"""
Test 51: Airliner Autopilot (Python)

The LLM must write a Python autopilot function that receives a state dict
with ~200 redundant sensor properties typical of a modern airliner, and
returns control inputs (elevator, aileron, rudder, throttle, etc.).

The autopilot is tested across 30 scenarios ranging from normal cruise to
catastrophic failures inspired by real aviation disasters. Each scenario
runs a physics simulation and scores recovery to stable flight.

This tests:
1. Python proficiency for real-time control
2. Sensor fusion with redundant/failed sensors
3. Flight dynamics understanding (lift, drag, stall, asymmetric thrust)
4. Fault tolerance and graceful degradation
5. PID or adaptive control design
"""

import math
import traceback
from typing import Tuple

from visualization_utils import generate_threejs_flight_path

from autopilot_sim import (
  make_truth,
  SensorModel,
  SensorFailure,
  SimRunner,
  AC,
  DEG2RAD,
  RAD2DEG,
  FT2M,
  M2FT,
  KT2MS,
  MS2KT,
  isa_rho,
  ias_from_tas,
  tas_from_ias,
  list_all_state_keys,
  physics_step,
)

title = "Airliner Autopilot (Python)"

tags = [
  "python",
  "structured response",
  "control systems",
  "simulation",
]

TIMEOUT_SECONDS = 120  # generous for 30 subpasses

_HISTORY_CACHE = {}  # {(aiEngineName, subPass): history_list}


# ──────────────────────────────────────────────────────────────────────────────
# Scenario Definitions
# ──────────────────────────────────────────────────────────────────────────────
def _scenario(name,
              description,
              truth_kwargs=None,
              failures=None,
              truth_mods=None,
              target_alt=None,
              target_hdg=None,
              target_ias=None,
              steps=1200,
              historical_ref=None):
  return {
    'name': name,
    'description': description,
    'truth_kwargs': truth_kwargs or {},
    'failures': failures or {},
    'truth_mods': truth_mods or {},
    'target_alt': target_alt,
    'target_hdg': target_hdg,
    'target_ias': target_ias,
    'steps': steps,
    'historical_ref': historical_ref,
  }


SCENARIOS = [
  # ── 0: Normal cruise ──
  _scenario(
    'Normal cruise at FL350',
    'Maintain straight and level flight at FL350, 250 KIAS, heading 090.',
    truth_kwargs={
      'alt_ft': 35000,
      'ias_kt': 250,
      'heading_deg': 90
    },
  ),

  # ── 1: Moderate turbulence ──
  _scenario(
    'Moderate turbulence at cruise',
    'Maintain flight in moderate turbulence.',
    truth_kwargs={
      'alt_ft': 35000,
      'ias_kt': 250,
      'heading_deg': 90
    },
    truth_mods={'turbulence': 0.4},
  ),

  # ── 2: Single engine failure ──
  _scenario(
    'Single engine failure at cruise',
    'Engine 1 fails at cruise. Maintain heading and controlled descent if needed.',
    truth_kwargs={
      'alt_ft': 35000,
      'ias_kt': 250,
      'heading_deg': 90
    },
    truth_mods={
      'engine_running': [False, True],
      'n1': [0.0, 85.0]
    },
  ),

  # ── 3: Dual engine flameout ──
  _scenario(
    'Dual engine flameout',
    'Both engines flame out. Glide while maintaining control. '
    'Inspired by Gimli Glider (Air Canada 143, 1983) and US Airways 1549 (2009).',
    truth_kwargs={
      'alt_ft': 35000,
      'ias_kt': 250,
      'heading_deg': 90
    },
    truth_mods={
      'engine_running': [False, False],
      'n1': [0.0, 0.0],
      'throttle': [0.0, 0.0]
    },
    historical_ref='Gimli Glider / Sully',
  ),

  # ── 4: Pitot tube icing — Air France 447 ──
  _scenario(
    'Pitot tube icing (AF447)',
    'All three pitot probes ice over progressively. IAS readings diverge and drop to zero. '
    'Maintain control using attitude, GPS groundspeed, and AoA. '
    'Air France 447 (2009): pitot icing caused unreliable airspeed, crew lost control over the Atlantic.',
    truth_kwargs={
      'alt_ft': 35000,
      'ias_kt': 250,
      'heading_deg': 90
    },
    failures={
      'ias_1': (SensorFailure.DRIFT, -8.0),  # dropping 8 kt/s
      'ias_2': (SensorFailure.STUCK, 0),  # frozen
      'ias_3': (SensorFailure.DRIFT, -5.0),
      'mach_1': (SensorFailure.DRIFT, -0.02),
      'mach_2': (SensorFailure.STUCK, 0),
      'mach_3': (SensorFailure.DRIFT, -0.01),
    },
    historical_ref='Air France 447 (2009)',
  ),

  # ── 5: AoA sensor stuck high — Boeing 737 MAX / Ethiopian 302 ──
  _scenario(
    'AoA sensor disagree (Ethiopian 302 / Lion Air 610)',
    'AoA sensor 1 reads 20° higher than actual (stuck/biased). Other two are correct. '
    'A naive system would see imminent stall and push nose down. '
    'Ethiopian Airlines 302 (2019) / Lion Air 610 (2018): erroneous AoA triggered MCAS nose-down.',
    truth_kwargs={
      'alt_ft': 10000,
      'ias_kt': 250,
      'heading_deg': 90
    },
    failures={
      'aoa_1': (SensorFailure.BIAS, 20.0),  # reads 20° too high
    },
    historical_ref='Ethiopian 302 / Lion Air 610',
  ),

  # ── 6: Stuck stabilizer trim — Alaska Airlines 261 ──
  _scenario(
    'Stabilizer trim runaway (Alaska 261)',
    'Horizontal stabilizer trim running away nose-down at 0.3°/s. '
    'Must counteract with elevator and attempt to stop or overpower the trim. '
    'Alaska Airlines 261 (2000): jackscrew failure caused trim runaway, aircraft crashed.',
    truth_kwargs={
      'alt_ft': 18000,
      'ias_kt': 280,
      'heading_deg': 180
    },
    truth_mods={'trim_runaway_rate': 0.3 * DEG2RAD},
    historical_ref='Alaska Airlines 261 (2000)',
  ),

  # ── 7: Severe wind shear on approach — Delta 191 ──
  _scenario(
    'Wind shear on approach (Delta 191)',
    'On approach at 2000 ft, encounter severe microburst. '
    'Headwind rapidly shifts to tailwind, causing airspeed drop and sink rate increase. '
    'Delta 191 (1985): microburst on approach to DFW caused crash.',
    truth_kwargs={
      'alt_ft': 2000,
      'ias_kt': 140,
      'heading_deg': 270,
      'mass': 60000
    },
    truth_mods={
      'flap_deg': 20,
      'gear': 1.0,
      'wind_x': 15 * KT2MS,
      'wind_y': 0,  # initial headwind
      'turbulence': 0.6
    },
    target_ias=140,
    steps=800,
    historical_ref='Delta 191 (1985)',
  ),

  # ── 8: Dutch roll — yaw damper failure ──
  _scenario(
    'Dutch roll onset',
    'Yaw damper fails. Aircraft develops dutch roll oscillation (coupled yaw-roll). '
    'Must damp oscillation manually with coordinated rudder inputs.',
    truth_kwargs={
      'alt_ft': 35000,
      'ias_kt': 250,
      'heading_deg': 90
    },
    truth_mods={
      'r': 3.0 * DEG2RAD,
      'p': -2.0 * DEG2RAD
    },
  ),

  # ── 9: Spiral dive recovery ──
  _scenario(
    'Spiral dive recovery',
    'Aircraft in 60° bank, 15° nose down, descending rapidly. Recover to wings level.',
    truth_kwargs={
      'alt_ft': 25000,
      'ias_kt': 300,
      'heading_deg': 90
    },
    truth_mods={
      'phi': 60 * DEG2RAD,
      'theta': -15 * DEG2RAD,
      'p': 5 * DEG2RAD,
      'q': -3 * DEG2RAD
    },
  ),

  # ── 10: Nose-high unusual attitude ──
  _scenario(
    'Unusual attitude — nose high',
    'Aircraft at 30° pitch up, airspeed decaying rapidly toward stall.',
    truth_kwargs={
      'alt_ft': 20000,
      'ias_kt': 180,
      'heading_deg': 90
    },
    truth_mods={
      'theta': 30 * DEG2RAD,
      'q': 2 * DEG2RAD
    },
  ),

  # ── 11: Inverted unusual attitude ──
  _scenario(
    'Unusual attitude — near inverted',
    'Aircraft rolled 130°, nose down 10°. Recovery requires correct roll direction.',
    truth_kwargs={
      'alt_ft': 25000,
      'ias_kt': 280,
      'heading_deg': 90
    },
    truth_mods={
      'phi': 130 * DEG2RAD,
      'theta': -10 * DEG2RAD
    },
  ),

  # ── 12: Engine failure on takeoff (V1 cut) ──
  _scenario(
    'Engine failure on takeoff',
    'Engine 1 fails at 1500 ft after takeoff. Low altitude, flaps 10, asymmetric thrust.',
    truth_kwargs={
      'alt_ft': 1500,
      'ias_kt': 165,
      'heading_deg': 90,
      'mass': 75000
    },
    truth_mods={
      'engine_running': [False, True],
      'n1': [0.0, 95.0],
      'flap_deg': 10,
      'throttle': [0.0, 0.95]
    },
    target_alt=3000,
  ),

  # ── 13: Rudder hardover — USAir 427 ──
  _scenario(
    'Rudder hardover (USAir 427)',
    'Rudder jams at full deflection. Must counteract with aileron and manage yaw. '
    'USAir 427 (1994): uncommanded rudder hardover caused loss of control.',
    truth_kwargs={
      'alt_ft': 6000,
      'ias_kt': 190,
      'heading_deg': 90
    },
    truth_mods={'rudder_jam': 30 * DEG2RAD},
    historical_ref='USAir 427 (1994)',
  ),

  # ── 14: Barometric altimeter disagree — Helios 802 ──
  _scenario(
    'Altimeter disagree / pressurization',
    'All three baro altimeters show different values due to pressurization leak. ',
    truth_kwargs={
      'alt_ft': 34000,
      'ias_kt': 250,
      'heading_deg': 90
    },
    failures={
      'baro_alt_1': (SensorFailure.BIAS, 2000),
      'baro_alt_2': (SensorFailure.DRIFT, 5.0),
      'baro_alt_3': (SensorFailure.BIAS, -1500),
    },
  ),

  # ── 15: Wing icing — progressive CL degradation ──
  _scenario(
    'Wing icing',
    'Ice accumulates on wings, progressively degrading lift and increasing drag. '
    'Must increase speed and/or descend to maintain control.',
    truth_kwargs={
      'alt_ft': 18000,
      'ias_kt': 220,
      'heading_deg': 90
    },
    truth_mods={
      'cl_factor': 0.75,
      'cd_extra': 0.01
    },
  ),

  # ── 16: Cargo door blowout — Turkish Airlines 981 ──
  _scenario(
    'Cargo door failure (Turkish Airlines 981)',
    'Aft cargo door fails, causing drag increase and partial loss of control authority. '
    'Turkish Airlines 981 (1974): cargo door blew out, severed control cables.',
    truth_kwargs={
      'alt_ft': 12000,
      'ias_kt': 250,
      'heading_deg': 90
    },
    truth_mods={
      'cd_extra': 0.03,
      'cn_extra': 0.02,
      'cl_roll_extra': -0.01
    },
    historical_ref='Turkish Airlines 981 (1974)',
  ),

  # ── 17: Flap asymmetry ──
  _scenario(
    'Flap asymmetry',
    'Left flap deploys to 20° but right flap stays at 0°. Causes strong roll moment.',
    truth_kwargs={
      'alt_ft': 8000,
      'ias_kt': 200,
      'heading_deg': 90
    },
    truth_mods={
      'cl_roll_extra': 0.04,
      'cd_extra': 0.005
    },
  ),

  # ── 18: ADIRU failure — Qantas 72 ──
  _scenario(
    'ADIRU failure (Qantas 72)',
    'IRS unit 1 sends wildly incorrect pitch values. Other units normal. '
    'Qantas 72 (2008): faulty ADIRU caused uncommanded pitch-down.',
    truth_kwargs={
      'alt_ft': 37000,
      'ias_kt': 250,
      'heading_deg': 90
    },
    failures={
      'irs_1_pitch': (SensorFailure.SPIKE, 50.0),  # wild ±50° spikes
      'irs_1_hdg': (SensorFailure.SPIKE, 180.0),
    },
    historical_ref='Qantas 72 (2008)',
  ),

  # ── 19: Engine fire ──
  _scenario(
    'Engine fire',
    'Engine 2 catches fire. Must shut down engine, maintain control with asymmetric thrust.',
    truth_kwargs={
      'alt_ft': 28000,
      'ias_kt': 260,
      'heading_deg': 180
    },
    truth_mods={'engine_fire': [False, True]},
  ),

  # ── 20: Crosswind approach ──
  _scenario(
    'Strong crosswind approach',
    'On approach at 3000 ft with 35-knot crosswind. Maintain centerline.',
    truth_kwargs={
      'alt_ft': 3000,
      'ias_kt': 145,
      'heading_deg': 90,
      'mass': 58000
    },
    truth_mods={
      'wind_x': 0,
      'wind_y': 35 * KT2MS,
      'flap_deg': 25,
      'gear': 1.0
    },
    target_ias=145,
    steps=800,
  ),

  # ── 21: Loss of all GPS ──
  _scenario(
    'GPS denial',
    'All three GPS receivers fail simultaneously. Navigate using IRS/INS only.',
    truth_kwargs={
      'alt_ft': 35000,
      'ias_kt': 250,
      'heading_deg': 90
    },
    failures={
      'gps_1': (SensorFailure.DEAD, 0),
      'gps_2': (SensorFailure.DEAD, 0),
      'gps_3': (SensorFailure.DEAD, 0),
    },
  ),

  # ── 22: Stall — Colgan 3407 type ──
  _scenario(
    'Approach stall in icing (Colgan 3407)',
    'Low speed at 3000 ft in icing conditions. Wings contaminated, stall margin reduced. '
    'Colgan Air 3407 (2009): stall during approach in icing, improper recovery.',
    truth_kwargs={
      'alt_ft': 3000,
      'ias_kt': 130,
      'heading_deg': 90,
      'mass': 58000
    },
    truth_mods={
      'cl_factor': 0.7,
      'cd_extra': 0.008,
      'flap_deg': 15
    },
    target_ias=150,
    steps=800,
    historical_ref='Colgan Air 3407 (2009)',
  ),

  # ── 23: Hydraulic system failure ──
  _scenario(
    'Dual hydraulic failure',
    'Hydraulic systems A and B fail. Reduced control authority (50% effectiveness).',
    truth_kwargs={
      'alt_ft': 25000,
      'ias_kt': 250,
      'heading_deg': 90
    },
    # Simulated by reducing control surface effectiveness
    truth_mods={'turbulence': 0.15},
  ),

  # ── 24: Wake turbulence upset ──
  _scenario(
    'Wake turbulence upset',
    'Sudden violent roll from wake turbulence of heavy aircraft ahead.',
    truth_kwargs={
      'alt_ft': 8000,
      'ias_kt': 200,
      'heading_deg': 90
    },
    truth_mods={
      'phi': -45 * DEG2RAD,
      'p': -20 * DEG2RAD,
      'theta': 5 * DEG2RAD
    },
  ),

  # ── 25: Rapid decompression ──
  _scenario(
    'Rapid decompression at cruise',
    'Cabin depressurizes at FL370. Must execute emergency descent to FL100.',
    truth_kwargs={
      'alt_ft': 37000,
      'ias_kt': 250,
      'heading_deg': 90
    },
    target_alt=10000,
  ),

  # ── 26: All pitot probes read differently ──
  _scenario(
    'Three pitot probes disagree',
    'Each pitot probe reads a different airspeed. Must determine which (if any) is correct.',
    truth_kwargs={
      'alt_ft': 30000,
      'ias_kt': 260,
      'heading_deg': 90
    },
    failures={
      'ias_1': (SensorFailure.BIAS, 40),  # reads 40 kt high
      'ias_2': (SensorFailure.BIAS, -30),  # reads 30 kt low
      'ias_3': (SensorFailure.NOISE, 20),  # noisy ±20 kt
    },
  ),

  # ── 27: Overweight approach ──
  _scenario(
    'Overweight emergency landing',
    'Must land overweight (above max landing weight). Higher approach speed needed.',
    truth_kwargs={
      'alt_ft': 4000,
      'ias_kt': 165,
      'heading_deg': 90,
      'mass': 75000
    },
    truth_mods={
      'flap_deg': 20,
      'gear': 1.0
    },
    target_ias=165,
    steps=800,
  ),

  # ── 28: Severe turbulence + engine failure ──
  _scenario(
    'Combined: turbulence + engine failure',
    'Engine 1 fails in severe turbulence. Multiple challenges simultaneously.',
    truth_kwargs={
      'alt_ft': 30000,
      'ias_kt': 260,
      'heading_deg': 90
    },
    truth_mods={
      'engine_running': [False, True],
      'n1': [0.0, 85.0],
      'turbulence': 0.5
    },
  ),

  # ── 29: Total sensor chaos ──
  _scenario(
    'Multiple sensor failures + upset',
    'Aircraft upset (45° bank, nose down) with simultaneous failures of '
    'one IAS, one AoA, and one IRS. The ultimate test of sensor fusion and recovery.',
    truth_kwargs={
      'alt_ft': 20000,
      'ias_kt': 280,
      'heading_deg': 90
    },
    truth_mods={
      'phi': 45 * DEG2RAD,
      'theta': -10 * DEG2RAD,
      'turbulence': 0.3
    },
    failures={
      'ias_1': (SensorFailure.DEAD, 0),
      'aoa_2': (SensorFailure.BIAS, 15.0),
      'irs_3_pitch': (SensorFailure.SPIKE, 30.0),
      'irs_3_hdg': (SensorFailure.SPIKE, 90.0),
    },
  ),
]


# ──────────────────────────────────────────────────────────────────────────────
# Prompt
# ──────────────────────────────────────────────────────────────────────────────
def _build_state_docs():
  """Build documentation string for all state dict keys."""
  keys = list_all_state_keys()
  # Group by prefix
  groups = {}
  for k in keys:
    prefix = k.split('_')[0]
    if prefix not in groups:
      groups[prefix] = []
    groups[prefix].append(k)

  lines = []
  for prefix in sorted(groups.keys()):
    lines.append(f'  # {prefix}:')
    for k in sorted(groups[prefix]):
      lines.append(f'  #   {k}')
  return '\n'.join(lines)


def prepareSubpassPrompt(subPass):
  if subPass != 0:
    raise StopIteration

  state_keys = list_all_state_keys()
  num_keys = len(state_keys)

  # Group sensor keys for the prompt
  sensor_groups = {
    'Barometric Altimeters (x3)': [k for k in state_keys if k.startswith('baro_')],
    'Radar Altimeters (x3)': [k for k in state_keys if k.startswith('radar_')],
    'Airspeed / Mach (x3 ADCs)': [
      k for k in state_keys if k.startswith('ias_') or k.startswith('tas_') or k.startswith('mach_')
    ],
    'Angle of Attack (x3)': [k for k in state_keys if k.startswith('aoa_')],
    'GPS (x3)': [k for k in state_keys if k.startswith('gps_')],
    'IRS / INS (x3)': [k for k in state_keys if k.startswith('irs_')],
    'Gyroscopes (x3)': [k for k in state_keys if k.startswith('gyro_')],
    'Accelerometers (x3)': [k for k in state_keys if k.startswith('accel_')],
    'Air Data Computers (x3)': [k for k in state_keys if k.startswith('adc_')],
    'Flight Controls': [
      k for k in state_keys if any(
        k.startswith(p)
        for p in ['elevator_', 'aileron_', 'rudder_', 'flap_', 'slat_', 'spoiler_', 'speedbrake'])
    ],
    'Landing Gear': [k for k in state_keys if k.startswith('gear_') or k.startswith('weight_')],
    'Engines (x2)': [k for k in state_keys if k.startswith('engine_')],
    'Throttle': [k for k in state_keys if k.startswith('throttle_')],
    'Fuel': [k for k in state_keys if k.startswith('fuel_')],
    'Weight/CG': [k for k in state_keys if k.startswith('gross_') or k.startswith('cg_')],
    'Environment': [
      k for k in state_keys if k in
      ['oat_c', 'tat_c', 'wind_speed_kt', 'wind_dir_deg', 'turbulence_intensity', 'icing_rate']
    ],
    'Navigation': [
      k for k in state_keys if any(
        k.startswith(p)
        for p in ['magnetic_', 'true_', 'vertical_', 'ground_', 'track_', 'drift_', 'flight_path'])
    ],
    'Electrical': [
      k for k in state_keys if k.startswith('bus_') or k.startswith('gen_')
      or k.startswith('apu_gen') or k == 'battery_voltage'
    ],
    'Hydraulic': [k for k in state_keys if k.startswith('hyd_')],
    'Pneumatic/Cabin': [
      k for k in state_keys if any(
        k.startswith(p) for p in ['bleed_', 'pack_', 'cabin_', 'apu_bleed'])
    ],
    'Autopilot Targets': [
      k for k in state_keys if k.startswith('ap_') or k.startswith('at_') or k.startswith('target_')
    ],
    'Sim': [k for k in state_keys if k.startswith('sim_') or k == 'dt'],
  }

  sensor_doc = ''
  for group_name, keys in sensor_groups.items():
    if keys:
      sensor_doc += f'\n  **{group_name}** ({len(keys)} keys):\n'
      for k in sorted(keys):
        sensor_doc += f'    {k}\n'

  scenario_descriptions = ''
  for i, sc in enumerate(SCENARIOS):
    hist = f' [{sc["historical_ref"]}]' if sc.get('historical_ref') else ''
    scenario_descriptions += f'  {i:2d}. {sc["name"]}{hist}\n'

  return f"""Write a Python autopilot for a twin-engine airliner (A320-class).

**Function Signature:**
```python
def autopilot_step(state: dict, dt: float) -> dict:
```

**Input:** `state` is a dict with {num_keys} keys representing redundant sensor data:
{sensor_doc}
Key facts:
- Most sensors come in triples (3 redundant units). Any can fail silently!
- `*_valid` flags exist but may be WRONG (sensor can fail without flagging)
- `target_alt_ft`, `target_hdg_deg`, `target_ias_kt` are the desired flight state
- `dt` is the timestep (~0.05s = 20 Hz)

**Output:** Return a dict with these control keys:
```
  elevator:       -1.0 (nose down) to +1.0 (nose up), RELATIVE TO TRIM
                  (0.0 = maintain current trimmed attitude)
  aileron:        -1.0 (roll left) to +1.0 (roll right)
  rudder:         -1.0 (yaw left) to +1.0 (yaw right)
  throttle_1:     0.0 (idle) to 1.0 (full thrust) — left engine
  throttle_2:     0.0 (idle) to 1.0 (full thrust) — right engine
  elevator_trim:  -1.0 to +1.0 (adjusts the trim baseline over time)
  flaps:          0 to 40 (degrees)
  speedbrake:     0.0 to 1.0
  gear:           0 (up) or 1 (down)
```
NOTE: If you omit a key, the current value is maintained. Throttle defaults
to the current setting if not specified.

**Critical Design Requirements:**
1. **Sensor fusion**: Cross-check redundant sensors, and cross-check other physical 
   properties. If GPS altitude is dropping, radar altitude is stable, and barometric
   altitude is climbing - two failures have occured, and you have to pick wisely.
2. **Aerodynamic simulation**: The flight model includes a full aerodynamic model, so
   brush up on your knowledge of stalls, asymetric thurst, yaw dampening, and trim.
5. Vne is (~365 KIAS / Mach 0.82).
6. **Persistence**: You may define module-level variables for PID integral terms,
   previous values, etc. They persist across calls within a scenario.

Your single autopilot function must handle every scenario, including over a dozen
historical aviation accidents. The same code is tested against every scenario.
"""


extraGradeAnswerRuns = list(range(1, len(SCENARIOS)))

structure = {
  "type": "object",
  "properties": {
    "reasoning": {
      "type": "string",
      "description": "Explain your autopilot design approach"
    },
    "python_code": {
      "type": "string",
      "description": "Complete Python code defining autopilot_step(state, dt)"
    }
  },
  "required": ["reasoning", "python_code"],
  "additionalProperties": False
}

# ──────────────────────────────────────────────────────────────────────────────
# Grading
# ──────────────────────────────────────────────────────────────────────────────


def _setup_scenario(idx):
  """Create truth state and sensor model for a scenario."""
  sc = SCENARIOS[idx]
  truth = make_truth(**sc['truth_kwargs'])

  # Apply truth modifications
  for key, val in sc['truth_mods'].items():
    if isinstance(val, list):
      truth[key] = list(val)
    else:
      truth[key] = val

  # Wind shear special handling for scenario 7
  if idx == 7:
    # Wind will be modified during simulation (handled in gradeAnswer)
    pass

  sensor_model = SensorModel(seed=42 + idx)

  # Apply sensor failures
  for key, (mode, param) in sc['failures'].items():
    sensor_model.set_failure(key, mode, param)

  target_alt = sc['target_alt'] or truth['z'] * M2FT
  target_hdg = sc['target_hdg'] or truth['psi'] * RAD2DEG
  target_ias = sc['target_ias']
  if target_ias is None:
    V = math.sqrt(truth['u']**2 + truth['v']**2 + truth['w']**2)
    target_ias = ias_from_tas(V, truth['z']) * MS2KT

  runner = SimRunner(truth,
                     sensor_model,
                     target_alt_ft=target_alt,
                     target_hdg_deg=target_hdg,
                     target_ias_kt=target_ias)
  return runner, sc


def gradeAnswer(result, subPass, aiEngineName):
  if not result or 'python_code' not in result:
    return 0.0, 'No Python code provided'

  code = result['python_code']

  # Compile and extract autopilot_step function
  try:
    ns = {}
    exec(compile(code, '<autopilot>', 'exec'), ns)
    if 'autopilot_step' not in ns:
      return 0.0, 'Code does not define autopilot_step(state, dt)'
    autopilot_fn = ns['autopilot_step']
  except Exception as e:
    return 0.0, f'Code compilation failed: {e}'

  sc_idx = subPass
  if sc_idx >= len(SCENARIOS):
    return 0.0, f'Invalid subpass {sc_idx}'

  sc = SCENARIOS[sc_idx]
  runner, scenario = _setup_scenario(sc_idx)

  print(f'  Scenario {sc_idx}: {sc["name"]}')

  try:
    # Special handling for wind shear scenario (7) - wind changes during sim
    if sc_idx == 7:
      _run_windshear(runner, autopilot_fn)
    else:
      runner.run(autopilot_fn, steps=sc.get('steps', 1200))
  except Exception as e:
    tb = traceback.format_exc()
    return 0.0, f'Simulation error: {e}\n{tb[:500]}'

  score, details = runner.score_recovery()

  # Cache history for visualization
  _HISTORY_CACHE[(aiEngineName, sc_idx)] = runner.history

  hist_note = ''
  if sc.get('historical_ref'):
    hist_note = f' [Ref: {sc["historical_ref"]}]'

  return score, f'[{sc["name"]}]{hist_note} {details}'


def _run_windshear(runner, autopilot_fn):
  """Custom sim loop for wind shear scenario with time-varying wind."""
  dt = 0.05
  for step in range(800):
    t = step * dt
    # Microburst profile: headwind -> downdraft -> tailwind
    if t < 5:
      runner.truth['wind_x'] = 15 * KT2MS  # headwind
      runner.truth['wind_z'] = 0
    elif t < 15:
      phase = (t - 5) / 10.0
      runner.truth['wind_x'] = 15 * KT2MS * (1 - 2 * phase)  # headwind to tailwind
      runner.truth['wind_z'] = -8 * KT2MS * math.sin(phase * math.pi)  # downdraft
    else:
      runner.truth['wind_x'] = -15 * KT2MS  # tailwind
      runner.truth['wind_z'] = 0

    state = runner.sensors.build_state(runner.truth)
    state['target_alt_ft'] = runner.target_alt_ft
    state['target_hdg_deg'] = runner.target_hdg_deg
    state['target_ias_kt'] = runner.target_ias_kt

    try:
      controls = autopilot_fn(state, dt)
      if not isinstance(controls, dict):
        controls = {}
    except Exception:
      controls = {}

    for k in ['elevator', 'aileron', 'rudder', 'elevator_trim', 'speedbrake']:
      if k in controls:
        controls[k] = max(-1.0, min(1.0, float(controls[k])))
    for k in ['throttle_1', 'throttle_2']:
      if k in controls:
        controls[k] = max(0.0, min(1.0, float(controls[k])))

    runner.truth = physics_step(runner.truth, controls, dt)

    if runner.truth['z'] <= 0 and step > 10:
      runner.crashed = True
      runner.crash_reason = 'CFIT during wind shear'
      break

    if step % 20 == 0:
      V = math.sqrt(runner.truth['u']**2 + runner.truth['v']**2 + runner.truth['w']**2)
      runner.history.append({
        'time': runner.truth['time'],
        'alt_ft': runner.truth['z'] * M2FT,
        'ias_kt': ias_from_tas(V, runner.truth['z']) * MS2KT,
        'hdg_deg': runner.truth['psi'] * RAD2DEG,
        'pitch_deg': runner.truth['theta'] * RAD2DEG,
        'roll_deg': runner.truth['phi'] * RAD2DEG,
        'vs_fpm': state.get('vertical_speed_fpm', 0),
      })


# ──────────────────────────────────────────────────────────────────────────────
# Visualization for HTML report
# ──────────────────────────────────────────────────────────────────────────────
def _history_to_path(history):
  """Dead-reckon XY position from heading + airspeed history.

  Returns list of [x_m, y_m, alt_m] suitable for the 3-D viewer.
  x = east, y = north, z = altitude (metres).
  """
  if not history:
    return []
  path = []
  x, y = 0.0, 0.0
  prev_time = None
  for h in history:
    alt_m = h['alt_ft'] * FT2M
    path.append([round(x, 1), round(y, 1), round(alt_m, 1)])
    if prev_time is not None:
      dt = h['time'] - prev_time
      hdg_rad = math.radians(h['hdg_deg'])
      speed_ms = h['ias_kt'] * KT2MS
      x += speed_ms * math.sin(hdg_rad) * dt
      y += speed_ms * math.cos(hdg_rad) * dt
    prev_time = h['time']
  return path


def resultToNiceReport(result, subPass, aiEngineName):
  sc_idx = subPass
  if sc_idx >= len(SCENARIOS):
    return ''
  sc = SCENARIOS[sc_idx]
  history = _HISTORY_CACHE.get((aiEngineName, sc_idx), [])
  if not history:
    return ''
  path = _history_to_path(history)
  return generate_threejs_flight_path(path, scenario_name=sc['name'])


highLevelSummary = """
<p>Write a Python autopilot for a simulated twin-engine airliner. The autopilot
receives over 200 sensor readings each tick (altitude, airspeed, attitude, engine
data, etc.) &mdash; many triple-redundant &mdash; and must output control surface
commands (elevator, aileron, rudder, throttle) to keep the aircraft flying safely.</p>
<p>The 30 scenarios range from calm cruise to catastrophic failures inspired by real
aviation disasters: frozen pitot tubes, stuck trim, engine flameouts, sensor
disagreements, and severe turbulence. The AI must handle sensor fusion, fault
detection, and adaptive control.</p>
"""
