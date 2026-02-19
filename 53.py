"""
Test 53: Spacecraft Orbital Docking Autopilot (Python)

The LLM must write a Python autopilot to dock a spacecraft with a space station.
Uses true Keplerian orbital mechanics — "thrust toward target" often moves you
AWAY due to orbital dynamics. The LLM must understand this.

Tested across 25 scenarios with varying starting positions, orientations,
sensor failures, thruster failures, and uncommanded thrust events.
"""

import math
import traceback

from visualization_utils import generate_threejs_docking_viz

from orbital_sim import (
  make_dock_truth,
  DockSensorModel,
  DockSimRunner,
  SC,
  STATION,
  DEG2RAD,
  RAD2DEG,
  MU_EARTH,
  R_EARTH,
  dock_physics_step,
  list_dock_state_keys,
  eci_to_lvlh,
  _vec_norm,
)

title = "Spacecraft Orbital Docking Autopilot (Python)"
TIMEOUT_SECONDS = 120

_HISTORY_CACHE = {}   # {(aiEngineName, subPass): history_list}
_OUTCOME_CACHE = {}   # {(aiEngineName, subPass): (docked, crashed, crash_reason)}


# ──────────────────────────────────────────────────────────────────────────────
# Scenarios
# ──────────────────────────────────────────────────────────────────────────────
def _sc(name, desc, **kw):
  return {'name': name, 'description': desc, **kw}


SCENARIOS = [
  # 0: V-bar approach (behind station, same orbit)
  _sc('V-bar approach — 200m behind',
      'Classic V-bar approach: 200m behind in along-track. Easiest case.',
      offset=[0, -200, 0]),

  # 1: V-bar 500m
  _sc('V-bar approach — 500m behind',
      'V-bar approach from 500m behind. Longer approach, more fuel management.',
      offset=[0, -500, 0]),

  # 2: V-bar 50m (close range)
  _sc('V-bar close approach — 50m',
      'Short-range docking from 50m. Precision control needed.',
      offset=[0, -50, 0]),

  # 3: R-bar approach (below station)
  _sc('R-bar approach — 200m below',
      'Approach from below (radial). Must understand R-bar dynamics.',
      offset=[-200, 0, 0]),

  # 4: Above station
  _sc('Approach from above — 200m',
      'Starting 200m above station in radial direction.',
      offset=[200, 0, 0]),

  # 5: Cross-track offset
  _sc('Cross-track offset — 100m',
      'Starting 100m out of plane. Must correct cross-track before approach.',
      offset=[0, -200, 100]),

  # 6: Ahead of station
  _sc('Ahead of station — 200m',
      'Starting 200m ahead. Counter-intuitive: must slow down to approach.',
      offset=[0, 200, 0]),

  # 7: Diagonal approach
  _sc('Diagonal — 150m radial + 150m along-track',
      'Starting offset in both radial and along-track.',
      offset=[150, -150, 0]),

  # 8: Far approach
  _sc('Long-range — 2km behind',
      'Starting 2km behind. Requires efficient phasing maneuver.',
      offset=[0, -2000, 0],
      steps=7200),

  # 9: Initial drift velocity
  _sc('Drifting away — 200m with 0.5 m/s radial',
      'Starting 200m behind but drifting radially at 0.5 m/s.',
      offset=[0, -200, 0],
      dv=[0.5, 0, 0]),

  # 10: Closing too fast
  _sc('Closing too fast — 2 m/s approach',
      'Starting 100m behind but closing at 2 m/s. Must brake.',
      offset=[0, -100, 0],
      dv=[0, 2.0, 0]),

  # 11: Range sensor failure
  _sc('Range sensor 1 failed',
      'Range sensor 1 stuck. Use other sensors for range.',
      offset=[0, -200, 0],
      failures={'range_1': ('stuck', 0)}),

  # 12: Radar failure
  _sc('Radar relative position failed',
      'Radar relative position all axes dead. Use lidar + range sensors.',
      offset=[0, -200, 0],
      failures={
        'radar_rel_x': ('dead', 0),
        'radar_rel_y': ('dead', 0),
        'radar_rel_z': ('dead', 0),
        'radar_relv_x': ('dead', 0),
        'radar_relv_y': ('dead', 0),
        'radar_relv_z': ('dead', 0),
      }),

  # 13: Gyro bias
  _sc('Gyro 1 bias + gyro 2 drift',
      'Gyroscope 1 has 0.5°/s bias, gyro 2 drifts. Gyro 3 normal.',
      offset=[0, -200, 0],
      failures={
        'gyro_1_yaw': ('bias', 0.5),
        'gyro_2_pitch': ('drift', 0.01)
      }),

  # 14: Star tracker failure
  _sc('Star tracker 1 dead',
      'Star tracker 1 fails. Use star tracker 2 and gyros for attitude.',
      offset=[0, -200, 0],
      failures={
        'star_1_roll': ('dead', 0),
        'star_1_pitch': ('dead', 0),
        'star_1_yaw': ('dead', 0),
      }),

  # 15: GPS denied
  _sc('Both GPS receivers failed',
      'No GPS. Must use relative sensors and ground data.',
      offset=[0, -200, 0],
      failures={
        'gps_1_x': ('dead', 0),
        'gps_1_y': ('dead', 0),
        'gps_1_z': ('dead', 0),
        'gps_2_x': ('dead', 0),
        'gps_2_y': ('dead', 0),
        'gps_2_z': ('dead', 0),
      }),

  # 16: Docking camera failure
  _sc('Docking camera failed',
      'Docking camera dead. Must use radar/lidar for final approach.',
      offset=[0, -50, 0],
      failures={
        'dock_cam_x': ('dead', 0),
        'dock_cam_y': ('dead', 0),
        'dock_cam_z': ('dead', 0),
      }),

  # 17: Uncommanded thrust
  _sc('Uncommanded +Y thrust (0.5N)',
      'Parasitic thruster leak applying 0.5N along-track continuously.',
      offset=[0, -200, 0],
      uncommanded=[0, 0.5, 0]),

  # 18: Uncommanded cross-track
  _sc('Uncommanded cross-track thrust',
      'Thruster leak applying 0.3N cross-track.',
      offset=[0, -200, 0],
      uncommanded=[0, 0, 0.3]),

  # 19: Flickering sensors
  _sc('Flickering range + radar sensors',
      'Range and radar sensors have intermittent dropouts.',
      offset=[0, -200, 0],
      flicker=['range_1', 'range_2', 'radar_rel_x', 'radar_rel_y', 'radar_rel_z']),

  # 20: Low fuel
  _sc('Low fuel — only 50 kg remaining',
      'Very limited fuel. Must be extremely efficient.',
      offset=[0, -100, 0],
      fuel=50),

  # 21: Combined: offset + sensor fail
  _sc('Cross-track + radar failure',
      'Starting with cross-track offset and radar dead.',
      offset=[0, -200, 80],
      failures={
        'radar_rel_x': ('dead', 0),
        'radar_rel_y': ('dead', 0),
        'radar_rel_z': ('dead', 0),
      }),

  # 22: Different orbital parameters
  _sc('Higher orbit — 450km',
      'Station in 450km orbit instead of 408km. Different dynamics.',
      offset=[0, -200, 0],
      station_alt=450),

  # 23: All range sensors noisy
  _sc('All range sensors very noisy',
      'All 3 range sensors have high noise (±5m). Need filtering.',
      offset=[0, -200, 0],
      failures={
        'range_1': ('noise', 5.0),
        'range_2': ('noise', 5.0),
        'range_3': ('noise', 5.0),
      }),

  # 24: Combined chaos
  _sc('Multi-failure + uncommanded thrust + offset',
      'Radar dead, gyro 1 biased, 0.3N parasitic thrust, starting 300m with drift.',
      offset=[100, -300, 50],
      dv=[0.2, 0, 0.1],
      uncommanded=[0, 0.3, 0],
      failures={
        'radar_rel_x': ('dead', 0),
        'radar_rel_y': ('dead', 0),
        'radar_rel_z': ('dead', 0),
        'gyro_1_yaw': ('bias', 0.3)
      }),
]


# ──────────────────────────────────────────────────────────────────────────────
# Prompt
# ──────────────────────────────────────────────────────────────────────────────
def prepareSubpassPrompt(subPass):
  if subPass != 0:
    raise StopIteration

  state_keys = list_dock_state_keys()
  num_keys = len(state_keys)

  sensor_groups = {
    'Range Finders (x3)': [k for k in state_keys if k.startswith('range_')],
    'Radar Relative Pos/Vel': [k for k in state_keys if k.startswith('radar_')],
    'Lidar Relative Pos/Vel': [k for k in state_keys if k.startswith('lidar_')],
    'Gyroscopes (x3)': [k for k in state_keys if k.startswith('gyro_')],
    'Accelerometers (x3)': [k for k in state_keys if k.startswith('accel_')],
    'Star Trackers (x2)': [k for k in state_keys if k.startswith('star_')],
    'GPS (x2)': [k for k in state_keys if k.startswith('gps_')],
    'Ground Station Data': [k for k in state_keys if k.startswith('ground_')],
    'Docking Camera': [k for k in state_keys if k.startswith('dock_')],
    'Fuel': [k for k in state_keys if k.startswith('fuel_')],
    'Targets': [k for k in state_keys if k.startswith('target_')],
  }

  sensor_doc = ''
  for gname, keys in sensor_groups.items():
    if keys:
      sensor_doc += f'\n  **{gname}** ({len(keys)} keys):\n'
      for k in sorted(keys):
        sensor_doc += f'    {k}\n'

  scenario_list = ''
  for i, sc in enumerate(SCENARIOS):
    scenario_list += f'  {i:2d}. {sc["name"]}\n'

  return f"""Write a Python autopilot to dock a spacecraft with a space station in orbit.

**CRITICAL ORBITAL MECHANICS INSIGHT:**
In orbit, relative motion is COUNTER-INTUITIVE:
- Thrusting TOWARD the target (prograde if behind) raises your orbit → you SLOW DOWN relative to target
- To catch up: thrust RETROGRADE (backward!) to lower orbit → you go FASTER
- Radial thrusts cause oscillation, not direct movement
- Cross-track corrections require thrusts at specific orbital positions
- Near the target (<50m), approximate as linear (CW equations work)
- Far away (>200m), full orbital mechanics dominate

**Function Signature:**
```python
def autopilot_step(state: dict, dt: float) -> dict:
```

**Input:** `state` is a dict with {num_keys} keys:
{sensor_doc}
Key sensor notes:
- Relative position/velocity in LVLH frame: x=radial (up), y=along-track (velocity dir), z=cross-track
- NEGATIVE y = behind station (most common start). Must CLOSE the y gap.
- Range finders give scalar distance. Radar/lidar give 3D relative pos/vel.
- All sensors can fail silently! Cross-check redundant sensors.
- Docking camera only active within 500m. Gives bearing to docking port.
- Ground station data updates every ~30s (stale but reliable).

**Output:** Return a dict:
```
  thrust_x:       -1.0 to +1.0 (radial: + = away from Earth)
  thrust_y:       -1.0 to +1.0 (along-track: + = prograde)
  thrust_z:       -1.0 to +1.0 (cross-track: + = north)
  att_rate_roll:  rad/s (max ±0.035)
  att_rate_pitch: rad/s (max ±0.035)
  att_rate_yaw:   rad/s (max ±0.035)
```
Each thrust axis: 220N per thruster, Isp=290s. Fuel is LIMITED (~500 kg).

**Docking Success Criteria:**
- Range < 2m AND relative speed < 0.3 m/s

**Approach Strategy Hints:**
1. Far (>200m): Use Hohmann-like transfer — retrograde to lower orbit if behind
2. Medium (50-200m): Gentle corrections, V-bar approach preferred
3. Close (<50m): Linear regime, direct thrust corrections OK
4. Final (<10m): Very gentle, < 0.1 m/s approach speed

**Test Scenarios ({len(SCENARIOS)} total):**
{scenario_list}
Your single autopilot function must handle ALL scenarios.
Persistence: module-level variables persist across calls within a scenario.
"""


extraGradeAnswerRuns = list(range(1, len(SCENARIOS)))

structure = {
  "type": "object",
  "properties": {
    "reasoning": {
      "type": "string",
      "description": "Explain your docking autopilot design"
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
  sc = SCENARIOS[idx]
  truth = make_dock_truth(
    chaser_offset_lvlh=sc.get('offset', [0, -200, 0]),
    chaser_dv_lvlh=sc.get('dv', [0, 0, 0]),
    station_alt_km=sc.get('station_alt', 408),
  )
  if 'fuel' in sc:
    truth['fuel_mass'] = sc['fuel']
  if 'uncommanded' in sc:
    truth['uncommanded_thrust'] = list(sc['uncommanded'])

  sensor = DockSensorModel(seed=42 + idx)
  for key, (mode, param) in sc.get('failures', {}).items():
    sensor.set_failure(key, mode, param)
  for key in sc.get('flicker', []):
    sensor.set_flicker(key)

  runner = DockSimRunner(truth, sensor)
  return runner, sc


def gradeAnswer(result, subPass, aiEngineName):
  if not result or 'python_code' not in result:
    return 0.0, 'No Python code provided'

  code = result['python_code']
  try:
    ns = {}
    exec(compile(code, '<dock_autopilot>', 'exec'), ns)
    if 'autopilot_step' not in ns:
      return 0.0, 'Code does not define autopilot_step(state, dt)'
    autopilot_fn = ns['autopilot_step']
  except Exception as e:
    return 0.0, f'Compilation failed: {e}'

  sc_idx = subPass
  if sc_idx >= len(SCENARIOS):
    return 0.0, f'Invalid subpass {sc_idx}'

  sc = SCENARIOS[sc_idx]
  runner, _ = _setup_scenario(sc_idx)
  steps = sc.get('steps', 3600)
  print(f'  Scenario {sc_idx}: {sc["name"]}')

  try:
    runner.run(autopilot_fn, steps=steps)
  except Exception as e:
    tb = traceback.format_exc()
    return 0.0, f'Simulation error: {e}\n{tb[:500]}'

  score, details = runner.score()

  # Cache for visualization
  _HISTORY_CACHE[(aiEngineName, sc_idx)] = runner.history
  _OUTCOME_CACHE[(aiEngineName, sc_idx)] = (
    runner.docked, runner.crashed, runner.crash_reason)

  return score, f'[{sc["name"]}] {details}'


# ──────────────────────────────────────────────────────────────────────────────
# Visualization for HTML report
# ──────────────────────────────────────────────────────────────────────────────
def resultToNiceReport(result, subPass, aiEngineName):
  sc_idx = subPass
  if sc_idx >= len(SCENARIOS):
    return ''
  sc = SCENARIOS[sc_idx]
  history = _HISTORY_CACHE.get((aiEngineName, sc_idx), [])
  if not history:
    return ''
  # Extract LVLH relative positions [x_rad, y_along, z_cross]
  path = [
    [round(h['rel_pos'][0], 2),
     round(h['rel_pos'][1], 2),
     round(h['rel_pos'][2], 2)]
    for h in history
  ]
  docked, crashed, crash_reason = _OUTCOME_CACHE.get(
    (aiEngineName, sc_idx), (False, False, ''))
  return generate_threejs_docking_viz(
    path,
    scenario_name=sc['name'],
    docked=docked,
    crashed=crashed,
    crash_reason=crash_reason,
  )
