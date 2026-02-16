"""
Test 54: Aircraft Autoland (Python)

The LLM must write a Python autopilot to land an airliner using ILS.
Reuses the flight physics from test 51, adds ILS glideslope/localizer,
ground contact, braking, reverse thrust, and spoilers.

Tested across 25 scenarios including normal approaches, engine failures,
windshear, sensor failures, slippery runways, and historical disasters.
"""

import math
import traceback

from autoland_sim import (
  make_autoland_truth,
  AutolandSensorModel,
  AutolandRunner,
  autoland_physics_step,
  list_autoland_state_keys,
  ILS,
  AC,
  DEG2RAD,
  RAD2DEG,
  FT2M,
  M2FT,
  KT2MS,
  MS2KT,
)

title = "Aircraft Autoland (Python)"
TIMEOUT_SECONDS = 120


# ──────────────────────────────────────────────────────────────────────────────
# Scenarios
# ──────────────────────────────────────────────────────────────────────────────
def _sc(name, desc, **kw):
  return {'name': name, 'description': desc, **kw}


SCENARIOS = [
  # 0: Normal ILS approach
  _sc('Normal ILS approach CAT I',
      'Standard 3° glideslope approach from 10nm, calm conditions.',
      dist_nm=10,
      ias_kt=160),

  # 1: Short final
  _sc('Short final — 3nm',
      'Starting on short final, 3nm out, already configured.',
      dist_nm=3,
      ias_kt=145),

  # 2: High approach
  _sc('High on glideslope',
      'Starting 300ft above glideslope. Must capture from above.',
      dist_nm=8,
      alt_offset_ft=300,
      ias_kt=160),

  # 3: Low approach
  _sc('Low on glideslope',
      'Starting 200ft below glideslope. Must capture from below.',
      dist_nm=8,
      alt_offset_ft=-200,
      ias_kt=160),

  # 4: Offset from centerline
  _sc('Offset from centerline',
      'Starting 100m right of centerline. Must track localizer.',
      dist_nm=8,
      offset_m=100,
      ias_kt=160),

  # 5: Engine failure on final (Kegworth / BA 38)
  _sc('Single engine failure on final',
      'Left engine fails at 5nm. Land on one engine.',
      dist_nm=8,
      ias_kt=160,
      engine_fail={
        'engine': 0,
        'at_dist_nm': 5
      },
      historical_ref='Kegworth (1989) / BA 38 (2008)'),

  # 6: Windshear on final (Delta 191)
  _sc('Windshear on short final',
      'Microburst wind shear at 500ft AGL. Requires escape or recovery.',
      dist_nm=5,
      ias_kt=150,
      windshear={
        'start_alt_ft': 500,
        'headwind_change_kt': -40,
        'downdraft_fps': 20
      },
      historical_ref='Delta 191 (1985)'),

  # 7: Crosswind landing
  _sc('Strong crosswind — 25kt',
      'Landing with 25kt crosswind from the left.',
      dist_nm=5,
      ias_kt=150,
      crosswind_kt=25),

  # 8: Gusty conditions
  _sc('Gusty crosswind — 15kt gusting 30',
      'Variable crosswind with gusts. Maintain stable approach.',
      dist_nm=5,
      ias_kt=150,
      crosswind_kt=15,
      gust_kt=15),

  # 9: Low visibility CAT III
  _sc('CAT III low visibility',
      'Autoland in CAT III conditions. Decision height 50ft.',
      dist_nm=8,
      ias_kt=150),

  # 10: Slippery runway
  _sc('Slippery wet runway',
      'Wet runway, braking action poor. Need maximum braking distance.',
      dist_nm=5,
      ias_kt=145,
      runway_friction=0.3,
      historical_ref='Overruns at LGA, MDW'),

  # 11: Contaminated runway (ice)
  _sc('Icy runway',
      'Ice-contaminated runway. Braking extremely poor.',
      dist_nm=5,
      ias_kt=145,
      runway_friction=0.1),

  # 12: Glideslope sensor failure
  _sc('Glideslope receiver 1+2 failed',
      'Two of three glideslope receivers fail. Use remaining + radar alt.',
      dist_nm=8,
      ias_kt=160,
      failures={
        'gs_1': ('dead', 0),
        'gs_2': ('dead', 0)
      }),

  # 13: Localizer failure
  _sc('Localizer 1 stuck + 2 biased',
      'Localizer 1 stuck, 2 has 0.5 dot bias. Only 3 is reliable.',
      dist_nm=8,
      ias_kt=160,
      failures={
        'loc_1': ('stuck', 0),
        'loc_2': ('bias', 0.5)
      }),

  # 14: Radar altimeter failure
  _sc('All radar altimeters failed',
      'No radar alt. Must use baro alt and glideslope for flare.',
      dist_nm=5,
      ias_kt=150,
      failures={
        'ralt_1': ('dead', 0),
        'ralt_2': ('dead', 0),
        'ralt_3': ('dead', 0),
      }),

  # 15: Flight controls sticking
  _sc('Elevator partially jammed',
      'Elevator control surface partially stuck. Reduced authority.',
      dist_nm=5,
      ias_kt=150,
      elevator_jam=0.3),

  # 16: Aileron jam
  _sc('Aileron stuck 5° left',
      'Left aileron jammed at 5° deflection. Causes rolling tendency.',
      dist_nm=5,
      ias_kt=150,
      aileron_jam=5 * DEG2RAD),

  # 17: Fast approach
  _sc('Unstabilised — too fast',
      'Starting at 180kt, must slow to Vref before landing.',
      dist_nm=5,
      ias_kt=180),

  # 18: Heavy aircraft
  _sc('Heavy weight landing',
      'Landing at maximum landing weight. Higher Vref, longer rollout.',
      dist_nm=8,
      ias_kt=160,
      mass=75000),

  # 19: Asymmetric flaps
  _sc('Asymmetric flap extension',
      'Left flap stuck at 10°, right at 20°. Rolling moment.',
      dist_nm=5,
      ias_kt=155,
      flap_asym=True),

  # 20: Dual engine failure (Gimli / Sully)
  _sc('Dual engine failure — glide approach',
      'Both engines flamed out. Must glide to runway.',
      dist_nm=4,
      ias_kt=200,
      dual_engine_fail=True,
      historical_ref='Gimli Glider (1983) / US Airways 1549 (2009)'),

  # 21: Go-around scenario
  _sc('Missed approach required',
      'Unstable approach at 500ft. Should go around and re-approach.',
      dist_nm=3,
      ias_kt=180,
      alt_offset_ft=200),

  # 22: Turbulence on approach
  _sc('Severe turbulence on final',
      'Strong turbulence causing pitch/roll disturbances.',
      dist_nm=5,
      ias_kt=150,
      turbulence=True),

  # 23: Flickering ILS
  _sc('ILS signal flickering',
      'Both glideslope and localizer signals intermittent.',
      dist_nm=5,
      ias_kt=150,
      flicker_ils=True),

  # 24: Combined: engine fail + crosswind + slippery
  _sc('Engine failure + crosswind + wet runway',
      'Single engine failure with 20kt crosswind on wet runway.',
      dist_nm=5,
      ias_kt=155,
      engine_fail={
        'engine': 0,
        'at_dist_nm': 4
      },
      crosswind_kt=20,
      runway_friction=0.4,
      historical_ref='Multiple incidents'),
]


# ──────────────────────────────────────────────────────────────────────────────
# Prompt
# ──────────────────────────────────────────────────────────────────────────────
def prepareSubpassPrompt(subPass):
  if subPass != 0:
    raise StopIteration

  state_keys = list_autoland_state_keys()
  num_keys = len(state_keys)

  sensor_groups = {
    'ILS Glideslope (x3)': [k for k in state_keys if 'glideslope' in k],
    'ILS Localizer (x3)': [k for k in state_keys if 'localizer' in k],
    'Radio/Radar Altimeters': [k for k in state_keys if 'ralt' in k or 'radio_alt' in k],
    'Baro Altimeters (x3)': [k for k in state_keys if 'baro_alt' in k],
    'Airspeed (x3)': [k for k in state_keys if 'ias_' in k or 'tas_' in k or 'mach_' in k],
    'AoA (x3)': [k for k in state_keys if 'aoa_' in k],
    'GPS': [k for k in state_keys if 'gps_' in k],
    'IRS/Attitude': [k for k in state_keys if 'irs_' in k],
    'DME': [k for k in state_keys if 'dme_' in k],
    'Engine': [k for k in state_keys if 'engine_' in k or 'n1_' in k or 'egt_' in k],
    'Ground': [
      k for k in state_keys if k in
      ['weight_on_wheels', 'on_ground', 'groundspeed_kt', 'radio_alt_ft', 'below_decision_height']
    ],
    'Targets': [k for k in state_keys if k.startswith('target_')],
  }

  sensor_doc = ''
  for gname, keys in sensor_groups.items():
    if keys:
      sensor_doc += f'\n  **{gname}** ({len(keys)} keys):\n'
      for k in sorted(set(keys)):
        sensor_doc += f'    {k}\n'

  scenario_list = ''
  for i, sc in enumerate(SCENARIOS):
    ref = f' [{sc.get("historical_ref", "")}]' if sc.get('historical_ref') else ''
    scenario_list += f'  {i:2d}. {sc["name"]}{ref}\n'

  return f"""Write a Python autopilot to land an airliner using ILS approach and autoland.

**Function Signature:**
```python
def autopilot_step(state: dict, dt: float) -> dict:
```

**Input:** `state` is a dict with {num_keys} keys from redundant sensors:
{sensor_doc}
**ILS Approach Key Concepts:**
- `glideslope_N_dots`: deviation from 3° glideslope (+above/-below), 3 receivers
- `localizer_N_dots`: deviation from runway centerline (+right/-left), 3 receivers
- `radio_alt_ft`: height above ground (critical for flare)
- `dme_N_nm`: distance to runway threshold
- `weight_on_wheels`: True when landed
- `target_vref_kt`: reference approach speed (typically 137kt)
- `decision_height_ft`: minimum altitude to continue approach (200ft CAT I)
- Elevator command 0 = trimmed flight. Positive = pitch up.

**Output:** Return a dict with any of:
```
  elevator:     -1.0 (nose down) to +1.0 (nose up), relative to trim
  aileron:      -1.0 (roll left) to +1.0 (roll right)
  rudder:       -1.0 (yaw left) to +1.0 (yaw right)
  throttle_1:   0.0 to 1.0 (left engine)
  throttle_2:   0.0 to 1.0 (right engine)
  elevator_trim: -1.0 to +1.0
  flaps:        degrees (0-40)
  gear:         0.0 (up) or 1.0 (down)
  speedbrake:   0.0 (retracted) to 1.0 (full)
  spoiler:      0.0 or 1.0 (ground spoilers, deploy after touchdown)
  autobrake:    0.0 to 1.0 (braking force after touchdown)
  reverse_1:    0.0 to 1.0 (left engine reverse thrust)
  reverse_2:    0.0 to 1.0 (right engine reverse thrust)
```

**Approach Phases:**
1. **Intercept** (>1000ft): Track glideslope and localizer, configure for landing
2. **Approach** (1000ft-200ft): Stabilise on glideslope, speed, config
3. **Flare** (<50ft radio alt): Reduce descent rate, idle thrust
4. **Touchdown & Rollout**: Deploy spoilers, reverse thrust, autobrake
5. **Go-around**: If unstable below 500ft, apply TOGA thrust, pitch up, retract

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
      "description": "Explain your autoland design"
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
  gs_deg = sc.get('gs_deg', 3.0)
  dist_nm = sc.get('dist_nm', 10)
  ias = sc.get('ias_kt', 160)
  mass = sc.get('mass', 65000)
  rwy_hdg = sc.get('rwy_heading', 90)

  dist_m = dist_nm * 1852
  alt_ft = dist_m * math.tan(gs_deg * DEG2RAD) * M2FT + 50
  alt_ft += sc.get('alt_offset_ft', 0)

  truth = make_autoland_truth(dist_nm=dist_nm,
                              alt_ft=alt_ft,
                              ias_kt=ias,
                              heading_deg=rwy_hdg,
                              mass=mass,
                              gs_deg=gs_deg,
                              rwy_heading=rwy_hdg)

  if sc.get('offset_m'):
    truth['rwy_offset_m'] = sc['offset_m']

  if sc.get('dual_engine_fail'):
    truth['engine_running'] = [False, False]
    truth['n1'] = [0, 0]
    truth['throttle'] = [0, 0]

  if sc.get('elevator_jam') is not None:
    truth['elevator_jam'] = sc['elevator_jam']

  if sc.get('aileron_jam') is not None:
    truth['aileron_jam'] = sc['aileron_jam']

  if sc.get('flap_asym'):
    truth['flap_deg'] = 15  # average
    truth['cl_roll_extra'] = 0.02  # rolling moment from asymmetric flaps

  sensor = AutolandSensorModel(seed=42 + idx)
  for key, (mode, param) in sc.get('failures', {}).items():
    sensor.set_failure(key, mode, param)

  if sc.get('flicker_ils'):
    for i in range(1, 4):
      sensor.set_flicker(f'gs_{i}')
      sensor.set_flicker(f'loc_{i}')

  runner = AutolandRunner(truth,
                          sensor,
                          target_hdg_deg=rwy_hdg,
                          target_ias_kt=sc.get('target_ias', 137))
  return runner, sc


def gradeAnswer(result, subPass, aiEngineName):
  if not result or 'python_code' not in result:
    return 0.0, 'No Python code provided'

  code = result['python_code']
  try:
    ns = {}
    exec(compile(code, '<autoland>', 'exec'), ns)
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
  print(f'  Scenario {sc_idx}: {sc["name"]}')

  try:
    if sc.get('engine_fail'):
      _run_engine_fail(runner, autopilot_fn, sc)
    elif sc.get('windshear'):
      _run_windshear(runner, autopilot_fn, sc)
    elif sc.get('crosswind_kt'):
      _run_crosswind(runner, autopilot_fn, sc)
    elif sc.get('turbulence'):
      _run_turbulence(runner, autopilot_fn, sc)
    else:
      runner.run(autopilot_fn, steps=sc.get('steps', 6000))
  except Exception as e:
    tb = traceback.format_exc()
    return 0.0, f'Simulation error: {e}\n{tb[:500]}'

  score, details = runner.score()
  hist = f' [Ref: {sc.get("historical_ref", "")}]' if sc.get('historical_ref') else ''
  return score, f'[{sc["name"]}]{hist} {details}'


def _run_engine_fail(runner, autopilot_fn, sc):
  """Engine fails at specified distance."""
  fail_dist = sc['engine_fail']['at_dist_nm'] * 1852
  engine_idx = sc['engine_fail']['engine']
  failed = False
  dt = 0.05
  friction = sc.get('runway_friction', 1.0)

  for step in range(6000):
    state = runner.sensors.build_state(runner.truth)
    state['target_hdg_deg'] = runner.target_hdg_deg
    state['target_ias_kt'] = runner.target_ias_kt

    if not failed and runner.truth['rwy_dist_m'] < fail_dist:
      runner.truth['engine_running'][engine_idx] = False
      runner.truth['n1'][engine_idx] = 0
      failed = True

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

    runner.truth = autoland_physics_step(runner.truth, controls, dt)

    # Apply crosswind if present
    if sc.get('crosswind_kt'):
      _apply_crosswind(runner.truth, sc['crosswind_kt'], sc.get('gust_kt', 0), dt)

    alt_ft = runner.truth['z'] * M2FT
    V = math.sqrt(runner.truth['u']**2 + runner.truth['v']**2 + runner.truth['w']**2)
    ias_kt = runner.truth.get('ias_kt_approx', V * MS2KT)

    if alt_ft < -10:
      runner.crashed = True
      runner.crash_reason = 'CFIT'
      break
    if runner.truth['on_ground'] and abs(runner.truth.get('touchdown_vs', 0)) > 600:
      runner.crashed = True
      runner.crash_reason = f'Hard landing: {abs(runner.truth["touchdown_vs"]):.0f} fpm'
      break
    if runner.truth.get('stopped'):
      runner.landed = True
      break

    if step % 20 == 0:
      runner.history.append({
        'time': runner.truth['time'],
        'alt_ft': alt_ft,
        'rwy_dist_m': runner.truth['rwy_dist_m'],
      })


def _run_windshear(runner, autopilot_fn, sc):
  """Windshear encounter on approach."""
  ws = sc['windshear']
  dt = 0.05
  for step in range(6000):
    state = runner.sensors.build_state(runner.truth)
    state['target_hdg_deg'] = runner.target_hdg_deg
    state['target_ias_kt'] = runner.target_ias_kt

    alt_ft = runner.truth['z'] * M2FT

    # Apply windshear effect
    if alt_ft < ws['start_alt_ft'] and alt_ft > 50:
      # Headwind decreases (performance-decreasing shear)
      shear_factor = (ws['start_alt_ft'] - alt_ft) / ws['start_alt_ft']
      runner.truth['u'] += ws['headwind_change_kt'] * KT2MS * shear_factor * dt * 0.1
      runner.truth['w'] += ws['downdraft_fps'] * FT2M * shear_factor * dt * 0.05

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

    runner.truth = autoland_physics_step(runner.truth, controls, dt)

    if alt_ft < -10:
      runner.crashed = True
      runner.crash_reason = 'CFIT during windshear'
      break
    if runner.truth['on_ground'] and abs(runner.truth.get('touchdown_vs', 0)) > 600:
      runner.crashed = True
      runner.crash_reason = f'Hard landing: {abs(runner.truth["touchdown_vs"]):.0f} fpm'
      break
    if runner.truth.get('stopped'):
      runner.landed = True
      break

    if step % 20 == 0:
      runner.history.append({
        'time': runner.truth['time'],
        'alt_ft': alt_ft,
        'rwy_dist_m': runner.truth['rwy_dist_m'],
      })


def _apply_crosswind(truth, base_kt, gust_kt, dt):
  """Apply lateral wind to truth state."""
  import random
  wind_kt = base_kt + random.gauss(0, gust_kt * 0.3)
  wind_ms = wind_kt * KT2MS
  truth['rwy_offset_m'] += wind_ms * dt * 0.02  # small lateral push


def _run_crosswind(runner, autopilot_fn, sc):
  """Approach with crosswind."""
  dt = 0.05
  for step in range(6000):
    state = runner.sensors.build_state(runner.truth)
    state['target_hdg_deg'] = runner.target_hdg_deg
    state['target_ias_kt'] = runner.target_ias_kt

    _apply_crosswind(runner.truth, sc['crosswind_kt'], sc.get('gust_kt', 0), dt)

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

    runner.truth = autoland_physics_step(runner.truth, controls, dt)

    alt_ft = runner.truth['z'] * M2FT
    if alt_ft < -10:
      runner.crashed = True
      runner.crash_reason = 'CFIT'
      break
    if runner.truth['on_ground'] and abs(runner.truth.get('touchdown_vs', 0)) > 600:
      runner.crashed = True
      runner.crash_reason = f'Hard landing: {abs(runner.truth["touchdown_vs"]):.0f} fpm'
      break
    if runner.truth.get('stopped'):
      runner.landed = True
      break

    if step % 20 == 0:
      runner.history.append({
        'time': runner.truth['time'],
        'alt_ft': alt_ft,
        'rwy_dist_m': runner.truth['rwy_dist_m'],
      })


def _run_turbulence(runner, autopilot_fn, sc):
  """Approach with turbulence disturbances."""
  import random
  dt = 0.05
  for step in range(6000):
    state = runner.sensors.build_state(runner.truth)
    state['target_hdg_deg'] = runner.target_hdg_deg
    state['target_ias_kt'] = runner.target_ias_kt

    # Add turbulence
    runner.truth['p'] += random.gauss(0, 0.02) * dt
    runner.truth['q'] += random.gauss(0, 0.01) * dt
    runner.truth['r'] += random.gauss(0, 0.01) * dt
    runner.truth['u'] += random.gauss(0, 1.0) * dt
    runner.truth['w'] += random.gauss(0, 0.5) * dt

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

    runner.truth = autoland_physics_step(runner.truth, controls, dt)

    alt_ft = runner.truth['z'] * M2FT
    if alt_ft < -10:
      runner.crashed = True
      runner.crash_reason = 'CFIT in turbulence'
      break
    if runner.truth['on_ground'] and abs(runner.truth.get('touchdown_vs', 0)) > 600:
      runner.crashed = True
      runner.crash_reason = f'Hard landing: {abs(runner.truth["touchdown_vs"]):.0f} fpm'
      break
    if runner.truth.get('stopped'):
      runner.landed = True
      break

    if step % 20 == 0:
      runner.history.append({
        'time': runner.truth['time'],
        'alt_ft': alt_ft,
        'rwy_dist_m': runner.truth['rwy_dist_m'],
      })
