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

from visualization_utils import generate_threejs_flight_path

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
  TAXI_SPEED_KT,
  RWY_REMAINING_FRAC,
  HARD_LANDING_FPM,
  CRASH_LANDING_FPM,
  EMAS_LENGTH_M,
  GO_AROUND_ALT_FT,
  GO_AROUND_MAX,
  GO_AROUND_PENALTY,
)

title = "Aircraft Autoland (Python)"
TIMEOUT_SECONDS = 120

_HISTORY_CACHE = {}  # {(aiEngineName, subPass): history_list}


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

**Go-Around:**
If the approach is unstable or unsafe, abort the landing:
- Apply TOGA thrust (throttle 1.0), pitch up, retract speedbrake
- Climb to {GO_AROUND_ALT_FT}ft AGL — this completes the go-around
- The sim will then **restart** from a stable approach position with the
  **same failures and weather**. Your autopilot must land on the retry.
- Each go-around costs a **{GO_AROUND_PENALTY} point penalty** on a perfect landing.
- After **{GO_AROUND_MAX} go-arounds** the scenario fails (score 0.0).

**Scoring:**
- **1.0** — Normal landing: stopped on the runway, or slowed below {TAXI_SPEED_KT}kt
  with >{RWY_REMAINING_FRAC:.0%} runway remaining (pilot takes over to taxi).
  Minus {GO_AROUND_PENALTY} per go-around.
- **0.5** — Hard landing: touchdown vertical speed {HARD_LANDING_FPM}–{CRASH_LANDING_FPM} fpm.
  Aircraft needs inspection but is reusable, everyone walks away.
- **0.1** — Minor overrun: aircraft rolls off the end but is caught by the
  EMAS arrestor bed (within {EMAS_LENGTH_M}m past runway end). Everyone safe,
  aircraft damaged.
- **0.0** — Catastrophic: touchdown >{CRASH_LANDING_FPM} fpm, major overrun
  beyond EMAS, runway excursion, CFIT, or failure to land. Write-off or
  loss of life.

**Test Scenarios ({len(SCENARIOS)} total):**
{scenario_list}
Your single autopilot function must handle ALL scenarios.
Persistence: module-level variables persist across calls within a scenario,
but are **reset** between scenarios and between go-around attempts.
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


def _clamp_controls(controls):
  """Clamp control surface commands to valid ranges."""
  for k in ['elevator', 'aileron', 'rudder', 'elevator_trim', 'speedbrake']:
    if k in controls:
      controls[k] = max(-1.0, min(1.0, float(controls[k])))
  for k in ['throttle_1', 'throttle_2']:
    if k in controls:
      controls[k] = max(0.0, min(1.0, float(controls[k])))
  return controls


def _apply_scenario_effects(truth, sc, dt):
  """Apply all scenario-specific environmental/failure effects in one place."""
  import random as _rng

  # Engine failure at specified distance
  ef = sc.get('engine_fail')
  if ef:
    fail_dist_m = ef['at_dist_nm'] * 1852
    eidx = ef['engine']
    if truth['rwy_dist_m'] < fail_dist_m and truth['engine_running'][eidx]:
      truth['engine_running'][eidx] = False
      truth['n1'][eidx] = 0

  # Windshear
  ws = sc.get('windshear')
  if ws:
    alt_ft = truth['z'] * M2FT
    if 50 < alt_ft < ws['start_alt_ft']:
      factor = (ws['start_alt_ft'] - alt_ft) / ws['start_alt_ft']
      truth['u'] += ws['headwind_change_kt'] * KT2MS * factor * dt * 0.1
      truth['w'] += ws['downdraft_fps'] * FT2M * factor * dt * 0.05

  # Crosswind / gusts
  cw = sc.get('crosswind_kt')
  if cw:
    gust = sc.get('gust_kt', 0)
    wind_kt = cw + _rng.gauss(0, gust * 0.3)
    truth['rwy_offset_m'] += wind_kt * KT2MS * dt * 0.02

  # Turbulence
  if sc.get('turbulence'):
    truth['p'] += _rng.gauss(0, 0.02) * dt
    truth['q'] += _rng.gauss(0, 0.01) * dt
    truth['r'] += _rng.gauss(0, 0.01) * dt
    truth['u'] += _rng.gauss(0, 1.0) * dt
    truth['w'] += _rng.gauss(0, 0.5) * dt


def _run_scenario(runner, autopilot_fn, sc, max_steps=6000, dt=0.05):
  """Unified sim loop for all scenarios.

  Handles scenario effects, end-condition checks (crash, overrun, taxi-speed
  exit, full stop) and go-around detection.  Sets runner flags so the caller
  can inspect runner.went_around / runner.landed / runner.crashed.
  """
  was_below_1000 = False

  for step in range(max_steps):
    state = runner.sensors.build_state(runner.truth)
    state['target_hdg_deg'] = runner.target_hdg_deg
    state['target_ias_kt'] = runner.target_ias_kt

    _apply_scenario_effects(runner.truth, sc, dt)

    try:
      controls = autopilot_fn(state, dt)
      if not isinstance(controls, dict):
        controls = {}
    except Exception:
      controls = {}

    _clamp_controls(controls)
    runner.truth = autoland_physics_step(runner.truth, controls, dt)

    alt_ft = runner.truth['z'] * M2FT
    V = math.sqrt(runner.truth['u']**2 + runner.truth['v']**2 + runner.truth['w']**2)

    if alt_ft < 1000:
      was_below_1000 = True

    # ── Go-around detection ──
    if was_below_1000 and not runner.truth['on_ground'] and alt_ft > GO_AROUND_ALT_FT:
      runner.went_around = True
      return

    # ── Crash checks ──
    if alt_ft < -10:
      runner.crashed = True
      runner.crash_reason = 'CFIT: below ground level'
      return

    if runner.truth['on_ground'] and abs(runner.truth.get('touchdown_vs', 0)) > CRASH_LANDING_FPM:
      runner.crashed = True
      runner.crash_reason = f'Catastrophic landing: {abs(runner.truth["touchdown_vs"]):.0f} fpm'
      return

    if runner.truth['on_ground'] and abs(runner.truth['rwy_offset_m']) > ILS['runway_width_m'] / 2:
      runner.crashed = True
      runner.crash_reason = f'Runway excursion: {runner.truth["rwy_offset_m"]:.0f}m offset'
      return

    overrun_m = -runner.truth['rwy_dist_m'] - ILS['runway_length_m']
    if runner.truth['on_ground'] and overrun_m > EMAS_LENGTH_M:
      runner.crashed = True
      runner.crash_reason = f'Major overrun: {overrun_m:.0f}m past end (beyond EMAS)'
      return
    if runner.truth['on_ground'] and overrun_m > 0 and runner.truth.get('stopped'):
      runner.landed = True
      runner.outcome = 'overrun'
      return

    if V * MS2KT > AC['Vne_kt']:
      runner.crashed = True
      runner.crash_reason = f'Overspeed: {V * MS2KT:.0f} kt'
      return

    # ── Taxi-speed exit ──
    if runner.truth['on_ground'] and not runner.truth.get('stopped'):
      V_kt = math.sqrt(runner.truth['u']**2 + runner.truth['v']**2) * MS2KT
      remaining_m = ILS['runway_length_m'] + runner.truth['rwy_dist_m']
      remaining_frac = remaining_m / ILS['runway_length_m']
      if V_kt < TAXI_SPEED_KT and remaining_frac > RWY_REMAINING_FRAC:
        runner.landed = True
        runner.taxi_exit = True
        runner.outcome = 'taxi_exit'
        return

    # ── Full stop ──
    if runner.truth.get('stopped'):
      runner.landed = True
      runner.outcome = 'stopped'
      return

    # ── History ──
    if step % 20 == 0:
      runner.history.append({
        'time': runner.truth['time'],
        'alt_ft': alt_ft,
        'rwy_dist_m': runner.truth['rwy_dist_m'],
        'rwy_offset_m': runner.truth.get('rwy_offset_m', 0),
        'on_ground': runner.truth['on_ground'],
      })


def gradeAnswer(result, subPass, aiEngineName):
  if not result or 'python_code' not in result:
    return 0.0, 'No Python code provided'

  code = result['python_code']

  sc_idx = subPass
  if sc_idx >= len(SCENARIOS):
    return 0.0, f'Invalid subpass {sc_idx}'
  sc = SCENARIOS[sc_idx]
  hist_ref = f' [Ref: {sc.get("historical_ref", "")}]' if sc.get('historical_ref') else ''

  # ── Go-around restart loop ──
  for ga_attempt in range(GO_AROUND_MAX + 1):
    # Re-exec code each attempt so module-level state is fresh
    try:
      ns = {}
      exec(compile(code, '<autoland>', 'exec'), ns)
      if 'autopilot_step' not in ns:
        return 0.0, 'Code does not define autopilot_step(state, dt)'
      autopilot_fn = ns['autopilot_step']
    except Exception as e:
      return 0.0, f'Compilation failed: {e}'

    runner, _ = _setup_scenario(sc_idx)
    runner.go_arounds = ga_attempt
    print(f'  Scenario {sc_idx}: {sc["name"]} (attempt {ga_attempt + 1})')

    try:
      _run_scenario(runner, autopilot_fn, sc)
    except Exception as e:
      tb = traceback.format_exc()
      return 0.0, f'Simulation error: {e}\n{tb[:500]}'

    if runner.went_around:
      print(f'    Go-around #{ga_attempt + 1} — restarting from approach')
      continue

    score, details = runner.score()
    _HISTORY_CACHE[(aiEngineName, sc_idx)] = runner.history
    return score, f'[{sc["name"]}]{hist_ref} {details}'

  # Exhausted all go-around attempts
  return 0.0, f'[{sc["name"]}]{hist_ref} FAIL: {GO_AROUND_MAX} go-arounds exhausted'


# ──────────────────────────────────────────────────────────────────────────────
# Visualization for HTML report
# ──────────────────────────────────────────────────────────────────────────────
def _history_to_path(history):
  """Convert autoland history to [x, y, z] path in metres.

  x = distance along runway (rwy_dist_m, positive = before threshold),
  y = lateral offset (rwy_offset_m),
  z = altitude in metres.
  """
  if not history:
    return []
  path = []
  for h in history:
    alt_m = h['alt_ft'] * FT2M
    path.append([
      round(h['rwy_dist_m'], 1),
      round(h.get('rwy_offset_m', 0), 1),
      round(alt_m, 1),
    ])
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
  runway = {
    'x': -ILS['runway_length_m'] / 2,  # center of runway
    'y': 0,
    'length': ILS['runway_length_m'],
    'width': ILS['runway_width_m'],
  }
  return generate_threejs_flight_path(
    path, scenario_name=sc['name'], runway=runway)
