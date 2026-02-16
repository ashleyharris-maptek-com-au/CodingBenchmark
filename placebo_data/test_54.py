"""Test 54: Aircraft Autoland - Placebo responses."""


def get_response(model_name, subpass):
  if model_name == 'naive':
    return _naive(subpass)
  elif model_name == 'naive-optimised':
    return _naive(subpass)
  elif model_name == 'best-published':
    return _best_published(subpass)
  elif model_name == 'random':
    return _random(subpass)
  elif model_name == 'human':
    return _human(subpass)
  return None, ''


NAIVE_CODE = r'''
# Naive autoland: VS-based descent + glideslope correction
_int_gs = 0.0
_int_loc = 0.0
_phase = 'approach'
_thr = None  # initialise from current state

def autopilot_step(state, dt):
    global _int_gs, _int_loc, _phase, _thr

    alt = state.get('radio_alt_ft', state.get('baro_alt_1_ft', 1000))
    gs_dots = state.get('glideslope_1_dots', 0)
    loc_dots = state.get('localizer_1_dots', 0)
    ias = state.get('ias_1_kt', 150)
    vs = state.get('vertical_speed_fpm', 0)
    vref = state.get('target_vref_kt', 137)
    wow = state.get('weight_on_wheels', False)

    # Initialise throttle from current state (don't jerk)
    if _thr is None:
        _thr = state.get('engine_1_n1_pct', 60) / 100.0

    controls = {}

    # Phase detection
    if wow or state.get('on_ground', False):
        _phase = 'rollout'
    elif alt < 40 and _phase == 'approach':
        _phase = 'flare'

    if _phase == 'rollout':
        controls['spoiler'] = 1.0
        controls['autobrake'] = 0.8
        controls['reverse_1'] = 0.8
        controls['reverse_2'] = 0.8
        controls['throttle_1'] = 0.0
        controls['throttle_2'] = 0.0
        controls['elevator'] = 0.0
        return controls

    if _phase == 'flare':
        controls['elevator'] = 0.08
        controls['throttle_1'] = 0.0
        controls['throttle_2'] = 0.0
        controls['aileron'] = -loc_dots * 0.02
        controls['rudder'] = -loc_dots * 0.01
        return controls

    # === Approach phase ===
    # Target VS for 3-degree glideslope at current groundspeed
    gs_kt = state.get('groundspeed_kt', ias)
    target_vs = -gs_kt * 5.3  # ~5.3 fpm per kt for 3 degrees

    # Adjust target VS based on glideslope dots
    # Above GS -> descend faster, below GS -> descend slower
    target_vs -= gs_dots * 200  # 200 fpm correction per dot

    # Elevator: control descent rate (VS)
    vs_err = target_vs - vs  # negative when descending too fast
    elevator = vs_err * 0.0003
    elevator = max(-0.3, min(0.3, elevator))
    controls['elevator'] = elevator

    # Throttle: control speed (slow adjustment from current)
    target_ias = vref + 5
    spd_err = target_ias - ias  # negative when too fast
    _thr += spd_err * 0.0005 * dt
    _thr = max(0.0, min(0.9, _thr))
    controls['throttle_1'] = _thr
    controls['throttle_2'] = _thr

    # Localizer tracking via aileron/rudder
    # Also use heading to damp lateral oscillation
    hdg = state.get('irs_1_heading_deg', 0)
    target_hdg = state.get('target_hdg_deg', 90)
    hdg_err = hdg - target_hdg
    if hdg_err > 180: hdg_err -= 360
    if hdg_err < -180: hdg_err += 360

    _int_loc += loc_dots * dt
    _int_loc = max(-3, min(3, _int_loc))
    aileron = -loc_dots * 0.05 - _int_loc * 0.01 - hdg_err * 0.01
    controls['aileron'] = max(-0.3, min(0.3, aileron))
    controls['rudder'] = -loc_dots * 0.02 - hdg_err * 0.005

    # Landing config
    controls['gear'] = 1.0
    controls['flaps'] = 30

    return controls
'''


def _naive(subpass):
  reasoning = ('Naive autoland: uses glideslope/localizer sensor 1 only, '
               'PI tracking, basic flare at 50ft radio alt, spoilers/reverse '
               'on touchdown. No sensor fusion, no go-around logic.')
  return {"reasoning": reasoning, "python_code": NAIVE_CODE}, reasoning


def _best_published(subpass):
  reasoning = 'Best: Full autoland with sensor fusion and go-around. TODO.'
  code = 'def autopilot_step(state, dt): return {"throttle_1": 0.3, "throttle_2": 0.3}'
  return {"reasoning": reasoning, "python_code": code}, reasoning


def _random(subpass):
  reasoning = 'Random: no control.'
  code = 'def autopilot_step(state, dt): return {}'
  return {"reasoning": reasoning, "python_code": code}, reasoning


def _human(subpass):
  reasoning = 'Human skeleton.'
  code = 'def autopilot_step(state, dt): return {"throttle_1": 0.3, "throttle_2": 0.3}'
  return {"reasoning": reasoning, "python_code": code}, reasoning
