"""Test 51: Airliner Autopilot (Python) - Placebo responses."""


def get_response(model_name, subpass):
  if model_name == 'naive':
    return _naive(subpass)
  elif model_name == 'naive-optimised':
    return _naive_optimised(subpass)
  elif model_name == 'best-published':
    return _best_published(subpass)
  elif model_name == 'random':
    return _random(subpass)
  elif model_name == 'human':
    return _human(subpass)
  return None, ''


NAIVE_CODE = r'''
# Naive PID autopilot - uses only sensor 1, no redundancy, no fault tolerance
_prev_alt_err = 0
_prev_hdg_err = 0
_int_alt = 0
_throttle = None  # will be set from initial state

def autopilot_step(state, dt):
    global _prev_alt_err, _prev_hdg_err, _int_alt, _throttle

    # Just use sensor 1 for everything (no redundancy check!)
    alt = state.get('baro_alt_1_ft', 0)
    ias = state.get('ias_1_kt', 250)
    hdg = state.get('irs_1_heading_deg', 0)
    roll = state.get('irs_1_roll_deg', 0)
    vs = state.get('vertical_speed_fpm', 0)

    tgt_alt = state.get('target_alt_ft', alt)
    tgt_hdg = state.get('target_hdg_deg', hdg)
    tgt_ias = state.get('target_ias_kt', 250)

    # Initialize throttle from current engine state
    if _throttle is None:
        _throttle = state.get('throttle_1_pos', 0.5)

    # Altitude hold: error -> target VS -> elevator
    alt_err = tgt_alt - alt
    _int_alt += alt_err * dt
    _int_alt = max(-5000, min(5000, _int_alt))
    d_alt = (alt_err - _prev_alt_err) / max(dt, 0.001)
    _prev_alt_err = alt_err

    tgt_vs = max(-2000, min(2000, alt_err * 0.3))
    vs_err = tgt_vs - vs

    # Elevator is RELATIVE TO TRIM (0 = level flight)
    elevator = vs_err * 0.0003 + _int_alt * 0.000001
    elevator = max(-0.5, min(0.5, elevator))

    # Heading hold via bank angle
    hdg_err = tgt_hdg - hdg
    if hdg_err > 180: hdg_err -= 360
    if hdg_err < -180: hdg_err += 360
    _prev_hdg_err = hdg_err

    tgt_roll = max(-25, min(25, hdg_err * 1.0))
    roll_err = tgt_roll - roll
    aileron = roll_err * 0.015
    aileron = max(-1, min(1, aileron))

    # Yaw damper
    yaw_rate = state.get('gyro_1_yaw_rate_dps', 0)
    rudder = -yaw_rate * 0.03
    rudder = max(-1, min(1, rudder))

    # Speed hold: adjust throttle from current baseline
    spd_err = tgt_ias - ias
    _throttle += spd_err * 0.0002 * dt
    _throttle = max(0.05, min(1.0, _throttle))

    return {
        'elevator': elevator,
        'aileron': aileron,
        'rudder': rudder,
        'throttle_1': _throttle,
        'throttle_2': _throttle,
        'elevator_trim': 0,
        'flaps': state.get('flap_pos_left_deg', 0),
        'speedbrake': 0,
        'gear': 1 if state.get('gear_handle_down', False) else 0,
    }
'''


def _naive(subpass):
  reasoning = ('Naive PID autopilot using only sensor 1 (no redundancy). '
               'Simple altitude/heading/speed hold with P and D terms. '
               'No sensor fusion, no stall protection, no fault tolerance.')
  return {"reasoning": reasoning, "python_code": NAIVE_CODE}, reasoning


def _naive_optimised(subpass):
  return _naive(subpass)


def _best_published(subpass):
  reasoning = ('Best published: Robust adaptive autopilot with '
               'triple-redundant sensor voting, Kalman filter state estimation, '
               'and gain-scheduled PID with envelope protection. '
               'Based on modern FBW architecture (Airbus Normal Law). '
               'TODO: Full implementation pending.')
  code = '# TODO: Implement robust adaptive autopilot\ndef autopilot_step(state, dt): return {}'
  return {"reasoning": reasoning, "python_code": code}, reasoning


def _random(subpass):
  reasoning = 'Random: returns zero controls (no input).'
  code = 'def autopilot_step(state, dt): return {}'
  return {"reasoning": reasoning, "python_code": code}, reasoning


def _human(subpass):
  reasoning = 'Human starting point for autopilot. Fill in sensor fusion and control logic.'
  code = r'''
# Human autopilot skeleton
def autopilot_step(state, dt):
    # TODO: Sensor fusion - cross-check redundant sensors
    # TODO: State estimation (altitude, speed, attitude)
    # TODO: Envelope protection (stall, overspeed)
    # TODO: Attitude control (PID for pitch/roll/yaw)
    # TODO: Altitude hold
    # TODO: Heading hold
    # TODO: Speed hold
    # TODO: Engine failure handling (asymmetric thrust)
    return {
        'elevator': 0, 'aileron': 0, 'rudder': 0,
        'throttle_1': 0.7, 'throttle_2': 0.7,
        'elevator_trim': 0, 'flaps': 0, 'speedbrake': 0, 'gear': 0,
    }
'''
  return {"reasoning": reasoning, "python_code": code}, reasoning
