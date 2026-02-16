"""Test 52: Self-Driving Car Highway Autopilot - Placebo responses."""


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
# Naive car autopilot - uses vision system 1, simple P lane keeping
_filtered_offset = 0.0

def autopilot_step(state, dt):
    global _filtered_offset

    # Lane offset from vision (positive = right of center)
    offset = state.get('vision_1_lane_offset_m', 0)
    # Low-pass filter to reject noise
    _filtered_offset = _filtered_offset * 0.85 + offset * 0.15

    # Heading correction (GPS heading: 0 = straight ahead on road)
    heading = state.get('gps_1_heading_deg', 0)

    # PD steering: P on offset + D via heading (heading error = drift rate)
    steering = -_filtered_offset * 0.08 - heading * 0.008
    steering = max(-0.2, min(0.2, steering))

    # Speed control
    speed = state.get('vehicle_speed_1_kmh', 100)
    target = state.get('target_speed_kmh', 110)

    # Check for obstacles
    obs_dist = state.get('vision_1_obstacle_dist_m', 200)
    obs_detected = state.get('vision_1_obstacle_detected', False)

    throttle = 0.0
    brake = 0.0

    if obs_detected and obs_dist < 50:
        # Brake proportional to closeness
        brake = max(0, (50 - obs_dist) / 50.0)
        throttle = 0.0
    else:
        speed_err = target - speed
        if speed_err > 0:
            throttle = min(1.0, speed_err * 0.02)
        else:
            brake = min(1.0, -speed_err * 0.01)

    # Reduce speed on low friction
    mu = state.get('estimated_mu', 1.0)
    if mu < 0.5:
        target_reduced = target * mu
        if speed > target_reduced:
            brake = min(0.3, (speed - target_reduced) * 0.01)
            throttle = 0

    return {
        'steering': steering,
        'throttle': throttle,
        'brake': brake,
    }
'''


def _naive(subpass):
  reasoning = ('Naive car autopilot: uses vision system 1 only for lane keeping, '
               'simple proportional speed control, basic obstacle braking. '
               'No sensor fusion, no redundancy, no advanced traction management.')
  return {"reasoning": reasoning, "python_code": NAIVE_CODE}, reasoning


def _best_published(subpass):
  reasoning = 'Best published: MPC-based with sensor fusion. TODO.'
  code = 'def autopilot_step(state, dt): return {"steering": 0, "throttle": 0.3, "brake": 0}'
  return {"reasoning": reasoning, "python_code": code}, reasoning


def _random(subpass):
  reasoning = 'Random: no control input.'
  code = 'def autopilot_step(state, dt): return {}'
  return {"reasoning": reasoning, "python_code": code}, reasoning


def _human(subpass):
  reasoning = 'Human skeleton.'
  code = 'def autopilot_step(state, dt): return {"steering": 0, "throttle": 0.3, "brake": 0}'
  return {"reasoning": reasoning, "python_code": code}, reasoning
