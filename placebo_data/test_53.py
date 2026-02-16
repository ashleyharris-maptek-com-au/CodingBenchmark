"""Test 53: Spacecraft Orbital Docking Autopilot - Placebo responses."""


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
# Naive docking autopilot - simple proportional control in LVLH
# Uses sensor 1 for everything, no redundancy

def autopilot_step(state, dt):
    # Get relative position from radar (sensor 1)
    rx = state.get('radar_rel_pos_x_m', 0)
    ry = state.get('radar_rel_pos_y_m', 0)
    rz = state.get('radar_rel_pos_z_m', 0)
    
    vx = state.get('radar_rel_vel_x_ms', 0)
    vy = state.get('radar_rel_vel_y_ms', 0)
    vz = state.get('radar_rel_vel_z_ms', 0)
    
    rng = state.get('range_1_m', 1000)
    
    # Simple PD control on each axis
    # Scale gains based on range (gentler when close)
    if rng < 10:
        kp = 0.002
        kd = 0.05
    elif rng < 100:
        kp = 0.005
        kd = 0.02
    else:
        kp = 0.01
        kd = 0.01
    
    # Thrust commands: try to zero out relative position
    # For along-track (y): negative ry means behind, need positive vy to close
    thrust_x = -rx * kp - vx * kd
    thrust_y = -ry * kp - vy * kd
    thrust_z = -rz * kp - vz * kd
    
    # Clamp
    thrust_x = max(-1, min(1, thrust_x))
    thrust_y = max(-1, min(1, thrust_y))
    thrust_z = max(-1, min(1, thrust_z))
    
    return {
        'thrust_x': thrust_x,
        'thrust_y': thrust_y,
        'thrust_z': thrust_z,
        'att_rate_roll': 0,
        'att_rate_pitch': 0,
        'att_rate_yaw': 0,
    }
'''


def _naive(subpass):
  reasoning = ('Naive PD controller in LVLH frame. Directly thrusts to reduce '
               'relative position and velocity. Does NOT account for orbital '
               'mechanics (Hohmann transfers, CW equations). No sensor fusion.')
  return {"reasoning": reasoning, "python_code": NAIVE_CODE}, reasoning


def _best_published(subpass):
  reasoning = 'Best: CW-guided approach with Kalman filter. TODO.'
  code = 'def autopilot_step(state, dt): return {}'
  return {"reasoning": reasoning, "python_code": code}, reasoning


def _random(subpass):
  reasoning = 'Random: no thrust.'
  code = 'def autopilot_step(state, dt): return {}'
  return {"reasoning": reasoning, "python_code": code}, reasoning


def _human(subpass):
  reasoning = 'Human skeleton.'
  code = 'def autopilot_step(state, dt): return {}'
  return {"reasoning": reasoning, "python_code": code}, reasoning
