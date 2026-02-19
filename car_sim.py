"""
Self-Driving Car Highway Simulator

Simplified 2D vehicle dynamics for a passenger car on a multi-lane freeway.
All units metric (m, m/s, m/s², rad, kg).

Provides:
- Vehicle physics (bicycle model with tire slip)
- ~150 sensor properties including redundant sensors and image parser outputs
- Failure injection (sensor, traction, obstacles)
- Scenario-based testing
"""

import math
import random
from typing import Dict, Optional, List, Tuple

# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────
G = 9.80665
DEG2RAD = math.pi / 180.0
RAD2DEG = 180.0 / math.pi

# ──────────────────────────────────────────────────────────────────────────────
# Vehicle Parameters (mid-size sedan)
# ──────────────────────────────────────────────────────────────────────────────
CAR = {
  'mass': 1600,  # kg
  'wheelbase': 2.7,  # m
  'track_width': 1.55,  # m (distance between left/right wheels)
  'Lf': 1.35,  # CG to front axle
  'Lr': 1.35,  # CG to rear axle
  'Iz': 2500,  # yaw inertia kg·m²
  'Cf': 80000,  # front cornering stiffness N/rad
  'Cr': 85000,  # rear cornering stiffness N/rad
  'Cd': 0.32,  # drag coefficient
  'A_front': 2.2,  # frontal area m²
  'rho_air': 1.225,  # air density kg/m³
  'tire_radius': 0.33,  # m
  'max_engine_force': 7000,  # N (at wheels)
  'max_brake_force': 15000,  # N (per axle)
  'max_steer_angle': 35 * DEG2RAD,
  'steer_rate_limit': 60 * DEG2RAD,  # rad/s
  'brake_rate_limit': 50000,  # N/s
  'throttle_rate_limit': 2.0,  # per second
  'wheel_speed_sensors': 4,
}

# Lane geometry
LANE_WIDTH = 3.7  # m
NUM_LANES = 3
ROAD_WIDTH = LANE_WIDTH * NUM_LANES


# ──────────────────────────────────────────────────────────────────────────────
# Road Model
# ──────────────────────────────────────────────────────────────────────────────
class RoadModel:
  """Straight/gently curved multi-lane freeway."""

  def __init__(self, curvature=0.0, speed_limit_kmh=110, num_lanes=3):
    self.curvature = curvature  # 1/m (0 = straight, positive = left curve)
    self.speed_limit = speed_limit_kmh / 3.6  # m/s
    self.speed_limit_kmh = speed_limit_kmh
    self.num_lanes = num_lanes
    self.lane_width = LANE_WIDTH

  def lane_center_y(self, lane_idx):
    """Y coordinate of lane center (0-indexed from left)."""
    return (lane_idx + 0.5) * self.lane_width

  def road_left_y(self):
    return 0.0

  def road_right_y(self):
    return self.num_lanes * self.lane_width


# ──────────────────────────────────────────────────────────────────────────────
# Obstacle
# ──────────────────────────────────────────────────────────────────────────────
class Obstacle:

  def __init__(self, x, y, vx=0, vy=0, width=2.0, length=4.5, label='vehicle'):
    self.x = x
    self.y = y
    self.vx = vx
    self.vy = vy
    self.width = width
    self.length = length
    self.label = label

  def update(self, dt):
    self.x += self.vx * dt
    self.y += self.vy * dt


# ──────────────────────────────────────────────────────────────────────────────
# Vehicle Truth State
# ──────────────────────────────────────────────────────────────────────────────
def make_car_truth(speed_ms=30.0, lane=1, heading_offset=0.0, road=None):
  """Create initial truth state for the car."""
  if road is None:
    road = RoadModel()
  y = road.lane_center_y(lane)
  return {
    'x': 0.0,  # longitudinal position m
    'y': y,  # lateral position m
    'vx': speed_ms,  # forward speed m/s
    'vy': 0.0,  # lateral speed m/s
    'yaw': heading_offset,  # heading rad (0 = forward along road)
    'yaw_rate': 0.0,  # rad/s
    'steer_angle': 0.0,  # front wheel angle rad
    'throttle': 0.3,  # 0-1
    'brake_force': 0.0,  # N total
    'wheel_speeds': [speed_ms / CAR['tire_radius']] * 4,  # rad/s [FL, FR, RL, RR]
    'mass': CAR['mass'],
    'mu': 1.0,  # tire-road friction coefficient (1.0 = dry)
    'time': 0.0,
    # Failure state
    'steer_jam': None,
    'brake_fail': False,
    'throttle_stuck': None,
  }


# ──────────────────────────────────────────────────────────────────────────────
# Physics
# ──────────────────────────────────────────────────────────────────────────────
def car_physics_step(truth, controls, road, dt=0.02):
  """Kinematic car model with realistic longitudinal dynamics.

  Lateral: kinematic bicycle (yaw_rate = V * tan(steer) / L), with
  traction-dependent response lag and yaw damping for stability.
  Longitudinal: force-based (engine, brake, drag, rolling resistance).
  """
  t = dict(truth)
  t['wheel_speeds'] = list(truth['wheel_speeds'])
  car = CAR
  mu = t['mu']

  # Apply controls with rate limiting
  def rl(cur, tgt, rate):
    d = tgt - cur
    mx = rate * dt
    return cur + max(min(d, mx), -mx)

  # Steering
  tgt_steer = controls.get('steering', 0.0) * car['max_steer_angle']
  if t['steer_jam'] is not None:
    t['steer_angle'] = t['steer_jam']
  else:
    t['steer_angle'] = rl(t['steer_angle'], tgt_steer, car['steer_rate_limit'])

  # Throttle
  tgt_thr = max(0, min(1, controls.get('throttle', t['throttle'])))
  if t['throttle_stuck'] is not None:
    t['throttle'] = t['throttle_stuck']
  else:
    t['throttle'] = rl(t['throttle'], tgt_thr, car['throttle_rate_limit'])

  # Brake
  tgt_brake = max(0, min(1, controls.get('brake', 0.0))) * car['max_brake_force'] * 2
  if t['brake_fail']:
    t['brake_force'] = 0
  else:
    t['brake_force'] = rl(t['brake_force'], tgt_brake, car['brake_rate_limit'])

  V = max(t['vx'], 0.1)

  # ── Longitudinal dynamics (force-based) ──
  Fx_engine = t['throttle'] * car['max_engine_force'] * mu  # traction limited
  Fx_brake = -t['brake_force'] * min(mu, 1.0) * (1 if t['vx'] > 0.5 else 0)
  Fx_drag = -0.5 * car['rho_air'] * car['Cd'] * car['A_front'] * V * V
  Fx_roll = -t['mass'] * G * 0.015 * (1 if V > 0.5 else 0)
  ax = (Fx_engine + Fx_brake + Fx_drag + Fx_roll) / t['mass']
  t['vx'] += ax * dt
  t['vx'] = max(t['vx'], 0)

  # ── Lateral dynamics (kinematic + damping) ──
  # Desired yaw rate from kinematic bicycle model
  yaw_rate_desired = V * math.tan(t['steer_angle']) / car['wheelbase']
  # Add road curvature: car must yaw to follow curve
  yaw_rate_desired += V * road.curvature
  # Traction limits lateral response (low mu = sluggish + possible overshoot)
  response_rate = 3.0 * mu  # how fast yaw rate converges (rad/s²)
  yaw_err = yaw_rate_desired - t['yaw_rate']
  max_yaw_change = response_rate * dt
  t['yaw_rate'] += max(-max_yaw_change, min(max_yaw_change, yaw_err))

  # On very low mu, add random yaw disturbance (skidding)
  if mu < 0.3 and abs(t['yaw_rate']) > 0.01:
    t['yaw_rate'] += random.gauss(0, 0.02 * (0.3 - mu))

  # Position update
  t['yaw'] += t['yaw_rate'] * dt
  cos_y, sin_y = math.cos(t['yaw']), math.sin(t['yaw'])
  t['x'] += V * cos_y * dt
  t['y'] += V * sin_y * dt
  t['vy'] = V * math.sin(t['yaw'] - truth['yaw']) / max(dt, 0.001) * 0.1  # approx lateral V

  # Wheel speeds
  r = car['tire_radius']
  if t['brake_force'] > 5000 and mu < 0.5:
    for i in range(4):
      t['wheel_speeds'][i] = (V / r) * (0.3 + random.gauss(0, 0.2))
  else:
    base = V / r
    yr_diff = t['yaw_rate'] * car['track_width'] / (2 * r)
    t['wheel_speeds'][0] = base - yr_diff
    t['wheel_speeds'][1] = base + yr_diff
    t['wheel_speeds'][2] = base - yr_diff * 0.3
    t['wheel_speeds'][3] = base + yr_diff * 0.3

  t['time'] += dt
  return t


# ──────────────────────────────────────────────────────────────────────────────
# Sensor Model
# ──────────────────────────────────────────────────────────────────────────────
class CarSensorModel:
  """Generates sensor readings from car truth state."""

  def __init__(self, seed=42):
    self.rng = random.Random(seed)
    self.failures = {}
    self.stuck_vals = {}
    self.drift_accum = {}
    self.flicker_keys = set()  # keys that randomly drop out

  def set_failure(self, key, mode, param=0.0):
    self.failures[key] = (mode, param)

  def set_flicker(self, key):
    self.flicker_keys.add(key)

  def _apply_failure(self, key, true_val, noise_std=0.0):
    val = true_val + self.rng.gauss(0, noise_std)

    # Flicker: randomly invalid
    if key in self.flicker_keys:
      if self.rng.random() < 0.15:
        return val + self.rng.gauss(0, abs(true_val) * 0.5 + 10), False

    if key not in self.failures:
      return val, True
    mode, param = self.failures[key]
    if mode == 'stuck':
      if key not in self.stuck_vals:
        self.stuck_vals[key] = val
      return self.stuck_vals[key], True
    elif mode == 'bias':
      return val + param, True
    elif mode == 'drift':
      self.drift_accum[key] = self.drift_accum.get(key, 0) + param * 0.02
      return val + self.drift_accum[key], True
    elif mode == 'dead':
      return 0.0, False
    elif mode == 'noise':
      return val + self.rng.gauss(0, param), True
    elif mode == 'spike':
      if self.rng.random() < 0.05:
        return val + self.rng.gauss(0, param), True
      return val, True
    return val, True

  def build_state(self, truth, road, obstacles=None):
    """Build full sensor state dict from physics truth."""
    t = truth
    s = {}
    V = math.sqrt(t['vx']**2 + t['vy']**2)
    speed_kmh = V * 3.6

    # ── Wheel Speed Sensors (4) ──
    for i, name in enumerate(['fl', 'fr', 'rl', 'rr']):
      v, ok = self._apply_failure(
        f'wheel_speed_{name}',
        t['wheel_speeds'][i] * CAR['tire_radius'] * 3.6,  # km/h
        0.2)
      s[f'wheel_speed_{name}_kmh'] = max(0, v)
      s[f'wheel_speed_{name}_valid'] = ok

    # ── Gyroscopes (3) ──
    for i in range(1, 4):
      v, ok = self._apply_failure(f'gyro_{i}', t['yaw_rate'] * RAD2DEG, 0.05)
      s[f'gyro_{i}_yaw_rate_dps'] = v
      s[f'gyro_{i}_valid'] = ok

    # ── Accelerometers (3) ──
    ax_true = (t['vx'] - getattr(self, '_prev_vx', t['vx'])) / 0.02
    ay_true = (t['vy'] - getattr(self, '_prev_vy', t['vy'])) / 0.02
    self._prev_vx, self._prev_vy = t['vx'], t['vy']
    for i in range(1, 4):
      s[f'accel_{i}_longitudinal_ms2'] = ax_true + self.rng.gauss(0, 0.05)
      s[f'accel_{i}_lateral_ms2'] = ay_true + self.rng.gauss(0, 0.05)
      s[f'accel_{i}_valid'] = True

    # ── GPS (2) ──
    for i in range(1, 3):
      gx, ok = self._apply_failure(f'gps_{i}_x', t['x'], 1.5)
      s[f'gps_{i}_x_m'] = gx
      gy, _ = self._apply_failure(f'gps_{i}_y', t['y'], 0.8)
      s[f'gps_{i}_y_m'] = gy
      s[f'gps_{i}_speed_kmh'] = speed_kmh + self.rng.gauss(0, 1.0)
      s[f'gps_{i}_heading_deg'] = t['yaw'] * RAD2DEG + self.rng.gauss(0, 0.5)
      s[f'gps_{i}_valid'] = ok

    # ── Steering Angle Sensor (2) ──
    for i in range(1, 3):
      v, ok = self._apply_failure(f'steer_{i}', t['steer_angle'] * RAD2DEG, 0.1)
      s[f'steering_angle_{i}_deg'] = v
      s[f'steering_angle_{i}_valid'] = ok

    # ── Brake Pressure (2) ──
    for i in range(1, 3):
      bp = t['brake_force'] / (CAR['max_brake_force'] * 2) * 100  # percent
      s[f'brake_pressure_{i}_pct'] = bp + self.rng.gauss(0, 0.5)
      s[f'brake_pressure_{i}_valid'] = True

    # ── Vehicle speed (computed, 2 sources) ──
    s['vehicle_speed_1_kmh'] = speed_kmh + self.rng.gauss(0, 0.5)
    s['vehicle_speed_2_kmh'] = speed_kmh + self.rng.gauss(0, 0.5)

    # ── Laser Range Finders (12 beams at different angles) ──
    beam_angles = [-90, -60, -45, -30, -15, -5, 5, 15, 30, 45, 60, 90]  # degrees
    for angle in beam_angles:
      key = f'lidar_{angle:+04d}_deg'
      dist = self._compute_lidar(t, road, obstacles, angle * DEG2RAD)
      v, ok = self._apply_failure(f'lidar_{angle}', dist, 0.1)
      s[f'{key}_range_m'] = max(0, v)
      s[f'{key}_valid'] = ok

    # ── Image Parser Results (3 independent vision systems) ──
    for i in range(1, 4):
      # Left lane line
      dist_to_left = t['y'] - self._nearest_line_left(t['y'], road)
      v, ok = self._apply_failure(f'vision_{i}_left_line', dist_to_left, 0.05)
      s[f'vision_{i}_left_line_dist_m'] = v
      s[f'vision_{i}_left_line_type'] = self._line_type(t['y'], road, 'left')
      s[f'vision_{i}_left_line_valid'] = ok and self.rng.random() > 0.02

      # Right lane line
      dist_to_right = self._nearest_line_right(t['y'], road) - t['y']
      v, ok = self._apply_failure(f'vision_{i}_right_line', dist_to_right, 0.05)
      s[f'vision_{i}_right_line_dist_m'] = v
      s[f'vision_{i}_right_line_type'] = self._line_type(t['y'], road, 'right')
      s[f'vision_{i}_right_line_valid'] = ok and self.rng.random() > 0.02

      # Obstacle detection
      front_obs = self._nearest_obstacle_front(t, obstacles)
      if front_obs:
        obs_dist, obs_speed_rel = front_obs
        s[f'vision_{i}_obstacle_detected'] = True
        s[f'vision_{i}_obstacle_dist_m'] = obs_dist + self.rng.gauss(0, 0.5)
        s[f'vision_{i}_obstacle_rel_speed_kmh'] = obs_speed_rel * 3.6 + self.rng.gauss(0, 1)
      else:
        s[f'vision_{i}_obstacle_detected'] = False
        s[f'vision_{i}_obstacle_dist_m'] = 200.0
        s[f'vision_{i}_obstacle_rel_speed_kmh'] = 0.0
      s[f'vision_{i}_obstacle_valid'] = True

      # Speed limit sign
      v_sl, ok = self._apply_failure(f'vision_{i}_speed', road.speed_limit_kmh, 0)
      s[f'vision_{i}_speed_limit_kmh'] = v_sl
      s[f'vision_{i}_speed_limit_valid'] = ok

      # Lane departure warning
      lane_center_dist = self._distance_to_nearest_lane_center(t['y'], road)
      s[f'vision_{i}_lane_offset_m'] = lane_center_dist + self.rng.gauss(0, 0.03)
      s[f'vision_{i}_road_curvature_1pm'] = road.curvature + self.rng.gauss(0, 0.0001)

    # ── Traction / Surface ──
    s['estimated_mu'] = t['mu'] + self.rng.gauss(0, 0.05)  # estimated friction
    s['rain_sensor'] = t['mu'] < 0.7
    s['temperature_c'] = 20 if t['mu'] > 0.5 else -2  # heuristic

    # ── Targets ──
    s['target_speed_kmh'] = road.speed_limit_kmh
    s['target_lane'] = self._current_lane(t['y'], road)
    s['current_lane'] = self._current_lane(t['y'], road)

    # ── Metadata ──
    s['sim_time_s'] = t['time']
    s['dt'] = 0.02

    return s

  def _compute_lidar(self, truth, road, obstacles, angle):
    """Compute distance to nearest object at given angle from vehicle."""
    max_range = 150.0
    cos_a, sin_a = math.cos(truth['yaw'] + angle), math.sin(truth['yaw'] + angle)
    min_dist = max_range

    # Check road boundaries
    if abs(sin_a) > 0.001:
      d_left = (0 - truth['y']) / sin_a
      d_right = (road.road_right_y() - truth['y']) / sin_a
      for d in [d_left, d_right]:
        if 0 < d < min_dist:
          min_dist = d

    # Check obstacles
    if obstacles:
      for obs in obstacles:
        dx = obs.x - truth['x']
        dy = obs.y - truth['y']
        dist = math.sqrt(dx * dx + dy * dy)
        # Simple angle check
        obs_angle = math.atan2(dy, dx) - truth['yaw']
        if abs(obs_angle - angle) < 0.15 and dist < min_dist:
          min_dist = dist - obs.length / 2

    return max(0, min_dist)

  def _nearest_line_left(self, y, road):
    """Y coordinate of nearest lane line to the left of car."""
    best = 0.0
    for i in range(road.num_lanes + 1):
      line_y = i * road.lane_width
      if line_y <= y:
        best = line_y
    return best

  def _nearest_line_right(self, y, road):
    for i in range(road.num_lanes + 1):
      line_y = i * road.lane_width
      if line_y >= y:
        return line_y
    return road.road_right_y()

  def _line_type(self, y, road, side):
    lane = self._current_lane(y, road)
    if side == 'left':
      return 'solid' if lane == 0 else 'dashed'
    else:
      return 'solid' if lane == road.num_lanes - 1 else 'dashed'

  def _current_lane(self, y, road):
    lane = int(y / road.lane_width)
    return max(0, min(road.num_lanes - 1, lane))

  def _distance_to_nearest_lane_center(self, y, road):
    lane = self._current_lane(y, road)
    center = road.lane_center_y(lane)
    return y - center

  def _nearest_obstacle_front(self, truth, obstacles):
    if not obstacles:
      return None
    best = None
    for obs in obstacles:
      dx = obs.x - truth['x']
      dy = abs(obs.y - truth['y'])
      if dx > 0 and dy < 3.0:  # within ~1 lane
        rel_speed = truth['vx'] - obs.vx
        if best is None or dx < best[0]:
          best = (dx, rel_speed)
    return best


# ──────────────────────────────────────────────────────────────────────────────
# Simulation Runner
# ──────────────────────────────────────────────────────────────────────────────
class CarSimRunner:

  def __init__(self, truth, sensor_model, road, obstacles=None):
    self.truth = truth
    self.sensors = sensor_model
    self.road = road
    self.obstacles = obstacles or []
    self.crashed = False
    self.crash_reason = ''
    self.history = []

  def run(self, autopilot_fn, steps=2000, dt=0.02):
    for step in range(steps):
      # Update obstacles
      for obs in self.obstacles:
        obs.update(dt)

      state = self.sensors.build_state(self.truth, self.road, self.obstacles)
      try:
        controls = autopilot_fn(state, dt)
        if not isinstance(controls, dict):
          controls = {}
      except Exception:
        controls = {}

      for k in ['steering', 'brake', 'throttle']:
        if k in controls:
          controls[k] = max(-1.0, min(1.0, float(controls[k])))

      self.truth = car_physics_step(self.truth, controls, self.road, dt)

      # Check crashes
      y = self.truth['y']
      if y < -3.0 or y > self.road.road_right_y() + 3.0:
        self.crashed = True
        self.crash_reason = f'Left roadway (y={y:.1f}m)'
        break

      # Check collision with obstacles
      for obs in self.obstacles:
        dx = abs(self.truth['x'] - obs.x)
        dy = abs(self.truth['y'] - obs.y)
        if dx < (obs.length / 2 + 2.5) and dy < (obs.width / 2 + 1.0):
          self.crashed = True
          self.crash_reason = f'Collision with {obs.label}'
          break
      if self.crashed:
        break

      # Record history
      if step % 25 == 0:
        V = math.sqrt(self.truth['vx']**2 + self.truth['vy']**2)
        self.history.append({
          'time': self.truth['time'],
          'x': self.truth['x'],
          'y': self.truth['y'],
          'speed_kmh': V * 3.6,
          'yaw_deg': self.truth['yaw'] * RAD2DEG,
          'lane': self.sensors._current_lane(self.truth['y'], self.road),
        })

    return self.sensors.build_state(self.truth, self.road, self.obstacles), self.history

  def score(self):
    if self.crashed:
      return 0.0, f'Crash: {self.crash_reason}'
    if not self.history:
      return 0.0, 'No history'

    recent = self.history[-10:] if len(self.history) >= 10 else self.history
    score = 1.0
    details = []

    # Lane keeping
    avg_offset = sum(
      abs(self.sensors._distance_to_nearest_lane_center(h['y'], self.road))
      for h in recent) / len(recent)
    if avg_offset > 1.5:
      score -= 0.3
      details.append(f'lane offset {avg_offset:.1f}m')
    elif avg_offset > 0.5:
      score -= 0.1

    # Speed management
    tgt = self.road.speed_limit * 3.6
    avg_spd = sum(h['speed_kmh'] for h in recent) / len(recent)
    spd_err = abs(avg_spd - tgt)
    if avg_spd > tgt * 1.15:
      score -= 0.2
      details.append(f'speeding {avg_spd:.0f}/{tgt:.0f} km/h')
    elif avg_spd < tgt * 0.5 and tgt > 30:
      score -= 0.15
      details.append(f'too slow {avg_spd:.0f}/{tgt:.0f} km/h')

    # Stability
    yaw_var = sum((h['yaw_deg'] - recent[0]['yaw_deg'])**2 for h in recent) / len(recent)
    if yaw_var > 100:
      score -= 0.2
      details.append(f'unstable yaw')

    # Still on road
    last_y = recent[-1]['y']
    if last_y < 0.5 or last_y > self.road.road_right_y() - 0.5:
      score -= 0.2
      details.append('near edge')

    score = max(0.0, score)
    summary = (f'Speed: {avg_spd:.0f} km/h (tgt {tgt:.0f}), '
               f'Lane offset: {avg_offset:.2f}m, Yaw var: {yaw_var:.1f}')
    if details:
      summary += ' | ' + ', '.join(details)
    return score, summary


def list_car_state_keys():
  t = make_car_truth()
  sm = CarSensorModel()
  road = RoadModel()
  s = sm.build_state(t, road, [])
  return sorted(s.keys())
