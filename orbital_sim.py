"""
Spacecraft Orbital Docking Simulator

Models two spacecraft in orbit: a chaser vehicle docking with a target station.
Uses true Keplerian orbital mechanics with gravitational acceleration,
NOT simplified linear relative motion (Hill/CW equations).

The key insight: in orbit, "thrust toward the target" often moves you AWAY
due to orbital mechanics. Prograde thrust raises orbit (slows you down),
retrograde lowers it (speeds you up). The LLM must understand this.

Coordinate system: Earth-Centered Inertial (ECI)
  x: toward vernal equinox, y: in equatorial plane, z: toward north pole
All units SI: meters, m/s, radians, kg, seconds.
"""

import math
import random
from typing import Dict, List, Tuple, Optional

# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────
MU_EARTH = 3.986004418e14  # m³/s² gravitational parameter
R_EARTH = 6.371e6          # m mean radius
DEG2RAD = math.pi / 180.0
RAD2DEG = 180.0 / math.pi

# ──────────────────────────────────────────────────────────────────────────────
# Spacecraft Parameters (Soyuz/Dragon-class)
# ──────────────────────────────────────────────────────────────────────────────
SC = {
  'mass': 7000,                # kg dry mass
  'fuel_mass': 500,            # kg propellant
  'thrust_per_jet': 220,       # N per RCS thruster
  'num_thrusters': 24,         # 4 per axis (± in each of 6 DOF)
  'isp': 290,                  # s specific impulse
  'max_attitude_rate': 2.0 * DEG2RAD,  # rad/s max rotation rate
  'attitude_deadband': 0.5 * DEG2RAD,
  'docking_port_axis': [1, 0, 0],  # +x axis
  # Docking tolerances
  'dock_range_m': 2.0,        # must be within 2m
  'dock_speed_ms': 0.3,       # max approach speed
  'dock_lateral_m': 0.5,      # max lateral offset
  'dock_angle_deg': 5.0,      # max angular misalignment
}

# Target station parameters
STATION = {
  'orbit_alt_km': 408,         # ISS-like orbit
  'orbit_inc_deg': 51.6,       # ISS inclination
  'docking_axis': [-1, 0, 0],  # -x axis (faces chaser)
  'mass': 420000,              # kg
}

# ──────────────────────────────────────────────────────────────────────────────
# Orbital Mechanics Helpers
# ──────────────────────────────────────────────────────────────────────────────
def _vec_add(a, b):
  return [a[0]+b[0], a[1]+b[1], a[2]+b[2]]

def _vec_sub(a, b):
  return [a[0]-b[0], a[1]-b[1], a[2]-b[2]]

def _vec_scale(a, s):
  return [a[0]*s, a[1]*s, a[2]*s]

def _vec_dot(a, b):
  return a[0]*b[0] + a[1]*b[1] + a[2]*b[2]

def _vec_cross(a, b):
  return [a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0]]

def _vec_norm(a):
  return math.sqrt(a[0]**2 + a[1]**2 + a[2]**2)

def _vec_unit(a):
  n = _vec_norm(a)
  if n < 1e-12:
    return [0, 0, 0]
  return [a[0]/n, a[1]/n, a[2]/n]

def circular_orbit_state(alt_km, inc_deg, raan_deg=0, arg_lat_deg=0):
  """Position and velocity for a circular orbit at given altitude/inclination."""
  r = (R_EARTH + alt_km * 1000)
  v = math.sqrt(MU_EARTH / r)
  inc = inc_deg * DEG2RAD
  raan = raan_deg * DEG2RAD
  u = arg_lat_deg * DEG2RAD  # argument of latitude

  # Position in orbital plane
  x_orb = r * math.cos(u)
  y_orb = r * math.sin(u)

  # Velocity in orbital plane (circular)
  vx_orb = -v * math.sin(u)
  vy_orb = v * math.cos(u)

  # Rotate to ECI through RAAN and inclination
  cos_O, sin_O = math.cos(raan), math.sin(raan)
  cos_i, sin_i = math.cos(inc), math.sin(inc)

  pos = [
    cos_O * x_orb - sin_O * cos_i * y_orb,
    sin_O * x_orb + cos_O * cos_i * y_orb,
    sin_i * y_orb,
  ]
  vel = [
    cos_O * vx_orb - sin_O * cos_i * vy_orb,
    sin_O * vx_orb + cos_O * cos_i * vy_orb,
    sin_i * vy_orb,
  ]
  return pos, vel


def grav_accel(pos):
  """Gravitational acceleration at position (ECI)."""
  r = _vec_norm(pos)
  if r < 1e3:
    return [0, 0, 0]
  a = -MU_EARTH / (r * r * r)
  return [pos[0] * a, pos[1] * a, pos[2] * a]


def rk4_step(pos, vel, accel_fn, dt):
  """4th order Runge-Kutta integration step for orbital mechanics."""
  def derivs(p, v):
    a = accel_fn(p)
    return v, a

  v1, a1 = derivs(pos, vel)
  p2 = _vec_add(pos, _vec_scale(v1, dt/2))
  v2_tmp = _vec_add(vel, _vec_scale(a1, dt/2))
  v2, a2 = derivs(p2, v2_tmp)

  p3 = _vec_add(pos, _vec_scale(v2, dt/2))
  v3_tmp = _vec_add(vel, _vec_scale(a2, dt/2))
  v3, a3 = derivs(p3, v3_tmp)

  p4 = _vec_add(pos, _vec_scale(v3, dt))
  v4_tmp = _vec_add(vel, _vec_scale(a3, dt))
  v4, a4 = derivs(p4, v4_tmp)

  new_pos = [pos[i] + dt/6 * (v1[i] + 2*v2[i] + 2*v3[i] + v4[i]) for i in range(3)]
  new_vel = [vel[i] + dt/6 * (a1[i] + 2*a2[i] + 2*a3[i] + a4[i]) for i in range(3)]
  return new_pos, new_vel


# ──────────────────────────────────────────────────────────────────────────────
# LVLH Frame (Local Vertical Local Horizontal) for relative navigation
# ──────────────────────────────────────────────────────────────────────────────
def eci_to_lvlh(target_pos, target_vel, chaser_pos, chaser_vel):
  """Convert chaser position/velocity to LVLH frame centered on target.
  
  LVLH: x = radial (away from Earth), y = along-track (velocity direction),
         z = cross-track (completes right-hand system).
  Returns (relative_pos_lvlh, relative_vel_lvlh).
  """
  # LVLH basis vectors
  r_hat = _vec_unit(target_pos)               # radial
  h = _vec_cross(target_pos, target_vel)       # angular momentum
  h_hat = _vec_unit(h)                         # cross-track
  v_hat = _vec_cross(h_hat, r_hat)             # along-track

  # Relative position/velocity in ECI
  dr = _vec_sub(chaser_pos, target_pos)
  dv = _vec_sub(chaser_vel, target_vel)

  # Project onto LVLH
  rel_pos = [_vec_dot(dr, r_hat), _vec_dot(dr, v_hat), _vec_dot(dr, h_hat)]
  # Account for rotating frame
  omega = _vec_norm(h) / (_vec_norm(target_pos) ** 2)
  rel_vel = [
    _vec_dot(dv, r_hat) - omega * _vec_dot(dr, v_hat),
    _vec_dot(dv, v_hat) + omega * _vec_dot(dr, r_hat),
    _vec_dot(dv, h_hat),
  ]
  return rel_pos, rel_vel


# ──────────────────────────────────────────────────────────────────────────────
# Docking Truth State
# ──────────────────────────────────────────────────────────────────────────────
def make_dock_truth(chaser_offset_lvlh=None, chaser_dv_lvlh=None,
                    station_alt_km=408, station_inc_deg=51.6,
                    station_raan_deg=0, station_arg_lat_deg=0):
  """Create initial truth state for docking scenario."""
  if chaser_offset_lvlh is None:
    chaser_offset_lvlh = [0, -200, 0]  # 200m behind in V-bar
  if chaser_dv_lvlh is None:
    chaser_dv_lvlh = [0, 0, 0]

  # Station state
  s_pos, s_vel = circular_orbit_state(station_alt_km, station_inc_deg,
                                       station_raan_deg, station_arg_lat_deg)

  # Convert LVLH offset to ECI
  r_hat = _vec_unit(s_pos)
  h = _vec_cross(s_pos, s_vel)
  h_hat = _vec_unit(h)
  v_hat = _vec_cross(h_hat, r_hat)

  c_pos = _vec_add(s_pos, _vec_add(
    _vec_scale(r_hat, chaser_offset_lvlh[0]),
    _vec_add(_vec_scale(v_hat, chaser_offset_lvlh[1]),
             _vec_scale(h_hat, chaser_offset_lvlh[2]))))

  omega = _vec_norm(h) / (_vec_norm(s_pos) ** 2)
  c_vel = _vec_add(s_vel, _vec_add(
    _vec_scale(r_hat, chaser_dv_lvlh[0] + omega * chaser_offset_lvlh[1]),
    _vec_add(_vec_scale(v_hat, chaser_dv_lvlh[1] - omega * chaser_offset_lvlh[0]),
             _vec_scale(h_hat, chaser_dv_lvlh[2]))))

  return {
    'station_pos': list(s_pos),
    'station_vel': list(s_vel),
    'chaser_pos': list(c_pos),
    'chaser_vel': list(c_vel),
    'chaser_att': [0.0, 0.0, 0.0],    # roll, pitch, yaw (relative to LVLH)
    'chaser_att_rate': [0.0, 0.0, 0.0],
    'fuel_mass': SC['fuel_mass'],
    'thruster_failed': [False] * SC['num_thrusters'],
    'uncommanded_thrust': [0.0, 0.0, 0.0],  # parasitic thrust vector (N)
    'time': 0.0,
  }


# ──────────────────────────────────────────────────────────────────────────────
# Physics Step
# ──────────────────────────────────────────────────────────────────────────────
def dock_physics_step(truth, controls, dt=1.0):
  """Advance both spacecraft one timestep. Controls: thrust in body frame."""
  t = dict(truth)
  t['station_pos'] = list(truth['station_pos'])
  t['station_vel'] = list(truth['station_vel'])
  t['chaser_pos'] = list(truth['chaser_pos'])
  t['chaser_vel'] = list(truth['chaser_vel'])
  t['chaser_att'] = list(truth['chaser_att'])
  t['chaser_att_rate'] = list(truth['chaser_att_rate'])
  t['thruster_failed'] = list(truth['thruster_failed'])

  sc = SC

  # ── Attitude control ──
  for i in range(3):
    tgt_rate = controls.get(f'att_rate_{["roll","pitch","yaw"][i]}', 0.0)
    tgt_rate = max(-sc['max_attitude_rate'], min(sc['max_attitude_rate'], tgt_rate))
    # Simple first-order convergence
    t['chaser_att_rate'][i] += (tgt_rate - t['chaser_att_rate'][i]) * min(1, dt * 2)
    t['chaser_att'][i] += t['chaser_att_rate'][i] * dt

  # ── Thrust (in LVLH frame for simplicity) ──
  thrust_lvlh = [0, 0, 0]
  for i, axis in enumerate(['x', 'y', 'z']):
    cmd = controls.get(f'thrust_{axis}', 0.0)
    cmd = max(-1.0, min(1.0, cmd))
    thrust_lvlh[i] = cmd * sc['thrust_per_jet']

  # Add uncommanded thrust
  for i in range(3):
    thrust_lvlh[i] += t['uncommanded_thrust'][i]

  # Fuel consumption
  thrust_mag = _vec_norm(thrust_lvlh)
  if thrust_mag > 0 and t['fuel_mass'] > 0:
    # Tsiolkovsky: dm = F * dt / (Isp * g0)
    dm = thrust_mag * dt / (sc['isp'] * 9.80665)
    t['fuel_mass'] = max(0, t['fuel_mass'] - dm)
  elif t['fuel_mass'] <= 0:
    thrust_lvlh = [0, 0, 0]
    t['fuel_mass'] = 0

  # Convert LVLH thrust to ECI
  r_hat = _vec_unit(t['station_pos'])
  h = _vec_cross(t['station_pos'], t['station_vel'])
  h_hat = _vec_unit(h)
  v_hat = _vec_cross(h_hat, r_hat)

  total_mass = sc['mass'] + t['fuel_mass']
  accel_eci = [0, 0, 0]
  for i in range(3):
    f = thrust_lvlh[i] / total_mass
    accel_eci[0] += f * [r_hat, v_hat, h_hat][i][0]
    accel_eci[1] += f * [r_hat, v_hat, h_hat][i][1]
    accel_eci[2] += f * [r_hat, v_hat, h_hat][i][2]

  # ── Propagate orbits ──
  # Station (no thrust, just gravity)
  t['station_pos'], t['station_vel'] = rk4_step(
    t['station_pos'], t['station_vel'], grav_accel, dt)

  # Chaser (gravity + thrust)
  def chaser_accel(pos):
    g = grav_accel(pos)
    return _vec_add(g, accel_eci)

  t['chaser_pos'], t['chaser_vel'] = rk4_step(
    t['chaser_pos'], t['chaser_vel'], chaser_accel, dt)

  t['time'] += dt
  return t


# ──────────────────────────────────────────────────────────────────────────────
# Sensor Model
# ──────────────────────────────────────────────────────────────────────────────
class DockSensorModel:
  def __init__(self, seed=42):
    self.rng = random.Random(seed)
    self.failures = {}
    self.flicker_keys = set()
    self.stuck_vals = {}
    self.drift_accum = {}

  def set_failure(self, key, mode, param=0.0):
    self.failures[key] = (mode, param)

  def set_flicker(self, key):
    self.flicker_keys.add(key)

  def _apply(self, key, true_val, noise_std=0.0):
    val = true_val + self.rng.gauss(0, noise_std)
    if key in self.flicker_keys and self.rng.random() < 0.1:
      return val + self.rng.gauss(0, abs(true_val) * 0.3 + 5), False
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
      self.drift_accum[key] = self.drift_accum.get(key, 0) + param * 1.0
      return val + self.drift_accum[key], True
    elif mode == 'dead':
      return 0.0, False
    elif mode == 'noise':
      return val + self.rng.gauss(0, param), True
    elif mode == 'spike':
      if self.rng.random() < 0.03:
        return val + self.rng.gauss(0, param), True
      return val, True
    return val, True

  def build_state(self, truth):
    """Build sensor state dict from truth."""
    t = truth
    s = {}

    # Compute LVLH relative state
    rel_pos, rel_vel = eci_to_lvlh(
      t['station_pos'], t['station_vel'],
      t['chaser_pos'], t['chaser_vel'])
    rng = _vec_norm(rel_pos)

    # ── Range finders (3 redundant) ──
    for i in range(1, 4):
      v, ok = self._apply(f'range_{i}', rng, 0.05)
      s[f'range_{i}_m'] = max(0, v)
      s[f'range_{i}_valid'] = ok

    # ── Relative position LVLH (2 sources: radar + lidar) ──
    for src in ['radar', 'lidar']:
      for j, axis in enumerate(['x', 'y', 'z']):
        v, ok = self._apply(f'{src}_rel_{axis}', rel_pos[j], 0.1)
        s[f'{src}_rel_pos_{axis}_m'] = v
        s[f'{src}_rel_{axis}_valid'] = ok

    # ── Relative velocity LVLH (2 sources) ──
    for src in ['radar', 'lidar']:
      for j, axis in enumerate(['x', 'y', 'z']):
        v, ok = self._apply(f'{src}_relv_{axis}', rel_vel[j], 0.005)
        s[f'{src}_rel_vel_{axis}_ms'] = v
        s[f'{src}_relv_{axis}_valid'] = ok

    # ── Gyroscopes (3 redundant) ──
    for i in range(1, 4):
      for j, axis in enumerate(['roll', 'pitch', 'yaw']):
        v, ok = self._apply(f'gyro_{i}_{axis}', t['chaser_att_rate'][j] * RAD2DEG, 0.01)
        s[f'gyro_{i}_{axis}_dps'] = v
        s[f'gyro_{i}_{axis}_valid'] = ok

    # ── Accelerometers (3 redundant) ──
    # Approximate from thrust (in reality would include gravity gradient)
    for i in range(1, 4):
      for j, axis in enumerate(['x', 'y', 'z']):
        s[f'accel_{i}_{axis}_ms2'] = self.rng.gauss(0, 0.0001)
        s[f'accel_{i}_{axis}_valid'] = True

    # ── Star trackers (2) ── attitude determination
    for i in range(1, 3):
      for j, axis in enumerate(['roll', 'pitch', 'yaw']):
        v, ok = self._apply(f'star_{i}_{axis}', t['chaser_att'][j] * RAD2DEG, 0.05)
        s[f'star_tracker_{i}_{axis}_deg'] = v
        s[f'star_tracker_{i}_{axis}_valid'] = ok

    # ── GPS (2 receivers) ── absolute position
    for i in range(1, 3):
      for j, axis in enumerate(['x', 'y', 'z']):
        v, ok = self._apply(f'gps_{i}_{axis}', t['chaser_pos'][j], 5.0)
        s[f'gps_{i}_pos_{axis}_m'] = v
        s[f'gps_{i}_{axis}_valid'] = ok

    # ── Ground station orbital elements (updates every 30s) ──
    step = int(t['time'])
    if step % 30 == 0 or not hasattr(self, '_last_ground'):
      r_mag = _vec_norm(t['chaser_pos'])
      v_mag = _vec_norm(t['chaser_vel'])
      self._last_ground = {
        'semi_major_axis_m': 1 / (2/r_mag - v_mag**2/MU_EARTH),
        'altitude_km': (r_mag - R_EARTH) / 1000,
        'speed_ms': v_mag,
      }
    for k, v in self._last_ground.items():
      val, ok = self._apply(f'ground_{k}', v, abs(v) * 0.001)
      s[f'ground_{k}'] = val
      s[f'ground_{k}_valid'] = ok

    # ── Docking camera (bearing to docking port) ──
    if rng < 500 and rng > 0.1:
      bearing = [rel_pos[j] / rng for j in range(3)]
      for j, axis in enumerate(['x', 'y', 'z']):
        v, ok = self._apply(f'dock_cam_{axis}', bearing[j], 0.002)
        s[f'dock_camera_bearing_{axis}'] = v
        s[f'dock_camera_bearing_{axis}_valid'] = ok
      s['dock_camera_range_m'] = rng + self.rng.gauss(0, 0.02)
      s['dock_camera_active'] = True
    else:
      for axis in ['x', 'y', 'z']:
        s[f'dock_camera_bearing_{axis}'] = 0
        s[f'dock_camera_bearing_{axis}_valid'] = False
      s['dock_camera_range_m'] = 0
      s['dock_camera_active'] = False

    # ── Fuel ──
    s['fuel_remaining_kg'] = t['fuel_mass'] + self.rng.gauss(0, 0.5)
    s['fuel_remaining_pct'] = t['fuel_mass'] / SC['fuel_mass'] * 100

    # ── Targets ──
    s['target_dock_range_m'] = SC['dock_range_m']
    s['target_dock_speed_ms'] = SC['dock_speed_ms']

    # ── Time ──
    s['sim_time_s'] = t['time']
    s['dt'] = 1.0

    return s


# ──────────────────────────────────────────────────────────────────────────────
# Simulation Runner
# ──────────────────────────────────────────────────────────────────────────────
class DockSimRunner:
  def __init__(self, truth, sensor_model):
    self.truth = truth
    self.sensors = sensor_model
    self.docked = False
    self.crashed = False
    self.crash_reason = ''
    self.history = []
    self.min_range = float('inf')

  def run(self, autopilot_fn, steps=3600, dt=1.0):
    for step in range(steps):
      state = self.sensors.build_state(self.truth)
      try:
        controls = autopilot_fn(state, dt)
        if not isinstance(controls, dict):
          controls = {}
      except Exception:
        controls = {}

      # Clamp controls
      for k in ['thrust_x', 'thrust_y', 'thrust_z']:
        if k in controls:
          controls[k] = max(-1.0, min(1.0, float(controls[k])))
      for k in ['att_rate_roll', 'att_rate_pitch', 'att_rate_yaw']:
        if k in controls:
          controls[k] = max(-0.035, min(0.035, float(controls[k])))

      self.truth = dock_physics_step(self.truth, controls, dt)

      # Compute relative state
      rel_pos, rel_vel = eci_to_lvlh(
        self.truth['station_pos'], self.truth['station_vel'],
        self.truth['chaser_pos'], self.truth['chaser_vel'])
      rng = _vec_norm(rel_pos)
      approach_speed = -_vec_dot(rel_vel, _vec_unit(rel_pos)) if rng > 0.1 else 0

      self.min_range = min(self.min_range, rng)

      # Check docking
      if rng < SC['dock_range_m']:
        rel_speed = _vec_norm(rel_vel)
        if rel_speed < SC['dock_speed_ms']:
          self.docked = True
          break
        elif rel_speed > 2.0:
          self.crashed = True
          self.crash_reason = f'Collision: approach speed {rel_speed:.2f} m/s > 2.0 m/s'
          break

      # Check if drifted way too far
      if rng > 50000:
        self.crashed = True
        self.crash_reason = f'Lost: range {rng/1000:.1f} km, too far'
        break

      # Check fuel
      if self.truth['fuel_mass'] <= 0 and rng > SC['dock_range_m']:
        self.crashed = True
        self.crash_reason = f'Out of fuel at range {rng:.0f} m'
        break

      # Record history
      if step % 30 == 0:
        self.history.append({
          'time': self.truth['time'],
          'range_m': rng,
          'rel_pos': list(rel_pos),
          'rel_vel': list(rel_vel),
          'fuel_kg': self.truth['fuel_mass'],
        })

  def score(self):
    if self.crashed:
      return 0.0, f'Crash: {self.crash_reason}'

    if self.docked:
      # Bonus for fuel efficiency
      fuel_used_pct = (SC['fuel_mass'] - self.truth['fuel_mass']) / SC['fuel_mass'] * 100
      score = 1.0
      if fuel_used_pct > 80:
        score -= 0.1
      return score, f'Docked! Fuel used: {fuel_used_pct:.0f}%, Time: {self.truth["time"]:.0f}s'

    # Didn't dock but didn't crash — partial credit for getting closer
    if self.min_range < 10:
      score = 0.7
    elif self.min_range < 50:
      score = 0.5
    elif self.min_range < 200:
      score = 0.3
    else:
      score = 0.1

    return score, f'Did not dock. Min range: {self.min_range:.1f} m, Final fuel: {self.truth["fuel_mass"]:.0f} kg'


def list_dock_state_keys():
  t = make_dock_truth()
  sm = DockSensorModel()
  s = sm.build_state(t)
  return sorted(s.keys())
