"""
Autopilot Flight Simulator

Simplified 6-DOF flight dynamics for a twin-engine airliner (A320-class).
Provides:
- Full aircraft state dict with ~200 redundant sensor properties
- Physics simulation (lift, drag, thrust, gravity, moments)
- Sensor model with noise and configurable failures
- Scenario-based testing with failure injection
"""

import math
import random
from typing import Dict, Optional, List, Tuple, Any

# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────
FT2M = 0.3048
M2FT = 1.0 / FT2M
KT2MS = 0.514444
MS2KT = 1.0 / KT2MS
DEG2RAD = math.pi / 180.0
RAD2DEG = 180.0 / math.pi
FPM2MS = 0.00508
MS2FPM = 1.0 / FPM2MS
G = 9.80665
RHO_SL = 1.225  # kg/m³ sea level ISA
T_SL = 288.15  # K sea level ISA
LAPSE = 0.0065  # K/m troposphere lapse rate
P_SL = 101325.0  # Pa


# ──────────────────────────────────────────────────────────────────────────────
# Atmosphere
# ──────────────────────────────────────────────────────────────────────────────
def isa_rho(alt_m):
  """ISA air density at altitude (meters)."""
  alt_m = max(0, min(alt_m, 20000))
  if alt_m <= 11000:
    T = T_SL - LAPSE * alt_m
    rho = RHO_SL * (T / T_SL)**(G / (LAPSE * 287.05) - 1.0)
  else:
    T = 216.65
    rho_11 = RHO_SL * (216.65 / T_SL)**(G / (LAPSE * 287.05) - 1.0)
    rho = rho_11 * math.exp(-G * (alt_m - 11000) / (287.05 * T))
  return max(rho, 0.01)


def isa_temp(alt_m):
  if alt_m <= 11000:
    return T_SL - LAPSE * alt_m
  return 216.65


def isa_pressure(alt_m):
  alt_m = max(0, min(alt_m, 20000))
  if alt_m <= 11000:
    T = T_SL - LAPSE * alt_m
    return P_SL * (T / T_SL)**(G / (LAPSE * 287.05))
  else:
    P_11 = P_SL * (216.65 / T_SL)**(G / (LAPSE * 287.05))
    return P_11 * math.exp(-G * (alt_m - 11000) / (287.05 * 216.65))


def speed_of_sound(alt_m):
  return math.sqrt(1.4 * 287.05 * isa_temp(alt_m))


def ias_from_tas(tas_ms, alt_m):
  """Convert TAS (m/s) to IAS (m/s) using density ratio."""
  rho = isa_rho(alt_m)
  return tas_ms * math.sqrt(rho / RHO_SL)


def tas_from_ias(ias_ms, alt_m):
  rho = isa_rho(alt_m)
  return ias_ms * math.sqrt(RHO_SL / rho)


# ──────────────────────────────────────────────────────────────────────────────
# Aircraft Parameters (A320-class twin-engine airliner)
# ──────────────────────────────────────────────────────────────────────────────
AC = {
  'S': 122.6,  # wing area m²
  'b': 34.1,  # wingspan m
  'c': 4.19,  # mean aerodynamic chord m
  'AR': 9.5,  # aspect ratio
  'e': 0.85,  # Oswald efficiency
  'mass_empty': 42000,
  'mass_typical': 65000,
  'mass_mtow': 78000,
  'num_engines': 2,
  'max_thrust_per_engine': 120000,  # N
  'engine_y_offset': 5.8,  # m from centerline
  # Aero coefficients (clean config)
  'CL0': 0.28,
  'CLa': 5.5,  # per rad
  'CLde': 0.6,  # per rad elevator deflection
  'CLflap': 0.02,  # per degree of flap
  'alpha_stall_clean': 14 * DEG2RAD,
  'alpha_stall_flap': 18 * DEG2RAD,
  'CD0': 0.022,
  'CDflap': 0.0003,  # per degree of flap
  'CDgear': 0.015,
  'CDspeedbrake': 0.04,
  # Pitch moment
  'CM0': 0.02,
  'CMa': -1.2,  # per rad (stable)
  'CMde': -1.8,  # per rad
  'CMq': -20.0,  # pitch damping
  'CMtrim': -0.15,  # per unit trim
  # Roll moment
  'Clb': -0.08,  # dihedral effect per rad
  'Clda': 0.15,  # per rad aileron
  'Clp': -0.5,  # roll damping
  # Yaw moment
  'Cnb': 0.12,  # weathercock stability per rad
  'Cndr': -0.08,  # per rad rudder
  'Cnr': -0.15,  # yaw damping
  # Side force
  'CYb': -0.6,  # per rad sideslip
  'CYdr': 0.15,  # per rad rudder
  # Control limits (radians)
  'elevator_max': 17 * DEG2RAD,
  'aileron_max': 25 * DEG2RAD,
  'rudder_max': 30 * DEG2RAD,
  'trim_max': 4 * DEG2RAD,
  # Actuator rate limits (rad/s)
  'elevator_rate': 20 * DEG2RAD,
  'aileron_rate': 30 * DEG2RAD,
  'rudder_rate': 20 * DEG2RAD,
  'throttle_rate': 0.5,  # per second
  # Structural limits
  'g_limit_pos': 2.5,
  'g_limit_neg': -1.0,
  'Vne_kt': 365,  # never exceed IAS
  'Mmo': 0.82,  # max operating Mach
}


# ──────────────────────────────────────────────────────────────────────────────
# Physics Truth State
# ──────────────────────────────────────────────────────────────────────────────
def make_truth(alt_ft=35000, ias_kt=250, heading_deg=90, mass=65000):
  """Create initial truth state at given conditions, properly trimmed."""
  alt_m = alt_ft * FT2M
  tas = tas_from_ias(ias_kt * KT2MS, alt_m)
  hdg = heading_deg * DEG2RAD
  # Compute trim alpha for level flight
  rho = isa_rho(alt_m)
  q_bar = 0.5 * rho * tas**2
  S = AC['S']
  CL_needed = mass * G / (q_bar * S)
  alpha = (CL_needed - AC['CL0']) / AC['CLa']
  alpha = max(-5 * DEG2RAD, min(alpha, 12 * DEG2RAD))
  # Compute trim elevator to zero pitching moment
  # CM = CM0 + CMa*alpha + CMde*de = 0  =>  de = -(CM0 + CMa*alpha) / CMde
  de_trim = -(AC['CM0'] + AC['CMa'] * alpha) / AC['CMde']
  de_trim = max(-AC['elevator_max'], min(AC['elevator_max'], de_trim))
  # Compute trim throttle to match drag (accounting for altitude thrust lapse)
  CD = AC['CD0'] + CL_needed**2 / (math.pi * AC['AR'] * AC['e'])
  drag = q_bar * S * CD
  thrust_needed = drag + mass * G * math.sin(alpha)  # small alpha correction
  thrust_ratio = (rho / RHO_SL)**0.7  # turbofan thrust lapse
  max_thrust_at_alt = AC['max_thrust_per_engine'] * thrust_ratio
  thrust_per_engine = thrust_needed / AC['num_engines']
  n1_trim = max(20, min(100, thrust_per_engine / max_thrust_at_alt * 100))
  throttle_trim = n1_trim / 100.0
  return {
    'x': 0.0,
    'y': 0.0,
    'z': alt_m,
    'u': tas * math.cos(alpha),  # body fwd
    'v': 0.0,  # body side
    'w': tas * math.sin(alpha),  # body down-ish
    'phi': 0.0,
    'theta': alpha,
    'psi': hdg,
    'p': 0.0,
    'q': 0.0,
    'r': 0.0,
    'mass': float(mass),
    'fuel_left': mass * 0.15,
    'fuel_right': mass * 0.15,
    'fuel_center': mass * 0.05,
    # Control surface positions (radians) - trimmed for level flight
    'de': de_trim,
    'da': 0.0,
    'dr': 0.0,
    'dt_trim': 0.0,
    'de_trim_base': de_trim,  # baseline elevator for trimmed flight
    'flap_deg': 0.0,
    'slat_deg': 0.0,
    'speedbrake': 0.0,
    'gear': 0.0,  # 0=up, 1=down
    # Engine states - trimmed for level flight
    'throttle': [throttle_trim, throttle_trim],
    'engine_running': [True, True],
    'engine_fire': [False, False],
    'n1': [n1_trim, n1_trim],
    # Wind
    'wind_x': 0.0,
    'wind_y': 0.0,
    'wind_z': 0.0,
    'turbulence': 0.0,
    # Failure modifiers
    'cl_factor': 1.0,  # icing degrades lift
    'cd_extra': 0.0,  # extra drag
    'cm_extra': 0.0,  # extra pitch moment (stuck trim etc)
    'cn_extra': 0.0,  # extra yaw moment
    'cl_roll_extra': 0.0,  # extra roll moment
    'elevator_jam': None,  # if not None, elevator stuck at this value
    'rudder_jam': None,
    'aileron_jam': None,
    'trim_runaway_rate': 0.0,  # rad/s trim movement
    'time': 0.0,
  }


# ──────────────────────────────────────────────────────────────────────────────
# Aerodynamics
# ──────────────────────────────────────────────────────────────────────────────
def aero_coefficients(alpha, beta, de, da, dr, p, q, r, V, flap_deg, speedbrake, gear, cl_factor,
                      cd_extra, cm_extra, cn_extra, cl_roll_extra):
  """Compute aero force and moment coefficients."""
  ac = AC
  c, b = ac['c'], ac['b']
  V = max(V, 1.0)
  inv2V = 1.0 / (2.0 * V)

  # Lift
  CL = (ac['CL0'] + ac['CLa'] * alpha + ac['CLde'] * de + ac['CLflap'] * flap_deg)
  CL *= cl_factor
  # Stall model
  alpha_stall = ac['alpha_stall_flap'] if flap_deg > 5 else ac['alpha_stall_clean']
  if alpha > alpha_stall:
    overshoot = alpha - alpha_stall
    CL -= ac['CLa'] * overshoot * 1.5  # post-stall lift loss
    CL = max(CL, 0.1)
  elif alpha < -10 * DEG2RAD:
    CL = max(CL, -0.5)

  # Drag
  CL_for_drag = max(CL, 0)
  CDi = CL_for_drag**2 / (math.pi * ac['AR'] * ac['e'])
  CD = (ac['CD0'] + CDi + ac['CDflap'] * flap_deg + ac['CDgear'] * gear +
        ac['CDspeedbrake'] * speedbrake + cd_extra)
  # Post-stall drag increase
  if alpha > alpha_stall:
    CD += 0.1 * (alpha - alpha_stall) / DEG2RAD

  # Pitch moment
  CM = (ac['CM0'] + ac['CMa'] * alpha + ac['CMde'] * de + ac['CMq'] * q * c * inv2V + cm_extra)
  # Side force
  CY = ac['CYb'] * beta + ac['CYdr'] * dr
  # Roll moment
  Cl = (ac['Clb'] * beta + ac['Clda'] * da + ac['Clp'] * p * b * inv2V + cl_roll_extra)
  # Yaw moment
  Cn = (ac['Cnb'] * beta + ac['Cndr'] * dr + ac['Cnr'] * r * b * inv2V + cn_extra)

  return CL, CD, CM, CY, Cl, Cn


# ──────────────────────────────────────────────────────────────────────────────
# Physics Step
# ──────────────────────────────────────────────────────────────────────────────
def physics_step(truth, controls, dt=0.05):
  """Advance physics one timestep. Controls: dict with normalized inputs."""
  ac = AC
  t = dict(truth)  # shallow copy

  # ── Apply controls with rate limiting and jams ──
  def rate_limit(current, target, rate, dt_):
    diff = target - current
    max_change = rate * dt_
    return current + max(min(diff, max_change), -max_change)

  # Elevator (command is RELATIVE to trim base: 0 = trimmed flight)
  # Negate: positive command = nose up, but CMde < 0 so we need negative de for nose-up
  base_de = t.get('de_trim_base', 0.0)
  tgt_de = base_de - controls.get('elevator', 0.0) * ac['elevator_max']
  if t['elevator_jam'] is not None:
    t['de'] = t['elevator_jam']
  else:
    t['de'] = rate_limit(t['de'], tgt_de, ac['elevator_rate'], dt)

  # Aileron
  tgt_da = controls.get('aileron', 0.0) * ac['aileron_max']
  if t['aileron_jam'] is not None:
    t['da'] = t['aileron_jam']
  else:
    t['da'] = rate_limit(t['da'], tgt_da, ac['aileron_rate'], dt)

  # Rudder (negate: positive command = yaw right, but Cndr < 0 so need negative dr)
  tgt_dr = -controls.get('rudder', 0.0) * ac['rudder_max']
  if t['rudder_jam'] is not None:
    t['dr'] = t['rudder_jam']
  else:
    t['dr'] = rate_limit(t['dr'], tgt_dr, ac['rudder_rate'], dt)

  # Trim (with possible runaway)
  tgt_trim = controls.get('elevator_trim', 0.0) * ac['trim_max']
  t['dt_trim'] = rate_limit(t['dt_trim'], tgt_trim, 2 * DEG2RAD, dt)
  t['dt_trim'] += t['trim_runaway_rate'] * dt
  t['dt_trim'] = max(-ac['trim_max'], min(t['dt_trim'], ac['trim_max']))

  # Throttle
  for i in range(ac['num_engines']):
    key = f'throttle_{i+1}'
    tgt = controls.get(key, t['throttle'][i])
    tgt = max(0.0, min(1.0, tgt))
    t['throttle'][i] = rate_limit(t['throttle'][i], tgt, ac['throttle_rate'], dt)
    # Engine spool
    tgt_n1 = t['throttle'][i] * 100.0 if t['engine_running'][i] else 0.0
    t['n1'][i] += (tgt_n1 - t['n1'][i]) * min(1.0, dt * 1.5)

  # Flaps
  tgt_flap = max(0, min(40, controls.get('flaps', t['flap_deg'])))
  t['flap_deg'] = rate_limit(t['flap_deg'], tgt_flap, 2.0, dt)  # 2 deg/s

  # Speedbrake
  t['speedbrake'] = max(0, min(1, controls.get('speedbrake', t['speedbrake'])))

  # Gear
  tgt_gear = 1.0 if controls.get('gear', 0) else 0.0
  t['gear'] = rate_limit(t['gear'], tgt_gear, 0.1, dt)  # ~10s transit

  # ── Compute forces ──
  u, v, w = t['u'], t['v'], t['w']
  V = math.sqrt(u * u + v * v + w * w)
  V = max(V, 1.0)
  alpha = math.atan2(w, max(u, 0.1))
  beta = math.asin(max(-1, min(1, v / V)))

  rho = isa_rho(t['z'])
  qbar = 0.5 * rho * V * V
  S = ac['S']

  CL, CD, CM, CY, Cl, Cn = aero_coefficients(alpha, beta,
                                             t['de'] + t['dt_trim'] * ac['CMtrim'] / ac['CMde'],
                                             t['da'], t['dr'], t['p'], t['q'], t['r'], V,
                                             t['flap_deg'], t['speedbrake'], t['gear'],
                                             t['cl_factor'], t['cd_extra'], t['cm_extra'],
                                             t['cn_extra'], t['cl_roll_extra'])

  # Aero forces in wind axes -> body axes
  L = qbar * S * CL  # lift (perpendicular to V)
  D = qbar * S * CD  # drag (along V)
  Y = qbar * S * CY  # side force

  ca, sa = math.cos(alpha), math.sin(alpha)
  Fx_aero = -D * ca + L * sa
  Fz_aero = -D * sa - L * ca
  Fy_aero = Y

  # Engine thrust (body x-axis) — thrust lapse with altitude
  thrust_ratio = (rho / RHO_SL)**0.7  # turbofan thrust lapse
  Fx_thrust = 0.0
  yaw_thrust = 0.0
  for i in range(ac['num_engines']):
    thrust = (t['n1'][i] / 100.0) * ac['max_thrust_per_engine'] * thrust_ratio
    if not t['engine_running'][i]:
      thrust = 0.0
    Fx_thrust += thrust
    # Asymmetric thrust creates yaw moment
    sign = 1.0 if i == 0 else -1.0  # engine 1 left, engine 2 right
    yaw_thrust += thrust * sign * ac['engine_y_offset']

  # Gravity in body axes
  sp, cp = math.sin(t['theta']), math.cos(t['theta'])
  sr, cr = math.sin(t['phi']), math.cos(t['phi'])
  Fx_grav = -t['mass'] * G * sp
  Fy_grav = t['mass'] * G * cp * sr
  Fz_grav = t['mass'] * G * cp * cr

  # Wind and turbulence
  turb_scale = t['turbulence'] * V * 0.05
  rng = random.Random()
  wx = t['wind_x'] + rng.gauss(0, turb_scale)
  wy = t['wind_y'] + rng.gauss(0, turb_scale)
  wz = t['wind_z'] + rng.gauss(0, turb_scale)

  # Total forces
  Fx = Fx_aero + Fx_thrust + Fx_grav
  Fy = Fy_aero + Fy_grav
  Fz = Fz_aero + Fz_grav

  mass = t['mass']
  Ix, Iy, Iz = mass * 25, mass * 45, mass * 55  # approximate MOI
  Ixz = mass * 1.5

  # Moments
  L_moment = qbar * S * ac['b'] * Cl
  M_moment = qbar * S * ac['c'] * CM
  N_moment = qbar * S * ac['b'] * Cn + yaw_thrust

  # Turbulence moments
  if t['turbulence'] > 0:
    turb_m = t['turbulence'] * qbar * S * 0.01
    L_moment += rng.gauss(0, turb_m * ac['b'])
    M_moment += rng.gauss(0, turb_m * ac['c'])
    N_moment += rng.gauss(0, turb_m * ac['b'])

  # ── Integrate ──
  # Linear accelerations (body axes)
  ax = Fx / mass + t['r'] * v - t['q'] * w
  ay = Fy / mass - t['r'] * u + t['p'] * w
  az = Fz / mass + t['q'] * u - t['p'] * v

  t['u'] += ax * dt + rng.gauss(0, turb_scale) * dt
  t['v'] += ay * dt + rng.gauss(0, turb_scale * 0.3) * dt
  t['w'] += az * dt + rng.gauss(0, turb_scale * 0.5) * dt

  # Angular accelerations (simplified, ignoring cross-coupling for speed)
  pdot = (L_moment - (Iz - Iy) * t['q'] * t['r']) / Ix
  qdot = (M_moment - (Ix - Iz) * t['p'] * t['r']) / Iy
  rdot = (N_moment - (Iy - Ix) * t['p'] * t['q']) / Iz

  t['p'] += pdot * dt
  t['q'] += qdot * dt
  t['r'] += rdot * dt

  # Euler angle rates
  sp, cp = math.sin(t['phi']), math.cos(t['phi'])
  tt = math.tan(max(min(t['theta'], 1.4), -1.4))
  ct = math.cos(max(min(t['theta'], 1.4), -1.4))
  ct = max(ct, 0.01)

  phi_dot = t['p'] + (t['q'] * sp + t['r'] * cp) * tt
  theta_dot = t['q'] * cp - t['r'] * sp
  psi_dot = (t['q'] * sp + t['r'] * cp) / ct

  t['phi'] += phi_dot * dt
  t['theta'] += theta_dot * dt
  t['psi'] += psi_dot * dt

  # Wrap psi to [0, 2pi)
  t['psi'] = t['psi'] % (2 * math.pi)
  # Clamp phi to [-pi, pi]
  if t['phi'] > math.pi:
    t['phi'] -= 2 * math.pi
  elif t['phi'] < -math.pi:
    t['phi'] += 2 * math.pi

  # Position update (NED frame)
  cp_, sp_ = math.cos(t['psi']), math.sin(t['psi'])
  ct_ = math.cos(t['theta'])
  st_ = math.sin(t['theta'])
  sr_, cr_ = math.sin(t['phi']), math.cos(t['phi'])

  # Body to NED velocity
  vn = (ct_ * cp_) * t['u'] + (sp_ * sr_ * ct_ - cr_ * sp_) * t['v'] + (cr_ * st_ * cp_ +
                                                                        sr_ * sp_) * t['w']
  ve = (ct_ * sp_) * t['u'] + (sp_ * sr_ * sp_ + cr_ * cp_) * t['v'] + (cr_ * st_ * sp_ -
                                                                        sr_ * cp_) * t['w']
  # NB: z positive up in our convention
  vd = -st_ * t['u'] + sr_ * ct_ * t['v'] + cr_ * ct_ * t['w']

  t['x'] += (vn + wx) * dt
  t['y'] += (ve + wy) * dt
  t['z'] += (-vd + wz) * dt  # z positive up
  t['z'] = max(t['z'], 0)  # ground

  # Fuel burn (very simplified)
  ff_total = 0
  for i in range(ac['num_engines']):
    if t['engine_running'][i]:
      ff = (t['n1'][i] / 100.0) * 0.8  # ~0.8 kg/s per engine at full thrust
      ff_total += ff
  fuel_burn = ff_total * dt
  if t['fuel_left'] + t['fuel_right'] + t['fuel_center'] > fuel_burn:
    # Burn from center first, then even from wings
    if t['fuel_center'] > fuel_burn * 0.5:
      t['fuel_center'] -= fuel_burn * 0.5
      t['fuel_left'] -= fuel_burn * 0.25
      t['fuel_right'] -= fuel_burn * 0.25
    else:
      t['fuel_left'] -= fuel_burn * 0.5
      t['fuel_right'] -= fuel_burn * 0.5
    t['fuel_left'] = max(t['fuel_left'], 0)
    t['fuel_right'] = max(t['fuel_right'], 0)
    t['fuel_center'] = max(t['fuel_center'], 0)
    t['mass'] -= fuel_burn
  else:
    # Fuel exhaustion - engines flame out
    for i in range(ac['num_engines']):
      t['engine_running'][i] = False

  t['time'] += dt
  return t


# ──────────────────────────────────────────────────────────────────────────────
# Sensor State Builder
# ──────────────────────────────────────────────────────────────────────────────
class SensorFailure:
  """Defines a sensor failure mode."""
  NONE = 'none'
  STUCK = 'stuck'  # frozen at a value
  BIAS = 'bias'  # constant offset
  DRIFT = 'drift'  # growing offset
  DEAD = 'dead'  # reads 0
  SPIKE = 'spike'  # occasional wild values
  NOISE = 'noise'  # excessive noise


class SensorModel:
  """Generates sensor readings from truth state with noise and failures."""

  def __init__(self, seed=42):
    self.rng = random.Random(seed)
    self.failures = {}  # key -> (mode, param)
    self.stuck_vals = {}  # key -> stuck value
    self.drift_accum = {}  # key -> accumulated drift
    self.terrain_alt_m = 0.0  # ground elevation

  def set_failure(self, key, mode, param=0.0):
    self.failures[key] = (mode, param)

  def set_flicker(self, key):
    """Mark a sensor key as flickering (intermittent dropouts)."""
    self.failures[key] = ('flicker', 0)

  def _apply_failure(self, key, true_val, noise_std=0.0):
    """Apply noise and failure to a sensor reading."""
    val = true_val + self.rng.gauss(0, noise_std)
    if key not in self.failures:
      return val, True

    mode, param = self.failures[key]
    if mode == SensorFailure.STUCK:
      if key not in self.stuck_vals:
        self.stuck_vals[key] = val
      return self.stuck_vals[key], True  # valid flag stays true (unflagged!)
    elif mode == SensorFailure.BIAS:
      return val + param, True
    elif mode == SensorFailure.DRIFT:
      self.drift_accum[key] = self.drift_accum.get(key, 0.0) + param * 0.05
      return val + self.drift_accum[key], True
    elif mode == SensorFailure.DEAD:
      return 0.0, False
    elif mode == SensorFailure.SPIKE:
      if self.rng.random() < 0.05:
        return val + self.rng.gauss(0, param), True
      return val, True
    elif mode == SensorFailure.NOISE:
      return val + self.rng.gauss(0, param), True
    elif mode == 'flicker':
      if self.rng.random() < 0.15:
        return val + self.rng.gauss(0, abs(true_val) * 0.3 + 5), False
      return val, True
    return val, True

  def build_state(self, truth):
    """Build the full sensor state dict from physics truth."""
    t = truth
    V = math.sqrt(t['u']**2 + t['v']**2 + t['w']**2)
    V = max(V, 1.0)
    alpha = math.atan2(t['w'], max(t['u'], 0.1))
    beta = math.asin(max(-1, min(1, t['v'] / V)))
    alt_m = t['z']
    alt_ft = alt_m * M2FT
    agl_ft = (alt_m - self.terrain_alt_m) * M2FT
    rho = isa_rho(alt_m)
    tas_kt = V * MS2KT
    ias_kt = ias_from_tas(V, alt_m) * MS2KT
    mach = V / speed_of_sound(alt_m)

    # Heading, track, etc
    hdg_deg = (t['psi'] * RAD2DEG) % 360
    pitch_deg = t['theta'] * RAD2DEG
    roll_deg = t['phi'] * RAD2DEG
    # Vertical speed
    cp_ = math.cos(t['psi'])
    sp_ = math.sin(t['psi'])
    ct_ = math.cos(t['theta'])
    st_ = math.sin(t['theta'])
    sr_ = math.sin(t['phi'])
    cr_ = math.cos(t['phi'])
    vd = -st_ * t['u'] + sr_ * ct_ * t['v'] + cr_ * ct_ * t['w']
    vs_fpm = -vd * MS2FPM  # positive = climbing

    gs_kt = math.sqrt((ct_ * cp_ * t['u'] + (sp_ * sr_ * ct_ - cr_ * sp_) * t['v'])**2 +
                      (ct_ * sp_ * t['u'] + (sp_ * sr_ * sp_ + cr_ * cp_) * t['v'])**2) * MS2KT

    oat_c = isa_temp(alt_m) - 273.15
    tat_c = oat_c + (tas_kt * KT2MS)**2 / (2 * 1005)

    # Fake lat/lon from x/y (meters from origin, ~1deg = 111km)
    lat = 33.0 + t['x'] / 111000
    lon = -117.0 + t['y'] / (111000 * math.cos(33 * DEG2RAD))

    s = {}

    # ── Barometric Altimeters (3) ──
    for i in range(1, 4):
      v, ok = self._apply_failure(f'baro_alt_{i}', alt_ft, 10)
      s[f'baro_alt_{i}_ft'] = v
      s[f'baro_alt_{i}_valid'] = ok
      s[f'baro_setting_{i}_inhg'] = 29.92  # standard

    # ── Radar Altimeters (3) ──
    for i in range(1, 4):
      v, ok = self._apply_failure(f'radar_alt_{i}', max(0, agl_ft), 2)
      s[f'radar_alt_{i}_ft'] = v
      s[f'radar_alt_{i}_valid'] = ok and agl_ft < 2500

    # ── Airspeed (3 ADCs) ──
    for i in range(1, 4):
      vi, ok = self._apply_failure(f'ias_{i}', ias_kt, 1.0)
      s[f'ias_{i}_kt'] = max(0, vi)
      s[f'ias_{i}_valid'] = ok
      vt, _ = self._apply_failure(f'tas_{i}', tas_kt, 1.5)
      s[f'tas_{i}_kt'] = max(0, vt)
      vm, _ = self._apply_failure(f'mach_{i}', mach, 0.002)
      s[f'mach_{i}'] = max(0, vm)

    # ── Angle of Attack (3) ──
    for i in range(1, 4):
      v, ok = self._apply_failure(f'aoa_{i}', alpha * RAD2DEG, 0.3)
      s[f'aoa_{i}_deg'] = v
      s[f'aoa_{i}_valid'] = ok

    # ── GPS (3) ──
    for i in range(1, 4):
      glat, ok = self._apply_failure(f'gps_{i}', lat, 0.00001)
      s[f'gps_{i}_lat'] = glat
      glon, _ = self._apply_failure(f'gps_{i}_lon', lon, 0.00001)
      s[f'gps_{i}_lon'] = glon
      galt, _ = self._apply_failure(f'gps_{i}_alt', alt_ft, 30)
      s[f'gps_{i}_alt_ft'] = galt
      s[f'gps_{i}_gs_kt'] = gs_kt + self.rng.gauss(0, 2)
      s[f'gps_{i}_track_deg'] = hdg_deg + self.rng.gauss(0, 0.5)
      s[f'gps_{i}_valid'] = ok

    # ── IRS / INS (3) ──
    for i in range(1, 4):
      s[f'irs_{i}_lat'] = lat + self.rng.gauss(0, 0.0001)
      s[f'irs_{i}_lon'] = lon + self.rng.gauss(0, 0.0001)
      irh, ok = self._apply_failure(f'irs_{i}_hdg', hdg_deg, 0.2)
      s[f'irs_{i}_heading_deg'] = irh % 360
      irp, _ = self._apply_failure(f'irs_{i}_pitch', pitch_deg, 0.1)
      s[f'irs_{i}_pitch_deg'] = irp
      irr, _ = self._apply_failure(f'irs_{i}_roll', roll_deg, 0.1)
      s[f'irs_{i}_roll_deg'] = irr
      s[f'irs_{i}_gs_kt'] = gs_kt + self.rng.gauss(0, 1)
      s[f'irs_{i}_vs_fpm'] = vs_fpm + self.rng.gauss(0, 20)
      s[f'irs_{i}_track_deg'] = (hdg_deg + self.rng.gauss(0, 0.3)) % 360
      s[f'irs_{i}_valid'] = ok

    # ── Gyroscopes (3) ──
    for i in range(1, 4):
      gp, ok = self._apply_failure(f'gyro_{i}_p', t['p'] * RAD2DEG, 0.05)
      s[f'gyro_{i}_pitch_rate_dps'] = gp
      gr, _ = self._apply_failure(f'gyro_{i}_r', t['q'] * RAD2DEG, 0.05)
      s[f'gyro_{i}_roll_rate_dps'] = gr
      gy, _ = self._apply_failure(f'gyro_{i}_y', t['r'] * RAD2DEG, 0.05)
      s[f'gyro_{i}_yaw_rate_dps'] = gy
      s[f'gyro_{i}_valid'] = ok

    # ── Accelerometers (3) ──
    # Compute body-axis accelerations
    ax_g = (t['u'] - getattr(self, '_prev_u', t['u'])) / (G * 0.05) if hasattr(self,
                                                                               '_prev_u') else 0
    ay_g = (t['v'] - getattr(self, '_prev_v', t['v'])) / (G * 0.05) if hasattr(self,
                                                                               '_prev_v') else 0
    az_g = (t['w'] - getattr(self, '_prev_w', t['w'])) / (G * 0.05) - 1.0  # subtract gravity
    self._prev_u, self._prev_v, self._prev_w = t['u'], t['v'], t['w']
    for i in range(1, 4):
      s[f'accel_{i}_x_g'] = ax_g + self.rng.gauss(0, 0.01)
      s[f'accel_{i}_y_g'] = ay_g + self.rng.gauss(0, 0.01)
      s[f'accel_{i}_z_g'] = az_g + self.rng.gauss(0, 0.01)
      s[f'accel_{i}_valid'] = True

    # ── Air Data Computers (3) ──
    for i in range(1, 4):
      s[f'adc_{i}_oat_c'] = oat_c + self.rng.gauss(0, 0.5)
      s[f'adc_{i}_tat_c'] = tat_c + self.rng.gauss(0, 0.5)
      s[f'adc_{i}_pressure_alt_ft'] = alt_ft + self.rng.gauss(0, 15)
      s[f'adc_{i}_density_alt_ft'] = alt_ft + self.rng.gauss(0, 20)
      s[f'adc_{i}_valid'] = True

    # ── Flight Control Positions ──
    s['elevator_pos_deg'] = t['de'] * RAD2DEG
    s['aileron_pos_deg'] = t['da'] * RAD2DEG
    s['rudder_pos_deg'] = t['dr'] * RAD2DEG
    s['elevator_trim_deg'] = t['dt_trim'] * RAD2DEG
    s['flap_pos_left_deg'] = t['flap_deg']
    s['flap_pos_right_deg'] = t['flap_deg']
    s['slat_pos_left_deg'] = min(t['flap_deg'] * 0.7, 27)
    s['slat_pos_right_deg'] = min(t['flap_deg'] * 0.7, 27)
    s['spoiler_pos_left'] = t['speedbrake']
    s['spoiler_pos_right'] = t['speedbrake']
    s['speedbrake_handle'] = t['speedbrake']

    # ── Landing Gear ──
    s['gear_nose_pos'] = t['gear']
    s['gear_left_pos'] = t['gear']
    s['gear_right_pos'] = t['gear']
    s['gear_nose_locked'] = t['gear'] > 0.99 or t['gear'] < 0.01
    s['gear_left_locked'] = t['gear'] > 0.99 or t['gear'] < 0.01
    s['gear_right_locked'] = t['gear'] > 0.99 or t['gear'] < 0.01
    s['gear_handle_down'] = t['gear'] > 0.5
    on_ground = t['z'] < (self.terrain_alt_m + 0.5)
    s['weight_on_wheels_left'] = on_ground
    s['weight_on_wheels_right'] = on_ground
    s['weight_on_wheels_nose'] = on_ground

    # ── Engines ──
    for i in range(1, AC['num_engines'] + 1):
      idx = i - 1
      n1 = t['n1'][idx]
      running = t['engine_running'][idx]
      s[f'engine_{i}_n1_pct'] = n1 if running else 0
      s[f'engine_{i}_n2_pct'] = n1 * 1.02 if running else 0
      s[f'engine_{i}_egt_c'] = 400 + n1 * 4 if running else oat_c
      s[f'engine_{i}_ff_kgph'] = n1 * 30 if running else 0
      s[f'engine_{i}_oil_press_psi'] = 45 + n1 * 0.3 if running else 0
      s[f'engine_{i}_oil_temp_c'] = 80 + n1 * 0.5 if running else oat_c
      s[f'engine_{i}_vibration'] = 0.5 + self.rng.gauss(0, 0.1) if running else 0
      s[f'engine_{i}_running'] = running
      s[f'engine_{i}_fire'] = t['engine_fire'][idx]
      s[f'engine_{i}_thrust_pct'] = n1 if running else 0
      s[f'engine_{i}_epr'] = 1.0 + n1 * 0.006 if running else 1.0
      s[f'engine_{i}_bleed_air_psi'] = 35 if running and n1 > 50 else 0

    # ── Throttle ──
    for i in range(1, AC['num_engines'] + 1):
      s[f'throttle_{i}_pos'] = t['throttle'][i - 1]

    # ── Fuel ──
    s['fuel_tank_left_kg'] = t['fuel_left']
    s['fuel_tank_right_kg'] = t['fuel_right']
    s['fuel_tank_center_kg'] = t['fuel_center']
    s['fuel_total_kg'] = t['fuel_left'] + t['fuel_right'] + t['fuel_center']

    # ── Weight and CG ──
    s['gross_weight_kg'] = t['mass']
    fuel_total = t['fuel_left'] + t['fuel_right'] + t['fuel_center']
    s['cg_pct_mac'] = 25.0 + (t['fuel_center'] / max(fuel_total, 1)) * 5

    # ── Environment ──
    s['oat_c'] = oat_c + self.rng.gauss(0, 0.3)
    s['tat_c'] = tat_c + self.rng.gauss(0, 0.3)
    s['wind_speed_kt'] = math.sqrt(t['wind_x']**2 + t['wind_y']**2) * MS2KT
    s['wind_dir_deg'] = math.degrees(math.atan2(t['wind_y'], t['wind_x'])) % 360
    s['turbulence_intensity'] = t['turbulence']
    s['icing_rate'] = 0.0  # set per scenario

    # ── Navigation ──
    s['magnetic_heading_deg'] = (hdg_deg + 12) % 360  # ~12 deg variation
    s['true_heading_deg'] = hdg_deg
    s['magnetic_variation_deg'] = 12.0
    s['vertical_speed_fpm'] = vs_fpm
    s['ground_speed_kt'] = gs_kt
    s['track_deg'] = hdg_deg
    s['drift_angle_deg'] = 0.0
    s['flight_path_angle_deg'] = math.degrees(math.atan2(-vd, max(V * ct_, 1)))

    # ── Electrical ──
    s['bus_1_voltage'] = 115.0
    s['bus_2_voltage'] = 115.0
    s['bus_ess_voltage'] = 28.0
    s['gen_1_active'] = t['engine_running'][0]
    s['gen_2_active'] = t['engine_running'][1]
    s['apu_gen_active'] = False
    s['battery_voltage'] = 24.0

    # ── Hydraulic ──
    for sys_name in ['a', 'b', 'c']:
      s[f'hyd_{sys_name}_press_psi'] = 3000 + self.rng.gauss(0, 20)
      s[f'hyd_{sys_name}_qty_pct'] = 98 + self.rng.gauss(0, 1)

    # ── Pneumatic ──
    bleed1 = 35 if t['engine_running'][0] and t['n1'][0] > 50 else 0
    bleed2 = 35 if t['engine_running'][1] and t['n1'][1] > 50 else 0
    s['bleed_1_press_psi'] = bleed1
    s['bleed_2_press_psi'] = bleed2
    s['apu_bleed_press_psi'] = 0
    s['pack_1_active'] = bleed1 > 10 or bleed2 > 10
    s['pack_2_active'] = bleed2 > 10 or bleed1 > 10
    s['cabin_alt_ft'] = min(8000, alt_ft * 0.22)
    s['cabin_vs_fpm'] = vs_fpm * 0.15

    # ── Autopilot mode (informational) ──
    s['ap_engaged'] = True
    s['at_engaged'] = True
    s['target_alt_ft'] = alt_ft  # set per scenario
    s['target_hdg_deg'] = hdg_deg
    s['target_ias_kt'] = ias_kt
    s['target_vs_fpm'] = 0

    # ── Sim metadata ──
    s['sim_time_s'] = t['time']
    s['dt'] = 0.05

    return s


# ──────────────────────────────────────────────────────────────────────────────
# Simulation Runner
# ──────────────────────────────────────────────────────────────────────────────
class SimRunner:
  """Runs a scenario with a provided autopilot function."""

  def __init__(self,
               truth,
               sensor_model,
               target_alt_ft=None,
               target_hdg_deg=None,
               target_ias_kt=None):
    self.truth = truth
    self.sensors = sensor_model
    self.target_alt_ft = target_alt_ft or truth['z'] * M2FT
    self.target_hdg_deg = target_hdg_deg or truth['psi'] * RAD2DEG
    self.target_ias_kt = target_ias_kt or (
      ias_from_tas(math.sqrt(truth['u']**2 + truth['v']**2 + truth['w']**2), truth['z']) * MS2KT)
    self.crashed = False
    self.crash_reason = ''
    self.max_g = 1.0
    self.history = []

  def run(self, autopilot_fn, steps=1200, dt=0.05):
    """Run simulation for given steps. Returns (final_state, history)."""
    for step in range(steps):
      state = self.sensors.build_state(self.truth)
      state['target_alt_ft'] = self.target_alt_ft
      state['target_hdg_deg'] = self.target_hdg_deg
      state['target_ias_kt'] = self.target_ias_kt

      # Call autopilot
      try:
        controls = autopilot_fn(state, dt)
        if not isinstance(controls, dict):
          controls = {}
      except Exception:
        controls = {}

      # Sanitize controls
      for k in ['elevator', 'aileron', 'rudder', 'elevator_trim', 'speedbrake']:
        if k in controls:
          controls[k] = max(-1.0, min(1.0, float(controls[k])))
      for k in ['throttle_1', 'throttle_2']:
        if k in controls:
          controls[k] = max(0.0, min(1.0, float(controls[k])))
      if 'flaps' in controls:
        controls['flaps'] = max(0, min(40, float(controls['flaps'])))

      # Step physics
      self.truth = physics_step(self.truth, controls, dt)

      # Check crash conditions
      if self.truth['z'] <= 0 and step > 10:
        self.crashed = True
        self.crash_reason = 'CFIT: Aircraft hit the ground'
        break

      # Check G limits
      V = math.sqrt(self.truth['u']**2 + self.truth['v']**2 + self.truth['w']**2)
      if V > 1:
        alpha = math.atan2(self.truth['w'], max(self.truth['u'], 0.1))
        rho = isa_rho(self.truth['z'])
        qbar = 0.5 * rho * V * V
        CL_approx = (self.truth['mass'] * G * math.cos(self.truth['phi']) / max(qbar * AC['S'], 1))
        n_load = qbar * AC['S'] * CL_approx / (self.truth['mass'] * G)
        self.max_g = max(self.max_g, abs(n_load))

      # Check structural overspeed
      ias_ms = ias_from_tas(V, self.truth['z'])
      if ias_ms * MS2KT > AC['Vne_kt'] * 1.2:
        self.crashed = True
        self.crash_reason = f'Structural failure: IAS {ias_ms * MS2KT:.0f} > Vne'
        break

      if self.max_g > AC['g_limit_pos'] * 1.5:
        self.crashed = True
        self.crash_reason = f'Structural failure: {self.max_g:.1f}G exceeded'
        break

      # Record history every 20 steps (~1s)
      if step % 20 == 0:
        self.history.append({
          'time': self.truth['time'],
          'alt_ft': self.truth['z'] * M2FT,
          'ias_kt': ias_from_tas(V, self.truth['z']) * MS2KT,
          'hdg_deg': self.truth['psi'] * RAD2DEG,
          'pitch_deg': self.truth['theta'] * RAD2DEG,
          'roll_deg': self.truth['phi'] * RAD2DEG,
          'vs_fpm': state.get('vertical_speed_fpm', 0),
        })

    return self.sensors.build_state(self.truth), self.history

  def score_recovery(self):
    """Score how well the aircraft recovered to straight and level."""
    if self.crashed:
      return 0.0, f'Crash: {self.crash_reason}'

    if not self.history:
      return 0.0, 'No simulation history'

    # Check last 10 seconds of data
    recent = self.history[-10:] if len(self.history) >= 10 else self.history

    avg_alt = sum(h['alt_ft'] for h in recent) / len(recent)
    avg_roll = sum(abs(h['roll_deg']) for h in recent) / len(recent)
    avg_pitch = sum(abs(h['pitch_deg']) for h in recent) / len(recent)
    avg_vs = sum(abs(h['vs_fpm']) for h in recent) / len(recent)
    last = recent[-1]

    # Alt deviation from target
    alt_err = abs(avg_alt - self.target_alt_ft)
    hdg_err = abs(((last['hdg_deg'] - self.target_hdg_deg + 180) % 360) - 180)
    ias_err = abs(last['ias_kt'] - self.target_ias_kt)

    score = 1.0
    details = []

    # Roll: should be near wings-level
    if avg_roll > 30:
      score -= 0.4
      details.append(f'excessive roll {avg_roll:.0f}°')
    elif avg_roll > 10:
      score -= 0.15
      details.append(f'roll {avg_roll:.0f}°')

    # Pitch: should be reasonable
    if avg_pitch > 20:
      score -= 0.3
      details.append(f'excessive pitch {avg_pitch:.0f}°')
    elif avg_pitch > 8:
      score -= 0.1
      details.append(f'pitch {avg_pitch:.0f}°')

    # VS: should be stable
    if avg_vs > 2000:
      score -= 0.3
      details.append(f'unstable VS {avg_vs:.0f} fpm')
    elif avg_vs > 500:
      score -= 0.1
      details.append(f'VS {avg_vs:.0f} fpm')

    # Altitude error (allow generous tolerance for emergencies)
    if alt_err > 5000:
      score -= 0.2
      details.append(f'alt err {alt_err:.0f} ft')
    elif alt_err > 1000:
      score -= 0.1

    # Heading error
    if hdg_err > 30:
      score -= 0.1
      details.append(f'hdg err {hdg_err:.0f}°')

    # Speed
    if last['ias_kt'] < 100 or last['ias_kt'] > 400:
      score -= 0.2
      details.append(f'speed {last["ias_kt"]:.0f} kt')

    score = max(0.0, score)
    summary = (f'Alt: {avg_alt:.0f} ft (tgt {self.target_alt_ft:.0f}), '
               f'HDG: {last["hdg_deg"]:.0f}°, IAS: {last["ias_kt"]:.0f} kt, '
               f'Roll: {avg_roll:.1f}°, VS: {avg_vs:.0f} fpm')
    if details:
      summary += ' | Issues: ' + ', '.join(details)

    return score, summary


def list_all_state_keys():
  """Return a sorted list of all keys in the sensor state dict."""
  t = make_truth()
  sm = SensorModel()
  s = sm.build_state(t)
  return sorted(s.keys())
