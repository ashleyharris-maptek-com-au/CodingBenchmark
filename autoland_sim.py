"""
Autoland Simulator — extends autopilot_sim.py for ILS approach and landing.

Adds:
- ILS glideslope and localizer deviation sensors (triple redundant)
- Decision height, runway geometry
- Ground contact physics (braking, reverse thrust, spoilers)
- Flare maneuver detection and scoring
- Runway-relative coordinate system

Reuses aircraft parameters, physics, and sensor model from autopilot_sim.py.
"""

import math
import random
from typing import Dict, Optional, Tuple, List

from autopilot_sim import (
  AC,
  DEG2RAD,
  RAD2DEG,
  FT2M,
  M2FT,
  KT2MS,
  MS2KT,
  G,
  RHO_SL,
  isa_rho,
  ias_from_tas,
  tas_from_ias,
  make_truth,
  physics_step,
  SensorModel,
  SensorFailure,
  list_all_state_keys as _base_state_keys,
)

# ──────────────────────────────────────────────────────────────────────────────
# ILS / Runway Parameters
# ──────────────────────────────────────────────────────────────────────────────
ILS = {
  'glideslope_deg': 3.0,  # standard 3° glideslope
  'runway_heading_deg': 90.0,  # runway 09
  'runway_length_m': 3000,  # 3km runway
  'runway_width_m': 45,
  'runway_elev_ft': 0,  # sea level
  'touchdown_zone_m': 300,  # first 300m is touchdown zone
  'decision_height_ft': 200,  # CAT I decision height
  'localizer_range_m': 20000,  # localizer signal range
  'glideslope_range_m': 15000,
  'threshold_crossing_ft': 50,  # height over threshold
}


def _gs_alt_at_dist(dist_m, gs_deg=3.0, threshold_ft=50):
  """Glideslope altitude (ft) at given distance from threshold."""
  return dist_m * math.tan(gs_deg * DEG2RAD) * M2FT + threshold_ft


# ──────────────────────────────────────────────────────────────────────────────
# Autoland Truth State (extends aircraft truth)
# ──────────────────────────────────────────────────────────────────────────────
def make_autoland_truth(dist_nm=10,
                        alt_ft=None,
                        ias_kt=160,
                        heading_deg=90,
                        mass=65000,
                        gs_deg=3.0,
                        rwy_heading=90):
  """Create initial truth state on approach.
  
  dist_nm: distance from runway threshold in nautical miles
  """
  dist_m = dist_nm * 1852
  if alt_ft is None:
    alt_ft = _gs_alt_at_dist(dist_m, gs_deg)

  truth = make_truth(alt_ft=alt_ft, ias_kt=ias_kt, heading_deg=heading_deg, mass=mass)

  # Override trim for approach configuration (3° descent with flaps/gear)
  # make_truth computes level-flight trim which gives unrealistic pitch at approach speed.
  # Real approach: ~3° pitch, ~3° alpha, ~3° FPA, throttle ~0.45
  fpa = -gs_deg * DEG2RAD  # flight path angle (negative = descending)
  alpha_app = 5.0 * DEG2RAD  # typical approach alpha with flaps
  theta_app = alpha_app + fpa  # pitch = alpha + FPA
  V = tas_from_ias(ias_kt * KT2MS, alt_ft * FT2M)
  truth['u'] = V * math.cos(alpha_app)
  truth['w'] = V * math.sin(alpha_app)
  truth['theta'] = theta_app
  # Approach throttle: enough thrust to maintain speed on glideslope
  # Roughly 40-55% for a descent with flaps/gear drag
  truth['throttle'] = [0.45, 0.45]
  truth['n1'] = [45, 45]
  truth['de_trim_base'] = -alpha_app * 0.8  # approximate trim elevator for approach

  # Approach config
  truth['flap_deg'] = 20.0
  truth['gear'] = 1.0

  # Runway-relative position
  truth['rwy_dist_m'] = dist_m  # distance to threshold (along runway axis)
  truth['rwy_offset_m'] = 0.0  # lateral offset from centerline
  truth['on_ground'] = False
  truth['touchdown'] = False
  truth['touchdown_vs'] = 0.0
  truth['touchdown_x'] = 0.0
  truth['stopped'] = False

  # Autoland-specific controls
  truth['reverse_thrust'] = [0.0, 0.0]
  truth['spoiler_deployed'] = False
  truth['autobrake'] = 0.0

  # ILS config
  truth['gs_deg'] = gs_deg
  truth['rwy_heading'] = rwy_heading * DEG2RAD

  return truth


# ──────────────────────────────────────────────────────────────────────────────
# Autoland Physics Step
# ──────────────────────────────────────────────────────────────────────────────
def autoland_physics_step(truth, controls, dt=0.05):
  """Simplified energy-based approach/landing physics.

  Uses a stable point-mass model instead of the full 6-DOF aero model
  which is tuned for cruise and unstable at approach speeds.
  
  Elevator controls pitch rate (→ VS), throttle controls speed,
  aileron controls heading rate (→ lateral offset).
  """
  t = dict(truth)
  for k in ['throttle', 'engine_running', 'engine_fire', 'n1', 'reverse_thrust']:
    if k in truth and isinstance(truth[k], list):
      t[k] = list(truth[k])

  # Apply autoland-specific controls
  if controls.get('spoiler', 0) > 0.5:
    t['speedbrake'] = 1.0
    t['spoiler_deployed'] = True
  t['autobrake'] = max(0, min(1, controls.get('autobrake', t['autobrake'])))
  if t['on_ground']:
    for i in range(AC['num_engines']):
      rev = controls.get(f'reverse_{i+1}', 0)
      t['reverse_thrust'][i] = max(0, min(1, rev))

  # ── Rate-limited controls ──
  def rl(cur, tgt, rate):
    d = tgt - cur
    mx = rate * dt
    return cur + max(min(d, mx), -mx)

  # Throttle
  for i in range(AC['num_engines']):
    key = f'throttle_{i+1}'
    if key in controls and t['engine_running'][i]:
      tgt = max(0, min(1, float(controls[key])))
      t['throttle'][i] = rl(t['throttle'][i], tgt, 1.0)
      t['n1'][i] = t['throttle'][i] * 100

  # Elevator → pitch rate (2°/s per unit, rate limited)
  elev_cmd = max(-1, min(1, controls.get('elevator', 0.0)))
  tgt_q = elev_cmd * 3.0 * DEG2RAD  # max 3°/s pitch rate
  t['q'] = rl(t['q'], tgt_q, 5.0 * DEG2RAD)
  t['theta'] += t['q'] * dt

  # Aileron → roll/heading rate
  ail_cmd = max(-1, min(1, controls.get('aileron', 0.0)))
  rud_cmd = max(-1, min(1, controls.get('rudder', 0.0)))
  tgt_yaw_rate = (ail_cmd * 2.0 + rud_cmd * 1.0) * DEG2RAD
  t['r'] = rl(t['r'], tgt_yaw_rate, 3.0 * DEG2RAD)
  t['psi'] += t['r'] * dt
  t['phi'] = ail_cmd * 25 * DEG2RAD  # approximate bank for display

  if not t['on_ground']:
    V = math.sqrt(t['u']**2 + t['w']**2)
    V = max(V, 10)

    # ── Speed dynamics ──
    avg_thr = sum(t['throttle'][i]
                  for i in range(AC['num_engines']) if t['engine_running'][i]) / max(
                    1, sum(t['engine_running']))
    # Thrust - drag balance
    rho = isa_rho(t['z'])
    thrust = avg_thr * AC['max_thrust_per_engine'] * AC['num_engines'] * (rho / RHO_SL)**0.7
    drag = 0.5 * rho * V**2 * AC['S'] * (AC['CD0'] + 0.02 * t['gear'] + 0.01 * t['speedbrake'])
    accel = (thrust - drag) / t['mass']
    t['u'] += accel * dt
    t['u'] = max(t['u'], 30)  # minimum forward speed

    # ── Vertical dynamics ──
    # VS driven by pitch attitude: descend when nose low, climb when nose high
    # FPA ≈ theta - alpha, where alpha ≈ 5° in approach
    alpha = 5.0 * DEG2RAD
    fpa = t['theta'] - alpha
    target_vd = -V * math.sin(fpa)  # vertical velocity (positive = up)
    # Smooth convergence to target VS
    vd_current = -t['w'] * math.cos(t['theta']) + t['u'] * math.sin(t['theta'])
    vd_new = vd_current + (target_vd - vd_current) * min(1, 2.0 * dt)
    # Update w from vd
    t['w'] = V * math.sin(alpha)  # maintain alpha
    t['z'] += vd_new * dt

    # Record vertical speed for sensors
    t['_vs_mps'] = vd_new

  else:
    # ── Ground roll ──
    V = math.sqrt(t['u']**2 + t['v']**2)
    t['z'] = max(t['z'], ILS['runway_elev_ft'] * FT2M + 0.1)
    t['w'] = 0
    t['theta'] = max(0, t['theta'] - 0.5 * DEG2RAD * dt)
    t['q'] = 0
    t['_vs_mps'] = 0

    # Braking
    brake_force = t['autobrake'] * 0.3 * t['mass'] * G
    if t['spoiler_deployed']:
      brake_force *= 1.5
    for i in range(AC['num_engines']):
      if t['engine_running'][i] and t['reverse_thrust'][i] > 0:
        brake_force += t['reverse_thrust'][i] * AC['max_thrust_per_engine'] * 0.5
    if V > 0.5:
      t['u'] = max(0, t['u'] - (brake_force / t['mass']) * dt)
    if V < 2.0:
      t['stopped'] = True
      t['u'] = 0
      t['v'] = 0

  # ── Update runway-relative position ──
  V = math.sqrt(t['u']**2 + t['v']**2 + t['w']**2)
  cos_diff = math.cos(t['psi'] - t['rwy_heading'])
  sin_diff = math.sin(t['psi'] - t['rwy_heading'])
  t['rwy_dist_m'] -= V * cos_diff * dt
  t['rwy_offset_m'] += V * sin_diff * dt

  # ── Ground contact detection ──
  alt_ft = t['z'] * M2FT
  if alt_ft <= ILS['runway_elev_ft'] + 1 and not t['on_ground']:
    vs_fpm = t.get('_vs_mps', 0) * 60 * M2FT
    t['on_ground'] = True
    t['touchdown'] = True
    t['touchdown_vs'] = vs_fpm
    t['touchdown_x'] = t['rwy_dist_m']
    t['z'] = ILS['runway_elev_ft'] * FT2M + 0.1

  t['time'] += dt
  return t


# ──────────────────────────────────────────────────────────────────────────────
# Autoland Sensor Model (extends base)
# ──────────────────────────────────────────────────────────────────────────────
class AutolandSensorModel(SensorModel):
  """Adds ILS glideslope, localizer, radar altimeter, and landing sensors."""

  def build_state(self, truth):
    """Build full sensor state including ILS signals."""
    s = super().build_state(truth)

    t = truth
    alt_ft = t['z'] * M2FT
    dist_m = t['rwy_dist_m']
    offset_m = t['rwy_offset_m']
    gs_deg = t.get('gs_deg', 3.0)

    # ── ILS Glideslope Deviation (3 receivers) ──
    # Positive = above glideslope, negative = below
    gs_alt_target = _gs_alt_at_dist(dist_m, gs_deg)
    gs_dev_dots = (alt_ft - gs_alt_target) / 50.0  # ~50ft per dot
    gs_dev_dots = max(-3.0, min(3.0, gs_dev_dots))
    gs_valid = dist_m < ILS['glideslope_range_m'] and dist_m > 50

    for i in range(1, 4):
      key = f'gs_{i}'
      if key in self.failures:
        v, ok = self._apply_failure(key, gs_dev_dots, 0.02)
      else:
        v = gs_dev_dots + self.rng.gauss(0, 0.02)
        ok = gs_valid
      s[f'glideslope_{i}_dots'] = v
      s[f'glideslope_{i}_valid'] = ok

    # ── ILS Localizer Deviation (3 receivers) ──
    # Positive = right of centerline, negative = left
    loc_dev_dots = offset_m / 5.0  # ~5m per dot
    loc_dev_dots = max(-3.0, min(3.0, loc_dev_dots))
    loc_valid = dist_m < ILS['localizer_range_m'] and dist_m > 50

    for i in range(1, 4):
      key = f'loc_{i}'
      if key in self.failures:
        v, ok = self._apply_failure(key, loc_dev_dots, 0.01)
      else:
        v = loc_dev_dots + self.rng.gauss(0, 0.01)
        ok = loc_valid
      s[f'localizer_{i}_dots'] = v
      s[f'localizer_{i}_valid'] = ok

    # ── Radar Altimeter (already in base, but add runway-aware version) ──
    s['radio_alt_ft'] = max(0, alt_ft - ILS['runway_elev_ft'])

    # ── Distance to threshold ──
    s['dme_1_nm'] = dist_m / 1852 + self.rng.gauss(0, 0.01)
    s['dme_2_nm'] = dist_m / 1852 + self.rng.gauss(0, 0.01)

    # ── Runway-relative ──
    s['rwy_offset_m'] = offset_m + self.rng.gauss(0, 0.5)
    s['rwy_dist_m'] = dist_m

    # ── Ground contact ──
    s['weight_on_wheels'] = t['on_ground']
    s['on_ground'] = t['on_ground']
    s['groundspeed_kt'] = math.sqrt(t['u']**2 + t['v']**2) * MS2KT

    # ── Decision height ──
    s['decision_height_ft'] = ILS['decision_height_ft']
    s['below_decision_height'] = alt_ft < ILS['decision_height_ft']

    # ── Targets ──
    s['target_gs_dots'] = 0.0  # on glideslope
    s['target_loc_dots'] = 0.0  # on centerline
    s['target_vref_kt'] = 137  # Vref for landing config

    return s


# ──────────────────────────────────────────────────────────────────────────────
# Simulation Runner
# ──────────────────────────────────────────────────────────────────────────────
class AutolandRunner:

  def __init__(self, truth, sensor_model, target_alt_ft=None, target_hdg_deg=90, target_ias_kt=137):
    self.truth = truth
    self.sensors = sensor_model
    self.target_alt_ft = target_alt_ft  # None = follow glideslope
    self.target_hdg_deg = target_hdg_deg
    self.target_ias_kt = target_ias_kt
    self.crashed = False
    self.crash_reason = ''
    self.history = []
    self.landed = False

  def run(self, autopilot_fn, steps=3000, dt=0.05):
    for step in range(steps):
      state = self.sensors.build_state(self.truth)
      state['target_hdg_deg'] = self.target_hdg_deg
      state['target_ias_kt'] = self.target_ias_kt
      if self.target_alt_ft is not None:
        state['target_alt_ft'] = self.target_alt_ft

      try:
        controls = autopilot_fn(state, dt)
        if not isinstance(controls, dict):
          controls = {}
      except Exception:
        controls = {}

      # Clamp standard controls
      for k in ['elevator', 'aileron', 'rudder', 'elevator_trim', 'speedbrake']:
        if k in controls:
          controls[k] = max(-1.0, min(1.0, float(controls[k])))
      for k in ['throttle_1', 'throttle_2']:
        if k in controls:
          controls[k] = max(0.0, min(1.0, float(controls[k])))

      self.truth = autoland_physics_step(self.truth, controls, dt)

      alt_ft = self.truth['z'] * M2FT
      V = math.sqrt(self.truth['u']**2 + self.truth['v']**2 + self.truth['w']**2)
      ias_kt = ias_from_tas(V, self.truth['z']) * MS2KT

      # Check crash conditions
      if alt_ft < -10:
        self.crashed = True
        self.crash_reason = 'CFIT: Below ground level'
        break

      if self.truth['on_ground'] and abs(self.truth['touchdown_vs']) > 800:
        self.crashed = True
        self.crash_reason = f'Hard landing: {abs(self.truth["touchdown_vs"]):.0f} fpm VS'
        break

      if self.truth['on_ground'] and abs(self.truth['rwy_offset_m']) > ILS['runway_width_m'] / 2:
        self.crashed = True
        self.crash_reason = f'Runway excursion: {self.truth["rwy_offset_m"]:.0f}m offset'
        break

      if self.truth['on_ground'] and self.truth['rwy_dist_m'] < -ILS['runway_length_m']:
        self.crashed = True
        self.crash_reason = f'Overrun: {-self.truth["rwy_dist_m"] - ILS["runway_length_m"]:.0f}m past end'
        break

      if ias_kt > AC['Vne_kt']:
        self.crashed = True
        self.crash_reason = f'Overspeed: {ias_kt:.0f} kt'
        break

      # Check successful landing
      if self.truth.get('stopped'):
        self.landed = True
        break

      # Record history
      if step % 20 == 0:
        self.history.append({
          'time': self.truth['time'],
          'alt_ft': alt_ft,
          'ias_kt': ias_kt,
          'rwy_dist_m': self.truth['rwy_dist_m'],
          'rwy_offset_m': self.truth['rwy_offset_m'],
          'on_ground': self.truth['on_ground'],
        })

  def score(self):
    if self.crashed:
      return 0.0, f'Crash: {self.crash_reason}'

    if self.landed:
      score = 1.0
      details = []

      # Touchdown quality
      vs = abs(self.truth.get('touchdown_vs', 0))
      if vs > 400:
        score -= 0.3
        details.append(f'hard touchdown {vs:.0f} fpm')
      elif vs > 250:
        score -= 0.1
        details.append(f'firm touchdown {vs:.0f} fpm')

      # Touchdown position
      tdx = self.truth.get('touchdown_x', 0)
      if tdx > ILS['runway_length_m'] * 0.5:
        score -= 0.2
        details.append(f'late touchdown at {tdx:.0f}m')
      elif tdx < 0:
        score -= 0.3
        details.append(f'short of runway by {-tdx:.0f}m')

      # Centerline tracking
      offset = abs(self.truth.get('rwy_offset_m', 0))
      if offset > 15:
        score -= 0.2
        details.append(f'offset {offset:.0f}m from centerline')

      score = max(0.0, score)
      summary = f'Landed! VS: {vs:.0f}fpm, TDZ: {tdx:.0f}m, Offset: {offset:.1f}m'
      if details:
        summary += ' | ' + ', '.join(details)
      return score, summary

    # Didn't land, didn't crash - partial credit
    alt = self.truth['z'] * M2FT
    dist = self.truth['rwy_dist_m']
    if dist < 500 and alt < 500:
      return 0.4, f'Close but no landing: alt {alt:.0f}ft, dist {dist:.0f}m'
    return 0.1, f'No landing attempt: alt {alt:.0f}ft, dist {dist:.0f}m'


def list_autoland_state_keys():
  t = make_autoland_truth()
  sm = AutolandSensorModel(seed=42)
  s = sm.build_state(t)
  return sorted(s.keys())
