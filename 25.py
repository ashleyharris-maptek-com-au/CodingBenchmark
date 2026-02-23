"""
Test 25: Asteroid Interception - N-Body Solar System (C# Implementation)
"""

import random
import subprocess
import sys
import os
import time
import math
import json
import uuid
from typing import List, Tuple, Dict, Optional
from pathlib import Path

# Import our native compiler helper
from native_compiler import CSharpCompiler, CompilationError, ExecutionError, describe_this_pc

title = "Asteroid Interception (C#)"

# Timeout in seconds (5 minutes)
TIMEOUT_SECONDS = 30

# Seed for reproducibility
RANDOM_SEED = 25252525

# Physical constants (SI units)
G = 6.67430e-11  # Gravitational constant
AU = 1.496e11  # Astronomical unit in meters
DAY = 86400.0  # Seconds per day
YEAR = 365.25 * DAY

# J2000 epoch (2000-01-01 12:00 TT)
J2000_JD = 2451545.0
# UNIX timestamp for 2000-01-01 12:00:00 UTC
J2000_UNIX_UTC = 946728000

# Bodies: (display_name, skyfield_target, mass_kg)
# Masses from IAU 2015 nominal values
BODY_CONFIG = [
  ("Sun",     "sun",                1.98892e30),
  ("Mercury", "mercury barycenter", 3.3011e23),
  ("Venus",   "venus barycenter",   4.8675e24),
  ("Earth",   "earth",              5.97237e24),
  ("Moon",    "moon",               7.342e22),
  ("Mars",    "mars barycenter",    6.4171e23),
  ("Jupiter", "jupiter barycenter", 1.89819e27),
  ("Saturn",  "saturn barycenter",  5.6834e26),
  ("Uranus",  "uranus barycenter",  8.6810e25),
  ("Neptune", "neptune barycenter", 1.02413e26),
]

# Lazy-loaded ephemeris cache
_eph_cache = {}


def _load_ephemeris():
  """Lazily load JPL DE421 ephemeris via skyfield."""
  if 'eph' not in _eph_cache:
    from skyfield.api import load
    _eph_cache['ts'] = load.timescale()
    _eph_cache['eph'] = load('de421.bsp')
  return _eph_cache['eph'], _eph_cache['ts']


def _get_body_state_at_jd(sf_name: str, jd: float):
  """Return (x, y, z, vx, vy, vz) in SI (metres, m/s) relative to SSB."""
  eph, ts = _load_ephemeris()
  t = ts.tt_jd(jd)
  pos = eph[sf_name].at(t)
  x, y, z = pos.position.km
  vx, vy, vz = pos.velocity.km_per_s
  return x * 1e3, y * 1e3, z * 1e3, vx * 1e3, vy * 1e3, vz * 1e3


class Body:
  """A celestial body in the simulation."""

  def __init__(self, name: str, mass: float, x: float, y: float, z: float, vx: float, vy: float,
               vz: float):
    self.name = name
    self.mass = mass
    self.x, self.y, self.z = x, y, z
    self.vx, self.vy, self.vz = vx, vy, vz


class NBodySimulation:
  """Restricted N-body simulation: planets from ephemeris, test particles integrated."""

  def __init__(self, seed: int = 0, epoch_jd: float = J2000_JD):
    self.rng = random.Random(seed)
    self.bodies: List[Body] = []
    self.time = 0.0
    self.epoch_jd = epoch_jd
    self._n_eph = 0  # number of ephemeris-backed bodies (first N in self.bodies)
    self._eph_table = None  # precomputed ephemeris: dict[body_idx] -> (px,py,pz,vx,vy,vz) arrays
    self._eph_dt = 3600.0  # precomputed table timestep
    self._eph_len = 0  # number of precomputed steps
    self._step_idx = 0  # current step index into table
    self._init_solar_system()

  def _init_solar_system(self):
    """Initialize solar system from JPL DE421 ephemeris at J2000."""
    for name, sf_name, mass in BODY_CONFIG:
      x, y, z, vx, vy, vz = _get_body_state_at_jd(sf_name, self.epoch_jd)
      self.bodies.append(Body(name, mass, x, y, z, vx, vy, vz))
    self._n_eph = len(BODY_CONFIG)

  def precompute_ephemeris(self, duration: float, dt: float = 3600.0):
    """Batch-precompute planet positions for entire simulation duration."""
    import numpy as np
    eph, ts = _load_ephemeris()
    n_steps = int(duration / dt) + 2
    jds = self.epoch_jd + np.arange(n_steps) * (dt / DAY)
    times = ts.tt_jd(jds)
    self._eph_table = {}
    for idx, (name, sf_name, mass) in enumerate(BODY_CONFIG):
      pos = eph[sf_name].at(times)
      px, py, pz = pos.position.km
      vx, vy, vz = pos.velocity.km_per_s
      self._eph_table[idx] = (px * 1e3, py * 1e3, pz * 1e3, vx * 1e3, vy * 1e3, vz * 1e3)
    self._eph_dt = dt
    self._eph_len = n_steps
    self._step_idx = 0
    # Set initial positions from table
    self._sync_ephemeris_bodies(0)

  def _sync_ephemeris_bodies(self, step_idx: int):
    """Update ephemeris body positions/velocities from precomputed table."""
    if self._eph_table is None:
      return
    idx = min(step_idx, self._eph_len - 1)
    for i in range(self._n_eph):
      px, py, pz, vx, vy, vz = self._eph_table[i]
      b = self.bodies[i]
      b.x, b.y, b.z = float(px[idx]), float(py[idx]), float(pz[idx])
      b.vx, b.vy, b.vz = float(vx[idx]), float(vy[idx]), float(vz[idx])

  def get_earth(self) -> Body:
    """Get Earth body."""
    for b in self.bodies:
      if b.name == "Earth":
        return b
    return None

  def add_asteroid(self, x: float, y: float, z: float, vx: float, vy: float, vz: float,
                   mass: float):
    """Add an asteroid to the simulation."""
    self.bodies.append(Body("Asteroid", mass, x, y, z, vx, vy, vz))

  def add_spacecraft(self, x: float, y: float, z: float, vx: float, vy: float, vz: float,
                     mass: float):
    """Add a spacecraft to the simulation."""
    self.bodies.append(Body("Spacecraft", mass, x, y, z, vx, vy, vz))

  def get_body(self, name: str) -> Optional[Body]:
    """Get body by name."""
    for b in self.bodies:
      if b.name == name:
        return b
    return None

  def _accel_on_dynamic(self):
    """Compute gravitational accelerations on dynamic (non-ephemeris) bodies only."""
    n = len(self.bodies)
    n_dyn = n - self._n_eph
    if n_dyn <= 0:
      return [], [], []
    ax = [0.0] * n_dyn
    ay = [0.0] * n_dyn
    az = [0.0] * n_dyn
    for di in range(n_dyn):
      i = self._n_eph + di
      bi = self.bodies[i]
      # Gravity from all ephemeris bodies
      for j in range(self._n_eph):
        bj = self.bodies[j]
        dx = bj.x - bi.x
        dy = bj.y - bi.y
        dz = bj.z - bi.z
        r2 = dx * dx + dy * dy + dz * dz
        if r2 < 1e6:
          continue
        r = math.sqrt(r2)
        f = G * bj.mass / (r * r2)
        ax[di] += f * dx
        ay[di] += f * dy
        az[di] += f * dz
      # Gravity from other dynamic bodies
      for dj in range(di + 1, n_dyn):
        j = self._n_eph + dj
        bj = self.bodies[j]
        dx = bj.x - bi.x
        dy = bj.y - bi.y
        dz = bj.z - bi.z
        r2 = dx * dx + dy * dy + dz * dz
        if r2 < 1e6:
          continue
        r = math.sqrt(r2)
        inv_r3 = 1.0 / (r * r2)
        fx = G * inv_r3 * dx
        fy = G * inv_r3 * dy
        fz = G * inv_r3 * dz
        ax[di] += bj.mass * fx
        ay[di] += bj.mass * fy
        az[di] += bj.mass * fz
        ax[dj] -= bi.mass * fx
        ay[dj] -= bi.mass * fy
        az[dj] -= bi.mass * fz
    return ax, ay, az

  def step(self, dt: float):
    """Advance simulation by dt using leapfrog for dynamic bodies, ephemeris for planets."""
    n_dyn = len(self.bodies) - self._n_eph
    half_dt = dt * 0.5

    if n_dyn > 0:
      # Half-kick
      ax, ay, az = self._accel_on_dynamic()
      for di in range(n_dyn):
        i = self._n_eph + di
        self.bodies[i].vx += ax[di] * half_dt
        self.bodies[i].vy += ay[di] * half_dt
        self.bodies[i].vz += az[di] * half_dt
      # Drift dynamic bodies
      for di in range(n_dyn):
        i = self._n_eph + di
        self.bodies[i].x += self.bodies[i].vx * dt
        self.bodies[i].y += self.bodies[i].vy * dt
        self.bodies[i].z += self.bodies[i].vz * dt

    # Advance time and update ephemeris bodies
    self._step_idx += 1
    self.time += dt
    self._sync_ephemeris_bodies(self._step_idx)

    if n_dyn > 0:
      # Half-kick with updated planet positions
      ax, ay, az = self._accel_on_dynamic()
      for di in range(n_dyn):
        i = self._n_eph + di
        self.bodies[i].vx += ax[di] * half_dt
        self.bodies[i].vy += ay[di] * half_dt
        self.bodies[i].vz += az[di] * half_dt

  def reverse_step(self, dt: float):
    """Reverse one leapfrog KDK step (for backward integration).
    Exactly undoes a forward step(): undo 2nd half-kick, undo drift,
    step ephemeris back, undo 1st half-kick."""
    n_dyn = len(self.bodies) - self._n_eph
    half_dt = dt * 0.5

    if n_dyn > 0:
      # Undo second half-kick (current planet positions)
      ax, ay, az = self._accel_on_dynamic()
      for di in range(n_dyn):
        i = self._n_eph + di
        self.bodies[i].vx -= ax[di] * half_dt
        self.bodies[i].vy -= ay[di] * half_dt
        self.bodies[i].vz -= az[di] * half_dt
      # Undo drift
      for di in range(n_dyn):
        i = self._n_eph + di
        self.bodies[i].x -= self.bodies[i].vx * dt
        self.bodies[i].y -= self.bodies[i].vy * dt
        self.bodies[i].z -= self.bodies[i].vz * dt

    # Step back in time and update ephemeris
    self._step_idx -= 1
    self.time -= dt
    self._sync_ephemeris_bodies(self._step_idx)

    if n_dyn > 0:
      # Undo first half-kick (previous planet positions)
      ax, ay, az = self._accel_on_dynamic()
      for di in range(n_dyn):
        i = self._n_eph + di
        self.bodies[i].vx -= ax[di] * half_dt
        self.bodies[i].vy -= ay[di] * half_dt
        self.bodies[i].vz -= az[di] * half_dt

  def run(self, duration: float, dt: float = 3600.0):
    """Run simulation for duration seconds."""
    # Auto-precompute ephemeris if not done yet
    if self._eph_table is None:
      self.precompute_ephemeris(self.time + duration + DAY, dt)
    steps = int(duration / dt)
    for _ in range(steps):
      self.step(dt)

  def distance(self, body1: str, body2: str) -> float:
    """Calculate distance between two bodies."""
    b1 = self.get_body(body1)
    b2 = self.get_body(body2)
    if not b1 or not b2:
      return float('inf')
    dx = b1.x - b2.x
    dy = b1.y - b2.y
    dz = b1.z - b2.z
    return math.sqrt(dx * dx + dy * dy + dz * dz)


def generate_impact_asteroid(sim: NBodySimulation, warning_time: float,
                             seed: int,
                             asteroid_mass: float = 1e9) -> Tuple[float, float, float, float, float, float]:
  """
    Generate an asteroid guaranteed to hit Earth at t=warning_time.

    Method: place asteroid at Earth's exact position at impact time with a
    random incoming velocity, then reverse-integrate the N-body simulation
    for the full warning period.  The resulting t=0 state is returned as
    initial conditions.  Forward-integrating from these conditions will
    reproduce the Earth impact.
    """
  rng = random.Random(seed)
  dt = 3600.0  # 1-hour integration steps

  # --- Impact conditions at t = warning_time ---
  impact_jd = sim.epoch_jd + warning_time / DAY
  ex, ey, ez, evx, evy, evz = _get_body_state_at_jd("earth", impact_jd)

  # Small random offset within Earth radius (so impact point varies)
  earth_radius = 6.371e6  # metres
  theta = rng.uniform(0, 2 * math.pi)
  phi = math.acos(rng.uniform(-1, 1))
  r = earth_radius * rng.uniform(0, 1)**(1.0 / 3.0)
  ix = ex + r * math.sin(phi) * math.cos(theta)
  iy = ey + r * math.sin(phi) * math.sin(theta)
  iz = ez + r * math.cos(phi)

  # Random incoming velocity relative to Earth (15-35 km/s from random dir)
  speed = rng.uniform(15000, 35000)
  vt = rng.uniform(0, 2 * math.pi)
  vp = math.acos(rng.uniform(-1, 1))
  rel_vx = speed * math.sin(vp) * math.cos(vt)
  rel_vy = speed * math.sin(vp) * math.sin(vt)
  rel_vz = speed * math.cos(vp)
  # Barycentric velocity = Earth velocity + relative velocity
  avx = evx + rel_vx
  avy = evy + rel_vy
  avz = evz + rel_vz

  # --- Reverse-integrate from impact back to t=0 ---
  rev = NBodySimulation(seed, epoch_jd=sim.epoch_jd)
  impact_steps = int(warning_time / dt)
  rev.precompute_ephemeris(warning_time + DAY, dt)

  # Jump ephemeris to impact time
  rev._step_idx = impact_steps
  rev.time = warning_time
  rev._sync_ephemeris_bodies(impact_steps)

  # Add asteroid at impact state (mass irrelevant for test-particle dynamics)
  rev.add_asteroid(ix, iy, iz, avx, avy, avz, asteroid_mass)

  # Walk backwards to t=0
  for _ in range(impact_steps):
    rev.reverse_step(dt)

  ast = rev.get_body("Asteroid")
  return ast.x, ast.y, ast.z, ast.vx, ast.vy, ast.vz


def format_scenario(sim: NBodySimulation, warning_days: float, impactor_mass: float,
                    delta_v_budget: float, asteroid_mass: float, t0_unix_utc: int) -> str:
  """Format scenario as input string."""
  lines = []

  # Header
  lines.append(
    f"{len(sim.bodies)} {warning_days:.1f} {impactor_mass:.0f} {delta_v_budget:.1f} {asteroid_mass:.2e} {t0_unix_utc}"
  )

  # Bodies
  for b in sim.bodies:
    lines.append(
      f"{b.name} {b.mass:.6e} {b.x:.6e} {b.y:.6e} {b.z:.6e} {b.vx:.6e} {b.vy:.6e} {b.vz:.6e}")

  # Solar system model
  lines.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "de421.bsp")))

  return "\n".join(lines)


# Test configurations
TEST_CASES = [
  # Subpass 0: Long warning, heavy impactor
  {
    "warning_days": 3650, # 10 years
    "impactor_mass": 10000,  # 10 tons
    "delta_v": 5000,  # 5 km/s
    "asteroid_mass": 1e9,  # 1000 tons
    "description": "10 years, 10t impactor, 5km/s dV",
    "start_offset_days": 0
  },
  # Subpass 1: Long warning
  {
    "warning_days": 3650,
    "impactor_mass": 5000,
    "delta_v": 4000,
    "asteroid_mass": 1e9,
    "description": "10 years, 5t impactor, 4km/s dV",
    "start_offset_days": 200
  },
  # Subpass 2: One year, lighter
  {
    "warning_days": 500,
    "impactor_mass": 2000,
    "delta_v": 3500,
    "asteroid_mass": 5e9,
    "description": "500 days, 2t impactor, 3.5km/s dV",
    "start_offset_days": 450
  },
  # Subpass 3: Two years
  {
    "warning_days": 730,
    "impactor_mass": 1000,
    "delta_v": 3000,
    "asteroid_mass": 1e10,
    "description": "2 years, 1t impactor, 3km/s dV",
    "start_offset_days": 800
  },
  # Subpass 4: Three years
  {
    "warning_days": 1095,
    "impactor_mass": 500,
    "delta_v": 2500,
    "asteroid_mass": 5e10,
    "description": "3 years, 500kg impactor, 2.5km/s dV",
    "start_offset_days": 1200
  },
  # Subpass 5: Five years
  {
    "warning_days": 1825,
    "impactor_mass": 300,
    "delta_v": 2000,
    "asteroid_mass": 1e11,
    "description": "5 years, 300kg impactor, 2km/s dV",
    "start_offset_days": 1800
  },
  # Extreme cases
  {
    "warning_days": 2555,  # 7 years
    "impactor_mass": 200,
    "delta_v": 1500,
    "asteroid_mass": 5e11,
    "description": "7 years, 200kg impactor, 1.5km/s dV",
    "start_offset_days": 2600
  },
  {
    "warning_days": 3650,  # 10 years
    "impactor_mass": 100,
    "delta_v": 1000,
    "asteroid_mass": 1e12,
    "description": "10 years, 100kg impactor, 1km/s dV",
    "start_offset_days": 3600
  },
  {
    "warning_days": 5475,  # 15 years
    "impactor_mass": 50,
    "delta_v": 800,
    "asteroid_mass": 5e12,
    "description": "15 years, 50kg impactor, 800m/s dV",
    "start_offset_days": 4800
  },
  {
    "warning_days": 7300,  # 20 years
    "impactor_mass": 20,
    "delta_v": 500,
    "asteroid_mass": 1e13,
    "description": "20 years, 20kg impactor, 500m/s dV",
    "start_offset_days": 6200
  },
  {
    "warning_days": 10950,  # 30 years
    "impactor_mass": 10,
    "delta_v": 300,
    "asteroid_mass": 5e13,
    "description": "30 years, 10kg impactor, 300m/s dV - decade-long intercept",
    "start_offset_days": 7800
  },
]

# Cache scenarios
SCENARIO_CACHE = {}

# Cache mission visualization data keyed by (subPass, aiEngineName)
LAST_MISSION_VIZ = {}


def get_scenario(subpass: int) -> Tuple[NBodySimulation, str]:
  """Get or generate scenario for subpass."""
  if subpass not in SCENARIO_CACHE:
    case = TEST_CASES[subpass]
    if "start_offset_days" not in case:
      raise ValueError("Each TEST_CASES entry must define start_offset_days")
    offset_days = case["start_offset_days"]
    epoch_jd = J2000_JD + offset_days
    t0_unix_utc = int(J2000_UNIX_UTC + offset_days * DAY)

    sim = NBodySimulation(RANDOM_SEED + subpass, epoch_jd=epoch_jd)

    # Generate asteroid
    warning_time = case["warning_days"] * DAY
    ax, ay, az, avx, avy, avz = generate_impact_asteroid(sim, warning_time,
                                                         RANDOM_SEED + subpass + 1000,
                                                         case["asteroid_mass"])
    sim.add_asteroid(ax, ay, az, avx, avy, avz, case["asteroid_mass"])

    input_str = format_scenario(sim, case["warning_days"], case["impactor_mass"], case["delta_v"],
                                case["asteroid_mass"], t0_unix_utc)

    SCENARIO_CACHE[subpass] = (sim, input_str)

  return SCENARIO_CACHE[subpass]


def prepareSubpassPrompt(subPass: int) -> str:
  """Generate the prompt for subpass 0 that handles all interception complexities."""
  if subPass != 0:
    raise StopIteration

  return f"""You are writing C# code to calculate an asteroid interception trajectory.

You must write a C# solver that can handle an interception and redirection scenarios, from
short warning times (1-2 years), large impactors, generous delta-v, simple trajectories to 
very long warning times (30-50 years), very small impactors, very tight delta-v, 
advanced trajectory optimization with multi-flyby maneuvers and gravitational assist sequences required.

**Problem:**
Calculate a spacecraft trajectory to intercept and deflect an asteroid on collision course 
with Earth using kinetic impact. 

**The solver must output:**
1. Launch window and initial trajectory
2. Mid-course correction burn(s)
3. Final approach for kinetic impact

**Input format (stdin, whitespace seperated decimal/scientific notation numbers):**
```
num_bodies warning_days impactor_mass delta_v_budget asteroid_mass t0_unix_utc
body_name mass x y z vx vy vz  (for each body)
de421.bsp_file_path
```

All values are in SI units unless otherwise specified. Eg mass of earth is 5.972e24 kg.
All positions + velocities are relative to solar system barycenter.
Delta_v budget is in m/s.

The body name for the asteriod is "Asteroid". The body name for where your spacecraft originates
and you must redirect the asteroid to avoid is "Earth".

**Output format (stdout):**
```
num_burns
launch_time burn_dv_x burn_dv_y burn_dv_z
correction_time1 burn_dv_x burn_dv_y burn_dv_z
...
impact_time impact_dv_x impact_dv_y impact_dv_z
```

**Time units:**
- All times are in **days since scenario start (t=0 at the provided initial state)**.
- The **launch window is [0, warning_days]** (inclusive).
- **t0_unix_utc** is the UTC time of the provided initial state (t=0).

The simulation uses the JPL DE421 model of the solar system with Sun, Earth, Moon, and major planets,
and a KDK integrator to calculate trajectories. The path to the DE421 model is provided in the input
so your code can precisely match the simulation at extreme precision.

**Success Criteria:**
- Spacecraft intercepts asteroid before Earth impact (within 1000km)
- Kinetic impact transfers enough momentum to deflect asteroid sufficiently to miss Earth
- Total delta-v usage stays within budget
- Launch occurs within specified window

**Simulation host environment**
{describe_this_pc()}

**C# Compiler**
{CSharpCompiler("test").describe()}

**Requirements:**
1. Program must compile with C# .NET/csc
2. Read from stdin, write to stdout
3. Handle variable warning times and delta-v budgets
4. Complete analysis within 5 minutes
5. Must handle varying mission complexities efficiently

Write complete, compilable C# code with a static void Main method.
"""


# List of subpasses to grade the single answer against all difficulty levels
extraGradeAnswerRuns = list(range(len(TEST_CASES)))

structure = {
  "type": "object",
  "properties": {
    "reasoning": {
      "type":
      "string",
      "description":
      "Explain your algorithm approach and how it adapts to different interception complexities"
    },
    "csharp_code": {
      "type": "string",
      "description": "Complete C# code with Main method that handles all scales"
    }
  },
  "required": ["reasoning", "csharp_code"],
  "additionalProperties": False
}


def parse_burns(output: str) -> List[Tuple[float, float, float, float]]:
  """Parse burn sequence from output."""
  lines = output.strip().split('\n')
  if not lines:
    return []

  try:
    num_burns = int(lines[0].strip())
    burns = []
    for i in range(1, min(num_burns + 1, len(lines))):
      parts = lines[i].strip().split()
      if len(parts) >= 4:
        day = float(parts[0])
        dvx = float(parts[1])
        dvy = float(parts[2])
        dvz = float(parts[3])
        burns.append((day, dvx, dvy, dvz))
    return burns
  except:
    return []


def simulate_mission(sim: NBodySimulation, burns: List[Tuple[float, float, float, float]],
                     impactor_mass: float, warning_days: float) -> Tuple[bool, float, float, dict]:
  """
    Simulate the interception mission.
    Returns (intercepted, closest_approach, earth_miss_distance, viz_data).
    viz_data contains sampled trajectory for visualization.
    """
  # Create fresh simulation (same epoch as scenario)
  test_sim = NBodySimulation(sim.rng.randint(0, 1000000), epoch_jd=sim.epoch_jd)

  # Precompute ephemeris for entire mission duration upfront
  total_duration = (warning_days + 100) * DAY
  dt = 3600.0  # 1 hour steps
  test_sim.precompute_ephemeris(total_duration + DAY, dt)

  # Copy asteroid
  orig_asteroid = sim.get_body("Asteroid")
  if orig_asteroid:
    test_sim.add_asteroid(orig_asteroid.x, orig_asteroid.y, orig_asteroid.z, orig_asteroid.vx,
                          orig_asteroid.vy, orig_asteroid.vz, orig_asteroid.mass)

  # Also run an undeflected asteroid sim for "original path" visualization
  undeflected_sim = NBodySimulation(sim.rng.randint(0, 1000000), epoch_jd=sim.epoch_jd)
  undeflected_sim.precompute_ephemeris(total_duration + DAY, dt)
  if orig_asteroid:
    undeflected_sim.add_asteroid(orig_asteroid.x, orig_asteroid.y, orig_asteroid.z, orig_asteroid.vx,
                                 orig_asteroid.vy, orig_asteroid.vz, orig_asteroid.mass)

  # Add spacecraft at Earth
  earth = test_sim.get_earth()
  if earth and burns:
    # First burn is launch
    day, dvx, dvy, dvz = burns[0]
    test_sim.add_spacecraft(earth.x, earth.y, earth.z, earth.vx + dvx, earth.vy + dvy,
                            earth.vz + dvz, impactor_mass)
  else:
    return False, float('inf'), 0, {}

  # Sort remaining burns by day
  remaining_burns = sorted(burns[1:], key=lambda b: b[0])
  burn_idx = 0

  closest_approach = float('inf')

  # --- Trajectory recording ---
  total_days = int(warning_days) + 100
  # Sample interval: aim for ~500 frames max
  sample_interval = max(1, total_days // 500)
  # Body names for planets we track
  planet_names = [cfg[0] for cfg in BODY_CONFIG]
  # Trajectory storage: {body_name: [[x,y,z], ...]}
  trajectories = {name: [] for name in planet_names}
  trajectories["Asteroid"] = []
  trajectories["Asteroid_Original"] = []
  trajectories["Spacecraft"] = []
  time_days = []
  # Key events
  launch_day = burns[0][0] if burns else 0
  intercept_day = None
  burn_events = [(b[0], math.sqrt(b[1]**2 + b[2]**2 + b[3]**2)) for b in burns]

  earth_radius = 6.371e6  # metres
  impact_day_deflected = None
  impact_day_original = None

  def _closest_approach_linear(a0, a1, e0, e1):
    """Closest approach between two points moving linearly over one day."""
    r0x = a0[0] - e0[0]
    r0y = a0[1] - e0[1]
    r0z = a0[2] - e0[2]
    r1x = a1[0] - e1[0]
    r1y = a1[1] - e1[1]
    r1z = a1[2] - e1[2]
    vx = r1x - r0x
    vy = r1y - r0y
    vz = r1z - r0z
    vv = vx * vx + vy * vy + vz * vz
    if vv == 0.0:
      t = 0.0
    else:
      t = -(r0x * vx + r0y * vy + r0z * vz) / vv
      t = max(0.0, min(1.0, t))
    rx = r0x + t * vx
    ry = r0y + t * vy
    rz = r0z + t * vz
    dist = math.sqrt(rx * rx + ry * ry + rz * rz)
    return dist, t

  def _record_frame(day_num):
    """Record positions of all bodies at current state."""
    time_days.append(day_num)
    for name in planet_names:
      b = test_sim.get_body(name)
      if b:
        trajectories[name].append([b.x / AU, b.y / AU, b.z / AU])
      else:
        # Pad with last known or zero
        prev = trajectories[name][-1] if trajectories[name] else [0, 0, 0]
        trajectories[name].append(prev)
    # Asteroid (deflected)
    ast = test_sim.get_body("Asteroid")
    if ast:
      trajectories["Asteroid"].append([ast.x / AU, ast.y / AU, ast.z / AU])
    else:
      prev = trajectories["Asteroid"][-1] if trajectories["Asteroid"] else [0, 0, 0]
      trajectories["Asteroid"].append(prev)
    # Asteroid (original)
    ast_orig = undeflected_sim.get_body("Asteroid")
    if ast_orig:
      trajectories["Asteroid_Original"].append([ast_orig.x / AU, ast_orig.y / AU, ast_orig.z / AU])
    else:
      prev = trajectories["Asteroid_Original"][-1] if trajectories["Asteroid_Original"] else [0, 0, 0]
      trajectories["Asteroid_Original"].append(prev)
    # Spacecraft
    sc = test_sim.get_body("Spacecraft")
    if sc:
      trajectories["Spacecraft"].append([sc.x / AU, sc.y / AU, sc.z / AU])
    else:
      # Use None sentinel to indicate not present
      trajectories["Spacecraft"].append(None)

  # Record frame 0
  _record_frame(0)

  # Run simulation
  for day in range(total_days):
    # Apply any burns scheduled for this day
    while burn_idx < len(remaining_burns) and remaining_burns[burn_idx][0] <= day:
      _, dvx, dvy, dvz = remaining_burns[burn_idx]
      sc = test_sim.get_body("Spacecraft")
      if sc:
        sc.vx += dvx
        sc.vy += dvy
        sc.vz += dvz
      burn_idx += 1

    # Start-of-day positions
    earth0 = test_sim.get_earth()
    ast0 = test_sim.get_body("Asteroid")
    earth0_o = undeflected_sim.get_earth()
    ast0_o = undeflected_sim.get_body("Asteroid")

    # Run one day
    test_sim.run(DAY, dt)
    undeflected_sim.run(DAY, dt)

    # End-of-day positions
    earth1 = test_sim.get_earth()
    ast1 = test_sim.get_body("Asteroid")
    earth1_o = undeflected_sim.get_earth()
    ast1_o = undeflected_sim.get_body("Asteroid")

    # Check asteroid-Earth impact time (linear within-day interpolation)
    if impact_day_deflected is None and earth0 and ast0 and earth1 and ast1:
      dist, t = _closest_approach_linear(
        (ast0.x, ast0.y, ast0.z),
        (ast1.x, ast1.y, ast1.z),
        (earth0.x, earth0.y, earth0.z),
        (earth1.x, earth1.y, earth1.z),
      )
      if dist <= earth_radius:
        impact_day_deflected = day + t

    if impact_day_original is None and earth0_o and ast0_o and earth1_o and ast1_o:
      dist, t = _closest_approach_linear(
        (ast0_o.x, ast0_o.y, ast0_o.z),
        (ast1_o.x, ast1_o.y, ast1_o.z),
        (earth0_o.x, earth0_o.y, earth0_o.z),
        (earth1_o.x, earth1_o.y, earth1_o.z),
      )
      if dist <= earth_radius:
        impact_day_original = day + t

    # Check closest approach
    dist = test_sim.distance("Spacecraft", "Asteroid")
    closest_approach = min(closest_approach, dist)

    # Check for intercept (within 1000 km)
    if dist < 1e6:
      if intercept_day is None:
        intercept_day = day + 1
      # Intercept! Calculate deflection
      sc = test_sim.get_body("Spacecraft")
      ast = test_sim.get_body("Asteroid")
      if sc and ast:
        # Momentum transfer
        rel_vx = sc.vx - ast.vx
        rel_vy = sc.vy - ast.vy
        rel_vz = sc.vz - ast.vz

        # Deflection velocity
        dv = impactor_mass / ast.mass
        ast.vx += dv * rel_vx
        ast.vy += dv * rel_vy
        ast.vz += dv * rel_vz

        # Remove spacecraft
        test_sim.bodies = [b for b in test_sim.bodies if b.name != "Spacecraft"]

    # Record frame at sample interval
    if (day + 1) % sample_interval == 0 or day == total_days - 1:
      _record_frame(day + 1)

  # Check if asteroid misses Earth
  earth = test_sim.get_earth()
  asteroid = test_sim.get_body("Asteroid")
  if earth and asteroid:
    dx = asteroid.x - earth.x
    dy = asteroid.y - earth.y
    dz = asteroid.z - earth.z
    miss_dist = math.sqrt(dx * dx + dy * dy + dz * dz)
  else:
    miss_dist = 0

  intercepted = closest_approach < 1e6

  # Build viz data
  viz_data = {
    "time_days": time_days,
    "warning_days": warning_days,
    "total_days": total_days,
    "planet_names": planet_names,
    "trajectories": trajectories,
    "launch_day": launch_day,
    "intercept_day": intercept_day,
    "impact_day_deflected": impact_day_deflected,
    "impact_day_original": impact_day_original,
    "burn_events": burn_events,
    "intercepted": intercepted,
    "closest_approach_km": closest_approach / 1e3,
    "miss_dist_km": miss_dist / 1e3,
  }

  return intercepted, closest_approach, miss_dist, viz_data


def run_solver(code: str, input_data: str, engine_name: str) -> Tuple[str, str, float]:
  """Compile and run C# solver."""
  compiler = CSharpCompiler(engine_name)

  if not compiler.find_compiler():
    return "", "No C# compiler found", 0

  try:
    exe_path = compiler.compile(code)
    stdout, stderr, exec_time, retcode = compiler.execute(exe_path, input_data, TIMEOUT_SECONDS)
    if retcode != 0:
      return stdout, f"Runtime error: {stderr[:500]}", exec_time
    return stdout, "", exec_time
  except CompilationError as e:
    return "", f"Compilation error: {str(e)[:500]}", 0
  except ExecutionError as e:
    return "", str(e), TIMEOUT_SECONDS


def gradeAnswer(result: dict, subPass: int, aiEngineName: str) -> tuple:
  """Grade the asteroid interception solver."""
  if not result:
    return 0.0, "No result provided"

  if "csharp_code" not in result:
    return 0.0, "No C# code provided"

  case = TEST_CASES[subPass]
  description = case["description"]
  sim, input_data = get_scenario(subPass)

  code = result["csharp_code"]

  # Run solver
  stdout, error, exec_time = run_solver(code, input_data, aiEngineName)

  if error:
    return 0.0, f"[{description}] {error}"

  # Parse burns
  burns = parse_burns(stdout)
  if not burns:
    return 0.0, f"[{description}] No valid burn sequence"

  # Check delta-V budget
  total_dv = sum(math.sqrt(b[1]**2 + b[2]**2 + b[3]**2) for b in burns)
  if total_dv > case["delta_v"] * 1.1:  # 10% tolerance
    return 0.1, f"[{description}] Delta-V exceeded: {total_dv:.0f} > {case['delta_v']}"

  # Simulate mission
  intercepted, closest, miss_dist, viz_data = simulate_mission(sim, burns, case["impactor_mass"],
                                                               case["warning_days"])
  LAST_MISSION_VIZ[(subPass, aiEngineName)] = viz_data

  earth_radius = 6.371e6

  if intercepted and miss_dist > earth_radius:
    score = 1.0
    status = "SUCCESS - intercepted and deflected"
  elif intercepted:
    score = 0.7
    status = f"Intercepted but insufficient deflection ({miss_dist/1e6:.0f} km)"
  elif closest < 1e8:  # Within 100,000 km
    score = 0.4
    status = f"Close approach {closest/1e6:.0f} km"
  else:
    score = 0.0
    status = f"Missed - closest {closest/1e9:.1f} million km"

  explanation = (f"[{description}] {status}, "
                 f"dV used: {total_dv:.0f}/{case['delta_v']}m/s, "
                 f"Burns: {len(burns)}, Time: {exec_time:.2f}s")

  return score, explanation


def resultToNiceReport(result: dict, subPass: int, aiEngineName: str) -> str:
  """Generate HTML report with orbital trajectory visualization."""
  if not result:
    return "<p style='color:red'>No result provided</p>"

  case = TEST_CASES[subPass]
  html = f"<h4>Asteroid Interception - {case['description']}</h4>"

  # Show reasoning summary
  if "reasoning" in result and subPass == 0:
    reasoning = result['reasoning'][:500] + ('...'
                                             if len(result.get('reasoning', '')) > 500 else '')
    reasoning_escaped = reasoning.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    html += f"<p><strong>Strategy:</strong> {reasoning_escaped}</p>"

  # Show code in collapsible
  if "csharp_code" in result and subPass == 0:
    code = result["csharp_code"]
    code_escaped = code.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    html += f"<details><summary>View C# Code ({len(code)} chars)</summary><pre>{code_escaped}</pre></details>"

  # 3D visualization
  viz_data = LAST_MISSION_VIZ.get((subPass, aiEngineName))
  if viz_data and viz_data.get("time_days"):
    html += _generate_mission_3d(viz_data, case)
  else:
    html += "<p style='color:#94a3b8;'>No trajectory data available for visualization.</p>"

  return html


def _generate_mission_3d(viz_data: dict, case: dict) -> str:
  """Generate Three.js interactive solar system mission visualization."""
  uid = f"mission3d_{uuid.uuid4().hex[:8]}"

  # Planet visual config: name -> (color_hex, base_radius_AU)
  planet_vis = {
    "Sun":     ("ffcc00", 0.04),
    "Mercury": ("b0b0b0", 0.008),
    "Venus":   ("e8c060", 0.012),
    "Earth":   ("4488ff", 0.013),
    "Moon":    ("aaaaaa", 0.005),
    "Mars":    ("dd4422", 0.010),
    "Jupiter": ("d4a06a", 0.028),
    "Saturn":  ("e8d080", 0.024),
    "Uranus":  ("88ccdd", 0.018),
    "Neptune": ("4466cc", 0.017),
  }

  data_json = json.dumps(viz_data, separators=(',', ':'))

  return f'''
  <div style="margin:16px 0;padding:12px;border:1px solid #1f2937;border-radius:8px;background:#0b1120;">
    <h5 style="margin:0 0 8px 0;color:#e2e8f0;">3D Mission Trajectory</h5>
    <div id="{uid}" style="width:100%;height:550px;border:1px solid #334155;position:relative;"></div>
    <!-- Controls bar -->
    <div id="{uid}_ctrl" style="margin-top:8px;display:flex;flex-wrap:wrap;gap:6px;align-items:center;font-size:12px;color:#cbd5e1;">
      <!-- Playback -->
      <button id="{uid}_play" style="padding:3px 10px;background:#1e3a5f;color:#e2e8f0;border:1px solid #334155;border-radius:4px;cursor:pointer;">&#9654; Play</button>
      <select id="{uid}_speed" style="padding:2px 4px;background:#1e293b;color:#e2e8f0;border:1px solid #334155;border-radius:4px;">
        <option value="0.25">0.25x</option><option value="0.5">0.5x</option><option value="1" selected>1x</option><option value="2">2x</option><option value="5">5x</option><option value="10">10x</option>
      </select>
      <!-- Time slider -->
      <input id="{uid}_slider" type="range" min="0" max="100" value="0" style="flex:1;min-width:120px;accent-color:#3b82f6;">
      <span id="{uid}_day" style="min-width:90px;">Day 0</span>
      <!-- Jump buttons -->
      <span style="color:#64748b;">|</span>
      <button class="{uid}_jmp" data-t="start" style="padding:2px 6px;background:#1e293b;color:#94a3b8;border:1px solid #334155;border-radius:3px;cursor:pointer;font-size:11px;">Start</button>
      <button class="{uid}_jmp" data-t="launch" style="padding:2px 6px;background:#1e293b;color:#94a3b8;border:1px solid #334155;border-radius:3px;cursor:pointer;font-size:11px;">Launch</button>
      <button class="{uid}_jmp" data-t="intercept" style="padding:2px 6px;background:#1e293b;color:#94a3b8;border:1px solid #334155;border-radius:3px;cursor:pointer;font-size:11px;">Intercept</button>
      <button class="{uid}_jmp" data-t="impact" style="padding:2px 6px;background:#1e293b;color:#94a3b8;border:1px solid #334155;border-radius:3px;cursor:pointer;font-size:11px;">Impact</button>
      <button class="{uid}_jmp" data-t="end" style="padding:2px 6px;background:#1e293b;color:#94a3b8;border:1px solid #334155;border-radius:3px;cursor:pointer;font-size:11px;">End</button>
    </div>
    <!-- Camera & toggles -->
    <div style="margin-top:6px;display:flex;flex-wrap:wrap;gap:6px;align-items:center;font-size:12px;color:#cbd5e1;">
      <span style="color:#64748b;">Camera:</span>
      <button class="{uid}_cam" data-b="Sun" style="padding:2px 6px;background:#1e293b;color:#ffcc00;border:1px solid #334155;border-radius:3px;cursor:pointer;font-size:11px;">Sun</button>
      <button class="{uid}_cam" data-b="Earth" style="padding:2px 6px;background:#1e293b;color:#4488ff;border:1px solid #334155;border-radius:3px;cursor:pointer;font-size:11px;">Earth</button>
      <button class="{uid}_cam" data-b="Spacecraft" style="padding:2px 6px;background:#1e293b;color:#44ff44;border:1px solid #334155;border-radius:3px;cursor:pointer;font-size:11px;">Spacecraft</button>
      <button class="{uid}_cam" data-b="Asteroid" style="padding:2px 6px;background:#1e293b;color:#ff4444;border:1px solid #334155;border-radius:3px;cursor:pointer;font-size:11px;">Asteroid</button>
      <button class="{uid}_cam" data-b="overview" style="padding:2px 6px;background:#1e293b;color:#e2e8f0;border:1px solid #334155;border-radius:3px;cursor:pointer;font-size:11px;">Overview</button>
      <span style="color:#64748b;">|</span>
      <label style="cursor:pointer;"><input type="checkbox" class="{uid}_tog" data-g="planets" checked style="accent-color:#3b82f6;"> Planet paths</label>
      <label style="cursor:pointer;"><input type="checkbox" class="{uid}_tog" data-g="ast_orig" checked style="accent-color:#ff8800;"> Asteroid original</label>
      <label style="cursor:pointer;"><input type="checkbox" class="{uid}_tog" data-g="ast_mod" checked style="accent-color:#ff4444;"> Asteroid modified</label>
      <label style="cursor:pointer;"><input type="checkbox" class="{uid}_tog" data-g="sc_path" checked style="accent-color:#44ff44;"> Spacecraft path</label>
      <span style="color:#64748b;">|</span>
      <label style="cursor:pointer;">Scale: <input id="{uid}_scale" type="range" min="0" max="100" value="60" style="width:80px;accent-color:#a78bfa;"></label>
    </div>
    <div id="{uid}_log" style="margin-top:8px;padding:8px;border:1px solid #334155;background:#0f172a;color:#cbd5e1;font-size:12px;border-radius:6px;"></div>
  </div>
  <script src="https://cdn.jsdelivr.net/npm/three@0.132.2/build/three.min.js"></script>
  <script src="https://cdn.jsdelivr.net/npm/three@0.132.2/examples/js/controls/OrbitControls.js"></script>
  <script>
  (function(){{
    const uid='{uid}';
    const D={data_json};
    const PVIS={json.dumps(planet_vis, separators=(',', ':'))};
    const el=document.getElementById(uid);
    if(!el||typeof THREE==='undefined') return;

    // Scene setup
    const scene=new THREE.Scene();
    scene.background=new THREE.Color(0x060612);
    const cam=new THREE.PerspectiveCamera(50,el.clientWidth/el.clientHeight,0.001,2000);
    cam.position.set(3,2,3);
    const renderer=new THREE.WebGLRenderer({{antialias:true}});
    renderer.setSize(el.clientWidth,el.clientHeight);
    el.appendChild(renderer.domElement);
    const OC=THREE.OrbitControls||window.OrbitControls;
    const controls=OC?new OC(cam,renderer.domElement):null;
    if(controls){{controls.enableDamping=true;controls.dampingFactor=0.08;}}

    // Lights
    scene.add(new THREE.AmbientLight(0x303050,0.4));
    const sunLight=new THREE.PointLight(0xffffff,1.5,0);
    scene.add(sunLight);

    // State
    const td=D.time_days;
    const nFrames=td.length;
    function dayToFrame(targetDay){{
      let best=0;
      for(let i=0;i<nFrames;i++){{
        if(Math.abs(td[i]-targetDay)<Math.abs(td[best]-targetDay)) best=i;
      }}
      return best;
    }}
    let curFrame=0;
    let playing=false;
    let speed=1;
    let bodyScale=1.0;
    let followTarget=null;
    const impactDayDef=D.impact_day_deflected;
    const impactDayOrig=D.impact_day_original;
    const impactFrameDef=(impactDayDef!==null&&impactDayDef!==undefined)?dayToFrame(impactDayDef):null;
    const impactFrameOrig=(impactDayOrig!==null&&impactDayOrig!==undefined)?dayToFrame(impactDayOrig):null;

    // Body meshes
    const bodyMeshes={{}};
    const planetNames=D.planet_names;
    // Create planet spheres
    for(const pn of planetNames){{
      const cfg=PVIS[pn]||['888888',0.01];
      const color=parseInt(cfg[0],16);
      const baseR=cfg[1];
      let mat;
      if(pn==='Sun'){{
        mat=new THREE.MeshBasicMaterial({{color:color}});
      }}else{{
        mat=new THREE.MeshStandardMaterial({{color:color,roughness:0.7,metalness:0.1,emissive:color,emissiveIntensity:0.15}});
      }}
      const mesh=new THREE.Mesh(new THREE.SphereGeometry(baseR,16,12),mat);
      mesh.userData.baseR=baseR;
      scene.add(mesh);
      bodyMeshes[pn]=mesh;
    }}
    // Asteroid sphere
    const astMesh=new THREE.Mesh(new THREE.SphereGeometry(0.012,12,8),new THREE.MeshStandardMaterial({{color:0xff4444,emissive:0xff2222,emissiveIntensity:0.3}}));
    astMesh.userData.baseR=0.012;
    scene.add(astMesh);
    bodyMeshes['Asteroid']=astMesh;
    // Spacecraft sphere
    const scMesh=new THREE.Mesh(new THREE.SphereGeometry(0.008,8,6),new THREE.MeshBasicMaterial({{color:0x44ff44}}));
    scMesh.userData.baseR=0.008;
    scene.add(scMesh);
    bodyMeshes['Spacecraft']=scMesh;

    // Path lines
    const pathGroups={{}};
    function makePath(key,positions,color,dashed,maxFrame){{
      if(!positions||positions.length<2) return null;
      const pts=[];
      const limit=(maxFrame!==null&&maxFrame!==undefined)?Math.min(maxFrame,positions.length-1):positions.length-1;
      for(let i=0;i<=limit;i++){{
        const p=positions[i];
        if(p) pts.push(new THREE.Vector3(p[0],p[2],p[1]));
      }}
      if(pts.length<2) return null;
      const geom=new THREE.BufferGeometry().setFromPoints(pts);
      let mat;
      if(dashed){{
        mat=new THREE.LineDashedMaterial({{color:color,dashSize:0.02,gapSize:0.01,transparent:true,opacity:0.5}});
      }}else{{
        mat=new THREE.LineBasicMaterial({{color:color,transparent:true,opacity:0.4}});
      }}
      const line=new THREE.Line(geom,mat);
      if(dashed) line.computeLineDistances();
      scene.add(line);
      return line;
    }}
    // Planet orbit paths (full trajectory)
    const planetPaths=[];
    for(const pn of planetNames){{
      if(pn==='Moon') continue; // too close to Earth to see path
      const line=makePath(pn,D.trajectories[pn],parseInt((PVIS[pn]||['888888'])[0],16),false);
      if(line) planetPaths.push(line);
    }}
    pathGroups['planets']=planetPaths;
    // Asteroid original path
    const astOrigLine=makePath('ast_orig',D.trajectories['Asteroid_Original'],0xff8800,true,impactFrameOrig);
    pathGroups['ast_orig']=astOrigLine?[astOrigLine]:[];
    // Asteroid modified path
    const astModLine=makePath('ast_mod',D.trajectories['Asteroid'],0xff4444,false,impactFrameDef);
    pathGroups['ast_mod']=astModLine?[astModLine]:[];
    // Spacecraft path (filter nulls)
    const scPts=D.trajectories['Spacecraft'].filter(p=>p!==null);
    const scLine=makePath('sc_path',scPts,0x44ff44,false);
    pathGroups['sc_path']=scLine?[scLine]:[];
    // Burn markers (red dots)
    const burnMarkers=[];
    function addBurnMarker(pos){{
      const m=new THREE.Mesh(new THREE.SphereGeometry(0.007,8,6),new THREE.MeshBasicMaterial({{color:0xff3333}}));
      m.userData.baseR=0.007;
      m.position.set(pos[0],pos[2],pos[1]);
      scene.add(m);
      burnMarkers.push(m);
      pathGroups['sc_path'].push(m);
    }}
    const burnEvents=D.burn_events||[];
    for(let i=0;i<burnEvents.length;i++){{
      const day=burnEvents[i][0];
      const frame=dayToFrame(day);
      const spos=D.trajectories['Spacecraft'][frame];
      if(spos) addBurnMarker(spos);
    }}

    // --- Update positions for frame ---
    function setFrame(fi){{
      curFrame=Math.max(0,Math.min(fi,nFrames-1));
      // Planets
      for(const pn of planetNames){{
        const pos=D.trajectories[pn][curFrame];
        if(pos){{
          bodyMeshes[pn].position.set(pos[0],pos[2],pos[1]);
        }}
      }}
      // Sun light follows Sun
      const sunPos=D.trajectories['Sun'][curFrame];
      if(sunPos) sunLight.position.set(sunPos[0],sunPos[2],sunPos[1]);
      // Asteroid
      const apos=D.trajectories['Asteroid'][curFrame];
      const asteroidVisible=(impactFrameDef===null||curFrame<=impactFrameDef);
      if(apos&&asteroidVisible){{
        astMesh.position.set(apos[0],apos[2],apos[1]);
        astMesh.visible=true;
      }}else{{
        astMesh.visible=false;
      }}
      // Spacecraft
      const spos=D.trajectories['Spacecraft'][curFrame];
      if(spos){{
        scMesh.position.set(spos[0],spos[2],spos[1]);
        scMesh.visible=true;
      }}else{{
        scMesh.visible=false;
      }}
      // Update slider & label
      const slider=document.getElementById(uid+'_slider');
      if(slider){{slider.max=nFrames-1;slider.value=curFrame;}}
      const dayLabel=document.getElementById(uid+'_day');
      if(dayLabel) dayLabel.textContent='Day '+td[curFrame];
      // Camera follow
      if(followTarget && bodyMeshes[followTarget]){{
        const m=bodyMeshes[followTarget];
        if(m.visible && controls){{
          controls.target.copy(m.position);
        }}
      }}
    }}

    // --- Scale update ---
    function updateScale(){{
      for(const [name,mesh] of Object.entries(bodyMeshes)){{
        const br=mesh.userData.baseR||0.01;
        const s=br*bodyScale;
        mesh.scale.set(s/br,s/br,s/br);
      }}
      for(const m of burnMarkers){{
        const br=m.userData.baseR||0.007;
        const s=br*bodyScale;
        m.scale.set(s/br,s/br,s/br);
      }}
    }}

    // --- Jump helpers ---
    const impactTarget=(impactFrameDef!==null&&impactFrameDef!==undefined)?impactFrameDef:dayToFrame(D.warning_days);
    const jumpTargets={{
      'start':0,
      'launch':dayToFrame(D.launch_day||0),
      'intercept':D.intercept_day?dayToFrame(D.intercept_day):dayToFrame(D.warning_days),
      'impact':impactTarget,
      'end':nFrames-1
    }};

    // --- Event log ---
    const logEl=document.getElementById(uid+'_log');
    if(logEl){{
      const events=[];
      const burns=D.burn_events||[];
      if(burns.length===0){{
        events.push({{label:'No burns', day:null}});
      }}else{{
        for(let i=0;i<burns.length;i++){{
          const b=burns[i];
          const label=(i===0)?'Launch':'Burn '+i;
          events.push({{label:label, day:b[0], dv:b[1]}});
        }}
      }}
      if(D.intercept_day!==null&&D.intercept_day!==undefined){{
        events.push({{label:'Spacecraft intercepts asteroid', day:D.intercept_day}});
      }}else{{
        events.push({{label:'No intercept', day:null}});
      }}
      if(D.impact_day_original!==null&&D.impact_day_original!==undefined){{
        events.push({{label:'Asteroid impacts Earth (original path)', day:D.impact_day_original}});
      }}else{{
        events.push({{label:'Asteroid misses Earth (original path)', day:null}});
      }}
      let html='<strong style="color:#e2e8f0;">Event Log</strong>';
      html+='<ul style="margin:6px 0 0 16px;padding:0;">';
      for(const ev of events){{
        const dayText=(ev.day!==null&&ev.day!==undefined)?('Day '+ev.day.toFixed(2)):'—';
        const dvText=(ev.dv!==null&&ev.dv!==undefined)?(' | Δv '+ev.dv.toFixed(1)+' m/s'):'';
        html+='<li style="margin:2px 0;">'+ev.label+' — '+dayText+dvText+'</li>';
      }}
      html+='</ul>';
      logEl.innerHTML=html;
    }}

    // --- Controls wiring ---
    const playBtn=document.getElementById(uid+'_play');
    playBtn.addEventListener('click',()=>{{
      playing=!playing;
      playBtn.innerHTML=playing?'&#9646;&#9646; Pause':'&#9654; Play';
    }});
    document.getElementById(uid+'_speed').addEventListener('change',function(){{speed=parseFloat(this.value);}});
    document.getElementById(uid+'_slider').addEventListener('input',function(){{
      setFrame(parseInt(this.value));playing=false;playBtn.innerHTML='&#9654; Play';
    }});
    // Jump buttons
    document.querySelectorAll('.'+uid+'_jmp').forEach(btn=>{{
      btn.addEventListener('click',()=>{{
        const t=btn.dataset.t;
        if(jumpTargets[t]!==undefined) setFrame(jumpTargets[t]);
      }});
    }});
    // Camera buttons
    document.querySelectorAll('.'+uid+'_cam').forEach(btn=>{{
      btn.addEventListener('click',()=>{{
        const b=btn.dataset.b;
        if(b==='overview'){{
          followTarget=null;
          cam.position.set(3,2,3);
          if(controls){{controls.target.set(0,0,0);controls.update();}}
        }}else{{
          followTarget=b;
          const m=bodyMeshes[b];
          if(m&&m.visible&&controls){{
            controls.target.copy(m.position);
            const dir=cam.position.clone().sub(controls.target).normalize();
            cam.position.copy(controls.target).add(dir.multiplyScalar(0.3));
            controls.update();
          }}
        }}
      }});
    }});
    // Path toggles
    document.querySelectorAll('.'+uid+'_tog').forEach(cb=>{{
      cb.addEventListener('change',()=>{{
        const g=cb.dataset.g;
        const lines=pathGroups[g]||[];
        for(const l of lines) l.visible=cb.checked;
      }});
    }});
    // Scale slider
    const scaleSlider=document.getElementById(uid+'_scale');
    scaleSlider.addEventListener('input',()=>{{
      const v=parseInt(scaleSlider.value);
      // Map 0-100 to scale 0.1x to 50x (logarithmic)
      bodyScale=Math.pow(10,(v-50)/25);
      updateScale();
    }});
    // Initialize scale
    bodyScale=Math.pow(10,(60-50)/25);
    updateScale();

    // --- Animation loop ---
    let lastTime=0;
    let accumDt=0;
    const frameDuration=0.05; // seconds per frame at 1x speed
    function animate(t){{
      requestAnimationFrame(animate);
      const dt=(t-lastTime)/1000;
      lastTime=t;
      if(playing&&dt<0.5){{
        accumDt+=dt*speed;
        while(accumDt>=frameDuration){{
          accumDt-=frameDuration;
          if(curFrame<nFrames-1){{
            setFrame(curFrame+1);
          }}else{{
            playing=false;
            playBtn.innerHTML='&#9654; Play';
            break;
          }}
        }}
      }}
      if(controls) controls.update();
      renderer.render(scene,cam);
    }}
    // Resize handler
    let lastW=0,lastH=0;
    function resize(){{
      const w=el.clientWidth,h=el.clientHeight;
      if(w>0&&h>0&&(w!==lastW||h!==lastH)){{
        lastW=w;lastH=h;
        cam.aspect=w/h;cam.updateProjectionMatrix();
        renderer.setSize(w,h,false);
      }}
    }}
    new ResizeObserver(resize).observe(el);
    resize();

    // Set initial frame
    setFrame(0);
    requestAnimationFrame(animate);
  }})();
  </script>
  '''


highLevelSummary = """
<p>Launch an interceptor spacecraft to collide with a dangerous asteroid before it
reaches Earth. The AI must plan a trajectory through the solar system, accounting
for the gravitational pull of the Sun and planets, to hit a moving target millions
of kilometres away.</p>
<p>Harder subpasses give less warning time or tighter fuel budgets, potentially
requiring planetary gravity assists (slingshots) to reach the asteroid. The AI
must implement orbital mechanics and trajectory optimisation from scratch.</p>
"""

if __name__ == "__main__":
  print("=== Ephemeris-Backed N-Body Validation ===")
  print(f"Mode: Planets from DE421 ephemeris, test particles integrated with leapfrog KDK")
  print(f"Bodies: {len(BODY_CONFIG)} ephemeris + dynamic test particles")
  print(f"Epoch: J2000.0 (JD {J2000_JD}), dt=3600s")
  print()

  # --- Test 1: Verify planet positions are exact (ephemeris-backed) ---
  print("--- Test 1: Planet position accuracy (should be ~0) ---")
  sim = NBodySimulation(RANDOM_SEED)
  dur = 365 * 30 * DAY  # 30 years
  dt = 3600.0
  t0 = time.time()
  sim.precompute_ephemeris(dur + DAY, dt)
  sim.run(dur, dt)
  wall_time = time.time() - t0
  print(f"  30-year simulation: {wall_time:.2f}s wall time")

  end_jd = J2000_JD + 365 * 30
  max_err_au = 0.0
  for name, sf_name, mass in BODY_CONFIG:
    body = sim.get_body(name)
    if not body:
      continue
    sx, sy, sz, _, _, _ = _get_body_state_at_jd(sf_name, end_jd)
    pos_err = math.sqrt((body.x - sx)**2 + (body.y - sy)**2 + (body.z - sz)**2)
    err_au = pos_err / AU
    print(f"  {name:10s}: pos err {err_au:.10f} AU ({pos_err:.2f} m)")
    max_err_au = max(max_err_au, err_au)

  print(f"  Max planet error: {max_err_au:.10f} AU")
  if max_err_au > 1e-6:
    print("  *** FAIL: Planet positions should be near-zero error! ***")
    sys.exit(1)
  print("  PASS: Planet positions match ephemeris exactly.")
  print()

  # --- Test 2: Asteroid hits Earth in undeflected simulation ---
  print("--- Test 2: Asteroid impact verification (no spacecraft) ---")
  earth_radius = 6.371e6  # metres
  all_pass = True
  # Test a few subpasses with varying warning times
  test_subpasses = [0, 1, 3, 5]
  for sp in test_subpasses:
    case = TEST_CASES[sp]
    warning_days = case["warning_days"]
    warning_time = warning_days * DAY
    t0 = time.time()

    # Clear cache so we regenerate
    SCENARIO_CACHE.pop(sp, None)
    test_sim_obj, _ = get_scenario(sp)

    # Create fresh sim and forward-integrate (no spacecraft)
    fwd = NBodySimulation(test_sim_obj.rng.randint(0, 1000000), epoch_jd=test_sim_obj.epoch_jd)
    fwd.precompute_ephemeris(warning_time + 200 * DAY, dt)
    orig_ast = test_sim_obj.get_body("Asteroid")
    fwd.add_asteroid(orig_ast.x, orig_ast.y, orig_ast.z,
                     orig_ast.vx, orig_ast.vy, orig_ast.vz, orig_ast.mass)

    # Track minimum distance to Earth over the mission
    min_dist = float('inf')
    min_dist_day = 0
    total_sim_days = warning_days + 100
    for d in range(total_sim_days):
      fwd.run(DAY, dt)
      earth_b = fwd.get_earth()
      ast_b = fwd.get_body("Asteroid")
      if earth_b and ast_b:
        dist = math.sqrt((ast_b.x - earth_b.x)**2 + (ast_b.y - earth_b.y)**2 + (ast_b.z - earth_b.z)**2)
        if dist < min_dist:
          min_dist = dist
          min_dist_day = d + 1

    wall_time = time.time() - t0
    hit = min_dist < earth_radius * 2  # within 2 Earth radii counts as impact
    status = "HIT" if hit else "MISS"
    if not hit:
      all_pass = False
    print(f"  Subpass {sp} ({warning_days:5d}d): closest = {min_dist/1e3:.0f} km on day {min_dist_day} "
          f"({min_dist/earth_radius:.2f} R_earth) [{status}] {wall_time:.1f}s")

  if all_pass:
    print("  PASS: All tested subpasses produce Earth impact.")
  else:
    print("  *** FAIL: Some asteroids missed Earth! ***")
    sys.exit(1)
  print()

  # --- Test 3: Deflection signal detection ---
  print("--- Test 3: Deflection signal over increasing durations ---")
  for dur_days, label in [(365, "1 year"), (365*5, "5 years"), (365*10, "10 years"), (365*30, "30 years")]:
    duration = dur_days * DAY

    # Sim A: undeflected asteroid
    sim_a = NBodySimulation(RANDOM_SEED)
    sim_a.precompute_ephemeris(duration + DAY, dt)
    ax = 2.0 * AU
    ay = 0.0
    az = 0.0
    v_circ = math.sqrt(G * 1.98892e30 / (2.0 * AU))
    avx = 0.0
    avy = v_circ
    avz = 0.0
    sim_a.add_asteroid(ax, ay, az, avx, avy, avz, 1e10)

    # Sim B: deflected asteroid (tiny velocity nudge)
    sim_b = NBodySimulation(RANDOM_SEED)
    sim_b.precompute_ephemeris(duration + DAY, dt)
    nudge = 0.01  # 1 cm/s nudge
    sim_b.add_asteroid(ax, ay, az, avx + nudge, avy, avz, 1e10)

    t0 = time.time()
    sim_a.run(duration, dt)
    sim_b.run(duration, dt)
    wall_time = time.time() - t0

    ast_a = sim_a.get_body("Asteroid")
    ast_b = sim_b.get_body("Asteroid")
    sep = math.sqrt((ast_a.x - ast_b.x)**2 + (ast_a.y - ast_b.y)**2 + (ast_a.z - ast_b.z)**2)
    sep_au = sep / AU

    print(f"  {label:10s}: deflection = {sep_au:.6f} AU ({sep/1e3:.1f} km), wall time {wall_time:.1f}s")

  print()
  print("=== Validation complete ===")
