"""
Test 6: Orbital Rendezvous Travelling Salesman Problem

The LLM must write a Python solver that plans an optimal route to visit
multiple space stations in Earth orbit, minimizing total delta-V (fuel).

This combines TSP with orbital mechanics:
- Stations orbit Earth following Kepler's laws
- Transfer costs depend on orbital positions and timing
- Must account for orbital dynamics, not just distances

Subpasses increase the number of stations to visit.
Solver times out after 5 minutes.
"""

import math
import time
import random
import json
import uuid
from typing import List, Tuple, Dict, Optional

from native_compiler import CSharpCompiler, compile_and_run, describe_this_pc

title = "Orbital TSP - Space Station Route Planning (C#)"

# Timeout in seconds (5 minutes)
TIMEOUT_SECONDS = 30

# Standard gravitational parameter for Earth (km^3/s^2)
MU = 398600.4418
EARTH_RADIUS = 6371  # km


# Orbit data: [Pos X, Y, Z, Vel X, Y, Z] at epoch T=0 (km, km/s)
def _kepler_to_state(mu: float, a_km: float, e: float, i_rad: float, raan_rad: float,
                     argp_rad: float, nu_rad: float) -> List[float]:
  """Convert Keplerian elements to ECI state vector.

  Elements:
  - a_km: semi-major axis (km)
  - e: eccentricity
  - i_rad: inclination (rad)
  - raan_rad: RAAN (rad)
  - argp_rad: argument of periapsis (rad)
  - nu_rad: true anomaly (rad)
  """
  p = a_km * (1.0 - e * e)
  r_pf = p / (1.0 + e * math.cos(nu_rad))

  # Perifocal position/velocity
  x_pf = r_pf * math.cos(nu_rad)
  y_pf = r_pf * math.sin(nu_rad)
  z_pf = 0.0

  v_pf_scale = math.sqrt(mu / p)
  vx_pf = -v_pf_scale * math.sin(nu_rad)
  vy_pf = v_pf_scale * (e + math.cos(nu_rad))
  vz_pf = 0.0

  # Rotation matrix from perifocal to ECI: R3(raan) * R1(i) * R3(argp)
  cO, sO = math.cos(raan_rad), math.sin(raan_rad)
  ci, si = math.cos(i_rad), math.sin(i_rad)
  co, so = math.cos(argp_rad), math.sin(argp_rad)

  r11 = cO * co - sO * so * ci
  r12 = -cO * so - sO * co * ci
  r13 = sO * si
  r21 = sO * co + cO * so * ci
  r22 = -sO * so + cO * co * ci
  r23 = -cO * si
  r31 = so * si
  r32 = co * si
  r33 = ci

  x = r11 * x_pf + r12 * y_pf + r13 * z_pf
  y = r21 * x_pf + r22 * y_pf + r23 * z_pf
  z = r31 * x_pf + r32 * y_pf + r33 * z_pf

  vx = r11 * vx_pf + r12 * vy_pf + r13 * vz_pf
  vy = r21 * vx_pf + r22 * vy_pf + r23 * vz_pf
  vz = r31 * vx_pf + r32 * vy_pf + r33 * vz_pf

  return [x, y, z, vx, vy, vz]


def generate_random_orbit(rng: random.Random,
                          *,
                          min_perigee_alt_km: float = 600.0,
                          max_apogee_alt_km: float = 80000.0) -> List[float]:
  """Generate a random long-lived Earth orbit as a state vector.

  Constraints:
  - Perigee altitude >= min_perigee_alt_km (avoids atmospheric drag for geologic times)
  - Apogee altitude <= max_apogee_alt_km (well below the Moon, avoids lunar dynamics)
  """
  # Choose inclination including retrograde
  i_deg = rng.uniform(0.0, 180.0)
  i = math.radians(i_deg)

  raan = rng.uniform(0.0, 2.0 * math.pi)
  argp = rng.uniform(0.0, 2.0 * math.pi)
  nu = rng.uniform(0.0, 2.0 * math.pi)

  # Semi-major axis baseline from altitude
  # Spread: LEO -> MEO-ish, avoid GEO-synchronous special-casing but allow it.
  base_alt_km = rng.uniform(700.0, 40000.0)

  # Eccentricity spread: near-circular to moderately elliptical
  e = rng.uniform(0.0, 0.6)

  rp_min = EARTH_RADIUS + float(min_perigee_alt_km)
  ra_max = EARTH_RADIUS + float(max_apogee_alt_km)

  # Solve for a such that constraints hold: rp = a(1-e) and ra = a(1+e)
  # Start with base altitude and adjust if needed.
  a = EARTH_RADIUS + base_alt_km
  a = max(a, rp_min / max(1e-6, (1.0 - e)))
  a = min(a, ra_max / max(1e-6, (1.0 + e)))

  # If constraints still infeasible (high e with low max apogee), reduce e.
  # (rare but possible if we clipped a hard)
  if a * (1.0 - e) < rp_min or a * (1.0 + e) > ra_max:
    e = min(e, 0.3)
    a = max(a, rp_min / (1.0 - e))
    a = min(a, ra_max / (1.0 + e))

  # Final sanity clamps
  rp = a * (1.0 - e)
  ra = a * (1.0 + e)
  if rp < rp_min:
    a = rp_min / (1.0 - e)
  if ra > ra_max:
    a = ra_max / (1.0 + e)

  return _kepler_to_state(MU, a, e, i, raan, argp, nu)


def generate_random_orbits(seed: int, n: int) -> Tuple[List[float], List[List[float]]]:
  """Generate a start orbit and N station orbits deterministically from seed."""
  rng = random.Random(int(seed))
  start_orbit = generate_random_orbit(
    rng,
    min_perigee_alt_km=700.0,
    max_apogee_alt_km=20000.0,
  )
  orbits = [generate_random_orbit(rng) for _ in range(n)]
  return start_orbit, orbits


# Default dataset (seeded for repeatability). Hardest test requires 100 stations.
START_ORBIT, ORBITS = generate_random_orbits(seed=6, n=100)

# Number of stations for each subpass
STATION_COUNTS = [2, 3, 4, 5, 6, 8, 10, 15, 20, 50, 100]

_DATASET_CACHE: Dict[int, Tuple[List[float], List[List[float]]]] = {}


def _get_dataset_for_subpass(subpass: int) -> Tuple[List[float], List[List[float]]]:
  """Return (start_orbit, orbits) for a given subpass.

  This makes each difficulty level start from a different (but deterministic) orbit
  and uses a different station set.
  """
  seed = 6000 + int(subpass)
  if seed in _DATASET_CACHE:
    return _DATASET_CACHE[seed]
  ds = generate_random_orbits(seed=seed, n=100)
  _DATASET_CACHE[seed] = ds
  return ds


def get_orbit_info(orbit: List[float]) -> Dict:
  """Calculate orbital parameters from state vector."""
  x, y, z, vx, vy, vz = orbit
  r = math.sqrt(x**2 + y**2 + z**2)
  v = math.sqrt(vx**2 + vy**2 + vz**2)

  # Specific orbital energy
  energy = v**2 / 2 - MU / r

  # Semi-major axis
  if abs(energy) > 1e-10:
    a = -MU / (2 * energy)
  else:
    a = float('inf')

  # Orbital period
  if a > 0 and a != float('inf'):
    period = 2 * math.pi * math.sqrt(a**3 / MU)
  else:
    period = float('inf')

  altitude = r - EARTH_RADIUS

  return {
    "altitude_km": altitude,
    "semi_major_axis_km": a,
    "period_seconds": period,
    "velocity_km_s": v
  }


def format_orbits_for_prompt(num_stations: int) -> str:
  """Format orbit data for the prompt."""
  lines = []
  for i in range(num_stations):
    orbit = ORBITS[i]
    info = get_orbit_info(orbit)
    lines.append(f"Station {i}: pos=({orbit[0]:.1f}, {orbit[1]:.1f}, {orbit[2]:.1f}) km, "
                 f"vel=({orbit[3]:.3f}, {orbit[4]:.3f}, {orbit[5]:.3f}) km/s, "
                 f"altitude≈{info['altitude_km']:.0f}km")
  return "\n".join(lines)


def prepareSubpassPrompt(subPass: int) -> str:
  """Generate the prompt for subpass 0 that handles all station counts."""
  if subPass != 0:
    raise StopIteration

  start_info = get_orbit_info(START_ORBIT)

  return f"""You are solving an Orbital Travelling Salesman Problem in C#.

You must write a C# solver that can handle ANY number of stations from trivial to ludicrous scale:
- **Trivial**: 2-5 stations (simple brute-force permutations feasible)
- **Medium**: 6-10 stations (requires heuristics, some optimization)
- **Large**: 15-20 stations (complex orbital mechanics, needs efficient algorithms)
- **Extreme**: 50-100 stations (very complex, requires advanced heuristics)

**The Challenge:**
Your program will be tested with 2 to 100 stations. The same algorithm must work efficiently across ALL scales.

**Input format (stdin):**
Line 1: N mu
Line 2: start orbit: x y z vx vy vz
Lines 3..N+2: station orbits: x y z vx vy vz

**Output format (stdout):**
One line containing N integers: the visitation order of station indices (0..N-1).


**Starting position (epoch T=0):**
Position: ({START_ORBIT[0]:.1f}, {START_ORBIT[1]:.1f}, {START_ORBIT[2]:.1f}) km
Velocity: ({START_ORBIT[3]:.3f}, {START_ORBIT[4]:.3f}, {START_ORBIT[5]:.3f}) km/s
Altitude: ~{start_info['altitude_km']:.0f} km

**Physics constants:**
- Earth gravitational parameter μ = {MU} km³/s²
- Earth radius = {EARTH_RADIUS} km

**Critical Requirements:**
1. **Scalability**: Your algorithm must adapt based on number of stations
2. **Performance**: Must complete within 5 minutes even for 100 stations
3. **Accuracy**: Should provide reasonable delta-V estimates

**Environment:**
{describe_this_pc()}

**C# Compiler:**
{CSharpCompiler("test_engine").describe()}

Write complete, compilable C# code with a static void Main method.
"""


# List of subpasses to grade the single answer against all difficulty levels
extraGradeAnswerRuns = list(range(len(STATION_COUNTS)))

structure = {
  "type": "object",
  "properties": {
    "reasoning": {
      "type":
      "string",
      "description":
      "Explain your orbital mechanics approach and how it adapts to different station counts"
    },
    "csharp_code": {
      "type": "string",
      "description": "Complete C# code with Main method that handles all scales"
    }
  },
  "required": ["reasoning", "csharp_code"],
  "additionalProperties": False
}


def estimate_transfer_delta_v(orbit1: List[float], orbit2: List[float]) -> float:
  """
  Estimate delta-V for transfer between two orbits.
  Properly accounts for plane changes which dominate delta-V requirements.
  """
  # Extract orbital parameters
  r1_vec = orbit1[:3]
  r2_vec = orbit2[:3]
  v1_vec = orbit1[3:6]
  v2_vec = orbit2[3:6]

  # Calculate orbital parameters
  r1 = math.sqrt(r1_vec[0]**2 + r1_vec[1]**2 + r1_vec[2]**2)
  r2 = math.sqrt(r2_vec[0]**2 + r2_vec[1]**2 + r2_vec[2]**2)
  v1 = math.sqrt(v1_vec[0]**2 + v1_vec[1]**2 + v1_vec[2]**2)
  v2 = math.sqrt(v2_vec[0]**2 + v2_vec[1]**2 + v2_vec[2]**2)

  # Calculate orbital angular momentum vectors (normal to orbital planes)
  def cross_product(a, b):
    return [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]

  h1_vec = cross_product(r1_vec, v1_vec)
  h2_vec = cross_product(r2_vec, v2_vec)

  # Calculate orbital inclinations (angle between orbital planes)
  h1_mag = math.sqrt(h1_vec[0]**2 + h1_vec[1]**2 + h1_vec[2]**2)
  h2_mag = math.sqrt(h2_vec[0]**2 + h2_vec[1]**2 + h2_vec[2]**2)

  if h1_mag > 0 and h2_mag > 0:
    # Normalize angular momentum vectors
    h1_norm = [h1_vec[0] / h1_mag, h1_vec[1] / h1_mag, h1_vec[2] / h1_mag]
    h2_norm = [h2_vec[0] / h2_mag, h2_vec[1] / h2_mag, h2_vec[2] / h2_mag]

    # Calculate plane change angle (angle between orbital planes)
    cos_plane_change = (h1_norm[0] * h2_norm[0] + h1_norm[1] * h2_norm[1] + h1_norm[2] * h2_norm[2])
    cos_plane_change = max(-1, min(1, cos_plane_change))  # Clamp to valid range
    plane_change_angle = math.acos(cos_plane_change)
  else:
    plane_change_angle = 0

  # Hohmann transfer for in-plane component
  a_transfer = (r1 + r2) / 2
  if a_transfer > 0:
    v_t1 = math.sqrt(MU * (2 / r1 - 1 / a_transfer))
    v_t2 = math.sqrt(MU * (2 / r2 - 1 / a_transfer))
    v_c1 = math.sqrt(MU / r1)
    v_c2 = math.sqrt(MU / r2)

    # In-plane delta-V requirements
    dv1 = abs(v_t1 - v_c1)
    dv2 = abs(v_c2 - v_t2)
    in_plane_dv = dv1 + dv2
  else:
    # Fallback for edge cases
    in_plane_dv = abs(v2 - v1)

  # Plane change delta-V (most expensive part)
  # Plane changes are typically done at the node where orbits intersect
  # The cost depends on the velocity at that point
  v_at_node = math.sqrt(MU * (2 / r1 + 2 / r2) / 2)  # Approximate velocity at node
  plane_change_dv = 2 * v_at_node * math.sin(plane_change_angle / 2)

  # Total delta-V is combination of in-plane and plane change maneuvers
  # In reality, these can be combined, but for estimation we sum them
  total_dv = in_plane_dv + plane_change_dv

  # Add penalty for large velocity magnitude differences (captures other effects)
  vel_magnitude_diff = abs(v2 - v1)
  velocity_penalty = vel_magnitude_diff * 0.1

  return total_dv + velocity_penalty


def get_orbit_characteristics(orbit: List[float]) -> Dict:
  """Extract key orbital characteristics for clustering."""
  x, y, z, vx, vy, vz = orbit
  r = math.sqrt(x**2 + y**2 + z**2)
  v = math.sqrt(vx**2 + vy**2 + vz**2)

  # Calculate orbital inclination (angle between orbital plane and equatorial plane)
  # Angular momentum vector (normal to orbital plane)
  h_vec = [y * vz - z * vy, z * vx - x * vz, x * vy - y * vx]
  h_mag = math.sqrt(h_vec[0]**2 + h_vec[1]**2 + h_vec[2]**2)

  if h_mag > 0:
    # Inclination is angle between angular momentum and z-axis (Earth's rotation axis)
    cos_inclination = h_vec[2] / h_mag
    cos_inclination = max(-1, min(1, cos_inclination))
    inclination = math.degrees(math.acos(cos_inclination))

    # Determine if retrograde (inclination > 90°)
    is_retrograde = inclination > 90

    # For retrograde orbits, use retrograde inclination (180° - inclination) for clustering
    # This groups retrograde orbits together by their "retrograde-ness"
    if is_retrograde:
      retrograde_inclination = 180 - inclination
      effective_inclination = retrograde_inclination
    else:
      effective_inclination = inclination
  else:
    inclination = 0
    is_retrograde = False
    effective_inclination = 0

  # Semi-major axis (approximate)
  energy = v**2 / 2 - MU / r
  if abs(energy) > 1e-10:
    a = -MU / (2 * energy)
  else:
    a = r

  return {
    'altitude': r - EARTH_RADIUS,
    'inclination': inclination,
    'effective_inclination': effective_inclination,  # For clustering prograde/retrograde separately
    'is_retrograde': is_retrograde,
    'semi_major_axis': a,
    'radius': r
  }


def cluster_orbits(station_orbits: List[List[float]]) -> List[List[int]]:
  """Group orbits by similar altitude and inclination, separating prograde/retrograde."""
  if not station_orbits:
    return []

  # Extract characteristics
  characteristics = []
  for i, orbit in enumerate(station_orbits):
    char = get_orbit_characteristics(orbit)
    characteristics.append((i, char))

  # Separate into prograde and retrograde groups first
  prograde_orbits = [(i, char) for i, char in characteristics if not char['is_retrograde']]
  retrograde_orbits = [(i, char) for i, char in characteristics if char['is_retrograde']]

  clusters = []

  # Cluster prograde orbits
  if prograde_orbits:
    prograde_orbits.sort(key=lambda x: x[1]['altitude'])
    prograde_clusters = []
    current_cluster = []
    last_altitude = None
    last_inclination = None

    for station_idx, char in prograde_orbits:
      if (last_altitude is None or abs(char['altitude'] - last_altitude) > 500
          or abs(char['effective_inclination'] - last_inclination) > 10):

        # Start new cluster
        if current_cluster:
          prograde_clusters.append(current_cluster)
        current_cluster = [station_idx]
      else:
        # Add to current cluster
        current_cluster.append(station_idx)

      last_altitude = char['altitude']
      last_inclination = char['effective_inclination']

    # Add final prograde cluster
    if current_cluster:
      prograde_clusters.append(current_cluster)

    clusters.extend(prograde_clusters)

  # Cluster retrograde orbits separately
  if retrograde_orbits:
    retrograde_orbits.sort(key=lambda x: x[1]['altitude'])
    retrograde_clusters = []
    current_cluster = []
    last_altitude = None
    last_inclination = None

    for station_idx, char in retrograde_orbits:
      if (last_altitude is None or abs(char['altitude'] - last_altitude) > 500
          or abs(char['effective_inclination'] - last_inclination) > 10):

        # Start new cluster
        if current_cluster:
          retrograde_clusters.append(current_cluster)
        current_cluster = [station_idx]
      else:
        # Add to current cluster
        current_cluster.append(station_idx)

      last_altitude = char['altitude']
      last_inclination = char['effective_inclination']

    # Add final retrograde cluster
    if current_cluster:
      retrograde_clusters.append(current_cluster)

    clusters.extend(retrograde_clusters)

  return clusters


def get_baseline_solution(num_stations: int, start_orbit: List[float],
                          orbits: List[List[float]]) -> Tuple[List[int], float]:
  """
  Get baseline solution using intelligent orbital clustering.
  Groups similar orbits together, then finds optimal sequence between clusters.
  """
  station_orbits = orbits[:num_stations]

  if num_stations <= 2:
    # For small numbers, just try all permutations
    if num_stations == 1:
      return [0], estimate_transfer_delta_v(start_orbit, station_orbits[0])
    elif num_stations == 2:
      # Try both orders
      dv1 = estimate_transfer_delta_v(start_orbit, station_orbits[0]) + \
            estimate_transfer_delta_v(station_orbits[0], station_orbits[1])
      dv2 = estimate_transfer_delta_v(start_orbit, station_orbits[1]) + \
            estimate_transfer_delta_v(station_orbits[1], station_orbits[0])
      if dv1 <= dv2:
        return [0, 1], dv1
      else:
        return [1, 0], dv2

  # Cluster orbits by similarity
  clusters = cluster_orbits(station_orbits)

  # Find optimal order to visit clusters
  # Use simple heuristic: start with cluster closest to start, then move to next closest
  remaining_clusters = clusters.copy()
  cluster_order = []
  current_orbit = start_orbit

  while remaining_clusters:
    best_cluster = None
    best_cost = float('inf')

    # Find closest cluster to current position
    for cluster in remaining_clusters:
      # Find best representative station in this cluster
      for station_idx in cluster:
        cost = estimate_transfer_delta_v(current_orbit, station_orbits[station_idx])
        if cost < best_cost:
          best_cost = cost
          best_cluster = cluster

    cluster_order.append(best_cluster)
    remaining_clusters.remove(best_cluster)

    # Set current position to the last station in this cluster
    current_orbit = station_orbits[best_cluster[-1]]

  # Now optimize within each cluster (visit stations in optimal order)
  final_order = []
  total_dv = 0
  current_orbit = start_orbit

  for cluster in cluster_order:
    if len(cluster) == 1:
      # Single station - just visit it
      station_idx = cluster[0]
      dv = estimate_transfer_delta_v(current_orbit, station_orbits[station_idx])
      total_dv += dv
      final_order.append(station_idx)
      current_orbit = station_orbits[station_idx]
    else:
      # Multiple stations - find optimal order within cluster
      # Since they're similar, use simple greedy
      cluster_remaining = cluster.copy()
      while cluster_remaining:
        best_station = None
        best_dv = float('inf')

        for station_idx in cluster_remaining:
          dv = estimate_transfer_delta_v(current_orbit, station_orbits[station_idx])
          if dv < best_dv:
            best_dv = dv
            best_station = station_idx

        total_dv += best_dv
        final_order.append(best_station)
        cluster_remaining.remove(best_station)
        current_orbit = station_orbits[best_station]

  return final_order, total_dv


def format_input(num_stations: int, start_orbit: List[float], orbits: List[List[float]]) -> str:
  lines = [f"{num_stations} {MU}"]
  lines.append(" ".join(f"{v:.6f}" for v in start_orbit))
  for orbit in orbits[:num_stations]:
    lines.append(" ".join(f"{v:.6f}" for v in orbit))
  return "\n".join(lines)


def parse_order_output(output: str, expected_n: int) -> tuple:
  tokens = output.strip().split()
  if not tokens:
    return None, "Empty output"

  try:
    values = [int(t) for t in tokens]
  except ValueError:
    return None, "Output contains non-integer tokens"

  if len(values) == expected_n + 1 and values[0] == expected_n:
    values = values[1:]

  if len(values) != expected_n:
    return None, f"Expected {expected_n} indices, got {len(values)}"

  return values, ""


def execute_solver(code: str,
                   num_stations: int,
                   start_orbit: List[float],
                   orbits: List[List[float]],
                   timeout: int = TIMEOUT_SECONDS) -> tuple:
  """Execute the LLM's solver. Returns (order, error, exec_time)."""
  input_data = format_input(num_stations, start_orbit, orbits)
  run = compile_and_run(code, "csharp", "test_engine", input_data=input_data, timeout=timeout)

  if not run:
    return None, run.error_message(), run.exec_time

  order, parse_error = parse_order_output(run.stdout, num_stations)
  if parse_error:
    return None, parse_error, run.exec_time

  return order, None, run.exec_time


def validate_solution(order: List[int], num_stations: int) -> Tuple[bool, str]:
  """Validate the visitation order format."""
  if not isinstance(order, list):
    return False, "Order must be a list"
  if not isinstance(order, list):
    return False, "visit_order must be a list"

  if len(order) != num_stations:
    return False, f"visit_order must have {num_stations} stations, got {len(order)}"

  if set(order) != set(range(num_stations)):
    return False, f"visit_order must contain each station 0 to {num_stations-1} exactly once"

  return True, ""


def evaluate_order(order: List[int], start_orbit: List[float], orbits: List[List[float]]) -> float:
  """Evaluate the delta-V for a given visitation order."""
  current_orbit = start_orbit
  total_dv = 0

  for station in order:
    dv = estimate_transfer_delta_v(current_orbit, orbits[station])
    total_dv += dv
    current_orbit = orbits[station]

  return total_dv


def gradeAnswer(result: dict, subPass: int, aiEngineName: str) -> tuple:
  """
    Grade the orbital TSP solver.
    
    Scoring based on delta-V vs baseline:
    - 1.0: <= baseline delta-V (found good or better solution)
    - 0.85: <= 1.2x baseline
    - 0.7: <= 1.5x baseline
    - 0.5: Valid solution but poor
    - 0.0: Invalid or error
    """
  if not result:
    return 0.0, "No result provided"

  if "csharp_code" not in result:
    return 0.0, "No C# code provided"

  num_stations = STATION_COUNTS[subPass]
  code = result["csharp_code"]

  start_orbit, orbits = _get_dataset_for_subpass(subPass)

  # Execute solver
  order, error, exec_time = execute_solver(code, num_stations, start_orbit, orbits)

  if error:
    return 0.0, f"[{num_stations} stations] {error}"

  # Validate solution
  is_valid, validation_error = validate_solution(order, num_stations)
  if not is_valid:
    return 0.0, f"[{num_stations} stations] Invalid: {validation_error}"

  # Evaluate the order
  solution_dv = evaluate_order(order, start_orbit, orbits)

  # Get baseline
  baseline_order, baseline_dv = get_baseline_solution(num_stations, start_orbit, orbits)

  # Score
  ratio = solution_dv / baseline_dv if baseline_dv > 0 else float('inf')

  if ratio <= 1.01:
    score = 1.0
    quality = "excellent (≤ baseline)"
  elif ratio <= 1.2:
    score = 0.2
    quality = "good (≤ 1.2x baseline)"
  elif ratio <= 1.5:
    score = 0.1
    quality = "acceptable (≤ 1.5x baseline)"
  else:
    score = 0.0
    quality = f"poor ({ratio:.1f}x baseline)"

  explanation = (f"[{num_stations} stations] Order: {order}, "
                 f"Delta-V: {solution_dv:.2f} km/s, Baseline: {baseline_dv:.2f} km/s, "
                 f"Time: {exec_time:.1f}s - {quality}")

  return score, explanation


def _state_to_elements(sv):
  """Convert state vector [x,y,z,vx,vy,vz] (km, km/s) to orbital elements (a, e, i, raan, argp)."""
  rx, ry, rz = sv[0], sv[1], sv[2]
  vx, vy, vz = sv[3], sv[4], sv[5]

  r = math.sqrt(rx**2 + ry**2 + rz**2)
  v = math.sqrt(vx**2 + vy**2 + vz**2)

  # Angular momentum h = r x v
  hx = ry * vz - rz * vy
  hy = rz * vx - rx * vz
  hz = rx * vy - ry * vx
  h = math.sqrt(hx**2 + hy**2 + hz**2)

  # Node vector n = k x h = [-hy, hx, 0]
  nx, ny = -hy, hx
  n = math.sqrt(nx**2 + ny**2)

  # Eccentricity vector: e = (v x h)/mu - r_hat
  vhx = vy * hz - vz * hy
  vhy = vz * hx - vx * hz
  vhz = vx * hy - vy * hx
  ex = vhx / MU - rx / r
  ey = vhy / MU - ry / r
  ez = vhz / MU - rz / r
  e = math.sqrt(ex**2 + ey**2 + ez**2)

  # Semi-major axis
  energy = v**2 / 2 - MU / r
  if abs(energy) > 1e-10:
    a = -MU / (2 * energy)
  else:
    a = r

  # Inclination
  inc = math.acos(max(-1, min(1, hz / h)))

  # RAAN
  if n > 1e-10:
    raan = math.acos(max(-1, min(1, nx / n)))
    if ny < 0:
      raan = 2 * math.pi - raan
  else:
    raan = 0.0

  # Argument of periapsis
  if n > 1e-10 and e > 1e-10:
    dot_ne = nx * ex + ny * ey
    cos_argp = dot_ne / (n * e)
    argp = math.acos(max(-1, min(1, cos_argp)))
    if ez < 0:
      argp = 2 * math.pi - argp
  else:
    argp = 0.0

  return a, e, inc, raan, argp


def _orbit_ring_points(a, e, inc, raan, argp, n_pts=64, scale=1.0):
  """Sample n_pts points around an orbit ellipse, scaled by scale factor."""
  if e >= 1.0:
    e = 0.99
  p = a * (1.0 - e * e)
  if p <= 0:
    return []

  cO, sO = math.cos(raan), math.sin(raan)
  ci, si = math.cos(inc), math.sin(inc)
  co, so = math.cos(argp), math.sin(argp)

  r11 = cO * co - sO * so * ci
  r12 = -cO * so - sO * co * ci
  r21 = sO * co + cO * so * ci
  r22 = -sO * so + cO * co * ci
  r31 = so * si
  r32 = co * si

  points = []
  for k in range(n_pts + 1):
    nu = 2.0 * math.pi * k / n_pts
    denom = 1.0 + e * math.cos(nu)
    if denom <= 0:
      continue
    r_pf = p / denom
    x_pf = r_pf * math.cos(nu)
    y_pf = r_pf * math.sin(nu)

    x = (r11 * x_pf + r12 * y_pf) * scale
    y = (r21 * x_pf + r22 * y_pf) * scale
    z = (r31 * x_pf + r32 * y_pf) * scale
    points.append([round(x, 4), round(y, 4), round(z, 4)])
  return points


def _ring_point_nearest_dir(ring, direction):
  """Find index of ring point most aligned with direction. Returns (index, point)."""
  best_dot = -float('inf')
  best_idx = 0
  for i, p in enumerate(ring):
    r = math.sqrt(p[0] * p[0] + p[1] * p[1] + p[2] * p[2])
    if r > 0:
      dot = (p[0] * direction[0] + p[1] * direction[1] + p[2] * direction[2]) / r
      if dot > best_dot:
        best_dot = dot
        best_idx = i
  return best_idx, ring[best_idx]


def _compute_transfer_arc(dep_ring, arr_ring, sv_dep, sv_arr, n_pts=100):
  """Compute a Hohmann transfer arc between two orbit rings.
  
  Departure/arrival are chosen at the line of nodes (h_dep × h_arr) so that
  orbit tangent directions are naturally compatible for 180° sweep.
  Uses Hermite spline for direction + cosine radius interpolation.
  
  Returns:
    list of [x,y,z] arc points (already in scaled coordinates)
  """
  if not dep_ring or not arr_ring or len(dep_ring) < 3 or len(arr_ring) < 3:
    return []

  # Angular momenta h = r × v (in km, unscaled)
  h_dep = [
    sv_dep[1] * sv_dep[5] - sv_dep[2] * sv_dep[4], sv_dep[2] * sv_dep[3] - sv_dep[0] * sv_dep[5],
    sv_dep[0] * sv_dep[4] - sv_dep[1] * sv_dep[3]
  ]
  h_arr = [
    sv_arr[1] * sv_arr[5] - sv_arr[2] * sv_arr[4], sv_arr[2] * sv_arr[3] - sv_arr[0] * sv_arr[5],
    sv_arr[0] * sv_arr[4] - sv_arr[1] * sv_arr[3]
  ]

  # Line of nodes = h_dep × h_arr
  node = [
    h_dep[1] * h_arr[2] - h_dep[2] * h_arr[1], h_dep[2] * h_arr[0] - h_dep[0] * h_arr[2],
    h_dep[0] * h_arr[1] - h_dep[1] * h_arr[0]
  ]
  node_mag = math.sqrt(sum(x * x for x in node))

  if node_mag < 1e-10:
    # Coplanar orbits: use satellite position as departure direction
    node = [sv_dep[0], sv_dep[1], sv_dep[2]]
    node_mag = math.sqrt(sum(x * x for x in node))
    if node_mag < 1e-10:
      return []

  node_hat = [node[i] / node_mag for i in range(3)]

  # Departure: ring point nearest +node, Arrival: ring point nearest -node
  dep_idx, dep_pt = _ring_point_nearest_dir(dep_ring, node_hat)
  arr_idx, arr_pt = _ring_point_nearest_dir(arr_ring, [-node_hat[i] for i in range(3)])

  dep_r = math.sqrt(sum(x * x for x in dep_pt))
  arr_r = math.sqrt(sum(x * x for x in arr_pt))
  if dep_r < 1e-10 or arr_r < 1e-10:
    return []

  dep_hat = [dep_pt[i] / dep_r for i in range(3)]
  arr_hat = [arr_pt[i] / arr_r for i in range(3)]

  # Orbit tangent at departure from adjacent ring points
  nd = len(dep_ring)
  td = [dep_ring[(dep_idx + 1) % nd][i] - dep_ring[(dep_idx - 1) % nd][i] for i in range(3)]
  td_dot_r = sum(td[i] * dep_hat[i] for i in range(3))
  td = [td[i] - td_dot_r * dep_hat[i] for i in range(3)]
  td_mag = math.sqrt(sum(x * x for x in td))
  if td_mag < 1e-12:
    return []
  td = [td[i] / td_mag for i in range(3)]

  # Orbit tangent at arrival from adjacent ring points
  na = len(arr_ring)
  ta = [arr_ring[(arr_idx + 1) % na][i] - arr_ring[(arr_idx - 1) % na][i] for i in range(3)]
  ta_dot_r = sum(ta[i] * arr_hat[i] for i in range(3))
  ta = [ta[i] - ta_dot_r * arr_hat[i] for i in range(3)]
  ta_mag = math.sqrt(sum(x * x for x in ta))
  if ta_mag < 1e-12:
    return []
  ta = [ta[i] / ta_mag for i in range(3)]

  # Hermite scale factor = angle between dep_hat and arr_hat
  dot_da = max(-1.0, min(1.0, sum(dep_hat[i] * arr_hat[i] for i in range(3))))
  angle = math.acos(dot_da)
  if angle < 0.01:
    angle = math.pi
  S = angle

  # Hermite spline direction + cosine radius
  points = []
  for k in range(n_pts + 1):
    t = k / n_pts
    t2 = t * t
    t3 = t2 * t
    h00 = 2.0 * t3 - 3.0 * t2 + 1.0
    h10 = t3 - 2.0 * t2 + t
    h01 = -2.0 * t3 + 3.0 * t2
    h11 = t3 - t2

    dx = h00 * dep_hat[0] + h10 * S * td[0] + h01 * arr_hat[0] + h11 * S * ta[0]
    dy = h00 * dep_hat[1] + h10 * S * td[1] + h01 * arr_hat[1] + h11 * S * ta[1]
    dz = h00 * dep_hat[2] + h10 * S * td[2] + h01 * arr_hat[2] + h11 * S * ta[2]

    d_mag = math.sqrt(dx * dx + dy * dy + dz * dz)
    if d_mag < 1e-12:
      d_mag = 1e-12
    dx /= d_mag
    dy /= d_mag
    dz /= d_mag

    r = (dep_r + arr_r) / 2.0 + (dep_r - arr_r) / 2.0 * math.cos(math.pi * t)
    points.append([round(r * dx, 4), round(r * dy, 4), round(r * dz, 4)])

  return points


def _build_viz_data(start_orbit, orbits, num_stations, order=None):
  """Build visualization data: orbit rings, positions, path."""
  scale = 1.0 / EARTH_RADIUS  # normalize so Earth radius = 1.0

  # Start orbit ring
  try:
    a, e, inc, raan, argp = _state_to_elements(start_orbit)
    start_ring = _orbit_ring_points(a, e, inc, raan, argp, n_pts=64, scale=scale)
  except Exception:
    start_ring = []

  start_pos = [round(start_orbit[i] * scale, 4) for i in range(3)]

  # Station orbit rings and positions
  orbit_rings = []
  station_positions = []
  for i in range(num_stations):
    sv = orbits[i]
    station_positions.append([round(sv[j] * scale, 4) for j in range(3)])
    try:
      a, e, inc, raan, argp = _state_to_elements(sv)
      orbit_rings.append(_orbit_ring_points(a, e, inc, raan, argp, n_pts=64, scale=scale))
    except Exception:
      orbit_rings.append([])

  # Transfer arcs (if we have a valid order)
  transfer_arcs = None
  if order is not None:
    transfer_arcs = []
    prev_ring = start_ring
    prev_sv = start_orbit
    for idx in order:
      if 0 <= idx < num_stations:
        next_ring = orbit_rings[idx]
        next_sv = orbits[idx]
        arc_points = _compute_transfer_arc(prev_ring, next_ring, prev_sv, next_sv, n_pts=40)
        if arc_points:
          transfer_arcs.append(arc_points)
        prev_ring = next_ring
        prev_sv = next_sv

  return {
    'earthRadius': 1.0,
    'orbitRings': orbit_rings,
    'stationPositions': station_positions,
    'startPosition': start_pos,
    'startRing': start_ring,
    'transferArcs': transfer_arcs,
  }


def _generate_orbital_viz_html(viz_data, name="Orbital TSP"):
  """Generate three.js HTML for orbital TSP visualization."""
  viz_id = str(uuid.uuid4())[:8]
  n_stations = len(viz_data['stationPositions'])
  arcs = viz_data['transferArcs']
  path_label = f", {len(arcs)} transfers" if arcs else ""

  data_json = json.dumps(viz_data)

  # The JS template uses {{ }} for literal braces inside the f-string
  return f"""
    <div class="orbital-tsp-visualization" style="margin: 15px 0;">
        <details>
            <summary style="cursor: pointer; padding: 8px; background: #e8e8e8; border-radius: 4px; font-weight: bold; color: #333; border: 1px solid #ccc;">
                &#128752; 3D Orbital View: {name} ({n_stations} stations{path_label})
            </summary>
            <div style="margin-top: 10px;">
                <div id="orb-{viz_id}" style="width: 100%; height: 500px; border: 1px solid #333; background: #080818; border-radius: 4px; position: relative; display: flex; align-items: center; justify-content: center; color: #999;">
                    <span class="viz-placeholder">Scroll here to activate 3D view</span>
                </div>
                <div style="margin-top: 8px; font-size: 12px; color: #666; background: #f8f8f8; padding: 5px; border-radius: 3px;">
                    Blue sphere = Earth | colored rings = station orbits | white ring = start orbit | green sphere = start | gradient line = solution path (green&#8594;red) | Drag to rotate, scroll to zoom, right-drag to pan
                </div>
            </div>
        </details>
    </div>
    <script src="https://cdn.jsdelivr.net/npm/three@0.132.2/build/three.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/three@0.132.2/examples/js/controls/OrbitControls.js"></script>
    <script>
    (function() {{
        var vizId = 'orb_{viz_id}';
        var D = {data_json};
        var scene, camera, renderer, controls, animId;
        var active = false;

        function go() {{
            if (active) return;
            active = true;
            var el = document.getElementById('orb-{viz_id}');
            if (!el || typeof THREE === 'undefined') return;
            var ph = el.querySelector('.viz-placeholder');
            if (ph) ph.style.display = 'none';

            scene = new THREE.Scene();
            scene.background = new THREE.Color(0x080818);
            camera = new THREE.PerspectiveCamera(50, el.clientWidth / el.clientHeight, 0.01, 500);
            renderer = new THREE.WebGLRenderer({{ antialias: true }});
            renderer.setSize(el.clientWidth, el.clientHeight);
            el.appendChild(renderer.domElement);
            controls = new THREE.OrbitControls(camera, renderer.domElement);
            controls.enableDamping = true;
            controls.dampingFactor = 0.05;

            var maxR = D.earthRadius, i, j, r, ring, geom, mat, pos, positions, colors, t, n;
            for (i = 0; i < D.orbitRings.length; i++) {{
                ring = D.orbitRings[i];
                for (j = 0; j < ring.length; j++) {{
                    r = Math.sqrt(ring[j][0]*ring[j][0]+ring[j][1]*ring[j][1]+ring[j][2]*ring[j][2]);
                    if (r > maxR) maxR = r;
                }}
            }}
            if (maxR < 2) maxR = 2;
            camera.position.set(maxR*1.2, maxR*0.8, maxR*1.2);
            camera.lookAt(0,0,0);

            scene.add(new THREE.AmbientLight(0x404060));
            var dl = new THREE.DirectionalLight(0xffffff, 0.8);
            dl.position.set(5,3,5);
            scene.add(dl);

            // Earth
            scene.add(new THREE.Mesh(
                new THREE.SphereGeometry(D.earthRadius, 32, 32),
                new THREE.MeshPhongMaterial({{ color: 0x2244aa, emissive: 0x112244, transparent: true, opacity: 0.85 }})
            ));

            // Equatorial ring
            var eq = [];
            for (i = 0; i <= 64; i++) {{
                var a = 2*Math.PI*i/64;
                eq.push(Math.cos(a)*D.earthRadius*1.02, 0, Math.sin(a)*D.earthRadius*1.02);
            }}
            geom = new THREE.BufferGeometry();
            geom.setAttribute('position', new THREE.Float32BufferAttribute(eq, 3));
            scene.add(new THREE.Line(geom, new THREE.LineBasicMaterial({{ color: 0x336699, transparent: true, opacity: 0.5 }})));

            var nS = D.stationPositions.length;
            // Station orbit rings
            for (i = 0; i < D.orbitRings.length; i++) {{
                ring = D.orbitRings[i];
                if (!ring || ring.length < 2) continue;
                positions = new Float32Array(ring.length*3);
                for (j = 0; j < ring.length; j++) {{ positions[j*3]=ring[j][0]; positions[j*3+1]=ring[j][1]; positions[j*3+2]=ring[j][2]; }}
                geom = new THREE.BufferGeometry();
                geom.setAttribute('position', new THREE.BufferAttribute(positions, 3));
                var hue = i / nS;
                mat = new THREE.LineBasicMaterial({{ color: new THREE.Color().setHSL(hue, 0.7, 0.5), transparent: true, opacity: 0.3 }});
                scene.add(new THREE.Line(geom, mat));
            }}

            // Start orbit ring (white)
            if (D.startRing && D.startRing.length > 1) {{
                positions = new Float32Array(D.startRing.length*3);
                for (j = 0; j < D.startRing.length; j++) {{ positions[j*3]=D.startRing[j][0]; positions[j*3+1]=D.startRing[j][1]; positions[j*3+2]=D.startRing[j][2]; }}
                geom = new THREE.BufferGeometry();
                geom.setAttribute('position', new THREE.BufferAttribute(positions, 3));
                scene.add(new THREE.Line(geom, new THREE.LineBasicMaterial({{ color: 0xffffff, transparent: true, opacity: 0.6 }})));
            }}

            // Station markers
            var ms = Math.max(0.03, maxR*0.012);
            for (i = 0; i < nS; i++) {{
                pos = D.stationPositions[i];
                var sc = new THREE.Color().setHSL(i/nS, 0.8, 0.6);
                var sp = new THREE.Mesh(new THREE.SphereGeometry(ms, 8, 8), new THREE.MeshBasicMaterial({{ color: sc }}));
                sp.position.set(pos[0], pos[1], pos[2]);
                scene.add(sp);
            }}

            // Start marker (green)
            pos = D.startPosition;
            var ss = new THREE.Mesh(new THREE.SphereGeometry(ms*2.5, 12, 12), new THREE.MeshBasicMaterial({{ color: 0x00ff00 }}));
            ss.position.set(pos[0], pos[1], pos[2]);
            scene.add(ss);

            // Transfer arcs (gradient green -> yellow -> red)
            if (D.transferArcs && D.transferArcs.length > 0) {{
                var nArcs = D.transferArcs.length;
                for (var ai = 0; ai < nArcs; ai++) {{
                    var arc = D.transferArcs[ai];
                    var arcLen = arc.length;
                    positions = new Float32Array(arcLen*3);
                    colors = new Float32Array(arcLen*3);
                    for (i = 0; i < arcLen; i++) {{
                        positions[i*3]=arc[i][0]; positions[i*3+1]=arc[i][1]; positions[i*3+2]=arc[i][2];
                        // Color based on position in overall journey
                        t = (ai + i/arcLen) / nArcs;
                        colors[i*3] = Math.min(1.0, t*2);
                        colors[i*3+1] = Math.min(1.0, (1-t)*2);
                        colors[i*3+2] = 0.1;
                    }}
                    geom = new THREE.BufferGeometry();
                    geom.setAttribute('position', new THREE.BufferAttribute(positions, 3));
                    geom.setAttribute('color', new THREE.BufferAttribute(colors, 3));
                    scene.add(new THREE.Line(geom, new THREE.LineBasicMaterial({{ vertexColors: true }})));
                }}
            }}

            function anim() {{ if (!active) return; animId = requestAnimationFrame(anim); controls.update(); renderer.render(scene, camera); }}
            anim();
        }}

        function stop() {{
            if (!active) return;
            active = false;
            if (animId) {{ cancelAnimationFrame(animId); animId = null; }}
            var el = document.getElementById('orb-{viz_id}');
            if (renderer) {{ renderer.dispose(); if (renderer.domElement && renderer.domElement.parentNode) renderer.domElement.parentNode.removeChild(renderer.domElement); renderer = null; }}
            if (scene) {{ scene.traverse(function(o) {{ if (o.geometry) o.geometry.dispose(); if (o.material) {{ if (Array.isArray(o.material)) o.material.forEach(function(m){{m.dispose();}}); else o.material.dispose(); }} }}); scene = null; }}
            camera = null; controls = null;
            if (el) {{ var ph = el.querySelector('.viz-placeholder'); if (ph) ph.style.display = ''; }}
        }}

        if (window.VizManager) {{ window.VizManager.register({{ id: vizId, containerId: 'orb-{viz_id}', activate: go, dispose: stop }}); }}
        else {{ go(); }}
    }})();
    </script>"""


def resultToNiceReport(result: dict, subPass: int, aiEngineName: str) -> str:
  """Generate HTML report with 3D orbital visualization."""
  if not result:
    return "<p style='color:red'>No result provided</p>"

  num_stations = STATION_COUNTS[subPass]

  html = f"<h4>Orbital TSP - {num_stations} stations</h4>"

  if subPass == 0:
    if "reasoning" in result:
      reasoning = result['reasoning'][:500] + ('...'
                                               if len(result.get('reasoning', '')) > 500 else '')
      html += f"<p><strong>Approach:</strong> {reasoning}</p>"

    if "csharp_code" in result:
      code = result["csharp_code"]
      code_escaped = code.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
      html += f"<details><summary>View Code ({len(code)} chars)</summary><pre>{code_escaped}</pre></details>"

  # Run solver to get visitation order for visualization
  start_orbit, orbits = _get_dataset_for_subpass(subPass)
  order = None
  if "csharp_code" in result:
    try:
      order, error, _ = execute_solver(result["csharp_code"], num_stations, start_orbit, orbits)
      if error:
        order = None
    except Exception:
      order = None

  viz_data = _build_viz_data(start_orbit, orbits, num_stations, order)
  html += _generate_orbital_viz_html(viz_data, name=f"{num_stations} stations (subpass {subPass})")

  return html


highLevelSummary = """
Orbital TSP combines the Travelling Salesman Problem with orbital mechanics.

**Challenges:**
- Stations are moving targets (orbiting at ~7 km/s)
- Transfer costs depend on timing and orbital geometry
- Plane changes are very expensive
- Need to solve Lambert's problem for optimal transfers

**Key orbital mechanics:**
- **Two-body problem**: Kepler's laws govern motion
- **Hohmann transfer**: Minimum fuel for coplanar circular orbits
- **Lambert's problem**: Find orbit connecting two positions in given time
- **Delta-V**: Velocity change = fuel consumption

**Approaches:**
- Simplify: Use energy/distance heuristics
- Full solution: Implement orbital propagation + Lambert solver
- Optimization: Try multiple orderings, pick minimum delta-V

The baseline uses a greedy nearest-neighbor approach with simplified
delta-V estimation based on orbital energy differences.
"""
