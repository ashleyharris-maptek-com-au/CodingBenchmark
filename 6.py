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
import subprocess
import sys
import tempfile
import os
import time
from typing import List, Tuple, Dict

title = "Orbital TSP - Space Station Route Planning"

# Timeout in seconds (5 minutes)
TIMEOUT_SECONDS = 30

# Standard gravitational parameter for Earth (km^3/s^2)
MU = 398600.4418
EARTH_RADIUS = 6371  # km

# Orbit data: [Pos X, Y, Z, Vel X, Y, Z] at epoch T=0 (km, km/s)
ORBITS = [
  [4579.848378, 5233.075386, 578.192725, -5.209886, 4.167153, 3.551514],
  [5315.246203, -4099.931110, -1906.074155, 1.612607, 4.727540, -5.671965],
  [7078.137000, 0.000000, 0.000000, 0.000000, -1.122156, 7.419911],
  [-3589.068500, 6216.448994, 0.000000, -6.453475, -3.725916, 0.000000],
  [-5021.137307, -2809.845016, 4456.930482, 1.463799, -6.765963, -2.616458],
  [-3789.068500, 0.000000, -6562.859155, -0.000000, -7.252499, -0.000000],
  [-6607.161632, -1921.218246, 3836.584185, 3.874056, -2.671117, 5.334098],
  [-53.281442, -4752.411653, -6655.348928, 4.056966, -4.639096, 3.280179],
  [-7661.829065, -1642.577396, 2820.670269, -2.051378, -1.486347, -6.437742],
  [-1362.468620, 5363.320462, -4332.803085, -6.670648, -3.059719, -1.689831],
]

# Starting orbit (spacecraft initial position)
START_ORBIT = [-1845.998197, -6653.941516, -3451.314228, 4.305530, 1.936911, -5.512077]

# Number of stations for each subpass
STATION_COUNTS = [2, 3, 4, 5, 6, 8, 10, 15, 20, 50, 100]


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

  return f"""You are solving an Orbital Travelling Salesman Problem.

You must write a Python solver that can handle ANY number of stations from trivial to ludicrous scale:
- **Trivial**: 2-5 stations (simple brute-force permutations feasible)
- **Medium**: 6-10 stations (requires heuristics, some optimization)
- **Large**: 15-20 stations (complex orbital mechanics, needs efficient algorithms)
- **Extreme**: 50-100 stations (very complex, requires advanced heuristics)

**The Challenge:**
Your `solve_orbital_tsp(start_orbit, station_orbits, mu)` function will be tested with 2 to 100 stations. The same function must work efficiently across ALL scales.

**Starting position (epoch T=0):**
Position: ({START_ORBIT[0]:.1f}, {START_ORBIT[1]:.1f}, {START_ORBIT[2]:.1f}) km
Velocity: ({START_ORBIT[3]:.3f}, {START_ORBIT[4]:.3f}, {START_ORBIT[5]:.3f}) km/s
Altitude: ~{start_info['altitude_km']:.0f} km

**Station orbits:**
- {len(ORBITS)} available station orbits with varying altitudes and inclinations
- Your function will receive subsets of these stations to visit
- All orbits follow Kepler's laws with Earth's gravity

**Physics constants:**
- Earth gravitational parameter μ = {MU} km³/s²
- Earth radius = {EARTH_RADIUS} km
- Earth-Centered Inertial (ECI) coordinates
- Two-body dynamics (ignore Moon, Sun, atmosphere)
- Instantaneous impulsive burns (delta-V applied instantly)

**Input:**
- `start_orbit`: [x, y, z, vx, vy, vz] - spacecraft initial state vector (6 floats: position + velocity)
- `station_orbits`: List of [x, y, z, vx, vy, vz] for each station (each is 6 floats)
- `mu`: Gravitational parameter (single float)

**Note:** All orbit data are 6-element lists containing position (x,y,z) and velocity (vx,vy,vz) components.

**Output:**
- Dict with:
  - `"visit_order"`: list of station indices in visitation order
  - `"total_delta_v"`: estimated total delta-V in km/s
  - `"reasoning"`: brief explanation of approach

**Critical Requirements:**
1. **Scalability**: Your algorithm must adapt based on number of stations
2. **Performance**: Must complete within 5 minutes even for 100 stations
3. **Accuracy**: Should provide reasonable delta-V estimates

**Constraints:**
- Use only Python standard library
- Handle varying numbers of stations efficiently
- Provide reasonable orbital transfer estimates

Write complete, runnable Python code with the solve_orbital_tsp function.
Include adaptive logic that chooses different strategies based on station count.
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
    "python_code": {
      "type": "string",
      "description": "Complete Python code with solve_orbital_tsp function that handles all scales"
    }
  },
  "required": ["reasoning", "python_code"],
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


def get_baseline_solution(num_stations: int) -> Tuple[List[int], float]:
  """
  Get baseline solution using intelligent orbital clustering.
  Groups similar orbits together, then finds optimal sequence between clusters.
  """
  station_orbits = ORBITS[:num_stations]

  if num_stations <= 2:
    # For small numbers, just try all permutations
    if num_stations == 1:
      return [0], estimate_transfer_delta_v(START_ORBIT, station_orbits[0])
    elif num_stations == 2:
      # Try both orders
      dv1 = estimate_transfer_delta_v(START_ORBIT, station_orbits[0]) + \
            estimate_transfer_delta_v(station_orbits[0], station_orbits[1])
      dv2 = estimate_transfer_delta_v(START_ORBIT, station_orbits[1]) + \
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
  current_orbit = START_ORBIT

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
  current_orbit = START_ORBIT

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


def execute_solver(code: str, num_stations: int, timeout: int = TIMEOUT_SECONDS) -> tuple:
  """Execute the LLM's solver. Returns (result_dict, error, exec_time)."""
  from solver_utils import execute_solver_with_data

  station_orbits = ORBITS[:num_stations]

  # Use the common utility with debugger isolation
  data_dict = {'MU': MU, 'start_orbit': START_ORBIT, 'station_orbits': station_orbits}

  return execute_solver_with_data(code, data_dict, 'solve_orbital_tsp', timeout)


def validate_solution(result: Dict, num_stations: int) -> Tuple[bool, str]:
  """Validate the solution format."""
  if not isinstance(result, dict):
    return False, "Result must be a dict"

  if "visit_order" not in result:
    return False, "Missing 'visit_order' in result"

  order = result["visit_order"]
  if not isinstance(order, list):
    return False, "visit_order must be a list"

  if len(order) != num_stations:
    return False, f"visit_order must have {num_stations} stations, got {len(order)}"

  if set(order) != set(range(num_stations)):
    return False, f"visit_order must contain each station 0 to {num_stations-1} exactly once"

  return True, ""


def evaluate_order(order: List[int]) -> float:
  """Evaluate the delta-V for a given visitation order."""
  current_orbit = START_ORBIT
  total_dv = 0

  for station in order:
    dv = estimate_transfer_delta_v(current_orbit, ORBITS[station])
    total_dv += dv
    current_orbit = ORBITS[station]

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

  if "python_code" not in result:
    return 0.0, "No Python code provided"

  num_stations = STATION_COUNTS[subPass]
  code = result["python_code"]

  # Execute solver
  solution, error, exec_time = execute_solver(code, num_stations)

  if error:
    return 0.0, f"[{num_stations} stations] {error}"

  # Validate solution
  is_valid, validation_error = validate_solution(solution, num_stations)
  if not is_valid:
    return 0.0, f"[{num_stations} stations] Invalid: {validation_error}"

  # Evaluate the order
  order = solution["visit_order"]
  solution_dv = evaluate_order(order)

  # Get baseline
  baseline_order, baseline_dv = get_baseline_solution(num_stations)

  # Score
  ratio = solution_dv / baseline_dv if baseline_dv > 0 else float('inf')

  if ratio <= 1.0:
    score = 1.0
    quality = "excellent (≤ baseline)"
  elif ratio <= 1.2:
    score = 0.85
    quality = "good (≤ 1.2x baseline)"
  elif ratio <= 1.5:
    score = 0.7
    quality = "acceptable (≤ 1.5x baseline)"
  else:
    score = 0.5
    quality = f"poor ({ratio:.1f}x baseline)"

  explanation = (f"[{num_stations} stations] Order: {order}, "
                 f"Delta-V: {solution_dv:.2f} km/s, Baseline: {baseline_dv:.2f} km/s, "
                 f"Time: {exec_time:.1f}s - {quality}")

  return score, explanation


def resultToNiceReport(result: dict, subPass: int, aiEngineName: str) -> str:
  """Generate HTML report."""
  if not result:
    return "<p style='color:red'>No result provided</p>"

  num_stations = STATION_COUNTS[subPass]

  html = f"<h4>Orbital TSP - {num_stations} stations</h4>"

  if subPass == 0:
    if "reasoning" in result:
      reasoning = result['reasoning'][:500] + ('...'
                                               if len(result.get('reasoning', '')) > 500 else '')
      html += f"<p><strong>Approach:</strong> {reasoning}</p>"

    if "python_code" in result:
      code = result["python_code"]
      code_escaped = code.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
      html += f"<details><summary>View Code ({len(code)} chars)</summary><pre>{code_escaped}</pre></details>"

  # TODO Add a visualisation, once an LLM gets more than 0.

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
