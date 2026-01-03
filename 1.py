import math
import random
import subprocess
import sys
import tempfile
import os
import time

title = "Travelling Salesman Problem Solver"

# Seed for reproducible city generation
RANDOM_SEED = 42

# Timeout in seconds (30 seconds)
TIMEOUT_SECONDS = 30

# City counts for each subpass
CITY_COUNTS = [
  10, 20, 30, 40, 100, 200, 300, 400, 500, 750, 1_000, 5_000, 10_000, 50_000, 100_000, 500_000,
  1_000_000, 5_000_000, 10_000_000, 50_000_000, 100_000_000
]


# Pre-generate cities for each subpass (deterministic)
def generate_cities(n: int, seed: int = RANDOM_SEED) -> list:
  """Generate n cities with random coordinates in 2D space [0, 1000]."""
  rng = random.Random(seed + n)  # Different seed per city count
  return [(rng.uniform(0, 1000), rng.uniform(0, 1000)) for _ in range(n)]


# Cache generated cities
CITIES_CACHE = {}


def format_cities_for_prompt(cities: list) -> str:
  """Format cities as a Python list literal for the prompt."""
  lines = ["["]
  for i, (x, y) in enumerate(cities):
    lines.append(f"    ({x:.2f}, {y:.2f}),  # City {i}")
  lines.append("]")
  return "\n".join(lines)


def prepareSubpassPrompt(subPass: int) -> str:
  """Generate the prompt for subpass 0 that handles all city counts."""
  if subPass != 0:
    raise StopIteration

  return f"""You are solving the Travelling Salesman Problem (TSP).

You must write a Python solver that can handle ANY number of cities from trivial to ludicrous scale:
- **Trivial**: 10-40 cities (brute-force feasible for smallest cases)
- **Medium**: 100-1000 cities (requires heuristics)
- **Large**: 5000-100000 cities (requires efficient heuristics)
- **Extreme**: 500000-5000000 cities (requires very fast approximations)
- **Epic**: 10000000+ cities (requires highly optimized approximations)

**The Challenge:**
Your `solve_tsp(cities)` function will be tested with city counts ranging from 10 to 10000000. The same function must work efficiently across ALL scales.

**Input:**
- `cities`: List of (x, y) tuples representing city coordinates in 2D space [0, 1000]
- The cost between cities is Euclidean distance

**Output:**
- List of city indices representing the route (starting at city 0, visiting all cities exactly once)
- The return list should NOT include returning to city 0 (that's implicit)

**Critical Requirements:**
1. **Scalability**: Your algorithm must adapt based on problem size
2. **Performance**: Must complete within 30 seconds even for 500,000 cities
3. **Quality**: Should produce reasonable routes at all scales

**Constraints:**
- Use only Python standard library (math, itertools, heapq, random, etc.)
- Must handle edge cases (empty list, single city)
- Route must be valid (visits each city exactly once)

Write complete, runnable Python code with the solve_tsp(cities) function.
Include adaptive logic that chooses different strategies based on the number of cities.
"""


# List of subpasses to grade the single answer against all difficulty levels
extraGradeAnswerRuns = list(range(1, len(CITY_COUNTS)))

structure = {
  "type": "object",
  "properties": {
    "reasoning": {
      "type": "string",
      "description": "Explain your approach and how it adapts to different problem sizes"
    },
    "python_code": {
      "type": "string",
      "description": "Complete Python code with solve_tsp(cities) function that handles all scales"
    }
  },
  "required": ["reasoning", "python_code"],
  "additionalProperties": False
}


def calculate_route_distance(cities: list, route: list) -> float:
  """Calculate total distance of a route (including return to start)."""
  if not route:
    return float('inf')

  total = 0.0
  for i in range(len(route)):
    c1 = cities[route[i]]
    c2 = cities[route[(i + 1) % len(route)]]
    dx = c1[0] - c2[0]
    dy = c1[1] - c2[1]
    total += math.sqrt(dx * dx + dy * dy)

  return total


def validate_route(cities: list, route: list) -> tuple:
  """
    Validate that a route is valid.
    Returns (is_valid, error_message).
    """
  n = len(cities)

  if not isinstance(route, list):
    return False, f"Route must be a list, got {type(route).__name__}"

  if len(route) != n:
    return False, f"Route must visit all {n} cities, got {len(route)} cities"

  if set(route) != set(range(n)):
    missing = set(range(n)) - set(route)
    extra = set(route) - set(range(n))
    msg = ""
    if missing:
      msg += f"Missing cities: {sorted(missing)[:5]}{'...' if len(missing) > 5 else ''}. "
    if extra:
      msg += f"Invalid cities: {sorted(extra)[:5]}{'...' if len(extra) > 5 else ''}."
    return False, msg

  if len(route) != len(set(route)):
    return False, "Route contains duplicate cities"

  return True, ""


def _orient(a, b, c) -> float:
  return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def _on_segment(a, b, p, eps: float) -> bool:
  return (min(a[0], b[0]) - eps <= p[0] <= max(a[0], b[0]) + eps
          and min(a[1], b[1]) - eps <= p[1] <= max(a[1], b[1]) + eps)


def _segments_intersect(a, b, c, d, eps: float = 1e-9) -> bool:
  if max(min(a[0], b[0]), min(c[0], d[0])) > min(max(a[0], b[0]), max(c[0], d[0])) + eps:
    return False
  if max(min(a[1], b[1]), min(c[1], d[1])) > min(max(a[1], b[1]), max(c[1], d[1])) + eps:
    return False

  o1 = _orient(a, b, c)
  o2 = _orient(a, b, d)
  o3 = _orient(c, d, a)
  o4 = _orient(c, d, b)

  def sgn(v: float) -> int:
    if abs(v) <= eps:
      return 0
    return 1 if v > 0 else -1

  s1, s2, s3, s4 = sgn(o1), sgn(o2), sgn(o3), sgn(o4)

  if s1 == 0 and _on_segment(a, b, c, eps):
    return True
  if s2 == 0 and _on_segment(a, b, d, eps):
    return True
  if s3 == 0 and _on_segment(c, d, a, eps):
    return True
  if s4 == 0 and _on_segment(c, d, b, eps):
    return True

  return (s1 * s2 < 0) and (s3 * s4 < 0)


def find_edge_crossing(cities: list,
                       route: list,
                       full_check_n: int = 2000,
                       sample_edges: int = 1000,
                       seed: int = RANDOM_SEED):
  n = len(route)
  if n < 4:
    return None

  def endpoints(edge_idx: int):
    a_i = route[edge_idx]
    b_i = route[(edge_idx + 1) % n]
    return cities[a_i], cities[b_i], a_i, b_i

  if n <= full_check_n:
    for i in range(n):
      a, b, a_i, b_i = endpoints(i)
      for j in range(i + 1, n):
        if j == i + 1:
          continue
        if i == 0 and j == n - 1:
          continue
        c, d, c_i, d_i = endpoints(j)
        if _segments_intersect(a, b, c, d):
          return (i, j, (a_i, b_i), (c_i, d_i))
    return None

  k = min(sample_edges, n)
  rng = random.Random(seed + n)
  edge_indices = rng.sample(range(n), k)

  for ii in range(k):
    i = edge_indices[ii]
    a, b, a_i, b_i = endpoints(i)
    for jj in range(ii + 1, k):
      j = edge_indices[jj]
      diff = abs(i - j)
      if diff == 1 or diff == 0 or diff == n - 1:
        continue
      c, d, c_i, d_i = endpoints(j)
      if _segments_intersect(a, b, c, d):
        return (i, j, (a_i, b_i), (c_i, d_i))
  return None


def get_baseline_distance(cities: list) -> float:
  """Get baseline distance using fast heuristic based on city density."""
  if not cities:
    return 0.0

  n = len(cities)

  # For small n, use the exact algorithm (fast enough)
  if n <= 100:
    # Use nearest neighbor for small graphs
    visited = [False] * n
    route = [0]
    visited[0] = True
    current = 0

    for _ in range(n - 1):
      best_next = None
      best_dist = float('inf')

      for j in range(n):
        if not visited[j]:
          dx = cities[current][0] - cities[j][0]
          dy = cities[current][1] - cities[j][1]
          dist = math.sqrt(dx * dx + dy * dy)
          if dist < best_dist:
            best_dist = dist
            best_next = j

      if best_next is not None:
        visited[best_next] = True
        route.append(best_next)
        current = best_next

    return calculate_route_distance(cities, route)

  # For large n, use density-based heuristic
  # Cities are uniformly distributed in 1000x1000 space
  # Average distance between nearby cities ≈ sqrt(area/n) = 1000/√n
  # TSP route length ≈ n * average_distance * efficiency_factor

  # Average spacing between cities
  avg_spacing = 1000.0 / math.sqrt(n)

  # Efficiency factor accounts for:
  # - TSP route is more efficient than random walk
  # - But less efficient than perfect grid
  # For random points, efficiency ≈ 0.6-0.8
  efficiency = 0.25

  # Estimated route length
  estimated_length = n * avg_spacing * efficiency

  # Add some variance based on n (larger problems have more optimization opportunities)
  if n > 100000:
    estimated_length *= 0.9  # Very large problems have more room for optimization
  elif n > 10000:
    estimated_length *= 0.95

  return estimated_length


def execute_solver(code: str, cities: list, timeout: int = TIMEOUT_SECONDS) -> tuple:
  """
    Execute the LLM's solver code safely in a subprocess.
    Returns (route, error_message, execution_time).
    """
  from solver_utils import execute_solver_with_data

  # Use the common utility with debugger isolation
  data_dict = {'cities': cities}
  return execute_solver_with_data(code, data_dict, 'solve_tsp', timeout)


lastRoute = None


def gradeAnswer(result: dict, subPass: int, aiEngineName: str) -> tuple:
  """
    Grade the LLM's TSP solver.
    
    Scoring:
    - 0.0: Invalid route or solver error
    - 0.5: Valid route but worse than 2x baseline
    - 0.7: Valid route within 2x baseline
    - 0.85: Valid route within 1.5x baseline
    - 1.0: Valid route within 1.1x baseline (near-optimal)
    """
  if not result:
    return 0.0, "No result provided"

  if "python_code" not in result:
    return 0.0, "No Python code provided"

  global CITIES_CACHE
  global lastRoute
  n = CITY_COUNTS[subPass]
  if subPass not in CITIES_CACHE:
    print(f"Lazily generating cities for subPass {subPass}, which is {CITY_COUNTS[subPass]} cities")
    CITIES_CACHE[subPass] = generate_cities(n)

  cities = CITIES_CACHE[subPass]
  code = result["python_code"]

  # Execute the solver
  route, error, exec_time = execute_solver(code, cities)

  if error:
    lastRoute = error
    return 0.0, f"[{n} cities] {error}"

  # Validate the route
  is_valid, validation_error = validate_route(cities, route)
  if not is_valid:
    return 0.0, f"[{n} cities] Invalid route: {validation_error}"

  lastRoute = route

  crossing = find_edge_crossing(cities, route)
  if crossing:
    i, j, e1, e2 = crossing
    return 0.0, f"[{n} cities] Route has edge crossing between edges {i}({e1[0]}→{e1[1]}) and {j}({e2[0]}→{e2[1]})"

  # Calculate distances
  route_distance = calculate_route_distance(cities, route)
  baseline_distance = get_baseline_distance(cities)

  ratio = route_distance / baseline_distance if baseline_distance > 0 else float('inf')

  # Score based on how close to baseline (or better)
  if ratio <= 1.1:
    score = 1.0
    quality = "excellent (within 10% of baseline)"
  elif ratio <= 1.5:
    score = 0.5
    quality = "good (within 50% of baseline)"
  elif ratio <= 2.0:
    score = 0.1
    quality = "acceptable (within 2x baseline)"
  else:
    score = 0.0
    quality = f"poor ({ratio:.1f}x baseline)"

  explanation = (f"[{n} cities] Route distance: {route_distance:.2f}, "
                 f"Baseline: {baseline_distance:.2f}, "
                 f"Ratio: {ratio:.2f}x, "
                 f"Time: {exec_time:.1f}s - {quality}")

  return score, explanation


def resultToNiceReport(result: dict, subPass: int, aiEngineName: str) -> str:
  """Generate a nice HTML report for the result."""
  n = CITY_COUNTS[subPass]
  global CITIES_CACHE

  html = f"<h4>TSP Solver - {n} Cities</h4>"

  if subPass == 0:
    if "reasoning" in result:
      html += f"<p><strong>Approach:</strong> {result['reasoning'][:500]}{'...' if len(result.get('reasoning', '')) > 500 else ''}</p>"

    if "python_code" in result:
      code = result["python_code"]
      # Escape HTML
      code_escaped = code.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
      html += f"<details><summary>View Code ({len(code)} chars)</summary><pre>{code_escaped}</pre></details>"

  if isinstance(lastRoute, str):
    return html + f"<p style='color:red'>Error: {lastRoute}</p>"

  if not result:
    return html + "<p style='color:red'>No result provided</p>"

  if subPass not in CITIES_CACHE:
    return html + "<p style='color:red'>Cities data not available for this problem size</p>"

  # Add SVG rendering of the route
  cities = CITIES_CACHE[subPass]

  if lastRoute:
    if len(lastRoute) <= 1000:  # Only render for smaller routes to avoid bloat
      html += generate_route_svg(cities, lastRoute, n)
    else:
      html += f"<p>Route too large ({len(lastRoute)} cities) to render SVG</p>"
  else:
    html += "<p style='color:red'>Could not render route: No valid route generated</p>"

  return html


def generate_route_svg(cities: list, route: list, city_count: int) -> str:
  """Generate SVG visualization of the TSP route."""
  if not cities or not route:
    return "<p>No route to visualize</p>"

  # Find bounds
  x_coords = [c[0] for c in cities]
  y_coords = [c[1] for c in cities]
  min_x, max_x = min(x_coords), max(x_coords)
  min_y, max_y = min(y_coords), max(y_coords)

  # Add padding
  padding = 20
  width = 400
  height = 400

  # Scale coordinates to fit SVG
  scale_x = (width - 2 * padding) / (max_x - min_x) if max_x != min_x else 1
  scale_y = (height - 2 * padding) / (max_y - min_y) if max_y != min_y else 1
  scale = min(scale_x, scale_y)

  def transform_point(x, y):
    tx = padding + (x - min_x) * scale
    ty = padding + (y - min_y) * scale
    return tx, ty

  # Generate route path
  path_points = []
  for i in range(len(route)):
    city_idx = route[i]
    x, y = transform_point(cities[city_idx][0], cities[city_idx][1])
    path_points.append(f"{x},{y}")

  # Close the loop back to start
  start_x, start_y = transform_point(cities[route[0]][0], cities[route[0]][1])
  path_points.append(f"{start_x},{start_y}")

  path_data = " L ".join(path_points)

  # Generate city points
  city_circles = ""
  for i, (x, y) in enumerate(cities):
    tx, ty = transform_point(x, y)
    color = "#ff4444" if i == route[0] else "#4444ff"  # Start city in red
    city_circles += f'<circle cx="{tx}" cy="{ty}" r="3" fill="{color}" title="City {i}"/>'

  svg_html = f'''
  <div style="margin: 10px 0; width: 100%">
    <h5>Route Visualization ({city_count} cities)</h5>
    <svg width="100%" style="border: 1px solid #ccc; background: white;" viewBox="0 0 {width} {height}">
      <path d="M {path_data}" stroke="#333" stroke-width="1.5" fill="none" opacity="0.7"/>
      {city_circles}
    </svg>
    <p style="font-size: 12px; color: #666;">
      <span style="color: #ff4444;">● Start city</span> | 
      <span style="color: #4444ff;">● Other cities</span> | 
      Route length: {len(route)} cities
    </p>
  </div>'''

  return svg_html


highLevelSummary = """
This test evaluates whether LLMs can write efficient algorithms that scale.

The Travelling Salesman Problem is NP-hard, meaning brute-force solutions 
become intractable very quickly:
- 10 cities: 3.6 million routes (feasible)
- 20 cities: 2.4 × 10^18 routes (impossible to brute-force)
- 1000 cities: astronomical number of routes

To pass this test, the LLM must discover and implement heuristics such as:
- Nearest Neighbor
- 2-opt local search
- Christofides algorithm
- Genetic algorithms
- Simulated annealing

The baseline uses a simple nearest-neighbor heuristic. LLMs are scored on
whether their solution is valid and how close it comes to the baseline distance.
"""
