"""
Test 34: Capacitated Vehicle Routing Problem (Python Implementation)

The LLM must write Python code to find optimal routes for vehicles with
capacity constraints to serve all customers. This is NP-Hard.

Subpasses increase customer count and tighten capacity, requiring savings
algorithm, sweep algorithm, or metaheuristics like genetic algorithms.

Solver times out after 5 minutes.
"""

import random
import subprocess
import time
import math
from typing import List, Tuple, Dict, Any
from solver_utils import StreamingInputFile

title = "Vehicle Routing Problem (Python)"
TIMEOUT_SECONDS = 30
RANDOM_SEED = 34343434


def generate_vrp(num_customers: int, num_vehicles: int, capacity: int,
                 seed: int) -> Tuple[List[Tuple[float, float]], List[int], Tuple[float, float]]:
  """Generate VRP instance with depot and customers."""
  rng = random.Random(seed)

  # Depot at center
  depot = (50.0, 50.0)

  # Random customer locations
  customers = [(rng.uniform(0, 100), rng.uniform(0, 100)) for _ in range(num_customers)]

  # Random demands (ensure feasible)
  max_demand = capacity // 3
  demands = [rng.randint(1, max_demand) for _ in range(num_customers)]

  return customers, demands, depot


def distance(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
  return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)


TEST_CASES = [
  {
    "customers": 15,
    "vehicles": 3,
    "capacity": 50,
    "desc": "15 customers, 3 vehicles"
  },
  {
    "customers": 25,
    "vehicles": 4,
    "capacity": 60,
    "desc": "25 customers, 4 vehicles"
  },
  {
    "customers": 40,
    "vehicles": 5,
    "capacity": 80,
    "desc": "40 customers, 5 vehicles"
  },
  {
    "customers": 60,
    "vehicles": 6,
    "capacity": 100,
    "desc": "60 customers, 6 vehicles"
  },
  {
    "customers": 80,
    "vehicles": 8,
    "capacity": 120,
    "desc": "80 customers, 8 vehicles"
  },
  {
    "customers": 100,
    "vehicles": 10,
    "capacity": 150,
    "desc": "100 customers, 10 vehicles"
  },
  {
    "customers": 150,
    "vehicles": 12,
    "capacity": 200,
    "desc": "150 customers, 12 vehicles"
  },
  {
    "customers": 200,
    "vehicles": 15,
    "capacity": 250,
    "desc": "200 customers, 15 vehicles"
  },
  {
    "customers": 300,
    "vehicles": 20,
    "capacity": 300,
    "desc": "300 customers, 20 vehicles"
  },
  {
    "customers": 500,
    "vehicles": 25,
    "capacity": 400,
    "desc": "500 customers, 25 vehicles"
  },
  {
    "customers": 1000,
    "vehicles": 40,
    "capacity": 500,
    "desc": "1000 customers, 40 vehicles"
  },
  # Ludicrous cases for streaming
  {
    "customers": 5000,
    "vehicles": 100,
    "capacity": 800,
    "desc": "5K customers (~100KB)"
  },
  {
    "customers": 20000,
    "vehicles": 200,
    "capacity": 1000,
    "desc": "20K customers (~500KB)"
  },
  {
    "customers": 100000,
    "vehicles": 500,
    "capacity": 1500,
    "desc": "100K customers (~2.5MB)"
  },
  {
    "customers": 1000000,
    "vehicles": 2000,
    "capacity": 2000,
    "desc": "1M customers (~25MB)"
  },
  {
    "customers": 10000000,
    "vehicles": 10000,
    "capacity": 3000,
    "desc": "10M customers (~250MB)"
  },
  {
    "customers": 50000000,
    "vehicles": 30000,
    "capacity": 5000,
    "desc": "50M customers (~1.25GB)"
  },
]

INSTANCE_CACHE: Dict[int, Any] = {}
_INPUT_FILE_CACHE: Dict[int, StreamingInputFile] = {}
STREAMING_THRESHOLD_CUSTOMERS = 10_000


def get_instance(subpass: int):
  if subpass not in INSTANCE_CACHE:
    case = TEST_CASES[subpass]
    customers, demands, depot = generate_vrp(case["customers"], case["vehicles"], case["capacity"],
                                             RANDOM_SEED + subpass)
    INSTANCE_CACHE[subpass] = (customers, demands, depot, case["vehicles"], case["capacity"])
  return INSTANCE_CACHE[subpass]


def _should_use_streaming(subpass: int) -> bool:
  return TEST_CASES[subpass]["customers"] > STREAMING_THRESHOLD_CUSTOMERS


def _get_streaming_input(subpass: int) -> StreamingInputFile:
  if subpass in _INPUT_FILE_CACHE:
    return _INPUT_FILE_CACHE[subpass]

  case = TEST_CASES[subpass]
  cache_key = f"vrp34|c={case['customers']}|v={case['vehicles']}|cap={case['capacity']}|seed={RANDOM_SEED + subpass}"

  def generator():
    customers, demands, depot, num_vehicles, capacity = get_instance(subpass)
    yield f"{len(customers)} {num_vehicles} {capacity}\n"
    yield f"{depot[0]:.2f} {depot[1]:.2f}\n"
    for (x, y), d in zip(customers, demands):
      yield f"{x:.2f} {y:.2f} {d}\n"

  input_file = StreamingInputFile(cache_key, generator, "test34_vrp")
  _INPUT_FILE_CACHE[subpass] = input_file
  return input_file


def format_input(customers: List[Tuple[float, float]], demands: List[int],
                 depot: Tuple[float, float], num_vehicles: int, capacity: int) -> str:
  lines = [f"{len(customers)} {num_vehicles} {capacity}"]
  lines.append(f"{depot[0]:.2f} {depot[1]:.2f}")
  for (x, y), d in zip(customers, demands):
    lines.append(f"{x:.2f} {y:.2f} {d}")
  return "\n".join(lines)


def prepareSubpassPrompt(subPass: int) -> str:
  """Generate the prompt for subpass 0 that handles all difficulty levels."""
  if subPass != 0:
    raise StopIteration

  return f"""You are writing CSHARP code to solve the Minimum Spanning Tree problem.

You must write a CSHARP solver that can handle ANY problem complexity from trivial to ludicrous scale:
- **Trivial**: Simple instances, basic algorithms
- **Medium**: Moderate instances, efficient algorithms  
- **Large**: Complex instances, advanced algorithms
- **Extreme**: Massive instances, very fast heuristics

**The Challenge:**
Your CSHARP solver will be tested with instances ranging from simple to very complex cases. The same algorithm must work efficiently across ALL problem complexities.

**Problem:**
Find a subset of edges that connects all vertices together, without any cycles, and with the minimum possible total edge weight.

**Input format (stdin):**
```
[Input format varies by problem]
```

**Output format (stdout):**
```
[Output format varies by problem]
```

**Critical Requirements:**
1. **Scalability**: Your algorithm must adapt based on problem size and complexity
2. **Performance**: Must complete within 5 minutes even for the largest instances
3. **Quality**: Find optimal or near-optimal solutions

**Algorithm Strategy Recommendations:**
Small graphs (100 nodes): Kruskal's, Medium (1000 nodes): optimized union-find, Large (10000 nodes): very fast implementations, Extreme (100000+ nodes): streaming algorithms

**Implementation Hints:**
- Detect problem complexity and choose appropriate algorithm
- Use efficient data structures and algorithms
- Implement adaptive quality vs speed tradeoffs
- For very large instances, focus on fast heuristics
- Handle edge cases appropriately
- Use fast I/O for large inputs

**Requirements:**
1. Program must compile with appropriate compiler
2. Read from stdin, write to stdout
3. Handle variable problem sizes
4. Complete within 5 minutes
5. Must handle varying problem complexities efficiently

Write complete, compilable CSHARP code.
Include adaptive logic that chooses different strategies based on problem complexity.
"""# List of subpasses to grade the single answer against all difficulty levels


extraGradeAnswerRuns = list(range(len(TEST_CASES)))

structure = {
  "type": "object",
  "properties": {
    "reasoning": {
      "type": "string",
      "description": "Explain your algorithm approach and how it adapts to different problem sizes"
    },
    "python_code": {
      "type": "string",
      "description": "Complete Python code with solver function that handles all scales"
    }
  },
  "required": ["reasoning", "python_code"],
  "additionalProperties": False
},


def verify_routes(customers: List[Tuple[float, float]], demands: List[int],
                  depot: Tuple[float, float], capacity: int, num_vehicles: int,
                  routes: List[List[int]]) -> Tuple[bool, float, str]:
  """Verify routes are valid and calculate total distance."""
  if len(routes) > num_vehicles:
    return False, 0, f"Too many routes: {len(routes)} > {num_vehicles}"

  visited = set()
  total_dist = 0

  for route in routes:
    route_demand = 0
    prev = depot

    for cid in route:
      if cid < 0 or cid >= len(customers):
        return False, 0, f"Invalid customer ID: {cid}"
      if cid in visited:
        return False, 0, f"Customer {cid} visited twice"
      visited.add(cid)
      route_demand += demands[cid]
      total_dist += distance(prev, customers[cid])
      prev = customers[cid]

    total_dist += distance(prev, depot)

    if route_demand > capacity:
      return False, 0, f"Route exceeds capacity: {route_demand} > {capacity}"

  if len(visited) != len(customers):
    return False, 0, f"Not all customers visited: {len(visited)}/{len(customers)}"

  return True, total_dist, "Valid"


def nearest_neighbor_distance(customers: List[Tuple[float, float]], demands: List[int],
                              depot: Tuple[float, float], capacity: int) -> float:
  """Simple nearest neighbor heuristic for comparison."""
  unvisited = set(range(len(customers)))
  total_dist = 0

  while unvisited:
    route_demand = 0
    current = depot
    route_dist = 0

    while unvisited:
      best_cid = None
      best_dist = float('inf')

      for cid in unvisited:
        if route_demand + demands[cid] <= capacity:
          d = distance(current, customers[cid])
          if d < best_dist:
            best_dist = d
            best_cid = cid

      if best_cid is None:
        break

      route_dist += best_dist
      current = customers[best_cid]
      route_demand += demands[best_cid]
      unvisited.remove(best_cid)

    route_dist += distance(current, depot)
    total_dist += route_dist

  return total_dist


def gradeAnswer(result: dict, subPass: int, aiEngineName: str) -> tuple:
  if not result or "python_code" not in result:
    return 0.0, "No Python code provided"

  case = TEST_CASES[subPass]
  use_streaming = _should_use_streaming(subPass)

  try:
    if use_streaming:
      t = time.time()
      streaming_input = _get_streaming_input(subPass)
      print(f"  Generating/caching input file for {case['desc']}...")
      input_file_path = streaming_input.generate()
      file_size_mb = streaming_input.get_size_bytes() / (1024 * 1024)
      print(f"  Input file: {file_size_mb:.1f} MB")
      if time.time() - t > 1:
        print(f"  Time to generate: {time.time() - t:.2f}s")

      # Skip verification for very large cases
      if case["customers"] > 100_000:
        start = time.time()
        with open(input_file_path, 'r') as f:
          proc = subprocess.run(["python", "-c", result["python_code"]],
                                stdin=f,
                                capture_output=True,
                                text=True,
                                timeout=TIMEOUT_SECONDS)
        exec_time = time.time() - start
        if proc.returncode == 0:
          lines = proc.stdout.strip().split('\n')
          if lines:
            reported_dist = float(lines[0])
            return 0.8, f"[{case['desc']}] Distance {reported_dist:.1f} in {exec_time:.2f}s (verification skipped)"
        return 0.0, f"Runtime error: {proc.stderr[:200]}"

      start = time.time()
      with open(input_file_path, 'r') as f:
        proc = subprocess.run(["python", "-c", result["python_code"]],
                              stdin=f,
                              capture_output=True,
                              text=True,
                              timeout=TIMEOUT_SECONDS)
      exec_time = time.time() - start
    else:
      customers, demands, depot, num_vehicles, capacity = get_instance(subPass)
      input_data = format_input(customers, demands, depot, num_vehicles, capacity)

      start = time.time()
      proc = subprocess.run(["python", "-c", result["python_code"]],
                            input=input_data,
                            capture_output=True,
                            text=True,
                            timeout=TIMEOUT_SECONDS)
      exec_time = time.time() - start

    if proc.returncode != 0:
      return 0.0, f"Runtime error: {proc.stderr[:200]}"

    lines = proc.stdout.strip().split('\n')
    reported_dist = float(lines[0])
    num_routes = int(lines[1])

    routes = []
    for i in range(2, 2 + num_routes):
      if i < len(lines):
        parts = lines[i].replace("route", "").replace(":", "").split()
        route = [int(x) for x in parts if x.isdigit() or (x.startswith('-') and x[1:].isdigit())]
        routes.append(route)

    valid, actual_dist, msg = verify_routes(customers, demands, depot, capacity, num_vehicles,
                                            routes)

    if not valid:
      return 0.2, f"[{case['desc']}] {msg}"

    nn_dist = nearest_neighbor_distance(customers, demands, depot, capacity)
    ratio = actual_dist / nn_dist if nn_dist > 0 else 1.0
    score = min(1.0, 1.5 - ratio * 0.5)

    return max(
      0.5,
      score), f"[{case['desc']}] Distance {actual_dist:.1f} (NN: {nn_dist:.1f}), {exec_time:.2f}s"

  except subprocess.TimeoutExpired:
    return 0.1, f"[{case['desc']}] Timeout"
  except Exception as e:
    return 0.0, f"[{case['desc']}] Error: {str(e)[:100]}"


def resultToNiceReport(result: dict, subPass: int, aiEngineName: str) -> str:
  if not result:
    return "<p style='color:red'>No result provided</p>"
  case = TEST_CASES[subPass]
  html = f"<h4>Vehicle Routing - {case['desc']}</h4>"
  if "reasoning" in result:
    r = result['reasoning'][:400] + ('...' if len(result.get('reasoning', '')) > 400 else '')
    html += f"<p><strong>Approach:</strong> {r.replace('<', '&lt;').replace('>', '&gt;')}</p>"
  if "python_code" in result:
    code = result["python_code"].replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    html += f"<details><summary>View Python Code ({len(result['python_code'])} chars)</summary><pre>{code}</pre></details>"
  return html


highLevelSummary = """
VRP finds optimal routes for vehicles to service customers.

**Algorithms:**
- **Nearest Neighbor**: Simple greedy heuristic
- **Clarke-Wright Savings**: Merge routes by savings
- **Genetic Algorithms**: Evolutionary optimization
"""


def setup():
  """Pre-generate and cache all streaming input files for parallel test execution."""
  print(f"  Pre-generating streaming input files for {len(TEST_CASES)} test cases...")
  for subpass in range(len(TEST_CASES)):
    if _should_use_streaming(subpass):
      streaming_input = _get_streaming_input(subpass)
      input_path = streaming_input.generate()
      size_mb = streaming_input.get_size_bytes() / (1024 * 1024)
      print(f"    Subpass {subpass}: {size_mb:.1f} MB cached")
