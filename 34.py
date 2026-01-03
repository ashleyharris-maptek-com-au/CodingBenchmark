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
from typing import List, Tuple

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
]

INSTANCE_CACHE = {}


def get_instance(subpass: int):
  if subpass not in INSTANCE_CACHE:
    case = TEST_CASES[subpass]
    customers, demands, depot = generate_vrp(case["customers"], case["vehicles"], case["capacity"],
                                             RANDOM_SEED + subpass)
    INSTANCE_CACHE[subpass] = (customers, demands, depot, case["vehicles"], case["capacity"])
  return INSTANCE_CACHE[subpass]


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
  customers, demands, depot, num_vehicles, capacity = get_instance(subPass)
  input_data = format_input(customers, demands, depot, num_vehicles, capacity)

  try:
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


def output_example_html(score: float, explanation: str, result: dict, subPass: int) -> str:
  case = TEST_CASES[subPass]
  code = result.get("python_code", "").replace("&", "&amp;").replace("<",
                                                                     "&lt;").replace(">", "&gt;")
  color = "green" if score >= 0.8 else "orange" if score >= 0.4 else "red"
  return f'<div class="result"><h4>Subpass {subPass}: {case["desc"]}</h4><p style="color:{color}">Score: {score:.2f}</p><p>{explanation}</p><details><summary>Code</summary><pre>{code}</pre></details></div>'


def output_header_html() -> str:
  return "<h2>Test 34: Vehicle Routing Problem (Python)</h2><p>NP-Hard logistics optimization.</p>"


def output_summary_html(results: list) -> str:
  total = sum(r[0] for r in results)
  return f'<div class="summary"><p>Total: {total:.2f}/{len(results)}</p></div>'
