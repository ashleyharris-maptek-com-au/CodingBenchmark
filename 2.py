import math
import random
import os
import time
from pathlib import Path

from native_compiler import CSharpCompiler, compile_and_run, describe_this_pc
from solver_utils import StreamingInputFile
from collections import defaultdict

title = "Chinese Postman Problem Solver (C#)"

# Seed for reproducible graph generation
RANDOM_SEED = 12345

# Timeout in seconds (30 seconds for testing)
TIMEOUT_SECONDS = 30

# Graph complexity for each subpass: (num_nodes, approx_edges)
GRAPH_CONFIGS = [
  (6, 8),  # Small - easy
  (10, 16),  # Medium
  (15, 28),  # Larger
  (20, 45),  # Complex
  (35, 80),  # Large
  (50, 120),  # Very large
  (200, 600),  # Extreme 1
  (500, 2000),  # Extreme 2
  (1000, 5000),  # Extreme 3
  (5000, 30000),  # Extreme 4
  (10000, 80000),  # Extreme 5 - 10k nodes, 80k edges
  (50000, 500000),  # Epic - 50k nodes, 500k edges
  (100000, 1000000),  # Legendary - 100k nodes, 1M edges
]


def generate_connected_graph(num_nodes: int, num_edges: int, seed: int) -> list:
  """
    Generate a connected undirected weighted graph.
    Returns list of (node1, node2, weight) tuples.
    """
  rng = random.Random(seed)
  edges = []
  edge_set = set()

  # First ensure connectivity with a spanning tree
  nodes = list(range(num_nodes))
  rng.shuffle(nodes)
  connected = {nodes[0]}

  for i in range(1, num_nodes):
    # Connect new node to a random connected node
    new_node = nodes[i]
    existing = rng.choice(list(connected))
    weight = rng.randint(1, 100)
    edge = tuple(sorted([new_node, existing]))
    edges.append((edge[0], edge[1], weight))
    edge_set.add(edge)
    connected.add(new_node)

  # Add remaining edges randomly
  remaining = num_edges - len(edges)
  attempts = 0
  while len(edges) < num_edges and attempts < 1000:
    a = rng.randint(0, num_nodes - 1)
    b = rng.randint(0, num_nodes - 1)
    if a != b:
      edge = tuple(sorted([a, b]))
      if edge not in edge_set:
        weight = rng.randint(1, 100)
        edges.append((edge[0], edge[1], weight))
        edge_set.add(edge)
    attempts += 1

  return edges


# Pre-generate graphs for each subpass
GRAPHS_CACHE = {}
for i, (nodes, edges) in enumerate(GRAPH_CONFIGS):
  if nodes < 1000:
    GRAPHS_CACHE[i] = generate_connected_graph(nodes, edges, RANDOM_SEED + i * 100)


def format_graph_for_prompt(edges: list, num_nodes: int) -> str:
  """Format graph as adjacency list representation for the prompt."""
  lines = ["{"]
  adj = defaultdict(list)
  for a, b, w in edges:
    adj[a].append((b, w))
    adj[b].append((a, w))

  for node in range(num_nodes):
    neighbors = adj[node]
    neighbor_str = ", ".join(f"({n}, {w})" for n, w in sorted(neighbors))
    lines.append(f"    {node}: [{neighbor_str}],")
  lines.append("}")
  return "\n".join(lines)


def format_edges_for_prompt(edges: list) -> str:
  """Format edges as a list of tuples."""
  lines = ["["]
  for a, b, w in edges:
    lines.append(f"    ({a}, {b}, {w}),  # Edge between node {a} and {b}, weight {w}")
  lines.append("]")
  return "\n".join(lines)


def prepareSubpassPrompt(subPass: int) -> str:
  """Generate the prompt for subpass 0 that handles all graph sizes."""
  if subPass != 0:
    raise StopIteration

  return f"""You are solving the Chinese Postman Problem (Route Inspection Problem) in C#.

You must write a C# solver that can handle ANY graph size from trivial to ludicrous scale:
- **Trivial**: 6-20 nodes, 8-45 edges (small graphs, exact algorithms feasible)
- **Medium**: 35-200 nodes, 80-600 edges (requires efficient algorithms)
- **Large**: 500-1000 nodes, 2000-5000 edges (requires optimized implementations)
- **Extreme**: 5000-10000 nodes, 30000-80000 edges (requires very fast algorithms)
- **Epic**: 50000-100000 nodes, 500000-1000000 edges (requires highly optimized algorithms)

**The Challenge:**
Your program will be tested with graphs ranging from 6 nodes to 100000 nodes. The same function must work efficiently across ALL scales.

**Input format (stdin):**
Line 1: N M (number of nodes, number of edges)
Lines 2..M+1: u v w (edge endpoints, 0-indexed, weight integer)

**Output format (stdout):**
A sequence of node indices (whitespace-separated) representing a route that starts at node 0,
traverses EVERY edge at least once, and returns to node 0.
You may optionally prefix with a single integer L (route length). If present, it must match the number of nodes listed.

**Critical Requirements:**
1. **Scalability**: Your algorithm must adapt based on graph size and complexity
2. **Performance**: Must complete within 30 seconds even for large graphs
3. **Correctness**: Must traverse every edge at least once

**Environment:**
{describe_this_pc()}

**C# Compiler:**
{CSharpCompiler("test_engine").describe()}

Write complete, compilable C# code with a static void Main method.
"""


# List of subpasses to grade the single answer against all difficulty levels
extraGradeAnswerRuns = list(range(1, len(GRAPH_CONFIGS)))

structure = {
  "type": "object",
  "properties": {
    "reasoning": {
      "type":
      "string",
      "description":
      "Explain your approach to solving the Chinese Postman Problem and how it adapts to different graph sizes"
    },
    "csharp_code": {
      "type": "string",
      "description": "Complete C# code with Main method that handles all scales"
    }
  },
  "required": ["reasoning", "csharp_code"],
  "additionalProperties": False
}


def build_adjacency(num_nodes: int, edges: list) -> dict:
  """Build adjacency list with edge weights."""
  adj = defaultdict(list)
  for a, b, w in edges:
    adj[a].append((b, w))
    adj[b].append((a, w))
  return adj


def get_edge_weight(edges: list, a: int, b: int) -> int:
  """Get weight of edge between a and b."""
  for e1, e2, w in edges:
    if (e1 == a and e2 == b) or (e1 == b and e2 == a):
      return w
  return float('inf')


def validate_route(num_nodes: int, edges: list, route: list) -> tuple:
  """
  Validate that a route covers all edges.
  Returns (is_valid, error_message, edges_covered).
  """
  if not isinstance(route, list):
    return False, f"Route must be a list, got {type(route).__name__}", set()

  if len(route) < 2:
    return False, "Route must have at least 2 nodes", set()

  if route[0] != 0:
    return False, f"Route must start at node 0, got {route[0]}", set()

  if route[-1] != 0:
    return False, f"Route must end at node 0, got {route[-1]}", set()

  # Check all nodes are valid
  for node in route:
    if not isinstance(node, int) or node < 0 or node >= num_nodes:
      return False, f"Invalid node in route: {node}", set()

  # Build set of edges that need to be covered
  required_edges = set()
  for a, b, _ in edges:
    required_edges.add(tuple(sorted([a, b])))

  # Track which edges are covered by the route
  covered_edges = set()
  adj = build_adjacency(num_nodes, edges)

  for i in range(len(route) - 1):
    a, b = route[i], route[i + 1]
    edge = tuple(sorted([a, b]))

    # Check edge exists in graph
    if edge not in required_edges:
      return False, f"Route uses non-existent edge ({a}, {b})", covered_edges

    covered_edges.add(edge)

  # Check all edges are covered
  missing = required_edges - covered_edges
  if missing:
    missing_list = list(missing)[:5]
    return False, f"Route doesn't cover all edges. Missing: {missing_list}{'...' if len(missing) > 5 else ''}", covered_edges

  return True, "", covered_edges


def calculate_route_distance(edges: list, route: list) -> float:
  """Calculate total distance of a route."""
  edge_weights = {}
  for a, b, w in edges:
    edge_weights[tuple(sorted([a, b]))] = w

  total = 0
  for i in range(len(route) - 1):
    edge = tuple(sorted([route[i], route[i + 1]]))
    total += edge_weights.get(edge, 0)

  return total


def get_baseline_distance(num_nodes: int, edges: list) -> float:
  """
    Get baseline distance using naive greedy approach.
    This is the placebo solver's expected result.
    """
  # Sum of all edge weights (minimum if Eulerian)
  total_weight = sum(w for _, _, w in edges)

  return total_weight * 1.5


STREAMING_THRESHOLD_EDGES = 200_000
_INPUT_FILE_CACHE = {}


def format_input(num_nodes: int, edges: list) -> str:
  lines = [f"{num_nodes} {len(edges)}"]
  for a, b, w in edges:
    lines.append(f"{a} {b} {w}")
  return "\n".join(lines)


def _should_use_streaming(subpass: int) -> bool:
  _, edge_count = GRAPH_CONFIGS[subpass]
  return edge_count > STREAMING_THRESHOLD_EDGES


def _get_streaming_input(subpass: int, edges: list, num_nodes: int) -> StreamingInputFile:
  if subpass in _INPUT_FILE_CACHE:
    return _INPUT_FILE_CACHE[subpass]

  cache_key = f"cpp2|n={num_nodes}|m={len(edges)}|seed={RANDOM_SEED + subpass * 100}"

  def generator():
    yield f"{num_nodes} {len(edges)}\n"
    for a, b, w in edges:
      yield f"{a} {b} {w}\n"

  input_file = StreamingInputFile(cache_key, generator, "test2_cpp")
  _INPUT_FILE_CACHE[subpass] = input_file
  return input_file


def parse_route_output(output: str) -> tuple:
  tokens = output.strip().split()
  if not tokens:
    return None, "Empty output"

  try:
    values = [int(t) for t in tokens]
  except ValueError:
    return None, "Output contains non-integer tokens"

  if len(values) >= 2 and values[0] == len(values) - 1:
    values = values[1:]

  return values, ""


lastRoute = None


def gradeAnswer(result: dict, subPass: int, aiEngineName: str) -> tuple:
  if not result:
    return 0.0, "No result provided"

  if "csharp_code" not in result:
    return 0.0, "No C# code provided"

  global GRAPHS_CACHE
  if subPass not in GRAPHS_CACHE:
    nodes, edges_count = GRAPH_CONFIGS[subPass]
    print(f"Loading graph config {subPass} ({nodes} nodes, {edges_count} edges) into cache...")
    GRAPHS_CACHE[subPass] = generate_connected_graph(nodes, edges_count,
                                                     RANDOM_SEED + subPass * 100)

  num_nodes, _ = GRAPH_CONFIGS[subPass]
  edges = GRAPHS_CACHE[subPass]
  code = result["csharp_code"]

  if _should_use_streaming(subPass):
    streaming_input = _get_streaming_input(subPass, edges, num_nodes)
    input_file_path = streaming_input.generate()
    run = compile_and_run(code,
                          "csharp",
                          aiEngineName,
                          input_file=input_file_path,
                          timeout=TIMEOUT_SECONDS)
  else:
    input_data = format_input(num_nodes, edges)
    run = compile_and_run(code,
                          "csharp",
                          aiEngineName,
                          input_data=input_data,
                          timeout=TIMEOUT_SECONDS)

  if not run:
    return 0.0, f"[{num_nodes} nodes, {len(edges)} edges] {run.error_message()}"

  route, parse_error = parse_route_output(run.stdout)
  exec_time = run.exec_time
  if parse_error:
    return 0.0, f"[{num_nodes} nodes, {len(edges)} edges] {parse_error}"

  global lastRoute
  lastRoute = route

  # Validate the route
  is_valid, validation_error, _ = validate_route(num_nodes, edges, route)
  if not is_valid:
    return 0.0, f"[{num_nodes} nodes, {len(edges)} edges] Invalid route: {validation_error}"

  # Calculate distances
  route_distance = calculate_route_distance(edges, route)
  baseline_distance = get_baseline_distance(num_nodes, edges)

  ratio = route_distance / baseline_distance if baseline_distance > 0 else float('inf')

  # Score based on how close to baseline (or better)
  if ratio <= 1.05:
    score = 1.0
    quality = "excellent (within 5% of baseline)"
  elif ratio <= 1.3:
    score = 0.5
    quality = "good (within 30% of baseline)"
  elif ratio <= 1.5:
    score = 0.25
    quality = "acceptable (within 2x baseline)"
  else:
    score = 0.0
    quality = f"poor ({ratio:.1f}x baseline)"

  explanation = (f"[{num_nodes} nodes, {len(edges)} edges] Route distance: {route_distance:.0f}, "
                 f"Baseline: {baseline_distance:.0f}, "
                 f"Ratio: {ratio:.2f}x, "
                 f"Time: {exec_time:.1f}s - {quality}")

  return score, explanation


def resultToNiceReport(result: dict, subPass: int, aiEngineName: str) -> str:
  """Generate a nice HTML report for the result."""
  if not result:
    return "<p style='color:red'>No result provided</p>"

  num_nodes, _ = GRAPH_CONFIGS[subPass]
  edges = GRAPHS_CACHE[subPass]

  html = ""
  if subPass == 0:

    html += f"<h4>Chinese Postman Problem - {num_nodes} nodes, {len(edges)} edges</h4>"

    if "reasoning" in result:
      reasoning = result['reasoning'][:500] + ('...'
                                               if len(result.get('reasoning', '')) > 500 else '')
      html += f"<p><strong>Approach:</strong> {reasoning}</p>"

    if "csharp_code" in result:
      code = result["csharp_code"]
      code_escaped = code.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
      html += f"<details><summary>View Code ({len(code)} chars)</summary><pre>{code_escaped}</pre></details>"

  # Add SVG rendering of the route
  if lastRoute:
    if len(lastRoute) <= 1000:  # Only render for smaller routes to avoid bloat
      html += generate_route_svg(edges, lastRoute, num_nodes)
    else:
      html += f"<p>Route too large ({len(lastRoute)} nodes) to render SVG</p>"
  else:
    html += "<p style='color:red'>Could not render route: No valid route generated</p>"

  return html


def force_directed_layout(num_nodes: int, edges: list, iterations: int = 50) -> dict:
  """Simple force-directed layout algorithm to minimize edge crossings."""
  import math
  import random
  from collections import defaultdict

  # Initialize positions randomly
  positions = {i: [random.uniform(50, 350), random.uniform(50, 350)] for i in range(num_nodes)}

  # Build adjacency list
  adj = defaultdict(list)
  for a, b, _ in edges:
    adj[a].append(b)
    adj[b].append(a)

  # Force-directed parameters
  k_repulsion = 2000.0  # Repulsion force constant
  k_attraction = 0.05  # Attraction force constant
  cooling = 0.95  # Cooling factor
  temperature = 100.0  # Initial temperature

  for iteration in range(iterations):
    forces = {i: [0.0, 0.0] for i in range(num_nodes)}

    # Repulsive forces between all nodes
    for i in range(num_nodes):
      for j in range(i + 1, num_nodes):
        dx = positions[j][0] - positions[i][0]
        dy = positions[j][1] - positions[i][1]
        dist_sq = dx * dx + dy * dy

        if dist_sq < 1:  # Avoid division by very small numbers
          dist_sq = 1

        # Coulomb's law: F = k/r^2
        force = k_repulsion / dist_sq
        dist = math.sqrt(dist_sq)

        # Apply force in opposite directions
        forces[i][0] -= force * dx / dist
        forces[i][1] -= force * dy / dist
        forces[j][0] += force * dx / dist
        forces[j][1] += force * dy / dist

    # Attractive forces for connected nodes
    for a, b, _ in edges:
      dx = positions[b][0] - positions[a][0]
      dy = positions[b][1] - positions[a][1]
      dist = math.sqrt(dx * dx + dy * dy)

      if dist < 1:
        dist = 1

      # Hooke's law: F = k * distance
      force = k_attraction * dist

      # Apply force towards each other
      forces[a][0] += force * dx / dist
      forces[a][1] += force * dy / dist
      forces[b][0] -= force * dx / dist
      forces[b][1] -= force * dy / dist

    # Update positions with temperature cooling
    for i in range(num_nodes):
      # Limit force magnitude
      force_mag = math.sqrt(forces[i][0]**2 + forces[i][1]**2)
      if force_mag > temperature:
        forces[i][0] = forces[i][0] * temperature / force_mag
        forces[i][1] = forces[i][1] * temperature / force_mag

      positions[i][0] += forces[i][0]
      positions[i][1] += forces[i][1]

      # Keep within bounds
      positions[i][0] = max(20, min(380, positions[i][0]))
      positions[i][1] = max(20, min(380, positions[i][1]))

    # Cool down
    temperature *= cooling

  return positions


def generate_route_svg(edges: list, route: list, node_count: int) -> str:
  """Generate SVG visualization of the Chinese Postman route."""
  if not edges or not route:
    return "<p>No route to visualize</p>"

  # Use force-directed layout for better positioning
  positions = force_directed_layout(node_count, edges, iterations=50)

  # Generate edge lines (all edges in the graph)
  edge_lines = ""
  edge_weights = {}
  for a, b, w in edges:
    x1, y1 = positions[a]
    x2, y2 = positions[b]
    edge_weights[(a, b)] = w
    edge_weights[(b, a)] = w
    edge_lines += f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="#ddd" stroke-width="1" opacity="0.5"/>'

  # Generate route path (highlighted)
  route_path = ""
  for i in range(len(route) - 1):
    a, b = route[i], route[i + 1]
    x1, y1 = positions[a]
    x2, y2 = positions[b]
    # Make route edges thicker and colored
    route_path += f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="#ff4444" stroke-width="2.5" opacity="0.8"/>'

  # Generate node circles
  node_circles = ""
  for i in range(node_count):
    x, y = positions[i]
    color = "#ff4444" if i == 0 else "#4444ff"  # Start node in red
    node_circles += f'<circle cx="{x}" cy="{y}" r="5" fill="{color}" stroke="white" stroke-width="1" title="Node {i}"/>'
    node_circles += f'<text x="{x}" y="{y-8}" text-anchor="middle" font-size="10" fill="#333">{i}</text>'

  svg_html = f'''
  <div style="margin: 10px 0; width: 100%">
    <h5>Route Visualization ({node_count} nodes, {len(edges)} edges)</h5>
    <svg width="100%" style="border: 1px solid #ccc; background: white;" viewBox="0 0 400 400">
      {edge_lines}
      {route_path}
      {node_circles}
    </svg>
    <p style="font-size: 12px; color: #666;">
      <span style="color: #ff4444;">● Start node (0)</span> | 
      <span style="color: #4444ff;">● Other nodes</span> | 
      <span style="color: #ff4444; font-weight: bold;">— Route path</span> | 
      Route length: {len(route)} nodes
    </p>
  </div>'''

  return svg_html


highLevelSummary = """
The Chinese Postman Problem (Route Inspection Problem) asks:
Find the shortest closed walk that traverses every edge of a graph at least once.

**Key concepts:**
- If all vertices have even degree → Eulerian circuit exists (optimal)
- If odd-degree vertices exist → must duplicate some edges

**Algorithm approaches:**
1. Find all odd-degree vertices
2. Find minimum-weight perfect matching of odd vertices
3. Duplicate matched path edges
4. Find Eulerian circuit in the resulting multigraph

The optimal solution uses:
- Dijkstra/Floyd-Warshall for shortest paths between odd vertices
- Minimum-weight perfect matching (Blossom algorithm)
- Hierholzer's algorithm for Eulerian circuit

The baseline uses a greedy matching approach which is suboptimal.
"""
