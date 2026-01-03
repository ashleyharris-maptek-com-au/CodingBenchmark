import math
import random
import subprocess
import sys
import tempfile
import os
import time
from collections import defaultdict

title = "Graph Layout - Edge Crossing Minimization"

# Seed for reproducible graph generation
RANDOM_SEED = 54321

# Timeout in seconds (0.5 minutes)
TIMEOUT_SECONDS = 30

# Graph configs: (num_nodes, edges_description, is_planar_possible)
# We'll use a mix of planar graphs (can have 0 crossings) and non-planar
GRAPH_CONFIGS = [
  (6, "tree", True),  # Simple tree - trivially planar
  (8, "cycle_with_chords", True),  # Cycle with some chords - planar
  (10, "grid", True),  # 2x5 grid - planar
  (12, "dense_planar", True),  # Dense but still planar
  (15, "k5_subdivision", False),  # K5 subdivision - not planar, but 1-planar
  (20, "random_dense", False),  # Random dense graph - likely many crossings
  (100, "random_dense", False),  # Extreme 1
  (500, "random_dense", False),  # Extreme 2
  (1000, "random_dense", False),  # Extreme 3
  (5000, "random_dense", False),  # Extreme 4
  (10000, "random_dense", False),  # Extreme 5 - 10k nodes
  (20000, "random_dense", False),  # Extreme 6 - 20k nodes
  (50000, "random_dense", False),  # Extreme 7 - 50k nodes
  (100000, "random_dense", False),  # Extreme 8 - 100k nodes
  (200000, "random_dense", False),  # Extreme 9 - 200k nodes
]


def generate_tree(n: int, rng: random.Random) -> list:
  """Generate a random tree with n nodes."""
  edges = []
  for i in range(1, n):
    parent = rng.randint(0, i - 1)
    weight = rng.randint(1, 10)
    edges.append((parent, i, weight))
  return edges


def generate_cycle_with_chords(n: int, rng: random.Random) -> list:
  """Generate a cycle with some non-crossing chords."""
  edges = []
  # Create cycle
  for i in range(n):
    edges.append((i, (i + 1) % n, 1))
  # Add some chords (carefully to keep it planar)
  for i in range(0, n - 3, 3):
    edges.append((i, i + 2, 1))
  return edges


def generate_grid(rows: int, cols: int) -> list:
  """Generate a grid graph."""
  edges = []
  for r in range(rows):
    for c in range(cols):
      node = r * cols + c
      if c < cols - 1:
        edges.append((node, node + 1, 1))
      if r < rows - 1:
        edges.append((node, node + cols, 1))
  return edges


def generate_dense_planar(n: int, rng: random.Random) -> list:
  """Generate a dense planar graph (maximal planar = triangulation)."""
  # Start with a triangle
  edges = [(0, 1, 1), (1, 2, 1), (2, 0, 1)]
  edge_set = {(0, 1), (1, 0), (1, 2), (2, 1), (2, 0), (0, 2)}

  # Add nodes one by one, connecting to 3 existing nodes
  for i in range(3, n):
    # Pick 3 connected nodes to form a face
    existing = list(range(i))
    rng.shuffle(existing)
    connected = existing[:min(3, len(existing))]
    for j in connected:
      if (i, j) not in edge_set:
        edges.append((i, j, 1))
        edge_set.add((i, j))
        edge_set.add((j, i))
  return edges


def generate_k5_subdivision(n: int, rng: random.Random) -> list:
  """Generate K5 with subdivided edges - non-planar but 1-planar."""
  # K5 on first 5 nodes
  edges = []
  for i in range(5):
    for j in range(i + 1, 5):
      edges.append((i, j, 1))

  # Add more nodes as subdivisions
  node_id = 5
  while node_id < n:
    # Subdivide a random edge
    if edges:
      idx = rng.randint(0, len(edges) - 1)
      a, b, w = edges[idx]
      edges[idx] = (a, node_id, 1)
      edges.append((node_id, b, 1))
      node_id += 1
  return edges


def generate_random_dense(n: int, rng: random.Random) -> list:
  """Generate a random dense graph."""
  edges = []
  edge_set = set()

  # Ensure connectivity first
  for i in range(1, n):
    parent = rng.randint(0, i - 1)
    edges.append((parent, i, 1))
    edge_set.add(tuple(sorted([parent, i])))

  # Add random edges up to ~2n edges total
  target = min(2 * n, n * (n - 1) // 4)
  attempts = 0
  while len(edges) < target and attempts < 500:
    a = rng.randint(0, n - 1)
    b = rng.randint(0, n - 1)
    if a != b:
      edge = tuple(sorted([a, b]))
      if edge not in edge_set:
        edges.append((a, b, 1))
        edge_set.add(edge)
    attempts += 1
  return edges


def generate_graph(config_idx: int) -> tuple:
  """Generate graph for given config index. Returns (num_nodes, edges)."""
  rng = random.Random(RANDOM_SEED + config_idx * 1000)

  n, graph_type, _ = GRAPH_CONFIGS[config_idx]

  if graph_type == "tree":
    edges = generate_tree(n, rng)
  elif graph_type == "cycle_with_chords":
    edges = generate_cycle_with_chords(n, rng)
  elif graph_type == "grid":
    # Make a grid close to n nodes
    rows = int(math.sqrt(n))
    cols = (n + rows - 1) // rows
    edges = generate_grid(rows, cols)
    n = rows * cols  # Actual node count
  elif graph_type == "dense_planar":
    edges = generate_dense_planar(n, rng)
  elif graph_type == "k5_subdivision":
    edges = generate_k5_subdivision(n, rng)
  elif graph_type == "random_dense":
    edges = generate_random_dense(n, rng)
  else:
    edges = generate_tree(n, rng)

  return n, edges


GRAPHS_CACHE = {}


def format_edges_for_prompt(edges: list) -> str:
  """Format edges as a list."""
  lines = ["["]
  for a, b, w in edges[:50]:  # Limit display
    lines.append(f"    ({a}, {b}),")
  if len(edges) > 50:
    lines.append(f"    # ... {len(edges) - 50} more edges ...")
  lines.append("]")
  return "\n".join(lines)


def prepareSubpassPrompt(subPass: int) -> str:
  """Generate the prompt for subpass 0 that handles all graph sizes."""
  if subPass != 0:
    raise StopIteration

  return f"""You are solving a Graph Layout problem (Edge Crossing Minimization).

You must write a Python solver that can handle ANY graph size from trivial to ludicrous scale:
- **Trivial**: 6-15 nodes, small trees and cycles (planar, 0 crossings achievable)
- **Medium**: 20-100 nodes, dense planar and non-planar graphs
- **Large**: 500-1000 nodes, random dense graphs (many crossings unavoidable)
- **Extreme**: 5000-10000 nodes, very large random graphs

**The Challenge:**
Your `layout_graph(num_nodes, edges)` function will be tested with graphs ranging from 6 nodes to 10,000 nodes. The same function must work efficiently across ALL scales.

**Input:**
- `num_nodes`: Number of nodes in the undirected graph (labeled 0 to num_nodes-1)
- `edges`: List of (node1, node2) tuples (unweighted edges)

**Output:**
- List of (x, y) coordinate tuples, one per node
- Coordinates can be any floats
- Goal: Minimize edge crossings when edges are drawn as straight lines

**Critical Requirements:**
1. **Scalability**: Your algorithm must adapt based on graph size and density
2. **Performance**: Must complete within 30 seconds even for 10,000 node graphs
3. **Quality**: Should produce reasonable layouts with minimal crossings
4. **Robustness**: Handle disconnected components, self-loops, and parallel edges gracefully

**Constraints:**
- Use only Python standard library
- Must handle edge cases (empty graphs, single nodes)
- Layout should be valid (coordinates for all nodes)

Write complete, runnable Python code with the layout_graph(num_nodes, edges) function.
"""


# List of subpasses to grade the single answer against all difficulty levels
extraGradeAnswerRuns = list(range(1, len(GRAPH_CONFIGS)))

structure = {
  "type": "object",
  "properties": {
    "reasoning": {
      "type": "string",
      "description": "Explain your layout approach and how it adapts to different graph sizes"
    },
    "python_code": {
      "type":
      "string",
      "description":
      "Complete Python code with layout_graph(num_nodes, edges) function that handles all scales"
    }
  },
  "required": ["reasoning", "python_code"],
  "additionalProperties": False
}


def segments_intersect(p1, p2, p3, p4) -> bool:
  """
    Check if line segment (p1, p2) intersects with (p3, p4).
    Returns True if they cross (not just touch at endpoints).
    """

  def ccw(A, B, C):
    return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])

  def on_segment(p, q, r):
    return (min(p[0], r[0]) <= q[0] <= max(p[0], r[0])
            and min(p[1], r[1]) <= q[1] <= max(p[1], r[1]))

  # Check if segments share an endpoint (not a crossing)
  endpoints = {p1, p2, p3, p4}
  if len(endpoints) < 4:
    return False

  d1 = ccw(p1, p2, p3) != ccw(p1, p2, p4)
  d2 = ccw(p3, p4, p1) != ccw(p3, p4, p2)

  return d1 and d2


def count_crossings(positions: list, edges: list, max_edges_to_check: int = 1000) -> int:
  """Count the number of edge crossings in a layout with sampling for large graphs."""
  crossings = 0
  edge_list = [(a, b) for a, b, *_ in edges]

  # For very large graphs, sample edges to estimate crossings
  if len(edge_list) > max_edges_to_check:
    import random
    # Sample edges for faster estimation
    sampled_indices = random.sample(range(len(edge_list)), max_edges_to_check)
    sampled_edges = [edge_list[i] for i in sampled_indices]

    for i, (a1, b1) in enumerate(sampled_edges):
      for j, (a2, b2) in enumerate(sampled_edges):
        if j <= i:
          continue
        # Skip if edges share a vertex
        if a1 == a2 or a1 == b2 or b1 == a2 or b1 == b2:
          continue

        p1, p2 = positions[a1], positions[b1]
        p3, p4 = positions[a2], positions[b2]

        if segments_intersect(p1, p2, p3, p4):
          crossings += 1

    # Scale up the estimate
    scaling_factor = (len(edge_list) * (len(edge_list) - 1) / 2) / (max_edges_to_check *
                                                                    (max_edges_to_check - 1) / 2)
    estimated_crossings = int(crossings * scaling_factor)
    return estimated_crossings

  # Full calculation for smaller graphs
  for i, (a1, b1) in enumerate(edge_list):
    for j, (a2, b2) in enumerate(edge_list):
      if j <= i:
        continue
      # Skip if edges share a vertex
      if a1 == a2 or a1 == b2 or b1 == a2 or b1 == b2:
        continue

      p1, p2 = positions[a1], positions[b1]
      p3, p4 = positions[a2], positions[b2]

      if segments_intersect(p1, p2, p3, p4):
        crossings += 1

  return crossings


def get_baseline_crossings(num_nodes: int, edges: list) -> int:
  # Use a simple formula-based baseline instead of actual calculation
  # For dense graphs, approximate crossings as O(E^2)
  # For sparse graphs, use a lower estimate
  edge_count = len(edges)

  # Use a conservative estimate
  return max(1, edge_count // 16)


def validate_layout(num_nodes: int, positions: list) -> tuple:
  """Validate that layout is valid. Returns (is_valid, error_message)."""
  if not isinstance(positions, list):
    return False, f"Positions must be a list, got {type(positions).__name__}"

  if len(positions) != num_nodes:
    return False, f"Expected {num_nodes} positions, got {len(positions)}"

  for i, pos in enumerate(positions):
    if not isinstance(pos, (list, tuple)) or len(pos) != 2:
      return False, f"Position {i} must be (x, y) tuple, got {pos}"
    try:
      x, y = float(pos[0]), float(pos[1])
      if math.isnan(x) or math.isnan(y) or math.isinf(x) or math.isinf(y):
        return False, f"Position {i} has invalid coordinates: {pos}"
    except (TypeError, ValueError):
      return False, f"Position {i} has non-numeric coordinates: {pos}"

  return True, ""


def execute_solver(code: str, num_nodes: int, edges: list, timeout: int = TIMEOUT_SECONDS) -> tuple:
  """Execute the LLM's solver code. Returns (positions, error, exec_time)."""
  from solver_utils import execute_solver_with_data

  edges_simple = [(a, b) for a, b, *_ in edges]

  # Reduce timeout for larger graphs to prevent hanging
  if num_nodes > 1000:
    timeout = min(timeout, 10)  # Max 10 seconds for large graphs
  elif num_nodes > 100:
    timeout = min(timeout, 20)  # Max 20 seconds for medium graphs

  data_dict = {
    'num_nodes': num_nodes,
    'edges': edges_simple,
  }

  result, error, exec_time = execute_solver_with_data(code, data_dict, 'layout_graph', timeout)

  if error:
    return None, error, exec_time

  if not isinstance(result, list):
    return None, f"Invalid output: expected list, got {type(result).__name__}", exec_time

  # Convert back to tuples for consistency
  positions = [tuple(p) for p in result]
  return positions, None, exec_time


lastPositions = None


def gradeAnswer(result: dict, subPass: int, aiEngineName: str) -> tuple:
  """
    Grade the LLM's graph layout solver.
    
    Scoring based on crossing count vs baseline:
    - 1.0: Zero crossings (for planar) or ≤ 10% of baseline
    - 0.85: ≤ 50% of baseline crossings
    - 0.7: ≤ baseline crossings
    - 0.5: > baseline but valid layout
    - 0.0: Invalid layout or error
    """
  global lastPositions, GRAPHS_CACHE
  if not result:
    return 0.0, "No result provided"

  if "python_code" not in result:
    return 0.0, "No Python code provided"

  if subPass not in GRAPHS_CACHE:

    print(f"Lazily generating graph for subPass {subPass}")
    GRAPHS_CACHE[subPass] = generate_graph(subPass)

  num_nodes, edges = GRAPHS_CACHE[subPass]
  _, graph_type, is_planar = GRAPH_CONFIGS[subPass]
  code = result["python_code"]

  # Execute solver
  positions, error, exec_time = execute_solver(code, num_nodes, edges)
  if error:
    return 0.0, f"[{graph_type}, {num_nodes} nodes] {error}"

  lastPositions = positions
  # Validate layout
  is_valid, validation_error = validate_layout(num_nodes, positions)
  if not is_valid:
    return 0.0, f"[{graph_type}, {num_nodes} nodes] Invalid layout: {validation_error}"

  # Count crossings
  t1 = time.time()
  crossings = count_crossings(positions, edges)
  crossing_time = time.time() - t1
  print(f"Crossing count time: {crossing_time:.4f}s")

  baseline_crossings = get_baseline_crossings(num_nodes, edges)

  # Score
  if crossings == 0:
    score = 1.0
    quality = "perfect (0 crossings)"
  elif baseline_crossings == 0:
    # Baseline had 0, but we have some
    score = 0.0
    quality = f"suboptimal ({crossings} crossings, baseline had 0)"
  elif crossings <= baseline_crossings * 0.1:
    score = 1.0
    quality = f"excellent (≤10% of baseline)"
  elif crossings <= baseline_crossings * 0.5:
    score = 0.5
    quality = f"good (≤50% of baseline)"
  elif crossings <= baseline_crossings:
    score = 0.25
    quality = f"acceptable (≤ baseline)"
  else:
    score = 0.0
    quality = f"poor ({crossings} vs {baseline_crossings} baseline)"

  explanation = (f"[{graph_type}, {num_nodes} nodes, {len(edges)} edges] "
                 f"Crossings: {crossings}, Baseline: {baseline_crossings}, "
                 f"Time: {exec_time:.1f}s - {quality}")

  return score, explanation


def resultToNiceReport(result: dict, subPass: int, aiEngineName: str) -> str:
  """Generate HTML report for result."""
  if not result:
    return "<p style='color:red'>No result provided</p>"

  num_nodes, edges = GRAPHS_CACHE[subPass]
  _, graph_type, is_planar = GRAPH_CONFIGS[subPass]

  html = f"<h4>Graph Layout - {graph_type} ({num_nodes} nodes, {len(edges)} edges)</h4>"

  if subPass == 0:
    if "reasoning" in result:
      reasoning = result['reasoning'][:500] + ('...'
                                               if len(result.get('reasoning', '')) > 500 else '')
      html += f"<p><strong>Approach:</strong> {reasoning}</p>"

    if "python_code" in result:
      code = result["python_code"]
      code_escaped = code.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
      html += f"<details><summary>View Code ({len(code)} chars)</summary><pre>{code_escaped}</pre></details>"

  if num_nodes <= 200:
    html += svg_graph(num_nodes, edges, lastPositions)
  else:
    html += "<p style='color:orange'>Graph too large to display (＞200 nodes)</p>"

  return html


def svg_graph(num_nodes: int, edges: list, positions: list) -> str:
  """Generate SVG visualization of the graph layout."""
  if not positions:
    return "<p style='color:red'>Could not render graph: No valid layout generated</p>"

  # Find bounds of positions
  x_coords = [pos[0] for pos in positions]
  y_coords = [pos[1] for pos in positions]
  min_x, max_x = min(x_coords), max(x_coords)
  min_y, max_y = min(y_coords), max(y_coords)

  # Add padding
  padding = 30
  width = 400
  height = 400

  # Scale coordinates to fit SVG
  if max_x != min_x:
    scale_x = (width - 2 * padding) / (max_x - min_x)
  else:
    scale_x = 1

  if max_y != min_y:
    scale_y = (height - 2 * padding) / (max_y - min_y)
  else:
    scale_y = 1

  scale = min(scale_x, scale_y)

  def transform_point(x, y):
    tx = padding + (x - min_x) * scale
    ty = padding + (y - min_y) * scale
    return tx, ty

  # Generate edge lines
  edge_lines = ""
  for a, b, _ in edges:
    x1, y1 = transform_point(positions[a][0], positions[a][1])
    x2, y2 = transform_point(positions[b][0], positions[b][1])
    edge_lines += f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="#666" stroke-width="1" opacity="0.7"/>'

  # Generate node circles
  node_circles = ""
  for i, pos in enumerate(positions):
    x, y = transform_point(pos[0], pos[1])
    color = "#ff4444" if i == 0 else "#4444ff"  # First node in red
    node_circles += f'<circle cx="{x}" cy="{y}" r="4" fill="{color}" stroke="white" stroke-width="1" title="Node {i}"/>'
    node_circles += f'<text x="{x}" y="{y-6}" text-anchor="middle" font-size="9" fill="#333">{i}</text>'

  svg_html = f'''
  <div style="margin: 10px 0; width: 100%">
    <h5>Graph Layout Visualization</h5>
    <svg width="100%" style="border: 1px solid #ccc; background: white;" viewBox="0 0 {width} {height}">
      {edge_lines}
      {node_circles}
    </svg>
    <p style="font-size: 12px; color: #666;">
      <span style="color: #ff4444;">● Node 0</span> | 
      <span style="color: #4444ff;">● Other nodes</span> | 
      Layout: {len(positions)} nodes positioned
    </p>
  </div>'''

  return svg_html


highLevelSummary = """
Graph layout with edge crossing minimization is related to 1-planar graphs.

**Key concepts:**
- Planar graphs can be drawn with 0 crossings
- 1-planar graphs can be drawn with each edge crossed at most once
- General graphs may require many crossings

**Algorithm approaches:**
- Force-directed (Fruchterman-Reingold, spring embeddings)
- Spectral methods (eigenvector-based positioning)
- Tutte's barycentric method (for 3-connected planar graphs)
- Stress majorization

**Complexity:**
- Testing planarity: O(n) - linear time
- Testing 1-planarity: NP-complete
- Minimizing crossings: NP-hard in general

The baseline uses a simple circular layout.
"""
