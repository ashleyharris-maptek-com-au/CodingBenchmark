"""
Test 32: Steiner Tree (C++ Implementation)

The LLM must write C++ code to find the minimum-weight tree connecting a set
of terminal vertices in a weighted graph. This is NP-Hard.

Subpasses increase graph size and terminal count, requiring Dreyfus-Wagner DP,
approximation algorithms, or ILP formulations.

Solver times out after 5 minutes.
"""

import random
import subprocess
import time
import math
from typing import List, Tuple, Set, Dict, Any
from native_compiler import CppCompiler, CompilationError, ExecutionError, describe_this_pc
from solver_utils import StreamingInputFile

title = "Steiner Tree (C++)"

tags = [
  "cpp",
  "structured response",
  "np hard",
  "graph theory",
]
TIMEOUT_SECONDS = 30
RANDOM_SEED = 32323232


def _generate_planted_meta(num_vertices: int, num_edges: int, num_terminals: int,
                           seed: int) -> Tuple[List[int], List[Tuple[int, int, int]], int, int]:
  """Generate terminals and a planted Steiner tree with known weight."""
  rng = random.Random(seed)
  terminals = rng.sample(range(num_vertices), num_terminals)
  terminal_set = set(terminals)

  non_terminals = [v for v in range(num_vertices) if v not in terminal_set]
  steiner_count = min(len(non_terminals), max(1, num_terminals // 4))
  steiner_nodes = rng.sample(non_terminals, steiner_count) if non_terminals else []

  tree_nodes = terminals + steiner_nodes
  rng.shuffle(tree_nodes)

  planted_edges: List[Tuple[int, int, int]] = []
  if tree_nodes:
    for idx in range(1, len(tree_nodes)):
      v = tree_nodes[idx]
      u = rng.choice(tree_nodes[:idx])
      weight = rng.randint(1, 6)
      planted_edges.append((min(u, v), max(u, v), weight))

  planted_weight = sum(w for _, _, w in planted_edges)
  min_edges = len(planted_edges)
  max_possible = (num_vertices * (num_vertices - 1)) // 2
  target_edges = min(num_edges, max_possible)
  decoy_count = max(0, target_edges - min_edges)

  return terminals, planted_edges, planted_weight, decoy_count


def _build_edges_from_meta(num_vertices: int, planted_edges: List[Tuple[int, int, int]],
                           decoy_count: int, seed: int) -> List[Tuple[int, int, int]]:
  rng = random.Random(seed + 991)
  edges = list(planted_edges)
  planted_set = { (u, v) for u, v, _ in planted_edges }
  edge_set = set(planted_set)

  while len(edges) < len(planted_edges) + decoy_count:
    u = rng.randint(0, num_vertices - 1)
    v = rng.randint(0, num_vertices - 1)
    if u == v:
      continue
    a, b = (u, v) if u < v else (v, u)
    if (a, b) in edge_set:
      continue
    weight = rng.randint(50, 120)
    edges.append((a, b, weight))
    edge_set.add((a, b))

  return edges


def generate_graph(num_vertices: int, num_edges: int, num_terminals: int,
                   seed: int) -> Tuple[List[Tuple[int, int, int]], List[int], List[Tuple[int, int, int]], int, int]:
  """Generate weighted graph with planted Steiner tree + decoy edges."""
  terminals, planted_edges, planted_weight, decoy_count = _generate_planted_meta(
    num_vertices, num_edges, num_terminals, seed
  )
  edges = _build_edges_from_meta(num_vertices, planted_edges, decoy_count, seed)
  return edges, terminals, planted_edges, planted_weight, decoy_count


TEST_CASES = [
  {
    "vertices": 20,
    "edges": 40,
    "terminals": 5,
    "desc": "20V, 5 terminals"
  },
  {
    "vertices": 50,
    "edges": 120,
    "terminals": 8,
    "desc": "50V, 8 terminals"
  },
  {
    "vertices": 100,
    "edges": 300,
    "terminals": 12,
    "desc": "100V, 12 terminals"
  },
  {
    "vertices": 200,
    "edges": 800,
    "terminals": 15,
    "desc": "200V, 15 terminals"
  },
  {
    "vertices": 500,
    "edges": 2500,
    "terminals": 20,
    "desc": "500V, 20 terminals"
  },
  {
    "vertices": 1000,
    "edges": 6000,
    "terminals": 25,
    "desc": "1KV, 25 terminals"
  },
  {
    "vertices": 2000,
    "edges": 15000,
    "terminals": 30,
    "desc": "2KV, 30 terminals"
  },
  {
    "vertices": 5000,
    "edges": 40000,
    "terminals": 40,
    "desc": "5KV, 40 terminals"
  },
  {
    "vertices": 10000,
    "edges": 100000,
    "terminals": 50,
    "desc": "10KV, 50 terminals"
  },
  {
    "vertices": 20000,
    "edges": 250000,
    "terminals": 75,
    "desc": "20KV, 75 terminals"
  },
  {
    "vertices": 50000,
    "edges": 750000,
    "terminals": 100,
    "desc": "50KV, 100 terminals"
  },
  # Ludicrous cases for streaming
  {
    "vertices": 100000,
    "edges": 2000000,
    "terminals": 150,
    "desc": "100KV, 2M edges"
  },
  {
    "vertices": 500000,
    "edges": 10000000,
    "terminals": 200,
    "desc": "500KV, 10M edges"
  },
  {
    "vertices": 1000000,
    "edges": 30000000,
    "terminals": 300,
    "desc": "1MV, 30M edges"
  },
  {
    "vertices": 2000000,
    "edges": 80000000,
    "terminals": 500,
    "desc": "2MV, 80M edges (~1GB)"
  },
]

INSTANCE_CACHE: Dict[int, Any] = {}
_INPUT_FILE_CACHE: Dict[int, StreamingInputFile] = {}
LAST_STEINER_VIZ: Dict[Tuple[int, str], dict] = {}
STREAMING_THRESHOLD_EDGES = 1_000_000


def get_instance(subpass: int, include_edges: bool = True):
  if subpass not in INSTANCE_CACHE:
    case = TEST_CASES[subpass]
    terminals, planted_edges, planted_weight, decoy_count = _generate_planted_meta(
      case["vertices"], case["edges"], case["terminals"], RANDOM_SEED + subpass
    )
    INSTANCE_CACHE[subpass] = {
      "num_vertices": case["vertices"],
      "edges": None,
      "terminals": terminals,
      "planted_edges": planted_edges,
      "planted_weight": planted_weight,
      "decoy_count": decoy_count,
    }

  cached = INSTANCE_CACHE[subpass]
  if include_edges and cached["edges"] is None:
    cached["edges"] = _build_edges_from_meta(
      cached["num_vertices"], cached["planted_edges"], cached["decoy_count"],
      RANDOM_SEED + subpass
    )

  return (
    cached["num_vertices"],
    cached["edges"],
    cached["terminals"],
    cached["planted_edges"],
    cached["planted_weight"],
    cached["decoy_count"],
  )


def _should_use_streaming(subpass: int) -> bool:
  return TEST_CASES[subpass]["edges"] > STREAMING_THRESHOLD_EDGES


def _get_streaming_input(subpass: int) -> StreamingInputFile:
  if subpass in _INPUT_FILE_CACHE:
    return _INPUT_FILE_CACHE[subpass]

  case = TEST_CASES[subpass]
  cache_key = f"steiner32|v={case['vertices']}|e={case['edges']}|t={case['terminals']}|seed={RANDOM_SEED + subpass}"

  def generator():
    num_vertices, _, terminals, planted_edges, _, decoy_count = get_instance(
      subpass, include_edges=False
    )
    total_edges = len(planted_edges) + decoy_count
    yield f"{num_vertices} {total_edges} {len(terminals)}\n"
    for u, v, w in planted_edges:
      yield f"{u} {v} {w}\n"

    planted_set = {(u, v) for u, v, _ in planted_edges}
    rng = random.Random(RANDOM_SEED + subpass + 991)
    count = 0
    u = 0
    v = 1
    while count < decoy_count:
      if u >= num_vertices - 1:
        u = 0
        v = 1
      if v >= num_vertices:
        u += 1
        v = u + 1
        continue
      if (u, v) not in planted_set:
        weight = rng.randint(50, 120)
        yield f"{u} {v} {weight}\n"
        count += 1
      v += 1

    yield " ".join(map(str, terminals)) + "\n"

  input_file = StreamingInputFile(cache_key, generator, "test32_steiner")
  _INPUT_FILE_CACHE[subpass] = input_file
  return input_file


def format_input(num_vertices: int, edges: List[Tuple[int, int, int]], terminals: List[int]) -> str:
  lines = [f"{num_vertices} {len(edges)} {len(terminals)}"]
  for u, v, w in edges:
    lines.append(f"{u} {v} {w}")
  lines.append(" ".join(map(str, terminals)))
  return "\n".join(lines)


def prepareSubpassPrompt(subPass: int) -> str:
  """Generate the prompt for subpass 0 that handles all Steiner tree complexities."""
  if subPass != 0:
    raise StopIteration

  return f"""You are writing C++ code to solve the Steiner Tree problem (NP-Hard).

**Problem:**
Given a weighted undirected graph and a set of terminal vertices, find a minimum-weight
tree that connects all terminals. The tree may include non-terminal (Steiner) vertices.

**Input (stdin):**
```
num_vertices num_edges num_terminals
u v weight          (one edge per line, 0-indexed)
terminal_1 terminal_2 ... terminal_K
```

**Output (stdout):**
```
total_weight edge_count
u1 v1
u2 v2
...
```
Edges listed must form a connected, acyclic subgraph that includes all terminals.


**Environment:**
{describe_this_pc()}

**C++ Compiler:**
{CppCompiler("test_engine").describe()}

Be sure that any deviation from the C++ standard library is supported by the given compiler,
as referencing the wrong intrinsics or non-standard header like 'bits/stdc++.h' could fail your submission.

**Hints (not required):**
- Exact DP (Dreyfus-Wagner) for small terminal sets.
- MST / shortest-path heuristics for large graphs.
- Use adjacency lists and fast I/O.

Write complete, compilable C++ code with a main() function.
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
      "Explain your algorithm approach and how it adapts to different Steiner tree complexities"
    },
    "cpp_code": {
      "type": "string",
      "description": "Complete C++ code with main() function that handles all scales"
    }
  },
  "required": ["reasoning", "cpp_code"],
  "additionalProperties": False
}


def verify_steiner_tree(num_vertices: int, edges: List[Tuple[int, int, int]], terminals: Set[int],
                        tree_edges: List[Tuple[int, int]]) -> Tuple[bool, int, str]:
  """Verify tree connects all terminals. Returns (valid, weight, message)."""
  edge_weights = {}
  for u, v, w in edges:
    edge_weights[(min(u, v), max(u, v))] = w

  # Build adjacency from tree
  adj = {}
  total_weight = 0
  for u, v in tree_edges:
    key = (min(u, v), max(u, v))
    if key not in edge_weights:
      return False, 0, f"Edge ({u},{v}) not in graph"
    total_weight += edge_weights[key]
    adj.setdefault(u, []).append(v)
    adj.setdefault(v, []).append(u)

  if not tree_edges:
    return len(terminals) <= 1, 0, "Empty tree"

  # BFS to check connectivity
  start = tree_edges[0][0]
  visited = {start}
  queue = [start]
  while queue:
    v = queue.pop(0)
    for u in adj.get(v, []):
      if u not in visited:
        visited.add(u)
        queue.append(u)

  missing = terminals - visited
  if missing:
    return False, total_weight, f"Terminals not connected: {missing}"

  return True, total_weight, "Valid"


def _verify_planted_tree(terminals: Set[int], tree_edges: List[Tuple[int, int]],
                         planted_edges: List[Tuple[int, int, int]],
                         planted_weight: int) -> Tuple[bool, int, str]:
  planted_set = {(min(u, v), max(u, v)) for u, v, _ in planted_edges}
  planted_weights = { (min(u, v), max(u, v)): w for u, v, w in planted_edges }

  if len(tree_edges) != len(set(tree_edges)):
    return False, 0, "Duplicate edges in tree"

  total_weight = 0
  adj: Dict[int, List[int]] = {}
  for u, v in tree_edges:
    key = (min(u, v), max(u, v))
    if key not in planted_set:
      return False, 0, f"Edge ({u},{v}) not in planted tree"
    total_weight += planted_weights[key]
    adj.setdefault(u, []).append(v)
    adj.setdefault(v, []).append(u)

  if not tree_edges:
    return len(terminals) <= 1, 0, "Empty tree"

  start = tree_edges[0][0]
  visited = {start}
  queue = [start]
  while queue:
    v = queue.pop(0)
    for u in adj.get(v, []):
      if u not in visited:
        visited.add(u)
        queue.append(u)

  missing = terminals - visited
  if missing:
    return False, total_weight, f"Terminals not connected: {missing}"

  if total_weight != planted_weight:
    return False, total_weight, "Weight mismatch"

  return True, total_weight, "Valid"


def mst_upper_bound(num_vertices: int, edges: List[Tuple[int, int, int]],
                    terminals: List[int]) -> int:
  """Compute MST on metric closure of terminals as upper bound."""
  from heapq import heappush, heappop

  # Dijkstra from each terminal
  adj = [[] for _ in range(num_vertices)]
  for u, v, w in edges:
    adj[u].append((v, w))
    adj[v].append((u, w))

  def dijkstra(start):
    dist = [float('inf')] * num_vertices
    dist[start] = 0
    pq = [(0, start)]
    while pq:
      d, u = heappop(pq)
      if d > dist[u]:
        continue
      for v, w in adj[u]:
        if dist[u] + w < dist[v]:
          dist[v] = dist[u] + w
          heappush(pq, (dist[v], v))
    return dist

  # Build metric closure
  term_dists = {t: dijkstra(t) for t in terminals}

  # MST on terminals
  if len(terminals) <= 1:
    return 0

  in_mst = {terminals[0]}
  mst_weight = 0
  while len(in_mst) < len(terminals):
    best = float('inf')
    best_t = None
    for t in terminals:
      if t not in in_mst:
        for s in in_mst:
          if term_dists[s][t] < best:
            best = term_dists[s][t]
            best_t = t
    if best_t:
      in_mst.add(best_t)
      mst_weight += best

  return mst_weight


def gradeAnswer(result: dict, subPass: int, aiEngineName: str) -> tuple:
  if not result or "cpp_code" not in result:
    return 0.0, "No C++ code provided"

  case = TEST_CASES[subPass]
  use_streaming = _should_use_streaming(subPass)

  compiler = CppCompiler(aiEngineName)
  if not compiler.find_compiler():
    return 0.0, "No C++ compiler found"

  t1 = time.time()

  try:
    exe_path = compiler.compile(result["cpp_code"])
  except CompilationError as e:
    return 0.0, f"Compilation error: {str(e)[:200]}"

  d = time.time() - t1
  if d > 1: print(f"  Compilation time: {d:.2f}s")

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

      start = time.time()
      stdout, stderr, exec_time, return_code = compiler.execute(exe_path,
                                                                timeout=TIMEOUT_SECONDS,
                                                                stdin_file=input_file_path)

      if return_code != 0:
        return 0.0, f"Runtime error: {stderr[:200]}"

      proc_stdout = stdout
    else:
      num_vertices, edges, terminals, _, _, _ = get_instance(subPass)
      input_data = format_input(num_vertices, edges, terminals)

      start = time.time()
      proc = subprocess.run([str(exe_path)],
                            input=input_data,
                            capture_output=True,
                            text=True,
                            timeout=TIMEOUT_SECONDS)
      exec_time = time.time() - start

      if proc.returncode != 0:
        return 0.0, f"Runtime error: {proc.stderr[:200]}"
      proc_stdout = proc.stdout

    lines = proc_stdout.strip().split('\n')
    header = lines[0].split()
    reported_weight = int(header[0])
    num_tree_edges = int(header[1])
    num_vertices, edges, terminals, planted_edges, planted_weight, _ = get_instance(subPass)

    tree_edges = []
    for i in range(1, num_tree_edges + 1):
      if i < len(lines):
        parts = lines[i].split()
        tree_edges.append((int(parts[0]), int(parts[1])))

    t = time.time()
    if use_streaming:
      valid, actual_weight, msg = _verify_planted_tree(
        set(terminals), tree_edges, planted_edges, planted_weight
      )
    else:
      valid, actual_weight, msg = verify_steiner_tree(num_vertices, edges, set(terminals), tree_edges)
    d = time.time() - t
    if d > 1:
      print(f"  Verification time: {d:.2f}s")

    if num_vertices <= 150 and not use_streaming:
      LAST_STEINER_VIZ[(subPass, aiEngineName)] = _build_steiner_viz(
        num_vertices, edges, terminals, tree_edges, reported_weight, actual_weight, valid, msg
      )

    if not valid:
      return 0.2, f"[{case['desc']}] {msg}"

    if use_streaming:
      return 1.0, f"[{case['desc']}] Weight {actual_weight}, {exec_time:.2f}s"

    upper = mst_upper_bound(num_vertices, edges, terminals)
    ratio = actual_weight / upper if upper > 0 else 1.0
    score = min(1.0, 1.5 - ratio * 0.5)

    return max(
      0.5, score), f"[{case['desc']}] Weight {actual_weight} (MST bound: {upper}), {exec_time:.2f}s"

  except subprocess.TimeoutExpired:
    return 0.0, f"[{case['desc']}] Timeout"
  except Exception as e:
    return 0.0, f"[{case['desc']}] Error: {str(e)[:100]}"


def resultToNiceReport(result: dict, subPass: int, aiEngineName: str) -> str:
  if not result:
    return "<p style='color:red'>No result provided</p>"
  case = TEST_CASES[subPass]
  html = f"<h4>Steiner Tree - {case['desc']}</h4>"
  if "reasoning" in result:
    r = result['reasoning'][:400] + ('...' if len(result.get('reasoning', '')) > 400 else '')
    html += f"<p><strong>Approach:</strong> {r.replace('<', '&lt;').replace('>', '&gt;')}</p>"
  if "cpp_code" in result:
    code = result["cpp_code"].replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    html += f"<details><summary>View C++ Code ({len(result['cpp_code'])} chars)</summary><pre>{code}</pre></details>"
  viz = LAST_STEINER_VIZ.get((subPass, aiEngineName))
  if viz:
    html += _generate_steiner_svg(viz)
  return html


def _build_steiner_viz(num_vertices: int, edges: List[Tuple[int, int, int]],
                       terminals: List[int], tree_edges: List[Tuple[int, int]],
                       reported_weight: int, actual_weight: int, valid: bool,
                       msg: str) -> dict:
  terminal_set = set(terminals)
  tree_set = {tuple(sorted(edge)) for edge in tree_edges}
  edge_lookup = {}
  for u, v, w in edges:
    edge_lookup[(min(u, v), max(u, v))] = w

  return {
    "num_vertices": num_vertices,
    "edges": edges,
    "terminals": terminals,
    "tree_edges": list(tree_set),
    "reported_weight": reported_weight,
    "actual_weight": actual_weight,
    "valid": valid,
    "message": msg,
    "edge_weights": edge_lookup,
    "terminal_set": terminal_set,
  }


def _generate_steiner_svg(viz: dict) -> str:
  num_vertices = viz["num_vertices"]
  terminals = viz["terminal_set"]
  tree_edges = set(viz["tree_edges"])
  edges = viz["edges"]

  width = 720
  height = 520
  cx = width / 2
  cy = height / 2
  radius = min(width, height) * 0.36

  positions = {}
  for i in range(num_vertices):
    angle = (2 * math.pi * i) / max(1, num_vertices)
    positions[i] = (cx + radius * math.cos(angle), cy + radius * math.sin(angle))

  edge_lines = []
  for u, v, _ in edges:
    x1, y1 = positions[u]
    x2, y2 = positions[v]
    edge_lines.append(
      f"<line x1='{x1:.2f}' y1='{y1:.2f}' x2='{x2:.2f}' y2='{y2:.2f}' "
      "stroke='#1f2937' stroke-width='1' opacity='0.35'/>"
    )

  tree_lines = []
  for u, v in tree_edges:
    if u not in positions or v not in positions:
      continue
    x1, y1 = positions[u]
    x2, y2 = positions[v]
    tree_lines.append(
      f"<line x1='{x1:.2f}' y1='{y1:.2f}' x2='{x2:.2f}' y2='{y2:.2f}' "
      "stroke='#22c55e' stroke-width='2.2' opacity='0.9'/>"
    )

  node_circles = []
  for i in range(num_vertices):
    x, y = positions[i]
    is_terminal = i in terminals
    fill = "#facc15" if is_terminal else "#94a3b8"
    stroke = "#1f2937"
    r = 6 if is_terminal else 4
    node_circles.append(
      f"<circle cx='{x:.2f}' cy='{y:.2f}' r='{r}' fill='{fill}' stroke='{stroke}' stroke-width='1'/>"
    )

  status = "VALID" if viz["valid"] else "INVALID"
  status_color = "#22c55e" if viz["valid"] else "#f97316"

  legend = (
    "<g font-family='sans-serif' font-size='11' fill='#cbd5f5'>"
    "<rect x='14' y='14' width='12' height='12' fill='#22c55e'/><text x='32' y='24'>Tree edge</text>"
    "<rect x='14' y='32' width='12' height='12' fill='#1f2937' opacity='0.7'/>"
    "<text x='32' y='42'>Graph edge</text>"
    "<circle cx='20' cy='56' r='5' fill='#facc15' stroke='#1f2937' stroke-width='1'/>"
    "<text x='32' y='60'>Terminal</text>"
    "<circle cx='20' cy='74' r='4' fill='#94a3b8' stroke='#1f2937' stroke-width='1'/>"
    "<text x='32' y='78'>Steiner/other</text>"
    "</g>"
  )

  return (
    "<div style='margin:12px 0;padding:10px;border:1px solid #1f2937;"
    "border-radius:8px;background:#0b1120;'>"
    f"<div style='color:#e2e8f0;font-size:13px;margin-bottom:6px;'>"
    f"<strong>Steiner Tree Visualization</strong> &mdash; "
    f"<span style='color:{status_color};'>{status}</span> "
    f"(reported={viz['reported_weight']}, actual={viz['actual_weight']})</div>"
    f"<div style='color:#94a3b8;font-size:11px;margin-bottom:6px;'>{viz['message']}</div>"
    f"<svg width='100%' viewBox='0 0 {width} {height}' "
    "style='background:#0b1120;border:1px solid #334155;border-radius:6px;'>"
    + "".join(edge_lines)
    + "".join(tree_lines)
    + "".join(node_circles)
    + legend
    + "</svg>"
    + "</div>"
  )


highLevelSummary = """
<p>Connect a set of important nodes in a weighted graph using the cheapest possible
tree of edges. Unlike a minimum spanning tree, the solution may pass through
intermediate nodes that aren&rsquo;t in the required set if doing so saves cost
&mdash; like building a road network that connects key cities via junctions.</p>
<p>This is NP-hard. The test uses planted instances with a known optimal tree for
strict grading. Subpasses increase the graph size.</p>
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
