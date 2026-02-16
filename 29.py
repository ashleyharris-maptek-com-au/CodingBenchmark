"""
Test 29: Maximum Clique (C# Implementation)

The LLM must write C# code to find the largest complete subgraph (clique)
in an undirected graph. This is NP-Hard.

Subpasses increase graph density and size, requiring algorithms like
Bron-Kerbosch with pivoting, branch-and-bound, or MaxSAT formulation.

Solver times out after 5 minutes.
"""

import random
import subprocess
import time
import math
from typing import List, Tuple, Set, Dict, Any
from native_compiler import CSharpCompiler, CompilationError, ExecutionError, describe_this_pc
from solver_utils import StreamingInputFile

title = "Maximum Clique (C#)"
TIMEOUT_SECONDS = 300
RANDOM_SEED = 29292929


def generate_graph(num_vertices: int, edge_prob: float, clique_size: int,
                   seed: int) -> Tuple[List[Tuple[int, int]], List[int]]:
  if clique_size < 1 or clique_size > num_vertices:
    raise ValueError(f"Invalid clique_size {clique_size} for {num_vertices} vertices")

  rng = random.Random(seed)
  clique_nodes = sorted(rng.sample(range(num_vertices), clique_size))
  clique_set = set(clique_nodes)
  non_clique_nodes = [v for v in range(num_vertices) if v not in clique_set]

  forbidden_for = {v: rng.choice(clique_nodes) for v in non_clique_nodes}

  edges: List[Tuple[int, int]] = []

  # Fully connect the planted clique
  for i, u in enumerate(clique_nodes):
    for v in clique_nodes[i + 1:]:
      edges.append((u, v))

  # Random edges between clique and non-clique, excluding each node's forbidden clique vertex
  for v in non_clique_nodes:
    forbidden = forbidden_for[v]
    for c in clique_nodes:
      if c == forbidden:
        continue
      if rng.random() < edge_prob:
        edges.append((min(v, c), max(v, c)))

  # Random edges among non-clique nodes, disallowing edges within the same forbidden group
  for i, u in enumerate(non_clique_nodes):
    forbidden_u = forbidden_for[u]
    for v in non_clique_nodes[i + 1:]:
      if forbidden_u == forbidden_for[v]:
        continue
      if rng.random() < edge_prob:
        edges.append((u, v))

  return edges, clique_nodes


TEST_CASES = [
  {
    "vertices": 20,
    "edge_prob": 0.6,
    "clique_size": 18,
    "desc": "20 vertices, dense"
  },
  {
    "vertices": 40,
    "edge_prob": 0.5,
    "clique_size": 30,
    "desc": "40 vertices, medium density"
  },
  {
    "vertices": 60,
    "edge_prob": 0.45,
    "clique_size": 37,
    "desc": "60 vertices"
  },
  {
    "vertices": 80,
    "edge_prob": 0.4,
    "clique_size": 42,
    "desc": "80 vertices"
  },
  {
    "vertices": 100,
    "edge_prob": 0.35,
    "clique_size": 45,
    "desc": "100 vertices"
  },
  {
    "vertices": 150,
    "edge_prob": 0.3,
    "clique_size": 55,
    "desc": "150 vertices"
  },
  {
    "vertices": 200,
    "edge_prob": 0.25,
    "clique_size": 60,
    "desc": "200 vertices"
  },
  {
    "vertices": 300,
    "edge_prob": 0.2,
    "clique_size": 70,
    "desc": "300 vertices"
  },
  {
    "vertices": 500,
    "edge_prob": 0.15,
    "clique_size": 85,
    "desc": "500 vertices"
  },
  {
    "vertices": 750,
    "edge_prob": 0.12,
    "clique_size": 100,
    "desc": "750 vertices"
  },
  {
    "vertices": 1000,
    "edge_prob": 0.1,
    "clique_size": 110,
    "desc": "1000 vertices, sparse"
  },
  # Ludicrous cases for streaming
  {
    "vertices": 5000,
    "edge_prob": 0.05,
    "clique_size": 260,
    "desc": "5K vertices (~625K edges)"
  },
  {
    "vertices": 10000,
    "edge_prob": 0.03,
    "clique_size": 310,
    "desc": "10K vertices (~1.5M edges)"
  },
  {
    "vertices": 20000,
    "edge_prob": 0.02,
    "clique_size": 410,
    "desc": "20K vertices (~4M edges)"
  },
  {
    "vertices": 50000,
    "edge_prob": 0.01,
    "clique_size": 510,
    "desc": "50K vertices (~12.5M edges)"
  },
  {
    "vertices": 100000,
    "edge_prob": 0.004,
    "clique_size": 410,
    "desc": "100K vertices (~20M edges)"
  },
  {
    "vertices": 200000,
    "edge_prob": 0.003,
    "clique_size": 610,
    "desc": "200K vertices (~60M edges)"
  },
  {
    "vertices": 500000,
    "edge_prob": 0.001,
    "clique_size": 510,
    "desc": "500K vertices (~125M edges, >1GB)"
  },
]

GRAPH_CACHE: Dict[int, Any] = {}
_INPUT_FILE_CACHE: Dict[int, StreamingInputFile] = {}
STREAMING_THRESHOLD_EDGES = 1_000_000
LAST_CLIQUE_VIZ: Dict[Tuple[int, str], dict] = {}


def get_graph(subpass: int) -> Tuple[int, List[Tuple[int, int]], List[int]]:
  if subpass not in GRAPH_CACHE:
    case = TEST_CASES[subpass]
    edges, clique_nodes = generate_graph(case["vertices"], case["edge_prob"],
                                         case["clique_size"], RANDOM_SEED + subpass)
    GRAPH_CACHE[subpass] = (case["vertices"], edges, clique_nodes)
  return GRAPH_CACHE[subpass]


def _estimate_edges(subpass: int) -> int:
  case = TEST_CASES[subpass]
  n = case["vertices"]
  p = case["edge_prob"]
  k = case["clique_size"]
  clique_edges = k * (k - 1) / 2
  outside = n - k
  if outside <= 0:
    return int(clique_edges)
  between = outside * max(k - 1, 0) * p
  if k <= 1:
    outside_pairs_allowed = 0
  else:
    outside_pairs = outside * (outside - 1) / 2
    outside_pairs_allowed = outside_pairs * (1 - 1 / k)
  return int(clique_edges + between + outside_pairs_allowed * p)


def _should_use_streaming(subpass: int) -> bool:
  return _estimate_edges(subpass) > STREAMING_THRESHOLD_EDGES


def _get_streaming_input(subpass: int) -> StreamingInputFile:
  if subpass in _INPUT_FILE_CACHE:
    return _INPUT_FILE_CACHE[subpass]

  case = TEST_CASES[subpass]
  cache_key = (
    f"graph29|v={case['vertices']}|p={case['edge_prob']}|k={case['clique_size']}|"
    f"seed={RANDOM_SEED + subpass}"
  )

  def generator():
    num_vertices, edges, _ = get_graph(subpass)
    yield f"{num_vertices} {len(edges)}\n"
    for u, v in edges:
      yield f"{u} {v}\n"

  input_file = StreamingInputFile(cache_key, generator, "test29_graphs")
  _INPUT_FILE_CACHE[subpass] = input_file
  return input_file


def format_input(num_vertices: int, edges: List[Tuple[int, int]]) -> str:
  lines = [f"{num_vertices} {len(edges)}"]
  for u, v in edges:
    lines.append(f"{u} {v}")
  return "\n".join(lines)


def prepareSubpassPrompt(subPass: int) -> str:
  """Generate the prompt for subpass 0 that handles all clique complexities."""
  if subPass != 0:
    raise StopIteration

  return f"""You are writing C# code to solve the Maximum Clique problem, which is NP Hard.

Your C# maximum clique finder will be tested with graphs ranging from sparse to very dense instances. 
The same algorithm must work efficiently across ALL graph complexities, from 20 vertices to 1 million.

**Problem:**
Find the largest complete subgraph (clique) in an undirected graph. 
A clique is a set of vertices where every pair of vertices is connected by an edge.

**Input format (stdin):**
```
num_vertices num_edges
u v  (for each edge, 0-indexed)
```

**Output format (stdout):**
```
clique_size
vertex_1 vertex_2 ... vertex_k  (vertices in the maximum clique, space-separated)
```

**Success Criteria:**
- All vertices in output form a complete subgraph
- No larger clique exists in the graph
- Output format is correct

**Failure Criteria:**
- Output vertices don't form a clique
- Larger clique exists
- Timeout without solution
- Invalid output format

**Requirements:**
1. Program must compile with .NET/csc
2. Read from stdin, write to stdout
3. Handle variable graph sizes and densities
4. Complete within 5 minutes 
5. Must handle varying graph complexities efficiently

Hints:
- You may wish to drop to a hueristic if time is running out for partial score.
- Use threads to process in parallel.

**Environment:**
{describe_this_pc()}

**C# Compiler:**
{CSharpCompiler("test_engine").describe()}

Write complete, compilable C# code with a Main method.
"""


extraGradeAnswerRuns = list(range(len(TEST_CASES)))

structure = {
  "type": "object",
  "properties": {
    "reasoning": {
      "type":
      "string",
      "description":
      "Explain your algorithm approach and how it adapts to different clique complexities"
    },
    "csharp_code": {
      "type": "string",
      "description": "Complete C# code with Main method that handles all scales"
    }
  },
  "required": ["reasoning", "csharp_code"],
  "additionalProperties": False
}


def verify_clique(num_vertices: int, edges: List[Tuple[int, int]],
                  clique: List[int]) -> Tuple[bool, str]:
  edge_set = set((min(u, v), max(u, v)) for u, v in edges)

  for v in clique:
    if v < 0 or v >= num_vertices:
      return False, f"Invalid vertex {v}"

  for i, u in enumerate(clique):
    for v in clique[i + 1:]:
      if (min(u, v), max(u, v)) not in edge_set:
        return False, f"Vertices {u} and {v} not connected"

  return True, "Valid clique"


def greedy_clique_size(num_vertices: int, edges: List[Tuple[int, int]]) -> int:
  adj = [set() for _ in range(num_vertices)]
  for u, v in edges:
    adj[u].add(v)
    adj[v].add(u)

  clique = []
  candidates = list(range(num_vertices))
  candidates.sort(key=lambda v: -len(adj[v]))

  for v in candidates:
    if all(v in adj[u] for u in clique):
      clique.append(v)

  return len(clique)


def gradeAnswer(result: dict, subPass: int, aiEngineName: str) -> tuple:
  if not result or "csharp_code" not in result:
    return 0.0, "No C# code provided"

  case = TEST_CASES[subPass]
  use_streaming = _should_use_streaming(subPass)

  compiler = CSharpCompiler(aiEngineName)
  if not compiler.find_compiler():
    return 0.0, "No C# compiler found"

  try:
    exe_path = compiler.compile(result["csharp_code"])
  except CompilationError as e:
    return 0.0, f"Compilation error: {str(e)[:200]}"

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

      stdout, stderr, exec_time, retcode = compiler.execute(exe_path,
                                                            timeout=TIMEOUT_SECONDS,
                                                            stdin_file=input_file_path)
    else:
      num_vertices, edges, _ = get_graph(subPass)
      input_data = format_input(num_vertices, edges)
      stdout, stderr, exec_time, retcode = compiler.execute(exe_path, input_data, TIMEOUT_SECONDS)

    if retcode != 0:
      return 0.0, f"Runtime error: {stderr[:200]}"

    lines = stdout.strip().split('\n')
    clique_size = int(lines[0])
    clique = list(map(int, lines[1].split())) if len(lines) > 1 else []

    num_vertices, edges, true_clique = get_graph(subPass)
    valid, msg = verify_clique(num_vertices, edges, clique)
    if case["vertices"] <= 1000:
      viz = _build_clique_viz(num_vertices, edges, clique, true_clique, valid, msg)
      LAST_CLIQUE_VIZ[(subPass, aiEngineName)] = viz
    if not valid:
      return 0.0, f"[{case['desc']}] {msg}"

    ratio = len(clique) / len(true_clique) if len(true_clique) > 0 else 1.0
    if ratio < 0.7:
      return 0.0, f"[{case['desc']}] Answered Clique size {len(clique)} (true: {len(true_clique)}), {exec_time:.2f}s"

    return ratio, f"[{case['desc']}] Answered Clique size {len(clique)} (true: {len(true_clique)}), {exec_time:.2f}s"

  except ExecutionError as e:
    return 0.0, f"[{case['desc']}] {str(e)[:100]}"
  except Exception as e:
    return 0.0, f"[{case['desc']}] Error: {str(e)[:100]}"


def resultToNiceReport(result: dict, subPass: int, aiEngineName: str) -> str:
  if not result:
    return "<p style='color:red'>No result provided</p>"
  case = TEST_CASES[subPass]
  html = f"<h4>Maximum Clique - {case['desc']}</h4>"
  if "reasoning" in result:
    r = result['reasoning'][:400] + ('...' if len(result.get('reasoning', '')) > 400 else '')
    html += f"<p><strong>Approach:</strong> {r.replace('<', '&lt;').replace('>', '&gt;')}</p>"
  if "csharp_code" in result:
    code = result["csharp_code"].replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    html += f"<details><summary>View C# Code ({len(result['csharp_code'])} chars)</summary><pre>{code}</pre></details>"
  viz = LAST_CLIQUE_VIZ.get((subPass, aiEngineName))
  if viz and viz.get("num_vertices", 0) <= 200:
    html += _generate_clique_svg(viz)
  return html


def _build_clique_viz(num_vertices: int, edges: List[Tuple[int, int]],
                      clique: List[int], true_clique: List[int],
                      valid: bool, msg: str) -> dict:
  true_set = set(true_clique)
  chosen_set = set(clique)
  return {
    "num_vertices": num_vertices,
    "edges": edges,
    "true_clique": true_clique,
    "chosen": clique,
    "missed": sorted(true_set - chosen_set),
    "extra": sorted(chosen_set - true_set),
    "valid": valid,
    "msg": msg,
  }


def _generate_clique_svg(viz: dict) -> str:
  num_vertices = viz["num_vertices"]
  edges = viz["edges"]
  true_clique = set(viz["true_clique"])
  chosen = set(viz["chosen"])
  missed = set(viz["missed"])
  extra = set(viz["extra"])

  size = 600
  cx = cy = size / 2.0
  inner_r = size * 0.22
  outer_r = size * 0.42

  clique_nodes = [v for v in range(num_vertices) if v in true_clique]
  other_nodes = [v for v in range(num_vertices) if v not in true_clique]

  positions = [None] * num_vertices
  for i, v in enumerate(clique_nodes):
    angle = 2 * math.pi * i / max(1, len(clique_nodes))
    x = cx + inner_r * math.cos(angle)
    y = cy + inner_r * math.sin(angle)
    positions[v] = (x, y)
  for i, v in enumerate(other_nodes):
    angle = 2 * math.pi * i / max(1, len(other_nodes))
    x = cx + outer_r * math.cos(angle)
    y = cy + outer_r * math.sin(angle)
    positions[v] = (x, y)

  if num_vertices <= 50:
    node_r = 5.0
  elif num_vertices <= 200:
    node_r = 4.0
  elif num_vertices <= 500:
    node_r = 3.0
  else:
    node_r = 2.2

  edge_lines = []
  for u, v in edges:
    x1, y1 = positions[u]
    x2, y2 = positions[v]
    in_clique = u in true_clique and v in true_clique
    stroke = "#38bdf8" if in_clique else "#334155"
    opacity = "0.7" if in_clique else "0.22"
    width = "1.1" if in_clique else "0.6"
    edge_lines.append(
      f"<line x1='{x1:.2f}' y1='{y1:.2f}' x2='{x2:.2f}' y2='{y2:.2f}' "
      f"stroke='{stroke}' stroke-width='{width}' stroke-opacity='{opacity}' />"
    )

  node_circles = []
  for v, (x, y) in enumerate(positions):
    if v in true_clique and v in chosen:
      fill = "#22c55e"
      stroke = "#14532d"
    elif v in true_clique and v in missed:
      fill = "#ef4444"
      stroke = "#7f1d1d"
    elif v in extra:
      fill = "#f59e0b"
      stroke = "#78350f"
    else:
      fill = "#475569"
      stroke = "#0f172a"
    node_circles.append(
      f"<circle cx='{x:.2f}' cy='{y:.2f}' r='{node_r:.2f}' fill='{fill}' stroke='{stroke}' stroke-width='0.7' />"
    )

  status = "Valid clique" if viz.get("valid") else viz.get("msg", "Invalid clique")
  svg = "\n".join([
    "<div style='margin:12px 0;padding:10px;border:1px solid #1f2937;border-radius:8px;background:#0b1120;'>",
    f"<div style='color:#e2e8f0;font-size:13px;margin-bottom:6px;'><strong>Graph Visualization</strong> — {status}</div>",
    f"<div style='color:#94a3b8;font-size:12px;margin-bottom:6px;'>"
    f"Nodes: {num_vertices} | Edges: {len(edges)} | True clique: {len(true_clique)} | Reported: {len(chosen)} | Missed: {len(missed)} | Extra: {len(extra)}</div>",
    f"<svg width='100%' viewBox='0 0 {size} {size}' style='background:#0b1120;border:1px solid #334155;border-radius:6px;'>",
    "<g>",
    *edge_lines,
    "</g>",
    "<g>",
    *node_circles,
    "</g>",
    "</svg>",
    "<div style='color:#64748b;font-size:11px;margin-top:6px;'>Green nodes are correctly selected clique vertices. Red nodes are missed clique vertices. Amber nodes are incorrect extras.</div>",
    "</div>",
  ])
  return svg


highLevelSummary = """
Maximum Clique finds the largest complete subgraph in a graph.


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
