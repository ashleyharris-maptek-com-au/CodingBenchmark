"""
Test 28: Minimum Vertex Cover (Rust Implementation)

Solver times out after 5 minutes.
"""

import random
import subprocess
import time
import math
import itertools
from typing import List, Tuple, Set, Dict, Any, Optional
from native_compiler import RustCompiler, CompilationError, ExecutionError,describe_this_pc
from solver_utils import StreamingInputFile

title = "Minimum Vertex Cover (Rust)"
TIMEOUT_SECONDS = 30
RANDOM_SEED = 28282828


def generate_graph(num_vertices: int, num_edges: int, seed: int) -> List[Tuple[int, int]]:
  rng = random.Random(seed)
  edges = set()
  while len(edges) < num_edges:
    u = rng.randint(0, num_vertices - 1)
    v = rng.randint(0, num_vertices - 1)
    if u != v:
      edges.add((min(u, v), max(u, v)))
  return list(edges)


TEST_CASES = [
  {
    "vertices": 10,
    "edges": 30,
    "desc": "10 vertices, 30 edges"
  },
  {
    "vertices": 15,
    "edges": 45,
    "desc": "15 vertices, 45 edges"
  },
  {
    "vertices": 20,
    "edges": 40,
    "desc": "20 vertices, 40 edges"
  },
  {
    "vertices": 50,
    "edges": 150,
    "desc": "50 vertices, 150 edges"
  },
  {
    "vertices": 100,
    "edges": 400,
    "desc": "100 vertices, 400 edges"
  },
  {
    "vertices": 200,
    "edges": 1000,
    "desc": "200 vertices, 1K edges"
  },
  {
    "vertices": 500,
    "edges": 3000,
    "desc": "500 vertices, 3K edges"
  },
  {
    "vertices": 1000,
    "edges": 8000,
    "desc": "1K vertices, 8K edges"
  },
  {
    "vertices": 2000,
    "edges": 20000,
    "desc": "2K vertices, 20K edges"
  },
  {
    "vertices": 5000,
    "edges": 60000,
    "desc": "5K vertices, 60K edges"
  },
  {
    "vertices": 10000,
    "edges": 150000,
    "desc": "10K vertices, 150K edges"
  },
  {
    "vertices": 20000,
    "edges": 400000,
    "desc": "20K vertices, 400K edges"
  },
  {
    "vertices": 50000,
    "edges": 1000000,
    "desc": "50K vertices, 1M edges"
  },
  # Ludicrous cases for streaming
  {
    "vertices": 100000,
    "edges": 5000000,
    "desc": "100K vertices, 5M edges"
  },
  {
    "vertices": 500000,
    "edges": 25000000,
    "desc": "500K vertices, 25M edges"
  },
  {
    "vertices": 1000000,
    "edges": 50000000,
    "desc": "1M vertices, 50M edges"
  },
  {
    "vertices": 2000000,
    "edges": 150000000,
    "desc": "2M vertices, 150M edges (>1GB)"
  },
]

GRAPH_CACHE: Dict[int, Any] = {}
_INPUT_FILE_CACHE: Dict[int, StreamingInputFile] = {}
LAST_COVER_VIZ: Dict[Tuple[int, str], dict] = {}
STREAMING_THRESHOLD_EDGES = 1_000_000


def _case_seed(case: dict, subpass: int) -> int:
  return case.get("seed", RANDOM_SEED + case["vertices"] * 1000003 + case["edges"])


def get_graph(subpass: int) -> Tuple[int, List[Tuple[int, int]]]:
  if subpass not in GRAPH_CACHE:
    case = TEST_CASES[subpass]
    seed = _case_seed(case, subpass)
    edges = generate_graph(case["vertices"], case["edges"], seed)
    GRAPH_CACHE[subpass] = (case["vertices"], edges)
  return GRAPH_CACHE[subpass]


def _should_use_streaming(subpass: int) -> bool:
  return TEST_CASES[subpass]["edges"] > STREAMING_THRESHOLD_EDGES


def _get_streaming_input(subpass: int) -> StreamingInputFile:
  if subpass in _INPUT_FILE_CACHE:
    return _INPUT_FILE_CACHE[subpass]

  case = TEST_CASES[subpass]
  seed = _case_seed(case, subpass)
  cache_key = f"graph28|v={case['vertices']}|e={case['edges']}|seed={seed}"

  def generator():
    num_vertices, edges = get_graph(subpass)
    yield f"{num_vertices} {len(edges)}\n"
    for u, v in edges:
      yield f"{u} {v}\n"

  input_file = StreamingInputFile(cache_key, generator, "test28_graphs")
  _INPUT_FILE_CACHE[subpass] = input_file
  return input_file


def format_input(num_vertices: int, edges: List[Tuple[int, int]]) -> str:
  lines = [f"{num_vertices} {len(edges)}"]
  for u, v in edges:
    lines.append(f"{u} {v}")
  return "\n".join(lines)


def prepareSubpassPrompt(subPass: int) -> str:
  """Generate the prompt for subpass 0 that handles all vertex cover complexities."""
  if subPass != 0:
    raise StopIteration

  return f"""You are writing Rust code to solve the Minimum Vertex Cover problem.

Your code may see graphs with up to 2M vertices and 15M edges.

**Problem:**
Find the minimum set of vertices such that every edge has at least one endpoint in the set. 
This is NP-Hard and requires sophisticated algorithms for larger instances.

**Input format (stdin):**
```
num_vertices num_edges
u v  (for each edge, 0-indexed)
```

**Output format (stdout):**
```
vertex_count
vertex_1 vertex_2 ... vertex_k  (vertices in the cover, space-separated)
```

**Success Criteria:**
- All edges are covered by selected vertices
- Vertex set size is minimized (or near-minimized for large graphs)
- Output format is correct

**Failure Criteria:**
- Not all edges are covered
- Timeout without solution
- Invalid output format

**Environment:**
{describe_this_pc()}

**Rust Compiler:**
{RustCompiler("test_engine").describe()}

Be aware that default warnings are enabled and will cause a compilation failure,
so ensure that you write warning-free code.

Write complete, compilable Rust code with a main() function.

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
      "Explain your algorithm approach and how it adapts to different vertex cover complexities"
    },
    "rust_code": {
      "type": "string",
      "description": "Complete Rust code with main function that handles all scales"
    }
  },
  "required": ["reasoning", "rust_code"],
  "additionalProperties": False
}


def verify_cover(num_vertices: int, edges: List[Tuple[int, int]],
                 cover: Set[int]) -> Tuple[bool, str]:
  for v in cover:
    if v < 0 or v >= num_vertices:
      return False, f"Invalid vertex {v}"

  for u, v in edges:
    if u not in cover and v not in cover:
      return False, f"Edge ({u},{v}) not covered"

  return True, "Valid cover"


def _build_cover_viz(num_vertices: int, edges: List[Tuple[int, int]], cover: Set[int],
                     valid: bool, msg: str) -> dict:
  cover_list = [False] * num_vertices
  invalid_nodes: List[int] = []
  for v in cover:
    if v < 0 or v >= num_vertices:
      invalid_nodes.append(v)
    else:
      cover_list[v] = True

  uncovered_edges: List[Tuple[int, int]] = []
  for u, v in edges:
    if not cover_list[u] and not cover_list[v]:
      uncovered_edges.append((u, v))

  return {
    "num_vertices": num_vertices,
    "edges": edges,
    "cover": cover_list,
    "invalid_nodes": invalid_nodes,
    "uncovered_edges": uncovered_edges,
    "valid": valid,
    "msg": msg,
  }


def greedy_cover_size(edges: List[Tuple[int, int]]) -> int:
  """Upper bound using greedy 2-approximation."""
  cover = set()
  for u, v in edges:
    if u not in cover and v not in cover:
      cover.add(u)
      cover.add(v)
  return len(cover)


def exact_min_vertex_cover(num_vertices: int, edges: List[Tuple[int, int]]) -> int:
  if not edges:
    return 0
  if num_vertices <= 0:
    return 0

  edge_masks = [0] * num_vertices
  for idx, (u, v) in enumerate(edges):
    bit = 1 << idx
    edge_masks[u] |= bit
    edge_masks[v] |= bit

  full_mask = (1 << len(edges)) - 1
  vertices = range(num_vertices)
  for k in range(num_vertices + 1):
    for combo in itertools.combinations(vertices, k):
      cover_mask = 0
      for v in combo:
        cover_mask |= edge_masks[v]
        if cover_mask == full_mask:
          return k
  return num_vertices


def estimate_cover_size(num_vertices: int, num_edges: int) -> int:
  if num_vertices <= 0:
    return 0
  if num_edges <= 0:
    return 0
  avg_degree = 2.0 * num_edges / num_vertices
  avg_degree = max(avg_degree, 1e-6)
  alpha_est = 2.0 * math.log(max(avg_degree, 1.000001)) / avg_degree * num_vertices
  cover_est = num_vertices - alpha_est
  cover_est = max(1.0, min(float(num_vertices), cover_est))
  return int(cover_est)


def gradeAnswer(result: dict, subPass: int, aiEngineName: str) -> tuple:
  if not result or "rust_code" not in result:
    return 0.0, "No Rust code provided"

  case = TEST_CASES[subPass]
  use_streaming = _should_use_streaming(subPass)

  compiler = RustCompiler(aiEngineName)
  if not compiler.find_compiler():
    return 0.0, "No Rust compiler found"

  try:
    exe_path = compiler.compile(result["rust_code"])
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

      start = time.time()
      stdout, stderr, exec_time, return_code = compiler.execute(exe_path,
                                                                timeout=TIMEOUT_SECONDS,
                                                                stdin_file=input_file_path)

      if return_code != 0:
        return 0.0, f"Runtime error: {stderr[:200]}"

      lines = stdout.strip().split('\n')
      cover_size = int(lines[0])
      cover = set(map(int, lines[1].split())) if len(lines) > 1 else set()

      # Skip full verification for very large graphs
      #if case["edges"] > 10_000_000:
      #  if len(cover) == cover_size and all(0 <= v < case["vertices"] for v in cover):
      #    return 0.8, f"[{case['desc']}] Cover size {cover_size} in {exec_time:.2f}s (verification skipped)"
      #  else:
      #    return 0.0, f"[{case['desc']}] Invalid output format"
    else:
      num_vertices, edges = get_graph(subPass)
      input_data = format_input(num_vertices, edges)

      start = time.time()
      proc = subprocess.run([str(exe_path)],
                            input=input_data,
                            capture_output=True,
                            text=True,
                            timeout=TIMEOUT_SECONDS)
      exec_time = time.time() - start

      if proc.returncode != 0:
        return 0.0, f"Runtime error: {proc.stderr[:200]}"

      lines = proc.stdout.strip().split('\n')
      cover_size = int(lines[0])
      cover = set(map(int, lines[1].split())) if len(lines) > 1 else set()

    if cover_size != len(cover):
      return 0.0, f"[{case['desc']}] Output count mismatch: {cover_size} vs {len(cover)}"

    num_vertices = case["vertices"]
    num_edges = case["edges"]
    if num_vertices > 100_000:
      if not all(0 <= v < num_vertices for v in cover):
        return 0.0, f"[{case['desc']}] Invalid vertex in output"
      estimate = estimate_cover_size(num_vertices, num_edges)
      ratio = len(cover) / estimate if estimate > 0 else 1.0
      score = min(1.0, 1.5 - ratio)
      return max(0.0, score), (
        f"[{case['desc']}] Cover size {len(cover)} (estimate: {estimate}), {exec_time:.2f}s"
      )

    num_vertices, edges = get_graph(subPass)
    valid, msg = verify_cover(num_vertices, edges, cover)
    if case["vertices"] <= 1000:
      viz = LAST_COVER_VIZ[(subPass, aiEngineName)] = _build_cover_viz(
        num_vertices, edges, cover, valid, msg
      )

      if viz.get("uncovered_edges"):
        return 0.0, f"Unconvered edges remain: {len(viz['uncovered_edges'])}"

    if not valid:
      return 0.0, f"[{case['desc']}] {msg}"

    if num_vertices <= 20:
      optimal_size = exact_min_vertex_cover(num_vertices, edges)
      score = min(1.0, optimal_size / max(len(cover), 1))
      return max(
        0.5,
        score), f"[{case['desc']}] Cover size {len(cover)} (optimal: {optimal_size}), {exec_time:.2f}s"
    else:
      greedy_size = greedy_cover_size(edges)

      
      if len(cover) < greedy_size * 0.8:
        score = 1.0
      elif len(cover) < greedy_size * 1.2:
        score = 0.5
      else:
        score = 0.0

      return score, f"[{case['desc']}] Cover size {len(cover)} (greedy: {greedy_size}), {exec_time:.2f}s"

  except subprocess.TimeoutExpired:
    return 0.0, f"[{case['desc']}] Timeout"
  except ExecutionError as e:
    return 0.0, f"[{case['desc']}] {str(e)[:100]}"
  except Exception as e:
    return 0.0, f"[{case['desc']}] Error: {str(e)[:100]}"


def resultToNiceReport(result: dict, subPass: int, aiEngineName: str) -> str:
  if not result:
    return "<p style='color:red'>No result provided</p>"
  case = TEST_CASES[subPass]
  html = f"<h4>Minimum Vertex Cover - {case['desc']}</h4>"
  if "reasoning" in result and subPass == 0:
    r = result['reasoning'][:400] + ('...' if len(result.get('reasoning', '')) > 400 else '')
    html += f"<p><strong>Approach:</strong> {r.replace('<', '&lt;').replace('>', '&gt;')}</p>"
  if "rust_code" in result and subPass == 0:
    code = result["rust_code"].replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    html += f"<details><summary>View Rust Code ({len(result['rust_code'])} chars)</summary><pre>{code}</pre></details>"
  viz = LAST_COVER_VIZ.get((subPass, aiEngineName))
  if viz and viz.get("num_vertices", 0) <= 1000:
    html += _generate_cover_svg(viz)
  return html


def _generate_cover_svg(viz: dict) -> str:
  num_vertices = viz["num_vertices"]
  edges = viz["edges"]
  cover = viz["cover"]
  invalid_nodes = set(viz["invalid_nodes"])
  uncovered_edges = set(tuple(sorted(e)) for e in viz["uncovered_edges"])

  size = 600
  cx = cy = size / 2.0
  radius = size * 0.42

  positions = []
  for i in range(num_vertices):
    angle = 2 * math.pi * i / max(1, num_vertices)
    x = cx + radius * math.cos(angle)
    y = cy + radius * math.sin(angle)
    positions.append((x, y))

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
    is_bad = tuple(sorted((u, v))) in uncovered_edges
    stroke = "#ef4444" if is_bad else "#334155"
    opacity = "0.85" if is_bad else "0.25"
    width = "1.4" if is_bad else "0.6"
    edge_lines.append(
      f"<line x1='{x1:.2f}' y1='{y1:.2f}' x2='{x2:.2f}' y2='{y2:.2f}' "
      f"stroke='{stroke}' stroke-width='{width}' stroke-opacity='{opacity}' />"
    )

  node_circles = []
  for i, (x, y) in enumerate(positions):
    fill = "#22c55e" if cover[i] else "#475569"
    if i in invalid_nodes:
      node_circles.append(
        f"<circle cx='{x:.2f}' cy='{y:.2f}' r='{node_r:.2f}' fill='{fill}' "
        f"stroke='#ef4444' stroke-width='1.3' />"
      )
    else:
      node_circles.append(
        f"<circle cx='{x:.2f}' cy='{y:.2f}' r='{node_r:.2f}' fill='{fill}' stroke='#0f172a' stroke-width='0.4' />"
      )

  uncovered_count = len(uncovered_edges)
  status = "Valid cover" if viz.get("valid") else viz.get("msg", "Invalid cover")
  cover_size = sum(1 for v in cover if v)

  svg = "\n".join([
    "<div style='margin:12px 0;padding:10px;border:1px solid #1f2937;border-radius:8px;background:#0b1120;'>",
    f"<div style='color:#e2e8f0;font-size:13px;margin-bottom:6px;'><strong>Graph Visualization</strong> — {status}</div>",
    f"<div style='color:#94a3b8;font-size:12px;margin-bottom:6px;'>"
    f"Nodes: {num_vertices} | Edges: {len(edges)} | Cover size: {cover_size} | Uncovered edges: {uncovered_count}</div>",
    f"<svg width='100%' viewBox='0 0 {size} {size}' style='background:#0b1120;border:1px solid #334155;border-radius:6px;'>",
    "<g>",
    *edge_lines,
    "</g>",
    "<g>",
    *node_circles,
    "</g>",
    "</svg>",
    "<div style='color:#64748b;font-size:11px;margin-top:6px;'>Green nodes are in the cover. Red edges indicate uncovered constraints.</div>",
    "</div>",
  ])

  return svg


highLevelSummary = """
Minimum Vertex Cover finds the smallest set of vertices covering all edges.

**Algorithms:**
- **2-Approximation**: Greedy edge-picking, O(V+E)
- **Branch and Bound**: Exact but exponential
- **Kernelization**: Reduce problem size for FPT algorithms
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
