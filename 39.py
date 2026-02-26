"""
Test 39: Graph Bisection (Rust Implementation)

The model must write Rust code to split graph vertices into two equal parts
while minimizing cut edges (edges crossing between the two parts).

This benchmark uses planted instances where the optimal cut value is known,
allowing strict pass/fail grading.
"""

import math
import random
import subprocess
import time
from typing import List, Tuple, Set, Dict, Any, Iterable
from native_compiler import RustCompiler, CompilationError, ExecutionError
from solver_utils import StreamingInputFile

title = "Graph Bisection (Rust)"

tags = [
  "rust",
  "structured response",
  "np hard",
  "graph theory",
]
TIMEOUT_SECONDS = 30
RANDOM_SEED = 39393939


def _build_components(component_size: int, seed: int) -> Tuple[List[int], List[int]]:
  labels = list(range(2 * component_size))
  rng = random.Random(seed ^ 0x9E3779B1)
  rng.shuffle(labels)
  return labels[:component_size], labels[component_size:]


def _iter_component_edges(nodes: List[int], extra_out: int, window: int,
                          seed: int) -> Iterable[Tuple[int, int]]:
  """Connected sparse component: chain + random local forward chords."""
  n = len(nodes)
  rng = random.Random(seed)

  # Connected backbone
  for i in range(n - 1):
    u = nodes[i]
    v = nodes[i + 1]
    if u < v:
      yield (u, v)
    else:
      yield (v, u)

  # Additional local edges (always i < j in local indexing, so no duplicates)
  for i in range(n):
    start = i + 2
    if start >= n:
      continue
    end = min(n, start + window)
    choices = end - start
    take = min(extra_out, choices)
    if take <= 0:
      continue
    for j in rng.sample(range(start, end), take):
      u = nodes[i]
      v = nodes[j]
      if u < v:
        yield (u, v)
      else:
        yield (v, u)


def _iter_graph_edges(component_size: int, extra_out: int, window: int,
                      seed: int) -> Iterable[Tuple[int, int]]:
  """Planted bisection instance with known optimum cut=0.

  Build two connected components of equal size and no cross edges.
  A perfectly aligned bisection along component boundaries has cut 0 and is optimal.
  """
  left_nodes, right_nodes = _build_components(component_size, seed)
  yield from _iter_component_edges(left_nodes, extra_out, window, seed ^ 0x1111)
  yield from _iter_component_edges(right_nodes, extra_out, window, seed ^ 0x2222)


def _component_edge_count(component_size: int, extra_out: int, window: int) -> int:
  count = max(0, component_size - 1)
  for i in range(component_size):
    start = i + 2
    if start >= component_size:
      continue
    end = min(component_size, start + window)
    choices = max(0, end - start)
    count += min(extra_out, choices)
  return count


def _get_case_params(subpass: int) -> Tuple[int, int, int, int, int]:
  case = TEST_CASES[subpass]
  component_size = case["component_size"]
  extra_out = case["extra_out"]
  window = case["window"]
  if component_size < 4:
    raise ValueError(f"Invalid component_size={component_size} for subpass {subpass}")
  if extra_out < 0:
    raise ValueError(f"Invalid extra_out={extra_out} for subpass {subpass}")
  if window < 2:
    raise ValueError(f"Invalid window={window} for subpass {subpass}")
  num_vertices = 2 * component_size
  num_edges = 2 * _component_edge_count(component_size, extra_out, window)
  return component_size, extra_out, window, num_vertices, num_edges


TEST_CASES = [
  {
    "component_size": 20,
    "extra_out": 2,
    "window": 16,
    "desc": "tiny: 40 vertices"
  },
  {
    "component_size": 40,
    "extra_out": 2,
    "window": 24,
    "desc": "small: 80 vertices"
  },
  {
    "component_size": 80,
    "extra_out": 2,
    "window": 24,
    "desc": "160 vertices"
  },
  {
    "component_size": 160,
    "extra_out": 2,
    "window": 32,
    "desc": "320 vertices"
  },
  {
    "component_size": 320,
    "extra_out": 2,
    "window": 32,
    "desc": "640 vertices"
  },
  {
    "component_size": 800,
    "extra_out": 2,
    "window": 40,
    "desc": "1.6K vertices"
  },
  {
    "component_size": 2000,
    "extra_out": 2,
    "window": 48,
    "desc": "4K vertices"
  },
  {
    "component_size": 5000,
    "extra_out": 2,
    "window": 48,
    "desc": "10K vertices"
  },
  {
    "component_size": 15000,
    "extra_out": 2,
    "window": 64,
    "desc": "30K vertices"
  },
  {
    "component_size": 40000,
    "extra_out": 2,
    "window": 64,
    "desc": "80K vertices"
  },
  {
    "component_size": 100000,
    "extra_out": 2,
    "window": 64,
    "desc": "200K vertices"
  },
  {
    "component_size": 200000,
    "extra_out": 2,
    "window": 64,
    "desc": "400K vertices"
  },
  {
    "component_size": 333334,
    "extra_out": 2,
    "window": 64,
    "desc": "~666K vertices"
  },
]

GRAPH_CACHE: Dict[int, Any] = {}
_INPUT_FILE_CACHE: Dict[int, StreamingInputFile] = {}
STREAMING_THRESHOLD_EDGES = 1_000_000
LAST_BISECTION_VIZ: Dict[Tuple[int, str], dict] = {}


def get_graph(subpass: int) -> Tuple[int, List[Tuple[int, int]], Set[int], Set[int]]:
  if subpass not in GRAPH_CACHE:
    component_size, extra_out, window, num_vertices, _ = _get_case_params(subpass)
    edges = list(_iter_graph_edges(component_size, extra_out, window, RANDOM_SEED + subpass))
    left_nodes, right_nodes = _build_components(component_size, RANDOM_SEED + subpass)
    GRAPH_CACHE[subpass] = (num_vertices, edges, set(left_nodes), set(right_nodes))
  return GRAPH_CACHE[subpass]


def _estimate_edges(subpass: int) -> int:
  _, _, _, _, num_edges = _get_case_params(subpass)
  return num_edges


def _should_use_streaming(subpass: int) -> bool:
  return _estimate_edges(subpass) > STREAMING_THRESHOLD_EDGES


def _get_streaming_input(subpass: int) -> StreamingInputFile:
  if subpass in _INPUT_FILE_CACHE:
    return _INPUT_FILE_CACHE[subpass]

  component_size, extra_out, window, num_vertices, num_edges = _get_case_params(subpass)
  cache_key = (
    f"graph39_v2|c={component_size}|out={extra_out}|win={window}|seed={RANDOM_SEED + subpass}"
  )

  def generator():
    yield f"{num_vertices} {num_edges}\n"
    for u, v in _iter_graph_edges(component_size, extra_out, window, RANDOM_SEED + subpass):
      yield f"{u} {v}\n"

  input_file = StreamingInputFile(cache_key, generator, "test39_graphs")
  _INPUT_FILE_CACHE[subpass] = input_file
  return input_file


def format_input(num_vertices: int, edges: List[Tuple[int, int]]) -> str:
  lines = [f"{num_vertices} {len(edges)}"]
  for u, v in edges:
    lines.append(f"{u} {v}")
  return "\n".join(lines)


def prepareSubpassPrompt(subPass: int) -> str:
  """Generate the prompt for subpass 0."""
  if subPass != 0:
    raise StopIteration

  return """You are writing Rust code to solve the **Graph Bisection** problem.

Given an undirected graph with an even number of vertices, split the vertices into two equal-size parts to minimize cut edges (edges with endpoints in different parts).

The evaluator is strict: only a minimum-cut valid bisection is accepted.

**Input format (stdin)**
```
n m
u v
u v
... (m lines total)
```
- `n` vertices labeled `0..n-1` (n is even)
- `m` undirected edges

**Output format (stdout)**
```
cut_size
v1 v2 v3 ... v(n/2)
```
- first line: reported cut size
- second line: vertices in partition A (exactly n/2 distinct indices)
- partition B is all remaining vertices

**What is checked**
1. Output format is valid.
2. Partition size is exactly `n/2` and vertices are valid/distinct.
3. Reported cut matches the actual cut.
4. The cut is minimum for the instance.

Write complete Rust code with `main` that reads stdin and writes stdout.
"""


# List of subpasses to grade the single answer against all difficulty levels


extraGradeAnswerRuns = list(range(len(TEST_CASES)))

structure = {
  "type": "object",
  "properties": {
    "reasoning": {
      "type": "string",
      "description": "Explain your graph bisection approach"
    },
    "rust_code": {
      "type": "string",
      "description": "Complete Rust code with main function that handles all scales"
    }
  },
  "required": ["reasoning", "rust_code"],
  "additionalProperties": False
}


def _parse_output(stdout: str) -> Tuple[int, List[int]]:
  lines = [line.strip() for line in stdout.strip().splitlines() if line.strip()]
  if not lines:
    raise ValueError("No output")

  first = lines[0].split()
  if len(first) != 1:
    raise ValueError("First line must contain exactly one integer")
  reported_cut = int(first[0])
  if reported_cut < 0:
    raise ValueError("Cut size cannot be negative")

  vertices: List[int] = []
  for line in lines[1:]:
    vertices.extend(int(tok) for tok in line.split())

  return reported_cut, vertices


def calculate_cut(num_vertices: int, edges: Iterable[Tuple[int, int]],
                  partition_a_vertices: List[int]) -> Tuple[bool, int, str]:
  target = num_vertices // 2
  if len(partition_a_vertices) != target:
    return False, 0, f"Partition A must contain exactly {target} vertices, got {len(partition_a_vertices)}"

  in_a = bytearray(num_vertices)
  for v in partition_a_vertices:
    if v < 0 or v >= num_vertices:
      return False, 0, f"Invalid vertex {v}"
    if in_a[v]:
      return False, 0, f"Duplicate vertex {v}"
    in_a[v] = 1

  cut = 0
  for u, v in edges:
    if in_a[u] != in_a[v]:
      cut += 1

  return True, cut, "Valid bisection"


def _build_bisection_viz(num_vertices: int, edges: List[Tuple[int, int]], partition_a: List[int],
                         left_component: Set[int], valid: bool, msg: str,
                         reported_cut: int, actual_cut: int) -> dict:
  a_set = set(partition_a)
  b_set = set(range(num_vertices)) - a_set
  cut_edges = [(u, v) for (u, v) in edges if (u in a_set) != (v in a_set)]
  return {
    "num_vertices": num_vertices,
    "edges": edges,
    "partition_a": sorted(a_set),
    "partition_b": sorted(b_set),
    "left_component": sorted(left_component),
    "right_component": sorted(set(range(num_vertices)) - left_component),
    "cut_edges": cut_edges,
    "valid": valid,
    "msg": msg,
    "reported_cut": reported_cut,
    "actual_cut": actual_cut,
  }


def gradeAnswer(result: dict, subPass: int, aiEngineName: str) -> tuple:
  if not result or "rust_code" not in result:
    return 0.0, "No Rust code provided"

  case = TEST_CASES[subPass]
  component_size, extra_out, window, num_vertices, _ = _get_case_params(subPass)
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
    else:
      num_vertices, edges, _, _ = get_graph(subPass)
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
      stdout = proc.stdout

    try:
      reported_cut, partition_a = _parse_output(stdout)
    except Exception as e:
      return 0.0, f"[{case['desc']}] Invalid output format: {str(e)[:120]}"

    valid, actual_cut, msg = calculate_cut(
      num_vertices,
      _iter_graph_edges(component_size, extra_out, window, RANDOM_SEED + subPass),
      partition_a,
    )

    if valid and reported_cut != actual_cut:
      valid = False
      msg = f"Reported cut {reported_cut}, actual cut is {actual_cut}"

    if valid and actual_cut != 0:
      valid = False
      msg = f"Cut is {actual_cut}, but optimum is 0"

    if num_vertices <= 600:
      _, viz_edges, left_component, _ = get_graph(subPass)
      LAST_BISECTION_VIZ[(subPass, aiEngineName)] = _build_bisection_viz(
        num_vertices, viz_edges, partition_a, left_component, valid, msg, reported_cut, actual_cut
      )

    if not valid:
      return 0.0, f"[{case['desc']}] FAIL: {msg}"
    return 1.0, f"[{case['desc']}] PASS: minimum cut 0 in {exec_time:.2f}s"

  except subprocess.TimeoutExpired:
    return 0.0, f"[{case['desc']}] FAIL: Timeout"
  except ExecutionError as e:
    return 0.0, f"[{case['desc']}] FAIL: {str(e)[:100]}"
  except Exception as e:
    return 0.0, f"[{case['desc']}] Error: {str(e)[:100]}"


def resultToNiceReport(result: dict, subPass: int, aiEngineName: str) -> str:
  if not result:
    return "<p style='color:red'>No result provided</p>"
  case = TEST_CASES[subPass]
  component_size, _, _, num_vertices, num_edges = _get_case_params(subPass)
  html = f"<h4>Graph Bisection - {case['desc']}</h4>"
  html += (
    f"<p style='font-size:12px;color:#475569;margin:6px 0;'>"
    f"Vertices: {num_vertices:,} | Edges: ~{num_edges:,} | Target partition size: {num_vertices // 2:,} | "
    f"Known optimum cut: 0</p>"
  )
  if "reasoning" in result:
    r = result['reasoning'][:400] + ('...' if len(result.get('reasoning', '')) > 400 else '')
    html += f"<p><strong>Approach:</strong> {r.replace('<', '&lt;').replace('>', '&gt;')}</p>"
  if "rust_code" in result:
    code = result["rust_code"].replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    html += f"<details><summary>View Rust Code ({len(result['rust_code'])} chars)</summary><pre>{code}</pre></details>"
  viz = LAST_BISECTION_VIZ.get((subPass, aiEngineName))
  if viz and viz.get("num_vertices", 0) <= 600:
    html += _generate_bisection_svg(viz)
  return html


highLevelSummary = """
<p>Split a graph&rsquo;s nodes into two equally-sized groups so that as few edges
as possible cross between the groups. Think of dividing a school into two teams
where friends want to stay together &mdash; minimise the number of friendships
that get split.</p>
<p>The test uses planted graphs that can be perfectly bisected with zero crossing
edges, so grading is strict pass/fail. Subpasses increase the graph size.</p>
"""


def _generate_bisection_svg(viz: dict) -> str:
  num_vertices = viz["num_vertices"]
  edges = viz["edges"]
  partition_a = set(viz["partition_a"])
  left_component = set(viz["left_component"])
  cut_edges = set(tuple(sorted(e)) for e in viz["cut_edges"])

  size = 640
  left_cx = size * 0.3
  right_cx = size * 0.7
  cy = size * 0.5
  ring_r = size * 0.22

  left_nodes = [v for v in range(num_vertices) if v in left_component]
  right_nodes = [v for v in range(num_vertices) if v not in left_component]

  positions: List[Tuple[float, float]] = [(0.0, 0.0)] * num_vertices
  for i, v in enumerate(left_nodes):
    angle = 2 * math.pi * i / max(1, len(left_nodes))
    positions[v] = (left_cx + ring_r * math.cos(angle), cy + ring_r * math.sin(angle))
  for i, v in enumerate(right_nodes):
    angle = 2 * math.pi * i / max(1, len(right_nodes))
    positions[v] = (right_cx + ring_r * math.cos(angle), cy + ring_r * math.sin(angle))

  if num_vertices <= 80:
    node_r = 4.8
  elif num_vertices <= 200:
    node_r = 3.6
  else:
    node_r = 2.8

  edge_lines = []
  for u, v in edges:
    x1, y1 = positions[u]
    x2, y2 = positions[v]
    is_cut = tuple(sorted((u, v))) in cut_edges
    stroke = "#ef4444" if is_cut else "#334155"
    opacity = "0.9" if is_cut else "0.2"
    width = "1.3" if is_cut else "0.55"
    edge_lines.append(
      f"<line x1='{x1:.2f}' y1='{y1:.2f}' x2='{x2:.2f}' y2='{y2:.2f}' "
      f"stroke='{stroke}' stroke-width='{width}' stroke-opacity='{opacity}' />"
    )

  node_circles = []
  for v, (x, y) in enumerate(positions):
    in_a = v in partition_a
    in_left = v in left_component
    fill = "#22c55e" if in_a else "#f59e0b"
    stroke = "#0ea5e9" if in_left else "#a78bfa"
    node_circles.append(
      f"<circle cx='{x:.2f}' cy='{y:.2f}' r='{node_r:.2f}' fill='{fill}' stroke='{stroke}' stroke-width='0.8' />"
    )

  status = "Valid minimum bisection" if viz.get("valid") else viz.get("msg", "Invalid")
  svg = "\n".join([
    "<div style='margin:12px 0;padding:10px;border:1px solid #1f2937;border-radius:8px;background:#0b1120;'>",
    f"<div style='color:#e2e8f0;font-size:13px;margin-bottom:6px;'><strong>Graph Visualization</strong> — {status}</div>",
    f"<div style='color:#94a3b8;font-size:12px;margin-bottom:6px;'>"
    f"Vertices: {num_vertices} | Reported cut: {viz.get('reported_cut')} | "
    f"Actual cut: {viz.get('actual_cut')} | Cut edges: {len(cut_edges)}</div>",
    f"<svg width='100%' viewBox='0 0 {size} {size}' style='background:#0b1120;border:1px solid #334155;border-radius:6px;'>",
    "<g>",
    *edge_lines,
    "</g>",
    "<g>",
    *node_circles,
    "</g>",
    "</svg>",
    "<div style='color:#64748b;font-size:11px;margin-top:6px;'>"
    "Node fill: green=reported partition A, amber=partition B. "
    "Node stroke: cyan/purple shows planted component membership. "
    "Red edges are crossing (cut) edges.</div>",
    "</div>",
  ])
  return svg


def setup():
  """Pre-generate and cache all streaming input files for parallel test execution."""
  print(f"  Pre-generating streaming input files for {len(TEST_CASES)} test cases...")
  for subpass in range(len(TEST_CASES)):
    if _should_use_streaming(subpass):
      streaming_input = _get_streaming_input(subpass)
      input_path = streaming_input.generate()
      size_mb = streaming_input.get_size_bytes() / (1024 * 1024)
      print(f"    Subpass {subpass}: {size_mb:.1f} MB cached")
