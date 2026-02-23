"""
Test 37: Feedback Vertex Set (C# Implementation)

The LLM must write C# code to find the minimum set of vertices whose removal
makes the graph acyclic (a DAG or forest). This is NP-Hard.

Subpasses increase graph complexity, requiring iterative compression,
branch-and-bound, or 2-approximation algorithms.

Solver times out after 5 minutes.
"""

import math
import random
import time
from typing import List, Tuple, Set, Dict, Any, Iterable
from native_compiler import CSharpCompiler, CompilationError, ExecutionError, describe_this_pc
from solver_utils import StreamingInputFile

title = "Feedback Vertex Set (C#)"
TIMEOUT_SECONDS = 30
RANDOM_SEED = 37373737


def _iter_graph_edges(dag_vertices: int, feedback_vertices: int, dag_extra_out: int,
                      dag_window: int, seed: int) -> Iterable[Tuple[int, int]]:
  """Generate planted directed-FVS instances.

  Construction:
  1) Random DAG subgraph on `dag_vertices` vertices (backbone + forward random edges).
  2) Add `feedback_vertices` new vertices (breaker vertices).
  3) For each breaker b_i, create two directed cycles that share only b_i:
       a1 -> c1 -> b_i -> a1
       a2 -> c2 -> b_i -> a2

  All cycles in gadget i can be destroyed with one deletion (b_i), but require at
  least two deletions if b_i is not removed. This makes the unique minimum FVS the
  set of all breaker vertices.
  """
  if dag_vertices < 4 * feedback_vertices + 1:
    raise ValueError("dag_vertices must be at least 4 * feedback_vertices + 1")
  if dag_window < 2:
    raise ValueError("dag_window must be >= 2")

  rng = random.Random(seed)
  topo_order = list(range(dag_vertices))
  rng.shuffle(topo_order)

  # Backbone chain guarantees DAG connectivity in topological order.
  for i in range(dag_vertices - 1):
    yield (topo_order[i], topo_order[i + 1])

  # Extra forward DAG edges (still acyclic).
  for i in range(dag_vertices):
    start = i + 2
    if start >= dag_vertices:
      continue
    end = min(dag_vertices, start + dag_window)
    choices = end - start
    take = min(dag_extra_out, choices)
    if take <= 0:
      continue
    for j in rng.sample(range(start, end), take):
      yield (topo_order[i], topo_order[j])

  # Cycle gadgets around breaker vertices.
  for i in range(feedback_vertices):
    b = dag_vertices + i
    a1 = topo_order[4 * i]
    c1 = topo_order[4 * i + 1]
    a2 = topo_order[4 * i + 2]
    c2 = topo_order[4 * i + 3]
    yield (c1, b)
    yield (b, a1)
    yield (c2, b)
    yield (b, a2)


def _get_case_params(subpass: int) -> Tuple[int, int, int, int, int, int]:
  case = TEST_CASES[subpass]
  dag_vertices = case["dag_vertices"]
  feedback_vertices = case["feedback_vertices"]
  dag_extra_out = case["dag_extra_out"]
  dag_window = case["dag_window"]

  if dag_vertices < 4 * feedback_vertices + 1:
    raise ValueError(f"Invalid case {subpass}: dag_vertices too small for gadget layout")

  dag_backbone_edges = max(0, dag_vertices - 1)
  dag_extra_edges = 0
  for i in range(dag_vertices):
    start = i + 2
    if start >= dag_vertices:
      continue
    end = min(dag_vertices, start + dag_window)
    choices = max(0, end - start)
    dag_extra_edges += min(dag_extra_out, choices)

  gadget_edges = 4 * feedback_vertices
  num_vertices = dag_vertices + feedback_vertices
  num_edges = dag_backbone_edges + dag_extra_edges + gadget_edges
  return dag_vertices, feedback_vertices, dag_extra_out, dag_window, num_vertices, num_edges


TEST_CASES = [
  {
    "dag_vertices": 40,
    "feedback_vertices": 4,
    "dag_extra_out": 2,
    "dag_window": 32,
    "desc": "tiny: DAG=40, breakers=4"
  },
  {
    "dag_vertices": 80,
    "feedback_vertices": 8,
    "dag_extra_out": 2,
    "dag_window": 32,
    "desc": "small: DAG=80, breakers=8"
  },
  {
    "dag_vertices": 160,
    "feedback_vertices": 12,
    "dag_extra_out": 2,
    "dag_window": 48,
    "desc": "DAG=160, breakers=12"
  },
  {
    "dag_vertices": 320,
    "feedback_vertices": 20,
    "dag_extra_out": 2,
    "dag_window": 48,
    "desc": "DAG=320, breakers=20"
  },
  {
    "dag_vertices": 800,
    "feedback_vertices": 35,
    "dag_extra_out": 2,
    "dag_window": 64,
    "desc": "DAG=800, breakers=35"
  },
  {
    "dag_vertices": 2000,
    "feedback_vertices": 70,
    "dag_extra_out": 2,
    "dag_window": 64,
    "desc": "DAG=2K, breakers=70"
  },
  {
    "dag_vertices": 5000,
    "feedback_vertices": 120,
    "dag_extra_out": 2,
    "dag_window": 64,
    "desc": "DAG=5K, breakers=120"
  },
  {
    "dag_vertices": 10000,
    "feedback_vertices": 180,
    "dag_extra_out": 2,
    "dag_window": 64,
    "desc": "DAG=10K, breakers=180"
  },
  {
    "dag_vertices": 25000,
    "feedback_vertices": 320,
    "dag_extra_out": 2,
    "dag_window": 64,
    "desc": "DAG=25K, breakers=320"
  },
  {
    "dag_vertices": 50000,
    "feedback_vertices": 520,
    "dag_extra_out": 2,
    "dag_window": 64,
    "desc": "DAG=50K, breakers=520"
  },
  {
    "dag_vertices": 100000,
    "feedback_vertices": 900,
    "dag_extra_out": 2,
    "dag_window": 96,
    "desc": "DAG=100K, breakers=900"
  },
  {
    "dag_vertices": 250000,
    "feedback_vertices": 1800,
    "dag_extra_out": 2,
    "dag_window": 96,
    "desc": "DAG=250K, breakers=1.8K"
  },
  {
    "dag_vertices": 500000,
    "feedback_vertices": 3000,
    "dag_extra_out": 2,
    "dag_window": 96,
    "desc": "DAG=500K, breakers=3K"
  },
  {
    "dag_vertices": 1000000,
    "feedback_vertices": 5000,
    "dag_extra_out": 2,
    "dag_window": 96,
    "desc": "DAG=1M, breakers=5K"
  },
]

GRAPH_CACHE: Dict[int, Any] = {}
_INPUT_FILE_CACHE: Dict[int, StreamingInputFile] = {}
STREAMING_THRESHOLD_EDGES = 1_000_000
LAST_FVS_VIZ: Dict[Tuple[int, str], dict] = {}


def get_graph(subpass: int) -> Tuple[int, List[Tuple[int, int]], int, int]:
  if subpass not in GRAPH_CACHE:
    dag_vertices, feedback_vertices, dag_extra_out, dag_window, num_vertices, _ = _get_case_params(subpass)
    edges = list(
      _iter_graph_edges(dag_vertices, feedback_vertices, dag_extra_out, dag_window, RANDOM_SEED + subpass)
    )
    GRAPH_CACHE[subpass] = (num_vertices, edges, dag_vertices, feedback_vertices)
  return GRAPH_CACHE[subpass]


def _should_use_streaming(subpass: int) -> bool:
  return _estimate_edges(subpass) > STREAMING_THRESHOLD_EDGES


def _estimate_edges(subpass: int) -> int:
  _, _, _, _, _, num_edges = _get_case_params(subpass)
  return num_edges


def _get_streaming_input(subpass: int) -> StreamingInputFile:
  if subpass in _INPUT_FILE_CACHE:
    return _INPUT_FILE_CACHE[subpass]

  dag_vertices, feedback_vertices, dag_extra_out, dag_window, num_vertices, num_edges = _get_case_params(subpass)
  cache_key = (
    f"graph37_v2|dag={dag_vertices}|fvs={feedback_vertices}|"
    f"out={dag_extra_out}|win={dag_window}|seed={RANDOM_SEED + subpass}"
  )

  def generator():
    yield f"{num_vertices} {num_edges}\n"
    for u, v in _iter_graph_edges(dag_vertices, feedback_vertices, dag_extra_out, dag_window,
                                  RANDOM_SEED + subpass):
      yield f"{u} {v}\n"

  input_file = StreamingInputFile(cache_key, generator, "test37_graphs")
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

  return f"""You are writing C# code to solve the **Directed Feedback Vertex Set** problem.

Given a directed graph, find the smallest set of vertices to remove so the remaining graph has **no directed cycles** (i.e., it becomes a DAG).

The evaluator is strict: only a minimum-size valid feedback vertex set is accepted.

**Input format (stdin)**
```
n m
u v
u v
... (m lines total)
```
- `n`: number of vertices (0-indexed: `0..n-1`)
- `m`: number of directed edges
- each line `u v` means an edge `u -> v`

**Output format (stdout)**
```
k
v1 v2 v3 ... vk
```
- `k`: number of removed vertices
- second line: exactly `k` distinct vertex indices (any order)

**What is checked**
1. Output format is valid.
2. Vertex indices are valid and distinct.
3. Removing those vertices makes the graph acyclic.
4. The set size is minimum.

**Requirements**
1. Provide complete C# code with `Main`.
2. Read from stdin and write to stdout.
3. Handle small and very large graphs.
4. Finish within the test timeout.

**Environment**
{describe_this_pc()}

**C# Compiler**
{CSharpCompiler("test_engine").describe()}
"""


# List of subpasses to grade the single answer against all difficulty levels


extraGradeAnswerRuns = list(range(len(TEST_CASES)))

structure = {
  "type": "object",
  "properties": {
    "reasoning": {
      "type": "string",
      "description": "Explain your approach for minimum directed feedback vertex set"
    },
    "csharp_code": {
      "type": "string",
      "description": "Complete C# code with Main method that handles all scales"
    }
  },
  "required": ["reasoning", "csharp_code"],
  "additionalProperties": False
}


def _parse_output(stdout: str) -> Tuple[int, List[int]]:
  lines = [line.strip() for line in stdout.strip().splitlines() if line.strip()]
  if not lines:
    raise ValueError("No output")

  first = lines[0].split()
  if len(first) != 1:
    raise ValueError("First line must contain exactly one integer")
  set_size = int(first[0])
  if set_size < 0:
    raise ValueError("Set size cannot be negative")

  vertices: List[int] = []
  for line in lines[1:]:
    vertices.extend(int(tok) for tok in line.split())

  if len(vertices) != set_size:
    raise ValueError(f"Expected {set_size} vertices, got {len(vertices)}")
  return set_size, vertices


def _expected_feedback_set(dag_vertices: int, feedback_vertices: int) -> Set[int]:
  return set(range(dag_vertices, dag_vertices + feedback_vertices))


def _build_fvs_viz(num_vertices: int, edges: List[Tuple[int, int]], dag_vertices: int,
                   feedback_vertices: int, chosen_vertices: List[int], valid: bool,
                   msg: str) -> dict:
  expected = _expected_feedback_set(dag_vertices, feedback_vertices)
  chosen_set = set(chosen_vertices)
  cycle_edges = [(u, v) for (u, v) in edges if u >= dag_vertices or v >= dag_vertices]

  return {
    "num_vertices": num_vertices,
    "edges": edges,
    "dag_vertices": dag_vertices,
    "feedback_vertices": feedback_vertices,
    "chosen": sorted(chosen_set),
    "expected": sorted(expected),
    "missing": sorted(expected - chosen_set),
    "extras": sorted(chosen_set - expected),
    "cycle_edges": cycle_edges,
    "valid": valid,
    "msg": msg,
  }


def gradeAnswer(result: dict, subPass: int, aiEngineName: str) -> tuple:
  if not result or "csharp_code" not in result:
    return 0.0, "No C# code provided"

  case = TEST_CASES[subPass]
  dag_vertices, feedback_vertices, _, _, num_vertices, _ = _get_case_params(subPass)
  expected_set = _expected_feedback_set(dag_vertices, feedback_vertices)
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
      num_vertices, edges, _, _ = get_graph(subPass)
      input_data = format_input(num_vertices, edges)
      stdout, stderr, exec_time, retcode = compiler.execute(exe_path, input_data, TIMEOUT_SECONDS)

    if retcode != 0:
      return 0.0, f"Runtime error: {stderr[:200]}"

    try:
      set_size, chosen_vertices = _parse_output(stdout)
    except Exception as e:
      return 0.0, f"[{case['desc']}] Invalid output format: {str(e)[:120]}"

    chosen_set = set()
    for v in chosen_vertices:
      if v < 0 or v >= num_vertices:
        return 0.0, f"[{case['desc']}] Invalid vertex {v}"
      if v in chosen_set:
        return 0.0, f"[{case['desc']}] Duplicate vertex {v}"
      chosen_set.add(v)

    if set_size != feedback_vertices:
      valid = False
      msg = f"Reported size {set_size}, but optimum is {feedback_vertices}"
    elif chosen_set != expected_set:
      missing = sorted(expected_set - chosen_set)
      extras = sorted(chosen_set - expected_set)
      miss_preview = ", ".join(map(str, missing[:5])) if missing else "none"
      extra_preview = ", ".join(map(str, extras[:5])) if extras else "none"
      valid = False
      msg = f"Incorrect minimum set (missing: {miss_preview}; extras: {extra_preview})"
    else:
      valid = True
      msg = "Valid minimum feedback vertex set"

    if num_vertices <= 500:
      _, viz_edges, dag_v, fb_v = get_graph(subPass)
      LAST_FVS_VIZ[(subPass, aiEngineName)] = _build_fvs_viz(
        num_vertices, viz_edges, dag_v, fb_v, chosen_vertices, valid, msg
      )

    if not valid:
      return 0.0, f"[{case['desc']}] FAIL: {msg}"
    return 1.0, f"[{case['desc']}] PASS: minimum FVS size {feedback_vertices} in {exec_time:.2f}s"

  except ExecutionError as e:
    return 0.1, f"[{case['desc']}] {str(e)[:100]}"
  except Exception as e:
    return 0.0, f"[{case['desc']}] Error: {str(e)[:100]}"


def resultToNiceReport(result: dict, subPass: int, aiEngineName: str) -> str:
  if not result:
    return "<p style='color:red'>No result provided</p>"
  case = TEST_CASES[subPass]
  dag_vertices, feedback_vertices, _, _, num_vertices, num_edges = _get_case_params(subPass)
  html = f"<h4>Feedback Vertex Set - {case['desc']}</h4>"
  html += (
    f"<p style='font-size:12px;color:#475569;margin:6px 0;'>"
    f"Graph: {num_vertices:,} vertices, ~{num_edges:,} edges | DAG subgraph: {dag_vertices:,} | "
    f"cycle-breaker vertices: {feedback_vertices:,}</p>"
  )
  if "reasoning" in result:
    r = result['reasoning'][:400] + ('...' if len(result.get('reasoning', '')) > 400 else '')
    html += f"<p><strong>Approach:</strong> {r.replace('<', '&lt;').replace('>', '&gt;')}</p>"
  if "csharp_code" in result:
    code = result["csharp_code"].replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    html += f"<details><summary>View C# Code ({len(result['csharp_code'])} chars)</summary><pre>{code}</pre></details>"
  viz = LAST_FVS_VIZ.get((subPass, aiEngineName))
  if viz and viz.get("num_vertices", 0) <= 500:
    html += _generate_fvs_svg(viz)
  return html


highLevelSummary = """
<p>Find the smallest set of nodes to remove from a directed graph so that no cycles
remain. Think of breaking deadlocks in a dependency chain &mdash; remove as few
tasks as possible so that the rest can be completed in a straight order.</p>
<p>This is NP-hard. The test uses planted instances where a known small set of
nodes creates all the cycles, enabling strict pass/fail grading. Subpasses
increase the graph size.</p>
"""


def _generate_fvs_svg(viz: dict) -> str:
  num_vertices = viz["num_vertices"]
  edges = viz["edges"]
  dag_vertices = viz["dag_vertices"]
  chosen = set(viz["chosen"])
  expected = set(viz["expected"])
  missing = set(viz["missing"])
  extras = set(viz["extras"])
  cycle_edges = set(tuple(e) for e in viz["cycle_edges"])

  size = 620
  cx = cy = size / 2.0
  outer_r = size * 0.43
  inner_r = size * 0.2

  positions: List[Tuple[float, float]] = [(0.0, 0.0)] * num_vertices
  for i in range(dag_vertices):
    angle = 2 * math.pi * i / max(1, dag_vertices)
    x = cx + outer_r * math.cos(angle)
    y = cy + outer_r * math.sin(angle)
    positions[i] = (x, y)

  breaker_count = max(1, num_vertices - dag_vertices)
  for idx, v in enumerate(range(dag_vertices, num_vertices)):
    angle = 2 * math.pi * idx / breaker_count
    x = cx + inner_r * math.cos(angle)
    y = cy + inner_r * math.sin(angle)
    positions[v] = (x, y)

  if num_vertices <= 80:
    node_r = 4.8
  elif num_vertices <= 200:
    node_r = 3.8
  else:
    node_r = 3.0

  edge_lines = []
  for u, v in edges:
    x1, y1 = positions[u]
    x2, y2 = positions[v]
    gadget_edge = (u, v) in cycle_edges
    if gadget_edge:
      stroke = "#38bdf8"
      width = "1.0"
      opacity = "0.55"
    else:
      stroke = "#334155"
      width = "0.55"
      opacity = "0.23"
    edge_lines.append(
      f"<line x1='{x1:.2f}' y1='{y1:.2f}' x2='{x2:.2f}' y2='{y2:.2f}' "
      f"stroke='{stroke}' stroke-width='{width}' stroke-opacity='{opacity}' marker-end='url(#arrowHead)' />"
    )

  node_circles = []
  for v, (x, y) in enumerate(positions):
    if v in expected and v in chosen:
      fill = "#22c55e"
      stroke = "#14532d"
    elif v in missing:
      fill = "#ef4444"
      stroke = "#7f1d1d"
    elif v in extras:
      fill = "#f59e0b"
      stroke = "#78350f"
    elif v >= dag_vertices:
      fill = "#0ea5e9"
      stroke = "#0c4a6e"
    else:
      fill = "#64748b"
      stroke = "#0f172a"
    node_circles.append(
      f"<circle cx='{x:.2f}' cy='{y:.2f}' r='{node_r:.2f}' fill='{fill}' stroke='{stroke}' stroke-width='0.7' />"
    )

  status = "Valid minimum FVS" if viz.get("valid") else viz.get("msg", "Invalid set")
  svg = "\n".join([
    "<div style='margin:12px 0;padding:10px;border:1px solid #1f2937;border-radius:8px;background:#0b1120;'>",
    f"<div style='color:#e2e8f0;font-size:13px;margin-bottom:6px;'><strong>Graph Visualization</strong> — {status}</div>",
    f"<div style='color:#94a3b8;font-size:12px;margin-bottom:6px;'>"
    f"Vertices: {num_vertices} | DAG nodes: {dag_vertices} | Breakers: {len(expected)} | "
    f"Chosen: {len(chosen)} | Missing: {len(missing)} | Extras: {len(extras)}</div>",
    f"<svg width='100%' viewBox='0 0 {size} {size}' style='background:#0b1120;border:1px solid #334155;border-radius:6px;'>",
    "<defs>",
    "<marker id='arrowHead' markerWidth='8' markerHeight='8' refX='6' refY='3' orient='auto' markerUnits='strokeWidth'>",
    "<path d='M0,0 L0,6 L6,3 z' fill='#475569' />",
    "</marker>",
    "</defs>",
    "<g>",
    *edge_lines,
    "</g>",
    "<g>",
    *node_circles,
    "</g>",
    "</svg>",
    "<div style='color:#64748b;font-size:11px;margin-top:6px;'>"
    "Green nodes are correctly selected breaker vertices. Red are missed breakers. "
    "Amber are incorrect extra removals. Cyan edges are cycle-gadget edges.</div>",
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
