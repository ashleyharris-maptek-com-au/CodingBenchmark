"""
Test 36: Maximum Independent Set (C++ Implementation)

The LLM must write C++ code to find the largest set of vertices with no edges
between them. This is NP-Hard (complement of Maximum Clique).

Subpasses increase graph size, requiring branch-and-bound, dynamic programming
on tree decompositions, or sophisticated local search.

Solver times out after 5 minutes.
"""

import math
import random
import subprocess
import time
import hashlib
from typing import List, Tuple, Dict, Any, Iterable, Optional
from native_compiler import CppCompiler, CompilationError, ExecutionError, describe_this_pc
from solver_utils import StreamingInputFile, GradeCache

title = "Maximum Independent Set (C++)"

tags = [
  "cpp",
  "structured response",
  "np hard",
  "graph theory",
]
TIMEOUT_SECONDS = 30
RANDOM_SEED = 36363636

_grade_cache = GradeCache("test36")


def _cache_key_parts(result: dict, subPass: int) -> tuple:
  case = TEST_CASES[subPass]
  code = result.get("cpp_code", "")
  return (
    hashlib.sha256(code.encode("utf-8")).hexdigest()[:16],
    f"k={case['solution_vertices']}|d={case['base_degree']}|seed={RANDOM_SEED + subPass}",
  )


def _iter_graph_edges(solution_vertices: int, base_degree: int,
                      seed: int) -> Iterable[Tuple[int, int]]:
  """Generate planted-MIS graph edges.

  Construction:
  1) Start with hidden solution vertices S = {0..k-1} (size k).
  2) Build a random-looking degree-d base graph on S.
  3) Subdivide each base edge with a new vertex x_e.
  4) Connect subdivision vertices that share an endpoint in the base graph.

  This guarantees alpha(G)=k, achieved by selecting all solution vertices.
  """
  if base_degree % 2 != 0:
    raise ValueError("base_degree must be even")
  if solution_vertices <= base_degree:
    raise ValueError("solution_vertices must be greater than base_degree")

  rng = random.Random(seed)
  order = list(range(solution_vertices))
  rng.shuffle(order)

  half = base_degree // 2
  incident_subdivisions: List[List[int]] = [[] for _ in range(solution_vertices)]

  subdivision_id = solution_vertices
  for i in range(solution_vertices):
    u = order[i]
    for step in range(1, half + 1):
      v = order[(i + step) % solution_vertices]
      x = subdivision_id
      subdivision_id += 1
      incident_subdivisions[u].append(x)
      incident_subdivisions[v].append(x)
      yield (u, x)
      yield (v, x)

  for u in range(solution_vertices):
    subs = incident_subdivisions[u]
    for i in range(len(subs)):
      a = subs[i]
      for b in subs[i + 1:]:
        if a < b:
          yield (a, b)
        else:
          yield (b, a)


def _get_case_params(subpass: int) -> Tuple[int, int, int, int]:
  case = TEST_CASES[subpass]
  solution_vertices = case["solution_vertices"]
  base_degree = case["base_degree"]
  if base_degree % 2 != 0 or base_degree < 2:
    raise ValueError(f"Invalid base_degree={base_degree} for subpass {subpass}")
  if solution_vertices <= base_degree:
    raise ValueError(f"Invalid solution_vertices={solution_vertices} for subpass {subpass}")

  subdivisions = solution_vertices * base_degree // 2
  num_vertices = solution_vertices + subdivisions
  num_edges = solution_vertices * base_degree + solution_vertices * (base_degree * (base_degree - 1) // 2)
  return solution_vertices, base_degree, num_vertices, num_edges


TEST_CASES = [
  {
    "solution_vertices": 20,
    "base_degree": 4,
    "desc": "tiny (60 vertices), planted MIS=20"
  },
  {
    "solution_vertices": 40,
    "base_degree": 4,
    "desc": "small (120 vertices), planted MIS=40"
  },
  {
    "solution_vertices": 80,
    "base_degree": 4,
    "desc": "240 vertices, planted MIS=80"
  },
  {
    "solution_vertices": 160,
    "base_degree": 4,
    "desc": "480 vertices, planted MIS=160"
  },
  {
    "solution_vertices": 320,
    "base_degree": 4,
    "desc": "960 vertices, planted MIS=320"
  },
  {
    "solution_vertices": 800,
    "base_degree": 4,
    "desc": "2.4K vertices, planted MIS=800"
  },
  {
    "solution_vertices": 2000,
    "base_degree": 4,
    "desc": "6K vertices, planted MIS=2000"
  },
  {
    "solution_vertices": 5000,
    "base_degree": 4,
    "desc": "15K vertices, planted MIS=5000"
  },
  {
    "solution_vertices": 15000,
    "base_degree": 4,
    "desc": "45K vertices, planted MIS=15K"
  },
  {
    "solution_vertices": 40000,
    "base_degree": 4,
    "desc": "120K vertices, planted MIS=40K"
  },
  {
    "solution_vertices": 100000,
    "base_degree": 4,
    "desc": "300K vertices, planted MIS=100K"
  },
  {
    "solution_vertices": 200000,
    "base_degree": 4,
    "desc": "600K vertices, planted MIS=200K"
  },
  {
    "solution_vertices": 333334,
    "base_degree": 4,
    "desc": "~1M vertices, planted MIS=333,334"
  },
]

GRAPH_CACHE: Dict[int, Any] = {}
_INPUT_FILE_CACHE: Dict[int, StreamingInputFile] = {}
STREAMING_THRESHOLD_EDGES = 1_000_000
LAST_MIS_VIZ: Dict[Tuple[int, str], dict] = {}


def get_graph(subpass: int) -> Tuple[int, List[Tuple[int, int]], int]:
  if subpass not in GRAPH_CACHE:
    solution_vertices, base_degree, num_vertices, _ = _get_case_params(subpass)
    edges = list(_iter_graph_edges(solution_vertices, base_degree, RANDOM_SEED + subpass))
    GRAPH_CACHE[subpass] = (num_vertices, edges, solution_vertices)
  return GRAPH_CACHE[subpass]


def _estimate_edges(subpass: int) -> int:
  _, _, _, num_edges = _get_case_params(subpass)
  return num_edges


def _should_use_streaming(subpass: int) -> bool:
  return _estimate_edges(subpass) > STREAMING_THRESHOLD_EDGES


def _get_streaming_input(subpass: int) -> StreamingInputFile:
  if subpass in _INPUT_FILE_CACHE:
    return _INPUT_FILE_CACHE[subpass]

  solution_vertices, base_degree, num_vertices, num_edges = _get_case_params(subpass)
  cache_key = (
    f"graph36_v2|k={solution_vertices}|d={base_degree}|"
    f"seed={RANDOM_SEED + subpass}"
  )

  def generator():
    yield f"{num_vertices} {num_edges}\n"
    for u, v in _iter_graph_edges(solution_vertices, base_degree, RANDOM_SEED + subpass):
      yield f"{u} {v}\n"

  input_file = StreamingInputFile(cache_key, generator, "test36_graphs")
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

  return f"""You are writing C++ code to solve the **Maximum Independent Set** problem.

Given an undirected graph, find the largest set of vertices such that no two chosen vertices share an edge.

The evaluator is strict: only an exact maximum independent set is accepted.

**Input format (stdin)**
```
n m
u v
u v
... (m lines total)
```
- `n` = number of vertices (0-indexed: 0..n-1)
- `m` = number of undirected edges
- each edge appears as two integers `u v`

**Output format (stdout)**
```
k
v1 v2 v3 ... vk
```
- `k` = size of your independent set
- second line lists exactly `k` distinct vertex indices (space-separated)
- vertices may be printed in any order

**What will be checked**
1. Output format is valid.
2. Every printed vertex is in range.
3. No edge exists between any two printed vertices.
4. `k` is the true maximum for the input graph.

**Requirements**
1. Use standard C++ (C++17-compatible).
2. Read from stdin and write to stdout.
3. Handle small and very large graphs.
4. Finish within the test timeout.

**Environment**
{describe_this_pc()}

**C++ Compiler**
{CppCompiler("test_engine").describe()}

Write complete, compilable C++ code with `main()`.
"""


# List of subpasses to grade the single answer against all difficulty levels


extraGradeAnswerRuns = list(range(len(TEST_CASES)))

structure = {
  "type": "object",
  "properties": {
    "reasoning": {
      "type": "string",
      "description": "Explain your approach for finding maximum independent sets"
    },
    "cpp_code": {
      "type": "string",
      "description": "Complete C++ code with main() function that handles all scales"
    }
  },
  "required": ["reasoning", "cpp_code"],
  "additionalProperties": False
}


def _parse_output(stdout: str) -> Tuple[int, List[int]]:
  lines = [line.strip() for line in stdout.strip().splitlines() if line.strip()]
  if not lines:
    raise ValueError("No output")

  size_tokens = lines[0].split()
  if len(size_tokens) != 1:
    raise ValueError("First line must contain exactly one integer")
  set_size = int(size_tokens[0])

  vertices: List[int] = []
  for line in lines[1:]:
    vertices.extend(int(tok) for tok in line.split())

  if set_size < 0:
    raise ValueError("Set size cannot be negative")
  if len(vertices) != set_size:
    raise ValueError(f"Expected {set_size} vertices, got {len(vertices)}")
  return set_size, vertices


def verify_independent_set(num_vertices: int, edges: Iterable[Tuple[int, int]],
                           ind_vertices: List[int],
                           expected_size: Optional[int] = None) -> Tuple[bool, str]:
  if expected_size is not None and len(ind_vertices) != expected_size:
    return False, f"Set has size {len(ind_vertices)} but optimum is {expected_size}"

  chosen = bytearray(num_vertices)
  for v in ind_vertices:
    if v < 0 or v >= num_vertices:
      return False, f"Invalid vertex {v}"
    if chosen[v]:
      return False, f"Duplicate vertex {v}"
    chosen[v] = 1

  for u, v in edges:
    if chosen[u] and chosen[v]:
        return False, f"Edge ({u},{v}) in independent set"

  return True, "Valid maximum independent set"


def _build_mis_viz(num_vertices: int, edges: List[Tuple[int, int]], chosen_vertices: List[int],
                   solution_vertices: int, valid: bool, msg: str) -> dict:
  chosen = sorted(set(chosen_vertices))
  chosen_set = set(chosen)
  truth_set = set(range(solution_vertices))

  missed = sorted(v for v in truth_set if v not in chosen_set)
  extras = sorted(v for v in chosen_set if v >= solution_vertices)
  wrong_solution_nodes = sorted(v for v in chosen_set if v < solution_vertices and v not in truth_set)

  conflict_edges = []
  for u, v in edges:
    if u in chosen_set and v in chosen_set:
      conflict_edges.append((u, v))

  return {
    "num_vertices": num_vertices,
    "edges": edges,
    "chosen": chosen,
    "solution_vertices": solution_vertices,
    "missed": missed,
    "extras": extras,
    "wrong_solution_nodes": wrong_solution_nodes,
    "conflict_edges": conflict_edges,
    "valid": valid,
    "msg": msg,
  }


def gradeAnswer(result: dict, subPass: int, aiEngineName: str) -> tuple:
  if not result or "cpp_code" not in result:
    return 0.0, "No C++ code provided"

  cache_parts = _cache_key_parts(result, subPass)
  cached = _grade_cache.get_grade(*cache_parts)
  if cached is not None:
    return cached

  case = TEST_CASES[subPass]
  solution_vertices, base_degree, num_vertices, _ = _get_case_params(subPass)
  use_streaming = _should_use_streaming(subPass)

  compiler = CppCompiler(aiEngineName)
  if not compiler.find_compiler():
    return 0.0, "No C++ compiler found"

  try:
    exe_path = compiler.compile(result["cpp_code"])
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
      num_vertices, edges, _ = get_graph(subPass)
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
      set_size, ind_vertices = _parse_output(stdout)
    except Exception as e:
      return 0.0, f"[{case['desc']}] Invalid output format: {str(e)[:120]}"

    # Cheap early fail before scanning edges
    if set_size != solution_vertices:
      valid = False
      msg = f"Reported size {set_size}, but optimum is {solution_vertices}"
    else:
      valid, msg = verify_independent_set(
        num_vertices,
        _iter_graph_edges(solution_vertices, base_degree, RANDOM_SEED + subPass),
        ind_vertices,
        expected_size=solution_vertices,
      )

    if num_vertices <= 500:
      _, viz_edges, _ = get_graph(subPass)
      LAST_MIS_VIZ[(subPass, aiEngineName)] = _build_mis_viz(
        num_vertices, viz_edges, ind_vertices, solution_vertices, valid, msg
      )

    if not valid:
      grade = (0.0, f"[{case['desc']}] FAIL: {msg}")
      _grade_cache.put_grade(grade, *cache_parts)
      return grade

    grade = (1.0,
             f"[{case['desc']}] PASS: maximum independent set size {solution_vertices} in {exec_time:.2f}s")
    _grade_cache.put_grade(grade, *cache_parts)
    return grade

  except subprocess.TimeoutExpired:
    grade = (0.1, f"[{case['desc']}] Timeout")
    _grade_cache.put_grade(grade, *cache_parts)
    return grade
  except ExecutionError as e:
    grade = (0.1, f"[{case['desc']}] {str(e)[:100]}")
    _grade_cache.put_grade(grade, *cache_parts)
    return grade
  except Exception as e:
    grade = (0.0, f"[{case['desc']}] Error: {str(e)[:100]}")
    _grade_cache.put_grade(grade, *cache_parts)
    return grade


def resultToNiceReport(result: dict, subPass: int, aiEngineName: str) -> str:
  if not result:
    return "<p style='color:red'>No result provided</p>"
  cache_parts = _cache_key_parts(result, subPass)
  cached = _grade_cache.get_report(*cache_parts)
  if cached is not None:
    return cached
  case = TEST_CASES[subPass]
  solution_vertices, _, num_vertices, estimated_edges = _get_case_params(subPass)
  html = f"<h4>Maximum Independent Set - {case['desc']}</h4>"
  html += (
    f"<p style='font-size:12px;color:#475569;margin:6px 0;'>"
    f"Graph: {num_vertices:,} vertices, ~{estimated_edges:,} edges, planted optimum size = {solution_vertices:,}</p>"
  )
  if "reasoning" in result:
    r = result['reasoning'][:400] + ('...' if len(result.get('reasoning', '')) > 400 else '')
    html += f"<p><strong>Approach:</strong> {r.replace('<', '&lt;').replace('>', '&gt;')}</p>"
  if "cpp_code" in result:
    code = result["cpp_code"].replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    html += f"<details><summary>View C++ Code ({len(result['cpp_code'])} chars)</summary><pre>{code}</pre></details>"
  viz = LAST_MIS_VIZ.get((subPass, aiEngineName))
  if viz and viz.get("num_vertices", 0) <= 500:
    html += _generate_mis_svg(viz)
  _grade_cache.put_report(html, *cache_parts)
  return html


def _generate_mis_svg(viz: dict) -> str:
  num_vertices = viz["num_vertices"]
  edges = viz["edges"]
  chosen = set(viz["chosen"])
  solution_vertices = viz["solution_vertices"]
  truth = set(range(solution_vertices))
  missed = set(viz["missed"])
  extras = set(viz["extras"])
  conflict_edges = set(tuple(sorted(e)) for e in viz["conflict_edges"])

  size = 620
  cx = cy = size / 2.0
  inner_r = size * 0.23
  outer_r = size * 0.43

  positions: List[Tuple[float, float]] = [(0.0, 0.0)] * num_vertices
  for i in range(solution_vertices):
    angle = 2 * math.pi * i / max(1, solution_vertices)
    x = cx + inner_r * math.cos(angle)
    y = cy + inner_r * math.sin(angle)
    positions[i] = (x, y)

  subdiv_count = max(1, num_vertices - solution_vertices)
  for idx, v in enumerate(range(solution_vertices, num_vertices)):
    angle = 2 * math.pi * idx / subdiv_count
    x = cx + outer_r * math.cos(angle)
    y = cy + outer_r * math.sin(angle)
    positions[v] = (x, y)

  if num_vertices <= 80:
    node_r = 5.0
  elif num_vertices <= 200:
    node_r = 3.8
  else:
    node_r = 2.8

  edge_lines = []
  for u, v in edges:
    x1, y1 = positions[u]
    x2, y2 = positions[v]
    is_conflict = tuple(sorted((u, v))) in conflict_edges
    both_subdiv = u >= solution_vertices and v >= solution_vertices
    if is_conflict:
      stroke = "#ef4444"
      width = "1.5"
      opacity = "0.95"
    elif both_subdiv:
      stroke = "#1d4ed8"
      width = "0.55"
      opacity = "0.18"
    else:
      stroke = "#334155"
      width = "0.65"
      opacity = "0.3"
    edge_lines.append(
      f"<line x1='{x1:.2f}' y1='{y1:.2f}' x2='{x2:.2f}' y2='{y2:.2f}' "
      f"stroke='{stroke}' stroke-width='{width}' stroke-opacity='{opacity}' />"
    )

  node_circles = []
  for v, (x, y) in enumerate(positions):
    if v in truth and v in chosen:
      fill = "#22c55e"  # correct solution vertex chosen
      stroke = "#14532d"
    elif v in missed:
      fill = "#ef4444"  # missed solution vertex
      stroke = "#7f1d1d"
    elif v in extras:
      fill = "#f59e0b"  # chosen subdivision vertex (incorrect)
      stroke = "#78350f"
    elif v < solution_vertices:
      fill = "#94a3b8"  # unchosen solution vertex
      stroke = "#334155"
    else:
      fill = "#334155"  # subdivision vertex
      stroke = "#0f172a"
    node_circles.append(
      f"<circle cx='{x:.2f}' cy='{y:.2f}' r='{node_r:.2f}' fill='{fill}' stroke='{stroke}' stroke-width='0.7' />"
    )

  status = "Valid maximum independent set" if viz.get("valid") else viz.get("msg", "Invalid set")
  svg = "\n".join([
    "<div style='margin:12px 0;padding:10px;border:1px solid #1f2937;border-radius:8px;background:#0b1120;'>",
    f"<div style='color:#e2e8f0;font-size:13px;margin-bottom:6px;'><strong>Graph Visualization</strong> — {status}</div>",
    f"<div style='color:#94a3b8;font-size:12px;margin-bottom:6px;'>"
    f"Vertices: {num_vertices} | Solution vertices: {solution_vertices} | Chosen: {len(chosen)} | "
    f"Missed: {len(missed)} | Extras: {len(extras)} | Conflicts: {len(conflict_edges)}</div>",
    f"<svg width='100%' viewBox='0 0 {size} {size}' style='background:#0b1120;border:1px solid #334155;border-radius:6px;'>",
    "<g>",
    *edge_lines,
    "</g>",
    "<g>",
    *node_circles,
    "</g>",
    "</svg>",
    "<div style='color:#64748b;font-size:11px;margin-top:6px;'>"
    "Green nodes are correctly chosen solution vertices. Red nodes are missed solution vertices. "
    "Amber nodes are incorrectly chosen subdivision vertices. Red edges mark independence violations.</div>",
    "</div>",
  ])
  return svg


highLevelSummary = """
<p>Find the largest group of nodes in a graph such that no two nodes in the group
are connected by an edge. Think of seating people at a dinner party where certain
pairs don&rsquo;t get along &mdash; seat as many guests as possible with no
feuding neighbours at the same table.</p>
<p>This is NP-hard. The test uses planted instances with a known optimal answer for
strict pass/fail grading. Subpasses increase the graph size.</p>
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
