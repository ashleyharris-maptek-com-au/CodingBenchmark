"""
Test 35: Minimum Dominating Set (Rust Implementation)

The LLM must write Rust code to find the smallest set of vertices such that
every vertex is either in the set or adjacent to a vertex in the set. NP-Hard.

Subpasses increase graph size, requiring greedy approximation, ILP,
or branch-and-bound with clever pruning.

Solver times out after 5 minutes.
"""

import random
import subprocess
import time
import math
import hashlib
from typing import List, Tuple, Set, Dict, Any
from native_compiler import RustCompiler, CompilationError, ExecutionError,describe_this_pc
from solver_utils import StreamingInputFile, GradeCache

_grade_cache = GradeCache('test35')

title = "Minimum Dominating Set (Rust)"

tags = [
  "rust",
  "structured response",
  "np hard",
  "graph theory",
]
TIMEOUT_SECONDS = 30
RANDOM_SEED = 35353535


def _generate_planted_meta(num_vertices: int, edge_prob: float, seed: int) -> Tuple[
    List[int], List[int], Dict[int, Set[int]], List[Set[int]]]:
  """Generate planted dominating set metadata (no full edge list)."""
  rng = random.Random(seed)
  if num_vertices <= 0:
    return [], [], {}, []

  dom_frac = max(0.02, min(0.2, edge_prob * 0.8))
  dom_size = max(1, min(num_vertices, int(num_vertices * dom_frac)))
  dom_vertices = rng.sample(range(num_vertices), dom_size)
  dom_set = set(dom_vertices)

  parent = [0] * num_vertices
  extra_dom_neighbors: List[Set[int]] = [set() for _ in range(num_vertices)]
  for v in range(num_vertices):
    if v in dom_set:
      parent[v] = v
    else:
      parent[v] = rng.choice(dom_vertices)
      if rng.random() < 0.35:
        extra = rng.choice(dom_vertices)
        if extra != parent[v]:
          extra_dom_neighbors[v].add(extra)
      if rng.random() < 0.1:
        extra = rng.choice(dom_vertices)
        if extra != parent[v]:
          extra_dom_neighbors[v].add(extra)

  dom_adj: Dict[int, Set[int]] = {v: set() for v in dom_vertices}
  for i in range(len(dom_vertices)):
    u = dom_vertices[i]
    for j in range(i + 1, len(dom_vertices)):
      v = dom_vertices[j]
      if rng.random() < edge_prob:
        dom_adj[u].add(v)
        dom_adj[v].add(u)

  return dom_vertices, parent, dom_adj, extra_dom_neighbors


def _build_edges_from_meta(num_vertices: int, dom_vertices: List[int], parent: List[int],
                           dom_adj: Dict[int, Set[int]],
                           extra_dom_neighbors: List[Set[int]]) -> List[Tuple[int, int]]:
  edges = set()

  for v in range(num_vertices):
    if parent[v] != v:
      a, b = sorted((parent[v], v))
      edges.add((a, b))

  for v in range(num_vertices):
    for d in extra_dom_neighbors[v]:
      a, b = sorted((d, v))
      edges.add((a, b))

  for u in dom_vertices:
    for v in dom_adj.get(u, set()):
      if u < v:
        edges.add((u, v))

  return list(edges)


def generate_graph(num_vertices: int, edge_prob: float, seed: int) -> Tuple[
    List[Tuple[int, int]], List[int], List[int], Dict[int, Set[int]], List[Set[int]]]:
  """Generate a graph with a planted dominating set and mild perturbations."""
  dom_vertices, parent, dom_adj, extra_dom_neighbors = _generate_planted_meta(
    num_vertices, edge_prob, seed
  )
  edges = _build_edges_from_meta(
    num_vertices, dom_vertices, parent, dom_adj, extra_dom_neighbors
  )
  return edges, dom_vertices, parent, dom_adj, extra_dom_neighbors


TEST_CASES = [
  {
    "vertices": 20,
    "edge_prob": 0.3,
    "desc": "20 vertices"
  },
  {
    "vertices": 40,
    "edge_prob": 0.25,
    "desc": "40 vertices"
  },
  {
    "vertices": 60,
    "edge_prob": 0.2,
    "desc": "60 vertices"
  },
  {
    "vertices": 100,
    "edge_prob": 0.15,
    "desc": "100 vertices"
  },
  {
    "vertices": 150,
    "edge_prob": 0.12,
    "desc": "150 vertices"
  },
  {
    "vertices": 200,
    "edge_prob": 0.1,
    "desc": "200 vertices"
  },
  {
    "vertices": 300,
    "edge_prob": 0.08,
    "desc": "300 vertices"
  },
  {
    "vertices": 500,
    "edge_prob": 0.05,
    "desc": "500 vertices"
  },
  {
    "vertices": 1000,
    "edge_prob": 0.03,
    "desc": "1000 vertices"
  },
  {
    "vertices": 2000,
    "edge_prob": 0.02,
    "desc": "2000 vertices"
  },
  {
    "vertices": 5000,
    "edge_prob": 0.01,
    "desc": "5000 vertices"
  },
  # Ludicrous cases for streaming
  {
    "vertices": 10000,
    "edge_prob": 0.008,
    "desc": "10K vertices (~400K edges)"
  },
  {
    "vertices": 50000,
    "edge_prob": 0.002,
    "desc": "50K vertices (~2.5M edges)"
  },
  {
    "vertices": 100000,
    "edge_prob": 0.001,
    "desc": "100K vertices (~5M edges)"
  },
  {
    "vertices": 500000,
    "edge_prob": 0.0004,
    "desc": "500K vertices (~50M edges)"
  },
  {
    "vertices": 1000000,
    "edge_prob": 0.0003,
    "desc": "1M vertices (~150M edges, >1GB)"
  },
  {
    "vertices": 2000000,
    "edge_prob": 0.0003,
    "desc": "2M vertices (~300M edges, >2GB)"
  },
  {
    "vertices": 3000000,
    "edge_prob": 0.0003,
    "desc": "3M vertices (~450M edges, >3GB)"
  },
  {
    "vertices": 4000000,
    "edge_prob": 0.0003,
    "desc": "4M vertices (~600M edges, >4GB)"
  },
  {
    "vertices": 5_000_000,
    "edge_prob": 0.0003,
    "desc": "5M vertices (~750M edges, >5GB)"
  },
  {
    "vertices": 10_000_000,
    "edge_prob": 0.0002,
    "desc": "10M vertices (~2B edges, >10GB)"
  },
]

GRAPH_CACHE: Dict[int, Any] = {}
_INPUT_FILE_CACHE: Dict[int, StreamingInputFile] = {}
LAST_DOM_VIZ: Dict[Tuple[int, str], dict] = {}
STREAMING_THRESHOLD_EDGES = 1_000


def get_graph(subpass: int, include_edges: bool = True) -> Tuple[
    int, List[Tuple[int, int]], List[int], List[int], Dict[int, Set[int]], List[Set[int]]]:
  if subpass not in GRAPH_CACHE:
    case = TEST_CASES[subpass]
    num_vertices = case["vertices"]
    dom_vertices, parent, dom_adj, extra_dom_neighbors = _generate_planted_meta(
      num_vertices, case["edge_prob"], RANDOM_SEED + subpass
    )
    GRAPH_CACHE[subpass] = {
      "num_vertices": num_vertices,
      "edges": None,
      "dom_vertices": dom_vertices,
      "parent": parent,
      "dom_adj": dom_adj,
      "extra_dom_neighbors": extra_dom_neighbors,
    }

  cached = GRAPH_CACHE[subpass]
  if include_edges and cached["edges"] is None:
    cached["edges"] = _build_edges_from_meta(
      cached["num_vertices"],
      cached["dom_vertices"],
      cached["parent"],
      cached["dom_adj"],
      cached["extra_dom_neighbors"],
    )

  return (
    cached["num_vertices"],
    cached["edges"],
    cached["dom_vertices"],
    cached["parent"],
    cached["dom_adj"],
    cached["extra_dom_neighbors"],
  )


def _estimate_edges(subpass: int) -> int:
  num_vertices, _, dom_vertices, _, dom_adj, extra_dom_neighbors = get_graph(
    subpass, include_edges=False
  )
  dom_edge_count = sum(len(dom_adj[v]) for v in dom_vertices) // 2
  extra_count = sum(len(extra_dom_neighbors[v]) for v in range(num_vertices))
  return (num_vertices - len(dom_vertices)) + dom_edge_count + extra_count


def _should_use_streaming(subpass: int) -> bool:
  return _estimate_edges(subpass) > STREAMING_THRESHOLD_EDGES


def _get_streaming_input(subpass: int) -> StreamingInputFile:
  if subpass in _INPUT_FILE_CACHE:
    return _INPUT_FILE_CACHE[subpass]

  case = TEST_CASES[subpass]
  cache_key = f"graph35|v={case['vertices']}|p={case['edge_prob']}|seed={RANDOM_SEED + subpass}"

  def generator():
    num_vertices, _, dom_vertices, parent, dom_adj, extra_dom_neighbors = get_graph(
      subpass, include_edges=False
    )
    dom_edge_count = sum(len(dom_adj[v]) for v in dom_vertices) // 2
    extra_count = sum(len(extra_dom_neighbors[v]) for v in range(num_vertices))
    total_edges = (num_vertices - len(dom_vertices)) + dom_edge_count + extra_count
    yield f"{num_vertices} {total_edges}\n"

    for v in range(num_vertices):
      if parent[v] != v:
        yield f"{parent[v]} {v}\n"

    for u in dom_vertices:
      for v in dom_adj[u]:
        if u < v:
          yield f"{u} {v}\n"

    for v in range(num_vertices):
      for d in extra_dom_neighbors[v]:
        yield f"{d} {v}\n"

  input_file = StreamingInputFile(cache_key, generator, "test35_graphs")
  _INPUT_FILE_CACHE[subpass] = input_file
  return input_file


def format_input(num_vertices: int, edges: List[Tuple[int, int]]) -> str:
  lines = [f"{num_vertices} {len(edges)}"]
  for u, v in edges:
    lines.append(f"{u} {v}")
  return "\n".join(lines)


def prepareSubpassPrompt(subPass: int) -> str:
  """Generate the prompt for subpass 0 that handles all difficulty levels."""
  if subPass != 0:
    raise StopIteration

  return f"""You are writing Rust code to solve the Minimum Dominating Set problem (NP-hard).

**Problem:**
Given an undirected graph, find a small set of vertices S such that every vertex is
either in S or adjacent to at least one vertex in S. Smaller sets score better.

**Input format (stdin):**
```
num_vertices num_edges
u1 v1
u2 v2
...
```
Vertices are 0-indexed.

**Output format (stdout):**
```
set_size
v1 v2 v3 ...
```
Where the second line lists the vertices in the dominating set (0-indexed).

**Requirements:**
1. Read from stdin, write to stdout.
2. Output a valid dominating set.
3. Heuristics are acceptable for large graphs; smaller sets score higher.
4. Must run within 5 minutes.

**Target environment specs**
{describe_this_pc()}

**Rust Compiler information**
{RustCompiler("test_engine").describe()}


Write complete, compilable Rust code with a main() function.
"""# List of subpasses to grade the single answer against all difficulty levels


extraGradeAnswerRuns = list(range(len(TEST_CASES)))

structure = {
  "type": "object",
  "properties": {
    "reasoning": {
      "type": "string",
      "description": "Explain your algorithm approach and how it adapts to different problem sizes"
    },
    "rust_code": {
      "type": "string",
      "description": "Complete Rust code with main function that handles all scales"
    }
  },
  "required": ["reasoning", "rust_code"],
  "additionalProperties": False
}


def verify_dominating_set(num_vertices: int, edges: List[Tuple[int, int]],
                          dom_set: Set[int]) -> Tuple[bool, str]:
  adj = [set() for _ in range(num_vertices)]
  for u, v in edges:
    adj[u].add(v)
    adj[v].add(u)

  for v in dom_set:
    if v < 0 or v >= num_vertices:
      return False, f"Invalid vertex {v}"

  for v in range(num_vertices):
    if v not in dom_set and not any(u in dom_set for u in adj[v]):
      return False, f"Vertex {v} not dominated"

  return True, "Valid"


def greedy_dominating_set(num_vertices: int, edges: List[Tuple[int, int]]) -> int:
  adj = [set() for _ in range(num_vertices)]
  for u, v in edges:
    adj[u].add(v)
    adj[v].add(u)

  dom_set = set()
  dominated = set()

  while len(dominated) < num_vertices:
    best_v = -1
    best_count = -1

    for v in range(num_vertices):
      if v not in dom_set:
        new_dominated = {v} | adj[v]
        count = len(new_dominated - dominated)
        if count > best_count:
          best_count = count
          best_v = v

    if best_v >= 0:
      dom_set.add(best_v)
      dominated.add(best_v)
      dominated.update(adj[best_v])

  return len(dom_set)


def _verify_planted_domination(num_vertices: int, dom_set: Set[int], parent: List[int],
                               extra_dom_neighbors: List[Set[int]]) -> Tuple[bool, str]:
  for v in dom_set:
    if v < 0 or v >= num_vertices:
      return False, f"Invalid vertex {v}"

  for v in range(num_vertices):
    if v in dom_set:
      continue
    if parent[v] in dom_set:
      continue
    if any(d in dom_set for d in extra_dom_neighbors[v]):
      continue
    return False, f"Vertex {v} not dominated"

  return True, "Valid"


def _cache_key_parts(result: dict, subPass: int) -> tuple:
  """Build cache key parts from code hash + test case params."""
  case = TEST_CASES[subPass]
  code = result.get("rust_code", "")
  return (
    hashlib.sha256(code.encode('utf-8')).hexdigest()[:16],
    f"v={case['vertices']}|p={case['edge_prob']}|seed={RANDOM_SEED + subPass}",
  )


def gradeAnswer(result: dict, subPass: int, aiEngineName: str) -> tuple:
  if not result or "rust_code" not in result:
    return 0.0, "No Rust code provided"

  cache_parts = _cache_key_parts(result, subPass)
  cached = _grade_cache.get_grade(*cache_parts)
  if cached is not None:
    return cached

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
        grade = (0.0, f"Runtime error: {stderr[:200]}")
        _grade_cache.put_grade(grade, *cache_parts)
        return grade

      lines = stdout.strip().split('\n')
      set_size = int(lines[0])
      dom_set = set(map(int, lines[1].split())) if len(lines) > 1 else set()

      # Use planted structure for verification on streaming graphs
      num_vertices, _, _, parent, _, extra_dom_neighbors = get_graph(
        subPass, include_edges=False
      )
      if len(dom_set) != set_size:
        grade = (0.0, f"[{case['desc']}] Invalid output format")
        _grade_cache.put_grade(grade, *cache_parts)
        return grade

      t = time.time()
      valid, msg = _verify_planted_domination(
        num_vertices, dom_set, parent, extra_dom_neighbors
      )
      verify_time = time.time() - t
      if verify_time > 1:
        print(f"  _verify_planted_domination time: {verify_time:.2f}s")

      if not valid:
        grade = (0.0, f"[{case['desc']}] {msg}")
        _grade_cache.put_grade(grade, *cache_parts)
        return grade
      grade = (1.0, f"[{case['desc']}] Size {set_size} in {exec_time:.2f}s")
      _grade_cache.put_grade(grade, *cache_parts)
      return grade
    else:
      num_vertices, edges, _, _, _, _ = get_graph(subPass)
      input_data = format_input(num_vertices, edges)

      start = time.time()
      proc = subprocess.run([str(exe_path)],
                            input=input_data,
                            capture_output=True,
                            text=True,
                            timeout=TIMEOUT_SECONDS)
      exec_time = time.time() - start

      if proc.returncode != 0:
        grade = (0.0, f"Runtime error: {proc.stderr[:200]}")
        _grade_cache.put_grade(grade, *cache_parts)
        return grade

      lines = proc.stdout.strip().split('\n')
      set_size = int(lines[0])
      dom_set = set(map(int, lines[1].split())) if len(lines) > 1 else set()

    num_vertices, edges, planted_dom, _, _, _ = get_graph(subPass)

    t = time.time()
    valid, msg = verify_dominating_set(num_vertices, edges, dom_set)
    verify_time = time.time() - t
    if verify_time > 1:
      print(f"  verify_dominating_set time: {verify_time:.2f}s")

    if not valid:
      grade = (0.0, f"[{case['desc']}] {msg}")
      _grade_cache.put_grade(grade, *cache_parts)
      return grade

    if num_vertices <= 150 and not use_streaming:
      LAST_DOM_VIZ[(subPass, aiEngineName)] = _build_dom_viz(
        num_vertices, edges, dom_set, planted_dom, valid, msg
      )

    t = time.time()
    greedy_size = greedy_dominating_set(num_vertices, edges)
    verify_time = time.time() - t
    if verify_time > 1:
      print(f"  greedy_dominating_set time: {verify_time:.2f}s")

    ratio = len(dom_set) / greedy_size if greedy_size > 0 else 1.0
    score = min(1.0, 1.5 - ratio * 0.5)

    grade = (max(0.5, score), f"[{case['desc']}] Size {len(dom_set)} (greedy: {greedy_size}), {exec_time:.2f}s")
    _grade_cache.put_grade(grade, *cache_parts)
    return grade

  except subprocess.TimeoutExpired:
    grade = (0.0, f"[{case['desc']}] Timeout")
    _grade_cache.put_grade(grade, *cache_parts)
    return grade
  except ExecutionError as e:
    grade = (0.0, f"[{case['desc']}] {str(e)[:100]}")
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
  html = f"<h4>Minimum Dominating Set - {case['desc']}</h4>"
  if "reasoning" in result and subPass ==0:
    r = result['reasoning'][:400] + ('...' if len(result.get('reasoning', '')) > 400 else '')
    html += f"<p><strong>Approach:</strong> {r.replace('<', '&lt;').replace('>', '&gt;')}</p>"
  if "rust_code" in result and subPass ==0:
    code = result["rust_code"].replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    html += f"<details><summary>View Rust Code ({len(result['rust_code'])} chars)</summary><pre>{code}</pre></details>"
  viz = LAST_DOM_VIZ.get((subPass, aiEngineName))
  if viz and subPass == 0:
    html += _generate_dom_svg(viz)

  _grade_cache.put_report(html, *cache_parts)
  return html


def _build_dom_viz(num_vertices: int, edges: List[Tuple[int, int]],
                   dom_set: Set[int], planted_dom: List[int], valid: bool, msg: str) -> dict:
  adj = [set() for _ in range(num_vertices)]
  for u, v in edges:
    adj[u].add(v)
    adj[v].add(u)

  dominated = set(dom_set)
  for v in dom_set:
    dominated.update(adj[v])

  return {
    "num_vertices": num_vertices,
    "edges": edges,
    "dom_set": dom_set,
    "planted_dom": planted_dom,
    "dominated": dominated,
    "valid": valid,
    "message": msg,
  }


def _generate_dom_svg(viz: dict) -> str:
  num_vertices = viz["num_vertices"]
  dom_set = viz["dom_set"]
  dominated = viz["dominated"]
  edges = viz["edges"]
  planted_dom = viz["planted_dom"]
  planted_set = set(planted_dom)

  width = 520
  height = 360
  cx = width / 2
  cy = height / 2
  outer_radius = min(width, height) * 0.36
  inner_radius = outer_radius * 0.38

  positions = {}
  planted = planted_dom
  outer_vertices = [i for i in range(num_vertices) if i not in planted_set]

  for idx, v in enumerate(planted):
    angle = (2 * math.pi * idx) / max(1, len(planted))
    positions[v] = (cx + inner_radius * math.cos(angle), cy + inner_radius * math.sin(angle))

  for idx, v in enumerate(outer_vertices):
    angle = (2 * math.pi * idx) / max(1, len(outer_vertices))
    positions[v] = (cx + outer_radius * math.cos(angle), cy + outer_radius * math.sin(angle))

  edge_lines = []
  for u, v in edges:
    x1, y1 = positions[u]
    x2, y2 = positions[v]
    is_planted_edge = u in planted_set or v in planted_set
    stroke = "#22c55e" if is_planted_edge else "#1f2937"
    opacity = "0.65" if is_planted_edge else "0.35"
    width = "1.6" if is_planted_edge else "1"
    edge_lines.append(
      f"<line x1='{x1:.2f}' y1='{y1:.2f}' x2='{x2:.2f}' y2='{y2:.2f}' "
      f"stroke='{stroke}' stroke-width='{width}' opacity='{opacity}'/>"
    )

  node_circles = []
  for i in range(num_vertices):
    x, y = positions[i]
    if i in dom_set:
      fill = "#22c55e"
      r = 7
    elif i in dominated:
      fill = "#facc15"
      r = 5
    else:
      fill = "#94a3b8"
      r = 4
    if i in planted_set:
      r = max(r, 8)
    node_circles.append(
      f"<circle cx='{x:.2f}' cy='{y:.2f}' r='{r}' fill='{fill}' stroke='#0f172a' stroke-width='1'/>"
    )

  status = "VALID" if viz["valid"] else "INVALID"
  status_color = "#22c55e" if viz["valid"] else "#f97316"

  legend = (
    "<g font-family='sans-serif' font-size='11' fill='#cbd5f5'>"
    "<circle cx='20' cy='18' r='7' fill='#22c55e' stroke='#0f172a'/>"
    "<text x='32' y='22'>Dominating vertex (solution)</text>"
    "<circle cx='20' cy='38' r='7' fill='none' stroke='#22c55e' stroke-width='2'/>"
    "<text x='32' y='42'>Planted min set (center)</text>"
    "<circle cx='20' cy='58' r='5' fill='#facc15' stroke='#0f172a'/>"
    "<text x='32' y='62'>Dominated vertex</text>"
    "<circle cx='20' cy='78' r='4' fill='#94a3b8' stroke='#0f172a'/>"
    "<text x='32' y='82'>Undominated</text>"
    "</g>"
  )

  return (
    f"<div style='color:#e2e8f0;font-size:13px;margin-bottom:6px;'>"
    f"<strong>Dominating Set Visualization</strong> &mdash; "
    f"<span style='color:{status_color};'>{status}</span> "
    f"(size {len(dom_set)})</div>"
    f"<div style='color:#94a3b8;font-size:11px;margin-bottom:6px;'>{viz['message']}</div>"
    f"<svg width='{width}' height='{height}' viewBox='0 0 {width} {height}' "
    "preserveAspectRatio='xMidYMid meet' "
    "style='background:#0b1120;border:1px solid #334155;border-radius:6px;display:block;'>"
    + "".join(edge_lines)
    + "".join(node_circles)
    + legend
    + "</svg>"
  )


highLevelSummary = """
<p>Find the smallest set of nodes such that every node in the graph is either in
the set or directly connected to a node in the set. Think of placing Wi-Fi
routers so that every room is within range of at least one router.</p>
<p>This is NP-hard. The test uses planted instances with a known optimal set for
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
