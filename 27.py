"""
Test 27: Graph Coloring (C++ Implementation)

The LLM must write C++ code to color a graph with exactly k colors such that
no two adjacent vertices share the same color. This is NP-Complete.

Subpasses increase graph size and decrease available colors, requiring
sophisticated algorithms like DSatur, backtracking with pruning, or
constraint propagation.

Solver times out after 5 minutes.
"""

import random
import subprocess
import sys
import os
import time
import math
import hashlib
from typing import List, Tuple, Set, Dict, Optional, Any
from pathlib import Path

from native_compiler import CppCompiler, CompilationError, ExecutionError,describe_this_pc
from solver_utils import StreamingInputFile

title = "Graph Coloring (C++)"

tags = [
  "cpp",
  "structured response",
  "np hard",
  "graph theory",
]
TIMEOUT_SECONDS = 30
RANDOM_SEED = 27272727


def _edge_generator(num_vertices: int, edge_probability: float, seed: int):
  """Yield edges (u,v) for an Erdos-Renyi graph using geometric skip.
  O(m) time, O(1) memory.  Based on Batagelj & Brandes (2005)."""
  n = num_vertices
  p = edge_probability
  if p <= 0 or n < 2:
    return
  if p >= 1.0:
    for i in range(n):
      for j in range(i + 1, n):
        yield (i, j)
    return
  rng = random.Random(seed)
  lp = math.log(1.0 - p)
  v = 1
  w = -1
  while v < n:
    r = rng.random()
    if r <= 0:
      r = 1e-300
    w = w + 1 + int(math.log(r) / lp)
    while w >= v and v < n:
      w -= v
      v += 1
    if v < n:
      yield (w, v)


def _edge_count(num_vertices: int, edge_probability: float, seed: int) -> int:
  """Count edges without storing them.  Same RNG sequence as _edge_generator."""
  n = num_vertices
  p = edge_probability
  if p <= 0 or n < 2:
    return 0
  if p >= 1.0:
    return n * (n - 1) // 2
  rng = random.Random(seed)
  lp = math.log(1.0 - p)
  count = 0
  v = 1
  w = -1
  while v < n:
    r = rng.random()
    if r <= 0:
      r = 1e-300
    w = w + 1 + int(math.log(r) / lp)
    while w >= v and v < n:
      w -= v
      v += 1
    if v < n:
      count += 1
  return count


def generate_graph(num_vertices: int, edge_probability: float, seed: int) -> List[Tuple[int, int]]:
  """Generate random graph with given edge probability.  O(m) via geometric skip."""
  return list(_edge_generator(num_vertices, edge_probability, seed))


def chromatic_number_upper_bound(num_vertices: int, edges: List[Tuple[int, int]]) -> int:
  """Estimate upper bound on chromatic number using greedy."""
  adj = [set() for _ in range(num_vertices)]
  for u, v in edges:
    adj[u].add(v)
    adj[v].add(u)

  colors = [-1] * num_vertices
  for v in range(num_vertices):
    neighbor_colors = {colors[n] for n in adj[v] if colors[n] >= 0}
    c = 0
    while c in neighbor_colors:
      c += 1
    colors[v] = c
  return max(colors) + 1 if colors else 1


TEST_CASES = [
  {
    "vertices": 20,
    "edge_prob": 0.3,
    "colors": 4,
    "desc": "20 vertices, 4 colors"
  },
  {
    "vertices": 50,
    "edge_prob": 0.25,
    "colors": 5,
    "desc": "50 vertices, 5 colors"
  },
  {
    "vertices": 100,
    "edge_prob": 0.15,
    "colors": 6,
    "desc": "100 vertices, 6 colors"
  },
  {
    "vertices": 200,
    "edge_prob": 0.1,
    "colors": 7,
    "desc": "200 vertices, 7 colors"
  },
  {
    "vertices": 500,
    "edge_prob": 0.05,
    "colors": 8,
    "desc": "500 vertices, 8 colors"
  },
  {
    "vertices": 1000,
    "edge_prob": 0.03,
    "colors": 10,
    "desc": "1K vertices, 10 colors"
  },
  {
    "vertices": 2000,
    "edge_prob": 0.02,
    "colors": 12,
    "desc": "2K vertices, 12 colors"
  },
  {
    "vertices": 5000,
    "edge_prob": 0.01,
    "colors": 15,
    "desc": "5K vertices, 15 colors"
  },
  {
    "vertices": 10000,
    "edge_prob": 0.005,
    "colors": 18,
    "desc": "10K vertices, 18 colors"
  },
  {
    "vertices": 25000,
    "edge_prob": 0.002,
    "colors": 20,
    "desc": "25K vertices, 20 colors"
  },
  {
    "vertices": 50000,
    "edge_prob": 0.001,
    "colors": 25,
    "desc": "50K vertices, 25 colors"
  },
  # Ludicrous cases for streaming
  {
    "vertices": 100000,
    "edge_prob": 0.0008,
    "colors": 30,
    "desc": "100K vertices, 30 colors (~4M edges)"
  },
  {
    "vertices": 500000,
    "edge_prob": 0.0002,
    "colors": 40,
    "desc": "500K vertices, 40 colors (~25M edges)"
  },
  {
    "vertices": 1000000,
    "edge_prob": 0.0001,
    "colors": 50,
    "desc": "1M vertices, 50 colors (~50M edges)"
  },
  {
    "vertices": 2000000,
    "edge_prob": 0.00008,
    "colors": 60,
    "desc": "2M vertices, 60 colors (~160M edges, >1GB)"
  },
]

GRAPH_CACHE: Dict[int, Any] = {}
_INPUT_FILE_CACHE: Dict[int, StreamingInputFile] = {}
LAST_COLORING_VIZ: Dict[Tuple[int, str], dict] = {}

# Threshold for streaming (file-based) input
STREAMING_THRESHOLD_EDGES = 1_000_000

# Persistent cache directory in repo
_PERSISTENT_CACHE_DIR = Path(__file__).parent / "cached_graphs" / "test27"


def _get_persistent_cache_path(cache_key: str) -> Path:
  """Get path to persistent cache file in repo."""
  key_hash = hashlib.sha256(cache_key.encode('utf-8')).hexdigest()[:24]
  return _PERSISTENT_CACHE_DIR / f"{key_hash}.txt"


def _ensure_persistent_cache_dir():
  """Ensure persistent cache directory exists."""
  _PERSISTENT_CACHE_DIR.mkdir(parents=True, exist_ok=True)


def get_graph(subpass: int) -> Tuple[int, List[Tuple[int, int]], int]:
  if subpass not in GRAPH_CACHE:
    case = TEST_CASES[subpass]
    edges = generate_graph(case["vertices"], case["edge_prob"], RANDOM_SEED + subpass)
    GRAPH_CACHE[subpass] = (case["vertices"], edges, case["colors"])
  return GRAPH_CACHE[subpass]


def _estimate_edges(subpass: int) -> int:
  """Estimate number of edges for a test case."""
  case = TEST_CASES[subpass]
  n = case["vertices"]
  p = case["edge_prob"]
  return int(n * (n - 1) / 2 * p)


def _should_use_streaming(subpass: int) -> bool:
  return _estimate_edges(subpass) > STREAMING_THRESHOLD_EDGES


def _get_streaming_input(subpass: int) -> StreamingInputFile:
  if subpass in _INPUT_FILE_CACHE:
    return _INPUT_FILE_CACHE[subpass]

  case = TEST_CASES[subpass]
  cache_key = f"graph27v2|v={case['vertices']}|p={case['edge_prob']}|c={case['colors']}|seed={RANDOM_SEED + subpass}"

  # Check persistent cache first
  persistent_path = _get_persistent_cache_path(cache_key)
  if persistent_path.exists():
    # Create a StreamingInputFile that points to the persistent cache
    class PersistentCacheFile:

      def __init__(self, path):
        self._path = path
        self._is_generated = True

      def get_file_path(self):
        return self._path

      def open_for_read(self):
        return open(self._path, 'r', encoding='utf-8')

      def get_size_bytes(self):
        return self._path.stat().st_size

    _INPUT_FILE_CACHE[subpass] = PersistentCacheFile(persistent_path)
    return _INPUT_FILE_CACHE[subpass]

  def generator():
    n = case["vertices"]
    p = case["edge_prob"]
    nc = case["colors"]
    s = RANDOM_SEED + subpass
    # First pass: count edges (O(m) time, O(1) memory)
    t0 = time.time()
    edge_count = _edge_count(n, p, s)
    count_time = time.time() - t0
    if count_time > 1:
      print(f"  Edge counting: {count_time:.2f}s")
    yield f"{n} {edge_count} {nc}\n"
    # Second pass: stream edges (same RNG sequence reproduces same edges)
    for u, v in _edge_generator(n, p, s):
      yield f"{u} {v}\n"

  input_file = StreamingInputFile(cache_key, generator, "test27_graphs")
  _INPUT_FILE_CACHE[subpass] = input_file

  # After generation, copy to persistent cache
  def copy_to_persistent():
    try:
      _ensure_persistent_cache_dir()
      temp_path = input_file.get_file_path()
      if temp_path.exists():
        import shutil
        shutil.copy2(temp_path, persistent_path)
        print(f"  Cached to repo: {persistent_path.name}")
    except Exception as e:
      print(f"  Warning: Failed to copy to persistent cache: {e}")

  # Monkey-patch generate to also copy to persistent cache
  original_generate = input_file.generate

  def generate_with_copy(force=False):
    result = original_generate(force)
    copy_to_persistent()
    return result

  input_file.generate = generate_with_copy

  return input_file


def format_input(num_vertices: int, edges: List[Tuple[int, int]], num_colors: int) -> str:
  lines = [f"{num_vertices} {len(edges)} {num_colors}"]
  for u, v in edges:
    lines.append(f"{u} {v}")
  return "\n".join(lines)


def prepareSubpassPrompt(subPass: int) -> str:
  """Generate the prompt for subpass 0 that handles all coloring complexities."""
  if subPass != 0:
    raise StopIteration

  return f"""You are writing C++ code to solve the Graph Coloring problem.

**The Challenge:**
Your C++ graph colorer will be tested with graphs ranging from simple 50 nodes  
to massive million-node graphs requiring 60 colours.

**Problem:**
Color a graph with exactly k colours such that no two adjacent vertices share the same colour. 
This is NP-Complete and requires sophisticated algorithms for larger instances.

**Input format (stdin):**
```
num_vertices num_edges num_colors
u v  (for each edge, 0-indexed)
```

**Output format (stdout):**
```
color_0 color_1 ... color_(n-1)  (color assignment for each vertex)
```

**Success Criteria:**
- All vertices assigned valid colors (0 to k-1)
- No adjacent vertices share the same color
- All vertices are colored (complete coloring)

**Failure Criteria:**
- Cannot find valid coloring within time limit
- Invalid color assignment (adjacent vertices same color)
- Incomplete coloring

**Environment:**
{describe_this_pc()}

**C++ Compiler:**
{CppCompiler("test_engine").describe()}

Be sure that any deviation from the C++ standard library is supported by the given compiler,
as referencing the wrong intrinsics or non-standard header like 'bits/stdc++.h' could fail your submission.

**Requirements:**
1. Program must compile with g++ or MSVC (C++17)
2. Read from stdin, write to stdout
3. Handle variable graph sizes and color counts
4. Complete within 5 minutes
5. Must handle varying graph complexities efficiently

Write complete, compilable C++ code with a main() function.
Include adaptive logic that chooses different strategies based on graph complexity.
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
      "Explain your algorithm approach and how it adapts to different coloring complexities"
    },
    "cpp_code": {
      "type": "string",
      "description": "Complete C++ code with main() function that handles all scales"
    }
  },
  "required": ["reasoning", "cpp_code"],
  "additionalProperties": False
}


def verify_coloring(num_vertices: int, edges: List[Tuple[int, int]], colors: List[int],
                    num_colors: int) -> Tuple[bool, str]:
  if len(colors) != num_vertices:
    return False, f"Wrong number of colors: {len(colors)} vs {num_vertices}"

  for c in colors:
    if c < 0 or c >= num_colors:
      return False, f"Invalid color {c}, must be in [0, {num_colors-1}]"

  for u, v in edges:
    if colors[u] == colors[v]:
      return False, f"Adjacent vertices {u} and {v} have same color {colors[u]}"

  return True, "Valid coloring"


def _build_coloring_viz(num_vertices: int, edges: List[Tuple[int, int]], colors: List[int],
                        num_colors: int, valid: bool, msg: str) -> dict:
  colors_list: List[Optional[int]] = []
  for i in range(num_vertices):
    colors_list.append(colors[i] if i < len(colors) else None)

  invalid_nodes: List[int] = []
  for i, c in enumerate(colors_list):
    if c is None or c < 0 or c >= num_colors:
      invalid_nodes.append(i)

  invalid_edges: List[Tuple[int, int]] = []
  for u, v in edges:
    cu = colors_list[u]
    cv = colors_list[v]
    if cu is None or cv is None:
      continue
    if cu < 0 or cu >= num_colors or cv < 0 or cv >= num_colors:
      continue
    if cu == cv:
      invalid_edges.append((u, v))

  return {
    "num_vertices": num_vertices,
    "edges": edges,
    "colors": colors_list,
    "num_colors": num_colors,
    "invalid_nodes": invalid_nodes,
    "invalid_edges": invalid_edges,
    "valid": valid,
    "msg": msg,
  }


def gradeAnswer(result: dict, subPass: int, aiEngineName: str) -> tuple:
  if not result or "cpp_code" not in result:
    return 0.0, "No C++ code provided"

  case = TEST_CASES[subPass]
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
      # Large case: use file-based streaming input
      t = time.time()
      streaming_input = _get_streaming_input(subPass)
      print(f"  Generating/caching input file for {case['desc']}...")
      input_file_path = streaming_input.generate()
      file_size_mb = streaming_input.get_size_bytes() / (1024 * 1024)
      print(f"  Input file: {file_size_mb:.1f} MB")
      generate_time = time.time() - t
      if generate_time > 1:
        print(f"  Time to generate: {generate_time:.2f}s")

      start = time.time()
      stdout, stderr, exec_time, return_code = compiler.execute(exe_path,
                                                                timeout=TIMEOUT_SECONDS,
                                                                stdin_file=input_file_path)

      if return_code != 0:
        return 0.0, f"Runtime error: {stderr[:200]}"

      colors = list(map(int, stdout.strip().split()))
      # Skip full verification for very large graphs
      estimated_edges = _estimate_edges(subPass)
      if estimated_edges > 10_000_000:
        if len(colors) == case["vertices"] and all(0 <= c < case["colors"] for c in colors):
          return 1.0, f"[{case['desc']}] Coloring produced in {exec_time:.2f}s (verification skipped)"
        else:
          return 0.0, f"[{case['desc']}] Invalid output format"
      else:
        num_vertices, edges, num_colors = get_graph(subPass)
        valid, msg = verify_coloring(num_vertices, edges, colors, num_colors)
    else:
      # Small case: use in-memory string input
      num_vertices, edges, num_colors = get_graph(subPass)
      input_data = format_input(num_vertices, edges, num_colors)

      start = time.time()
      proc = subprocess.run([str(exe_path)],
                            input=input_data,
                            capture_output=True,
                            text=True,
                            timeout=TIMEOUT_SECONDS)
      exec_time = time.time() - start

      if proc.returncode != 0:
        return 0.0, f"Runtime error: {proc.stderr[:200]}"

      colors = list(map(int, proc.stdout.strip().split()))
      valid, msg = verify_coloring(num_vertices, edges, colors, num_colors)

    if case["vertices"] <= 1000:
      LAST_COLORING_VIZ[(subPass, aiEngineName)] = _build_coloring_viz(
        num_vertices, edges, colors, num_colors, valid, msg
      )

    if valid:
      return 1.0, f"[{case['desc']}] Valid coloring in {exec_time:.2f}s"
    else:
      return 0.0, f"[{case['desc']}] {msg}"

  except subprocess.TimeoutExpired:
    return 0.0, f"[{case['desc']}] Timeout"
  except ExecutionError as e:
    return 0.0, f"[{case['desc']}] {str(e)[:100]}"
  except Exception as e:
    return 0.0, f"[{case['desc']}] Error: {str(e)[:100]}"


def resultToNiceReport(result: dict, subPass: int, aiEngineName: str) -> str:
  """Generate HTML report for graph coloring."""
  if not result:
    return "<p style='color:red'>No result provided</p>"
  case = TEST_CASES[subPass]
  html = f"<h4>Graph Coloring - {case['desc']}</h4>"
  if "reasoning" in result and subPass == 0:
    r = result['reasoning'][:400] + ('...' if len(result.get('reasoning', '')) > 400 else '')
    html += f"<p><strong>Approach:</strong> {r.replace('<', '&lt;').replace('>', '&gt;')}</p>"
  if "cpp_code" in result and subPass == 0:
    code = result["cpp_code"].replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    html += f"<details><summary>View C++ Code ({len(result['cpp_code'])} chars)</summary><pre>{code}</pre></details>"
  viz = LAST_COLORING_VIZ.get((subPass, aiEngineName))
  if viz and viz.get("num_vertices", 0) <= 1000:
    html += _generate_graph_svg(viz)
  return html


def _generate_graph_svg(viz: dict) -> str:
  num_vertices = viz["num_vertices"]
  edges = viz["edges"]
  colors = viz["colors"]
  num_colors = max(1, viz["num_colors"])
  invalid_nodes = set(viz["invalid_nodes"])
  invalid_edges = set(tuple(sorted(e)) for e in viz["invalid_edges"])

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

  def color_for(c: Optional[int]) -> str:
    if c is None or c < 0 or c >= num_colors:
      return "#111827"
    hue = (c * 360.0 / num_colors) % 360.0
    return f"hsl({hue:.0f},70%,55%)"

  edge_lines = []
  for u, v in edges:
    x1, y1 = positions[u]
    x2, y2 = positions[v]
    is_bad = tuple(sorted((u, v))) in invalid_edges
    stroke = "#ef4444" if is_bad else "#334155"
    opacity = "0.85" if is_bad else "0.25"
    width = "1.4" if is_bad else "0.6"
    edge_lines.append(
      f"<line x1='{x1:.2f}' y1='{y1:.2f}' x2='{x2:.2f}' y2='{y2:.2f}' "
      f"stroke='{stroke}' stroke-width='{width}' stroke-opacity='{opacity}' />"
    )

  node_circles = []
  for i, (x, y) in enumerate(positions):
    fill = color_for(colors[i])
    if i in invalid_nodes:
      node_circles.append(
        f"<circle cx='{x:.2f}' cy='{y:.2f}' r='{node_r:.2f}' fill='{fill}' "
        f"stroke='#ef4444' stroke-width='1.3' />"
      )
    else:
      node_circles.append(
        f"<circle cx='{x:.2f}' cy='{y:.2f}' r='{node_r:.2f}' fill='{fill}' stroke='#0f172a' stroke-width='0.4' />"
      )

  invalid_edge_count = len(invalid_edges)
  invalid_node_count = len(invalid_nodes)
  status = "Valid coloring" if viz.get("valid") else viz.get("msg", "Invalid coloring")

  svg = "\n".join([
    "<div style='margin:12px 0;padding:10px;border:1px solid #1f2937;border-radius:8px;background:#0b1120;'>",
    f"<div style='color:#e2e8f0;font-size:13px;margin-bottom:6px;'><strong>Graph Visualization</strong> — {status}</div>",
    f"<div style='color:#94a3b8;font-size:12px;margin-bottom:6px;'>"
    f"Nodes: {num_vertices} | Edges: {len(edges)} | Invalid nodes: {invalid_node_count} | Invalid edges: {invalid_edge_count}</div>",
    f"<svg width='100%' viewBox='0 0 {size} {size}' style='background:#0b1120;border:1px solid #334155;border-radius:6px;'>",
    "<g>",
    *edge_lines,
    "</g>",
    "<g>",
    *node_circles,
    "</g>",
    "</svg>",
    "<div style='color:#64748b;font-size:11px;margin-top:6px;'>Red edges highlight same-color conflicts. Red strokes on nodes mark invalid/missing colors.</div>",
    "</div>",
  ])

  return svg


highLevelSummary = """
<p>Colour the nodes of a graph using as few colours as possible, with the rule that
no two connected nodes may share the same colour. This is the same idea as
colouring a map so that no neighbouring countries are the same colour.</p>
<p>Finding the minimum number of colours is NP-hard. Subpasses increase the graph
size and connectivity. The AI&rsquo;s colouring must be valid (no clashes) and
use as few colours as possible.</p>
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
