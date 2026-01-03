"""
Test 37: Feedback Vertex Set (C# Implementation)

The LLM must write C# code to find the minimum set of vertices whose removal
makes the graph acyclic (a DAG or forest). This is NP-Hard.

Subpasses increase graph complexity, requiring iterative compression,
branch-and-bound, or 2-approximation algorithms.

Solver times out after 5 minutes.
"""

import random
import subprocess
import time
from typing import List, Tuple, Set, Dict, Any
from native_compiler import CSharpCompiler, CompilationError, ExecutionError
from solver_utils import StreamingInputFile

title = "Feedback Vertex Set (C#)"
TIMEOUT_SECONDS = 30
RANDOM_SEED = 37373737


def generate_graph(num_vertices: int, num_edges: int, seed: int) -> List[Tuple[int, int]]:
  """Generate directed graph with cycles."""
  rng = random.Random(seed)
  edges = set()

  # Create some cycles
  for _ in range(num_vertices // 3):
    cycle_len = rng.randint(3, min(10, num_vertices))
    start = rng.randint(0, num_vertices - 1)
    for i in range(cycle_len):
      u = (start + i) % num_vertices
      v = (start + i + 1) % num_vertices
      edges.add((u, v))

  # Add random edges
  while len(edges) < num_edges:
    u = rng.randint(0, num_vertices - 1)
    v = rng.randint(0, num_vertices - 1)
    if u != v:
      edges.add((u, v))

  return list(edges)


TEST_CASES = [
  {
    "vertices": 20,
    "edges": 50,
    "desc": "20 vertices, 50 edges"
  },
  {
    "vertices": 40,
    "edges": 120,
    "desc": "40 vertices, 120 edges"
  },
  {
    "vertices": 60,
    "edges": 200,
    "desc": "60 vertices, 200 edges"
  },
  {
    "vertices": 100,
    "edges": 400,
    "desc": "100 vertices, 400 edges"
  },
  {
    "vertices": 150,
    "edges": 700,
    "desc": "150 vertices, 700 edges"
  },
  {
    "vertices": 200,
    "edges": 1000,
    "desc": "200 vertices, 1K edges"
  },
  {
    "vertices": 300,
    "edges": 1800,
    "desc": "300 vertices, 1.8K edges"
  },
  {
    "vertices": 500,
    "edges": 3500,
    "desc": "500 vertices, 3.5K edges"
  },
  {
    "vertices": 750,
    "edges": 6000,
    "desc": "750 vertices, 6K edges"
  },
  {
    "vertices": 1000,
    "edges": 10000,
    "desc": "1K vertices, 10K edges"
  },
  {
    "vertices": 2000,
    "edges": 25000,
    "desc": "2K vertices, 25K edges"
  },
  # Ludicrous cases for streaming
  {
    "vertices": 5000,
    "edges": 100000,
    "desc": "5K vertices, 100K edges"
  },
  {
    "vertices": 10000,
    "edges": 500000,
    "desc": "10K vertices, 500K edges"
  },
  {
    "vertices": 50000,
    "edges": 2000000,
    "desc": "50K vertices, 2M edges"
  },
  {
    "vertices": 100000,
    "edges": 10000000,
    "desc": "100K vertices, 10M edges"
  },
  {
    "vertices": 500000,
    "edges": 50000000,
    "desc": "500K vertices, 50M edges"
  },
  {
    "vertices": 1000000,
    "edges": 150000000,
    "desc": "1M vertices, 150M edges (>1GB)"
  },
]

GRAPH_CACHE: Dict[int, Any] = {}
_INPUT_FILE_CACHE: Dict[int, StreamingInputFile] = {}
STREAMING_THRESHOLD_EDGES = 1_000_000


def get_graph(subpass: int) -> Tuple[int, List[Tuple[int, int]]]:
  if subpass not in GRAPH_CACHE:
    case = TEST_CASES[subpass]
    edges = generate_graph(case["vertices"], case["edges"], RANDOM_SEED + subpass)
    GRAPH_CACHE[subpass] = (case["vertices"], edges)
  return GRAPH_CACHE[subpass]


def _should_use_streaming(subpass: int) -> bool:
  return TEST_CASES[subpass]["edges"] > STREAMING_THRESHOLD_EDGES


def _get_streaming_input(subpass: int) -> StreamingInputFile:
  if subpass in _INPUT_FILE_CACHE:
    return _INPUT_FILE_CACHE[subpass]

  case = TEST_CASES[subpass]
  cache_key = f"graph37|v={case['vertices']}|e={case['edges']}|seed={RANDOM_SEED + subpass}"

  def generator():
    num_vertices, edges = get_graph(subpass)
    yield f"{num_vertices} {len(edges)}\n"
    for u, v in edges:
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
  """Generate the prompt for subpass 0 that handles all difficulty levels."""
  if subPass != 0:
    raise StopIteration

  return f"""You are writing CSHARP code to solve the Set Cover problem.

You must write a CSHARP solver that can handle ANY problem complexity from trivial to ludicrous scale:
- **Trivial**: Simple instances, basic algorithms
- **Medium**: Moderate instances, efficient algorithms  
- **Large**: Complex instances, advanced algorithms
- **Extreme**: Massive instances, very fast heuristics

**The Challenge:**
Your CSHARP solver will be tested with instances ranging from simple to very complex cases. The same algorithm must work efficiently across ALL problem complexities.

**Problem:**
Given a universe of elements and a collection of sets, find the smallest subcollection of sets whose union contains all elements.

**Input format (stdin):**
```
[Input format varies by problem]
```

**Output format (stdout):**
```
[Output format varies by problem]
```

**Critical Requirements:**
1. **Scalability**: Your algorithm must adapt based on problem size and complexity
2. **Performance**: Must complete within 5 minutes even for the largest instances
3. **Quality**: Find optimal or near-optimal solutions

**Algorithm Strategy Recommendations:**
Small instances (50 elements): exact branch-and-bound, Medium (500 elements): greedy approximation, Large (5000 elements): advanced heuristics, Extreme (50000+ elements): very fast heuristics

**Implementation Hints:**
- Detect problem complexity and choose appropriate algorithm
- Use efficient data structures and algorithms
- Implement adaptive quality vs speed tradeoffs
- For very large instances, focus on fast heuristics
- Handle edge cases appropriately
- Use fast I/O for large inputs

**Requirements:**
1. Program must compile with appropriate compiler
2. Read from stdin, write to stdout
3. Handle variable problem sizes
4. Complete within 5 minutes
5. Must handle varying problem complexities efficiently

Write complete, compilable CSHARP code.
Include adaptive logic that chooses different strategies based on problem complexity.
"""# List of subpasses to grade the single answer against all difficulty levels


extraGradeAnswerRuns = list(range(len(TEST_CASES)))

structure = {
  "type": "object",
  "properties": {
    "reasoning": {
      "type": "string",
      "description": "Explain your algorithm approach and how it adapts to different problem sizes"
    },
    "csharp_code": {
      "type": "string",
      "description": "Complete C# code with Main method that handles all scales"
    }
  },
  "required": ["reasoning", "csharp_code"],
  "additionalProperties": False
},


def has_cycle(num_vertices: int, edges: List[Tuple[int, int]], removed: Set[int]) -> bool:
  """Check if graph has cycle after removing vertices."""
  adj = [[] for _ in range(num_vertices)]
  for u, v in edges:
    if u not in removed and v not in removed:
      adj[u].append(v)

  WHITE, GRAY, BLACK = 0, 1, 2
  color = [WHITE] * num_vertices

  def dfs(v):
    if v in removed:
      return False
    color[v] = GRAY
    for u in adj[v]:
      if color[u] == GRAY:
        return True
      if color[u] == WHITE and dfs(u):
        return True
    color[v] = BLACK
    return False

  for v in range(num_vertices):
    if v not in removed and color[v] == WHITE:
      if dfs(v):
        return True
  return False


def greedy_fvs(num_vertices: int, edges: List[Tuple[int, int]]) -> int:
  """Greedy FVS for comparison."""
  removed = set()
  remaining_edges = list(edges)

  while has_cycle(num_vertices, remaining_edges, removed):
    # Count in-degree and out-degree for each vertex
    in_deg = [0] * num_vertices
    out_deg = [0] * num_vertices
    for u, v in remaining_edges:
      if u not in removed and v not in removed:
        out_deg[u] += 1
        in_deg[v] += 1

    # Pick vertex with max min(in, out) - heuristic for cycle involvement
    best_v = -1
    best_score = -1
    for v in range(num_vertices):
      if v not in removed:
        score = min(in_deg[v], out_deg[v])
        if score > best_score:
          best_score = score
          best_v = v

    if best_v >= 0:
      removed.add(best_v)
    else:
      break

  return len(removed)


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
      num_vertices, edges = get_graph(subPass)
      input_data = format_input(num_vertices, edges)
      stdout, stderr, exec_time, retcode = compiler.execute(exe_path, input_data, TIMEOUT_SECONDS)

    if retcode != 0:
      return 0.0, f"Runtime error: {stderr[:200]}"

    lines = stdout.strip().split('\n')
    set_size = int(lines[0])
    fvs = set(map(int, lines[1].split())) if len(lines) > 1 and lines[1].strip() else set()

    # Skip full verification for very large graphs
    if case["edges"] > 10_000_000:
      if len(fvs) == set_size:
        return 0.8, f"[{case['desc']}] FVS size {set_size} in {exec_time:.2f}s (verification skipped)"
      else:
        return 0.2, f"[{case['desc']}] Invalid output format"

    num_vertices, edges = get_graph(subPass)
    for v in fvs:
      if v < 0 or v >= num_vertices:
        return 0.2, f"[{case['desc']}] Invalid vertex {v}"

    if has_cycle(num_vertices, edges, fvs):
      return 0.2, f"[{case['desc']}] Graph still has cycles"

    greedy_size = greedy_fvs(num_vertices, edges)
    ratio = len(fvs) / greedy_size if greedy_size > 0 else 1.0
    score = min(1.0, 1.5 - ratio * 0.5)

    return max(
      0.5, score), f"[{case['desc']}] FVS size {len(fvs)} (greedy: {greedy_size}), {exec_time:.2f}s"

  except ExecutionError as e:
    return 0.1, f"[{case['desc']}] {str(e)[:100]}"
  except Exception as e:
    return 0.0, f"[{case['desc']}] Error: {str(e)[:100]}"


def resultToNiceReport(result: dict, subPass: int, aiEngineName: str) -> str:
  if not result:
    return "<p style='color:red'>No result provided</p>"
  case = TEST_CASES[subPass]
  html = f"<h4>Feedback Vertex Set - {case['desc']}</h4>"
  if "reasoning" in result:
    r = result['reasoning'][:400] + ('...' if len(result.get('reasoning', '')) > 400 else '')
    html += f"<p><strong>Approach:</strong> {r.replace('<', '&lt;').replace('>', '&gt;')}</p>"
  if "csharp_code" in result:
    code = result["csharp_code"].replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    html += f"<details><summary>View C# Code ({len(result['csharp_code'])} chars)</summary><pre>{code}</pre></details>"
  return html


highLevelSummary = """
Feedback Vertex Set finds minimum vertices to remove to make graph acyclic.

**Algorithms:**
- **Greedy**: Remove high-degree vertices in cycles
- **2-Approximation**: For undirected graphs
- **FPT**: Parameterized by solution size
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
