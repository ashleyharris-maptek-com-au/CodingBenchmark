"""
Test 28: Minimum Vertex Cover (Rust Implementation)

The LLM must write Rust code to find the minimum set of vertices such that
every edge has at least one endpoint in the set. This is NP-Hard.

Subpasses increase graph complexity, requiring algorithms like branch-and-bound,
kernelization, or approximation with local search refinement.

Solver times out after 5 minutes.
"""

import random
import subprocess
import time
from typing import List, Tuple, Set, Dict, Any
from native_compiler import RustCompiler, CompilationError, ExecutionError
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
  cache_key = f"graph28|v={case['vertices']}|e={case['edges']}|seed={RANDOM_SEED + subpass}"

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

You must write a Rust solver that can handle ANY graph complexity from trivial to ludicrous scale:
- **Trivial**: Small graphs (10-20 vertices), exact algorithms, brute force
- **Medium**: Moderate graphs (50-100 vertices), branch-and-bound, kernelization
- **Large**: Complex graphs (200-500 vertices), advanced heuristics, approximation
- **Extreme**: Massive graphs (1000+ vertices), very fast heuristics, local search

**The Challenge:**
Your Rust vertex cover solver will be tested with graphs ranging from simple to massive instances. The same algorithm must work efficiently across ALL graph complexities.

**Problem:**
Find the minimum set of vertices such that every edge has at least one endpoint in the set. This is NP-Hard and requires sophisticated algorithms for larger instances.

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

**Critical Requirements:**
1. **Scalability**: Your algorithm must adapt based on graph size and density
2. **Performance**: Must complete within 5 minutes even for massive graphs
3. **Quality**: Find minimum or near-minimum vertex covers

**Algorithm Strategy Recommendations:**
- **Small graphs (≤50 vertices)**: Can use exact branch-and-bound algorithms
- **Medium graphs (50-200 vertices)**: Kernelization + branch-and-bound
- **Large graphs (200-1000 vertices)**: 2-approximation algorithms, local search
- **Very Large graphs (>1000 vertices)**: Very fast heuristics, greedy approaches

**Key Techniques:**
- **Branch-and-bound**: Systematic search with pruning
- **Kernelization**: Reduce problem size while preserving optimality
- **2-approximation**: Greedy algorithm with guaranteed factor 2
- **Local search**: Hill climbing, simulated annealing for refinement
- **LP relaxation**: Linear programming relaxation for bounds

**Implementation Hints:**
- Detect graph complexity and choose appropriate algorithm
- Use efficient adjacency list representation
- Implement adaptive quality vs speed tradeoffs
- For very large graphs, focus on fast approximation algorithms
- Handle edge cases: empty graphs, complete graphs, bipartite graphs
- Use fast I/O for large inputs

**Success Criteria:**
- All edges are covered by selected vertices
- Vertex set size is minimized (or near-minimized for large graphs)
- Output format is correct

**Failure Criteria:**
- Not all edges are covered
- Timeout without solution
- Invalid output format

**Requirements:**
1. Program must compile with rustc (edition 2021)
2. Read from stdin, write to stdout
3. Handle variable graph sizes and densities
4. Complete within 5 minutes
5. Must handle varying graph complexities efficiently

Write complete, compilable Rust code with a main function.
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


def greedy_cover_size(edges: List[Tuple[int, int]]) -> int:
  """Upper bound using greedy 2-approximation."""
  cover = set()
  for u, v in edges:
    if u not in cover and v not in cover:
      cover.add(u)
      cover.add(v)
  return len(cover)


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
      if case["edges"] > 10_000_000:
        if len(cover) == cover_size and all(0 <= v < case["vertices"] for v in cover):
          return 0.8, f"[{case['desc']}] Cover size {cover_size} in {exec_time:.2f}s (verification skipped)"
        else:
          return 0.2, f"[{case['desc']}] Invalid output format"
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

    num_vertices, edges = get_graph(subPass)
    valid, msg = verify_cover(num_vertices, edges, cover)
    if not valid:
      return 0.2, f"[{case['desc']}] {msg}"

    greedy_size = greedy_cover_size(edges)
    ratio = len(cover) / greedy_size if greedy_size > 0 else 1.0
    score = min(1.0, 1.5 - ratio)

    return max(
      0.5,
      score), f"[{case['desc']}] Cover size {len(cover)} (greedy: {greedy_size}), {exec_time:.2f}s"

  except subprocess.TimeoutExpired:
    return 0.1, f"[{case['desc']}] Timeout"
  except ExecutionError as e:
    return 0.1, f"[{case['desc']}] {str(e)[:100]}"
  except Exception as e:
    return 0.0, f"[{case['desc']}] Error: {str(e)[:100]}"


def resultToNiceReport(result: dict, subPass: int, aiEngineName: str) -> str:
  if not result:
    return "<p style='color:red'>No result provided</p>"
  case = TEST_CASES[subPass]
  html = f"<h4>Minimum Vertex Cover - {case['desc']}</h4>"
  if "reasoning" in result:
    r = result['reasoning'][:400] + ('...' if len(result.get('reasoning', '')) > 400 else '')
    html += f"<p><strong>Approach:</strong> {r.replace('<', '&lt;').replace('>', '&gt;')}</p>"
  if "rust_code" in result:
    code = result["rust_code"].replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    html += f"<details><summary>View Rust Code ({len(result['rust_code'])} chars)</summary><pre>{code}</pre></details>"
  return html


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
