"""
Test 39: Graph Bisection (Rust Implementation)

The LLM must write Rust code to partition a graph into two equal-sized parts
minimizing the number of edges between them. This is NP-Hard.

Subpasses increase graph size, requiring Kernighan-Lin, spectral methods,
multilevel approaches, or simulated annealing.

Solver times out after 5 minutes.
"""

import random
import subprocess
import time
from typing import List, Tuple, Set, Dict, Any
from native_compiler import RustCompiler, CompilationError, ExecutionError
from solver_utils import StreamingInputFile

title = "Graph Bisection (Rust)"
TIMEOUT_SECONDS = 30
RANDOM_SEED = 39393939


def generate_graph(num_vertices: int, edge_prob: float, seed: int) -> List[Tuple[int, int]]:
  """Generate random graph (must have even vertices)."""
  rng = random.Random(seed)
  edges = []
  for i in range(num_vertices):
    for j in range(i + 1, num_vertices):
      if rng.random() < edge_prob:
        edges.append((i, j))
  return edges


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
]

GRAPH_CACHE: Dict[int, Any] = {}
_INPUT_FILE_CACHE: Dict[int, StreamingInputFile] = {}
STREAMING_THRESHOLD_EDGES = 1_000_000


def get_graph(subpass: int) -> Tuple[int, List[Tuple[int, int]]]:
  if subpass not in GRAPH_CACHE:
    case = TEST_CASES[subpass]
    edges = generate_graph(case["vertices"], case["edge_prob"], RANDOM_SEED + subpass)
    GRAPH_CACHE[subpass] = (case["vertices"], edges)
  return GRAPH_CACHE[subpass]


def _estimate_edges(subpass: int) -> int:
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
  cache_key = f"graph39|v={case['vertices']}|p={case['edge_prob']}|seed={RANDOM_SEED + subpass}"

  def generator():
    num_vertices, edges = get_graph(subpass)
    yield f"{num_vertices} {len(edges)}\n"
    for u, v in edges:
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
  """Generate the prompt for subpass 0 that handles all difficulty levels."""
  if subPass != 0:
    raise StopIteration

  return f"""You are writing RUST code to solve the Shortest Path with Negative Weights problem.

You must write a RUST solver that can handle ANY problem complexity from trivial to ludicrous scale:
- **Trivial**: Simple instances, basic algorithms
- **Medium**: Moderate instances, efficient algorithms  
- **Large**: Complex instances, advanced algorithms
- **Extreme**: Massive instances, very fast heuristics

**The Challenge:**
Your RUST solver will be tested with instances ranging from simple to very complex cases. The same algorithm must work efficiently across ALL problem complexities.

**Problem:**
Find shortest paths in a graph that may contain negative edge weights (but no negative cycles).

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
Small graphs (50 vertices): Bellman-Ford, Medium (500 vertices): SPFA, Large (5000 vertices): very fast implementations, Extreme (50000+ vertices): approximation algorithms

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

Write complete, compilable RUST code.
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
    "rust_code": {
      "type": "string",
      "description": "Complete Rust code with main function that handles all scales"
    }
  },
  "required": ["reasoning", "rust_code"],
  "additionalProperties": False
},


def calculate_cut(num_vertices: int, edges: List[Tuple[int, int]],
                  partition_a: Set[int]) -> Tuple[bool, int, str]:
  """Calculate cut size and validate partition."""
  partition_b = set(range(num_vertices)) - partition_a

  # Check balance
  size_diff = abs(len(partition_a) - len(partition_b))
  if size_diff > 1:
    return False, 0, f"Unbalanced: |A|={len(partition_a)}, |B|={len(partition_b)}"

  # Count cut edges
  cut = 0
  for u, v in edges:
    if (u in partition_a) != (v in partition_a):
      cut += 1

  return True, cut, "Valid"


def random_bisection_cut(num_vertices: int, edges: List[Tuple[int, int]], seed: int) -> int:
  """Random bisection for comparison."""
  rng = random.Random(seed)
  vertices = list(range(num_vertices))
  rng.shuffle(vertices)
  partition_a = set(vertices[:num_vertices // 2])

  cut = 0
  for u, v in edges:
    if (u in partition_a) != (v in partition_a):
      cut += 1
  return cut


def greedy_bisection_cut(num_vertices: int, edges: List[Tuple[int, int]]) -> int:
  """Greedy bisection starting from random, then KL-style improvements."""
  adj = [set() for _ in range(num_vertices)]
  for u, v in edges:
    adj[u].add(v)
    adj[v].add(u)

  # Start with degree-based partition
  degrees = [(len(adj[v]), v) for v in range(num_vertices)]
  degrees.sort(reverse=True)

  partition_a = set()
  for _, v in degrees:
    if len(partition_a) < num_vertices // 2:
      partition_a.add(v)

  # Simple swaps
  for _ in range(100):
    improved = False
    best_swap = None
    best_gain = 0

    for a in list(partition_a)[:20]:
      for b in range(num_vertices):
        if b not in partition_a:
          # Calculate gain of swapping a and b
          gain = 0
          for n in adj[a]:
            if n in partition_a:
              gain += 1
            else:
              gain -= 1
          for n in adj[b]:
            if n not in partition_a:
              gain += 1
            else:
              gain -= 1

          if gain > best_gain:
            best_gain = gain
            best_swap = (a, b)
            improved = True

    if improved and best_swap:
      a, b = best_swap
      partition_a.remove(a)
      partition_a.add(b)
    else:
      break

  cut = 0
  for u, v in edges:
    if (u in partition_a) != (v in partition_a):
      cut += 1
  return cut


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
      reported_cut = int(lines[0])
      partition_a = set(map(int, lines[1].split())) if len(lines) > 1 else set()

      # Skip full verification for very large graphs
      if _estimate_edges(subPass) > 10_000_000:
        if len(partition_a) == case["vertices"] // 2:
          return 0.8, f"[{case['desc']}] Cut {reported_cut} in {exec_time:.2f}s (verification skipped)"
        else:
          return 0.2, f"[{case['desc']}] Invalid partition size"
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
      reported_cut = int(lines[0])
      partition_a = set(map(int, lines[1].split())) if len(lines) > 1 else set()

    num_vertices, edges = get_graph(subPass)
    valid, actual_cut, msg = calculate_cut(num_vertices, edges, partition_a)
    if not valid:
      return 0.2, f"[{case['desc']}] {msg}"

    greedy_cut = greedy_bisection_cut(num_vertices, edges)
    ratio = actual_cut / greedy_cut if greedy_cut > 0 else 1.0
    score = min(1.0, 1.5 - ratio * 0.5)

    return max(0.5,
               score), f"[{case['desc']}] Cut {actual_cut} (greedy: {greedy_cut}), {exec_time:.2f}s"

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
  html = f"<h4>Graph Bisection - {case['desc']}</h4>"
  if "reasoning" in result:
    r = result['reasoning'][:400] + ('...' if len(result.get('reasoning', '')) > 400 else '')
    html += f"<p><strong>Approach:</strong> {r.replace('<', '&lt;').replace('>', '&gt;')}</p>"
  if "rust_code" in result:
    code = result["rust_code"].replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    html += f"<details><summary>View Rust Code ({len(result['rust_code'])} chars)</summary><pre>{code}</pre></details>"
  return html


highLevelSummary = """
Graph Bisection partitions vertices into two equal sets minimizing cut edges.

**Algorithms:**
- **Kernighan-Lin**: Iterative improvement heuristic
- **Spectral Bisection**: Using graph Laplacian eigenvectors
- **METIS**: Multilevel graph partitioning
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
