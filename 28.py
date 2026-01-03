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
from typing import List, Tuple, Set
from native_compiler import RustCompiler, CompilationError

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
]

GRAPH_CACHE = {}


def get_graph(subpass: int) -> Tuple[int, List[Tuple[int, int]]]:
  if subpass not in GRAPH_CACHE:
    case = TEST_CASES[subpass]
    edges = generate_graph(case["vertices"], case["edges"], RANDOM_SEED + subpass)
    GRAPH_CACHE[subpass] = (case["vertices"], edges)
  return GRAPH_CACHE[subpass]


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
  num_vertices, edges = get_graph(subPass)

  compiler = RustCompiler(aiEngineName)
  if not compiler.find_compiler():
    return 0.0, "No Rust compiler found"

  try:
    exe_path = compiler.compile(result["rust_code"])
  except CompilationError as e:
    return 0.0, f"Compilation error: {str(e)[:200]}"

  input_data = format_input(num_vertices, edges)

  try:
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

    valid, msg = verify_cover(num_vertices, edges, cover)
    if not valid:
      return 0.2, f"[{case['desc']}] {msg}"

    greedy_size = greedy_cover_size(edges)
    ratio = len(cover) / greedy_size if greedy_size > 0 else 1.0
    score = min(1.0, 1.5 - ratio)  # Better than greedy = higher score

    return max(
      0.5,
      score), f"[{case['desc']}] Cover size {len(cover)} (greedy: {greedy_size}), {exec_time:.2f}s"

  except subprocess.TimeoutExpired:
    return 0.1, f"[{case['desc']}] Timeout"
  except Exception as e:
    return 0.0, f"[{case['desc']}] Error: {str(e)[:100]}"


def output_example_html(score: float, explanation: str, result: dict, subPass: int) -> str:
  case = TEST_CASES[subPass]
  code = result.get("rust_code", "").replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
  color = "green" if score >= 0.8 else "orange" if score >= 0.4 else "red"
  return f'<div class="result"><h4>Subpass {subPass}: {case["desc"]}</h4><p style="color:{color}">Score: {score:.2f}</p><p>{explanation}</p><details><summary>Code</summary><pre>{code}</pre></details></div>'


def output_header_html() -> str:
  return "<h2>Test 28: Minimum Vertex Cover (Rust)</h2><p>NP-Hard optimization problem.</p>"


def output_summary_html(results: list) -> str:
  total = sum(r[0] for r in results)
  return f'<div class="summary"><p>Total: {total:.2f}/{len(results)}</p></div>'
