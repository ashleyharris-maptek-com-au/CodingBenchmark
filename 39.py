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
from typing import List, Tuple, Set
from native_compiler import RustCompiler, CompilationError

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
]

GRAPH_CACHE = {}


def get_graph(subpass: int) -> Tuple[int, List[Tuple[int, int]]]:
  if subpass not in GRAPH_CACHE:
    case = TEST_CASES[subpass]
    edges = generate_graph(case["vertices"], case["edge_prob"], RANDOM_SEED + subpass)
    GRAPH_CACHE[subpass] = (case["vertices"], edges)
  return GRAPH_CACHE[subpass]


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
    reported_cut = int(lines[0])
    partition_a = set(map(int, lines[1].split())) if len(lines) > 1 else set()

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
  except Exception as e:
    return 0.0, f"[{case['desc']}] Error: {str(e)[:100]}"


def output_example_html(score: float, explanation: str, result: dict, subPass: int) -> str:
  case = TEST_CASES[subPass]
  code = result.get("rust_code", "").replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
  color = "green" if score >= 0.8 else "orange" if score >= 0.4 else "red"
  return f'<div class="result"><h4>Subpass {subPass}: {case["desc"]}</h4><p style="color:{color}">Score: {score:.2f}</p><p>{explanation}</p><details><summary>Code</summary><pre>{code}</pre></details></div>'


def output_header_html() -> str:
  return "<h2>Test 39: Graph Bisection (Rust)</h2><p>NP-Hard graph partitioning problem.</p>"


def output_summary_html(results: list) -> str:
  total = sum(r[0] for r in results)
  return f'<div class="summary"><p>Total: {total:.2f}/{len(results)}</p></div>'
