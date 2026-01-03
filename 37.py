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
from typing import List, Tuple, Set
from native_compiler import CSharpCompiler, CompilationError

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
  num_vertices, edges = get_graph(subPass)

  compiler = CSharpCompiler(aiEngineName)
  if not compiler.find_compiler():
    return 0.0, "No C# compiler found"

  try:
    exe_path = compiler.compile(result["csharp_code"])
  except CompilationError as e:
    return 0.0, f"Compilation error: {str(e)[:200]}"

  input_data = format_input(num_vertices, edges)

  try:
    start = time.time()
    stdout, stderr, exec_time, retcode = compiler.execute(exe_path, input_data, TIMEOUT_SECONDS)

    if retcode != 0:
      return 0.0, f"Runtime error: {stderr[:200]}"

    lines = stdout.strip().split('\n')
    set_size = int(lines[0])
    fvs = set(map(int, lines[1].split())) if len(lines) > 1 and lines[1].strip() else set()

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

  except Exception as e:
    return 0.0, f"[{case['desc']}] Error: {str(e)[:100]}"


def output_example_html(score: float, explanation: str, result: dict, subPass: int) -> str:
  case = TEST_CASES[subPass]
  code = result.get("csharp_code", "").replace("&", "&amp;").replace("<",
                                                                     "&lt;").replace(">", "&gt;")
  color = "green" if score >= 0.8 else "orange" if score >= 0.4 else "red"
  return f'<div class="result"><h4>Subpass {subPass}: {case["desc"]}</h4><p style="color:{color}">Score: {score:.2f}</p><p>{explanation}</p><details><summary>Code</summary><pre>{code}</pre></details></div>'


def output_header_html() -> str:
  return "<h2>Test 37: Feedback Vertex Set (C#)</h2><p>NP-Hard cycle elimination problem.</p>"


def output_summary_html(results: list) -> str:
  total = sum(r[0] for r in results)
  return f'<div class="summary"><p>Total: {total:.2f}/{len(results)}</p></div>'
