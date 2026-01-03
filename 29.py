"""
Test 29: Maximum Clique (C# Implementation)

The LLM must write C# code to find the largest complete subgraph (clique)
in an undirected graph. This is NP-Hard.

Subpasses increase graph density and size, requiring algorithms like
Bron-Kerbosch with pivoting, branch-and-bound, or MaxSAT formulation.

Solver times out after 5 minutes.
"""

import random
import subprocess
import time
from typing import List, Tuple, Set
from native_compiler import CSharpCompiler, CompilationError

title = "Maximum Clique (C#)"
TIMEOUT_SECONDS = 30
RANDOM_SEED = 29292929


def generate_graph(num_vertices: int, edge_prob: float, seed: int) -> List[Tuple[int, int]]:
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
    "edge_prob": 0.6,
    "desc": "20 vertices, dense"
  },
  {
    "vertices": 40,
    "edge_prob": 0.5,
    "desc": "40 vertices, medium density"
  },
  {
    "vertices": 60,
    "edge_prob": 0.45,
    "desc": "60 vertices"
  },
  {
    "vertices": 80,
    "edge_prob": 0.4,
    "desc": "80 vertices"
  },
  {
    "vertices": 100,
    "edge_prob": 0.35,
    "desc": "100 vertices"
  },
  {
    "vertices": 150,
    "edge_prob": 0.3,
    "desc": "150 vertices"
  },
  {
    "vertices": 200,
    "edge_prob": 0.25,
    "desc": "200 vertices"
  },
  {
    "vertices": 300,
    "edge_prob": 0.2,
    "desc": "300 vertices"
  },
  {
    "vertices": 500,
    "edge_prob": 0.15,
    "desc": "500 vertices"
  },
  {
    "vertices": 750,
    "edge_prob": 0.12,
    "desc": "750 vertices"
  },
  {
    "vertices": 1000,
    "edge_prob": 0.1,
    "desc": "1000 vertices, sparse"
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
  """Generate the prompt for subpass 0 that handles all clique complexities."""
  if subPass != 0:
    raise StopIteration

  return f"""You are writing C# code to solve the Maximum Clique problem.

You must write a C# solver that can handle ANY graph complexity from trivial to ludicrous scale:
- **Trivial**: Small sparse graphs (10-20 vertices), basic backtracking
- **Medium**: Moderate graphs (50-100 vertices), Bron-Kerbosch with pivoting
- **Large**: Dense graphs (200-500 vertices), advanced branch-and-bound
- **Extreme**: Very dense graphs (1000+ vertices), heuristics and approximation

**The Challenge:**
Your C# maximum clique finder will be tested with graphs ranging from sparse to very dense instances. The same algorithm must work efficiently across ALL graph complexities.

**Problem:**
Find the largest complete subgraph (clique) in an undirected graph. A clique is a set of vertices where every pair of vertices is connected by an edge. This is NP-Hard.

**Input format (stdin):**
```
num_vertices num_edges
u v  (for each edge, 0-indexed)
```

**Output format (stdout):**
```
clique_size
vertex_1 vertex_2 ... vertex_k  (vertices in the maximum clique, space-separated)
```

**Critical Requirements:**
1. **Scalability**: Your algorithm must adapt based on graph size and density
2. **Performance**: Must complete within 5 minutes even for very dense graphs
3. **Quality**: Find maximum or near-maximum cliques

**Algorithm Strategy Recommendations:**
- **Small sparse graphs (≤50 vertices)**: Can use basic backtracking algorithms
- **Medium graphs (50-200 vertices)**: Bron-Kerbosch algorithm with pivoting
- **Large dense graphs (200-1000 vertices)**: Advanced branch-and-bound with pruning
- **Very dense graphs (>1000 vertices)**: Fast heuristics, approximation algorithms

**Key Techniques:**
- **Bron-Kerbosch**: Recursive backtracking with pivot selection
- **Pivoting**: Choose pivot vertex to reduce recursive calls
- **Branch-and-bound**: Prune search space using upper bounds
- **Degeneracy ordering**: Process vertices in order of increasing degree
- **Heuristics**: Greedy algorithms for initial bounds

**Implementation Hints:**
- Detect graph complexity and choose appropriate algorithm
- Use efficient adjacency matrix for dense graphs, adjacency lists for sparse
- Implement adaptive quality vs speed tradeoffs
- For very dense graphs, focus on fast heuristics
- Handle edge cases: empty graphs, complete graphs, disconnected graphs
- Use fast I/O for large inputs

**Success Criteria:**
- All vertices in output form a complete subgraph
- No larger clique exists in the graph
- Output format is correct

**Failure Criteria:**
- Output vertices don't form a clique
- Larger clique exists
- Timeout without solution
- Invalid output format

**Requirements:**
1. Program must compile with .NET/csc
2. Read from stdin, write to stdout
3. Handle variable graph sizes and densities
4. Complete within 5 minutes
5. Must handle varying graph complexities efficiently

Write complete, compilable C# code with a Main method.
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
      "Explain your algorithm approach and how it adapts to different clique complexities"
    },
    "csharp_code": {
      "type": "string",
      "description": "Complete C# code with Main method that handles all scales"
    }
  },
  "required": ["reasoning", "csharp_code"],
  "additionalProperties": False
}


def verify_clique(num_vertices: int, edges: List[Tuple[int, int]],
                  clique: List[int]) -> Tuple[bool, str]:
  edge_set = set((min(u, v), max(u, v)) for u, v in edges)

  for v in clique:
    if v < 0 or v >= num_vertices:
      return False, f"Invalid vertex {v}"

  for i, u in enumerate(clique):
    for v in clique[i + 1:]:
      if (min(u, v), max(u, v)) not in edge_set:
        return False, f"Vertices {u} and {v} not connected"

  return True, "Valid clique"


def greedy_clique_size(num_vertices: int, edges: List[Tuple[int, int]]) -> int:
  adj = [set() for _ in range(num_vertices)]
  for u, v in edges:
    adj[u].add(v)
    adj[v].add(u)

  clique = []
  candidates = list(range(num_vertices))
  candidates.sort(key=lambda v: -len(adj[v]))

  for v in candidates:
    if all(v in adj[u] for u in clique):
      clique.append(v)

  return len(clique)


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
    clique_size = int(lines[0])
    clique = list(map(int, lines[1].split())) if len(lines) > 1 else []

    valid, msg = verify_clique(num_vertices, edges, clique)
    if not valid:
      return 0.2, f"[{case['desc']}] {msg}"

    greedy_size = greedy_clique_size(num_vertices, edges)
    ratio = len(clique) / greedy_size if greedy_size > 0 else 1.0
    score = min(1.0, 0.5 + ratio * 0.5)

    return score, f"[{case['desc']}] Clique size {len(clique)} (greedy: {greedy_size}), {exec_time:.2f}s"

  except Exception as e:
    return 0.0, f"[{case['desc']}] Error: {str(e)[:100]}"


def output_example_html(score: float, explanation: str, result: dict, subPass: int) -> str:
  case = TEST_CASES[subPass]
  code = result.get("csharp_code", "").replace("&", "&amp;").replace("<",
                                                                     "&lt;").replace(">", "&gt;")
  color = "green" if score >= 0.8 else "orange" if score >= 0.4 else "red"
  return f'<div class="result"><h4>Subpass {subPass}: {case["desc"]}</h4><p style="color:{color}">Score: {score:.2f}</p><p>{explanation}</p><details><summary>Code</summary><pre>{code}</pre></details></div>'


def output_header_html() -> str:
  return "<h2>Test 29: Maximum Clique (C#)</h2><p>NP-Hard graph problem.</p>"


def output_summary_html(results: list) -> str:
  total = sum(r[0] for r in results)
  return f'<div class="summary"><p>Total: {total:.2f}/{len(results)}</p></div>'
