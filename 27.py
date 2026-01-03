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
from typing import List, Tuple, Set, Dict, Optional
from pathlib import Path

from native_compiler import CppCompiler, CompilationError

title = "Graph Coloring (C++)"
TIMEOUT_SECONDS = 30
RANDOM_SEED = 27272727


def generate_graph(num_vertices: int, edge_probability: float, seed: int) -> List[Tuple[int, int]]:
  """Generate random graph with given edge probability."""
  rng = random.Random(seed)
  edges = []
  for i in range(num_vertices):
    for j in range(i + 1, num_vertices):
      if rng.random() < edge_probability:
        edges.append((i, j))
  return edges


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
]

GRAPH_CACHE = {}


def get_graph(subpass: int) -> Tuple[int, List[Tuple[int, int]], int]:
  if subpass not in GRAPH_CACHE:
    case = TEST_CASES[subpass]
    edges = generate_graph(case["vertices"], case["edge_prob"], RANDOM_SEED + subpass)
    GRAPH_CACHE[subpass] = (case["vertices"], edges, case["colors"])
  return GRAPH_CACHE[subpass]


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

You must write a C++ solver that can handle ANY graph complexity from trivial to ludicrous scale:
- **Trivial**: Small graphs (10-20 vertices, many colors), basic backtracking
- **Medium**: Moderate graphs (50-100 vertices, moderate colors), DSatur algorithm
- **Large**: Complex graphs (200-500 vertices, few colors), advanced heuristics
- **Extreme**: Massive graphs (1000-5000 vertices, very few colors), sophisticated optimization

**The Challenge:**
Your C++ graph colorer will be tested with graphs ranging from simple coloring tasks to massive graphs with very few available colors. The same algorithm must work efficiently across ALL graph complexities.

**Problem:**
Color a graph with exactly k colors such that no two adjacent vertices share the same color. This is NP-Complete and requires sophisticated algorithms for larger instances.

**Input format (stdin):**
```
num_vertices num_edges num_colors
u v  (for each edge, 0-indexed)
```

**Output format (stdout):**
```
color_0 color_1 ... color_(n-1)  (color assignment for each vertex)
```

**Critical Requirements:**
1. **Scalability**: Your algorithm must adapt based on graph size and color availability
2. **Performance**: Must complete within 5 minutes even for massive graphs
3. **Quality**: Find valid colorings or correctly report impossibility

**Algorithm Strategy Recommendations:**
- **Small graphs (≤50 vertices, ≥5 colors)**: Can use backtracking with pruning
- **Medium graphs (50-200 vertices, 3-4 colors)**: DSatur algorithm, constraint propagation
- **Large graphs (200-1000 vertices, 2-3 colors)**: Advanced heuristics, local search
- **Very Large graphs (>1000 vertices, 2 colors)**: Very fast heuristics, approximation methods

**Key Techniques:**
- **DSatur**: Degree of Saturation algorithm for greedy coloring
- **Backtracking**: Systematic search with pruning
- **Constraint propagation**: Forward checking to reduce search space
- **Local search**: Hill climbing, tabu search for optimization
- **Heuristics**: Largest degree ordering, Welsh-Powell algorithm

**Implementation Hints:**
- Detect graph complexity and choose appropriate coloring algorithm
- Use efficient adjacency list representation
- Implement adaptive quality vs speed tradeoffs
- For very large graphs, focus on fast heuristics
- Handle edge cases: bipartite graphs, complete graphs, impossible coloring
- Use fast I/O for large inputs

**Success Criteria:**
- All vertices assigned valid colors (0 to k-1)
- No adjacent vertices share the same color
- All vertices are colored (complete coloring)

**Failure Criteria:**
- Cannot find valid coloring within time limit
- Invalid color assignment (adjacent vertices same color)
- Incomplete coloring

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


def gradeAnswer(result: dict, subPass: int, aiEngineName: str) -> tuple:
  if not result or "cpp_code" not in result:
    return 0.0, "No C++ code provided"

  case = TEST_CASES[subPass]
  num_vertices, edges, num_colors = get_graph(subPass)

  compiler = CppCompiler(aiEngineName)
  if not compiler.find_compiler():
    return 0.0, "No C++ compiler found"

  try:
    exe_path = compiler.compile(result["cpp_code"])
  except CompilationError as e:
    return 0.0, f"Compilation error: {str(e)[:200]}"

  input_data = format_input(num_vertices, edges, num_colors)

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

    colors = list(map(int, proc.stdout.strip().split()))
    valid, msg = verify_coloring(num_vertices, edges, colors, num_colors)

    if valid:
      return 1.0, f"[{case['desc']}] Valid coloring in {exec_time:.2f}s"
    else:
      return 0.2, f"[{case['desc']}] {msg}"

  except subprocess.TimeoutExpired:
    return 0.1, f"[{case['desc']}] Timeout"
  except Exception as e:
    return 0.0, f"[{case['desc']}] Error: {str(e)[:100]}"


def output_example_html(score: float, explanation: str, result: dict, subPass: int) -> str:
  case = TEST_CASES[subPass]
  code = result.get("cpp_code", "").replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
  color = "green" if score >= 0.8 else "orange" if score >= 0.4 else "red"
  return f'<div class="result"><h4>Subpass {subPass}: {case["desc"]}</h4><p style="color:{color}">Score: {score:.2f}</p><p>{explanation}</p><details><summary>Code</summary><pre>{code}</pre></details></div>'


def output_header_html() -> str:
  return "<h2>Test 27: Graph Coloring (C++)</h2><p>NP-Complete graph coloring problem.</p>"


def output_summary_html(results: list) -> str:
  total = sum(r[0] for r in results)
  return f'<div class="summary"><p>Total: {total:.2f}/{len(results)}</p></div>'
