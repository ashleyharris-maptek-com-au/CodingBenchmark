"""
Test 18: Minimum Cut Problem (C++ Implementation)

The LLM must write C++ code that finds the minimum edge cut to split a graph into two parts.
Given a graph, remove the minimum number of edges to disconnect it into two non-empty components.

This tests:
1. C++ proficiency
2. Graph algorithms (Karger's algorithm, Stoer-Wagner, Ford-Fulkerson based)
3. Efficient implementation for large graphs

Input format (stdin):
Line 1: N M (number of nodes, number of edges)
Lines 2 to M+1: u v (edge from node u to node v, 0-indexed)

Output format (stdout):
Line 1: K (number of edges in minimum cut)
Lines 2 to K+1: u v (edges to cut)

Subpasses test increasingly complex graphs.
Solver times out after 5 minutes.
"""

import random
import subprocess
import sys
import os
import time
from typing import List, Tuple, Set, Dict
from pathlib import Path

# Import our native compiler helper
from native_compiler import CppCompiler, CompilationError, ExecutionError

title = "Minimum Cut Problem (C++)"

# Timeout in seconds (5 minutes)
TIMEOUT_SECONDS = 30

# Seed for reproducibility
RANDOM_SEED = 18181818


def generate_connected_graph(num_nodes: int, num_edges: int, seed: int) -> List[Tuple[int, int]]:
  """Generate a connected undirected graph."""
  rng = random.Random(seed)
  edges = set()

  # First, create a spanning tree to ensure connectivity
  nodes = list(range(num_nodes))
  rng.shuffle(nodes)

  for i in range(1, num_nodes):
    parent = rng.randint(0, i - 1)
    u, v = min(nodes[parent], nodes[i]), max(nodes[parent], nodes[i])
    edges.add((u, v))

  # Add remaining edges randomly
  attempts = 0
  while len(edges) < num_edges and attempts < num_edges * 10:
    u = rng.randint(0, num_nodes - 1)
    v = rng.randint(0, num_nodes - 1)
    if u != v:
      edge = (min(u, v), max(u, v))
      edges.add(edge)
    attempts += 1

  return list(edges)


def generate_graph_with_known_cut(num_nodes: int, num_edges: int, cut_size: int,
                                  seed: int) -> Tuple[List[Tuple[int, int]], int]:
  """
    Generate a graph with a known minimum cut.
    Creates two clusters connected by exactly cut_size edges.
    """
  rng = random.Random(seed)

  # Split nodes into two groups
  group1_size = num_nodes // 2
  group2_size = num_nodes - group1_size

  group1 = list(range(group1_size))
  group2 = list(range(group1_size, num_nodes))

  edges = set()

  # Create dense connections within each group (spanning trees + extra)
  # Group 1
  for i in range(1, len(group1)):
    parent = rng.randint(0, i - 1)
    u, v = min(group1[parent], group1[i]), max(group1[parent], group1[i])
    edges.add((u, v))

  # Group 2
  for i in range(1, len(group2)):
    parent = rng.randint(0, i - 1)
    u, v = min(group2[parent], group2[i]), max(group2[parent], group2[i])
    edges.add((u, v))

  # Add more intra-group edges
  intra_edges = num_edges - cut_size - (num_nodes - 2)  # subtract spanning tree edges
  for _ in range(max(0, intra_edges // 2)):
    # Add to group 1
    if len(group1) > 1:
      u, v = rng.sample(group1, 2)
      edges.add((min(u, v), max(u, v)))
    # Add to group 2
    if len(group2) > 1:
      u, v = rng.sample(group2, 2)
      edges.add((min(u, v), max(u, v)))

  # Add exactly cut_size edges between groups
  cut_edges = set()
  attempts = 0
  while len(cut_edges) < cut_size and attempts < cut_size * 100:
    u = rng.choice(group1)
    v = rng.choice(group2)
    edge = (min(u, v), max(u, v))
    if edge not in edges:
      cut_edges.add(edge)
      edges.add(edge)
    attempts += 1

  return list(edges), len(cut_edges)


def format_graph_input(num_nodes: int, edges: List[Tuple[int, int]]) -> str:
  """Format graph as input string."""
  lines = [f"{num_nodes} {len(edges)}"]
  for u, v in edges:
    lines.append(f"{u} {v}")
  return "\n".join(lines)


def compute_min_cut_bfs(num_nodes: int, edges: List[Tuple[int, int]],
                        cut_edges: Set[Tuple[int, int]]) -> Tuple[bool, Set[int], Set[int]]:
  """
    Verify if removing cut_edges disconnects the graph.
    Returns (is_valid_cut, component1, component2).
    """
  # Build adjacency list without cut edges
  adj = {i: [] for i in range(num_nodes)}
  for u, v in edges:
    edge = (min(u, v), max(u, v))
    if edge not in cut_edges:
      adj[u].append(v)
      adj[v].append(u)

  # BFS from node 0
  visited = {0}
  queue = [0]
  while queue:
    node = queue.pop(0)
    for neighbor in adj[node]:
      if neighbor not in visited:
        visited.add(neighbor)
        queue.append(neighbor)

  component1 = visited
  component2 = set(range(num_nodes)) - visited

  # Valid if both components are non-empty
  is_valid = len(component1) > 0 and len(component2) > 0
  return is_valid, component1, component2


def verify_cut_edges(num_nodes: int, edges: List[Tuple[int, int]],
                     cut_edges: List[Tuple[int, int]]) -> Tuple[bool, str]:
  """
    Verify that the proposed cut edges are valid.
    Returns (is_valid, error_message).
    """
  edge_set = set((min(u, v), max(u, v)) for u, v in edges)
  cut_set = set()

  for u, v in cut_edges:
    edge = (min(u, v), max(u, v))
    if edge not in edge_set:
      return False, f"Edge ({u}, {v}) is not in the graph"
    cut_set.add(edge)

  is_disconnected, comp1, comp2 = compute_min_cut_bfs(num_nodes, edges, cut_set)

  if not is_disconnected:
    return False, "Cut does not disconnect the graph"

  return True, f"Valid cut: components of size {len(comp1)} and {len(comp2)}"


# Test configurations
TEST_CASES = [
  # Subpass 0: Simple graph
  {
    "num_nodes": 6,
    "edges": generate_graph_with_known_cut(6, 10, 2, RANDOM_SEED)[0],
    "known_cut": 2,
    "description": "6 nodes, 10 edges, min cut ~2"
  },
  # Subpass 1: Small graph
  {
    "num_nodes": 10,
    "edges": generate_graph_with_known_cut(10, 20, 3, RANDOM_SEED + 1)[0],
    "known_cut": 3,
    "description": "10 nodes, 20 edges, min cut ~3"
  },
  # Subpass 2: Medium graph
  {
    "num_nodes": 20,
    "edges": generate_graph_with_known_cut(20, 50, 4, RANDOM_SEED + 2)[0],
    "known_cut": 4,
    "description": "20 nodes, 50 edges, min cut ~4"
  },
  # Subpass 3: Larger graph
  {
    "num_nodes": 50,
    "edges": generate_graph_with_known_cut(50, 150, 5, RANDOM_SEED + 3)[0],
    "known_cut": 5,
    "description": "50 nodes, 150 edges, min cut ~5"
  },
  # Subpass 4: Complex graph
  {
    "num_nodes": 100,
    "edges": generate_graph_with_known_cut(100, 400, 6, RANDOM_SEED + 4)[0],
    "known_cut": 6,
    "description": "100 nodes, 400 edges, min cut ~6"
  },
  # Subpass 5: Large graph
  {
    "num_nodes": 200,
    "edges": generate_graph_with_known_cut(200, 1000, 8, RANDOM_SEED + 5)[0],
    "known_cut": 8,
    "description": "200 nodes, 1000 edges, min cut ~8"
  },
  # Extreme cases
  {
    "num_nodes": 500,
    "edges": generate_graph_with_known_cut(500, 3000, 10, RANDOM_SEED + 6)[0],
    "known_cut": 10,
    "description": "500 nodes, 3000 edges"
  },
  {
    "num_nodes": 1000,
    "edges": generate_graph_with_known_cut(1000, 8000, 12, RANDOM_SEED + 7)[0],
    "known_cut": 12,
    "description": "1000 nodes, 8000 edges"
  },
  {
    "num_nodes": 5000,
    "edges": generate_graph_with_known_cut(5000, 50000, 15, RANDOM_SEED + 8)[0],
    "known_cut": 15,
    "description": "5000 nodes, 50000 edges"
  },
  {
    "num_nodes": 10000,
    "edges": generate_graph_with_known_cut(10000, 100000, 20, RANDOM_SEED + 9)[0],
    "known_cut": 20,
    "description": "10000 nodes, 100000 edges"
  },
  {
    "num_nodes": 50000,
    "edges": generate_graph_with_known_cut(50000, 500000, 25, RANDOM_SEED + 10)[0],
    "known_cut": 25,
    "description": "50000 nodes, 500000 edges"
  },
]


def prepareSubpassPrompt(subPass: int) -> str:
  """Generate the prompt for subpass 0 that handles all graph sizes."""
  if subPass != 0:
    raise StopIteration

  return f"""You are writing a C++ program to solve the Minimum Edge Cut problem.

You must write a C++ solver that can handle ANY graph size from trivial to ludicrous scale:
- **Trivial**: Small graphs (4-10 nodes, 5-20 edges), simple cases
- **Medium**: Medium graphs (20-50 nodes, 50-200 edges), moderate complexity
- **Large**: Large graphs (100-500 nodes, 200-2000 edges), complex structures
- **Extreme**: Massive graphs (1000-50000 nodes, 2000-500000 edges), very complex optimization

**The Challenge:**
Your C++ program will be tested with graphs ranging from small toy examples to massive real-world networks. The same program must work efficiently across ALL scales.

**Problem:**
Given an undirected connected graph, find the minimum number of edges to remove
such that the graph becomes disconnected (split into two non-empty components).

**Input format (stdin):**
- Line 1: N M (number of nodes, number of edges)
- Lines 2 to M+1: u v (undirected edge between nodes u and v, 0-indexed)

**Output format (stdout):**
- Line 1: K (number of edges in the minimum cut)
- Lines 2 to K+1: u v (each edge in the cut, in any order)

**Critical Requirements:**
1. **Scalability**: Your algorithm must adapt based on graph size and density
2. **Performance**: Must complete within 5 minutes even for massive graphs
3. **Correctness**: Must find a valid minimum cut that disconnects the graph

**Algorithm Strategy Recommendations:**
- **Small graphs (≤50 nodes)**: Can use exact methods (try all possible cuts)
- **Medium graphs (50-500 nodes)**: Karger's randomized algorithm O(n²m) expected
- **Large graphs (500-5000 nodes)**: Stoer-Wagner algorithm O(nm + n² log n) deterministic
- **Very Large graphs (>5000 nodes)**: Fast approximations, heuristic methods

**Key Algorithms:**
- **Karger's randomized algorithm**: Randomly contract edges until 2 nodes remain
- **Stoer-Wagner algorithm**: Deterministic minimum cut algorithm
- **Ford-Fulkerson based**: Run max-flow between all pairs, min of max-flows = min-cut
- **Push-relabel**: Efficient max-flow implementation

**Implementation Hints:**
- Detect graph size and choose appropriate algorithm
- Use efficient data structures (adjacency lists)
- Implement adaptive quality vs speed tradeoffs
- For very large graphs, consider approximation algorithms
- Handle edge cases: already disconnected graphs, single nodes
- Use fast I/O for large inputs

**Requirements:**
1. Program must compile with g++ or MSVC (C++17)
2. Read from stdin, write to stdout
3. Handle graphs up to 50,000 nodes and 500,000 edges
4. Complete within 5 minutes
5. Output must be a valid cut that disconnects the graph
6. Minimize the number of cut edges
7. Must handle varying graph sizes efficiently

**Example:**
Input:
```
4 5
0 1
1 2
2 3
3 0
0 2
```

Output (one possible answer):
```
2
0 1
3 0
```

Write complete, compilable C++ code with a main() function.
Include adaptive logic that chooses different algorithms based on graph scale.
"""


# List of subpasses to grade the single answer against all difficulty levels
extraGradeAnswerRuns = list(range(len(TEST_CASES)))

structure = {
  "type": "object",
  "properties": {
    "reasoning": {
      "type": "string",
      "description": "Explain your algorithm choice and how it adapts to different graph sizes"
    },
    "cpp_code": {
      "type": "string",
      "description": "Complete C++ code with main() function that handles all scales"
    }
  },
  "required": ["reasoning", "cpp_code"],
  "additionalProperties": False
}


def execute_cpp_solver(code: str,
                       input_data: str,
                       engine_name: str,
                       timeout: float = TIMEOUT_SECONDS) -> Tuple[str, str, float, bool]:
  """
    Compile and execute C++ solver.
    
    Returns:
        Tuple of (stdout, error_message, execution_time, success)
    """
  compiler = CppCompiler(engine_name)

  # Check if compiler is available
  if not compiler.find_compiler():
    return "", "No C++ compiler found", 0, False

  try:
    # Compile
    exe_path = compiler.compile(code)

    # Execute
    stdout, stderr, exec_time, return_code = compiler.execute(exe_path, input_data, timeout)

    if return_code != 0:
      return stdout, f"Runtime error (exit code {return_code}): {stderr[:500]}", exec_time, False

    return stdout, "", exec_time, True

  except CompilationError as e:
    return "", str(e), 0, False
  except ExecutionError as e:
    return "", str(e), TIMEOUT_SECONDS, False
  except Exception as e:
    return "", f"Unexpected error: {str(e)}", 0, False


def parse_output(output: str) -> Tuple[List[Tuple[int, int]], str]:
  """
    Parse solver output.
    
    Returns:
        Tuple of (cut_edges, error_message)
    """
  lines = output.strip().split('\n')
  if not lines:
    return [], "Empty output"

  try:
    k = int(lines[0].strip())
    if k < 0:
      return [], f"Invalid cut size: {k}"

    cut_edges = []
    for i in range(1, min(k + 1, len(lines))):
      parts = lines[i].strip().split()
      if len(parts) >= 2:
        u, v = int(parts[0]), int(parts[1])
        cut_edges.append((u, v))

    if len(cut_edges) != k:
      return cut_edges, f"Expected {k} edges, got {len(cut_edges)}"

    return cut_edges, ""

  except ValueError as e:
    return [], f"Parse error: {str(e)}"


def gradeAnswer(result: dict, subPass: int, aiEngineName: str) -> tuple:
  """
    Grade the C++ minimum cut solver.
    
    Scoring:
    - 1.0: Optimal cut (matches or beats known minimum)
    - 0.8: Valid cut within 1.5x optimal
    - 0.6: Valid cut within 2x optimal
    - 0.4: Valid cut but suboptimal
    - 0.0: Invalid cut or error
    """
  if not result:
    return 0.0, "No result provided"

  if "cpp_code" not in result:
    return 0.0, "No C++ code provided"

  case = TEST_CASES[subPass]
  num_nodes = case["num_nodes"]
  edges = case["edges"]
  known_cut = case["known_cut"]
  description = case["description"]

  code = result["cpp_code"]
  input_data = format_graph_input(num_nodes, edges)

  # Execute solver
  stdout, error, exec_time, success = execute_cpp_solver(code, input_data, aiEngineName)

  if not success:
    return 0.0, f"[{description}] {error}"

  # Parse output
  cut_edges, parse_error = parse_output(stdout)
  if parse_error and not cut_edges:
    return 0.0, f"[{description}] {parse_error}"

  # Verify cut
  is_valid, verify_msg = verify_cut_edges(num_nodes, edges, cut_edges)
  if not is_valid:
    return 0.0, f"[{description}] Invalid cut: {verify_msg}"

  # Score based on cut size
  cut_size = len(cut_edges)
  ratio = cut_size / known_cut if known_cut > 0 else float('inf')

  if cut_size <= known_cut:
    score = 1.0
    quality = "optimal"
  elif ratio <= 1.5:
    score = 0.8
    quality = f"good ({ratio:.1f}x optimal)"
  elif ratio <= 2.0:
    score = 0.6
    quality = f"acceptable ({ratio:.1f}x optimal)"
  else:
    score = 0.4
    quality = f"valid but suboptimal ({ratio:.1f}x optimal)"

  explanation = (f"[{description}] Cut size: {cut_size}, "
                 f"Known min: {known_cut}, "
                 f"Time: {exec_time:.2f}s - {quality}")

  return score, explanation


def output_example_html(score: float, explanation: str, result: dict, subPass: int) -> str:
  """Generate HTML for result display."""
  case = TEST_CASES[subPass]

  code = result.get("cpp_code", "No code provided")
  reasoning = result.get("reasoning", "No reasoning provided")

  # Escape HTML
  code = code.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
  reasoning = reasoning.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

  score_color = "green" if score >= 0.8 else "orange" if score >= 0.4 else "red"

  return f"""
    <div class="result" style="margin: 10px; padding: 10px; border: 1px solid #ccc;">
        <h4>Subpass {subPass}: {case['description']}</h4>
        <p><strong>Score:</strong> <span style="color: {score_color};">{score:.2f}</span></p>
        <p><strong>Details:</strong> {explanation}</p>
        <details>
            <summary>Reasoning</summary>
            <pre style="background: #f5f5f5; padding: 10px; overflow-x: auto;">{reasoning}</pre>
        </details>
        <details>
            <summary>C++ Code</summary>
            <pre style="background: #f0f0f0; padding: 10px; overflow-x: auto;"><code>{code}</code></pre>
        </details>
    </div>
    """


def output_header_html() -> str:
  """Generate HTML header."""
  return """
    <h2>Test 18: Minimum Cut Problem (C++)</h2>
    <p>Testing C++ implementation of minimum edge cut algorithm.</p>
    """


def output_summary_html(results: list) -> str:
  """Generate summary HTML."""
  if not results:
    return "<p>No results</p>"

  total_score = sum(r[0] for r in results)
  max_score = len(results)
  avg_score = total_score / max_score if max_score > 0 else 0

  return f"""
    <div class="summary" style="margin: 10px; padding: 15px; background: #e8f4e8; border-radius: 5px;">
        <h3>Summary</h3>
        <p><strong>Total Score:</strong> {total_score:.2f} / {max_score}</p>
        <p><strong>Average Score:</strong> {avg_score:.2%}</p>
        <p><strong>Subpasses Completed:</strong> {len(results)}</p>
    </div>
    """
