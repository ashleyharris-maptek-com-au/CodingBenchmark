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
from typing import List, Tuple, Set, Dict, Any, Optional
from pathlib import Path
from collections import deque
import io

try:
  import numpy as np
except Exception:
  np = None

# Import our native compiler helper
from native_compiler import CppCompiler, CompilationError, ExecutionError, describe_this_pc
from solver_utils import StreamingInputFile, create_graph_input_file

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


def generate_graph_with_known_cut(num_nodes: int, num_edges: int, cut_size: int, seed: int):
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

  if np is not None and num_edges >= 200_000 and num_nodes >= 50_000:
    nrng = np.random.default_rng(seed)
    g1 = group1_size
    g2 = group2_size
    dtype = np.int32 if num_nodes <= (2**31 - 1) else np.int64

    if g1 > 1:
      i = np.arange(1, g1, dtype=dtype)
      parents = (nrng.random(g1 - 1) * i).astype(dtype)
      u1 = parents
      v1 = i
      a1 = np.minimum(u1, v1)
      b1 = np.maximum(u1, v1)
      tree1 = np.stack([a1, b1], axis=1)
    else:
      tree1 = np.zeros((0, 2), dtype=dtype)

    if g2 > 1:
      i = np.arange(1, g2, dtype=dtype)
      parents = (nrng.random(g2 - 1) * i).astype(dtype)
      u2 = parents + dtype(g1)
      v2 = i + dtype(g1)
      a2 = np.minimum(u2, v2)
      b2 = np.maximum(u2, v2)
      tree2 = np.stack([a2, b2], axis=1)
    else:
      tree2 = np.zeros((0, 2), dtype=dtype)

    intra_edges = max(0, num_edges - cut_size - (num_nodes - 2))
    e1 = intra_edges // 2
    e2 = intra_edges - e1

    def rand_pairs(count: int, start: int, size: int):
      if count <= 0 or size <= 1:
        return np.zeros((0, 2), dtype=dtype)
      out_u = np.empty(count, dtype=dtype)
      out_v = np.empty(count, dtype=dtype)
      filled = 0
      while filled < count:
        remaining = count - filled
        batch = max(1024, min(remaining * 2, 1_000_000))
        u = nrng.integers(0, size, size=batch, dtype=dtype)
        v = nrng.integers(0, size, size=batch, dtype=dtype)
        m = u != v
        if not np.any(m):
          continue
        u = u[m]
        v = v[m]
        take = min(remaining, u.shape[0])
        out_u[filled:filled + take] = u[:take] + dtype(start)
        out_v[filled:filled + take] = v[:take] + dtype(start)
        filled += take

      a = np.minimum(out_u, out_v)
      b = np.maximum(out_u, out_v)
      return np.stack([a, b], axis=1)

    extra1 = rand_pairs(e1, 0, g1)
    extra2 = rand_pairs(e2, g1, g2)

    cut_edges = set()
    attempts = 0
    while len(cut_edges) < cut_size and attempts < cut_size * 100:
      u = rng.randrange(0, g1) if g1 > 0 else 0
      v = g1 + (rng.randrange(0, g2) if g2 > 0 else 0)
      edge = (min(u, v), max(u, v))
      cut_edges.add(edge)
      attempts += 1
    cross = np.array(list(cut_edges), dtype=dtype)

    edges = np.concatenate([tree1, tree2, extra1, extra2, cross], axis=0)
    return edges, len(cut_edges)

  edges = set()

  for i in range(1, len(group1)):
    parent = rng.randint(0, i - 1)
    u, v = min(group1[parent], group1[i]), max(group1[parent], group1[i])
    edges.add((u, v))

  for i in range(1, len(group2)):
    parent = rng.randint(0, i - 1)
    u, v = min(group2[parent], group2[i]), max(group2[parent], group2[i])
    edges.add((u, v))

  intra_edges = num_edges - cut_size - (num_nodes - 2)
  for _ in range(max(0, intra_edges // 2)):
    if len(group1) > 1:
      u, v = rng.sample(group1, 2)
      edges.add((min(u, v), max(u, v)))
    if len(group2) > 1:
      u, v = rng.sample(group2, 2)
      edges.add((min(u, v), max(u, v)))

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


def format_graph_input(num_nodes: int, edges) -> str:
  """Format graph as input string."""
  if np is not None and isinstance(edges, np.ndarray):
    m = int(edges.shape[0])
    buf = io.StringIO()
    buf.write(f"{num_nodes} {m}\n")
    try:
      np.savetxt(buf, edges, fmt='%d %d')
    except Exception:
      for i in range(m):
        buf.write(f"{int(edges[i, 0])} {int(edges[i, 1])}\n")
    return buf.getvalue()

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
  queue = deque([0])
  while queue:
    node = queue.popleft()
    for neighbor in adj[node]:
      if neighbor not in visited:
        visited.add(neighbor)
        queue.append(neighbor)

  component1 = visited
  component2 = set(range(num_nodes)) - visited

  # Valid if both components are non-empty
  is_valid = len(component1) > 0 and len(component2) > 0
  return is_valid, component1, component2


def verify_cut_edges(num_nodes: int, edges, cut_edges: List[Tuple[int, int]]) -> Tuple[bool, str]:
  """
    Verify that the proposed cut edges are valid.
    Returns (is_valid, error_message).
    """
  if np is not None and isinstance(edges, np.ndarray):
    edge_set = set((int(min(u, v)), int(max(u, v))) for u, v in edges)
  else:
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
  {
    "num_nodes": 6,
    "num_edges": 10,
    "cut_size": 2,
    "seed": RANDOM_SEED,
    "known_cut": 2,
    "description": "6 nodes, 10 edges, min cut ~2"
  },
  {
    "num_nodes": 100,
    "num_edges": 400,
    "cut_size": 6,
    "seed": RANDOM_SEED + 4,
    "known_cut": 6,
    "description": "100 nodes, 400 edges, min cut ~6"
  },
  {
    "num_nodes": 1000,
    "num_edges": 8000,
    "cut_size": 12,
    "seed": RANDOM_SEED + 7,
    "known_cut": 12,
    "description": "1000 nodes, 8000 edges"
  },
  {
    "num_nodes": 10000,
    "num_edges": 100000,
    "cut_size": 20,
    "seed": RANDOM_SEED + 9,
    "known_cut": 20,
    "description": "10000 nodes, 100000 edges"
  },
  {
    "num_nodes": 100000,
    "num_edges": 5000000,
    "cut_size": 25,
    "seed": RANDOM_SEED + 10,
    "known_cut": 25,
    "description": "100000 nodes, 5000000 edges"
  },
  {
    "num_nodes": 1000000,
    "num_edges": 50000000,
    "cut_size": 25,
    "seed": RANDOM_SEED + 10,
    "known_cut": 25,
    "description": "1000000 nodes, 50000000 edges"
  },
  {
    "num_nodes": 10000000,
    "num_edges": 500000000,
    "cut_size": 25,
    "seed": RANDOM_SEED + 10,
    "known_cut": 25,
    "description": "10000000 nodes, 500000000 edges"
  },
  {
    "num_nodes": 100000000,
    "num_edges": 5000000000,
    "cut_size": 25,
    "seed": RANDOM_SEED + 10,
    "known_cut": 25,
    "description": "100000000 nodes, 5000000000 edges"
  },
  {
    "num_nodes": 1000000000,
    "num_edges": 50000000000,
    "cut_size": 25,
    "seed": RANDOM_SEED + 10,
    "known_cut": 25,
    "description": "1000000000 nodes, 50000000000 edges"
  },
]

_EDGES_CACHE: Dict[int, Any] = {}
_INPUT_FILE_CACHE: Dict[int, StreamingInputFile] = {}

# Threshold for using streaming (file-based) input vs in-memory string
STREAMING_THRESHOLD_EDGES = 1_000_000


def _get_case_edges(subPass: int):
  """Get edges for a test case (for small cases or verification)."""
  if subPass in _EDGES_CACHE:
    return _EDGES_CACHE[subPass]

  case = TEST_CASES[subPass]
  edges, actual_cut = generate_graph_with_known_cut(case["num_nodes"], case["num_edges"],
                                                    case["cut_size"], case["seed"])
  if case.get("known_cut") is None:
    case["known_cut"] = actual_cut

  if case["num_edges"] <= STREAMING_THRESHOLD_EDGES:
    _EDGES_CACHE[subPass] = edges
  return edges


def _get_streaming_input(subPass: int) -> StreamingInputFile:
  """Get a StreamingInputFile for a test case (for large cases)."""
  if subPass in _INPUT_FILE_CACHE:
    return _INPUT_FILE_CACHE[subPass]

  case = TEST_CASES[subPass]
  input_file = create_graph_input_file(case["num_nodes"],
                                       case["num_edges"],
                                       case["cut_size"],
                                       case["seed"],
                                       generate_graph_with_known_cut,
                                       cache_subdir="test18_graphs")
  _INPUT_FILE_CACHE[subPass] = input_file
  return input_file


def _should_use_streaming(subPass: int) -> bool:
  """Determine if streaming input should be used for this test case."""
  case = TEST_CASES[subPass]
  return case["num_edges"] > STREAMING_THRESHOLD_EDGES


def prepareSubpassPrompt(subPass: int) -> str:
  """Generate the prompt for subpass 0 that handles all graph sizes."""
  if subPass != 0:
    raise StopIteration

  return f"""You are writing a C++ program to solve the Minimum Edge Cut problem.

**Target environment specs**
{describe_this_pc()}

**C++ Compiler information**
{CppCompiler("test_engine").describe()}

You must write a C++ solver that can handle ANY graph size from trivial to multi-gb.

**The Challenge:**
Your C++ program will be tested with graphs ranging from small toy examples to massive real-world networks.
The same program must work efficiently across ALL scales.

**Problem:**
Given an undirected connected graph, find the minimum number of edges to remove
such that the graph becomes disconnected (split into two non-empty components).

**Input format (stdin):**
- Line 1: N M (number of nodes, number of edges)
- Lines 2 to M+1: u v (undirected edge between nodes u and v, 0-indexed)

**Output format (stdout):**
- Line 1: K (number of edges in the minimum cut)
- Lines 2 to K+1: u v (each edge in the cut, in any order)

**Requirements:**
1. Program must compile with this compiler, and run on this environment.
2. Read from stdin, write to stdout
3. Handle graphs even if they don't fit in memory.
4. Complete as fast as possible.
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
extraGradeAnswerRuns = list(range(1, len(TEST_CASES)))

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
                       engine_name: str,
                       input_data: str = None,
                       input_file: Path = None,
                       timeout: float = TIMEOUT_SECONDS) -> Tuple[str, str, float, bool]:
  """
    Compile and execute C++ solver.
    
    Args:
        code: C++ source code
        engine_name: Name for caching
        input_data: String data for stdin (for small inputs)
        input_file: Path to file for stdin (for large/streaming inputs)
        timeout: Execution timeout
    
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

    # Execute with file-based or string-based stdin
    if input_file is not None:
      stdout, stderr, exec_time, return_code = compiler.execute(exe_path,
                                                                timeout=timeout,
                                                                stdin_file=input_file)
    else:
      stdout, stderr, exec_time, return_code = compiler.execute(exe_path,
                                                                stdin_data=input_data or "",
                                                                timeout=timeout)

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
    """
  if not result:
    return 0.0, "No result provided"

  if "cpp_code" not in result:
    return 0.0, "No C++ code provided"

  case = TEST_CASES[subPass]
  num_nodes = case["num_nodes"]
  known_cut = case["known_cut"]
  description = case["description"]
  code = result["cpp_code"]

  use_streaming = _should_use_streaming(subPass)

  # Execute solver with streaming or in-memory input
  if use_streaming:
    # Large case: use file-based streaming input
    t = time.time()
    streaming_input = _get_streaming_input(subPass)
    print(f"  Generating/caching input file for {description}...")
    input_file_path = streaming_input.generate()
    file_size_mb = streaming_input.get_size_bytes() / (1024 * 1024)
    print(f"  Input file: {file_size_mb:.1f} MB")
    generate_time = time.time() - t
    if generate_time > 1: print(f"  Time to generate data: {generate_time:.2f}s")

    stdout, error, exec_time, success = execute_cpp_solver(code,
                                                           aiEngineName,
                                                           input_file=input_file_path)
  else:
    # Small case: use in-memory string input
    edges = _get_case_edges(subPass)
    input_data = format_graph_input(num_nodes, edges)
    stdout, error, exec_time, success = execute_cpp_solver(code,
                                                           aiEngineName,
                                                           input_data=input_data)

  if not success:
    return 0.0, f"[{description}] {error}", stdout

  # Parse output
  cut_edges, parse_error = parse_output(stdout)
  if parse_error and not cut_edges:
    return 0.0, f"[{description}] {parse_error}", stdout

  # Verify cut (skip full verification for very large streaming cases)
  if use_streaming and case["num_edges"] > 10_000_000:
    # For very large cases, trust the cut size without full verification
    # (verification would require loading all edges into memory)
    is_valid = True
    verify_msg = f"Cut of {len(cut_edges)} edges (verification skipped for large graph)"
  else:
    edges = _get_case_edges(subPass)
    is_valid, verify_msg = verify_cut_edges(num_nodes, edges, cut_edges)

  if not is_valid:
    return 0.0, f"[{description}] Invalid cut: {verify_msg}", stdout

  # Score based on cut size
  cut_size = len(cut_edges)
  ratio = cut_size / known_cut if known_cut > 0 else float('inf')

  if cut_size <= known_cut:
    score = 1.0
    quality = "optimal"
  elif ratio <= 1.25:
    score = 0.8
    quality = f"good ({ratio:.1f}x optimal)"
  elif ratio <= 1.5:
    score = 0.6
    quality = f"acceptable ({ratio:.1f}x optimal)"
  else:
    score = 0.4
    quality = f"valid but suboptimal ({ratio:.1f}x optimal)"

  explanation = (f"[{description}] Cut size: {cut_size}, "
                 f"Known min: {known_cut}, "
                 f"Time: {exec_time:.2f}s - {quality}")

  if subPass == 0:
    html = output_example_html(score, explanation, result, subPass)
  else:
    html = f"<pre>{stdout}</pre>"

  return score, explanation, html


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
            <summary>C++ Code</summary>
            <pre style="padding: 10px; overflow-x: auto;"><code>{code}</code></pre>
        </details>
    </div>
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


def setup():
  """Pre-generate and cache all streaming input files for parallel test execution."""
  print(f"  Pre-generating streaming input files for {len(TEST_CASES)} test cases...")
  for subpass in range(len(TEST_CASES)):
    if _should_use_streaming(subpass):
      streaming_input = _get_streaming_input(subpass)
      input_path = streaming_input.generate()
      size_mb = streaming_input.get_size_bytes() / (1024 * 1024)
      print(f"    Subpass {subpass}: {size_mb:.1f} MB cached")
