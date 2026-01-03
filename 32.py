"""
Test 32: Steiner Tree (C++ Implementation)

The LLM must write C++ code to find the minimum-weight tree connecting a set
of terminal vertices in a weighted graph. This is NP-Hard.

Subpasses increase graph size and terminal count, requiring Dreyfus-Wagner DP,
approximation algorithms, or ILP formulations.

Solver times out after 5 minutes.
"""

import random
import subprocess
import time
from typing import List, Tuple, Set
from native_compiler import CppCompiler, CompilationError

title = "Steiner Tree (C++)"
TIMEOUT_SECONDS = 30
RANDOM_SEED = 32323232


def generate_graph(num_vertices: int, num_edges: int, num_terminals: int,
                   seed: int) -> Tuple[List[Tuple[int, int, int]], List[int]]:
  """Generate weighted graph and terminal vertices."""
  rng = random.Random(seed)

  # Ensure connected by creating spanning tree first
  edges = []
  in_tree = {0}
  for v in range(1, num_vertices):
    u = rng.choice(list(in_tree))
    weight = rng.randint(1, 100)
    edges.append((u, v, weight))
    in_tree.add(v)

  # Add random edges
  while len(edges) < num_edges:
    u = rng.randint(0, num_vertices - 1)
    v = rng.randint(0, num_vertices - 1)
    if u != v:
      weight = rng.randint(1, 100)
      edges.append((min(u, v), max(u, v), weight))

  terminals = rng.sample(range(num_vertices), num_terminals)
  return edges, terminals


TEST_CASES = [
  {
    "vertices": 20,
    "edges": 40,
    "terminals": 5,
    "desc": "20V, 5 terminals"
  },
  {
    "vertices": 50,
    "edges": 120,
    "terminals": 8,
    "desc": "50V, 8 terminals"
  },
  {
    "vertices": 100,
    "edges": 300,
    "terminals": 12,
    "desc": "100V, 12 terminals"
  },
  {
    "vertices": 200,
    "edges": 800,
    "terminals": 15,
    "desc": "200V, 15 terminals"
  },
  {
    "vertices": 500,
    "edges": 2500,
    "terminals": 20,
    "desc": "500V, 20 terminals"
  },
  {
    "vertices": 1000,
    "edges": 6000,
    "terminals": 25,
    "desc": "1KV, 25 terminals"
  },
  {
    "vertices": 2000,
    "edges": 15000,
    "terminals": 30,
    "desc": "2KV, 30 terminals"
  },
  {
    "vertices": 5000,
    "edges": 40000,
    "terminals": 40,
    "desc": "5KV, 40 terminals"
  },
  {
    "vertices": 10000,
    "edges": 100000,
    "terminals": 50,
    "desc": "10KV, 50 terminals"
  },
  {
    "vertices": 20000,
    "edges": 250000,
    "terminals": 75,
    "desc": "20KV, 75 terminals"
  },
  {
    "vertices": 50000,
    "edges": 750000,
    "terminals": 100,
    "desc": "50KV, 100 terminals"
  },
]

INSTANCE_CACHE = {}


def get_instance(subpass: int):
  if subpass not in INSTANCE_CACHE:
    case = TEST_CASES[subpass]
    edges, terminals = generate_graph(case["vertices"], case["edges"], case["terminals"],
                                      RANDOM_SEED + subpass)
    INSTANCE_CACHE[subpass] = (case["vertices"], edges, terminals)
  return INSTANCE_CACHE[subpass]


def format_input(num_vertices: int, edges: List[Tuple[int, int, int]], terminals: List[int]) -> str:
  lines = [f"{num_vertices} {len(edges)} {len(terminals)}"]
  for u, v, w in edges:
    lines.append(f"{u} {v} {w}")
  lines.append(" ".join(map(str, terminals)))
  return "\n".join(lines)


def prepareSubpassPrompt(subPass: int) -> str:
  """Generate the prompt for subpass 0 that handles all Steiner tree complexities."""
  if subPass != 0:
    raise StopIteration

  return f"""You are writing C++ code to solve the Steiner Tree problem.

You must write a C++ solver that can handle ANY Steiner tree complexity from trivial to ludicrous scale:
- **Trivial**: Small graphs (20 vertices, 5 terminals), exact algorithms, brute force
- **Medium**: Moderate graphs (100 vertices, 20 terminals), Dreyfus-Wagner DP
- **Large**: Complex graphs (500 vertices, 50 terminals), approximation algorithms, heuristics
- **Extreme**: Massive graphs (2000+ vertices, 200+ terminals), very fast heuristics, ILP formulations

**The Challenge:**
Your C++ Steiner tree solver will be tested with graphs ranging from simple to very complex instances. The same algorithm must work efficiently across ALL graph complexities.

**Problem:**
Find the minimum-weight tree connecting a set of terminal vertices in a weighted graph. The tree can use any vertices (terminals and Steiner vertices) to achieve minimum total weight. This is NP-Hard.

**Input format (stdin):**
```
num_vertices num_edges num_terminals
u v weight  (for each edge, 0-indexed)
terminal_1 terminal_2 ... terminal_K
```

**Output format (stdout):**
```
tree_weight
edge_1 edge_2 ... edge_M  (edges in the Steiner tree)
```

**Critical Requirements:**
1. **Scalability**: Your algorithm must adapt based on graph size and terminal count
2. **Performance**: Must complete within 5 minutes even for massive graphs
3. **Quality**: Find minimum or near-minimum weight Steiner trees

**Algorithm Strategy Recommendations:**
- **Small graphs (≤50 vertices, ≤10 terminals)**: Can use exact DP, brute force
- **Medium graphs (50-200 vertices, 10-50 terminals)**: Dreyfus-Wagner DP, state space search
- **Large graphs (200-1000 vertices, 50-100 terminals)**: 2-approximation algorithms, heuristics
- **Very Large graphs (>1000 vertices, >100 terminals)**: Very fast heuristics, ILP formulations

**Key Techniques:**
- **Dreyfus-Wagner**: Dynamic programming over terminal subsets
- **Minimum Spanning Tree**: MST-based approximations
- **Prune and Search**: Branch-and-bound with lower bounds
- **ILP Formulation**: Integer linear programming for exact solutions
- **Heuristics**: Distance network, shortest path heuristics

**Implementation Hints:**
- Detect graph complexity and choose appropriate algorithm
- Use efficient graph data structures: adjacency lists, priority queues
- Implement adaptive quality vs speed tradeoffs
- For very large graphs, focus on approximation algorithms
- Handle edge cases: disconnected graphs, single terminal
- Use fast I/O for large inputs

**Success Criteria:**
- All terminals are connected in the tree
- Tree weight is minimized (or near-minimized for large graphs)
- Output format is correct

**Failure Criteria:**
- Not all terminals are connected
- Tree contains cycles
- Timeout without solution
- Invalid output format

**Requirements:**
1. Program must compile with g++ or MSVC (C++17)
2. Read from stdin, write to stdout
3. Handle variable graph sizes and terminal counts
4. Complete within 5 minutes
5. Must handle varying Steiner tree complexities efficiently

Write complete, compilable C++ code with a main() function.
Include adaptive logic that chooses different strategies based on Steiner tree complexity.
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
      "Explain your algorithm approach and how it adapts to different Steiner tree complexities"
    },
    "cpp_code": {
      "type": "string",
      "description": "Complete C++ code with main() function that handles all scales"
    }
  },
  "required": ["reasoning", "cpp_code"],
  "additionalProperties": False
}


def verify_steiner_tree(num_vertices: int, edges: List[Tuple[int, int, int]], terminals: Set[int],
                        tree_edges: List[Tuple[int, int]]) -> Tuple[bool, int, str]:
  """Verify tree connects all terminals. Returns (valid, weight, message)."""
  edge_weights = {}
  for u, v, w in edges:
    edge_weights[(min(u, v), max(u, v))] = w

  # Build adjacency from tree
  adj = {}
  total_weight = 0
  for u, v in tree_edges:
    key = (min(u, v), max(u, v))
    if key not in edge_weights:
      return False, 0, f"Edge ({u},{v}) not in graph"
    total_weight += edge_weights[key]
    adj.setdefault(u, []).append(v)
    adj.setdefault(v, []).append(u)

  if not tree_edges:
    return len(terminals) <= 1, 0, "Empty tree"

  # BFS to check connectivity
  start = tree_edges[0][0]
  visited = {start}
  queue = [start]
  while queue:
    v = queue.pop(0)
    for u in adj.get(v, []):
      if u not in visited:
        visited.add(u)
        queue.append(u)

  missing = terminals - visited
  if missing:
    return False, total_weight, f"Terminals not connected: {missing}"

  return True, total_weight, "Valid"


def mst_upper_bound(num_vertices: int, edges: List[Tuple[int, int, int]],
                    terminals: List[int]) -> int:
  """Compute MST on metric closure of terminals as upper bound."""
  from heapq import heappush, heappop

  # Dijkstra from each terminal
  adj = [[] for _ in range(num_vertices)]
  for u, v, w in edges:
    adj[u].append((v, w))
    adj[v].append((u, w))

  def dijkstra(start):
    dist = [float('inf')] * num_vertices
    dist[start] = 0
    pq = [(0, start)]
    while pq:
      d, u = heappop(pq)
      if d > dist[u]:
        continue
      for v, w in adj[u]:
        if dist[u] + w < dist[v]:
          dist[v] = dist[u] + w
          heappush(pq, (dist[v], v))
    return dist

  # Build metric closure
  term_dists = {t: dijkstra(t) for t in terminals}

  # MST on terminals
  if len(terminals) <= 1:
    return 0

  in_mst = {terminals[0]}
  mst_weight = 0
  while len(in_mst) < len(terminals):
    best = float('inf')
    best_t = None
    for t in terminals:
      if t not in in_mst:
        for s in in_mst:
          if term_dists[s][t] < best:
            best = term_dists[s][t]
            best_t = t
    if best_t:
      in_mst.add(best_t)
      mst_weight += best

  return mst_weight


def gradeAnswer(result: dict, subPass: int, aiEngineName: str) -> tuple:
  if not result or "cpp_code" not in result:
    return 0.0, "No C++ code provided"

  case = TEST_CASES[subPass]
  num_vertices, edges, terminals = get_instance(subPass)

  compiler = CppCompiler(aiEngineName)
  if not compiler.find_compiler():
    return 0.0, "No C++ compiler found"

  try:
    exe_path = compiler.compile(result["cpp_code"])
  except CompilationError as e:
    return 0.0, f"Compilation error: {str(e)[:200]}"

  input_data = format_input(num_vertices, edges, terminals)

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
    header = lines[0].split()
    reported_weight = int(header[0])
    num_tree_edges = int(header[1])

    tree_edges = []
    for i in range(1, num_tree_edges + 1):
      if i < len(lines):
        parts = lines[i].split()
        tree_edges.append((int(parts[0]), int(parts[1])))

    valid, actual_weight, msg = verify_steiner_tree(num_vertices, edges, set(terminals), tree_edges)

    if not valid:
      return 0.2, f"[{case['desc']}] {msg}"

    upper = mst_upper_bound(num_vertices, edges, terminals)
    ratio = actual_weight / upper if upper > 0 else 1.0
    score = min(1.0, 1.5 - ratio * 0.5)

    return max(
      0.5, score), f"[{case['desc']}] Weight {actual_weight} (MST bound: {upper}), {exec_time:.2f}s"

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
  return "<h2>Test 32: Steiner Tree (C++)</h2><p>NP-Hard network design problem.</p>"


def output_summary_html(results: list) -> str:
  total = sum(r[0] for r in results)
  return f'<div class="summary"><p>Total: {total:.2f}/{len(results)}</p></div>'
