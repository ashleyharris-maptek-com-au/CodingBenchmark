"""
Test 33: Quadratic Assignment Problem (C# Implementation)

The LLM must write C# code to assign facilities to locations minimizing
the sum of flow*distance products. This is NP-Hard.

Subpasses increase problem size, requiring simulated annealing, tabu search,
genetic algorithms, or branch-and-bound with bounds.

Solver times out after 5 minutes.
"""

import random
import subprocess
import time
from typing import List, Tuple, Dict, Any
from native_compiler import CSharpCompiler, CompilationError, ExecutionError
from solver_utils import StreamingInputFile

title = "Quadratic Assignment Problem (C#)"
TIMEOUT_SECONDS = 30
RANDOM_SEED = 33333333


def generate_qap(n: int, seed: int) -> Tuple[List[List[int]], List[List[int]]]:
  """Generate flow and distance matrices."""
  rng = random.Random(seed)

  # Flow matrix (asymmetric)
  flow = [[rng.randint(0, 100) if i != j else 0 for j in range(n)] for i in range(n)]

  # Distance matrix (symmetric)
  dist = [[0] * n for _ in range(n)]
  for i in range(n):
    for j in range(i + 1, n):
      d = rng.randint(1, 100)
      dist[i][j] = d
      dist[j][i] = d

  return flow, dist


def evaluate_assignment(flow: List[List[int]], dist: List[List[int]], perm: List[int]) -> int:
  """Calculate total cost of assignment."""
  n = len(perm)
  cost = 0
  for i in range(n):
    for j in range(n):
      cost += flow[i][j] * dist[perm[i]][perm[j]]
  return cost


TEST_CASES = [
  {
    "n": 8,
    "desc": "8 facilities"
  },
  {
    "n": 10,
    "desc": "10 facilities"
  },
  {
    "n": 12,
    "desc": "12 facilities"
  },
  {
    "n": 15,
    "desc": "15 facilities"
  },
  {
    "n": 20,
    "desc": "20 facilities"
  },
  {
    "n": 25,
    "desc": "25 facilities"
  },
  {
    "n": 30,
    "desc": "30 facilities"
  },
  {
    "n": 40,
    "desc": "40 facilities"
  },
  {
    "n": 50,
    "desc": "50 facilities"
  },
  {
    "n": 75,
    "desc": "75 facilities"
  },
  {
    "n": 100,
    "desc": "100 facilities"
  },
  # Ludicrous cases for streaming
  {
    "n": 200,
    "desc": "200 facilities (~800KB)"
  },
  {
    "n": 500,
    "desc": "500 facilities (~5MB)"
  },
  {
    "n": 1000,
    "desc": "1K facilities (~20MB)"
  },
  {
    "n": 2000,
    "desc": "2K facilities (~80MB)"
  },
  {
    "n": 5000,
    "desc": "5K facilities (~500MB)"
  },
  {
    "n": 7000,
    "desc": "7K facilities (~1GB)"
  },
]

INSTANCE_CACHE: Dict[int, Any] = {}
_INPUT_FILE_CACHE: Dict[int, StreamingInputFile] = {}
STREAMING_THRESHOLD_N = 200


def get_instance(subpass: int):
  if subpass not in INSTANCE_CACHE:
    case = TEST_CASES[subpass]
    flow, dist = generate_qap(case["n"], RANDOM_SEED + subpass)
    INSTANCE_CACHE[subpass] = (flow, dist)
  return INSTANCE_CACHE[subpass]


def _should_use_streaming(subpass: int) -> bool:
  return TEST_CASES[subpass]["n"] > STREAMING_THRESHOLD_N


def _get_streaming_input(subpass: int) -> StreamingInputFile:
  if subpass in _INPUT_FILE_CACHE:
    return _INPUT_FILE_CACHE[subpass]

  case = TEST_CASES[subpass]
  cache_key = f"qap33|n={case['n']}|seed={RANDOM_SEED + subpass}"

  def generator():
    flow, dist = get_instance(subpass)
    n = len(flow)
    yield f"{n}\n"
    for row in flow:
      yield " ".join(map(str, row)) + "\n"
    for row in dist:
      yield " ".join(map(str, row)) + "\n"

  input_file = StreamingInputFile(cache_key, generator, "test33_qap")
  _INPUT_FILE_CACHE[subpass] = input_file
  return input_file


def format_input(flow: List[List[int]], dist: List[List[int]]) -> str:
  n = len(flow)
  lines = [str(n)]
  for row in flow:
    lines.append(" ".join(map(str, row)))
  for row in dist:
    lines.append(" ".join(map(str, row)))
  return "\n".join(lines)


def prepareSubpassPrompt(subPass: int) -> str:
  """Generate the prompt for subpass 0 that handles all difficulty levels."""
  if subPass != 0:
    raise StopIteration

  return f"""You are writing RUST code to solve the Maximum Flow problem.

You must write a RUST solver that can handle ANY problem complexity from trivial to ludicrous scale:
- **Trivial**: Simple instances, basic algorithms
- **Medium**: Moderate instances, efficient algorithms  
- **Large**: Complex instances, advanced algorithms
- **Extreme**: Massive instances, very fast heuristics

**The Challenge:**
Your RUST solver will be tested with instances ranging from simple to very complex cases. The same algorithm must work efficiently across ALL problem complexities.

**Problem:**
Find the maximum possible flow from a source to a sink in a flow network.

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
Small networks (50 nodes): Ford-Fulkerson, Medium (500 nodes): Dinic's algorithm, Large (5000 nodes): Push-relabel, Extreme (50000+ nodes): very fast heuristics

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
    "csharp_code": {
      "type": "string",
      "description": "Complete C# code with Main method that handles all scales"
    }
  },
  "required": ["reasoning", "csharp_code"],
  "additionalProperties": False
},


def greedy_assignment(flow: List[List[int]], dist: List[List[int]]) -> Tuple[List[int], int]:
  """Simple greedy for comparison."""
  n = len(flow)
  # Sort facilities by total flow
  facility_flow = [(sum(flow[i]) + sum(flow[j][i] for j in range(n)), i) for i in range(n)]
  facility_flow.sort(reverse=True)

  # Sort locations by total distance (prefer central)
  loc_dist = [(sum(dist[i]), i) for i in range(n)]
  loc_dist.sort()

  perm = [0] * n
  for rank, (_, fac) in enumerate(facility_flow):
    perm[fac] = loc_dist[rank][1]

  return perm, evaluate_assignment(flow, dist, perm)


def gradeAnswer(result: dict, subPass: int, aiEngineName: str) -> tuple:
  if not result or "csharp_code" not in result:
    return 0.0, "No C# code provided"

  case = TEST_CASES[subPass]
  use_streaming = _should_use_streaming(subPass)

  compiler = CSharpCompiler(aiEngineName)
  if not compiler.find_compiler():
    return 0.0, "No C# compiler found"

  try:
    exe_path = compiler.compile(result["csharp_code"])
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
      stdout, stderr, exec_time, retcode = compiler.execute(exe_path,
                                                            timeout=TIMEOUT_SECONDS,
                                                            stdin_file=input_file_path)

      if retcode != 0:
        return 0.0, f"Runtime error: {stderr[:200]}"

      # Skip verification for very large cases
      if case["n"] > 1000:
        lines = stdout.strip().split('\n')
        if lines:
          reported_cost = int(lines[0])
          return 0.8, f"[{case['desc']}] Cost {reported_cost} in {exec_time:.2f}s (verification skipped)"
        return 0.2, f"[{case['desc']}] No output"
    else:
      flow, dist = get_instance(subPass)
      input_data = format_input(flow, dist)
      start = time.time()
      stdout, stderr, exec_time, retcode = compiler.execute(exe_path, input_data, TIMEOUT_SECONDS)

      if retcode != 0:
        return 0.0, f"Runtime error: {stderr[:200]}"

    lines = stdout.strip().split('\n')
    reported_cost = int(lines[0])
    perm = list(map(int, lines[1].split()))
    flow, dist = get_instance(subPass)
    n = len(flow)

    if len(perm) != n or set(perm) != set(range(n)):
      return 0.2, f"[{case['desc']}] Invalid permutation"

    actual_cost = evaluate_assignment(flow, dist, perm)
    if actual_cost != reported_cost:
      return 0.3, f"[{case['desc']}] Cost mismatch: {actual_cost} vs {reported_cost}"

    _, greedy_cost = greedy_assignment(flow, dist)
    ratio = actual_cost / greedy_cost if greedy_cost > 0 else 1.0
    score = min(1.0, 1.5 - ratio * 0.5)

    return max(
      0.5, score), f"[{case['desc']}] Cost {actual_cost} (greedy: {greedy_cost}), {exec_time:.2f}s"

  except Exception as e:
    return 0.0, f"[{case['desc']}] Error: {str(e)[:100]}"


def resultToNiceReport(result: dict, subPass: int, aiEngineName: str) -> str:
  if not result:
    return "<p style='color:red'>No result provided</p>"
  case = TEST_CASES[subPass]
  html = f"<h4>Quadratic Assignment - {case['desc']}</h4>"
  if "reasoning" in result:
    r = result['reasoning'][:400] + ('...' if len(result.get('reasoning', '')) > 400 else '')
    html += f"<p><strong>Approach:</strong> {r.replace('<', '&lt;').replace('>', '&gt;')}</p>"
  if "csharp_code" in result:
    code = result["csharp_code"].replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    html += f"<details><summary>View C# Code ({len(result['csharp_code'])} chars)</summary><pre>{code}</pre></details>"
  return html


highLevelSummary = """
QAP assigns facilities to locations minimizing flow*distance cost.

**Algorithms:**
- **Simulated Annealing**: Good quality solutions
- **Tabu Search**: Avoids local minima
- **Branch and Bound**: Exact but slow for large instances
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
