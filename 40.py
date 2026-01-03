"""
Test 40: Integer Linear Programming (C++ Implementation)

The LLM must write C++ code to solve integer linear programs - optimize a
linear objective subject to linear constraints with integer variables. NP-Hard.

Subpasses increase problem size, requiring branch-and-bound, cutting planes,
or sophisticated MIP techniques.

Solver times out after 5 minutes.
"""

import random
import subprocess
import time
from typing import List, Tuple, Optional
from native_compiler import CppCompiler, CompilationError

title = "Integer Linear Programming (C++)"
TIMEOUT_SECONDS = 30
RANDOM_SEED = 40404040


def generate_ilp(num_vars: int, num_constraints: int, seed: int):
  """Generate ILP instance: maximize c'x subject to Ax <= b, x >= 0, x integer."""
  rng = random.Random(seed)

  # Objective coefficients
  c = [rng.randint(1, 20) for _ in range(num_vars)]

  # Constraint matrix and RHS
  A = []
  b = []
  for _ in range(num_constraints):
    row = [rng.randint(0, 10) for _ in range(num_vars)]
    # Ensure feasibility by setting RHS based on a feasible point
    feasible_x = [rng.randint(0, 5) for _ in range(num_vars)]
    rhs = sum(row[j] * feasible_x[j] for j in range(num_vars)) + rng.randint(5, 20)
    A.append(row)
    b.append(rhs)

  # Variable upper bounds
  upper = [rng.randint(10, 50) for _ in range(num_vars)]

  return c, A, b, upper


TEST_CASES = [
  {
    "vars": 5,
    "constraints": 5,
    "desc": "5 vars, 5 constraints"
  },
  {
    "vars": 10,
    "constraints": 10,
    "desc": "10 vars, 10 constraints"
  },
  {
    "vars": 15,
    "constraints": 15,
    "desc": "15 vars, 15 constraints"
  },
  {
    "vars": 20,
    "constraints": 25,
    "desc": "20 vars, 25 constraints"
  },
  {
    "vars": 30,
    "constraints": 40,
    "desc": "30 vars, 40 constraints"
  },
  {
    "vars": 50,
    "constraints": 60,
    "desc": "50 vars, 60 constraints"
  },
  {
    "vars": 75,
    "constraints": 100,
    "desc": "75 vars, 100 constraints"
  },
  {
    "vars": 100,
    "constraints": 150,
    "desc": "100 vars, 150 constraints"
  },
  {
    "vars": 150,
    "constraints": 200,
    "desc": "150 vars, 200 constraints"
  },
  {
    "vars": 200,
    "constraints": 300,
    "desc": "200 vars, 300 constraints"
  },
  {
    "vars": 300,
    "constraints": 500,
    "desc": "300 vars, 500 constraints"
  },
]

INSTANCE_CACHE = {}


def get_instance(subpass: int):
  if subpass not in INSTANCE_CACHE:
    case = TEST_CASES[subpass]
    c, A, b, upper = generate_ilp(case["vars"], case["constraints"], RANDOM_SEED + subpass)
    INSTANCE_CACHE[subpass] = (c, A, b, upper)
  return INSTANCE_CACHE[subpass]


def format_input(c: List[int], A: List[List[int]], b: List[int], upper: List[int]) -> str:
  num_vars = len(c)
  num_constraints = len(A)

  lines = [f"{num_vars} {num_constraints}"]
  lines.append(" ".join(map(str, c)))
  lines.append(" ".join(map(str, upper)))
  for i in range(num_constraints):
    lines.append(" ".join(map(str, A[i])) + f" {b[i]}")
  return "\n".join(lines)


def prepareSubpassPrompt(subPass: int) -> str:
  """Generate the prompt for subpass 0 that handles all difficulty levels."""
  if subPass != 0:
    raise StopIteration

  return f"""You are writing PYTHON code to solve the Longest Common Subsequence problem.

You must write a PYTHON solver that can handle ANY problem complexity from trivial to ludicrous scale:
- **Trivial**: Simple instances, basic algorithms
- **Medium**: Moderate instances, efficient algorithms  
- **Large**: Complex instances, advanced algorithms
- **Extreme**: Massive instances, very fast heuristics

**The Challenge:**
Your PYTHON solver will be tested with instances ranging from simple to very complex cases. The same algorithm must work efficiently across ALL problem complexities.

**Problem:**
Find the longest subsequence common to two sequences. A subsequence is a sequence that appears in the same relative order, but not necessarily contiguous.

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
Small sequences (100 chars): classic DP, Medium (1000 chars): optimized DP, Large (10000 chars): Hirschberg's algorithm, Extreme (100000+ chars): very fast heuristics

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

Write complete, compilable PYTHON code.
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
    "cpp_code": {
      "type": "string",
      "description": "Complete C++ code with main() function that handles all scales"
    }
  },
  "required": ["reasoning", "cpp_code"],
  "additionalProperties": False
}


def verify_solution(c: List[int], A: List[List[int]], b: List[int], upper: List[int],
                    x: List[int]) -> Tuple[bool, int, str]:
  """Verify solution feasibility and calculate objective."""
  num_vars = len(c)

  if len(x) != num_vars:
    return False, 0, f"Wrong number of variables: {len(x)} vs {num_vars}"

  # Check bounds
  for i, val in enumerate(x):
    if val < 0:
      return False, 0, f"x[{i}] = {val} < 0"
    if val > upper[i]:
      return False, 0, f"x[{i}] = {val} > upper[{i}] = {upper[i]}"

  # Check constraints
  for i, row in enumerate(A):
    lhs = sum(row[j] * x[j] for j in range(num_vars))
    if lhs > b[i]:
      return False, 0, f"Constraint {i} violated: {lhs} > {b[i]}"

  # Calculate objective
  obj = sum(c[j] * x[j] for j in range(num_vars))

  return True, obj, "Valid"


def greedy_ilp(c: List[int], A: List[List[int]], b: List[int], upper: List[int]) -> int:
  """Greedy heuristic for comparison."""
  num_vars = len(c)
  x = [0] * num_vars

  # Sort variables by c[i] / (sum of coefficients in constraints)
  efficiency = []
  for j in range(num_vars):
    constraint_usage = sum(A[i][j] for i in range(len(A))) + 1
    efficiency.append((c[j] / constraint_usage, j))
  efficiency.sort(reverse=True)

  for _, j in efficiency:
    # Try to increase x[j] as much as possible
    max_increase = upper[j]
    for i, row in enumerate(A):
      if row[j] > 0:
        current_lhs = sum(row[k] * x[k] for k in range(num_vars))
        slack = b[i] - current_lhs
        max_increase = min(max_increase, slack // row[j])
    x[j] = max(0, max_increase)

  return sum(c[j] * x[j] for j in range(num_vars))


def gradeAnswer(result: dict, subPass: int, aiEngineName: str) -> tuple:
  if not result or "cpp_code" not in result:
    return 0.0, "No C++ code provided"

  case = TEST_CASES[subPass]
  c, A, b, upper = get_instance(subPass)

  compiler = CppCompiler(aiEngineName)
  if not compiler.find_compiler():
    return 0.0, "No C++ compiler found"

  try:
    exe_path = compiler.compile(result["cpp_code"])
  except CompilationError as e:
    return 0.0, f"Compilation error: {str(e)[:200]}"

  input_data = format_input(c, A, b, upper)

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
    if not lines:
      return 0.0, "No output"

    if lines[0].strip() == "INFEASIBLE":
      return 0.3, f"[{case['desc']}] Reports infeasible (instance is feasible), {exec_time:.2f}s"

    reported_obj = int(lines[0])
    x = list(map(int, lines[1].split())) if len(lines) > 1 else []

    valid, actual_obj, msg = verify_solution(c, A, b, upper, x)
    if not valid:
      return 0.2, f"[{case['desc']}] {msg}"

    greedy_obj = greedy_ilp(c, A, b, upper)
    ratio = actual_obj / greedy_obj if greedy_obj > 0 else 1.0
    score = min(1.0, 0.5 + ratio * 0.5)

    return score, f"[{case['desc']}] Objective {actual_obj} (greedy: {greedy_obj}), {exec_time:.2f}s"

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
  return "<h2>Test 40: Integer Linear Programming (C++)</h2><p>NP-Hard optimization problem.</p>"


def output_summary_html(results: list) -> str:
  total = sum(r[0] for r in results)
  return f'<div class="summary"><p>Total: {total:.2f}/{len(results)}</p></div>'
