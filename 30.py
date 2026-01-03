"""
Test 30: 3-SAT Solver (Python Implementation)

The LLM must write Python code to solve the Boolean Satisfiability problem
in 3-CNF form. This is the canonical NP-Complete problem.

Subpasses increase clause count and variable count near the phase transition
(clause/variable ratio ~4.26), requiring DPLL, CDCL, or stochastic local search.

Solver times out after 5 minutes.
"""

import random
import subprocess
import time
from typing import List, Tuple, Set, Optional, Dict, Any
from solver_utils import StreamingInputFile

title = "3-SAT Solver (Python)"
TIMEOUT_SECONDS = 30
RANDOM_SEED = 30303030


def generate_3sat(num_vars: int, num_clauses: int, seed: int) -> List[Tuple[int, int, int]]:
  """Generate random 3-SAT instance. Literals are 1-indexed, negative = negation."""
  rng = random.Random(seed)
  clauses = []
  for _ in range(num_clauses):
    vars_in_clause = rng.sample(range(1, num_vars + 1), 3)
    clause = tuple(v if rng.random() > 0.5 else -v for v in vars_in_clause)
    clauses.append(clause)
  return clauses


def is_satisfiable_dpll(clauses: List[Tuple[int, int, int]], num_vars: int) -> Optional[List[bool]]:
  """Simple DPLL for verification (only for small instances)."""
  if num_vars > 30:
    return None  # Too large for simple verification

  def evaluate(assignment):
    for clause in clauses:
      satisfied = False
      for lit in clause:
        var = abs(lit) - 1
        val = assignment[var]
        if (lit > 0 and val) or (lit < 0 and not val):
          satisfied = True
          break
      if not satisfied:
        return False
    return True

  for i in range(2**num_vars):
    assignment = [(i >> j) & 1 == 1 for j in range(num_vars)]
    if evaluate(assignment):
      return assignment
  return None


TEST_CASES = [
  {
    "vars": 20,
    "clauses": 85,
    "desc": "20 vars, 85 clauses"
  },
  {
    "vars": 50,
    "clauses": 213,
    "desc": "50 vars, 213 clauses"
  },
  {
    "vars": 100,
    "clauses": 426,
    "desc": "100 vars, 426 clauses"
  },
  {
    "vars": 200,
    "clauses": 852,
    "desc": "200 vars, 852 clauses"
  },
  {
    "vars": 500,
    "clauses": 2130,
    "desc": "500 vars, 2130 clauses"
  },
  {
    "vars": 1000,
    "clauses": 4260,
    "desc": "1K vars, 4260 clauses"
  },
  {
    "vars": 2000,
    "clauses": 8520,
    "desc": "2K vars, 8520 clauses"
  },
  {
    "vars": 5000,
    "clauses": 21300,
    "desc": "5K vars, 21K clauses"
  },
  {
    "vars": 10000,
    "clauses": 42600,
    "desc": "10K vars, 42K clauses"
  },
  {
    "vars": 20000,
    "clauses": 85200,
    "desc": "20K vars, 85K clauses"
  },
  {
    "vars": 50000,
    "clauses": 213000,
    "desc": "50K vars, 213K clauses"
  },
  # Ludicrous cases for streaming
  {
    "vars": 100000,
    "clauses": 426000,
    "desc": "100K vars, 426K clauses"
  },
  {
    "vars": 500000,
    "clauses": 2130000,
    "desc": "500K vars, 2.1M clauses"
  },
  {
    "vars": 1000000,
    "clauses": 4260000,
    "desc": "1M vars, 4.3M clauses"
  },
  {
    "vars": 5000000,
    "clauses": 21300000,
    "desc": "5M vars, 21M clauses"
  },
  {
    "vars": 10000000,
    "clauses": 42600000,
    "desc": "10M vars, 43M clauses (~600MB)"
  },
  {
    "vars": 20000000,
    "clauses": 85200000,
    "desc": "20M vars, 85M clauses (~1.2GB)"
  },
]

SAT_CACHE: Dict[int, Any] = {}
_INPUT_FILE_CACHE: Dict[int, StreamingInputFile] = {}
STREAMING_THRESHOLD_CLAUSES = 500_000


def get_sat_instance(subpass: int) -> Tuple[int, List[Tuple[int, int, int]]]:
  if subpass not in SAT_CACHE:
    case = TEST_CASES[subpass]
    clauses = generate_3sat(case["vars"], case["clauses"], RANDOM_SEED + subpass)
    SAT_CACHE[subpass] = (case["vars"], clauses)
  return SAT_CACHE[subpass]


def _should_use_streaming(subpass: int) -> bool:
  return TEST_CASES[subpass]["clauses"] > STREAMING_THRESHOLD_CLAUSES


def _get_streaming_input(subpass: int) -> StreamingInputFile:
  if subpass in _INPUT_FILE_CACHE:
    return _INPUT_FILE_CACHE[subpass]

  case = TEST_CASES[subpass]
  cache_key = f"sat30|v={case['vars']}|c={case['clauses']}|seed={RANDOM_SEED + subpass}"

  def generator():
    num_vars, clauses = get_sat_instance(subpass)
    yield f"{num_vars} {len(clauses)}\n"
    for clause in clauses:
      yield f"{clause[0]} {clause[1]} {clause[2]}\n"

  input_file = StreamingInputFile(cache_key, generator, "test30_sat")
  _INPUT_FILE_CACHE[subpass] = input_file
  return input_file


def format_input(num_vars: int, clauses: List[Tuple[int, int, int]]) -> str:
  lines = [f"{num_vars} {len(clauses)}"]
  for clause in clauses:
    lines.append(f"{clause[0]} {clause[1]} {clause[2]}")
  return "\n".join(lines)


def prepareSubpassPrompt(subPass: int) -> str:
  """Generate the prompt for subpass 0 that handles all SAT complexities."""
  if subPass != 0:
    raise StopIteration

  return f"""You are writing Python code to solve the 3-SAT problem.

You must write a Python solver that can handle ANY SAT complexity from trivial to ludicrous scale:
- **Trivial**: Small instances (20 variables, 80 clauses), basic DPLL, brute force
- **Medium**: Medium instances (100 variables, 400 clauses), optimized DPLL, unit propagation
- **Large**: Complex instances (500 variables, 2000 clauses), CDCL algorithm, clause learning
- **Extreme**: Massive instances (2000+ variables, 8000+ clauses), stochastic local search, heuristics

**The Challenge:**
Your Python 3-SAT solver will be tested with instances ranging from simple to very complex Boolean formulas. The same algorithm must work efficiently across ALL SAT complexities.

**Problem:**
Solve the Boolean Satisfiability problem in 3-CNF form. Given a Boolean formula in conjunctive normal form where each clause has exactly 3 literals, determine if there exists an assignment that makes the formula true. This is the canonical NP-Complete problem.

**Input format (stdin):**
```
num_variables num_clauses
lit1 lit2 lit3  (for each clause, 1-indexed, negative = negation)
...
```

**Output format (stdout):**
```
SATISFIABLE or UNSATISFIABLE
[var1 var2 ... varN]  (only if SATISFIABLE, 1=true, 0=false)
```

**Critical Requirements:**
1. **Scalability**: Your algorithm must adapt based on variable and clause count
2. **Performance**: Must complete within 5 minutes even for massive instances
3. **Quality**: Find satisfying assignments or correctly prove unsatisfiability

**Algorithm Strategy Recommendations:**
- **Small instances (≤50 variables)**: Can use basic DPLL with backtracking
- **Medium instances (50-200 variables)**: Optimized DPLL with unit propagation, pure literals
- **Large instances (200-1000 variables)**: CDCL algorithm with clause learning, conflict analysis
- **Very Large instances (>1000 variables)**: Stochastic local search, WalkSAT, heuristics

**Key Techniques:**
- **DPLL**: Davis-Putnam-Logemann-Loveland algorithm with backtracking
- **Unit propagation**: Forced assignments from unit clauses
- **Pure literal elimination**: Variables appearing with only one polarity
- **CDCL**: Conflict-Driven Clause Learning with non-chronological backtracking
- **VSIDS**: Variable State Independent Decaying Sum for heuristic ordering

**Implementation Hints:**
- Detect formula complexity and choose appropriate SAT algorithm
- Use efficient data structures: implication graphs, watched literals
- Implement adaptive quality vs speed tradeoffs
- For very large instances, focus on stochastic local search
- Handle edge cases: empty formulas, tautologies, contradictions
- Use fast parsing for large inputs

**Success Criteria:**
- Correctly determine satisfiability
- If SATISFIABLE, provide valid variable assignment
- Complete within time limit

**Failure Criteria:**
- Incorrect satisfiability determination
- Invalid variable assignment
- Timeout without conclusion

**Requirements:**
1. Program must run with Python 3.x
2. Read from stdin, write to stdout
3. Handle variable numbers and clause counts efficiently
4. Complete within 5 minutes
5. Must handle varying SAT complexities efficiently

Write complete, runnable Python code.
Include adaptive logic that chooses different strategies based on SAT complexity.
"""
  # List of subpasses to grade the single answer against all difficulty levels


extraGradeAnswerRuns = list(range(len(TEST_CASES)))

structure = {
  "type": "object",
  "properties": {
    "reasoning": {
      "type": "string",
      "description":
      "Explain your algorithm approach and how it adapts to different SAT complexities"
    },
    "python_code": {
      "type": "string",
      "description": "Complete Python code with solver function that handles all scales"
    }
  },
  "required": ["reasoning", "python_code"],
  "additionalProperties": False
}


def verify_assignment(clauses: List[Tuple[int, int, int]], assignment: List[bool]) -> bool:
  for clause in clauses:
    satisfied = False
    for lit in clause:
      var_idx = abs(lit) - 1
      if var_idx >= len(assignment):
        return False
      val = assignment[var_idx]
      if (lit > 0 and val) or (lit < 0 and not val):
        satisfied = True
        break
    if not satisfied:
      return False
  return True


def gradeAnswer(result: dict, subPass: int, aiEngineName: str) -> tuple:
  if not result or "python_code" not in result:
    return 0.0, "No Python code provided"

  case = TEST_CASES[subPass]
  use_streaming = _should_use_streaming(subPass)

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

      # For very large cases, skip verification
      if case["clauses"] > 10_000_000:
        start = time.time()
        with open(input_file_path, 'r') as f:
          proc = subprocess.run(["python", "-c", result["python_code"]],
                                stdin=f,
                                capture_output=True,
                                text=True,
                                timeout=TIMEOUT_SECONDS)
        exec_time = time.time() - start
        if proc.returncode == 0:
          return 0.8, f"[{case['desc']}] Completed in {exec_time:.2f}s (verification skipped)"
        else:
          return 0.0, f"Runtime error: {proc.stderr[:200]}"

      start = time.time()
      with open(input_file_path, 'r') as f:
        proc = subprocess.run(["python", "-c", result["python_code"]],
                              stdin=f,
                              capture_output=True,
                              text=True,
                              timeout=TIMEOUT_SECONDS)
      exec_time = time.time() - start
    else:
      num_vars, clauses = get_sat_instance(subPass)
      input_data = format_input(num_vars, clauses)

      start = time.time()
      proc = subprocess.run(["python", "-c", result["python_code"]],
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

    if lines[0].strip() == "UNSAT":
      # Can't easily verify UNSAT, give partial credit
      return 0.5, f"[{case['desc']}] Reports UNSAT, {exec_time:.2f}s"

    if lines[0].strip() == "SAT":
      if len(lines) < 2:
        return 0.2, f"[{case['desc']}] SAT but no assignment"

      assignment = [x == "1" for x in lines[1].split()]
      if len(assignment) < num_vars:
        return 0.3, f"[{case['desc']}] Incomplete assignment"

      if verify_assignment(clauses, assignment):
        return 1.0, f"[{case['desc']}] Valid SAT assignment, {exec_time:.2f}s"
      else:
        return 0.2, f"[{case['desc']}] Invalid assignment"

    return 0.1, f"[{case['desc']}] Unknown output format"

  except subprocess.TimeoutExpired:
    return 0.1, f"[{case['desc']}] Timeout"
  except Exception as e:
    return 0.0, f"[{case['desc']}] Error: {str(e)[:100]}"


def resultToNiceReport(result: dict, subPass: int, aiEngineName: str) -> str:
  if not result:
    return "<p style='color:red'>No result provided</p>"
  case = TEST_CASES[subPass]
  html = f"<h4>3-SAT Solver - {case['desc']}</h4>"
  if "reasoning" in result:
    r = result['reasoning'][:400] + ('...' if len(result.get('reasoning', '')) > 400 else '')
    html += f"<p><strong>Approach:</strong> {r.replace('<', '&lt;').replace('>', '&gt;')}</p>"
  if "python_code" in result:
    code = result["python_code"].replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    html += f"<details><summary>View Python Code ({len(result['python_code'])} chars)</summary><pre>{code}</pre></details>"
  return html


highLevelSummary = """
3-SAT determines if a boolean formula in CNF with 3 literals per clause is satisfiable.

**Algorithms:**
- **DPLL**: Backtracking with unit propagation
- **CDCL**: Conflict-driven clause learning (modern SAT solvers)
- **WalkSAT**: Randomized local search
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
