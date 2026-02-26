"""
Test 30: 3-SAT Solver (Python Implementation)

The LLM must write Python code to solve the Boolean Satisfiability problem
in 3-CNF (Conjunctive Normal Form).

3-SAT is the canonical NP-Complete problem.  You have N boolean variables and
M "clauses".  Each clause contains exactly 3 "literals" (a variable or its
negation) and requires at least one of them to be True.  The solver must find
a True/False assignment for every variable that satisfies all clauses at once.

All instances are generated with a planted satisfying assignment hidden from
the solver, so the correct answer is always SAT.  The solver must discover a
valid assignment on its own.

Subpasses scale from 20 variables to 20M variables at the phase-transition
ratio (~4.26 clauses per variable), where random 3-SAT is hardest.

Solver times out after 5 minutes.
"""

import random
import subprocess
import time
import math
from typing import List, Tuple, Set, Optional, Dict, Any
from solver_utils import StreamingInputFile

title = "3-SAT Solver (Python)"

tags = [
  "python",
  "structured response",
  "np hard",
  "constraint satisfaction",
]
TIMEOUT_SECONDS = 300
RANDOM_SEED = 30303030


def generate_3sat(num_vars: int, num_clauses: int,
                  seed: int) -> Tuple[List[Tuple[int, int, int]], List[bool]]:
  """Generate a 3-SAT instance with a planted satisfying assignment.

  1. Pick a random truth assignment (the planted solution).
  2. For each clause, pick 3 distinct variables with random polarities.
  3. If the clause is unsatisfied by the planted assignment, flip one
     randomly chosen literal so it agrees with the assignment.

  This guarantees every clause is satisfiable while keeping the clause
  distribution close to random 3-SAT at the phase-transition ratio.

  Returns (clauses, planted_assignment).
  """
  rng = random.Random(seed)
  assignment = [rng.choice([True, False]) for _ in range(num_vars)]

  clauses = []
  for _ in range(num_clauses):
    vars_in_clause = rng.sample(range(1, num_vars + 1), 3)
    lits = [v if rng.random() > 0.5 else -v for v in vars_in_clause]
    if not any((l > 0 and assignment[abs(l) - 1]) or
               (l < 0 and not assignment[abs(l) - 1]) for l in lits):
      fix = rng.randrange(3)
      v = abs(lits[fix])
      lits[fix] = v if assignment[v - 1] else -v
    clauses.append(tuple(lits))

  return clauses, assignment


def _get_planted_assignment(num_vars: int, seed: int) -> List[bool]:
  """Regenerate just the planted assignment from the seed (no clause generation)."""
  rng = random.Random(seed)
  return [rng.choice([True, False]) for _ in range(num_vars)]


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
LAST_SAT_VIZ: Dict[Tuple[int, str], dict] = {}
STREAMING_THRESHOLD_CLAUSES = 500_000


def get_sat_instance(subpass: int) -> Tuple[int, List[Tuple[int, int, int]], List[bool]]:
  """Get or generate SAT instance.  Returns (num_vars, clauses, planted_assignment)."""
  if subpass not in SAT_CACHE:
    case = TEST_CASES[subpass]
    clauses, assignment = generate_3sat(case["vars"], case["clauses"], RANDOM_SEED + subpass)
    SAT_CACHE[subpass] = (case["vars"], clauses, assignment)
  return SAT_CACHE[subpass]


def _should_use_streaming(subpass: int) -> bool:
  return TEST_CASES[subpass]["clauses"] > STREAMING_THRESHOLD_CLAUSES


def _get_streaming_input(subpass: int) -> StreamingInputFile:
  if subpass in _INPUT_FILE_CACHE:
    return _INPUT_FILE_CACHE[subpass]

  case = TEST_CASES[subpass]
  cache_key = f"sat30v2|v={case['vars']}|c={case['clauses']}|seed={RANDOM_SEED + subpass}"

  def generator():
    # Stream clauses directly using the same RNG sequence as generate_3sat,
    # so we never need to hold all clauses in memory for huge instances.
    num_vars = case["vars"]
    num_clauses = case["clauses"]
    seed = RANDOM_SEED + subpass
    rng = random.Random(seed)
    assignment = [rng.choice([True, False]) for _ in range(num_vars)]

    yield f"{num_vars} {num_clauses}\n"
    for _ in range(num_clauses):
      vars_in_clause = rng.sample(range(1, num_vars + 1), 3)
      lits = [v if rng.random() > 0.5 else -v for v in vars_in_clause]
      if not any((l > 0 and assignment[abs(l) - 1]) or
                 (l < 0 and not assignment[abs(l) - 1]) for l in lits):
        fix = rng.randrange(3)
        v = abs(lits[fix])
        lits[fix] = v if assignment[v - 1] else -v
      yield f"{lits[0]} {lits[1]} {lits[2]}\n"

  input_file = StreamingInputFile(cache_key, generator, "test30_sat")
  _INPUT_FILE_CACHE[subpass] = input_file
  return input_file


def _verify_streaming(subpass: int, assignment: List[bool]) -> Tuple[bool, int, int]:
  """Verify assignment by streaming through the cached input file (memory-efficient)."""
  input_file = _get_streaming_input(subpass)
  file_path = input_file.get_file_path()
  sat_count = 0
  total = 0
  with open(file_path, 'r') as f:
    f.readline()  # skip header
    for line in f:
      parts = line.split()
      if len(parts) < 3:
        continue
      total += 1
      for p in parts[:3]:
        lit = int(p)
        var_idx = abs(lit) - 1
        val = assignment[var_idx]
        if (lit > 0 and val) or (lit < 0 and not val):
          sat_count += 1
          break
  return sat_count == total, sat_count, total


def format_input(num_vars: int, clauses: List[Tuple[int, int, int]]) -> str:
  lines = [f"{num_vars} {len(clauses)}"]
  for clause in clauses:
    lines.append(f"{clause[0]} {clause[1]} {clause[2]}")
  return "\n".join(lines)


def prepareSubpassPrompt(subPass: int) -> str:
  """Generate the prompt for subpass 0."""
  if subPass != 0:
    raise StopIteration

  return f"""You are writing Python code to solve the 3-SAT (Boolean Satisfiability) problem, which is NP-Complete.

**What is 3-SAT?**
You have N boolean variables (x1, x2, ..., xN), each of which is either True or False.
You are given M "clauses".  Each clause is a constraint containing exactly 3 "literals".
A literal is either:
  - A positive integer  (e.g.  5) meaning "variable 5 must be True", or
  - A negative integer (e.g. -5) meaning "variable 5 must be False".

A clause is **satisfied** if ANY of its 3 literals evaluates to True.
The formula is **satisfiable** if ALL clauses can be satisfied simultaneously
by some assignment of True/False to the variables.

**Example:**
  Variables: x1, x2, x3
  Clause "1 -2 3" means: x1=True  OR  x2=False  OR  x3=True.
  If x1=True, x2=True, x3=False  →  clause satisfied (x1=True is enough).

Your task: find an assignment of True/False to every variable that satisfies
every clause.  All test instances are guaranteed to be satisfiable.

**Input format (stdin):**
```
num_variables num_clauses
lit1 lit2 lit3
lit1 lit2 lit3
...
```
Variables are 1-indexed.  Negative literals mean negation.

**Output format (stdout):**
```
SAT
v1 v2 v3 ... vN
```
Where each vi is 1 (True) or 0 (False) for variable i, space-separated.

**Grading:**
- Output "SAT" on the first line, then the assignment on the second line.
- Every clause must be satisfied by your assignment.
- Timeouts, crashes, wrong format, or unsatisfied clauses all score 0.

**Requirements:**
1. Read from stdin, write to stdout.
2. Handle up to 20M variables and 85M clauses.
3. Complete within 5 minutes.

Write complete, runnable Python code.
"""


extraGradeAnswerRuns = list(range(len(TEST_CASES)))

structure = {
  "type": "object",
  "properties": {
    "reasoning": {
      "type": "string",
      "description": "Explain your algorithm approach for solving 3-SAT at scale"
    },
    "python_code": {
      "type": "string",
      "description":
      "Complete Python code that reads a 3-SAT instance from stdin and outputs a satisfying assignment"
    }
  },
  "required": ["reasoning", "python_code"],
  "additionalProperties": False
}


def verify_assignment(clauses: List[Tuple[int, int, int]],
                      assignment: List[bool]) -> Tuple[bool, int, int]:
  """Verify assignment.  Returns (all_satisfied, num_satisfied, total_clauses)."""
  sat_count = 0
  for clause in clauses:
    for lit in clause:
      var_idx = abs(lit) - 1
      if var_idx >= len(assignment):
        return False, sat_count, len(clauses)
      val = assignment[var_idx]
      if (lit > 0 and val) or (lit < 0 and not val):
        sat_count += 1
        break
  return sat_count == len(clauses), sat_count, len(clauses)


def gradeAnswer(result: dict, subPass: int, aiEngineName: str) -> tuple:
  if not result or "python_code" not in result:
    return 0.0, "No Python code provided"

  case = TEST_CASES[subPass]
  num_vars = case["vars"]
  use_streaming = _should_use_streaming(subPass)

  try:
    # --- run solver ---
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
      with open(input_file_path, 'r') as f:
        proc = subprocess.run(["python", "-c", result["python_code"]],
                              stdin=f,
                              capture_output=True,
                              text=True,
                              timeout=TIMEOUT_SECONDS)
      exec_time = time.time() - start
    else:
      _, clauses, _ = get_sat_instance(subPass)
      input_data = format_input(num_vars, clauses)

      start = time.time()
      proc = subprocess.run(["python", "-c", result["python_code"]],
                            input=input_data,
                            capture_output=True,
                            text=True,
                            timeout=TIMEOUT_SECONDS)
      exec_time = time.time() - start

    if proc.returncode != 0:
      return 0.0, f"[{case['desc']}] Runtime error: {proc.stderr[:200]}"

    # --- parse output ---
    lines = proc.stdout.strip().split('\n')
    if not lines:
      return 0.0, f"[{case['desc']}] No output"

    first_line = lines[0].strip().upper()
    if first_line in ("UNSAT", "UNSATISFIABLE"):
      return 0.0, f"[{case['desc']}] Reports UNSAT (all instances are SAT), {exec_time:.2f}s"

    if first_line not in ("SAT", "SATISFIABLE"):
      return 0.0, f"[{case['desc']}] Unrecognised first line: {lines[0].strip()[:60]}"

    if len(lines) < 2:
      return 0.0, f"[{case['desc']}] SAT but no assignment on second line"

    try:
      values = lines[1].split()
      assignment = [x == "1" for x in values]
    except Exception:
      return 0.0, f"[{case['desc']}] Could not parse assignment"

    if len(assignment) < num_vars:
      return 0.0, f"[{case['desc']}] Assignment has {len(assignment)} values, need {num_vars}"

    # --- verify ---
    if use_streaming:
      all_sat, sat_count, total = _verify_streaming(subPass, assignment)
    else:
      _, clauses, _ = get_sat_instance(subPass)
      all_sat, sat_count, total = verify_assignment(clauses, assignment)

    # Build visualisation for small cases
    if case["vars"] <= 100 and not use_streaming:
      _, clauses, _ = get_sat_instance(subPass)
      LAST_SAT_VIZ[(subPass, aiEngineName)] = _build_sat_viz(
        num_vars, clauses, assignment, all_sat, sat_count, total
      )

    if all_sat:
      return 1.0, f"[{case['desc']}] All {total} clauses satisfied, {exec_time:.2f}s"
    else:
      return 0.0, f"[{case['desc']}] {total - sat_count}/{total} clauses unsatisfied, {exec_time:.2f}s"

  except subprocess.TimeoutExpired:
    return 0.0, f"[{case['desc']}] Timeout ({TIMEOUT_SECONDS}s)"
  except Exception as e:
    return 0.0, f"[{case['desc']}] Error: {str(e)[:100]}"


def resultToNiceReport(result: dict, subPass: int, aiEngineName: str) -> str:
  if not result:
    return "<p style='color:red'>No result provided</p>"
  case = TEST_CASES[subPass]
  html = f"<h4>3-SAT Solver - {case['desc']}</h4>"
  if "reasoning" in result and subPass == 0:
    r = result['reasoning'][:400] + ('...' if len(result.get('reasoning', '')) > 400 else '')
    html += f"<p><strong>Approach:</strong> {r.replace('<', '&lt;').replace('>', '&gt;')}</p>"
  if "python_code" in result and subPass == 0:
    code = result["python_code"].replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    html += f"<details><summary>View Python Code ({len(result['python_code'])} chars)</summary><pre>{code}</pre></details>"
  viz = LAST_SAT_VIZ.get((subPass, aiEngineName))
  if viz and viz.get("num_vars", 0) <= 100:
    html += _generate_sat_svg(viz)
  return html


def _build_sat_viz(num_vars: int, clauses: List[Tuple[int, int, int]],
                   assignment: List[bool], all_sat: bool,
                   sat_count: int, total: int) -> dict:
  """Build visualization data for the SAT solution."""
  unsat_clauses = []
  for i, clause in enumerate(clauses):
    satisfied = False
    for lit in clause:
      var_idx = abs(lit) - 1
      val = assignment[var_idx] if var_idx < len(assignment) else False
      if (lit > 0 and val) or (lit < 0 and not val):
        satisfied = True
        break
    if not satisfied:
      unsat_clauses.append((i, clause))

  return {
    "num_vars": num_vars,
    "num_clauses": len(clauses),
    "clauses": clauses,
    "assignment": assignment[:num_vars],
    "all_sat": all_sat,
    "sat_count": sat_count,
    "total": total,
    "unsat_clauses": unsat_clauses,
  }


def _generate_sat_svg(viz: dict) -> str:
  """Generate an SVG visualization of the 3-SAT solution.

  Layout:
  - Top bar: variable assignments (green = True, grey = False)
  - Grid: each small cell is one clause (green = satisfied, red = unsatisfied)
  - Below: details of any unsatisfied clauses
  """
  num_vars = viz["num_vars"]
  num_clauses = viz["num_clauses"]
  assignment = viz["assignment"]
  clauses = viz["clauses"]
  sat_count = viz["sat_count"]
  total = viz["total"]
  unsat_clauses = viz["unsat_clauses"]

  # Variable assignment bar
  cell_w = max(4, min(14, 600 // max(1, num_vars)))
  var_bar_w = cell_w * num_vars

  # Clause satisfaction grid
  cols = min(num_clauses, max(20, int(math.sqrt(num_clauses * 2))))
  rows = (num_clauses + cols - 1) // cols
  clause_cell = max(3, min(10, 600 // max(1, cols)))
  grid_w = clause_cell * cols
  grid_h = clause_cell * rows

  total_w = max(var_bar_w, grid_w, 200)
  label_h = 16
  var_bar_h = cell_w
  gap = 6
  total_h = label_h + var_bar_h + gap + label_h + grid_h + 4

  # Variable cells
  var_cells = []
  for i in range(num_vars):
    x = i * cell_w
    fill = "#22c55e" if assignment[i] else "#475569"
    var_cells.append(
      f"<rect x='{x}' y='{label_h}' width='{cell_w}' height='{var_bar_h}' "
      f"fill='{fill}' stroke='#0f172a' stroke-width='0.3' />"
    )

  # Clause satisfaction per clause
  clause_sat = []
  for clause in clauses:
    satisfied = False
    for lit in clause:
      var_idx = abs(lit) - 1
      val = assignment[var_idx] if var_idx < len(assignment) else False
      if (lit > 0 and val) or (lit < 0 and not val):
        satisfied = True
        break
    clause_sat.append(satisfied)

  clause_y_offset = label_h + var_bar_h + gap + label_h
  clause_cells = []
  for i, sat in enumerate(clause_sat):
    col = i % cols
    row = i // cols
    x = col * clause_cell
    y = clause_y_offset + row * clause_cell
    fill = "#166534" if sat else "#dc2626"
    clause_cells.append(
      f"<rect x='{x}' y='{y}' width='{clause_cell}' height='{clause_cell}' "
      f"fill='{fill}' stroke='#0f172a' stroke-width='0.2' />"
    )

  if viz["all_sat"]:
    status = f"All {total} clauses satisfied"
  else:
    status = f"{total - sat_count}/{total} clauses UNSATISFIED"

  parts = [
    "<div style='margin:12px 0;padding:10px;border:1px solid #1f2937;border-radius:8px;background:#0b1120;'>",
    f"<div style='color:#e2e8f0;font-size:13px;margin-bottom:6px;'>"
    f"<strong>3-SAT Visualization</strong> &mdash; {status}</div>",
    f"<div style='color:#94a3b8;font-size:12px;margin-bottom:6px;'>"
    f"Variables: {num_vars} | Clauses: {num_clauses} | "
    f"Satisfied: {sat_count}/{num_clauses}</div>",
    f"<svg width='100%' viewBox='0 0 {total_w} {total_h}' "
    f"style='background:#0b1120;border:1px solid #334155;border-radius:6px;'>",
    f"<text x='0' y='12' fill='#94a3b8' font-size='10' font-family='sans-serif'>"
    f"Variable assignments (green=True, grey=False):</text>",
    "<g>",
    *var_cells,
    "</g>",
    f"<text x='0' y='{clause_y_offset - 2}' fill='#94a3b8' font-size='10' font-family='sans-serif'>"
    f"Clause satisfaction (green=satisfied, red=unsatisfied):</text>",
    "<g>",
    *clause_cells,
    "</g>",
    "</svg>",
  ]

  if unsat_clauses:
    parts.append("<div style='color:#fca5a5;font-size:11px;margin-top:6px;font-family:monospace;'>")
    for idx, clause in unsat_clauses[:20]:
      lits_str = ", ".join(
        f"x{abs(l)}={'T' if l > 0 else 'F'} (actual={'T' if assignment[abs(l)-1] else 'F'})"
        for l in clause
      )
      parts.append(f"Clause {idx}: needs {lits_str}<br>")
    if len(unsat_clauses) > 20:
      parts.append(f"... and {len(unsat_clauses) - 20} more unsatisfied clauses")
    parts.append("</div>")

  parts.append(
    "<div style='color:#64748b;font-size:11px;margin-top:6px;'>"
    "Top bar: variable assignments. Grid: each cell is one clause.</div>"
  )
  parts.append("</div>")
  return "\n".join(parts)


highLevelSummary = """
<p>Given a logic puzzle made of true/false variables &mdash; where each rule says
&ldquo;at least one of these three things must be true&rdquo; and all rules must hold
simultaneously &mdash; find an assignment that satisfies every rule.</p>
<p>This is the classic 3-SAT problem, one of the first problems proven to be
NP-complete. Every test instance has a hidden satisfying assignment the AI must
discover. Subpasses increase the number of variables and clauses.</p>
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
