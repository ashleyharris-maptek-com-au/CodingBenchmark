"""
Test 40: Integer Linear Programming (C++ Implementation)

The model must write C++ code for a 0/1 ILP family:
maximize c^T x subject to pairwise constraints x_u + x_v <= 1.

Instances are planted with a known unique optimum, enabling strict pass/fail
grading and fast verification.
"""

import random
import subprocess
import time
from typing import List, Tuple, Dict, Any, Iterable
from native_compiler import CppCompiler, CompilationError, ExecutionError
from solver_utils import StreamingInputFile

title = "Integer Linear Programming (C++)"
TIMEOUT_SECONDS = 30
RANDOM_SEED = 40404040


def _iter_pairs(pair_count: int, seed: int) -> Iterable[Tuple[int, int]]:
  labels = list(range(2 * pair_count))
  rng = random.Random(seed ^ 0xA5A5A5A5)
  rng.shuffle(labels)
  for i in range(pair_count):
    yield labels[2 * i], labels[2 * i + 1]


def generate_ilp(pair_count: int, seed: int) -> Tuple[List[int], List[Tuple[int, int]], List[int], int]:
  """Generate planted 0/1 ILP with unique optimum.

  Objective: maximize sum(c_i * x_i)
  Constraints: for each pair (u,v), x_u + x_v <= 1, and x_i in {0,1}

  Since each pair is independent and one side gets strictly larger coefficient,
  the unique optimum is selecting the larger-coefficient variable per pair.
  """
  rng = random.Random(seed)
  num_vars = 2 * pair_count
  c = [0] * num_vars
  expected_x = [0] * num_vars
  pairs = list(_iter_pairs(pair_count, seed))

  optimum_obj = 0
  for u, v in pairs:
    high = rng.randint(8, 20)
    low = rng.randint(0, 6)
    if rng.random() < 0.5:
      c[u] = high
      c[v] = low
      expected_x[u] = 1
    else:
      c[u] = low
      c[v] = high
      expected_x[v] = 1
    optimum_obj += high

  return c, pairs, expected_x, optimum_obj


def _get_case_params(subpass: int) -> Tuple[int, int, int]:
  pair_count = TEST_CASES[subpass]["pairs"]
  if pair_count <= 0:
    raise ValueError(f"Invalid pairs={pair_count} for subpass {subpass}")
  return pair_count, 2 * pair_count, pair_count


TEST_CASES = [
  {
    "pairs": 10,
    "desc": "20 vars, 10 constraints"
  },
  {
    "pairs": 20,
    "desc": "40 vars, 20 constraints"
  },
  {
    "pairs": 40,
    "desc": "80 vars, 40 constraints"
  },
  {
    "pairs": 80,
    "desc": "160 vars, 80 constraints"
  },
  {
    "pairs": 160,
    "desc": "320 vars, 160 constraints"
  },
  {
    "pairs": 320,
    "desc": "640 vars, 320 constraints"
  },
  {
    "pairs": 640,
    "desc": "1.28K vars, 640 constraints"
  },
  {
    "pairs": 1250,
    "desc": "2.5K vars, 1.25K constraints"
  },
  {
    "pairs": 2500,
    "desc": "5K vars, 2.5K constraints"
  },
  {
    "pairs": 5000,
    "desc": "10K vars, 5K constraints"
  },
  {
    "pairs": 10000,
    "desc": "20K vars, 10K constraints"
  },
  {
    "pairs": 20000,
    "desc": "40K vars, 20K constraints"
  },
  {
    "pairs": 50000,
    "desc": "100K vars, 50K constraints"
  },
  {
    "pairs": 100000,
    "desc": "200K vars, 100K constraints"
  },
]

INSTANCE_CACHE: Dict[int, Any] = {}
_INPUT_FILE_CACHE: Dict[int, StreamingInputFile] = {}
STREAMING_THRESHOLD_PAIRS = 20_000
LAST_ILP_VIZ: Dict[Tuple[int, str], dict] = {}


def get_instance(subpass: int):
  if subpass not in INSTANCE_CACHE:
    pair_count, _, _ = _get_case_params(subpass)
    c, pairs, expected_x, optimum_obj = generate_ilp(pair_count, RANDOM_SEED + subpass)
    INSTANCE_CACHE[subpass] = (c, pairs, expected_x, optimum_obj)
  return INSTANCE_CACHE[subpass]


def _should_use_streaming(subpass: int) -> bool:
  pair_count, _, _ = _get_case_params(subpass)
  return pair_count > STREAMING_THRESHOLD_PAIRS


def _get_streaming_input(subpass: int) -> StreamingInputFile:
  if subpass in _INPUT_FILE_CACHE:
    return _INPUT_FILE_CACHE[subpass]

  pair_count, num_vars, num_constraints = _get_case_params(subpass)
  cache_key = f"ilp40_v2|pairs={pair_count}|seed={RANDOM_SEED + subpass}"

  def generator():
    c, pairs, _, _ = get_instance(subpass)
    yield f"{num_vars} {num_constraints}\n"
    yield " ".join(map(str, c)) + "\n"
    for u, v in pairs:
      yield f"{u} {v}\n"

  input_file = StreamingInputFile(cache_key, generator, "test40_ilp")
  _INPUT_FILE_CACHE[subpass] = input_file
  return input_file


def format_input(c: List[int], pairs: List[Tuple[int, int]]) -> str:
  num_vars = len(c)
  num_constraints = len(pairs)

  lines = [f"{num_vars} {num_constraints}"]
  lines.append(" ".join(map(str, c)))
  for u, v in pairs:
    lines.append(f"{u} {v}")
  return "\n".join(lines)


def prepareSubpassPrompt(subPass: int) -> str:
  """Generate the prompt for subpass 0."""
  if subPass != 0:
    raise StopIteration

  return """You are writing C++ code to solve a **0/1 Integer Linear Program**.

Maximize:
  sum(c_i * x_i)

Subject to:
  x_u + x_v <= 1   (for each constraint pair u, v)
  x_i in {0, 1}

**Input format (stdin)**
```
n m
c_0 c_1 ... c_(n-1)
u_0 v_0
u_1 v_1
... (m lines total)
```
- `n` = number of binary variables
- `m` = number of pair constraints

**Output format (stdout)**
```
objective_value
x_0 x_1 ... x_(n-1)
```

**What is checked**
1. Output format is valid.
2. `x_i` are binary and satisfy all constraints.
3. Reported objective matches the assignment.
4. Assignment achieves the true optimum for the instance.

Write complete C++17 code with `main()` that reads stdin and writes stdout.
"""


# List of subpasses to grade the single answer against all difficulty levels


extraGradeAnswerRuns = list(range(len(TEST_CASES)))

structure = {
  "type": "object",
  "properties": {
    "reasoning": {
      "type": "string",
      "description": "Explain your 0/1 ILP approach"
    },
    "cpp_code": {
      "type": "string",
      "description": "Complete C++ code with main() function that handles all scales"
    }
  },
  "required": ["reasoning", "cpp_code"],
  "additionalProperties": False
}


def _parse_output(stdout: str, num_vars: int) -> Tuple[int, List[int]]:
  lines = [line.strip() for line in stdout.strip().splitlines() if line.strip()]
  if not lines:
    raise ValueError("No output")

  first = lines[0].split()
  if len(first) != 1:
    raise ValueError("First line must contain exactly one integer objective")
  reported_obj = int(first[0])

  x: List[int] = []
  for line in lines[1:]:
    x.extend(int(tok) for tok in line.split())

  if len(x) != num_vars:
    raise ValueError(f"Expected {num_vars} variable values, got {len(x)}")
  return reported_obj, x


def verify_solution(c: List[int], pairs: List[Tuple[int, int]], x: List[int],
                    expected_x: List[int], optimum_obj: int,
                    reported_obj: int) -> Tuple[bool, int, str]:
  num_vars = len(c)

  if len(x) != num_vars:
    return False, 0, f"Wrong number of variables: {len(x)} vs {num_vars}"

  for i, val in enumerate(x):
    if val not in (0, 1):
      return False, 0, f"x[{i}] = {val}, expected binary value 0/1"

  for u, v in pairs:
    if x[u] + x[v] > 1:
      return False, 0, f"Constraint violated: x[{u}] + x[{v}] = {x[u] + x[v]} > 1"

  actual_obj = sum(c[i] * x[i] for i in range(num_vars))
  if reported_obj != actual_obj:
    return False, actual_obj, f"Reported objective {reported_obj}, but actual is {actual_obj}"

  if actual_obj != optimum_obj:
    return False, actual_obj, f"Objective {actual_obj}, but optimum is {optimum_obj}"

  if x != expected_x:
    return False, actual_obj, "Assignment differs from planted optimum"

  return True, actual_obj, "Valid optimal assignment"


def _build_ilp_viz(c: List[int], pairs: List[Tuple[int, int]], expected_x: List[int],
                   chosen_x: List[int], valid: bool, msg: str,
                   reported_obj: int, actual_obj: int, optimum_obj: int) -> dict:
  bad_pairs = []
  for u, v in pairs:
    if u < len(chosen_x) and v < len(chosen_x) and chosen_x[u] + chosen_x[v] > 1:
      bad_pairs.append((u, v))

  diff_indices = []
  for i in range(min(len(expected_x), len(chosen_x))):
    if expected_x[i] != chosen_x[i]:
      diff_indices.append(i)

  return {
    "c": c,
    "pairs": pairs,
    "expected_x": expected_x,
    "chosen_x": chosen_x,
    "bad_pairs": bad_pairs,
    "diff_indices": diff_indices,
    "valid": valid,
    "msg": msg,
    "reported_obj": reported_obj,
    "actual_obj": actual_obj,
    "optimum_obj": optimum_obj,
  }


def gradeAnswer(result: dict, subPass: int, aiEngineName: str) -> tuple:
  if not result or "cpp_code" not in result:
    return 0.0, "No C++ code provided"

  case = TEST_CASES[subPass]
  use_streaming = _should_use_streaming(subPass)

  compiler = CppCompiler(aiEngineName)
  if not compiler.find_compiler():
    return 0.0, "No C++ compiler found"

  try:
    exe_path = compiler.compile(result["cpp_code"])
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
      stdout, stderr, exec_time, return_code = compiler.execute(exe_path,
                                                                timeout=TIMEOUT_SECONDS,
                                                                stdin_file=input_file_path)

      if return_code != 0:
        return 0.0, f"Runtime error: {stderr[:200]}"
      proc_stdout = stdout
    else:
      c, pairs, _, _ = get_instance(subPass)
      input_data = format_input(c, pairs)

      start = time.time()
      proc = subprocess.run([str(exe_path)],
                            input=input_data,
                            capture_output=True,
                            text=True,
                            timeout=TIMEOUT_SECONDS)
      exec_time = time.time() - start

      if proc.returncode != 0:
        return 0.0, f"Runtime error: {proc.stderr[:200]}"
      proc_stdout = proc.stdout

    lines = proc_stdout.strip().split('\n')
    if not lines:
      return 0.0, "No output"

    c, pairs, expected_x, optimum_obj = get_instance(subPass)
    try:
      reported_obj, x = _parse_output(proc_stdout, len(c))
    except Exception as e:
      return 0.0, f"[{case['desc']}] Invalid output format: {str(e)[:120]}"

    valid, actual_obj, msg = verify_solution(c, pairs, x, expected_x, optimum_obj, reported_obj)

    if len(c) <= 200:
      LAST_ILP_VIZ[(subPass, aiEngineName)] = _build_ilp_viz(
        c, pairs, expected_x, x, valid, msg, reported_obj, actual_obj, optimum_obj
      )

    if not valid:
      return 0.0, f"[{case['desc']}] FAIL: {msg}"

    return 1.0, f"[{case['desc']}] PASS: objective {actual_obj} (optimal) in {exec_time:.2f}s"

  except subprocess.TimeoutExpired:
    return 0.0, f"[{case['desc']}] FAIL: Timeout"
  except ExecutionError as e:
    return 0.0, f"[{case['desc']}] FAIL: {str(e)[:100]}"
  except Exception as e:
    return 0.0, f"[{case['desc']}] Error: {str(e)[:100]}"


def resultToNiceReport(result: dict, subPass: int, aiEngineName: str) -> str:
  if not result:
    return "<p style='color:red'>No result provided</p>"
  case = TEST_CASES[subPass]
  _, num_vars, num_constraints = _get_case_params(subPass)
  html = f"<h4>Integer Linear Programming - {case['desc']}</h4>"
  html += (
    f"<p style='font-size:12px;color:#475569;margin:6px 0;'>"
    f"Binary variables: {num_vars:,} | Pair constraints: {num_constraints:,} | "
    f"Planted optimum is unique</p>"
  )
  if "reasoning" in result:
    r = result['reasoning'][:400] + ('...' if len(result.get('reasoning', '')) > 400 else '')
    html += f"<p><strong>Approach:</strong> {r.replace('<', '&lt;').replace('>', '&gt;')}</p>"
  if "cpp_code" in result:
    code = result["cpp_code"].replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    html += f"<details><summary>View C++ Code ({len(result['cpp_code'])} chars)</summary><pre>{code}</pre></details>"
  viz = LAST_ILP_VIZ.get((subPass, aiEngineName))
  if viz and len(viz.get("c", [])) <= 200:
    html += _generate_ilp_svg(viz)
  return html


highLevelSummary = """
0/1 Integer Linear Programming optimizes a linear objective under linear constraints
with binary decision variables.

This test uses planted pairwise constraints with known unique optimum for strict
pass/fail grading.
"""


def _generate_ilp_svg(viz: dict) -> str:
  c = viz["c"]
  pairs = viz["pairs"]
  expected_x = viz["expected_x"]
  chosen_x = viz["chosen_x"]
  bad_pairs = set(tuple(sorted(p)) for p in viz["bad_pairs"])
  diff_indices = set(viz["diff_indices"])

  pair_count = len(pairs)
  size = 680
  margin = 24
  row_h = 14
  col_gap = 260
  width = size
  height = margin * 2 + pair_count * row_h + 34

  row_lines = []
  for i, (u, v) in enumerate(pairs[:120]):
    y = margin + i * row_h
    u_x = 40
    v_x = 40 + col_gap
    is_bad = tuple(sorted((u, v))) in bad_pairs
    edge_color = "#ef4444" if is_bad else "#334155"
    edge_w = "1.5" if is_bad else "0.8"
    row_lines.append(
      f"<line x1='{u_x+8}' y1='{y+6}' x2='{v_x-8}' y2='{y+6}' stroke='{edge_color}' stroke-width='{edge_w}' stroke-opacity='0.8' />"
    )

    def node_fill(idx: int) -> str:
      if idx in diff_indices:
        return "#f59e0b"
      if chosen_x[idx] == 1 and expected_x[idx] == 1:
        return "#22c55e"
      if chosen_x[idx] == 1 and expected_x[idx] == 0:
        return "#ef4444"
      if chosen_x[idx] == 0 and expected_x[idx] == 1:
        return "#ef4444"
      return "#475569"

    for x0, idx in ((u_x, u), (v_x, v)):
      row_lines.append(
        f"<circle cx='{x0}' cy='{y+6}' r='5.0' fill='{node_fill(idx)}' stroke='#0f172a' stroke-width='0.8' />"
      )
      row_lines.append(
        f"<text x='{x0+10}' y='{y+9}' font-size='9' fill='#cbd5e1'>v{idx} c={c[idx]} x={chosen_x[idx]}</text>"
      )

  shown_pairs = min(pair_count, 120)
  status = "Valid optimal assignment" if viz.get("valid") else viz.get("msg", "Invalid")
  svg = "\n".join([
    "<div style='margin:12px 0;padding:10px;border:1px solid #1f2937;border-radius:8px;background:#0b1120;'>",
    f"<div style='color:#e2e8f0;font-size:13px;margin-bottom:6px;'><strong>Constraint Pair Visualization</strong> — {status}</div>",
    f"<div style='color:#94a3b8;font-size:12px;margin-bottom:6px;'>"
    f"Pairs: {pair_count} | Reported obj: {viz.get('reported_obj')} | "
    f"Actual obj: {viz.get('actual_obj')} | Optimum: {viz.get('optimum_obj')} | "
    f"Mismatched vars: {len(diff_indices)} | Violated pairs: {len(bad_pairs)}</div>",
    f"<svg width='100%' viewBox='0 0 {width} {height}' style='background:#0b1120;border:1px solid #334155;border-radius:6px;'>",
    *row_lines,
    "</svg>",
    "<div style='color:#64748b;font-size:11px;margin-top:6px;'>"
    "Green nodes match planted optimum (selected). Gray are correctly unselected. "
    "Amber/Red indicate wrong variable decisions. Red pair lines mark violated constraints." +
    (f" Showing first {shown_pairs} of {pair_count} pairs." if pair_count > shown_pairs else "") +
    "</div>",
    "</div>",
  ])
  return svg


def setup():
  """Pre-generate and cache all streaming input files for parallel test execution."""
  print(f"  Pre-generating streaming input files for {len(TEST_CASES)} test cases...")
  for subpass in range(len(TEST_CASES)):
    if _should_use_streaming(subpass):
      streaming_input = _get_streaming_input(subpass)
      input_path = streaming_input.generate()
      size_mb = streaming_input.get_size_bytes() / (1024 * 1024)
      print(f"    Subpass {subpass}: {size_mb:.1f} MB cached")
