"""
Test 9: 1D Cutting Stock Problem (Minimum Cut Solver)

The LLM must write a Python solver for the cutting stock problem:
Given a list of piece lengths needed and a fixed wholesale stock length,
minimize the number of stock pieces purchased.

This is a classic bin-packing/optimization problem.

Subpasses test increasingly complex cutting lists.
Solver times out after 5 minutes.
"""

import random
import time
import hashlib
import tempfile
from collections import Counter
from pathlib import Path
from typing import List, Tuple, Dict

from native_compiler import CSharpCompiler, compile_and_run, describe_this_pc
from solver_utils import GradeCache

title = "1D Cutting Stock - Minimum Waste (C#)"

tags = [
  "csharp",
  "structured response",
  "optimization",
  "packing",
]

TIMEOUT_SECONDS = 300

RANDOM_SEED = 88888

_grade_cache = GradeCache('test9')


def _cache_key_parts(result: dict, subPass: int) -> tuple:
  case = TEST_CASES[subPass]
  code = result.get("csharp_code", "")
  return (
    hashlib.sha256(code.encode('utf-8')).hexdigest()[:16],
    f"stock={case['stock_length']}|cuts={len(case['cuts'])}|seed={RANDOM_SEED + subPass}",
  )


# Test configurations: (stock_length, cuts_list, description)
def generate_cuts(num_cuts: int, stock_length: int, seed: int) -> List[int]:
  """Generate random cut lengths that fit in stock."""
  rng = random.Random(seed)
  cuts = []
  for _ in range(num_cuts):
    cut = rng.randint(stock_length // 10, int(stock_length * 0.51))
    cuts.append(cut)
  return cuts


class _LazyCutsList(list):
  """List that defers generation of large cut arrays until first access."""

  def __init__(self, num_cuts, stock_length, seed):
    super().__init__()
    self._params = (num_cuts, stock_length, seed)
    self._generated = False

  def _ensure(self):
    if not self._generated:
      self._generated = True
      t = time.time()
      self.extend(generate_cuts(*self._params))
      elapsed = time.time() - t
      if elapsed > 0.5:
        print(f"  Generated {self._params[0]} cuts in {elapsed:.1f}s")

  def __len__(self):
    if not self._generated:
      return self._params[0]
    return super().__len__()

  def __iter__(self):
    self._ensure()
    return super().__iter__()

  def __getitem__(self, idx):
    self._ensure()
    return super().__getitem__(idx)


# Pre-defined test cases
TEST_CASES = [
  # Subpass 0: Simple - few cuts
  {
    "stock_length": 100,
    "cuts": [30, 40, 50, 25, 35],
    "description": "5 cuts, stock=100"
  },
  # Subpass 1: Medium - more cuts
  {
    "stock_length": 100,
    "cuts": [20, 30, 40, 15, 25, 35, 45, 10, 50, 30],
    "description": "10 cuts, stock=100"
  },
  # Subpass 2: Larger stock, various sizes
  {
    "stock_length": 200,
    "cuts": [80, 60, 40, 120, 75, 45, 90, 55, 100, 65, 35, 85],
    "description": "12 cuts, stock=200"
  },
  # Subpass 3: Many small cuts
  {
    "stock_length": 100,
    "cuts": generate_cuts(20, 100, RANDOM_SEED),
    "description": "20 cuts, stock=100"
  },
  # Subpass 4: Industrial scale
  {
    "stock_length": 1000,
    "cuts": generate_cuts(30, 1000, RANDOM_SEED + 1),
    "description": "30 cuts, stock=1000"
  },
  # Subpass 5: Large problem
  {
    "stock_length": 500,
    "cuts": generate_cuts(50, 500, RANDOM_SEED + 2),
    "description": "50 cuts, stock=500"
  },
  # Extreme cases
  {
    "stock_length": 1000,
    "cuts": generate_cuts(100, 1000, RANDOM_SEED + 3),
    "description": "100 cuts, stock=1000"
  },
  {
    "stock_length": 1000,
    "cuts": generate_cuts(500, 1000, RANDOM_SEED + 4),
    "description": "500 cuts, stock=1000"
  },
  # Large cases - use lazy generation to avoid slow module import
  {
    "stock_length": 10000,
    "cuts": _LazyCutsList(1000, 10000, RANDOM_SEED + 5),
    "description": "1000 cuts, stock=10000"
  },
  {
    "stock_length": 10000,
    "cuts": _LazyCutsList(5000, 10000, RANDOM_SEED + 6),
    "description": "5000 cuts, stock=10000"
  },
  {
    "stock_length": 100000,
    "cuts": _LazyCutsList(10000, 100000, RANDOM_SEED + 7),
    "description": "10000 cuts, stock=100000"
  },
  {
    "stock_length": 1000,
    "cuts": _LazyCutsList(100000, 1000, RANDOM_SEED + 8),
    "description": "100000 cuts, stock=1000"
  },
  {
    "stock_length": 1000,
    "cuts": _LazyCutsList(500000, 1000, RANDOM_SEED + 9),
    "description": "500000 cuts, stock=1000"
  },
  {
    "stock_length": 1000,
    "cuts": _LazyCutsList(1000000, 1000, RANDOM_SEED + 9),
    "description": "1000000 cuts, stock=1000"
  },
  {
    "stock_length": 1000,
    "cuts": _LazyCutsList(10000000, 1000, RANDOM_SEED + 10),
    "description": "10000000 cuts, stock=1000"
  },
]


def format_cuts_for_prompt(cuts: List[int]) -> str:
  """Format cuts list for prompt."""
  if len(cuts) <= 20:
    return str(cuts)
  else:
    return f"[{', '.join(map(str, cuts[:15]))}, ... ({len(cuts)} total)]"


def prepareSubpassPrompt(subPass: int) -> str:
  """Generate the prompt for subpass 0 that handles all cutting problem sizes."""
  if subPass != 0:
    raise StopIteration

  return f"""You are solving the 1D Cutting Stock Problem in C#.

**Input format (stdin):**
Line 1: N stock_length
Line 2: N cut lengths separated by spaces

**Output format (stdout):**
Line 1: num_stocks
Next num_stocks lines: space-separated cut indices assigned to that stock (0-indexed)

**Example:**
For cuts=[30, 40, 50] and stock_length=100:
Input:
3 100
30 40 50

Output:
2
0 1
2

**Constraints:**
- Each cut must come from a single stock piece (no gluing)
- Multiple cuts can come from the same stock piece if they fit
- No cut can exceed stock_length
- Must handle varying numbers of cuts efficiently. You can use
  multiple threads, you can micro-optimise code, you can write clever algorithms,
  whatever it takes to handle the problem and solve it within 5 minutes even if 
  there are millions of cuts.

**Environment:**
{describe_this_pc()}

**C# Compiler:**
{CSharpCompiler("test_engine").describe()}

Write complete, compilable C# code with a static void Main method.
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
      "Explain your cutting stock algorithm and how it adapts to different problem sizes"
    },
    "csharp_code": {
      "type": "string",
      "description": "Complete C# code with Main method that handles all scales"
    }
  },
  "required": ["reasoning", "csharp_code"],
  "additionalProperties": False
}


def validate_solution(solution: Dict, cuts: List[int], stock_length: int) -> Tuple[bool, str]:
  """Validate cutting stock solution."""
  if not isinstance(solution, dict):
    return False, "Solution must be a dict"

  if "num_stocks" not in solution:
    return False, "Missing 'num_stocks'"

  if "assignments" not in solution:
    return False, "Missing 'assignments'"

  num_stocks = solution["num_stocks"]
  assignments = solution["assignments"]

  if not isinstance(num_stocks, int) or num_stocks < 1:
    return False, f"num_stocks must be positive int, got {num_stocks}"

  if not isinstance(assignments, list):
    return False, "assignments must be a list"

  if len(assignments) != num_stocks:
    return False, f"assignments length {len(assignments)} != num_stocks {num_stocks}"

  # Check all cuts are assigned exactly once
  all_assigned = []
  for i, stock_cuts in enumerate(assignments):
    if not isinstance(stock_cuts, list):
      return False, f"Stock {i} assignments must be a list"

    # Check each stock doesn't exceed capacity
    total = 0
    for cut_idx in stock_cuts:
      if not isinstance(cut_idx, int):
        return False, f"Cut index must be int, got {cut_idx}"
      if cut_idx < 0 or cut_idx >= len(cuts):
        return False, f"Invalid cut index {cut_idx}"
      total += cuts[cut_idx]
      all_assigned.append(cut_idx)

    if total > stock_length:
      return False, f"Stock {i} exceeds capacity: {total} > {stock_length}"

  # Check all cuts assigned exactly once (O(n) via Counter)
  counts = Counter(all_assigned)
  expected = set(range(len(cuts)))
  assigned_set = set(all_assigned)
  if assigned_set != expected or len(all_assigned) != len(cuts):
    missing = expected - assigned_set
    duplicates = {k for k, v in counts.items() if v > 1}
    msg = ""
    if missing:
      missing_sample = sorted(missing)[:10]
      msg += f"Missing {len(missing)} cuts: {missing_sample}{'...' if len(missing) > 10 else ''}. "
    if duplicates:
      dup_sample = sorted(duplicates)[:10]
      msg += f"Duplicate {len(duplicates)} cuts: {dup_sample}{'...' if len(duplicates) > 10 else ''}."
    return False, msg or "Not all cuts assigned correctly"

  return True, ""


def compute_waste(assignments: List[List[int]], cuts: List[int], stock_length: int) -> int:
  """Compute total waste."""
  total_waste = 0
  for stock_cuts in assignments:
    used = sum(cuts[i] for i in stock_cuts)
    total_waste += stock_length - used
  return total_waste


MAX_REPACK_NODES = 1_000_000  # backtracking node limit to prevent hangs


def _min_bins_exact(pieces: List[int], capacity: int, max_bins: int) -> int:
  """Find minimum number of bins needed for pieces via backtracking.
  Returns min bins needed, or max_bins+1 if impossible within max_bins.
  pieces must be sorted descending for best pruning."""

  n = len(pieces)
  if n == 0:
    return 0

  # Quick lower bound: ceiling of total / capacity
  total = sum(pieces)
  lb = (total + capacity - 1) // capacity
  if lb > max_bins:
    return max_bins + 1

  # Pre-compute suffix sums (avoids O(n) sum per recursive call)
  suffix = [0] * (n + 1)
  for i in range(n - 1, -1, -1):
    suffix[i] = suffix[i + 1] + pieces[i]

  best = [max_bins + 1]
  nodes = [0]
  bins = [0] * max_bins  # used space per bin

  def solve(idx):
    if nodes[0] >= MAX_REPACK_NODES:
      return  # bail out — too complex
    nodes[0] += 1

    if idx == n:
      used = sum(1 for b in bins if b > 0)
      if used < best[0]:
        best[0] = used
      return

    p = pieces[idx]

    tried_empty = False
    for j in range(min(best[0], max_bins)):
      if bins[j] + p > capacity:
        continue
      if bins[j] == 0:
        if tried_empty:
          continue  # all empty bins are equivalent
        tried_empty = True
      bins[j] += p
      solve(idx + 1)
      bins[j] -= p
      if best[0] <= lb:
        return  # can't do better than lower bound

  solve(0)
  if nodes[0] >= MAX_REPACK_NODES:
    return max_bins  # inconclusive — assume no saving (no penalty)
  return best[0]


def check_top_waste_repack(assignments: List[List[int]], cuts: List[int],
                           stock_length: int) -> Tuple[int, int]:
  """Pick the 5 stocks with highest waste, strip their cuts, and exhaustively
  repack.  Returns (original_count, optimal_count) for those stocks.
  
  Grading rule applied by caller:
    optimal <= 3  (saved 2+)  -> score 0
    optimal == 4  (saved 1)   -> score capped at 0.5
    optimal == 5  (no saving)  -> no penalty
  """
  N_FOCUS = 5
  # Compute waste per stock
  waste_per_stock = []
  for i, stock_cuts in enumerate(assignments):
    used = sum(cuts[idx] for idx in stock_cuts)
    waste_per_stock.append((stock_length - used, i))

  # Sort by waste descending, pick top 5
  waste_per_stock.sort(reverse=True)
  focus = waste_per_stock[:N_FOCUS]
  focus_indices = [idx for _, idx in focus]

  # Gather all cut lengths from those stocks
  focus_pieces = []
  for si in focus_indices:
    for cut_idx in assignments[si]:
      focus_pieces.append(cuts[cut_idx])

  if not focus_pieces:
    return N_FOCUS, N_FOCUS

  original_count = len(focus_indices)

  # Cap pieces to prevent exponential blowup in exact solver
  if len(focus_pieces) > 20:
    return original_count, original_count  # too many pieces, skip repack check

  # Sort descending for best backtracking pruning
  focus_pieces.sort(reverse=True)

  optimal = _min_bins_exact(focus_pieces, stock_length, original_count)

  return original_count, optimal


def get_baseline_solution(cuts: List[int], stock_length: int) -> int:
  """
    Compute baseline using First Fit Decreasing (FFD).
    Returns number of stocks used.
    """
  sorted_cuts = sorted(enumerate(cuts), key=lambda x: -x[1])
  stocks = []  # List of remaining capacity per stock

  for idx, length in sorted_cuts:
    placed = False
    for i, remaining in enumerate(stocks):
      if remaining >= length:
        stocks[i] -= length
        placed = True
        break

    if not placed:
      stocks.append(stock_length - length)

  return len(stocks)


def format_input(cuts: List[int], stock_length: int) -> str:
  lines = [f"{len(cuts)} {stock_length}"]
  lines.append(" ".join(str(c) for c in cuts))
  return "\n".join(lines)


_INPUT_FILE_CACHE: Dict[int, Path] = {}


def _get_or_create_input_file(cuts, stock_length: int, subPass: int) -> Path:
  """Write input to temp file for large inputs (avoids huge in-memory strings)."""
  if subPass in _INPUT_FILE_CACHE and _INPUT_FILE_CACHE[subPass].exists():
    return _INPUT_FILE_CACHE[subPass]

  path = Path(tempfile.gettempdir()) / f"test9_input_sp{subPass}.txt"
  t = time.time()
  with open(path, 'w') as f:
    f.write(f"{len(cuts)} {stock_length}\n")
    # Write in chunks to avoid creating one massive string
    chunk_size = 100000
    n = len(cuts)
    for start in range(0, n, chunk_size):
      end = min(start + chunk_size, n)
      chunk = " ".join(str(cuts[i]) for i in range(start, end))
      if start > 0:
        f.write(" ")
      f.write(chunk)
    f.write("\n")
  elapsed = time.time() - t
  if elapsed > 0.5:
    sz = path.stat().st_size / 1e6
    print(f"  Input file for subpass {subPass}: {sz:.1f}MB in {elapsed:.1f}s")
  _INPUT_FILE_CACHE[subPass] = path
  return path


def parse_assignments_output(output: str, num_cuts: int) -> tuple:
  text = output.strip()
  if not text:
    return None, "Empty output"

  lines = [l for l in text.splitlines() if l.strip()]
  if not lines:
    return None, "No output lines"

  try:
    num_stocks = int(lines[0])
  except ValueError:
    return None, "First line must be num_stocks integer"

  if len(lines) < 1 + num_stocks:
    return None, f"Expected {num_stocks} assignment lines, got {len(lines) - 1}"

  assignments = []
  for i in range(1, 1 + num_stocks):
    parts = lines[i].split()
    try:
      indices = [int(p) for p in parts]
    except ValueError:
      return None, f"Non-integer index in assignment line {i}"
    assignments.append(indices)

  # Compute waste for the solution dict
  all_assigned = [idx for stock in assignments for idx in stock]
  out = {
    'num_stocks': num_stocks,
    'assignments': assignments,
    'waste': 0,  # will be computed by caller
  }
  return out, None


def execute_solver(code: str,
                   cuts: List[int],
                   stock_length: int,
                   ai_engine_name: str,
                   subPass: int = -1,
                   timeout: int = TIMEOUT_SECONDS) -> tuple:
  """Execute the LLM's solver. Returns (solution, error, exec_time)."""
  num_cuts = len(cuts)

  # Use file-based stdin for large inputs to avoid huge in-memory strings
  if num_cuts >= 5000:
    t = time.time()
    input_file = _get_or_create_input_file(cuts, stock_length, subPass)
    fmt_time = time.time() - t

    t = time.time()
    run = compile_and_run(code, "csharp", ai_engine_name,
                          input_file=input_file, timeout=timeout)
    run_time = time.time() - t
  else:
    t = time.time()
    input_data = format_input(cuts, stock_length)
    fmt_time = time.time() - t

    t = time.time()
    run = compile_and_run(code, "csharp", ai_engine_name,
                          input_data=input_data, timeout=timeout)
    run_time = time.time() - t

  if not run:
    total = fmt_time + run_time
    if total > 0.5:
      print(f"  Solver[{num_cuts}]: format={fmt_time:.1f}s, run={run_time:.1f}s (failed)")
    return None, run.error_message(), run.exec_time

  t = time.time()
  solution, parse_error = parse_assignments_output(run.stdout, num_cuts)
  parse_time = time.time() - t

  total = fmt_time + run_time + parse_time
  if total > 1:
    print(f"  Solver[{num_cuts}]: format={fmt_time:.1f}s, run={run_time:.1f}s, parse={parse_time:.1f}s")

  if parse_error:
    return None, parse_error, run.exec_time

  return solution, None, run.exec_time


lastSolution = None


def gradeAnswer(result: dict, subPass: int, aiEngineName: str) -> tuple:
  """
    Grade the cutting stock solver.
    
    Scoring based on stocks used vs baseline, with a trivial-improvement penalty:
    - 1.0: <= baseline stocks AND no trivially saveable stocks
    - 0.5: could trivially save exactly 1 stock (passable but not optimal)
    - 0.0: could trivially save 2+ stocks (lazy packing)
    - Ratio-based penalties still apply on top.
    """
  grade_t0 = time.time()
  if not result:
    return 0.0, "No result provided"

  if "csharp_code" not in result:
    return 0.0, "No C# code provided"

  cache_parts = _cache_key_parts(result, subPass)
  cached = _grade_cache.get_grade(*cache_parts)
  if cached is not None:
    return cached

  case = TEST_CASES[subPass]
  cuts = case["cuts"]
  stock_length = case["stock_length"]
  description = case["description"]
  code = result["csharp_code"]
  print(f"  [SP{subPass}] Grading {description} (cache miss)")

  # Execute solver
  t = time.time()
  solution, error, exec_time = execute_solver(code, cuts, stock_length, aiEngineName,
                                              subPass=subPass)
  execTime = time.time() - t
  if execTime > 1:
    print(f"  Execution took: {execTime:.1f}s")

  global lastSolution
  lastSolution = solution

  if error:
    grade = (0.0, f"[{description}] {error}")
    _grade_cache.put_grade(grade, *cache_parts)
    return grade

  # Validate solution
  t = time.time()
  is_valid, validation_error = validate_solution(solution, cuts, stock_length)
  validationTime = time.time() - t
  if validationTime > 1:
    print(f"  Validation took: {validationTime:.1f}s")
  if not is_valid:
    grade = (0.0, f"[{description}] Invalid: {validation_error}")
    _grade_cache.put_grade(grade, *cache_parts)
    return grade

  # Compare to baseline
  num_stocks = solution["num_stocks"]

  t = time.time()
  waste = compute_waste(solution["assignments"], cuts, stock_length)
  wasteTime = time.time() - t
  if wasteTime > 1:
    print(f"  Waste calculation took: {wasteTime:.1f}s")

  # ── Repack penalty ──
  # If total waste < 1 stock length, packing is tight enough — no penalty.
  # Otherwise, take the 5 stocks with the most waste, strip their cuts,
  # and exhaustively repack.  If we can fit them in fewer stocks, penalise.
  penalty_note = ""
  if waste < stock_length:
    penalty_note = " (waste < 1 stock, no repack check)"
    score = 1.0
  else:
    t = time.time()
    orig, optimal = check_top_waste_repack(solution["assignments"], cuts, stock_length)
    repackCheckTime = time.time() - t
    if repackCheckTime > 1:
      print(f"  Repack check took: {repackCheckTime:.1f}s")
    saved = orig - optimal
    if saved >= 2:
      score = 0.0
      penalty_note = f" TERRIBLE PACKING. Top-5 waste stocks repacked {orig}→{optimal} (saved {saved}) → 0"
    elif saved == 1:
      score = 0.5
      penalty_note = f" SUBOPTIMAL PACKING. Top-5 waste stocks repacked {orig}→{optimal} (saved 1) → score of 0.5"
    else:
      penalty_note = f" Decent packing. Worst 5 packed stocks couldn't be improved on.)"
      score = 1.0

  explanation = (f"[{description}] Stocks used: {num_stocks},  "
                 f"Waste: {waste}, Time: {exec_time:.1f}s - {penalty_note}")

  grade = (score, explanation)
  _grade_cache.put_grade(grade, *cache_parts)
  grade_total = time.time() - grade_t0
  if grade_total > 1:
    print(f"  [SP{subPass}] Total gradeAnswer: {grade_total:.1f}s")
  return grade


def resultToNiceReport(result: dict, subPass: int, aiEngineName: str) -> str:
  """Generate HTML report."""
  if not result:
    return "<p style='color:red'>No result provided</p>"

  cache_parts = _cache_key_parts(result, subPass)
  cached = _grade_cache.get_report(*cache_parts)
  if cached is not None:
    return cached

  case = TEST_CASES[subPass]

  html = f"<h4>Cutting Stock - {case['description']}</h4>"

  if subPass == 0:
    if "reasoning" in result:
      reasoning = result['reasoning'][:500] + ('...'
                                               if len(result.get('reasoning', '')) > 500 else '')
      html += f"<p><strong>Algorithm:</strong> {reasoning}</p>"

    if "csharp_code" in result:
      code = result["csharp_code"]
      code_escaped = code.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
      html += f"<details><summary>View Code ({len(code)} chars)</summary><pre>{code_escaped}</pre></details>"

  if lastSolution and subPass < 7:
    html += generate_cutting_stock_svg(case["cuts"], case["stock_length"], lastSolution,
                                       case["description"])
  elif lastSolution and subPass >= 7:
    html += f"<p>Solution too large to render SVG ({len(case['cuts'])} cuts)</p>"

  _grade_cache.put_report(html, *cache_parts)
  return html


def generate_cutting_stock_svg(cuts: List[int], stock_length: int, solution: Dict,
                               description: str) -> str:
  """Generate SVG visualization of cutting stock assignments."""
  if not solution or not isinstance(solution, dict):
    return "<p>No solution to visualize</p>"

  assignments = solution.get("assignments")
  if not isinstance(assignments, list) or not assignments:
    return "<p>No assignments to visualize</p>"

  num_stocks = len(assignments)

  # Keep SVG manageable
  if num_stocks > 60 or len(cuts) > 200:
    return f"<p>Solution too large to render SVG ({num_stocks} stocks, {len(cuts)} cuts)</p>"

  svg_width = 800
  left_label_w = 110
  right_label_w = 90
  bar_h = 16
  gap = 10
  pad = 12
  header_h = 30

  height = header_h + pad * 2 + num_stocks * (bar_h + gap) + 30

  # Colors for cuts
  palette = [
    "#60a5fa", "#34d399", "#fbbf24", "#f472b6", "#a78bfa", "#fb7185", "#22c55e", "#38bdf8",
    "#f97316", "#e879f9"
  ]

  total_waste = 0
  bars_svg = []

  for i, stock_cuts in enumerate(assignments):
    if not isinstance(stock_cuts, list):
      continue

    used = sum(cuts[idx] for idx in stock_cuts)
    waste = max(0, stock_length - used)
    total_waste += waste

    y = header_h + pad + i * (bar_h + gap)
    x0 = left_label_w
    bar_w = svg_width - left_label_w - right_label_w

    # Outline for full stock
    bars_svg.append(
      f'<rect x="{x0}" y="{y}" width="{bar_w}" height="{bar_h}" fill="#ffffff" stroke="#333" stroke-width="1"/>'
    )

    # Label
    bars_svg.append(
      f'<text x="{x0 - 6}" y="{y + bar_h - 3}" text-anchor="end" font-size="11" fill="#333">Stock {i + 1}</text>'
    )

    cursor = 0
    # Render cut segments
    for j, cut_idx in enumerate(stock_cuts):
      length = cuts[cut_idx]
      seg_w = (length / stock_length) * bar_w if stock_length else 0
      color = palette[(cut_idx + j) % len(palette)]
      bars_svg.append(
        f'<rect x="{x0 + (cursor / stock_length) * bar_w}" y="{y}" width="{seg_w}" height="{bar_h}" fill="{color}" opacity="0.85"/>'
      )

      # Draw cut boundary line at end of segment
      cursor += length
      end_x = x0 + (cursor / stock_length) * bar_w
      bars_svg.append(
        f'<line x1="{end_x}" y1="{y}" x2="{end_x}" y2="{y + bar_h}" stroke="#111" stroke-width="1" opacity="0.6"/>'
      )

      # Label length if it fits
      if seg_w >= 22:
        text_x = x0 + ((cursor - length / 2) / stock_length) * bar_w
        bars_svg.append(
          f'<text x="{text_x}" y="{y + bar_h - 4}" text-anchor="middle" font-size="10" fill="#111">{length}</text>'
        )

    # Waste segment (highlight)
    if waste > 0:
      waste_x = x0 + (used / stock_length) * bar_w if stock_length else x0
      waste_w = (waste / stock_length) * bar_w if stock_length else 0
      bars_svg.append(
        f'<rect x="{waste_x}" y="{y}" width="{waste_w}" height="{bar_h}" fill="#fee2e2" stroke="#ef4444" stroke-width="1" opacity="0.95"/>'
      )
      if waste_w >= 22:
        bars_svg.append(
          f'<text x="{waste_x + waste_w / 2}" y="{y + bar_h - 4}" text-anchor="middle" font-size="10" fill="#991b1b">waste {waste}</text>'
        )

    # Right-side summary
    bars_svg.append(
      f'<text x="{x0 + bar_w + 6}" y="{y + bar_h - 3}" text-anchor="start" font-size="11" fill="#333">{used}/{stock_length}</text>'
    )

  svg_html = f'''
  <div style="margin: 10px 0; width: 100%">
    <h5>Cut Plan Visualization ({description})</h5>
    <svg width="100%" style="border: 1px solid #ccc; background: white;" viewBox="0 0 {svg_width} {height}">
      <text x="{left_label_w}" y="18" text-anchor="start" font-size="12" fill="#333">Stock length = {stock_length}</text>
      <text x="{svg_width - right_label_w}" y="18" text-anchor="start" font-size="12" fill="#333">used/stock</text>
      {''.join(bars_svg)}
      <text x="{left_label_w}" y="{height - 10}" text-anchor="start" font-size="12" fill="#333">Total waste: {total_waste}</text>
    </svg>
    <p style="font-size: 12px; color: #666;">
      Colored segments are cuts (labels show lengths when there is room). 
      <span style="color: #ef4444; font-weight: bold;">Red</span> segment is waste.
    </p>
  </div>'''

  return svg_html


highLevelSummary = """
<p>Given fixed-length stock material and a shopping list of smaller pieces to cut,
use as few stock pieces as possible. Every requested piece must be cut from
somewhere, but any leftover material is wasted &mdash; so packing the cuts
tightly matters.</p>
<p>This is essentially a bin-packing problem and is NP-hard. Subpasses scale from
a handful of cuts up to millions, forcing the AI to write code that is both
correct and fast. The baseline uses a simple first-fit-decreasing heuristic.</p>
"""
