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
from typing import List, Tuple, Dict

from native_compiler import CSharpCompiler, compile_and_run, describe_this_pc

title = "1D Cutting Stock - Minimum Waste (C#)"

# Timeout in seconds (30 seconds)
TIMEOUT_SECONDS = 30

# Seed for reproducibility
RANDOM_SEED = 88888


# Test configurations: (stock_length, cuts_list, description)
def generate_cuts(num_cuts: int, stock_length: int, seed: int) -> List[int]:
  """Generate random cut lengths that fit in stock."""
  rng = random.Random(seed)
  cuts = []
  for _ in range(num_cuts):
    cut = rng.randint(stock_length // 10, int(stock_length * 0.51))
    cuts.append(cut)
  return cuts


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
  {
    "stock_length": 10000,
    "cuts": generate_cuts(1000, 10000, RANDOM_SEED + 5),
    "description": "1000 cuts, stock=10000"
  },
  {
    "stock_length": 10000,
    "cuts": generate_cuts(5000, 10000, RANDOM_SEED + 6),
    "description": "5000 cuts, stock=10000"
  },
  {
    "stock_length": 100000,
    "cuts": generate_cuts(10000, 100000, RANDOM_SEED + 7),
    "description": "10000 cuts, stock=100000"
  },
  {
    "stock_length": 1000,
    "cuts": generate_cuts(100000, 1000, RANDOM_SEED + 8),
    "description": "100000 cuts, stock=1000"
  },
  {
    "stock_length": 1000,
    "cuts": generate_cuts(500000, 1000, RANDOM_SEED + 9),
    "description": "500000 cuts, stock=1000"
  },
  {
    "stock_length": 1000,
    "cuts": generate_cuts(1000000, 1000, RANDOM_SEED + 9),
    "description": "1000000 cuts, stock=1000"
  },
  {
    "stock_length": 1000,
    "cuts": generate_cuts(10000000, 1000, RANDOM_SEED + 10),
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

You must write a C# solver that can handle ANY problem size from trivial to ludicrous scale:
- **Trivial**: 3-8 cuts, short stock lengths
- **Medium**: 10-25 cuts, moderate stock lengths
- **Large**: 50-100 cuts, long stock lengths
- **Extreme**: 200-500 cuts, very long stock lengths
- **Ludicrous**: 10000+ cuts, very long stock lengths

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
- Must handle varying numbers of cuts efficiently

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

  # Check all cuts assigned exactly once
  if sorted(all_assigned) != list(range(len(cuts))):
    missing = set(range(len(cuts))) - set(all_assigned)
    duplicate = [x for x in all_assigned if all_assigned.count(x) > 1]
    msg = ""
    if missing:
      msg += f"Missing cuts: {missing}. "
    if duplicate:
      msg += f"Duplicate cuts: {set(duplicate)}."
    return False, msg or "Not all cuts assigned correctly"

  return True, ""


def compute_waste(assignments: List[List[int]], cuts: List[int], stock_length: int) -> int:
  """Compute total waste."""
  total_waste = 0
  for stock_cuts in assignments:
    used = sum(cuts[i] for i in stock_cuts)
    total_waste += stock_length - used
  return total_waste


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

  best = [max_bins + 1]
  bins = [0] * max_bins  # used space per bin

  def solve(idx):
    if idx == n:
      # Count non-empty bins
      used = sum(1 for b in bins if b > 0)
      if used < best[0]:
        best[0] = used
      return

    p = pieces[idx]
    # Pruning: remaining items need at least ceil(remaining_total / capacity) more bins
    remaining = sum(pieces[idx:])
    free = sum(capacity - b for b in bins[:best[0] - 1]
               if b > 0) + capacity * (best[0] - 1 - sum(1 for b in bins if b > 0))

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
                   timeout: int = TIMEOUT_SECONDS) -> tuple:
  """Execute the LLM's solver. Returns (solution, error, exec_time)."""
  input_data = format_input(cuts, stock_length)
  run = compile_and_run(code, "csharp", ai_engine_name, input_data=input_data, timeout=timeout)

  if not run:
    return None, run.error_message(), run.exec_time

  solution, parse_error = parse_assignments_output(run.stdout, len(cuts))
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
  if not result:
    return 0.0, "No result provided"

  if "csharp_code" not in result:
    return 0.0, "No C# code provided"

  case = TEST_CASES[subPass]
  cuts = case["cuts"]
  stock_length = case["stock_length"]
  description = case["description"]
  code = result["csharp_code"]

  # Execute solver
  solution, error, exec_time = execute_solver(code, cuts, stock_length, aiEngineName)

  global lastSolution
  lastSolution = solution

  if error:
    return 0.0, f"[{description}] {error}"

  # Validate solution
  is_valid, validation_error = validate_solution(solution, cuts, stock_length)
  if not is_valid:
    return 0.0, f"[{description}] Invalid: {validation_error}"

  # Compare to baseline
  num_stocks = solution["num_stocks"]
  baseline_stocks = get_baseline_solution(cuts, stock_length)
  waste = compute_waste(solution["assignments"], cuts, stock_length)

  ratio = num_stocks / baseline_stocks if baseline_stocks > 0 else float('inf')

  if ratio <= 1.0:
    score = 1.0
    quality = "excellent (≤ baseline)"
  elif ratio <= 1.1:
    score = 0.85
    quality = "good (≤ 1.1x baseline)"
  elif ratio <= 1.25:
    score = 0.5
    quality = "acceptable (≤ 1.25x baseline)"
  else:
    score = 0.0
    quality = f"valid but crazily inefficient ({ratio:.2f}x baseline)"

  # ── Repack penalty ──
  # If total waste < 1 stock length, packing is tight enough — no penalty.
  # Otherwise, take the 5 stocks with the most waste, strip their cuts,
  # and exhaustively repack.  If we can fit them in fewer stocks, penalise.
  penalty_note = ""
  if waste < stock_length:
    penalty_note = " (waste < 1 stock, no repack check)"
  elif num_stocks >= 5 and num_stocks <= 500 and len(cuts) <= 10000:
    orig, optimal = check_top_waste_repack(solution["assignments"], cuts, stock_length)
    saved = orig - optimal
    if saved >= 2:
      score = 0.0
      penalty_note = f" REPACK PENALTY: top-5 waste stocks repacked {orig}→{optimal} (saved {saved}) → 0"
    elif saved == 1:
      score = min(score, 0.5)
      penalty_note = f" REPACK PENALTY: top-5 waste stocks repacked {orig}→{optimal} (saved 1) → capped 0.5"
    else:
      penalty_note = f" (repack check: top-5 waste stocks already optimal at {optimal})"

  explanation = (f"[{description}] Stocks used: {num_stocks}, Baseline: {baseline_stocks}, "
                 f"Waste: {waste}, Time: {exec_time:.1f}s - {quality}.{penalty_note}")

  return score, explanation


def resultToNiceReport(result: dict, subPass: int, aiEngineName: str) -> str:
  """Generate HTML report."""
  if not result:
    return "<p style='color:red'>No result provided</p>"

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
The 1D Cutting Stock Problem is a classic optimization problem.

**Problem:** Given stock pieces of fixed length and a list of cuts needed,
minimize the number of stock pieces purchased.

**This is equivalent to bin packing:** Fit items (cuts) into bins (stocks).

**Algorithms:**
- **First Fit Decreasing (FFD)**: Sort by size, place in first bin that fits
- **Best Fit Decreasing (BFD)**: Place in bin with least remaining space
- **Branch and Bound**: Optimal but exponential time
- **Column Generation**: For very large instances

**Complexity:** NP-hard in general, but good heuristics exist.

The baseline uses FFD which typically achieves ~11/9 × OPT + 6/9.
"""
