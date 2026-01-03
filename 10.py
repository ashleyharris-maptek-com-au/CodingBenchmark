"""
Test 10: 2D Cutting Stock Problem (Rectangle Packing)

The LLM must write a Python solver for the 2D cutting stock problem:
Given a board size (width x height) and a list of rectangular pieces needed,
minimize the number of boards purchased.

This is a 2D bin packing problem - significantly harder than 1D.

Subpasses test increasingly complex cutting lists.
Solver times out after 5 minutes.
"""

import random
import subprocess
import sys
import tempfile
import os
import time
from typing import List, Tuple, Dict

title = "2D Cutting Stock - Rectangle Packing"

# Timeout in seconds (30 seconds)
TIMEOUT_SECONDS = 30

# Seed for reproducibility
RANDOM_SEED = 77777


def generate_rectangles(num_rects: int, board_w: int, board_h: int,
                        seed: int) -> List[Tuple[int, int]]:
  """Generate random rectangles that fit on board."""
  rng = random.Random(seed)
  rects = []
  for _ in range(num_rects):
    # Generate rectangles between 10% and 60% of board dimensions
    w = rng.randint(board_w // 10, int(board_w * 0.6))
    h = rng.randint(board_h // 10, int(board_h * 0.6))
    rects.append((w, h))
  return rects


# Pre-defined test cases
TEST_CASES = [
  # Subpass 0: Simple - few rectangles
  {
    "board_size": (100, 100),
    "rectangles": [(30, 40), (50, 30), (40, 50), (20, 60)],
    "description": "4 rectangles, 100x100 board"
  },
  # Subpass 1: More rectangles
  {
    "board_size": (100, 100),
    "rectangles": [(30, 30), (40, 20), (25, 45), (35, 35), (20, 50), (45, 25), (30, 40), (50, 20)],
    "description": "8 rectangles, 100x100 board"
  },
  # Subpass 2: Larger board
  {
    "board_size": (200, 150),
    "rectangles": [(60, 40), (80, 50), (40, 70), (50, 60), (70, 30), (45, 55), (55, 45), (35, 65),
                   (65, 35), (50, 50)],
    "description":
    "10 rectangles, 200x150 board"
  },
  # Subpass 3: Many small rectangles
  {
    "board_size": (100, 100),
    "rectangles": generate_rectangles(15, 100, 100, RANDOM_SEED),
    "description": "15 rectangles, 100x100 board"
  },
  # Subpass 4: Industrial scale
  {
    "board_size": (200, 200),
    "rectangles": generate_rectangles(20, 200, 200, RANDOM_SEED + 1),
    "description": "20 rectangles, 200x200 board"
  },
  # Subpass 5: Complex problem
  {
    "board_size": (150, 150),
    "rectangles": generate_rectangles(25, 150, 150, RANDOM_SEED + 2),
    "description": "25 rectangles, 150x150 board"
  },
  # Extreme cases
  {
    "board_size": (500, 500),
    "rectangles": generate_rectangles(50, 500, 500, RANDOM_SEED + 3),
    "description": "50 rectangles, 500x500 board"
  },
  {
    "board_size": (1000, 1000),
    "rectangles": generate_rectangles(100, 1000, 1000, RANDOM_SEED + 4),
    "description": "100 rectangles, 1000x1000 board"
  },
  {
    "board_size": (2000, 2000),
    "rectangles": generate_rectangles(500, 2000, 2000, RANDOM_SEED + 5),
    "description": "500 rectangles, 2000x2000 board"
  },
  {
    "board_size": (5000, 5000),
    "rectangles": generate_rectangles(1000, 5000, 5000, RANDOM_SEED + 6),
    "description": "1000 rectangles, 5000x5000 board"
  },
  {
    "board_size": (10000, 10000),
    "rectangles": generate_rectangles(5000, 10000, 10000, RANDOM_SEED + 7),
    "description": "5000 rectangles, 10000x10000 board"
  },
  {
    "board_size": (10000, 10000),
    "rectangles": generate_rectangles(10000, 10000, 10000, RANDOM_SEED + 7),
    "description": "10000 rectangles, 10000x10000 board"
  },
  {
    "board_size": (10000, 10000),
    "rectangles": generate_rectangles(50000, 10000, 10000, RANDOM_SEED + 7),
    "description": "50000 rectangles, 10000x10000 board"
  },
]


def format_rectangles_for_prompt(rects: List[Tuple[int, int]]) -> str:
  """Format rectangles list for prompt."""
  if len(rects) <= 15:
    return str(rects)
  else:
    return f"[{', '.join(str(r) for r in rects[:10])}, ... ({len(rects)} total)]"


def prepareSubpassPrompt(subPass: int) -> str:
  """Generate the prompt for subpass 0 that handles all problem sizes."""
  if subPass != 0:
    raise StopIteration

  return f"""You are solving the 2D Cutting Stock Problem (Rectangle Bin Packing).

You must write a Python solver that can handle ANY problem size from trivial to ludicrous scale:
- **Trivial**: 4-8 rectangles, 100x100-200x150 boards (small problems, exact methods feasible)
- **Ludicrous**: 500-5000 rectangles, 2000x2000-10000x10000 boards (requires very fast heuristics)

**The Challenge:**
Your `solve_2d_cutting(rectangles, board_width, board_height)` function will be tested with problems 
ranging from 4 rectangles to 5000 rectangles on boards up to 10000x10000. The same function must work 
efficiently across all scales.

**Input:**
- `rectangles`: List of (width, height) tuples for pieces to cut
- `board_width`, `board_height`: Dimensions of wholesale boards

**Output:**
- Dictionary with:
  - `"num_boards"`: Number of boards used (int)
  - `"placements"`: List of placements, one per rectangle
    - Each placement: (board_index, x, y, rotated)
    - board_index: which board (0-indexed)
    - x, y: position of rectangle's bottom-left corner on that board
    - rotated: True if rectangle was rotated 90°

**Critical Requirements:**
1. **Scalability**: Your algorithm must adapt based on problem size and board dimensions
2. **Performance**: Must complete within 30 seconds even for 5000 rectangles
3. **Quality**: Minimize number of boards used while producing valid placements

**Constraints:**
- Use only Python standard library or numpy
- Rectangles may be rotated 90° if it helps
- No overlapping allowed
- Rectangles must not exceed board boundaries

Write complete, runnable Python code with the solve_2d_cutting function.
"""


# List of subpasses to grade the single answer against all difficulty levels
extraGradeAnswerRuns = list(range(len(TEST_CASES)))

structure = {
  "type": "object",
  "properties": {
    "reasoning": {
      "type": "string",
      "description":
      "Explain your 2D packing algorithm and how it adapts to different problem sizes"
    },
    "python_code": {
      "type":
      "string",
      "description":
      "Complete Python code with solve_2d_cutting(rectangles, board_width, board_height) function that handles all scales"
    }
  },
  "required": ["reasoning", "python_code"],
  "additionalProperties": False
}


def validate_solution(solution: Dict, rects: List[Tuple[int, int]], board_w: int,
                      board_h: int) -> Tuple[bool, str]:
  """Validate 2D cutting solution."""
  if not isinstance(solution, dict):
    return False, "Solution must be a dict"

  if "num_boards" not in solution:
    return False, "Missing 'num_boards'"

  if "placements" not in solution:
    return False, "Missing 'placements'"

  num_boards = solution["num_boards"]
  placements = solution["placements"]

  if not isinstance(num_boards, int) or num_boards < 1:
    return False, f"num_boards must be positive int, got {num_boards}"

  if not isinstance(placements, list):
    return False, "placements must be a list"

  if len(placements) != len(rects):
    return False, f"placements count {len(placements)} != rectangles count {len(rects)}"

  # Track occupied areas per board
  boards = [[] for _ in range(num_boards)]  # List of placed rectangles per board

  for i, placement in enumerate(placements):
    if not isinstance(placement, (list, tuple)) or len(placement) != 4:
      return False, f"Placement {i} must be (board_idx, x, y, rotated)"

    board_idx, x, y, rotated = placement

    if not isinstance(board_idx, int) or board_idx < 0 or board_idx >= num_boards:
      return False, f"Placement {i}: invalid board_idx {board_idx}"

    # Get rectangle dimensions (possibly rotated)
    orig_w, orig_h = rects[i]
    if rotated:
      w, h = orig_h, orig_w
    else:
      w, h = orig_w, orig_h

    # Check bounds
    if x < 0 or y < 0:
      return False, f"Placement {i}: negative position ({x}, {y})"

    if x + w > board_w or y + h > board_h:
      return False, f"Placement {i}: exceeds board bounds ({x}+{w}>{board_w} or {y}+{h}>{board_h})"

    # Check overlap with other rectangles on same board
    for j, (ox, oy, ow, oh) in boards[board_idx]:
      # Check if rectangles overlap
      if not (x + w <= ox or ox + ow <= x or y + h <= oy or oy + oh <= y):
        return False, f"Placement {i} overlaps with placement {j} on board {board_idx}"

    boards[board_idx].append((i, (x, y, w, h)))

  return True, ""


def compute_waste(placements: List, rects: List[Tuple[int, int]], board_w: int,
                  board_h: int) -> int:
  """Compute total waste area."""
  num_boards = max(p[0] for p in placements) + 1 if placements else 0
  total_board_area = num_boards * board_w * board_h
  total_rect_area = sum(w * h for w, h in rects)
  return total_board_area - total_rect_area


def get_baseline_solution(rects: List[Tuple[int, int]], board_w: int, board_h: int) -> int:
  """
    Compute baseline using Shelf Next Fit Decreasing Height (SNFDH).
    Returns number of boards used.
    """
  # Sort rectangles by height (descending)
  sorted_rects = sorted(enumerate(rects), key=lambda x: -max(x[1]))

  boards = []  # Each board: list of shelves, each shelf: (y, height, x_used)

  if len(rects) > 1000:
    # For large problems, use a simpler heuristic
    return len(rects) // 10 + 1  # Rough estimate

  for idx, (w, h) in sorted_rects:
    # Try both orientations
    placed = False

    for rotated in [False, True]:
      if rotated:
        rw, rh = h, w
      else:
        rw, rh = w, h

      if rw > board_w or rh > board_h:
        continue

      # Try to fit in existing board
      for board in boards:
        # Try existing shelves
        for shelf in board:
          shelf_y, shelf_h, x_used = shelf
          if rh <= shelf_h and x_used + rw <= board_w:
            shelf[2] = x_used + rw
            placed = True
            break

        if placed:
          break

        # Try new shelf on this board
        total_shelf_height = sum(s[1] for s in board)
        if total_shelf_height + rh <= board_h:
          board.append([total_shelf_height, rh, rw])
          placed = True
          break

      if placed:
        break

    if not placed:
      # New board
      boards.append([[0, max(h, w) if w > board_w else h, min(w, h) if w > board_w else w]])

  return len(boards)


def execute_solver(code: str,
                   rects: List[Tuple[int, int]],
                   board_w: int,
                   board_h: int,
                   timeout: int = TIMEOUT_SECONDS) -> tuple:
  """Execute the LLM's solver. Returns (solution, error, exec_time)."""
  from solver_utils import execute_solver_with_data

  data_dict = {
    'rectangles': rects,
    'board_width': board_w,
    'board_height': board_h,
  }

  result, error, exec_time = execute_solver_with_data(code, data_dict, 'solve_2d_cutting', timeout)

  if error:
    return None, error, exec_time

  if not isinstance(result, dict):
    return None, f"Invalid result type: expected dict, got {type(result).__name__}", exec_time

  # Normalize to ensure JSON-serializable primitives (e.g., numpy int -> int)
  try:
    placements = result.get('placements')
    if not isinstance(placements, list):
      return None, "Invalid result: missing or non-list 'placements'", exec_time

    normalized_placements = []
    for p in placements:
      if isinstance(p, (list, tuple)) and len(p) == 4:
        board_idx, x, y, rotated = p
      elif isinstance(p, dict):
        board_idx = p.get('board_idx', p.get('board_index', p.get('board', 0)))
        x = p.get('x', 0)
        y = p.get('y', 0)
        rotated = p.get('rotated', p.get('rotation', False))
      else:
        return None, "Invalid placement: expected (board_idx, x, y, rotated)", exec_time

      normalized_placements.append([
        int(board_idx),
        float(x),
        float(y),
        bool(rotated),
      ])

    out = {
      'num_boards': int(result.get('num_boards')),
      'placements': normalized_placements,
    }
  except Exception as e:
    return None, f"Invalid result format: {e}", exec_time

  return out, None, exec_time


lastSolution = None


def gradeAnswer(result: dict, subPass: int, aiEngineName: str) -> tuple:
  """
    Grade the 2D cutting stock solver.
    
    Scoring based on boards used vs baseline:
    - 1.0: <= baseline boards
    - 0.85: <= 1.15x baseline
    - 0.7: <= 1.5x baseline
    - 0.0: Inefficient, invalid, or error
    """
  if not result:
    return 0.0, "No result provided"

  if "python_code" not in result:
    return 0.0, "No Python code provided"

  case = TEST_CASES[subPass]
  rects = case["rectangles"]
  board_w, board_h = case["board_size"]
  description = case["description"]
  code = result["python_code"]

  # Execute solver
  solution, error, exec_time = execute_solver(code, rects, board_w, board_h)

  global lastSolution
  lastSolution = solution

  if error:
    return 0.0, f"[{description}] {error}"

  # Validate solution
  is_valid, validation_error = validate_solution(solution, rects, board_w, board_h)
  if not is_valid:
    return 0.0, f"[{description}] Invalid: {validation_error}"

  # Compare to baseline
  num_boards = solution["num_boards"]
  baseline_boards = get_baseline_solution(rects, board_w, board_h)
  waste = compute_waste(solution["placements"], rects, board_w, board_h)

  ratio = num_boards / baseline_boards if baseline_boards > 0 else float('inf')

  if ratio <= 1.0:
    score = 1.0
    quality = "excellent (≤ baseline)"
  elif ratio <= 1.15:
    score = 0.85
    quality = "good (≤ 1.15x baseline)"
  elif ratio <= 1.5:
    score = 0.7
    quality = "acceptable (≤ 1.5x baseline)"
  else:
    score = 0.0
    quality = f"inefficient ({ratio:.2f}x baseline)"

  explanation = (f"[{description}] Boards: {num_boards}, Baseline: {baseline_boards}, "
                 f"Waste: {waste}, Time: {exec_time:.1f}s - {quality}")

  return score, explanation


def resultToNiceReport(result: dict, subPass: int, aiEngineName: str) -> str:
  """Generate HTML report."""
  if not result:
    return "<p style='color:red'>No result provided</p>"

  case = TEST_CASES[subPass]

  html = f"<h4>2D Cutting Stock - {case['description']}</h4>"

  if subPass == 0:

    if "reasoning" in result:
      reasoning = result['reasoning'][:500] + ('...'
                                               if len(result.get('reasoning', '')) > 500 else '')
      html += f"<p><strong>Algorithm:</strong> {reasoning}</p>"

    if "python_code" in result:
      code = result["python_code"]
      code_escaped = code.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
      html += f"<details><summary>View Code ({len(code)} chars)</summary><pre>{code_escaped}</pre></details>"

  if lastSolution and isinstance(lastSolution, dict):
    rects = case["rectangles"]
    board_w, board_h = case["board_size"]
    if len(rects) <= 200 and int(lastSolution.get('num_boards', 0)) <= 12:
      html += generate_2d_cutting_stock_svg(rects, board_w, board_h, lastSolution,
                                            case["description"])
    else:
      html += f"<p>Solution too large to render SVG ({len(rects)} rectangles, {lastSolution.get('num_boards')} boards)</p>"

  return html


def generate_2d_cutting_stock_svg(rects: List[Tuple[int, int]], board_w: int, board_h: int,
                                  solution: Dict, description: str) -> str:
  if not solution or not isinstance(solution, dict):
    return "<p>No solution to visualize</p>"

  placements = solution.get('placements')
  if not isinstance(placements, list) or not placements:
    return "<p>No placements to visualize</p>"

  try:
    num_boards = int(solution.get('num_boards', 0))
  except Exception:
    num_boards = 0

  if num_boards <= 0:
    return "<p>No boards to visualize</p>"

  per_board = [[] for _ in range(num_boards)]
  for i, p in enumerate(placements):
    if not isinstance(p, (list, tuple)) or len(p) != 4:
      continue
    b, x, y, rotated = int(p[0]), float(p[1]), float(p[2]), bool(p[3])
    if b < 0 or b >= num_boards:
      continue
    w, h = rects[i]
    if rotated:
      w, h = h, w
    per_board[b].append((i, x, y, float(w), float(h)))

  def color_for(i: int) -> str:
    r = (i * 97) % 200 + 30
    g = (i * 57) % 200 + 30
    b = (i * 17) % 200 + 30
    return f"#{r:02x}{g:02x}{b:02x}"

  show_labels = len(rects) <= 30
  html = f"<h5>Layout</h5><p>{description}</p>"
  for board_idx, items in enumerate(per_board):
    html += f"<div style='margin:10px 0'>"
    html += f"<div><strong>Board {board_idx + 1}/{num_boards}</strong></div>"
    html += (
      f"<svg width='420' style='max-width:100%; height:auto; border:1px solid #ddd; background:#fafafa' "
      f"viewBox='0 0 {board_w} {board_h}' xmlns='http://www.w3.org/2000/svg'>")
    html += f"<rect x='0' y='0' width='{board_w}' height='{board_h}' fill='white' stroke='black' stroke-width='{max(board_w, board_h) * 0.002:.3f}'/>"

    for i, x, y, w, h in items:
      y_svg = board_h - y - h
      fill = color_for(i)
      html += (
        f"<rect x='{x:.3f}' y='{y_svg:.3f}' width='{w:.3f}' height='{h:.3f}' "
        f"fill='{fill}' fill-opacity='0.6' stroke='{fill}' stroke-opacity='0.9' stroke-width='{max(board_w, board_h) * 0.0015:.3f}'/>"
      )
      if show_labels:
        tx = x + w / 2
        ty = y_svg + h / 2
        html += (f"<text x='{tx:.3f}' y='{ty:.3f}' text-anchor='middle' dominant-baseline='middle' "
                 f"font-size='{max(board_w, board_h) * 0.03:.3f}' fill='black'>{i}</text>")

    html += "</svg></div>"

  return html


highLevelSummary = """
The 2D Cutting Stock Problem is a harder variant of bin packing.

**Problem:** Pack rectangles into minimum number of fixed-size boards.

**Key considerations:**
- Rectangles may be rotated 90°
- No overlapping allowed
- Guillotine vs non-guillotine cuts

**Algorithms:**
- **Shelf algorithms**: NFDH, FFDH, BFDH - pack in horizontal shelves
- **Guillotine**: Only edge-to-edge cuts (practical for real cutting)
- **Maximal rectangles**: Track free spaces, best-fit placement
- **Bottom-Left (BL)**: Place at lowest, then leftmost position
- **Skyline**: Track top profile of placed rectangles

**Complexity:** NP-hard, but good heuristics exist.

The baseline uses Shelf Next Fit Decreasing Height (SNFDH).
"""
