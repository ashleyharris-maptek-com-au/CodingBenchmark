"""
Test 14: Minesweeper Solver

The LLM must write a Python solver that analyzes a Minesweeper board state
and identifies cells that are guaranteed to be safe (non-mines).

Board representation:
- ' ' (space): Unknown/unrevealed cell
- '0'-'8': Revealed cell showing count of adjacent mines
- '*': Flagged mine (known mine location)

The solver should use constraint satisfaction to deduce safe cells.

Subpasses test increasingly large boards.
Solver times out after 5 minutes.
"""

import random
import time
from typing import List, Tuple, Set

from native_compiler import CSharpCompiler, compile_and_run, describe_this_pc
from solver_utils import StreamingInputFile

title = "Minesweeper Solver (C#)"

tags = [
  "csharp",
  "structured response",
  "constraint satisfaction",
  "game ai",
]

# Timeout in seconds (5 minutes)
TIMEOUT_SECONDS = 30

# Seed for reproducibility
RANDOM_SEED = 33333


def generate_minesweeper_board(width: int, height: int, num_mines: int, reveal_ratio: float,
                               seed: int) -> Tuple[List[List[str]], Set[Tuple[int, int]]]:
  """
    Generate a Minesweeper board with some cells revealed.
    Returns (board, safe_unrevealed_cells).
    """
  rng = random.Random(seed)

  # Place mines
  all_cells = [(x, y) for x in range(width) for y in range(height)]
  mine_cells = set(rng.sample(all_cells, min(num_mines, len(all_cells))))

  # Calculate numbers for each cell
  def count_adjacent_mines(x, y):
    count = 0
    for dx in [-1, 0, 1]:
      for dy in [-1, 0, 1]:
        if dx == 0 and dy == 0:
          continue
        nx, ny = x + dx, y + dy
        if (nx, ny) in mine_cells:
          count += 1
    return count

  # Create full solution board
  solution = [[' ' for _ in range(width)] for _ in range(height)]
  for y in range(height):
    for x in range(width):
      if (x, y) in mine_cells:
        solution[y][x] = '*'
      else:
        solution[y][x] = str(count_adjacent_mines(x, y))

  # Create visible board with some cells revealed
  board = [[' ' for _ in range(width)] for _ in range(height)]

  # Reveal some non-mine cells
  non_mine_cells = [(x, y) for x, y in all_cells if (x, y) not in mine_cells]
  num_reveal = int(len(non_mine_cells) * reveal_ratio)
  revealed = set(rng.sample(non_mine_cells, num_reveal))

  # Also flag some mines (make them known)
  num_flag = int(len(mine_cells) * 0.3)  # Flag 30% of mines
  flagged = set(rng.sample(list(mine_cells), min(num_flag, len(mine_cells))))

  for y in range(height):
    for x in range(width):
      if (x, y) in revealed:
        board[y][x] = solution[y][x]
      elif (x, y) in flagged:
        board[y][x] = '*'
      else:
        board[y][x] = ' '

  # Calculate which unrevealed cells are actually safe
  safe_unrevealed = set()
  for y in range(height):
    for x in range(width):
      if board[y][x] == ' ' and (x, y) not in mine_cells:
        safe_unrevealed.add((x, y))

  return board, safe_unrevealed


def board_to_string(board: List[List[str]]) -> str:
  """Convert board to string representation."""
  return '\n'.join(''.join(row) for row in board)


def string_to_board(s: str) -> List[List[str]]:
  """Convert string to board."""
  return [list(row) for row in s.strip().split('\n')]


# Test configurations
TEST_CASES = [{
  "board": lambda: generate_minesweeper_board(8, 8, 10, 0.4, RANDOM_SEED)[0],
  "safe_cells": lambda: generate_minesweeper_board(8, 8, 10, 0.4, RANDOM_SEED)[1],
  "description": "8x8 board, 10 mines"
}, {
  "board": lambda: generate_minesweeper_board(24, 24, 80, 0.25, RANDOM_SEED + 3)[0],
  "safe_cells": lambda: generate_minesweeper_board(24, 24, 80, 0.25, RANDOM_SEED + 3)[1],
  "description": "24x24 board, 80 mines"
}, {
  "board":
  lambda: generate_minesweeper_board(50, 50, 400, 0.15, RANDOM_SEED + 5)[0],
  "safe_cells":
  lambda: generate_minesweeper_board(50, 50, 400, 0.15, RANDOM_SEED + 5)[1],
  "description":
  "50x50 board, 400 mines"
}, {
  "board":
  lambda: generate_minesweeper_board(100, 100, 1500, 0.1, RANDOM_SEED + 6)[0],
  "safe_cells":
  lambda: generate_minesweeper_board(100, 100, 1500, 0.1, RANDOM_SEED + 6)[1],
  "description":
  "100x100 board, 1500 mines"
}, {
  "board":
  lambda: generate_minesweeper_board(200, 200, 6000, 0.08, RANDOM_SEED + 7)[0],
  "safe_cells":
  lambda: generate_minesweeper_board(200, 200, 6000, 0.08, RANDOM_SEED + 7)[1],
  "description":
  "200x200 board, 6000 mines"
}, {
  "board":
  lambda: generate_minesweeper_board(500, 500, 40000, 0.05, RANDOM_SEED + 8)[0],
  "safe_cells":
  lambda: generate_minesweeper_board(500, 500, 40000, 0.05, RANDOM_SEED + 8)[1],
  "description":
  "500x500 board, 40000 mines"
}, {
  "board":
  lambda: generate_minesweeper_board(1000, 1000, 150000, 0.03, RANDOM_SEED + 9)[0],
  "safe_cells":
  lambda: generate_minesweeper_board(1000, 1000, 150000, 0.03, RANDOM_SEED + 9)[1],
  "description":
  "1000x1000 board, 150000 mines (1M cells)"
}, {
  "board":
  lambda: generate_minesweeper_board(2000, 2000, 600000, 0.02, RANDOM_SEED + 10)[0],
  "safe_cells":
  lambda: generate_minesweeper_board(2000, 2000, 600000, 0.02, RANDOM_SEED + 10)[1],
  "description":
  "2000x2000 board, 600000 mines (4M cells)"
}, {
  "board":
  lambda: generate_minesweeper_board(5000, 5000, 1000000, 0.02, RANDOM_SEED + 11)[0],
  "safe_cells":
  lambda: generate_minesweeper_board(5000, 5000, 1000000, 0.02, RANDOM_SEED + 11)[1],
  "description":
  "5000x5000 board, 1000000 mines (25M cells)"
}, {
  "board":
  lambda: generate_minesweeper_board(10000, 10000, 5000000, 0.005, RANDOM_SEED + 12)[0],
  "safe_cells":
  lambda: generate_minesweeper_board(10000, 10000, 5000000, 0.005, RANDOM_SEED + 12)[1],
  "description":
  "10000x10000 board, 5000000 mines (100M cells)"
}]


def prepareSubpassPrompt(subPass: int) -> str:
  """Generate the prompt for subpass 0 that handles all board sizes."""
  if subPass != 0:
    raise StopIteration

  return f"""You are writing a Minesweeper solver in C# that must handle ANY board complexity from trivial to ludicrous scale.

- **Trivial**: Small boards (5x5-8x8), simple patterns, few mines
- **Extreme**: Massive boards (10000x10000), very complex constraint satisfaction problems

**Input format (stdin):**
Line 1: H W (height and width)
Next H lines: the board row strings (each W characters)

**Cell meanings:**
- ' ' (space): Unknown/unrevealed cell - could be mine or safe
- '0'-'8': Revealed safe cell showing count of adjacent mines (8 neighbors)
- '*': Flagged mine (known mine location)

**Output format (stdout):**
Line 1: N (number of safe cells found)
Next N lines: x y (column row, 0-indexed)
Only output cells that are currently unknown (' ') and provably safe.
Do NOT guess - only return cells you can prove are safe.

**Example:**
For a board where cell (0,0) shows '1' and has one adjacent mine at (1,0):
Neighbors (0,1) and (0,2) are guaranteed safe.

**Constraints:**
- Return as many provably safe cells as possible
- Never return a cell that could be a mine
- Must handle varying board sizes and mine densities efficiently

**Environment:**
{describe_this_pc()}

**C# Compiler:**
{CSharpCompiler("test_engine").describe()}

Write complete, compilable C# code with a static void Main method.
Include adaptive logic that chooses different strategies based on board complexity.
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
      "Explain your Minesweeper solving algorithm and how it adapts to different board complexities"
    },
    "csharp_code": {
      "type": "string",
      "description": "Complete C# code with Main method that handles all scales"
    }
  },
  "required": ["reasoning", "csharp_code"],
  "additionalProperties": False
}


def validate_solution(moves: List[Tuple[int, int]], board: List[List[str]],
                      safe_cells: Set[Tuple[int, int]]) -> Tuple[bool, str, int, int]:
  """
    Validate Minesweeper solution.
    Returns (is_valid, error, correct_count, incorrect_count).
    """
  if not isinstance(moves, list):
    return False, "Moves must be a list", 0, 0

  height = len(board)
  width = len(board[0]) if board else 0

  correct = 0
  incorrect = 0

  for move in moves:
    if not isinstance(move, (list, tuple)) or len(move) != 2:
      continue

    x, y = int(move[0]), int(move[1])

    # Check bounds
    if x < 0 or x >= width or y < 0 or y >= height:
      continue

    # Check if cell is unknown
    if board[y][x] != ' ':
      continue  # Not an unknown cell, skip

    # Check if actually safe
    if (x, y) in safe_cells:
      correct += 1
    else:
      incorrect += 1

  if incorrect > 0:
    return False, f"Returned {incorrect} unsafe cell(s) (mines!)", correct, incorrect

  return True, "", correct, incorrect


def get_baseline_safe_cells(board: List[List[str]]) -> Set[Tuple[int, int]]:
  """
    Basic constraint satisfaction to find provably safe cells.
    Uses simple rule: if a number has all its mines accounted for, 
    remaining neighbors are safe.
    """
  height = len(board)
  width = len(board[0]) if board else 0

  safe = set()

  def get_neighbors(x, y):
    neighbors = []
    for dx in [-1, 0, 1]:
      for dy in [-1, 0, 1]:
        if dx == 0 and dy == 0:
          continue
        nx, ny = x + dx, y + dy
        if 0 <= nx < width and 0 <= ny < height:
          neighbors.append((nx, ny))
    return neighbors

  # Iterate until no changes
  changed = True
  iterations = 0
  max_iterations = 100

  while changed and iterations < max_iterations:
    changed = False
    iterations += 1

    for y in range(height):
      for x in range(width):
        cell = board[y][x]
        if cell not in '012345678':
          continue

        mine_count = int(cell)
        neighbors = get_neighbors(x, y)

        unknown = [(nx, ny) for nx, ny in neighbors
                   if board[ny][nx] == ' ' and (nx, ny) not in safe]
        flagged = sum(1 for nx, ny in neighbors if board[ny][nx] == '*')

        # If flagged mines equal the count, all unknown are safe
        if flagged == mine_count and unknown:
          for pos in unknown:
            if pos not in safe:
              safe.add(pos)
              changed = True

  return safe


STREAMING_THRESHOLD_CELLS = 500_000
_INPUT_FILE_CACHE = {}


def format_input(board: List[List[str]]) -> str:
  height = len(board)
  width = len(board[0]) if board else 0
  lines = [f"{height} {width}"]
  for row in board:
    lines.append("".join(row))
  return "\n".join(lines)


def _get_streaming_input(subpass: int, board: List[List[str]]) -> StreamingInputFile:
  if subpass in _INPUT_FILE_CACHE:
    return _INPUT_FILE_CACHE[subpass]

  height = len(board)
  width = len(board[0]) if board else 0
  cache_key = f"mine14|h={height}|w={width}|seed={RANDOM_SEED + subpass}"

  def generator():
    yield f"{height} {width}\n"
    for row in board:
      yield "".join(row) + "\n"

  input_file = StreamingInputFile(cache_key, generator, "test14_mine")
  _INPUT_FILE_CACHE[subpass] = input_file
  return input_file


def parse_moves_output(output: str) -> tuple:
  text = output.strip()
  if not text:
    return [], None

  lines = [l for l in text.splitlines() if l.strip()]
  if not lines:
    return [], None

  try:
    n = int(lines[0])
  except ValueError:
    return None, "First line must be count of safe cells"

  if n == 0:
    return [], None

  if len(lines) < 1 + n:
    return None, f"Expected {n} coordinate lines, got {len(lines) - 1}"

  moves = []
  for i in range(1, 1 + n):
    parts = lines[i].split()
    if len(parts) < 2:
      return None, f"Invalid coordinate line {i}"
    try:
      x, y = int(parts[0]), int(parts[1])
      moves.append((x, y))
    except ValueError:
      return None, f"Non-integer coordinate at line {i}"

  return moves, None


def execute_solver(code: str,
                   board: List[List[str]],
                   subpass: int,
                   ai_engine_name: str,
                   timeout: int = TIMEOUT_SECONDS) -> tuple:
  """Execute the LLM's solver."""
  height = len(board)
  width = len(board[0]) if board else 0
  total_cells = height * width

  if total_cells > STREAMING_THRESHOLD_CELLS:
    streaming_input = _get_streaming_input(subpass, board)
    input_file_path = streaming_input.generate()
    run = compile_and_run(code,
                          "csharp",
                          ai_engine_name,
                          input_file=input_file_path,
                          timeout=timeout)
  else:
    input_data = format_input(board)
    run = compile_and_run(code, "csharp", ai_engine_name, input_data=input_data, timeout=timeout)

  if not run:
    return None, run.error_message(), run.exec_time

  moves, parse_error = parse_moves_output(run.stdout)
  if parse_error:
    return None, parse_error, run.exec_time

  return moves, None, run.exec_time


lastGame = None


def _render_minesweeper_grid_html(board: List[List[str]], moves: List[Tuple[int, int]]) -> str:
  height = len(board)
  width = len(board[0]) if height else 0
  total = height * width

  if total == 0:
    return "<p>(empty board)</p>"

  max_cells_full = 10_000
  if total <= max_cells_full:
    view_x0, view_y0, view_w, view_h = 0, 0, width, height
  else:
    win = 60
    if moves:
      mx, my = moves[0]
      cx = max(0, min(width - 1, int(mx)))
      cy = max(0, min(height - 1, int(my)))
    else:
      cx, cy = width // 2, height // 2
    view_x0 = max(0, min(width - win, cx - win // 2))
    view_y0 = max(0, min(height - win, cy - win // 2))
    view_w = min(win, width)
    view_h = min(win, height)

  move_set = set()
  for x, y in moves:
    if 0 <= x < width and 0 <= y < height:
      move_set.add((int(x), int(y)))

  cell_px = 14
  grid_style = (f"display:grid;grid-template-columns:repeat({view_w},{cell_px}px);"
                f"grid-auto-rows:{cell_px}px;gap:1px;background:#cfcfcf;"
                "border:1px solid #cfcfcf;overflow:auto;max-height:700px;max-width:100%;")
  cell_base = (
    f"width:{cell_px}px;height:{cell_px}px;display:flex;align-items:center;justify-content:center;"
    "font:700 11px/1 monospace;user-select:none;")

  num_colors = {
    '0': '#777',
    '1': '#1976d2',
    '2': '#2e7d32',
    '3': '#d32f2f',
    '4': '#512da8',
    '5': '#6d4c41',
    '6': '#00838f',
    '7': '#424242',
    '8': '#000000',
  }

  cells = []
  for y in range(view_y0, view_y0 + view_h):
    row = board[y]
    for x in range(view_x0, view_x0 + view_w):
      c = row[x]
      if c == ' ':
        bg = '#e0e0e0'
        fg = '#333'
        text = ''
      elif c == '*':
        bg = '#f5f5f5'
        fg = '#111'
        text = '*'
      else:
        bg = '#ffffff'
        fg = num_colors.get(c, '#111')
        text = c

      if (x, y) in move_set:
        bg = '#c8e6c9'

      title = f"({x},{y}) '{c if c != ' ' else ' '}'"
      style = cell_base + f"background:{bg};color:{fg};"
      cells.append(f"<div title=\"{title}\" style=\"{style}\">{text}</div>")

  view_note = ""
  if total > max_cells_full:
    view_note = (f"<div style='margin:8px 0;color:#666;font-size:12px;'>"
                 f"Showing {view_w}x{view_h} window at x=[{view_x0},{view_x0+view_w-1}], "
                 f"y=[{view_y0},{view_y0+view_h-1}] (board is {width}x{height})."
                 f"</div>")

  legend = ("<div style='margin:8px 0;color:#666;font-size:12px;'>"
            "Legend: unknown=grey, revealed=white, flagged='*', returned-safe=green."
            "</div>")

  return (
    "<div style='margin: 12px 0; padding: 10px; border: 1px solid #ddd; border-radius: 4px; background: white;'>"
    "<h5 style='margin:0 0 8px 0;color:#333;'>Board Visualization</h5>" + legend + view_note +
    f"<div style=\"{grid_style}\">" + "".join(cells) + "</div>" + "</div>")


def gradeAnswer(result: dict, subPass: int, aiEngineName: str) -> tuple:
  """Grade the Minesweeper solver."""

  global lastGame

  if not result:
    return 0.0, "No result provided"

  if "csharp_code" not in result:
    return 0.0, "No C# code provided"

  case = TEST_CASES[subPass]
  board = case["board"]()
  safe_cells = case["safe_cells"]()
  description = case["description"]
  code = result["csharp_code"]

  # Execute solver
  moves, error, exec_time = execute_solver(code, board, subPass, aiEngineName)

  if error:
    return 0.0, f"[{description}] {error}"

  # Validate solution
  is_valid, validation_error, correct, incorrect = validate_solution(moves, board, safe_cells)

  lastGame = (board, moves, correct, incorrect)

  if not is_valid:
    return 0.0, f"[{description}] {validation_error}"

  # Get baseline
  baseline_safe = get_baseline_safe_cells(board)
  baseline_count = len(baseline_safe)
  total_safe = len(safe_cells)

  # Score based on how many safe cells found vs baseline
  if baseline_count == 0:
    # No baseline deductions possible
    if correct > 0:
      score = 1.0
      quality = "found safe cells where baseline couldn't"
    else:
      score = 0.5
      quality = "no deductions (matches baseline)"
  else:
    ratio = correct / baseline_count
    if ratio >= 1.0:
      score = 1.0
      quality = "excellent (≥ baseline)"
    elif ratio >= 0.7:
      score = 0.85
      quality = "good (≥70% of baseline)"
    elif ratio >= 0.4:
      score = 0.7
      quality = "acceptable (≥40% of baseline)"
    elif correct > 0:
      score = 0.5
      quality = f"partial ({correct}/{baseline_count})"
    else:
      score = 0.3
      quality = "no safe cells found"

  explanation = (f"[{description}] Found: {correct} safe, Baseline: {baseline_count}, "
                 f"Total safe: {total_safe}, Time: {exec_time:.1f}s - {quality}")

  return score, explanation


def resultToNiceReport(result: dict, subPass: int, aiEngineName: str) -> str:
  """Generate HTML report."""
  if not result:
    return "<p style='color:red'>No result provided</p>"

  case = TEST_CASES[subPass]

  html = f"<h4>Minesweeper Solver - {case['description']}</h4>"

  if subPass == 0:
    if "reasoning" in result:
      reasoning = result['reasoning'][:500] + ('...'
                                               if len(result.get('reasoning', '')) > 500 else '')
      html += f"<p><strong>Algorithm:</strong> {reasoning}</p>"

    if "csharp_code" in result:
      code = result["csharp_code"]
      code_escaped = code.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
      html += f"<details><summary>View Code ({len(code)} chars)</summary><pre>{code_escaped}</pre></details>"

  if lastGame:
    (board, moves, correct, incorrect) = lastGame
    html += _render_minesweeper_grid_html(board, moves)

    html += f"<p>Moves: {len(moves)}, Correct: {correct}, Incorrect: {incorrect}</p>"

  return html


highLevelSummary = """
<p>Given a partially-revealed Minesweeper board, deduce which hidden cells are safe
to click and which contain mines. Each visible number tells you exactly how many
of its neighbours are mined &mdash; the AI must combine these clues to make
guaranteed-safe moves without guessing.</p>
<p>Larger boards and denser mine layouts make the logic harder. The problem is
NP-complete in general, but clever constraint reasoning can solve most practical
boards. The baseline uses only the simplest single-cell deduction rules.</p>
"""
