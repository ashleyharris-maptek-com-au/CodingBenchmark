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
import subprocess
import sys
import tempfile
import os
import time
from typing import List, Tuple, Set

title = "Minesweeper Solver"

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

  return f"""You are writing a Minesweeper solver that must handle ANY board complexity from trivial 
  to ludicrous scale:

- **Trivial**: Small boards (5x5-8x8), simple patterns, few mines
- **Extreme**: Massive boards (50x50-100x100), very complex constraint satisfaction problems

**The Challenge:**
Your `solve_minesweeper(board)` function will be tested with boards ranging from 5x5 to 100x100 with varying mine densities. The same function must work efficiently across ALL scales.

**Input:**
- `board`: 2D list of characters representing the board state

**Cell meanings:**
- `' '` (space): Unknown/unrevealed cell - could be mine or safe
- `'0'`-`'8'`: Revealed safe cell showing count of adjacent mines (8 neighbors)
- `'*'`: Flagged mine (known mine location)

**Output:**
- List of (x, y) coordinates of cells **guaranteed** to be safe
- x is column (0 = leftmost), y is row (0 = topmost)
- Only return cells that are currently unknown (' ') and provably safe
- Do NOT guess - only return cells you can prove are safe

**Example:**
```
  012
0 1*
1 11
2   
```
Cell (0,0) shows '1' and has one adjacent mine at (1,0). So (0,1) and (0,2) are guaranteed safe.

**Constraints:**
- Use only Python standard library or numpy
- Return as many provably safe cells as possible
- Never return a cell that could be a mine
- Must handle varying board sizes and mine densities efficiently

Write complete, runnable Python code with the solve_minesweeper function.
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
    "python_code": {
      "type":
      "string",
      "description":
      "Complete Python code with solve_minesweeper(board) function that handles all scales"
    }
  },
  "required": ["reasoning", "python_code"],
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


def execute_solver(code: str, board: List[List[str]], timeout: int = TIMEOUT_SECONDS) -> tuple:
  """Execute the LLM's solver."""
  from solver_utils import execute_solver_with_data

  data_dict = {
    'board': board,
  }

  result, error, exec_time = execute_solver_with_data(code, data_dict, 'solve_minesweeper', timeout)

  if error:
    return None, error, exec_time

  moves = None
  if isinstance(result, dict):
    moves = result.get('moves', [])
  else:
    moves = result

  if not isinstance(moves, list):
    return None, f"Invalid output: expected list, got {type(moves).__name__}", exec_time

  try:
    return [(int(m[0]), int(m[1])) for m in moves], None, exec_time
  except Exception as e:
    return None, f"Invalid move format: {e}", exec_time


lastGame = None


def gradeAnswer(result: dict, subPass: int, aiEngineName: str) -> tuple:
  """Grade the Minesweeper solver."""

  global lastGame

  if not result:
    return 0.0, "No result provided"

  if "python_code" not in result:
    return 0.0, "No Python code provided"

  case = TEST_CASES[subPass]
  board = case["board"]()
  safe_cells = case["safe_cells"]()
  description = case["description"]
  code = result["python_code"]

  # Execute solver
  moves, error, exec_time = execute_solver(code, board)

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

    if "python_code" in result:
      code = result["python_code"]
      code_escaped = code.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
      html += f"<details><summary>View Code ({len(code)} chars)</summary><pre>{code_escaped}</pre></details>"

  if lastGame:
    (board, moves, correct, incorrect) = lastGame
    if len(board) < 1000:
      html += "<p>Table map of the game board goes here.</p>"
    else:
      html += "<p>Board too large to display.</p>"

    html += f"<p>Moves: {len(moves)}, Correct: {correct}, Incorrect: {incorrect}</p>"

  return html


highLevelSummary = """
Minesweeper solving is a constraint satisfaction problem (CSP).

**Rules:**
- Each numbered cell indicates exactly that many mines in its 8 neighbors
- Goal: Find cells guaranteed to be safe without guessing

**Deduction techniques:**

1. **Basic satisfaction:** If numbered cell's adjacent mine count equals flagged neighbors,
   all other neighbors are safe.

2. **Basic flagging:** If numbered cell's unknown neighbors equal remaining mine count,
   all unknowns are mines.

3. **Subset/superset analysis:** Compare constraints between neighboring numbered cells.
   If one's unknowns are a subset of another's, combine constraints.

4. **CSP solving:** Model as constraint satisfaction, use backtracking with arc consistency.

5. **Probabilistic (for guessing):** When no certain moves, calculate mine probability per cell.

**Complexity:** Minesweeper is NP-complete in general, but local constraint propagation
works well for most practical boards.

The baseline uses simple constraint satisfaction - if a number's mines are all flagged,
remaining neighbors are safe.
"""
