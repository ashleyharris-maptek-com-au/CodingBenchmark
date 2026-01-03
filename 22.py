"""
Test 22: Tetris Game (C# Implementation)

The LLM must write C# code that plays Tetris optimally.
The program receives pieces one at a time and must decide placement.

Standard Tetris rules:
- 10-wide board, pieces fall from top
- Clear lines by filling them completely
- Game over when pieces stack to the top

Input format (stdin):
Line 1: width height num_pieces
For each piece:
  Line: piece_type (I, O, T, S, Z, J, L)
  Program outputs: rotation column
  Then receives: result (ok, cleared_N, gameover)

Output format (stdout):
For each piece: rotation column
  - rotation: 0-3 (number of 90° clockwise rotations)
  - column: 0 to width-1 (leftmost column of piece)

Subpasses test increasingly long games.
Solver times out after 5 minutes.
"""

skip = True

import random
import subprocess
import sys
import os
import time
from typing import List, Tuple, Set, Dict, Optional
from pathlib import Path

# Import our native compiler helper
from native_compiler import CSharpCompiler, CompilationError, ExecutionError

title = "Tetris Game (C#)"

# Timeout in seconds (5 minutes)
TIMEOUT_SECONDS = 30

# Seed for reproducibility
RANDOM_SEED = 22222222

# Tetris piece definitions (each rotation state)
# Coordinates relative to piece origin, (row, col) where row 0 is bottom
PIECES = {
  'I': [
    [(0, 0), (0, 1), (0, 2), (0, 3)],  # Horizontal
    [(0, 0), (1, 0), (2, 0), (3, 0)],  # Vertical
    [(0, 0), (0, 1), (0, 2), (0, 3)],  # Horizontal
    [(0, 0), (1, 0), (2, 0), (3, 0)],  # Vertical
  ],
  'O': [
    [(0, 0), (0, 1), (1, 0), (1, 1)],  # Square (all rotations same)
    [(0, 0), (0, 1), (1, 0), (1, 1)],
    [(0, 0), (0, 1), (1, 0), (1, 1)],
    [(0, 0), (0, 1), (1, 0), (1, 1)],
  ],
  'T': [
    [(0, 0), (0, 1), (0, 2), (1, 1)],  # T up
    [(0, 0), (1, 0), (2, 0), (1, 1)],  # T right
    [(1, 0), (1, 1), (1, 2), (0, 1)],  # T down
    [(0, 1), (1, 0), (1, 1), (2, 1)],  # T left
  ],
  'S': [
    [(0, 0), (0, 1), (1, 1), (1, 2)],  # S horizontal
    [(0, 1), (1, 0), (1, 1), (2, 0)],  # S vertical
    [(0, 0), (0, 1), (1, 1), (1, 2)],
    [(0, 1), (1, 0), (1, 1), (2, 0)],
  ],
  'Z': [
    [(0, 1), (0, 2), (1, 0), (1, 1)],  # Z horizontal
    [(0, 0), (1, 0), (1, 1), (2, 1)],  # Z vertical
    [(0, 1), (0, 2), (1, 0), (1, 1)],
    [(0, 0), (1, 0), (1, 1), (2, 1)],
  ],
  'J': [
    [(0, 0), (1, 0), (1, 1), (1, 2)],  # J
    [(0, 0), (0, 1), (1, 0), (2, 0)],
    [(0, 0), (0, 1), (0, 2), (1, 2)],
    [(0, 1), (1, 1), (2, 0), (2, 1)],
  ],
  'L': [
    [(0, 2), (1, 0), (1, 1), (1, 2)],  # L
    [(0, 0), (1, 0), (2, 0), (2, 1)],
    [(0, 0), (0, 1), (0, 2), (1, 0)],
    [(0, 0), (0, 1), (1, 1), (2, 1)],
  ],
}


def get_piece_width(piece_type: str, rotation: int) -> int:
  """Get width of piece in given rotation."""
  coords = PIECES[piece_type][rotation % 4]
  return max(c[1] for c in coords) + 1


def get_piece_height(piece_type: str, rotation: int) -> int:
  """Get height of piece in given rotation."""
  coords = PIECES[piece_type][rotation % 4]
  return max(c[0] for c in coords) + 1


class TetrisBoard:
  """Simulates a Tetris board."""

  def __init__(self, width: int, height: int):
    self.width = width
    self.height = height
    self.board = [[False] * width for _ in range(height)]
    self.lines_cleared = 0
    self.pieces_placed = 0
    self.game_over = False

  def get_column_height(self, col: int) -> int:
    """Get height of highest block in column."""
    for row in range(self.height - 1, -1, -1):
      if self.board[row][col]:
        return row + 1
    return 0

  def can_place(self, piece_type: str, rotation: int, col: int) -> bool:
    """Check if piece can be placed at column."""
    coords = PIECES[piece_type][rotation % 4]
    piece_width = max(c[1] for c in coords) + 1

    if col < 0 or col + piece_width > self.width:
      return False

    # Find landing row
    landing_row = self._find_landing_row(piece_type, rotation, col)
    if landing_row < 0:
      return False

    # Check if piece fits
    for r, c in coords:
      board_row = landing_row + r
      board_col = col + c
      if board_row >= self.height:
        return False

    return True

  def _find_landing_row(self, piece_type: str, rotation: int, col: int) -> int:
    """Find the row where piece lands."""
    coords = PIECES[piece_type][rotation % 4]

    # Start from top and drop
    for start_row in range(self.height - 1, -1, -1):
      can_place_here = True
      for r, c in coords:
        board_row = start_row + r
        board_col = col + c
        if board_row < 0 or board_col < 0 or board_col >= self.width:
          can_place_here = False
          break
        if board_row < self.height and self.board[board_row][board_col]:
          can_place_here = False
          break

      if not can_place_here:
        return start_row + 1

    return 0

  def place_piece(self, piece_type: str, rotation: int, col: int) -> Tuple[bool, int]:
    """
        Place a piece. Returns (success, lines_cleared).
        """
    if self.game_over:
      return False, 0

    coords = PIECES[piece_type][rotation % 4]
    piece_width = max(c[1] for c in coords) + 1

    if col < 0 or col + piece_width > self.width:
      return False, 0

    landing_row = self._find_landing_row(piece_type, rotation, col)

    # Place the piece
    for r, c in coords:
      board_row = landing_row + r
      board_col = col + c
      if board_row >= self.height:
        self.game_over = True
        return False, 0
      self.board[board_row][board_col] = True

    self.pieces_placed += 1

    # Check for line clears
    lines = self._clear_lines()
    self.lines_cleared += lines

    return True, lines

  def _clear_lines(self) -> int:
    """Clear full lines and return count."""
    lines_cleared = 0
    row = 0
    while row < self.height:
      if all(self.board[row]):
        # Remove this line
        del self.board[row]
        self.board.append([False] * self.width)
        lines_cleared += 1
      else:
        row += 1
    return lines_cleared

  def get_aggregate_height(self) -> int:
    """Sum of all column heights."""
    return sum(self.get_column_height(c) for c in range(self.width))

  def get_holes(self) -> int:
    """Count holes (empty cells with filled cells above)."""
    holes = 0
    for col in range(self.width):
      found_block = False
      for row in range(self.height - 1, -1, -1):
        if self.board[row][col]:
          found_block = True
        elif found_block:
          holes += 1
    return holes


def generate_piece_sequence(num_pieces: int, seed: int) -> List[str]:
  """Generate random piece sequence."""
  rng = random.Random(seed)
  piece_types = list(PIECES.keys())
  return [rng.choice(piece_types) for _ in range(num_pieces)]


# Test configurations
TEST_CASES = [
  # Subpass 0: Very short game
  {
    "width": 10,
    "height": 20,
    "num_pieces": 20,
    "description": "20 pieces"
  },
  # Subpass 1: Short game
  {
    "width": 10,
    "height": 20,
    "num_pieces": 50,
    "description": "50 pieces"
  },
  # Subpass 2: Medium game
  {
    "width": 10,
    "height": 20,
    "num_pieces": 100,
    "description": "100 pieces"
  },
  # Subpass 3: Longer game
  {
    "width": 10,
    "height": 20,
    "num_pieces": 200,
    "description": "200 pieces"
  },
  # Subpass 4: Long game
  {
    "width": 10,
    "height": 20,
    "num_pieces": 500,
    "description": "500 pieces"
  },
  # Subpass 5: Very long game
  {
    "width": 10,
    "height": 20,
    "num_pieces": 1000,
    "description": "1000 pieces"
  },
  # Extreme cases
  {
    "width": 10,
    "height": 20,
    "num_pieces": 2500,
    "description": "2500 pieces"
  },
  {
    "width": 10,
    "height": 20,
    "num_pieces": 5000,
    "description": "5000 pieces"
  },
  {
    "width": 10,
    "height": 20,
    "num_pieces": 10000,
    "description": "10000 pieces"
  },
  {
    "width": 10,
    "height": 20,
    "num_pieces": 50000,
    "description": "50000 pieces"
  },
  {
    "width": 10,
    "height": 20,
    "num_pieces": 100000,
    "description": "100000 pieces (marathon)"
  },
]

# Cache piece sequences
PIECE_CACHE = {}


def get_pieces(subpass: int) -> List[str]:
  """Get piece sequence for subpass."""
  if subpass not in PIECE_CACHE:
    case = TEST_CASES[subpass]
    PIECE_CACHE[subpass] = generate_piece_sequence(case["num_pieces"], RANDOM_SEED + subpass)
  return PIECE_CACHE[subpass]


def prepareSubpassPrompt(subPass: int) -> str:
  """Generate the prompt for subpass 0 that handles all game complexities."""
  if subPass != 0:
    raise StopIteration

  return f"""You are writing C# code to play Tetris optimally.

You must write a C# solver that can handle ANY game complexity from trivial to ludicrous scale:
- **Trivial**: Short games (10-20 pieces), standard 10x20 board, basic patterns
- **Medium**: Medium games (50-100 pieces), standard board, moderate complexity
- **Large**: Long games (200-500 pieces), standard board, complex stacking strategies
- **Extreme**: Very long games (1000+ pieces), wide boards, very complex optimization

**The Challenge:**
Your C# Tetris player will be tested with games ranging from short practice sessions to marathon games. The same algorithm must work efficiently across ALL game lengths and board configurations.

**Problem:**
Standard Tetris rules apply - 10-wide board (or variable width), pieces fall from top, clear lines by filling them completely, game over when pieces stack to the top. The program receives pieces one at a time and must decide optimal placement.

**Input format (stdin):**
```
width height num_pieces
For each piece:
  Line: piece_type (I, O, T, S, Z, J, L)
  Program outputs: rotation column
  Then receives: result (ok, cleared_N, gameover)
```

**Output format (stdout):**
For each piece: rotation column
  - rotation: 0-3 (number of 90° clockwise rotations)
  - column: 0 to width-1 (leftmost column of piece)

**Critical Requirements:**
1. **Scalability**: Your algorithm must adapt based on game length and board size
2. **Performance**: Must make decisions quickly even for very long games
3. **Quality**: Maximize lines cleared while avoiding game over

**Algorithm Strategy Recommendations:**
- **Short games (≤50 pieces)**: Can use exhaustive search, look-ahead planning
- **Medium games (50-200 pieces)**: Heuristic evaluation, limited look-ahead
- **Long games (200-500 pieces)**: Fast heuristics, pattern recognition
- **Very Long games (>500 pieces)**: Very fast heuristics, reactive strategies

**Key Techniques:**
- **Board evaluation**: Height, holes, wells, bumpiness metrics
- **Piece placement**: Try all valid rotations and positions
- **Look-ahead**: Simulate next few pieces for better decisions
- **Pattern recognition**: Common Tetris patterns and strategies
- **Adaptive strategy**: Adjust based on current board state

**Implementation Hints:**
- Detect game length and choose appropriate algorithm complexity
- Use efficient board representation and piece placement checking
- Implement adaptive quality vs speed tradeoffs
- For very long games, focus on fast evaluation
- Handle edge cases: no valid moves, emergency situations
- Use fast I/O for interactive gameplay

**Piece Types:**
- I: 4x1 line piece
- O: 2x2 square piece  
- T: T-shaped piece
- S: S-shaped piece (right-leaning)
- Z: Z-shaped piece (left-leaning)
- J: L-shaped piece (left-leaning)
- L: L-shaped piece (right-leaning)

**Requirements:**
1. Program must compile with .NET/csc
2. Read from stdin, write to stdout
3. Handle variable board widths and long game sequences
4. Complete within 5 minutes
5. Must handle varying game lengths efficiently

Write complete, compilable C# code with a Main method.
Include adaptive logic that chooses different strategies based on game complexity.
"""


def run_tetris_game(code: str, case: dict, subpass: int,
                    engine_name: str) -> Tuple[int, int, int, str, float]:
  """
    Compile and run Tetris game.
    
    Returns:
        Tuple of (pieces_placed, lines_cleared, score, end_reason, exec_time)
    """
  compiler = CSharpCompiler(engine_name)

  if not compiler.find_compiler():
    return 0, 0, 0, "No C# compiler found", 0

  try:
    exe_path = compiler.compile(code)
  except CompilationError as e:
    return 0, 0, 0, f"Compilation error: {str(e)[:500]}", 0

  pieces = get_pieces(subpass)
  board = TetrisBoard(case["width"], case["height"])

  start_time = time.time()

  try:
    # Determine how to run the executable
    if compiler._compiler_type == 'dotnet':
      cmd = ['dotnet', str(exe_path)]
    else:
      cmd = [str(exe_path)]

    process = subprocess.Popen(cmd,
                               stdin=subprocess.PIPE,
                               stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE,
                               text=True)

    # Send header
    process.stdin.write(f"{case['width']} {case['height']} {len(pieces)}\n")
    process.stdin.flush()

    end_reason = "completed"

    for piece_idx, piece_type in enumerate(pieces):
      if board.game_over:
        end_reason = "gameover"
        break

      # Check timeout
      if time.time() - start_time > TIMEOUT_SECONDS:
        end_reason = "timeout"
        break

      # Send piece
      process.stdin.write(f"{piece_type}\n")
      process.stdin.flush()

      # Read move
      try:
        line = process.stdout.readline()
        if not line:
          end_reason = "no_output"
          break

        parts = line.strip().split()
        if len(parts) < 2:
          end_reason = f"invalid_output: {line.strip()}"
          break

        rotation = int(parts[0])
        column = int(parts[1])

        # Place piece
        success, lines = board.place_piece(piece_type, rotation, column)

        # Send result
        if not success:
          process.stdin.write("gameover\n")
          process.stdin.flush()
          end_reason = "gameover"
          break
        elif lines > 0:
          process.stdin.write(f"cleared_{lines}\n")
        else:
          process.stdin.write("ok\n")
        process.stdin.flush()

      except Exception as e:
        end_reason = f"error: {str(e)}"
        break

    process.terminate()
    try:
      process.wait(timeout=2)
    except:
      process.kill()

    exec_time = time.time() - start_time

    # Calculate score
    score = board.lines_cleared * 100 + board.pieces_placed

    return board.pieces_placed, board.lines_cleared, score, end_reason, exec_time

  except Exception as e:
    return 0, 0, 0, f"Execution error: {str(e)}", time.time() - start_time


def gradeAnswer(result: dict, subPass: int, aiEngineName: str) -> tuple:
  """
    Grade the C# Tetris player.
    
    Scoring based on:
    - Pieces placed / total pieces
    - Bonus for lines cleared
    """
  if not result:
    return 0.0, "No result provided"

  if "csharp_code" not in result:
    return 0.0, "No C# code provided"

  case = TEST_CASES[subPass]
  description = case["description"]
  total_pieces = case["num_pieces"]

  code = result["csharp_code"]

  # Run game
  pieces_placed, lines_cleared, score, end_reason, exec_time = run_tetris_game(
    code, case, subPass, aiEngineName)

  # Score based on survival
  survival_rate = pieces_placed / total_pieces if total_pieces > 0 else 0

  # Bonus for lines cleared (efficiency)
  line_bonus = min(0.2, lines_cleared / (pieces_placed + 1) * 0.5) if pieces_placed > 0 else 0

  score = min(1.0, survival_rate + line_bonus)

  # Perfect score for completing all pieces
  if end_reason == "completed":
    score = 1.0

  explanation = (f"[{description}] Placed: {pieces_placed}/{total_pieces}, "
                 f"Lines: {lines_cleared}, End: {end_reason}, "
                 f"Time: {exec_time:.2f}s")

  return score, explanation


def resultToNiceReport(result: dict, subPass: int, aiEngineName: str) -> str:
  if not result:
    return "<p style='color:red'>No result provided</p>"
  case = TEST_CASES[subPass]
  html = f"<h4>Tetris AI - {case['description']}</h4>"
  if "reasoning" in result:
    r = result['reasoning'][:400] + ('...' if len(result.get('reasoning', '')) > 400 else '')
    html += f"<p><strong>Approach:</strong> {r.replace('<', '&lt;').replace('>', '&gt;')}</p>"
  if "csharp_code" in result:
    code = result["csharp_code"].replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    html += f"<details><summary>View C# Code ({len(result['csharp_code'])} chars)</summary><pre>{code}</pre></details>"
  return html


highLevelSummary = """
Tetris AI evaluates board states to maximize line clears.

**Key heuristics:**
- Aggregate height, holes, bumpiness
- Line clear potential
- Lookahead for upcoming pieces
"""
