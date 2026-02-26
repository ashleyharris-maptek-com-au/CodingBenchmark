import random
import subprocess
import sys
import os
import time
import threading
from queue import Queue, Empty
from typing import List, Tuple, Set, Dict, Optional
from pathlib import Path

# Import our native compiler helper
from native_compiler import CSharpCompiler, CompilationError, ExecutionError, describe_this_pc

title = "Tetris Game (C#)"

tags = [
  "csharp",
  "structured response",
  "game ai",
  "algorithm design",
]

# Timeout in seconds (5 minutes)
TIMEOUT_SECONDS = 300

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


LAST_GAME_STATS: Dict[Tuple[int, str], Dict] = {}


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
    self.board: List[List[Optional[str]]] = [[None] * width for _ in range(height)]
    self.lines_cleared = 0
    self.pieces_placed = 0
    self.game_over = False

  def get_column_height(self, col: int) -> int:
    """Get height of highest block in column."""
    for row in range(self.height - 1, -1, -1):
      if self.board[row][col] is not None:
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
        if board_row < self.height and self.board[board_row][board_col] is not None:
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
      self.board[board_row][board_col] = piece_type

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
      if all(cell is not None for cell in self.board[row]):
        # Remove this line
        del self.board[row]
        self.board.append([None] * self.width)
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
        if self.board[row][col] is not None:
          found_block = True
        elif found_block:
          holes += 1
    return holes

  def get_bumpiness(self) -> int:
    """Sum of absolute height differences between adjacent columns."""
    heights = [self.get_column_height(c) for c in range(self.width)]
    return sum(abs(heights[i] - heights[i + 1]) for i in range(len(heights) - 1))


def create_tetris_svg(board: List[List[Optional[str]]], width: int, height: int) -> str:
  """Create an SVG visualization of the Tetris board."""
  if not board:
    return "<p>No board state available to render.</p>"

  piece_colors = {
    "I": "#38bdf8",
    "O": "#facc15",
    "T": "#a855f7",
    "S": "#22c55e",
    "Z": "#ef4444",
    "J": "#6366f1",
    "L": "#f97316",
  }

  cell_size = max(8, min(24, 480 // max(width, height)))
  margin = 14
  svg_width = width * cell_size + margin * 2
  svg_height = height * cell_size + margin * 2

  svg_elements = [
    f'<rect width="{svg_width}" height="{svg_height}" fill="#0b1120" stroke="#334155" stroke-width="2"/>'
  ]

  for x in range(width + 1):
    x_pos = margin + x * cell_size
    svg_elements.append(
      f'<line x1="{x_pos}" y1="{margin}" x2="{x_pos}" y2="{margin + height * cell_size}" stroke="#1e293b" stroke-width="1"/>'
    )

  for y in range(height + 1):
    y_pos = margin + y * cell_size
    svg_elements.append(
      f'<line x1="{margin}" y1="{y_pos}" x2="{margin + width * cell_size}" y2="{y_pos}" stroke="#1e293b" stroke-width="1"/>'
    )

  for row in range(height):
    for col in range(width):
      if board[row][col] is None:
        continue
      x = margin + col * cell_size
      y = margin + (height - 1 - row) * cell_size
      piece_type = board[row][col]
      color = piece_colors.get(piece_type, "#94a3b8")
      svg_elements.append(
        f'<rect x="{x + 1}" y="{y + 1}" width="{cell_size - 2}" height="{cell_size - 2}" fill="{color}" stroke="#0f172a" stroke-width="1" rx="2"/>'
      )

  svg_html = f'''
    <div style="margin: 16px 0; padding: 12px; border: 1px solid #1f2937; border-radius: 8px; background: #0b1120;">
      <h5 style="margin: 0 0 10px 0; color: #e2e8f0;">Final Board State</h5>
      <svg width="{svg_width}" height="{svg_height}" style="border: 1px solid #334155; background: #0b1120;">
        {"".join(svg_elements)}
      </svg>
    </div>
  '''

  return svg_html


def read_line_with_timeout(process: subprocess.Popen, timeout_seconds: float) -> Tuple[Optional[str], bool, float]:
  """Read a line from process.stdout with timeout. Returns (line, timed_out, duration)."""
  if not process.stdout:
    return None, False, 0.0

  queue: Queue = Queue(maxsize=1)

  def _reader():
    try:
      queue.put(process.stdout.readline())
    except Exception:
      queue.put(None)

  start_time = time.time()
  thread = threading.Thread(target=_reader, daemon=True)
  thread.start()
  try:
    line = queue.get(timeout=timeout_seconds)
    return line, False, time.time() - start_time
  except Empty:
    return None, True, time.time() - start_time


def generate_piece_sequence(num_pieces: int, seed: int) -> List[str]:
  """Generate random piece sequence."""
  rng = random.Random(seed)
  piece_types = list(PIECES.keys())
  return [rng.choice(piece_types) for _ in range(num_pieces)]


# Test configurations
TEST_CASES = [
  {
    "width": 10,
    "height": 20,
    "num_pieces": 20,
    "description": "20 pieces"
  },
  {
    "width": 10,
    "height": 25,
    "num_pieces": 50,
    "description": "50 pieces"
  },
  {
    "width": 12,
    "height": 20,
    "num_pieces": 100,
    "description": "100 pieces"
  },
  {
    "width": 10,
    "height": 20,
    "num_pieces": 200,
    "description": "200 pieces"
  },
  {
    "width": 14,
    "height": 25,
    "num_pieces": 500,
    "description": "500 pieces"
  },
  {
    "width": 10,
    "height": 20,
    "num_pieces": 1000,
    "description": "1000 pieces"
  },
  {
    "width": 10,
    "height": 20,
    "num_pieces": 2500,
    "description": "2500 pieces"
  },
  {
    "width": 20,
    "height": 30,
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
    "width": 24,
    "height": 36,
    "num_pieces": 100000,
    "description": "100000 pieces (marathon)"
  },
]

extraGradeAnswerRuns = list(range(len(TEST_CASES)))

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

Your C# Tetris player will be tested with games ranging from short practice sessions to marathon games. 

**Problem:**
Standard Tetris rules apply - variable width board (typically 10-20 columns), 
pieces fall from top, clear lines by filling them completely, game over when pieces stack to the top. 
The program receives pieces one at a time and must decide optimal placement.

**Input format (stdin, whitespace seperated decimal integers):**
```
width height num_pieces
For each piece:
  Line: piece_type (I, O, T, S, Z, J, L)
  Program outputs: rotation column
  Then receives: result (ok, cleared_N, gameover)
```

**Output format (stdout, 1 line per piece, whitespace seperated decimal integers):**
rotation column
  - rotation: 0-3 (number of 90° clockwise rotations)
  - column: 0 to width-1 (leftmost column of piece)

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
4. Must place a peice within 5 seconds of it being written to STDIN.

**Environment:**
{describe_this_pc()}

**C# Compiler:**
{CSharpCompiler("test_engine").describe()}


Write complete, compilable C# code with a Main method.
Include adaptive logic that chooses different strategies based on game complexity.
"""

structure = {
  "type": "object",
  "properties": {
    "reasoning": {
      "type": "string",
      "description": "Explain your strategy"
    },
    "csharp_code": {
      "type": "string",
      "description": "Complete C# code with Main method and processes each tetris block."
    }
  },
  "required": ["reasoning", "csharp_code"],
  "additionalProperties": False
}



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
  response_times: List[float] = []
  piece_counts: Dict[str, int] = {piece: 0 for piece in PIECES}
  stall_timeouts = 0

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
        line, timed_out, duration = read_line_with_timeout(process, 5.0)
        response_times.append(duration)
        if timed_out:
          stall_timeouts += 1
          rotation = 0
          piece_width = get_piece_width(piece_type, rotation)
          column = max(0, (case["width"] - piece_width) // 2)
        else:
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
        piece_counts[piece_type] += 1

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

    LAST_GAME_STATS[(subpass, engine_name)] = {
      "board": board.board,
      "width": case["width"],
      "height": case["height"],
      "lines_cleared": board.lines_cleared,
      "pieces_placed": board.pieces_placed,
      "total_pieces": len(pieces),
      "end_reason": end_reason,
      "exec_time": exec_time,
      "response_times": response_times,
      "piece_counts": piece_counts,
      "stall_timeouts": stall_timeouts,
      "aggregate_height": board.get_aggregate_height(),
      "holes": board.get_holes(),
      "bumpiness": board.get_bumpiness(),
    }

    # Calculate score
    score = board.lines_cleared * 100 + board.pieces_placed

    return board.pieces_placed, board.lines_cleared, score, end_reason, exec_time

  except Exception as e:
    exec_time = time.time() - start_time
    LAST_GAME_STATS[(subpass, engine_name)] = {
      "board": board.board if "board" in locals() else [],
      "width": case["width"],
      "height": case["height"],
      "lines_cleared": board.lines_cleared if "board" in locals() else 0,
      "pieces_placed": board.pieces_placed if "board" in locals() else 0,
      "total_pieces": len(pieces) if "pieces" in locals() else 0,
      "end_reason": f"Execution error: {str(e)}",
      "exec_time": exec_time,
      "response_times": response_times if "response_times" in locals() else [],
      "piece_counts": piece_counts if "piece_counts" in locals() else {},
      "stall_timeouts": stall_timeouts if "stall_timeouts" in locals() else 0,
      "aggregate_height": board.get_aggregate_height() if "board" in locals() else 0,
      "holes": board.get_holes() if "board" in locals() else 0,
      "bumpiness": board.get_bumpiness() if "board" in locals() else 0,
    }
    return 0, 0, 0, f"Execution error: {str(e)}", exec_time


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

  if lines_cleared < 1:
    score = 0
  elif end_reason == "completed":
    score = 1.0
  else:
    score = pieces_placed / total_pieces / 2

  stats = LAST_GAME_STATS.get((subPass, aiEngineName), {})
  stall_timeouts = stats.get("stall_timeouts", 0)
  stall_note = f", Stall timeouts: {stall_timeouts}" if stall_timeouts else ""
  explanation = (f"[{description}] Placed: {pieces_placed}/{total_pieces}, "
                 f"Lines: {lines_cleared}, End: {end_reason}, "
                 f"Time: {exec_time:.2f}s{stall_note}")

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
  stats = LAST_GAME_STATS.get((subPass, aiEngineName))
  if stats:
    response_times = stats.get("response_times", [])
    avg_rt = (sum(response_times) / len(response_times)) if response_times else 0
    max_rt = max(response_times) if response_times else 0
    median_rt = 0
    if response_times:
      sorted_times = sorted(response_times)
      mid = len(sorted_times) // 2
      median_rt = (sorted_times[mid] if len(sorted_times) % 2 == 1
                   else (sorted_times[mid - 1] + sorted_times[mid]) / 2)
    pieces_placed = stats.get("pieces_placed", 0)
    total_pieces = stats.get("total_pieces", 0)
    lines_cleared = stats.get("lines_cleared", 0)
    exec_time = stats.get("exec_time", 0)
    survival_rate = (pieces_placed / total_pieces) if total_pieces else 0
    lines_per_100 = (lines_cleared / pieces_placed * 100) if pieces_placed else 0
    piece_counts = stats.get("piece_counts", {})
    stall_timeouts = stats.get("stall_timeouts", 0)
    distribution = ", ".join(
      f"{k}: {v}" for k, v in piece_counts.items() if v > 0) or "None"
    avg_rt_ms = avg_rt * 1000
    median_rt_ms = median_rt * 1000
    max_rt_ms = max_rt * 1000

    html += f"""
      <div style="margin: 14px 0; padding: 12px; border: 1px solid #1f2937; border-radius: 8px; background: #0b1120; color: #e2e8f0;">
        <h5 style="margin: 0 0 10px 0; color: #e2e8f0;">Run Statistics</h5>
        <table style="border-collapse: collapse; width: 100%; font-size: 13px; color: #e2e8f0;">
          <tr>
            <td style="padding: 4px 8px; color: #94a3b8;">Pieces placed</td>
            <td style="padding: 4px 8px; font-weight: 600;">{pieces_placed} / {total_pieces}</td>
            <td style="padding: 4px 8px; color: #94a3b8;">Lines cleared</td>
            <td style="padding: 4px 8px; font-weight: 600;">{lines_cleared}</td>
          </tr>
          <tr>
            <td style="padding: 4px 8px; color: #94a3b8;">Avg response time</td>
            <td style="padding: 4px 8px; font-weight: 600;">{avg_rt_ms:.1f} ms</td>
            <td style="padding: 4px 8px; color: #94a3b8;">Median / Max</td>
            <td style="padding: 4px 8px; font-weight: 600;">{median_rt_ms:.1f} ms / {max_rt_ms:.1f} ms</td>
          </tr>
          <tr>
            <td style="padding: 4px 8px; color: #94a3b8;">Total execution time</td>
            <td style="padding: 4px 8px; font-weight: 600;">{exec_time:.2f}s</td>
            <td style="padding: 4px 8px; color: #94a3b8;">Survival rate</td>
            <td style="padding: 4px 8px; font-weight: 600;">{survival_rate:.1%}</td>
          </tr>
          <tr>
            <td style="padding: 4px 8px; color: #94a3b8;">Lines per 100 pieces</td>
            <td style="padding: 4px 8px; font-weight: 600;">{lines_per_100:.2f}</td>
            <td style="padding: 4px 8px; color: #94a3b8;">Aggregate height</td>
            <td style="padding: 4px 8px; font-weight: 600;">{stats.get('aggregate_height', 0)}</td>
          </tr>
          <tr>
            <td style="padding: 4px 8px; color: #94a3b8;">Holes</td>
            <td style="padding: 4px 8px; font-weight: 600;">{stats.get('holes', 0)}</td>
            <td style="padding: 4px 8px; color: #94a3b8;">Bumpiness</td>
            <td style="padding: 4px 8px; font-weight: 600;">{stats.get('bumpiness', 0)}</td>
          </tr>
          <tr>
            <td style="padding: 4px 8px; color: #94a3b8;">Stall timeouts</td>
            <td style="padding: 4px 8px; font-weight: 600;">{stall_timeouts}</td>
            <td style="padding: 4px 8px; color: #94a3b8;">Stall behavior</td>
            <td style="padding: 4px 8px; font-weight: 600;">Auto-drop in center column</td>
          </tr>
          <tr>
            <td style="padding: 4px 8px; color: #94a3b8;">Piece distribution</td>
            <td colspan="3" style="padding: 4px 8px; font-weight: 600;">{distribution}</td>
          </tr>
        </table>
      </div>
    """
    html += create_tetris_svg(stats.get("board"), stats.get("width"), stats.get("height"))
  return html


highLevelSummary = """
<p>Write an AI that plays Tetris. Given the current board state and the next piece,
choose where to place it to survive as long as possible and clear as many lines
as you can. Good play means keeping the board flat, avoiding buried holes, and
setting up multi-line clears.</p>
<p>Subpasses increase the board size and speed, demanding smarter lookahead and
evaluation. The AI must balance short-term survival with long-term board health.</p>
"""
