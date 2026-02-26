import time
import random
import sys
from typing import List, Tuple, Set

from native_compiler import RustCompiler, compile_and_run, describe_this_pc
from solver_utils import StreamingInputFile

title = "Maze Solver (Rust)"

tags = [
  "rust",
  "structured response",
  "algorithm design",
]

# Timeout in seconds (30 seconds)
TIMEOUT_SECONDS = 30

# Seed for reproducibility
RANDOM_SEED = 88888


def generate_random_maze(width: int, height: int, wall_density: float, seed: int) -> str:
  """Generate a random maze with guaranteed path from A to B."""
  rng = random.Random(seed)

  # Start with all walls
  maze = [['#' for _ in range(width)] for _ in range(height)]

  # Carve paths using randomized DFS
  def carve(x, y):
    maze[y][x] = ' '
    directions = [(0, 2), (2, 0), (0, -2), (-2, 0)]
    rng.shuffle(directions)
    for dx, dy in directions:
      nx, ny = x + dx, y + dy
      if 1 <= nx < width - 1 and 1 <= ny < height - 1 and maze[ny][nx] == '#':
        maze[y + dy // 2][x + dx // 2] = ' '
        carve(nx, ny)

  # Start carving from (1, 1)
  sys.setrecursionlimit(1000000000)
  try:
    carve(1, 1)
  except RecursionError:
    # For very large mazes, use iterative approach
    stack = [(1, 1)]
    maze[1][1] = ' '
    while stack:
      x, y = stack[-1]
      directions = [(0, 2), (2, 0), (0, -2), (-2, 0)]
      rng.shuffle(directions)
      found = False
      for dx, dy in directions:
        nx, ny = x + dx, y + dy
        if 1 <= nx < width - 1 and 1 <= ny < height - 1 and maze[ny][nx] == '#':
          maze[y + dy // 2][x + dx // 2] = ' '
          maze[ny][nx] = ' '
          stack.append((nx, ny))
          found = True
          break
      if not found:
        stack.pop()

  # Place A at top-left area, B at bottom-right area
  maze[1][1] = 'A'

  # Find a valid position for B in bottom-right quadrant
  for y in range(height - 2, height // 2, -1):
    for x in range(width - 2, width // 2, -1):
      if maze[y][x] == ' ':
        maze[y][x] = 'B'
        break
    else:
      continue
    break
  else:
    # Fallback
    maze[height - 2][width - 2] = 'B'

  return '\n'.join(''.join(row) for row in maze)


# Test mazes - increasing complexity
MAZES = [
  # Subpass 0: Simple 5x5
  """\
#####
#A  #
# # #
#  B#
#####""",

  # Subpass 1: 7x7 with more walls
  """\
#######
#A#   #
# # # #
#   # #
# ### #
#    B#
#######""",

  # Subpass 2: 10x10 maze
  """\
##########
#A       #
# ###### #
# #    # #
# # ## # #
# #  # # #
# ## # # #
#    #  B#
# ########
##########""",

  # Subpass 3: 15x15 maze
  """\
###############
#A    #       #
# ### # ##### #
# #   #     # #
# # ####### # #
# #       # # #
# ####### # # #
#       # # # #
# ##### # # # #
# #   # # # # #
# # # # # # # #
# # #   #   # #
# # ##### ### #
#     #      B#
###############""",

  # Subpass 4: 20x20 maze with multiple paths
  """\
####################
#A #     #    #    #
# ## ### # ## # ## #
#    #   # #  #  # #
# #### ### # ## ## #
#    #     #  #    #
# ## # ##### ##### #
# #  #     #     # #
# # ###### ##### # #
# #      #     # # #
# ###### # ### # # #
#      # #   # # # #
# #### # ### # # # #
# #    #   # # # # #
# # ###### # # # # #
# #        # # #   #
# ########## # ### #
#            #    B#
# ##################
####################""",

  # Subpass 5: 25x25 complex maze
  """\
#########################
#A  #     #       #     #
# # # ### # ##### # ### #
# #   #   #     # #   # #
# ##### ##### # # ### # #
#     #     # # #   # # #
# ### ##### # # ### # # #
# #       # # #   # # # #
# # ##### # # ### # # # #
# # #   # # #   # # # # #
# # # # # # ### # # # # #
# # # # # #   # # # # # #
# # # # # ### # # # # # #
# # # #     # # # # # # #
# # # ####### # # # # # #
# # #         # # # #   #
# # ########### # # ### #
# #           # # #   # #
# ########### # # ### # #
#     #     # # #   # # #
# ### # ### # # ### # # #
# #   #   # # #   # #   #
# # ##### # # ### # ### #
# #       #     #      B#
#########################""",
]

EXTREME_MAZE_SIZES = [
  (100, 100),  # Subpass 6: 100x100
  (500, 500),  # Subpass 7: 500x500
  (1000, 1000),  # Subpass 8: 1000x1000
  (5000, 5000),  # Subpass 9: 5000x5000
  (10000, 10000),  # Subpass 10: 10000x10000 - 100 million cells
  (20000, 20000),  # Subpass 11: 20000x20000 - 400 million cells
  (50000, 50000),  # Subpass 12: 50000x50000 - 2.5 billion cells
  (100000, 100000),  # Subpass 13: 100000x100000 - 10 billion cells
]


def get_maze(subpass: int) -> str:
  """Get maze for subpass, generating extreme ones on demand (cached)."""
  if subpass in _MAZE_CACHE:
    return _MAZE_CACHE[subpass]

  if subpass < len(MAZES):
    _MAZE_CACHE[subpass] = ensure_solvable_maze(MAZES[subpass])
    return _MAZE_CACHE[subpass]

  extreme_idx = subpass - len(MAZES)
  if extreme_idx < len(EXTREME_MAZE_SIZES):
    w, h = EXTREME_MAZE_SIZES[extreme_idx]
    _MAZE_CACHE[subpass] = generate_random_maze(w, h, 0.3, RANDOM_SEED + subpass)
    return _MAZE_CACHE[subpass]

  raise StopIteration


_MAZE_CACHE = {}


def _reachable_cells(lines: List[str], start: Tuple[int, int]) -> Set[Tuple[int, int]]:
  height = len(lines)
  width = max(len(line) for line in lines) if lines else 0

  def is_open(x: int, y: int) -> bool:
    if y < 0 or y >= height:
      return False
    if x < 0 or x >= len(lines[y]):
      return False
    return lines[y][x] != '#'

  q = [start]
  seen = {start}
  qi = 0
  while qi < len(q):
    x, y = q[qi]
    qi += 1
    for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
      nx, ny = x + dx, y + dy
      if (nx, ny) not in seen and is_open(nx, ny):
        seen.add((nx, ny))
        q.append((nx, ny))
  return seen


def ensure_solvable_maze(maze: str, max_breaks: int = 64) -> str:
  """Ensure there is a path from A to B by opening a minimal connecting wall if needed."""
  lines = maze.split('\n')
  info = get_maze_info(maze)
  start = info.get('start')
  end = info.get('end')
  if not start or not end:
    return maze

  height = len(lines)
  width = max(len(line) for line in lines) if lines else 0

  for _ in range(max_breaks):
    reach_a = _reachable_cells(lines, start)
    if end in reach_a:
      return '\n'.join(lines)

    reach_b = _reachable_cells(lines, end)
    # Find a wall cell adjacent to both components and open it
    opened = False
    for y in range(1, height - 1):
      row = lines[y]
      for x in range(1, min(width - 1, len(row) - 1)):
        if row[x] != '#':
          continue

        adj_a = False
        adj_b = False
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
          nx, ny = x + dx, y + dy
          if (nx, ny) in reach_a:
            adj_a = True
          if (nx, ny) in reach_b:
            adj_b = True
        if adj_a and adj_b:
          lines[y] = row[:x] + ' ' + row[x + 1:]
          opened = True
          break
      if opened:
        break

    if not opened:
      return maze

  return '\n'.join(lines)


def get_maze_info(maze: str) -> dict:
  """Extract maze information."""
  lines = maze.strip().split('\n')
  height = len(lines)
  width = max(len(line) for line in lines)

  start = None
  end = None

  for y, line in enumerate(lines):
    for x, char in enumerate(line):
      if char == 'A':
        start = (x, y)
      elif char == 'B':
        end = (x, y)

  return {
    "width": width,
    "height": height,
    "start": start,
    "end": end,
  }


# Pre-process mazes
MAZE_INFO = [get_maze_info(maze) for maze in MAZES]


def prepareSubpassPrompt(subPass: int) -> str:
  """Generate the prompt for subpass 0 that handles all maze sizes."""
  if subPass != 0:
    raise StopIteration

  return f"""You are writing a maze solver in Rust that must handle ANY maze complexity from trivial to ludicrous scale.

Your program will be tested with mazes ranging from 5x5 to 50000x50000.
The same program must work efficiently across ALL scales.

**Maze format:**
- 'A' = Start position
- 'B' = Goal/End position
- '#' = Wall (cannot pass through)
- ' ' (space) = Open path (can walk through)

**Input format (stdin):**
Line 1: H W (height and width)
Next H lines: the maze row strings (each W characters)

**Output format (stdout):**
Line 1: N (number of steps in path)
Next N lines: x y (0-indexed coordinates from top-left)
Path should include both start (A) and end (B) positions.
Movement is only up/down/left/right (no diagonals).
Output nothing (or 0) if no path exists.

**Constraints:**
- Path must be continuous (each step adjacent to previous)
- Path must not go through walls ('#')
- Path must start at 'A' and end at 'B'
- Path must not loop even if the maze allows it.
- If multiple solutions exist, return any one.

**Environment:**
{describe_this_pc()}

**Rust Compiler:**
{RustCompiler("test_engine").describe()}

Write complete, compilable Rust code with a main() function.
"""


# List of subpasses to grade the single answer against all difficulty levels
extraGradeAnswerRuns = list(range(len(MAZES) + len(EXTREME_MAZE_SIZES)))

structure = {
  "type": "object",
  "properties": {
    "reasoning": {
      "type": "string",
      "description": "Explain your maze-solving algorithm and how it adapts to different maze sizes"
    },
    "rust_code": {
      "type": "string",
      "description": "Complete Rust code with main() that handles all scales"
    }
  },
  "required": ["reasoning", "rust_code"],
  "additionalProperties": False
}


def parse_maze(maze_string: str) -> Tuple[List[str], Tuple[int, int], Tuple[int, int]]:
  """Parse maze string into grid and find start/end positions."""
  lines = maze_string.strip().split('\n')
  start = None
  end = None

  for y, line in enumerate(lines):
    for x, char in enumerate(line):
      if char == 'A':
        start = (x, y)
      elif char == 'B':
        end = (x, y)

  return lines, start, end


def validate_path(maze_string: str, path: List[Tuple[int, int]]) -> Tuple[bool, str]:
  """Validate that a path is valid for the maze."""
  if not isinstance(path, list):
    return False, f"Path must be a list, got {type(path).__name__}"

  if len(path) == 0:
    return False, "Empty path"

  lines, start, end = parse_maze(maze_string)
  height = len(lines)

  # Check start position
  if tuple(path[0]) != start:
    return False, f"Path must start at A {start}, got {path[0]}"

  # Check end position
  if tuple(path[-1]) != end:
    return False, f"Path must end at B {end}, got {path[-1]}"

  # Check each position
  for i, pos in enumerate(path):
    if not isinstance(pos, (list, tuple)) or len(pos) != 2:
      return False, f"Position {i} must be (x, y), got {pos}"

    x, y = int(pos[0]), int(pos[1])

    # Bounds check
    if y < 0 or y >= height:
      return False, f"Position {i}: y={y} out of bounds"
    if x < 0 or x >= len(lines[y]):
      return False, f"Position {i}: x={x} out of bounds"

    # Wall check
    char = lines[y][x]
    if char == '#':
      return False, f"Position {i}: ({x}, {y}) is a wall"

    # Adjacency check (except for first position)
    if i > 0:
      prev_x, prev_y = int(path[i - 1][0]), int(path[i - 1][1])
      dx = abs(x - prev_x)
      dy = abs(y - prev_y)
      if dx + dy != 1:
        return False, f"Position {i}: ({x}, {y}) not adjacent to ({prev_x}, {prev_y})"

  return True, ""


STREAMING_THRESHOLD_CHARS = 500_000
_INPUT_FILE_CACHE = {}


def format_input(maze_string: str) -> str:
  lines = maze_string.strip().split('\n')
  height = len(lines)
  width = max(len(line) for line in lines)
  result_lines = [f"{height} {width}"]
  for line in lines:
    result_lines.append(line.ljust(width))
  return "\n".join(result_lines)


def _get_streaming_input(subpass: int, maze_string: str) -> StreamingInputFile:
  if subpass in _INPUT_FILE_CACHE:
    return _INPUT_FILE_CACHE[subpass]

  lines = maze_string.strip().split('\n')
  height = len(lines)
  width = max(len(line) for line in lines)
  cache_key = f"maze8|h={height}|w={width}|seed={RANDOM_SEED + subpass}"

  def generator():
    yield f"{height} {width}\n"
    for line in lines:
      yield line.ljust(width) + "\n"

  input_file = StreamingInputFile(cache_key, generator, "test8_maze")
  _INPUT_FILE_CACHE[subpass] = input_file
  return input_file


def parse_path_output(output: str) -> tuple:
  text = output.strip()
  if not text:
    return [], None

  lines = [l for l in text.splitlines() if l.strip()]
  if not lines:
    return [], None

  try:
    n = int(lines[0])
  except ValueError:
    return None, "First line must be path length"

  if n == 0:
    return [], None

  if len(lines) < 1 + n:
    return None, f"Expected {n} coordinate lines, got {len(lines) - 1}"

  path = []
  for i in range(1, 1 + n):
    parts = lines[i].split()
    if len(parts) < 2:
      return None, f"Invalid coordinate line {i}"
    try:
      x, y = int(parts[0]), int(parts[1])
      path.append((x, y))
    except ValueError:
      return None, f"Non-integer coordinate at line {i}"

  return path, None


def execute_solver(code: str,
                   maze_string: str,
                   subpass: int,
                   ai_engine_name: str,
                   timeout: int = TIMEOUT_SECONDS) -> tuple:
  """Execute the LLM's solver. Returns (path, error, exec_time)."""
  if len(maze_string) > STREAMING_THRESHOLD_CHARS:
    streaming_input = _get_streaming_input(subpass, maze_string)
    input_file_path = streaming_input.generate()
    run = compile_and_run(code, "rust", ai_engine_name, input_file=input_file_path, timeout=timeout)
  else:
    input_data = format_input(maze_string)
    run = compile_and_run(code, "rust", ai_engine_name, input_data=input_data, timeout=timeout)

  if not run:
    return None, run.error_message(), run.exec_time

  path, parse_error = parse_path_output(run.stdout)
  if parse_error:
    return None, parse_error, run.exec_time

  return path, None, run.exec_time


lastPath = None


def threeJs_visualisation(path, width: int, height: int, start, end, maze_string: str) -> str:
  from visualization_utils import generate_threejs_maze_visualization
  return generate_threejs_maze_visualization(path,
                                             width,
                                             height,
                                             start,
                                             end,
                                             maze_string,
                                             name="Maze")


def gradeAnswer(result: dict, subPass: int, aiEngineName: str) -> tuple:
  if not result:
    return 0.0, "No result provided"

  if "rust_code" not in result:
    return 0.0, "No Rust code provided"

  maze = get_maze(subPass)
  info = get_maze_info(maze)
  code = result["rust_code"]

  # Execute solver
  path, error, exec_time = execute_solver(code, maze, subPass, aiEngineName)

  global lastPath
  lastPath = path

  if error:
    return 0.0, f"[{info['width']}x{info['height']}] {error}"

  # Validate path
  is_valid, validation_error = validate_path(maze, path)
  if not is_valid:
    return 0.0, f"[{info['width']}x{info['height']}] Invalid path: {validation_error}"

  quality = f"path length {len(path)}"

  explanation = (f"[{info['width']}x{info['height']}] Path found, "
                 f"Time: {exec_time:.1f}s - {quality}")

  return 1.0, explanation


def resultToNiceReport(result: dict, subPass: int, aiEngineName: str) -> str:
  """Generate HTML report."""
  if not result:
    return "<p style='color:red'>No result provided</p>"

  maze = get_maze(subPass)
  info = get_maze_info(maze)

  html = f"<h4>Maze Solver - {info['width']}x{info['height']}</h4>"

  if subPass == 0:
    if "reasoning" in result:
      reasoning = result['reasoning'][:500] + ('...'
                                               if len(result.get('reasoning', '')) > 500 else '')
      html += f"<p><strong>Algorithm:</strong> {reasoning}</p>"

    if "rust_code" in result:
      code = result["rust_code"]
      code_escaped = code.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
      html += f"<details><summary>View Code ({len(code)} chars)</summary><pre>{code_escaped}</pre></details>"

  if info['width'] <= 100:
    html += threeJs_visualisation(lastPath, info['width'], info['height'], info['start'],
                                  info['end'], maze)

  return html


highLevelSummary = """
<p>Find the shortest path through a maze from a start cell to an end cell.
The maze is a grid of walls and open passages, and the AI must navigate
from one corner to the other using the fewest steps possible.</p>
<p>Subpasses increase the maze size dramatically, testing whether the AI's
approach scales to very large mazes. The baseline finds the true shortest
path using breadth-first search.</p>
"""
