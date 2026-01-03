import subprocess
import sys
import tempfile
import os
import time
import random
from typing import List, Tuple, Set

title = "Maze Solver"

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
  sys.setrecursionlimit(max(10000, width * height))
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
]


def get_maze(subpass: int) -> str:
  """Get maze for subpass, generating extreme ones on demand."""
  if subpass < len(MAZES):
    if subpass not in _MAZE_CACHE:
      _MAZE_CACHE[subpass] = ensure_solvable_maze(MAZES[subpass])
    return _MAZE_CACHE[subpass]

  extreme_idx = subpass - len(MAZES)
  if extreme_idx < len(EXTREME_MAZE_SIZES):
    w, h = EXTREME_MAZE_SIZES[extreme_idx]
    return generate_random_maze(w, h, 0.3, RANDOM_SEED + subpass)

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

  return f"""You are writing a maze solver that must handle ANY maze complexity from trivial to ludicrous scale:

Your `solve_maze(maze_string)` function will be tested with mazes ranging from 5x5 to 50000x50000. 
The same function must work efficiently across ALL scales.

**Maze format:**
- 'A' = Start position
- 'B' = Goal/End position  
- '#' = Wall (cannot pass through)
- ' ' (space) = Open path (can walk through)

**Input:**
- `maze_string`: Multi-line string representing the maze

**Output:**
- List of (x, y) coordinates representing the path from A to B
- Coordinates are 0-indexed: (0,0) is top-left
- Path should include both start (A) and end (B) positions
- Movement is only up/down/left/right (no diagonals)

**Example output for a simple maze:**
```python
[(1, 1), (2, 1), (3, 1), (3, 2), (3, 3)]  # path from A to B
```

**Constraints:**
- Use only Python standard library and numpy
- Return empty list [] if no path exists
- Path must be continuous (each step adjacent to previous)
- Path must not go through walls ('#')
- Path must start at 'A' and end at 'B'
- Path must not loop even if the maze allows it.
- If multiple solutions exist, return any one.

Write complete, runnable Python code with the solve_maze function.
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
    "python_code": {
      "type":
      "string",
      "description":
      "Complete Python code with solve_maze(maze_string) function that handles all scales"
    }
  },
  "required": ["reasoning", "python_code"],
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


def execute_solver(code: str, maze_string: str, timeout: int = TIMEOUT_SECONDS) -> tuple:
  """Execute the LLM's solver. Returns (path, error, exec_time)."""
  from solver_utils import execute_solver_with_data

  data_dict = {
    'maze_string': maze_string,
  }

  result, error, exec_time = execute_solver_with_data(code, data_dict, 'solve_maze', timeout)

  if error:
    return None, error, exec_time

  if not isinstance(result, list):
    return None, f"Invalid output: expected list, got {type(result).__name__}", exec_time

  # Convert to tuples for consistency
  path = [tuple(p) for p in result]
  return path, None, exec_time


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

  if "python_code" not in result:
    return 0.0, "No Python code provided"

  maze = get_maze(subPass)
  info = get_maze_info(maze)
  code = result["python_code"]

  # Execute solver
  path, error, exec_time = execute_solver(code, maze)

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

    if "python_code" in result:
      code = result["python_code"]
      code_escaped = code.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
      html += f"<details><summary>View Code ({len(code)} chars)</summary><pre>{code_escaped}</pre></details>"

  html += threeJs_visualisation(lastPath, info['width'], info['height'], info['start'], info['end'],
                                maze)

  return html


highLevelSummary = """
Maze solving is a classic pathfinding problem.

**Algorithms:**
- **BFS (Breadth-First Search)**: Finds shortest path, O(V+E)
- **DFS (Depth-First Search)**: Finds a path (not shortest), O(V+E)
- **A* Search**: Optimal with admissible heuristic
- **Dijkstra**: For weighted graphs

**Key concepts:**
- Graph representation of maze
- Visited set to avoid cycles
- Queue (BFS) vs Stack (DFS)
- Manhattan distance heuristic for A*

The baseline uses simple BFS for optimal shortest paths.
"""
