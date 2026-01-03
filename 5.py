"""
Test 5: Hamiltonian Path/Cycle on 2D Grid with Obstacles

The LLM must write a Python solver that finds a Hamiltonian path (visiting every
non-obstacle cell exactly once) on a 2D grid. Optionally forms a cycle back to start.

Subpasses grow in grid size and number of obstacles.
Solver times out after 5 minutes.
"""

from pickle import FALSE
import random
import subprocess
import sys
import tempfile
import os
import time
from typing import List, Tuple, Set, Optional

title = "Hamiltonian Path on Grid with Obstacles"

# Seed for reproducibility
RANDOM_SEED = 77777

# Timeout in seconds (30 seconds)
TIMEOUT_SECONDS = 30

# Grid configurations: (width, height, num_obstacles, require_cycle)
GRID_CONFIGS = [
  (4, 4, 2, False),  # Small grid, few obstacles
  (5, 5, 4, False),  # Medium grid
  (6, 5, 6, False),  # 
  (6, 6, 8, False),  # Larger grid
  (7, 7, 12, False),  # Large with cycle requirement
  (8, 8, 16, False),  # Large grid, many obstacles
  (10, 10, 25, False),  # Extreme 1 - 100 cells
  (15, 15, 50, False),  # Extreme 2 - 225 cells
  (20, 20, 100, False),  # Extreme 3 - 400 cells
  (30, 30, 200, False),  # Extreme 4 - 900 cells
  (50, 50, 500, False),  # Extreme 5 - 2500 cells, NP-hard territory
]


def generate_obstacles_constructive(width: int,
                                    height: int,
                                    max_obstacles: int,
                                    seed: int,
                                    require_cycle: bool = False) -> Set[Tuple[int, int]]:
  """
  Generate obstacles by walking a random path covering most of the grid,
  then marking unvisited cells as obstacles.
  
  This approach guarantees solvability by construction.
  """
  rng = random.Random(seed)

  # Start with all cells free
  all_cells = [(x, y) for x in range(width) for y in range(height)]
  free_cells = set(all_cells)

  # Walk a random path covering most of the grid
  path = random_walk_cover(width, height, (0, 0), rng, require_cycle)

  # Mark cells not in the path as obstacles
  path_set = set(path)
  obstacles = set(all_cells) - path_set

  return obstacles


def random_walk_cover(width: int, height: int, start: Tuple[int, int], rng: random.Random,
                      require_cycle: bool) -> List[Tuple[int, int]]:
  """Walk a random path covering most of the grid with backtracking at dead ends."""
  all_cells = {(x, y) for x in range(width) for y in range(height)}
  visited = {start}
  path = [start]
  current = start
  target_coverage = 0.9  # Cover 90% of the grid

  # Continue until we cover most cells or get stuck
  while len(visited) < len(all_cells) * target_coverage:
    # Get unvisited neighbors
    neighbors = []
    for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
      nx, ny = current[0] + dx, current[1] + dy
      neighbor = (nx, ny)
      if (0 <= nx < width and 0 <= ny < height and neighbor not in visited):
        neighbors.append(neighbor)

    if neighbors:
      # Choose a random neighbor
      next_cell = rng.choice(neighbors)
      path.append(next_cell)
      visited.add(next_cell)
      current = next_cell
    else:
      # Dead end - backtrack to find unvisited cells
      if len(path) > 1:
        # Remove current position and backtrack
        path.pop()
        if path:
          current = path[-1]
        else:
          break  # No more backtracking possible
      else:
        break  # Stuck at start with no moves

  # If cycle is required, try to adjust the path
  if require_cycle and len(path) > 3:
    # Try to find a way back to start
    end = path[-1]
    if are_adjacent(end, start):
      # Already a cycle - done!
      return path
    else:
      # Try to find a simple modification to make it a cycle
      return adjust_path_to_cycle(path, width, height, rng)

  return path


def adjust_path_to_cycle(path: List[Tuple[int, int]], width: int, height: int,
                         rng: random.Random) -> List[Tuple[int, int]]:
  """Try to adjust a path to form a cycle by finding shortcuts."""
  if len(path) < 4:
    return path  # Too short to make a meaningful cycle

  start = path[0]

  # Look for opportunities to create cycles by finding adjacent cells
  for i in range(len(path) - 2):
    for j in range(i + 2, len(path)):
      if are_adjacent(path[i], path[j]):
        # Found a potential cycle - create a loop
        new_path = path[:i + 1] + path[j:]
        if len(new_path) >= 3 and are_adjacent(new_path[-1], new_path[0]):
          return new_path

  # If no simple cycle found, just return the original path
  return path


def construct_hamiltonian_path(width: int, height: int, free_cells: Set[Tuple[int, int]],
                               start: Tuple[int, int], require_cycle: bool,
                               rng: random.Random) -> List[Tuple[int, int]]:
  """
  Construct a Hamiltonian path/cycle using a randomized backtracking approach.
  """
  if not free_cells:
    return []

  # For small grids, use simple construction
  if width * height <= 100:
    return construct_simple_path(width, height, free_cells, start, require_cycle, rng)

  # For larger grids, use a greedy approach with backtracking
  return construct_greedy_path(width, height, free_cells, start, require_cycle, rng)


def construct_simple_path(width: int, height: int, free_cells: Set[Tuple[int, int]],
                          start: Tuple[int, int], require_cycle: bool,
                          rng: random.Random) -> List[Tuple[int, int]]:
  """Simple construction for small grids using backtracking with limits."""
  target_length = len(free_cells)
  path = [start]
  visited = {start}
  max_attempts = target_length * 100  # Prevent infinite recursion
  attempts = 0

  def dfs(current: Tuple[int, int]) -> bool:
    nonlocal attempts
    attempts += 1

    if attempts > max_attempts:
      return False  # Give up to prevent infinite recursion

    if len(path) == target_length:
      # Check cycle requirement
      if require_cycle:
        return are_adjacent(current, start)
      return True

    # Get neighbors in random order
    neighbors = []
    for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
      nx, ny = current[0] + dx, current[1] + dy
      neighbor = (nx, ny)
      if (0 <= nx < width and 0 <= ny < height and neighbor in free_cells
          and neighbor not in visited):
        neighbors.append(neighbor)

    rng.shuffle(neighbors)

    for neighbor in neighbors:
      path.append(neighbor)
      visited.add(neighbor)

      if dfs(neighbor):
        return True

      # Backtrack
      path.pop()
      visited.remove(neighbor)

    return False

  if dfs(start):
    return path
  return []


def construct_greedy_path(width: int, height: int, free_cells: Set[Tuple[int, int]],
                          start: Tuple[int, int], require_cycle: bool,
                          rng: random.Random) -> List[Tuple[int, int]]:
  """Greedy construction for larger grids with some backtracking."""
  path = [start]
  visited = {start}
  current = start
  target_length = len(free_cells)
  max_steps = target_length * 10  # Prevent infinite loops
  steps = 0

  while len(path) < target_length and steps < max_steps:
    steps += 1

    # Get unvisited neighbors
    neighbors = []
    for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
      nx, ny = current[0] + dx, current[1] + dy
      neighbor = (nx, ny)
      if (0 <= nx < width and 0 <= ny < height and neighbor in free_cells
          and neighbor not in visited):
        neighbors.append(neighbor)

    if not neighbors:
      # Dead end - try some limited backtracking
      if len(path) > 1:
        # Backtrack a few steps and try different direction
        backtrack_steps = min(5, len(path) - 1)
        for _ in range(backtrack_steps):
          visited.remove(path.pop())

        if path:
          current = path[-1]
          continue
      else:
        break

    # Choose neighbor with some heuristics
    if len(neighbors) == 1:
      next_cell = neighbors[0]
    else:
      # Prefer neighbors with fewer unvisited neighbors (Warnsdorff's rule)
      def count_unvisited_neighbors(cell):
        count = 0
        for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
          nx, ny = cell[0] + dx, cell[1] + dy
          neighbor = (nx, ny)
          if (0 <= nx < width and 0 <= ny < height and neighbor in free_cells
              and neighbor not in visited):
            count += 1
        return count

      neighbors.sort(key=count_unvisited_neighbors)
      next_cell = neighbors[0]

    path.append(next_cell)
    visited.add(next_cell)
    current = next_cell

  # Check if we succeeded
  if len(path) == target_length:
    if require_cycle and not are_adjacent(current, start):
      # Try to fix the cycle by local adjustments
      return adjust_to_cycle(path, width, height, free_cells, rng)
    return path

  return []


def adjust_to_cycle(path: List[Tuple[int, int]], width: int, height: int,
                    free_cells: Set[Tuple[int, int]], rng: random.Random) -> List[Tuple[int, int]]:
  """Try to adjust a path to form a cycle."""
  if len(path) < 3:
    return []

  # Try to find a way to connect end to start
  start, end = path[0], path[-1]

  # Look for a simple 2-opt move
  for i in range(1, len(path) - 1):
    for j in range(i + 1, len(path)):
      if are_adjacent(path[i - 1], path[j]) and are_adjacent(path[i], path[j - 1]):
        # Found a 2-opt improvement
        new_path = path[:i] + path[i:j][::-1] + path[j:]
        if are_adjacent(new_path[-1], new_path[0]):
          return new_path

  return []


def are_adjacent(cell1: Tuple[int, int], cell2: Tuple[int, int]) -> bool:
  """Check if two cells are adjacent."""
  return abs(cell1[0] - cell2[0]) + abs(cell1[1] - cell2[1]) == 1


def generate_obstacles_simple(width: int,
                              height: int,
                              num_obstacles: int,
                              seed: int,
                              require_cycle: bool = False) -> Set[Tuple[int, int]]:
  """Fallback simple obstacle generation."""
  rng = random.Random(seed)
  obstacles = set()

  # Never block (0, 0) - the start position
  all_cells = [(x, y) for x in range(width) for y in range(height) if (x, y) != (0, 0)]
  rng.shuffle(all_cells)

  for cell in all_cells[:num_obstacles]:
    obstacles.add(cell)

  return obstacles


def generate_obstacles(width: int, height: int, num_obstacles: int,
                       seed: int) -> Set[Tuple[int, int]]:
  """
  Generate obstacle positions ensuring a path is still possible.
  Uses constructive approach: build a valid path first, then mark unused cells as obstacles.
  """
  return generate_obstacles_constructive(width, height, num_obstacles, seed, False)


def is_connected(width: int, height: int, obstacles: Set[Tuple[int, int]]) -> bool:
  """Check if non-obstacle cells form a connected region."""
  start = None
  for x in range(width):
    for y in range(height):
      if (x, y) not in obstacles:
        start = (x, y)
        break
    if start:
      break

  if not start:
    return False

  # BFS to count reachable cells
  visited = {start}
  queue = [start]

  while queue:
    x, y = queue.pop(0)
    for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
      nx, ny = x + dx, y + dy
      if (0 <= nx < width and 0 <= ny < height and (nx, ny) not in obstacles
          and (nx, ny) not in visited):
        visited.add((nx, ny))
        queue.append((nx, ny))

  total_free = width * height - len(obstacles)
  return len(visited) == total_free


def has_hamiltonian_path_potential(width: int,
                                   height: int,
                                   obstacles: Set[Tuple[int, int]],
                                   require_cycle: bool = False) -> bool:
  """Advanced check for Hamiltonian path feasibility."""
  # 1. Check connectivity
  if not is_connected(width, height, obstacles):
    return False

  # 2. Check parity balance
  black_count = 0
  white_count = 0

  for x in range(width):
    for y in range(height):
      if (x, y) not in obstacles:
        if (x + y) % 2 == 0:
          black_count += 1
        else:
          white_count += 1

  # For Hamiltonian path: difference at most 1
  # For Hamiltonian cycle: counts must be equal
  if require_cycle:
    if black_count != white_count:
      return False
  else:
    if abs(black_count - white_count) > 1:
      return False

  # 3. Check for dead ends (cells with only 1 neighbor)
  dead_ends = []
  free_cells = [(x, y) for x in range(width) for y in range(height) if (x, y) not in obstacles]

  for x, y in free_cells:
    neighbors = 0
    for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
      nx, ny = x + dx, y + dy
      if (0 <= nx < width and 0 <= ny < height and (nx, ny) not in obstacles):
        neighbors += 1

    if neighbors == 1:
      dead_ends.append((x, y))

  # For Hamiltonian path: at most 2 dead ends (start and end)
  # For Hamiltonian cycle: NO dead ends allowed
  if require_cycle:
    if len(dead_ends) > 0:
      return False
  else:
    if len(dead_ends) > 2:
      return False

  # 4. For cycles, check for 1-wide bottlenecks (articulation points)
  if require_cycle:
    if has_articulation_point(width, height, obstacles):
      return False

  # 5. Check for forced paths that create contradictions
  # This is the critical check for the pattern you identified
  if has_forced_path_contradiction(width, height, obstacles, dead_ends, require_cycle):
    return False

  # 6. Advanced checks for subtle unsolvable patterns
  if require_cycle:
    # Check for degree-2 cycles that are too small
    if has_small_degree2_cycle(width, height, obstacles):
      return False
    # Check for separated regions that can only connect through one point
    if has_separation_constraint(width, height, obstacles):
      return False
  else:
    # For paths, check for trapped regions
    if has_trapped_region(width, height, obstacles):
      return False

  return True


def has_small_degree2_cycle(width: int, height: int, obstacles: Set[Tuple[int, int]]) -> bool:
  """Check for small cycles formed by degree-2 vertices that prevent Hamiltonian cycles."""
  free_cells = {(x, y) for x in range(width) for y in range(height) if (x, y) not in obstacles}

  # Create adjacency map
  neighbors = {}
  degree2_vertices = []

  for x, y in free_cells:
    neighbors[(x, y)] = []
    for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
      nx, ny = x + dx, y + dy
      if (0 <= nx < width and 0 <= ny < height and (nx, ny) not in obstacles):
        neighbors[(x, y)].append((nx, ny))

    if len(neighbors[(x, y)]) == 2:
      degree2_vertices.append((x, y))

  # Look for small cycles formed by degree-2 vertices
  visited = set()

  for start in degree2_vertices:
    if start in visited:
      continue

    # Follow the degree-2 chain
    current = start
    path = [current]
    visited.add(current)

    while True:
      next_cells = [n for n in neighbors[current] if n not in path]
      if len(next_cells) != 1:
        break

      current = next_cells[0]
      path.append(current)
      visited.add(current)

      # Check if we've formed a cycle
      if current in neighbors[start] and len(path) > 2:
        # Found a cycle - check if it's too small
        if len(path) < len(free_cells):
          # Small cycle formed by degree-2 vertices - impossible for Hamiltonian cycle
          return True

      if len(path) > len(free_cells):
        break

  return False


def has_separation_constraint(width: int, height: int, obstacles: Set[Tuple[int, int]]) -> bool:
  """Check if the grid can be separated into regions that only connect through constrained points."""
  free_cells = {(x, y) for x in range(width) for y in range(height) if (x, y) not in obstacles}

  # Try to find a cut that separates the grid into two regions
  # with limited connection points

  # Simple heuristic: look for "bridge" patterns
  for x, y in free_cells:
    # Check if this cell is a bridge between two large regions
    neighbor_count = 0
    for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
      nx, ny = x + dx, y + dy
      if (0 <= nx < width and 0 <= ny < height and (nx, ny) not in obstacles):
        neighbor_count += 1

    if neighbor_count == 2:
      # This could be a bridge - check if removing it disconnects the grid
      temp_obstacles = obstacles | {(x, y)}
      if not is_connected(width, height, temp_obstacles):
        # Check the sizes of the resulting components
        components = get_connected_components(width, height, temp_obstacles)
        if len(components) == 2:
          # If both components are substantial, this creates a separation constraint
          comp1_size, comp2_size = len(components[0]), len(components[1])
          if min(comp1_size, comp2_size) >= 3:
            return True

  return False


def has_trapped_region(width: int, height: int, obstacles: Set[Tuple[int, int]]) -> bool:
  """Check for trapped regions that can only be entered/exited through one cell."""
  free_cells = {(x, y) for x in range(width) for y in range(height) if (x, y) not in obstacles}

  # For each cell, check if it's the only entrance to a region
  for x, y in free_cells:
    # Temporarily remove this cell and check connectivity
    temp_obstacles = obstacles | {(x, y)}
    components = get_connected_components(width, height, temp_obstacles)

    if len(components) > 1:
      # This cell connects multiple regions
      # Check if any region is "trapped" (small and only accessible through this cell)
      for component in components:
        if len(component) >= 2 and len(component) <= 4:
          # Small region that's only accessible through this cell
          # Check if all cells in this region have degree <= 2
          trapped = True
          for cx, cy in component:
            degree = 0
            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
              nx, ny = cx + dx, cy + dy
              if (0 <= nx < width and 0 <= ny < height and (nx, ny) not in obstacles):
                degree += 1
            if degree > 2:
              trapped = False
              break

          if trapped:
            return True

  return False


def get_connected_components(width: int, height: int,
                             obstacles: Set[Tuple[int, int]]) -> List[Set[Tuple[int, int]]]:
  """Get all connected components of the grid."""
  free_cells = {(x, y) for x in range(width) for y in range(height) if (x, y) not in obstacles}
  components = []
  visited = set()

  for start in free_cells:
    if start in visited:
      continue

    # BFS to find this component
    component = set()
    queue = [start]
    visited.add(start)

    while queue:
      x, y = queue.pop(0)
      component.add((x, y))

      for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
        nx, ny = x + dx, y + dy
        if (0 <= nx < width and 0 <= ny < height and (nx, ny) not in obstacles
            and (nx, ny) not in visited):
          visited.add((nx, ny))
          queue.append((nx, ny))

    components.append(component)

  return components


def has_articulation_point(width: int, height: int, obstacles: Set[Tuple[int, int]]) -> bool:
  """Check if the graph has an articulation point (1-wide bottleneck)."""
  free_cells = {(x, y) for x in range(width) for y in range(height) if (x, y) not in obstacles}

  if len(free_cells) < 3:
    return False  # Too small to have meaningful articulation points

  # Create adjacency map
  neighbors = {}
  for x, y in free_cells:
    neighbors[(x, y)] = []
    for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
      nx, ny = x + dx, y + dy
      if (0 <= nx < width and 0 <= ny < height and (nx, ny) not in obstacles):
        neighbors[(x, y)].append((nx, ny))

  # Use DFS to find articulation points
  visited = {}
  discovery_time = {}
  low = {}
  parent = {}
  time = 0
  articulation_points = set()

  def dfs(u: Tuple[int, int], root: bool = False) -> None:
    nonlocal time
    visited[u] = True
    discovery_time[u] = time
    low[u] = time
    time += 1
    children = 0

    for v in neighbors[u]:
      if v not in visited:
        parent[v] = u
        children += 1
        dfs(v, False)

        # Update low value
        low[u] = min(low[u], low[v])

        # Check for articulation point
        if not root and low[v] >= discovery_time[u]:
          articulation_points.add(u)
        elif root and children > 1:
          articulation_points.add(u)
      elif v != parent.get(u, None):
        # Back edge
        low[u] = min(low[u], discovery_time[v])

  # Run DFS from each unvisited node (handle disconnected graphs, though we checked connectivity)
  for cell in free_cells:
    if cell not in visited:
      dfs(cell, True)

  return len(articulation_points) > 0


def has_forced_path_contradiction(width: int, height: int, obstacles: Set[Tuple[int, int]],
                                  dead_ends: List[Tuple[int, int]], require_cycle: bool) -> bool:
  """Check for forced path patterns that make Hamiltonian path impossible."""
  free_cells = {(x, y) for x in range(width) for y in range(height) if (x, y) not in obstacles}

  # Create adjacency map
  neighbors = {}
  for x, y in free_cells:
    neighbors[(x, y)] = []
    for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
      nx, ny = x + dx, y + dy
      if (0 <= nx < width and 0 <= ny < height and (nx, ny) not in obstacles):
        neighbors[(x, y)].append((nx, ny))

  # Check for the specific pattern you identified:
  # Dead end -> neighbor -> sequence of degree-2 cells -> another dead end
  if len(dead_ends) == 2:
    for dead_end in dead_ends:
      # Check if this dead end leads to the other through forced path
      if has_forced_path_between(dead_ends[0], dead_ends[1], neighbors, free_cells):
        return True

  # More generally, check for any forced path that would require revisiting cells
  for start in free_cells:
    visited = set([start])
    path = [start]
    if follows_forced_path(start, visited, path, neighbors, free_cells):
      # If we return to a visited cell before covering all cells, it's impossible
      if len(visited) < len(free_cells):
        return True

  return False


def has_forced_path_between(start: Tuple[int, int], end: Tuple[int, int], neighbors: dict,
                            free_cells: Set[Tuple[int, int]]) -> bool:
  """Check if there's a forced path between two dead ends."""
  if start == end:
    return False

  visited = set([start])
  current = start

  while current != end:
    # Find next cell in forced path
    next_cells = [n for n in neighbors[current] if n not in visited]

    # If we have a choice, path isn't forced
    if len(next_cells) != 1:
      return False

    current = next_cells[0]
    visited.add(current)

    # If we've visited too many cells without reaching the end
    if len(visited) > len(free_cells):
      return False

  return True


def follows_forced_path(current: Tuple[int, int], visited: Set[Tuple[int, int]],
                        path: List[Tuple[int, int]], neighbors: dict,
                        free_cells: Set[Tuple[int, int]]) -> bool:
  """Check if we're following a forced path that leads to contradiction."""
  # Get unvisited neighbors
  unvisited_neighbors = [n for n in neighbors[current] if n not in visited]

  # If no unvisited neighbors and we haven't covered all cells, this is a dead end
  if len(unvisited_neighbors) == 0:
    return len(visited) < len(free_cells)

  # If more than 1 unvisited neighbor, path isn't forced here
  if len(unvisited_neighbors) > 1:
    return False

  # Follow the forced path
  next_cell = unvisited_neighbors[0]
  visited.add(next_cell)
  path.append(next_cell)

  # Check if we've created a cycle before covering all cells
  if len(visited) < len(free_cells):
    # Check if next_cell has been visited before (cycle detection)
    if next_cell in path[:-1]:
      return True

  return follows_forced_path(next_cell, visited, path, neighbors, free_cells)


# Pre-generate grids ensuring they're solvable
GRIDS_CACHE = {}
for i, (w, h, num_obs, cycle) in enumerate(GRID_CONFIGS):
  seed = RANDOM_SEED + i * 100
  obstacles = generate_obstacles_constructive(w, h, num_obs, seed, cycle)
  #print(f"Generated {w}x{h} grid with {num_obs} obstacles (cycle={cycle})")
  # Store the configuration
  GRIDS_CACHE[i] = (w, h, obstacles, cycle)


def format_grid_visual(width: int, height: int, obstacles: Set[Tuple[int, int]]) -> str:
  """Create ASCII visualization of the grid."""
  lines = []
  for y in range(height - 1, -1, -1):  # Top to bottom
    row = ""
    for x in range(width):
      if (x, y) == (0, 0):
        row += "S "  # Start
      elif (x, y) in obstacles:
        row += "# "  # Obstacle
      else:
        row += ". "  # Free cell
    lines.append(row)
  return "\n".join(lines)


def format_obstacles_for_prompt(obstacles: Set[Tuple[int, int]]) -> str:
  """Format obstacles as a set literal."""
  sorted_obs = sorted(obstacles)
  if len(sorted_obs) <= 20:
    return "{" + ", ".join(f"({x}, {y})" for x, y in sorted_obs) + "}"
  else:
    preview = sorted_obs[:15]
    return "{" + ", ".join(f"({x}, {y})"
                           for x, y in preview) + f", ... ({len(sorted_obs)} total)" + "}"


def prepareSubpassPrompt(subPass: int) -> str:
  """Generate the prompt for subpass 0 that handles all grid sizes."""
  if subPass != 0:
    raise StopIteration

  return f"""You are solving a Hamiltonian Path problem on 2D grids with obstacles.

You must write a Python solver that can handle ANY grid size from trivial to ludicrous scale:

Your `find_hamiltonian_path(width, height, obstacles, require_cycle)` function will be tested with grids ranging from 4x4 to 50x50. 
The same function must work efficiently across ALL scales.

**Input:**
- `width`, `height`: Grid dimensions
- `obstacles`: Set of (x, y) tuples for blocked cells
- `require_cycle`: Boolean - if True, must form cycle back to start

**Output:**
- List of (x, y) coordinates representing the path
- Path must start at (0, 0)
- Path must visit every non-obstacle cell exactly once
- Movement is only to adjacent cells (up/down/left/right)
- If require_cycle is True, last cell must be adjacent to (0, 0)

**Critical Requirements:**
1. **Scalability**: Your algorithm must adapt based on grid size and obstacle density
2. **Performance**: Must complete within 5 minutes even for 2500 cell grids
3. **Correctness**: Must find valid paths or correctly report impossibility

**Constraints:**
- Use only Python standard library
- Return empty list [] if no path exists
- Must handle both path-only and cycle requirements

Write complete, runnable Python code with the find_hamiltonian_path function.
Include adaptive logic that chooses different strategies based on grid size.
"""


# List of subpasses to grade the single answer against all difficulty levels
extraGradeAnswerRuns = list(range(1, len(GRID_CONFIGS)))

structure = {
  "type": "object",
  "properties": {
    "reasoning": {
      "type":
      "string",
      "description":
      "Explain your approach to finding Hamiltonian paths and how it adapts to different grid sizes"
    },
    "python_code": {
      "type":
      "string",
      "description":
      "Complete Python code with find_hamiltonian_path(width, height, obstacles, require_cycle) function that handles all scales"
    }
  },
  "required": ["reasoning", "python_code"],
  "additionalProperties": False
}


def validate_path(path: List[Tuple[int, int]], width: int, height: int,
                  obstacles: Set[Tuple[int, int]], require_cycle: bool) -> Tuple[bool, str]:
  """
    Validate a Hamiltonian path.
    Returns (is_valid, error_message).
    """
  if not isinstance(path, list):
    return False, f"Path must be a list, got {type(path).__name__}"

  if len(path) == 0:
    return False, "Empty path returned"

  # Check start
  if path[0] != (0, 0):
    return False, f"Path must start at (0, 0), got {path[0]}"

  # Expected path length
  expected_len = width * height - len(obstacles)
  if len(path) != expected_len:
    return False, f"Path length {len(path)} != expected {expected_len} free cells"

  # Check each cell
  visited = set()
  for i, (x, y) in enumerate(path):
    # Valid coordinates
    if not (0 <= x < width and 0 <= y < height):
      return False, f"Cell {i}: ({x}, {y}) is out of bounds"

    # Not an obstacle
    if (x, y) in obstacles:
      return False, f"Cell {i}: ({x}, {y}) is an obstacle"

    # Not visited before
    if (x, y) in visited:
      return False, f"Cell {i}: ({x}, {y}) is visited twice"

    visited.add((x, y))

    # Check adjacency (except for first cell)
    if i > 0:
      px, py = path[i - 1]
      if abs(x - px) + abs(y - py) != 1:
        return False, f"Cell {i}: ({x}, {y}) is not adjacent to ({px}, {py})"

  # Check cycle requirement
  if require_cycle:
    last_x, last_y = path[-1]
    if abs(last_x - 0) + abs(last_y - 0) != 1:
      return False, f"Cycle required but last cell ({last_x}, {last_y}) not adjacent to (0, 0)"

  return True, ""


def execute_solver(code: str,
                   width: int,
                   height: int,
                   obstacles: Set[Tuple[int, int]],
                   require_cycle: bool,
                   timeout: int = TIMEOUT_SECONDS) -> tuple:
  """Execute the LLM's solver. Returns (path, error, exec_time)."""
  from solver_utils import execute_solver_with_data

  # Use the common utility with debugger isolation
  data_dict = {
    'width': width,
    'height': height,
    'obstacles': obstacles,
    'require_cycle': require_cycle
  }

  result, error, exec_time = execute_solver_with_data(code, data_dict, 'find_hamiltonian_path',
                                                      timeout)

  if error:
    return None, error, exec_time

  if result is None:
    return None, "Solver returned no result", exec_time

  # Convert to tuples for consistency
  path = [tuple(p) for p in result]
  return path, None, exec_time


lastPath = None


def gradeAnswer(result: dict, subPass: int, aiEngineName: str) -> tuple:
  """
    Grade the Hamiltonian path solver.
    
    Scoring:
    - 1.0: Valid complete Hamiltonian path found
    - 0.5: Partial path (visits >50% of cells correctly)
    - 0.0: No valid path or error
    """
  if not result:
    return 0.0, "No result provided"

  if "python_code" not in result:
    return 0.0, "No Python code provided"

  width, height, obstacles, require_cycle = GRIDS_CACHE[subPass]
  code = result["python_code"]

  # Execute solver
  path, error, exec_time = execute_solver(code, width, height, obstacles, require_cycle)

  global lastPath
  lastPath = path

  if error:
    return 0.0, f"[{width}x{height}, {len(obstacles)} obstacles] {error}"

  # Validate path
  is_valid, validation_error = validate_path(path, width, height, obstacles, require_cycle)

  if is_valid:
    cycle_str = " (cycle)" if require_cycle else ""
    return 1.0, f"[{width}x{height}, {len(obstacles)} obstacles] Valid path{cycle_str}, {len(path)} cells, Time: {exec_time:.1f}s"

  # Check for partial credit
  if path and len(path) > 0:
    expected_len = width * height - len(obstacles)
    valid_steps = 0
    visited = set()
    prev = None

    for x, y in path:
      if (0 <= x < width and 0 <= y < height and (x, y) not in obstacles and (x, y) not in visited):
        if prev is None or abs(x - prev[0]) + abs(y - prev[1]) == 1:
          valid_steps += 1
          visited.add((x, y))
          prev = (x, y)
        else:
          break
      else:
        break

    if valid_steps > expected_len * 0.5:
      return 0.5, f"[{width}x{height}] Partial path: {valid_steps}/{expected_len} cells valid. {validation_error}"

  return 0.0, f"[{width}x{height}, {len(obstacles)} obstacles] Invalid: {validation_error}"


def create_svg_representation(path: List[Tuple[int, int]], width: int, height: int,
                              obstacles: Set[Tuple[int, int]], require_cycle: bool) -> str:
  """Create SVG visualization of the Hamiltonian path."""

  # Scale factor for visualization
  cell_size = min(600 // max(width, height), 40)
  margin = 20

  svg_width = width * cell_size + 2 * margin
  svg_height = height * cell_size + 2 * margin

  # Create SVG elements
  svg_elements = []

  if not path: path = []

  # Background
  svg_elements.append(
    f'<rect width="{svg_width}" height="{svg_height}" fill="#f8f8f8" stroke="#ccc" stroke-width="2"/>'
  )

  # Grid lines
  for i in range(width + 1):
    x = margin + i * cell_size
    svg_elements.append(
      f'<line x1="{x}" y1="{margin}" x2="{x}" y2="{margin + height * cell_size}" stroke="#ddd" stroke-width="1"/>'
    )

  for i in range(height + 1):
    y = margin + i * cell_size
    svg_elements.append(
      f'<line x1="{margin}" y1="{y}" x2="{margin + width * cell_size}" y2="{y}" stroke="#ddd" stroke-width="1"/>'
    )

  # Obstacles
  for ox, oy in obstacles:
    x = margin + ox * cell_size
    y = margin + (height - 1 - oy) * cell_size  # Flip Y for SVG coordinates
    svg_elements.append(
      f'<rect x="{x + 2}" y="{y + 2}" width="{cell_size - 4}" height="{cell_size - 4}" fill="#333" rx="2"/>'
    )

  # Path
  if len(path) > 1:
    # Create path lines
    path_coords = []
    for i, (px, py) in enumerate(path):
      x = margin + px * cell_size + cell_size // 2
      y = margin + (height - 1 - py) * cell_size + cell_size // 2
      path_coords.append((x, y))

    # Draw path lines
    for i in range(len(path_coords) - 1):
      x1, y1 = path_coords[i]
      x2, y2 = path_coords[i + 1]
      # Color gradient along the path
      progress = i / max(len(path_coords) - 1, 1)
      red = int(255 * (1 - progress))
      green = int(100 + 155 * progress)
      color = f"rgb({red}, {green}, 100)"
      svg_elements.append(
        f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}" stroke-width="3" stroke-linecap="round"/>'
      )

    # Draw cycle connection if required
    if require_cycle and len(path_coords) > 2:
      x1, y1 = path_coords[-1]
      x2, y2 = path_coords[0]
      svg_elements.append(
        f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="#ff6b6b" stroke-width="3" stroke-dasharray="5,5" stroke-linecap="round"/>'
      )

  # Path points
  for i, (px, py) in enumerate(path):
    x = margin + px * cell_size + cell_size // 2
    y = margin + (height - 1 - py) * cell_size + cell_size // 2

    if i == 0:
      # Start point
      svg_elements.append(
        f'<circle cx="{x}" cy="{y}" r="6" fill="#4CAF50" stroke="#2E7D32" stroke-width="2"/>')
      svg_elements.append(
        f'<text x="{x}" y="{y - 10}" text-anchor="middle" font-size="12" font-weight="bold" fill="#2E7D32">S</text>'
      )
    elif i == len(path) - 1:
      # End point
      svg_elements.append(
        f'<circle cx="{x}" cy="{y}" r="6" fill="#FF5722" stroke="#D84315" stroke-width="2"/>')
      svg_elements.append(
        f'<text x="{x}" y="{y - 10}" text-anchor="middle" font-size="12" font-weight="bold" fill="#D84315">E</text>'
      )
    else:
      # Intermediate points
      progress = i / max(len(path) - 1, 1)
      red = int(255 * (1 - progress))
      green = int(100 + 155 * progress)
      color = f"rgb({red}, {green}, 100)"
      svg_elements.append(
        f'<circle cx="{x}" cy="{y}" r="3" fill="{color}" stroke="white" stroke-width="1"/>')

  # Path length indicator
  expected_len = width * height - len(obstacles)
  completeness = len(path) / expected_len if expected_len > 0 else 0

  svg_html = f'''
    <div style="margin: 15px 0; padding: 10px; border: 1px solid #ddd; border-radius: 4px; background: white;">
      <h5 style="margin: 0 0 10px 0; color: #333;">Path Visualization</h5>
      <svg width="{svg_width}" height="{svg_height}" style="border: 1px solid #ccc; background: white;">
        {"".join(svg_elements)}
      </svg>
      <div style="margin-top: 8px; font-size: 12px; color: #666;">
        🟢 Start (S) → 🔴 End (E) | Path: {len(path)}/{expected_len} cells ({completeness:.1%} complete)
        {' | Cycle completed' if require_cycle and len(path) == expected_len else ''}
      </div>
    </div>
  '''

  return svg_html


def resultToNiceReport(result: dict, subPass: int, aiEngineName: str) -> str:
  """Generate HTML report."""
  if not result:
    return "<p style='color:red'>No result provided</p>"

  width, height, obstacles, require_cycle = GRIDS_CACHE[subPass]

  html = f"<h4>Hamiltonian Path - {width}x{height} grid, {len(obstacles)} obstacles"
  if require_cycle:
    html += " (cycle required)"
  html += "</h4>"

  if subPass == 0:
    if "reasoning" in result:
      reasoning = result['reasoning'][:500] + ('...'
                                               if len(result.get('reasoning', '')) > 500 else '')
      html += f"<p><strong>Approach:</strong> {reasoning}</p>"

    if "python_code" in result:
      code = result["python_code"]
      code_escaped = code.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
      html += f"<details><summary>View Code ({len(code)} chars)</summary><pre>{code_escaped}</pre></details>"

  if not lastPath or len(lastPath) < 100:
    html += create_svg_representation(lastPath, width, height, obstacles, require_cycle)

  return html


highLevelSummary = """
Finding a Hamiltonian path on a grid with obstacles is NP-complete.

**Problem:**
- Visit every free cell exactly once
- Move only to adjacent cells (no diagonals)
- Start at (0, 0)
- Optionally return to start (Hamiltonian cycle)

**Approaches:**
- **Backtracking/DFS**: Basic exhaustive search
- **Warnsdorff's rule**: Prefer cells with fewer unvisited neighbors
- **Connectivity pruning**: Abandon paths that disconnect remaining cells
- **Dead-end detection**: Avoid creating isolated cells

**Complexity:**
- NP-complete in general
- Grid structure provides some exploitable properties
- Obstacles can make problems harder or easier

The baseline uses simple DFS without optimizations.
"""
