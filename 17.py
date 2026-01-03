"""
Test 17: 3D AABB Bin Packing

The LLM must write a Python solver that packs axis-aligned bounding boxes
into a container as efficiently as possible.

Given a container size and a list of box dimensions, find positions for
as many boxes as possible without overlap.

Subpasses test increasingly complex packing problems.
Solver times out after 5 minutes.
"""

import random
import subprocess
import sys
import tempfile
import os
import time
from typing import List, Tuple, Dict

title = "3D AABB Bin Packing"

# Timeout in seconds (5 minutes)
TIMEOUT_SECONDS = 30

# Seed for reproducibility
RANDOM_SEED = 99999


def generate_boxes(num_boxes: int, container: Tuple[int, int, int], min_size: int, max_size: int,
                   seed: int) -> List[Tuple[int, int, int]]:
  """Generate random box dimensions."""
  rng = random.Random(seed)
  boxes = []
  for _ in range(num_boxes):
    w = rng.randint(min_size, min(max_size, container[0]))
    h = rng.randint(min_size, min(max_size, container[1]))
    d = rng.randint(min_size, min(max_size, container[2]))
    boxes.append((w, h, d))
  return boxes


# Test configurations
TEST_CASES = [
  # Subpass 0: Simple - few boxes
  {
    "container": (10, 10, 10),
    "boxes": [(3, 3, 3), (4, 4, 4), (2, 5, 3), (3, 2, 4)],
    "description": "4 boxes in 10x10x10"
  },
  # Subpass 1: More boxes
  {
    "container": (10, 10, 10),
    "boxes": generate_boxes(8, (10, 10, 10), 2, 5, RANDOM_SEED),
    "description": "8 boxes in 10x10x10"
  },
  # Subpass 2: Larger container
  {
    "container": (15, 15, 15),
    "boxes": generate_boxes(12, (15, 15, 15), 2, 6, RANDOM_SEED + 1),
    "description": "12 boxes in 15x15x15"
  },
  # Subpass 3: Many small boxes
  {
    "container": (20, 20, 20),
    "boxes": generate_boxes(20, (20, 20, 20), 2, 5, RANDOM_SEED + 2),
    "description": "20 boxes in 20x20x20"
  },
  # Subpass 4: Mixed sizes
  {
    "container": (25, 25, 25),
    "boxes": generate_boxes(25, (25, 25, 25), 2, 8, RANDOM_SEED + 3),
    "description": "25 boxes in 25x25x25"
  },
  # Subpass 5: Large problem
  {
    "container": (30, 30, 30),
    "boxes": generate_boxes(35, (30, 30, 30), 2, 8, RANDOM_SEED + 4),
    "description": "35 boxes in 30x30x30"
  },
  # Extreme cases
  {
    "container": (50, 50, 50),
    "boxes": generate_boxes(100, (50, 50, 50), 2, 10, RANDOM_SEED + 5),
    "description": "100 boxes in 50x50x50"
  },
  {
    "container": (100, 100, 100),
    "boxes": generate_boxes(500, (100, 100, 100), 2, 15, RANDOM_SEED + 6),
    "description": "500 boxes in 100x100x100"
  },
  {
    "container": (200, 200, 200),
    "boxes": generate_boxes(1000, (200, 200, 200), 3, 20, RANDOM_SEED + 7),
    "description": "1000 boxes in 200x200x200"
  },
  {
    "container": (500, 500, 500),
    "boxes": generate_boxes(5000, (500, 500, 500), 5, 30, RANDOM_SEED + 8),
    "description": "5000 boxes in 500x500x500"
  },
  {
    "container": (1000, 1000, 1000),
    "boxes": generate_boxes(10000, (1000, 1000, 1000), 5, 50, RANDOM_SEED + 9),
    "description": "10000 boxes in 1000x1000x1000 (1B cubic units)"
  },
]


def format_boxes_for_prompt(boxes: List[Tuple[int, int, int]]) -> str:
  """Format boxes for prompt display."""
  if len(boxes) <= 15:
    return str(boxes)
  return f"[{', '.join(str(b) for b in boxes[:10])}, ... ({len(boxes)} total)]"


def prepareSubpassPrompt(subPass: int) -> str:
  """Generate the prompt for subpass 0 that handles all packing complexities."""
  if subPass != 0:
    raise StopIteration

  return f"""You are solving a 3D Bin Packing problem with axis-aligned boxes.

Need a function that works on all scales, from: 10,000 boxes in 1,000,000 cubic units,
down to 4 boxes in a 10x10x10

**The Challenge:**
Your `pack_boxes(boxes, container)` function will be tested with containers ranging 
from 5x5x5 to 500x500x500 and varying numbers of boxes. The same function must work efficiently 
across ALL scales, from small to very large problems.

**Input:**
- `boxes`: List of (width, height, depth) tuples
- `container`: (W, H, D) tuple for container dimensions
- Container origin at (0, 0, 0), extends to (W, H, D)

**Output:**
- Dict with:
  - `"packed_count"`: Number of boxes successfully packed
  - `"placements"`: List of placements for packed boxes:
    - `"box_index"`: Index of the box in input list
    - `"position"`: (x, y, z) - corner position of box
    - `"rotated"`: (rot_x, rot_y, rot_z) - which axes to swap (optional)

**Critical Requirements:**
1. **Scalability**: Your algorithm must adapt based on container size and number of boxes
2. **Performance**: Must complete within 5 minutes even for very large containers
3. **Quality**: Maximize number of boxes packed while ensuring valid placements

**Rotation Options:**
- (0, 0, 0): No rotation - (w, h, d) as given
- (1, 0, 0): Swap height and depth - (w, d, h)
- (0, 1, 0): Swap width and depth - (d, h, w)
- etc. (6 possible orientations for a box)

**Constraints:**
- Use only Python standard library and numpy
- Boxes must be entirely within container bounds
- No two boxes may overlap
- Boxes are axis-aligned (no arbitrary rotation)
- Maximize number of boxes packed
- Must handle varying container sizes and box counts efficiently

Write complete, runnable Python code with the pack_boxes function.
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
      "Explain your 3D packing algorithm and how it adapts to different container sizes and box counts"
    },
    "python_code": {
      "type":
      "string",
      "description":
      "Complete Python code with pack_boxes(boxes, container) function that handles all scales"
    }
  },
  "required": ["reasoning", "python_code"],
  "additionalProperties": False
}


def boxes_overlap(pos1, size1, pos2, size2) -> bool:
  """Check if two axis-aligned boxes overlap."""
  for i in range(3):
    if pos1[i] + size1[i] <= pos2[i] or pos2[i] + size2[i] <= pos1[i]:
      return False
  return True


def box_in_container(pos, size, container) -> bool:
  """Check if box fits within container."""
  for i in range(3):
    if pos[i] < 0 or pos[i] + size[i] > container[i]:
      return False
  return True


def validate_solution(solution: Dict, boxes: List[Tuple],
                      container: Tuple) -> Tuple[bool, str, int]:
  """
    Validate packing solution.
    Returns (is_valid, error, packed_count).
    """
  if not isinstance(solution, dict):
    return False, "Solution must be a dict", 0

  if "packed_count" not in solution or "placements" not in solution:
    return False, "Missing 'packed_count' or 'placements'", 0

  placements = solution["placements"]

  placed_boxes = []  # List of (position, size)
  valid_count = 0
  used_indices = set()

  for p in placements:
    if not isinstance(p, dict):
      continue

    box_idx = p.get("box_index", -1)
    pos = p.get("position", (0, 0, 0))

    if box_idx < 0 or box_idx >= len(boxes):
      continue

    if box_idx in used_indices:
      continue  # Already placed this box

    # Get box size (handle rotation if specified)
    size = list(boxes[box_idx])
    rotated = p.get("rotated", (0, 0, 0))
    if rotated and any(rotated):
      # Simple rotation handling - just try the given size
      pass

    pos = tuple(int(x) for x in pos)
    size = tuple(size)

    # Check container bounds
    if not box_in_container(pos, size, container):
      continue

    # Check overlap with already placed boxes
    overlaps = False
    for prev_pos, prev_size in placed_boxes:
      if boxes_overlap(pos, size, prev_pos, prev_size):
        overlaps = True
        break

    if not overlaps:
      placed_boxes.append((pos, size))
      used_indices.add(box_idx)
      valid_count += 1

  return True, "", valid_count


def greedy_pack(boxes: List[Tuple], container: Tuple) -> int:
  """
    Greedy first-fit packing.
    Returns number of boxes packed.
    """
  # Sort by volume descending
  indexed_boxes = sorted(enumerate(boxes), key=lambda x: -x[1][0] * x[1][1] * x[1][2])

  placed = []  # List of (position, size)

  for idx, (w, h, d) in indexed_boxes:
    # Try to find a valid position
    placed_box = False

    # Generate candidate positions (extreme points approach simplified)
    candidates = [(0, 0, 0)]
    for pos, size in placed:
      # Add corners of placed boxes
      candidates.append((pos[0] + size[0], pos[1], pos[2]))
      candidates.append((pos[0], pos[1] + size[1], pos[2]))
      candidates.append((pos[0], pos[1], pos[2] + size[2]))

    # Sort by z, then y, then x (bottom-left-back preference)
    candidates.sort(key=lambda p: (p[2], p[1], p[0]))

    for cx, cy, cz in candidates:
      pos = (cx, cy, cz)
      size = (w, h, d)

      if not box_in_container(pos, size, container):
        continue

      overlaps = False
      for prev_pos, prev_size in placed:
        if boxes_overlap(pos, size, prev_pos, prev_size):
          overlaps = True
          break

      if not overlaps:
        placed.append((pos, size))
        placed_box = True
        break

    # If not placed, try more positions
    if not placed_box:
      for x in range(0, container[0] - w + 1, max(1, w // 2)):
        if placed_box:
          break
        for y in range(0, container[1] - h + 1, max(1, h // 2)):
          if placed_box:
            break
          for z in range(0, container[2] - d + 1, max(1, d // 2)):
            pos = (x, y, z)
            size = (w, h, d)

            overlaps = False
            for prev_pos, prev_size in placed:
              if boxes_overlap(pos, size, prev_pos, prev_size):
                overlaps = True
                break

            if not overlaps:
              placed.append((pos, size))
              placed_box = True
              break

  return len(placed)


def execute_solver(code: str,
                   boxes: List[Tuple],
                   container: Tuple,
                   timeout: int = TIMEOUT_SECONDS) -> tuple:
  """Execute the LLM's solver."""
  from solver_utils import execute_solver_with_data

  data_dict = {
    'boxes': boxes,
    'container': container,
  }

  result, error, exec_time = execute_solver_with_data(code, data_dict, 'pack_boxes', timeout)

  if error:
    return None, error, exec_time

  if not isinstance(result, dict):
    return None, f"Invalid result type: expected dict, got {type(result).__name__}", exec_time

  try:
    placements = result.get('placements')
    if not isinstance(placements, list):
      return None, "Invalid result: missing or non-list 'placements'", exec_time

    normalized_placements = []
    for p in placements:
      if not isinstance(p, dict):
        continue
      idx = int(p.get('box_index', 0))
      pos = p.get('position', [0, 0, 0])
      rot = p.get('rotated', [0, 0, 0])
      normalized_placements.append({
        'box_index': idx,
        'position': [int(pos[0]), int(pos[1]), int(pos[2])],
        'rotated': [int(rot[0]), int(rot[1]), int(rot[2])],
      })

    out = {
      'packed_count': int(result.get('packed_count', len(normalized_placements))),
      'placements': normalized_placements,
    }
  except Exception as e:
    return None, f"Invalid result format: {e}", exec_time

  return out, None, exec_time


def gradeAnswer(result: dict, subPass: int, aiEngineName: str) -> tuple:
  """Grade the 3D packing solver."""
  if not result:
    return 0.0, "No result provided"

  if "python_code" not in result:
    return 0.0, "No Python code provided"

  case = TEST_CASES[subPass]
  boxes = case["boxes"]
  container = case["container"]
  description = case["description"]
  code = result["python_code"]

  # Execute solver
  solution, error, exec_time = execute_solver(code, boxes, container)

  if error:
    return 0.0, f"[{description}] {error}"

  # Validate solution
  is_valid, validation_error, packed_count = validate_solution(solution, boxes, container)

  if not is_valid:
    return 0.0, f"[{description}] {validation_error}"

  # Get baseline
  baseline_count = greedy_pack(boxes, container)

  if packed_count == 0:
    return 0.0, f"[{description}] No boxes packed"

  # Score based on boxes packed vs baseline
  ratio = packed_count / baseline_count if baseline_count > 0 else 1.0

  if ratio >= 1.0:
    score = 1.0
    quality = "excellent (≥ baseline)"
  elif ratio >= 0.8:
    score = 0.85
    quality = "good (≥ 80% of baseline)"
  elif ratio >= 0.6:
    score = 0.7
    quality = "acceptable (≥ 60% of baseline)"
  elif ratio > 0:
    score = 0.5
    quality = f"partial ({packed_count}/{baseline_count})"
  else:
    score = 0.0
    quality = "failed"

  explanation = (
    f"[{description}] Packed: {packed_count}/{len(boxes)}, Baseline: {baseline_count}, "
    f"Time: {exec_time:.1f}s - {quality}")

  return score, explanation


def resultToNiceReport(result: dict, subPass: int, aiEngineName: str) -> str:
  """Generate HTML report."""
  if not result:
    return "<p style='color:red'>No result provided</p>"

  case = TEST_CASES[subPass]

  html = f"<h4>3D AABB Packing - {case['description']}</h4>"

  if subPass == 0:
    if "reasoning" in result:
      reasoning = result['reasoning'][:500] + ('...'
                                               if len(result.get('reasoning', '')) > 500 else '')
      html += f"<p><strong>Algorithm:</strong> {reasoning}</p>"

    if "python_code" in result:
      code = result["python_code"]
      code_escaped = code.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
      html += f"<details><summary>View Code ({len(code)} chars)</summary><pre>{code_escaped}</pre></details>"

  # TODO: Add three.js rendering here, for at least the trivial cases.

  return html


highLevelSummary = """
3D Bin Packing with axis-aligned boxes is a classic optimization problem.

**Problem:** Pack rectangular boxes into a container, maximizing utilization.

**Algorithms:**
1. **First Fit Decreasing (FFD):** Sort boxes by volume, place each in first valid spot
2. **Bottom-Left-Back (BLB):** Place at lowest, leftmost, backmost position
3. **Extreme Points:** Track valid placement points at box corners
4. **Guillotine:** Recursively split remaining space with cuts
5. **Maximal Spaces:** Track largest empty rectangular regions

**Key considerations:**
- Box rotation (6 orientations possible)
- Placement order affects packing efficiency
- Trade-off between speed and optimality

**Complexity:** NP-hard, but heuristics work well in practice.

The baseline uses greedy first-fit with extreme points heuristic.
"""
