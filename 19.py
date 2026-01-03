"""
Test 19: 3D Dirt Excavation Problem (C# Implementation)

The LLM must write C# code that solves an excavation planning problem:
Given overlapping 3D dirt volumes with a target point buried at the bottom,
plan the optimal sequence of dig-and-dump operations to expose the target.

Constraints:
- Dirt can only be carried a limited distance uphill (max_uphill_distance)
- Dirt must be dumped on ground or stable surface (not on other dirt to be dug)
- Can only dig dirt that is exposed from above (no tunneling)
- Goal: Minimize total work (volume * distance moved)

Input format (stdin):
Line 1: N max_uphill target_x target_y target_z
  - N: number of dirt volumes
  - max_uphill: maximum uphill carrying distance
  - target_x/y/z: target point coordinates
Line 2 to N+1: x1 y1 z1 x2 y2 z2
  - Axis-aligned box from (x1,y1,z1) to (x2,y2,z2)
  - z is vertical (up)

Output format (stdout):
Line 1: M (number of operations)
Lines 2 to M+1: dig_index dump_x dump_y dump_z
  - dig_index: which volume to dig (0-indexed)
  - dump_x/y/z: where to dump the dirt

Subpasses test increasingly complex excavation scenarios.
Solver times out after 5 minutes.
"""

skip = True

import random
import subprocess
import sys
import os
import time
import math
from typing import List, Tuple, Set, Dict, Optional
from pathlib import Path

# Import our native compiler helper
from native_compiler import CSharpCompiler, CompilationError, ExecutionError

title = "3D Dirt Excavation Problem (C#)"

# Timeout in seconds (5 minutes)
TIMEOUT_SECONDS = 30

# Seed for reproducibility
RANDOM_SEED = 19191919


class Box:
  """Axis-aligned 3D box."""

  def __init__(self, x1: float, y1: float, z1: float, x2: float, y2: float, z2: float):
    self.x1, self.y1, self.z1 = min(x1, x2), min(y1, y2), min(z1, z2)
    self.x2, self.y2, self.z2 = max(x1, x2), max(y1, y2), max(z1, z2)

  def volume(self) -> float:
    return (self.x2 - self.x1) * (self.y2 - self.y1) * (self.z2 - self.z1)

  def center(self) -> Tuple[float, float, float]:
    return ((self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2, (self.z1 + self.z2) / 2)

  def contains_point(self, x: float, y: float, z: float) -> bool:
    return (self.x1 <= x <= self.x2 and self.y1 <= y <= self.y2 and self.z1 <= z <= self.z2)

  def overlaps(self, other: 'Box') -> bool:
    return not (self.x2 <= other.x1 or other.x2 <= self.x1 or self.y2 <= other.y1
                or other.y2 <= self.y1 or self.z2 <= other.z1 or other.z2 <= self.z1)

  def is_above(self, other: 'Box') -> bool:
    """Check if self is directly above other (blocks access from top)."""
    # Overlaps in x and y, and self is higher
    xy_overlap = not (self.x2 <= other.x1 or other.x2 <= self.x1 or self.y2 <= other.y1
                      or other.y2 <= self.y1)
    return xy_overlap and self.z1 >= other.z2


def generate_dirt_pile(num_volumes: int, pile_radius: float, pile_height: float,
                       seed: int) -> Tuple[List[Box], Tuple[float, float, float]]:
  """
    Generate a pile of overlapping dirt volumes with a buried target.
    Returns (list of boxes, target point).
    """
  rng = random.Random(seed)
  boxes = []

  # Generate overlapping boxes forming a pile
  for i in range(num_volumes):
    # Random position within pile area
    cx = rng.uniform(-pile_radius * 0.8, pile_radius * 0.8)
    cy = rng.uniform(-pile_radius * 0.8, pile_radius * 0.8)

    # Height based on distance from center (higher in middle)
    dist_from_center = math.sqrt(cx * cx + cy * cy)
    max_z = pile_height * (1 - dist_from_center / pile_radius) * 0.8
    cz = rng.uniform(0, max(0.5, max_z))

    # Random box size
    size_x = rng.uniform(pile_radius * 0.1, pile_radius * 0.4)
    size_y = rng.uniform(pile_radius * 0.1, pile_radius * 0.4)
    size_z = rng.uniform(pile_height * 0.05, pile_height * 0.2)

    box = Box(cx - size_x / 2, cy - size_y / 2, cz, cx + size_x / 2, cy + size_y / 2, cz + size_z)
    boxes.append(box)

  # Find target point - bottom center area, covered by dirt
  target_x = rng.uniform(-pile_radius * 0.3, pile_radius * 0.3)
  target_y = rng.uniform(-pile_radius * 0.3, pile_radius * 0.3)
  target_z = 0.1  # Just above ground

  # Ensure at least one box covers the target
  if not any(b.contains_point(target_x, target_y, target_z) for b in boxes):
    # Add a box covering the target
    boxes.append(Box(target_x - 1, target_y - 1, 0, target_x + 1, target_y + 1, pile_height * 0.3))

  return boxes, (target_x, target_y, target_z)


def format_input(boxes: List[Box], max_uphill: float, target: Tuple[float, float, float]) -> str:
  """Format problem as input string."""
  lines = [f"{len(boxes)} {max_uphill:.2f} {target[0]:.2f} {target[1]:.2f} {target[2]:.2f}"]
  for b in boxes:
    lines.append(f"{b.x1:.2f} {b.y1:.2f} {b.z1:.2f} {b.x2:.2f} {b.y2:.2f} {b.z2:.2f}")
  return "\n".join(lines)


def boxes_covering_target(boxes: List[Box], target: Tuple[float, float, float],
                          removed: Set[int]) -> List[int]:
  """Find indices of boxes that cover the target point (not yet removed)."""
  tx, ty, tz = target
  covering = []
  for i, b in enumerate(boxes):
    if i not in removed:
      # Check if box is above target (blocks vertical access)
      if (b.x1 <= tx <= b.x2 and b.y1 <= ty <= b.y2 and b.z1 <= tz):
        covering.append(i)
  return covering


def is_exposed_from_above(box_idx: int, boxes: List[Box], removed: Set[int]) -> bool:
  """Check if a box can be dug (exposed from above)."""
  box = boxes[box_idx]
  for i, other in enumerate(boxes):
    if i != box_idx and i not in removed:
      if other.is_above(box):
        return False
  return True


def distance_3d(p1: Tuple[float, float, float], p2: Tuple[float, float, float]) -> float:
  """Calculate 3D distance between points."""
  return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2 + (p1[2] - p2[2])**2)


def uphill_distance(from_z: float, to_z: float) -> float:
  """Calculate uphill component of movement."""
  return max(0, to_z - from_z)


# Test configurations
TEST_CASES = [
  # Subpass 0: Simple - few volumes
  {
    "boxes": generate_dirt_pile(5, 5.0, 3.0, RANDOM_SEED)[0],
    "target": generate_dirt_pile(5, 5.0, 3.0, RANDOM_SEED)[1],
    "max_uphill": 5.0,
    "description": "5 dirt volumes, small pile"
  },
  # Subpass 1: More volumes
  {
    "boxes": generate_dirt_pile(10, 8.0, 5.0, RANDOM_SEED + 1)[0],
    "target": generate_dirt_pile(10, 8.0, 5.0, RANDOM_SEED + 1)[1],
    "max_uphill": 4.0,
    "description": "10 dirt volumes"
  },
  # Subpass 2: Medium pile
  {
    "boxes": generate_dirt_pile(20, 10.0, 8.0, RANDOM_SEED + 2)[0],
    "target": generate_dirt_pile(20, 10.0, 8.0, RANDOM_SEED + 2)[1],
    "max_uphill": 3.0,
    "description": "20 dirt volumes"
  },
  # Subpass 3: Larger pile
  {
    "boxes": generate_dirt_pile(35, 15.0, 10.0, RANDOM_SEED + 3)[0],
    "target": generate_dirt_pile(35, 15.0, 10.0, RANDOM_SEED + 3)[1],
    "max_uphill": 3.0,
    "description": "35 dirt volumes"
  },
  # Subpass 4: Complex pile
  {
    "boxes": generate_dirt_pile(50, 20.0, 12.0, RANDOM_SEED + 4)[0],
    "target": generate_dirt_pile(50, 20.0, 12.0, RANDOM_SEED + 4)[1],
    "max_uphill": 2.5,
    "description": "50 dirt volumes"
  },
  # Subpass 5: Large pile
  {
    "boxes": generate_dirt_pile(75, 25.0, 15.0, RANDOM_SEED + 5)[0],
    "target": generate_dirt_pile(75, 25.0, 15.0, RANDOM_SEED + 5)[1],
    "max_uphill": 2.0,
    "description": "75 dirt volumes"
  },
  # Extreme cases
  {
    "boxes": generate_dirt_pile(100, 30.0, 20.0, RANDOM_SEED + 6)[0],
    "target": generate_dirt_pile(100, 30.0, 20.0, RANDOM_SEED + 6)[1],
    "max_uphill": 2.0,
    "description": "100 dirt volumes"
  },
  {
    "boxes": generate_dirt_pile(250, 50.0, 30.0, RANDOM_SEED + 7)[0],
    "target": generate_dirt_pile(250, 50.0, 30.0, RANDOM_SEED + 7)[1],
    "max_uphill": 2.0,
    "description": "250 dirt volumes"
  },
  {
    "boxes": generate_dirt_pile(500, 75.0, 40.0, RANDOM_SEED + 8)[0],
    "target": generate_dirt_pile(500, 75.0, 40.0, RANDOM_SEED + 8)[1],
    "max_uphill": 1.5,
    "description": "500 dirt volumes"
  },
  {
    "boxes": generate_dirt_pile(1000, 100.0, 50.0, RANDOM_SEED + 9)[0],
    "target": generate_dirt_pile(1000, 100.0, 50.0, RANDOM_SEED + 9)[1],
    "max_uphill": 1.5,
    "description": "1000 dirt volumes"
  },
  {
    "boxes": generate_dirt_pile(5000, 200.0, 80.0, RANDOM_SEED + 10)[0],
    "target": generate_dirt_pile(5000, 200.0, 80.0, RANDOM_SEED + 10)[1],
    "max_uphill": 1.0,
    "description": "5000 dirt volumes"
  },
]


def prepareSubpassPrompt(subPass: int) -> str:
  """Generate the prompt for subpass 0 that handles all excavation complexities."""
  if subPass != 0:
    raise StopIteration

  return f"""You are writing a C# program to solve a 3D Dirt Excavation problem.

You must write a C# solver that can handle ANY problem size from trivial to ludicrous scale:
- **Trivial**: Simple stacks (3-5 volumes), easy access, basic planning
- **Medium**: Moderate stacks (10-25 volumes), some obstacles, intermediate planning
- **Large**: Complex stacks (50-200 volumes), many obstacles, sophisticated planning
- **Extreme**: Massive stacks (500-5000 volumes), very complex 3D geometry, advanced optimization

**The Challenge:**
Your C# program will be tested with excavation problems ranging from simple 3-volume stacks to massive 5000-volume complexes. The same program must work efficiently across ALL scales.

**Problem:**
You have a pile of overlapping 3D dirt volumes (axis-aligned boxes). A target point is buried at the bottom. You must plan dig-and-dump operations to expose the target.

**Constraints:**
1. Can only dig dirt that is exposed from above (no tunneling)
2. Dug dirt must be dumped somewhere - at a location not blocking future digging
3. Maximum uphill carry distance: varies by test case
4. Dumped dirt becomes a new obstacle (plan carefully!)
5. Goal: Expose the target with minimum total work (volume × distance)

**Input format (stdin):**
```
N max_uphill target_x target_y target_z
x1 y1 z1 x2 y2 z2  (for each volume)
```

**Output format (stdout):**
```
M                    (number of operations)
dig_index dump_x dump_y dump_z  (for each operation)
```

**Critical Requirements:**
1. **Scalability**: Your algorithm must adapt based on number of volumes and complexity
2. **Performance**: Must complete within 5 minutes even for massive excavation problems
3. **Quality**: Minimize total work while ensuring target exposure

**Algorithm Strategy Recommendations:**
- **Small problems (≤10 volumes)**: Can use exact planning, try all reasonable sequences
- **Medium problems (10-50 volumes)**: Greedy heuristics with local optimization
- **Large problems (50-200 volumes)**: Advanced heuristics, search algorithms
- **Very Large problems (>200 volumes)**: Fast heuristics, approximation methods

**Key Techniques:**
- **Greedy approach**: Always dig the highest exposed volume blocking the target
- **Top-down planning**: Dig from highest to lowest volumes
- **Uphill constraint**: Consider maximum carry distance when dumping
- **Obstacle management**: Plan dump locations to avoid blocking future operations
- **Work optimization**: Balance volume vs distance in cost calculations

**Implementation Hints:**
- Detect problem complexity and choose appropriate algorithm
- Use efficient 3D geometry calculations
- Implement adaptive quality vs speed tradeoffs
- For very large problems, consider simplified heuristics
- Handle edge cases: already exposed targets, impossible problems
- Use fast I/O for large inputs

**Requirements:**
1. Program must compile with .NET/csc
2. Read from stdin, write to stdout
3. Handle up to 5000 volumes
4. Complete within 5 minutes
5. All dug volumes must have been exposed from above
6. Target must be exposed after all operations
7. Must handle varying numbers of volumes and uphill constraints efficiently

Write complete, compilable C# code with a Main method.
Include adaptive logic that chooses different strategies based on problem scale.
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
      "Explain your algorithm choice and how it adapts to different excavation complexities"
    },
    "csharp_code": {
      "type": "string",
      "description": "Complete C# code with Main method that handles all scales"
    }
  },
  "required": ["reasoning", "csharp_code"],
  "additionalProperties": False
}


def execute_csharp_solver(code: str,
                          input_data: str,
                          engine_name: str,
                          timeout: float = TIMEOUT_SECONDS) -> Tuple[str, str, float, bool]:
  """
    Compile and execute C# solver.
    
    Returns:
        Tuple of (stdout, error_message, execution_time, success)
    """
  compiler = CSharpCompiler(engine_name)

  # Check if compiler is available
  if not compiler.find_compiler():
    return "", "No C# compiler found", 0, False

  try:
    # Compile
    exe_path = compiler.compile(code)

    # Execute
    stdout, stderr, exec_time, return_code = compiler.execute(exe_path, input_data, timeout)

    if return_code != 0:
      return stdout, f"Runtime error (exit code {return_code}): {stderr[:500]}", exec_time, False

    return stdout, "", exec_time, True

  except CompilationError as e:
    return "", str(e), 0, False
  except ExecutionError as e:
    return "", str(e), TIMEOUT_SECONDS, False
  except Exception as e:
    return "", f"Unexpected error: {str(e)}", 0, False


def parse_output(output: str) -> Tuple[List[Tuple[int, float, float, float]], str]:
  """
    Parse solver output.
    
    Returns:
        Tuple of (operations, error_message)
    """
  lines = output.strip().split('\n')
  if not lines:
    return [], "Empty output"

  try:
    m = int(lines[0].strip())
    if m < 0:
      return [], f"Invalid operation count: {m}"

    operations = []
    for i in range(1, min(m + 1, len(lines))):
      parts = lines[i].strip().split()
      if len(parts) >= 4:
        dig_idx = int(parts[0])
        dump_x = float(parts[1])
        dump_y = float(parts[2])
        dump_z = float(parts[3])
        operations.append((dig_idx, dump_x, dump_y, dump_z))

    if len(operations) != m:
      return operations, f"Expected {m} operations, got {len(operations)}"

    return operations, ""

  except ValueError as e:
    return [], f"Parse error: {str(e)}"


def verify_solution(boxes: List[Box], target: Tuple[float, float, float], max_uphill: float,
                    operations: List[Tuple[int, float, float, float]]) -> Tuple[bool, str, float]:
  """
    Verify that the solution is valid and calculate total work.
    
    Returns:
        Tuple of (is_valid, message, total_work)
    """
  removed = set()
  total_work = 0.0
  dump_locations = []  # Track where dirt was dumped

  for op_idx, (dig_idx, dump_x, dump_y, dump_z) in enumerate(operations):
    # Check dig index is valid
    if dig_idx < 0 or dig_idx >= len(boxes):
      return False, f"Op {op_idx}: Invalid dig index {dig_idx}", 0

    if dig_idx in removed:
      return False, f"Op {op_idx}: Volume {dig_idx} already removed", 0

    # Check if volume is exposed from above
    if not is_exposed_from_above(dig_idx, boxes, removed):
      return False, f"Op {op_idx}: Volume {dig_idx} is not exposed from above", 0

    # Check uphill constraint
    box = boxes[dig_idx]
    box_center = box.center()
    uphill = uphill_distance(box_center[2], dump_z)
    if uphill > max_uphill:
      return False, f"Op {op_idx}: Uphill distance {uphill:.1f} exceeds max {max_uphill:.1f}", 0

    # Calculate work (volume * distance)
    dist = distance_3d(box_center, (dump_x, dump_y, dump_z))
    work = box.volume() * dist
    total_work += work

    # Mark as removed
    removed.add(dig_idx)
    dump_locations.append((dump_x, dump_y, dump_z))

  # Check if target is now exposed
  covering = boxes_covering_target(boxes, target, removed)
  if covering:
    return False, f"Target still covered by volumes: {covering[:5]}", total_work

  return True, "Valid solution", total_work


def gradeAnswer(result: dict, subPass: int, aiEngineName: str) -> tuple:
  """
    Grade the C# excavation solver.
    
    Scoring based on total work (volume * distance):
    - 1.0: Exposes target with efficient work
    - Scaled by comparison to naive solution
    - 0.0: Invalid solution or error
    """
  if not result:
    return 0.0, "No result provided"

  if "csharp_code" not in result:
    return 0.0, "No C# code provided"

  case = TEST_CASES[subPass]
  boxes = case["boxes"]
  target = case["target"]
  max_uphill = case["max_uphill"]
  description = case["description"]

  code = result["csharp_code"]
  input_data = format_input(boxes, max_uphill, target)

  # Execute solver
  stdout, error, exec_time, success = execute_csharp_solver(code, input_data, aiEngineName)

  if not success:
    return 0.0, f"[{description}] {error}"

  # Parse output
  operations, parse_error = parse_output(stdout)
  if parse_error and not operations:
    return 0.0, f"[{description}] {parse_error}"

  # Verify solution
  is_valid, verify_msg, total_work = verify_solution(boxes, target, max_uphill, operations)

  if not is_valid:
    return 0.0, f"[{description}] Invalid: {verify_msg}"

  # Calculate baseline work (naive greedy)
  baseline_work = sum(b.volume() * 10.0 for b in boxes)  # Assume avg 10 units distance

  # Score based on efficiency
  if total_work <= 0:
    score = 1.0
  else:
    efficiency = baseline_work / total_work if total_work > 0 else float('inf')
    if efficiency >= 2.0:
      score = 1.0
    elif efficiency >= 1.0:
      score = 0.8 + 0.2 * (efficiency - 1.0)
    elif efficiency >= 0.5:
      score = 0.5 + 0.3 * (efficiency - 0.5) * 2
    else:
      score = 0.3 + 0.2 * efficiency * 2

  explanation = (f"[{description}] Operations: {len(operations)}, "
                 f"Total work: {total_work:.1f}, "
                 f"Time: {exec_time:.2f}s - Score: {score:.2f}")

  return score, explanation


def resultToNiceReport(result: dict, subPass: int, aiEngineName: str) -> str:
  if not result:
    return "<p style='color:red'>No result provided</p>"
  case = TEST_CASES[subPass]
  html = f"<h4>3D Dirt Excavation - {case['description']}</h4>"
  if "reasoning" in result:
    r = result['reasoning'][:400] + ('...' if len(result.get('reasoning', '')) > 400 else '')
    html += f"<p><strong>Approach:</strong> {r.replace('<', '&lt;').replace('>', '&gt;')}</p>"
  if "csharp_code" in result:
    code = result["csharp_code"].replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    html += f"<details><summary>View C# Code ({len(result['csharp_code'])} chars)</summary><pre>{code}</pre></details>"
  return html


highLevelSummary = """
3D Dirt Excavation optimizes material movement in mining/construction.

**Key concepts:**
- Block model representation of terrain
- Precedence constraints (can't dig below unsupported blocks)
- Minimizing haul distance and equipment moves
"""
