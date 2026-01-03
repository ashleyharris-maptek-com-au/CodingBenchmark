"""
Test 12: 3D Bin Packing - Polyhedra in Bounding Box

The LLM must write a Python solver that packs as many copies of a given
polyhedron as possible into an axis-aligned bounding box (AABB).

Each placement is defined by a translation (x, y, z) and a quaternion rotation (w, x, y, z).

Subpasses test different polyhedra and box sizes.
Solver times out after 5 minutes.
"""

import math
import random
import subprocess
import sys
import tempfile
import os
import time
from typing import List, Tuple, Dict, Optional

title = "3D Bin Packing - Polyhedra in Box"

# Timeout in seconds (5 minutes)
TIMEOUT_SECONDS = 30

# Seed for reproducibility
RANDOM_SEED = 55555


def make_box_mesh(sx: float, sy: float, sz: float) -> Dict:
  """Create a box mesh centered at origin."""
  hx, hy, hz = sx / 2, sy / 2, sz / 2
  vertices = [
    [-hx, -hy, -hz],
    [hx, -hy, -hz],
    [hx, hy, -hz],
    [-hx, hy, -hz],
    [-hx, -hy, hz],
    [hx, -hy, hz],
    [hx, hy, hz],
    [-hx, hy, hz],
  ]
  faces = [
    [0, 2, 1],
    [0, 3, 2],
    [4, 5, 6],
    [4, 6, 7],
    [0, 1, 5],
    [0, 5, 4],
    [2, 3, 7],
    [2, 7, 6],
    [0, 4, 7],
    [0, 7, 3],
    [1, 2, 6],
    [1, 6, 5],
  ]
  return {"vertices": vertices, "faces": faces}


def make_tetrahedron_mesh(size: float) -> Dict:
  """Create a regular tetrahedron mesh."""
  a = size / math.sqrt(2)
  vertices = [
    [a, 0, -a / math.sqrt(2)],
    [-a, 0, -a / math.sqrt(2)],
    [0, a, a / math.sqrt(2)],
    [0, -a, a / math.sqrt(2)],
  ]
  faces = [[0, 1, 2], [0, 3, 1], [0, 2, 3], [1, 3, 2]]
  return {"vertices": vertices, "faces": faces}


def make_octahedron_mesh(size: float) -> Dict:
  """Create a regular octahedron mesh."""
  s = size
  vertices = [
    [s, 0, 0],
    [-s, 0, 0],
    [0, s, 0],
    [0, -s, 0],
    [0, 0, s],
    [0, 0, -s],
  ]
  faces = [
    [0, 2, 4],
    [2, 1, 4],
    [1, 3, 4],
    [3, 0, 4],
    [2, 0, 5],
    [1, 2, 5],
    [3, 1, 5],
    [0, 3, 5],
  ]
  return {"vertices": vertices, "faces": faces}


def make_prism_mesh(base: float, height: float) -> Dict:
  """Create a triangular prism mesh."""
  h = base * math.sqrt(3) / 2
  vertices = [
    [0, 0, 0],
    [base, 0, 0],
    [base / 2, h, 0],
    [0, 0, height],
    [base, 0, height],
    [base / 2, h, height],
  ]
  faces = [
    [0, 2, 1],
    [3, 4, 5],
    [0, 1, 4],
    [0, 4, 3],
    [1, 2, 5],
    [1, 5, 4],
    [2, 0, 3],
    [2, 3, 5],
  ]
  return {"vertices": vertices, "faces": faces}


def mesh_bounds(mesh: Dict) -> Tuple[List[float], List[float]]:
  """Get AABB of mesh."""
  verts = mesh["vertices"]
  min_pt = [min(v[i] for v in verts) for i in range(3)]
  max_pt = [max(v[i] for v in verts) for i in range(3)]
  return min_pt, max_pt


def mesh_dimensions(mesh: Dict) -> Tuple[float, float, float]:
  """Get dimensions of mesh bounding box."""
  min_pt, max_pt = mesh_bounds(mesh)
  return (max_pt[0] - min_pt[0], max_pt[1] - min_pt[1], max_pt[2] - min_pt[2])


# Test configurations
TEST_CASES = [
  # Subpass 0: Simple - small boxes in box
  {
    "polyhedron": make_box_mesh(2, 2, 2),
    "container": (10, 10, 10),
    "description": "2x2x2 boxes in 10x10x10 container"
  },
  # Subpass 1: Tetrahedra in box
  {
    "polyhedron": make_tetrahedron_mesh(2),
    "container": (10, 10, 10),
    "description": "Tetrahedra (size=2) in 10x10x10 container"
  },
  # Subpass 2: Rectangular boxes
  {
    "polyhedron": make_box_mesh(3, 2, 1),
    "container": (12, 10, 8),
    "description": "3x2x1 boxes in 12x10x8 container"
  },
  # Subpass 3: Octahedra
  {
    "polyhedron": make_octahedron_mesh(1.5),
    "container": (10, 10, 10),
    "description": "Octahedra (size=1.5) in 10x10x10 container"
  },
  # Subpass 4: Prisms
  {
    "polyhedron": make_prism_mesh(2, 3),
    "container": (12, 12, 12),
    "description": "Triangular prisms in 12x12x12 container"
  },
  # Subpass 5: Larger container
  {
    "polyhedron": make_box_mesh(2, 3, 2),
    "container": (15, 15, 15),
    "description": "2x3x2 boxes in 15x15x15 container"
  },
  # Extreme cases
  {
    "polyhedron": make_tetrahedron_mesh(2),
    "container": (50, 50, 50),
    "description": "2x2x2 boxes in 50x50x50 container (~15k boxes)"
  },
  {
    "polyhedron": make_tetrahedron_mesh(3),
    "container": (100, 100, 100),
    "description": "2x2x2 boxes in 100x100x100 container (~125k boxes)"
  },
  {
    "polyhedron": make_tetrahedron_mesh(1),
    "container": (100, 100, 100),
    "description": "Tetrahedra in 100x100x100 container"
  },
  {
    "polyhedron": make_prism_mesh(1, 1),
    "container": (200, 200, 200),
    "description": "1x1 prisms in 200x200x200 container (~8M prisms)"
  },
  {
    "polyhedron": make_prism_mesh(1, 1),
    "container": (500, 500, 500),
    "description": "1x1 prisms in 500x500x500 container (~125M prisms)"
  },
]


def format_mesh_for_prompt(mesh: Dict) -> str:
  """Format mesh for prompt."""
  verts = mesh["vertices"]
  faces = mesh["faces"]
  v_str = ",\n        ".join(f"[{v[0]:.2f}, {v[1]:.2f}, {v[2]:.2f}]" for v in verts)
  f_str = ", ".join(str(f) for f in faces)
  return f'{{\n    "vertices": [\n        {v_str}\n    ],\n    "faces": [{f_str}]\n}}'


def prepareSubpassPrompt(subPass: int) -> str:
  """Generate the prompt for subpass 0 that handles all problem sizes."""
  if subPass != 0:
    raise StopIteration

  return f"""You are solving a 3D Bin Packing problem.

You must write a Python solver that can handle ANY problem size from trivial to ludicrous scale:
- **Trivial**: Small containers (5x5x5), simple polyhedra (tetrahedra, cubes)
- **Medium**: Medium containers (10x10x10-20x20x20), moderate complexity polyhedra
- **Large**: Large containers (50x50x50-100x100x100), complex polyhedra
- **Extreme**: Massive containers (200x200x200-500x500x500), very complex geometries

**The Challenge:**
Your `pack_polyhedra(polyhedron, container_size)` function will be tested with containers 
ranging from 5x5x5 to 500x500x500 and various polyhedron complexities. The same function must 
adapt its strategy based on the problem scale and work efficiently across ALL scales.

**Input:**
- `polyhedron`: Dict with "vertices" (list of [x, y, z]) and "faces" (list of triangles)
- `container_size`: Tuple (width, height, depth) of axis-aligned bounding box
- Container origin at (0, 0, 0), extends to (width, height, depth)

**Output:**
- Dict with:
  - `"count"`: Number of polyhedra packed
  - `"placements"`: List of placement dicts:
    - `"translation"`: [x, y, z] - position offset
    - `"quaternion"`: [w, x, y, z] - rotation quaternion (w is scalar part)

**Critical Requirements:**
1. **Scalability**: Your algorithm must adapt based on container size and polyhedron complexity
2. **Performance**: Must complete within 5 minutes even for massive containers
3. **Quality**: Maximize packing count while ensuring valid placements

**Quaternion Rotation Reference:**
- Identity (no rotation): [1, 0, 0, 0]
- 90° around Z: [0.707, 0, 0, 0.707]
- 180° around Z: [0, 0, 0, 1]
- To rotate a point: p' = q * p * q^(-1)

**Constraints:**
- Libraries allowed: any python standard library + numpy, scipy
- All polyhedra must be entirely within the container bounds
- No two polyhedra may intersect/overlap
- Must handle varying container sizes and polyhedron complexities

Write complete, runnable Python code with the pack_polyhedra function.
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
      "Explain your 3D packing algorithm and how it adapts to different container sizes and polyhedron complexities"
    },
    "python_code": {
      "type":
      "string",
      "description":
      "Complete Python code with pack_polyhedra(polyhedron, container_size) function that handles all scales"
    }
  },
  "required": ["reasoning", "python_code"],
  "additionalProperties": False
}


def quaternion_multiply(q1, q2):
  """Multiply two quaternions [w, x, y, z]."""
  w1, x1, y1, z1 = q1
  w2, x2, y2, z2 = q2
  return [
    w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
    w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
    w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
    w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
  ]


def quaternion_rotate_point(q, p):
  """Rotate point p by quaternion q."""
  # p as quaternion [0, px, py, pz]
  p_quat = [0, p[0], p[1], p[2]]
  # q conjugate
  q_conj = [q[0], -q[1], -q[2], -q[3]]
  # q * p * q^-1
  temp = quaternion_multiply(q, p_quat)
  result = quaternion_multiply(temp, q_conj)
  return [result[1], result[2], result[3]]


def transform_mesh(mesh: Dict, translation: List[float], quaternion: List[float]) -> Dict:
  """Apply quaternion rotation then translation to mesh."""
  new_verts = []
  for v in mesh["vertices"]:
    rotated = quaternion_rotate_point(quaternion, v)
    new_verts.append([
      rotated[0] + translation[0],
      rotated[1] + translation[1],
      rotated[2] + translation[2],
    ])
  return {"vertices": new_verts, "faces": mesh["faces"]}


def mesh_in_aabb(mesh: Dict, container: Tuple[float, float, float]) -> bool:
  """Check if mesh is entirely within AABB from origin to container."""
  for v in mesh["vertices"]:
    if v[0] < -0.001 or v[0] > container[0] + 0.001:
      return False
    if v[1] < -0.001 or v[1] > container[1] + 0.001:
      return False
    if v[2] < -0.001 or v[2] > container[2] + 0.001:
      return False
  return True


def aabb_intersect(min1, max1, min2, max2) -> bool:
  """Check if two AABBs intersect."""
  for i in range(3):
    if max1[i] < min2[i] - 0.001 or min1[i] > max2[i] + 0.001:
      return False
  return True


def meshes_intersect_simple(mesh1: Dict, mesh2: Dict) -> bool:
  """
    Simple intersection check using AABB and vertex-in-mesh tests.
    This is a simplified check - not exact for all cases.
    """
  # AABB check first
  min1, max1 = mesh_bounds(mesh1)
  min2, max2 = mesh_bounds(mesh2)

  if not aabb_intersect(min1, max1, min2, max2):
    return False

  # For simplicity, if AABBs overlap, consider them intersecting
  # A full implementation would do triangle-triangle intersection
  return True


def validate_solution(solution: Dict, poly: Dict, container: Tuple) -> Tuple[bool, str, int]:
  """Validate 3D packing solution. Returns (is_valid, error, valid_count)."""
  if not isinstance(solution, dict):
    return False, "Solution must be a dict", 0

  if "count" not in solution or "placements" not in solution:
    return False, "Missing 'count' or 'placements'", 0

  count = solution["count"]
  placements = solution["placements"]

  if not isinstance(count, int) or count < 0:
    return False, f"count must be non-negative int", 0

  if len(placements) != count:
    return False, f"placements length {len(placements)} != count {count}", 0

  placed_meshes = []
  valid_count = 0

  if len(placements) > 1000:
    placementsToCheck = random.sample(placements, 1000)
  else:
    placementsToCheck = placements

  for p in placementsToCheck:
    if not isinstance(p, dict):
      continue

    trans = p.get("translation", [0, 0, 0])
    quat = p.get("quaternion", [1, 0, 0, 0])

    if len(trans) != 3 or len(quat) != 4:
      continue

    # Transform mesh
    transformed = transform_mesh(poly, trans, quat)

    # Check containment
    if not mesh_in_aabb(transformed, container):
      continue

    # Check overlap with previously placed (using AABB approximation)
    overlaps = False
    for prev_mesh in placed_meshes:
      if meshes_intersect_simple(transformed, prev_mesh):
        overlaps = True
        break

    if not overlaps:
      placed_meshes.append(transformed)
      valid_count += 1

  if overlaps:
    return False, f"Placements overlap: {valid_count}/{count} valid placements", valid_count

  if len(placements) > 1000:
    valid_count = int(valid_count * len(placements) / 1000)

  return True, "", valid_count


def get_baseline_count(poly: Dict, container: Tuple) -> int:
  """Get baseline packing count using simple grid placement."""
  dims = mesh_dimensions(poly)
  min_pt, max_pt = mesh_bounds(poly)

  # Calculate offset to make all vertices positive
  offset = [-min_pt[0], -min_pt[1], -min_pt[2]]

  # Grid placement with small gap
  gap = 0.1
  count = 0

  x = offset[0]
  while x + dims[0] <= container[0]:
    y = offset[1]
    while y + dims[1] <= container[1]:
      z = offset[2]
      while z + dims[2] <= container[2]:
        count += 1
        z += dims[2] + gap
      y += dims[1] + gap
    x += dims[0] + gap

  return max(1, count)


def execute_solver(code: str,
                   poly: Dict,
                   container: Tuple,
                   timeout: int = TIMEOUT_SECONDS) -> tuple:
  """Execute the LLM's solver."""
  from solver_utils import execute_solver_with_data

  data_dict = {
    'polyhedron': poly,
    'container_size': container,
  }

  result, error, exec_time = execute_solver_with_data(code, data_dict, 'pack_polyhedra', timeout)

  if error:
    return None, error, exec_time

  if not isinstance(result, dict):
    return None, f"Invalid result type: expected dict, got {type(result).__name__}", exec_time

  # Normalize output to match expected schema
  try:
    placements = result.get('placements')
    if not isinstance(placements, list):
      return None, "Invalid result: missing or non-list 'placements'", exec_time

    normalized_placements = []
    for p in placements:
      if not isinstance(p, dict):
        continue
      translation = p.get('translation', [0, 0, 0])
      quaternion = p.get('quaternion', [1, 0, 0, 0])
      normalized_placements.append({
        'translation': [float(translation[0]),
                        float(translation[1]),
                        float(translation[2])],
        'quaternion':
        [float(quaternion[0]),
         float(quaternion[1]),
         float(quaternion[2]),
         float(quaternion[3])],
      })

    out = {
      'count': int(result.get('count', len(normalized_placements))),
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
  poly = case["polyhedron"]
  container = case["container"]
  description = case["description"]
  code = result["python_code"]

  # Execute solver
  solution, error, exec_time = execute_solver(code, poly, container)

  if error:
    return 0.0, f"[{description}] {error}"

  # Validate solution
  t1 = time.time()
  is_valid, validation_error, valid_count = validate_solution(solution, poly, container)
  t2 = time.time()
  validation_time = t2 - t1
  if validation_time > 1.0:
    print(f"Validation took: {validation_time:.4f}s")

  if not is_valid:
    return 0.0, f"[{description}] Invalid: {validation_error}"

  # Compare to baseline
  t1 = time.time()
  baseline_count = get_baseline_count(poly, container)
  t2 = time.time()
  baseline_time = t2 - t1
  if baseline_time > 1.0:
    print(f"Baseline calculation took: {baseline_time:.4f}s")

  if valid_count == 0:
    return 0.0, f"[{description}] No valid placements"

  ratio = valid_count / baseline_count if baseline_count > 0 else 0

  if ratio >= 1.0:
    score = 1.0
    quality = "excellent (≥ baseline)"
  elif ratio >= 0.8:
    score = 0.85
    quality = "good (≥ 80% baseline)"
  elif ratio >= 0.5:
    score = 0.7
    quality = "acceptable (≥ 50% baseline)"
  elif ratio > 0:
    score = 0.5
    quality = f"low ({ratio:.0%} of baseline)"
  else:
    score = 0.0
    quality = "no valid placements"

  msg = validation_error if validation_error else ""
  explanation = (f"[{description}] Packed: {valid_count}, Baseline: {baseline_count}, "
                 f"Time: {exec_time:.1f}s - {quality} {msg}")

  return score, explanation


def resultToNiceReport(result: dict, subPass: int, aiEngineName: str) -> str:
  """Generate HTML report."""
  if not result:
    return "<p style='color:red'>No result provided</p>"

  case = TEST_CASES[subPass]

  html = f"<h4>3D Bin Packing - {case['description']}</h4>"

  if subPass == 0:
    if "reasoning" in result:
      reasoning = result['reasoning'][:500] + ('...'
                                               if len(result.get('reasoning', '')) > 500 else '')
      html += f"<p><strong>Algorithm:</strong> {reasoning}</p>"

    if "python_code" in result:
      code = result["python_code"]
      code_escaped = code.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
      html += f"<details><summary>View Code ({len(code)} chars)</summary><pre>{code_escaped}</pre></details>"

  # Add Three.js visualization here (placeholder for actual 3D rendering)
  html += "<p><em>3D visualization would be generated here using Three.js</em></p>"

  return html


highLevelSummary = """
3D Bin Packing with polyhedra is a challenging geometric optimization problem.

**Problem:** Pack as many copies of a polyhedron into a box as possible.

**Key challenges:**
- 3D rotation adds complexity (quaternions)
- Collision detection in 3D
- Irregular shapes don't pack efficiently

**Algorithms:**
- **Grid-based**: For regular shapes, use grid placement
- **Layer-by-layer**: Pack 2D layers, stack them
- **Greedy bottom-left-back**: Place at first valid position
- **Genetic algorithms**: For complex shapes

**Required operations:**
- Quaternion rotation
- AABB containment check
- Mesh-mesh intersection (simplified: AABB overlap)

The baseline uses simple grid placement without rotation optimization.
"""
