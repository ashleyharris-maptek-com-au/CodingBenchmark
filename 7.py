"""
Test 7: CSG Union of Polyhedra

The LLM must write a Python solver that computes the union of two 3D polyhedra.
Each polyhedron is defined by vertices (3D points) and faces (triangles).

The result should be a single mesh representing the boolean union.

Subpasses test increasingly complex geometry.
Solver times out after 5 minutes.
"""

import math
import subprocess
import sys
import tempfile
import os
import time
import json
from typing import List, Tuple, Dict

title = "CSG Union of Two Polyhedra"

# Timeout in seconds (30 seconds for testing)
TIMEOUT_SECONDS = 30


def make_box(cx, cy, cz, sx, sy, sz) -> Dict:
  """Create a box mesh centered at (cx,cy,cz) with size (sx,sy,sz)."""
  hx, hy, hz = sx / 2, sy / 2, sz / 2
  vertices = [
    [cx - hx, cy - hy, cz - hz],  # 0
    [cx + hx, cy - hy, cz - hz],  # 1
    [cx + hx, cy + hy, cz - hz],  # 2
    [cx - hx, cy + hy, cz - hz],  # 3
    [cx - hx, cy - hy, cz + hz],  # 4
    [cx + hx, cy - hy, cz + hz],  # 5
    [cx + hx, cy + hy, cz + hz],  # 6
    [cx - hx, cy + hy, cz + hz],  # 7
  ]
  # Triangulated faces (2 triangles per face)
  faces = [
    [0, 2, 1],
    [0, 3, 2],  # bottom
    [4, 5, 6],
    [4, 6, 7],  # top
    [0, 1, 5],
    [0, 5, 4],  # front
    [2, 3, 7],
    [2, 7, 6],  # back
    [0, 4, 7],
    [0, 7, 3],  # left
    [1, 2, 6],
    [1, 6, 5],  # right
  ]
  return {"vertices": vertices, "faces": faces}


def make_tetrahedron(cx, cy, cz, size) -> Dict:
  """Create a regular tetrahedron centered at (cx,cy,cz)."""
  a = size / math.sqrt(2)
  vertices = [
    [cx + a, cy, cz - a / math.sqrt(2)],
    [cx - a, cy, cz - a / math.sqrt(2)],
    [cx, cy + a, cz + a / math.sqrt(2)],
    [cx, cy - a, cz + a / math.sqrt(2)],
  ]
  faces = [
    [0, 1, 2],
    [0, 3, 1],
    [0, 2, 3],
    [1, 3, 2],
  ]
  return {"vertices": vertices, "faces": faces}


def make_octahedron(cx, cy, cz, size) -> Dict:
  """Create a regular octahedron."""
  s = size
  vertices = [
    [cx + s, cy, cz],
    [cx - s, cy, cz],
    [cx, cy + s, cz],
    [cx, cy - s, cz],
    [cx, cy, cz + s],
    [cx, cy, cz - s],
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


def make_pyramid(cx, cy, cz, base_size, height) -> Dict:
  """Create a square pyramid."""
  h = base_size / 2
  vertices = [
    [cx - h, cy - h, cz],  # 0 base
    [cx + h, cy - h, cz],  # 1 base
    [cx + h, cy + h, cz],  # 2 base
    [cx - h, cy + h, cz],  # 3 base
    [cx, cy, cz + height],  # 4 apex
  ]
  faces = [
    [0, 2, 1],
    [0, 3, 2],  # base
    [0, 1, 4],  # front
    [1, 2, 4],  # right
    [2, 3, 4],  # back
    [3, 0, 4],  # left
  ]
  return {"vertices": vertices, "faces": faces}


# Test configurations: pairs of meshes to union
TEST_CONFIGS = [
  # Subpass 0: Two overlapping boxes (simple)
  {
    "name": "overlapping_boxes",
    "mesh_a": lambda: make_box(0, 0, 0, 2, 2, 2),
    "mesh_b": lambda: make_box(1, 1, 1, 2, 2, 2),
  },
  # Subpass 1: Box and tetrahedron
  {
    "name": "box_tetrahedron",
    "mesh_a": lambda: make_box(0, 0, 0, 3, 3, 3),
    "mesh_b": lambda: make_tetrahedron(1.5, 1.5, 1.5, 2),
  },
  # Subpass 2: Two tetrahedra
  {
    "name": "two_tetrahedra",
    "mesh_a": lambda: make_tetrahedron(0, 0, 0, 3),
    "mesh_b": lambda: make_tetrahedron(1, 1, 0, 3),
  },
  # Subpass 3: Box and octahedron
  {
    "name": "box_octahedron",
    "mesh_a": lambda: make_box(0, 0, 0, 2, 2, 2),
    "mesh_b": lambda: make_octahedron(0, 0, 0, 1.5),
  },
  # Subpass 4: Two pyramids
  {
    "name": "two_pyramids",
    "mesh_a": lambda: make_pyramid(0, 0, 0, 3, 2),
    "mesh_b": lambda: make_pyramid(1, 1, 1, 3, 2),
  },
  # Subpass 5: Complex - three boxes (union of union)
  {
    "name": "three_boxes",
    "mesh_a": lambda: make_box(0, 0, 0, 2, 4, 2),
    "mesh_b": lambda: make_box(0, 0, 0, 4, 2, 2),
  },
  # Extreme cases - high vertex/face counts
  {
    "name": "large_mesh_union",
    "mesh_a": lambda: make_subdivided_box(0, 0, 0, 10, 10, 10, 10),
    "mesh_b": lambda: make_subdivided_box(5, 5, 5, 10, 10, 10, 10),
  },
  {
    "name": "huge_mesh_union",
    "mesh_a": lambda: make_subdivided_box(0, 0, 0, 20, 20, 20, 50),
    "mesh_b": lambda: make_subdivided_box(10, 10, 10, 20, 20, 20, 50),
  },
  {
    "name": "massive_mesh_union_with_pyramid",
    "mesh_a": lambda: make_subdivided_box(0, 0, 0, 50, 50, 50, 100),
    "mesh_b": lambda: make_pyramid(1, 1, 25, 30, 20),
  },
  {
    "name": "massive_mesh_union",
    "mesh_a": lambda: make_subdivided_box(0, 0, 0, 50, 50, 50, 100),
    "mesh_b": lambda: make_subdivided_box(25, 25, 25, 50, 50, 50, 100),
  },
  {
    "name": "extreme_mesh_union",
    "mesh_a": lambda: make_subdivided_box(0, 0, 0, 100, 100, 100, 500),
    "mesh_b": lambda: make_subdivided_box(50, 50, 50, 100, 100, 100, 500),
  },
  {
    "name": "insane_mesh_union",
    "mesh_a": lambda: make_subdivided_box(0, 0, 0, 200, 200, 200, 1000),
    "mesh_b": lambda: make_subdivided_box(100, 100, 100, 200, 200, 200, 1000),
  },
]


def make_subdivided_box(cx, cy, cz, sx, sy, sz, subdivisions) -> Dict:
  """Create a box with subdivided faces for high vertex count."""
  vertices = []
  faces = []
  step = 1.0 / subdivisions

  # Generate grid of vertices on each face
  for face_idx in range(6):
    for i in range(subdivisions + 1):
      for j in range(subdivisions + 1):
        u, v = i * step, j * step
        if face_idx == 0:  # front
          x, y, z = cx - sx / 2 + u * sx, cy - sy / 2 + v * sy, cz - sz / 2
        elif face_idx == 1:  # back
          x, y, z = cx - sx / 2 + u * sx, cy - sy / 2 + v * sy, cz + sz / 2
        elif face_idx == 2:  # left
          x, y, z = cx - sx / 2, cy - sy / 2 + u * sy, cz - sz / 2 + v * sz
        elif face_idx == 3:  # right
          x, y, z = cx + sx / 2, cy - sy / 2 + u * sy, cz - sz / 2 + v * sz
        elif face_idx == 4:  # bottom
          x, y, z = cx - sx / 2 + u * sx, cy - sy / 2, cz - sz / 2 + v * sz
        else:  # top
          x, y, z = cx - sx / 2 + u * sx, cy + sy / 2, cz - sz / 2 + v * sz
        vertices.append([x, y, z])

  # Generate triangular faces
  verts_per_face = (subdivisions + 1)**2
  for face_idx in range(6):
    base = face_idx * verts_per_face
    for i in range(subdivisions):
      for j in range(subdivisions):
        v0 = base + i * (subdivisions + 1) + j
        v1 = v0 + 1
        v2 = v0 + subdivisions + 1
        v3 = v2 + 1
        faces.append([v0, v2, v1])
        faces.append([v1, v2, v3])

  return {"vertices": vertices, "faces": faces}


# Pre-generate meshes
MESHES_CACHE = {}


def format_mesh_for_prompt(mesh: Dict, name: str) -> str:
  """Format mesh as Python dict literal."""
  v = mesh["vertices"]
  f = mesh["faces"]
  lines = [f"{name} = {{"]
  lines.append(f'    "vertices": [')
  for i, vert in enumerate(v):
    lines.append(f"        [{vert[0]:.3f}, {vert[1]:.3f}, {vert[2]:.3f}],  # v{i}")
  lines.append("    ],")
  lines.append(f'    "faces": [')
  for face in f:
    lines.append(f"        {face},")
  lines.append("    ]")
  lines.append("}")
  return "\n".join(lines)


def prepareSubpassPrompt(subPass: int) -> str:
  """Generate the prompt for subpass 0 that handles all mesh complexities."""
  if subPass != 0:
    raise StopIteration

  return f"""You are implementing a CSG (Constructive Solid Geometry) Union operation.

You must write a Python solver that can handle ANY mesh complexity from trivial (6 vertex unit cubes) to 
ludicrous scale (millions of vertices):
Your `csg_union(mesh_a, mesh_b)` function will be tested with meshes ranging from simple boxes to complex 
polyhedra. The same function must work efficiently across ALL scales.

**Input:**
- `mesh_a`, `mesh_b`: Each a dict with:
  - `"vertices"`: List of [x, y, z] coordinates
  - `"faces"`: List of triangles (3 vertex indices each)

**Output:**
- Union mesh in the same format (vertices and faces)
- Must be a valid closed mesh (watertight)
- No duplicate vertices at same position
- Face normals should be consistent (outward-facing)

**Critical Requirements:**
1. **Scalability**: Your algorithm must adapt based on mesh complexity
2. **Performance**: Must complete within 5 minutes even for very complex meshes
3. **Correctness**: Must handle overlapping/intersecting geometry properly

**Constraints:**
- Use only Python standard library + numpy if needed for vector math.
- Result must be a valid mesh dict with "vertices" and "faces"

Write complete, runnable Python code with the csg_union function.
Include adaptive logic that chooses different strategies based on mesh complexity.
"""


# List of subpasses to grade the single answer against all difficulty levels
extraGradeAnswerRuns = list(range(len(TEST_CONFIGS)))

structure = {
  "type": "object",
  "properties": {
    "reasoning": {
      "type":
      "string",
      "description":
      "Explain your CSG algorithm approach and how it adapts to different mesh complexities"
    },
    "python_code": {
      "type":
      "string",
      "description":
      "Complete Python code with csg_union(mesh_a, mesh_b) function that handles all scales"
    }
  },
  "required": ["reasoning", "python_code"],
  "additionalProperties": False
}


def compute_volume(mesh: Dict) -> float:
  """Compute signed volume of a mesh using divergence theorem."""
  vertices = mesh["vertices"]
  faces = mesh["faces"]

  volume = 0.0
  for face in faces:
    v0 = vertices[face[0]]
    v1 = vertices[face[1]]
    v2 = vertices[face[2]]

    # Signed volume of tetrahedron with origin
    volume += (v0[0] * (v1[1] * v2[2] - v1[2] * v2[1]) + v0[1] *
               (v1[2] * v2[0] - v1[0] * v2[2]) + v0[2] * (v1[0] * v2[1] - v1[1] * v2[0])) / 6.0

  return abs(volume)


def compute_bounding_box(mesh: Dict) -> Tuple:
  """Compute axis-aligned bounding box."""
  vertices = mesh["vertices"]
  min_x = min(v[0] for v in vertices)
  max_x = max(v[0] for v in vertices)
  min_y = min(v[1] for v in vertices)
  max_y = max(v[1] for v in vertices)
  min_z = min(v[2] for v in vertices)
  max_z = max(v[2] for v in vertices)
  return (min_x, max_x, min_y, max_y, min_z, max_z)


def bounding_boxes_overlap(bb1: Tuple, bb2: Tuple) -> bool:
  """Check if two bounding boxes overlap."""
  return (bb1[0] <= bb2[1] and bb1[1] >= bb2[0] and bb1[2] <= bb2[3] and bb1[3] >= bb2[2]
          and bb1[4] <= bb2[5] and bb1[5] >= bb2[4])


def get_expected_union_volume(mesh_a: Dict, mesh_b: Dict) -> float:
  """
    Estimate expected union volume.
    Union volume = V(A) + V(B) - V(intersection)
    For overlapping convex shapes, we estimate intersection.
  """
  vol_a = compute_volume(mesh_a)
  vol_b = compute_volume(mesh_b)

  bb_a = compute_bounding_box(mesh_a)
  bb_b = compute_bounding_box(mesh_b)

  if not bounding_boxes_overlap(bb_a, bb_b):
    # No overlap - union is sum
    return vol_a + vol_b

  # For simple cases, try exact calculation
  if is_simple_case(mesh_a, mesh_b):
    exact_volume = calculate_exact_union_volume(mesh_a, mesh_b)
    if exact_volume is not None:
      return exact_volume

  # Estimate intersection volume from bounding box overlap
  overlap_x = max(0, min(bb_a[1], bb_b[1]) - max(bb_a[0], bb_b[0]))
  overlap_y = max(0, min(bb_a[3], bb_b[3]) - max(bb_a[2], bb_b[2]))
  overlap_z = max(0, min(bb_a[5], bb_b[5]) - max(bb_a[4], bb_b[4]))

  bb_overlap_vol = overlap_x * overlap_y * overlap_z

  # Actual intersection is typically less than BB overlap
  # Use rough estimate: ~50% of BB overlap for convex shapes
  estimated_intersection = bb_overlap_vol * 0.5

  return vol_a + vol_b - estimated_intersection


def is_simple_case(mesh_a: Dict, mesh_b: Dict) -> bool:
  """Check if this is a simple case we can calculate exactly."""
  # Simple cases: axis-aligned boxes and regular tetrahedrons
  return (is_axis_aligned_box(mesh_a) or is_regular_tetrahedron(mesh_a)) and \
         (is_axis_aligned_box(mesh_b) or is_regular_tetrahedron(mesh_b))


def is_axis_aligned_box(mesh: Dict) -> bool:
  """Check if mesh is an axis-aligned box."""
  vertices = mesh["vertices"]
  faces = mesh["faces"]

  # Box should have 8 vertices and 6 faces
  if len(vertices) != 8 or len(faces) != 6:
    return False

  # Check if all vertices have coordinates that are just min/max of x, y, z
  x_coords = [v[0] for v in vertices]
  y_coords = [v[1] for v in vertices]
  z_coords = [v[2] for v in vertices]

  # Should only have 2 unique values per coordinate (min and max)
  if len(set(x_coords)) != 2 or len(set(y_coords)) != 2 or len(set(z_coords)) != 2:
    return False

  return True


def is_regular_tetrahedron(mesh: Dict) -> bool:
  """Check if mesh is a regular tetrahedron."""
  vertices = mesh["vertices"]
  faces = mesh["faces"]

  # Tetrahedron should have 4 vertices and 4 faces
  if len(vertices) != 4 or len(faces) != 4:
    return False

  # Check if all edges have approximately the same length
  edge_lengths = []
  for i in range(4):
    for j in range(i + 1, 4):
      v1, v2 = vertices[i], vertices[j]
      length = ((v1[0] - v2[0])**2 + (v1[1] - v2[1])**2 + (v1[2] - v2[2])**2)**0.5
      edge_lengths.append(length)

  # All edges should be roughly equal
  avg_length = sum(edge_lengths) / len(edge_lengths)
  return all(abs(l - avg_length) < 0.01 for l in edge_lengths)


def calculate_exact_union_volume(mesh_a: Dict, mesh_b: Dict) -> float:
  """Calculate exact union volume for simple cases."""
  vol_a = compute_volume(mesh_a)
  vol_b = compute_volume(mesh_b)

  # Box-Box intersection
  if is_axis_aligned_box(mesh_a) and is_axis_aligned_box(mesh_b):
    return calculate_box_box_union(mesh_a, mesh_b)

  # Box-Tetrahedron intersection
  if is_axis_aligned_box(mesh_a) and is_regular_tetrahedron(mesh_b):
    return calculate_box_tetra_union(mesh_a, mesh_b)
  if is_axis_aligned_box(mesh_b) and is_regular_tetrahedron(mesh_a):
    return calculate_box_tetra_union(mesh_b, mesh_a)

  # Tetrahedron-Tetrahedron intersection (complex, fall back to estimate)
  if is_regular_tetrahedron(mesh_a) and is_regular_tetrahedron(mesh_b):
    # For tetrahedron-tetrahedron, use a better estimate based on overlap
    return vol_a + vol_b - estimate_tetra_tetra_intersection(mesh_a, mesh_b)

  return None


def calculate_box_box_union(box_a: Dict, box_b: Dict) -> float:
  """Calculate exact union volume of two axis-aligned boxes."""
  vol_a = compute_volume(box_a)
  vol_b = compute_volume(box_b)

  # Calculate intersection box
  vertices_a = box_a["vertices"]
  vertices_b = box_b["vertices"]

  min_a = [min(v[i] for v in vertices_a) for i in range(3)]
  max_a = [max(v[i] for v in vertices_a) for i in range(3)]
  min_b = [min(v[i] for v in vertices_b) for i in range(3)]
  max_b = [max(v[i] for v in vertices_b) for i in range(3)]

  # Intersection box dimensions
  inter_min = [max(min_a[i], min_b[i]) for i in range(3)]
  inter_max = [min(max_a[i], max_b[i]) for i in range(3)]

  # Check if they actually intersect
  if any(inter_min[i] >= inter_max[i] for i in range(3)):
    return vol_a + vol_b  # No intersection

  # Intersection volume
  inter_vol = (inter_max[0] - inter_min[0]) * (inter_max[1] - inter_min[1]) * (inter_max[2] -
                                                                               inter_min[2])

  return vol_a + vol_b - inter_vol


def calculate_box_tetra_union(box: Dict, tetra: Dict) -> float:
  """Calculate union volume of box and tetrahedron."""
  vol_box = compute_volume(box)
  vol_tetra = compute_volume(tetra)

  # Estimate tetrahedron volume inside box
  tetra_inside_vol = estimate_tetra_in_box(tetra, box)

  return vol_box + vol_tetra - tetra_inside_vol


def estimate_tetra_in_box(tetra: Dict, box: Dict) -> float:
  """Estimate how much of tetrahedron is inside the box."""
  vertices = tetra["vertices"]
  box_vertices = box["vertices"]

  # Get box bounds
  min_box = [min(v[i] for v in box_vertices) for i in range(3)]
  max_box = [max(v[i] for v in box_vertices) for i in range(3)]

  # Count vertices inside box
  inside_count = 0
  for v in vertices:
    if all(min_box[i] <= v[i] <= max_box[i] for i in range(3)):
      inside_count += 1

  vol_tetra = compute_volume(tetra)

  if inside_count == 4:
    # All vertices inside - tetrahedron is fully inside box
    return vol_tetra
  elif inside_count == 0:
    # No vertices inside - estimate intersection based on bounding box overlap
    bb_tetra = compute_bounding_box(tetra)
    bb_box = compute_bounding_box(box)

    overlap_x = max(0, min(bb_tetra[1], bb_box[1]) - max(bb_tetra[0], bb_box[0]))
    overlap_y = max(0, min(bb_tetra[3], bb_box[3]) - max(bb_tetra[2], bb_box[2]))
    overlap_z = max(0, min(bb_tetra[5], bb_box[5]) - max(bb_tetra[4], bb_box[4]))

    bb_overlap_vol = overlap_x * overlap_y * overlap_z
    # Tetrahedron fills about 1/3 of its bounding box
    return bb_overlap_vol * 0.33
  else:
    # Partial overlap - use proportional estimate
    return vol_tetra * (inside_count / 4.0)


def estimate_tetra_tetra_intersection(tetra_a: Dict, tetra_b: Dict) -> float:
  """Better estimate for tetrahedron-tetrahedron intersection."""
  # Use bounding box overlap but with better scaling
  bb_a = compute_bounding_box(tetra_a)
  bb_b = compute_bounding_box(tetra_b)

  overlap_x = max(0, min(bb_a[1], bb_b[1]) - max(bb_a[0], bb_b[0]))
  overlap_y = max(0, min(bb_a[3], bb_b[3]) - max(bb_a[2], bb_b[2]))
  overlap_z = max(0, min(bb_a[5], bb_b[5]) - max(bb_a[4], bb_b[4]))

  bb_overlap_vol = overlap_x * overlap_y * overlap_z

  # Two tetrahedra fill about 2/3 of their combined bounding box when overlapping
  return bb_overlap_vol * 0.67


def execute_solver(code: str, mesh_a: Dict, mesh_b: Dict, timeout: int = TIMEOUT_SECONDS) -> tuple:
  """Execute the LLM's solver. Returns (result_mesh, error, exec_time)."""
  from solver_utils import execute_solver_with_data

  # Build data dictionary for solver
  data_dict = {'mesh_a': mesh_a, 'mesh_b': mesh_b}

  # Execute using common utility with __main__ block stripping
  result, error, exec_time = execute_solver_with_data(code, data_dict, 'csg_union', timeout)

  if error:
    return None, error, exec_time

  return result, None, exec_time


def validate_mesh(mesh: Dict) -> Tuple[bool, str]:
  """Validate mesh structure."""
  if not isinstance(mesh, dict):
    return False, "Result must be a dict"

  if "vertices" not in mesh or "faces" not in mesh:
    return False, "Mesh must have 'vertices' and 'faces'"

  vertices = mesh["vertices"]
  faces = mesh["faces"]

  if not isinstance(vertices, list) or len(vertices) < 4:
    return False, "Need at least 4 vertices for a valid polyhedron"

  if not isinstance(faces, list) or len(faces) < 4:
    return False, "Need at least 4 faces for a valid polyhedron"

  # Check vertex format
  for i, v in enumerate(vertices):
    if not isinstance(v, (list, tuple)) or len(v) != 3:
      return False, f"Vertex {i} must be [x, y, z]"

  # Check face format and indices
  num_verts = len(vertices)
  for i, f in enumerate(faces):
    if not isinstance(f, (list, tuple)) or len(f) < 3:
      return False, f"Face {i} must have at least 3 vertices"
    for idx in f:
      if not isinstance(idx, int) or idx < 0 or idx >= num_verts:
        return False, f"Face {i} has invalid vertex index {idx}"

  return True, ""


lastGeneratedMesh = None


def gradeAnswer(result: dict, subPass: int, aiEngineName: str) -> tuple:
  """
    Grade the CSG union solver.
    
    Scoring based on:
    - Valid mesh structure
    - Volume accuracy compared to expected
    """
  global lastGeneratedMesh
  if not result:
    return 0.0, "No result provided"

  if "python_code" not in result:
    return 0.0, "No Python code provided"

  config = TEST_CONFIGS[subPass]
  MESHES_CACHE[subPass] = {
    "name": config["name"],
    "mesh_a": config["mesh_a"](),
    "mesh_b": config["mesh_b"](),
  }

  data = MESHES_CACHE[subPass]
  mesh_a = data["mesh_a"]
  mesh_b = data["mesh_b"]
  name = data["name"]
  code = result["python_code"]

  # Execute solver
  result_mesh, error, exec_time = execute_solver(code, mesh_a, mesh_b)

  if error:
    return 0.0, f"[{name}] {error}"

  # Validate mesh structure
  is_valid, validation_error = validate_mesh(result_mesh)
  if not is_valid:
    return 0.0, f"[{name}] Invalid mesh: {validation_error}"

  # Store last generated mesh for visualization
  lastGeneratedMesh = result_mesh

  # Compute volumes
  vol_a = compute_volume(mesh_a)
  vol_b = compute_volume(mesh_b)
  result_vol = compute_volume(result_mesh)
  expected_vol = get_expected_union_volume(mesh_a, mesh_b)

  # Minimum valid volume: at least as big as the larger input
  min_valid_vol = max(vol_a, vol_b) * 0.9

  # Maximum valid volume: sum of both (no intersection)
  max_valid_vol = (vol_a + vol_b) * 1.1

  if result_vol < min_valid_vol:
    return 0.3, f"[{name}] Volume too small: {result_vol:.2f} < min {min_valid_vol:.2f}"

  if result_vol > max_valid_vol:
    return 0.3, f"[{name}] Volume too large: {result_vol:.2f} > max {max_valid_vol:.2f}"

  # Score based on volume accuracy
  vol_error = abs(result_vol - expected_vol) / expected_vol if expected_vol > 0 else 0

  if vol_error < 0.1:
    score = 1.0
    quality = "excellent (< 10% error)"
  elif vol_error < 0.25:
    score = 0.3
    quality = "good (< 25% error)"
  elif vol_error < 0.5:
    score = 0.1
    quality = "acceptable (< 50% error)"
  else:
    score = 0
    quality = f"poor ({vol_error*100:.0f}% error)"

  explanation = (f"[{name}] Result volume: {result_vol:.2f}, Expected: ~{expected_vol:.2f}, "
                 f"Inputs: {vol_a:.2f} + {vol_b:.2f}, "
                 f"Time: {exec_time:.1f}s - {quality}")

  return score, explanation


def resultToNiceReport(result: dict, subPass: int, aiEngineName: str) -> str:
  """Generate HTML report."""
  if not result:
    return "<p style='color:red'>No result provided</p>"

  data = MESHES_CACHE[subPass]
  name = data["name"]

  html = f"<h4>CSG Union - {name}</h4>"

  if subPass == 0:

    if "reasoning" in result:
      reasoning = result['reasoning'][:500] + ('...'
                                               if len(result.get('reasoning', '')) > 500 else '')
      html += f"<p><strong>Approach:</strong> {reasoning}</p>"

    if "python_code" in result:
      code = result["python_code"]
      code_escaped = code.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
      html += f"<details><summary>View Code ({len(code)} chars)</summary><pre>{code_escaped}</pre></details>"

  input_mesh_a = data.get("mesh_a")
  input_mesh_b = data.get("mesh_b")
  output_mesh = lastGeneratedMesh
  if input_mesh_a is not None and input_mesh_b is not None and output_mesh is not None:
    from visualization_utils import generate_threejs_csg_visualization
    html += generate_threejs_csg_visualization(input_mesh_a, input_mesh_b, output_mesh, name)
  else:
    html += "<p style='color:red'>No mesh data available</p>"

  return html


highLevelSummary = """
CSG (Constructive Solid Geometry) Union combines two 3D shapes into one.

**The challenge:**
- Handle intersecting geometry correctly
- Split faces along intersection curves
- Determine which faces are inside/outside
- Produce valid watertight mesh

**Approaches:**
1. **BSP Trees**: Binary space partitioning for face classification
2. **Mesh boolean libraries**: trimesh, pycsg, pymesh
3. **External tools**: OpenSCAD, CGAL
4. **Voxelization**: Convert to voxels, union, convert back

**Key operations:**
- Triangle-triangle intersection
- Point-in-polyhedron tests
- Edge-face intersection
- Mesh merging and cleanup

The placebo uses an existing CSG library for correctness.
"""
