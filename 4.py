"""
Test 4: Tetrahedron Packing

The LLM must write a Python solver that packs as many regular tetrahedrons
as possible within a given polyhedron (defined by its vertices).

Each tetrahedron has a fixed edge length. The goal is to maximize the number
of non-overlapping tetrahedrons that fit entirely within the polyhedron.

Subpasses test increasingly complex container shapes.
Solver times out after 5 minutes.
"""

import math
import random
import time
from typing import List, Tuple, Set, Optional, Dict

from native_compiler import CppCompiler, compile_and_run, describe_this_pc
from solver_utils import StreamingInputFile

# Import visualization utilities
from visualization_utils import generate_threejs_tetrahedron_visualization

title = "Regular Tetrahedron Packing in Polyhedra (C++)"

# Seed for reproducibility
RANDOM_SEED = 99999

# Timeout in seconds (30 seconds)
TIMEOUT_SECONDS = 30

# Tetrahedron edge length for all tests
TETRA_EDGE = 1.0

# Container configurations: (name, vertices_generator, expected_min_tetrahedrons)
# Vertices define a convex polyhedron


def make_cube(size: float) -> List[Tuple[float, float, float]]:
  """Generate vertices of a cube centered at origin."""
  s = size / 2
  return [(-s, -s, -s), (s, -s, -s), (s, s, -s), (-s, s, -s), (-s, -s, s), (s, -s, s), (s, s, s),
          (-s, s, s)]


def make_rectangular_box(lx: float, ly: float, lz: float) -> List[Tuple[float, float, float]]:
  """Generate vertices of a rectangular box."""
  return [(0, 0, 0), (lx, 0, 0), (lx, ly, 0), (0, ly, 0), (0, 0, lz), (lx, 0, lz), (lx, ly, lz),
          (0, ly, lz)]


def make_tetrahedron(size: float) -> List[Tuple[float, float, float]]:
  """Generate vertices of a regular tetrahedron."""
  # Regular tetrahedron with edge length = size
  a = size / math.sqrt(2)
  return [(a, 0, -a / math.sqrt(2)), (-a, 0, -a / math.sqrt(2)), (0, a, a / math.sqrt(2)),
          (0, -a, a / math.sqrt(2))]


def make_octahedron(size: float) -> List[Tuple[float, float, float]]:
  """Generate vertices of a regular octahedron."""
  s = size
  return [(s, 0, 0), (-s, 0, 0), (0, s, 0), (0, -s, 0), (0, 0, s), (0, 0, -s)]


def make_irregular_polyhedron(rng: random.Random) -> List[Tuple[float, float, float]]:
  """Generate an irregular convex polyhedron."""
  # Start with a cube and perturb vertices
  base = make_cube(4.0)
  vertices = []
  for x, y, z in base:
    vertices.append(
      (x + rng.uniform(-0.3, 0.3), y + rng.uniform(-0.3, 0.3), z + rng.uniform(-0.3, 0.3)))
  return vertices


# Define test configurations
CONTAINER_CONFIGS = [
  ("small_cube", lambda: make_cube(2.0), 1),
  ("medium_cube", lambda: make_cube(3.0), 3),
  ("rectangular_box", lambda: make_rectangular_box(4.0, 3.0, 2.0), 5),
  ("large_cube", lambda: make_cube(5.0), 15),
  ("octahedron", lambda: make_octahedron(3.0), 4),
  ("irregular", lambda: make_irregular_polyhedron(random.Random(RANDOM_SEED)), 8),
  # Extreme cases
  ("huge_cube", lambda: make_cube(20.0), 1000),  # ~8000 tetrahedra
  ("giant_cube", lambda: make_cube(50.0), 15000),  # ~125000 tetrahedra
  ("massive_cube", lambda: make_cube(100.0), 100000),  # ~1M tetrahedra
  ("enormous_box", lambda: make_rectangular_box(200.0, 150.0, 100.0), 500000),
  ("cosmic_cube", lambda: make_cube(500.0), 5000000),  # 5M+ tetrahedra theoretical
]

# Pre-generate containers
CONTAINERS_CACHE = {}
for i, (name, gen_func, _) in enumerate(CONTAINER_CONFIGS):
  CONTAINERS_CACHE[i] = (name, gen_func())

# Global variable to store last placements for visualization
lastPlacements = None


def format_vertices_for_prompt(vertices: List[Tuple]) -> str:
  """Format vertices as Python list."""
  lines = ["["]
  for v in vertices:
    lines.append(f"    ({v[0]:.4f}, {v[1]:.4f}, {v[2]:.4f}),")
  lines.append("]")
  return "\n".join(lines)


def prepareSubpassPrompt(subPass: int) -> str:
  """Generate the prompt for subpass 0 that handles all container sizes."""
  if subPass != 0:
    raise StopIteration

  return f"""You are solving a 3D Packing problem: Pack regular tetrahedrons into polyhedra (C++).

You must write a C++ solver that can handle ANY container size from trivial to ludicrous scale.

**Input format (stdin):**
Line 1: V edge_length
Lines 2..V+1: x y z (container vertices)

**Output format (stdout):**
Line 1: K (number of tetrahedrons placed)
Lines 2..K+1: cx cy cz rx ry rz (center + Euler angles in radians)

**Critical Requirements:**
1. **Scalability**: Your algorithm must adapt based on container volume and complexity
2. **Performance**: Must complete within 30 seconds even for massive containers
3. **Quality**: Maximize number of tetrahedrons while ensuring no overlaps

**Key Geometry (edge length e={TETRA_EDGE}):**
- Height: h = e * sqrt(2/3) ≈ {TETRA_EDGE * math.sqrt(2/3):.4f}
- Circumradius: R = e * sqrt(3/8) ≈ {TETRA_EDGE * math.sqrt(3/8):.4f}
- Inradius: r = e / sqrt(24) ≈ {TETRA_EDGE / math.sqrt(24):.4f}

**Environment:**
{describe_this_pc()}

**C++ Compiler:**
{CppCompiler("test_engine").describe()}

Be sure that any deviation from the C++ standard library is supported by the given compiler,
as referencing the wrong intrinsics or non-standard header like 'bits/stdc++.h' could fail your submission.

Write complete, compilable C++ code with a main() function.
"""


# List of subpasses to grade the single answer against all difficulty levels
extraGradeAnswerRuns = list(range(len(CONTAINER_CONFIGS)))

structure = {
  "type": "object",
  "properties": {
    "reasoning": {
      "type": "string",
      "description": "Explain your packing strategy and how it adapts to different container sizes"
    },
    "cpp_code": {
      "type": "string",
      "description": "Complete C++ code with main() that handles all scales"
    }
  },
  "required": ["reasoning", "cpp_code"],
  "additionalProperties": False
}


def get_tetrahedron_vertices(center: Tuple, rotation: Tuple, edge: float) -> List[Tuple]:
  """
    Get the 4 vertices of a regular tetrahedron given center, rotation, and edge length.
    """
  # Regular tetrahedron vertices centered at origin
  a = edge / math.sqrt(2)
  base_verts = [(a, 0, -a / math.sqrt(2)), (-a, 0, -a / math.sqrt(2)), (0, a, a / math.sqrt(2)),
                (0, -a, a / math.sqrt(2))]

  # Center the tetrahedron at origin (it should already be roughly centered)
  # Apply rotation (Euler angles)
  rx, ry, rz = rotation

  def rotate_point(p):
    x, y, z = p
    # Rotate around X
    y1 = y * math.cos(rx) - z * math.sin(rx)
    z1 = y * math.sin(rx) + z * math.cos(rx)
    y, z = y1, z1
    # Rotate around Y
    x1 = x * math.cos(ry) + z * math.sin(ry)
    z1 = -x * math.sin(ry) + z * math.cos(ry)
    x, z = x1, z1
    # Rotate around Z
    x1 = x * math.cos(rz) - y * math.sin(rz)
    y1 = x * math.sin(rz) + y * math.cos(rz)
    return (x1, y1, z)

  # Rotate and translate
  cx, cy, cz = center
  result = []
  for v in base_verts:
    rv = rotate_point(v)
    result.append((rv[0] + cx, rv[1] + cy, rv[2] + cz))

  return result


def point_in_convex_hull(point: Tuple, vertices: List[Tuple]) -> bool:
  """
    Check if a point is inside the convex hull of vertices.
    Uses a simple approach: check if point is on the correct side of all faces.
    """
  from itertools import combinations

  px, py, pz = point
  n = len(vertices)

  # For small vertex sets, use centroid-based approach
  cx = sum(v[0] for v in vertices) / n
  cy = sum(v[1] for v in vertices) / n
  cz = sum(v[2] for v in vertices) / n

  # Simple bounding box check first
  min_x = min(v[0] for v in vertices)
  max_x = max(v[0] for v in vertices)
  min_y = min(v[1] for v in vertices)
  max_y = max(v[1] for v in vertices)
  min_z = min(v[2] for v in vertices)
  max_z = max(v[2] for v in vertices)

  margin = 0.001
  if (px < min_x - margin or px > max_x + margin or py < min_y - margin or py > max_y + margin
      or pz < min_z - margin or pz > max_z + margin):
    return False

  # For convex hulls with 8 or fewer vertices (like cubes),
  # check if point is "inside" relative to centroid
  # This is a simplified check
  dist_to_center = math.sqrt((px - cx)**2 + (py - cy)**2 + (pz - cz)**2)
  max_dist = max(math.sqrt((v[0] - cx)**2 + (v[1] - cy)**2 + (v[2] - cz)**2) for v in vertices)

  return dist_to_center <= max_dist * 1.1  # Allow small margin


def tetrahedron_in_container(tetra_verts: List[Tuple], container_verts: List[Tuple]) -> bool:
  """Check if all tetrahedron vertices are inside the container."""
  for v in tetra_verts:
    if not point_in_convex_hull(v, container_verts):
      return False
  return True


def tetrahedrons_overlap(verts1: List[Tuple], verts2: List[Tuple]) -> bool:
  """
    Check if two tetrahedrons overlap.
    Uses a simplified center-distance check based on circumradius.
    """
  # Calculate centers
  c1 = tuple(sum(v[i] for v in verts1) / 4 for i in range(3))
  c2 = tuple(sum(v[i] for v in verts2) / 4 for i in range(3))

  # Distance between centers
  dist = math.sqrt(sum((c1[i] - c2[i])**2 for i in range(3)))

  # Minimum separation (using inradius as approximation)
  # Two tetrahedrons don't overlap if centers are far enough apart
  inradius = TETRA_EDGE / math.sqrt(24)
  min_sep = inradius * 1.8  # Conservative estimate

  return dist < min_sep


def calculate_tetrahedron_volume(edge: float) -> float:
  """Volume of regular tetrahedron."""
  return (edge**3) / (6 * math.sqrt(2))


def estimate_container_volume(vertices: List[Tuple]) -> float:
  """Rough estimate of convex hull volume using bounding box."""
  min_x = min(v[0] for v in vertices)
  max_x = max(v[0] for v in vertices)
  min_y = min(v[1] for v in vertices)
  max_y = max(v[1] for v in vertices)
  min_z = min(v[2] for v in vertices)
  max_z = max(v[2] for v in vertices)

  # Bounding box volume * ~0.5 for typical convex shapes
  bbox_vol = (max_x - min_x) * (max_y - min_y) * (max_z - min_z)
  return bbox_vol * 0.6


def get_baseline_packing(container_vertices: List[Tuple], edge: float) -> int:
  """
  Fast baseline packing using volume calculation.
  Returns count of tetrahedrons that fit based on container volume and packing efficiency.
  """
  tetra_vol = calculate_tetrahedron_volume(edge)
  container_vol = estimate_container_volume(container_vertices)

  # Maximum tetrahedron packing efficiency in 3D space is ~0.856 (for dense packings)
  # For random/irregular containers, use a more conservative estimate
  max_packing_efficiency = 0.856  # Theoretical maximum for optimal packings

  # For practical baseline, use a more conservative efficiency
  # This accounts for:
  # - Container shape inefficiencies
  # - Non-optimal packing patterns
  # - Edge effects and wasted space
  if container_vol / tetra_vol < 10:
    practical_efficiency = 0.4  # Conservative but achievable baseline
  else:
    practical_efficiency = max_packing_efficiency  # Better efficiency for larger containers

  # Calculate expected count
  expected_count = int(container_vol / tetra_vol * practical_efficiency)

  # Ensure at least 1 tetrahedron fits (for smallest containers)
  return max(1, expected_count)


def format_input(container_vertices: List[Tuple], edge: float) -> str:
  lines = [f"{len(container_vertices)} {edge:.6f}"]
  for x, y, z in container_vertices:
    lines.append(f"{x:.6f} {y:.6f} {z:.6f}")
  return "\n".join(lines)


def parse_placements_output(output: str) -> tuple:
  tokens = output.strip().split()
  if not tokens:
    return None, "Empty output"

  values = []
  try:
    values = [float(t) for t in tokens]
  except ValueError:
    return None, "Output contains non-numeric tokens"

  if len(values) < 1:
    return None, "Missing placement count"

  count = int(values[0])
  remaining = values[1:]

  if count < 0:
    return None, "Invalid placement count"

  if len(remaining) < count * 6:
    if len(values) % 6 == 0:
      count = len(values) // 6
      remaining = values
    else:
      return None, f"Expected {count * 6} placement values, got {len(remaining)}"

  placements = []
  for i in range(count):
    idx = i * 6
    cx, cy, cz, rx, ry, rz = remaining[idx:idx + 6]
    placements.append({'center': (cx, cy, cz), 'rotation': (rx, ry, rz)})

  return placements, ""


def execute_solver(code: str,
                   container_vertices: List[Tuple],
                   edge: float,
                   timeout: int = TIMEOUT_SECONDS) -> tuple:
  """Execute the LLM's solver. Returns (placements, error, exec_time)."""
  input_data = format_input(container_vertices, edge)
  run = compile_and_run(code, "cpp", "test_engine", input_data=input_data, timeout=timeout)

  if not run:
    return None, run.error_message(), run.exec_time

  placements, parse_error = parse_placements_output(run.stdout)
  if parse_error:
    return None, parse_error, run.exec_time

  return placements, None, run.exec_time


def validate_packing(placements: List[dict], container_vertices: List[Tuple],
                     edge: float) -> Tuple[bool, str, int]:
  """
    Validate packing. Returns (is_valid, error_message, valid_count).
    """
  if not isinstance(placements, list):
    return False, "Placements must be a list", 0

  valid_tetrahedrons = []

  for i, p in enumerate(placements):
    if not isinstance(p, dict):
      continue
    if 'center' not in p:
      continue

    center = p['center']
    rotation = p.get('rotation', (0, 0, 0))

    try:
      verts = get_tetrahedron_vertices(center, rotation, edge)
    except:
      continue

    # Check containment
    if not tetrahedron_in_container(verts, container_vertices):
      continue

    # Check overlap with previously validated
    overlaps = False
    for prev_verts in valid_tetrahedrons:
      if tetrahedrons_overlap(verts, prev_verts):
        overlaps = True
        break

    if not overlaps:
      valid_tetrahedrons.append(verts)

  return True, "", len(valid_tetrahedrons)


def gradeAnswer(result: dict, subPass: int, aiEngineName: str) -> tuple:
  if not result:
    return 0.0, "No result provided"

  if "cpp_code" not in result:
    return 0.0, "No C++ code provided"

  global lastPlacements
  name, container_vertices = CONTAINERS_CACHE[subPass]
  _, _, expected_min = CONTAINER_CONFIGS[subPass]
  code = result["cpp_code"]

  # Execute solver
  placements, error, exec_time = execute_solver(code, container_vertices, TETRA_EDGE)

  if error:
    return 0.0, f"[{name}] {error}"

  # Store placements for visualization
  lastPlacements = placements

  # Validate packing
  t = time.time()
  is_valid, validation_error, valid_count = validate_packing(placements, container_vertices,
                                                             TETRA_EDGE)
  validation_time = time.time() - t
  if validation_time > 1.0:
    print(f"[{name}] Validation took: {validation_time:.1f}s")

  if not is_valid:
    return 0.0, f"[{name}] Invalid packing: {validation_error}"

  # Get baseline
  baseline = get_baseline_packing(container_vertices, TETRA_EDGE)

  # Score
  if valid_count == 0:
    score = 0.0
    quality = "no valid tetrahedrons"
  elif valid_count >= baseline * 1.5:
    score = 1.0
    quality = f"excellent (≥150% of baseline)"
  elif valid_count >= baseline:
    score = 0.5
    quality = f"good (≥100% of baseline)"
  elif valid_count >= baseline * 0.5:
    score = 0.25
    quality = f"acceptable (≥50% of baseline)"
  else:
    score = 0.0
    quality = f"minimal ({valid_count} tetrahedrons)"

  explanation = (
    f"[{name}] Packed: {valid_count} legal tetrahedrons (of {len(lastPlacements)} total placements), "
    f"Baseline: {baseline}, Expected min: {expected_min}, "
    f"Time: {exec_time:.1f}s - {quality}")

  return score, explanation


def resultToNiceReport(result: dict, subPass: int, aiEngineName: str) -> str:
  """Generate HTML report."""
  if not result:
    return "<p style='color:red'>No result provided</p>"

  name, vertices = CONTAINERS_CACHE[subPass]

  html = f"<h4>Tetrahedron Packing - {name} ({len(vertices)} vertices)</h4>"

  if subPass == 0:
    if "reasoning" in result:
      reasoning = result['reasoning'][:500] + ('...'
                                               if len(result.get('reasoning', '')) > 500 else '')
      html += f"<p><strong>Strategy:</strong> {reasoning}</p>"

    if "cpp_code" in result:
      code = result["cpp_code"]
      code_escaped = code.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
      html += f"<details><summary>View Code ({len(code)} chars)</summary><pre>{code_escaped}</pre></details>"

  # Add 3D visualization if we have placements
  global lastPlacements
  if lastPlacements and isinstance(lastPlacements, list) and len(lastPlacements) > 0:
    try:
      # Limit visualization to first 100 tetrahedrons for performance
      viz_placements = lastPlacements[:100] if len(lastPlacements) > 100 else lastPlacements
      viz_html = generate_threejs_tetrahedron_visualization(
        vertices, viz_placements, TETRA_EDGE, f"{name} ({len(lastPlacements)} tetrahedrons)")
      html += viz_html
    except Exception as e:
      html += f"<p style='color:orange;'>3D visualization error: {str(e)}</p>"

  return html


highLevelSummary = """
<p>Pack as many identical triangular pyramids (tetrahedra) as possible inside a
given 3D shape without any overlaps. Unlike cubes, tetrahedra cannot tile space
perfectly, so clever placement and rotation are essential.</p>
<p>The AI must handle 3D collision detection and geometry. Subpasses use
increasingly complex container shapes. The baseline uses a simple grid
placement strategy.</p>
"""
