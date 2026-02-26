"""
Test 15: Tetrahedron Shadow Covering

The LLM must write a Python solver that places tetrahedrons in 3D space such that
their combined shadow (projection along a sun vector) covers a given 2D polygon.

Inputs: Target polygon, sun direction vector
Output: List of tetrahedron placements (position + quaternion rotation)

Goals:
- Minimize number of tetrahedrons used
- Shadows must completely cover the target polygon
- Tetrahedrons must not intersect each other

Solver times out after 5 minutes.
"""

import hashlib
import json
import math
import os
import random
import time
from typing import List, Tuple, Dict, Set

import numpy as np
from PIL import Image, ImageDraw

from native_compiler import RustCompiler, compile_and_run, describe_this_pc
from solver_utils import GradeCache

title = "Tetrahedron Shadow Covering (Rust)"

# Timeout in seconds (5 minutes)
TIMEOUT_SECONDS = 30

# Seed for reproducibility
RANDOM_SEED = 22222

# Standard tetrahedron vertices (regular, centered at origin, edge length ~2)
TETRA_VERTICES = [
  [1.0, 0.0, -1.0 / math.sqrt(2)],
  [-1.0, 0.0, -1.0 / math.sqrt(2)],
  [0.0, 1.0, 1.0 / math.sqrt(2)],
  [0.0, -1.0, 1.0 / math.sqrt(2)],
]

TETRA_SIZE = 2.0  # Approximate size
MAX_TETRA_SCALE = 10.0  # Maximum allowed scale factor per tetrahedron

# Numpy array of tetrahedron vertices for vectorized ops: shape (4, 3)
TETRA_VERTICES_NP = np.array(TETRA_VERTICES, dtype=np.float64)


def make_rectangle_poly(w: float, h: float) -> List[Tuple[float, float]]:
  """Create rectangle polygon."""
  return [(0, 0), (w, 0), (w, h), (0, h)]


def make_triangle_poly(base: float, height: float) -> List[Tuple[float, float]]:
  """Create triangle polygon."""
  return [(0, 0), (base, 0), (base / 2, height)]


def make_hexagon_poly(size: float) -> List[Tuple[float, float]]:
  """Create regular hexagon polygon."""
  points = []
  for i in range(6):
    angle = math.pi / 3 * i
    points.append((size * math.cos(angle) + size, size * math.sin(angle) + size))
  return points


def make_l_shape_poly(w: float, h: float, notch: float) -> List[Tuple[float, float]]:
  """Create L-shaped polygon."""
  return [(0, 0), (w, 0), (w, notch), (notch, notch), (notch, h), (0, h)]


# Test configurations
TEST_CASES = [
  # Subpass 0: Small square
  {
    "polygon": make_rectangle_poly(3, 3),
    "sun_vector": [0, 0, -1],  # Straight down
    "description": "3x3 square, sun straight down"
  },
  # Subpass 1: Rectangle with angled sun
  {
    "polygon": make_rectangle_poly(4, 3),
    "sun_vector": [0.3, 0, -1],  # Slightly angled
    "description": "4x3 rectangle, angled sun"
  },
  # Subpass 2: Triangle
  {
    "polygon": make_triangle_poly(5, 4),
    "sun_vector": [0, 0.2, -1],
    "description": "Triangle, angled sun"
  },
  # Subpass 3: Larger rectangle
  {
    "polygon": make_rectangle_poly(6, 5),
    "sun_vector": [0.2, 0.2, -1],
    "description": "6x5 rectangle, diagonal sun"
  },
  # Subpass 4: Hexagon
  {
    "polygon": make_hexagon_poly(3),
    "sun_vector": [0, 0, -1],
    "description": "Hexagon, sun straight down"
  },
  # Subpass 5: L-shape (complex)
  {
    "polygon": make_l_shape_poly(6, 6, 3),
    "sun_vector": [0.1, 0.1, -1],
    "description": "L-shape, slightly angled sun"
  },
  # Extreme cases
  {
    "polygon": make_rectangle_poly(50, 50),
    "sun_vector": [0.1, 0.1, -1],
    "description": "50x50 rectangle"
  },
  {
    "polygon": make_hexagon_poly(100),
    "sun_vector": [0, 0, -1],
    "description": "100x100 hexagon"
  },
  {
    "polygon": make_triangle_poly(500, 500),
    "sun_vector": [0.2, 0.2, -1],
    "description": "500x500 triangle"
  },
  {
    "polygon": make_rectangle_poly(1000, 1000),
    "sun_vector": [0.1, 0, -1],
    "description": "1000x1000 rectangle"
  },
  {
    "polygon": make_hexagon_poly(5000),
    "sun_vector": [0.3, 0, -1],
    "description": "5000x5000 hexagon (~25M sq units)"
  },
]


def polygon_area(poly: List[Tuple[float, float]]) -> float:
  """Calculate polygon area using shoelace formula."""
  n = len(poly)
  area = 0.0
  for i in range(n):
    j = (i + 1) % n
    area += poly[i][0] * poly[j][1]
    area -= poly[j][0] * poly[i][1]
  return abs(area) / 2.0


def polygon_bounds(poly: List[Tuple[float, float]]) -> Tuple[float, float, float, float]:
  """Get bounding box (min_x, min_y, max_x, max_y)."""
  xs = [p[0] for p in poly]
  ys = [p[1] for p in poly]
  return (min(xs), min(ys), max(xs), max(ys))


def format_polygon(poly: List[Tuple[float, float]]) -> str:
  """Format polygon for display."""
  return "[" + ", ".join(f"({x:.1f}, {y:.1f})" for x, y in poly) + "]"


def prepareSubpassPrompt(subPass: int) -> str:
  """Generate the prompt for subpass 0 that handles all problem complexities."""
  if subPass != 0:
    raise StopIteration

  return f"""You are solving a Shadow Covering problem with tetrahedrons in Rust.

**The Challenge:**
Your program will be tested with polygons ranging from simple shapes to very complex geometries.
The same program must work efficiently across ALL scales.

**Input format (stdin):**
Line 1: NV (number of polygon vertices)
Next NV lines: x y (polygon vertices on z=0 plane)
Last line: sx sy sz (sun direction vector, points from sun toward ground)

**Output format (stdout):**
Line 1: count (number of tetrahedrons)
Next count lines: px py pz qw qx qy qz scale
  px,py,pz: center position
  qw,qx,qy,qz: rotation quaternion (w is scalar)
  scale: uniform scale factor

**Standard Tetrahedron Reference:**
Vertices (before transform):
  [1.0, 0.0, -0.707]
  [-1.0, 0.0, -0.707]
  [0.0, 1.0, 0.707]
  [0.0, -1.0, 0.707]

**Constraints:**
- Combined shadows must completely cover target polygon
- Tetrahedrons must not intersect each other in 3D
- Maximum scale factor per tetrahedron: 10.0
- Minimize number of tetrahedrons

**Environment:**
{describe_this_pc()}

**Rust Compiler:**
{RustCompiler("test_engine").describe()}

Be aware that default warnings are enabled and will cause a compilation failure,
so ensure that you write warning-free code.

Write complete, compilable Rust code with a main() function.
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
      "Explain your shadow covering algorithm and how it adapts to different polygon complexities"
    },
    "rust_code": {
      "type": "string",
      "description": "Complete Rust code with main() that handles all scales"
    }
  },
  "required": ["reasoning", "rust_code"],
  "additionalProperties": False
}


def quaternion_rotate_point(q, p):
  """Rotate point p by quaternion q [w,x,y,z]."""
  w, qx, qy, qz = q
  px, py, pz = p

  # q * p * q^-1
  # Using quaternion multiplication formula
  t0 = w * w - qx * qx - qy * qy - qz * qz
  t1 = 2 * (qx * px + qy * py + qz * pz)
  t2 = 2 * w

  rx = t0 * px + t1 * qx + t2 * (qy * pz - qz * py)
  ry = t0 * py + t1 * qy + t2 * (qz * px - qx * pz)
  rz = t0 * pz + t1 * qz + t2 * (qx * py - qy * px)

  return [rx, ry, rz]


def transform_tetrahedron(position, quaternion, scale):
  """Get transformed tetrahedron vertices."""
  vertices = []
  for v in TETRA_VERTICES:
    # Scale
    scaled = [v[i] * scale for i in range(3)]
    # Rotate
    rotated = quaternion_rotate_point(quaternion, scaled)
    # Translate
    translated = [rotated[i] + position[i] for i in range(3)]
    vertices.append(translated)
  return vertices


def project_point_to_shadow(point, sun_vector):
  """Project 3D point to 2D shadow on z=0 plane."""
  if abs(sun_vector[2]) < 0.001:
    return (point[0], point[1])

  # Shadow projection
  t = -point[2] / sun_vector[2]
  shadow_x = point[0] + t * sun_vector[0]
  shadow_y = point[1] + t * sun_vector[1]
  return (shadow_x, shadow_y)


def convex_hull_2d(points):
  """Compute convex hull of 2D points using Graham scan."""
  if len(points) < 3:
    return points

  def cross(o, a, b):
    return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

  points = sorted(set(points))
  if len(points) <= 1:
    return points

  lower = []
  for p in points:
    while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
      lower.pop()
    lower.append(p)

  upper = []
  for p in reversed(points):
    while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
      upper.pop()
    upper.append(p)

  return lower[:-1] + upper[:-1]


def get_tetrahedron_shadow(position, quaternion, scale, sun_vector):
  """Get shadow polygon of a tetrahedron."""
  vertices = transform_tetrahedron(position, quaternion, scale)
  shadow_points = [project_point_to_shadow(v, sun_vector) for v in vertices]
  return convex_hull_2d(shadow_points)


def check_coverage_rasterized(shadows, target_polygon, resolution=None):
  """
    Check shadow coverage using PIL rasterization.
    Returns (coverage_ratio, covered_pixels, total_pixels).
    """
  bounds = polygon_bounds(target_polygon)
  min_x, min_y, max_x, max_y = bounds

  width = max_x - min_x
  height = max_y - min_y

  if width <= 0 or height <= 0:
    return 1.0, 0, 0

  # Scale resolution with target size for accuracy, capped for performance
  if resolution is None:
    resolution = max(100, min(500, int(max(width, height))))

  scale_x = resolution / width
  scale_y = resolution / height

  def world_to_pixel(poly):
    return [(int((x - min_x) * scale_x), int((y - min_y) * scale_y)) for x, y in poly]

  # Rasterize target polygon
  target_img = Image.new('L', (resolution, resolution), 0)
  target_draw = ImageDraw.Draw(target_img)
  target_draw.polygon(world_to_pixel(target_polygon), fill=255)
  target_arr = np.array(target_img)

  # Rasterize union of all shadows
  shadow_img = Image.new('L', (resolution, resolution), 0)
  shadow_draw = ImageDraw.Draw(shadow_img)
  for shadow in shadows:
    if len(shadow) >= 3:
      shadow_draw.polygon(world_to_pixel(shadow), fill=255)
  shadow_arr = np.array(shadow_img)

  # Compute coverage
  target_mask = target_arr > 0
  total_in_target = int(np.count_nonzero(target_mask))
  if total_in_target == 0:
    return 1.0, 0, 0

  covered = int(np.count_nonzero(target_mask & (shadow_arr > 0)))
  return covered / total_in_target, covered, total_in_target


def _quat_rotate_batch(q, points):
  """
  Rotate an array of points by quaternion q [w,x,y,z] using numpy.
  q: (4,) array  [w, x, y, z]
  points: (N, 3) array
  Returns: (N, 3) array
  """
  w, qx, qy, qz = q
  px, py, pz = points[:, 0], points[:, 1], points[:, 2]

  t0 = w * w - qx * qx - qy * qy - qz * qz
  t1 = 2 * (qx * px + qy * py + qz * pz)
  t2 = 2 * w

  rx = t0 * px + t1 * qx + t2 * (qy * pz - qz * py)
  ry = t0 * py + t1 * qy + t2 * (qz * px - qx * pz)
  rz = t0 * pz + t1 * qz + t2 * (qx * py - qy * px)

  return np.column_stack([rx, ry, rz])


def _batch_compute_shadows(placements, sun_vector):
  """
  Vectorized shadow computation for all placements at once.
  Returns list of shadow polygons (as lists of tuples).
  """
  n = len(placements)
  if n == 0:
    return []

  sun = np.array(sun_vector, dtype=np.float64)
  # For each placement, transform 4 tetra vertices -> project -> convex hull
  shadows = []
  for p in placements:
    pos = np.array(p.get("position", [0, 0, 1]), dtype=np.float64)
    quat = np.array(p.get("quaternion", [1, 0, 0, 0]), dtype=np.float64)
    scale = p.get("scale", 1.0)

    # Scale + rotate + translate all 4 vertices at once
    scaled = TETRA_VERTICES_NP * scale
    rotated = _quat_rotate_batch(quat, scaled)
    transformed = rotated + pos

    # Project to shadow plane (z=0)
    if abs(sun[2]) < 0.001:
      shadow_pts = transformed[:, :2]
    else:
      t = -transformed[:, 2] / sun[2]
      shadow_x = transformed[:, 0] + t * sun[0]
      shadow_y = transformed[:, 1] + t * sun[1]
      shadow_pts = np.column_stack([shadow_x, shadow_y])

    pts_tuples = [tuple(row) for row in shadow_pts]
    hull = convex_hull_2d(pts_tuples)
    shadows.append(hull)

  return shadows


def validate_solution(solution, target_polygon, sun_vector):
  """Validate shadow covering solution."""
  if not isinstance(solution, dict):
    return False, "Solution must be a dict", 0, 0

  if "count" not in solution or "placements" not in solution:
    return False, "Missing 'count' or 'placements'", 0, 0

  count = solution["count"]
  placements = solution["placements"]

  if len(placements) != count:
    return False, f"Placement count mismatch", 0, 0

  # Validate scales
  for i, p in enumerate(placements):
    scale = p.get("scale", 1.0)
    if scale > MAX_TETRA_SCALE:
      return False, f"Tetrahedron {i} scale {scale:.1f} exceeds max {MAX_TETRA_SCALE}", 0, 0
    if scale <= 0:
      return False, f"Tetrahedron {i} has invalid scale {scale}", 0, 0

  # Batch compute all shadows using numpy
  shadows = _batch_compute_shadows(placements, sun_vector)

  # Check coverage
  coverage, covered, total = check_coverage_rasterized(shadows, target_polygon)

  return True, "", coverage, count


def get_baseline_count(target_polygon, sun_vector):
  """Estimate baseline tetrahedron count using simple grid placement."""
  bounds = polygon_bounds(target_polygon)
  area = polygon_area(target_polygon)

  # Estimate shadow size of one scaled tetrahedron
  shadow_per_tetra = TETRA_SIZE * TETRA_SIZE * 0.8  # Rough estimate

  # Estimate count needed
  return max(1, int(area / shadow_per_tetra) + 1)


def format_input(target_polygon, sun_vector) -> str:
  lines = [str(len(target_polygon))]
  for x, y in target_polygon:
    lines.append(f"{x:.6f} {y:.6f}")
  lines.append(f"{sun_vector[0]:.6f} {sun_vector[1]:.6f} {sun_vector[2]:.6f}")
  return "\n".join(lines)


def parse_placements_output(output: str) -> tuple:
  text = output.strip()
  if not text:
    return None, "Empty output"

  lines = [l for l in text.splitlines() if l.strip()]
  if not lines:
    return None, "No output lines"

  try:
    count = int(lines[0])
  except ValueError:
    return None, "First line must be count integer"

  if count == 0:
    return {'count': 0, 'placements': []}, None

  if len(lines) < 1 + count:
    return None, f"Expected {count} placement lines, got {len(lines) - 1}"

  placements = []
  for i in range(1, 1 + count):
    parts = lines[i].split()
    if len(parts) < 8:
      return None, f"Placement line {i} needs 8 values (px py pz qw qx qy qz scale)"
    try:
      px, py, pz = float(parts[0]), float(parts[1]), float(parts[2])
      qw, qx, qy, qz = float(parts[3]), float(parts[4]), float(parts[5]), float(parts[6])
      scale = float(parts[7])
    except ValueError:
      return None, f"Invalid values in placement line {i}"
    placements.append({
      'position': [px, py, pz],
      'quaternion': [qw, qx, qy, qz],
      'scale': scale,
    })

  return {'count': count, 'placements': placements}, None


def execute_solver(code: str,
                   target_polygon,
                   sun_vector,
                   ai_engine_name: str,
                   timeout: int = TIMEOUT_SECONDS):
  """Execute the LLM's solver."""
  input_data = format_input(target_polygon, sun_vector)
  run = compile_and_run(code, "rust", ai_engine_name, input_data=input_data, timeout=timeout)

  if not run:
    return None, run.error_message(), run.exec_time

  solution, parse_error = parse_placements_output(run.stdout)
  if parse_error:
    return None, parse_error, run.exec_time

  return solution, None, run.exec_time


_grade_cache = GradeCache('test15')


def _cache_key_parts(result, subPass):
  """Build cache key from code hash + subpass."""
  code = result.get("rust_code", "") if isinstance(result, dict) else ""
  code_hash = hashlib.sha256(code.encode('utf-8')).hexdigest()[:16]
  return (code_hash, str(subPass))


def gradeAnswer(result: dict, subPass: int, aiEngineName: str) -> tuple:
  """Grade the shadow covering solver."""
  if not result:
    return 0.0, "No result provided"

  if "rust_code" not in result:
    return 0.0, "No Rust code provided"

  # Check cache
  cache_parts = _cache_key_parts(result, subPass)
  cached = _grade_cache.get_grade(*cache_parts)
  if cached is not None:
    return cached

  case = TEST_CASES[subPass]
  target = case["polygon"]
  sun = case["sun_vector"]
  description = case["description"]
  code = result["rust_code"]

  # Execute solver
  solution, error, exec_time = execute_solver(code, target, sun, aiEngineName)

  global lastSolution
  global lastCase
  lastSolution = solution
  lastCase = case

  if error:
    grade = (0.0, f"[{description}] {error}")
    _grade_cache.put_grade(grade, *cache_parts)
    return grade

  # Validate solution
  is_valid, validation_error, coverage, count = validate_solution(solution, target, sun)

  if not is_valid:
    grade = (0.0, f"[{description}] {validation_error}")
    _grade_cache.put_grade(grade, *cache_parts)
    return grade

  # Score based on coverage
  if coverage >= 0.95:
    coverage_score = 1.0
  elif coverage >= 0.9:
    coverage_score = 0.85
  elif coverage >= 0.8:
    coverage_score = 0.7
  elif coverage >= 0.7:
    coverage_score = 0.5
  else:
    coverage_score = 0.0

  quality = f"{coverage*100:.0f}% coverage with {count} tetrahedrons"

  explanation = (f"[{description}] {quality}, Time: {exec_time:.1f}s")

  grade = (coverage_score, explanation)
  _grade_cache.put_grade(grade, *cache_parts)
  return grade


lastSolution = None
lastCase = None


def resultToNiceReport(result: dict, subPass: int, aiEngineName: str) -> str:
  """Generate HTML report."""
  if not result:
    return "<p style='color:red'>No result provided</p>"

  cache_parts = _cache_key_parts(result, subPass)
  cached = _grade_cache.get_report(*cache_parts)
  if cached is not None:
    return cached

  case = TEST_CASES[subPass]

  html = f"<h4>Shadow Covering - {case['description']}</h4>"

  if subPass == 0:

    if "reasoning" in result:
      reasoning = result['reasoning'][:500] + ('...'
                                               if len(result.get('reasoning', '')) > 500 else '')
      html += f"<p><strong>Algorithm:</strong> {reasoning}</p>"

    if "rust_code" in result:
      code = result["rust_code"]
      code_escaped = code.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
      html += f"<details><summary>View Code ({len(code)} chars)</summary><pre>{code_escaped}</pre></details>"

  global lastSolution
  if lastSolution and isinstance(lastSolution, dict):
    try:
      placements = lastSolution.get('placements')
      if isinstance(placements, list) and len(placements) > 0:
        target_poly = case.get('polygon', [])
        sun = case.get('sun_vector', [0, 0, -1])

        max_show = 200
        if len(placements) > max_show:
          viz_placements = random.sample(placements, max_show)
        else:
          viz_placements = placements

        bounds = polygon_bounds(target_poly) if target_poly else (0.0, 0.0, 1.0, 1.0)
        min_x, min_y, max_x, max_y = bounds
        cx = (min_x + max_x) / 2.0
        cy = (min_y + max_y) / 2.0

        viz_id = f"shadow3d_{subPass}_{abs(hash((aiEngineName, subPass, time.time()))) % 10000000}"

        poly_data = json.dumps([list(p) for p in target_poly])
        placements_data = json.dumps(viz_placements)
        sun_data = json.dumps(list(sun))
        tetra_data = json.dumps([list(v) for v in TETRA_VERTICES])

        html += f"""
        <div class="shadow-visualization" style="margin: 15px 0;">
            <details>
                <summary style="cursor: pointer; padding: 8px; background: #e8e8e8; border-radius: 4px; font-weight: bold; color: #333; border: 1px solid #ccc;">
                    3D Visualization: {len(viz_placements)}/{len(placements)} tetrahedrons
                </summary>
                <div style="margin-top: 10px;">
                    <div id="shadow3d-renderer-{viz_id}" style="width: 100%; height: 500px; border: 1px solid #ccc; background: #fafafa; border-radius: 4px; display: flex; align-items: center; justify-content: center; color: #999;">
                        <span class="viz-placeholder">Scroll here to activate 3D view</span>
                    </div>
                    <div style="margin-top: 8px; font-size: 12px; color: #666; background: #f8f8f8; padding: 5px; border-radius: 3px;">
                        Left-drag rotate, Right-drag pan, Scroll zoom
                    </div>
                </div>
            </details>
        </div>

        <script src="https://cdn.jsdelivr.net/npm/three@0.132.2/build/three.min.js"></script>
        <script src="https://cdn.jsdelivr.net/npm/three@0.132.2/examples/js/controls/OrbitControls.js"></script>
        <script>
        (function() {{
            const vizId = 'shadow3d-renderer-{viz_id}';
            const targetPolyData = {poly_data};
            const placementsData = {placements_data};
            const sunVectorData = {sun_data};
            const tetraVertsData = {tetra_data};
            const centerXYData = [{cx}, {cy}];
            const boundsData = {{ minx: {min_x}, miny: {min_y}, maxx: {max_x}, maxy: {max_y} }};
            
            let scene, camera, renderer, controls, animationId;
            let isActive = false;
            
            function activate() {{
                if (isActive) return;
                isActive = true;
                
                if (typeof THREE === 'undefined') return;
                const containerEl = document.getElementById(vizId);
                if (!containerEl) return;
                
                const placeholder = containerEl.querySelector('.viz-placeholder');
                if (placeholder) placeholder.style.display = 'none';

                scene = new THREE.Scene();
                scene.background = new THREE.Color(0xf0f0f0);

                renderer = new THREE.WebGLRenderer({{ antialias: true }});
                renderer.setSize(containerEl.clientWidth, containerEl.clientHeight);
                containerEl.appendChild(renderer.domElement);

                camera = new THREE.PerspectiveCamera(60, containerEl.clientWidth / containerEl.clientHeight, 0.1, 1e12);

                controls = new THREE.OrbitControls(camera, renderer.domElement);
                controls.enableDamping = true;
                controls.dampingFactor = 0.05;

                const ambient = new THREE.AmbientLight(0xffffff, 0.6);
                scene.add(ambient);
                const dir = new THREE.DirectionalLight(0xffffff, 0.8);
                dir.position.set(1, 2, 3);
                scene.add(dir);

                const root = new THREE.Group();
                scene.add(root);
                root.position.set(-centerXYData[0], -centerXYData[1], 0);

                if (targetPolyData && targetPolyData.length >= 3) {{
                    const pts = [];
                    for (let i = 0; i < targetPolyData.length; i++) {{
                        const p = targetPolyData[i];
                        pts.push(new THREE.Vector3(p[0], p[1], 0));
                    }}
                    const geomLine = new THREE.BufferGeometry().setFromPoints(pts);
                    const matLine = new THREE.LineBasicMaterial({{ color: 0x111111 }});
                    const line = new THREE.LineLoop(geomLine, matLine);
                    root.add(line);

                    const planeW = Math.max(1, (boundsData.maxx - boundsData.minx) * 1.1);
                    const planeH = Math.max(1, (boundsData.maxy - boundsData.miny) * 1.1);
                    const grid = new THREE.GridHelper(Math.max(planeW, planeH), 20, 0x999999, 0xcccccc);
                    grid.rotation.x = Math.PI / 2;
                    grid.position.set(centerXYData[0], centerXYData[1], 0);
                    root.add(grid);
                }}

                const positions = [];
                for (let i = 0; i < tetraVertsData.length; i++) {{
                    const v = tetraVertsData[i];
                    positions.push(v[0], v[1], v[2]);
                }}
                const indices = [0, 1, 2, 0, 2, 3, 0, 3, 1, 1, 3, 2];
                const tetraGeom = new THREE.BufferGeometry();
                tetraGeom.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
                tetraGeom.setIndex(indices);
                tetraGeom.computeVertexNormals();

                for (let i = 0; i < placementsData.length; i++) {{
                    const p = placementsData[i] || {{}};
                    const pos = p.position || [0, 0, 1];
                    const q = p.quaternion || [1, 0, 0, 0];
                    const s = (p.scale === undefined) ? 1.0 : p.scale;
                    const color = new THREE.Color().setHSL((i * 37 % 360) / 360, 0.6, 0.55);
                    const mat = new THREE.MeshPhongMaterial({{ color: color, transparent: true, opacity: 0.75, side: THREE.DoubleSide }});
                    const m = new THREE.Mesh(tetraGeom, mat);
                    m.position.set(pos[0], pos[1], pos[2]);
                    m.quaternion.set(q[1], q[2], q[3], q[0]);
                    m.scale.set(s, s, s);
                    root.add(m);
                }}

                const sunDir = new THREE.Vector3(sunVectorData[0], sunVectorData[1], sunVectorData[2]);
                if (sunDir.length() > 0) {{
                    sunDir.normalize();
                    const arrow = new THREE.ArrowHelper(sunDir.clone().multiplyScalar(-1), new THREE.Vector3(centerXYData[0], centerXYData[1], 0), Math.max(1, Math.max(boundsData.maxx - boundsData.minx, boundsData.maxy - boundsData.miny) * 0.25), 0xff8800);
                    root.add(arrow);
                }}

                const box = new THREE.Box3().setFromObject(root);
                const center = box.getCenter(new THREE.Vector3());
                const size = box.getSize(new THREE.Vector3());
                const maxDim = Math.max(size.x, size.y, size.z);
                const fov = camera.fov * Math.PI / 180;
                let cameraZ = Math.abs(maxDim / 2 / Math.tan(fov / 2));
                cameraZ *= 1.6;
                camera.position.set(center.x + cameraZ, center.y + cameraZ, center.z + cameraZ);
                camera.lookAt(center);
                controls.target.copy(center);
                controls.update();

                function animate() {{
                    if (!isActive) return;
                    animationId = requestAnimationFrame(animate);
                    controls.update();
                    renderer.render(scene, camera);
                }}
                animate();
            }}
            
            function dispose() {{
                if (!isActive) return;
                isActive = false;
                
                if (animationId) {{
                    cancelAnimationFrame(animationId);
                    animationId = null;
                }}
                
                const containerEl = document.getElementById(vizId);
                
                if (renderer) {{
                    renderer.dispose();
                    if (renderer.domElement && renderer.domElement.parentNode) {{
                        renderer.domElement.parentNode.removeChild(renderer.domElement);
                    }}
                    renderer = null;
                }}
                
                if (scene) {{
                    scene.traverse(function(object) {{
                        if (object.geometry) object.geometry.dispose();
                        if (object.material) {{
                            if (Array.isArray(object.material)) {{
                                object.material.forEach(m => m.dispose());
                            }} else {{
                                object.material.dispose();
                            }}
                        }}
                    }});
                    scene = null;
                }}
                
                camera = null;
                controls = null;
                
                if (containerEl) {{
                    const placeholder = containerEl.querySelector('.viz-placeholder');
                    if (placeholder) placeholder.style.display = '';
                }}
            }}
            
            if (window.VizManager) {{
                window.VizManager.register({{
                    id: vizId,
                    containerId: vizId,
                    activate: activate,
                    dispose: dispose
                }});
            }} else {{
                activate();
            }}
        }})();
        </script>
        """
    except Exception as e:
      html += f"<p style='color:orange;'>3D visualization error: {str(e)}</p>"

  _grade_cache.put_report(html, *cache_parts)
  return html


highLevelSummary = """
<p>Place 3D tetrahedra (triangular pyramids) above a flat surface so that their
shadows, cast by a directional light, completely cover a given 2D target shape.
Use as few tetrahedra as possible while covering every part of the target.</p>
<p>The AI must reason about 3D-to-2D projection, rotation, and coverage geometry.
Subpasses use more complex target shapes and trickier light angles. The baseline
uses a simple grid of tetrahedra above the target.</p>
"""
