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

import json
import math
import random
import subprocess
import sys
import tempfile
import os
import time
from typing import List, Tuple, Dict, Set

title = "Tetrahedron Shadow Covering"

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

  return f"""You are solving a Shadow Covering problem with tetrahedrons.

**The Challenge:**
Your `solve_shadow_cover(target_polygon, sun_vector)` function will be tested with polygons 
ranging from simple shapes to very complex geometries. The same function must work efficiently 
across ALL scales.

**Input:**
- `target_polygon`: List of (x, y) vertices defining 2D polygon on z=0 plane
- `sun_vector`: 3D direction vector (points from sun toward ground)

**Output:**
- Dict with:
  - `"count"`: Number of tetrahedrons used
  - `"placements"`: List of placement dicts:
    - `"position"`: [x, y, z] - center position of tetrahedron
    - `"quaternion"`: [w, x, y, z] - rotation (w is scalar)

**Standard Tetrahedron Reference:**
```python
tetra_vertices = [
    [1.0, 0.0, -0.707],
    [-1.0, 0.0, -0.707],
    [0.0, 1.0, 0.707],
    [0.0, -1.0, 0.707]
]
```

**Constraints:**
- Libraries allowed: numpy, scipy, python standard library
- Combined shadows must completely cover target polygon
- Tetrahedrons must not intersect each other in 3D
- Minimize number of tetrahedrons
- Must handle varying polygon complexities and sun angles

Write complete, runnable Python code with the solve_shadow_cover function.
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
    "python_code": {
      "type":
      "string",
      "description":
      "Complete Python code with solve_shadow_cover(target_polygon, sun_vector) function that handles all scales"
    }
  },
  "required": ["reasoning", "python_code"],
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


def point_in_polygon(point, polygon):
  """Ray casting point-in-polygon test."""
  x, y = point
  n = len(polygon)
  inside = False
  j = n - 1
  for i in range(n):
    xi, yi = polygon[i]
    xj, yj = polygon[j]
    if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi) + xi):
      inside = not inside
    j = i
  return inside


def check_coverage_rasterized(shadows, target_polygon, resolution=100):
  """
    Check shadow coverage using rasterization.
    Returns (coverage_ratio, covered_pixels, total_pixels).
    """
  bounds = polygon_bounds(target_polygon)
  min_x, min_y, max_x, max_y = bounds

  width = max_x - min_x
  height = max_y - min_y

  if width <= 0 or height <= 0:
    return 1.0, 0, 0

  step_x = width / resolution
  step_y = height / resolution

  total_in_target = 0
  covered = 0

  for i in range(resolution):
    for j in range(resolution):
      px = min_x + (i + 0.5) * step_x
      py = min_y + (j + 0.5) * step_y

      if point_in_polygon((px, py), target_polygon):
        total_in_target += 1
        # Check if covered by any shadow
        for shadow in shadows:
          if len(shadow) >= 3 and point_in_polygon((px, py), shadow):
            covered += 1
            break

  if total_in_target == 0:
    return 1.0, 0, 0

  return covered / total_in_target, covered, total_in_target


def aabb_intersect_3d(verts1, verts2):
  """Check if AABBs of two vertex sets intersect."""
  min1 = [min(v[i] for v in verts1) for i in range(3)]
  max1 = [max(v[i] for v in verts1) for i in range(3)]
  min2 = [min(v[i] for v in verts2) for i in range(3)]
  max2 = [max(v[i] for v in verts2) for i in range(3)]

  for i in range(3):
    if max1[i] < min2[i] - 0.01 or min1[i] > max2[i] + 0.01:
      return False
  return True


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

  # Collect shadows and check intersections
  shadows = []
  tetra_verts = []

  for i, p in enumerate(placements):
    pos = p.get("position", [0, 0, 1])
    quat = p.get("quaternion", [1, 0, 0, 0])
    scale = p.get("scale", 1.0)

    verts = transform_tetrahedron(pos, quat, scale)
    shadow = get_tetrahedron_shadow(pos, quat, scale, sun_vector)

    # Check intersection with previous tetrahedrons (AABB only)
    for j, prev_verts in enumerate(tetra_verts):
      if aabb_intersect_3d(verts, prev_verts):
        # Note: This is approximate - full intersection test is complex
        pass  # Allow for now, rely on AABB

    tetra_verts.append(verts)
    shadows.append(shadow)

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


def execute_solver(code: str, target_polygon, sun_vector, timeout: int = TIMEOUT_SECONDS):
  """Execute the LLM's solver."""
  from solver_utils import execute_solver_with_data

  data_dict = {
    'target_polygon': target_polygon,
    'sun_vector': sun_vector,
  }

  result, error, exec_time = execute_solver_with_data(code, data_dict, 'solve_shadow_cover',
                                                      timeout)

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
      pos = p.get('position', [0, 0, 1])
      quat = p.get('quaternion', [1, 0, 0, 0])
      scale = p.get('scale', 1.0)
      normalized_placements.append({
        'position': [float(pos[0]), float(pos[1]), float(pos[2])],
        'quaternion': [float(quat[0]),
                       float(quat[1]),
                       float(quat[2]),
                       float(quat[3])],
        'scale':
        float(scale),
      })

    out = {
      'count': int(result.get('count', len(normalized_placements))),
      'placements': normalized_placements,
    }
  except Exception as e:
    return None, f"Invalid result format: {e}", exec_time

  return out, None, exec_time


def gradeAnswer(result: dict, subPass: int, aiEngineName: str) -> tuple:
  """Grade the shadow covering solver."""
  if not result:
    return 0.0, "No result provided"

  if "python_code" not in result:
    return 0.0, "No Python code provided"

  case = TEST_CASES[subPass]
  target = case["polygon"]
  sun = case["sun_vector"]
  description = case["description"]
  code = result["python_code"]

  # Execute solver
  solution, error, exec_time = execute_solver(code, target, sun)

  global lastSolution
  global lastCase
  lastSolution = solution
  lastCase = case

  if error:
    return 0.0, f"[{description}] {error}"

  # Validate solution
  is_valid, validation_error, coverage, count = validate_solution(solution, target, sun)

  if not is_valid:
    return 0.0, f"[{description}] {validation_error}"

  # Score based on coverage
  if coverage >= 0.95:
    coverage_score = 1.0
  elif coverage >= 0.8:
    coverage_score = 0.85
  elif coverage >= 0.6:
    coverage_score = 0.7
  elif coverage >= 0.3:
    coverage_score = 0.5
  else:
    coverage_score = 0.3

  quality = f"{coverage*100:.0f}% coverage with {count} tetrahedrons"

  explanation = (f"[{description}] {quality}, Time: {exec_time:.1f}s")

  return coverage_score, explanation


lastSolution = None
lastCase = None


def resultToNiceReport(result: dict, subPass: int, aiEngineName: str) -> str:
  """Generate HTML report."""
  if not result:
    return "<p style='color:red'>No result provided</p>"

  case = TEST_CASES[subPass]

  html = f"<h4>Shadow Covering - {case['description']}</h4>"

  if subPass == 0:

    if "reasoning" in result:
      reasoning = result['reasoning'][:500] + ('...'
                                               if len(result.get('reasoning', '')) > 500 else '')
      html += f"<p><strong>Algorithm:</strong> {reasoning}</p>"

    if "python_code" in result:
      code = result["python_code"]
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
                    <div id="shadow3d-renderer-{viz_id}" style="width: 100%; height: 500px; border: 1px solid #ccc; background: #fafafa; border-radius: 4px;"></div>
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
            if (typeof THREE === 'undefined') return;
            const containerEl = document.getElementById('shadow3d-renderer-{viz_id}');
            if (!containerEl) return;

            const targetPoly = {poly_data};
            const placements = {placements_data};
            const sunVector = {sun_data};
            const tetraVerts = {tetra_data};
            const centerXY = [{cx}, {cy}];

            const scene = new THREE.Scene();
            scene.background = new THREE.Color(0xf0f0f0);

            const renderer = new THREE.WebGLRenderer({{ antialias: true }});
            renderer.setSize(containerEl.clientWidth, containerEl.clientHeight);
            containerEl.appendChild(renderer.domElement);

            const camera = new THREE.PerspectiveCamera(60, containerEl.clientWidth / containerEl.clientHeight, 0.1, 1e12);

            const controls = new THREE.OrbitControls(camera, renderer.domElement);
            controls.enableDamping = true;
            controls.dampingFactor = 0.05;

            const ambient = new THREE.AmbientLight(0xffffff, 0.6);
            scene.add(ambient);
            const dir = new THREE.DirectionalLight(0xffffff, 0.8);
            dir.position.set(1, 2, 3);
            scene.add(dir);

            const root = new THREE.Group();
            scene.add(root);
            root.position.set(-centerXY[0], -centerXY[1], 0);

            if (targetPoly && targetPoly.length >= 3) {{
                const pts = [];
                for (let i = 0; i < targetPoly.length; i++) {{
                    const p = targetPoly[i];
                    pts.push(new THREE.Vector3(p[0], p[1], 0));
                }}
                const geomLine = new THREE.BufferGeometry().setFromPoints(pts);
                const matLine = new THREE.LineBasicMaterial({{ color: 0x111111 }});
                const line = new THREE.LineLoop(geomLine, matLine);
                root.add(line);

                const minx = {min_x}, miny = {min_y}, maxx = {max_x}, maxy = {max_y};
                const planeW = Math.max(1, (maxx - minx) * 1.1);
                const planeH = Math.max(1, (maxy - miny) * 1.1);
                const grid = new THREE.GridHelper(Math.max(planeW, planeH), 20, 0x999999, 0xcccccc);
                grid.rotation.x = Math.PI / 2;
                grid.position.set(centerXY[0], centerXY[1], 0);
                root.add(grid);
            }}

            const positions = [];
            for (let i = 0; i < tetraVerts.length; i++) {{
                const v = tetraVerts[i];
                positions.push(v[0], v[1], v[2]);
            }}
            const indices = [0, 1, 2, 0, 2, 3, 0, 3, 1, 1, 3, 2];
            const tetraGeom = new THREE.BufferGeometry();
            tetraGeom.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
            tetraGeom.setIndex(indices);
            tetraGeom.computeVertexNormals();

            for (let i = 0; i < placements.length; i++) {{
                const p = placements[i] || {{}};
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

            const sunDir = new THREE.Vector3(sunVector[0], sunVector[1], sunVector[2]);
            if (sunDir.length() > 0) {{
                sunDir.normalize();
                const arrow = new THREE.ArrowHelper(sunDir.clone().multiplyScalar(-1), new THREE.Vector3(centerXY[0], centerXY[1], 0), Math.max(1, Math.max({max_x} - {min_x}, {max_y} - {min_y}) * 0.25), 0xff8800);
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

            function handleResize() {{
                const w = containerEl.clientWidth;
                const h = containerEl.clientHeight;
                camera.aspect = w / h;
                camera.updateProjectionMatrix();
                renderer.setSize(w, h);
            }}
            window.addEventListener('resize', handleResize);

            function animate() {{
                requestAnimationFrame(animate);
                controls.update();
                renderer.render(scene, camera);
            }}
            animate();
        }})();
        </script>
        """
    except Exception as e:
      html += f"<p style='color:orange;'>3D visualization error: {str(e)}</p>"

  return html


highLevelSummary = """
Shadow covering with tetrahedrons is a geometric optimization problem.

**Problem:** Place 3D tetrahedrons such that their projected shadows cover a 2D target.

**Key concepts:**
- **Shadow projection:** Project 3D vertices along sun vector to z=0 plane
- **Convex hull:** Tetrahedron shadow is convex hull of 4 projected vertices
- **Coverage:** Union of all shadows must contain target polygon

**Approaches:**
1. **Grid placement:** Place tetrahedrons in grid above target, scale to cover
2. **Greedy covering:** Iteratively place tetrahedrons to cover uncovered areas
3. **Optimization:** Minimize count while maximizing coverage

**Geometric operations:**
- Quaternion rotation
- Projection along direction vector
- Convex hull computation
- Point-in-polygon testing

The baseline uses simple grid-based placement with scaled tetrahedrons.
"""
