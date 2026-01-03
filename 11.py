"""
Test 11: 2D Polygon Cutting Stock Problem

The LLM must write a Python solver for cutting small polygons from large polygons.
Given a large stock polygon shape and a list of small polygons needed,
minimize the number of stock polygons purchased.

This is more complex than rectangular cutting - arbitrary polygon shapes.

Subpasses test increasingly complex polygon sets.
Solver times out after 5 minutes.
"""

import math
import random
import subprocess
import sys
import tempfile
import os
import time
from typing import List, Tuple, Dict

title = "2D Polygon Cutting Stock"

# Timeout in seconds (30 seconds for testing)
TIMEOUT_SECONDS = 30

# Seed for reproducibility
RANDOM_SEED = 66666


def make_rectangle(w: float, h: float) -> List[Tuple[float, float]]:
  """Create rectangle polygon centered at origin."""
  return [(0, 0), (w, 0), (w, h), (0, h)]


def make_triangle(base: float, height: float) -> List[Tuple[float, float]]:
  """Create isoceles triangle."""
  return [(0, 0), (base, 0), (base / 2, height)]


def make_hexagon(size: float) -> List[Tuple[float, float]]:
  """Create regular hexagon."""
  points = []
  for i in range(6):
    angle = math.pi / 3 * i
    points.append((size * math.cos(angle), size * math.sin(angle)))
  return points


def make_pentagon(size: float) -> List[Tuple[float, float]]:
  """Create regular pentagon."""
  points = []
  for i in range(5):
    angle = math.pi / 2 + 2 * math.pi / 5 * i
    points.append((size * math.cos(angle), size * math.sin(angle)))
  return points


def make_l_shape(w: float, h: float, notch: float) -> List[Tuple[float, float]]:
  """Create L-shaped polygon."""
  return [(0, 0), (w, 0), (w, notch), (notch, notch), (notch, h), (0, h)]


def translate_polygon(poly: List[Tuple[float, float]], dx: float,
                      dy: float) -> List[Tuple[float, float]]:
  """Translate polygon by (dx, dy)."""
  return [(x + dx, y + dy) for x, y in poly]


def rotate_polygon(poly: List[Tuple[float, float]], angle: float) -> List[Tuple[float, float]]:
  """Rotate polygon around origin by angle (radians)."""
  cos_a = math.cos(angle)
  sin_a = math.sin(angle)
  return [(x * cos_a - y * sin_a, x * sin_a + y * cos_a) for x, y in poly]


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


# Test configurations
TEST_CASES = [
  # Subpass 0: Simple - rectangles from rectangle
  {
    "stock_polygon":
    make_rectangle(100, 100),
    "pieces": [
      make_rectangle(30, 40),
      make_rectangle(25, 35),
      make_rectangle(40, 30),
      make_rectangle(20, 50),
    ],
    "description":
    "4 rectangles from 100x100 square"
  },
  # Subpass 1: Triangles from rectangle
  {
    "stock_polygon":
    make_rectangle(100, 80),
    "pieces": [
      make_triangle(30, 40),
      make_triangle(40, 35),
      make_triangle(25, 45),
      make_triangle(35, 30),
      make_triangle(30, 35),
    ],
    "description":
    "5 triangles from 100x80 rectangle"
  },
  # Subpass 2: Mixed shapes
  {
    "stock_polygon":
    make_rectangle(120, 100),
    "pieces": [
      make_rectangle(40, 30),
      make_triangle(35, 40),
      make_rectangle(30, 45),
      make_triangle(40, 35),
      make_rectangle(25, 50),
      make_triangle(30, 40),
    ],
    "description":
    "6 mixed shapes from 120x100 rectangle"
  },
  # Subpass 3: Hexagonal stock
  {
    "stock_polygon":
    make_hexagon(50),
    "pieces": [
      make_triangle(25, 30),
      make_triangle(20, 25),
      make_rectangle(20, 25),
      make_triangle(25, 25),
      make_rectangle(15, 20),
    ],
    "description":
    "5 pieces from hexagon stock"
  },
  # Subpass 4: Many small pieces
  {
    "stock_polygon":
    make_rectangle(100, 100),
    "pieces": [
      make_rectangle(20, 25),
      make_rectangle(25, 20),
      make_triangle(22, 28),
      make_rectangle(18, 30),
      make_triangle(25, 22),
      make_rectangle(22, 22),
      make_triangle(20, 25),
      make_rectangle(28, 18),
      make_triangle(24, 26),
      make_rectangle(20, 28),
    ],
    "description":
    "10 pieces from 100x100 square"
  },
  # Subpass 5: Complex - L-shapes and pentagons
  {
    "stock_polygon":
    make_rectangle(150, 120),
    "pieces": [
      make_l_shape(40, 50, 20),
      make_pentagon(18),
      make_rectangle(35, 40),
      make_l_shape(35, 45, 18),
      make_triangle(40, 45),
      make_pentagon(15),
      make_rectangle(30, 35),
      make_l_shape(30, 40, 15),
    ],
    "description":
    "8 complex pieces from 150x120 rectangle"
  },
  # Extreme cases
  {
    "stock_polygon":
    make_rectangle(500, 500),
    "pieces": [
      make_rectangle(
        random.Random(RANDOM_SEED + 6).randint(20, 80),
        random.Random(RANDOM_SEED + 6 + i).randint(20, 80)) for i in range(50)
    ],
    "description":
    "50 pieces from 500x500"
  },
  {
    "stock_polygon":
    make_triangle(1000, 1000),
    "pieces": [
      make_rectangle(
        random.Random(RANDOM_SEED + 7).randint(30, 100),
        random.Random(RANDOM_SEED + 7 + i).randint(30, 100)) for i in range(100)
    ],
    "description":
    "100 pieces from 1000x1000"
  },
  {
    "stock_polygon":
    make_rectangle(2000, 2000),
    "pieces": [
      make_rectangle(
        random.Random(RANDOM_SEED + 8).randint(40, 150),
        random.Random(RANDOM_SEED + 8 + i).randint(40, 150)) for i in range(500)
    ],
    "description":
    "500 pieces from 2000x2000"
  },
  {
    "stock_polygon":
    make_pentagon(5000),
    "pieces": [
      make_rectangle(
        random.Random(RANDOM_SEED + 9).randint(50, 200),
        random.Random(RANDOM_SEED + 9 + i).randint(50, 200)) for i in range(1000)
    ],
    "description":
    "1000 pieces from 5000x5000"
  },
  {
    "stock_polygon":
    make_triangle(10000, 10000),
    "pieces": [
      make_rectangle(
        random.Random(RANDOM_SEED + 10).randint(50, 300),
        random.Random(RANDOM_SEED + 10 + i).randint(50, 300)) for i in range(5000)
    ],
    "description":
    "5000 pieces from 10000x10000"
  },
]


def prepareSubpassPrompt(subPass: int) -> str:
  if subPass != 0:
    raise StopIteration

  return f"""You are solving a 2D Polygon Cutting Stock Problem.

**The Challenge:**
Your `solve_polygon_cutting(stock_polygon, pieces)` function will be tested with problems 
ranging from 2 simple pieces to 100+ complex polygons. The same function must work efficiently 
across ALL scales, from small test cases to large industrial problems with thousands of pieces 
and varying polygon complexities.

**Input:**
- `stock_polygon`: List of (x, y) vertices defining the stock shape
- `pieces`: List of polygons, each a list of (x, y) vertices

**Output:**
- Dict with:
  - `"num_stocks"`: Number of stock polygons used
  - `"placements"`: List of placement dicts, one per piece:
    - `"stock_index"`: Which stock (0-indexed)
    - `"position"`: (x, y) translation to apply to piece
    - `"rotation"`: Rotation angle in radians

**Critical Requirements:**
1. **Scalability**: Your algorithm must adapt based on number and complexity of pieces
2. **Performance**: Must complete within 5 minutes even for 10000+ complex pieces
3. **Quality**: Minimize stock usage while ensuring valid placements

**Constraints:**
- Use only Python standard library + numpy
- Each placed piece must fit entirely within the stock polygon
- Pieces on same stock must not overlap
- Pieces may be rotated to any angle
- Must handle varying numbers and complexities of pieces

Write complete, runnable Python code with the solve_polygon_cutting function.
"""


# List of subpasses to grade the single answer against all difficulty levels
extraGradeAnswerRuns = list(range(1, len(TEST_CASES)))

structure = {
  "type": "object",
  "properties": {
    "reasoning": {
      "type":
      "string",
      "description":
      "Explain your polygon packing algorithm and how it adapts to different problem complexities"
    },
    "python_code": {
      "type":
      "string",
      "description":
      "Complete Python code with solve_polygon_cutting(stock_polygon, pieces) function that handles all scales"
    }
  },
  "required": ["reasoning", "python_code"],
  "additionalProperties": False
}


def point_in_polygon(point: Tuple[float, float], polygon: List[Tuple[float, float]]) -> bool:
  """Ray casting algorithm for point-in-polygon test."""
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


def polygon_in_polygon(inner: List[Tuple[float, float]], outer: List[Tuple[float, float]]) -> bool:
  """Check if inner polygon is entirely within outer polygon."""
  for point in inner:
    if not point_in_polygon(point, outer):
      return False
  return True


def segments_intersect(p1, p2, p3, p4) -> bool:
  """Check if line segment p1-p2 intersects with p3-p4."""

  def ccw(A, B, C):
    return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])

  return ccw(p1, p3, p4) != ccw(p2, p3, p4) and ccw(p1, p2, p3) != ccw(p1, p2, p4)


def polygons_overlap(poly1: List[Tuple[float, float]], poly2: List[Tuple[float, float]]) -> bool:
  """Check if two polygons overlap (share interior area)."""
  # Check if any vertex of one is inside the other
  for p in poly1:
    if point_in_polygon(p, poly2):
      return True
  for p in poly2:
    if point_in_polygon(p, poly1):
      return True

  # Check if any edges intersect
  n1, n2 = len(poly1), len(poly2)
  for i in range(n1):
    p1, p2 = poly1[i], poly1[(i + 1) % n1]
    for j in range(n2):
      p3, p4 = poly2[j], poly2[(j + 1) % n2]
      if segments_intersect(p1, p2, p3, p4):
        return True

  return False


def apply_placement(piece: List[Tuple[float, float]], position: Tuple[float, float],
                    rotation: float) -> List[Tuple[float, float]]:
  """Apply rotation then translation to a piece."""
  rotated = rotate_polygon(piece, rotation)
  return translate_polygon(rotated, position[0], position[1])


def validate_solution(solution: Dict, stock: List[Tuple[float, float]],
                      pieces: List[List[Tuple[float, float]]]) -> Tuple[bool, str]:
  """Validate polygon cutting solution."""
  if not isinstance(solution, dict):
    return False, "Solution must be a dict"

  if "num_stocks" not in solution:
    return False, "Missing 'num_stocks'"

  if "placements" not in solution:
    return False, "Missing 'placements'"

  num_stocks = solution["num_stocks"]
  placements = solution["placements"]

  if not isinstance(num_stocks, int) or num_stocks < 1:
    return False, f"num_stocks must be positive int, got {num_stocks}"

  if len(placements) != len(pieces):
    return False, f"placements count {len(placements)} != pieces count {len(pieces)}"

  # Track placed pieces per stock for overlap checking
  stocks_pieces = [[] for _ in range(num_stocks)]

  for i, placement in enumerate(placements):
    if not isinstance(placement, dict):
      return False, f"Placement {i} must be a dict"

    stock_idx = placement.get("stock_index", 0)
    position = placement.get("position", (0, 0))
    rotation = placement.get("rotation", 0)

    if stock_idx < 0 or stock_idx >= num_stocks:
      return False, f"Placement {i}: invalid stock_index {stock_idx}"

    # Apply transformation
    placed = apply_placement(pieces[i], position, rotation)

    # Check containment in stock
    if not polygon_in_polygon(placed, stock):
      return False, f"Placement {i}: piece not contained in stock"

    # Check overlap with other pieces on same stock
    for j, other_placed in stocks_pieces[stock_idx]:
      if polygons_overlap(placed, other_placed):
        return False, f"Placement {i} overlaps with placement {j}"

    stocks_pieces[stock_idx].append((i, placed))

  return True, ""


def get_baseline_solution(stock: List[Tuple[float, float]],
                          pieces: List[List[Tuple[float, float]]]) -> int:
  """
    Greedy first-fit baseline.
    Returns number of stocks used.
    """
  stock_bounds = polygon_bounds(stock)
  stock_w = stock_bounds[2] - stock_bounds[0]
  stock_h = stock_bounds[3] - stock_bounds[1]

  # Simple grid-based placement
  stocks_pieces = []  # List of lists of placed polygons

  for piece in pieces:
    placed = False

    # Try each existing stock
    for stock_idx, stock_placed in enumerate(stocks_pieces):
      # Try rotations
      for rot in [0, math.pi / 2, math.pi, 3 * math.pi / 2]:
        rotated = rotate_polygon(piece, rot)
        bounds = polygon_bounds(rotated)
        pw = bounds[2] - bounds[0]
        ph = bounds[3] - bounds[1]

        # Grid search for position
        for y in range(0, int(stock_h - ph) + 1, 5):
          for x in range(0, int(stock_w - pw) + 1, 5):
            # Offset to positive coordinates
            tx = x - bounds[0]
            ty = y - bounds[1]
            candidate = translate_polygon(rotated, tx, ty)

            # Check containment
            if not polygon_in_polygon(candidate, stock):
              continue

            # Check overlap
            overlaps = False
            for _, other in stock_placed:
              if polygons_overlap(candidate, other):
                overlaps = True
                break

            if not overlaps:
              stock_placed.append((len(pieces), candidate))
              placed = True
              break
          if placed:
            break
        if placed:
          break
      if placed:
        break

    if not placed:
      # New stock
      rotated = piece
      bounds = polygon_bounds(rotated)
      tx = -bounds[0]
      ty = -bounds[1]
      candidate = translate_polygon(rotated, tx, ty)
      stocks_pieces.append([(len(pieces), candidate)])

  return len(stocks_pieces)


def execute_solver(code: str,
                   stock: List[Tuple[float, float]],
                   pieces: List[List[Tuple[float, float]]],
                   timeout: int = TIMEOUT_SECONDS) -> tuple:
  """Execute the LLM's solver."""
  from solver_utils import execute_solver_with_data

  data_dict = {
    'stock_polygon': stock,
    'pieces': pieces,
  }

  result, error, exec_time = execute_solver_with_data(code, data_dict, 'solve_polygon_cutting',
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
      if isinstance(p, dict):
        stock_index = int(p.get('stock_index', 0))
        position = p.get('position', (0, 0))
        rotation = p.get('rotation', 0)
      elif isinstance(p, (list, tuple)) and len(p) >= 3:
        stock_index = int(p[0])
        position = p[1]
        rotation = p[2]
      else:
        stock_index = 0
        position = (0, 0)
        rotation = 0

      normalized_placements.append({
        'stock_index': stock_index,
        'position': [float(position[0]), float(position[1])],
        'rotation': float(rotation),
      })

    out = {
      'num_stocks': int(result.get('num_stocks')),
      'placements': normalized_placements,
    }
  except Exception as e:
    return None, f"Invalid result format: {e}", exec_time

  return out, None, exec_time


lastSolution = None


def gradeAnswer(result: dict, subPass: int, aiEngineName: str) -> tuple:
  """Grade the polygon cutting solver."""
  if not result:
    return 0.0, "No result provided"

  if "python_code" not in result:
    return 0.0, "No Python code provided"

  case = TEST_CASES[subPass]
  stock = case["stock_polygon"]
  pieces = case["pieces"]
  description = case["description"]
  code = result["python_code"]

  # Execute solver
  solution, error, exec_time = execute_solver(code, stock, pieces)

  global lastSolution
  lastSolution = solution

  if error:
    return 0.0, f"[{description}] {error}"

  # Validate solution
  t1 = time.time()
  is_valid, validation_error = validate_solution(solution, stock, pieces)
  validation_time = time.time() - t1
  if validation_time > 1: print(f"Validation took {validation_time:.2f}s")
  if not is_valid:
    return 0.0, f"[{description}] Invalid: {validation_error}"

  # Compare to baseline
  num_stocks = solution["num_stocks"]
  t2 = time.time()
  baseline_stocks = get_baseline_solution(stock, pieces)
  baseline_time = time.time() - t2
  if baseline_time > 1: print(f"Baseline calculation took {baseline_time:.2f}s")

  ratio = num_stocks / baseline_stocks if baseline_stocks > 0 else float('inf')

  if ratio <= 1.0:
    score = 1.0
    quality = "excellent (≤ baseline)"
  elif ratio <= 1.2:
    score = 0.85
    quality = "good (≤ 1.2x baseline)"
  elif ratio <= 1.5:
    score = 0.7
    quality = "acceptable (≤ 1.5x baseline)"
  else:
    score = 0.5
    quality = f"valid but inefficient ({ratio:.2f}x baseline)"

  explanation = (f"[{description}] Stocks: {num_stocks}, Baseline: {baseline_stocks}, "
                 f"Time: {exec_time:.1f}s - {quality}")

  return score, explanation


def resultToNiceReport(result: dict, subPass: int, aiEngineName: str) -> str:
  """Generate HTML report."""
  if not result:
    return "<p style='color:red'>No result provided</p>"

  case = TEST_CASES[subPass]

  html = f"<h4>Polygon Cutting - {case['description']}</h4>"

  if "reasoning" in result:
    reasoning = result['reasoning'][:500] + ('...'
                                             if len(result.get('reasoning', '')) > 500 else '')
    html += f"<p><strong>Algorithm:</strong> {reasoning}</p>"

  if "python_code" in result:
    code = result["python_code"]
    code_escaped = code.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    html += f"<details><summary>View Code ({len(code)} chars)</summary><pre>{code_escaped}</pre></details>"

  if lastSolution and isinstance(lastSolution, dict):
    stock = case["stock_polygon"]
    pieces = case["pieces"]
    num_stocks = int(lastSolution.get('num_stocks', 0) or 0)
    if len(pieces) <= 200 and num_stocks <= 12:
      html += generate_polygon_cutting_svg(stock, pieces, lastSolution, case["description"])
    else:
      html += f"<p>Solution too large to render SVG ({len(pieces)} pieces, {num_stocks} stocks)</p>"

  return html


def generate_polygon_cutting_svg(stock: List[Tuple[float, float]], pieces: List[List[Tuple[float,
                                                                                           float]]],
                                 solution: Dict, description: str) -> str:
  if not solution or not isinstance(solution, dict):
    return "<p>No solution to visualize</p>"

  placements = solution.get('placements')
  if not isinstance(placements, list) or not placements:
    return "<p>No placements to visualize</p>"

  try:
    num_stocks = int(solution.get('num_stocks', 0))
  except Exception:
    num_stocks = 0

  if num_stocks <= 0:
    return "<p>No stocks to visualize</p>"

  min_x, min_y, max_x, max_y = polygon_bounds(stock)
  w = max(1e-6, max_x - min_x)
  h = max(1e-6, max_y - min_y)
  pad = max(w, h) * 0.05 + 1e-3
  min_x -= pad
  max_x += pad
  min_y -= pad
  max_y += pad
  y0 = min_y + max_y

  def pts(poly: List[Tuple[float, float]]) -> str:
    return ' '.join(f"{x:.3f},{(y0 - y):.3f}" for x, y in poly)

  def color_for(i: int) -> str:
    r = (i * 97) % 200 + 30
    g = (i * 57) % 200 + 30
    b = (i * 17) % 200 + 30
    return f"#{r:02x}{g:02x}{b:02x}"

  per_stock = [[] for _ in range(num_stocks)]
  for i, p in enumerate(placements):
    if i >= len(pieces):
      break
    if not isinstance(p, dict):
      continue
    stock_idx = int(p.get('stock_index', 0))
    if stock_idx < 0 or stock_idx >= num_stocks:
      continue
    position = p.get('position', [0, 0])
    rotation = float(p.get('rotation', 0.0))
    placed_poly = apply_placement(pieces[i], (float(position[0]), float(position[1])), rotation)
    per_stock[stock_idx].append((i, placed_poly))

  show_labels = len(pieces) <= 30
  html = f"<h5>Layout</h5><p>{description}</p>"

  view_w = max_x - min_x
  view_h = max_y - min_y

  for stock_idx, placed in enumerate(per_stock):
    html += f"<div style='margin:10px 0'>"
    html += f"<div><strong>Stock {stock_idx + 1}/{num_stocks}</strong></div>"
    html += (
      f"<svg width='420' style='max-width:100%; height:auto; border:1px solid #ddd; background:#fafafa' "
      f"viewBox='{min_x:.3f} {min_y:.3f} {view_w:.3f} {view_h:.3f}' xmlns='http://www.w3.org/2000/svg'>"
    )

    html += f"<polygon points='{pts(stock)}' fill='#eeeeee' stroke='black' stroke-width='{max(view_w, view_h) * 0.002:.3f}'/>"

    for i, poly in placed:
      fill = color_for(i)
      html += (
        f"<polygon points='{pts(poly)}' fill='{fill}' fill-opacity='0.65' "
        f"stroke='{fill}' stroke-opacity='0.9' stroke-width='{max(view_w, view_h) * 0.0015:.3f}'/>")
      if show_labels:
        cx = sum(x for x, _ in poly) / max(1, len(poly))
        cy = sum(y for _, y in poly) / max(1, len(poly))
        html += (
          f"<text x='{cx:.3f}' y='{(y0 - cy):.3f}' text-anchor='middle' dominant-baseline='middle' "
          f"font-size='{max(view_w, view_h) * 0.03:.3f}' fill='black'>{i}</text>")

    html += "</svg></div>"

  return html


highLevelSummary = """
2D Polygon Cutting Stock is a generalization of rectangle packing.

**Problem:** Cut arbitrary polygons from stock polygons, minimizing waste.

**Key challenges:**
- Polygon containment testing
- Polygon-polygon intersection detection
- Rotation optimization
- Irregular shapes don't pack as efficiently

**Algorithms:**
- **Greedy First Fit**: Place each piece in first stock where it fits
- **Bottom-Left**: Try positions from bottom-left corner
- **No-Fit Polygon (NFP)**: Compute valid placement regions
- **Genetic algorithms**: For complex instances

**Required geometry operations:**
- Point-in-polygon (ray casting)
- Line segment intersection
- Polygon area calculation
- Bounding box computation

The baseline uses greedy first-fit with grid search for positions.
"""
