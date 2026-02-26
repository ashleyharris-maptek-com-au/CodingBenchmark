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
import time
from typing import List, Tuple, Dict

from native_compiler import RustCompiler, compile_and_run, describe_this_pc
from solver_utils import parse_freeform_response

title = "2D Polygon Cutting Stock (Rust)"

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

  return f"""You are solving a 2D Polygon Cutting Stock Problem in Rust.

**The Challenge:**
Your program will be tested with problems ranging from 2 simple pieces to 5000+ complex polygons.
The same program must work efficiently across ALL scales.

**Input format (stdin):**
Line 1: SV NP (number of stock polygon vertices, number of pieces)
Next SV lines: x y (stock polygon vertices)
For each piece:
  Line: PV (number of vertices in this piece)
  Next PV lines: x y (piece polygon vertices)

**Output format (stdout):**
Line 1: num_stocks
Next NP lines: stock_index x y rotation (one per piece)
  stock_index: 0-indexed stock polygon
  x y: translation to apply
  rotation: angle in radians

**Critical Requirements:**
1. **Scalability**: Your algorithm must adapt based on number and complexity of pieces
2. **Performance**: Must complete within 5 minutes even for 5000+ complex pieces
3. **Quality**: Minimize stock usage while ensuring valid placements

**Constraints:**
- Each placed piece must fit entirely within the stock polygon
- Pieces on same stock must not overlap
- Pieces may be rotated to any angle

**Environment:**
{describe_this_pc()}

**Rust Compiler:**
{RustCompiler("test_engine").describe()}

Write complete, compilable Rust code with a main() function.
"""


# List of subpasses to grade the single answer against all difficulty levels
extraGradeAnswerRuns = list(range(1, len(TEST_CASES)))

structure = None


def _extract_freeform(result):
  if isinstance(result, dict):
    discussion = result.get("reasoning") or result.get("discussion") or ""
    code = result.get("rust_code") or result.get("code") or ""
    return discussion, code, ""
  if isinstance(result, str) and result.strip() == "__content_violation__":
    return "", "", "Content violation"
  parsed = parse_freeform_response(result or "")
  return parsed.get("discussion", ""), parsed.get("code", ""), ""


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


def check_top_waste_repack_poly(solution: Dict, stock: List[Tuple[float, float]],
                                pieces: List[List[Tuple[float, float]]]) -> Tuple[int, int]:
  """Pick the 5 stocks with highest waste, strip their pieces, and repack
  using the greedy baseline.  Returns (original_count, optimal_count)."""
  N_FOCUS = 5
  num_stocks = solution["num_stocks"]
  placements = solution["placements"]
  stock_area = polygon_area(stock)

  if num_stocks < N_FOCUS:
    return num_stocks, num_stocks

  # Group piece indices by stock
  stocks = {}
  for i, p in enumerate(placements):
    si = p["stock_index"]
    if si not in stocks:
      stocks[si] = []
    stocks[si].append(i)

  # Waste per stock
  waste_per = []
  for si, indices in stocks.items():
    used = sum(polygon_area(pieces[i]) for i in indices)
    waste_per.append((stock_area - used, si))
  waste_per.sort(reverse=True)

  focus = waste_per[:N_FOCUS]
  focus_pieces = []
  for _, si in focus:
    for idx in stocks[si]:
      focus_pieces.append(pieces[idx])

  if not focus_pieces:
    return N_FOCUS, N_FOCUS

  # Area-based lower bound — if pieces can't even fit in fewer by area, skip
  total_piece_area = sum(polygon_area(p) for p in focus_pieces)
  area_lb = max(1, math.ceil(total_piece_area / stock_area))
  if area_lb >= N_FOCUS:
    return N_FOCUS, N_FOCUS

  # For small piece counts, run greedy baseline on the subset
  if len(focus_pieces) <= 80:
    optimal = get_baseline_solution(stock, focus_pieces)
    return N_FOCUS, min(optimal, N_FOCUS)

  # For larger counts, use area bound with 65% packing efficiency assumption
  est = max(area_lb, math.ceil(total_piece_area / (stock_area * 0.65)))
  return N_FOCUS, min(est, N_FOCUS)


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


def format_input(stock: List[Tuple[float, float]], pieces: List[List[Tuple[float, float]]]) -> str:
  lines = [f"{len(stock)} {len(pieces)}"]
  for x, y in stock:
    lines.append(f"{x:.6f} {y:.6f}")
  for piece in pieces:
    lines.append(str(len(piece)))
    for x, y in piece:
      lines.append(f"{x:.6f} {y:.6f}")
  return "\n".join(lines)


def parse_placements_output(output: str, num_pieces: int) -> tuple:
  text = output.strip()
  if not text:
    return None, "Empty output"

  lines = [l for l in text.splitlines() if l.strip()]
  if not lines:
    return None, "No output lines"

  try:
    num_stocks = int(lines[0])
  except ValueError:
    return None, "First line must be num_stocks integer"

  if len(lines) < 1 + num_pieces:
    return None, f"Expected {num_pieces} placement lines, got {len(lines) - 1}"

  placements = []
  for i in range(1, 1 + num_pieces):
    parts = lines[i].split()
    if len(parts) < 4:
      return None, f"Placement line {i} needs 4 values (stock_index x y rotation)"
    try:
      stock_index = int(parts[0])
      x = float(parts[1])
      y = float(parts[2])
      rotation = float(parts[3])
    except ValueError:
      return None, f"Invalid values in placement line {i}"
    placements.append({
      'stock_index': stock_index,
      'position': [x, y],
      'rotation': rotation,
    })

  return {'num_stocks': num_stocks, 'placements': placements}, None


def execute_solver(code: str,
                   stock: List[Tuple[float, float]],
                   pieces: List[List[Tuple[float, float]]],
                   ai_engine_name: str,
                   timeout: int = TIMEOUT_SECONDS) -> tuple:
  """Execute the LLM's solver."""
  input_data = format_input(stock, pieces)
  run = compile_and_run(code, "rust", ai_engine_name, input_data=input_data, timeout=timeout)

  if not run:
    return None, run.error_message(), run.exec_time

  solution, parse_error = parse_placements_output(run.stdout, len(pieces))
  if parse_error:
    return None, parse_error, run.exec_time

  return solution, None, run.exec_time


lastSolution = None


def gradeAnswer(result: dict, subPass: int, aiEngineName: str) -> tuple:
  """Grade the polygon cutting solver."""
  if not result:
    return 0.0, "No result provided"

  discussion, code, parse_error = _extract_freeform(result)
  if parse_error:
    return 0.0, parse_error
  if not code:
    return 0.0, "No Rust code provided"

  case = TEST_CASES[subPass]
  stock = case["stock_polygon"]
  pieces = case["pieces"]
  description = case["description"]
  code = code

  # Execute solver
  solution, error, exec_time = execute_solver(code, stock, pieces, aiEngineName)

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

  # ── Repack penalty ──
  # If total waste < 1 stock area, packing is tight — no penalty.
  # Otherwise, take the 5 stocks with the most waste, strip their pieces,
  # and repack.  If fewer stocks needed, penalise.
  stock_area = polygon_area(stock)
  total_piece_area = sum(polygon_area(p) for p in pieces)
  waste_area = num_stocks * stock_area - total_piece_area
  penalty_note = ""
  if waste_area < stock_area:
    penalty_note = " (waste < 1 stock area, no repack check)"
  elif num_stocks >= 5 and num_stocks <= 200 and len(pieces) <= 5000:
    orig, optimal = check_top_waste_repack_poly(solution, stock, pieces)
    saved = orig - optimal
    if saved >= 2:
      score = 0.0
      penalty_note = f" REPACK PENALTY: top-5 waste stocks repacked {orig}→{optimal} (saved {saved}) → 0"
    elif saved == 1:
      score = min(score, 0.5)
      penalty_note = f" REPACK PENALTY: top-5 waste stocks repacked {orig}→{optimal} (saved 1) → capped 0.5"
    else:
      penalty_note = f" (repack check: top-5 waste stocks already optimal at {optimal})"

  explanation = (f"[{description}] Stocks: {num_stocks}, Baseline: {baseline_stocks}, "
                 f"Time: {exec_time:.1f}s - {quality}.{penalty_note}")

  return score, explanation


def resultToNiceReport(result: dict, subPass: int, aiEngineName: str) -> str:
  """Generate HTML report."""
  if not result:
    return "<p style='color:red'>No result provided</p>"

  case = TEST_CASES[subPass]

  html = f"<h4>Polygon Cutting - {case['description']}</h4>"

  discussion, code, _ = _extract_freeform(result)
  if discussion:
    reasoning = discussion[:500] + ('...' if len(discussion) > 500 else '')
    html += f"<p><strong>Algorithm:</strong> {reasoning}</p>"

  if code:
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
<p>Cut a set of irregular polygon shapes out of stock sheets with as little waste
as possible. Unlike rectangle packing, the pieces can be any shape &mdash; stars,
L-shapes, curves &mdash; so fitting them together is much harder.</p>
<p>The AI must handle polygon geometry (containment, intersection, rotation) and
plan efficient layouts. Subpasses increase the number and complexity of the
pieces. The baseline uses a simple greedy placement strategy.</p>
"""
