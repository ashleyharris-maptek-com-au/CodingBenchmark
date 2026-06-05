"""
Shared JavaScript/TypeScript benchmark problems.

The numbered tests at the end of the benchmark import this module and bind one
problem to one runtime. Keeping the problem logic shared makes JavaScript and
TypeScript directly comparable while still producing separate benchmark tests.
"""

import html
import json
import math
import os
import random
import time
from collections import deque
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from solver_utils import normalize_code_result, GradeCache
from javascript_typescript_runner import (
  describe_javascript_runtime,
  describe_typescript_runtime,
  execute_javascript,
  execute_typescript,
)

TIMEOUT_SECONDS = 45
RANDOM_SEED = 560000
_LAST_VIZ: Dict[Tuple[str, str, int, str], str] = {}
_GRADE_CACHE = GradeCache("script_benchmark_problems")

LANGUAGES = {
  "javascript": {
    "name": "javascript",
    "label": "JavaScript",
    "code_key": "javascript_code",
    "tags": ["javascript", "nodejs"],
    "execute": execute_javascript,
    "describe": describe_javascript_runtime,
    "read_hint": "const fs = require('fs'); const input = fs.readFileSync(0, 'utf8');",
  },
  "typescript": {
    "name":
    "typescript",
    "label":
    "TypeScript",
    "code_key":
    "typescript_code",
    "tags": ["typescript", "nodejs"],
    "execute":
    execute_typescript,
    "describe":
    describe_typescript_runtime,
    "read_hint":
    "declare const require: any; const fs = require('fs'); const input: string = fs.readFileSync(0, 'utf8');",
  },
}


def _merge_tags(*tag_groups: Sequence[str]) -> List[str]:
  tags = []
  seen = set()
  for group in tag_groups:
    for tag in group:
      key = tag.lower()
      if key not in seen:
        tags.append(tag)
        seen.add(key)
  return tags


def configure_problem(problem_key: str,
                      language_key: str,
                      extra_tags: Optional[Sequence[str]] = None) -> Dict[str, Any]:
  problem = PROBLEMS[problem_key]
  language = LANGUAGES[language_key]

  def prepareSubpassPrompt(subPass: int) -> str:
    if subPass != 0:
      raise StopIteration
    return _build_prompt(problem, language)

  def gradeAnswer(result: dict, subPass: int, aiEngineName: str) -> tuple:
    return _grade_problem(problem, language, result, subPass, aiEngineName)

  def resultToNiceReport(result: dict, subPass: int, aiEngineName: str) -> str:
    return _nice_report(problem, language, result, subPass, aiEngineName)

  return {
    "title": f"{problem['title']} ({language['label']})",
    "tags": _merge_tags(language["tags"], problem["tags"], extra_tags or []),
    "TIMEOUT_SECONDS": TIMEOUT_SECONDS,
    "structure": None,
    "prepareSubpassPrompt": prepareSubpassPrompt,
    "gradeAnswer": gradeAnswer,
    "resultToNiceReport": resultToNiceReport,
    "extraGradeAnswerRuns": list(range(len(problem["cases"]))),
    "highLevelSummary": problem["summary"],
  }


def _build_prompt(problem: Dict[str, Any], language: Dict[str, Any]) -> str:
  return f"""Write a complete Node.js {language['label']} program for this benchmark task.

{problem['prompt']}

**Input format:**
The program receives one JSON document on stdin. Use this pattern to read it:
```
{language['read_hint']}
const data = JSON.parse(input);
```

**Output format:**
Print one JSON document to stdout. Do not print logs or explanatory text.

**Runtime environment:**
{language['describe']()}

The grader will run the same program on increasingly large scenarios. Use algorithms
that scale; avoid browser APIs, network access, and external npm dependencies.
Return the complete runnable {language['label']} source code.
"""


def _grade_cache_key_parts(problem: Dict[str, Any], language: Dict[str, Any], subPass: int,
                           aiEngineName: str, code: str, case: Dict[str, Any]) -> tuple:
  return (
    "script-benchmark-grade-v1",
    f"problem={problem['key']}",
    f"language={language['name']}",
    f"model={aiEngineName}",
    f"subpass={subPass}",
    json.dumps(case, sort_keys=True, separators=(",", ":")),
    code,
  )


def _restore_cached_viz(problem: Dict[str, Any], language: Dict[str, Any], subPass: int,
                        aiEngineName: str, viz: str) -> None:
  viz_key = (problem["key"], language["name"], subPass, aiEngineName)
  if viz:
    _LAST_VIZ[viz_key] = viz
  else:
    _LAST_VIZ.pop(viz_key, None)


def _grade_problem(problem: Dict[str, Any], language: Dict[str, Any], result: dict, subPass: int,
                   aiEngineName: str) -> tuple:
  result = normalize_code_result(result, language["code_key"])
  if result and language["code_key"] not in result:
    for alt_key in ("js_code", "javascript", "ts_code", "typescript"):
      if isinstance(result.get(alt_key), str) and result.get(alt_key).strip():
        result[language["code_key"]] = result[alt_key]
        break
  if not result or language["code_key"] not in result:
    return 0.0, f"No {language['label']} code provided"

  skip_reason_fn = problem.get("skip_local_grade_reason")
  if callable(skip_reason_fn):
    skip_reason = skip_reason_fn(subPass)
    if skip_reason:
      return 1.0, skip_reason

  case = problem["make_case"](subPass)
  input_data = json.dumps(case, separators=(",", ":"))
  code = result[language["code_key"]]
  cache_parts = _grade_cache_key_parts(problem, language, subPass, aiEngineName, code, case)

  def compute_grade_record() -> Dict[str, Any]:
    run = language["execute"](code, aiEngineName, input_data=input_data, timeout=TIMEOUT_SECONDS)
    if not run:
      return {
        "score": 0.0,
        "explanation": f"{run.error_stage}: {run.error_message()}",
        "viz": "",
      }

    try:
      output = _parse_json_output(run.stdout)
    except Exception as e:
      sample = run.stdout.strip().replace("\n", " ")[:160]
      return {
        "score": 0.0,
        "explanation": f"Could not parse JSON output: {e}. Output: {sample}",
        "viz": "",
      }

    start = time.time()
    score, explanation, viz = problem["grade_output"](case, output)
    grade_time = time.time() - start
    timing = f"run {run.exec_time:.2f}s"
    if grade_time > 1:
      timing += f", grade {grade_time:.2f}s"
    return {
      "score": score,
      "explanation": f"{explanation} ({timing})",
      "viz": viz,
    }

  record = _GRADE_CACHE.get_or_compute_json("grade_record", compute_grade_record, *cache_parts)
  _restore_cached_viz(problem, language, subPass, aiEngineName, str(record.get("viz", "") or ""))
  return float(record.get("score", 0.0)), record.get("explanation", "No explanation")


def _nice_report(problem: Dict[str, Any], language: Dict[str, Any], result: dict, subPass: int,
                 aiEngineName: str) -> str:
  result = normalize_code_result(result, language["code_key"])
  if result and language["code_key"] not in result:
    for alt_key in ("js_code", "javascript", "ts_code", "typescript"):
      if isinstance(result.get(alt_key), str) and result.get(alt_key).strip():
        result[language["code_key"]] = result[alt_key]
        break
  case_desc = f"subpass {subPass}"
  try:
    case_info = problem["cases"][subPass]
    if isinstance(case_info, (list, tuple)) and case_info:
      case_desc = str(case_info[-1])
  except Exception:
    pass
  out = [f"<h4>{html.escape(problem['title'])} - {html.escape(case_desc)}</h4>"]
  if result and "reasoning" in result and subPass == 0:
    reasoning = html.escape(str(result["reasoning"])[:500])
    out.append(f"<p><strong>Approach:</strong> {reasoning}</p>")
  if result and language["code_key"] in result and subPass == 0:
    code = html.escape(result[language["code_key"]])
    out.append(
      f"<details><summary>View {language['label']} code ({len(result[language['code_key']])} chars)</summary><pre>{code}</pre></details>"
    )
  viz = _LAST_VIZ.get((problem["key"], language["name"], subPass, aiEngineName))
  if viz:
    out.append(viz)
  return "\n".join(out)


def _parse_json_output(stdout: str) -> Any:
  text = stdout.strip()
  if not text:
    raise ValueError("empty stdout")
  try:
    return json.loads(text)
  except Exception:
    pass
  for left, right in (("{", "}"), ("[", "]")):
    s = text.find(left)
    e = text.rfind(right)
    if s >= 0 and e > s:
      return json.loads(text[s:e + 1])
  raise ValueError("no JSON object or array found")


def _clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
  return max(lo, min(hi, x))


def _dist(a: Sequence[float], b: Sequence[float]) -> float:
  dx = float(a[0]) - float(b[0])
  dy = float(a[1]) - float(b[1])
  return math.hypot(dx, dy)


def _point_from_any(value: Any) -> Optional[Tuple[int, int]]:
  if isinstance(value, dict):
    if "x" in value and "y" in value:
      try:
        return int(value["x"]), int(value["y"])
      except Exception:
        return None
  if isinstance(value, (list, tuple)) and len(value) >= 2:
    try:
      return int(value[0]), int(value[1])
    except Exception:
      return None
  return None


def _normalise_routes(output: Any) -> Optional[List[List[int]]]:
  if isinstance(output, dict):
    raw = output.get("routes", output.get("route", output.get("assignments")))
  else:
    raw = output
  if raw is None:
    return None
  if isinstance(raw, dict):
    ordered = []
    for key in sorted(raw.keys(), key=lambda x: str(x)):
      ordered.append(raw[key])
    raw = ordered
  if not isinstance(raw, list):
    return None
  if raw and all(not isinstance(x, (list, tuple)) for x in raw):
    raw = [raw]
  routes = []
  for route in raw:
    if not isinstance(route, (list, tuple)):
      return None
    ids = []
    for v in route:
      try:
        ids.append(int(v["id"] if isinstance(v, dict) and "id" in v else v))
      except Exception:
        return None
    routes.append(ids)
  return routes


# ---------------------------------------------------------------------------
# Delivery route planner
# ---------------------------------------------------------------------------

DELIVERY_CASES = [
  (30, 2, "30 packages, 2 vans"),
  (80, 3, "80 packages, 3 vans"),
  (180, 5, "180 packages, 5 vans"),
  (400, 8, "400 packages, 8 vans"),
  (800, 10, "800 packages, 10 vans"),
  (1400, 14, "1.4K packages, 14 vans"),
  (2200, 18, "2.2K packages, 18 vans"),
  (3600, 24, "3.6K packages, 24 vans"),
  (10000, 50, "10K packages, 50 vans"),
  (50000, 100, "50K packages, 100 vans"),
  (100000, 200, "100K packages, 200 vans"),
  (200000, 400, "200K packages, 400 vans"),
  (500000, 1000, "500K packages, 1000 vans"),
  (1000000, 2000, "1M packages, 2000 vans"),
]
DELIVERY_EXACT_PACKAGE_LIMIT = int(os.environ.get("DELIVERY_EXACT_PACKAGE_LIMIT", "3600"))
DELIVERY_ANGLE_SORT_PACKAGE_LIMIT = int(os.environ.get("DELIVERY_ANGLE_SORT_PACKAGE_LIMIT", "50000"))
DELIVERY_SKIP_LOCAL_GRADE_PACKAGE_LIMIT = int(
  os.environ.get("DELIVERY_SKIP_LOCAL_GRADE_PACKAGE_LIMIT", "200000"))


def _delivery_skip_reason(subpass: int) -> Optional[str]:
  package_count, truck_count, desc = DELIVERY_CASES[subpass]
  if package_count <= DELIVERY_SKIP_LOCAL_GRADE_PACKAGE_LIMIT:
    return None
  return (
    f"[{desc}] local grade skipped for {package_count} packages and {truck_count} vans; "
    "smaller delivery subpasses validate correctness and route quality.")


def _make_delivery_case(subpass: int) -> Dict[str, Any]:
  n, trucks, desc = DELIVERY_CASES[subpass]
  rng = random.Random(RANDOM_SEED + 1000 + subpass)
  depot = [500.0, 500.0]
  cluster_count = max(3, min(18, int(math.sqrt(n) // 2)))
  centers = [(rng.uniform(80, 920), rng.uniform(80, 920)) for _ in range(cluster_count)]
  packages = []
  for i in range(n):
    cx, cy = centers[i % cluster_count]
    x = max(0.0, min(1000.0, rng.gauss(cx, 55 + 8 * subpass)))
    y = max(0.0, min(1000.0, rng.gauss(cy, 55 + 8 * subpass)))
    packages.append({"id": i, "x": round(x, 3), "y": round(y, 3)})
  rng.shuffle(packages)
  return {
    "desc": desc,
    "depot": depot,
    "truckCount": trucks,
    "packages": packages,
    "objective": "minimize total depot-to-depot route distance",
  }


def _delivery_distance(case: Dict[str, Any], routes: List[List[int]]) -> float:
  packages = {int(p["id"]): (float(p["x"]), float(p["y"])) for p in case["packages"]}
  depot = case["depot"]
  total = 0.0
  for route in routes:
    cur = depot
    for pid in route:
      nxt = packages[pid]
      total += _dist(cur, nxt)
      cur = nxt
    total += _dist(cur, depot)
  return total


def _delivery_reference(case: Dict[str, Any]) -> List[List[int]]:
  packages = {int(p["id"]): (float(p["x"]), float(p["y"])) for p in case["packages"]}
  depot = case["depot"]
  trucks = int(case["truckCount"])
  ordered = sorted(packages.keys(),
                   key=lambda i: math.atan2(packages[i][1] - depot[1], packages[i][0] - depot[0]))
  routes: List[List[int]] = []
  for t in range(trucks):
    chunk = ordered[(len(ordered) * t) // trucks:(len(ordered) * (t + 1)) // trucks]
    remaining = set(chunk)
    route = []
    cur = depot
    while remaining:
      best = min(remaining, key=lambda pid: _dist(cur, packages[pid]))
      route.append(best)
      remaining.remove(best)
      cur = packages[best]
    routes.append(route)
  return routes


def _delivery_reference_fast(case: Dict[str, Any]) -> List[List[int]]:
  packages = case["packages"]
  depot = case["depot"]
  trucks = int(case["truckCount"])
  if len(packages) <= DELIVERY_ANGLE_SORT_PACKAGE_LIMIT:
    ordered = [
      int(p["id"]) for p in sorted(
        packages,
        key=lambda p: math.atan2(float(p["y"]) - depot[1], float(p["x"]) - depot[0]))
    ]
  else:
    bucket_count = max(128, trucks * 16)
    buckets: List[List[int]] = [[] for _ in range(bucket_count)]
    for p in packages:
      angle = math.atan2(float(p["y"]) - depot[1], float(p["x"]) - depot[0])
      bucket = int(((angle + math.pi) / (2 * math.pi)) * bucket_count)
      buckets[min(bucket_count - 1, max(0, bucket))].append(int(p["id"]))
    ordered = [pid for bucket in buckets for pid in bucket]

  routes: List[List[int]] = []
  for t in range(trucks):
    routes.append(ordered[(len(ordered) * t) // trucks:(len(ordered) * (t + 1)) // trucks])
  return routes


def _grade_delivery(case: Dict[str, Any], output: Any) -> tuple:
  routes = _normalise_routes(output)
  if routes is None:
    return 0.0, "Expected JSON with routes: [[packageId,...],...]", ""
  all_ids = [pid for route in routes for pid in route]
  expected = {int(p["id"]) for p in case["packages"]}
  if set(all_ids) != expected or len(all_ids) != len(expected):
    return 0.0, f"Routes must visit every package exactly once; got {len(all_ids)} visits", ""
  if len(routes) > int(case["truckCount"]):
    return 0.0, f"Used {len(routes)} routes but only {case['truckCount']} trucks are available", ""

  large_case = len(case["packages"]) > DELIVERY_EXACT_PACKAGE_LIMIT
  solver_distance = _delivery_distance(case, routes)
  ref_routes = _delivery_reference_fast(case) if large_case else _delivery_reference(case)
  ref_distance = _delivery_distance(case, ref_routes)
  score = _clamp(ref_distance / max(solver_distance, 1e-9))
  mode = "angular fast" if large_case else "exact nearest-neighbor"
  explanation = (
    f"[{case['desc']}] distance {solver_distance:.1f}, reference {ref_distance:.1f}, "
    f"score {score:.3f} ({mode} grading)"
  )
  return score, explanation, _delivery_svg(case, routes, solver_distance, ref_distance)


def _delivery_svg(case: Dict[str, Any], routes: List[List[int]], solver_dist: float,
                  ref_dist: float) -> str:
  packages = {int(p["id"]): (float(p["x"]), float(p["y"])) for p in case["packages"]}
  if len(packages) > 500:
    return f"<p>Route visualization skipped for {len(packages)} packages.</p>"
  colors = ["#2563eb", "#dc2626", "#16a34a", "#9333ea", "#ea580c", "#0891b2", "#be123c", "#4f46e5"]
  lines = [
    "<div style='margin:12px 0'>",
    f"<p><strong>Total distance:</strong> {solver_dist:.1f} (reference {ref_dist:.1f})</p>",
    "<svg width='100%' viewBox='0 0 1040 1040' style='background:#f8fafc;border:1px solid #cbd5e1'>",
    "<rect x='20' y='20' width='1000' height='1000' fill='white' stroke='#94a3b8' />",
  ]
  depot = case["depot"]
  for ri, route in enumerate(routes):
    pts = [depot] + [packages[pid] for pid in route] + [depot]
    color = colors[ri % len(colors)]
    path = " ".join(f"{20 + p[0]:.1f},{1020 - p[1]:.1f}" for p in pts)
    lines.append(
      f"<polyline points='{path}' fill='none' stroke='{color}' stroke-width='2' opacity='0.65' />")
  for pid, (x, y) in packages.items():
    lines.append(f"<circle cx='{20 + x:.1f}' cy='{1020 - y:.1f}' r='3' fill='#0f172a' />")
  lines.append(
    f"<rect x='{16 + depot[0]:.1f}' y='{1016 - depot[1]:.1f}' width='8' height='8' fill='#facc15' stroke='#854d0e' />"
  )
  lines.append("</svg></div>")
  return "\n".join(lines)


# ---------------------------------------------------------------------------
# Wildfire firebreak planner
# ---------------------------------------------------------------------------

WILDFIRE_CASES = [
  (32, 22, 10, 3, 1, "32x22 forest, 3 towns"),
  (52, 36, 18, 5, 2, "52x36 forest, 5 towns"),
  (80, 54, 28, 8, 2, "80x54 forest, 8 towns"),
  (110, 74, 42, 12, 3, "110x74 forest, 12 towns"),
  (150, 96, 62, 18, 4, "150x96 forest, 18 towns"),
  (200, 128, 90, 26, 5, "200x128 forest, 26 towns"),
  (260, 160, 125, 36, 6, "260x160 forest, 36 towns"),
]


def _make_wildfire_case(subpass: int) -> Dict[str, Any]:
  w, h, budget, town_count, ignition_count, desc = WILDFIRE_CASES[subpass]
  rng = random.Random(RANDOM_SEED + 2000 + subpass)
  grid = [["." for _ in range(w)] for _ in range(h)]
  for y in range(h):
    for x in range(w):
      if rng.random() < 0.045 + 0.006 * subpass:
        grid[y][x] = "~"

  def open_cell() -> Tuple[int, int]:
    while True:
      x = rng.randrange(2, w - 2)
      y = rng.randrange(2, h - 2)
      if grid[y][x] == ".":
        return x, y

  ignitions = []
  for _ in range(ignition_count):
    x, y = open_cell()
    ignitions.append([x, y])
  towns = []
  for i in range(town_count):
    x, y = open_cell()
    value = rng.randint(3, 12)
    towns.append({"id": i, "x": x, "y": y, "value": value})
  return {
    "desc": desc,
    "width": w,
    "height": h,
    "cutBudget": budget,
    "grid": ["".join(row) for row in grid],
    "ignitions": ignitions,
    "towns": towns,
    "objective": "choose forest cells to cut so the fire burns the least town value",
  }


def _simulate_fire(case: Dict[str, Any],
                   cuts: Optional[Iterable[Tuple[int, int]]] = None) -> Tuple[int, int, set]:
  w, h = int(case["width"]), int(case["height"])
  blocked = set()
  for y, row in enumerate(case["grid"]):
    for x, c in enumerate(row):
      if c == "~":
        blocked.add((x, y))
  if cuts:
    blocked.update(cuts)
  q = deque()
  burned = set()
  for p in case["ignitions"]:
    pt = (int(p[0]), int(p[1]))
    if pt not in blocked:
      q.append(pt)
      burned.add(pt)
  while q:
    x, y = q.popleft()
    for nx, ny in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
      if 0 <= nx < w and 0 <= ny < h and (nx, ny) not in blocked and (nx, ny) not in burned:
        burned.add((nx, ny))
        q.append((nx, ny))
  town_value = 0
  for t in case["towns"]:
    if (int(t["x"]), int(t["y"])) in burned:
      town_value += int(t["value"])
  return town_value, len(burned), burned


def _grade_wildfire(case: Dict[str, Any], output: Any) -> tuple:
  raw = output.get("firebreaks", output.get("cuts", output)) if isinstance(output, dict) else output
  if not isinstance(raw, list):
    return 0.0, "Expected firebreaks as [[x,y],...]", ""
  budget = int(case["cutBudget"])
  if len(raw) > budget:
    return 0.0, f"Used {len(raw)} firebreak cells but budget is {budget}", ""

  w, h = int(case["width"]), int(case["height"])
  water = {(x, y) for y, row in enumerate(case["grid"]) for x, c in enumerate(row) if c == "~"}
  towns = {(int(t["x"]), int(t["y"])) for t in case["towns"]}
  ignitions = {(int(p[0]), int(p[1])) for p in case["ignitions"]}
  cuts = set()
  invalid = 0
  for item in raw:
    pt = _point_from_any(item)
    if pt is None:
      invalid += 1
      continue
    x, y = pt
    if not (0 <= x < w and 0 <= y < h) or pt in water or pt in towns or pt in ignitions:
      invalid += 1
      continue
    cuts.add(pt)

  baseline_value, baseline_cells, _ = _simulate_fire(case)
  solver_value, solver_cells, burned = _simulate_fire(case, cuts)
  if baseline_value <= 0:
    score = 1.0 if invalid == 0 else 0.75
  else:
    saved = max(0, baseline_value - solver_value)
    score = _clamp(saved / baseline_value)
    if invalid:
      score *= max(0.0, 1.0 - invalid / max(1, budget))
  explanation = (
    f"[{case['desc']}] town value burned {solver_value}/{baseline_value}, burned cells {solver_cells}/{baseline_cells}, valid cuts {len(cuts)}"
  )
  return score, explanation, _wildfire_svg(case, cuts, burned, solver_value, baseline_value)


def _wildfire_svg(case: Dict[str, Any], cuts: set, burned: set, solver_value: int,
                  baseline_value: int) -> str:
  w, h = int(case["width"]), int(case["height"])
  if w * h > 16000:
    return f"<p>Fire map visualization skipped for {w}x{h} grid.</p>"
  cell = max(3, min(12, 780 / max(w, h)))
  width = int(w * cell)
  height = int(h * cell)
  towns = {(int(t["x"]), int(t["y"])): int(t["value"]) for t in case["towns"]}
  ignitions = {(int(p[0]), int(p[1])) for p in case["ignitions"]}
  parts = [
    f"<div><p><strong>Burned town value:</strong> {solver_value} (baseline {baseline_value})</p>",
    f"<svg width='100%' viewBox='0 0 {width} {height}' style='background:#ecfdf5;border:1px solid #94a3b8'>",
  ]
  for y, row in enumerate(case["grid"]):
    for x, ch in enumerate(row):
      color = "#bbf7d0"
      if ch == "~":
        color = "#60a5fa"
      if (x, y) in burned:
        color = "#fb923c"
      if (x, y) in cuts:
        color = "#1f2937"
      if (x, y) in towns:
        color = "#ef4444" if (x, y) in burned else "#22c55e"
      if (x, y) in ignitions:
        color = "#7f1d1d"
      parts.append(
        f"<rect x='{x * cell:.1f}' y='{y * cell:.1f}' width='{cell:.1f}' height='{cell:.1f}' fill='{color}' />"
      )
  parts.append("</svg></div>")
  return "\n".join(parts)


# ---------------------------------------------------------------------------
# Warehouse robot assignment
# ---------------------------------------------------------------------------

WAREHOUSE_CASES = [
  (28, 20, 2, 28, "28 jobs, 2 robots"),
  (44, 30, 3, 70, "70 jobs, 3 robots"),
  (64, 42, 5, 150, "150 jobs, 5 robots"),
  (90, 60, 8, 320, "320 jobs, 8 robots"),
  (120, 84, 10, 620, "620 jobs, 10 robots"),
  (160, 110, 14, 1000, "1K jobs, 14 robots"),
]
WAREHOUSE_EXACT_JOB_LIMIT = int(os.environ.get("WAREHOUSE_EXACT_JOB_LIMIT", "150"))


def _make_warehouse_case(subpass: int) -> Dict[str, Any]:
  w, h, robot_count, job_count, desc = WAREHOUSE_CASES[subpass]
  rng = random.Random(RANDOM_SEED + 3000 + subpass)
  grid = [["." for _ in range(w)] for _ in range(h)]
  for _ in range(max(4, w * h // 550)):
    x0 = rng.randrange(2, max(3, w - 10))
    y0 = rng.randrange(2, max(3, h - 6))
    rw = rng.randrange(2, min(8, max(3, w // 9)))
    rh = rng.randrange(2, min(6, max(3, h // 8)))
    for y in range(y0, min(h - 2, y0 + rh)):
      for x in range(x0, min(w - 2, x0 + rw)):
        grid[y][x] = "#"

  open_cells = _largest_open_component(grid)

  def open_cell() -> Tuple[int, int]:
    return rng.choice(open_cells)

  robots = []
  for i in range(robot_count):
    x, y = open_cell()
    robots.append({"id": i, "x": x, "y": y})
  jobs = []
  used = {(r["x"], r["y"]) for r in robots}
  for i in range(job_count):
    x, y = open_cell()
    while (x, y) in used:
      x, y = open_cell()
    used.add((x, y))
    jobs.append({"id": i, "x": x, "y": y, "priority": rng.randint(1, 5)})
  return {
    "desc": desc,
    "width": w,
    "height": h,
    "grid": ["".join(row) for row in grid],
    "robots": robots,
    "jobs": jobs,
    "objective": "assign every job to a robot and minimize total walking distance",
  }


def _largest_open_component(grid: List[List[str]]) -> List[Tuple[int, int]]:
  h = len(grid)
  w = len(grid[0]) if h else 0
  seen = set()
  best: List[Tuple[int, int]] = []
  for sy in range(h):
    for sx in range(w):
      if grid[sy][sx] != "." or (sx, sy) in seen:
        continue
      q = deque([(sx, sy)])
      seen.add((sx, sy))
      cells = []
      while q:
        x, y = q.popleft()
        cells.append((x, y))
        for nx, ny in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
          if 0 <= nx < w and 0 <= ny < h and grid[ny][nx] == "." and (nx, ny) not in seen:
            seen.add((nx, ny))
            q.append((nx, ny))
      if len(cells) > len(best):
        best = cells
  return best


def _distance_grid(case: Dict[str, Any], start: Tuple[int, int], target: Tuple[int, int],
                   cache: Dict[Tuple[int, int], Dict[Tuple[int, int], int]]) -> int:
  if start not in cache:
    w, h = int(case["width"]), int(case["height"])
    blocked = {(x, y) for y, row in enumerate(case["grid"]) for x, c in enumerate(row) if c == "#"}
    dist = {start: 0}
    q = deque([start])
    while q:
      x, y = q.popleft()
      nd = dist[(x, y)] + 1
      for nx, ny in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
        if 0 <= nx < w and 0 <= ny < h and (nx, ny) not in blocked and (nx, ny) not in dist:
          dist[(nx, ny)] = nd
          q.append((nx, ny))
    cache[start] = dist
  return cache[start].get(target, 10**9)


def _warehouse_route_cost(case: Dict[str, Any], assignments: List[List[int]]) -> int:
  robots = [(int(r["x"]), int(r["y"])) for r in case["robots"]]
  jobs = {int(j["id"]): (int(j["x"]), int(j["y"])) for j in case["jobs"]}
  cache: Dict[Tuple[int, int], Dict[Tuple[int, int], int]] = {}
  total = 0
  for ri, route in enumerate(assignments):
    if ri >= len(robots):
      return 10**12
    cur = robots[ri]
    for jid in route:
      d = _distance_grid(case, cur, jobs[jid], cache)
      if d >= 10**9:
        return 10**12
      total += d
      cur = jobs[jid]
  return total


def _distance_manhattan(a: Tuple[int, int], b: Tuple[int, int]) -> int:
  return abs(int(a[0]) - int(b[0])) + abs(int(a[1]) - int(b[1]))


def _warehouse_route_cost_fast(case: Dict[str, Any], assignments: List[List[int]]) -> int:
  robots = [(int(r["x"]), int(r["y"])) for r in case["robots"]]
  jobs = {int(j["id"]): (int(j["x"]), int(j["y"])) for j in case["jobs"]}
  total = 0
  for ri, route in enumerate(assignments):
    if ri >= len(robots):
      return 10**12
    cur = robots[ri]
    for jid in route:
      if jid not in jobs:
        return 10**12
      total += _distance_manhattan(cur, jobs[jid])
      cur = jobs[jid]
  return total


def _warehouse_reference(case: Dict[str, Any]) -> List[List[int]]:
  robots = [(int(r["x"]), int(r["y"])) for r in case["robots"]]
  jobs = {int(j["id"]): (int(j["x"]), int(j["y"])) for j in case["jobs"]}
  cache: Dict[Tuple[int, int], Dict[Tuple[int, int], int]] = {}
  assignments = [[] for _ in robots]
  current = list(robots)
  remaining = set(jobs.keys())
  while remaining:
    best = None
    for ri, pos in enumerate(current):
      for jid in remaining:
        d = _distance_grid(case, pos, jobs[jid], cache)
        if best is None or d < best[0]:
          best = (d, ri, jid)
    _, ri, jid = best
    assignments[ri].append(jid)
    current[ri] = jobs[jid]
    remaining.remove(jid)
  return assignments


def _warehouse_reference_fast(case: Dict[str, Any]) -> List[List[int]]:
  robots = [(int(r["x"]), int(r["y"])) for r in case["robots"]]
  jobs = {int(j["id"]): (int(j["x"]), int(j["y"])) for j in case["jobs"]}
  assignments = [[] for _ in robots]
  current = list(robots)
  remaining = set(jobs.keys())
  while remaining:
    best = None
    for ri, pos in enumerate(current):
      for jid in remaining:
        d = _distance_manhattan(pos, jobs[jid])
        if best is None or d < best[0]:
          best = (d, ri, jid)
    _, ri, jid = best
    assignments[ri].append(jid)
    current[ri] = jobs[jid]
    remaining.remove(jid)
  return assignments


def _grade_warehouse(case: Dict[str, Any], output: Any) -> tuple:
  assignments = _normalise_routes(output)
  if assignments is None:
    return 0.0, "Expected assignments/routes as [[jobId,...],...]", ""
  all_ids = [jid for route in assignments for jid in route]
  expected = {int(j["id"]) for j in case["jobs"]}
  if set(all_ids) != expected or len(all_ids) != len(expected):
    return 0.0, f"Assignments must include every job exactly once; got {len(all_ids)} visits", ""
  if len(assignments) > len(case["robots"]):
    return 0.0, f"Used {len(assignments)} robots but only {len(case['robots'])} are available", ""

  while len(assignments) < len(case["robots"]):
    assignments.append([])
  large_case = len(case["jobs"]) > WAREHOUSE_EXACT_JOB_LIMIT
  route_cost = _warehouse_route_cost_fast if large_case else _warehouse_route_cost
  reference = _warehouse_reference_fast if large_case else _warehouse_reference
  solver_cost = route_cost(case, assignments)
  if solver_cost >= 10**12:
    return 0.0, "One or more assigned jobs are unreachable", ""
  ref = reference(case)
  ref_cost = route_cost(case, ref)
  score = _clamp(ref_cost / max(1, solver_cost))
  mode = "Manhattan approximate" if large_case else "exact grid"
  explanation = (
    f"[{case['desc']}] travel {solver_cost}, reference {ref_cost}, "
    f"score {score:.3f} ({mode} grading)")
  return score, explanation, _warehouse_svg(case, assignments, solver_cost, ref_cost)


def _warehouse_svg(case: Dict[str, Any], assignments: List[List[int]], solver_cost: int,
                   ref_cost: int) -> str:
  w, h = int(case["width"]), int(case["height"])
  if w * h > 6500:
    return f"<p>Warehouse visualization skipped for {w}x{h} grid.</p>"
  cell = max(4, min(16, 760 / max(w, h)))
  jobs = {int(j["id"]): (int(j["x"]), int(j["y"])) for j in case["jobs"]}
  robots = [(int(r["x"]), int(r["y"])) for r in case["robots"]]
  colors = ["#2563eb", "#dc2626", "#16a34a", "#9333ea", "#ea580c", "#0891b2"]
  parts = [
    f"<div><p><strong>Travel:</strong> {solver_cost} (reference {ref_cost})</p>",
    f"<svg width='100%' viewBox='0 0 {w * cell:.0f} {h * cell:.0f}' style='background:white;border:1px solid #94a3b8'>",
  ]
  for y, row in enumerate(case["grid"]):
    for x, ch in enumerate(row):
      if ch == "#":
        parts.append(
          f"<rect x='{x * cell:.1f}' y='{y * cell:.1f}' width='{cell:.1f}' height='{cell:.1f}' fill='#475569' />"
        )
  for ri, route in enumerate(assignments):
    pts = [robots[ri]] + [jobs[jid] for jid in route if jid in jobs]
    if len(pts) > 1:
      path = " ".join(f"{(x + 0.5) * cell:.1f},{(y + 0.5) * cell:.1f}" for x, y in pts[:200])
      parts.append(
        f"<polyline points='{path}' fill='none' stroke='{colors[ri % len(colors)]}' stroke-width='2' opacity='0.65' />"
      )
  for jid, (x, y) in jobs.items():
    parts.append(
      f"<circle cx='{(x + 0.5) * cell:.1f}' cy='{(y + 0.5) * cell:.1f}' r='{cell * 0.22:.1f}' fill='#0f172a' />"
    )
  for x, y in robots:
    parts.append(
      f"<rect x='{(x + 0.2) * cell:.1f}' y='{(y + 0.2) * cell:.1f}' width='{cell * 0.6:.1f}' height='{cell * 0.6:.1f}' fill='#facc15' stroke='#854d0e' />"
    )
  parts.append("</svg></div>")
  return "\n".join(parts)


# ---------------------------------------------------------------------------
# Drone search coverage
# ---------------------------------------------------------------------------

DRONE_CASES = [
  (30, 22, 2, 55, 4, "2 drones, 55 steps"),
  (48, 34, 3, 85, 6, "3 drones, 85 steps"),
  (70, 48, 5, 125, 9, "5 drones, 125 steps"),
  (96, 66, 7, 180, 12, "7 drones, 180 steps"),
  (130, 90, 10, 260, 16, "10 drones, 260 steps"),
  (180, 120, 14, 380, 22, "14 drones, 380 steps"),
]


def _make_drone_case(subpass: int) -> Dict[str, Any]:
  w, h, drone_count, steps, hotspot_count, desc = DRONE_CASES[subpass]
  rng = random.Random(RANDOM_SEED + 4000 + subpass)
  obstacles = set()
  for _ in range(int(w * h * (0.035 + subpass * 0.003))):
    obstacles.add((rng.randrange(w), rng.randrange(h)))
  starts = []
  for i in range(drone_count):
    pt = (rng.randrange(2, w - 2), rng.randrange(2, h - 2))
    while pt in obstacles:
      pt = (rng.randrange(2, w - 2), rng.randrange(2, h - 2))
    starts.append({"id": i, "x": pt[0], "y": pt[1]})
  hotspots = [(rng.randrange(w), rng.randrange(h), rng.randint(40, 100), rng.uniform(4.0, 14.0))
              for _ in range(hotspot_count)]
  probability = []
  for y in range(h):
    row = []
    for x in range(w):
      if (x, y) in obstacles:
        row.append(-1)
        continue
      value = 1
      for hx, hy, amp, sigma in hotspots:
        value += int(amp * math.exp(-((x - hx)**2 + (y - hy)**2) / (2 * sigma * sigma)))
      row.append(min(255, value))
    probability.append(row)
  return {
    "desc": desc,
    "width": w,
    "height": h,
    "maxStepsPerDrone": steps,
    "drones": starts,
    "probability": probability,
    "objective": "choose adjacent grid paths that maximize unique search probability covered",
  }


def _coverage_score(case: Dict[str, Any],
                    paths: List[List[Tuple[int, int]]]) -> Tuple[int, bool, str, set]:
  w, h = int(case["width"]), int(case["height"])
  max_steps = int(case["maxStepsPerDrone"])
  probs = case["probability"]
  covered = set()
  for di, path in enumerate(paths):
    if di >= len(case["drones"]):
      return 0, False, "too many drone paths", covered
    if len(path) > max_steps + 1:
      return 0, False, f"drone {di} path has {len(path)-1} steps, max {max_steps}", covered
    start = (int(case["drones"][di]["x"]), int(case["drones"][di]["y"]))
    if not path:
      path = [start]
    if path[0] != start:
      path = [start] + path
    prev = path[0]
    for x, y in path:
      if not (0 <= x < w and 0 <= y < h) or probs[y][x] < 0:
        return 0, False, f"drone {di} enters blocked/out-of-bounds cell", covered
      if abs(x - prev[0]) + abs(y - prev[1]) > 1:
        return 0, False, f"drone {di} makes a non-adjacent move", covered
      covered.add((x, y))
      prev = (x, y)
  total = sum(probs[y][x] for x, y in covered)
  return total, True, "ok", covered


def _normalise_paths(output: Any) -> Optional[List[List[Tuple[int, int]]]]:
  raw = output.get("paths", output.get("drones", output)) if isinstance(output, dict) else output
  if isinstance(raw, dict):
    raw = [raw[k] for k in sorted(raw.keys(), key=lambda x: str(x))]
  if not isinstance(raw, list):
    return None
  paths = []
  for path in raw:
    if isinstance(path, dict) and "path" in path:
      path = path["path"]
    if not isinstance(path, list):
      return None
    coords = []
    for item in path:
      pt = _point_from_any(item)
      if pt is None:
        return None
      coords.append(pt)
    paths.append(coords)
  return paths


def _drone_reference(case: Dict[str, Any]) -> List[List[Tuple[int, int]]]:
  probs = case["probability"]
  w, h = int(case["width"]), int(case["height"])
  steps = int(case["maxStepsPerDrone"])
  covered = set()
  paths = []
  for d in case["drones"]:
    cur = (int(d["x"]), int(d["y"]))
    path = [cur]
    covered.add(cur)
    for _ in range(steps):
      candidates = [
        cur, (cur[0] + 1, cur[1]), (cur[0] - 1, cur[1]), (cur[0], cur[1] + 1), (cur[0], cur[1] - 1)
      ]
      candidates = [(x, y) for x, y in candidates if 0 <= x < w and 0 <= y < h and probs[y][x] >= 0]
      best = max(candidates,
                 key=lambda p: (0 if p in covered else probs[p[1]][p[0]], probs[p[1]][p[0]]))
      cur = best
      path.append(cur)
      covered.add(cur)
    paths.append(path)
  return paths


def _grade_drone(case: Dict[str, Any], output: Any) -> tuple:
  paths = _normalise_paths(output)
  if paths is None:
    return 0.0, "Expected paths as [[[x,y],...],...]", ""
  solver_cov, valid, msg, covered = _coverage_score(case, paths)
  if not valid:
    return 0.0, msg, ""
  ref_paths = _drone_reference(case)
  ref_cov, _, _, _ = _coverage_score(case, ref_paths)
  score = _clamp(solver_cov / max(1, ref_cov))
  explanation = f"[{case['desc']}] coverage {solver_cov}, reference {ref_cov}, score {score:.3f}"
  return score, explanation, _drone_svg(case, paths, covered, solver_cov, ref_cov)


def _drone_svg(case: Dict[str, Any], paths: List[List[Tuple[int, int]]], covered: set,
               solver_cov: int, ref_cov: int) -> str:
  w, h = int(case["width"]), int(case["height"])
  if w * h > 9000:
    return f"<p>Search heatmap visualization skipped for {w}x{h} grid.</p>"
  cell = max(3, min(14, 760 / max(w, h)))
  colors = ["#2563eb", "#dc2626", "#16a34a", "#9333ea", "#ea580c", "#0891b2"]
  parts = [
    f"<div><p><strong>Coverage:</strong> {solver_cov} (reference {ref_cov})</p>",
    f"<svg width='100%' viewBox='0 0 {w * cell:.0f} {h * cell:.0f}' style='background:white;border:1px solid #94a3b8'>",
  ]
  probs = case["probability"]
  maxp = max(max(row) for row in probs)
  for y, row in enumerate(probs):
    for x, v in enumerate(row):
      if v < 0:
        color = "#334155"
      else:
        shade = int(245 - 170 * (v / max(1, maxp)))
        color = f"rgb(255,{shade},{shade})"
      if (x, y) in covered:
        color = "#fde047"
      parts.append(
        f"<rect x='{x * cell:.1f}' y='{y * cell:.1f}' width='{cell:.1f}' height='{cell:.1f}' fill='{color}' />"
      )
  for i, path in enumerate(paths):
    if len(path) > 1:
      pts = " ".join(f"{(x + 0.5) * cell:.1f},{(y + 0.5) * cell:.1f}" for x, y in path[:500])
      parts.append(
        f"<polyline points='{pts}' fill='none' stroke='{colors[i % len(colors)]}' stroke-width='2' opacity='0.8' />"
      )
  parts.append("</svg></div>")
  return "\n".join(parts)


# ---------------------------------------------------------------------------
# Mission schedule optimizer
# --------------------------------------------------------------------------

SCHEDULE_CASES = [
  (24, 3, 0.10, "24 tasks, 3 workstations"),
  (70, 5, 0.08, "70 tasks, 5 workstations"),
  (160, 8, 0.055, "160 tasks, 8 workstations"),
  (420, 12, 0.035, "420 tasks, 12 workstations"),
  (900, 18, 0.025, "900 tasks, 18 workstations"),
  (1800, 28, 0.018, "1.8K tasks, 28 workstations"),
]


def _make_schedule_case(subpass: int) -> Dict[str, Any]:
  n, machines, dep_prob, desc = SCHEDULE_CASES[subpass]
  rng = random.Random(RANDOM_SEED + 5000 + subpass)
  tasks = []
  for i in range(n):
    deps = []
    max_back = min(i, 40 + subpass * 20)
    for j in range(max(0, i - max_back), i):
      if rng.random() < dep_prob:
        deps.append(j)
    duration = rng.randint(2, 18 + subpass * 2)
    horizon = max(80, int(n * 8 / machines) + 60 + subpass * 25)
    deadline = rng.randint(duration + 8, horizon)
    tasks.append({
      "id": i,
      "duration": duration,
      "deps": deps,
      "deadline": deadline,
      "weight": rng.randint(1, 8),
    })
  return {
    "desc": desc,
    "machineCount": machines,
    "tasks": tasks,
    "objective": "schedule all tasks after dependencies to minimize weighted lateness and makespan",
  }


def _topological_order(tasks: List[Dict[str, Any]]) -> List[int]:
  remaining_deps = {int(t["id"]): set(map(int, t["deps"])) for t in tasks}
  children = {int(t["id"]): [] for t in tasks}
  for t in tasks:
    tid = int(t["id"])
    for d in t["deps"]:
      children[int(d)].append(tid)
  ready = [tid for tid, deps in remaining_deps.items() if not deps]
  ready.sort()
  order = []
  while ready:
    tid = ready.pop(0)
    order.append(tid)
    for c in children[tid]:
      remaining_deps[c].discard(tid)
      if not remaining_deps[c]:
        ready.append(c)
    ready.sort()
  return order


def _schedule_from_order(case: Dict[str, Any], order: List[int]) -> List[Dict[str, int]]:
  tasks = {int(t["id"]): t for t in case["tasks"]}
  machines = int(case["machineCount"])
  machine_free = [0] * machines
  finish = {}
  sched = []
  for tid in order:
    if tid not in tasks:
      continue
    task = tasks[tid]
    ready = 0
    for d in task["deps"]:
      ready = max(ready, finish.get(int(d), 0))
    m = min(range(machines), key=lambda mi: max(machine_free[mi], ready))
    start = max(machine_free[m], ready)
    finish[tid] = start + int(task["duration"])
    machine_free[m] = finish[tid]
    sched.append({"id": tid, "start": start, "machine": m})
  return sched


def _normalise_schedule(case: Dict[str, Any], output: Any) -> Optional[List[Dict[str, int]]]:
  if isinstance(output, dict) and "schedule" in output:
    raw = output["schedule"]
  elif isinstance(output, dict) and "order" in output:
    try:
      return _schedule_from_order(case, [int(x) for x in output["order"]])
    except Exception:
      return None
  else:
    raw = output
  if isinstance(raw, list) and raw and all(not isinstance(x, dict) for x in raw):
    try:
      return _schedule_from_order(case, [int(x) for x in raw])
    except Exception:
      return None
  if not isinstance(raw, list):
    return None
  sched = []
  for item in raw:
    if not isinstance(item, dict):
      return None
    try:
      sched.append({
        "id": int(item["id"]),
        "start": int(item["start"]),
        "machine": int(item["machine"]),
      })
    except Exception:
      return None
  return sched


def _schedule_cost(case: Dict[str, Any],
                   sched: List[Dict[str, int]]) -> Tuple[Optional[float], str, int]:
  tasks = {int(t["id"]): t for t in case["tasks"]}
  if {s["id"] for s in sched} != set(tasks.keys()) or len(sched) != len(tasks):
    return None, "schedule must include every task exactly once", 0
  machines = int(case["machineCount"])
  intervals = [[] for _ in range(machines)]
  finish = {}
  for s in sched:
    tid, start, machine = int(s["id"]), int(s["start"]), int(s["machine"])
    if start < 0 or machine < 0 or machine >= machines:
      return None, "negative start or invalid machine", 0
    dur = int(tasks[tid]["duration"])
    finish[tid] = start + dur
    intervals[machine].append((start, start + dur, tid))
  for machine_intervals in intervals:
    machine_intervals.sort()
    for a, b in zip(machine_intervals, machine_intervals[1:]):
      if a[1] > b[0]:
        return None, f"machine overlap between tasks {a[2]} and {b[2]}", 0
  for tid, task in tasks.items():
    start = next(s["start"] for s in sched if s["id"] == tid)
    for dep in task["deps"]:
      if finish[int(dep)] > start:
        return None, f"task {tid} starts before dependency {dep} finishes", 0
  makespan = max(finish.values()) if finish else 0
  lateness = 0
  for tid, task in tasks.items():
    lateness += max(0, finish[tid] - int(task["deadline"])) * int(task["weight"])
  cost = lateness + makespan * 0.05
  return cost, "ok", makespan


def _schedule_reference(case: Dict[str, Any]) -> List[Dict[str, int]]:
  candidates = []
  for rule in ("id", "deadline", "weighted_deadline", "shortest", "weighted_shortest", "critical",
               "slack"):
    sched = _schedule_with_rule(case, rule)
    cost, _, _ = _schedule_cost(case, sched)
    if cost is not None:
      candidates.append((cost, sched))
  if not candidates:
    order = _topological_order(case["tasks"])
    return _schedule_from_order(case, order)
  candidates.sort(key=lambda x: x[0])
  return candidates[0][1]


def _schedule_with_rule(case: Dict[str, Any], rule: str) -> List[Dict[str, int]]:
  tasks = {int(t["id"]): t for t in case["tasks"]}
  deps_left = {tid: set(map(int, t["deps"])) for tid, t in tasks.items()}
  children = {tid: [] for tid in tasks}
  for tid, task in tasks.items():
    for d in task["deps"]:
      children[int(d)].append(tid)
  critical = _critical_lengths(tasks, children)
  machines = int(case["machineCount"])
  machine_free = [0] * machines
  finish = {}
  ready = {tid for tid, deps in deps_left.items() if not deps}
  unscheduled = set(tasks.keys())
  sched = []
  while unscheduled:
    if not ready:
      order = _topological_order(case["tasks"])
      return _schedule_from_order(case, order)
    tid = min(ready, key=lambda i: _schedule_priority_key(rule, tasks[i], critical[i], finish))
    ready.remove(tid)
    unscheduled.remove(tid)
    task = tasks[tid]
    dep_ready = max((finish[int(d)] for d in task["deps"]), default=0)
    m = min(range(machines), key=lambda mi: max(machine_free[mi], dep_ready))
    start = max(machine_free[m], dep_ready)
    end = start + int(task["duration"])
    finish[tid] = end
    machine_free[m] = end
    sched.append({"id": tid, "start": start, "machine": m})
    for c in children[tid]:
      deps_left[c].discard(tid)
      if not deps_left[c] and c in unscheduled:
        ready.add(c)
  return sched


def _critical_lengths(tasks: Dict[int, Dict[str, Any]],
                      children: Dict[int, List[int]]) -> Dict[int, int]:
  order = sorted(tasks.keys(), reverse=True)
  critical = {}
  for tid in order:
    child_best = max((critical.get(c, 0) for c in children[tid]), default=0)
    critical[tid] = int(tasks[tid]["duration"]) + child_best
  return critical


def _schedule_priority_key(rule: str, task: Dict[str, Any], critical: int,
                           finish: Dict[int, int]) -> tuple:
  duration = int(task["duration"])
  deadline = int(task["deadline"])
  weight = int(task["weight"])
  dep_ready = max((finish.get(int(d), 0) for d in task["deps"]), default=0)
  if rule == "deadline":
    return (deadline, -weight, duration, int(task["id"]))
  if rule == "weighted_deadline":
    return (deadline / max(1, weight), duration, int(task["id"]))
  if rule == "shortest":
    return (duration, deadline, int(task["id"]))
  if rule == "weighted_shortest":
    return (duration / max(1, weight), deadline, int(task["id"]))
  if rule == "critical":
    return (-critical, deadline, int(task["id"]))
  if rule == "slack":
    return (deadline - dep_ready - duration - critical, deadline, int(task["id"]))
  return (int(task["id"]), )


def _grade_schedule(case: Dict[str, Any], output: Any) -> tuple:
  sched = _normalise_schedule(case, output)
  if sched is None:
    return 0.0, "Expected schedule [{id,start,machine},...] or order [taskId,...]", ""
  solver_cost, msg, solver_makespan = _schedule_cost(case, sched)
  if solver_cost is None:
    return 0.0, msg, ""
  ref = _schedule_reference(case)
  ref_cost, _, ref_makespan = _schedule_cost(case, ref)
  score = _clamp(max(1.0, ref_cost) / max(1.0, solver_cost))
  explanation = (
    f"[{case['desc']}] cost {solver_cost:.1f}, makespan {solver_makespan}; reference {ref_cost:.1f}, makespan {ref_makespan}; score {score:.3f}"
  )
  return score, explanation, _schedule_svg(case, sched, solver_cost, ref_cost)


def _schedule_svg(case: Dict[str, Any], sched: List[Dict[str, int]], solver_cost: float,
                  ref_cost: float) -> str:
  if len(case["tasks"]) > 180:
    return f"<p>Gantt visualization skipped for {len(case['tasks'])} tasks.</p>"
  tasks = {int(t["id"]): t for t in case["tasks"]}
  machines = int(case["machineCount"])
  makespan = max((int(s["start"]) + int(tasks[s["id"]]["duration"]) for s in sched), default=1)
  width = 900
  row_h = 22
  height = machines * row_h + 30
  parts = [
    f"<div><p><strong>Cost:</strong> {solver_cost:.1f} (reference {ref_cost:.1f})</p>",
    f"<svg width='100%' viewBox='0 0 {width} {height}' style='background:white;border:1px solid #94a3b8'>",
  ]
  for m in range(machines):
    y = 20 + m * row_h
    parts.append(f"<text x='4' y='{y + 13}' font-size='10' fill='#334155'>M{m}</text>")
    parts.append(f"<line x1='32' y1='{y + 10}' x2='{width - 8}' y2='{y + 10}' stroke='#e2e8f0' />")
  for s in sched:
    t = tasks[int(s["id"])]
    x = 34 + int(s["start"]) / max(1, makespan) * (width - 48)
    ww = max(2, int(t["duration"]) / max(1, makespan) * (width - 48))
    y = 20 + int(s["machine"]) * row_h
    late = int(s["start"]) + int(t["duration"]) > int(t["deadline"])
    color = "#ef4444" if late else "#2563eb"
    parts.append(
      f"<rect x='{x:.1f}' y='{y:.1f}' width='{ww:.1f}' height='14' rx='2' fill='{color}' opacity='0.75' />"
    )
  parts.append("</svg></div>")
  return "\n".join(parts)


PROBLEMS = {
  "delivery": {
    "key":
    "delivery",
    "title":
    "Delivery Route Planner",
    "tags": [
      "combinatorial optimization",
      "route planning",
      "vehicle routing",
      "spatial data",
      "heuristics",
    ],
    "cases":
    DELIVERY_CASES,
    "make_case":
    _make_delivery_case,
    "grade_output":
    _grade_delivery,
    "skip_local_grade_reason":
    _delivery_skip_reason,
    "summary":
    "Plan multi-vehicle delivery routes over clustered city coordinates.",
    "prompt":
    """Plan delivery routes for a fleet of vans.

The input JSON contains:
- `depot`: [x, y]
- `truckCount`: maximum number of vans/routes
- `packages`: objects `{id, x, y}`

Return JSON:
```
{"routes": [[packageId, packageId, ...], ...]}
```
Every package must appear exactly once. Each route starts and ends at the depot.
The score compares your total distance to a deterministic reference planner.""",
  },
  "wildfire": {
    "key":
    "wildfire",
    "title":
    "Wildfire Firebreak Planner",
    "tags": [
      "grid problems",
      "simulation",
      "resource allocation",
      "pathfinding",
      "risk optimization",
    ],
    "cases":
    WILDFIRE_CASES,
    "make_case":
    _make_wildfire_case,
    "grade_output":
    _grade_wildfire,
    "summary":
    "Choose firebreak cells on a forest map to protect towns from spreading fire.",
    "prompt":
    """Choose where to cut firebreaks before a wildfire spreads.

The input JSON contains:
- `width`, `height`
- `grid`: strings where `.` is burnable forest and `~` is water/rock
- `ignitions`: starting fire cells `[x, y]`
- `towns`: objects `{id, x, y, value}`
- `cutBudget`: maximum number of forest cells you may cut

Fire spreads in four directions through burnable cells. Water/rock and your cut
cells block fire. Return JSON:
```
{"firebreaks": [[x, y], [x, y], ...]}
```
The score is the fraction of otherwise-burned town value saved.""",
  },
  "warehouse": {
    "key":
    "warehouse",
    "title":
    "Warehouse Robot Job Planner",
    "tags": [
      "multi-agent planning",
      "warehouse logistics",
      "grid problems",
      "pathfinding",
      "route planning",
    ],
    "cases":
    WAREHOUSE_CASES,
    "make_case":
    _make_warehouse_case,
    "grade_output":
    _grade_warehouse,
    "summary":
    "Assign warehouse pick jobs to robots moving through aisles and blocked shelves.",
    "prompt":
    """Assign warehouse pick jobs to robots.

The input JSON contains:
- `grid`: strings where `.` is open floor and `#` is blocked shelving
- `robots`: objects `{id, x, y}`
- `jobs`: objects `{id, x, y, priority}`

Return JSON:
```
{"assignments": [[jobId, jobId, ...], ...]}
```
The array index is the robot id. Every job must appear exactly once. The grader
uses shortest grid walking distance through open cells and compares total travel
to a deterministic greedy reference.""",
  },
  "drone": {
    "key":
    "drone",
    "title":
    "Drone Search Coverage Planner",
    "tags": [
      "coverage planning",
      "grid problems",
      "pathfinding",
      "multi-agent planning",
      "resource allocation",
    ],
    "cases":
    DRONE_CASES,
    "make_case":
    _make_drone_case,
    "grade_output":
    _grade_drone,
    "summary":
    "Route search drones over a probability heatmap with obstacles and battery limits.",
    "prompt":
    """Plan grid paths for search drones.

The input JSON contains:
- `probability`: 2D array; -1 is an obstacle, positive values are search value
- `drones`: starting cells `{id, x, y}`
- `maxStepsPerDrone`

Return JSON:
```
{"paths": [[[x,y], [x,y], ...], ...]}
```
Each path belongs to the drone with the same index. Moves must be up/down/left/right
or stay in place, must stay inside the grid, and must avoid obstacles. The score is
unique probability covered compared with a greedy reference search.""",
  },
  "schedule": {
    "key":
    "schedule",
    "title":
    "Space Mission Work Schedule",
    "tags": [
      "scheduling",
      "precedence constraints",
      "constraint solving",
      "parallel machines",
      "combinatorial optimization",
    ],
    "cases":
    SCHEDULE_CASES,
    "make_case":
    _make_schedule_case,
    "grade_output":
    _grade_schedule,
    "summary":
    "Schedule dependent mission tasks across parallel workstations with deadlines.",
    "prompt":
    """Schedule mission-control work across parallel workstations.

The input JSON contains:
- `machineCount`
- `tasks`: objects `{id, duration, deps, deadline, weight}`

A task may start only after every dependency has finished. Return either:
```
{"schedule": [{"id": taskId, "start": integerTime, "machine": machineIndex}, ...]}
```
or a simpler topological order:
```
{"order": [taskId, taskId, ...]}
```
If you return only an order, the grader will place tasks greedily in that order.
The score compares weighted lateness plus makespan against a deterministic list
scheduler reference.""",
  },
}
