"""
Test 38: Exact Cover (Python Implementation)

The model must write Python code that selects set indices forming an exact cover:
every universe element appears in exactly one chosen set.

This benchmark uses planted instances with a known unique solution, allowing
strict pass/fail grading.
"""

import random
import subprocess
import time
from typing import List, Set, Tuple, Dict, Any, Iterable
from solver_utils import StreamingInputFile

title = "Exact Cover (Python)"
TIMEOUT_SECONDS = 30
RANDOM_SEED = 38383838


def _iter_instance_sets(solution_sets: int, block_size: int, decoy_sets: int,
                        max_blocks_per_decoy: int, seed: int) -> Iterable[List[int]]:
  """Yield sets for a planted unique exact-cover instance.

  Construction:
  - Universe is partitioned into `solution_sets` disjoint blocks of size `block_size`.
  - The first `solution_sets` sets are exactly those blocks (the planted solution).
  - Each block has an anchor element (first element) that appears only in its own
    planted set, making the exact cover unique.
  - Additional decoy sets contain only non-anchor elements.
  """
  if solution_sets <= 0:
    raise ValueError("solution_sets must be > 0")
  if block_size < 2:
    raise ValueError("block_size must be >= 2")
  if max_blocks_per_decoy < 2:
    raise ValueError("max_blocks_per_decoy must be >= 2")

  rng = random.Random(seed)

  # Planted sets (indices 0..solution_sets-1)
  for b in range(solution_sets):
    start = b * block_size
    yield list(range(start, start + block_size))

  # Decoy sets (no anchors, so they can never replace planted sets in an exact cover)
  for _ in range(decoy_sets):
    blocks_to_mix = rng.randint(2, min(max_blocks_per_decoy, solution_sets))
    chosen_blocks = rng.sample(range(solution_sets), blocks_to_mix)
    elems = set()
    for b in chosen_blocks:
      start = b * block_size
      picks = 1 if block_size <= 3 else rng.randint(1, min(3, block_size - 1))
      while picks > 0:
        e = start + 1 + rng.randrange(block_size - 1)  # skip anchor (start)
        if e not in elems:
          elems.add(e)
          picks -= 1
    yield sorted(elems)


def _get_case_params(subpass: int) -> Tuple[int, int, int, int, int, int]:
  case = TEST_CASES[subpass]
  solution_sets = case["solution_sets"]
  block_size = case["block_size"]
  decoy_sets = case["decoy_sets"]
  max_blocks_per_decoy = case["max_blocks_per_decoy"]

  if solution_sets <= 0:
    raise ValueError(f"Invalid solution_sets={solution_sets} for subpass {subpass}")
  if block_size < 2:
    raise ValueError(f"Invalid block_size={block_size} for subpass {subpass}")
  if max_blocks_per_decoy < 2:
    raise ValueError(f"Invalid max_blocks_per_decoy={max_blocks_per_decoy} for subpass {subpass}")

  universe_size = solution_sets * block_size
  num_sets = solution_sets + decoy_sets
  return solution_sets, block_size, decoy_sets, max_blocks_per_decoy, universe_size, num_sets


TEST_CASES = [
  {
    "solution_sets": 8,
    "block_size": 5,
    "decoy_sets": 24,
    "max_blocks_per_decoy": 3,
    "desc": "tiny: universe=40, sets=32"
  },
  {
    "solution_sets": 16,
    "block_size": 5,
    "decoy_sets": 64,
    "max_blocks_per_decoy": 3,
    "desc": "small: universe=80, sets=80"
  },
  {
    "solution_sets": 24,
    "block_size": 6,
    "decoy_sets": 180,
    "max_blocks_per_decoy": 3,
    "desc": "universe=144, sets=204"
  },
  {
    "solution_sets": 40,
    "block_size": 6,
    "decoy_sets": 360,
    "max_blocks_per_decoy": 3,
    "desc": "universe=240, sets=400"
  },
  {
    "solution_sets": 70,
    "block_size": 6,
    "decoy_sets": 840,
    "max_blocks_per_decoy": 4,
    "desc": "universe=420, sets=910"
  },
  {
    "solution_sets": 120,
    "block_size": 7,
    "decoy_sets": 1680,
    "max_blocks_per_decoy": 4,
    "desc": "universe=840, sets=1.8K"
  },
  {
    "solution_sets": 200,
    "block_size": 7,
    "decoy_sets": 3200,
    "max_blocks_per_decoy": 4,
    "desc": "universe=1.4K, sets=3.4K"
  },
  {
    "solution_sets": 350,
    "block_size": 7,
    "decoy_sets": 7000,
    "max_blocks_per_decoy": 4,
    "desc": "universe=2.45K, sets=7.35K"
  },
  {
    "solution_sets": 600,
    "block_size": 8,
    "decoy_sets": 15000,
    "max_blocks_per_decoy": 4,
    "desc": "universe=4.8K, sets=15.6K"
  },
  {
    "solution_sets": 1000,
    "block_size": 8,
    "decoy_sets": 30000,
    "max_blocks_per_decoy": 4,
    "desc": "universe=8K, sets=31K"
  },
  {
    "solution_sets": 2000,
    "block_size": 8,
    "decoy_sets": 80000,
    "max_blocks_per_decoy": 4,
    "desc": "universe=16K, sets=82K"
  },
  {
    "solution_sets": 5000,
    "block_size": 8,
    "decoy_sets": 250000,
    "max_blocks_per_decoy": 5,
    "desc": "universe=40K, sets=255K"
  },
  {
    "solution_sets": 10000,
    "block_size": 8,
    "decoy_sets": 600000,
    "max_blocks_per_decoy": 5,
    "desc": "universe=80K, sets=610K"
  },
  {
    "solution_sets": 20000,
    "block_size": 8,
    "decoy_sets": 2000000,
    "max_blocks_per_decoy": 5,
    "desc": "universe=160K, sets=2.02M"
  },
  {
    "solution_sets": 50000,
    "block_size": 8,
    "decoy_sets": 5000000,
    "max_blocks_per_decoy": 5,
    "desc": "universe=400K, sets=5.05M"
  },
  {
    "solution_sets": 100000,
    "block_size": 8,
    "decoy_sets": 12000000,
    "max_blocks_per_decoy": 5,
    "desc": "universe=800K, sets=12.1M"
  },
]

INSTANCE_CACHE: Dict[int, Any] = {}
_INPUT_FILE_CACHE: Dict[int, StreamingInputFile] = {}
STREAMING_THRESHOLD_SETS = 50_000
LAST_EXACT_COVER_VIZ: Dict[Tuple[int, str], dict] = {}


def get_instance(subpass: int):
  if subpass not in INSTANCE_CACHE:
    solution_sets, block_size, decoy_sets, max_blocks_per_decoy, universe_size, _ = _get_case_params(subpass)
    sets = list(_iter_instance_sets(solution_sets, block_size, decoy_sets, max_blocks_per_decoy,
                                    RANDOM_SEED + subpass))
    INSTANCE_CACHE[subpass] = (universe_size, sets, solution_sets)
  return INSTANCE_CACHE[subpass]


def _should_use_streaming(subpass: int) -> bool:
  return _estimate_sets(subpass) > STREAMING_THRESHOLD_SETS


def _estimate_sets(subpass: int) -> int:
  solution_sets, _, decoy_sets, _, _, _ = _get_case_params(subpass)
  return solution_sets + decoy_sets


def _get_streaming_input(subpass: int) -> StreamingInputFile:
  if subpass in _INPUT_FILE_CACHE:
    return _INPUT_FILE_CACHE[subpass]

  solution_sets, block_size, decoy_sets, max_blocks_per_decoy, universe_size, num_sets = _get_case_params(subpass)
  cache_key = (
    f"exactcover38_v2|sol={solution_sets}|b={block_size}|d={decoy_sets}|"
    f"mix={max_blocks_per_decoy}|seed={RANDOM_SEED + subpass}"
  )

  def generator():
    yield f"{universe_size} {num_sets}\n"
    for s in _iter_instance_sets(solution_sets, block_size, decoy_sets, max_blocks_per_decoy,
                                 RANDOM_SEED + subpass):
      yield " ".join(map(str, s)) + "\n"

  input_file = StreamingInputFile(cache_key, generator, "test38_exactcover")
  _INPUT_FILE_CACHE[subpass] = input_file
  return input_file


def format_input(universe_size: int, sets: List[List[int]]) -> str:
  lines = [f"{universe_size} {len(sets)}"]
  for s in sets:
    lines.append(" ".join(map(str, s)))
  return "\n".join(lines)


def prepareSubpassPrompt(subPass: int) -> str:
  """Generate the prompt for subpass 0."""
  if subPass != 0:
    raise StopIteration

  return """You are writing Python code to solve the **Exact Cover** problem.

Given a universe of elements and a list of sets, choose set indices such that:
1) every element is covered, and
2) no element is covered more than once.

The evaluator is strict: only the exact expected solution is accepted.

**Input format (stdin)**
```
U S
e1 e2 e3 ...
e1 e2 ...
... (S lines total)
```
- `U` = number of elements in the universe (elements are `0..U-1`)
- `S` = number of available sets
- each following line is one set, listed as space-separated element IDs

**Output format (stdout)**
```
SOLUTION
i1 i2 i3 ... ik
```
or
```
NO SOLUTION
```

- `i1..ik` are 0-based set indices into the input list.

**What is checked**
1. Output format is valid.
2. Indices are valid and distinct.
3. Chosen indices are exactly the planted exact-cover solution.

Write complete Python code that reads stdin and writes stdout.
"""


# List of subpasses to grade the single answer against all difficulty levels


extraGradeAnswerRuns = list(range(len(TEST_CASES)))

structure = {
  "type": "object",
  "properties": {
    "reasoning": {
      "type": "string",
      "description": "Explain your exact cover approach"
    },
    "python_code": {
      "type": "string",
      "description": "Complete Python code with solver function that handles all scales"
    }
  },
  "required": ["reasoning", "python_code"],
  "additionalProperties": False
}


def _parse_solution_output(stdout: str) -> Tuple[str, List[int]]:
  lines = [line.strip() for line in stdout.strip().splitlines() if line.strip()]
  if not lines:
    raise ValueError("No output")

  header = lines[0].upper()
  if header not in ("SOLUTION", "NO SOLUTION"):
    raise ValueError("First line must be SOLUTION or NO SOLUTION")

  indices: List[int] = []
  for line in lines[1:]:
    indices.extend(int(tok) for tok in line.split())

  return header, indices


def _expected_solution_set(solution_sets: int) -> Set[int]:
  return set(range(solution_sets))


def _build_exact_cover_viz(universe_size: int, sets: List[List[int]], solution_sets: int,
                           chosen_indices: List[int], valid: bool, msg: str) -> dict:
  expected = _expected_solution_set(solution_sets)
  chosen = set(chosen_indices)
  return {
    "universe_size": universe_size,
    "sets": sets,
    "solution_sets": solution_sets,
    "chosen": sorted(chosen),
    "expected": sorted(expected),
    "missing": sorted(expected - chosen),
    "extras": sorted(chosen - expected),
    "valid": valid,
    "msg": msg,
  }


def gradeAnswer(result: dict, subPass: int, aiEngineName: str) -> tuple:
  if not result or "python_code" not in result:
    return 0.0, "No Python code provided"

  case = TEST_CASES[subPass]
  solution_sets, _, _, _, universe_size, num_sets = _get_case_params(subPass)
  expected_solution = _expected_solution_set(solution_sets)
  use_streaming = _should_use_streaming(subPass)

  try:
    if use_streaming:
      t = time.time()
      streaming_input = _get_streaming_input(subPass)
      print(f"  Generating/caching input file for {case['desc']}...")
      input_file_path = streaming_input.generate()
      file_size_mb = streaming_input.get_size_bytes() / (1024 * 1024)
      print(f"  Input file: {file_size_mb:.1f} MB")
      if time.time() - t > 1:
        print(f"  Time to generate: {time.time() - t:.2f}s")

      start = time.time()
      with open(input_file_path, 'r') as f:
        proc = subprocess.run(["python", "-c", result["python_code"]],
                              stdin=f,
                              capture_output=True,
                              text=True,
                              timeout=TIMEOUT_SECONDS)
      exec_time = time.time() - start
    else:
      universe_size, sets, _ = get_instance(subPass)
      input_data = format_input(universe_size, sets)

      start = time.time()
      proc = subprocess.run(["python", "-c", result["python_code"]],
                            input=input_data,
                            capture_output=True,
                            text=True,
                            timeout=TIMEOUT_SECONDS)
      exec_time = time.time() - start

    if proc.returncode != 0:
      return 0.0, f"Runtime error: {proc.stderr[:200]}"

    try:
      status, indices = _parse_solution_output(proc.stdout)
    except Exception as e:
      return 0.0, f"[{case['desc']}] Invalid output format: {str(e)[:120]}"

    if status == "NO SOLUTION":
      valid = False
      msg = "Reported NO SOLUTION, but instance has a planted exact cover"
      if universe_size <= 400 and num_sets <= 500:
        u, s, sol_k = get_instance(subPass)
        LAST_EXACT_COVER_VIZ[(subPass, aiEngineName)] = _build_exact_cover_viz(
          u, s, sol_k, [], valid, msg
        )
      return 0.0, f"[{case['desc']}] FAIL: {msg}"

    seen = set()
    for idx in indices:
      if idx < 0 or idx >= num_sets:
        return 0.0, f"[{case['desc']}] FAIL: invalid set index {idx}"
      if idx in seen:
        return 0.0, f"[{case['desc']}] FAIL: duplicate set index {idx}"
      seen.add(idx)

    if len(indices) != solution_sets:
      valid = False
      msg = f"Reported {len(indices)} sets, but minimum exact cover uses {solution_sets} sets"
    else:
      chosen = set(indices)
      if chosen == expected_solution:
        valid = True
        msg = "Valid planted exact cover"
      else:
        missing = sorted(expected_solution - chosen)
        extras = sorted(chosen - expected_solution)
        miss_preview = ", ".join(map(str, missing[:6])) if missing else "none"
        extra_preview = ", ".join(map(str, extras[:6])) if extras else "none"
        valid = False
        msg = f"Wrong index set (missing: {miss_preview}; extras: {extra_preview})"

    if universe_size <= 400 and num_sets <= 500:
      u, s, sol_k = get_instance(subPass)
      LAST_EXACT_COVER_VIZ[(subPass, aiEngineName)] = _build_exact_cover_viz(
        u, s, sol_k, indices, valid, msg
      )

    if not valid:
      return 0.0, f"[{case['desc']}] FAIL: {msg}"
    return 1.0, f"[{case['desc']}] PASS: exact cover found with {solution_sets} sets in {exec_time:.2f}s"

  except subprocess.TimeoutExpired:
    return 0.0, f"[{case['desc']}] FAIL: Timeout"
  except Exception as e:
    return 0.0, f"[{case['desc']}] Error: {str(e)[:100]}"


def resultToNiceReport(result: dict, subPass: int, aiEngineName: str) -> str:
  if not result:
    return "<p style='color:red'>No result provided</p>"
  case = TEST_CASES[subPass]
  solution_sets, block_size, decoy_sets, _, universe_size, num_sets = _get_case_params(subPass)
  html = f"<h4>Exact Cover - {case['desc']}</h4>"
  html += (
    f"<p style='font-size:12px;color:#475569;margin:6px 0;'>"
    f"Universe: {universe_size:,} elements | Total sets: {num_sets:,} | "
    f"Planted sets: {solution_sets:,} (block size {block_size}) | Decoys: {decoy_sets:,}</p>"
  )
  if "reasoning" in result:
    r = result['reasoning'][:400] + ('...' if len(result.get('reasoning', '')) > 400 else '')
    html += f"<p><strong>Approach:</strong> {r.replace('<', '&lt;').replace('>', '&gt;')}</p>"
  if "python_code" in result:
    code = result["python_code"].replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    html += f"<details><summary>View Python Code ({len(result['python_code'])} chars)</summary><pre>{code}</pre></details>"
  viz = LAST_EXACT_COVER_VIZ.get((subPass, aiEngineName))
  if viz and viz.get("universe_size", 0) <= 400 and len(viz.get("sets", [])) <= 500:
    html += _generate_exact_cover_svg(viz)
  return html


highLevelSummary = """
Exact Cover asks for a subset of sets that covers each universe element exactly once.

This test uses planted, uniquely-solvable instances for strict pass/fail grading.
"""


def _generate_exact_cover_svg(viz: dict) -> str:
  universe_size = viz["universe_size"]
  sets = viz["sets"]
  solution_sets = viz["solution_sets"]
  chosen = set(viz["chosen"])
  expected = set(viz["expected"])
  missing = set(viz["missing"])
  extras = set(viz["extras"])

  display_elements = min(universe_size, 70)
  display_sets = min(len(sets), 80)
  cell = 8
  margin_x = 70
  margin_y = 44
  width = margin_x + display_elements * cell + 24
  height = margin_y + display_sets * cell + 40

  def color_for_row(set_idx: int) -> str:
    if set_idx in expected and set_idx in chosen:
      return "#22c55e"
    if set_idx in missing:
      return "#ef4444"
    if set_idx in extras:
      return "#f59e0b"
    if set_idx < solution_sets:
      return "#64748b"
    return "#334155"

  cells = []
  for r in range(display_sets):
    row_color = color_for_row(r)
    for e in sets[r]:
      if e >= display_elements:
        continue
      x = margin_x + e * cell
      y = margin_y + r * cell
      cells.append(
        f"<rect x='{x}' y='{y}' width='{cell-1}' height='{cell-1}' fill='{row_color}' fill-opacity='0.9' />"
      )

  axis = []
  for e in range(0, display_elements, 5):
    x = margin_x + e * cell
    axis.append(f"<text x='{x}' y='32' font-size='8' fill='#94a3b8'>{e}</text>")
  for r in range(0, display_sets, 5):
    y = margin_y + r * cell + 7
    axis.append(f"<text x='8' y='{y}' font-size='8' fill='#94a3b8'>{r}</text>")

  status = "Valid exact cover" if viz.get("valid") else viz.get("msg", "Invalid")
  footer = []
  if universe_size > display_elements or len(sets) > display_sets:
    footer.append(
      f"Showing first {display_sets} sets × first {display_elements} elements "
      f"(full instance: {len(sets)} × {universe_size})."
    )

  svg = "\n".join([
    "<div style='margin:12px 0;padding:10px;border:1px solid #1f2937;border-radius:8px;background:#0b1120;'>",
    f"<div style='color:#e2e8f0;font-size:13px;margin-bottom:6px;'><strong>Set-Element Matrix</strong> — {status}</div>",
    f"<div style='color:#94a3b8;font-size:12px;margin-bottom:6px;'>"
    f"Universe: {universe_size} | Sets: {len(sets)} | Expected picks: {len(expected)} | "
    f"Chosen: {len(chosen)} | Missing: {len(missing)} | Extras: {len(extras)}</div>",
    f"<svg width='100%' viewBox='0 0 {width} {height}' style='background:#0b1120;border:1px solid #334155;border-radius:6px;'>",
    f"<rect x='{margin_x-1}' y='{margin_y-1}' width='{display_elements * cell + 2}' height='{display_sets * cell + 2}' fill='none' stroke='#334155' stroke-width='1' />",
    *cells,
    *axis,
    "</svg>",
    "<div style='color:#64748b;font-size:11px;margin-top:6px;'>"
    "Rows are sets and columns are elements. Green rows are correctly chosen planted sets, "
    "red rows are missing planted sets, amber rows are incorrect extra selections." +
    (f" {' '.join(footer)}" if footer else "") +
    "</div>",
    "</div>",
  ])
  return svg


def setup():
  """Pre-generate and cache all streaming input files for parallel test execution."""
  print(f"  Pre-generating streaming input files for {len(TEST_CASES)} test cases...")
  for subpass in range(len(TEST_CASES)):
    if _should_use_streaming(subpass):
      streaming_input = _get_streaming_input(subpass)
      input_path = streaming_input.generate()
      size_mb = streaming_input.get_size_bytes() / (1024 * 1024)
      print(f"    Subpass {subpass}: {size_mb:.1f} MB cached")
