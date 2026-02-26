"""
Test 20: Drillhole Data Validation (Rust Implementation)

The LLM must write Rust code that identifies suspect data entries in drillhole assay data.
Given N drillholes with spatial paths and assay measurements, find entries that don't
correlate well with neighboring drillhole data (likely typos or measurement errors).

Each drillhole:
- Has a 3D path (start point, direction vector, length)
- Contains assay data: measurements at intervals along the hole
- Measurements include properties like: radioactivity, density, conductivity, etc.

Goal: Identify the most suspect data entries based on spatial correlation analysis.

Input format (stdin):
Line 1: N num_properties
Line 2: property_names (space-separated)
Lines 3 to N+2: hole_id start_x start_y start_z dir_x dir_y dir_z length num_samples
  Following num_samples lines: depth prop1 prop2 ... propN

Output format (stdout):
Line 1: M (number of suspect entries)
Lines 2 to M+1: hole_id sample_index property_name confidence_score
  (sorted by confidence score descending)

Subpasses test increasingly complex datasets.
Solver times out after 5 minutes.
"""

import random
import subprocess
import sys
import os
import time
import math
import json
from typing import List, Tuple, Dict, Set, Optional, Any
from pathlib import Path

# Import our native compiler helper
from native_compiler import RustCompiler, compile_and_run, describe_this_pc
from solver_utils import StreamingInputFile

title = "Drillhole Data Validation (Rust)"

tags = [
  "rust",
  "structured response",
  "algorithm design",
]

# Timeout in seconds (5 minutes)
TIMEOUT_SECONDS = 300

# Seed for reproducibility
RANDOM_SEED = 20202020


class DrillHole:
  """Represents a drillhole with path and assay data."""

  def __init__(self, hole_id: int, start: Tuple[float, float, float],
               direction: Tuple[float, float, float], length: float):
    self.hole_id = hole_id
    self.start = start
    # Normalize direction
    mag = math.sqrt(sum(d * d for d in direction))
    self.direction = tuple(d / mag for d in direction) if mag > 0 else (0, 0, -1)
    self.length = length
    self.samples: List[Dict[str, float]] = []  # List of {depth, prop1, prop2, ...}

  def point_at_depth(self, depth: float) -> Tuple[float, float, float]:
    """Get 3D point at given depth along hole."""
    return (self.start[0] + self.direction[0] * depth, self.start[1] + self.direction[1] * depth,
            self.start[2] + self.direction[2] * depth)

  def add_sample(self, depth: float, properties: Dict[str, float]):
    """Add a sample at given depth."""
    sample = {"depth": depth}
    sample.update(properties)
    self.samples.append(sample)


def distance_3d(p1: Tuple[float, float, float], p2: Tuple[float, float, float]) -> float:
  """Calculate 3D distance."""
  return math.sqrt(sum((a - b)**2 for a, b in zip(p1, p2)))


def generate_drillhole_data(
    num_holes: int, num_properties: int, samples_per_hole: int, error_rate: float,
    seed: int) -> Tuple[List[DrillHole], List[str], List[Tuple[int, int, str]]]:
  """
    Generate drillhole dataset with some intentional errors.
    
    Returns:
        Tuple of (holes, property_names, injected_errors)
        injected_errors: List of (hole_id, sample_idx, property_name) for ground truth
    """
  rng = random.Random(seed)

  # Property names
  all_properties = [
    "radioactivity", "density", "conductivity", "porosity", "magnetic_susceptibility", "albedo",
    "hardness", "moisture", "copper_ppm", "gold_ppb", "iron_percent", "sulfur_percent"
  ]
  property_names = all_properties[:num_properties]

  holes = []
  injected_errors = []

  # Generate base geological model (smooth spatial variation)
  # Using simple gradient + noise model
  def base_value(prop_idx: int, x: float, y: float, z: float) -> float:
    # Each property has different spatial correlation
    freq = 0.001 * (prop_idx + 1)
    base = 50 + 30 * math.sin(freq * x) * math.cos(freq * y) + 20 * math.sin(freq * z * 0.5)
    return max(0, base + rng.gauss(0, 5))

  # Generate holes in a grid-like pattern with some randomness
  grid_size = int(math.ceil(math.sqrt(num_holes)))
  spacing = 100.0

  for i in range(num_holes):
    gx = i % grid_size
    gy = i // grid_size

    # Start position with jitter
    start_x = gx * spacing + rng.uniform(-20, 20)
    start_y = gy * spacing + rng.uniform(-20, 20)
    start_z = rng.uniform(0, 50)  # Surface elevation varies

    # Direction (mostly down, but can be angled)
    dir_x = rng.uniform(-0.3, 0.3)
    dir_y = rng.uniform(-0.3, 0.3)
    dir_z = -1.0 + rng.uniform(-0.2, 0)  # Mostly downward

    length = rng.uniform(100, 300)

    hole = DrillHole(i, (start_x, start_y, start_z), (dir_x, dir_y, dir_z), length)

    # Generate samples along the hole
    sample_interval = length / samples_per_hole
    for j in range(samples_per_hole):
      depth = j * sample_interval + rng.uniform(0, sample_interval * 0.5)
      if depth > length:
        depth = length - 0.1

      point = hole.point_at_depth(depth)

      # Generate property values based on spatial position
      props = {}
      for k, prop_name in enumerate(property_names):
        value = base_value(k, point[0], point[1], point[2])

        # Inject errors randomly
        if rng.random() < error_rate:
          # Typo: off by order of magnitude or wrong sign
          original_value = value
          error_type = rng.choice(["magnitude", "digit", "sign"])
          if error_type == "magnitude":
            value *= rng.choice([0.1, 10, 100])
          elif error_type == "digit":
            value = value + rng.choice([-1, 1]) * 10**rng.randint(1, 3)
          else:
            value = -abs(value)

          if abs(original_value - value) > abs(original_value * 0.1):
            injected_errors.append((i, j, prop_name))

        props[prop_name] = round(value, 2)

      hole.add_sample(depth, props)

    holes.append(hole)

  return holes, property_names, injected_errors


def format_input(holes: List[DrillHole], property_names: List[str]) -> str:
  """Format drillhole data as input string."""
  lines = [f"{len(holes)} {len(property_names)}"]
  lines.append(" ".join(property_names))

  for hole in holes:
    start = hole.start
    dir = hole.direction
    lines.append(f"{hole.hole_id} {start[0]:.2f} {start[1]:.2f} {start[2]:.2f} "
                 f"{dir[0]:.4f} {dir[1]:.4f} {dir[2]:.4f} {hole.length:.2f} {len(hole.samples)}")
    for sample in hole.samples:
      values = [f"{sample['depth']:.2f}"]
      for prop in property_names:
        values.append(f"{sample[prop]:.2f}")
      lines.append(" ".join(values))

  return "\n".join(lines)


# Test configurations
TEST_CASES = [
  # Subpass 0: Simple - few holes
  {
    "num_holes": 5,
    "num_properties": 3,
    "samples_per_hole": 10,
    "error_rate": 0.05,
    "description": "5 holes, 3 properties, 50 samples"
  },
  # Subpass 1: More holes
  {
    "num_holes": 10,
    "num_properties": 4,
    "samples_per_hole": 15,
    "error_rate": 0.004,
    "description": "10 holes, 4 properties, 150 samples"
  },
  # Subpass 2: Medium dataset
  {
    "num_holes": 20,
    "num_properties": 5,
    "samples_per_hole": 20,
    "error_rate": 0.003,
    "description": "20 holes, 5 properties, 400 samples"
  },
  # Subpass 3: Larger dataset
  {
    "num_holes": 35,
    "num_properties": 6,
    "samples_per_hole": 25,
    "error_rate": 0.003,
    "description": "35 holes, 6 properties, 875 samples"
  },
  # Subpass 4: Complex
  {
    "num_holes": 50,
    "num_properties": 8,
    "samples_per_hole": 30,
    "error_rate": 0.002,
    "description": "50 holes, 8 properties, 1500 samples"
  },
  # Subpass 5: Large
  {
    "num_holes": 75,
    "num_properties": 10,
    "samples_per_hole": 40,
    "error_rate": 0.002,
    "description": "75 holes, 10 properties, 3000 samples"
  },
  # Extreme cases
  {
    "num_holes": 100,
    "num_properties": 12,
    "samples_per_hole": 50,
    "error_rate": 0.0015,
    "description": "100 holes, 12 properties, 5000 samples"
  },
  {
    "num_holes": 200,
    "num_properties": 12,
    "samples_per_hole": 60,
    "error_rate": 0.001,
    "description": "200 holes, 12 properties, 12000 samples"
  },
  {
    "num_holes": 500,
    "num_properties": 12,
    "samples_per_hole": 80,
    "error_rate": 0.0008,
    "description": "500 holes, 12 properties, 40000 samples"
  },
  {
    "num_holes": 1000,
    "num_properties": 12,
    "samples_per_hole": 100,
    "error_rate": 0.0005,
    "description": "1000 holes, 12 properties, 100000 samples"
  },
  {
    "num_holes": 2000,
    "num_properties": 12,
    "samples_per_hole": 150,
    "error_rate": 0.0003,
    "description": "2000 holes, 12 properties, 300000 samples"
  },
  # Ludicrous cases for streaming
  {
    "num_holes": 5000,
    "num_properties": 12,
    "samples_per_hole": 200,
    "error_rate": 0.0002,
    "description": "5K holes, 12 props, 1M samples"
  },
  {
    "num_holes": 10000,
    "num_properties": 12,
    "samples_per_hole": 300,
    "error_rate": 0.0001,
    "description": "10K holes, 12 props, 3M samples"
  },
  {
    "num_holes": 20000,
    "num_properties": 12,
    "samples_per_hole": 400,
    "error_rate": 0.00008,
    "description": "20K holes, 12 props, 8M samples (~1GB)"
  },
]

# Pre-generate test data
TEST_DATA_CACHE: Dict[int, Any] = {}
_INPUT_FILE_CACHE: Dict[int, StreamingInputFile] = {}
STREAMING_THRESHOLD_SAMPLES = 500_000


def get_test_data(subpass: int):
  """Get or generate test data for subpass."""
  if subpass not in TEST_DATA_CACHE:
    case = TEST_CASES[subpass]
    holes, props, errors = generate_drillhole_data(case["num_holes"], case["num_properties"],
                                                   case["samples_per_hole"], case["error_rate"],
                                                   RANDOM_SEED + subpass)
    TEST_DATA_CACHE[subpass] = (holes, props, errors)
  return TEST_DATA_CACHE[subpass]


def _estimate_samples(subpass: int) -> int:
  case = TEST_CASES[subpass]
  return case["num_holes"] * case["samples_per_hole"]


def _should_use_streaming(subpass: int) -> bool:
  return _estimate_samples(subpass) > STREAMING_THRESHOLD_SAMPLES


def _get_streaming_input(subpass: int) -> StreamingInputFile:
  if subpass in _INPUT_FILE_CACHE:
    return _INPUT_FILE_CACHE[subpass]

  case = TEST_CASES[subpass]
  cache_key = f"drillhole20|h={case['num_holes']}|p={case['num_properties']}|s={case['samples_per_hole']}|seed={RANDOM_SEED + subpass}"

  def generator():
    holes, property_names, _ = get_test_data(subpass)
    yield f"{len(holes)} {len(property_names)}\n"
    yield " ".join(property_names) + "\n"
    for hole in holes:
      start = hole.start
      dir = hole.direction
      yield f"{hole.hole_id} {start[0]:.2f} {start[1]:.2f} {start[2]:.2f} {dir[0]:.4f} {dir[1]:.4f} {dir[2]:.4f} {hole.length:.2f} {len(hole.samples)}\n"
      for sample in hole.samples:
        values = [f"{sample['depth']:.2f}"]
        for prop in property_names:
          values.append(f"{sample[prop]:.2f}")
        yield " ".join(values) + "\n"

  input_file = StreamingInputFile(cache_key, generator, "test20_drillholes")
  _INPUT_FILE_CACHE[subpass] = input_file
  return input_file


def prepareSubpassPrompt(subPass: int) -> str:
  """Generate the prompt for subpass 0 that handles all data validation complexities."""
  if subPass != 0:
    raise StopIteration

  return f"""You are writing a Rust program to validate drillhole assay data.


**The Challenge:**
Your Rust program will be tested with drillhole datasets ranging from small exploration projects to 
massive mining surveys. The same program must work efficiently across ALL scales from 10 holes to
200,000 holes.

**Problem:**
Given drillhole data from a mining exploration project, identify samples that likely contain typos or 
measurement errors. These outliers don't correlate well with spatially neighboring data points.

**Target environment specs**
{describe_this_pc()}

**Rust Compiler information**
{RustCompiler("test_engine").describe()}

**Drillhole Data:**
- Each hole has a 3D path: start point (x,y,z) + direction vector + length
- Samples are taken at intervals along each hole
- Each sample has measurements for multiple properties

**Input format (stdin):**
```
num_holes num_properties
property_name_1 property_name_2 ...
hole_id start_x start_y start_z dir_x dir_y dir_z length num_samples
depth prop1 prop2 ...
depth prop1 prop2 ...
etc.
hole_id start_x start_y start_z dir_x dir_y dir_z length num_samples
```

**Output format (stdout):**
```
M                                             (number of suspect entries found)
hole_id sample_idx property_name confidence   (for each suspect, sorted by confidence desc)
```

**Requirements:**
1. Program must compile with rustc and only the standard rust library.
2. Read from stdin, write to stdout
3. Handle up to thousands of holes with thousands of samples each
4. Complete within 5 minutes
5. Report at least the most obvious outliers
6. Confidence score should reflect how anomalous the value is
7. Must handle varying dataset sizes efficiently

Write complete, compilable Rust code with a main function.
Include adaptive logic that chooses different strategies based on dataset scale.

The program should be robust enough to handle edge cases like missing data, duplicate entries, and varying data distributions.
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
      "Explain your algorithm choice and how it adapts to different dataset complexities"
    },
    "rust_code": {
      "type": "string",
      "description": "Complete Rust code with main function that handles all scales"
    }
  },
  "required": ["reasoning", "rust_code"],
  "additionalProperties": False
}


def execute_rust_solver(code: str,
                        input_data: str,
                        engine_name: str,
                        timeout: float = TIMEOUT_SECONDS) -> Tuple[str, str, float, bool]:
  """
    Compile and execute Rust solver.
    
    Returns:
        Tuple of (stdout, error_message, execution_time, success)
    """
  compiler = RustCompiler(engine_name)

  # Check if compiler is available
  if not compiler.find_compiler():
    return "", "No Rust compiler (rustc) found", 0, False

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


def parse_output(output: str) -> Tuple[List[Tuple[int, int, str, float]], str]:
  """
    Parse solver output.
    
    Returns:
        Tuple of (suspect_entries, error_message)
        suspect_entries: List of (hole_id, sample_idx, property, confidence)
    """
  lines = output.strip().split('\n')
  if not lines:
    return [], "Empty output"

  try:
    m = int(lines[0].strip())
    if m < 0:
      return [], f"Invalid suspect count: {m}"

    suspects = []
    for i in range(1, min(m + 1, len(lines))):
      parts = lines[i].strip().split()
      if len(parts) >= 4:
        hole_id = int(parts[0])
        sample_idx = int(parts[1])
        prop_name = parts[2]
        confidence = float(parts[3])
        suspects.append((hole_id, sample_idx, prop_name, confidence))

    return suspects, ""

  except ValueError as e:
    return [], f"Parse error: {str(e)}"


def gradeAnswer(result: dict, subPass: int, aiEngineName: str) -> tuple:
  """
    Grade the Rust drillhole validator.
    
    Scoring based on:
    - Precision: What fraction of reported suspects are actual errors
    - Recall: What fraction of actual errors were found
    - F1 score combining both
    """
  if not result:
    return 0.0, "No result provided", "No result provided"

  if "rust_code" not in result:
    return 0.0, "No Rust code provided", "No result provided"

  case = TEST_CASES[subPass]
  holes, property_names, injected_errors = get_test_data(subPass)
  description = case["description"]
  use_streaming = _should_use_streaming(subPass)
  code = result["rust_code"]

  # Get input (streaming file or in-memory string)
  if use_streaming:
    t = time.time()
    streaming_input = _get_streaming_input(subPass)
    print(f"  Generating/caching input file for {description}...")
    input_file_path = streaming_input.generate()
    file_size_mb = streaming_input.get_size_bytes() / (1024 * 1024)
    print(f"  Input file: {file_size_mb:.1f} MB")
    if time.time() - t > 1:
      print(f"  Time to generate: {time.time() - t:.2f}s")

    run_result = compile_and_run(code,
                                 'rust',
                                 aiEngineName,
                                 input_file=input_file_path,
                                 timeout=TIMEOUT_SECONDS)
  else:
    input_data = format_input(holes, property_names)
    run_result = compile_and_run(code,
                                 'rust',
                                 aiEngineName,
                                 input_data=input_data,
                                 timeout=TIMEOUT_SECONDS)

  if not run_result.success:
    error_msg = run_result.error_message(200)
    if run_result.error_stage == 'compilation':
      return 0.0, f"Compilation error: {error_msg}", f"<pre>{error_msg}</pre>"
    elif run_result.error_stage == 'compiler_missing':
      return 0.0, error_msg, "No Rust compiler found"
    else:
      return 0.0, f"[{description}] {error_msg}", f"<pre>{error_msg}</pre>"

  stdout = run_result.stdout
  exec_time = run_result.exec_time

  # Parse output
  t = time.time()
  suspects, parse_error = parse_output(stdout)
  parseTime = time.time() - t
  if parseTime > 1:
    print(f"  Parse time: {parseTime:.2f}s")
    
  if parse_error and not suspects:
    return 0.0, f"[{description}] {parse_error}", parse_error

  # Convert injected errors to set for comparison
  error_set = set(injected_errors)

  # Convert suspects to comparable format
  suspect_set = set((s[0], s[1], s[2]) for s in suspects)

  # Calculate precision and recall
  true_positives = len(error_set & suspect_set)
  false_positives = len(suspect_set - error_set)
  false_negatives = len(error_set - suspect_set)

  precision = true_positives / (true_positives + false_positives) if (true_positives +
                                                                      false_positives) > 0 else 0
  recall = true_positives / (true_positives + false_negatives) if (true_positives +
                                                                   false_negatives) > 0 else 0
  f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

  # Score based on F1
  if f1 >= 0.8:
    score = 1.0
  elif f1 >= 0.5:
    score = 0.6 + 0.4 * (f1 - 0.5) / 0.3
  elif f1 >= 0.2:
    score = 0.3 + 0.3 * (f1 - 0.2) / 0.3
  else:
    score = f1 * 1.5  # Some credit for trying

  explanation = (f"[{description}] Found: {len(suspects)}, "
                 f"Actual errors: {len(injected_errors)}, "
                 f"TP: {true_positives}, FP: {false_positives}, FN: {false_negatives}, "
                 f"P: {precision:.2f}, R: {recall:.2f}, F1: {f1:.2f}, "
                 f"Time: {exec_time:.2f}s")

  html = ""

  if subPass == 0:
    html = output_example_html(score, explanation, result, subPass)

  # Rich visualization
  html += _generate_drillhole_visualization(
    subPass, holes, property_names, injected_errors,
    suspects, true_positives, false_positives, false_negatives,
    precision, recall, f1, score, exec_time, description
  )

  return score, explanation, html


def _generate_drillhole_visualization(
    subPass, holes, property_names, injected_errors,
    suspects, true_positives, false_positives, false_negatives,
    precision, recall, f1, score, exec_time, description):
  """Generate rich HTML visualization for drillhole validation results."""
  import json as _json

  num_holes = len(holes)
  error_set = set(injected_errors)
  suspect_set = set((s[0], s[1], s[2]) for s in suspects)
  tp_set = error_set & suspect_set
  fn_set = error_set - suspect_set
  fp_set = suspect_set - error_set

  score_color = "lime" if score >= 0.8 else "orange" if score >= 0.4 else "red"
  viz_id = f"viz20_{subPass}"

  # Stats bar (always shown)
  stats_html = f"""
  <div style="display:flex;gap:16px;flex-wrap:wrap;align-items:center;margin:8px 0;font-family:monospace;font-size:13px;color:#ccc;">
    <span style="color:{score_color};font-weight:bold;font-size:15px;">{score:.2f}</span>
    <span>({description})</span>
    <span>Found: {len(suspects)}</span>
    <span>Actual: {len(injected_errors)}</span>
    <span style="color:#4f4;">TP: {true_positives}</span>
    <span style="color:#f84;">FP: {false_positives}</span>
    <span style="color:#f44;">FN: {false_negatives}</span>
    <span>P: {precision:.2f}</span>
    <span>R: {recall:.2f}</span>
    <span>F1: {f1:.2f}</span>
    <span>Time: {exec_time:.2f}s</span>
  </div>"""

  # For large subpasses (>=100 holes), show collapsed error table
  if num_holes >= 100:
    rows = []
    # Show up to 50 errors
    all_errors = []
    for e in sorted(tp_set):
      all_errors.append((*e, "TP", "#4f4"))
    for e in sorted(fn_set):
      all_errors.append((*e, "Missed", "#f44"))
    for e in sorted(fp_set):
      all_errors.append((*e, "FP", "#f84"))
    for hid, sid, prop, label, color in all_errors[:50]:
      rows.append(f'<tr><td>{hid}</td> <td>{sid}</td> <td>{prop}</td>'
                   f'<td style="color:{color}">{label}</td></tr>')
    if len(all_errors) > 50:
      rows.append(f'<tr><td colspan="4">... and {len(all_errors)-50} more</td></tr>')

    table_html = ""
    if rows:
      table_html = f"""<details style="margin:4px 0;">
        <summary style="cursor:pointer;color:#aaa;">Error details ({len(all_errors)} entries)</summary>
        <table style="border-collapse:collapse;font-size:12px;font-family:monospace;margin:4px 0;">
          <tr style="color:#888;"><th style="padding:2px 8px;text-align:left;">Hole</th>
          <th style="padding:2px 8px;text-align:left;">Sample</th>
          <th style="padding:2px 8px;text-align:left;">Property</th>
          <th style="padding:2px 8px;text-align:left;">Status</th></tr>
          {''.join(rows)}
        </table>
      </details>"""

    return stats_html + table_html

  # ---- 3D visualization for small subpasses ----

  # Build hole data for JS: each hole = {start, end, segments: [{p1, p2, values: {prop: val}}], errors: [...]}
  holes_json = []
  for hole in holes:
    start = hole.start
    end_pt = hole.point_at_depth(hole.length)
    segments = []
    for j, sample in enumerate(hole.samples):
      p = hole.point_at_depth(sample["depth"])
      # next point: midpoint to next sample, or end of hole
      if j + 1 < len(hole.samples):
        next_depth = hole.samples[j + 1]["depth"]
      else:
        next_depth = hole.length
      p2 = hole.point_at_depth((sample["depth"] + next_depth) / 2.0)
      vals = {}
      for prop in property_names:
        vals[prop] = sample[prop]
      # Error classification for this sample
      err_labels = []
      for prop in property_names:
        key = (hole.hole_id, j, prop)
        if key in tp_set:
          err_labels.append({"prop": prop, "type": "tp"})
        elif key in fn_set:
          err_labels.append({"prop": prop, "type": "fn"})
        elif key in fp_set:
          err_labels.append({"prop": prop, "type": "fp"})
      segments.append({
        "p1": [round(p[0], 2), round(p[1], 2), round(p[2], 2)],
        "p2": [round(p2[0], 2), round(p2[1], 2), round(p2[2], 2)],
        "vals": vals,
        "errs": err_labels
      })
    holes_json.append({
      "id": hole.hole_id,
      "start": [round(start[0], 2), round(start[1], 2), round(start[2], 2)],
      "end": [round(end_pt[0], 2), round(end_pt[1], 2), round(end_pt[2], 2)],
      "segs": segments
    })

  # Compute global min/max per property for color mapping
  prop_ranges = {}
  for prop in property_names:
    vals = []
    for hole in holes:
      for s in hole.samples:
        vals.append(s[prop])
    if vals:
      p5 = sorted(vals)[max(0, int(len(vals) * 0.02))]
      p95 = sorted(vals)[min(len(vals) - 1, int(len(vals) * 0.98))]
      prop_ranges[prop] = [round(p5, 2), round(p95, 2)]
    else:
      prop_ranges[prop] = [0, 1]

  data_json = _json.dumps(holes_json)
  props_json = _json.dumps(property_names)
  ranges_json = _json.dumps(prop_ranges)

  return stats_html + f"""
  <div id="{viz_id}_wrap" style="position:relative;margin:4px 0;">
    <div style="display:flex;gap:4px;flex-wrap:wrap;margin-bottom:4px;" id="{viz_id}_btns"></div>
    <div style="display:flex;gap:8px;">
      <div id="{viz_id}" style="width:700px;height:450px;background:#1a1a2e;border-radius:4px;"></div>
      <div id="{viz_id}_legend" style="width:160px;font-size:11px;font-family:monospace;color:#ccc;"></div>
    </div>
  </div>
  <script>
  (function() {{
    var container = document.getElementById('{viz_id}');
    if (!container) return;
    function loadThree(cb) {{
      if (window.THREE) {{ cb(); return; }}
      var s = document.createElement('script');
      s.src = 'https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js';
      s.onload = function() {{
        var s2 = document.createElement('script');
        s2.src = 'https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js';
        s2.onload = cb;
        document.head.appendChild(s2);
      }};
      document.head.appendChild(s);
    }}
    loadThree(function() {{
      var THREE = window.THREE;
      var holesData = {data_json};
      var propNames = {props_json};
      var propRanges = {ranges_json};
      var currentProp = propNames[0];

      var W = 700, H = 450;
      var scene = new THREE.Scene();
      scene.background = new THREE.Color(0x1a1a2e);
      var camera = new THREE.PerspectiveCamera(50, W / H, 0.1, 50000);
      var renderer = new THREE.WebGLRenderer({{ antialias: true }});
      renderer.setSize(W, H);
      container.appendChild(renderer.domElement);
      var controls = new THREE.OrbitControls(camera, renderer.domElement);
      controls.enableDamping = true;

      // Lighting
      scene.add(new THREE.AmbientLight(0xffffff, 0.6));
      var dl = new THREE.DirectionalLight(0xffffff, 0.8);
      dl.position.set(1, 1, 1);
      scene.add(dl);

      // Compute bounds
      var minX=1e9, minY=1e9, minZ=1e9, maxX=-1e9, maxY=-1e9, maxZ=-1e9;
      holesData.forEach(function(h) {{
        [h.start, h.end].forEach(function(p) {{
          if(p[0]<minX)minX=p[0]; if(p[0]>maxX)maxX=p[0];
          if(p[1]<minY)minY=p[1]; if(p[1]>maxY)maxY=p[1];
          if(p[2]<minZ)minZ=p[2]; if(p[2]>maxZ)maxZ=p[2];
        }});
      }});
      var cx=(minX+maxX)/2, cy=(minY+maxY)/2, cz=(minZ+maxZ)/2;
      var span = Math.max(maxX-minX, maxY-minY, maxZ-minZ, 1);
      controls.target.set(cx, cz, -cy);
      camera.position.set(cx + span*0.8, cz + span*0.6, -cy + span*0.8);

      // Color ramp: blue -> cyan -> green -> yellow -> red
      function valToColor(v, lo, hi) {{
        var t = (hi > lo) ? Math.max(0, Math.min(1, (v - lo) / (hi - lo))) : 0.5;
        var r, g, b;
        if (t < 0.25) {{ r=0; g=t*4; b=1; }}
        else if (t < 0.5) {{ r=0; g=1; b=1-(t-0.25)*4; }}
        else if (t < 0.75) {{ r=(t-0.5)*4; g=1; b=0; }}
        else {{ r=1; g=1-(t-0.75)*4; b=0; }}
        return new THREE.Color(r, g, b);
      }}

      // Build hole geometries
      var segmentLines = []; // [{{line, hole, segIdx}}]
      var errorMarkers = []; // [{{mesh, hole, segIdx, errType, prop}}]

      holesData.forEach(function(h) {{
        // Hole collar marker (small sphere at start)
        var cg = new THREE.SphereGeometry(span * 0.005, 8, 8);
        var cm = new THREE.Mesh(cg, new THREE.MeshPhongMaterial({{ color: 0xffffff }}));
        cm.position.set(h.start[0], h.start[2], -h.start[1]);
        scene.add(cm);

        h.segs.forEach(function(seg, si) {{
          // Segment line (thick via cylinder)
          var p1 = new THREE.Vector3(seg.p1[0], seg.p1[2], -seg.p1[1]);
          var p2 = new THREE.Vector3(seg.p2[0], seg.p2[2], -seg.p2[1]);
          var dir = new THREE.Vector3().subVectors(p2, p1);
          var len = dir.length();
          if (len < 0.001) return;
          var mid = new THREE.Vector3().addVectors(p1, p2).multiplyScalar(0.5);
          var cylGeo = new THREE.CylinderGeometry(span * 0.003, span * 0.003, len, 6);
          var mat = new THREE.MeshPhongMaterial({{ color: 0x888888 }});
          var cyl = new THREE.Mesh(cylGeo, mat);
          cyl.position.copy(mid);
          // Orient cylinder along segment
          var axis = new THREE.Vector3(0, 1, 0);
          var quat = new THREE.Quaternion().setFromUnitVectors(axis, dir.normalize());
          cyl.quaternion.copy(quat);
          scene.add(cyl);
          segmentLines.push({{ mesh: cyl, hole: h.id, segIdx: si, vals: seg.vals }});

          // Error markers
          seg.errs.forEach(function(err) {{
            var markerSize = span * 0.012;
            var color = err.type === 'tp' ? 0x44ff44 : err.type === 'fn' ? 0xff4444 : 0xff8844;
            var geo = err.type === 'tp'
              ? new THREE.SphereGeometry(markerSize, 8, 8)
              : err.type === 'fn'
                ? new THREE.OctahedronGeometry(markerSize)
                : new THREE.BoxGeometry(markerSize, markerSize, markerSize);
            var mmat = new THREE.MeshPhongMaterial({{
              color: color, transparent: true, opacity: 0.25,
              emissive: color, emissiveIntensity: 0.3
            }});
            var marker = new THREE.Mesh(geo, mmat);
            marker.position.copy(mid);
            marker.visible = true;
            scene.add(marker);
            errorMarkers.push({{ mesh: marker, hole: h.id, segIdx: si, errType: err.type, prop: err.prop }});
          }});
        }});
      }});

      // Update colors based on selected property
      function updateColors(prop) {{
        var range = propRanges[prop] || [0, 100];
        segmentLines.forEach(function(sl) {{
          var v = sl.vals[prop];
          if (v !== undefined) {{
            sl.mesh.material.color.copy(valToColor(v, range[0], range[1]));
            sl.mesh.material.emissive.copy(valToColor(v, range[0], range[1]));
            sl.mesh.material.emissiveIntensity = 0.15;
          }} else {{
            sl.mesh.material.color.set(0x444444);
            sl.mesh.material.emissive.set(0x000000);
          }}
        }});
        // Show/hide error markers: only show markers for current property or "all"
        errorMarkers.forEach(function(em) {{
          em.mesh.visible = (em.prop === prop);
        }});
        updateLegend(prop, range);
      }}

      // Legend
      function updateLegend(prop, range) {{
        var leg = document.getElementById('{viz_id}_legend');
        if (!leg) return;
        var canvas = document.createElement('canvas');
        canvas.width = 20; canvas.height = 200;
        var ctx = canvas.getContext('2d');
        for (var i = 0; i < 200; i++) {{
          var t = 1 - i / 199;
          var r, g, b;
          if (t < 0.25) {{ r=0; g=t*4; b=1; }}
          else if (t < 0.5) {{ r=0; g=1; b=1-(t-0.25)*4; }}
          else if (t < 0.75) {{ r=(t-0.5)*4; g=1; b=0; }}
          else {{ r=1; g=1-(t-0.75)*4; b=0; }}
          ctx.fillStyle = 'rgb('+Math.round(r*255)+','+Math.round(g*255)+','+Math.round(b*255)+')';
          ctx.fillRect(0, i, 20, 1);
        }}
        leg.innerHTML = '<div style="margin-bottom:6px;font-weight:bold;color:#fff;">' + prop + '</div>'
          + '<div style="display:flex;gap:6px;align-items:stretch;">'
          + '<img src="' + canvas.toDataURL() + '" style="width:16px;height:150px;border-radius:2px;">'
          + '<div style="display:flex;flex-direction:column;justify-content:space-between;height:150px;">'
          + '<span>' + range[1].toFixed(1) + '</span>'
          + '<span>' + ((range[0]+range[1])/2).toFixed(1) + '</span>'
          + '<span>' + range[0].toFixed(1) + '</span>'
          + '</div></div>'
          + '<div style="margin-top:10px;">'
          + '<div><span style="color:#4f4;">&#9679;</span> Detected (TP)</div>'
          + '<div><span style="color:#f44;">&#9670;</span> Missed (FN)</div>'
          + '<div><span style="color:#f84;">&#9632;</span> False Pos (FP)</div>'
          + '</div>';
      }}

      // Property buttons
      var btnDiv = document.getElementById('{viz_id}_btns');
      propNames.forEach(function(prop) {{
        var btn = document.createElement('button');
        btn.textContent = prop;
        btn.style.cssText = 'padding:3px 10px;font-size:11px;cursor:pointer;border:1px solid #555;'
          + 'border-radius:3px;background:#333;color:#ccc;font-family:monospace;';
        btn.onclick = function() {{
          currentProp = prop;
          updateColors(prop);
          // Highlight active button
          btnDiv.querySelectorAll('button').forEach(function(b) {{
            b.style.background = b.textContent === prop ? '#556' : '#333';
            b.style.color = b.textContent === prop ? '#fff' : '#ccc';
          }});
        }};
        btnDiv.appendChild(btn);
      }});
      // "All errors" button
      var allBtn = document.createElement('button');
      allBtn.textContent = 'All Errors';
      allBtn.style.cssText = 'padding:3px 10px;font-size:11px;cursor:pointer;border:1px solid #555;'
        + 'border-radius:3px;background:#333;color:#f88;font-family:monospace;';
      allBtn.onclick = function() {{
        errorMarkers.forEach(function(em) {{ em.mesh.visible = true; }});
        btnDiv.querySelectorAll('button').forEach(function(b) {{
          b.style.background = b.textContent === 'All Errors' ? '#543' : '#333';
          b.style.color = b.textContent === 'All Errors' ? '#f88' : '#ccc';
        }});
      }};
      btnDiv.appendChild(allBtn);

      // Initial state
      updateColors(currentProp);
      btnDiv.querySelector('button').style.background = '#556';
      btnDiv.querySelector('button').style.color = '#fff';

      // Animate
      function animate() {{
        requestAnimationFrame(animate);
        controls.update();
        renderer.render(scene, camera);
      }}
      animate();
    }});
  }})();
  </script>"""


def output_example_html(score: float, explanation: str, result: dict, subPass: int) -> str:
  """Generate HTML for result display."""
  case = TEST_CASES[subPass]

  code = result.get("rust_code", "No code provided")
  reasoning = result.get("reasoning", "No reasoning provided")

  # Escape HTML
  code = code.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
  reasoning = reasoning.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

  score_color = "green" if score >= 0.8 else "orange" if score >= 0.4 else "red"

  return f"""
    <div class="result" style="margin: 10px; padding: 10px; border: 1px solid #ccc;">
        <h4>Subpass {subPass}: {case['description']}</h4>
        <p><strong>Score:</strong> <span style="color: {score_color};">{score:.2f}</span></p>
        <p><strong>Details:</strong> {explanation}</p>
        <details>
            <summary>Reasoning</summary>
            <pre style="background: #f5f5f5; padding: 10px; overflow-x: auto;">{reasoning}</pre>
        </details>
        <details>
            <summary>Rust Code</summary>
            <pre style="background: #f0f0f0; padding: 10px; overflow-x: auto;"><code>{code}</code></pre>
        </details>
    </div>
    """


def output_header_html() -> str:
  """Generate HTML header."""
  return """
    <h2>Test 20: Drillhole Data Validation (Rust)</h2>
    <p>Testing Rust implementation of spatial anomaly detection in assay data.</p>
    """


def output_summary_html(results: list) -> str:
  """Generate summary HTML."""
  if not results:
    return "<p>No results</p>"

  total_score = sum(r[0] for r in results)
  max_score = len(results)
  avg_score = total_score / max_score if max_score > 0 else 0

  return f"""
    <div class="summary" style="margin: 10px; padding: 15px; background: #e8f4e8; border-radius: 5px;">
        <h3>Summary</h3>
        <p><strong>Total Score:</strong> {total_score:.2f} / {max_score}</p>
        <p><strong>Average Score:</strong> {avg_score:.2%}</p>
        <p><strong>Subpasses Completed:</strong> {len(results)}</p>
    </div>
    """


def setup():
  """Pre-generate and cache all streaming input files for parallel test execution."""
  print(f"  Pre-generating streaming input files for {len(TEST_CASES)} test cases...")
  for subpass in range(len(TEST_CASES)):
    if _should_use_streaming(subpass):
      streaming_input = _get_streaming_input(subpass)
      input_path = streaming_input.generate()
      size_mb = streaming_input.get_size_bytes() / (1024 * 1024)
      print(f"    Subpass {subpass}: {size_mb:.1f} MB cached")


highLevelSummary = """
<p>Given a set of mining drillholes &mdash; each a 3D path with measurements taken
at intervals along its length &mdash; identify the data entries that are most likely
to be errors (typos, instrument glitches, or transcription mistakes).</p>
<p>The AI must compare each measurement against its spatial neighbours: a reading
that disagrees with nearby drillholes is suspect. Subpasses increase the number
of drillholes and properties, requiring efficient spatial reasoning at scale.</p>
"""
