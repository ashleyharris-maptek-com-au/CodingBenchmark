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
from typing import List, Tuple, Dict, Set, Optional
from pathlib import Path

# Import our native compiler helper
from native_compiler import RustCompiler, CompilationError, ExecutionError

title = "Drillhole Data Validation (Rust)"

# Timeout in seconds (5 minutes)
TIMEOUT_SECONDS = 30

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
    freq = 0.01 * (prop_idx + 1)
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
          error_type = rng.choice(["magnitude", "digit", "sign"])
          if error_type == "magnitude":
            value *= rng.choice([0.1, 10, 100])
          elif error_type == "digit":
            value = value + rng.choice([-1, 1]) * 10**rng.randint(1, 3)
          else:
            value = -abs(value)

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
    "error_rate": 0.04,
    "description": "10 holes, 4 properties, 150 samples"
  },
  # Subpass 2: Medium dataset
  {
    "num_holes": 20,
    "num_properties": 5,
    "samples_per_hole": 20,
    "error_rate": 0.03,
    "description": "20 holes, 5 properties, 400 samples"
  },
  # Subpass 3: Larger dataset
  {
    "num_holes": 35,
    "num_properties": 6,
    "samples_per_hole": 25,
    "error_rate": 0.03,
    "description": "35 holes, 6 properties, 875 samples"
  },
  # Subpass 4: Complex
  {
    "num_holes": 50,
    "num_properties": 8,
    "samples_per_hole": 30,
    "error_rate": 0.02,
    "description": "50 holes, 8 properties, 1500 samples"
  },
  # Subpass 5: Large
  {
    "num_holes": 75,
    "num_properties": 10,
    "samples_per_hole": 40,
    "error_rate": 0.02,
    "description": "75 holes, 10 properties, 3000 samples"
  },
  # Extreme cases
  {
    "num_holes": 100,
    "num_properties": 12,
    "samples_per_hole": 50,
    "error_rate": 0.015,
    "description": "100 holes, 12 properties, 5000 samples"
  },
  {
    "num_holes": 200,
    "num_properties": 12,
    "samples_per_hole": 60,
    "error_rate": 0.01,
    "description": "200 holes, 12 properties, 12000 samples"
  },
  {
    "num_holes": 500,
    "num_properties": 12,
    "samples_per_hole": 80,
    "error_rate": 0.008,
    "description": "500 holes, 12 properties, 40000 samples"
  },
  {
    "num_holes": 1000,
    "num_properties": 12,
    "samples_per_hole": 100,
    "error_rate": 0.005,
    "description": "1000 holes, 12 properties, 100000 samples"
  },
  {
    "num_holes": 2000,
    "num_properties": 12,
    "samples_per_hole": 150,
    "error_rate": 0.003,
    "description": "2000 holes, 12 properties, 300000 samples"
  },
]

# Pre-generate test data
TEST_DATA_CACHE = {}


def get_test_data(subpass: int):
  """Get or generate test data for subpass."""
  if subpass not in TEST_DATA_CACHE:
    case = TEST_CASES[subpass]
    holes, props, errors = generate_drillhole_data(case["num_holes"], case["num_properties"],
                                                   case["samples_per_hole"], case["error_rate"],
                                                   RANDOM_SEED + subpass)
    TEST_DATA_CACHE[subpass] = (holes, props, errors)
  return TEST_DATA_CACHE[subpass]


def prepareSubpassPrompt(subPass: int) -> str:
  """Generate the prompt for subpass 0 that handles all data validation complexities."""
  if subPass != 0:
    raise StopIteration

  return f"""You are writing a Rust program to validate drillhole assay data.

You must write a Rust solver that can handle ANY problem size from trivial to ludicrous scale:
- **Trivial**: Small datasets (10-20 holes, 10-50 samples each), basic validation
- **Medium**: Medium datasets (50-200 holes, 50-100 samples each), moderate complexity
- **Large**: Large datasets (500-1000 holes, 100-150 samples each), complex patterns
- **Extreme**: Massive datasets (1500-2000 holes, 150 samples each), very complex statistical analysis

**The Challenge:**
Your Rust program will be tested with drillhole datasets ranging from small exploration projects to massive mining surveys. The same program must work efficiently across ALL scales.

**Problem:**
Given drillhole data from a mining exploration project, identify samples that likely contain typos or measurement errors. These outliers don't correlate well with spatially neighboring data points.

**Drillhole Data:**
- Each hole has a 3D path: start point (x,y,z) + direction vector + length
- Samples are taken at intervals along each hole
- Each sample has measurements for multiple properties

**Input format (stdin):**
```
N num_properties
property1 property2 ...
hole_id start_x start_y start_z dir_x dir_y dir_z length num_samples
depth prop1 prop2 ...
...
```

**Output format (stdout):**
```
M                                    (number of suspect entries found)
hole_id sample_idx property confidence   (for each suspect, sorted by confidence desc)
```

**Critical Requirements:**
1. **Scalability**: Your algorithm must adapt based on number of holes and samples
2. **Performance**: Must complete within 5 minutes even for massive datasets
3. **Quality**: Identify genuine outliers while minimizing false positives

**Algorithm Strategy Recommendations:**
- **Small problems (≤1000 samples)**: Can use exact statistical methods, thorough analysis
- **Medium problems (1000-10000 samples)**: Efficient spatial indexing, statistical tests
- **Large problems (10000-50000 samples)**: Fast heuristics, approximate methods
- **Very Large problems (>50000 samples)**: Very fast heuristics, sampling methods

**Key Techniques:**
- **K-nearest neighbors**: For each sample point, find K nearest neighbors (from other holes)
- **Statistical comparison**: Compare property values to neighbors' values
- **Z-score calculation**: Calculate deviation from expected value
- **Spatial indexing**: Efficient neighbor search for large datasets
- **Correlation analysis**: Consider property correlations with depth/geology

**Implementation Hints:**
- Detect dataset size and choose appropriate algorithm
- Use efficient spatial data structures (KD-trees, grid indexing)
- Implement adaptive quality vs speed tradeoffs
- For very large datasets, consider sampling or approximation
- Handle edge cases: sparse data, missing values, different property scales
- Use fast I/O for large inputs

**Example approach:**
- Build spatial index of all sample points
- For each (hole, sample, property), compute local statistics
- Score = |value - local_mean| / local_stddev
- Report entries with score > 3.0 (3 sigma outliers)

**Requirements:**
1. Program must compile with rustc (edition 2021)
2. Read from stdin, write to stdout
3. Handle up to 2000 holes with 150 samples each
4. Complete within 5 minutes
5. Report at least the most obvious outliers
6. Confidence score should reflect how anomalous the value is
7. Must handle varying dataset sizes efficiently

Write complete, compilable Rust code with a main function.
Include adaptive logic that chooses different strategies based on dataset scale.
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
    return 0.0, "No result provided"

  if "rust_code" not in result:
    return 0.0, "No Rust code provided"

  case = TEST_CASES[subPass]
  holes, property_names, injected_errors = get_test_data(subPass)
  description = case["description"]

  code = result["rust_code"]
  input_data = format_input(holes, property_names)

  # Execute solver
  stdout, error, exec_time, success = execute_rust_solver(code, input_data, aiEngineName)

  if not success:
    return 0.0, f"[{description}] {error}"

  # Parse output
  suspects, parse_error = parse_output(stdout)
  if parse_error and not suspects:
    return 0.0, f"[{description}] {parse_error}"

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

  return score, explanation


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
