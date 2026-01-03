"""
Test 26: 3D Point Clustering (Rust Implementation)

The LLM must write Rust code that clusters millions to billions of 3D points
into exactly N clusters, outputting the cluster index for each point.

Points are provided via stdin as binary float32 triples for efficiency.
Cluster assignments are output via stdout as binary uint32 indices.

The algorithm must:
1. Handle streaming input (points may not fit in memory)
2. Produce exactly N clusters
3. Minimize within-cluster variance (k-means objective)
4. Complete within timeout

Subpasses increase difficulty:
- Point counts from 1M to 10B
- Cluster counts from 10 to 10,000
- Memory constraints become critical

Solver times out after 5 minutes.
"""

import random
import subprocess
import sys
import os
import time
import math
import struct
from typing import List, Tuple, Dict, Optional
from pathlib import Path
import tempfile

# Import our native compiler helper
from native_compiler import RustCompiler, CompilationError, ExecutionError

title = "3D Point Clustering (Rust)"

# Timeout in seconds (5 minutes)
TIMEOUT_SECONDS = 30

# Seed for reproducibility
RANDOM_SEED = 26262626


def generate_clustered_points(
    num_points: int, num_clusters: int,
    seed: int) -> Tuple[List[Tuple[float, float, float]], List[Tuple[float, float, float]]]:
  """
    Generate points with natural clustering around centroids.
    Returns (points, true_centroids).
    """
  rng = random.Random(seed)

  # Generate cluster centroids spread across space
  centroids = []
  space_size = 1000.0 * (num_clusters**0.33)  # Scale space with clusters

  for _ in range(num_clusters):
    cx = rng.uniform(-space_size, space_size)
    cy = rng.uniform(-space_size, space_size)
    cz = rng.uniform(-space_size, space_size)
    centroids.append((cx, cy, cz))

  # Generate points around centroids
  points = []
  cluster_std = space_size / (num_clusters**0.5) * 0.3

  for i in range(num_points):
    # Pick a random cluster
    c_idx = rng.randint(0, num_clusters - 1)
    cx, cy, cz = centroids[c_idx]

    # Generate point with Gaussian noise around centroid
    x = cx + rng.gauss(0, cluster_std)
    y = cy + rng.gauss(0, cluster_std)
    z = cz + rng.gauss(0, cluster_std)
    points.append((x, y, z))

  return points, centroids


def write_points_binary(points: List[Tuple[float, float, float]], file_path: str):
  """Write points to binary file as float32 triples."""
  with open(file_path, 'wb') as f:
    for x, y, z in points:
      f.write(struct.pack('<fff', x, y, z))


def read_clusters_binary(file_path: str, num_points: int) -> List[int]:
  """Read cluster assignments from binary file as uint32."""
  clusters = []
  with open(file_path, 'rb') as f:
    for _ in range(num_points):
      data = f.read(4)
      if len(data) < 4:
        break
      clusters.append(struct.unpack('<I', data)[0])
  return clusters


def format_text_input(num_points: int, num_clusters: int, points: List[Tuple[float, float,
                                                                             float]]) -> str:
  """Format as text for smaller test cases."""
  lines = [f"{num_points} {num_clusters}"]
  for x, y, z in points:
    lines.append(f"{x:.6f} {y:.6f} {z:.6f}")
  return "\n".join(lines)


def parse_text_output(output: str, num_points: int) -> List[int]:
  """Parse cluster assignments from text output."""
  clusters = []
  for line in output.strip().split('\n'):
    line = line.strip()
    if line:
      try:
        clusters.append(int(line))
      except:
        pass
  return clusters


def calculate_wcss(points: List[Tuple[float, float, float]], clusters: List[int],
                   num_clusters: int) -> float:
  """Calculate within-cluster sum of squares."""
  if len(clusters) != len(points):
    return float('inf')

  # Calculate centroids
  cluster_sums = [[0.0, 0.0, 0.0] for _ in range(num_clusters)]
  cluster_counts = [0] * num_clusters

  for i, (x, y, z) in enumerate(points):
    c = clusters[i]
    if 0 <= c < num_clusters:
      cluster_sums[c][0] += x
      cluster_sums[c][1] += y
      cluster_sums[c][2] += z
      cluster_counts[c] += 1

  centroids = []
  for i in range(num_clusters):
    if cluster_counts[i] > 0:
      centroids.append(
        (cluster_sums[i][0] / cluster_counts[i], cluster_sums[i][1] / cluster_counts[i],
         cluster_sums[i][2] / cluster_counts[i]))
    else:
      centroids.append((0, 0, 0))

  # Calculate WCSS
  wcss = 0.0
  for i, (x, y, z) in enumerate(points):
    c = clusters[i]
    if 0 <= c < num_clusters:
      cx, cy, cz = centroids[c]
      wcss += (x - cx)**2 + (y - cy)**2 + (z - cz)**2

  return wcss


# Test configurations - from millions to billions
TEST_CASES = [
  # Subpass 0: Small test
  {
    "num_points": 10_000,
    "num_clusters": 10,
    "use_binary": False,
    "description": "10K points, 10 clusters"
  },
  # Subpass 1: Medium
  {
    "num_points": 100_000,
    "num_clusters": 25,
    "use_binary": False,
    "description": "100K points, 25 clusters"
  },
  # Subpass 2: 1 million
  {
    "num_points": 1_000_000,
    "num_clusters": 50,
    "use_binary": True,
    "description": "1M points, 50 clusters (~12MB)"
  },
  # Subpass 3: 5 million
  {
    "num_points": 5_000_000,
    "num_clusters": 100,
    "use_binary": True,
    "description": "5M points, 100 clusters (~60MB)"
  },
  # Subpass 4: 10 million
  {
    "num_points": 10_000_000,
    "num_clusters": 200,
    "use_binary": True,
    "description": "10M points, 200 clusters (~120MB)"
  },
  # Subpass 5: 50 million
  {
    "num_points": 50_000_000,
    "num_clusters": 500,
    "use_binary": True,
    "description": "50M points, 500 clusters (~600MB)"
  },
  # Extreme cases
  {
    "num_points": 100_000_000,
    "num_clusters": 1000,
    "use_binary": True,
    "description": "100M points, 1K clusters (~1.2GB)"
  },
  {
    "num_points": 250_000_000,
    "num_clusters": 2000,
    "use_binary": True,
    "description": "250M points, 2K clusters (~3GB)"
  },
  {
    "num_points": 500_000_000,
    "num_clusters": 5000,
    "use_binary": True,
    "description": "500M points, 5K clusters (~6GB)"
  },
  {
    "num_points": 1_000_000_000,
    "num_clusters": 7500,
    "use_binary": True,
    "description": "1B points, 7.5K clusters (~12GB)"
  },
  {
    "num_points": 2_000_000_000,
    "num_clusters": 10000,
    "use_binary": True,
    "description": "2B points, 10K clusters (~24GB)"
  },
]

# Cache for generated points (only for smaller cases)
POINTS_CACHE = {}


def get_test_data(
  subpass: int
) -> Tuple[List[Tuple[float, float, float]], List[Tuple[float, float, float]], int, int]:
  """Get or generate test data for subpass."""
  case = TEST_CASES[subpass]
  num_points = case["num_points"]
  num_clusters = case["num_clusters"]

  # Only cache smaller cases
  if num_points <= 1_000_000:
    if subpass not in POINTS_CACHE:
      points, centroids = generate_clustered_points(num_points, num_clusters, RANDOM_SEED + subpass)
      POINTS_CACHE[subpass] = (points, centroids)
    return POINTS_CACHE[subpass][0], POINTS_CACHE[subpass][1], num_points, num_clusters
  else:
    # Generate on demand for large cases
    points, centroids = generate_clustered_points(num_points, num_clusters, RANDOM_SEED + subpass)
    return points, centroids, num_points, num_clusters


def prepareSubpassPrompt(subPass: int) -> str:
  """Generate the prompt for subpass 0 that handles all clustering complexities."""
  if subPass != 0:
    raise StopIteration

  return f"""You are writing Rust code to cluster 3D points into exactly N clusters.

You must write a Rust solver that can handle ANY clustering scenario from trivial to ludicrous scale:
- **Trivial**: Small datasets (1M points, 10 clusters), basic k-means, memory-friendly
- **Medium**: Moderate datasets (10M points, 100 clusters), optimized k-means, streaming
- **Large**: Large datasets (100M points, 1000 clusters), mini-batch k-means, memory constraints
- **Extreme**: Massive datasets (1B-10B points, 10000 clusters), streaming algorithms, very tight memory

**The Challenge:**
Your Rust clustering algorithm will be tested with datasets ranging from millions to billions of 3D points. The same algorithm must work efficiently across ALL scales while respecting memory constraints.

**Problem:**
Given millions to billions of 3D points, partition them into exactly N clusters minimizing within-cluster variance (k-means objective). Points are provided as binary float32 triples for efficiency.

**Input format (stdin):**
```
num_points num_clusters
[x y z] (as binary float32 triples, num_points total)
```

**Output format (stdout):**
```
[cluster_index] (as binary uint32, one per point)
```

**Critical Requirements:**
1. **Scalability**: Your algorithm must adapt based on point count and cluster count
2. **Performance**: Must complete within 5 minutes even for massive datasets
3. **Memory**: Must handle streaming input for datasets that don't fit in memory
4. **Quality**: Minimize within-cluster variance while producing exactly N clusters

**Algorithm Strategy Recommendations:**
- **Small datasets (≤10M points)**: Can use standard k-means with full data in memory
- **Medium datasets (10M-100M points)**: Mini-batch k-means, streaming updates
- **Large datasets (100M-1B points)**: Streaming k-means with limited memory buffers
- **Very Large datasets (>1B points)**: Very fast streaming, approximate clustering

**Key Techniques:**
- **Streaming algorithms**: Process points in batches without storing all data
- **Mini-batch k-means**: Update centroids incrementally
- **Memory management**: Use fixed-size buffers for large datasets
- **Binary I/O**: Efficient reading/writing of binary data
- **Convergence detection**: Monitor when clustering stabilizes

**Implementation Hints:**
- Detect dataset size and choose appropriate clustering algorithm
- Use efficient binary I/O (std::io::BufReader/BufWriter)
- Implement streaming updates for large datasets
- For very large datasets, use approximation algorithms
- Handle edge cases: empty clusters, convergence issues
- Use fast random access for centroid updates

**Binary Format Details:**
- Input: Points as little-endian f32 triples (12 bytes per point)
- Output: Cluster indices as little-endian u32 (4 bytes per point)
- Total input size: num_points × 12 bytes
- Total output size: num_points × 4 bytes

**Memory Constraints:**
- Small datasets: Can load all points into memory
- Medium datasets: Use streaming with moderate buffers
- Large datasets: Streaming with minimal memory footprint
- Very Large datasets: Extremely memory-efficient streaming

**Requirements:**
1. Program must compile with rustc (edition 2021)
2. Read binary input from stdin, write binary output to stdout
3. Handle variable point counts and cluster numbers
4. Complete within 5 minutes
5. Must handle varying dataset sizes efficiently
6. Produce exactly N clusters

Write complete, compilable Rust code with a main function.
Include adaptive logic that chooses different strategies based on clustering complexity.
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
      "Explain your algorithm approach and how it adapts to different clustering complexities"
    },
    "rust_code": {
      "type": "string",
      "description": "Complete Rust code with main function that handles all scales"
    }
  },
  "required": ["reasoning", "rust_code"],
  "additionalProperties": False
}


def run_clustering(code: str, case: dict, subpass: int,
                   engine_name: str) -> Tuple[List[int], str, float]:
  """Compile and run clustering algorithm."""
  compiler = RustCompiler(engine_name)

  if not compiler.find_compiler():
    return [], "No Rust compiler found", 0

  try:
    exe_path = compiler.compile(code)
  except CompilationError as e:
    return [], f"Compilation error: {str(e)[:500]}", 0

  num_points = case["num_points"]
  num_clusters = case["num_clusters"]

  # Generate test data
  points, _, _, _ = get_test_data(subpass)

  # Prepare input
  if case["use_binary"]:
    # Use temporary files for binary I/O
    with tempfile.NamedTemporaryFile(mode='wb', delete=False, suffix='.bin') as f:
      input_file = f.name
      # Write header as text first line, then binary
      header = f"{num_points} {num_clusters}\n".encode()
      f.write(header)
      for x, y, z in points:
        f.write(struct.pack('<fff', x, y, z))

    output_file = input_file + '.out'

    start_time = time.time()
    try:
      with open(input_file, 'rb') as fin:
        with open(output_file, 'w') as fout:
          result = subprocess.run([str(exe_path)],
                                  stdin=fin,
                                  stdout=fout,
                                  stderr=subprocess.PIPE,
                                  timeout=TIMEOUT_SECONDS)
      exec_time = time.time() - start_time

      if result.returncode != 0:
        return [], f"Runtime error: {result.stderr.decode()[:500]}", exec_time

      # Parse output
      with open(output_file, 'r') as f:
        clusters = parse_text_output(f.read(), num_points)

      # Cleanup
      os.unlink(input_file)
      if os.path.exists(output_file):
        os.unlink(output_file)

      return clusters, "", exec_time

    except subprocess.TimeoutExpired:
      return [], "Timeout", TIMEOUT_SECONDS
    except Exception as e:
      return [], str(e), time.time() - start_time
  else:
    # Text mode for smaller cases
    input_data = format_text_input(num_points, num_clusters, points)

    start_time = time.time()
    try:
      result = subprocess.run([str(exe_path)],
                              input=input_data,
                              capture_output=True,
                              text=True,
                              timeout=TIMEOUT_SECONDS)
      exec_time = time.time() - start_time

      if result.returncode != 0:
        return [], f"Runtime error: {result.stderr[:500]}", exec_time

      clusters = parse_text_output(result.stdout, num_points)
      return clusters, "", exec_time

    except subprocess.TimeoutExpired:
      return [], "Timeout", TIMEOUT_SECONDS
    except Exception as e:
      return [], str(e), time.time() - start_time


def gradeAnswer(result: dict, subPass: int, aiEngineName: str) -> tuple:
  """Grade the clustering solution."""
  if not result:
    return 0.0, "No result provided"

  if "rust_code" not in result:
    return 0.0, "No Rust code provided"

  case = TEST_CASES[subPass]
  description = case["description"]
  num_points = case["num_points"]
  num_clusters = case["num_clusters"]

  code = result["rust_code"]

  # For very large cases, just check compilation
  if num_points > 100_000_000:
    compiler = RustCompiler(aiEngineName)
    if not compiler.find_compiler():
      return 0.0, f"[{description}] No Rust compiler"
    try:
      compiler.compile(code)
      return 0.5, f"[{description}] Compiled (too large to run)"
    except CompilationError as e:
      return 0.0, f"[{description}] Compile error: {str(e)[:200]}"

  # Run clustering
  clusters, error, exec_time = run_clustering(code, case, subPass, aiEngineName)

  if error:
    return 0.0, f"[{description}] {error}"

  if len(clusters) != num_points:
    return 0.1, f"[{description}] Wrong output count: {len(clusters)} vs {num_points}"

  # Validate cluster indices
  invalid = sum(1 for c in clusters if c < 0 or c >= num_clusters)
  if invalid > 0:
    return 0.2, f"[{description}] {invalid} invalid cluster indices"

  # Check all clusters used
  used_clusters = len(set(clusters))
  if used_clusters < num_clusters * 0.5:
    score = 0.3 + 0.2 * (used_clusters / num_clusters)
    return score, f"[{description}] Only {used_clusters}/{num_clusters} clusters used"

  # Calculate WCSS for scoring (sample for large cases)
  points, _, _, _ = get_test_data(subPass)

  if num_points > 1_000_000:
    # Sample for WCSS calculation
    sample_size = 100_000
    step = num_points // sample_size
    sample_points = [points[i] for i in range(0, num_points, step)][:sample_size]
    sample_clusters = [clusters[i] for i in range(0, num_points, step)][:sample_size]
    wcss = calculate_wcss(sample_points, sample_clusters, num_clusters)
  else:
    wcss = calculate_wcss(points, clusters, num_clusters)

  # Score based on completion and cluster usage
  cluster_coverage = used_clusters / num_clusters
  score = 0.7 + 0.3 * min(1.0, cluster_coverage)

  explanation = (f"[{description}] Clusters used: {used_clusters}/{num_clusters}, "
                 f"WCSS: {wcss:.2e}, Time: {exec_time:.2f}s")

  return score, explanation


def resultToNiceReport(result: dict, subPass: int, aiEngineName: str) -> str:
  """Generate HTML report with clustering visualization."""
  if not result:
    return "<p style='color:red'>No result provided</p>"

  case = TEST_CASES[subPass]
  html = f"<h4>3D Point Clustering - {case['description']}</h4>"

  if "reasoning" in result:
    reasoning = result['reasoning'][:500] + ('...'
                                             if len(result.get('reasoning', '')) > 500 else '')
    reasoning_escaped = reasoning.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    html += f"<p><strong>Approach:</strong> {reasoning_escaped}</p>"

  if "rust_code" in result:
    code = result["rust_code"]
    code_escaped = code.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    html += f"<details><summary>View Rust Code ({len(code)} chars)</summary><pre>{code_escaped}</pre></details>"

  # SVG clustering visualization
  html += _generate_clustering_svg(case)
  return html


def _generate_clustering_svg(case: dict) -> str:
  """Generate SVG showing clustering concept."""
  return f'''
  <details>
    <summary>📊 Clustering Diagram</summary>
    <svg viewBox="0 0 300 200" style="max-width:300px; margin:10px auto; display:block; background:#f8f8f8; border-radius:8px;">
      <circle cx="50" cy="50" r="25" fill="none" stroke="#4488ff" stroke-width="2" stroke-dasharray="4"/>
      <circle cx="45" cy="45" r="3" fill="#4488ff"/><circle cx="55" cy="50" r="3" fill="#4488ff"/>
      <circle cx="50" cy="58" r="3" fill="#4488ff"/><circle cx="42" cy="52" r="3" fill="#4488ff"/>
      <circle cx="150" cy="80" r="30" fill="none" stroke="#44aa44" stroke-width="2" stroke-dasharray="4"/>
      <circle cx="145" cy="75" r="3" fill="#44aa44"/><circle cx="155" cy="82" r="3" fill="#44aa44"/>
      <circle cx="148" cy="90" r="3" fill="#44aa44"/><circle cx="160" cy="72" r="3" fill="#44aa44"/>
      <circle cx="250" cy="120" r="20" fill="none" stroke="#ff8844" stroke-width="2" stroke-dasharray="4"/>
      <circle cx="245" cy="115" r="3" fill="#ff8844"/><circle cx="255" cy="122" r="3" fill="#ff8844"/>
      <circle cx="250" cy="128" r="3" fill="#ff8844"/>
      <text x="150" y="180" fill="#666" font-size="11" text-anchor="middle">K={case.get('num_clusters', '?')} clusters | {case.get('num_points', '?')} points</text>
    </svg>
  </details>
  '''


highLevelSummary = """
3D point clustering partitions spatial data into meaningful groups.

**Algorithms:**
- **K-means**: Iterative centroid optimization, O(nki) per iteration
- **K-means++**: Smart initialization for better convergence
- **DBSCAN**: Density-based clustering, handles arbitrary shapes

**Key metrics:**
- WCSS (Within-Cluster Sum of Squares): Lower is better
- Silhouette score: Measures cluster separation
"""
