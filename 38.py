"""
Test 38: Exact Cover Problem (Python Implementation)

The LLM must write Python code to find a subset of sets that partitions
a universe exactly (each element covered exactly once). This is NP-Complete.

This is the foundation of Knuth's Algorithm X / Dancing Links, used in
Sudoku solvers and many combinatorial problems.

Solver times out after 5 minutes.
"""

import random
import subprocess
import time
from typing import List, Set, Tuple, Dict, Any
from solver_utils import StreamingInputFile

title = "Exact Cover Problem (Python)"
TIMEOUT_SECONDS = 30
RANDOM_SEED = 38383838


def generate_exact_cover(universe_size: int, num_sets: int, set_size_range: Tuple[int, int],
                         seed: int) -> Tuple[Set[int], List[Set[int]]]:
  """Generate exact cover instance with a known solution."""
  rng = random.Random(seed)
  universe = set(range(universe_size))

  # Create a valid solution first
  remaining = set(universe)
  solution_sets = []
  while remaining:
    size = min(rng.randint(*set_size_range), len(remaining))
    s = set(rng.sample(list(remaining), size))
    solution_sets.append(s)
    remaining -= s

  # Add more random sets
  all_sets = list(solution_sets)
  while len(all_sets) < num_sets:
    size = rng.randint(*set_size_range)
    s = set(rng.sample(list(universe), min(size, universe_size)))
    if s and s not in all_sets:
      all_sets.append(s)

  rng.shuffle(all_sets)
  return universe, all_sets


TEST_CASES = [
  {
    "universe": 20,
    "sets": 30,
    "size_range": (2, 5),
    "desc": "20 elements, 30 sets"
  },
  {
    "universe": 40,
    "sets": 80,
    "size_range": (3, 7),
    "desc": "40 elements, 80 sets"
  },
  {
    "universe": 60,
    "sets": 150,
    "size_range": (3, 8),
    "desc": "60 elements, 150 sets"
  },
  {
    "universe": 100,
    "sets": 300,
    "size_range": (4, 10),
    "desc": "100 elements, 300 sets"
  },
  {
    "universe": 150,
    "sets": 500,
    "size_range": (4, 12),
    "desc": "150 elements, 500 sets"
  },
  {
    "universe": 200,
    "sets": 800,
    "size_range": (5, 15),
    "desc": "200 elements, 800 sets"
  },
  {
    "universe": 300,
    "sets": 1500,
    "size_range": (5, 18),
    "desc": "300 elements, 1.5K sets"
  },
  {
    "universe": 500,
    "sets": 3000,
    "size_range": (6, 20),
    "desc": "500 elements, 3K sets"
  },
  {
    "universe": 750,
    "sets": 5000,
    "size_range": (6, 25),
    "desc": "750 elements, 5K sets"
  },
  {
    "universe": 1000,
    "sets": 8000,
    "size_range": (8, 30),
    "desc": "1K elements, 8K sets"
  },
  {
    "universe": 2000,
    "sets": 20000,
    "size_range": (10, 40),
    "desc": "2K elements, 20K sets"
  },
  # Ludicrous cases for streaming
  {
    "universe": 5000,
    "sets": 100000,
    "size_range": (15, 50),
    "desc": "5K elements, 100K sets (~5MB)"
  },
  {
    "universe": 10000,
    "sets": 500000,
    "size_range": (20, 60),
    "desc": "10K elements, 500K sets (~30MB)"
  },
  {
    "universe": 50000,
    "sets": 2000000,
    "size_range": (25, 80),
    "desc": "50K elements, 2M sets (~150MB)"
  },
  {
    "universe": 100000,
    "sets": 5000000,
    "size_range": (30, 100),
    "desc": "100K elements, 5M sets (~500MB)"
  },
  {
    "universe": 200000,
    "sets": 15000000,
    "size_range": (40, 120),
    "desc": "200K elements, 15M sets (~1.5GB)"
  },
]

INSTANCE_CACHE: Dict[int, Any] = {}
_INPUT_FILE_CACHE: Dict[int, StreamingInputFile] = {}
STREAMING_THRESHOLD_SETS = 50_000


def get_instance(subpass: int):
  if subpass not in INSTANCE_CACHE:
    case = TEST_CASES[subpass]
    universe, sets = generate_exact_cover(case["universe"], case["sets"], case["size_range"],
                                          RANDOM_SEED + subpass)
    INSTANCE_CACHE[subpass] = (universe, sets)
  return INSTANCE_CACHE[subpass]


def _should_use_streaming(subpass: int) -> bool:
  return TEST_CASES[subpass]["sets"] > STREAMING_THRESHOLD_SETS


def _get_streaming_input(subpass: int) -> StreamingInputFile:
  if subpass in _INPUT_FILE_CACHE:
    return _INPUT_FILE_CACHE[subpass]

  case = TEST_CASES[subpass]
  cache_key = f"exactcover38|u={case['universe']}|s={case['sets']}|r={case['size_range']}|seed={RANDOM_SEED + subpass}"

  def generator():
    universe, sets = get_instance(subpass)
    yield f"{len(universe)} {len(sets)}\n"
    for s in sets:
      yield " ".join(map(str, sorted(s))) + "\n"

  input_file = StreamingInputFile(cache_key, generator, "test38_exactcover")
  _INPUT_FILE_CACHE[subpass] = input_file
  return input_file


def format_input(universe: Set[int], sets: List[Set[int]]) -> str:
  lines = [f"{len(universe)} {len(sets)}"]
  for s in sets:
    lines.append(" ".join(map(str, sorted(s))))
  return "\n".join(lines)


def prepareSubpassPrompt(subPass: int) -> str:
  """Generate the prompt for subpass 0 that handles all difficulty levels."""
  if subPass != 0:
    raise StopIteration

  return f"""You are writing CPP code to solve the Maximum Matching problem.

You must write a CPP solver that can handle ANY problem complexity from trivial to ludicrous scale:
- **Trivial**: Simple instances, basic algorithms
- **Medium**: Moderate instances, efficient algorithms  
- **Large**: Complex instances, advanced algorithms
- **Extreme**: Massive instances, very fast heuristics

**The Challenge:**
Your CPP solver will be tested with instances ranging from simple to very complex cases. The same algorithm must work efficiently across ALL problem complexities.

**Problem:**
Find a maximum set of edges without common vertices in a graph.

**Input format (stdin):**
```
[Input format varies by problem]
```

**Output format (stdout):**
```
[Output format varies by problem]
```

**Critical Requirements:**
1. **Scalability**: Your algorithm must adapt based on problem size and complexity
2. **Performance**: Must complete within 5 minutes even for the largest instances
3. **Quality**: Find optimal or near-optimal solutions

**Algorithm Strategy Recommendations:**
Small graphs (50 vertices): augmenting path algorithms, Medium (500 vertices): Hopcroft-Karp, Large (5000 vertices): very fast implementations, Extreme (50000+ vertices): streaming algorithms

**Implementation Hints:**
- Detect problem complexity and choose appropriate algorithm
- Use efficient data structures and algorithms
- Implement adaptive quality vs speed tradeoffs
- For very large instances, focus on fast heuristics
- Handle edge cases appropriately
- Use fast I/O for large inputs

**Requirements:**
1. Program must compile with appropriate compiler
2. Read from stdin, write to stdout
3. Handle variable problem sizes
4. Complete within 5 minutes
5. Must handle varying problem complexities efficiently

Write complete, compilable CPP code.
Include adaptive logic that chooses different strategies based on problem complexity.
"""# List of subpasses to grade the single answer against all difficulty levels


extraGradeAnswerRuns = list(range(len(TEST_CASES)))

structure = {
  "type": "object",
  "properties": {
    "reasoning": {
      "type": "string",
      "description": "Explain your algorithm approach and how it adapts to different problem sizes"
    },
    "python_code": {
      "type": "string",
      "description": "Complete Python code with solver function that handles all scales"
    }
  },
  "required": ["reasoning", "python_code"],
  "additionalProperties": False
}


def verify_exact_cover(universe: Set[int], sets: List[Set[int]],
                       indices: List[int]) -> Tuple[bool, str]:
  if not indices:
    return len(universe) == 0, "Empty solution"

  for idx in indices:
    if idx < 0 or idx >= len(sets):
      return False, f"Invalid set index {idx}"

  covered = set()
  for idx in indices:
    overlap = covered & sets[idx]
    if overlap:
      return False, f"Set {idx} overlaps with previous: {overlap}"
    covered |= sets[idx]

  if covered != universe:
    missing = universe - covered
    extra = covered - universe
    if missing:
      return False, f"Missing elements: {missing}"
    if extra:
      return False, f"Extra elements: {extra}"

  return True, "Valid exact cover"


def gradeAnswer(result: dict, subPass: int, aiEngineName: str) -> tuple:
  if not result or "python_code" not in result:
    return 0.0, "No Python code provided"

  case = TEST_CASES[subPass]
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

      # Skip verification for very large cases
      if case["sets"] > 1_000_000:
        start = time.time()
        with open(input_file_path, 'r') as f:
          proc = subprocess.run(["python", "-c", result["python_code"]],
                                stdin=f,
                                capture_output=True,
                                text=True,
                                timeout=TIMEOUT_SECONDS)
        exec_time = time.time() - start
        if proc.returncode == 0:
          lines = proc.stdout.strip().split('\n')
          if lines and lines[0].strip() == "SOLUTION":
            return 0.8, f"[{case['desc']}] Found solution in {exec_time:.2f}s (verification skipped)"
          elif lines and lines[0].strip() == "NO SOLUTION":
            return 0.5, f"[{case['desc']}] Reports no solution in {exec_time:.2f}s"
        return 0.0, f"Runtime error: {proc.stderr[:200]}"

      start = time.time()
      with open(input_file_path, 'r') as f:
        proc = subprocess.run(["python", "-c", result["python_code"]],
                              stdin=f,
                              capture_output=True,
                              text=True,
                              timeout=TIMEOUT_SECONDS)
      exec_time = time.time() - start
    else:
      universe, sets = get_instance(subPass)
      input_data = format_input(universe, sets)

      start = time.time()
      proc = subprocess.run(["python", "-c", result["python_code"]],
                            input=input_data,
                            capture_output=True,
                            text=True,
                            timeout=TIMEOUT_SECONDS)
      exec_time = time.time() - start

    if proc.returncode != 0:
      return 0.0, f"Runtime error: {proc.stderr[:200]}"

    lines = proc.stdout.strip().split('\n')
    if not lines:
      return 0.0, "No output"

    if lines[0].strip() == "NO SOLUTION":
      return 0.3, f"[{case['desc']}] Reports no solution (instance has one), {exec_time:.2f}s"

    if lines[0].strip() == "SOLUTION":
      if len(lines) < 2:
        return 0.2, f"[{case['desc']}] SOLUTION but no indices"

      indices = list(map(int, lines[1].split()))
      valid, msg = verify_exact_cover(universe, sets, indices)

      if valid:
        return 1.0, f"[{case['desc']}] Valid exact cover with {len(indices)} sets, {exec_time:.2f}s"
      else:
        return 0.2, f"[{case['desc']}] {msg}"

    return 0.1, f"[{case['desc']}] Unknown output format"

  except subprocess.TimeoutExpired:
    return 0.1, f"[{case['desc']}] Timeout"
  except Exception as e:
    return 0.0, f"[{case['desc']}] Error: {str(e)[:100]}"


def resultToNiceReport(result: dict, subPass: int, aiEngineName: str) -> str:
  if not result:
    return "<p style='color:red'>No result provided</p>"
  case = TEST_CASES[subPass]
  html = f"<h4>Exact Cover - {case['desc']}</h4>"
  if "reasoning" in result:
    r = result['reasoning'][:400] + ('...' if len(result.get('reasoning', '')) > 400 else '')
    html += f"<p><strong>Approach:</strong> {r.replace('<', '&lt;').replace('>', '&gt;')}</p>"
  if "python_code" in result:
    code = result["python_code"].replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    html += f"<details><summary>View Python Code ({len(result['python_code'])} chars)</summary><pre>{code}</pre></details>"
  return html


highLevelSummary = """
Exact Cover finds subsets that partition a universe exactly once.

**Algorithms:**
- **Algorithm X**: Knuth's dancing links (DLX)
- **Backtracking**: Systematic search with pruning
- **Reduction from SAT**: Many problems reduce to exact cover
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
