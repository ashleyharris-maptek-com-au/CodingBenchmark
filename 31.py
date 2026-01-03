"""
Test 31: Subset Sum (Rust Implementation)

The LLM must write Rust code to find a subset of integers that sums to
exactly a target value. This is NP-Complete.

Subpasses increase set size and target magnitude, requiring meet-in-the-middle,
dynamic programming with optimizations, or pseudo-polynomial algorithms.

Solver times out after 5 minutes.
"""

import random
import subprocess
import time
from typing import List, Tuple, Optional
from native_compiler import RustCompiler, CompilationError

title = "Subset Sum (Rust)"
TIMEOUT_SECONDS = 30
RANDOM_SEED = 31313131


def generate_subset_sum(n: int, max_val: int, seed: int) -> Tuple[List[int], int]:
  """Generate subset sum instance with a known solution."""
  rng = random.Random(seed)
  nums = [rng.randint(1, max_val) for _ in range(n)]

  # Pick random subset as solution
  solution_size = rng.randint(n // 4, 3 * n // 4)
  solution_indices = rng.sample(range(n), solution_size)
  target = sum(nums[i] for i in solution_indices)

  return nums, target


TEST_CASES = [
  {
    "n": 20,
    "max_val": 100,
    "desc": "20 numbers, small values"
  },
  {
    "n": 30,
    "max_val": 1000,
    "desc": "30 numbers, medium values"
  },
  {
    "n": 40,
    "max_val": 10000,
    "desc": "40 numbers (meet-in-middle threshold)"
  },
  {
    "n": 50,
    "max_val": 100000,
    "desc": "50 numbers, large values"
  },
  {
    "n": 100,
    "max_val": 1000000,
    "desc": "100 numbers, 1M max"
  },
  {
    "n": 200,
    "max_val": 10000000,
    "desc": "200 numbers, 10M max"
  },
  {
    "n": 500,
    "max_val": 100000000,
    "desc": "500 numbers, 100M max"
  },
  {
    "n": 1000,
    "max_val": 1000000000,
    "desc": "1K numbers, 1B max"
  },
  {
    "n": 2000,
    "max_val": 10000000000,
    "desc": "2K numbers, 10B max"
  },
  {
    "n": 5000,
    "max_val": 100000000000,
    "desc": "5K numbers, 100B max"
  },
  {
    "n": 10000,
    "max_val": 1000000000000,
    "desc": "10K numbers, 1T max"
  },
]

INSTANCE_CACHE = {}


def get_instance(subpass: int) -> Tuple[List[int], int]:
  if subpass not in INSTANCE_CACHE:
    case = TEST_CASES[subpass]
    nums, target = generate_subset_sum(case["n"], case["max_val"], RANDOM_SEED + subpass)
    INSTANCE_CACHE[subpass] = (nums, target)
  return INSTANCE_CACHE[subpass]


def format_input(nums: List[int], target: int) -> str:
  lines = [f"{len(nums)} {target}"]
  lines.append(" ".join(map(str, nums)))
  return "\n".join(lines)


def prepareSubpassPrompt(subPass: int) -> str:
  """Generate the prompt for subpass 0 that handles all subset sum complexities."""
  if subPass != 0:
    raise StopIteration

  return f"""You are writing Rust code to solve the Subset Sum problem.

You must write a Rust solver that can handle ANY subset sum complexity from trivial to ludicrous scale:
- **Trivial**: Small sets (20 numbers), exact backtracking, brute force
- **Medium**: Moderate sets (100 numbers), meet-in-the-middle, optimized DP
- **Large**: Complex sets (1000 numbers), pseudo-polynomial DP, bitset optimization
- **Extreme**: Massive sets (10000+ numbers), approximation algorithms, randomized methods

**The Challenge:**
Your Rust subset sum solver will be tested with instances ranging from simple to very complex sets. The same algorithm must work efficiently across ALL problem complexities.

**Problem:**
Given a set of integers and a target value, determine if there exists a subset that sums to exactly the target value. This is NP-Complete and requires sophisticated algorithms for larger instances.

**Input format (stdin):**
```
num_numbers target
num1 num2 ... numN
```

**Output format (stdout):**
```
YES or NO
[index1 index2 ... indexK]  (only if YES, 0-indexed indices of subset elements)
```

**Critical Requirements:**
1. **Scalability**: Your algorithm must adapt based on set size and target magnitude
2. **Performance**: Must complete within 5 minutes even for massive sets
3. **Quality**: Find exact solutions or correctly report impossibility

**Algorithm Strategy Recommendations:**
- **Small sets (≤50 numbers)**: Can use exact backtracking with pruning
- **Medium sets (50-500 numbers)**: Meet-in-the-middle algorithm, optimized DP
- **Large sets (500-5000 numbers)**: Pseudo-polynomial DP with bitset optimization
- **Very Large sets (>5000 numbers)**: Approximation algorithms, randomized methods

**Key Techniques:**
- **Backtracking**: Systematic search with pruning based on partial sums
- **Meet-in-the-middle**: Split set in half, compute all subset sums of each half
- **Dynamic Programming**: DP table for reachable sums, optimized with bitsets
- **Approximation**: PTAS (Polynomial Time Approximation Scheme) for large instances
- **Randomization**: Randomized algorithms for very large instances

**Implementation Hints:**
- Detect set complexity and choose appropriate algorithm
- Use efficient data structures: bitsets, hash sets for DP
- Implement adaptive quality vs speed tradeoffs
- For very large sets, focus on approximation algorithms
- Handle edge cases: empty set, zero target, negative numbers
- Use fast I/O for large inputs

**Success Criteria:**
- Correctly determine if subset sum exists
- If YES, provide valid subset indices
- Complete within time limit

**Failure Criteria:**
- Incorrect subset sum determination
- Invalid subset indices
- Timeout without conclusion

**Requirements:**
1. Program must compile with rustc (edition 2021)
2. Read from stdin, write to stdout
3. Handle variable set sizes and target values
4. Complete within 5 minutes
5. Must handle varying subset sum complexities efficiently

Write complete, compilable Rust code with a main function.
Include adaptive logic that chooses different strategies based on subset sum complexity.
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
      "Explain your algorithm approach and how it adapts to different subset sum complexities"
    },
    "rust_code": {
      "type": "string",
      "description": "Complete Rust code with main function that handles all scales"
    }
  },
  "required": ["reasoning", "rust_code"],
  "additionalProperties": False
}


def verify_subset(nums: List[int], target: int, indices: List[int]) -> Tuple[bool, str]:
  if not indices:
    return target == 0, "Empty subset"

  for idx in indices:
    if idx < 0 or idx >= len(nums):
      return False, f"Invalid index {idx}"

  if len(indices) != len(set(indices)):
    return False, "Duplicate indices"

  actual_sum = sum(nums[i] for i in indices)
  if actual_sum != target:
    return False, f"Sum {actual_sum} != target {target}"

  return True, "Valid subset"


def gradeAnswer(result: dict, subPass: int, aiEngineName: str) -> tuple:
  if not result or "rust_code" not in result:
    return 0.0, "No Rust code provided"

  case = TEST_CASES[subPass]
  nums, target = get_instance(subPass)

  compiler = RustCompiler(aiEngineName)
  if not compiler.find_compiler():
    return 0.0, "No Rust compiler found"

  try:
    exe_path = compiler.compile(result["rust_code"])
  except CompilationError as e:
    return 0.0, f"Compilation error: {str(e)[:200]}"

  input_data = format_input(nums, target)

  try:
    start = time.time()
    proc = subprocess.run([str(exe_path)],
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

    if lines[0].strip() == "NO":
      return 0.3, f"[{case['desc']}] Reports NO (instance has solution), {exec_time:.2f}s"

    if lines[0].strip() == "YES":
      if len(lines) < 2:
        return 0.2, f"[{case['desc']}] YES but no indices"

      indices = list(map(int, lines[1].split()))
      valid, msg = verify_subset(nums, target, indices)

      if valid:
        return 1.0, f"[{case['desc']}] Valid subset of {len(indices)} elements, {exec_time:.2f}s"
      else:
        return 0.2, f"[{case['desc']}] {msg}"

    return 0.1, f"[{case['desc']}] Unknown output format"

  except subprocess.TimeoutExpired:
    return 0.1, f"[{case['desc']}] Timeout"
  except Exception as e:
    return 0.0, f"[{case['desc']}] Error: {str(e)[:100]}"


def output_example_html(score: float, explanation: str, result: dict, subPass: int) -> str:
  case = TEST_CASES[subPass]
  code = result.get("rust_code", "").replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
  color = "green" if score >= 0.8 else "orange" if score >= 0.4 else "red"
  return f'<div class="result"><h4>Subpass {subPass}: {case["desc"]}</h4><p style="color:{color}">Score: {score:.2f}</p><p>{explanation}</p><details><summary>Code</summary><pre>{code}</pre></details></div>'


def output_header_html() -> str:
  return "<h2>Test 31: Subset Sum (Rust)</h2><p>NP-Complete number theory problem.</p>"


def output_summary_html(results: list) -> str:
  total = sum(r[0] for r in results)
  return f'<div class="summary"><p>Total: {total:.2f}/{len(results)}</p></div>'
