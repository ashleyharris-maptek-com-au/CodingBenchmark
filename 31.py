"""
Test 31: Subset Sum (Rust Implementation)

The LLM must write Rust code to find a subset of integers that sums to
exactly a target value. This is NP-Complete.

Solver times out after 5 minutes.
"""

import random
import subprocess
import time
from typing import List, Tuple, Optional, Dict, Any
from native_compiler import RustCompiler, CompilationError, ExecutionError,describe_this_pc
from solver_utils import StreamingInputFile

title = "Subset Sum (Rust)"

tags = [
  "rust",
  "structured response",
  "np hard",
  "algorithm design",
]
TIMEOUT_SECONDS = 300
RANDOM_SEED = 31313131


def generate_subset_sum(n: int, max_val: int,
                        seed: int) -> Tuple[List[int], int, List[int]]:
  """Generate subset sum instance with a known solution."""
  rng = random.Random(seed)
  nums = [rng.randint(1, max_val) for _ in range(n)]

  # Pick random subset as solution
  solution_size = rng.randint(n // 4, 3 * n // 4)
  solution_indices = rng.sample(range(n), solution_size)
  target = sum(nums[i] for i in solution_indices)

  return nums, target, solution_indices


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
  # Ludicrous cases for streaming
  {
    "n": 50000,
    "max_val": 10000000000000,
    "desc": "50K numbers, 10T max"
  },
  {
    "n": 200000,
    "max_val": 100000000000000,
    "desc": "200K numbers, 100T max"
  },
  {
    "n": 1000000,
    "max_val": 1000000000000000,
    "desc": "1M numbers, 1P max"
  },
  {
    "n": 5000000,
    "max_val": 10000000000000000,
    "desc": "5M numbers (~100MB)"
  },
  {
    "n": 20000000,
    "max_val": 100000000000000000,
    "desc": "20M numbers (~400MB)"
  },
  {
    "n": 50000000,
    "max_val": 1000000000000000000,
    "desc": "50M numbers (~1GB)"
  },
]

INSTANCE_CACHE: Dict[int, Any] = {}
_INPUT_FILE_CACHE: Dict[int, StreamingInputFile] = {}
LAST_SUBSET_VIZ: Dict[Tuple[int, str], dict] = {}
STREAMING_THRESHOLD_N = 100_000


def get_instance(subpass: int) -> Tuple[List[int], int, List[int]]:
  if subpass not in INSTANCE_CACHE:
    case = TEST_CASES[subpass]
    nums, target, solution_indices = generate_subset_sum(
      case["n"], case["max_val"], RANDOM_SEED + subpass
    )
    INSTANCE_CACHE[subpass] = (nums, target, solution_indices)
  return INSTANCE_CACHE[subpass]


def _should_use_streaming(subpass: int) -> bool:
  return TEST_CASES[subpass]["n"] > STREAMING_THRESHOLD_N


def _get_streaming_input(subpass: int) -> StreamingInputFile:
  if subpass in _INPUT_FILE_CACHE:
    return _INPUT_FILE_CACHE[subpass]

  case = TEST_CASES[subpass]
  cache_key = f"subset31|n={case['n']}|max={case['max_val']}|seed={RANDOM_SEED + subpass}"

  def generator():
    nums, target, _ = get_instance(subpass)
    yield f"{len(nums)} {target}\n"
    yield " ".join(map(str, nums)) + "\n"

  input_file = StreamingInputFile(cache_key, generator, "test31_subset")
  _INPUT_FILE_CACHE[subpass] = input_file
  return input_file


def format_input(nums: List[int], target: int) -> str:
  lines = [f"{len(nums)} {target}"]
  lines.append(" ".join(map(str, nums)))
  return "\n".join(lines)


def prepareSubpassPrompt(subPass: int) -> str:
  """Generate the prompt for subpass 0 that handles all subset sum complexities."""
  if subPass != 0:
    raise StopIteration

  return f"""You are writing Rust code to solve the Subset Sum problem.

Your Rust subset sum solver will be tested with instances ranging from simple to very 
complex sets. The same algorithm must work efficiently across ALL problem complexities.

Given a set of integers and a target value, determine if there exists a subset 
that sums to exactly the target value. Simple sets (<100 numbers) you can probably
brute force, however this code will also need to run on sets with millions of
numbers. This is NP-Complete and requires sophisticated algorithms for larger instances:
additionally; threading, memory optimization, vectorization and parallelization are all 
strongly suggested.

**Input format (stdin, space seperated decimal integers):**
```
num_numbers target
num1 num2 ... numN
```

**Output format (stdout):**
```
YES or NO
[index1 index2 ... indexK]  (only if YES, 0-indexed indices of subset elements)
```

**Implementation Hints:**
- Detect set complexity and choose appropriate algorithm
- Use efficient data structures: bitsets, hash sets for DP
- Implement adaptive quality vs speed tradeoffs
- For very large sets, focus on approximation algorithms
- Handle edge cases: empty set, zero target, all numbers add to target.
- Use fast I/O as the input size can approach 1gb.

**Success Criteria:**
- Correctly determine if subset sum exists
- If YES, provide valid subset indices
- Complete within time limit

**Failure Criteria:**
- Incorrect subset sum determination
- Invalid subset indices
- Timeout without conclusion

**Requirements:**
1. Program must compile with rustc
2. Read from stdin, write to stdout
3. Handle variable set sizes and target values
4. Complete within 5 minutes
5. Must handle varying subset sum complexities efficiently

**Environment:**
{describe_this_pc()}

**Rust Compiler:**
{RustCompiler("test_engine").describe()}

Be aware that default warnings are enabled and will cause a compilation failure,
so ensure that you write warning-free code.

Write complete, compilable Rust code with a main() function.
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
  use_streaming = _should_use_streaming(subPass)

  compiler = RustCompiler(aiEngineName)
  if not compiler.find_compiler():
    return 0.0, "No Rust compiler found"

  try:
    exe_path = compiler.compile(result["rust_code"])
  except CompilationError as e:
    return 0.0, f"Compilation error: {str(e)[:200]}"

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
      stdout, stderr, exec_time, return_code = compiler.execute(exe_path,
                                                                timeout=TIMEOUT_SECONDS,
                                                                stdin_file=input_file_path)

      if return_code != 0:
        return 0.0, f"Runtime error: {stderr[:200]}"

      # Skip verification for very large cases
      if case["n"] > 1_000_000:
        lines = stdout.strip().split('\n')
        if lines and lines[0].strip() == "YES":
          return 0.8, f"[{case['desc']}] Reports YES in {exec_time:.2f}s (verification skipped)"
        elif lines and lines[0].strip() == "NO":
          return 0.5, f"[{case['desc']}] Reports NO in {exec_time:.2f}s"
        else:
          return 0.2, f"[{case['desc']}] Unknown output"

      proc_stdout = stdout
    else:
      nums, target, solution_indices = get_instance(subPass)
      input_data = format_input(nums, target)

      start = time.time()
      proc = subprocess.run([str(exe_path)],
                            input=input_data,
                            capture_output=True,
                            text=True,
                            timeout=TIMEOUT_SECONDS)
      exec_time = time.time() - start

      if proc.returncode != 0:
        return 0.0, f"Runtime error: {proc.stderr[:200]}"
      proc_stdout = proc.stdout

    lines = proc_stdout.strip().split('\n')
    if not lines:
      return 0.0, "No output"

    if lines[0].strip() == "NO":
      return 0.3, f"[{case['desc']}] Reports NO (instance has solution), {exec_time:.2f}s"

    if lines[0].strip() == "YES":
      if len(lines) < 2:
        return 0.2, f"[{case['desc']}] YES but no indices"

      indices = list(map(int, lines[1].split()))
      nums, target, solution_indices = get_instance(subPass)
      valid, msg = verify_subset(nums, target, indices)

      if case["n"] <= 200 and not use_streaming:
        LAST_SUBSET_VIZ[(subPass, aiEngineName)] = _build_subset_viz(
          nums, target, solution_indices, indices, valid, msg
        )

      if valid:
        return 1.0, f"[{case['desc']}] Valid subset of {len(indices)} elements, {exec_time:.2f}s"
      else:
        return 0.0, f"[{case['desc']}] {msg}"

    return 0.0, f"[{case['desc']}] Unknown output format"

  except subprocess.TimeoutExpired:
    return 0.0, f"[{case['desc']}] Timeout"
  except Exception as e:
    return 0.0, f"[{case['desc']}] Error: {str(e)[:100]}"


def resultToNiceReport(result: dict, subPass: int, aiEngineName: str) -> str:
  if not result:
    return "<p style='color:red'>No result provided</p>"
  case = TEST_CASES[subPass]
  html = f"<h4>Subset Sum - {case['desc']}</h4>"
  if "reasoning" in result and subPass ==0:
    r = result['reasoning'][:400] + ('...' if len(result.get('reasoning', '')) > 400 else '')
    html += f"<p><strong>Approach:</strong> {r.replace('<', '&lt;').replace('>', '&gt;')}</p>"
  if "rust_code" in result and subPass ==0:
    code = result["rust_code"].replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    html += f"<details><summary>View Rust Code ({len(result['rust_code'])} chars)</summary><pre>{code}</pre></details>"
  viz = LAST_SUBSET_VIZ.get((subPass, aiEngineName))
  if viz:
    html += _generate_subset_viz_html(viz)
  return html


def _build_subset_viz(nums: List[int], target: int, solution_indices: List[int],
                      chosen_indices: List[int], valid: bool, msg: str) -> dict:
  solution_set = set(solution_indices)
  chosen_set = set(chosen_indices)
  items = []
  for i, val in enumerate(nums):
    in_solution = i in solution_set
    in_chosen = i in chosen_set
    if in_solution and in_chosen:
      status = "tp"
    elif in_solution and not in_chosen:
      status = "fn"
    elif not in_solution and in_chosen:
      status = "fp"
    else:
      status = "tn"
    items.append({"index": i, "value": val, "status": status})

  return {
    "n": len(nums),
    "target": target,
    "valid": valid,
    "message": msg,
    "items": items,
    "chosen_sum": sum(nums[i] for i in chosen_set if i < len(nums)),
    "solution_sum": sum(nums[i] for i in solution_set if i < len(nums)),
  }


def _generate_subset_viz_html(viz: dict) -> str:
  color_map = {
    "tp": ("#22c55e", "Chosen & in true solution"),
    "tn": ("#9ca3af", "Not chosen & not in solution"),
    "fp": ("#f59e0b", "Chosen but NOT in solution"),
    "fn": ("#facc15", "In solution but NOT chosen"),
  }
  max_items = 200
  items = viz["items"][:max_items]
  total = viz["n"]

  chips = []
  for item in items:
    fill = color_map[item["status"]][0]
    chips.append(
      "<div style='display:flex;flex-direction:column;align-items:center;gap:2px;"
      "padding:6px 4px;border-radius:6px;background:#0f172a;border:1px solid #1f2937;'>"
      f"<div style='font-size:11px;color:#e2e8f0'>{item['value']}</div>"
      f"<div style='font-size:9px;color:#94a3b8'>#{item['index']}</div>"
      f"<div style='width:14px;height:14px;border-radius:50%;background:{fill};'></div>"
      "</div>"
    )

  legend_items = []
  for key in ("tp", "fp", "fn", "tn"):
    color, label = color_map[key]
    legend_items.append(
      "<div style='display:flex;align-items:center;gap:6px;'>"
      f"<span style='width:12px;height:12px;border-radius:3px;background:{color};display:inline-block;'></span>"
      f"<span style='color:#cbd5f5;font-size:11px;'>{label}</span>"
      "</div>"
    )

  status = "VALID" if viz["valid"] else "INVALID"
  status_color = "#22c55e" if viz["valid"] else "#f97316"
  header = (
    f"<div style='color:#e2e8f0;font-size:13px;margin-bottom:6px;'>"
    f"<strong>Subset Visualization</strong> &mdash; "
    f"<span style='color:{status_color};'>{status}</span> "
    f"(target={viz['target']}, chosen_sum={viz['chosen_sum']})</div>"
  )
  if viz.get("message"):
    header += (
      f"<div style='color:#94a3b8;font-size:11px;margin-bottom:6px;'>"
      f"{viz['message']}</div>"
    )

  more_note = ""
  if total > max_items:
    more_note = (
      f"<div style='color:#94a3b8;font-size:11px;margin-top:6px;'>"
      f"Showing first {max_items} of {total} numbers.</div>"
    )

  return (
    "<div style='margin:12px 0;padding:10px;border:1px solid #1f2937;"
    "border-radius:8px;background:#0b1120;'>"
    f"{header}"
    "<div style='display:flex;flex-wrap:wrap;gap:6px;margin-bottom:8px;'>"
    + "".join(legend_items) +
    "</div>"
    "<div style='display:grid;grid-template-columns:repeat(auto-fill, minmax(70px, 1fr));gap:6px;'>"
    + "".join(chips) +
    "</div>"
    f"{more_note}"
    "</div>"
  )


highLevelSummary = """
<p>Given a list of numbers and a target total, find a subset of those numbers that
adds up to exactly the target. It sounds simple, but with large lists the number
of possible subsets is astronomical.</p>
<p>This is a classic NP-hard problem. The test uses planted instances with a known
solution for strict pass/fail grading. Subpasses increase the list size and
target magnitude.</p>
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
