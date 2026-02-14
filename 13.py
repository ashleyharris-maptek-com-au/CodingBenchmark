"""
Test 13: Longest Common Substring

The LLM must write a Python solver that finds the longest common substring
across all strings in a given list.

A substring must appear contiguously in all strings.

Subpasses test increasingly difficult cases (longer strings, more strings).
Solver times out after 5 minutes.
"""

import random
import string
import time
from typing import List, Tuple, Optional

from native_compiler import CppCompiler, compile_and_run, describe_this_pc
from solver_utils import StreamingInputFile

title = "Longest Common Substring (C++)"

# Timeout in seconds (0.5 minutes)
TIMEOUT_SECONDS = 300

# Seed for reproducibility
RANDOM_SEED = 44444


def generate_strings_with_common(num_strings: int, string_len: int, common_len: int,
                                 seed: int) -> Tuple[List[str], str]:
  """Generate strings that share a common substring."""
  rng = random.Random(seed)

  # Generate the common substring
  common = ''.join(rng.choices(string.ascii_lowercase, k=common_len))

  strings = []
  for _ in range(num_strings):
    # Generate prefix and suffix
    prefix_len = rng.randint(0, string_len - common_len)
    suffix_len = string_len - common_len - prefix_len

    prefix = ''.join(rng.choices(string.ascii_lowercase, k=prefix_len))
    suffix = ''.join(rng.choices(string.ascii_lowercase, k=suffix_len))

    strings.append(prefix + common + suffix)

  return strings, common


def generate_dna_strings(num_strings: int, string_len: int, common_len: int,
                         seed: int) -> Tuple[List[str], str]:
  """Generate DNA-like strings (ACGT) with common substring."""
  rng = random.Random(seed)
  alphabet = 'ACGT'

  common = ''.join(rng.choices(alphabet, k=common_len))

  strings = []
  for _ in range(num_strings):
    prefix_len = rng.randint(0, string_len - common_len)
    suffix_len = string_len - common_len - prefix_len

    prefix = ''.join(rng.choices(alphabet, k=prefix_len))
    suffix = ''.join(rng.choices(alphabet, k=suffix_len))

    strings.append(prefix + common + suffix)

  return strings, common


# Test configurations
TEST_CASES = [
  # Subpass 0: Simple - short strings
  {
    "strings": lambda: ["abcdefg", "xbcdefy", "zbcdefa"],
    "expected": lambda: "bcdef",
    "description": "3 short strings (7 chars)"
  },
  # Subpass 1: Medium strings
  {
    "strings": lambda: generate_strings_with_common(3, 50, 15, RANDOM_SEED)[0],
    "expected": lambda: generate_strings_with_common(3, 50, 15, RANDOM_SEED)[1],
    "description": "3 strings, 50 chars each"
  },
  # Subpass 2: More strings
  {
    "strings": lambda: generate_strings_with_common(5, 100, 20, RANDOM_SEED + 1)[0],
    "expected": lambda: generate_strings_with_common(5, 100, 20, RANDOM_SEED + 1)[1],
    "description": "5 strings, 100 chars each"
  },
  # Subpass 3: DNA sequences
  {
    "strings": lambda: generate_dna_strings(4, 200, 30, RANDOM_SEED + 2)[0],
    "expected": lambda: generate_dna_strings(4, 200, 30, RANDOM_SEED + 2)[1],
    "description": "4 DNA strings, 200 chars each"
  },
  # Subpass 4: Longer strings (harder)
  {
    "strings": lambda: generate_strings_with_common(4, 500, 40, RANDOM_SEED + 3)[0],
    "expected": lambda: generate_strings_with_common(4, 500, 40, RANDOM_SEED + 3)[1],
    "description": "4 strings, 500 chars each"
  },
  # Subpass 5: Large problem (requires efficient algorithm)
  {
    "strings": lambda: generate_strings_with_common(5, 1000, 50, RANDOM_SEED + 4)[0],
    "expected": lambda: generate_strings_with_common(5, 1000, 50, RANDOM_SEED + 4)[1],
    "description": "5 strings, 1000 chars each"
  },
  # Extreme cases
  {
    "strings": lambda: generate_strings_with_common(50, 5000, 100, RANDOM_SEED + 5)[0],
    "expected": lambda: generate_strings_with_common(50, 5000, 100, RANDOM_SEED + 5)[1],
    "description": "50 strings, 5000 chars each"
  },
  {
    "strings": lambda: generate_strings_with_common(10, 10000, 200, RANDOM_SEED + 6)[0],
    "expected": lambda: generate_strings_with_common(10, 10000, 200, RANDOM_SEED + 6)[1],
    "description": "10 strings, 10000 chars each"
  },
  {
    "strings": lambda: generate_strings_with_common(50, 50000, 500, RANDOM_SEED + 7)[0],
    "expected": lambda: generate_strings_with_common(50, 50000, 500, RANDOM_SEED + 7)[1],
    "description": "50 strings, 50000 chars each"
  },
  {
    "strings": lambda: generate_strings_with_common(100, 100000, 1000, RANDOM_SEED + 8)[0],
    "expected": lambda: generate_strings_with_common(100, 100000, 1000, RANDOM_SEED + 8)[1],
    "description": "100 strings, 100000 chars each"
  },
  {
    "strings": lambda: generate_strings_with_common(50, 1000000, 5000, RANDOM_SEED + 9)[0],
    "expected": lambda: generate_strings_with_common(50, 1000000, 5000, RANDOM_SEED + 9)[1],
    "description": "50 strings, 1000000 chars each (1MB per string)"
  },
  {
    "strings": lambda: generate_strings_with_common(100, 10000000, 10000, RANDOM_SEED + 10)[0],
    "expected": lambda: generate_strings_with_common(100, 10000000, 10000, RANDOM_SEED + 10)[1],
    "description": "100 strings, 10000000 chars each (10KB common)"
  },
]


def format_strings_for_prompt(strings: List[str], max_display: int = 100) -> str:
  """Format strings for prompt display."""
  result = []
  for i, s in enumerate(strings):
    if len(s) <= max_display:
      result.append(f'    "{s}"')
    else:
      result.append(f'    "{s[:50]}...{s[-30:]}"  # ({len(s)} chars)')
  return ",\n".join(result)


def prepareSubpassPrompt(subPass: int) -> str:
  """Generate the prompt for subpass 0 that handles all problem sizes."""
  if subPass != 0:
    raise StopIteration

  return f"""You are solving the Longest Common Substring problem in C++.

You must write a C++ solver that can handle ANY problem size from trivial to ludicrous scale:
- **Trivial**: 2-3 short strings (10-50 chars each)
- **Medium**: 3-5 medium strings (100-500 chars each)
- **Large**: 5-10 long strings (1000-5000 chars each)
- **Extreme**: 100 very long strings (10MB+ each), massive search space

**Input format (stdin):**
Line 1: N (number of strings)
Next N lines: one string per line

**Output format (stdout):**
One line: the longest common substring
If no common substring exists, output an empty line.

**Example:**
Input:
3
abcdef
xbcdey
zbcdew

Output:
bcde

**Critical Requirements:**
1. **Scalability**: Your algorithm must adapt based on number and length of strings
2. **Performance**: Must complete within 300 seconds even for very long strings
3. **Correctness**: Must find the actual longest common substring

**Environment:**
{describe_this_pc()}

**C++ Compiler:**
{CppCompiler("test_engine").describe()}

Be sure that any deviation from the C++ standard library is supported by the given compiler,
as referencing the wrong intrinsics or non-standard header like 'bits/stdc++.h' could fail your submission.

Write complete, compilable C++ code with a main() function.
Include adaptive logic that chooses different strategies based on problem scale.
"""


# List of subpasses to grade the single answer against all difficulty levels
extraGradeAnswerRuns = list(range(len(TEST_CASES)))

structure = {
  "type": "object",
  "properties": {
    "reasoning": {
      "type": "string",
      "description": "Explain your substring algorithm and how it adapts to different problem sizes"
    },
    "cpp_code": {
      "type": "string",
      "description": "Complete C++ code with main() that handles all scales"
    }
  },
  "required": ["reasoning", "cpp_code"],
  "additionalProperties": False
}


def is_common_substring(sub: str, strings: List[str]) -> bool:
  """Check if substring appears in all strings."""
  return all(sub in s for s in strings)


def find_lcs_naive(strings: List[str]) -> str:
  """Naive O(n²m) algorithm for reference."""
  if not strings:
    return ""

  shortest = min(strings, key=len)
  best = ""

  for i in range(len(shortest)):
    for j in range(i + 1, len(shortest) + 1):
      sub = shortest[i:j]
      if len(sub) > len(best) and is_common_substring(sub, strings):
        best = sub

  return best


def validate_solution(result: str, strings: List[str], expected: str) -> Tuple[bool, str, int]:
  """Validate LCS solution. Returns (is_valid, error, length)."""
  if not isinstance(result, str):
    return False, f"Result must be string, got {type(result).__name__}", 0

  # Check that result is actually a common substring
  if result and not is_common_substring(result, strings):
    return False, f"'{result[:50]}...' is not a common substring", 0

  return True, "", len(result)


STREAMING_THRESHOLD_CHARS = 1_000_000
_INPUT_FILE_CACHE = {}


def format_input(strings: List[str]) -> str:
  lines = [str(len(strings))]
  for s in strings:
    lines.append(s)
  return "\n".join(lines)


def _get_streaming_input(subpass: int, strings: List[str]) -> StreamingInputFile:
  if subpass in _INPUT_FILE_CACHE:
    return _INPUT_FILE_CACHE[subpass]

  total_len = sum(len(s) for s in strings)
  cache_key = f"lcs13|n={len(strings)}|total={total_len}|seed={RANDOM_SEED + subpass}"

  def generator():
    yield f"{len(strings)}\n"
    for s in strings:
      yield s + "\n"

  input_file = StreamingInputFile(cache_key, generator, "test13_lcs")
  _INPUT_FILE_CACHE[subpass] = input_file
  return input_file


def execute_solver(code: str,
                   strings: List[str],
                   subpass: int,
                   ai_engine_name: str,
                   timeout: int = TIMEOUT_SECONDS) -> tuple:
  """Execute the LLM's solver."""
  total_chars = sum(len(s) for s in strings)
  if total_chars > STREAMING_THRESHOLD_CHARS:
    streaming_input = _get_streaming_input(subpass, strings)
    input_file_path = streaming_input.generate()
    run = compile_and_run(code, "cpp", ai_engine_name, input_file=input_file_path, timeout=timeout)
  else:
    input_data = format_input(strings)
    run = compile_and_run(code, "cpp", ai_engine_name, input_data=input_data, timeout=timeout)

  if not run:
    return None, run.error_message(), run.exec_time

  result = run.stdout.strip()
  return result, None, run.exec_time


def gradeAnswer(result: dict, subPass: int, aiEngineName: str) -> tuple:
  """Grade the LCS solver."""
  if not result:
    return 0.0, "No result provided"

  if "cpp_code" not in result:
    return 0.0, "No C++ code provided"

  case = TEST_CASES[subPass]
  strings = case["strings"]()
  expected = case["expected"]()
  description = case["description"]
  code = result["cpp_code"]

  # Execute solver
  solution, error, exec_time = execute_solver(code, strings, subPass, aiEngineName)

  if error:
    return 0.0, f"[{description}] {error}"

  # Validate solution
  is_valid, validation_error, length = validate_solution(solution, strings, expected)

  if not is_valid:
    return 0.0, f"[{description}] Invalid: {validation_error}"

  expected_len = len(expected)

  # Score based on length found vs expected
  if length >= expected_len:
    score = 1.0
    quality = "optimal"
  elif length >= expected_len * 0.9:
    score = 0.85
    quality = "good (≥90% of optimal)"
  elif length >= expected_len * 0.7:
    score = 0.7
    quality = "acceptable (≥70% of optimal)"
  elif length > 0:
    score = 0.5
    quality = f"partial ({length}/{expected_len} chars)"
  else:
    score = 0.0
    quality = "no common substring found"

  explanation = (f"[{description}] Found: {length} chars, Expected: {expected_len} chars, "
                 f"Time: {exec_time:.1f}s - {quality}")

  return score, explanation


def resultToNiceReport(result: dict, subPass: int, aiEngineName: str) -> str:
  """Generate HTML report."""
  if not result:
    return "<p style='color:red'>No result provided</p>"

  case = TEST_CASES[subPass]

  html = f"<h4>Longest Common Substring - {case['description']}</h4>"

  if subPass == 0:
    if "reasoning" in result:
      reasoning = result['reasoning'][:500] + ('...'
                                               if len(result.get('reasoning', '')) > 500 else '')
      html += f"<p><strong>Algorithm:</strong> {reasoning}</p>"

    if "cpp_code" in result:
      code = result["cpp_code"]
      code_escaped = code.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
      html += f"<details><summary>View Code ({len(code)} chars)</summary><pre>{code_escaped}</pre></details>"

  return html


highLevelSummary = """
Longest Common Substring finds the longest contiguous sequence appearing in all input strings.

**Algorithms by complexity:**

1. **Naive O(n²·m):** Check all substrings of shortest string against others.

2. **DP O(n·m²):** Build suffix table, find longest common suffix ending at each position.

3. **Binary search + Rolling hash O(n·m·log m):**
   - Binary search on substring length
   - Use Rabin-Karp rolling hash to find all substrings of given length
   - Check if any hash appears in all strings

4. **Generalized Suffix Array O(n·m·log(n·m)):**
   - Concatenate strings with unique separators
   - Build suffix array and LCP array
   - Find longest common prefix spanning all original strings

**Key insight:** Binary search works because if a common substring of length k exists, 
then common substrings of all lengths < k also exist.

The baseline uses Python's difflib which works for small inputs but times out on large ones.
"""
