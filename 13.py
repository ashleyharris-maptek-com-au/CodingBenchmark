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
import subprocess
import sys
import tempfile
import os
import time
from typing import List, Tuple, Optional

title = "Longest Common Substring"

# Timeout in seconds (0.5 minutes)
TIMEOUT_SECONDS = 30

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

  return f"""You are solving the Longest Common Substring problem.

You must write a Python solver that can handle ANY problem size from trivial to ludicrous scale:
- **Trivial**: 2-3 short strings (10-50 chars each), simple cases
- **Medium**: 3-5 medium strings (100-500 chars each), moderate complexity
- **Large**: 5-10 long strings (1000-5000 chars each), complex patterns
- **Extreme**: 100 very long strings (10mb+ each), massive search space

**The Challenge:**
Your `longest_common_substring(strings)` function will be tested with problems ranging 
from a few short strings to many very long strings. The same function must work efficiently 
across ALL scales - from trivial to ludicrously large inputs, maintaining correctness and performance.

**Input:**
- `strings`: List of strings to find common substring in

**Output:**
- Longest common substring (as a string)
- If multiple substrings have same max length, return any one
- If no common substring exists (other than empty), return ""

**Critical Requirements:**
1. **Scalability**: Your algorithm must adapt based on number and length of strings
2. **Performance**: Must complete within 5 minutes even for very long strings
3. **Correctness**: Must find the actual longest common substring

**Example:**
```python
strings = ["abcdef", "xbcdey", "zbcdew"]
# Returns: "bcde" (length 4)
```

**Constraints:**
- Use only Python standard library
- Return the actual substring string, not just its length
- Must handle varying numbers and lengths of strings efficiently

Write complete, runnable Python code with the longest_common_substring function.
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
    "python_code": {
      "type":
      "string",
      "description":
      "Complete Python code with longest_common_substring(strings) function that handles all scales"
    }
  },
  "required": ["reasoning", "python_code"],
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


def execute_solver(code: str, strings: List[str], timeout: int = TIMEOUT_SECONDS) -> tuple:
  """Execute the LLM's solver."""
  from solver_utils import execute_solver_with_data

  data_dict = {
    'strings': strings,
  }

  result, error, exec_time = execute_solver_with_data(code, data_dict, 'longest_common_substring',
                                                      timeout)

  if error:
    return None, error, exec_time

  if not isinstance(result, str):
    return None, f"Invalid result type: expected str, got {type(result).__name__}", exec_time

  return result, None, exec_time


def gradeAnswer(result: dict, subPass: int, aiEngineName: str) -> tuple:
  """Grade the LCS solver."""
  if not result:
    return 0.0, "No result provided"

  if "python_code" not in result:
    return 0.0, "No Python code provided"

  case = TEST_CASES[subPass]
  strings = case["strings"]()
  expected = case["expected"]()
  description = case["description"]
  code = result["python_code"]

  # Execute solver
  solution, error, exec_time = execute_solver(code, strings)

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

    if "python_code" in result:
      code = result["python_code"]
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
