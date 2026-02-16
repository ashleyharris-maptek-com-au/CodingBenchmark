"""Test 13: Longest Common Substring - Placebo responses for all control types."""


def get_response(model_name, subpass):
  """Return (result_dict, reasoning_string) for the given control type."""
  if model_name == 'naive':
    return _naive(subpass)
  elif model_name == 'naive-optimised':
    return _naive_optimised(subpass)
  elif model_name == 'best-published':
    return _best_published(subpass)
  elif model_name == 'random':
    return _random(subpass)
  elif model_name == 'human':
    return _human(subpass)
  return None, ''


def _naive(subpass):
  reasoning = "Using Python's difflib.SequenceMatcher for LCS: 1) Start with first string as reference. 2) Find longest match between current result and each subsequent string. 3) Update result to the common substring found. 4) This uses difflib's efficient matching but may timeout on very large inputs."
  code = 'from difflib import SequenceMatcher\n\ndef longest_common_substring(strings):\n    """\n    Find longest common substring using difflib.SequenceMatcher.\n    """\n    if not strings:\n        return ""\n    \n    if len(strings) == 1:\n        return strings[0]\n    \n    def lcs_two(s1, s2):\n        """Find LCS of two strings using SequenceMatcher."""\n        matcher = SequenceMatcher(None, s1, s2)\n        match = matcher.find_longest_match(0, len(s1), 0, len(s2))\n        return s1[match.a:match.a + match.size]\n    \n    # Start with all substrings of first string as candidates\n    # Then filter by checking each subsequent string\n    result = strings[0]\n    \n    for s in strings[1:]:\n        result = lcs_two(result, s)\n        if not result:\n            return ""\n    \n    return result\n'
  return {"reasoning": reasoning, 'cpp_code': code}, reasoning


def _naive_optimised(subpass):
  # TODO: Same algorithm as naive but hyper-optimised for C++
  # (parallel, SIMD, small data types, register packing, etc.)
  return _naive(subpass)


def _best_published(subpass):
  reasoning = (
    "Best published: Suffix array LCS (Kasai et al. 2001, ALENEX 2001:181-192). "
    "TODO: Full implementation pending."
  )
  code = '// TODO: Implement Suffix array LCS'
  return {"reasoning": reasoning, 'cpp_code': code}, reasoning


def _random(subpass):
  reasoning = 'Random: pseudorandom output with fixed seed 42.'
  code = '#include <iostream>\n#include <random>\nusing namespace std;\nint main() {\n    mt19937 rng(42);\n    string line;\n    getline(cin, line);\n    cout << "0" << endl;\n    return 0;\n}'
  return {"reasoning": reasoning, 'cpp_code': code}, reasoning


def _human(subpass):
  reasoning = 'Human starting point for Longest Common Substring. Fill in the TODOs.'
  code = '// TODO: Human attempt at Longest Common Substring\n#include <iostream>\n#include <vector>\n#include <algorithm>\nusing namespace std;\nint main() {\n    ios_base::sync_with_stdio(false);\n    cin.tie(nullptr);\n    // TODO: Parse input\n    // TODO: Implement solution\n    // TODO: Output result\n    return 0;\n}'
  return {"reasoning": reasoning, 'cpp_code': code}, reasoning
