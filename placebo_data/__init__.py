"""
Placebo data provider for Algorithm Benchmarks.

Provides pre-computed responses for multiple control types:
  - naive:           Simple, dumb approach (<20 lines), no optimisation. Times out on large data.
  - naive-optimised: Same algorithm as naive, but hyper-optimised (parallel, SIMD, small types).
  - best-published:  References a published paper/algorithm. Partial attempts with TODOs.
  - random:          Solves using only seeded pseudorandom generation.
  - human:           Starting point / skeleton for a human attempt.

Each test has its own module: placebo_data/test_NN.py
"""

from importlib import import_module
from typing import Optional, Union, Tuple

# Cache for loaded test modules
_module_cache = {}


def get_placebo_response(model_name: str, question_num: int,
                         subpass: int) -> Tuple[Optional[Union[dict, str]], str]:
  """
  Placebo data provider function.

  Args:
      model_name: The control type name (e.g. 'naive', 'naive-optimised', etc.)
      question_num: The test number (1-45)
      subpass: The subpass index

  Returns:
      Tuple of (result_dict, reasoning_string)
  """
  # Load the per-test module
  mod_name = f"test_{question_num:02d}"
  if mod_name not in _module_cache:
    try:
      _module_cache[mod_name] = import_module(f".{mod_name}", package="placebo_data")
    except ImportError:
      return None, f"No placebo data for test {question_num}"

  mod = _module_cache[mod_name]
  try:
    return mod.get_response(model_name, subpass)
  except Exception as e:
    return None, f"Error getting placebo response: {e}"
