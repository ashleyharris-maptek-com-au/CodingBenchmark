#!/usr/bin/env python3
"""
CodingBenchmark - Algorithm Design and Optimization LLM Benchmark

Tests LLM capabilities in algorithm design and optimization.
The LLM must write a Python solver for various complicated software
engineering tasks. All these tasks require efficient algorithmic thinking,
as a dumb solution is possible, 

This extends the abstract TestRunner framework with TSP-specific:
- Code execution and validation
- Route distance calculation
- Timeout enforcement
"""

import sys
import os
import re

# Add LLMBenchCore to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'LLMBenchCore'))

from LLMBenchCore import BenchmarkRunner, run_benchmark_main
from LLMBenchCore.AiEnginePlacebo import set_placebo_data_provider
from placebo_data import get_placebo_response

_VOLATILE_PROMPT_LINE_RE = re.compile(
  r"^(Platform|Architecture|CPU|Core count|Memory|SIMD|SIMD detail|Node\.js): .*$",
  re.MULTILINE)
_COMPILER_PROMPT_LINE_RE = re.compile(
  r"^(\*\*(?:C\+\+|Rust|C#) Compiler(?: information)?(?::)?\*\*)\s*\r?\n[^\r\n]*",
  re.MULTILINE)


def _normalize_prompt_for_cache(prompt: str) -> str:
  prompt = str(prompt)
  prompt = _VOLATILE_PROMPT_LINE_RE.sub(lambda m: f"{m.group(1)}: <cache-ignored>", prompt)
  prompt = _COMPILER_PROMPT_LINE_RE.sub(lambda m: f"{m.group(1)}\n<cache-ignored>", prompt)
  return prompt.strip()


class TSPBenchmarkRunner(BenchmarkRunner):
  """
    Algorithm Design and Optimization Benchmark runner.
    
    Tests whether LLMs can design and optimize algorithms for complex software engineering tasks.
    """

  def get_benchmark_title(self) -> str:
    return "Algorithm Design and Optimization Benchmark"

  def get_benchmark_subtitle(self) -> str:
    return "Complex Software Engineering Problems - Algorithm Optimization"

  def get_benchmark_description(self) -> str:
    return """<p>Can LLMs design and optimize algorithms for complex software engineering tasks?</p>
        <p>The LLM must write efficient solutions that scale well and meet performance requirements.</p>"""

  def normalize_prompt_for_cache(self, prompt: str) -> str:
    return _normalize_prompt_for_cache(prompt)


if __name__ == "__main__":
  # Register placebo data provider for "Human with tools" baseline
  set_placebo_data_provider([
    {"name": "naive", "description": "Simple, obvious approach (~20 lines). No optimisation. Times out on large inputs."},
    {"name": "naive-optimised", "description": "Same algorithm as naive, but hyper-optimised (parallel, SIMD, small types)."},
    {"name": "best-published", "description": "References a published paper/algorithm."},
    {"name": "random", "description": "Lets just guess."},
    {"name": "human", "description": "A professional human software engineer spent ~1hr trying to solve it."},
  ], get_placebo_response)

  runner = TSPBenchmarkRunner()
  run_benchmark_main(runner, __file__)
