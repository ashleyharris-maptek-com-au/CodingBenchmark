# TSP Benchmark

A programming benchmark that tests LLM capabilities in algorithm design and optimization using the classic Travelling Salesman Problem.

## Overview

This benchmark evaluates whether LLMs can:
- Write efficient algorithms that scale
- Discover and implement heuristics for NP-hard problems
- Optimize code to meet time constraints

The LLM must write a Python solver for TSP that handles varying city counts within a **5-minute timeout**.

## Test Structure

### Test 1: Travelling Salesman Problem

**Subpasses by city count:**
| Subpass | Cities | Difficulty |
|---------|--------|------------|
| 0 | 10 | Easy - brute force possible |
| 1 | 20 | Medium - needs heuristics |
| 2 | 30 | Medium-Hard |
| 3 | 40 | Hard |
| 4 | 100 | Very Hard |
| 5 | 1000 | Extreme - needs efficient implementation |

**Scoring:**
- **1.0**: Route within 10% of baseline (excellent)
- **0.85**: Route within 50% of baseline (good)
- **0.7**: Route within 2x baseline (acceptable)
- **0.5**: Valid route but poor quality
- **0.0**: Invalid route or solver error/timeout

## Installation

```bash
# Clone with submodules
git clone --recursive <repo-url>

# Install dependencies
pip install -r requirements.txt
```

## Usage

```bash
# Run all tests on all available models
python TSPBenchmark.py

# Run specific models
python TSPBenchmark.py -m gpt-5-nano
python TSPBenchmark.py -m "claude-*"

# Run in parallel
python TSPBenchmark.py --parallel

# List available models
python TSPBenchmark.py --list-models

# Force bypass cache
python TSPBenchmark.py --force
```

## Baseline (Placebo)

The "Human with tools" baseline uses a naive **nearest-neighbor heuristic**:
- O(n²) time complexity
- No optimizations (2-opt, etc.)
- Provides a reasonable approximation

LLMs are expected to match or beat this baseline.

## What We're Measuring

1. **Algorithm Knowledge**: Does the LLM know TSP heuristics?
2. **Code Quality**: Can it write correct, runnable Python?
3. **Scalability Awareness**: Does it adapt approach for larger inputs?
4. **Optimization Skills**: Can it implement efficient data structures?

## Expected Approaches

Successful solvers typically implement:
- **Nearest Neighbor**: O(n²), simple greedy
- **2-opt**: Local search improvement
- **Christofides**: Near-optimal for metric TSP
- **Genetic Algorithms**: Population-based optimization
- **Simulated Annealing**: Probabilistic local search

## License

MIT License - see LLMBenchCore/LICENSE for details.
