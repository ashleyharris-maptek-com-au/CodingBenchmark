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
from typing import List, Set, Tuple

title = "Exact Cover Problem (Python)"
TIMEOUT_SECONDS = 30
RANDOM_SEED = 38383838


def generate_exact_cover(universe_size: int, num_sets: int,
                         set_size_range: Tuple[int, int],
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
]

INSTANCE_CACHE = {}


def get_instance(subpass: int):
    if subpass not in INSTANCE_CACHE:
        case = TEST_CASES[subpass]
        universe, sets = generate_exact_cover(case["universe"], case["sets"],
                                              case["size_range"],
                                              RANDOM_SEED + subpass)
        INSTANCE_CACHE[subpass] = (universe, sets)
    return INSTANCE_CACHE[subpass]


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
},
        "python_code": {
            "type": "string"
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
    universe, sets = get_instance(subPass)
    input_data = format_input(universe, sets)

    try:
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


def output_example_html(score: float, explanation: str, result: dict,
                        subPass: int) -> str:
    case = TEST_CASES[subPass]
    code = result.get("python_code", "").replace("&", "&amp;").replace(
        "<", "&lt;").replace(">", "&gt;")
    color = "green" if score >= 0.8 else "orange" if score >= 0.4 else "red"
    return f'<div class="result"><h4>Subpass {subPass}: {case["desc"]}</h4><p style="color:{color}">Score: {score:.2f}</p><p>{explanation}</p><details><summary>Code</summary><pre>{code}</pre></details></div>'


def output_header_html() -> str:
    return "<h2>Test 38: Exact Cover Problem (Python)</h2><p>NP-Complete set partitioning problem.</p>"


def output_summary_html(results: list) -> str:
    total = sum(r[0] for r in results)
    return f'<div class="summary"><p>Total: {total:.2f}/{len(results)}</p></div>'
