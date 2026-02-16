"""Test 30: 3-SAT Solver - Placebo responses for all control types."""


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
  reasoning = 'Using DPLL with unit propagation: 1) Unit propagation: if clause has one unset literal, set it. 2) Pure literal elimination. 3) Choose unset variable, try both values. 4) Backtrack on conflict.'
  code = 'import sys\ndef solve():\n    line = input().split()\n    n, m = int(line[0]), int(line[1])\n    clauses = []\n    for _ in range(m):\n        clauses.append(list(map(int, input().split())))\n    assign = [None] * (n + 1)\n    def check():\n        for c in clauses:\n            sat = False\n            for lit in c:\n                v = abs(lit)\n                if assign[v] is not None:\n                    if (lit > 0 and assign[v]) or (lit < 0 and not assign[v]):\n                        sat = True; break\n            if not sat and all(assign[abs(l)] is not None for l in c):\n                return False\n        return True\n    def dpll(var):\n        if var > n:\n            return all(any((l > 0 and assign[abs(l)]) or (l < 0 and not assign[abs(l)]) for l in c) for c in clauses)\n        for val in [True, False]:\n            assign[var] = val\n            if check() and dpll(var + 1): return True\n        assign[var] = None\n        return False\n    if dpll(1):\n        print("SAT")\n        print(" ".join("1" if assign[i] else "0" for i in range(1, n + 1)))\n    else:\n        print("UNSAT")\nif __name__ == "__main__": solve()'
  return {"reasoning": reasoning, 'python_code': code}, reasoning


def _naive_optimised(subpass):
  # TODO: Same algorithm as naive but hyper-optimised for Python
  # (parallel, SIMD, small data types, register packing, etc.)
  return _naive(subpass)


def _best_published(subpass):
  reasoning = (
    "Best published: CDCL SAT solver (Marques-Silva & Sakallah 1999, IEEE Trans. Computers 48(5):506-521). "
    "TODO: Full implementation pending."
  )
  code = '# TODO: Implement CDCL SAT solver'
  return {"reasoning": reasoning, 'python_code': code}, reasoning


def _random(subpass):
  reasoning = 'Random: pseudorandom output with fixed seed 42.'
  code = 'import random\nrandom.seed(42)\ndef solve():\n    header = input().split()\n    print("0")\nif __name__ == "__main__":\n    solve()'
  return {"reasoning": reasoning, 'python_code': code}, reasoning


def _human(subpass):
  reasoning = 'Human starting point for 3-SAT Solver. Fill in the TODOs.'
  code = '# TODO: Human attempt at 3-SAT Solver\ndef solve():\n    header = input().split()\n    # TODO: Parse input\n    # TODO: Implement solution\n    # TODO: Output result\n    print("0")\nif __name__ == "__main__":\n    solve()'
  return {"reasoning": reasoning, 'python_code': code}, reasoning
