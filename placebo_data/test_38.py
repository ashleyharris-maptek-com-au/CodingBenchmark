"""Test 38: Exact Cover - Placebo responses for all control types."""


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
  reasoning = 'Using backtracking for exact cover: 1) Pick element with fewest covering sets (MRV). 2) Try each set containing that element. 3) Remove covered elements and conflicting sets. 4) Recurse, backtrack on failure.'
  code = 'def solve():\n    header = input().split()\n    universe_size, num_sets = int(header[0]), int(header[1])\n    sets = []\n    for i in range(num_sets):\n        s = set(map(int, input().split()))\n        sets.append(s)\n    universe = set(range(universe_size))\n    def backtrack(remaining, available, chosen):\n        if not remaining: return chosen\n        elem = min(remaining, key=lambda e: sum(1 for i in available if e in sets[i]))\n        for i in available:\n            if elem in sets[i]:\n                if sets[i] & remaining == sets[i] & universe:\n                    new_remaining = remaining - sets[i]\n                    new_available = [j for j in available if not (sets[j] & sets[i])]\n                    result = backtrack(new_remaining, new_available, chosen + [i])\n                    if result is not None: return result\n        return None\n    result = backtrack(universe, list(range(num_sets)), [])\n    if result: print("SOLUTION"); print(" ".join(map(str, result)))\n    else: print("NO SOLUTION")\nif __name__ == "__main__": solve()'
  return {"reasoning": reasoning, 'python_code': code}, reasoning


def _naive_optimised(subpass):
  # TODO: Same algorithm as naive but hyper-optimised for Python
  # (parallel, SIMD, small data types, register packing, etc.)
  return _naive(subpass)


def _best_published(subpass):
  reasoning = (
    "Best published: Algorithm X / dancing links (Knuth 2000, Millennial Perspectives in Computer Science). "
    "TODO: Full implementation pending."
  )
  code = '# TODO: Implement Algorithm X / dancing links'
  return {"reasoning": reasoning, 'python_code': code}, reasoning


def _random(subpass):
  reasoning = 'Random: pseudorandom output with fixed seed 42.'
  code = 'import random\nrandom.seed(42)\ndef solve():\n    header = input().split()\n    print("0")\nif __name__ == "__main__":\n    solve()'
  return {"reasoning": reasoning, 'python_code': code}, reasoning


def _human(subpass):
  reasoning = 'Human starting point for Exact Cover. Fill in the TODOs.'
  code = '# TODO: Human attempt at Exact Cover\ndef solve():\n    header = input().split()\n    # TODO: Parse input\n    # TODO: Implement solution\n    # TODO: Output result\n    print("0")\nif __name__ == "__main__":\n    solve()'
  return {"reasoning": reasoning, 'python_code': code}, reasoning
