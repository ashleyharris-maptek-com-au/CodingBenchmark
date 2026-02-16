"""Test 1: TSP - Placebo responses for all control types."""


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
  reasoning = 'Nearest-neighbor heuristic in C++. O(n^2), no optimisation. Times out on large inputs.'
  code = '#include <iostream>\n#include <vector>\n#include <cmath>\nusing namespace std;\nint main() {\n    int n; cin >> n;\n    vector<double> x(n), y(n);\n    for (int i = 0; i < n; i++) cin >> x[i] >> y[i];\n    vector<bool> vis(n, false);\n    vis[0] = true;\n    int cur = 0;\n    cout << 0;\n    for (int step = 1; step < n; step++) {\n        int best = -1; double bd = 1e18;\n        for (int j = 0; j < n; j++) {\n            if (!vis[j]) {\n                double d = sqrt((x[cur]-x[j])*(x[cur]-x[j])+(y[cur]-y[j])*(y[cur]-y[j]));\n                if (d < bd) { bd = d; best = j; }\n            }\n        }\n        vis[best] = true; cur = best;\n        cout << " " << best;\n    }\n    cout << endl;\n}'
  return {"reasoning": reasoning, 'cpp_code': code}, reasoning


def _naive_optimised(subpass):
  # TODO: Same algorithm as naive but hyper-optimised for C++
  # (parallel, SIMD, small data types, register packing, etc.)
  return _naive(subpass)


def _best_published(subpass):
  reasoning = (
    "Best published: Lin-Kernighan heuristic (Lin & Kernighan 1973, Operations Research 21(2):498-516). "
    "TODO: Full implementation pending."
  )
  code = '// TODO: Implement Lin-Kernighan heuristic'
  return {"reasoning": reasoning, 'cpp_code': code}, reasoning


def _random(subpass):
  reasoning = 'Random: pseudorandom output with fixed seed 42.'
  code = '#include <iostream>\n#include <random>\nusing namespace std;\nint main() {\n    mt19937 rng(42);\n    string line;\n    getline(cin, line);\n    cout << "0" << endl;\n    return 0;\n}'
  return {"reasoning": reasoning, 'cpp_code': code}, reasoning


def _human(subpass):
  reasoning = 'Human starting point for TSP. Fill in the TODOs.'
  code = '// TODO: Human attempt at TSP\n#include <iostream>\n#include <vector>\n#include <algorithm>\nusing namespace std;\nint main() {\n    ios_base::sync_with_stdio(false);\n    cin.tie(nullptr);\n    // TODO: Parse input\n    // TODO: Implement solution\n    // TODO: Output result\n    return 0;\n}'
  return {"reasoning": reasoning, 'cpp_code': code}, reasoning
