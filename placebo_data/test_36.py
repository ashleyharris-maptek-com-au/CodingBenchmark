"""Test 36: Max Independent Set - Placebo responses for all control types."""


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
  reasoning = 'Using greedy independent set: 1) Sort vertices by degree ascending. 2) Pick minimum degree vertex not adjacent to any selected. 3) Add to independent set, remove neighbors.'
  code = '#include <iostream>\n#include <vector>\n#include <set>\n#include <algorithm>\nusing namespace std;\nint main() {\n    int n, m; cin >> n >> m;\n    vector<set<int>> adj(n);\n    for (int i = 0; i < m; i++) {\n        int u, v; cin >> u >> v;\n        adj[u].insert(v); adj[v].insert(u);\n    }\n    set<int> indSet, available;\n    for (int i = 0; i < n; i++) available.insert(i);\n    while (!available.empty()) {\n        int best = -1, minDeg = n + 1;\n        for (int v : available) {\n            int deg = 0;\n            for (int u : adj[v]) if (available.count(u)) deg++;\n            if (deg < minDeg) { minDeg = deg; best = v; }\n        }\n        indSet.insert(best);\n        available.erase(best);\n        for (int u : adj[best]) available.erase(u);\n    }\n    cout << indSet.size() << "\\n";\n    for (int v : indSet) cout << v << " ";\n    cout << "\\n";\n    return 0;\n}'
  return {"reasoning": reasoning, 'cpp_code': code}, reasoning


def _naive_optimised(subpass):
  # TODO: Same algorithm as naive but hyper-optimised for C++
  # (parallel, SIMD, small data types, register packing, etc.)
  return _naive(subpass)


def _best_published(subpass):
  reasoning = (
    "Best published: Boppana-Halldorsson MIS (Boppana & Halldorsson 1992, BIT 32(2):180-196). "
    "TODO: Full implementation pending."
  )
  code = '// TODO: Implement Boppana-Halldorsson MIS'
  return {"reasoning": reasoning, 'cpp_code': code}, reasoning


def _random(subpass):
  reasoning = 'Random: pseudorandom output with fixed seed 42.'
  code = '#include <iostream>\n#include <random>\nusing namespace std;\nint main() {\n    mt19937 rng(42);\n    string line;\n    getline(cin, line);\n    cout << "0" << endl;\n    return 0;\n}'
  return {"reasoning": reasoning, 'cpp_code': code}, reasoning


def _human(subpass):
  reasoning = 'Human starting point for Max Independent Set. Fill in the TODOs.'
  code = '// TODO: Human attempt at Max Independent Set\n#include <iostream>\n#include <vector>\n#include <algorithm>\nusing namespace std;\nint main() {\n    ios_base::sync_with_stdio(false);\n    cin.tie(nullptr);\n    // TODO: Parse input\n    // TODO: Implement solution\n    // TODO: Output result\n    return 0;\n}'
  return {"reasoning": reasoning, 'cpp_code': code}, reasoning
