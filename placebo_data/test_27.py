"""Test 27: Graph Coloring - Placebo responses for all control types."""


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
  reasoning = 'Using DSatur (saturation degree) greedy coloring: 1) Build adjacency list from edges. 2) Pick uncolored vertex with max saturation (distinct neighbor colors). 3) Assign smallest valid color not used by neighbors. 4) Repeat until all vertices colored.'
  code = '#include <iostream>\n#include <vector>\n#include <set>\nusing namespace std;\nint main() {\n    int n, m, k; cin >> n >> m >> k;\n    vector<set<int>> adj(n);\n    for (int i = 0; i < m; i++) {\n        int u, v; cin >> u >> v;\n        adj[u].insert(v); adj[v].insert(u);\n    }\n    vector<int> color(n, -1);\n    for (int i = 0; i < n; i++) {\n        set<int> used;\n        for (int nb : adj[i]) if (color[nb] >= 0) used.insert(color[nb]);\n        for (int c = 0; c < k; c++) {\n            if (used.find(c) == used.end()) { color[i] = c; break; }\n        }\n    }\n    for (int i = 0; i < n; i++) cout << color[i] << (i < n-1 ? " " : "\\n");\n    return 0;\n}'
  return {"reasoning": reasoning, 'cpp_code': code}, reasoning


def _naive_optimised(subpass):
  # TODO: Same algorithm as naive but hyper-optimised for C++
  # (parallel, SIMD, small data types, register packing, etc.)
  return _naive(subpass)


def _best_published(subpass):
  reasoning = (
    "Best published: DSatur coloring (Brelaz 1979, Communications of ACM 22(4):251-256). "
    "TODO: Full implementation pending."
  )
  code = '// TODO: Implement DSatur coloring'
  return {"reasoning": reasoning, 'cpp_code': code}, reasoning


def _random(subpass):
  reasoning = 'Random: pseudorandom output with fixed seed 42.'
  code = '#include <iostream>\n#include <random>\nusing namespace std;\nint main() {\n    mt19937 rng(42);\n    string line;\n    getline(cin, line);\n    cout << "0" << endl;\n    return 0;\n}'
  return {"reasoning": reasoning, 'cpp_code': code}, reasoning


def _human(subpass):
  reasoning = 'Human starting point for Graph Coloring. Fill in the TODOs.'
  code = '// TODO: Human attempt at Graph Coloring\n#include <iostream>\n#include <vector>\n#include <algorithm>\nusing namespace std;\nint main() {\n    ios_base::sync_with_stdio(false);\n    cin.tie(nullptr);\n    // TODO: Parse input\n    // TODO: Implement solution\n    // TODO: Output result\n    return 0;\n}'
  return {"reasoning": reasoning, 'cpp_code': code}, reasoning
