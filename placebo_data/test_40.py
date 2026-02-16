"""Test 40: Integer Linear Programming - Placebo responses for all control types."""


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
  reasoning = 'Using greedy heuristic for ILP: 1) Sort variables by objective coefficient / constraint usage. 2) Greedily increase each variable while maintaining feasibility. 3) Round to integers.'
  code = '#include <iostream>\n#include <vector>\n#include <algorithm>\nusing namespace std;\nint main() {\n    int n, m; cin >> n >> m;\n    vector<int> c(n), upper(n);\n    for (int i = 0; i < n; i++) cin >> c[i];\n    for (int i = 0; i < n; i++) cin >> upper[i];\n    vector<vector<int>> A(m, vector<int>(n));\n    vector<int> b(m);\n    for (int i = 0; i < m; i++) {\n        for (int j = 0; j < n; j++) cin >> A[i][j];\n        cin >> b[i];\n    }\n    vector<pair<double, int>> eff(n);\n    for (int j = 0; j < n; j++) {\n        double usage = 1;\n        for (int i = 0; i < m; i++) usage += A[i][j];\n        eff[j] = {(double)c[j] / usage, j};\n    }\n    sort(eff.rbegin(), eff.rend());\n    vector<int> x(n, 0);\n    for (auto [_, j] : eff) {\n        int maxInc = upper[j];\n        for (int i = 0; i < m; i++) {\n            if (A[i][j] > 0) {\n                int lhs = 0;\n                for (int k = 0; k < n; k++) lhs += A[i][k] * x[k];\n                maxInc = min(maxInc, (b[i] - lhs) / A[i][j]);\n            }\n        }\n        x[j] = max(0, maxInc);\n    }\n    long long obj = 0;\n    for (int j = 0; j < n; j++) obj += c[j] * x[j];\n    cout << obj << "\\n";\n    for (int j = 0; j < n; j++) cout << x[j] << (j < n-1 ? " " : "\\n");\n    return 0;\n}'
  return {"reasoning": reasoning, 'cpp_code': code}, reasoning


def _naive_optimised(subpass):
  # TODO: Same algorithm as naive but hyper-optimised for C++
  # (parallel, SIMD, small data types, register packing, etc.)
  return _naive(subpass)


def _best_published(subpass):
  reasoning = (
    "Best published: Branch-and-bound ILP (Land & Doig 1960, Econometrica 28(3):497-520). "
    "TODO: Full implementation pending."
  )
  code = '// TODO: Implement Branch-and-bound ILP'
  return {"reasoning": reasoning, 'cpp_code': code}, reasoning


def _random(subpass):
  reasoning = 'Random: pseudorandom output with fixed seed 42.'
  code = '#include <iostream>\n#include <random>\nusing namespace std;\nint main() {\n    mt19937 rng(42);\n    string line;\n    getline(cin, line);\n    cout << "0" << endl;\n    return 0;\n}'
  return {"reasoning": reasoning, 'cpp_code': code}, reasoning


def _human(subpass):
  reasoning = 'Human starting point for Integer Linear Programming. Fill in the TODOs.'
  code = '// TODO: Human attempt at Integer Linear Programming\n#include <iostream>\n#include <vector>\n#include <algorithm>\nusing namespace std;\nint main() {\n    ios_base::sync_with_stdio(false);\n    cin.tie(nullptr);\n    // TODO: Parse input\n    // TODO: Implement solution\n    // TODO: Output result\n    return 0;\n}'
  return {"reasoning": reasoning, 'cpp_code': code}, reasoning
