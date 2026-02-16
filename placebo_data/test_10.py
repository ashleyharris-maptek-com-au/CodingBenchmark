"""Test 10: 2D Cutting Stock - Placebo responses for all control types."""


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
  reasoning = 'Shelf FFDH for 2D rectangle packing in C++. Sorts by height, packs into shelves.'
  code = '#include <iostream>\n#include <vector>\n#include <algorithm>\nusing namespace std;\nint main() {\n    int n; double W, H;\n    cin >> n >> W >> H;\n    vector<double> w(n), h(n);\n    for (int i = 0; i < n; i++) cin >> w[i] >> h[i];\n    vector<int> idx(n); iota(idx.begin(), idx.end(), 0);\n    sort(idx.begin(), idx.end(), [&](int a, int b){ return h[a] > h[b]; });\n    // Shelves: (y_start, shelf_height, x_used)\n    vector<tuple<double,double,double>> shelves;\n    vector<tuple<int,int,double,double>> placements; // board, idx, x, y\n    int boards = 1;\n    shelves.push_back({0, 0, 0});\n    for (int i : idx) {\n        bool placed = false;\n        for (auto& [sy,sh,sx] : shelves) {\n            if (sx + w[i] <= W && (sh == 0 || h[i] <= sh)) {\n                if (sh == 0) sh = h[i];\n                placements.push_back({0, i, sx, sy});\n                sx += w[i]; placed = true; break;\n            }\n        }\n        if (!placed) {\n            double ny = 0; for (auto& [sy,sh,sx] : shelves) ny = max(ny, sy+sh);\n            if (ny + h[i] <= H) {\n                shelves.push_back({ny, h[i], w[i]});\n                placements.push_back({0, i, 0, ny});\n            } else {\n                boards++;\n                shelves.clear(); shelves.push_back({0, h[i], w[i]});\n                placements.push_back({boards-1, i, 0, 0});\n            }\n        }\n    }\n    cout << boards << endl;\n    for (auto& [b, id, x, y] : placements)\n        cout << b << " " << id << " " << x << " " << y << endl;\n}'
  return {"reasoning": reasoning, 'cpp_code': code}, reasoning


def _naive_optimised(subpass):
  # TODO: Same algorithm as naive but hyper-optimised for C++
  # (parallel, SIMD, small data types, register packing, etc.)
  return _naive(subpass)


def _best_published(subpass):
  reasoning = (
    "Best published: Lodi et al. 2D bin packing (Lodi et al. 2002, European J. OR 141(2):241-252). "
    "TODO: Full implementation pending."
  )
  code = '// TODO: Implement Lodi et al. 2D bin packing'
  return {"reasoning": reasoning, 'cpp_code': code}, reasoning


def _random(subpass):
  reasoning = 'Random: pseudorandom output with fixed seed 42.'
  code = '#include <iostream>\n#include <random>\nusing namespace std;\nint main() {\n    mt19937 rng(42);\n    string line;\n    getline(cin, line);\n    cout << "0" << endl;\n    return 0;\n}'
  return {"reasoning": reasoning, 'cpp_code': code}, reasoning


def _human(subpass):
  reasoning = 'Human starting point for 2D Cutting Stock. Fill in the TODOs.'
  code = '// TODO: Human attempt at 2D Cutting Stock\n#include <iostream>\n#include <vector>\n#include <algorithm>\nusing namespace std;\nint main() {\n    ios_base::sync_with_stdio(false);\n    cin.tie(nullptr);\n    // TODO: Parse input\n    // TODO: Implement solution\n    // TODO: Output result\n    return 0;\n}'
  return {"reasoning": reasoning, 'cpp_code': code}, reasoning
