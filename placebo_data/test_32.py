"""Test 32: Steiner Tree - Placebo responses for all control types."""


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
  reasoning = 'Using MST-based 2-approximation for Steiner tree: 1) Compute shortest paths between all terminals. 2) Build complete graph on terminals with shortest path distances. 3) Find MST of terminal graph. 4) Map back to original edges.'
  code = '#include <iostream>\n#include <vector>\n#include <queue>\n#include <set>\nusing namespace std;\nint main() {\n    int n, m, t; cin >> n >> m >> t;\n    vector<vector<pair<int,int>>> adj(n);\n    for (int i = 0; i < m; i++) {\n        int u, v, w; cin >> u >> v >> w;\n        adj[u].push_back({v, w}); adj[v].push_back({u, w});\n    }\n    vector<int> terms(t);\n    for (int i = 0; i < t; i++) cin >> terms[i];\n    set<int> termSet(terms.begin(), terms.end());\n    // Simple: just connect terminals greedily via Dijkstra\n    vector<pair<int,int>> treeEdges;\n    set<int> inTree; inTree.insert(terms[0]);\n    int totalWeight = 0;\n    while (inTree.size() < termSet.size()) {\n        int bestU = -1, bestV = -1, bestW = 1e9;\n        for (int s : inTree) {\n            vector<int> dist(n, 1e9); dist[s] = 0;\n            priority_queue<pair<int,int>, vector<pair<int,int>>, greater<>> pq;\n            pq.push({0, s});\n            while (!pq.empty()) {\n                auto [d, u] = pq.top(); pq.pop();\n                if (d > dist[u]) continue;\n                for (auto [v, w] : adj[u]) {\n                    if (dist[u] + w < dist[v]) { dist[v] = dist[u] + w; pq.push({dist[v], v}); }\n                }\n            }\n            for (int v : terms) {\n                if (inTree.find(v) == inTree.end() && dist[v] < bestW) {\n                    bestW = dist[v]; bestU = s; bestV = v;\n                }\n            }\n        }\n        if (bestV >= 0) { inTree.insert(bestV); totalWeight += bestW; treeEdges.push_back({bestU, bestV}); }\n    }\n    cout << totalWeight << " " << treeEdges.size() << "\\n";\n    for (auto [u, v] : treeEdges) cout << u << " " << v << "\\n";\n    return 0;\n}'
  return {"reasoning": reasoning, 'cpp_code': code}, reasoning


def _naive_optimised(subpass):
  # TODO: Same algorithm as naive but hyper-optimised for C++
  # (parallel, SIMD, small data types, register packing, etc.)
  return _naive(subpass)


def _best_published(subpass):
  reasoning = (
    "Best published: Dreyfus-Wagner Steiner tree (Dreyfus & Wagner 1971, Networks 1(3):195-207). "
    "TODO: Full implementation pending."
  )
  code = '// TODO: Implement Dreyfus-Wagner Steiner tree'
  return {"reasoning": reasoning, 'cpp_code': code}, reasoning


def _random(subpass):
  reasoning = 'Random: pseudorandom output with fixed seed 42.'
  code = '#include <iostream>\n#include <random>\nusing namespace std;\nint main() {\n    mt19937 rng(42);\n    string line;\n    getline(cin, line);\n    cout << "0" << endl;\n    return 0;\n}'
  return {"reasoning": reasoning, 'cpp_code': code}, reasoning


def _human(subpass):
  reasoning = 'Human starting point for Steiner Tree. Fill in the TODOs.'
  code = '// TODO: Human attempt at Steiner Tree\n#include <iostream>\n#include <vector>\n#include <algorithm>\nusing namespace std;\nint main() {\n    ios_base::sync_with_stdio(false);\n    cin.tie(nullptr);\n    // TODO: Parse input\n    // TODO: Implement solution\n    // TODO: Output result\n    return 0;\n}'
  return {"reasoning": reasoning, 'cpp_code': code}, reasoning
