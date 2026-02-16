"""Test 18: Minimum Cut - Placebo responses for all control types."""


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
  reasoning = 'Using Stoer-Wagner algorithm for minimum cut in C++: 1) Stoer-Wagner finds global min-cut in O(VE + VÂ² log V) time. 2) Iteratively contract vertices while tracking minimum cut. 3) Use adjacency list with edge weights for efficiency. 4) Track which edges form the minimum cut found.'
  code = '#include <iostream>\n#include <vector>\n#include <queue>\n#include <algorithm>\n#include <climits>\n#include <cstring>\nusing namespace std;\n\nconst int MAXN = 50005;\nint n, m;\nvector<pair<int, int>> adj[MAXN];  // {neighbor, edge_index}\nvector<pair<int, int>> edges;  // original edges\nbool merged[MAXN];\nint parent[MAXN];\nint weight[MAXN];  // weight to contracted graph\nbool inA[MAXN];\n\n// Union-Find for vertex merging\nint uf_find(int x) {\n    if (parent[x] != x) parent[x] = uf_find(parent[x]);\n    return parent[x];\n}\n\nvoid uf_union(int x, int y) {\n    parent[uf_find(x)] = uf_find(y);\n}\n\n// One phase of Stoer-Wagner\npair<int, pair<int,int>> minCutPhase(int start, vector<int>& active) {\n    fill(weight, weight + n, 0);\n    fill(inA, inA + n, false);\n    \n    priority_queue<pair<int,int>> pq;\n    int s = -1, t = -1;\n    int lastWeight = 0;\n    \n    for (int v : active) {\n        pq.push({0, v});\n    }\n    \n    int added = 0;\n    while (!pq.empty() && added < (int)active.size()) {\n        auto [w, u] = pq.top();\n        pq.pop();\n        \n        if (inA[u]) continue;\n        inA[u] = true;\n        added++;\n        s = t;\n        t = u;\n        lastWeight = -w;\n        \n        for (auto [v, eidx] : adj[u]) {\n            int rv = uf_find(v);\n            if (!inA[rv] && !merged[rv]) {\n                weight[rv]++;\n                pq.push({-weight[rv], rv});\n            }\n        }\n    }\n    \n    return {lastWeight, {s, t}};\n}\n\nint main() {\n    ios_base::sync_with_stdio(false);\n    cin.tie(nullptr);\n    \n    cin >> n >> m;\n    edges.resize(m);\n    \n    for (int i = 0; i < m; i++) {\n        int u, v;\n        cin >> u >> v;\n        edges[i] = {u, v};\n        adj[u].push_back({v, i});\n        adj[v].push_back({u, i});\n    }\n    \n    // Initialize union-find\n    for (int i = 0; i < n; i++) {\n        parent[i] = i;\n        merged[i] = false;\n    }\n    \n    int minCut = INT_MAX;\n    int bestT = -1;\n    vector<int> component;  // vertices in the cut component\n    \n    vector<int> active;\n    for (int i = 0; i < n; i++) active.push_back(i);\n    \n    // Run n-1 phases\n    for (int phase = 0; phase < n - 1 && active.size() > 1; phase++) {\n        auto [cutWeight, st] = minCutPhase(active[0], active);\n        int s = st.first, t = st.second;\n        \n        if (cutWeight < minCut && cutWeight > 0) {\n            minCut = cutWeight;\n            bestT = t;\n            // Remember component containing t\n            component.clear();\n            component.push_back(t);\n        }\n        \n        // Merge s and t\n        if (s >= 0 && t >= 0) {\n            uf_union(t, s);\n            merged[t] = true;\n            \n            // Update active list\n            active.erase(remove(active.begin(), active.end(), t), active.end());\n        }\n    }\n    \n    // Find cut edges - edges between component containing bestT and rest\n    if (minCut == INT_MAX || minCut == 0) {\n        // Graph might already be disconnected or single node\n        // Find any edge that can disconnect\n        if (m > 0) {\n            cout << 1 << "\\n";\n            cout << edges[0].first << " " << edges[0].second << "\\n";\n        } else {\n            cout << 0 << "\\n";\n        }\n        return 0;\n    }\n    \n    // Simple approach: find edges crossing to isolated vertex group\n    // Use BFS to find actual components after removing minimum edges\n    vector<pair<int,int>> cutEdges;\n    \n    // For simplicity, find edges incident to bestT in contracted graph\n    // This is approximate but fast\n    for (int i = 0; i < m && (int)cutEdges.size() < minCut; i++) {\n        int u = edges[i].first, v = edges[i].second;\n        // Check if edge crosses between original bestT group and others\n        if ((u == bestT || v == bestT) && uf_find(u) != uf_find(v)) {\n            cutEdges.push_back(edges[i]);\n        }\n    }\n    \n    // If we didn\'t find enough, just output first few edges\n    if (cutEdges.empty()) {\n        for (int i = 0; i < min(m, minCut); i++) {\n            cutEdges.push_back(edges[i]);\n        }\n    }\n    \n    cout << cutEdges.size() << "\\n";\n    for (auto [u, v] : cutEdges) {\n        cout << u << " " << v << "\\n";\n    }\n    \n    return 0;\n}\n'
  return {"reasoning": reasoning, 'cpp_code': code}, reasoning


def _naive_optimised(subpass):
  # TODO: Same algorithm as naive but hyper-optimised for C++
  # (parallel, SIMD, small data types, register packing, etc.)
  return _naive(subpass)


def _best_published(subpass):
  reasoning = (
    "Best published: Stoer-Wagner minimum cut (Stoer & Wagner 1997, J. ACM 44(4):585-591). "
    "TODO: Full implementation pending."
  )
  code = '// TODO: Implement Stoer-Wagner minimum cut'
  return {"reasoning": reasoning, 'cpp_code': code}, reasoning


def _random(subpass):
  reasoning = 'Random: pseudorandom output with fixed seed 42.'
  code = '#include <iostream>\n#include <random>\nusing namespace std;\nint main() {\n    mt19937 rng(42);\n    string line;\n    getline(cin, line);\n    cout << "0" << endl;\n    return 0;\n}'
  return {"reasoning": reasoning, 'cpp_code': code}, reasoning


def _human(subpass):
  reasoning = 'Human starting point for Minimum Cut. Fill in the TODOs.'
  code = '// TODO: Human attempt at Minimum Cut\n#include <iostream>\n#include <vector>\n#include <algorithm>\nusing namespace std;\nint main() {\n    ios_base::sync_with_stdio(false);\n    cin.tie(nullptr);\n    // TODO: Parse input\n    // TODO: Implement solution\n    // TODO: Output result\n    return 0;\n}'
  return {"reasoning": reasoning, 'cpp_code': code}, reasoning
