"""Test 37: Feedback Vertex Set - Placebo responses for all control types."""


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
  reasoning = 'Using SCC-based greedy for FVS: 1) Find strongly connected components. 2) For each non-trivial SCC, greedily remove high-degree vertices. 3) Repeat until acyclic.'
  code = 'using System;\nusing System.Collections.Generic;\nusing System.Linq;\nclass Program {\n    static void Main() {\n        var header = Console.ReadLine().Split().Select(int.Parse).ToArray();\n        int n = header[0], m = header[1];\n        var adj = new List<int>[n];\n        var inDeg = new int[n];\n        var outDeg = new int[n];\n        for (int i = 0; i < n; i++) adj[i] = new List<int>();\n        for (int i = 0; i < m; i++) {\n            var e = Console.ReadLine().Split().Select(int.Parse).ToArray();\n            adj[e[0]].Add(e[1]); outDeg[e[0]]++; inDeg[e[1]]++;\n        }\n        var removed = new HashSet<int>();\n        bool hasCycle = true;\n        while (hasCycle) {\n            hasCycle = false;\n            var color = new int[n];\n            for (int s = 0; s < n; s++) {\n                if (removed.Contains(s) || color[s] != 0) continue;\n                var stack = new Stack<int>(); stack.Push(s);\n                while (stack.Count > 0) {\n                    int u = stack.Peek();\n                    if (color[u] == 0) { color[u] = 1; }\n                    bool found = false;\n                    foreach (int v in adj[u]) {\n                        if (!removed.Contains(v)) {\n                            if (color[v] == 1) { removed.Add(u); hasCycle = true; break; }\n                            if (color[v] == 0) { stack.Push(v); found = true; break; }\n                        }\n                    }\n                    if (hasCycle) break;\n                    if (!found) { color[u] = 2; stack.Pop(); }\n                }\n                if (hasCycle) break;\n            }\n        }\n        Console.WriteLine(removed.Count);\n        Console.WriteLine(string.Join(" ", removed));\n    }\n}'
  return {"reasoning": reasoning, 'csharp_code': code}, reasoning


def _naive_optimised(subpass):
  # TODO: Same algorithm as naive but hyper-optimised for C#
  # (parallel, SIMD, small data types, register packing, etc.)
  return _naive(subpass)


def _best_published(subpass):
  reasoning = (
    "Best published: Iterative compression FVS (Chen et al. 2008, J. ACM 55(5):1-19). "
    "TODO: Full implementation pending."
  )
  code = '// TODO: Implement Iterative compression FVS'
  return {"reasoning": reasoning, 'csharp_code': code}, reasoning


def _random(subpass):
  reasoning = 'Random: pseudorandom output with fixed seed 42.'
  code = 'using System;\nclass Program {\n    static void Main() {\n        var rng = new Random(42);\n        Console.ReadLine();\n        Console.WriteLine("0");\n    }\n}'
  return {"reasoning": reasoning, 'csharp_code': code}, reasoning


def _human(subpass):
  reasoning = 'Human starting point for Feedback Vertex Set. Fill in the TODOs.'
  code = '// TODO: Human attempt at Feedback Vertex Set\nusing System;\nusing System.Collections.Generic;\nusing System.Linq;\nclass Program {\n    static void Main() {\n        // TODO: Parse input\n        // TODO: Implement solution\n        // TODO: Output result\n    }\n}'
  return {"reasoning": reasoning, 'csharp_code': code}, reasoning
