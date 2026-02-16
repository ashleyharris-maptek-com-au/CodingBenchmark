"""Test 2: Chinese Postman - Placebo responses for all control types."""


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
  reasoning = 'Greedy Eulerian circuit in C#. Hierholzer with greedy odd-vertex pairing. No optimisation.'
  code = 'using System;\nusing System.Collections.Generic;\nusing System.Linq;\nclass Program {\n    static void Main() {\n        var h = Console.ReadLine().Split().Select(int.Parse).ToArray();\n        int n = h[0], m = h[1];\n        var adj = new List<int>[n];\n        for (int i = 0; i < n; i++) adj[i] = new List<int>();\n        for (int i = 0; i < m; i++) {\n            var e = Console.ReadLine().Split().Select(int.Parse).ToArray();\n            adj[e[0]].Add(e[1]); adj[e[1]].Add(e[0]);\n        }\n        // Greedy DFS traversal covering all edges\n        var stack = new Stack<int>();\n        var route = new List<int>();\n        stack.Push(0);\n        while (stack.Count > 0) {\n            int v = stack.Peek();\n            if (adj[v].Count > 0) {\n                int u = adj[v][0]; adj[v].RemoveAt(0); adj[u].Remove(v);\n                stack.Push(u);\n            } else { route.Add(stack.Pop()); }\n        }\n        route.Reverse();\n        Console.WriteLine(string.Join(" ", route));\n    }\n}'
  return {"reasoning": reasoning, 'csharp_code': code}, reasoning


def _naive_optimised(subpass):
  # TODO: Same algorithm as naive but hyper-optimised for C#
  # (parallel, SIMD, small data types, register packing, etc.)
  return _naive(subpass)


def _best_published(subpass):
  reasoning = (
    "Best published: Edmonds minimum weight perfect matching (Edmonds 1965, Canadian J. Math 17:449-467). "
    "TODO: Full implementation pending."
  )
  code = '// TODO: Implement Edmonds minimum weight perfect matching'
  return {"reasoning": reasoning, 'csharp_code': code}, reasoning


def _random(subpass):
  reasoning = 'Random: pseudorandom output with fixed seed 42.'
  code = 'using System;\nclass Program {\n    static void Main() {\n        var rng = new Random(42);\n        Console.ReadLine();\n        Console.WriteLine("0");\n    }\n}'
  return {"reasoning": reasoning, 'csharp_code': code}, reasoning


def _human(subpass):
  reasoning = 'Human starting point for Chinese Postman. Fill in the TODOs.'
  code = '// TODO: Human attempt at Chinese Postman\nusing System;\nusing System.Collections.Generic;\nusing System.Linq;\nclass Program {\n    static void Main() {\n        // TODO: Parse input\n        // TODO: Implement solution\n        // TODO: Output result\n    }\n}'
  return {"reasoning": reasoning, 'csharp_code': code}, reasoning
