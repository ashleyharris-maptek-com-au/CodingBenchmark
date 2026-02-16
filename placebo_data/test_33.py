"""Test 33: Quadratic Assignment - Placebo responses for all control types."""


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
  reasoning = 'Using greedy assignment for QAP: 1) Sort facilities by total flow. 2) Sort locations by total distance. 3) Assign high-flow facilities to central locations.'
  code = 'using System;\nusing System.Linq;\nclass Program {\n    static void Main() {\n        int n = int.Parse(Console.ReadLine());\n        var flow = new int[n, n];\n        var dist = new int[n, n];\n        for (int i = 0; i < n; i++) {\n            var row = Console.ReadLine().Split().Select(int.Parse).ToArray();\n            for (int j = 0; j < n; j++) flow[i, j] = row[j];\n        }\n        for (int i = 0; i < n; i++) {\n            var row = Console.ReadLine().Split().Select(int.Parse).ToArray();\n            for (int j = 0; j < n; j++) dist[i, j] = row[j];\n        }\n        var facFlow = Enumerable.Range(0, n).Select(i => (Enumerable.Range(0, n).Sum(j => flow[i, j] + flow[j, i]), i)).OrderByDescending(x => x.Item1).ToList();\n        var locDist = Enumerable.Range(0, n).Select(i => (Enumerable.Range(0, n).Sum(j => dist[i, j]), i)).OrderBy(x => x.Item1).ToList();\n        var perm = new int[n];\n        for (int r = 0; r < n; r++) perm[facFlow[r].i] = locDist[r].i;\n        long cost = 0;\n        for (int i = 0; i < n; i++) for (int j = 0; j < n; j++) cost += flow[i, j] * dist[perm[i], perm[j]];\n        Console.WriteLine(cost);\n        Console.WriteLine(string.Join(" ", perm));\n    }\n}'
  return {"reasoning": reasoning, 'csharp_code': code}, reasoning


def _naive_optimised(subpass):
  # TODO: Same algorithm as naive but hyper-optimised for C#
  # (parallel, SIMD, small data types, register packing, etc.)
  return _naive(subpass)


def _best_published(subpass):
  reasoning = (
    "Best published: Simulated annealing QAP (Burkard & Rendl 1984, European J. OR 17(2):169-174). "
    "TODO: Full implementation pending."
  )
  code = '// TODO: Implement Simulated annealing QAP'
  return {"reasoning": reasoning, 'csharp_code': code}, reasoning


def _random(subpass):
  reasoning = 'Random: pseudorandom output with fixed seed 42.'
  code = 'using System;\nclass Program {\n    static void Main() {\n        var rng = new Random(42);\n        Console.ReadLine();\n        Console.WriteLine("0");\n    }\n}'
  return {"reasoning": reasoning, 'csharp_code': code}, reasoning


def _human(subpass):
  reasoning = 'Human starting point for Quadratic Assignment. Fill in the TODOs.'
  code = '// TODO: Human attempt at Quadratic Assignment\nusing System;\nusing System.Collections.Generic;\nusing System.Linq;\nclass Program {\n    static void Main() {\n        // TODO: Parse input\n        // TODO: Implement solution\n        // TODO: Output result\n    }\n}'
  return {"reasoning": reasoning, 'csharp_code': code}, reasoning
