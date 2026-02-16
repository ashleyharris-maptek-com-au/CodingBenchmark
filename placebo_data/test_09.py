"""Test 9: 1D Cutting Stock - Placebo responses for all control types."""


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
  reasoning = 'First Fit Decreasing for 1D cutting stock in C#. Simple bin packing heuristic.'
  code = 'using System;\nusing System.Collections.Generic;\nusing System.Linq;\nclass Program {\n    static void Main() {\n        var h = Console.ReadLine().Split();\n        int n = int.Parse(h[0]); double stock = double.Parse(h[1]);\n        var cuts = new double[n];\n        for (int i = 0; i < n; i++) cuts[i] = double.Parse(Console.ReadLine().Trim());\n        var idx = Enumerable.Range(0, n).OrderByDescending(i => cuts[i]).ToArray();\n        var bins = new List<(double rem, List<int> items)>();\n        foreach (int i in idx) {\n            bool placed = false;\n            for (int b = 0; b < bins.Count; b++) {\n                if (bins[b].rem >= cuts[i]) {\n                    bins[b] = (bins[b].rem - cuts[i], bins[b].items);\n                    bins[b].items.Add(i); placed = true; break;\n                }\n            }\n            if (!placed) bins.Add((stock - cuts[i], new List<int>{i}));\n        }\n        Console.WriteLine(bins.Count);\n        for (int b = 0; b < bins.Count; b++)\n            Console.WriteLine(string.Join(" ", bins[b].items));\n    }\n}'
  return {"reasoning": reasoning, 'csharp_code': code}, reasoning


def _naive_optimised(subpass):
  # TODO: Same algorithm as naive but hyper-optimised for C#
  # (parallel, SIMD, small data types, register packing, etc.)
  return _naive(subpass)


def _best_published(subpass):
  reasoning = (
    "Best published: Martello-Toth bin packing (Martello & Toth 1990, Discrete Applied Mathematics 28(1):59-70). "
    "TODO: Full implementation pending."
  )
  code = '// TODO: Implement Martello-Toth bin packing'
  return {"reasoning": reasoning, 'csharp_code': code}, reasoning


def _random(subpass):
  reasoning = 'Random: pseudorandom output with fixed seed 42.'
  code = 'using System;\nclass Program {\n    static void Main() {\n        var rng = new Random(42);\n        Console.ReadLine();\n        Console.WriteLine("0");\n    }\n}'
  return {"reasoning": reasoning, 'csharp_code': code}, reasoning


def _human(subpass):
  reasoning = 'Human starting point for 1D Cutting Stock. Fill in the TODOs.'
  code = '// TODO: Human attempt at 1D Cutting Stock\nusing System;\nusing System.Collections.Generic;\nusing System.Linq;\nclass Program {\n    static void Main() {\n        // TODO: Parse input\n        // TODO: Implement solution\n        // TODO: Output result\n    }\n}'
  return {"reasoning": reasoning, 'csharp_code': code}, reasoning
