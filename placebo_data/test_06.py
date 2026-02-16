"""Test 6: Orbital TSP - Placebo responses for all control types."""


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
  reasoning = 'Greedy nearest-neighbor orbital TSP in C#. Simplified Hohmann delta-V estimates. No Lambert solver.'
  code = 'using System;\nusing System.Collections.Generic;\nusing System.Linq;\nclass Program {\n    static double DeltaV(double[] a, double[] b, double mu) {\n        double r1 = Math.Sqrt(a[0]*a[0]+a[1]*a[1]+a[2]*a[2]);\n        double r2 = Math.Sqrt(b[0]*b[0]+b[1]*b[1]+b[2]*b[2]);\n        double at = (r1+r2)/2;\n        double v1 = Math.Sqrt(mu/r1); double v2 = Math.Sqrt(mu/r2);\n        double vt1 = Math.Sqrt(mu*(2/r1-1/at)); double vt2 = Math.Sqrt(mu*(2/r2-1/at));\n        return Math.Abs(vt1-v1) + Math.Abs(v2-vt2);\n    }\n    static void Main() {\n        var h = Console.ReadLine().Split();\n        int n = int.Parse(h[0]); double mu = double.Parse(h[1]);\n        var start = Console.ReadLine().Split().Select(double.Parse).ToArray();\n        var orbits = new double[n][];\n        for (int i = 0; i < n; i++)\n            orbits[i] = Console.ReadLine().Split().Select(double.Parse).ToArray();\n        var visited = new bool[n]; var order = new List<int>();\n        var cur = start; double total = 0;\n        for (int step = 0; step < n; step++) {\n            int best = -1; double bd = 1e18;\n            for (int j = 0; j < n; j++)\n                if (!visited[j]) { double d = DeltaV(cur, orbits[j], mu); if (d<bd){bd=d;best=j;} }\n            visited[best] = true; order.Add(best); total += bd; cur = orbits[best];\n        }\n        Console.WriteLine(string.Join(" ", order));\n        Console.WriteLine(total);\n    }\n}'
  return {"reasoning": reasoning, 'csharp_code': code}, reasoning


def _naive_optimised(subpass):
  # TODO: Same algorithm as naive but hyper-optimised for C#
  # (parallel, SIMD, small data types, register packing, etc.)
  return _naive(subpass)


def _best_published(subpass):
  reasoning = (
    "Best published: Lambert solver + branch-and-bound (Izzo 2015, Celestial Mechanics 121(1):1-15). "
    "TODO: Full implementation pending."
  )
  code = '// TODO: Implement Lambert solver + branch-and-bound'
  return {"reasoning": reasoning, 'csharp_code': code}, reasoning


def _random(subpass):
  reasoning = 'Random: pseudorandom output with fixed seed 42.'
  code = 'using System;\nclass Program {\n    static void Main() {\n        var rng = new Random(42);\n        Console.ReadLine();\n        Console.WriteLine("0");\n    }\n}'
  return {"reasoning": reasoning, 'csharp_code': code}, reasoning


def _human(subpass):
  reasoning = 'Human starting point for Orbital TSP. Fill in the TODOs.'
  code = '// TODO: Human attempt at Orbital TSP\nusing System;\nusing System.Collections.Generic;\nusing System.Linq;\nclass Program {\n    static void Main() {\n        // TODO: Parse input\n        // TODO: Implement solution\n        // TODO: Output result\n    }\n}'
  return {"reasoning": reasoning, 'csharp_code': code}, reasoning
