"""Test 12: 3D Bin Packing - Placebo responses for all control types."""


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
  reasoning = 'Grid-based 3D bin packing in C#. Places polyhedra in axis-aligned grid. No rotation.'
  code = 'using System;\nusing System.Collections.Generic;\nusing System.Linq;\nclass Program {\n    static void Main() {\n        var h = Console.ReadLine().Split().Select(double.Parse).ToArray();\n        double cw = h[0], ch = h[1], cd = h[2];\n        int nv = int.Parse(Console.ReadLine().Trim());\n        var vx = new double[nv]; var vy = new double[nv]; var vz = new double[nv];\n        for (int i = 0; i < nv; i++) {\n            var p = Console.ReadLine().Split().Select(double.Parse).ToArray();\n            vx[i] = p[0]; vy[i] = p[1]; vz[i] = p[2];\n        }\n        double pw = vx.Max()-vx.Min(), ph = vy.Max()-vy.Min(), pd = vz.Max()-vz.Min();\n        double ox = -vx.Min(), oy = -vy.Min(), oz = -vz.Min();\n        double gap = 0.1;\n        var placements = new List<string>();\n        for (double x = ox; x+pw <= cw; x += pw+gap)\n            for (double y = oy; y+ph <= ch; y += ph+gap)\n                for (double z = oz; z+pd <= cd; z += pd+gap)\n                    placements.Add($"{x:F3} {y:F3} {z:F3} 1 0 0 0");\n        Console.WriteLine(placements.Count);\n        foreach (var p in placements) Console.WriteLine(p);\n    }\n}'
  return {"reasoning": reasoning, 'csharp_code': code}, reasoning


def _naive_optimised(subpass):
  # TODO: Same algorithm as naive but hyper-optimised for C#
  # (parallel, SIMD, small data types, register packing, etc.)
  return _naive(subpass)


def _best_published(subpass):
  reasoning = (
    "Best published: Cagan et al. 3D packing (Cagan et al. 2002, Computer-Aided Design 34(8):597-611). "
    "TODO: Full implementation pending."
  )
  code = '// TODO: Implement Cagan et al. 3D packing'
  return {"reasoning": reasoning, 'csharp_code': code}, reasoning


def _random(subpass):
  reasoning = 'Random: pseudorandom output with fixed seed 42.'
  code = 'using System;\nclass Program {\n    static void Main() {\n        var rng = new Random(42);\n        Console.ReadLine();\n        Console.WriteLine("0");\n    }\n}'
  return {"reasoning": reasoning, 'csharp_code': code}, reasoning


def _human(subpass):
  reasoning = 'Human starting point for 3D Bin Packing. Fill in the TODOs.'
  code = '// TODO: Human attempt at 3D Bin Packing\nusing System;\nusing System.Collections.Generic;\nusing System.Linq;\nclass Program {\n    static void Main() {\n        // TODO: Parse input\n        // TODO: Implement solution\n        // TODO: Output result\n    }\n}'
  return {"reasoning": reasoning, 'csharp_code': code}, reasoning
