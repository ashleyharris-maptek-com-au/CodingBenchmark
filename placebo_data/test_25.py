"""Test 25: Asteroid Interception - Placebo responses for all control types."""


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
  reasoning = 'Using simplified Hohmann transfer for asteroid interception: 1) Parse solar system state and asteroid position/velocity. 2) Estimate asteroid future position at 60% of warning time. 3) Calculate transfer trajectory direction and required velocity. 4) Launch burn: escape Earth velocity + transfer velocity. 5) Mid-course correction at 50% of transfer time. 6) Final approach burn at 90% of transfer time. 7) Scale burns to stay within delta-V budget.'
  code = 'using System;\nusing System.Collections.Generic;\nusing System.Globalization;\n\nclass AsteroidInterceptor {\n    const double G = 6.67430e-11;\n    const double AU = 1.496e11;\n    const double DAY = 86400.0;\n    \n    static double sunMass = 1.989e30;\n    static double[] earthPos = new double[3];\n    static double[] earthVel = new double[3];\n    static double[] asteroidPos = new double[3];\n    static double[] asteroidVel = new double[3];\n    static double warningDays, impactorMass, deltaVBudget, asteroidMass;\n    \n    static double Magnitude(double[] v) {\n        return Math.Sqrt(v[0]*v[0] + v[1]*v[1] + v[2]*v[2]);\n    }\n    \n    static double[] Normalize(double[] v) {\n        double m = Magnitude(v);\n        if (m < 1e-10) return new double[] {0, 0, 0};\n        return new double[] {v[0]/m, v[1]/m, v[2]/m};\n    }\n    \n    static void Main() {\n        CultureInfo.CurrentCulture = CultureInfo.InvariantCulture;\n        \n        // Parse header\n        var header = Console.ReadLine().Split();\n        int numBodies = int.Parse(header[0]);\n        warningDays = double.Parse(header[1]);\n        impactorMass = double.Parse(header[2]);\n        deltaVBudget = double.Parse(header[3]);\n        asteroidMass = double.Parse(header[4]);\n        \n        // Parse bodies\n        for (int i = 0; i < numBodies; i++) {\n            var parts = Console.ReadLine().Split();\n            string name = parts[0];\n            double mass = double.Parse(parts[1]);\n            double x = double.Parse(parts[2]);\n            double y = double.Parse(parts[3]);\n            double z = double.Parse(parts[4]);\n            double vx = double.Parse(parts[5]);\n            double vy = double.Parse(parts[6]);\n            double vz = double.Parse(parts[7]);\n            \n            if (name == "Earth") {\n                earthPos = new double[] {x, y, z};\n                earthVel = new double[] {vx, vy, vz};\n            } else if (name == "Asteroid") {\n                asteroidPos = new double[] {x, y, z};\n                asteroidVel = new double[] {vx, vy, vz};\n            } else if (name == "Sun") {\n                sunMass = mass;\n            }\n        }\n        \n        // Simple intercept calculation\n        // Estimate where asteroid will be at intercept time\n        double interceptDays = warningDays * 0.6; // Intercept at 60% of warning time\n        \n        double[] futureAsteroid = new double[3];\n        for (int i = 0; i < 3; i++) {\n            futureAsteroid[i] = asteroidPos[i] + asteroidVel[i] * interceptDays * DAY;\n        }\n        \n        // Calculate required velocity for Hohmann-like transfer\n        double[] toAsteroid = new double[] {\n            futureAsteroid[0] - earthPos[0],\n            futureAsteroid[1] - earthPos[1],\n            futureAsteroid[2] - earthPos[2]\n        };\n        \n        double transferDist = Magnitude(toAsteroid);\n        double transferTime = interceptDays * DAY;\n        \n        // Required average velocity\n        double reqSpeed = transferDist / transferTime;\n        double[] transferDir = Normalize(toAsteroid);\n        \n        // Launch burn - escape Earth and head toward asteroid\n        double[] launchDV = new double[3];\n        for (int i = 0; i < 3; i++) {\n            launchDV[i] = transferDir[i] * reqSpeed - earthVel[i];\n        }\n        \n        double launchMag = Magnitude(launchDV);\n        \n        // Scale if over budget\n        if (launchMag > deltaVBudget * 0.8) {\n            double scale = deltaVBudget * 0.8 / launchMag;\n            for (int i = 0; i < 3; i++) launchDV[i] *= scale;\n            launchMag = Magnitude(launchDV);\n        }\n        \n        // Mid-course correction at halfway point\n        double mcDays = interceptDays * 0.5;\n        double remainingDV = deltaVBudget - launchMag;\n        \n        // Small correction toward asteroid\n        double[] mcDV = new double[] {\n            transferDir[0] * remainingDV * 0.3,\n            transferDir[1] * remainingDV * 0.3,\n            transferDir[2] * remainingDV * 0.3\n        };\n        \n        // Final approach burn\n        double finalDays = interceptDays * 0.9;\n        double[] finalDV = new double[] {\n            transferDir[0] * remainingDV * 0.5,\n            transferDir[1] * remainingDV * 0.5,\n            transferDir[2] * remainingDV * 0.5\n        };\n        \n        // Output burn sequence\n        Console.WriteLine("3");\n        Console.WriteLine($"0 {launchDV[0]:F2} {launchDV[1]:F2} {launchDV[2]:F2}");\n        Console.WriteLine($"{mcDays:F0} {mcDV[0]:F2} {mcDV[1]:F2} {mcDV[2]:F2}");\n        Console.WriteLine($"{finalDays:F0} {finalDV[0]:F2} {finalDV[1]:F2} {finalDV[2]:F2}");\n    }\n}\n'
  return {"reasoning": reasoning, 'csharp_code': code}, reasoning


def _naive_optimised(subpass):
  # TODO: Same algorithm as naive but hyper-optimised for C#
  # (parallel, SIMD, small data types, register packing, etc.)
  return _naive(subpass)


def _best_published(subpass):
  reasoning = (
    "Best published: Lambert + GA trajectory (Izzo et al. 2007, J. Global Optimization 38(2):283-296). "
    "TODO: Full implementation pending."
  )
  code = '// TODO: Implement Lambert + GA trajectory'
  return {"reasoning": reasoning, 'csharp_code': code}, reasoning


def _random(subpass):
  reasoning = 'Random: pseudorandom output with fixed seed 42.'
  code = 'using System;\nclass Program {\n    static void Main() {\n        var rng = new Random(42);\n        Console.ReadLine();\n        Console.WriteLine("0");\n    }\n}'
  return {"reasoning": reasoning, 'csharp_code': code}, reasoning


def _human(subpass):
  reasoning = 'Human starting point for Asteroid Interception. Fill in the TODOs.'
  code = '// TODO: Human attempt at Asteroid Interception\nusing System;\nusing System.Collections.Generic;\nusing System.Linq;\nclass Program {\n    static void Main() {\n        // TODO: Parse input\n        // TODO: Implement solution\n        // TODO: Output result\n    }\n}'
  return {"reasoning": reasoning, 'csharp_code': code}, reasoning
