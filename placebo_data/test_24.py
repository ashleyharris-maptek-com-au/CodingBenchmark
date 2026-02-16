"""Test 24: 3D Lunar Lander - Placebo responses for all control types."""


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
  reasoning = 'Using 3D PD controller for lunar lander in C++: 1) Parse world dimensions, obstacles, and target position. 2) For each state update with 13 values (pos, vel, angles, rates, fuel). 3) Calculate 3D velocity error toward target + gravity compensation. 4) Compute desired pitch and yaw from velocity error direction. 5) PD control for pitch, yaw, roll (keep roll level). 6) Thrust when aligned, careful control near landing. 7) Flush output immediately.'
  code = '#include <iostream>\n#include <cmath>\n#include <sstream>\n#include <string>\nusing namespace std;\n\ndouble targetX, targetY, targetZ;\ndouble gravity, maxThrust;\n\nint main() {\n    ios_base::sync_with_stdio(false);\n    cin.tie(nullptr);\n    \n    // Parse header\n    double width, depth, height, fuel;\n    cin >> width >> depth >> height >> gravity >> maxThrust >> fuel;\n    \n    double startX, startY, startZ;\n    cin >> startX >> startY >> startZ >> targetX >> targetY >> targetZ;\n    \n    int numObstacles;\n    cin >> numObstacles;\n    for (int i = 0; i < numObstacles; i++) {\n        double ox, oy, oz, r;\n        cin >> ox >> oy >> oz >> r;\n    }\n    \n    string marker;\n    cin >> marker; // STATE\n    \n    // Process state updates\n    while (true) {\n        double x, y, z, vx, vy, vz;\n        double pitch, yaw, roll;\n        double pitchRate, yawRate, rollRate;\n        double fuelLeft;\n        \n        if (!(cin >> x >> y >> z >> vx >> vy >> vz \n              >> pitch >> yaw >> roll \n              >> pitchRate >> yawRate >> rollRate >> fuelLeft)) {\n            break;\n        }\n        \n        if (fuelLeft <= 0) {\n            cout << "0.0 0.0 0.0 0.0\\n";\n            cout.flush();\n            continue;\n        }\n        \n        // Calculate error to target\n        double dx = targetX - x;\n        double dy = targetY - y;\n        double dz = targetZ - z;\n        double dist = sqrt(dx*dx + dy*dy + dz*dz);\n        \n        // Desired velocity toward target\n        double scale = 0.1;\n        double desiredVx = dx * scale;\n        double desiredVy = dy * scale;\n        double desiredVz = dz * scale + gravity * 2.0; // Counter gravity\n        \n        // Velocity error\n        double vxErr = desiredVx - vx;\n        double vyErr = desiredVy - vy;\n        double vzErr = desiredVz - vz;\n        \n        // Calculate desired orientation\n        // We want thrust to point in direction of velocity error\n        double horizErr = sqrt(vxErr*vxErr + vyErr*vyErr);\n        double desiredPitch = atan2(vzErr, horizErr);\n        double desiredYaw = atan2(vxErr, vyErr);\n        \n        // Angle errors\n        double pitchErr = desiredPitch - pitch;\n        double yawErr = desiredYaw - yaw;\n        \n        // Normalize yaw error to [-pi, pi]\n        while (yawErr > M_PI) yawErr -= 2*M_PI;\n        while (yawErr < -M_PI) yawErr += 2*M_PI;\n        \n        // PD control for rotation\n        double pitchCmd = max(-1.0, min(1.0, pitchErr * 2.0 - pitchRate * 0.5));\n        double yawCmd = max(-1.0, min(1.0, yawErr * 2.0 - yawRate * 0.5));\n        double rollCmd = max(-1.0, min(1.0, -roll * 2.0 - rollRate * 0.5)); // Keep level\n        \n        // Thrust control\n        double velErr = sqrt(vxErr*vxErr + vyErr*vyErr + vzErr*vzErr);\n        double speed = sqrt(vx*vx + vy*vy + vz*vz);\n        \n        double thrust;\n        if (dist < 50.0) {\n            // Landing phase\n            if (speed > 3.0) {\n                thrust = min(1.0, (speed - 2.0) / maxThrust);\n            } else if (vz < -1.0) {\n                thrust = min(0.6, (-vz - 0.5) / maxThrust * 2.0);\n            } else {\n                thrust = 0.1;\n            }\n        } else if (abs(pitchErr) < 0.3 && abs(yawErr) < 0.3) {\n            // Aligned - thrust\n            thrust = min(1.0, max(0.2, velErr / maxThrust * 0.5));\n        } else {\n            // Turning - minimal thrust\n            thrust = 0.1;\n        }\n        \n        cout << thrust << " " << pitchCmd << " " << yawCmd << " " << rollCmd << "\\n";\n        cout.flush();\n    }\n    \n    return 0;\n}\n'
  return {"reasoning": reasoning, 'cpp_code': code}, reasoning


def _naive_optimised(subpass):
  # TODO: Same algorithm as naive but hyper-optimised for C++
  # (parallel, SIMD, small data types, register packing, etc.)
  return _naive(subpass)


def _best_published(subpass):
  reasoning = (
    "Best published: Model predictive control (Camacho & Bordons 2007, Springer). "
    "TODO: Full implementation pending."
  )
  code = '// TODO: Implement Model predictive control'
  return {"reasoning": reasoning, 'cpp_code': code}, reasoning


def _random(subpass):
  reasoning = 'Random: pseudorandom output with fixed seed 42.'
  code = '#include <iostream>\n#include <random>\nusing namespace std;\nint main() {\n    mt19937 rng(42);\n    string line;\n    getline(cin, line);\n    cout << "0" << endl;\n    return 0;\n}'
  return {"reasoning": reasoning, 'cpp_code': code}, reasoning


def _human(subpass):
  reasoning = 'Human starting point for 3D Lunar Lander. Fill in the TODOs.'
  code = '// TODO: Human attempt at 3D Lunar Lander\n#include <iostream>\n#include <vector>\n#include <algorithm>\nusing namespace std;\nint main() {\n    ios_base::sync_with_stdio(false);\n    cin.tie(nullptr);\n    // TODO: Parse input\n    // TODO: Implement solution\n    // TODO: Output result\n    return 0;\n}'
  return {"reasoning": reasoning, 'cpp_code': code}, reasoning
