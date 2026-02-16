"""
Test 46: N-Body Gravitational Simulation (GLSL Compute Shader)

The LLM must write a GLSL compute shader that performs one timestep of
gravitational N-body simulation. The shader is dispatched repeatedly
(ping-pong) for multiple timesteps, then final positions are compared
to a CPU double-precision reference.

This tests:
1. GLSL compute shader proficiency
2. Understanding of GPU parallel patterns (all-pairs interaction)
3. Numerical stability and gravitational force computation
4. Performance at scale (O(N^2) per step)

Buffer layout:
  Binding 0 (read):  Body data - N * 2 vec4s: [pos.xyz, mass], [vel.xyz, 0]
  Binding 1 (write): Output body data - same layout
  Binding 2 (uniform): Params - uvec4(N, timesteps, 0, 0), vec4(dt, softening, G, 0)
"""

import math
import random
import struct
import time
from typing import Tuple, Optional

import numpy as np

from shader_test_utils import compile_glsl, validate_spirv
from compute_test_utils import ComputeShaderRunner, grade_compute_pingpong

title = "N-Body Gravitational Simulation (GLSL Compute)"

RANDOM_SEED = 46464646
TIMEOUT_SECONDS = 60

# Subpass configurations: (N_bodies, timesteps, dt, softening)
SUBPASSES = [
  (64, 10, 0.001, 0.01),
  (128, 20, 0.001, 0.01),
  (256, 50, 0.001, 0.01),
  (512, 100, 0.001, 0.01),
  (1024, 100, 0.001, 0.01),
  (2048, 200, 0.001, 0.01),
  (4096, 200, 0.001, 0.01),
]

G_CONSTANT = 6.674e-3  # Scaled G for visible dynamics


def generate_bodies(n, seed=RANDOM_SEED):
  """Generate N bodies with random positions, velocities, and masses."""
  rng = random.Random(seed + n)
  bodies = []
  for _ in range(n):
    # Positions in [-10, 10]
    x = rng.uniform(-10, 10)
    y = rng.uniform(-10, 10)
    z = rng.uniform(-10, 10)
    mass = rng.uniform(0.1, 10.0)
    # Small initial velocities
    vx = rng.uniform(-0.5, 0.5)
    vy = rng.uniform(-0.5, 0.5)
    vz = rng.uniform(-0.5, 0.5)
    bodies.append((x, y, z, mass, vx, vy, vz))
  return bodies


def bodies_to_buffer(bodies):
  """Pack bodies into GPU buffer format: [pos.xyz, mass, vel.xyz, 0] per body."""
  data = bytearray()
  for x, y, z, mass, vx, vy, vz in bodies:
    data += struct.pack('ffff', x, y, z, mass)
    data += struct.pack('ffff', vx, vy, vz, 0.0)
  return bytes(data)


def buffer_to_bodies(data, n):
  """Unpack GPU buffer to body list."""
  bodies = []
  for i in range(n):
    off = i * 32
    px, py, pz, mass = struct.unpack_from('ffff', data, off)
    vx, vy, vz, _ = struct.unpack_from('ffff', data, off + 16)
    bodies.append((px, py, pz, mass, vx, vy, vz))
  return bodies


def make_params_buffer(n, timesteps, dt, softening):
  """Create the uniform params buffer."""
  data = struct.pack('IIII', n, timesteps, 0, 0)
  data += struct.pack('ffff', dt, softening, G_CONSTANT, 0.0)
  return data


def cpu_reference(bodies, timesteps, dt, softening, G=G_CONSTANT):
  """Run N-body simulation on CPU with double precision."""
  n = len(bodies)
  pos = np.array([(b[0], b[1], b[2]) for b in bodies], dtype=np.float64)
  vel = np.array([(b[4], b[5], b[6]) for b in bodies], dtype=np.float64)
  mass = np.array([b[3] for b in bodies], dtype=np.float64)

  for _ in range(timesteps):
    acc = np.zeros_like(pos)
    for i in range(n):
      for j in range(n):
        if i == j:
          continue
        diff = pos[j] - pos[i]
        dist_sq = np.dot(diff, diff) + softening * softening
        dist = math.sqrt(dist_sq)
        force_mag = G * mass[j] / dist_sq
        acc[i] += force_mag * diff / dist

    vel += acc * dt
    pos += vel * dt

  result = []
  for i in range(n):
    result.append((pos[i, 0], pos[i, 1], pos[i, 2], mass[i], vel[i, 0], vel[i, 1], vel[i, 2]))
  return result


def compare_bodies(gpu_bodies, ref_bodies, pos_tol=0.1, vel_tol=0.5):
  """Compare GPU and CPU body results. Returns (score, description)."""
  n = len(ref_bodies)
  pos_errors = []
  vel_errors = []

  for i in range(n):
    gx, gy, gz = gpu_bodies[i][0], gpu_bodies[i][1], gpu_bodies[i][2]
    rx, ry, rz = ref_bodies[i][0], ref_bodies[i][1], ref_bodies[i][2]
    pos_err = math.sqrt((gx - rx)**2 + (gy - ry)**2 + (gz - rz)**2)
    pos_errors.append(pos_err)

    gvx, gvy, gvz = gpu_bodies[i][4], gpu_bodies[i][5], gpu_bodies[i][6]
    rvx, rvy, rvz = ref_bodies[i][4], ref_bodies[i][5], ref_bodies[i][6]
    vel_err = math.sqrt((gvx - rvx)**2 + (gvy - rvy)**2 + (gvz - rvz)**2)
    vel_errors.append(vel_err)

  avg_pos_err = np.mean(pos_errors)
  max_pos_err = np.max(pos_errors)
  avg_vel_err = np.mean(vel_errors)
  pos_ok = sum(1 for e in pos_errors if e < pos_tol)

  if avg_pos_err < pos_tol * 0.1 and avg_vel_err < vel_tol * 0.1:
    score = 1.0
    quality = "excellent"
  elif avg_pos_err < pos_tol and avg_vel_err < vel_tol:
    score = 0.8
    quality = "good"
  elif pos_ok > n * 0.9:
    score = 0.5
    quality = "partial (>90% bodies within tolerance)"
  elif pos_ok > n * 0.5:
    score = 0.3
    quality = "poor (>50% bodies within tolerance)"
  else:
    score = 0.0
    quality = "failed"

  desc = (f"{n} bodies: {quality}. "
          f"Avg pos error: {avg_pos_err:.6f}, max: {max_pos_err:.6f}, "
          f"avg vel error: {avg_vel_err:.6f}, "
          f"{pos_ok}/{n} within pos tolerance {pos_tol}")
  return score, desc


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

GLSL_INTERFACE = """**GLSL Compute Shader Interface:**

```
#version 450

layout(local_size_x = 256) in;

// Each body is stored as two consecutive vec4s:
//   vec4(position.xyz, mass)
//   vec4(velocity.xyz, 0.0)
// So body i starts at index i*2 in the buffer.

layout(set = 0, binding = 0) readonly buffer InputBodies {
    vec4 inData[];   // length = N * 2
};

layout(set = 0, binding = 1) buffer OutputBodies {
    vec4 outData[];  // length = N * 2
};

layout(set = 0, binding = 2) uniform Params {
    uvec4 counts;    // counts.x = N (number of bodies)
    vec4 physics;    // physics.x = dt, physics.y = softening, physics.z = G
};
```

**Task:** Write a compute shader that performs ONE timestep of gravitational
N-body simulation using the leapfrog/Euler integration method:

1. For each body i (gl_GlobalInvocationID.x):
   - Read its position and velocity from inData
   - Compute gravitational acceleration from ALL other bodies:
     acc += G * mass_j * (pos_j - pos_i) / (|pos_j - pos_i|^2 + softening^2)^(3/2)
   - Update velocity: vel += acc * dt
   - Update position: pos += vel * dt
   - Write updated position and velocity to outData

The shader will be dispatched ceil(N/256) workgroups in X.
It will be run multiple times with output feeding back as input (ping-pong)
for the required number of timesteps.

**Critical Requirements:**
- Guard against gl_GlobalInvocationID.x >= N
- Use the softening parameter to avoid singularities
- Skip self-interaction (i == j)
- Write BOTH position+mass and velocity vec4s to output
"""


def prepareSubpassPrompt(subPass):
  if subPass != 0:
    raise StopIteration

  configs = []
  for i, (n, steps, dt, soft) in enumerate(SUBPASSES):
    configs.append(f"  Subpass {i}: N={n}, timesteps={steps}, dt={dt}, softening={soft}")

  return f"""Write a GLSL compute shader for gravitational N-body simulation.

{GLSL_INTERFACE}

**Test Configurations:**
{chr(10).join(configs)}

G = {G_CONSTANT} (scaled gravitational constant)

Write the complete GLSL compute shader source code.
"""


extraGradeAnswerRuns = list(range(1, len(SUBPASSES)))

structure = {
  "type": "object",
  "properties": {
    "reasoning": {
      "type": "string",
      "description": "Explain your approach to the N-body compute shader"
    },
    "shader_code": {
      "type": "string",
      "description": "Complete GLSL compute shader source code"
    }
  },
  "required": ["reasoning", "shader_code"],
  "additionalProperties": False
}

_runner_cache = None
_ref_cache = {}


def gradeAnswer(result, subPass, aiEngineName):
  global _runner_cache, _ref_cache

  if not result or "shader_code" not in result:
    return 0.0, "No shader code provided"

  code = result["shader_code"]

  # Compile GLSL to SPIR-V
  try:
    spirv = compile_glsl(code, stage="comp")
  except RuntimeError as e:
    return 0.0, f"GLSL compilation failed: {e}"

  valid, err = validate_spirv(spirv)
  if not valid:
    return 0.0, f"SPIR-V validation failed: {err}"

  n, timesteps, dt, softening = SUBPASSES[subPass]

  # Generate test data
  bodies = generate_bodies(n)
  input_data = bodies_to_buffer(bodies)
  params = make_params_buffer(n, timesteps, dt, softening)

  # CPU reference (cached)
  cache_key = (n, timesteps)
  if cache_key not in _ref_cache:
    print(f"  Computing CPU reference for N={n}, steps={timesteps}...")
    _ref_cache[cache_key] = cpu_reference(bodies, timesteps, dt, softening)
  ref_bodies = _ref_cache[cache_key]

  # Run on GPU
  try:
    if _runner_cache is None:
      _runner_cache = ComputeShaderRunner()

    workgroups = ((n + 255) // 256, 1, 1)

    def verify_fn(output_bytes):
      gpu_bodies = buffer_to_bodies(output_bytes, n)
      # Scale tolerance with timesteps (error accumulates)
      pos_tol = 0.1 * (1 + timesteps / 100)
      vel_tol = 0.5 * (1 + timesteps / 100)
      return compare_bodies(gpu_bodies, ref_bodies, pos_tol, vel_tol)

    return grade_compute_pingpong(spirv,
                                  input_data,
                                  extra_buffers={2: params},
                                  extra_types={2: 'uniform'},
                                  workgroups=workgroups,
                                  iterations=timesteps,
                                  verify_fn=verify_fn,
                                  runner=_runner_cache,
                                  timeout=TIMEOUT_SECONDS)

  except Exception as e:
    return 0.0, f"GPU execution failed: {e}"
