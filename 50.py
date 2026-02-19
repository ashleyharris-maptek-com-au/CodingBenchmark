"""
Test 50: Boid Flocking Simulation (GLSL Compute Shader)

The LLM must write a GLSL compute shader that performs one step of a
3D boid flocking simulation with separation, alignment, and cohesion rules.
The shader is dispatched repeatedly (ping-pong) for multiple timesteps.

This tests:
1. GLSL compute shader proficiency
2. Understanding of flocking algorithms (Reynolds' boids)
3. O(N^2) neighbor search on GPU
4. Vector math and boundary handling

Buffer layout:
  Binding 0 (read):  Input boids - N * 2 vec4s: [pos.xyz, 0], [vel.xyz, 0]
  Binding 1 (write): Output boids - same layout
  Binding 2 (uniform): Params:
      uvec4(N, 0, 0, 0)
      vec4(dt, separationRadius, alignmentRadius, cohesionRadius)
      vec4(separationWeight, alignmentWeight, cohesionWeight, maxSpeed)
      vec4(boundSize, boundForce, 0, 0)
"""

import math
import random
import struct
import time
from typing import Tuple, Optional

import numpy as np

from shader_test_utils import compile_glsl, validate_spirv
from compute_test_utils import ComputeShaderRunner, grade_compute_pingpong

title = "Boid Flocking Simulation (GLSL Compute)"

RANDOM_SEED = 50505050
TIMEOUT_SECONDS = 60

# Flocking parameters
SEP_RADIUS = 2.0
ALIGN_RADIUS = 5.0
COHESION_RADIUS = 8.0
SEP_WEIGHT = 1.5
ALIGN_WEIGHT = 1.0
COHESION_WEIGHT = 1.0
MAX_SPEED = 5.0
BOUND_SIZE = 50.0
BOUND_FORCE = 0.5

# Subpass configurations: (N_boids, timesteps, dt)
SUBPASSES = [
  (128, 50, 0.05),
  (256, 100, 0.05),
  (512, 100, 0.05),
  (1024, 200, 0.05),
  (2048, 200, 0.05),
  (4096, 200, 0.05),
]


def generate_boids(n, seed=RANDOM_SEED):
  """Generate N boids with random positions and velocities."""
  rng = random.Random(seed + n)
  boids = []
  for _ in range(n):
    x = rng.uniform(-BOUND_SIZE * 0.5, BOUND_SIZE * 0.5)
    y = rng.uniform(-BOUND_SIZE * 0.5, BOUND_SIZE * 0.5)
    z = rng.uniform(-BOUND_SIZE * 0.5, BOUND_SIZE * 0.5)
    vx = rng.uniform(-1, 1)
    vy = rng.uniform(-1, 1)
    vz = rng.uniform(-1, 1)
    boids.append((x, y, z, vx, vy, vz))
  return boids


def boids_to_buffer(boids):
  """Pack boids into GPU buffer: [pos.xyz, 0, vel.xyz, 0] per boid."""
  data = bytearray()
  for x, y, z, vx, vy, vz in boids:
    data += struct.pack('ffff', x, y, z, 0.0)
    data += struct.pack('ffff', vx, vy, vz, 0.0)
  return bytes(data)


def buffer_to_boids(data, n):
  """Unpack buffer to boid list."""
  boids = []
  for i in range(n):
    off = i * 32
    px, py, pz, _ = struct.unpack_from('ffff', data, off)
    vx, vy, vz, _ = struct.unpack_from('ffff', data, off + 16)
    boids.append((px, py, pz, vx, vy, vz))
  return boids


def make_params_buffer(n, dt):
  """Create the uniform params buffer."""
  data = struct.pack('IIII', n, 0, 0, 0)
  data += struct.pack('ffff', dt, SEP_RADIUS, ALIGN_RADIUS, COHESION_RADIUS)
  data += struct.pack('ffff', SEP_WEIGHT, ALIGN_WEIGHT, COHESION_WEIGHT, MAX_SPEED)
  data += struct.pack('ffff', BOUND_SIZE, BOUND_FORCE, 0.0, 0.0)
  return data


def clamp_speed(vx, vy, vz, max_speed):
  """Clamp velocity vector to max speed."""
  speed = math.sqrt(vx * vx + vy * vy + vz * vz)
  if speed > max_speed and speed > 0:
    scale = max_speed / speed
    return vx * scale, vy * scale, vz * scale
  return vx, vy, vz


def cpu_boid_step(boids, dt):
  """One step of boid simulation on CPU."""
  n = len(boids)
  new_boids = []

  for i in range(n):
    px, py, pz, vx, vy, vz = boids[i]

    # Separation, alignment, cohesion accumulators
    sep_x, sep_y, sep_z = 0.0, 0.0, 0.0
    align_x, align_y, align_z = 0.0, 0.0, 0.0
    coh_x, coh_y, coh_z = 0.0, 0.0, 0.0
    sep_count = 0
    align_count = 0
    coh_count = 0

    for j in range(n):
      if i == j:
        continue
      jx, jy, jz, jvx, jvy, jvz = boids[j]
      dx = px - jx
      dy = py - jy
      dz = pz - jz
      dist = math.sqrt(dx * dx + dy * dy + dz * dz)
      if dist < 1e-6:
        continue

      # Separation
      if dist < SEP_RADIUS:
        sep_x += dx / dist
        sep_y += dy / dist
        sep_z += dz / dist
        sep_count += 1

      # Alignment
      if dist < ALIGN_RADIUS:
        align_x += jvx
        align_y += jvy
        align_z += jvz
        align_count += 1

      # Cohesion
      if dist < COHESION_RADIUS:
        coh_x += jx
        coh_y += jy
        coh_z += jz
        coh_count += 1

    # Apply forces
    ax, ay, az = 0.0, 0.0, 0.0

    if sep_count > 0:
      ax += SEP_WEIGHT * sep_x / sep_count
      ay += SEP_WEIGHT * sep_y / sep_count
      az += SEP_WEIGHT * sep_z / sep_count

    if align_count > 0:
      ax += ALIGN_WEIGHT * (align_x / align_count - vx)
      ay += ALIGN_WEIGHT * (align_y / align_count - vy)
      az += ALIGN_WEIGHT * (align_z / align_count - vz)

    if coh_count > 0:
      ax += COHESION_WEIGHT * (coh_x / coh_count - px)
      ay += COHESION_WEIGHT * (coh_y / coh_count - py)
      az += COHESION_WEIGHT * (coh_z / coh_count - pz)

    # Boundary force (push back toward center)
    half = BOUND_SIZE * 0.5
    if abs(px) > half * 0.8:
      ax -= BOUND_FORCE * (px / half)
    if abs(py) > half * 0.8:
      ay -= BOUND_FORCE * (py / half)
    if abs(pz) > half * 0.8:
      az -= BOUND_FORCE * (pz / half)

    # Update velocity
    nvx = vx + ax * dt
    nvy = vy + ay * dt
    nvz = vz + az * dt
    nvx, nvy, nvz = clamp_speed(nvx, nvy, nvz, MAX_SPEED)

    # Update position
    npx = px + nvx * dt
    npy = py + nvy * dt
    npz = pz + nvz * dt

    new_boids.append((npx, npy, npz, nvx, nvy, nvz))

  return new_boids


def cpu_reference(boids, timesteps, dt):
  """Run boid simulation on CPU."""
  b = list(boids)
  for _ in range(timesteps):
    b = cpu_boid_step(b, dt)
  return b


def compare_boids(gpu_boids, ref_boids, pos_tol=1.0, vel_tol=2.0):
  """Compare GPU and CPU boid results. Returns (score, description)."""
  n = len(ref_boids)
  pos_errors = []
  vel_errors = []

  for i in range(n):
    gx, gy, gz = gpu_boids[i][0], gpu_boids[i][1], gpu_boids[i][2]
    rx, ry, rz = ref_boids[i][0], ref_boids[i][1], ref_boids[i][2]
    pos_err = math.sqrt((gx - rx)**2 + (gy - ry)**2 + (gz - rz)**2)
    pos_errors.append(pos_err)

    gvx, gvy, gvz = gpu_boids[i][3], gpu_boids[i][4], gpu_boids[i][5]
    rvx, rvy, rvz = ref_boids[i][3], ref_boids[i][4], ref_boids[i][5]
    vel_err = math.sqrt((gvx - rvx)**2 + (gvy - rvy)**2 + (gvz - rvz)**2)
    vel_errors.append(vel_err)

  avg_pos = np.mean(pos_errors)
  max_pos = np.max(pos_errors)
  avg_vel = np.mean(vel_errors)
  pos_ok = sum(1 for e in pos_errors if e < pos_tol)

  if avg_pos < pos_tol * 0.1 and avg_vel < vel_tol * 0.1:
    score = 1.0
    quality = "excellent"
  elif avg_pos < pos_tol and avg_vel < vel_tol:
    score = 0.8
    quality = "good"
  elif pos_ok > n * 0.9:
    score = 0.5
    quality = "partial"
  elif pos_ok > n * 0.5:
    score = 0.3
    quality = "poor"
  else:
    score = 0.0
    quality = "failed"

  desc = (f"{n} boids: {quality}. "
          f"Avg pos error: {avg_pos:.4f}, max: {max_pos:.4f}, "
          f"avg vel error: {avg_vel:.4f}, "
          f"{pos_ok}/{n} within tolerance {pos_tol}")
  return score, desc


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

GLSL_INTERFACE = """**GLSL Compute Shader Interface:**

```glsl
#version 450

layout(local_size_x = 256) in;

// Each boid stored as two vec4s: [pos.xyz, 0], [vel.xyz, 0]
// Boid i is at index i*2 and i*2+1
layout(set = 0, binding = 0) readonly buffer InputBoids {
    vec4 inData[];   // length = N * 2
};

layout(set = 0, binding = 1) buffer OutputBoids {
    vec4 outData[];  // length = N * 2
};

layout(set = 0, binding = 2) uniform Params {
    uvec4 counts;      // counts.x = N
    vec4 radii;        // radii.x=dt, radii.y=sepRadius, radii.z=alignRadius, radii.w=cohRadius
    vec4 weights;      // weights.x=sepWeight, weights.y=alignWeight, weights.z=cohWeight, weights.w=maxSpeed
    vec4 bounds;       // bounds.x=boundSize, bounds.y=boundForce
};
```

**Task:** Write a compute shader that performs ONE timestep of 3D boid
flocking simulation (Reynolds' rules). Each invocation handles one boid
(gl_GlobalInvocationID.x), reading its position/velocity from inData and
writing updated values to outData. Use separation/align/cohesion with the
provided radii and weights, average neighbor contributions, and apply the
boundary force when a component exceeds boundSize*0.4 (push toward center
using boundForce and boundSize*0.5 scaling). Update velocity by accel*dt,
clamp speed to maxSpeed, then update position by vel*dt.

**Critical Requirements:**
- Guard against gl_GlobalInvocationID.x >= N
- Skip self (i == j) and zero-distance neighbors
- Average the accumulated forces by neighbor count (avoid division by zero)
- Clamp velocity magnitude to maxSpeed
"""


def prepareSubpassPrompt(subPass):
  if subPass != 0:
    raise StopIteration

  configs = []
  for i, (n, steps, dt) in enumerate(SUBPASSES):
    configs.append(f"  Subpass {i}: N={n}, timesteps={steps}, dt={dt}")

  return f"""Write a GLSL compute shader for 3D boid flocking simulation.

{GLSL_INTERFACE}

**Flocking Parameters (constant across all subpasses):**
  Separation radius: {SEP_RADIUS}, weight: {SEP_WEIGHT}
  Alignment radius: {ALIGN_RADIUS}, weight: {ALIGN_WEIGHT}
  Cohesion radius: {COHESION_RADIUS}, weight: {COHESION_WEIGHT}
  Max speed: {MAX_SPEED}
  Bound size: {BOUND_SIZE}, bound force: {BOUND_FORCE}

**Test Configurations:**
{chr(10).join(configs)}

Write the complete GLSL compute shader source code.
The shader performs ONE step. It will be dispatched multiple times with
ping-pong buffers for the required number of timesteps.
"""


extraGradeAnswerRuns = list(range(1, len(SUBPASSES)))

structure = {
  "type": "object",
  "properties": {
    "reasoning": {
      "type": "string",
      "description": "Explain your approach to the boid flocking compute shader"
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

  n, timesteps, dt = SUBPASSES[subPass]

  # Generate test data
  boids = generate_boids(n)
  input_data = boids_to_buffer(boids)
  params = make_params_buffer(n, dt)

  # CPU reference (cached)
  cache_key = (n, timesteps)
  if cache_key not in _ref_cache:
    print(f"  Computing CPU boid reference for N={n}, steps={timesteps}...")
    _ref_cache[cache_key] = cpu_reference(boids, timesteps, dt)
  ref_boids = _ref_cache[cache_key]

  # Run on GPU
  try:
    if _runner_cache is None:
      _runner_cache = ComputeShaderRunner()

    workgroups = ((n + 255) // 256, 1, 1)

    def verify_fn(output_bytes):
      gpu_boids = buffer_to_boids(output_bytes, n)
      # Scale tolerance with timesteps
      pos_tol = 1.0 * (1 + timesteps / 100)
      vel_tol = 2.0 * (1 + timesteps / 100)
      return compare_boids(gpu_boids, ref_boids, pos_tol, vel_tol)

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
