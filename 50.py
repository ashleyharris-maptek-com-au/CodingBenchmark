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

import base64
import json
import math
import os
import random
import struct
import subprocess
import sys
import tempfile
import time
from typing import Tuple, Optional

import numpy as np

from shader_test_utils import compile_glsl, validate_spirv
from compute_test_utils import ComputeShaderRunner, grade_compute_pingpong
from native_compiler import CppCompiler, CompilationError, ExecutionError

title = "Boid Flocking Simulation (GLSL Compute)"

tags = [
  "glsl",
  "structured response",
  "gpu compute",
  "simulation",
]

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

# ── C++ reference implementation ──────────────────────────────────────────────
_BOID_CPP_SOURCE = r"""
#include <cstdio>
#include <cmath>
#include <vector>

int main() {
    int N, timesteps, capture_every;
    double dt;
    double SEP_R, ALIGN_R, COH_R;
    double SEP_W, ALIGN_W, COH_W, MAX_SPD;
    double BOUND_SZ, BOUND_F;

    if (scanf("%d %d %d", &N, &timesteps, &capture_every) != 3) return 1;
    if (scanf("%lf", &dt) != 1) return 1;
    if (scanf("%lf %lf %lf", &SEP_R, &ALIGN_R, &COH_R) != 3) return 1;
    if (scanf("%lf %lf %lf %lf", &SEP_W, &ALIGN_W, &COH_W, &MAX_SPD) != 4) return 1;
    if (scanf("%lf %lf", &BOUND_SZ, &BOUND_F) != 2) return 1;

    std::vector<double> px(N), py(N), pz(N);
    std::vector<double> vx(N), vy(N), vz(N);
    for (int i = 0; i < N; i++) {
        if (scanf("%lf %lf %lf %lf %lf %lf",
                  &px[i], &py[i], &pz[i], &vx[i], &vy[i], &vz[i]) != 6) return 1;
    }

    std::vector<double> npx(N), npy(N), npz(N);
    std::vector<double> nvx(N), nvy(N), nvz(N);

    double half = BOUND_SZ * 0.5;
    double boundary_thresh = half * 0.8;

    for (int step = 0; step < timesteps; step++) {
        for (int i = 0; i < N; i++) {
            double sx=0, sy=0, sz=0;
            double ax_a=0, ay_a=0, az_a=0;
            double cx=0, cy=0, cz=0;
            int sc=0, ac=0, cc=0;

            for (int j = 0; j < N; j++) {
                if (i == j) continue;
                double dx = px[i] - px[j];
                double dy = py[i] - py[j];
                double dz = pz[i] - pz[j];
                double dist = sqrt(dx*dx + dy*dy + dz*dz);
                if (dist < 1e-6) continue;

                if (dist < SEP_R) {
                    sx += dx / dist;
                    sy += dy / dist;
                    sz += dz / dist;
                    sc++;
                }
                if (dist < ALIGN_R) {
                    ax_a += vx[j];
                    ay_a += vy[j];
                    az_a += vz[j];
                    ac++;
                }
                if (dist < COH_R) {
                    cx += px[j];
                    cy += py[j];
                    cz += pz[j];
                    cc++;
                }
            }

            double acx=0, acy=0, acz=0;
            if (sc > 0) {
                acx += SEP_W * sx / sc;
                acy += SEP_W * sy / sc;
                acz += SEP_W * sz / sc;
            }
            if (ac > 0) {
                acx += ALIGN_W * (ax_a / ac - vx[i]);
                acy += ALIGN_W * (ay_a / ac - vy[i]);
                acz += ALIGN_W * (az_a / ac - vz[i]);
            }
            if (cc > 0) {
                acx += COH_W * (cx / cc - px[i]);
                acy += COH_W * (cy / cc - py[i]);
                acz += COH_W * (cz / cc - pz[i]);
            }

            if (fabs(px[i]) > boundary_thresh) acx -= BOUND_F * (px[i] / half);
            if (fabs(py[i]) > boundary_thresh) acy -= BOUND_F * (py[i] / half);
            if (fabs(pz[i]) > boundary_thresh) acz -= BOUND_F * (pz[i] / half);

            double tvx = vx[i] + acx * dt;
            double tvy = vy[i] + acy * dt;
            double tvz = vz[i] + acz * dt;
            double spd = sqrt(tvx*tvx + tvy*tvy + tvz*tvz);
            if (spd > MAX_SPD && spd > 0) {
                double s = MAX_SPD / spd;
                tvx *= s; tvy *= s; tvz *= s;
            }
            npx[i] = px[i] + tvx * dt;
            npy[i] = py[i] + tvy * dt;
            npz[i] = pz[i] + tvz * dt;
            nvx[i] = tvx; nvy[i] = tvy; nvz[i] = tvz;
        }

        for (int i = 0; i < N; i++) {
            px[i]=npx[i]; py[i]=npy[i]; pz[i]=npz[i];
            vx[i]=nvx[i]; vy[i]=nvy[i]; vz[i]=nvz[i];
        }

        if (capture_every > 0 && (step % capture_every == 0 || step == timesteps - 1)) {
            printf("FRAME\n");
            for (int i = 0; i < N; i++) {
                printf("%.17g %.17g %.17g %.17g %.17g %.17g\n",
                       px[i], py[i], pz[i], vx[i], vy[i], vz[i]);
            }
        }
    }

    printf("FINAL\n");
    for (int i = 0; i < N; i++) {
        printf("%.17g %.17g %.17g %.17g %.17g %.17g\n",
               px[i], py[i], pz[i], vx[i], vy[i], vz[i]);
    }
    return 0;
}
"""

_cpp_compiler = None
_cpp_exe = None


def _get_cpp_exe():
  """Compile the C++ boid reference binary (cached by source hash)."""
  global _cpp_compiler, _cpp_exe
  if _cpp_exe is not None and _cpp_exe.exists():
    return _cpp_exe
  _cpp_compiler = CppCompiler("_reference")
  if _cpp_compiler.find_compiler() is None:
    raise CompilationError("No C++ compiler found for boid reference")
  _cpp_exe = _cpp_compiler.compile(_BOID_CPP_SOURCE)
  return _cpp_exe


def _parse_cpp_boid_output(stdout, n):
  """Parse C++ output into (final_boids, frames_list)."""
  frames = []
  final = None
  current_section = None
  current_boids = []

  for line in stdout.split("\n"):
    line = line.strip()
    if line == "FRAME":
      if current_section == "FRAME" and len(current_boids) == n:
        frames.append(current_boids)
      current_section = "FRAME"
      current_boids = []
    elif line == "FINAL":
      if current_section == "FRAME" and len(current_boids) == n:
        frames.append(current_boids)
      current_section = "FINAL"
      current_boids = []
    elif line:
      vals = line.split()
      if len(vals) == 6:
        current_boids.append(tuple(float(v) for v in vals))

  if current_section == "FINAL" and len(current_boids) == n:
    final = current_boids

  return final, frames


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


def _cpu_boid_step_python(boids, dt):
  """Fallback: one step of boid simulation on CPU (pure Python)."""
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


def _cpu_reference_python(boids, timesteps, dt):
  """Fallback: pure Python boid simulation."""
  b = list(boids)
  for _ in range(timesteps):
    b = _cpu_boid_step_python(b, dt)
  return b


def _cpu_reference_with_frames_python(boids, timesteps, dt, max_frames=60):
  """Fallback: pure Python boid simulation with frame capture."""
  b = list(boids)
  frames = []
  if timesteps <= max_frames:
    capture_every = 1
  else:
    capture_every = max(1, timesteps // max_frames)
  for step in range(timesteps):
    b = _cpu_boid_step_python(b, dt)
    if step % capture_every == 0 or step == timesteps - 1:
      frames.append([(x, y, z, vx, vy, vz) for x, y, z, vx, vy, vz in b])
  return b, frames


def _build_cpp_stdin(boids, timesteps, dt, capture_every):
  """Build stdin text for the C++ boid reference."""
  lines = [
    f"{len(boids)} {timesteps} {capture_every}",
    f"{dt!r}",
    f"{SEP_RADIUS!r} {ALIGN_RADIUS!r} {COHESION_RADIUS!r}",
    f"{SEP_WEIGHT!r} {ALIGN_WEIGHT!r} {COHESION_WEIGHT!r} {MAX_SPEED!r}",
    f"{BOUND_SIZE!r} {BOUND_FORCE!r}",
  ]
  for x, y, z, vx, vy, vz in boids:
    lines.append(f"{x!r} {y!r} {z!r} {vx!r} {vy!r} {vz!r}")
  return "\n".join(lines) + "\n"


def cpu_reference(boids, timesteps, dt):
  """Run boid simulation using compiled C++ (falls back to Python)."""
  n = len(boids)
  try:
    exe = _get_cpp_exe()
  except (CompilationError, Exception) as e:
    print(f"  C++ boid reference unavailable ({e}), falling back to Python...")
    return _cpu_reference_python(boids, timesteps, dt)

  stdin_data = _build_cpp_stdin(boids, timesteps, dt, 0)
  compiler = _cpp_compiler or CppCompiler("_reference")
  try:
    stdout, stderr, elapsed, rc = compiler.execute(exe, stdin_data=stdin_data, timeout=600)
  except ExecutionError as e:
    print(f"  C++ boid execution failed ({e}), falling back to Python...")
    return _cpu_reference_python(boids, timesteps, dt)

  if rc != 0:
    print(f"  C++ boid reference failed (rc={rc}): {stderr[:300]}")
    return _cpu_reference_python(boids, timesteps, dt)

  final, _ = _parse_cpp_boid_output(stdout, n)
  if final is None or len(final) != n:
    print(f"  C++ boid output parse error, falling back to Python...")
    return _cpu_reference_python(boids, timesteps, dt)
  return final


def cpu_reference_with_frames(boids, timesteps, dt, max_frames=60):
  """Run boid simulation with frame capture using compiled C++ (falls back to Python)."""
  n = len(boids)
  if timesteps <= max_frames:
    capture_every = 1
  else:
    capture_every = max(1, timesteps // max_frames)

  try:
    exe = _get_cpp_exe()
  except (CompilationError, Exception) as e:
    print(f"  C++ boid reference unavailable ({e}), falling back to Python...")
    return _cpu_reference_with_frames_python(boids, timesteps, dt, max_frames)

  stdin_data = _build_cpp_stdin(boids, timesteps, dt, capture_every)
  compiler = _cpp_compiler or CppCompiler("_reference")
  try:
    stdout, stderr, elapsed, rc = compiler.execute(exe, stdin_data=stdin_data, timeout=600)
  except ExecutionError as e:
    print(f"  C++ boid execution failed ({e}), falling back to Python...")
    return _cpu_reference_with_frames_python(boids, timesteps, dt, max_frames)

  if rc != 0:
    print(f"  C++ boid reference failed (rc={rc}): {stderr[:300]}")
    return _cpu_reference_with_frames_python(boids, timesteps, dt, max_frames)

  final, frames = _parse_cpp_boid_output(stdout, n)
  if final is None or len(final) != n:
    print(f"  C++ boid output parse error, falling back to Python...")
    return _cpu_reference_with_frames_python(boids, timesteps, dt, max_frames)
  return final, frames


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
_REPORT_CACHE = {}


def _grade_answer_inner(result, subPass, aiEngineName):
  global _runner_cache, _ref_cache

  if not result or "shader_code" not in result:
    return 0.0, "No shader code provided", {"error": "no_shader_code"}

  code = result["shader_code"]

  # Compile GLSL to SPIR-V
  try:
    spirv = compile_glsl(code, stage="comp")
  except RuntimeError as e:
    return 0.0, f"GLSL compilation failed: {e}", {"error": str(e)}

  valid, err = validate_spirv(spirv)
  if not valid:
    return 0.0, f"SPIR-V validation failed: {err}", {"error": err}

  n, timesteps, dt = SUBPASSES[subPass]

  # Generate test data
  boids = generate_boids(n)
  input_data = boids_to_buffer(boids)
  params = make_params_buffer(n, dt)

  # Decide whether to capture intermediate frames (only for small subpasses)
  max_viz_frames = 60
  want_viz = (n <= 512)  # subpasses 0-2
  if want_viz:
    capture_every = max(1, timesteps // max_viz_frames)
  else:
    capture_every = 0

  # CPU reference (cached)
  cache_key = (n, timesteps, capture_every)
  if cache_key not in _ref_cache:
    print(f"  Computing CPU boid reference for N={n}, steps={timesteps}...")
    if want_viz:
      ref_final, ref_frames = cpu_reference_with_frames(boids, timesteps, dt, max_frames=max_viz_frames)
    else:
      ref_final = cpu_reference(boids, timesteps, dt)
      ref_frames = []
    _ref_cache[cache_key] = (ref_final, ref_frames)
  ref_boids, ref_frames = _ref_cache[cache_key]

  report_data = {
    "n": n,
    "timesteps": timesteps,
    "dt": dt,
    "want_viz": want_viz,
    "capture_every": capture_every,
  }

  # Run on GPU
  try:
    if _runner_cache is None:
      _runner_cache = ComputeShaderRunner()

    workgroups = ((n + 255) // 256, 1, 1)

    def verify_fn(result):
      # result is (final_bytes, captured_frames_list) when capture_every > 0
      # result is just final_bytes when capture_every == 0
      if capture_every > 0:
        final_bytes, gpu_frame_bytes = result
      else:
        final_bytes = result
        gpu_frame_bytes = []

      gpu_boids = buffer_to_boids(final_bytes, n)
      pos_tol = 1.0 * (1 + timesteps / 100)
      vel_tol = 2.0 * (1 + timesteps / 100)
      score, desc = compare_boids(gpu_boids, ref_boids, pos_tol, vel_tol)

      # Compute per-boid position errors for stats
      pos_errors = []
      for i in range(n):
        gx, gy, gz = gpu_boids[i][0], gpu_boids[i][1], gpu_boids[i][2]
        rx, ry, rz = ref_boids[i][0], ref_boids[i][1], ref_boids[i][2]
        pos_errors.append(math.sqrt((gx-rx)**2 + (gy-ry)**2 + (gz-rz)**2))

      report_data.update({
        "pos_tol": float(pos_tol),
        "vel_tol": float(vel_tol),
        "pos_errors": pos_errors,
        "desc": desc,
        "score": score,
      })

      if want_viz:
        report_data.update({
          "initial_b64": base64.b64encode(boids_to_buffer(boids)).decode('ascii'),
          "gpu_final_b64": base64.b64encode(final_bytes).decode('ascii'),
          "ref_final_b64": base64.b64encode(boids_to_buffer(ref_boids)).decode('ascii'),
          "gpu_frames_b64": [base64.b64encode(fb).decode('ascii') for fb in gpu_frame_bytes],
          "ref_frames_b64": [base64.b64encode(boids_to_buffer(f)).decode('ascii') for f in ref_frames],
        })

      return score, desc

    grade = grade_compute_pingpong(spirv,
                                  input_data,
                                  extra_buffers={2: params},
                                  extra_types={2: 'uniform'},
                                  workgroups=workgroups,
                                  iterations=timesteps,
                                  verify_fn=verify_fn,
                                  runner=_runner_cache,
                                  timeout=TIMEOUT_SECONDS,
                                  capture_every=capture_every)
    return grade[0], grade[1], report_data

  except Exception as e:
    return 0.0, f"GPU execution failed: {e}", {"error": str(e)}


def gradeAnswer(result, subPass, aiEngineName):
  """Run grading in an isolated subprocess to survive GPU hangs/TDRs."""
  if not result or "shader_code" not in result:
    return 0.0, "No shader code provided"

  payload = {
    "shader_code": result.get("shader_code", ""),
    "subPass": subPass,
    "aiEngineName": aiEngineName,
  }

  with tempfile.TemporaryDirectory() as tmp_dir:
    in_path = os.path.join(tmp_dir, "grade_input.json")
    out_path = os.path.join(tmp_dir, "grade_output.json")
    with open(in_path, "w", encoding="utf-8") as f:
      json.dump(payload, f)

    cmd = [sys.executable, __file__, "--grade", in_path, out_path]
    try:
      subprocess.run(
        cmd,
        check=False,
        timeout=TIMEOUT_SECONDS + 10,
        creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0,
      )
    except subprocess.TimeoutExpired:
      return 0.0, "GPU execution timed out or hung (subprocess killed)"
    except Exception as e:
      return 0.0, f"Subprocess failed: {e}"

    if not os.path.exists(out_path):
      return 0.0, "Subprocess produced no result (crash or TDR)"

    try:
      with open(out_path, "r", encoding="utf-8") as f:
        out = json.load(f)
      score = out.get("score", 0.0)
      explanation = out.get("explanation", "No explanation")
      details = out.get("details", {}) or {}
      n = details.get("n")
      if n is not None:
        n = int(n)
        report = {
          "n": n,
          "timesteps": int(details.get("timesteps", 0)),
          "dt": float(details.get("dt", 0)),
          "pos_tol": float(details.get("pos_tol", 0)),
          "vel_tol": float(details.get("vel_tol", 0)),
          "pos_errors": details.get("pos_errors", []),
          "desc": details.get("desc", ""),
          "score": score,
        }

        want_viz = details.get("want_viz")
        if want_viz:
          initial_b64 = details.get("initial_b64")
          gpu_final_b64 = details.get("gpu_final_b64")
          ref_final_b64 = details.get("ref_final_b64")
          gpu_frames_b64 = details.get("gpu_frames_b64", [])
          ref_frames_b64 = details.get("ref_frames_b64", [])
          if initial_b64 and gpu_final_b64 and ref_final_b64:
            report.update({
              "initial_boids": buffer_to_boids(base64.b64decode(initial_b64), n),
              "gpu_boids": buffer_to_boids(base64.b64decode(gpu_final_b64), n),
              "ref_boids": buffer_to_boids(base64.b64decode(ref_final_b64), n),
              "gpu_frames": [buffer_to_boids(base64.b64decode(b), n) for b in gpu_frames_b64],
              "ref_frames": [buffer_to_boids(base64.b64decode(b), n) for b in ref_frames_b64],
            })
        _REPORT_CACHE[(aiEngineName, subPass)] = report
      return score, explanation
    except Exception as e:
      return 0.0, f"Failed to read subprocess result: {e}"


def _run_grade_subprocess(in_path: str, out_path: str) -> int:
  try:
    with open(in_path, "r", encoding="utf-8") as f:
      payload = json.load(f)
    result = {"shader_code": payload.get("shader_code", "")}
    subPass = int(payload.get("subPass", 0))
    aiEngineName = payload.get("aiEngineName", "")
    score, explanation, details = _grade_answer_inner(result, subPass, aiEngineName)
    with open(out_path, "w", encoding="utf-8") as f:
      json.dump({"score": score, "explanation": explanation, "details": details}, f)
    return 0
  except Exception as e:
    try:
      with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"score": 0.0, "explanation": f"Subprocess error: {e}",
                   "details": {"error": str(e)}}, f)
    except Exception:
      pass
    return 1


if __name__ == "__main__":
  if len(sys.argv) >= 4 and sys.argv[1] == "--grade":
    sys.exit(_run_grade_subprocess(sys.argv[2], sys.argv[3]))


def _stats_only_report(report):
  """Generate a compact stats-only HTML report for large subpasses."""
  from html import escape
  n = report['n']
  timesteps = report['timesteps']
  dt = report['dt']
  pos_tol = report['pos_tol']
  vel_tol = report['vel_tol']
  score = report['score']
  desc = escape(report['desc'])
  pos_errors = report['pos_errors']

  avg_pos = sum(pos_errors) / len(pos_errors)
  max_pos = max(pos_errors)
  within = sum(1 for e in pos_errors if e < pos_tol)
  pct = within / n * 100

  score_color = "#22c55e" if score >= 0.8 else "#eab308" if score >= 0.3 else "#ef4444"
  bar_pct = within / n * 100

  return f"""
  <div style='padding:10px;border:1px solid #e5e7eb;border-radius:8px;background:#f8fafc;margin:10px 0;'>
    <div style='font-weight:600;margin-bottom:6px;'>
      N={n} boids &middot; {timesteps} steps &middot; dt={dt}
      <span style='color:{score_color};margin-left:12px;'>score={score:.2f}</span>
    </div>
    <div style='font-size:12px;color:#475569;line-height:1.6;'>
      Avg pos error: <b>{avg_pos:.4f}</b> &middot;
      Max pos error: <b>{max_pos:.4f}</b> &middot;
      Within tolerance ({pos_tol:.2f}): <b>{within}/{n}</b> ({pct:.1f}%)
    </div>
    <div style='margin-top:6px;height:8px;background:#1e293b;border-radius:4px;overflow:hidden;'>
      <div style='height:100%;width:{bar_pct:.1f}%;background:{score_color};border-radius:4px;'></div>
    </div>
    <div style='font-size:11px;color:#64748b;margin-top:4px;'>{desc}</div>
  </div>
  """


def resultToNiceReport(result, subPass, aiEngineName):
  from visualization_utils import generate_threejs_boid_visualization

  report = _REPORT_CACHE.get((aiEngineName, subPass))
  if not report:
    return ''

  n = report['n']

  # For large subpasses (N > 512), show stats only
  if n > 512:
    return _stats_only_report(report)

  # Full 3D visualization for small subpasses
  timesteps = report['timesteps']
  dt = report['dt']
  initial_boids = report['initial_boids']
  gpu_frames = report.get('gpu_frames', [])
  ref_frames = report.get('ref_frames', [])
  gpu_boids = report['gpu_boids']
  ref_boids = report['ref_boids']
  pos_tol = report['pos_tol']
  score = report['score']

  def round_boids(boid_list):
    return [[round(b[0], 2), round(b[1], 2), round(b[2], 2),
             round(b[3], 2), round(b[4], 2), round(b[5], 2)] for b in boid_list]

  viz_gpu_frames = [round_boids(f) for f in gpu_frames]
  viz_ref_frames = [round_boids(f) for f in ref_frames]
  viz_initial = round_boids(initial_boids)
  viz_gpu_final = round_boids(gpu_boids)
  viz_ref_final = round_boids(ref_boids)

  return generate_threejs_boid_visualization(
    initial_boids=viz_initial,
    gpu_frames=viz_gpu_frames,
    ref_frames=viz_ref_frames,
    gpu_final=viz_gpu_final,
    ref_final=viz_ref_final,
    n_total=n,
    n_shown=n,
    timesteps=timesteps,
    dt=dt,
    pos_tol=pos_tol,
    score=score,
    bound_size=BOUND_SIZE,
  )


def setup():
  """Pre-compile C++ reference and compute all CPU boid references (called by --setup)."""
  print("  Compiling C++ boid reference...")
  try:
    _get_cpp_exe()
    print("  C++ boid reference compiled OK")
  except Exception as e:
    print(f"  WARNING: C++ compilation failed ({e}), will use Python fallback")

  print("  Pre-computing CPU boid references for all subpasses...")
  for i, (n, timesteps, dt) in enumerate(SUBPASSES):
    want_viz = (n <= 512)
    max_viz_frames = 60
    if want_viz:
      capture_every = max(1, timesteps // max_viz_frames)
    else:
      capture_every = 0
    cache_key = (n, timesteps, capture_every)
    if cache_key in _ref_cache:
      print(f"    Subpass {i}: N={n}, steps={timesteps} - cached (mem)")
      continue
    print(f"    Subpass {i}: N={n}, steps={timesteps} - computing...")
    t0 = time.time()
    boids = generate_boids(n)
    if want_viz:
      ref_final, ref_frames = cpu_reference_with_frames(boids, timesteps, dt, max_frames=max_viz_frames)
    else:
      ref_final = cpu_reference(boids, timesteps, dt)
      ref_frames = []
    _ref_cache[cache_key] = (ref_final, ref_frames)
    elapsed = time.time() - t0
    print(f"    Subpass {i}: done in {elapsed:.1f}s")


highLevelSummary = """
<p>Write a GPU compute shader in GLSL that simulates flocking behaviour (boids).
Each boid follows three simple rules &mdash; steer towards neighbours, match their
velocity, and avoid collisions &mdash; yet the flock as a whole produces complex,
lifelike motion.</p>
<p>The shader runs iteratively on the GPU and final boid positions are compared
against a CPU reference. Subpasses increase the flock size (up to thousands) and
timestep count, testing parallel neighbourhood queries and numerical accuracy.</p>
"""
