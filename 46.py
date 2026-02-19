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
import hashlib
from html import escape
from typing import Tuple, Optional

import numpy as np

from shader_test_utils import compile_glsl, validate_spirv
from compute_test_utils import ComputeShaderRunner, grade_compute_pingpong
from solver_utils import GradeCache

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


def compare_bodies(gpu_bodies, ref_bodies, pos_tol=0.1, vel_tol=0.5, return_details=False):
  """Compare GPU and CPU body results. Returns (score, description[, details])."""
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
  if return_details:
    return score, desc, {
      "pos_errors": pos_errors,
      "vel_errors": vel_errors,
      "avg_pos_err": avg_pos_err,
      "max_pos_err": max_pos_err,
      "avg_vel_err": avg_vel_err,
    }
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
N-body simulation using the leapfrog/Euler integration method. Each invocation
handles one body (gl_GlobalInvocationID.x), reading its position and velocity
from inData and writing updated values to outData. The acceleration should be
computed from all other bodies using:

  acc += G * mass_j * (pos_j - pos_i) / (|pos_j - pos_i|^2 + softening^2)^(3/2)

Velocity is advanced by acc * dt and position is advanced by vel * dt. The
shader will be dispatched ceil(N/256) workgroups in X and run multiple times
with ping-pong buffers for the required number of timesteps.

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
_REPORT_CACHE = {}
_grade_cache = GradeCache('test46')


def _cache_key_parts(result, subPass):
  code = result.get("shader_code", "")
  n, timesteps, dt, softening = SUBPASSES[subPass]
  return (
    hashlib.sha256(code.encode('utf-8')).hexdigest()[:16],
    f"n={n}|steps={timesteps}|dt={dt}|soft={softening}|g={G_CONSTANT}",
  )


def gradeAnswer(result, subPass, aiEngineName):
  global _runner_cache, _ref_cache

  if not result or "shader_code" not in result:
    return 0.0, "No shader code provided"

  cache_parts = _cache_key_parts(result, subPass)
  cached = _grade_cache.get_grade(*cache_parts)
  if cached is not None:
    return cached

  code = result["shader_code"]

  # Compile GLSL to SPIR-V
  try:
    spirv = compile_glsl(code, stage="comp")
  except RuntimeError as e:
    grade = (0.0, f"GLSL compilation failed: {e}")
    _grade_cache.put_grade(grade, *cache_parts)
    return grade

  valid, err = validate_spirv(spirv)
  if not valid:
    grade = (0.0, f"SPIR-V validation failed: {err}")
    _grade_cache.put_grade(grade, *cache_parts)
    return grade

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
      score, desc, details = compare_bodies(
        gpu_bodies, ref_bodies, pos_tol, vel_tol, return_details=True)
      _REPORT_CACHE[(aiEngineName, subPass)] = {
        "n": n,
        "timesteps": timesteps,
        "dt": dt,
        "softening": softening,
        "pos_tol": pos_tol,
        "vel_tol": vel_tol,
        "gpu_bodies": gpu_bodies,
        "ref_bodies": ref_bodies,
        "details": details,
        "score": score,
        "desc": desc,
      }
      return score, desc

    grade = grade_compute_pingpong(spirv,
                                   input_data,
                                   extra_buffers={2: params},
                                   extra_types={2: 'uniform'},
                                   workgroups=workgroups,
                                   iterations=timesteps,
                                   verify_fn=verify_fn,
                                   runner=_runner_cache,
                                   timeout=TIMEOUT_SECONDS)
    _grade_cache.put_grade(grade, *cache_parts)
    return grade

  except Exception as e:
    grade = (0.0, f"GPU execution failed: {e}")
    _grade_cache.put_grade(grade, *cache_parts)
    return grade


def _build_svg_projection(samples, width=520, height=360):
  if not samples:
    return ""
  xs = [s[0] for s in samples]
  ys = [s[1] for s in samples]
  min_x, max_x = min(xs), max(xs)
  min_y, max_y = min(ys), max(ys)
  span_x = max(max_x - min_x, 1e-6)
  span_y = max(max_y - min_y, 1e-6)

  def map_x(x):
    return 20 + (x - min_x) / span_x * (width - 40)

  def map_y(y):
    return height - 20 - (y - min_y) / span_y * (height - 40)

  circles = []
  for x, y, err, ok in samples:
    cx = map_x(x)
    cy = map_y(y)
    color = "#22c55e" if ok else "#ef4444"
    alpha = 0.35 if ok else 0.65
    r = 2.2 if ok else 2.8
    circles.append(
      f"<circle cx='{cx:.2f}' cy='{cy:.2f}' r='{r:.2f}' fill='{color}' "
      f"fill-opacity='{alpha}' />")

  return (
    f"<svg width='{width}' height='{height}' viewBox='0 0 {width} {height}' "
    f"style='background:#0b1220;border:1px solid #1f2937;border-radius:6px'>"
    f"<rect x='10' y='10' width='{width-20}' height='{height-20}' "
    f"fill='none' stroke='#1f2937' stroke-width='1' />"
    + "".join(circles) + "</svg>")


def resultToNiceReport(result, subPass, aiEngineName):
  cache_parts = _cache_key_parts(result or {}, subPass)
  cached = _grade_cache.get_report(*cache_parts)
  if cached is not None:
    return cached

  report = _REPORT_CACHE.get((aiEngineName, subPass))
  if not report:
    return ""

  n = report["n"]
  details = report["details"]
  pos_errors = details["pos_errors"]
  vel_errors = details["vel_errors"]
  pos_tol = report["pos_tol"]
  vel_tol = report["vel_tol"]

  samples = []
  step = max(1, n // 300)
  for i in range(0, n, step):
    gx, gy, gz = report["gpu_bodies"][i][0:3]
    rx, ry, rz = report["ref_bodies"][i][0:3]
    err = pos_errors[i]
    ok = err < pos_tol
    samples.append((gx, gy, err, ok))

  xy_svg = _build_svg_projection(samples)

  samples_xz = []
  for i in range(0, n, step):
    gx, gy, gz = report["gpu_bodies"][i][0:3]
    err = pos_errors[i]
    ok = err < pos_tol
    samples_xz.append((gx, gz, err, ok))
  xz_svg = _build_svg_projection(samples_xz)

  avg_pos = details["avg_pos_err"]
  max_pos = details["max_pos_err"]
  avg_vel = details["avg_vel_err"]
  within = sum(1 for e in pos_errors if e < pos_tol)
  desc = escape(report["desc"])

  html = f"""
  <div style='display:grid;gap:12px;margin:10px 0;'>
    <div style='padding:10px;border:1px solid #e5e7eb;border-radius:8px;background:#f8fafc;'>
      <div style='font-weight:600;margin-bottom:4px;'>N-body validation snapshot</div>
      <div style='font-size:12px;color:#475569;'>
        N={n} · timesteps={report['timesteps']} · dt={report['dt']} · softening={report['softening']}<br>
        Pos tol={pos_tol:.3f} · Vel tol={vel_tol:.3f} · Within tol: {within}/{n}<br>
        Avg pos err={avg_pos:.4f} · Max pos err={max_pos:.4f} · Avg vel err={avg_vel:.4f}
      </div>
      <div style='margin-top:6px;font-size:12px;color:#0f172a;'>{desc}</div>
    </div>
    <div style='display:grid;grid-template-columns:1fr 1fr;gap:12px;'>
      <div>
        <div style='font-size:12px;color:#64748b;margin-bottom:4px;'>GPU positions XY (green=within tol, red=out)</div>
        {xy_svg}
      </div>
      <div>
        <div style='font-size:12px;color:#64748b;margin-bottom:4px;'>GPU positions XZ (green=within tol, red=out)</div>
        {xz_svg}
      </div>
    </div>
  </div>
  """

  _grade_cache.put_report(html, *cache_parts)
  return html
