"""
Test 48: 2D Fluid Simulation - Lattice Boltzmann (SPIR-V Assembly Compute Shader)

The LLM must write a SPIR-V assembly compute shader that performs one step
of a 2D Lattice Boltzmann Method (LBM) simulation using the D2Q9 model.
The shader is dispatched repeatedly (ping-pong) for multiple timesteps.

This tests:
1. SPIR-V assembly compute shader proficiency
2. Understanding of Lattice Boltzmann method (streaming + collision)
3. Numerical precision in fluid dynamics
4. Performance on 2D grid computation

Buffer layout:
  Binding 0 (read):  Input grid - W*H*9 floats (D2Q9 distribution functions)
  Binding 1 (write): Output grid - W*H*9 floats
  Binding 2 (uniform): Params - uvec4(W, H, 0, 0), vec4(omega, 0, 0, 0)
                        omega = 1/(3*viscosity + 0.5) relaxation parameter
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
import zlib
from typing import Tuple, Optional

import numpy as np

from shader_test_utils import assemble_spirv, validate_spirv
from compute_test_utils import ComputeShaderRunner, grade_compute_pingpong

title = "2D Fluid Simulation - Lattice Boltzmann (SPIR-V ASM Compute)"

tags = [
  "spirv",
  "structured response",
  "gpu compute",
  "simulation",
]

RANDOM_SEED = 48484848
TIMEOUT_SECONDS = 60

# D2Q9 velocity directions: (cx, cy) and weights
D2Q9_CX = [0, 1, 0, -1, 0, 1, -1, -1, 1]
D2Q9_CY = [0, 0, 1, 0, -1, 1, 1, -1, -1]
D2Q9_W = [4.0 / 9, 1.0 / 9, 1.0 / 9, 1.0 / 9, 1.0 / 9, 1.0 / 36, 1.0 / 36, 1.0 / 36, 1.0 / 36]

# Subpass configurations: (width, height, timesteps, viscosity)
SUBPASSES = [
  (32, 32, 50, 0.1),
  (64, 64, 100, 0.1),
  (128, 128, 100, 0.05),
  (128, 128, 200, 0.05),
  (256, 256, 200, 0.02),
]


def init_grid(w, h, seed=RANDOM_SEED):
  """Initialize D2Q9 grid with a density perturbation in the center."""
  rng = random.Random(seed + w * h)
  # Start near equilibrium with a Gaussian density bump
  grid = np.zeros((h, w, 9), dtype=np.float64)

  for y in range(h):
    for x in range(w):
      # Base density with Gaussian perturbation
      dx = x - w / 2
      dy = y - h / 2
      rho = 1.0 + 0.2 * math.exp(-(dx * dx + dy * dy) / (w * 0.1)**2)
      # Small velocity perturbation
      ux = 0.02 * math.sin(2 * math.pi * y / h) + rng.gauss(0, 0.001)
      uy = 0.02 * math.cos(2 * math.pi * x / w) + rng.gauss(0, 0.001)

      # Equilibrium distribution
      for k in range(9):
        cu = D2Q9_CX[k] * ux + D2Q9_CY[k] * uy
        grid[y, x,
             k] = D2Q9_W[k] * rho * (1.0 + 3.0 * cu + 4.5 * cu * cu - 1.5 * (ux * ux + uy * uy))

  return grid


def grid_to_buffer(grid):
  """Pack grid (H,W,9) float64 -> float32 bytes."""
  return grid.astype(np.float32).tobytes()


def buffer_to_grid(data, w, h):
  """Unpack buffer to grid (H,W,9)."""
  arr = np.frombuffer(data, dtype=np.float32).copy()
  return arr.reshape(h, w, 9)


def make_params_buffer(w, h, viscosity):
  """Params: uvec4(W, H, 0, 0), vec4(omega, 0, 0, 0)."""
  omega = 1.0 / (3.0 * viscosity + 0.5)
  data = struct.pack('IIII', w, h, 0, 0)
  data += struct.pack('ffff', omega, 0.0, 0.0, 0.0)
  return data


def cpu_lbm_step(grid, w, h, omega):
  """One LBM step: collision + streaming with bounce-back boundaries."""
  new_grid = np.zeros_like(grid)

  for y in range(h):
    for x in range(w):
      # Compute macroscopic quantities
      rho = 0.0
      ux = 0.0
      uy = 0.0
      for k in range(9):
        rho += grid[y, x, k]
        ux += D2Q9_CX[k] * grid[y, x, k]
        uy += D2Q9_CY[k] * grid[y, x, k]
      if rho > 0:
        ux /= rho
        uy /= rho

      # Collision (BGK)
      for k in range(9):
        cu = D2Q9_CX[k] * ux + D2Q9_CY[k] * uy
        feq = D2Q9_W[k] * rho * (1.0 + 3.0 * cu + 4.5 * cu * cu - 1.5 * (ux * ux + uy * uy))
        f_out = grid[y, x, k] + omega * (feq - grid[y, x, k])

        # Streaming: move to neighbor
        nx = (x + D2Q9_CX[k]) % w
        ny = (y + D2Q9_CY[k]) % h
        new_grid[ny, nx, k] = f_out

  return new_grid


def cpu_reference(grid, w, h, timesteps, viscosity):
  """Run LBM for timesteps on CPU."""
  omega = 1.0 / (3.0 * viscosity + 0.5)
  g = grid.copy()
  for _ in range(timesteps):
    g = cpu_lbm_step(g, w, h, omega)
  return g


def compute_mass(grid):
  """Compute total mass (sum of all distribution functions)."""
  return np.sum(grid)


def compare_grids(gpu_grid, ref_grid, mass_tol=0.01, val_tol=0.05):
  """Compare GPU and CPU grids. Returns (score, description)."""
  gpu_mass = compute_mass(gpu_grid)
  ref_mass = compute_mass(ref_grid)
  mass_err = abs(gpu_mass - ref_mass) / max(abs(ref_mass), 1e-10)

  # Per-cell comparison
  diff = np.abs(gpu_grid.astype(np.float64) - ref_grid)
  max_err = np.max(diff)
  avg_err = np.mean(diff)
  rel_err = np.mean(diff / (np.abs(ref_grid) + 1e-10))

  if mass_err < mass_tol * 0.01 and rel_err < val_tol * 0.1:
    score = 1.0
    quality = "excellent"
  elif mass_err < mass_tol and rel_err < val_tol:
    score = 0.8
    quality = "good"
  elif mass_err < mass_tol * 5:
    score = 0.4
    quality = "partial (mass roughly conserved)"
  else:
    score = 0.0
    quality = "failed"

  w, h = gpu_grid.shape[1], gpu_grid.shape[0]
  desc = (f"{w}x{h} grid: {quality}. "
          f"Mass error: {mass_err:.6f}, avg cell error: {avg_err:.6f}, "
          f"max cell error: {max_err:.6f}, mean relative error: {rel_err:.6f}")
  return score, desc


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

SPIRV_INTERFACE = """**SPIR-V Assembly Compute Shader Interface:**

The shader uses the D2Q9 Lattice Boltzmann model for 2D fluid simulation.

**D2Q9 velocity directions (index k -> (cx, cy)):**
```
k=0: ( 0, 0)  w=4/9    (rest)
k=1: ( 1, 0)  w=1/9    (east)
k=2: ( 0, 1)  w=1/9    (north)
k=3: (-1, 0)  w=1/9    (west)
k=4: ( 0,-1)  w=1/9    (south)
k=5: ( 1, 1)  w=1/36   (NE)
k=6: (-1, 1)  w=1/36   (NW)
k=7: (-1,-1)  w=1/36   (SW)
k=8: ( 1,-1)  w=1/36   (SE)
```

**Buffer layout:**
- Binding 0 (storage, read-only): float array, length W*H*9
  Index = (y * W + x) * 9 + k
- Binding 1 (storage, read-write): float array, length W*H*9 (output, NOT writeonly)
- Binding 2 (uniform): { uvec4(W, H, 0, 0), vec4(omega, 0, 0, 0) }

**Algorithm (per cell at gl_GlobalInvocationID.xy):**
Compute one LBM step (collision + streaming) for each lattice cell. Guard against
out-of-bounds (x >= W or y >= H). For the cell distributions f[0..8], compute
macroscopic density rho and velocity (ux, uy) using the D2Q9 directions. Apply
the BGK collision update with:

  cu = cx[k]*ux + cy[k]*uy
  feq = w[k] * rho * (1 + 3*cu + 4.5*cu*cu - 1.5*(ux*ux + uy*uy))
  f_out = f[k] + omega * (feq - f[k])

Then stream each f_out to the neighbor cell (x+cx[k], y+cy[k]) with periodic
wraparound in both axes. Write to the output buffer at the corresponding
neighbor index for k.

**SPIR-V requirements:**
- OpCapability Shader
- OpExecutionMode with LocalSize 16 16 1
- Use gl_GlobalInvocationID for (x, y)
- Declare runtime arrays for storage buffers with OpDecorate ArrayStride 4
- Use OpSMod or manual modulo for periodic boundary conditions
"""


def prepareSubpassPrompt(subPass):
  if subPass != 0:
    raise StopIteration

  configs = []
  for i, (w, h, steps, visc) in enumerate(SUBPASSES):
    omega = 1.0 / (3.0 * visc + 0.5)
    configs.append(f"  Subpass {i}: {w}x{h} grid, {steps} steps, "
                   f"viscosity={visc}, omega={omega:.4f}")

  return f"""Write a SPIR-V assembly compute shader for 2D Lattice Boltzmann fluid simulation.

{SPIRV_INTERFACE}

**Test Configurations:**
{chr(10).join(configs)}

Write the complete SPIR-V assembly text (starting with "; SPIR-V").
The shader performs ONE LBM step (collision + streaming). It will be
dispatched repeatedly for the required number of timesteps with ping-pong buffers.
"""


extraGradeAnswerRuns = list(range(1, len(SUBPASSES)))

structure = {
  "type": "object",
  "properties": {
    "reasoning": {
      "type": "string",
      "description": "Explain your approach to the LBM compute shader"
    },
    "spirv_code": {
      "type": "string",
      "description": "Complete SPIR-V assembly text for the compute shader"
    }
  },
  "required": ["reasoning", "spirv_code"],
  "additionalProperties": False
}

_runner_cache = None
_ref_cache = {}
_REPORT_CACHE = {}


def _grade_answer_inner(result, subPass, aiEngineName):
  global _runner_cache, _ref_cache

  if not result or "spirv_code" not in result:
    return 0.0, "No SPIR-V code provided", {"error": "no_spirv_code"}

  code = result["spirv_code"]

  # Assemble SPIR-V
  try:
    spirv = assemble_spirv(code)
  except RuntimeError as e:
    return 0.0, f"SPIR-V assembly failed: {e}", {"error": str(e)}

  valid, err = validate_spirv(spirv)
  if not valid:
    return 0.0, f"SPIR-V validation failed: {err}", {"error": err}

  w, h, timesteps, viscosity = SUBPASSES[subPass]

  # Generate initial grid
  grid = init_grid(w, h)
  input_data = grid_to_buffer(grid)
  params = make_params_buffer(w, h, viscosity)

  # CPU reference (cached)
  cache_key = (w, h, timesteps, viscosity)
  if cache_key not in _ref_cache:
    print(f"  Computing CPU LBM reference for {w}x{h}, {timesteps} steps...")
    _ref_cache[cache_key] = cpu_reference(grid, w, h, timesteps, viscosity)
  ref_grid = _ref_cache[cache_key]
  ref_bytes = ref_grid.astype(np.float32).tobytes()
  ref_b64 = base64.b64encode(ref_bytes).decode('ascii')

  report_data = {
    "w": w,
    "h": h,
    "timesteps": timesteps,
    "viscosity": viscosity,
    "ref_grid_b64": ref_b64,
  }

  # Run on GPU
  try:
    if _runner_cache is None:
      _runner_cache = ComputeShaderRunner()

    workgroups = ((w + 15) // 16, (h + 15) // 16, 1)

    def verify_fn(output_bytes):
      gpu_grid = buffer_to_grid(output_bytes, w, h)
      # Scale tolerance with timesteps
      val_tol = 0.05 * (1 + timesteps / 200)
      score, desc = compare_grids(gpu_grid, ref_grid, mass_tol=0.01, val_tol=val_tol)
      report_data.update({
        "gpu_grid_b64": base64.b64encode(output_bytes).decode('ascii'),
        "score": score,
        "desc": desc,
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
                                   timeout=TIMEOUT_SECONDS)
    return grade[0], grade[1], report_data

  except Exception as e:
    return 0.0, f"GPU execution failed: {e}", {"error": str(e)}


def gradeAnswer(result, subPass, aiEngineName):
  """Run grading in an isolated subprocess to survive GPU hangs/TDRs."""
  if not result or "spirv_code" not in result:
    return 0.0, "No SPIR-V code provided"

  payload = {
    "spirv_code": result.get("spirv_code", ""),
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
      w = details.get("w")
      h = details.get("h")
      timesteps = details.get("timesteps")
      viscosity = details.get("viscosity")
      gpu_b64 = details.get("gpu_grid_b64")
      ref_b64 = details.get("ref_grid_b64")
      if w and h and gpu_b64:
        gpu_grid = buffer_to_grid(base64.b64decode(gpu_b64), int(w), int(h))
        report = {
          "gpu_grid": gpu_grid,
          "w": int(w),
          "h": int(h),
          "timesteps": int(timesteps),
          "viscosity": float(viscosity),
          "score": score,
          "desc": details.get("desc", ""),
        }
        _REPORT_CACHE[(aiEngineName, subPass)] = report
      if w and h and timesteps is not None and viscosity is not None and ref_b64:
        ref_grid = buffer_to_grid(base64.b64decode(ref_b64), int(w), int(h))
        _ref_cache[(int(w), int(h), int(timesteps), float(viscosity))] = ref_grid
      return score, explanation
    except Exception as e:
      return 0.0, f"Failed to read subprocess result: {e}"


def _run_grade_subprocess(in_path: str, out_path: str) -> int:
  try:
    with open(in_path, "r", encoding="utf-8") as f:
      payload = json.load(f)
    result = {"spirv_code": payload.get("spirv_code", "")}
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


# ---------------------------------------------------------------------------
# Visualization helpers
# ---------------------------------------------------------------------------

def _extract_density(grid):
  """Sum distribution functions to get macroscopic density per cell."""
  return np.sum(grid, axis=2)


def _cpu_lbm_step_fast(grid, w, h, omega):
  """Vectorized LBM step (numpy) for visualization frame generation."""
  rho = np.sum(grid, axis=2)
  ux = sum(D2Q9_CX[k] * grid[:, :, k] for k in range(9))
  uy = sum(D2Q9_CY[k] * grid[:, :, k] for k in range(9))
  mask = rho > 1e-10
  ux[mask] /= rho[mask]
  uy[mask] /= rho[mask]

  new_grid = np.zeros_like(grid)
  for k in range(9):
    cu = D2Q9_CX[k] * ux + D2Q9_CY[k] * uy
    feq = D2Q9_W[k] * rho * (1.0 + 3.0 * cu + 4.5 * cu**2
                              - 1.5 * (ux**2 + uy**2))
    f_out = grid[:, :, k] + omega * (feq - grid[:, :, k])
    new_grid[:, :, k] = np.roll(
      np.roll(f_out, D2Q9_CX[k], axis=1), D2Q9_CY[k], axis=0)

  return new_grid


def _downsample(data_2d, max_dim=96):
  """Stride-based downsampling for visualization."""
  h, w = data_2d.shape
  if max(h, w) <= max_dim:
    return data_2d
  step = max(1, max(h, w) // max_dim)
  return data_2d[::step, ::step]


def _colormap_fluid(t):
  """Ocean-thermal colormap for fluid density."""
  t = max(0.0, min(1.0, t))
  stops = [
    (0.00, (8, 12, 60)),  (0.15, (15, 40, 130)),
    (0.30, (20, 100, 180)), (0.50, (40, 180, 170)),
    (0.65, (120, 210, 100)), (0.80, (220, 220, 60)),
    (0.95, (250, 160, 40)), (1.00, (255, 255, 255)),
  ]
  for i in range(len(stops) - 1):
    t0, c0 = stops[i]
    t1, c1 = stops[i + 1]
    if t <= t1:
      s = (t - t0) / (t1 - t0)
      return tuple(int(c0[j] + s * (c1[j] - c0[j])) for j in range(3))
  return stops[-1][1]


def _colormap_error(t):
  """Error heatmap: black -> red -> orange -> yellow -> white."""
  t = max(0.0, min(1.0, t))
  stops = [
    (0.00, (0, 0, 4)),    (0.25, (100, 8, 8)),
    (0.50, (200, 40, 15)), (0.75, (250, 180, 40)),
    (1.00, (255, 255, 255)),
  ]
  for i in range(len(stops) - 1):
    t0, c0 = stops[i]
    t1, c1 = stops[i + 1]
    if t <= t1:
      s = (t - t0) / (t1 - t0)
      return tuple(int(c0[j] + s * (c1[j] - c0[j])) for j in range(3))
  return stops[-1][1]


def _png_chunk(tag, data):
  body = tag + data
  return (struct.pack('>I', len(data)) + body
          + struct.pack('>I', zlib.crc32(body) & 0xFFFFFFFF))


def _make_heatmap_png(data_2d, colormap_fn, max_dim=96,
                      vmin=None, vmax=None):
  """Render a 2D numpy array as a base64-encoded PNG heatmap."""
  data_2d = _downsample(data_2d, max_dim)
  h, w = data_2d.shape
  if vmin is None:
    vmin = float(np.min(data_2d))
  if vmax is None:
    vmax = float(np.max(data_2d))
  if vmax - vmin < 1e-10:
    vmax = vmin + 1.0
  data_2d = data_2d[::-1]  # flip y so row 0 is at bottom
  norm = np.clip((data_2d - vmin) / (vmax - vmin), 0.0, 1.0)

  raw = bytearray()
  for y in range(h):
    raw.append(0)  # PNG row filter: None
    for x in range(w):
      r, g, b = colormap_fn(float(norm[y, x]))
      raw.extend([r, g, b])

  sig = b'\x89PNG\r\n\x1a\n'
  ihdr = _png_chunk(b'IHDR', struct.pack('>IIBBBBB', w, h, 8, 2, 0, 0, 0))
  idat = _png_chunk(b'IDAT', zlib.compress(bytes(raw), 9))
  iend = _png_chunk(b'IEND', b'')
  return base64.b64encode(sig + ihdr + idat + iend).decode('ascii')


def _frame_html(b64_png, label, px=140):
  return (
    f"<div style='text-align:center;flex-shrink:0;'>"
    f"<img src='data:image/png;base64,{b64_png}' "
    f"style='width:{px}px;height:{px}px;image-rendering:pixelated;"
    f"border:1px solid #334155;border-radius:4px;display:block;'>"
    f"<div style='font-size:10px;color:#94a3b8;margin-top:3px;'>"
    f"{label}</div></div>"
  )


# ---------------------------------------------------------------------------
# HTML report
# ---------------------------------------------------------------------------

def resultToNiceReport(result, subPass, aiEngineName):
  from html import escape

  report = _REPORT_CACHE.get((aiEngineName, subPass))
  if not report:
    return "<div style='color:#94a3b8;'>No visualization data captured</div>"

  w = report["w"]
  h = report["h"]
  timesteps = report["timesteps"]
  viscosity = report["viscosity"]
  omega = 1.0 / (3.0 * viscosity + 0.5)
  score = report["score"]
  desc = escape(report["desc"])
  gpu_grid = report["gpu_grid"]

  # Recompute initial grid and fetch cached CPU reference
  init_g = init_grid(w, h)
  ref_grid = _ref_cache.get((w, h, timesteps, viscosity))
  init_rho = _extract_density(init_g)
  gpu_rho = _extract_density(gpu_grid)
  ref_rho = _extract_density(ref_grid) if ref_grid is not None else None

  # Consistent color range across all density panels
  rho_min = float(np.min(init_rho))
  rho_max = float(np.max(init_rho))
  for rho in [gpu_rho, ref_rho]:
    if rho is not None:
      rho_min = min(rho_min, float(np.min(rho)))
      rho_max = max(rho_max, float(np.max(rho)))

  # --- Evolution filmstrip (fast vectorized CPU LBM) ---
  n_frames = 6
  frame_times = sorted(set(
    int(timesteps * i / (n_frames - 1)) for i in range(n_frames)))

  evo_frames = []
  g = init_g.astype(np.float64).copy()
  step_done = 0
  for ft in frame_times:
    while step_done < ft:
      g = _cpu_lbm_step_fast(g, w, h, omega)
      step_done += 1
    rho = _extract_density(g)
    b64 = _make_heatmap_png(rho, _colormap_fluid,
                            vmin=rho_min, vmax=rho_max)
    evo_frames.append((f"t={ft}", b64))

  evo_panels = []
  for i, (label, b64) in enumerate(evo_frames):
    if i > 0:
      evo_panels.append(
        "<div style='color:#475569;font-size:18px;"
        "align-self:center;padding:0 1px;'>&#x203A;</div>")
    evo_panels.append(_frame_html(b64, label))

  evo_html = (
    "<div style='font-size:11px;color:#94a3b8;font-weight:600;"
    "margin-bottom:4px;'>Density evolution (CPU reference)</div>"
    "<div style='display:flex;gap:4px;align-items:flex-start;"
    "flex-wrap:wrap;margin-bottom:14px;'>"
    + "".join(evo_panels) + "</div>"
  )

  # --- Verification row: CPU ref | GPU output | Error | Stats ---
  verify_panels = []

  if ref_rho is not None:
    b64 = _make_heatmap_png(ref_rho, _colormap_fluid,
                            vmin=rho_min, vmax=rho_max)
    verify_panels.append(_frame_html(b64, f"CPU ref (t={timesteps})"))

  b64 = _make_heatmap_png(gpu_rho, _colormap_fluid,
                          vmin=rho_min, vmax=rho_max)
  verify_panels.append(_frame_html(b64, f"GPU output (t={timesteps})"))

  if ref_rho is not None:
    error = np.abs(gpu_rho - ref_rho)
    b64 = _make_heatmap_png(error, _colormap_error)
    verify_panels.append(_frame_html(b64, "|GPU &#x2212; CPU| error"))

  # Stats card
  status = "PASS" if score > 0 else "FAIL"
  sc = "#22c55e" if score > 0 else "#f97316"
  gpu_mass = float(np.sum(gpu_grid))
  ref_mass = float(np.sum(ref_grid)) if ref_grid is not None else gpu_mass
  mass_pct = abs(gpu_mass - ref_mass) / max(abs(ref_mass), 1e-10) * 100

  stats = (
    f"<div style='align-self:center;padding:10px 14px;font-size:11px;"
    f"color:#cbd5e1;background:#1e293b;border-radius:6px;"
    f"border:1px solid #334155;min-width:130px;'>"
    f"<div style='font-weight:700;color:{sc};font-size:15px;"
    f"margin-bottom:6px;'>{status}</div>"
    f"<div>Score: <b>{score:.1f}</b></div>"
    f"<div>Mass &#x394;: {mass_pct:.4f}%</div>"
    f"<div style='margin-top:4px;border-top:1px solid #334155;"
    f"padding-top:4px;'>Grid: {w}&#xd7;{h}</div>"
    f"<div>Steps: {timesteps}</div>"
    f"<div>&#x3bd;: {viscosity}</div>"
    f"<div>&#x3c9;: {omega:.4f}</div>"
    f"</div>"
  )
  verify_panels.append(stats)

  verify_html = (
    "<div style='font-size:11px;color:#94a3b8;font-weight:600;"
    "margin-bottom:4px;'>GPU vs CPU verification</div>"
    "<div style='display:flex;gap:8px;align-items:flex-start;"
    "flex-wrap:wrap;'>"
    + "".join(verify_panels) + "</div>"
  )

  return f"""
  <div style='margin:10px 0;padding:14px;border:1px solid #1f2937;
              border-radius:8px;background:#0f172a;'>
    <div style='font-weight:600;color:#e2e8f0;font-size:14px;
                margin-bottom:2px;'>
      Lattice Boltzmann D2Q9 Fluid Simulation</div>
    <div style='font-size:12px;color:#64748b;margin-bottom:12px;'>
      {w}&#xd7;{h} grid &#xb7; {timesteps} timesteps &#xb7;
      viscosity={viscosity} &#xb7; &#x3c9;={omega:.4f}</div>
    {evo_html}
    {verify_html}
    <div style='font-size:11px;color:#64748b;margin-top:10px;'>
      {desc}</div>
  </div>
  """


highLevelSummary = """
<p>Write a GPU compute shader in SPIR-V assembly that simulates 2D fluid flow using
the Lattice Boltzmann method. The shader iterates a grid of fluid cells, streaming
and colliding particle distributions each timestep, producing realistic fluid
behaviour from simple local rules.</p>
<p>The final grid state is compared against a CPU reference. Subpasses increase the
grid resolution and timestep count, testing both correctness and numerical
stability of the simulation.</p>
"""
