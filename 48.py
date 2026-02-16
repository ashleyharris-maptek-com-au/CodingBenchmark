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

import math
import random
import struct
import time
from typing import Tuple, Optional

import numpy as np

from shader_test_utils import assemble_spirv, validate_spirv
from compute_test_utils import ComputeShaderRunner, grade_compute_pingpong

title = "2D Fluid Simulation - Lattice Boltzmann (SPIR-V ASM Compute)"

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

**Algorithm per cell (x, y) = gl_GlobalInvocationID.xy:**
1. Guard: if x >= W or y >= H, return
2. Compute macroscopic density rho = sum(f[k] for k=0..8)
3. Compute macroscopic velocity: ux = sum(cx[k]*f[k])/rho, uy = sum(cy[k]*f[k])/rho
4. BGK collision: for each k:
     cu = cx[k]*ux + cy[k]*uy
     feq = w[k] * rho * (1 + 3*cu + 4.5*cu*cu - 1.5*(ux*ux + uy*uy))
     f_out = f[k] + omega * (feq - f[k])
5. Streaming: write f_out to neighbor cell (x+cx[k], y+cy[k]) with periodic wrap
   output[(((y+cy[k]) mod H) * W + ((x+cx[k]) mod W)) * 9 + k] = f_out

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


def gradeAnswer(result, subPass, aiEngineName):
  global _runner_cache, _ref_cache

  if not result or "spirv_code" not in result:
    return 0.0, "No SPIR-V code provided"

  code = result["spirv_code"]

  # Assemble SPIR-V
  try:
    spirv = assemble_spirv(code)
  except RuntimeError as e:
    return 0.0, f"SPIR-V assembly failed: {e}"

  valid, err = validate_spirv(spirv)
  if not valid:
    return 0.0, f"SPIR-V validation failed: {err}"

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

  # Run on GPU
  try:
    if _runner_cache is None:
      _runner_cache = ComputeShaderRunner()

    workgroups = ((w + 15) // 16, (h + 15) // 16, 1)

    def verify_fn(output_bytes):
      gpu_grid = buffer_to_grid(output_bytes, w, h)
      # Scale tolerance with timesteps
      val_tol = 0.05 * (1 + timesteps / 200)
      return compare_grids(gpu_grid, ref_grid, mass_tol=0.01, val_tol=val_tol)

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
