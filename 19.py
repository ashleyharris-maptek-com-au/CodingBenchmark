"""
Test 19: 3D Voxel Mining Optimisation (C# Implementation)

The LLM must write C# code that solves a voxel-based open-pit mining problem:
Given a 3D voxel grid containing ore bodies and waste rock, plan dig-and-haul
operations to maximise profit by delivering valuable ore to a processing plant.

Voxel encoding: each byte stores massInTonnes (lower 4 bits) and valuePerTonne
(upper 4 bits).  Z resolution is 10x finer than X/Y so a 10×10×100 grid is a cube.

Constraints:
- Can only dig voxels exposed from above or a side
- Dug material must be hauled along a 4-connected X/Y path
- Max slope: 1 Z-voxel per X/Y step
- Dumped dirt raises the ground, becoming a new obstacle
- Processing plant at (0, 0) buys valuable ore, penalises junk

Solver times out after 5 minutes.
"""

import random
import subprocess
import sys
import os
import time
import json
import tempfile
from typing import List, Tuple, Set, Dict, Optional
from pathlib import Path

from native_compiler import CSharpCompiler, CompilationError, ExecutionError, describe_this_pc
from solver_utils import parse_freeform_response

title = "3D Voxel Mining (C#)"

TIMEOUT_SECONDS = 300
RANDOM_SEED = 19191919

# Block sizes: Z is 10x finer than X/Y
X_BLOCK_SIZE = 1.0
Y_BLOCK_SIZE = 1.0
Z_BLOCK_SIZE = 0.1

# Economics
FUEL_FLAT = 1        # $ per ton per flat XY step
FUEL_UPHILL = 5      # $ per ton per uphill XY step
REVENUE_PER_MASS_VALUE = 1000  # $ per ton per value unit at plant
JUNK_PENALTY = 1000  # $ per ton for value=0 ore at plant


def _voxel_index(x: int, y: int, z: int, ydim: int, zdim: int) -> int:
  return x * ydim * zdim + y * zdim + z


def _encode_voxel(mass: int, value: int) -> int:
  """Encode mass (0-15) and value (0-15) into a single byte."""
  return (value & 0xF) * 16 + (mass & 0xF)


def _decode_voxel(byte: int) -> Tuple[int, int]:
  """Decode byte into (massInTonnes, valuePerTonne)."""
  return byte % 16, byte // 16


def generate_world(xdim: int, ydim: int, zdim: int, seed: int,
                   num_ore_bodies: int = 5) -> Tuple[bytearray, List[List[int]], list]:
  """
  Generate a voxel world with terrain, waste rock, and ore bodies.

  Returns (voxels, heightmap, ore_body_info).
  voxels is in x-major order: voxels[x * ydim * zdim + y * zdim + z].
  heightmap[x][y] = number of filled Z voxels in that column.
  """
  rng = random.Random(seed)
  voxels = bytearray(xdim * ydim * zdim)

  # Generate heightmap: base height + gentle noise
  base_height = zdim * 2 // 3
  heightmap = [[0] * ydim for _ in range(xdim)]
  for x in range(xdim):
    for y in range(ydim):
      variation = rng.randint(-zdim // 20, zdim // 20)
      heightmap[x][y] = max(1, min(zdim - 1, base_height + variation))

  # Fill below surface with waste rock (value=0, mass=3-8)
  for x in range(xdim):
    for y in range(ydim):
      h = heightmap[x][y]
      for z in range(h):
        mass = rng.randint(3, 15)
        voxels[_voxel_index(x, y, z, ydim, zdim)] = _encode_voxel(mass, 0)

  # Place ore bodies as ellipsoids
  ore_info = []
  for _ in range(num_ore_bodies):
    cx = rng.randint(max(2, xdim // 5), xdim - max(2, xdim // 5))
    cy = rng.randint(max(2, ydim // 5), ydim - max(2, ydim // 5))
    surf = heightmap[min(cx, xdim - 1)][min(cy, ydim - 1)]
    depth = rng.randint(max(1, zdim // 15), max(2, surf // 2))
    cz = max(1, surf - depth)
    rx = rng.randint(1, max(1, xdim // 8))
    ry = rng.randint(1, max(1, ydim // 8))
    rz = rng.randint(2, max(3, zdim // 12))
    ore_value = rng.randint(3, 15)
    ore_mass = rng.randint(8, 15)

    for x in range(max(0, cx - rx), min(xdim, cx + rx + 1)):
      for y in range(max(0, cy - ry), min(ydim, cy + ry + 1)):
        for z in range(max(0, cz - rz), min(zdim, cz + rz + 1)):
          dx = (x - cx) / max(1, rx)
          dy = (y - cy) / max(1, ry)
          dz = (z - cz) / max(1, rz)
          if dx * dx + dy * dy + dz * dz <= 1.0:
            if z < heightmap[x][y]:
              voxels[_voxel_index(x, y, z, ydim, zdim)] = _encode_voxel(ore_mass, ore_value)

    ore_info.append({"cx": cx, "cy": cy, "cz": cz, "rx": rx, "ry": ry, "rz": rz,
                     "value": ore_value, "mass": ore_mass})

  return voxels, heightmap, ore_info


def build_heightmap(voxels: bytearray, xdim: int, ydim: int, zdim: int) -> List[List[int]]:
  """Rebuild heightmap from voxel data (top of highest non-empty voxel + 1)."""
  hm = [[0] * ydim for _ in range(xdim)]
  for x in range(xdim):
    for y in range(ydim):
      for z in range(zdim - 1, -1, -1):
        if voxels[_voxel_index(x, y, z, ydim, zdim)] != 0:
          hm[x][y] = z + 1
          break
  return hm


def format_input_bytes(voxels: bytearray, xdim: int, ydim: int, zdim: int) -> bytes:
  """Build the binary input: text header line followed by raw voxel bytes."""
  header = f"{xdim} {ydim} {zdim} {X_BLOCK_SIZE} {Y_BLOCK_SIZE} {Z_BLOCK_SIZE}\n"
  return header.encode('ascii') + bytes(voxels)


def compute_total_ore_value(voxels: bytearray, xdim: int, ydim: int, zdim: int) -> float:
  """Sum of mass*value for every non-zero-value voxel (theoretical max revenue)."""
  total = 0.0
  for b in voxels:
    if b == 0:
      continue
    mass, value = _decode_voxel(b)
    if value > 0:
      total += mass * value * REVENUE_PER_MASS_VALUE
  return total


# ── Test configurations ──────────────────────────────────────────────────
# (xdim, ydim) are in XY voxels; zdim = physical_z * 10 for Z precision
TEST_CASES = [
  {"xdim": 10, "ydim": 10, "zdim": 100, "ore_bodies": 2,
   "desc": "10×10 tiny pit (10 KB)"},
  {"xdim": 15, "ydim": 15, "zdim": 150, "ore_bodies": 3,
   "desc": "15×15 small pit (34 KB)"},
  {"xdim": 20, "ydim": 20, "zdim": 200, "ore_bodies": 4,
   "desc": "20×20 pit (80 KB)"},
  {"xdim": 30, "ydim": 30, "zdim": 300, "ore_bodies": 5,
   "desc": "30×30 medium pit (270 KB)"},
  {"xdim": 50, "ydim": 50, "zdim": 500, "ore_bodies": 8,
   "desc": "50×50 large pit (1.25 MB)"},
  {"xdim": 75, "ydim": 75, "zdim": 750, "ore_bodies": 10,
   "desc": "75×75 pit (4.2 MB)"},
  {"xdim": 100, "ydim": 100, "zdim": 1000, "ore_bodies": 12,
   "desc": "100×100 pit (10 MB)"},
  {"xdim": 150, "ydim": 150, "zdim": 1500, "ore_bodies": 15,
   "desc": "150×150 large mine (34 MB)"},
  {"xdim": 250, "ydim": 250, "zdim": 1000, "ore_bodies": 20,
   "desc": "250×250 large mine (62 MB)"},
  {"xdim": 500, "ydim": 500, "zdim": 1000, "ore_bodies": 50,
   "desc": "500×500 large mine (250 MB)"},
  {"xdim": 1000, "ydim": 1000, "zdim": 1000, "ore_bodies": 100,
   "desc": "1000×1000 large mine (1 GB)"},
  {"xdim": 2000, "ydim": 2000, "zdim": 2000, "ore_bodies": 200,
   "desc": "2000×2000 large mine (8 GB)"},
  {"xdim": 5000, "ydim": 5000, "zdim": 2000, "ore_bodies": 200,
   "desc": "5000×5000 large mine (50 GB)"},
]

# Store last grading result for visualisation
lastGradeResult = None


def prepareSubpassPrompt(subPass: int) -> str:
  """Generate the prompt for subpass 0 that handles all excavation complexities."""
  if subPass != 0:
    raise StopIteration

  return f"""You are writing a C# program to solve a mining problem in a simplified 3D voxel space.

**Constraints:**
1. Can only dig dirt that is exposed from above or a side.
2. Dug dirt must be dumped somewhere - at a location not blocking future digging
3. Maximum uphill slope - 1 Z voxel per X/Y voxel.
4. Dumped dirt becomes a new obstacle (plan carefully!)
5. Goal is to get valuable ore to a processing plant located at x=0, y=0
6. Ground is stable and solid, and you can dig under any voxel without fear of it collapsing
   so long as it's connected to the earth. If you disconnect a voxel(s), it will collapse.

The voxel grid size has 10x more precision in Z than X/Y. So 10x10x100 is a cube.

**Input header format (stdin - space delimited. Ends with newline):**
```
XVoxelCount YVoxelCount ZVoxelCount XBlockSize YBlockSize ZBlockSize
```
Counts are human readable integers. BlockSize are human readable decimal numbers.

After the newline, voxels then follow in bytes, like this python psuedocode:
```
for x in range(XVoxelCount):
  for y in range(YVoxelCount):
    for z in range(ZVoxelCount):
      d = readByte(stdin)
      massInTonnes = d % 16
      valuePerTonne = d // 16
      addVoxel(d)
```

Each peice of dirt needs to be moved to its new location - you need to trace the
path in X and Y (4-connected / manhatten steps) and ensure that the path doesn't climb uphill 
or downhill more than 1 Z voxel for every X/Y voxel.

**Output format (stdout / newline seperated):**
```
M                    (number of operations)
N digX digY digZ stepX1 stepY1 stepX2 stepY2 (N steps) destX destY (for each operation)
```

If dumping valuable ore at the processing plant, the voxel ceases to exist. If a voxel is
dumped elsewhere, it is dumped on top of the ground, raising the ground by 1 voxel. It can
be dug again and moved later if required.

Every unit of mass requires $1 of fuel to move one voxel in X or Y when flat, or $5 of fuel
to go up a 10% slope. Downhill is free.

Getting valuable ore to your processing plant pays $1000 per unit of mass per unit of value.
The most valuable ore (#15) pays $15,000 per ton, so the most a single dump truck load can gross
is $225,000.

(Dumping junk ore at the processing plant causes damage to the plant, costing $1000 per ton.)

**Requirements:**
1. Program must compile with .NET/csc
2. Read from stdin, write to stdout
3. World size could be anything from 10x10x10 to 5000x5000x5000, your code should handle both extremes.
4. Your code should complete in under 5 minutes
5. Maximize profit.

**Environment:**
{describe_this_pc()}

**C# Compiler:**
{CSharpCompiler("test_engine").describe()}


Write complete, compilable C# code with a Main method.
Include adaptive logic that chooses different strategies based on problem scale.
"""


# List of subpasses to grade the single answer against all difficulty levels
extraGradeAnswerRuns = list(range(len(TEST_CASES)))

structure = None


def _extract_freeform(result):
  if isinstance(result, dict):
    discussion = result.get("reasoning") or result.get("discussion") or ""
    code = result.get("csharp_code") or result.get("code") or ""
    return discussion, code, ""
  if isinstance(result, str) and result.strip() == "__content_violation__":
    return "", "", "Content violation"
  parsed = parse_freeform_response(result or "")
  return parsed.get("discussion", ""), parsed.get("code", ""), ""


def execute_solver_binary(code: str, input_bytes: bytes, engine_name: str,
                          timeout: float = TIMEOUT_SECONDS) -> Tuple[str, str, float, bool]:
  """
  Compile and execute C# solver with binary stdin.
  Returns (stdout, error_message, execution_time, success).
  """
  compiler = CSharpCompiler(engine_name)
  if not compiler.find_compiler():
    return "", "No C# compiler found", 0, False

  try:
    exe_path = compiler.compile(code)

    # Write binary input to a temp file and use stdin_file for streaming
    with tempfile.NamedTemporaryFile(delete=False, suffix='.bin') as tf:
      tf.write(input_bytes)
      tmp_path = Path(tf.name)

    try:
      # Use the base class execute with stdin_file support
      # CSharpCompiler.execute doesn't have stdin_file, so call subprocess directly
      import platform
      is_windows = platform.system() == "Windows"

      if hasattr(compiler, '_compiler_type'):
        if compiler._compiler_type == 'dotnet' and str(exe_path).endswith('.dll'):
          cmd = ['dotnet', str(exe_path)]
        elif compiler._compiler_type == 'mono' and not is_windows:
          cmd = ['mono', str(exe_path)]
        else:
          cmd = [str(exe_path)]
      else:
        cmd = [str(exe_path)]

      start_time = time.time()
      with open(tmp_path, 'rb') as stdin_handle:
        process = subprocess.Popen(
          cmd, stdin=stdin_handle, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
          creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if is_windows else 0)
        stdout_bytes, stderr_bytes = process.communicate(timeout=timeout)

      exec_time = time.time() - start_time
      stdout = stdout_bytes.decode('utf-8', errors='replace')
      stderr = stderr_bytes.decode('utf-8', errors='replace')

      if process.returncode != 0:
        return stdout, f"Runtime error (exit {process.returncode}): {stderr[:500]}", exec_time, False
      return stdout, "", exec_time, True

    finally:
      try:
        tmp_path.unlink()
      except Exception:
        pass

  except CompilationError as e:
    return "", str(e), 0, False
  except subprocess.TimeoutExpired:
    return "", f"Execution timed out after {timeout}s", timeout, False
  except Exception as e:
    return "", f"Unexpected error: {str(e)}", 0, False


def parse_output(output: str) -> Tuple[list, str]:
  """
  Parse solver output.
  Each operation line: N digX digY digZ stepX1 stepY1 ... stepXN stepYN destX destY

  Returns (operations, error_message).
  Each operation is a dict: {dig: (x,y,z), path: [(x,y)...], dest: (x,y)}
  """
  lines = output.strip().split('\n')
  if not lines:
    return [], "Empty output"

  try:
    m = int(lines[0].strip())
    if m < 0:
      return [], f"Invalid operation count: {m}"

    operations = []
    for i in range(1, min(m + 1, len(lines))):
      parts = lines[i].strip().split()
      if len(parts) < 6:
        return operations, f"Op {i-1}: too few fields ({len(parts)})"

      n_steps = int(parts[0])
      dig_x, dig_y, dig_z = int(parts[1]), int(parts[2]), int(parts[3])

      # n_steps intermediate (x,y) pairs + 1 final (destX, destY) = 2*n_steps + 2 values
      expected = 4 + 2 * n_steps + 2
      if len(parts) < expected:
        return operations, f"Op {i-1}: expected {expected} fields, got {len(parts)}"

      path = []
      idx = 4
      for _ in range(n_steps):
        sx, sy = int(parts[idx]), int(parts[idx + 1])
        path.append((sx, sy))
        idx += 2

      dest_x, dest_y = int(parts[idx]), int(parts[idx + 1])

      operations.append({
        "dig": (dig_x, dig_y, dig_z),
        "path": path,
        "dest": (dest_x, dest_y),
      })

    if len(operations) != m:
      return operations, f"Expected {m} operations, got {len(operations)}"

    return operations, ""

  except (ValueError, IndexError) as e:
    return [], f"Parse error: {str(e)}"


def is_exposed(voxels: bytearray, x: int, y: int, z: int,
               xdim: int, ydim: int, zdim: int) -> bool:
  """Check if voxel at (x,y,z) is exposed from above or any side."""
  # Above
  if z + 1 >= zdim or voxels[_voxel_index(x, y, z + 1, ydim, zdim)] == 0:
    return True
  # 4 horizontal neighbours
  for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
    nx, ny = x + dx, y + dy
    if nx < 0 or nx >= xdim or ny < 0 or ny >= ydim:
      return True  # edge of world = exposed
    if voxels[_voxel_index(nx, ny, z, ydim, zdim)] == 0:
      return True
  return False


def validate_and_score(voxels_orig: bytearray, operations: list,
                       xdim: int, ydim: int, zdim: int) -> Tuple[bool, str, float, float, int]:
  """
  Validate all operations and compute profit.
  Returns (is_valid, message, profit, revenue, ops_count).
  """
  voxels = bytearray(voxels_orig)  # mutable copy
  heightmap = build_heightmap(voxels, xdim, ydim, zdim)

  total_revenue = 0.0
  total_fuel = 0.0
  total_penalty = 0.0

  for op_idx, op in enumerate(operations):
    dx, dy, dz = op["dig"]
    path = op["path"]
    dest_x, dest_y = op["dest"]

    # Bounds check dig position
    if not (0 <= dx < xdim and 0 <= dy < ydim and 0 <= dz < zdim):
      return False, f"Op {op_idx}: dig ({dx},{dy},{dz}) out of bounds", 0, 0, op_idx

    vi = _voxel_index(dx, dy, dz, ydim, zdim)
    voxel_byte = voxels[vi]
    if voxel_byte == 0:
      return False, f"Op {op_idx}: dig ({dx},{dy},{dz}) is empty", 0, 0, op_idx

    mass, value = _decode_voxel(voxel_byte)

    # Check exposed
    if not is_exposed(voxels, dx, dy, dz, xdim, ydim, zdim):
      return False, f"Op {op_idx}: voxel ({dx},{dy},{dz}) not exposed", 0, 0, op_idx

    # Remove the voxel
    voxels[vi] = 0
    # Update heightmap if this was the top voxel
    if dz == heightmap[dx][dy] - 1:
      h = dz - 1
      while h >= 0 and voxels[_voxel_index(dx, dy, h, ydim, zdim)] == 0:
        h -= 1
      heightmap[dx][dy] = h + 1

    # Validate path: starts at (dx, dy), goes through path steps, ends at (dest_x, dest_y)
    full_path = [(dx, dy)] + path + [(dest_x, dest_y)]
    step_fuel = 0.0

    for si in range(len(full_path) - 1):
      fx, fy = full_path[si]
      tx, ty = full_path[si + 1]

      # Same position = no movement (e.g. dig at plant, deliver to plant)
      if fx == tx and fy == ty:
        continue

      # Must be 4-connected
      manhattan = abs(tx - fx) + abs(ty - fy)
      if manhattan != 1:
        return False, f"Op {op_idx} step {si}: ({fx},{fy})->({tx},{ty}) not 4-connected (dist={manhattan})", 0, 0, op_idx

      # Bounds check
      if not (0 <= tx < xdim and 0 <= ty < ydim):
        return False, f"Op {op_idx} step {si}: ({tx},{ty}) out of bounds", 0, 0, op_idx

      # Height at from and to positions
      h_from = heightmap[fx][fy]
      h_to = heightmap[tx][ty]

      # Slope constraint: max 1 Z voxel difference per XY step
      if abs(h_to - h_from) > 1:
        return False, f"Op {op_idx} step {si}: slope {h_to - h_from} exceeds ±1 ({fx},{fy} h={h_from} -> {tx},{ty} h={h_to})", 0, 0, op_idx

      # Fuel cost
      if h_to > h_from:
        step_fuel += mass * FUEL_UPHILL
      elif h_to == h_from:
        step_fuel += mass * FUEL_FLAT
      # downhill is free

    total_fuel += step_fuel

    # Handle destination
    is_at_plant = (dest_x == 0 and dest_y == 0)
    if is_at_plant:
      if value > 0:
        total_revenue += mass * value * REVENUE_PER_MASS_VALUE
      else:
        total_penalty += mass * JUNK_PENALTY
    else:
      # Dump on top of ground at destination
      if not (0 <= dest_x < xdim and 0 <= dest_y < ydim):
        return False, f"Op {op_idx}: dest ({dest_x},{dest_y}) out of bounds", 0, 0, op_idx
      dump_z = heightmap[dest_x][dest_y]
      if dump_z >= zdim:
        return False, f"Op {op_idx}: dest column ({dest_x},{dest_y}) full (h={dump_z})", 0, 0, op_idx
      voxels[_voxel_index(dest_x, dest_y, dump_z, ydim, zdim)] = voxel_byte
      heightmap[dest_x][dest_y] = dump_z + 1

  profit = total_revenue - total_fuel - total_penalty
  return True, "Valid", profit, total_revenue, len(operations)


def gradeAnswer(result: dict, subPass: int, aiEngineName: str) -> tuple:
  """
  Grade the C# voxel mining solver.
  Score is profit / theoretical_max_revenue, clamped to [0, 1].
  """
  global lastGradeResult

  if not result:
    return 0.0, "No result provided"

  discussion, code, parse_error = _extract_freeform(result)
  if parse_error:
    return 0.0, parse_error
  if not code:
    return 0.0, "No C# code provided"

  case = TEST_CASES[subPass]
  xdim, ydim, zdim = case["xdim"], case["ydim"], case["zdim"]
  desc = case["desc"]
  seed = RANDOM_SEED + subPass

  # Generate world
  t0 = time.time()
  voxels, heightmap, ore_info = generate_world(xdim, ydim, zdim, seed, case["ore_bodies"])
  gen_time = time.time() - t0

  input_bytes = format_input_bytes(voxels, xdim, ydim, zdim)
  theoretical_max = compute_total_ore_value(voxels, xdim, ydim, zdim)

  code = code

  # Execute solver
  stdout, error, exec_time, success = execute_solver_binary(code, input_bytes, aiEngineName)

  if not success:
    return 0.0, f"[{desc}] {error}"

  # Parse output
  operations, parse_error = parse_output(stdout)
  if parse_error and not operations:
    return 0.0, f"[{desc}] {parse_error}"

  # Validate and score
  is_valid, msg, profit, revenue, ops_count = validate_and_score(
    voxels, operations, xdim, ydim, zdim)

  if not is_valid:
    return 0.0, f"[{desc}] Invalid: {msg}"

  # Score: profit as fraction of theoretical max revenue
  if theoretical_max <= 0:
    score = 1.0 if profit >= 0 else 0.0
  else:
    score = max(0.0, min(1.0, profit / theoretical_max))

  lastGradeResult = {
    "subPass": subPass,
    "voxels": voxels,
    "heightmap": heightmap,
    "ore_info": ore_info,
    "operations": operations,
    "xdim": xdim, "ydim": ydim, "zdim": zdim,
    "profit": profit,
    "revenue": revenue,
    "theoretical_max": theoretical_max,
    "score": score,
  }

  explanation = (f"[{desc}] Ops: {ops_count}, Profit: ${profit:,.0f}, "
                 f"Revenue: ${revenue:,.0f} / ${theoretical_max:,.0f} max, "
                 f"Gen: {gen_time:.1f}s, Run: {exec_time:.1f}s — Score: {score:.3f}")

  return score, explanation


def _generate_mining_visualization(grade_result: dict) -> str:
  """Generate a three.js visualization of the mining world and operations."""
  xdim = grade_result["xdim"]
  ydim = grade_result["ydim"]
  zdim = grade_result["zdim"]
  voxels = grade_result["voxels"]
  heightmap = grade_result["heightmap"]
  ore_info = grade_result["ore_info"]
  operations = grade_result["operations"]

  # For large worlds, subsample columns
  max_render_dim = 60
  step_x = max(1, xdim // max_render_dim)
  step_y = max(1, ydim // max_render_dim)

  # Collect terrain columns: [x, y, height] - rendered as solid pillars
  columns = []       # [x, y, h] for terrain columns
  ore_voxels = []    # [x, y, z, value] for visible ore
  dug_positions = []

  for op in operations:
    dug_positions.append(list(op["dig"]))

  plant_h = heightmap[0][0] if xdim > 0 and ydim > 0 else 0

  max_ore_render = 3000
  for x in range(0, xdim, step_x):
    for y in range(0, ydim, step_y):
      h = heightmap[x][y]
      if h <= 0:
        continue
      columns.append([x, y, h])

      # Collect ore voxels in this column
      if len(ore_voxels) < max_ore_render:
        for z2 in range(h - 1, -1, -1):
          vi2 = _voxel_index(x, y, z2, ydim, zdim)
          b2 = voxels[vi2]
          if b2 != 0:
            _m2, v2 = _decode_voxel(b2)
            if v2 > 0:
              ore_voxels.append([x, y, z2, v2])
              if len(ore_voxels) >= max_ore_render:
                break

  ore_centres = [[o["cx"], o["cy"], o["cz"], o["rx"], o["ry"], o["rz"], o["value"]] for o in ore_info]

  # Serialize to JSON
  col_json = json.dumps(columns)
  ore_vox_json = json.dumps(ore_voxels)
  dug_json = json.dumps(dug_positions[:500])
  ore_json = json.dumps(ore_centres)

  viz_id = f"mv{id(grade_result)}"
  # Physical scale: X/Y blocks are 1m, Z blocks are 0.1m
  # For display, normalise so the world looks roughly cubic
  xs = X_BLOCK_SIZE * step_x
  ys = Y_BLOCK_SIZE * step_y
  zs = Z_BLOCK_SIZE

  html = f"""
<div id="{viz_id}" style="width:100%;height:500px;position:relative;background:#1a1a2e;border-radius:8px;overflow:hidden;">
<canvas id="c{viz_id}" style="width:100%;height:100%;"></canvas>
</div>
<script>
(function() {{
  const container = document.getElementById('{viz_id}');
  const canvas = document.getElementById('c{viz_id}');
  if (!container || !canvas) return;

  function loadThree(cb) {{
    if (window.THREE) {{ cb(); return; }}
    const s = document.createElement('script');
    s.src = 'https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js';
    s.onload = function() {{
      const s2 = document.createElement('script');
      s2.src = 'https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js';
      s2.onload = cb;
      document.head.appendChild(s2);
    }};
    document.head.appendChild(s);
  }}

  loadThree(function() {{
    const THREE = window.THREE;
    const W = container.clientWidth, H = container.clientHeight;
    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x1a1a2e);
    scene.fog = new THREE.Fog(0x1a1a2e, 50, 200);

    const camera = new THREE.PerspectiveCamera(50, W/H, 0.1, 5000);
    const renderer = new THREE.WebGLRenderer({{canvas: canvas, antialias: true}});
    renderer.setSize(W, H);
    renderer.setPixelRatio(window.devicePixelRatio);

    const controls = new THREE.OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;

    scene.add(new THREE.AmbientLight(0x606060, 0.7));
    const dLight = new THREE.DirectionalLight(0xffffff, 0.9);
    dLight.position.set(30, 50, 20);
    scene.add(dLight);
    const dLight2 = new THREE.DirectionalLight(0xaabbff, 0.3);
    dLight2.position.set(-20, 30, -10);
    scene.add(dLight2);

    const xs = {xs};  // x step size
    const ys = {ys};  // y step size
    const zs = {zs};  // z voxel size
    const stepX = {step_x};
    const stepY = {step_y};

    // Terrain columns - each is a solid pillar from z=0 to z=height
    const colData = {col_json};
    colData.forEach(function(c) {{
      const x = c[0], y = c[1], h = c[2];
      const colH = h * zs;
      if (colH <= 0) return;
      const geom = new THREE.BoxGeometry(xs * 0.98, 0.1, ys * 0.98);
      const mat = new THREE.MeshLambertMaterial({{color: 0x8B7355}});
      const mesh = new THREE.Mesh(geom, mat);
      // Position at centre of column
      mesh.position.set(x * {X_BLOCK_SIZE}, colH -0.05, y * {Y_BLOCK_SIZE});
      scene.add(mesh);
    }});

    // Ore voxels embedded in terrain (brighter, slightly protruding)
    const oreVox = {ore_vox_json};
    oreVox.forEach(function(v) {{
      const x = v[0], y = v[1], z = v[2], val = v[3];
      const geom = new THREE.BoxGeometry(xs * 1.01, zs * 1.01, ys * 1.01);
      let color;
      if (val > 10) color = 0xffd700;
      else if (val > 5) color = 0xff8c00;
      else color = 0xcd853f;
      const mat = new THREE.MeshLambertMaterial({{color: color, emissive: color, emissiveIntensity: 0.3}});
      const mesh = new THREE.Mesh(geom, mat);
      mesh.position.set(x * {X_BLOCK_SIZE}, z * zs + zs/2, y * {Y_BLOCK_SIZE});
      scene.add(mesh);
    }});

    // Dug positions (red translucent)
    const dugData = {dug_json};
    dugData.forEach(function(d) {{
      const geom = new THREE.BoxGeometry(xs * 0.8, zs * 0.8, ys * 0.8);
      const mat = new THREE.MeshBasicMaterial({{color: 0xff3333, transparent: true, opacity: 0.5}});
      const mesh = new THREE.Mesh(geom, mat);
      mesh.position.set(d[0] * {X_BLOCK_SIZE}, d[2] * zs + zs/2, d[1] * {Y_BLOCK_SIZE});
      scene.add(mesh);
    }});

    // Processing plant at (0,0) - ON TOP of terrain
    const plantH = {plant_h} * zs;
    // Building body
    const bldgGeom = new THREE.BoxGeometry(2, 3, 2);
    const bldgMat = new THREE.MeshLambertMaterial({{color: 0xcc4444}});
    const bldg = new THREE.Mesh(bldgGeom, bldgMat);
    bldg.position.set(0, plantH + 1.5, 0);
    scene.add(bldg);
    // Chimney
    const chimGeom = new THREE.CylinderGeometry(0.3, 0.3, 3, 8);
    const chimMat = new THREE.MeshLambertMaterial({{color: 0x666666}});
    const chim = new THREE.Mesh(chimGeom, chimMat);
    chim.position.set(0.5, plantH + 4.5, 0.5);
    scene.add(chim);
    // Label
    const lCanvas = document.createElement('canvas');
    lCanvas.width = 256; lCanvas.height = 64;
    const ctx = lCanvas.getContext('2d');
    ctx.fillStyle = '#cc4444';
    ctx.fillRect(0, 0, 256, 64);
    ctx.fillStyle = '#ffffff';
    ctx.font = 'bold 28px Arial';
    ctx.textAlign = 'center';
    ctx.fillText('PROCESSING PLANT', 128, 42);
    const labelTex = new THREE.CanvasTexture(lCanvas);
    const labelGeom = new THREE.PlaneGeometry(4, 1);
    const labelMat = new THREE.MeshBasicMaterial({{map: labelTex, transparent: true, side: THREE.DoubleSide}});
    const label = new THREE.Mesh(labelGeom, labelMat);
    label.position.set(0, plantH + 6.5, 0);
    scene.add(label);

    // Ore body outlines (wireframe ellipsoids)
    const oreData = {ore_json};
    oreData.forEach(function(o) {{
      const cx=o[0], cy=o[1], cz=o[2], rx=o[3], ry=o[4], rz=o[5], val=o[6];
      const sGeom = new THREE.SphereGeometry(1, 12, 12);
      sGeom.scale(rx * {X_BLOCK_SIZE}, rz * zs, ry * {Y_BLOCK_SIZE});
      const brightness = Math.min(1, val / 15);
      const sMat = new THREE.MeshBasicMaterial({{
        color: new THREE.Color(brightness, brightness * 0.6, 0),
        wireframe: true, transparent: true, opacity: 0.5
      }});
      const sMesh = new THREE.Mesh(sGeom, sMat);
      sMesh.position.set(cx * {X_BLOCK_SIZE}, cz * zs, cy * {Y_BLOCK_SIZE});
      scene.add(sMesh);
    }});

    // Ground plane
    const gpGeom = new THREE.PlaneGeometry({xdim * X_BLOCK_SIZE} * 1.5, {ydim * Y_BLOCK_SIZE} * 1.5);
    const gpMat = new THREE.MeshLambertMaterial({{color: 0x2a2a1e, side: THREE.DoubleSide}});
    const gp = new THREE.Mesh(gpGeom, gpMat);
    gp.rotation.x = -Math.PI / 2;
    gp.position.set({xdim * X_BLOCK_SIZE / 2}, -0.01, {ydim * Y_BLOCK_SIZE / 2});
    scene.add(gp);

    // Camera
    const midX = {xdim * X_BLOCK_SIZE / 2};
    const midZ = {ydim * Y_BLOCK_SIZE / 2};
    const surfH = {zdim * Z_BLOCK_SIZE * 2 / 3};
    const viewDist = Math.max({xdim * X_BLOCK_SIZE}, {ydim * Y_BLOCK_SIZE}) * 1.8;
    camera.position.set(midX + viewDist * 0.6, surfH + viewDist * 0.5, midZ + viewDist * 0.6);
    controls.target.set(midX, surfH * 0.5, midZ);
    controls.update();

    function animate() {{
      requestAnimationFrame(animate);
      controls.update();
      renderer.render(scene, camera);
    }}
    animate();

    new ResizeObserver(function() {{
      const w = container.clientWidth, h = container.clientHeight;
      camera.aspect = w / h;
      camera.updateProjectionMatrix();
      renderer.setSize(w, h);
    }}).observe(container);
  }});
}})();
</script>
"""
  return html


def resultToNiceReport(result: dict, subPass: int, aiEngineName: str) -> str:
  global lastGradeResult

  if not result:
    return "<p style='color:red'>No result provided</p>"

  case = TEST_CASES[subPass]
  html = f"<h4>3D Voxel Mining - {case['desc']}</h4>"

  discussion, code, _ = _extract_freeform(result)
  if discussion and subPass == 0:
    r = discussion[:400] + ('...' if len(discussion) > 400 else '')
    html += f"<p><strong>Strategy:</strong> {r.replace('<', '&lt;').replace('>', '&gt;')}</p>"

  # Add visualization if we have grade results
  if lastGradeResult and lastGradeResult.get("subPass") == subPass:
    gr = lastGradeResult
    html += f"""<p><strong>Results:</strong> Profit: ${gr['profit']:,.0f},
    Revenue: ${gr['revenue']:,.0f} / ${gr['theoretical_max']:,.0f} theoretical max,
    Operations: {len(gr['operations'])}, Score: {gr['score']:.3f}</p>"""
    try:
      html += _generate_mining_visualization(gr)
    except Exception as e:
      html += f"<p style='color:orange'>Visualization error: {str(e)}</p>"

  if code and subPass == 0:
    code_escaped = code.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    html += f"<details><summary>View C# Code ({len(code)} chars)</summary><pre>{code_escaped}</pre></details>"

  return html


highLevelSummary = """
<p>Plan an open-pit mine in a 3D block model. Each block has a value (ore worth
money, waste costs money to move) and you can only dig a block once the blocks
above it have been removed. The goal is to maximise profit by choosing which
blocks to extract and where to dump the waste.</p>
<p>The AI must balance revenue against haulage costs, respect slope constraints for
road access, and avoid dumping waste where it blocks future digging. Subpasses
increase the pit size and complexity.</p>
"""
