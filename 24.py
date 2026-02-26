"""
Test 24: 3D Lunar Lander Game (C++ Implementation)

The LLM must write C++ code that plays Lunar Lander in 3D space in real-time.
"""

import random
import subprocess
import sys
import os
import time
import math
import threading
import json
import struct
import tempfile
import hashlib
import numpy as np
from queue import Queue, Empty
from typing import List, Tuple, Set, Dict, Optional
from pathlib import Path

# Import our native compiler helper
from native_compiler import CppCompiler, CompilationError, ExecutionError, describe_this_pc
from solver_utils import GradeCache

title = "3D Lunar Lander Game (C++)"

tags = [
  "cpp",
  "structured response",
  "game ai",
  "simulation",
]

# Timeout in seconds
TIMEOUT_SECONDS = 3600

# Seed for reproducibility
RANDOM_SEED = 24242424

MAX_VIS_VOLUME = 2_000_000
LAST_LANDER_STATS: Dict[Tuple[int, str], Dict] = {}

_grade_cache = GradeCache("test24")


def _cache_key_parts(result: dict, subPass: int) -> tuple:
  case = TEST_CASES[subPass]
  code = result.get("cpp_code", "")
  return (
    hashlib.sha256(code.encode("utf-8")).hexdigest()[:16],
    (f"w={case['width']}|d={case['depth']}|h={case['height']}|"
     f"g={case['gravity']}|th={case['max_thrust']}|fuel={case['fuel']}|"
     f"t={case['max_time']}|seed={RANDOM_SEED + subPass}"),
  )


def _perlin_fade(t: float) -> float:
  return t * t * t * (t * (t * 6 - 15) + 10)


def _perlin_lerp(a: float, b: float, t: float) -> float:
  return a + _perlin_fade(t) * (b - a)


def _perlin_grad3(hash_val: int, x: float, y: float, z: float) -> float:
  h = hash_val & 15
  u = x if h < 8 else y
  v = y if h < 4 else (x if h == 12 or h == 14 else z)
  return (u if (h & 1) == 0 else -u) + (v if (h & 2) == 0 else -v)


class PerlinNoise3D:
  """Simple 3D Perlin noise with seeded permutation table."""

  def __init__(self, seed: int):
    rng = random.Random(seed)
    self.perm = list(range(256))
    rng.shuffle(self.perm)
    self.perm = self.perm + self.perm

  def noise(self, x: float, y: float, z: float) -> float:
    xi = int(math.floor(x)) & 255
    yi = int(math.floor(y)) & 255
    zi = int(math.floor(z)) & 255
    xf = x - math.floor(x)
    yf = y - math.floor(y)
    zf = z - math.floor(z)
    p = self.perm
    aaa = p[p[p[xi] + yi] + zi]
    aba = p[p[p[xi] + yi + 1] + zi]
    aab = p[p[p[xi] + yi] + zi + 1]
    abb = p[p[p[xi] + yi + 1] + zi + 1]
    baa = p[p[p[xi + 1] + yi] + zi]
    bba = p[p[p[xi + 1] + yi + 1] + zi]
    bab = p[p[p[xi + 1] + yi] + zi + 1]
    bbb = p[p[p[xi + 1] + yi + 1] + zi + 1]
    x1 = _perlin_lerp(_perlin_grad3(aaa, xf, yf, zf), _perlin_grad3(baa, xf - 1, yf, zf), xf)
    x2 = _perlin_lerp(_perlin_grad3(aba, xf, yf - 1, zf), _perlin_grad3(bba, xf - 1, yf - 1, zf), xf)
    y1 = _perlin_lerp(x1, x2, yf)
    x1 = _perlin_lerp(_perlin_grad3(aab, xf, yf, zf - 1), _perlin_grad3(bab, xf - 1, yf, zf - 1), xf)
    x2 = _perlin_lerp(_perlin_grad3(abb, xf, yf - 1, zf - 1), _perlin_grad3(bbb, xf - 1, yf - 1, zf - 1), xf)
    y2 = _perlin_lerp(x1, x2, yf)
    return _perlin_lerp(y1, y2, zf)

  def octave(self, x: float, y: float, z: float, octaves: int = 4, persistence: float = 0.5) -> float:
    total = 0.0
    amplitude = 1.0
    frequency = 1.0
    max_val = 0.0
    for _ in range(octaves):
      total += self.noise(x * frequency, y * frequency, z * frequency) * amplitude
      max_val += amplitude
      amplitude *= persistence
      frequency *= 2.0
    return total / max_val if max_val > 0 else 0.0


TERRAIN_CACHE_DIR = os.path.join(tempfile.gettempdir(), "lander3d_terrain")


class Lander3DWorld:
  """Generates and stores a 3D Lunar Lander world with binary terrain file.

  Terrain is stored as a binary file with 3x uint16 per (x, y) cell:
    for x in range(width):
      for y in range(depth):
        a = uint16  # ground rock top
        b = uint16  # floating rock bottom
        c = uint16  # floating rock top
    z < a: rock (ground)
    a < z < b: open air
    b < z < c: rock (floating feature)
    c < z: open air

  Generated using 3D Perlin noise for interesting terrain including
  caves, overhangs, and floating rock formations.
  """

  def __init__(self, width: int, depth: int, height: int, gravity: float, max_thrust: float,
               fuel: float, seed: int):
    self.width = width
    self.depth = depth
    self.height = height
    self.gravity = gravity
    self.max_thrust = max_thrust
    self.initial_fuel = fuel
    self.seed = seed
    self.rng = random.Random(seed)

    self.terrain_path: Optional[str] = None
    self._terrain_data: Optional[bytes] = None

    self._generate_terrain()

    # Place start/target inside safe margins with a meaningful 2D traverse
    margin = max(10, int(min(width, depth) * 0.15))
    max_attempts = 50
    min_sep = min(width, depth) * 0.35
    sx = sy = tx = ty = 0
    for _ in range(max_attempts):
      sx = self.rng.randint(margin, width - margin - 1)
      sy = self.rng.randint(margin, depth - margin - 1)
      # Pick target by offset so it's not colinear with axes
      angle = self.rng.uniform(0.3, 2.8)
      dist = self.rng.uniform(min_sep, min(width, depth) * 0.75)
      tx = int(sx + math.cos(angle) * dist)
      ty = int(sy + math.sin(angle) * dist)
      if (margin <= tx < width - margin) and (margin <= ty < depth - margin):
        if math.hypot(tx - sx, ty - sy) >= min_sep:
          break
    self.start_x = sx
    self.start_y = sy
    self.start_z = max(10.0, height * 0.85)

    a, _, _ = self._get_cell(tx, ty)
    self.target_x = tx
    self.target_y = ty
    self.target_z = float(a) + 1.0

    # Clear landing zone
    self._clear_landing_zone()

    # Initial horizontal velocity: non-zero, not aimed at target or edges
    dx = self.target_x - self.start_x
    dy = self.target_y - self.start_y
    dist_xy = math.hypot(dx, dy)
    if dist_xy < 1:
      dist_xy = 1.0
    # Perpendicular direction with a bias away from edges
    dir_x = -dy / dist_xy
    dir_y = dx / dist_xy
    center_x = (width - 1) / 2.0
    center_y = (depth - 1) / 2.0
    to_center_x = center_x - self.start_x
    to_center_y = center_y - self.start_y
    if dir_x * to_center_x + dir_y * to_center_y < 0:
      dir_x *= -1
      dir_y *= -1

    speed = max(8.0, min(0.12 * max(width, depth), 80.0))
    self.start_vx = dir_x * speed
    self.start_vy = dir_y * speed
    self.start_vz = 0.0

  def _terrain_cache_path(self) -> str:
    """Get deterministic cache path based on dimensions and seed."""
    os.makedirs(TERRAIN_CACHE_DIR, exist_ok=True)
    key = f"{self.width}_{self.depth}_{self.height}_{self.seed}"
    h = hashlib.md5(key.encode()).hexdigest()[:12]
    return os.path.join(TERRAIN_CACHE_DIR, f"terrain_{key}_{h}.bin")

  def _generate_terrain(self):
    """Generate 3D terrain using Perlin noise on a coarse grid, then interpolate.

    Uses bilinear interpolation from a ~40x40 control grid to fill the full
    resolution binary file. This keeps generation fast even for huge maps
    (seconds instead of hours for billion-cell worlds).
    """
    cache_path = self._terrain_cache_path()
    expected_size = self.width * self.depth * 6

    if os.path.exists(cache_path) and os.path.getsize(cache_path) == expected_size:
      self.terrain_path = cache_path
      # For huge files, don't load into memory - use file-backed access
      if expected_size > 500_000_000:
        self._terrain_data = None
        self._terrain_file = cache_path
      else:
        with open(cache_path, 'rb') as f:
          self._terrain_data = f.read()
      return

    w, d, h = self.width, self.depth, self.height
    perlin = PerlinNoise3D(self.seed)

    # Coarse control grid - ~40 points per axis, minimum 2
    ctrl_nx = max(2, min(60, w // max(1, w // 40) + 2))
    ctrl_ny = max(2, min(60, d // max(1, d // 40) + 2))
    step_x = w / (ctrl_nx - 1)
    step_y = d / (ctrl_ny - 1)

    # Scale factors for Perlin noise sampling
    noise_scale = 8.0

    print(f"  Generating 3D terrain {w}x{d}x{h} ({expected_size / 1e9:.2f} GB), "
          f"ctrl grid {ctrl_nx}x{ctrl_ny}...")
    t0 = time.time()

    # Sample Perlin noise on coarse grid for ground height and floating rock
    ground_base = h * 0.18
    ground_var = h * 0.08
    mid_z = h * 0.45

    # Control grids: ground_a[ci][cj], rock_b[ci][cj], rock_c[ci][cj]
    ctrl_a = [[0] * ctrl_ny for _ in range(ctrl_nx)]
    ctrl_b = [[0] * ctrl_ny for _ in range(ctrl_nx)]
    ctrl_c = [[0] * ctrl_ny for _ in range(ctrl_nx)]

    for ci in range(ctrl_nx):
      nx = ci / (ctrl_nx - 1) * noise_scale
      for cj in range(ctrl_ny):
        ny = cj / (ctrl_ny - 1) * noise_scale

        # Ground height from 2D noise (z=0 slice)
        gn = perlin.octave(nx, ny, 0.0, octaves=4, persistence=0.5)
        a_val = ground_base + gn * ground_var
        a_val = max(1, min(h - 4, int(a_val)))

        # Floating rock from 3D noise at mid-altitude
        rn = perlin.octave(nx * 1.5, ny * 1.5, mid_z / max(1, h // 5),
                           octaves=3, persistence=0.45)

        if rn > 0.15:
          thickness = int((rn - 0.15) * h * 0.3)
          thickness = max(1, min(thickness, h // 4))
          rb = int(mid_z - thickness / 2)
          rt = int(mid_z + thickness / 2)
          b_val = max(a_val + 2, min(h - 2, rb))
          c_val = max(b_val + 1, min(h - 1, rt))
        else:
          b_val = a_val + 1
          c_val = b_val

        ctrl_a[ci][cj] = a_val
        ctrl_b[ci][cj] = b_val
        ctrl_c[ci][cj] = c_val

    elapsed_ctrl = time.time() - t0
    print(f"  Control grid sampled in {elapsed_ctrl:.2f}s, interpolating to full resolution...")

    # Convert control grids to numpy arrays for vectorized interpolation
    np_a = np.array(ctrl_a, dtype=np.float64)
    np_b = np.array(ctrl_b, dtype=np.float64)
    np_c = np.array(ctrl_c, dtype=np.float64)

    # Precompute y interpolation indices and weights
    y_coords = np.arange(d, dtype=np.float64) / step_y
    y_ci0 = np.minimum(y_coords.astype(np.int32), ctrl_ny - 2)
    y_ci1 = y_ci0 + 1
    y_ty = y_coords - y_ci0.astype(np.float64)

    # Write in x-row chunks
    with open(cache_path, 'wb') as f:
      for x in range(w):
        ci_f = x / step_x
        ci0 = min(int(ci_f), ctrl_nx - 2)
        ci1 = ci0 + 1
        tx = ci_f - ci0

        # Vectorized bilinear interpolation for entire row
        # For each of a, b, c: lerp in y for ci0 and ci1, then lerp in x
        a00 = np_a[ci0][y_ci0]
        a01 = np_a[ci0][y_ci1]
        a10 = np_a[ci1][y_ci0]
        a11 = np_a[ci1][y_ci1]
        a_y0 = a00 + (a01 - a00) * y_ty
        a_y1 = a10 + (a11 - a10) * y_ty
        a_row = a_y0 + (a_y1 - a_y0) * tx

        b00 = np_b[ci0][y_ci0]
        b01 = np_b[ci0][y_ci1]
        b10 = np_b[ci1][y_ci0]
        b11 = np_b[ci1][y_ci1]
        b_y0 = b00 + (b01 - b00) * y_ty
        b_y1 = b10 + (b11 - b10) * y_ty
        b_row = b_y0 + (b_y1 - b_y0) * tx

        c00 = np_c[ci0][y_ci0]
        c01 = np_c[ci0][y_ci1]
        c10 = np_c[ci1][y_ci0]
        c11 = np_c[ci1][y_ci1]
        c_y0 = c00 + (c01 - c00) * y_ty
        c_y1 = c10 + (c11 - c10) * y_ty
        c_row = c_y0 + (c_y1 - c_y0) * tx

        # Enforce ordering: a < b <= c, clamp to valid range
        a_int = np.clip(a_row.astype(np.int32), 1, h - 4)
        b_int = np.clip(b_row.astype(np.int32), a_int + 1, h - 2)
        # If b was equal to a+1 in ctrl (no floating rock), keep b=c
        c_int = np.clip(c_row.astype(np.int32), b_int, h - 1)

        # Interleave a,b,c into uint16 array: [a0,b0,c0, a1,b1,c1, ...]
        row_data = np.empty(d * 3, dtype=np.uint16)
        row_data[0::3] = a_int.astype(np.uint16)
        row_data[1::3] = b_int.astype(np.uint16)
        row_data[2::3] = c_int.astype(np.uint16)

        f.write(row_data.tobytes())

        if w >= 1000 and x % (w // 10) == 0:
          pct = x / w * 100
          elapsed = time.time() - t0
          print(f"    {pct:.0f}% ({elapsed:.1f}s)")

    elapsed = time.time() - t0
    print(f"  Terrain generated in {elapsed:.1f}s")

    self.terrain_path = cache_path
    if expected_size > 500_000_000:
      self._terrain_data = None
      self._terrain_file = cache_path
    else:
      with open(cache_path, 'rb') as f:
        self._terrain_data = f.read()

  def _get_cell(self, x: int, y: int) -> Tuple[int, int, int]:
    """Get (a, b, c) terrain values for cell (x, y)."""
    x = max(0, min(x, self.width - 1))
    y = max(0, min(y, self.depth - 1))
    offset = (x * self.depth + y) * 6
    if self._terrain_data is not None and offset + 6 <= len(self._terrain_data):
      a, b, c = struct.unpack_from('<HHH', self._terrain_data, offset)
      return a, b, c
    # File-backed access for huge terrains
    if self.terrain_path and os.path.exists(self.terrain_path):
      with open(self.terrain_path, 'rb') as f:
        f.seek(offset)
        data = f.read(6)
        if len(data) == 6:
          a, b, c = struct.unpack('<HHH', data)
          return a, b, c
    return 1, 2, 2

  def _clear_landing_zone(self):
    """Flatten terrain and remove floating rocks near landing pad."""
    pad_r = max(5, min(self.width, self.depth) * 0.02)
    pad_r_int = int(pad_r) + 1
    tx, ty = self.target_x, self.target_y

    # Find the ground height at the exact target
    a0, _, _ = self._get_cell(tx, ty)

    if self._terrain_data is not None:
      # In-memory modification
      data = bytearray(self._terrain_data)
      for dx in range(-pad_r_int, pad_r_int + 1):
        for dy in range(-pad_r_int, pad_r_int + 1):
          ix = tx + dx
          iy = ty + dy
          if ix < 0 or ix >= self.width or iy < 0 or iy >= self.depth:
            continue
          if math.sqrt(dx * dx + dy * dy) > pad_r:
            continue
          offset = (ix * self.depth + iy) * 6
          a = a0
          b = a + 1
          c = b
          struct.pack_into('<HHH', data, offset, a, b, c)
      self._terrain_data = bytes(data)
      if self.terrain_path:
        with open(self.terrain_path, 'wb') as f:
          f.write(self._terrain_data)
    elif self.terrain_path and os.path.exists(self.terrain_path):
      # File-backed modification for huge terrains
      cell_data = struct.pack('<HHH', a0, a0 + 1, a0 + 1)
      with open(self.terrain_path, 'r+b') as f:
        for dx in range(-pad_r_int, pad_r_int + 1):
          for dy in range(-pad_r_int, pad_r_int + 1):
            ix = tx + dx
            iy = ty + dy
            if ix < 0 or ix >= self.width or iy < 0 or iy >= self.depth:
              continue
            if math.sqrt(dx * dx + dy * dy) > pad_r:
              continue
            offset = (ix * self.depth + iy) * 6
            f.seek(offset)
            f.write(cell_data)

    # Update target_z to match flattened ground
    self.target_z = float(a0) + 1.0

  def is_solid(self, x: float, y: float, z: float) -> bool:
    """Check if position (x,y,z) is inside solid terrain."""
    if x < 0 or x >= self.width or y < 0 or y >= self.depth:
      return True
    if z >= self.height or z < 0:
      return True
    ix = int(x)
    iy = int(y)
    a, b, c = self._get_cell(ix, iy)
    zi = int(z)
    # z < a: rock, a <= z < b: open, b <= z < c: rock, c <= z: open
    if zi < a:
      return True
    if zi >= b and zi < c:
      return True
    return False

  def is_landing_zone(self, x: float, y: float, z: float) -> bool:
    """Check if position is near target landing zone."""
    pad_r = max(5, min(self.width, self.depth) * 0.02)
    dist_xy = math.sqrt((x - self.target_x)**2 + (y - self.target_y)**2)
    return dist_xy < pad_r and z <= self.target_z + 3

  def terrain_file_size_gb(self) -> float:
    """Return terrain file size in GB."""
    return (self.width * self.depth * 6) / (1024**3)

  def startup_delay(self) -> float:
    """Calculate startup delay: 5s + 10s per GB of terrain file."""
    return 5.0 + 10.0 * self.terrain_file_size_gb()


class Lander3DSimulator:
  """Simulates 3D lander physics with proper thrust vectoring."""

  def __init__(self, world: Lander3DWorld):
    self.world = world
    self.x = float(world.start_x)
    self.y = float(world.start_y)
    self.z = float(world.start_z)
    self.vx = float(getattr(world, "start_vx", 0.0))
    self.vy = float(getattr(world, "start_vy", 0.0))
    self.vz = float(getattr(world, "start_vz", 0.0))

    self.pitch = 0.0
    self.yaw = 0.0
    self.roll = 0.0

    self.pitch_rate = 0.0
    self.yaw_rate = 0.0
    self.roll_rate = 0.0

    self.fuel = world.initial_fuel
    self.crashed = False
    self.landed = False
    self.time = 0.0

  def step(self, thrust: float, pitch_cmd: float, yaw_cmd: float, roll_cmd: float, dt: float = 0.1):
    """Simulate one time step."""
    if self.crashed or self.landed:
      return

    thrust = max(0.0, min(1.0, thrust))
    pitch_cmd = max(-1.0, min(1.0, pitch_cmd))
    yaw_cmd = max(-1.0, min(1.0, yaw_cmd))
    roll_cmd = max(-1.0, min(1.0, roll_cmd))

    # Rotation dynamics
    rot_accel = 2.0
    self.pitch_rate += pitch_cmd * rot_accel * dt
    self.yaw_rate += yaw_cmd * rot_accel * dt
    self.roll_rate += roll_cmd * rot_accel * dt

    damping = 0.95
    self.pitch_rate *= damping
    self.yaw_rate *= damping
    self.roll_rate *= damping

    self.pitch += self.pitch_rate * dt
    self.yaw += self.yaw_rate * dt
    self.roll += self.roll_rate * dt

    # Clamp pitch to avoid flipping
    self.pitch = max(-math.pi / 2, min(math.pi / 2, self.pitch))

    # Thrust along lander's "up" axis (Z-up when pitch=yaw=0)
    if self.fuel > 0 and thrust > 0:
      f = thrust * self.world.max_thrust
      self.fuel = max(0, self.fuel - thrust * dt)

      # Thrust direction: default up (0,0,1), rotated by pitch then yaw
      tx = f * (-math.sin(self.pitch) * math.cos(self.yaw))
      ty = f * (-math.sin(self.pitch) * math.sin(self.yaw))
      tz = f * math.cos(self.pitch)

      self.vx += tx * dt
      self.vy += ty * dt
      self.vz += tz * dt

    # Gravity (Z-down)
    self.vz -= self.world.gravity * dt

    self.x += self.vx * dt
    self.y += self.vy * dt
    self.z += self.vz * dt

    self.time += dt

    # Collision
    if self.world.is_solid(self.x, self.y, self.z):
      vel = math.sqrt(self.vx**2 + self.vy**2 + self.vz**2)
      if self.world.is_landing_zone(self.x, self.y, self.z) and vel < 3.0:
        self.landed = True
      else:
        self.crashed = True

  def get_state_string(self) -> str:
    """Get current state as string."""
    return (f"{self.x:.2f} {self.y:.2f} {self.z:.2f} "
            f"{self.vx:.4f} {self.vy:.4f} {self.vz:.4f} "
            f"{self.pitch:.4f} {self.yaw:.4f} {self.roll:.4f} "
            f"{self.pitch_rate:.4f} {self.yaw_rate:.4f} {self.roll_rate:.4f} "
            f"{self.fuel:.2f}")

  def distance_to_target(self) -> float:
    dx = self.x - self.world.target_x
    dy = self.y - self.world.target_y
    dz = self.z - self.world.target_z
    return math.sqrt(dx * dx + dy * dy + dz * dz)


def create_3d_visualization(world: Lander3DWorld,
                            path: List[Tuple[float, float, float]],
                            landed: bool, crashed: bool) -> str:
  """Render world + trajectory as an interactive Three.js scene (inline HTML)."""
  if world.width * world.depth > MAX_VIS_VOLUME:
    return "<p style='color:#94a3b8;'>World too large to render 3D visualization.</p>"

  # Sample terrain cells for visualization
  sample = max(1, max(world.width, world.depth) // 80)
  xs = list(range(0, world.width, sample))
  ys = list(range(0, world.depth, sample))
  heights = []
  floating_verts = []
  for ix in xs:
    row = []
    for iy in ys:
      a, b, c = world._get_cell(ix, iy)
      row.append(a)
      if c > b + 1:
        floating_verts.append([ix, iy, b, c])
    heights.append(row)

  xs_json = json.dumps(xs)
  ys_json = json.dumps(ys)
  heights_json = json.dumps(heights)
  floating_json = json.dumps(floating_verts)

  # Downsample path for rendering
  max_path_points = 500
  step = max(1, len(path) // max_path_points)
  sampled_path = path[::step]
  path_json = json.dumps([[p[0], p[1], p[2]] for p in sampled_path])

  path_color = "#38bdf8" if landed else "#ef4444" if crashed else "#f59e0b"
  uid = f"lander3d_{id(world)}_{random.randint(0,99999)}"

  return f'''
    <div style="margin:16px 0;padding:12px;border:1px solid #1f2937;border-radius:8px;background:#0b1120;">
      <h5 style="margin:0 0 10px 0;color:#e2e8f0;">3D Lander Trajectory</h5>
      <div id="{uid}" style="width:100%;height:500px;border:1px solid #334155;"></div>
      <script src="https://cdn.jsdelivr.net/npm/three@0.132.2/build/three.min.js"></script>
      <script src="https://cdn.jsdelivr.net/npm/three@0.132.2/examples/js/controls/OrbitControls.js"></script>
      <script>
      (function(){{
        const el=document.getElementById('{uid}');
        const W={world.width},D={world.depth},H={world.height};
        const scene=new THREE.Scene();
        scene.background=new THREE.Color(0x0b1120);
        const cam=new THREE.PerspectiveCamera(60,el.clientWidth/el.clientHeight,0.1,W*5);
        cam.position.set(W*0.5,D*1.2,H*1.5);
        cam.lookAt(W/2,D/2,H*0.15);
        const renderer=new THREE.WebGLRenderer({{antialias:true}});
        renderer.domElement.style.width='100%';
        renderer.domElement.style.height='100%';
        el.appendChild(renderer.domElement);
        let controls=null;
        const OrbitControlsCtor = (typeof THREE.OrbitControls === 'function') ? THREE.OrbitControls :
          ((typeof OrbitControls === 'function') ? OrbitControls : null);
        if (OrbitControlsCtor) {{
          controls=new OrbitControlsCtor(cam,renderer.domElement);
          controls.target.set(W/2,D/2,H*0.15);
          controls.update();
        }} else {{
          console.warn('OrbitControls not available; rendering without controls.');
        }}

        // Lights
        scene.add(new THREE.AmbientLight(0x5b6475,.2));
        const hemi=new THREE.HemisphereLight(0xbfd8ff,0x2b3444,1.1);
        scene.add(hemi);
        const dl=new THREE.DirectionalLight(0xffffff,.8);
        dl.position.set(W*0.35,D*0.9,H*0.9);
        scene.add(dl);
        const fill=new THREE.DirectionalLight(0x9fb3c8,0.8);
        fill.position.set(-W*0.4,-D*0.2,H*0.6);
        scene.add(fill);

        // Ground heightfield mesh
        const gx={xs_json};
        const gy={ys_json};
        const gh={heights_json};
        if(gx.length>1 && gy.length>1){{
          const nx=gx.length, ny=gy.length;
          const positions=new Float32Array(nx*ny*3);
          let p=0;
          for(let i=0;i<nx;i++){{
            for(let j=0;j<ny;j++){{
              positions[p++]=gx[i];
              positions[p++]=gh[i][j];
              positions[p++]=gy[j];
            }}
          }}
          const indices=new Uint32Array((nx-1)*(ny-1)*6);
          let k=0;
          for(let i=0;i<nx-1;i++){{
            for(let j=0;j<ny-1;j++){{
              const a=i*ny+j;
              const b=(i+1)*ny+j;
              const c=(i+1)*ny+(j+1);
              const d=i*ny+(j+1);
              indices[k++]=a;indices[k++]=d;indices[k++]=b;
              indices[k++]=b;indices[k++]=d;indices[k++]=c;
            }}
          }}
          const tg=new THREE.BufferGeometry();
          tg.setAttribute('position',new THREE.BufferAttribute(positions,3));
          tg.setIndex(new THREE.BufferAttribute(indices,1));
          tg.computeVertexNormals();
          const tm=new THREE.MeshStandardMaterial({{
            color:0x6b7280,roughness:0.85,metalness:0.02,
            emissive:0x0b0f14,emissiveIntensity:0.4,side:THREE.DoubleSide
          }});
          const mesh=new THREE.Mesh(tg,tm);
          scene.add(mesh);
        }}

        // Floating rock layers (rendered as translucent boxes)
        const fv={floating_json};
        for(const f of fv){{
          const bx={sample},by=f[3]-f[2],bz={sample};
          const sg=new THREE.BoxGeometry(bx,by,bz);
          const sm=new THREE.MeshStandardMaterial({{color:0x94a3b8,transparent:true,opacity:0.35}});
          const s=new THREE.Mesh(sg,sm);
          s.position.set(f[0]+bx/2,(f[2]+f[3])/2,f[1]+bz/2);
          scene.add(s);
        }}

        // Path
        const pp={path_json};
        if(pp.length>1){{
          const pts=pp.map(p=>new THREE.Vector3(p[0],p[2],p[1]));
          const lg=new THREE.BufferGeometry().setFromPoints(pts);
          const lm=new THREE.LineBasicMaterial({{color:'{path_color}'}});
          scene.add(new THREE.Line(lg,lm));
        }}

        // Start marker
        const ss=new THREE.Mesh(new THREE.SphereGeometry(Math.max(2,W*0.005),8,8),new THREE.MeshBasicMaterial({{color:0xfbbf24}}));
        ss.position.set({world.start_x},{world.start_z},{world.start_y});
        scene.add(ss);

        // Target marker
        const ts=new THREE.Mesh(new THREE.SphereGeometry(Math.max(2,W*0.005),8,8),new THREE.MeshBasicMaterial({{color:0x22c55e}}));
        ts.position.set({world.target_x},{world.target_z},{world.target_y});
        scene.add(ts);

        // Landing pad ring
        const rg=new THREE.RingGeometry(Math.max(3,W*0.01),Math.max(5,W*0.02),32);
        const rm=new THREE.MeshBasicMaterial({{color:0x22c55e,side:THREE.DoubleSide,transparent:true,opacity:0.6}});
        const ring=new THREE.Mesh(rg,rm);
        ring.rotation.x=-Math.PI/2;
        ring.position.set({world.target_x},{world.target_z+0.5},{world.target_y});
        scene.add(ring);

        let lastW=0,lastH=0;
        function resizeIfNeeded(){{
          const w=Math.floor(el.clientWidth);
          const h=Math.floor(el.clientHeight);
          if(w>0&&h>0&&(w!==lastW||h!==lastH)){{
            lastW=w;lastH=h;
            cam.aspect=w/h;cam.updateProjectionMatrix();
            renderer.setSize(w,h,false);
          }}
        }}
        resizeIfNeeded();
        function animate(){{requestAnimationFrame(animate);resizeIfNeeded();if(controls){{controls.update();}}renderer.render(scene,cam);}}
        animate();
      }})();
      </script>
    </div>
  '''


# Test configurations
TEST_CASES = [
  {
    "width": 100,
    "depth": 100,
    "height": 50,
    "gravity": 1.62,
    "max_thrust": 5.0,
    "fuel": 50.0,
    "max_time": 300,
    "description": "100x100x50 - simple 3D landing"
  },
  {
    "width": 200,
    "depth": 200,
    "height": 100,
    "gravity": 1.62,
    "max_thrust": 6.0,
    "fuel": 100.0,
    "max_time": 450,
    "description": "200x200x100 - short 3D hop"
  },
  {
    "width": 500,
    "depth": 500,
    "height": 500,
    "gravity": 1.62,
    "max_thrust": 8.0,
    "fuel": 250.0,
    "max_time": 600,
    "description": "500x500x500 - 3D maneuvering"
  },
  {
    "width": 1000,
    "depth": 1000,
    "height": 400,
    "gravity": 1.62,
    "max_thrust": 12.0,
    "fuel": 500.0,
    "max_time": 900,
    "description": "1km - cross-crater 3D"
  },
  {
    "width": 2000,
    "depth": 2000,
    "height": 1000,
    "gravity": 1.62,
    "max_thrust": 18.0,
    "fuel": 1000.0,
    "max_time": 1200,
    "description": "2km x 2km x 1km - regional 3D"
  },
  {
    "width": 5000,
    "depth": 5000,
    "height": 2000,
    "gravity": 1.62,
    "max_thrust": 25.0,
    "fuel": 3000.0,
    "max_time": 1800,
    "description": "5km - long range 3D"
  },
  {
    "width": 10000,
    "depth": 10000,
    "height": 500,
    "gravity": 1.62,
    "max_thrust": 40.0,
    "fuel": 8000.0,
    "max_time": 2400,
    "description": "10km - continental 3D"
  },
  {
    "width": 25000,
    "depth": 25000,
    "height": 1000,
    "gravity": 1.62,
    "max_thrust": 60.0,
    "fuel": 20000.0,
    "max_time": 3000,
    "description": "25km - suborbital 3D"
  },
  {
    "width": 50000,
    "depth": 50000,
    "height": 2500,
    "gravity": 1.62,
    "max_thrust": 100.0,
    "fuel": 50000.0,
    "max_time": 3000,
    "description": "50km - orbital 3D"
  },
  {
    "width": 100000,
    "depth": 10000,
    "height": 5000,
    "gravity": 1.62,
    "max_thrust": 150.0,
    "fuel": 120000.0,
    "max_time": 3600,
    "description": "100km - trans-lunar 3D"
  },
]

WORLD_CACHE = {}


def get_world(subpass: int) -> Lander3DWorld:
  """Get or generate world for subpass."""
  if subpass not in WORLD_CACHE:
    case = TEST_CASES[subpass]
    WORLD_CACHE[subpass] = Lander3DWorld(case["width"], case["depth"], case["height"],
                                         case["gravity"], case["max_thrust"], case["fuel"],
                                         RANDOM_SEED + subpass)
  return WORLD_CACHE[subpass]


def format_input(world: Lander3DWorld) -> List[str]:
  """Format world as list of input lines (each ending with newline).

  Format:
    width depth height gravity max_thrust fuel
    start_x start_y start_z target_x target_y target_z
    terrain_file_path    (on its own line to handle spaces in path)
    STATE
  """
  lines = []
  lines.append(
    f"{world.width} {world.depth} {world.height} {world.gravity} {world.max_thrust} {world.initial_fuel}\n"
  )
  lines.append(
    f"{world.start_x} {world.start_y} {world.start_z:.2f} {world.target_x} {world.target_y} {world.target_z:.2f}\n"
  )
  lines.append(f"{world.terrain_path}\n")
  lines.append("STATE\n")
  return lines


def prepareSubpassPrompt(subPass: int) -> str:
  """Generate the prompt for subpass 0 that handles all landing complexities."""
  if subPass != 0:
    raise StopIteration

  return f"""You are writing C++ code to play a 3D Lunar Lander game in real-time.

Your C++ controller will be tested with scenarios ranging from tiny (100m) to huge (100km).
The same code must work for all scales.

Control a lunar lander in full 3D space to safely land on a target pad.
The program receives a terrain file path and continuous state updates via stdin, and must
output thrust/rotation commands to stdout in real-time.

Data is streamed to stdin in real time, and you must write to stdout in real time.
If your code doesn't read from stdin fast enough and the IO buffer fills up, the ship
will likely crash.

Be very careful if writing single threaded / blocking code, this has many failure modes
that async code doesn't have. Multithreaded code is HIGHLY recommended.

**Coordinate System:**
- X: horizontal (East), Y: horizontal (North), Z: UP.
- Gravity acts in the -Z direction.
- All positions are in meters. Velocities in m/s. Angles in radians.

**Physics:**
- Gravity = 1.62 m/s² (lunar) in -Z direction.
- Thrust is along the lander's local "up" axis. When pitch=0 and yaw=0, thrust is
  straight up (+Z). Tilting via pitch/yaw redirects thrust.
- Pitch: tilts the lander. +pitch tilts forward, redirecting thrust partially into -X.
  Pitch is clamped to [-pi/2, pi/2].
- Yaw: rotates heading around Z.
- Roll: rolls around the thrust axis (cosmetic only, does not affect thrust direction).
- Rotation dynamics: commands apply angular acceleration (2 rad/s² per unit).
  Angular rates are damped by 0.95 per tick.
- Fuel: 100% thrust burns 1 unit of fuel per second.

**Input format (stdin):**
Initial header (whitespace separated decimal numbers, except terrain path):
```
width depth height gravity max_thrust fuel
start_x start_y start_z target_x target_y target_z
terrain_file_path
STATE
```

The terrain_file_path is on its own line (the path may contain spaces).
Read the entire line with getline() or equivalent.

**Terrain file format (binary):**
The terrain file defines the 3D world as a binary file. It is a sequence of
uint16_t triplets (little-endian), one per (x, y) cell, stored in row-major order:

```cpp
for (int x = 0; x < width; x++) {{
  for (int y = 0; y < depth; y++) {{
    uint16_t a = read_uint16_le();  // ground rock top
    uint16_t b = read_uint16_le();  // floating rock bottom
    uint16_t c = read_uint16_le();  // floating rock top
  }}
}}
```

For each column (x, y), the vertical structure is:
- z < a: SOLID ROCK (ground)
- a <= z < b: OPEN AIR
- b <= z < c: SOLID ROCK (floating terrain feature, overhang, arch, etc.)
- c <= z: OPEN AIR

If b == c, there is no floating rock in that column (just ground + open sky).
The file size is exactly width * depth * 6 bytes.

You have a startup grace period to parse this file before state updates begin.
The grace period is 5 seconds plus 10 seconds per gigabyte of terrain file.
For a 100x100 map this is ~5s. For a 100000x10000 map (~6GB) this is ~65s.
You may continue parsing in a background thread after the simulation begins,
but you MUST be reading stdin and writing stdout by then to avoid a crash.

Then continuous state updates (every 0.1 seconds, whitespace separated):
```
x y z vx vy vz pitch yaw roll pitch_rate yaw_rate roll_rate fuel_remaining
```

**Output format (stdout):**
Your code needs to output the following at a frequency of at least 10hz:
```
thrust pitch_cmd yaw_cmd roll_cmd
```
- thrust: 0.0 to 1.0 (fraction of max_thrust)
- pitch_cmd: -1.0 to 1.0 (pitch angular acceleration command)
- yaw_cmd: -1.0 to 1.0 (yaw angular acceleration command)
- roll_cmd: -1.0 to 1.0 (roll angular acceleration command)

To be crystal clear: four decimal numbers separated by spaces, followed by a newline.

If your code stalls / freezes for 100ms or more (misses 2 consecutive updates), the
engine will cut thrust automatically. A long freeze and eventually the lander will crash.

**Landing:**
- The target is at (target_x, target_y, target_z) which is just above the terrain.
- You must land within a small radius of the target with total speed < 3 m/s.
- Landing zone radius is approximately max(5, min(width,depth)*0.02) meters.

**Terrain collisions:**
- You collide if you enter any solid cell as defined by the terrain file.
- Out of bounds (x<0, x>=width, y<0, y>=depth, z<0, z>=height) is also a collision.

**Success Conditions:**
- Land on target with |velocity| < 3 m/s
- Stay within landing zone boundaries

**Failure Conditions:**
- Crash into terrain or ground too hard
- Run out of fuel
- Exceed simulation time limit

**Environment:**
{describe_this_pc()}

**C++ Compiler:**
{CppCompiler("test_engine").describe()}

Write complete, compilable C++ code with a main() function.

Be sure that any deviation from the C++ standard library is supported by the given compiler,
as referencing the wrong intrinsics or non-standard header like 'bits/stdc++.h' could fail your submission.

"""


extraGradeAnswerRuns = list(range(len(TEST_CASES)))

structure = {
  "type": "object",
  "properties": {
    "reasoning": {
      "type":
      "string",
      "description":
      "Explain your algorithm approach and how it adapts to different 3D landing complexities"
    },
    "cpp_code": {
      "type": "string",
      "description": "Complete C++ code with main() function that handles all scales"
    }
  },
  "required": ["reasoning", "cpp_code"],
  "additionalProperties": False
}


def run_lander_simulation(code: str, case: dict, subpass: int,
                          engine_name: str) -> Tuple[bool, float, str, float]:
  """Compile and run 3D lander simulation with async IO."""
  compiler = CppCompiler(engine_name)

  if not compiler.find_compiler():
    return False, float('inf'), "No C++ compiler found", 0

  try:
    exe_path = compiler.compile(code)
  except CompilationError as e:
    return False, float('inf'), f"Compilation error: <br><pre>{str(e)[:500]}</pre>", 0

  world = get_world(subpass)
  sim = Lander3DSimulator(world)

  start_time = time.time()

  try:
    process = subprocess.Popen([str(exe_path)],
                               stdin=subprocess.PIPE,
                               stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE,
                               text=True,
                               bufsize=1)

    stdout_queue: Queue = Queue()
    stdin_queue: Queue = Queue(maxsize=200)
    stop_event = threading.Event()
    dropped_inputs = 0
    debug_log: List[str] = []
    log_enabled = subpass == 0
    log_limit = 400

    def log_event(message: str) -> None:
      if not log_enabled:
        return
      if len(debug_log) < log_limit:
        debug_log.append(message)

    def stdout_reader():
      if not process.stdout:
        return
      while not stop_event.is_set():
        try:
          line = process.stdout.readline()
          if not line:
            break
          stdout_queue.put(line)
          log_event(f"stdout: {line.strip()}")
        except Exception:
          break

    def stdin_writer():
      if not process.stdin:
        return
      while not stop_event.is_set():
        try:
          line = stdin_queue.get(timeout=0.1)
        except Empty:
          continue
        if line is None:
          break
        try:
          process.stdin.write(line)
          process.stdin.flush()
          log_event(f"stdin: {line.strip()}")
        except Exception:
          break

    reader_thread = threading.Thread(target=stdout_reader, daemon=True)
    writer_thread = threading.Thread(target=stdin_writer, daemon=True)
    reader_thread.start()
    writer_thread.start()

    # Send initial header (non-blocking, may drop if buffer fills)
    header_lines = format_input(world)
    for line in header_lines:
      try:
        stdin_queue.put(line, timeout=5)
      except Exception:
        dropped_inputs += 1
        log_event("stdin_drop: header")

    # Grace period for terrain file parsing: 5s + 10s per GB
    startup_delay = world.startup_delay()
    log_event(f"startup_delay: {startup_delay:.1f}s (terrain {world.terrain_file_size_gb():.3f} GB)")
    time.sleep(startup_delay)

    # Drain any commands sent during startup (before sim begins)
    while True:
      try:
        stdout_queue.get_nowait()
      except Empty:
        break

    max_time = case["max_time"]
    dt = 0.1
    end_reason = "timeout"
    path: List[Tuple[float, float, float]] = []
    ticks = 0
    valid_commands = 0
    invalid_commands = 0
    no_command_ticks = 0
    commands_received = 0
    command_intervals: List[float] = []
    last_command_time: Optional[float] = None
    process_exited = False

    next_tick = time.time()
    while sim.time < max_time:
      now = time.time()
      if now < next_tick:
        time.sleep(min(0.01, next_tick - now))
        continue
      next_tick += dt

      if sim.crashed:
        end_reason = "crashed"
        break
      if sim.landed:
        end_reason = "landed"
        break

      if process.poll() is not None:
        process_exited = True
        log_event(f"process_exit: {process.returncode}")

      if time.time() - start_time > TIMEOUT_SECONDS:
        end_reason = "real_timeout"
        break

      # Send state (non-blocking)
      state_line = sim.get_state_string() + "\n"
      try:
        stdin_queue.put_nowait(state_line)
      except Exception:
        dropped_inputs += 1
        log_event("stdin_drop: state")

      # Drain output queue and keep most recent command
      latest_line = None
      while True:
        try:
          latest_line = stdout_queue.get_nowait()
        except Empty:
          break

      if latest_line:
        commands_received += 1
        now_time = time.time()
        if last_command_time is not None:
          command_intervals.append(now_time - last_command_time)
        last_command_time = now_time
        parts = latest_line.strip().split()
        if len(parts) >= 4:
          try:
            thrust = float(parts[0])
            pitch_cmd = float(parts[1])
            yaw_cmd = float(parts[2])
            roll_cmd = float(parts[3])
            valid_commands += 1
          except Exception:
            thrust = pitch_cmd = yaw_cmd = roll_cmd = 0.0
            invalid_commands += 1
            log_event(f"parse_error: {latest_line.strip()}")
        else:
          thrust = pitch_cmd = yaw_cmd = roll_cmd = 0.0
          invalid_commands += 1
          log_event(f"invalid_cmd: {latest_line.strip()}")
      else:
        thrust = pitch_cmd = yaw_cmd = roll_cmd = 0.0
        no_command_ticks += 1
        log_event("no_cmd")

      sim.step(thrust, pitch_cmd, yaw_cmd, roll_cmd, dt)
      path.append((sim.x, sim.y, sim.z))
      ticks += 1

    # Send END signal (non-blocking)
    try:
      stdin_queue.put_nowait("END\n")
    except Exception:
      dropped_inputs += 1
      log_event("stdin_drop: END")

    stop_event.set()
    try:
      stdin_queue.put_nowait(None)
    except Exception:
      pass

    process.terminate()
    try:
      process.wait(timeout=2)
    except:
      process.kill()

    exec_time = time.time() - start_time
    distance = sim.distance_to_target()

    avg_command_interval = (sum(command_intervals) / len(command_intervals)) if command_intervals else 0
    LAST_LANDER_STATS[(subpass, engine_name)] = {
      "world": world,
      "path": path,
      "ticks": ticks,
      "sim_time": sim.time,
      "valid_commands": valid_commands,
      "invalid_commands": invalid_commands,
      "commands_received": commands_received,
      "no_command_ticks": no_command_ticks,
      "avg_command_interval": avg_command_interval,
      "dropped_inputs": dropped_inputs,
      "process_exited": process_exited,
      "end_reason": end_reason,
      "exec_time": exec_time,
      "distance": distance,
      "landed": sim.landed,
      "crashed": sim.crashed,
      "fuel_remaining": sim.fuel,
      "final_vx": sim.vx,
      "final_vy": sim.vy,
      "final_vz": sim.vz,
      "debug_log": debug_log,
    }

    return sim.landed, distance, end_reason, exec_time

  except Exception as e:
    return False, float('inf'), f"Execution error: {str(e)}", time.time() - start_time


def gradeAnswer(result: dict, subPass: int, aiEngineName: str) -> tuple:
  """Grade the C++ 3D lander controller."""
  if not result:
    return 0.0, "No result provided"

  if "cpp_code" not in result:
    return 0.0, "No C++ code provided"

  cache_parts = _cache_key_parts(result, subPass)
  cached = _grade_cache.get_grade(*cache_parts)
  if cached is not None:
    return cached

  case = TEST_CASES[subPass]
  description = case["description"]
  t = time.time()
  world = get_world(subPass)
  gen_time = time.time() - t
  if gen_time > 1:
    print(f"Generating the 3D world took {gen_time:.2f}s for subpass {subPass}")
  max_distance = math.sqrt(world.width**2 + world.depth**2 + world.height**2)

  code = result["cpp_code"]

  landed, distance, end_reason, exec_time = run_lander_simulation(code, case, subPass, aiEngineName)

  if landed:
    score = 1.0
  elif end_reason == "crashed":
    score = 0
  elif end_reason in ("timeout", "real_timeout"):
    score = max(0.05, 0.3 * (1 - distance / max_distance))
  else:
    score = 0.0

  explanation = (f"[{description}] End: {end_reason}, "
                 f"Distance: {distance:.1f}m, "
                 f"Time: {exec_time:.2f}s")

  grade = (score, explanation)
  _grade_cache.put_grade(grade, *cache_parts)
  return grade


def resultToNiceReport(result: dict, subPass: int, aiEngineName: str) -> str:
  if not result:
    return "<p style='color:red'>No result provided</p>"
  cache_parts = _cache_key_parts(result, subPass)
  cached = _grade_cache.get_report(*cache_parts)
  if cached is not None:
    return cached
  case = TEST_CASES[subPass]
  html = f"<h4>3D Lunar Lander - {case['description']}</h4>"
  if "reasoning" in result:
    r = result['reasoning'][:400] + ('...' if len(result.get('reasoning', '')) > 400 else '')
    html += f"<p><strong>Approach:</strong> {r.replace('<', '&lt;').replace('>', '&gt;')}</p>"
  if "cpp_code" in result:
    code = result["cpp_code"].replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    html += f"<details><summary>View C++ Code ({len(result['cpp_code'])} chars)</summary><pre>{code}</pre></details>"
  stats = LAST_LANDER_STATS.get((subPass, aiEngineName))
  if stats:
    tick_rate = stats["ticks"] / stats["sim_time"] if stats["sim_time"] > 0 else 0
    responsiveness = (stats["valid_commands"] / stats["ticks"]) if stats["ticks"] > 0 else 0
    html += f"""
      <div style="margin: 14px 0; padding: 12px; border: 1px solid #1f2937; border-radius: 8px; background: #0b1120; color: #e2e8f0;">
        <h5 style="margin: 0 0 10px 0; color: #e2e8f0;">Session Stats</h5>
        <table style="border-collapse: collapse; width: 100%; font-size: 13px; color: #e2e8f0;">
          <tr>
            <td style="padding: 4px 8px; color: #94a3b8;">Ticks</td>
            <td style="padding: 4px 8px; font-weight: 600;">{stats['ticks']}</td>
            <td style="padding: 4px 8px; color: #94a3b8;">Sim time</td>
            <td style="padding: 4px 8px; font-weight: 600;">{stats['sim_time']:.1f}s</td>
          </tr>
          <tr>
            <td style="padding: 4px 8px; color: #94a3b8;">Valid / Invalid commands</td>
            <td style="padding: 4px 8px; font-weight: 600;">{stats['valid_commands']} / {stats['invalid_commands']}</td>
            <td style="padding: 4px 8px; color: #94a3b8;">No-command ticks</td>
            <td style="padding: 4px 8px; font-weight: 600;">{stats['no_command_ticks']}</td>
          </tr>
          <tr>
            <td style="padding: 4px 8px; color: #94a3b8;">Tick rate</td>
            <td style="padding: 4px 8px; font-weight: 600;">{tick_rate:.1f} Hz</td>
            <td style="padding: 4px 8px; color: #94a3b8;">Responsiveness</td>
            <td style="padding: 4px 8px; font-weight: 600;">{responsiveness:.1%}</td>
          </tr>
          <tr>
            <td style="padding: 4px 8px; color: #94a3b8;">Avg command interval</td>
            <td style="padding: 4px 8px; font-weight: 600;">{stats['avg_command_interval']:.2f}s</td>
            <td style="padding: 4px 8px; color: #94a3b8;">Dropped state updates</td>
            <td style="padding: 4px 8px; font-weight: 600;">{stats['dropped_inputs']}</td>
          </tr>
          <tr>
            <td style="padding: 4px 8px; color: #94a3b8;">Fuel remaining</td>
            <td style="padding: 4px 8px; font-weight: 600;">{stats['fuel_remaining']:.1f}</td>
            <td style="padding: 4px 8px; color: #94a3b8;">Final velocity</td>
            <td style="padding: 4px 8px; font-weight: 600;">{stats['final_vx']:.1f}, {stats['final_vy']:.1f}, {stats['final_vz']:.1f}</td>
          </tr>
          <tr>
            <td style="padding: 4px 8px; color: #94a3b8;">Process exited early</td>
            <td style="padding: 4px 8px; font-weight: 600;">{str(stats['process_exited'])}</td>
            <td style="padding: 4px 8px; color: #94a3b8;">End reason</td>
            <td style="padding: 4px 8px; font-weight: 600;">{stats['end_reason']}</td>
          </tr>
          <tr>
            <td style="padding: 4px 8px; color: #94a3b8;">Terrain file</td>
            <td style="padding: 4px 8px; font-weight: 600;">{stats['world'].terrain_file_size_gb()*1024:.3f} MB</td>
            <td style="padding: 4px 8px; color: #94a3b8;">Startup delay</td>
            <td style="padding: 4px 8px; font-weight: 600;">{stats['world'].startup_delay():.1f}s</td>
          </tr>
        </table>
      </div>
    """
    html += create_3d_visualization(stats["world"], stats["path"], stats["landed"], stats["crashed"])
    if subPass == 0 and stats.get("end_reason") == "crashed":
      debug_log = stats.get("debug_log", [])
      if debug_log:
        log_text = "\n".join(debug_log)
        log_text = log_text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        html += (
          "<details><summary>Crash Debug Log (subpass 0)</summary>"
          f"<pre style='white-space: pre-wrap; background: #0b1120; color: #e2e8f0; padding: 10px; border-radius: 6px;'>"
          f"{log_text}</pre></details>"
        )

  _grade_cache.put_report(html, *cache_parts)
  return html


highLevelSummary = """
<p>Land a spacecraft on a 3D lunar surface with full attitude control. Unlike the
2D version, the lander can pitch, yaw, and roll, and the terrain is a detailed
3D heightmap with floating rock obstacles. The AI must parse a binary terrain
file and navigate to a safe touchdown.</p>
<p>Subpasses vary the terrain complexity, starting conditions, and obstacle density.
The AI must handle 6-degree-of-freedom physics and real-time control in three
dimensions.</p>
"""
