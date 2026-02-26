"""
Test 23: Lunar Lander Game (Rust Implementation)

The LLM must write Rust code that plays Lunar Lander in real-time.
"""

import random
import subprocess
import sys
import os
import time
import math
import threading
from queue import Queue, Empty
from typing import List, Tuple, Set, Dict, Optional
from pathlib import Path

# Import our native compiler helper
from native_compiler import RustCompiler, CompilationError, ExecutionError, describe_this_pc
from solver_utils import parse_freeform_response

title = "Lunar Lander Game"

tags = [
  "rust",
  "freeform response",
  "game ai",
  "simulation",
]

# Timeout in seconds (60 minutes)
TIMEOUT_SECONDS = 3600

# Seed for reproducibility
RANDOM_SEED = 232323

MAX_SVG_CELLS = 200_000
MAX_SVG_DIM = 900

LAST_LANDER_STATS: Dict[Tuple[int, str], Dict] = {}


class LunarLanderWorld:
  """Represents a lunar landing world."""

  def __init__(self, width: int, height: int, gravity: float, max_thrust: float, fuel: float):
    self.width = width
    self.height = height
    self.gravity = gravity
    self.max_thrust = max_thrust
    self.initial_fuel = fuel

    # Generate landing site
    self.start_x = width // 4
    self.start_y = max(1, int(height * 0.2))
    self.target_x = 3 * width // 4
    self.target_y = height // 2

    # Create map
    self.map = [['.' for _ in range(width)] for _ in range(height)]

    # Add terrain and landing zone
    self._add_obstacles()
    self._place_landing_zone()

  def _add_obstacles(self):
    """Add blobby terrain using a smoothed noise heightmap."""
    rng = random.Random(RANDOM_SEED)
    step = max(12, self.width // 60)
    control_points = []
    for x in range(0, self.width + step, step):
      base = int(self.height * 0.7)
      variance = int(self.height * 0.12)
      control_points.append(base + rng.randint(-variance, variance))

    heights = [0] * self.width
    for x in range(self.width):
      idx = x // step
      t = (x % step) / step
      h0 = control_points[idx]
      h1 = control_points[min(idx + 1, len(control_points) - 1)]
      heights[x] = int(h0 * (1 - t) + h1 * t)

    # Smooth heights
    window = max(3, step // 2)
    smoothed = []
    for x in range(self.width):
      left = max(0, x - window)
      right = min(self.width, x + window + 1)
      smoothed.append(int(sum(heights[left:right]) / (right - left)))

    ground_heights = []
    for x in range(self.width):
      ground_y = max(2, min(self.height - 1, smoothed[x]))
      ground_heights.append(ground_y)
      for y in range(ground_y, self.height):
        self.map[y][x] = '#'

    # Add some random rock outcrops above the ground
    rock_count = rng.randint(2, 6)
    max_radius = max(2, min(self.height // 10, 30))
    for _ in range(rock_count):
      x = rng.randint(0, self.width - 1)
      ground_y = ground_heights[x]
      r = rng.randint(2, max_radius)
      center_y = max(1, ground_y - rng.randint(r + 1, r * 3))
      for dx in range(-r, r + 1):
        for dy in range(-r, r + 1):
          if dx * dx + dy * dy <= r * r:
            px = x + dx
            py = center_y + dy
            if 0 <= px < self.width and 0 <= py < self.height:
              if self.map[py][px] == '.':
                self.map[py][px] = '#'

  def _place_landing_zone(self):
    """Flatten terrain around the landing zone and place target markers."""
    pad_half = 1
    pad_left = max(1, self.target_x - pad_half)
    pad_right = min(self.width - 2, self.target_x + pad_half)

    # Find terrain height at target and flatten a small pad
    terrain_y = None
    for y in range(self.height):
      if self.map[y][self.target_x] == '#':
        terrain_y = y
        break
    if terrain_y is None:
      terrain_y = int(self.height * 0.75)

    self.target_y = max(1, terrain_y - 1)
    for x in range(pad_left, pad_right + 1):
      for y in range(self.target_y, self.height):
        self.map[y][x] = '#'
      self.map[self.target_y][x] = 'L'
      for y in range(0, self.target_y):
        if self.map[y][x] == '#':
          self.map[y][x] = '.'

    self.map[self.target_y][self.target_x] = 'T'

  def get_map_string(self) -> str:
    """Get the map as a string."""
    return '\n'.join(''.join(row) for row in self.map)


class LanderSimulator:
  """Simulates the lunar lander physics."""

  def __init__(self, world: LunarLanderWorld):
    self.world = world
    self.reset()

  def reset(self):
    """Reset the simulation."""
    self.x = float(self.world.start_x)
    self.y = float(self.world.start_y)
    self.vx = 0.0
    self.vy = 0.0
    self.angle = 0.0  # 0 = pointing up
    self.angular_vel = 0.0
    self.fuel = self.world.initial_fuel
    self.time = 0.0
    self.crashed = False
    self.landed = False

  def step(self, thrust: float, turn: float, dt: float):
    """Advance simulation by one time step."""
    if self.crashed or self.landed:
      return

    # Update angle
    self.angle += turn * dt
    self.angle = max(-math.pi, min(math.pi, self.angle))

    # Apply thrust
    if self.fuel > 0 and thrust > 0:
      # Thrust in facing direction
      fx = math.sin(self.angle) * thrust * self.world.max_thrust
      fy = -math.cos(self.angle) * thrust * self.world.max_thrust
      self.fuel -= thrust * dt
    else:
      fx = fy = 0.0

    # Apply gravity
    fy += self.world.gravity

    # Update velocity
    self.vx += fx * dt
    self.vy += fy * dt

    # Update position
    self.x += self.vx * dt
    self.y += self.vy * dt

    # Check boundaries
    if self.x < 0 or self.x >= self.world.width or self.y < 0 or self.y >= self.world.height:
      self.crashed = True
      return

    # Check collision with obstacles
    map_x = int(self.x)
    map_y = int(self.y)
    if 0 <= map_x < self.world.width and 0 <= map_y < self.world.height:
      if self.world.map[map_y][map_x] == '#':
        self.crashed = True
        return

    # Check landing
    if self.world.map[map_y][map_x] in ['T', 'L']:
      if abs(self.vy) < 2.0 and abs(self.vx) < 2.0:
        self.landed = True
      else:
        self.crashed = True

    self.time += dt

  def get_state_string(self) -> str:
    """Get current state as a string for the program."""
    return f"{self.x:.2f} {self.y:.2f} {self.vx:.2f} {self.vy:.2f} {self.angle:.3f} {self.angular_vel:.3f} {self.fuel:.1f}"

  def distance_to_target(self) -> float:
    """Get distance to target."""
    return math.sqrt((self.x - self.world.target_x)**2 + (self.y - self.world.target_y)**2)


def create_lander_svg(world: LunarLanderWorld,
                      path: List[Tuple[float, float]],
                      landed: bool,
                      crashed: bool) -> str:
  """Render the world and lander path as SVG for reasonably sized maps."""
  if world.width * world.height > MAX_SVG_CELLS:
    return "<p>World too large to render (visualization skipped).</p>"

  scale = min(MAX_SVG_DIM / world.width, MAX_SVG_DIM / world.height, 12)
  cell_size = max(3, scale)
  margin = 16
  svg_width = int(world.width * cell_size + margin * 2)
  svg_height = int(world.height * cell_size + margin * 2)

  svg_elements = [
    f'<rect width="{svg_width}" height="{svg_height}" fill="#0b1120" stroke="#334155" stroke-width="2"/>'
  ]

  for y in range(world.height):
    for x in range(world.width):
      cell = world.map[y][x]
      if cell == '.':
        continue
      color = "#1f2937"
      if cell == '#':
        color = "#475569"
      elif cell in ('T', 'L'):
        color = "#22c55e"
      px = margin + x * cell_size
      py = margin + y * cell_size
      svg_elements.append(
        f'<rect x="{px}" y="{py}" width="{cell_size}" height="{cell_size}" fill="{color}" />'
      )

  if path:
    points = []
    for x, y in path:
      px = margin + x * cell_size + cell_size / 2
      py = margin + y * cell_size + cell_size / 2
      points.append(f"{px:.1f},{py:.1f}")
    path_color = "#38bdf8" if landed else "#ef4444" if crashed else "#f59e0b"
    svg_elements.append(
      f'<polyline points="{" ".join(points)}" fill="none" stroke="{path_color}" stroke-width="2" opacity="0.9" />'
    )

  start_x = margin + world.start_x * cell_size + cell_size / 2
  start_y = margin + world.start_y * cell_size + cell_size / 2
  target_x = margin + world.target_x * cell_size + cell_size / 2
  target_y = margin + world.target_y * cell_size + cell_size / 2
  svg_elements.append(
    f'<circle cx="{start_x:.1f}" cy="{start_y:.1f}" r="{cell_size * 0.6:.1f}" fill="#fbbf24" />'
  )
  svg_elements.append(
    f'<circle cx="{target_x:.1f}" cy="{target_y:.1f}" r="{cell_size * 0.6:.1f}" fill="#22c55e" />'
  )

  svg_html = f'''
    <div style="margin: 16px 0; padding: 12px; border: 1px solid #1f2937; border-radius: 8px; background: #0b1120;">
      <h5 style="margin: 0 0 10px 0; color: #e2e8f0;">Lander Trajectory</h5>
      <svg width="{svg_width}" height="{svg_height}" style="border: 1px solid #334155; background: #0b1120;">
        {"".join(svg_elements)}
      </svg>
    </div>
  '''

  return svg_html


def generate_world(width: int, height: int, gravity: float, max_thrust: float, fuel: float,
                   seed: int) -> LunarLanderWorld:
  """Generate a lunar lander world."""
  rng = random.Random(seed)
  world = LunarLanderWorld(width, height, gravity, max_thrust, fuel)
  return world


# Test cases with varying difficulty
TEST_CASES = [
  # Subpass 0: Tiny world
  {
    "width": 100,
    "height": 50,
    "gravity": 1.62,  # Moon gravity
    "max_thrust": 5.0,
    "fuel": 50.0,
    "max_time": 30.0,
    "description": "100x50 - simple landing"
  },
  # Subpass 1: Small world
  {
    "width": 200,
    "height": 100,
    "gravity": 1.62,
    "max_thrust": 5.0,
    "fuel": 100.0,
    "max_time": 45.0,
    "description": "200x100 - short hop"
  },
  # Subpass 2: Medium world
  {
    "width": 500,
    "height": 200,
    "gravity": 1.62,
    "max_thrust": 8.0,
    "fuel": 200.0,
    "max_time": 60.0,
    "description": "500x200 - orbital insertion"
  },
  # Subpass 3: Larger world
  {
    "width": 1000,
    "height": 400,
    "gravity": 1.62,
    "max_thrust": 10.0,
    "fuel": 400.0,
    "max_time": 90.0,
    "description": "1000x400 - cross-crater hop"
  },
  # Subpass 4: Large world
  {
    "width": 2000,
    "height": 800,
    "gravity": 1.62,
    "max_thrust": 15.0,
    "fuel": 800.0,
    "max_time": 120.0,
    "description": "2km x 800m - regional transfer"
  },
  # Subpass 5: Very large
  {
    "width": 5000,
    "height": 2000,
    "gravity": 1.62,
    "max_thrust": 20.0,
    "fuel": 2000.0,
    "max_time": 180.0,
    "description": "5km x 2km - long range"
  },
  # Extreme cases
  {
    "width": 10000,
    "height": 4000,
    "gravity": 1.62,
    "max_thrust": 30.0,
    "fuel": 5000.0,
    "max_time": 240.0,
    "description": "10km x 4km - continental"
  },
  {
    "width": 25000,
    "height": 10000,
    "gravity": 1.62,
    "max_thrust": 50.0,
    "fuel": 15000.0,
    "max_time": 300.0,
    "description": "25km x 10km - suborbital"
  },
  {
    "width": 50000,
    "height": 20000,
    "gravity": 1.62,
    "max_thrust": 80.0,
    "fuel": 40000.0,
    "max_time": 300.0,
    "description": "50km x 20km - orbital transfer"
  },
  {
    "width": 100000,
    "height": 40000,
    "gravity": 1.62,
    "max_thrust": 100.0,
    "fuel": 100000.0,
    "max_time": 300.0,
    "description": "100km x 40km - trans-lunar injection"
  },
  {
    "width": 500000,
    "height": 200000,
    "gravity": 1.62,
    "max_thrust": 200.0,
    "fuel": 500000.0,
    "max_time": 300.0,
    "description": "500km x 200km - interplanetary scale"
  },
]

# Cache worlds
WORLD_CACHE = {}

extraGradeAnswerRuns = list(range(len(TEST_CASES)))


def get_world(subpass: int) -> LunarLanderWorld:
  """Get or generate world for subpass."""
  if subpass not in WORLD_CACHE:
    case = TEST_CASES[subpass]
    WORLD_CACHE[subpass] = generate_world(case["width"], case["height"], case["gravity"],
                                          case["max_thrust"], case["fuel"], RANDOM_SEED + subpass)
  return WORLD_CACHE[subpass]


def prepareSubpassPrompt(subPass: int) -> str:
  """Generate the prompt for subpass 0 that handles all landing complexities."""
  if subPass != 0:
    raise StopIteration

  return f"""You are writing Rust code to play Lunar Lander in real-time.

Your Rust lunar lander controller will be tested with scenarios ranging from simple 
training environments to very complex landing sites.

Control a lunar lander to safely land on a target pad. 
The program receives a 2D voxel map and continuous state updates, and must output thrust/turn 
commands to land safely on the target.

Data is streamed to stdin in real time, and you must write to stdout in real time. If your code
doesn't read from stdin fast enough and the io buffer fills up, the ship will likely crash. 

Be very careful if writing single threaded / blocking code, this has many failure modes that
async code doesn't have. Multithreaded / asyncio code is HIGHLY recommended.

**Physics:**
- Gravity pulls lander down constantly
- Thrust applies force in lander's facing direction
- Fuel is limited
- Must land softly on target (low velocity)

**Input format (stdin, whitespace seperated decimal numbers):**
Initial map:
```
width height gravity max_thrust fuel
start_x start_y target_x target_y
map rows (. = empty, # = obstacle, S = start, T = target, L = landing zone)
```

Then you recieve continuous state updates (every 0.1 seconds):
```
x y vx vy angle angular_vel fuel_remaining
```
x and y are in meters, from the worlds top left, and y increases downward.
vx and vy are in m/s.
angle is in radians, 0 is facing up.
angular_vel is in radians per second. + is clockwise, - is counterclockwise.
fuel_remaining is seconds at max_thrust.

**Output format (stdout):**
Your code needs to output the following instructions at a frequency of at least 20hz: 
```
thrust turn
```
- thrust: 0.0 to 1.0 (fraction of max thrust)
- turn: -1.0 to 1.0 (turning rate, negative = left, positive = right)

To be crystal clear: that's two decimal numbers seperated by a space, followed by a newline.

If your code stalls / freezes for 100ms or more (so it misses 2 consecutive updates), the engine
will automatically cut power. A long freeze and eventually the lander will crash.

Map size can vary from 100x50 to multi-gb maps, so be efficient with memory and processing.

Thrust and gravity are in m/s/s. 100% engine burn burns 1 unit of fuel per second.

Lander size is approximatly 1x1m, and the 2D map grid size is 1m x 1m. 

Turn is in radians per second. The lander can not rotate fully around and will clamp rotation to 
facing down if attempted.

Y increases downward. Positive vy means moving downward. Gravity is +g.

**Success Conditions:**
- Land on target with |velocity| < 2m/s
- Stay within landing zone boundaries
- Complete landing within time limit

**Failure Conditions:**
- Crash into obstacle or ground too hard
- Run out of fuel
- Exceed time limit

**Environment:**
{describe_this_pc()}

**Rust Compiler:**
{RustCompiler("test_engine").describe()}

Write complete, compilable Rust code with a main function. Assume only the standard library is available.
"""


# List of subpasses to grade the single answer against all difficulty levels
extraGradeAnswerRuns = list(range(len(TEST_CASES)))

structure = None


def _extract_freeform(result):
  if isinstance(result, dict):
    discussion = result.get("reasoning") or result.get("discussion") or ""
    code = result.get("rust_code") or result.get("code") or ""
    return discussion, code, ""
  if isinstance(result, str) and result.strip() == "__content_violation__":
    return "", "", "Content violation"
  parsed = parse_freeform_response(result or "")
  return parsed.get("discussion", ""), parsed.get("code", ""), ""


def run_lander_simulation(code: str, case: dict, subpass: int,
                          engine_name: str) -> Tuple[bool, float, str, float]:
  """
    Compile and run lander simulation.
    
    Returns:
        Tuple of (landed, distance_to_target, end_reason, exec_time)
    """
  compiler = RustCompiler(engine_name)

  if not compiler.find_compiler():
    return False, float('inf'), "No Rust compiler found", 0

  try:
    exe_path = compiler.compile(code)
  except CompilationError as e:
    return False, float('inf'), f"Compilation error: <br><pre>{str(e)[:500]}</pre>", 0

  world = get_world(subpass)
  sim = LanderSimulator(world)

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

    # Send initial data (non-blocking, may drop if buffer fills)
    header_lines = [
      f"{world.width} {world.height} {world.gravity} {world.max_thrust} {world.initial_fuel}\n",
      f"{world.start_x} {world.start_y} {world.target_x} {world.target_y}\n",
      *(line + "\n" for line in world.get_map_string().splitlines()),
    ]
    for line in header_lines:
      try:
        stdin_queue.put(line,timeout=5)
      except Exception:
        dropped_inputs += 1
        log_event("stdin_drop: header")

    max_time = case["max_time"]
    dt = 0.1
    end_reason = "timeout"
    path: List[Tuple[float, float]] = []
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

      # Check real-time timeout
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
        if len(parts) >= 2:
          try:
            thrust = float(parts[0])
            turn = float(parts[1])
            valid_commands += 1
          except Exception:
            thrust = 0.0
            turn = 0.0
            invalid_commands += 1
            log_event(f"parse_error: {latest_line.strip()}")
        else:
          thrust = 0.0
          turn = 0.0
          invalid_commands += 1
          log_event(f"invalid_cmd: {latest_line.strip()}")
      else:
        thrust = 0.0
        turn = 0.0
        no_command_ticks += 1
        log_event("no_cmd")

      sim.step(thrust, turn, dt)
      path.append((sim.x, sim.y))
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
      "debug_log": debug_log,
    }

    return sim.landed, distance, end_reason, exec_time

  except Exception as e:
    return False, float('inf'), f"Execution error: {str(e)}", time.time() - start_time


def gradeAnswer(result: dict, subPass: int, aiEngineName: str) -> tuple:
  """
    Grade the Rust lander controller.
    
    Scoring:
    - 1.0: Successful landing
    - Partial: Based on distance to target
    - 0.0: Crash or no valid output
    """
  if not result:
    return 0.0, "No result provided"

  discussion, code, parse_error = _extract_freeform(result)
  if parse_error:
    return 0.0, parse_error
  if not code:
    return 0.0, "No Rust code provided"

  case = TEST_CASES[subPass]
  description = case["description"]
  t = time.time()
  world = get_world(subPass)
  getWorldTime = time.time() - t
  if getWorldTime > 1:
    print(f"Generating the world took {getWorldTime:.2f}s for subpass {subPass}")
  max_distance = math.sqrt(world.width**2 + world.height**2)

  code = code

  # Run simulation
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

  return score, explanation


def resultToNiceReport(result: dict, subPass: int, aiEngineName: str) -> str:
  if not result:
    return "<p style='color:red'>No result provided</p>"
  case = TEST_CASES[subPass]
  html = f"<h4>Lunar Lander - {case['description']}</h4>"
  discussion, code, _ = _extract_freeform(result)
  if discussion:
    r = discussion[:400] + ('...' if len(discussion) > 400 else '')
    html += f"<p><strong>Approach:</strong> {r.replace('<', '&lt;').replace('>', '&gt;')}</p>"
  if code:
    code_escaped = code.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    html += f"<details><summary>View Rust Code ({len(code)} chars)</summary><pre>{code_escaped}</pre></details>"
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
            <td style="padding: 4px 8px; font-weight: 600;">{stats['final_vx']:.1f}, {stats['final_vy']:.1f}</td>
          </tr>
          <tr>
            <td style="padding: 4px 8px; color: #94a3b8;">Process exited early</td>
            <td style="padding: 4px 8px; font-weight: 600;">{str(stats['process_exited'])}</td>
            <td style="padding: 4px 8px; color: #94a3b8;">End reason</td>
            <td style="padding: 4px 8px; font-weight: 600;">{stats['end_reason']}</td>
          </tr>
        </table>
      </div>
    """
    html += create_lander_svg(stats["world"], stats["path"], stats["landed"], stats["crashed"])
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
  return html


highLevelSummary = """
<p>Land a spacecraft softly on the Moon. The lander starts above the surface with
some velocity, and the AI must control the main thruster and side jets to touch
down gently &mdash; too fast and it crashes, too much fuel wasted and it runs dry.</p>
<p>Subpasses vary the starting altitude, velocity, and terrain. The AI must handle
2D physics, fuel management, and real-time control to achieve a safe landing.</p>
"""
