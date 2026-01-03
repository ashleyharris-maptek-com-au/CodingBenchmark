"""
Test 23: Lunar Lander Game (Rust Implementation)

The LLM must write Rust code that plays Lunar Lander in real-time.
The program receives a 2D voxel map and continuous state updates,
and must output thrust/turn commands to land safely on the target.

Physics:
- Gravity pulls lander down constantly
- Thrust applies force in lander's facing direction
- Fuel is limited
- Must land softly on target (low velocity)

Input format (stdin):
Initial map:
  Line 1: width height gravity max_thrust fuel
  Line 2: start_x start_y target_x target_y
  Lines 3+: map rows (. = empty, # = obstacle, S = start, T = target, L = landing zone)

Then continuous state updates:
  x y vx vy angle angular_vel fuel_remaining
  
Output format (stdout):
For each state update, immediately output:
  thrust turn
  - thrust: 0.0 to 1.0 (fraction of max thrust)
  - turn: -1.0 to 1.0 (turning rate, negative = left, positive = right)

Success: Land on target with |velocity| < threshold
Failure: Crash into obstacle, run out of fuel, or exceed time limit
"""

import random
import subprocess
import sys
import os
import time
import math
from typing import List, Tuple, Set, Dict, Optional
from pathlib import Path

# Import our native compiler helper
from native_compiler import RustCompiler, CompilationError, ExecutionError

title = "Lunar Lander Game"

# Timeout in seconds (5 minutes)
TIMEOUT_SECONDS = 30

# Seed for reproducibility
RANDOM_SEED = 232323


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
    self.start_y = height // 4
    self.target_x = 3 * width // 4
    self.target_y = height // 2

    # Create map
    self.map = [['.' for _ in range(width)] for _ in range(height)]

    # Add landing zone
    self.map[self.target_y][self.target_x] = 'T'
    self.map[self.target_y][self.target_x + 1] = 'L'
    self.map[self.target_y][self.target_x - 1] = 'L'

    # Add some obstacles
    self._add_obstacles()

  def _add_obstacles(self):
    """Add random obstacles to the map."""
    rng = random.Random(RANDOM_SEED)

    for _ in range(self.width * self.height // 50):
      x = rng.randint(0, self.width - 1)
      y = rng.randint(0, self.height - 1)

      # Don't place obstacles on start, target, or landing zone
      if (x, y) not in [(self.start_x, self.start_y), (self.target_x, self.target_y),
                        (self.target_x + 1, self.target_y), (self.target_x - 1, self.target_y)]:
        self.map[y][x] = '#'

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

You must write a Rust solver that can handle ANY landing scenario from trivial to ludicrous scale:
- **Trivial**: Simple maps (20x20), clear paths, generous fuel, basic physics
- **Medium**: Complex maps (50x50), obstacles, moderate fuel, realistic physics
- **Large**: Very complex maps (100x100), many obstacles, tight fuel constraints
- **Extreme**: Massive maps (200x200+), dense obstacles, very tight fuel, precision required

**The Challenge:**
Your Rust lunar lander controller will be tested with scenarios ranging from simple training environments to very complex landing sites. The same algorithm must work efficiently across ALL map complexities and fuel constraints.

**Problem:**
Control a lunar lander to safely land on a target pad. The program receives a 2D voxel map and continuous state updates, and must output thrust/turn commands to land safely on the target.

**Physics:**
- Gravity pulls lander down constantly
- Thrust applies force in lander's facing direction
- Fuel is limited
- Must land softly on target (low velocity)

**Input format (stdin):**
Initial map:
```
width height gravity max_thrust fuel
start_x start_y target_x target_y
map rows (. = empty, # = obstacle, S = start, T = target, L = landing zone)
```

Then continuous state updates:
```
x y vx vy angle angular_vel fuel_remaining
```

**Output format (stdout):**
For each state update, immediately output:
```
thrust turn
```
- thrust: 0.0 to 1.0 (fraction of max thrust)
- turn: -1.0 to 1.0 (turning rate, negative = left, positive = right)

**Critical Requirements:**
1. **Scalability**: Your algorithm must adapt based on map size and complexity
2. **Performance**: Must make decisions in real-time even for large maps
3. **Quality**: Land safely on target while conserving fuel

**Algorithm Strategy Recommendations:**
- **Simple maps (≤30x30)**: Can use pathfinding, precise trajectory planning
- **Medium maps (30x30-80x80)**: Heuristic navigation, reactive control
- **Large maps (80x80-150x150)**: Fast heuristics, simplified physics models
- **Very Large maps (>150x150)**: Very fast heuristics, basic control strategies

**Key Techniques:**
- **Trajectory planning**: Calculate optimal path to target
- **Fuel management**: Balance thrust usage vs fuel conservation
- **Obstacle avoidance**: Navigate around terrain features
- **Real-time control**: React quickly to state updates
- **Physics simulation**: Predict lander movement under thrust/gravity

**Implementation Hints:**
- Detect map complexity and choose appropriate algorithm
- Use efficient pathfinding and collision detection
- Implement adaptive control based on fuel remaining
- For very large maps, focus on reactive control
- Handle edge cases: fuel exhaustion, impossible landings
- Use fast I/O for real-time interaction

**Success Conditions:**
- Land on target with |velocity| < threshold
- Stay within landing zone boundaries
- Complete landing within time limit

**Failure Conditions:**
- Crash into obstacle or ground too hard
- Run out of fuel
- Exceed time limit

**Requirements:**
1. Program must compile with rustc (edition 2021)
2. Read from stdin, write to stdout
3. Handle variable map sizes and fuel constraints
4. Complete within 5 minutes
5. Must handle varying map complexities efficiently

Write complete, compilable Rust code with a main function.
Include adaptive logic that chooses different strategies based on landing complexity.
"""


# List of subpasses to grade the single answer against all difficulty levels
extraGradeAnswerRuns = list(range(len(TEST_CASES)))

structure = {
  "type": "object",
  "properties": {
    "reasoning": {
      "type":
      "string",
      "description":
      "Explain your algorithm approach and how it adapts to different landing complexities"
    },
    "rust_code": {
      "type": "string",
      "description": "Complete Rust code with main function that handles all scales"
    }
  },
  "required": ["reasoning", "rust_code"],
  "additionalProperties": False
}


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
    return False, float('inf'), f"Compilation error: {str(e)[:500]}", 0

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

    # Send initial data
    header = f"{world.width} {world.height} {world.gravity} {world.max_thrust} {world.initial_fuel}\n"
    header += f"{world.start_x} {world.start_y} {world.target_x} {world.target_y}\n"
    header += world.get_map_string() + "\n"
    header += "STATE\n"

    process.stdin.write(header)
    process.stdin.flush()

    max_time = case["max_time"]
    dt = 0.1
    end_reason = "timeout"

    while sim.time < max_time:
      if sim.crashed:
        end_reason = "crashed"
        break
      if sim.landed:
        end_reason = "landed"
        break

      # Check real-time timeout
      if time.time() - start_time > TIMEOUT_SECONDS:
        end_reason = "real_timeout"
        break

      # Send state
      state_line = sim.get_state_string() + "\n"
      try:
        process.stdin.write(state_line)
        process.stdin.flush()
      except:
        end_reason = "pipe_error"
        break

      # Read command (with short timeout)
      try:
        # Non-blocking read attempt
        import select
        if sys.platform != 'win32':
          ready, _, _ = select.select([process.stdout], [], [], 0.1)
          if ready:
            line = process.stdout.readline()
          else:
            line = ""
        else:
          line = process.stdout.readline()

        if line:
          parts = line.strip().split()
          if len(parts) >= 2:
            thrust = float(parts[0])
            turn = float(parts[1])
            sim.step(thrust, turn, dt)
          else:
            sim.step(0, 0, dt)
        else:
          sim.step(0, 0, dt)

      except Exception as e:
        sim.step(0, 0, dt)

    # Send END signal
    try:
      process.stdin.write("END\n")
      process.stdin.flush()
    except:
      pass

    process.terminate()
    try:
      process.wait(timeout=2)
    except:
      process.kill()

    exec_time = time.time() - start_time
    distance = sim.distance_to_target()

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

  if "rust_code" not in result:
    return 0.0, "No Rust code provided"

  case = TEST_CASES[subPass]
  description = case["description"]
  world = get_world(subPass)
  max_distance = math.sqrt(world.width**2 + world.height**2)

  code = result["rust_code"]

  # Run simulation
  landed, distance, end_reason, exec_time = run_lander_simulation(code, case, subPass, aiEngineName)

  if landed:
    score = 1.0
  elif end_reason == "crashed":
    # Partial credit based on how close
    score = max(0.1, 0.5 * (1 - distance / max_distance))
  elif end_reason in ("timeout", "real_timeout"):
    score = max(0.05, 0.3 * (1 - distance / max_distance))
  else:
    score = 0.0

  explanation = (f"[{description}] End: {end_reason}, "
                 f"Distance: {distance:.1f}m, "
                 f"Time: {exec_time:.2f}s")

  return score, explanation


def output_example_html(score: float, explanation: str, result: dict, subPass: int) -> str:
  """Generate HTML for result display."""
  case = TEST_CASES[subPass]

  code = result.get("rust_code", "No code provided")
  reasoning = result.get("reasoning", "No reasoning provided")

  code = code.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
  reasoning = reasoning.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

  score_color = "green" if score >= 0.8 else "orange" if score >= 0.4 else "red"

  return f"""
    <div class="result" style="margin: 10px; padding: 10px; border: 1px solid #ccc;">
        <h4>Subpass {subPass}: {case['description']}</h4>
        <p><strong>Score:</strong> <span style="color: {score_color};">{score:.2f}</span></p>
        <p><strong>Details:</strong> {explanation}</p>
        <details>
            <summary>Reasoning</summary>
            <pre style="background: #f5f5f5; padding: 10px; overflow-x: auto;">{reasoning}</pre>
        </details>
        <details>
            <summary>Rust Code</summary>
            <pre style="background: #f0f0f0; padding: 10px; overflow-x: auto;"><code>{code}</code></pre>
        </details>
    </div>
    """


def output_header_html() -> str:
  """Generate HTML header."""
  return """
    <h2>Test 23: Lunar Lander Game (Rust)</h2>
    <p>Testing Rust implementation of real-time lunar lander control.</p>
    """


def output_summary_html(results: list) -> str:
  """Generate HTML summary."""
  total_score = sum(r[0] for r in results)
  avg_score = total_score / len(results) if results else 0

  return f"""
    <div class="summary" style="margin: 20px; padding: 15px; border: 2px solid #333; background: #f9f9f9;">
        <h3>Summary</h3>
        <p><strong>Average Score:</strong> {avg_score:.3f}</p>
        <p><strong>Total Score:</strong> {total_score:.3f}</p>
        <p><strong>Tests Completed:</strong> {len(results)}/{len(TEST_CASES)}</p>
    </div>
    """
