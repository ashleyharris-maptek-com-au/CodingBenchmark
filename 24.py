"""
Test 24: 3D Lunar Lander Game (C++ Implementation)

The LLM must write C++ code that plays Lunar Lander in 3D space in real-time.
The program receives a 3D voxel map and continuous state updates,
and must output thrust/rotation commands to land safely on the target.

Physics:
- Gravity pulls lander down constantly (negative Z)
- Thrust applies force in lander's facing direction
- Lander can rotate around all 3 axes (pitch, yaw, roll)
- Fuel is limited
- Must land softly on target (low velocity)

Input format (stdin):
Initial map:
  Line 1: width depth height gravity max_thrust fuel
  Line 2: start_x start_y start_z target_x target_y target_z
  Line 3: num_obstacles
  Following lines: obstacle positions (x y z) or "VOXELS" for voxel data
  Then: STATE marker

Then continuous state updates:
  x y z vx vy vz pitch yaw roll pitch_rate yaw_rate roll_rate fuel_remaining

Output format (stdout):
For each state update, immediately output:
  thrust pitch_cmd yaw_cmd roll_cmd
  - thrust: 0.0 to 1.0 (fraction of max thrust)
  - pitch_cmd: -1.0 to 1.0 (pitch rate command)
  - yaw_cmd: -1.0 to 1.0 (yaw rate command)  
  - roll_cmd: -1.0 to 1.0 (roll rate command)

Success: Land on target with |velocity| < threshold
Failure: Crash into obstacle, run out of fuel, or exceed time limit

Subpasses test increasingly large 3D worlds.
Solver times out after 5 minutes.
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
from native_compiler import CppCompiler, CompilationError, ExecutionError

title = "3D Lunar Lander Game (C++)"

# Timeout in seconds (5 minutes)
TIMEOUT_SECONDS = 30

# Seed for reproducibility
RANDOM_SEED = 24242424


class Lander3DWorld:
  """Generates and stores a 3D Lunar Lander world."""

  def __init__(self, width: int, depth: int, height: int, gravity: float, max_thrust: float,
               fuel: float, seed: int):
    self.width = width
    self.depth = depth
    self.height = height
    self.gravity = gravity
    self.max_thrust = max_thrust
    self.initial_fuel = fuel
    self.rng = random.Random(seed)

    # Generate obstacles (spherical regions)
    self.obstacles = []
    self._generate_terrain()

    # Place start and target
    self.start_x = width // 4
    self.start_y = depth // 2
    self.start_z = height - height // 10

    self.target_x = 3 * width // 4
    self.target_y = depth // 2
    self.target_z = self._find_ground_level(self.target_x, self.target_y) + 2

  def _generate_terrain(self):
    """Generate terrain with ground and obstacles."""
    # Ground is at z=0 (implicit)

    # Add floating obstacles
    num_obstacles = max(5, (self.width * self.depth * self.height) // 50000)
    for _ in range(num_obstacles):
      ox = self.rng.uniform(0, self.width)
      oy = self.rng.uniform(0, self.depth)
      oz = self.rng.uniform(self.height * 0.1, self.height * 0.7)
      radius = self.rng.uniform(
        min(self.width, self.depth, self.height) * 0.01,
        min(self.width, self.depth, self.height) * 0.05)
      self.obstacles.append((ox, oy, oz, radius))

  def _find_ground_level(self, x: float, y: float) -> float:
    """Find ground level at position (always 0 for flat terrain)."""
    return 0.0

  def is_obstacle(self, x: float, y: float, z: float) -> bool:
    """Check if position is in an obstacle or out of bounds."""
    # Check bounds
    if x < 0 or x >= self.width or y < 0 or y >= self.depth:
      return True
    if z < 0:  # Ground
      return True
    if z >= self.height:
      return True

    # Check spherical obstacles
    for ox, oy, oz, radius in self.obstacles:
      dist = math.sqrt((x - ox)**2 + (y - oy)**2 + (z - oz)**2)
      if dist < radius:
        return True
    return False

  def is_landing_zone(self, x: float, y: float, z: float) -> bool:
    """Check if position is near target landing zone."""
    landing_radius = max(5, min(self.width, self.depth) * 0.02)
    dist = math.sqrt((x - self.target_x)**2 + (y - self.target_y)**2)
    return dist < landing_radius and z <= self.target_z + 2


class Lander3DSimulator:
  """Simulates 3D lander physics."""

  def __init__(self, world: Lander3DWorld):
    self.world = world
    self.x = float(world.start_x)
    self.y = float(world.start_y)
    self.z = float(world.start_z)
    self.vx = 0.0
    self.vy = 0.0
    self.vz = 0.0

    # Euler angles (radians)
    self.pitch = 0.0  # Nose up/down
    self.yaw = 0.0  # Left/right heading
    self.roll = 0.0  # Roll around forward axis

    # Angular velocities
    self.pitch_rate = 0.0
    self.yaw_rate = 0.0
    self.roll_rate = 0.0

    self.fuel = world.initial_fuel
    self.crashed = False
    self.landed = False
    self.time = 0.0
    self.max_landing_velocity = 3.0

  def step(self, thrust: float, pitch_cmd: float, yaw_cmd: float, roll_cmd: float, dt: float = 0.1):
    """Simulate one time step."""
    if self.crashed or self.landed:
      return

    # Clamp inputs
    thrust = max(0.0, min(1.0, thrust))
    pitch_cmd = max(-1.0, min(1.0, pitch_cmd))
    yaw_cmd = max(-1.0, min(1.0, yaw_cmd))
    roll_cmd = max(-1.0, min(1.0, roll_cmd))

    # Apply rotation commands
    rot_accel = 2.0  # Rotation acceleration factor
    self.pitch_rate += pitch_cmd * rot_accel * dt
    self.yaw_rate += yaw_cmd * rot_accel * dt
    self.roll_rate += roll_cmd * rot_accel * dt

    # Damping
    damping = 0.95
    self.pitch_rate *= damping
    self.yaw_rate *= damping
    self.roll_rate *= damping

    # Update angles
    self.pitch += self.pitch_rate * dt
    self.yaw += self.yaw_rate * dt
    self.roll += self.roll_rate * dt

    # Apply thrust if fuel available
    if self.fuel > 0 and thrust > 0:
      thrust_force = thrust * self.world.max_thrust
      fuel_used = thrust * dt * 0.1
      self.fuel = max(0, self.fuel - fuel_used)

      # Thrust direction based on orientation
      # Simplified: thrust is along lander's "up" which we compute from Euler angles
      # For simplicity, just use pitch and yaw
      ax = thrust_force * math.sin(self.yaw) * math.cos(self.pitch)
      ay = thrust_force * math.cos(self.yaw) * math.cos(self.pitch)
      az = thrust_force * math.sin(self.pitch)

      self.vx += ax * dt
      self.vy += ay * dt
      self.vz += az * dt

    # Apply gravity (negative Z)
    self.vz -= self.world.gravity * dt

    # Update position
    self.x += self.vx * dt
    self.y += self.vy * dt
    self.z += self.vz * dt

    self.time += dt

    # Check collision
    if self.world.is_obstacle(self.x, self.y, self.z):
      velocity = math.sqrt(self.vx**2 + self.vy**2 + self.vz**2)
      if self.world.is_landing_zone(self.x, self.y,
                                    self.z) and velocity < self.max_landing_velocity:
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
    """Calculate distance to target."""
    dx = self.x - self.world.target_x
    dy = self.y - self.world.target_y
    dz = self.z - self.world.target_z
    return math.sqrt(dx * dx + dy * dy + dz * dz)


# Test configurations
TEST_CASES = [
  # Subpass 0: Tiny world
  {
    "width": 100,
    "depth": 100,
    "height": 50,
    "gravity": 1.62,
    "max_thrust": 5.0,
    "fuel": 50.0,
    "max_time": 30.0,
    "description": "100x100x50 - simple 3D landing"
  },
  # Subpass 1: Small world
  {
    "width": 200,
    "depth": 200,
    "height": 100,
    "gravity": 1.62,
    "max_thrust": 6.0,
    "fuel": 100.0,
    "max_time": 45.0,
    "description": "200x200x100 - short 3D hop"
  },
  # Subpass 2: Medium world
  {
    "width": 500,
    "depth": 500,
    "height": 200,
    "gravity": 1.62,
    "max_thrust": 8.0,
    "fuel": 250.0,
    "max_time": 60.0,
    "description": "500x500x200 - 3D maneuvering"
  },
  # Subpass 3: Larger world
  {
    "width": 1000,
    "depth": 1000,
    "height": 400,
    "gravity": 1.62,
    "max_thrust": 12.0,
    "fuel": 500.0,
    "max_time": 90.0,
    "description": "1km³ - cross-crater 3D"
  },
  # Subpass 4: Large world
  {
    "width": 2000,
    "depth": 2000,
    "height": 1000,
    "gravity": 1.62,
    "max_thrust": 18.0,
    "fuel": 1000.0,
    "max_time": 120.0,
    "description": "2km x 2km x 1km - regional 3D"
  },
  # Subpass 5: Very large
  {
    "width": 5000,
    "depth": 5000,
    "height": 2000,
    "gravity": 1.62,
    "max_thrust": 25.0,
    "fuel": 3000.0,
    "max_time": 180.0,
    "description": "5km³ - long range 3D"
  },
  # Extreme cases
  {
    "width": 10000,
    "depth": 10000,
    "height": 5000,
    "gravity": 1.62,
    "max_thrust": 40.0,
    "fuel": 8000.0,
    "max_time": 240.0,
    "description": "10km³ - continental 3D"
  },
  {
    "width": 25000,
    "depth": 25000,
    "height": 10000,
    "gravity": 1.62,
    "max_thrust": 60.0,
    "fuel": 20000.0,
    "max_time": 300.0,
    "description": "25km³ - suborbital 3D"
  },
  {
    "width": 50000,
    "depth": 50000,
    "height": 25000,
    "gravity": 1.62,
    "max_thrust": 100.0,
    "fuel": 50000.0,
    "max_time": 300.0,
    "description": "50km³ - orbital 3D"
  },
  {
    "width": 100000,
    "depth": 100000,
    "height": 50000,
    "gravity": 1.62,
    "max_thrust": 150.0,
    "fuel": 120000.0,
    "max_time": 300.0,
    "description": "100km³ - trans-lunar 3D"
  },
  {
    "width": 500000,
    "depth": 500000,
    "height": 200000,
    "gravity": 1.62,
    "max_thrust": 300.0,
    "fuel": 600000.0,
    "max_time": 300.0,
    "description": "500km³ - interplanetary 3D"
  },
]

# Cache worlds
WORLD_CACHE = {}


def get_world(subpass: int) -> Lander3DWorld:
  """Get or generate world for subpass."""
  if subpass not in WORLD_CACHE:
    case = TEST_CASES[subpass]
    WORLD_CACHE[subpass] = Lander3DWorld(case["width"], case["depth"], case["height"],
                                         case["gravity"], case["max_thrust"], case["fuel"],
                                         RANDOM_SEED + subpass)
  return WORLD_CACHE[subpass]


def format_input(world: Lander3DWorld) -> str:
  """Format world as input string."""
  lines = []
  lines.append(
    f"{world.width} {world.depth} {world.height} {world.gravity} {world.max_thrust} {world.initial_fuel}"
  )
  lines.append(
    f"{world.start_x} {world.start_y} {world.start_z} {world.target_x} {world.target_y} {world.target_z}"
  )
  lines.append(str(len(world.obstacles)))
  for ox, oy, oz, radius in world.obstacles:
    lines.append(f"{ox:.2f} {oy:.2f} {oz:.2f} {radius:.2f}")
  lines.append("STATE")
  return "\n".join(lines)


def prepareSubpassPrompt(subPass: int) -> str:
  """Generate the prompt for subpass 0 that handles all landing complexities."""
  if subPass != 0:
    raise StopIteration

  return f"""You are writing C++ code to play 3D Lunar Lander in real-time.

You must write a C++ solver that can handle ANY landing scenario from trivial to ludicrous scale:
- **Trivial**: Small 3D maps (20x20x20), clear paths, generous fuel, basic physics
- **Medium**: Complex 3D maps (50x50x50), obstacles, moderate fuel, realistic physics
- **Large**: Very complex 3D maps (100x100x100), many obstacles, tight fuel constraints
- **Extreme**: Massive 3D maps (200x200x200+), dense obstacles, very tight fuel, precision required

**The Challenge:**
Your C++ 3D lunar lander controller will be tested with scenarios ranging from simple training environments to very complex landing sites. The same algorithm must work efficiently across ALL map complexities and fuel constraints.

**Problem:**
Control a lunar lander in full 3D space to safely land on a target pad. The program receives a 3D voxel map and continuous state updates, and must output thrust/rotation commands to land safely on the target.

**Physics:**
- Gravity pulls lander down constantly (negative Z direction)
- Thrust applies force in lander's facing direction
- Lander can rotate around all 3 axes (pitch, yaw, roll)
- Fuel is limited
- Must land softly on target (low velocity)

**Input format (stdin):**
Initial map:
```
width depth height gravity max_thrust fuel
start_x start_y start_z target_x target_y target_z
num_obstacles
obstacle positions (x y z) or "VOXELS" for voxel data
STATE
```

Then continuous state updates:
```
x y z vx vy vz pitch yaw roll pitch_rate yaw_rate roll_rate fuel_remaining
```

**Output format (stdout):**
For each state update, immediately output:
```
thrust pitch_cmd yaw_cmd roll_cmd
```
- thrust: 0.0 to 1.0 (fraction of max thrust)
- pitch_cmd: -1.0 to 1.0 (pitch rate command)
- yaw_cmd: -1.0 to 1.0 (yaw rate command)
- roll_cmd: -1.0 to 1.0 (roll rate command)

**Critical Requirements:**
1. **Scalability**: Your algorithm must adapt based on 3D map size and complexity
2. **Performance**: Must make decisions in real-time even for large 3D maps
3. **Quality**: Land safely on target while conserving fuel

**Algorithm Strategy Recommendations:**
- **Small maps (≤30x30x30)**: Can use 3D pathfinding, precise trajectory planning
- **Medium maps (30x30x30-80x80x80)**: Heuristic navigation, reactive control
- **Large maps (80x80x80-150x150x150)**: Fast heuristics, simplified physics models
- **Very Large maps (>150x150x150)**: Very fast heuristics, basic control strategies

**Key Techniques:**
- **3D Trajectory planning**: Calculate optimal path to target in 3D space
- **Fuel management**: Balance thrust usage vs fuel conservation
- **Obstacle avoidance**: Navigate around 3D terrain features
- **Real-time control**: React quickly to state updates
- **3D Physics simulation**: Predict lander movement under thrust/gravity
- **Quaternion math**: Handle 3D rotations efficiently

**Implementation Hints:**
- Detect 3D map complexity and choose appropriate algorithm
- Use efficient 3D pathfinding and collision detection
- Implement adaptive control based on fuel remaining
- For very large maps, focus on reactive control
- Handle edge cases: fuel exhaustion, impossible landings
- Use fast I/O for real-time interaction
- Consider using quaternions for 3D rotations

**Success Conditions:**
- Land on target with |velocity| < threshold in all dimensions
- Stay within landing zone boundaries
- Complete landing within time limit

**Failure Conditions:**
- Crash into obstacle or ground too hard
- Run out of fuel
- Exceed time limit

**Requirements:**
1. Program must compile with g++ or MSVC (C++17)
2. Read from stdin, write to stdout
3. Handle variable 3D map sizes and fuel constraints
4. Complete within 5 minutes
5. Must handle varying 3D map complexities efficiently

Write complete, compilable C++ code with a main() function.
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
  """
    Compile and run 3D lander simulation.
    
    Returns:
        Tuple of (landed, distance_to_target, end_reason, exec_time)
    """
  compiler = CppCompiler(engine_name)

  if not compiler.find_compiler():
    return False, float('inf'), "No C++ compiler found", 0

  try:
    exe_path = compiler.compile(code)
  except CompilationError as e:
    return False, float('inf'), f"Compilation error: {str(e)[:500]}", 0

  world = get_world(subpass)
  sim = Lander3DSimulator(world)
  input_data = format_input(world)

  start_time = time.time()

  try:
    process = subprocess.Popen([str(exe_path)],
                               stdin=subprocess.PIPE,
                               stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE,
                               text=True,
                               bufsize=1)

    # Send initial data
    process.stdin.write(input_data + "\n")
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

      # Read command
      try:
        line = process.stdout.readline()
        if line:
          parts = line.strip().split()
          if len(parts) >= 4:
            thrust = float(parts[0])
            pitch_cmd = float(parts[1])
            yaw_cmd = float(parts[2])
            roll_cmd = float(parts[3])
            sim.step(thrust, pitch_cmd, yaw_cmd, roll_cmd, dt)
          else:
            sim.step(0, 0, 0, 0, dt)
        else:
          sim.step(0, 0, 0, 0, dt)
      except Exception as e:
        sim.step(0, 0, 0, 0, dt)

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
  """Grade the C++ 3D lander controller."""
  if not result:
    return 0.0, "No result provided"

  if "cpp_code" not in result:
    return 0.0, "No C++ code provided"

  case = TEST_CASES[subPass]
  description = case["description"]
  world = get_world(subPass)
  max_distance = math.sqrt(world.width**2 + world.depth**2 + world.height**2)

  code = result["cpp_code"]

  landed, distance, end_reason, exec_time = run_lander_simulation(code, case, subPass, aiEngineName)

  if landed:
    score = 1.0
  elif end_reason == "crashed":
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

  code = result.get("cpp_code", "No code provided")
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
            <summary>C++ Code</summary>
            <pre style="background: #f0f0f0; padding: 10px; overflow-x: auto;"><code>{code}</code></pre>
        </details>
    </div>
    """


def output_header_html() -> str:
  return """
    <h2>Test 24: 3D Lunar Lander Game (C++)</h2>
    <p>Testing C++ implementation of real-time 3D lunar lander control.</p>
    """


def output_summary_html(results: list) -> str:
  if not results:
    return "<p>No results</p>"

  total_score = sum(r[0] for r in results)
  max_score = len(results)
  avg_score = total_score / max_score if max_score > 0 else 0

  return f"""
    <div class="summary" style="margin: 10px; padding: 15px; background: #e8f4e8; border-radius: 5px;">
        <h3>Summary</h3>
        <p><strong>Total Score:</strong> {total_score:.2f} / {max_score}</p>
        <p><strong>Average Score:</strong> {avg_score:.2%}</p>
        <p><strong>Subpasses Completed:</strong> {len(results)}</p>
    </div>
    """
