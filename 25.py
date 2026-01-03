"""
Test 25: Asteroid Interception - N-Body Solar System (C# Implementation)

The LLM must write C# code that calculates a spacecraft trajectory to intercept
and deflect an asteroid on collision course with Earth using kinetic impact.

The simulation uses a simplified N-body model of the solar system with:
- Sun, Earth, Moon, and major planets
- Asteroid on Earth-impact trajectory
- Spacecraft with limited delta-v budget

The solver must output:
1. Launch window and initial trajectory
2. Mid-course correction burns
3. Final approach for kinetic impact

Grading:
- Simulation runs backward to place asteroid
- Runs forward to verify Earth impact trajectory
- Checks if spacecraft intercepts asteroid
- Verifies deflection is sufficient to miss Earth

Subpasses increase difficulty with:
- Longer warning times (years to decades)
- Lighter impactors (less momentum transfer)
- Limited delta-v budgets
- Complex multi-flyby trajectories

Solver times out after 5 minutes.
"""

import random
import subprocess
import sys
import os
import time
import math
from typing import List, Tuple, Dict, Optional
from pathlib import Path

# Import our native compiler helper
from native_compiler import CSharpCompiler, CompilationError, ExecutionError

title = "Asteroid Interception (C#)"

# Timeout in seconds (5 minutes)
TIMEOUT_SECONDS = 30

# Seed for reproducibility
RANDOM_SEED = 25252525

# Physical constants (SI units)
G = 6.67430e-11  # Gravitational constant
AU = 1.496e11  # Astronomical unit in meters
DAY = 86400.0  # Seconds per day
YEAR = 365.25 * DAY

# Simplified solar system data (semi-major axis in AU, mass in kg)
SOLAR_SYSTEM = {
  "Sun": {
    "mass": 1.989e30,
    "sma": 0.0,
    "period": 0
  },
  "Mercury": {
    "mass": 3.301e23,
    "sma": 0.387,
    "period": 87.97 * DAY
  },
  "Venus": {
    "mass": 4.867e24,
    "sma": 0.723,
    "period": 224.7 * DAY
  },
  "Earth": {
    "mass": 5.972e24,
    "sma": 1.0,
    "period": 365.25 * DAY
  },
  "Moon": {
    "mass": 7.342e22,
    "sma": 1.00257,
    "period": 27.3 * DAY
  },  # Relative to Earth
  "Mars": {
    "mass": 6.417e23,
    "sma": 1.524,
    "period": 687.0 * DAY
  },
  "Jupiter": {
    "mass": 1.898e27,
    "sma": 5.203,
    "period": 4333.0 * DAY
  },
  "Saturn": {
    "mass": 5.683e26,
    "sma": 9.537,
    "period": 10759.0 * DAY
  },
}


class Body:
  """A celestial body in the simulation."""

  def __init__(self, name: str, mass: float, x: float, y: float, z: float, vx: float, vy: float,
               vz: float):
    self.name = name
    self.mass = mass
    self.x, self.y, self.z = x, y, z
    self.vx, self.vy, self.vz = vx, vy, vz


class NBodySimulation:
  """Simplified N-body solar system simulation."""

  def __init__(self, seed: int):
    self.rng = random.Random(seed)
    self.bodies: List[Body] = []
    self.time = 0.0
    self._init_solar_system()

  def _init_solar_system(self):
    """Initialize solar system bodies in circular orbits."""
    # Sun at origin
    self.bodies.append(Body("Sun", SOLAR_SYSTEM["Sun"]["mass"], 0, 0, 0, 0, 0, 0))

    # Planets in circular orbits (simplified)
    for name, data in SOLAR_SYSTEM.items():
      if name == "Sun" or name == "Moon":
        continue

      sma = data["sma"] * AU
      mass = data["mass"]

      # Random initial angle
      angle = self.rng.uniform(0, 2 * math.pi)

      # Position
      x = sma * math.cos(angle)
      y = sma * math.sin(angle)
      z = 0

      # Circular orbital velocity
      v_orbital = math.sqrt(G * SOLAR_SYSTEM["Sun"]["mass"] / sma)
      vx = -v_orbital * math.sin(angle)
      vy = v_orbital * math.cos(angle)
      vz = 0

      self.bodies.append(Body(name, mass, x, y, z, vx, vy, vz))

  def get_earth(self) -> Body:
    """Get Earth body."""
    for b in self.bodies:
      if b.name == "Earth":
        return b
    return None

  def add_asteroid(self, x: float, y: float, z: float, vx: float, vy: float, vz: float,
                   mass: float):
    """Add an asteroid to the simulation."""
    self.bodies.append(Body("Asteroid", mass, x, y, z, vx, vy, vz))

  def add_spacecraft(self, x: float, y: float, z: float, vx: float, vy: float, vz: float,
                     mass: float):
    """Add a spacecraft to the simulation."""
    self.bodies.append(Body("Spacecraft", mass, x, y, z, vx, vy, vz))

  def get_body(self, name: str) -> Optional[Body]:
    """Get body by name."""
    for b in self.bodies:
      if b.name == name:
        return b
    return None

  def step(self, dt: float):
    """Advance simulation by dt seconds using Verlet integration."""
    n = len(self.bodies)

    # Calculate accelerations
    ax = [0.0] * n
    ay = [0.0] * n
    az = [0.0] * n

    for i in range(n):
      for j in range(n):
        if i == j:
          continue

        dx = self.bodies[j].x - self.bodies[i].x
        dy = self.bodies[j].y - self.bodies[i].y
        dz = self.bodies[j].z - self.bodies[i].z

        r2 = dx * dx + dy * dy + dz * dz
        r = math.sqrt(r2)

        if r < 1e6:  # Collision threshold
          continue

        # Gravitational acceleration
        a = G * self.bodies[j].mass / r2
        ax[i] += a * dx / r
        ay[i] += a * dy / r
        az[i] += a * dz / r

    # Update velocities and positions
    for i in range(n):
      self.bodies[i].vx += ax[i] * dt
      self.bodies[i].vy += ay[i] * dt
      self.bodies[i].vz += az[i] * dt

      self.bodies[i].x += self.bodies[i].vx * dt
      self.bodies[i].y += self.bodies[i].vy * dt
      self.bodies[i].z += self.bodies[i].vz * dt

    self.time += dt

  def run(self, duration: float, dt: float = 3600.0):
    """Run simulation for duration seconds."""
    steps = int(duration / dt)
    for _ in range(steps):
      self.step(dt)

  def distance(self, body1: str, body2: str) -> float:
    """Calculate distance between two bodies."""
    b1 = self.get_body(body1)
    b2 = self.get_body(body2)
    if not b1 or not b2:
      return float('inf')

    dx = b1.x - b2.x
    dy = b1.y - b2.y
    dz = b1.z - b2.z
    return math.sqrt(dx * dx + dy * dy + dz * dz)


def generate_impact_asteroid(sim: NBodySimulation, warning_time: float,
                             seed: int) -> Tuple[float, float, float, float, float, float]:
  """
    Generate an asteroid on Earth-impact trajectory.
    Returns initial position and velocity.
    """
  rng = random.Random(seed)
  earth = sim.get_earth()

  # Start with Earth's future position
  future_sim = NBodySimulation(seed)
  future_sim.run(warning_time)
  future_earth = future_sim.get_earth()

  # Asteroid approaches from outer solar system
  approach_angle = rng.uniform(0, 2 * math.pi)
  approach_dist = rng.uniform(3, 5) * AU  # From 3-5 AU out

  # Position when detected
  ax = future_earth.x + approach_dist * math.cos(approach_angle)
  ay = future_earth.y + approach_dist * math.sin(approach_angle)
  az = rng.uniform(-0.1, 0.1) * AU  # Slight inclination

  # Velocity toward Earth (simplified)
  speed = rng.uniform(15000, 35000)  # 15-35 km/s
  dx = future_earth.x - ax
  dy = future_earth.y - ay
  dz = future_earth.z - az
  dist = math.sqrt(dx * dx + dy * dy + dz * dz)

  avx = speed * dx / dist
  avy = speed * dy / dist
  avz = speed * dz / dist

  return ax, ay, az, avx, avy, avz


def format_scenario(sim: NBodySimulation, warning_days: float, impactor_mass: float,
                    delta_v_budget: float, asteroid_mass: float) -> str:
  """Format scenario as input string."""
  lines = []

  # Header
  lines.append(
    f"{len(sim.bodies)} {warning_days:.1f} {impactor_mass:.0f} {delta_v_budget:.1f} {asteroid_mass:.2e}"
  )

  # Bodies
  for b in sim.bodies:
    lines.append(
      f"{b.name} {b.mass:.6e} {b.x:.6e} {b.y:.6e} {b.z:.6e} {b.vx:.6e} {b.vy:.6e} {b.vz:.6e}")

  return "\n".join(lines)


# Test configurations
TEST_CASES = [
  # Subpass 0: Short warning, heavy impactor
  {
    "warning_days": 180,
    "impactor_mass": 10000,  # 10 tons
    "delta_v": 5000,  # 5 km/s
    "asteroid_mass": 1e9,  # 1000 tons
    "description": "180 days, 10t impactor, 5km/s dV"
  },
  # Subpass 1: Medium warning
  {
    "warning_days": 365,
    "impactor_mass": 5000,
    "delta_v": 4000,
    "asteroid_mass": 1e9,
    "description": "1 year, 5t impactor, 4km/s dV"
  },
  # Subpass 2: One year, lighter
  {
    "warning_days": 500,
    "impactor_mass": 2000,
    "delta_v": 3500,
    "asteroid_mass": 5e9,
    "description": "500 days, 2t impactor, 3.5km/s dV"
  },
  # Subpass 3: Two years
  {
    "warning_days": 730,
    "impactor_mass": 1000,
    "delta_v": 3000,
    "asteroid_mass": 1e10,
    "description": "2 years, 1t impactor, 3km/s dV"
  },
  # Subpass 4: Three years
  {
    "warning_days": 1095,
    "impactor_mass": 500,
    "delta_v": 2500,
    "asteroid_mass": 5e10,
    "description": "3 years, 500kg impactor, 2.5km/s dV"
  },
  # Subpass 5: Five years
  {
    "warning_days": 1825,
    "impactor_mass": 300,
    "delta_v": 2000,
    "asteroid_mass": 1e11,
    "description": "5 years, 300kg impactor, 2km/s dV"
  },
  # Extreme cases
  {
    "warning_days": 2555,  # 7 years
    "impactor_mass": 200,
    "delta_v": 1500,
    "asteroid_mass": 5e11,
    "description": "7 years, 200kg impactor, 1.5km/s dV"
  },
  {
    "warning_days": 3650,  # 10 years
    "impactor_mass": 100,
    "delta_v": 1000,
    "asteroid_mass": 1e12,
    "description": "10 years, 100kg impactor, 1km/s dV"
  },
  {
    "warning_days": 5475,  # 15 years
    "impactor_mass": 50,
    "delta_v": 800,
    "asteroid_mass": 5e12,
    "description": "15 years, 50kg impactor, 800m/s dV"
  },
  {
    "warning_days": 7300,  # 20 years
    "impactor_mass": 20,
    "delta_v": 500,
    "asteroid_mass": 1e13,
    "description": "20 years, 20kg impactor, 500m/s dV"
  },
  {
    "warning_days": 10950,  # 30 years
    "impactor_mass": 10,
    "delta_v": 300,
    "asteroid_mass": 5e13,
    "description": "30 years, 10kg impactor, 300m/s dV - decade-long intercept"
  },
]

# Cache scenarios
SCENARIO_CACHE = {}


def get_scenario(subpass: int) -> Tuple[NBodySimulation, str]:
  """Get or generate scenario for subpass."""
  if subpass not in SCENARIO_CACHE:
    case = TEST_CASES[subpass]
    sim = NBodySimulation(RANDOM_SEED + subpass)

    # Generate asteroid
    warning_time = case["warning_days"] * DAY
    ax, ay, az, avx, avy, avz = generate_impact_asteroid(sim, warning_time,
                                                         RANDOM_SEED + subpass + 1000)
    sim.add_asteroid(ax, ay, az, avx, avy, avz, case["asteroid_mass"])

    input_str = format_scenario(sim, case["warning_days"], case["impactor_mass"], case["delta_v"],
                                case["asteroid_mass"])

    SCENARIO_CACHE[subpass] = (sim, input_str)

  return SCENARIO_CACHE[subpass]


def prepareSubpassPrompt(subPass: int) -> str:
  """Generate the prompt for subpass 0 that handles all interception complexities."""
  if subPass != 0:
    raise StopIteration

  return f"""You are writing C# code to calculate an asteroid interception trajectory.

You must write a C# solver that can handle ANY interception scenario from trivial to ludicrous scale:
- **Trivial**: Short warning times (1-2 years), large impactors, generous delta-v, simple trajectories
- **Medium**: Moderate warning times (5-10 years), medium impactors, limited delta-v, basic flybys
- **Large**: Long warning times (15-25 years), small impactors, tight delta-v, complex multi-flyby trajectories
- **Extreme**: Very long warning times (30-50 years), very small impactors, very tight delta-v, advanced trajectory optimization

**The Challenge:**
Your C# asteroid interceptor will be tested with scenarios ranging from simple deflection missions to complex multi-flyby interception missions. The same algorithm must work efficiently across ALL mission complexities and constraints.

**Problem:**
Calculate a spacecraft trajectory to intercept and deflect an asteroid on collision course with Earth using kinetic impact. The simulation uses a simplified N-body model of the solar system with Sun, Earth, Moon, and major planets.

**The solver must output:**
1. Launch window and initial trajectory
2. Mid-course correction burns
3. Final approach for kinetic impact

**Input format (stdin):**
```
num_bodies
body_name mass x y z vx vy vz  (for each body)
spacecraft_mass delta_v_budget
warning_time_years
```

**Output format (stdout):**
```
launch_time burn_dv_x burn_dv_y burn_dv_z
correction_time1 burn_dv_x burn_dv_y burn_dv_z
...
impact_time impact_dv_x impact_dv_y impact_dv_z
```

**Critical Requirements:**
1. **Scalability**: Your algorithm must adapt based on warning time and delta-v constraints
2. **Performance**: Must complete within 5 minutes even for complex trajectory optimization
3. **Quality**: Successfully intercept asteroid and deflect it sufficiently to miss Earth

**Algorithm Strategy Recommendations:**
- **Short warning times (≤5 years)**: Direct intercept trajectories, high-thrust burns
- **Medium warning times (5-15 years)**: Basic gravity assists, moderate optimization
- **Long warning times (15-30 years)**: Complex multi-flyby trajectories, advanced optimization
- **Very long warning times (>30 years)**: Advanced trajectory optimization, multiple assists

**Key Techniques:**
- **N-body simulation**: Accurate gravitational modeling
- **Trajectory optimization**: Find optimal intercept trajectory
- **Gravity assists**: Use planetary flybys for delta-v savings
- **Kinetic impact modeling**: Calculate deflection effectiveness
- **Launch window analysis**: Find optimal launch times

**Implementation Hints:**
- Detect mission complexity and choose appropriate trajectory planning method
- Use efficient numerical integration for N-body simulation
- Implement adaptive optimization based on delta-v constraints
- For very complex missions, use advanced trajectory optimization techniques
- Handle edge cases: impossible missions, insufficient delta-v
- Use fast I/O for large trajectory calculations

**Success Criteria:**
- Spacecraft intercepts asteroid before Earth impact
- Kinetic impact deflects asteroid sufficiently to miss Earth
- Total delta-v usage stays within budget
- Launch occurs within specified window

**Failure Criteria:**
- Miss intercept opportunity
- Insufficient deflection
- Exceed delta-v budget
- Launch window missed

**Requirements:**
1. Program must compile with .NET/csc
2. Read from stdin, write to stdout
3. Handle variable warning times and delta-v budgets
4. Complete within 5 minutes
5. Must handle varying mission complexities efficiently

Write complete, compilable C# code with a Main method.
Include adaptive logic that chooses different strategies based on mission complexity.
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
      "Explain your algorithm approach and how it adapts to different interception complexities"
    },
    "csharp_code": {
      "type": "string",
      "description": "Complete C# code with Main method that handles all scales"
    }
  },
  "required": ["reasoning", "csharp_code"],
  "additionalProperties": False
}


def parse_burns(output: str) -> List[Tuple[float, float, float, float]]:
  """Parse burn sequence from output."""
  lines = output.strip().split('\n')
  if not lines:
    return []

  try:
    num_burns = int(lines[0].strip())
    burns = []
    for i in range(1, min(num_burns + 1, len(lines))):
      parts = lines[i].strip().split()
      if len(parts) >= 4:
        day = float(parts[0])
        dvx = float(parts[1])
        dvy = float(parts[2])
        dvz = float(parts[3])
        burns.append((day, dvx, dvy, dvz))
    return burns
  except:
    return []


def simulate_mission(sim: NBodySimulation, burns: List[Tuple[float, float, float, float]],
                     impactor_mass: float, warning_days: float) -> Tuple[bool, float, float]:
  """
    Simulate the interception mission.
    Returns (intercepted, closest_approach, earth_miss_distance).
    """
  # Create fresh simulation
  test_sim = NBodySimulation(sim.rng.randint(0, 1000000))

  # Copy asteroid
  orig_asteroid = sim.get_body("Asteroid")
  if orig_asteroid:
    test_sim.add_asteroid(orig_asteroid.x, orig_asteroid.y, orig_asteroid.z, orig_asteroid.vx,
                          orig_asteroid.vy, orig_asteroid.vz, orig_asteroid.mass)

  # Add spacecraft at Earth
  earth = test_sim.get_earth()
  if earth and burns:
    # First burn is launch
    day, dvx, dvy, dvz = burns[0]
    test_sim.add_spacecraft(earth.x, earth.y, earth.z, earth.vx + dvx, earth.vy + dvy,
                            earth.vz + dvz, impactor_mass)
  else:
    return False, float('inf'), 0

  # Sort remaining burns by day
  remaining_burns = sorted(burns[1:], key=lambda b: b[0])
  burn_idx = 0

  closest_approach = float('inf')
  dt = 3600.0  # 1 hour steps

  # Run simulation
  for day in range(int(warning_days) + 100):
    # Apply any burns scheduled for this day
    while burn_idx < len(remaining_burns) and remaining_burns[burn_idx][0] <= day:
      _, dvx, dvy, dvz = remaining_burns[burn_idx]
      sc = test_sim.get_body("Spacecraft")
      if sc:
        sc.vx += dvx
        sc.vy += dvy
        sc.vz += dvz
      burn_idx += 1

    # Run one day
    test_sim.run(DAY, dt)

    # Check closest approach
    dist = test_sim.distance("Spacecraft", "Asteroid")
    closest_approach = min(closest_approach, dist)

    # Check for intercept (within 1000 km)
    if dist < 1e6:
      # Intercept! Calculate deflection
      sc = test_sim.get_body("Spacecraft")
      ast = test_sim.get_body("Asteroid")
      if sc and ast:
        # Momentum transfer
        rel_vx = sc.vx - ast.vx
        rel_vy = sc.vy - ast.vy
        rel_vz = sc.vz - ast.vz

        # Deflection velocity
        dv = impactor_mass / ast.mass
        ast.vx += dv * rel_vx
        ast.vy += dv * rel_vy
        ast.vz += dv * rel_vz

        # Remove spacecraft
        test_sim.bodies = [b for b in test_sim.bodies if b.name != "Spacecraft"]

  # Check if asteroid misses Earth
  earth = test_sim.get_earth()
  asteroid = test_sim.get_body("Asteroid")
  if earth and asteroid:
    dx = asteroid.x - earth.x
    dy = asteroid.y - earth.y
    dz = asteroid.z - earth.z
    miss_dist = math.sqrt(dx * dx + dy * dy + dz * dz)
  else:
    miss_dist = 0

  intercepted = closest_approach < 1e6
  return intercepted, closest_approach, miss_dist


def run_solver(code: str, input_data: str, engine_name: str) -> Tuple[str, str, float]:
  """Compile and run C# solver."""
  compiler = CSharpCompiler(engine_name)

  if not compiler.find_compiler():
    return "", "No C# compiler found", 0

  try:
    exe_path = compiler.compile(code)
    stdout, stderr, exec_time, retcode = compiler.execute(exe_path, input_data, TIMEOUT_SECONDS)
    if retcode != 0:
      return stdout, f"Runtime error: {stderr[:500]}", exec_time
    return stdout, "", exec_time
  except CompilationError as e:
    return "", f"Compilation error: {str(e)[:500]}", 0
  except ExecutionError as e:
    return "", str(e), TIMEOUT_SECONDS


def gradeAnswer(result: dict, subPass: int, aiEngineName: str) -> tuple:
  """Grade the asteroid interception solver."""
  if not result:
    return 0.0, "No result provided"

  if "csharp_code" not in result:
    return 0.0, "No C# code provided"

  case = TEST_CASES[subPass]
  description = case["description"]
  sim, input_data = get_scenario(subPass)

  code = result["csharp_code"]

  # Run solver
  stdout, error, exec_time = run_solver(code, input_data, aiEngineName)

  if error:
    return 0.0, f"[{description}] {error}"

  # Parse burns
  burns = parse_burns(stdout)
  if not burns:
    return 0.0, f"[{description}] No valid burn sequence"

  # Check delta-V budget
  total_dv = sum(math.sqrt(b[1]**2 + b[2]**2 + b[3]**2) for b in burns)
  if total_dv > case["delta_v"] * 1.1:  # 10% tolerance
    return 0.1, f"[{description}] Delta-V exceeded: {total_dv:.0f} > {case['delta_v']}"

  # Simulate mission
  intercepted, closest, miss_dist = simulate_mission(sim, burns, case["impactor_mass"],
                                                     case["warning_days"])

  earth_radius = 6.371e6

  if intercepted and miss_dist > earth_radius:
    score = 1.0
    status = "SUCCESS - intercepted and deflected"
  elif intercepted:
    score = 0.7
    status = f"Intercepted but insufficient deflection ({miss_dist/1e6:.0f} km)"
  elif closest < 1e8:  # Within 100,000 km
    score = 0.4
    status = f"Close approach {closest/1e6:.0f} km"
  else:
    score = 0.1
    status = f"Missed - closest {closest/1e9:.1f} million km"

  explanation = (f"[{description}] {status}, "
                 f"dV used: {total_dv:.0f}/{case['delta_v']}m/s, "
                 f"Burns: {len(burns)}, Time: {exec_time:.2f}s")

  return score, explanation


def output_example_html(score: float, explanation: str, result: dict, subPass: int) -> str:
  """Generate HTML for result display."""
  case = TEST_CASES[subPass]

  code = result.get("csharp_code", "No code provided")
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
            <summary>C# Code</summary>
            <pre style="background: #f0f0f0; padding: 10px; overflow-x: auto;"><code>{code}</code></pre>
        </details>
    </div>
    """


def output_header_html() -> str:
  return """
    <h2>Test 25: Asteroid Interception (C#)</h2>
    <p>Testing C# implementation of N-body trajectory planning for asteroid deflection.</p>
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
