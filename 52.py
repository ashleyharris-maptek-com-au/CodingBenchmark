"""
Test 52: Self-Driving Car Highway Autopilot (Python)

The LLM must write a Python autopilot for a car on a multi-lane freeway.
It receives ~150 sensor properties including redundant wheel speeds, gyros,
accelerometers, laser range finders, and black-box image parser results.

Tested across 25 scenarios: normal driving, sensor failures, sudden obstacles,
loss of traction (ice/aquaplaning), flickering sensors, tire blowouts, etc.
"""

import math
import traceback

from car_sim import (
  make_car_truth, CarSensorModel, CarSimRunner, RoadModel, Obstacle,
  CAR, DEG2RAD, RAD2DEG, car_physics_step, list_car_state_keys,
)

title = "Self-Driving Car Highway Autopilot (Python)"
TIMEOUT_SECONDS = 90

# ──────────────────────────────────────────────────────────────────────────────
# Scenarios
# ──────────────────────────────────────────────────────────────────────────────
def _sc(name, desc, **kw):
  return {'name': name, 'description': desc, **kw}

SCENARIOS = [
  # 0: Normal cruise
  _sc('Normal highway cruise',
      'Maintain lane and speed on straight dry freeway at 110 km/h.',
      speed=30.6, lane=1),

  # 1: Gentle curve
  _sc('Gentle curve',
      'Follow a gentle left curve while maintaining lane.',
      speed=30.6, lane=1, curvature=0.002),

  # 2: Slow vehicle ahead
  _sc('Slow vehicle ahead',
      'Vehicle ahead doing 70 km/h. Maintain safe following distance or change lanes.',
      speed=30.6, lane=1,
      obstacles=[{'x': 80, 'y': 5.55, 'vx': 19.4}]),

  # 3: Stopped vehicle ahead
  _sc('Emergency stop — stopped vehicle',
      'Vehicle stopped 120m ahead in your lane. Emergency braking required.',
      speed=30.6, lane=1,
      obstacles=[{'x': 120, 'y': 5.55, 'vx': 0}]),

  # 4: Cut-in vehicle
  _sc('Cut-in from adjacent lane',
      'Vehicle merges into your lane 40m ahead at slightly lower speed.',
      speed=30.6, lane=1,
      obstacles=[{'x': 40, 'y': 7.4, 'vx': 27, 'vy': -1.5}]),

  # 5: Ice patch — low friction
  _sc('Ice patch — reduced traction',
      'Sudden ice patch (μ=0.2). Maintain control without spinning.',
      speed=30.6, lane=1, mu=0.2),

  # 6: Aquaplaning
  _sc('Aquaplaning — very low friction',
      'Heavy rain, aquaplaning (μ=0.15). Reduce speed, maintain lane.',
      speed=30.6, lane=1, mu=0.15),

  # 7: Single wheel speed sensor failure
  _sc('Wheel speed sensor failure (FL)',
      'Front-left wheel speed sensor fails (reads 0). Other 3 normal.',
      speed=30.6, lane=1,
      failures={'wheel_speed_fl': ('dead', 0)}),

  # 8: Two wheel speed sensors fail
  _sc('Two wheel speed sensors fail',
      'FL and RR wheel speed sensors fail. Use remaining + other sensors.',
      speed=30.6, lane=1,
      failures={'wheel_speed_fl': ('dead', 0), 'wheel_speed_rr': ('dead', 0)}),

  # 9: Gyro failure
  _sc('Gyro sensor failure',
      'Gyro 1 stuck, gyro 2 has large bias. Gyro 3 normal.',
      speed=30.6, lane=1,
      failures={'gyro_1': ('stuck', 0), 'gyro_2': ('bias', 5.0)}),

  # 10: GPS denied
  _sc('GPS signal lost',
      'Both GPS receivers fail. Navigate using other sensors.',
      speed=30.6, lane=1,
      failures={'gps_1_x': ('dead', 0), 'gps_2_x': ('dead', 0)}),

  # 11: Vision system partial failure
  _sc('Vision system 1 failure',
      'Camera/vision system 1 outputs garbage. Systems 2 and 3 normal.',
      speed=30.6, lane=1,
      failures={
        'vision_1_left_line': ('noise', 2.0),
        'vision_1_right_line': ('noise', 2.0),
        'vision_1_speed': ('dead', 0),
      }),

  # 12: All vision flickering
  _sc('All vision systems flickering',
      'All 3 vision systems have intermittent dropouts (15% of readings invalid).',
      speed=30.6, lane=1, flicker_vision=True),

  # 13: Lidar partial failure
  _sc('Front lidar beams fail',
      'Forward-facing lidar beams (-5° and +5°) fail. Side beams work.',
      speed=30.6, lane=1,
      failures={'lidar_-5': ('dead', 0), 'lidar_5': ('dead', 0)}),

  # 14: Steering sensor disagree
  _sc('Steering angle sensors disagree',
      'Steering sensor 1 reads 10° offset. Sensor 2 normal.',
      speed=30.6, lane=1,
      failures={'steer_1': ('bias', 10.0)}),

  # 15: Brake failure
  _sc('Total brake failure',
      'Brakes fail completely. Must use engine braking and steering to slow down safely.',
      speed=30.6, lane=1, brake_fail=True),

  # 16: Throttle stuck
  _sc('Throttle stuck at 30%',
      'Throttle jams at 30%. Cannot accelerate or coast. Use brakes to control speed.',
      speed=30.6, lane=1, throttle_stuck=0.3),

  # 17: Sudden obstacle (debris)
  _sc('Road debris',
      'Large debris appears 60m ahead. Must brake and/or swerve.',
      speed=30.6, lane=1,
      obstacles=[{'x': 60, 'y': 5.55, 'vx': 0, 'label': 'debris', 'width': 1.5, 'length': 1.0}]),

  # 18: Multi-vehicle traffic
  _sc('Dense traffic',
      'Vehicles in all lanes at varying speeds. Maintain safe following.',
      speed=30.6, lane=1,
      obstacles=[
        {'x': 60, 'y': 1.85, 'vx': 28},    # left lane
        {'x': 50, 'y': 5.55, 'vx': 25},    # center lane (your lane)
        {'x': 70, 'y': 9.25, 'vx': 30},    # right lane
      ]),

  # 19: Lane departure — wind gust
  _sc('Crosswind gust',
      'Strong crosswind pushes car laterally. Maintain lane.',
      speed=30.6, lane=1, crosswind=True),

  # 20: Steering jam
  _sc('Steering partially jammed',
      'Steering jams at 2° right. Must compensate with available range.',
      speed=30.6, lane=1, steer_jam=2 * DEG2RAD),

  # 21: Combined: ice + obstacle
  _sc('Ice + stopped vehicle',
      'Low friction surface with stopped vehicle 100m ahead.',
      speed=30.6, lane=1, mu=0.3,
      obstacles=[{'x': 100, 'y': 5.55, 'vx': 0}]),

  # 22: Sensor flickering + curve
  _sc('Flickering sensors in curve',
      'Gentle curve with intermittent sensor dropouts.',
      speed=30.6, lane=1, curvature=0.003, flicker_all=True),

  # 23: Night / reduced vision
  _sc('Reduced visibility',
      'Night driving: vision systems have increased noise, range reduced.',
      speed=30.6, lane=1,
      failures={
        'vision_1_left_line': ('noise', 0.3),
        'vision_2_left_line': ('noise', 0.3),
        'vision_3_left_line': ('noise', 0.3),
        'vision_1_right_line': ('noise', 0.3),
        'vision_2_right_line': ('noise', 0.3),
        'vision_3_right_line': ('noise', 0.3),
      }),

  # 24: Combined chaos
  _sc('Multi-failure + obstacle',
      'Gyro 1 dead, wheel speed FL dead, ice (μ=0.4), vehicle ahead doing 60 km/h.',
      speed=30.6, lane=1, mu=0.4,
      failures={'gyro_1': ('dead', 0), 'wheel_speed_fl': ('dead', 0)},
      obstacles=[{'x': 80, 'y': 5.55, 'vx': 16.7}]),
]

# ──────────────────────────────────────────────────────────────────────────────
# Prompt
# ──────────────────────────────────────────────────────────────────────────────
def prepareSubpassPrompt(subPass):
  if subPass != 0:
    raise StopIteration

  state_keys = list_car_state_keys()
  num_keys = len(state_keys)

  sensor_groups = {
    'Wheel Speed (x4)': [k for k in state_keys if k.startswith('wheel_speed_')],
    'Gyroscopes (x3)': [k for k in state_keys if k.startswith('gyro_')],
    'Accelerometers (x3)': [k for k in state_keys if k.startswith('accel_')],
    'GPS (x2)': [k for k in state_keys if k.startswith('gps_')],
    'Steering Angle (x2)': [k for k in state_keys if k.startswith('steering_')],
    'Brake Pressure (x2)': [k for k in state_keys if k.startswith('brake_')],
    'Vehicle Speed (x2)': [k for k in state_keys if k.startswith('vehicle_speed')],
    'Lidar Beams (12 angles)': [k for k in state_keys if k.startswith('lidar_')],
    'Vision Systems (x3)': [k for k in state_keys if k.startswith('vision_')],
    'Traction/Surface': [k for k in state_keys if k in ['estimated_mu', 'rain_sensor', 'temperature_c']],
    'Targets': [k for k in state_keys if k.startswith('target_') or k.startswith('current_')],
    'Sim': [k for k in state_keys if k in ['sim_time_s', 'dt']],
  }

  sensor_doc = ''
  for gname, keys in sensor_groups.items():
    if keys:
      sensor_doc += f'\n  **{gname}** ({len(keys)} keys):\n'
      for k in sorted(keys):
        sensor_doc += f'    {k}\n'

  scenario_list = ''
  for i, sc in enumerate(SCENARIOS):
    scenario_list += f'  {i:2d}. {sc["name"]}\n'

  return f"""Write a Python autopilot for a self-driving car on a multi-lane freeway.
All units are METRIC (m, m/s, km/h, degrees).

**Function Signature:**
```python
def autopilot_step(state: dict, dt: float) -> dict:
```

**Input:** `state` is a dict with {num_keys} keys from redundant sensors:
{sensor_doc}
Key facts:
- Wheel speed sensors (4): FL, FR, RL, RR in km/h. Any can fail.
- Gyroscopes (3): yaw rate in degrees/second. Any can fail silently.
- Accelerometers (3): longitudinal and lateral in m/s². Any can fail.
- GPS (2): position in meters, speed in km/h. Can lose signal.
- Lidar (12 beams): range in meters at angles from -90° to +90°.
- Vision (3 independent systems): lane line distances, types (solid/dashed),
  obstacle detection (bool + distance + relative speed), speed limit signs,
  lane offset, road curvature. ANY can fail or flicker!
- `estimated_mu`: estimated tire-road friction (1.0=dry, <0.3=ice).
- `target_speed_kmh`: desired speed. `target_lane`: desired lane (0-2).
- `*_valid` flags exist but may be WRONG on failed sensors.
- Sensors can FLICKER: randomly return invalid data for a few frames.

**Output:** Return a dict:
```
  steering:   -1.0 (full left) to +1.0 (full right)
              0.0 = straight ahead
  throttle:   0.0 (no power) to 1.0 (full power)
  brake:      0.0 (no brake) to 1.0 (emergency stop)
```
If you omit a key, the current value is maintained.

**Critical Design Requirements:**
1. **Sensor fusion**: Cross-check redundant sensors. Median/voting for disagreements.
2. **Lane keeping**: Stay centered in lane using vision line distances and lidar.
3. **Speed control**: Maintain target speed, respect speed limits.
4. **Obstacle avoidance**: If obstacle detected, brake and/or lane change.
5. **Traction management**: On low-mu surfaces, reduce speed, gentler inputs.
6. **Graceful degradation**: Handle sensor failures, flickering data.
7. **Persistence**: Module-level variables persist across calls for filtering/PID.

**Test Scenarios ({len(SCENARIOS)} total):**
{scenario_list}
Your single autopilot function must handle ALL scenarios.
"""

extraGradeAnswerRuns = list(range(1, len(SCENARIOS)))

structure = {
  "type": "object",
  "properties": {
    "reasoning": {
      "type": "string",
      "description": "Explain your autopilot design"
    },
    "python_code": {
      "type": "string",
      "description": "Complete Python code defining autopilot_step(state, dt)"
    }
  },
  "required": ["reasoning", "python_code"],
  "additionalProperties": False
}

# ──────────────────────────────────────────────────────────────────────────────
# Grading
# ──────────────────────────────────────────────────────────────────────────────
def _setup_scenario(idx):
  sc = SCENARIOS[idx]
  road = RoadModel(
    curvature=sc.get('curvature', 0.0),
    speed_limit_kmh=sc.get('speed_limit', 110),
  )
  truth = make_car_truth(
    speed_ms=sc.get('speed', 30.6),
    lane=sc.get('lane', 1),
    road=road,
  )

  if 'mu' in sc:
    truth['mu'] = sc['mu']
  if sc.get('brake_fail'):
    truth['brake_fail'] = True
  if sc.get('throttle_stuck') is not None:
    truth['throttle_stuck'] = sc.get('throttle_stuck')
  if sc.get('steer_jam') is not None:
    truth['steer_jam'] = sc['steer_jam']

  sensor = CarSensorModel(seed=42 + idx)
  for key, (mode, param) in sc.get('failures', {}).items():
    sensor.set_failure(key, mode, param)

  if sc.get('flicker_vision'):
    for i in range(1, 4):
      for suffix in ['left_line', 'right_line', 'speed']:
        sensor.set_flicker(f'vision_{i}_{suffix}')

  if sc.get('flicker_all'):
    for i in range(1, 4):
      for suffix in ['left_line', 'right_line', 'speed']:
        sensor.set_flicker(f'vision_{i}_{suffix}')
    sensor.set_flicker('gyro_1')
    sensor.set_flicker('wheel_speed_fl')

  obstacles = []
  for odef in sc.get('obstacles', []):
    obstacles.append(Obstacle(
      x=odef.get('x', 100), y=odef.get('y', 5.55),
      vx=odef.get('vx', 0), vy=odef.get('vy', 0),
      width=odef.get('width', 2.0), length=odef.get('length', 4.5),
      label=odef.get('label', 'vehicle'),
    ))

  runner = CarSimRunner(truth, sensor, road, obstacles)
  return runner, sc


def gradeAnswer(result, subPass, aiEngineName):
  if not result or 'python_code' not in result:
    return 0.0, 'No Python code provided'

  code = result['python_code']
  try:
    ns = {}
    exec(compile(code, '<car_autopilot>', 'exec'), ns)
    if 'autopilot_step' not in ns:
      return 0.0, 'Code does not define autopilot_step(state, dt)'
    autopilot_fn = ns['autopilot_step']
  except Exception as e:
    return 0.0, f'Compilation failed: {e}'

  sc_idx = subPass
  if sc_idx >= len(SCENARIOS):
    return 0.0, f'Invalid subpass {sc_idx}'

  sc = SCENARIOS[sc_idx]
  runner, scenario = _setup_scenario(sc_idx)
  print(f'  Scenario {sc_idx}: {sc["name"]}')

  try:
    # Special handling for crosswind scenario
    if sc.get('crosswind'):
      _run_crosswind(runner, autopilot_fn)
    else:
      runner.run(autopilot_fn, steps=sc.get('steps', 2000))
  except Exception as e:
    tb = traceback.format_exc()
    return 0.0, f'Simulation error: {e}\n{tb[:500]}'

  score, details = runner.score()
  return score, f'[{sc["name"]}] {details}'


def _run_crosswind(runner, autopilot_fn):
  """Custom loop with time-varying lateral wind force."""
  dt = 0.02
  for step in range(2000):
    t = step * dt
    # Gust profile: ramps up, holds, ramps down
    if 5 < t < 10:
      runner.truth['vy'] += 0.15 * dt  # lateral push
    elif 15 < t < 20:
      runner.truth['vy'] -= 0.1 * dt  # second gust opposite

    state = runner.sensors.build_state(runner.truth, runner.road, runner.obstacles)
    try:
      controls = autopilot_fn(state, dt)
      if not isinstance(controls, dict):
        controls = {}
    except Exception:
      controls = {}

    for k in ['steering', 'brake', 'throttle']:
      if k in controls:
        controls[k] = max(-1.0, min(1.0, float(controls[k])))

    runner.truth = car_physics_step(runner.truth, controls, runner.road, dt)

    y = runner.truth['y']
    if y < -1.0 or y > runner.road.road_right_y() + 1.0:
      runner.crashed = True
      runner.crash_reason = f'Left roadway during crosswind (y={y:.1f}m)'
      break

    if step % 25 == 0:
      V = math.sqrt(runner.truth['vx']**2 + runner.truth['vy']**2)
      runner.history.append({
        'time': runner.truth['time'],
        'x': runner.truth['x'],
        'y': runner.truth['y'],
        'speed_kmh': V * 3.6,
        'yaw_deg': runner.truth['yaw'] * RAD2DEG,
        'lane': runner.sensors._current_lane(runner.truth['y'], runner.road),
      })
