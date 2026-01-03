"""
Test 21: N-Dimensional Snake Game (C++ Implementation)

The LLM must write C++ code that plays a snake game in N-dimensional space.
Like classic Nokia snake, but generalized to arbitrary dimensions.

The snake:
- Occupies a series of connected integer grid points in N-D space
- Moves one step per turn along one axis (+1 or -1)
- Grows when eating food
- Dies if it hits an obstacle or itself

Goal: Collect as much food as possible without dying.

Input format (stdin):
Line 1: D max_turns (D = number of dimensions, max_turns = turn limit)
Line 2: bounds (D integers: size of space in each dimension)
Line 3: snake_length start_pos (length, then D integers for head position)
Line 4: F (number of food items)
Lines 5 to 4+F: food positions (D integers each)
Line 5+F: O (number of obstacles)
Following O lines: obstacle positions (D integers each)

Output format (stdout):
Each turn, output: axis direction
  - axis: 0 to D-1 (which dimension to move along)
  - direction: -1 or +1 (which way to move)

The game ends when:
- Snake eats all food (win)
- Snake hits obstacle or itself (lose)
- Max turns reached

Subpasses test increasingly complex scenarios.
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
from collections import deque

# Import our native compiler helper
from native_compiler import CppCompiler, CompilationError, ExecutionError

title = "N-Dimensional Snake Game (C++)"

# Timeout in seconds (5 minutes)
TIMEOUT_SECONDS = 30

# Seed for reproducibility
RANDOM_SEED = 21212121


def generate_snake_game(dimensions: int, bounds: List[int], num_food: int, num_obstacles: int,
                        snake_length: int,
                        seed: int) -> Tuple[List[int], List[Tuple], List[Tuple], Tuple]:
  """
    Generate a snake game setup.
    
    Returns:
        Tuple of (bounds, food_positions, obstacle_positions, snake_start)
    """
  rng = random.Random(seed)

  # Generate snake start position (centered-ish, away from edges)
  snake_start = tuple(
    rng.randint(snake_length + 2, b - snake_length - 3) if b > snake_length * 2 + 5 else b // 2
    for b in bounds)

  # Generate initial snake body (straight line along first axis)
  snake_body = set()
  for i in range(snake_length):
    pos = list(snake_start)
    pos[0] -= i
    if pos[0] >= 0:
      snake_body.add(tuple(pos))

  # Generate food positions (not on snake or obstacles)
  food = []
  attempts = 0
  while len(food) < num_food and attempts < num_food * 100:
    pos = tuple(rng.randint(1, b - 2) for b in bounds)
    if pos not in snake_body and pos not in food:
      food.append(pos)
    attempts += 1

  # Generate obstacle positions (not on snake, food, or other obstacles)
  obstacles = []
  attempts = 0
  occupied = snake_body | set(food)
  while len(obstacles) < num_obstacles and attempts < num_obstacles * 100:
    pos = tuple(rng.randint(0, b - 1) for b in bounds)
    if pos not in occupied and pos not in obstacles:
      obstacles.append(pos)
      occupied.add(pos)
    attempts += 1

  return bounds, food, obstacles, snake_start


def format_input(dimensions: int, bounds: List[int], food: List[Tuple], obstacles: List[Tuple],
                 snake_start: Tuple, snake_length: int, max_turns: int) -> str:
  """Format game setup as input string."""
  lines = [f"{dimensions} {max_turns}"]
  lines.append(" ".join(map(str, bounds)))
  lines.append(f"{snake_length} " + " ".join(map(str, snake_start)))
  lines.append(str(len(food)))
  for f in food:
    lines.append(" ".join(map(str, f)))
  lines.append(str(len(obstacles)))
  for o in obstacles:
    lines.append(" ".join(map(str, o)))
  return "\n".join(lines)


class SnakeGame:
  """Simulates the N-D snake game."""

  def __init__(self, dimensions: int, bounds: List[int], food: List[Tuple], obstacles: Set[Tuple],
               snake_start: Tuple, snake_length: int):
    self.dimensions = dimensions
    self.bounds = bounds
    self.food = set(food)
    self.obstacles = obstacles
    self.snake = deque()  # Front is head

    # Initialize snake body
    for i in range(snake_length):
      pos = list(snake_start)
      pos[0] -= i
      if all(0 <= p < b for p, b in zip(pos, bounds)):
        self.snake.append(tuple(pos))

    self.alive = True
    self.food_eaten = 0
    self.turns = 0

  def get_head(self) -> Tuple:
    return self.snake[0] if self.snake else None

  def move(self, axis: int, direction: int) -> Tuple[bool, str]:
    """
        Execute a move.
        Returns (success, message).
        """
    if not self.alive:
      return False, "Snake is dead"

    if axis < 0 or axis >= self.dimensions:
      return False, f"Invalid axis {axis}"

    if direction not in (-1, 1):
      return False, f"Invalid direction {direction}"

    head = list(self.get_head())
    head[axis] += direction
    new_head = tuple(head)

    # Check bounds
    if not all(0 <= p < b for p, b in zip(new_head, self.bounds)):
      self.alive = False
      return False, "Hit wall"

    # Check obstacle
    if new_head in self.obstacles:
      self.alive = False
      return False, "Hit obstacle"

    # Check self (excluding tail which will move)
    body_without_tail = set(list(self.snake)[:-1])
    if new_head in body_without_tail:
      self.alive = False
      return False, "Hit self"

    # Move snake
    self.snake.appendleft(new_head)

    # Check food
    if new_head in self.food:
      self.food.remove(new_head)
      self.food_eaten += 1
      # Don't remove tail - snake grows
    else:
      self.snake.pop()  # Remove tail

    self.turns += 1
    return True, "OK"

  def is_won(self) -> bool:
    return len(self.food) == 0


# Test configurations
TEST_CASES = [
  # Subpass 0: Simple 2D
  {
    "dimensions": 2,
    "bounds": [20, 20],
    "num_food": 5,
    "num_obstacles": 10,
    "snake_length": 3,
    "max_turns": 200,
    "description": "2D 20x20, 5 food, 10 obstacles"
  },
  # Subpass 1: 2D larger
  {
    "dimensions": 2,
    "bounds": [50, 50],
    "num_food": 15,
    "num_obstacles": 50,
    "snake_length": 5,
    "max_turns": 500,
    "description": "2D 50x50, 15 food, 50 obstacles"
  },
  # Subpass 2: 3D small
  {
    "dimensions": 3,
    "bounds": [15, 15, 15],
    "num_food": 10,
    "num_obstacles": 30,
    "snake_length": 3,
    "max_turns": 300,
    "description": "3D 15x15x15, 10 food, 30 obstacles"
  },
  # Subpass 3: 3D larger
  {
    "dimensions": 3,
    "bounds": [25, 25, 25],
    "num_food": 20,
    "num_obstacles": 100,
    "snake_length": 4,
    "max_turns": 500,
    "description": "3D 25x25x25, 20 food, 100 obstacles"
  },
  # Subpass 4: 4D
  {
    "dimensions": 4,
    "bounds": [12, 12, 12, 12],
    "num_food": 15,
    "num_obstacles": 80,
    "snake_length": 3,
    "max_turns": 400,
    "description": "4D 12^4, 15 food, 80 obstacles"
  },
  # Subpass 5: 5D
  {
    "dimensions": 5,
    "bounds": [10, 10, 10, 10, 10],
    "num_food": 20,
    "num_obstacles": 150,
    "snake_length": 3,
    "max_turns": 500,
    "description": "5D 10^5, 20 food, 150 obstacles"
  },
  # Extreme cases
  {
    "dimensions": 6,
    "bounds": [8, 8, 8, 8, 8, 8],
    "num_food": 25,
    "num_obstacles": 200,
    "snake_length": 4,
    "max_turns": 600,
    "description": "6D 8^6, 25 food, 200 obstacles"
  },
  {
    "dimensions": 8,
    "bounds": [6, 6, 6, 6, 6, 6, 6, 6],
    "num_food": 30,
    "num_obstacles": 300,
    "snake_length": 3,
    "max_turns": 800,
    "description": "8D 6^8, 30 food, 300 obstacles"
  },
  {
    "dimensions": 10,
    "bounds": [5, 5, 5, 5, 5, 5, 5, 5, 5, 5],
    "num_food": 40,
    "num_obstacles": 500,
    "snake_length": 3,
    "max_turns": 1000,
    "description": "10D 5^10, 40 food, 500 obstacles"
  },
  {
    "dimensions": 15,
    "bounds": [4] * 15,
    "num_food": 50,
    "num_obstacles": 800,
    "snake_length": 3,
    "max_turns": 1500,
    "description": "15D 4^15, 50 food, 800 obstacles"
  },
  {
    "dimensions": 20,
    "bounds": [3] * 20,
    "num_food": 60,
    "num_obstacles": 1000,
    "snake_length": 2,
    "max_turns": 2000,
    "description": "20D 3^20, 60 food, 1000 obstacles"
  },
]

# Pre-generate test data
TEST_DATA_CACHE = {}


def get_test_data(subpass: int):
  """Get or generate test data for subpass."""
  if subpass not in TEST_DATA_CACHE:
    case = TEST_CASES[subpass]
    bounds, food, obstacles, snake_start = generate_snake_game(case["dimensions"], case["bounds"],
                                                               case["num_food"],
                                                               case["num_obstacles"],
                                                               case["snake_length"],
                                                               RANDOM_SEED + subpass)
    TEST_DATA_CACHE[subpass] = (bounds, food, obstacles, snake_start)
  return TEST_DATA_CACHE[subpass]


def prepareSubpassPrompt(subPass: int) -> str:
  """Generate the prompt for subpass 0 that handles all dimensional complexities."""
  if subPass != 0:
    raise StopIteration

  return f"""You are writing C++ code to play an N-Dimensional Snake Game.

You must write a C++ solver that can handle ANY dimensional complexity from trivial to ludicrous scale:
- **Trivial**: 2D games (8x8), few obstacles, basic pathfinding
- **Medium**: 3D-4D games (6x6x6 to 8x8x8x8), moderate obstacles, intermediate planning
- **Large**: 6D-8D games (4x4x4x4x4x4 to 6x6x6x6x6x6x6x6), many obstacles, complex pathfinding
- **Extreme**: 10D-20D games (3x3x3... to 4x4x4...), massive state spaces, very complex optimization

**The Challenge:**
Your C++ snake player will be tested with games ranging from simple 2D boards to massive 20D hypercubes. The same algorithm must work efficiently across ALL dimensionalities.

**Problem:**
Like classic Nokia snake, but generalized to arbitrary dimensions. The snake occupies a series of connected integer grid points in N-D space, moves one step per turn along one axis (+1 or -1), grows when eating food, and dies if it hits an obstacle or itself.

**Input format (stdin):**
```
D max_turns (D = number of dimensions, max_turns = turn limit)
bounds (D integers: size of space in each dimension)
snake_length start_pos (length, then D integers for head position)
F (number of food items)
F lines: food positions (D integers each)
O (number of obstacles)
O lines: obstacle positions (D integers each)
```

**Output format (stdout):**
Each turn, output: axis direction
  - axis: 0 to D-1 (which dimension to move along)
  - direction: -1 or +1 (which way to move)

**Critical Requirements:**
1. **Scalability**: Your algorithm must adapt based on number of dimensions and board size
2. **Performance**: Must complete within 5 minutes even for 20D games
3. **Quality**: Collect maximum food while avoiding collisions

**Algorithm Strategy Recommendations:**
- **Low dimensions (2D-3D)**: Can use A* pathfinding, exhaustive search
- **Medium dimensions (4D-6D)**: Heuristic pathfinding, limited look-ahead
- **High dimensions (7D-10D)**: Fast heuristics, greedy approaches
- **Very High dimensions (>10D)**: Very fast heuristics, possibly random walk with bias

**Key Techniques:**
- **State representation**: Efficient N-dimensional position storage
- **Pathfinding**: A* or Dijkstra adapted for N-D space
- **Collision detection**: Fast checking for self-collision and obstacles
- **Food targeting**: Prioritize nearest/safest food items
- **Survival strategy**: Balance food collection with safety

**Implementation Hints:**
- Detect dimension count and choose appropriate algorithm complexity
- Use efficient data structures for N-D positions (vectors, hash sets)
- Implement adaptive look-ahead based on dimensionality
- For very high dimensions, consider simplified heuristics
- Handle edge cases: no valid moves, trapped situations
- Use fast I/O for large inputs

**Game ends when:**
- Snake eats all food (win)
- Snake hits obstacle or itself (lose)
- Max turns reached

**Requirements:**
1. Program must compile with g++ or MSVC (C++17)
2. Read from stdin, write to stdout
3. Handle up to 20 dimensions with large board sizes
4. Complete within 5 minutes
5. Must handle varying dimensionalities efficiently

Write complete, compilable C++ code with a main() function.
Include adaptive logic that chooses different strategies based on dimensional complexity.
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
      "Explain your algorithm approach and how it adapts to different dimensional complexities"
    },
    "cpp_code": {
      "type": "string",
      "description": "Complete C++ code with main() function that handles all scales"
    }
  },
  "required": ["reasoning", "cpp_code"],
  "additionalProperties": False
}


def execute_snake_game(code: str, input_data: str, case: dict,
                       engine_name: str) -> Tuple[int, int, str, float]:
  """
    Compile C++ code and run snake game simulation.
    
    Returns:
        Tuple of (food_eaten, turns_taken, end_reason, exec_time)
    """
  compiler = CppCompiler(engine_name)

  if not compiler.find_compiler():
    return 0, 0, "No C++ compiler found", 0

  try:
    exe_path = compiler.compile(code)
  except CompilationError as e:
    return 0, 0, f"Compilation error: {str(e)[:500]}", 0

  # Run the game interactively
  bounds, food, obstacles, snake_start = get_test_data(TEST_CASES.index(case))

  game = SnakeGame(case["dimensions"], bounds, food, set(obstacles), snake_start,
                   case["snake_length"])

  start_time = time.time()

  try:
    process = subprocess.Popen([str(exe_path)],
                               stdin=subprocess.PIPE,
                               stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE,
                               text=True)

    # Send input
    process.stdin.write(input_data + "\n")
    process.stdin.flush()

    # Read moves and simulate
    max_turns = case["max_turns"]
    end_reason = "max_turns"

    for turn in range(max_turns):
      if not game.alive:
        break
      if game.is_won():
        end_reason = "won"
        break

      # Read next move with timeout
      try:
        line = process.stdout.readline()
        if not line:
          end_reason = "no_output"
          break

        parts = line.strip().split()
        if len(parts) < 2:
          end_reason = f"invalid_output: {line.strip()}"
          break

        axis = int(parts[0])
        direction = int(parts[1])

        success, msg = game.move(axis, direction)
        if not success:
          end_reason = msg
          break

      except Exception as e:
        end_reason = f"error: {str(e)}"
        break

    process.terminate()
    try:
      process.wait(timeout=2)
    except:
      process.kill()

    exec_time = time.time() - start_time
    return game.food_eaten, game.turns, end_reason, exec_time

  except Exception as e:
    return 0, 0, f"Execution error: {str(e)}", time.time() - start_time


def gradeAnswer(result: dict, subPass: int, aiEngineName: str) -> tuple:
  """
    Grade the C++ snake player.
    
    Scoring based on food collected:
    - 1.0: Collected all food
    - Proportional to food/total for partial
    """
  if not result:
    return 0.0, "No result provided"

  if "cpp_code" not in result:
    return 0.0, "No C++ code provided"

  case = TEST_CASES[subPass]
  bounds, food, obstacles, snake_start = get_test_data(subPass)
  description = case["description"]

  code = result["cpp_code"]
  input_data = format_input(case["dimensions"], bounds, food, obstacles, snake_start,
                            case["snake_length"], case["max_turns"])

  # Run game
  food_eaten, turns, end_reason, exec_time = execute_snake_game(code, input_data, case,
                                                                aiEngineName)

  total_food = len(food)

  if total_food == 0:
    score = 1.0 if end_reason == "won" else 0.5
  else:
    score = food_eaten / total_food

  # Bonus for winning
  if end_reason == "won":
    score = 1.0

  # Penalty for early death
  if end_reason in ("Hit wall", "Hit obstacle", "Hit self"):
    score *= 0.8  # Keep some credit for food eaten before death

  explanation = (f"[{description}] Food: {food_eaten}/{total_food}, "
                 f"Turns: {turns}, End: {end_reason}, "
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
  """Generate HTML header."""
  return """
    <h2>Test 21: N-Dimensional Snake Game (C++)</h2>
    <p>Testing C++ implementation of snake game in arbitrary dimensions.</p>
    """


def output_summary_html(results: list) -> str:
  """Generate summary HTML."""
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
