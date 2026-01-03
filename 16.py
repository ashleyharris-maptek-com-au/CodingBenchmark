"""
Test 16: Job-Shop Scheduling

The LLM must write a Python solver for job-shop scheduling:
Given a list of jobs (each with multiple tasks) and a number of machines,
schedule tasks to minimize makespan (total completion time).

Each job has a sequence of tasks that must be executed in order.
Each task has a duration and must be assigned to a specific machine.

Subpasses test increasingly complex scheduling problems.
Solver times out after 5 minutes.
"""

import random
import subprocess
import sys
import tempfile
import os
import time
from typing import List, Tuple, Dict

title = "Job-Shop Scheduling"

# Timeout in seconds (5 minutes)
TIMEOUT_SECONDS = 30

# Seed for reproducibility
RANDOM_SEED = 11111


def generate_jobs(num_jobs: int, tasks_per_job: int, num_machines: int, min_duration: int,
                  max_duration: int, seed: int) -> List[List[Dict]]:
  """
    Generate random job-shop problem.
    Each job is a list of tasks, each task has 'machine' and 'duration'.
    """
  rng = random.Random(seed)

  jobs = []
  for _ in range(num_jobs):
    job = []
    # Each task in job must be on different machine (classic job-shop)
    machines = list(range(num_machines))
    rng.shuffle(machines)

    for i in range(min(tasks_per_job, num_machines)):
      task = {"machine": machines[i], "duration": rng.randint(min_duration, max_duration)}
      job.append(task)
    jobs.append(job)

  return jobs


# Test configurations
TEST_CASES = [
  # Subpass 0: Simple - 3 jobs, 2 machines
  {
    "jobs": [
      [{
        "machine": 0,
        "duration": 3
      }, {
        "machine": 1,
        "duration": 2
      }],
      [{
        "machine": 1,
        "duration": 2
      }, {
        "machine": 0,
        "duration": 4
      }],
      [{
        "machine": 0,
        "duration": 2
      }, {
        "machine": 1,
        "duration": 3
      }],
    ],
    "num_machines":
    2,
    "description":
    "3 jobs, 2 machines"
  },
  # Subpass 1: Medium - 4 jobs, 3 machines
  {
    "jobs": generate_jobs(4, 3, 3, 2, 5, RANDOM_SEED),
    "num_machines": 3,
    "description": "4 jobs, 3 machines"
  },
  # Subpass 2: Larger - 5 jobs, 4 machines
  {
    "jobs": generate_jobs(5, 4, 4, 2, 6, RANDOM_SEED + 1),
    "num_machines": 4,
    "description": "5 jobs, 4 machines"
  },
  # Subpass 3: Complex - 6 jobs, 4 machines
  {
    "jobs": generate_jobs(6, 4, 4, 3, 8, RANDOM_SEED + 2),
    "num_machines": 4,
    "description": "6 jobs, 4 machines"
  },
  # Subpass 4: Large - 8 jobs, 5 machines
  {
    "jobs": generate_jobs(8, 5, 5, 2, 7, RANDOM_SEED + 3),
    "num_machines": 5,
    "description": "8 jobs, 5 machines"
  },
  # Subpass 5: Very large - 10 jobs, 6 machines
  {
    "jobs": generate_jobs(10, 6, 6, 3, 10, RANDOM_SEED + 4),
    "num_machines": 6,
    "description": "10 jobs, 6 machines"
  },
  # Extreme cases
  {
    "jobs": generate_jobs(20, 8, 8, 2, 15, RANDOM_SEED + 5),
    "num_machines": 8,
    "description": "20 jobs, 8 machines"
  },
  {
    "jobs": generate_jobs(50, 10, 10, 2, 20, RANDOM_SEED + 6),
    "num_machines": 10,
    "description": "50 jobs, 10 machines"
  },
  {
    "jobs": generate_jobs(100, 15, 15, 3, 25, RANDOM_SEED + 7),
    "num_machines": 15,
    "description": "100 jobs, 15 machines"
  },
  {
    "jobs": generate_jobs(500, 20, 20, 2, 30, RANDOM_SEED + 8),
    "num_machines": 20,
    "description": "500 jobs, 20 machines (10k tasks)"
  },
  {
    "jobs": generate_jobs(1000, 50, 50, 2, 50, RANDOM_SEED + 9),
    "num_machines": 50,
    "description": "1000 jobs, 50 machines (50k tasks)"
  },
]


def format_jobs_for_prompt(jobs: List[List[Dict]]) -> str:
  """Format jobs for prompt display."""
  lines = []
  for i, job in enumerate(jobs):
    tasks = ", ".join(f"(M{t['machine']}, d={t['duration']})" for t in job)
    lines.append(f"    Job {i}: [{tasks}]")
  return "\n".join(lines)


def prepareSubpassPrompt(subPass: int) -> str:
  """Generate the prompt for subpass 0 that handles all scheduling complexities."""
  if subPass != 0:
    raise StopIteration

  return f"""You are solving a Job-Shop Scheduling problem.

You must write a Python solver that can handle ANY problem size from trivial to extremely large scale:
- **Trivial**: 2-3 jobs, 2-3 machines, 5-10 tasks total (simple cases)
- **Extreme**: 1000+ jobs, 50+ machines, 50k+ tasks total (massive optimization)

**The Challenge:**
Your `schedule_jobs(jobs, num_machines)` function will be tested with problems ranging in scope.
The same function must work efficiently across ALL scales.

**Input:**
- `jobs`: List of jobs, each job is a list of tasks with machine and duration
- `num_machines`: Number of available machines (numbered 0 to num_machines-1)

**Output:**
- Dict with:
  - `"makespan"`: Total completion time (when last task finishes)
  - `"schedule"`: List of lists - for each job, for each task: (start_time, machine)

**Critical Requirements:**
1. **Scalability**: Your algorithm must adapt based on number of jobs, machines, and tasks
2. **Performance**: Must complete within 5 minutes even for very large scheduling problems
3. **Quality**: Minimize makespan while respecting all constraints

**Key Constraints:**
1. Each task in a job must run on its specified machine
2. Tasks within a job must execute in order (task i before task i+1)
3. A machine can only run one task at a time
4. Tasks cannot be interrupted once started

**Example output:**
```python
{{
    "makespan": 12,
    "schedule": [
        [(0, 0), (3, 1)],     # Job 0: task 0 starts at t=0 on M0, task 1 at t=3 on M1
        [(0, 1), (5, 0)],     # Job 1: task 0 starts at t=0 on M1, task 1 at t=5 on M0
    ]
}}
```

**Constraints:**
- Use only Python standard library or numpy
- Must handle varying numbers of jobs, machines, and tasks efficiently
- Must respect all scheduling constraints
- Minimize makespan (total completion time)

Write complete, runnable Python code with the schedule_jobs function.
Include adaptive logic that chooses different strategies based on problem scale.
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
      "Explain your scheduling algorithm and how it adapts to different problem complexities"
    },
    "python_code": {
      "type":
      "string",
      "description":
      "Complete Python code with schedule_jobs(jobs, num_machines) function that handles all scales"
    }
  },
  "required": ["reasoning", "python_code"],
  "additionalProperties": False
}


def validate_schedule(schedule: List[List[Tuple]], jobs: List[List[Dict]],
                      num_machines: int) -> Tuple[bool, str, int]:
  """
    Validate job-shop schedule.
    Returns (is_valid, error, makespan).
    """
  if not isinstance(schedule, list):
    return False, "Schedule must be a list", 0

  if len(schedule) != len(jobs):
    return False, f"Schedule has {len(schedule)} jobs, expected {len(jobs)}", 0

  # Track machine usage: list of (start, end) intervals per machine
  machine_intervals = [[] for _ in range(num_machines)]
  makespan = 0

  for job_idx, (job_schedule, job) in enumerate(zip(schedule, jobs)):
    if len(job_schedule) != len(job):
      return False, f"Job {job_idx}: {len(job_schedule)} tasks scheduled, expected {len(job)}", 0

    prev_end = 0

    for task_idx, (task_sched, task) in enumerate(zip(job_schedule, job)):
      if not isinstance(task_sched, (list, tuple)) or len(task_sched) != 2:
        return False, f"Job {job_idx} task {task_idx}: invalid format", 0

      start_time, machine = int(task_sched[0]), int(task_sched[1])
      duration = task["duration"]
      expected_machine = task["machine"]
      end_time = start_time + duration

      # Check machine assignment
      if machine != expected_machine:
        return False, f"Job {job_idx} task {task_idx}: wrong machine {machine}, expected {expected_machine}", 0

      # Check precedence within job
      if start_time < prev_end:
        return False, f"Job {job_idx} task {task_idx}: starts at {start_time} before previous ends at {prev_end}", 0

      # Check machine conflicts
      for interval_start, interval_end in machine_intervals[machine]:
        if not (end_time <= interval_start or start_time >= interval_end):
          return False, f"Job {job_idx} task {task_idx}: conflicts with another task on machine {machine}", 0

      machine_intervals[machine].append((start_time, end_time))
      prev_end = end_time
      makespan = max(makespan, end_time)

  return True, "", makespan


def greedy_schedule(jobs: List[List[Dict]], num_machines: int) -> int:
  """
    Greedy first-fit scheduling.
    Returns makespan.
    """
  # Track when each machine is free
  machine_free = [0] * num_machines
  # Track when each job's previous task ends
  job_ready = [0] * len(jobs)
  # Track next task index for each job
  job_task_idx = [0] * len(jobs)

  total_tasks = sum(len(job) for job in jobs)
  scheduled = 0

  while scheduled < total_tasks:
    # Find task that can start earliest
    best_job = -1
    best_start = float('inf')

    for j, job in enumerate(jobs):
      if job_task_idx[j] >= len(job):
        continue

      task = job[job_task_idx[j]]
      machine = task["machine"]

      # Earliest this task can start
      earliest = max(job_ready[j], machine_free[machine])

      if earliest < best_start:
        best_start = earliest
        best_job = j

    if best_job < 0:
      break

    # Schedule this task
    task = jobs[best_job][job_task_idx[best_job]]
    machine = task["machine"]
    duration = task["duration"]
    start = max(job_ready[best_job], machine_free[machine])
    end = start + duration

    machine_free[machine] = end
    job_ready[best_job] = end
    job_task_idx[best_job] += 1
    scheduled += 1

  return max(machine_free)


def execute_solver(code: str,
                   jobs: List[List[Dict]],
                   num_machines: int,
                   timeout: int = TIMEOUT_SECONDS) -> tuple:
  """Execute the LLM's solver."""
  from solver_utils import execute_solver_with_data

  data_dict = {
    'jobs': jobs,
    'num_machines': num_machines,
  }

  result, error, exec_time = execute_solver_with_data(code, data_dict, 'schedule_jobs', timeout)

  if error:
    return None, error, exec_time

  if not isinstance(result, dict):
    return None, f"Invalid result type: expected dict, got {type(result).__name__}", exec_time

  try:
    schedule = result.get('schedule')
    if not isinstance(schedule, list):
      return None, "Invalid result: missing or non-list 'schedule'", exec_time

    normalized_schedule = []
    makespan = 0

    for job_idx, job_sched in enumerate(schedule):
      if not isinstance(job_sched, list):
        normalized_schedule.append([])
        continue

      normalized_job = []
      for task_idx, t in enumerate(job_sched):
        if not isinstance(t, (list, tuple)) or len(t) != 2:
          normalized_job.append((0, int(jobs[job_idx][task_idx]['machine'])))
          continue
        start_time = int(t[0])
        machine = int(t[1])
        normalized_job.append((start_time, machine))

        duration = int(jobs[job_idx][task_idx].get('duration', 0))
        makespan = max(makespan, start_time + duration)

      normalized_schedule.append(normalized_job)

    out = {
      'makespan': int(result.get('makespan', makespan)),
      'schedule': normalized_schedule,
    }
  except Exception as e:
    return None, f"Invalid result format: {e}", exec_time

  return out, None, exec_time


def gradeAnswer(result: dict, subPass: int, aiEngineName: str) -> tuple:
  """Grade the job-shop scheduler."""
  if not result:
    return 0.0, "No result provided"

  if "python_code" not in result:
    return 0.0, "No Python code provided"

  case = TEST_CASES[subPass]
  jobs = case["jobs"]
  num_machines = case["num_machines"]
  description = case["description"]
  code = result["python_code"]

  # Execute solver
  solution, error, exec_time = execute_solver(code, jobs, num_machines)

  if error:
    return 0.0, f"[{description}] {error}"

  # Validate schedule
  schedule = solution.get("schedule", [])
  is_valid, validation_error, actual_makespan = validate_schedule(schedule, jobs, num_machines)

  if not is_valid:
    return 0.0, f"[{description}] Invalid: {validation_error}"

  # Get baseline
  baseline_makespan = greedy_schedule(jobs, num_machines)

  # Score based on makespan vs baseline
  ratio = actual_makespan / baseline_makespan if baseline_makespan > 0 else 1.0

  if ratio <= 1.0:
    score = 1.0
    quality = "excellent (≤ baseline)"
  elif ratio <= 1.1:
    score = 0.85
    quality = "good (≤ 1.1x baseline)"
  elif ratio <= 1.25:
    score = 0.7
    quality = "acceptable (≤ 1.25x baseline)"
  else:
    score = 0.5
    quality = f"valid but slow ({ratio:.2f}x baseline)"

  explanation = (f"[{description}] Makespan: {actual_makespan}, Baseline: {baseline_makespan}, "
                 f"Time: {exec_time:.1f}s - {quality}")

  return score, explanation


def resultToNiceReport(result: dict, subPass: int, aiEngineName: str) -> str:
  """Generate HTML report."""
  if not result:
    return "<p style='color:red'>No result provided</p>"

  case = TEST_CASES[subPass]

  html = f"<h4>Job-Shop Scheduling - {case['description']}</h4>"

  if subPass == 0:
    if "reasoning" in result:
      reasoning = result['reasoning'][:500] + ('...'
                                               if len(result.get('reasoning', '')) > 500 else '')
      html += f"<p><strong>Algorithm:</strong> {reasoning}</p>"

    if "python_code" in result:
      code = result["python_code"]
      code_escaped = code.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
      html += f"<details><summary>View Code ({len(code)} chars)</summary><pre>{code_escaped}</pre></details>"

  return html


highLevelSummary = """
Job-Shop Scheduling is a classic NP-hard optimization problem.

**Problem:** Assign tasks to machines, respecting precedence and resource constraints,
to minimize makespan (total completion time).

**Constraints:**
- Each job has ordered sequence of tasks
- Each task requires specific machine
- Machine can only run one task at a time
- Tasks are non-preemptive

**Algorithms:**
1. **Priority dispatch rules:** SPT, LPT, FIFO, Most Work Remaining
2. **Shifting Bottleneck:** Iteratively solve single-machine subproblems
3. **Genetic algorithms:** Evolve operation permutations
4. **Constraint programming:** Model and solve with CP-SAT
5. **Branch and bound:** Exact but exponential

**Complexity:** NP-hard, but good heuristics achieve near-optimal results.

The baseline uses greedy first-fit scheduling - always schedule the task
that can start earliest.
"""
