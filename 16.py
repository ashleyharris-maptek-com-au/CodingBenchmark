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
import time
from typing import List, Tuple, Dict

import hashlib
import os
import tempfile
from pathlib import Path

from native_compiler import CppCompiler, compile_and_run, describe_this_pc

title = "Job-Shop Scheduling (C++)"

# Timeout in seconds (5 minutes)
TIMEOUT_SECONDS = 30

# Seed for reproducibility
RANDOM_SEED = 11111

_BASELINE_MAKESPAN_CACHE: Dict[Tuple[int, str], int] = {}


def _baseline_cache_dir() -> Path:
  p = Path(tempfile.gettempdir()) / "CodingBenchmark" / "test16_baseline_cache"
  try:
    p.mkdir(parents=True, exist_ok=True)
  except Exception:
    pass
  return p


def _jobs_fingerprint(jobs: List[List[Dict]], num_machines: int) -> str:
  h = hashlib.sha256()
  h.update(f"m={num_machines};j={len(jobs)};".encode("utf-8"))
  for job in jobs:
    h.update(f"[{len(job)}]".encode("utf-8"))
    for t in job:
      h.update(f"{int(t['machine'])},{int(t['duration'])};".encode("utf-8"))
  return h.hexdigest()


def _get_baseline_makespan_cached(subpass: int, jobs: List[List[Dict]],
                                  num_machines: int) -> Tuple[int, bool, float]:
  """Return (baseline_makespan, from_cache, compute_time_seconds)."""
  fp = _jobs_fingerprint(jobs, num_machines)
  mem_key = (subpass, fp)
  if mem_key in _BASELINE_MAKESPAN_CACHE:
    return _BASELINE_MAKESPAN_CACHE[mem_key], True, 0.0

  cache_dir = _baseline_cache_dir()
  cache_path = cache_dir / f"s{subpass}_{fp}.txt"
  try:
    text = cache_path.read_text(encoding="utf-8").strip()
    if text:
      baseline_makespan = int(text)
      _BASELINE_MAKESPAN_CACHE[mem_key] = baseline_makespan
      return baseline_makespan, True, 0.0
  except Exception:
    pass

  t0 = time.time()
  baseline_makespan = greedy_schedule(jobs, num_machines)
  dt = time.time() - t0
  _BASELINE_MAKESPAN_CACHE[mem_key] = baseline_makespan
  try:
    tmp_path = cache_path.with_suffix(cache_path.suffix + f".{os.getpid()}.tmp")
    tmp_path.write_text(str(int(baseline_makespan)), encoding="utf-8")
    os.replace(str(tmp_path), str(cache_path))
  except Exception:
    pass
  return baseline_makespan, False, dt


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
    "jobs":
    lambda: [
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
    "jobs": lambda: generate_jobs(4, 3, 3, 2, 5, RANDOM_SEED),
    "num_machines": 3,
    "description": "4 jobs, 3 machines"
  },
  # Subpass 2: Larger - 5 jobs, 4 machines
  {
    "jobs": lambda: generate_jobs(5, 4, 4, 2, 6, RANDOM_SEED + 1),
    "num_machines": 4,
    "description": "5 jobs, 4 machines"
  },
  # Subpass 3: Complex - 6 jobs, 4 machines
  {
    "jobs": lambda: generate_jobs(6, 4, 4, 3, 8, RANDOM_SEED + 2),
    "num_machines": 4,
    "description": "6 jobs, 4 machines"
  },
  # Subpass 4: Large - 8 jobs, 5 machines
  {
    "jobs": lambda: generate_jobs(8, 5, 5, 2, 7, RANDOM_SEED + 3),
    "num_machines": 5,
    "description": "8 jobs, 5 machines"
  },
  # Subpass 5: Very large - 10 jobs, 6 machines
  {
    "jobs": lambda: generate_jobs(10, 6, 6, 3, 10, RANDOM_SEED + 4),
    "num_machines": 6,
    "description": "10 jobs, 6 machines"
  },
  # Extreme cases
  {
    "jobs": lambda: generate_jobs(20, 8, 8, 2, 15, RANDOM_SEED + 5),
    "num_machines": 8,
    "description": "20 jobs, 8 machines"
  },
  {
    "jobs": lambda: generate_jobs(50, 10, 10, 2, 20, RANDOM_SEED + 6),
    "num_machines": 10,
    "description": "50 jobs, 10 machines"
  },
  {
    "jobs": lambda: generate_jobs(100, 15, 15, 3, 25, RANDOM_SEED + 7),
    "num_machines": 15,
    "description": "100 jobs, 15 machines"
  },
  {
    "jobs": lambda: generate_jobs(500, 20, 20, 2, 300, RANDOM_SEED + 8),
    "num_machines": 20,
    "description": "500 jobs, 20 machines (10k tasks)"
  },
  {
    "jobs": lambda: generate_jobs(1000, 50, 50, 2, 5000, RANDOM_SEED + 9),
    "num_machines": 50,
    "description": "1000 jobs, 50 machines (50k tasks)"
  },
  {
    "jobs": lambda: generate_jobs(10000, 50, 50, 200, 5000, RANDOM_SEED + 10),
    "num_machines": 50,
    "description": "10000 jobs, 50 machines (50k tasks)"
  },
  {
    "jobs": lambda: generate_jobs(100000, 50, 50, 2000, 50000, RANDOM_SEED + 11),
    "num_machines": 50,
    "description": "100000 jobs, 50 machines (50k tasks)"
  },
  {
    "jobs": lambda: generate_jobs(1000000, 50, 50, 20000, 500000, RANDOM_SEED + 12),
    "num_machines": 50,
    "description": "1000000 jobs, 50 machines (50k tasks)"
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

  return f"""You are solving a Job-Shop Scheduling problem in C++.

You must write a C++ solver that can handle ANY problem size from trivial to extremely large scale:
- **Trivial**: 2-3 jobs, 2-3 machines, 5-10 tasks total
- **Extreme**: 1000+ jobs, 50+ machines, 50k+ tasks total

**Input format (stdin):**
Line 1: J M (number of jobs, number of machines)
For each job:
  Line: T (number of tasks in this job)
  Next T lines: machine duration

**Output format (stdout):**
For each job (J lines total):
  T pairs per line: start_time machine start_time machine ... (all tasks for that job)

**Example:**
Input:
2 2
2
0 3
1 2
2
1 2
0 4

Output:
0 0 3 1
0 1 5 0

(Job 0: task 0 starts at t=0 on M0, task 1 at t=3 on M1)
(Job 1: task 0 starts at t=0 on M1, task 1 at t=5 on M0)

**Key Constraints:**
1. Each task in a job must run on its specified machine
2. Tasks within a job must execute in order (task i before task i+1)
3. A machine can only run one task at a time
4. Tasks cannot be interrupted once started

**Environment:**
{describe_this_pc()}

**C++ Compiler:**
{CppCompiler("test_engine").describe()}

Write complete, compilable C++ code with a main() function.
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
    "cpp_code": {
      "type": "string",
      "description": "Complete C++ code with main() that handles all scales"
    }
  },
  "required": ["reasoning", "cpp_code"],
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
  # Store (start, end, job_idx, task_idx) for better error messages
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

      if start_time < 0:
        return False, f"Job {job_idx} task {task_idx}: negative start time {start_time}", 0

      # Check machine assignment
      if machine != expected_machine:
        return False, f"Job {job_idx} task {task_idx}: wrong machine {machine}, expected {expected_machine}", 0

      # Check precedence within job
      if start_time < prev_end:
        return False, f"Job {job_idx} task {task_idx}: starts at {start_time} before previous ends at {prev_end}", 0

      machine_intervals[machine].append((start_time, end_time, job_idx, task_idx))
      prev_end = end_time
      makespan = max(makespan, end_time)

  # Check machine conflicts efficiently: sort by start time and ensure no overlap.
  for m in range(num_machines):
    intervals = machine_intervals[m]
    if len(intervals) <= 1:
      continue
    intervals.sort(key=lambda x: x[0])
    prev_s, prev_e, prev_j, prev_t = intervals[0]
    for s, e, j, t in intervals[1:]:
      if s < prev_e:
        return False, (
          f"Job {j} task {t}: conflicts with Job {prev_j} task {prev_t} on machine {m}"), 0
      prev_s, prev_e, prev_j, prev_t = s, e, j, t

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


def format_input(jobs: List[List[Dict]], num_machines: int) -> str:
  lines = [f"{len(jobs)} {num_machines}"]
  for job in jobs:
    lines.append(str(len(job)))
    for task in job:
      lines.append(f"{task['machine']} {task['duration']}")
  return "\n".join(lines)


def parse_schedule_output(output: str, jobs: List[List[Dict]]) -> tuple:
  text = output.strip()
  if not text:
    return None, "Empty output"

  lines = [l for l in text.splitlines() if l.strip()]
  if len(lines) < len(jobs):
    return None, f"Expected {len(jobs)} schedule lines, got {len(lines)}"

  schedule = []
  makespan = 0

  for job_idx in range(len(jobs)):
    parts = lines[job_idx].split()
    num_tasks = len(jobs[job_idx])
    if len(parts) < num_tasks * 2:
      return None, f"Job {job_idx}: expected {num_tasks * 2} values, got {len(parts)}"

    job_sched = []
    for t in range(num_tasks):
      try:
        start_time = int(parts[t * 2])
        machine = int(parts[t * 2 + 1])
      except ValueError:
        return None, f"Job {job_idx} task {t}: non-integer values"
      job_sched.append((start_time, machine))
      duration = jobs[job_idx][t]['duration']
      makespan = max(makespan, start_time + duration)

    schedule.append(job_sched)

  return {'makespan': makespan, 'schedule': schedule}, None


def execute_solver(code: str,
                   jobs: List[List[Dict]],
                   num_machines: int,
                   ai_engine_name: str,
                   timeout: int = TIMEOUT_SECONDS) -> tuple:
  """Execute the LLM's solver."""
  input_data = format_input(jobs, num_machines)
  run = compile_and_run(code, "cpp", ai_engine_name, input_data=input_data, timeout=timeout)

  if not run:
    return None, run.error_message(), run.exec_time

  solution, parse_error = parse_schedule_output(run.stdout, jobs)
  if parse_error:
    return None, parse_error, run.exec_time

  return solution, None, run.exec_time


def gradeAnswer(result: dict, subPass: int, aiEngineName: str) -> tuple:
  if not result:
    return 0.0, "No result provided"

  if "cpp_code" not in result:
    return 0.0, "No C++ code provided"

  case = TEST_CASES[subPass]
  jobsLambda = case["jobs"]
  num_machines = case["num_machines"]
  description = case["description"]
  code = result["cpp_code"]

  time1 = time.time()
  jobs = jobsLambda()
  time2 = time.time() - time1
  if time2 > 1:
    print(f"Job generation time: {time2:.2f} seconds")

  # Execute solver
  solution, error, exec_time = execute_solver(code, jobs, num_machines, aiEngineName)

  if error:
    return 0.0, f"[{description}] {error}"

  # Validate schedule
  schedule = solution.get("schedule", [])
  time3 = time.time()
  is_valid, validation_error, actual_makespan = validate_schedule(schedule, jobs, num_machines)
  time4 = time.time() - time3
  if time4 > 1:
    print(f"Validation time: {time4:.2f} seconds")

  if not is_valid:
    return 0.0, f"[{description}] Invalid: {validation_error}"

  # Get baseline
  baseline_makespan, baseline_cached, baseline_time = _get_baseline_makespan_cached(
    subPass, jobs, num_machines)
  if baseline_cached:
    pass
  elif baseline_time > 1:
    print(f"Baseline calculation time: {baseline_time:.2f} seconds")

  # Score based on makespan vs baseline
  ratio = actual_makespan / baseline_makespan if baseline_makespan > 0 else 1.0

  if ratio <= 1.0:
    score = 1.0
    quality = "excellent (≤ baseline)"
  elif ratio <= 1.1:
    score = 0.5
    quality = "good (≤ 1.1x baseline)"
  elif ratio <= 1.25:
    score = 0.1
    quality = "acceptable (≤ 1.25x baseline)"
  else:
    score = 0.0
    quality = f"valid but inefficient ({ratio:.2f}x baseline)"

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

    if "cpp_code" in result:
      code = result["cpp_code"]
      code_escaped = code.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
      html += f"<details><summary>View Code ({len(code)} chars)</summary><pre>{code_escaped}</pre></details>"

  return html


highLevelSummary = """
<p>Schedule a set of jobs across several machines to finish everything as quickly
as possible. Each job is a sequence of tasks that must run in order, and each
task needs a specific machine &mdash; but a machine can only do one thing at a time.</p>
<p>This classic factory-scheduling problem is NP-hard. Subpasses increase the
number of jobs and machines. The baseline greedily starts whichever task can
begin earliest.</p>
"""
