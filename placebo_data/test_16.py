"""Test 16: Job-Shop Scheduling - Placebo responses for all control types."""


def get_response(model_name, subpass):
  """Return (result_dict, reasoning_string) for the given control type."""
  if model_name == 'naive':
    return _naive(subpass)
  elif model_name == 'naive-optimised':
    return _naive_optimised(subpass)
  elif model_name == 'best-published':
    return _best_published(subpass)
  elif model_name == 'random':
    return _random(subpass)
  elif model_name == 'human':
    return _human(subpass)
  return None, ''


def _naive(subpass):
  reasoning = "Using greedy first-fit scheduling: 1) Track when each machine becomes free. 2) Track when each job's previous task ends. 3) At each step, find the task that can start earliest. 4) Schedule it and update machine/job availability."
  code = 'def schedule_jobs(jobs, num_machines):\n    """\n    Schedule jobs using greedy first-fit.\n    Returns dict with makespan and schedule.\n    """\n    # Track when each machine is free\n    machine_free = [0] * num_machines\n    # Track when each job\'s previous task ends\n    job_ready = [0] * len(jobs)\n    # Track next task index for each job\n    job_task_idx = [0] * len(jobs)\n    \n    # Schedule storage: schedule[job][task] = (start_time, machine)\n    schedule = [[] for _ in range(len(jobs))]\n    \n    total_tasks = sum(len(job) for job in jobs)\n    scheduled = 0\n    \n    while scheduled < total_tasks:\n        # Find task that can start earliest\n        best_job = -1\n        best_start = float(\'inf\')\n        \n        for j, job in enumerate(jobs):\n            if job_task_idx[j] >= len(job):\n                continue\n            \n            task = job[job_task_idx[j]]\n            machine = task["machine"]\n            \n            # Earliest this task can start\n            earliest = max(job_ready[j], machine_free[machine])\n            \n            if earliest < best_start:\n                best_start = earliest\n                best_job = j\n        \n        if best_job < 0:\n            break\n        \n        # Schedule this task\n        task = jobs[best_job][job_task_idx[best_job]]\n        machine = task["machine"]\n        duration = task["duration"]\n        start = max(job_ready[best_job], machine_free[machine])\n        end = start + duration\n        \n        schedule[best_job].append((start, machine))\n        machine_free[machine] = end\n        job_ready[best_job] = end\n        job_task_idx[best_job] += 1\n        scheduled += 1\n    \n    makespan = max(machine_free)\n    \n    return {\n        "makespan": makespan,\n        "schedule": schedule\n    }\n'
  return {"reasoning": reasoning, 'cpp_code': code}, reasoning


def _naive_optimised(subpass):
  # TODO: Same algorithm as naive but hyper-optimised for C++
  # (parallel, SIMD, small data types, register packing, etc.)
  return _naive(subpass)


def _best_published(subpass):
  reasoning = (
    "Best published: Shifting bottleneck heuristic (Adams et al. 1988, Management Science 34(3):391-401). "
    "TODO: Full implementation pending."
  )
  code = '// TODO: Implement Shifting bottleneck heuristic'
  return {"reasoning": reasoning, 'cpp_code': code}, reasoning


def _random(subpass):
  reasoning = 'Random: pseudorandom output with fixed seed 42.'
  code = '#include <iostream>\n#include <random>\nusing namespace std;\nint main() {\n    mt19937 rng(42);\n    string line;\n    getline(cin, line);\n    cout << "0" << endl;\n    return 0;\n}'
  return {"reasoning": reasoning, 'cpp_code': code}, reasoning


def _human(subpass):
  reasoning = 'Human starting point for Job-Shop Scheduling. Fill in the TODOs.'
  code = '// TODO: Human attempt at Job-Shop Scheduling\n#include <iostream>\n#include <vector>\n#include <algorithm>\nusing namespace std;\nint main() {\n    ios_base::sync_with_stdio(false);\n    cin.tie(nullptr);\n    // TODO: Parse input\n    // TODO: Implement solution\n    // TODO: Output result\n    return 0;\n}'
  return {"reasoning": reasoning, 'cpp_code': code}, reasoning
