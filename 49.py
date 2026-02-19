"""
Test 49: Parallel Prefix Sum (SPIR-V Binary Compute Shader)

The LLM must write raw SPIR-V binary (hex-encoded) for a compute shader
that performs an inclusive prefix sum (scan) on an array of uint32 values.

This tests:
1. SPIR-V binary format knowledge (magic number, headers, opcodes)
2. Workgroup shared memory and barriers
3. Parallel prefix sum algorithm (Hillis-Steele or Blelloch)
4. Correct results at scale

Buffer layout:
  Binding 0 (read):      Input uint32 array, length N
  Binding 1 (readwrite): Output uint32 array, length N
  Binding 2 (uniform):   Params - uint32[4]: [N, 0, 0, 0]
"""

import struct
import time
from typing import Tuple, Optional

import numpy as np

from shader_test_utils import validate_spirv
from compute_test_utils import ComputeShaderRunner, grade_compute

title = "Parallel Prefix Sum (SPIR-V Binary Compute)"

RANDOM_SEED = 49494949
TIMEOUT_SECONDS = 60

# Subpass configurations: (N_elements,)
SUBPASSES = [
    (256,),
    (1024,),
    (4096,),
    (16384,),
    (65536,),
]


def generate_input(n, seed=RANDOM_SEED):
  """Generate N uint32 values for prefix sum."""
  rng = np.random.RandomState(seed + n)
  # Use small values to avoid overflow for large N
  max_val = max(1, min(100, 2**31 // n))
  return rng.randint(0, max_val, size=n, dtype=np.uint32)


def cpu_prefix_sum(arr):
  """Inclusive prefix sum on CPU."""
  return np.cumsum(arr).astype(np.uint32)


def input_to_buffer(arr):
  """Pack uint32 array to bytes."""
  return arr.tobytes()


def buffer_to_array(data, n):
  """Unpack bytes to uint32 array."""
  return np.frombuffer(data[:n * 4], dtype=np.uint32).copy()


def make_params_buffer(n):
  """Create params uniform buffer."""
  return struct.pack('4I', n, 0, 0, 0)


def compare_arrays(gpu_arr, ref_arr):
  """Compare GPU and CPU prefix sums. Returns (score, description)."""
  n = len(ref_arr)
  if len(gpu_arr) < n:
    return 0.0, f"Output too short: got {len(gpu_arr)}, expected {n}"

  gpu_arr = gpu_arr[:n]
  matches = np.sum(gpu_arr == ref_arr)
  total = n

  if matches == total:
    return 1.0, f"Perfect: all {n} elements match exactly"

  # Check how far the first mismatch is
  first_mismatch = np.argmax(gpu_arr != ref_arr)

  # Partial credit based on fraction correct
  frac = matches / total
  if frac > 0.99:
    score = 0.9
  elif frac > 0.9:
    score = 0.5
  elif frac > 0.5:
    score = 0.2
  else:
    score = 0.0

  desc = (f"{matches}/{total} elements correct ({frac:.1%}). "
          f"First mismatch at index {first_mismatch}: "
          f"got {gpu_arr[first_mismatch]}, expected {ref_arr[first_mismatch]}")
  return score, desc


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

SPIRV_BINARY_INTERFACE = """**SPIR-V Binary (hex-encoded) Compute Shader Interface:**

You must produce raw SPIR-V binary encoded as a hex string (no spaces, no 0x prefix).
The binary must start with the SPIR-V magic number 0x07230203.

**Required layout:**
```
Binding 0 (set=0): Storage buffer (read-only) - uint array, length N
Binding 1 (set=0): Storage buffer (read-write) - uint array, length N
Binding 2 (set=0): Uniform buffer - uint4: [N, 0, 0, 0]

Local size: 256 x 1 x 1
```

**Algorithm:** Inclusive prefix sum (scan)
- Input:  [a0, a1, a2, a3, ...]
- Output: [a0, a0+a1, a0+a1+a2, a0+a1+a2+a3, ...]

Compute an inclusive scan over the input array. For N <= 256, use a shared-memory
scan within a single workgroup (synchronize as needed). For N > 256, a naive
fallback (each thread sums all previous elements) is acceptable but slow; full
marks require a multi-workgroup scan.

**SPIR-V binary encoding notes:**
- Little-endian uint32 words
- Magic: 0x07230203
- Version: 0x00010000 (1.0)
- Bound: upper bound on all IDs used + 1
- Schema: 0
- OpCapability Shader = [0x00020011, 0x00000001]
- OpMemoryModel Logical GLSL450 = [0x00030006, 0x00000001, 0x00000001]
- OpEntryPoint GLCompute = opcode 0x0000000F, execution model 5
- OpExecutionMode LocalSize = opcode 0x00000010, mode 17 (LocalSize)
- OpDecorate Binding = decoration 33
- OpDecorate DescriptorSet = decoration 34
- OpDecorate ArrayStride = decoration 6
- OpDecorate BufferBlock or Block = decoration 2 or 3
- OpTypeInt 32 0 = unsigned int
- OpTypePointer StorageBuffer = storage class 12
- OpVariable StorageBuffer = storage class 12

Encode every uint32 word as exactly 8 hex characters (little-endian).
"""


def prepareSubpassPrompt(subPass):
  if subPass != 0:
    raise StopIteration

  configs = []
  for i, (n,) in enumerate(SUBPASSES):
    configs.append(f"  Subpass {i}: N={n:,} elements")

  return f"""Write a SPIR-V binary compute shader (hex-encoded) for parallel inclusive prefix sum.

{SPIRV_BINARY_INTERFACE}

**Test Configurations:**
{chr(10).join(configs)}

Provide the complete SPIR-V binary as a hex string.
Each uint32 word must be 8 hex digits, little-endian byte order.
The entire binary is one continuous hex string with no spaces or line breaks.
"""


extraGradeAnswerRuns = list(range(1, len(SUBPASSES)))

structure = {
    "type": "object",
    "properties": {
        "reasoning": {
            "type": "string",
            "description": "Explain your approach to the prefix sum SPIR-V binary"
        },
        "spirv_hex": {
            "type": "string",
            "description": "Complete SPIR-V binary as a hex-encoded string (no spaces)"
        }
    },
    "required": ["reasoning", "spirv_hex"],
    "additionalProperties": False
}

_runner_cache = None
_REPORT_CACHE = {}


def gradeAnswer(result, subPass, aiEngineName):
  global _runner_cache

  if not result or "spirv_hex" not in result:
    return 0.0, "No SPIR-V hex provided"

  hex_str = result["spirv_hex"].strip().replace(" ", "").replace("\n", "")

  # Validate hex string
  if len(hex_str) % 2 != 0:
    return 0.0, "Hex string has odd length"

  try:
    spirv = bytes.fromhex(hex_str)
  except ValueError as e:
    return 0.0, f"Invalid hex: {e}"

  if len(spirv) < 20:
    return 0.0, f"SPIR-V binary too short ({len(spirv)} bytes)"

  # Check magic number
  magic = struct.unpack_from('<I', spirv, 0)[0]
  if magic != 0x07230203:
    return 0.0, f"Bad SPIR-V magic: {magic:#010x} (expected 0x07230203)"

  valid, err = validate_spirv(spirv)
  if not valid:
    return 0.0, f"SPIR-V validation failed: {err}"

  (n,) = SUBPASSES[subPass]

  # Generate test data
  input_arr = generate_input(n)
  ref_arr = cpu_prefix_sum(input_arr)

  input_buf = input_to_buffer(input_arr)
  output_size = n * 4
  params_buf = make_params_buffer(n)

  workgroups = ((n + 255) // 256, 1, 1)

  try:
    if _runner_cache is None:
      _runner_cache = ComputeShaderRunner()

    def verify_fn(results):
      gpu_arr = buffer_to_array(results[1], n)
      score, desc = compare_arrays(gpu_arr, ref_arr)
      _REPORT_CACHE[(aiEngineName, subPass)] = {
        "n": n,
        "input_arr": input_arr,
        "ref_arr": ref_arr,
        "gpu_arr": gpu_arr,
        "score": score,
        "desc": desc,
      }
      return score, desc

    return grade_compute(
        spirv,
        buffers={0: input_buf, 1: output_size, 2: params_buf},
        buffer_types={0: 'read', 1: 'readwrite', 2: 'uniform'},
        workgroups=workgroups,
        read_back=[1],
        verify_fn=verify_fn,
        runner=_runner_cache,
        timeout=TIMEOUT_SECONDS)

  except Exception as e:
    return 0.0, f"GPU execution failed: {e}"


# ---------------------------------------------------------------------------
# Visualization helpers
# ---------------------------------------------------------------------------

def _sparkline_svg(values, width=220, height=48, color="#22c55e", label=""):
  if len(values) == 0:
    return ""
  vmin = float(np.min(values))
  vmax = float(np.max(values))
  if vmax - vmin < 1e-9:
    vmax = vmin + 1.0
  xs = np.linspace(0, width, num=len(values))
  ys = height - (values - vmin) / (vmax - vmin) * height
  points = " ".join(f"{x:.1f},{y:.1f}" for x, y in zip(xs, ys))
  return (
    f"<svg width='{width}' height='{height}' viewBox='0 0 {width} {height}' "
    f"style='display:block;background:#0b1220;border:1px solid #1f2937;"
    f"border-radius:4px;'>"
    f"<polyline fill='none' stroke='{color}' stroke-width='1.5' "
    f"points='{points}' />"
    f"</svg>"
    f"<div style='font-size:10px;color:#94a3b8;margin-top:4px;'>{label}</div>"
  )


def _block_accuracy_svg(matches, width=300, height=18):
  if len(matches) == 0:
    return ""
  n = len(matches)
  blocks = 64
  block_size = max(1, n // blocks)
  acc = []
  for i in range(0, n, block_size):
    chunk = matches[i:i + block_size]
    acc.append(float(np.mean(chunk)))
  bar_w = width / len(acc)
  rects = []
  for i, a in enumerate(acc):
    if a >= 0.99:
      color = "#22c55e"
    elif a >= 0.9:
      color = "#84cc16"
    elif a >= 0.5:
      color = "#f59e0b"
    else:
      color = "#ef4444"
    rects.append(
      f"<rect x='{i * bar_w:.2f}' y='0' width='{bar_w + 0.2:.2f}' height='{height}' "
      f"fill='{color}' />")
  return (
    f"<svg width='{width}' height='{height}' viewBox='0 0 {width} {height}' "
    f"style='display:block;background:#0b1220;border:1px solid #1f2937;"
    f"border-radius:4px;'>" + "".join(rects) + "</svg>"
  )


def resultToNiceReport(result, subPass, aiEngineName):
  from html import escape

  report = _REPORT_CACHE.get((aiEngineName, subPass))
  if not report:
    return "<div style='color:#94a3b8;'>No visualization data captured</div>"

  n = report["n"]
  input_arr = report["input_arr"]
  ref_arr = report["ref_arr"]
  gpu_arr = report["gpu_arr"]
  score = report["score"]
  desc = escape(report["desc"])

  matches = (gpu_arr[:n] == ref_arr)
  match_count = int(np.sum(matches))
  first_bad = int(np.argmax(~matches)) if match_count < n else None

  sample_n = min(n, 256)
  input_sample = input_arr[:sample_n].astype(np.float64)
  ref_sample = ref_arr[:sample_n].astype(np.float64)
  gpu_sample = gpu_arr[:sample_n].astype(np.float64)

  input_svg = _sparkline_svg(input_sample, color="#38bdf8", label="Input values")
  ref_svg = _sparkline_svg(ref_sample, color="#22c55e", label="CPU prefix sum")
  gpu_svg = _sparkline_svg(gpu_sample, color="#f97316", label="GPU prefix sum")
  block_svg = _block_accuracy_svg(matches)

  status = "PASS" if score >= 1.0 else "PARTIAL" if score > 0 else "FAIL"
  sc = "#22c55e" if score >= 1.0 else "#f59e0b" if score > 0 else "#ef4444"

  mismatch_line = "All elements match" if first_bad is None else (
    f"First mismatch @ {first_bad}: got {gpu_arr[first_bad]}, expected {ref_arr[first_bad]}")

  return f"""
  <div style='margin:10px 0;padding:14px;border:1px solid #1f2937;
              border-radius:8px;background:#0f172a;'>
    <div style='font-weight:600;color:#e2e8f0;font-size:14px;margin-bottom:2px;'>
      Parallel Prefix Sum (inclusive scan)</div>
    <div style='font-size:12px;color:#64748b;margin-bottom:10px;'>
      N={n:,} elements · Correct {match_count:,}/{n:,} ·
      <span style='color:{sc};font-weight:700;'>{status}</span></div>

    <div style='display:flex;gap:10px;flex-wrap:wrap;align-items:flex-start;'>
      <div>{input_svg}</div>
      <div>{ref_svg}</div>
      <div>{gpu_svg}</div>
    </div>

    <div style='margin-top:10px;'>
      <div style='font-size:11px;color:#94a3b8;font-weight:600;margin-bottom:4px;'>
        Block accuracy (green = correct, red = wrong)</div>
      {block_svg}
    </div>

    <div style='margin-top:8px;font-size:11px;color:#94a3b8;'>
      {escape(mismatch_line)}</div>
    <div style='margin-top:6px;font-size:11px;color:#64748b;'>
      {desc}</div>
  </div>
  """
