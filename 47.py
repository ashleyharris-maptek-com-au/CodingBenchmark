"""
Test 47: Hash Mining / Proof of Work (WGSL Compute Shader)

The LLM must write a WGSL compute shader that searches for a uint32 nonce
such that a custom hash of (base_data XOR nonce) has D trailing zero bits.
This is a classic GPU-parallel brute-force search problem.

This tests:
1. WGSL compute shader proficiency
2. Bitwise operations and hash function implementation on GPU
3. Parallel search with result reporting (no atomics needed)
4. Performance at scale (embarrassingly parallel)

Buffer layout:
  Binding 0 (read):     Base data - 16 uint32 values to hash against
  Binding 1 (readwrite): Result buffer - uint32[4]: [found_flag, nonce, hash_lo, hash_hi]
  Binding 2 (uniform):  Params - uint32[4]: [difficulty, range_start, range_size, 0]
"""

import json
import os
import struct
import subprocess
import sys
import tempfile
import time
from typing import Tuple, Optional

import numpy as np

from shader_test_utils import compile_wgsl
from compute_test_utils import ComputeShaderRunner, grade_compute

title = "Hash Mining / Proof of Work (WGSL Compute)"

tags = [
  "wgsl",
  "structured response",
  "gpu compute",
  "algorithm design",
]

RANDOM_SEED = 47474747
TIMEOUT_SECONDS = 60

# Subpass configurations: (difficulty_bits, search_range_size)
# difficulty_bits = number of trailing zero bits required in hash output
SUBPASSES = [
  (4, 1 << 16),  # 16 trailing zero nibble - easy
  (8, 1 << 20),  # 8 trailing zero bits
  (12, 1 << 24),  # 12 trailing zero bits
  (16, 1 << 28),  # 16 trailing zero bits
  (20, 1 << 30),  # 20 trailing zero bits - hard
]


def generate_base_data(seed=RANDOM_SEED):
  """Generate 16 uint32 values as base hash input."""
  import random
  rng = random.Random(seed)
  return [rng.getrandbits(32) for _ in range(16)]


def custom_hash(data_words, nonce):
  """
  Simple but non-trivial hash function (must match GPU implementation).
  Based on a reduced-round ChaCha-like mixing function.
  Input: 16 uint32 words XORed with nonce, output: 2 uint32 words.
  """
  M = 0xFFFFFFFF
  state = list(data_words)

  # XOR nonce into all words
  for i in range(16):
    state[i] = (state[i] ^ (nonce + i)) & M

  # 8 rounds of mixing
  for _ in range(8):
    for i in range(0, 16, 4):
      a, b, c, d = state[i], state[i + 1], state[i + 2], state[i + 3]
      a = (a + b) & M
      d ^= a
      d = ((d << 16) | (d >> 16)) & M
      c = (c + d) & M
      b ^= c
      b = ((b << 12) | (b >> 20)) & M
      a = (a + b) & M
      d ^= a
      d = ((d << 8) | (d >> 24)) & M
      c = (c + d) & M
      b ^= c
      b = ((b << 7) | (b >> 25)) & M
      state[i], state[i + 1], state[i + 2], state[i + 3] = a, b, c, d

  # Fold to 2 words
  h0 = 0
  h1 = 0
  for i in range(0, 16, 2):
    h0 ^= state[i]
    h1 ^= state[i + 1]

  return h0 & M, h1 & M


def count_trailing_zeros(h0, h1):
  """Count trailing zero bits in the 64-bit hash [h0, h1] (h0 is low)."""
  if h0 == 0:
    tz = 32
    val = h1
  else:
    tz = 0
    val = h0

  if val == 0:
    return 64

  while (val & 1) == 0:
    tz += 1
    val >>= 1
  return tz


def find_nonce_cpu(base_data, difficulty, max_range):
  """CPU reference: find a nonce with required trailing zeros."""
  for nonce in range(max_range):
    h0, h1 = custom_hash(base_data, nonce)
    if count_trailing_zeros(h0, h1) >= difficulty:
      return nonce, h0, h1
  return None, 0, 0


def base_data_to_buffer(base_data):
  """Pack base data into buffer."""
  return struct.pack('16I', *base_data)


def make_result_buffer():
  """Create zeroed result buffer: [found, nonce, hash_lo, hash_hi]."""
  return struct.pack('4I', 0, 0, 0, 0)


def make_params_buffer(difficulty, range_start, range_size):
  """Create params uniform buffer."""
  return struct.pack('4I', difficulty, range_start, range_size, 0)


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

WGSL_INTERFACE = """**WGSL Compute Shader Interface:**

```wgsl
@group(0) @binding(0) var<storage, read>       baseData : array<u32, 16>;
@group(0) @binding(1) var<storage, read_write>  result   : array<u32, 4>;
@group(0) @binding(2) var<uniform>              params   : vec4<u32>;
// params.x = difficulty, params.y = range_start, params.z = range_size

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) { ... }
```

**Task:** Write a compute shader that searches for a uint32 nonce such that
the custom hash of (baseData XOR nonce) has at least `difficulty` trailing
zero bits in the 64-bit output.

**Hash function (MUST match exactly):**
```
function custom_hash(data[16], nonce) -> (h0, h1):
    state[0..15] = data[0..15]
    for i in 0..15: state[i] ^= (nonce + i)

    // 8 rounds of quarter-round mixing (ChaCha-like)
    repeat 8 times:
        for each group of 4 consecutive elements (i, i+1, i+2, i+3):
            a += b;  d ^= a;  d = rotl(d, 16)
            c += d;  b ^= c;  b = rotl(b, 12)
            a += b;  d ^= a;  d = rotl(d, 8)
            c += d;  b ^= c;  b = rotl(b, 7)

    // Fold 16 words to 2:
    h0 = state[0] ^ state[2] ^ state[4] ^ state[6] ^ state[8] ^ state[10] ^ state[12] ^ state[14]
    h1 = state[1] ^ state[3] ^ state[5] ^ state[7] ^ state[9] ^ state[11] ^ state[13] ^ state[15]
    return (h0, h1)  // h0 is low 32 bits
```

**Trailing zero count:** Count trailing zero bits of the 64-bit value [h0 | (h1 << 32)].
That is, count from h0's LSB first; if h0 == 0, add 32 and continue counting from h1.

**Algorithm:**
1. Each thread computes: my_nonce = params.y + gid.x  (range_start + thread ID)
2. Guard: if gid.x >= params.z, return (out of range)
3. Compute hash, count trailing zeros
4. If trailing_zeros >= difficulty:
   Write result[0]=1u, result[1]=nonce, result[2]=h0, result[3]=h1
   (Race writes are acceptable - any valid nonce will be verified on CPU)

**Important WGSL notes:**
- Use `var state : array<u32, 16>` for the mutable hash state.
- WGSL rotation: `(x << shift) | (x >> (32u - shift))` — all values must be u32.
- The inner quarter-round loop MUST be unrolled (write out all 4 groups
  of state[0..3], state[4..7], state[8..11], state[12..15] explicitly).
- Do NOT use atomics. Plain stores to result[] are sufficient.
- All integer literals must be typed: use `0u`, `1u`, `16u`, `32u` etc.

The shader will be dispatched with ceil(range_size / 256) workgroups.
Only ONE dispatch is performed - find the nonce in a single pass.
"""


def prepareSubpassPrompt(subPass):
  if subPass != 0:
    raise StopIteration

  configs = []
  for i, (diff, rng) in enumerate(SUBPASSES):
    configs.append(f"  Subpass {i}: difficulty={diff} bits, search_range={rng:,}")

  return f"""Write a WGSL compute shader for hash mining (proof-of-work).

{WGSL_INTERFACE}

**Test Configurations:**
{chr(10).join(configs)}

Write the complete WGSL compute shader source code.
The shader must implement the exact hash function described above.
"""


extraGradeAnswerRuns = list(range(1, len(SUBPASSES)))

structure = {
  "type": "object",
  "properties": {
    "reasoning": {
      "type": "string",
      "description": "Explain your approach to the hash mining compute shader"
    },
    "shader_code": {
      "type": "string",
      "description": "Complete WGSL compute shader source code"
    }
  },
  "required": ["reasoning", "shader_code"],
  "additionalProperties": False
}

_runner_cache = None
_base_data = None
_ref_cache = {}
_REPORT_CACHE = {}


def _grade_answer_inner(result, subPass, aiEngineName):
  global _runner_cache, _base_data, _ref_cache

  if not result or "shader_code" not in result:
    return 0.0, "No shader code provided"

  code = result["shader_code"]

  # Validate WGSL
  try:
    wgsl = compile_wgsl(code)
  except RuntimeError as e:
    return 0.0, f"WGSL compilation failed: {e}", {}

  difficulty, range_size = SUBPASSES[subPass]

  # Generate base data (same for all subpasses)
  if _base_data is None:
    _base_data = generate_base_data()

  # First verify a CPU solution exists
  cache_key = (difficulty, range_size)
  if cache_key not in _ref_cache:
    print(f"  Finding CPU reference nonce for difficulty={difficulty}...")
    nonce, h0, h1 = find_nonce_cpu(_base_data, difficulty, min(range_size, 1 << 24))
    if nonce is not None:
      _ref_cache[cache_key] = (nonce, h0, h1)
      print(f"    CPU found nonce={nonce}, hash=({h0:#010x}, {h1:#010x}), "
            f"tz={count_trailing_zeros(h0, h1)}")
    else:
      _ref_cache[cache_key] = None
      print(f"    WARNING: CPU couldn't find nonce in range")

  # Prepare buffers
  base_buf = base_data_to_buffer(_base_data)
  result_buf = make_result_buffer()
  params_buf = make_params_buffer(difficulty, 0, range_size)
  workgroups = ((range_size + 255) // 256, 1, 1)
  # Cap workgroups to reasonable maximum
  workgroups = (min(workgroups[0], 65535), 1, 1)

  # Run on GPU
  try:
    if _runner_cache is None:
      _runner_cache = ComputeShaderRunner()

    details = {
      "difficulty": difficulty,
      "range_size": range_size,
      "found": False,
      "nonce": None,
      "hash_lo": None,
      "hash_hi": None,
      "trailing_zeros": None,
      "cpu_ref": _ref_cache.get(cache_key),
    }

    def verify_fn(results):
      out = results[1]
      found, nonce, h0, h1 = struct.unpack_from('4I', out, 0)

      if found == 0:
        details.update({
          "found": False,
          "nonce": None,
          "hash_lo": None,
          "hash_hi": None,
          "trailing_zeros": None,
        })
        return 0.0, f"No nonce found (difficulty={difficulty} bits)"

      # Verify the hash on CPU
      cpu_h0, cpu_h1 = custom_hash(_base_data, nonce)
      tz = count_trailing_zeros(cpu_h0, cpu_h1)

      details.update({
        "found": True,
        "nonce": int(nonce),
        "hash_lo": int(cpu_h0),
        "hash_hi": int(cpu_h1),
        "trailing_zeros": int(tz),
      })

      if tz < difficulty:
        return 0.0, (f"Nonce {nonce} invalid: hash=({cpu_h0:#010x}, {cpu_h1:#010x}), "
                     f"trailing zeros={tz}, need {difficulty}")

      return 1.0, (f"Found valid nonce {nonce}: "
                   f"hash=({cpu_h0:#010x}, {cpu_h1:#010x}), "
                   f"trailing zeros={tz} >= {difficulty}")

    score, explanation = grade_compute(wgsl,
                         buffers={
                           0: base_buf,
                           1: result_buf,
                           2: params_buf
                         },
                         buffer_types={
                           0: 'read',
                           1: 'readwrite',
                           2: 'uniform'
                         },
                         workgroups=workgroups,
                         read_back=[1],
                         verify_fn=verify_fn,
                         runner=_runner_cache,
                         timeout=TIMEOUT_SECONDS)
    return score, explanation, details

  except Exception as e:
    return 0.0, f"GPU execution failed: {e}", {"error": str(e)}


def gradeAnswer(result, subPass, aiEngineName):
  """Run grading in an isolated subprocess to survive GPU hangs/TDRs."""
  if not result or "shader_code" not in result:
    return 0.0, "No shader code provided"

  payload = {
    "shader_code": result.get("shader_code", ""),
    "subPass": subPass,
    "aiEngineName": aiEngineName,
  }

  with tempfile.TemporaryDirectory() as tmp_dir:
    in_path = os.path.join(tmp_dir, "grade_input.json")
    out_path = os.path.join(tmp_dir, "grade_output.json")
    with open(in_path, "w", encoding="utf-8") as f:
      json.dump(payload, f)

    cmd = [sys.executable, __file__, "--grade", in_path, out_path]
    try:
      subprocess.run(
        cmd,
        check=False,
        timeout=TIMEOUT_SECONDS + 10,
        creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0,
      )
    except subprocess.TimeoutExpired:
      return 0.0, "GPU execution timed out or hung (subprocess killed)"
    except Exception as e:
      return 0.0, f"Subprocess failed: {e}"

    if not os.path.exists(out_path):
      return 0.0, "Subprocess produced no result (crash or TDR)"

    try:
      with open(out_path, "r", encoding="utf-8") as f:
        out = json.load(f)
      score = out.get("score", 0.0)
      explanation = out.get("explanation", "No explanation")
      _REPORT_CACHE[(aiEngineName, subPass)] = out.get("details", {})
      return score, explanation
    except Exception as e:
      return 0.0, f"Failed to read subprocess result: {e}"


def _run_grade_subprocess(in_path: str, out_path: str) -> int:
  try:
    with open(in_path, "r", encoding="utf-8") as f:
      payload = json.load(f)
    result = {"shader_code": payload.get("shader_code", "")}
    subPass = int(payload.get("subPass", 0))
    aiEngineName = payload.get("aiEngineName", "")
    score, explanation, details = _grade_answer_inner(result, subPass, aiEngineName)
    with open(out_path, "w", encoding="utf-8") as f:
      json.dump({"score": score, "explanation": explanation, "details": details}, f)
    return 0
  except Exception as e:
    try:
      with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"score": 0.0, "explanation": f"Subprocess error: {e}",
                   "details": {"error": str(e)}}, f)
    except Exception:
      pass
    return 1


if __name__ == "__main__":
  if len(sys.argv) >= 4 and sys.argv[1] == "--grade":
    sys.exit(_run_grade_subprocess(sys.argv[2], sys.argv[3]))


def resultToNiceReport(result, subPass, aiEngineName):
  report = _REPORT_CACHE.get((aiEngineName, subPass), {})
  difficulty = report.get("difficulty")
  range_size = report.get("range_size")
  found = report.get("found")
  nonce = report.get("nonce")
  h0 = report.get("hash_lo")
  h1 = report.get("hash_hi")
  tz = report.get("trailing_zeros")
  cpu_ref = report.get("cpu_ref")
  error = report.get("error")

  status = "VALID" if found and tz is not None and difficulty is not None and tz >= difficulty else "FAIL"
  status_color = "#22c55e" if status == "VALID" else "#f97316"

  def fmt_hex(val):
    return f"0x{val:08x}" if isinstance(val, int) else "-"

  ref_line = ""
  if isinstance(cpu_ref, (list, tuple)) and cpu_ref and cpu_ref[0] is not None:
    ref_line = (f"CPU ref nonce {cpu_ref[0]} | hash=({fmt_hex(cpu_ref[1])}, {fmt_hex(cpu_ref[2])})")
  elif cpu_ref is None:
    ref_line = "CPU ref: not found in limited search"
 
  if range_size is None:
    range_size = 0

  return f"""
  <div style='margin:10px 0;padding:10px;border:1px solid #e5e7eb;border-radius:8px;background:#f8fafc;'>
    <div style='font-weight:600;margin-bottom:4px;'>Hash mining validation</div>
    <div style='font-size:12px;color:#475569;'>
      Difficulty: {difficulty} bits · Range: {range_size:,} · Status:
      <span style='color:{status_color};font-weight:700;'>{status}</span>
    </div>
    <div style='margin-top:6px;font-size:12px;color:#0f172a;'>
      Found nonce: {nonce if nonce is not None else '-'}<br>
      Hash: ({fmt_hex(h0)}, {fmt_hex(h1)})<br>
      Trailing zeros: {tz if tz is not None else '-'} (needs {difficulty})<br>
      {ref_line}
    </div>
    {f"<div style='margin-top:6px;font-size:12px;color:#b91c1c;'>Error: {error}</div>" if error else ""}
  </div>
  """


highLevelSummary = """
<p>Write a GPU compute shader in WGSL that performs hash mining &mdash; the same kind
of brute-force search used in cryptocurrency proof-of-work. The shader must find a
number (nonce) whose hash has a specified number of trailing zero bits, searching
millions of candidates in parallel on the GPU.</p>
<p>Subpasses increase the difficulty (more trailing zeros required), demanding that the
shader correctly implement bitwise hashing and parallel search. The found nonce is
verified on the CPU.</p>
"""
