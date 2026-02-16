"""
Compute Shader Test Utilities

Shared module for GPU compute tests (46+). Provides:
- ComputeShaderRunner: dispatches compute shaders via wgpu with storage buffers
- Buffer management for input/output data
- Compilation helpers (reuses shader_test_utils for GLSL/HLSL/SPIR-V)
- Grading helpers for comparing GPU output to CPU reference
"""

import struct
import time
from typing import Optional, Tuple, List, Dict, Any, Union

import numpy as np

from shader_test_utils import (
    assemble_spirv, validate_spirv, compile_glsl, compile_hlsl
)


class ComputeShaderRunner:
  """
  Runs compute shaders on the GPU via wgpu-py (WebGPU over Vulkan/Metal/DX12).

  Usage:
      runner = ComputeShaderRunner()
      outputs = runner.run(
          spirv_binary,
          buffers={0: input_data_bytes, 1: output_size_int, 2: params_bytes},
          buffer_types={0: 'read', 1: 'readwrite', 2: 'uniform'},
          workgroups=(num_groups_x, num_groups_y, num_groups_z),
          read_back=[1],  # which bindings to read back
      )
      result = outputs[1]  # bytes
  """

  def __init__(self):
    import wgpu
    self._wgpu = wgpu

    adapter = wgpu.gpu.request_adapter_sync(power_preference="high-performance")
    if adapter is None:
      raise RuntimeError("No GPU adapter found")
    self.device = adapter.request_device_sync(
      required_limits={
        "max_storage_buffer_binding_size": 256 * 1024 * 1024,
        "max_buffer_size": 256 * 1024 * 1024,
        "max_compute_workgroup_size_x": 1024,
        "max_compute_workgroup_size_y": 1024,
        "max_compute_workgroup_size_z": 64,
        "max_compute_invocations_per_workgroup": 1024,
        "max_compute_workgroups_per_dimension": 65535,
      }
    )

  def run(self,
          spirv_binary: bytes,
          buffers: Dict[int, Union[bytes, int]],
          buffer_types: Dict[int, str],
          workgroups: Tuple[int, int, int] = (1, 1, 1),
          read_back: Optional[List[int]] = None,
          dispatches: int = 1,
          entry_point: str = "main") -> Dict[int, bytes]:
    """
    Dispatch a compute shader.

    Args:
        spirv_binary: Compiled SPIR-V binary for the compute shader
        buffers: Dict mapping binding index to either:
                 - bytes: initial data to upload
                 - int: size in bytes for an output-only buffer (zeroed)
        buffer_types: Dict mapping binding index to type string:
                      'read' = read-only storage buffer
                      'readwrite' = read-write storage buffer
                      'uniform' = uniform buffer
        workgroups: (x, y, z) workgroup counts
        read_back: List of binding indices to read back after dispatch.
                   Defaults to all 'readwrite' bindings.
        dispatches: Number of times to dispatch (for iterative simulations).
                    For dispatches > 1 with ping-pong, use run_pingpong instead.
        entry_point: Shader entry point name

    Returns:
        Dict mapping binding index to output bytes for each read-back buffer
    """
    wgpu = self._wgpu

    if read_back is None:
      read_back = [k for k, v in buffer_types.items() if v == 'readwrite']

    # Create shader module
    shader_module = self.device.create_shader_module(code=spirv_binary)

    # Create GPU buffers
    gpu_buffers = {}
    for binding, data in buffers.items():
      btype = buffer_types.get(binding, 'readwrite')
      if isinstance(data, int):
        size = data
        init_data = None
      else:
        size = len(data)
        init_data = data

      if btype == 'uniform':
        usage = wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST
      elif btype == 'read':
        usage = (wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_DST)
      else:  # readwrite
        usage = (wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_DST
                 | wgpu.BufferUsage.COPY_SRC)

      if init_data is not None:
        gpu_buffers[binding] = self.device.create_buffer_with_data(
          data=init_data, usage=usage)
      else:
        gpu_buffers[binding] = self.device.create_buffer(
          size=size, usage=usage)

    # Create bind group layout
    layout_entries = []
    for binding in sorted(buffers.keys()):
      btype = buffer_types.get(binding, 'readwrite')
      if btype == 'uniform':
        layout_entries.append({
          "binding": binding,
          "visibility": wgpu.ShaderStage.COMPUTE,
          "buffer": {"type": wgpu.BufferBindingType.uniform},
        })
      elif btype == 'read':
        layout_entries.append({
          "binding": binding,
          "visibility": wgpu.ShaderStage.COMPUTE,
          "buffer": {"type": wgpu.BufferBindingType.read_only_storage},
        })
      else:
        layout_entries.append({
          "binding": binding,
          "visibility": wgpu.ShaderStage.COMPUTE,
          "buffer": {"type": wgpu.BufferBindingType.storage},
        })

    bind_group_layout = self.device.create_bind_group_layout(entries=layout_entries)
    pipeline_layout = self.device.create_pipeline_layout(
      bind_group_layouts=[bind_group_layout])

    # Create compute pipeline
    compute_pipeline = self.device.create_compute_pipeline(
      layout=pipeline_layout,
      compute={"module": shader_module, "entry_point": entry_point},
    )

    # Create bind group
    bind_entries = []
    for binding in sorted(buffers.keys()):
      buf = gpu_buffers[binding]
      bind_entries.append({
        "binding": binding,
        "resource": {"buffer": buf, "offset": 0, "size": buf.size},
      })
    bind_group = self.device.create_bind_group(
      layout=bind_group_layout, entries=bind_entries)

    # Dispatch
    for _ in range(dispatches):
      encoder = self.device.create_command_encoder()
      compute_pass = encoder.begin_compute_pass()
      compute_pass.set_pipeline(compute_pipeline)
      compute_pass.set_bind_group(0, bind_group)
      compute_pass.dispatch_workgroups(*workgroups)
      compute_pass.end()
      self.device.queue.submit([encoder.finish()])

    # Read back results
    results = {}
    for binding in read_back:
      buf = gpu_buffers[binding]
      data = self.device.queue.read_buffer(buf)
      results[binding] = bytes(data)

    return results

  def run_pingpong(self,
                   spirv_binary: bytes,
                   buf_a_data: bytes,
                   buf_b_size: int,
                   extra_buffers: Optional[Dict[int, bytes]] = None,
                   extra_types: Optional[Dict[int, str]] = None,
                   workgroups: Tuple[int, int, int] = (1, 1, 1),
                   iterations: int = 1,
                   entry_point: str = "main") -> bytes:
    """
    Run a compute shader with ping-pong buffers for iterative simulation.

    Binding 0 = input buffer (read-only storage)
    Binding 1 = output buffer (read-write storage)
    Additional bindings from extra_buffers (e.g. binding 2 = uniform params)

    After each iteration, input and output are swapped.

    Args:
        spirv_binary: SPIR-V binary
        buf_a_data: Initial data for the ping-pong buffer
        buf_b_size: Size of the second ping-pong buffer (usually same as len(buf_a_data))
        extra_buffers: Additional buffers {binding: data_bytes}
        extra_types: Types for extra buffers {binding: 'uniform'|'read'|'readwrite'}
        workgroups: Dispatch dimensions
        iterations: Number of ping-pong iterations
        entry_point: Shader entry point

    Returns:
        Final output buffer contents as bytes
    """
    wgpu = self._wgpu

    shader_module = self.device.create_shader_module(code=spirv_binary)

    # Create ping-pong buffers
    usage_rw = (wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_DST
                | wgpu.BufferUsage.COPY_SRC)
    buf_a = self.device.create_buffer_with_data(data=buf_a_data, usage=usage_rw)
    buf_b = self.device.create_buffer(size=buf_b_size, usage=usage_rw)

    # Create extra GPU buffers
    extra_gpu = {}
    if extra_buffers:
      for binding, data in extra_buffers.items():
        btype = (extra_types or {}).get(binding, 'uniform')
        if btype == 'uniform':
          usage = wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST
        else:
          usage = wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_DST
        extra_gpu[binding] = self.device.create_buffer_with_data(data=data, usage=usage)

    # Build bind group layout
    layout_entries = [
      {"binding": 0, "visibility": wgpu.ShaderStage.COMPUTE,
       "buffer": {"type": wgpu.BufferBindingType.read_only_storage}},
      {"binding": 1, "visibility": wgpu.ShaderStage.COMPUTE,
       "buffer": {"type": wgpu.BufferBindingType.storage}},
    ]
    for binding in sorted(extra_gpu.keys()):
      btype = (extra_types or {}).get(binding, 'uniform')
      if btype == 'uniform':
        layout_entries.append({
          "binding": binding, "visibility": wgpu.ShaderStage.COMPUTE,
          "buffer": {"type": wgpu.BufferBindingType.uniform}})
      else:
        layout_entries.append({
          "binding": binding, "visibility": wgpu.ShaderStage.COMPUTE,
          "buffer": {"type": wgpu.BufferBindingType.read_only_storage}})

    bind_group_layout = self.device.create_bind_group_layout(entries=layout_entries)
    pipeline_layout = self.device.create_pipeline_layout(
      bind_group_layouts=[bind_group_layout])
    compute_pipeline = self.device.create_compute_pipeline(
      layout=pipeline_layout,
      compute={"module": shader_module, "entry_point": entry_point})

    # Helper to make bind group
    def make_bind_group(in_buf, out_buf):
      entries = [
        {"binding": 0, "resource": {"buffer": in_buf, "offset": 0, "size": in_buf.size}},
        {"binding": 1, "resource": {"buffer": out_buf, "offset": 0, "size": out_buf.size}},
      ]
      for binding in sorted(extra_gpu.keys()):
        ebuf = extra_gpu[binding]
        entries.append({"binding": binding,
                        "resource": {"buffer": ebuf, "offset": 0, "size": ebuf.size}})
      return self.device.create_bind_group(layout=bind_group_layout, entries=entries)

    # Create both bind groups
    bg_ab = make_bind_group(buf_a, buf_b)
    bg_ba = make_bind_group(buf_b, buf_a)

    # Run iterations
    for i in range(iterations):
      bg = bg_ab if (i % 2 == 0) else bg_ba
      encoder = self.device.create_command_encoder()
      compute_pass = encoder.begin_compute_pass()
      compute_pass.set_pipeline(compute_pipeline)
      compute_pass.set_bind_group(0, bg)
      compute_pass.dispatch_workgroups(*workgroups)
      compute_pass.end()
      self.device.queue.submit([encoder.finish()])

    # Read final output (in buf_b if even iterations, buf_a if odd)
    final_buf = buf_b if (iterations % 2 != 0) else buf_a
    data = self.device.queue.read_buffer(final_buf)
    return bytes(data)


def grade_compute(spirv_binary: bytes,
                  buffers: Dict[int, Union[bytes, int]],
                  buffer_types: Dict[int, str],
                  workgroups: Tuple[int, int, int],
                  read_back: List[int],
                  verify_fn,
                  runner: Optional[ComputeShaderRunner] = None,
                  timeout: float = 30.0,
                  dispatches: int = 1) -> Tuple[float, str]:
  """
  Grade a compute shader by running it and verifying output.

  Args:
      spirv_binary: SPIR-V binary
      buffers, buffer_types, workgroups, read_back: passed to runner.run()
      verify_fn: Callable(Dict[int, bytes]) -> (float, str) that checks output
      runner: Optional pre-created ComputeShaderRunner
      timeout: Time limit in seconds
      dispatches: Number of dispatches

  Returns:
      (score, explanation)
  """
  # Validate SPIR-V
  valid, err = validate_spirv(spirv_binary)
  if not valid:
    return 0.0, f"SPIR-V validation failed: {err}"

  # Run
  try:
    if runner is None:
      runner = ComputeShaderRunner()
    t0 = time.perf_counter()
    results = runner.run(spirv_binary, buffers, buffer_types, workgroups,
                         read_back, dispatches)
    elapsed = time.perf_counter() - t0
  except Exception as e:
    return 0.0, f"Compute dispatch failed: {e}"

  if elapsed > timeout:
    return 0.0, f"Timeout: {elapsed:.1f}s > {timeout:.1f}s limit"

  # Verify
  try:
    score, desc = verify_fn(results)
    return score, f"{desc} (GPU time: {elapsed:.3f}s)"
  except Exception as e:
    return 0.0, f"Verification failed: {e}"


def grade_compute_pingpong(spirv_binary: bytes,
                           initial_data: bytes,
                           extra_buffers: Optional[Dict[int, bytes]],
                           extra_types: Optional[Dict[int, str]],
                           workgroups: Tuple[int, int, int],
                           iterations: int,
                           verify_fn,
                           runner: Optional[ComputeShaderRunner] = None,
                           timeout: float = 30.0) -> Tuple[float, str]:
  """
  Grade a ping-pong compute shader (iterative simulation).

  Args:
      spirv_binary: SPIR-V binary
      initial_data: Initial buffer data
      extra_buffers: Additional uniform/read buffers
      extra_types: Types for extra buffers
      workgroups: Dispatch dimensions
      iterations: Number of simulation steps
      verify_fn: Callable(bytes) -> (float, str) that checks final output
      runner: Optional pre-created runner
      timeout: Time limit

  Returns:
      (score, explanation)
  """
  valid, err = validate_spirv(spirv_binary)
  if not valid:
    return 0.0, f"SPIR-V validation failed: {err}"

  try:
    if runner is None:
      runner = ComputeShaderRunner()
    t0 = time.perf_counter()
    result = runner.run_pingpong(
      spirv_binary, initial_data, len(initial_data),
      extra_buffers, extra_types, workgroups, iterations)
    elapsed = time.perf_counter() - t0
  except Exception as e:
    return 0.0, f"Compute dispatch failed: {e}"

  if elapsed > timeout:
    return 0.0, f"Timeout: {elapsed:.1f}s > {timeout:.1f}s limit"

  try:
    score, desc = verify_fn(result)
    return score, f"{desc} (GPU time: {elapsed:.3f}s)"
  except Exception as e:
    return 0.0, f"Verification failed: {e}"
