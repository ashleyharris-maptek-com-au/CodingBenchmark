"""
Shader Test Utilities

Shared module for tests 41-60. Provides:
- SPIR-V assembly/validation via Vulkan SDK tools (spirv-as, spirv-val)
- Offscreen GPU rendering via wgpu-py (WebGPU/Vulkan backend)
- Smart image comparison tolerant to 1px offset and minor RGB differences
- UV sphere mesh generation with positions, normals, UVs, tangents, colors
"""

import hashlib
import math
import os
import struct
import subprocess
import shutil
import tempfile
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any

import numpy as np
from PIL import Image

# ---------------------------------------------------------------------------
# SPIR-V Tools (from Vulkan SDK)
# ---------------------------------------------------------------------------

_spirv_tools_cache: Dict[str, Optional[str]] = {}


def _find_vulkan_sdk() -> Optional[str]:
  """Locate Vulkan SDK installation."""
  sdk = os.environ.get("VULKAN_SDK")
  if sdk and os.path.isdir(sdk):
    return sdk
  # Common Windows paths
  for base in [r"C:\VulkanSDK", os.path.expandvars(r"%LOCALAPPDATA%\VulkanSDK")]:
    if os.path.isdir(base):
      versions = sorted(os.listdir(base), reverse=True)
      for v in versions:
        candidate = os.path.join(base, v)
        if os.path.isdir(os.path.join(candidate, "Bin")):
          return candidate
  return None


def find_spirv_tool(name: str) -> Optional[str]:
  """Find a SPIR-V tool executable (e.g. 'spirv-as', 'spirv-val')."""
  if name in _spirv_tools_cache:
    return _spirv_tools_cache[name]

  # Try PATH first
  exe = shutil.which(name)
  if exe:
    _spirv_tools_cache[name] = exe
    return exe

  # Try Vulkan SDK
  sdk = _find_vulkan_sdk()
  if sdk:
    for subdir in ["Bin", "bin"]:
      candidate = os.path.join(sdk, subdir, name + (".exe" if os.name == "nt" else ""))
      if os.path.isfile(candidate):
        _spirv_tools_cache[name] = candidate
        return candidate

  _spirv_tools_cache[name] = None
  return None


def assemble_spirv(text: str, target_env: str = "vulkan1.0") -> bytes:
  """
  Assemble SPIR-V assembly text into SPIR-V binary using spirv-as.

  Args:
      text: SPIR-V assembly text
      target_env: Target environment (e.g. 'vulkan1.0', 'spv1.3')

  Returns:
      SPIR-V binary bytes

  Raises:
      RuntimeError: If spirv-as is not found or assembly fails
  """
  spirv_as = find_spirv_tool("spirv-as")
  if not spirv_as:
    raise RuntimeError("spirv-as not found. Install the Vulkan SDK and ensure VULKAN_SDK is set.")

  with tempfile.NamedTemporaryFile(mode="w", suffix=".spvasm", delete=False, encoding="utf-8") as f:
    f.write(text)
    src_path = f.name

  out_path = src_path + ".spv"

  try:
    result = subprocess.run([spirv_as, "--target-env", target_env, "-o", out_path, src_path],
                            capture_output=True,
                            text=True,
                            timeout=30)
    if result.returncode != 0:
      error = result.stderr or result.stdout or "Unknown error"
      raise RuntimeError(f"SPIR-V assembly failed:\n{error[:2000]}")

    with open(out_path, "rb") as f:
      return f.read()
  finally:
    for p in [src_path, out_path]:
      try:
        os.unlink(p)
      except Exception:
        pass


def validate_spirv(binary: bytes, target_env: str = "vulkan1.0") -> Tuple[bool, str]:
  """
  Validate SPIR-V binary using spirv-val.

  Returns:
      (is_valid, error_message)
  """
  spirv_val = find_spirv_tool("spirv-val")
  if not spirv_val:
    return True, "spirv-val not found, skipping validation"

  with tempfile.NamedTemporaryFile(suffix=".spv", delete=False) as f:
    f.write(binary)
    spv_path = f.name

  try:
    result = subprocess.run([spirv_val, "--target-env", target_env, spv_path],
                            capture_output=True,
                            text=True,
                            timeout=30)
    if result.returncode == 0:
      return True, ""
    return False, (result.stderr or result.stdout or "Validation failed")[:2000]
  finally:
    try:
      os.unlink(spv_path)
    except Exception:
      pass


# ---------------------------------------------------------------------------
# GLSL / HLSL Compilation (via glslangValidator from Vulkan SDK)
# ---------------------------------------------------------------------------


def compile_glsl(source: str, stage: str = "frag", target_env: str = "vulkan1.0") -> bytes:
  """
  Compile GLSL source to SPIR-V binary using glslangValidator.

  Args:
      source: GLSL shader source text
      stage: Shader stage ('vert', 'frag', 'geom', 'comp', etc.)
      target_env: Target environment ('vulkan1.0', 'vulkan1.1', etc.)

  Returns:
      SPIR-V binary bytes

  Raises:
      RuntimeError: If glslangValidator is not found or compilation fails
  """
  glslang = find_spirv_tool("glslangValidator")
  if not glslang:
    raise RuntimeError(
      "glslangValidator not found. Install the Vulkan SDK and ensure VULKAN_SDK is set.")

  with tempfile.NamedTemporaryFile(mode="w", suffix=f".{stage}", delete=False,
                                   encoding="utf-8") as f:
    f.write(source)
    src_path = f.name

  out_path = src_path + ".spv"

  try:
    result = subprocess.run(
      [glslang, "-V", "--target-env", target_env, "-S", stage, "-o", out_path, src_path],
      capture_output=True,
      text=True,
      timeout=30)
    if result.returncode != 0:
      error = result.stderr or result.stdout or "Unknown error"
      raise RuntimeError(f"GLSL compilation failed:\n{error[:2000]}")

    with open(out_path, "rb") as f:
      return f.read()
  finally:
    for p in [src_path, out_path]:
      try:
        os.unlink(p)
      except Exception:
        pass


def compile_hlsl(source: str,
                 stage: str = "frag",
                 entry_point: str = "main",
                 target_env: str = "vulkan1.0") -> bytes:
  """
  Compile HLSL source to SPIR-V binary using glslangValidator.

  Args:
      source: HLSL shader source text
      stage: Shader stage ('vert', 'frag', 'geom', 'comp', etc.)
      entry_point: Entry point function name (default 'main')
      target_env: Target environment ('vulkan1.0', 'vulkan1.1', etc.)

  Returns:
      SPIR-V binary bytes

  Raises:
      RuntimeError: If glslangValidator is not found or compilation fails
  """
  glslang = find_spirv_tool("glslangValidator")
  if not glslang:
    raise RuntimeError(
      "glslangValidator not found. Install the Vulkan SDK and ensure VULKAN_SDK is set.")

  with tempfile.NamedTemporaryFile(mode="w",
                                   suffix=f".{stage}.hlsl",
                                   delete=False,
                                   encoding="utf-8") as f:
    f.write(source)
    src_path = f.name

  out_path = src_path + ".spv"

  try:
    result = subprocess.run([
      glslang, "-V", "-D", "--target-env", target_env, "-S", stage, "-e", entry_point, "-o",
      out_path, src_path
    ],
                            capture_output=True,
                            text=True,
                            timeout=30)
    if result.returncode != 0:
      error = result.stderr or result.stdout or "Unknown error"
      raise RuntimeError(f"HLSL compilation failed:\n{error[:2000]}")

    with open(out_path, "rb") as f:
      return f.read()
  finally:
    for p in [src_path, out_path]:
      try:
        os.unlink(p)
      except Exception:
        pass


# ---------------------------------------------------------------------------
# Smart Image Comparison
# ---------------------------------------------------------------------------


def compare_images(rendered: np.ndarray,
                   reference: np.ndarray,
                   color_tolerance: int = 2,
                   spatial_tolerance: int = 1) -> Tuple[float, str]:
  """
  Smart image comparison that tolerates small spatial offsets and minor color differences.

  For each pixel in the rendered image, checks a neighborhood in the reference image.
  A pixel matches if any neighbor is within color_tolerance in all RGB channels.

  Args:
      rendered: HxWx3 or HxWx4 uint8 numpy array
      reference: HxWx3 or HxWx4 uint8 numpy array
      color_tolerance: Max per-channel difference allowed (default 2)
      spatial_tolerance: Pixel offset radius to check (default 1 = 3x3 neighborhood)

  Returns:
      (score, description) where score is 0.0 to 1.0
  """
  # Ensure RGB (drop alpha if present)
  if rendered.ndim == 3 and rendered.shape[2] == 4:
    rendered = rendered[:, :, :3]
  if reference.ndim == 3 and reference.shape[2] == 4:
    reference = reference[:, :, :3]

  if rendered.shape != reference.shape:
    return 0.0, f"Shape mismatch: rendered {rendered.shape} vs reference {reference.shape}"

  h, w, c = rendered.shape
  rendered = rendered.astype(np.int16)
  reference = reference.astype(np.int16)

  # Build padded reference for neighborhood lookup
  pad = spatial_tolerance
  ref_padded = np.pad(reference, ((pad, pad), (pad, pad), (0, 0)), mode="edge")

  matched = np.zeros((h, w), dtype=bool)

  # Check all offsets in the neighborhood
  for dy in range(-pad, pad + 1):
    for dx in range(-pad, pad + 1):
      ref_shifted = ref_padded[pad + dy:pad + dy + h, pad + dx:pad + dx + w, :]
      diff = np.abs(rendered - ref_shifted)
      pixel_ok = np.all(diff <= color_tolerance, axis=2)
      matched |= pixel_ok

  match_count = np.sum(matched)
  total = h * w
  score = float(match_count) / total

  # Also compute mean color distance for non-matching pixels for diagnostics
  if match_count < total:
    non_matched = ~matched
    nm_count = np.sum(non_matched)
    # For non-matched pixels, find their best (minimum) neighbor distance
    best_dist = np.full((h, w), 1e9)
    for dy in range(-pad, pad + 1):
      for dx in range(-pad, pad + 1):
        ref_shifted = ref_padded[pad + dy:pad + dy + h, pad + dx:pad + dx + w, :]
        dist = np.max(np.abs(rendered - ref_shifted), axis=2).astype(float)
        best_dist = np.minimum(best_dist, dist)
    avg_err = float(np.mean(best_dist[non_matched]))
    desc = (
      f"{score:.1%} pixels match (tol: color={color_tolerance}, spatial={spatial_tolerance}). "
      f"{nm_count} mismatched pixels, avg best-neighbor error: {avg_err:.1f}")
  else:
    desc = f"Perfect match (tol: color={color_tolerance}, spatial={spatial_tolerance})"

  return score, desc


# ---------------------------------------------------------------------------
# Sphere Mesh Generation
# ---------------------------------------------------------------------------


def generate_uv_sphere(
  stacks: int = 32,
  sectors: int = 64
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
  """
  Generate a UV sphere mesh.

  Returns:
      (positions, normals, uvs, tangents, colors, indices)
      - positions: (N, 3) float32 - vertex positions
      - normals:   (N, 3) float32 - vertex normals
      - uvs:       (N, 2) float32 - texture coordinates
      - tangents:  (N, 4) float32 - tangent vectors (xyz + handedness w)
      - colors:    (N, 3) float32 - per-vertex colors (varied for testing)
      - indices:   (M,)   uint32  - triangle indices
  """
  positions = []
  normals = []
  uvs = []
  tangents = []
  colors = []
  indices = []

  for i in range(stacks + 1):
    phi = math.pi * i / stacks  # 0 to pi
    v = i / stacks

    for j in range(sectors + 1):
      theta = 2.0 * math.pi * j / sectors  # 0 to 2pi
      u = j / sectors

      # Position on unit sphere
      x = math.sin(phi) * math.cos(theta)
      y = math.cos(phi)
      z = math.sin(phi) * math.sin(theta)

      positions.append([x, y, z])
      normals.append([x, y, z])  # Normal = position for unit sphere
      uvs.append([u, v])

      # Tangent: derivative with respect to theta
      tx = -math.sin(theta)
      ty = 0.0
      tz = math.cos(theta)
      length = math.sqrt(tx * tx + ty * ty + tz * tz)
      if length > 1e-6:
        tx /= length
        ty /= length
        tz /= length
      tangents.append([tx, ty, tz, 1.0])  # w=1.0 for handedness

      # Varied per-vertex color based on position
      r = 0.5 + 0.5 * x
      g = 0.5 + 0.5 * y
      b = 0.5 + 0.5 * z
      colors.append([r, g, b])

  # Generate triangle indices
  for i in range(stacks):
    for j in range(sectors):
      first = i * (sectors + 1) + j
      second = first + sectors + 1

      indices.extend([first, second, first + 1])
      indices.extend([second, second + 1, first + 1])

  return (
    np.array(positions, dtype=np.float32),
    np.array(normals, dtype=np.float32),
    np.array(uvs, dtype=np.float32),
    np.array(tangents, dtype=np.float32),
    np.array(colors, dtype=np.float32),
    np.array(indices, dtype=np.uint32),
  )


# ---------------------------------------------------------------------------
# Fixed Vertex Shader (SPIR-V assembly)
# ---------------------------------------------------------------------------

# This vertex shader:
# - Input locations: 0=position(vec3), 1=normal(vec3), 2=uv(vec2), 3=tangent(vec4), 4=color(vec3)
# - Uniform binding 0: {model(mat4), view(mat4), proj(mat4), lightPos(vec4), camPos(vec4), params(vec4)}
# - Output locations: 0=worldPos(vec3), 1=normal(vec3), 2=uv(vec2), 3=tangent(vec3), 4=color(vec3)
VERTEX_SHADER_SPIRV_ASM = """\
; SPIR-V
; Version: 1.0
; Generator: hand-written
               OpCapability Shader
               OpMemoryModel Logical GLSL450
               OpEntryPoint Vertex %main "main" %inPos %inNormal %inUV %inTangent %inColor %gl_Position %outWorldPos %outNormal %outUV %outTangent %outColor
               OpDecorate %inPos Location 0
               OpDecorate %inNormal Location 1
               OpDecorate %inUV Location 2
               OpDecorate %inTangent Location 3
               OpDecorate %inColor Location 4
               OpDecorate %gl_Position BuiltIn Position
               OpDecorate %outWorldPos Location 0
               OpDecorate %outNormal Location 1
               OpDecorate %outUV Location 2
               OpDecorate %outTangent Location 3
               OpDecorate %outColor Location 4
               OpDecorate %UBO Block
               OpMemberDecorate %UBO 0 ColMajor
               OpMemberDecorate %UBO 0 Offset 0
               OpMemberDecorate %UBO 0 MatrixStride 16
               OpMemberDecorate %UBO 1 ColMajor
               OpMemberDecorate %UBO 1 Offset 64
               OpMemberDecorate %UBO 1 MatrixStride 16
               OpMemberDecorate %UBO 2 ColMajor
               OpMemberDecorate %UBO 2 Offset 128
               OpMemberDecorate %UBO 2 MatrixStride 16
               OpMemberDecorate %UBO 3 Offset 192
               OpMemberDecorate %UBO 4 Offset 208
               OpMemberDecorate %UBO 5 Offset 224
               OpDecorate %ubo DescriptorSet 0
               OpDecorate %ubo Binding 0

       %void = OpTypeVoid
       %func = OpTypeFunction %void
      %float = OpTypeFloat 32
    %v2float = OpTypeVector %float 2
    %v3float = OpTypeVector %float 3
    %v4float = OpTypeVector %float 4
  %mat4float = OpTypeMatrix %v4float 4

 %ptr_in_v3 = OpTypePointer Input %v3float
 %ptr_in_v2 = OpTypePointer Input %v2float
 %ptr_in_v4 = OpTypePointer Input %v4float
%ptr_out_v4 = OpTypePointer Output %v4float
%ptr_out_v3 = OpTypePointer Output %v3float
%ptr_out_v2 = OpTypePointer Output %v2float

      %inPos = OpVariable %ptr_in_v3 Input
   %inNormal = OpVariable %ptr_in_v3 Input
       %inUV = OpVariable %ptr_in_v2 Input
  %inTangent = OpVariable %ptr_in_v4 Input
    %inColor = OpVariable %ptr_in_v3 Input

%gl_Position = OpVariable %ptr_out_v4 Output
 %outWorldPos = OpVariable %ptr_out_v3 Output
  %outNormal = OpVariable %ptr_out_v3 Output
      %outUV = OpVariable %ptr_out_v2 Output
 %outTangent = OpVariable %ptr_out_v3 Output
  %outColor  = OpVariable %ptr_out_v3 Output

        %UBO = OpTypeStruct %mat4float %mat4float %mat4float %v4float %v4float %v4float
    %ptr_ubo = OpTypePointer Uniform %UBO
        %ubo = OpVariable %ptr_ubo Uniform

       %uint = OpTypeInt 32 0
        %int = OpTypeInt 32 1
      %int_0 = OpConstant %int 0
      %int_1 = OpConstant %int 1
      %int_2 = OpConstant %int 2
    %float_0 = OpConstant %float 0
    %float_1 = OpConstant %float 1

 %ptr_u_mat4 = OpTypePointer Uniform %mat4float

       %main = OpFunction %void None %func
      %entry = OpLabel

               ; Load vertex attributes
       %pos3 = OpLoad %v3float %inPos
       %norm = OpLoad %v3float %inNormal
         %uv = OpLoad %v2float %inUV
       %tang = OpLoad %v4float %inTangent
        %col = OpLoad %v3float %inColor

               ; Load matrices
  %model_ptr = OpAccessChain %ptr_u_mat4 %ubo %int_0
      %model = OpLoad %mat4float %model_ptr
   %view_ptr = OpAccessChain %ptr_u_mat4 %ubo %int_1
       %view = OpLoad %mat4float %view_ptr
   %proj_ptr = OpAccessChain %ptr_u_mat4 %ubo %int_2
       %proj = OpLoad %mat4float %proj_ptr

               ; Compute world position: model * vec4(pos, 1.0)
       %pos4 = OpCompositeConstruct %v4float %pos3 %float_1
   %worldPos = OpMatrixTimesVector %v4float %model %pos4
  %viewPos   = OpMatrixTimesVector %v4float %view %worldPos
   %clipPos  = OpMatrixTimesVector %v4float %proj %viewPos

               ; Store gl_Position
               OpStore %gl_Position %clipPos

               ; Pass world position (xyz of worldPos)
  %wp3       = OpVectorShuffle %v3float %worldPos %worldPos 0 1 2
               OpStore %outWorldPos %wp3

               ; Pass normal (assume no non-uniform scale for simplicity)
  %norm4     = OpCompositeConstruct %v4float %norm %float_0
  %wNorm4    = OpMatrixTimesVector %v4float %model %norm4
  %wNorm3    = OpVectorShuffle %v3float %wNorm4 %wNorm4 0 1 2
               OpStore %outNormal %wNorm3

               ; Pass UV
               OpStore %outUV %uv

               ; Pass tangent xyz
  %tang3     = OpVectorShuffle %v3float %tang %tang 0 1 2
               OpStore %outTangent %tang3

               ; Pass color
               OpStore %outColor %col

               OpReturn
               OpFunctionEnd
"""

# ---------------------------------------------------------------------------
# Uniform Buffer Layout
# ---------------------------------------------------------------------------
# struct UBO {
#   mat4 model;       // offset 0   (64 bytes)
#   mat4 view;        // offset 64  (64 bytes)
#   mat4 projection;  // offset 128 (64 bytes)
#   vec4 lightPos;    // offset 192 (16 bytes)
#   vec4 cameraPos;   // offset 208 (16 bytes)
#   vec4 params;      // offset 224 (16 bytes)  -- (time, roughness, metalness, 0)
# };
# Total: 240 bytes

UBO_SIZE = 240


def _mat4_identity() -> np.ndarray:
  return np.eye(4, dtype=np.float32)


def _mat4_perspective(fov_y: float, aspect: float, near: float, far: float) -> np.ndarray:
  """Create a perspective projection matrix (Vulkan clip space: y-down, z 0..1)."""
  f = 1.0 / math.tan(fov_y / 2.0)
  m = np.zeros((4, 4), dtype=np.float32)
  m[0, 0] = f / aspect
  m[1, 1] = -f  # Vulkan y-flip
  m[2, 2] = far / (near - far)
  m[2, 3] = (near * far) / (near - far)
  m[3, 2] = -1.0
  return m


def _mat4_look_at(eye: np.ndarray, target: np.ndarray, up: np.ndarray) -> np.ndarray:
  """Create a look-at view matrix."""
  f = target - eye
  f = f / np.linalg.norm(f)
  s = np.cross(f, up)
  s = s / np.linalg.norm(s)
  u = np.cross(s, f)

  m = np.eye(4, dtype=np.float32)
  m[0, :3] = s
  m[1, :3] = u
  m[2, :3] = -f
  m[0, 3] = -np.dot(s, eye)
  m[1, 3] = -np.dot(u, eye)
  m[2, 3] = np.dot(f, eye)
  return m


def build_default_ubo(width: int = 512, height: int = 512) -> bytes:
  """Build the default uniform buffer for the standard sphere view."""
  model = _mat4_identity()
  eye = np.array([0.0, 0.0, 3.0], dtype=np.float32)
  target = np.array([0.0, 0.0, 0.0], dtype=np.float32)
  up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
  view = _mat4_look_at(eye, target, up)
  proj = _mat4_perspective(math.radians(45.0), width / height, 0.1, 100.0)

  light_pos = np.array([5.0, 5.0, 5.0, 1.0], dtype=np.float32)
  cam_pos = np.array([0.0, 0.0, 3.0, 1.0], dtype=np.float32)
  params = np.array([0.0, 0.5, 0.0, 0.0], dtype=np.float32)

  # Pack as column-major matrices (numpy default is row-major, transpose for column-major)
  data = bytearray()
  data += model.T.tobytes()  # 64 bytes
  data += view.T.tobytes()  # 64 bytes
  data += proj.T.tobytes()  # 64 bytes
  data += light_pos.tobytes()  # 16 bytes
  data += cam_pos.tobytes()  # 16 bytes
  data += params.tobytes()  # 16 bytes
  assert len(data) == UBO_SIZE
  return bytes(data)


# ---------------------------------------------------------------------------
# wgpu Offscreen Renderer
# ---------------------------------------------------------------------------


class ShaderRenderer:
  """
  Offscreen renderer using wgpu-py (WebGPU over Vulkan/Metal/DX12).
  Renders a UV sphere with a custom fragment shader.
  """

  def __init__(self, width: int = 512, height: int = 512):
    import wgpu

    self.width = width
    self.height = height
    self._wgpu = wgpu

    # Request adapter and device
    adapter = wgpu.gpu.request_adapter_sync(power_preference="high-performance")
    if adapter is None:
      raise RuntimeError("No GPU adapter found")
    self.device = adapter.request_device_sync()

    # Generate sphere mesh
    positions, normals, uvs, tangents, colors, indices = generate_uv_sphere()
    self._index_count = len(indices)

    # Interleave vertex data: pos(3) + normal(3) + uv(2) + tangent(4) + color(3) = 15 floats
    n_verts = len(positions)
    interleaved = np.zeros((n_verts, 15), dtype=np.float32)
    interleaved[:, 0:3] = positions
    interleaved[:, 3:6] = normals
    interleaved[:, 6:8] = uvs
    interleaved[:, 8:12] = tangents
    interleaved[:, 12:15] = colors
    self._vertex_stride = 15 * 4  # 60 bytes

    # Create GPU buffers
    self._vertex_buffer = self.device.create_buffer_with_data(data=interleaved.tobytes(),
                                                              usage=wgpu.BufferUsage.VERTEX)
    self._index_buffer = self.device.create_buffer_with_data(data=indices.tobytes(),
                                                             usage=wgpu.BufferUsage.INDEX)

    # Uniform buffer
    self._ubo_data = build_default_ubo(width, height)
    self._uniform_buffer = self.device.create_buffer_with_data(data=self._ubo_data,
                                                               usage=wgpu.BufferUsage.UNIFORM
                                                               | wgpu.BufferUsage.COPY_DST)

    # Create bind group layout
    self._bind_group_layout = self.device.create_bind_group_layout(
      entries=[{
        "binding": 0,
        "visibility": wgpu.ShaderStage.VERTEX | wgpu.ShaderStage.FRAGMENT,
        "buffer": {
          "type": wgpu.BufferBindingType.uniform
        },
      }])

    # Create bind group
    self._bind_group = self.device.create_bind_group(layout=self._bind_group_layout,
                                                     entries=[{
                                                       "binding": 0,
                                                       "resource": {
                                                         "buffer": self._uniform_buffer,
                                                         "offset": 0,
                                                         "size": UBO_SIZE
                                                       },
                                                     }])

    # Pipeline layout
    self._pipeline_layout = self.device.create_pipeline_layout(
      bind_group_layouts=[self._bind_group_layout])

    # Create render target texture
    self._color_texture = self.device.create_texture(
      size=(width, height, 1),
      format=wgpu.TextureFormat.rgba8unorm,
      usage=wgpu.TextureUsage.RENDER_ATTACHMENT | wgpu.TextureUsage.COPY_SRC,
    )
    self._color_view = self._color_texture.create_view()

    # Depth texture
    self._depth_texture = self.device.create_texture(
      size=(width, height, 1),
      format=wgpu.TextureFormat.depth24plus,
      usage=wgpu.TextureUsage.RENDER_ATTACHMENT,
    )
    self._depth_view = self._depth_texture.create_view()

    # Readback buffer
    # RGBA8 = 4 bytes per pixel, rows aligned to 256 bytes
    self._bytes_per_row = ((width * 4 + 255) // 256) * 256
    self._readback_buffer = self.device.create_buffer(
      size=self._bytes_per_row * height,
      usage=wgpu.BufferUsage.COPY_DST | wgpu.BufferUsage.COPY_SRC,
    )

    # Assemble the fixed vertex shader
    self._vert_spirv = assemble_spirv(VERTEX_SHADER_SPIRV_ASM)

  def render(self, frag_spirv_bytes: bytes) -> np.ndarray:
    """
    Render the sphere with the given fragment shader SPIR-V binary.

    Args:
        frag_spirv_bytes: Compiled SPIR-V binary for the fragment shader

    Returns:
        HxWx4 uint8 numpy array (RGBA)
    """
    wgpu = self._wgpu

    # Create shader modules
    vert_module = self.device.create_shader_module(code=self._vert_spirv)
    frag_module = self.device.create_shader_module(code=frag_spirv_bytes)

    # Vertex buffer layout
    vertex_attrs = [
      {
        "format": wgpu.VertexFormat.float32x3,
        "offset": 0,
        "shader_location": 0
      },  # position
      {
        "format": wgpu.VertexFormat.float32x3,
        "offset": 12,
        "shader_location": 1
      },  # normal
      {
        "format": wgpu.VertexFormat.float32x2,
        "offset": 24,
        "shader_location": 2
      },  # uv
      {
        "format": wgpu.VertexFormat.float32x4,
        "offset": 32,
        "shader_location": 3
      },  # tangent
      {
        "format": wgpu.VertexFormat.float32x3,
        "offset": 48,
        "shader_location": 4
      },  # color
    ]

    # Create render pipeline
    pipeline = self.device.create_render_pipeline(
      layout=self._pipeline_layout,
      vertex={
        "module":
        vert_module,
        "entry_point":
        "main",
        "buffers": [{
          "array_stride": self._vertex_stride,
          "step_mode": wgpu.VertexStepMode.vertex,
          "attributes": vertex_attrs,
        }],
      },
      primitive={
        "topology": wgpu.PrimitiveTopology.triangle_list,
        "cull_mode": wgpu.CullMode.back,
        "front_face": wgpu.FrontFace.ccw,
      },
      depth_stencil={
        "format": wgpu.TextureFormat.depth24plus,
        "depth_write_enabled": True,
        "depth_compare": wgpu.CompareFunction.less,
      },
      fragment={
        "module": frag_module,
        "entry_point": "main",
        "targets": [{
          "format": wgpu.TextureFormat.rgba8unorm
        }],
      },
    )

    # Record commands
    encoder = self.device.create_command_encoder()
    render_pass = encoder.begin_render_pass(
      color_attachments=[{
        "view": self._color_view,
        "resolve_target": None,
        "clear_value": (0.0, 0.0, 0.0, 1.0),
        "load_op": wgpu.LoadOp.clear,
        "store_op": wgpu.StoreOp.store,
      }],
      depth_stencil_attachment={
        "view": self._depth_view,
        "depth_clear_value": 1.0,
        "depth_load_op": wgpu.LoadOp.clear,
        "depth_store_op": wgpu.StoreOp.store,
      },
    )

    render_pass.set_pipeline(pipeline)
    render_pass.set_bind_group(0, self._bind_group)
    render_pass.set_vertex_buffer(0, self._vertex_buffer)
    render_pass.set_index_buffer(self._index_buffer, wgpu.IndexFormat.uint32)
    render_pass.draw_indexed(self._index_count)
    render_pass.end()

    # Copy texture to readback buffer
    encoder.copy_texture_to_buffer(
      {
        "texture": self._color_texture,
        "mip_level": 0,
        "origin": (0, 0, 0)
      },
      {
        "buffer": self._readback_buffer,
        "offset": 0,
        "bytes_per_row": self._bytes_per_row
      },
      (self.width, self.height, 1),
    )

    self.device.queue.submit([encoder.finish()])

    # Read back pixels
    data = self.device.queue.read_buffer(self._readback_buffer)
    # Parse into image array (accounting for row alignment)
    img = np.zeros((self.height, self.width, 4), dtype=np.uint8)
    for y in range(self.height):
      row_start = y * self._bytes_per_row
      row_data = data[row_start:row_start + self.width * 4]
      img[y] = np.frombuffer(row_data, dtype=np.uint8).reshape(self.width, 4)

    return img

  def render_to_image(self, frag_spirv_bytes: bytes) -> Image.Image:
    """Render and return a PIL Image."""
    pixels = self.render(frag_spirv_bytes)
    return Image.fromarray(pixels, "RGBA")


# ---------------------------------------------------------------------------
# Reference Image Management
# ---------------------------------------------------------------------------

REFERENCE_DIR = Path(__file__).parent / "reference_images"


def get_reference_path(test_num: int, subpass: int) -> Path:
  """Get path to a reference image."""
  return REFERENCE_DIR / str(test_num) / f"subpass_{subpass:02d}.png"


def save_reference(img: np.ndarray, test_num: int, subpass: int):
  """Save a reference image."""
  path = get_reference_path(test_num, subpass)
  path.parent.mkdir(parents=True, exist_ok=True)
  if img.ndim == 3 and img.shape[2] == 4:
    Image.fromarray(img, "RGBA").save(str(path))
  else:
    Image.fromarray(img, "RGB").save(str(path))


def load_reference(test_num: int, subpass: int) -> Optional[np.ndarray]:
  """Load a reference image. Returns None if not found."""
  path = get_reference_path(test_num, subpass)
  if not path.exists():
    return None
  img = Image.open(str(path))
  return np.array(img)


# ---------------------------------------------------------------------------
# High-level test helpers
# ---------------------------------------------------------------------------


def grade_shader(frag_spirv_text: str,
                 test_num: int,
                 subpass: int,
                 renderer: Optional['ShaderRenderer'] = None,
                 width: int = 512,
                 height: int = 512,
                 color_tolerance: int = 2,
                 spatial_tolerance: int = 1) -> Tuple[float, str]:
  """
  Grade a fragment shader by assembling, rendering, and comparing to reference.

  Args:
      frag_spirv_text: SPIR-V assembly text for the fragment shader
      test_num: Test number (for reference image lookup)
      subpass: Subpass number
      renderer: Optional pre-created ShaderRenderer (for reuse)
      width, height: Render resolution
      color_tolerance, spatial_tolerance: Comparison tolerances

  Returns:
      (score, explanation)
  """
  # Step 1: Assemble
  try:
    frag_spirv = assemble_spirv(frag_spirv_text)
  except RuntimeError as e:
    return 0.0, f"SPIR-V assembly failed: {e}"

  return grade_shader_binary(frag_spirv, test_num, subpass, renderer, width, height,
                             color_tolerance, spatial_tolerance)


def grade_shader_binary(frag_spirv: bytes,
                        test_num: int,
                        subpass: int,
                        renderer: Optional['ShaderRenderer'] = None,
                        width: int = 512,
                        height: int = 512,
                        color_tolerance: int = 2,
                        spatial_tolerance: int = 1,
                        ref_test_num: Optional[int] = None) -> Tuple[float, str]:
  """
  Grade a pre-compiled fragment shader SPIR-V binary by rendering and comparing.

  Args:
      frag_spirv: SPIR-V binary bytes for the fragment shader
      test_num: Test number (for reference image lookup, unless ref_test_num set)
      subpass: Subpass number
      renderer: Optional pre-created ShaderRenderer (for reuse)
      width, height: Render resolution
      color_tolerance, spatial_tolerance: Comparison tolerances
      ref_test_num: If set, look up reference images from this test number instead

  Returns:
      (score, explanation)
  """
  # Step 1: Validate
  valid, err = validate_spirv(frag_spirv)
  if not valid:
    return 0.0, f"SPIR-V validation failed: {err}"

  # Step 2: Render
  try:
    if renderer is None:
      renderer = ShaderRenderer(width, height)
    pixels = renderer.render(frag_spirv)
  except Exception as e:
    return 0.0, f"Rendering failed: {e}"

  # Step 3: Compare to reference
  lookup_test = ref_test_num if ref_test_num is not None else test_num
  reference = load_reference(lookup_test, subpass)
  if reference is None:
    # No reference - save this as reference and give full marks (first run)
    save_reference(pixels, lookup_test, subpass)
    return 1.0, "No reference image found - saved current render as reference (assumed correct)"

  score, desc = compare_images(pixels, reference, color_tolerance, spatial_tolerance)
  return score, desc
