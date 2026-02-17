"""
Test 41: SPIR-V Assembly Fragment Shaders

The LLM must write fragment shaders in SPIR-V assembly text.
Each subpass is a different shader task, all rendered on the same unit sphere
with the same vertex shader and uniform buffer.

Available fragment shader inputs:
  layout(location = 0) in vec3 worldPos;
  layout(location = 1) in vec3 normal;    -- interpolated world-space normal
  layout(location = 2) in vec2 uv;        -- texture coordinates [0,1]
  layout(location = 3) in vec3 tangent;   -- tangent vector
  layout(location = 4) in vec3 color;     -- per-vertex color (varies across sphere)

Available uniform buffer (set=0, binding=0):
  mat4 model;       // offset 0
  mat4 view;        // offset 64
  mat4 projection;  // offset 128
  vec4 lightPos;    // offset 192  (5, 5, 5, 1)
  vec4 cameraPos;   // offset 208  (0, 0, 3, 1)
  vec4 params;      // offset 224  (0, 0.5, 0, 0)

Fragment output:
  layout(location = 0) out vec4 outColor;  -- RGBA, format rgba8unorm
"""

import os
import sys
from typing import Tuple, Optional

from shader_test_utils import (
  ShaderRenderer, assemble_spirv, validate_spirv,
  compare_images, load_reference, save_reference, grade_shader
)

title = "SPIR-V Assembly Fragment Shaders"

# ---------------------------------------------------------------------------
# Common SPIR-V preamble description for prompts
# ---------------------------------------------------------------------------

SPIRV_INTERFACE_DESC = """\
You are writing a SPIR-V assembly text fragment shader. The shader will be assembled
with spirv-as (Vulkan 1.0 target) and rendered on a unit sphere.

**Available inputs (from vertex shader):**
```
layout(location = 0) in vec3 worldPos;   ; world-space position
layout(location = 1) in vec3 normal;     ; world-space normal (interpolated, may need normalize)
layout(location = 2) in vec2 uv;         ; texture coordinates in [0,1]
layout(location = 3) in vec3 tangent;    ; tangent vector
layout(location = 4) in vec3 color;      ; per-vertex color (smoothly varies: typical value: rgb = 0.5+0.5*xyz)
```

**Available uniform buffer (DescriptorSet 0, Binding 0):**
```
struct UBO {           // std140 layout
  mat4 model;          // offset 0   (column-major)
  mat4 view;           // offset 64
  mat4 projection;     // offset 128
  vec4 lightPos;       // offset 192  -- typical value: (5.0, 5.0, 5.0, 1.0)
  vec4 cameraPos;      // offset 208  -- typical value: (0.0, 0.0, 3.0, 1.0)
  vec4 params;         // offset 224  -- typical value: (0.0, 0.5, 0.0, 0.0)
};
```

**Required output:**
```
layout(location = 0) out vec4 outColor;  RGBA float, will be stored as rgba8unorm
```

**SPIR-V requirements:**
- Start with `; SPIR-V` header comment and version
- Use `OpCapability Shader` and `OpMemoryModel Logical GLSL450`
- Use `OpEntryPoint Fragment %main "main"` listing ALL input/output variables used
- Use `OpExecutionMode %main OriginUpperLeft`
- Declare all necessary decorations (Location, DescriptorSet, Binding, Block, Offsets)
- All input variables are `Input` storage class, output is `Output` storage class
- The uniform buffer variable is `Uniform` storage class
- Entry point function name must be "main"

**Write ONLY the complete SPIR-V assembly text.** Do not include any other text or explanation
outside the spirv_code field.
"""

# ---------------------------------------------------------------------------
# Subpass definitions
# ---------------------------------------------------------------------------

SUBPASSES = [
  # 0: Solid color
  {
    "description": "Solid Red",
    "prompt": f"""{SPIRV_INTERFACE_DESC}

**Task:** Output a solid red color: outColor = vec4(1.0, 0.0, 0.0, 1.0).
This is the simplest possible fragment shader. Just store the constant color.
""",
  },

  # 1: Normal visualization
  {
    "description": "Normal to RGB",
    "prompt": f"""{SPIRV_INTERFACE_DESC}

**Task:** Visualize the surface normal as color.
Map the normal from [-1,1] to [0,1] range: outColor.rgb = normal * 0.5 + 0.5, outColor.a = 1.0.
You must normalize the interpolated normal first.
""",
  },

  # 2: UV visualization
  {
    "description": "UV to RG",
    "prompt": f"""{SPIRV_INTERFACE_DESC}

**Task:** Visualize texture coordinates as color.
outColor = vec4(uv.x, uv.y, 0.0, 1.0).
""",
  },

  # 3: Vertex color passthrough
  {
    "description": "Vertex Color",
    "prompt": f"""{SPIRV_INTERFACE_DESC}

**Task:** Pass through the per-vertex color directly.
outColor = vec4(color.r, color.g, color.b, 1.0).
""",
  },

  # 4: Lambertian diffuse
  {
    "description": "Lambertian Diffuse",
    "prompt": f"""{SPIRV_INTERFACE_DESC}

**Task:** Implement Lambertian diffuse lighting.
1. Normalize the interpolated normal.
2. Compute light direction: L = normalize(lightPos.xyz - worldPos).
3. Compute diffuse factor: d = max(dot(N, L), 0.0).
4. Output: outColor = vec4(d, d, d, 1.0) (white light, grey sphere).

You will need OpExtInstImport "GLSL.std" for normalize, and use OpDot.
Use OpExtInst to call GLSL normalize (instruction 69) and FMax (instruction 40).
""",
  },

  # 5: Phong specular
  {
    "description": "Phong Specular",
    "prompt": f"""{SPIRV_INTERFACE_DESC}

**Task:** Implement Phong specular highlights.
1. Normalize the normal.
2. Compute light direction L = normalize(lightPos.xyz - worldPos).
3. Compute view direction V = normalize(cameraPos.xyz - worldPos).
4. Compute reflection R = reflect(-L, N).  (reflect = I - 2*dot(N,I)*N where I=-L)
5. Specular = pow(max(dot(R, V), 0.0), 32.0).
6. Diffuse = max(dot(N, L), 0.0).
7. outColor = vec4(0.2 + 0.5*diffuse + 0.8*specular) for each RGB, alpha = 1.0.

Use GLSL.std.450 extended instructions: Normalize(69), Reflect(71), Pow(26), FMax(40).
""",
  },

  # 6: Blinn-Phong
  {
    "description": "Blinn-Phong",
    "prompt": f"""{SPIRV_INTERFACE_DESC}

**Task:** Implement Blinn-Phong shading.
1. N = normalize(normal), L = normalize(lightPos.xyz - worldPos), V = normalize(cameraPos.xyz - worldPos).
2. H = normalize(L + V) (half-vector).
3. diffuse = max(dot(N, L), 0.0).
4. specular = pow(max(dot(N, H), 0.0), 64.0).
5. outColor = vec4(vec3(0.1 + 0.6*diffuse + 0.8*specular), 1.0).

Use GLSL.std.450: Normalize(69), Pow(26), FMax(40).
""",
  },

  # 7: Rim/Fresnel lighting
  {
    "description": "Rim Lighting",
    "prompt": f"""{SPIRV_INTERFACE_DESC}

**Task:** Implement rim (Fresnel) lighting effect.
1. N = normalize(normal), V = normalize(cameraPos.xyz - worldPos).
2. rim = 1.0 - max(dot(N, V), 0.0).
3. rim = pow(rim, 3.0) to sharpen the rim.
4. Combine with basic diffuse: L = normalize(lightPos.xyz - worldPos), diff = max(dot(N,L), 0.0).
5. Base color = vec3(0.2, 0.3, 0.8) (blue).
6. outColor = vec4(base * (0.3 + 0.5*diff) + vec3(1.0, 0.8, 0.5) * rim, 1.0).

Use GLSL.std.450: Normalize(69), Pow(26), FMax(40).
""",
  },

  # 8: Toon/cel shading
  {
    "description": "Toon Shading",
    "prompt": f"""{SPIRV_INTERFACE_DESC}

**Task:** Implement toon/cel shading with quantized lighting.
1. N = normalize(normal), L = normalize(lightPos.xyz - worldPos).
2. d = max(dot(N, L), 0.0).
3. Quantize to 4 bands: if d > 0.75 then shade = 1.0, else if d > 0.5 then shade = 0.7,
   else if d > 0.25 then shade = 0.4, else shade = 0.2.
4. Use vertex color as base: outColor = vec4(color * shade, 1.0).

Use OpSelect for conditional selection, or chain of mix operations.
The quantization can be done with floor(d * 4.0) / 4.0 as an alternative.
""",
  },

  # 9: Checkerboard pattern
  {
    "description": "Checkerboard",
    "prompt": f"""{SPIRV_INTERFACE_DESC}

**Task:** Render a checkerboard pattern using UV coordinates.
1. Scale UVs: su = uv.x * 8.0, sv = uv.y * 8.0.
2. checker = floor(su) + floor(sv).
3. If checker is even (fract(checker*0.5) < 0.25), use color1 = vec3(0.9, 0.9, 0.9).
   Otherwise use color2 = vec3(0.2, 0.2, 0.2).
4. Add slight diffuse shading: N = normalize(normal), L = normalize(lightPos.xyz - worldPos),
   shade = 0.5 + 0.5 * max(dot(N,L), 0.0).
5. outColor = vec4(checkerColor * shade, 1.0).

Use GLSL.std.450: Floor(8), Normalize(69), FMax(40).
""",
  },

  # 10: Procedural stripes
  {
    "description": "Horizontal Stripes",
    "prompt": f"""{SPIRV_INTERFACE_DESC}

**Task:** Render horizontal stripes using the UV v-coordinate.
1. stripe = sin(uv.y * 3.14159 * 16.0).
2. If stripe > 0.0, use color1 = vec3(0.9, 0.1, 0.1) (red).
   If stripe <= 0.0, use color2 = vec3(0.1, 0.1, 0.9) (blue).
3. Add diffuse: shade = 0.4 + 0.6 * max(dot(normalize(normal), normalize(lightPos.xyz - worldPos)), 0.0).
4. outColor = vec4(stripeColor * shade, 1.0).

Use GLSL.std.450: Sin(13), Normalize(69), FMax(40).
""",
  },

  # 11: Procedural dots/circles
  {
    "description": "Polka Dots",
    "prompt": f"""{SPIRV_INTERFACE_DESC}

**Task:** Render a polka dot pattern on the sphere using UV coordinates.
1. Scale UVs: su = uv.x * 6.0, sv = uv.y * 6.0.
2. Get fractional part: fu = fract(su) - 0.5, fv = fract(sv) - 0.5.
3. dist = sqrt(fu*fu + fv*fv).
4. If dist < 0.3, use dot color vec3(1.0, 0.8, 0.0) (yellow).
   Otherwise use background vec3(0.1, 0.1, 0.3) (dark blue).
5. Add diffuse shading: shade = 0.3 + 0.7*max(dot(N,L), 0.0).
6. outColor = vec4(chosenColor * shade, 1.0).

Use GLSL.std.450: Fract(10), Sqrt(31), Normalize(69), FMax(40).
""",
  },

  # 12: Hemisphere lighting
  {
    "description": "Hemisphere Lighting",
    "prompt": f"""{SPIRV_INTERFACE_DESC}

**Task:** Implement hemisphere (ambient) lighting.
1. N = normalize(normal).
2. skyColor = vec3(0.4, 0.6, 1.0) (light blue).
3. groundColor = vec3(0.3, 0.15, 0.05) (brown).
4. blend = N.y * 0.5 + 0.5 (map from [-1,1] to [0,1]).
5. ambient = mix(groundColor, skyColor, blend) = groundColor*(1-blend) + skyColor*blend.
6. Add slight directional: L = normalize(lightPos.xyz - worldPos), d = max(dot(N,L),0.0).
7. outColor = vec4(ambient + vec3(0.3)*d, 1.0).

Use GLSL.std.450: Normalize(69), FMax(40), FMix(46).
""",
  },

  # 13: World-space gradient
  {
    "description": "Y-Gradient",
    "prompt": f"""{SPIRV_INTERFACE_DESC}

**Task:** Color the sphere based on world-space Y position.
1. t = worldPos.y * 0.5 + 0.5 (map from [-1,1] to [0,1]).
2. Clamp t to [0,1]: t = clamp(t, 0.0, 1.0).
3. Bottom color = vec3(0.1, 0.4, 0.1) (green).
4. Top color = vec3(1.0, 1.0, 1.0) (white, like snow).
5. outColor = vec4(mix(bottomColor, topColor, t), 1.0).

Use GLSL.std.450: FClamp(43) or FMin/FMax, FMix(46).
""",
  },

  # 14: Distance-based fog
  {
    "description": "Fog Effect",
    "prompt": f"""{SPIRV_INTERFACE_DESC}

**Task:** Apply distance-based fog to a lit sphere.
1. Base lighting: N = normalize(normal), L = normalize(lightPos.xyz - worldPos).
   diff = max(dot(N,L), 0.0). baseColor = vec3(0.8, 0.2, 0.2) * (0.3 + 0.7*diff).
2. Compute distance from camera: dist = length(worldPos - cameraPos.xyz).
3. fogFactor = clamp((dist - 1.5) / 3.0, 0.0, 1.0).
   (fog starts at distance 1.5, fully fogged at 4.5)
4. fogColor = vec3(0.7, 0.7, 0.8).
5. outColor = vec4(mix(baseColor, fogColor, fogFactor), 1.0).

Use GLSL.std.450: Normalize(69), Length(66), FClamp(43), FMax(40), FMix(46).
""",
  },

  # 15: Gooch shading
  {
    "description": "Gooch Shading",
    "prompt": f"""{SPIRV_INTERFACE_DESC}

**Task:** Implement Gooch warm/cool shading.
1. N = normalize(normal), L = normalize(lightPos.xyz - worldPos).
2. t = (dot(N, L) + 1.0) * 0.5 (remap from [-1,1] to [0,1]).
3. coolColor = vec3(0.0, 0.0, 0.55) + 0.25 * vec3(0.8, 0.8, 0.8).
4. warmColor = vec3(0.3, 0.3, 0.0) + 0.5 * vec3(0.8, 0.8, 0.8).
   (So cool = (0.2, 0.2, 0.75), warm = (0.7, 0.7, 0.4))
5. outColor = vec4(mix(coolColor, warmColor, t), 1.0).

Use GLSL.std.450: Normalize(69), FMix(46).
""",
  },

  # 16: Mandelbrot on UVs
  {
    "description": "Mandelbrot Set",
    "prompt": f"""{SPIRV_INTERFACE_DESC}

**Task:** Render the Mandelbrot set mapped onto the sphere using UV coordinates.
1. Map UVs to complex plane: c_real = uv.x * 3.0 - 2.0, c_imag = uv.y * 2.0 - 1.0.
2. Iterate z = z*z + c starting from z = 0+0i, for up to 20 iterations.
   z_real_new = z_real*z_real - z_imag*z_imag + c_real
   z_imag_new = 2.0*z_real*z_imag + c_imag
   Stop if z_real*z_real + z_imag*z_imag > 4.0.
3. Color based on iteration count: t = iterations / 20.0.
4. outColor = vec4(t, t*0.5, 1.0-t, 1.0).
   (Blue for inside the set where t=1, yellow-ish for outside)

Use a loop (OpLoopMerge/OpBranch/OpBranchConditional) for the iterations.
Use OpFMul, OpFAdd, OpFSub for the complex arithmetic.
""",
  },

  # 17: Procedural brick pattern
  {
    "description": "Brick Pattern",
    "prompt": f"""{SPIRV_INTERFACE_DESC}

**Task:** Render a procedural brick pattern using UV coordinates.
1. Scale: bx = uv.x * 8.0, by = uv.y * 16.0.
2. For alternating rows, offset x: row = floor(by).
   If fract(row * 0.5) > 0.25, then bx = bx + 0.5.
3. fx = fract(bx), fy = fract(by).
4. If fx < 0.05 or fy < 0.1, use mortar color vec3(0.7, 0.7, 0.7).
   Otherwise use brick color vec3(0.7, 0.2, 0.1).
5. Add diffuse: shade = 0.4 + 0.6*max(dot(normalize(normal), normalize(lightPos.xyz-worldPos)), 0.0).
6. outColor = vec4(chosenColor * shade, 1.0).

Use GLSL.std.450: Floor(8), Fract(10), Normalize(69), FMax(40).
""",
  },

  # 18: Fresnel heatmap
  {
    "description": "Fresnel Heatmap",
    "prompt": f"""{SPIRV_INTERFACE_DESC}

**Task:** Create a heatmap based on the viewing angle (Fresnel-like).
1. N = normalize(normal), V = normalize(cameraPos.xyz - worldPos).
2. NdotV = max(dot(N, V), 0.0).
3. Map NdotV to a heatmap: 
   - NdotV near 0 (edges): red (1,0,0)
   - NdotV near 0.5 (mid): yellow (1,1,0)  
   - NdotV near 1.0 (center): blue (0,0,1)
4. Implement as: 
   if NdotV < 0.5: r=1, g=NdotV*2, b=0
   else: r=1.0-(NdotV-0.5)*2, g=1.0-(NdotV-0.5)*2, b=(NdotV-0.5)*2
5. outColor = vec4(r, g, b, 1.0).

Use GLSL.std.450: Normalize(69), FMax(40), FMin(37), FClamp(43).
""",
  },

  # 19: Combined diffuse + specular + ambient
  {
    "description": "Full Phong Model",
    "prompt": f"""{SPIRV_INTERFACE_DESC}

**Task:** Implement a complete Phong lighting model with colored materials.
1. N = normalize(normal), L = normalize(lightPos.xyz - worldPos), V = normalize(cameraPos.xyz - worldPos).
2. Material: ambientColor = vec3(0.1, 0.05, 0.05), diffuseColor = vertex color, specularColor = vec3(1.0, 1.0, 1.0).
3. Ambient = ambientColor.
4. Diffuse = diffuseColor * max(dot(N, L), 0.0).
5. R = reflect(-L, N): R = -L - 2.0*dot(N,-L)*N = 2.0*dot(N,L)*N - L.
6. Specular = specularColor * pow(max(dot(R, V), 0.0), 32.0).
7. outColor = vec4(clamp(ambient + diffuse + specular, 0.0, 1.0), 1.0).

Use GLSL.std.450: Normalize(69), Reflect(71), Pow(26), FMax(40), FClamp(43).
""",
  },
]

# ---------------------------------------------------------------------------
# Test interface (follows benchmark conventions)
# ---------------------------------------------------------------------------

structure = {
  "type": "object",
  "properties": {
    "reasoning": {
      "type": "string",
      "description": "Explain your approach to writing this SPIR-V assembly shader"
    },
    "spirv_code": {
      "type": "string",
      "description": "Complete SPIR-V assembly text for the fragment shader"
    }
  },
  "required": ["reasoning", "spirv_code"],
  "additionalProperties": False
}


def prepareSubpassPrompt(subPass: int) -> str:
  """Generate the prompt for the given subpass."""
  if subPass >= len(SUBPASSES):
    raise StopIteration
  return SUBPASSES[subPass]["prompt"]


# All subpasses are independent (each has its own prompt)
extraGradeAnswerRuns = []

# Shared renderer instance (created on first use)
_renderer_instance: Optional[ShaderRenderer] = None


def _get_renderer() -> ShaderRenderer:
  global _renderer_instance
  if _renderer_instance is None:
    _renderer_instance = ShaderRenderer(512, 512)
  return _renderer_instance


def gradeAnswer(result: dict, subPass: int, aiEngineName: str) -> tuple:
  """
  Grade the SPIR-V assembly fragment shader.
  Assembles it, renders on the sphere, compares to reference image.
  """
  if not result:
    return 0.0, "No result provided"

  if "spirv_code" not in result:
    return 0.0, "No SPIR-V code provided"

  spirv_text = result["spirv_code"]
  desc = SUBPASSES[subPass]["description"]

  try:
    renderer = _get_renderer()
  except Exception as e:
    return 0.0, f"[{desc}] Failed to create renderer: {e}"

  score, explanation = grade_shader(
    spirv_text, test_num=41, subpass=subPass, renderer=renderer,
    color_tolerance=2, spatial_tolerance=1
  )
  return score, f"[{desc}] {explanation}"


def resultToNiceReport(result: dict, subPass: int, aiEngineName: str) -> str:
  if not result:
    return "<p style='color:red'>No result provided</p>"
  desc = SUBPASSES[subPass]["description"]
  html = f"<h4>SPIR-V Shader - {desc}</h4>"
  if "reasoning" in result:
    r = result['reasoning'][:400] + ('...' if len(result.get('reasoning', '')) > 400 else '')
    html += f"<p><strong>Approach:</strong> {r}</p>"
  if "spirv_code" in result:
    code = result["spirv_code"].replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    html += f"<details><summary>View SPIR-V ({len(result['spirv_code'])} chars)</summary><pre>{code}</pre></details>"
  return html


highLevelSummary = """
SPIR-V Assembly Fragment Shaders tests the ability to write GPU shaders in raw SPIR-V assembly.

**Key concepts:**
- SPIR-V instruction set (OpLoad, OpStore, OpDot, OpFMul, etc.)
- GPU rendering pipeline (fragment shader inputs/outputs)
- Lighting models (Lambertian, Phong, Blinn-Phong)
- Procedural patterns (checkerboard, stripes, bricks, Mandelbrot)
- Vector math in SPIR-V (normalize, reflect, cross products)
"""
