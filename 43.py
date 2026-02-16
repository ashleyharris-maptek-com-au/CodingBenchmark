"""
Test 43: HLSL Fragment Shaders

The LLM must write fragment (pixel) shaders in HLSL.
Same 20 subpass tasks as test 41, but the shader source language is HLSL
instead of SPIR-V assembly. Compiled to SPIR-V via glslangValidator.

Uses test 41's reference images for comparison (identical rendering setup).

Available pixel shader inputs (Vulkan binding, HLSL register syntax):
  [[vk::location(0)]] float3 worldPos : TEXCOORD0;
  [[vk::location(1)]] float3 normal   : TEXCOORD1;
  [[vk::location(2)]] float2 uv       : TEXCOORD2;
  [[vk::location(3)]] float3 tangent  : TEXCOORD3;
  [[vk::location(4)]] float3 color    : TEXCOORD4;

Available constant buffer (set=0, binding=0):
  [[vk::binding(0, 0)]] cbuffer UBO {
    float4x4 model;       // offset 0
    float4x4 view;        // offset 64
    float4x4 projection;  // offset 128
    float4 lightPos;      // (5, 5, 5, 1)
    float4 cameraPos;     // (0, 0, 3, 1)
    float4 params;        // (0, 0.5, 0, 0)
  };

Pixel shader output:
  float4 : SV_Target0  -- RGBA, format rgba8unorm
"""

import os
import sys
from typing import Tuple, Optional

from shader_test_utils import (
    ShaderRenderer, compile_hlsl, validate_spirv,
    compare_images, load_reference, grade_shader_binary
)

title = "HLSL Fragment Shaders"

# ---------------------------------------------------------------------------
# Common HLSL interface description for prompts
# ---------------------------------------------------------------------------

HLSL_INTERFACE_DESC = """\
You are writing an HLSL pixel (fragment) shader for Vulkan. The shader will be
compiled with glslangValidator (-D flag for HLSL mode) targeting Vulkan 1.0,
then rendered on a unit sphere.

**Available inputs (from vertex shader, using Vulkan layout attributes):**
```hlsl
struct PSInput {
    [[vk::location(0)]] float3 worldPos : TEXCOORD0;   // world-space position
    [[vk::location(1)]] float3 normal   : TEXCOORD1;   // world-space normal (interpolated, may need normalize)
    [[vk::location(2)]] float2 uv       : TEXCOORD2;   // texture coordinates in [0,1]
    [[vk::location(3)]] float3 tangent  : TEXCOORD3;   // tangent vector
    [[vk::location(4)]] float3 color    : TEXCOORD4;   // per-vertex color (smoothly varies: rgb = 0.5+0.5*xyz)
};
```

**Available constant buffer (set 0, binding 0):**
```hlsl
[[vk::binding(0, 0)]]
cbuffer UBO : register(b0) {
    float4x4 model;          // offset 0   (column-major)
    float4x4 view;           // offset 64
    float4x4 projection;     // offset 128
    float4 lightPos;         // value: (5.0, 5.0, 5.0, 1.0)
    float4 cameraPos;        // value: (0.0, 0.0, 3.0, 1.0)
    float4 params;           // value: (0.0, 0.5, 0.0, 0.0)
};
```

**Required output:**
```hlsl
float4 main(PSInput input) : SV_Target0   // RGBA float, stored as rgba8unorm
```

**HLSL requirements:**
- Use Vulkan-specific attributes: `[[vk::location(N)]]` for inputs, `[[vk::binding(B, S)]]` for UBO
- Entry point must be named `main`
- Return type is `float4` with `SV_Target0` semantic
- Standard HLSL intrinsics are available: normalize, dot, reflect, pow, max, clamp, lerp, etc.
- Use `mul()` for matrix-vector operations if needed

**Write ONLY the complete HLSL source code.** Do not include any other text or explanation
outside the shader_code field.
"""

# ---------------------------------------------------------------------------
# Subpass definitions (same tasks as test 41, different language)
# ---------------------------------------------------------------------------

SUBPASSES = [
    {
        "description": "Solid Red",
        "prompt": f"""{HLSL_INTERFACE_DESC}

**Task:** Output a solid red color: return float4(1.0, 0.0, 0.0, 1.0).
This is the simplest possible pixel shader.
""",
    },
    {
        "description": "Normal to RGB",
        "prompt": f"""{HLSL_INTERFACE_DESC}

**Task:** Visualize the surface normal as color.
Map the normal from [-1,1] to [0,1] range: rgb = normalize(normal) * 0.5 + 0.5, a = 1.0.
""",
    },
    {
        "description": "UV to RG",
        "prompt": f"""{HLSL_INTERFACE_DESC}

**Task:** Visualize texture coordinates as color.
return float4(uv.x, uv.y, 0.0, 1.0).
""",
    },
    {
        "description": "Vertex Color",
        "prompt": f"""{HLSL_INTERFACE_DESC}

**Task:** Pass through the per-vertex color directly.
return float4(color.r, color.g, color.b, 1.0).
""",
    },
    {
        "description": "Lambertian Diffuse",
        "prompt": f"""{HLSL_INTERFACE_DESC}

**Task:** Implement Lambertian diffuse lighting.
1. N = normalize(normal).
2. L = normalize(lightPos.xyz - worldPos).
3. d = max(dot(N, L), 0.0).
4. return float4(d, d, d, 1.0).
""",
    },
    {
        "description": "Phong Specular",
        "prompt": f"""{HLSL_INTERFACE_DESC}

**Task:** Implement Phong specular highlights.
1. N = normalize(normal), L = normalize(lightPos.xyz - worldPos), V = normalize(cameraPos.xyz - worldPos).
2. R = reflect(-L, N).
3. specular = pow(max(dot(R, V), 0.0), 32.0).
4. diffuse = max(dot(N, L), 0.0).
5. return float4((0.2 + 0.5*diffuse + 0.8*specular).xxx, 1.0).
""",
    },
    {
        "description": "Blinn-Phong",
        "prompt": f"""{HLSL_INTERFACE_DESC}

**Task:** Implement Blinn-Phong shading.
1. N = normalize(normal), L = normalize(lightPos.xyz - worldPos), V = normalize(cameraPos.xyz - worldPos).
2. H = normalize(L + V).
3. diffuse = max(dot(N, L), 0.0), specular = pow(max(dot(N, H), 0.0), 64.0).
4. return float4((0.1 + 0.6*diffuse + 0.8*specular).xxx, 1.0).
""",
    },
    {
        "description": "Rim Lighting",
        "prompt": f"""{HLSL_INTERFACE_DESC}

**Task:** Implement rim (Fresnel) lighting.
1. N = normalize(normal), V = normalize(cameraPos.xyz - worldPos).
2. rim = pow(1.0 - max(dot(N, V), 0.0), 3.0).
3. L = normalize(lightPos.xyz - worldPos), diff = max(dot(N, L), 0.0).
4. base = float3(0.2, 0.3, 0.8).
5. return float4(base * (0.3 + 0.5*diff) + float3(1.0, 0.8, 0.5) * rim, 1.0).
""",
    },
    {
        "description": "Toon Shading",
        "prompt": f"""{HLSL_INTERFACE_DESC}

**Task:** Implement toon/cel shading with quantized lighting.
1. N = normalize(normal), L = normalize(lightPos.xyz - worldPos).
2. d = max(dot(N, L), 0.0).
3. Quantize: shade = d > 0.75 ? 1.0 : d > 0.5 ? 0.7 : d > 0.25 ? 0.4 : 0.2.
4. return float4(color * shade, 1.0).
""",
    },
    {
        "description": "Checkerboard",
        "prompt": f"""{HLSL_INTERFACE_DESC}

**Task:** Render a checkerboard pattern using UV coordinates.
1. su = uv.x * 8.0, sv = uv.y * 8.0.
2. checker = floor(su) + floor(sv).
3. If frac(checker*0.5) < 0.25, color1 = 0.9. Else color2 = 0.2.
4. shade = 0.5 + 0.5*max(dot(normalize(normal), normalize(lightPos.xyz - worldPos)), 0.0).
5. return float4(checkerColor * shade, 1.0).
""",
    },
    {
        "description": "Horizontal Stripes",
        "prompt": f"""{HLSL_INTERFACE_DESC}

**Task:** Render horizontal stripes using UV v-coordinate.
1. stripe = sin(uv.y * 3.14159 * 16.0).
2. If stripe > 0: red (0.9, 0.1, 0.1). Else: blue (0.1, 0.1, 0.9).
3. shade = 0.4 + 0.6*max(dot(normalize(normal), normalize(lightPos.xyz - worldPos)), 0.0).
4. return float4(stripeColor * shade, 1.0).
""",
    },
    {
        "description": "Polka Dots",
        "prompt": f"""{HLSL_INTERFACE_DESC}

**Task:** Render polka dots using UV coordinates.
1. su = uv.x * 6.0, sv = uv.y * 6.0.
2. fu = frac(su) - 0.5, fv = frac(sv) - 0.5.
3. dist = sqrt(fu*fu + fv*fv).
4. If dist < 0.3: yellow (1.0, 0.8, 0.0). Else: dark blue (0.1, 0.1, 0.3).
5. shade = 0.3 + 0.7*max(dot(normalize(normal), normalize(lightPos.xyz - worldPos)), 0.0).
6. return float4(dotColor * shade, 1.0).
""",
    },
    {
        "description": "Hemisphere Lighting",
        "prompt": f"""{HLSL_INTERFACE_DESC}

**Task:** Implement hemisphere (ambient) lighting.
1. N = normalize(normal).
2. skyColor = float3(0.4, 0.6, 1.0), groundColor = float3(0.3, 0.15, 0.05).
3. blend = N.y * 0.5 + 0.5.
4. ambient = lerp(groundColor, skyColor, blend).
5. L = normalize(lightPos.xyz - worldPos), d = max(dot(N, L), 0.0).
6. return float4(ambient + float3(0.3, 0.3, 0.3)*d, 1.0).
""",
    },
    {
        "description": "Y-Gradient",
        "prompt": f"""{HLSL_INTERFACE_DESC}

**Task:** Color based on world-space Y position.
1. t = clamp(worldPos.y * 0.5 + 0.5, 0.0, 1.0).
2. bottom = float3(0.1, 0.4, 0.1), top = float3(1.0, 1.0, 1.0).
3. return float4(lerp(bottom, top, t), 1.0).
""",
    },
    {
        "description": "Fog Effect",
        "prompt": f"""{HLSL_INTERFACE_DESC}

**Task:** Apply distance-based fog.
1. N = normalize(normal), L = normalize(lightPos.xyz - worldPos).
   diff = max(dot(N,L), 0.0). baseColor = float3(0.8, 0.2, 0.2) * (0.3 + 0.7*diff).
2. dist = length(worldPos - cameraPos.xyz).
3. fogFactor = clamp((dist - 1.5) / 3.0, 0.0, 1.0).
4. fogColor = float3(0.7, 0.7, 0.8).
5. return float4(lerp(baseColor, fogColor, fogFactor), 1.0).
""",
    },
    {
        "description": "Gooch Shading",
        "prompt": f"""{HLSL_INTERFACE_DESC}

**Task:** Implement Gooch warm/cool shading.
1. N = normalize(normal), L = normalize(lightPos.xyz - worldPos).
2. t = dot(N, L) * 0.5 + 0.5.
3. cool = float3(0.2, 0.2, 0.75), warm = float3(0.7, 0.7, 0.4).
4. return float4(lerp(cool, warm, t), 1.0).
""",
    },
    {
        "description": "Mandelbrot Set",
        "prompt": f"""{HLSL_INTERFACE_DESC}

**Task:** Render the Mandelbrot set mapped onto the sphere using UVs.
1. c_real = uv.x * 3.0 - 2.0, c_imag = uv.y * 2.0 - 1.0.
2. z = 0+0i. Iterate z = z*z + c for up to 20 iterations.
   Stop if |z|^2 > 4.0.
3. t = iterations / 20.0.
4. return float4(t, t*0.5, 1.0-t, 1.0).
""",
    },
    {
        "description": "Brick Pattern",
        "prompt": f"""{HLSL_INTERFACE_DESC}

**Task:** Render a procedural brick pattern.
1. bx = uv.x * 8.0, by = uv.y * 16.0.
2. row = floor(by). If frac(row * 0.5) > 0.25: bx += 0.5.
3. fx = frac(bx), fy = frac(by).
4. If fx < 0.05 or fy < 0.1: mortar (0.7, 0.7, 0.7). Else: brick (0.7, 0.2, 0.1).
5. shade = 0.4 + 0.6*max(dot(normalize(normal), normalize(lightPos.xyz - worldPos)), 0.0).
6. return float4(chosen * shade, 1.0).
""",
    },
    {
        "description": "Fresnel Heatmap",
        "prompt": f"""{HLSL_INTERFACE_DESC}

**Task:** Create a heatmap based on viewing angle.
1. N = normalize(normal), V = normalize(cameraPos.xyz - worldPos).
2. NdotV = max(dot(N, V), 0.0).
3. if NdotV < 0.5: r=1, g=NdotV*2, b=0.
   else: r=1-(NdotV-0.5)*2, g=1-(NdotV-0.5)*2, b=(NdotV-0.5)*2.
4. return float4(r, g, b, 1.0).
""",
    },
    {
        "description": "Full Phong Model",
        "prompt": f"""{HLSL_INTERFACE_DESC}

**Task:** Complete Phong lighting with colored materials.
1. N = normalize(normal), L = normalize(lightPos.xyz - worldPos), V = normalize(cameraPos.xyz - worldPos).
2. ambient = float3(0.1, 0.05, 0.05). diffuseColor = color. specularColor = float3(1,1,1).
3. diffuse = diffuseColor * max(dot(N, L), 0.0).
4. R = reflect(-L, N). specular = specularColor * pow(max(dot(R, V), 0.0), 32.0).
5. return float4(clamp(ambient + diffuse + specular, 0.0, 1.0), 1.0).
""",
    },
]

# ---------------------------------------------------------------------------
# Test interface
# ---------------------------------------------------------------------------

structure = {
    "type": "object",
    "properties": {
        "reasoning": {
            "type": "string",
            "description": "Explain your approach to writing this HLSL shader"
        },
        "shader_code": {
            "type": "string",
            "description": "Complete HLSL source code for the pixel shader"
        }
    },
    "required": ["reasoning", "shader_code"],
    "additionalProperties": False
}


def prepareSubpassPrompt(subPass: int) -> str:
    if subPass >= len(SUBPASSES):
        raise StopIteration
    return SUBPASSES[subPass]["prompt"]


extraGradeAnswerRuns = []
_renderer_instance = None


def _get_renderer():
    global _renderer_instance
    if _renderer_instance is None:
        from shader_test_utils import ShaderRenderer
        _renderer_instance = ShaderRenderer(512, 512)
    return _renderer_instance


def gradeAnswer(result: dict, subPass: int, aiEngineName: str) -> tuple:
    if not result:
        return 0.0, "No result provided"
    if "shader_code" not in result:
        return 0.0, "No shader code provided"

    desc = SUBPASSES[subPass]["description"]
    hlsl_src = result["shader_code"]

    # Compile HLSL to SPIR-V
    try:
        frag_spirv = compile_hlsl(hlsl_src, stage="frag", entry_point="main")
    except RuntimeError as e:
        return 0.0, f"[{desc}] HLSL compilation failed: {e}"

    # Grade using shared binary grader, referencing test 41's images
    try:
        renderer = _get_renderer()
    except Exception as e:
        return 0.0, f"[{desc}] Failed to create renderer: {e}"

    score, explanation = grade_shader_binary(
        frag_spirv, test_num=43, subpass=subPass, renderer=renderer,
        color_tolerance=2, spatial_tolerance=1, ref_test_num=41
    )
    return score, f"[{desc}] {explanation}"


def resultToNiceReport(result: dict, subPass: int, aiEngineName: str) -> str:
    if not result:
        return "<p style='color:red'>No result provided</p>"
    desc = SUBPASSES[subPass]["description"]
    html = f"<h4>HLSL Shader - {desc}</h4>"
    if "reasoning" in result:
        r = result['reasoning'][:400] + ('...' if len(result.get('reasoning', '')) > 400 else '')
        html += f"<p><strong>Approach:</strong> {r}</p>"
    if "shader_code" in result:
        code = result["shader_code"].replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        html += f"<details><summary>View HLSL ({len(result['shader_code'])} chars)</summary><pre>{code}</pre></details>"
    return html


highLevelSummary = """
HLSL Fragment Shaders tests the ability to write GPU pixel shaders in HLSL for Vulkan.

**Key concepts:**
- HLSL syntax and intrinsics (normalize, dot, reflect, pow, lerp, clamp, etc.)
- Vulkan-specific HLSL attributes ([[vk::location]], [[vk::binding]])
- Same 20 rendering tasks as test 41 (lighting, patterns, procedural effects)
- Compiled to SPIR-V via glslangValidator
"""
