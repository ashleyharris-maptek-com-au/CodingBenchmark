"""
Test 44: GLSL Fragment Shaders

The LLM must write fragment shaders in GLSL (Vulkan dialect).
Same 20 subpass tasks as test 41, but the shader source language is GLSL
instead of SPIR-V assembly. Compiled to SPIR-V via glslangValidator.

Uses test 41's reference images for comparison (identical rendering setup).

Available fragment shader inputs:
  layout(location = 0) in vec3 worldPos;
  layout(location = 1) in vec3 normal;
  layout(location = 2) in vec2 uv;
  layout(location = 3) in vec3 tangent;
  layout(location = 4) in vec3 color;

Available uniform buffer (set=0, binding=0):
  layout(set = 0, binding = 0) uniform UBO {
    mat4 model;
    mat4 view;
    mat4 projection;
    vec4 lightPos;      // (5, 5, 5, 1)
    vec4 cameraPos;     // (0, 0, 3, 1)
    vec4 params;        // (0, 0.5, 0, 0)
  } ubo;

Fragment output:
  layout(location = 0) out vec4 outColor;  -- RGBA, format rgba8unorm
"""

import os
import sys
from typing import Tuple, Optional

from shader_test_utils import (
    ShaderRenderer, compile_glsl, validate_spirv,
    compare_images, load_reference, grade_shader_binary
)

title = "GLSL Fragment Shaders"

# ---------------------------------------------------------------------------
# Common GLSL interface description for prompts
# ---------------------------------------------------------------------------

GLSL_INTERFACE_DESC = """\
You are writing a GLSL fragment shader for Vulkan (GLSL 450). The shader will be
compiled with glslangValidator targeting Vulkan 1.0, then rendered on a unit sphere.

**Available inputs (from vertex shader):**
```glsl
layout(location = 0) in vec3 worldPos;   // world-space position
layout(location = 1) in vec3 normal;     // world-space normal (interpolated, may need normalize)
layout(location = 2) in vec2 uv;         // texture coordinates in [0,1]
layout(location = 3) in vec3 tangent;    // tangent vector
layout(location = 4) in vec3 color;      // per-vertex color (smoothly varies: rgb = 0.5+0.5*xyz)
```

**Available uniform buffer (set 0, binding 0):**
```glsl
layout(set = 0, binding = 0) uniform UBO {
    mat4 model;          // offset 0   (column-major)
    mat4 view;           // offset 64
    mat4 projection;     // offset 128
    vec4 lightPos;       // value: (5.0, 5.0, 5.0, 1.0)
    vec4 cameraPos;      // value: (0.0, 0.0, 3.0, 1.0)
    vec4 params;         // value: (0.0, 0.5, 0.0, 0.0)
} ubo;
```

**Required output:**
```glsl
layout(location = 0) out vec4 outColor;  // RGBA float, stored as rgba8unorm
```

**GLSL requirements:**
- Must start with `#version 450`
- Use `layout(location = N)` for inputs/outputs
- Use `layout(set = 0, binding = 0)` for the UBO
- Standard GLSL built-ins available: normalize, dot, reflect, pow, max, clamp, mix, etc.
- Entry point must be `void main()`

**Write ONLY the complete GLSL source code.** Do not include any other text or explanation
outside the shader_code field.
"""

# ---------------------------------------------------------------------------
# Subpass definitions (same tasks as test 41, different language)
# ---------------------------------------------------------------------------

SUBPASSES = [
    {
        "description": "Solid Red",
        "prompt": f"""{GLSL_INTERFACE_DESC}

**Task:** Output a solid red color: outColor = vec4(1.0, 0.0, 0.0, 1.0).
This is the simplest possible fragment shader.
""",
    },
    {
        "description": "Normal to RGB",
        "prompt": f"""{GLSL_INTERFACE_DESC}

**Task:** Visualize the surface normal as color.
Map the normal from [-1,1] to [0,1] range: outColor.rgb = normalize(normal) * 0.5 + 0.5, alpha = 1.0.
""",
    },
    {
        "description": "UV to RG",
        "prompt": f"""{GLSL_INTERFACE_DESC}

**Task:** Visualize texture coordinates as color.
outColor = vec4(uv.x, uv.y, 0.0, 1.0).
""",
    },
    {
        "description": "Vertex Color",
        "prompt": f"""{GLSL_INTERFACE_DESC}

**Task:** Pass through the per-vertex color directly.
outColor = vec4(color, 1.0).
""",
    },
    {
        "description": "Lambertian Diffuse",
        "prompt": f"""{GLSL_INTERFACE_DESC}

**Task:** Implement Lambertian diffuse lighting.
1. N = normalize(normal).
2. L = normalize(ubo.lightPos.xyz - worldPos).
3. d = max(dot(N, L), 0.0).
4. outColor = vec4(d, d, d, 1.0).
""",
    },
    {
        "description": "Phong Specular",
        "prompt": f"""{GLSL_INTERFACE_DESC}

**Task:** Implement Phong specular highlights.
1. N = normalize(normal), L = normalize(ubo.lightPos.xyz - worldPos), V = normalize(ubo.cameraPos.xyz - worldPos).
2. R = reflect(-L, N).
3. specular = pow(max(dot(R, V), 0.0), 32.0).
4. diffuse = max(dot(N, L), 0.0).
5. outColor = vec4(vec3(0.2 + 0.5*diffuse + 0.8*specular), 1.0).
""",
    },
    {
        "description": "Blinn-Phong",
        "prompt": f"""{GLSL_INTERFACE_DESC}

**Task:** Implement Blinn-Phong shading.
1. N = normalize(normal), L = normalize(ubo.lightPos.xyz - worldPos), V = normalize(ubo.cameraPos.xyz - worldPos).
2. H = normalize(L + V).
3. diffuse = max(dot(N, L), 0.0), specular = pow(max(dot(N, H), 0.0), 64.0).
4. outColor = vec4(vec3(0.1 + 0.6*diffuse + 0.8*specular), 1.0).
""",
    },
    {
        "description": "Rim Lighting",
        "prompt": f"""{GLSL_INTERFACE_DESC}

**Task:** Implement rim (Fresnel) lighting.
1. N = normalize(normal), V = normalize(ubo.cameraPos.xyz - worldPos).
2. rim = pow(1.0 - max(dot(N, V), 0.0), 3.0).
3. L = normalize(ubo.lightPos.xyz - worldPos), diff = max(dot(N, L), 0.0).
4. base = vec3(0.2, 0.3, 0.8).
5. outColor = vec4(base * (0.3 + 0.5*diff) + vec3(1.0, 0.8, 0.5) * rim, 1.0).
""",
    },
    {
        "description": "Toon Shading",
        "prompt": f"""{GLSL_INTERFACE_DESC}

**Task:** Implement toon/cel shading with quantized lighting.
1. N = normalize(normal), L = normalize(ubo.lightPos.xyz - worldPos).
2. d = max(dot(N, L), 0.0).
3. Quantize: shade = d > 0.75 ? 1.0 : d > 0.5 ? 0.7 : d > 0.25 ? 0.4 : 0.2.
4. outColor = vec4(color * shade, 1.0).
""",
    },
    {
        "description": "Checkerboard",
        "prompt": f"""{GLSL_INTERFACE_DESC}

**Task:** Render a checkerboard pattern using UV coordinates.
1. su = uv.x * 8.0, sv = uv.y * 8.0.
2. checker = floor(su) + floor(sv).
3. If fract(checker*0.5) < 0.25: color1 = vec3(0.9). Else: color2 = vec3(0.2).
4. shade = 0.5 + 0.5*max(dot(normalize(normal), normalize(ubo.lightPos.xyz - worldPos)), 0.0).
5. outColor = vec4(checkerColor * shade, 1.0).
""",
    },
    {
        "description": "Horizontal Stripes",
        "prompt": f"""{GLSL_INTERFACE_DESC}

**Task:** Render horizontal stripes using UV v-coordinate.
1. stripe = sin(uv.y * 3.14159 * 16.0).
2. If stripe > 0: red vec3(0.9, 0.1, 0.1). Else: blue vec3(0.1, 0.1, 0.9).
3. shade = 0.4 + 0.6*max(dot(normalize(normal), normalize(ubo.lightPos.xyz - worldPos)), 0.0).
4. outColor = vec4(stripeColor * shade, 1.0).
""",
    },
    {
        "description": "Polka Dots",
        "prompt": f"""{GLSL_INTERFACE_DESC}

**Task:** Render polka dots using UV coordinates.
1. su = uv.x * 6.0, sv = uv.y * 6.0.
2. fu = fract(su) - 0.5, fv = fract(sv) - 0.5.
3. dist = sqrt(fu*fu + fv*fv).
4. If dist < 0.3: yellow vec3(1.0, 0.8, 0.0). Else: dark blue vec3(0.1, 0.1, 0.3).
5. shade = 0.3 + 0.7*max(dot(normalize(normal), normalize(ubo.lightPos.xyz - worldPos)), 0.0).
6. outColor = vec4(dotColor * shade, 1.0).
""",
    },
    {
        "description": "Hemisphere Lighting",
        "prompt": f"""{GLSL_INTERFACE_DESC}

**Task:** Implement hemisphere (ambient) lighting.
1. N = normalize(normal).
2. skyColor = vec3(0.4, 0.6, 1.0), groundColor = vec3(0.3, 0.15, 0.05).
3. blend = N.y * 0.5 + 0.5.
4. ambient = mix(groundColor, skyColor, blend).
5. L = normalize(ubo.lightPos.xyz - worldPos), d = max(dot(N, L), 0.0).
6. outColor = vec4(ambient + vec3(0.3)*d, 1.0).
""",
    },
    {
        "description": "Y-Gradient",
        "prompt": f"""{GLSL_INTERFACE_DESC}

**Task:** Color based on world-space Y position.
1. t = clamp(worldPos.y * 0.5 + 0.5, 0.0, 1.0).
2. bottom = vec3(0.1, 0.4, 0.1), top = vec3(1.0, 1.0, 1.0).
3. outColor = vec4(mix(bottom, top, t), 1.0).
""",
    },
    {
        "description": "Fog Effect",
        "prompt": f"""{GLSL_INTERFACE_DESC}

**Task:** Apply distance-based fog.
1. N = normalize(normal), L = normalize(ubo.lightPos.xyz - worldPos).
   diff = max(dot(N,L), 0.0). baseColor = vec3(0.8, 0.2, 0.2) * (0.3 + 0.7*diff).
2. dist = length(worldPos - ubo.cameraPos.xyz).
3. fogFactor = clamp((dist - 1.5) / 3.0, 0.0, 1.0).
4. fogColor = vec3(0.7, 0.7, 0.8).
5. outColor = vec4(mix(baseColor, fogColor, fogFactor), 1.0).
""",
    },
    {
        "description": "Gooch Shading",
        "prompt": f"""{GLSL_INTERFACE_DESC}

**Task:** Implement Gooch warm/cool shading.
1. N = normalize(normal), L = normalize(ubo.lightPos.xyz - worldPos).
2. t = dot(N, L) * 0.5 + 0.5.
3. cool = vec3(0.2, 0.2, 0.75), warm = vec3(0.7, 0.7, 0.4).
4. outColor = vec4(mix(cool, warm, t), 1.0).
""",
    },
    {
        "description": "Mandelbrot Set",
        "prompt": f"""{GLSL_INTERFACE_DESC}

**Task:** Render the Mandelbrot set mapped onto the sphere using UVs.
1. c_real = uv.x * 3.0 - 2.0, c_imag = uv.y * 2.0 - 1.0.
2. z = vec2(0). Iterate z = vec2(z.x*z.x - z.y*z.y + c_real, 2.0*z.x*z.y + c_imag) up to 20 times.
   Stop if dot(z,z) > 4.0.
3. t = float(iterations) / 20.0.
4. outColor = vec4(t, t*0.5, 1.0-t, 1.0).
""",
    },
    {
        "description": "Brick Pattern",
        "prompt": f"""{GLSL_INTERFACE_DESC}

**Task:** Render a procedural brick pattern.
1. bx = uv.x * 8.0, by = uv.y * 16.0.
2. row = floor(by). If fract(row * 0.5) > 0.25: bx += 0.5.
3. fx = fract(bx), fy = fract(by).
4. If fx < 0.05 or fy < 0.1: mortar vec3(0.7). Else: brick vec3(0.7, 0.2, 0.1).
5. shade = 0.4 + 0.6*max(dot(normalize(normal), normalize(ubo.lightPos.xyz - worldPos)), 0.0).
6. outColor = vec4(chosen * shade, 1.0).
""",
    },
    {
        "description": "Fresnel Heatmap",
        "prompt": f"""{GLSL_INTERFACE_DESC}

**Task:** Create a heatmap based on viewing angle.
1. N = normalize(normal), V = normalize(ubo.cameraPos.xyz - worldPos).
2. NdotV = max(dot(N, V), 0.0).
3. if NdotV < 0.5: r=1, g=NdotV*2, b=0.
   else: r=1-(NdotV-0.5)*2, g=1-(NdotV-0.5)*2, b=(NdotV-0.5)*2.
4. outColor = vec4(r, g, b, 1.0).
""",
    },
    {
        "description": "Full Phong Model",
        "prompt": f"""{GLSL_INTERFACE_DESC}

**Task:** Complete Phong lighting with colored materials.
1. N = normalize(normal), L = normalize(ubo.lightPos.xyz - worldPos), V = normalize(ubo.cameraPos.xyz - worldPos).
2. ambient = vec3(0.1, 0.05, 0.05). diffuseColor = color. specularColor = vec3(1.0).
3. diffuse = diffuseColor * max(dot(N, L), 0.0).
4. R = reflect(-L, N). specular = specularColor * pow(max(dot(R, V), 0.0), 32.0).
5. outColor = vec4(clamp(ambient + diffuse + specular, 0.0, 1.0), 1.0).
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
            "description": "Explain your approach to writing this GLSL shader"
        },
        "shader_code": {
            "type": "string",
            "description": "Complete GLSL source code for the fragment shader"
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
    glsl_src = result["shader_code"]

    # Compile GLSL to SPIR-V
    try:
        frag_spirv = compile_glsl(glsl_src, stage="frag")
    except RuntimeError as e:
        return 0.0, f"[{desc}] GLSL compilation failed: {e}"

    try:
        renderer = _get_renderer()
    except Exception as e:
        return 0.0, f"[{desc}] Failed to create renderer: {e}"

    score, explanation = grade_shader_binary(
        frag_spirv, test_num=44, subpass=subPass, renderer=renderer,
        color_tolerance=2, spatial_tolerance=1, ref_test_num=41
    )
    return score, f"[{desc}] {explanation}"


def resultToNiceReport(result: dict, subPass: int, aiEngineName: str) -> str:
    if not result:
        return "<p style='color:red'>No result provided</p>"
    desc = SUBPASSES[subPass]["description"]
    html = f"<h4>GLSL Shader - {desc}</h4>"
    if "reasoning" in result:
        r = result['reasoning'][:400] + ('...' if len(result.get('reasoning', '')) > 400 else '')
        html += f"<p><strong>Approach:</strong> {r}</p>"
    if "shader_code" in result:
        code = result["shader_code"].replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        html += f"<details><summary>View GLSL ({len(result['shader_code'])} chars)</summary><pre>{code}</pre></details>"
    return html


highLevelSummary = """
GLSL Fragment Shaders tests the ability to write GPU fragment shaders in GLSL (Vulkan dialect).

**Key concepts:**
- GLSL 450 syntax and built-in functions (normalize, dot, reflect, pow, mix, clamp, etc.)
- Vulkan layout qualifiers (location, set, binding)
- Same 20 rendering tasks as test 41 (lighting, patterns, procedural effects)
- Compiled to SPIR-V via glslangValidator
"""
