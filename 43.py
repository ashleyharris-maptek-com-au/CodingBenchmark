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

import json
import os
import subprocess
import sys
import tempfile
from typing import Tuple, Optional, Dict
from PIL import Image

from shader_test_utils import (
    ShaderRenderer, compile_hlsl, validate_spirv,
    compare_images, load_reference, save_reference, get_reference_path,
    image_pair_html
)

title = "Fragment Shaders (HLSL)"

TIMEOUT_SECONDS = 60

_renderer_instance: Optional[ShaderRenderer] = None
_OUTPUT_IMAGE_CACHE: Dict[Tuple[int, str], str] = {}


def _get_renderer() -> ShaderRenderer:
    global _renderer_instance
    if _renderer_instance is None:
        _renderer_instance = ShaderRenderer(512, 512)
    return _renderer_instance

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

**Task:** Output a solid red color (float4(1.0, 0.0, 0.0, 1.0)).
""",
    },
    {
        "description": "Normal to RGB",
        "prompt": f"""{HLSL_INTERFACE_DESC}

**Task:** Visualize the surface normal as color by mapping the normalized normal from [-1, 1] into [0, 1] and outputting it with alpha 1.0.
""",
    },
    {
        "description": "UV to RG",
        "prompt": f"""{HLSL_INTERFACE_DESC}

**Task:** Visualize texture coordinates as color (R=uv.x, G=uv.y, B=0, A=1).
""",
    },
    {
        "description": "Vertex Color",
        "prompt": f"""{HLSL_INTERFACE_DESC}

**Task:** Pass through the per-vertex color directly with alpha 1.0.
""",
    },
    {
        "description": "Lambertian Diffuse",
        "prompt": f"""{HLSL_INTERFACE_DESC}

**Task:** Implement Lambertian diffuse lighting using N=normalize(normal) and L=normalize(lightPos.xyz - worldPos), outputting grayscale intensity d=max(dot(N,L),0) with alpha 1.0.
""",
    },
    {
        "description": "Phong Specular",
        "prompt": f"""{HLSL_INTERFACE_DESC}

**Task:** Implement Phong specular highlights using N, L, V and R=reflect(-L,N), with diffuse=max(dot(N,L),0) and specular=pow(max(dot(R,V),0),32). Output grayscale intensity (0.2 + 0.5*diffuse + 0.8*specular) with alpha 1.0.
""",
    },
    {
        "description": "Blinn-Phong",
        "prompt": f"""{HLSL_INTERFACE_DESC}

**Task:** Implement Blinn-Phong shading using N, L, V and half-vector H=normalize(L+V), with diffuse=max(dot(N,L),0) and specular=pow(max(dot(N,H),0),64). Output grayscale intensity (0.1 + 0.6*diffuse + 0.8*specular) with alpha 1.0.
""",
    },
    {
        "description": "Rim Lighting",
        "prompt": f"""{HLSL_INTERFACE_DESC}

**Task:** Implement rim (Fresnel) lighting with rim = pow(1 - max(dot(N,V),0), 3). Use base color (0.2, 0.3, 0.8) modulated by (0.3 + 0.5*diffuse), and add rim tint (1.0, 0.8, 0.5) * rim. Output alpha 1.0.
""",
    },
    {
        "description": "Toon Shading",
        "prompt": f"""{HLSL_INTERFACE_DESC}

**Task:** Implement toon/cel shading by quantizing the Lambertian diffuse term d=max(dot(N,L),0) into bands with thresholds 0.75/0.5/0.25 and values 1.0/0.7/0.4/0.2. Multiply the vertex color by the band and output alpha 1.0.
""",
    },
    {
        "description": "Checkerboard",
        "prompt": f"""{HLSL_INTERFACE_DESC}

**Task:** Render a UV checkerboard at 8x8 frequency, alternating colors 0.9 and 0.2 by parity. Apply Lambertian shading with factor (0.5 + 0.5*diffuse) and output alpha 1.0.
""",
    },
    {
        "description": "Horizontal Stripes",
        "prompt": f"""{HLSL_INTERFACE_DESC}

**Task:** Render horizontal stripes driven by sin(uv.y * 3.14159 * 16.0), alternating red (0.9, 0.1, 0.1) and blue (0.1, 0.1, 0.9). Apply Lambertian shading with factor (0.4 + 0.6*diffuse) and output alpha 1.0.
""",
    },
    {
        "description": "Polka Dots",
        "prompt": f"""{HLSL_INTERFACE_DESC}

**Task:** Render a 6x6 UV polka-dot pattern with circular dots of radius 0.3 (in cell space). Dots are yellow (1.0, 0.8, 0.0) on a dark blue background (0.1, 0.1, 0.3). Apply Lambertian shading with factor (0.3 + 0.7*diffuse) and output alpha 1.0.
""",
    },
    {
        "description": "Hemisphere Lighting",
        "prompt": f"""{HLSL_INTERFACE_DESC}

**Task:** Implement hemisphere ambient lighting with skyColor (0.4, 0.6, 1.0) and groundColor (0.3, 0.15, 0.05) blended by N.y * 0.5 + 0.5. Add a small diffuse term 0.3*d and output alpha 1.0.
""",
    },
    {
        "description": "Y-Gradient",
        "prompt": f"""{HLSL_INTERFACE_DESC}

**Task:** Color based on world-space Y: t=clamp(worldPos.y * 0.5 + 0.5), lerp bottom (0.1, 0.4, 0.1) to top (1.0, 1.0, 1.0), output alpha 1.0.
""",
    },
    {
        "description": "Fog Effect",
        "prompt": f"""{HLSL_INTERFACE_DESC}

**Task:** Apply distance-based fog: base color is (0.8, 0.2, 0.2) with Lambertian factor (0.3 + 0.7*diffuse). Fog factor is clamp((dist - 1.5) / 3.0), fog color (0.7, 0.7, 0.8), and output the lerp with alpha 1.0.
""",
    },
    {
        "description": "Gooch Shading",
        "prompt": f"""{HLSL_INTERFACE_DESC}

**Task:** Implement Gooch warm/cool shading with cool (0.2, 0.2, 0.75), warm (0.7, 0.7, 0.4), and t = dot(N,L) * 0.5 + 0.5. Output alpha 1.0.
""",
    },
    {
        "description": "Mandelbrot Set",
        "prompt": f"""{HLSL_INTERFACE_DESC}

**Task:** Render the Mandelbrot set using UVs mapped to c = (uv.x*3-2, uv.y*2-1). Iterate up to 20 steps, escape when |z|^2 > 4, and color with t=iterations/20 as (t, t*0.5, 1.0-t), alpha 1.0.
""",
    },
    {
        "description": "Brick Pattern",
        "prompt": f"""{HLSL_INTERFACE_DESC}

**Task:** Render a procedural brick pattern at 8x16 UV scale with staggered rows (half-brick offset every other row). Mortar thickness is 0.05 in X and 0.1 in Y; mortar color is (0.7, 0.7, 0.7) and brick color is (0.7, 0.2, 0.1). Apply Lambertian shading with factor (0.4 + 0.6*diffuse) and output alpha 1.0.
""",
    },
    {
        "description": "Fresnel Heatmap",
        "prompt": f"""{HLSL_INTERFACE_DESC}

**Task:** Create a heatmap based on viewing angle using NdotV = max(dot(N, V), 0). If NdotV < 0.5 use (r=1, g=2*NdotV, b=0); otherwise use r=g=1-(NdotV-0.5)*2 and b=(NdotV-0.5)*2. Output alpha 1.0.
""",
    },
    {
        "description": "Full Phong Model",
        "prompt": f"""{HLSL_INTERFACE_DESC}

**Task:** Complete Phong lighting with colored materials: ambient (0.1, 0.05, 0.05), diffuse uses vertex color with max(dot(N,L),0), specular uses (1,1,1) with shininess 32 and R=reflect(-L,N). Clamp the sum to [0,1], output alpha 1.0.
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


def _grade_answer_inner(result: dict, subPass: int, aiEngineName: str) -> tuple:
    if not result:
        return 0.0, "No result provided", {"error": "no_result"}
    if "shader_code" not in result:
        return 0.0, "No shader code provided", {"error": "no_shader_code"}

    desc = SUBPASSES[subPass]["description"]
    hlsl_src = result["shader_code"]

    # Compile HLSL to SPIR-V
    try:
        frag_spirv = compile_hlsl(hlsl_src, stage="frag", entry_point="main")
    except RuntimeError as e:
        return 0.0, f"[{desc}] HLSL compilation failed: {e}", {"error": str(e)}

    try:
        renderer = _get_renderer()
    except Exception as e:
        return 0.0, f"[{desc}] Failed to create renderer: {e}", {"error": str(e)}

    valid, err = validate_spirv(frag_spirv)
    if not valid:
        return 0.0, f"[{desc}] SPIR-V validation failed: {err}", {"error": err}

    try:
        pixels = renderer.render(frag_spirv)
    except Exception as e:
        return 0.0, f"[{desc}] Rendering failed: {e}", {"error": str(e)}

    output_image = _save_rendered_image(43, subPass, aiEngineName, pixels)

    reference = load_reference(41, subPass)
    if reference is None:
        save_reference(pixels, 41, subPass)
        return 1.0, f"[{desc}] No reference - saved current render as reference", {
            "output_image": output_image
        }

    score, explanation = compare_images(pixels, reference, color_tolerance=2, spatial_tolerance=1)
    return score, f"[{desc}] {explanation}", {"output_image": output_image}


def gradeAnswer(result: dict, subPass: int, aiEngineName: str) -> tuple:
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
            details = out.get("details", {}) or {}
            output_image = details.get("output_image")
            if output_image:
                _OUTPUT_IMAGE_CACHE[(subPass, aiEngineName)] = output_image
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
    html += image_pair_html(
        _OUTPUT_IMAGE_CACHE.get((subPass, aiEngineName), ""),
        str(get_reference_path(41, subPass))
    )
    return html


def resultToImage(result: dict, subPass: int, aiEngineName: str) -> str:
    return _OUTPUT_IMAGE_CACHE.get((subPass, aiEngineName), "")


def getReferenceImage(subPass: int, aiEngineName: str) -> str:
    return str(get_reference_path(41, subPass))


def _save_rendered_image(test_num: int, subPass: int, aiEngineName: str, pixels) -> str:
    base_dir = os.path.dirname(__file__)
    out_dir = os.path.join(base_dir, "results", "models", aiEngineName, "renders")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"test{test_num}_subpass_{subPass:02d}.png")
    Image.fromarray(pixels, "RGBA").save(out_path)
    return out_path


highLevelSummary = """
<p>Write GPU pixel shaders in HLSL (Microsoft&rsquo;s high-level shading language)
targeting Vulkan. The same 20 visual effects from the SPIR-V assembly test
(lighting, patterns, fractals on a sphere) must be reproduced, but this time
using HLSL&rsquo;s C-like syntax with Vulkan-specific attributes.</p>
<p>The code is compiled to SPIR-V behind the scenes and rendered; the output is
compared pixel-by-pixel against the reference images.</p>
"""
