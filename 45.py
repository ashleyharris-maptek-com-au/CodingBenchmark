"""
Test 45: Binary SPIR-V Fragment Shaders

The LLM must write fragment shaders as raw SPIR-V binary, provided as a
hex-encoded string. This tests understanding of the SPIR-V binary format
at the byte level.

Same 20 subpass tasks as test 41. Uses test 41's reference images.

The SPIR-V binary format:
  - Little-endian 32-bit words
  - Magic number: 0x07230203
  - Header: magic, version, generator, bound, reserved (5 words)
  - Then instruction stream: each instruction is (word_count << 16 | opcode) followed by operands

Available fragment shader inputs (same as test 41):
  layout(location = 0) in vec3 worldPos;
  layout(location = 1) in vec3 normal;
  layout(location = 2) in vec2 uv;
  layout(location = 3) in vec3 tangent;
  layout(location = 4) in vec3 color;

Available uniform buffer (set=0, binding=0):
  Same layout as test 41 (model/view/proj + lightPos + cameraPos + params).

Fragment output:
  layout(location = 0) out vec4 outColor;
"""

import json
import os
import subprocess
import sys
import tempfile
from typing import Tuple, Optional, Dict
from PIL import Image

from shader_test_utils import (
    ShaderRenderer, validate_spirv,
    compare_images, load_reference, save_reference, get_reference_path,
    image_pair_html
)

title = "Fragment Shaders (SPIRV)"

TIMEOUT_SECONDS = 60

# ---------------------------------------------------------------------------
# Common binary SPIR-V description for prompts
# ---------------------------------------------------------------------------

BINARY_SPIRV_INTERFACE_DESC = """\
You are writing a SPIR-V fragment shader as raw binary data, encoded as a hex string.
The binary will be decoded from hex, validated with spirv-val, and rendered on a unit sphere.

**SPIR-V Binary Format (little-endian 32-bit words):**
Each 32-bit word is 8 hex characters (little-endian byte order).

Header (5 words):
```
03022307   // Magic number 0x07230203 (little-endian)
00000100   // Version 1.0 (0x00010000)
00000000   // Generator (0 = manual)
BBBBBBBB   // Bound (max ID + 1)
00000000   // Reserved
```

Instructions follow as: first word = (WordCount << 16) | Opcode, then operands.

**Key opcodes (decimal):**
- OpCapability (17): word_count=2, operand=capability (1=Shader)
- OpExtInstImport (11): result_id, name_string
- OpMemoryModel (14): word_count=3, addressing=0(Logical), model=1(GLSL450)
- OpEntryPoint (15): execution_model=4(Fragment), entry_id, name, interface_vars...
- OpExecutionMode (16): entry_id, mode=7(OriginUpperLeft)
- OpDecorate (71): target, decoration, operands
- OpTypeVoid (19), OpTypeFunction (33), OpTypeFloat (22), OpTypeVector (23)
- OpTypePointer (32), OpVariable (59), OpConstant (43)
- OpFunction (54), OpFunctionEnd (56), OpLabel (248)
- OpLoad (61), OpStore (62), OpReturn (253)
- OpCompositeConstruct (80), OpCompositeExtract (81)
- OpDot (148), OpFMul (133), OpFAdd (129), OpFSub (131), OpFNegate (127)
- OpExtInst (12): GLSL.std.450 ext instructions

**Decoration values:** Location=30, DescriptorSet=34, Binding=33, Block=2, Offset=35

**Available inputs (from vertex shader):**
```
location 0: vec3 worldPos
location 1: vec3 normal
location 2: vec2 uv
location 3: vec3 tangent
location 4: vec3 color
```

**UBO (set=0, binding=0):** mat4 model(0), mat4 view(64), mat4 projection(128),
vec4 lightPos(192), vec4 cameraPos(208), vec4 params(224).

**Required output:** location 0: vec4 outColor

**Write ONLY the complete hex string of the SPIR-V binary.** Each byte as two hex
characters, no spaces or newlines within the hex data. The hex string represents
the raw bytes of the .spv file.
"""

# ---------------------------------------------------------------------------
# Subpass definitions (same tasks, binary SPIR-V format)
# ---------------------------------------------------------------------------

SUBPASSES = [
    {
        "description": "Solid Red",
        "prompt": f"""{BINARY_SPIRV_INTERFACE_DESC}

**Task:** Output a solid red color (vec4(1.0, 0.0, 0.0, 1.0)).
""",
    },
    {
        "description": "Normal to RGB",
        "prompt": f"""{BINARY_SPIRV_INTERFACE_DESC}

**Task:** Visualize the surface normal as color by mapping the normalized normal from [-1, 1] into [0, 1] with alpha 1.0.
""",
    },
    {
        "description": "UV to RG",
        "prompt": f"""{BINARY_SPIRV_INTERFACE_DESC}

**Task:** Visualize texture coordinates as color (R=uv.x, G=uv.y, B=0, A=1).
""",
    },
    {
        "description": "Vertex Color",
        "prompt": f"""{BINARY_SPIRV_INTERFACE_DESC}

**Task:** Pass through the per-vertex color directly with alpha 1.0.
""",
    },
    {
        "description": "Lambertian Diffuse",
        "prompt": f"""{BINARY_SPIRV_INTERFACE_DESC}

**Task:** Implement Lambertian diffuse lighting using N=normalize(normal), L=normalize(lightPos.xyz - worldPos), and grayscale intensity d=max(dot(N,L),0) with alpha 1.0.
""",
    },
    {
        "description": "Phong Specular",
        "prompt": f"""{BINARY_SPIRV_INTERFACE_DESC}

**Task:** Implement Phong specular highlights using N, L, V and R=reflect(-L,N), with diffuse=max(dot(N,L),0) and specular=pow(max(dot(R,V),0),32). Output grayscale intensity (0.2 + 0.5*diffuse + 0.8*specular) with alpha 1.0.
""",
    },
    {
        "description": "Blinn-Phong",
        "prompt": f"""{BINARY_SPIRV_INTERFACE_DESC}

**Task:** Implement Blinn-Phong shading using N, L, V and half-vector H=normalize(L+V), with diffuse=max(dot(N,L),0) and specular=pow(max(dot(N,H),0),64). Output grayscale intensity (0.1 + 0.6*diffuse + 0.8*specular) with alpha 1.0.
""",
    },
    {
        "description": "Rim Lighting",
        "prompt": f"""{BINARY_SPIRV_INTERFACE_DESC}

**Task:** Implement rim (Fresnel) lighting with rim = pow(1 - max(dot(N,V),0), 3). Use base color (0.2, 0.3, 0.8) modulated by (0.3 + 0.5*diffuse), and add rim tint (1.0, 0.8, 0.5) * rim. Output alpha 1.0.
""",
    },
    {
        "description": "Toon Shading",
        "prompt": f"""{BINARY_SPIRV_INTERFACE_DESC}

**Task:** Implement toon/cel shading by quantizing the Lambertian diffuse term d=max(dot(N,L),0) into bands with thresholds 0.75/0.5/0.25 and values 1.0/0.7/0.4/0.2. Multiply the vertex color by the band and output alpha 1.0.
""",
    },
    {
        "description": "Checkerboard",
        "prompt": f"""{BINARY_SPIRV_INTERFACE_DESC}

**Task:** Render a UV checkerboard at 8x8 frequency, alternating colors 0.9 and 0.2 by parity. Apply Lambertian shading and output alpha 1.0.
""",
    },
    {
        "description": "Horizontal Stripes",
        "prompt": f"""{BINARY_SPIRV_INTERFACE_DESC}

**Task:** Render horizontal stripes driven by sin(uv.y * pi * 16), alternating red (0.9, 0.1, 0.1) and blue (0.1, 0.1, 0.9). Apply Lambertian shading and output alpha 1.0.
""",
    },
    {
        "description": "Polka Dots",
        "prompt": f"""{BINARY_SPIRV_INTERFACE_DESC}

**Task:** Render a 6x6 UV polka-dot pattern with circular dots of radius 0.3 (in cell space). Dots are yellow (1.0, 0.8, 0.0) on a dark blue background (0.1, 0.1, 0.3). Apply Lambertian shading and output alpha 1.0.
""",
    },
    {
        "description": "Hemisphere Lighting",
        "prompt": f"""{BINARY_SPIRV_INTERFACE_DESC}

**Task:** Implement hemisphere ambient lighting with skyColor (0.4, 0.6, 1.0) and groundColor (0.3, 0.15, 0.05) blended by N.y * 0.5 + 0.5. Add a small diffuse term 0.3*d and output alpha 1.0.
""",
    },
    {
        "description": "Y-Gradient",
        "prompt": f"""{BINARY_SPIRV_INTERFACE_DESC}

**Task:** Color based on world-space Y: t=clamp(worldPos.y * 0.5 + 0.5), mix bottom (0.1, 0.4, 0.1) to top (1.0, 1.0, 1.0), output alpha 1.0.
""",
    },
    {
        "description": "Fog Effect",
        "prompt": f"""{BINARY_SPIRV_INTERFACE_DESC}

**Task:** Apply distance-based fog: base color is (0.8, 0.2, 0.2) with Lambertian factor (0.3 + 0.7*diffuse). Fog factor is clamp((dist - 1.5) / 3.0), fog color (0.7, 0.7, 0.8), and output the mix with alpha 1.0.
""",
    },
    {
        "description": "Gooch Shading",
        "prompt": f"""{BINARY_SPIRV_INTERFACE_DESC}

**Task:** Implement Gooch warm/cool shading with cool (0.2, 0.2, 0.75), warm (0.7, 0.7, 0.4), and t = dot(N,L) * 0.5 + 0.5. Output alpha 1.0.
""",
    },
    {
        "description": "Mandelbrot Set",
        "prompt": f"""{BINARY_SPIRV_INTERFACE_DESC}

**Task:** Render the Mandelbrot set using UVs mapped to c = (uv.x*3-2, uv.y*2-1). Iterate up to 20 steps, escape when |z|^2 > 4, and color with t=iterations/20 as (t, t*0.5, 1.0-t), alpha 1.0.
""",
    },
    {
        "description": "Brick Pattern",
        "prompt": f"""{BINARY_SPIRV_INTERFACE_DESC}

**Task:** Render a procedural brick pattern at 8x16 UV scale with staggered rows (half-brick offset every other row). Mortar thickness is 0.05 in X and 0.1 in Y; mortar color is (0.7, 0.7, 0.7) and brick color is (0.7, 0.2, 0.1). Apply Lambertian shading and output alpha 1.0.
""",
    },
    {
        "description": "Fresnel Heatmap",
        "prompt": f"""{BINARY_SPIRV_INTERFACE_DESC}

**Task:** Create a heatmap based on viewing angle using NdotV = max(dot(N, V), 0). If NdotV < 0.5 use (r=1, g=2*NdotV, b=0); otherwise use r=g=1-(NdotV-0.5)*2 and b=(NdotV-0.5)*2. Output alpha 1.0.
""",
    },
    {
        "description": "Full Phong Model",
        "prompt": f"""{BINARY_SPIRV_INTERFACE_DESC}

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
            "description": "Explain your approach to constructing this SPIR-V binary"
        },
        "spirv_hex": {
            "type": "string",
            "description": "Complete SPIR-V binary as a hex-encoded string (no spaces)"
        }
    },
    "required": ["reasoning", "spirv_hex"],
    "additionalProperties": False
}


def prepareSubpassPrompt(subPass: int) -> str:
    if subPass >= len(SUBPASSES):
        raise StopIteration
    return SUBPASSES[subPass]["prompt"]


extraGradeAnswerRuns = []
_renderer_instance = None
_OUTPUT_IMAGE_CACHE: Dict[Tuple[int, str], str] = {}


def _get_renderer():
    global _renderer_instance
    if _renderer_instance is None:
        from shader_test_utils import ShaderRenderer
        _renderer_instance = ShaderRenderer(512, 512)
    return _renderer_instance


def _grade_answer_inner(result: dict, subPass: int, aiEngineName: str) -> tuple:
    if not result:
        return 0.0, "No result provided", {"error": "no_result"}
    if "spirv_hex" not in result:
        return 0.0, "No SPIR-V hex data provided", {"error": "no_spirv_hex"}

    desc = SUBPASSES[subPass]["description"]
    hex_str = result["spirv_hex"]

    # Strip whitespace and validate hex
    hex_str = hex_str.strip().replace(" ", "").replace("\n", "").replace("\r", "")
    try:
        frag_spirv = bytes.fromhex(hex_str)
    except ValueError as e:
        return 0.0, f"[{desc}] Invalid hex string: {e}", {"error": str(e)}

    # Check SPIR-V magic number
    if len(frag_spirv) < 20:
        return 0.0, f"[{desc}] Binary too short ({len(frag_spirv)} bytes, minimum 20)", {
            "error": "binary_too_short"
        }
    magic = int.from_bytes(frag_spirv[0:4], 'little')
    if magic != 0x07230203:
        return 0.0, f"[{desc}] Invalid SPIR-V magic: 0x{magic:08X} (expected 0x07230203)", {
            "error": "bad_magic"
        }

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

    output_image = _save_rendered_image(45, subPass, aiEngineName, pixels)

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
    if not result or "spirv_hex" not in result:
        return 0.0, "No SPIR-V hex data provided"

    payload = {
        "spirv_hex": result.get("spirv_hex", ""),
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
        result = {"spirv_hex": payload.get("spirv_hex", "")}
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
    html = f"<h4>Binary SPIR-V - {desc}</h4>"
    if "reasoning" in result:
        r = result['reasoning'][:400] + ('...' if len(result.get('reasoning', '')) > 400 else '')
        html += f"<p><strong>Approach:</strong> {r}</p>"
    if "spirv_hex" in result:
        hex_data = result["spirv_hex"][:200] + ('...' if len(result.get('spirv_hex', '')) > 200 else '')
        html += f"<details><summary>View hex ({len(result['spirv_hex'])} chars)</summary><pre>{hex_data}</pre></details>"
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
<p>Construct GPU fragment shaders as raw SPIR-V binary data &mdash; a stream of
little-endian 32-bit words encoding every instruction, type, and decoration by
hand. The same 20 visual effects (lighting, patterns, fractals) must be produced,
but the AI outputs hex-encoded binary rather than text assembly.</p>
<p>This is the most extreme shader test: every opcode, operand, and ID must be
numerically correct at the byte level. The binary is validated, rendered on a
sphere, and compared pixel-by-pixel against the reference images.</p>
"""
