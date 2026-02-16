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

import os
import sys
from typing import Tuple, Optional

from shader_test_utils import (
    ShaderRenderer, validate_spirv,
    compare_images, load_reference, grade_shader_binary
)

title = "Binary SPIR-V Fragment Shaders"

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

**Task:** Output a solid red color: outColor = vec4(1.0, 0.0, 0.0, 1.0).
This is the simplest possible fragment shader. Just store a constant red color.

Hint: A minimal shader needs ~25-30 instructions. The hex output should be
around 200-400 hex characters.
""",
    },
    {
        "description": "Normal to RGB",
        "prompt": f"""{BINARY_SPIRV_INTERFACE_DESC}

**Task:** Visualize the surface normal as color.
Normalize the interpolated normal, then map from [-1,1] to [0,1]: rgb = N*0.5+0.5, a=1.0.
""",
    },
    {
        "description": "UV to RG",
        "prompt": f"""{BINARY_SPIRV_INTERFACE_DESC}

**Task:** Visualize texture coordinates as color.
outColor = vec4(uv.x, uv.y, 0.0, 1.0).
""",
    },
    {
        "description": "Vertex Color",
        "prompt": f"""{BINARY_SPIRV_INTERFACE_DESC}

**Task:** Pass through the per-vertex color directly.
outColor = vec4(color.r, color.g, color.b, 1.0).
""",
    },
    {
        "description": "Lambertian Diffuse",
        "prompt": f"""{BINARY_SPIRV_INTERFACE_DESC}

**Task:** Implement Lambertian diffuse lighting.
N = normalize(normal), L = normalize(lightPos.xyz - worldPos).
d = max(dot(N, L), 0.0). outColor = vec4(d, d, d, 1.0).
""",
    },
    {
        "description": "Phong Specular",
        "prompt": f"""{BINARY_SPIRV_INTERFACE_DESC}

**Task:** Implement Phong specular highlights.
N, L, V = normalized normal, light dir, view dir.
R = reflect(-L, N). spec = pow(max(dot(R,V),0), 32). diff = max(dot(N,L),0).
outColor = vec4(vec3(0.2 + 0.5*diff + 0.8*spec), 1.0).
""",
    },
    {
        "description": "Blinn-Phong",
        "prompt": f"""{BINARY_SPIRV_INTERFACE_DESC}

**Task:** Implement Blinn-Phong shading.
H = normalize(L+V). diff = max(dot(N,L),0). spec = pow(max(dot(N,H),0), 64).
outColor = vec4(vec3(0.1 + 0.6*diff + 0.8*spec), 1.0).
""",
    },
    {
        "description": "Rim Lighting",
        "prompt": f"""{BINARY_SPIRV_INTERFACE_DESC}

**Task:** Implement rim (Fresnel) lighting.
rim = pow(1 - max(dot(N,V),0), 3). base = vec3(0.2, 0.3, 0.8).
outColor = vec4(base*(0.3+0.5*diff) + vec3(1,0.8,0.5)*rim, 1.0).
""",
    },
    {
        "description": "Toon Shading",
        "prompt": f"""{BINARY_SPIRV_INTERFACE_DESC}

**Task:** Toon shading with 4 quantized bands.
d = max(dot(N,L),0). shade = d>0.75?1.0 : d>0.5?0.7 : d>0.25?0.4 : 0.2.
outColor = vec4(color * shade, 1.0).
""",
    },
    {
        "description": "Checkerboard",
        "prompt": f"""{BINARY_SPIRV_INTERFACE_DESC}

**Task:** Checkerboard pattern using UV coordinates.
checker = floor(uv.x*8) + floor(uv.y*8). Even=white(0.9), odd=dark(0.2).
Add diffuse shading. outColor = vec4(checkerColor * shade, 1.0).
""",
    },
    {
        "description": "Horizontal Stripes",
        "prompt": f"""{BINARY_SPIRV_INTERFACE_DESC}

**Task:** Horizontal stripes via sin(uv.y * pi * 16).
Positive = red(0.9,0.1,0.1), negative = blue(0.1,0.1,0.9). Add diffuse.
""",
    },
    {
        "description": "Polka Dots",
        "prompt": f"""{BINARY_SPIRV_INTERFACE_DESC}

**Task:** Polka dots pattern. Scale UV by 6, check distance from cell center.
dist < 0.3 = yellow(1,0.8,0), else = dark blue(0.1,0.1,0.3). Add diffuse.
""",
    },
    {
        "description": "Hemisphere Lighting",
        "prompt": f"""{BINARY_SPIRV_INTERFACE_DESC}

**Task:** Hemisphere ambient lighting.
blend = N.y*0.5+0.5. ambient = mix(ground(0.3,0.15,0.05), sky(0.4,0.6,1.0), blend).
Add directional: outColor = vec4(ambient + vec3(0.3)*d, 1.0).
""",
    },
    {
        "description": "Y-Gradient",
        "prompt": f"""{BINARY_SPIRV_INTERFACE_DESC}

**Task:** Color by world Y position.
t = clamp(worldPos.y*0.5+0.5, 0, 1). Mix green(0.1,0.4,0.1) to white(1,1,1).
""",
    },
    {
        "description": "Fog Effect",
        "prompt": f"""{BINARY_SPIRV_INTERFACE_DESC}

**Task:** Distance-based fog.
baseColor from diffuse lit red sphere. dist = length(worldPos - camPos).
fog = clamp((dist-1.5)/3, 0, 1). Mix base with fogColor(0.7,0.7,0.8).
""",
    },
    {
        "description": "Gooch Shading",
        "prompt": f"""{BINARY_SPIRV_INTERFACE_DESC}

**Task:** Gooch warm/cool shading.
t = dot(N,L)*0.5+0.5. cool=(0.2,0.2,0.75), warm=(0.7,0.7,0.4).
outColor = vec4(mix(cool, warm, t), 1.0).
""",
    },
    {
        "description": "Mandelbrot Set",
        "prompt": f"""{BINARY_SPIRV_INTERFACE_DESC}

**Task:** Mandelbrot set on UV coordinates.
c = (uv.x*3-2, uv.y*2-1). Iterate z=z²+c up to 20 times.
t = iter/20. Color = (t, t*0.5, 1-t, 1).
""",
    },
    {
        "description": "Brick Pattern",
        "prompt": f"""{BINARY_SPIRV_INTERFACE_DESC}

**Task:** Procedural bricks.
Scale UV (8x, 16y), offset alternating rows. Mortar if fract < threshold.
brick=(0.7,0.2,0.1), mortar=(0.7,0.7,0.7). Add diffuse shading.
""",
    },
    {
        "description": "Fresnel Heatmap",
        "prompt": f"""{BINARY_SPIRV_INTERFACE_DESC}

**Task:** Heatmap from viewing angle.
NdotV = max(dot(N,V),0). Map to red→yellow→blue color ramp.
<0.5: (1, NdotV*2, 0). >=0.5: (1-x*2, 1-x*2, x*2) where x=NdotV-0.5.
""",
    },
    {
        "description": "Full Phong Model",
        "prompt": f"""{BINARY_SPIRV_INTERFACE_DESC}

**Task:** Complete Phong model with colored material.
ambient=(0.1,0.05,0.05), diffuse=color*NdotL, specular=white*pow(RdotV,32).
outColor = vec4(clamp(ambient+diffuse+specular, 0, 1), 1.0).
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


def _get_renderer():
    global _renderer_instance
    if _renderer_instance is None:
        from shader_test_utils import ShaderRenderer
        _renderer_instance = ShaderRenderer(512, 512)
    return _renderer_instance


def gradeAnswer(result: dict, subPass: int, aiEngineName: str) -> tuple:
    if not result:
        return 0.0, "No result provided"
    if "spirv_hex" not in result:
        return 0.0, "No SPIR-V hex data provided"

    desc = SUBPASSES[subPass]["description"]
    hex_str = result["spirv_hex"]

    # Strip whitespace and validate hex
    hex_str = hex_str.strip().replace(" ", "").replace("\n", "").replace("\r", "")
    try:
        frag_spirv = bytes.fromhex(hex_str)
    except ValueError as e:
        return 0.0, f"[{desc}] Invalid hex string: {e}"

    # Check SPIR-V magic number
    if len(frag_spirv) < 20:
        return 0.0, f"[{desc}] Binary too short ({len(frag_spirv)} bytes, minimum 20)"
    magic = int.from_bytes(frag_spirv[0:4], 'little')
    if magic != 0x07230203:
        return 0.0, f"[{desc}] Invalid SPIR-V magic: 0x{magic:08X} (expected 0x07230203)"

    try:
        renderer = _get_renderer()
    except Exception as e:
        return 0.0, f"[{desc}] Failed to create renderer: {e}"

    score, explanation = grade_shader_binary(
        frag_spirv, test_num=45, subpass=subPass, renderer=renderer,
        color_tolerance=2, spatial_tolerance=1, ref_test_num=41
    )
    return score, f"[{desc}] {explanation}"


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
    return html


highLevelSummary = """
Binary SPIR-V Fragment Shaders tests the ability to construct GPU shaders as raw SPIR-V binary data.

**Key concepts:**
- SPIR-V binary format (little-endian 32-bit words, magic number, header, instruction stream)
- Instruction encoding: (word_count << 16) | opcode
- Type system (void, float, vectors, pointers)
- Decoration system (locations, bindings, offsets)
- Same 20 rendering tasks as test 41
- This is the hardest shader test: requires byte-level understanding of SPIR-V
"""
