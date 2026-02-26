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

import json
import os
import subprocess
import sys
import tempfile
from typing import Tuple, Optional, Dict
from PIL import Image

from shader_test_utils import (
  ShaderRenderer, assemble_spirv, validate_spirv,
  compare_images, load_reference, save_reference, grade_shader,
  get_reference_path, image_pair_html
)

title = "Fragment Shaders (SPIR-V Assembly)"

TIMEOUT_SECONDS = 60

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

**Task:** Output a solid red color (vec4(1.0, 0.0, 0.0, 1.0)).
""",
  },

  # 1: Normal visualization
  {
    "description": "Normal to RGB",
    "prompt": f"""{SPIRV_INTERFACE_DESC}

**Task:** Visualize the surface normal as color by mapping the normalized normal from [-1, 1] into [0, 1] with alpha 1.0.
""",
  },

  # 2: UV visualization
  {
    "description": "UV to RG",
    "prompt": f"""{SPIRV_INTERFACE_DESC}

**Task:** Visualize texture coordinates as color (R=uv.x, G=uv.y, B=0, A=1).
""",
  },

  # 3: Vertex color passthrough
  {
    "description": "Vertex Color",
    "prompt": f"""{SPIRV_INTERFACE_DESC}

**Task:** Pass through the per-vertex color directly with alpha 1.0.
""",
  },

  # 4: Lambertian diffuse
  {
    "description": "Lambertian Diffuse",
    "prompt": f"""{SPIRV_INTERFACE_DESC}

**Task:** Implement Lambertian diffuse lighting using N=normalize(normal), L=normalize(lightPos.xyz - worldPos), and grayscale intensity d=max(dot(N,L),0) with alpha 1.0.
""",
  },

  # 5: Phong specular
  {
    "description": "Phong Specular",
    "prompt": f"""{SPIRV_INTERFACE_DESC}

**Task:** Implement Phong specular highlights using N, L, V and R=reflect(-L,N), with diffuse=max(dot(N,L),0) and specular=pow(max(dot(R,V),0),32). Output grayscale intensity (0.2 + 0.5*diffuse + 0.8*specular) with alpha 1.0.
""",
  },

  # 6: Blinn-Phong
  {
    "description": "Blinn-Phong",
    "prompt": f"""{SPIRV_INTERFACE_DESC}

**Task:** Implement Blinn-Phong shading using N, L, V and half-vector H=normalize(L+V), with diffuse=max(dot(N,L),0) and specular=pow(max(dot(N,H),0),64). Output grayscale intensity (0.1 + 0.6*diffuse + 0.8*specular) with alpha 1.0.
""",
  },

  # 7: Rim/Fresnel lighting
  {
    "description": "Rim Lighting",
    "prompt": f"""{SPIRV_INTERFACE_DESC}

**Task:** Implement rim (Fresnel) lighting with rim = pow(1 - max(dot(N,V),0), 3). Use base color (0.2, 0.3, 0.8) modulated by (0.3 + 0.5*diffuse), and add rim tint (1.0, 0.8, 0.5) * rim. Output alpha 1.0.
""",
  },

  # 8: Toon/cel shading
  {
    "description": "Toon Shading",
    "prompt": f"""{SPIRV_INTERFACE_DESC}

**Task:** Implement toon/cel shading by quantizing the Lambertian diffuse term d=max(dot(N,L),0) into bands with thresholds 0.75/0.5/0.25 and values 1.0/0.7/0.4/0.2. Multiply the vertex color by the band and output alpha 1.0.
""",
  },

  # 9: Checkerboard pattern
  {
    "description": "Checkerboard",
    "prompt": f"""{SPIRV_INTERFACE_DESC}

**Task:** Render a UV checkerboard at 8x8 frequency, alternating colors 0.9 and 0.2 by parity. Apply Lambertian shading with factor (0.5 + 0.5*diffuse) and output alpha 1.0.
""",
  },

  # 10: Procedural stripes
  {
    "description": "Horizontal Stripes",
    "prompt": f"""{SPIRV_INTERFACE_DESC}

**Task:** Render horizontal stripes driven by sin(uv.y * 3.14159 * 16.0), alternating red (0.9, 0.1, 0.1) and blue (0.1, 0.1, 0.9). Apply Lambertian shading with factor (0.4 + 0.6*diffuse) and output alpha 1.0.
""",
  },

  # 11: Procedural dots/circles
  {
    "description": "Polka Dots",
    "prompt": f"""{SPIRV_INTERFACE_DESC}

**Task:** Render a 6x6 UV polka-dot pattern with circular dots of radius 0.3 (in cell space). Dots are yellow (1.0, 0.8, 0.0) on a dark blue background (0.1, 0.1, 0.3). Apply Lambertian shading with factor (0.3 + 0.7*diffuse) and output alpha 1.0.
""",
  },

  # 12: Hemisphere lighting
  {
    "description": "Hemisphere Lighting",
    "prompt": f"""{SPIRV_INTERFACE_DESC}

**Task:** Implement hemisphere ambient lighting with skyColor (0.4, 0.6, 1.0) and groundColor (0.3, 0.15, 0.05) blended by N.y * 0.5 + 0.5. Add a small diffuse term 0.3*d and output alpha 1.0.
""",
  },

  # 13: World-space gradient
  {
    "description": "Y-Gradient",
    "prompt": f"""{SPIRV_INTERFACE_DESC}

**Task:** Color based on world-space Y: t=clamp(worldPos.y * 0.5 + 0.5), mix bottom (0.1, 0.4, 0.1) to top (1.0, 1.0, 1.0), output alpha 1.0.
""",
  },

  # 14: Distance-based fog
  {
    "description": "Fog Effect",
    "prompt": f"""{SPIRV_INTERFACE_DESC}

**Task:** Apply distance-based fog: base color is (0.8, 0.2, 0.2) with Lambertian factor (0.3 + 0.7*diffuse). Fog factor is clamp((dist - 1.5) / 3.0), fog color (0.7, 0.7, 0.8), and output the mix with alpha 1.0.
""",
  },

  # 15: Gooch shading
  {
    "description": "Gooch Shading",
    "prompt": f"""{SPIRV_INTERFACE_DESC}

**Task:** Implement Gooch warm/cool shading with cool (0.2, 0.2, 0.75), warm (0.7, 0.7, 0.4), and t = dot(N,L) * 0.5 + 0.5. Output alpha 1.0.
""",
  },

  # 16: Mandelbrot on UVs
  {
    "description": "Mandelbrot Set",
    "prompt": f"""{SPIRV_INTERFACE_DESC}

**Task:** Render the Mandelbrot set using UVs mapped to c = (uv.x*3-2, uv.y*2-1). Iterate up to 20 steps, escape when |z|^2 > 4, and color with t=iterations/20 as (t, t*0.5, 1.0-t), alpha 1.0.
""",
  },

  # 17: Procedural brick pattern
  {
    "description": "Brick Pattern",
    "prompt": f"""{SPIRV_INTERFACE_DESC}

**Task:** Render a procedural brick pattern at 8x16 UV scale with staggered rows (half-brick offset every other row). Mortar thickness is 0.05 in X and 0.1 in Y; mortar color is (0.7, 0.7, 0.7) and brick color is (0.7, 0.2, 0.1). Apply Lambertian shading with factor (0.4 + 0.6*diffuse) and output alpha 1.0.
""",
  },

  # 18: Fresnel heatmap
  {
    "description": "Fresnel Heatmap",
    "prompt": f"""{SPIRV_INTERFACE_DESC}

**Task:** Create a heatmap based on viewing angle using NdotV = max(dot(N, V), 0). If NdotV < 0.5 use (r=1, g=2*NdotV, b=0); otherwise use r=g=1-(NdotV-0.5)*2 and b=(NdotV-0.5)*2. Output alpha 1.0.
""",
  },

  # 19: Combined diffuse + specular + ambient
  {
    "description": "Full Phong Model",
    "prompt": f"""{SPIRV_INTERFACE_DESC}

**Task:** Complete Phong lighting with colored materials: ambient (0.1, 0.05, 0.05), diffuse uses vertex color with max(dot(N,L),0), specular uses (1,1,1) with shininess 32 and R=reflect(-L,N). Clamp the sum to [0,1], output alpha 1.0.
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
_OUTPUT_IMAGE_CACHE: Dict[Tuple[int, str], str] = {}


def _get_renderer() -> ShaderRenderer:
  global _renderer_instance
  if _renderer_instance is None:
    _renderer_instance = ShaderRenderer(512, 512)
  return _renderer_instance


def _grade_answer_inner(result: dict, subPass: int, aiEngineName: str) -> tuple:
  """
  Grade the SPIR-V assembly fragment shader.
  Assembles it, renders on the sphere, compares to reference image.
  """
  if not result:
    return 0.0, "No result provided", {"error": "no_result"}

  if "spirv_code" not in result:
    return 0.0, "No SPIR-V code provided", {"error": "no_spirv_code"}

  spirv_text = result["spirv_code"]
  desc = SUBPASSES[subPass]["description"]

  try:
    renderer = _get_renderer()
  except Exception as e:
    return 0.0, f"[{desc}] Failed to create renderer: {e}", {"error": str(e)}

  try:
    frag_spirv = assemble_spirv(spirv_text)
  except RuntimeError as e:
    return 0.0, f"[{desc}] SPIR-V assembly failed: {e}", {"error": str(e)}

  valid, err = validate_spirv(frag_spirv)
  if not valid:
    return 0.0, f"[{desc}] SPIR-V validation failed: {err}", {"error": err}

  try:
    pixels = renderer.render(frag_spirv)
  except Exception as e:
    return 0.0, f"[{desc}] Rendering failed: {e}", {"error": str(e)}

  output_image = _save_rendered_image(41, subPass, aiEngineName, pixels)

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
  if not result or "spirv_code" not in result:
    return 0.0, "No SPIR-V code provided"

  payload = {
    "spirv_code": result.get("spirv_code", ""),
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
    result = {"spirv_code": payload.get("spirv_code", "")}
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
  html = f"<h4>SPIR-V Shader - {desc}</h4>"
  if "reasoning" in result:
    r = result['reasoning'][:400] + ('...' if len(result.get('reasoning', '')) > 400 else '')
    html += f"<p><strong>Approach:</strong> {r}</p>"
  if "spirv_code" in result:
    code = result["spirv_code"].replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    html += f"<details><summary>View SPIR-V ({len(result['spirv_code'])} chars)</summary><pre>{code}</pre></details>"
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
<p>Write GPU fragment shaders in raw SPIR-V assembly &mdash; the low-level instruction
set that graphics cards actually execute. Each subpass asks for a different visual
effect (lighting, patterns, fractals) rendered onto a 3D sphere, and the output
image is compared pixel-by-pixel against a reference.</p>
<p>This is the hardest shader-language test because SPIR-V assembly is essentially
GPU machine code: every operation, type declaration, and memory access must be
spelled out by hand.</p>
"""
