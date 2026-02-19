"""
Test 42: SPIR-V Assembly Geometry Shaders

The LLM must write geometry shaders in SPIR-V assembly text.
Each subpass uses a different input primitive type and geometry task.
The vertex shader and fragment shader are fixed (provided by framework).
The AI writes only the geometry shader.

Input primitive types tested:
  - Points (point list)
  - Lines (line list)
  - Triangles (triangle list, sphere mesh)
  - Lines with adjacency (line list with adjacency, smooth curve)
  - Triangles with adjacency (triangle list with adjacency, grid mesh)

Vertex format (all primitive types):
  layout(location = 0) in vec3 position;
  layout(location = 1) in vec3 normal;
  layout(location = 2) in vec3 color;

The fixed vertex shader passes these through to the geometry shader as:
  gl_PerVertex { vec4 gl_Position; }  (position with w=1)
  layout(location = 0) vec3 worldPos;
  layout(location = 1) vec3 normal;
  layout(location = 2) vec3 color;

Available uniform buffer (set=0, binding=0) - same as test 41:
  mat4 model;       // offset 0
  mat4 view;        // offset 64
  mat4 projection;  // offset 128
  vec4 lightPos;    // offset 192  (5, 5, 5, 1)
  vec4 cameraPos;   // offset 208  (0, 0, 3, 1)
  vec4 params;      // offset 224  (0, 0.5, 0, 0)

Geometry shader must output:
  gl_PerVertex { vec4 gl_Position; }
  layout(location = 0) out vec3 outColor;

Fragment shader outputs: vec4(inColor, 1.0)
"""

import os
import sys
from typing import Tuple, Optional, Dict
from PIL import Image

from shader_test_utils import (
    assemble_spirv, validate_spirv, compare_images,
    load_reference, save_reference, UBO_SIZE,
    get_reference_path, image_pair_html
)

title = "SPIR-V Assembly Geometry Shaders"

# ---------------------------------------------------------------------------
# Fixed vertex shader (passes attributes through to geometry shader)
# ---------------------------------------------------------------------------

VERT_SHADER_ASM = """\
; SPIR-V
; Version: 1.0
               OpCapability Shader
               OpMemoryModel Logical GLSL450
               OpEntryPoint Vertex %main "main" %inPos %inNormal %inColor %gl_out %outPos %outNormal %outColor
               OpDecorate %inPos Location 0
               OpDecorate %inNormal Location 1
               OpDecorate %inColor Location 2
               OpDecorate %outPos Location 0
               OpDecorate %outNormal Location 1
               OpDecorate %outColor Location 2
               OpDecorate %gl_PerVertex_out Block
               OpMemberDecorate %gl_PerVertex_out 0 BuiltIn Position

       %void = OpTypeVoid
       %func = OpTypeFunction %void
      %float = OpTypeFloat 32
        %int = OpTypeInt 32 1
      %int_0 = OpConstant %int 0
    %v3float = OpTypeVector %float 3
    %v4float = OpTypeVector %float 4
  %ptr_in_v3 = OpTypePointer Input %v3float
 %ptr_out_v3 = OpTypePointer Output %v3float
 %ptr_out_v4 = OpTypePointer Output %v4float
%gl_PerVertex_out = OpTypeStruct %v4float
 %ptr_out_pv = OpTypePointer Output %gl_PerVertex_out
     %gl_out = OpVariable %ptr_out_pv Output
      %inPos = OpVariable %ptr_in_v3 Input
   %inNormal = OpVariable %ptr_in_v3 Input
    %inColor = OpVariable %ptr_in_v3 Input
     %outPos = OpVariable %ptr_out_v3 Output
  %outNormal = OpVariable %ptr_out_v3 Output
   %outColor = OpVariable %ptr_out_v3 Output
    %float_1 = OpConstant %float 1

       %main = OpFunction %void None %func
      %entry = OpLabel
        %pos = OpLoad %v3float %inPos
       %norm = OpLoad %v3float %inNormal
        %col = OpLoad %v3float %inColor
       %pos4 = OpCompositeConstruct %v4float %pos %float_1
   %gp_ptr   = OpAccessChain %ptr_out_v4 %gl_out %int_0
               OpStore %gp_ptr %pos4
               OpStore %outPos %pos
               OpStore %outNormal %norm
               OpStore %outColor %col
               OpReturn
               OpFunctionEnd
"""

# ---------------------------------------------------------------------------
# Fixed fragment shader (outputs inColor with alpha 1.0)
# ---------------------------------------------------------------------------

FRAG_SHADER_ASM = """\
; SPIR-V
; Version: 1.0
               OpCapability Shader
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %main "main" %inColor %outColor
               OpExecutionMode %main OriginUpperLeft
               OpDecorate %inColor Location 0
               OpDecorate %outColor Location 0
       %void = OpTypeVoid
       %func = OpTypeFunction %void
      %float = OpTypeFloat 32
    %v3float = OpTypeVector %float 3
    %v4float = OpTypeVector %float 4
  %ptr_in_v3 = OpTypePointer Input %v3float
 %ptr_out_v4 = OpTypePointer Output %v4float
    %inColor = OpVariable %ptr_in_v3 Input
   %outColor = OpVariable %ptr_out_v4 Output
    %float_1 = OpConstant %float 1
       %main = OpFunction %void None %func
      %entry = OpLabel
        %col = OpLoad %v3float %inColor
          %r = OpCompositeExtract %float %col 0
          %g = OpCompositeExtract %float %col 1
          %b = OpCompositeExtract %float %col 2
        %out = OpCompositeConstruct %v4float %r %g %b %float_1
               OpStore %outColor %out
               OpReturn
               OpFunctionEnd
"""

# ---------------------------------------------------------------------------
# Common prompt preamble
# ---------------------------------------------------------------------------

GEOM_INTERFACE_DESC = """\
You are writing a SPIR-V assembly geometry shader. It will be assembled with
spirv-as (Vulkan 1.0) and rendered using a fixed vertex shader and fragment shader.

**Geometry shader requirements:**
- `OpCapability Geometry`
- `OpMemoryModel Logical GLSL450`
- `OpEntryPoint Geometry %main "main"` listing ALL interface variables
- `OpExecutionMode %main <InputPrimitive>` (InputPoints/InputLines/Triangles/InputLinesAdjacency/InputTrianglesAdjacency)
- `OpExecutionMode %main <OutputPrimitive>` (OutputPoints/OutputLineStrip/OutputTriangleStrip)
- `OpExecutionMode %main OutputVertices <N>` (max vertices emitted)
- `OpExecutionMode %main Invocations 1`

**Input from vertex shader (per-vertex arrays):**
The input gl_Position is in a gl_PerVertex block (must be decorated with Block + BuiltIn Position).
```
%gl_PerVertex_in = OpTypeStruct %v4float
OpDecorate %gl_PerVertex_in Block
OpMemberDecorate %gl_PerVertex_in 0 BuiltIn Position
; Array size depends on input primitive:
;   InputPoints: 1, InputLines: 2, Triangles: 3
;   InputLinesAdjacency: 4, InputTrianglesAdjacency: 6
%arr_pv_in = OpTypeArray %gl_PerVertex_in %uint_N
%ptr_in_arr_pv = OpTypePointer Input %arr_pv_in
%gl_in = OpVariable %ptr_in_arr_pv Input
; Access: OpAccessChain %ptr_in_v4 %gl_in %int_i %int_0  -> gl_in[i].gl_Position
```

User-defined inputs are also arrays of the same size:
```
layout(location = 0) in vec3 worldPos[];   ; positions in clip space
layout(location = 1) in vec3 normal[];     ; vertex normals
layout(location = 2) in vec3 color[];      ; per-vertex colors
```

**Output (per emitted vertex):**
The output gl_Position is in a gl_PerVertex block:
```
%gl_PerVertex_out = OpTypeStruct %v4float
OpDecorate %gl_PerVertex_out Block
OpMemberDecorate %gl_PerVertex_out 0 BuiltIn Position
%ptr_out_pv = OpTypePointer Output %gl_PerVertex_out
%gl_out = OpVariable %ptr_out_pv Output
; Store: OpAccessChain %ptr_out_v4 %gl_out %int_0 -> gl_out.gl_Position
```
```
layout(location = 0) out vec3 outColor;  ; color for fragment shader
```

**Emit vertices with:** `OpEmitVertex` after storing gl_Position and outColor.
**End primitive with:** `OpEndPrimitive` after emitting a complete strip.

**Available UBO (set=0, binding=0) - accessible from geometry shader:**
Same layout as test 41 (model/view/proj matrices, lightPos, cameraPos, params).

**Write ONLY the complete SPIR-V assembly text for the geometry shader.**
"""

# ---------------------------------------------------------------------------
# Primitive type info for array sizes
# ---------------------------------------------------------------------------

PRIM_ARRAY_SIZES = {
    "points": 1,
    "lines": 2,
    "triangles": 3,
    "line_adjacency": 4,
    "triangle_adjacency": 6,
}

# ---------------------------------------------------------------------------
# Subpass definitions
# ---------------------------------------------------------------------------

SUBPASSES = [
    # ===== POINTS (subpasses 0-2) =====
    {
        "description": "Points to Quads",
        "topology": "points",
        "prompt": f"""{GEOM_INTERFACE_DESC}

**Input primitive:** InputPoints (1 vertex per invocation, array size 1)
**Geometry:** 25 colored points in a 5x5 grid spread across [-0.8, 0.8].

**Task:** Expand each point into a small colored quad (4 vertices as triangle strip).
1. Read gl_in[0].gl_Position (the point position).
2. Read color[0] (the point's color).
3. Define a half-size: h = 0.04.
4. Emit 4 vertices forming a quad centered on the point:
   - (x-h, y-h), (x+h, y-h), (x-h, y+h), (x+h, y+h)
   - All with z=0, w=1, and the same color.
5. Output as triangle strip, max 4 vertices.
""",
    },
    {
        "description": "Points to Diamonds",
        "topology": "points",
        "prompt": f"""{GEOM_INTERFACE_DESC}

**Input primitive:** InputPoints (1 vertex per invocation, array size 1)
**Geometry:** 25 colored points in a 5x5 grid.

**Task:** Expand each point into a diamond shape (4 triangles forming a diamond).
1. Read the point position and color.
2. Define half-size h = 0.05.
3. Emit 4 vertices as a triangle strip forming a diamond:
   - Top: (x, y+h), Right: (x+h, y), Bottom: (x, y-h), Left: (x-h, y)
   - Emit order for triangle strip: top, right, left, bottom.
4. Use the point's color for all vertices.
5. Output as triangle strip, max 4 vertices.
""",
    },
    {
        "description": "Points to Hexagons",
        "topology": "points",
        "prompt": f"""{GEOM_INTERFACE_DESC}

**Input primitive:** InputPoints (1 vertex per invocation, array size 1)
**Geometry:** 25 colored points in a 5x5 grid.

**Task:** Expand each point into a hexagon (circle approximation).
1. Read the point position and color.
2. Define radius r = 0.04.
3. Emit a triangle fan as triangle strip: center vertex + 7 edge vertices
   (0°, 60°, 120°, 180°, 240°, 300°, 360°) interleaved with center.
   Triangle strip pattern: center, edge0, edge1, center, edge1, edge2, ...
   OR: emit center, then all 7 edge vertices (center + 6 sides + close = 8 total as fan).
   Simplest: emit as triangle strip: V0=center, V1=edge0, V2=edge1, V3=edge2, V4=edge3, V5=edge4, V6=edge5, V7=edge0 (closing).
   Actually for a fan via triangle strip: center, e0, e1, center, e2, e3, center, e4, e5 won't work.
   Better: just emit 6 triangles via 6 separate triangle strips, or use a single strip:
   center, e0, e1, e2, e3, e4, e5, e0 where the strip auto-forms triangles.
   Wait - triangle strip: center(0), e0(1), e1(2) -> tri. Then e0, e1, e2 -> wrong.
   
   Simplest correct approach: emit 6 separate 3-vertex strips (18 vertices total, or use EndPrimitive between each).
   Or emit as: center, e0, e1, EndPrimitive, center, e1, e2, EndPrimitive, ... (18 verts, 6 primitives).
4. Use point color for center, slightly darker for edges.
5. OutputTriangleStrip, OutputVertices 18.
""",
    },

    # ===== LINES (subpasses 3-5) =====
    {
        "description": "Thick Lines",
        "topology": "lines",
        "prompt": f"""{GEOM_INTERFACE_DESC}

**Input primitive:** InputLines (2 vertices per invocation, array size 2)
**Geometry:** 8 radial line segments from center outward.

**Task:** Expand each thin line into a thick ribbon (quad).
1. Read gl_in[0].gl_Position and gl_in[1].gl_Position (line endpoints).
2. Read color[0] and color[1].
3. Compute line direction: dir = p1 - p0.
4. Compute perpendicular: perp = normalize(-dir.y, dir.x) * thickness, where thickness = 0.03.
5. Emit 4 vertices as triangle strip:
   - p0 + perp, p0 - perp, p1 + perp, p1 - perp.
6. Interpolate color from c0 to c1.
7. OutputTriangleStrip, OutputVertices 4.
""",
    },
    {
        "description": "Tapered Lines",
        "topology": "lines",
        "prompt": f"""{GEOM_INTERFACE_DESC}

**Input primitive:** InputLines (2 vertices per invocation, array size 2)
**Geometry:** 8 radial line segments from center outward.

**Task:** Expand each line into a tapered ribbon (thick at start, thin at end).
1. Read line endpoints and colors.
2. Compute perpendicular direction (same as thick lines).
3. Use thick_start = 0.04 at p0, thick_end = 0.005 at p1.
4. Emit 4 vertices as triangle strip:
   - p0 + perp*thick_start, p0 - perp*thick_start,
   - p1 + perp*thick_end, p1 - perp*thick_end.
5. Color: c0 at start vertices, c1 at end vertices.
6. OutputTriangleStrip, OutputVertices 4.
""",
    },
    {
        "description": "Arrow Lines",
        "topology": "lines",
        "prompt": f"""{GEOM_INTERFACE_DESC}

**Input primitive:** InputLines (2 vertices per invocation, array size 2)
**Geometry:** 8 radial line segments from center outward.

**Task:** Draw each line as a thick shaft with an arrowhead at the end.
1. Read line endpoints p0, p1 and colors c0, c1.
2. Compute direction dir = p1 - p0, and perpendicular perp = normalize(-dir.y, dir.x).
3. Shaft: thickness = 0.015. Arrow starts at 75% along the line.
   mid = p0 + 0.75 * (p1 - p0).
4. Emit shaft as triangle strip (4 verts): p0±perp*0.015, mid±perp*0.015. EndPrimitive.
5. Emit arrowhead as triangle strip (3 verts):
   mid + perp*0.05, mid - perp*0.05, p1 (tip). EndPrimitive.
6. Shaft colored c0, arrowhead colored c1.
7. OutputTriangleStrip, OutputVertices 7.
""",
    },

    # ===== TRIANGLES (subpasses 6-8) =====
    {
        "description": "Triangle Passthrough",
        "topology": "triangles",
        "prompt": f"""{GEOM_INTERFACE_DESC}

**Input primitive:** Triangles (3 vertices per invocation, array size 3)
**Geometry:** A sphere mesh (triangle list with indices).

**Task:** Identity/passthrough geometry shader. Simply re-emit the input triangle unchanged.
1. For each of the 3 input vertices (i = 0, 1, 2):
   - Store gl_in[i].gl_Position to output gl_Position.
   - Store color[i] to outColor.
   - OpEmitVertex.
2. OpEndPrimitive.
3. This should produce an image identical to rendering without a geometry shader.
4. OutputTriangleStrip, OutputVertices 3.
""",
    },
    {
        "description": "Exploded Triangles",
        "topology": "triangles",
        "prompt": f"""{GEOM_INTERFACE_DESC}

**Input primitive:** Triangles (3 vertices per invocation, array size 3)
**Geometry:** A sphere mesh.

**Task:** Explode each triangle outward along its face normal.
1. Read the 3 vertex positions from gl_in[0..2].gl_Position.
2. Compute face center: center = (p0 + p1 + p2) / 3.
3. Compute face normal using cross product: e1 = p1-p0, e2 = p2-p0, N = normalize(cross(e1, e2)).
4. Offset = N * 0.08 (push outward).
5. Emit 3 vertices, each shifted by the offset: gl_Position = pI + vec4(offset, 0).
6. Color each vertex with its original vertex color.
7. OutputTriangleStrip, OutputVertices 3.

Use GLSL.std.450 for Normalize (69), Cross (68) if needed, or compute manually.
The cross product of vec3 (a,b,c) x (d,e,f) = (bf-ce, cd-af, ae-bd).
""",
    },
    {
        "description": "Shrunk Triangles",
        "topology": "triangles",
        "prompt": f"""{GEOM_INTERFACE_DESC}

**Input primitive:** Triangles (3 vertices per invocation, array size 3)
**Geometry:** A sphere mesh.

**Task:** Shrink each triangle toward its centroid, creating visible gaps between faces.
1. Read the 3 vertex positions from gl_in[0..2].gl_Position.
2. Compute centroid: C = (p0 + p1 + p2) / 3.
3. Shrink factor: s = 0.7.
4. For each vertex i: new_pos = C + s * (pI - C).
5. Emit 3 shrunk vertices with their original colors.
6. OutputTriangleStrip, OutputVertices 3.
""",
    },

    # ===== LINE ADJACENCY (subpasses 9-11) =====
    {
        "description": "Smooth Thick Curve",
        "topology": "line_adjacency",
        "prompt": f"""{GEOM_INTERFACE_DESC}

**Input primitive:** InputLinesAdjacency (4 vertices per invocation, array size 4)
**Geometry:** A sine wave curve. Vertices: [prev, start, end, next].
  gl_in[0] = previous point, gl_in[1] = line start, gl_in[2] = line end, gl_in[3] = next point.

**Task:** Render a thick smooth curve using adjacency for proper miter joins.
1. Read all 4 positions. The actual line segment is from gl_in[1] to gl_in[2].
2. Compute tangent at start: t0 = normalize(gl_in[2].pos - gl_in[0].pos).
3. Compute tangent at end: t1 = normalize(gl_in[3].pos - gl_in[1].pos).
4. Compute perpendiculars: perp0 = vec2(-t0.y, t0.x), perp1 = vec2(-t1.y, t1.x).
5. Thickness = 0.025.
6. Emit 4 vertices as triangle strip:
   start + perp0*thickness, start - perp0*thickness,
   end + perp1*thickness, end - perp1*thickness.
7. Use colors from color[1] and color[2].
8. OutputTriangleStrip, OutputVertices 4.
""",
    },
    {
        "description": "Thick Bezier Curve",
        "topology": "line_adjacency",
        "prompt": f"""{GEOM_INTERFACE_DESC}

**Input primitive:** InputLinesAdjacency (4 vertices per invocation, array size 4)
**Geometry:** A sine wave curve with adjacency.

**Task:** Render a thick smooth curve by subdividing the segment into multiple quads.
1. Use the 4 control points for Catmull-Rom-style interpolation.
   P0=gl_in[0], P1=gl_in[1] (start), P2=gl_in[2] (end), P3=gl_in[3].
2. Subdivide the segment into 4 sub-segments (5 interpolated points along P1→P2).
3. For each interpolated point, compute a perpendicular for thickness = 0.02.
4. Emit a continuous triangle strip: for each of the 5 points, emit pos+perp and pos-perp.
5. Total: 10 vertices as a single triangle strip.
6. Interpolate color from color[1] to color[2].
7. OutputTriangleStrip, OutputVertices 10.

For Catmull-Rom: lerp between P1 and P2, using P0 and P3 to influence curvature.
Simplified: just linearly interpolate between P1 and P2 for positions, but use
(P2-P0) and (P3-P1) for tangent directions at start/end to get smooth perpendiculars.
""",
    },
    {
        "description": "Dashed Curve",
        "topology": "line_adjacency",
        "prompt": f"""{GEOM_INTERFACE_DESC}

**Input primitive:** InputLinesAdjacency (4 vertices per invocation, array size 4)
**Geometry:** A sine wave curve with adjacency.

**Task:** Render a dashed thick curve (alternating visible/invisible segments).
1. Read start = gl_in[1].gl_Position, end = gl_in[2].gl_Position.
2. Compute adjacency tangents for smooth perpendiculars (same as smooth thick curve).
3. Thickness = 0.02.
4. Split the segment into 3 equal parts. Only draw the 1st and 3rd parts (skip middle = dash gap).
5. For each drawn part, emit 4 vertices as a triangle strip quad.
6. Total: 2 quads × 4 vertices = 8 vertices, with EndPrimitive between them.
7. Color from color[1] and color[2], interpolated.
8. OutputTriangleStrip, OutputVertices 8.
""",
    },

    # ===== TRIANGLE ADJACENCY (subpasses 12-14) =====
    {
        "description": "Silhouette Edges",
        "topology": "triangle_adjacency",
        "prompt": f"""{GEOM_INTERFACE_DESC}

**Input primitive:** InputTrianglesAdjacency (6 vertices per invocation, array size 6)
**Geometry:** A grid mesh with triangle adjacency indices.
  Vertices 0,2,4 are the triangle. Vertices 1,3,5 are adjacent vertices.
  - Edge 0→2: adjacent vertex is 1
  - Edge 2→4: adjacent vertex is 3
  - Edge 4→0: adjacent vertex is 5

**Task:** Detect and render silhouette/boundary edges as thick lines.
1. The main triangle has vertices at indices 0, 2, 4 in gl_in.
2. For each edge of the triangle, check if the adjacent triangle faces away
   (or doesn't exist / is degenerate). A simple heuristic: if the adjacent
   vertex (1, 3, or 5) has the same position as one of the main vertices,
   it's a boundary edge.
3. For each silhouette/boundary edge found, emit a thick quad (4 vertices as
   triangle strip, thickness 0.015).
4. Also emit the original triangle (3 verts) with dimmed colors (color * 0.3).
5. Edge color: white (1,1,1). Triangle fill: dark version of vertex color.
6. OutputTriangleStrip, OutputVertices 15 (triangle=3 + up to 3 edges × 4=12).

Simplified approach: just emit all 3 edges as thick lines plus the filled triangle.
""",
    },
    {
        "description": "Wireframe from Adjacency",
        "topology": "triangle_adjacency",
        "prompt": f"""{GEOM_INTERFACE_DESC}

**Input primitive:** InputTrianglesAdjacency (6 vertices per invocation, array size 6)
**Geometry:** A grid mesh with adjacency.

**Task:** Render wireframe: emit each edge of the triangle as a thin quad.
1. Main triangle vertices are at gl_in indices 0, 2, 4.
2. For each of the 3 edges (0→2, 2→4, 4→0), emit a thin quad (thickness 0.008).
3. For each edge, compute perpendicular and emit 4 vertices as triangle strip.
4. Edge color = average of the two endpoint colors.
5. Emit EndPrimitive after each edge quad.
6. OutputTriangleStrip, OutputVertices 12 (3 edges × 4 verts).
""",
    },
    {
        "description": "Flat-Shaded Triangles",
        "topology": "triangle_adjacency",
        "prompt": f"""{GEOM_INTERFACE_DESC}

**Input primitive:** InputTrianglesAdjacency (6 vertices per invocation, array size 6)
**Geometry:** A grid mesh with adjacency.

**Task:** Flat-shade each triangle using the face normal for simple lighting.
1. Main triangle vertices at gl_in indices 0, 2, 4.
2. Compute face normal: N = normalize(cross(p2-p0, p4-p0)).
3. Light direction: L = normalize(vec3(1, 1, 1)).
4. Diffuse = max(dot(N, L), 0.0).
5. Base color = average of color[0], color[2], color[4].
6. Final color = base * (0.2 + 0.8 * diffuse).
7. Emit the 3 triangle vertices with flat-shaded color.
8. OutputTriangleStrip, OutputVertices 3.

Compute cross product manually: (a×b) = (a.y*b.z-a.z*b.y, a.z*b.x-a.x*b.z, a.x*b.y-a.y*b.x).
Use GLSL.std.450 Normalize(69), FMax(40) for the lighting math.
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
            "description": "Explain your approach to writing this SPIR-V geometry shader"
        },
        "spirv_code": {
            "type": "string",
            "description": "Complete SPIR-V assembly text for the geometry shader"
        }
    },
    "required": ["reasoning", "spirv_code"],
    "additionalProperties": False
}


def prepareSubpassPrompt(subPass: int) -> str:
    if subPass >= len(SUBPASSES):
        raise StopIteration
    return SUBPASSES[subPass]["prompt"]


extraGradeAnswerRuns = []

# Cached resources
_vert_spv = None
_frag_spv = None
_renderer = None
_OUTPUT_IMAGE_CACHE: Dict[Tuple[int, str], str] = {}

def _get_fixed_shaders():
    global _vert_spv, _frag_spv
    if _vert_spv is None:
        _vert_spv = assemble_spirv(VERT_SHADER_ASM)
        _frag_spv = assemble_spirv(FRAG_SHADER_ASM)
    return _vert_spv, _frag_spv


def _get_renderer():
    global _renderer
    if _renderer is None:
        from geometry_renderer import VulkanGeometryRenderer
        _renderer = VulkanGeometryRenderer(512, 512)
    return _renderer


def _get_geometry_data(topology):
    """Get vertex/index data for the given topology."""
    from geometry_renderer import (
        generate_point_grid, generate_line_segments,
        generate_sphere_triangles, generate_line_adjacency_curve,
        generate_triangle_adjacency_mesh,
    )
    if topology == "points":
        vb, vc = generate_point_grid(5, 5)
        return vb, vc, None, 0
    elif topology == "lines":
        vb, vc = generate_line_segments(8)
        return vb, vc, None, 0
    elif topology == "triangles":
        return generate_sphere_triangles()
    elif topology == "line_adjacency":
        vb, vc = generate_line_adjacency_curve(16)
        return vb, vc, None, 0
    elif topology == "triangle_adjacency":
        return generate_triangle_adjacency_mesh()
    raise ValueError(f"Unknown topology: {topology}")


def gradeAnswer(result: dict, subPass: int, aiEngineName: str) -> tuple:
    if not result:
        return 0.0, "No result provided"
    if "spirv_code" not in result:
        return 0.0, "No SPIR-V code provided"

    desc = SUBPASSES[subPass]["description"]
    topology = SUBPASSES[subPass]["topology"]
    geom_text = result["spirv_code"]

    # Assemble geometry shader
    try:
        geom_spv = assemble_spirv(geom_text)
    except RuntimeError as e:
        return 0.0, f"[{desc}] SPIR-V assembly failed: {e}"

    # Validate
    valid, err = validate_spirv(geom_spv)
    if not valid:
        return 0.0, f"[{desc}] SPIR-V validation failed: {err[:300]}"

    # Get fixed shaders
    try:
        vert_spv, frag_spv = _get_fixed_shaders()
    except RuntimeError as e:
        return 0.0, f"[{desc}] Fixed shader error: {e}"

    # Get geometry data
    vb_data, vert_count, ib_data, idx_count = _get_geometry_data(topology)

    # Render
    try:
        renderer = _get_renderer()
        pixels = renderer.render(
            vert_spv, geom_spv, frag_spv,
            vb_data, vert_count, topology,
            ib_data, idx_count,
        )
    except Exception as e:
        return 0.0, f"[{desc}] Rendering failed: {e}"

    _OUTPUT_IMAGE_CACHE[(subPass, aiEngineName)] = _save_rendered_image(
        42, subPass, aiEngineName, pixels
    )

    # Compare to reference
    import numpy as np
    reference = load_reference(42, subPass)
    if reference is None:
        save_reference(pixels, 42, subPass)
        return 1.0, f"[{desc}] No reference - saved current render as reference"

    score, comparison = compare_images(pixels, reference, color_tolerance=2, spatial_tolerance=1)
    return score, f"[{desc}] {comparison}"


def resultToNiceReport(result: dict, subPass: int, aiEngineName: str) -> str:
    if not result:
        return "<p style='color:red'>No result provided</p>"
    desc = SUBPASSES[subPass]["description"]
    topo = SUBPASSES[subPass]["topology"]
    html = f"<h4>Geometry Shader - {desc} ({topo})</h4>"
    if "reasoning" in result:
        r = result['reasoning'][:400] + ('...' if len(result.get('reasoning', '')) > 400 else '')
        html += f"<p><strong>Approach:</strong> {r}</p>"
    if "spirv_code" in result:
        code = result["spirv_code"].replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        html += f"<details><summary>View SPIR-V ({len(result['spirv_code'])} chars)</summary><pre>{code}</pre></details>"
    html += image_pair_html(
        _OUTPUT_IMAGE_CACHE.get((subPass, aiEngineName), ""),
        str(get_reference_path(42, subPass))
    )
    return html


def resultToImage(result: dict, subPass: int, aiEngineName: str) -> str:
    return _OUTPUT_IMAGE_CACHE.get((subPass, aiEngineName), "")


def getReferenceImage(subPass: int, aiEngineName: str) -> str:
    return str(get_reference_path(42, subPass))


def _save_rendered_image(test_num: int, subPass: int, aiEngineName: str, pixels) -> str:
    base_dir = os.path.dirname(__file__)
    out_dir = os.path.join(base_dir, "results", "models", aiEngineName, "renders")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"test{test_num}_subpass_{subPass:02d}.png")
    Image.fromarray(pixels, "RGBA").save(out_path)
    return out_path


highLevelSummary = """
SPIR-V Geometry Shaders tests the ability to write GPU geometry shaders in raw SPIR-V assembly.

**Key concepts:**
- Geometry shader input/output primitive types (points, lines, triangles, adjacency)
- gl_PerVertex blocks with BuiltIn Position decorations
- Array inputs for per-vertex data
- OpEmitVertex / OpEndPrimitive for generating new geometry
- Expanding points to quads, lines to thick ribbons, triangle manipulation
- Using adjacency information for smooth curves and edge detection
"""
