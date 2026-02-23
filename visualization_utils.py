"""
3D Visualization utilities for test reports using three.js.
Provides reusable components for rendering 3D objects in HTML reports.
"""

import json
import uuid
from typing import List, Tuple, Dict, Any


def generate_threejs_csg_visualization(mesh_a: Dict,
                                       mesh_b: Dict,
                                       result_mesh: Dict,
                                       name: str = "CSG Union") -> str:
  """
    Generate HTML/JavaScript for 3D CSG union visualization using three.js.
    
    Args:
        mesh_a: First input mesh with 'vertices' and 'faces'
        mesh_b: Second input mesh with 'vertices' and 'faces'
        result_mesh: Result mesh with 'vertices' and 'faces'
        name: Name for the visualization
        
    Returns:
        HTML string with three.js visualization wrapped in collapsible details
    """
  # Generate unique ID for this visualization
  viz_id = str(uuid.uuid4())[:8]

  # Function to convert mesh to Three.js geometry format
  def mesh_to_three_js(mesh, color, is_wireframe=False, mesh_suffix=""):
    vertices = mesh['vertices']
    faces = mesh['faces']

    # Convert vertices to flat array
    verts = []
    for v in vertices:
      verts.extend([v[0], v[1], v[2]])

    if is_wireframe:
      # For wireframe, create line segments from triangle edges
      edges = set()
      for face in faces:
        if len(face) >= 3:
          # Add edges of the triangle/polygon
          for i in range(len(face)):
            v1 = face[i]
            v2 = face[(i + 1) % len(face)]
            edge = tuple(sorted([v1, v2]))
            edges.add(edge)

      # Convert edges to flat array for LineSegments
      line_indices = []
      for v1, v2 in edges:
        line_indices.extend([v1, v2])

      geom = f"""
              const geometry{mesh_suffix} = new THREE.BufferGeometry();
              geometry{mesh_suffix}.setAttribute('position', new THREE.Float32BufferAttribute({verts}, 3));
              geometry{mesh_suffix}.setIndex({line_indices});
              
              const material{mesh_suffix} = new THREE.LineBasicMaterial({{'color': 0x{color}, 'transparent': true, 'opacity': 0.8}});
              
              const wireframe{mesh_suffix} = new THREE.LineSegments(geometry{mesh_suffix}, material{mesh_suffix});
              wireframe{mesh_suffix}.name = '{viz_id}{mesh_suffix.replace(f"_{viz_id}_", "")}';
              scene{viz_id}.add(wireframe{mesh_suffix});
          """
    else:
      # For solid mesh, convert faces to triangles
      indices = []
      for face in faces:
        if len(face) == 3:  # Triangle
          indices.extend(face)
        elif len(face) > 3:  # Triangulate polygon (naive fan triangulation)
          for i in range(1, len(face) - 1):
            indices.extend([face[0], face[i], face[i + 1]])

      geom = f"""
              const geometry{mesh_suffix} = new THREE.BufferGeometry();
              geometry{mesh_suffix}.setAttribute('position', new THREE.Float32BufferAttribute({verts}, 3));
              geometry{mesh_suffix}.setIndex({indices});
              geometry{mesh_suffix}.computeVertexNormals();
              
              const material{mesh_suffix} = new THREE.MeshPhongMaterial({{'color': 0x{color}, 'side': THREE.DoubleSide, 'flatShading': true, 'transparent': true, 'opacity': 0.8}});
              
              const mesh{mesh_suffix} = new THREE.Mesh(geometry{mesh_suffix}, material{mesh_suffix});
              mesh{mesh_suffix}.name = '{viz_id}{mesh_suffix.replace(f"_{viz_id}_", "")}';
              scene{viz_id}.add(mesh{mesh_suffix});
          """
    return geom

  # Convert mesh data to JSON for lazy initialization
  def mesh_to_json(mesh):
    return {'vertices': mesh['vertices'], 'faces': mesh['faces']}

  mesh_a_json = json.dumps(mesh_to_json(mesh_a))
  mesh_b_json = json.dumps(mesh_to_json(mesh_b))
  result_mesh_json = json.dumps(mesh_to_json(result_mesh))

  html = f"""
    <div class="csg-visualization" style="margin: 15px 0;">
        <details>
            <summary style="cursor: pointer; padding: 8px; background: #e8e8e8; border-radius: 4px; font-weight: bold; color: #333; border: 1px solid #ccc;">
                🎯 CSG Union Visualization: {name}
            </summary>
            <div style="margin-top: 10px;">
                <div id="csg-container-{viz_id}" style="width: 100%; height: 500px; position: relative;">
                    <div id="csg-renderer-{viz_id}" style="width: 100%; height: 100%; border: 1px solid #ccc; background: #fafafa; border-radius: 4px; display: flex; align-items: center; justify-content: center; color: #999;">
                        <span class="viz-placeholder">Scroll here to activate 3D view</span>
                    </div>
                    <div style="position: absolute; top: 10px; left: 10px; background: rgba(255,255,255,0.9); padding: 5px; border-radius: 3px; border: 1px solid #ddd;">
                        <button onclick="toggleWireframeA{viz_id}()" style="padding: 4px 8px; margin-right: 5px; background: #ffe6e6; border: 1px solid #ccc; border-radius: 3px; cursor: pointer;">Toggle A</button>
                        <button onclick="toggleWireframeB{viz_id}()" style="padding: 4px 8px; margin-right: 5px; background: #e6e6ff; border: 1px solid #ccc; border-radius: 3px; cursor: pointer;">Toggle B</button>
                        <select onchange="changeResultMode{viz_id}(this.value)" style="padding: 4px 8px; margin-right: 10px; background: #e6ffe6; border: 1px solid #ccc; border-radius: 3px; cursor: pointer;">
                            <option value="solid">Result: Solid</option>
                            <option value="wireframe">Result: Wireframe</option>
                            <option value="hidden">Result: Hidden</option>
                        </select>
                        <button onclick="resetCamera{viz_id}()" style="padding: 4px 8px; margin-right: 10px; background: #f0f0f0; border: 1px solid #ccc; border-radius: 3px; cursor: pointer;">Reset View</button>
                        <span style="margin-left: 20px; color: #666;">Drag to rotate, Scroll to zoom, Right-drag to pan</span>
                    </div>
                </div>
                <div style="margin-top: 8px; font-size: 12px; color: #666; background: #f8f8f8; padding: 5px; border-radius: 3px;">
                    🔴 Input Mesh A (wireframe) | 🔵 Input Mesh B (wireframe) | 🟢 Result Mesh (solid)
                </div>
            </div>
        </details>
    </div>
    
    <script src="https://cdn.jsdelivr.net/npm/three@0.132.2/build/three.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/three@0.132.2/examples/js/controls/OrbitControls.js"></script>
    <script>
    (function() {{
        const vizId = '{viz_id}';
        const meshAData = {mesh_a_json};
        const meshBData = {mesh_b_json};
        const resultMeshData = {result_mesh_json};
        
        let scene, camera, renderer, controls, animationId;
        let isActive = false;
        
        function meshToThreeJs(mesh, color, isWireframe, meshSuffix) {{
            const vertices = mesh.vertices;
            const faces = mesh.faces;
            
            const verts = [];
            for (const v of vertices) {{
                verts.push(v[0], v[1], v[2]);
            }}
            
            if (isWireframe) {{
                const edges = new Set();
                for (const face of faces) {{
                    if (face.length >= 3) {{
                        for (let i = 0; i < face.length; i++) {{
                            const v1 = face[i];
                            const v2 = face[(i + 1) % face.length];
                            const edge = v1 < v2 ? v1 + '_' + v2 : v2 + '_' + v1;
                            edges.add(edge);
                        }}
                    }}
                }}
                
                const lineIndices = [];
                for (const edge of edges) {{
                    const [v1, v2] = edge.split('_').map(Number);
                    lineIndices.push(v1, v2);
                }}
                
                const geometry = new THREE.BufferGeometry();
                geometry.setAttribute('position', new THREE.Float32BufferAttribute(verts, 3));
                geometry.setIndex(lineIndices);
                
                const material = new THREE.LineBasicMaterial({{color: parseInt(color, 16), transparent: true, opacity: 0.8}});
                const wireframe = new THREE.LineSegments(geometry, material);
                wireframe.name = vizId + meshSuffix;
                return wireframe;
            }} else {{
                const indices = [];
                for (const face of faces) {{
                    if (face.length === 3) {{
                        indices.push(...face);
                    }} else if (face.length > 3) {{
                        for (let i = 1; i < face.length - 1; i++) {{
                            indices.push(face[0], face[i], face[i + 1]);
                        }}
                    }}
                }}
                
                const geometry = new THREE.BufferGeometry();
                geometry.setAttribute('position', new THREE.Float32BufferAttribute(verts, 3));
                geometry.setIndex(indices);
                geometry.computeVertexNormals();
                
                const material = new THREE.MeshPhongMaterial({{color: parseInt(color, 16), side: THREE.DoubleSide, flatShading: true, transparent: true, opacity: 0.8}});
                const mesh = new THREE.Mesh(geometry, material);
                mesh.name = vizId + meshSuffix;
                return mesh;
            }}
        }}
        
        function activate() {{
            if (isActive) return;
            isActive = true;
            
            const container = document.getElementById('csg-renderer-' + vizId);
            if (!container || typeof THREE === 'undefined') return;
            
            // Clear placeholder
            const placeholder = container.querySelector('.viz-placeholder');
            if (placeholder) placeholder.style.display = 'none';
            
            // Scene setup
            scene = new THREE.Scene();
            scene.background = new THREE.Color(0xf0f0f0);
            
            // Camera
            camera = new THREE.PerspectiveCamera(75, container.clientWidth / container.clientHeight, 0.1, 1000);
            camera.position.set(5, 5, 5);
            camera.lookAt(0, 0, 0);
            
            // Renderer
            renderer = new THREE.WebGLRenderer({{ antialias: true }});
            renderer.setSize(container.clientWidth, container.clientHeight);
            container.appendChild(renderer.domElement);
            
            // Controls
            controls = new THREE.OrbitControls(camera, renderer.domElement);
            controls.enableDamping = true;
            controls.dampingFactor = 0.05;
            
            // Lighting
            const ambientLight = new THREE.AmbientLight(0x404040);
            scene.add(ambientLight);
            
            const directionalLight1 = new THREE.DirectionalLight(0xffffff, 0.8);
            directionalLight1.position.set(1, 1, 1);
            scene.add(directionalLight1);
            
            const directionalLight2 = new THREE.DirectionalLight(0xffffff, 0.5);
            directionalLight2.position.set(-1, -1, -1);
            scene.add(directionalLight2);
            
            // Add coordinate axes
            const axesHelper = new THREE.AxesHelper(5);
            scene.add(axesHelper);
            
            // Add meshes
            scene.add(meshToThreeJs(meshAData, 'ff0000', true, 'a'));
            scene.add(meshToThreeJs(meshBData, '0000ff', true, 'b'));
            scene.add(meshToThreeJs(resultMeshData, '00aa00', false, 'result'));
            const resultWire = meshToThreeJs(resultMeshData, '00aa00', true, 'result_wire');
            resultWire.visible = false;
            scene.add(resultWire);
            
            // Animation loop
            function animate() {{
                if (!isActive) return;
                animationId = requestAnimationFrame(animate);
                controls.update();
                renderer.render(scene, camera);
            }}
            animate();
        }}
        
        function dispose() {{
            if (!isActive) return;
            isActive = false;
            
            if (animationId) {{
                cancelAnimationFrame(animationId);
                animationId = null;
            }}
            
            const container = document.getElementById('csg-renderer-' + vizId);
            
            if (renderer) {{
                renderer.dispose();
                if (renderer.domElement && renderer.domElement.parentNode) {{
                    renderer.domElement.parentNode.removeChild(renderer.domElement);
                }}
                renderer = null;
            }}
            
            if (scene) {{
                scene.traverse(function(object) {{
                    if (object.geometry) object.geometry.dispose();
                    if (object.material) {{
                        if (Array.isArray(object.material)) {{
                            object.material.forEach(m => m.dispose());
                        }} else {{
                            object.material.dispose();
                        }}
                    }}
                }});
                scene = null;
            }}
            
            camera = null;
            controls = null;
            
            // Show placeholder again
            if (container) {{
                const placeholder = container.querySelector('.viz-placeholder');
                if (placeholder) placeholder.style.display = '';
            }}
        }}
        
        // Global functions for buttons
        window.resetCamera{viz_id} = function() {{
            if (camera && controls) {{
                camera.position.set(5, 5, 5);
                camera.lookAt(0, 0, 0);
                controls.update();
            }}
        }};
        
        window.toggleWireframeA{viz_id} = function() {{
            if (scene) {{
                const obj = scene.getObjectByName(vizId + 'a');
                if (obj) obj.visible = !obj.visible;
            }}
        }};
        
        window.toggleWireframeB{viz_id} = function() {{
            if (scene) {{
                const obj = scene.getObjectByName(vizId + 'b');
                if (obj) obj.visible = !obj.visible;
            }}
        }};
        
        window.changeResultMode{viz_id} = function(mode) {{
            if (!scene) return;
            const solid = scene.getObjectByName(vizId + 'result');
            const wire = scene.getObjectByName(vizId + 'result_wire');
            
            if (mode === 'solid') {{
                if (solid) solid.visible = true;
                if (wire) wire.visible = false;
            }} else if (mode === 'wireframe') {{
                if (solid) solid.visible = false;
                if (wire) wire.visible = true;
            }} else if (mode === 'hidden') {{
                if (solid) solid.visible = false;
                if (wire) wire.visible = false;
            }}
        }};
        
        // Register with VizManager
        if (window.VizManager) {{
            window.VizManager.register({{
                id: vizId,
                containerId: 'csg-renderer-' + vizId,
                activate: activate,
                dispose: dispose
            }});
        }} else {{
            // Fallback: activate immediately if no VizManager
            activate();
        }}
    }})();
    </script>
    """
  return html


def generate_threejs_maze_visualization(path,
                                        width: int,
                                        height: int,
                                        start,
                                        end,
                                        maze_string: str,
                                        name: str = "Maze") -> str:
  viz_id = str(uuid.uuid4())[:8]

  if width * height > 40000:
    return f"""
    <div class=\"maze-visualization\" style=\"margin: 15px 0;\">
        <details>
            <summary style=\"cursor: pointer; padding: 8px; background: #e8e8e8; border-radius: 4px; font-weight: bold; color: #333; border: 1px solid #ccc;\">🧩 {name}: {width}x{height}</summary>
            <div style=\"margin-top: 10px; color: #666;\">Visualization skipped for large maze.</div>
        </details>
    </div>
    """

  maze_lines = maze_string.strip().split("\n")
  maze_data = json.dumps(maze_lines)
  path_data = json.dumps([list(p) for p in path]) if path else "null"
  start_data = json.dumps(list(start) if start else None)
  end_data = json.dumps(list(end) if end else None)

  path_len = len(path) if path else 0
  html = f"""
    <div class=\"maze-visualization\" style=\"margin: 15px 0;\">
        <details>
            <summary style=\"cursor: pointer; padding: 8px; background: #e8e8e8; border-radius: 4px; font-weight: bold; color: #333; border: 1px solid #ccc;\">🧩 {name}: {width}x{height} (path {path_len})</summary>
            <div style=\"margin-top: 10px;\">
                <div id=\"maze-renderer-{viz_id}\" style=\"width: 100%; height: 420px; border: 1px solid #ccc; background: #fafafa; border-radius: 4px; position: relative; display: flex; align-items: center; justify-content: center; color: #999;\">
                    <span class="viz-placeholder">Scroll here to activate 3D view</span>
                </div>
                <div style=\"margin-top: 8px; font-size: 12px; color: #666; background: #f8f8f8; padding: 5px; border-radius: 3px;\">A=start, B=end, black=wall, white=open, purple=path, hot-pink=path-through-wall (invalid)</div>
            </div>
        </details>
    </div>

    <script src=\"https://cdn.jsdelivr.net/npm/three@0.132.2/build/three.min.js\"></script>
    <script src=\"https://cdn.jsdelivr.net/npm/three@0.132.2/examples/js/controls/OrbitControls.js\"></script>
    <script>
    (function() {{
        const vizId = 'maze_{viz_id}';
        const mazeWidth = {width};
        const mazeHeight = {height};
        const maze = {maze_data};
        const pathData = {path_data};
        const startData = {start_data};
        const endData = {end_data};
        
        let scene, camera, renderer, controls, animationId, texture;
        let isActive = false;
        
        function activate() {{
            if (isActive) return;
            isActive = true;
            
            if (typeof THREE === 'undefined') return;
            const container = document.getElementById('maze-renderer-{viz_id}');
            if (!container) return;
            
            // Clear placeholder
            const placeholder = container.querySelector('.viz-placeholder');
            if (placeholder) placeholder.style.display = 'none';

            const maxCanvas = 2048;
            const maxDim = Math.max(mazeWidth, mazeHeight);
            const cellSize = Math.max(1, Math.floor(maxCanvas / maxDim));

            const canvas = document.createElement('canvas');
            canvas.width = mazeWidth * cellSize;
            canvas.height = mazeHeight * cellSize;
            const ctx = canvas.getContext('2d');

            ctx.fillStyle = '#ffffff';
            ctx.fillRect(0, 0, canvas.width, canvas.height);

            for (let y = 0; y < mazeHeight; y++) {{
                const row = (maze[y] || '');
                for (let x = 0; x < mazeWidth; x++) {{
                    const ch = row[x] || '#';
                    if (ch === '#') {{
                        ctx.fillStyle = '#111111';
                    }} else {{
                        ctx.fillStyle = '#ffffff';
                    }}
                    ctx.fillRect(x * cellSize, y * cellSize, cellSize, cellSize);
                }}
            }}

            if (pathData && pathData.length) {{
                const violations = new Array(pathData.length);
                for (let i = 0; i < pathData.length; i++) {{
                    const p = pathData[i];
                    const x = p[0];
                    const y = p[1];
                    const row = (maze[y] || '');
                    const ch = row[x] || '#';
                    const isWall = (ch === '#');
                    violations[i] = isWall;

                    if (!isWall) {{
                        ctx.fillStyle = '#7c3aed';
                        ctx.fillRect(x * cellSize, y * cellSize, cellSize, cellSize);
                    }} else {{
                        // Path walks through a wall: render as an obvious error tile.
                        ctx.fillStyle = '#ff00cc';
                        ctx.fillRect(x * cellSize, y * cellSize, cellSize, cellSize);
                        ctx.strokeStyle = '#000000';
                        ctx.lineWidth = Math.max(1, Math.floor(cellSize / 10));
                        ctx.beginPath();
                        ctx.moveTo(x * cellSize + 1, y * cellSize + 1);
                        ctx.lineTo((x + 1) * cellSize - 1, (y + 1) * cellSize - 1);
                        ctx.moveTo((x + 1) * cellSize - 1, y * cellSize + 1);
                        ctx.lineTo(x * cellSize + 1, (y + 1) * cellSize - 1);
                        ctx.stroke();
                        ctx.strokeStyle = '#ffffff';
                        ctx.lineWidth = Math.max(1, Math.floor(cellSize / 16));
                        ctx.strokeRect(x * cellSize + 0.5, y * cellSize + 0.5, cellSize - 1, cellSize - 1);
                    }}
                }}

                // Export violations to 3D overlay drawing.
                window.__mazeVizViolations = window.__mazeVizViolations || {{}};
                window.__mazeVizViolations['{viz_id}'] = violations;
            }}

            if (startData) {{
                ctx.fillStyle = '#22c55e';
                ctx.fillRect(startData[0] * cellSize, startData[1] * cellSize, cellSize, cellSize);
            }}
            if (endData) {{
                ctx.fillStyle = '#ef4444';
                ctx.fillRect(endData[0] * cellSize, endData[1] * cellSize, cellSize, cellSize);
            }}

            texture = new THREE.CanvasTexture(canvas);
            texture.magFilter = THREE.NearestFilter;
            texture.minFilter = THREE.NearestFilter;
            texture.needsUpdate = true;

            scene = new THREE.Scene();
            scene.background = new THREE.Color(0xf0f0f0);

            renderer = new THREE.WebGLRenderer({{ antialias: true }});
            renderer.setSize(container.clientWidth, container.clientHeight);
            container.appendChild(renderer.domElement);

            const aspect = container.clientWidth / container.clientHeight;
            let viewW = mazeWidth;
            let viewH = mazeHeight;
            if (viewW / viewH < aspect) {{
                viewW = viewH * aspect;
            }} else {{
                viewH = viewW / aspect;
            }}

            camera = new THREE.OrthographicCamera(-viewW / 2, viewW / 2, viewH / 2, -viewH / 2, 0.1, 1000);
            camera.position.set(0, 0, 100);
            camera.lookAt(0, 0, 0);

            controls = new THREE.OrbitControls(camera, renderer.domElement);
            controls.enableRotate = false;
            controls.enableDamping = true;
            controls.dampingFactor = 0.05;
            controls.screenSpacePanning = true;

            const planeGeom = new THREE.PlaneGeometry(mazeWidth, mazeHeight);
            const planeMat = new THREE.MeshBasicMaterial({{ map: texture }});
            const plane = new THREE.Mesh(planeGeom, planeMat);
            scene.add(plane);

            if (pathData && pathData.length) {{
                const violations = (window.__mazeVizViolations && window.__mazeVizViolations['{viz_id}']) || [];

                // Split into valid/invalid segments so violations are visually unmistakable.
                const good = [];
                const bad = [];
                const badPts = [];

                function toWorld(p) {{
                    const px = p[0] - mazeWidth / 2 + 0.5;
                    const py = mazeHeight / 2 - p[1] - 0.5;
                    return [px, py, 0.25];
                }}

                for (let i = 0; i < pathData.length; i++) {{
                    if (violations[i]) {{
                        const w = toWorld(pathData[i]);
                        badPts.push(w[0], w[1], w[2]);
                    }}
                    if (i + 1 >= pathData.length) break;
                    const aBad = !!violations[i];
                    const bBad = !!violations[i + 1];
                    const segBad = aBad || bBad;
                    const a = toWorld(pathData[i]);
                    const b = toWorld(pathData[i + 1]);
                    if (segBad) {{
                        bad.push(a[0], a[1], a[2], b[0], b[1], b[2]);
                    }} else {{
                        good.push(a[0], a[1], a[2], b[0], b[1], b[2]);
                    }}
                }}

                if (good.length) {{
                    const geom = new THREE.BufferGeometry();
                    geom.setAttribute('position', new THREE.BufferAttribute(new Float32Array(good), 3));
                    const mat = new THREE.LineBasicMaterial({{ color: 0x7c3aed, linewidth: 1 }});
                    const line = new THREE.LineSegments(geom, mat);
                    scene.add(line);
                }}

                if (bad.length) {{
                    const geom = new THREE.BufferGeometry();
                    geom.setAttribute('position', new THREE.BufferAttribute(new Float32Array(bad), 3));
                    const mat = new THREE.LineBasicMaterial({{ color: 0xff00cc, linewidth: 2 }});
                    const line = new THREE.LineSegments(geom, mat);
                    scene.add(line);
                }}

                if (badPts.length) {{
                    const geom = new THREE.BufferGeometry();
                    geom.setAttribute('position', new THREE.BufferAttribute(new Float32Array(badPts), 3));
                    const mat = new THREE.PointsMaterial({{ color: 0xffea00, size: 0.35, sizeAttenuation: false }});
                    const pts = new THREE.Points(geom, mat);
                    scene.add(pts);
                }}
            }}

            function animate() {{
                if (!isActive) return;
                animationId = requestAnimationFrame(animate);
                controls.update();
                renderer.render(scene, camera);
            }}
            animate();
        }}
        
        function dispose() {{
            if (!isActive) return;
            isActive = false;
            
            if (animationId) {{
                cancelAnimationFrame(animationId);
                animationId = null;
            }}
            
            const container = document.getElementById('maze-renderer-{viz_id}');
            
            if (texture) {{
                texture.dispose();
                texture = null;
            }}
            
            if (renderer) {{
                renderer.dispose();
                if (renderer.domElement && renderer.domElement.parentNode) {{
                    renderer.domElement.parentNode.removeChild(renderer.domElement);
                }}
                renderer = null;
            }}
            
            if (scene) {{
                scene.traverse(function(object) {{
                    if (object.geometry) object.geometry.dispose();
                    if (object.material) {{
                        if (object.material.map) object.material.map.dispose();
                        if (Array.isArray(object.material)) {{
                            object.material.forEach(m => m.dispose());
                        }} else {{
                            object.material.dispose();
                        }}
                    }}
                }});
                scene = null;
            }}
            
            camera = null;
            controls = null;
            
            // Show placeholder again
            if (container) {{
                const placeholder = container.querySelector('.viz-placeholder');
                if (placeholder) placeholder.style.display = '';
            }}
        }}
        
        // Register with VizManager
        if (window.VizManager) {{
            window.VizManager.register({{
                id: vizId,
                containerId: 'maze-renderer-{viz_id}',
                activate: activate,
                dispose: dispose
            }});
        }} else {{
            // Fallback: activate immediately if no VizManager
            activate();
        }}
    }})();
    </script>
  """

  return html


def generate_threejs_tetrahedron_visualization(container_vertices: List[Tuple[float, float, float]],
                                               placements: List[Dict],
                                               edge_length: float,
                                               container_name: str = "Container") -> str:
  """
    Generate HTML/JavaScript for 3D visualization of tetrahedron packing using three.js.
    
    Args:
        container_vertices: List of (x, y, z) tuples defining the container
        placements: List of tetrahedron placements with 'center' and 'rotation'
        edge_length: Edge length of tetrahedrons
        container_name: Name for the visualization
        
    Returns:
        HTML string with three.js visualization
    """
  # Generate unique ID for this visualization
  viz_id = f"tetra_viz_{hash(container_name) % 10000}"

  # Convert data to JSON for JavaScript
  container_data = json.dumps(container_vertices)
  placements_data = json.dumps(placements)

  html = f"""
    <div class="tetrahedron-visualization" style="margin: 15px 0;">
        <details>
            <summary style="cursor: pointer; padding: 8px; background: #f0f0f0; border-radius: 4px; font-weight: bold;">
                📊 3D Visualization: {container_name} ({len(placements)} tetrahedrons)
            </summary>
            <div style="margin-top: 10px;">
                <div id="{viz_id}" style="width: 100%; height: 400px; border: 1px solid #ccc; background: #fafafa; border-radius: 4px; display: flex; align-items: center; justify-content: center; color: #999;">
                    <span class="viz-placeholder">Scroll here to activate 3D view</span>
                </div>
                <div style="margin-top: 8px; font-size: 12px; color: #666;">
                    🖱️ Left click + drag to rotate | Right click + drag to pan | Scroll to zoom
                </div>
            </div>
        </details>
    </div>
    
    <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
    <script>
    (function() {{
        const vizId = '{viz_id}';
        const containerVertices = {container_data};
        const placementsData = {placements_data};
        const edgeLengthData = {edge_length};
        
        let scene, camera, renderer, animationId, sceneCenter;
        let isActive = false;
        
        function activate() {{
            if (isActive) return;
            isActive = true;
            
            if (typeof THREE === 'undefined') return;
            const container = document.getElementById(vizId);
            if (!container) return;
            
            // Clear placeholder
            const placeholder = container.querySelector('.viz-placeholder');
            if (placeholder) placeholder.style.display = 'none';
            
            // Scene setup
            scene = new THREE.Scene();
            scene.background = new THREE.Color(0xf8f8f8);
            
            // Camera setup
            camera = new THREE.PerspectiveCamera(75, container.clientWidth / 400, 0.1, 1000);
            
            // Renderer setup
            renderer = new THREE.WebGLRenderer({{ antialias: true }});
            renderer.setSize(container.clientWidth, 400);
            container.appendChild(renderer.domElement);
            
            // Lighting
            const ambientLight = new THREE.AmbientLight(0xffffff, 0.6);
            scene.add(ambientLight);
            const directionalLight = new THREE.DirectionalLight(0xffffff, 0.4);
            directionalLight.position.set(10, 10, 5);
            scene.add(directionalLight);
            
            // Container wireframe - create proper edges for convex polyhedron
            const containerEdges = [];
            
            function shouldConnect(v1, v2, allVertices) {{
                const dist = Math.sqrt(
                    Math.pow(v1[0] - v2[0], 2) +
                    Math.pow(v1[1] - v2[1], 2) +
                    Math.pow(v1[2] - v2[2], 2)
                );
                
                const distances = [];
                for (let i = 0; i < allVertices.length; i++) {{
                    for (let j = i + 1; j < allVertices.length; j++) {{
                        const d = Math.sqrt(
                            Math.pow(allVertices[i][0] - allVertices[j][0], 2) +
                            Math.pow(allVertices[i][1] - allVertices[j][1], 2) +
                            Math.pow(allVertices[i][2] - allVertices[j][2], 2)
                        );
                        distances.push(d);
                    }}
                }}
                
                distances.sort((a, b) => a - b);
                const shortestDistances = distances.slice(0, Math.min(allVertices.length * 2, distances.length));
                const avgShortDist = shortestDistances.reduce((a, b) => a + b, 0) / shortestDistances.length;
                
                return dist < avgShortDist * 1.2;
            }}
            
            for (let i = 0; i < containerVertices.length; i++) {{
                for (let j = i + 1; j < containerVertices.length; j++) {{
                    if (shouldConnect(containerVertices[i], containerVertices[j], containerVertices)) {{
                        containerEdges.push([containerVertices[i], containerVertices[j]]);
                    }}
                }}
            }}
            
            const edgePositions = [];
            containerEdges.forEach(edge => {{
                edgePositions.push(...edge[0], ...edge[1]);
            }});
            
            const containerGeometry = new THREE.BufferGeometry();
            containerGeometry.setAttribute('position', new THREE.Float32BufferAttribute(edgePositions, 3));
            const containerMaterial = new THREE.LineBasicMaterial({{ color: 0x333333, linewidth: 2 }});
            const containerMesh = new THREE.LineSegments(containerGeometry, containerMaterial);
            scene.add(containerMesh);
            
            function createTetrahedron(center, rotation, edgeLength) {{
                const a = edgeLength / Math.sqrt(2);
                const vertices = [
                    [a, 0, -a / Math.sqrt(2)],
                    [-a, 0, -a / Math.sqrt(2)],
                    [0, a, a / Math.sqrt(2)],
                    [0, -a, a / Math.sqrt(2)]
                ];
                
                const geometry = new THREE.BufferGeometry();
                const positions = [];
                const indices = [];
                
                const faces = [[0, 1, 2], [0, 2, 3], [0, 3, 1], [1, 3, 2]];
                
                vertices.forEach(v => positions.push(...v));
                faces.forEach(face => indices.push(...face));
                
                geometry.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
                geometry.setIndex(indices);
                geometry.computeVertexNormals();
                
                const mesh = new THREE.Mesh(geometry, new THREE.MeshPhongMaterial({{ 
                    color: 0x4488ff, transparent: true, opacity: 0.8,
                    side: THREE.DoubleSide, shininess: 100, specular: 0x222222
                }}));
                
                const wireframeGeometry = new THREE.BufferGeometry();
                wireframeGeometry.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
                wireframeGeometry.setIndex(indices);
                const wireframeMaterial = new THREE.LineBasicMaterial({{ color: 0x000088, linewidth: 1 }});
                const wireframe = new THREE.LineSegments(wireframeGeometry, wireframeMaterial);
                
                mesh.position.set(...center);
                wireframe.position.set(...center);
                
                if (rotation && rotation.length >= 3) {{
                    mesh.rotation.set(rotation[0], rotation[1], rotation[2]);
                    wireframe.rotation.set(rotation[0], rotation[1], rotation[2]);
                }}
                
                const group = new THREE.Group();
                group.add(mesh);
                group.add(wireframe);
                return group;
            }}
            
            placementsData.forEach((placement, index) => {{
                const tetrahedron = createTetrahedron(placement.center, placement.rotation || [0, 0, 0], edgeLengthData);
                const hue = (index * 30) % 360;
                if (tetrahedron.children[0] && tetrahedron.children[0].material) {{
                    tetrahedron.children[0].material.color.setHSL(hue / 360, 0.6, 0.5);
                }}
                scene.add(tetrahedron);
            }});
            
            const box = new THREE.Box3().setFromObject(scene);
            sceneCenter = box.getCenter(new THREE.Vector3());
            const size = box.getSize(new THREE.Vector3());
            const maxDim = Math.max(size.x, size.y, size.z);
            const fov = camera.fov * (Math.PI / 180);
            let cameraZ = Math.abs(maxDim / 2 / Math.tan(fov / 2));
            cameraZ *= 1.5;
            camera.position.set(sceneCenter.x + cameraZ, sceneCenter.y + cameraZ, sceneCenter.z + cameraZ);
            camera.lookAt(sceneCenter);
            
            // Controls
            let mouseDown = false, mouseX = 0, mouseY = 0, isRightClick = false;
            
            container.addEventListener('mousedown', (e) => {{
                mouseDown = true; mouseX = e.clientX; mouseY = e.clientY; isRightClick = e.button === 2;
            }});
            container.addEventListener('contextmenu', (e) => e.preventDefault());
            container.addEventListener('mousemove', (e) => {{
                if (!mouseDown) return;
                const deltaX = e.clientX - mouseX, deltaY = e.clientY - mouseY;
                if (isRightClick) {{
                    camera.position.x -= deltaX * 0.01;
                    camera.position.y += deltaY * 0.01;
                }} else {{
                    const spherical = new THREE.Spherical();
                    spherical.setFromVector3(camera.position.clone().sub(sceneCenter));
                    spherical.theta -= deltaX * 0.005;
                    spherical.phi += deltaY * 0.005;
                    spherical.phi = Math.max(0.1, Math.min(Math.PI - 0.1, spherical.phi));
                    camera.position.copy(sceneCenter).add(new THREE.Vector3().setFromSpherical(spherical));
                    camera.lookAt(sceneCenter);
                }}
                mouseX = e.clientX; mouseY = e.clientY;
            }});
            container.addEventListener('mouseup', () => {{ mouseDown = false; }});
            container.addEventListener('wheel', (e) => {{
                e.preventDefault();
                const scale = e.deltaY > 0 ? 1.1 : 0.9;
                camera.position.multiplyScalar(scale);
            }});
            
            function animate() {{
                if (!isActive) return;
                animationId = requestAnimationFrame(animate);
                renderer.render(scene, camera);
            }}
            animate();
        }}
        
        function dispose() {{
            if (!isActive) return;
            isActive = false;
            
            if (animationId) {{
                cancelAnimationFrame(animationId);
                animationId = null;
            }}
            
            const container = document.getElementById(vizId);
            
            if (renderer) {{
                renderer.dispose();
                if (renderer.domElement && renderer.domElement.parentNode) {{
                    renderer.domElement.parentNode.removeChild(renderer.domElement);
                }}
                renderer = null;
            }}
            
            if (scene) {{
                scene.traverse(function(object) {{
                    if (object.geometry) object.geometry.dispose();
                    if (object.material) {{
                        if (Array.isArray(object.material)) {{
                            object.material.forEach(m => m.dispose());
                        }} else {{
                            object.material.dispose();
                        }}
                    }}
                }});
                scene = null;
            }}
            
            camera = null;
            sceneCenter = null;
            
            if (container) {{
                const placeholder = container.querySelector('.viz-placeholder');
                if (placeholder) placeholder.style.display = '';
            }}
        }}
        
        // Register with VizManager
        if (window.VizManager) {{
            window.VizManager.register({{
                id: vizId,
                containerId: vizId,
                activate: activate,
                dispose: dispose
            }});
        }} else {{
            activate();
        }}
    }})();
    </script>
    """

  return html


def generate_threejs_aabb_visualization(container: Tuple[int, int, int],
                                        boxes: List[Tuple[int, int, int]],
                                        placements: List[Dict],
                                        container_name: str = "Container") -> str:
  """
    Generate HTML/JavaScript for 3D AABB bin packing visualization using three.js.
    
    Args:
        container: (width, height, depth) of container
        boxes: List of (w, h, d) box dimensions
        placements: List of placed boxes with 'box_index' and 'position'
        container_name: Name for the visualization
        
    Returns:
        HTML string with three.js visualization
    """
  viz_id = f"aabb_viz_{hash(container_name) % 10000}"

  # Convert data to JSON for JavaScript
  container_data = json.dumps(container)
  boxes_data = json.dumps(boxes)
  placements_data = json.dumps(placements)

  html = f"""
    <div class="aabb-visualization" style="margin: 15px 0;">
        <details>
            <summary style="cursor: pointer; padding: 8px; background: #f0f0f0; border-radius: 4px; font-weight: bold;">
                📦 3D Bin Packing Visualization: {container_name} ({len(placements)} boxes)
            </summary>
            <div style="margin-top: 10px;">
                <div id="{viz_id}" style="width: 100%; height: 500px; border: 1px solid #ccc; background: #fafafa; border-radius: 4px; display: flex; align-items: center; justify-content: center; color: #999;">
                    <span class="viz-placeholder">Scroll here to activate 3D view</span>
                </div>
                <div style="margin-top: 8px; font-size: 12px; color: #666;">
                    🖱️ Left click + drag to rotate | Right click + drag to pan | Scroll to zoom
                </div>
            </div>
        </details>
    </div>
    
    <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
    <script>
    (function() {{
        const vizId = '{viz_id}';
        const containerData = {container_data};
        const boxesData = {boxes_data};
        const placementsData = {placements_data};
        
        let scene, camera, renderer, animationId, sceneCenter;
        let isActive = false;
        
        function activate() {{
            if (isActive) return;
            isActive = true;
            
            if (typeof THREE === 'undefined') return;
            const container = document.getElementById(vizId);
            if (!container) return;
            
            // Clear placeholder
            const placeholder = container.querySelector('.viz-placeholder');
            if (placeholder) placeholder.style.display = 'none';
            
            // Scene setup
            scene = new THREE.Scene();
            scene.background = new THREE.Color(0xf8f8f8);
            
            // Camera setup
            camera = new THREE.PerspectiveCamera(75, container.clientWidth / 500, 0.1, 10000);
            
            // Renderer setup
            renderer = new THREE.WebGLRenderer({{ antialias: true }});
            renderer.setSize(container.clientWidth, 500);
            container.appendChild(renderer.domElement);
            
            // Lighting
            const ambientLight = new THREE.AmbientLight(0xffffff, 0.6);
            scene.add(ambientLight);
            const directionalLight = new THREE.DirectionalLight(0xffffff, 0.4);
            directionalLight.position.set(10, 10, 5);
            scene.add(directionalLight);
            
            // Draw container wireframe
            const [cw, ch, cd] = containerData;
            const containerEdges = new THREE.EdgesGeometry(new THREE.BoxGeometry(cw, ch, cd));
            const containerMaterial = new THREE.LineBasicMaterial({{ color: 0x000000, linewidth: 2 }});
            const containerWireframe = new THREE.LineSegments(containerEdges, containerMaterial);
            containerWireframe.position.set(cw / 2, ch / 2, cd / 2);
            scene.add(containerWireframe);
            
            // Draw placed boxes
            placementsData.forEach((placement, index) => {{
                const boxIdx = placement.box_index;
                if (boxIdx < 0 || boxIdx >= boxesData.length) return;
                
                const [bw, bh, bd] = boxesData[boxIdx];
                const [px, py, pz] = placement.position;
                
                // Create box geometry
                const boxGeometry = new THREE.BoxGeometry(bw, bh, bd);
                
                // Vary color by index for visual distinction
                const hue = (index * 37) % 360;
                const boxMaterial = new THREE.MeshPhongMaterial({{ 
                    color: new THREE.Color().setHSL(hue / 360, 0.7, 0.6),
                    transparent: true,
                    opacity: 0.7,
                    side: THREE.DoubleSide
                }});
                
                const boxMesh = new THREE.Mesh(boxGeometry, boxMaterial);
                boxMesh.position.set(px + bw / 2, py + bh / 2, pz + bd / 2);
                scene.add(boxMesh);
                
                // Add wireframe for clarity
                const boxEdges = new THREE.EdgesGeometry(boxGeometry);
                const edgeMaterial = new THREE.LineBasicMaterial({{ color: 0x222222, linewidth: 1 }});
                const boxWireframe = new THREE.LineSegments(boxEdges, edgeMaterial);
                boxWireframe.position.set(px + bw / 2, py + bh / 2, pz + bd / 2);
                scene.add(boxWireframe);
            }});
            
            // Position camera
            const box = new THREE.Box3().setFromObject(scene);
            sceneCenter = box.getCenter(new THREE.Vector3());
            const size = box.getSize(new THREE.Vector3());
            const maxDim = Math.max(size.x, size.y, size.z);
            const fov = camera.fov * (Math.PI / 180);
            let cameraZ = Math.abs(maxDim / 2 / Math.tan(fov / 2));
            cameraZ *= 1.8;
            camera.position.set(sceneCenter.x + cameraZ * 0.7, sceneCenter.y + cameraZ * 0.7, sceneCenter.z + cameraZ * 0.7);
            camera.lookAt(sceneCenter);
            
            // Mouse controls
            let mouseDown = false, mouseX = 0, mouseY = 0, isRightClick = false;
            
            container.addEventListener('mousedown', (e) => {{
                mouseDown = true; mouseX = e.clientX; mouseY = e.clientY; isRightClick = e.button === 2;
            }});
            container.addEventListener('contextmenu', (e) => e.preventDefault());
            container.addEventListener('mousemove', (e) => {{
                if (!mouseDown) return;
                const deltaX = e.clientX - mouseX, deltaY = e.clientY - mouseY;
                if (isRightClick) {{
                    camera.position.x -= deltaX * (maxDim / 500);
                    camera.position.y += deltaY * (maxDim / 500);
                }} else {{
                    const spherical = new THREE.Spherical();
                    spherical.setFromVector3(camera.position.clone().sub(sceneCenter));
                    spherical.theta -= deltaX * 0.005;
                    spherical.phi += deltaY * 0.005;
                    spherical.phi = Math.max(0.1, Math.min(Math.PI - 0.1, spherical.phi));
                    camera.position.copy(sceneCenter).add(new THREE.Vector3().setFromSpherical(spherical));
                    camera.lookAt(sceneCenter);
                }}
                mouseX = e.clientX; mouseY = e.clientY;
            }});
            container.addEventListener('mouseup', () => {{ mouseDown = false; }});
            container.addEventListener('wheel', (e) => {{
                e.preventDefault();
                const scale = e.deltaY > 0 ? 1.1 : 0.9;
                const direction = camera.position.clone().sub(sceneCenter).normalize();
                const distance = camera.position.distanceTo(sceneCenter);
                camera.position.copy(sceneCenter).add(direction.multiplyScalar(distance * scale));
            }});
            
            function animate() {{
                if (!isActive) return;
                animationId = requestAnimationFrame(animate);
                renderer.render(scene, camera);
            }}
            animate();
        }}
        
        function dispose() {{
            if (!isActive) return;
            isActive = false;
            
            if (animationId) {{
                cancelAnimationFrame(animationId);
                animationId = null;
            }}
            
            const container = document.getElementById(vizId);
            
            if (renderer) {{
                renderer.dispose();
                if (renderer.domElement && renderer.domElement.parentNode) {{
                    renderer.domElement.parentNode.removeChild(renderer.domElement);
                }}
                renderer = null;
            }}
            
            if (scene) {{
                scene.traverse(function(object) {{
                    if (object.geometry) object.geometry.dispose();
                    if (object.material) {{
                        if (Array.isArray(object.material)) {{
                            object.material.forEach(m => m.dispose());
                        }} else {{
                            object.material.dispose();
                        }}
                    }}
                }});
                scene = null;
            }}
            
            camera = null;
            sceneCenter = null;
            
            if (container) {{
                const placeholder = container.querySelector('.viz-placeholder');
                if (placeholder) placeholder.style.display = '';
            }}
        }}
        
        // Register with VizManager
        if (window.VizManager) {{
            window.VizManager.register({{
                id: vizId,
                containerId: vizId,
                activate: activate,
                dispose: dispose
            }});
        }} else {{
            activate();
        }}
    }})();
    </script>
    """

  return html


def generate_threejs_graph_visualization(nodes: List[Tuple[float, float]],
                                         edges: List[Tuple[int, int]],
                                         graph_name: str = "Graph") -> str:
  """
    Generate HTML/JavaScript for 3D graph visualization using three.js.
    
    Args:
        nodes: List of (x, y) node positions
        edges: List of (node1, node2) edge connections
        graph_name: Name for the visualization
        
    Returns:
        HTML string with three.js visualization
    """
  viz_id = f"graph_viz_{hash(graph_name) % 10000}"

  nodes_data = json.dumps(nodes)
  edges_data = json.dumps(edges)

  html = f"""
    <div class="graph-visualization" style="margin: 15px 0;">
        <details>
            <summary style="cursor: pointer; padding: 8px; background: #f0f0f0; border-radius: 4px; font-weight: bold;">
                📊 3D Graph Visualization: {graph_name} ({len(nodes)} nodes, {len(edges)} edges)
            </summary>
            <div style="margin-top: 10px;">
                <div id="{viz_id}" style="width: 100%; height: 400px; border: 1px solid #ccc; background: #fafafa; border-radius: 4px; display: flex; align-items: center; justify-content: center; color: #999;">
                    <span class="viz-placeholder">Scroll here to activate 3D view</span>
                </div>
                <div style="margin-top: 8px; font-size: 12px; color: #666;">
                    🖱️ Left click + drag to rotate | Right click + drag to pan | Scroll to zoom
                </div>
            </div>
        </details>
    </div>
    
    <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
    <script>
    (function() {{
        const vizId = '{viz_id}';
        const nodesData = {nodes_data};
        const edgesData = {edges_data};
        
        let scene, camera, renderer, animationId, sceneCenter;
        let isActive = false;
        
        function activate() {{
            if (isActive) return;
            isActive = true;
            
            if (typeof THREE === 'undefined') return;
            const container = document.getElementById(vizId);
            if (!container) return;
            
            // Clear placeholder
            const placeholder = container.querySelector('.viz-placeholder');
            if (placeholder) placeholder.style.display = 'none';
            
            // Scene setup
            scene = new THREE.Scene();
            scene.background = new THREE.Color(0xf8f8f8);
            
            camera = new THREE.PerspectiveCamera(75, container.clientWidth / 400, 0.1, 1000);
            renderer = new THREE.WebGLRenderer({{ antialias: true }});
            renderer.setSize(container.clientWidth, 400);
            container.appendChild(renderer.domElement);
            
            // Lighting
            const ambientLight = new THREE.AmbientLight(0xffffff, 0.6);
            scene.add(ambientLight);
            
            // Create nodes
            const nodeGeometry = new THREE.SphereGeometry(0.1, 16, 16);
            nodesData.forEach((node, index) => {{
                const nodeMaterial = new THREE.MeshPhongMaterial({{ color: 0x4488ff }});
                const nodeMesh = new THREE.Mesh(nodeGeometry, nodeMaterial);
                nodeMesh.position.set(node[0], node[1], 0);
                scene.add(nodeMesh);
            }});
            
            // Create edges
            const edgeGeometry = new THREE.BufferGeometry();
            const edgePositions = [];
            edgesData.forEach(edge => {{
                const node1 = nodesData[edge[0]];
                const node2 = nodesData[edge[1]];
                edgePositions.push(node1[0], node1[1], 0, node2[0], node2[1], 0);
            }});
            
            edgeGeometry.setAttribute('position', new THREE.Float32BufferAttribute(edgePositions, 3));
            const edgeMaterial = new THREE.LineBasicMaterial({{ color: 0x333333, linewidth: 2 }});
            const edgeMesh = new THREE.LineSegments(edgeGeometry, edgeMaterial);
            scene.add(edgeMesh);
            
            // Camera positioning
            const box = new THREE.Box3().setFromObject(scene);
            sceneCenter = box.getCenter(new THREE.Vector3());
            const size = box.getSize(new THREE.Vector3());
            const maxDim = Math.max(size.x, size.y, size.z);
            const fov = camera.fov * (Math.PI / 180);
            let cameraZ = Math.abs(maxDim / 2 / Math.tan(fov / 2));
            cameraZ *= 1.5;
            camera.position.set(sceneCenter.x, sceneCenter.y, sceneCenter.z + cameraZ);
            camera.lookAt(sceneCenter);
            
            // Simple controls
            let mouseDown = false, mouseX = 0, mouseY = 0, isRightClick = false;
            
            container.addEventListener('mousedown', (e) => {{
                mouseDown = true; mouseX = e.clientX; mouseY = e.clientY; isRightClick = e.button === 2;
            }});
            container.addEventListener('contextmenu', (e) => e.preventDefault());
            container.addEventListener('mousemove', (e) => {{
                if (!mouseDown) return;
                const deltaX = e.clientX - mouseX, deltaY = e.clientY - mouseY;
                if (isRightClick) {{
                    camera.position.x -= deltaX * 0.01; camera.position.y += deltaY * 0.01;
                }} else {{
                    const spherical = new THREE.Spherical();
                    spherical.setFromVector3(camera.position.clone().sub(sceneCenter));
                    spherical.theta -= deltaX * 0.005; spherical.phi += deltaY * 0.005;
                    spherical.phi = Math.max(0.1, Math.min(Math.PI - 0.1, spherical.phi));
                    camera.position.copy(sceneCenter).add(new THREE.Vector3().setFromSpherical(spherical));
                    camera.lookAt(sceneCenter);
                }}
                mouseX = e.clientX; mouseY = e.clientY;
            }});
            container.addEventListener('mouseup', () => {{ mouseDown = false; }});
            container.addEventListener('wheel', (e) => {{
                e.preventDefault();
                const scale = e.deltaY > 0 ? 1.1 : 0.9;
                camera.position.multiplyScalar(scale);
            }});
            
            function animate() {{
                if (!isActive) return;
                animationId = requestAnimationFrame(animate);
                renderer.render(scene, camera);
            }}
            animate();
        }}
        
        function dispose() {{
            if (!isActive) return;
            isActive = false;
            
            if (animationId) {{
                cancelAnimationFrame(animationId);
                animationId = null;
            }}
            
            const container = document.getElementById(vizId);
            
            if (renderer) {{
                renderer.dispose();
                if (renderer.domElement && renderer.domElement.parentNode) {{
                    renderer.domElement.parentNode.removeChild(renderer.domElement);
                }}
                renderer = null;
            }}
            
            if (scene) {{
                scene.traverse(function(object) {{
                    if (object.geometry) object.geometry.dispose();
                    if (object.material) {{
                        if (Array.isArray(object.material)) {{
                            object.material.forEach(m => m.dispose());
                        }} else {{
                            object.material.dispose();
                        }}
                    }}
                }});
                scene = null;
            }}
            
            camera = null;
            sceneCenter = null;
            
            if (container) {{
                const placeholder = container.querySelector('.viz-placeholder');
                if (placeholder) placeholder.style.display = '';
            }}
        }}
        
        // Register with VizManager
        if (window.VizManager) {{
            window.VizManager.register({{
                id: vizId,
                containerId: vizId,
                activate: activate,
                dispose: dispose
            }});
        }} else {{
            activate();
        }}
    }})();
    </script>
    """

  return html


def generate_threejs_flight_path(path_points,
                                 scenario_name: str = "Flight Path",
                                 runway=None) -> str:
  """
  Generate a three.js 3D flight path visualization.

  Args:
      path_points: list of [x, y, z] coordinates (metres).
      scenario_name: label for the collapsible summary.
      runway: optional dict with keys 'x', 'y', 'length', 'width'
              to draw a runway rectangle on the ground plane.

  Returns:
      HTML string with embedded three.js visualization.
  """
  if not path_points or len(path_points) < 2:
    return ""

  viz_id = str(uuid.uuid4())[:8]
  pts_json = json.dumps(path_points)
  rwy_json = json.dumps(runway) if runway else "null"
  n_pts = len(path_points)

  html = f"""
    <div class="flight-path-visualization" style="margin: 15px 0;">
        <details>
            <summary style="cursor: pointer; padding: 8px; background: #e8e8e8;
                            border-radius: 4px; font-weight: bold; color: #333;
                            border: 1px solid #ccc;">
                &#9992; 3D Flight Path: {scenario_name} ({n_pts} samples)
            </summary>
            <div style="margin-top: 10px;">
                <div id="fp-container-{viz_id}"
                     style="width:100%; height:500px; position:relative;">
                    <div id="fp-renderer-{viz_id}"
                         style="width:100%; height:100%; border:1px solid #ccc;
                                background:#fafafa; border-radius:4px;
                                display:flex; align-items:center;
                                justify-content:center; color:#999;">
                        <span class="viz-placeholder">Scroll here to activate 3D view</span>
                    </div>
                    <div style="position:absolute; top:10px; left:10px;
                                background:rgba(255,255,255,0.9); padding:5px;
                                border-radius:3px; border:1px solid #ddd;
                                font-size:12px;">
                        <button onclick="fpReset_{viz_id}()"
                                style="padding:3px 8px; background:#f0f0f0;
                                       border:1px solid #ccc; border-radius:3px;
                                       cursor:pointer;">Reset View</button>
                        <span style="margin-left:12px; color:#666;">
                            Drag=rotate | Scroll=zoom | Right-drag=pan
                        </span>
                    </div>
                </div>
                <div style="margin-top:6px; font-size:12px; color:#666;
                            background:#f8f8f8; padding:5px; border-radius:3px;">
                    &#x1F7E2; Start &nbsp; &#x1F534; End &nbsp;
                    Line colour: green (start) &rarr; red (end)
                </div>
            </div>
        </details>
    </div>

    <script src="https://cdn.jsdelivr.net/npm/three@0.132.2/build/three.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/three@0.132.2/examples/js/controls/OrbitControls.js"></script>
    <script>
    (function() {{
        const vizId = 'fp_{viz_id}';
        const rawPts = {pts_json};
        const runwayData = {rwy_json};

        let scene, camera, renderer, controls, animationId;
        let isActive = false;

        function activate() {{
            if (isActive) return;
            isActive = true;
            if (typeof THREE === 'undefined') return;
            const container = document.getElementById('fp-renderer-{viz_id}');
            if (!container) return;
            const ph = container.querySelector('.viz-placeholder');
            if (ph) ph.style.display = 'none';

            let minX=Infinity,minY=Infinity,minZ=Infinity;
            let maxX=-Infinity,maxY=-Infinity,maxZ=-Infinity;
            for (const p of rawPts) {{
                if (p[0]<minX) minX=p[0]; if (p[0]>maxX) maxX=p[0];
                if (p[1]<minY) minY=p[1]; if (p[1]>maxY) maxY=p[1];
                if (p[2]<minZ) minZ=p[2]; if (p[2]>maxZ) maxZ=p[2];
            }}
            const cx=(minX+maxX)/2, cy=(minY+maxY)/2, cz=(minZ+maxZ)/2;
            const span = Math.max(maxX-minX, maxY-minY, maxZ-minZ, 1);
            const sc = 16.0 / span;

            function tr(p) {{ return [(p[0]-cx)*sc, (p[2]-cz)*sc, -(p[1]-cy)*sc]; }}

            scene = new THREE.Scene();
            scene.background = new THREE.Color(0xf0f0f0);

            camera = new THREE.PerspectiveCamera(
                60, container.clientWidth / container.clientHeight, 0.01, 500);
            camera.position.set(12, 10, 12);
            camera.lookAt(0, 0, 0);

            renderer = new THREE.WebGLRenderer({{ antialias: true }});
            renderer.setSize(container.clientWidth, container.clientHeight);
            container.appendChild(renderer.domElement);

            controls = new THREE.OrbitControls(camera, renderer.domElement);
            controls.enableDamping = true;
            controls.dampingFactor = 0.05;

            scene.add(new THREE.AmbientLight(0x808080));
            const dl = new THREE.DirectionalLight(0xffffff, 0.7);
            dl.position.set(5, 10, 5);
            scene.add(dl);

            const groundY = (minZ - cz) * sc;
            const gSize = 24;
            const gGeom = new THREE.PlaneGeometry(gSize, gSize);
            const gMat = new THREE.MeshLambertMaterial({{
                color: 0xbbddbb, transparent: true, opacity: 0.35,
                side: THREE.DoubleSide}});
            const ground = new THREE.Mesh(gGeom, gMat);
            ground.rotation.x = -Math.PI / 2;
            ground.position.y = groundY;
            scene.add(ground);

            const grid = new THREE.GridHelper(gSize, 20, 0x999999, 0xcccccc);
            grid.position.y = groundY;
            scene.add(grid);

            if (runwayData) {{
                const rc = tr([runwayData.x, runwayData.y, minZ]);
                const rl = runwayData.length * sc;
                const rw = runwayData.width * sc;
                const rGeom = new THREE.PlaneGeometry(rl, rw);
                const rMat = new THREE.MeshLambertMaterial({{
                    color: 0x555555, transparent: true, opacity: 0.7,
                    side: THREE.DoubleSide}});
                const rMesh = new THREE.Mesh(rGeom, rMat);
                rMesh.rotation.x = -Math.PI / 2;
                rMesh.position.set(rc[0], groundY + 0.01, rc[2]);
                scene.add(rMesh);

                const tGeom = new THREE.PlaneGeometry(0.1, rw);
                const tMat = new THREE.MeshBasicMaterial({{
                    color: 0xffffff, side: THREE.DoubleSide}});
                const tMesh = new THREE.Mesh(tGeom, tMat);
                tMesh.rotation.x = -Math.PI / 2;
                const thPt = tr([0, runwayData.y, minZ]);
                tMesh.position.set(thPt[0], groundY + 0.02, thPt[2]);
                scene.add(tMesh);
            }}

            const positions = [];
            const colors = [];
            const n = rawPts.length;
            for (let i = 0; i < n; i++) {{
                const t = tr(rawPts[i]);
                positions.push(t[0], t[1], t[2]);
                const frac = i / Math.max(n - 1, 1);
                colors.push(1 - frac, 0.2, frac);
            }}
            const segPos = [];
            const segCol = [];
            for (let i = 0; i < n - 1; i++) {{
                segPos.push(
                    positions[i*3], positions[i*3+1], positions[i*3+2],
                    positions[(i+1)*3], positions[(i+1)*3+1], positions[(i+1)*3+2]);
                segCol.push(
                    colors[i*3], colors[i*3+1], colors[i*3+2],
                    colors[(i+1)*3], colors[(i+1)*3+1], colors[(i+1)*3+2]);
            }}
            const lineGeom = new THREE.BufferGeometry();
            lineGeom.setAttribute('position',
                new THREE.Float32BufferAttribute(segPos, 3));
            lineGeom.setAttribute('color',
                new THREE.Float32BufferAttribute(segCol, 3));
            const lineMat = new THREE.LineBasicMaterial({{
                vertexColors: true, linewidth: 2 }});
            scene.add(new THREE.LineSegments(lineGeom, lineMat));

            const dropPos = [];
            const dropCol = [];
            for (let i = 0; i < n; i += Math.max(1, Math.floor(n / 30))) {{
                const t = tr(rawPts[i]);
                dropPos.push(t[0], t[1], t[2], t[0], groundY, t[2]);
                dropCol.push(0.6, 0.6, 0.6, 0.6, 0.6, 0.6);
            }}
            if (dropPos.length) {{
                const dGeom = new THREE.BufferGeometry();
                dGeom.setAttribute('position',
                    new THREE.Float32BufferAttribute(dropPos, 3));
                dGeom.setAttribute('color',
                    new THREE.Float32BufferAttribute(dropCol, 3));
                const dMat = new THREE.LineBasicMaterial({{
                    vertexColors: true, transparent: true, opacity: 0.3 }});
                scene.add(new THREE.LineSegments(dGeom, dMat));
            }}

            const sGeom = new THREE.SphereGeometry(0.25, 12, 12);
            const startPt = tr(rawPts[0]);
            const endPt = tr(rawPts[n - 1]);
            const ss = new THREE.Mesh(sGeom.clone(),
                new THREE.MeshLambertMaterial({{ color: 0x22cc22 }}));
            ss.position.set(startPt[0], startPt[1], startPt[2]);
            scene.add(ss);
            const es = new THREE.Mesh(sGeom.clone(),
                new THREE.MeshLambertMaterial({{ color: 0xcc2222 }}));
            es.position.set(endPt[0], endPt[1], endPt[2]);
            scene.add(es);

            function animate() {{
                if (!isActive) return;
                animationId = requestAnimationFrame(animate);
                controls.update();
                renderer.render(scene, camera);
            }}
            animate();
        }}

        function dispose() {{
            if (!isActive) return;
            isActive = false;
            if (animationId) {{ cancelAnimationFrame(animationId); animationId = null; }}
            const container = document.getElementById('fp-renderer-{viz_id}');
            if (renderer) {{
                renderer.dispose();
                if (renderer.domElement && renderer.domElement.parentNode)
                    renderer.domElement.parentNode.removeChild(renderer.domElement);
                renderer = null;
            }}
            if (scene) {{
                scene.traverse(function(o) {{
                    if (o.geometry) o.geometry.dispose();
                    if (o.material) {{
                        if (Array.isArray(o.material)) o.material.forEach(m=>m.dispose());
                        else o.material.dispose();
                    }}
                }});
                scene = null;
            }}
            camera = null; controls = null;
            if (container) {{
                const ph = container.querySelector('.viz-placeholder');
                if (ph) ph.style.display = '';
            }}
        }}

        window.fpReset_{viz_id} = function() {{
            if (camera && controls) {{
                camera.position.set(12, 10, 12);
                camera.lookAt(0, 0, 0);
                controls.update();
            }}
        }};

        if (window.VizManager) {{
            window.VizManager.register({{
                id: vizId,
                containerId: 'fp-renderer-{viz_id}',
                activate: activate,
                dispose: dispose
            }});
        }} else {{
            activate();
        }}
    }})();
    </script>
    """
  return html


def generate_threejs_car_path(path_points,
                              scenario_name: str = "Car Path",
                              road_width: float = 11.1,
                              lane_width: float = 3.7,
                              num_lanes: int = 3,
                              obstacles=None) -> str:
  """
  Generate a three.js 3D car path visualization.

  Args:
      path_points: list of [x, y] coordinates in metres (x=forward, y=lateral).
      scenario_name: label for the collapsible summary.
      road_width: total road width in metres.
      lane_width: width of each lane in metres.
      num_lanes: number of lanes.
      obstacles: optional list of dicts with 'x', 'y', 'width', 'length', 'label'.

  Returns:
      HTML string with embedded three.js visualization.
  """
  if not path_points or len(path_points) < 2:
    return ""

  viz_id = str(uuid.uuid4())[:8]
  pts_json = json.dumps(path_points)
  obs_json = json.dumps(obstacles or [])
  n_pts = len(path_points)

  html = f"""
    <div class="car-path-visualization" style="margin: 15px 0;">
        <details>
            <summary style="cursor: pointer; padding: 8px; background: #e8e8e8;
                            border-radius: 4px; font-weight: bold; color: #333;
                            border: 1px solid #ccc;">
                &#x1F697; Car Path: {scenario_name} ({n_pts} samples)
            </summary>
            <div style="margin-top: 10px;">
                <div id="cp-container-{viz_id}"
                     style="width:100%; height:500px; position:relative;">
                    <div id="cp-renderer-{viz_id}"
                         style="width:100%; height:100%; border:1px solid #ccc;
                                background:#fafafa; border-radius:4px;
                                display:flex; align-items:center;
                                justify-content:center; color:#999;">
                        <span class="viz-placeholder">Scroll here to activate 3D view</span>
                    </div>
                    <div style="position:absolute; top:10px; left:10px;
                                background:rgba(255,255,255,0.9); padding:5px;
                                border-radius:3px; border:1px solid #ddd;
                                font-size:12px;">
                        <button onclick="cpReset_{viz_id}()"
                                style="padding:3px 8px; background:#f0f0f0;
                                       border:1px solid #ccc; border-radius:3px;
                                       cursor:pointer;">Reset View</button>
                        <span style="margin-left:12px; color:#666;">
                            Drag=rotate | Scroll=zoom | Right-drag=pan
                        </span>
                    </div>
                </div>
                <div style="margin-top:6px; font-size:12px; color:#666;
                            background:#f8f8f8; padding:5px; border-radius:3px;">
                    &#x1F7E2; Start &nbsp; &#x1F534; End &nbsp;
                    &#x1F7E7; Obstacles &nbsp;
                    Line colour: green (start) &rarr; red (end)
                </div>
            </div>
        </details>
    </div>

    <script src="https://cdn.jsdelivr.net/npm/three@0.132.2/build/three.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/three@0.132.2/examples/js/controls/OrbitControls.js"></script>
    <script>
    (function() {{
        const vizId = 'cp_{viz_id}';
        const rawPts = {pts_json};
        const obstaclesData = {obs_json};
        const roadWidth = {road_width};
        const laneWidth = {lane_width};
        const numLanes = {num_lanes};

        let scene, camera, renderer, controls, animationId;
        let isActive = false;

        function activate() {{
            if (isActive) return;
            isActive = true;
            if (typeof THREE === 'undefined') return;
            const container = document.getElementById('cp-renderer-{viz_id}');
            if (!container) return;
            const ph = container.querySelector('.viz-placeholder');
            if (ph) ph.style.display = 'none';

            /* bounding box of car path */
            let minX = Infinity, maxX = -Infinity;
            for (const p of rawPts) {{
                if (p[0] < minX) minX = p[0];
                if (p[0] > maxX) maxX = p[0];
            }}
            /* include obstacles in x range */
            for (const o of obstaclesData) {{
                if (o.x - o.length/2 < minX) minX = o.x - o.length/2;
                if (o.x + o.length/2 > maxX) maxX = o.x + o.length/2;
            }}
            const xPad = 20;
            minX -= xPad;
            maxX += xPad;
            const roadLen = maxX - minX;
            const cx = (minX + maxX) / 2;
            const cy = roadWidth / 2;

            scene = new THREE.Scene();
            scene.background = new THREE.Color(0x87ceeb);  /* sky blue */

            const aspect = container.clientWidth / container.clientHeight;
            camera = new THREE.PerspectiveCamera(50, aspect, 0.1, 5000);
            /* top-down angled view */
            camera.position.set(cx, 60, cy + 40);
            camera.lookAt(cx, 0, cy);

            renderer = new THREE.WebGLRenderer({{ antialias: true }});
            renderer.setSize(container.clientWidth, container.clientHeight);
            container.appendChild(renderer.domElement);

            controls = new THREE.OrbitControls(camera, renderer.domElement);
            controls.enableDamping = true;
            controls.dampingFactor = 0.05;
            controls.target.set(cx, 0, cy);

            scene.add(new THREE.AmbientLight(0xcccccc));
            const dl = new THREE.DirectionalLight(0xffffff, 0.6);
            dl.position.set(cx, 50, -20);
            scene.add(dl);

            /* ground (grass) */
            const grassGeo = new THREE.PlaneGeometry(roadLen + 100, roadWidth + 40);
            const grassMat = new THREE.MeshLambertMaterial({{ color: 0x4a7c3f }});
            const grass = new THREE.Mesh(grassGeo, grassMat);
            grass.rotation.x = -Math.PI / 2;
            grass.position.set(cx, -0.02, cy);
            scene.add(grass);

            /* road surface */
            const roadGeo = new THREE.PlaneGeometry(roadLen, roadWidth);
            const roadMat = new THREE.MeshLambertMaterial({{ color: 0x444444 }});
            const road = new THREE.Mesh(roadGeo, roadMat);
            road.rotation.x = -Math.PI / 2;
            road.position.set(cx, -0.01, cy);
            scene.add(road);

            /* lane markings */
            for (let i = 0; i <= numLanes; i++) {{
                const ly = i * laneWidth;
                const isSolid = (i === 0 || i === numLanes);
                if (isSolid) {{
                    /* solid edge lines */
                    const lGeo = new THREE.PlaneGeometry(roadLen, 0.15);
                    const lMat = new THREE.MeshBasicMaterial({{ color: 0xffffff }});
                    const line = new THREE.Mesh(lGeo, lMat);
                    line.rotation.x = -Math.PI / 2;
                    line.position.set(cx, 0.001, ly);
                    scene.add(line);
                }} else {{
                    /* dashed lane lines */
                    const dashLen = 3.0;
                    const gapLen = 6.0;
                    const step = dashLen + gapLen;
                    for (let dx = minX; dx < maxX; dx += step) {{
                        const dGeo = new THREE.PlaneGeometry(dashLen, 0.12);
                        const dMat = new THREE.MeshBasicMaterial({{ color: 0xffffff }});
                        const dash = new THREE.Mesh(dGeo, dMat);
                        dash.rotation.x = -Math.PI / 2;
                        dash.position.set(dx + dashLen / 2, 0.001, ly);
                        scene.add(dash);
                    }}
                }}
            }}

            /* obstacles */
            for (const o of obstaclesData) {{
                const oLen = o.length || 4.5;
                const oW = o.width || 2.0;
                const oH = 1.5;
                const oGeo = new THREE.BoxGeometry(oLen, oH, oW);
                const oMat = new THREE.MeshLambertMaterial({{
                    color: 0xff6600, transparent: true, opacity: 0.85 }});
                const oMesh = new THREE.Mesh(oGeo, oMat);
                oMesh.position.set(o.x, oH / 2, o.y);
                scene.add(oMesh);

                /* wireframe outline */
                const oEdge = new THREE.EdgesGeometry(oGeo);
                const oLine = new THREE.LineSegments(oEdge,
                    new THREE.LineBasicMaterial({{ color: 0x993300 }}));
                oLine.position.copy(oMesh.position);
                scene.add(oLine);

                /* label */
                if (o.label) {{
                    const canvas = document.createElement('canvas');
                    canvas.width = 128; canvas.height = 32;
                    const ctx = canvas.getContext('2d');
                    ctx.fillStyle = '#ff6600';
                    ctx.fillRect(0, 0, 128, 32);
                    ctx.fillStyle = '#fff';
                    ctx.font = 'bold 18px Arial';
                    ctx.textAlign = 'center';
                    ctx.fillText(o.label, 64, 22);
                    const tex = new THREE.CanvasTexture(canvas);
                    const spMat = new THREE.SpriteMaterial({{ map: tex }});
                    const sprite = new THREE.Sprite(spMat);
                    sprite.position.set(o.x, oH + 1.0, o.y);
                    sprite.scale.set(4, 1, 1);
                    scene.add(sprite);
                }}
            }}

            /* car path line (gradient green to red) */
            const n = rawPts.length;
            const segPos = [];
            const segCol = [];
            for (let i = 0; i < n - 1; i++) {{
                const frac0 = i / Math.max(n - 1, 1);
                const frac1 = (i + 1) / Math.max(n - 1, 1);
                segPos.push(rawPts[i][0], 0.15, rawPts[i][1]);
                segPos.push(rawPts[i+1][0], 0.15, rawPts[i+1][1]);
                segCol.push(0, 1 - frac0, frac0);
                segCol.push(0, 1 - frac1, frac1);
            }}
            if (segPos.length) {{
                const lineGeom = new THREE.BufferGeometry();
                lineGeom.setAttribute('position',
                    new THREE.Float32BufferAttribute(segPos, 3));
                lineGeom.setAttribute('color',
                    new THREE.Float32BufferAttribute(segCol, 3));
                const lineMat = new THREE.LineBasicMaterial({{
                    vertexColors: true, linewidth: 2 }});
                scene.add(new THREE.LineSegments(lineGeom, lineMat));
            }}

            /* car silhouette along path (every Nth point) */
            const carGeo = new THREE.BoxGeometry(4.5, 0.6, 2.0);
            const carMat = new THREE.MeshLambertMaterial({{
                color: 0x2266cc, transparent: true, opacity: 0.25 }});
            const interval = Math.max(1, Math.floor(n / 15));
            for (let i = 0; i < n; i += interval) {{
                const car = new THREE.Mesh(carGeo.clone(), carMat.clone());
                car.position.set(rawPts[i][0], 0.3, rawPts[i][1]);
                if (rawPts[i].length > 2) {{
                    car.rotation.y = -rawPts[i][2] * Math.PI / 180;
                }}
                scene.add(car);
            }}

            /* start / end markers */
            const sGeo = new THREE.SphereGeometry(0.6, 12, 12);
            const ss = new THREE.Mesh(sGeo.clone(),
                new THREE.MeshLambertMaterial({{ color: 0x22cc22 }}));
            ss.position.set(rawPts[0][0], 0.6, rawPts[0][1]);
            scene.add(ss);
            const es = new THREE.Mesh(sGeo.clone(),
                new THREE.MeshLambertMaterial({{ color: 0xcc2222 }}));
            es.position.set(rawPts[n-1][0], 0.6, rawPts[n-1][1]);
            scene.add(es);

            function animate() {{
                if (!isActive) return;
                animationId = requestAnimationFrame(animate);
                controls.update();
                renderer.render(scene, camera);
            }}
            animate();
        }}

        function dispose() {{
            if (!isActive) return;
            isActive = false;
            if (animationId) {{ cancelAnimationFrame(animationId); animationId = null; }}
            const container = document.getElementById('cp-renderer-{viz_id}');
            if (renderer) {{
                renderer.dispose();
                if (renderer.domElement && renderer.domElement.parentNode)
                    renderer.domElement.parentNode.removeChild(renderer.domElement);
                renderer = null;
            }}
            if (scene) {{
                scene.traverse(function(o) {{
                    if (o.geometry) o.geometry.dispose();
                    if (o.material) {{
                        if (o.material.map) o.material.map.dispose();
                        if (Array.isArray(o.material)) o.material.forEach(m=>m.dispose());
                        else o.material.dispose();
                    }}
                }});
                scene = null;
            }}
            camera = null; controls = null;
            if (container) {{
                const ph = container.querySelector('.viz-placeholder');
                if (ph) ph.style.display = '';
            }}
        }}

        window.cpReset_{viz_id} = function() {{
            if (camera && controls) {{
                const pts = rawPts;
                let mnX=Infinity, mxX=-Infinity;
                for (const p of pts) {{ if(p[0]<mnX)mnX=p[0]; if(p[0]>mxX)mxX=p[0]; }}
                const ccx = (mnX+mxX)/2;
                camera.position.set(ccx, 60, roadWidth/2 + 40);
                controls.target.set(ccx, 0, roadWidth/2);
                controls.update();
            }}
        }};

        if (window.VizManager) {{
            window.VizManager.register({{
                id: vizId,
                containerId: 'cp-renderer-{viz_id}',
                activate: activate,
                dispose: dispose
            }});
        }} else {{
            activate();
        }}
    }})();
    </script>
    """
  return html


def generate_threejs_docking_viz(path_points,
                                 scenario_name: str = "Docking",
                                 docked: bool = False,
                                 crashed: bool = False,
                                 crash_reason: str = "") -> str:
  """
  Generate a three.js 3D orbital docking visualization.

  Args:
      path_points: list of [x_lvlh, y_lvlh, z_lvlh] in metres
                   (x=radial/up, y=along-track, z=cross-track).
      scenario_name: label for the collapsible summary.
      docked: True if spacecraft successfully docked.
      crashed: True if spacecraft crashed into station.
      crash_reason: text description of crash.

  Returns:
      HTML string with embedded three.js visualization.
  """
  if not path_points or len(path_points) < 2:
    return ""

  viz_id = str(uuid.uuid4())[:8]
  pts_json = json.dumps(path_points)
  n_pts = len(path_points)
  outcome = "DOCKED" if docked else ("CRASH: " + crash_reason if crashed else "timeout")
  outcome_color = "#22cc22" if docked else ("#cc2222" if crashed else "#cc8800")

  html = f"""
    <div class="docking-visualization" style="margin: 15px 0;">
        <details>
            <summary style="cursor: pointer; padding: 8px; background: #e8e8e8;
                            border-radius: 4px; font-weight: bold; color: #333;
                            border: 1px solid #ccc;">
                &#x1F6F0; Docking Path: {scenario_name} ({n_pts} samples)
                &mdash; <span style="color:{outcome_color}">{outcome}</span>
            </summary>
            <div style="margin-top: 10px;">
                <div id="dk-container-{viz_id}"
                     style="width:100%; height:550px; position:relative;">
                    <div id="dk-renderer-{viz_id}"
                         style="width:100%; height:100%; border:1px solid #ccc;
                                background:#000011; border-radius:4px;
                                display:flex; align-items:center;
                                justify-content:center; color:#999;">
                        <span class="viz-placeholder">Scroll here to activate 3D view</span>
                    </div>
                    <div style="position:absolute; top:10px; left:10px;
                                background:rgba(0,0,0,0.7); padding:5px 10px;
                                border-radius:3px; font-size:12px; color:#ccc;">
                        <button onclick="dkReset_{viz_id}()"
                                style="padding:3px 8px; background:#333;
                                       color:#ccc; border:1px solid #555;
                                       border-radius:3px; cursor:pointer;">Reset View</button>
                        <span style="margin-left:12px;">
                            Drag=rotate | Scroll=zoom | Right-drag=pan
                        </span>
                    </div>
                </div>
                <div style="margin-top:6px; font-size:12px; color:#666;
                            background:#f8f8f8; padding:5px; border-radius:3px;">
                    &#x1F7E2; Start &nbsp;
                    &#x1F534; End/Impact &nbsp;
                    &#x1F7E1; Station &nbsp;
                    LVLH frame: Y=along-track, Up=radial (away from Earth)
                </div>
            </div>
        </details>
    </div>

    <script src="https://cdn.jsdelivr.net/npm/three@0.132.2/build/three.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/three@0.132.2/examples/js/controls/OrbitControls.js"></script>
    <script>
    (function() {{
        const vizId = 'dk_{viz_id}';
        const rawPts = {pts_json};
        const didDock = {'true' if docked else 'false'};
        const didCrash = {'true' if crashed else 'false'};

        let scene, camera, renderer, controls, animationId;
        let isActive = false;

        function buildStation(sc) {{
            /* Simplified ISS-like model at origin.
               Oriented so docking port faces -Y (toward approaching chaser
               in V-bar approach where chaser is at negative along-track).
               Truss runs along Z (cross-track), modules along Y (along-track). */
            const station = new THREE.Group();

            const moduleMat = new THREE.MeshPhongMaterial({{
                color: 0xcccccc, specular: 0x333333, shininess: 30 }});
            const goldMat = new THREE.MeshPhongMaterial({{
                color: 0xddaa33, specular: 0x886622, shininess: 20 }});
            const panelMat = new THREE.MeshPhongMaterial({{
                color: 0x1a237e, specular: 0x111155, shininess: 10,
                side: THREE.DoubleSide }});
            const dockMat = new THREE.MeshPhongMaterial({{
                color: 0x44ff44, emissive: 0x115511 }});

            /* Main pressurised modules (along Y = along-track) */
            const mainLen = 6 * sc;
            const modR = 0.5 * sc;
            const mainGeo = new THREE.CylinderGeometry(modR, modR, mainLen, 12);
            mainGeo.rotateX(Math.PI / 2);           /* align along Z initially */
            mainGeo.rotateY(Math.PI / 2);            /* now along X ... */
            /* Actually let's just orient manually */
            const mainMod = new THREE.Mesh(
                new THREE.CylinderGeometry(modR, modR, mainLen, 12),
                moduleMat);
            mainMod.rotation.x = Math.PI / 2;        /* axis along Z(three)=Z(lvlh) cross-track */
            mainMod.rotation.z = Math.PI / 2;        /* nope, let me just position it along Y */
            mainMod.rotation.set(0, 0, 0);
            /* Cylinder default axis = Y. We want it along three.js X (=LVLH y along-track). */
            mainMod.rotation.z = Math.PI / 2;
            station.add(mainMod);

            /* Second module perpendicular (along Z = cross-track) — the truss */
            const trussLen = 12 * sc;
            const trussR = 0.15 * sc;
            const truss = new THREE.Mesh(
                new THREE.CylinderGeometry(trussR, trussR, trussLen, 8),
                goldMat);
            /* Default Y axis → rotate to Z axis */
            truss.rotation.x = Math.PI / 2;
            station.add(truss);

            /* Node module at junction */
            const node = new THREE.Mesh(
                new THREE.SphereGeometry(modR * 0.7, 12, 12), moduleMat);
            station.add(node);

            /* Forward module (along +X in three.js = +Y LVLH along-track) */
            const fwdMod = new THREE.Mesh(
                new THREE.CylinderGeometry(modR * 0.8, modR * 0.8, 3 * sc, 10),
                moduleMat);
            fwdMod.rotation.z = Math.PI / 2;
            fwdMod.position.x = 4.5 * sc;
            station.add(fwdMod);

            /* Docking port — faces -X in three.js (= -Y LVLH = toward approaching V-bar chaser) */
            const dockGeo = new THREE.TorusGeometry(modR * 0.5, modR * 0.1, 8, 16);
            const dockPort = new THREE.Mesh(dockGeo, dockMat);
            dockPort.rotation.y = Math.PI / 2;
            dockPort.position.x = -mainLen / 2 - modR * 0.1;
            station.add(dockPort);

            /* Docking guide cone */
            const coneGeo = new THREE.ConeGeometry(modR * 0.3, 0.8 * sc, 8, 1, true);
            const coneMat = new THREE.MeshPhongMaterial({{
                color: 0x44ff44, transparent: true, opacity: 0.25,
                side: THREE.DoubleSide }});
            const cone = new THREE.Mesh(coneGeo, coneMat);
            cone.rotation.z = -Math.PI / 2;
            cone.position.x = -mainLen / 2 - 0.5 * sc;
            station.add(cone);

            /* Solar panels (4 arrays at ends of truss) */
            const panelW = 4 * sc;
            const panelH = 1.5 * sc;
            const panelGeo = new THREE.PlaneGeometry(panelW, panelH);
            const panelPositions = [
                [0, 0, trussLen / 2 - 1 * sc],
                [0, 0, trussLen / 2 - 3.5 * sc],
                [0, 0, -trussLen / 2 + 1 * sc],
                [0, 0, -trussLen / 2 + 3.5 * sc],
            ];
            for (const pp of panelPositions) {{
                const panel = new THREE.Mesh(panelGeo.clone(), panelMat);
                panel.position.set(pp[0], pp[1], pp[2]);
                /* Panels face up (Y in three.js = radial) */
                panel.rotation.x = Math.PI / 2;
                station.add(panel);
                /* Back side slightly different */
                const panelBack = new THREE.Mesh(panelGeo.clone(),
                    new THREE.MeshPhongMaterial({{ color: 0x333344, side: THREE.DoubleSide }}));
                panelBack.position.set(pp[0], pp[1] - 0.02 * sc, pp[2]);
                panelBack.rotation.x = Math.PI / 2;
                station.add(panelBack);
            }}

            return station;
        }}

        function activate() {{
            if (isActive) return;
            isActive = true;
            if (typeof THREE === 'undefined') return;
            const container = document.getElementById('dk-renderer-{viz_id}');
            if (!container) return;
            const ph = container.querySelector('.viz-placeholder');
            if (ph) ph.style.display = 'none';

            /* Coordinate mapping: LVLH [x_rad, y_along, z_cross]
               → three.js: X = y_along, Y = x_rad (up), Z = z_cross */
            function lvlhToThree(p) {{ return [p[1], p[0], p[2]]; }}

            /* Determine scale: fit path into reasonable view */
            let maxDist = 1;
            for (const p of rawPts) {{
                const d = Math.sqrt(p[0]*p[0] + p[1]*p[1] + p[2]*p[2]);
                if (d > maxDist) maxDist = d;
            }}
            /* Scale so max distance maps to ~15 units */
            const sc = 15.0 / maxDist;
            /* Station model scale */
            const stSc = Math.max(0.3, Math.min(2.0, maxDist / 200));

            scene = new THREE.Scene();
            scene.background = new THREE.Color(0x000811);

            const aspect = container.clientWidth / container.clientHeight;
            camera = new THREE.PerspectiveCamera(55, aspect, 0.01, 2000);
            camera.position.set(18, 12, 18);
            camera.lookAt(0, 0, 0);

            renderer = new THREE.WebGLRenderer({{ antialias: true }});
            renderer.setSize(container.clientWidth, container.clientHeight);
            container.appendChild(renderer.domElement);

            controls = new THREE.OrbitControls(camera, renderer.domElement);
            controls.enableDamping = true;
            controls.dampingFactor = 0.05;

            /* Lights */
            scene.add(new THREE.AmbientLight(0x404050));
            const sun = new THREE.DirectionalLight(0xffffff, 1.0);
            sun.position.set(20, 15, 10);
            scene.add(sun);
            const fill = new THREE.DirectionalLight(0x334466, 0.3);
            fill.position.set(-10, -5, -10);
            scene.add(fill);

            /* Stars background — scattered points */
            const starCount = 800;
            const starPos = new Float32Array(starCount * 3);
            for (let i = 0; i < starCount; i++) {{
                const theta = Math.random() * Math.PI * 2;
                const phi = Math.acos(2 * Math.random() - 1);
                const r = 500 + Math.random() * 500;
                starPos[i*3] = r * Math.sin(phi) * Math.cos(theta);
                starPos[i*3+1] = r * Math.sin(phi) * Math.sin(theta);
                starPos[i*3+2] = r * Math.cos(phi);
            }}
            const starGeo = new THREE.BufferGeometry();
            starGeo.setAttribute('position', new THREE.BufferAttribute(starPos, 3));
            const starMat = new THREE.PointsMaterial({{ color: 0xffffff, size: 0.8, sizeAttenuation: false }});
            scene.add(new THREE.Points(starGeo, starMat));

            /* Earth — large sphere below (LVLH -x = three.js -Y) */
            const earthR = 40;
            const earthGeo = new THREE.SphereGeometry(earthR, 48, 32);
            /* Create a simple Earth-like look with vertex colors */
            const earthMat = new THREE.MeshPhongMaterial({{
                color: 0x2244aa, emissive: 0x050510, specular: 0x224488,
                shininess: 15 }});
            const earth = new THREE.Mesh(earthGeo, earthMat);
            earth.position.y = -(earthR + 25);
            scene.add(earth);
            /* Atmosphere glow */
            const atmosGeo = new THREE.SphereGeometry(earthR * 1.02, 32, 24);
            const atmosMat = new THREE.MeshPhongMaterial({{
                color: 0x4488ff, transparent: true, opacity: 0.15,
                side: THREE.BackSide }});
            const atmos = new THREE.Mesh(atmosGeo, atmosMat);
            atmos.position.copy(earth.position);
            scene.add(atmos);

            /* Station model */
            const station = buildStation(stSc);
            scene.add(station);

            /* Spacecraft path (gradient green → red) */
            const n = rawPts.length;
            const segPos = [];
            const segCol = [];
            for (let i = 0; i < n - 1; i++) {{
                const a = lvlhToThree(rawPts[i]);
                const b = lvlhToThree(rawPts[i+1]);
                const f0 = i / Math.max(n - 1, 1);
                const f1 = (i+1) / Math.max(n - 1, 1);
                segPos.push(a[0]*sc, a[1]*sc, a[2]*sc,
                            b[0]*sc, b[1]*sc, b[2]*sc);
                segCol.push(0, 1-f0, f0,  0, 1-f1, f1);
            }}
            if (segPos.length) {{
                const lg = new THREE.BufferGeometry();
                lg.setAttribute('position', new THREE.Float32BufferAttribute(segPos, 3));
                lg.setAttribute('color', new THREE.Float32BufferAttribute(segCol, 3));
                scene.add(new THREE.LineSegments(lg,
                    new THREE.LineBasicMaterial({{ vertexColors: true, linewidth: 2 }})));
            }}

            /* Start sphere (green) */
            const startPt = lvlhToThree(rawPts[0]);
            const sph = new THREE.SphereGeometry(0.3, 10, 10);
            const startM = new THREE.Mesh(sph.clone(),
                new THREE.MeshLambertMaterial({{ color: 0x22ff22, emissive: 0x114411 }}));
            startM.position.set(startPt[0]*sc, startPt[1]*sc, startPt[2]*sc);
            scene.add(startM);

            /* End sphere */
            const endPt = lvlhToThree(rawPts[n-1]);
            let endColor = 0xffaa00;   /* timeout = orange */
            let endEmit = 0x554400;
            if (didDock) {{ endColor = 0x22ff22; endEmit = 0x114411; }}
            if (didCrash) {{ endColor = 0xff2222; endEmit = 0x441111; }}
            const endM = new THREE.Mesh(sph.clone(),
                new THREE.MeshLambertMaterial({{ color: endColor, emissive: endEmit }}));
            endM.position.set(endPt[0]*sc, endPt[1]*sc, endPt[2]*sc);
            scene.add(endM);

            /* If crashed, add impact flash */
            if (didCrash) {{
                const flashGeo = new THREE.SphereGeometry(0.6, 12, 12);
                const flashMat = new THREE.MeshBasicMaterial({{
                    color: 0xff4400, transparent: true, opacity: 0.5 }});
                const flash = new THREE.Mesh(flashGeo, flashMat);
                flash.position.copy(endM.position);
                scene.add(flash);
                /* Debris ring */
                const ringGeo = new THREE.TorusGeometry(0.8, 0.1, 8, 24);
                const ringMat = new THREE.MeshBasicMaterial({{
                    color: 0xff6600, transparent: true, opacity: 0.4 }});
                const ring = new THREE.Mesh(ringGeo, ringMat);
                ring.position.copy(endM.position);
                scene.add(ring);
            }}

            /* If docked, add docking indicator */
            if (didDock) {{
                const dkRing = new THREE.Mesh(
                    new THREE.TorusGeometry(0.5, 0.08, 8, 24),
                    new THREE.MeshBasicMaterial({{ color: 0x44ff44, transparent: true, opacity: 0.6 }}));
                dkRing.rotation.y = Math.PI / 2;
                scene.add(dkRing);
            }}

            /* LVLH axis helper (small) */
            const axLen = 3;
            const axOrigin = new THREE.Vector3(0, 0, 0);
            /* Y along-track = three.js +X (blue) */
            scene.add(new THREE.ArrowHelper(
                new THREE.Vector3(1, 0, 0), axOrigin, axLen, 0x4444ff, 0.3, 0.15));
            /* X radial = three.js +Y (red) */
            scene.add(new THREE.ArrowHelper(
                new THREE.Vector3(0, 1, 0), axOrigin, axLen, 0xff4444, 0.3, 0.15));
            /* Z cross-track = three.js +Z (green) */
            scene.add(new THREE.ArrowHelper(
                new THREE.Vector3(0, 0, 1), axOrigin, axLen, 0x44ff44, 0.3, 0.15));

            function animate() {{
                if (!isActive) return;
                animationId = requestAnimationFrame(animate);
                /* Slowly rotate Earth to give life */
                earth.rotation.y += 0.001;
                atmos.rotation.y += 0.001;
                controls.update();
                renderer.render(scene, camera);
            }}
            animate();
        }}

        function dispose() {{
            if (!isActive) return;
            isActive = false;
            if (animationId) {{ cancelAnimationFrame(animationId); animationId = null; }}
            const container = document.getElementById('dk-renderer-{viz_id}');
            if (renderer) {{
                renderer.dispose();
                if (renderer.domElement && renderer.domElement.parentNode)
                    renderer.domElement.parentNode.removeChild(renderer.domElement);
                renderer = null;
            }}
            if (scene) {{
                scene.traverse(function(o) {{
                    if (o.geometry) o.geometry.dispose();
                    if (o.material) {{
                        if (Array.isArray(o.material)) o.material.forEach(m=>m.dispose());
                        else o.material.dispose();
                    }}
                }});
                scene = null;
            }}
            camera = null; controls = null;
            if (container) {{
                const ph = container.querySelector('.viz-placeholder');
                if (ph) ph.style.display = '';
            }}
        }}

        window.dkReset_{viz_id} = function() {{
            if (camera && controls) {{
                camera.position.set(18, 12, 18);
                camera.lookAt(0, 0, 0);
                controls.target.set(0, 0, 0);
                controls.update();
            }}
        }};

        if (window.VizManager) {{
            window.VizManager.register({{
                id: vizId,
                containerId: 'dk-renderer-{viz_id}',
                activate: activate,
                dispose: dispose
            }});
        }} else {{
            activate();
        }}
    }})();
    </script>
    """
  return html


def generate_threejs_boid_visualization(initial_boids: List,
                                        gpu_frames: List,
                                        ref_frames: List,
                                        gpu_final: List,
                                        ref_final: List,
                                        n_total: int,
                                        n_shown: int,
                                        timesteps: int,
                                        dt: float,
                                        pos_tol: float,
                                        score: float,
                                        bound_size: float) -> str:
  """
  Generate HTML/JS for 3D boid flocking visualization with playback controls.
  Shows both GPU (orange) and CPU reference (cyan) boids for comparison.

  Args:
      initial_boids: [[x,y,z,vx,vy,vz], ...] initial state
      gpu_frames: list of frames from GPU, each [[x,y,z,vx,vy,vz], ...]
      ref_frames: list of frames from CPU reference (same length as gpu_frames)
      gpu_final: [[x,y,z,vx,vy,vz], ...] GPU final state
      ref_final: [[x,y,z,vx,vy,vz], ...] CPU final state
      n_total: total boid count (before downsampling)
      n_shown: number of boids shown in visualization
      timesteps: total simulation steps
      dt: timestep
      pos_tol: position tolerance used for grading
      score: grade score
      bound_size: simulation bounding box size
  """
  viz_id = str(uuid.uuid4())[:8]

  gpu_frames_json = json.dumps(gpu_frames)
  ref_frames_json = json.dumps(ref_frames)
  gpu_final_json = json.dumps(gpu_final)
  ref_final_json = json.dumps(ref_final)
  num_frames = max(len(gpu_frames), len(ref_frames))

  score_color = "#22c55e" if score >= 0.8 else "#eab308" if score >= 0.3 else "#ef4444"

  html = f"""
  <div class="boid-visualization" style="margin: 15px 0;">
    <details>
      <summary style="cursor:pointer;padding:8px;background:#e8e8e8;border-radius:4px;font-weight:bold;color:#333;border:1px solid #ccc;">
        &#x1f426; Boid Flocking: N={n_total} ({n_shown} shown), {timesteps} steps
        <span style="color:{score_color};margin-left:8px;">score={score:.2f}</span>
      </summary>
      <div style="margin-top:10px;">
        <div id="boid-renderer-{viz_id}" style="width:100%;height:500px;border:1px solid #ccc;background:#111;border-radius:4px;position:relative;display:flex;align-items:center;justify-content:center;color:#999;">
          <span class="viz-placeholder">Scroll here to activate 3D view</span>
        </div>
        <div id="boid-controls-{viz_id}" style="display:none;margin-top:8px;padding:8px 12px;background:#f8f8f8;border:1px solid #e5e7eb;border-radius:6px;">
          <div style="display:flex;align-items:center;gap:10px;flex-wrap:wrap;">
            <button id="boid-play-{viz_id}" style="padding:4px 12px;background:#3b82f6;color:#fff;border:none;border-radius:4px;cursor:pointer;font-size:13px;">&#x25b6; Play</button>
            <input id="boid-scrub-{viz_id}" type="range" min="0" max="100" value="0" style="flex:1;min-width:120px;">
            <span id="boid-frame-{viz_id}" style="font-size:12px;color:#475569;min-width:80px;">Frame 0/{num_frames}</span>
            <label style="font-size:12px;color:#475569;">Speed:
              <select id="boid-speed-{viz_id}" style="font-size:12px;padding:2px 4px;">
                <option value="0.5">0.5x</option>
                <option value="1" selected>1x</option>
                <option value="2">2x</option>
                <option value="4">4x</option>
              </select>
            </label>
            <label style="font-size:12px;color:#475569;">
              <input id="boid-showcpu-{viz_id}" type="checkbox" checked style="margin-right:3px;">CPU ref
            </label>
            <label style="font-size:12px;color:#475569;">
              <input id="boid-showgpu-{viz_id}" type="checkbox" checked style="margin-right:3px;">GPU
            </label>
            <button id="boid-reset-{viz_id}" style="padding:4px 8px;background:#e5e7eb;border:1px solid #ccc;border-radius:4px;cursor:pointer;font-size:12px;">Reset View</button>
          </div>
        </div>
        <div style="margin-top:6px;font-size:11px;color:#64748b;">
          Bound={bound_size} &middot; dt={dt} &middot; pos_tol={pos_tol:.2f} &middot; {n_shown}/{n_total} boids shown &middot;
          <span style="color:#22d3ee;">&#x25cf; Cyan=CPU ref</span> &middot;
          <span style="color:#f97316;">&#x25cf; Orange=GPU</span> &middot;
          Drag to rotate &middot; Scroll to zoom &middot; Right-drag to pan
        </div>
      </div>
    </details>
  </div>

  <script src="https://cdn.jsdelivr.net/npm/three@0.132.2/build/three.min.js"></script>
  <script src="https://cdn.jsdelivr.net/npm/three@0.132.2/examples/js/controls/OrbitControls.js"></script>
  <script>
  (function() {{
    var vizId = '{viz_id}';
    var gpuFrames = {gpu_frames_json};
    var refFrames = {ref_frames_json};
    var gpuFinal = {gpu_final_json};
    var refFinal = {ref_final_json};
    var boundSize = {bound_size};
    var nShown = {n_shown};
    var totalFrames = Math.max(gpuFrames.length, refFrames.length);

    var scene, camera, renderer, controls, animationId;
    var isActive = false;
    var cpuMesh, gpuMesh;
    var currentFrame = 0;
    var playing = false;
    var playSpeed = 1;
    var lastPlayTime = 0;
    var showCpu = true;
    var showGpu = true;

    function createBoidInstances(count, color) {{
      var coneGeom = new THREE.ConeGeometry(0.3, 1.0, 6);
      coneGeom.rotateX(Math.PI / 2);
      var mat = new THREE.MeshPhongMaterial({{
        color: color, flatShading: true, transparent: true, opacity: 0.8
      }});
      var mesh = new THREE.InstancedMesh(coneGeom, mat, count);
      mesh.instanceMatrix.setUsage(35048);
      return mesh;
    }}

    function setBoidPositions(mesh, boids) {{
      var dummy = new THREE.Object3D();
      for (var i = 0; i < boids.length; i++) {{
        var b = boids[i];
        dummy.position.set(b[0], b[1], b[2]);
        var vx = b[3], vy = b[4], vz = b[5];
        var speed = Math.sqrt(vx*vx + vy*vy + vz*vz);
        if (speed > 0.001) {{
          dummy.lookAt(b[0] + vx, b[1] + vy, b[2] + vz);
        }}
        dummy.scale.set(1, 1, 1);
        dummy.updateMatrix();
        mesh.setMatrixAt(i, dummy.matrix);
      }}
      mesh.instanceMatrix.needsUpdate = true;
      mesh.count = boids.length;
    }}

    function getFrame(frames, idx, finalState) {{
      if (frames.length === 0) return finalState;
      var i = Math.min(idx, frames.length - 1);
      return frames[i];
    }}

    function updateFrame(frameIdx) {{
      currentFrame = Math.max(0, Math.min(frameIdx, totalFrames - 1));
      if (cpuMesh) setBoidPositions(cpuMesh, getFrame(refFrames, currentFrame, refFinal));
      if (gpuMesh) setBoidPositions(gpuMesh, getFrame(gpuFrames, currentFrame, gpuFinal));
      var scrub = document.getElementById('boid-scrub-' + vizId);
      if (scrub) scrub.value = (currentFrame / Math.max(1, totalFrames - 1) * 100).toFixed(0);
      var label = document.getElementById('boid-frame-' + vizId);
      if (label) label.textContent = 'Frame ' + (currentFrame + 1) + '/' + totalFrames;
    }}

    function activate() {{
      if (isActive) return;
      isActive = true;
      if (typeof THREE === 'undefined') return;
      var container = document.getElementById('boid-renderer-' + vizId);
      if (!container) return;

      var ph = container.querySelector('.viz-placeholder');
      if (ph) ph.style.display = 'none';
      var ctrlDiv = document.getElementById('boid-controls-' + vizId);
      if (ctrlDiv) ctrlDiv.style.display = 'block';

      scene = new THREE.Scene();
      scene.background = new THREE.Color(0x111827);

      camera = new THREE.PerspectiveCamera(60, container.clientWidth / container.clientHeight, 0.1, 500);
      camera.position.set(boundSize * 0.8, boundSize * 0.6, boundSize * 0.8);
      camera.lookAt(0, 0, 0);

      renderer = new THREE.WebGLRenderer({{ antialias: true }});
      renderer.setSize(container.clientWidth, container.clientHeight);
      container.appendChild(renderer.domElement);

      controls = new THREE.OrbitControls(camera, renderer.domElement);
      controls.enableDamping = true;
      controls.dampingFactor = 0.08;
      controls.target.set(0, 0, 0);

      scene.add(new THREE.AmbientLight(0x404060, 0.6));
      var dl = new THREE.DirectionalLight(0xffffff, 0.8);
      dl.position.set(boundSize, boundSize, boundSize);
      scene.add(dl);
      var dl2 = new THREE.DirectionalLight(0x8888ff, 0.3);
      dl2.position.set(-boundSize, -boundSize, 0);
      scene.add(dl2);

      var half = boundSize / 2;
      var boxGeom = new THREE.BoxGeometry(boundSize, boundSize, boundSize);
      var boxEdges = new THREE.EdgesGeometry(boxGeom);
      scene.add(new THREE.LineSegments(boxEdges, new THREE.LineBasicMaterial({{ color: 0x334155 }})));

      var grid = new THREE.GridHelper(boundSize, 10, 0x1e293b, 0x1e293b);
      grid.position.y = -half;
      scene.add(grid);

      var axes = new THREE.AxesHelper(half * 0.3);
      axes.position.set(-half, -half, -half);
      scene.add(axes);

      cpuMesh = createBoidInstances(nShown, 0x22d3ee);
      scene.add(cpuMesh);

      gpuMesh = createBoidInstances(nShown, 0xf97316);
      scene.add(gpuMesh);

      updateFrame(0);

      var playBtn = document.getElementById('boid-play-' + vizId);
      var scrub = document.getElementById('boid-scrub-' + vizId);
      var speedSel = document.getElementById('boid-speed-' + vizId);
      var cpuCb = document.getElementById('boid-showcpu-' + vizId);
      var gpuCb = document.getElementById('boid-showgpu-' + vizId);
      var resetBtn = document.getElementById('boid-reset-' + vizId);

      if (playBtn) playBtn.addEventListener('click', function() {{
        playing = !playing;
        playBtn.innerHTML = playing ? '&#x23f8; Pause' : '&#x25b6; Play';
        if (playing) lastPlayTime = performance.now();
      }});
      if (scrub) scrub.addEventListener('input', function() {{
        var f = Math.round(parseFloat(scrub.value) / 100 * (totalFrames - 1));
        updateFrame(f);
      }});
      if (speedSel) speedSel.addEventListener('change', function() {{
        playSpeed = parseFloat(speedSel.value);
      }});
      if (cpuCb) cpuCb.addEventListener('change', function() {{
        showCpu = cpuCb.checked;
        cpuMesh.visible = showCpu;
      }});
      if (gpuCb) gpuCb.addEventListener('change', function() {{
        showGpu = gpuCb.checked;
        gpuMesh.visible = showGpu;
      }});
      if (resetBtn) resetBtn.addEventListener('click', function() {{
        camera.position.set(boundSize * 0.8, boundSize * 0.6, boundSize * 0.8);
        camera.lookAt(0, 0, 0);
        controls.target.set(0, 0, 0);
        controls.update();
      }});

      function animate(now) {{
        if (!isActive) return;
        animationId = requestAnimationFrame(animate);
        if (playing) {{
          var elapsed = (now - lastPlayTime) / 1000;
          var framesToAdvance = elapsed * 30 * playSpeed;
          if (framesToAdvance >= 1) {{
            var next = currentFrame + Math.floor(framesToAdvance);
            lastPlayTime = now;
            if (next >= totalFrames) {{
              updateFrame(totalFrames - 1);
              playing = false;
              if (playBtn) playBtn.innerHTML = '&#x25b6; Play';
            }} else {{
              updateFrame(next);
            }}
          }}
        }}
        controls.update();
        renderer.render(scene, camera);
      }}
      animationId = requestAnimationFrame(animate);
    }}

    function dispose() {{
      if (!isActive) return;
      isActive = false;
      playing = false;
      if (animationId) {{ cancelAnimationFrame(animationId); animationId = null; }}
      var container = document.getElementById('boid-renderer-' + vizId);
      if (renderer) {{
        renderer.dispose();
        if (renderer.domElement && renderer.domElement.parentNode)
          renderer.domElement.parentNode.removeChild(renderer.domElement);
        renderer = null;
      }}
      if (scene) {{
        scene.traverse(function(obj) {{
          if (obj.geometry) obj.geometry.dispose();
          if (obj.material) {{
            if (Array.isArray(obj.material)) obj.material.forEach(function(m) {{ m.dispose(); }});
            else obj.material.dispose();
          }}
        }});
        scene = null;
      }}
      camera = null; controls = null;
      cpuMesh = null; gpuMesh = null;
      var ctrlDiv = document.getElementById('boid-controls-' + vizId);
      if (ctrlDiv) ctrlDiv.style.display = 'none';
      if (container) {{
        var ph = container.querySelector('.viz-placeholder');
        if (ph) ph.style.display = '';
      }}
    }}

    if (window.VizManager) {{
      window.VizManager.register({{
        id: vizId,
        containerId: 'boid-renderer-' + vizId,
        activate: activate,
        dispose: dispose
      }});
    }} else {{
      activate();
    }}
  }})();
  </script>
  """
  return html

