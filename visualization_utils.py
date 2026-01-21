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
    return {
      'vertices': mesh['vertices'],
      'faces': mesh['faces']
    }
  
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
                <div style=\"margin-top: 8px; font-size: 12px; color: #666; background: #f8f8f8; padding: 5px; border-radius: 3px;\">A=start, B=end, black=wall, white=open, purple=path</div>
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
                ctx.fillStyle = '#7c3aed';
                for (let i = 0; i < pathData.length; i++) {{
                    const p = pathData[i];
                    const x = p[0];
                    const y = p[1];
                    ctx.fillRect(x * cellSize, y * cellSize, cellSize, cellSize);
                }}
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
                const positions = new Float32Array(pathData.length * 3);
                for (let i = 0; i < pathData.length; i++) {{
                    const p = pathData[i];
                    const px = p[0] - mazeWidth / 2 + 0.5;
                    const py = mazeHeight / 2 - p[1] - 0.5;
                    positions[i * 3 + 0] = px;
                    positions[i * 3 + 1] = py;
                    positions[i * 3 + 2] = 0.2;
                }}
                const pathGeom = new THREE.BufferGeometry();
                pathGeom.setAttribute('position', new THREE.BufferAttribute(positions, 3));
                const pathMat = new THREE.LineBasicMaterial({{ color: 0x7c3aed }});
                const pathLine = new THREE.Line(pathGeom, pathMat);
                scene.add(pathLine);
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
