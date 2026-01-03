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

  html = f"""
    <div class="csg-visualization" style="margin: 15px 0;">
        <details>
            <summary style="cursor: pointer; padding: 8px; background: #e8e8e8; border-radius: 4px; font-weight: bold; color: #333; border: 1px solid #ccc;">
                🎯 CSG Union Visualization: {name}
            </summary>
            <div style="margin-top: 10px;">
                <div id="csg-container-{viz_id}" style="width: 100%; height: 500px; position: relative;">
                    <div id="csg-renderer-{viz_id}" style="width: 100%; height: 100%; border: 1px solid #ccc; background: #fafafa; border-radius: 4px;"></div>
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
        // Scene setup
        const scene{viz_id} = new THREE.Scene();
        scene{viz_id}.background = new THREE.Color(0xf0f0f0);
        
        // Camera
        const container{viz_id} = document.getElementById('csg-renderer-{viz_id}');
        const camera{viz_id} = new THREE.PerspectiveCamera(75, container{viz_id}.clientWidth / container{viz_id}.clientHeight, 0.1, 1000);
        camera{viz_id}.position.set(5, 5, 5);
        camera{viz_id}.lookAt(0, 0, 0);
        
        // Renderer
        const renderer{viz_id} = new THREE.WebGLRenderer({{ antialias: true }});
        renderer{viz_id}.setSize(container{viz_id}.clientWidth, container{viz_id}.clientHeight);
        container{viz_id}.appendChild(renderer{viz_id}.domElement);
        
        // Controls
        const controls{viz_id} = new THREE.OrbitControls(camera{viz_id}, renderer{viz_id}.domElement);
        controls{viz_id}.enableDamping = true;
        controls{viz_id}.dampingFactor = 0.05;
        
        // Lighting
        const ambientLight{viz_id} = new THREE.AmbientLight(0x404040);
        scene{viz_id}.add(ambientLight{viz_id});
        
        const directionalLight1{viz_id} = new THREE.DirectionalLight(0xffffff, 0.8);
        directionalLight1{viz_id}.position.set(1, 1, 1);
        scene{viz_id}.add(directionalLight1{viz_id});
        
        const directionalLight2{viz_id} = new THREE.DirectionalLight(0xffffff, 0.5);
        directionalLight2{viz_id}.position.set(-1, -1, -1);
        scene{viz_id}.add(directionalLight2{viz_id});
        
        // Add coordinate axes
        const axesHelper{viz_id} = new THREE.AxesHelper(5);
        scene{viz_id}.add(axesHelper{viz_id});
        
        // Add meshes
        {mesh_to_three_js(mesh_a, "ff0000", True, f"_{viz_id}_a")}
        {mesh_to_three_js(mesh_b, "0000ff", True, f"_{viz_id}_b")}
        {mesh_to_three_js(result_mesh, "00aa00", False, f"_{viz_id}_result")}
        {mesh_to_three_js(result_mesh, "00aa00", True, f"_{viz_id}_result_wire")}
        
        // Hide result wireframe by default
        const resultWireframe{viz_id} = scene{viz_id}.getObjectByName('{viz_id}result_wire');
        if (resultWireframe{viz_id}) {{
            resultWireframe{viz_id}.visible = false;
        }}
        
        // Handle window resize
        window.addEventListener('resize', onWindowResize{viz_id}, false);
        
        function onWindowResize{viz_id}() {{
            const width = container{viz_id}.clientWidth;
            const height = container{viz_id}.clientHeight;
            camera{viz_id}.aspect = width / height;
            camera{viz_id}.updateProjectionMatrix();
            renderer{viz_id}.setSize(width, height);
        }}
        
        // Animation loop
        function animate{viz_id}() {{
            requestAnimationFrame(animate{viz_id});
            controls{viz_id}.update();
            renderer{viz_id}.render(scene{viz_id}, camera{viz_id});
        }}
        animate{viz_id}();
        
        // Reset camera
        window.resetCamera{viz_id} = function() {{
            camera{viz_id}.position.set(5, 5, 5);
            camera{viz_id}.lookAt(0, 0, 0);
            controls{viz_id}.update();
        }};
        
        // Toggle wireframe A
        window.toggleWireframeA{viz_id} = function() {{
            const wireframeA{viz_id} = scene{viz_id}.getObjectByName('{viz_id}a');
            if (wireframeA{viz_id}) {{
                wireframeA{viz_id}.visible = !wireframeA{viz_id}.visible;
            }}
        }};
        
        // Toggle wireframe B
        window.toggleWireframeB{viz_id} = function() {{
            const wireframeB{viz_id} = scene{viz_id}.getObjectByName('{viz_id}b');
            if (wireframeB{viz_id}) {{
                wireframeB{viz_id}.visible = !wireframeB{viz_id}.visible;
            }}
        }};
        
        // Change result mode
        window.changeResultMode{viz_id} = function(mode) {{
            const resultSolid{viz_id} = scene{viz_id}.getObjectByName('{viz_id}result');
            const resultWireframe{viz_id} = scene{viz_id}.getObjectByName('{viz_id}result_wire');
            
            if (mode === 'solid') {{
                if (resultSolid{viz_id}) resultSolid{viz_id}.visible = true;
                if (resultWireframe{viz_id}) resultWireframe{viz_id}.visible = false;
            }} else if (mode === 'wireframe') {{
                if (resultSolid{viz_id}) resultSolid{viz_id}.visible = false;
                if (resultWireframe{viz_id}) resultWireframe{viz_id}.visible = true;
            }} else if (mode === 'hidden') {{
                if (resultSolid{viz_id}) resultSolid{viz_id}.visible = false;
                if (resultWireframe{viz_id}) resultWireframe{viz_id}.visible = false;
            }}
        }};
        
        // Initial camera position
        resetCamera{viz_id}();
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
                <div id=\"maze-renderer-{viz_id}\" style=\"width: 100%; height: 420px; border: 1px solid #ccc; background: #fafafa; border-radius: 4px; position: relative;\"></div>
                <div style=\"margin-top: 8px; font-size: 12px; color: #666; background: #f8f8f8; padding: 5px; border-radius: 3px;\">A=start, B=end, black=wall, white=open, purple=path</div>
            </div>
        </details>
    </div>

    <script src=\"https://cdn.jsdelivr.net/npm/three@0.132.2/build/three.min.js\"></script>
    <script src=\"https://cdn.jsdelivr.net/npm/three@0.132.2/examples/js/controls/OrbitControls.js\"></script>
    <script>
    (function() {{
        if (typeof THREE === 'undefined') return;
        const container = document.getElementById('maze-renderer-{viz_id}');
        if (!container) return;

        const width = {width};
        const height = {height};
        const maze = {maze_data};
        const path = {path_data};
        const start = {start_data};
        const end = {end_data};

        const maxCanvas = 2048;
        const maxDim = Math.max(width, height);
        const cellSize = Math.max(1, Math.floor(maxCanvas / maxDim));

        const canvas = document.createElement('canvas');
        canvas.width = width * cellSize;
        canvas.height = height * cellSize;
        const ctx = canvas.getContext('2d');

        ctx.fillStyle = '#ffffff';
        ctx.fillRect(0, 0, canvas.width, canvas.height);

        for (let y = 0; y < height; y++) {{
            const row = (maze[y] || '');
            for (let x = 0; x < width; x++) {{
                const ch = row[x] || '#';
                if (ch === '#') {{
                    ctx.fillStyle = '#111111';
                }} else {{
                    ctx.fillStyle = '#ffffff';
                }}
                ctx.fillRect(x * cellSize, y * cellSize, cellSize, cellSize);
            }}
        }}

        if (path && path.length) {{
            ctx.fillStyle = '#7c3aed';
            for (let i = 0; i < path.length; i++) {{
                const p = path[i];
                const x = p[0];
                const y = p[1];
                ctx.fillRect(x * cellSize, y * cellSize, cellSize, cellSize);
            }}
        }}

        if (start) {{
            ctx.fillStyle = '#22c55e';
            ctx.fillRect(start[0] * cellSize, start[1] * cellSize, cellSize, cellSize);
        }}
        if (end) {{
            ctx.fillStyle = '#ef4444';
            ctx.fillRect(end[0] * cellSize, end[1] * cellSize, cellSize, cellSize);
        }}

        const texture = new THREE.CanvasTexture(canvas);
        texture.magFilter = THREE.NearestFilter;
        texture.minFilter = THREE.NearestFilter;
        texture.needsUpdate = true;

        const scene = new THREE.Scene();
        scene.background = new THREE.Color(0xf0f0f0);

        const renderer = new THREE.WebGLRenderer({{ antialias: true }});
        renderer.setSize(container.clientWidth, container.clientHeight);
        container.appendChild(renderer.domElement);

        const aspect = container.clientWidth / container.clientHeight;
        let viewW = width;
        let viewH = height;
        if (viewW / viewH < aspect) {{
            viewW = viewH * aspect;
        }} else {{
            viewH = viewW / aspect;
        }}

        const camera = new THREE.OrthographicCamera(-viewW / 2, viewW / 2, viewH / 2, -viewH / 2, 0.1, 1000);
        camera.position.set(0, 0, 100);
        camera.lookAt(0, 0, 0);

        const controls = new THREE.OrbitControls(camera, renderer.domElement);
        controls.enableRotate = false;
        controls.enableDamping = true;
        controls.dampingFactor = 0.05;
        controls.screenSpacePanning = true;

        const planeGeom = new THREE.PlaneGeometry(width, height);
        const planeMat = new THREE.MeshBasicMaterial({{ map: texture }});
        const plane = new THREE.Mesh(planeGeom, planeMat);
        scene.add(plane);

        if (path && path.length) {{
            const positions = new Float32Array(path.length * 3);
            for (let i = 0; i < path.length; i++) {{
                const p = path[i];
                const px = p[0] - width / 2 + 0.5;
                const py = height / 2 - p[1] - 0.5;
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

        function handleResize() {{
            const w = container.clientWidth;
            const h = container.clientHeight;
            const a = w / h;
            let vw = width;
            let vh = height;
            if (vw / vh < a) {{
                vw = vh * a;
            }} else {{
                vh = vw / a;
            }}
            camera.left = -vw / 2;
            camera.right = vw / 2;
            camera.top = vh / 2;
            camera.bottom = -vh / 2;
            camera.updateProjectionMatrix();
            renderer.setSize(w, h);
        }}
        window.addEventListener('resize', handleResize);

        function animate() {{
            requestAnimationFrame(animate);
            controls.update();
            renderer.render(scene, camera);
        }}
        animate();
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
                <div id="{viz_id}" style="width: 100%; height: 400px; border: 1px solid #ccc; background: #fafafa; border-radius: 4px;"></div>
                <div style="margin-top: 8px; font-size: 12px; color: #666;">
                    🖱️ Left click + drag to rotate | Right click + drag to pan | Scroll to zoom
                </div>
            </div>
        </details>
    </div>
    
    <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
    <script>
    (function() {{
        // Only initialize if Three.js is available and element exists
        if (typeof THREE === 'undefined' || !document.getElementById('{viz_id}')) return;
        
        const containerVertices = {container_data};
        const placements = {placements_data};
        const edgeLength = {edge_length};
        
        // Debug output
        console.log('Container vertices:', containerVertices.length, containerVertices);
        console.log('Placements:', placements.length, placements);
        console.log('Edge length:', edgeLength);
        
        // Scene setup
        const scene = new THREE.Scene();
        scene.background = new THREE.Color(0xf8f8f8);
        
        // Camera setup
        const container = document.getElementById('{viz_id}');
        const camera = new THREE.PerspectiveCamera(75, container.clientWidth / 400, 0.1, 1000);
        
        // Renderer setup
        const renderer = new THREE.WebGLRenderer({{ antialias: true }});
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
        
        // Helper function to check if two vertices should be connected by an edge
        // For convex polyhedra, edges connect vertices that are "visible" to each other
        function shouldConnect(v1, v2, allVertices) {{
            const dist = Math.sqrt(
                Math.pow(v1[0] - v2[0], 2) +
                Math.pow(v1[1] - v2[1], 2) +
                Math.pow(v1[2] - v2[2], 2)
            );
            
            // Calculate all pairwise distances to find reasonable edge length
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
            
            // Sort distances to find the most common edge lengths
            distances.sort((a, b) => a - b);
            
            // Find the shortest distances (these are likely the real edges)
            const shortestDistances = distances.slice(0, Math.min(allVertices.length * 2, distances.length));
            const avgShortDist = shortestDistances.reduce((a, b) => a + b, 0) / shortestDistances.length;
            
            // Connect if distance is close to the shortest distances (actual edges)
            return dist < avgShortDist * 1.2; // Allow 20% tolerance
        }}
        
        // Generate container edges
        for (let i = 0; i < containerVertices.length; i++) {{
            for (let j = i + 1; j < containerVertices.length; j++) {{
                if (shouldConnect(containerVertices[i], containerVertices[j], containerVertices)) {{
                    containerEdges.push([containerVertices[i], containerVertices[j]]);
                }}
            }}
        }}
        
        // Create line segments for container
        const edgePositions = [];
        containerEdges.forEach(edge => {{
            edgePositions.push(...edge[0], ...edge[1]);
        }});
        
        const containerGeometry = new THREE.BufferGeometry();
        containerGeometry.setAttribute('position', new THREE.Float32BufferAttribute(edgePositions, 3));
        const containerMaterial = new THREE.LineBasicMaterial({{ color: 0x333333, linewidth: 2 }});
        const containerMesh = new THREE.LineSegments(containerGeometry, containerMaterial);
        scene.add(containerMesh);
        
        // Tetrahedron geometry
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
            
            // Tetrahedron faces (triangles)
            const faces = [
                [0, 1, 2], [0, 2, 3], [0, 3, 1], [1, 3, 2]
            ];
            
            vertices.forEach(v => positions.push(...v));
            faces.forEach(face => indices.push(...face));
            
            geometry.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
            geometry.setIndex(indices);
            geometry.computeVertexNormals();
            
            // Create mesh with more visible material
            const mesh = new THREE.Mesh(geometry, new THREE.MeshPhongMaterial({{ 
                color: 0x4488ff, 
                transparent: true, 
                opacity: 0.8,
                side: THREE.DoubleSide,
                shininess: 100,
                specular: 0x222222
            }}));
            
            // Add wireframe overlay for better visibility
            const wireframeGeometry = new THREE.BufferGeometry();
            wireframeGeometry.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
            wireframeGeometry.setIndex(indices);
            const wireframeMaterial = new THREE.LineBasicMaterial({{ color: 0x000088, linewidth: 1 }});
            const wireframe = new THREE.LineSegments(wireframeGeometry, wireframeMaterial);
            
            // Apply rotation and translation
            mesh.position.set(...center);
            wireframe.position.set(...center);
            
            if (rotation && rotation.length >= 3) {{
                mesh.rotation.set(rotation[0], rotation[1], rotation[2]);
                wireframe.rotation.set(rotation[0], rotation[1], rotation[2]);
            }}
            
            // Return group containing both mesh and wireframe
            const group = new THREE.Group();
            group.add(mesh);
            group.add(wireframe);
            
            return group;
        }}
        
        // Add tetrahedrons
        placements.forEach((placement, index) => {{
            const tetrahedron = createTetrahedron(placement.center, placement.rotation || [0, 0, 0], edgeLength);
            // Vary color slightly for visual distinction
            const hue = (index * 30) % 360;
            // Apply color to the mesh (first child of the group)
            if (tetrahedron.children[0] && tetrahedron.children[0].material) {{
                tetrahedron.children[0].material.color.setHSL(hue / 360, 0.6, 0.5);
            }}
            scene.add(tetrahedron);
        }});
        
        // Camera positioning
        const box = new THREE.Box3().setFromObject(scene);
        const center = box.getCenter(new THREE.Vector3());
        const size = box.getSize(new THREE.Vector3());
        const maxDim = Math.max(size.x, size.y, size.z);
        const fov = camera.fov * (Math.PI / 180);
        let cameraZ = Math.abs(maxDim / 2 / Math.tan(fov / 2));
        cameraZ *= 1.5; // Add some padding
        camera.position.set(center.x + cameraZ, center.y + cameraZ, center.z + cameraZ);
        camera.lookAt(center);
        
        // Controls
        let mouseDown = false;
        let mouseX = 0;
        let mouseY = 0;
        let isRightClick = false;
        
        container.addEventListener('mousedown', (e) => {{
            mouseDown = true;
            mouseX = e.clientX;
            mouseY = e.clientY;
            isRightClick = e.button === 2;
        }});
        
        container.addEventListener('contextmenu', (e) => e.preventDefault());
        
        container.addEventListener('mousemove', (e) => {{
            if (!mouseDown) return;
            
            const deltaX = e.clientX - mouseX;
            const deltaY = e.clientY - mouseY;
            
            if (isRightClick) {{
                // Pan
                const panSpeed = 0.01;
                camera.position.x -= deltaX * panSpeed;
                camera.position.y += deltaY * panSpeed;
            }} else {{
                // Rotate around center
                const rotateSpeed = 0.005;
                const spherical = new THREE.Spherical();
                spherical.setFromVector3(camera.position.clone().sub(center));
                spherical.theta -= deltaX * rotateSpeed;
                spherical.phi += deltaY * rotateSpeed;
                spherical.phi = Math.max(0.1, Math.min(Math.PI - 0.1, spherical.phi));
                camera.position.copy(center).add(new THREE.Vector3().setFromSpherical(spherical));
                camera.lookAt(center);
            }}
            
            mouseX = e.clientX;
            mouseY = e.clientY;
        }});
        
        container.addEventListener('mouseup', () => {{
            mouseDown = false;
        }});
        
        container.addEventListener('wheel', (e) => {{
            e.preventDefault();
            const zoomSpeed = 0.1;
            const scale = e.deltaY > 0 ? (1 + zoomSpeed) : (1 - zoomSpeed);
            camera.position.multiplyScalar(scale);
        }});
        
        // Handle resize
        function handleResize() {{
            camera.aspect = container.clientWidth / 400;
            camera.updateProjectionMatrix();
            renderer.setSize(container.clientWidth, 400);
        }}
        
        window.addEventListener('resize', handleResize);
        
        // Animation loop
        function animate() {{
            requestAnimationFrame(animate);
            renderer.render(scene, camera);
        }}
        
        animate();
        
        // Cleanup on page unload
        window.addEventListener('beforeunload', () => {{
            renderer.dispose();
        }});
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
                <div id="{viz_id}" style="width: 100%; height: 400px; border: 1px solid #ccc; background: #fafafa; border-radius: 4px;"></div>
                <div style="margin-top: 8px; font-size: 12px; color: #666;">
                    🖱️ Left click + drag to rotate | Right click + drag to pan | Scroll to zoom
                </div>
            </div>
        </details>
    </div>
    
    <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
    <script>
    (function() {{
        if (typeof THREE === 'undefined' || !document.getElementById('{viz_id}')) return;
        
        const nodes = {nodes_data};
        const edges = {edges_data};
        
        // Scene setup
        const scene = new THREE.Scene();
        scene.background = new THREE.Color(0xf8f8f8);
        
        const container = document.getElementById('{viz_id}');
        const camera = new THREE.PerspectiveCamera(75, container.clientWidth / 400, 0.1, 1000);
        const renderer = new THREE.WebGLRenderer({{ antialias: true }});
        renderer.setSize(container.clientWidth, 400);
        container.appendChild(renderer.domElement);
        
        // Lighting
        const ambientLight = new THREE.AmbientLight(0xffffff, 0.6);
        scene.add(ambientLight);
        
        // Create nodes
        const nodeGeometry = new THREE.SphereGeometry(0.1, 16, 16);
        nodes.forEach((node, index) => {{
            const nodeMaterial = new THREE.MeshPhongMaterial({{ color: 0x4488ff }});
            const nodeMesh = new THREE.Mesh(nodeGeometry, nodeMaterial);
            nodeMesh.position.set(node[0], node[1], 0);
            scene.add(nodeMesh);
        }});
        
        // Create edges
        const edgeGeometry = new THREE.BufferGeometry();
        const edgePositions = [];
        edges.forEach(edge => {{
            const node1 = nodes[edge[0]];
            const node2 = nodes[edge[1]];
            edgePositions.push(node1[0], node1[1], 0, node2[0], node2[1], 0);
        }});
        
        edgeGeometry.setAttribute('position', new THREE.Float32BufferAttribute(edgePositions, 3));
        const edgeMaterial = new THREE.LineBasicMaterial({{ color: 0x333333, linewidth: 2 }});
        const edgeMesh = new THREE.LineSegments(edgeGeometry, edgeMaterial);
        scene.add(edgeMesh);
        
        // Camera positioning
        const box = new THREE.Box3().setFromObject(scene);
        const center = box.getCenter(new THREE.Vector3());
        const size = box.getSize(new THREE.Vector3());
        const maxDim = Math.max(size.x, size.y, size.z);
        const fov = camera.fov * (Math.PI / 180);
        let cameraZ = Math.abs(maxDim / 2 / Math.tan(fov / 2));
        cameraZ *= 1.5;
        camera.position.set(center.x, center.y, center.z + cameraZ);
        camera.lookAt(center);
        
        // Simple controls (same as tetrahedron visualization)
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
                spherical.setFromVector3(camera.position.clone().sub(center));
                spherical.theta -= deltaX * 0.005; spherical.phi += deltaY * 0.005;
                spherical.phi = Math.max(0.1, Math.min(Math.PI - 0.1, spherical.phi));
                camera.position.copy(center).add(new THREE.Vector3().setFromSpherical(spherical));
                camera.lookAt(center);
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
            requestAnimationFrame(animate);
            renderer.render(scene, camera);
        }}
        animate();
    }})();
    </script>
    """

  return html
