from pathlib import Path

path = Path(r'd:\CodingBenchmark\visualization_utils.py')
text = path.read_text(encoding='utf-8')


def replace_once(old: str, new: str, label: str) -> None:
    global text
    if old not in text:
        raise SystemExit(f'Anchor not found for {label}')
    text = text.replace(old, new, 1)

replace_once(
'''import json
import uuid
from typing import List, Tuple, Dict, Any


''',
'''import hashlib
import json
import uuid
from typing import List, Tuple, Dict, Any

try:
  from LLMBenchCore import ResultPaths as rp
except Exception:
  rp = None

_REPORT_PAYLOAD_INLINE_LIMIT_BYTES = 8 * 1024


def _build_viz_payload_loader(viz_id: str, payload: Dict[str, Any]) -> str:
  payload_json = json.dumps(payload, separators=(",", ":"), ensure_ascii=False)
  payload_bytes = len(payload_json.encode("utf-8"))
  payload_key = f"{viz_id}_{hashlib.sha256(payload_json.encode('utf-8')).hexdigest()[:12]}"

  inline_loader = f"""
        const payloadKey = {json.dumps(payload_key)};
        function loadPayload() {{
            return Promise.resolve({payload_json});
        }}
  """

  if rp is None or rp.get_current_model() is None or payload_bytes <= _REPORT_PAYLOAD_INLINE_LIMIT_BYTES:
    return inline_loader

  try:
    payload_path = rp.result_path(f"report_payloads/{payload_key}.js")
    payload_href = rp.report_relpath(payload_path)
    payload_script = (
      "window.__llmbenchVizPayloads=window.__llmbenchVizPayloads||{};"
      + f"window.__llmbenchVizPayloads[{json.dumps(payload_key)}]={payload_json};")
    with open(payload_path, "w", encoding="utf-8", newline="") as handle:
      handle.write(payload_script)
    return f"""
        const payloadKey = {json.dumps(payload_key)};
        const payloadScriptSrc = {json.dumps(payload_href)};
        let payloadLoadPromise = null;
        function getPayloadSync() {{
            return window.__llmbenchVizPayloads && window.__llmbenchVizPayloads[payloadKey];
        }}
        function loadPayload() {{
            const existingPayload = getPayloadSync();
            if (existingPayload) {{
                return Promise.resolve(existingPayload);
            }}
            if (payloadLoadPromise) {{
                return payloadLoadPromise;
            }}
            payloadLoadPromise = new Promise((resolve, reject) => {{
                const scriptId = 'llmbench-payload-' + payloadKey;
                const existingScript = document.getElementById(scriptId);
                const onLoad = function() {{
                    const loadedPayload = getPayloadSync();
                    if (loadedPayload) {{
                        resolve(loadedPayload);
                    }} else {{
                        reject(new Error('Visualization payload missing after script load: ' + payloadKey));
                    }}
                }};
                const onError = function() {{
                    reject(new Error('Failed to load visualization payload: ' + payloadScriptSrc));
                }};
                if (existingScript) {{
                    window.setTimeout(function() {{
                        const loadedPayload = getPayloadSync();
                        if (loadedPayload) {{
                            resolve(loadedPayload);
                        }} else {{
                            existingScript.addEventListener('load', onLoad, { once: true });
                            existingScript.addEventListener('error', onError, { once: true });
                        }}
                    }}, 0);
                    return;
                }}
                const script = document.createElement('script');
                script.id = scriptId;
                script.src = payloadScriptSrc;
                script.async = true;
                script.onload = onLoad;
                script.onerror = onError;
                document.head.appendChild(script);
            }});
            return payloadLoadPromise;
        }}
  """
  except Exception:
    return inline_loader


''',
'imports+helper')

for old, new, label in [
    (
        '''  maze_lines = maze_string.strip().split("\\n")
  maze_data = json.dumps(maze_lines)
  path_data = json.dumps([list(p) for p in path]) if path else "null"
  start_data = json.dumps(list(start) if start else None)
  end_data = json.dumps(list(end) if end else None)

''',
        '''  maze_lines = maze_string.strip().split("\\n")
  payload_loader_js = _build_viz_payload_loader(
    viz_id, {
      'maze': maze_lines,
      'pathData': [list(p) for p in path] if path else None,
      'startData': list(start) if start else None,
      'endData': list(end) if end else None,
    })

''',
        'maze payload python'),
    (
        '''        const vizId = 'maze_{viz_id}';
        const mazeWidth = {width};
        const mazeHeight = {height};
        const maze = {maze_data};
        const pathData = {path_data};
        const startData = {start_data};
        const endData = {end_data};
        
''',
        '''        const vizId = 'maze_{viz_id}';
        const mazeWidth = {width};
        const mazeHeight = {height};
        {payload_loader_js}
        
''',
        'maze payload js top'),
    (
        '''        function activate() {{
            if (isActive) return;
            isActive = true;
            
            if (typeof THREE === 'undefined') return;
            const container = document.getElementById('maze-renderer-{viz_id}');
            if (!container) return;
            
            // Clear placeholder
            const placeholder = container.querySelector('.viz-placeholder');
            if (placeholder) placeholder.style.display = 'none';

            const maxCanvas = 2048;
''',
        '''        async function activate() {{
            if (isActive) return;
            isActive = true;
            
            if (typeof THREE === 'undefined') return;
            const container = document.getElementById('maze-renderer-{viz_id}');
            if (!container) return;
            
            // Clear placeholder
            const placeholder = container.querySelector('.viz-placeholder');
            if (placeholder) placeholder.style.display = 'none';
            let payload;
            try {{
                payload = await loadPayload();
            }} catch (err) {{
                console.warn('Visualization payload failed to load:', err);
                if (placeholder) placeholder.textContent = 'Visualization data failed to load';
                isActive = false;
                return;
            }}
            if (!isActive) return;
            const maze = payload.maze;
            const pathData = payload.pathData;
            const startData = payload.startData;
            const endData = payload.endData;

            const maxCanvas = 2048;
''',
        'maze activate'),
    (
        '''  # Convert data to JSON for JavaScript
  container_data = json.dumps(container_vertices)
  placements_data = json.dumps(placements)

''',
        '''  payload_loader_js = _build_viz_payload_loader(
    viz_id, {
      'containerVertices': container_vertices,
      'placementsData': placements,
    })

''',
        'tetra payload python'),
    (
        '''        const vizId = '{viz_id}';
        const containerVertices = {container_data};
        const placementsData = {placements_data};
        const edgeLengthData = {edge_length};
        
''',
        '''        const vizId = '{viz_id}';
        {payload_loader_js}
        const edgeLengthData = {edge_length};
        
''',
        'tetra payload js top'),
    (
        '''        function activate() {{
            if (isActive) return;
            isActive = true;
            
            if (typeof THREE === 'undefined') return;
            const container = document.getElementById(vizId);
            if (!container) return;
            
            // Clear placeholder
            const placeholder = container.querySelector('.viz-placeholder');
            if (placeholder) placeholder.style.display = 'none';
            
            // Scene setup
''',
        '''        async function activate() {{
            if (isActive) return;
            isActive = true;
            
            if (typeof THREE === 'undefined') return;
            const container = document.getElementById(vizId);
            if (!container) return;
            
            // Clear placeholder
            const placeholder = container.querySelector('.viz-placeholder');
            if (placeholder) placeholder.style.display = 'none';
            let payload;
            try {{
                payload = await loadPayload();
            }} catch (err) {{
                console.warn('Visualization payload failed to load:', err);
                if (placeholder) placeholder.textContent = 'Visualization data failed to load';
                isActive = false;
                return;
            }}
            if (!isActive) return;
            const containerVertices = payload.containerVertices;
            const placementsData = payload.placementsData;
            
            // Scene setup
''',
        'tetra activate'),
    (
        '''  pts_json = json.dumps(path_points)
  rwy_json = json.dumps(runway) if runway else "null"
  n_pts = len(path_points)

''',
        '''  payload_loader_js = _build_viz_payload_loader(
    viz_id, {
      'rawPts': path_points,
      'runwayData': runway,
    })
  n_pts = len(path_points)

''',
        'flight payload python'),
    (
        '''        const vizId = 'fp_{viz_id}';
        const rawPts = {pts_json};
        const runwayData = {rwy_json};

''',
        '''        const vizId = 'fp_{viz_id}';
        {payload_loader_js}
        let rawPts = null;
        let runwayData = null;

''',
        'flight payload js top'),
    (
        '''        function activate() {{
            if (isActive) return;
            isActive = true;
            if (typeof THREE === 'undefined') return;
            const container = document.getElementById('fp-renderer-{viz_id}');
            if (!container) return;
            const ph = container.querySelector('.viz-placeholder');
            if (ph) ph.style.display = 'none';

            let minX=Infinity,minY=Infinity,minZ=Infinity;
''',
        '''        async function activate() {{
            if (isActive) return;
            isActive = true;
            if (typeof THREE === 'undefined') return;
            const container = document.getElementById('fp-renderer-{viz_id}');
            if (!container) return;
            const ph = container.querySelector('.viz-placeholder');
            if (ph) ph.style.display = 'none';
            let payload;
            try {{
                payload = await loadPayload();
            }} catch (err) {{
                console.warn('Visualization payload failed to load:', err);
                if (ph) ph.textContent = 'Visualization data failed to load';
                isActive = false;
                return;
            }}
            if (!isActive) return;
            rawPts = payload.rawPts || [];
            runwayData = payload.runwayData || null;

            let minX=Infinity,minY=Infinity,minZ=Infinity;
''',
        'flight activate'),
    (
        '''  pts_json = json.dumps(path_points)
  obs_json = json.dumps(obstacles or [])
  n_pts = len(path_points)

''',
        '''  payload_loader_js = _build_viz_payload_loader(
    viz_id, {
      'rawPts': path_points,
      'obstaclesData': obstacles or [],
    })
  n_pts = len(path_points)

''',
        'car payload python'),
    (
        '''        const vizId = 'cp_{viz_id}';
        const rawPts = {pts_json};
        const obstaclesData = {obs_json};
        const roadWidth = {road_width};
''',
        '''        const vizId = 'cp_{viz_id}';
        {payload_loader_js}
        let rawPts = [];
        let obstaclesData = [];
        const roadWidth = {road_width};
''',
        'car payload js top'),
    (
        '''        function activate() {{
            if (isActive) return;
            isActive = true;
            if (typeof THREE === 'undefined') return;
            const container = document.getElementById('cp-renderer-{viz_id}');
            if (!container) return;
            const ph = container.querySelector('.viz-placeholder');
            if (ph) ph.style.display = 'none';

            /* bounding box of car path */
''',
        '''        async function activate() {{
            if (isActive) return;
            isActive = true;
            if (typeof THREE === 'undefined') return;
            const container = document.getElementById('cp-renderer-{viz_id}');
            if (!container) return;
            const ph = container.querySelector('.viz-placeholder');
            if (ph) ph.style.display = 'none';
            let payload;
            try {{
                payload = await loadPayload();
            }} catch (err) {{
                console.warn('Visualization payload failed to load:', err);
                if (ph) ph.textContent = 'Visualization data failed to load';
                isActive = false;
                return;
            }}
            if (!isActive) return;
            rawPts = payload.rawPts || [];
            obstaclesData = payload.obstaclesData || [];

            /* bounding box of car path */
''',
        'car activate'),
    (
        '''  pts_json = json.dumps(path_points)
  n_pts = len(path_points)
  outcome = "DOCKED" if docked else ("CRASH: " + crash_reason if crashed else "timeout")
''',
        '''  payload_loader_js = _build_viz_payload_loader(viz_id, {'rawPts': path_points})
  n_pts = len(path_points)
  outcome = "DOCKED" if docked else ("CRASH: " + crash_reason if crashed else "timeout")
''',
        'docking payload python'),
    (
        '''        const vizId = 'dk_{viz_id}';
        const rawPts = {pts_json};
        const didDock = {'true' if docked else 'false'};
''',
        '''        const vizId = 'dk_{viz_id}';
        {payload_loader_js}
        let rawPts = [];
        const didDock = {'true' if docked else 'false'};
''',
        'docking payload js top'),
    (
        '''        function activate() {{
            if (isActive) return;
            isActive = true;
            if (typeof THREE === 'undefined') return;
            const container = document.getElementById('dk-renderer-{viz_id}');
            if (!container) return;
            const ph = container.querySelector('.viz-placeholder');
            if (ph) ph.style.display = 'none';

            /* Coordinate mapping: LVLH [x_rad, y_along, z_cross]
''',
        '''        async function activate() {{
            if (isActive) return;
            isActive = true;
            if (typeof THREE === 'undefined') return;
            const container = document.getElementById('dk-renderer-{viz_id}');
            if (!container) return;
            const ph = container.querySelector('.viz-placeholder');
            if (ph) ph.style.display = 'none';
            let payload;
            try {{
                payload = await loadPayload();
            }} catch (err) {{
                console.warn('Visualization payload failed to load:', err);
                if (ph) ph.textContent = 'Visualization data failed to load';
                isActive = false;
                return;
            }}
            if (!isActive) return;
            rawPts = payload.rawPts || [];

            /* Coordinate mapping: LVLH [x_rad, y_along, z_cross]
''',
        'docking activate'),
    (
        '''  gpu_frames_json = json.dumps(gpu_frames)
  ref_frames_json = json.dumps(ref_frames)
  gpu_final_json = json.dumps(gpu_final)
  ref_final_json = json.dumps(ref_final)
  num_frames = max(len(gpu_frames), len(ref_frames))

''',
        '''  payload_loader_js = _build_viz_payload_loader(
    viz_id, {
      'gpuFrames': gpu_frames,
      'refFrames': ref_frames,
      'gpuFinal': gpu_final,
      'refFinal': ref_final,
    })
  num_frames = max(len(gpu_frames), len(ref_frames))

''',
        'boid payload python'),
    (
        '''    var vizId = '{viz_id}';
    var gpuFrames = {gpu_frames_json};
    var refFrames = {ref_frames_json};
    var gpuFinal = {gpu_final_json};
    var refFinal = {ref_final_json};
    var boundSize = {bound_size};
    var nShown = {n_shown};
    var totalFrames = Math.max(gpuFrames.length, refFrames.length);

''',
        '''    var vizId = '{viz_id}';
    {payload_loader_js}
    var gpuFrames = [];
    var refFrames = [];
    var gpuFinal = [];
    var refFinal = [];
    var boundSize = {bound_size};
    var nShown = {n_shown};
    var totalFrames = 0;

''',
        'boid payload js top'),
    (
        '''    function activate() {{
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
''',
        '''    async function activate() {{
      if (isActive) return;
      isActive = true;
      if (typeof THREE === 'undefined') return;
      var container = document.getElementById('boid-renderer-' + vizId);
      if (!container) return;

      var ph = container.querySelector('.viz-placeholder');
      if (ph) ph.style.display = 'none';
      var ctrlDiv = document.getElementById('boid-controls-' + vizId);
      if (ctrlDiv) ctrlDiv.style.display = 'block';
      var payload;
      try {{
        payload = await loadPayload();
      }} catch (err) {{
        console.warn('Visualization payload failed to load:', err);
        if (ph) ph.textContent = 'Visualization data failed to load';
        if (ctrlDiv) ctrlDiv.style.display = 'none';
        isActive = false;
        return;
      }}
      if (!isActive) return;
      gpuFrames = payload.gpuFrames || [];
      refFrames = payload.refFrames || [];
      gpuFinal = payload.gpuFinal || [];
      refFinal = payload.refFinal || [];
      totalFrames = Math.max(gpuFrames.length, refFrames.length, 1);

      scene = new THREE.Scene();
''',
        'boid activate'),
]:
    replace_once(old, new, label)

path.write_text(text, encoding='utf-8', newline='\n')
print('Updated visualization_utils.py heavy visualizations')
