import math
import random
import os
import time
from pathlib import Path

from native_compiler import CSharpCompiler, compile_and_run, describe_this_pc
from solver_utils import StreamingInputFile
from collections import defaultdict

title = "Chinese Postman Problem Solver (C#)"

tags = [
  "csharp",
  "structured response",
  "graph theory",
  "optimization",
]

# Seed for reproducible graph generation
RANDOM_SEED = 12345

# Timeout in seconds (30 seconds for testing)
TIMEOUT_SECONDS = 30

# Graph complexity for each subpass: (num_nodes, approx_edges)
GRAPH_CONFIGS = [
  (6, 8),  # Small - easy
  (10, 16),  # Medium
  (15, 28),  # Larger
  (20, 45),  # Complex
  (35, 80),  # Large
  (50, 120),  # Very large
  (200, 600),  # Extreme 1
  (500, 2000),  # Extreme 2
  (1000, 5000),  # Extreme 3
  (5000, 30000),  # Extreme 4
  (10000, 80000),  # Extreme 5 - 10k nodes, 80k edges
  (50000, 500000),  # Epic - 50k nodes, 500k edges
  (100000, 1000000),  # Legendary - 100k nodes, 1M edges
]


def generate_connected_graph(num_nodes: int, num_edges: int, seed: int) -> list:
  """
    Generate a connected undirected weighted graph.
    Returns list of (node1, node2, weight) tuples.
    """
  rng = random.Random(seed)
  edges = []
  edge_set = set()

  # First ensure connectivity with a spanning tree
  nodes = list(range(num_nodes))
  rng.shuffle(nodes)
  connected = {nodes[0]}

  for i in range(1, num_nodes):
    # Connect new node to a random connected node
    new_node = nodes[i]
    existing = rng.choice(list(connected))
    weight = rng.randint(1, 100)
    edge = tuple(sorted([new_node, existing]))
    edges.append((edge[0], edge[1], weight))
    edge_set.add(edge)
    connected.add(new_node)

  # Add remaining edges randomly
  remaining = num_edges - len(edges)
  attempts = 0
  while len(edges) < num_edges and attempts < 1000:
    a = rng.randint(0, num_nodes - 1)
    b = rng.randint(0, num_nodes - 1)
    if a != b:
      edge = tuple(sorted([a, b]))
      if edge not in edge_set:
        weight = rng.randint(1, 100)
        edges.append((edge[0], edge[1], weight))
        edge_set.add(edge)
    attempts += 1

  return edges


# Pre-generate graphs for each subpass
GRAPHS_CACHE = {}
for i, (nodes, edges) in enumerate(GRAPH_CONFIGS):
  if nodes < 1000:
    GRAPHS_CACHE[i] = generate_connected_graph(nodes, edges, RANDOM_SEED + i * 100)


def format_graph_for_prompt(edges: list, num_nodes: int) -> str:
  """Format graph as adjacency list representation for the prompt."""
  lines = ["{"]
  adj = defaultdict(list)
  for a, b, w in edges:
    adj[a].append((b, w))
    adj[b].append((a, w))

  for node in range(num_nodes):
    neighbors = adj[node]
    neighbor_str = ", ".join(f"({n}, {w})" for n, w in sorted(neighbors))
    lines.append(f"    {node}: [{neighbor_str}],")
  lines.append("}")
  return "\n".join(lines)


def format_edges_for_prompt(edges: list) -> str:
  """Format edges as a list of tuples."""
  lines = ["["]
  for a, b, w in edges:
    lines.append(f"    ({a}, {b}, {w}),  # Edge between node {a} and {b}, weight {w}")
  lines.append("]")
  return "\n".join(lines)


def prepareSubpassPrompt(subPass: int) -> str:
  """Generate the prompt for subpass 0 that handles all graph sizes."""
  if subPass != 0:
    raise StopIteration

  return f"""You are solving the Chinese Postman Problem (Route Inspection Problem) in C#.

You must write a C# solver that can handle ANY graph size from trivial to ludicrous scale:
- **Trivial**: 6-20 nodes, 8-45 edges (small graphs, exact algorithms feasible)
- **Medium**: 35-200 nodes, 80-600 edges (requires efficient algorithms)
- **Large**: 500-1000 nodes, 2000-5000 edges (requires optimized implementations)
- **Extreme**: 5000-10000 nodes, 30000-80000 edges (requires very fast algorithms)
- **Epic**: 50000-100000 nodes, 500000-1000000 edges (requires highly optimized algorithms)

**The Challenge:**
Your program will be tested with graphs ranging from 6 nodes to 100000 nodes. The same function must work efficiently across ALL scales.

**Input format (stdin):**
Line 1: N M (number of nodes, number of edges)
Lines 2..M+1: u v w (edge endpoints, 0-indexed, weight integer)

**Output format (stdout):**
A sequence of node indices (whitespace-separated) representing a route that starts at node 0,
traverses EVERY edge at least once, and returns to node 0.
You may optionally prefix with a single integer L (route length). If present, it must match the number of nodes listed.

**Critical Requirements:**
1. **Scalability**: Your algorithm must adapt based on graph size and complexity
2. **Performance**: Must complete within 30 seconds even for large graphs
3. **Correctness**: Must traverse every edge at least once

**Environment:**
{describe_this_pc()}

**C# Compiler:**
{CSharpCompiler("test_engine").describe()}

Write complete, compilable C# code with a static void Main method.
"""


# List of subpasses to grade the single answer against all difficulty levels
extraGradeAnswerRuns = list(range(1, len(GRAPH_CONFIGS)))

structure = {
  "type": "object",
  "properties": {
    "reasoning": {
      "type":
      "string",
      "description":
      "Explain your approach to solving the Chinese Postman Problem and how it adapts to different graph sizes"
    },
    "csharp_code": {
      "type": "string",
      "description": "Complete C# code with Main method that handles all scales"
    }
  },
  "required": ["reasoning", "csharp_code"],
  "additionalProperties": False
}


def build_adjacency(num_nodes: int, edges: list) -> dict:
  """Build adjacency list with edge weights."""
  adj = defaultdict(list)
  for a, b, w in edges:
    adj[a].append((b, w))
    adj[b].append((a, w))
  return adj


def get_edge_weight(edges: list, a: int, b: int) -> int:
  """Get weight of edge between a and b."""
  for e1, e2, w in edges:
    if (e1 == a and e2 == b) or (e1 == b and e2 == a):
      return w
  return float('inf')


def validate_route(num_nodes: int, edges: list, route: list) -> tuple:
  """
  Validate that a route covers all edges.
  Returns (is_valid, error_message, edges_covered).
  """
  if not isinstance(route, list):
    return False, f"Route must be a list, got {type(route).__name__}", set()

  if len(route) < 2:
    return False, "Route must have at least 2 nodes", set()

  if route[0] != 0:
    return False, f"Route must start at node 0, got {route[0]}", set()

  if route[-1] != 0:
    return False, f"Route must end at node 0, got {route[-1]}", set()

  # Check all nodes are valid
  for node in route:
    if not isinstance(node, int) or node < 0 or node >= num_nodes:
      return False, f"Invalid node in route: {node}", set()

  # Build set of edges that need to be covered
  required_edges = set()
  for a, b, _ in edges:
    required_edges.add(tuple(sorted([a, b])))

  # Track which edges are covered by the route
  covered_edges = set()
  adj = build_adjacency(num_nodes, edges)

  for i in range(len(route) - 1):
    a, b = route[i], route[i + 1]
    edge = tuple(sorted([a, b]))

    # Check edge exists in graph
    if edge not in required_edges:
      return False, f"Route uses non-existent edge ({a}, {b})", covered_edges

    covered_edges.add(edge)

  # Check all edges are covered
  missing = required_edges - covered_edges
  if missing:
    missing_list = list(missing)[:5]
    return False, f"Route doesn't cover all edges. Missing: {missing_list}{'...' if len(missing) > 5 else ''}", covered_edges

  return True, "", covered_edges


def calculate_route_distance(edges: list, route: list) -> float:
  """Calculate total distance of a route."""
  edge_weights = {}
  for a, b, w in edges:
    edge_weights[tuple(sorted([a, b]))] = w

  total = 0
  for i in range(len(route) - 1):
    edge = tuple(sorted([route[i], route[i + 1]]))
    total += edge_weights.get(edge, 0)

  return total


def get_baseline_distance(num_nodes: int, edges: list) -> float:
  """
    Get baseline distance using naive greedy approach.
    This is the placebo solver's expected result.
    """
  # Sum of all edge weights (minimum if Eulerian)
  total_weight = sum(w for _, _, w in edges)

  return total_weight * 1.5


STREAMING_THRESHOLD_EDGES = 200_000
_INPUT_FILE_CACHE = {}


def format_input(num_nodes: int, edges: list) -> str:
  lines = [f"{num_nodes} {len(edges)}"]
  for a, b, w in edges:
    lines.append(f"{a} {b} {w}")
  return "\n".join(lines)


def _should_use_streaming(subpass: int) -> bool:
  _, edge_count = GRAPH_CONFIGS[subpass]
  return edge_count > STREAMING_THRESHOLD_EDGES


def _get_streaming_input(subpass: int, edges: list, num_nodes: int) -> StreamingInputFile:
  if subpass in _INPUT_FILE_CACHE:
    return _INPUT_FILE_CACHE[subpass]

  cache_key = f"cpp2|n={num_nodes}|m={len(edges)}|seed={RANDOM_SEED + subpass * 100}"

  def generator():
    yield f"{num_nodes} {len(edges)}\n"
    for a, b, w in edges:
      yield f"{a} {b} {w}\n"

  input_file = StreamingInputFile(cache_key, generator, "test2_cpp")
  _INPUT_FILE_CACHE[subpass] = input_file
  return input_file


def parse_route_output(output: str) -> tuple:
  tokens = output.strip().split()
  if not tokens:
    return None, "Empty output"

  try:
    values = [int(t) for t in tokens]
  except ValueError:
    return None, "Output contains non-integer tokens"

  if len(values) >= 2 and values[0] == len(values) - 1:
    values = values[1:]

  return values, ""


lastRoute = None


def gradeAnswer(result: dict, subPass: int, aiEngineName: str) -> tuple:
  if not result:
    return 0.0, "No result provided"

  if "csharp_code" not in result:
    return 0.0, "No C# code provided"

  global GRAPHS_CACHE
  if subPass not in GRAPHS_CACHE:
    nodes, edges_count = GRAPH_CONFIGS[subPass]
    print(f"Loading graph config {subPass} ({nodes} nodes, {edges_count} edges) into cache...")
    GRAPHS_CACHE[subPass] = generate_connected_graph(nodes, edges_count,
                                                     RANDOM_SEED + subPass * 100)

  num_nodes, _ = GRAPH_CONFIGS[subPass]
  edges = GRAPHS_CACHE[subPass]
  code = result["csharp_code"]

  if _should_use_streaming(subPass):
    streaming_input = _get_streaming_input(subPass, edges, num_nodes)
    input_file_path = streaming_input.generate()
    run = compile_and_run(code,
                          "csharp",
                          aiEngineName,
                          input_file=input_file_path,
                          timeout=TIMEOUT_SECONDS)
  else:
    input_data = format_input(num_nodes, edges)
    run = compile_and_run(code,
                          "csharp",
                          aiEngineName,
                          input_data=input_data,
                          timeout=TIMEOUT_SECONDS)

  if not run:
    return 0.0, f"[{num_nodes} nodes, {len(edges)} edges] {run.error_message()}"

  route, parse_error = parse_route_output(run.stdout)
  exec_time = run.exec_time
  if parse_error:
    return 0.0, f"[{num_nodes} nodes, {len(edges)} edges] {parse_error}"

  global lastRoute
  lastRoute = route

  # Validate the route
  is_valid, validation_error, _ = validate_route(num_nodes, edges, route)
  if not is_valid:
    return 0.0, f"[{num_nodes} nodes, {len(edges)} edges] Invalid route: {validation_error}"

  # Calculate distances
  route_distance = calculate_route_distance(edges, route)
  baseline_distance = get_baseline_distance(num_nodes, edges)

  ratio = route_distance / baseline_distance if baseline_distance > 0 else float('inf')

  # Score based on how close to baseline (or better)
  if ratio <= 1.05:
    score = 1.0
    quality = "excellent (within 5% of baseline)"
  elif ratio <= 1.3:
    score = 0.5
    quality = "good (within 30% of baseline)"
  elif ratio <= 1.5:
    score = 0.25
    quality = "acceptable (within 2x baseline)"
  else:
    score = 0.0
    quality = f"poor ({ratio:.1f}x baseline)"

  explanation = (f"[{num_nodes} nodes, {len(edges)} edges] Route distance: {route_distance:.0f}, "
                 f"Baseline: {baseline_distance:.0f}, "
                 f"Ratio: {ratio:.2f}x, "
                 f"Time: {exec_time:.1f}s - {quality}")

  return score, explanation


def resultToNiceReport(result: dict, subPass: int, aiEngineName: str) -> str:
  """Generate a nice HTML report for the result."""
  if not result:
    return "<p style='color:red'>No result provided</p>"

  num_nodes, _ = GRAPH_CONFIGS[subPass]
  edges = GRAPHS_CACHE[subPass]

  html = ""
  if subPass == 0:

    html += f"<h4>Chinese Postman Problem - {num_nodes} nodes, {len(edges)} edges</h4>"

    if "reasoning" in result:
      reasoning = result['reasoning'][:500] + ('...'
                                               if len(result.get('reasoning', '')) > 500 else '')
      html += f"<p><strong>Approach:</strong> {reasoning}</p>"

    if "csharp_code" in result:
      code = result["csharp_code"]
      code_escaped = code.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
      html += f"<details><summary>View Code ({len(code)} chars)</summary><pre>{code_escaped}</pre></details>"

  # Add SVG rendering of the route
  if lastRoute:
    if len(lastRoute) <= 1000:  # Only render for smaller routes to avoid bloat
      html += generate_route_svg(edges, lastRoute, num_nodes)
    else:
      html += f"<p>Route too large ({len(lastRoute)} nodes) to render SVG</p>"
  else:
    html += "<p style='color:red'>Could not render route: No valid route generated</p>"

  return html


def force_directed_layout(num_nodes: int, edges: list, iterations: int = 50) -> dict:
  """Simple force-directed layout algorithm to minimize edge crossings."""
  import math
  import random
  from collections import defaultdict

  # Initialize positions randomly
  positions = {i: [random.uniform(50, 350), random.uniform(50, 350)] for i in range(num_nodes)}

  # Build adjacency list
  adj = defaultdict(list)
  for a, b, _ in edges:
    adj[a].append(b)
    adj[b].append(a)

  # Force-directed parameters
  k_repulsion = 2000.0  # Repulsion force constant
  k_attraction = 0.05  # Attraction force constant
  cooling = 0.95  # Cooling factor
  temperature = 100.0  # Initial temperature

  for iteration in range(iterations):
    forces = {i: [0.0, 0.0] for i in range(num_nodes)}

    # Repulsive forces between all nodes
    for i in range(num_nodes):
      for j in range(i + 1, num_nodes):
        dx = positions[j][0] - positions[i][0]
        dy = positions[j][1] - positions[i][1]
        dist_sq = dx * dx + dy * dy

        if dist_sq < 1:  # Avoid division by very small numbers
          dist_sq = 1

        # Coulomb's law: F = k/r^2
        force = k_repulsion / dist_sq
        dist = math.sqrt(dist_sq)

        # Apply force in opposite directions
        forces[i][0] -= force * dx / dist
        forces[i][1] -= force * dy / dist
        forces[j][0] += force * dx / dist
        forces[j][1] += force * dy / dist

    # Attractive forces for connected nodes
    for a, b, _ in edges:
      dx = positions[b][0] - positions[a][0]
      dy = positions[b][1] - positions[a][1]
      dist = math.sqrt(dx * dx + dy * dy)

      if dist < 1:
        dist = 1

      # Hooke's law: F = k * distance
      force = k_attraction * dist

      # Apply force towards each other
      forces[a][0] += force * dx / dist
      forces[a][1] += force * dy / dist
      forces[b][0] -= force * dx / dist
      forces[b][1] -= force * dy / dist

    # Update positions with temperature cooling
    for i in range(num_nodes):
      # Limit force magnitude
      force_mag = math.sqrt(forces[i][0]**2 + forces[i][1]**2)
      if force_mag > temperature:
        forces[i][0] = forces[i][0] * temperature / force_mag
        forces[i][1] = forces[i][1] * temperature / force_mag

      positions[i][0] += forces[i][0]
      positions[i][1] += forces[i][1]

      # Keep within bounds
      positions[i][0] = max(20, min(380, positions[i][0]))
      positions[i][1] = max(20, min(380, positions[i][1]))

    # Cool down
    temperature *= cooling

  return positions


def generate_route_svg(edges: list, route: list, node_count: int) -> str:
  """Generate interactive path traversal visualization for Chinese Postman route."""
  if not edges or not route:
    return "<p>No route to visualize</p>"

  import json as _json

  # Use force-directed layout for better positioning
  positions = force_directed_layout(node_count, edges, iterations=50)

  # Count how many times each edge is traversed
  edge_visits = defaultdict(int)
  for i in range(len(route) - 1):
    e = tuple(sorted([route[i], route[i + 1]]))
    edge_visits[e] += 1

  # Build data for JS
  pos_list = [positions[i] for i in range(node_count)]
  edges_data = [[a, b, w] for a, b, w in edges]
  visits_data = {f"{a},{b}": c for (a, b), c in edge_visits.items()}

  uid = f"cpv_{abs(hash(tuple(route[:20]))) % 10000000}"

  return f"""
  <div style="margin:10px 0;">
    <h5>Path Traversal ({node_count} nodes, {len(edges)} edges, {len(route)-1} steps)</h5>
    <canvas id="{uid}" width="600" height="450" style="border:1px solid #ccc;background:#fafafa;width:100%;max-width:700px;"></canvas>
    <div style="display:flex;align-items:center;gap:8px;margin-top:6px;max-width:700px;">
      <button id="{uid}_play" style="padding:4px 12px;cursor:pointer;">▶</button>
      <input id="{uid}_slider" type="range" min="0" max="{len(route)-1}" value="0" style="flex:1;">
      <span id="{uid}_lbl" style="font-size:12px;color:#555;min-width:80px;">Step 0/{len(route)-1}</span>
      <select id="{uid}_spd" style="font-size:12px;"><option value="1">1x</option><option value="3" selected>3x</option><option value="10">10x</option><option value="50">50x</option></select>
    </div>
    <div style="font-size:11px;color:#666;margin-top:4px;">
      <span style="color:#ff4444;">●</span> Start (node 0) &nbsp;
      <span style="color:#4444ff;">●</span> Other nodes &nbsp;
      Edge color: <span style="color:#3388cc;">■</span> 1 visit → <span style="color:#cc3333;">■</span> 2+ visits &nbsp;
      Arrows show direction.
    </div>
  </div>
  <script>
  (function(){{
    var pos={_json.dumps(pos_list)},
        edges={_json.dumps(edges_data)},
        route={_json.dumps(route)},
        visits={_json.dumps(visits_data)};
    var canvas=document.getElementById('{uid}'),
        ctx=canvas.getContext('2d'),
        slider=document.getElementById('{uid}_slider'),
        lbl=document.getElementById('{uid}_lbl'),
        playBtn=document.getElementById('{uid}_play'),
        spdSel=document.getElementById('{uid}_spd');
    var step=0, playing=false, timer=null;
    var W=canvas.width, H=canvas.height, pad=30;
    // Map positions to canvas
    var xs=pos.map(p=>p[0]), ys=pos.map(p=>p[1]);
    var mnx=Math.min(...xs), mxx=Math.max(...xs), mny=Math.min(...ys), mxy=Math.max(...ys);
    var sx=mnx===mxx?1:(W-2*pad)/(mxx-mnx), sy=mny===mxy?1:(H-2*pad)/(mxy-mny);
    var sc=Math.min(sx,sy);
    function tx(i){{return pad+(pos[i][0]-mnx)*sc;}}
    function ty(i){{return pad+(pos[i][1]-mny)*sc;}}
    function edgeKey(a,b){{return Math.min(a,b)+','+Math.max(a,b);}}
    function lerpColor(t){{// blue->cyan->yellow->red
      t=Math.max(0,Math.min(1,t));
      var r,g,b;
      if(t<0.33){{var s=t/0.33; r=Math.round(50+s*0);g=Math.round(136+s*119);b=Math.round(204-s*4);}}
      else if(t<0.66){{var s=(t-0.33)/0.33; r=Math.round(50+s*205);g=Math.round(255-s*55);b=Math.round(200-s*200);}}
      else{{var s=(t-0.66)/0.34; r=Math.round(255-s*55);g=Math.round(200-s*160);b=Math.round(0);}}
      return 'rgb('+r+','+g+','+b+')';
    }}
    function drawArrow(x1,y1,x2,y2,color,width,offset){{
      var dx=x2-x1, dy=y2-y1, len=Math.sqrt(dx*dx+dy*dy);
      if(len<1)return;
      // Offset perpendicular for multi-visit
      var nx=-dy/len*offset, ny=dx/len*offset;
      x1+=nx;y1+=ny;x2+=nx;y2+=ny;
      ctx.strokeStyle=color; ctx.lineWidth=width;
      ctx.beginPath(); ctx.moveTo(x1,y1); ctx.lineTo(x2,y2); ctx.stroke();
      // Arrowhead at 70% along
      var mx=x1+dx*0.7+nx*0, my=y1+dy*0.7+ny*0;
      var aLen=6, aAng=0.5;
      var ux=dx/len, uy=dy/len;
      ctx.fillStyle=color; ctx.beginPath();
      ctx.moveTo(mx+ux*aLen, my+uy*aLen);
      ctx.lineTo(mx-ux*aLen-uy*aLen*aAng, my-uy*aLen+ux*aLen*aAng);
      ctx.lineTo(mx-ux*aLen+uy*aLen*aAng, my-uy*aLen-ux*aLen*aAng);
      ctx.fill();
    }}
    function draw(){{
      ctx.clearRect(0,0,W,H);
      // Draw all graph edges (light)
      ctx.strokeStyle='#ddd'; ctx.lineWidth=1;
      for(var i=0;i<edges.length;i++){{
        var a=edges[i][0],b=edges[i][1];
        ctx.beginPath(); ctx.moveTo(tx(a),ty(a)); ctx.lineTo(tx(b),ty(b)); ctx.stroke();
        // Weight label
        var mx=(tx(a)+tx(b))/2, my=(ty(a)+ty(b))/2;
        ctx.fillStyle='#bbb'; ctx.font='9px sans-serif'; ctx.textAlign='center';
        ctx.fillText(edges[i][2], mx, my-3);
      }}
      // Draw traversed edges up to current step
      var edgeCount={{}}; // count visits per directed edge at this step
      for(var i=0;i<step;i++){{
        var a=route[i], b=route[i+1];
        var dk=a+'>'+b;
        if(!edgeCount[dk])edgeCount[dk]=0;
        edgeCount[dk]++;
        var ek=edgeKey(a,b);
        var totalVisits=parseInt(visits[ek])||1;
        var thisVisitNum=edgeCount[dk];
        // Offset based on direction and visit number
        var offset=0;
        var revK=b+'>'+a;
        var revCount=edgeCount[revK]||0;
        // Positive offset for a->b, negative for b->a
        if(a<b) offset=(thisVisitNum-1)*3+1.5;
        else offset=-((thisVisitNum-1)*3+1.5);
        if(totalVisits===1 && !revCount) offset=0;
        var t=i/(Math.max(1,step-1));
        var color=lerpColor(t);
        drawArrow(tx(a),ty(a),tx(b),ty(b),color,2,offset);
      }}
      // Edge visit count badges
      var shown={{}};
      for(var i=0;i<step;i++){{
        var ek=edgeKey(route[i],route[i+1]);
        if(!shown[ek]){{
          shown[ek]=0;
        }}
        shown[ek]++;
      }}
      for(var ek in shown){{
        if(shown[ek]>1){{
          var parts=ek.split(',');
          var a=parseInt(parts[0]),b=parseInt(parts[1]);
          var mx=(tx(a)+tx(b))/2, my=(ty(a)+ty(b))/2;
          ctx.fillStyle='#c00'; ctx.font='bold 10px sans-serif'; ctx.textAlign='center';
          ctx.fillText('x'+shown[ek], mx, my+12);
        }}
      }}
      // Draw nodes
      for(var i=0;i<pos.length;i++){{
        var x=tx(i),y=ty(i);
        ctx.beginPath(); ctx.arc(x,y,5,0,Math.PI*2);
        ctx.fillStyle=(i===0)?'#ff4444':'#4466cc';
        ctx.fill(); ctx.strokeStyle='#fff'; ctx.lineWidth=1; ctx.stroke();
        ctx.fillStyle='#333'; ctx.font='10px sans-serif'; ctx.textAlign='center';
        ctx.fillText(i, x, y-8);
      }}
      // Current position marker
      if(step<route.length){{
        var ci=route[step];
        ctx.beginPath(); ctx.arc(tx(ci),ty(ci),9,0,Math.PI*2);
        ctx.strokeStyle='#ff8800'; ctx.lineWidth=3; ctx.stroke();
      }}
      lbl.textContent='Step '+step+'/'+(route.length-1)+' (node '+route[Math.min(step,route.length-1)]+')';
    }}
    function tick(){{
      if(!playing)return;
      var spd=parseInt(spdSel.value)||1;
      spd /= 100;
      step=Math.min(step+spd, route.length-1);
      slider.value=step;
      draw();
      if(step>=route.length-1){{playing=false;playBtn.textContent='▶';}}
      else timer=requestAnimationFrame(tick);
    }}
    slider.oninput=function(){{step=parseInt(this.value);draw();}};
    playBtn.onclick=function(){{
      if(playing){{playing=false;playBtn.textContent='▶';return;}}
      if(step>=route.length-1)step=0;
      playing=true;playBtn.textContent='⏸';
      timer=requestAnimationFrame(tick);
    }};
    draw();
  }})();
  </script>
  """


highLevelSummary = """
<p>Imagine a mail carrier who must walk along every street in a neighbourhood and
return home. Some streets may need to be walked twice if the network doesn't
allow a perfect loop &mdash; the goal is to minimise that extra walking.</p>
<p>The AI must figure out which streets (edges) to re-traverse and produce a
complete route. Subpasses increase the graph size, making the problem harder
to solve efficiently.</p>
"""
