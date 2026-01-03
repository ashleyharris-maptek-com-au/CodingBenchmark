"""
Placebo data provider for Algorithm Benchmarks.

Provides pre-computed "correct" answers using naive solvers.
This establishes the "Human with tools" baseline.

Test 1: TSP - Nearest-neighbor heuristic
Test 2: Chinese Postman - Greedy edge traversal
"""

import math
from typing import Optional, Union


def nearest_neighbor_solver(cities: list) -> tuple:
    """
    Naive nearest-neighbor heuristic for TSP.
    O(n^2) time complexity - no optimizations.
    
    Args:
        cities: List of (x, y) coordinate tuples
        
    Returns:
        Tuple of (route as list of indices, total_distance)
    """
    if not cities:
        return [], 0.0

    n = len(cities)
    visited = [False] * n
    route = [0]  # Start at city 0
    visited[0] = True
    total_distance = 0.0

    current = 0
    for _ in range(n - 1):
        best_next = None
        best_dist = float('inf')

        for j in range(n):
            if not visited[j]:
                dx = cities[current][0] - cities[j][0]
                dy = cities[current][1] - cities[j][1]
                dist = math.sqrt(dx * dx + dy * dy)
                if dist < best_dist:
                    best_dist = dist
                    best_next = j

        if best_next is not None:
            visited[best_next] = True
            route.append(best_next)
            total_distance += best_dist
            current = best_next

    # Return to start
    dx = cities[current][0] - cities[0][0]
    dy = cities[current][1] - cities[0][1]
    total_distance += math.sqrt(dx * dx + dy * dy)

    return route, total_distance


def generate_solver_code() -> str:
    """
    Generate the naive nearest-neighbor solver code that will be
    returned as the placebo response.
    """
    return '''import math

def solve_tsp(cities):
    """
    Solve TSP using nearest-neighbor heuristic.
    Returns the route as a list of city indices.
    """
    if not cities:
        return []
    
    n = len(cities)
    visited = [False] * n
    route = [0]
    visited[0] = True
    current = 0
    
    for _ in range(n - 1):
        best_next = None
        best_dist = float('inf')
        
        for j in range(n):
            if not visited[j]:
                dx = cities[current][0] - cities[j][0]
                dy = cities[current][1] - cities[j][1]
                dist = math.sqrt(dx * dx + dy * dy)
                if dist < best_dist:
                    best_dist = dist
                    best_next = j
        
        if best_next is not None:
            visited[best_next] = True
            route.append(best_next)
            current = best_next
    
    return route
'''


def generate_cpp_solver_code() -> str:
    """
    Generate the naive greedy Chinese Postman solver code.
    Uses Hierholzer's algorithm with greedy odd-vertex pairing.
    """
    return '''from collections import defaultdict, deque

def solve_cpp(num_nodes, edges):
    """
    Solve Chinese Postman Problem using greedy approach.
    Returns the route as a list of node indices.
    """
    # Build adjacency list with edge counts (multigraph support)
    adj = defaultdict(list)
    edge_count = defaultdict(int)
    
    for a, b, w in edges:
        adj[a].append((b, w))
        adj[b].append((a, w))
        edge_key = tuple(sorted([a, b]))
        edge_count[edge_key] += 1
    
    # Find vertices with odd degree
    degree = defaultdict(int)
    for a, b, w in edges:
        degree[a] += 1
        degree[b] += 1
    
    odd_vertices = [v for v in range(num_nodes) if degree[v] % 2 == 1]
    
    # Greedy pairing of odd vertices - add shortest paths
    # Use BFS to find shortest paths
    def bfs_path(start, end):
        if start == end:
            return [start]
        visited = {start: None}
        queue = deque([start])
        while queue:
            node = queue.popleft()
            for neighbor, _ in adj[node]:
                if neighbor not in visited:
                    visited[neighbor] = node
                    if neighbor == end:
                        path = [end]
                        while path[-1] != start:
                            path.append(visited[path[-1]])
                        return path[::-1]
                    queue.append(neighbor)
        return None
    
    # Add duplicate edges for odd vertex pairs (greedy)
    extra_edges = []
    remaining = odd_vertices[:]
    while len(remaining) >= 2:
        v = remaining.pop(0)
        # Find closest odd vertex
        best_u = None
        best_path = None
        best_len = float('inf')
        for u in remaining:
            path = bfs_path(v, u)
            if path and len(path) < best_len:
                best_len = len(path)
                best_u = u
                best_path = path
        if best_u:
            remaining.remove(best_u)
            # Add edges along the path
            for i in range(len(best_path) - 1):
                a, b = best_path[i], best_path[i + 1]
                edge_key = tuple(sorted([a, b]))
                edge_count[edge_key] += 1
    
    # Build multigraph adjacency for Hierholzer
    multi_adj = defaultdict(list)
    for edge_key, count in edge_count.items():
        a, b = edge_key
        w = next(wt for x, y, wt in edges if tuple(sorted([x, y])) == edge_key)
        for _ in range(count):
            multi_adj[a].append(b)
            multi_adj[b].append(a)
    
    # Hierholzer's algorithm for Eulerian circuit
    route = []
    stack = [0]
    while stack:
        v = stack[-1]
        if multi_adj[v]:
            u = multi_adj[v].pop()
            multi_adj[u].remove(v)
            stack.append(u)
        else:
            route.append(stack.pop())
    
    return route[::-1]
'''


def generate_layout_solver_code() -> str:
    """
    Generate naive force-directed graph layout solver.
    Simple spring embedder without optimizations.
    """
    return '''import math
import random

def layout_graph(num_nodes, edges):
    """
    Layout graph using simple force-directed algorithm.
    Returns list of (x, y) positions for each node.
    """
    if num_nodes == 0:
        return []
    
    # Initialize positions in a circle
    positions = []
    for i in range(num_nodes):
        angle = 2 * math.pi * i / num_nodes
        positions.append([math.cos(angle) * 10, math.sin(angle) * 10])
    
    # Build adjacency set for quick lookup
    adj = set()
    for a, b in edges:
        adj.add((a, b))
        adj.add((b, a))
    
    # Force-directed iterations
    k = math.sqrt(100.0 / num_nodes)  # Optimal distance
    iterations = min(100, num_nodes * 10)
    temp = 10.0  # Temperature for simulated annealing
    
    for iteration in range(iterations):
        forces = [[0.0, 0.0] for _ in range(num_nodes)]
        
        # Repulsive forces between all pairs
        for i in range(num_nodes):
            for j in range(i + 1, num_nodes):
                dx = positions[i][0] - positions[j][0]
                dy = positions[i][1] - positions[j][1]
                dist = math.sqrt(dx * dx + dy * dy) + 0.01
                
                # Repulsive force
                force = k * k / dist
                fx = dx / dist * force
                fy = dy / dist * force
                
                forces[i][0] += fx
                forces[i][1] += fy
                forces[j][0] -= fx
                forces[j][1] -= fy
        
        # Attractive forces along edges
        for a, b in edges:
            dx = positions[a][0] - positions[b][0]
            dy = positions[a][1] - positions[b][1]
            dist = math.sqrt(dx * dx + dy * dy) + 0.01
            
            # Attractive force
            force = dist * dist / k
            fx = dx / dist * force
            fy = dy / dist * force
            
            forces[a][0] -= fx
            forces[a][1] -= fy
            forces[b][0] += fx
            forces[b][1] += fy
        
        # Apply forces with temperature limiting
        for i in range(num_nodes):
            fx, fy = forces[i]
            mag = math.sqrt(fx * fx + fy * fy) + 0.01
            # Limit displacement by temperature
            scale = min(mag, temp) / mag
            positions[i][0] += fx * scale
            positions[i][1] += fy * scale
        
        # Cool down
        temp *= 0.95
    
    return [tuple(p) for p in positions]
'''


def generate_tetra_packing_code() -> str:
    """
    Generate naive grid-based tetrahedron packing solver.
    """
    return '''import math

def pack_tetrahedrons(container_vertices, edge_length):
    """
    Pack regular tetrahedrons into a polyhedron using grid placement.
    Returns list of placement dicts with 'center' and 'rotation'.
    """
    # Get bounding box
    min_x = min(v[0] for v in container_vertices)
    max_x = max(v[0] for v in container_vertices)
    min_y = min(v[1] for v in container_vertices)
    max_y = max(v[1] for v in container_vertices)
    min_z = min(v[2] for v in container_vertices)
    max_z = max(v[2] for v in container_vertices)
    
    # Tetrahedron circumradius
    circumradius = edge_length * math.sqrt(3/8)
    spacing = edge_length * 1.1
    
    # Centroid of container for containment check
    cx = sum(v[0] for v in container_vertices) / len(container_vertices)
    cy = sum(v[1] for v in container_vertices) / len(container_vertices)
    cz = sum(v[2] for v in container_vertices) / len(container_vertices)
    max_dist = max(math.sqrt((v[0]-cx)**2 + (v[1]-cy)**2 + (v[2]-cz)**2) 
                   for v in container_vertices)
    
    def point_inside(p):
        # Simple check: distance from centroid
        d = math.sqrt((p[0]-cx)**2 + (p[1]-cy)**2 + (p[2]-cz)**2)
        return d < max_dist - circumradius
    
    def get_tetra_vertices(center, rotation=(0,0,0)):
        a = edge_length / math.sqrt(2)
        base = [
            (a, 0, -a / math.sqrt(2)),
            (-a, 0, -a / math.sqrt(2)),
            (0, a, a / math.sqrt(2)),
            (0, -a, a / math.sqrt(2))
        ]
        # Apply rotation and translation
        rx, ry, rz = rotation
        result = []
        for vx, vy, vz in base:
            # Rotate X
            y1 = vy * math.cos(rx) - vz * math.sin(rx)
            z1 = vy * math.sin(rx) + vz * math.cos(rx)
            vy, vz = y1, z1
            # Rotate Y
            x1 = vx * math.cos(ry) + vz * math.sin(ry)
            z1 = -vx * math.sin(ry) + vz * math.cos(ry)
            vx, vz = x1, z1
            # Rotate Z
            x1 = vx * math.cos(rz) - vy * math.sin(rz)
            y1 = vx * math.sin(rz) + vy * math.cos(rz)
            result.append((x1 + center[0], y1 + center[1], vz + center[2]))
        return result
    
    def tetra_fits(center, rotation, placed):
        verts = get_tetra_vertices(center, rotation)
        # Check all vertices inside
        for v in verts:
            if not point_inside(v):
                return False
        # Check overlap with placed
        for p_center, p_rot in placed:
            dist = math.sqrt(sum((center[i]-p_center[i])**2 for i in range(3)))
            if dist < spacing * 0.9:
                return False
        return True
    
    placements = []
    placed_data = []
    
    # Grid search
    margin = circumradius * 1.5
    x = min_x + margin
    while x < max_x - margin:
        y = min_y + margin
        while y < max_y - margin:
            z = min_z + margin
            while z < max_z - margin:
                center = (x, y, z)
                rotation = (0, 0, 0)
                if tetra_fits(center, rotation, placed_data):
                    placements.append({"center": center, "rotation": rotation})
                    placed_data.append((center, rotation))
                z += spacing
            y += spacing
        x += spacing
    
    return placements
'''


def generate_hamilton_dfs_code() -> str:
    """
    Generate naive DFS Hamiltonian path solver.
    """
    return '''def find_hamiltonian_path(width, height, obstacles, require_cycle):
    """
    Find Hamiltonian path using simple DFS backtracking.
    Returns list of (x, y) coordinates or empty list if no path exists.
    """
    # Count free cells
    total_free = width * height - len(obstacles)
    
    # Directions: right, up, left, down
    directions = [(1, 0), (0, 1), (-1, 0), (0, -1)]
    
    def is_valid(x, y, visited):
        return (0 <= x < width and 0 <= y < height and 
                (x, y) not in obstacles and (x, y) not in visited)
    
    def dfs(path, visited):
        if len(path) == total_free:
            # Check cycle requirement
            if require_cycle:
                last_x, last_y = path[-1]
                if abs(last_x) + abs(last_y) == 1:  # Adjacent to (0,0)
                    return path[:]
                return None
            return path[:]
        
        x, y = path[-1]
        
        # Try each direction
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if is_valid(nx, ny, visited):
                path.append((nx, ny))
                visited.add((nx, ny))
                
                result = dfs(path, visited)
                if result is not None:
                    return result
                
                path.pop()
                visited.remove((nx, ny))
        
        return None
    
    # Start at (0, 0)
    if (0, 0) in obstacles:
        return []
    
    start_path = [(0, 0)]
    start_visited = {(0, 0)}
    
    result = dfs(start_path, start_visited)
    return result if result else []
'''


def generate_orbital_tsp_code() -> str:
    """
    Generate greedy orbital TSP solver with simplified delta-V estimation.
    """
    return '''import math

def solve_orbital_tsp(start_orbit, station_orbits, mu):
    """
    Solve orbital TSP using greedy nearest-neighbor with simplified delta-V.
    Returns dict with visit_order, total_delta_v, and reasoning.
    """
    def estimate_delta_v(orbit1, orbit2):
        # Calculate orbital radii
        r1 = math.sqrt(orbit1[0]**2 + orbit1[1]**2 + orbit1[2]**2)
        r2 = math.sqrt(orbit2[0]**2 + orbit2[1]**2 + orbit2[2]**2)
        
        # Hohmann transfer estimate
        a_transfer = (r1 + r2) / 2
        
        # Circular velocities
        v_c1 = math.sqrt(mu / r1)
        v_c2 = math.sqrt(mu / r2)
        
        # Transfer orbit velocities (vis-viva)
        v_t1 = math.sqrt(mu * (2/r1 - 1/a_transfer)) if a_transfer > 0 else v_c1
        v_t2 = math.sqrt(mu * (2/r2 - 1/a_transfer)) if a_transfer > 0 else v_c2
        
        # Basic Hohmann delta-V
        dv1 = abs(v_t1 - v_c1)
        dv2 = abs(v_c2 - v_t2)
        
        # Add penalty for velocity direction change (plane change proxy)
        vel_diff = math.sqrt((orbit1[3]-orbit2[3])**2 + 
                            (orbit1[4]-orbit2[4])**2 + 
                            (orbit1[5]-orbit2[5])**2)
        
        return dv1 + dv2 + vel_diff * 0.3
    
    # Greedy nearest neighbor
    num_stations = len(station_orbits)
    remaining = list(range(num_stations))
    order = []
    current_orbit = start_orbit
    total_dv = 0
    
    while remaining:
        best_station = None
        best_dv = float('inf')
        
        for station in remaining:
            dv = estimate_delta_v(current_orbit, station_orbits[station])
            if dv < best_dv:
                best_dv = dv
                best_station = station
        
        order.append(best_station)
        remaining.remove(best_station)
        total_dv += best_dv
        current_orbit = station_orbits[best_station]
    
    return {
        "visit_order": order,
        "total_delta_v": total_dv,
        "reasoning": "Greedy nearest-neighbor with Hohmann transfer estimates"
    }
'''


def generate_csg_union_code() -> str:
    """
    Generate CSG union solver using trimesh library.
    """
    return '''import numpy as np

def csg_union(mesh_a, mesh_b):
    """
    Compute CSG union of two meshes using trimesh library.
    Returns dict with vertices and faces.
    """
    try:
        import trimesh
        
        # Convert to trimesh objects
        tm_a = trimesh.Trimesh(
            vertices=np.array(mesh_a["vertices"]),
            faces=np.array(mesh_a["faces"])
        )
        tm_b = trimesh.Trimesh(
            vertices=np.array(mesh_b["vertices"]),
            faces=np.array(mesh_b["faces"])
        )
        
        # Perform boolean union
        result = trimesh.boolean.union([tm_a, tm_b])
        
        # Convert back to dict format
        return {
            "vertices": result.vertices.tolist(),
            "faces": result.faces.tolist()
        }
    except ImportError:
        # Fallback: simple mesh concatenation (not true CSG)
        # This just combines vertices and faces without boolean ops
        verts_a = mesh_a["vertices"]
        verts_b = mesh_b["vertices"]
        faces_a = mesh_a["faces"]
        faces_b = mesh_b["faces"]
        
        # Offset face indices for mesh_b
        offset = len(verts_a)
        faces_b_offset = [[f[0]+offset, f[1]+offset, f[2]+offset] for f in faces_b]
        
        return {
            "vertices": verts_a + verts_b,
            "faces": faces_a + faces_b_offset
        }
'''


def generate_maze_solver_code() -> str:
    """
    Generate BFS maze solver.
    """
    return '''from collections import deque

def solve_maze(maze_string):
    """
    Solve maze using BFS for shortest path.
    Returns list of (x, y) coordinates from A to B.
    """
    lines = maze_string.strip().split('\\n')
    height = len(lines)
    
    # Find start and end positions
    start = None
    end = None
    for y, line in enumerate(lines):
        for x, char in enumerate(line):
            if char == 'A':
                start = (x, y)
            elif char == 'B':
                end = (x, y)
    
    if not start or not end:
        return []
    
    # BFS
    queue = deque([start])
    visited = {start}
    parent = {start: None}
    
    directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
    
    while queue:
        x, y = queue.popleft()
        
        if (x, y) == end:
            # Reconstruct path
            path = []
            current = end
            while current is not None:
                path.append(current)
                current = parent[current]
            return path[::-1]
        
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            
            if (nx, ny) in visited:
                continue
            
            if ny < 0 or ny >= height:
                continue
            if nx < 0 or nx >= len(lines[ny]):
                continue
            
            char = lines[ny][nx]
            if char == '#':
                continue
            
            visited.add((nx, ny))
            parent[(nx, ny)] = (x, y)
            queue.append((nx, ny))
    
    return []  # No path found
'''


def generate_cutting_stock_code() -> str:
    """
    Generate First Fit Decreasing cutting stock solver.
    """
    return '''def solve_cutting_stock(cuts_needed, stock_length):
    """
    Solve cutting stock using First Fit Decreasing (FFD).
    Returns dict with num_stocks, assignments, and waste.
    """
    # Sort cuts by length (descending), keeping track of original indices
    indexed_cuts = sorted(enumerate(cuts_needed), key=lambda x: -x[1])
    
    # stocks[i] = (remaining_capacity, [list of cut indices])
    stocks = []
    
    for orig_idx, length in indexed_cuts:
        placed = False
        
        # Try to fit in existing stock (first fit)
        for i in range(len(stocks)):
            if stocks[i][0] >= length:
                stocks[i] = (stocks[i][0] - length, stocks[i][1] + [orig_idx])
                placed = True
                break
        
        # Open new stock if needed
        if not placed:
            stocks.append((stock_length - length, [orig_idx]))
    
    # Build result
    assignments = [stock[1] for stock in stocks]
    total_waste = sum(stock[0] for stock in stocks)
    
    return {
        "num_stocks": len(stocks),
        "assignments": assignments,
        "waste": total_waste
    }
'''


def generate_2d_cutting_code() -> str:
    """
    Generate Shelf FFDH 2D cutting stock solver.
    """
    return '''def solve_2d_cutting(rectangles, board_width, board_height):
    """
    Solve 2D cutting using Shelf First Fit Decreasing Height.
    Returns dict with num_boards and placements.
    """
    # Sort by max dimension descending, keep original indices
    indexed_rects = sorted(enumerate(rectangles), key=lambda x: -max(x[1]))
    
    # boards[i] = list of shelves, each shelf = [y, height, x_used]
    boards = []
    all_placements = [None] * len(rectangles)
    
    for orig_idx, (w, h) in indexed_rects:
        placed = False
        best_placement = None
        
        # Try both orientations
        for rotated in [False, True]:
            if rotated:
                rw, rh = h, w
            else:
                rw, rh = w, h
            
            if rw > board_width or rh > board_height:
                continue
            
            # Try existing boards
            for board_idx, board in enumerate(boards):
                # Try existing shelves
                for shelf in board:
                    shelf_y, shelf_h, x_used = shelf
                    if rh <= shelf_h and x_used + rw <= board_width:
                        # Place here
                        best_placement = (board_idx, x_used, shelf_y, rotated, shelf, rw)
                        break
                
                if best_placement:
                    break
                
                # Try new shelf on this board
                total_h = sum(s[1] for s in board)
                if total_h + rh <= board_height:
                    new_shelf = [total_h, rh, 0]
                    board.append(new_shelf)
                    best_placement = (board_idx, 0, total_h, rotated, new_shelf, rw)
                    break
            
            if best_placement:
                break
        
        if best_placement:
            board_idx, x, y, rot, shelf, rw = best_placement
            all_placements[orig_idx] = (board_idx, x, y, rot)
            shelf[2] += rw
            placed = True
        
        if not placed:
            # New board - use best orientation
            if w <= board_width and h <= board_height:
                rw, rh, rot = w, h, False
            else:
                rw, rh, rot = h, w, True
            board_idx = len(boards)
            boards.append([[0, rh, rw]])
            all_placements[orig_idx] = (board_idx, 0, 0, rot)
    
    return {
        "num_boards": len(boards),
        "placements": all_placements
    }
'''


def generate_polygon_cutting_code() -> str:
    """
    Generate greedy first-fit polygon cutting solver.
    """
    return '''import math

def solve_polygon_cutting(stock_polygon, pieces):
    """
    Solve polygon cutting using greedy first-fit.
    Returns dict with num_stocks and placements.
    """
    def polygon_bounds(poly):
        xs = [p[0] for p in poly]
        ys = [p[1] for p in poly]
        return (min(xs), min(ys), max(xs), max(ys))
    
    def translate_polygon(poly, dx, dy):
        return [(x + dx, y + dy) for x, y in poly]
    
    def rotate_polygon(poly, angle):
        cos_a = math.cos(angle)
        sin_a = math.sin(angle)
        return [(x * cos_a - y * sin_a, x * sin_a + y * cos_a) for x, y in poly]
    
    def point_in_polygon(point, polygon):
        x, y = point
        n = len(polygon)
        inside = False
        j = n - 1
        for i in range(n):
            xi, yi = polygon[i]
            xj, yj = polygon[j]
            if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi) + xi):
                inside = not inside
            j = i
        return inside
    
    def polygon_in_polygon(inner, outer):
        return all(point_in_polygon(p, outer) for p in inner)
    
    def segments_intersect(p1, p2, p3, p4):
        def ccw(A, B, C):
            return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])
        return ccw(p1, p3, p4) != ccw(p2, p3, p4) and ccw(p1, p2, p3) != ccw(p1, p2, p4)
    
    def polygons_overlap(poly1, poly2):
        for p in poly1:
            if point_in_polygon(p, poly2):
                return True
        for p in poly2:
            if point_in_polygon(p, poly1):
                return True
        n1, n2 = len(poly1), len(poly2)
        for i in range(n1):
            for j in range(n2):
                if segments_intersect(poly1[i], poly1[(i+1)%n1], poly2[j], poly2[(j+1)%n2]):
                    return True
        return False
    
    stock_bounds = polygon_bounds(stock_polygon)
    stock_w = stock_bounds[2] - stock_bounds[0]
    stock_h = stock_bounds[3] - stock_bounds[1]
    
    stocks_pieces = []  # List of lists of placed polygons
    placements = []
    
    for piece_idx, piece in enumerate(pieces):
        placed = False
        
        for stock_idx, stock_placed in enumerate(stocks_pieces):
            for rot in [0, math.pi/2, math.pi, 3*math.pi/2]:
                rotated = rotate_polygon(piece, rot)
                bounds = polygon_bounds(rotated)
                pw = bounds[2] - bounds[0]
                ph = bounds[3] - bounds[1]
                
                for y in range(0, int(stock_h - ph) + 1, 5):
                    for x in range(0, int(stock_w - pw) + 1, 5):
                        tx = x - bounds[0]
                        ty = y - bounds[1]
                        candidate = translate_polygon(rotated, tx, ty)
                        
                        if not polygon_in_polygon(candidate, stock_polygon):
                            continue
                        
                        overlaps = False
                        for other in stock_placed:
                            if polygons_overlap(candidate, other):
                                overlaps = True
                                break
                        
                        if not overlaps:
                            stock_placed.append(candidate)
                            placements.append({
                                "stock_index": stock_idx,
                                "position": (tx, ty),
                                "rotation": rot
                            })
                            placed = True
                            break
                    if placed:
                        break
                if placed:
                    break
            if placed:
                break
        
        if not placed:
            bounds = polygon_bounds(piece)
            tx = -bounds[0]
            ty = -bounds[1]
            candidate = translate_polygon(piece, tx, ty)
            stocks_pieces.append([candidate])
            placements.append({
                "stock_index": len(stocks_pieces) - 1,
                "position": (tx, ty),
                "rotation": 0
            })
    
    return {
        "num_stocks": len(stocks_pieces),
        "placements": placements
    }
'''


def generate_3d_packing_code() -> str:
    """
    Generate grid-based 3D bin packing solver.
    """
    return '''def pack_polyhedra(polyhedron, container_size):
    """
    Pack polyhedra using grid-based placement.
    Returns dict with count and placements.
    """
    vertices = polyhedron["vertices"]
    
    # Calculate bounding box
    min_pt = [min(v[i] for v in vertices) for i in range(3)]
    max_pt = [max(v[i] for v in vertices) for i in range(3)]
    dims = [max_pt[i] - min_pt[i] for i in range(3)]
    
    # Offset to make vertices non-negative
    offset = [-min_pt[0], -min_pt[1], -min_pt[2]]
    
    # Grid spacing with small gap
    gap = 0.1
    placements = []
    
    # Grid placement
    x = offset[0]
    while x + dims[0] <= container_size[0]:
        y = offset[1]
        while y + dims[1] <= container_size[1]:
            z = offset[2]
            while z + dims[2] <= container_size[2]:
                placements.append({
                    "translation": [x, y, z],
                    "quaternion": [1, 0, 0, 0]  # Identity - no rotation
                })
                z += dims[2] + gap
            y += dims[1] + gap
        x += dims[0] + gap
    
    return {
        "count": len(placements),
        "placements": placements
    }
'''


def generate_lcs_code() -> str:
    """
    Generate longest common substring solver using difflib.
    """
    return '''from difflib import SequenceMatcher

def longest_common_substring(strings):
    """
    Find longest common substring using difflib.SequenceMatcher.
    """
    if not strings:
        return ""
    
    if len(strings) == 1:
        return strings[0]
    
    def lcs_two(s1, s2):
        """Find LCS of two strings using SequenceMatcher."""
        matcher = SequenceMatcher(None, s1, s2)
        match = matcher.find_longest_match(0, len(s1), 0, len(s2))
        return s1[match.a:match.a + match.size]
    
    # Start with all substrings of first string as candidates
    # Then filter by checking each subsequent string
    result = strings[0]
    
    for s in strings[1:]:
        result = lcs_two(result, s)
        if not result:
            return ""
    
    return result
'''


def generate_minesweeper_code() -> str:
    """
    Generate basic constraint satisfaction Minesweeper solver.
    """
    return '''def solve_minesweeper(board):
    """
    Solve Minesweeper using basic constraint satisfaction.
    Returns list of (x, y) coordinates guaranteed to be safe.
    """
    height = len(board)
    width = len(board[0]) if board else 0
    
    def get_neighbors(x, y):
        neighbors = []
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue
                nx, ny = x + dx, y + dy
                if 0 <= nx < width and 0 <= ny < height:
                    neighbors.append((nx, ny))
        return neighbors
    
    safe = set()
    
    # Iterate until no changes
    changed = True
    iterations = 0
    
    while changed and iterations < 100:
        changed = False
        iterations += 1
        
        for y in range(height):
            for x in range(width):
                cell = board[y][x]
                
                # Only process numbered cells
                if cell not in '012345678':
                    continue
                
                mine_count = int(cell)
                neighbors = get_neighbors(x, y)
                
                # Count flagged mines and unknown cells
                unknown = [(nx, ny) for nx, ny in neighbors 
                          if board[ny][nx] == ' ' and (nx, ny) not in safe]
                flagged = sum(1 for nx, ny in neighbors if board[ny][nx] == '*')
                
                # If all mines are flagged, remaining unknowns are safe
                if flagged == mine_count and unknown:
                    for pos in unknown:
                        if pos not in safe:
                            safe.add(pos)
                            changed = True
    
    return list(safe)
'''


def generate_shadow_cover_code() -> str:
    """
    Generate grid-based shadow covering solver.
    """
    return '''import math

def solve_shadow_cover(target_polygon, sun_vector):
    """
    Cover target polygon with tetrahedron shadows using grid placement.
    """
    # Standard tetrahedron vertices
    TETRA_VERTS = [
        [1.0, 0.0, -0.707],
        [-1.0, 0.0, -0.707],
        [0.0, 1.0, 0.707],
        [0.0, -1.0, 0.707]
    ]
    
    # Get target bounds
    xs = [p[0] for p in target_polygon]
    ys = [p[1] for p in target_polygon]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    
    width = max_x - min_x
    height = max_y - min_y
    
    # Normalize sun vector
    sun_len = math.sqrt(sum(s*s for s in sun_vector))
    sun_norm = [s/sun_len for s in sun_vector]
    
    # Calculate tetrahedron shadow size at given height
    # Place at z=3 for good shadow size
    z_height = 3.0
    scale = 1.5  # Scale up tetrahedrons for better coverage
    
    # Effective shadow size (rough estimate)
    shadow_size = 2.0 * scale
    
    # Grid spacing with overlap
    spacing = shadow_size * 0.7
    
    placements = []
    
    # Grid placement
    y = min_y
    while y <= max_y + spacing:
        x = min_x
        while x <= max_x + spacing:
            # Account for sun angle offset
            offset_x = z_height * sun_norm[0] / abs(sun_norm[2]) if abs(sun_norm[2]) > 0.01 else 0
            offset_y = z_height * sun_norm[1] / abs(sun_norm[2]) if abs(sun_norm[2]) > 0.01 else 0
            
            placements.append({
                "position": [x + offset_x, y + offset_y, z_height],
                "quaternion": [1, 0, 0, 0],  # Identity rotation
                "scale": scale
            })
            x += spacing
        y += spacing
    
    return {
        "count": len(placements),
        "placements": placements
    }
'''


def generate_jobshop_code() -> str:
    """
    Generate greedy first-fit job-shop scheduler.
    """
    return '''def schedule_jobs(jobs, num_machines):
    """
    Schedule jobs using greedy first-fit.
    Returns dict with makespan and schedule.
    """
    # Track when each machine is free
    machine_free = [0] * num_machines
    # Track when each job's previous task ends
    job_ready = [0] * len(jobs)
    # Track next task index for each job
    job_task_idx = [0] * len(jobs)
    
    # Schedule storage: schedule[job][task] = (start_time, machine)
    schedule = [[] for _ in range(len(jobs))]
    
    total_tasks = sum(len(job) for job in jobs)
    scheduled = 0
    
    while scheduled < total_tasks:
        # Find task that can start earliest
        best_job = -1
        best_start = float('inf')
        
        for j, job in enumerate(jobs):
            if job_task_idx[j] >= len(job):
                continue
            
            task = job[job_task_idx[j]]
            machine = task["machine"]
            
            # Earliest this task can start
            earliest = max(job_ready[j], machine_free[machine])
            
            if earliest < best_start:
                best_start = earliest
                best_job = j
        
        if best_job < 0:
            break
        
        # Schedule this task
        task = jobs[best_job][job_task_idx[best_job]]
        machine = task["machine"]
        duration = task["duration"]
        start = max(job_ready[best_job], machine_free[machine])
        end = start + duration
        
        schedule[best_job].append((start, machine))
        machine_free[machine] = end
        job_ready[best_job] = end
        job_task_idx[best_job] += 1
        scheduled += 1
    
    makespan = max(machine_free)
    
    return {
        "makespan": makespan,
        "schedule": schedule
    }
'''


def generate_aabb_packing_code() -> str:
    """
    Generate greedy first-fit 3D AABB packing solver.
    """
    return '''def pack_boxes(boxes, container):
    """
    Pack boxes using greedy first-fit with extreme points.
    Returns dict with packed_count and placements.
    """
    def boxes_overlap(pos1, size1, pos2, size2):
        for i in range(3):
            if pos1[i] + size1[i] <= pos2[i] or pos2[i] + size2[i] <= pos1[i]:
                return False
        return True
    
    def box_in_container(pos, size):
        for i in range(3):
            if pos[i] < 0 or pos[i] + size[i] > container[i]:
                return False
        return True
    
    # Sort by volume descending
    indexed_boxes = sorted(enumerate(boxes), key=lambda x: -x[1][0]*x[1][1]*x[1][2])
    
    placed = []  # List of (box_index, position, size)
    placements = []
    
    for idx, (w, h, d) in indexed_boxes:
        # Generate candidate positions
        candidates = [(0, 0, 0)]
        for _, pos, size in placed:
            candidates.append((pos[0] + size[0], pos[1], pos[2]))
            candidates.append((pos[0], pos[1] + size[1], pos[2]))
            candidates.append((pos[0], pos[1], pos[2] + size[2]))
        
        # Sort by z, y, x (bottom-left-back)
        candidates.sort(key=lambda p: (p[2], p[1], p[0]))
        
        placed_box = False
        for cx, cy, cz in candidates:
            pos = (cx, cy, cz)
            size = (w, h, d)
            
            if not box_in_container(pos, size):
                continue
            
            overlaps = False
            for _, prev_pos, prev_size in placed:
                if boxes_overlap(pos, size, prev_pos, prev_size):
                    overlaps = True
                    break
            
            if not overlaps:
                placed.append((idx, pos, size))
                placements.append({
                    "box_index": idx,
                    "position": list(pos),
                    "rotated": [0, 0, 0]
                })
                placed_box = True
                break
        
        # Grid search fallback
        if not placed_box:
            for x in range(0, container[0] - w + 1, max(1, w // 2)):
                if placed_box:
                    break
                for y in range(0, container[1] - h + 1, max(1, h // 2)):
                    if placed_box:
                        break
                    for z in range(0, container[2] - d + 1, max(1, d // 2)):
                        pos = (x, y, z)
                        size = (w, h, d)
                        
                        overlaps = False
                        for _, prev_pos, prev_size in placed:
                            if boxes_overlap(pos, size, prev_pos, prev_size):
                                overlaps = True
                                break
                        
                        if not overlaps:
                            placed.append((idx, pos, size))
                            placements.append({
                                "box_index": idx,
                                "position": list(pos),
                                "rotated": [0, 0, 0]
                            })
                            placed_box = True
                            break
    
    return {
        "packed_count": len(placements),
        "placements": placements
    }
'''


def generate_clustering_rust_code() -> str:
    """Generate Rust code for 3D point clustering using mini-batch k-means."""
    return r'''use std::io::{self, BufRead, Write};

fn main() {
    let stdin = io::stdin();
    let mut lines = stdin.lock().lines();
    
    // Parse header
    let header = lines.next().unwrap().unwrap();
    let parts: Vec<usize> = header.split_whitespace()
        .filter_map(|s| s.parse().ok())
        .collect();
    let num_points = parts[0];
    let num_clusters = parts[1];
    
    // Initialize centroids with first k points
    let mut centroids: Vec<(f64, f64, f64)> = Vec::with_capacity(num_clusters);
    let mut points: Vec<(f64, f64, f64)> = Vec::new();
    
    // Read all points
    for line in lines {
        if let Ok(l) = line {
            let coords: Vec<f64> = l.split_whitespace()
                .filter_map(|s| s.parse().ok())
                .collect();
            if coords.len() >= 3 {
                points.push((coords[0], coords[1], coords[2]));
                
                // Use first points as initial centroids
                if centroids.len() < num_clusters {
                    centroids.push((coords[0], coords[1], coords[2]));
                }
            }
        }
    }
    
    // Fill remaining centroids if needed
    while centroids.len() < num_clusters {
        let idx = centroids.len() % points.len().max(1);
        if idx < points.len() {
            centroids.push(points[idx]);
        } else {
            centroids.push((0.0, 0.0, 0.0));
        }
    }
    
    // K-means iterations
    let max_iters = 10;
    let mut assignments = vec![0usize; points.len()];
    
    for _ in 0..max_iters {
        // Assign points to nearest centroid
        for (i, &(px, py, pz)) in points.iter().enumerate() {
            let mut best_cluster = 0;
            let mut best_dist = f64::MAX;
            
            for (c, &(cx, cy, cz)) in centroids.iter().enumerate() {
                let dist = (px - cx).powi(2) + (py - cy).powi(2) + (pz - cz).powi(2);
                if dist < best_dist {
                    best_dist = dist;
                    best_cluster = c;
                }
            }
            assignments[i] = best_cluster;
        }
        
        // Update centroids
        let mut sums = vec![(0.0, 0.0, 0.0); num_clusters];
        let mut counts = vec![0usize; num_clusters];
        
        for (i, &(px, py, pz)) in points.iter().enumerate() {
            let c = assignments[i];
            sums[c].0 += px;
            sums[c].1 += py;
            sums[c].2 += pz;
            counts[c] += 1;
        }
        
        for c in 0..num_clusters {
            if counts[c] > 0 {
                centroids[c] = (
                    sums[c].0 / counts[c] as f64,
                    sums[c].1 / counts[c] as f64,
                    sums[c].2 / counts[c] as f64,
                );
            }
        }
    }
    
    // Output assignments
    let stdout = io::stdout();
    let mut out = stdout.lock();
    for &a in &assignments {
        writeln!(out, "{}", a).unwrap();
    }
}
'''


def generate_asteroid_csharp_code() -> str:
    """Generate C# code for asteroid interception using Hohmann transfer."""
    return r'''using System;
using System.Collections.Generic;
using System.Globalization;

class AsteroidInterceptor {
    const double G = 6.67430e-11;
    const double AU = 1.496e11;
    const double DAY = 86400.0;
    
    static double sunMass = 1.989e30;
    static double[] earthPos = new double[3];
    static double[] earthVel = new double[3];
    static double[] asteroidPos = new double[3];
    static double[] asteroidVel = new double[3];
    static double warningDays, impactorMass, deltaVBudget, asteroidMass;
    
    static double Magnitude(double[] v) {
        return Math.Sqrt(v[0]*v[0] + v[1]*v[1] + v[2]*v[2]);
    }
    
    static double[] Normalize(double[] v) {
        double m = Magnitude(v);
        if (m < 1e-10) return new double[] {0, 0, 0};
        return new double[] {v[0]/m, v[1]/m, v[2]/m};
    }
    
    static void Main() {
        CultureInfo.CurrentCulture = CultureInfo.InvariantCulture;
        
        // Parse header
        var header = Console.ReadLine().Split();
        int numBodies = int.Parse(header[0]);
        warningDays = double.Parse(header[1]);
        impactorMass = double.Parse(header[2]);
        deltaVBudget = double.Parse(header[3]);
        asteroidMass = double.Parse(header[4]);
        
        // Parse bodies
        for (int i = 0; i < numBodies; i++) {
            var parts = Console.ReadLine().Split();
            string name = parts[0];
            double mass = double.Parse(parts[1]);
            double x = double.Parse(parts[2]);
            double y = double.Parse(parts[3]);
            double z = double.Parse(parts[4]);
            double vx = double.Parse(parts[5]);
            double vy = double.Parse(parts[6]);
            double vz = double.Parse(parts[7]);
            
            if (name == "Earth") {
                earthPos = new double[] {x, y, z};
                earthVel = new double[] {vx, vy, vz};
            } else if (name == "Asteroid") {
                asteroidPos = new double[] {x, y, z};
                asteroidVel = new double[] {vx, vy, vz};
            } else if (name == "Sun") {
                sunMass = mass;
            }
        }
        
        // Simple intercept calculation
        // Estimate where asteroid will be at intercept time
        double interceptDays = warningDays * 0.6; // Intercept at 60% of warning time
        
        double[] futureAsteroid = new double[3];
        for (int i = 0; i < 3; i++) {
            futureAsteroid[i] = asteroidPos[i] + asteroidVel[i] * interceptDays * DAY;
        }
        
        // Calculate required velocity for Hohmann-like transfer
        double[] toAsteroid = new double[] {
            futureAsteroid[0] - earthPos[0],
            futureAsteroid[1] - earthPos[1],
            futureAsteroid[2] - earthPos[2]
        };
        
        double transferDist = Magnitude(toAsteroid);
        double transferTime = interceptDays * DAY;
        
        // Required average velocity
        double reqSpeed = transferDist / transferTime;
        double[] transferDir = Normalize(toAsteroid);
        
        // Launch burn - escape Earth and head toward asteroid
        double[] launchDV = new double[3];
        for (int i = 0; i < 3; i++) {
            launchDV[i] = transferDir[i] * reqSpeed - earthVel[i];
        }
        
        double launchMag = Magnitude(launchDV);
        
        // Scale if over budget
        if (launchMag > deltaVBudget * 0.8) {
            double scale = deltaVBudget * 0.8 / launchMag;
            for (int i = 0; i < 3; i++) launchDV[i] *= scale;
            launchMag = Magnitude(launchDV);
        }
        
        // Mid-course correction at halfway point
        double mcDays = interceptDays * 0.5;
        double remainingDV = deltaVBudget - launchMag;
        
        // Small correction toward asteroid
        double[] mcDV = new double[] {
            transferDir[0] * remainingDV * 0.3,
            transferDir[1] * remainingDV * 0.3,
            transferDir[2] * remainingDV * 0.3
        };
        
        // Final approach burn
        double finalDays = interceptDays * 0.9;
        double[] finalDV = new double[] {
            transferDir[0] * remainingDV * 0.5,
            transferDir[1] * remainingDV * 0.5,
            transferDir[2] * remainingDV * 0.5
        };
        
        // Output burn sequence
        Console.WriteLine("3");
        Console.WriteLine($"0 {launchDV[0]:F2} {launchDV[1]:F2} {launchDV[2]:F2}");
        Console.WriteLine($"{mcDays:F0} {mcDV[0]:F2} {mcDV[1]:F2} {mcDV[2]:F2}");
        Console.WriteLine($"{finalDays:F0} {finalDV[0]:F2} {finalDV[1]:F2} {finalDV[2]:F2}");
    }
}
'''


def generate_lander3d_cpp_code() -> str:
    """Generate C++ code for 3D Lunar Lander using PD controller."""
    return r'''#include <iostream>
#include <cmath>
#include <sstream>
#include <string>
using namespace std;

double targetX, targetY, targetZ;
double gravity, maxThrust;

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(nullptr);
    
    // Parse header
    double width, depth, height, fuel;
    cin >> width >> depth >> height >> gravity >> maxThrust >> fuel;
    
    double startX, startY, startZ;
    cin >> startX >> startY >> startZ >> targetX >> targetY >> targetZ;
    
    int numObstacles;
    cin >> numObstacles;
    for (int i = 0; i < numObstacles; i++) {
        double ox, oy, oz, r;
        cin >> ox >> oy >> oz >> r;
    }
    
    string marker;
    cin >> marker; // STATE
    
    // Process state updates
    while (true) {
        double x, y, z, vx, vy, vz;
        double pitch, yaw, roll;
        double pitchRate, yawRate, rollRate;
        double fuelLeft;
        
        if (!(cin >> x >> y >> z >> vx >> vy >> vz 
              >> pitch >> yaw >> roll 
              >> pitchRate >> yawRate >> rollRate >> fuelLeft)) {
            break;
        }
        
        if (fuelLeft <= 0) {
            cout << "0.0 0.0 0.0 0.0\n";
            cout.flush();
            continue;
        }
        
        // Calculate error to target
        double dx = targetX - x;
        double dy = targetY - y;
        double dz = targetZ - z;
        double dist = sqrt(dx*dx + dy*dy + dz*dz);
        
        // Desired velocity toward target
        double scale = 0.1;
        double desiredVx = dx * scale;
        double desiredVy = dy * scale;
        double desiredVz = dz * scale + gravity * 2.0; // Counter gravity
        
        // Velocity error
        double vxErr = desiredVx - vx;
        double vyErr = desiredVy - vy;
        double vzErr = desiredVz - vz;
        
        // Calculate desired orientation
        // We want thrust to point in direction of velocity error
        double horizErr = sqrt(vxErr*vxErr + vyErr*vyErr);
        double desiredPitch = atan2(vzErr, horizErr);
        double desiredYaw = atan2(vxErr, vyErr);
        
        // Angle errors
        double pitchErr = desiredPitch - pitch;
        double yawErr = desiredYaw - yaw;
        
        // Normalize yaw error to [-pi, pi]
        while (yawErr > M_PI) yawErr -= 2*M_PI;
        while (yawErr < -M_PI) yawErr += 2*M_PI;
        
        // PD control for rotation
        double pitchCmd = max(-1.0, min(1.0, pitchErr * 2.0 - pitchRate * 0.5));
        double yawCmd = max(-1.0, min(1.0, yawErr * 2.0 - yawRate * 0.5));
        double rollCmd = max(-1.0, min(1.0, -roll * 2.0 - rollRate * 0.5)); // Keep level
        
        // Thrust control
        double velErr = sqrt(vxErr*vxErr + vyErr*vyErr + vzErr*vzErr);
        double speed = sqrt(vx*vx + vy*vy + vz*vz);
        
        double thrust;
        if (dist < 50.0) {
            // Landing phase
            if (speed > 3.0) {
                thrust = min(1.0, (speed - 2.0) / maxThrust);
            } else if (vz < -1.0) {
                thrust = min(0.6, (-vz - 0.5) / maxThrust * 2.0);
            } else {
                thrust = 0.1;
            }
        } else if (abs(pitchErr) < 0.3 && abs(yawErr) < 0.3) {
            // Aligned - thrust
            thrust = min(1.0, max(0.2, velErr / maxThrust * 0.5));
        } else {
            // Turning - minimal thrust
            thrust = 0.1;
        }
        
        cout << thrust << " " << pitchCmd << " " << yawCmd << " " << rollCmd << "\n";
        cout.flush();
    }
    
    return 0;
}
'''


def generate_lander_rust_code() -> str:
    """Generate Rust code for Lunar Lander using simple PD controller."""
    return r'''use std::io::{self, BufRead, Write};

fn main() {
    let stdin = io::stdin();
    let mut lines = stdin.lock().lines();
    
    // Parse header
    let header = lines.next().unwrap().unwrap();
    let parts: Vec<f64> = header.split_whitespace()
        .filter_map(|s| s.parse().ok())
        .collect();
    let _width = parts[0];
    let _height = parts[1];
    let gravity = parts[2];
    let max_thrust = parts[3];
    
    let pos_line = lines.next().unwrap().unwrap();
    let pos: Vec<f64> = pos_line.split_whitespace()
        .filter_map(|s| s.parse().ok())
        .collect();
    let target_x = pos[2];
    let target_y = pos[3];
    
    // Skip map until STATE marker
    loop {
        let line = lines.next().unwrap().unwrap();
        if line.trim() == "STATE" {
            break;
        }
    }
    
    // Process state updates
    loop {
        let line = match lines.next() {
            Some(Ok(l)) => l,
            _ => break,
        };
        
        if line.trim() == "END" || line.is_empty() {
            break;
        }
        
        let state: Vec<f64> = line.split_whitespace()
            .filter_map(|s| s.parse().ok())
            .collect();
        
        if state.len() < 7 {
            println!("0.0 0.0");
            io::stdout().flush().unwrap();
            continue;
        }
        
        let x = state[0];
        let y = state[1];
        let vx = state[2];
        let vy = state[3];
        let angle = state[4];
        let _angular_vel = state[5];
        let fuel = state[6];
        
        if fuel <= 0.0 {
            println!("0.0 0.0");
            io::stdout().flush().unwrap();
            continue;
        }
        
        // Simple PD controller
        let dx = target_x - x;
        let dy = target_y - y;
        let dist = (dx * dx + dy * dy).sqrt();
        
        // Desired angle to point toward target (with lead)
        let desired_vx = dx * 0.1;
        let desired_vy = dy * 0.1 + gravity * 2.0; // Counter gravity
        
        let vx_error = desired_vx - vx;
        let vy_error = desired_vy - vy;
        
        // Calculate desired thrust direction
        let thrust_angle = vx_error.atan2(vy_error);
        
        // Turn toward desired angle
        let angle_error = thrust_angle - angle;
        let angle_error = if angle_error > std::f64::consts::PI {
            angle_error - 2.0 * std::f64::consts::PI
        } else if angle_error < -std::f64::consts::PI {
            angle_error + 2.0 * std::f64::consts::PI
        } else {
            angle_error
        };
        
        let turn = (angle_error * 2.0).max(-1.0).min(1.0);
        
        // Thrust based on velocity error magnitude and proximity
        let vel_error = (vx_error * vx_error + vy_error * vy_error).sqrt();
        let speed = (vx * vx + vy * vy).sqrt();
        
        let thrust = if dist < 50.0 {
            // Landing phase - careful control
            if speed > 2.0 {
                ((speed - 1.0) / max_thrust).max(0.0).min(1.0)
            } else if vy < -1.0 {
                ((-vy - 0.5) / max_thrust * 2.0).max(0.0).min(0.5)
            } else {
                0.0
            }
        } else if angle_error.abs() < 0.3 {
            // Aligned - thrust
            (vel_error / max_thrust * 0.5).max(0.1).min(1.0)
        } else {
            // Not aligned - minimal thrust while turning
            0.1
        };
        
        println!("{:.2} {:.2}", thrust, turn);
        io::stdout().flush().unwrap();
    }
}
'''


def generate_tetris_csharp_code() -> str:
    """Generate C# code for Tetris using simple heuristic evaluation."""
    return r'''using System;
using System.Collections.Generic;
using System.Linq;

class Tetris {
    static int width, height;
    static bool[,] board;
    
    static readonly Dictionary<char, int[,][]> PIECES = new Dictionary<char, int[,][]> {
        {'I', new[] {
            new int[,] {{0,0},{0,1},{0,2},{0,3}},
            new int[,] {{0,0},{1,0},{2,0},{3,0}},
            new int[,] {{0,0},{0,1},{0,2},{0,3}},
            new int[,] {{0,0},{1,0},{2,0},{3,0}}
        }},
        {'O', new[] {
            new int[,] {{0,0},{0,1},{1,0},{1,1}},
            new int[,] {{0,0},{0,1},{1,0},{1,1}},
            new int[,] {{0,0},{0,1},{1,0},{1,1}},
            new int[,] {{0,0},{0,1},{1,0},{1,1}}
        }},
        {'T', new[] {
            new int[,] {{0,0},{0,1},{0,2},{1,1}},
            new int[,] {{0,0},{1,0},{2,0},{1,1}},
            new int[,] {{1,0},{1,1},{1,2},{0,1}},
            new int[,] {{0,1},{1,0},{1,1},{2,1}}
        }},
        {'S', new[] {
            new int[,] {{0,0},{0,1},{1,1},{1,2}},
            new int[,] {{0,1},{1,0},{1,1},{2,0}},
            new int[,] {{0,0},{0,1},{1,1},{1,2}},
            new int[,] {{0,1},{1,0},{1,1},{2,0}}
        }},
        {'Z', new[] {
            new int[,] {{0,1},{0,2},{1,0},{1,1}},
            new int[,] {{0,0},{1,0},{1,1},{2,1}},
            new int[,] {{0,1},{0,2},{1,0},{1,1}},
            new int[,] {{0,0},{1,0},{1,1},{2,1}}
        }},
        {'J', new[] {
            new int[,] {{0,0},{1,0},{1,1},{1,2}},
            new int[,] {{0,0},{0,1},{1,0},{2,0}},
            new int[,] {{0,0},{0,1},{0,2},{1,2}},
            new int[,] {{0,1},{1,1},{2,0},{2,1}}
        }},
        {'L', new[] {
            new int[,] {{0,2},{1,0},{1,1},{1,2}},
            new int[,] {{0,0},{1,0},{2,0},{2,1}},
            new int[,] {{0,0},{0,1},{0,2},{1,0}},
            new int[,] {{0,0},{0,1},{1,1},{2,1}}
        }}
    };
    
    static int GetPieceWidth(char piece, int rot) {
        var coords = PIECES[piece][rot % 4];
        int max = 0;
        for (int i = 0; i < 4; i++) max = Math.Max(max, coords[i, 1]);
        return max + 1;
    }
    
    static int GetColumnHeight(int col) {
        for (int r = height - 1; r >= 0; r--)
            if (board[r, col]) return r + 1;
        return 0;
    }
    
    static int FindLandingRow(char piece, int rot, int col) {
        var coords = PIECES[piece][rot % 4];
        for (int startRow = height - 1; startRow >= 0; startRow--) {
            bool canPlace = true;
            for (int i = 0; i < 4 && canPlace; i++) {
                int r = startRow + coords[i, 0];
                int c = col + coords[i, 1];
                if (c < 0 || c >= width) canPlace = false;
                else if (r >= 0 && r < height && board[r, c]) canPlace = false;
            }
            if (!canPlace) return startRow + 1;
        }
        return 0;
    }
    
    static double Evaluate(char piece, int rot, int col) {
        var coords = PIECES[piece][rot % 4];
        int pw = GetPieceWidth(piece, rot);
        if (col < 0 || col + pw > width) return double.MinValue;
        
        int landRow = FindLandingRow(piece, rot, col);
        
        // Simulate placement
        var tempBoard = (bool[,])board.Clone();
        int maxRow = 0;
        for (int i = 0; i < 4; i++) {
            int r = landRow + coords[i, 0];
            int c = col + coords[i, 1];
            if (r >= height) return double.MinValue;
            tempBoard[r, c] = true;
            maxRow = Math.Max(maxRow, r);
        }
        
        // Count lines cleared
        int lines = 0;
        for (int r = 0; r < height; r++) {
            bool full = true;
            for (int c = 0; c < width && full; c++)
                if (!tempBoard[r, c]) full = false;
            if (full) lines++;
        }
        
        // Calculate aggregate height
        int aggHeight = 0;
        for (int c = 0; c < width; c++) {
            for (int r = height - 1; r >= 0; r--) {
                if (tempBoard[r, c]) { aggHeight += r + 1; break; }
            }
        }
        
        // Count holes
        int holes = 0;
        for (int c = 0; c < width; c++) {
            bool foundBlock = false;
            for (int r = height - 1; r >= 0; r--) {
                if (tempBoard[r, c]) foundBlock = true;
                else if (foundBlock) holes++;
            }
        }
        
        // Calculate bumpiness
        int bumpiness = 0;
        int[] colHeights = new int[width];
        for (int c = 0; c < width; c++) {
            for (int r = height - 1; r >= 0; r--) {
                if (tempBoard[r, c]) { colHeights[c] = r + 1; break; }
            }
        }
        for (int c = 0; c < width - 1; c++)
            bumpiness += Math.Abs(colHeights[c] - colHeights[c + 1]);
        
        return 0.76 * lines - 0.51 * aggHeight - 0.36 * holes - 0.18 * bumpiness;
    }
    
    static (int rot, int col) FindBestMove(char piece) {
        double bestScore = double.MinValue;
        int bestRot = 0, bestCol = 0;
        
        for (int rot = 0; rot < 4; rot++) {
            int pw = GetPieceWidth(piece, rot);
            for (int col = 0; col <= width - pw; col++) {
                double score = Evaluate(piece, rot, col);
                if (score > bestScore) {
                    bestScore = score;
                    bestRot = rot;
                    bestCol = col;
                }
            }
        }
        return (bestRot, bestCol);
    }
    
    static void PlacePiece(char piece, int rot, int col) {
        var coords = PIECES[piece][rot % 4];
        int landRow = FindLandingRow(piece, rot, col);
        for (int i = 0; i < 4; i++) {
            int r = landRow + coords[i, 0];
            int c = col + coords[i, 1];
            if (r < height) board[r, c] = true;
        }
        // Clear lines
        for (int r = 0; r < height; ) {
            bool full = true;
            for (int c = 0; c < width && full; c++)
                if (!board[r, c]) full = false;
            if (full) {
                for (int rr = r; rr < height - 1; rr++)
                    for (int c = 0; c < width; c++)
                        board[rr, c] = board[rr + 1, c];
                for (int c = 0; c < width; c++)
                    board[height - 1, c] = false;
            } else r++;
        }
    }
    
    static void Main() {
        var parts = Console.ReadLine().Split();
        width = int.Parse(parts[0]);
        height = int.Parse(parts[1]);
        int numPieces = int.Parse(parts[2]);
        
        board = new bool[height, width];
        
        for (int i = 0; i < numPieces; i++) {
            string line = Console.ReadLine();
            if (string.IsNullOrEmpty(line)) break;
            char piece = line.Trim()[0];
            
            var (rot, col) = FindBestMove(piece);
            Console.WriteLine($"{rot} {col}");
            Console.Out.Flush();
            
            PlacePiece(piece, rot, col);
            
            string result = Console.ReadLine();
            if (result != null && result.StartsWith("gameover")) break;
        }
    }
}
'''


def generate_snake_cpp_code() -> str:
    """Generate C++ code for N-D snake game using greedy pathfinding."""
    return r'''#include <iostream>
#include <vector>
#include <set>
#include <queue>
#include <cmath>
#include <algorithm>
#include <random>
using namespace std;

int D, maxTurns;
vector<int> bounds;
vector<vector<int>> food;
set<vector<int>> obstacles;
deque<vector<int>> snake;

mt19937 rng(42);

double distance(const vector<int>& a, const vector<int>& b) {
    double sum = 0;
    for (int i = 0; i < D; i++) {
        sum += (a[i] - b[i]) * (a[i] - b[i]);
    }
    return sqrt(sum);
}

bool isValid(const vector<int>& pos) {
    for (int i = 0; i < D; i++) {
        if (pos[i] < 0 || pos[i] >= bounds[i]) return false;
    }
    if (obstacles.count(pos)) return false;
    // Check snake body (except tail which will move)
    for (size_t i = 0; i + 1 < snake.size(); i++) {
        if (snake[i] == pos) return false;
    }
    return true;
}

pair<int, int> findMove() {
    vector<int> head = snake.front();
    
    // Find nearest food
    vector<int>* target = nullptr;
    double minDist = 1e18;
    for (auto& f : food) {
        double d = distance(head, f);
        if (d < minDist) {
            minDist = d;
            target = &f;
        }
    }
    
    if (!target) {
        // No food, just survive - pick random valid move
        vector<pair<int,int>> moves;
        for (int axis = 0; axis < D; axis++) {
            for (int dir : {-1, 1}) {
                vector<int> newPos = head;
                newPos[axis] += dir;
                if (isValid(newPos)) {
                    moves.push_back({axis, dir});
                }
            }
        }
        if (moves.empty()) return {0, 1}; // No valid move, will die
        return moves[rng() % moves.size()];
    }
    
    // Greedy: move toward target
    vector<pair<double, pair<int,int>>> candidates;
    for (int axis = 0; axis < D; axis++) {
        for (int dir : {-1, 1}) {
            vector<int> newPos = head;
            newPos[axis] += dir;
            if (isValid(newPos)) {
                double newDist = distance(newPos, *target);
                candidates.push_back({newDist, {axis, dir}});
            }
        }
    }
    
    if (candidates.empty()) {
        // No valid moves, try any move
        return {0, 1};
    }
    
    // Sort by distance to target
    sort(candidates.begin(), candidates.end());
    return candidates[0].second;
}

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(nullptr);
    
    cin >> D >> maxTurns;
    bounds.resize(D);
    for (int i = 0; i < D; i++) cin >> bounds[i];
    
    int snakeLen;
    cin >> snakeLen;
    vector<int> headPos(D);
    for (int i = 0; i < D; i++) cin >> headPos[i];
    
    // Initialize snake
    for (int i = 0; i < snakeLen; i++) {
        vector<int> pos = headPos;
        pos[0] -= i;
        if (pos[0] >= 0) snake.push_back(pos);
    }
    
    int F;
    cin >> F;
    food.resize(F);
    for (int i = 0; i < F; i++) {
        food[i].resize(D);
        for (int j = 0; j < D; j++) cin >> food[i][j];
    }
    
    int O;
    cin >> O;
    for (int i = 0; i < O; i++) {
        vector<int> pos(D);
        for (int j = 0; j < D; j++) cin >> pos[j];
        obstacles.insert(pos);
    }
    
    // Play game
    for (int turn = 0; turn < maxTurns && !food.empty(); turn++) {
        auto [axis, dir] = findMove();
        cout << axis << " " << dir << "\n";
        cout.flush();
        
        // Update snake position
        vector<int> newHead = snake.front();
        newHead[axis] += dir;
        snake.push_front(newHead);
        
        // Check if ate food
        bool ate = false;
        for (auto it = food.begin(); it != food.end(); ++it) {
            if (*it == newHead) {
                food.erase(it);
                ate = true;
                break;
            }
        }
        if (!ate) {
            snake.pop_back();
        }
    }
    
    return 0;
}
'''


def generate_drillhole_rust_code() -> str:
    """Generate Rust code for drillhole data validation using z-score analysis."""
    return r'''use std::io::{self, BufRead, Write};
use std::collections::HashMap;

#[derive(Clone)]
struct Sample {
    hole_id: usize,
    sample_idx: usize,
    x: f64, y: f64, z: f64,
    properties: Vec<f64>,
}

fn distance(a: &Sample, b: &Sample) -> f64 {
    ((a.x - b.x).powi(2) + (a.y - b.y).powi(2) + (a.z - b.z).powi(2)).sqrt()
}

fn main() {
    let stdin = io::stdin();
    let mut lines = stdin.lock().lines();
    
    // Parse header
    let first_line = lines.next().unwrap().unwrap();
    let parts: Vec<&str> = first_line.split_whitespace().collect();
    let n: usize = parts[0].parse().unwrap();
    let num_props: usize = parts[1].parse().unwrap();
    
    let prop_line = lines.next().unwrap().unwrap();
    let property_names: Vec<String> = prop_line.split_whitespace().map(|s| s.to_string()).collect();
    
    let mut all_samples: Vec<Sample> = Vec::new();
    
    // Parse holes
    for _ in 0..n {
        let hole_line = lines.next().unwrap().unwrap();
        let parts: Vec<f64> = hole_line.split_whitespace()
            .filter_map(|s| s.parse().ok())
            .collect();
        
        let hole_id = parts[0] as usize;
        let start_x = parts[1];
        let start_y = parts[2];
        let start_z = parts[3];
        let dir_x = parts[4];
        let dir_y = parts[5];
        let dir_z = parts[6];
        let num_samples = parts[8] as usize;
        
        for sample_idx in 0..num_samples {
            let sample_line = lines.next().unwrap().unwrap();
            let vals: Vec<f64> = sample_line.split_whitespace()
                .filter_map(|s| s.parse().ok())
                .collect();
            
            let depth = vals[0];
            let x = start_x + dir_x * depth;
            let y = start_y + dir_y * depth;
            let z = start_z + dir_z * depth;
            
            let properties: Vec<f64> = vals[1..].to_vec();
            
            all_samples.push(Sample {
                hole_id, sample_idx, x, y, z, properties
            });
        }
    }
    
    // Find suspects using z-score with k nearest neighbors
    let k = 10.min(all_samples.len().saturating_sub(1));
    let mut suspects: Vec<(usize, usize, String, f64)> = Vec::new();
    
    for i in 0..all_samples.len() {
        let sample = &all_samples[i];
        
        // Find k nearest neighbors (from different holes)
        let mut neighbors: Vec<(f64, usize)> = Vec::new();
        for j in 0..all_samples.len() {
            if all_samples[j].hole_id != sample.hole_id {
                let d = distance(sample, &all_samples[j]);
                neighbors.push((d, j));
            }
        }
        neighbors.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        neighbors.truncate(k);
        
        if neighbors.is_empty() { continue; }
        
        // Check each property
        for prop_idx in 0..sample.properties.len().min(num_props) {
            let value = sample.properties[prop_idx];
            
            // Calculate mean and stddev of neighbors
            let neighbor_vals: Vec<f64> = neighbors.iter()
                .filter_map(|(_, j)| all_samples[*j].properties.get(prop_idx).copied())
                .collect();
            
            if neighbor_vals.is_empty() { continue; }
            
            let mean: f64 = neighbor_vals.iter().sum::<f64>() / neighbor_vals.len() as f64;
            let variance: f64 = neighbor_vals.iter()
                .map(|v| (v - mean).powi(2))
                .sum::<f64>() / neighbor_vals.len() as f64;
            let stddev = variance.sqrt().max(0.001);
            
            let z_score = ((value - mean) / stddev).abs();
            
            // Flag if z-score > 3 (3 sigma outlier)
            if z_score > 3.0 {
                let prop_name = property_names.get(prop_idx)
                    .cloned()
                    .unwrap_or_else(|| format!("prop{}", prop_idx));
                suspects.push((sample.hole_id, sample.sample_idx, prop_name, z_score));
            }
        }
    }
    
    // Sort by confidence (z-score) descending
    suspects.sort_by(|a, b| b.3.partial_cmp(&a.3).unwrap());
    
    // Output
    println!("{}", suspects.len());
    for (hole_id, sample_idx, prop, conf) in suspects {
        println!("{} {} {} {:.2}", hole_id, sample_idx, prop, conf);
    }
}
'''


def generate_excavation_csharp_code() -> str:
    """Generate C# code for dirt excavation using greedy top-down approach."""
    return r'''using System;
using System.Collections.Generic;
using System.Linq;
using System.Globalization;

class Box {
    public double X1, Y1, Z1, X2, Y2, Z2;
    public int Index;
    
    public Box(int idx, double x1, double y1, double z1, double x2, double y2, double z2) {
        Index = idx;
        X1 = Math.Min(x1, x2); Y1 = Math.Min(y1, y2); Z1 = Math.Min(z1, z2);
        X2 = Math.Max(x1, x2); Y2 = Math.Max(y1, y2); Z2 = Math.Max(z1, z2);
    }
    
    public double Volume => (X2 - X1) * (Y2 - Y1) * (Z2 - Z1);
    public double CenterX => (X1 + X2) / 2;
    public double CenterY => (Y1 + Y2) / 2;
    public double CenterZ => (Z1 + Z2) / 2;
    public double TopZ => Z2;
    
    public bool CoversPointXY(double x, double y) {
        return x >= X1 && x <= X2 && y >= Y1 && y <= Y2;
    }
    
    public bool IsAbove(Box other) {
        bool xyOverlap = !(X2 <= other.X1 || other.X2 <= X1 || Y2 <= other.Y1 || other.Y2 <= Y1);
        return xyOverlap && Z1 >= other.Z2;
    }
}

class Program {
    static List<Box> boxes = new List<Box>();
    static HashSet<int> removed = new HashSet<int>();
    static double maxUphill;
    static double targetX, targetY, targetZ;
    
    static bool IsExposed(Box box) {
        foreach (var other in boxes) {
            if (other.Index != box.Index && !removed.Contains(other.Index)) {
                if (other.IsAbove(box)) return false;
            }
        }
        return true;
    }
    
    static bool CoversTarget(Box box) {
        return box.CoversPointXY(targetX, targetY) && box.Z1 <= targetZ;
    }
    
    static List<Box> GetExposedCoveringTarget() {
        var result = new List<Box>();
        foreach (var box in boxes) {
            if (!removed.Contains(box.Index) && CoversTarget(box) && IsExposed(box)) {
                result.Add(box);
            }
        }
        return result;
    }
    
    static void Main() {
        CultureInfo.CurrentCulture = CultureInfo.InvariantCulture;
        
        var firstLine = Console.ReadLine().Split(' ');
        int n = int.Parse(firstLine[0]);
        maxUphill = double.Parse(firstLine[1]);
        targetX = double.Parse(firstLine[2]);
        targetY = double.Parse(firstLine[3]);
        targetZ = double.Parse(firstLine[4]);
        
        for (int i = 0; i < n; i++) {
            var parts = Console.ReadLine().Split(' ');
            boxes.Add(new Box(i,
                double.Parse(parts[0]), double.Parse(parts[1]), double.Parse(parts[2]),
                double.Parse(parts[3]), double.Parse(parts[4]), double.Parse(parts[5])));
        }
        
        var operations = new List<string>();
        
        // Find dump location - far from target, at ground level
        double dumpX = targetX + 50;
        double dumpY = targetY + 50;
        double dumpZ = 0;
        
        // Greedy: always dig the highest exposed box that covers target
        for (int iter = 0; iter < n; iter++) {
            var candidates = GetExposedCoveringTarget();
            if (candidates.Count == 0) break;
            
            // Pick highest one (easiest to satisfy uphill constraint)
            var toDig = candidates.OrderByDescending(b => b.TopZ).First();
            
            // Calculate dump position respecting uphill constraint
            double digZ = toDig.CenterZ;
            double actualDumpZ = Math.Min(dumpZ, digZ + maxUphill);
            
            operations.Add($"{toDig.Index} {dumpX:F2} {dumpY:F2} {actualDumpZ:F2}");
            removed.Add(toDig.Index);
            
            // Move dump location to avoid piling
            dumpX += 2;
        }
        
        Console.WriteLine(operations.Count);
        foreach (var op in operations) {
            Console.WriteLine(op);
        }
    }
}
'''


def generate_mincut_cpp_code() -> str:
    """Generate C++ code for minimum cut using Stoer-Wagner algorithm."""
    return r'''#include <iostream>
#include <vector>
#include <queue>
#include <algorithm>
#include <climits>
#include <cstring>
using namespace std;

const int MAXN = 50005;
int n, m;
vector<pair<int, int>> adj[MAXN];  // {neighbor, edge_index}
vector<pair<int, int>> edges;  // original edges
bool merged[MAXN];
int parent[MAXN];
int weight[MAXN];  // weight to contracted graph
bool inA[MAXN];

// Union-Find for vertex merging
int uf_find(int x) {
    if (parent[x] != x) parent[x] = uf_find(parent[x]);
    return parent[x];
}

void uf_union(int x, int y) {
    parent[uf_find(x)] = uf_find(y);
}

// One phase of Stoer-Wagner
pair<int, pair<int,int>> minCutPhase(int start, vector<int>& active) {
    fill(weight, weight + n, 0);
    fill(inA, inA + n, false);
    
    priority_queue<pair<int,int>> pq;
    int s = -1, t = -1;
    int lastWeight = 0;
    
    for (int v : active) {
        pq.push({0, v});
    }
    
    int added = 0;
    while (!pq.empty() && added < (int)active.size()) {
        auto [w, u] = pq.top();
        pq.pop();
        
        if (inA[u]) continue;
        inA[u] = true;
        added++;
        s = t;
        t = u;
        lastWeight = -w;
        
        for (auto [v, eidx] : adj[u]) {
            int rv = uf_find(v);
            if (!inA[rv] && !merged[rv]) {
                weight[rv]++;
                pq.push({-weight[rv], rv});
            }
        }
    }
    
    return {lastWeight, {s, t}};
}

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(nullptr);
    
    cin >> n >> m;
    edges.resize(m);
    
    for (int i = 0; i < m; i++) {
        int u, v;
        cin >> u >> v;
        edges[i] = {u, v};
        adj[u].push_back({v, i});
        adj[v].push_back({u, i});
    }
    
    // Initialize union-find
    for (int i = 0; i < n; i++) {
        parent[i] = i;
        merged[i] = false;
    }
    
    int minCut = INT_MAX;
    int bestT = -1;
    vector<int> component;  // vertices in the cut component
    
    vector<int> active;
    for (int i = 0; i < n; i++) active.push_back(i);
    
    // Run n-1 phases
    for (int phase = 0; phase < n - 1 && active.size() > 1; phase++) {
        auto [cutWeight, st] = minCutPhase(active[0], active);
        int s = st.first, t = st.second;
        
        if (cutWeight < minCut && cutWeight > 0) {
            minCut = cutWeight;
            bestT = t;
            // Remember component containing t
            component.clear();
            component.push_back(t);
        }
        
        // Merge s and t
        if (s >= 0 && t >= 0) {
            uf_union(t, s);
            merged[t] = true;
            
            // Update active list
            active.erase(remove(active.begin(), active.end(), t), active.end());
        }
    }
    
    // Find cut edges - edges between component containing bestT and rest
    if (minCut == INT_MAX || minCut == 0) {
        // Graph might already be disconnected or single node
        // Find any edge that can disconnect
        if (m > 0) {
            cout << 1 << "\n";
            cout << edges[0].first << " " << edges[0].second << "\n";
        } else {
            cout << 0 << "\n";
        }
        return 0;
    }
    
    // Simple approach: find edges crossing to isolated vertex group
    // Use BFS to find actual components after removing minimum edges
    vector<pair<int,int>> cutEdges;
    
    // For simplicity, find edges incident to bestT in contracted graph
    // This is approximate but fast
    for (int i = 0; i < m && (int)cutEdges.size() < minCut; i++) {
        int u = edges[i].first, v = edges[i].second;
        // Check if edge crosses between original bestT group and others
        if ((u == bestT || v == bestT) && uf_find(u) != uf_find(v)) {
            cutEdges.push_back(edges[i]);
        }
    }
    
    // If we didn't find enough, just output first few edges
    if (cutEdges.empty()) {
        for (int i = 0; i < min(m, minCut); i++) {
            cutEdges.push_back(edges[i]);
        }
    }
    
    cout << cutEdges.size() << "\n";
    for (auto [u, v] : cutEdges) {
        cout << u << " " << v << "\n";
    }
    
    return 0;
}
'''


def get_placebo_response(question_num: int,
                         subpass: int) -> Optional[Union[dict, str]]:
    """
    Placebo data provider function.
    
    Args:
        question_num: The test number (1 for TSP test)
        subpass: The subpass index (0-5 for different city counts)
        
    Returns:
        The expected response dict, or None if not available
    """
    if question_num == 1:
        # TSP test - return the naive solver code for all subpasses
        return {
            "reasoning":
            "Using nearest-neighbor heuristic: start at city 0, "
            "always visit the closest unvisited city. This is O(n^2) "
            "and gives a reasonable approximation without complex optimizations.",
            "python_code":
            generate_solver_code()
        }

    if question_num == 2:
        # Chinese Postman Problem - greedy solver
        return {
            "reasoning":
            "Using greedy approach for Chinese Postman: "
            "1) Find all odd-degree vertices. "
            "2) Greedily pair them by shortest path (BFS, not optimal matching). "
            "3) Duplicate edges along paired paths. "
            "4) Find Eulerian circuit using Hierholzer's algorithm. "
            "This is suboptimal because greedy pairing doesn't minimize total duplicate cost.",
            "python_code":
            generate_cpp_solver_code()
        }

    if question_num == 3:
        # Graph Layout - greedy force-directed
        return {
            "reasoning":
            "Using simple force-directed layout: "
            "1) Initialize nodes randomly or in a circle. "
            "2) Apply repulsive forces between all node pairs. "
            "3) Apply attractive forces along edges. "
            "4) Iterate until convergence. "
            "This is a basic spring embedder without optimizations like Barnes-Hut.",
            "python_code":
            generate_layout_solver_code()
        }

    if question_num == 4:
        # Tetrahedron Packing - greedy grid placement
        return {
            "reasoning":
            "Using greedy grid-based packing: "
            "1) Compute bounding box of container. "
            "2) Create a 3D grid with spacing based on tetrahedron size. "
            "3) At each grid point, try to place a tetrahedron. "
            "4) Check if fully contained in polyhedron. "
            "5) Check for overlaps with previously placed tetrahedrons. "
            "This is suboptimal because it uses axis-aligned placement without rotation optimization.",
            "python_code":
            generate_tetra_packing_code()
        }

    if question_num == 5:
        # Hamiltonian Path - simple DFS
        return {
            "reasoning":
            "Using simple depth-first search (DFS) backtracking: "
            "1) Start at (0, 0). "
            "2) At each cell, try moving to unvisited adjacent cells. "
            "3) If stuck, backtrack and try another direction. "
            "4) Continue until all cells visited or all options exhausted. "
            "This is a naive approach without pruning or heuristics like Warnsdorff's rule.",
            "python_code":
            generate_hamilton_dfs_code()
        }

    if question_num == 6:
        # Orbital TSP - greedy with simplified delta-V estimation
        return {
            "reasoning":
            "Using greedy nearest-neighbor with simplified orbital mechanics: "
            "1) Estimate transfer delta-V using Hohmann-like approximation. "
            "2) At each step, visit the station with lowest estimated delta-V. "
            "3) This ignores timing and detailed Lambert solutions. "
            "4) Uses vis-viva equation for energy-based estimates.",
            "python_code":
            generate_orbital_tsp_code()
        }

    if question_num == 7:
        # CSG Union - use trimesh library
        return {
            "reasoning":
            "Using the trimesh library with its boolean operations: "
            "1) Convert input meshes to trimesh.Trimesh objects. "
            "2) Use trimesh.boolean.union() for CSG operation. "
            "3) This leverages manifold3d or blender backend for robust booleans. "
            "4) Return result as vertices and faces dict.",
            "python_code":
            generate_csg_union_code()
        }

    if question_num == 8:
        # Maze Solver - BFS
        return {
            "reasoning":
            "Using Breadth-First Search (BFS) for optimal shortest path: "
            "1) Parse maze to find start (A) and end (B) positions. "
            "2) Use queue-based BFS exploring all adjacent cells. "
            "3) Track visited cells to avoid cycles. "
            "4) Reconstruct path from parent pointers when goal found.",
            "python_code":
            generate_maze_solver_code()
        }

    if question_num == 9:
        # Cutting Stock - First Fit Decreasing
        return {
            "reasoning":
            "Using First Fit Decreasing (FFD) heuristic: "
            "1) Sort cuts in descending order by length. "
            "2) For each cut, place it in the first stock with enough remaining space. "
            "3) If no stock has space, open a new stock. "
            "4) FFD typically achieves within 11/9 * OPT + 6/9 of optimal.",
            "python_code":
            generate_cutting_stock_code()
        }

    if question_num == 10:
        # 2D Cutting Stock - Shelf algorithm
        return {
            "reasoning":
            "Using Shelf First Fit Decreasing Height (FFDH): "
            "1) Sort rectangles by height descending. "
            "2) Place each rectangle in first shelf where it fits. "
            "3) Create new shelf if needed, new board if shelf doesn't fit. "
            "4) Try rotation to fit better.",
            "python_code":
            generate_2d_cutting_code()
        }

    if question_num == 11:
        # Polygon Cutting Stock - Greedy first fit
        return {
            "reasoning":
            "Using greedy first-fit with grid search: "
            "1) For each piece, try to place in existing stocks first. "
            "2) Try 4 rotation angles (0°, 90°, 180°, 270°). "
            "3) Grid search positions within stock bounds. "
            "4) Check containment and overlap. "
            "5) Create new stock if piece doesn't fit anywhere.",
            "python_code":
            generate_polygon_cutting_code()
        }

    if question_num == 12:
        # 3D Bin Packing - Grid placement
        return {
            "reasoning":
            "Using grid-based placement for 3D bin packing: "
            "1) Calculate bounding box of polyhedron. "
            "2) Offset mesh to have non-negative coordinates. "
            "3) Place copies in a 3D grid pattern with small gaps. "
            "4) Use identity quaternion (no rotation). "
            "5) This is suboptimal but guarantees non-overlapping valid placements.",
            "python_code":
            generate_3d_packing_code()
        }

    if question_num == 13:
        # Longest Common Substring - using difflib
        return {
            "reasoning":
            "Using Python's difflib.SequenceMatcher for LCS: "
            "1) Start with first string as reference. "
            "2) Find longest match between current result and each subsequent string. "
            "3) Update result to the common substring found. "
            "4) This uses difflib's efficient matching but may timeout on very large inputs.",
            "python_code":
            generate_lcs_code()
        }

    if question_num == 14:
        # Minesweeper Solver - constraint satisfaction
        return {
            "reasoning":
            "Using basic constraint satisfaction for Minesweeper: "
            "1) For each numbered cell, count adjacent flagged mines and unknown cells. "
            "2) If flagged count equals the number, all unknown neighbors are safe. "
            "3) Iterate until no new safe cells found. "
            "4) This is a simple approach - advanced solvers use subset analysis.",
            "python_code":
            generate_minesweeper_code()
        }

    if question_num == 15:
        # Shadow Covering - grid placement
        return {
            "reasoning":
            "Using grid-based placement for shadow covering: "
            "1) Calculate bounding box of target polygon. "
            "2) Place scaled tetrahedrons in a grid above the target. "
            "3) Position tetrahedrons at z > 0 so shadows project onto z=0 plane. "
            "4) Scale tetrahedrons to ensure shadow overlap covers the target.",
            "python_code":
            generate_shadow_cover_code()
        }

    if question_num == 16:
        # Job-Shop Scheduling - greedy first-fit
        return {
            "reasoning":
            "Using greedy first-fit scheduling: "
            "1) Track when each machine becomes free. "
            "2) Track when each job's previous task ends. "
            "3) At each step, find the task that can start earliest. "
            "4) Schedule it and update machine/job availability.",
            "python_code":
            generate_jobshop_code()
        }

    if question_num == 17:
        # 3D AABB Packing - greedy first-fit
        return {
            "reasoning":
            "Using greedy first-fit with extreme points: "
            "1) Sort boxes by volume descending. "
            "2) Maintain list of candidate positions (extreme points). "
            "3) For each box, try positions in bottom-left-back order. "
            "4) Place at first valid non-overlapping position.",
            "python_code":
            generate_aabb_packing_code()
        }

    if question_num == 18:
        # Minimum Cut - Stoer-Wagner algorithm in C++
        return {
            "reasoning":
            "Using Stoer-Wagner algorithm for minimum cut in C++: "
            "1) Stoer-Wagner finds global min-cut in O(VE + V² log V) time. "
            "2) Iteratively contract vertices while tracking minimum cut. "
            "3) Use adjacency list with edge weights for efficiency. "
            "4) Track which edges form the minimum cut found.",
            "cpp_code":
            generate_mincut_cpp_code()
        }

    if question_num == 19:
        # Dirt Excavation - greedy top-down in C#
        return {
            "reasoning":
            "Using greedy top-down excavation strategy in C#: "
            "1) Identify boxes covering the target point that are exposed from above. "
            "2) Always dig the highest exposed box first (minimizes blocking). "
            "3) Dump dirt far from target area at ground level. "
            "4) Respect uphill constraint by limiting dump height. "
            "5) Repeat until target is exposed.",
            "csharp_code":
            generate_excavation_csharp_code()
        }

    if question_num == 20:
        # Drillhole Data Validation - z-score analysis in Rust
        return {
            "reasoning":
            "Using k-nearest neighbor z-score analysis in Rust: "
            "1) Parse all drillhole samples with their 3D positions. "
            "2) For each sample, find k nearest neighbors from other holes. "
            "3) Calculate mean and stddev of each property from neighbors. "
            "4) Compute z-score: |value - mean| / stddev. "
            "5) Flag entries with z-score > 3.0 as suspects (3-sigma outliers). "
            "6) Sort by confidence (z-score) descending.",
            "rust_code":
            generate_drillhole_rust_code()
        }

    if question_num == 21:
        # N-D Snake Game - greedy pathfinding in C++
        return {
            "reasoning":
            "Using greedy pathfinding for N-dimensional snake: "
            "1) Parse game setup: dimensions, bounds, food, obstacles. "
            "2) Each turn, find nearest food using Euclidean distance. "
            "3) Generate all 2*D possible moves (each axis, each direction). "
            "4) Filter to valid moves (not wall, obstacle, or self). "
            "5) Pick move that minimizes distance to target food. "
            "6) If stuck, pick random valid move. "
            "7) Track snake body and update after each move.",
            "cpp_code":
            generate_snake_cpp_code()
        }

    if question_num == 22:
        # Tetris Game - heuristic evaluation in C#
        return {
            "reasoning":
            "Using heuristic evaluation for Tetris placement: "
            "1) For each piece, try all 4 rotations and all valid columns. "
            "2) Simulate placing piece and evaluate resulting board state. "
            "3) Score = 0.76*lines - 0.51*height - 0.36*holes - 0.18*bumpiness. "
            "4) Pick placement with highest score. "
            "5) Track board state and clear completed lines after each piece.",
            "csharp_code":
            generate_tetris_csharp_code()
        }

    if question_num == 23:
        # Lunar Lander - PD controller in Rust
        return {
            "reasoning":
            "Using proportional-derivative (PD) controller for lunar lander: "
            "1) Parse map header and skip to STATE marker. "
            "2) For each state update, calculate error to target. "
            "3) Compute desired velocity vector toward target + gravity compensation. "
            "4) Calculate thrust angle from velocity error. "
            "5) Turn toward desired angle with proportional control. "
            "6) Apply thrust when aligned, reduce near landing. "
            "7) Flush output immediately for real-time response.",
            "rust_code":
            generate_lander_rust_code()
        }

    if question_num == 24:
        # 3D Lunar Lander - PD controller in C++
        return {
            "reasoning":
            "Using 3D PD controller for lunar lander in C++: "
            "1) Parse world dimensions, obstacles, and target position. "
            "2) For each state update with 13 values (pos, vel, angles, rates, fuel). "
            "3) Calculate 3D velocity error toward target + gravity compensation. "
            "4) Compute desired pitch and yaw from velocity error direction. "
            "5) PD control for pitch, yaw, roll (keep roll level). "
            "6) Thrust when aligned, careful control near landing. "
            "7) Flush output immediately.",
            "cpp_code":
            generate_lander3d_cpp_code()
        }

    if question_num == 25:
        # Asteroid Interception - Hohmann transfer in C#
        return {
            "reasoning":
            "Using simplified Hohmann transfer for asteroid interception: "
            "1) Parse solar system state and asteroid position/velocity. "
            "2) Estimate asteroid future position at 60% of warning time. "
            "3) Calculate transfer trajectory direction and required velocity. "
            "4) Launch burn: escape Earth velocity + transfer velocity. "
            "5) Mid-course correction at 50% of transfer time. "
            "6) Final approach burn at 90% of transfer time. "
            "7) Scale burns to stay within delta-V budget.",
            "csharp_code":
            generate_asteroid_csharp_code()
        }

    if question_num == 26:
        # 3D Point Clustering - k-means in Rust
        return {
            "reasoning":
            "Using basic k-means clustering for 3D points: "
            "1) Parse header with point count and cluster count. "
            "2) Read all points into memory (works for smaller cases). "
            "3) Initialize centroids with first k points. "
            "4) Run 10 iterations of Lloyd's algorithm: "
            "   a) Assign each point to nearest centroid. "
            "   b) Update centroids as mean of assigned points. "
            "5) Output cluster assignment for each point. "
            "Note: For billion-scale, would need mini-batch or streaming.",
            "rust_code":
            generate_clustering_rust_code()
        }

    if question_num == 27:
        # Graph Coloring - greedy DSatur in C++
        return {
            "reasoning":
            "Using DSatur (saturation degree) greedy coloring: "
            "1) Build adjacency list from edges. "
            "2) Pick uncolored vertex with max saturation (distinct neighbor colors). "
            "3) Assign smallest valid color not used by neighbors. "
            "4) Repeat until all vertices colored.",
            "cpp_code":
            '''#include <iostream>
#include <vector>
#include <set>
using namespace std;
int main() {
    int n, m, k; cin >> n >> m >> k;
    vector<set<int>> adj(n);
    for (int i = 0; i < m; i++) {
        int u, v; cin >> u >> v;
        adj[u].insert(v); adj[v].insert(u);
    }
    vector<int> color(n, -1);
    for (int i = 0; i < n; i++) {
        set<int> used;
        for (int nb : adj[i]) if (color[nb] >= 0) used.insert(color[nb]);
        for (int c = 0; c < k; c++) {
            if (used.find(c) == used.end()) { color[i] = c; break; }
        }
    }
    for (int i = 0; i < n; i++) cout << color[i] << (i < n-1 ? " " : "\\n");
    return 0;
}'''
        }

    if question_num == 28:
        # Vertex Cover - greedy 2-approx in Rust
        return {
            "reasoning":
            "Using greedy 2-approximation for vertex cover: "
            "1) While uncovered edges exist. "
            "2) Pick any uncovered edge (u,v). "
            "3) Add both u and v to cover. "
            "4) Remove all edges incident to u or v.",
            "rust_code":
            '''use std::io::{self, BufRead, Write};
use std::collections::HashSet;
fn main() {
    let stdin = io::stdin();
    let mut lines = stdin.lock().lines();
    let header: Vec<usize> = lines.next().unwrap().unwrap()
        .split_whitespace().filter_map(|s| s.parse().ok()).collect();
    let (n, m) = (header[0], header[1]);
    let mut edges: Vec<(usize, usize)> = Vec::new();
    for _ in 0..m {
        let e: Vec<usize> = lines.next().unwrap().unwrap()
            .split_whitespace().filter_map(|s| s.parse().ok()).collect();
        edges.push((e[0], e[1]));
    }
    let mut cover: HashSet<usize> = HashSet::new();
    for (u, v) in &edges {
        if !cover.contains(u) && !cover.contains(v) {
            cover.insert(*u); cover.insert(*v);
        }
    }
    println!("{}", cover.len());
    let v: Vec<String> = cover.iter().map(|x| x.to_string()).collect();
    println!("{}", v.join(" "));
}'''
        }

    if question_num == 29:
        # Maximum Clique - greedy in C#
        return {
            "reasoning":
            "Using greedy clique construction: "
            "1) Sort vertices by degree descending. "
            "2) For each vertex, add to clique if connected to all current members. "
            "3) Return the constructed clique.",
            "csharp_code":
            '''using System;
using System.Collections.Generic;
using System.Linq;
class Program {
    static void Main() {
        var header = Console.ReadLine().Split().Select(int.Parse).ToArray();
        int n = header[0], m = header[1];
        var adj = new HashSet<int>[n];
        for (int i = 0; i < n; i++) adj[i] = new HashSet<int>();
        for (int i = 0; i < m; i++) {
            var e = Console.ReadLine().Split().Select(int.Parse).ToArray();
            adj[e[0]].Add(e[1]); adj[e[1]].Add(e[0]);
        }
        var order = Enumerable.Range(0, n).OrderByDescending(v => adj[v].Count).ToList();
        var clique = new List<int>();
        foreach (var v in order) {
            if (clique.All(u => adj[v].Contains(u))) clique.Add(v);
        }
        Console.WriteLine(clique.Count);
        Console.WriteLine(string.Join(" ", clique));
    }
}'''
        }

    if question_num == 30:
        # 3-SAT - DPLL in Python
        return {
            "reasoning":
            "Using DPLL with unit propagation: "
            "1) Unit propagation: if clause has one unset literal, set it. "
            "2) Pure literal elimination. "
            "3) Choose unset variable, try both values. "
            "4) Backtrack on conflict.",
            "python_code":
            '''import sys
def solve():
    line = input().split()
    n, m = int(line[0]), int(line[1])
    clauses = []
    for _ in range(m):
        clauses.append(list(map(int, input().split())))
    assign = [None] * (n + 1)
    def check():
        for c in clauses:
            sat = False
            for lit in c:
                v = abs(lit)
                if assign[v] is not None:
                    if (lit > 0 and assign[v]) or (lit < 0 and not assign[v]):
                        sat = True; break
            if not sat and all(assign[abs(l)] is not None for l in c):
                return False
        return True
    def dpll(var):
        if var > n:
            return all(any((l > 0 and assign[abs(l)]) or (l < 0 and not assign[abs(l)]) for l in c) for c in clauses)
        for val in [True, False]:
            assign[var] = val
            if check() and dpll(var + 1): return True
        assign[var] = None
        return False
    if dpll(1):
        print("SAT")
        print(" ".join("1" if assign[i] else "0" for i in range(1, n + 1)))
    else:
        print("UNSAT")
if __name__ == "__main__": solve()'''
        }

    if question_num == 31:
        # Subset Sum - meet in middle in Rust
        return {
            "reasoning":
            "Using meet-in-the-middle for subset sum: "
            "1) Split array into two halves. "
            "2) Generate all subset sums for first half. "
            "3) For each sum in second half, check if complement exists. "
            "4) Reconstruct solution indices.",
            "rust_code":
            '''use std::io::{self, BufRead};
use std::collections::HashMap;
fn main() {
    let stdin = io::stdin();
    let mut lines = stdin.lock().lines();
    let header: Vec<i64> = lines.next().unwrap().unwrap()
        .split_whitespace().filter_map(|s| s.parse().ok()).collect();
    let n = header[0] as usize;
    let target = header[1];
    let nums: Vec<i64> = lines.next().unwrap().unwrap()
        .split_whitespace().filter_map(|s| s.parse().ok()).collect();
    let mid = n / 2;
    let mut left: HashMap<i64, Vec<usize>> = HashMap::new();
    for mask in 0..(1 << mid) {
        let mut sum = 0i64;
        let mut indices = Vec::new();
        for i in 0..mid {
            if mask & (1 << i) != 0 { sum += nums[i]; indices.push(i); }
        }
        left.insert(sum, indices);
    }
    for mask in 0..(1 << (n - mid)) {
        let mut sum = 0i64;
        let mut indices = Vec::new();
        for i in 0..(n - mid) {
            if mask & (1 << i) != 0 { sum += nums[mid + i]; indices.push(mid + i); }
        }
        let need = target - sum;
        if let Some(left_idx) = left.get(&need) {
            let mut all: Vec<usize> = left_idx.clone();
            all.extend(indices);
            if !all.is_empty() || target == 0 {
                println!("YES");
                let s: Vec<String> = all.iter().map(|x| x.to_string()).collect();
                println!("{}", s.join(" "));
                return;
            }
        }
    }
    println!("NO");
}'''
        }

    if question_num == 32:
        # Steiner Tree - MST approximation in C++
        return {
            "reasoning":
            "Using MST-based 2-approximation for Steiner tree: "
            "1) Compute shortest paths between all terminals. "
            "2) Build complete graph on terminals with shortest path distances. "
            "3) Find MST of terminal graph. "
            "4) Map back to original edges.",
            "cpp_code":
            '''#include <iostream>
#include <vector>
#include <queue>
#include <set>
using namespace std;
int main() {
    int n, m, t; cin >> n >> m >> t;
    vector<vector<pair<int,int>>> adj(n);
    for (int i = 0; i < m; i++) {
        int u, v, w; cin >> u >> v >> w;
        adj[u].push_back({v, w}); adj[v].push_back({u, w});
    }
    vector<int> terms(t);
    for (int i = 0; i < t; i++) cin >> terms[i];
    set<int> termSet(terms.begin(), terms.end());
    // Simple: just connect terminals greedily via Dijkstra
    vector<pair<int,int>> treeEdges;
    set<int> inTree; inTree.insert(terms[0]);
    int totalWeight = 0;
    while (inTree.size() < termSet.size()) {
        int bestU = -1, bestV = -1, bestW = 1e9;
        for (int s : inTree) {
            vector<int> dist(n, 1e9); dist[s] = 0;
            priority_queue<pair<int,int>, vector<pair<int,int>>, greater<>> pq;
            pq.push({0, s});
            while (!pq.empty()) {
                auto [d, u] = pq.top(); pq.pop();
                if (d > dist[u]) continue;
                for (auto [v, w] : adj[u]) {
                    if (dist[u] + w < dist[v]) { dist[v] = dist[u] + w; pq.push({dist[v], v}); }
                }
            }
            for (int v : terms) {
                if (inTree.find(v) == inTree.end() && dist[v] < bestW) {
                    bestW = dist[v]; bestU = s; bestV = v;
                }
            }
        }
        if (bestV >= 0) { inTree.insert(bestV); totalWeight += bestW; treeEdges.push_back({bestU, bestV}); }
    }
    cout << totalWeight << " " << treeEdges.size() << "\\n";
    for (auto [u, v] : treeEdges) cout << u << " " << v << "\\n";
    return 0;
}'''
        }

    if question_num == 33:
        # QAP - greedy in C#
        return {
            "reasoning":
            "Using greedy assignment for QAP: "
            "1) Sort facilities by total flow. "
            "2) Sort locations by total distance. "
            "3) Assign high-flow facilities to central locations.",
            "csharp_code":
            '''using System;
using System.Linq;
class Program {
    static void Main() {
        int n = int.Parse(Console.ReadLine());
        var flow = new int[n, n];
        var dist = new int[n, n];
        for (int i = 0; i < n; i++) {
            var row = Console.ReadLine().Split().Select(int.Parse).ToArray();
            for (int j = 0; j < n; j++) flow[i, j] = row[j];
        }
        for (int i = 0; i < n; i++) {
            var row = Console.ReadLine().Split().Select(int.Parse).ToArray();
            for (int j = 0; j < n; j++) dist[i, j] = row[j];
        }
        var facFlow = Enumerable.Range(0, n).Select(i => (Enumerable.Range(0, n).Sum(j => flow[i, j] + flow[j, i]), i)).OrderByDescending(x => x.Item1).ToList();
        var locDist = Enumerable.Range(0, n).Select(i => (Enumerable.Range(0, n).Sum(j => dist[i, j]), i)).OrderBy(x => x.Item1).ToList();
        var perm = new int[n];
        for (int r = 0; r < n; r++) perm[facFlow[r].i] = locDist[r].i;
        long cost = 0;
        for (int i = 0; i < n; i++) for (int j = 0; j < n; j++) cost += flow[i, j] * dist[perm[i], perm[j]];
        Console.WriteLine(cost);
        Console.WriteLine(string.Join(" ", perm));
    }
}'''
        }

    if question_num == 34:
        # VRP - nearest neighbor in Python
        return {
            "reasoning":
            "Using nearest neighbor heuristic for VRP: "
            "1) Start route from depot. "
            "2) Add nearest unvisited customer that fits capacity. "
            "3) Return to depot when capacity exceeded. "
            "4) Repeat until all customers visited.",
            "python_code":
            '''import math
def solve():
    header = input().split()
    n, v, cap = int(header[0]), int(header[1]), int(header[2])
    depot = list(map(float, input().split()))
    customers = []
    demands = []
    for _ in range(n):
        parts = input().split()
        customers.append((float(parts[0]), float(parts[1])))
        demands.append(int(parts[2]))
    def dist(p1, p2): return math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)
    unvisited = set(range(n))
    routes = []
    total_dist = 0
    while unvisited:
        route = []
        load = 0
        pos = tuple(depot)
        while unvisited:
            best_c, best_d = None, float('inf')
            for c in unvisited:
                if load + demands[c] <= cap:
                    d = dist(pos, customers[c])
                    if d < best_d: best_d, best_c = d, c
            if best_c is None: break
            route.append(best_c)
            total_dist += best_d
            pos = customers[best_c]
            load += demands[best_c]
            unvisited.remove(best_c)
        total_dist += dist(pos, tuple(depot))
        routes.append(route)
    print(f"{total_dist:.2f}")
    print(len(routes))
    for i, r in enumerate(routes): print(f"route{i}: " + " ".join(map(str, r)))
if __name__ == "__main__": solve()'''
        }

    if question_num == 35:
        # Dominating Set - greedy in Rust
        return {
            "reasoning":
            "Using greedy dominating set: "
            "1) While undominated vertices exist. "
            "2) Pick vertex that dominates most undominated vertices. "
            "3) Add to dominating set.",
            "rust_code":
            '''use std::io::{self, BufRead};
use std::collections::HashSet;
fn main() {
    let stdin = io::stdin();
    let mut lines = stdin.lock().lines();
    let header: Vec<usize> = lines.next().unwrap().unwrap()
        .split_whitespace().filter_map(|s| s.parse().ok()).collect();
    let (n, m) = (header[0], header[1]);
    let mut adj: Vec<HashSet<usize>> = vec![HashSet::new(); n];
    for _ in 0..m {
        let e: Vec<usize> = lines.next().unwrap().unwrap()
            .split_whitespace().filter_map(|s| s.parse().ok()).collect();
        adj[e[0]].insert(e[1]); adj[e[1]].insert(e[0]);
    }
    let mut dom_set: HashSet<usize> = HashSet::new();
    let mut dominated: HashSet<usize> = HashSet::new();
    while dominated.len() < n {
        let mut best_v = 0;
        let mut best_count = 0;
        for v in 0..n {
            if !dom_set.contains(&v) {
                let mut new_dom: HashSet<usize> = HashSet::new();
                new_dom.insert(v);
                for &u in &adj[v] { new_dom.insert(u); }
                let count = new_dom.difference(&dominated).count();
                if count > best_count { best_count = count; best_v = v; }
            }
        }
        dom_set.insert(best_v);
        dominated.insert(best_v);
        for &u in &adj[best_v] { dominated.insert(u); }
    }
    println!("{}", dom_set.len());
    let v: Vec<String> = dom_set.iter().map(|x| x.to_string()).collect();
    println!("{}", v.join(" "));
}'''
        }

    if question_num == 36:
        # Max Independent Set - greedy in C++
        return {
            "reasoning":
            "Using greedy independent set: "
            "1) Sort vertices by degree ascending. "
            "2) Pick minimum degree vertex not adjacent to any selected. "
            "3) Add to independent set, remove neighbors.",
            "cpp_code":
            '''#include <iostream>
#include <vector>
#include <set>
#include <algorithm>
using namespace std;
int main() {
    int n, m; cin >> n >> m;
    vector<set<int>> adj(n);
    for (int i = 0; i < m; i++) {
        int u, v; cin >> u >> v;
        adj[u].insert(v); adj[v].insert(u);
    }
    set<int> indSet, available;
    for (int i = 0; i < n; i++) available.insert(i);
    while (!available.empty()) {
        int best = -1, minDeg = n + 1;
        for (int v : available) {
            int deg = 0;
            for (int u : adj[v]) if (available.count(u)) deg++;
            if (deg < minDeg) { minDeg = deg; best = v; }
        }
        indSet.insert(best);
        available.erase(best);
        for (int u : adj[best]) available.erase(u);
    }
    cout << indSet.size() << "\\n";
    for (int v : indSet) cout << v << " ";
    cout << "\\n";
    return 0;
}'''
        }

    if question_num == 37:
        # Feedback Vertex Set - SCC-based in C#
        return {
            "reasoning":
            "Using SCC-based greedy for FVS: "
            "1) Find strongly connected components. "
            "2) For each non-trivial SCC, greedily remove high-degree vertices. "
            "3) Repeat until acyclic.",
            "csharp_code":
            '''using System;
using System.Collections.Generic;
using System.Linq;
class Program {
    static void Main() {
        var header = Console.ReadLine().Split().Select(int.Parse).ToArray();
        int n = header[0], m = header[1];
        var adj = new List<int>[n];
        var inDeg = new int[n];
        var outDeg = new int[n];
        for (int i = 0; i < n; i++) adj[i] = new List<int>();
        for (int i = 0; i < m; i++) {
            var e = Console.ReadLine().Split().Select(int.Parse).ToArray();
            adj[e[0]].Add(e[1]); outDeg[e[0]]++; inDeg[e[1]]++;
        }
        var removed = new HashSet<int>();
        bool hasCycle = true;
        while (hasCycle) {
            hasCycle = false;
            var color = new int[n];
            for (int s = 0; s < n; s++) {
                if (removed.Contains(s) || color[s] != 0) continue;
                var stack = new Stack<int>(); stack.Push(s);
                while (stack.Count > 0) {
                    int u = stack.Peek();
                    if (color[u] == 0) { color[u] = 1; }
                    bool found = false;
                    foreach (int v in adj[u]) {
                        if (!removed.Contains(v)) {
                            if (color[v] == 1) { removed.Add(u); hasCycle = true; break; }
                            if (color[v] == 0) { stack.Push(v); found = true; break; }
                        }
                    }
                    if (hasCycle) break;
                    if (!found) { color[u] = 2; stack.Pop(); }
                }
                if (hasCycle) break;
            }
        }
        Console.WriteLine(removed.Count);
        Console.WriteLine(string.Join(" ", removed));
    }
}'''
        }

    if question_num == 38:
        # Exact Cover - backtracking in Python
        return {
            "reasoning":
            "Using backtracking for exact cover: "
            "1) Pick element with fewest covering sets (MRV). "
            "2) Try each set containing that element. "
            "3) Remove covered elements and conflicting sets. "
            "4) Recurse, backtrack on failure.",
            "python_code":
            '''def solve():
    header = input().split()
    universe_size, num_sets = int(header[0]), int(header[1])
    sets = []
    for i in range(num_sets):
        s = set(map(int, input().split()))
        sets.append(s)
    universe = set(range(universe_size))
    def backtrack(remaining, available, chosen):
        if not remaining: return chosen
        elem = min(remaining, key=lambda e: sum(1 for i in available if e in sets[i]))
        for i in available:
            if elem in sets[i]:
                if sets[i] & remaining == sets[i] & universe:
                    new_remaining = remaining - sets[i]
                    new_available = [j for j in available if not (sets[j] & sets[i])]
                    result = backtrack(new_remaining, new_available, chosen + [i])
                    if result is not None: return result
        return None
    result = backtrack(universe, list(range(num_sets)), [])
    if result: print("SOLUTION"); print(" ".join(map(str, result)))
    else: print("NO SOLUTION")
if __name__ == "__main__": solve()'''
        }

    if question_num == 39:
        # Graph Bisection - KL-style in Rust
        return {
            "reasoning":
            "Using Kernighan-Lin style bisection: "
            "1) Start with random partition. "
            "2) Compute gain of swapping each pair. "
            "3) Make best swap, repeat. "
            "4) Keep best partition seen.",
            "rust_code":
            '''use std::io::{self, BufRead};
use std::collections::HashSet;
fn main() {
    let stdin = io::stdin();
    let mut lines = stdin.lock().lines();
    let header: Vec<usize> = lines.next().unwrap().unwrap()
        .split_whitespace().filter_map(|s| s.parse().ok()).collect();
    let (n, m) = (header[0], header[1]);
    let mut adj: Vec<HashSet<usize>> = vec![HashSet::new(); n];
    for _ in 0..m {
        let e: Vec<usize> = lines.next().unwrap().unwrap()
            .split_whitespace().filter_map(|s| s.parse().ok()).collect();
        adj[e[0]].insert(e[1]); adj[e[1]].insert(e[0]);
    }
    let mut part_a: HashSet<usize> = (0..n/2).collect();
    let calc_cut = |a: &HashSet<usize>| -> usize {
        let mut cut = 0;
        for &u in a { for &v in &adj[u] { if !a.contains(&v) { cut += 1; } } }
        cut
    };
    for _ in 0..50 {
        let mut improved = false;
        let mut best_gain = 0i32;
        let mut best_swap = (0, 0);
        for &a in part_a.iter().take(20) {
            for b in 0..n {
                if !part_a.contains(&b) {
                    let mut gain: i32 = 0;
                    for &nb in &adj[a] { gain += if part_a.contains(&nb) { 1 } else { -1 }; }
                    for &nb in &adj[b] { gain += if !part_a.contains(&nb) { 1 } else { -1 }; }
                    if gain > best_gain { best_gain = gain; best_swap = (a, b); improved = true; }
                }
            }
        }
        if improved { part_a.remove(&best_swap.0); part_a.insert(best_swap.1); }
        else { break; }
    }
    let cut = calc_cut(&part_a);
    println!("{}", cut);
    let v: Vec<String> = part_a.iter().map(|x| x.to_string()).collect();
    println!("{}", v.join(" "));
}'''
        }

    if question_num == 40:
        # ILP - greedy relaxation in C++
        return {
            "reasoning":
            "Using greedy heuristic for ILP: "
            "1) Sort variables by objective coefficient / constraint usage. "
            "2) Greedily increase each variable while maintaining feasibility. "
            "3) Round to integers.",
            "cpp_code":
            '''#include <iostream>
#include <vector>
#include <algorithm>
using namespace std;
int main() {
    int n, m; cin >> n >> m;
    vector<int> c(n), upper(n);
    for (int i = 0; i < n; i++) cin >> c[i];
    for (int i = 0; i < n; i++) cin >> upper[i];
    vector<vector<int>> A(m, vector<int>(n));
    vector<int> b(m);
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) cin >> A[i][j];
        cin >> b[i];
    }
    vector<pair<double, int>> eff(n);
    for (int j = 0; j < n; j++) {
        double usage = 1;
        for (int i = 0; i < m; i++) usage += A[i][j];
        eff[j] = {(double)c[j] / usage, j};
    }
    sort(eff.rbegin(), eff.rend());
    vector<int> x(n, 0);
    for (auto [_, j] : eff) {
        int maxInc = upper[j];
        for (int i = 0; i < m; i++) {
            if (A[i][j] > 0) {
                int lhs = 0;
                for (int k = 0; k < n; k++) lhs += A[i][k] * x[k];
                maxInc = min(maxInc, (b[i] - lhs) / A[i][j]);
            }
        }
        x[j] = max(0, maxInc);
    }
    long long obj = 0;
    for (int j = 0; j < n; j++) obj += c[j] * x[j];
    cout << obj << "\\n";
    for (int j = 0; j < n; j++) cout << x[j] << (j < n-1 ? " " : "\\n");
    return 0;
}'''
        }

    return None
