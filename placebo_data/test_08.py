"""Test 8: Maze Solver - Placebo responses for all control types."""


def get_response(model_name, subpass):
  """Return (result_dict, reasoning_string) for the given control type."""
  if model_name == 'naive':
    return _naive(subpass)
  elif model_name == 'naive-optimised':
    return _naive_optimised(subpass)
  elif model_name == 'best-published':
    return _best_published(subpass)
  elif model_name == 'random':
    return _random(subpass)
  elif model_name == 'human':
    return _human(subpass)
  return None, ''


def _naive(subpass):
  reasoning = 'BFS maze solver in Rust. Finds shortest path from A to B. Simple and correct but no optimisation.'
  code = 'use std::io::{self, Read, Write};\nuse std::collections::VecDeque;\nfn main() {\n    let mut input = String::new();\n    io::stdin().read_to_string(&mut input).unwrap();\n    let lines: Vec<&str> = input.lines().collect();\n    let h = lines.len();\n    let mut start = (0,0); let mut end = (0,0);\n    for (y, line) in lines.iter().enumerate() {\n        for (x, ch) in line.chars().enumerate() {\n            if ch == \'A\' { start = (x,y); } else if ch == \'B\' { end = (x,y); }\n        }\n    }\n    let mut vis = vec![vec![false; 200]; h];\n    let mut par: Vec<Vec<(i32,i32)>> = vec![vec![(-1,-1); 200]; h];\n    vis[start.1][start.0] = true;\n    let mut q = VecDeque::new(); q.push_back(start);\n    while let Some((x,y)) = q.pop_front() {\n        if (x,y) == end { break; }\n        for (dx,dy) in [(1i32,0),(0,1),(-1,0),(0,-1)] {\n            let (nx,ny) = (x as i32+dx, y as i32+dy);\n            if nx>=0 && ny>=0 && (ny as usize)<h && (nx as usize)<lines[ny as usize].len() {\n                let (ux,uy) = (nx as usize, ny as usize);\n                let ch = lines[uy].as_bytes()[ux];\n                if !vis[uy][ux] && ch != b\'#\' { vis[uy][ux]=true; par[uy][ux]=(x as i32,y as i32); q.push_back((ux,uy)); }\n            }\n        }\n    }\n    let mut path = vec![end]; let mut c = end;\n    while c != start { let p = par[c.1][c.0]; c = (p.0 as usize, p.1 as usize); path.push(c); }\n    path.reverse();\n    let stdout = io::stdout(); let mut out = stdout.lock();\n    for (x,y) in &path { writeln!(out, "{} {}", x, y).unwrap(); }\n}'
  return {"reasoning": reasoning, 'rust_code': code}, reasoning


def _naive_optimised(subpass):
  # TODO: Same algorithm as naive but hyper-optimised for Rust
  # (parallel, SIMD, small data types, register packing, etc.)
  return _naive(subpass)


def _best_published(subpass):
  reasoning = (
    "Best published: A* with Manhattan distance (Hart et al. 1968, IEEE Trans SSC 4(2):100-107). "
    "TODO: Full implementation pending."
  )
  code = '// TODO: Implement A* with Manhattan distance'
  return {"reasoning": reasoning, 'rust_code': code}, reasoning


def _random(subpass):
  reasoning = 'Random: pseudorandom output with fixed seed 42.'
  code = 'use std::io::{self, BufRead};\nfn main() {\n    let stdin = io::stdin();\n    let _h = stdin.lock().lines().next();\n    println!("0");\n}'
  return {"reasoning": reasoning, 'rust_code': code}, reasoning


def _human(subpass):
  reasoning = 'Human starting point for Maze Solver. Fill in the TODOs.'
  code = '// TODO: Human attempt at Maze Solver\nuse std::io::{self, BufRead, Write};\nfn main() {\n    let stdin = io::stdin();\n    let mut lines = stdin.lock().lines();\n    // TODO: Parse input\n    // TODO: Implement solution\n    // TODO: Output result\n}'
  return {"reasoning": reasoning, 'rust_code': code}, reasoning
