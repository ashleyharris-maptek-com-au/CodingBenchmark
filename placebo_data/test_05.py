"""Test 5: Hamiltonian Path - Placebo responses for all control types."""


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
  reasoning = 'DFS backtracking Hamiltonian path in Rust. Exponential worst case. Times out on large grids.'
  code = 'use std::io::{self, BufRead, Write};\nuse std::collections::HashSet;\nfn main() {\n    let stdin = io::stdin();\n    let mut lines = stdin.lock().lines();\n    let h: Vec<usize> = lines.next().unwrap().unwrap()\n        .split_whitespace().filter_map(|s| s.parse().ok()).collect();\n    let (w, he) = (h[0], h[1]);\n    let mut obs = HashSet::new();\n    let no: usize = lines.next().unwrap().unwrap().trim().parse().unwrap();\n    for _ in 0..no {\n        let o: Vec<usize> = lines.next().unwrap().unwrap()\n            .split_whitespace().filter_map(|s| s.parse().ok()).collect();\n        obs.insert((o[0], o[1]));\n    }\n    let total = w * he - obs.len();\n    let mut path = vec![(0usize, 0usize)];\n    let mut visited = HashSet::new(); visited.insert((0,0));\n    fn dfs(path: &mut Vec<(usize,usize)>, vis: &mut HashSet<(usize,usize)>,\n           w: usize, h: usize, obs: &HashSet<(usize,usize)>, total: usize) -> bool {\n        if path.len() == total { return true; }\n        let (x,y) = *path.last().unwrap();\n        for (dx,dy) in [(1i32,0),(0,1),(-1,0),(0,-1)] {\n            let (nx,ny) = (x as i32+dx, y as i32+dy);\n            if nx>=0 && ny>=0 && (nx as usize)<w && (ny as usize)<h {\n                let p = (nx as usize, ny as usize);\n                if !obs.contains(&p) && !vis.contains(&p) {\n                    vis.insert(p); path.push(p);\n                    if dfs(path, vis, w, h, obs, total) { return true; }\n                    path.pop(); vis.remove(&p);\n                }\n            }\n        }\n        false\n    }\n    let stdout = io::stdout(); let mut out = stdout.lock();\n    if dfs(&mut path, &mut visited, w, he, &obs, total) {\n        for (x,y) in &path { writeln!(out, "{} {}", x, y).unwrap(); }\n    }\n}'
  return {"reasoning": reasoning, 'rust_code': code}, reasoning


def _naive_optimised(subpass):
  # TODO: Same algorithm as naive but hyper-optimised for Rust
  # (parallel, SIMD, small data types, register packing, etc.)
  return _naive(subpass)


def _best_published(subpass):
  reasoning = (
    "Best published: Warnsdorff's rule with backtracking (Warnsdorff 1823). "
    "TODO: Full implementation pending."
  )
  code = "// TODO: Implement Warnsdorff's rule with backtracking"
  return {"reasoning": reasoning, 'rust_code': code}, reasoning


def _random(subpass):
  reasoning = 'Random: pseudorandom output with fixed seed 42.'
  code = 'use std::io::{self, BufRead};\nfn main() {\n    let stdin = io::stdin();\n    let _h = stdin.lock().lines().next();\n    println!("0");\n}'
  return {"reasoning": reasoning, 'rust_code': code}, reasoning


def _human(subpass):
  reasoning = 'Human starting point for Hamiltonian Path. Fill in the TODOs.'
  code = '// TODO: Human attempt at Hamiltonian Path\nuse std::io::{self, BufRead, Write};\nfn main() {\n    let stdin = io::stdin();\n    let mut lines = stdin.lock().lines();\n    // TODO: Parse input\n    // TODO: Implement solution\n    // TODO: Output result\n}'
  return {"reasoning": reasoning, 'rust_code': code}, reasoning
