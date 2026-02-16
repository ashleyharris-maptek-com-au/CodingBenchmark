"""Test 28: Vertex Cover - Placebo responses for all control types."""


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
  reasoning = 'Using greedy 2-approximation for vertex cover: 1) While uncovered edges exist. 2) Pick any uncovered edge (u,v). 3) Add both u and v to cover. 4) Remove all edges incident to u or v.'
  code = 'use std::io::{self, BufRead, Write};\nuse std::collections::HashSet;\nfn main() {\n    let stdin = io::stdin();\n    let mut lines = stdin.lock().lines();\n    let header: Vec<usize> = lines.next().unwrap().unwrap()\n        .split_whitespace().filter_map(|s| s.parse().ok()).collect();\n    let (n, m) = (header[0], header[1]);\n    let mut edges: Vec<(usize, usize)> = Vec::new();\n    for _ in 0..m {\n        let e: Vec<usize> = lines.next().unwrap().unwrap()\n            .split_whitespace().filter_map(|s| s.parse().ok()).collect();\n        edges.push((e[0], e[1]));\n    }\n    let mut cover: HashSet<usize> = HashSet::new();\n    for (u, v) in &edges {\n        if !cover.contains(u) && !cover.contains(v) {\n            cover.insert(*u); cover.insert(*v);\n        }\n    }\n    println!("{}", cover.len());\n    let v: Vec<String> = cover.iter().map(|x| x.to_string()).collect();\n    println!("{}", v.join(" "));\n}'
  return {"reasoning": reasoning, 'rust_code': code}, reasoning


def _naive_optimised(subpass):
  # TODO: Same algorithm as naive but hyper-optimised for Rust
  # (parallel, SIMD, small data types, register packing, etc.)
  return _naive(subpass)


def _best_published(subpass):
  reasoning = (
    "Best published: Nemhauser-Trotter kernelization (Nemhauser & Trotter 1975, Mathematical Programming 8(1):232-248). "
    "TODO: Full implementation pending."
  )
  code = '// TODO: Implement Nemhauser-Trotter kernelization'
  return {"reasoning": reasoning, 'rust_code': code}, reasoning


def _random(subpass):
  reasoning = 'Random: pseudorandom output with fixed seed 42.'
  code = 'use std::io::{self, BufRead};\nfn main() {\n    let stdin = io::stdin();\n    let _h = stdin.lock().lines().next();\n    println!("0");\n}'
  return {"reasoning": reasoning, 'rust_code': code}, reasoning


def _human(subpass):
  reasoning = 'Human starting point for Vertex Cover. Fill in the TODOs.'
  code = '// TODO: Human attempt at Vertex Cover\nuse std::io::{self, BufRead, Write};\nfn main() {\n    let stdin = io::stdin();\n    let mut lines = stdin.lock().lines();\n    // TODO: Parse input\n    // TODO: Implement solution\n    // TODO: Output result\n}'
  return {"reasoning": reasoning, 'rust_code': code}, reasoning
