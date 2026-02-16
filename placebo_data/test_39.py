"""Test 39: Graph Bisection - Placebo responses for all control types."""


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
  reasoning = 'Using Kernighan-Lin style bisection: 1) Start with random partition. 2) Compute gain of swapping each pair. 3) Make best swap, repeat. 4) Keep best partition seen.'
  code = 'use std::io::{self, BufRead};\nuse std::collections::HashSet;\nfn main() {\n    let stdin = io::stdin();\n    let mut lines = stdin.lock().lines();\n    let header: Vec<usize> = lines.next().unwrap().unwrap()\n        .split_whitespace().filter_map(|s| s.parse().ok()).collect();\n    let (n, m) = (header[0], header[1]);\n    let mut adj: Vec<HashSet<usize>> = vec![HashSet::new(); n];\n    for _ in 0..m {\n        let e: Vec<usize> = lines.next().unwrap().unwrap()\n            .split_whitespace().filter_map(|s| s.parse().ok()).collect();\n        adj[e[0]].insert(e[1]); adj[e[1]].insert(e[0]);\n    }\n    let mut part_a: HashSet<usize> = (0..n/2).collect();\n    let calc_cut = |a: &HashSet<usize>| -> usize {\n        let mut cut = 0;\n        for &u in a { for &v in &adj[u] { if !a.contains(&v) { cut += 1; } } }\n        cut\n    };\n    for _ in 0..50 {\n        let mut improved = false;\n        let mut best_gain = 0i32;\n        let mut best_swap = (0, 0);\n        for &a in part_a.iter().take(20) {\n            for b in 0..n {\n                if !part_a.contains(&b) {\n                    let mut gain: i32 = 0;\n                    for &nb in &adj[a] { gain += if part_a.contains(&nb) { 1 } else { -1 }; }\n                    for &nb in &adj[b] { gain += if !part_a.contains(&nb) { 1 } else { -1 }; }\n                    if gain > best_gain { best_gain = gain; best_swap = (a, b); improved = true; }\n                }\n            }\n        }\n        if improved { part_a.remove(&best_swap.0); part_a.insert(best_swap.1); }\n        else { break; }\n    }\n    let cut = calc_cut(&part_a);\n    println!("{}", cut);\n    let v: Vec<String> = part_a.iter().map(|x| x.to_string()).collect();\n    println!("{}", v.join(" "));\n}'
  return {"reasoning": reasoning, 'rust_code': code}, reasoning


def _naive_optimised(subpass):
  # TODO: Same algorithm as naive but hyper-optimised for Rust
  # (parallel, SIMD, small data types, register packing, etc.)
  return _naive(subpass)


def _best_published(subpass):
  reasoning = (
    "Best published: Kernighan-Lin bisection (Kernighan & Lin 1970, Bell System Tech J. 49(2):291-307). "
    "TODO: Full implementation pending."
  )
  code = '// TODO: Implement Kernighan-Lin bisection'
  return {"reasoning": reasoning, 'rust_code': code}, reasoning


def _random(subpass):
  reasoning = 'Random: pseudorandom output with fixed seed 42.'
  code = 'use std::io::{self, BufRead};\nfn main() {\n    let stdin = io::stdin();\n    let _h = stdin.lock().lines().next();\n    println!("0");\n}'
  return {"reasoning": reasoning, 'rust_code': code}, reasoning


def _human(subpass):
  reasoning = 'Human starting point for Graph Bisection. Fill in the TODOs.'
  code = '// TODO: Human attempt at Graph Bisection\nuse std::io::{self, BufRead, Write};\nfn main() {\n    let stdin = io::stdin();\n    let mut lines = stdin.lock().lines();\n    // TODO: Parse input\n    // TODO: Implement solution\n    // TODO: Output result\n}'
  return {"reasoning": reasoning, 'rust_code': code}, reasoning
