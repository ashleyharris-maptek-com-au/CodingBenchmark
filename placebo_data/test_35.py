"""Test 35: Dominating Set - Placebo responses for all control types."""


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
  reasoning = 'Using greedy dominating set: 1) While undominated vertices exist. 2) Pick vertex that dominates most undominated vertices. 3) Add to dominating set.'
  code = 'use std::io::{self, BufRead};\nuse std::collections::HashSet;\nfn main() {\n    let stdin = io::stdin();\n    let mut lines = stdin.lock().lines();\n    let header: Vec<usize> = lines.next().unwrap().unwrap()\n        .split_whitespace().filter_map(|s| s.parse().ok()).collect();\n    let (n, m) = (header[0], header[1]);\n    let mut adj: Vec<HashSet<usize>> = vec![HashSet::new(); n];\n    for _ in 0..m {\n        let e: Vec<usize> = lines.next().unwrap().unwrap()\n            .split_whitespace().filter_map(|s| s.parse().ok()).collect();\n        adj[e[0]].insert(e[1]); adj[e[1]].insert(e[0]);\n    }\n    let mut dom_set: HashSet<usize> = HashSet::new();\n    let mut dominated: HashSet<usize> = HashSet::new();\n    while dominated.len() < n {\n        let mut best_v = 0;\n        let mut best_count = 0;\n        for v in 0..n {\n            if !dom_set.contains(&v) {\n                let mut new_dom: HashSet<usize> = HashSet::new();\n                new_dom.insert(v);\n                for &u in &adj[v] { new_dom.insert(u); }\n                let count = new_dom.difference(&dominated).count();\n                if count > best_count { best_count = count; best_v = v; }\n            }\n        }\n        dom_set.insert(best_v);\n        dominated.insert(best_v);\n        for &u in &adj[best_v] { dominated.insert(u); }\n    }\n    println!("{}", dom_set.len());\n    let v: Vec<String> = dom_set.iter().map(|x| x.to_string()).collect();\n    println!("{}", v.join(" "));\n}'
  return {"reasoning": reasoning, 'rust_code': code}, reasoning


def _naive_optimised(subpass):
  # TODO: Same algorithm as naive but hyper-optimised for Rust
  # (parallel, SIMD, small data types, register packing, etc.)
  return _naive(subpass)


def _best_published(subpass):
  reasoning = (
    "Best published: Greedy + local search domination (Grandoni 2006, J. Discrete Algorithms 4(2):209-214). "
    "TODO: Full implementation pending."
  )
  code = '// TODO: Implement Greedy + local search domination'
  return {"reasoning": reasoning, 'rust_code': code}, reasoning


def _random(subpass):
  reasoning = 'Random: pseudorandom output with fixed seed 42.'
  code = 'use std::io::{self, BufRead};\nfn main() {\n    let stdin = io::stdin();\n    let _h = stdin.lock().lines().next();\n    println!("0");\n}'
  return {"reasoning": reasoning, 'rust_code': code}, reasoning


def _human(subpass):
  reasoning = 'Human starting point for Dominating Set. Fill in the TODOs.'
  code = '// TODO: Human attempt at Dominating Set\nuse std::io::{self, BufRead, Write};\nfn main() {\n    let stdin = io::stdin();\n    let mut lines = stdin.lock().lines();\n    // TODO: Parse input\n    // TODO: Implement solution\n    // TODO: Output result\n}'
  return {"reasoning": reasoning, 'rust_code': code}, reasoning
