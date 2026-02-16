"""Test 3: Graph Layout - Placebo responses for all control types."""


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
  reasoning = 'Simple force-directed layout in Rust. O(n^2) per iteration, no Barnes-Hut. Times out on large graphs.'
  code = 'use std::io::{self, BufRead, Write};\nfn main() {\n    let stdin = io::stdin();\n    let mut lines = stdin.lock().lines();\n    let h: Vec<usize> = lines.next().unwrap().unwrap()\n        .split_whitespace().filter_map(|s| s.parse().ok()).collect();\n    let (n, m) = (h[0], h[1]);\n    let mut adj = vec![vec![]; n];\n    for _ in 0..m {\n        let e: Vec<usize> = lines.next().unwrap().unwrap()\n            .split_whitespace().filter_map(|s| s.parse().ok()).collect();\n        adj[e[0]].push(e[1]); adj[e[1]].push(e[0]);\n    }\n    let mut px: Vec<f64> = (0..n).map(|i| (i as f64 * 6.28 / n as f64).cos() * 10.0).collect();\n    let mut py: Vec<f64> = (0..n).map(|i| (i as f64 * 6.28 / n as f64).sin() * 10.0).collect();\n    let k = (100.0 / n as f64).sqrt();\n    for _ in 0..100 {\n        let mut fx = vec![0.0f64; n]; let mut fy = vec![0.0f64; n];\n        for i in 0..n { for j in (i+1)..n {\n            let dx = px[i]-px[j]; let dy = py[i]-py[j];\n            let d = (dx*dx+dy*dy).sqrt().max(0.01);\n            let f = k*k/d; fx[i]+=dx/d*f; fy[i]+=dy/d*f; fx[j]-=dx/d*f; fy[j]-=dy/d*f;\n        }}\n        for i in 0..n { for &j in &adj[i] {\n            let dx = px[i]-px[j]; let dy = py[i]-py[j];\n            let d = (dx*dx+dy*dy).sqrt().max(0.01);\n            let f = d*d/k; fx[i]-=dx/d*f; fy[i]-=dy/d*f;\n        }}\n        for i in 0..n { px[i]+=fx[i].max(-5.0).min(5.0); py[i]+=fy[i].max(-5.0).min(5.0); }\n    }\n    let stdout = io::stdout(); let mut out = stdout.lock();\n    for i in 0..n { writeln!(out, "{:.4} {:.4}", px[i], py[i]).unwrap(); }\n}'
  return {"reasoning": reasoning, 'rust_code': code}, reasoning


def _naive_optimised(subpass):
  # TODO: Same algorithm as naive but hyper-optimised for Rust
  # (parallel, SIMD, small data types, register packing, etc.)
  return _naive(subpass)


def _best_published(subpass):
  reasoning = (
    "Best published: Stress majorization layout (Gansner et al. 2005, GD 2004 LNCS 3383:239-250). "
    "TODO: Full implementation pending."
  )
  code = '// TODO: Implement Stress majorization layout'
  return {"reasoning": reasoning, 'rust_code': code}, reasoning


def _random(subpass):
  reasoning = 'Random: pseudorandom output with fixed seed 42.'
  code = 'use std::io::{self, BufRead};\nfn main() {\n    let stdin = io::stdin();\n    let _h = stdin.lock().lines().next();\n    println!("0");\n}'
  return {"reasoning": reasoning, 'rust_code': code}, reasoning


def _human(subpass):
  reasoning = 'Human starting point for Graph Layout. Fill in the TODOs.'
  code = '// TODO: Human attempt at Graph Layout\nuse std::io::{self, BufRead, Write};\nfn main() {\n    let stdin = io::stdin();\n    let mut lines = stdin.lock().lines();\n    // TODO: Parse input\n    // TODO: Implement solution\n    // TODO: Output result\n}'
  return {"reasoning": reasoning, 'rust_code': code}, reasoning
