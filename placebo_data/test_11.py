"""Test 11: Polygon Cutting - Placebo responses for all control types."""


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
  reasoning = 'Greedy first-fit polygon nesting in Rust. Grid search with 4 rotations. Very slow.'
  code = 'use std::io::{self, BufRead, Write};\nfn main() {\n    let stdin = io::stdin();\n    let mut lines = stdin.lock().lines();\n    let h: Vec<usize> = lines.next().unwrap().unwrap()\n        .split_whitespace().filter_map(|s| s.parse().ok()).collect();\n    let n_pieces = h[0];\n    // Read stock polygon (simplified: bounding box only)\n    let sverts: usize = lines.next().unwrap().unwrap().trim().parse().unwrap();\n    let mut sx = vec![0.0f64; sverts]; let mut sy = vec![0.0f64; sverts];\n    for i in 0..sverts {\n        let p: Vec<f64> = lines.next().unwrap().unwrap()\n            .split_whitespace().filter_map(|s| s.parse().ok()).collect();\n        sx[i] = p[0]; sy[i] = p[1];\n    }\n    let sw = sx.iter().cloned().fold(f64::NEG_INFINITY, f64::max) - sx.iter().cloned().fold(f64::INFINITY, f64::min);\n    let sh = sy.iter().cloned().fold(f64::NEG_INFINITY, f64::max) - sy.iter().cloned().fold(f64::INFINITY, f64::min);\n    // Greedy: just place each piece sequentially, new stock when full\n    let mut stocks = 1; let mut used_x = 0.0f64;\n    let stdout = io::stdout(); let mut out = stdout.lock();\n    writeln!(out, "{}", n_pieces).unwrap();\n    for _i in 0..n_pieces {\n        let pv: usize = lines.next().unwrap().unwrap().trim().parse().unwrap();\n        let mut px = vec![0.0f64; pv]; let mut py = vec![0.0f64; pv];\n        for j in 0..pv {\n            let p: Vec<f64> = lines.next().unwrap().unwrap()\n                .split_whitespace().filter_map(|s| s.parse().ok()).collect();\n            px[j] = p[0]; py[j] = p[1];\n        }\n        let pw = px.iter().cloned().fold(f64::NEG_INFINITY, f64::max) - px.iter().cloned().fold(f64::INFINITY, f64::min);\n        if used_x + pw > sw { stocks += 1; used_x = 0.0; }\n        writeln!(out, "{} {:.2} 0.0 0", stocks-1, used_x).unwrap();\n        used_x += pw + 1.0;\n    }\n}'
  return {"reasoning": reasoning, 'rust_code': code}, reasoning


def _naive_optimised(subpass):
  # TODO: Same algorithm as naive but hyper-optimised for Rust
  # (parallel, SIMD, small data types, register packing, etc.)
  return _naive(subpass)


def _best_published(subpass):
  reasoning = (
    "Best published: No-fit polygon nesting (Burke et al. 2007, European J. OR 179(1):27-49). "
    "TODO: Full implementation pending."
  )
  code = '// TODO: Implement No-fit polygon nesting'
  return {"reasoning": reasoning, 'rust_code': code}, reasoning


def _random(subpass):
  reasoning = 'Random: pseudorandom output with fixed seed 42.'
  code = 'use std::io::{self, BufRead};\nfn main() {\n    let stdin = io::stdin();\n    let _h = stdin.lock().lines().next();\n    println!("0");\n}'
  return {"reasoning": reasoning, 'rust_code': code}, reasoning


def _human(subpass):
  reasoning = 'Human starting point for Polygon Cutting. Fill in the TODOs.'
  code = '// TODO: Human attempt at Polygon Cutting\nuse std::io::{self, BufRead, Write};\nfn main() {\n    let stdin = io::stdin();\n    let mut lines = stdin.lock().lines();\n    // TODO: Parse input\n    // TODO: Implement solution\n    // TODO: Output result\n}'
  return {"reasoning": reasoning, 'rust_code': code}, reasoning
