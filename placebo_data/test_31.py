"""Test 31: Subset Sum - Placebo responses for all control types."""


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
  reasoning = 'Using meet-in-the-middle for subset sum: 1) Split array into two halves. 2) Generate all subset sums for first half. 3) For each sum in second half, check if complement exists. 4) Reconstruct solution indices.'
  code = 'use std::io::{self, BufRead};\nuse std::collections::HashMap;\nfn main() {\n    let stdin = io::stdin();\n    let mut lines = stdin.lock().lines();\n    let header: Vec<i64> = lines.next().unwrap().unwrap()\n        .split_whitespace().filter_map(|s| s.parse().ok()).collect();\n    let n = header[0] as usize;\n    let target = header[1];\n    let nums: Vec<i64> = lines.next().unwrap().unwrap()\n        .split_whitespace().filter_map(|s| s.parse().ok()).collect();\n    let mid = n / 2;\n    let mut left: HashMap<i64, Vec<usize>> = HashMap::new();\n    for mask in 0..(1 << mid) {\n        let mut sum = 0i64;\n        let mut indices = Vec::new();\n        for i in 0..mid {\n            if mask & (1 << i) != 0 { sum += nums[i]; indices.push(i); }\n        }\n        left.insert(sum, indices);\n    }\n    for mask in 0..(1 << (n - mid)) {\n        let mut sum = 0i64;\n        let mut indices = Vec::new();\n        for i in 0..(n - mid) {\n            if mask & (1 << i) != 0 { sum += nums[mid + i]; indices.push(mid + i); }\n        }\n        let need = target - sum;\n        if let Some(left_idx) = left.get(&need) {\n            let mut all: Vec<usize> = left_idx.clone();\n            all.extend(indices);\n            if !all.is_empty() || target == 0 {\n                println!("YES");\n                let s: Vec<String> = all.iter().map(|x| x.to_string()).collect();\n                println!("{}", s.join(" "));\n                return;\n            }\n        }\n    }\n    println!("NO");\n}'
  return {"reasoning": reasoning, 'rust_code': code}, reasoning


def _naive_optimised(subpass):
  # TODO: Same algorithm as naive but hyper-optimised for Rust
  # (parallel, SIMD, small data types, register packing, etc.)
  return _naive(subpass)


def _best_published(subpass):
  reasoning = (
    "Best published: Horowitz-Sahni meet-in-the-middle (Horowitz & Sahni 1974, J. ACM 21(2):277-292). "
    "TODO: Full implementation pending."
  )
  code = '// TODO: Implement Horowitz-Sahni meet-in-the-middle'
  return {"reasoning": reasoning, 'rust_code': code}, reasoning


def _random(subpass):
  reasoning = 'Random: pseudorandom output with fixed seed 42.'
  code = 'use std::io::{self, BufRead};\nfn main() {\n    let stdin = io::stdin();\n    let _h = stdin.lock().lines().next();\n    println!("0");\n}'
  return {"reasoning": reasoning, 'rust_code': code}, reasoning


def _human(subpass):
  reasoning = 'Human starting point for Subset Sum. Fill in the TODOs.'
  code = '// TODO: Human attempt at Subset Sum\nuse std::io::{self, BufRead, Write};\nfn main() {\n    let stdin = io::stdin();\n    let mut lines = stdin.lock().lines();\n    // TODO: Parse input\n    // TODO: Implement solution\n    // TODO: Output result\n}'
  return {"reasoning": reasoning, 'rust_code': code}, reasoning
