"""Test 20: Drillhole Validation - Placebo responses for all control types."""


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
  reasoning = 'Using k-nearest neighbor z-score analysis in Rust: 1) Parse all drillhole samples with their 3D positions. 2) For each sample, find k nearest neighbors from other holes. 3) Calculate mean and stddev of each property from neighbors. 4) Compute z-score: |value - mean| / stddev. 5) Flag entries with z-score > 3.0 as suspects (3-sigma outliers). 6) Sort by confidence (z-score) descending.'
  code = 'use std::io::{self, BufRead, Write};\nuse std::collections::HashMap;\n\n#[derive(Clone)]\nstruct Sample {\n    hole_id: usize,\n    sample_idx: usize,\n    x: f64, y: f64, z: f64,\n    properties: Vec<f64>,\n}\n\nfn distance(a: &Sample, b: &Sample) -> f64 {\n    ((a.x - b.x).powi(2) + (a.y - b.y).powi(2) + (a.z - b.z).powi(2)).sqrt()\n}\n\nfn main() {\n    let stdin = io::stdin();\n    let mut lines = stdin.lock().lines();\n    \n    // Parse header\n    let first_line = lines.next().unwrap().unwrap();\n    let parts: Vec<&str> = first_line.split_whitespace().collect();\n    let n: usize = parts[0].parse().unwrap();\n    let num_props: usize = parts[1].parse().unwrap();\n    \n    let prop_line = lines.next().unwrap().unwrap();\n    let property_names: Vec<String> = prop_line.split_whitespace().map(|s| s.to_string()).collect();\n    \n    let mut all_samples: Vec<Sample> = Vec::new();\n    \n    // Parse holes\n    for _ in 0..n {\n        let hole_line = lines.next().unwrap().unwrap();\n        let parts: Vec<f64> = hole_line.split_whitespace()\n            .filter_map(|s| s.parse().ok())\n            .collect();\n        \n        let hole_id = parts[0] as usize;\n        let start_x = parts[1];\n        let start_y = parts[2];\n        let start_z = parts[3];\n        let dir_x = parts[4];\n        let dir_y = parts[5];\n        let dir_z = parts[6];\n        let num_samples = parts[8] as usize;\n        \n        for sample_idx in 0..num_samples {\n            let sample_line = lines.next().unwrap().unwrap();\n            let vals: Vec<f64> = sample_line.split_whitespace()\n                .filter_map(|s| s.parse().ok())\n                .collect();\n            \n            let depth = vals[0];\n            let x = start_x + dir_x * depth;\n            let y = start_y + dir_y * depth;\n            let z = start_z + dir_z * depth;\n            \n            let properties: Vec<f64> = vals[1..].to_vec();\n            \n            all_samples.push(Sample {\n                hole_id, sample_idx, x, y, z, properties\n            });\n        }\n    }\n    \n    // Find suspects using z-score with k nearest neighbors\n    let k = 10.min(all_samples.len().saturating_sub(1));\n    let mut suspects: Vec<(usize, usize, String, f64)> = Vec::new();\n    \n    for i in 0..all_samples.len() {\n        let sample = &all_samples[i];\n        \n        // Find k nearest neighbors (from different holes)\n        let mut neighbors: Vec<(f64, usize)> = Vec::new();\n        for j in 0..all_samples.len() {\n            if all_samples[j].hole_id != sample.hole_id {\n                let d = distance(sample, &all_samples[j]);\n                neighbors.push((d, j));\n            }\n        }\n        neighbors.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());\n        neighbors.truncate(k);\n        \n        if neighbors.is_empty() { continue; }\n        \n        // Check each property\n        for prop_idx in 0..sample.properties.len().min(num_props) {\n            let value = sample.properties[prop_idx];\n            \n            // Calculate mean and stddev of neighbors\n            let neighbor_vals: Vec<f64> = neighbors.iter()\n                .filter_map(|(_, j)| all_samples[*j].properties.get(prop_idx).copied())\n                .collect();\n            \n            if neighbor_vals.is_empty() { continue; }\n            \n            let mean: f64 = neighbor_vals.iter().sum::<f64>() / neighbor_vals.len() as f64;\n            let variance: f64 = neighbor_vals.iter()\n                .map(|v| (v - mean).powi(2))\n                .sum::<f64>() / neighbor_vals.len() as f64;\n            let stddev = variance.sqrt().max(0.001);\n            \n            let z_score = ((value - mean) / stddev).abs();\n            \n            // Flag if z-score > 3 (3 sigma outlier)\n            if z_score > 3.0 {\n                let prop_name = property_names.get(prop_idx)\n                    .cloned()\n                    .unwrap_or_else(|| format!("prop{}", prop_idx));\n                suspects.push((sample.hole_id, sample.sample_idx, prop_name, z_score));\n            }\n        }\n    }\n    \n    // Sort by confidence (z-score) descending\n    suspects.sort_by(|a, b| b.3.partial_cmp(&a.3).unwrap());\n    \n    // Output\n    println!("{}", suspects.len());\n    for (hole_id, sample_idx, prop, conf) in suspects {\n        println!("{} {} {} {:.2}", hole_id, sample_idx, prop, conf);\n    }\n}\n'
  return {"reasoning": reasoning, 'rust_code': code}, reasoning


def _naive_optimised(subpass):
  # TODO: Same algorithm as naive but hyper-optimised for Rust
  # (parallel, SIMD, small data types, register packing, etc.)
  return _naive(subpass)


def _best_published(subpass):
  reasoning = (
    "Best published: Spatial LISA outlier detection (Anselin 1995, Geographical Analysis 27(2):93-115). "
    "TODO: Full implementation pending."
  )
  code = '// TODO: Implement Spatial LISA outlier detection'
  return {"reasoning": reasoning, 'rust_code': code}, reasoning


def _random(subpass):
  reasoning = 'Random: pseudorandom output with fixed seed 42.'
  code = 'use std::io::{self, BufRead};\nfn main() {\n    let stdin = io::stdin();\n    let _h = stdin.lock().lines().next();\n    println!("0");\n}'
  return {"reasoning": reasoning, 'rust_code': code}, reasoning


def _human(subpass):
  reasoning = 'Human starting point for Drillhole Validation. Fill in the TODOs.'
  code = '// TODO: Human attempt at Drillhole Validation\nuse std::io::{self, BufRead, Write};\nfn main() {\n    let stdin = io::stdin();\n    let mut lines = stdin.lock().lines();\n    // TODO: Parse input\n    // TODO: Implement solution\n    // TODO: Output result\n}'
  return {"reasoning": reasoning, 'rust_code': code}, reasoning
