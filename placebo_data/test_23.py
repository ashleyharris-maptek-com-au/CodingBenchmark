"""Test 23: Lunar Lander - Placebo responses for all control types."""


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
  reasoning = 'Using proportional-derivative (PD) controller for lunar lander: 1) Parse map header and skip to STATE marker. 2) For each state update, calculate error to target. 3) Compute desired velocity vector toward target + gravity compensation. 4) Calculate thrust angle from velocity error. 5) Turn toward desired angle with proportional control. 6) Apply thrust when aligned, reduce near landing. 7) Flush output immediately for real-time response.'
  code = 'use std::io::{self, BufRead, Write};\n\nfn main() {\n    let stdin = io::stdin();\n    let mut lines = stdin.lock().lines();\n    \n    // Parse header\n    let header = lines.next().unwrap().unwrap();\n    let parts: Vec<f64> = header.split_whitespace()\n        .filter_map(|s| s.parse().ok())\n        .collect();\n    let _width = parts[0];\n    let _height = parts[1];\n    let gravity = parts[2];\n    let max_thrust = parts[3];\n    \n    let pos_line = lines.next().unwrap().unwrap();\n    let pos: Vec<f64> = pos_line.split_whitespace()\n        .filter_map(|s| s.parse().ok())\n        .collect();\n    let target_x = pos[2];\n    let target_y = pos[3];\n    \n    // Skip map until STATE marker\n    loop {\n        let line = lines.next().unwrap().unwrap();\n        if line.trim() == "STATE" {\n            break;\n        }\n    }\n    \n    // Process state updates\n    loop {\n        let line = match lines.next() {\n            Some(Ok(l)) => l,\n            _ => break,\n        };\n        \n        if line.trim() == "END" || line.is_empty() {\n            break;\n        }\n        \n        let state: Vec<f64> = line.split_whitespace()\n            .filter_map(|s| s.parse().ok())\n            .collect();\n        \n        if state.len() < 7 {\n            println!("0.0 0.0");\n            io::stdout().flush().unwrap();\n            continue;\n        }\n        \n        let x = state[0];\n        let y = state[1];\n        let vx = state[2];\n        let vy = state[3];\n        let angle = state[4];\n        let _angular_vel = state[5];\n        let fuel = state[6];\n        \n        if fuel <= 0.0 {\n            println!("0.0 0.0");\n            io::stdout().flush().unwrap();\n            continue;\n        }\n        \n        // Simple PD controller\n        let dx = target_x - x;\n        let dy = target_y - y;\n        let dist = (dx * dx + dy * dy).sqrt();\n        \n        // Desired angle to point toward target (with lead)\n        let desired_vx = dx * 0.1;\n        let desired_vy = dy * 0.1 + gravity * 2.0; // Counter gravity\n        \n        let vx_error = desired_vx - vx;\n        let vy_error = desired_vy - vy;\n        \n        // Calculate desired thrust direction\n        let thrust_angle = vx_error.atan2(vy_error);\n        \n        // Turn toward desired angle\n        let angle_error = thrust_angle - angle;\n        let angle_error = if angle_error > std::f64::consts::PI {\n            angle_error - 2.0 * std::f64::consts::PI\n        } else if angle_error < -std::f64::consts::PI {\n            angle_error + 2.0 * std::f64::consts::PI\n        } else {\n            angle_error\n        };\n        \n        let turn = (angle_error * 2.0).max(-1.0).min(1.0);\n        \n        // Thrust based on velocity error magnitude and proximity\n        let vel_error = (vx_error * vx_error + vy_error * vy_error).sqrt();\n        let speed = (vx * vx + vy * vy).sqrt();\n        \n        let thrust = if dist < 50.0 {\n            // Landing phase - careful control\n            if speed > 2.0 {\n                ((speed - 1.0) / max_thrust).max(0.0).min(1.0)\n            } else if vy < -1.0 {\n                ((-vy - 0.5) / max_thrust * 2.0).max(0.0).min(0.5)\n            } else {\n                0.0\n            }\n        } else if angle_error.abs() < 0.3 {\n            // Aligned - thrust\n            (vel_error / max_thrust * 0.5).max(0.1).min(1.0)\n        } else {\n            // Not aligned - minimal thrust while turning\n            0.1\n        };\n        \n        println!("{:.2} {:.2}", thrust, turn);\n        io::stdout().flush().unwrap();\n    }\n}\n'
  return {"reasoning": reasoning, 'rust_code': code}, reasoning


def _naive_optimised(subpass):
  # TODO: Same algorithm as naive but hyper-optimised for Rust
  # (parallel, SIMD, small data types, register packing, etc.)
  return _naive(subpass)


def _best_published(subpass):
  reasoning = (
    "Best published: Pontryagin optimal control (Bryson & Ho 1975, Hemisphere Publishing). "
    "TODO: Full implementation pending."
  )
  code = '// TODO: Implement Pontryagin optimal control'
  return {"reasoning": reasoning, 'rust_code': code}, reasoning


def _random(subpass):
  reasoning = 'Random: pseudorandom output with fixed seed 42.'
  code = 'use std::io::{self, BufRead};\nfn main() {\n    let stdin = io::stdin();\n    let _h = stdin.lock().lines().next();\n    println!("0");\n}'
  return {"reasoning": reasoning, 'rust_code': code}, reasoning


def _human(subpass):
  reasoning = 'Human starting point for Lunar Lander. Fill in the TODOs.'
  code = '// TODO: Human attempt at Lunar Lander\nuse std::io::{self, BufRead, Write};\nfn main() {\n    let stdin = io::stdin();\n    let mut lines = stdin.lock().lines();\n    // TODO: Parse input\n    // TODO: Implement solution\n    // TODO: Output result\n}'
  return {"reasoning": reasoning, 'rust_code': code}, reasoning
