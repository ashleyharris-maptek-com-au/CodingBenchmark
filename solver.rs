
use std::io::{self, Read, Write};
use std::fs::File;
use std::env;

struct Xorshift {
    state: u32,
}

impl Xorshift {
    fn new(seed: u32) -> Self {
        let mut s = seed;
        if s == 0 { s = 1; }
        Xorshift { state: s }
    }
    fn next(&mut self) -> u32 {
        let mut x = self.state;
        x ^= x << 13;
        x ^= x >> 17;
        x ^= x << 5;
        self.state = x;
        x
    }
    fn next_range(&mut self, limit: usize) -> usize {
        if limit == 0 { return 0; }
        (self.next() as usize) % limit
    }
    fn next_f64(&mut self) -> f64 {
        (self.next() as f64) / (u32::MAX as f64)
    }
}

struct FastScanner<R> {
    reader: R,
    buffer: Vec<u8>,
    pos: usize,
    cap: usize,
}

impl<R: Read> FastScanner<R> {
    fn new(reader: R) -> Self {
        FastScanner {
            reader,
            buffer: vec![0; 1024 * 1024],
            pos: 0,
            cap: 0,
        }
    }

    fn next_int(&mut self) -> Option<i32> {
        loop {
            if self.pos >= self.cap {
                self.cap = self.reader.read(&mut self.buffer).unwrap_or(0);
                self.pos = 0;
                if self.cap == 0 {
                    return None;
                }
            }
            while self.pos < self.cap && self.buffer[self.pos] <= b' ' {
                self.pos += 1;
            }
            if self.pos < self.cap {
                break;
            }
        }
        
        let mut neg = false;
        if self.buffer[self.pos] == b'-' {
            neg = true;
            self.pos += 1;
        } else if self.buffer[self.pos] == b'+' {
            self.pos += 1;
        }
        
        let mut val = 0;
        loop {
            if self.pos >= self.cap {
                self.cap = self.reader.read(&mut self.buffer).unwrap_or(0);
                self.pos = 0;
                if self.cap == 0 {
                    break;
                }
            }
            let c = self.buffer[self.pos];
            if c > b' ' {
                val = val * 10 + (c - b'0') as i32;
                self.pos += 1;
            } else {
                break;
            }
        }
        if neg {
            Some(-val)
        } else {
            Some(val)
        }
    }
}

fn main() -> io::Result<()> {
    let args_env: Vec<String> = env::args().collect();
    let input: Box<dyn Read> = if args_env.len() > 1 {
        Box::new(File::open(&args_env[1])?)
    } else {
        Box::new(io::stdin())
    };
    
    let mut scanner = FastScanner::new(input);
    let num_vars = match scanner.next_int() {
        Some(v) => v as usize,
        None => return Ok(()),
    };
    let num_clauses = match scanner.next_int() {
        Some(c) => c as usize,
        None => return Ok(()),
    };
    
    let mut clauses = Vec::with_capacity(3 * num_clauses);
    for _ in 0..(3 * num_clauses) {
        if let Some(val) = scanner.next_int() {
            clauses.push(val);
        } else {
            break;
        }
    }
    
    let mut pos_counts = vec![0; num_vars + 1];
    let mut neg_counts = vec![0; num_vars + 1];
    for &lit in &clauses {
        if lit > 0 {
            pos_counts[lit as usize] += 1;
        } else {
            neg_counts[(-lit) as usize] += 1;
        }
    }
    
    let mut assign = vec![0u8; num_vars + 1];
    for i in 1..=num_vars {
        if pos_counts[i] >= neg_counts[i] {
            assign[i] = 1;
        } else {
            assign[i] = 0;
        }
    }
    
    let mut counts = vec![0; 2 * num_vars + 2];
    for &lit in &clauses {
        let lit_idx = (lit + num_vars as i32) as usize;
        counts[lit_idx] += 1;
    }
    
    let mut offsets = vec![0; 2 * num_vars + 2];
    let mut sum = 0;
    for i in 0..(2 * num_vars + 2) {
        offsets[i] = sum;
        sum += counts[i];
    }
    
    let mut args = vec![0; 3 * num_clauses];
    let mut current_offsets = offsets.clone();
    for j in 0..(3 * num_clauses) {
        let lit = clauses[j];
        let lit_idx = (lit + num_vars as i32) as usize;
        let pos = current_offsets[lit_idx];
        args[pos] = j;
        current_offsets[lit_idx] += 1;
    }
    
    let mut clause_idx_of_occ = vec![0; 3 * num_clauses];
    for i in 0..(3 * num_clauses) {
        clause_idx_of_occ[i] = args[i] / 3;
    }
    
    let mut sat_count = vec![0u8; num_clauses];
    let mut unsat_list = Vec::new();
    let mut is_unsat = vec![0u8; num_clauses];
    
    for i in 0..num_clauses {
        let idx = 3 * i;
        let lit1 = clauses[idx];
        let lit2 = clauses[idx+1];
        let lit3 = clauses[idx+2];
        
        let mut c = 0;
        if lit1 > 0 {
            if assign[lit1 as usize] == 1 { c += 1; }
        } else {
            if assign[(-lit1) as usize] == 0 { c += 1; }
        }
        
        if lit2 > 0 {
            if assign[lit2 as usize] == 1 { c += 1; }
        } else {
            if assign[(-lit2) as usize] == 0 { c += 1; }
        }
        
        if lit3 > 0 {
            if assign[lit3 as usize] == 1 { c += 1; }
        } else {
            if assign[(-lit3) as usize] == 0 { c += 1; }
        }
        
        sat_count[i] = c;
        if c == 0 {
            is_unsat[i] = 1;
            unsat_list.push(i);
        }
    }
    
    // Hill Climbing Pass
    loop {
        let mut flips = 0;
        for x in 1..=num_vars {
            let true_lit = if assign[x] == 1 { x as i32 } else { -(x as i32) };
            let false_lit = -true_lit;
            
            let mut break_count = 0;
            let start = offsets[(true_lit + num_vars as i32) as usize];
            let end = offsets[(true_lit + num_vars as i32 + 1) as usize];
            for k in start..end {
                let c_prime = clause_idx_of_occ[k];
                if sat_count[c_prime] == 1 {
                    break_count += 1;
                }
            }
            
            let mut make_count = 0;
            let start = offsets[(false_lit + num_vars as i32) as usize];
            let end = offsets[(false_lit + num_vars as i32 + 1) as usize];
            for k in start..end {
                let c_prime = clause_idx_of_occ[k];
                if sat_count[c_prime] == 0 {
                    make_count += 1;
                }
            }
            
            if make_count > break_count {
                assign[x] = 1 - assign[x];
                flips += 1;
                
                let new_true_lit = if assign[x] == 1 { x as i32 } else { -(x as i32) };
                let start = offsets[(new_true_lit + num_vars as i32) as usize];
                let end = offsets[(new_true_lit + num_vars as i32 + 1) as usize];
                for k in start..end {
                    let c_prime = clause_idx_of_occ[k];
                    if sat_count[c_prime] == 0 {
                        is_unsat[c_prime] = 0;
                    }
                    sat_count[c_prime] += 1;
                }
                
                let new_false_lit = -new_true_lit;
                let start = offsets[(new_false_lit + num_vars as i32) as usize];
                let end = offsets[(new_false_lit + num_vars as i32 + 1) as usize];
                for k in start..end {
                    let c_prime = clause_idx_of_occ[k];
                    sat_count[c_prime] -= 1;
                    if sat_count[c_prime] == 0 {
                        if is_unsat[c_prime] == 0 {
                            is_unsat[c_prime] = 1;
                            unsat_list.push(c_prime);
                        }
                    }
                }
            }
        }
        
        unsat_list.clear();
        for i in 0..num_clauses {
            if sat_count[i] == 0 {
                is_unsat[i] = 1;
                unsat_list.push(i);
            } else {
                is_unsat[i] = 0;
            }
        }
        
        if flips == 0 || unsat_list.is_empty() {
            break;
        }
    }
    
    let mut rng = Xorshift::new(12345);
    let mut steps = 0;
    let p = 0.35;
    let max_steps = 100_000_000;
    
    while !unsat_list.is_empty() && steps < max_steps {
        let c = unsat_list[unsat_list.len() - 1];
        if is_unsat[c] == 0 {
            unsat_list.pop();
            continue;
        }
        
        steps += 1;
        if steps % 2000000 == 0 {
            let mut i = 0;
            let mut j = 0;
            while i < unsat_list.len() {
                let cl = unsat_list[i];
                if is_unsat[cl] == 1 {
                    unsat_list[j] = cl;
                    j += 1;
                }
                i += 1;
            }
            unsat_list.truncate(j);
        }
        
        let idx = 3 * c;
        let v1 = clauses[idx].abs() as usize;
        let v2 = clauses[idx+1].abs() as usize;
        let v3 = clauses[idx+2].abs() as usize;
        
        let candidates = [v1, v2, v3];
        let var_to_flip;
        
        if rng.next_f64() < p {
            var_to_flip = candidates[rng.next_range(3)];
        } else {
            let mut best_break = 999999;
            let mut break_counts = [0, 0, 0];
            
            for idx_cand in 0..3 {
                let var = candidates[idx_cand];
                let true_lit = if assign[var] == 1 { var as i32 } else { -(var as i32) };
                let start = offsets[(true_lit + num_vars as i32) as usize];
                let end = offsets[(true_lit + num_vars as i32 + 1) as usize];
                let mut br = 0;
                for k in start..end {
                    let c_prime = clause_idx_of_occ[k];
                    if sat_count[c_prime] == 1 {
                        br += 1;
                    }
                }
                break_counts[idx_cand] = br;
                if br < best_break {
                    best_break = br;
                }
            }
            
            if best_break == 0 {
                let mut zeros = Vec::with_capacity(3);
                for idx_cand in 0..3 {
                    if break_counts[idx_cand] == 0 {
                        zeros.push(candidates[idx_cand]);
                    }
                }
                var_to_flip = zeros[rng.next_range(zeros.len())];
            } else {
                let mut mins = Vec::with_capacity(3);
                for idx_cand in 0..3 {
                    if break_counts[idx_cand] == best_break {
                        mins.push(candidates[idx_cand]);
                    }
                }
                var_to_flip = mins[rng.next_range(mins.len())];
            }
        }
        
        let old_val = assign[var_to_flip];
        let new_val = 1 - old_val;
        assign[var_to_flip] = new_val;
        
        let new_true_lit = if new_val == 1 { var_to_flip as i32 } else { -(var_to_flip as i32) };
        let start = offsets[(new_true_lit + num_vars as i32) as usize];
        let end = offsets[(new_true_lit + num_vars as i32 + 1) as usize];
        for k in start..end {
            let c_prime = clause_idx_of_occ[k];
            if sat_count[c_prime] == 0 {
                is_unsat[c_prime] = 0;
            }
            sat_count[c_prime] += 1;
        }
        
        let new_false_lit = -new_true_lit;
        let start = offsets[(new_false_lit + num_vars as i32) as usize];
        let end = offsets[(new_false_lit + num_vars as i32 + 1) as usize];
        for k in start..end {
            let c_prime = clause_idx_of_occ[k];
            sat_count[c_prime] -= 1;
            if sat_count[c_prime] == 0 {
                if is_unsat[c_prime] == 0 {
                    is_unsat[c_prime] = 1;
                    unsat_list.push(c_prime);
                }
            }
        }
    }
    
    let num_unsat = sat_count.iter().filter(|&&c| c == 0).count();
    if num_unsat == 0 {
        println!("SAT");
        let mut out = io::stdout();
        let mut buffer = String::new();
        for i in 1..=num_vars {
            buffer.push_str(&assign[i].to_string());
            if i < num_vars {
                buffer.push(' ');
            }
            if buffer.len() > 65536 {
                out.write_all(buffer.as_bytes())?;
                buffer.clear();
            }
        }
        if !buffer.is_empty() {
            out.write_all(buffer.as_bytes())?;
        }
        out.write_all(b"\n")?;
    } else {
        std::process::exit(1);
    }
    Ok(())
}
