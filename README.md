# Ash's LLM Coding Benchmark

A growing suite of **54 tests** that measure LLM ability to design algorithms, write systems code, and reason about performance constraints. The benchmark now spans classic algorithms, systems engineering, GPU compute, physics simulation, and shader authoring (GLSL/HLSL/SPIR-V).

Live report example: <https://ashleyharris-maptek-com-au.github.io/CodingBenchmark/results/gpt-5-mini.html>

## What this benchmark covers

- Algorithm design and optimization under time limits
- NP-hard heuristics and approximation strategies
- Robustness and input handling for large files/streams
- Numerical simulation correctness (physics, fluids, orbital mechanics)
- GPU compute and shader authoring (GLSL/HLSL/SPIR-V)
- Visualization and report generation for interpretability

Each test defines its own prompt, input generation, scoring, and report visualization. Many tests include multiple **subpasses** to stress scaling behavior.

## Repository layout

- `CodingBenchmark.py` - main entrypoint
- `LLMBenchCore/` - runner framework, model adapters, caching, reporting
- `placebo_data/` - baseline implementations ("Human with tools")
- `results/` - HTML reports and run artifacts
- `visualization_utils.py` - shared rendering helpers for reports
- `1.py` ... `54.py` - individual tests

## Installation

```bash
git clone --recursive <repo-url>
pip install -r requirements.txt
```

## Usage

Run the benchmark via `CodingBenchmark.py` (the CLI has evolved from the original single-test script).

```bash
# Run all tests on all models
python CodingBenchmark.py

# Run specific models
python CodingBenchmark.py -m gpt-5-nano
python CodingBenchmark.py -m "claude-*"

# Run specific tests
python CodingBenchmark.py -t 1,2,3
python CodingBenchmark.py -t 5-10

# Run a specific subpass (test.subpass)
python CodingBenchmark.py -t 2.3
python CodingBenchmark.py -t 2.0,2.3,2.5

# List available models
python CodingBenchmark.py --list-models

# Force bypass cache
python CodingBenchmark.py --force

# Offline mode (cache only)
python CodingBenchmark.py --offline

# Parallel execution (one process per model)
python CodingBenchmark.py --parallel

# Batch mode (where supported by provider)
python CodingBenchmark.py --batch
python CodingBenchmark.py --batch -m "gpt-*,claude-*"

# Import results from a cancelled batch
python CodingBenchmark.py --import-batch <batch_id_or_jsonl> --import-model gpt-5.2

# Setup mode: download/build reference data
python CodingBenchmark.py --setup
```

Full CLI options live in `LLMBenchCore/TestRunner.py` (`create_argument_parser`).

## Scoring and reporting

- Each test returns a score per subpass (0.0 to 1.0).
- Subpass scores roll up to a test score.
- Reports are generated as HTML with per-test visualizations where applicable.

Outputs are written to `results/`, and model-specific reports are saved under `results/models/<model>/`.

## Baselines (Placebo)

The benchmark includes multiple baseline behaviors to control against in `placebo_data/`:

- naive
- naive-optimised
- best-published
- random
- human

These are used for sanity checks and to provide a reference for LLM performance. Consider these a work in progress at the moment.

## License

MIT License - see `LLMBenchCore/LICENSE` for details.
