"""Test 49: Parallel Prefix Sum (SPIR-V Binary Compute) - Placebo responses."""


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
  reasoning = 'Naive parallel prefix sum in raw SPIR-V binary. Each thread sums all preceding elements (O(N) per thread, O(N^2) total).'
  code = ''  # SPIR-V binary too complex to hand-write as placebo
  return {"reasoning": reasoning, "spirv_hex": code}, reasoning


def _naive_optimised(subpass):
  return _naive(subpass)


def _best_published(subpass):
  reasoning = (
    "Best published: Work-efficient parallel scan "
    "(Blelloch 1990, 'Prefix Sums and Their Applications', CMU-CS-90-190). "
    "TODO: Full SPIR-V binary implementation pending."
  )
  code = ''
  return {"reasoning": reasoning, "spirv_hex": code}, reasoning


def _random(subpass):
  reasoning = 'Random: empty SPIR-V binary (will fail validation).'
  code = ''
  return {"reasoning": reasoning, "spirv_hex": code}, reasoning


def _human(subpass):
  reasoning = 'Human starting point for prefix sum SPIR-V binary compute shader.'
  code = ''
  return {"reasoning": reasoning, "spirv_hex": code}, reasoning
