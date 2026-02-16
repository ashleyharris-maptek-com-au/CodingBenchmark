"""Test 48: 2D Fluid Simulation - Lattice Boltzmann (SPIR-V ASM Compute) - Placebo responses."""


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
  reasoning = 'Naive D2Q9 LBM in SPIR-V assembly. Each thread handles one grid cell: collision (BGK) + streaming with periodic boundaries.'
  code = '; TODO: Full SPIR-V assembly LBM implementation'
  return {"reasoning": reasoning, "spirv_code": code}, reasoning


def _naive_optimised(subpass):
  return _naive(subpass)


def _best_published(subpass):
  reasoning = (
    "Best published: GPU LBM with AA pattern "
    "(Bailey et al. 2009, 'Accelerating Lattice Boltzmann Fluid Flow Simulations Using Graphics Processors'). "
    "TODO: Full implementation pending."
  )
  code = '; TODO: Implement optimised GPU LBM'
  return {"reasoning": reasoning, "spirv_code": code}, reasoning


def _random(subpass):
  reasoning = 'Random: copy input to output unchanged.'
  code = '; TODO: Passthrough SPIR-V compute shader'
  return {"reasoning": reasoning, "spirv_code": code}, reasoning


def _human(subpass):
  reasoning = 'Human starting point for LBM SPIR-V assembly compute shader.'
  code = '; TODO: Implement D2Q9 Lattice Boltzmann in SPIR-V assembly'
  return {"reasoning": reasoning, "spirv_code": code}, reasoning
