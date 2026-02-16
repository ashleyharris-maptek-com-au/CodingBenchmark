"""Test 43: HLSL Fragment Shaders - Placebo responses (correctness test)."""


def get_response(model_name, subpass):
  """All control types return the same for correctness tests."""
  # Shader/correctness tests: answer is right or wrong, no optimization axis.
  # All control types produce the same code.
  return None, 'No pre-built placebo for HLSL Fragment Shaders subpass ' + str(subpass)
