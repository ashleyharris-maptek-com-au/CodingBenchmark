"""Test 46: N-Body Gravitational Simulation (GLSL Compute) - Placebo responses."""


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
  reasoning = 'Naive O(N^2) all-pairs gravitational N-body in GLSL compute. Each thread computes force from all other bodies.'
  code = r'''#version 450
layout(local_size_x = 256) in;
layout(set = 0, binding = 0) readonly buffer In { vec4 inData[]; };
layout(set = 0, binding = 1) buffer Out { vec4 outData[]; };
layout(set = 0, binding = 2) uniform Params { uvec4 counts; vec4 physics; };
void main() {
    uint i = gl_GlobalInvocationID.x;
    uint N = counts.x;
    if (i >= N) return;
    vec3 pos = inData[i*2].xyz;
    float mass_i = inData[i*2].w;
    vec3 vel = inData[i*2+1].xyz;
    float dt = physics.x;
    float soft = physics.y;
    float G = physics.z;
    vec3 acc = vec3(0.0);
    for (uint j = 0; j < N; j++) {
        if (j == i) continue;
        vec3 diff = inData[j*2].xyz - pos;
        float distSq = dot(diff, diff) + soft * soft;
        float dist = sqrt(distSq);
        acc += G * inData[j*2].w * diff / (distSq * dist);
    }
    vel += acc * dt;
    pos += vel * dt;
    outData[i*2] = vec4(pos, mass_i);
    outData[i*2+1] = vec4(vel, 0.0);
}'''
  return {"reasoning": reasoning, "shader_code": code}, reasoning


def _naive_optimised(subpass):
  # TODO: Use shared memory tiling to reduce global memory reads
  return _naive(subpass)


def _best_published(subpass):
  reasoning = (
    "Best published: Barnes-Hut tree algorithm on GPU "
    "(Burtscher & Pingali 2011, 'An Efficient CUDA Implementation of the Tree-Based Barnes Hut n-Body Algorithm'). "
    "TODO: Full implementation pending.")
  code = '// TODO: Implement Barnes-Hut on GPU'
  return {"reasoning": reasoning, "shader_code": code}, reasoning


def _random(subpass):
  reasoning = 'Random: just copy input to output unchanged.'
  code = r'''#version 450
layout(local_size_x = 256) in;
layout(set = 0, binding = 0) readonly buffer In { vec4 inData[]; };
layout(set = 0, binding = 1) buffer Out { vec4 outData[]; };
layout(set = 0, binding = 2) uniform Params { uvec4 counts; vec4 physics; };
void main() {
    uint i = gl_GlobalInvocationID.x;
    if (i >= counts.x) return;
    outData[i*2] = inData[i*2];
    outData[i*2+1] = inData[i*2+1];
}'''
  return {"reasoning": reasoning, "shader_code": code}, reasoning


def _human(subpass):
  reasoning = 'Human starting point for N-body compute shader. Fill in the force computation.'
  code = r'''#version 450
layout(local_size_x = 256) in;
layout(set = 0, binding = 0) readonly buffer In { vec4 inData[]; };
layout(set = 0, binding = 1) buffer Out { vec4 outData[]; };
layout(set = 0, binding = 2) uniform Params { uvec4 counts; vec4 physics; };
void main() {
    uint i = gl_GlobalInvocationID.x;
    uint N = counts.x;
    if (i >= N) return;
    vec3 pos = inData[i*2].xyz;
    float mass_i = inData[i*2].w;
    vec3 vel = inData[i*2+1].xyz;
    // TODO: Compute gravitational acceleration from all other bodies
    vec3 acc = vec3(0.0);
    // TODO: Update velocity and position
    outData[i*2] = vec4(pos, mass_i);
    outData[i*2+1] = vec4(vel, 0.0);
}'''
  return {"reasoning": reasoning, "shader_code": code}, reasoning
