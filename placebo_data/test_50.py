"""Test 50: Boid Flocking Simulation (GLSL Compute) - Placebo responses."""


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
  reasoning = 'Naive O(N^2) boid flocking in GLSL compute. Each thread checks all other boids for separation/alignment/cohesion.'
  code = r'''#version 450
layout(local_size_x = 256) in;
layout(set = 0, binding = 0) readonly buffer In { vec4 inData[]; };
layout(set = 0, binding = 1) buffer Out { vec4 outData[]; };
layout(set = 0, binding = 2) uniform Params {
    uvec4 counts;
    vec4 radii;
    vec4 weights;
    vec4 bounds;
};
void main() {
    uint i = gl_GlobalInvocationID.x;
    uint N = counts.x;
    if (i >= N) return;
    vec3 pos = inData[i*2].xyz;
    vec3 vel = inData[i*2+1].xyz;
    float dt = radii.x;
    float sepR = radii.y, aliR = radii.z, cohR = radii.w;
    float sepW = weights.x, aliW = weights.y, cohW = weights.z, maxSpd = weights.w;
    vec3 sep = vec3(0), ali = vec3(0), coh = vec3(0);
    int sc = 0, ac = 0, cc = 0;
    for (uint j = 0; j < N; j++) {
        if (j == i) continue;
        vec3 other = inData[j*2].xyz;
        vec3 diff = pos - other;
        float d = length(diff);
        if (d < 1e-6) continue;
        if (d < sepR) { sep += diff / d; sc++; }
        if (d < aliR) { ali += inData[j*2+1].xyz; ac++; }
        if (d < cohR) { coh += other; cc++; }
    }
    vec3 acc = vec3(0);
    if (sc > 0) acc += sepW * sep / float(sc);
    if (ac > 0) acc += aliW * (ali / float(ac) - vel);
    if (cc > 0) acc += cohW * (coh / float(cc) - pos);
    float hb = bounds.x * 0.5;
    if (abs(pos.x) > hb * 0.8) acc.x -= bounds.y * (pos.x / hb);
    if (abs(pos.y) > hb * 0.8) acc.y -= bounds.y * (pos.y / hb);
    if (abs(pos.z) > hb * 0.8) acc.z -= bounds.y * (pos.z / hb);
    vel += acc * dt;
    float spd = length(vel);
    if (spd > maxSpd) vel = vel * (maxSpd / spd);
    pos += vel * dt;
    outData[i*2] = vec4(pos, 0.0);
    outData[i*2+1] = vec4(vel, 0.0);
}'''
  return {"reasoning": reasoning, "shader_code": code}, reasoning


def _naive_optimised(subpass):
  # TODO: Use shared memory tiling for neighbor search
  return _naive(subpass)


def _best_published(subpass):
  reasoning = ("Best published: Spatial hashing for boid neighbor search on GPU "
               "(Green 2010, 'Particle Simulation using CUDA', NVIDIA GPU Computing SDK). "
               "TODO: Full implementation pending.")
  code = '// TODO: Implement spatial hash boid simulation'
  return {"reasoning": reasoning, "shader_code": code}, reasoning


def _random(subpass):
  reasoning = 'Random: copy input to output unchanged.'
  code = r'''#version 450
layout(local_size_x = 256) in;
layout(set = 0, binding = 0) readonly buffer In { vec4 inData[]; };
layout(set = 0, binding = 1) buffer Out { vec4 outData[]; };
layout(set = 0, binding = 2) uniform Params { uvec4 counts; vec4 radii; vec4 weights; vec4 bounds; };
void main() {
    uint i = gl_GlobalInvocationID.x;
    if (i >= counts.x) return;
    outData[i*2] = inData[i*2];
    outData[i*2+1] = inData[i*2+1];
}'''
  return {"reasoning": reasoning, "shader_code": code}, reasoning


def _human(subpass):
  reasoning = 'Human starting point for boid flocking GLSL compute shader.'
  code = r'''#version 450
layout(local_size_x = 256) in;
layout(set = 0, binding = 0) readonly buffer In { vec4 inData[]; };
layout(set = 0, binding = 1) buffer Out { vec4 outData[]; };
layout(set = 0, binding = 2) uniform Params {
    uvec4 counts;
    vec4 radii;
    vec4 weights;
    vec4 bounds;
};
void main() {
    uint i = gl_GlobalInvocationID.x;
    uint N = counts.x;
    if (i >= N) return;
    vec3 pos = inData[i*2].xyz;
    vec3 vel = inData[i*2+1].xyz;
    // TODO: Compute separation, alignment, cohesion forces
    // TODO: Apply boundary forces
    // TODO: Update velocity (clamp to maxSpeed) and position
    outData[i*2] = vec4(pos, 0.0);
    outData[i*2+1] = vec4(vel, 0.0);
}'''
  return {"reasoning": reasoning, "shader_code": code}, reasoning
