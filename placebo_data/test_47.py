"""Test 47: Hash Mining / Proof of Work (HLSL Compute) - Placebo responses."""


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
  reasoning = 'Naive HLSL hash mining: each thread checks one nonce with the ChaCha-like hash, unrolled quarter-rounds, race-based result write.'
  code = r'''[[vk::binding(0, 0)]] StructuredBuffer<uint> baseData;
[[vk::binding(1, 0)]] RWStructuredBuffer<uint> result;
[[vk::binding(2, 0)]] cbuffer Params { uint4 params; };
uint rotl(uint v, uint n) { return (v << n) | (v >> (32 - n)); }
uint countTZ(uint v) {
    if (v == 0) return 32;
    uint c = 0;
    if ((v & 0xFFFF) == 0) { c += 16; v >>= 16; }
    if ((v & 0xFF) == 0) { c += 8; v >>= 8; }
    if ((v & 0xF) == 0) { c += 4; v >>= 4; }
    if ((v & 0x3) == 0) { c += 2; v >>= 2; }
    if ((v & 0x1) == 0) { c += 1; }
    return c;
}
[numthreads(256, 1, 1)]
void main(uint3 dtid : SV_DispatchThreadID) {
    uint nonce = params.y + dtid.x;
    if (nonce >= params.y + params.z) return;
    uint s[16];
    for (uint i = 0; i < 16; i++) s[i] = baseData[i] ^ (nonce + i);
    for (uint r = 0; r < 8; r++) {
        s[0] += s[1]; s[3] ^= s[0]; s[3] = rotl(s[3], 16);
        s[2] += s[3]; s[1] ^= s[2]; s[1] = rotl(s[1], 12);
        s[0] += s[1]; s[3] ^= s[0]; s[3] = rotl(s[3], 8);
        s[2] += s[3]; s[1] ^= s[2]; s[1] = rotl(s[1], 7);
        s[4] += s[5]; s[7] ^= s[4]; s[7] = rotl(s[7], 16);
        s[6] += s[7]; s[5] ^= s[6]; s[5] = rotl(s[5], 12);
        s[4] += s[5]; s[7] ^= s[4]; s[7] = rotl(s[7], 8);
        s[6] += s[7]; s[5] ^= s[6]; s[5] = rotl(s[5], 7);
        s[8] += s[9]; s[11] ^= s[8]; s[11] = rotl(s[11], 16);
        s[10] += s[11]; s[9] ^= s[10]; s[9] = rotl(s[9], 12);
        s[8] += s[9]; s[11] ^= s[8]; s[11] = rotl(s[11], 8);
        s[10] += s[11]; s[9] ^= s[10]; s[9] = rotl(s[9], 7);
        s[12] += s[13]; s[15] ^= s[12]; s[15] = rotl(s[15], 16);
        s[14] += s[15]; s[13] ^= s[14]; s[13] = rotl(s[13], 12);
        s[12] += s[13]; s[15] ^= s[12]; s[15] = rotl(s[15], 8);
        s[14] += s[15]; s[13] ^= s[14]; s[13] = rotl(s[13], 7);
    }
    uint h0 = s[0]^s[2]^s[4]^s[6]^s[8]^s[10]^s[12]^s[14];
    uint h1 = s[1]^s[3]^s[5]^s[7]^s[9]^s[11]^s[13]^s[15];
    uint tz = countTZ(h0);
    if (h0 == 0) tz = 32 + countTZ(h1);
    if (tz >= params.x) {
        result[0] = 1;
        result[1] = nonce;
        result[2] = h0;
        result[3] = h1;
    }
}'''
  return {"reasoning": reasoning, "shader_code": code}, reasoning


def _naive_optimised(subpass):
  # TODO: Unroll loops, use wave intrinsics for early exit
  return _naive(subpass)


def _best_published(subpass):
  reasoning = (
    "Best published: GPU hash mining techniques from Bitcoin/cryptocurrency mining literature. "
    "TODO: Full implementation pending.")
  code = '// TODO: Implement optimised GPU hash mining'
  return {"reasoning": reasoning, "shader_code": code}, reasoning


def _random(subpass):
  reasoning = 'Random: always reports nonce 0 (almost certainly wrong).'
  code = r'''[[vk::binding(0, 0)]] StructuredBuffer<uint> baseData;
[[vk::binding(1, 0)]] RWStructuredBuffer<uint> result;
[[vk::binding(2, 0)]] cbuffer Params { uint4 params; };
[numthreads(256, 1, 1)]
void main(uint3 dtid : SV_DispatchThreadID) {
    if (dtid.x == 0) { result[0] = 1; result[1] = 0; result[2] = 0; result[3] = 0; }
}'''
  return {"reasoning": reasoning, "shader_code": code}, reasoning


def _human(subpass):
  reasoning = 'Human starting point for hash mining HLSL compute shader.'
  code = r'''[[vk::binding(0, 0)]] StructuredBuffer<uint> baseData;
[[vk::binding(1, 0)]] RWStructuredBuffer<uint> result;
[[vk::binding(2, 0)]] cbuffer Params { uint4 params; };
// TODO: Implement rotl helper
// TODO: Implement ChaCha-like hash
[numthreads(256, 1, 1)]
void main(uint3 dtid : SV_DispatchThreadID) {
    uint nonce = params.y + dtid.x;
    // TODO: Hash and check trailing zeros
    // TODO: Report via InterlockedCompareExchange
}'''
  return {"reasoning": reasoning, "shader_code": code}, reasoning
