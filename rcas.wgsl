struct RcasConsts { sharp: vec4<f32> };
@group(0) @binding(0) var<uniform> rc : RcasConsts;
@group(0) @binding(1) var sampLinear: sampler;
@group(0) @binding(2) var inTex: texture_2d<f32>;
@group(0) @binding(3) var outImg: texture_storage_2d<rgba8unorm, write>;

@compute @workgroup_size(8,8,1)
fn rcasMain(@builtin(global_invocation_id) gid: vec3<u32>) {
  let size = textureDimensions(outImg);
  if (gid.x >= size.x || gid.y >= size.y) { return; }
  let texSize = vec2<f32>(textureDimensions(inTex));
  let uv = (vec2<f32>(gid.xy) + 0.5) / texSize;
  // 5-tap sharpen kernel (center + 4 cardinals)
  let c = textureSampleLevel(inTex, sampLinear, uv, 0.0).rgb;
  let off = 1.0 / texSize;
  let n = textureSampleLevel(inTex, sampLinear, uv + vec2<f32>(0.0, -off.y), 0.0).rgb;
  let s = textureSampleLevel(inTex, sampLinear, uv + vec2<f32>(0.0,  off.y), 0.0).rgb;
  let w = textureSampleLevel(inTex, sampLinear, uv + vec2<f32>(-off.x, 0.0), 0.0).rgb;
  let e = textureSampleLevel(inTex, sampLinear, uv + vec2<f32>( off.x, 0.0), 0.0).rgb;
  let mn = min(min(min(n, s), w), e);
  let mx = max(max(max(n, s), w), e);
  let range = max(mx - mn, vec3<f32>(1e-6));
  let localConstrast = clamp((c - mn) / range, vec3<f32>(0.0), vec3<f32>(1.0));
  let att = rc.sharp.x; // attenuation parameter
  let detail = (n + s + w + e) * 0.25 - c;
  let sharpened = c + detail * (1.0 / att) * localConstrast;
  textureStore(outImg, vec2<i32>(gid.xy), vec4<f32>(sharpened, 1.0));
}
