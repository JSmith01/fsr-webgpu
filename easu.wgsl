struct EasuConsts { c0: vec4<f32>, c1: vec4<f32>, c2: vec4<f32>, c3: vec4<f32> };
@group(0) @binding(0) var<uniform> ec : EasuConsts;
@group(0) @binding(1) var sampLinear: sampler;
@group(0) @binding(2) var inTex: texture_2d<f32>;
@group(0) @binding(3) var outImg: texture_storage_2d<rgba8unorm, write>;

fn luminance(c: vec3<f32>) -> f32 { return dot(c, vec3<f32>(0.299,0.587,0.114)); }

@compute @workgroup_size(8,8,1)
fn easuMain(@builtin(global_invocation_id) gid: vec3<u32>) {
  let outSize = textureDimensions(outImg);
  if (gid.x >= outSize.x || gid.y >= outSize.y) { return; }
  // Map output pixel center to input uv
  let ox = f32(gid.x) + 0.5;
  let oy = f32(gid.y) + 0.5;
  let sx = ec.c0.x; let sy = ec.c0.y; let px = ec.c0.z; let py = ec.c0.w;
  let inPx = vec2<f32>(ox * sx + px, oy * sy + py);
  let base = inPx * vec2<f32>(ec.c1.x, ec.c1.y);
  let off = vec2<f32>(ec.c1.x, ec.c1.y);
  // Sample pattern (EASU taps). Real FSR uses 12 taps with directional weights; we approximate full adaptation.
  var taps: array<vec3<f32>, 12>;
  var lum: array<f32,12>;
  // Offsets chosen per reference (rotated grid)
  let o = array<vec2<f32>,12>(
    vec2<f32>(-1.0,-1.0), vec2<f32>(0.0,-1.0), vec2<f32>(1.0,-1.0),
    vec2<f32>(-1.0, 0.0), vec2<f32>(0.0, 0.0), vec2<f32>(1.0, 0.0),
    vec2<f32>(-1.0, 1.0), vec2<f32>(0.0, 1.0), vec2<f32>(1.0, 1.0),
    vec2<f32>(-2.0, 0.0), vec2<f32>(2.0, 0.0), vec2<f32>(0.0, 2.0)
  );
  for (var i: i32 = 0; i < 12; i = i + 1) {
    let suv = base + o[i] * off;
    let c = textureSampleLevel(inTex, sampLinear, suv, 0.0).rgb;
    taps[i] = c;
    lum[i] = luminance(c);
  }
  // Edge detection: compute gradients
  let lC = lum[4];
  let gx = (lum[5] - lum[3]) + 0.5*(lum[2]-lum[0]) + 0.5*(lum[8]-lum[6]);
  let gy = (lum[7] - lum[1]) + 0.5*(lum[6]-lum[0]) + 0.5*(lum[8]-lum[2]);
  let edgeMag = clamp(abs(gx) + abs(gy), 0.0, 1.0);
  // Directional weights: reduce along strongest gradient
  let dir = normalize(vec2<f32>(gx, gy) + 1e-6);

  // Weighted accumulation
  var accum = vec3<f32>(0.0);
  var wsum = 0.0;
  for (var i: i32 = 0; i < 9; i = i + 1) { // 3x3 core
    let delta = o[i];
    let d = dot(dir, normalize(vec2<f32>(delta.x, delta.y) + vec2<f32>(1e-6)));
    let aniso = 1.0 - edgeMag * abs(d);
    let dist2 = dot(delta, delta);
    let w = aniso * (1.0 / (1.0 + dist2));
    accum += taps[i] * w;
    wsum += w;
  }
  let color = accum / max(wsum, 1e-6);
  textureStore(outImg, vec2<i32>(gid.xy), vec4<f32>(color, 1.0));
}
