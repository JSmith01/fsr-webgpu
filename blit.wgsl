@group(0) @binding(0) var smp: sampler;
@group(0) @binding(1) var tex: texture_2d<f32>;

struct VSOut {
  @builtin(position) pos : vec4<f32>,
  @location(0) uv : vec2<f32>,
};

@vertex fn vs(@builtin(vertex_index) i:u32)->VSOut {
  let p = array<vec2<f32>,6>(
    vec2<f32>(-1.,-1.), vec2<f32>(1.,-1.), vec2<f32>(-1.,1.),
    vec2<f32>(-1.,1.), vec2<f32>(1.,-1.), vec2<f32>(1.,1.)
  );
  let uv = (p[i] * 0.5) + vec2<f32>(0.5,0.5);

  return VSOut(vec4<f32>(p[i],0,1), uv);
}

@fragment fn fs(in:VSOut)->@location(0) vec4<f32> {
  return textureSample(tex, smp, in.uv);
}
