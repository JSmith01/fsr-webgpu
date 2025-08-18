// language=WGSL
const VERTEX_COUNT = 6;
const shaderCode = `
struct VSOut {
  @builtin(position) pos : vec4<f32>,
  @location(0) uv : vec2<f32>,
};

const positions = array<vec2<f32>,6>(
  vec2(-1.0,-1.0), vec2( 1.0,-1.0), vec2(-1.0, 1.0),
  vec2(-1.0, 1.0), vec2( 1.0,-1.0), vec2( 1.0, 1.0)
);

@vertex
fn vs(@builtin(vertex_index) vid: u32) -> VSOut {
  var out : VSOut;
  let p = positions[vid];
  out.pos = vec4(p, 0.0, 1.0);
  out.uv = vec2((p.x + 1.0) * 0.5, 1.0 - (p.y + 1.0) * 0.5);

  return out;
}

@group(0) @binding(0) var smp : sampler;
@group(0) @binding(1) var tex : texture_external;

const rgbToY = vec3<f32>(0.2126, 0.7152,  0.0722);
//const rgbToY = vec3<f32>(0.299, 0.587, 0.114);
const contrast = 1.2; // 1.0 = no contrast, 1.2 = 120% contrast

@fragment
fn fs(in: VSOut) -> @location(0) vec4<f32> {
  let c = textureSampleBaseClampToEdge(tex, smp, in.uv);
//  return c;
  let y = (dot(c.rgb, rgbToY) - 0.5) * contrast + 0.5;
  
  return vec4(vec3(y), 1.0);
}
`;

async function createProcessor() {
    if (!createProcessor.isSupported) {
        throw new Error('WebGPU and/or WebCodecs are not supported in this browser');
    }

    const adapter = await navigator.gpu.requestAdapter();
    if (!adapter) {
        throw new Error("No suitable GPU adapter found.");
    }

    const device = await adapter.requestDevice();
    const format = navigator.gpu.getPreferredCanvasFormat();

    const shader = device.createShaderModule({ code: shaderCode });

    const pipeline = device.createRenderPipeline({
        layout: 'auto',
        vertex: { module: shader, entryPoint: 'vs' },
        fragment: {
            module: shader,
            entryPoint: 'fs',
            targets: [{ format }]
        },
        primitive: { topology: 'triangle-list' }
    });

    const sampler = device.createSampler({
        magFilter: 'linear',
        minFilter: 'linear'
    });

    /** @param {HTMLCanvasElement|OffscreenCanvas} canvas */
    return function createProcessFrame(canvas) {
        const context = canvas.getContext('webgpu');
        context.configure({ device, format, alphaMode: 'opaque' });

        return processFrame;

        /** @param {VideoFrame} source */
        function processFrame(source) {
            const externalTex = device.importExternalTexture({ source });
            const encoder = device.createCommandEncoder();
            const view = context.getCurrentTexture().createView();
            const pass = encoder.beginRenderPass({
                colorAttachments: [{
                    view,
                    loadOp: 'clear',
                    clearValue: { r:0, g:0, b:0, a:1 },
                    storeOp: 'store'
                }]
            });
            const bindGroup = device.createBindGroup({
                layout: pipeline.getBindGroupLayout(0),
                entries: [
                    { binding: 0, resource: sampler },
                    { binding: 1, resource: externalTex }
                ]
            });
            pass.setPipeline(pipeline);
            pass.setBindGroup(0, bindGroup);
            pass.draw(VERTEX_COUNT);
            pass.end();
            device.queue.submit([encoder.finish()]);
        }
    }
}

createProcessor.isSupported = navigator.gpu && typeof VideoFrame === 'function';

export { createProcessor };
