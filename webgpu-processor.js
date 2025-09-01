// webgpu-processor.js
const DEBUG = (typeof window !== 'undefined' && (new URLSearchParams(window.location.search).has('debug') || window.DEBUG_WEBGPU)) || false;
const UPSCALE_FACTOR = 2; // has to be integer

// language=WGSL
const upscaleShaderCode = `
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
@group(0) @binding(2) var<storage, read_write> size : vec2<u32>;

fn FsrEasuCF(coord: vec2<f32>) -> vec3<f32> {
    return textureSampleBaseClampToEdge(tex, smp, coord).rgb;
}

// ---- Pack EASU constants (no swizzles used) ----
struct EasuConsts {
    con0: vec4<f32>, // xy = output->input scale, zw = first pixel offset correction
    con1: vec4<f32>, // (1/x, 1/y, 1/x, -1/y)
    con2: vec4<f32>, // (-1/x, 2/y, 1/x, 2/y)
    con3: vec4<f32>, // (0, 4/y, 0, 0)
};

fn FsrEasuCon(
    inputViewportInPixels: vec2<f32>, // source (pre-upscale) size
    inputSizeInPixels: vec2<f32>,     // same as above in this use
    outputSizeInPixels: vec2<f32>     // target size (2x)
) -> EasuConsts {
    let con0 = vec4<f32>(
        inputViewportInPixels.x / outputSizeInPixels.x,
        inputViewportInPixels.y / outputSizeInPixels.y,
        0.5 * inputViewportInPixels.x / outputSizeInPixels.x - 0.5,
        0.5 * inputViewportInPixels.y / outputSizeInPixels.y - 0.5
    );

    // Replace vec4(1,1,1,-1)/inputSize.xyxy explicitly:
    let con1 = vec4<f32>(
        1.0 / inputSizeInPixels.x,
        1.0 / inputSizeInPixels.y,
        1.0 / inputSizeInPixels.x,
       -1.0 / inputSizeInPixels.y
    );

    let con2 = vec4<f32>(
       -1.0 / inputSizeInPixels.x,
        2.0 / inputSizeInPixels.y,
        1.0 / inputSizeInPixels.x,
        2.0 / inputSizeInPixels.y
    );

    let con3 = vec4<f32>(
        0.0,
        4.0 / inputSizeInPixels.y,
        0.0,
        0.0
    );

    return EasuConsts(con0, con1, con2, con3);
}

// ---- Tap accumulation (no swizzles) ----
fn FsrEasuTapF(
    aC: ptr<function, vec3<f32>>,
    aW: ptr<function, f32>,
    off: vec2<f32>,
    dir: vec2<f32>,
    len: vec2<f32>,
    lob: f32,
    clp: f32,
    c: vec3<f32>
) {
    // Rotate offset by direction
    let v = vec2<f32>(
        dot(off, dir),
        dot(off, vec2<f32>(-dir.y, dir.x))
    );
    // Anisotropy
    let va = v * len;
    // Distance^2 (clipped)
    let d2 = min(dot(va, va), clp);

    // Approximate lanczos2 windowed polynomial
    var wB = 0.4 * d2 - 1.0;
    var wA = lob * d2 - 1.0;
    wB = wB * wB;
    wA = wA * wA;
    wB = 1.5625 * wB - 0.5625;
    let w = wB * wA;

    (*aC) = (*aC) + c * w;
    (*aW) = (*aW) + w;
}

// ---- Accumulate direction and length ----
fn FsrEasuSetF(
    dir: ptr<function, vec2<f32>>,
    len: ptr<function, f32>,
    w: f32,
    lA: f32, lB: f32, lC: f32, lD: f32, lE: f32
) {
    // X axis
    var lenX = max(abs(lD - lC), abs(lC - lB));
    let dirX = lD - lB;
    (*dir).x = (*dir).x + dirX * w;
    lenX = clamp(abs(dirX) / lenX, 0.0, 1.0);
    lenX = lenX * lenX;
    (*len) = (*len) + lenX * w;

    // Y axis
    var lenY = max(abs(lE - lC), abs(lC - lA));
    let dirY = lE - lA;
    (*dir).y = (*dir).y + dirY * w;
    lenY = clamp(abs(dirY) / lenY, 0.0, 1.0);
    lenY = lenY * lenY;
    (*len) = (*len) + lenY * w;
}

// ---- Main EASU filter (full expansion, no swizzles) ----
fn FsrEasuF(ip: vec2<f32>, consts: EasuConsts) -> vec3<f32> {
    let con0 = consts.con0;
    let con1 = consts.con1;
    let con2 = consts.con2;
    let con3 = consts.con3;

    // Input pixel/subpixel
    var pp = ip * vec2<f32>(con0.x, con0.y) + vec2<f32>(con0.z, con0.w);
    let fp = floor(pp);
    pp = pp - fp;

    // Upper-left of 'F' tap
    let p0 = fp * vec2<f32>(con1.x, con1.y) + vec2<f32>(con1.z, con1.w);

    // Derived centers
    let p1 = p0 + vec2<f32>(con2.x, con2.y);
    let p2 = p0 + vec2<f32>(con2.z, con2.w);
    let p3 = p0 + vec2<f32>(con3.x, con3.y);

    // Offsets (replacing vec4(-.5,.5,-.5,.5) * con1.xxyy)
    let off = vec4<f32>(
        -0.5 * con1.x, // x uses 1/x
         0.5 * con1.y, // y uses 1/y
        -0.5 * con1.x, // z uses 1/x
         0.5 * con1.y  // w uses 1/y
    );

    // Sample taps (explicit vec2 builds instead of swizzles)
    let bC = FsrEasuCF(p0 + vec2<f32>(off.x, off.w)); let bL = bC.g + 0.5 * (bC.r + bC.b);
    let cC = FsrEasuCF(p0 + vec2<f32>(off.y, off.w)); let cL = cC.g + 0.5 * (cC.r + cC.b);
    let iC = FsrEasuCF(p1 + vec2<f32>(off.x, off.w)); let iL = iC.g + 0.5 * (iC.r + iC.b);
    let jC = FsrEasuCF(p1 + vec2<f32>(off.y, off.w)); let jL = jC.g + 0.5 * (jC.r + jC.b);
    let fC = FsrEasuCF(p1 + vec2<f32>(off.y, off.z)); let fL = fC.g + 0.5 * (fC.r + fC.b);
    let eC = FsrEasuCF(p1 + vec2<f32>(off.x, off.z)); let eL = eC.g + 0.5 * (eC.r + eC.b);
    let kC = FsrEasuCF(p2 + vec2<f32>(off.x, off.w)); let kL = kC.g + 0.5 * (kC.r + kC.b);
    let lC = FsrEasuCF(p2 + vec2<f32>(off.y, off.w)); let lL = lC.g + 0.5 * (lC.r + lC.b);
    let hC = FsrEasuCF(p2 + vec2<f32>(off.y, off.z)); let hL = hC.g + 0.5 * (hC.r + hC.b);
    let gC = FsrEasuCF(p2 + vec2<f32>(off.x, off.z)); let gL = gC.g + 0.5 * (gC.r + gC.b);
    let oC = FsrEasuCF(p3 + vec2<f32>(off.y, off.z)); let oL = oC.g + 0.5 * (oC.r + oC.b);
    let nC = FsrEasuCF(p3 + vec2<f32>(off.x, off.z)); let nL = nC.g + 0.5 * (nC.r + nC.b);

    // Gradient accumulation
    var dir = vec2<f32>(0.0, 0.0);
    var len: f32 = 0.0;

    FsrEasuSetF(&dir, &len, (1.0 - pp.x) * (1.0 - pp.y), bL, eL, fL, gL, jL);
    FsrEasuSetF(&dir, &len,      pp.x     * (1.0 - pp.y), cL, fL, gL, hL, kL);
    FsrEasuSetF(&dir, &len, (1.0 - pp.x) *      pp.y    , fL, iL, jL, kL, nL);
    FsrEasuSetF(&dir, &len,      pp.x     *      pp.y    , gL, jL, kL, lL, oL);

    // Normalize dir with safe handling near zero
    let dir2 = dir * dir;
    var lenSq = dir2.x + dir2.y;
    var invLen: f32;
    if (lenSq < (1.0 / 32768.0)) {
        invLen = 1.0;
        dir.x = 1.0;
    } else {
        invLen = inverseSqrt(lenSq);
    }
    dir = dir * vec2<f32>(invLen, invLen);

    // Shape anisotropy
    len = len * 0.5;
    len = len * len;
    let stretch = dot(dir, dir) / max(abs(dir.x), abs(dir.y));
    let len2 = vec2<f32>(1.0 + (stretch - 1.0) * len, 1.0 - 0.5 * len);
    let lob = 0.5 - 0.29 * len;
    let clp = 1.0 / lob;

    // Min/max of the 4 nearest for deringing
    let min4 = min(min(fC, gC), min(jC, kC));
    let max4 = max(max(fC, gC), max(jC, kC));

    // Accumulate weighted taps
    var aC = vec3<f32>(0.0, 0.0, 0.0);
    var aW: f32 = 0.0;

    FsrEasuTapF(&aC, &aW, vec2<f32>( 0.0, -1.0) - pp, dir, len2, lob, clp, bC);
    FsrEasuTapF(&aC, &aW, vec2<f32>( 1.0, -1.0) - pp, dir, len2, lob, clp, cC);
    FsrEasuTapF(&aC, &aW, vec2<f32>(-1.0,  1.0) - pp, dir, len2, lob, clp, iC);
    FsrEasuTapF(&aC, &aW, vec2<f32>( 0.0,  1.0) - pp, dir, len2, lob, clp, jC);
    FsrEasuTapF(&aC, &aW, vec2<f32>( 0.0,  0.0) - pp, dir, len2, lob, clp, fC);
    FsrEasuTapF(&aC, &aW, vec2<f32>(-1.0,  0.0) - pp, dir, len2, lob, clp, eC);
    FsrEasuTapF(&aC, &aW, vec2<f32>( 1.0,  1.0) - pp, dir, len2, lob, clp, kC);
    FsrEasuTapF(&aC, &aW, vec2<f32>( 2.0,  1.0) - pp, dir, len2, lob, clp, lC);
    FsrEasuTapF(&aC, &aW, vec2<f32>( 2.0,  0.0) - pp, dir, len2, lob, clp, hC);
    FsrEasuTapF(&aC, &aW, vec2<f32>( 1.0,  0.0) - pp, dir, len2, lob, clp, gC);
    FsrEasuTapF(&aC, &aW, vec2<f32>( 1.0,  2.0) - pp, dir, len2, lob, clp, oC);
    FsrEasuTapF(&aC, &aW, vec2<f32>( 0.0,  2.0) - pp, dir, len2, lob, clp, nC);

    // Normalize & dering
    let result = min(max4, max(min4, aC / aW));
    return result;
}

// ---- Fragment entry ----
@fragment
fn fs(@builtin(position) fragCoord: vec4<f32>) -> @location(0) vec4<f32> {
    // source_dims from texture, target_dims = 2x
    let source_dims = vec2<f32>(textureDimensions(tex));
    let target_dims = source_dims * 2.0;

    // Build EASU constants and run filter
    let consts = FsrEasuCon(source_dims, source_dims, target_dims);
    let rgb = FsrEasuF(fragCoord.xy, consts);

    return vec4<f32>(rgb, 1.0);
}
`;

// language=WGSL
const rcasShaderCode = `
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
@group(0) @binding(1) var upTex : texture_2d<f32>;

// WGSL fragment shader version of your GLSL CAS filter

// Sharpness coefficient should be in [-0.125, -0.2]
const cas_sharpness: f32 = -0.2;

// Get RGB from texture
fn getRgb(coord: vec2<f32>) -> vec3<f32> {
    return textureSampleLevel(upTex, smp, coord, 0.0).xyz;
}

// CAS algorithm
fn getCas(uv: vec2<f32>) -> vec3<f32> {
    var col = getRgb(uv);

    var max_g = col.y;
    var min_g = col.y;

    // Query texture size at mip 0
    let texSize: vec2<u32> = textureDimensions(upTex, 0u);
    let texSizeF: vec2<f32> = vec2<f32>(texSize);

    // Equivalent to vec4(1,0,1,-1) / texture_dims.xxyy in GLSL
    let uvoff = vec4<f32>(
        1.0 / texSizeF.x,
        0.0,
        1.0 / texSizeF.x,
       -1.0 / texSizeF.y
    );

    var colw: vec3<f32>;

    // up
    var col1 = getRgb(uv + uvoff.yw);
    max_g = max(max_g, col1.y);
    min_g = min(min_g, col1.y);
    colw = col1;

    // right
    col1 = getRgb(uv + uvoff.xy);
    max_g = max(max_g, col1.y);
    min_g = min(min_g, col1.y);
    colw += col1;

    // down
    col1 = getRgb(uv + uvoff.yz);
    max_g = max(max_g, col1.y);
    min_g = min(min_g, col1.y);
    colw += col1;

    // left
    col1 = getRgb(uv - uvoff.xy);
    max_g = max(max_g, col1.y);
    min_g = min(min_g, col1.y);
    colw += col1;

    let d_min_g = min_g;
    let d_max_g = 1.0 - max_g;

    var A: f32;
    if (d_max_g < d_min_g) {
        A = d_max_g / max_g;
    } else {
        A = d_min_g / max_g;
    }
    A = sqrt(A) * cas_sharpness;

    let col_out = (col + colw * A) / (1.0 + 4.0 * A);

    return col_out;
}

@fragment
fn fs(@builtin(position) fragCoord: vec4<f32>) -> @location(0) vec4<f32> {
    // Normalize pixel coords to [0,1] range
    let texSize: vec2<u32> = textureDimensions(upTex, 0u);
    let texSizeF: vec2<f32> = vec2<f32>(texSize);

    let coord = fragCoord.xy / texSizeF;
    return vec4<f32>(getCas(coord), 1.0);
}
`;

const VERTEX_COUNT = 6;

async function createProcessor() {
    if (DEBUG) console.log('[WebGPU] createProcessor start');

    if (!createProcessor.isSupported) throw new Error('WebGPU/WebCodecs unsupported');

    const adapter = await navigator.gpu.requestAdapter();
    if (!adapter) throw new Error('No adapter');

    if (DEBUG) console.log('[WebGPU] Adapter:', adapter);

    const device = await adapter.requestDevice();
    if (DEBUG) {
        console.log('[WebGPU] Device acquired');
        device.addEventListener('uncapturederror', (e) => {
            console.error('[WebGPU] Uncaptured error:', e.error || e);
        });
        device.lost.then((info) => {
            console.error('[WebGPU] Device lost:', info);
        });
    }
    const format = navigator.gpu.getPreferredCanvasFormat();
    if (DEBUG) console.log('[WebGPU] Preferred canvas format:', format);

    // Shaders & pipelines
    const upscaleModule = device.createShaderModule({ code: upscaleShaderCode });
    const grayscaleModule = device.createShaderModule({ code: rcasShaderCode });
    if (DEBUG) {
        try {
            const info1 = await upscaleModule.getCompilationInfo();
            if (info1.messages?.length) console.log('[WGSL] upscaleModule info:', info1.messages);

            const info2 = await grayscaleModule.getCompilationInfo();
            if (info2.messages?.length) console.log('[WGSL] grayscaleModule info:', info2.messages);
        } catch (e) {
            console.warn('[WGSL] getCompilationInfo not available or failed:', e);
        }
    }

    const upscalePipeline = device.createRenderPipeline({
        layout: 'auto',
        vertex: { module: upscaleModule, entryPoint: 'vs' },
        fragment: { module: upscaleModule, entryPoint: 'fs', targets: [{ format: 'rgba8unorm' }] },
        primitive: { topology: 'triangle-list' }
    });

    const grayscalePipeline = device.createRenderPipeline({
        layout: 'auto',
        vertex: { module: grayscaleModule, entryPoint: 'vs' },
        fragment: {
            module: grayscaleModule,
            entryPoint: 'fs',
            targets: [{ format }]
        },
        primitive: { topology: 'triangle-list' }
    });

    const sampler = device.createSampler({
        magFilter: 'linear',
        minFilter: 'linear'
    });

    // Uniform buffer for source size (vec2<f32>)
    const upscaleUniformBuffer = device.createBuffer({
        size: 8,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
    });

    // Intermediate (will be (srcW*2, srcH*2))
    let intermediateTex = null;
    let currentSrcW = 0, currentSrcH = 0;

    function ensureIntermediate(srcW, srcH) {
            if (DEBUG) console.log('[WebGPU] ensureIntermediate src', srcW, srcH);
        if (srcW === currentSrcW && srcH === currentSrcH && intermediateTex) return;

        currentSrcW = srcW; currentSrcH = srcH;
        if (intermediateTex) intermediateTex.destroy?.();
        intermediateTex = device.createTexture({
            size: { width: srcW * UPSCALE_FACTOR, height: srcH * UPSCALE_FACTOR },
            format: 'rgba8unorm',
            usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING
        });

        if (DEBUG) console.log('[WebGPU] Recreated intermediate texture', intermediateTex);
    }

    /** @param {HTMLCanvasElement|OffscreenCanvas} canvas */
    return function createProcessFrame(canvas) {
        if (DEBUG) console.log('[WebGPU] createProcessFrame for canvas', canvas.width, canvas.height);

        const context = canvas.getContext('webgpu');
        context.configure({ device, format, alphaMode: 'opaque' });

        return processFrame;

        /** @param {VideoFrame} source */
        function processFrame(source) {
            if (DEBUG) console.log('[WebGPU] processFrame source', { w: source.displayWidth || source.codedWidth, h: source.displayHeight || source.codedHeight, timestamp: source.timestamp });

            const srcW = source.displayWidth || source.codedWidth;
            const srcH = source.displayHeight || source.codedHeight;
            ensureIntermediate(srcW, srcH);

            const dstW = srcW * UPSCALE_FACTOR;
            const dstH = srcH * UPSCALE_FACTOR;

            if (canvas.width !== dstW || canvas.height !== dstH) {
                if (DEBUG) console.log('[WebGPU] Resize canvas', canvas.width, canvas.height, '->', dstW, dstH);

                canvas.width = dstW;
                canvas.height = dstH;
            }

            device.queue.writeBuffer(upscaleUniformBuffer, 0, new Float32Array([srcW, srcH]));

            const externalTex = device.importExternalTexture({ source });
            if (DEBUG && (!srcW || !srcH)) console.warn('[WebGPU] Source dimensions invalid:', srcW, srcH);

            const encoder = device.createCommandEncoder({ label: 'main-encoder' });

            // Pass 1: upscale to intermediate
            {
                const pass = encoder.beginRenderPass({
                    colorAttachments: [{
                        view: intermediateTex.createView(),
                        loadOp: 'clear',
                        clearValue: DEBUG ? { r:0, g:1, b:0, a:1 } : { r:0, g:0, b:0, a:1 }, // green to verify pass 1
                        storeOp: 'store'
                    }]
                });
                const bindGroup = device.createBindGroup({
                    layout: upscalePipeline.getBindGroupLayout(0),
                    entries: [
                        { binding: 0, resource: sampler },
                        { binding: 1, resource: externalTex },
                    ]
                });
                pass.setPipeline(upscalePipeline);
                pass.setBindGroup(0, bindGroup);
                pass.draw(VERTEX_COUNT);
                pass.end();
            }

            // Pass 2: grayscale to canvas
            {
                const currentTex = context.getCurrentTexture();
                if (DEBUG && !currentTex) console.warn('[WebGPU] getCurrentTexture() returned null');

                const pass = encoder.beginRenderPass({
                    colorAttachments: [{
                        view: currentTex.createView(),
                        loadOp: 'clear',
                        clearValue: DEBUG ? { r:1, g:0, b:1, a:1 } : { r:0, g:0, b:0, a:1 }, // magenta to verify presentation
                        storeOp: 'store'
                    }]
                });
                const bindGroup = device.createBindGroup({
                    layout: grayscalePipeline.getBindGroupLayout(0),
                    entries: [
                        { binding: 0, resource: sampler },
                        { binding: 1, resource: intermediateTex.createView() }
                    ]
                });
                pass.setPipeline(grayscalePipeline);
                pass.setBindGroup(0, bindGroup);
                pass.draw(VERTEX_COUNT);
                pass.end();
            }

            device.queue.submit([encoder.finish()]);
            if (DEBUG) device.queue.onSubmittedWorkDone().then(() => console.log('[WebGPU] Frame submitted'));
        }
    }
}

createProcessor.isSupported = !!navigator.gpu && typeof VideoFrame === 'function';

export { createProcessor };
