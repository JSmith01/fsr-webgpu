precision mediump float;
varying vec2 v_texcoord;

// sharpness coefficient should be in [-.125, -.2]
const float cas_sharpness = -0.18;

uniform sampler2D i_texture;

uniform vec2 canvas_dims;
uniform vec2 texture_dims;

vec3 getRgb(vec2 coord) {
    return texture2D(i_texture, coord).xyz;
}

vec3 getCas(vec2 uv) {
    vec3 col = getRgb(uv);

    // CAS algorithm
    float max_g = col.y;
    float min_g = col.y;
    vec4 uvoff = vec4(1,0,1,-1) / texture_dims.xxyy;
    vec3 colw;

    // up
    vec3 col1 = getRgb(uv + uvoff.yw);
    max_g = max(max_g, col1.y);
    min_g = min(min_g, col1.y);
    colw = col1;

    // right
    col1 = getRgb(uv+uvoff.xy);
    max_g = max(max_g, col1.y);
    min_g = min(min_g, col1.y);
    colw += col1;

    // down
    col1 = getRgb(uv+uvoff.yz);
    max_g = max(max_g, col1.y);
    min_g = min(min_g, col1.y);
    colw += col1;

    // left
    col1 = getRgb(uv-uvoff.xy);
    max_g = max(max_g, col1.y);
    min_g = min(min_g, col1.y);
    colw += col1;

    float d_min_g = min_g;
    float d_max_g = 1.-max_g;

    float A;
    if (d_max_g < d_min_g) {
        A = d_max_g / max_g;
    } else {
        A = d_min_g / max_g;
    }
    A = sqrt(A) * cas_sharpness;

    vec3 col_out = (col + colw * A) / (1. + 4. * A);

    return col_out;
}

void main() {
    vec2 coord = gl_FragCoord.xy / canvas_dims * vec2(1.0, -1.0) + vec2(0.0, 1.0);
    gl_FragColor = vec4(getCas(coord), 1.0);
}
