/* ═══════════════════════════════════════════════════════════════════════
   schlieren.frag — Digital Schlieren / BOS WebGL Fragment Shader
   ═══════════════════════════════════════════════════════════════════════
   Implements Background-Oriented Schlieren (BOS) on the GPU.
   Visualises refractive-index gradients (density ripples) caused by
   gas concentration differences — the "invisible made visible."

   Two-pass design (managed from useWebGL.js):
     Pass 1 : render current camera frame → FBO A
     Pass 2 : this shader compares FBO A against the reference frame
              (stored in u_reference) and renders the Schlieren output

   Uniforms
   ────────
   u_current    : sampler2D — current camera frame
   u_reference  : sampler2D — reference (blank scene) frame
   u_sensitivity: float     — gradient amplification (default 4.0)
   u_time       : float     — elapsed seconds
   u_mode       : int       — 0=Schlieren only, 1=blend with RGB, 2=false-colour
   u_blend      : float     — RGB/Schlieren mix when mode=1
   ═══════════════════════════════════════════════════════════════════════ */

#version 300 es
precision highp float;

uniform sampler2D u_current;
uniform sampler2D u_reference;
uniform float     u_sensitivity;
uniform float     u_time;
uniform int       u_mode;
uniform float     u_blend;

in  vec2 v_uv;
out vec4 fragColor;

/* ── Luminance helper ─────────────────────────────────────────────── */
float luma(vec3 c) {
    return dot(c, vec3(0.299, 0.587, 0.114));
}

/* ── Sobel gradient magnitude ─────────────────────────────────────── */
float sobelMag(sampler2D tex, vec2 uv) {
    vec2 d = 1.0 / vec2(textureSize(tex, 0));
    float tl = luma(texture(tex, uv + vec2(-d.x,  d.y)).rgb);
    float tc = luma(texture(tex, uv + vec2( 0.0,  d.y)).rgb);
    float tr = luma(texture(tex, uv + vec2( d.x,  d.y)).rgb);
    float ml = luma(texture(tex, uv + vec2(-d.x,  0.0)).rgb);
    float mr = luma(texture(tex, uv + vec2( d.x,  0.0)).rgb);
    float bl = luma(texture(tex, uv + vec2(-d.x, -d.y)).rgb);
    float bc = luma(texture(tex, uv + vec2( 0.0, -d.y)).rgb);
    float br = luma(texture(tex, uv + vec2( d.x, -d.y)).rgb);
    float gx = -tl - 2.0*ml - bl + tr + 2.0*mr + br;
    float gy = -tl - 2.0*tc - tr + bl + 2.0*bc + br;
    return sqrt(gx*gx + gy*gy);
}

/* ── False-colour Schlieren palette (dark-blue → cyan → white) ────── */
vec3 schlierenColor(float v) {
    /* mimics real Schlieren photography colour scale */
    vec3 c;
    c.b = clamp(1.0 - v * 2.0, 0.0, 1.0);
    c.g = clamp(v * 2.0 - 0.3, 0.0, 1.0);
    c.r = clamp(v * 4.0 - 2.5, 0.0, 1.0);
    return c;
}

void main() {
    vec2 uv = v_uv;
    vec4 current   = texture(u_current,   uv);
    vec4 reference = texture(u_reference, uv);

    /* ── Temporal difference (BOS core) ─────────────────────────────── */
    float temporal = abs(luma(current.rgb) - luma(reference.rgb));

    /* ── Spatial Sobel gradient of current frame ─────────────────────── */
    float spatial  = sobelMag(u_current, uv);

    /* ── Combined gradient signal ────────────────────────────────────── */
    float combined = mix(spatial, temporal, 0.55);
    float enhanced = clamp(combined * u_sensitivity, 0.0, 1.0);

    /* ── Subtle noise shimmer to simulate real Schlieren "speckle" ────── */
    float speckle  = fract(sin(dot(uv * 100.0 + u_time * 0.1,
                                   vec2(12.9898, 78.233))) * 43758.5453);
    enhanced = mix(enhanced, enhanced + speckle * 0.02 * enhanced, 0.3);

    /* ── Output based on mode ──────────────────────────────────────────
       0 = pure Schlieren false-colour
       1 = blend with original RGB (like a "heat vision" overlay)
       2 = full false-colour + RGB blend                               */
    vec3 schlieren = schlierenColor(enhanced);
    vec3 result;

    if (u_mode == 0) {
        result = schlieren;
    } else if (u_mode == 1) {
        result = mix(current.rgb, schlieren, u_blend);
    } else {
        /* Mode 2: screen-blend so bright Schlieren regions appear on top */
        result = 1.0 - (1.0 - current.rgb) * (1.0 - schlieren * u_blend);
    }

    fragColor = vec4(result, 1.0);
}
