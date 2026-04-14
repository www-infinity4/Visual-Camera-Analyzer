/* ═══════════════════════════════════════════════════════════════════════
   ogi.frag — Optical Gas Imaging WebGL Fragment Shader
   ═══════════════════════════════════════════════════════════════════════
   Runs in a WebGL2 context on the camera <canvas> element.
   Performs per-pixel spectral band isolation and false-colour plume
   rendering entirely on the GPU — no CPU involvement per frame.

   Inputs (uniforms set from React useWebGL hook)
   ───────────────────────────────────────────────
   u_frame         : sampler2D  — current camera frame (RGB)
   u_time          : float      — elapsed seconds (for plume animation)
   u_threshold     : float      — absorption threshold 0–1  (default 0.25)
   u_blend_alpha   : float      — plume opacity 0–1         (default 0.55)
   u_plume_color   : vec3       — RGB plume colour 0–1
   u_anomaly_map   : sampler2D  — greyscale anomaly heat map from Python
   u_schlieren     : bool       — enable Schlieren density ripples overlay
   u_blur_radius   : float      — soft-edge kernel radius in UV units
   ═══════════════════════════════════════════════════════════════════════ */

#version 300 es
precision highp float;

uniform sampler2D u_frame;
uniform sampler2D u_anomaly_map;
uniform float     u_time;
uniform float     u_threshold;
uniform float     u_blend_alpha;
uniform vec3      u_plume_color;
uniform bool      u_schlieren;
uniform float     u_blur_radius;

in  vec2 v_uv;
out vec4 fragColor;

/* ── Gaussian soft-blur kernel (5-tap, separable approximation) ───── */
vec4 sampleBlurred(sampler2D tex, vec2 uv, float radius) {
    vec2 texelSize = radius / vec2(textureSize(tex, 0));
    vec4 result = vec4(0.0);
    float weights[5];
    weights[0] = 0.0625;
    weights[1] = 0.25;
    weights[2] = 0.375;
    weights[3] = 0.25;
    weights[4] = 0.0625;
    float offsets[5];
    offsets[0] = -2.0;
    offsets[1] = -1.0;
    offsets[2] =  0.0;
    offsets[3] =  1.0;
    offsets[4] =  2.0;
    for (int i = 0; i < 5; i++) {
        for (int j = 0; j < 5; j++) {
            vec2 offset = vec2(offsets[i], offsets[j]) * texelSize;
            result += texture(tex, uv + offset) * weights[i] * weights[j];
        }
    }
    return result;
}

/* ── Schlieren ripple pattern ─────────────────────────────────────── */
float schlierenRipple(vec2 uv, float t) {
    float ripple = sin(uv.x * 40.0 + t * 2.0) * 0.5 + 0.5;
    ripple      *= cos(uv.y * 30.0 - t * 1.5) * 0.5 + 0.5;
    return ripple * 0.06;  /* subtle — just a shimmer */
}

/* ── Screen blend mode ────────────────────────────────────────────── */
vec3 screenBlend(vec3 base, vec3 overlay) {
    return 1.0 - (1.0 - base) * (1.0 - overlay);
}

void main() {
    vec2 uv = v_uv;

    /* 1. Sample the camera frame */
    vec4 frameColor = texture(u_frame, uv);

    /* 2. Sample the anomaly heat-map (from Python OGI / noise-calibration) */
    float anomaly = sampleBlurred(u_anomaly_map, uv, u_blur_radius).r;

    /* 3. Apply threshold + soft clip */
    float gasStrength = clamp(
        (anomaly - u_threshold) / max(1.0 - u_threshold, 0.001),
        0.0, 1.0
    );

    /* 4. Plume drift animation — slight vertical oscillation */
    float drift = sin(uv.x * 8.0 + u_time * 0.8) * 0.012
                + cos(uv.y * 6.0 - u_time * 0.6) * 0.008;
    float animatedStrength = clamp(gasStrength + drift * gasStrength, 0.0, 1.0);

    /* 5. Build plume colour layer */
    vec3 plumeTint = u_plume_color * animatedStrength * u_blend_alpha;

    /* 6. Screen-blend plume over camera frame */
    vec3 blended = screenBlend(frameColor.rgb, plumeTint);

    /* 7. Optional Schlieren density-ripple overlay */
    if (u_schlieren && gasStrength > 0.05) {
        float ripple = schlierenRipple(uv + vec2(drift), u_time);
        blended = mix(blended, vec3(0.9, 0.95, 1.0), ripple * gasStrength);
    }

    /* 8. Vignette (darkens corners — makes it look like a lens filter) */
    vec2  centered  = uv - 0.5;
    float vignette  = 1.0 - dot(centered, centered) * 0.5;
    blended *= vignette;

    fragColor = vec4(clamp(blended, 0.0, 1.0), frameColor.a);
}
