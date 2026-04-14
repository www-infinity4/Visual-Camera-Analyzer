/* ═══════════════════════════════════════════════════════════════════════
   shared.vert — Shared Vertex Shader for all WebGL passes
   ═══════════════════════════════════════════════════════════════════════
   A full-screen quad vertex shader used by ogi.frag and schlieren.frag.
   Renders a clip-space quad and passes texture coordinates to the
   fragment shader.
   ═══════════════════════════════════════════════════════════════════════ */

#version 300 es

in  vec2 a_position;
out vec2 v_uv;

void main() {
    /* a_position is in clip-space [-1,+1]; convert to UV [0,1] */
    v_uv        = a_position * 0.5 + 0.5;
    /* Flip Y: WebGL origin is bottom-left, canvas/video origin is top-left */
    v_uv.y      = 1.0 - v_uv.y;
    gl_Position = vec4(a_position, 0.0, 1.0);
}
