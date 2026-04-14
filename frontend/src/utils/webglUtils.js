/**
 * webglUtils.js — WebGL2 helper utilities for OGI + Schlieren shaders
 *
 * Handles shader compilation, program linking, full-screen quad setup,
 * texture upload from HTMLVideoElement, and the render loop.
 *
 * Usage (from useWebGL hook):
 *   const gl  = canvas.getContext('webgl2');
 *   const ogi = createProgram(gl, vertSrc, ogiFrag);
 *   const tex = createVideoTexture(gl);
 *   renderFrame(gl, ogi, tex, video, uniforms);
 */

/**
 * Compile a WebGL2 shader.
 * @param {WebGL2RenderingContext} gl
 * @param {number} type  gl.VERTEX_SHADER | gl.FRAGMENT_SHADER
 * @param {string} src   GLSL source
 */
export function compileShader(gl, type, src) {
  const shader = gl.createShader(type);
  gl.shaderSource(shader, src);
  gl.compileShader(shader);
  if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
    const info = gl.getShaderInfoLog(shader);
    gl.deleteShader(shader);
    throw new Error(`Shader compile error:\n${info}`);
  }
  return shader;
}

/**
 * Link a WebGL2 program from pre-compiled shaders.
 */
export function linkProgram(gl, vert, frag) {
  const prog = gl.createProgram();
  gl.attachShader(prog, vert);
  gl.attachShader(prog, frag);
  gl.linkProgram(prog);
  if (!gl.getProgramParameter(prog, gl.LINK_STATUS)) {
    const info = gl.getProgramInfoLog(prog);
    gl.deleteProgram(prog);
    throw new Error(`Program link error:\n${info}`);
  }
  return prog;
}

/**
 * Compile + link a full shader program from GLSL source strings.
 */
export function createProgram(gl, vertSrc, fragSrc) {
  const vert = compileShader(gl, gl.VERTEX_SHADER,   vertSrc);
  const frag = compileShader(gl, gl.FRAGMENT_SHADER, fragSrc);
  return linkProgram(gl, vert, frag);
}

/**
 * Set up a full-screen quad VAO.
 * Returns { vao, buffer } — call gl.bindVertexArray(vao) before drawing.
 */
export function createFullScreenQuad(gl, program) {
  const positions = new Float32Array([-1,-1,  1,-1,  -1,1,  1,1]);
  const buf = gl.createBuffer();
  gl.bindBuffer(gl.ARRAY_BUFFER, buf);
  gl.bufferData(gl.ARRAY_BUFFER, positions, gl.STATIC_DRAW);

  const vao    = gl.createVertexArray();
  gl.bindVertexArray(vao);
  const posLoc = gl.getAttribLocation(program, 'a_position');
  gl.enableVertexAttribArray(posLoc);
  gl.vertexAttribPointer(posLoc, 2, gl.FLOAT, false, 0, 0);
  gl.bindVertexArray(null);
  return { vao, buffer: buf };
}

/**
 * Create a WebGL2 texture suitable for video frame upload.
 */
export function createVideoTexture(gl) {
  const tex = gl.createTexture();
  gl.bindTexture(gl.TEXTURE_2D, tex);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S,     gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T,     gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
  return tex;
}

/**
 * Upload a video frame or canvas to a WebGL texture.
 * @param {WebGL2RenderingContext} gl
 * @param {WebGLTexture} tex
 * @param {HTMLVideoElement|HTMLCanvasElement|ImageBitmap} source
 */
export function uploadVideoFrame(gl, tex, source) {
  gl.bindTexture(gl.TEXTURE_2D, tex);
  gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, source);
}

/**
 * Set a collection of uniforms on the active program.
 * @param {WebGL2RenderingContext} gl
 * @param {WebGLProgram} prog
 * @param {Object} uniforms  { name: value }
 *   value can be: number, boolean, [number,number,number] (vec3), or WebGLTexture
 */
export function setUniforms(gl, prog, uniforms) {
  for (const [name, value] of Object.entries(uniforms)) {
    const loc = gl.getUniformLocation(prog, name);
    if (loc === null) continue;
    if (Array.isArray(value)) {
      if (value.length === 2) gl.uniform2fv(loc, value);
      else if (value.length === 3) gl.uniform3fv(loc, value);
      else if (value.length === 4) gl.uniform4fv(loc, value);
    } else if (typeof value === 'boolean') {
      gl.uniform1i(loc, value ? 1 : 0);
    } else if (typeof value === 'number') {
      // Distinguish int uniforms by name convention: u_*i, u_mode, u_schlieren
      if (name.endsWith('_i') || name === 'u_mode' || name === 'u_frame'
          || name === 'u_anomaly_map' || name === 'u_reference'
          || name === 'u_schlieren') {
        gl.uniform1i(loc, value);
      } else {
        gl.uniform1f(loc, value);
      }
    }
  }
}

/**
 * Render one frame using a shader program.
 * @param {WebGL2RenderingContext} gl
 * @param {WebGLProgram} prog
 * @param {{ vao: WebGLVertexArrayObject }} quad
 * @param {Object} uniforms
 */
export function renderFrame(gl, prog, quad, uniforms) {
  gl.useProgram(prog);
  gl.bindVertexArray(quad.vao);
  setUniforms(gl, prog, uniforms);
  gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
  gl.bindVertexArray(null);
}
