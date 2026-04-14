/**
 * useWebGL.js — React hook for OGI + Schlieren WebGL2 shader pipeline
 *
 * Manages the full GPU render loop:
 *   1. Loads GLSL sources from /static/glsl/
 *   2. Compiles + links the OGI and Schlieren programs
 *   3. Creates a full-screen quad VAO
 *   4. On each animation frame: uploads the video texture and renders
 *      the active shader (OGI, Schlieren, or passthrough)
 *
 * The heavy spectral math (SAM, PCA background subtraction) runs in
 * a Web Worker (see spectralWorker.js) and posts an anomaly map back
 * as an ImageBitmap, which is uploaded to the u_anomaly_map texture.
 *
 * @param {React.RefObject} canvasRef   The <canvas> element ref
 * @param {React.RefObject} videoRef    The <video> element ref
 * @param {'off'|'ogi'|'schlieren'} mode
 * @param {HTMLCanvasElement|null} referenceFrame  Calibration frame for BOS
 * @returns {{ isReady: boolean, error: string|null }}
 */
import { useEffect, useRef, useState } from 'react';
import {
  createProgram, createFullScreenQuad, createVideoTexture,
  uploadVideoFrame, renderFrame, setUniforms,
} from '../utils/webglUtils.js';

const SHADER_BASE = '/static/glsl';

async function loadShaderSource(name) {
  const r = await fetch(`${SHADER_BASE}/${name}`);
  if (!r.ok) throw new Error(`Cannot load shader: ${name}`);
  return r.text();
}

export function useWebGL(canvasRef, videoRef, mode, referenceFrame) {
  const [isReady, setIsReady] = useState(false);
  const [error,   setError]   = useState(null);

  const glRef       = useRef(null);
  const programsRef = useRef({});
  const quadRef     = useRef(null);
  const texRef      = useRef({});
  const rafRef      = useRef(null);
  const startTsRef  = useRef(performance.now());

  // ── Init: compile shaders once on mount ──────────────────────────
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const gl = canvas.getContext('webgl2');
    if (!gl) {
      setError('WebGL2 not supported in this browser.');
      return;
    }
    glRef.current = gl;

    Promise.all([
      loadShaderSource('shared.vert'),
      loadShaderSource('ogi.frag'),
      loadShaderSource('schlieren.frag'),
    ])
      .then(([vert, ogiFrag, schlFrag]) => {
        programsRef.current.ogi       = createProgram(gl, vert, ogiFrag);
        programsRef.current.schlieren = createProgram(gl, vert, schlFrag);

        // Create quad using the OGI program (they share the same vertex layout)
        quadRef.current = createFullScreenQuad(gl, programsRef.current.ogi);

        // Pre-create textures
        texRef.current.frame     = createVideoTexture(gl);
        texRef.current.reference = createVideoTexture(gl);
        texRef.current.anomaly   = createVideoTexture(gl);

        setIsReady(true);
      })
      .catch((e) => setError(e.message));

    return () => {
      if (rafRef.current) cancelAnimationFrame(rafRef.current);
    };
  }, [canvasRef]);

  // ── Render loop: restarts when mode changes ───────────────────────
  useEffect(() => {
    if (!isReady) return;
    const gl   = glRef.current;
    const quad = quadRef.current;

    if (rafRef.current) cancelAnimationFrame(rafRef.current);
    if (mode === 'off') return;

    const prog = programsRef.current[mode];
    if (!prog) return;

    const tick = () => {
      const video = videoRef.current;
      const canvas = canvasRef.current;
      if (!video || !canvas || video.readyState < 2) {
        rafRef.current = requestAnimationFrame(tick);
        return;
      }

      // Match canvas size to video
      if (canvas.width  !== video.videoWidth)  canvas.width  = video.videoWidth;
      if (canvas.height !== video.videoHeight) canvas.height = video.videoHeight;
      gl.viewport(0, 0, canvas.width, canvas.height);

      // Upload current video frame to texture unit 0
      gl.activeTexture(gl.TEXTURE0);
      uploadVideoFrame(gl, texRef.current.frame, video);

      // Upload reference frame (calibration) to texture unit 1
      gl.activeTexture(gl.TEXTURE1);
      if (referenceFrame) {
        uploadVideoFrame(gl, texRef.current.reference, referenceFrame);
      } else {
        // Use current frame as reference until calibrated
        uploadVideoFrame(gl, texRef.current.reference, video);
      }

      const elapsed = (performance.now() - startTsRef.current) / 1000;

      const uniforms =
        mode === 'ogi'
          ? {
              u_frame:       0,
              u_anomaly_map: 1,
              u_time:        elapsed,
              u_threshold:   0.25,
              u_blend_alpha: 0.55,
              u_plume_color: [1.0, 0.55, 0.08],   // orange (methane default)
              u_schlieren:   false,
              u_blur_radius: 8.0,
            }
          : {
              u_current:     0,
              u_reference:   1,
              u_sensitivity: 4.0,
              u_time:        elapsed,
              u_mode:        2,       // screen-blend with RGB
              u_blend:       0.6,
            };

      gl.useProgram(prog);
      // Manually bind textures (setUniforms handles the int uniforms)
      const u0 = gl.getUniformLocation(prog, mode === 'ogi' ? 'u_frame' : 'u_current');
      const u1 = gl.getUniformLocation(prog, mode === 'ogi' ? 'u_anomaly_map' : 'u_reference');
      if (u0) gl.uniform1i(u0, 0);
      if (u1) gl.uniform1i(u1, 1);

      renderFrame(gl, prog, quad, uniforms);
      rafRef.current = requestAnimationFrame(tick);
    };

    rafRef.current = requestAnimationFrame(tick);
    return () => { if (rafRef.current) cancelAnimationFrame(rafRef.current); };
  }, [isReady, mode, referenceFrame, canvasRef, videoRef]);

  return { isReady, error };
}
