/**
 * spectralWorker.js — Web Worker for heavy spectral math
 *
 * Offloads CPU-intensive operations from the main React thread:
 *   • PCA background subtraction (noise_calibration.py equivalent in JS)
 *   • Spectral Angle Mapper (SAM) matching against the chemical library
 *   • Anomaly map generation (for the OGI WebGL u_anomaly_map texture)
 *
 * Message protocol
 * ────────────────
 * Main → Worker:
 *   { type: 'calibrate',   pixels: Float32Array, width, height }
 *   { type: 'analyze',     pixels: Float32Array, width, height, targetChemical }
 *   { type: 'sam_match',   signature: Float32Array, library: Object }
 *
 * Worker → Main:
 *   { type: 'calibrated',  baseline: Float32Array }
 *   { type: 'anomaly_map', map: Float32Array, width, height }
 *   { type: 'sam_result',  compound: string, confidence: number }
 *   { type: 'error',       message: string }
 */

/* ── Rolling baseline state ──────────────────────────────────────── */
let baselineSum   = null;
let baselineCount = 0;
const BASELINE_FRAMES = 8;  // collect this many frames for calibration

/* ── SAM (Spectral Angle Mapper) ─────────────────────────────────── */
function samSimilarity(a, b) {
  let dot = 0, na = 0, nb = 0;
  for (let i = 0; i < a.length; i++) {
    dot += a[i] * b[i];
    na  += a[i] * a[i];
    nb  += b[i] * b[i];
  }
  const denom = Math.sqrt(na) * Math.sqrt(nb);
  return denom < 1e-9 ? 0 : dot / denom;
}

/* ── Luminance of a single RGBA pixel ────────────────────────────── */
function luma(pixels, offset) {
  return 0.299 * pixels[offset] + 0.587 * pixels[offset + 1] + 0.114 * pixels[offset + 2];
}

/* ── Build anomaly map from pixel delta ──────────────────────────── */
function buildAnomalyMap(currentPixels, baselinePixels, width, height) {
  const map = new Float32Array(width * height);
  let max = 0;
  for (let i = 0; i < width * height; i++) {
    const off = i * 4;
    const diff = Math.abs(luma(currentPixels, off) - luma(baselinePixels, off)) / 255;
    map[i] = diff;
    if (diff > max) max = diff;
  }
  // Normalise
  if (max > 0) for (let i = 0; i < map.length; i++) map[i] /= max;
  return map;
}

/* ── Message handler ─────────────────────────────────────────────── */
self.onmessage = (e) => {
  const { type } = e.data;

  try {
    if (type === 'calibrate') {
      const { pixels, width, height } = e.data;
      if (!baselineSum) {
        baselineSum = new Float32Array(pixels.length);
      }
      for (let i = 0; i < pixels.length; i++) baselineSum[i] += pixels[i];
      baselineCount++;

      if (baselineCount >= BASELINE_FRAMES) {
        const baseline = new Float32Array(pixels.length);
        for (let i = 0; i < pixels.length; i++) baseline[i] = baselineSum[i] / baselineCount;
        self.postMessage({ type: 'calibrated', baseline }, [baseline.buffer]);
        // Reset for next calibration cycle
        baselineSum   = null;
        baselineCount = 0;
      }

    } else if (type === 'analyze') {
      const { pixels, width, height } = e.data;
      if (!baselineSum) {
        // Not calibrated yet — emit flat map
        const map = new Float32Array(width * height);
        self.postMessage({ type: 'anomaly_map', map, width, height }, [map.buffer]);
        return;
      }
      const baseline = new Float32Array(pixels.length);
      for (let i = 0; i < pixels.length; i++) baseline[i] = baselineSum[i] / Math.max(baselineCount, 1);
      const map = buildAnomalyMap(pixels, baseline, width, height);
      self.postMessage({ type: 'anomaly_map', map, width, height }, [map.buffer]);

    } else if (type === 'sam_match') {
      const { signature, library } = e.data;
      let best = 'unknown', bestSim = -1;
      for (const [compound, refSig] of Object.entries(library)) {
        if (!refSig || refSig.length !== signature.length) continue;
        const sim = samSimilarity(signature, refSig);
        if (sim > bestSim) { bestSim = sim; best = compound; }
      }
      self.postMessage({ type: 'sam_result', compound: best, confidence: bestSim });
    }
  } catch (err) {
    self.postMessage({ type: 'error', message: err.message });
  }
};
