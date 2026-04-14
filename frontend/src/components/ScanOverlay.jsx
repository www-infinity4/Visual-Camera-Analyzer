/**
 * ScanOverlay.jsx — Camera feed with WebGL OGI / Schlieren shaders
 *
 * Integration checklist items addressed here:
 *  ✔ Item 2 — WebGL shaders run in useWebGL hook (GPU, not CPU)
 *  ✔ Item 3 — Calibration flow: 2-second blank-surface capture
 *  ✔ Item 4 — Gemma-labelled plume text from /api/fusion
 */
import React, { useCallback, useRef, useState } from 'react';
import { useWebGL } from '../hooks/useWebGL.js';

const CALIBRATION_DURATION_MS = 2000;
const CALIBRATION_INTERVAL_MS = 500;

export default function ScanOverlay({ ogiActive, schlierenActive, fusionLabel }) {
  const videoRef     = useRef(null);
  const canvasRef    = useRef(null);
  const [stream, setStream]         = useState(null);
  const [calibState, setCalibState] = useState('idle');  // idle | calibrating | done
  const [calibPct,   setCalibPct]   = useState(0);
  const [refFrame,   setRefFrame]   = useState(null);

  const shaderMode = ogiActive       ? 'ogi'
                   : schlierenActive ? 'schlieren'
                   : 'off';

  const { isReady, error } = useWebGL(canvasRef, videoRef, shaderMode, refFrame);

  /* ── Start / stop camera ───────────────────────────────────────── */
  const toggleCamera = useCallback(async () => {
    if (stream) {
      stream.getTracks().forEach(t => t.stop());
      setStream(null);
      if (videoRef.current) videoRef.current.srcObject = null;
      return;
    }
    try {
      const s = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: 'environment', width: { ideal: 1280 }, height: { ideal: 720 } },
      });
      setStream(s);
      if (videoRef.current) videoRef.current.srcObject = s;
    } catch (e) {
      alert(`Camera error: ${e.message}`);
    }
  }, [stream]);

  /* ── Calibration — item 3 of checklist ────────────────────────── */
  const startCalibration = useCallback(() => {
    if (!stream) {
      alert('Start the camera first, then point it at a blank surface.');
      return;
    }
    setCalibState('calibrating');
    setCalibPct(0);

    let frames = 0;
    const total = CALIBRATION_DURATION_MS / CALIBRATION_INTERVAL_MS;

    const iv = setInterval(() => {
      frames++;
      setCalibPct(Math.round((frames / total) * 100));

      // Capture a reference frame for Schlieren BOS
      const video = videoRef.current;
      if (video && video.videoWidth) {
        const tmp    = document.createElement('canvas');
        tmp.width    = video.videoWidth;
        tmp.height   = video.videoHeight;
        tmp.getContext('2d').drawImage(video, 0, 0);
        setRefFrame(tmp);
      }

      if (frames >= total) {
        clearInterval(iv);
        setCalibState('done');
      }
    }, CALIBRATION_INTERVAL_MS);
  }, [stream]);

  return (
    <div style={{ position: 'relative', background: '#000', borderRadius: 8, overflow: 'hidden', aspectRatio: '16/9' }}>
      {/* Camera video */}
      <video
        ref={videoRef}
        autoPlay muted playsInline
        style={{ position: 'absolute', inset: 0, width: '100%', height: '100%', objectFit: 'cover' }}
      />

      {/* WebGL canvas overlay */}
      <canvas
        ref={canvasRef}
        style={{
          position: 'absolute', inset: 0,
          width: '100%', height: '100%',
          pointerEvents: 'none',
          opacity: (ogiActive || schlierenActive) ? 1 : 0,
          transition: 'opacity 0.4s ease',
        }}
      />

      {/* Gemma-labelled plume annotation — item 4 ─────────────────── */}
      {fusionLabel && (
        <div style={{
          position: 'absolute', top: 10, left: 10,
          background: 'rgba(0,0,0,0.72)',
          border: '1px solid rgba(255,255,255,0.2)',
          color: '#e8e8f0', fontFamily: 'monospace', fontSize: '0.72rem',
          padding: '4px 10px', borderRadius: 4,
          backdropFilter: 'blur(4px)',
        }}>
          {fusionLabel}
        </div>
      )}

      {/* Calibration banner ────────────────────────────────────────── */}
      {calibState === 'calibrating' && (
        <div style={{
          position: 'absolute', bottom: 0, left: 0, right: 0,
          background: 'rgba(255,180,0,0.9)', color: '#000',
          fontFamily: 'monospace', fontSize: '0.78rem',
          textAlign: 'center', padding: '0.4rem',
          letterSpacing: '0.06em', textTransform: 'uppercase',
        }}>
          ▶ CALIBRATING — hold on blank surface… {calibPct}%
        </div>
      )}
      {calibState === 'done' && (
        <div style={{
          position: 'absolute', bottom: 0, left: 0, right: 0,
          background: 'rgba(68,255,136,0.85)', color: '#000',
          fontFamily: 'monospace', fontSize: '0.78rem',
          textAlign: 'center', padding: '0.4rem',
          letterSpacing: '0.06em',
        }}>
          ✔ CALIBRATION COMPLETE — Schlieren reference captured
        </div>
      )}

      {/* WebGL not ready */}
      {error && (
        <div style={{
          position: 'absolute', top: 10, right: 10,
          background: 'rgba(255,50,50,0.8)', color: '#fff',
          fontFamily: 'monospace', fontSize: '0.68rem',
          padding: '3px 8px', borderRadius: 3,
        }}>
          ⚠ {error}
        </div>
      )}

      {/* Controls bar */}
      <div style={{
        position: 'absolute', bottom: calibState !== 'idle' ? 36 : 10, right: 10,
        display: 'flex', gap: 6,
      }}>
        <button className={`btn btn-blue${stream ? ' active' : ''}`} onClick={toggleCamera}>
          {stream ? '◼ Stop' : '⬤ Camera'}
        </button>
        {stream && (
          <button className="btn btn-orange" onClick={startCalibration}
            disabled={calibState === 'calibrating'}>
            Calibrate
          </button>
        )}
      </div>
    </div>
  );
}
