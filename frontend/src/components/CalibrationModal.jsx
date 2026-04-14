/**
 * CalibrationModal.jsx — 2-second blank-surface calibration flow
 *
 * Integration checklist item 3:
 *   "The app should ask the user to point at a clear area for 2 seconds
 *    to calibrate the rare earth sensor noise before it starts scanning."
 *
 * Shown automatically on first launch if the camera stream is active.
 * Posts collected frames to the spectral Web Worker for PCA background
 * subtraction baseline (see spectralWorker.js).
 */
import React, { useCallback, useEffect, useRef, useState } from 'react';

const DURATION_MS   = 2000;
const INTERVAL_MS   =  400;
const TOTAL_FRAMES  = DURATION_MS / INTERVAL_MS;  // 5 frames

export default function CalibrationModal({ videoRef, workerRef, onComplete, onDismiss }) {
  const [phase,    setPhase]    = useState('prompt');   // prompt | counting | done
  const [progress, setProgress] = useState(0);
  const timerRef = useRef(null);

  const startCapture = useCallback(() => {
    setPhase('counting');
    setProgress(0);
    let frames = 0;

    timerRef.current = setInterval(() => {
      frames++;
      setProgress(Math.round((frames / TOTAL_FRAMES) * 100));

      // Capture and send to spectral worker
      const video = videoRef?.current;
      if (video && video.videoWidth && workerRef?.current) {
        const tmp = document.createElement('canvas');
        tmp.width  = video.videoWidth;
        tmp.height = video.videoHeight;
        tmp.getContext('2d').drawImage(video, 0, 0);
        const imageData = tmp.getContext('2d').getImageData(0, 0, tmp.width, tmp.height);
        workerRef.current.postMessage(
          { type: 'calibrate', pixels: imageData.data, width: tmp.width, height: tmp.height },
          [imageData.data.buffer],
        );
      }

      if (frames >= TOTAL_FRAMES) {
        clearInterval(timerRef.current);
        setPhase('done');
        setTimeout(() => onComplete?.(), 800);
      }
    }, INTERVAL_MS);
  }, [videoRef, workerRef, onComplete]);

  useEffect(() => () => clearInterval(timerRef.current), []);

  const overlay = {
    position: 'fixed', inset: 0, zIndex: 10000,
    background: 'rgba(0,0,0,0.82)',
    display: 'flex', alignItems: 'center', justifyContent: 'center',
    fontFamily: 'monospace',
  };
  const box = {
    background: '#111118', border: '1px solid rgba(255,255,255,0.15)',
    borderRadius: 10, padding: '2rem', maxWidth: 380, width: '90%',
    color: '#e8e8f0', textAlign: 'center',
  };

  return (
    <div style={overlay}>
      <div style={box}>
        {phase === 'prompt' && (
          <>
            <div style={{ fontSize: '2rem', marginBottom: '0.5rem' }}>🔬</div>
            <h2 style={{ fontSize: '1rem', letterSpacing: '0.1em', textTransform: 'uppercase', marginBottom: '0.75rem' }}>
              Sensor Calibration Required
            </h2>
            <p style={{ fontSize: '0.8rem', color: '#8888a8', lineHeight: 1.6, marginBottom: '1.25rem' }}>
              Point the camera at a <strong style={{ color: '#e8e8f0' }}>blank surface</strong> (white wall,
              clear floor) and press <strong style={{ color: '#44ff88' }}>Start</strong>.
              The system will capture a 2-second background baseline to subtract
              normal rare-earth hardware noise from the chemical signal.
            </p>
            <div style={{ display: 'flex', gap: 8, justifyContent: 'center' }}>
              <button className="btn btn-green" onClick={startCapture}>▶ Start</button>
              <button className="btn btn-blue"  onClick={onDismiss}>Skip</button>
            </div>
          </>
        )}

        {phase === 'counting' && (
          <>
            <div style={{ fontSize: '2rem', marginBottom: '0.5rem', animation: 'blink-status 0.5s step-end infinite' }}>📡</div>
            <h2 style={{ fontSize: '1rem', letterSpacing: '0.1em', textTransform: 'uppercase', color: '#ffee44', marginBottom: '0.75rem' }}>
              Capturing Baseline…
            </h2>
            <div style={{
              background: '#1a1a2a', borderRadius: 6, overflow: 'hidden',
              height: 12, marginBottom: '0.75rem',
            }}>
              <div style={{
                height: '100%', background: '#44ff88',
                width: `${progress}%`, transition: 'width 0.35s linear',
              }} />
            </div>
            <p style={{ fontSize: '0.75rem', color: '#8888a8' }}>
              Hold steady… {progress}%
            </p>
          </>
        )}

        {phase === 'done' && (
          <>
            <div style={{ fontSize: '2rem', marginBottom: '0.5rem' }}>✅</div>
            <h2 style={{ fontSize: '1rem', letterSpacing: '0.1em', textTransform: 'uppercase', color: '#44ff88' }}>
              Calibration Complete
            </h2>
            <p style={{ fontSize: '0.78rem', color: '#8888a8', marginTop: '0.5rem' }}>
              Background noise baseline captured. Schlieren reference frame stored.
            </p>
          </>
        )}
      </div>
    </div>
  );
}
