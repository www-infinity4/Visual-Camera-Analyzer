/**
 * App.jsx — REE Digital Twin · Chemical Vision Analyzer
 *
 * Top-level React application.  Wires together:
 *   • useElementStream  — SSE live data from /stream/elements
 *   • ScanOverlay       — camera feed + WebGL OGI/Schlieren shaders
 *   • CalibrationModal  — 2-second blank-surface baseline capture
 *   • FusionPanel       — Gemma 2 / SAM fusion result + label
 *   • ElementCard grid  — 17 live REE Digital Twin cards
 *   • PointCloud        — 3D emission point cloud
 *   • spectralWorker    — Web Worker for off-thread SAM / PCA math
 *
 * Performance strategy (addresses "frame rate" concern):
 *   1. WebGL shaders run on GPU (OGI / Schlieren) — never on main thread.
 *   2. SAM + PCA background subtraction run in spectralWorker.js Web Worker.
 *   3. React element-grid updates are batched via useTransition so the
 *      17-card render never blocks animation frames.
 *   4. ElementCard is wrapped in React.memo — only re-renders if data changes.
 */
import React, {
  useCallback, useEffect, useMemo, useRef, useState, useTransition,
} from 'react';
import { useElementStream } from './hooks/useElementStream.js';
import ElementCard          from './components/ElementCard.jsx';
import FusionPanel          from './components/FusionPanel.jsx';
import PointCloud           from './components/PointCloud.jsx';
import ScanOverlay          from './components/ScanOverlay.jsx';
import CalibrationModal     from './components/CalibrationModal.jsx';
import axios                from 'axios';

// ── Web Worker ───────────────────────────────────────────────────────
let spectralWorkerInstance = null;
function getWorker() {
  if (!spectralWorkerInstance) {
    spectralWorkerInstance = new Worker(
      new URL('./workers/spectralWorker.js', import.meta.url),
      { type: 'module' }
    );
  }
  return spectralWorkerInstance;
}

// ── Ammonium readings helper ──────────────────────────────────────────
async function fetchAmmonia(mode, ppm = 15) {
  const { data } = await axios.get('/api/ammonia', { params: { mode, ppm } });
  return data;
}

// ────────────────────────────────────────────────────────────────────
export default function App() {
  const { elements, status, lastTs } = useElementStream();
  const [, startTransition]          = useTransition();

  const [ogiActive,        setOgiActive]        = useState(false);
  const [schlierenActive,  setSchlierenActive]  = useState(false);
  const [showCalib,        setShowCalib]         = useState(false);
  const [searchQuery,      setSearchQuery]       = useState('');
  const [ammoniaMode,      setAmmoniaMode]       = useState('gas');
  const [ammoniaData,      setAmmoniaData]       = useState(null);
  const [fusionLabel,      setFusionLabel]       = useState(null);
  const [scanResult,       setScanResult]        = useState(null);

  const videoRef  = useRef(null);
  const workerRef = useRef(getWorker());

  // Refresh ammonia on mode change
  useEffect(() => {
    fetchAmmonia(ammoniaMode).then(setAmmoniaData).catch(() => {});
  }, [ammoniaMode]);

  // Trigger UV scan
  const triggerScan = useCallback(async () => {
    try {
      const { data } = await axios.post('/api/scan');
      setScanResult(data);
    } catch (_) {}
  }, []);

  // Filtered element list for search
  const filteredElements = useMemo(() => {
    const q = searchQuery.toLowerCase();
    if (!q) return elements;
    return Object.fromEntries(
      Object.entries(elements).filter(([name, el]) =>
        name.includes(q) || (el.name ?? '').toLowerCase().includes(q)
      )
    );
  }, [elements, searchQuery]);

  const toggleOGI = () => {
    setOgiActive(v => !v);
    if (!ogiActive) setSchlierenActive(false);
  };

  const toggleSchlieren = () => {
    if (!schlierenActive) {
      setShowCalib(true);   // prompt calibration before enabling Schlieren
      return;
    }
    setSchlierenActive(false);
  };

  const onCalibComplete = () => {
    setShowCalib(false);
    setSchlierenActive(true);
  };

  return (
    <div className="dashboard">
      {/* Global scanline */}
      <div className="scanline-global" />

      {/* Calibration modal */}
      {showCalib && (
        <CalibrationModal
          videoRef={videoRef}
          workerRef={workerRef}
          onComplete={onCalibComplete}
          onDismiss={() => setShowCalib(false)}
        />
      )}

      {/* Header */}
      <header>
        <h1>⚗ REE Digital Twin · Chemical Vision Analyzer</h1>
        <span className={`status-bar${status === 'disconnected' ? ' disconnected' : ''}`}>
          {status === 'live' ? '● LIVE' : status === 'connecting' ? '● CONNECTING…' : '● DISCONNECTED'}
        </span>
      </header>

      {/* Camera + side panel */}
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 320px', gap: '1rem', marginBottom: '1.25rem' }}>
        <ScanOverlay
          ogiActive={ogiActive}
          schlierenActive={schlierenActive}
          fusionLabel={fusionLabel}
          videoRef={videoRef}
        />

        <div style={{ display: 'flex', flexDirection: 'column', gap: '0.75rem' }}>
          {/* Fusion panel */}
          <div className="panel-box">
            <h3>⚗ Sensor Fusion (Gemma 2 / SAM)</h3>
            <FusionPanel onLabelUpdate={setFusionLabel} />
          </div>

          {/* Ammonia suite */}
          <div className="panel-box">
            <h3>NH₃ / NH₄NO₃ Suite</h3>
            <div style={{ display: 'flex', gap: 6, marginBottom: 6 }}>
              {['gas', 'ion'].map(m => (
                <button
                  key={m}
                  className={`btn btn-blue${ammoniaMode === m ? ' active' : ''}`}
                  style={{ padding: '0.25rem 0.75rem', fontSize: '0.68rem' }}
                  onClick={() => startTransition(() => setAmmoniaMode(m))}
                >
                  {m === 'gas' ? 'Gas (NH₃)' : 'Ion (NH₄NO₃)'}
                </button>
              ))}
            </div>
            {ammoniaData && (
              <div style={{ fontFamily: 'monospace', fontSize: '0.7rem', color: '#8888a8', lineHeight: 1.6 }}>
                {Object.entries(ammoniaData.readings ?? {}).slice(0, 6).map(([k, v]) => (
                  <div key={k} style={{ display: 'flex', justifyContent: 'space-between' }}>
                    <span>{k}</span>
                    <span style={{ color: '#e8e8f0' }}>{typeof v === 'number' ? v.toFixed(3) : String(v)}</span>
                  </div>
                ))}
              </div>
            )}
          </div>

          {/* UV scan */}
          <div className="panel-box">
            <h3>UV Camera Scan</h3>
            <div style={{ fontFamily: 'monospace', fontSize: '0.7rem', color: '#8888a8', marginBottom: 6 }}>
              Scans: {scanResult?.scan_count ?? 0} &nbsp;|&nbsp;
              Detections: {scanResult?.detection_count ?? 0}
            </div>
            {scanResult?.detections?.slice(0, 3).map(d => (
              <div key={d.id} style={{ fontFamily: 'monospace', fontSize: '0.67rem', color: '#8888a8' }}>
                {d.label} {Math.round(d.confidence * 100)}% @ ({d.bounding_box.x},{d.bounding_box.y})
              </div>
            ))}
            <button className="btn btn-green" onClick={triggerScan}
              style={{ marginTop: 8, padding: '0.3rem 0.8rem', fontSize: '0.72rem' }}>
              ▶ UV Scan
            </button>
          </div>
        </div>
      </div>

      {/* Controls */}
      <div className="controls">
        <button className={`btn ${ogiActive ? 'btn-orange' : 'btn-blue'}`} onClick={toggleOGI}>
          {ogiActive ? '◼ OGI Off' : 'OGI Filter'}
        </button>
        <button className={`btn ${schlierenActive ? 'btn-orange' : 'btn-blue'}`} onClick={toggleSchlieren}>
          {schlierenActive ? '◼ Schlieren Off' : 'Schlieren'}
        </button>
        <button className="btn btn-orange" onClick={() => setShowCalib(true)}>
          Calibrate
        </button>
        <input
          type="text"
          id="searchBar"
          placeholder="Filter elements…"
          value={searchQuery}
          onChange={e => startTransition(() => setSearchQuery(e.target.value))}
        />
        {lastTs && (
          <span style={{ fontFamily: 'monospace', fontSize: '0.68rem', color: '#8888a8' }}>
            Updated {new Date(lastTs * 1000).toLocaleTimeString()}
          </span>
        )}
      </div>

      {/* REE Element grid */}
      <div className="element-grid">
        {Object.entries(filteredElements).map(([name, el]) => (
          <ElementCard
            key={name}
            name={name}
            data={el}
            showOGI={ogiActive}
            showSchlieren={schlierenActive}
          />
        ))}
        {Object.keys(filteredElements).length === 0 && (
          <div style={{ fontFamily: 'monospace', fontSize: '0.8rem', color: '#8888a8', padding: '2rem', gridColumn: '1/-1' }}>
            {status === 'connecting' ? 'Connecting to Digital Twin server…' : 'No elements match filter.'}
          </div>
        )}
      </div>

      {/* 3D Point cloud */}
      <PointCloud elements={elements} />
    </div>
  );
}
