/**
 * FusionPanel.jsx — Sensor-LLM result panel
 *
 * Displays the output of /api/fusion and drives the Gemma 2 label
 * shown on the ScanOverlay camera feed (item 4 of integration checklist).
 */
import React, { useCallback, useEffect, useState, useTransition } from 'react';
import axios from 'axios';

const HAZARD_COLORS = {
  none:     '#44ff88',
  low:      '#ffee44',
  medium:   '#ff9944',
  high:     '#ff5533',
  critical: '#ff2244',
};

const CHEMICAL_OPTIONS = [
  'Ammonia', 'Ammonium Nitrate', 'Methane', 'Urea',
  'Neodymium', 'Cerium', 'Lanthanum', 'Terbium',
  'Dysprosium', 'Europium',
];

export default function FusionPanel({ onLabelUpdate }) {
  const [target,   setTarget]   = useState('Ammonia');
  const [ppm,      setPpm]      = useState(30);
  const [result,   setResult]   = useState(null);
  const [loading,  setLoading]  = useState(false);
  const [, startTransition]     = useTransition();

  const fetchFusion = useCallback(async (tgt = target, concentration = ppm) => {
    setLoading(true);
    try {
      const { data } = await axios.get('/api/fusion', {
        params: { target: tgt, ppm: concentration },
      });
      setResult(data);

      // Build the Gemma-labelled annotation for the camera overlay
      const conf  = Math.round((data.confidence ?? 0) * 100);
      const label = `⚗ ${data.identified_compound ?? '—'} · ${data.hazard_level?.toUpperCase() ?? 'NONE'} · ${conf}% (${data.backend_used ?? 'sam'})`;
      if (onLabelUpdate) onLabelUpdate(label);
    } catch (_) {
      if (onLabelUpdate) onLabelUpdate(null);
    } finally {
      setLoading(false);
    }
  }, [target, ppm, onLabelUpdate]);

  // Auto-run on mount and when target/ppm change
  useEffect(() => { fetchFusion(); }, [target, ppm]); // eslint-disable-line

  const hazardColor = HAZARD_COLORS[result?.hazard_level ?? 'none'] ?? '#44ff88';

  return (
    <div className="fusion-panel">
      <div style={{ display: 'flex', gap: 8, marginBottom: 8, flexWrap: 'wrap', alignItems: 'center' }}>
        <select
          value={target}
          onChange={e => startTransition(() => setTarget(e.target.value))}
          className="ctrl-select"
        >
          {CHEMICAL_OPTIONS.map(c => <option key={c}>{c}</option>)}
        </select>

        <input
          type="number" min="0" max="500" step="5"
          value={ppm}
          onChange={e => setPpm(Number(e.target.value))}
          style={{
            width: 70, background: '#111', border: '1px solid rgba(255,255,255,0.15)',
            color: '#e8e8f0', fontFamily: 'monospace', fontSize: '0.72rem',
            padding: '0.3rem 0.5rem', borderRadius: 4,
          }}
          title="Simulated concentration (ppm)"
        />

        <button
          className="btn btn-green"
          onClick={() => fetchFusion()}
          disabled={loading}
          style={{ padding: '0.3rem 0.8rem' }}
        >
          {loading ? '⟳' : '⚗ Fuse'}
        </button>
      </div>

      {result ? (
        <>
          <div style={{ marginBottom: 4 }}>
            <span className="compound">{result.identified_compound ?? '—'}</span>
            <span style={{ color: '#8888a8', marginLeft: 8, fontSize: '0.7rem' }}>
              {Math.round((result.confidence ?? 0) * 100)}% confidence
            </span>
          </div>

          <div style={{ display: 'flex', gap: 12, marginBottom: 6, fontSize: '0.72rem' }}>
            <span style={{ color: '#8888a8' }}>HAZARD</span>
            <span style={{ color: hazardColor, fontWeight: 700 }}>
              {(result.hazard_level ?? 'none').toUpperCase()}
            </span>
            <span style={{ color: '#8888a8' }}>BACKEND</span>
            <span style={{ color: '#e8e8f0' }}>{result.backend_used ?? '—'}</span>
          </div>

          {result.sensor_summaries?.length > 0 && (
            <details style={{ marginBottom: 4 }}>
              <summary style={{ color: '#8888a8', cursor: 'pointer', fontSize: '0.68rem' }}>
                Sensor summaries ({result.sensor_summaries.length})
              </summary>
              <ul style={{ margin: '4px 0 0 12px', padding: 0, color: '#8888a8', fontSize: '0.68rem', lineHeight: 1.6 }}>
                {result.sensor_summaries.map((s, i) => <li key={i}>{s}</li>)}
              </ul>
            </details>
          )}

          {result.reasoning && (
            <p style={{ color: '#8888a8', fontSize: '0.67rem', lineHeight: 1.5, marginTop: 4 }}>
              {result.reasoning}
            </p>
          )}
        </>
      ) : (
        <span style={{ color: '#8888a8' }}>—</span>
      )}
    </div>
  );
}
