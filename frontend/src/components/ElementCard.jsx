/**
 * ElementCard.jsx — REE Digital Twin element card component
 *
 * Renders a single rare-earth element's live readings with:
 *  • Animated excitation state (excited / peak-emission / radiation-jitter)
 *  • SVG RF waveform driven by live flux data
 *  • OGI plume and Schlieren overlay CSS classes
 *  • Detection badge ("EXCITED" / "PEAK EMISSION")
 *
 * CSS custom properties --element-color, --rf-color, --glitch-deg are
 * set inline so the animations in emissions.css adapt per-card.
 */
import React, { memo } from 'react';
import { mapElementSignature, buildWavePath } from '../utils/elementMapper.js';

const ElementCard = memo(function ElementCard({ name, data, showOGI, showSchlieren }) {
  if (!data) return null;

  const sig = mapElementSignature(
    data.excitation  ?? 45_000,
    data.flux        ?? 0,
    data.rf_signature ?? 'Steady State',
    data.spectral_nm ?? 450,
  );

  const tempC  = ((data.excitation ?? 45_000) / 1000).toFixed(1);
  const fluxMA = ((data.flux ?? 0) / 1000).toFixed(1);

  const cardClass = [
    'element-card',
    sig.isExcited ? 'excited'           : '',
    sig.isPeak    ? 'peak-emission'     : '',
    sig.isExcited ? 'radiation-jitter'  : '',
  ].filter(Boolean).join(' ');

  const cardStyle = {
    '--element-color': sig.hex,
    '--rf-color':      sig.rfColor,
    '--glitch-deg':    `${sig.glitchEffect}deg`,
  };

  const wavePath = buildWavePath(data.rf_signature ?? 'Steady State', sig.amplitude, sig.t);

  return (
    <div className={cardClass} style={cardStyle} title={data.primary_use ?? ''}>
      {/* RF scan-line overlay (always present) */}
      <div className="rf-overlay" />

      {/* OGI plume — toggled via CSS class */}
      {showOGI && (
        <div
          className={`ogi-plume${sig.isExcited ? ' active' : ''}`}
          style={{ '--ogi-color': `${sig.hex.replace('hsl', 'hsla').replace(')', ',0.25)')}` }}
        />
      )}

      {/* Schlieren density-ripple overlay */}
      {showSchlieren && (
        <div className={`schlieren-overlay${sig.isExcited ? ' active' : ''}`} />
      )}

      {/* Detection badge */}
      <div className="detection-badge">{sig.badge}</div>

      {/* Periodic-table symbol */}
      <div className="element-symbol">{data.symbol ?? name.slice(0, 2).toUpperCase()}</div>

      {/* Element name */}
      <h2>{data.name ?? name}</h2>

      {/* Stats */}
      <div className="stat">
        <span>TEMP</span>
        <span className="value">{tempC}°C</span>
      </div>
      <div className="stat">
        <span>FLUX</span>
        <span className="value">{fluxMA} mA</span>
      </div>
      <div className="stat">
        <span>RF</span>
        <span className="value">{data.rf_signature ?? '—'}</span>
      </div>
      <div className="stat">
        <span>λ</span>
        <span className="value">{(data.spectral_nm ?? 0).toFixed(1)} nm</span>
      </div>

      {/* SVG RF waveform */}
      <div className="waveform-container">
        <svg
          viewBox="0 0 100 20"
          preserveAspectRatio="none"
          style={{ color: sig.rfColor }}
        >
          <path d={wavePath} />
        </svg>
      </div>
    </div>
  );
});

export default ElementCard;
