/**
 * elementMapper.js — Spectral Logic (React-side copy)
 *
 * Identical logic to static/js/elementMapper.js; imported by React
 * components so the bundle is self-contained.
 *
 * Maps raw Android /sys/ thermal + battery flux values into visual and
 * RF metadata for the Element Card UI.
 */

/**
 * @param {number} thermal    Raw /sys/class/thermal/thermal_zoneX/temp value (mC)
 * @param {number} flux       Raw /sys/class/power_supply/battery/current_now (µA)
 * @param {string} [rfType]   RF signature type
 * @param {number} [spectralNm] Primary spectral line (nm)
 */
export function mapElementSignature(
  thermal,
  flux,
  rfType = 'Steady State',
  spectralNm = 450
) {
  // 1. Normalise thermal (30 000 mC idle → 70 000 mC peak)
  const t = Math.min(Math.max((thermal - 30_000) / 40_000, 0), 1);

  // 2. Map temperature to visible-spectrum hue (blue=cool, red=hot)
  const hue        = Math.round((1 - t) * 240);
  const saturation = Math.round(70 + t * 20);
  const lightness  = Math.round(55 - t * 10);
  const hex        = `hsl(${hue}, ${saturation}%, ${lightness}%)`;

  // 3. Secondary accent from the element's spectral line
  const specHue = Math.round(270 - ((spectralNm - 380) / 320) * 270);
  const rfColor = `hsl(${specHue}, 90%, 65%)`;

  // 4. RF amplitude from current (0–1)
  const amplitude = Math.min(Math.abs(flux) / 1_000_000, 1.0);

  // 5. Opacity
  const opacity = t > 0.1 ? 0.2 + t * 0.6 : 0.15;

  // 6. Flags
  const isExcited   = t > 0.7;
  const isPeak      = t > 0.9;
  const glitchEffect = isExcited ? parseFloat((t * 10).toFixed(2)) : 0;
  const badge = isPeak ? 'PEAK EMISSION' : isExcited ? 'EXCITED' : 'STABLE';

  // 7. SVG waveform path
  const wavePathD = buildWavePath(rfType, amplitude, t);

  // 8. Canvas screen-blend overlay colour
  const overlayRgba = thermalToScreenOverlay(t, opacity);

  return {
    hex, rfColor,
    amplitude:    parseFloat(amplitude.toFixed(4)),
    opacity:      parseFloat(opacity.toFixed(3)),
    isExcited, isPeak, glitchEffect, badge,
    wavePathD, overlayRgba,
    t,
  };
}

/**
 * Build an SVG <path d="…"> for the inline RF waveform.
 * @param {string} rfType
 * @param {number} amplitude  0–1
 * @param {number} t          normalised temperature 0–1
 */
export function buildWavePath(rfType, amplitude, t) {
  const mid  = 10;
  const h    = 8 * amplitude;
  const pts  = [];

  for (let x = 0; x <= 100; x += 2) {
    let y;
    switch (rfType) {
      case 'Pulsed':
        y = mid + (x % 20 < 4 ? -h : 0);
        break;
      case 'Oscillatory':
        y = mid - Math.sin((x / 100) * Math.PI * 6) * h;
        break;
      case 'Harmonic':
        y = mid - (Math.sin((x / 100) * Math.PI * 6)
                 + 0.4 * Math.sin((x / 100) * Math.PI * 12)) * h * 0.7;
        break;
      case 'Decay Pattern':
        y = mid - Math.sin((x / 100) * Math.PI * 5) * h * Math.exp(-x / 60);
        break;
      case 'Wideband Noise':
        // Deterministic pseudo-noise using sine superposition
        y = mid - (Math.sin(x * 1.1) + Math.sin(x * 2.3) + Math.sin(x * 3.7)) * h / 3;
        break;
      default: // Steady State
        y = mid - t * h;
    }
    pts.push(`${x},${y.toFixed(1)}`);
  }
  return `M${pts.join(' L')}`;
}

/**
 * Convert normalised temperature to an RGBA string for Canvas
 * globalCompositeOperation='screen' overlay.
 */
export function thermalToScreenOverlay(t, opacity) {
  const r = Math.round(255 * t);
  const g = Math.round(180 * (1 - t) + 60 * t);
  const b = Math.round(255 * (1 - t));
  return `rgba(${r},${g},${b},${opacity.toFixed(2)})`;
}

/**
 * Build a simple Three.js-style point for a 3D point cloud.
 * Returns {x, y, z, color} in normalised coordinates.
 */
export function buildPointCloudPoint(elementName, sig, index, total) {
  const phi   = Math.acos(1 - 2 * (index + 0.5) / total);   // Fibonacci sphere
  const theta = Math.PI * (1 + Math.sqrt(5)) * index;
  const r     = 0.4 + sig.amplitude * 0.4;
  return {
    x:     r * Math.sin(phi) * Math.cos(theta),
    y:     r * Math.cos(phi),
    z:     r * Math.sin(phi) * Math.sin(theta),
    color: sig.hex,
    label: elementName,
    size:  2 + sig.t * 6,
  };
}
