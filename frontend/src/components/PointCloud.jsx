/**
 * PointCloud.jsx — 3D REE Emission Point Cloud (Canvas 2D, 60 fps)
 *
 * Renders all 17 REEs as glowing points on a rotating sphere projection.
 * Point size and glow are proportional to the element's excitation state.
 * Heavy math is avoided: no Three.js required — just canvas arc + shadows.
 *
 * For a true 3D version, upgrade to Three.js PointsMaterial using the
 * buildPointCloudPoint() helper from elementMapper.js.
 */
import React, { useEffect, useRef } from 'react';
import { mapElementSignature } from '../utils/elementMapper.js';

let pcAngle = 0;

export default function PointCloud({ elements }) {
  const canvasRef = useRef(null);
  const rafRef    = useRef(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const draw = () => {
      const W = canvas.offsetWidth  || 800;
      const H = canvas.offsetHeight || 200;
      if (canvas.width !== W) canvas.width = W;
      if (canvas.height !== H) canvas.height = H;

      const ctx = canvas.getContext('2d');
      ctx.fillStyle = '#000';
      ctx.fillRect(0, 0, W, H);

      pcAngle += 0.006;
      const entries = Object.entries(elements);
      if (!entries.length) { rafRef.current = requestAnimationFrame(draw); return; }

      entries.forEach(([name, el], i) => {
        const sig = mapElementSignature(
          el.excitation ?? 45_000,
          el.flux       ?? 0,
          el.rf_signature ?? 'Steady State',
          el.spectral_nm ?? 450,
        );

        // Fibonacci sphere projection
        const phi   = Math.acos(1 - 2 * (i + 0.5) / entries.length);
        const theta = Math.PI * (1 + Math.sqrt(5)) * i + pcAngle;
        const r     = 55 + sig.amplitude * 55;
        const x     = W / 2 + r * Math.sin(phi) * Math.cos(theta) * (W / 350);
        const y     = H / 2 + r * Math.cos(phi) * 0.35;
        const size  = 2 + sig.t * 7;

        ctx.shadowColor = sig.isExcited ? sig.hex : 'transparent';
        ctx.shadowBlur  = sig.isExcited ? 14 : 0;

        ctx.beginPath();
        ctx.arc(x, y, size, 0, Math.PI * 2);
        ctx.fillStyle = sig.isExcited
          ? sig.hex
          : sig.hex.replace('hsl(', 'hsla(').replace(')', ',0.45)');
        ctx.fill();

        // Label excited elements
        if (sig.t > 0.45) {
          ctx.shadowBlur = 0;
          ctx.fillStyle  = '#bbb';
          ctx.font       = '9px monospace';
          ctx.fillText(el.symbol ?? name.slice(0, 2).toUpperCase(), x + size + 2, y + 3);
        }
      });

      ctx.shadowBlur = 0;
      rafRef.current = requestAnimationFrame(draw);
    };

    rafRef.current = requestAnimationFrame(draw);
    return () => { if (rafRef.current) cancelAnimationFrame(rafRef.current); };
  }, [elements]);

  return (
    <canvas
      ref={canvasRef}
      className="point-cloud-canvas"
      title="REE Emission Point Cloud — point size = excitation level"
    />
  );
}
