/**
 * elementMapper.js — Spectral Logic for the REE Digital Twin UI
 * ═══════════════════════════════════════════════════════════════
 * Maps raw Android hardware sensor readings from /sys/ sysfs files into
 * visual and RF metadata that drives the Element Card UI components.
 *
 * Input sources (from the Python element_server.py bridge):
 *   thermal  ← /sys/class/thermal/thermal_zone*/temp       (milli-°C)
 *   flux     ← /sys/class/power_supply/battery/current_now (µA)
 *
 * Output properties fed into the React/Vue/vanilla DOM:
 *   hex          → HSL colour string (blue=cool, red=excited)
 *   amplitude    → RF amplitude scalar (0–1) from current
 *   opacity      → Detection confidence visual weight
 *   isExcited    → Boolean — high emission state flag
 *   glitchEffect → CSS skew/jitter magnitude
 *   wavePathD    → SVG <path d="…"> for the inline RF waveform
 *   rfColor      → Secondary accent colour for RF overlay
 *   badge        → Human-readable excitation state label
 */

/**
 * Maps Android hardware sensor data to Rare Earth Element signatures.
 *
 * @param {number} thermal  - Raw value from /sys/class/thermal/thermal_zone (in mC)
 * @param {number} flux     - Raw value from /sys/class/power_supply/battery/current_now (in µA)
 * @param {string} rfType   - RF signature type: "Wideband Noise" | "Oscillatory" |
 *                            "Pulsed" | "Steady State" | "Decay Pattern" | "Harmonic"
 * @param {number} spectralNm - Element's primary spectral line (nm), 300–700
 * @returns {ElementSignature} Visual and RF metadata for the Digital Twin UI
 */
const mapElementSignature = (thermal, flux, rfType = "Steady State", spectralNm = 450) => {
    // ─── 1. Normalise Thermal ─────────────────────────────────────────────
    // Range: 30 000 mC (30°C, idle) → 70 000 mC (70°C, peak load)
    const normalizedTemp = Math.min(Math.max((thermal - 30000) / 40000, 0), 1);

    // ─── 2. Map Temperature to Visible Spectrum ───────────────────────────
    // Blue (hue 240) = cool/stable,  Red (hue 0) = hot/excited
    // Simulates the "Excitation State" of the mineral's electron shell
    const hue = (1 - normalizedTemp) * 240;          // 240 → 0
    const saturation = 70 + normalizedTemp * 20;     // 70 → 90 %
    const lightness  = 45 + (1 - normalizedTemp) * 10; // 55 → 45 %
    const hex = `hsl(${Math.round(hue)}, ${Math.round(saturation)}%, ${Math.round(lightness)}%)`;

    // Secondary accent derived from the element's spectral line (nm → hue)
    // Visible spectrum: 380 nm → hue 270 (violet), 700 nm → hue 0 (red)
    const specHue = Math.round(270 - ((spectralNm - 380) / 320) * 270);
    const rfColor = `hsl(${specHue}, 90%, 65%)`;

    // ─── 3. Map Flux (Current) to RF Amplitude ────────────────────────────
    // Absolute value because current is negative when discharging
    const rfAmplitude = Math.min(Math.abs(flux) / 1_000_000, 1.0);

    // ─── 4. Detection Confidence → Opacity ───────────────────────────────
    const opacity = normalizedTemp > 0.1 ? 0.2 + normalizedTemp * 0.6 : 0.15;

    // ─── 5. Glitch/Jitter Magnitude ───────────────────────────────────────
    // Applied as CSS transform: skewX(glitchEffect deg)
    const glitchEffect = normalizedTemp > 0.7 ? normalizedTemp * 10 : 0;

    // ─── 6. Derived flags ─────────────────────────────────────────────────
    const isExcited   = normalizedTemp > 0.7;        // ≥ 58 000 mC (58°C)
    const isPeak      = normalizedTemp > 0.9;        // ≥ 66 000 mC (66°C)
    const badge       = isPeak ? "PEAK EMISSION" : isExcited ? "EXCITED" : "STABLE";

    // ─── 7. SVG Waveform Path ─────────────────────────────────────────────
    const wavePathD   = buildWavePath(rfType, rfAmplitude, normalizedTemp);

    // ─── 8. Canvas globalCompositeOperation overlay colour ────────────────
    // Used with 'screen' blending on the camera feed (hot = orange-white)
    const overlayRgba = tempToScreenOverlay(normalizedTemp, opacity);

    return {
        hex,
        rfColor,
        amplitude:      parseFloat(rfAmplitude.toFixed(4)),
        opacity:        parseFloat(opacity.toFixed(3)),
        isExcited,
        isPeak,
        glitchEffect:   parseFloat(glitchEffect.toFixed(2)),
        wavePathD,
        badge,
        normalizedTemp: parseFloat(normalizedTemp.toFixed(4)),
        overlayRgba,
        // Raw derived values (useful for 3D point cloud / Three.js)
        emissionIntensity: normalizedTemp,
        magneticFlux:      rfAmplitude,
        spectralHue:       specHue,
    };
};

/**
 * Convert normalised temperature to an RGBA string for Canvas screen-blend.
 *
 * At low temp → transparent blue tint
 * At high temp → bright orange-white (simulates infrared heat leakage)
 *
 * @param {number} t    - Normalised temperature 0–1
 * @param {number} base - Base opacity
 * @returns {string} - CSS rgba() string
 */
const tempToScreenOverlay = (t, base = 0.4) => {
    const r = Math.round(t * 255);
    const g = Math.round(t * 120);
    const b = Math.round((1 - t) * 200);
    const a = parseFloat((base * t).toFixed(3));
    return `rgba(${r}, ${g}, ${b}, ${a})`;
};

/**
 * Build an SVG <path d="…"> string for the RF waveform display.
 *
 * The path fits a 100×20 viewBox.  Different RF types produce different
 * shapes matching the physical waveform of that emission mode.
 *
 * @param {string} rfType    - One of the six RF signature types
 * @param {number} amplitude - 0–1 amplitude scalar
 * @param {number} temp      - Normalised temperature 0–1 (adds noise)
 * @returns {string}
 */
const buildWavePath = (rfType, amplitude, temp) => {
    const A = amplitude * 8;   // max ±8 units in a 0-20 viewBox (mid = 10)
    const mid = 10;

    switch (rfType) {
        case "Oscillatory":
        case "Steady State": {
            // Clean sine wave — Q bezier approximation
            return (
                `M0 ${mid} ` +
                `Q 12.5 ${mid - A} 25 ${mid} ` +
                `Q 37.5 ${mid + A} 50 ${mid} ` +
                `Q 62.5 ${mid - A} 75 ${mid} ` +
                `Q 87.5 ${mid + A} 100 ${mid}`
            );
        }
        case "Pulsed": {
            // Spike train
            const spike = A * 1.5;
            return (
                `M0 ${mid} L20 ${mid} L21 ${mid - spike} L22 ${mid} ` +
                `L45 ${mid} L46 ${mid - spike} L47 ${mid} ` +
                `L70 ${mid} L71 ${mid - spike} L72 ${mid} L100 ${mid}`
            );
        }
        case "Decay Pattern": {
            // Exponentially decaying oscillation
            const pts = [];
            for (let x = 0; x <= 100; x += 2) {
                const decay  = Math.exp(-x / 40);
                const y      = mid + A * decay * Math.sin((x / 100) * 4 * Math.PI);
                pts.push(`${x} ${y.toFixed(2)}`);
            }
            return "M" + pts.join(" L");
        }
        case "Harmonic": {
            // Sum of fundamental + harmonics
            const pts = [];
            for (let x = 0; x <= 100; x += 2) {
                const t = (x / 100) * 2 * Math.PI;
                const y = mid + A * (
                    Math.sin(t) + 0.5 * Math.sin(2 * t) + 0.33 * Math.sin(3 * t)
                ) / 1.83;   // normalise sum to ≈ 1
                pts.push(`${x} ${y.toFixed(2)}`);
            }
            return "M" + pts.join(" L");
        }
        case "Wideband Noise": {
            // Jagged noise line (deterministic for consistent rendering)
            const pts = [];
            let seed = 42;
            const rand = () => {
                seed = (seed * 1103515245 + 12345) & 0x7fffffff;
                return (seed / 0x7fffffff) * 2 - 1;
            };
            for (let x = 0; x <= 100; x += 2) {
                const y = mid + A * rand() * (0.5 + temp * 0.5);
                pts.push(`${x} ${y.toFixed(2)}`);
            }
            return "M" + pts.join(" L");
        }
        default:
            return `M0 ${mid} L100 ${mid}`;
    }
};

/**
 * Apply computed signature to a DOM element card.
 *
 * @param {HTMLElement} cardEl  - The .element-card DOM node
 * @param {object}      sig     - Output of mapElementSignature()
 */
const applySignatureToCard = (cardEl, sig) => {
    if (!cardEl) return;

    // Set CSS custom property for glow colour
    cardEl.style.setProperty("--element-color", sig.hex);
    cardEl.style.setProperty("--rf-color", sig.rfColor);
    cardEl.style.opacity = sig.opacity;

    // Glitch / jitter transform
    if (sig.glitchEffect > 0) {
        cardEl.style.transform = `skewX(${sig.glitchEffect}deg)`;
    } else {
        cardEl.style.transform = "";
    }

    // Toggle CSS classes
    cardEl.classList.toggle("excited",          sig.isExcited);
    cardEl.classList.toggle("radiation-jitter", sig.isPeak);
};

/**
 * Build a Three.js-ready point cloud dataset from all element signatures.
 *
 * Each point represents one REE element in 3-D space:
 *   x = spectral hue / 360    (spectral position)
 *   y = normalizedTemp         (excitation height)
 *   z = magneticFlux           (RF emission depth)
 *
 * @param {object} signaturesMap - { elementName: ElementSignature, … }
 * @returns {Array<{x, y, z, color}>}
 */
const buildPointCloud = (signaturesMap) =>
    Object.entries(signaturesMap).map(([name, sig]) => ({
        name,
        x: sig.spectralHue / 360,
        y: sig.emissionIntensity,
        z: sig.magneticFlux,
        color: sig.hex,
        size: 0.05 + sig.emissionIntensity * 0.15,
    }));

// ─── Exports ────────────────────────────────────────────────────────────────

if (typeof module !== "undefined" && module.exports) {
    // Node / CommonJS
    module.exports = { mapElementSignature, buildWavePath, buildPointCloud, applySignatureToCard, tempToScreenOverlay };
} else if (typeof window !== "undefined") {
    // Browser global
    window.VCA = window.VCA || {};
    Object.assign(window.VCA, { mapElementSignature, buildWavePath, buildPointCloud, applySignatureToCard, tempToScreenOverlay });
}
