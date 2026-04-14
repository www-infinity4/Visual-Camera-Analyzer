"""
Rare Earth Signal Generator — Digital Mine
══════════════════════════════════════════
Generates synthetic spectral "fingerprints" for rare earth elements (REEs)
based on their unique 4f electronic transition absorption signatures.

What makes REEs special
───────────────────────
Rare earth elements (lanthanides, Z = 57–71) have partially filled 4f
electron shells that are shielded from the chemical environment by outer
5s and 5p electrons.  This produces extremely narrow, sharp absorption
lines that are:
  • Element-specific  — no two REEs share the same set of lines
  • Stable            — barely shift with changes in host matrix or temperature
  • Detectable at ppm levels in natural minerals, ore samples, and soils

These "narrow spikes" are the reason satellite hyperspectral surveys can
locate REE deposits from orbit — the same signatures that DARPA's AI
mineralogy programme uses to create "digital mines."

The 4f Signal Model
───────────────────
Each absorption feature is modelled as a Gaussian dip in an otherwise flat
reflectance spectrum:

    signal(x) = 1 + background_noise(x) − Σ Aᵢ · exp(−(x−λᵢ)²/(2σᵢ²))

where λᵢ = peak wavelength (nm), Aᵢ = absorption intensity, σᵢ = line width.

Built-in REE Library
────────────────────
  Neodymium  (Nd, Z=60) : 580, 740, 800 nm   (magnets, lasers)
  Praseodymium (Pr, Z=59): 445, 480, 590 nm  (glass colouring, magnets)
  Samarium   (Sm, Z=62) : 950, 1080, 1250 nm (permanent magnets, neutron absorbers)
  Dysprosium (Dy, Z=66) : 1300, 1670, 1700 nm (EV motors, wind turbines)
  Europium   (Eu, Z=63) : 395, 465, 615 nm   (phosphors, LEDs)
  Terbium    (Tb, Z=65) : 350, 487, 544 nm   (green phosphors)
  Ytterbium  (Yb, Z=70) : 976, 1030 nm       (fibre lasers)
  Erbium     (Er, Z=68) : 520, 800, 1530 nm  (fibre amplifiers, lasers)

References
──────────
[1] DARPA REE AI: https://www.defenseone.com/technology/2024/07/
    darpa-wants-use-ai-find-new-rare-minerals/397830/
[2] 4f transitions: https://www.nature.com/articles/s41598-024-71395-2
[3] REE hyperspectral detection:
    https://www.sciencedirect.com/science/article/pii/S305099552600036X
[4] Digital twin mineralogy: https://medium.com/mineshift/
    revolutionizing-rare-earth-extraction-with-digital-twins-and-predictive-control-59b61e4c8a35
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.ndimage import gaussian_filter1d


# ---------------------------------------------------------------------------
# REE Spectral Library
# ---------------------------------------------------------------------------
# Format: element_name → List of (peak_nm, intensity, sigma_nm)
# intensity = fractional absorption depth (0–1)
# sigma_nm  = Gaussian half-width (nm) — REE lines are characteristically narrow

REE_LIBRARY: Dict[str, List[Tuple[float, float, float]]] = {
    "neodymium":     [(580.0, 0.35, 6),  (740.0, 0.4, 8),  (800.0, 0.6, 10)],
    "praseodymium":  [(445.0, 0.5, 5),   (480.0, 0.4, 6),  (590.0, 0.3, 7)],
    "samarium":      [(950.0, 0.25, 10), (1080.0, 0.3, 12),(1250.0, 0.5, 15)],
    "dysprosium":    [(1300.0, 0.2, 5),  (1670.0, 0.6, 18),(1700.0, 0.7, 20)],
    "europium":      [(395.0, 0.7, 4),   (465.0, 0.5, 5),  (615.0, 0.9, 6)],
    "terbium":       [(350.0, 0.4, 4),   (487.0, 0.6, 5),  (544.0, 0.8, 5)],
    "ytterbium":     [(976.0, 0.9, 3),   (1030.0, 0.5, 5)],
    "erbium":        [(520.0, 0.3, 4),   (800.0, 0.4, 6),  (1530.0, 0.8, 8)],
    "holmium":       [(450.0, 0.3, 5),   (537.0, 0.5, 6),  (641.0, 0.4, 7)],
    "thulium":       [(690.0, 0.35, 5),  (1210.0, 0.6, 8), (1720.0, 0.7, 12)],
    "cerium":        [(254.0, 0.8, 10),  (320.0, 0.5, 12)],
    "lanthanum":     [(138.0, 0.9, 8)],   # deep UV
    "gadolinium":    [(272.0, 0.6, 6),   (313.0, 0.4, 7)],
}


# ---------------------------------------------------------------------------
# Signal generator function (matches the requirement's sketch)
# ---------------------------------------------------------------------------


def generate_rare_earth_signal(
    material: str = "neodymium",
    background_noise: float = 0.05,
    wavelength_min_nm: float = 400.0,
    wavelength_max_nm: float = 2500.0,
    num_points: int = 2048,
    concentration_scale: float = 1.0,
    rng: Optional[np.random.Generator] = None,
) -> List[Tuple[float, float]]:
    """
    Simulate the 4f-transition spectral fingerprint of a rare earth element.

    Each REE produces characteristic narrow absorption dips at specific
    wavelengths.  These "invisible" signals are what AI can detect in
    hyperspectral imagery where the human eye would see only soil.

    Args:
        material:           REE name (key in REE_LIBRARY).  Case-insensitive.
        background_noise:   Gaussian noise std for the baseline (0–1 reflectance).
        wavelength_min_nm:  Start of spectral range (nm).
        wavelength_max_nm:  End of spectral range (nm).
        num_points:         Number of spectral channels.
        concentration_scale: Scales absorption depth (1.0 = standard ore grade).
        rng:                NumPy Generator for reproducible results.

    Returns:
        List of (wavelength_nm, reflectance) tuples.
        Reflectance is 1.0 for a perfect white reflector; absorption dips
        appear as sharp downward spikes.

    Example:
        >>> digital_neodymium = generate_rare_earth_signal("neodymium")
        >>> wl, ref = zip(*digital_neodymium)
        >>> print(f"Deepest dip at {wl[ref.index(min(ref))]:.0f} nm")
    """
    if rng is None:
        rng = np.random.default_rng()

    x = np.linspace(wavelength_min_nm, wavelength_max_nm, num_points)
    signal = np.ones(num_points) + rng.normal(0.0, background_noise, num_points)

    key = material.lower().strip()
    if key in REE_LIBRARY:
        for peak_nm, intensity, sigma_nm in REE_LIBRARY[key]:
            dip = (intensity * concentration_scale) * np.exp(
                -((x - peak_nm) ** 2) / (2 * sigma_nm ** 2)
            )
            signal -= dip

    signal = np.clip(signal, 0.0, 2.0)  # allow slight over-reflection for noise
    return list(zip(x.tolist(), signal.tolist()))


# ---------------------------------------------------------------------------
# Noise / SNR filter
# ---------------------------------------------------------------------------


@dataclass
class PeakDetectionResult:
    """Result of REE peak detection in a spectrum."""

    material: str
    peaks_detected: List[Tuple[float, float]]   # (wavelength_nm, depth)
    snr: float                                   # signal-to-noise ratio
    is_detected: bool                            # True if SNR > threshold
    confidence: float                            # 0–1


class RareEarthDetector:
    """
    Filters REE signals from background noise.

    Uses the 3-sigma SNR criterion: a peak is real if its depth exceeds
    3× the baseline noise standard deviation.  This is the same statistical
    test used in spectroscopic trace analysis (IUPAC LOD = 3σ).

    Implements a "Digital Mine" scanner: given a spectrum, determine which
    (if any) REE is present and report a confidence score.
    """

    SNR_THRESHOLD_DB = 3.0    # minimum SNR for a reliable detection

    def __init__(
        self,
        snr_threshold: float = SNR_THRESHOLD_DB,
        peak_tolerance_nm: float = 20.0,
        smooth_sigma: float = 1.0,
    ):
        """
        Args:
            snr_threshold:     Minimum SNR (linear ratio) for a valid detection.
            peak_tolerance_nm: Wavelength tolerance when matching library peaks.
            smooth_sigma:      Gaussian smoothing σ before peak detection (nm units).
        """
        self.snr_threshold = snr_threshold
        self.peak_tolerance_nm = peak_tolerance_nm
        self.smooth_sigma = smooth_sigma

    def estimate_noise(self, reflectance: np.ndarray) -> float:
        """
        Estimate baseline noise from the upper quartile of reflectance
        (where no absorption dips are expected).
        """
        upper = reflectance[reflectance > np.percentile(reflectance, 75)]
        return float(np.std(upper)) if len(upper) > 2 else 1e-6

    def detect_peaks(
        self,
        wavelengths: np.ndarray,
        reflectance: np.ndarray,
    ) -> List[Tuple[float, float]]:
        """
        Find absorption dips (negative peaks) in the reflectance spectrum.

        Args:
            wavelengths:  1-D array of wavelength values (nm).
            reflectance:  1-D reflectance array (same length).

        Returns:
            List of (wavelength_nm, absorption_depth) for each dip.
        """
        if self.smooth_sigma > 0:
            refl = gaussian_filter1d(reflectance, sigma=self.smooth_sigma)
        else:
            refl = reflectance.copy()

        noise = self.estimate_noise(refl)
        baseline = float(np.median(refl))
        threshold_depth = self.snr_threshold * noise

        dips: List[Tuple[float, float]] = []
        n = len(refl)
        for i in range(1, n - 1):
            depth = baseline - refl[i]
            if depth < threshold_depth:
                continue
            if refl[i] < refl[i - 1] and refl[i] < refl[i + 1]:
                dips.append((float(wavelengths[i]), round(depth, 6)))

        return dips

    def identify(
        self,
        wavelengths: np.ndarray,
        reflectance: np.ndarray,
    ) -> List[PeakDetectionResult]:
        """
        Identify REEs by matching detected dips against the library.

        Args:
            wavelengths:  Wavelength array (nm).
            reflectance:  Reflectance array (0–1+ range).

        Returns:
            List of PeakDetectionResult sorted by confidence (descending).
        """
        noise = self.estimate_noise(reflectance)
        dips = self.detect_peaks(wavelengths, reflectance)

        results: List[PeakDetectionResult] = []
        for material, library_peaks in REE_LIBRARY.items():
            matched: List[Tuple[float, float]] = []
            for lib_nm, lib_intensity, _ in library_peaks:
                for dip_nm, dip_depth in dips:
                    if abs(dip_nm - lib_nm) <= self.peak_tolerance_nm:
                        matched.append((dip_nm, dip_depth))
                        break

            if not matched:
                continue

            # Confidence = fraction of library peaks matched,
            # weighted by average depth / expected intensity
            frac_matched = len(matched) / len(library_peaks)
            avg_depth = float(np.mean([d for _, d in matched]))
            avg_expected = float(np.mean([i for _, i, _ in library_peaks]))
            depth_ratio = min(1.0, avg_depth / (avg_expected + 1e-9))
            confidence = round(frac_matched * 0.6 + depth_ratio * 0.4, 4)

            # SNR: ratio of average dip depth to noise
            snr = avg_depth / (noise + 1e-9) if noise > 0 else 0.0

            results.append(
                PeakDetectionResult(
                    material=material,
                    peaks_detected=matched,
                    snr=round(snr, 2),
                    is_detected=snr >= self.snr_threshold,
                    confidence=confidence,
                )
            )

        results.sort(key=lambda r: r.confidence, reverse=True)
        return results

    def to_text_summary(self, results: List[PeakDetectionResult]) -> str:
        """Structured text summary for Sensor-LLM fusion."""
        detected = [r for r in results if r.is_detected]
        if not detected:
            return "REE detector: no rare earth element signatures detected above noise floor."
        parts = [
            f"{r.material.title()} (SNR={r.snr:.1f}, confidence={r.confidence:.0%})"
            for r in detected[:3]
        ]
        return (
            f"REE detector: identified {len(detected)} rare earth signature(s): "
            + "; ".join(parts) + "."
        )


# ---------------------------------------------------------------------------
# Digital Mine Scanner — spatial REE mapping
# ---------------------------------------------------------------------------


class DigitalMineScanner:
    """
    Maps REE abundance across a 2-D spatial grid using synthetic spectral data.

    This implements the "Digital Mine" concept:
    • Each grid cell's spectrum is generated by the REE signal generator.
    • The RareEarthDetector identifies which elements are present.
    • A confidence heatmap is produced for each REE.

    Analogous to DARPA's AI satellite mineralogy work: the model does not
    "see" physical material — it reasons from spectral signatures alone.
    """

    def __init__(
        self,
        detector: Optional[RareEarthDetector] = None,
        wavelength_min_nm: float = 400.0,
        wavelength_max_nm: float = 2500.0,
        num_spectral_points: int = 512,
    ):
        self.detector = detector or RareEarthDetector()
        self.wavelength_min_nm = wavelength_min_nm
        self.wavelength_max_nm = wavelength_max_nm
        self.num_spectral_points = num_spectral_points
        self._wavelengths = np.linspace(
            wavelength_min_nm, wavelength_max_nm, num_spectral_points
        )

    def scan_pixel(
        self,
        spectrum: np.ndarray,
        target_element: Optional[str] = None,
    ) -> Dict:
        """
        Analyse a single pixel spectrum.

        Args:
            spectrum:        1-D reflectance array (num_spectral_points,).
            target_element:  If set, return only this element's result.

        Returns:
            Dict with detection results and best-match element.
        """
        results = self.detector.identify(self._wavelengths, spectrum)
        detected = [r for r in results if r.is_detected]

        if target_element:
            target_key = target_element.lower()
            for r in results:
                if r.material == target_key:
                    return {
                        "target": target_element,
                        "detected": r.is_detected,
                        "confidence": r.confidence,
                        "snr": r.snr,
                    }
            return {"target": target_element, "detected": False, "confidence": 0.0, "snr": 0.0}

        best = results[0] if results else None
        return {
            "best_match": best.material if best else None,
            "confidence": best.confidence if best else 0.0,
            "snr": best.snr if best else 0.0,
            "all_detected": [r.material for r in detected],
        }

    def scan_grid(
        self,
        spectra_grid: np.ndarray,
        target_element: str,
    ) -> np.ndarray:
        """
        Produce a 2-D confidence heatmap for a target REE across a spatial grid.

        Args:
            spectra_grid:    3-D array (rows, cols, num_spectral_points).
            target_element:  REE name to map.

        Returns:
            2-D float32 confidence map (rows, cols) in [0, 1].
        """
        rows, cols, _ = spectra_grid.shape
        heatmap = np.zeros((rows, cols), dtype=np.float32)
        for r in range(rows):
            for c in range(cols):
                result = self.scan_pixel(spectra_grid[r, c], target_element)
                heatmap[r, c] = result["confidence"]
        return heatmap

    def simulate_deposit(
        self,
        grid_rows: int = 10,
        grid_cols: int = 10,
        target_element: str = "neodymium",
        deposit_region: Optional[Tuple[int, int, int, int]] = None,
        ore_grade_scale: float = 2.0,
        background_noise: float = 0.06,
        rng: Optional[np.random.Generator] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Simulate a spatial ore deposit for demonstration.

        Places a high-concentration REE deposit in a sub-region of the grid,
        surrounded by background soil/rock reflectance.

        Args:
            grid_rows, grid_cols: Spatial dimensions.
            target_element:       REE to simulate.
            deposit_region:       (r0, r1, c0, c1) bounding box for the deposit.
            ore_grade_scale:      Concentration multiplier in the deposit region.
            background_noise:     Baseline noise for off-deposit cells.
            rng:                  NumPy Generator for reproducibility.

        Returns:
            Tuple of (spectra_grid (H, W, bands), confidence_map (H, W)).
        """
        if rng is None:
            rng = np.random.default_rng(42)
        if deposit_region is None:
            r0, r1 = grid_rows // 4, 3 * grid_rows // 4
            c0, c1 = grid_cols // 4, 3 * grid_cols // 4
            deposit_region = (r0, r1, c0, c1)

        r0, r1, c0, c1 = deposit_region
        spectra = np.zeros((grid_rows, grid_cols, self.num_spectral_points))

        for r in range(grid_rows):
            for c in range(grid_cols):
                in_deposit = r0 <= r < r1 and c0 <= c < c1
                scale = ore_grade_scale if in_deposit else 0.05
                noise = background_noise * (0.5 if in_deposit else 1.0)
                sig_pairs = generate_rare_earth_signal(
                    target_element,
                    background_noise=noise,
                    wavelength_min_nm=self.wavelength_min_nm,
                    wavelength_max_nm=self.wavelength_max_nm,
                    num_points=self.num_spectral_points,
                    concentration_scale=scale,
                    rng=rng,
                )
                spectra[r, c, :] = np.array([s for _, s in sig_pairs])

        heatmap = self.scan_grid(spectra, target_element)
        return spectra, heatmap
