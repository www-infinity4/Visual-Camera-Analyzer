"""
Spectral Analysis Module

Provides a lightweight DIY spectrometer simulation and spectral-feature
extraction from camera images.

Background:
- A simple DIY spectrometer can be built with a diffraction grating and
  smartphone camera to resolve the emission spectrum of fluorescent materials.
- Cat urine under UV illumination shows a characteristic emission peak near
  450 nm (blue-white) with secondary shoulders in the yellow-green region.
- This module can analyse a horizontal "spectral stripe" from a diffraction-
  grating image, or approximate spectral content from a standard RGB image
  using known camera colour-matching functions.

References:
- Public Lab DIY spectrometer: https://publiclab.org/wiki/spectrometer
- Spectral Sciences TRACER chemical mapping camera documentation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

try:
    import cv2 as _cv2

    _CV2_AVAILABLE = True
except ImportError:  # pragma: no cover
    _CV2_AVAILABLE = False


@dataclass
class SpectralProfile:
    """Holds a 1-D intensity profile mapped to estimated wavelengths."""

    wavelengths: np.ndarray  # nm
    intensities: np.ndarray  # normalised 0–1
    peak_wavelength: float  # nm at maximum intensity
    peak_intensity: float  # 0–1


# Approximate urine fluorescence peak under UV (nm)
URINE_PEAK_NM = 450.0
URINE_PEAK_TOLERANCE_NM = 30.0

# Approximate wavelength boundaries for R, G, B camera channels
# (centre wavelengths of broadband colour filters)
_CHANNEL_WAVELENGTHS = {
    "B": 450.0,
    "G": 540.0,
    "R": 610.0,
}


class SpectralAnalyzer:
    """
    Extracts and analyses spectral information from camera images.

    Two analysis modes are supported:

    1. **RGB approximation** – treats each colour channel as a broadband
       spectral bin at its approximate centre wavelength.  Quick and works
       with any camera.

    2. **Spectrometer stripe** – analyses a horizontal row of pixels from
       a diffraction-grating spectrometer image where each column corresponds
       to a different wavelength (requires calibration data).
    """

    def __init__(
        self,
        wavelength_min: float = 300.0,
        wavelength_max: float = 700.0,
        num_bins: int = 100,
        urine_peak_nm: float = URINE_PEAK_NM,
        peak_tolerance_nm: float = URINE_PEAK_TOLERANCE_NM,
    ):
        """
        Args:
            wavelength_min: Minimum wavelength for spectral range (nm).
            wavelength_max: Maximum wavelength for spectral range (nm).
            num_bins: Number of bins in the spectral profile.
            urine_peak_nm: Expected fluorescence peak for urine (nm).
            peak_tolerance_nm: Tolerance for peak matching (nm).
        """
        self.wavelength_min = wavelength_min
        self.wavelength_max = wavelength_max
        self.num_bins = num_bins
        self.urine_peak_nm = urine_peak_nm
        self.peak_tolerance_nm = peak_tolerance_nm
        self.wavelengths = np.linspace(wavelength_min, wavelength_max, num_bins)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def analyse_rgb_patch(self, patch: np.ndarray) -> SpectralProfile:
        """
        Approximate spectral profile from a BGR image patch using colour
        channel intensities as broadband spectral bins.

        Args:
            patch: BGR image patch (NumPy array, uint8).

        Returns:
            SpectralProfile with three representative bins (B, G, R).
        """
        if patch is None or patch.size == 0:
            raise ValueError("Patch is empty or None.")

        # Mean intensity per channel (OpenCV stores channels as B, G, R)
        b_mean = float(np.mean(patch[:, :, 0])) / 255.0
        g_mean = float(np.mean(patch[:, :, 1])) / 255.0
        r_mean = float(np.mean(patch[:, :, 2])) / 255.0

        channel_wl = np.array(
            [
                _CHANNEL_WAVELENGTHS["B"],
                _CHANNEL_WAVELENGTHS["G"],
                _CHANNEL_WAVELENGTHS["R"],
            ]
        )
        channel_intensity = np.array([b_mean, g_mean, r_mean])

        # Interpolate onto the full wavelength grid
        intensities = np.interp(self.wavelengths, channel_wl, channel_intensity)

        peak_idx = int(np.argmax(intensities))
        return SpectralProfile(
            wavelengths=self.wavelengths,
            intensities=intensities,
            peak_wavelength=float(self.wavelengths[peak_idx]),
            peak_intensity=float(intensities[peak_idx]),
        )

    def analyse_spectrometer_stripe(
        self,
        stripe: np.ndarray,
        calibration: Optional[Tuple[float, float]] = None,
    ) -> SpectralProfile:
        """
        Analyse a horizontal pixel row from a diffraction-grating spectrometer.

        Each column in the stripe image corresponds to a different wavelength.
        A linear calibration (slope, intercept) maps pixel column index to nm.

        Args:
            stripe: Single-row (or averaged) BGR image of the spectral stripe.
            calibration: (slope, intercept) for pixel→nm mapping.
                         Defaults to a linear mapping across wavelength_min–max.

        Returns:
            SpectralProfile across the calibrated wavelength range.
        """
        if stripe is None or stripe.size == 0:
            raise ValueError("Stripe is empty or None.")

        # Collapse to 1-D luminance profile
        if stripe.ndim == 3:
            gray = np.mean(stripe, axis=(0, 2))  # average over rows and channels
        else:
            gray = np.mean(stripe, axis=0)

        num_pixels = len(gray)

        if calibration is None:
            # Default: linear mapping from pixel 0 → wavelength_min, last pixel → wavelength_max
            slope = (self.wavelength_max - self.wavelength_min) / max(num_pixels - 1, 1)
            intercept = self.wavelength_min
        else:
            slope, intercept = calibration

        pixel_wavelengths = intercept + slope * np.arange(num_pixels)
        intensities_norm = gray / max(gray.max(), 1e-9)

        # Resample onto standard grid
        intensities = np.interp(
            self.wavelengths, pixel_wavelengths, intensities_norm
        )

        peak_idx = int(np.argmax(intensities))
        return SpectralProfile(
            wavelengths=self.wavelengths,
            intensities=intensities,
            peak_wavelength=float(self.wavelengths[peak_idx]),
            peak_intensity=float(intensities[peak_idx]),
        )

    def matches_urine_signature(self, profile: SpectralProfile) -> bool:
        """
        Check whether a spectral profile matches the known urine fluorescence
        signature (peak near 450 nm under UV).

        Args:
            profile: SpectralProfile to evaluate.

        Returns:
            True if the peak wavelength is within tolerance of the urine peak.
        """
        return abs(profile.peak_wavelength - self.urine_peak_nm) <= self.peak_tolerance_nm

    def generate_heatmap(
        self,
        profiles: List[SpectralProfile],
        positions: List[Tuple[int, int]],
        image_shape: Tuple[int, int],
    ) -> np.ndarray:
        """
        Generate a 2-D heatmap of urine-signature match scores for spatial
        mapping of hazardous regions.

        Args:
            profiles: Spectral profiles for each sampled patch.
            positions: (x, y) centre pixel for each profile.
            image_shape: (height, width) of the output heatmap.

        Returns:
            Floating-point heatmap array (0–1) of the same spatial extent.
        """
        heatmap = np.zeros(image_shape, dtype=np.float32)

        for profile, (px, py) in zip(profiles, positions):
            # Score: inversely proportional to distance from urine peak
            distance = abs(profile.peak_wavelength - self.urine_peak_nm)
            score = max(0.0, 1.0 - distance / (self.peak_tolerance_nm * 2))
            if 0 <= py < image_shape[0] and 0 <= px < image_shape[1]:
                heatmap[py, px] = score

        return heatmap
