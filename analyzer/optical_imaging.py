"""
Optical Imaging Effects — OGI & Schlieren Simulation
═════════════════════════════════════════════════════
Provides digital signal-processing routines that simulate two key
"invisible-made-visible" imaging modalities used in chemical detection:

1. **Optical Gas Imaging (OGI)**
   Physical OGI cameras (e.g. Opgal EyeCGas) use cooled InSb detectors
   sensitive to the 3.2–3.4 µm band where hydrocarbons absorb IR strongly.
   The software equivalent reads a thermal / spectral channel and renders the
   gas as a smoke-like coloured plume on the camera feed.

   This module implements:
   • ``OGIFilter.apply()``  — applies per-pixel spectral band masking and
     false-colour mapping to a hyperspectral frame or a thermal image.
   • ``OGIFilter.plume_mask()`` — returns a single-channel 0/1 mask of the
     gas region so the frontend can overlay the ``.ogi-plume`` CSS class.

2. **Schlieren Imaging Simulation**
   Classical Schlieren optics (BOS — Background-Oriented Schlieren) visualise
   refractive-index gradients caused by density differences in a fluid/gas.
   The digital implementation uses **optical flow** between successive frames
   to detect regions where the refractive index changes rapidly — the
   "shadows" of invisible chemical leaks.

   This module implements:
   • ``SchlierenProcessor.compute()`` — computes a Schlieren-like gradient
     magnitude image from two consecutive frames using the Sobel operator
     and temporal differencing.

References
──────────
[1] OGI hydrocarbons: https://www.opgal.com/blog/blog/beyond-the-naked-eye-how-thermal-imaging-detects-invisible-leaks/
[2] Schlieren imaging:  https://www.reddit.com/r/Physics/comments/1iwlhsw/
[3] RT-CAN cross-attention RGB-thermal: https://arxiv.org/html/2403.17712v1
[4] Background subtraction: https://www.sciencedirect.com/article/abs/pii/S1077314224001802
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np
from scipy.ndimage import gaussian_filter


# ---------------------------------------------------------------------------
# Chemical OGI colour mapping
# ---------------------------------------------------------------------------


# Maps chemical / band label → RGB plume colour (0–255)
OGI_PALETTE: dict = {
    "methane":          (255, 140,  20),   # orange
    "ammonia":          (180, 255, 100),   # yellow-green
    "co2":              ( 80, 180, 255),   # sky-blue
    "voc":              (255,  80, 200),   # pink-purple
    "h2s":              (255, 230,  40),   # yellow
    "ammonium_nitrate": (200, 120, 255),   # violet
    "default":          (160, 230, 255),   # pale-blue
}


# ---------------------------------------------------------------------------
# 1. OGI Filter
# ---------------------------------------------------------------------------


@dataclass
class OGIFilter:
    """
    Applies Optical Gas Imaging false-colour rendering.

    Parameters
    ──────────
    target_chemical : str
        Name of the target chemical (used to select plume colour from
        ``OGI_PALETTE``).
    detection_threshold : float
        Spectral band intensity (0–1) above which a pixel is considered
        to contain the target gas.
    blend_alpha : float
        Opacity of the false-colour plume overlay (0 = invisible, 1 = opaque).
    blur_sigma : float
        Gaussian blur applied to the plume mask before blending, simulating
        the soft-edged "smoke" appearance in real OGI footage.
    """

    target_chemical: str = "methane"
    detection_threshold: float = 0.25
    blend_alpha: float = 0.55
    blur_sigma: float = 8.0

    def _plume_color(self) -> Tuple[int, int, int]:
        key = self.target_chemical.lower().replace(" ", "_")
        return OGI_PALETTE.get(key, OGI_PALETTE["default"])

    def plume_mask(
        self,
        signal_band: np.ndarray,
    ) -> np.ndarray:
        """
        Compute a soft plume mask from a single spectral or thermal channel.

        Parameters
        ──────────
        signal_band : np.ndarray, shape (H, W)
            Normalised (0–1) intensity for the absorption band of interest.
            E.g. the 3.3 µm SWIR band for methane, or the 1510 nm band for NH₃.

        Returns
        ───────
        mask : np.ndarray, shape (H, W), dtype float32
            Values in [0, 1]; 1 = strong gas presence, 0 = background.
        """
        band = np.asarray(signal_band, dtype=float)
        # Threshold + soft-clip
        mask = np.clip((band - self.detection_threshold) / (1.0 - self.detection_threshold + 1e-9), 0, 1)
        # Gaussian blur for smoke-like appearance
        if self.blur_sigma > 0:
            mask = gaussian_filter(mask.astype(np.float32), sigma=self.blur_sigma)
        return mask.astype(np.float32)

    def apply(
        self,
        rgb_frame: np.ndarray,
        signal_band: np.ndarray,
    ) -> np.ndarray:
        """
        Blend the false-colour gas plume onto an RGB camera frame.

        Parameters
        ──────────
        rgb_frame  : np.ndarray, shape (H, W, 3), dtype uint8
        signal_band: np.ndarray, shape (H, W),    dtype float32

        Returns
        ───────
        composite : np.ndarray, shape (H, W, 3), dtype uint8
            RGB frame with OGI plume overlay.
        """
        frame = rgb_frame.astype(np.float32) / 255.0
        mask = self.plume_mask(signal_band)[..., np.newaxis]   # (H, W, 1)

        r, g, b = self._plume_color()
        plume_color = np.array([b / 255.0, g / 255.0, r / 255.0], dtype=np.float32)  # BGR order

        # Screen blending: out = 1 − (1−src)×(1−overlay)
        plume_layer = plume_color * mask * self.blend_alpha
        composite = np.clip(1.0 - (1.0 - frame) * (1.0 - plume_layer), 0, 1)
        return (composite * 255).astype(np.uint8)

    def apply_to_hsi(
        self,
        rgb_frame: np.ndarray,
        hsi_datacube: np.ndarray,
        wavelengths_nm: np.ndarray,
        target_wavelength_nm: float = 1510.0,
        band_width_nm: float = 20.0,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract the relevant spectral band from an HSI datacube and apply OGI.

        Parameters
        ──────────
        hsi_datacube        : np.ndarray, shape (H, W, bands)
        wavelengths_nm      : np.ndarray, shape (bands,)
        target_wavelength_nm: float — centre wavelength of the absorption band
        band_width_nm       : float — half-width of the band window

        Returns
        ───────
        (composite_rgb, plume_mask)
        """
        lo = target_wavelength_nm - band_width_nm
        hi = target_wavelength_nm + band_width_nm
        band_idx = np.where((wavelengths_nm >= lo) & (wavelengths_nm <= hi))[0]

        if len(band_idx) == 0:
            return rgb_frame.copy(), np.zeros(rgb_frame.shape[:2], dtype=np.float32)

        signal_band = hsi_datacube[:, :, band_idx].mean(axis=2)
        # Normalise to 0–1
        vmin, vmax = signal_band.min(), signal_band.max()
        if vmax > vmin:
            signal_band = (signal_band - vmin) / (vmax - vmin)
        else:
            signal_band = np.zeros_like(signal_band)

        # Invert: high absorption → high gas signal
        signal_band = 1.0 - signal_band

        composite = self.apply(rgb_frame, signal_band)
        return composite, self.plume_mask(signal_band)


# ---------------------------------------------------------------------------
# 2. Schlieren Processor
# ---------------------------------------------------------------------------


@dataclass
class SchlierenProcessor:
    """
    Digital Schlieren Imaging simulation using temporal frame differencing
    and spatial gradient magnitude.

    Classical BOS (Background-Oriented Schlieren) detects refractive-index
    gradients by comparing a reference background with a "disturbed"
    background seen through a gas medium.  The digital equivalent tracks
    inter-frame changes in pixel intensity to reveal the "shadows" of
    density gradients (chemical plumes, thermal convection, etc.).

    Parameters
    ──────────
    sensitivity : float
        Amplification applied to the gradient image before display
        (higher = more sensitive to weak disturbances).
    blur_sigma : float
        Pre-filter applied to each frame to reduce sensor noise before
        computing gradients.
    temporal_weight : float
        Blending weight between spatial gradient (current frame) and
        temporal difference (between frames).  0 = pure spatial,
        1 = pure temporal.
    """

    sensitivity: float = 4.0
    blur_sigma: float = 1.5
    temporal_weight: float = 0.5
    _prev_frame: Optional[np.ndarray] = field(default=None, init=False, repr=False)

    def _to_gray(self, frame: np.ndarray) -> np.ndarray:
        """Convert BGR/RGB frame to float32 greyscale [0,1]."""
        if frame.ndim == 3:
            # Luminance weights (ITU-R BT.601)
            gray = (
                0.299 * frame[:, :, 2].astype(float) +
                0.587 * frame[:, :, 1].astype(float) +
                0.114 * frame[:, :, 0].astype(float)
            ) / 255.0
        else:
            gray = frame.astype(float) / (frame.max() + 1e-9)
        return gray.astype(np.float32)

    def _sobel_gradient(self, gray: np.ndarray) -> np.ndarray:
        """Return Sobel gradient magnitude (0–1)."""
        if self.blur_sigma > 0:
            gray = gaussian_filter(gray, sigma=self.blur_sigma)
        # Simple finite-difference Sobel
        gx = np.zeros_like(gray)
        gy = np.zeros_like(gray)
        gx[:, 1:-1] = gray[:, 2:] - gray[:, :-2]
        gy[1:-1, :] = gray[2:, :] - gray[:-2, :]
        mag = np.sqrt(gx ** 2 + gy ** 2)
        top = np.percentile(mag, 99) + 1e-9
        return np.clip(mag / top, 0, 1)

    def compute(self, frame: np.ndarray) -> np.ndarray:
        """
        Compute a Schlieren-like visualisation image.

        Parameters
        ──────────
        frame : np.ndarray, shape (H, W, 3) or (H, W), dtype uint8

        Returns
        ───────
        schlieren : np.ndarray, shape (H, W, 3), dtype uint8
            False-colour Schlieren image.  Regions of high refractive-index
            gradient appear bright (white/yellow); stable regions are dark.
        """
        gray = self._to_gray(frame)
        spatial = self._sobel_gradient(gray)

        if self._prev_frame is not None:
            temporal = np.abs(gray - self._prev_frame)
            combined = (
                (1.0 - self.temporal_weight) * spatial
                + self.temporal_weight * temporal
            )
        else:
            combined = spatial

        self._prev_frame = gray.copy()

        # Amplify and clip
        enhanced = np.clip(combined * self.sensitivity, 0, 1)

        # False colour: dark blue → cyan → white (like real Schlieren photos)
        h, w = enhanced.shape
        out = np.zeros((h, w, 3), dtype=np.uint8)
        v = enhanced
        out[:, :, 0] = (np.clip(1.0 - v * 2, 0, 1) * 255).astype(np.uint8)   # B
        out[:, :, 1] = (np.clip(v * 2 - 0.5, 0, 1) * 255).astype(np.uint8)   # G
        out[:, :, 2] = (np.clip(v * 4 - 2, 0, 1) * 255).astype(np.uint8)     # R
        return out

    def compute_mask(self, frame: np.ndarray, threshold: float = 0.15) -> np.ndarray:
        """
        Return a binary mask (H, W) uint8 where 255 = Schlieren anomaly detected.

        Useful for triggering the ``.schlieren-overlay`` CSS class in the
        React frontend.
        """
        gray = self._to_gray(frame)
        spatial = self._sobel_gradient(gray)
        if self._prev_frame is not None:
            temporal = np.abs(gray - self._prev_frame)
            combined = (
                (1.0 - self.temporal_weight) * spatial
                + self.temporal_weight * temporal
            )
        else:
            combined = spatial
        self._prev_frame = gray.copy()
        return ((combined * self.sensitivity) > threshold).astype(np.uint8) * 255


# ---------------------------------------------------------------------------
# 3. Cross-attention RGB-Thermal dual-stream (lightweight simulation)
# ---------------------------------------------------------------------------


class RGBThermalCrossAttention:
    """
    Lightweight simulation of the RT-CAN cross-attention dual-stream architecture.

    In a full RT-CAN implementation, two CNN encoders process the RGB stream
    and the thermal/spectral stream separately, then cross-attention layers
    let each stream "attend" to the other.  Here we implement the same
    *concept* using simple per-pixel weighting, suitable for real-time use
    without a GPU.

    The attention weight for each pixel is derived from the thermal anomaly
    strength: pixels with high thermal anomaly receive higher weight in the
    fusion, pulling the output towards the thermal channel where "textureless"
    gas regions are more visible.

    Reference
    ─────────
    RT-CAN: https://arxiv.org/html/2403.17712v1
    """

    def __init__(self, thermal_weight: float = 0.6) -> None:
        """
        Parameters
        ──────────
        thermal_weight : float
            Base weight given to the thermal/spectral channel when anomaly is
            detected (0.0 = RGB only, 1.0 = thermal only).
        """
        self.thermal_weight = thermal_weight

    def fuse(
        self,
        rgb_frame: np.ndarray,
        thermal_frame: np.ndarray,
        anomaly_mask: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Fuse RGB and thermal frames using attention-weighted blending.

        Parameters
        ──────────
        rgb_frame     : np.ndarray (H, W, 3) uint8 — standard camera feed
        thermal_frame : np.ndarray (H, W, 3) uint8 — false-colour thermal / OGI
        anomaly_mask  : np.ndarray (H, W)    float32 in [0,1], optional.
                        If supplied, pixels with high anomaly score are
                        weighted towards the thermal channel.

        Returns
        ───────
        fused : np.ndarray (H, W, 3) uint8
        """
        rgb = rgb_frame.astype(np.float32)
        thm = thermal_frame.astype(np.float32)

        if anomaly_mask is not None:
            w = np.clip(anomaly_mask * self.thermal_weight, 0, 1)[..., np.newaxis]
        else:
            w = self.thermal_weight

        fused = np.clip(rgb * (1.0 - w) + thm * w, 0, 255).astype(np.uint8)
        return fused
