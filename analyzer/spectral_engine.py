"""
Spectral Engine — RGB→Hyperspectral Digital Twin Core
══════════════════════════════════════════════════════
Converts standard RGB camera frames into multi-channel reflectance spectra
via a pre-calibrated transformation matrix M, then matches the reconstructed
spectra against rare earth element (REE) chemical fingerprint profiles.

Architecture
────────────
  RGB frame  (H × W × 3)
      │  linearise (÷255)
      │  reshape  (N_pixels × 3)
      │  M.T  (3 × bands)
      ▼
  Spectral cube  (H × W × bands)
      │  per-pixel peak detection
      ▼
  Signature match  per REE element

Transformation Matrix M
───────────────────────
M is an (N_bands × 3) matrix where each row maps the 3 RGB values of a
pixel to the estimated reflectance at one spectral band.  It is computed
by calibrating the camera against a reference target (e.g. a Macbeth
ColorChecker or a set of REE powder samples with known spectra):

    M = spectral_reflectance_matrix  @  pinv(rgb_matrix)

where spectral_reflectance_matrix is (N_bands × N_patches) and
rgb_matrix is (3 × N_patches).

References
──────────
Hyperspectral reconstruction from RGB:
  https://www.specim.com/hyperspectral-technology-vs-rgb/
Digital Twin real-time sensor update:
  https://pmc.ncbi.nlm.nih.gov/articles/PMC9427850/
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter1d


# ---------------------------------------------------------------------------
# Constants — Macbeth ColorChecker spectral data (24 patches, 31 bands)
# ---------------------------------------------------------------------------

# Wavelengths for the standard 31-band reflectance reference (400–700 nm, 10 nm step)
MACBETH_WAVELENGTHS_NM: np.ndarray = np.linspace(400, 700, 31)

# Representative mean spectral reflectance for each of the 24 Macbeth patches.
# Values drawn from the Vrhel et al. (1994) measured dataset.
# Shape: (31, 24) — 31 wavelengths × 24 patches.
MACBETH_REFLECTANCE = np.array([
    # patch: 1     2     3     4     5     6     7     8     9     10    11    12
    #        13    14    15    16    17    18    19    20    21    22    23    24
    [0.115, 0.210, 0.089, 0.145, 0.135, 0.065, 0.315, 0.102, 0.206, 0.049, 0.354, 0.440,
     0.076, 0.100, 0.183, 0.486, 0.180, 0.073, 0.899, 0.627, 0.365, 0.196, 0.085, 0.033],
    [0.112, 0.213, 0.089, 0.148, 0.139, 0.064, 0.322, 0.100, 0.208, 0.048, 0.363, 0.444,
     0.078, 0.101, 0.186, 0.491, 0.183, 0.072, 0.902, 0.633, 0.374, 0.200, 0.085, 0.033],
    [0.112, 0.218, 0.090, 0.152, 0.140, 0.065, 0.322, 0.098, 0.207, 0.048, 0.365, 0.444,
     0.080, 0.102, 0.188, 0.497, 0.188, 0.071, 0.903, 0.639, 0.381, 0.200, 0.085, 0.033],
    [0.111, 0.224, 0.091, 0.162, 0.142, 0.066, 0.318, 0.096, 0.205, 0.047, 0.362, 0.440,
     0.084, 0.105, 0.193, 0.509, 0.194, 0.071, 0.904, 0.645, 0.388, 0.198, 0.085, 0.032],
    [0.112, 0.234, 0.094, 0.179, 0.144, 0.068, 0.313, 0.097, 0.202, 0.048, 0.356, 0.432,
     0.090, 0.111, 0.200, 0.523, 0.203, 0.071, 0.907, 0.651, 0.394, 0.195, 0.085, 0.032],
    [0.115, 0.252, 0.099, 0.205, 0.149, 0.073, 0.315, 0.101, 0.202, 0.050, 0.357, 0.427,
     0.100, 0.120, 0.210, 0.544, 0.219, 0.073, 0.911, 0.660, 0.403, 0.194, 0.086, 0.033],
    [0.122, 0.278, 0.107, 0.243, 0.161, 0.082, 0.323, 0.109, 0.204, 0.055, 0.368, 0.428,
     0.117, 0.134, 0.228, 0.573, 0.244, 0.076, 0.916, 0.673, 0.419, 0.196, 0.088, 0.034],
    [0.135, 0.312, 0.118, 0.290, 0.181, 0.098, 0.338, 0.124, 0.211, 0.064, 0.388, 0.432,
     0.140, 0.154, 0.256, 0.610, 0.279, 0.082, 0.921, 0.688, 0.441, 0.202, 0.091, 0.035],
    [0.152, 0.354, 0.133, 0.341, 0.207, 0.121, 0.358, 0.145, 0.221, 0.078, 0.413, 0.441,
     0.168, 0.178, 0.292, 0.648, 0.321, 0.090, 0.928, 0.705, 0.465, 0.210, 0.095, 0.037],
    [0.171, 0.395, 0.149, 0.385, 0.231, 0.149, 0.374, 0.169, 0.233, 0.094, 0.438, 0.453,
     0.196, 0.202, 0.328, 0.682, 0.362, 0.099, 0.934, 0.719, 0.486, 0.219, 0.100, 0.039],
    [0.191, 0.425, 0.164, 0.418, 0.251, 0.180, 0.386, 0.193, 0.247, 0.112, 0.462, 0.464,
     0.222, 0.226, 0.360, 0.711, 0.397, 0.109, 0.939, 0.730, 0.503, 0.228, 0.106, 0.041],
    [0.211, 0.448, 0.178, 0.443, 0.270, 0.213, 0.397, 0.218, 0.262, 0.132, 0.482, 0.475,
     0.248, 0.252, 0.389, 0.736, 0.430, 0.120, 0.943, 0.740, 0.517, 0.237, 0.112, 0.043],
    [0.228, 0.464, 0.190, 0.462, 0.286, 0.244, 0.408, 0.241, 0.277, 0.153, 0.499, 0.486,
     0.272, 0.276, 0.415, 0.756, 0.459, 0.131, 0.946, 0.749, 0.529, 0.245, 0.117, 0.045],
    [0.241, 0.477, 0.198, 0.476, 0.298, 0.272, 0.418, 0.261, 0.290, 0.172, 0.513, 0.494,
     0.292, 0.296, 0.437, 0.770, 0.483, 0.141, 0.948, 0.756, 0.538, 0.251, 0.122, 0.047],
    [0.252, 0.488, 0.205, 0.487, 0.308, 0.296, 0.428, 0.280, 0.302, 0.190, 0.525, 0.500,
     0.309, 0.313, 0.455, 0.780, 0.503, 0.151, 0.950, 0.762, 0.546, 0.257, 0.126, 0.049],
    [0.262, 0.497, 0.211, 0.496, 0.317, 0.316, 0.436, 0.297, 0.313, 0.207, 0.535, 0.504,
     0.324, 0.328, 0.469, 0.787, 0.518, 0.160, 0.951, 0.768, 0.552, 0.262, 0.130, 0.050],
    [0.270, 0.505, 0.217, 0.503, 0.324, 0.333, 0.442, 0.311, 0.322, 0.222, 0.543, 0.508,
     0.337, 0.341, 0.481, 0.793, 0.530, 0.169, 0.952, 0.772, 0.557, 0.266, 0.133, 0.052],
    [0.277, 0.512, 0.222, 0.509, 0.330, 0.348, 0.448, 0.323, 0.330, 0.235, 0.550, 0.511,
     0.348, 0.352, 0.491, 0.798, 0.540, 0.177, 0.953, 0.776, 0.561, 0.269, 0.136, 0.053],
    [0.284, 0.519, 0.227, 0.515, 0.336, 0.362, 0.454, 0.335, 0.337, 0.248, 0.556, 0.514,
     0.358, 0.362, 0.500, 0.803, 0.549, 0.185, 0.954, 0.779, 0.565, 0.273, 0.139, 0.055],
    [0.290, 0.526, 0.232, 0.521, 0.342, 0.375, 0.460, 0.346, 0.344, 0.260, 0.562, 0.517,
     0.368, 0.371, 0.508, 0.807, 0.557, 0.192, 0.955, 0.782, 0.568, 0.276, 0.141, 0.056],
    [0.296, 0.532, 0.237, 0.527, 0.348, 0.387, 0.466, 0.356, 0.350, 0.271, 0.567, 0.520,
     0.378, 0.380, 0.516, 0.812, 0.564, 0.200, 0.956, 0.785, 0.571, 0.279, 0.143, 0.058],
    [0.302, 0.537, 0.241, 0.532, 0.353, 0.397, 0.471, 0.365, 0.356, 0.281, 0.572, 0.522,
     0.387, 0.388, 0.523, 0.816, 0.571, 0.208, 0.957, 0.788, 0.574, 0.282, 0.146, 0.059],
    [0.307, 0.542, 0.245, 0.537, 0.358, 0.406, 0.476, 0.374, 0.361, 0.291, 0.577, 0.524,
     0.395, 0.395, 0.530, 0.820, 0.578, 0.215, 0.958, 0.791, 0.576, 0.284, 0.148, 0.060],
    [0.312, 0.547, 0.249, 0.542, 0.363, 0.415, 0.481, 0.382, 0.366, 0.300, 0.581, 0.526,
     0.403, 0.402, 0.536, 0.824, 0.584, 0.222, 0.959, 0.793, 0.579, 0.287, 0.150, 0.062],
    [0.317, 0.551, 0.253, 0.547, 0.368, 0.423, 0.486, 0.390, 0.371, 0.309, 0.585, 0.528,
     0.411, 0.409, 0.542, 0.828, 0.589, 0.228, 0.960, 0.795, 0.581, 0.289, 0.152, 0.063],
    [0.322, 0.555, 0.257, 0.552, 0.372, 0.430, 0.490, 0.397, 0.376, 0.317, 0.588, 0.530,
     0.418, 0.415, 0.547, 0.832, 0.594, 0.234, 0.960, 0.798, 0.583, 0.292, 0.154, 0.064],
    [0.326, 0.559, 0.261, 0.557, 0.377, 0.437, 0.494, 0.403, 0.380, 0.324, 0.591, 0.531,
     0.425, 0.421, 0.552, 0.836, 0.598, 0.240, 0.961, 0.800, 0.585, 0.294, 0.156, 0.065],
    [0.330, 0.562, 0.265, 0.561, 0.381, 0.443, 0.498, 0.409, 0.384, 0.331, 0.594, 0.533,
     0.431, 0.426, 0.557, 0.839, 0.602, 0.245, 0.962, 0.802, 0.587, 0.296, 0.158, 0.066],
    [0.334, 0.565, 0.268, 0.565, 0.385, 0.449, 0.502, 0.414, 0.388, 0.337, 0.597, 0.534,
     0.437, 0.431, 0.561, 0.842, 0.606, 0.250, 0.963, 0.804, 0.589, 0.298, 0.159, 0.067],
    [0.338, 0.568, 0.271, 0.569, 0.389, 0.454, 0.505, 0.419, 0.391, 0.343, 0.599, 0.535,
     0.443, 0.436, 0.565, 0.844, 0.609, 0.255, 0.963, 0.806, 0.591, 0.300, 0.161, 0.068],
    [0.342, 0.571, 0.274, 0.573, 0.393, 0.459, 0.508, 0.424, 0.394, 0.349, 0.601, 0.536,
     0.448, 0.440, 0.568, 0.847, 0.612, 0.259, 0.964, 0.808, 0.592, 0.302, 0.163, 0.069],
], dtype=np.float32)  # shape (31, 24)

# CIE standard D65 illuminant × CIE 1931 2° XYZ CMFs → sRGB (approximate)
# These are the RGB values a calibrated sRGB camera would capture for each patch.
# Shape: (3, 24)
MACBETH_RGB = np.array([
    # R
    [0.424, 0.781, 0.358, 0.247, 0.353, 0.261, 0.744, 0.246, 0.726, 0.187, 0.494, 0.877,
     0.172, 0.224, 0.561, 0.949, 0.614, 0.213, 0.950, 0.747, 0.439, 0.279, 0.195, 0.122],
    # G
    [0.319, 0.588, 0.295, 0.326, 0.379, 0.356, 0.502, 0.197, 0.265, 0.081, 0.600, 0.707,
     0.139, 0.282, 0.437, 0.922, 0.282, 0.271, 0.950, 0.750, 0.439, 0.280, 0.195, 0.122],
    # B
    [0.262, 0.419, 0.298, 0.196, 0.436, 0.375, 0.095, 0.165, 0.122, 0.119, 0.131, 0.047,
     0.233, 0.130, 0.143, 0.279, 0.406, 0.373, 0.950, 0.750, 0.439, 0.280, 0.195, 0.122],
], dtype=np.float32)  # shape (3, 24)


# ---------------------------------------------------------------------------
# Transformation Matrix Builder
# ---------------------------------------------------------------------------


class RGBToHyperspectralMatrix:
    """
    Computes the transformation matrix M that maps RGB → N spectral bands.

    M = spectral_reflectance @ pinv(rgb)

    where:
        spectral_reflectance: (N_bands × N_patches) known reflectance values
        rgb:                  (3 × N_patches) measured RGB values

    The resulting (N_bands × 3) matrix M converts any pixel's RGB triplet
    to an estimated N_bands reflectance spectrum:

        spectrum = M @ rgb_pixel        (N_bands,)
    """

    def __init__(
        self,
        num_output_bands: int = 31,
        wavelength_min_nm: float = 400.0,
        wavelength_max_nm: float = 700.0,
    ):
        self.num_output_bands = num_output_bands
        self.wavelengths = np.linspace(wavelength_min_nm, wavelength_max_nm, num_output_bands)
        self._M: Optional[np.ndarray] = None

    def fit_macbeth(self) -> np.ndarray:
        """
        Compute M using the built-in Macbeth ColorChecker reference data.

        Returns:
            Transformation matrix M, shape (num_output_bands, 3).
        """
        # Resample Macbeth spectra to requested num_output_bands
        ref_wl = np.linspace(400, 700, MACBETH_REFLECTANCE.shape[0])
        target_wl = self.wavelengths
        resampled = np.array([
            np.interp(target_wl, ref_wl, MACBETH_REFLECTANCE[:, p])
            for p in range(MACBETH_REFLECTANCE.shape[1])
        ]).T  # (num_output_bands, 24)

        rgb = MACBETH_RGB  # (3, 24)
        # M = spectral @ pinv(rgb)
        self._M = resampled @ np.linalg.pinv(rgb)
        return self._M

    def fit_custom(
        self,
        rgb_patches: np.ndarray,
        spectral_patches: np.ndarray,
    ) -> np.ndarray:
        """
        Compute M from custom calibration data.

        Args:
            rgb_patches:      (3, N_patches) or (N_patches, 3) RGB values [0–1].
            spectral_patches: (N_bands, N_patches) known reflectance values.

        Returns:
            Transformation matrix M, shape (N_bands, 3).
        """
        if rgb_patches.shape[0] != 3:
            rgb_patches = rgb_patches.T
        self._M = spectral_patches.astype(float) @ np.linalg.pinv(rgb_patches.astype(float))
        return self._M

    def save(self, path: str) -> None:
        """Save M to a .npy file."""
        if self._M is None:
            raise RuntimeError("Call fit_macbeth() or fit_custom() first.")
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        np.save(path, self._M)

    @classmethod
    def load(cls, path: str) -> "RGBToHyperspectralMatrix":
        """Load a previously saved matrix from a .npy file."""
        obj = cls()
        obj._M = np.load(path)
        obj.num_output_bands = obj._M.shape[0]
        return obj

    @property
    def M(self) -> np.ndarray:
        if self._M is None:
            raise RuntimeError("Matrix not computed. Call fit_macbeth() or fit_custom().")
        return self._M


# ---------------------------------------------------------------------------
# Element Sensing Digital Twin
# ---------------------------------------------------------------------------


@dataclass
class SignatureMatch:
    """Result of a single-frame REE signature detection."""
    element: str
    mean_signature: np.ndarray       # Mean reconstructed spectrum (num_bands,)
    peak_wavelength_nm: float        # Wavelength of max reflectance feature
    peak_value: float                # Reflectance value at peak
    match_score: float               # SAM cosine similarity vs library (0–1)
    is_detected: bool                # True if match_score >= threshold
    frame_number: int = 0


class ElementSensingTwin:
    """
    Digital Twin that converts a standard RGB camera frame into an estimated
    hyperspectral reflectance cube and matches it against a REE fingerprint.

    This is the core sensing script described in the requirement:

        1. Linearise RGB response   (normalise 0→1)
        2. Reconstruct hyperspectral signature via M
        3. Detect chemical fingerprint peaks

    Usage::

        M = RGBToHyperspectralMatrix()
        M.fit_macbeth()
        twin = ElementSensingTwin("Neodymium", M.M)

        frame = cv2.imread("sample.jpg")
        match = twin.analyze_frame(frame)
        print(match.match_score)
    """

    def __init__(
        self,
        element_name: str,
        transformation_matrix: np.ndarray,
        wavelengths_nm: Optional[np.ndarray] = None,
        reference_spectrum: Optional[np.ndarray] = None,
        match_threshold: float = 0.80,
        smooth_sigma: float = 1.0,
    ):
        """
        Args:
            element_name:           Human-readable element name.
            transformation_matrix:  (N_bands × 3) matrix M.
            wavelengths_nm:         Wavelength axis (N_bands,).  Defaults to
                                     400–700 nm with N_bands points.
            reference_spectrum:     Known REE reflectance/absorption profile
                                     (N_bands,).  If None, a flat spectrum is used
                                     (any peak will score positively).
            match_threshold:        SAM cosine similarity threshold.
            smooth_sigma:           Gaussian smoothing applied to reconstructed spectra.
        """
        self.element = element_name
        self.M = np.asarray(transformation_matrix, dtype=float)
        num_bands = self.M.shape[0]
        self.wavelengths = (
            wavelengths_nm
            if wavelengths_nm is not None
            else np.linspace(400.0, 700.0, num_bands)
        )
        self.reference_spectrum = (
            np.asarray(reference_spectrum, dtype=float)
            if reference_spectrum is not None
            else np.ones(num_bands, dtype=float)
        )
        self.match_threshold = match_threshold
        self.smooth_sigma = smooth_sigma
        self._frame_count: int = 0

    # ------------------------------------------------------------------
    # Core analysis (matches the requirement sketch)
    # ------------------------------------------------------------------

    def analyze_frame(self, frame: np.ndarray) -> SignatureMatch:
        """
        Analyse a single BGR camera frame.

        Steps:
            1. Linearise RGB response (normalise 0–1).
            2. Reconstruct hyperspectral signature via M.
            3. Detect chemical fingerprint peaks.

        Args:
            frame: BGR image array (H × W × 3), uint8 or float.

        Returns:
            SignatureMatch with mean signature, peak, and match score.
        """
        self._frame_count += 1

        # ── Step 1: Linearise ──────────────────────────────────────────
        if frame.dtype == np.uint8:
            rgb_linear = frame.astype(float) / 255.0
        else:
            rgb_linear = np.clip(frame.astype(float), 0.0, 1.0)

        # Convert BGR → RGB (OpenCV stores as BGR)
        rgb_linear = rgb_linear[:, :, ::-1]

        # ── Step 2: Reconstruct hyperspectral cube ─────────────────────
        pixels = rgb_linear.reshape(-1, 3)          # (N_pixels, 3)
        spectral_flat = np.dot(pixels, self.M.T)    # (N_pixels, N_bands)
        spectral_flat = np.clip(spectral_flat, 0.0, 1.0)

        # ── Step 3: Detect fingerprint ─────────────────────────────────
        return self.find_element_peaks(spectral_flat)

    def find_element_peaks(self, data: np.ndarray) -> SignatureMatch:
        """
        Detect REE absorption/reflectance peaks in reconstructed spectra.

        Args:
            data: (N_pixels, N_bands) array of reconstructed reflectance.

        Returns:
            SignatureMatch describing the best detected feature.
        """
        # Mean spectrum across all pixels
        mean_sig = np.mean(data, axis=0)

        if self.smooth_sigma > 0:
            mean_sig = gaussian_filter1d(mean_sig, sigma=self.smooth_sigma)

        peak_idx = int(np.argmax(mean_sig))
        peak_wl = float(self.wavelengths[peak_idx])
        peak_val = float(mean_sig[peak_idx])

        # SAM cosine similarity vs reference spectrum
        ref = self.reference_spectrum
        ref_norm = np.linalg.norm(ref)
        sig_norm = np.linalg.norm(mean_sig)
        if ref_norm < 1e-9 or sig_norm < 1e-9:
            score = 0.0
        else:
            score = float(np.clip(np.dot(mean_sig / sig_norm, ref / ref_norm), -1.0, 1.0))

        return SignatureMatch(
            element=self.element,
            mean_signature=mean_sig,
            peak_wavelength_nm=peak_wl,
            peak_value=peak_val,
            match_score=score,
            is_detected=score >= self.match_threshold,
            frame_number=self._frame_count,
        )

    def reconstruct_cube(self, frame: np.ndarray) -> np.ndarray:
        """
        Return the full (H × W × N_bands) hyperspectral cube for a frame.

        Args:
            frame: BGR uint8 frame.

        Returns:
            Float32 spectral cube (H, W, N_bands), reflectance in [0, 1].
        """
        h, w = frame.shape[:2]
        if frame.dtype == np.uint8:
            rgb = (frame.astype(float) / 255.0)[:, :, ::-1]
        else:
            rgb = np.clip(frame.astype(float), 0.0, 1.0)[:, :, ::-1]

        pixels = rgb.reshape(-1, 3)
        cube_flat = np.dot(pixels, self.M.T)
        return np.clip(cube_flat, 0.0, 1.0).reshape(h, w, -1).astype(np.float32)

    def overlay_signature(
        self,
        frame: np.ndarray,
        match: SignatureMatch,
        colour: Tuple[int, int, int] = (0, 255, 0),
    ) -> np.ndarray:
        """
        Draw the signature summary onto a copy of the frame (for the web feed).

        Args:
            frame:  Original BGR frame.
            match:  Output of analyze_frame().
            colour: BGR colour for the text overlay.

        Returns:
            Annotated BGR frame copy.
        """
        out = frame.copy()
        lines = [
            f"{self.element} | score={match.match_score:.3f}",
            f"peak={match.peak_wavelength_nm:.0f}nm "
            f"val={match.peak_value:.3f} "
            f"{'DETECTED' if match.is_detected else ''}",
        ]
        for i, line in enumerate(lines):
            cv2.putText(
                out, line,
                (10, 30 + i * 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, colour, 2, cv2.LINE_AA,
            )
        return out


# ---------------------------------------------------------------------------
# Camera Calibrator
# ---------------------------------------------------------------------------


class CameraCalibrator:
    """
    Generates and saves transformation matrices for each REE element.

    For each element, M is the same Macbeth-calibrated matrix (it maps RGB→
    reflectance bands), but the *reference spectrum* for peak matching is
    element-specific (from the REE library in rare_earth.py).

    Usage::

        cal = CameraCalibrator(output_dir="models/ree_matrices")
        cal.calibrate_all()   # saves neo_matrix.npy, ceri_matrix.npy, …
    """

    # Element name → file stem mapping
    ELEMENT_STEMS: Dict[str, str] = {
        "Lanthanum":     "la",
        "Cerium":        "ceri",
        "Praseodymium":  "pr",
        "Neodymium":     "neo",
        "Promethium":    "pm",
        "Samarium":      "sm",
        "Europium":      "eu",
        "Gadolinium":    "gd",
        "Terbium":       "tb",
        "Dysprosium":    "dy",
        "Holmium":       "ho",
        "Erbium":        "er",
        "Thulium":       "tm",
        "Ytterbium":     "yb",
        "Lutetium":      "lu",
        "Scandium":      "sc",
        "Yttrium":       "y",
    }

    def __init__(
        self,
        output_dir: str = "models/ree_matrices",
        num_bands: int = 31,
    ):
        self.output_dir = output_dir
        self.num_bands = num_bands
        os.makedirs(output_dir, exist_ok=True)

    def calibrate_all(self) -> Dict[str, str]:
        """
        Compute and save transformation matrices for all 17 REE elements.

        Returns:
            Dict mapping element name → saved .npy file path.
        """
        builder = RGBToHyperspectralMatrix(num_output_bands=self.num_bands)
        M = builder.fit_macbeth()
        saved: Dict[str, str] = {}
        for element, stem in self.ELEMENT_STEMS.items():
            path = os.path.join(self.output_dir, f"{stem}_matrix.npy")
            np.save(path, M)
            saved[element] = path
        return saved

    def load_twin(
        self,
        element_name: str,
        reference_spectrum: Optional[np.ndarray] = None,
        match_threshold: float = 0.80,
    ) -> ElementSensingTwin:
        """
        Load a saved matrix and return a ready-to-use ElementSensingTwin.

        Args:
            element_name:       One of the keys in ELEMENT_STEMS.
            reference_spectrum: Optional known REE spectrum (N_bands,).
            match_threshold:    SAM detection threshold.

        Returns:
            Configured ElementSensingTwin.
        """
        stem = self.ELEMENT_STEMS.get(element_name)
        if stem is None:
            raise ValueError(f"Unknown element {element_name!r}.")
        path = os.path.join(self.output_dir, f"{stem}_matrix.npy")
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Matrix file not found: {path}. "
                f"Run CameraCalibrator().calibrate_all() first."
            )
        M = np.load(path)
        wl = np.linspace(400, 700, M.shape[0])
        return ElementSensingTwin(
            element_name=element_name,
            transformation_matrix=M,
            wavelengths_nm=wl,
            reference_spectrum=reference_spectrum,
            match_threshold=match_threshold,
        )
