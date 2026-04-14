"""
Hyperspectral Imaging Module

Full-pipeline hyperspectral data cube processing inspired by:
  - SpecimONE  : inline material classification for industrial sorting/QC
  - TRACER     : Dispersive Transform Spectroscopy (DTS) for hazard detection
  - imec       : snapshot mosaic hyperspectral sensors for art/materials

Architecture
────────────
  DataCube (H × W × λ)
      │
      ├─ band_extraction()       → select spectral bands of interest
      ├─ spectral_indices()      → ratios / derivatives (NDVI-style)
      ├─ abundance_map()         → pixel-level SAM scoring
      ├─ unmix()                 → linear spectral unmixing (endmembers)
      ├─ tracer_dts()            → Dispersive Transform Spectroscopy
      └─ export_false_colour()   → false-colour composite image

DataCube format
───────────────
Shape : (height, width, num_bands)  – "band-interleaved-by-line" (BIL-style)
Dtype : float32 or float64
Units : Reflectance [0–1] or radiance [W/(m²·sr·μm)]

References
──────────
SpecimONE : https://www.specim.com/specim-one/
TRACER    : https://www.spectral.com/tracer
imec HSI  : https://www.imechyperspectral.com/
SAM       : Kruse et al. (1993), Remote Sensing of Environment 44, 145–163.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.ndimage import gaussian_filter, gaussian_filter1d


# ---------------------------------------------------------------------------
# DataCube container
# ---------------------------------------------------------------------------


@dataclass
class DataCube:
    """
    Container for a hyperspectral image datacube.

    Attributes:
        data:        3-D float array (height, width, num_bands).
        wavelengths: 1-D array of centre wavelengths (nm) for each band.
        metadata:    Arbitrary sensor / acquisition metadata dict.
    """

    data: np.ndarray
    wavelengths: np.ndarray
    metadata: Dict = field(default_factory=dict)

    @property
    def height(self) -> int:
        return self.data.shape[0]

    @property
    def width(self) -> int:
        return self.data.shape[1]

    @property
    def num_bands(self) -> int:
        return self.data.shape[2]

    def pixel(self, row: int, col: int) -> np.ndarray:
        """Return the spectral vector for a single pixel."""
        return self.data[row, col, :]

    def band_image(self, band_index: int) -> np.ndarray:
        """Return a 2-D greyscale image for one spectral band."""
        return self.data[:, :, band_index]

    def wavelength_to_band(self, wavelength_nm: float) -> int:
        """Find the band index closest to a target wavelength (nm)."""
        return int(np.argmin(np.abs(self.wavelengths - wavelength_nm)))


# ---------------------------------------------------------------------------
# Hyperspectral Imager
# ---------------------------------------------------------------------------


class HyperspectralImager:
    """
    End-to-end hyperspectral image processing pipeline.

    Covers the full workflow from raw datacube to chemical maps:

      1. Radiometric pre-processing (dark subtraction, flat-field)
      2. Band selection and spectral index computation
      3. Pixel-level SAM classification
      4. Linear spectral unmixing
      5. TRACER-style Dispersive Transform Spectroscopy
      6. False-colour composite rendering
    """

    def __init__(
        self,
        wavelengths: Optional[np.ndarray] = None,
        num_bands: int = 128,
        wavelength_min_nm: float = 400.0,
        wavelength_max_nm: float = 1000.0,
    ):
        """
        Args:
            wavelengths: Explicit array of band centre wavelengths (nm).
                         If None, linearly spaced from min to max.
            num_bands: Used only when wavelengths is None.
            wavelength_min_nm: Minimum wavelength (nm).
            wavelength_max_nm: Maximum wavelength (nm).
        """
        if wavelengths is not None:
            self.wavelengths = np.asarray(wavelengths, dtype=float)
        else:
            self.wavelengths = np.linspace(
                wavelength_min_nm, wavelength_max_nm, num_bands
            )

    # ------------------------------------------------------------------
    # Simulation / acquisition
    # ------------------------------------------------------------------

    def simulate(
        self,
        height: int = 64,
        width: int = 64,
        noise_level: float = 0.03,
        rng: Optional[np.random.Generator] = None,
    ) -> DataCube:
        """
        Generate a synthetic datacube for testing and development.

        Inserts three synthetic spectral "target" regions to simulate
        chemical inclusions (e.g. a urine stain, a solvent spill, background).

        Args:
            height: Image height in pixels.
            width:  Image width in pixels.
            noise_level: Gaussian noise σ.
            rng: Random number generator for reproducibility.

        Returns:
            DataCube with simulated reflectance data.
        """
        if rng is None:
            rng = np.random.default_rng(42)

        num_bands = len(self.wavelengths)

        # Smooth background (broadband reflectance)
        base = rng.random((height, width, num_bands)).astype(np.float32) * 0.3
        for b in range(num_bands):
            base[:, :, b] = gaussian_filter(base[:, :, b], sigma=3)

        # Target region 1: UV-fluorescent urine spot (peak ~450 nm)
        peak_band = self.wavelength_to_band(450.0)
        base[10:25, 10:25, peak_band - 5 : peak_band + 5] += 0.6

        # Target region 2: NIR-absorbing solvent patch (~900 nm)
        peak_nir = self.wavelength_to_band(900.0)
        base[35:50, 35:50, peak_nir - 3 : peak_nir + 3] += 0.5

        # Add photon shot noise
        noise = rng.normal(0, noise_level, base.shape).astype(np.float32)
        data = np.clip(base + noise, 0.0, 1.0)

        return DataCube(
            data=data,
            wavelengths=self.wavelengths,
            metadata={"simulated": True, "noise_level": noise_level},
        )

    # ------------------------------------------------------------------
    # Pre-processing
    # ------------------------------------------------------------------

    def dark_subtract(self, cube: DataCube, dark_frame: np.ndarray) -> DataCube:
        """Subtract a dark-current reference frame from the datacube."""
        corrected = np.clip(cube.data - dark_frame[..., np.newaxis], 0.0, None)
        return DataCube(corrected, cube.wavelengths, cube.metadata)

    def flat_field_correct(
        self, cube: DataCube, white_reference: np.ndarray
    ) -> DataCube:
        """
        Divide by a white-reference flat-field to convert radiance → reflectance.

        Args:
            cube: Input DataCube (radiance).
            white_reference: 1-D mean spectrum of a calibration target (bands,).

        Returns:
            DataCube with values normalised to [0, 1] reflectance.
        """
        ref = white_reference[np.newaxis, np.newaxis, :]
        corrected = np.clip(cube.data / (ref + 1e-9), 0.0, 1.0)
        return DataCube(corrected, cube.wavelengths, cube.metadata)

    def spectral_smooth(self, cube: DataCube, sigma: float = 1.0) -> DataCube:
        """Apply Gaussian smoothing along the spectral axis."""
        smoothed = gaussian_filter1d(cube.data, sigma=sigma, axis=2)
        return DataCube(smoothed.astype(np.float32), cube.wavelengths, cube.metadata)

    # ------------------------------------------------------------------
    # Band selection & spectral indices
    # ------------------------------------------------------------------

    def wavelength_to_band(self, wavelength_nm: float) -> int:
        """Find the band index closest to a target wavelength (nm)."""
        return int(np.argmin(np.abs(self.wavelengths - wavelength_nm)))

    def extract_bands(
        self, cube: DataCube, band_indices: List[int]
    ) -> np.ndarray:
        """
        Extract specific spectral bands as a new array.

        Args:
            cube: Source DataCube.
            band_indices: List of band indices to extract.

        Returns:
            3-D array (H, W, len(band_indices)).
        """
        return cube.data[:, :, band_indices]

    def band_ratio_index(
        self, cube: DataCube, band_a: int, band_b: int
    ) -> np.ndarray:
        """
        Compute a normalised difference index between two bands:
            NDI = (A − B) / (A + B + ε)

        Analogous to NDVI for vegetation, but applicable to any two bands
        to highlight specific chemical absorption features.

        Args:
            cube: DataCube.
            band_a: Index of band A (numerator / first).
            band_b: Index of band B (denominator / second).

        Returns:
            2-D float32 NDI map (H, W), range [−1, 1].
        """
        a = cube.data[:, :, band_a].astype(float)
        b = cube.data[:, :, band_b].astype(float)
        ndi = (a - b) / (a + b + 1e-9)
        return ndi.astype(np.float32)

    # ------------------------------------------------------------------
    # Chemical mapping (SAM pixel classification)
    # ------------------------------------------------------------------

    def abundance_map(
        self,
        cube: DataCube,
        reference_spectrum: np.ndarray,
        normalise_pixels: bool = True,
    ) -> np.ndarray:
        """
        Compute a per-pixel SAM similarity map against a reference spectrum.

        Equivalent to SpecimONE's material-classification scoring at
        inference time.

        Args:
            cube: DataCube to analyse.
            reference_spectrum: 1-D reference spectrum (num_bands,).
            normalise_pixels: If True, Z-score normalise each pixel before SAM.

        Returns:
            2-D float32 similarity map (H, W) in [0, 1].
        """
        h, w, b = cube.data.shape
        flat = cube.data.reshape(-1, b).astype(float)

        if normalise_pixels:
            means = flat.mean(axis=1, keepdims=True)
            stds = flat.std(axis=1, keepdims=True) + 1e-9
            flat = (flat - means) / stds

        ref = np.asarray(reference_spectrum, dtype=float)
        if normalise_pixels:
            ref = (ref - ref.mean()) / (ref.std() + 1e-9)

        ref_unit = ref / (np.linalg.norm(ref) + 1e-9)
        norms = np.linalg.norm(flat, axis=1, keepdims=True) + 1e-9
        scores = (flat / norms) @ ref_unit

        return np.clip(scores.reshape(h, w), 0.0, 1.0).astype(np.float32)

    # ------------------------------------------------------------------
    # Linear spectral unmixing
    # ------------------------------------------------------------------

    def unmix(
        self,
        cube: DataCube,
        endmembers: Dict[str, np.ndarray],
    ) -> Dict[str, np.ndarray]:
        """
        Non-negative least-squares (NNLS) linear spectral unmixing.

        Each pixel spectrum is modelled as a linear combination of pure
        endmember spectra (e.g. urine, detergent, carpet fibre).
        Returns one abundance map per endmember.

        Args:
            cube: DataCube to unmix.
            endmembers: Dict mapping chemical name → 1-D endmember spectrum.

        Returns:
            Dict mapping chemical name → 2-D abundance map (H, W) [0, 1].
        """
        from scipy.optimize import nnls

        h, w, b = cube.data.shape
        em_names = list(endmembers.keys())
        em_matrix = np.column_stack(
            [endmembers[n].astype(float) for n in em_names]
        )  # shape (bands, n_endmembers)

        flat = cube.data.reshape(-1, b).astype(float)
        abundance_flat = np.zeros((flat.shape[0], len(em_names)), dtype=float)

        for i, pixel in enumerate(flat):
            coeffs, _ = nnls(em_matrix, pixel)
            # Normalise so abundances sum to ≤ 1
            total = coeffs.sum() + 1e-9
            abundance_flat[i] = coeffs / total

        maps = {}
        for j, name in enumerate(em_names):
            maps[name] = abundance_flat[:, j].reshape(h, w).astype(np.float32)

        return maps

    # ------------------------------------------------------------------
    # TRACER-style Dispersive Transform Spectroscopy
    # ------------------------------------------------------------------

    def tracer_dts(
        self,
        cube: DataCube,
        target_spectra: Dict[str, np.ndarray],
        sam_threshold: float = 0.85,
    ) -> Dict[str, np.ndarray]:
        """
        TRACER-inspired chemical detection using Dispersive Transform
        Spectroscopy (DTS).

        TRACER (by Spectral Sciences) detects gaseous solvents and solid
        explosives by computing a dispersive transform — essentially a
        matched filter in spectral space — on the full datacube.

        This implementation uses SAM as the matched filter kernel:
          - Each target chemical has a known spectral signature.
          - Every pixel is scored against every target.
          - A binary detection map is produced per chemical.

        Args:
            cube: DataCube (should be dark-subtracted and flat-field corrected).
            target_spectra: Dict mapping chemical name → reference spectrum.
            sam_threshold: SAM cosine similarity threshold for a detection.

        Returns:
            Dict mapping chemical name → 2-D binary detection map (H, W).
        """
        detection_maps: Dict[str, np.ndarray] = {}
        for chem_name, ref in target_spectra.items():
            score_map = self.abundance_map(cube, ref, normalise_pixels=True)
            detection_map = (score_map >= sam_threshold).astype(np.uint8)
            detection_maps[chem_name] = detection_map
        return detection_maps

    # ------------------------------------------------------------------
    # False-colour composite
    # ------------------------------------------------------------------

    def false_colour_composite(
        self,
        cube: DataCube,
        red_band: int,
        green_band: int,
        blue_band: int,
    ) -> np.ndarray:
        """
        Produce a false-colour BGR image by mapping three spectral bands
        to the display RGB channels.

        This technique is standard in remote sensing and industrial HSI
        (SpecimONE, ENVI) to reveal chemical features invisible to the
        naked eye.

        Args:
            cube: DataCube.
            red_band: Band index to display as red.
            green_band: Band index to display as green.
            blue_band: Band index to display as blue.

        Returns:
            8-bit BGR image (H, W, 3) suitable for display with OpenCV.
        """
        def _to_uint8(arr: np.ndarray) -> np.ndarray:
            lo, hi = arr.min(), arr.max()
            if hi - lo < 1e-9:
                return np.zeros_like(arr, dtype=np.uint8)
            return ((arr - lo) / (hi - lo) * 255).astype(np.uint8)

        r = _to_uint8(cube.data[:, :, red_band])
        g = _to_uint8(cube.data[:, :, green_band])
        b = _to_uint8(cube.data[:, :, blue_band])

        # OpenCV BGR ordering
        return np.stack([b, g, r], axis=2)

    # ------------------------------------------------------------------
    # Thermal infrared fingerprinting
    # ------------------------------------------------------------------

    def thermal_ir_fingerprint(
        self,
        cube: DataCube,
        thermal_band_indices: Optional[List[int]] = None,
    ) -> np.ndarray:
        """
        Extract thermal infrared (TIR) chemical signatures.

        Chemical substances absorb and emit thermal IR radiation at specific
        wavelengths determined by their molecular vibrational modes.  This
        is the physical basis of TRACER's TIR chemical detection and the
        OPUS FT-IR system's chemical mapping.

        Args:
            cube: DataCube covering the TIR range (8–14 µm / 8000–14000 nm).
                  For SWIR work, supply SWIR bands (1400–3000 nm).
            thermal_band_indices: List of band indices to include in the
                                  fingerprint.  Defaults to all bands in the
                                  upper half of the wavelength range.

        Returns:
            2-D float32 TIR activity map (H, W) — higher values indicate
            stronger thermal emission features.
        """
        if thermal_band_indices is None:
            mid = len(self.wavelengths) // 2
            thermal_band_indices = list(range(mid, len(self.wavelengths)))

        tir_data = cube.data[:, :, thermal_band_indices]
        # Mean radiance in the TIR window as activity proxy
        activity = tir_data.mean(axis=2).astype(np.float32)
        # Normalise to [0, 1]
        lo, hi = activity.min(), activity.max()
        if hi - lo > 1e-9:
            activity = (activity - lo) / (hi - lo)
        return activity
