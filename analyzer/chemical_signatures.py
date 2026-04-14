"""
Chemical Spectral Signature Library

Provides a curated library of known chemical spectral fingerprints drawn from
open-source reference databases, alongside SAM and PCA-based matching tools.

Open-source databases referenced
─────────────────────────────────
SDBS   – Spectral Database for Organic Compounds (AIST, Japan)
         https://sdbs.db.aist.go.jp/  (IR, Raman, NMR)
PubChem– NIH open chemistry database: structures, properties, spectra
         https://pubchem.ncbi.nlm.nih.gov/
ORD    – Open Reaction Database: reactions and physicochemical properties
         https://open-reaction-database.org/
spectrai – Open-source deep learning for spectral data
         https://github.com/conor-horgan/spectrai

SAM (Spectral Angle Mapper)
───────────────────────────
The standard algorithm for spectral identification in hyperspectral systems
(SpecimONE, ENVI, PySptools).  Measures the angle between two spectral
vectors in N-dimensional space — invariant to illumination intensity.

PCA (Principal Component Analysis)
───────────────────────────────────
Reduces the dimensionality of high-band-count data before SAM matching,
suppressing noise and speeding up classification.  This is the standard
pre-processing step in PySptools and ENVI hyperspectral analysis workflows.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.ndimage import gaussian_filter1d


# ---------------------------------------------------------------------------
# Chemical entry dataclass
# ---------------------------------------------------------------------------


@dataclass
class ChemicalSignature:
    """Spectral signature for a single chemical compound."""

    name: str
    cas_number: str                         # CAS registry number
    formula: str                            # Molecular formula
    description: str                        # Human-readable description
    # Raman: wavenumber (cm⁻¹) → normalised intensity
    raman_peaks: Dict[float, float] = field(default_factory=dict)
    # IR/FTIR: wavenumber (cm⁻¹) → normalised absorbance
    ir_peaks: Dict[float, float] = field(default_factory=dict)
    # LWIR: wavelength (µm) → absorption strength
    lwir_bands: Dict[float, float] = field(default_factory=dict)
    # UV-Vis fluorescence: wavelength (nm) → emission intensity
    uv_vis_peaks: Dict[float, float] = field(default_factory=dict)
    # IMS reduced ion mobility K₀ (cm²/(V·s))
    ims_k0: Optional[float] = None
    # PID correction factor relative to isobutylene
    pid_correction_factor: Optional[float] = None
    # Hazard classification: safe / irritant / toxic / explosive / narcotic
    hazard_class: str = "unknown"
    # Database cross-reference IDs
    pubchem_cid: Optional[int] = None
    sdbs_id: Optional[str] = None


# ---------------------------------------------------------------------------
# Built-in chemical library
# ---------------------------------------------------------------------------
# Spectral data are representative values from SDBS / PubChem references.
# All peak intensities are normalised to [0, 1] relative to the base peak.


CHEMICAL_LIBRARY: Dict[str, ChemicalSignature] = {
    # ── Urine markers ────────────────────────────────────────────────────
    "uric_acid": ChemicalSignature(
        name="Uric Acid",
        cas_number="69-93-2",
        formula="C5H4N4O3",
        description=(
            "Primary fluorescent marker in dried cat urine under UV illumination. "
            "Fluoresces yellow-green at ~450 nm when excited at 365-395 nm."
        ),
        raman_peaks={630.0: 0.6, 970.0: 1.0, 1260.0: 0.8, 1590.0: 0.5},
        ir_peaks={1660.0: 1.0, 1720.0: 0.8, 3100.0: 0.6},
        uv_vis_peaks={290.0: 1.0, 450.0: 0.7},   # emission under 365 nm excitation
        hazard_class="safe",
        pubchem_cid=1175,
        sdbs_id="NMR-CDS-00-040",
    ),
    "ammonia": ChemicalSignature(
        name="Ammonia",
        cas_number="7664-41-7",
        formula="NH3",
        description=(
            "Produced by bacterial decomposition of urea in cat urine. "
            "Strong LWIR absorption at 9.0 µm; high PID response at 11.7 eV lamp."
        ),
        raman_peaks={3350.0: 1.0, 1628.0: 0.4},
        ir_peaks={965.0: 1.0, 3336.0: 0.9},
        lwir_bands={9.0: 1.0, 10.75: 0.4},
        ims_k0=2.70,
        pid_correction_factor=10.0,
        hazard_class="irritant",
        pubchem_cid=222,
        sdbs_id="IR-NIDA-29513",
    ),

    # ── Common solvents / VOCs ────────────────────────────────────────────
    "ethanol": ChemicalSignature(
        name="Ethanol",
        cas_number="64-17-5",
        formula="C2H5OH",
        description="Common solvent; false positive under UV cleaning sprays.",
        raman_peaks={884.0: 1.0, 1055.0: 0.9, 1092.0: 0.7, 2930.0: 0.6},
        ir_peaks={1046.0: 1.0, 3350.0: 0.8},
        lwir_bands={9.5: 1.0},
        pid_correction_factor=9.0,
        hazard_class="safe",
        pubchem_cid=702,
        sdbs_id="IR-NIDA-29754",
    ),
    "benzene": ChemicalSignature(
        name="Benzene",
        cas_number="71-43-2",
        formula="C6H6",
        description="Aromatic hydrocarbon; characteristic Raman ring-breathing mode.",
        raman_peaks={992.0: 1.0, 3062.0: 0.6, 1178.0: 0.4},
        ir_peaks={671.0: 1.0, 1479.0: 0.7},
        lwir_bands={14.8: 0.8},
        pid_correction_factor=0.53,
        hazard_class="toxic",
        pubchem_cid=241,
        sdbs_id="IR-NIDA-29247",
    ),
    "acetone": ChemicalSignature(
        name="Acetone",
        cas_number="67-64-1",
        formula="C3H6O",
        description="Common cleaning solvent; UV fluorescence false positive.",
        raman_peaks={787.0: 0.8, 1710.0: 1.0, 2921.0: 0.6},
        ir_peaks={1715.0: 1.0, 3000.0: 0.5},
        lwir_bands={8.0: 1.0, 8.5: 0.6},
        pid_correction_factor=1.1,
        hazard_class="safe",
        pubchem_cid=180,
        sdbs_id="IR-NIDA-29244",
    ),

    # ── Threat substances ─────────────────────────────────────────────────
    "rdx": ChemicalSignature(
        name="RDX (Cyclonite)",
        cas_number="121-82-4",
        formula="C3H6N6O6",
        description="Military explosive; detected by IMS and Raman.",
        raman_peaks={885.0: 1.0, 1270.0: 0.7, 1570.0: 0.5},
        ir_peaks={1256.0: 1.0, 1540.0: 0.8},
        ims_k0=1.53,
        hazard_class="explosive",
        pubchem_cid=8490,
    ),
    "tatp": ChemicalSignature(
        name="TATP (Triacetone Triperoxide)",
        cas_number="17088-37-8",
        formula="C9H18O6",
        description="Peroxide explosive; strong Raman O-O stretch.",
        raman_peaks={829.0: 1.0, 1080.0: 0.6},
        ir_peaks={879.0: 1.0, 1200.0: 0.7},
        ims_k0=1.74,
        hazard_class="explosive",
        pubchem_cid=102138,
    ),

    # ── Environmental / atmospheric ────────────────────────────────────────
    "methane": ChemicalSignature(
        name="Methane",
        cas_number="74-82-8",
        formula="CH4",
        description="Greenhouse gas; LWIR absorption at 7.7 µm.",
        raman_peaks={1306.0: 1.0, 2917.0: 0.5},
        ir_peaks={3018.0: 1.0, 1306.0: 0.8},
        lwir_bands={7.7: 1.0},
        hazard_class="safe",
        pubchem_cid=297,
    ),

    # ── Cleaning products (UV false positives) ──────────────────────────
    "optical_brightener": ChemicalSignature(
        name="Optical Brightener (stilbene derivative)",
        cas_number="1533-45-5",
        formula="C28H20N2O2S2",
        description=(
            "Fabric brightener / laundry detergent additive that fluoresces "
            "bright blue-white under UV — the most common false positive in "
            "UV urine detection."
        ),
        uv_vis_peaks={350.0: 0.5, 430.0: 1.0},
        hazard_class="safe",
        pubchem_cid=5488061,
    ),
}


# ---------------------------------------------------------------------------
# Spectral Angle Mapper (SAM)
# ---------------------------------------------------------------------------


def spectral_angle_mapper(
    observed: np.ndarray,
    reference: np.ndarray,
) -> float:
    """
    Compute the Spectral Angle Mapper (SAM) similarity score.

    Returns cosine similarity in [−1, 1]; 1.0 = perfect match.
    Invariant to illumination scaling (multiplicative gain).

    This is the standard algorithm implemented in:
    - PySptools (https://pysptools.sourceforge.io/)
    - ENVI (Harris Geospatial)
    - Resonon SpectroNon

    Args:
        observed:  1-D observed spectral vector.
        reference: 1-D reference spectral vector (same length).

    Returns:
        Cosine similarity scalar.
    """
    obs = np.asarray(observed, dtype=float)
    ref = np.asarray(reference, dtype=float)
    if obs.shape != ref.shape:
        raise ValueError(f"Shape mismatch: {obs.shape} vs {ref.shape}.")
    n_obs = np.linalg.norm(obs)
    n_ref = np.linalg.norm(ref)
    if n_obs < 1e-12 or n_ref < 1e-12:
        return 0.0
    return float(np.clip(np.dot(obs / n_obs, ref / n_ref), -1.0, 1.0))


# ---------------------------------------------------------------------------
# Principal Component Analysis (PCA) pre-processing
# ---------------------------------------------------------------------------


class SpectralPCA:
    """
    PCA dimensionality reduction for hyperspectral data.

    Reduces the number of spectral bands before SAM matching to:
    1. Remove sensor noise concentrated in minor principal components.
    2. Speed up pixel-level classification over large datacubes.
    3. Suppress correlated background variation.

    This mirrors the pre-processing workflow in PySptools, ENVI, and the
    spectrai deep-learning framework.

    References:
        PySptools: https://pysptools.sourceforge.io/
        spectrai:  https://github.com/conor-horgan/spectrai
    """

    def __init__(self, n_components: int = 10):
        """
        Args:
            n_components: Number of principal components to retain.
        """
        self.n_components = n_components
        self._components: Optional[np.ndarray] = None   # (n_components, bands)
        self._mean: Optional[np.ndarray] = None
        self._explained_variance_ratio: Optional[np.ndarray] = None

    def fit(self, datacube: np.ndarray) -> "SpectralPCA":
        """
        Fit PCA on a datacube or a set of spectra.

        Args:
            datacube: 3-D (H, W, bands) or 2-D (N, bands) array.

        Returns:
            self (for method chaining).
        """
        if datacube.ndim == 3:
            h, w, b = datacube.shape
            X = datacube.reshape(-1, b).astype(float)
        else:
            X = datacube.astype(float)

        self._mean = X.mean(axis=0)
        X_centred = X - self._mean

        # Economy SVD: O(N·bands·k) vs full O(bands³)
        n_comp = min(self.n_components, X_centred.shape[0], X_centred.shape[1])
        U, S, Vt = np.linalg.svd(X_centred, full_matrices=False)

        self._components = Vt[:n_comp]
        total_var = float((S ** 2).sum()) + 1e-12
        self._explained_variance_ratio = (S[:n_comp] ** 2) / total_var
        return self

    def transform(self, spectra: np.ndarray) -> np.ndarray:
        """
        Project spectra into PCA space.

        Args:
            spectra: 2-D (N, bands) or 3-D (H, W, bands) array.

        Returns:
            2-D projected array (N, n_components) or 3-D (H, W, n_components).
        """
        if self._components is None:
            raise RuntimeError("Call fit() before transform().")
        original_shape = spectra.shape
        if spectra.ndim == 3:
            h, w, b = spectra.shape
            flat = spectra.reshape(-1, b).astype(float)
        else:
            flat = spectra.astype(float)

        projected = (flat - self._mean) @ self._components.T

        if len(original_shape) == 3:
            return projected.reshape(original_shape[0], original_shape[1], -1)
        return projected

    def fit_transform(self, datacube: np.ndarray) -> np.ndarray:
        """Fit and transform in one call."""
        return self.fit(datacube).transform(datacube)

    @property
    def explained_variance_ratio(self) -> Optional[np.ndarray]:
        """Fraction of variance explained by each principal component."""
        return self._explained_variance_ratio


# ---------------------------------------------------------------------------
# Spectral Library Matcher
# ---------------------------------------------------------------------------


class SpectralLibraryMatcher:
    """
    Matches observed spectra against the built-in chemical library using SAM.

    Optionally applies PCA pre-processing to denoise the observed spectrum
    before matching — the standard workflow in PySptools.
    """

    def __init__(
        self,
        library: Optional[Dict[str, ChemicalSignature]] = None,
        num_bands: int = 128,
        wavelength_min_nm: float = 200.0,
        wavelength_max_nm: float = 1000.0,
        pca_components: Optional[int] = None,
        sam_threshold: float = 0.90,
    ):
        """
        Args:
            library: Chemical signature library.  Defaults to CHEMICAL_LIBRARY.
            num_bands: Number of spectral bands in observed spectra.
            wavelength_min_nm: Minimum wavelength axis value (nm).
            wavelength_max_nm: Maximum wavelength axis value (nm).
            pca_components: If set, apply PCA with this many components
                            before matching.
            sam_threshold: Cosine similarity threshold for a positive match.
        """
        self.library = library or CHEMICAL_LIBRARY
        self.num_bands = num_bands
        self.wavelengths = np.linspace(wavelength_min_nm, wavelength_max_nm, num_bands)
        self.pca_components = pca_components
        self.sam_threshold = sam_threshold
        self._pca: Optional[SpectralPCA] = None

    def _build_reference_vector(self, signature: ChemicalSignature) -> np.ndarray:
        """
        Convert a ChemicalSignature's UV-Vis or Raman peaks into a dense
        spectral vector on the shared wavelength grid.

        Priority: uv_vis_peaks → raman_peaks (converted to nm equivalent).
        """
        vector = np.zeros(self.num_bands, dtype=float)
        peak_dict = signature.uv_vis_peaks if signature.uv_vis_peaks else {}
        if not peak_dict and signature.raman_peaks:
            # Approximate: map wavenumber (cm⁻¹) to nm via excitation 785 nm
            excitation_cm1 = 1e7 / 785.0
            for wn, intensity in signature.raman_peaks.items():
                emission_cm1 = excitation_cm1 - wn
                if emission_cm1 > 0:
                    emission_nm = 1e7 / emission_cm1
                    if self.wavelengths[0] <= emission_nm <= self.wavelengths[-1]:
                        peak_dict[emission_nm] = intensity

        for wavelength_nm, intensity in peak_dict.items():
            if self.wavelengths[0] <= wavelength_nm <= self.wavelengths[-1]:
                idx = int(np.argmin(np.abs(self.wavelengths - wavelength_nm)))
                vector[idx] = max(vector[idx], intensity)

        # Smooth to simulate instrument lineshape
        if vector.max() > 0:
            vector = gaussian_filter1d(vector, sigma=2.0)
        return vector

    def match(
        self,
        observed_spectrum: np.ndarray,
        top_n: int = 5,
    ) -> List[Tuple[str, float, ChemicalSignature]]:
        """
        Match an observed spectrum against the full chemical library.

        Args:
            observed_spectrum: 1-D array of length num_bands.
            top_n: Return only the top-N matches.

        Returns:
            List of (chemical_name, sam_score, ChemicalSignature) sorted by
            score descending.
        """
        obs = np.asarray(observed_spectrum, dtype=float)
        if len(obs) != self.num_bands:
            obs = np.interp(
                np.linspace(0, 1, self.num_bands),
                np.linspace(0, 1, len(obs)),
                obs,
            )

        results = []
        for key, sig in self.library.items():
            ref_vec = self._build_reference_vector(sig)
            if ref_vec.max() == 0:
                continue
            score = spectral_angle_mapper(obs, ref_vec)
            results.append((sig.name, score, sig))

        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_n]

    def match_all_sensors(
        self,
        hsi_spectrum: Optional[np.ndarray] = None,
        raman_peaks_cm1: Optional[List[float]] = None,
        ims_k0: Optional[float] = None,
        pid_ppm: Optional[float] = None,
        lwir_band_um: Optional[float] = None,
    ) -> List[Tuple[str, float]]:
        """
        Cross-sensor chemical identification using multi-modal evidence fusion.

        Scores each library entry across all available sensor modalities and
        returns a combined confidence score.

        Args:
            hsi_spectrum:     HSI/UV-Vis spectral vector.
            raman_peaks_cm1:  List of detected Raman shift peaks (cm⁻¹).
            ims_k0:           IMS reduced ion mobility (cm²/(V·s)).
            pid_ppm:          PID VOC concentration (ppm).
            lwir_band_um:     Dominant LWIR absorption wavelength (µm).

        Returns:
            List of (chemical_name, combined_confidence 0–1) sorted descending.
        """
        scores: Dict[str, float] = {key: 0.0 for key in self.library}
        evidence_count: Dict[str, int] = {key: 0 for key in self.library}

        # ── HSI / UV-Vis ────────────────────────────────────────────────
        if hsi_spectrum is not None:
            for key, sig in self.library.items():
                ref = self._build_reference_vector(sig)
                if ref.max() > 0:
                    s = max(0.0, spectral_angle_mapper(hsi_spectrum, ref))
                    scores[key] += s
                    evidence_count[key] += 1

        # ── Raman ───────────────────────────────────────────────────────
        if raman_peaks_cm1:
            for key, sig in self.library.items():
                if not sig.raman_peaks:
                    continue
                matches = 0
                for detected_wn in raman_peaks_cm1:
                    for ref_wn in sig.raman_peaks:
                        if abs(detected_wn - ref_wn) <= 15.0:
                            matches += 1
                if matches:
                    raman_score = min(1.0, matches / max(len(sig.raman_peaks), 1))
                    scores[key] += raman_score
                    evidence_count[key] += 1

        # ── IMS ─────────────────────────────────────────────────────────
        if ims_k0 is not None:
            for key, sig in self.library.items():
                if sig.ims_k0 is None:
                    continue
                diff = abs(ims_k0 - sig.ims_k0)
                if diff <= 0.1:
                    ims_score = 1.0 - diff / 0.1
                    scores[key] += ims_score
                    evidence_count[key] += 1

        # ── LWIR ────────────────────────────────────────────────────────
        if lwir_band_um is not None:
            for key, sig in self.library.items():
                if not sig.lwir_bands:
                    continue
                for band_um in sig.lwir_bands:
                    if abs(lwir_band_um - band_um) <= 0.3:
                        lwir_score = 1.0 - abs(lwir_band_um - band_um) / 0.3
                        scores[key] += lwir_score * sig.lwir_bands[band_um]
                        evidence_count[key] += 1
                        break

        # ── Combine: average across evidence modalities ──────────────────
        combined: List[Tuple[str, float]] = []
        for key, sig in self.library.items():
            n = evidence_count[key]
            if n == 0:
                continue
            avg_score = scores[key] / n
            combined.append((sig.name, round(avg_score, 4)))

        combined.sort(key=lambda x: x[1], reverse=True)
        return combined
