"""
Multi-Sensor Chemical Analysis Engine

Implements three core sensor types that convert raw physical signals into
readable chemical signatures, following the modular signal-to-signature
conversion architecture:

  ┌─────────────────────────────────────────────────────────────────┐
  │  SENSOR TYPE            PHYSICAL SIGNAL      OUTPUT             │
  │  HyperspectralSensor    3D datacube (x,y,λ)  Spatial signature  │
  │  SpectroscopicSensor    1D wavelength array  Molecular fingerpt  │
  │  GasSensorArray         Voltage readings     Concentration (ppm) │
  └─────────────────────────────────────────────────────────────────┘

Hardware integration notes
──────────────────────────
Nano 33 BLE Sense / Arduino Nicla Vision
  Replace simulated data with an I2C read from the onboard sensors:
  
      import serial
      port = serial.Serial('/dev/ttyACM0', 115200, timeout=1)
      raw = port.readline().decode().strip().split(',')
      voltages = [float(v) for v in raw]

ESP8266 / ESP32 (MicroPython)
  Use machine.I2C or machine.ADC for multi-channel gas sensors:
  
      from machine import ADC, Pin
      adc = ADC(Pin(34))
      voltage = adc.read() / 4095.0 * 3.3  # 12-bit, 3.3 V reference

Raspberry Pi + Specim / Resonon camera SDK
  Interface via the vendor SDK to fill the HSI datacube:
  
      from specim_sdk import Camera
      cam = Camera()
      datacube = cam.capture()   # returns np.ndarray (H, W, bands)

Signal conditioning
───────────────────
All sensors implement optional Z-score normalisation (zero mean, unit
variance) to remove baseline drift — a standard practice in forensic and
analytical chemistry.  A Gaussian low-pass pre-filter reduces high-frequency
sensor noise before normalisation.

References
──────────
[1] Spectral Angle Mapper: https://resonon.com/blog-spectronon-hyperspectral-imaging-software
[2] Nano 33 BLE multi-sensor edge: https://www.instructables.com/Habitat-Signature-Analyzer-HSA-a-Multi-Sensor-Edge/
[3] Z-score baseline correction: https://www.sciencedirect.com/science/article/abs/pii/S0925400520305803
[4] Electrochemical gas arrays: https://pmc.ncbi.nlm.nih.gov/articles/PMC10054971/
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.ndimage import gaussian_filter1d


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------


class ChemicalSensor(ABC):
    """
    Abstract base class for signal-to-chemical-signature conversion.

    All sensor types share a common interface:
      raw_data  →  process_signal()  →  chemical signature (np.ndarray)
    """

    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def process_signal(self, raw_data) -> np.ndarray:
        """Convert a raw physical signal into a normalised chemical signature."""

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name!r})"


# ---------------------------------------------------------------------------
# Sensor 1 – Hyperspectral Camera (Visual / Spatial signal)
# ---------------------------------------------------------------------------


class HyperspectralSensor(ChemicalSensor):
    """
    Processes a 3-D datacube (height × width × spectral_bands) captured by
    a hyperspectral camera (e.g. Specim FX series, Resonon Pika).

    The sensor extracts a mean spectral signature from any rectangular region
    of interest (ROI) in the image, applies Gaussian smoothing to reduce
    photon-noise, and optionally Z-score normalises the result.

    Typical use:
        sensor = HyperspectralSensor("Specim FX10")
        cube = sdk.capture()   # shape (H, W, 224)
        sig  = sensor.process_signal(cube)
    """

    def __init__(
        self,
        name: str = "Hyperspectral Camera",
        roi: Optional[Tuple[int, int, int, int]] = None,
        smooth_sigma: float = 1.0,
        normalise: bool = True,
    ):
        """
        Args:
            name: Human-readable sensor name.
            roi: Region of interest (row_start, row_end, col_start, col_end).
                 Defaults to the central 20×20 pixel block.
            smooth_sigma: Gaussian smoothing σ applied along the spectral axis.
            normalise: If True, apply Z-score normalisation to the signature.
        """
        super().__init__(name)
        self.roi = roi
        self.smooth_sigma = smooth_sigma
        self.normalise = normalise

    def process_signal(self, datacube: np.ndarray) -> np.ndarray:
        """
        Extract a mean spectral signature from the datacube ROI.

        Args:
            datacube: 3-D NumPy array of shape (H, W, bands), dtype float32/64.

        Returns:
            1-D spectral signature array of length ``bands``.
        """
        if datacube.ndim != 3:
            raise ValueError(
                f"Expected a 3-D datacube (H, W, bands), got shape {datacube.shape}."
            )

        h, w, bands = datacube.shape
        if self.roi is None:
            # Default: centre 20×20 patch (or full image if smaller)
            r0 = max(0, h // 2 - 10)
            r1 = min(h, h // 2 + 10)
            c0 = max(0, w // 2 - 10)
            c1 = min(w, w // 2 + 10)
        else:
            r0, r1, c0, c1 = self.roi

        signature = np.mean(datacube[r0:r1, c0:c1, :], axis=(0, 1))

        if self.smooth_sigma > 0:
            signature = gaussian_filter1d(signature, sigma=self.smooth_sigma)

        if self.normalise:
            signature = _zscore(signature)

        return signature

    def extract_pixel_signature(
        self, datacube: np.ndarray, row: int, col: int
    ) -> np.ndarray:
        """
        Return the full spectral vector for a single pixel.

        Args:
            datacube: 3-D datacube (H, W, bands).
            row: Pixel row index.
            col: Pixel column index.

        Returns:
            1-D spectral signature for that pixel.
        """
        if datacube.ndim != 3:
            raise ValueError("Expected 3-D datacube.")
        sig = datacube[row, col, :].astype(float)
        if self.smooth_sigma > 0:
            sig = gaussian_filter1d(sig, sigma=self.smooth_sigma)
        if self.normalise:
            sig = _zscore(sig)
        return sig

    def chemical_map(
        self,
        datacube: np.ndarray,
        library_spectrum: np.ndarray,
        threshold: float = 0.95,
    ) -> np.ndarray:
        """
        Produce a 2-D spatial map of SAM similarity scores for every pixel.

        This mirrors SpecimONE's inline material-classification approach:
        each pixel's spectral vector is compared to a target library spectrum.

        Args:
            datacube: 3-D datacube (H, W, bands).
            library_spectrum: Reference spectral signature (bands,).
            threshold: SAM cosine similarity threshold.

        Returns:
            2-D float32 array (H, W) with similarity scores in [0, 1].
        """
        h, w, bands = datacube.shape
        flat = datacube.reshape(-1, bands).astype(float)
        scores = _sam_similarity_batch(flat, library_spectrum)
        return scores.reshape(h, w).astype(np.float32)


# ---------------------------------------------------------------------------
# Sensor 2 – Near-Infrared / Raman / Forensic (Wavelength signal)
# ---------------------------------------------------------------------------


class SpectroscopicSensor(ChemicalSensor):
    """
    Processes a 1-D wavelength-intensity array from near-infrared (NIR),
    Raman, or UV-Vis spectrometers.

    Signal conditioning pipeline:
      1. Gaussian low-pass filter  – removes high-frequency photon shot noise.
      2. Z-score normalisation     – removes baseline drift (DC offset & slope).
      3. Peak detection            – identifies molecular absorption/emission peaks.

    Matches instruments such as:
    - Ocean Insight USB4000 (UV-Vis-NIR)
    - Wasatch Photonics Raman spectrometer
    - imec hyperspectral sensor snapshot mosaic
    - OPUS FT-IR (Fourier-transform infrared)

    Reference: Z-score normalisation for baseline drift:
    https://www.sciencedirect.com/science/article/abs/pii/S0925400520305803
    """

    def __init__(
        self,
        name: str = "NIR Spectrometer",
        smooth_sigma: float = 2.0,
        peak_min_height: float = 1.5,
        peak_min_distance: int = 10,
    ):
        """
        Args:
            name: Human-readable sensor name.
            smooth_sigma: Gaussian σ for noise suppression (spectral bins).
            peak_min_height: Minimum Z-score height to report a peak.
            peak_min_distance: Minimum number of bins between adjacent peaks.
        """
        super().__init__(name)
        self.smooth_sigma = smooth_sigma
        self.peak_min_height = peak_min_height
        self.peak_min_distance = peak_min_distance

    def process_signal(self, wavelengths: np.ndarray) -> np.ndarray:
        """
        Normalise a raw wavelength-intensity spectrum.

        Args:
            wavelengths: 1-D array of raw intensity values (any length).

        Returns:
            Z-score normalised, noise-filtered 1-D spectrum.
        """
        spectrum = np.asarray(wavelengths, dtype=float)
        if spectrum.ndim != 1:
            raise ValueError("Expected a 1-D wavelength intensity array.")
        if len(spectrum) < 3:
            raise ValueError("Spectrum must have at least 3 data points.")

        # Step 1: Gaussian low-pass filter (removes high-frequency noise)
        if self.smooth_sigma > 0:
            spectrum = gaussian_filter1d(spectrum, sigma=self.smooth_sigma)

        # Step 2: Z-score normalisation (removes baseline drift)
        spectrum = _zscore(spectrum)

        return spectrum

    def find_peaks(self, normalised_spectrum: np.ndarray) -> np.ndarray:
        """
        Identify molecular fingerprint peaks in a normalised spectrum.

        Uses a simple local-maximum search with minimum-height and
        minimum-distance constraints (avoids scipy.signal dependency).

        Args:
            normalised_spectrum: Output of process_signal().

        Returns:
            1-D integer array of peak indices.
        """
        peaks = []
        n = len(normalised_spectrum)
        for i in range(1, n - 1):
            if normalised_spectrum[i] < self.peak_min_height:
                continue
            if normalised_spectrum[i] <= normalised_spectrum[i - 1]:
                continue
            if normalised_spectrum[i] <= normalised_spectrum[i + 1]:
                continue
            # Enforce minimum distance from last accepted peak
            if peaks and (i - peaks[-1]) < self.peak_min_distance:
                # Keep the taller one
                if normalised_spectrum[i] > normalised_spectrum[peaks[-1]]:
                    peaks[-1] = i
            else:
                peaks.append(i)
        return np.array(peaks, dtype=int)


# ---------------------------------------------------------------------------
# Sensor 3 – Electrochemical / Gas Sensor Array (Electrical signal)
# ---------------------------------------------------------------------------


@dataclass
class GasReading:
    """Result from a single gas sensor channel."""

    channel: int
    voltage: float
    concentration_ppm: float
    gas_name: str = "unknown"


class GasSensorArray(ChemicalSensor):
    """
    Converts a multi-channel voltage array from an electrochemical gas sensor
    (e.g. MQ-series, SGP30, CCS811, or a custom I2C array on a Nano 33 BLE)
    into gas concentrations in parts-per-million (ppm).

    Each channel has its own voltage-to-ppm sensitivity calibration:
        concentration_ppm = (voltage - offset) * sensitivity

    Calibration defaults to 12.5 ppm/V as a linear approximation, matching
    typical MQ-series sensor sensitivity curves in their linear operating range.

    Hardware wiring (Nano 33 BLE / ESP8266):
        - Connect AOUT of each MQ sensor to an ADC pin.
        - 3.3 V reference; 12-bit ADC (ESP32) or 10-bit (ESP8266/Nano 33).
        - Read voltage:
              nano33: voltage = analogRead(A0) / 1023.0 * 3.3
              esp32:  voltage = analogRead(34) / 4095.0 * 3.3
        - Pass voltage list to process_signal().

    Reference: Electrochemical sensor arrays:
    https://pmc.ncbi.nlm.nih.gov/articles/PMC10054971/
    """

    DEFAULT_SENSITIVITY = 12.5  # ppm per volt

    def __init__(
        self,
        name: str = "Gas Sensor Array",
        channel_names: Optional[List[str]] = None,
        sensitivities: Optional[List[float]] = None,
        offsets: Optional[List[float]] = None,
    ):
        """
        Args:
            name: Human-readable sensor name.
            channel_names: Gas names for each channel (e.g. ["NH3", "H2S"]).
            sensitivities: ppm-per-volt sensitivity per channel.
            offsets: Voltage offset per channel (zero-concentration baseline).
        """
        super().__init__(name)
        self.channel_names = channel_names or []
        self.sensitivities = sensitivities or []
        self.offsets = offsets or []

    def process_signal(self, voltages) -> np.ndarray:
        """
        Convert voltage readings to gas concentrations in ppm.

        Args:
            voltages: List or array of voltage values, one per sensor channel.

        Returns:
            1-D float array of concentrations in ppm (one per channel).
        """
        v = np.asarray(voltages, dtype=float)
        if v.ndim != 1:
            raise ValueError("Expected a 1-D array of voltage readings.")

        n = len(v)
        offsets = np.array(
            self.offsets[:n] + [0.0] * max(0, n - len(self.offsets))
        )
        sensitivities = np.array(
            self.sensitivities[:n]
            + [self.DEFAULT_SENSITIVITY] * max(0, n - len(self.sensitivities))
        )

        concentration = (v - offsets) * sensitivities
        return np.clip(concentration, 0.0, None)  # ppm cannot be negative

    def get_readings(self, voltages) -> List[GasReading]:
        """
        Return structured GasReading objects for each channel.

        Args:
            voltages: List or array of voltage values.

        Returns:
            List of GasReading objects.
        """
        ppm = self.process_signal(voltages)
        readings = []
        for i, (v, c) in enumerate(zip(voltages, ppm)):
            gas = self.channel_names[i] if i < len(self.channel_names) else "unknown"
            readings.append(
                GasReading(channel=i, voltage=float(v), concentration_ppm=float(c), gas_name=gas)
            )
        return readings


# ---------------------------------------------------------------------------
# Spectral library simulation helper
# ---------------------------------------------------------------------------


def simulate_hsi_sensor(
    spatial_res: Tuple[int, int] = (64, 64),
    bands: int = 128,
    noise_level: float = 0.05,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """
    Simulate a Hyperspectral Image (HSI) datacube.

    In a real deployment, replace this function with a call to the camera SDK:
        from specim_sdk import Camera
        return Camera().capture()   # returns np.ndarray (H, W, bands)

    Args:
        spatial_res: (height, width) of the simulated image.
        bands: Number of spectral bands.
        noise_level: Gaussian noise standard deviation added to simulate
                     photon shot noise and dark current.
        rng: Optional NumPy random Generator for reproducible results.

    Returns:
        Simulated datacube of shape (*spatial_res, bands), float32.
    """
    if rng is None:
        rng = np.random.default_rng()
    cube = rng.random((*spatial_res, bands)).astype(np.float32)
    if noise_level > 0:
        cube += rng.normal(0, noise_level, cube.shape).astype(np.float32)
    return np.clip(cube, 0.0, 1.0)


def match_spectral_signature(
    observed_spectrum: np.ndarray,
    library_spectrum: np.ndarray,
    threshold: float = 0.95,
) -> Tuple[bool, float]:
    """
    Spectral Angle Mapper (SAM) signature matching.

    Computes the cosine similarity between the observed and library spectral
    vectors.  A high cosine similarity (close to 1.0) indicates that the
    spectra point in the same direction in N-dimensional space — a reliable
    indicator of chemical identity regardless of illumination intensity.

    SAM is the standard algorithm used by tools such as:
    - Resonon SpectroNon  (https://resonon.com/blog-spectronon-hyperspectral-imaging-software)
    - ENVI (Harris Geospatial)
    - SpecimONE

    Args:
        observed_spectrum: 1-D array of measured spectral intensities.
        library_spectrum:  1-D reference array from a spectral library.
        threshold:         Cosine similarity threshold for a positive match.

    Returns:
        Tuple of (is_match: bool, similarity: float).
    """
    obs = np.asarray(observed_spectrum, dtype=float)
    lib = np.asarray(library_spectrum, dtype=float)

    if obs.shape != lib.shape:
        raise ValueError(
            f"Shape mismatch: observed {obs.shape} vs library {lib.shape}."
        )

    obs_norm_val = np.linalg.norm(obs)
    lib_norm_val = np.linalg.norm(lib)

    if obs_norm_val == 0 or lib_norm_val == 0:
        return False, 0.0

    obs_unit = obs / obs_norm_val
    lib_unit = lib / lib_norm_val
    similarity = float(np.dot(obs_unit, lib_unit))
    # Clamp to [-1, 1] to guard against floating-point rounding
    similarity = max(-1.0, min(1.0, similarity))

    return similarity >= threshold, similarity


# ---------------------------------------------------------------------------
# Multi-sensor detection cycle
# ---------------------------------------------------------------------------


def run_detection_cycle(
    hsi_bands: int = 128,
    nir_points: int = 1024,
    gas_voltages: Optional[List[float]] = None,
    spectral_library: Optional[Dict[str, np.ndarray]] = None,
    rng: Optional[np.random.Generator] = None,
    verbose: bool = True,
) -> Dict:
    """
    Execute one full multi-sensor detection cycle.

    1. Acquire (or simulate) raw signals from all three sensor types.
    2. Process each signal through the appropriate sensor class.
    3. Match the hyperspectral pixel signature against a spectral library.
    4. Return a structured results dict.

    Replace the simulated data sources with hardware reads for deployment:
        - HSI:  camera SDK datacube
        - NIR:  serial/USB spectrometer
        - Gas:  I2C / ADC voltage readings from Nano 33 BLE or ESP8266

    Args:
        hsi_bands: Number of spectral bands for the simulated datacube.
        nir_points: Number of data points from the NIR spectrometer.
        gas_voltages: Voltage readings from the gas sensor array.
                      Defaults to a four-channel simulated reading.
        spectral_library: Dict mapping chemical name → reference spectrum.
                          Built-in mock library used if None.
        rng: Optional NumPy Generator for reproducible simulation.
        verbose: If True, print a summary to stdout.

    Returns:
        Dict with keys:
            hsi_signature, nir_fingerprint, nir_peaks,
            gas_concentrations_ppm, library_matches, gas_readings
    """
    if rng is None:
        rng = np.random.default_rng()

    if gas_voltages is None:
        gas_voltages = [0.1, 0.45, 0.22, 0.8]

    if spectral_library is None:
        spectral_library = {
            "Methane": rng.random(hsi_bands),
            "Water Vapor": rng.random(hsi_bands),
            "Ammonia (urine)": rng.random(hsi_bands),
        }

    # ── 1. Acquire signals ──────────────────────────────────────────────
    hsi_data = simulate_hsi_sensor(bands=hsi_bands, rng=rng)
    nir_data = rng.random(nir_points)

    # ── 2. Initialise sensors ───────────────────────────────────────────
    camera = HyperspectralSensor(
        "Visual HSI",
        roi=(20, 40, 20, 40),
        normalise=True,
    )
    spectrometer = SpectroscopicSensor("NIR Scanner")
    gas_box = GasSensorArray(
        "Pollution / Odour Array",
        channel_names=["NH3", "H2S", "CH4", "VOC"],
    )

    # ── 3. Process signals ──────────────────────────────────────────────
    hsi_signature = camera.process_signal(hsi_data)
    nir_fingerprint = spectrometer.process_signal(nir_data)
    nir_peaks = spectrometer.find_peaks(nir_fingerprint)
    gas_ppm = gas_box.process_signal(gas_voltages)
    gas_readings = gas_box.get_readings(gas_voltages)

    # ── 4. Pixel-level library matching (SAM) ──────────────────────────
    pixel_signal = camera.extract_pixel_signature(hsi_data, row=32, col=32)
    library_matches: Dict[str, Tuple[bool, float]] = {}
    for chem_name, ref_spectrum in spectral_library.items():
        # Normalise the library entry with the same pipeline
        ref_norm = gaussian_filter1d(ref_spectrum.astype(float), sigma=1.0)
        ref_norm = _zscore(ref_norm)
        is_match, similarity = match_spectral_signature(
            pixel_signal, ref_norm, threshold=0.90
        )
        library_matches[chem_name] = (is_match, round(similarity, 4))

    # ── 5. Report ───────────────────────────────────────────────────────
    if verbose:
        print(f"\n{'─'*60}")
        print(f"  Multi-Sensor Chemical Analysis — Detection Cycle")
        print(f"{'─'*60}")
        print(f"  [HSI]  Visual signature shape : {hsi_signature.shape}")
        print(f"  [NIR]  Molecular fingerprint peak : band {nir_peaks[0] if len(nir_peaks) else 'none'}")
        print(f"  [GAS]  Concentrations (ppm)   : {gas_ppm.round(2)}")
        for name, (matched, sim) in library_matches.items():
            flag = "✓ MATCH" if matched else "✗ no match"
            print(f"  [SAM]  {name:<22} similarity={sim:.4f}  {flag}")
        print(f"{'─'*60}\n")

    return {
        "hsi_signature": hsi_signature,
        "nir_fingerprint": nir_fingerprint,
        "nir_peaks": nir_peaks,
        "gas_concentrations_ppm": gas_ppm,
        "gas_readings": gas_readings,
        "library_matches": library_matches,
    }


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _zscore(arr: np.ndarray) -> np.ndarray:
    """Z-score normalise: zero mean, unit variance.  Safe if std ≈ 0."""
    std = arr.std()
    if std < 1e-9:
        return arr - arr.mean()
    return (arr - arr.mean()) / std


def _sam_similarity_batch(
    spectra: np.ndarray, reference: np.ndarray
) -> np.ndarray:
    """
    Vectorised SAM cosine similarity of many spectra against one reference.

    Args:
        spectra: 2-D array (N, bands).
        reference: 1-D array (bands,).

    Returns:
        1-D array of similarity scores (N,) in [-1, 1].
    """
    ref_norm = reference / (np.linalg.norm(reference) + 1e-9)
    norms = np.linalg.norm(spectra, axis=1, keepdims=True) + 1e-9
    unit_spectra = spectra / norms
    return unit_spectra @ ref_norm


# ---------------------------------------------------------------------------
# Script entry point (matches the example in the requirement)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    run_detection_cycle()
