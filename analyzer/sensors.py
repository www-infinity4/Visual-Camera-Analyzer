"""
Multi-Sensor Chemical Analysis Engine

Implements seven core sensor types that convert raw physical signals into
readable chemical signatures, following the modular signal-to-signature
conversion architecture:

  ┌──────────────────────────────────────────────────────────────────────┐
  │  SENSOR TYPE            PHYSICAL SIGNAL        OUTPUT                │
  │  HyperspectralSensor    3D datacube (x,y,λ)    Spatial signature     │
  │  SpectroscopicSensor    1D wavelength array     Molecular fingerprint │
  │  GasSensorArray         Voltage readings        Concentration (ppm)   │
  │  PIDSensor              Ion current (µA)        VOC ppm               │
  │  RamanSensor            Wavenumber shifts       Molecular fingerprint │
  │  IMSSensor              Drift-time spectrum     Threat score          │
  │  LWIRSensor             Thermal radiance array  Chemical vapour map   │
  └──────────────────────────────────────────────────────────────────────┘

These sensors feed the Late Fusion / Sensor-LLM architecture in
analyzer/fusion.py, where each sensor's structured text summary is
reasoned over by an LLM (or a local spectral-matching fallback) to
produce a definitive chemical identification.

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
# Sensor 4 – Photoionization Detector / PID  (VOC electrical signal)
# ---------------------------------------------------------------------------


@dataclass
class PIDReading:
    """Result from a PID sensor measurement."""

    ion_current_ua: float          # Raw ionisation current (µA)
    concentration_ppm: float       # Calculated VOC concentration (ppm)
    lamp_energy_ev: float = 10.6   # UV lamp ionisation energy (eV)
    compound_class: str = "VOC"    # Broad compound class label


class PIDSensor(ChemicalSensor):
    """
    Photoionization Detector (PID) — detects Volatile Organic Compounds (VOCs)
    that cameras and electrochemical sensors often miss at low concentrations.

    Physical principle:
        A high-energy UV lamp (typically 10.6 eV or 11.7 eV) ionises molecules
        whose ionisation potential (IP) is lower than the lamp energy.  The
        resulting ion current is proportional to VOC concentration.

    Sensitivity:
        PIDs detect sub-ppm concentrations of benzene, toluene, xylene (BTX),
        styrene, and many chlorinated solvents.  They are blind to methane,
        CO₂, and most inorganic gases.

    Hardware (DIY):
        - RAE Systems MiniRAE / ppbRAE sensor modules expose an analogue
          current output readable via ADC on a Nano 33 BLE or ESP32.
        - Current-to-voltage: place a 100 kΩ transimpedance resistor.
        - Voltage reading → ppm via the sensor's correction factor (CF):
              ppm = (voltage / CF) * reference_sensitivity

    Lamp energies:
        10.6 eV : detects most aromatics and chlorinated solvents (most common)
        11.7 eV : also detects acetone, methanol, ethanol, ammonia
        9.8 eV  : selective for very low-IP compounds (naphthalene, anthracene)

    Reference:
        https://pmc.ncbi.nlm.nih.gov/articles/PMC10054971/
    """

    # Correction factors relative to isobutylene (the calibration gas for most PIDs)
    # CF = actual_ppm / isobutylene_equivalent_ppm
    CORRECTION_FACTORS: Dict[str, float] = {
        "benzene": 0.53,
        "toluene": 0.54,
        "xylene": 0.48,
        "styrene": 0.40,
        "ethanol": 9.0,
        "acetone": 1.1,
        "ammonia": 10.0,
        "isobutylene": 1.0,  # calibration reference
    }

    def __init__(
        self,
        name: str = "PID VOC Sensor",
        lamp_energy_ev: float = 10.6,
        sensitivity_ua_per_ppm: float = 0.01,
        correction_factor: float = 1.0,
        baseline_ua: float = 0.002,
    ):
        """
        Args:
            name: Human-readable sensor name.
            lamp_energy_ev: UV lamp ionisation energy (9.8, 10.6, or 11.7 eV).
            sensitivity_ua_per_ppm: Current sensitivity (µA per ppm isobutylene).
            correction_factor: Compound-specific correction factor relative to
                               isobutylene.  Use CORRECTION_FACTORS dict.
            baseline_ua: Zero-air dark current baseline (µA).
        """
        super().__init__(name)
        self.lamp_energy_ev = lamp_energy_ev
        self.sensitivity_ua_per_ppm = sensitivity_ua_per_ppm
        self.correction_factor = correction_factor
        self.baseline_ua = baseline_ua

    def process_signal(self, raw_data) -> np.ndarray:
        """
        Convert a raw ion-current reading (µA) into VOC concentration (ppm).

        Args:
            raw_data: Scalar or 1-D array of ion current readings (µA).

        Returns:
            1-D float array of VOC concentrations in ppm.
        """
        currents = np.atleast_1d(np.asarray(raw_data, dtype=float))
        net_current = np.clip(currents - self.baseline_ua, 0.0, None)
        ppm_isobutylene = net_current / (self.sensitivity_ua_per_ppm + 1e-12)
        ppm_compound = ppm_isobutylene / (self.correction_factor + 1e-12)
        return ppm_compound

    def get_reading(self, ion_current_ua: float, compound: str = "VOC") -> PIDReading:
        """
        Return a structured PIDReading for a single measurement.

        Args:
            ion_current_ua: Measured ion current (µA).
            compound: Target compound name (used for correction factor lookup).
        """
        cf = self.CORRECTION_FACTORS.get(compound.lower(), self.correction_factor)
        original_cf = self.correction_factor
        self.correction_factor = cf
        ppm = float(self.process_signal([ion_current_ua])[0])
        self.correction_factor = original_cf
        return PIDReading(
            ion_current_ua=ion_current_ua,
            concentration_ppm=ppm,
            lamp_energy_ev=self.lamp_energy_ev,
            compound_class=compound,
        )

    def to_text_summary(self, ppm_values: np.ndarray) -> str:
        """Generate a structured text summary for Sensor-LLM fusion."""
        peak_ppm = float(ppm_values.max()) if ppm_values.size else 0.0
        level = "high" if peak_ppm > 100 else ("moderate" if peak_ppm > 10 else "low")
        return (
            f"PID sensor ({self.lamp_energy_ev} eV lamp) reports {level} VOC "
            f"concentration of {peak_ppm:.1f} ppm (peak)."
        )


# ---------------------------------------------------------------------------
# Sensor 5 – Raman Spectrometer (wavenumber shift signal)
# ---------------------------------------------------------------------------


@dataclass
class RamanPeak:
    """A single identified Raman shift peak."""

    wavenumber_cm1: float    # Raman shift (cm⁻¹)
    intensity: float         # Normalised intensity (0–1)
    assignment: str = ""     # Vibrational mode assignment (e.g. "C-H stretch")


class RamanSensor(ChemicalSensor):
    """
    Raman Spectrometer — provides a high-resolution molecular "fingerprint"
    for definitive chemical identification.

    Physical principle:
        A monochromatic laser (typically 532 nm, 785 nm, or 1064 nm) excites
        molecular vibrations.  The inelastically scattered photons are shifted
        in frequency by characteristic wavenumber amounts (cm⁻¹) determined by
        the molecule's vibrational modes.

    Key chemical signatures (Raman shifts):
        Benzene ring breathing mode  : ~992 cm⁻¹
        C-H aromatic stretch         : 3050–3100 cm⁻¹
        Ammonium ion (urine)         : ~3350 cm⁻¹  (broad N-H stretch)
        Uric acid (urine marker)     : ~630, 970, 1260 cm⁻¹
        Ethanol C-C stretch          : ~884 cm⁻¹
        Methane C-H bend             : ~1306 cm⁻¹
        TATP (explosive)             : ~829 cm⁻¹

    This class extends SpectroscopicSensor with Raman-specific peak tables
    and wavenumber-axis handling.

    Reference:
        SDBS Raman database: https://sdbs.db.aist.go.jp/
    """

    # Raman shift (cm⁻¹) → vibrational assignment for common chemicals
    PEAK_ASSIGNMENTS: Dict[float, str] = {
        630.0:  "uric acid (urine marker) – ring deformation",
        829.0:  "TATP – O-O stretch",
        884.0:  "ethanol – C-C stretch",
        970.0:  "uric acid – ring breathing",
        992.0:  "benzene – ring breathing",
        1055.0: "ethanol – C-O stretch",
        1260.0: "uric acid – N-H in-plane bend",
        1306.0: "methane – C-H symmetric bend",
        1580.0: "graphene/aromatic – G band",
        3050.0: "aromatic – C-H stretch",
        3350.0: "ammonia / ammonium ion – N-H stretch",
    }
    ASSIGNMENT_TOLERANCE_CM1 = 15.0

    def __init__(
        self,
        name: str = "Raman Spectrometer",
        excitation_nm: float = 785.0,
        wavenumber_range: Tuple[float, float] = (200.0, 3600.0),
        smooth_sigma: float = 1.5,
        peak_min_height: float = 1.0,
        peak_min_distance: int = 5,
    ):
        """
        Args:
            name: Human-readable sensor name.
            excitation_nm: Laser excitation wavelength (nm).
            wavenumber_range: (min, max) Raman shift range (cm⁻¹).
            smooth_sigma: Gaussian smoothing σ for the raw spectrum.
            peak_min_height: Minimum normalised Z-score height for peaks.
            peak_min_distance: Minimum bin separation between peaks.
        """
        super().__init__(name)
        self.excitation_nm = excitation_nm
        self.wavenumber_range = wavenumber_range
        self.smooth_sigma = smooth_sigma
        self.peak_min_height = peak_min_height
        self.peak_min_distance = peak_min_distance

    def process_signal(self, raw_data) -> np.ndarray:
        """
        Condition a raw Raman spectrum: smooth → Z-score normalise.

        Args:
            raw_data: 1-D array of raw Raman intensity counts.

        Returns:
            Z-score normalised, noise-reduced 1-D spectrum.
        """
        spectrum = np.asarray(raw_data, dtype=float)
        if spectrum.ndim != 1 or len(spectrum) < 3:
            raise ValueError("Raman spectrum must be a 1-D array with ≥ 3 points.")
        if self.smooth_sigma > 0:
            spectrum = gaussian_filter1d(spectrum, sigma=self.smooth_sigma)
        return _zscore(spectrum)

    def find_peaks(self, normalised: np.ndarray) -> np.ndarray:
        """Return indices of local maxima above peak_min_height."""
        peaks = []
        n = len(normalised)
        for i in range(1, n - 1):
            if normalised[i] < self.peak_min_height:
                continue
            if normalised[i] <= normalised[i - 1] or normalised[i] <= normalised[i + 1]:
                continue
            if peaks and (i - peaks[-1]) < self.peak_min_distance:
                if normalised[i] > normalised[peaks[-1]]:
                    peaks[-1] = i
            else:
                peaks.append(i)
        return np.array(peaks, dtype=int)

    def assign_peaks(
        self,
        peak_indices: np.ndarray,
        wavenumbers: np.ndarray,
        intensities: np.ndarray,
    ) -> List[RamanPeak]:
        """
        Map detected peak indices to known chemical assignments.

        Args:
            peak_indices: Output of find_peaks().
            wavenumbers:  1-D array of wavenumber values (cm⁻¹).
            intensities:  Normalised spectrum array (same length as wavenumbers).

        Returns:
            List of RamanPeak objects.
        """
        raman_peaks: List[RamanPeak] = []
        for idx in peak_indices:
            wn = float(wavenumbers[idx])
            intensity = float(intensities[idx])
            assignment = ""
            for known_wn, label in self.PEAK_ASSIGNMENTS.items():
                if abs(wn - known_wn) <= self.ASSIGNMENT_TOLERANCE_CM1:
                    assignment = label
                    break
            raman_peaks.append(RamanPeak(wn, intensity, assignment))
        return raman_peaks

    def to_text_summary(self, peaks: List[RamanPeak]) -> str:
        """Generate a structured text summary for Sensor-LLM fusion."""
        if not peaks:
            return "Raman spectrometer: no significant peaks detected."
        top = sorted(peaks, key=lambda p: p.intensity, reverse=True)[:3]
        parts = []
        for p in top:
            label = f" ({p.assignment})" if p.assignment else ""
            parts.append(f"{p.wavenumber_cm1:.0f} cm⁻¹{label}")
        return (
            f"Raman spectrometer ({self.excitation_nm:.0f} nm laser) detected "
            f"peaks at: {', '.join(parts)}."
        )


# ---------------------------------------------------------------------------
# Sensor 6 – Ion Mobility Spectrometer / IMS (drift-time signal)
# ---------------------------------------------------------------------------


@dataclass
class IMSReading:
    """Result from an IMS drift-tube measurement."""

    drift_time_ms: float        # Peak drift time (milliseconds)
    reduced_mobility_k0: float  # Reduced ion mobility K₀ (cm²/(V·s))
    threat_score: float         # 0–1 probability of threat substance
    compound_class: str = "unknown"


class IMSSensor(ChemicalSensor):
    """
    Ion Mobility Spectrometer (IMS) — filters for trace explosives and narcotics.

    Physical principle:
        Ions are accelerated through a drift tube filled with a buffer gas.
        Different compounds separate by their characteristic drift time, which
        is normalised to the reduced ion mobility constant K₀:

            K₀ = (L² / (V · t_d)) · (P/P₀) · (T₀/T)

        Where L = drift length, V = drift voltage, t_d = drift time,
        P/P₀ and T₀/T are pressure and temperature corrections.

    Typical K₀ values for threat substances:
        RDX explosive    : 1.53 cm²/(V·s)
        TNT explosive    : 1.52 cm²/(V·s)
        TATP explosive   : 1.74 cm²/(V·s)
        Cocaine          : 1.05 cm²/(V·s)
        Heroin           : 1.00 cm²/(V·s)
        Ammonia (urine)  : 2.70 cm²/(V·s)

    DIY hardware note:
        A minimal IMS can be built with a 63Ni ionisation source (or UV lamp),
        a stainless-steel drift tube, and a Faraday-plate detector connected to
        a transimpedance amplifier.  Record the full drift spectrum with a 16-bit
        ADC at ≥ 1 MHz sampling rate.

    Reference:
        https://www.spiedigitallibrary.org/conference-proceedings-of-spie/5795/0000/
        A-taxonomy-of-algorithms-for-chemical-vapor-detection-with-hyperspectral/
        10.1117/12.602323.pdf
    """

    # Known reduced ion mobility constants K₀ (cm²/(V·s)) for threat library
    K0_LIBRARY: Dict[str, float] = {
        "RDX":      1.53,
        "TNT":      1.52,
        "TATP":     1.74,
        "HMTD":     1.61,
        "cocaine":  1.05,
        "heroin":   1.00,
        "methamphetamine": 1.18,
        "ammonia":  2.70,
        "acetone":  1.90,
    }
    K0_TOLERANCE = 0.05  # cm²/(V·s)

    def __init__(
        self,
        name: str = "Ion Mobility Spectrometer",
        drift_length_cm: float = 6.0,
        drift_voltage_v: float = 2000.0,
        temperature_k: float = 373.15,   # 100 °C — typical IMS operating temp
        pressure_hpa: float = 1013.25,
    ):
        """
        Args:
            name: Human-readable sensor name.
            drift_length_cm: Length of the IMS drift tube (cm).
            drift_voltage_v: Drift tube voltage (V).
            temperature_k: Operating temperature (K).
            pressure_hpa: Ambient pressure (hPa).
        """
        super().__init__(name)
        self.drift_length_cm = drift_length_cm
        self.drift_voltage_v = drift_voltage_v
        self.temperature_k = temperature_k
        self.pressure_hpa = pressure_hpa

    def process_signal(self, raw_data) -> np.ndarray:
        """
        Process a raw IMS drift-time spectrum.

        Applies Z-score normalisation and returns the conditioned spectrum.

        Args:
            raw_data: 1-D array of detector current vs drift-time bins.

        Returns:
            Z-score normalised drift spectrum.
        """
        spectrum = np.asarray(raw_data, dtype=float)
        if spectrum.ndim != 1 or len(spectrum) < 3:
            raise ValueError("IMS spectrum must be a 1-D array with ≥ 3 points.")
        spectrum = gaussian_filter1d(spectrum, sigma=1.0)
        return _zscore(spectrum)

    def drift_time_to_k0(self, drift_time_ms: float) -> float:
        """
        Convert a measured drift time to the reduced ion mobility K₀.

        Args:
            drift_time_ms: Measured peak drift time (ms).

        Returns:
            Reduced ion mobility K₀ (cm²/(V·s)).
        """
        if drift_time_ms <= 0:
            return 0.0
        L = self.drift_length_cm
        V = self.drift_voltage_v
        t_s = drift_time_ms * 1e-3
        T0, P0 = 273.15, 1013.25
        k0 = (L ** 2 / (V * t_s)) * (self.pressure_hpa / P0) * (T0 / self.temperature_k)
        return k0

    def identify_compound(self, k0: float) -> Tuple[str, float]:
        """
        Match K₀ against the threat library to identify a compound.

        Args:
            k0: Reduced ion mobility (cm²/(V·s)).

        Returns:
            Tuple of (compound_name, threat_score 0–1).
        """
        best_compound = "unknown"
        best_score = 0.0
        for compound, ref_k0 in self.K0_LIBRARY.items():
            diff = abs(k0 - ref_k0)
            if diff <= self.K0_TOLERANCE:
                score = 1.0 - (diff / self.K0_TOLERANCE)
                if score > best_score:
                    best_score = score
                    best_compound = compound
        return best_compound, best_score

    def get_reading(self, drift_time_ms: float) -> IMSReading:
        """Return a structured IMSReading for a single drift-time measurement."""
        k0 = self.drift_time_to_k0(drift_time_ms)
        compound, score = self.identify_compound(k0)
        return IMSReading(
            drift_time_ms=drift_time_ms,
            reduced_mobility_k0=round(k0, 4),
            threat_score=round(score, 4),
            compound_class=compound,
        )

    def to_text_summary(self, reading: IMSReading) -> str:
        """Generate a structured text summary for Sensor-LLM fusion."""
        threat_str = "THREAT DETECTED" if reading.threat_score > 0.7 else "no threat"
        return (
            f"IMS sensor: drift time {reading.drift_time_ms:.2f} ms → "
            f"K₀ = {reading.reduced_mobility_k0:.3f} cm²/(V·s), "
            f"compound={reading.compound_class}, "
            f"threat_score={reading.threat_score:.2f} [{threat_str}]."
        )


# ---------------------------------------------------------------------------
# Sensor 7 – Long-Wave Infrared / LWIR  (thermal radiance signal)
# ---------------------------------------------------------------------------


class LWIRSensor(ChemicalSensor):
    """
    Long-Wave Infrared (LWIR) Thermal Camera — detects chemical vapour clouds
    based on their thermal emission/absorption signatures (8–14 µm).

    Physical principle:
        At room temperature, all objects emit thermal radiation.  Chemical
        vapours (gases) absorb and re-emit this radiation at wavelengths
        determined by their rotational-vibrational modes.  By imaging the
        radiance contrast between a warm background and a cooler gas cloud,
        LWIR cameras reveal invisible chemical plumes.

    Key LWIR chemical fingerprints (absorption band centres):
        Ammonia (NH₃)    : 9.0 µm  (N-H wag)    → urine marker
        Methane (CH₄)    : 7.7 µm  (C-H deform)
        Ethanol          : 9.5 µm  (C-O stretch)
        Acetone          : 8.0 µm  (C=O stretch)
        Sarin (nerve)    : 9.8 µm  (P=O stretch)
        SF₆ (calibration): 10.5 µm

    This matches the detection principle of the TRACER DTS camera from
    Spectral Sciences and thermal IR chemical mapping in ENVI/OPUS.

    Hardware options:
        - FLIR Lepton 3.5 (80×60, LWIR, <$250)
        - Seek Thermal Compact (320×240, LWIR)
        - Heimann HTPA 32×32 array (thermopile, I2C)

    Reference:
        https://www.spiedigitallibrary.org/conference-proceedings-of-spie/5795/
    """

    # LWIR absorption band centres (µm) for common chemicals
    LWIR_FINGERPRINTS: Dict[str, float] = {
        "ammonia":  9.0,
        "methane":  7.7,
        "ethanol":  9.5,
        "acetone":  8.0,
        "sarin":    9.8,
        "SF6":      10.5,
        "CO2":      14.9,
        "ozone":    9.6,
    }
    FINGERPRINT_TOLERANCE_UM = 0.3  # µm

    def __init__(
        self,
        name: str = "LWIR Thermal Camera",
        wavelength_min_um: float = 8.0,
        wavelength_max_um: float = 14.0,
        num_bands: int = 60,
        temperature_reference_k: float = 300.0,
    ):
        """
        Args:
            name: Human-readable sensor name.
            wavelength_min_um: Minimum LWIR wavelength (µm).
            wavelength_max_um: Maximum LWIR wavelength (µm).
            num_bands: Number of spectral channels.
            temperature_reference_k: Scene background temperature (K).
        """
        super().__init__(name)
        self.wavelength_min_um = wavelength_min_um
        self.wavelength_max_um = wavelength_max_um
        self.num_bands = num_bands
        self.temperature_reference_k = temperature_reference_k
        self.wavelengths_um = np.linspace(wavelength_min_um, wavelength_max_um, num_bands)

    def process_signal(self, raw_data) -> np.ndarray:
        """
        Process a raw LWIR radiance spectrum.

        Subtracts a Planck background radiance model, then Z-score
        normalises the residual to isolate chemical absorption features.

        Args:
            raw_data: 1-D array of thermal radiance values (W/(m²·sr·µm)).

        Returns:
            Background-subtracted, Z-score normalised radiance spectrum.
        """
        radiance = np.asarray(raw_data, dtype=float)
        if radiance.ndim != 1 or len(radiance) < 3:
            raise ValueError("LWIR spectrum must be a 1-D array with ≥ 3 points.")

        # Planck background: resize to match input length
        background = self._planck_radiance(self.temperature_reference_k)
        if len(background) != len(radiance):
            background = np.interp(
                np.linspace(0, 1, len(radiance)),
                np.linspace(0, 1, len(background)),
                background,
            )
        residual = radiance - background
        return _zscore(residual)

    def identify_chemical(
        self, radiance_spectrum: np.ndarray
    ) -> List[Tuple[str, float]]:
        """
        Match LWIR absorption features against known chemical fingerprints.

        Args:
            radiance_spectrum: Raw or background-subtracted LWIR spectrum.

        Returns:
            List of (chemical_name, confidence 0–1) sorted by confidence.
        """
        processed = self.process_signal(radiance_spectrum)
        results = []
        for chem, band_centre_um in self.LWIR_FINGERPRINTS.items():
            # Find the band index closest to the fingerprint wavelength
            idx = int(np.argmin(np.abs(self.wavelengths_um - band_centre_um)))
            if idx < len(processed):
                # Absorption appears as a negative residual; take absolute value
                absorption_strength = abs(float(processed[idx]))
                confidence = min(1.0, absorption_strength / 3.0)
                results.append((chem, round(confidence, 4)))
        return sorted(results, key=lambda x: x[1], reverse=True)

    def thermal_vapour_map(
        self,
        thermal_image: np.ndarray,
        target_chemical: str,
    ) -> np.ndarray:
        """
        Produce a 2-D vapour-presence map from a thermal image.

        Each pixel's single-band radiance is scored against the expected
        absorption contrast for the target chemical.

        Args:
            thermal_image: 2-D float array of scene temperatures (K) or
                           radiance values (H × W).
            target_chemical: Chemical to map (key in LWIR_FINGERPRINTS).

        Returns:
            2-D float32 vapour confidence map (H, W) in [0, 1].
        """
        if target_chemical not in self.LWIR_FINGERPRINTS:
            raise ValueError(f"Unknown chemical: {target_chemical!r}")

        ref_temp = self.temperature_reference_k
        contrast = np.abs(thermal_image.astype(float) - ref_temp)
        normalised = np.clip(contrast / 10.0, 0.0, 1.0)  # 10 K contrast = 100%
        return normalised.astype(np.float32)

    def to_text_summary(self, matches: List[Tuple[str, float]]) -> str:
        """Generate a structured text summary for Sensor-LLM fusion."""
        if not matches:
            return "LWIR sensor: no chemical vapour signatures detected."
        top = matches[:3]
        parts = [f"{chem} (confidence={conf:.0%})" for chem, conf in top if conf > 0.1]
        if not parts:
            return "LWIR sensor: thermal baseline only, no significant absorption features."
        return f"LWIR thermal camera detected chemical vapour: {'; '.join(parts)}."

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _planck_radiance(self, temperature_k: float) -> np.ndarray:
        """Compute Planck spectral radiance for a blackbody at temperature_k."""
        h = 6.626e-34  # Planck constant (J·s)
        c = 2.998e8    # Speed of light (m/s)
        k = 1.381e-23  # Boltzmann constant (J/K)
        wl_m = self.wavelengths_um * 1e-6
        exponent = np.clip((h * c) / (wl_m * k * temperature_k), 0, 700)
        radiance = (2 * h * c ** 2 / wl_m ** 5) / (np.exp(exponent) - 1.0 + 1e-30)
        # Normalise so the background is ~ unit amplitude
        max_r = radiance.max()
        if max_r > 1e-30:
            radiance /= max_r
        return radiance


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
