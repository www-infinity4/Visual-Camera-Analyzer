"""
Virtual Digital Twin Sensors
════════════════════════════
Pure-software simulations of physical chemical sensors.  Each class is a
"Digital Twin" — a mathematical model that generates the same kind of signal
a real hardware sensor would produce, without requiring any physical device.

This enables:
  • Rapid prototyping before hardware is sourced
  • Training and testing the Sensor-LLM pipeline on reproducible data
  • Satellite / remote-sensing workflows where only spectral data is available
  • "Digital Mine" surveys for rare earth elements via synthetic spectral matching

Sensor Digital Twins
────────────────────
  VirtualAmmoniaGasSensor  – MQ-137 / electrochemical EC sensor simulation
  VirtualSpectralSensor    – Hyperspectral / SWIR reflectance simulation
  ChemicalSignalGenerator  – General Gaussian-peak signal engine (NH₃ series)
  MetalOxideSensor         – MOS resistance-change simulation (ppb range)

Integration helper
──────────────────
  synthesize_data_for_gemma()  – Formats all sensor readings as a structured
                                  text prompt ready for Gemma 2 / any LLM.

References
──────────
Virtual / synthetic spectral libraries:
  USGS spectral library  : https://www.usgs.gov/labs/spectroscopy-lab
  ECOSTRESS             : https://ecostress.jpl.nasa.gov/
  DARPA REE AI project  : https://www.defenseone.com/technology/2024/07/
                           darpa-wants-use-ai-find-new-rare-minerals/397830/
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.ndimage import gaussian_filter1d


# ---------------------------------------------------------------------------
# 1. Virtual Ammonia Gas Sensor  (Digital Twin of MQ-137 / EC sensor)
# ---------------------------------------------------------------------------


class VirtualAmmoniaGasSensor:
    """
    Simulates an MQ-137 metal-oxide or electrochemical ammonia sensor.

    Response model
    ──────────────
    The real MQ-137 produces a resistance change that follows a log-linear
    power-law curve.  In this Digital Twin we model the sensed value as:

        sensed_ppm = (true_ppm × sensitivity) + baseline + gaussian_noise

    The Gaussian noise term mimics real sensor fluctuations caused by
    temperature drift, humidity cross-sensitivity, and ADC quantisation.

    Safety reference thresholds (OSHA / ACGIH):
        8-hr TWA (PEL)  :  25 ppm   – permissible exposure limit
        STEL            :  35 ppm   – short-term 15-min limit
        IDLH            : 300 ppm   – immediately dangerous to life/health
    """

    # OSHA / ACGIH ammonia safety thresholds (ppm)
    THRESHOLD_TWA_PPM: float = 25.0
    THRESHOLD_STEL_PPM: float = 35.0
    THRESHOLD_IDLH_PPM: float = 300.0

    def __init__(
        self,
        baseline_ppm: float = 0.5,
        sensitivity: float = 1.05,
        noise_std: float = 0.05,
        rng: Optional[np.random.Generator] = None,
    ):
        """
        Args:
            baseline_ppm:  Zero-air background reading (ppm).
            sensitivity:   Gain factor (>1 means slight over-read, as real sensors do).
            noise_std:     Gaussian noise standard deviation (ppm).
            rng:           NumPy random Generator for reproducible simulations.
        """
        self.baseline_ppm = baseline_ppm
        self.sensitivity = sensitivity
        self.noise_std = noise_std
        self._rng = rng or np.random.default_rng()

    def generate_signal(self, true_concentration: float) -> float:
        """
        Generate a simulated sensor reading for a known concentration.

        Args:
            true_concentration: Actual NH₃ concentration (ppm).

        Returns:
            Simulated sensor reading (ppm), always ≥ 0.
        """
        noise = self._rng.normal(0.0, self.noise_std)
        sensed = (true_concentration * self.sensitivity) + self.baseline_ppm + noise
        return round(max(0.0, sensed), 3)

    def hazard_level(self, ppm: float) -> str:
        """Return a safety classification string for a given ppm reading."""
        if ppm >= self.THRESHOLD_IDLH_PPM:
            return "IDLH – EVACUATE IMMEDIATELY"
        if ppm >= self.THRESHOLD_STEL_PPM:
            return "STEL exceeded – short-term overexposure"
        if ppm >= self.THRESHOLD_TWA_PPM:
            return "TWA exceeded – reduce exposure"
        return "safe"

    def to_text_summary(self, ppm: float) -> str:
        """Structured text for Sensor-LLM fusion."""
        hazard = self.hazard_level(ppm)
        return (
            f"Virtual NH₃ gas sensor reading: {ppm:.3f} ppm "
            f"[{hazard}]."
        )


# ---------------------------------------------------------------------------
# 2. Virtual Spectral Sensor  (Digital Twin of SWIR hyperspectral camera)
# ---------------------------------------------------------------------------


class VirtualSpectralSensor:
    """
    Simulates the reflectance signature of a spectrometer or hyperspectral
    camera covering 400–2500 nm (Visible to Short-Wave Infrared / SWIR).

    Key absorption signatures modelled
    ────────────────────────────────────
    Ammonia (NH₃)
      • 1510 nm  — N-H overtone stretch (strong absorption dip, SWIR)
      • 1960 nm  — N-H combination band

    Ammonium Nitrate (NH₄NO₃)
      • 800 nm   — NO₃⁻ electronic transition / NIR fingerprint
      • 1040 nm  — N-H stretch overtone
      • 2050 nm  — N-O combination band

    Urea (CH₄N₂O)
      • 1500 nm  — N-H overtone
      • 2170 nm  — C=O combination band

    Optical brightener (false positive)
      • 430 nm   — stilbene UV-Vis emission (blue-white fluorescence)

    References:
      USGS spectral library / ECOSTRESS panel measurements.
    """

    # Substance library: name → list of (centre_nm, depth, half_width_nm)
    SUBSTANCE_LIBRARY: Dict[str, List[Tuple[float, float, float]]] = {
        "ammonia": [
            (1510.0, 0.6, 15.0),
            (1960.0, 0.3, 20.0),
        ],
        "ammonium_nitrate": [
            (800.0,  0.4, 10.0),
            (1040.0, 0.3, 12.0),
            (2050.0, 0.35, 18.0),
        ],
        "urea": [
            (1500.0, 0.45, 14.0),
            (2170.0, 0.25, 22.0),
        ],
        "optical_brightener": [
            (430.0,  0.8, 20.0),
        ],
        "methane": [
            (1670.0, 0.7, 8.0),
            (2310.0, 0.9, 12.0),
        ],
        "ethanol": [
            (1450.0, 0.5, 10.0),
            (2100.0, 0.4, 15.0),
        ],
    }

    def __init__(
        self,
        resolution: int = 128,
        wavelength_min_nm: float = 400.0,
        wavelength_max_nm: float = 2500.0,
        noise_std: float = 0.01,
        rng: Optional[np.random.Generator] = None,
    ):
        """
        Args:
            resolution:         Number of spectral channels.
            wavelength_min_nm:  Start of spectral range (nm).
            wavelength_max_nm:  End of spectral range (nm).
            noise_std:          Gaussian photon-noise std (reflectance units).
            rng:                NumPy Generator for reproducibility.
        """
        self.resolution = resolution
        self.wavelength_min_nm = wavelength_min_nm
        self.wavelength_max_nm = wavelength_max_nm
        self.noise_std = noise_std
        self._rng = rng or np.random.default_rng()
        self.wavelengths = np.linspace(wavelength_min_nm, wavelength_max_nm, resolution)

    def get_signature(
        self,
        substance: str,
        concentration_scale: float = 1.0,
    ) -> List[Tuple[float, float]]:
        """
        Generate a reflectance spectrum for the specified substance.

        Args:
            substance:           Chemical name (key in SUBSTANCE_LIBRARY).
                                 Case-insensitive.
            concentration_scale: Scales absorption depth (0 = no absorption,
                                  1 = nominal, >1 = higher concentration).

        Returns:
            List of (wavelength_nm, reflectance) tuples.  Reflectance is
            1.0 for a perfect reflector and <1.0 where the chemical absorbs.
        """
        signal = np.ones(self.resolution, dtype=float)

        # Add photon shot noise
        if self.noise_std > 0:
            signal += self._rng.normal(0.0, self.noise_std, self.resolution)

        key = substance.lower().replace(" ", "_")
        if key in self.SUBSTANCE_LIBRARY:
            for centre_nm, depth, half_width_nm in self.SUBSTANCE_LIBRARY[key]:
                dip = (depth * concentration_scale) * np.exp(
                    -((self.wavelengths - centre_nm) ** 2) / (2 * half_width_nm ** 2)
                )
                signal -= dip

        signal = np.clip(signal, 0.0, 1.0)
        return list(zip(self.wavelengths.tolist(), signal.tolist()))

    def get_array(
        self, substance: str, concentration_scale: float = 1.0
    ) -> np.ndarray:
        """Return the reflectance spectrum as a plain NumPy array."""
        pairs = self.get_signature(substance, concentration_scale)
        return np.array([r for _, r in pairs], dtype=float)

    def dominant_dip_nm(self, signature: List[Tuple[float, float]]) -> float:
        """Return the wavelength (nm) of the deepest absorption dip."""
        wl = np.array([w for w, _ in signature])
        refl = np.array([r for _, r in signature])
        return float(wl[np.argmin(refl)])

    def to_text_summary(self, substance: str, concentration_scale: float = 1.0) -> str:
        """Structured text for Sensor-LLM fusion."""
        sig = self.get_signature(substance, concentration_scale)
        dip_nm = self.dominant_dip_nm(sig)
        return (
            f"Virtual spectral sensor: reflectance signature for {substance!r}; "
            f"dominant absorption dip at {dip_nm:.1f} nm "
            f"(concentration_scale={concentration_scale:.2f})."
        )


# ---------------------------------------------------------------------------
# 3. Chemical Signal Generator  (Gaussian-peak signal engine)
# ---------------------------------------------------------------------------


# Built-in signal library: chemical name → {peak index, width, modifier}
# The "peak index" is a position in a 1024-point signal array.
# This is a frequency / wavenumber / field-sweep index — not a wavelength.
CHEMICAL_SIGNAL_LIBRARY: Dict[str, Dict] = {
    # Ammonia series
    "ammonia":          {"peak": 320, "width": 15,  "modifier": 1.0},
    "ammonium_nitrate": {"peak": 320, "width": 20,  "modifier": 1.4},   # wider, higher
    "ammonium_chloride":{"peak": 315, "width": 18,  "modifier": 1.1},
    "urea":             {"peak": 345, "width": 12,  "modifier": 0.9},
    # Common solvents
    "ethanol":          {"peak": 410, "width": 14,  "modifier": 0.8},
    "acetone":          {"peak": 380, "width": 11,  "modifier": 1.05},
    "benzene":          {"peak": 460, "width": 8,   "modifier": 1.2},
    # Explosives
    "rdx":              {"peak": 520, "width": 6,   "modifier": 0.95},
    "tatp":             {"peak": 490, "width": 7,   "modifier": 1.0},
    # Rare earths (dominant spectral index)
    "neodymium":        {"peak": 200, "width": 5,   "modifier": 0.7},
    "samarium":         {"peak": 280, "width": 7,   "modifier": 0.6},
    "dysprosium":       {"peak": 340, "width": 4,   "modifier": 0.55},
}


def generate_chemical_signal(
    chemical_name: str,
    intensity: float = 1.0,
    signal_length: int = 1024,
    noise_std: float = 0.02,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """
    Create a synthetic 1-D signal array — the "digital fingerprint" — for a
    chemical compound.

    This is the core of the Software-Defined Sensor Suite.  Each chemical has
    a unique Gaussian peak defined by its position (peak index), width, and
    modifier.  The "Fine Adjustment" between related compounds (e.g. NH₃ →
    NH₄NO₃) is encoded as a change in peak width and modifier: the nitrate
    salt produces a wider, higher peak than the gas form.

    Args:
        chemical_name:  Key in CHEMICAL_SIGNAL_LIBRARY.
        intensity:      Overall signal amplitude scaler.
        signal_length:  Number of data points (default 1024).
        noise_std:      Gaussian baseline noise standard deviation.
        rng:            NumPy Generator for reproducible simulations.

    Returns:
        1-D float64 NumPy array of length ``signal_length``.

    Example:
        >>> nh3   = generate_chemical_signal("ammonia")
        >>> nh4no3 = generate_chemical_signal("ammonium_nitrate")
        >>> print(f"NH₃ peak:     {nh3.max():.4f}")
        >>> print(f"NH₄NO₃ peak:  {nh4no3.max():.4f}  (wider & higher)")
    """
    if rng is None:
        rng = np.random.default_rng()

    signal = rng.normal(0.0, noise_std, signal_length)
    key = chemical_name.lower().replace(" ", "_")

    if key in CHEMICAL_SIGNAL_LIBRARY:
        target = CHEMICAL_SIGNAL_LIBRARY[key]
        x = np.arange(signal_length, dtype=float)
        peak = (
            target["modifier"]
            * intensity
            * np.exp(
                -((x - target["peak"]) ** 2) / (2 * target["width"] ** 2)
            )
        )
        signal += peak

    return signal


class ChemicalSignalGenerator:
    """
    Object-oriented wrapper around :func:`generate_chemical_signal`.

    Supports batch generation, SNR calculation, and comparison between
    related chemicals to identify the "fine adjustment" (the shift in
    peak width/height that distinguishes, for example, NH₃ gas from
    the NH₄NO₃ solid/particulate form).
    """

    def __init__(
        self,
        signal_length: int = 1024,
        noise_std: float = 0.02,
        rng: Optional[np.random.Generator] = None,
    ):
        self.signal_length = signal_length
        self.noise_std = noise_std
        self._rng = rng or np.random.default_rng()

    def generate(self, chemical_name: str, intensity: float = 1.0) -> np.ndarray:
        """Generate a signal for one chemical."""
        return generate_chemical_signal(
            chemical_name, intensity, self.signal_length, self.noise_std, self._rng
        )

    def generate_batch(
        self,
        chemicals: List[str],
        intensities: Optional[List[float]] = None,
    ) -> Dict[str, np.ndarray]:
        """Generate signals for multiple chemicals in one call."""
        if intensities is None:
            intensities = [1.0] * len(chemicals)
        return {
            name: self.generate(name, intensity)
            for name, intensity in zip(chemicals, intensities)
        }

    def snr(self, signal: np.ndarray) -> float:
        """
        Signal-to-Noise Ratio: ratio of peak amplitude to baseline RMS.

        SNR > 3 is considered a reliable detection (3-sigma rule).
        """
        peak = float(signal.max())
        # Estimate noise from the lower 25 % of the signal
        noise_rms = float(np.std(signal[signal < np.percentile(signal, 25)]))
        if noise_rms < 1e-12:
            return float("inf")
        return peak / noise_rms

    def compare(
        self, chem_a: str, chem_b: str
    ) -> Dict:
        """
        Compare two chemicals' peak parameters (the "fine adjustment").

        Returns a dict explaining how chem_b differs from chem_a.
        """
        key_a = chem_a.lower().replace(" ", "_")
        key_b = chem_b.lower().replace(" ", "_")
        lib = CHEMICAL_SIGNAL_LIBRARY

        if key_a not in lib or key_b not in lib:
            return {"error": "one or both chemicals not in library"}

        a, b = lib[key_a], lib[key_b]
        return {
            "chemical_a": chem_a,
            "chemical_b": chem_b,
            "peak_shift": b["peak"] - a["peak"],
            "width_change": b["width"] - a["width"],
            "modifier_change": round(b["modifier"] - a["modifier"], 4),
            "interpretation": (
                "wider and higher peak → denser/solid phase"
                if b["width"] > a["width"] and b["modifier"] > a["modifier"]
                else "narrower/lower peak → gaseous / dilute phase"
            ),
        }

    @staticmethod
    def list_chemicals() -> List[str]:
        """Return all chemical names in the built-in library."""
        return list(CHEMICAL_SIGNAL_LIBRARY.keys())

    def to_text_summary(self, chemical_name: str, intensity: float = 1.0) -> str:
        """Structured text for Sensor-LLM fusion."""
        sig = self.generate(chemical_name, intensity)
        peak_val = float(sig.max())
        snr = self.snr(sig)
        key = chemical_name.lower().replace(" ", "_")
        info = CHEMICAL_SIGNAL_LIBRARY.get(key, {})
        peak_idx = info.get("peak", "N/A")
        width = info.get("width", "N/A")
        return (
            f"Chemical signal generator: {chemical_name!r} — "
            f"peak at index {peak_idx}, width={width}, "
            f"amplitude={peak_val:.4f}, SNR={snr:.1f}."
        )


# ---------------------------------------------------------------------------
# 4. Signal Integration Script — the "Brain" for Gemma 2
# ---------------------------------------------------------------------------


def synthesize_data_for_gemma(
    substance_name: str,
    actual_ppm: float,
    spectral_sensor: Optional[VirtualSpectralSensor] = None,
    gas_sensor: Optional[VirtualAmmoniaGasSensor] = None,
    include_signal_fingerprint: bool = True,
    hazard_threshold_ppm: float = 25.0,
) -> str:
    """
    Collect virtual sensor readings and format them as a structured text
    prompt ready to be fed into Gemma 2 (or any LLM) for reasoning.

    This is the "Signal Integration Script" — it bridges the Software-Defined
    Sensor Suite and the language model's reasoning engine.

    Args:
        substance_name:          Target chemical (e.g. "Ammonia").
        actual_ppm:              Simulated true concentration (ppm).
        spectral_sensor:         Optional pre-configured VirtualSpectralSensor.
        gas_sensor:              Optional pre-configured VirtualAmmoniaGasSensor.
        include_signal_fingerprint: Include Gaussian peak parameters.
        hazard_threshold_ppm:   Concentration above which to flag as hazardous.

    Returns:
        Formatted multi-line string suitable as an LLM prompt.

    Example:
        >>> prompt = synthesize_data_for_gemma("Ammonia", 30)
        >>> print(prompt)
    """
    if gas_sensor is None:
        gas_sensor = VirtualAmmoniaGasSensor()
    if spectral_sensor is None:
        spectral_sensor = VirtualSpectralSensor()

    gas_val = gas_sensor.generate_signal(actual_ppm)
    spectral_sig = spectral_sensor.get_signature(substance_name)
    dip_nm = spectral_sensor.dominant_dip_nm(spectral_sig)
    hazard_str = "YES – exceeds safety threshold" if gas_val >= hazard_threshold_ppm else "NO – within safe limits"

    fingerprint_block = ""
    if include_signal_fingerprint:
        key = substance_name.lower().replace(" ", "_")
        info = CHEMICAL_SIGNAL_LIBRARY.get(key, {})
        if info:
            fingerprint_block = (
                f"SIGNAL FINGERPRINT: peak_index={info['peak']}, "
                f"width={info['width']}, modifier={info['modifier']}\n"
            )

    prompt = (
        f"\n{'═'*50}\n"
        f"  SENSOR DATA REPORT\n"
        f"{'═'*50}\n"
        f"  IDENTIFIED SUBSTANCE   : {substance_name}\n"
        f"  GAS SENSOR (ppm)       : {gas_val}\n"
        f"  DOMINANT SPECTRAL DIP  : {dip_nm:.1f} nm\n"
        f"  HAZARDOUS (>{hazard_threshold_ppm} ppm) : {hazard_str}\n"
        f"  {fingerprint_block}"
        f"{'─'*50}\n"
        f"  ANALYSIS REQUEST:\n"
        f"  Compare this data to chemical safety standards for\n"
        f"  {substance_name}. Determine if the substance is:\n"
        f"  (a) Pure {substance_name} gas, (b) a nitrate/salt form,\n"
        f"  or (c) a mixture.  Flag if concentration is hazardous.\n"
        f"{'═'*50}\n"
    )
    return prompt
