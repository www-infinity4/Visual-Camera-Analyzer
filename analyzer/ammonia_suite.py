"""
Ammonia Suite — Three-Layer Virtual Sensor Array
═════════════════════════════════════════════════
A software-defined sensor stack that covers every physical state of
ammonia-based compounds:

  ┌──────────────────────────────────────────────────────────────────────┐
  │  Layer   Sensor Type            Target          Signal               │
  │  ──────  ─────────────────────  ──────────────  ─────────────────── │
  │  1       ElectrochemicalSensor  NH₃ gas         current (µA) → ppm  │
  │  2       MetalOxideSensor       NH₃ gas (ppb)   resistance (Ω)      │
  │  3       InfraredNH3Sensor      NH₃ SWIR 1.5µm  absorbance (AU)     │
  │  ──────  ─────────────────────  ──────────────  ─────────────────── │
  │  ION     IonSelectiveElectrode  NH₄⁺ / NO₃⁻    voltage (mV) → mM   │
  └──────────────────────────────────────────────────────────────────────┘

Mode Switching
──────────────
The AmmoniaSuite class switches between two detection domains:

  mode="gas"  → layers 1, 2, 3  (vapour-phase NH₃ detection)
  mode="ion"  → IonSelectiveElectrode  (solid/dissolved NH₄NO₃ detection)

This mirrors the physical reality: NH₃ is a gas, while NH₄NO₃ is a solid
or dissolved salt requiring ion-exchange detection.

  ┌────────────────────────────────────────────────────────────────────┐
  │  Change          NH₃                    NH₄NO₃                   │
  │  State           Gas / Vapour           Solid / Particulate       │
  │  Sensor Type     EC / MOS gas sensor    Ion Selective Electrode   │
  │  Visual Signal   SWIR abs 1.5 µm        NIR fingerprint 700-800nm │
  │  Output          ppm current            mV ion activity           │
  └────────────────────────────────────────────────────────────────────┘

The AmmoniaSuite also generates the Gaussian-peak "digital fingerprint"
signal for each mode using :func:`generate_chemical_signal`, enabling
the LLM fusion engine to reason about related compounds without hardware.

Hardware integration notes
──────────────────────────
EC sensor   : Alphasense NH3-A1 or SGX SensorTech EC4-NH3
              → Connect to Nano 33 BLE via I2C transimpedance amplifier
MOS sensor  : Figaro TGS2444 (NH3-selective, 10 ppb–300 ppm)
              → Ratiometric ADC read: Rs = (Vcc/Vout − 1) × Rload
IR sensor   : Alphasense IRcycle NH3 or Integrated Device
              → SPI/UART to Raspberry Pi; returns absorbance directly
ISE         : Vernier Ion Selective Electrode (NH4⁺) or HANNA HI4101
              → Nernst equation: V = E₀ + (RT/zF) × ln(activity)

References
──────────
[1] DOL ammonia sensors: https://www.dol-sensors.com/content-pages/ammonia-measurement-sensors
[2] Timberline NH3 detection: https://www.timberlineinstruments.com/ammonia-detection-equipment/
[3] NIR ammonia absorption: https://pmc.ncbi.nlm.nih.gov/articles/PMC11125007/
[4] GfG electrochemical NH3: https://www.gfgsafety.com/
[5] Ion selective electrode: https://forum.arduino.cc/t/does-anyone-know-of-any-sensors-that-can-detect-ammonia-dissolved-in-water/1211307
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.ndimage import gaussian_filter1d

from analyzer.virtual_sensors import (
    CHEMICAL_SIGNAL_LIBRARY,
    generate_chemical_signal,
)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _zscore(arr: np.ndarray) -> np.ndarray:
    """Z-score normalise: zero mean, unit variance."""
    std = arr.std()
    if std < 1e-9:
        return arr - arr.mean()
    return (arr - arr.mean()) / std


# ---------------------------------------------------------------------------
# Layer 1 – Electrochemical (EC) Gas Sensor
# ---------------------------------------------------------------------------


@dataclass
class ECReading:
    """Single reading from the electrochemical sensor."""
    raw_current_ua: float
    concentration_ppm: float
    temperature_c: float = 20.0
    humidity_pct: float = 50.0


class ElectrochemicalSensor:
    """
    Digital Twin of an electrochemical NH₃ gas sensor (e.g. Alphasense NH3-A1).

    The sensing reaction at the working electrode:
        NH₃  →  ½ N₂  +  3 H⁺  +  3 e⁻

    Current model:
        I (µA) = sensitivity × ppm × Tc × Hc + dark_current + noise

    where Tc = temperature correction, Hc = humidity correction.

    Accuracy: ±2 % of reading or ±0.5 ppm (whichever is larger).
    """

    DEFAULT_SENSITIVITY_UA_PPM = 0.06   # µA per ppm (Alphasense NH3-A1 typical)
    TEMP_COEFFICIENT = 0.002            # 0.2 % per °C deviation from 20 °C

    def __init__(
        self,
        sensitivity_ua_ppm: float = DEFAULT_SENSITIVITY_UA_PPM,
        dark_current_ua: float = 0.003,
        noise_std_ua: float = 0.001,
        rng: Optional[np.random.Generator] = None,
    ):
        self.sensitivity_ua_ppm = sensitivity_ua_ppm
        self.dark_current_ua = dark_current_ua
        self.noise_std_ua = noise_std_ua
        self._rng = rng or np.random.default_rng()

    def measure(
        self,
        ppm: float,
        temperature_c: float = 20.0,
        humidity_pct: float = 50.0,
    ) -> ECReading:
        """
        Simulate a sensor measurement at given concentration and conditions.

        Args:
            ppm:           True NH₃ concentration (ppm).
            temperature_c: Ambient temperature (°C).
            humidity_pct:  Relative humidity (%).

        Returns:
            ECReading with raw current and calculated concentration.
        """
        tc = 1.0 + self.TEMP_COEFFICIENT * (temperature_c - 20.0)
        hc = 1.0 + 0.001 * (humidity_pct - 50.0)    # ±0.1 % per %RH
        signal_ua = self.sensitivity_ua_ppm * ppm * tc * hc
        noise = self._rng.normal(0.0, self.noise_std_ua)
        current = signal_ua + self.dark_current_ua + noise
        measured_ppm = max(
            0.0, (current - self.dark_current_ua) / (self.sensitivity_ua_ppm * tc * hc)
        )
        return ECReading(
            raw_current_ua=round(current, 6),
            concentration_ppm=round(measured_ppm, 3),
            temperature_c=temperature_c,
            humidity_pct=humidity_pct,
        )

    def to_text_summary(self, reading: ECReading) -> str:
        return (
            f"EC sensor: {reading.concentration_ppm:.3f} ppm NH₃ "
            f"(I={reading.raw_current_ua:.4f} µA, "
            f"T={reading.temperature_c}°C, RH={reading.humidity_pct}%)."
        )


# ---------------------------------------------------------------------------
# Layer 2 – Metal Oxide Sensor (MOS)
# ---------------------------------------------------------------------------


@dataclass
class MOSReading:
    """Single reading from the metal-oxide sensor."""
    resistance_ohm: float
    ratio_rs_r0: float        # Rs/R0 ratio (normalised to clean air)
    concentration_ppb: float  # Estimated concentration (ppb — sub-ppm)
    sensitivity_class: str    # "low" / "medium" / "high"


class MetalOxideSensor:
    """
    Digital Twin of a metal-oxide semiconductor (MOS) NH₃ sensor
    (e.g. Figaro TGS2444, MQ-137).

    Physical principle:
        At high temperatures (~200–400 °C), NH₃ reduces the surface
        resistance of the metal-oxide (SnO₂ or WO₃) film.  The ratio
        Rs/R0 (measured resistance / clean-air resistance) follows a
        power-law curve:

            ppm = a × (Rs/R0)^b

    MOS sensors are highly sensitive (ppb range) but cross-sensitive
    to humidity and require a warm-up period of ~30 s after power-on.

    Heater wiring (ESP32):
        - Heater pin → 5 V (via 33 Ω resistor for ~150 mW)
        - AOUT → ADC GPIO (3.3 V, 12-bit)
        - Rs = (Vcc/Vadc − 1) × Rload
    """

    # Default power-law coefficients for TGS2444 in clean air → NH₃
    DEFAULT_A = 100.0   # clean-air baseline (Ω equivalent)
    DEFAULT_B = -1.5    # slope on log-log curve

    def __init__(
        self,
        r0_clean_air_ohm: float = 10_000.0,
        a: float = DEFAULT_A,
        b: float = DEFAULT_B,
        noise_std_ohm: float = 50.0,
        rng: Optional[np.random.Generator] = None,
    ):
        """
        Args:
            r0_clean_air_ohm: Baseline resistance in clean air (Ω).
            a, b:             Power-law calibration coefficients.
            noise_std_ohm:    Resistance noise (Ω).
        """
        self.r0 = r0_clean_air_ohm
        self.a = a
        self.b = b
        self.noise_std_ohm = noise_std_ohm
        self._rng = rng or np.random.default_rng()

    def measure(self, ppb: float) -> MOSReading:
        """
        Simulate a MOS resistance measurement.

        Args:
            ppb: True NH₃ concentration in parts-per-billion.

        Returns:
            MOSReading with resistance, ratio, and estimated ppb.
        """
        ppm = ppb / 1000.0
        # Rs decreases as NH₃ concentration increases (reducing gas)
        rs_r0 = max(0.05, 1.0 / (1.0 + ppm / self.a))
        rs = rs_r0 * self.r0 + self._rng.normal(0.0, self.noise_std_ohm)
        rs = max(0.0, rs)
        actual_ratio = rs / self.r0
        # Inverse model: ppm = a × (ratio)^b  → rearranged
        est_ppm = max(0.0, self.a * (actual_ratio ** self.b) - self.a)
        est_ppb = est_ppm * 1000.0

        if est_ppb < 50:
            sens = "low"
        elif est_ppb < 500:
            sens = "medium"
        else:
            sens = "high"

        return MOSReading(
            resistance_ohm=round(rs, 1),
            ratio_rs_r0=round(actual_ratio, 4),
            concentration_ppb=round(est_ppb, 2),
            sensitivity_class=sens,
        )

    def to_text_summary(self, reading: MOSReading) -> str:
        return (
            f"MOS sensor (Figaro TGS2444): Rs/R0={reading.ratio_rs_r0:.4f}, "
            f"estimated NH₃ = {reading.concentration_ppb:.1f} ppb "
            f"[{reading.sensitivity_class} sensitivity]."
        )


# ---------------------------------------------------------------------------
# Layer 3 – Infrared / SWIR Sensor  (1450–1560 nm absorption window)
# ---------------------------------------------------------------------------


@dataclass
class IRReading:
    """Single reading from the infrared NH₃ sensor."""
    absorbance_au: float       # Absorbance in Beer-Lambert units (AU)
    concentration_ppm: float   # Calculated via Beer-Lambert law
    wavelength_nm: float       # Centre of measurement window (nm)


class InfraredNH3Sensor:
    """
    Digital Twin of an infrared / SWIR optical NH₃ sensor.

    Target absorption window: 1450–1560 nm (N-H overtone stretch)
    This is the "visual" component of the Ammonia Suite, capturing the
    characteristic IR signature that hyperspectral cameras detect.

    Beer-Lambert law model:
        A = ε × c × l

    where:
        A = absorbance (AU)
        ε = molar absorptivity (AU/(ppm·cm))
        c = concentration (ppm)
        l = optical path length (cm)

    References:
        [3] NIR ammonia absorption: https://pmc.ncbi.nlm.nih.gov/articles/PMC11125007/
    """

    NH3_ABSORPTION_NM = 1510.0          # Primary SWIR absorption peak (nm)
    NH3_MOLAR_ABS = 2.5e-4              # Typical ε for NH₃ at 1510 nm (AU/ppm/cm)

    def __init__(
        self,
        wavelength_nm: float = NH3_ABSORPTION_NM,
        path_length_cm: float = 10.0,
        molar_absorptivity: float = NH3_MOLAR_ABS,
        noise_std_au: float = 5e-4,
        rng: Optional[np.random.Generator] = None,
    ):
        """
        Args:
            wavelength_nm:       Centre wavelength of the measurement (nm).
            path_length_cm:      Optical cell path length (cm).
            molar_absorptivity:  ε in AU per ppm per cm.
            noise_std_au:        Absorbance noise standard deviation.
        """
        self.wavelength_nm = wavelength_nm
        self.path_length_cm = path_length_cm
        self.molar_absorptivity = molar_absorptivity
        self.noise_std_au = noise_std_au
        self._rng = rng or np.random.default_rng()

    def measure(self, ppm: float) -> IRReading:
        """
        Simulate an IR absorbance measurement.

        Args:
            ppm: True NH₃ concentration (ppm).

        Returns:
            IRReading with absorbance and estimated concentration.
        """
        A_true = self.molar_absorptivity * ppm * self.path_length_cm
        noise = self._rng.normal(0.0, self.noise_std_au)
        A_meas = max(0.0, A_true + noise)
        est_ppm = A_meas / max(
            self.molar_absorptivity * self.path_length_cm, 1e-12
        )
        return IRReading(
            absorbance_au=round(A_meas, 6),
            concentration_ppm=round(est_ppm, 3),
            wavelength_nm=self.wavelength_nm,
        )

    def to_text_summary(self, reading: IRReading) -> str:
        return (
            f"IR sensor ({reading.wavelength_nm:.0f} nm): "
            f"absorbance = {reading.absorbance_au:.5f} AU → "
            f"{reading.concentration_ppm:.3f} ppm NH₃."
        )


# ---------------------------------------------------------------------------
# ION Layer – Ion Selective Electrode (ISE) for NH₄⁺ / NO₃⁻
# ---------------------------------------------------------------------------


@dataclass
class ISEReading:
    """Single reading from an ion selective electrode."""
    raw_voltage_mv: float       # Measured EMF (mV)
    ion_activity_mm: float      # Ion activity (millimolar)
    ion_name: str               # "NH4+" or "NO3-"
    nernst_slope_mv: float      # Actual Nernst slope (mV/decade)


class IonSelectiveElectrode:
    """
    Digital Twin of an Ion Selective Electrode (ISE) for detecting
    ammonium (NH₄⁺) or nitrate (NO₃⁻) ions in solution.

    Used when operating in "ion" mode (solid / dissolved NH₄NO₃).

    Nernst equation:
        E = E₀ + (RT / zF) × ln(activity)
        E = E₀ + S × log₁₀(activity)   [Nernst slope S ≈ 59.2 mV/decade at 25°C]

    Hardware:
        - Vernier Ion Selective Electrode (NH₄⁺) or HANNA HI4101
        - Connect to Arduino via differential ADC (ADS1115)
        - Reference electrode required (Ag/AgCl double-junction)

    References:
        [6] Arduino ISE forum: https://forum.arduino.cc/t/does-anyone-know-of-any-sensors-that-can-detect-ammonia-dissolved-in-water/1211307
    """

    IDEAL_NERNST_SLOPE = 59.16   # mV/decade at 25 °C (z=1 ion)
    R = 8.314    # J/(mol·K)
    F = 96485.0  # C/mol

    def __init__(
        self,
        ion_name: str = "NH4+",
        e0_mv: float = 200.0,
        nernst_slope_mv: float = IDEAL_NERNST_SLOPE,
        noise_std_mv: float = 0.5,
        rng: Optional[np.random.Generator] = None,
    ):
        """
        Args:
            ion_name:          "NH4+" or "NO3-".
            e0_mv:             Standard electrode potential (mV).
            nernst_slope_mv:   Actual Nernst slope (mV/decade); should be
                                near 59.2 mV for monovalent ions at 25 °C.
            noise_std_mv:      EMF measurement noise (mV).
        """
        self.ion_name = ion_name
        self.e0_mv = e0_mv
        self.nernst_slope_mv = nernst_slope_mv
        self.noise_std_mv = noise_std_mv
        self._rng = rng or np.random.default_rng()

    def measure(self, activity_mm: float) -> ISEReading:
        """
        Simulate an ISE measurement for a given ion activity.

        Args:
            activity_mm: True ion activity in millimolar (mM).

        Returns:
            ISEReading with raw voltage and estimated activity.
        """
        activity_mol = activity_mm * 1e-3   # mM → mol/L
        if activity_mol <= 0:
            log_a = -9.0
        else:
            log_a = np.log10(activity_mol)

        v_true = self.e0_mv + self.nernst_slope_mv * log_a
        noise = self._rng.normal(0.0, self.noise_std_mv)
        v_meas = v_true + noise

        # Invert Nernst equation to recover activity
        log_a_est = (v_meas - self.e0_mv) / self.nernst_slope_mv
        est_mol = 10 ** log_a_est
        est_mm = est_mol * 1e3

        return ISEReading(
            raw_voltage_mv=round(v_meas, 3),
            ion_activity_mm=round(max(0.0, est_mm), 4),
            ion_name=self.ion_name,
            nernst_slope_mv=self.nernst_slope_mv,
        )

    def to_text_summary(self, reading: ISEReading) -> str:
        return (
            f"ISE ({reading.ion_name}): {reading.raw_voltage_mv:.2f} mV → "
            f"{reading.ion_activity_mm:.4f} mM ion activity."
        )


# ---------------------------------------------------------------------------
# AmmoniaSuite — Master controller
# ---------------------------------------------------------------------------


class AmmoniaSuite:
    """
    Virtual Sensor Array for Ammonia-Based Compounds.

    Orchestrates all three gas-phase sensing layers (EC + MOS + IR) when
    in "gas" mode, and switches to the Ion Selective Electrode in "ion"
    mode for solid/dissolved NH₄NO₃ detection.

    Quick start::

        suite = AmmoniaSuite(mode="gas")
        suite.read_signal(15)
        # → "NH3 Gas Concentration: 18.0 ppm"

        suite = AmmoniaSuite(mode="ion")
        suite.read_signal(0.85)
        # → "Nitrate Ion Activity: 0.7225 mV"

        # Full multi-layer reading:
        result = AmmoniaSuite(mode="gas").full_reading(ppm=30.0)
    """

    VALID_MODES = ("gas", "ion")

    def __init__(
        self,
        mode: str = "gas",
        rng: Optional[np.random.Generator] = None,
    ):
        """
        Args:
            mode: "gas"  → EC + MOS + IR (NH₃ vapour)
                  "ion"  → ISE (NH₄⁺ / NO₃⁻ in solution or solid)
        """
        if mode not in self.VALID_MODES:
            raise ValueError(f"mode must be one of {self.VALID_MODES}, got {mode!r}.")
        self.mode = mode
        _rng = rng or np.random.default_rng()

        # Initialise all sensor layers
        self._ec  = ElectrochemicalSensor(rng=_rng)
        self._mos = MetalOxideSensor(rng=_rng)
        self._ir  = InfraredNH3Sensor(rng=_rng)
        self._ise = IonSelectiveElectrode(ion_name="NH4+", rng=_rng)

    # ------------------------------------------------------------------
    # Simple API matching the requirement's AmmoniaSuite sketch
    # ------------------------------------------------------------------

    def read_signal(self, raw_input: float) -> str:
        """
        Minimal interface matching the requirement's sketch.

        In "gas" mode, raw_input is a concentration (ppm).
        In "ion" mode, raw_input is a raw electrode voltage reading.

        Returns:
            Human-readable measurement string.
        """
        if self.mode == "gas":
            # EC sensor converts current → ppm; here raw_input is already ppm
            ppm = self._ec.measure(raw_input).concentration_ppm
            return f"NH3 Gas Concentration: {ppm:.1f} ppm"
        else:
            # ISE: raw_input is raw voltage (mV); derive activity
            activity = raw_input * 0.85
            return f"Nitrate Ion Activity: {activity:.4f} mV"

    # ------------------------------------------------------------------
    # Full multi-layer reading
    # ------------------------------------------------------------------

    def full_reading(
        self,
        ppm: float = 0.0,
        ppb: Optional[float] = None,
        ion_activity_mm: float = 0.0,
        temperature_c: float = 20.0,
        humidity_pct: float = 50.0,
    ) -> Dict:
        """
        Run all active sensor layers and return structured results.

        Args:
            ppm:              NH₃ gas concentration (ppm) for gas mode.
            ppb:              NH₃ concentration in ppb (overrides ppm if set).
            ion_activity_mm:  NH₄⁺ / NO₃⁻ ion activity (mM) for ion mode.
            temperature_c:    Ambient temperature (°C).
            humidity_pct:     Relative humidity (%).

        Returns:
            Dict with sensor readings and chemical signal fingerprint.
        """
        if ppb is not None:
            ppm = ppb / 1000.0

        result: Dict = {"mode": self.mode}

        if self.mode == "gas":
            ec_reading  = self._ec.measure(ppm, temperature_c, humidity_pct)
            mos_reading = self._mos.measure(ppm * 1000.0)  # ppm → ppb
            ir_reading  = self._ir.measure(ppm)

            # Chemical signal fingerprint
            signal_nh3 = generate_chemical_signal("ammonia", intensity=ppm / 25.0 or 0.1)

            result.update({
                "ec":             ec_reading,
                "mos":            mos_reading,
                "ir":             ir_reading,
                "signal_peak":    float(signal_nh3.max()),
                "signal_snr":     _signal_snr(signal_nh3),
                "chemical":       "NH3",
                "text_summaries": [
                    self._ec.to_text_summary(ec_reading),
                    self._mos.to_text_summary(mos_reading),
                    self._ir.to_text_summary(ir_reading),
                ],
            })
        else:
            ise_reading = self._ise.measure(ion_activity_mm)

            # Chemical signal fingerprint for NH₄NO₃
            scale = ion_activity_mm / 10.0 if ion_activity_mm > 0 else 0.1
            signal_salt = generate_chemical_signal("ammonium_nitrate", intensity=scale)

            result.update({
                "ise":            ise_reading,
                "signal_peak":    float(signal_salt.max()),
                "signal_snr":     _signal_snr(signal_salt),
                "chemical":       "NH4NO3",
                "text_summaries": [
                    self._ise.to_text_summary(ise_reading),
                ],
            })

        return result

    def to_gemma_prompt(
        self,
        ppm: float = 0.0,
        ion_activity_mm: float = 0.0,
    ) -> str:
        """
        Format all sensor readings as a structured text prompt for Gemma 2.

        Args:
            ppm:             NH₃ concentration (gas mode).
            ion_activity_mm: Ion activity (ion mode).

        Returns:
            Formatted multi-line string for LLM input.
        """
        data = self.full_reading(ppm=ppm, ion_activity_mm=ion_activity_mm)
        summaries = "\n    ".join(data.get("text_summaries", []))
        chemical = data["chemical"]
        snr = data["signal_snr"]
        peak = data["signal_peak"]
        mode_desc = (
            "gas-phase electrochemical + MOS + IR"
            if self.mode == "gas"
            else "ion-selective electrode"
        )

        prompt = (
            f"\n{'═'*52}\n"
            f"  AMMONIA SUITE — SENSOR DATA REPORT\n"
            f"{'═'*52}\n"
            f"  Detection mode   : {self.mode.upper()} ({mode_desc})\n"
            f"  Target chemical  : {chemical}\n"
            f"  Signal peak      : {peak:.4f}\n"
            f"  Signal SNR       : {snr:.1f} dB\n"
            f"  ─── Sensor readings ───\n"
            f"    {summaries}\n"
            f"{'─'*52}\n"
            f"  TASK: Determine whether this reading indicates\n"
            f"  (a) pure NH₃ gas, (b) NH₄NO₃ solid/dissolved,\n"
            f"  or (c) a mixture.  Flag if hazardous (>25 ppm).\n"
            f"{'═'*52}\n"
        )
        return prompt


# ---------------------------------------------------------------------------
# Module-level convenience function (matches the requirement's sketch)
# ---------------------------------------------------------------------------


def gemma_filter(sensor_data: str, backend: str = "local") -> str:
    """
    Stub function that formats a Gemma 2 reasoning request.

    For full LLM integration use :class:`analyzer.fusion.SensorFusionLLM`.

    Args:
        sensor_data: Structured sensor text (output of to_gemma_prompt()).
        backend:     "local" (rule-based) or "ollama" (Gemma 2 via Ollama).

    Returns:
        Identification string.
    """
    prompt = f"Analyze this chemical data: {sensor_data}. Is this pure Ammonia or a Nitrate salt?"

    if backend == "local":
        # Simple rule-based local reasoning
        if "NH4NO3" in sensor_data or "ammonium_nitrate" in sensor_data.lower():
            return "Local filter identifies: Ammonium Nitrate (nitrate salt form)"
        if "NH3" in sensor_data or "ammonia" in sensor_data.lower():
            return "Local filter identifies: Ammonia gas (vapour phase)"
        return "Local filter: insufficient signal for identification"

    # Ollama/LLM path — handled by SensorFusionLLM.analyze_with_llm()
    return f"LLM Reasoning requested: '{prompt[:80]}...'"


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _signal_snr(signal: np.ndarray) -> float:
    """SNR in dB: 20 × log10(peak / noise_rms)."""
    peak = float(signal.max())
    noise = float(np.std(signal[signal < np.percentile(signal, 25)]))
    if noise < 1e-12:
        return 99.9
    return round(20.0 * np.log10(peak / noise), 2)
