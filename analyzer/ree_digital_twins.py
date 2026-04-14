"""
Rare Earth Element Digital Twin Framework
═════════════════════════════════════════
Core Sense-Model-Transmit architecture for all 17 REE Digital Twins.

Each element subclass implements the standardised loop:

    raw_data  →  sense()  →  model()  →  transmit()  →  RF signal

RF Signature Types
──────────────────
  WIDEBAND_NOISE  : flat-spectrum Gaussian noise (processing runoff, diffuse)
  OSCILLATORY     : sinusoidal carrier (gas emissions, periodic cycles)
  PULSED          : Gaussian pulse train (discrete particle events)
  STEADY_STATE    : constant-amplitude CW tone (continuous mining dust)
  DECAY_PATTERN   : exponentially decaying oscillation (radioactive decay)
  HARMONIC        : fundamental + harmonics (heat radiation / phonon modes)

References
──────────
Digital Twin framework: https://pmc.ncbi.nlm.nih.gov/articles/PMC9427850/
GNU Radio modulation:   https://www.youtube.com/watch?v=BUQkb1StMR8
REE hyperspectral:      https://www.mdpi.com/2072-4292/17/21/3615
RF-emission mapping:    https://arxiv.org/html/2601.01321v1
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------


class RFSignatureType(str, Enum):
    WIDEBAND_NOISE = "Wideband Noise"
    OSCILLATORY    = "Oscillatory"
    PULSED         = "Pulsed"
    STEADY_STATE   = "Steady State"
    DECAY_PATTERN  = "Decay Pattern"
    HARMONIC       = "Harmonic"


class REECategory(str, Enum):
    LREE = "Light Rare Earth"   # La, Ce, Pr, Nd, Pm, Sm, Eu, Gd, Sc
    HREE = "Heavy Rare Earth"   # Tb, Dy, Ho, Er, Tm, Yb, Lu, Y


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class REEProperties:
    """Physicochemical and sensing properties of one rare earth element."""
    symbol: str
    atomic_number: int
    name: str
    category: REECategory
    primary_spectral_line_nm: float
    rf_signature_type: RFSignatureType
    emission_focus: str                          # Environmental monitoring focus
    primary_use: str                             # Main technological application
    ore_minerals: List[str]
    secondary_spectral_lines_nm: List[float] = field(default_factory=list)
    ionization_energy_ev: float = 0.0
    abundance_ppm_earth_crust: float = 0.0
    hazard_notes: str = ""
    rf_carrier_hz: float = 1000.0               # Default RF carrier (Hz, baseband)
    rf_harmonic_count: int = 3                  # Harmonics for HARMONIC type


@dataclass
class SenseResult:
    """Output of the Sense step."""
    element: str
    raw_value: float
    processed_value: float
    spectral_intensity: float
    units: str = "normalised"


@dataclass
class ModelResult:
    """Output of the Model step."""
    element: str
    baseline_value: float
    current_value: float
    deviation: float
    predicted_concentration: float
    anomaly_detected: bool
    confidence: float


@dataclass
class TransmitResult:
    """Output of the Transmit step — a baseband RF waveform."""
    element: str
    rf_signal: np.ndarray
    sample_rate_hz: float
    carrier_freq_hz: float
    rf_type: RFSignatureType
    duration_s: float
    amplitude: float


@dataclass
class DTCycleResult:
    """Complete result of one Sense-Model-Transmit cycle."""
    sense:    SenseResult
    model:    ModelResult
    transmit: TransmitResult
    gemma_prompt: str = ""


# ---------------------------------------------------------------------------
# RF Signal Generator
# ---------------------------------------------------------------------------


class RFSignalGenerator:
    """
    Generates baseband RF waveforms corresponding to each REE emission type.

    All signals are sampled at sample_rate_hz and last duration_s seconds.
    They represent the electromagnetic emission profile of a mining/processing
    site, suitable for modulation into a real RF carrier via GNU Radio or USRP.
    """

    def __init__(
        self,
        sample_rate_hz: float = 44100.0,
        duration_s: float = 1.0,
        rng: Optional[np.random.Generator] = None,
    ):
        self.sample_rate_hz = sample_rate_hz
        self.duration_s = duration_s
        self._rng = rng or np.random.default_rng()

    @property
    def num_samples(self) -> int:
        return int(self.sample_rate_hz * self.duration_s)

    def generate(
        self,
        rf_type: RFSignatureType,
        carrier_hz: float = 1000.0,
        amplitude: float = 1.0,
        harmonic_count: int = 3,
    ) -> np.ndarray:
        """
        Generate a normalised baseband waveform.

        Args:
            rf_type:        Waveform type (see RFSignatureType).
            carrier_hz:     Carrier / fundamental frequency (Hz).
            amplitude:      Peak amplitude.
            harmonic_count: Number of harmonics for HARMONIC type.

        Returns:
            1-D float32 array of length num_samples.
        """
        t = np.linspace(0, self.duration_s, self.num_samples, endpoint=False)

        if rf_type == RFSignatureType.WIDEBAND_NOISE:
            sig = self._rng.normal(0, 1.0, self.num_samples)
            sig = gaussian_filter1d(sig, sigma=2.0)          # flatten spectrum

        elif rf_type == RFSignatureType.OSCILLATORY:
            sig = np.sin(2 * np.pi * carrier_hz * t)

        elif rf_type == RFSignatureType.PULSED:
            pulse_interval = max(1, int(self.sample_rate_hz / carrier_hz))
            sig = np.zeros(self.num_samples)
            sigma = pulse_interval / 6.0
            for start in range(0, self.num_samples, pulse_interval):
                idx = np.arange(self.num_samples)
                sig += np.exp(-((idx - start) ** 2) / (2 * sigma ** 2))

        elif rf_type == RFSignatureType.STEADY_STATE:
            sig = np.ones(self.num_samples)

        elif rf_type == RFSignatureType.DECAY_PATTERN:
            tau = self.duration_s / 3.0
            sig = np.exp(-t / tau) * np.cos(2 * np.pi * carrier_hz * t)

        elif rf_type == RFSignatureType.HARMONIC:
            sig = np.zeros(self.num_samples)
            for n in range(1, harmonic_count + 1):
                sig += (1.0 / n) * np.sin(2 * np.pi * n * carrier_hz * t)

        else:
            sig = np.zeros(self.num_samples)

        # Normalise and scale
        peak = np.abs(sig).max()
        if peak > 1e-9:
            sig = sig / peak
        return (sig * amplitude).astype(np.float32)


# ---------------------------------------------------------------------------
# Base Digital Twin
# ---------------------------------------------------------------------------


class REEDigitalTwin(ABC):
    """
    Abstract base class for a REE Digital Twin.

    Implements the Sense-Model-Transmit loop.  Subclasses (one per element)
    supply the REEProperties descriptor; the base class handles all signal
    processing and RF generation logic.
    """

    # Baseline concentration (normalised) used by the model step.
    BASELINE = 0.1

    def __init__(
        self,
        properties: REEProperties,
        sample_rate_hz: float = 44100.0,
        rng: Optional[np.random.Generator] = None,
    ):
        self.props = properties
        self._rng = rng or np.random.default_rng()
        self._rf_gen = RFSignalGenerator(
            sample_rate_hz=sample_rate_hz,
            rng=self._rng,
        )
        self._baseline = self.BASELINE

    # ------------------------------------------------------------------
    # Step 1 – Sense
    # ------------------------------------------------------------------

    def sense(self, raw_input: float) -> SenseResult:
        """
        Convert a raw environmental signal into a processed spectral intensity.

        The raw_input can be:
          • A simulated concentration value (float, 0–1 range)
          • A voltage from an ADC (0–3.3 V)
          • A reflectance value from a pixel

        Args:
            raw_input: Raw scalar sensor reading.

        Returns:
            SenseResult with processed intensity.
        """
        # Add instrument noise
        noise = float(self._rng.normal(0, 0.02))
        processed = float(np.clip(raw_input + noise, 0.0, 1.0))

        # Spectral intensity at the element's primary emission line
        spectral_intensity = processed * np.exp(
            -((processed - 0.5) ** 2) / 0.1
        )

        return SenseResult(
            element=self.props.name,
            raw_value=float(raw_input),
            processed_value=processed,
            spectral_intensity=round(spectral_intensity, 5),
        )

    # ------------------------------------------------------------------
    # Step 2 – Model  (compare vs baseline; detect anomalies)
    # ------------------------------------------------------------------

    def model(self, sense_result: SenseResult) -> ModelResult:
        """
        Compare the sensed value against the baseline model and predict
        current concentration.

        Args:
            sense_result: Output of sense().

        Returns:
            ModelResult with deviation and anomaly flag.
        """
        deviation = sense_result.processed_value - self._baseline
        anomaly = abs(deviation) > 0.25          # 25 % deviation threshold
        # Update baseline with exponential moving average (α = 0.05)
        self._baseline = 0.95 * self._baseline + 0.05 * sense_result.processed_value

        confidence = float(np.clip(1.0 - abs(deviation) / 2.0, 0.0, 1.0))
        predicted_ppm = sense_result.processed_value * self.props.abundance_ppm_earth_crust

        return ModelResult(
            element=self.props.name,
            baseline_value=round(self._baseline, 5),
            current_value=sense_result.processed_value,
            deviation=round(deviation, 5),
            predicted_concentration=round(predicted_ppm, 4),
            anomaly_detected=anomaly,
            confidence=round(confidence, 4),
        )

    # ------------------------------------------------------------------
    # Step 3 – Transmit  (RF waveform generation)
    # ------------------------------------------------------------------

    def transmit(self, model_result: ModelResult) -> TransmitResult:
        """
        Convert the modelled emission level into an RF baseband signal.

        The signal amplitude is proportional to the current measured value,
        so higher concentrations produce stronger RF emissions — mimicking
        how a real USRP/GNU Radio SDR would broadcast the emission level.

        Args:
            model_result: Output of model().

        Returns:
            TransmitResult with the RF waveform array.
        """
        amplitude = float(np.clip(model_result.current_value, 0.0, 1.0))
        signal = self._rf_gen.generate(
            rf_type=self.props.rf_signature_type,
            carrier_hz=self.props.rf_carrier_hz,
            amplitude=amplitude,
            harmonic_count=self.props.rf_harmonic_count,
        )
        return TransmitResult(
            element=self.props.name,
            rf_signal=signal,
            sample_rate_hz=self._rf_gen.sample_rate_hz,
            carrier_freq_hz=self.props.rf_carrier_hz,
            rf_type=self.props.rf_signature_type,
            duration_s=self._rf_gen.duration_s,
            amplitude=amplitude,
        )

    # ------------------------------------------------------------------
    # Full cycle
    # ------------------------------------------------------------------

    def run_cycle(self, raw_input: float) -> DTCycleResult:
        """
        Execute one complete Sense → Model → Transmit cycle.

        Args:
            raw_input: Raw environmental sensor reading.

        Returns:
            DTCycleResult with all three step outputs and a Gemma prompt.
        """
        s = self.sense(raw_input)
        m = self.model(s)
        t = self.transmit(m)
        prompt = self._build_gemma_prompt(s, m, t)
        return DTCycleResult(sense=s, model=m, transmit=t, gemma_prompt=prompt)

    # ------------------------------------------------------------------
    # Gemma-2 prompt builder
    # ------------------------------------------------------------------

    def _build_gemma_prompt(
        self,
        s: SenseResult,
        m: ModelResult,
        t: TransmitResult,
    ) -> str:
        return (
            f"\n{'═'*52}\n"
            f"  REE DIGITAL TWIN — {self.props.name.upper()} ({self.props.symbol})\n"
            f"{'═'*52}\n"
            f"  Atomic number  : {self.props.atomic_number}\n"
            f"  Category       : {self.props.category.value}\n"
            f"  Spectral line  : {self.props.primary_spectral_line_nm} nm\n"
            f"  Emission focus : {self.props.emission_focus}\n"
            f"  ─── SENSE ───\n"
            f"  Raw input      : {s.raw_value:.4f}\n"
            f"  Processed      : {s.processed_value:.4f}\n"
            f"  Spectral intensity: {s.spectral_intensity:.5f}\n"
            f"  ─── MODEL ───\n"
            f"  Baseline       : {m.baseline_value:.5f}\n"
            f"  Deviation      : {m.deviation:+.5f}\n"
            f"  Anomaly        : {'YES ⚠' if m.anomaly_detected else 'NO'}\n"
            f"  Confidence     : {m.confidence:.1%}\n"
            f"  ─── TRANSMIT ───\n"
            f"  RF type        : {t.rf_type.value}\n"
            f"  Carrier        : {t.carrier_freq_hz:.0f} Hz\n"
            f"  Amplitude      : {t.amplitude:.4f}\n"
            f"{'─'*52}\n"
            f"  TASK: Given the above sensor data for {self.props.name},\n"
            f"  determine whether these readings indicate normal background\n"
            f"  levels or an elevated emission event at a mining/processing\n"
            f"  site. Primary use: {self.props.primary_use}.\n"
            f"{'═'*52}\n"
        )

    def to_text_summary(self, result: DTCycleResult) -> str:
        """Short text summary for Sensor-LLM late fusion."""
        return (
            f"{self.props.name} DT: intensity={result.sense.spectral_intensity:.4f}, "
            f"deviation={result.model.deviation:+.4f}, "
            f"anomaly={'YES' if result.model.anomaly_detected else 'no'}, "
            f"RF={result.transmit.rf_type.value}."
        )


# ---------------------------------------------------------------------------
# Emissions DataFrame (from requirement)
# ---------------------------------------------------------------------------


def build_emissions_dataframe() -> pd.DataFrame:
    """
    Build the REE emissions DataFrame from the requirement specification.

    Returns:
        pandas DataFrame with Element, Primary_Spectral_Line_nm,
        RF_Signature_Type, Emission_Focus, and Category columns.
    """
    data = {
        "Element": [
            "Lanthanum", "Cerium", "Praseodymium", "Neodymium", "Promethium",
            "Samarium", "Europium", "Gadolinium", "Terbium", "Dysprosium",
            "Holmium", "Erbium", "Thulium", "Ytterbium", "Lutetium",
            "Scandium", "Yttrium",
        ],
        "Symbol": [
            "La", "Ce", "Pr", "Nd", "Pm",
            "Sm", "Eu", "Gd", "Tb", "Dy",
            "Ho", "Er", "Tm", "Yb", "Lu",
            "Sc", "Y",
        ],
        "Atomic_Number": [57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 21, 39],
        "Category": [
            "LREE", "LREE", "LREE", "LREE", "LREE",
            "LREE", "LREE", "LREE", "HREE", "HREE",
            "HREE", "HREE", "HREE", "HREE", "HREE",
            "LREE", "HREE",
        ],
        "Primary_Spectral_Line_nm": [
            394.9, 413.7, 440.8, 430.3, 463.0,
            359.3, 459.4, 342.2, 432.6, 353.1,
            345.6, 337.2, 313.1, 398.8, 451.8,
            361.3, 360.1,
        ],
        "RF_Signature_Type": [
            "Wideband Noise", "Oscillatory", "Pulsed", "Steady State", "Decay Pattern",
            "Harmonic", "Wideband Noise", "Pulsed", "Steady State", "Oscillatory",
            "Harmonic", "Decay Pattern", "Pulsed", "Steady State", "Wideband Noise",
            "Oscillatory", "Harmonic",
        ],
        "Emission_Focus": [
            "Processing runoff", "Gas emissions", "Mining dust", "Mining dust density",
            "Radioactive decay", "Heat radiation", "Luminescence output",
            "Processing runoff", "Heat radiation", "Magnetic field drift",
            "Optical emission", "Fibre amplifier leakage", "X-ray emission",
            "Laser output", "Processing runoff", "Alloy dust", "Phosphor emission",
        ],
        "Primary_Use": [
            "Camera lens glass", "Catalytic converters / polishing",
            "NdPr magnets (EV motors)", "High-strength magnets (wind turbines)",
            "Radioactive tracer (medical)", "Neutron absorber (nuclear)",
            "Red phosphors (screens)", "MRI contrast agent",
            "Green phosphors (LEDs)", "EV motor magnets",
            "Medical lasers", "Fibre optic amplifiers (EDFA)",
            "Surgical lasers", "Fibre lasers",
            "Specialty alloys", "Lightweight Al-Sc alloys (aerospace)",
            "Phosphors / LEDs",
        ],
    }
    return pd.DataFrame(data)


# ---------------------------------------------------------------------------
# Twin Suite — manages all 17 instances
# ---------------------------------------------------------------------------


class REETwinSuite:
    """
    Registry and runner for all 17 REE Digital Twins.

    Holds one twin instance per element and can execute batch detection
    cycles, produce the emissions DataFrame, and generate Gemma-2 prompts.
    """

    def __init__(self, rng: Optional[np.random.Generator] = None):
        self._rng = rng or np.random.default_rng()
        self._twins: Dict[str, REEDigitalTwin] = {}

    def register(self, twin: REEDigitalTwin) -> None:
        """Register a twin (called by each element module's __init__)."""
        self._twins[twin.props.name] = twin

    def get(self, element_name: str) -> REEDigitalTwin:
        """Retrieve a twin by element name (case-insensitive)."""
        for name, twin in self._twins.items():
            if name.lower() == element_name.lower():
                return twin
        raise KeyError(f"No twin registered for {element_name!r}.")

    def run_all(self, raw_inputs: Optional[Dict[str, float]] = None) -> Dict[str, DTCycleResult]:
        """
        Run one Sense-Model-Transmit cycle for every registered twin.

        Args:
            raw_inputs: Dict mapping element name → raw_input value.
                        If None, uses a random value for each element.

        Returns:
            Dict mapping element name → DTCycleResult.
        """
        results: Dict[str, DTCycleResult] = {}
        for name, twin in self._twins.items():
            raw = (raw_inputs or {}).get(name, float(self._rng.random()))
            results[name] = twin.run_cycle(raw)
        return results

    def emissions_dataframe(self) -> pd.DataFrame:
        """Return the standard REE emissions DataFrame."""
        return build_emissions_dataframe()

    def anomalies(
        self, results: Dict[str, DTCycleResult]
    ) -> List[str]:
        """Return list of element names where an anomaly was detected."""
        return [name for name, r in results.items() if r.model.anomaly_detected]

    @property
    def registered_elements(self) -> List[str]:
        return list(self._twins.keys())
