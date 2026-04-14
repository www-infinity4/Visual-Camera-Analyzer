"""
SensorFusionLLM — Late Fusion Architecture
═══════════════════════════════════════════
Implements a "Sensor-to-Text" + LLM reasoning pipeline that fuses heterogeneous
sensor signals into a single chemical identification.

Architecture (Late Fusion)
──────────────────────────
Rather than merging raw sensor arrays (which differ in units, scale, and noise
floor), each sensor module produces a concise natural-language "summary".
A master reasoning engine (the LLM) then reads those summaries and returns a
structured chemical identification — filtering for the target compound.

    ┌──────────────────────────────────────────────────────────────────────┐
    │ Sensor Modules                  Fusion Engine        Output          │
    │ ─────────────────────────────── ──────────────────── ─────────────  │
    │ HyperspectralSensor  ──┐                                             │
    │ PIDSensor            ──┤  text summaries  ──▶ LLM (Gemma 2 /        │
    │ RamanSensor          ──┤  (sensor-to-text)    GPT-4 / local SAM) ──▶│
    │ IMSSensor            ──┤                                  │          │
    │ LWIRSensor           ──┘                     ChemicalIdentification  │
    └──────────────────────────────────────────────────────────────────────┘

Two reasoning backends
──────────────────────
1. **LLM backend** (Gemma 2 / GPT-4 / any OpenAI-compatible endpoint):
   Sends the structured sensor report to a local Ollama server or a remote
   OpenAI-compatible API.  Requires the ``FUSION_LLM_URL`` environment
   variable (default: ``http://localhost:11434/api/generate`` for Ollama
   running Gemma 2).

2. **Local SAM fallback** (no internet / no GPU required):
   When no LLM is available the fusion engine falls back to a Spectral Angle
   Mapper comparison between the observed hyperspectral signature and the
   built-in chemical spectral library.  Returns the best-matching compound
   with a cosine-similarity confidence score.

Supported sensor inputs (all optional)
───────────────────────────────────────
  hsi_signature   np.ndarray (N,)   — mean spectral signature from HSI camera
  pid_ppm         float             — VOC concentration from PID sensor (ppm)
  raman_shift_cm  float             — dominant Raman wavenumber (cm⁻¹)
  ims_score       float             — IMS threat score 0–1
  lwir_peak_um    float             — LWIR absorption peak (µm)
  gas_ppm         float             — electrochemical gas sensor reading (ppm)
  thermal_mC      int               — device thermal zone temp (milli-°C)

Open-source databases
─────────────────────
  SDBS  — Spectral Database for Organic Compounds (IR, Raman, NMR)
          https://sdbs.db.aist.go.jp/
  PubChem — NIH open chemistry database
          https://pubchem.ncbi.nlm.nih.gov/
  spectrai — open-source deep learning for spectral data
          https://github.com/conor-horgan/spectrai

References
──────────
[1] Late fusion multimodal: https://machinelearning.apple.com/research/multimodal-sensor-fusion
[2] Sensor fusion + LLM: https://medium.com/@hhroberthdaniel/sensor-fusion-with-language-models-d6d512bcac2e
[3] SAM algorithm: https://resonon.com/blog-spectronon-hyperspectral-imaging-software
[4] PySptools unmixing: https://pysptools.sourceforge.io/
[5] Gemma 2: https://blog.google/innovation-and-ai/technology/developers-tools/google-gemma-2/
"""

from __future__ import annotations

import json
import os
import time
import urllib.request
import urllib.error
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from analyzer.chemical_signatures import SignatureLibrary, spectral_angle_mapper


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class SensorBundle:
    """Container for raw or pre-processed readings from all sensor types."""

    # Hyperspectral / visual
    hsi_signature: Optional[np.ndarray] = None   # shape (N,)
    hsi_wavelengths_nm: Optional[np.ndarray] = None  # shape (N,) — optional axis labels

    # PID (Photoionisation Detector) — VOCs
    pid_ppm: Optional[float] = None

    # Raman / FTIR spectrometer
    raman_shift_cm: Optional[float] = None        # dominant peak (cm⁻¹)
    raman_intensity: Optional[float] = None

    # Ion Mobility Spectrometry
    ims_score: Optional[float] = None             # 0–1 threat score

    # Thermal Infrared (LWIR)
    lwir_peak_um: Optional[float] = None          # µm

    # Electrochemical gas sensor
    gas_ppm: Optional[float] = None

    # Android hardware proxy (thermal + current)
    thermal_mC: Optional[int] = None              # milli-°C
    battery_flux_uA: Optional[int] = None         # µA

    # Extra free-form metadata
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ChemicalIdentification:
    """Result of the fusion engine's reasoning pass."""

    target_chemical: str
    identified_compound: str
    confidence: float                             # 0.0 – 1.0
    is_hazardous: bool
    hazard_level: str                             # "none" | "low" | "medium" | "high" | "critical"
    reasoning: str                                # LLM text or SAM explanation
    backend_used: str                             # "llm" | "sam_fallback"
    sensor_summaries: List[str] = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "target_chemical": self.target_chemical,
            "identified_compound": self.identified_compound,
            "confidence": round(self.confidence, 4),
            "is_hazardous": self.is_hazardous,
            "hazard_level": self.hazard_level,
            "reasoning": self.reasoning,
            "backend_used": self.backend_used,
            "sensor_summaries": self.sensor_summaries,
            "timestamp": self.timestamp,
        }


# ---------------------------------------------------------------------------
# Sensor-to-Text layer
# ---------------------------------------------------------------------------


class SensorTextEncoder:
    """
    Converts individual sensor readings into concise natural-language summaries
    that an LLM can parse without domain-specific training.

    Each ``describe_*`` method returns a one-sentence summary or ``None`` if
    the sensor was not active.
    """

    # Hazard thresholds (conservative, safety-first)
    PID_HAZARD_PPM = 25.0
    GAS_HAZARD_PPM = 25.0
    IMS_HAZARD_THRESHOLD = 0.6

    def describe_hsi(
        self,
        signature: np.ndarray,
        wavelengths: Optional[np.ndarray] = None,
    ) -> str:
        """Summarise a hyperspectral reflectance signature."""
        peak_idx = int(np.argmin(signature))         # deepest absorption dip
        trough_idx = int(np.argmax(signature))       # highest reflectance
        peak_nm = (
            float(wavelengths[peak_idx]) if wavelengths is not None else float(peak_idx)
        )
        refl_nm = (
            float(wavelengths[trough_idx]) if wavelengths is not None else float(trough_idx)
        )
        mean_r = float(np.mean(signature))
        return (
            f"HSI camera detected a reflectance signature with a primary absorption dip "
            f"at {peak_nm:.1f} nm, peak reflectance at {refl_nm:.1f} nm, "
            f"and a mean reflectance of {mean_r:.3f}."
        )

    def describe_pid(self, ppm: float) -> str:
        level = "HIGH" if ppm > self.PID_HAZARD_PPM else "normal"
        return (
            f"PID sensor reports a VOC concentration of {ppm:.1f} ppm "
            f"(level: {level}, hazard threshold: {self.PID_HAZARD_PPM} ppm)."
        )

    def describe_raman(self, shift_cm: float, intensity: Optional[float] = None) -> str:
        intensity_str = f", intensity {intensity:.3f}" if intensity is not None else ""
        return (
            f"Raman spectrometer identified a dominant peak at {shift_cm:.1f} cm\u207b\u00b9"
            f"{intensity_str}. "
            f"Notable reference shifts: benzene ring ~1008 cm\u207b\u00b9, "
            f"N-H bend ~1450-1560 cm\u207b\u00b9, C=O stretch ~1700-1750 cm\u207b\u00b9."
        )

    def describe_ims(self, score: float) -> str:
        flag = "POSITIVE DETECTION" if score >= self.IMS_HAZARD_THRESHOLD else "below threshold"
        return (
            f"IMS (Ion Mobility Spectrometer) threat score: {score:.3f} "
            f"({flag}, threshold: {self.IMS_HAZARD_THRESHOLD})."
        )

    def describe_lwir(self, peak_um: float) -> str:
        return (
            f"Thermal IR (LWIR) sensor detected a chemical vapor absorption band "
            f"at {peak_um:.2f} \u00b5m. "
            f"Reference: NH\u2083 absorbs at ~10 \u00b5m; CO\u2082 at ~15 \u00b5m; "
            f"ammonia-related bands appear 8\u201312 \u00b5m."
        )

    def describe_gas(self, ppm: float) -> str:
        level = "HAZARDOUS" if ppm > self.GAS_HAZARD_PPM else "safe"
        return (
            f"Electrochemical gas sensor measures {ppm:.1f} ppm "
            f"(status: {level})."
        )

    def describe_thermal(self, thermal_mC: int, flux_uA: Optional[int] = None) -> str:
        temp_c = thermal_mC / 1000.0
        flux_str = ""
        if flux_uA is not None:
            flux_str = f", battery flux: {flux_uA} \u00b5A"
        return (
            f"Hardware thermal sensor: {temp_c:.1f}\u00b0C ({thermal_mC} m\u00b0C){flux_str}."
        )

    def encode(self, bundle: SensorBundle) -> List[str]:
        """Return a list of sensor summary strings for the LLM prompt."""
        summaries: List[str] = []

        if bundle.hsi_signature is not None:
            summaries.append(
                self.describe_hsi(bundle.hsi_signature, bundle.hsi_wavelengths_nm)
            )
        if bundle.pid_ppm is not None:
            summaries.append(self.describe_pid(bundle.pid_ppm))
        if bundle.raman_shift_cm is not None:
            summaries.append(
                self.describe_raman(bundle.raman_shift_cm, bundle.raman_intensity)
            )
        if bundle.ims_score is not None:
            summaries.append(self.describe_ims(bundle.ims_score))
        if bundle.lwir_peak_um is not None:
            summaries.append(self.describe_lwir(bundle.lwir_peak_um))
        if bundle.gas_ppm is not None:
            summaries.append(self.describe_gas(bundle.gas_ppm))
        if bundle.thermal_mC is not None:
            summaries.append(
                self.describe_thermal(bundle.thermal_mC, bundle.battery_flux_uA)
            )

        for key, val in bundle.metadata.items():
            summaries.append(f"Additional sensor [{key}]: {val}.")

        return summaries


# ---------------------------------------------------------------------------
# Local SAM fallback
# ---------------------------------------------------------------------------


class LocalSAMFusion:
    """
    Pure-Python spectral matching fallback.

    When no LLM is available, compares the observed HSI signature against the
    built-in SignatureLibrary using the Spectral Angle Mapper (SAM).  The
    best-matching compound name and cosine similarity score are returned.

    If no HSI signature is present, falls back to heuristic rules based on
    available scalar sensor readings (PID, Raman, IMS, LWIR, gas).
    """

    def __init__(self, library: Optional[SignatureLibrary] = None) -> None:
        self.library = library or SignatureLibrary()

    def _sam_best_match(
        self, observed: np.ndarray
    ) -> Tuple[str, float]:
        """Return (best_compound_name, similarity) using SAM."""
        best_name = "unknown"
        best_sim = -1.0

        for chem in self.library.chemicals.values():
            if not chem.uv_vis_peaks:
                continue
            # Build a reference vector at the same resolution as observed
            ref = np.zeros_like(observed)
            n = len(observed)
            for wl_idx, (wl_nm, intensity) in enumerate(chem.uv_vis_peaks.items()):
                idx = min(int(wl_idx * n / max(len(chem.uv_vis_peaks), 1)), n - 1)
                ref[idx] = intensity

            if np.linalg.norm(ref) == 0:
                continue
            sim = spectral_angle_mapper(observed, ref)
            if sim > best_sim:
                best_sim = sim
                best_name = chem.name

        return best_name, float(best_sim)

    def _heuristic_identify(self, bundle: SensorBundle, target: str) -> Tuple[str, float]:
        """Rule-based identification when HSI is absent."""
        scores: Dict[str, float] = {}

        if bundle.pid_ppm is not None and bundle.pid_ppm > 25:
            scores["Ammonia"] = scores.get("Ammonia", 0) + 0.4
        if bundle.raman_shift_cm is not None:
            if 900 < bundle.raman_shift_cm < 1100:
                scores["Ammonia"] = scores.get("Ammonia", 0) + 0.3
            if 1300 < bundle.raman_shift_cm < 1500:
                scores["Ammonium Nitrate"] = scores.get("Ammonium Nitrate", 0) + 0.3
        if bundle.ims_score is not None and bundle.ims_score > 0.6:
            scores[target] = scores.get(target, 0) + 0.5
        if bundle.lwir_peak_um is not None and 9 < bundle.lwir_peak_um < 11:
            scores["Ammonia"] = scores.get("Ammonia", 0) + 0.3
        if bundle.gas_ppm is not None and bundle.gas_ppm > 25:
            scores["Ammonia"] = scores.get("Ammonia", 0) + 0.3

        if not scores:
            return "unknown", 0.0
        best = max(scores, key=lambda k: scores[k])
        return best, min(scores[best], 1.0)

    def fuse(
        self, bundle: SensorBundle, target_chemical: str
    ) -> ChemicalIdentification:
        """Run local SAM / heuristic fusion and return an identification."""
        if bundle.hsi_signature is not None:
            compound, confidence = self._sam_best_match(bundle.hsi_signature)
            reasoning = (
                f"Local SAM spectral matching against built-in chemical library. "
                f"Best cosine similarity: {confidence:.4f}. "
                f"Filtering for target: '{target_chemical}'."
            )
        else:
            compound, confidence = self._heuristic_identify(bundle, target_chemical)
            reasoning = (
                f"Heuristic rule-based fusion (no HSI data). "
                f"Target filter: '{target_chemical}'. "
                f"Scalar sensor votes produced confidence {confidence:.4f}."
            )

        is_hazardous, hazard_level = self._hazard_assessment(bundle, compound)

        return ChemicalIdentification(
            target_chemical=target_chemical,
            identified_compound=compound,
            confidence=confidence,
            is_hazardous=is_hazardous,
            hazard_level=hazard_level,
            reasoning=reasoning,
            backend_used="sam_fallback",
        )

    @staticmethod
    def _hazard_assessment(
        bundle: SensorBundle, compound: str
    ) -> Tuple[bool, str]:
        """Simple hazard level determination."""
        hazardous_compounds = {
            "Ammonia", "Ammonium Nitrate", "Methane", "Benzene",
            "Toluene", "Acetone", "Ethanol",
        }
        is_hazardous = compound in hazardous_compounds

        ppm = bundle.pid_ppm or bundle.gas_ppm or 0.0
        if ppm > 100 or (bundle.ims_score or 0) > 0.8:
            level = "critical"
        elif ppm > 50 or (bundle.ims_score or 0) > 0.6:
            level = "high"
        elif ppm > 25 or is_hazardous:
            level = "medium"
        elif is_hazardous:
            level = "low"
        else:
            level = "none"

        return is_hazardous, level


# ---------------------------------------------------------------------------
# LLM Backend
# ---------------------------------------------------------------------------


class LLMBackend:
    """
    Sends structured sensor reports to a local or remote LLM endpoint.

    Supported endpoints
    ───────────────────
    • Ollama (Gemma 2, LLaMA 3, Mistral …): http://localhost:11434/api/generate
    • OpenAI-compatible (GPT-4o, local vLLM, LM Studio): any /v1/chat/completions URL

    Configuration via environment variables
    ────────────────────────────────────────
    FUSION_LLM_URL    — base URL of the LLM server
                        default: http://localhost:11434/api/generate  (Ollama)
    FUSION_LLM_MODEL  — model name to use
                        default: gemma2  (for Ollama; use gpt-4o for OpenAI)
    FUSION_LLM_KEY    — API key (not required for Ollama or local servers)
    FUSION_LLM_TIMEOUT— HTTP timeout in seconds, default: 30
    """

    DEFAULT_OLLAMA_URL = "http://localhost:11434/api/generate"
    DEFAULT_MODEL = "gemma2"
    SYSTEM_PROMPT = (
        "You are a chemical analysis expert assistant. "
        "You will be given structured sensor data from a multi-sensor chemical detection "
        "system. Analyse the data, identify the most likely compound(s) present, "
        "and assess any safety hazards. "
        "Respond ONLY in valid JSON with the following keys: "
        "identified_compound (string), confidence (0.0-1.0), is_hazardous (bool), "
        "hazard_level (none/low/medium/high/critical), reasoning (string)."
    )

    def __init__(self) -> None:
        self.url = os.environ.get("FUSION_LLM_URL", self.DEFAULT_OLLAMA_URL)
        self.model = os.environ.get("FUSION_LLM_MODEL", self.DEFAULT_MODEL)
        self.api_key = os.environ.get("FUSION_LLM_KEY", "")
        self.timeout = int(os.environ.get("FUSION_LLM_TIMEOUT", "30"))
        self._is_openai_compat = "chat/completions" in self.url

    def _build_ollama_payload(self, prompt: str) -> bytes:
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "format": "json",
        }
        return json.dumps(payload).encode()

    def _build_openai_payload(self, prompt: str) -> bytes:
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": self.SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            "response_format": {"type": "json_object"},
        }
        return json.dumps(payload).encode()

    def _parse_response(self, raw: bytes) -> Dict[str, Any]:
        data = json.loads(raw.decode())
        if self._is_openai_compat:
            text = data["choices"][0]["message"]["content"]
        else:
            text = data.get("response", "{}")
        return json.loads(text)

    def query(self, prompt: str) -> Optional[Dict[str, Any]]:
        """
        Send ``prompt`` to the LLM endpoint and return the parsed JSON dict,
        or ``None`` if the request fails (network error, timeout, parse error).
        """
        payload = (
            self._build_openai_payload(prompt)
            if self._is_openai_compat
            else self._build_ollama_payload(prompt)
        )
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        req = urllib.request.Request(
            self.url, data=payload, headers=headers, method="POST"
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                return self._parse_response(resp.read())
        except (urllib.error.URLError, json.JSONDecodeError, KeyError):
            return None


# ---------------------------------------------------------------------------
# Main SensorFusionLLM class
# ---------------------------------------------------------------------------


class SensorFusionLLM:
    """
    Late Fusion engine that combines multi-sensor text summaries and reasons
    over them using a local or remote LLM, with a local SAM fallback.

    Usage
    ─────
    ::

        from analyzer.fusion import SensorFusionLLM, SensorBundle

        fusion = SensorFusionLLM()

        bundle = SensorBundle(
            pid_ppm=32.0,
            raman_shift_cm=966.0,
            gas_ppm=30.0,
        )

        result = fusion.analyze(bundle, target_chemical="Ammonia")
        print(result.identified_compound, result.hazard_level)

    The LLM backend is tried first.  If unavailable (Ollama not running, no
    internet, timeout), the local SAM + heuristic fallback is used automatically.
    """

    def __init__(
        self,
        library: Optional[SignatureLibrary] = None,
        use_llm: bool = True,
    ) -> None:
        self.encoder = SensorTextEncoder()
        self.sam_fallback = LocalSAMFusion(library=library)
        self.llm = LLMBackend() if use_llm else None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_sensor_summaries(self, bundle: SensorBundle) -> List[str]:
        """Return a list of natural-language sensor summaries."""
        return self.encoder.encode(bundle)

    def build_prompt(
        self, summaries: List[str], target_chemical: str
    ) -> str:
        """Format sensor summaries as a structured LLM prompt."""
        report = "\n".join(f"  • {s}" for s in summaries)
        return (
            f"--- SENSOR DATA REPORT ---\n"
            f"FILTER TARGET: {target_chemical}\n"
            f"SENSOR SUMMARIES:\n{report}\n"
            f"\n"
            f"Based on the above multi-sensor report, identify the chemical compound "
            f"present. Filter specifically for '{target_chemical}' and related "
            f"compounds. Assess whether levels are hazardous (threshold: >25 ppm "
            f"for ammonia-class compounds). "
            f"Respond in valid JSON only."
        )

    def analyze(
        self,
        bundle: SensorBundle,
        target_chemical: str = "Ammonia",
    ) -> ChemicalIdentification:
        """
        Run the full Late Fusion pipeline.

        1. Encode all sensor readings as text summaries.
        2. Build a structured LLM prompt.
        3. Query the LLM backend (Gemma 2 / GPT-4 / Ollama).
        4. If LLM is unavailable, fall back to local SAM matching.

        Parameters
        ──────────
        bundle           : SensorBundle — all active sensor readings
        target_chemical  : str — the compound to filter for

        Returns
        ───────
        ChemicalIdentification with compound name, confidence, hazard level,
        reasoning, and which backend was used.
        """
        summaries = self.get_sensor_summaries(bundle)

        llm_result: Optional[Dict[str, Any]] = None
        if self.llm is not None and summaries:
            prompt = self.build_prompt(summaries, target_chemical)
            llm_result = self.llm.query(prompt)

        if llm_result is not None:
            is_haz = bool(llm_result.get("is_hazardous", False))
            return ChemicalIdentification(
                target_chemical=target_chemical,
                identified_compound=str(
                    llm_result.get("identified_compound", "unknown")
                ),
                confidence=float(llm_result.get("confidence", 0.0)),
                is_hazardous=is_haz,
                hazard_level=str(llm_result.get("hazard_level", "none")),
                reasoning=str(llm_result.get("reasoning", "")),
                backend_used="llm",
                sensor_summaries=summaries,
            )

        # ── Local SAM / heuristic fallback ──────────────────────────────
        result = self.sam_fallback.fuse(bundle, target_chemical)
        result.sensor_summaries = summaries
        return result

    def analyze_ree(
        self,
        element_name: str,
        spectral_signature: np.ndarray,
        wavelengths_nm: Optional[np.ndarray] = None,
        thermal_mC: Optional[int] = None,
        flux_uA: Optional[int] = None,
    ) -> ChemicalIdentification:
        """
        Convenience wrapper for REE (Rare Earth Element) identification.

        Packages the spectral signature (from :mod:`analyzer.rare_earth` or
        :mod:`analyzer.ree_elements`) into a SensorBundle and runs the full
        fusion pipeline targeting the given element.
        """
        bundle = SensorBundle(
            hsi_signature=spectral_signature,
            hsi_wavelengths_nm=wavelengths_nm,
            thermal_mC=thermal_mC,
            battery_flux_uA=flux_uA,
        )
        return self.analyze(bundle, target_chemical=element_name)
