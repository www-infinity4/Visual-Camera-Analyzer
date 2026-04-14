"""Tests for analyzer/fusion.py"""
import numpy as np
import pytest

from analyzer.fusion import (
    ChemicalIdentification,
    LocalSAMFusion,
    SensorBundle,
    SensorFusionLLM,
    SensorTextEncoder,
)


class TestSensorTextEncoder:
    def setup_method(self):
        self.enc = SensorTextEncoder()

    def test_describe_hsi_returns_string(self):
        sig = np.array([0.8, 0.5, 0.3, 0.9, 0.6])
        wl  = np.array([400.0, 500.0, 600.0, 700.0, 800.0])
        out = self.enc.describe_hsi(sig, wl)
        assert isinstance(out, str)
        assert "nm" in out

    def test_describe_hsi_no_wavelengths(self):
        sig = np.array([0.8, 0.5, 0.3])
        out = self.enc.describe_hsi(sig)
        assert "reflectance" in out.lower()

    def test_describe_pid_high(self):
        out = self.enc.describe_pid(50.0)
        assert "HIGH" in out

    def test_describe_pid_normal(self):
        out = self.enc.describe_pid(5.0)
        assert "normal" in out

    def test_describe_raman(self):
        out = self.enc.describe_raman(1008.0, 0.75)
        assert "1008.0" in out
        assert "0.750" in out

    def test_describe_ims_positive(self):
        out = self.enc.describe_ims(0.85)
        assert "POSITIVE" in out

    def test_describe_ims_negative(self):
        out = self.enc.describe_ims(0.3)
        assert "below threshold" in out

    def test_describe_lwir(self):
        out = self.enc.describe_lwir(10.2)
        assert "10.20" in out

    def test_describe_gas(self):
        out = self.enc.describe_gas(30.0)
        assert "HAZARDOUS" in out

    def test_describe_thermal(self):
        out = self.enc.describe_thermal(45_000, flux_uA=200_000)
        assert "45.0" in out
        assert "200000" in out

    def test_encode_full_bundle(self):
        bundle = SensorBundle(
            pid_ppm=32.0,
            raman_shift_cm=966.0,
            gas_ppm=28.0,
            thermal_mC=52_000,
            battery_flux_uA=300_000,
        )
        summaries = self.enc.encode(bundle)
        assert len(summaries) == 4  # pid, raman, gas, thermal(+flux)
        assert all(isinstance(s, str) for s in summaries)

    def test_encode_empty_bundle_returns_empty(self):
        bundle = SensorBundle()
        assert self.enc.encode(bundle) == []

    def test_encode_metadata(self):
        bundle = SensorBundle(metadata={"custom_key": "hello"})
        summaries = self.enc.encode(bundle)
        assert any("custom_key" in s for s in summaries)


class TestLocalSAMFusion:
    def setup_method(self):
        self.sam = LocalSAMFusion()

    def test_fuse_with_hsi_returns_identification(self):
        sig = np.random.rand(50)
        bundle = SensorBundle(hsi_signature=sig)
        result = self.sam.fuse(bundle, "Ammonia")
        assert isinstance(result, ChemicalIdentification)
        assert result.backend_used == "sam_fallback"

    def test_fuse_heuristic_high_ppm(self):
        bundle = SensorBundle(pid_ppm=80.0, gas_ppm=60.0)
        result = self.sam.fuse(bundle, "Ammonia")
        assert result.confidence > 0
        assert result.hazard_level in {"medium", "high", "critical"}

    def test_fuse_heuristic_safe(self):
        bundle = SensorBundle(pid_ppm=2.0)
        result = self.sam.fuse(bundle, "Ammonia")
        assert result.confidence >= 0
        assert result.hazard_level in {"none", "low"}

    def test_fuse_empty_bundle(self):
        result = self.sam.fuse(SensorBundle(), "Ammonia")
        assert result.identified_compound == "unknown"
        assert result.confidence == 0.0

    def test_hazard_critical_threshold(self):
        bundle = SensorBundle(pid_ppm=150.0, ims_score=0.9)
        result = self.sam.fuse(bundle, "Ammonia")
        assert result.hazard_level == "critical"

    def test_is_hazardous_known_compound(self):
        is_haz, _ = LocalSAMFusion._hazard_assessment(SensorBundle(), "Ammonia")
        assert is_haz is True

    def test_is_not_hazardous_unknown(self):
        is_haz, _ = LocalSAMFusion._hazard_assessment(SensorBundle(), "Nitrogen")
        assert is_haz is False


class TestSensorFusionLLM:
    def setup_method(self):
        # Disable LLM backend for unit tests (no Ollama available in CI)
        self.fusion = SensorFusionLLM(use_llm=False)

    def test_analyze_returns_identification(self):
        bundle = SensorBundle(pid_ppm=30.0, gas_ppm=25.0)
        result = self.fusion.analyze(bundle, target_chemical="Ammonia")
        assert isinstance(result, ChemicalIdentification)

    def test_analyze_populates_sensor_summaries(self):
        bundle = SensorBundle(pid_ppm=30.0, raman_shift_cm=966.0)
        result = self.fusion.analyze(bundle)
        assert len(result.sensor_summaries) == 2

    def test_analyze_ree_neodymium(self):
        sig = np.random.rand(100)
        wl  = np.linspace(400, 2500, 100)
        result = self.fusion.analyze_ree("Neodymium", sig, wl, thermal_mC=55_000, flux_uA=400_000)
        assert result.target_chemical == "Neodymium"
        assert isinstance(result.confidence, float)

    def test_build_prompt_contains_target(self):
        summaries = ["PID sensor reports 30.0 ppm."]
        prompt = self.fusion.build_prompt(summaries, "Methane")
        assert "Methane" in prompt
        assert "30.0 ppm" in prompt

    def test_to_dict_serialisable(self):
        bundle = SensorBundle(gas_ppm=10.0)
        result = self.fusion.analyze(bundle)
        d = result.to_dict()
        import json
        json.dumps(d)   # must not raise
        assert "identified_compound" in d
        assert "confidence" in d
        assert "hazard_level" in d

    def test_analyze_no_sensors_unknown(self):
        result = self.fusion.analyze(SensorBundle(), "Ammonia")
        assert result.identified_compound == "unknown"
        assert result.confidence == 0.0
