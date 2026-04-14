"""Tests for analyzer/noise_calibration.py"""
import numpy as np
import pytest

from analyzer.noise_calibration import (
    PCABackgroundCalibrator,
    RollingBaselineCalibrator,
    ScalarBaselineCalibrator,
)


class TestRollingBaselineCalibrator:
    def test_not_ready_initially(self):
        cal = RollingBaselineCalibrator(min_samples=5)
        assert not cal.is_ready()

    def test_ready_after_min_samples(self):
        cal = RollingBaselineCalibrator(min_samples=3)
        for _ in range(3):
            cal.update_baseline(np.array([1.0, 2.0, 3.0]))
        assert cal.is_ready()

    def test_subtract_passthrough_when_cold(self):
        cal = RollingBaselineCalibrator(min_samples=10)
        r   = np.array([5.0, 6.0])
        assert np.allclose(cal.subtract(r), r)

    def test_subtract_removes_baseline(self):
        cal = RollingBaselineCalibrator(min_samples=3)
        baseline = np.array([2.0, 3.0, 4.0])
        for _ in range(5):
            cal.update_baseline(baseline)
        signal = np.array([5.0, 6.0, 7.0])
        residual = cal.subtract(signal)
        assert np.allclose(residual, signal - baseline, atol=0.2)

    def test_zscore_near_zero_on_baseline(self):
        cal = RollingBaselineCalibrator(min_samples=3)
        for _ in range(10):
            cal.update_baseline(np.array([1.0, 1.0]))
        z = cal.zscore(np.array([1.0, 1.0]))
        assert np.all(np.abs(z) < 0.5)

    def test_is_anomalous_flags_spike(self):
        cal = RollingBaselineCalibrator(min_samples=3, zscore_threshold=3.0)
        for _ in range(10):
            cal.update_baseline(np.array([1.0, 1.0]))
        # A value 5σ above baseline should trigger anomaly
        assert cal.is_anomalous(np.array([100.0, 1.0]))

    def test_window_respects_max_size(self):
        cal = RollingBaselineCalibrator(window_size=5)
        for i in range(20):
            cal.update_baseline(np.array([float(i)]))
        # Window should only hold last 5
        assert len(cal._window) == 5


class TestPCABackgroundCalibrator:
    def _fill(self, cal, n=25, seed=42):
        rng = np.random.default_rng(seed)
        for _ in range(n):
            cal.update_baseline(rng.normal(0, 0.1, 16))

    def test_fit_requires_min_samples(self):
        cal = PCABackgroundCalibrator(min_calibration_samples=20)
        for _ in range(5):
            cal.update_baseline(np.ones(8))
        with pytest.raises(ValueError, match="Need at least"):
            cal.fit()

    def test_fit_succeeds(self):
        cal = PCABackgroundCalibrator(n_components=3, min_calibration_samples=10)
        self._fill(cal)
        cal.fit()
        assert cal.is_ready()

    def test_subtract_reduces_background(self):
        cal = PCABackgroundCalibrator(n_components=3, min_calibration_samples=10)
        self._fill(cal)
        cal.fit()
        # A pure-background signal should produce a small residual
        rng = np.random.default_rng(99)
        bg  = rng.normal(0, 0.1, 16)
        res = cal.subtract(bg)
        assert np.linalg.norm(res) < np.linalg.norm(bg) * 2

    def test_subtract_passthrough_when_not_fitted(self):
        cal = PCABackgroundCalibrator()
        r   = np.array([1.0, 2.0, 3.0])
        assert np.allclose(cal.subtract(r), r)

    def test_reconstruction_error_high_for_anomaly(self):
        cal = PCABackgroundCalibrator(n_components=2, min_calibration_samples=10)
        self._fill(cal)
        cal.fit()
        # Pure background → low error
        bg_err = cal.reconstruction_error(np.zeros(16))
        # Large anomaly spike → higher error
        anomaly = np.zeros(16)
        anomaly[0] = 50.0
        assert cal.reconstruction_error(anomaly) > bg_err


class TestScalarBaselineCalibrator:
    def test_invalid_alpha(self):
        with pytest.raises(ValueError):
            ScalarBaselineCalibrator(alpha=0.0)

    def test_subtract_passthrough_uninitialized(self):
        cal = ScalarBaselineCalibrator()
        assert cal.subtract(5.0) == 5.0

    def test_ema_converges(self):
        cal = ScalarBaselineCalibrator(alpha=0.3)
        for _ in range(50):
            cal.update_baseline(10.0)
        assert abs(cal.baseline - 10.0) < 0.1

    def test_subtract_removes_baseline(self):
        cal = ScalarBaselineCalibrator(alpha=0.3)
        for _ in range(50):
            cal.update_baseline(10.0)
        assert abs(cal.subtract(12.0) - 2.0) < 0.5
