"""Tests for analyzer/optical_imaging.py"""
import numpy as np
import pytest

from analyzer.optical_imaging import (
    OGIFilter,
    RGBThermalCrossAttention,
    SchlierenProcessor,
)


def _blank_rgb(h=64, w=64):
    return np.zeros((h, w, 3), dtype=np.uint8)


def _bright_spot(h=64, w=64, cx=32, cy=32, radius=8):
    frame = _blank_rgb(h, w)
    for r in range(h):
        for c in range(w):
            if (r - cy) ** 2 + (c - cx) ** 2 < radius ** 2:
                frame[r, c] = [200, 200, 200]
    return frame


class TestOGIFilter:
    def setup_method(self):
        self.ogi = OGIFilter(target_chemical="ammonia", detection_threshold=0.25, blend_alpha=0.55)

    def test_plume_mask_shape(self):
        band = np.random.rand(64, 64).astype(np.float32)
        mask = self.ogi.plume_mask(band)
        assert mask.shape == (64, 64)
        assert mask.dtype == np.float32

    def test_plume_mask_range(self):
        band = np.random.rand(32, 32).astype(np.float32)
        mask = self.ogi.plume_mask(band)
        assert mask.min() >= 0.0
        assert mask.max() <= 1.0

    def test_plume_mask_zeros_below_threshold(self):
        band = np.zeros((32, 32), dtype=np.float32)
        mask = self.ogi.plume_mask(band)
        assert mask.max() < 0.05   # near-zero after blurring

    def test_apply_output_shape(self):
        frame = _blank_rgb()
        band  = np.random.rand(64, 64).astype(np.float32)
        out   = self.ogi.apply(frame, band)
        assert out.shape == frame.shape
        assert out.dtype == np.uint8

    def test_apply_no_change_on_zero_band(self):
        frame = _bright_spot()
        band  = np.zeros((64, 64), dtype=np.float32)
        out   = self.ogi.apply(frame, band)
        # With zero gas signal, output should be very close to input
        diff = np.abs(out.astype(int) - frame.astype(int))
        assert diff.mean() < 5.0

    def test_palette_fallback_for_unknown_chemical(self):
        ogi = OGIFilter(target_chemical="unknown_xyz")
        r, g, b = ogi._plume_color()
        assert 0 <= r <= 255
        assert 0 <= g <= 255
        assert 0 <= b <= 255

    def test_apply_to_hsi_returns_correct_shapes(self):
        h, w, bands = 32, 32, 20
        frame   = _blank_rgb(h, w)
        hsi     = np.random.rand(h, w, bands).astype(np.float32)
        wl      = np.linspace(1400, 1620, bands)
        composite, mask = self.ogi.apply_to_hsi(frame, hsi, wl, target_wavelength_nm=1510)
        assert composite.shape == (h, w, 3)
        assert mask.shape      == (h, w)

    def test_apply_to_hsi_missing_band(self):
        h, w, bands = 16, 16, 5
        frame = _blank_rgb(h, w)
        hsi   = np.random.rand(h, w, bands).astype(np.float32)
        wl    = np.linspace(400, 700, bands)
        # Target wavelength not in range → should return unchanged frame
        out, mask = self.ogi.apply_to_hsi(frame, hsi, wl, target_wavelength_nm=1510)
        np.testing.assert_array_equal(out, frame)
        assert mask.max() == 0


class TestSchlierenProcessor:
    def setup_method(self):
        self.proc = SchlierenProcessor(sensitivity=4.0)

    def test_compute_output_shape(self):
        frame = _blank_rgb()
        out   = self.proc.compute(frame)
        assert out.shape == (64, 64, 3)
        assert out.dtype == np.uint8

    def test_compute_changes_after_motion(self):
        frame1 = _blank_rgb()
        frame2 = _bright_spot()          # different frame
        self.proc.compute(frame1)
        out = self.proc.compute(frame2)
        # Should show some non-zero gradient
        assert out.max() > 0

    def test_compute_mask_shape(self):
        frame = _bright_spot()
        mask  = self.proc.compute_mask(frame, threshold=0.05)
        assert mask.shape == (64, 64)
        assert mask.dtype == np.uint8

    def test_first_frame_uses_spatial_only(self):
        proc  = SchlierenProcessor()
        frame = _bright_spot()
        out   = proc.compute(frame)
        assert out.shape == (64, 64, 3)

    def test_greyscale_input(self):
        gray  = np.random.randint(0, 255, (32, 32), dtype=np.uint8)
        out   = self.proc.compute(gray)
        assert out.shape == (32, 32, 3)


class TestRGBThermalCrossAttention:
    def test_fuse_output_shape(self):
        ca  = RGBThermalCrossAttention(thermal_weight=0.6)
        rgb = np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8)
        thm = np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8)
        out = ca.fuse(rgb, thm)
        assert out.shape == (32, 32, 3)
        assert out.dtype == np.uint8

    def test_fuse_with_anomaly_mask(self):
        ca   = RGBThermalCrossAttention()
        rgb  = np.zeros((16, 16, 3), dtype=np.uint8)
        thm  = np.full((16, 16, 3), 200, dtype=np.uint8)
        mask = np.ones((16, 16), dtype=np.float32)   # full thermal weight
        out  = ca.fuse(rgb, thm, anomaly_mask=mask)
        # With full mask, output should be close to thermal frame
        assert out.mean() > 50

    def test_fuse_zero_mask_returns_rgb(self):
        ca   = RGBThermalCrossAttention(thermal_weight=0.6)
        rgb  = np.full((16, 16, 3), 100, dtype=np.uint8)
        thm  = np.full((16, 16, 3), 200, dtype=np.uint8)
        mask = np.zeros((16, 16), dtype=np.float32)
        out  = ca.fuse(rgb, thm, anomaly_mask=mask)
        np.testing.assert_array_equal(out, rgb)
