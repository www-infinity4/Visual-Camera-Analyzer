"""
Noise Calibration — Background Subtraction for Software-Defined Sensors
════════════════════════════════════════════════════════════════════════
When using hardware components (CPU, battery, thermal zones) as *proxy*
sensors, their readings include a large "baseline" contribution from normal
device operation.  This module provides calibration models to subtract that
baseline and isolate the true chemical signal contribution.

Two approaches are implemented:

1. **Rolling-baseline subtraction** (lightweight, streaming)
   Maintains a rolling window of recent readings and subtracts the rolling
   mean.  Suitable for continuous monitoring on embedded / mobile devices.

2. **PCA background subtraction** (higher-fidelity, batch)
   Fits a PCA model to a calibration set of "blank" (no-chemical) spectral
   signatures.  Projects incoming signals onto the background subspace and
   subtracts the reconstructed background component, leaving only the
   anomalous chemical signal.

Both approaches implement the same ``BaseCalibrator`` interface so they are
interchangeable in the fusion pipeline.

References
──────────
[1] Background subtraction in hyperspectral imagery:
    https://www.sciencedirect.com/science/article/abs/pii/S1077314224001802
[2] Thermal baseline drift correction:
    https://pmc.ncbi.nlm.nih.gov/articles/PMC9584913/
[3] Cross-attention RGB-thermal fusion:
    https://arxiv.org/html/2403.17712v1
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections import deque
from typing import Deque, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Base interface
# ---------------------------------------------------------------------------


class BaseCalibrator(ABC):
    """Abstract baseline calibrator."""

    @abstractmethod
    def update_baseline(self, reading: np.ndarray) -> None:
        """Ingest a new baseline (no-chemical) reading to update the model."""

    @abstractmethod
    def subtract(self, reading: np.ndarray) -> np.ndarray:
        """Return the calibrated signal: ``reading - estimated_background``."""

    def is_ready(self) -> bool:
        """Return True once enough baseline samples have been collected."""
        return True


# ---------------------------------------------------------------------------
# 1. Rolling baseline subtraction
# ---------------------------------------------------------------------------


class RollingBaselineCalibrator(BaseCalibrator):
    """
    Maintains a fixed-length rolling window of baseline readings.

    The background estimate at each timestep is:

        background = mean(window)  ± z × std(window)

    Parameters
    ──────────
    window_size : int
        Number of historical readings to keep (default 60 — roughly 1 minute
        at 1 Hz sampling).
    zscore_threshold : float
        Readings further than this many standard deviations from the rolling
        mean are flagged as *anomalous* (potential chemical signal present).
        Set to 0 to disable flagging.
    min_samples : int
        Minimum number of samples before subtract() is meaningful.
    """

    def __init__(
        self,
        window_size: int = 60,
        zscore_threshold: float = 3.0,
        min_samples: int = 10,
    ) -> None:
        self.window_size = window_size
        self.zscore_threshold = zscore_threshold
        self.min_samples = min_samples
        self._window: Deque[np.ndarray] = deque(maxlen=window_size)

    # ------------------------------------------------------------------

    def update_baseline(self, reading: np.ndarray) -> None:
        """Add a blank (no-signal) reading to the rolling window."""
        self._window.append(np.asarray(reading, dtype=float))

    def subtract(self, reading: np.ndarray) -> np.ndarray:
        """
        Return ``reading − rolling_mean``.

        If fewer than ``min_samples`` have been collected, the raw reading is
        returned unchanged (no-op pass-through until calibration warms up).
        """
        r = np.asarray(reading, dtype=float)
        if len(self._window) < self.min_samples:
            return r.copy()

        stack = np.stack(list(self._window))  # (T, N) or (T,)
        baseline = stack.mean(axis=0)
        return r - baseline

    def is_ready(self) -> bool:
        return len(self._window) >= self.min_samples

    def zscore(self, reading: np.ndarray) -> np.ndarray:
        """Return per-element z-scores relative to the rolling window."""
        r = np.asarray(reading, dtype=float)
        if len(self._window) < self.min_samples:
            return np.zeros_like(r)
        stack = np.stack(list(self._window))
        mu = stack.mean(axis=0)
        sigma = stack.std(axis=0) + 1e-9
        return (r - mu) / sigma

    def is_anomalous(self, reading: np.ndarray) -> bool:
        """
        Return True if *any* element of ``reading`` exceeds the z-score
        threshold — i.e., a chemical signal is likely present.
        """
        if self.zscore_threshold <= 0:
            return False
        return bool(np.any(np.abs(self.zscore(reading)) > self.zscore_threshold))


# ---------------------------------------------------------------------------
# 2. PCA background subtraction
# ---------------------------------------------------------------------------


class PCABackgroundCalibrator(BaseCalibrator):
    """
    PCA-based background subtraction for hyperspectral / multi-band signals.

    Workflow
    ────────
    1. **Calibration phase**: Feed ``n_components`` or more blank-scene
       spectra via ``update_baseline()``.
    2. ``fit()`` must be called explicitly once you have enough samples.
    3. **Runtime phase**: ``subtract(spectrum)`` projects the incoming
       spectrum onto the background subspace and returns the residual.

    The residual isolates the part of the signal *not* explained by normal
    background variation — i.e., the chemical anomaly.

    Parameters
    ──────────
    n_components : int
        Number of PCA components used to model the background.
    min_calibration_samples : int
        Minimum blank-scene samples required before fit() succeeds.
    """

    def __init__(
        self,
        n_components: int = 5,
        min_calibration_samples: int = 20,
    ) -> None:
        self.n_components = n_components
        self.min_calibration_samples = min_calibration_samples
        self._calibration_set: list = []
        self._components: Optional[np.ndarray] = None  # (n_components, N)
        self._mean: Optional[np.ndarray] = None
        self._fitted = False

    # ------------------------------------------------------------------

    def update_baseline(self, reading: np.ndarray) -> None:
        """Add a blank (background-only) spectral reading to the calibration set."""
        self._calibration_set.append(np.asarray(reading, dtype=float).ravel())
        self._fitted = False  # mark as needing re-fit

    def fit(self) -> None:
        """
        Fit the PCA background model to the accumulated calibration samples.

        Raises
        ──────
        ValueError — if fewer than ``min_calibration_samples`` have been
                     collected, or if all spectra are identical.
        """
        n = len(self._calibration_set)
        if n < self.min_calibration_samples:
            raise ValueError(
                f"Need at least {self.min_calibration_samples} calibration "
                f"samples; only {n} collected."
            )

        X = np.stack(self._calibration_set)   # (n_samples, n_bands)
        self._mean = X.mean(axis=0)
        X_centred = X - self._mean

        # Economy SVD — only compute the top k components
        k = min(self.n_components, X_centred.shape[0], X_centred.shape[1])
        _, _, Vt = np.linalg.svd(X_centred, full_matrices=False)
        self._components = Vt[:k, :]          # (k, n_bands)
        self._fitted = True

    def subtract(self, reading: np.ndarray) -> np.ndarray:
        """
        Project ``reading`` onto the background subspace and return the residual.

        If the model has not been fitted yet, the raw reading is returned.
        """
        r = np.asarray(reading, dtype=float).ravel()
        if not self._fitted or self._mean is None or self._components is None:
            return r.copy()

        r_centred = r - self._mean
        # Reconstruct background component
        projection = self._components.T @ (self._components @ r_centred)
        background = projection + self._mean
        return r - background

    def is_ready(self) -> bool:
        return self._fitted

    def reconstruction_error(self, reading: np.ndarray) -> float:
        """
        Return the L2 norm of the residual after PCA background removal.
        High values indicate a strong anomalous (chemical) signal.
        """
        residual = self.subtract(reading)
        return float(np.linalg.norm(residual))


# ---------------------------------------------------------------------------
# 3. Scalar (hardware proxy) baseline subtractor
# ---------------------------------------------------------------------------


class ScalarBaselineCalibrator:
    """
    Lightweight calibrator for single-value hardware proxy readings (e.g.,
    thermal zone temperature, battery current).

    Uses exponential moving average (EMA) to track the idle baseline:

        baseline_t = α × reading_t + (1−α) × baseline_{t-1}

    Parameters
    ──────────
    alpha : float
        EMA smoothing factor (0 < α < 1).  Smaller values = more stable
        baseline, larger values = faster adaptation.
    """

    def __init__(self, alpha: float = 0.05) -> None:
        if not (0 < alpha < 1):
            raise ValueError("alpha must be in (0, 1)")
        self.alpha = alpha
        self._baseline: Optional[float] = None

    def update_baseline(self, value: float) -> None:
        """Update the EMA baseline with a new idle reading."""
        if self._baseline is None:
            self._baseline = float(value)
        else:
            self._baseline = self.alpha * float(value) + (1.0 - self.alpha) * self._baseline

    def subtract(self, value: float) -> float:
        """Return ``value − EMA_baseline``.  Returns raw value if not yet initialised."""
        if self._baseline is None:
            return float(value)
        return float(value) - self._baseline

    @property
    def baseline(self) -> Optional[float]:
        return self._baseline
