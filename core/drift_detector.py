"""
drift_detector.py — SCS ψ Semantic Drift Detector.

Monitors ψ (world semantic vector) stability using a sliding window
of Φ (cosine similarity) scores. When the coefficient of variation (CV)
exceeds a threshold, the detector flags semantic drift.

Drift detection is the trigger for δ-mem S-update damping:
  - Drift detected → reduce D-Core S beta to 0.2× (dampen noisy ingest)
  - Drift subsides → restore full S learning rate

v1.3 (2026-06-11): Exponential decay weighting for CV computation.
  Recovery from drift is now faster because old low-Φ values decay
  exponentially, letting recent high-Φ values dominate the CV.

Author: Taiji OS Team
Version: v1.3 — exponential decay CV (2026-06-11)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class DriftDetector:
    """Sliding-window ψ drift detector using Φ-value CV with exponential decay.

    Tracks a moving window of Φ (consistency) scores and computes the
    weighted coefficient of variation (std/mean) with exponential decay
    weighting. Recent Φ values receive higher weight, allowing faster
    recovery from drift when Φ values return to normal.

    v1.3 (2026-06-11): Exponential decay weighting for CV computation.
        Old low-Φ values from a drift episode decay exponentially,
        letting recent high-Φ values dominate the CV during recovery.

    Attributes:
        window_size: Number of recent Φ values to track (default 20).
        cv_threshold: CV above which drift is flagged (default 0.30).
        decay: Exponential decay factor for CV weighting (default 0.80).
            Weight for sample i (0=oldest, n-1=newest) = decay^(n-1-i).
            Lower values = faster forgetting of old samples.
        min_samples_before_detect: Minimum samples before drift detection
            is allowed (default 5). Prevents false positives from
            early-stage CV instability.
        hysteresis_rounds: Consecutive above-threshold rounds needed
            to confirm drift entry (default 2). Acts like a Schmitt
            trigger to filter transient fluctuations.
        phi_history: Circular buffer of recent Φ values.
        write_idx: Current write position in the circular buffer.
        count: Number of valid entries in the buffer.
    """

    window_size: int = 20
    cv_threshold: float = 0.30
    decay: float = 0.55                     # exponential decay factor
    min_samples_before_detect: int = 5
    hysteresis_rounds: int = 2
    phi_history: np.ndarray = field(init=False)
    write_idx: int = 0
    count: int = 0
    _drifting_streak: int = 0          # consecutive above-threshold rounds
    _currently_drifting: bool = False   # confirmed drift state (latched)

    def __post_init__(self):
        self.phi_history = np.zeros(self.window_size, dtype=np.float64)

    # ── Push new observation ───────────────────────────────────────────

    def push(self, phi_value: float) -> None:
        """Record a new Φ value into the sliding window.

        Called after each Ψ-check in the consistency loop.

        Args:
            phi_value: The Φ (cosine similarity) score from the latest step.
        """
        self.phi_history[self.write_idx] = float(phi_value)
        self.write_idx = (self.write_idx + 1) % self.window_size
        if self.count < self.window_size:
            self.count += 1

    # ── Weighted statistics (v1.3: exponential decay) ─────────────────

    def _compute_weighted_stats(self) -> tuple[float, float, float]:
        """Compute exponentially weighted mean, std, and CV.

        Weights follow w[i] = decay^(n-1-i) / sum(decay^k) for sample i
        where i=0 is the oldest sample and i=n-1 is the newest.
        The newest sample always gets weight 1.0 (after normalization).

        The circular buffer is reconstructed into chronological order:
        window_chrono[0] = oldest, window_chrono[n-1] = newest.

        Returns:
            (weighted_mean, weighted_std, weighted_cv)
        """
        n = self.count
        # Reconstruct chronological ordering from circular buffer.
        # Before buffer fills: phi_history[0:n] is already in chronological order.
        # After buffer wraps: start at write_idx and wrap around.
        if n < self.window_size:
            window = self.phi_history[:n]
        else:
            indices = [(self.write_idx + i) % self.window_size for i in range(n)]
            window = self.phi_history[indices]  # chronological: [oldest, ..., newest]

        # Build weights: [decay^(n-1), decay^(n-2), ..., decay^1, decay^0]
        exponents = np.arange(n - 1, -1, -1, dtype=np.float64)
        raw_weights = np.power(self.decay, exponents)
        weights = raw_weights / raw_weights.sum()

        weighted_mean = float(np.average(window, weights=weights))

        if weighted_mean < 1e-8:
            return 0.0, 0.0, 0.0

        # Weighted std: sqrt(Σ w[i] * (x[i] - mean)^2)
        deviations = window - weighted_mean
        weighted_var = float(np.average(deviations ** 2, weights=weights))
        weighted_std = np.sqrt(weighted_var)

        weighted_cv = weighted_std / weighted_mean
        return weighted_mean, weighted_std, weighted_cv

    # ── Drift detection ────────────────────────────────────────────────

    def is_drifting(self) -> bool:
        """Check whether ψ is currently drifting.

        Drift detection is gated by two safeguards:
          1. min_samples_before_detect:  Don't even look until enough
             samples have accumulated (prevents early CV instability).
          2. hysteresis_rounds:  Need N consecutive above-threshold
             rounds to confirm drift entry (Schmitt trigger).

        CV computation uses exponential decay weighting (v1.3):
          Recent Φ values dominate the CV, so drift recovery is faster
          when Φ values return to normal levels.

        Exit from drift state is immediate when CV drops below threshold
        (no exit hysteresis — we want to resume learning quickly).

        Returns:
            True if drift is confirmed, False otherwise.
        """
        if self.count < self.min_samples_before_detect:
            self._drifting_streak = 0
            self._currently_drifting = False
            return False

        _, _, cv = self._compute_weighted_stats()

        is_over_threshold = cv > self.cv_threshold

        if is_over_threshold:
            self._drifting_streak += 1
            if self._drifting_streak >= self.hysteresis_rounds:
                self._currently_drifting = True
        else:
            # CV dropped below threshold → immediately exit drift
            self._drifting_streak = 0
            self._currently_drifting = False

        return self._currently_drifting

    # ── Statistics ─────────────────────────────────────────────────────

    @property
    def current_cv(self) -> float:
        """Current weighted coefficient of variation of the Φ window (v1.3: exponential decay)."""
        if self.count < 3:
            return 0.0
        _, _, cv = self._compute_weighted_stats()
        return float(cv)

    @property
    def mean_phi(self) -> float:
        """Weighted mean Φ value in the current window (v1.3: exponential decay)."""
        if self.count == 0:
            return 0.0
        mean, _, _ = self._compute_weighted_stats()
        return float(mean)

    def stats(self) -> dict:
        """Return diagnostic statistics about the drift detector state.

        Returns:
            dict with keys: window_size, count, current_cv, mean_phi,
            is_drifting, cv_threshold, drifting_streak, min_samples.
        """
        return {
            "window_size": self.window_size,
            "count": self.count,
            "current_cv": round(self.current_cv, 4),
            "mean_phi": round(self.mean_phi, 4),
            "is_drifting": self.is_drifting(),
            "cv_threshold": self.cv_threshold,
            "drifting_streak": self._drifting_streak,
            "min_samples_before_detect": self.min_samples_before_detect,
            "decay": self.decay,
        }

    # ── Reset ──────────────────────────────────────────────────────────

    def reset(self) -> None:
        """Reset the drift detector to initial state."""
        self.phi_history = np.zeros(self.window_size, dtype=np.float64)
        self.write_idx = 0
        self.count = 0
        self._drifting_streak = 0
        self._currently_drifting = False


# ────────────────────────────────────────────────────────────────────────────
# Pre-computed ψ-stability regions (for analysis / visualization)
# ────────────────────────────────────────────────────────────────────────────

def analyze_stability(phi_sequence: list[float]) -> dict:
    """Analyze a Φ sequence for stability regions.

    Segments the sequence into stable (CV < 0.15), transitional
    (0.15 ≤ CV < 0.30), and drifting (CV ≥ 0.30) periods.

    Args:
        phi_sequence: List of Φ values in chronological order.

    Returns:
        dict with stability analysis.
    """
    detector = DriftDetector(window_size=min(len(phi_sequence), 20))
    regions = []
    current_region = None

    for i, phi in enumerate(phi_sequence):
        detector.push(phi)
        is_drifting = detector.is_drifting()
        cv = detector.current_cv

        tag = "drifting" if is_drifting else ("stable" if cv < 0.15 else "transitional")

        if current_region is None or current_region["tag"] != tag:
            if current_region is not None:
                current_region["end"] = i - 1
                regions.append(current_region)
            current_region = {"tag": tag, "start": i, "end": None, "cv": round(cv, 4)}

    if current_region is not None:
        current_region["end"] = len(phi_sequence) - 1
        regions.append(current_region)

    return {
        "total_steps": len(phi_sequence),
        "regions": regions,
        "stable_ratio": sum(
            1 for r in regions if r["tag"] == "stable"
        ) / max(len(regions), 1),
    }
