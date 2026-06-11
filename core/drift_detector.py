"""
drift_detector.py — SCS ψ Semantic Drift Detector.

Monitors ψ (world semantic vector) stability using a sliding window
of Φ (cosine similarity) scores. When the coefficient of variation (CV)
exceeds a threshold, the detector flags semantic drift.

Drift detection is the trigger for δ-mem S-update damping:
  - Drift detected → reduce D-Core S beta to 0.2× (dampen noisy ingest)
  - Drift subsides → restore full S learning rate

v1.4 (2026-06-11): Adaptive decay — three-stage decay auto-tuning.
  Decay factor γ is automatically adjusted based on the current SCL stage:
    STABLE   γ=0.70  (slow forgetting, steady state)
    DRIFTING  γ=0.35  (fast forgetting, quick adaptation)
    RECOVERY  γ=0.55  (balanced, recover from drift)
  Stage transitions are tracked internally; no external API changes needed
  when adaptive=True.

v1.3 (2026-06-11): Exponential decay weighting for CV computation.
  Recovery from drift is now faster because old low-Φ values decay
  exponentially, letting recent high-Φ values dominate the CV.

Author: Taiji OS Team
Version: v1.4 — adaptive decay auto-tuning (2026-06-11)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class DriftDetector:
    """Sliding-window ψ drift detector using Φ-value CV with decay weighting.

    Tracks a moving window of Φ (consistency) scores and computes the
    weighted coefficient of variation (std/mean) with decay weighting.
    Recent Φ values receive higher weight, allowing faster recovery
    from drift when Φ values return to normal.

    v1.4 (2026-06-11): Adaptive decay auto-tuning.
        When adaptive=True, the decay factor is automatically adjusted
        based on the current SCL stage — no manual tuning needed.
        Stage transitions: STABLE ↔ DRIFTING → RECOVERY → STABLE.

    v1.3 (2026-06-11): Exponential decay weighting for CV computation.
        Old low-Φ values from a drift episode decay exponentially,
        letting recent high-Φ values dominate the CV during recovery.

    Attributes:
        window_size: Number of recent Φ values to track (default 20).
        cv_threshold: CV above which drift is flagged (default 0.30).
        decay: Base exponential decay factor (default 0.55).
            When adaptive=False, this is the fixed decay used.
            When adaptive=True, this serves as the RECOVERY-stage decay.
        adaptive: Enable three-stage adaptive decay (default False).
            STABLE=0.70, DRIFTING=0.35, RECOVERY=decay (0.55).
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
    decay: float = 0.55                     # base decay (RECOVERY when adaptive)
    adaptive: bool = False                   # enable three-stage adaptive decay
    min_samples_before_detect: int = 5
    hysteresis_rounds: int = 2
    phi_history: np.ndarray = field(init=False)
    write_idx: int = 0
    count: int = 0
    _drifting_streak: int = 0          # consecutive above-threshold rounds
    _currently_drifting: bool = False   # confirmed drift state (latched)
    _stage: str = "STABLE"             # STABLE | DRIFTING | RECOVERY
    _was_drifting: bool = False        # tracks prior drift for RECOVERY detection
    _recovery_streak: int = 0          # consecutive CV < 0.15 rounds in RECOVERY

    def __post_init__(self):
        self.phi_history = np.zeros(self.window_size, dtype=np.float64)

    # ── Adaptive decay (v1.4) ──────────────────────────────────────────

    def _get_decay(self) -> float:
        """Return the effective decay factor.

        In adaptive mode, decay depends on the current stage.
        In fixed mode, returns the configured decay value.
        """
        if not self.adaptive:
            return self.decay
        if self._stage == "DRIFTING":
            return 0.35
        elif self._stage == "RECOVERY":
            return self.decay  # use base decay (0.55)
        else:  # STABLE
            return 0.70

    def _update_stage(self) -> None:
        """Update the adaptive stage based on drift detection state.

        Stage transitions:
          STABLE → DRIFTING: when drift is confirmed (hysteresis satisfied)
          DRIFTING → RECOVERY: when CV drops below threshold (immediate exit)
          RECOVERY → STABLE: when CV < 0.15 for 2 consecutive rounds
            (RECOVERY exit hysteresis prevents premature return)
        """
        if not self.adaptive:
            return
        if self._currently_drifting:
            self._was_drifting = True
            self._stage = "DRIFTING"
            self._recovery_streak = 0
        elif self._was_drifting:
            self._stage = "RECOVERY"
            if self.current_cv < 0.15:
                self._recovery_streak += 1
            else:
                self._recovery_streak = 0
            if self._recovery_streak >= 2:
                self._was_drifting = False
                self._stage = "STABLE"
                self._recovery_streak = 0
        else:
            self._stage = "STABLE"
            self._recovery_streak = 0

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
        effective_decay = self._get_decay()
        exponents = np.arange(n - 1, -1, -1, dtype=np.float64)
        raw_weights = np.power(effective_decay, exponents)
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

        self._update_stage()
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
            "decay": self._get_decay(),
            "adaptive": self.adaptive,
            "stage": self._stage,
        }

    # ── Reset ──────────────────────────────────────────────────────────

    def reset(self) -> None:
        """Reset the drift detector to initial state."""
        self.phi_history = np.zeros(self.window_size, dtype=np.float64)
        self.write_idx = 0
        self.count = 0
        self._drifting_streak = 0
        self._currently_drifting = False
        self._stage = "STABLE"
        self._was_drifting = False
        self._recovery_streak = 0


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
