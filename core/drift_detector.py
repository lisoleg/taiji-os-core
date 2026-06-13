"""
drift_detector.py — SCS ψ Semantic Drift Detector.

Monitors ψ (world semantic vector) stability using a sliding window
of Φ (cosine similarity) scores. When the coefficient of variation (CV)
exceeds a threshold, the detector flags semantic drift.

Drift detection is the trigger for δ-mem S-update damping:
  - Drift detected → reduce D-Core S beta to 0.2× (dampen noisy ingest)
  - Drift subsides → restore full S learning rate

v1.6 (2026-06-13): HyperParamAdapter — automatic adaptation of γ_max, γ_min, cv_mid
  based on multi-round CV history statistics. Tracks CV distribution over 200 rounds,
  adapts every 20 rounds using percentile-based heuristics. Eliminates need for
  manual tuning of the sigmoid hyperparameters.

v1.5 (2026-06-11): Continuous auto-tune decay — replaces hard-coded three-stage
  lookup with a smooth sigmoid-based formula.
    γ(CV, dCV/dt) = γ_max − Δγ × σ((CV−CV_mid)/T) × slope_factor(dCV/dt)
  This eliminates stage boundary discontinuities and adapts preemptively
  to CV trend direction (rising CV → lower γ, falling CV → higher γ).

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
Version: v1.6 — hyperparameter auto-adaptation (2026-06-13)
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

    v1.5 (2026-06-11): Continuous auto-tune decay.
        When auto_tune=True (recommended), the decay factor is a continuous
        sigmoid function of CV and its derivative — no hard stage boundaries.
        γ(CV, dCV/dt) = γ_max − Δγ × σ((CV−CV_mid)/T) × slope_factor(dCV/dt)

    v1.4 (2026-06-11): Adaptive three-stage decay.
        STABLE=0.70, DRIFTING=0.35, RECOVERY=base_decay (0.55).

    v1.3 (2026-06-11): Exponential decay weighting for CV computation.

    Attributes:
        window_size: Number of recent Φ values to track (default 20).
        cv_threshold: CV above which drift is flagged (default 0.30).
        decay: Base decay factor for CV weighting (default 0.55).
            When auto_tune=True, this serves as the fallback.
        adaptive: Enable adaptive decay mode (default True).
            When auto_tune=True, uses continuous sigmoid formula.
            When auto_tune=False, uses three-stage lookup (v1.4).
            When adaptive=False, uses fixed decay (v1.3).
        auto_tune: Use continuous sigmoid-based decay auto-tuning (default True).
            Requires adaptive=True to take effect.
        gamma_max: Upper bound for auto-tuned γ (default 0.85).
        gamma_min: Lower bound for auto-tuned γ (default 0.20).
        cv_mid: CV midpoint of sigmoid inflection (default 0.25).
        temperature: Steepness of sigmoid transition (default 0.08).
        slope_alpha: Maximum adjustment from CV slope (default 0.15).
        slope_k: Sensitivity to CV rate of change (default 20.0).
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
    decay: float = 0.55                     # base decay (fallback when auto_tune disabled)
    adaptive: bool = True                    # enable adaptive decay
    auto_tune: bool = True                   # v1.5: continuous sigmoid auto-tuning
    # ── Auto-tune hyperparams (v1.5) ──────────────────────────────────
    gamma_max: float = 0.85                  # upper bound: very stable → slow forgetting
    gamma_min: float = 0.20                  # lower bound: heavy drift → fast forgetting
    cv_mid: float = 0.25                     # sigmoid inflection point (CV where γ = midpoint)
    temperature: float = 0.08                # sigmoid steepness (smaller = sharper)
    slope_alpha: float = 0.15                # max slope adjustment fraction
    slope_k: float = 20.0                    # slope sensitivity gain
    # ── Detection params ──────────────────────────────────────────────
    min_samples_before_detect: int = 5
    hysteresis_rounds: int = 2
    phi_history: np.ndarray = field(init=False)
    write_idx: int = 0
    count: int = 0
    _drifting_streak: int = 0          # consecutive above-threshold rounds
    _currently_drifting: bool = False   # confirmed drift state (latched)
    _stage: str = "STABLE"             # STABLE | DRIFTING | RECOVERY (diagnostic only when auto_tune)
    _was_drifting: bool = False        # tracks prior drift for RECOVERY detection
    _recovery_streak: int = 0          # consecutive CV < 0.15 rounds in RECOVERY
    _prev_cv: float = 0.0              # v1.5: previous CV for slope computation
    _effective_decay: float = 0.55     # v1.5: cached effective decay for diagnostics
    adapter: Optional["HyperParamAdapter"] = None  # v1.6: auto-adapts gamma_max/min/cv_mid

    def __post_init__(self):
        self.phi_history = np.zeros(self.window_size, dtype=np.float64)

    # ── Adaptive decay (v1.5: continuous auto-tune) ───────────────────

    def _get_decay(self) -> float:
        """Return the effective decay factor for CV weighting.

        v1.5 auto_tune mode (default):
            γ(CV, dCV/dt) = γ_max − Δγ × σ((CV−CV_mid)/T) × slope_factor
            where slope_factor = 1.0 − α × tanh(k × dCV/dt)

        v1.4 three-stage mode (adaptive=True, auto_tune=False):
            STABLE=0.70, DRIFTING=0.35, RECOVERY=base_decay

        v1.3 fixed mode (adaptive=False):
            Returns configured decay value.

        The auto-tune formula ensures:
          - CV << cv_mid (very stable): γ ≈ γ_max (0.85, slow forgetting)
          - CV >> cv_mid (heavy drift): γ ≈ γ_min (0.20, fast forgetting)
          - dCV/dt > 0 (worsening): slope_factor < 1 → γ reduced further (preemptive)
          - dCV/dt < 0 (improving): slope_factor > 1 → γ increased (accelerate recovery)
        """
        if not self.adaptive:
            return self.decay

        if not self.auto_tune:
            # v1.4 three-stage lookup
            if self._stage == "DRIFTING":
                return 0.35
            elif self._stage == "RECOVERY":
                return self.decay
            else:
                return 0.70

        # ── v1.5 continuous auto-tune ─────────────────────────────────
        # current_cv → _compute_weighted_stats uses cached _effective_decay
        # (no recursion: the cached value was set by get_decay in the previous round)
        cv = self.current_cv

        # Sigmoid: maps CV → [0, 1], centered at cv_mid
        if self.temperature <= 0:
            sigmoid = 1.0 if cv > self.cv_mid else 0.0
        else:
            x = (cv - self.cv_mid) / self.temperature
            x = max(-50.0, min(50.0, x))
            sigmoid = 1.0 / (1.0 + np.exp(-x))

        # Base continuous γ from sigmoid
        delta_gamma = self.gamma_max - self.gamma_min
        gamma_continuous = self.gamma_max - delta_gamma * sigmoid

        # Slope factor: adjust based on CV trend direction
        # _prev_cv set in is_drifting() from previous round
        dcv_dt = cv - self._prev_cv
        slope_factor = 1.0 - self.slope_alpha * np.tanh(self.slope_k * dcv_dt)

        gamma_effective = gamma_continuous * slope_factor

        # Clamp to safe range
        gamma_effective = max(self.gamma_min, min(self.gamma_max, gamma_effective))

        self._effective_decay = float(gamma_effective)
        return self._effective_decay

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
        v1.6: After pushing, triggers HyperParamAdapter.adapt() if configured.

        Args:
            phi_value: The Φ (cosine similarity) score from the latest step.
        """
        self.phi_history[self.write_idx] = float(phi_value)
        self.write_idx = (self.write_idx + 1) % self.window_size
        if self.count < self.window_size:
            self.count += 1

        # v1.6: trigger hyperparameter auto-adaptation after CV update
        if self.adapter is not None and self.count >= 3:
            cv = self.current_cv
            self.adapter.push(cv)
            self.adapter.adapt(self)

    # ── Weighted statistics (v1.3: exponential decay) ─────────────────

    def _compute_weighted_stats(self) -> tuple[float, float, float]:
        """Compute exponentially weighted mean, std, and CV.

        Weights follow w[i] = decay^(n-1-i) / sum(decay^k) for sample i
        where i=0 is the oldest sample and i=n-1 is the newest.
        The newest sample always gets weight 1.0 (after normalization).

        The circular buffer is reconstructed into chronological order:
        window_chrono[0] = oldest, window_chrono[n-1] = newest.

        v1.5: Uses cached _effective_decay to avoid circular dependency.
              The decay for this computation is from the PREVIOUS round.
              _get_decay() updates _effective_decay for the NEXT round
              after the CV has been computed.

        Returns:
            (weighted_mean, weighted_std, weighted_cv)
        """
        n = self.count
        # Reconstruct chronological ordering from circular buffer.
        if n < self.window_size:
            window = self.phi_history[:n]
        else:
            indices = [(self.write_idx + i) % self.window_size for i in range(n)]
            window = self.phi_history[indices]  # chronological: [oldest, ..., newest]

        # Determine effective decay without calling _get_decay()
        # to avoid circular dependency (_get_decay → current_cv → this method)
        if not self.adaptive:
            effective_decay = self.decay
        elif not self.auto_tune:
            # v1.4 three-stage lookup (inlined to avoid recursion)
            if self._stage == "DRIFTING":
                effective_decay = 0.35
            elif self._stage == "RECOVERY":
                effective_decay = self.decay
            else:
                effective_decay = 0.70
        else:
            # v1.5 auto-tune: use cached value from previous round
            effective_decay = self._effective_decay
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

        # v1.5: track previous CV for slope computation in _get_decay()
        self._prev_cv = cv

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
            "auto_tune": self.auto_tune,
            "stage": self._stage,
            "prev_cv": round(self._prev_cv, 4),
            # v1.6: hyperparameter adapter state
            "adapter_enabled": self.adapter is not None,
            "gamma_max": self.gamma_max,
            "gamma_min": self.gamma_min,
            "cv_mid": self.cv_mid,
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
        self._prev_cv = 0.0
        self._effective_decay = self.decay


# ────────────────────────────────────────────────────────────────────────────
# v1.6 HyperParamAdapter — 自动调整 γ_max/γ_min/cv_mid
# ────────────────────────────────────────────────────────────────────────────


@dataclass
class HyperParamAdapter:
    """自动适配 DriftDetector 超参数的统计模块。

    v5.1 (2026-06-13): 基于多轮 CV 历史统计自动调整 sigmoid 公式的
    γ_max, γ_min, cv_mid 三个关键超参，消除手动调参需求。

    适配策略:
      - cv_mid: 设为 CV 分布的滚动分位数（默认 60 分位）
      - gamma_max: 基于稳定度比例自适应（越稳定 → 越高, 慢遗忘）
      - gamma_min: 基于最差 CV 自适应（越差 → 越低, 快遗忘）

    Attributes:
        history_size: CV 历史记录上限（默认 200）。
        adaptation_interval: 两次适配之间的最小轮数（默认 20）。
        cv_mid_quantile: 用于 cv_mid 的分位数（默认 0.60）。
        rounds_since_adapt: 距上次适配的轮数。
        cv_history: CV 历史值列表（最近 history_size 个）。
        cv_mid_bounds: cv_mid 的安全边界 (min, max)。
        gamma_max_bounds: gamma_max 的安全边界 (min, max)。
        gamma_min_bounds: gamma_min 的安全边界 (min, max)。
        _last_adapted: 最近一次适配的参数字典（诊断用）。
    """

    history_size: int = 200
    adaptation_interval: int = 20
    cv_mid_quantile: float = 0.60
    rounds_since_adapt: int = 0
    cv_history: list = field(default_factory=list)
    cv_mid_bounds: tuple = (0.15, 0.40)
    gamma_max_bounds: tuple = (0.70, 0.95)
    gamma_min_bounds: tuple = (0.10, 0.35)
    _last_adapted: dict = field(default_factory=dict)

    def push(self, cv: float) -> None:
        """记录一个新的 CV 值到历史缓冲区。

        Args:
            cv: 当前加权 CV 值。
        """
        self.cv_history.append(float(cv))
        if len(self.cv_history) > self.history_size:
            self.cv_history.pop(0)
        self.rounds_since_adapt += 1

    def should_adapt(self) -> bool:
        """检查是否满足适配条件。

        Returns:
            True 如果距上次适配 ≥ adaptation_interval 且历史 ≥ 20 条。
        """
        return (
            self.rounds_since_adapt >= self.adaptation_interval
            and len(self.cv_history) >= 20
        )

    def adapt(self, detector: "DriftDetector") -> dict:
        """计算适配后的超参并应用到 detector。

        使用最近 100 个 CV 值（不超过历史长度）计算统计量，
        根据稳定性指标调整三个关键超参。

        Args:
            detector: 要更新的 DriftDetector 实例。

        Returns:
            适配结果字典，包含 adapted 标志和新参数值。
        """
        if not self.should_adapt():
            return {"adapted": False}

        import numpy as np

        # 取最近最多 100 个 CV 做统计
        window_size = min(100, len(self.cv_history))
        cvs = np.array(self.cv_history[-window_size:])

        # ── cv_mid: 滚动分位数 ──────────────────────────────────────
        new_cv_mid = float(np.quantile(cvs, self.cv_mid_quantile))
        new_cv_mid = max(self.cv_mid_bounds[0], min(self.cv_mid_bounds[1], new_cv_mid))

        # ── gamma_max: 稳定度比例驱动 ─────────────────────────────
        stability_ratio = float(np.mean(cvs < 0.15))
        new_gamma_max = 0.70 + 0.25 * stability_ratio
        new_gamma_max = max(
            self.gamma_max_bounds[0], min(self.gamma_max_bounds[1], new_gamma_max)
        )

        # ── gamma_min: 最差 CV 驱动 ────────────────────────────────
        worst_cv = float(np.max(cvs[-20:]))
        if worst_cv > 0.40:
            new_gamma_min = 0.10
        elif worst_cv > 0.30:
            new_gamma_min = 0.15
        else:
            new_gamma_min = 0.20
        new_gamma_min = max(
            self.gamma_min_bounds[0], min(self.gamma_min_bounds[1], new_gamma_min)
        )

        # 应用到 detector
        detector.cv_mid = new_cv_mid
        detector.gamma_max = new_gamma_max
        detector.gamma_min = new_gamma_min

        self.rounds_since_adapt = 0
        self._last_adapted = {
            "adapted": True,
            "cv_mid": round(new_cv_mid, 4),
            "gamma_max": round(new_gamma_max, 4),
            "gamma_min": round(new_gamma_min, 4),
            "stability_ratio": round(stability_ratio, 4),
            "worst_cv": round(worst_cv, 4),
            "window_size": window_size,
        }
        return self._last_adapted

    def reset(self) -> None:
        """重置适配器到初始状态。"""
        self.cv_history.clear()
        self.rounds_since_adapt = 0
        self._last_adapted = {}

    def stats(self) -> dict:
        """返回适配器诊断统计。

        Returns:
            包含历史长度、最近适配参数等的字典。
        """
        return {
            "history_len": len(self.cv_history),
            "rounds_since_adapt": self.rounds_since_adapt,
            "last_adapted": self._last_adapted,
        }


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
