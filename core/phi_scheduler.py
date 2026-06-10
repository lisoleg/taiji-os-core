"""
core/phi_scheduler.py — Φ-Scheduler / FlowBreaker v4.1

Φ 门控调度器：检查候选语义向量与世界模型的语义一致性。

v4.1 升级:
  - 新增 adaptive 模式：滑动窗口 CV 公式，阈值随数据分布自适应
  - static 模式（默认）：硬编码阈值，向后兼容
  - 滑动窗口记录历史 Φ 值，计算变异系数 (CV = σ/μ)
  - 阈值调整公式: threshold = base * (1 + alpha * CV), clamped to [min, max]
"""

import numpy as np
from collections import deque
from typing import Optional


class PhiScheduler:
    """
    Φ-Scheduler / FlowBreaker v4.1

    两种模式:
      - "static"  (默认): 硬编码阈值，v3.2 行为
      - "adaptive": 基于滑动窗口 CV 动态调整阈值

    自适应公式:
      CV = σ_window / μ_window  (窗口Φ值的变异系数)
      threshold = base * (1 + alpha * CV)
      threshold = clamp(threshold, min_threshold, max_threshold)

    直觉: 当历史Φ值波动大时(CV高)，说明世界模型状态不稳定，
          适当降低门控阈值以避免过度拒绝；波动小时，严格门控。

    参数:
        mode          : "static" | "adaptive"
        threshold     : static 模式下的固定阈值（默认 0.65）
        window_size   : adaptive 模式的滑动窗口大小（默认 50）
        alpha         : CV 敏感度系数（默认 0.5）
        min_threshold : 自适应下限（默认 0.45）
        max_threshold : 自适应上限（默认 0.85）
    """

    def __init__(
        self,
        threshold: float = 0.65,
        mode: str = "static",
        window_size: int = 50,
        alpha: float = 0.5,
        min_threshold: float = 0.45,
        max_threshold: float = 0.85,
    ):
        self.mode = mode
        self.base_threshold = threshold
        self.threshold = threshold  # 当前有效阈值

        # adaptive 模式参数
        self.window_size = window_size
        self.alpha = alpha
        self.min_threshold = min_threshold
        self.max_threshold = max_threshold
        self._phi_window: deque = deque(maxlen=window_size)

        # 统计计数器
        self._accept_count = 0
        self._reject_count = 0

    # ------------------------------------------------------------------
    # 公开 API
    # ------------------------------------------------------------------

    def check(self, w, new_psi: np.ndarray) -> tuple[bool, float]:
        """
        检查候选向量的 Φ 值是否超过当前阈值。

        参数:
            w       : WorldModel 实例（含 phi() 方法）
            new_psi : 候选语义向量

        返回:
            (ok: bool, phi_val: float)
        """
        phi_val = w.phi(new_psi) if hasattr(w, "phi") else self._compute_phi(w, new_psi)

        # 如果是 adaptive 模式，先更新阈值再判定
        if self.mode == "adaptive":
            self._phi_window.append(phi_val)
            self._update_threshold()

        ok = phi_val >= self.threshold
        if ok:
            self._accept_count += 1
        else:
            self._reject_count += 1

        return ok, phi_val

    # ------------------------------------------------------------------
    # Adaptive 阈值更新
    # ------------------------------------------------------------------

    def _update_threshold(self) -> None:
        """基于滑动窗口的 Φ 值分布动态调整阈值。"""
        if len(self._phi_window) < 3:
            # 窗口太小时使用基础阈值
            self.threshold = self.base_threshold
            return

        values = np.array(self._phi_window)
        mu = np.mean(values)
        sigma = np.std(values)

        if mu < 1e-8:
            # 避免除零
            self.threshold = self.base_threshold
            return

        cv = sigma / mu  # 变异系数
        adjusted = self.base_threshold * (1.0 + self.alpha * cv)
        self.threshold = float(np.clip(adjusted, self.min_threshold, self.max_threshold))

    @staticmethod
    def _compute_phi(w, new_psi: np.ndarray) -> float:
        """独立计算余弦相似度（当 WorldModel 无 phi() 方法时使用）。"""
        psi = getattr(w, "psi", np.zeros_like(new_psi))
        dot = np.dot(psi, new_psi)
        norm = (np.linalg.norm(psi) * np.linalg.norm(new_psi)) + 1e-8
        return float(dot / norm)

    # ------------------------------------------------------------------
    # 统计
    # ------------------------------------------------------------------

    @property
    def acceptance_rate(self) -> float:
        total = self._accept_count + self._reject_count
        if total == 0:
            return 1.0
        return self._accept_count / total

    def stats(self) -> dict:
        """返回 Φ 调度器统计信息。"""
        result = {
            "mode": self.mode,
            "threshold": round(self.threshold, 4),
            "accept_count": self._accept_count,
            "reject_count": self._reject_count,
            "acceptance_rate": round(self.acceptance_rate, 4),
        }
        if self.mode == "adaptive" and len(self._phi_window) >= 3:
            vals = np.array(self._phi_window)
            result["window_size"] = len(vals)
            result["window_mean"] = round(float(np.mean(vals)), 4)
            result["window_std"] = round(float(np.std(vals)), 4)
            result["window_cv"] = round(
                float(np.std(vals) / (np.mean(vals) + 1e-8)), 4
            )
        return result

    def reset(self) -> None:
        """重置统计计数器和滑动窗口。"""
        self._phi_window.clear()
        self._accept_count = 0
        self._reject_count = 0
        if self.mode == "adaptive":
            self.threshold = self.base_threshold
