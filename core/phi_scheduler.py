"""core/phi_scheduler.py — Φ 语义连贯性门控调度器

v4.1: 支持 static（固定阈值）和 adaptive（滑动窗口动态阈值）两种模式。

Adaptive 公式（基于变异系数 CV）:
    recent   = phi_history[-window:]
    mean     = avg(recent)
    std      = std(recent)
    cv       = std / (mean + 1e-8)
    threshold = base_threshold * (1 + alpha * cv)
    threshold = clamp(threshold, min_threshold, max_threshold)

默认 mode="static"，行为与 v4 完全一致（threshold=0.65）。
"""

from __future__ import annotations

import numpy as np
from core.world_model import WorldModel


class PhiScheduler:
    """Φ 语义连贯性门控调度器。

    检查候选语义向量是否与当前世界模型保持足够的一致性。
    低于阈值时触发门控，拒绝本次更新。

    用法::

        # 静态模式（v4 兼容）
        phi = PhiScheduler(threshold=0.65)

        # 自适应模式
        phi = PhiScheduler(threshold=0.65, mode="adaptive", window=20, alpha=0.3)

        ok, phi_val = phi.check(world_model, new_psi)
    """

    def __init__(
        self,
        threshold: float = 0.65,
        mode: str = "static",
        window: int = 20,
        alpha: float = 0.3,
        min_threshold: float = 0.5,
        max_threshold: float = 0.9,
    ):
        """初始化 Φ 调度器。

        参数:
            threshold:     基础阈值（static 模式使用固定值；adaptive 模式作为基准）
            mode:          "static" | "adaptive"
            window:        adaptive 模式滑动窗口大小，必须 >= 5
            alpha:         adaptive 模式灵敏度系数，范围 (0, 1]
            min_threshold: adaptive 模式阈值下限
            max_threshold: adaptive 模式阈值上限
        """
        self.base_threshold = threshold
        self.mode = mode
        self.window = max(window, 5)
        self.alpha = alpha
        self.min_threshold = min_threshold
        self.max_threshold = max_threshold
        self._phi_history: list[float] = []
        self._current_threshold = threshold

    @property
    def history(self) -> list[float]:
        """Return a copy of the internal ϕ history."""
        return list(self._phi_history)

    # ------------------------------------------------------------------
    # 主门控入口
    # ------------------------------------------------------------------

    def check(self, w: WorldModel, new_psi: np.ndarray):
        """执行 Φ 门控检查。

        参数:
            w:       WorldModel 实例
            new_psi: 候选语义向量

        返回:
            (ok: bool, phi_val: float)
            ok=True  → 通过一致性检验，可以接受
            ok=False → Φ 值过低，触发门控拒绝
        """
        phi_val = w.phi(new_psi)
        self._phi_history.append(phi_val)

        # adaptive 模式：滑动窗口 >= window 时动态更新阈值
        if self.mode == "adaptive" and len(self._phi_history) >= self.window:
            self._update_adaptive_threshold()

        ok = phi_val >= self._current_threshold
        return ok, phi_val

    # ------------------------------------------------------------------
    # Adaptive 阈值计算
    # ------------------------------------------------------------------

    def _update_adaptive_threshold(self) -> None:
        """基于滑动窗口的变异系数（CV）计算动态阈值。

        变异系数越大 → 语义波动大 → 适当降低阈值（更宽松）；
        变异系数越小 → 语义稳定 → 阈值趋向基准值。
        """
        recent = self._phi_history[-self.window:]
        mean = sum(recent) / len(recent)
        variance = sum((x - mean) ** 2 for x in recent) / len(recent)
        std = variance ** 0.5
        # 变异系数：std / mean，避免除零
        cv = std / (mean + 1e-8)

        # 自适应公式: 波动大时降低阈值（放宽门控），波动小时恢复基准
        adaptive = self.base_threshold * (1 + self.alpha * cv)

        # 钳位到 [min, max] 区间
        self._current_threshold = max(
            self.min_threshold,
            min(self.max_threshold, adaptive),
        )

    # ------------------------------------------------------------------
    # 工具方法
    # ------------------------------------------------------------------

    def get_threshold(self) -> float:
        """返回当前生效的阈值（adaptive 模式下为动态计算值）。"""
        return self._current_threshold

    def reset(self) -> None:
        """重置历史窗口和阈值到初始状态。"""
        self._phi_history.clear()
        self._current_threshold = self.base_threshold
