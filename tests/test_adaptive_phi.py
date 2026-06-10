"""
tests/test_adaptive_phi.py — Φ Scheduler adaptive 模式测试
"""
import pytest, sys, os
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.phi_scheduler import PhiScheduler
from core.world_model import WorldModel


class TestPhiSchedulerStatic:
    """Φ Scheduler static 模式（默认）"""

    def test_default_threshold(self):
        phi = PhiScheduler()
        assert phi.threshold == 0.65
        assert phi.mode == "static"

    def test_custom_threshold(self):
        phi = PhiScheduler(threshold=0.80)
        assert phi.threshold == 0.80

    def test_check_accept_above_threshold(self):
        """Φ 值高于阈值时应通过"""
        phi = PhiScheduler(threshold=0.5)
        wm = WorldModel()
        wm.update("test text")
        # 直接使用 ψ 本身 → Φ = 1.0
        ok, phi_val = phi.check(wm, wm.psi)
        assert ok
        assert phi_val > 0.9

    def test_check_reject_below_threshold(self):
        """Φ 值低于阈值时应拒绝"""
        phi = PhiScheduler(threshold=0.5)
        wm = WorldModel()
        wm.update("test text")
        # 构造一个与 ψ 完全相反的向量
        far_psi = -wm.psi
        ok, phi_val = phi.check(wm, far_psi)
        assert not ok
        assert phi_val < 0

    def test_acceptance_rate(self):
        phi = PhiScheduler()
        wm = WorldModel()
        wm.update("base")
        phi.check(wm, wm.psi)  # Φ = 1.0 → 接受
        assert phi.acceptance_rate == 1.0
        phi.check(wm, -wm.psi)  # Φ = -1.0 → 拒绝
        assert phi.acceptance_rate == 0.5

    def test_reset(self):
        phi = PhiScheduler()
        wm = WorldModel()
        phi.check(wm, wm.encode("a"))
        phi.check(wm, wm.encode("b"))
        phi.reset()
        assert phi._accept_count == 0
        assert phi._reject_count == 0
        assert phi.acceptance_rate == 1.0

    def test_stats(self):
        phi = PhiScheduler()
        stats = phi.stats()
        assert "mode" in stats
        assert "threshold" in stats
        assert "accept_count" in stats
        assert "reject_count" in stats
        assert "acceptance_rate" in stats


class TestPhiSchedulerAdaptive:
    """Φ Scheduler adaptive 模式"""

    def test_init_adaptive(self):
        phi = PhiScheduler(mode="adaptive")
        assert phi.mode == "adaptive"
        assert phi.threshold == 0.65  # 初始值=基础值

    def test_adaptive_window_small(self):
        """窗口太小（<3）时使用基础阈值"""
        phi = PhiScheduler(mode="adaptive", window_size=50)
        wm = WorldModel()
        phi.check(wm, wm.encode("a"))
        phi.check(wm, wm.encode("b"))
        assert phi.threshold == phi.base_threshold  # 窗口=2, 使用基础值

    def test_adaptive_window_large(self):
        """窗口 ≥3 时开始调整阈值"""
        phi = PhiScheduler(mode="adaptive", window_size=50)
        wm = WorldModel()
        wm.update("base text for stable psi")
        # 添加多个高 Φ 值
        for _ in range(10):
            phi.check(wm, wm.psi + np.random.normal(0, 0.01, wm.dim))
        assert phi.threshold != phi.base_threshold  # 应该调整了
        # 高一致性 → CV 低 → 阈值应接近或略高于基础值
        assert phi.min_threshold <= phi.threshold <= phi.max_threshold

    def test_adaptive_bounds(self):
        """阈值始终在 [min, max] 范围内"""
        phi = PhiScheduler(mode="adaptive", min_threshold=0.3, max_threshold=0.9)
        wm = WorldModel()
        for i in range(20):
            # 交替高低 Φ
            vec = wm.psi if i % 2 == 0 else -wm.psi
            phi.check(wm, vec)
        assert phi.min_threshold <= phi.threshold <= phi.max_threshold

    def test_adaptive_stats(self):
        """adaptive 模式统计包含窗口信息"""
        phi = PhiScheduler(mode="adaptive")
        wm = WorldModel()
        wm.update("base")
        for _ in range(5):
            phi.check(wm, wm.psi + np.random.normal(0, 0.01, wm.dim))
        stats = phi.stats()
        assert "window_size" in stats
        assert "window_mean" in stats
        assert "window_std" in stats
        assert "window_cv" in stats

    def test_adaptive_reset(self):
        """reset 后阈值回到基础值"""
        phi = PhiScheduler(mode="adaptive")
        wm = WorldModel()
        wm.update("base")
        for _ in range(10):
            phi.check(wm, wm.psi + np.random.normal(0, 0.01, wm.dim))
        pre_reset = phi.threshold
        phi.reset()
        assert phi.threshold == phi.base_threshold
        assert len(phi._phi_window) == 0
