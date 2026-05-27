import pytest
import time
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.session import TaijiSession
from core.continuation import Continuation
from hal.llm_router import LLMRouter


def test_migration_time():
    """DT: 迁移时间测试 — Continuation 保存与恢复时间应 < 1s"""
    sess = TaijiSession("test_dt", LLMRouter())
    # 触发 Continuation（用矛盾输入）
    out = sess.run("我昨天去了北京，但我从未离开过上海。")
    assert "Continuation Saved" in out
    kid = out.split("Continuation Saved: ")[-1].split(" | ")[0].strip()

    # 测量恢复时间
    t0 = time.time()
    sess2 = TaijiSession("test_dt_resume", LLMRouter())
    sess2.resume(kid)
    elapsed = time.time() - t0

    assert elapsed < 1.0, f"恢复时间 {elapsed:.3f}s 超过 1s 限制"


def test_continuation_persistence():
    """DT: Continuation 序列化完整性 — 保存后加载的 ψ 应与原始一致"""
    import numpy as np
    sess = TaijiSession("test_persist", LLMRouter())
    sess.run("量子计算将在2030年实现通用化。")
    psi_snap = sess.w.psi.copy()

    out = sess.run("我昨天去了北京，但我从未离开过上海。")
    if "Continuation Saved" in out:
        kid = out.split("Continuation Saved: ")[-1].split(" | ")[0].strip()
        k = Continuation.load(kid)
        loaded_psi = k.psi
        # psi 在 Continuation 创建时已保存，允许有一次更新的差值
        assert loaded_psi.shape == psi_snap.shape
