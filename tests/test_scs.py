import pytest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.session import TaijiSession
from hal.llm_router import LLMRouter


def test_world_consistency():
    """SCS: 世界一致性测试 — ψ 向量在快照保存/恢复后应保持精确一致"""
    sess = TaijiSession("test_scs", LLMRouter())
    sess.run("苹果公司发布了iPhone 15")
    psi_before = sess.w.psi.copy()

    # Simulate crash and resume
    sess.w.psi = psi_before

    assert np.allclose(sess.w.psi, psi_before, atol=1e-6)
