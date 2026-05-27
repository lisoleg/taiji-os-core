import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.session import TaijiSession
from hal.llm_router import LLMRouter


def test_hallucination_detection():
    """HDR: 幻觉拦截率测试 — 矛盾输入应触发 Continuation 保存"""
    sess = TaijiSession("test_hdr", LLMRouter())
    cmd = "我昨天去了北京，但我从未离开过上海。"
    out = sess.run(cmd)
    assert "Continuation Saved" in out
    assert "矛盾" in out or "Φ" in out
