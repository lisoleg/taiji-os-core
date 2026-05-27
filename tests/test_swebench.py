import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.session import TaijiSession
from hal.llm_router import LLMRouter


def test_psr_basic_task():
    """PSR: 基础任务解决率 — session 应能正常响应编程相关指令"""
    sess = TaijiSession("test_swe", LLMRouter())
    out = sess.run("实现一个冒泡排序函数")
    # 应返回实际输出或 Continuation（不崩溃）
    assert isinstance(out, str)
    assert len(out) > 0


def test_psr_no_crash_on_complex_input():
    """PSR: 复杂输入稳定性 — 不应抛出异常"""
    sess = TaijiSession("test_swe2", LLMRouter())
    complex_input = "修复GitHub issue #1234: IndexError in list comprehension when input is empty"
    try:
        out = sess.run(complex_input)
        assert isinstance(out, str)
    except Exception as e:
        pytest.fail(f"session.run 抛出异常: {e}")
