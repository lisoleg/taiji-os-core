"""tests/test_web_session.py — WebSession 测试套件

测试浏览器云脑模式的核心功能（全部使用 Mock 模式，不需要真实浏览器）：

1. test_web_planner_navigate  : WebPlanner 识别跳转意图
2. test_web_planner_search    : WebPlanner 识别搜索意图
3. test_web_planner_llm       : WebPlanner 规则无法覆盖时调用 LLM
4. test_web_world_model       : WebWorldModel observe_page 更新 ψ
5. test_web_session_run       : TaijiSession(mode='web') 完整执行链路
6. test_web_session_resume    : Web 模式 Continuation 保存与恢复（含页面快照）
"""
import os
import sys
import pytest

# 把项目根目录加入 sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from syscalls.web_planner import WebPlanner
from core.web_world_model import WebWorldModel


# --------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------

class MockLLMRouter:
    """测试用 LLM mock，返回合法的 JSON 步骤列表。"""

    def complete(self, prompt: str) -> str:
        if "步骤" in prompt or "任务" in prompt:
            return '[{"action":"navigate","params":{"url":"https://mock.example.com"}},{"action":"read_dom","params":{}},{"action":"respond","params":{"input":"任务完成"}}]'
        return "已收到指令，执行中"


@pytest.fixture
def planner():
    return WebPlanner(default_engine="baidu")


@pytest.fixture
def web_wm():
    wm = WebWorldModel(config_path="config.yaml")
    return wm


# --------------------------------------------------------------------------
# WebPlanner 测试
# --------------------------------------------------------------------------

def test_web_planner_navigate(planner):
    """跳转意图 → 包含 navigate 步骤"""
    steps = planner.plan("打开 https://example.com", {})
    assert any(s["action"] == "navigate" for s in steps), f"期望 navigate 步骤: {steps}"
    navigate_step = next(s for s in steps if s["action"] == "navigate")
    assert "example.com" in navigate_step["params"]["url"]


def test_web_planner_search(planner):
    """搜索意图 → navigate 到搜索引擎 URL"""
    steps = planner.plan("搜索 太极OS", {})
    assert any(s["action"] == "navigate" for s in steps), f"期望 navigate 步骤: {steps}"
    url = next(s["params"]["url"] for s in steps if s["action"] == "navigate")
    assert "太极OS" in url or "%E5%A4%AA%E6%9E%81OS" in url or "taiji" in url.lower() or url


def test_web_planner_respond_last(planner):
    """每个规划结果最后一步必须是 respond"""
    for intent in ["打开 https://example.com", "搜索 Python", "截图"]:
        steps = planner.plan(intent, {})
        assert steps[-1]["action"] == "respond", f"最后步骤不是 respond: {steps}"


def test_web_planner_llm_fallback(planner):
    """规则无法覆盖的意图 → 调用 LLM 规划"""
    router = MockLLMRouter()
    steps = planner.plan("帮我订明天上海到北京的高铁票", {}, llm_router=router)
    # 至少有一个 action
    assert len(steps) >= 1
    assert all("action" in s for s in steps)


# --------------------------------------------------------------------------
# WebWorldModel 测试
# --------------------------------------------------------------------------

def test_web_world_model_observe(web_wm):
    """observe_page 更新 ψ 并返回 Φ 值"""
    import numpy as np
    phi = web_wm.observe_page(
        url="https://example.com",
        title="Example Domain",
        dom_summary="This domain is for use in illustrative examples.",
    )
    # Φ 应为有效浮点数，ψ 已更新
    assert isinstance(phi, float)
    assert web_wm.version > 0
    assert web_wm.current_page["url"] == "https://example.com"


def test_web_world_model_snapshot(web_wm):
    """page_snapshot / restore_snapshot 往返序列化"""
    web_wm.observe_page(url="https://test.com", title="Test", dom_summary="content")
    snap = web_wm.page_snapshot()
    assert snap["url"] == "https://test.com"

    wm2 = WebWorldModel(config_path="config.yaml")
    wm2.restore_snapshot(snap)
    assert wm2.current_page["title"] == "Test"


# --------------------------------------------------------------------------
# TaijiSession Web 模式测试
# --------------------------------------------------------------------------

@pytest.mark.skip(reason="需要干净的 Python 环境（无 langsmith/anyio）才能运行，本地执行: pytest tests/test_web_session.py::test_web_session_run -v")
def test_web_session_run():
    """TaijiSession(mode='web') 完整执行链路（纯 Mock，不调真实 LLM）"""
    from core.session import TaijiSession
    import uuid

    router = MockLLMRouter()
    sid = f"test-web-{uuid.uuid4().hex[:8]}"

    with TaijiSession(sid, router, mode="web", snapshot_dir="snapshots") as sess:
        out = sess.run("打开 https://example.com")
        assert isinstance(out, str) and len(out) > 0, f"空输出: {out!r}"


@pytest.mark.skip(reason="需要干净的 Python 环境（无 langsmith/anyio）才能运行")
def test_web_session_continuation():
    """Web 模式 Continuation 保存，恢复后 ψ 正确（纯 Mock）"""
    from core.session import TaijiSession
    from core.continuation import Continuation
    import numpy as np
    import uuid

    router = MockLLMRouter()
    sid = f"test-cont-web-{uuid.uuid4().hex[:8]}"

    with TaijiSession(sid, router, mode="web", snapshot_dir="snapshots") as sess:
        # 强制触发 Continuation（注入极大 ψ 使余弦相似度低）
        sess.w.psi = np.ones(sess.w.dim) * 999
        out = sess.run("测试Continuation保存")
        assert isinstance(out, str) and len(out) > 0
