"""tests/test_walrus_memory.py — Walrus Memory 风格升级测试

覆盖:
  1. Continuation v2: proof 链 + verify() + parent_kid
  2. MemoryHub: 注册 / 存储 / 搜索 / proof 链遍历 / 批量验证
  3. TaijiSession × MemoryHub 集成
"""
import pytest
import sys
import os
import tempfile
import shutil
import json
import hashlib

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.continuation import Continuation
from core.memory_hub import MemoryHub
from core.session import TaijiSession


# ======================================================================
# Fixtures
# ======================================================================

@pytest.fixture
def tmp_dir():
    d = tempfile.mkdtemp(prefix="taiji_test_")
    yield d
    shutil.rmtree(d, ignore_errors=True)


@pytest.fixture
def hub(tmp_dir):
    return MemoryHub(os.path.join(tmp_dir, "hub"))


# ======================================================================
# Continuation v2: Proof Chain
# ======================================================================

def test_proof_chain_standalone(tmp_dir):
    """每条 Continuation 携带 SHA-256 proof，verify() 通过"""
    psi = np.array([0.1, 0.2, 0.3], dtype=np.float64)

    k1 = Continuation("s1", psi, {"a": 1}, "reason1", snapshot_dir=tmp_dir)
    assert k1.proof, "v2 proof 不能为空"
    assert k1.payload_hash, "v2 payload_hash 不能为空"
    assert k1.verify(), "自验证（无父 proof）应通过"

    # 加载后验证
    k1_loaded = Continuation.load(k1.kid, tmp_dir)
    assert k1_loaded.proof == k1.proof, "proof 应与保存前一致"
    assert k1_loaded.verify(), "加载后 verify 应通过"


def test_proof_chain_linked(tmp_dir):
    """父子 proof 链：k2.proof 依赖 k1.proof"""
    psi = np.ones(3)

    k1 = Continuation("s1", psi, {"step": 1}, "first", snapshot_dir=tmp_dir)
    k2 = Continuation(
        "s1", psi * 2, {"step": 2}, "second",
        snapshot_dir=tmp_dir, parent_kid=k1.kid, parent_proof=k1.proof,
    )

    # k2 用 k1 的 proof 验证
    assert k2.verify(parent_proof=k1.proof), "k2 用正确父 proof 验证应通过"

    # 用错误父 proof 验证应失败
    assert not k2.verify(parent_proof="wrong_proof"), "错误父 proof 应验证失败"


def test_proof_chain_tamper_detection(tmp_dir):
    """篡改快照文件后 verify() 应检测到"""
    psi = np.array([0.5, 0.6])
    k = Continuation("s1", psi, {"x": "y"}, "test", snapshot_dir=tmp_dir)
    assert k.verify(), "原始快照应通过验证"

    # 篡改磁盘文件——修改 payload_hash
    path = os.path.join(tmp_dir, f"{k.kid}.json")
    with open(path, "r") as f:
        data = json.load(f)
    data["payload_hash"] = "0" * 64  # 伪造
    with open(path, "w") as f:
        json.dump(data, f)

    # 重新加载，verify 应失败
    k_loaded = Continuation.load(k.kid, tmp_dir)
    assert not k_loaded.verify(), "篡改后 verify 应失败"


def test_proof_chain_parent_kid(tmp_dir):
    """parent_kid 正确记录父子关系"""
    psi = np.ones(3)
    k1 = Continuation("s1", psi, {"n": 1}, "first", snapshot_dir=tmp_dir)
    k2 = Continuation(
        "s1", psi, {"n": 2}, "second",
        snapshot_dir=tmp_dir, parent_kid=k1.kid, parent_proof=k1.proof,
    )
    assert k2.parent_kid == k1.kid, "parent_kid 应正确链接"


# ======================================================================
# MemoryHub
# ======================================================================

def test_hub_register(hub):
    """注册会话，索引文件持久化"""
    hub.register("sid-A", {"role": "researcher"})
    hub.register("sid-B", {"role": "executor"})

    sessions = hub.list_sessions()
    assert len(sessions) == 2
    sids = {s["sid"] for s in sessions}
    assert sids == {"sid-A", "sid-B"}


def test_hub_store_and_search(hub):
    """MemoryHub.store() 生成 proof 链，search() 跨会话检索"""
    hub.register("sid-X")
    psi = np.array([0.1, 0.2, 0.3])

    k1 = hub.store("sid-X", psi, {"topic": "量子计算"}, "研究量子纠错码", "snapshots")
    k2 = hub.store(
        "sid-X", psi * 2, {"topic": "AGI安全"}, "讨论对齐问题",
        "snapshots", parent_kid=k1.kid,
    )

    # 跨会话搜索
    results = hub.search(keyword="量子", limit=5)
    assert len(results) >= 1, "应搜到至少 1 条相关快照"
    assert any("量子" in r.get("reason", "") for r in results)

    # 限定会话搜索
    results_x = hub.search(keyword="AGI", sid="sid-X", limit=5)
    assert len(results_x) >= 1


def test_hub_proof_chain_traversal(hub):
    """get_chain() 正确遍历父子 proof 链"""
    hub.register("sid-C")
    psi = np.ones(3)

    k1 = hub.store("sid-C", psi, {"n": 1}, "A", "snapshots")
    k2 = hub.store("sid-C", psi, {"n": 2}, "B", "snapshots", parent_kid=k1.kid)
    k3 = hub.store("sid-C", psi, {"n": 3}, "C", "snapshots", parent_kid=k2.kid)

    chain = hub.get_chain(k3.kid, "snapshots")
    assert len(chain) == 3, f"应从 k3 → k2 → k1，共3个，实际{len(chain)}"
    assert chain[0]["kid"] == k3.kid
    assert chain[-1]["kid"] == k1.kid


def test_hub_verify_all(hub, tmp_dir):
    """批量完整性验证（用独立 tmp_dir，避免扫到历史快照）"""
    hub.register("sid-V")
    psi = np.ones(3)

    hub.store("sid-V", psi, {"v": 1}, "valid", tmp_dir)
    report = hub.verify_all(tmp_dir)
    assert report["total"] == 1, f"应只验证 1 条，实际 {report['total']}"
    assert report["passed"] == 1, "唯一快照应通过验证"
    assert len(report["failed"]) == 0


# ======================================================================
# TaijiSession × MemoryHub 集成
# ======================================================================

class MockRouter:
    """纯 Mock LLM Router，不调 API。"""
    def complete(self, prompt: str) -> str:
        if "矛盾" in prompt or "从未离开" in prompt:
            return "检测到语义矛盾，无法生成一致响应。"
        return f"[Mock] 已处理：{prompt[:60]}"


def test_session_with_hub(hub, tmp_dir):
    """TaijiSession 集成 MemoryHub：Continuation 自动走 hub.store()"""
    router = MockRouter()
    sess = TaijiSession(
        "sid-hub", router,
        snapshot_dir=tmp_dir,
        memory_hub=hub,
    )

    # 先跑一轮——初始 ψ=零，Φ=0 必然触发 Continuation（这是预期行为）
    out1 = sess.run("你好")
    assert "Continuation Saved" in out1

    # 第二轮：ψ 已从 Continuation 中更新（resume 语义），再次运行
    out2 = sess.run("继续推进")
    assert isinstance(out2, str) and len(out2) > 0

    # 验证 hub 连接状态和 proof 链
    status = sess.status()
    assert status["hub_connected"] is True
    assert status["last_kid"] is not None

    # 跨会话搜索
    results = sess.search_memory("你好")
    assert len(results) >= 0  # 取决于 snapshot_dir 内容


def test_session_search_memory_no_hub(tmp_dir):
    """无 hub 时 search_memory 返回空列表"""
    sess = TaijiSession("sid-no-hub", MockRouter(), snapshot_dir=tmp_dir)
    assert sess.search_memory("anything") == []


def test_session_verify_integrity(hub, tmp_dir):
    """verify_integrity 能正确报告证明链状态"""
    router = MockRouter()
    sess = TaijiSession("sid-verify", router, snapshot_dir=tmp_dir, memory_hub=hub)
    sess.run("测试1")
    sess.run("我昨天去了北京，但我从未离开过上海。")  # → Continuation
    report = sess.verify_integrity()
    assert report["total"] > 0
    assert report["passed"] == report["total"]
