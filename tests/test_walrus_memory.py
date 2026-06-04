"""
tests/test_walrus_memory.py — Walrus Memory 层测试

测试内容：
  1. Continuation v2 proof 链
  2. Continuation v2 verify() 验证
  3. proof 篡改检测
  4. MemoryHub 存储 & 搜索
  5. MemoryHub verify_all() 批量验证
  6. Session + MemoryHub 集成
"""

import json
import os
import pytest
import numpy as np

# 将项目根目录加入 sys.path
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from core.continuation import Continuation
from core.memory_hub import MemoryHub


# ======================================================================
# 1. Continuation v2 — proof 链
# ======================================================================

def test_continuation_has_proof(tmp_path):
    """验证 Continuation v2 自动生成 SHA-256 proof。"""
    snap_dir = str(tmp_path / "snaps")
    psi = np.array([0.1, 0.2, 0.3])

    k = Continuation(
        sid="test-proof",
        psi=psi,
        env={"intent": "test"},
        reason="Test proof chain",
        snapshot_dir=snap_dir,
    )

    assert hasattr(k, "proof"), "Continuation 缺少 proof 字段"
    assert isinstance(k.proof, str), "proof 应为字符串"
    assert len(k.proof) == 64, "SHA-256 proof 应为 64 位十六进制"


def test_continuation_proof_chain(tmp_path):
    """验证 proof 形成链式结构：C2 的 proof 依赖 C1 的 proof。"""
    snap_dir = str(tmp_path / "snaps")
    psi = np.array([0.0])

    c1 = Continuation("chain-test", psi, {"step": 1}, "step 1", snapshot_dir=snap_dir)
    c2 = Continuation("chain-test", psi, {"step": 2}, "step 2",
                      snapshot_dir=snap_dir, parent_kid=c1.kid)

    # 重新计算 c2 的 proof
    import hashlib
    data2 = json.dumps({"step": 2}, ensure_ascii=False, sort_keys=True)
    expected_proof = hashlib.sha256((c1.proof + data2).encode("utf-8")).hexdigest()

    assert c2.parent_kid == c1.kid, "parent_kid 应指向 c1"
    assert c2.proof == expected_proof, "C2 proof 应基于 C1 proof + C2 数据计算"


def test_continuation_verify(tmp_path):
    """验证 Continuation v2 的 verify() 方法。"""
    snap_dir = str(tmp_path / "snaps")
    psi = np.array([0.5])

    k = Continuation("verify-test", psi, {"x": 42}, "verify", snapshot_dir=snap_dir)
    ok, msg = k.verify()
    assert ok, f"verify 应返回 True，但得到: {msg}"


def test_continuation_tamper_detection(tmp_path):
    """验证篡改检测：修改快照文件后 verify() 应返回 False。"""
    snap_dir = str(tmp_path / "snaps")
    psi = np.array([0.7])

    k = Continuation("tamper-test", psi, {"secret": 999}, "tamper", snapshot_dir=snap_dir)

    # 篡改磁盘文件
    path = os.path.join(snap_dir, f"{k.kid}.json")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    data["env"]["secret"] = 1000  # 篡改！
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    ok, msg = k.verify()
    assert not ok, "篡改后 verify 应返回 False"


# ======================================================================
# 2. MemoryHub 测试
# ======================================================================

def test_hub_register_and_store(tmp_path):
    """测试 MemoryHub 注册和存储。"""
    store_dir = str(tmp_path / "memory")
    hub = MemoryHub(store_dir=store_dir)

    hub.register("alice")
    result = hub.store("alice", {"intent": "search", "query": "量子计算"})

    assert "mid" in result
    assert "proof" in result
    assert len(result["proof"]) == 64


def test_hub_search(tmp_path):
    """测试 MemoryHub 关键词搜索。"""
    store_dir = str(tmp_path / "memory")
    hub = MemoryHub(store_dir=store_dir)

    hub.store("bob", {"topic": "量子计算", "result": "量子比特"})
    hub.store("bob", {"topic": "经典物理", "result": "牛顿力学"})
    hub.store("carol", {"topic": "量子纠缠", "result": "EPR佯谬"})

    results = hub.search("量子")
    assert len(results) >= 2, f"搜索'量子'应返回至少2条，实际: {len(results)}"

    results2 = hub.search("牛顿")
    assert len(results2) == 1

    results3 = hub.search("不存在")
    assert len(results3) == 0


def test_hub_verify_all(tmp_path):
    """测试 MemoryHub 批量验证。"""
    store_dir = str(tmp_path / "memory")
    hub = MemoryHub(store_dir=store_dir)

    hub.store("eve", {"a": 1})
    hub.store("eve", {"b": 2})

    report = hub.verify_all()
    assert report["total"] == 2
    assert report["valid"] == 2
    assert report["invalid"] == 0


# ======================================================================
# 3. Session + MemoryHub 集成
# ======================================================================

def test_session_with_memory_hub(tmp_path):
    """测试 TaijiSession 集成 MemoryHub 的基本流程。"""
    from core.session import TaijiSession
    from hal.llm_router import LLMRouter

    store_dir = str(tmp_path / "memory")
    snap_dir = str(tmp_path / "snaps")

    hub = MemoryHub(store_dir=store_dir)
    llm = LLMRouter()

    sess = TaijiSession(
        sid="hub-test",
        llm_router=llm,
        snapshot_dir=snap_dir,
        memory_hub=hub,
    )

    assert sess.memory_hub is not None
    assert sess.memory_hub is hub

    # 验证集成了 search_memory
    assert hasattr(sess, "search_memory"), "session 缺少 search_memory 方法"
    assert hasattr(sess, "verify_integrity"), "session 缺少 verify_integrity 方法"
