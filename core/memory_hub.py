"""
core/memory_hub.py — MemoryHub: Walrus Memory 共享记忆空间

概念映射（Walrus Memory → MemoryHub）：
  - Portable Memory   → 跨会话可移植的 Continuation proof 链
  - Integrity Proofs  → SHA-256 哈希链验证
  - Shared Memory     → 多 session 共享命名空间
  - MCP Native        → 通过 mcp_bridge.py 暴露为 MCP 工具

MemoryHub 提供：
  - register(sid)       : 注册会话到共享空间
  - store(sid, data)    : 存储记忆（自动附加 proof）
  - search(query)       : 关键词搜索共享记忆
  - verify_all()        : 批量完整性验证
  - list_sessions()     : 列出已注册会话
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from typing import Optional


class MemoryHub:
    """
    跨会话共享记忆空间。

    每条记忆都是一个不可变记录，包含：
      - id         : 唯一标识
      - sid        : 来源会话
      - data       : 记忆内容
      - proof      : SHA-256 完整性证明（链式）
      - prev_proof : 前一条记忆的 proof（形成链）
      - timestamp  : 创建时间戳

    使用方式:
        hub = MemoryHub(store_dir="memory_store")
        hub.register("alice")
        hub.store("alice", {"intent": "搜索量子计算", "result": "..."})
        results = hub.search("量子")
        hub.verify_all()
    """

    def __init__(self, store_dir: str = "memory_store"):
        self.store_dir = store_dir
        os.makedirs(self.store_dir, exist_ok=True)
        self._sessions: dict[str, list[str]] = {}   # sid → [memory_ids]
        self._last_proof: dict[str, str] = {}         # sid → latest proof hash

    # ------------------------------------------------------------------
    # 注册
    # ------------------------------------------------------------------

    def register(self, sid: str) -> str:
        """注册一个会话到共享记忆空间。返回注册确认。"""
        if sid not in self._sessions:
            self._sessions[sid] = []
            self._last_proof[sid] = ""  # 初始空 proof
        return f"registered:{sid}"

    # ------------------------------------------------------------------
    # 存储
    # ------------------------------------------------------------------

    def store(self, sid: str, data: dict) -> dict:
        """
        存储一条记忆，自动计算 proof 链。

        返回:
            {"mid": str, "proof": str, "prev_proof": str}
        """
        if sid not in self._sessions:
            self.register(sid)

        prev = self._last_proof.get(sid, "")

        # 构建内容 + 计算 proof
        content = json.dumps(data, ensure_ascii=False, sort_keys=True)
        chain_input = prev + content
        proof = hashlib.sha256(chain_input.encode("utf-8")).hexdigest()

        mid = f"{sid}_{len(self._sessions[sid]):05d}_{proof[:8]}"
        record = {
            "id": mid,
            "sid": sid,
            "data": data,
            "proof": proof,
            "prev_proof": prev,
            "timestamp": time.time(),
        }

        path = os.path.join(self.store_dir, f"{mid}.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(record, f, ensure_ascii=False, indent=2)

        self._sessions[sid].append(mid)
        self._last_proof[sid] = proof

        return {"mid": mid, "proof": proof, "prev_proof": prev}

    # ------------------------------------------------------------------
    # 搜索
    # ------------------------------------------------------------------

    def search(self, query: str) -> list[dict]:
        """
        在共享记忆中搜索关键词。

        参数:
            query: 搜索关键词（空格分隔多关键词，AND 逻辑）

        返回:
            匹配的记忆记录列表，按时间倒序
        """
        keywords = query.strip().split()
        results = []

        for fname in os.listdir(self.store_dir):
            if not fname.endswith(".json"):
                continue
            path = os.path.join(self.store_dir, fname)
            try:
                with open(path, "r", encoding="utf-8") as f:
                    record = json.load(f)
            except (json.JSONDecodeError, IOError):
                continue

            data_str = json.dumps(record.get("data", {}), ensure_ascii=False).lower()
            if all(kw.lower() in data_str for kw in keywords):
                results.append(record)

        results.sort(key=lambda r: r.get("timestamp", 0), reverse=True)
        return results

    # ------------------------------------------------------------------
    # 完整性验证
    # ------------------------------------------------------------------

    def verify(self, mid: str) -> tuple[bool, str]:
        """
        验证单条记忆的完整性。

        返回:
            (is_valid, message)
        """
        path = os.path.join(self.store_dir, f"{mid}.json")
        try:
            with open(path, "r", encoding="utf-8") as f:
                record = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError, IOError) as e:
            return False, f"读取失败: {e}"

        prev = record.get("prev_proof", "")
        data_str = json.dumps(record.get("data", {}), ensure_ascii=False, sort_keys=True)
        expected = hashlib.sha256((prev + data_str).encode("utf-8")).hexdigest()
        actual = record.get("proof", "")

        if expected != actual:
            return False, f"proof 不匹配: expected={expected[:16]}... actual={actual[:16]}..."
        return True, "ok"

    def verify_all(self) -> dict:
        """
        批量验证所有记忆的完整性。

        返回:
            {"total": int, "valid": int, "invalid": int, "failures": [str]}
        """
        total = 0
        valid = 0
        failures = []

        for fname in os.listdir(self.store_dir):
            if not fname.endswith(".json"):
                continue
            mid = fname[:-5]  # strip .json
            total += 1
            ok, msg = self.verify(mid)
            if ok:
                valid += 1
            else:
                failures.append(f"{mid}: {msg}")

        return {
            "total": total,
            "valid": valid,
            "invalid": total - valid,
            "failures": failures,
        }

    # ------------------------------------------------------------------
    # 会话管理
    # ------------------------------------------------------------------

    def list_sessions(self) -> list[dict]:
        """列出所有已注册会话及其记忆数量。"""
        result = []
        for sid, mem_ids in self._sessions.items():
            result.append({
                "sid": sid,
                "memory_count": len(mem_ids),
                "latest_proof": self._last_proof.get(sid, "")[:16] + "...",
            })
        return result

    def load_by_sid(self, sid: str) -> list[dict]:
        """加载特定会话的所有记忆。"""
        records = []
        for fname in os.listdir(self.store_dir):
            if not fname.endswith(".json"):
                continue
            path = os.path.join(self.store_dir, fname)
            try:
                with open(path, "r", encoding="utf-8") as f:
                    record = json.load(f)
            except (json.JSONDecodeError, IOError):
                continue
            if record.get("sid") == sid:
                records.append(record)
        records.sort(key=lambda r: r.get("timestamp", 0))
        return records

    # ------------------------------------------------------------------
    # 清理
    # ------------------------------------------------------------------

    def clear(self, sid: Optional[str] = None):
        """清空指定会话或全部记忆数据。"""
        if sid:
            for mid in self._sessions.get(sid, []):
                path = os.path.join(self.store_dir, f"{mid}.json")
                if os.path.exists(path):
                    os.remove(path)
            self._sessions.pop(sid, None)
            self._last_proof.pop(sid, None)
        else:
            for fname in os.listdir(self.store_dir):
                if fname.endswith(".json"):
                    os.remove(os.path.join(self.store_dir, fname))
            self._sessions.clear()
            self._last_proof.clear()
