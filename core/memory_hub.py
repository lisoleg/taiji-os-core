"""core/memory_hub.py — MemoryHub: Walrus Memory 风格的共享记忆空间

Walrus Memory 核心理念映射：
  - Shared Memory Space → 多 TaijiSession 共享同一 MemoryHub
  - 记忆可在 Agent、应用、工作流之间自由流动
  - 不绑定单一运行时、会话或服务提供方

MemoryHub 提供：
  1. 会话注册 / 注销
  2. 跨会话 Continuation 查询（按 sid / keyword / time range）
  3. 记忆图谱（parent_kid 链遍历）
  4. 完整性批量验证
"""
from __future__ import annotations

import os
import json
import hashlib
from datetime import datetime, timezone
from typing import Optional

from core.continuation import Continuation


class MemoryHub:
    """
    Walrus-style 共享记忆空间。

    用法:
        hub = MemoryHub("shared_memory")
        hub.register("session-001")
        # ... 各 TaijiSession 通过 hub 存取 Continuation ...
        results = hub.search("量子计算")
        hub.verify_all()  # 批量验证所有快照完整性
    """

    def __init__(self, hub_dir: str = "shared_memory"):
        self.hub_dir = hub_dir
        os.makedirs(hub_dir, exist_ok=True)
        self._sessions: dict[str, dict] = {}  # sid → metadata
        self._index_path = os.path.join(hub_dir, "_index.json")
        self._load_index()

    # ------------------------------------------------------------------
    # 会话管理
    # ------------------------------------------------------------------

    def register(self, sid: str, metadata: Optional[dict] = None) -> str:
        """注册一个 TaijiSession 到共享记忆空间。返回 hub 内 sid。"""
        entry = {
            "sid": sid,
            "registered_at": datetime.now(timezone.utc).isoformat(),
            "metadata": metadata or {},
            "continuation_count": 0,
        }
        self._sessions[sid] = entry
        self._save_index()
        return sid

    def unregister(self, sid: str):
        """注销会话（不删除其 Continuation 快照）。"""
        self._sessions.pop(sid, None)
        self._save_index()

    def register_page_table(self, sid: str, page_table) -> None:
        """为会话注册页表（用于 USCS 内核集成）。
        
        Args:
            sid: 会话 ID
            page_table: PageTable 实例
        """
        if sid in self._sessions:
            self._sessions[sid]["page_table_registered"] = True
            self._sessions[sid]["page_count"] = page_table.page_count()
            self._save_index()

    def list_sessions(self) -> list[dict]:
        """列出所有已注册会话。"""
        return list(self._sessions.values())

    # ------------------------------------------------------------------
    # Continuation 存取（代理到各 session 的 snapshot_dir）
    # ------------------------------------------------------------------

    def store(
        self,
        sid: str,
        psi,
        env: dict,
        reason: str,
        snapshot_dir: str,
        parent_kid: Optional[str] = None,
    ) -> Continuation:
        """
        存储 Continuation 到 hub 管理的会话。
        自动链接 parent_kid 的 proof 链。
        """
        parent_proof = None
        if parent_kid:
            try:
                parent = Continuation.load(parent_kid, snapshot_dir)
                parent_proof = parent.proof
            except Exception:
                pass

        k = Continuation(
            sid=sid,
            psi=psi,
            env=env,
            reason=reason,
            snapshot_dir=snapshot_dir,
            parent_kid=parent_kid,
            parent_proof=parent_proof,
        )
        if sid in self._sessions:
            self._sessions[sid]["continuation_count"] += 1
        self._save_index()
        return k

    def load(self, kid: str, snapshot_dir: str = "snapshots") -> Continuation:
        """从 hub 加载 Continuation。"""
        return Continuation.load(kid, snapshot_dir)

    # ------------------------------------------------------------------
    # 跨会话搜索（Walrus Memory 的核心能力）
    # ------------------------------------------------------------------

    def search(
        self,
        keyword: str = "",
        sid: Optional[str] = None,
        snapshot_dir: str = "snapshots",
        limit: int = 20,
    ) -> list[dict]:
        """
        跨会话搜索 Continuation 快照。

        参数:
            keyword: 搜索关键词（匹配 reason / env 内容）
            sid: 限定会话（None = 搜索所有会话）
            snapshot_dir: 快照目录
            limit: 最大返回数
        """
        results = []
        search_dirs = [snapshot_dir]
        # 也搜其他已知会话的快照目录
        for s in self._sessions:
            alt_dir = os.path.join(self.hub_dir, s)
            if os.path.isdir(alt_dir) and alt_dir not in search_dirs:
                search_dirs.append(alt_dir)

        for d in search_dirs:
            if not os.path.isdir(d):
                continue
            for fname in os.listdir(d):
                if not fname.endswith(".json") or fname.startswith("_"):
                    continue
                try:
                    k = Continuation.load(fname.replace(".json", ""), d)
                except Exception:
                    continue

                if sid and k.sid != sid:
                    continue
                if keyword:
                    text = json.dumps(k.env, ensure_ascii=False) + k.reason
                    if keyword.lower() not in text.lower():
                        continue

                results.append(k.to_audit_dict())
                if len(results) >= limit:
                    return results
        return results

    def get_chain(self, kid: str, snapshot_dir: str = "snapshots") -> list[dict]:
        """
        遍历 proof 链：从给定 kid 回溯所有祖先快照。
        这是 Walrus Memory "记忆图谱" 的太极实现。
        """
        chain = []
        visited = set()
        current_kid = kid
        while current_kid and current_kid not in visited:
            visited.add(current_kid)
            try:
                k = Continuation.load(current_kid, snapshot_dir)
            except Exception:
                break
            chain.append(k.to_audit_dict())
            current_kid = k.parent_kid
        return chain

    # ------------------------------------------------------------------
    # 批量完整性验证
    # ------------------------------------------------------------------

    def verify_all(self, snapshot_dir: str = "snapshots") -> dict:
        """
        验证所有 Continuation 的完整性（Walrus-style batch verify）。

        返回: {"total": N, "passed": M, "failed": [...], "skipped": K}
        """
        report = {"total": 0, "passed": 0, "failed": [], "skipped": 0}
        if not os.path.isdir(snapshot_dir):
            return report

        # 构建 proof map: kid → proof
        proof_map: dict[str, str] = {}
        for fname in os.listdir(snapshot_dir):
            if not fname.endswith(".json") or fname.startswith("_"):
                continue
            try:
                k = Continuation.load(fname.replace(".json", ""), snapshot_dir)
            except Exception:
                continue
            proof_map[k.kid] = k.proof

        # 逐条验证
        for kid, stored_proof in proof_map.items():
            report["total"] += 1
            try:
                k = Continuation.load(kid, snapshot_dir)
            except Exception:
                report["failed"].append({"kid": kid, "error": "load failed"})
                continue

            # 找到父 proof
            parent_proof = None
            if k.parent_kid and k.parent_kid in proof_map:
                parent_proof = proof_map[k.parent_kid]

            if k.verify(parent_proof):
                report["passed"] += 1
            else:
                report["failed"].append(
                    {
                        "kid": kid,
                        "expected_proof": k._compute_proof(parent_proof or "")[:16],
                        "stored_proof": stored_proof[:16],
                    }
                )

        return report

    # ------------------------------------------------------------------
    # 索引持久化
    # ------------------------------------------------------------------

    def _load_index(self):
        if os.path.exists(self._index_path):
            with open(self._index_path, "r", encoding="utf-8") as f:
                self._sessions = json.load(f)

    def _save_index(self):
        with open(self._index_path, "w", encoding="utf-8") as f:
            json.dump(self._sessions, f, ensure_ascii=False, indent=2)

    def __repr__(self):
        return f"<MemoryHub sessions={len(self._sessions)} dir={self.hub_dir}>"
