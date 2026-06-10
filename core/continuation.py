"""core/continuation.py — Continuation v2: Walrus Memory 风格的完整性证明链

Walrus Memory 核心理念映射：
  1. Integrity Proofs → SHA-256 哈希链，每条快照携带可验证完整性证明
  2. Portable Memory  → 支持跨会话、跨 Agent 的快照共享与验证
  3. Parent Chain    → 每个 Continuation 链接到父快照，形成不可篡改链

v2 新增:
  - proof: SHA-256(prev_proof || payload_hash) 链式完整性证明
  - parent_kid: 父快照 ID，构建记忆图谱
  - verify(): 独立验证快照未被篡改
  - payload_hash: 载荷哈希，用于审计
  - 兼容 v1 快照（无 proof 字段时跳过验证）
"""
import os
import json
import uuid
import hashlib
from datetime import datetime, timezone
from typing import Optional

import numpy as np


class Continuation:
    """
    Continuation (k): AGI 进程的可序列化快照 — Walrus Memory 风格。

    v2 特性:
      - 每条快照携带 SHA-256 完整性证明（proof 链）
      - 支持 parent_kid 构建记忆图谱
      - verify() 可独立验证，无需信任存储层
      - 向后兼容 v1 快照（无 proof 字段时降级为 trust-on-load）
    """

    def __init__(
        self,
        sid: str,
        psi: np.ndarray,
        env: dict,
        reason: str,
        snapshot_dir: str = "snapshots",
        parent_kid: Optional[str] = None,
        parent_proof: Optional[str] = None,
    ):
        self.kid = str(uuid.uuid4())[:8]
        self.sid = sid
        self.psi = psi
        self.env = env
        self.reason = reason
        self.ts = datetime.now(timezone.utc).isoformat()
        self.snapshot_dir = snapshot_dir
        self.parent_kid = parent_kid

        # ---- v2: Walrus-style integrity proof ----
        self.payload_hash = self._compute_payload_hash()
        self.proof = self._compute_proof(parent_proof or "")
        # ----                                   ----

        self._save()

    # ------------------------------------------------------------------
    # Integrity Proof (Walrus Memory style)
    # ------------------------------------------------------------------

    def _compute_payload_hash(self) -> str:
        """计算载荷 SHA-256 哈希（不含 proof 自身，避免循环）。"""
        payload = json.dumps(
            {
                "kid": self.kid,
                "sid": self.sid,
                "psi": self.psi.tolist(),
                "env": self.env,
                "reason": self.reason,
                "ts": self.ts,
                "parent_kid": self.parent_kid,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def _compute_proof(self, parent_proof: str) -> str:
        """proof = SHA-256(parent_proof || payload_hash)"""
        seed = f"{parent_proof}||{self.payload_hash}"
        return hashlib.sha256(seed.encode("utf-8")).hexdigest()

    def verify(self, parent_proof: Optional[str] = None) -> bool:
        """
        独立验证快照完整性。
        - 检查 payload_hash 是否匹配——证明内容未被篡改
        - 检查 proof 是否匹配 parent_proof——证明链未断裂
        - 不依赖任何外部信任，纯密码学验证
        """
        # 重新计算载荷哈希
        recomputed = self._compute_payload_hash()
        if recomputed != self.payload_hash:
            return False
        # 重新计算 proof
        expected_proof = self._compute_proof(parent_proof or "")
        if expected_proof != self.proof:
            return False
        return True

    # ------------------------------------------------------------------
    # 持久化
    # ------------------------------------------------------------------

    def _save(self):
        os.makedirs(self.snapshot_dir, exist_ok=True)
        path = os.path.join(self.snapshot_dir, f"{self.kid}.json")
        payload = {
            "kid": self.kid,
            "sid": self.sid,
            "psi": self.psi.tolist(),
            "env": self.env,
            "reason": self.reason,
            "ts": self.ts,
            # ---- v2 fields ----
            "parent_kid": self.parent_kid,
            "payload_hash": self.payload_hash,
            "proof": self.proof,
            "migration_meta": getattr(self, "_migration_meta", None),
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, kid: str, snapshot_dir: str = "snapshots") -> "Continuation":
        path = os.path.join(snapshot_dir, f"{kid}.json")
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        obj = object.__new__(cls)
        obj.kid = data["kid"]
        obj.sid = data["sid"]
        obj.psi = np.array(data["psi"])
        obj.env = data["env"]
        obj.reason = data["reason"]
        obj.ts = data["ts"]
        obj.snapshot_dir = snapshot_dir
        # ---- v2 fields (向后兼容) ----
        obj.parent_kid = data.get("parent_kid")
        obj.payload_hash = data.get("payload_hash", "")
        obj.proof = data.get("proof", "")
        obj._migration_meta = data.get("migration_meta")
        return obj

    # ------------------------------------------------------------------
    # 查询 / 审计
    # ------------------------------------------------------------------

    def to_audit_dict(self) -> dict:
        """返回审计视图：kid, proof, parent, ts."""
        return {
            "kid": self.kid,
            "sid": self.sid,
            "proof": self.proof[:16] + "..." if self.proof else None,
            "parent_kid": self.parent_kid,
            "reason": self.reason,
            "ts": self.ts,
        }

    @classmethod
    def list_all(cls, snapshot_dir: str = "snapshots") -> list["Continuation"]:
        """列出所有 Continuation 快照。"""
        if not os.path.isdir(snapshot_dir):
            return []
        result = []
        for fname in sorted(os.listdir(snapshot_dir)):
            if fname.endswith(".json"):
                kid = fname.replace(".json", "")
                try:
                    result.append(cls.load(kid, snapshot_dir))
                except Exception:
                    continue
        return result

    def __repr__(self):
        proof_short = self.proof[:8] if self.proof else "N/A"
        return (
            f"<Continuation kid={self.kid} sid={self.sid} "
            f"proof={proof_short}... parent={self.parent_kid}>"
        )
