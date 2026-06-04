"""core/continuation.py — Continuation v2: AGI 进程可序列化快照 with Walrus Memory proof 链

v2 新增（Walrus Memory 概念映射）：
  - Portable Memory   → 跨节点可迁移的 JSON 快照
  - Integrity Proofs  → SHA-256 哈希链验证
  - Shared Memory     → parent_kid 记忆图谱
  - verify()          → 单条完整性验证
  - load_all()        → 批量加载

每条 Continuation 包含：
  - kid       : 唯一标识（uuid8）
  - sid       : 来源会话
  - psi       : ψ 向量快照
  - env       : 环境状态
  - reason    : 中断原因
  - ts        : 时间戳
  - proof     : SHA-256 完整性证明（链式）
  - parent_kid: 父 Continuation ID（记忆图谱）
"""

import hashlib
import json
import os
import uuid
from datetime import datetime, timezone
from typing import Optional, Tuple

import numpy as np


class Continuation:
    """
    Continuation v2 (k): AGI 进程的可序列化快照。

    与 v1 的区别：
      - 新增 proof（SHA-256 链上完整性证明）
      - 新增 parent_kid（记忆图谱链接）
      - 新增 verify()、load_all() 方法
    """

    def __init__(
        self,
        sid: str,
        psi: np.ndarray,
        env: dict,
        reason: str,
        snapshot_dir: str = "snapshots",
        parent_kid: Optional[str] = None,
    ):
        self.kid = str(uuid.uuid4())[:8]
        self.sid = sid
        self.psi = psi
        self.env = env
        self.reason = reason
        self.ts = datetime.now(timezone.utc).isoformat()
        self.snapshot_dir = snapshot_dir
        self.parent_kid = parent_kid

        # 计算 proof：SHA-256(prev_proof + data)
        prev_proof = ""
        if parent_kid:
            parent_path = os.path.join(snapshot_dir, f"{parent_kid}.json")
            if os.path.exists(parent_path):
                try:
                    with open(parent_path, "r", encoding="utf-8") as f:
                        parent_data = json.load(f)
                    prev_proof = parent_data.get("proof", "")
                except (json.JSONDecodeError, IOError):
                    pass

        data_str = json.dumps(env, ensure_ascii=False, sort_keys=True)
        chain_input = prev_proof + data_str
        self.proof = hashlib.sha256(chain_input.encode("utf-8")).hexdigest()

        self._save()

    def _save(self):
        """持久化到磁盘。"""
        os.makedirs(self.snapshot_dir, exist_ok=True)
        path = os.path.join(self.snapshot_dir, f"{self.kid}.json")
        payload = {
            "kid": self.kid,
            "sid": self.sid,
            "psi": self.psi.tolist(),
            "env": self.env,
            "reason": self.reason,
            "ts": self.ts,
            "proof": self.proof,
            "parent_kid": self.parent_kid,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

    # ------------------------------------------------------------------
    # 验证
    # ------------------------------------------------------------------

    def verify(self) -> Tuple[bool, str]:
        """
        验证此 Continuation 的完整性。

        返回:
            (is_valid, message)

        验证逻辑：
          1. 从磁盘读取 parent 的 proof
          2. 重新计算 SHA-256(parent_proof + env_data)
          3. 对比存储的 proof
        """
        path = os.path.join(self.snapshot_dir, f"{self.kid}.json")
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError, IOError) as e:
            return False, f"读取快照失败: {e}"

        prev_proof = ""
        stored_parent = data.get("parent_kid")
        if stored_parent:
            parent_path = os.path.join(self.snapshot_dir, f"{stored_parent}.json")
            if os.path.exists(parent_path):
                try:
                    with open(parent_path, "r", encoding="utf-8") as f:
                        pdata = json.load(f)
                    prev_proof = pdata.get("proof", "")
                except (json.JSONDecodeError, IOError):
                    pass

        env_str = json.dumps(data.get("env", {}), ensure_ascii=False, sort_keys=True)
        expected = hashlib.sha256((prev_proof + env_str).encode("utf-8")).hexdigest()
        actual = data.get("proof", "")

        if expected != actual:
            return False, (
                f"proof 不匹配: expected={expected[:16]}... actual={actual[:16]}..."
            )
        return True, "ok"

    # ------------------------------------------------------------------
    # 加载
    # ------------------------------------------------------------------

    @classmethod
    def load(cls, kid: str, snapshot_dir: str = "snapshots") -> "Continuation":
        """从磁盘加载单个 Continuation。"""
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
        obj.proof = data.get("proof", "")
        obj.parent_kid = data.get("parent_kid")
        return obj

    @classmethod
    def load_all(cls, snapshot_dir: str = "snapshots") -> list:
        """批量加载所有 Continuations（按时间排序）。"""
        result = []
        if not os.path.exists(snapshot_dir):
            return result
        for fname in os.listdir(snapshot_dir):
            if not fname.endswith(".json"):
                continue
            kid = fname[:-5]
            try:
                c = cls.load(kid, snapshot_dir)
                result.append(c)
            except (json.JSONDecodeError, IOError, KeyError):
                continue
        result.sort(key=lambda c: c.ts, reverse=True)
        return result

    # ------------------------------------------------------------------
    # 显示
    # ------------------------------------------------------------------

    def __repr__(self):
        pp = self.proof[:8] if self.proof else "none"
        parent = f" parent={self.parent_kid}" if self.parent_kid else ""
        return (
            f"<Continuation kid={self.kid} sid={self.sid} "
            f"reason={self.reason} proof={pp}...{parent}>"
        )
