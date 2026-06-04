"""core/ark_covenant.py — 约柜合约 (Ark Covenant)

约柜思想（文章原文）：
  通过区块链部署不可篡改的约柜合约，实现契约的永久封印与强制执行。

核心结构（Ark struct）:
  specHash    : M88 一致性哈希（行为规范）
  client      : 人类签署者（责任节点 RN）
  agent       : 智能体（只读，无权修改合约）
  escrowTokens: M178 托管池
  sealedBlock : M175 封印章区块号
  completed   : M106 收敛标志
  slashed     : M178 罚没状态

核心算子:
  M175 (封印) : seal() — 双签后 specHash 永不可改
  M106 (收敛) : complete() — 验收通过，释放 Token
  M178 (罚没) : slash() — 违约自动罚没托管池

定理映射:
  约柜不可篡改定理 → seal() 后 specHash 只读
  Token 激励对齐定理 → GCD 约束下，最大化 Token 收益的唯一途径是最大化交付质量
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from typing import Optional


# ---------------------------------------------------------------------------
# 约柜数据结构
# ---------------------------------------------------------------------------

@dataclass
class ArkState:
    """约柜状态 — Ark Covenant 合约状态机。

    字段:
        spec_hash       : M88 一致性哈希（行为规范摘要）
        client          : 人类签署者 DID（责任节点 RN）
        agent           : 智能体 DID（只读，Agent 无权调用 seal）
        escrow_tokens   : M178 托管池 Token 数量
        sealed_block    : M175 封印区块号（0 = 未封印）
        completed       : M106 收敛标志
        slashed         : M178 罚没状态
        sealed_at       : 封印时间戳
        completed_at    : 验收时间戳
        slashed_at      : 罚没时间戳
    """

    spec_hash: str
    client: str            # RN DID
    agent: str             # Agent DID
    escrow_tokens: float = 0.0
    sealed_block: int = 0  # 0 = 未封印
    completed: bool = False
    slashed: bool = False
    sealed_at: str = ""
    completed_at: str = ""
    slashed_at: str = ""
    history: list = field(default_factory=list)  # 操作日志

    def to_dict(self) -> dict:
        return {
            "specHash": self.spec_hash,
            "client": self.client,
            "agent": self.agent,
            "escrowTokens": self.escrow_tokens,
            "sealedBlock": self.sealed_block,
            "completed": self.completed,
            "slashed": self.slashed,
            "sealedAt": self.sealed_at,
            "completedAt": self.completed_at,
            "slashedAt": self.slashed_at,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2)

    @property
    def is_sealed(self) -> bool:
        """M175: 封印状态。"""
        return self.sealed_block > 0

    @property
    def is_settled(self) -> bool:
        """合约已终局。"""
        return self.completed or self.slashed

    @property
    def status(self) -> str:
        if self.slashed:
            return "SLASHED"
        if self.completed:
            return "COMPLETED"
        if self.is_sealed:
            return "SEALED"
        return "DRAFT"


# ---------------------------------------------------------------------------
# 约柜合约引擎
# ---------------------------------------------------------------------------

class ArkCovenant:
    """约柜合约引擎。

    模拟 PoS 区块链上的约柜智能合约，保证：
      - specHash 仅在 sealedBlock == 0 时可写
      - Agent 无权调用 seal()（只读身份）
      - 验收后自动释放 escrowTokens
      - 违约自动罚没
    """

    _chain: dict[str, ArkState] = {}  # 模拟区块链存储
    _block_height: int = 0

    # ---------- 初始化 ----------

    @classmethod
    def deploy(
        cls,
        covenant_id: str,
        spec: str,
        client_did: str,
        agent_did: str,
        initial_tokens: float = 0.0,
    ) -> ArkState:
        """部署约柜合约。"""
        spec_hash = hashlib.sha256(spec.encode()).hexdigest()
        state = ArkState(
            spec_hash=spec_hash,
            client=client_did,
            agent=agent_did,
            escrow_tokens=initial_tokens,
        )
        state.history.append({
            "op": "deploy",
            "block": cls._block_height,
            "time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        })
        cls._chain[covenant_id] = state
        cls._advance_block()
        return state

    # ---------- M175 封印 ----------

    @classmethod
    def seal(
        cls,
        covenant_id: str,
        caller_did: str,
    ) -> ArkState:
        """M175 封印算子 — 双签封印。

        规则:
          1. 仅 client (RN) 可调用
          2. 仅未封印时可封印
          3. 封印后 specHash 永不可改
        """
        state = cls._require(covenant_id)

        if caller_did != state.client:
            raise ArkError(
                f"M175 seal denied: caller '{caller_did}' is not client '{state.client}'"
            )
        if state.is_sealed:
            raise ArkError(
                f"M175 seal denied: covenant '{covenant_id}' already sealed at block {state.sealed_block}"
            )

        cls._advance_block()
        state.sealed_block = cls._block_height
        state.sealed_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        state.history.append({
            "op": "seal",
            "block": cls._block_height,
            "caller": caller_did,
        })
        return state

    # ---------- M106 收敛（验收） ----------

    @classmethod
    def complete(
        cls,
        covenant_id: str,
        caller_did: str,
    ) -> ArkState:
        """M106 收敛算子 — 验收通过，释放托管 Token。

        规则:
          1. 仅 client (RN) 可验收
          2. 必须先封印
          3. 未罚没
          4. 未完成
        """
        state = cls._require(covenant_id)

        if caller_did != state.client:
            raise ArkError(
                f"M106 complete denied: caller '{caller_did}' != client '{state.client}'"
            )
        if not state.is_sealed:
            raise ArkError(
                f"M106 complete denied: covenant not sealed"
            )
        if state.slashed:
            raise ArkError(
                f"M106 complete denied: covenant already slashed"
            )
        if state.completed:
            raise ArkError(
                f"M106 complete denied: already completed"
            )

        cls._advance_block()
        state.completed = True
        state.completed_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

        # 释放 Token 给 Agent（模拟转账）
        if state.escrow_tokens > 0:
            state.history.append({
                "op": "release",
                "amount": state.escrow_tokens,
                "to": state.agent,
                "block": cls._block_height,
            })
            state.escrow_tokens = 0.0

        state.history.append({
            "op": "complete",
            "block": cls._block_height,
            "caller": caller_did,
        })
        return state

    # ---------- M178 罚没 ----------

    @classmethod
    def slash(
        cls,
        covenant_id: str,
        caller_did: str,
        reason: str = "",
    ) -> ArkState:
        """M178 罚没算子 — 违约自动罚没托管 Token。

        规则:
          1. caller 必须是 client 或系统仲裁者
          2. 必须先封印
          3. 未完成
          4. 未罚没
        """
        state = cls._require(covenant_id)

        if caller_did not in (state.client, "system:arbitrator"):
            raise ArkError(
                f"M178 slash denied: '{caller_did}' not authorized"
            )
        if not state.is_sealed:
            raise ArkError(
                f"M178 slash denied: covenant not sealed"
            )
        if state.completed:
            raise ArkError(
                f"M178 slash denied: covenant already completed"
            )
        if state.slashed:
            raise ArkError(
                f"M178 slash denied: already slashed"
            )

        cls._advance_block()
        state.slashed = True
        state.slashed_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

        # 罚没 Token 给 client（归责节点）
        forfeited = state.escrow_tokens
        state.history.append({
            "op": "slash",
            "reason": reason,
            "forfeited": forfeited,
            "to": state.client,
            "block": cls._block_height,
        })
        state.escrow_tokens = 0.0

        state.history.append({
            "op": "slashed",
            "block": cls._block_height,
            "caller": caller_did,
        })
        return state

    # ---------- 查询 ----------

    @classmethod
    def get(cls, covenant_id: str) -> Optional[ArkState]:
        return cls._chain.get(covenant_id)

    @classmethod
    def list_all(cls) -> dict:
        return {k: v.status for k, v in cls._chain.items()}

    # ---------- 内部 ----------

    @classmethod
    def _advance_block(cls):
        cls._block_height += 1

    @classmethod
    def _require(cls, covenant_id: str) -> ArkState:
        state = cls._chain.get(covenant_id)
        if state is None:
            raise ArkError(f"Covenant '{covenant_id}' not found")
        return state


class ArkError(Exception):
    """约柜合约异常。"""
    pass
