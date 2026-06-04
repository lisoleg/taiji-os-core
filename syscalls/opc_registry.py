"""syscalls/opc_registry.py — OPC 注册表 (人人即法人)

OPC（One Person, One Corporation / 人人即法人）制度（文章原文）:
  将传统「自然人→法人→AI Agent」三级结构压扁为
  「责任节点网络 + 受权代理附件」的扁平化责任体系。

核心能力:
  - 责任节点注册 (RN Registry)
  - AIC 凭证签发与验证
  - 临时法人人格授予 (AIC 签署)
  - 资产池管理
  - 代理权限管理

OPC 注册表是 AIC 凭证的生命周期管理中心。
"""

from __future__ import annotations

import hashlib
import json
import time
import uuid
from dataclasses import dataclass, field
from typing import Optional

from core.aic import AgentIdentityCredential, AICIssuer, HomotopyClassHasher


# ---------------------------------------------------------------------------
# 责任节点
# ---------------------------------------------------------------------------

@dataclass
class ResponsibleNode:
    """责任节点 (RN) — OPC 体系中的自然人/法人。

    属性:
        did           : 去中心化标识符
        name          : 自然人/法人名称
        type          : "natural" | "legal"
        asset_pool    : 资产池余额
        aic_issued    : 已签发的 AIC 凭证数
        created_at    : 注册时间
    """
    did: str
    name: str
    type: str = "natural"  # natural | legal
    asset_pool: float = 0.0
    aic_issued: int = 0
    created_at: str = ""
    active: bool = True
    metadata: dict = field(default_factory=dict)

    def __post_init__(self):
        if not self.created_at:
            self.created_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    def to_dict(self) -> dict:
        return {
            "did": self.did,
            "name": self.name,
            "type": self.type,
            "assetPool": self.asset_pool,
            "aicIssued": self.aic_issued,
            "createdAt": self.created_at,
            "active": self.active,
        }


# ---------------------------------------------------------------------------
# 临时法人人格 (AIC)
# ---------------------------------------------------------------------------

@dataclass
class TemporaryLegalPersonality:
    """临时法人人格 — OPC 体系中自然人签署 AIC 后获得。

    权利:
      - 接包
      - 收款

    义务:
      - 义务上限 = 资产池余额
      - Agent 行为追责回溯至 RN
    """
    rn_did: str
    aic_id: str
    asset_pool_cap: float   # 资产池上限
    active: bool = True
    issued_at: str = ""

    def __post_init__(self):
        if not self.issued_at:
            self.issued_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


# ---------------------------------------------------------------------------
# OPC 注册表
# ---------------------------------------------------------------------------

class OPCRegistry:
    """OPC 注册表 — 责任节点与 AIC 凭证的生命周期管理中心。

    操作:
      - register_node() : 注册责任节点
      - issue_aic()     : 签发 AIC 凭证
      - revoke_aic()    : 吊销 AIC 凭证
      - list_agents()   : 列出 RN 下的所有代理
      - deposit()       : 资产池充值
      - withdraw()      : 资产池提取
    """

    _nodes: dict = {}          # did → ResponsibleNode
    _aic_index: dict = {}      # credential_id → AgentIdentityCredential
    _rn_agents: dict = {}      # rn_did → [aic_ids]
    _personalities: dict = {}  # rn_did → TemporaryLegalPersonality

    # ---------- RN 注册 ----------

    @classmethod
    def register_node(
        cls,
        name: str,
        node_type: str = "natural",
        initial_assets: float = 0.0,
        metadata: dict = None,
    ) -> ResponsibleNode:
        """注册责任节点。"""
        did = f"did:taiji:rn:{uuid.uuid4().hex[:12]}"
        node = ResponsibleNode(
            did=did,
            name=name,
            type=node_type,
            asset_pool=initial_assets,
            metadata=metadata or {},
        )
        cls._nodes[did] = node
        return node

    @classmethod
    def get_node(cls, did: str) -> Optional[ResponsibleNode]:
        return cls._nodes.get(did)

    @classmethod
    def list_nodes(cls) -> list:
        return [n.to_dict() for n in cls._nodes.values()]

    # ---------- AIC 签发 ----------

    @classmethod
    def issue_aic(
        cls,
        rn_did: str,
        agent_name: str,
        capabilities: list,
        pi_spec: str,
        seed: str = "",
    ) -> Optional[AgentIdentityCredential]:
        """为责任节点签发 AIC 凭证。

        前提:
          1. RN 已注册
          2. RN 未冻结
        """
        node = cls._nodes.get(rn_did)
        if node is None:
            return None
        if not node.active:
            return None

        aic = AICIssuer.issue(
            agent_name=agent_name,
            owner_did=rn_did,
            capabilities=capabilities,
            pi_spec=pi_spec,
            seed=seed,
        )

        cls._aic_index[aic.credential_id] = aic

        if rn_did not in cls._rn_agents:
            cls._rn_agents[rn_did] = []
        cls._rn_agents[rn_did].append(aic.credential_id)

        node.aic_issued += 1
        return aic

    # ---------- 临时法人人格 ----------

    @classmethod
    def grant_personality(
        cls,
        rn_did: str,
        asset_cap: float,
    ) -> Optional[TemporaryLegalPersonality]:
        """授予临时法人人格 — 自然人签署 AIC 后获得经济能动性。"""
        node = cls._nodes.get(rn_did)
        if node is None:
            return None
        if not node.active:
            return None
        if asset_cap > node.asset_pool:
            return None  # 上限不能超过资产池

        personality = TemporaryLegalPersonality(
            rn_did=rn_did,
            aic_id=f"aic:personality:{uuid.uuid4().hex[:8]}",
            asset_pool_cap=asset_cap,
        )
        cls._personalities[rn_did] = personality
        return personality

    @classmethod
    def get_personality(cls, rn_did: str) -> Optional[TemporaryLegalPersonality]:
        return cls._personalities.get(rn_did)

    # ---------- AIC 查询/吊销 ----------

    @classmethod
    def get_aic(cls, credential_id: str) -> Optional[AgentIdentityCredential]:
        return cls._aic_index.get(credential_id)

    @classmethod
    def list_agents(cls, rn_did: str) -> list:
        """列出 RN 下的所有代理。"""
        aic_ids = cls._rn_agents.get(rn_did, [])
        return [cls._aic_index[aid].to_dict() for aid in aic_ids if aid in cls._aic_index]

    @classmethod
    def revoke_aic(cls, credential_id: str, rn_did: str) -> bool:
        """吊销 AIC 凭证（仅 RN 本人可吊销）。"""
        aic = cls._aic_index.get(credential_id)
        if aic is None:
            return False
        if aic.owner_did != rn_did:
            return False

        del cls._aic_index[credential_id]
        if rn_did in cls._rn_agents:
            cls._rn_agents[rn_did] = [
                aid for aid in cls._rn_agents[rn_did] if aid != credential_id
            ]
        return True

    # ---------- 资产池 ----------

    @classmethod
    def deposit(cls, rn_did: str, amount: float) -> bool:
        """资产池充值。"""
        node = cls._nodes.get(rn_did)
        if node is None:
            return False
        node.asset_pool += amount
        return True

    @classmethod
    def withdraw(cls, rn_did: str, amount: float) -> bool:
        """从资产池提取（不能超余额）。"""
        node = cls._nodes.get(rn_did)
        if node is None:
            return False
        if amount > node.asset_pool:
            return False
        node.asset_pool -= amount
        return True

    # ---------- 统计 ----------

    @classmethod
    def stats(cls) -> dict:
        return {
            "total_nodes": len(cls._nodes),
            "total_aic_issued": len(cls._aic_index),
            "total_personalities": len(cls._personalities),
            "active_nodes": sum(1 for n in cls._nodes.values() if n.active),
        }
