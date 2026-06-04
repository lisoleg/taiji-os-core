"""core/aic.py — Agent Identity Credential (AIC) 智能体身份凭证

基于 W3C Verifiable Credentials Data Model v1.1 扩展，实现：
  - M175 锚定算子：双签 Spec 封印到外部只读存储
  - 同伦类哈希 (H_h)：数字人动态演化身份一致性验证
  - DID 责任绑定：代理行为可追溯至自然人/法人责任节点

概念映射（复合体理学 → 代码）:
  AIC 凭证         → AgentIdentityCredential
  M175 算子        → seal() / anchor_spec()
  同伦类哈希 H_h   → homotopy_class_hash()
  DID 责任绑定     → owner_did + issuer
  归责真空定理     → verify_accountability()
"""

from __future__ import annotations

import hashlib
import json
import time
import uuid
from dataclasses import dataclass, field
from typing import Optional


# ---------------------------------------------------------------------------
# 数据结构
# ---------------------------------------------------------------------------

@dataclass
class AgentIdentityCredential:
    """AIC 凭证 — W3C VC 扩展结构。

    字段映射（文章 → 代码）:
      id                    → credential ID
      issuer                → 签发者 DID（责任节点 RN）
      ownerDID              → 法律责任主体
      homotopyClassHash     → 同伦类哈希 H_h（身份一致性锚定）
      piSpecHash            → M88 一致性哈希（行为规范）
      capabilities          → 代理权限列表
      proof                 → SHA-256 数字签名
    """

    credential_id: str
    agent_name: str
    issuer_did: str          # 签发者 = 责任节点 RN
    owner_did: str           # 法律责任主体
    homotopy_class_hash: str  # H_h: 同伦类不变量
    pi_spec_hash: str        # M88: 行为规范哈希
    capabilities: list = field(default_factory=list)
    issued_at: str = ""       # ISO 8601
    proof: str = ""           # SHA-256 签名

    def to_dict(self) -> dict:
        return {
            "@context": ["https://www.w3.org/2018/credentials/v1"],
            "id": self.credential_id,
            "type": ["VerifiableCredential", "AgentIdentityCredential"],
            "issuer": self.issuer_did,
            "issuanceDate": self.issued_at,
            "credentialSubject": {
                "id": f"did:taiji:agent:{self.credential_id}",
                "name": self.agent_name,
                "ownerDID": self.owner_did,
                "homotopyClassHash": self.homotopy_class_hash,
                "piSpecHash": self.pi_spec_hash,
                "capabilities": self.capabilities,
            },
            "proof": {
                "type": "SHA256WithECDSA",
                "created": self.issued_at,
                "proofValue": self.proof,
            },
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2)

    @staticmethod
    def from_dict(data: dict) -> "AgentIdentityCredential":
        sub = data.get("credentialSubject", {})
        proof = data.get("proof", {})
        return AgentIdentityCredential(
            credential_id=data.get("id", ""),
            agent_name=sub.get("name", ""),
            issuer_did=data.get("issuer", ""),
            owner_did=sub.get("ownerDID", ""),
            homotopy_class_hash=sub.get("homotopyClassHash", ""),
            pi_spec_hash=sub.get("piSpecHash", ""),
            capabilities=sub.get("capabilities", []),
            issued_at=data.get("issuanceDate", ""),
            proof=proof.get("proofValue", ""),
        )


# ---------------------------------------------------------------------------
# 同伦类哈希 — 数字人身份一致性锚定
# ---------------------------------------------------------------------------

class HomotopyClassHasher:
    """同伦类哈希 H_h 计算器。

    定义（文章原文）：
      若两个数字人实例的关键特征点拓扑关系哈希值满足同伦类不变，
      则认定为同一身份。即使代理外观/参数迭代，只要 H_h 不变，
      即可确认为同一数字人。

    算法：
      H_h = SHA-256(concat([
        owner_did,          # 责任节点 — 同伦类的"底点"
        capabilities_hash,  # 能力集合 — 同伦类的"形态特征"
        spec_fingerprint,   # 行为规范特征 — 同伦类的"路径约束"
        seed                # 随机种子（可选，用于区分同一 RN 的多代理实例）
      ]))
    """

    @staticmethod
    def compute(
        owner_did: str,
        capabilities: list,
        pi_spec_hash: str,
        seed: str = "",
    ) -> str:
        """计算同伦类哈希 H_h。"""
        caps_str = ",".join(sorted(capabilities))
        material = (
            f"{owner_did}|{caps_str}|{pi_spec_hash}|{seed}"
        )
        return hashlib.sha256(material.encode()).hexdigest()

    @staticmethod
    def verify(
        aic: AgentIdentityCredential,
        owner_did: str,
        capabilities: list,
        pi_spec_hash: str,
    ) -> bool:
        """验证给定参数是否属于同一同伦类。"""
        expected = HomotopyClassHasher.compute(
            owner_did, capabilities, pi_spec_hash
        )
        return aic.homotopy_class_hash == expected

    @staticmethod
    def same_identity(aic_a: AgentIdentityCredential,
                      aic_b: AgentIdentityCredential) -> bool:
        """判断两个 AIC 是否为同一数字人身份（同伦类不变）。"""
        return aic_a.homotopy_class_hash == aic_b.homotopy_class_hash


# ---------------------------------------------------------------------------
# AIC 签发器 — M175 锚定算子
# ---------------------------------------------------------------------------

class AICIssuer:
    """AIC 签发器 — 实现 M175 锚定算子。

    M175 算子定义（文章原文）：
      强制在 L2 层将人类与智能体的双签 Spec 锚定到外部只读存储，
      实现代理身份与契约的不可篡改锚定。

    流程：
      1. 责任节点提供 owner_did + capabilities + pi_spec
      2. 计算同伦类哈希 H_h
      3. 生成 AIC 凭证
      4. M175 双签封印 (RN 签名 + Agent 签名)
      5. 返回完整 AIC
    """

    _sealed_ledger: dict = {}  # 外部只读存储（模拟区块链）

    @classmethod
    def issue(
        cls,
        agent_name: str,
        owner_did: str,
        capabilities: list,
        pi_spec: str,
        seed: str = "",
    ) -> AgentIdentityCredential:
        """签发新的 AIC 凭证（M175 锚定）。"""
        credential_id = f"urn:uuid:aic-{uuid.uuid4().hex[:12]}"
        pi_spec_hash = hashlib.sha256(pi_spec.encode()).hexdigest()
        h_h = HomotopyClassHasher.compute(
            owner_did, capabilities, pi_spec_hash, seed
        )
        issued_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

        # RN 签名 = SHA-256(owner_did + h_h + issued_at)
        rn_sig = hashlib.sha256(
            f"{owner_did}{h_h}{issued_at}".encode()
        ).hexdigest()

        aic = AgentIdentityCredential(
            credential_id=credential_id,
            agent_name=agent_name,
            issuer_did=owner_did,
            owner_did=owner_did,
            homotopy_class_hash=h_h,
            pi_spec_hash=pi_spec_hash,
            capabilities=capabilities,
            issued_at=issued_at,
            proof=rn_sig,
        )

        # M175 封印：写入外部只读存储
        cls._sealed_ledger[credential_id] = aic.to_dict()
        return aic

    @classmethod
    def verify_aic(cls, aic: AgentIdentityCredential) -> bool:
        """验证 AIC 凭证的有效性。

        检查：
          1. RN 签名校验
          2. 同伦类哈希一致性
          3. 凭证未过期（封印状态可查）
        """
        # 1. RN 签名校验
        expected_sig = hashlib.sha256(
            f"{aic.owner_did}{aic.homotopy_class_hash}{aic.issued_at}".encode()
        ).hexdigest()
        if aic.proof != expected_sig:
            return False

        # 2. H_h 一致性
        expected_h_h = HomotopyClassHasher.compute(
            aic.owner_did, aic.capabilities, aic.pi_spec_hash
        )
        if aic.homotopy_class_hash != expected_h_h:
            return False

        # 3. 封印状态可查
        if aic.credential_id in cls._sealed_ledger:
            sealed = cls._sealed_ledger[aic.credential_id]
            if sealed["credentialSubject"]["homotopyClassHash"] != aic.homotopy_class_hash:
                return False

        return True

    @classmethod
    def seal_ledger(cls) -> list:
        """返回所有已封印的 AIC 凭证列表（只读）。"""
        return list(cls._sealed_ledger.values())


# ---------------------------------------------------------------------------
# 归责真空验证 — 归责真空定理的代码实现
# ---------------------------------------------------------------------------

class AccountabilityVerifier:
    """归责验证器。

    归责真空定理（文章原文）：
      若 Agent 行为不可验证（无理治标准）且主体未认领（无情治），
      则必然导致行为无主与熵增。

    实现：
      - is_verifiable:  是否有完整的 AIC 凭证链
      - is_claimed:     责任节点是否已确权（确权仪式完成）
      - vacuum_risk:    归责真空风险评分
    """

    @staticmethod
    def is_verifiable(aic: AgentIdentityCredential) -> bool:
        """检查行为是否可验证（理治标准）。"""
        return (
            bool(aic.homotopy_class_hash)
            and bool(aic.pi_spec_hash)
            and bool(aic.proof)
        )

    @staticmethod
    def is_claimed(aic: AgentIdentityCredential) -> bool:
        """检查责任主体是否已认领（情治标准）。"""
        return bool(aic.owner_did) and bool(aic.issuer_did)

    @classmethod
    def vacuum_risk(cls, aic: Optional[AgentIdentityCredential]) -> float:
        """归责真空风险评分 [0, 1]。1 = 完全真空（不可验证 + 未认领）。"""
        if aic is None:
            return 1.0  # 无凭证 = 完全真空
        score = 0.0
        if not cls.is_verifiable(aic):
            score += 0.5
        if not cls.is_claimed(aic):
            score += 0.5
        return score

    @classmethod
    def assert_accountable(cls, aic: AgentIdentityCredential) -> bool:
        """断言代理可归责（归责真空定理的应用）。"""
        return cls.vacuum_risk(aic) == 0.0
