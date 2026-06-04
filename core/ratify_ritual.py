"""core/ratify_ritual.py — 确权仪式 (Ratify Ritual)

确权仪式定义（文章原文）：
  部署 AI 代理前的强制流程，通过「Plan → Consult → Ratify」三步流程
  将主体意识强度写入凭证，避免事后推诿。

三步流程:
  Step 1 - Plan (集中):    明确资产池、用途、权限边界，生成初始 Spec
  Step 2 - Consult (到群众中去): 法务/安全评审 Spec 合规性与风险
  Step 3 - Ratify (再集中): 对 soul.md 哈希、同伦类哈希进行数字签名，注入 Runtime

确权仪式降推诿定理：
  完成确权仪式的主体，声称「不知 Agent 越界」在认知与法理上均不成立。

RatifyRitual 是三旋治理的「情治」落地实现：
  通过结构化的三步流程，让责任主体在认知层面明确认领代理，
  消除「我不知道」、「不是我授权的」等推诿心理。
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Optional


# ---------------------------------------------------------------------------
# 仪式阶段
# ---------------------------------------------------------------------------

class RitualPhase(Enum):
    PENDING = "pending"        # 未开始
    PLAN = "plan"              # 第一步：计划
    CONSULT = "consult"        # 第二步：评审
    RATIFY = "ratify"          # 第三步：封印
    COMPLETE = "complete"      # 完成
    ABORTED = "aborted"        # 中止


# ---------------------------------------------------------------------------
# Spec 定义
# ---------------------------------------------------------------------------

@dataclass
class AgentSpec:
    """代理行为规范 (Spec) — 确权仪式的核心文档。

    Plan 阶段产出，Consult 阶段评审，Ratify 阶段封印。
    """
    agent_name: str
    owner_did: str
    purpose: str              # 代理用途
    capabilities: list = field(default_factory=list)
    asset_pool: float = 0.0   # 资产池额度
    boundaries: list = field(default_factory=list)  # 权限边界
    constraints: list = field(default_factory=list)  # 约束条件
    risks: list = field(default_factory=list)        # 已知风险
    version: int = 1

    def to_text(self) -> str:
        """生成 Spec 文本（用于哈希锚定）。"""
        lines = [
            f"# Agent Spec: {self.agent_name}",
            f"Owner: {self.owner_did}",
            f"Purpose: {self.purpose}",
            f"Version: {self.version}",
            f"Asset Pool: {self.asset_pool}",
            "",
            "## Capabilities",
        ]
        for c in self.capabilities:
            lines.append(f"  - {c}")
        lines.append("")
        lines.append("## Boundaries")
        for b in self.boundaries:
            lines.append(f"  - {b}")
        lines.append("")
        lines.append("## Constraints")
        for c in self.constraints:
            lines.append(f"  - {c}")
        return "\n".join(lines)

    def to_soul_md(self) -> str:
        """生成 SOUL.md 格式。"""
        lines = [
            f"# SOUL.md — {self.agent_name}",
            f"Owner DID: {self.owner_did}",
            f"Purpose: {self.purpose}",
            f"Asset Pool Cap: {self.asset_pool}",
            "",
            "## Identity",
            f"- Agent Name: {self.agent_name}",
            f"- Version: {self.version}",
            "",
            "## Capabilities",
        ]
        for c in self.capabilities:
            lines.append(f"- {c}")
        lines.append("")
        lines.append("## Boundaries (Hard Limits)")
        for b in self.boundaries:
            lines.append(f"- {b}")
        lines.append("")
        lines.append("## Constraints")
        for c in self.constraints:
            lines.append(f"- {c}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# 评审意见
# ---------------------------------------------------------------------------

@dataclass
class ConsultOpinion:
    """Consult 阶段的评审意见。"""
    reviewer: str            # 评审人 DID
    approved: bool           # 是否通过
    severity: str = "info"   # info | warning | critical
    comments: str = ""
    conditions: list = field(default_factory=list)  # 附加条件
    timestamp: str = ""

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


# ---------------------------------------------------------------------------
# 确权仪式引擎
# ---------------------------------------------------------------------------

class RatifyRitual:
    """确权仪式引擎。

    三步流程:
      1. plan(spec)     → 产出 AgentSpec，进入 PLAN 阶段
      2. consult(...)   → 评审 Spec，收集意见
      3. ratify(signer) → 数字签名，封印 Spec 哈希和同伦类哈希

    用法:
        ritual = RatifyRitual()
        ritual.plan(spec)
        ritual.consult(reviewer_did, approved=True, ...)
        ritual.ratify(lambda data, did: sign(data, did))
    """

    def __init__(self):
        self.phase: RitualPhase = RitualPhase.PENDING
        self.spec: Optional[AgentSpec] = None
        self.spec_hash: str = ""
        self.opinions: list[ConsultOpinion] = []
        self.ratified: bool = False
        self.signature: str = ""  # 数字签名
        self.ratified_at: str = ""
        self._log: list[dict] = []

    # ---------- Step 1: Plan ----------

    def plan(self, spec: AgentSpec) -> bool:
        """Step 1 — Plan（集中）：定义代理 Spec。

        责任节点明确：
          - 资产池注资额
          - Agent 的用途
          - 权限边界

        返回 True 表示进入 PLAN 阶段。
        """
        if self.phase not in (RitualPhase.PENDING, RitualPhase.ABORTED):
            return False

        self.spec = spec
        self.spec_hash = hashlib.sha256(
            spec.to_text().encode()
        ).hexdigest()

        self.phase = RitualPhase.PLAN
        self._log.append({
            "phase": "plan",
            "spec_hash": self.spec_hash,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        })
        return True

    # ---------- Step 2: Consult ----------

    def consult(
        self,
        reviewer: str,
        approved: bool,
        comments: str = "",
        severity: str = "info",
        conditions: list = None,
    ) -> ConsultOpinion:
        """Step 2 — Consult（到群众中去）：法务/安全评审。

        评审 Spec 的合规性与风险，提出修改意见。

        返回评审意见。
        """
        if self.phase not in (RitualPhase.PLAN, RitualPhase.CONSULT):
            raise RitualError(
                f"Cannot consult in phase '{self.phase.value}'"
            )

        self.phase = RitualPhase.CONSULT
        opinion = ConsultOpinion(
            reviewer=reviewer,
            approved=approved,
            severity=severity,
            comments=comments,
            conditions=conditions or [],
        )
        self.opinions.append(opinion)
        self._log.append({
            "phase": "consult",
            "reviewer": reviewer,
            "approved": approved,
            "timestamp": opinion.timestamp,
        })
        return opinion

    def consult_auto(
        self,
        safety_check: Callable[[AgentSpec], ConsultOpinion],
    ) -> ConsultOpinion:
        """自动化评审 — 传入安全检查函数。"""
        if self.spec is None:
            raise RitualError("No spec to review")
        return self.consult(safety_check(self.spec).reviewer,
                            safety_check(self.spec).approved,
                            safety_check(self.spec).comments,
                            safety_check(self.spec).severity,
                            safety_check(self.spec).conditions)

    @property
    def all_approved(self) -> bool:
        """是否所有评审都通过。"""
        if not self.opinions:
            return False
        return all(o.approved for o in self.opinions)

    @property
    def critical_opinions(self) -> list:
        """返回 severity == 'critical' 的评审意见。"""
        return [o for o in self.opinions if o.severity == "critical"]

    # ---------- Step 3: Ratify ----------

    def ratify(
        self,
        signer: Callable[[str, str], str],
    ) -> bool:
        """Step 3 — Ratify（再集中）：数字签名封印。

        签名内容 = SHA-256(soul_md + spec_hash + all_opinions)

        参数:
            signer: 签名函数 (data_to_sign: str, owner_did: str) -> signature: str

        返回 True 表示封印完成。
        """
        if self.phase != RitualPhase.CONSULT:
            return False
        if not self.all_approved:
            return False
        if self.critical_opinions:
            return False

        if self.spec is None:
            return False

        soul_md = self.spec.to_soul_md()
        material = f"{soul_md}|{self.spec_hash}|{self._opinions_hash()}"
        data_to_sign = hashlib.sha256(material.encode()).hexdigest()

        try:
            self.signature = signer(data_to_sign, self.spec.owner_did)
        except Exception as e:
            self._log.append({
                "phase": "ratify",
                "status": "failed",
                "error": str(e),
            })
            return False

        self.ratified = True
        self.ratified_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        self.phase = RitualPhase.COMPLETE

        self._log.append({
            "phase": "ratify",
            "signature": self.signature[:16] + "...",
            "timestamp": self.ratified_at,
        })
        return True

    def ratify_simple(self) -> bool:
        """简化签名 — 使用 SHA-256 模拟数字签名。"""
        def _sha256_signer(data: str, did: str) -> str:
            return hashlib.sha256(f"{data}{did}".encode()).hexdigest()

        return self.ratify(_sha256_signer)

    # ---------- 查询 ----------

    def status(self) -> dict:
        return {
            "phase": self.phase.value,
            "spec_hash": self.spec_hash,
            "opinions_count": len(self.opinions),
            "all_approved": self.all_approved,
            "ratified": self.ratified,
            "signature": self.signature[:16] + "..." if self.signature else "",
        }

    def verify_ratification(self) -> bool:
        """验证确权是否完成。"""
        return self.phase == RitualPhase.COMPLETE and self.ratified

    def audit_trail(self) -> list:
        """返回完整审计链路。"""
        return self._log

    # ---------- 内部 ----------

    def _opinions_hash(self) -> str:
        opin_data = [
            {"r": o.reviewer, "a": o.approved, "s": o.severity}
            for o in self.opinions
        ]
        return hashlib.sha256(
            json.dumps(opin_data, sort_keys=True).encode()
        ).hexdigest()


class RitualError(Exception):
    """确权仪式异常。"""
    pass
