"""core/tri_spin_governor.py — 三旋治理 (Tri-spin Governance)

三旋治理架构（文章原文）:
  - 情治 (Consensus)   : 激活主体责任意识，通过认识递归迭代明确认领
  - 理治 (Cryptography) : 密码学锚定身份契约，M175 双签封印
  - 法治 (Statute)      : 确立行为归责底线，M106 验收 / M178 罚没

TriSpinGovernor 是硅基代理的治理核心框架，贯穿代理的完整生命周期：
  上线前 → 确权仪式 (情治+理治)
  执行中 → GCD 归约约束 (法治)
  结算时 → M106 验收 / M178 罚没 (法治)
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field
from typing import Optional

from core.aic import AgentIdentityCredential, AICIssuer, AccountabilityVerifier
from core.ark_covenant import ArkCovenant, ArkState
from core.gcd_engine import GCDEngine, GCDRegistry


# ---------------------------------------------------------------------------
# 治理记录结构
# ---------------------------------------------------------------------------

@dataclass
class SpinRecord:
    """三旋治理单次操作记录。"""
    spin: str          # "consensus" | "cryptography" | "statute"
    action: str        # 操作名称
    actor: str         # 执行者 DID
    detail: str        # 详细信息
    timestamp: str = ""
    block_height: int = 0

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


@dataclass
class GovernanceReport:
    """治理审计报告。"""
    agent_id: str
    total_actions: int
    consensus_ops: list = field(default_factory=list)     # 情治
    cryptography_ops: list = field(default_factory=list)   # 理治
    statute_ops: list = field(default_factory=list)        # 法治
    vacuum_risk: float = 1.0
    aic_valid: bool = False
    ark_status: str = "NONE"
    gcd_stats: dict = field(default_factory=dict)

    @property
    def accountable(self) -> bool:
        """是否可归责（归责真空定理）。"""
        return self.vacuum_risk == 0.0 and self.aic_valid

    def summary(self) -> str:
        lines = [
            f"=== Governance Report: {self.agent_id} ===",
            f"  可归责: {self.accountable}",
            f"  归责真空风险: {self.vacuum_risk:.2f}",
            f"  AIC 有效: {self.aic_valid}",
            f"  约柜状态: {self.ark_status}",
            f"  情治操作: {len(self.consensus_ops)}",
            f"  理治操作: {len(self.cryptography_ops)}",
            f"  法治操作: {len(self.statute_ops)}",
            f"  GCD 阻断: {self.gcd_stats.get('blocks', 0)}",
            f"  GCD 警告: {self.gcd_stats.get('warns', 0)}",
        ]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# 三旋治理引擎
# ---------------------------------------------------------------------------

class TriSpinGovernor:
    """三旋治理引擎。

    用法:
        gov = TriSpinGovernor()
        gov.bootstrap(agent_name="Daemon-1", owner_did="did:rn:alice", ...)

        # 执行中 — 情治: 确认主体认领
        gov.consensus_acknowledge("did:rn:alice")

        # 执行中 — 法治: GCD 约束
        result = gov.statute_check("browser.navigate", "pre", url="https://...")

        # 结算 — 法治: M106 验收
        gov.statute_complete("did:rn:alice")
    """

    def __init__(self, gcd_registry: Optional[GCDRegistry] = None):
        self.agent_id: str = ""
        self.aic: Optional[AgentIdentityCredential] = None
        self.ark: Optional[ArkState] = None
        self._ark_id: str = ""
        self._records: list[SpinRecord] = []
        self._consensus_confirmed: bool = False  # 情治: 主体已认领
        self._spec_text: str = ""

        # GCD 引擎
        self.gcd = GCDEngine(registry=gcd_registry)

    # ---------- 引导（Bootstrap） ----------

    def bootstrap(
        self,
        agent_name: str,
        owner_did: str,
        capabilities: list,
        spec_text: str,
        escrow_tokens: float = 0.0,
        seed: str = "",
    ) -> GovernanceReport:
        """引导三旋治理：签发 AIC + 部署约柜。

        流程:
          1. 理治: AICIssuer.issue() → M175 锚定
          2. 理治: ArkCovenant.deploy() → 约柜部署
          3. 情治: 初始化认领状态（待确权）
        """
        self.agent_id = f"taiji:agent:{agent_name}"
        self._spec_text = spec_text

        # 理治: AIC 签发
        self.aic = AICIssuer.issue(
            agent_name=agent_name,
            owner_did=owner_did,
            capabilities=capabilities,
            pi_spec=spec_text,
            seed=seed,
        )
        self._record("cryptography", "aic_issue", owner_did,
                      f"AIC issued: {self.aic.credential_id}")

        # 理治: 约柜部署
        self._ark_id = f"ark:{agent_name}"
        self.ark = ArkCovenant.deploy(
            covenant_id=self._ark_id,
            spec=spec_text,
            client_did=owner_did,
            agent_did=self.agent_id,
            initial_tokens=escrow_tokens,
        )
        self._record("cryptography", "ark_deploy", owner_did,
                      f"Ark deployed: {self._ark_id}")

        return self.report()

    # ---------- 情治 (Consensus) ----------

    def consensus_acknowledge(self, owner_did: str) -> bool:
        """情治: 责任主体明确认领代理。

        认识递归迭代的核心步骤 — 消除推诿心理。
        返回 True 表示认领成功。
        """
        if self.aic is None:
            self._record("consensus", "acknowledge_failed", owner_did,
                          "AIC not yet issued")
            return False

        if owner_did != self.aic.owner_did:
            self._record("consensus", "acknowledge_failed", owner_did,
                          f"DID mismatch: {owner_did} != {self.aic.owner_did}")
            return False

        self._consensus_confirmed = True
        self._record("consensus", "acknowledge", owner_did,
                      f"Agent {self.agent_id} acknowledged by {owner_did}")
        return True

    def consensus_verify(self) -> bool:
        """情治: 验证主体认领状态。"""
        return self._consensus_confirmed

    # ---------- 理治 (Cryptography) ----------

    def cryptography_seal(self, rn_did: str) -> bool:
        """理治: M175 封印 — 双签后 specHash 永不可改。

        调用约柜的 seal() 操作。
        """
        if self.ark is None:
            self._record("cryptography", "seal_failed", rn_did, "Ark not deployed")
            return False
        try:
            self.ark = ArkCovenant.seal(self._ark_id, rn_did)
            self._record("cryptography", "seal", rn_did,
                          f"Sealed at block {self.ark.sealed_block}")
            return True
        except Exception as e:
            self._record("cryptography", "seal_failed", rn_did, str(e))
            return False

    def cryptography_verify_aic(self) -> bool:
        """理治: 验证 AIC 有效性。"""
        if self.aic is None:
            return False
        return AICIssuer.verify_aic(self.aic)

    # ---------- 法治 (Statute) ----------

    def statute_check(self, tool_name: str, phase: str, *args, **kwargs):
        """法治: GCD 归约校验 — 实时约束工具调用。

        委托给 GCDEngine.check()。
        """
        result = self.gcd.check(tool_name, phase, *args, **kwargs)
        if result.blocked_count > 0:
            self._record("statute", "gcd_block", "system",
                          f"{tool_name}.{phase}: {result.violations}")
        elif result.warnings:
            self._record("statute", "gcd_warn", "system",
                          f"{tool_name}.{phase}: {result.warnings}")
        return result

    def statute_complete(self, rn_did: str) -> bool:
        """法治: M106 验收 — 通过后释放 Token。"""
        if self.ark is None:
            return False
        try:
            self.ark = ArkCovenant.complete(self._ark_id, rn_did)
            self._record("statute", "complete", rn_did,
                          f"M106 complete: tokens released")
            return True
        except Exception as e:
            self._record("statute", "complete_failed", rn_did, str(e))
            return False

    def statute_slash(self, rn_did: str, reason: str = "") -> bool:
        """法治: M178 罚没 — 违约自动罚没 Token。"""
        if self.ark is None:
            return False
        try:
            self.ark = ArkCovenant.slash(self._ark_id, rn_did, reason)
            self._record("statute", "slash", rn_did,
                          f"M178 slash: {reason}")
            return True
        except Exception as e:
            self._record("statute", "slash_failed", rn_did, str(e))
            return False

    # ---------- 报告 ----------

    def report(self) -> GovernanceReport:
        return GovernanceReport(
            agent_id=self.agent_id,
            total_actions=len(self._records),
            consensus_ops=[r for r in self._records if r.spin == "consensus"],
            cryptography_ops=[r for r in self._records if r.spin == "cryptography"],
            statute_ops=[r for r in self._records if r.spin == "statute"],
            vacuum_risk=AccountabilityVerifier.vacuum_risk(self.aic),
            aic_valid=self.cryptography_verify_aic(),
            ark_status=self.ark.status if self.ark else "NONE",
            gcd_stats=self.gcd.stats.copy(),
        )

    # ---------- 内部 ----------

    def _record(self, spin: str, action: str, actor: str, detail: str):
        self._records.append(SpinRecord(
            spin=spin, action=action, actor=actor, detail=detail,
        ))
