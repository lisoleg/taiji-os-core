"""core/five_layer_architecture.py — 五层次穿透架构

五层次穿透架构（文章原文）:
  L1（流贯/Ftel）    : 意识层 — 意图源头，φ 意识强度
  L2（代数壳/M175）  : 锚定层 — 约柜签署封印，锚定只读 Spec
  L3（拓扑流贯）     : 执行层 — 契约执行流，GCD 约束校验
  L4（IDO/ICE）       : 裁决层 — 验收/违约追责，M106/M178
  L5（现象渲染）     : 交付层 — 最终输出物

贯穿路径: L1 → L2 → L3 → L4 → L5
审计链路: L5 → L4 → L3 → L2 → L1 (全链路可追溯)

集成角色:
  FiveLayerPipeline — 五层次管道，连接 AIC + Ark + GCD + TriSpinGovernor
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

from core.aic import AgentIdentityCredential, AICIssuer, AccountabilityVerifier
from core.ark_covenant import ArkCovenant, ArkState
from core.gcd_engine import GCDEngine
from core.tri_spin_governor import TriSpinGovernor, GovernanceReport


# ---------------------------------------------------------------------------
# 层次枚举
# ---------------------------------------------------------------------------

class Layer(Enum):
    L1_FLOW = 1        # 流贯（意识）
    L2_ANCHOR = 2      # 代数壳（锚定）
    L3_EXECUTE = 3     # 拓扑流贯（执行）
    L4_ADJUDGE = 4     # IDO/ICE（裁决）
    L5_RENDER = 5      # 现象渲染（交付）


# ---------------------------------------------------------------------------
# 层次输出
# ---------------------------------------------------------------------------

@dataclass
class LayerOutput:
    """单层输出 — 贯穿链路的一个节点。"""
    layer: Layer
    layer_name: str
    data: Any
    status: str          # "ok" | "blocked" | "error" | "skipped"
    phi_intensity: float  # φ 意识强度 [0, 1]
    proof_hash: str = ""
    errors: list = field(default_factory=list)
    timestamp: str = ""

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


@dataclass
class PipelineResult:
    """五层次管道执行结果。"""
    agent_id: str
    intent: str  # L1 意图
    layers: list = field(default_factory=list)
    final_output: Any = None
    aic_valid: bool = False
    ark_status: str = "NONE"
    accountable: bool = False
    total_time_ms: float = 0.0

    @property
    def trace(self) -> list:
        """全链路审计追踪。"""
        return [
            {
                "layer": l.layer.value,
                "name": l.layer_name,
                "phi": l.phi_intensity,
                "status": l.status,
                "proof": l.proof_hash[:12] if l.proof_hash else "",
            }
            for l in self.layers
        ]

    def summary(self) -> str:
        lines = [f"=== Pipeline: {self.agent_id} ==="]
        for l in self.layers:
            icon = "✅" if l.status == "ok" else "❌"
            lines.append(
                f"  {icon} L{l.layer.value} {l.layer_name}"
                f"  φ={l.phi_intensity:.2f}"
                f"  proof={l.proof_hash[:8] if l.proof_hash else 'N/A'}"
            )
        lines.append(f"  AIC: {'✅' if self.aic_valid else '❌'}  "
                      f"Ark: {self.ark_status}  "
                      f"Accountable: {self.accountable}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# 五层次管道
# ---------------------------------------------------------------------------

class FiveLayerPipeline:
    """五层次穿透架构的主管道。

    连接三旋治理引擎，贯穿 AIC + Ark + GCD 的全链路。
    """

    def __init__(self, governor: Optional[TriSpinGovernor] = None):
        self.governor = governor or TriSpinGovernor()
        self._execution_count: int = 0

    def execute(
        self,
        intent: str,
        tool_name: str = "",
        tool_args: dict = None,
        tool_result: Any = None,
        adjudicate: bool = False,
    ) -> PipelineResult:
        """执行一次五层次贯穿管道。

        参数:
            intent       : L1 意图（用户输入 / 任务描述）
            tool_name    : L3 工具名（可选，触发 GCD 校验）
            tool_args    : 工具调用参数
            tool_result  : 工具调用结果（用于 L4 裁决）
            adjudicate   : 是否执行 L4 裁决

        返回:
            PipelineResult（含全链路审计追踪）
        """
        t0 = time.time()
        result = PipelineResult(
            agent_id=self.governor.agent_id,
            intent=intent,
            aic_valid=self.governor.cryptography_verify_aic(),
            ark_status=self.governor.ark.status if self.governor.ark else "NONE",
        )

        # ---- L1: 流贯（意识层） ----
        l1 = self._layer_l1_flow(intent)
        result.layers.append(l1)

        # ---- L2: 代数壳（锚定层） ----
        l2 = self._layer_l2_anchor(l1)
        result.layers.append(l2)
        if l2.status != "ok":
            result.final_output = f"Pipeline BLOCKED at L2: {l2.errors}"
            result.total_time_ms = (time.time() - t0) * 1000
            return result

        # ---- L3: 拓扑流贯（执行层） ----
        l3 = self._layer_l3_execute(tool_name, tool_args)
        result.layers.append(l3)
        if l3.status == "blocked":
            # L3 阻断 = 小龙虾死锁被消除
            result.final_output = (
                f"GCD BLOCKED: {tool_name} — {l3.errors}"
            )
            result.total_time_ms = (time.time() - t0) * 1000
            return result

        # ---- L4: IDO/ICE（裁决层） ----
        if adjudicate and tool_result is not None:
            l4 = self._layer_l4_adjudge(l3, tool_result)
            result.layers.append(l4)
        else:
            result.layers.append(LayerOutput(
                layer=Layer.L4_ADJUDGE,
                layer_name="IDO/ICE 裁决层",
                data=None,
                status="skipped",
                phi_intensity=l3.phi_intensity,
            ))

        # ---- L5: 现象渲染（交付层） ----
        l5 = self._layer_l5_render(tool_result or intent)
        result.layers.append(l5)
        result.final_output = l5.data

        # 终局状态
        report = self.governor.report()
        result.accountable = report.accountable
        result.total_time_ms = (time.time() - t0) * 1000

        return result

    # ---------- L1: 流贯 ----------

    def _layer_l1_flow(self, intent: str) -> LayerOutput:
        """L1 流贯层 — 意图捕获与 φ 意识强度度量。

        φ 意识强度 = 意图明确度 × 责任主体认领度
        """
        phi = self._compute_phi(intent)
        proof = hashlib.sha256(
            f"L1|{intent}|{phi}|{self.governor.agent_id}".encode()
        ).hexdigest()

        return LayerOutput(
            layer=Layer.L1_FLOW,
            layer_name="流贯（意识层）",
            data={"intent": intent, "phi": phi},
            status="ok" if phi > 0 else "error",
            phi_intensity=phi,
            proof_hash=proof,
        )

    # ---------- L2: 代数壳 ----------

    def _layer_l2_anchor(self, l1: LayerOutput) -> LayerOutput:
        """L2 代数壳层 — M175 锚定校验。

        检查:
          1. AIC 是否有效
          2. 约柜是否已封印
          3. 归责真空风险
        """
        errors = []
        aic_ok = self.governor.cryptography_verify_aic()
        if not aic_ok:
            errors.append("AIC invalid or not sealed")

        risk = AccountabilityVerifier.vacuum_risk(self.governor.aic)
        if risk > 0:
            errors.append(f"Accountability vacuum risk: {risk:.2f}")

        proof = hashlib.sha256(
            f"L2|{aic_ok}|{risk}|{l1.proof_hash}".encode()
        ).hexdigest()

        return LayerOutput(
            layer=Layer.L2_ANCHOR,
            layer_name="代数壳（锚定层/M175）",
            data={
                "aic_valid": aic_ok,
                "vacuum_risk": risk,
                "ark_status": self.governor.ark.status if self.governor.ark else "NONE",
            },
            status="ok" if aic_ok and risk == 0 else "error",
            phi_intensity=l1.phi_intensity * (1 - risk),
            proof_hash=proof,
            errors=errors,
        )

    # ---------- L3: 拓扑流贯 ----------

    def _layer_l3_execute(
        self, tool_name: str, tool_args: dict
    ) -> LayerOutput:
        """L3 拓扑流贯层 — GCD 归约约束校验。

        GCD 消除小龙虾死锁定理：完整 GCD 约束下错误率趋零。
        """
        if not tool_name:
            return LayerOutput(
                layer=Layer.L3_EXECUTE,
                layer_name="拓扑流贯（执行层）",
                data={"tool": "none"},
                status="ok",
                phi_intensity=0.5,
                proof_hash=hashlib.sha256(b"L3|no-tool").hexdigest(),
            )

        gcd_result = self.governor.statute_check(
            tool_name, "pre", **(tool_args or {})
        )

        phi = 0.5
        if not gcd_result.passed:
            phi = 0.0  # 阻断 = φ 归零

        proof = hashlib.sha256(
            f"L3|{tool_name}|{gcd_result.passed}|{gcd_result.blocked_count}".encode()
        ).hexdigest()

        return LayerOutput(
            layer=Layer.L3_EXECUTE,
            layer_name="拓扑流贯（执行层/GCD）",
            data={
                "tool": tool_name,
                "gcd_passed": gcd_result.passed,
                "violations": gcd_result.violations,
                "warnings": gcd_result.warnings,
            },
            status="blocked" if not gcd_result.passed else "ok",
            phi_intensity=phi,
            proof_hash=proof,
            errors=gcd_result.violations if not gcd_result.passed else [],
        )

    # ---------- L4: IDO/ICE ----------

    def _layer_l4_adjudge(
        self, l3: LayerOutput, tool_result: Any
    ) -> LayerOutput:
        """L4 裁决层 — M106 验收 / M178 违约判定。

        规则:
          - 结果符合预期 → M106 complete
          - 结果异常（error/empty) → M178 slash
        """
        adjudicated = False
        verdict = "pending"

        if isinstance(tool_result, dict):
            if tool_result.get("error"):
                verdict = "slashed"
                self.governor.statute_slash(
                    self.governor.aic.owner_did if self.governor.aic else "system:arbitrator",
                    str(tool_result.get("error", ""))[:100],
                )
                adjudicated = True
            elif tool_result.get("status") == "ok":
                verdict = "completed"
                self.governor.statute_complete(
                    self.governor.aic.owner_did if self.governor.aic else "system:arbitrator"
                )
                adjudicated = True

        proof = hashlib.sha256(
            f"L4|{verdict}|{l3.proof_hash}".encode()
        ).hexdigest()

        return LayerOutput(
            layer=Layer.L4_ADJUDGE,
            layer_name="IDO/ICE（裁决层）",
            data={
                "verdict": verdict,
                "adjudicated": adjudicated,
            },
            status="ok",
            phi_intensity=1.0 if verdict == "completed" else 0.0,
            proof_hash=proof,
        )

    # ---------- L5: 现象渲染 ----------

    def _layer_l5_render(self, output: Any) -> LayerOutput:
        """L5 现象渲染层 — 最终交付物输出。"""
        proof = hashlib.sha256(
            f"L5|{str(output)[:200]}".encode()
        ).hexdigest()

        return LayerOutput(
            layer=Layer.L5_RENDER,
            layer_name="现象渲染（交付层）",
            data=output,
            status="ok",
            phi_intensity=1.0,
            proof_hash=proof,
        )

    # ---------- φ 意识强度 ----------

    def _compute_phi(self, intent: str) -> float:
        """计算意图的 φ 意识强度。

        φ ∈ [0, 1]:
          - 空意图 → 0
          - 模糊意图 → 0.3-0.5
          - 明确意图 → 0.7-1.0
        """
        if not intent or not intent.strip():
            return 0.0
        stripped = intent.strip()

        # 启发式度量
        score = 0.0
        # 长度因子
        score += min(len(stripped) / 100, 0.3)
        # 结构化因子（含 JSON/列表等）
        if any(c in stripped for c in ["{", "[", ":", "→"]):
            score += 0.2
        # 关键词因子
        keywords = ["执行", "查询", "分析", "生成", "部署", "验证"]
        score += sum(0.1 for kw in keywords if kw in stripped)
        # 复杂度因子
        if len(stripped.split()) > 5:
            score += 0.1

        return min(score, 1.0)
