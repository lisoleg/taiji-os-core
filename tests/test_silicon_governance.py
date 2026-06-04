"""tests/test_silicon_governance.py — 硅基代理治理体系测试

覆盖模块:
  - core/aic.py          : AIC 凭证系统 (M175 + H_h + 归责真空)
  - core/ark_covenant.py : 约柜合约 (M175封印/M106验收/M178罚没)
  - core/gcd_engine.py   : GCD 归约算子 (Pre/Post校验)
  - core/tri_spin_governor.py : 三旋治理 (情治/理治/法治)
  - core/ratify_ritual.py: 确权仪式 (Plan→Consult→Ratify)
  - core/five_layer_architecture.py : 五层次穿透架构
  - syscalls/opc_registry.py : OPC 注册表

定理验证测试:
  - 归责真空定理
  - GCD 消除小龙虾死锁定理
  - 约柜不可篡改定理
  - 确权仪式降推诿定理
"""

import hashlib
import json
import time

import pytest


# ============================================================================
# AIC 凭证系统测试
# ============================================================================

class TestAIC:
    """AIC 凭证系统测试。"""

    def test_issue_aic(self):
        """M175: 签发 AIC 凭证。"""
        from core.aic import AICIssuer

        aic = AICIssuer.issue(
            agent_name="TestAgent",
            owner_did="did:taiji:rn:alice",
            capabilities=["text_generate", "file_read"],
            pi_spec="只读文本生成代理，不写入文件",
        )

        assert aic.credential_id.startswith("urn:uuid:aic-")
        assert aic.agent_name == "TestAgent"
        assert aic.owner_did == "did:taiji:rn:alice"
        assert len(aic.capabilities) == 2
        assert len(aic.homotopy_class_hash) == 64  # SHA-256
        assert len(aic.proof) == 64

    def test_homotopy_class_hash_same_identity(self):
        """H_h: 同一责任节点 + 相同能力 → 同伦类不变。"""
        from core.aic import AICIssuer, HomotopyClassHasher

        aic_v1 = AICIssuer.issue(
            "Agent-1", "did:rn:alice",
            ["read", "write"], "spec-v1",
        )
        aic_v2 = AICIssuer.issue(
            "Agent-1-v2", "did:rn:alice",
            ["read", "write"], "spec-v1",
        )

        # 不同凭证ID，但同伦类哈希相同 → 同一身份
        assert aic_v1.credential_id != aic_v2.credential_id
        assert aic_v1.homotopy_class_hash == aic_v2.homotopy_class_hash
        assert HomotopyClassHasher.same_identity(aic_v1, aic_v2)

    def test_homotopy_class_hash_different_identity(self):
        """不同责任节点 → 不同同伦类。"""
        from core.aic import AICIssuer, HomotopyClassHasher

        aic_alice = AICIssuer.issue(
            "Agent-A", "did:rn:alice", ["read"], "spec"
        )
        aic_bob = AICIssuer.issue(
            "Agent-B", "did:rn:bob", ["read"], "spec"
        )

        assert aic_alice.homotopy_class_hash != aic_bob.homotopy_class_hash
        assert not HomotopyClassHasher.same_identity(aic_alice, aic_bob)

    def test_verify_aic(self):
        """验证 M175 封印凭证的有效性。"""
        from core.aic import AICIssuer

        aic = AICIssuer.issue("Test", "did:rn:alice", ["read"], "spec")
        assert AICIssuer.verify_aic(aic)

    def test_aic_json_roundtrip(self):
        """AIC JSON 序列化/反序列化。"""
        from core.aic import AICIssuer, AgentIdentityCredential

        aic = AICIssuer.issue("Test", "did:rn:alice", ["read"], "spec")
        data = aic.to_dict()
        restored = AgentIdentityCredential.from_dict(data)

        assert restored.credential_id == aic.credential_id
        assert restored.homotopy_class_hash == aic.homotopy_class_hash
        assert restored.proof == aic.proof

    def test_vacuum_risk_zero(self):
        """归责真空定理: 完整凭证 → 风险 = 0。"""
        from core.aic import AICIssuer, AccountabilityVerifier

        aic = AICIssuer.issue("Test", "did:rn:alice", ["read"], "spec")
        assert AccountabilityVerifier.vacuum_risk(aic) == 0.0
        assert AccountabilityVerifier.assert_accountable(aic)

    def test_vacuum_risk_full(self):
        """归责真空定理: 无凭证 → 风险 = 1。"""
        from core.aic import AccountabilityVerifier

        assert AccountabilityVerifier.vacuum_risk(None) == 1.0


# ============================================================================
# GCD 归约算子测试
# ============================================================================

class TestGCD:
    """GCD 归约算子测试。"""

    def test_pre_url_valid(self):
        """Pre: 合法 URL 通过。"""
        from core.gcd_engine import GCDEngine

        engine = GCDEngine()
        result = engine.check("browser.navigate", "pre", url="https://example.com")
        assert result.passed

    def test_pre_url_empty_blocked(self):
        """Pre: 空 URL 阻断（GCD 消除小龙虾死锁）。"""
        from core.gcd_engine import GCDEngine

        engine = GCDEngine()
        result = engine.check("browser.navigate", "pre", url="")
        assert not result.passed
        assert result.blocked_count >= 1

    def test_pre_url_invalid_blocked(self):
        """Pre: 非法 URL 阻断。"""
        from core.gcd_engine import GCDEngine

        engine = GCDEngine()
        result = engine.check("browser.navigate", "pre", url="ftp://bad")
        assert not result.passed

    def test_pre_shell_dangerous_blocked(self):
        """Pre: 危险 shell 命令阻断。"""
        from core.gcd_engine import GCDEngine

        engine = GCDEngine()
        result = engine.check("shell.exec", "pre", command="rm -rf /")
        assert not result.passed
        assert "rm -rf" in str(result.violations)

    def test_pre_shell_safe_passes(self):
        """Pre: 安全 shell 命令通过。"""
        from core.gcd_engine import GCDEngine

        engine = GCDEngine()
        result = engine.check("shell.exec", "pre", command="ls -la")
        assert result.passed

    def test_pre_file_path_traversal_blocked(self):
        """Pre: 路径遍历 ../ 阻断。"""
        from core.gcd_engine import GCDEngine

        engine = GCDEngine()
        result = engine.check("file.read", "pre", path="../etc/shadow")
        assert not result.passed

    def test_pre_file_write_system_blocked(self):
        """Pre: 写入系统目录阻断。"""
        from core.gcd_engine import GCDEngine

        engine = GCDEngine()
        result = engine.check("file.write", "pre", path="/etc/crontab")
        assert not result.passed

    def test_gcd_stats_tracking(self):
        """GCD 统计追踪。"""
        from core.gcd_engine import GCDEngine

        engine = GCDEngine()
        engine.check("browser.navigate", "pre", url="https://ok.com")
        engine.check("browser.navigate", "pre", url="")

        assert engine.stats["pre_checks"] == 2
        assert engine.stats["blocks"] >= 1


# ============================================================================
# 约柜合约测试
# ============================================================================

class TestArkCovenant:
    """约柜合约测试。"""

    def test_deploy_and_seal(self):
        """M175: 部署 + 封印。"""
        from core.ark_covenant import ArkCovenant

        ark = ArkCovenant.deploy(
            "covenant-1", "spec-text", "did:rn:alice", "did:agent:1"
        )
        assert not ark.is_sealed

        ark = ArkCovenant.seal("covenant-1", "did:rn:alice")
        assert ark.is_sealed
        assert ark.sealed_block > 0

    def test_seal_twice_denied(self):
        """约柜不可篡改定理: 封印后不可再封印。"""
        from core.ark_covenant import ArkCovenant, ArkError

        ArkCovenant.deploy("c-2", "spec", "did:rn:alice", "did:agent:1")
        ArkCovenant.seal("c-2", "did:rn:alice")

        with pytest.raises(ArkError, match="already sealed"):
            ArkCovenant.seal("c-2", "did:rn:alice")

    def test_agent_cannot_seal(self):
        """约柜不可篡改定理: Agent 无权封印。"""
        from core.ark_covenant import ArkCovenant, ArkError

        ArkCovenant.deploy("c-3", "spec", "did:rn:alice", "did:agent:1")

        with pytest.raises(ArkError, match="not client"):
            ArkCovenant.seal("c-3", "did:agent:1")

    def test_complete_release_tokens(self):
        """M106: 验收通过 → 释放 Token。"""
        from core.ark_covenant import ArkCovenant

        ArkCovenant.deploy(
            "c-4", "spec", "did:rn:alice", "did:agent:1", initial_tokens=100
        )
        ArkCovenant.seal("c-4", "did:rn:alice")
        ark = ArkCovenant.complete("c-4", "did:rn:alice")

        assert ark.completed
        assert ark.escrow_tokens == 0  # Token 已释放

    def test_slash_forfeit_tokens(self):
        """M178: 罚没 → 没收 Token。"""
        from core.ark_covenant import ArkCovenant

        ArkCovenant.deploy(
            "c-5", "spec", "did:rn:alice", "did:agent:1", initial_tokens=50
        )
        ArkCovenant.seal("c-5", "did:rn:alice")
        ark = ArkCovenant.slash("c-5", "did:rn:alice", "违规操作")

        assert ark.slashed
        assert ark.escrow_tokens == 0

    def test_cannot_complete_after_slash(self):
        """罚没后不可验收。"""
        from core.ark_covenant import ArkCovenant, ArkError

        ArkCovenant.deploy("c-6", "spec", "did:rn:alice", "did:agent:1")
        ArkCovenant.seal("c-6", "did:rn:alice")
        ArkCovenant.slash("c-6", "did:rn:alice", "违约")

        with pytest.raises(ArkError, match="already slashed"):
            ArkCovenant.complete("c-6", "did:rn:alice")


# ============================================================================
# 三旋治理测试
# ============================================================================

class TestTriSpinGovernor:
    """三旋治理测试。"""

    @pytest.fixture
    def gov(self):
        from core.tri_spin_governor import TriSpinGovernor
        g = TriSpinGovernor()
        g.bootstrap(
            agent_name="Daemon-Test",
            owner_did="did:rn:alice",
            capabilities=["read", "write"],
            spec_text="安全沙箱代理，禁止系统修改",
            escrow_tokens=100,
        )
        return g

    def test_bootstrap_creates_aic_and_ark(self, gov):
        """引导: AIC + Ark 双创建。"""
        assert gov.aic is not None
        assert gov.ark is not None
        assert gov.aic.owner_did == "did:rn:alice"

    def test_consensus_acknowledge(self, gov):
        """情治: 主体认领。"""
        assert not gov.consensus_verify()
        ok = gov.consensus_acknowledge("did:rn:alice")
        assert ok
        assert gov.consensus_verify()

    def test_consensus_wrong_owner(self, gov):
        """情治: 非 RN 认领被拒绝。"""
        ok = gov.consensus_acknowledge("did:rn:bob")
        assert not ok

    def test_cryptography_seal(self, gov):
        """理治: M175 封印。"""
        ok = gov.cryptography_seal("did:rn:alice")
        assert ok
        assert gov.ark.is_sealed

    def test_statute_slash(self, gov):
        """法治: M178 罚没。"""
        gov.cryptography_seal("did:rn:alice")
        ok = gov.statute_slash("did:rn:alice", "违规")
        assert ok
        assert gov.ark.slashed

    def test_full_lifecycle(self, gov):
        """完整三旋治理生命周期。"""
        # 情治
        gov.consensus_acknowledge("did:rn:alice")
        # 理治
        gov.cryptography_seal("did:rn:alice")
        # 法治 - GCD
        result = gov.statute_check("browser.navigate", "pre", url="https://ok.com")
        assert result.passed
        # 法治 - 验收
        gov.statute_complete("did:rn:alice")
        assert gov.ark.completed

        report = gov.report()
        assert report.accountable


# ============================================================================
# 确权仪式测试
# ============================================================================

class TestRatifyRitual:
    """确权仪式测试。"""

    @pytest.fixture
    def spec(self):
        from core.ratify_ritual import AgentSpec
        return AgentSpec(
            agent_name="TestAgent",
            owner_did="did:rn:alice",
            purpose="自动化测试代理",
            capabilities=["read", "write", "shell"],
            boundaries=["no_system_modify"],
            constraints=["require_gcd"],
        )

    def test_plan_creates_spec_hash(self, spec):
        """Step 1: Plan → spec_hash 生成。"""
        from core.ratify_ritual import RatifyRitual
        ritual = RatifyRitual()
        ok = ritual.plan(spec)
        assert ok
        assert len(ritual.spec_hash) == 64

    def test_consult_and_approve(self, spec):
        """Step 2: Consult → 评审通过。"""
        from core.ratify_ritual import RatifyRitual
        ritual = RatifyRitual()
        ritual.plan(spec)
        opinion = ritual.consult("did:reviewer:1", True, "合规")
        assert opinion.approved
        assert ritual.all_approved

    def test_ratify_complete_cycle(self, spec):
        """确权仪式降推诿定理: 三步完整流程。"""
        from core.ratify_ritual import RatifyRitual
        ritual = RatifyRitual()
        ritual.plan(spec)
        ritual.consult("did:reviewer:1", True, "通过")
        ok = ritual.ratify_simple()
        assert ok
        assert ritual.verify_ratification()
        assert ritual.phase.value == "complete"

    def test_cannot_ratify_without_consult(self, spec):
        """跳过 Consult 无法 Ratify。"""
        from core.ratify_ritual import RatifyRitual
        ritual = RatifyRitual()
        ritual.plan(spec)
        assert not ritual.ratify_simple()  # 未进入 consult 阶段

    def test_cannot_ratify_with_critical(self, spec):
        """有 critical 评审意见无法封印。"""
        from core.ratify_ritual import RatifyRitual
        ritual = RatifyRitual()
        ritual.plan(spec)
        ritual.consult("did:reviewer:1", False, "风险过高", severity="critical")
        assert not ritual.ratify_simple()

    def test_soul_md_generation(self, spec):
        """Spec → SOUL.md 生成。"""
        md = spec.to_soul_md()
        assert "SOUL.md" in md
        assert "did:rn:alice" in md
        assert "NO_SYSTEM_MODIFY" in md.upper() or "no_system_modify" in md


# ============================================================================
# 五层次穿透架构测试
# ============================================================================

class TestFiveLayerArchitecture:
    """五层次穿透架构测试。"""

    @pytest.fixture
    def pipeline(self):
        from core.tri_spin_governor import TriSpinGovernor
        from core.five_layer_architecture import FiveLayerPipeline

        gov = TriSpinGovernor()
        gov.bootstrap(
            "FiveLayerTest", "did:rn:alice",
            ["read", "write"], "测试代理",
            escrow_tokens=50,
        )
        gov.consensus_acknowledge("did:rn:alice")
        gov.cryptography_seal("did:rn:alice")
        return FiveLayerPipeline(gov)

    def test_l1_l5_full_pipeline(self, pipeline):
        """L1 → L5 全链路贯穿。"""
        result = pipeline.execute(
            intent="分析数据并生成报告",
            tool_name="text.generate",
            tool_args={"prompt": "分析"},
        )
        assert len(result.layers) == 5
        assert result.layers[0].layer.value == 1  # L1
        assert result.layers[4].layer.value == 5  # L5

    def test_l3_gcd_block(self, pipeline):
        """L3 GCD 阻断: 非法 URL。"""
        result = pipeline.execute(
            intent="访问网站",
            tool_name="browser.navigate",
            tool_args={"url": ""},
        )
        l3 = result.layers[2]
        assert l3.status == "blocked"

    def test_pipeline_audit_trace(self, pipeline):
        """全链路审计追踪。"""
        result = pipeline.execute(
            intent="测试审计",
            tool_name="shell.exec",
            tool_args={"command": "ls"},
        )
        trace = result.trace
        assert len(trace) == 5
        assert trace[0]["name"] == "流贯（意识层）"
        assert trace[1]["name"] == "代数壳（锚定层/M175）"
        assert trace[2]["name"] == "拓扑流贯（执行层/GCD）"

    def test_pipeline_accountability_chain(self, pipeline):
        """归责真空定理: 完整链路 → 可归责。"""
        result = pipeline.execute(
            intent="安全任务",
            tool_name="file.read",
            tool_args={"path": "/tmp/data.txt"},
        )
        # 有完整 AIC + 完整链路 → 可归责
        assert result.aic_valid
        assert len(result.trace) == 5


# ============================================================================
# OPC 注册表测试
# ============================================================================

class TestOPCRegistry:
    """OPC 注册表测试。"""

    def test_register_node(self):
        """注册责任节点。"""
        from syscalls.opc_registry import OPCRegistry

        node = OPCRegistry.register_node("Alice", "natural", 1000)
        assert node.name == "Alice"
        assert node.asset_pool == 1000
        assert node.did.startswith("did:taiji:rn:")

    def test_issue_aic_from_rn(self):
        """从 RN 签发 AIC。"""
        from syscalls.opc_registry import OPCRegistry

        node = OPCRegistry.register_node("Bob", "natural", 500)
        aic = OPCRegistry.issue_aic(
            node.did, "BobAgent", ["read"], "安全代理"
        )
        assert aic is not None
        assert aic.owner_did == node.did
        assert node.aic_issued == 1

    def test_grant_personality(self):
        """授予临时法人人格。"""
        from syscalls.opc_registry import OPCRegistry

        node = OPCRegistry.register_node("Carol", "natural", 2000)
        p = OPCRegistry.grant_personality(node.did, 500)
        assert p is not None
        assert p.asset_pool_cap == 500

    def test_personality_cap_exceeds_pool(self):
        """法人人格上限 > 资产池 → 拒绝。"""
        from syscalls.opc_registry import OPCRegistry

        node = OPCRegistry.register_node("Dave", "natural", 100)
        p = OPCRegistry.grant_personality(node.did, 500)
        assert p is None

    def test_revoke_aic(self):
        """吊销 AIC。"""
        from syscalls.opc_registry import OPCRegistry

        node = OPCRegistry.register_node("Eve", "natural")
        aic = OPCRegistry.issue_aic(node.did, "EveAgent", ["read"], "spec")
        assert OPCRegistry.get_aic(aic.credential_id) is not None

        ok = OPCRegistry.revoke_aic(aic.credential_id, node.did)
        assert ok
        assert OPCRegistry.get_aic(aic.credential_id) is None

    def test_deposit_withdraw(self):
        """资产池充提。"""
        from syscalls.opc_registry import OPCRegistry

        node = OPCRegistry.register_node("Frank", "natural", 100)
        OPCRegistry.deposit(node.did, 200)
        assert OPCRegistry.get_node(node.did).asset_pool == 300

        OPCRegistry.withdraw(node.did, 50)
        assert OPCRegistry.get_node(node.did).asset_pool == 250

        assert not OPCRegistry.withdraw(node.did, 9999)  # 超提


# ============================================================================
# Session 治理集成测试
# ============================================================================

class TestSessionGovernance:
    """Session v4 三旋治理集成。"""

    def test_session_governance_bootstrap(self):
        """Session 在 tri-spin 模式下自动初始化治理。"""
        from core.session import TaijiSession

        class FakeLLM:
            pass

        session = TaijiSession(
            sid="gov-test-1",
            llm_router=FakeLLM(),
            governance="tri-spin",
            agent_spec="测试代理，只读操作",
            owner_did="did:rn:test",
        )

        assert session.tri_spin is not None
        assert session.pipeline is not None
        assert session.ritual is not None
        assert session.ritual.spec is not None

        status = session.status()
        assert "governance_report" in status

    def test_session_basic_no_governance(self):
        """basic 模式不启动治理。"""
        from core.session import TaijiSession

        class FakeLLM:
            pass

        session = TaijiSession(
            sid="basic-test-1",
            llm_router=FakeLLM(),
            governance="basic",
        )

        assert session.tri_spin is None
        assert session.pipeline is None
