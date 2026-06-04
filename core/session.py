"""core/session.py — TaijiSession v4: AGI 进程主控 + Walrus Memory + 硅基代理治理

三种工作模式：
  - "text"   : 原始文本推演（默认）
  - "web"    : 浏览器云脑（WebWorldModel + WebPlanner + PlaywrightExecutor）

Walrus Memory 集成：
  - memory_hub       : MemoryHub 实例（可选），跨会话共享记忆
  - _save_continuation() : 保存 Continuation 同时同步到 MemoryHub
  - search_memory()  : 代理到 MemoryHub.search()
  - verify_integrity(): 代理到 MemoryHub.verify_all()

硅基代理治理集成（v4 新增）:
  - tri_spin         : TriSpinGovernor 三旋治理引擎
  - pipeline         : FiveLayerPipeline 五层次穿透架构
  - ritual           : RatifyRitual 确权仪式
  - governance_mode  : "tri-spin" | "basic" | "none"
"""

from __future__ import annotations

from typing import Optional

from core.world_model import WorldModel
from core.carbon_silicon_gan import CarbonSiliconGAN
from core.continuation import Continuation
from core.closure_env import ClosureEnv
from core.self_model import SelfModel


class TaijiSession:
    """
    TaijiSession v4 (AGI Process):
    封装世界模型、自我模型、GAN推演、Continuation机制、MemoryHub 与
    三旋治理（五层次穿透架构）。

    参数:
        sid          : 会话 ID
        llm_router   : HAL LLMRouter 实例
        snapshot_dir : Continuation 快照目录
        mode         : "text"（纯文本）或 "web"（浏览器云脑）
        headless     : Web 模式下是否无头启动浏览器
        memory_hub   : MemoryHub 实例（可选，启用 Walrus Memory 集成）
        governance   : 治理模式 "tri-spin"（默认）| "basic" | "none"
        agent_spec   : (tri-spin 时必需) 代理行为规范文本
        owner_did    : (tri-spin 时必需) 责任节点 DID
        escrow_tokens: (tri-spin 时可选) 初始托管 Token 数
    """

    def __init__(
        self,
        sid: str,
        llm_router,
        snapshot_dir: str = "snapshots",
        mode: str = "text",
        headless: bool = True,
        memory_hub=None,
        governance: str = "tri-spin",
        agent_spec: str = "",
        owner_did: str = "",
        escrow_tokens: float = 0.0,
    ):
        self.sid = sid
        self.snapshot_dir = snapshot_dir
        self.mode = mode
        self.memory_hub = memory_hub
        self.governance_mode = governance
        self._last_kid: Optional[str] = None  # 用于记忆链

        # 三旋治理 初始化
        self.tri_spin = None
        self.pipeline = None
        self.ritual = None
        if governance == "tri-spin" and owner_did and agent_spec:
            from core.tri_spin_governor import TriSpinGovernor
            from core.five_layer_architecture import FiveLayerPipeline
            from core.ratify_ritual import RatifyRitual, AgentSpec

            self.tri_spin = TriSpinGovernor()
            self.tri_spin.bootstrap(
                agent_name=sid,
                owner_did=owner_did,
                capabilities=["text_generate", "web_browse", "file_ops", "shell_exec"],
                spec_text=agent_spec,
                escrow_tokens=escrow_tokens,
            )
            self.pipeline = FiveLayerPipeline(self.tri_spin)

            # 确权仪式
            spec = AgentSpec(
                agent_name=sid,
                owner_did=owner_did,
                purpose=f"Taiji Session: {sid}",
                capabilities=["text_generate", "web_browse", "file_ops", "shell_exec"],
                boundaries=["no_system_modify", "no_sensitive_read"],
                constraints=["obey_gcd", "respect_ark"],
            )
            self.ritual = RatifyRitual()
            self.ritual.plan(spec)

        # 根据模式选择 WorldModel 和执行器
        if mode == "web":
            from core.web_world_model import WebWorldModel
            from syscalls.browser_executor import PlaywrightExecutor
            from syscalls.web_planner import WebPlanner

            self.w = WebWorldModel()
            self._executor = PlaywrightExecutor(
                headless=headless, web_world_model=self.w
            )
            self._planner = WebPlanner()
        else:
            from syscalls.executor import Executor
            from syscalls.planner import Planner

            self.w = WorldModel()
            self._executor = Executor()
            self._planner = Planner()

        self.llm_router = llm_router
        self.self_model = SelfModel(sid)
        self.gan = CarbonSiliconGAN(llm_router, self.w)
        self.env = ClosureEnv(intent="idle")

        # 自动注册到 MemoryHub
        if self.memory_hub is not None:
            self.memory_hub.register(sid)

    # ------------------------------------------------------------------
    # 主执行入口
    # ------------------------------------------------------------------

    def run(self, user_input: str) -> str:
        """执行一轮推演，返回输出或 Continuation ID。

        tri-spin 模式下自动穿越五层次贯穿架构：
          L1 流贯 → L2 代数壳 → L3 拓扑流贯(GCD) → L4 裁决 → L5 交付
        """
        self.env.push("user", user_input)

        # 三旋治理: 情治校验
        if self.tri_spin and not self.tri_spin.consensus_verify():
            if self.ritual and not self.ritual.verify_ratification():
                return (
                    "Governance BLOCKED: 确权仪式未完成\n"
                    f"  状态: {self.ritual.status()['phase']}\n"
                    "  请先完成 RatifyRitual (Plan → Consult → Ratify)"
                )

        # 五层次管道 (tri-spin 模式)
        if self.pipeline:
            return self._run_governed(user_input)

        if self.mode == "web":
            return self._run_web(user_input)
        return self._run_text(user_input)

    def _run_governed(self, user_input: str) -> str:
        """三旋治理模式 — 五层次贯穿推演。"""
        from core.five_layer_architecture import PipelineResult

        # L1-L5 贯穿
        result: PipelineResult = self.pipeline.execute(
            intent=user_input,
            tool_name="text.generate",
            tool_args={"prompt": user_input},
            adjudicate=False,
        )

        # 如果 L3 GCD 阻断，返回阻断信息
        if result.layers and result.layers[-1].status == "blocked":
            return f"GCD BLOCKED: {result.final_output}"

        # GAN 推演（L3 通过后进行）
        gan_result, reason = self.gan.step(self.env.to_dict(), user_input)
        if gan_result:
            # L4 裁决（模拟验收）
            if self.tri_spin:
                self.tri_spin.statute_complete(
                    self.tri_spin.aic.owner_did if self.tri_spin.aic else ""
                )
            self.env.push("assistant", gan_result)
            return f"{gan_result}\n\n[Governance]\n{result.summary()}"
        else:
            return self._save_continuation(reason)

    def _run_text(self, user_input: str) -> str:
        """原始文本模式推演。"""
        result, reason = self.gan.step(self.env.to_dict(), user_input)
        if result:
            self.env.push("assistant", result)
            return result
        return self._save_continuation(reason)

    def _run_web(self, user_input: str) -> str:
        """浏览器云脑模式。"""
        steps = self._planner.plan(
            user_input,
            context=self.env.to_dict(),
            llm_router=self.llm_router,
        )
        exec_results = self._executor.execute(steps, llm_router=self.llm_router)

        respond_outputs = [
            r["output"]
            for r in exec_results
            if r["action"] == "respond" and r["status"] == "ok"
        ]
        candidate_text = respond_outputs[-1] if respond_outputs else str(exec_results)

        result, reason = self.gan.step(self.env.to_dict(), candidate_text)

        if result:
            self.env.push("assistant", result)
            summary = self._exec_summary(exec_results)
            return f"{result}\n\n[执行摘要]\n{summary}"
        else:
            env_dict = self.env.to_dict()
            if hasattr(self.w, "page_snapshot"):
                env_dict["__page_snapshot__"] = self.w.page_snapshot()
            return self._save_continuation(reason, extra_env=env_dict)

    # ------------------------------------------------------------------
    # Continuation 保存（v2 + MemoryHub）
    # ------------------------------------------------------------------

    def _save_continuation(self, reason: str, extra_env: dict = None) -> str:
        """
        保存 Continuation v2，并同步到 MemoryHub。

        返回格式: "Continuation Saved: <kid> | reason: <reason> [proof: <hash>]"
        """
        env_dict = extra_env if extra_env is not None else self.env.to_dict()

        k = Continuation(
            self.sid,
            self.w.psi.copy(),
            env_dict,
            reason,
            snapshot_dir=self.snapshot_dir,
            parent_kid=self._last_kid,
        )

        self._last_kid = k.kid

        # 同步到 MemoryHub
        if self.memory_hub is not None:
            self.memory_hub.store(self.sid, {
                "kid": k.kid,
                "reason": reason,
                "proof": k.proof,
                "parent_kid": k.parent_kid,
                "env_summary": str(env_dict)[:200],
            })

        return (
            f"Continuation Saved: {k.kid}"
            f" | reason: {reason} [proof: {k.proof[:12]}...]"
        )

    @staticmethod
    def _exec_summary(results: list) -> str:
        lines = []
        for r in results:
            status_icon = "✅" if r["status"] == "ok" else "❌"
            lines.append(
                f"{status_icon} {r['action']} ({r['elapsed_ms']}ms): "
                f"{str(r['output'])[:80]}"
            )
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Continuation 恢复
    # ------------------------------------------------------------------

    def resume(self, kid: str) -> "TaijiSession":
        """从 Continuation 快照恢复会话状态。"""
        k = Continuation.load(kid, snapshot_dir=self.snapshot_dir)
        self.w.psi = k.psi
        env_dict = k.env
        if self.mode == "web" and "__page_snapshot__" in env_dict:
            self.w.restore_snapshot(env_dict.pop("__page_snapshot__"))
        self.env = ClosureEnv.from_dict(env_dict)
        self._last_kid = kid
        return self

    # ------------------------------------------------------------------
    # Walrus Memory 集成
    # ------------------------------------------------------------------

    def search_memory(self, query: str) -> list:
        """
        在共享记忆中搜索（代理到 MemoryHub）。

        参数:
            query: 搜索关键词（空格分隔多关键词，AND 逻辑）

        返回:
            匹配的记忆记录列表
        """
        if self.memory_hub is None:
            return []
        return self.memory_hub.search(query)

    def verify_integrity(self) -> dict:
        """
        验证所有共享记忆的完整性。

        返回:
            {"total": int, "valid": int, "invalid": int, "failures": [str]}
        """
        if self.memory_hub is None:
            return {"total": 0, "valid": 0, "invalid": 0, "failures": []}
        return self.memory_hub.verify_all()

    # ------------------------------------------------------------------
    # 状态查询
    # ------------------------------------------------------------------

    def status(self) -> dict:
        s = {
            "sid": self.sid,
            "mode": self.mode,
            "governance": self.governance_mode,
            "world_version": self.w.version,
            "self_identity": self.self_model.identity(),
            "intent": self.env.intent,
            "history_len": len(self.env.history),
            "last_kid": self._last_kid,
        }
        if self.mode == "web" and hasattr(self.w, "current_page"):
            s["current_url"] = self.w.current_page.get("url", "")
        if self.memory_hub is not None:
            s["memory_count"] = len(
                self.memory_hub.load_by_sid(self.sid)
            )
        # 治理状态
        if self.tri_spin:
            report = self.tri_spin.report()
            s["governance_report"] = {
                "aiс_valid": report.aic_valid,
                "vacuum_risk": report.vacuum_risk,
                "ark_status": report.ark_status,
                "gcd_blocks": report.gcd_stats.get("blocks", 0),
                "accountable": report.accountable,
            }
        if self.ritual:
            s["ratify_status"] = self.ritual.status()
        return s

    # ------------------------------------------------------------------
    # 生命周期
    # ------------------------------------------------------------------

    def close(self) -> None:
        if self.mode == "web" and hasattr(self._executor, "close"):
            self._executor.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
