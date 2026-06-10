"""core/session.py — TaijiSession v3.1: AGI 进程主控 (Walrus Memory)

两种工作模式（通过 mode 参数切换）：
  - "text"   : 原始文本推演模式（默认，不需要浏览器）
  - "web"    : 浏览器云脑模式（WebWorldModel + WebPlanner + PlaywrightExecutor）

Walrus Memory v3.1 升级:
  - 可选 MemoryHub 集成，跨会话记忆共享
  - Continuation v2 带 SHA-256 integrity proof 链
  - parent_kid 记忆图谱
"""
from __future__ import annotations

from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from core.uscs_mmu import PageTable
    from core.preemptive_scheduler import PCB, PreemptiveScheduler

from core.world_model import WorldModel
from core.carbon_silicon_gan import CarbonSiliconGAN
from core.continuation import Continuation
from core.closure_env import ClosureEnv
from core.self_model import SelfModel


class TaijiSession:
    """
    TaijiSession (AGI Process) v3.1 — Walrus Memory 集成。

    参数:
        sid          : 会话 ID
        llm_router   : HAL LLMRouter 实例
        snapshot_dir : Continuation 快照目录
        mode         : "text"（纯文本）或 "web"（浏览器云脑）
        headless     : Web 模式下是否无头启动浏览器
        memory_hub   : 可选 MemoryHub 实例（跨会话记忆共享）
    """

    def __init__(
        self,
        sid: str,
        llm_router,
        snapshot_dir: str = "snapshots",
        mode: str = "text",
        headless: bool = True,
        memory_hub=None,
        page_table: Optional["PageTable"] = None,
        pcb: Optional["PCB"] = None,
        scheduler: Optional["PreemptiveScheduler"] = None,
    ):
        self.sid = sid
        self.snapshot_dir = snapshot_dir
        self.mode = mode
        self.memory_hub = memory_hub
        # ★ USCS 内核绑定 ★
        self.page_table = page_table  # PageTable，由 PreemptiveScheduler.register() 创建
        self.pcb = pcb                # PCB，由 PreemptiveScheduler.register() 创建
        self._scheduler = scheduler   # PreemptiveScheduler 引用
        self._last_kid: Optional[str] = None  # 上一个 Continuation ID（proof 链用）

        # 自动注册到 MemoryHub
        if memory_hub:
            memory_hub.register(sid, {"mode": mode})

        # 根据模式选择 WorldModel 和 执行器
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

    @property
    def pid(self) -> Optional[str]:
        """进程 ID（与 PCB 绑定后才有效）。"""
        return self.pcb.pid if self.pcb else None

    # ------------------------------------------------------------------
    # 主执行入口
    # ------------------------------------------------------------------

    def run(self, user_input: str) -> str:
        """执行一轮推演，返回输出或 Continuation ID。"""
        self.env.push("user", user_input)

        if self.mode == "web":
            return self._run_web(user_input)
        return self._run_text(user_input)

    def _run_text(self, user_input: str) -> str:
        """原始文本模式推演（Walrus Memory proof 链）。"""
        result, reason = self.gan.step(self.env.to_dict(), user_input)
        if result:
            self.env.push("assistant", result)
            return result
        # 打包 Continuation 快照（带 proof 链）
        k = self._save_continuation(self.env.to_dict(), reason)
        return f"Continuation Saved: {k.kid} | reason: {reason}"

    def _run_web(self, user_input: str) -> str:
        """
        浏览器云脑模式：
        意图 → WebPlanner → 步骤列表 → PlaywrightExecutor → 结果 → GAN 门控
        """
        # Step 1: 规划浏览器步骤
        steps = self._planner.plan(
            user_input,
            context=self.env.to_dict(),
            llm_router=self.llm_router,
        )

        # Step 2: 执行浏览器步骤
        exec_results = self._executor.execute(steps, llm_router=self.llm_router)

        # Step 3: 从执行结果中提取 respond 动作的输出作为候选回答
        respond_outputs = [
            r["output"]
            for r in exec_results
            if r["action"] == "respond" and r["status"] == "ok"
        ]
        candidate_text = respond_outputs[-1] if respond_outputs else str(exec_results)

        # Step 4: GAN Φ 门控（一致性检验）
        result, reason = self.gan.step(self.env.to_dict(), candidate_text)

        if result:
            self.env.push("assistant", result)
            # 附加执行摘要
            summary = self._exec_summary(exec_results)
            return f"{result}\n\n[执行摘要]\n{summary}"
        else:
            # 保存 Continuation（含浏览器快照 + Walrus proof 链）
            env_dict = self.env.to_dict()
            if hasattr(self.w, "page_snapshot"):
                env_dict["__page_snapshot__"] = self.w.page_snapshot()
            k = self._save_continuation(env_dict, reason)
            return f"Continuation Saved: {k.kid} | reason: {reason}"

    def _save_continuation(self, env_dict: dict, reason: str) -> Continuation:
        """
        保存 Continuation 快照，支持 Walrus Memory proof 链。

        如果集成了 MemoryHub，使用 hub.store()（自动链接 parent_kid 的 proof）。
        否则回退到原始的 Continuation() 构造。
        """
        if self.memory_hub:
            k = self.memory_hub.store(
                sid=self.sid,
                psi=self.w.psi.copy(),
                env=env_dict,
                reason=reason,
                snapshot_dir=self.snapshot_dir,
                parent_kid=self._last_kid,
            )
        else:
            parent_proof = None
            if self._last_kid:
                try:
                    prev = Continuation.load(self._last_kid, self.snapshot_dir)
                    parent_proof = prev.proof
                except Exception:
                    pass
            k = Continuation(
                self.sid,
                self.w.psi.copy(),
                env_dict,
                reason,
                snapshot_dir=self.snapshot_dir,
                parent_kid=self._last_kid,
                parent_proof=parent_proof,
            )
        self._last_kid = k.kid
        return k

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
        # 恢复浏览器快照（Web 模式）
        if self.mode == "web" and "__page_snapshot__" in env_dict:
            self.w.restore_snapshot(env_dict.pop("__page_snapshot__"))
        self.env = ClosureEnv.from_dict(env_dict)
        return self

    # ------------------------------------------------------------------
    # 状态查询
    # ------------------------------------------------------------------

    def status(self) -> dict:
        s = {
            "sid": self.sid,
            "mode": self.mode,
            "world_version": self.w.version,
            "self_identity": self.self_model.identity(),
            "intent": self.env.intent,
            "history_len": len(self.env.history),
            "last_kid": self._last_kid,
            "hub_connected": self.memory_hub is not None,
        }
        if self.mode == "web" and hasattr(self.w, "current_page"):
            s["current_url"] = self.w.current_page.get("url", "")
        return s

    # ------------------------------------------------------------------
    # Walrus Memory: 跨会话搜索
    # ------------------------------------------------------------------

    def search_memory(self, keyword: str, limit: int = 20) -> list[dict]:
        """通过 MemoryHub 跨会话搜索记忆。"""
        if not self.memory_hub:
            return []
        return self.memory_hub.search(keyword=keyword, limit=limit)

    def verify_integrity(self) -> dict:
        """验证本会话所有 Continuation 的 proof 链。"""
        if self.memory_hub:
            return self.memory_hub.verify_all(self.snapshot_dir)
        # 独立验证
        return self._standalone_verify()

    def _standalone_verify(self) -> dict:
        import os as _os
        report = {"total": 0, "passed": 0, "failed": [], "skipped": 0}
        if not _os.path.isdir(self.snapshot_dir):
            return report
        proof_map = {}
        for fname in _os.listdir(self.snapshot_dir):
            if fname.endswith(".json") and not fname.startswith("_"):
                kid = fname.replace(".json", "")
                try:
                    proof_map[kid] = Continuation.load(kid, self.snapshot_dir)
                except Exception:
                    continue
        for kid, k in proof_map.items():
            report["total"] += 1
            parent_proof = proof_map[k.parent_kid].proof if (k.parent_kid and k.parent_kid in proof_map) else None
            if k.verify(parent_proof):
                report["passed"] += 1
            else:
                report["failed"].append({"kid": kid})
        return report

    # ------------------------------------------------------------------
    # 生命周期
    # ------------------------------------------------------------------

    def close(self) -> None:
        """关闭浏览器（Web 模式）。"""
        if self.mode == "web" and hasattr(self._executor, "close"):
            self._executor.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
