"""core/session.py — TaijiSession: AGI 进程主控

两种工作模式（通过 mode 参数切换）：
  - "text"   : 原始文本推演模式（默认，不需要浏览器）
  - "web"    : 浏览器云脑模式（WebWorldModel + WebPlanner + PlaywrightExecutor）

Web 模式下，TaijiSession 的 run() 方法会：
  1. 用 WebPlanner 将意图转化为浏览器动作步骤列表
  2. 用 PlaywrightExecutor 执行这些步骤（或 Mock 执行）
  3. 将页面快照反馈给 WebWorldModel，更新 ψ
  4. 通过 CarbonSiliconGAN 的 Φ 门控决定是否输出或保存 Continuation
"""
from __future__ import annotations

from core.world_model import WorldModel
from core.carbon_silicon_gan import CarbonSiliconGAN
from core.continuation import Continuation
from core.closure_env import ClosureEnv
from core.self_model import SelfModel


class TaijiSession:
    """
    TaijiSession (AGI Process):
    一个完整的太极OS会话实例，封装世界模型、自我模型、GAN推演与Continuation机制。

    参数:
        sid          : 会话 ID
        llm_router   : HAL LLMRouter 实例
        snapshot_dir : Continuation 快照目录
        mode         : "text"（纯文本）或 "web"（浏览器云脑）
        headless     : Web 模式下是否无头启动浏览器
    """

    def __init__(
        self,
        sid: str,
        llm_router,
        snapshot_dir: str = "snapshots",
        mode: str = "text",
        headless: bool = True,
    ):
        self.sid = sid
        self.snapshot_dir = snapshot_dir
        self.mode = mode

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
        """原始文本模式推演。"""
        result, reason = self.gan.step(self.env.to_dict(), user_input)
        if result:
            self.env.push("assistant", result)
            return result
        # 打包 Continuation 快照
        k = Continuation(
            self.sid,
            self.w.psi.copy(),
            self.env.to_dict(),
            reason,
            snapshot_dir=self.snapshot_dir,
        )
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
            # 保存 Continuation（含浏览器快照）
            env_dict = self.env.to_dict()
            if hasattr(self.w, "page_snapshot"):
                env_dict["__page_snapshot__"] = self.w.page_snapshot()
            k = Continuation(
                self.sid,
                self.w.psi.copy(),
                env_dict,
                reason,
                snapshot_dir=self.snapshot_dir,
            )
            return f"Continuation Saved: {k.kid} | reason: {reason}"

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
        }
        if self.mode == "web" and hasattr(self.w, "current_page"):
            s["current_url"] = self.w.current_page.get("url", "")
        return s

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
