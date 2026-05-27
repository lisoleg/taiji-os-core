from core.world_model import WorldModel
from core.carbon_silicon_gan import CarbonSiliconGAN
from core.continuation import Continuation
from core.closure_env import ClosureEnv
from core.self_model import SelfModel


class TaijiSession:
    """
    TaijiSession (AGI Process):
    一个完整的太极OS会话实例，封装世界模型、自我模型、GAN推演与Continuation机制。
    """

    def __init__(self, sid: str, llm_router, snapshot_dir: str = "snapshots"):
        self.sid = sid
        self.snapshot_dir = snapshot_dir
        self.w = WorldModel()
        self.self_model = SelfModel(sid)
        self.gan = CarbonSiliconGAN(llm_router, self.w)
        self.env = ClosureEnv(intent="idle")

    def run(self, user_input: str) -> str:
        """执行一轮推演，返回输出或 Continuation ID。"""
        self.env.push("user", user_input)
        result, reason = self.gan.step(self.env.to_dict(), user_input)

        if result:
            self.env.push("assistant", result)
            return result
        else:
            # 打包 Continuation 快照
            k = Continuation(
                self.sid,
                self.w.psi.copy(),
                self.env.to_dict(),
                reason,
                snapshot_dir=self.snapshot_dir,
            )
            return f"Continuation Saved: {k.kid} | reason: {reason}"

    def resume(self, kid: str) -> "TaijiSession":
        """从 Continuation 快照恢复会话状态。"""
        import numpy as np
        k = Continuation.load(kid, snapshot_dir=self.snapshot_dir)
        self.w.psi = k.psi
        self.env = ClosureEnv.from_dict(k.env)
        return self

    def status(self) -> dict:
        return {
            "sid": self.sid,
            "world_version": self.w.version,
            "self_identity": self.self_model.identity(),
            "intent": self.env.intent,
            "history_len": len(self.env.history),
        }
