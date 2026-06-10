"""
core/carbon_silicon_gan.py — CarbonSiliconGAN v4.1 (历史别名)

Carbon-Silicon GAN:
  G-Core（生成侧）由 LLM 生成候选响应；
  D-Core（判别侧）使用 SelfConsistencyLoop 做语义矛盾检测，
  再经 Φ-Scheduler 过滤。

v4.1 术语诚实性:
  - CarbonSiliconGAN 保留为历史别名
  - D-Core 正式名称为 SelfConsistencyLoop（core/self_consistency_loop.py）
  - 本文件是向后兼容的别名 shim
"""

# 向后兼容：从新模块导入 D-Core
from core.self_consistency_loop import SelfConsistencyLoop
from core.phi_scheduler import PhiScheduler


class CarbonSiliconGAN:
    """
    Carbon-Silicon GAN v4.1

    G-Core（生成侧）由 LLM 生成候选响应；
    D-Core（判别侧）先做语义矛盾检测，再经 Φ-Scheduler 过滤。
    通过则更新世界模型，否则返回 None 并给出拒绝原因。

    v4.1 升级: D-Core 使用 SelfConsistencyLoop 语义检测
    """

    def __init__(self, llm_router, world_model):
        self.llm = llm_router
        self.w = world_model
        self.phi = PhiScheduler()
        self._d_core = SelfConsistencyLoop()  # 语义矛盾检测器

    def step(self, env, user_input: str):
        """
        执行一步 GAN 推演。
        返回 (output: str | None, reason: str)
        """
        # G-Core: 生成候选
        intent = env.get("intent", "") if isinstance(env, dict) else getattr(env, "intent", "")
        prompt = f"{intent}\n{user_input}"
        candidate = self.llm.complete(prompt)

        # D-Core Layer 1: SelfConsistencyLoop 语义矛盾检测
        prev_output = ""
        if isinstance(env, dict):
            history = env.get("history", [])
            if history:
                # 取最近的 assistant 输出作为上下文
                for msg in reversed(history):
                    if isinstance(msg, dict) and msg.get("role") == "assistant":
                        prev_output = msg.get("content", "")
                        break

        if prev_output:
            is_contra, verdict, method = self._d_core.detect_contradiction(
                prev_output, candidate
            )
            if is_contra:
                return None, f"D-Core: 语义矛盾 ({method})"

        # D-Core Layer 2: Φ Check — 世界一致性
        new_psi = self.w.encode(candidate)
        ok, phi_val = self.phi.check(self.w, new_psi)

        if ok:
            self.w.update(candidate)
            return candidate, "Accepted"
        else:
            return None, f"D-Core: Φ={phi_val:.3f} < threshold={self.phi.threshold:.3f}"

    # ------------------------------------------------------------------
    # 别名兼容
    # ------------------------------------------------------------------

    @property
    def d_core(self) -> SelfConsistencyLoop:
        """获取底层语义矛盾检测器。"""
        return self._d_core
