from core.phi_scheduler import PhiScheduler


class CarbonSiliconGAN:
    """
    Carbon-Silicon GAN:
    G-Core（生成侧）由 LLM 生成候选响应；
    D-Core（判别侧）先做语义矛盾检测，再经 Φ-Scheduler 过滤。
    通过则更新世界模型，否则返回 None 并给出拒绝原因。
    """

    def __init__(self, llm_router, world_model):
        self.llm = llm_router
        self.w = world_model
        self.phi = PhiScheduler()

    def step(self, env, user_input: str):
        """
        执行一步 GAN 推演。
        返回 (output: str | None, reason: str)
        """
        # G-Core: 生成候选
        intent = env.get("intent", "") if isinstance(env, dict) else getattr(env, "intent", "")
        prompt = f"{intent}\n{user_input}"
        candidate = self.llm.complete(prompt)

        # D-Core: 语义矛盾检测（简单规则）
        if "矛盾" in candidate or "未定义" in candidate:
            return None, "D-Core: 语义矛盾"

        # Φ Check: 世界一致性
        new_psi = self.w.encode(candidate)
        ok, phi_val = self.phi.check(self.w, new_psi)

        if ok:
            self.w.update(candidate)
            return candidate, "Accepted"
        else:
            return None, f"D-Core: Φ={phi_val:.3f} too low"
