"""core/self_consistency_loop.py — 自洽性推演循环

v4.8.0: 自适应三态衰减 CV → DriftDetector v1.4 (STABLE/DRIFT/RECOVERY)

G-Core 由 LLM 生成候选，D-Core 做语义矛盾检测，
经 PhiScheduler 门控过滤后更新世界模型。

δ-mem L1/L2 融合集成：
  - 每次 LLM 调用前从 S 矩阵读取残差注入 prompt
  - G-Core S 更新始终允许（保留学习能力）
  - D-Core S 更新在 ψ 漂移时降低 β（保留学习但阻尼噪音）
  - ψ 漂移检测：自适应指数衰减 CV (STABLE=0.70/DRIFT=0.35/RECOVERY=0.55)

D-Core 双模式：
  - "semantic" (默认): 调用 DeepSeek API 做零样本矛盾检测
  - "keyword": 回退到关键词匹配（简单规则，离线/测试用）
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np

from core.phi_scheduler import PhiScheduler
from core.embedding_adapter import (
    embed_to_key,
    embed_to_value,
    embed_to_query,
    residual_to_context,
    delta_to_attention_hint,
)
from core.drift_detector import DriftDetector

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------
# D-Core 语义检测 Prompt（零样本矛盾检测）
# ------------------------------------------------------------------

DCORE_SEMANTIC_PROMPT = """你是一个语义一致性检测器。请判断以下AI生成的响应是否包含逻辑矛盾、自我矛盾或与常识矛盾的内容。

上下文意图：{intent}
用户输入：{user_input}
AI响应：{candidate}

请严格按以下格式回答，不要输出其他内容：
VERDICT: CONTRADICTION 或 VERDICT: CONSISTENT
REASON: 如果CONTRADICTION简述矛盾点，否则写"无矛盾\""""


class SelfConsistencyLoop:
    """自洽性推演循环（v4.1 前称 CarbonSiliconGAN）。

    G-Core（生成侧）由 LLM 生成候选响应；
    D-Core（判别侧）先做语义矛盾检测，再经 PhiScheduler 过滤。
    通过则更新世界模型，否则返回 None 并给出拒绝原因。

    用法::

        loop = SelfConsistencyLoop(llm_router, world_model, dcore_mode="semantic")
        output, reason = loop.step(env_dict, "用户输入")
        if output:
            print(f"通过: {output}")
        else:
            print(f"拒绝: {reason}")
    """

    DCORE_MODES = ("semantic", "keyword")

    def __init__(
        self, llm_router, world_model, dcore_mode: str = "semantic",
        delta_fusion=None,
    ):
        """初始化推演循环（v4.2: δ-mem 集成 + 漂移检测）。

        参数:
            llm_router:   LLMRouter 实例，用于 G-Core 生成和 D-Core 检测
            world_model:  WorldModel 实例，用于 ψ 更新和 Φ 计算
            dcore_mode:   "semantic"（DeepSeek API 语义检测）或 "keyword"（关键词回退）
            delta_fusion: DeltaFusion 实例（可选），启用 δ-mem L1/L2 融合
        """
        self.llm = llm_router
        self.w = world_model
        self.dcore_mode = dcore_mode if dcore_mode in self.DCORE_MODES else "semantic"
        self.phi = PhiScheduler()

        # ── δ-mem L1/L2 融合 (v4.2) ──
        self.delta_fusion = delta_fusion

        # ── ψ 漂移检测器 (v4.8: 自适应三态衰减) ──
        self.drift_detector = DriftDetector(
            window_size=20, cv_threshold=0.30,
            min_samples_before_detect=5, hysteresis_rounds=2,
            adaptive=True,  # v4.8: 三态自适应 decay
        )

    # ------------------------------------------------------------------
    # 主推演入口（v4.2: δ-mem 注入 + 漂移检测）
    # ------------------------------------------------------------------

    def step(self, env, user_input: str):
        """执行一步自洽性推演。

        流程: δ-mem read → G-Core 生成 → δ-mem ingest → D-Core 矛盾检测
              → Φ 一致性门控 → drift 检测 → S 更新控制

        参数:
            env:        当前环境（dict 或 ClosureEnv）
            user_input: 用户输入文本

        返回:
            (output: str | None, reason: str)
            output 非 None 表示通过，None 表示被拒绝
        """
        intent = (
            env.get("intent", "")
            if isinstance(env, dict)
            else getattr(env, "intent", "")
        )
        prompt = f"{intent}\n{user_input}"

        # ── δ-mem L1: 读取残差注入 G-Core prompt ──
        augmented_prompt = prompt
        if self.delta_fusion is not None:
            q = embed_to_query(prompt, self.w)
            residual = self.delta_fusion.query(q)
            ctx = residual_to_context(residual)
            if ctx:
                augmented_prompt = f"[δ-mem: {ctx}]\n{prompt}"

        # G-Core: 生成候选
        candidate = self.llm.complete(augmented_prompt)

        # ── δ-mem L1: 更新 S 矩阵 (G-Core — always ingest) ──
        # v4.6.3: G-Core S update is always allowed, even during drift.
        # The S matrix benefits from continuous learning; drift protection
        # uses β reduction for D-Core ingest (more noise-prone detection phase).
        if self.delta_fusion is not None:
            k = embed_to_key(prompt, self.w)
            v = embed_to_value(candidate, self.w)
            self.delta_fusion.ingest(k, v)

        # D-Core: 语义矛盾检测
        is_contradiction, d_reason = self._d_core_detect(candidate, user_input, env)
        if is_contradiction:
            return None, f"D-Core: {d_reason}"

        # Φ Check: 世界一致性门控
        new_psi = self.w.encode(candidate)
        ok, phi_val = self.phi.check(self.w, new_psi)

        # ── ψ 漂移检测 (v4.6.3: 降 β — 阻尼 D-Core 噪音，不阻断学习) ──
        if self.delta_fusion is not None:
            self.drift_detector.push(phi_val)
            if self.drift_detector.is_drifting():
                # 降 β 策略：β ← β × 0.2 保持 D-Core S 学习但大幅阻尼
                # 这样可以(1)不完全丢失检测阶段的信号
                # (2)防止漂移期噪音过度污染 S 矩阵
                logger.info(
                    f"δ-mem: ψ drift confirmed (CV={self.drift_detector.current_cv:.3f}, "
                    f"streak={self.drift_detector._drifting_streak}), "
                    f"reducing D-Core S beta to 20%"
                )

        if ok:
            self.w.update(candidate)
            return candidate, "Accepted"
        else:
            return None, f"D-Core: Φ={phi_val:.3f} too low (threshold={self.phi.get_threshold():.3f})"

    # ------------------------------------------------------------------
    # D-Core 矛盾检测
    # ------------------------------------------------------------------

    def _d_core_detect(self, candidate: str, user_input: str, env) -> tuple:
        """检测候选输出是否包含矛盾。

        返回:
            (is_contradiction: bool, reason: str)
        """
        if self.dcore_mode == "semantic":
            return self._d_core_check_semantic(candidate, user_input, env)
        return self._d_core_check_keyword(candidate)

    def _d_core_check_semantic(
        self, candidate: str, user_input: str, env
    ) -> tuple:
        """使用 DeepSeek API 做零样本语义矛盾检测（v4.2: δ-mem 注入）。

        API 失败时自动回退到关键词检测，并记录 warning。
        """
        intent = (
            env.get("intent", "")
            if isinstance(env, dict)
            else getattr(env, "intent", "")
        )

        detection_prompt = DCORE_SEMANTIC_PROMPT.format(
            intent=intent or "通用对话",
            user_input=user_input[:500],
            candidate=candidate[:500],
        )

        # ── δ-mem L1: 读取 attention delta 注入 D-Core prompt ──
        augmented_detection = detection_prompt
        if self.delta_fusion is not None:
            q = embed_to_query(detection_prompt, self.w)
            k = embed_to_key(detection_prompt, self.w)
            attn_delta = self.delta_fusion.query_attention_delta(q, k)
            hint = delta_to_attention_hint(attn_delta)
            if hint:
                augmented_detection = f"[δ-mem signal: {hint}]\n{detection_prompt}"

        try:
            response = self.llm.complete(augmented_detection)
            # ── δ-mem: 更新 D-Core 结果到 S (v4.6.3: 降 β 替代完全暂停) ──
            if self.delta_fusion is not None:
                smatrix = self.delta_fusion.delta_layer.smatrix
                original_beta = smatrix.beta
                if self.drift_detector.is_drifting():
                    smatrix.beta = original_beta * 0.2  # Drift damping
                try:
                    k2 = embed_to_key(detection_prompt, self.w)
                    v2 = embed_to_value(response, self.w)
                    self.delta_fusion.ingest(k2, v2)
                finally:
                    smatrix.beta = original_beta
            return self._parse_dcore_response(response)
        except Exception as e:
            logger.warning(
                f"D-Core semantic detection failed: {e}, "
                f"falling back to keyword detection"
            )
            return self._d_core_check_keyword(candidate)

    def _d_core_check_keyword(self, candidate: str) -> tuple:
        """关键词回退：检测中文矛盾/未定义。

        保留 v4 原有规则，确保离线/测试可复现。
        """
        contradiction_keywords = ["矛盾", "未定义", "逻辑错误", "不一致"]
        for kw in contradiction_keywords:
            if kw in candidate:
                return True, f"关键词检测: 包含\"{kw}\""
        return False, "无矛盾"

    @staticmethod
    def _parse_dcore_response(response: str) -> tuple:
        """解析 D-Core API 返回的 VERDICT/REASON 格式。

        参数:
            response: LLM 返回的原始文本

        返回:
            (is_contradiction: bool, reason: str)
        """
        response_upper = response.upper().strip()

        # 查找 VERDICT 行
        if "VERDICT: CONTRADICTION" in response_upper:
            # 尝试提取 REASON
            reason = "语义矛盾"
            for line in response.split("\n"):
                if line.strip().upper().startswith("REASON:"):
                    reason = line.split(":", 1)[1].strip()
                    break
            return True, reason

        if "VERDICT: CONSISTENT" in response_upper:
            return False, "无矛盾"

        # 解析失败 → 保守策略：不阻断
        logger.warning(
            f"D-Core response parse failed, treating as consistent: "
            f"{response[:100]}"
        )
        return False, "解析失败，保守通过"
