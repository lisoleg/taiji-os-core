"""
core/gpt4_baseline.py — GPT-4 零样本矛盾检测 Baseline
使用 OpenAI API (GPT-4o) 作为外部验证 baseline

与 D-Core (SelfConsistencyLoop) 使用相同的 prompt，但通过 GPT-4 调用。
提供与 D-Core 相同的接口，方便在消融实验中进行对比。
"""

import os
import logging
import time

logger = logging.getLogger("taiji.gpt4_baseline")

# 复用 D-Core 的语义矛盾检测 prompt
from core.self_consistency_loop import DCORE_SEMANTIC_PROMPT


class GPT4ContradictionDetector:
    """
    GPT-4o 零样本矛盾检测器。

    使用与 D-Core 完全相同的检测 prompt，但通过 OpenAI GPT-4o API 调用。
    作为外部 baseline，用于验证 D-Core (DeepSeek) 的检测能力。

    参数:
        api_key  : OpenAI API key（默认从 OPENAI_API_KEY 环境变量读取）
        model    : 模型名称 (默认 gpt-4o)
    """

    def __init__(self, api_key: str = None, model: str = "gpt-4o"):
        self.model = model
        self._cache: dict[tuple, dict] = {}
        self._api_key = api_key or os.environ.get("OPENAI_API_KEY", "")

        if self._api_key:
            try:
                import openai
                self._client = openai.OpenAI(api_key=self._api_key)
                self._available = True
                logger.info(f"GPT4ContradictionDetector 已初始化: model={model}")
            except Exception as e:
                logger.warning(f"OpenAI 客户端初始化失败: {e}")
                self._client = None
                self._available = False
        else:
            self._client = None
            self._available = False
            logger.warning("未设置 OPENAI_API_KEY，GPT-4 baseline 不可用")

    # ------------------------------------------------------------------
    # 公开 API
    # ------------------------------------------------------------------

    def detect(self, statement_a: str, statement_b: str) -> dict:
        """
        检测两个陈述是否存在语义矛盾。

        返回:
            {
                "is_contradiction": bool,
                "verdict": str,         # "CONTRADICTION" | "CONSISTENT"
                "method": str,          # "gpt4-zero-shot" | "gpt4-cache" | "gpt4-error"
                "latency_ms": float,    # API 调用耗时
            }
        """
        # 缓存检查
        cache_key = (statement_a[:200], statement_b[:200])
        if cache_key in self._cache:
            cached = self._cache[cache_key]
            return {**cached, "method": "gpt4-cache"}

        if not self._available:
            return {
                "is_contradiction": False,
                "verdict": "ERROR",
                "method": "gpt4-error",
                "latency_ms": 0,
            }

        try:
            t0 = time.perf_counter()
            verdict = self._detect_api(statement_a, statement_b)
            latency = (time.perf_counter() - t0) * 1000

            result = {
                "is_contradiction": verdict == "CONTRADICTION",
                "verdict": verdict,
                "method": "gpt4-zero-shot",
                "latency_ms": round(latency, 1),
            }
            self._cache[cache_key] = {
                "is_contradiction": result["is_contradiction"],
                "verdict": result["verdict"],
                "latency_ms": result["latency_ms"],
            }
            return result

        except Exception as e:
            logger.warning(f"GPT-4 API 调用失败: {e}")
            return {
                "is_contradiction": False,
                "verdict": "ERROR",
                "method": "gpt4-error",
                "latency_ms": 0,
            }

    # ------------------------------------------------------------------
    # 内部方法
    # ------------------------------------------------------------------

    def _detect_api(self, statement_a: str, statement_b: str) -> str:
        """调用 GPT-4o API 做语义矛盾检测。"""
        import openai

        prompt = DCORE_SEMANTIC_PROMPT.format(
            statement_a=statement_a, statement_b=statement_b
        )
        resp = self._client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=50,
        )
        content = resp.choices[0].message.content.strip().upper()

        if "CONTRADICTION" in content:
            return "CONTRADICTION"
        elif "CONSISTENT" in content:
            return "CONSISTENT"
        # 模糊回复检查
        if any(kw in content for kw in ["矛盾", "冲突", "不一致"]):
            return "CONTRADICTION"
        if any(kw in content for kw in ["一致", "无矛盾", "协调"]):
            return "CONSISTENT"
        return "CONSISTENT"  # 默认保守

    # ------------------------------------------------------------------
    # 统计
    # ------------------------------------------------------------------

    @property
    def is_available(self) -> bool:
        return self._available

    def stats(self) -> dict:
        return {
            "model": self.model,
            "available": self._available,
            "cache_size": len(self._cache),
        }
