"""
core/self_consistency_loop.py — D-Core 语义矛盾检测器

Self-Consistency Loop (SCL) 是 CarbonSiliconGAN 的判别侧（D-Core）正式名称。
从 v4.0 起，D-Core 从简单字符串匹配升级为两层检测：

  Layer 1 — 语义矛盾检测 (Semantic Contradiction Detection):
    使用 DeepSeek API 零样本 prompt 判定文本对是否存在语义矛盾。
    在线时走 API，离线时回退到确定性关键词检测。

  Layer 2 — Φ 门控 (Semantic Consistency Gate):
    委托给 PhiScheduler，检查候选向量与世界模型的余弦相似度。

术语诚实性:
  - "CarbonSiliconGAN" → 保留为历史别名（carbon_silicon_gan.py 的 shim）
  - D-Core 正式名称为 SelfConsistencyLoop
  - G-Core 指 LLM 生成侧（不单独成类）

DCORE_SEMANTIC_PROMPT:
  零样本语义矛盾检测 prompt，要求模型输出结构化的 VERDICT。
  输出格式: VERDICT: CONTRADICTION | CONSISTENT
"""

import os
import yaml
import openai
import logging

logger = logging.getLogger("taiji.dcore")

# ------------------------------------------------------------------
# 零样本语义矛盾检测 Prompt
# ------------------------------------------------------------------

DCORE_SEMANTIC_PROMPT = """你是一个严格的语义一致性检测器。给定两个陈述，判定它们是否存在逻辑矛盾。

定义:
- CONTRADICTION: 两个陈述在逻辑上不可能同时为真（例如: "A在B" vs "A在C"，且B≠C）
- CONSISTENT: 两个陈述可以同时为真，即使它们讨论不同方面或细节层次不同

矛盾类型示例:
- 空间矛盾: "猫在沙发上" vs "猫在厨房里"
- 时间矛盾: "会议在上午10点开始" vs "会议下午3点才开始"
- 逻辑矛盾: "系统正常运行" vs "系统已崩溃"
- 数值矛盾: "共有5个用户" vs "注册了3个用户"
- 属性矛盾: "这件衣服是红色的" vs "这件衣服是蓝色的"

请严格按以下格式回复，不要加任何解释:
VERDICT: CONTRADICTION
或
VERDICT: CONSISTENT

陈述A: {statement_a}
陈述B: {statement_b}
"""

# 离线关键词回退列表（多层粒度）
CONTRADICTION_KEYWORDS = [
    # 中文显式矛盾词
    "矛盾", "不一致", "冲突", "相反", "不可能",
    # 英文显式矛盾词
    "contradiction", "inconsistent", "conflict", "impossible",
    # 自相矛盾标记
    "既是", "又不是", "既不是", "又是",
    # 不确定标记（可能表示幻觉）
    "未定义", "undefined", "无法确定",
]


def _load_config(path: str = "config.yaml") -> dict:
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    return {}


class SelfConsistencyLoop:
    """
    D-Core 语义矛盾检测器（Self-Consistency Loop）。
    
    两层检测管道:
      1. 语义矛盾检测: DeepSeek API 零样本 → 关键词回退
      2. Φ 门控: 委托 PhiScheduler（在 CarbonSiliconGAN.step() 中调用）
    
    参数:
        api_key      : DeepSeek API key（可选，不提供则离线运行）
        base_url     : API base URL
        model        : 模型名称
        online       : 强制在线/离线模式（None=自动检测）
    """

    def __init__(
        self,
        api_key: str = "",
        base_url: str = "https://api.deepseek.com/v1",
        model: str = "deepseek-chat",
        online: bool = None,
        config_path: str = "config.yaml",
    ):
        self.model = model
        self._online = online
        self._contradiction_cache: dict[tuple, str] = {}  # 确定性缓存

        # 尝试从 config 读取
        cfg = _load_config(config_path)
        dcore_cfg = cfg.get("dcore", {})
        if not api_key:
            api_key = os.environ.get(
                "DEEPSEEK_API_KEY", dcore_cfg.get("api_key", "")
            )
        if not base_url:
            base_url = dcore_cfg.get("base_url", base_url)
        if not model or model == "deepseek-chat":
            model = dcore_cfg.get("model", model)

        if api_key and not api_key.startswith("${"):
            self._client = openai.OpenAI(api_key=api_key, base_url=base_url)
            if online is None:
                self._online = True
        else:
            self._client = None
            if online is None:
                self._online = False

    # ------------------------------------------------------------------
    # 公开 API
    # ------------------------------------------------------------------

    def detect_contradiction(
        self, statement_a: str, statement_b: str
    ) -> tuple[bool, str, str]:
        """
        检测两个陈述是否存在语义矛盾。
        
        返回:
            (is_contradiction: bool, verdict: str, method: str)
            method ∈ {"api", "keyword", "cache"}
        """
        # 缓存检查
        cache_key = (statement_a[:200], statement_b[:200])
        if cache_key in self._contradiction_cache:
            cached = self._contradiction_cache[cache_key]
            return cached == "CONTRADICTION", cached, "cache"

        # Layer 1a: 在线语义检测
        if self._online:
            try:
                verdict = self._detect_api(statement_a, statement_b)
                self._contradiction_cache[cache_key] = verdict
                return verdict == "CONTRADICTION", verdict, "api"
            except Exception as e:
                logger.warning(
                    f"语义检测API调用失败: {e}, 回退到关键词检测"
                )

        # Layer 1b: 离线关键词检测
        verdict = self._detect_keyword(statement_a, statement_b)
        self._contradiction_cache[cache_key] = verdict
        return verdict == "CONTRADICTION", verdict, "keyword"

    def is_contradiction(self, statement_a: str, statement_b: str) -> bool:
        """便捷方法：仅返回是否矛盾。"""
        is_contra, _, _ = self.detect_contradiction(statement_a, statement_b)
        return is_contra

    # ------------------------------------------------------------------
    # 内部方法
    # ------------------------------------------------------------------

    def _detect_api(self, statement_a: str, statement_b: str) -> str:
        """调用 DeepSeek API 做语义矛盾检测。"""
        prompt = DCORE_SEMANTIC_PROMPT.format(
            statement_a=statement_a, statement_b=statement_b
        )
        resp = self._client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,  # zero temperature for deterministic output
            max_tokens=50,
        )
        content = resp.choices[0].message.content.strip().upper()
        # 解析 VERDICT
        if "CONTRADICTION" in content:
            return "CONTRADICTION"
        elif "CONSISTENT" in content:
            return "CONSISTENT"
        # 模糊回复：检查更多模式
        if any(kw in content for kw in ["矛盾", "冲突", "不一致"]):
            return "CONTRADICTION"
        if any(kw in content for kw in ["一致", "无矛盾", "协调"]):
            return "CONSISTENT"
        # 无法解析，回退到关键词
        return self._detect_keyword(statement_a, statement_b)

    @staticmethod
    def _detect_keyword(statement_a: str, statement_b: str) -> str:
        """离线关键词检测——比简单字符串匹配更健壮。"""
        combined = f"{statement_a} {statement_b}"
        for kw in CONTRADICTION_KEYWORDS:
            if kw.lower() in combined.lower():
                return "CONTRADICTION"
        return "CONSISTENT"

    # ------------------------------------------------------------------
    # 统计 & 调试
    # ------------------------------------------------------------------

    @property
    def mode(self) -> str:
        return "online" if self._online else "offline"

    def stats(self) -> dict:
        return {
            "mode": self.mode,
            "model": self.model if self._online else "keyword-fallback",
            "cache_size": len(self._contradiction_cache),
        }
