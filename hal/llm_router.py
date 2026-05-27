import os
import time
import yaml
import openai
from typing import Optional


def _load_config(path: str = "config.yaml") -> dict:
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    return {}


class LLMRouter:
    """
    HAL LLMRouter: 硬件抽象层 LLM 路由器。
    支持 DeepSeek / Claude / Local，含自动 fallback 逻辑。
    """

    def __init__(self, config_path: str = "config.yaml"):
        cfg = _load_config(config_path)
        self.cfg = cfg
        llm_cfg = cfg.get("llm", {})
        fallback_cfg = cfg.get("fallback", {})
        taiji_cfg = cfg.get("taiji", {})

        self.provider = llm_cfg.get("provider", "mock")
        self.base_url = llm_cfg.get("base_url", "")
        self.api_key = os.environ.get("DEEPSEEK_API_KEY", llm_cfg.get("api_key", ""))
        self.model = llm_cfg.get("model", "deepseek-reasoner")
        self.temperature = llm_cfg.get("temperature", 0.2)

        self.fallback_enabled = fallback_cfg.get("enabled", False)
        self.fallback_api_key = os.environ.get("CLAUDE_API_KEY", fallback_cfg.get("api_key", ""))
        self.fallback_base_url = fallback_cfg.get("base_url", "")
        self.fallback_model = fallback_cfg.get("model", "claude-sonnet-4-20260514")

        self.max_retry = taiji_cfg.get("max_retry", 3)

    def complete(self, prompt: str) -> str:
        """调用 LLM 生成补全，失败时自动 fallback。"""
        for attempt in range(self.max_retry):
            try:
                return self._call_primary(prompt)
            except Exception as e:
                if self.fallback_enabled and attempt == self.max_retry - 1:
                    try:
                        return self._call_fallback(prompt)
                    except Exception as fe:
                        return f"[LLMRouter Error] primary={e}, fallback={fe}"
                time.sleep(0.5 * (attempt + 1))
        return "[LLMRouter Error] max retries exceeded"

    def _call_primary(self, prompt: str) -> str:
        if self.provider == "mock" or not self.api_key or self.api_key.startswith("${"):
            return self._mock_response(prompt)
        client = openai.OpenAI(api_key=self.api_key, base_url=self.base_url)
        resp = client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.temperature,
        )
        return resp.choices[0].message.content

    def _call_fallback(self, prompt: str) -> str:
        if not self.fallback_api_key or self.fallback_api_key.startswith("${"):
            return self._mock_response(prompt)
        client = openai.OpenAI(api_key=self.fallback_api_key, base_url=self.fallback_base_url)
        resp = client.chat.completions.create(
            model=self.fallback_model,
            messages=[{"role": "user", "content": prompt}],
        )
        return resp.choices[0].message.content

    def _mock_response(self, prompt: str) -> str:
        """离线 mock：用于测试，不调用真实 API。"""
        keywords = ["矛盾", "未定义", "错误"]
        for kw in keywords:
            if kw in prompt:
                return f"检测到语义{kw}，无法生成一致响应。"
        return f"[Mock] 已处理输入：{prompt[:80]}"
