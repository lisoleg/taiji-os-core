"""syscalls/web_planner.py — WebPlanner: 自然语言意图 → 浏览器动作序列

将用户意图转化为浏览器操作步骤列表，供 PlaywrightExecutor 执行。

规划策略：
  - 规则层（无需 LLM）: 识别常见意图模式（搜索/打开/填写/截图等）
  - LLM 层（需要 llm_router）: 将复杂自然语言意图发给 DeepSeek 生成 JSON 步骤列表

步骤格式:
  {"action": "navigate", "params": {"url": "https://..."}}
  {"action": "click", "params": {"selector": "#btn-submit"}}
  {"action": "type", "params": {"selector": "#input", "text": "keyword"}}
  {"action": "read_dom", "params": {}}
  {"action": "screenshot", "params": {}}
  {"action": "respond", "params": {"input": "..."}}
"""
from __future__ import annotations

import json
import re
from typing import Dict, List, Optional


# 常见搜索引擎映射
_SEARCH_ENGINES = {
    "google": "https://www.google.com/search?q={}",
    "baidu": "https://www.baidu.com/s?wd={}",
    "bing": "https://www.bing.com/search?q={}",
}

# URL 直接跳转的触发词
_NAVIGATE_KEYWORDS = ("打开", "访问", "跳转", "go to", "navigate to", "open", "visit")
_SEARCH_KEYWORDS = ("搜索", "查一下", "search", "find", "查询", "搜一搜")
_SCREENSHOT_KEYWORDS = ("截图", "screenshot", "截屏", "snapshot")
_FILL_KEYWORDS = ("填写", "输入", "type", "fill", "write")
_CLICK_KEYWORDS = ("点击", "click", "按", "press")
_SCROLL_KEYWORDS = ("滚动", "scroll", "下拉", "翻页")


class WebPlanner:
    """
    WebPlanner — 将自然语言意图转化为浏览器动作步骤列表。

    优先用规则层快速匹配；规则无法覆盖时调用 LLM 生成结构化步骤。
    """

    SYSTEM_PROMPT = """你是太极OS的浏览器操控规划器。
用户给你一个任务意图，你需要把它转成浏览器操作步骤的JSON列表。

每个步骤的格式：
  {"action": "navigate", "params": {"url": "https://..."}}
  {"action": "click", "params": {"selector": "CSS_SELECTOR_OR_TEXT_DESCRIPTION"}}
  {"action": "type", "params": {"selector": "INPUT_SELECTOR", "text": "要输入的内容"}}
  {"action": "read_dom", "params": {}}
  {"action": "screenshot", "params": {}}
  {"action": "scroll", "params": {"direction": "down", "px": 300}}
  {"action": "wait", "params": {"ms": 1000}}
  {"action": "respond", "params": {"input": "最终总结任务结果"}}

规则：
1. 始终以 respond 步骤结尾，总结执行结果
2. 只返回 JSON 数组，不要有额外文字
3. selector 优先用文本描述如 'text=搜索' 或 'button[type=submit]'

现在，请为以下任务生成步骤列表："""

    def __init__(self, default_engine: str = "baidu"):
        self.default_engine = default_engine

    def plan(
        self,
        intent: str,
        context: Dict,
        llm_router=None,
    ) -> List[Dict]:
        """
        规划浏览器操作步骤。

        优先走规则层，失败时调用 LLM。
        """
        steps = self._rule_plan(intent)
        if steps:
            return steps

        # 规则无法覆盖 → 尝试 LLM 规划
        if llm_router:
            steps = self._llm_plan(intent, llm_router)
            if steps:
                return steps

        # 最终降级：直接让 LLM 回答
        return [{"action": "respond", "params": {"input": intent}}]

    # ------------------------------------------------------------------
    # 规则层规划
    # ------------------------------------------------------------------

    def _rule_plan(self, intent: str) -> List[Dict]:
        lower = intent.lower()

        # 1. 直接 URL
        url_match = re.search(r"https?://\S+", intent)
        if url_match:
            url = url_match.group()
            return [
                {"action": "navigate", "params": {"url": url}},
                {"action": "read_dom", "params": {}},
                {"action": "respond", "params": {"input": f"已打开 {url}，请告诉我下一步"}},
            ]

        # 2. 打开/访问 + 网站名
        if any(kw in lower for kw in _NAVIGATE_KEYWORDS):
            # 提取目标（简单启发：取关键词之后的部分）
            for kw in _NAVIGATE_KEYWORDS:
                if kw in lower:
                    target = intent[lower.index(kw) + len(kw):].strip()
                    if target:
                        # 如果像域名就直接用，否则搜索它
                        if re.match(r"[\w\-]+\.(com|cn|net|org|io|ai)", target):
                            url = f"https://{target}"
                        else:
                            url = _SEARCH_ENGINES[self.default_engine].format(
                                target.replace(" ", "+")
                            )
                        return [
                            {"action": "navigate", "params": {"url": url}},
                            {"action": "read_dom", "params": {}},
                            {"action": "respond", "params": {"input": f"已导航至 {url}"}},
                        ]

        # 3. 搜索意图
        if any(kw in lower for kw in _SEARCH_KEYWORDS):
            for kw in _SEARCH_KEYWORDS:
                if kw in lower:
                    query = intent[lower.index(kw) + len(kw):].strip()
                    if query:
                        url = _SEARCH_ENGINES[self.default_engine].format(
                            query.replace(" ", "+")
                        )
                        return [
                            {"action": "navigate", "params": {"url": url}},
                            {"action": "read_dom", "params": {}},
                            {"action": "respond", "params": {"input": f"搜索 '{query}' 完成"}},
                        ]

        # 4. 截图
        if any(kw in lower for kw in _SCREENSHOT_KEYWORDS):
            return [
                {"action": "screenshot", "params": {}},
                {"action": "respond", "params": {"input": "截图完成"}},
            ]

        # 5. 滚动
        if any(kw in lower for kw in _SCROLL_KEYWORDS):
            direction = "up" if "上" in lower or "up" in lower else "down"
            return [
                {"action": "scroll", "params": {"direction": direction, "px": 400}},
                {"action": "respond", "params": {"input": f"已向{direction}滚动"}},
            ]

        return []  # 规则无法覆盖

    # ------------------------------------------------------------------
    # LLM 层规划
    # ------------------------------------------------------------------

    def _llm_plan(self, intent: str, llm_router) -> List[Dict]:
        """调用 LLM 生成步骤 JSON 列表。"""
        prompt = f"{self.SYSTEM_PROMPT}\n\n任务：{intent}"
        try:
            raw = llm_router.complete(prompt)
            # 提取 JSON 数组部分
            match = re.search(r"\[.*\]", raw, re.DOTALL)
            if match:
                steps = json.loads(match.group())
                if isinstance(steps, list) and all(
                    isinstance(s, dict) and "action" in s for s in steps
                ):
                    return steps
        except Exception:
            pass
        return []
