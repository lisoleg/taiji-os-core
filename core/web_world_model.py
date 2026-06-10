"""core/web_world_model.py — WebWorldModel: 浏览器环境感知 ψ 状态

将网页 DOM 内容、URL 历史、截图描述向量化，作为 WorldModel 的浏览器扩展层。
继承 WorldModel 的所有能力（DeepSeek Embedding / 哈希回退 / Φ计算），
新增 observe_page() 方法，把浏览器快照融入 ψ 更新。
"""
from __future__ import annotations

import json
from typing import Optional

from core.world_model import WorldModel


class WebWorldModel(WorldModel):
    """
    WebWorldModel (ψ_web):
    在原有 WorldModel 基础上追加浏览器感知维度。

    感知信号 = URL + 页面标题 + DOM 摘要 + 截图文字 OCR（可选）

    这些信号被序列化为文本，通过 encode() 转成语义向量后融入 ψ。
    """

    def __init__(self, dim: int = 1536, config_path: str = "config.yaml"):
        super().__init__(dim=dim, config_path=config_path)
        # 当前浏览器快照（最近一次 observe_page 的结果）
        self.current_page: dict = {
            "url": "",
            "title": "",
            "dom_summary": "",
            "screenshot_desc": "",
        }

    # ------------------------------------------------------------------
    # 浏览器快照感知
    # ------------------------------------------------------------------

    def observe_page(
        self,
        url: str = "",
        title: str = "",
        dom_summary: str = "",
        screenshot_desc: str = "",
    ) -> float:
        """
        接收浏览器快照，更新 ψ 并返回 Φ 值（当前页面与记忆的一致性）。

        参数:
            url:            当前页面 URL
            title:          页面标题
            dom_summary:    DOM 关键节点文本摘要（最多 512 字符）
            screenshot_desc:屏幕截图的视觉描述（可选，由 LLM 生成）

        返回:
            phi_val: 本次更新前的余弦相似度（体现"意外程度"）
        """
        self.current_page = {
            "url": url,
            "title": title,
            "dom_summary": dom_summary[:512],
            "screenshot_desc": screenshot_desc[:256],
        }
        # 序列化为文本
        obs_text = self._serialize_page(self.current_page)
        # 先计算 Φ（更新前）
        candidate = self.encode(obs_text)
        phi_val = self.phi(candidate)
        # 更新 ψ
        self.update(obs_text)
        return phi_val

    @staticmethod
    def _serialize_page(page: dict) -> str:
        """将页面快照序列化为统一文本格式。"""
        parts = []
        if page.get("url"):
            parts.append(f"URL: {page['url']}")
        if page.get("title"):
            parts.append(f"Title: {page['title']}")
        if page.get("dom_summary"):
            parts.append(f"DOM: {page['dom_summary']}")
        if page.get("screenshot_desc"):
            parts.append(f"Visual: {page['screenshot_desc']}")
        return " | ".join(parts) if parts else "empty page"

    def page_snapshot(self) -> dict:
        """返回当前页面快照（供 Continuation 序列化使用）。"""
        return dict(self.current_page)

    def restore_snapshot(self, snapshot: dict) -> None:
        """从 Continuation 恢复页面快照。"""
        self.current_page = snapshot
