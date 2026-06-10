"""core/step_result.py — 统一步骤结果数据类

v4.1 新增：将 run() 的返回值从裸 str 升级为结构化 dataclass，
同时通过 to_legacy_str() 保证向后兼容。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional


@dataclass
class StepResult:
    """单步推演结果。

    属性:
        output:             主文本输出
        accepted:           是否通过
        phi_value:          Φ余弦相似度
        reason:             "Accepted" 或 拒绝原因
        world_version:      WorldModel版本号
        session_id:         Session ID
        continuation_kid:   Continuation 快照 ID（仅拒绝时有值）
        continuation_proof: Continuation 证明哈希（仅拒绝时有值）
        timestamp:          ISO 8601 UTC 时间戳
        mode:               "text" | "web" | "governed"
        extra:              Governance摘要 / 执行摘要
    """

    output: str
    accepted: bool
    phi_value: float
    reason: str
    world_version: int
    session_id: str
    continuation_kid: Optional[str] = None
    continuation_proof: Optional[str] = None
    timestamp: str = ""
    mode: str = "text"
    extra: str = ""

    def __post_init__(self) -> None:
        """自动填充时间戳（若未提供）。"""
        if not self.timestamp:
            self.timestamp = datetime.now(timezone.utc).isoformat()

    def to_legacy_str(self) -> str:
        """转换为 run() 的历史 str 格式，确保向后兼容。

        返回:
            与 v4 run() 返回值格式完全一致的字符串。
        """
        if self.accepted:
            if self.mode == "governed" and self.extra:
                return f"{self.output}\n\n[Governance]\n{self.extra}"
            elif self.mode == "web" and self.extra:
                return f"{self.output}\n\n[执行摘要]\n{self.extra}"
            return self.output
        # 拒绝时：与 _save_continuation 返回格式对齐
        proof_short = (self.continuation_proof or "")[:12]
        return (
            f"Continuation Saved: {self.continuation_kid}"
            f" | reason: {self.reason} [proof: {proof_short}...]"
        )
