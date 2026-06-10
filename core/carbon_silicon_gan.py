"""Backward-compatible alias: CarbonSiliconGAN → SelfConsistencyLoop

v4.1 重命名：CarbonSiliconGAN → SelfConsistencyLoop（诚实命名）
旧名称 CarbonSiliconGAN 保留为别名，确保所有旧代码不 break。

如需使用新的语义矛盾检测（DeepSeek API 零样本），请直接用 SelfConsistencyLoop。
"""

from __future__ import annotations

from core.self_consistency_loop import SelfConsistencyLoop as CarbonSiliconGAN

__all__ = ["CarbonSiliconGAN"]
