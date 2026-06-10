"""core/config_validator.py — 配置校验器（纯Python，无pydantic）

v4.1 新增：在运行前对 config.yaml 做结构化校验，
给出 errors（阻断级）和 warnings（提示级）。
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from typing import Any

import yaml


@dataclass
class ValidationResult:
    """校验结果。

    属性:
        valid:    是否通过校验（无 error 时为 True）
        errors:   阻断级错误列表
        warnings: 提示级警告列表
    """

    valid: bool = True
    errors: list = field(default_factory=list)
    warnings: list = field(default_factory=list)


class ConfigValidator:
    """config.yaml 校验器。

    用法::

        v = ConfigValidator.validate("config.yaml")
        if not v.valid:
            for e in v.errors: print("ERROR:", e)
        for w in v.warnings: print("WARN:", w)
    """

    # 允许的 phi_mode 枚举
    _PHI_MODES = {"static", "adaptive"}

    @classmethod
    def validate(cls, config_path: str) -> ValidationResult:
        """从文件路径校验配置。

        参数:
            config_path: config.yaml 文件路径

        返回:
            ValidationResult 实例
        """
        if not os.path.exists(config_path):
            result = ValidationResult(valid=False)
            result.errors.append(f"配置文件不存在: {config_path}")
            return result

        try:
            with open(config_path, "r", encoding="utf-8") as f:
                config = yaml.safe_load(f) or {}
        except yaml.YAMLError as exc:
            result = ValidationResult(valid=False)
            result.errors.append(f"YAML 解析失败: {exc}")
            return result

        return cls.validate_dict(config)

    @classmethod
    validate_dict = lambda cls, config: cls._validate_dict_impl(cls, config)  # noqa: E731

    @classmethod
    def _validate_dict_impl(cls, _self_or_cls, config: dict) -> ValidationResult:
        """从字典校验配置。

        参数:
            config: 已解析的配置字典

        返回:
            ValidationResult 实例
        """
        result = ValidationResult()

        taiji = config.get("taiji", {})
        embedding = config.get("embedding", {})
        llm = config.get("llm", {})

        # 委托给各子校验器
        cls._validate_phi(taiji, result)
        cls._validate_embedding(embedding, result)
        cls._validate_llm(llm, result)

        result.valid = len(result.errors) == 0
        return result

    # ------------------------------------------------------------------
    # 私有子校验器
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_phi(taiji: dict, result: ValidationResult) -> None:
        """校验 taiji.phi_* 相关配置。"""
        # phi_threshold ∈ [0, 1]
        threshold = taiji.get("phi_threshold")
        if threshold is not None:
            if not isinstance(threshold, (int, float)):
                result.errors.append("taiji.phi_threshold 必须为数字")
            elif not (0 <= threshold <= 1):
                result.errors.append(
                    f"taiji.phi_threshold={threshold} 不在 [0, 1] 范围内"
                )
        else:
            result.warnings.append("taiji.phi_threshold 未设置，将使用默认值 0.65")

        # phi_mode ∈ {"static", "adaptive"}
        mode = taiji.get("phi_mode")
        if mode is not None:
            if mode not in ConfigValidator._PHI_MODES:
                result.errors.append(
                    f"taiji.phi_mode='{mode}' 不合法，允许值: {ConfigValidator._PHI_MODES}"
                )
        else:
            result.warnings.append("taiji.phi_mode 未设置，将使用默认值 'static'")

        # adaptive 模式下的额外约束
        if mode == "adaptive":
            window = taiji.get("phi_adaptive_window")
            if window is not None:
                if not isinstance(window, int) or window < 5:
                    result.errors.append(
                        f"taiji.phi_adaptive_window={window} 必须 >= 5"
                    )

            alpha = taiji.get("phi_adaptive_alpha")
            if alpha is not None:
                if not isinstance(alpha, (int, float)) or not (0 < alpha <= 1):
                    result.errors.append(
                        f"taiji.phi_adaptive_alpha={alpha} 不在 (0, 1] 范围内"
                    )

            min_t = taiji.get("phi_adaptive_min")
            max_t = taiji.get("phi_adaptive_max")
            if min_t is not None and max_t is not None:
                if not (min_t < max_t):
                    result.errors.append(
                        f"taiji.phi_adaptive_min={min_t} 必须 < phi_adaptive_max={max_t}"
                    )

    @staticmethod
    def _validate_embedding(embedding: dict, result: ValidationResult) -> None:
        """校验 embedding 配置。"""
        model = embedding.get("model")
        if not model or (isinstance(model, str) and not model.strip()):
            result.errors.append("embedding.model 不能为空")

    @staticmethod
    def _validate_llm(llm: dict, result: ValidationResult) -> None:
        """校验 llm 配置。"""
        model = llm.get("model")
        if not model or (isinstance(model, str) and not model.strip()):
            result.errors.append("llm.model 不能为空")

        api_key = llm.get("api_key", "")
        if api_key and isinstance(api_key, str) and re.match(r"^\$\{.+\}$", api_key):
            result.warnings.append(
                "llm.api_key 仍为模板变量（${...}），运行前请替换为真实密钥"
            )
