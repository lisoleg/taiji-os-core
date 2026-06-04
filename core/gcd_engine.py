"""core/gcd_engine.py — GCD 归约算子 (Constraint Generation Dynamics)

GCD（Constraint Generation Dynamics）归约算子定义（文章原文）：
  作用于 Agent 的工具调用流程：
  - Pre 条件校验阻断非法输入
  - Post 条件校验阻断非法输出

GCD 消除小龙虾死锁定理：
  若 Agent 具备完整 GCD 约束，则其选错工具或传错参数的概率趋近于 0。

实现核心：
  GCDEngine — 为每次工具调用注入 Pre/Post 约束校验
  GCDConstraint — 单个约束定义（前置条件 + 后置条件 + 校验函数）
  GCDRegistry  — 注册和管理所有工具约束
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Callable, Optional


# ---------------------------------------------------------------------------
# 约束定义
# ---------------------------------------------------------------------------

@dataclass
class GCDConstraint:
    """GCD 约束 — Pre 条件 + Post 条件。

    参数:
        tool_name     : 绑定的工具名称（如 "browser.navigate"）
        pre_check     : 前置条件校验函数 (args, kwargs) -> (bool, reason)
        post_check    : 后置条件校验函数 (result, args, kwargs) -> (bool, reason)
        description   : 约束描述
        severity      : "block"（阻断）| "warn"（警告）| "log"（仅记录）
    """

    tool_name: str
    pre_check: Optional[Callable] = None
    post_check: Optional[Callable] = None
    description: str = ""
    severity: str = "block"  # block | warn | log

    def validate_pre(self, *args, **kwargs) -> tuple:
        """验证前置条件。返回 (pass, reason)。"""
        if self.pre_check is None:
            return (True, "no pre-check")
        try:
            return self.pre_check(*args, **kwargs)
        except Exception as e:
            return (False, f"pre-check error: {e}")

    def validate_post(self, result: Any, *args, **kwargs) -> tuple:
        """验证后置条件。返回 (pass, reason)。"""
        if self.post_check is None:
            return (True, "no post-check")
        try:
            return self.post_check(result, *args, **kwargs)
        except Exception as e:
            return (False, f"post-check error: {e}")


# ---------------------------------------------------------------------------
# GCD 注册表 — 内置约束库
# ---------------------------------------------------------------------------

class GCDRegistry:
    """GCD 约束注册表。

    内置约束覆盖常见的小龙虾死锁场景：
      - 空 URL / 非法 URL
      - 危险 shell 命令
      - 文件路径越界
      - 参数类型错误
    """

    def __init__(self):
        self._constraints: dict[str, list[GCDConstraint]] = {}
        self._register_builtins()

    # ---------- 注册 / 查询 ----------

    def register(self, constraint: GCDConstraint) -> None:
        if constraint.tool_name not in self._constraints:
            self._constraints[constraint.tool_name] = []
        self._constraints[constraint.tool_name].append(constraint)

    def get(self, tool_name: str) -> list[GCDConstraint]:
        return self._constraints.get(tool_name, [])

    def list_tools(self) -> list:
        return list(self._constraints.keys())

    # ---------- 内置约束 ----------

    def _register_builtins(self) -> None:
        # --- browser.navigate ---
        self.register(GCDConstraint(
            tool_name="browser.navigate",
            pre_check=_pre_url_valid,
            post_check=_post_page_loaded,
            description="URL 必须非空且格式合法; 返回内容非空",
            severity="block",
        ))
        # --- browser.click ---
        self.register(GCDConstraint(
            tool_name="browser.click",
            pre_check=_pre_selector_valid,
            description="CSS selector 非空; 禁止危险选择器",
            severity="block",
        ))
        # --- browser.type ---
        self.register(GCDConstraint(
            tool_name="browser.type",
            pre_check=_pre_text_safe,
            description="输入文本长度 ≤ 10000; 禁止注入 payload",
            severity="warn",
        ))
        # --- shell.exec ---
        self.register(GCDConstraint(
            tool_name="shell.exec",
            pre_check=_pre_shell_safe,
            description="禁止 rm -rf /; 禁止 fork bomb; 禁止反弹 shell",
            severity="block",
        ))
        # --- file.read ---
        self.register(GCDConstraint(
            tool_name="file.read",
            pre_check=_pre_path_safe,
            description="禁止读取 /etc/shadow; 禁止路径遍历 ../",
            severity="block",
        ))
        # --- file.write ---
        self.register(GCDConstraint(
            tool_name="file.write",
            pre_check=_pre_write_safe,
            description="禁止覆盖系统文件; 禁止写入 /etc /sys",
            severity="block",
        ))
        # --- api.call ---
        self.register(GCDConstraint(
            tool_name="api.call",
            pre_check=_pre_api_params,
            post_check=_post_http_status,
            description="API 参数必须为合法 JSON; 响应状态码 2xx",
            severity="warn",
        ))


# ---------------------------------------------------------------------------
# Pre-Check 函数
# ---------------------------------------------------------------------------

def _pre_url_valid(*args, **kwargs) -> tuple:
    url = kwargs.get("url", args[0] if args else "")
    if not url or not isinstance(url, str):
        return (False, "GCD: URL 为空或非字符串")
    if not re.match(r"^https?://", url):
        return (False, f"GCD: URL 格式不合法: {str(url)[:80]}")
    return (True, "ok")


def _pre_selector_valid(*args, **kwargs) -> tuple:
    selector = kwargs.get("selector", args[0] if args else "")
    if not selector or not isinstance(selector, str):
        return (False, "GCD: CSS selector 为空")
    # 禁止危险选择器注入
    dangerous = ["document.cookie", "eval(", "Function("]
    for d in dangerous:
        if d in selector:
            return (False, f"GCD: 危险选择器注入: {d}")
    return (True, "ok")


def _pre_text_safe(*args, **kwargs) -> tuple:
    text = kwargs.get("text", args[0] if args else "")
    if not isinstance(text, str):
        return (False, "GCD: 文本参数非字符串")
    if len(text) > 10000:
        return (False, f"GCD: 文本长度超限 ({len(text)} > 10000)")
    # 检测注入 payload
    injection_patterns = ["<script", "javascript:", "onerror=", "onload="]
    for p in injection_patterns:
        if p.lower() in text.lower():
            return (False, f"GCD: 检测到注入 payload: {p}")
    return (True, "ok")


def _pre_shell_safe(*args, **kwargs) -> tuple:
    cmd = kwargs.get("command", args[0] if args else "")
    if not isinstance(cmd, str):
        return (False, "GCD: 命令参数非字符串")
    dangerous = [
        "rm -rf /", "rm -rf --no-preserve-root",
        ":(){ :|:& };:",  # fork bomb
        "> /dev/sda", "mkfs.",
        "nc -e /bin/sh", "bash -i >&",  # reverse shell
        "chmod 777 /", "chown -R",
    ]
    lower = cmd.lower()
    for d in dangerous:
        if d.lower() in lower:
            return (False, f"GCD: 危险命令: {d}")
    return (True, "ok")


def _pre_path_safe(*args, **kwargs) -> tuple:
    path = kwargs.get("path", args[0] if args else "")
    if not isinstance(path, str):
        return (False, "GCD: 文件路径非字符串")
    # 路径遍历检测
    if ".." in path:
        return (False, f"GCD: 路径遍历拒绝: {path}")
    # 敏感文件检测
    sensitive = ["/etc/shadow", "/etc/passwd", "C:\\Windows\\System32\\config\\SAM"]
    for s in sensitive:
        if s.lower() in path.lower():
            return (False, f"GCD: 敏感文件拒绝: {s}")
    return (True, "ok")


def _pre_write_safe(*args, **kwargs) -> tuple:
    path = kwargs.get("path", args[0] if args else "")
    if not isinstance(path, str):
        return (False, "GCD: 写入路径非字符串")
    forbidden_prefixes = ["/etc/", "/sys/", "/proc/", "/boot/",
                          "C:\\Windows\\", "C:\\Windows\\System32\\"]
    for prefix in forbidden_prefixes:
        if path.replace("\\", "/").lower().startswith(prefix.lower()):
            return (False, f"GCD: 禁止写入系统路径: {prefix}")
    return (True, "ok")


def _pre_api_params(*args, **kwargs) -> tuple:
    """API 调用参数必须为合法 JSON-serializable。"""
    params = kwargs.get("params", args[0] if args else {})
    if params is not None:
        try:
            import json
            json.dumps(params)
        except (TypeError, ValueError) as e:
            return (False, f"GCD: API 参数不可序列化: {e}")
    return (True, "ok")


# ---------------------------------------------------------------------------
# Post-Check 函数
# ---------------------------------------------------------------------------

def _post_page_loaded(result: Any, *args, **kwargs) -> tuple:
    """浏览器页面加载后：内容非空。"""
    if isinstance(result, dict):
        content = result.get("content", result.get("html", ""))
        if not content:
            return (False, "GCD: 页面内容为空（可能被拦截或超时）")
    return (True, "ok")


def _post_http_status(result: Any, *args, **kwargs) -> tuple:
    """HTTP 响应：状态码 2xx。"""
    if isinstance(result, dict):
        status = result.get("status", result.get("status_code", 200))
        if not (200 <= status < 300):
            return (False, f"GCD: HTTP 状态码异常: {status}")
    return (True, "ok")


# ---------------------------------------------------------------------------
# GCD 引擎 — 主入口
# ---------------------------------------------------------------------------

@dataclass
class GCDResult:
    """GCD 校验结果。"""
    passed: bool
    tool_name: str
    phase: str          # "pre" | "post"
    violations: list = field(default_factory=list)
    warnings: list = field(default_factory=list)
    logs: list = field(default_factory=list)
    blocked_count: int = 0

    @property
    def is_clean(self) -> bool:
        return self.blocked_count == 0


class GCDEngine:
    """GCD 归约引擎。

    对每次工具调用实施 Pre + Post 约束校验。

    用法:
        engine = GCDEngine()
        result = engine.check("browser.navigate", "pre", url="https://example.com")
        if not result.passed:
            raise GCDViolationError(result)
    """

    def __init__(self, registry: Optional[GCDRegistry] = None):
        self.registry = registry or GCDRegistry()
        self.stats = {"pre_checks": 0, "post_checks": 0,
                      "blocks": 0, "warns": 0}

    def check(
        self,
        tool_name: str,
        phase: str = "pre",
        *args,
        **kwargs,
    ) -> GCDResult:
        """对一次工具调用执行 GCD 校验。

        参数:
            tool_name: 工具名
            phase:     "pre"（调用前）| "post"（调用后）
            *args, **kwargs: 传递给约束校验函数的参数

        返回:
            GCDResult（passed=False 时调用应被阻断）
        """
        result = GCDResult(
            passed=True, tool_name=tool_name, phase=phase
        )
        constraints = self.registry.get(tool_name)

        for c in constraints:
            if phase == "pre":
                ok, reason = c.validate_pre(*args, **kwargs)
            else:
                ok, reason = c.validate_post(*args, **kwargs)

            if not ok:
                if c.severity == "block":
                    result.blocked_count += 1
                    result.violations.append(f"[BLOCK] {c.tool_name}: {reason}")
                    self.stats["blocks"] += 1
                elif c.severity == "warn":
                    result.warnings.append(f"[WARN] {c.tool_name}: {reason}")
                    self.stats["warns"] += 1
                else:
                    result.logs.append(f"[LOG] {c.tool_name}: {reason}")

        result.passed = result.blocked_count == 0

        if phase == "pre":
            self.stats["pre_checks"] += 1
        else:
            self.stats["post_checks"] += 1

        return result

    def wrap_call(
        self, tool_name: str, func: Callable, *args, **kwargs
    ) -> Any:
        """包裹一次工具调用，自动注入 Pre/Post 校验。

        返回:
            (result, GCDResult) 元组
        """
        from dataclasses import dataclass as dc

        pre_result = self.check(tool_name, "pre", *args, **kwargs)
        if not pre_result.passed:
            raise GCDViolationError(pre_result)

        func_result = func(*args, **kwargs)

        post_result = self.check(tool_name, "post", func_result, *args, **kwargs)
        if not post_result.passed:
            raise GCDViolationError(post_result)

        return func_result


class GCDViolationError(Exception):
    """GCD 约束违反异常。"""

    def __init__(self, result: GCDResult):
        self.result = result
        msg = (
            f"GCD Violation [{result.tool_name}] {result.phase}: "
            f"{len(result.violations)} blocks, {len(result.warnings)} warns"
        )
        super().__init__(msg)
