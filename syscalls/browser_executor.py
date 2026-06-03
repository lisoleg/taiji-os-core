"""syscalls/browser_executor.py — PlaywrightExecutor: 浏览器动作执行器

将 Planner 生成的"浏览器操作步骤"转化为真实的 Playwright 指令。

支持动作类型:
  - navigate   : 跳转到 URL
  - click      : 点击元素（CSS selector / XPath）
  - type       : 在输入框填写文本
  - read_dom   : 读取页面 DOM 文本摘要
  - screenshot : 截图（Base64 PNG）
  - scroll     : 滚动页面
  - wait       : 等待元素出现 / 固定时间
  - eval_js    : 执行 JavaScript 片段

Playwright 未安装时自动回退到 MockBrowserExecutor（保证测试可通过）。
"""
from __future__ import annotations

import time
import json
from typing import Any, Dict, List, Optional


# --------------------------------------------------------------------------
# 尝试导入 Playwright；不可用时使用 mock
# --------------------------------------------------------------------------

try:
    from playwright.sync_api import sync_playwright, Page, Browser  # type: ignore
    _PLAYWRIGHT_AVAILABLE = True
except ImportError:
    _PLAYWRIGHT_AVAILABLE = False


class PlaywrightExecutor:
    """
    浏览器动作执行器（PlaywrightExecutor）。

    懒加载浏览器：第一次执行浏览器动作时才启动，避免不必要的开销。
    自动降级：Playwright 未安装时回退到 MockBrowserExecutor。
    """

    def __init__(self, headless: bool = True, web_world_model=None):
        self._headless = headless
        self._pw = None          # sync_playwright 上下文
        self._browser: Optional["Browser"] = None
        self._page: Optional["Page"] = None
        self._web_wm = web_world_model  # WebWorldModel 实例（可选）

        # 降级标志
        self._mock_mode = not _PLAYWRIGHT_AVAILABLE

    # ------------------------------------------------------------------
    # 生命周期
    # ------------------------------------------------------------------

    def start(self) -> None:
        """启动浏览器（若尚未启动）。"""
        if self._mock_mode or self._browser:
            return
        self._pw = sync_playwright().start()
        self._browser = self._pw.chromium.launch(headless=self._headless)
        self._page = self._browser.new_page()

    def close(self) -> None:
        """关闭浏览器。"""
        if self._browser:
            self._browser.close()
            self._browser = None
        if self._pw:
            self._pw.stop()
            self._pw = None
        self._page = None

    # ------------------------------------------------------------------
    # 主入口：执行步骤列表
    # ------------------------------------------------------------------

    def execute(self, steps: List[Dict], llm_router=None) -> List[Dict]:
        """
        执行 Planner 生成的步骤列表。

        每个 step 格式:
          {"action": "navigate", "params": {"url": "https://example.com"}}

        返回结果列表，每项包含 action / status / output / elapsed_ms。
        """
        if not self._mock_mode:
            self.start()

        results = []
        for step in steps:
            action = step.get("action", "noop")
            params = step.get("params", {})
            t0 = time.time()
            try:
                output = self._dispatch(action, params, llm_router)
                status = "ok"
            except Exception as e:
                output = f"[ERROR] {e}"
                status = "error"
            elapsed = time.time() - t0
            results.append({
                "action": action,
                "status": status,
                "output": output,
                "elapsed_ms": round(elapsed * 1000, 2),
            })
        return results

    # ------------------------------------------------------------------
    # 动作分发
    # ------------------------------------------------------------------

    def _dispatch(self, action: str, params: Dict, llm_router=None) -> Any:
        if self._mock_mode:
            return self._mock_dispatch(action, params, llm_router)
        return self._real_dispatch(action, params, llm_router)

    # --- 真实 Playwright 实现 ------------------------------------------

    def _real_dispatch(self, action: str, params: Dict, llm_router=None) -> Any:
        page = self._page
        assert page is not None, "Browser not started"

        if action == "navigate":
            url = params["url"]
            page.goto(url, wait_until="domcontentloaded", timeout=30_000)
            title = page.title()
            dom = self._dom_summary(page)
            self._update_wm(page.url, title, dom)
            return {"url": page.url, "title": title, "dom_summary": dom[:256]}

        elif action == "click":
            selector = params.get("selector", "")
            page.click(selector, timeout=10_000)
            return f"Clicked: {selector}"

        elif action == "type":
            selector = params.get("selector", "")
            text = params.get("text", "")
            page.fill(selector, text, timeout=10_000)
            return f"Typed '{text}' into {selector}"

        elif action == "read_dom":
            dom = self._dom_summary(page)
            self._update_wm(page.url, page.title(), dom)
            return dom[:1024]

        elif action == "screenshot":
            screenshot_bytes = page.screenshot(type="png")
            import base64
            b64 = base64.b64encode(screenshot_bytes).decode()
            return {"format": "png_base64", "data": b64[:256] + "...[truncated]"}

        elif action == "scroll":
            direction = params.get("direction", "down")
            px = params.get("px", 300)
            dy = px if direction == "down" else -px
            page.mouse.wheel(0, dy)
            return f"Scrolled {direction} {px}px"

        elif action == "wait":
            selector = params.get("selector")
            ms = params.get("ms", 1000)
            if selector:
                page.wait_for_selector(selector, timeout=ms)
                return f"Element appeared: {selector}"
            else:
                page.wait_for_timeout(ms)
                return f"Waited {ms}ms"

        elif action == "eval_js":
            script = params.get("script", "")
            result = page.evaluate(script)
            return str(result)[:512]

        elif action == "respond" and llm_router:
            return llm_router.complete(params.get("input", ""))

        return f"[Noop] action={action}"

    # --- Mock 实现（Playwright 未安装时）-------------------------------

    def _mock_dispatch(self, action: str, params: Dict, llm_router=None) -> Any:
        if action == "navigate":
            url = params.get("url", "")
            return {"url": url, "title": f"[Mock] {url}", "dom_summary": "Mock DOM content"}
        elif action == "click":
            return f"[Mock] Clicked: {params.get('selector', '')}"
        elif action == "type":
            return f"[Mock] Typed '{params.get('text', '')}' into {params.get('selector', '')}"
        elif action == "read_dom":
            return "[Mock] <html><body><p>Mock page content</p></body></html>"
        elif action == "screenshot":
            return {"format": "mock", "data": "mock_screenshot_data"}
        elif action == "scroll":
            return f"[Mock] Scrolled {params.get('direction', 'down')}"
        elif action == "wait":
            return f"[Mock] Waited {params.get('ms', 1000)}ms"
        elif action == "eval_js":
            return f"[Mock] JS result: {params.get('script', '')[:50]}"
        elif action == "respond" and llm_router:
            return llm_router.complete(params.get("input", ""))
        elif action == "analyze_requirements":
            return f"[Mock Analyzed] {params.get('input', '')[:80]}"
        elif action == "generate_solution":
            return "[Mock] Solution generated"
        elif action == "verify_solution":
            return "[Mock] Solution verified"
        elif action == "retrieve":
            return f"[Mock] Retrieved: {params.get('query', '')}"
        return f"[Mock Noop] action={action}"

    # ------------------------------------------------------------------
    # 辅助
    # ------------------------------------------------------------------

    @staticmethod
    def _dom_summary(page: "Page", max_chars: int = 2000) -> str:
        """提取页面主要文本内容（去掉脚本/样式）。"""
        try:
            text = page.evaluate("""() => {
                const clone = document.cloneNode(true);
                const scripts = clone.querySelectorAll('script, style, noscript');
                scripts.forEach(s => s.remove());
                return document.body ? document.body.innerText : '';
            }""")
            return str(text)[:max_chars]
        except Exception:
            return page.content()[:max_chars]

    def _update_wm(self, url: str, title: str, dom: str) -> None:
        """将页面快照推送到 WebWorldModel（若已注入）。"""
        if self._web_wm is not None:
            self._web_wm.observe_page(url=url, title=title, dom_summary=dom)

    @property
    def mock_mode(self) -> bool:
        return self._mock_mode
