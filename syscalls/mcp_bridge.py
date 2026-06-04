"""
syscalls/mcp_bridge.py — MCP Bridge: Walrus Memory MCP 原生桥接

将太极OS的能力暴露为标准 MCP (Model Context Protocol) 工具，
通过 stdio JSON-RPC 与外部 AI agents (Claude Desktop, ChatGPT, etc.) 通信。

暴露的 MCP 工具：
  - taiji.run(query)       : 执行一轮推演
  - taiji.status(sid)      : 查询会话状态
  - taiji.resume(kid)      : 从 Continuation 恢复
  - taiji.memory_search(q) : 搜索共享记忆
  - taiji.verify(mid)      : 验证记忆完整性
  - taiji.list_sessions()  : 列出已注册会话

用法:
    # 作为 MCP Server 运行
    python -m syscalls.mcp_bridge

    # 或在代码中嵌入
    from syscalls.mcp_bridge import MCPBridge
    bridge = MCPBridge(session, memory_hub)
    bridge.serve()
"""

from __future__ import annotations

import json
import sys
from typing import Any, Optional


class MCPBridge:
    """
    MCP 桥接：将 TaijiSession + MemoryHub 暴露为 MCP 工具。

    通过 stdio JSON-RPC 协议与外部 MCP 客户端通信。
    支持 6 个工具 + initialize/notifications/list_tools/call_tool 标准方法。
    """

    def __init__(self, session: Any = None, memory_hub: Any = None):
        self.session = session
        self.memory_hub = memory_hub
        self._server_name = "taiji-os"
        self._server_version = "3.0.0"

    # ------------------------------------------------------------------
    # JSON-RPC 主循环
    # ------------------------------------------------------------------

    def serve(self):
        """启动 stdio MCP 服务（阻塞直到 stdin 关闭）。"""
        for line in sys.stdin:
            line = line.strip()
            if not line:
                continue
            try:
                request = json.loads(line)
            except json.JSONDecodeError:
                self._respond_error(None, -32700, "Parse error")
                continue

            self._handle(request)

    def _handle(self, request: dict):
        """处理单条 JSON-RPC 请求。"""
        rpc_id = request.get("id")
        method = request.get("method", "")
        params = request.get("params", {})

        handler = self._DISPATCH.get(method)
        if handler is None:
            self._respond_error(rpc_id, -32601, f"Method not found: {method}")
            return

        try:
            result = handler(self, **params)
            self._respond(rpc_id, result)
        except Exception as e:
            self._respond_error(rpc_id, -32603, str(e))

    # ------------------------------------------------------------------
    # MCP 标准方法
    # ------------------------------------------------------------------

    def _mcp_initialize(self, **params) -> dict:
        return {
            "protocolVersion": "2024-11-05",
            "serverInfo": {
                "name": self._server_name,
                "version": self._server_version,
            },
            "capabilities": {
                "tools": {},
            },
        }

    def _mcp_initialized(self, **params) -> dict:
        return {}

    def _mcp_list_tools(self, **params) -> dict:
        return {
            "tools": [
                {
                    "name": "taiji.run",
                    "description": "执行一轮太极OS推演（文本或浏览器云脑模式）。参数: sid(会话ID), query(输入)",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "sid": {"type": "string", "description": "会话 ID"},
                            "query": {"type": "string", "description": "输入指令或问题"},
                        },
                        "required": ["sid", "query"],
                    },
                },
                {
                    "name": "taiji.status",
                    "description": "查询指定会话状态。参数: sid(会话ID)",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "sid": {"type": "string", "description": "会话 ID"},
                        },
                        "required": ["sid"],
                    },
                },
                {
                    "name": "taiji.resume",
                    "description": "从 Continuation 快照恢复会话。参数: kid(Continuation ID)",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "kid": {"type": "string", "description": "Continuation ID (kid)"},
                        },
                        "required": ["kid"],
                    },
                },
                {
                    "name": "taiji.memory_search",
                    "description": "在共享记忆中搜索。参数: q(搜索关键词)",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "q": {"type": "string", "description": "搜索关键词"},
                        },
                        "required": ["q"],
                    },
                },
                {
                    "name": "taiji.verify",
                    "description": "验证记忆完整性。参数: mid(记忆ID), 无mid时验证全部",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "mid": {"type": "string", "description": "记忆 ID（可选，不提供时验证全部）"},
                        },
                    },
                },
                {
                    "name": "taiji.list_sessions",
                    "description": "列出所有已注册会话及记忆统计",
                    "inputSchema": {
                        "type": "object",
                        "properties": {},
                    },
                },
            ]
        }

    def _mcp_call_tool(self, name: str, arguments: dict = None, **params) -> dict:
        if arguments is None:
            arguments = {}

        tool_handlers = {
            "taiji.run": self._tool_run,
            "taiji.status": self._tool_status,
            "taiji.resume": self._tool_resume,
            "taiji.memory_search": self._tool_memory_search,
            "taiji.verify": self._tool_verify,
            "taiji.list_sessions": self._tool_list_sessions,
        }

        handler = tool_handlers.get(name)
        if handler is None:
            raise ValueError(f"Unknown tool: {name}")

        result = handler(**arguments)
        return {
            "content": [
                {
                    "type": "text",
                    "text": json.dumps(result, ensure_ascii=False, indent=2),
                }
            ]
        }

    # ------------------------------------------------------------------
    # 工具实现
    # ------------------------------------------------------------------

    def _tool_run(self, sid: str, query: str) -> dict:
        if self.session is None:
            # 延迟初始化
            from core.session import TaijiSession
            from hal.llm_router import LLMRouter
            llm = LLMRouter()
            self.session = TaijiSession(sid, llm)
        output = self.session.run(query)
        return {"sid": sid, "output": output}

    def _tool_status(self, sid: str) -> dict:
        if self.session is None:
            return {"error": "no active session"}
        return self.session.status()

    def _tool_resume(self, kid: str) -> dict:
        if self.session is None:
            return {"error": "no active session"}
        self.session.resume(kid)
        return {"kid": kid, "status": "resumed"}

    def _tool_memory_search(self, q: str) -> dict:
        if self.memory_hub is None:
            from core.memory_hub import MemoryHub
            self.memory_hub = MemoryHub()
        results = self.memory_hub.search(q)
        return {"query": q, "count": len(results), "results": results}

    def _tool_verify(self, mid: Optional[str] = None) -> dict:
        if self.memory_hub is None:
            from core.memory_hub import MemoryHub
            self.memory_hub = MemoryHub()
        if mid:
            ok, msg = self.memory_hub.verify(mid)
            return {"mid": mid, "valid": ok, "message": msg}
        else:
            return self.memory_hub.verify_all()

    def _tool_list_sessions(self) -> dict:
        if self.memory_hub is None:
            from core.memory_hub import MemoryHub
            self.memory_hub = MemoryHub()
        sessions = self.memory_hub.list_sessions()
        return {"sessions": sessions}

    # ------------------------------------------------------------------
    # 响应输出
    # ------------------------------------------------------------------

    def _respond(self, rpc_id, result: dict):
        resp = {"jsonrpc": "2.0", "id": rpc_id, "result": result}
        sys.stdout.write(json.dumps(resp, ensure_ascii=False) + "\n")
        sys.stdout.flush()

    def _respond_error(self, rpc_id, code: int, message: str):
        resp = {
            "jsonrpc": "2.0",
            "id": rpc_id,
            "error": {"code": code, "message": message},
        }
        sys.stdout.write(json.dumps(resp, ensure_ascii=False) + "\n")
        sys.stdout.flush()

    # ------------------------------------------------------------------
    # 方法分发表
    # ------------------------------------------------------------------

    _DISPATCH = {
        "initialize":        _mcp_initialize,
        "notifications/initialized": _mcp_initialized,
        "tools/list":        _mcp_list_tools,
        "tools/call":        _mcp_call_tool,
    }


# ------------------------------------------------------------------
# 独立运行入口
# ------------------------------------------------------------------
if __name__ == "__main__":
    import os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

    from core.memory_hub import MemoryHub
    hub = MemoryHub()
    bridge = MCPBridge(memory_hub=hub)
    bridge.serve()
