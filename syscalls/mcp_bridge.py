"""syscalls/mcp_bridge.py — MCP Bridge: Walrus Memory 风格的 MCP 原生支持

Walrus Memory 理念映射：
  - MCP Native → 太极OS 作为 MCP Server 暴露工具
  - 支持 Claude、ChatGPT、Gemini 等任何 MCP 客户端接入

MCP Bridge 提供以下 tools:
  - taiji.run         : 执行一轮太极推演
  - taiji.status      : 查询会话状态
  - taiji.resume      : 从 Continuation 恢复
  - taiji.memory_search: 跨会话搜索记忆（Walrus 核心能力）
  - taiji.verify      : 验证快照完整性
  - taiji.list_sessions: 列出所有已注册会话

协议: MCP JSON-RPC 2.0 (stdio transport)
"""
from __future__ import annotations

import json
import sys
from typing import Optional


class MCPBridge:
    """
    MCP (Model Context Protocol) stdio bridge.

    启动方式:
        python -m syscalls.mcp_bridge --hub-dir shared_memory

    MCP 客户端连接后自动进入 JSON-RPC 循环。
    """

    def __init__(self, hub_dir: str = "shared_memory", config_path: str = "config.yaml"):
        from core.memory_hub import MemoryHub

        self.hub = MemoryHub(hub_dir)
        self.config_path = config_path
        self._sessions: dict[str, object] = {}  # sid → TaijiSession
        self._tools = self._build_tools()

    # ------------------------------------------------------------------
    # Tool 定义
    # ------------------------------------------------------------------

    def _build_tools(self) -> list[dict]:
        return [
            {
                "name": "taiji.run",
                "description": "在太极OS中运行一轮AGI推演。支持 text 和 web 两种模式",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "sid": {"type": "string", "description": "会话ID（留空自动创建）"},
                        "input": {"type": "string", "description": "用户输入 / 意图"},
                        "mode": {
                            "type": "string",
                            "enum": ["text", "web"],
                            "default": "text",
                        },
                    },
                    "required": ["input"],
                },
            },
            {
                "name": "taiji.status",
                "description": "查询太极OS会话状态（ψ版本、意图、历史长度）",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "sid": {"type": "string", "description": "会话ID"},
                    },
                    "required": ["sid"],
                },
            },
            {
                "name": "taiji.resume",
                "description": "从 Continuation 快照恢复太极OS会话",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "sid": {"type": "string", "description": "会话ID"},
                        "kid": {"type": "string", "description": "Continuation ID"},
                    },
                    "required": ["sid", "kid"],
                },
            },
            {
                "name": "taiji.memory_search",
                "description": "跨会话搜索太极OS记忆（Walrus Memory 风格）",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "keyword": {"type": "string", "description": "搜索关键词"},
                        "sid": {"type": "string", "description": "限定会话（可选）"},
                        "limit": {"type": "integer", "default": 10},
                    },
                    "required": ["keyword"],
                },
            },
            {
                "name": "taiji.verify",
                "description": "验证所有 Continuation 快照的完整性",
                "inputSchema": {
                    "type": "object",
                    "properties": {},
                },
            },
            {
                "name": "taiji.list_sessions",
                "description": "列出所有已注册的太极OS会话",
                "inputSchema": {
                    "type": "object",
                    "properties": {},
                },
            },
        ]

    # ------------------------------------------------------------------
    # Tool 实现
    # ------------------------------------------------------------------

    def _handle_taiji_run(self, args: dict) -> str:
        sid = args.get("sid") or f"mcp-{id(args)}"
        user_input = args["input"]
        mode = args.get("mode", "text")

        # 获取或创建 session
        if sid not in self._sessions:
            from core.session import TaijiSession
            from hal.llm_router import LLMRouter

            router = LLMRouter(self.config_path)
            sess = TaijiSession(sid, router, mode=mode)
            self._sessions[sid] = sess
            self.hub.register(sid, {"mode": mode})

        sess = self._sessions[sid]
        result = sess.run(user_input)
        return result

    def _handle_taiji_status(self, args: dict) -> dict:
        sid = args["sid"]
        if sid not in self._sessions:
            return {"error": f"session {sid} not found"}
        return self._sessions[sid].status()

    def _handle_taiji_resume(self, args: dict) -> str:
        sid = args["sid"]
        kid = args["kid"]
        if sid not in self._sessions:
            from core.session import TaijiSession
            from hal.llm_router import LLMRouter

            router = LLMRouter(self.config_path)
            sess = TaijiSession(sid, router)
            self._sessions[sid] = sess
            self.hub.register(sid)
        self._sessions[sid].resume(kid)
        return f"Session {sid} resumed from Continuation {kid}"

    def _handle_memory_search(self, args: dict) -> list[dict]:
        return self.hub.search(
            keyword=args.get("keyword", ""),
            sid=args.get("sid"),
            limit=args.get("limit", 10),
        )

    def _handle_verify(self, _args: dict) -> dict:
        return self.hub.verify_all()

    def _handle_list_sessions(self, _args: dict) -> list[dict]:
        return self.hub.list_sessions()

    # ------------------------------------------------------------------
    # JSON-RPC 循环
    # ------------------------------------------------------------------

    def _dispatch(self, method: str, args: dict) -> dict:
        handlers = {
            "taiji.run": self._handle_taiji_run,
            "taiji.status": self._handle_taiji_status,
            "taiji.resume": self._handle_taiji_resume,
            "taiji.memory_search": self._handle_memory_search,
            "taiji.verify": self._handle_verify,
            "taiji.list_sessions": self._handle_list_sessions,
        }
        handler = handlers.get(method)
        if not handler:
            return {"error": f"unknown method: {method}"}
        try:
            result = handler(args)
            return {"result": result}
        except Exception as e:
            return {"error": str(e)}

    def serve(self):
        """
        启动 MCP stdio JSON-RPC 服务循环。

        协议: 每行一个 JSON-RPC 请求 → 返回 JSON-RPC 响应
        """
        for line in sys.stdin:
            line = line.strip()
            if not line:
                continue
            try:
                request = json.loads(line)
            except json.JSONDecodeError:
                self._respond({"error": "invalid JSON"})
                continue

            req_id = request.get("id")
            method = request.get("method", "")

            if method == "tools/list":
                self._respond({"id": req_id, "result": {"tools": self._tools}})
            elif method == "tools/call":
                params = request.get("params", {})
                tool_name = params.get("name", "")
                tool_args = params.get("arguments", {})
                result = self._dispatch(tool_name, tool_args)
                self._respond({"id": req_id, "result": result})
            elif method == "initialize":
                self._respond({
                    "id": req_id,
                    "result": {
                        "protocolVersion": "2024-11-05",
                        "serverInfo": {"name": "taiji-os", "version": "3.1"},
                        "capabilities": {"tools": {}},
                    },
                })
            else:
                self._respond({"id": req_id, "error": f"unknown method: {method}"})

    @staticmethod
    def _respond(data: dict):
        sys.stdout.write(json.dumps(data, ensure_ascii=False) + "\n")
        sys.stdout.flush()


# ------------------------------------------------------------------
# CLI 入口
# ------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="太极OS MCP Bridge (Walrus Memory)")
    parser.add_argument("--hub-dir", default="shared_memory", help="MemoryHub 目录")
    parser.add_argument("--config", default="config.yaml", help="配置文件路径")
    parser.add_argument("--list-tools", action="store_true", help="仅打印 tool 列表后退出")
    args = parser.parse_args()

    bridge = MCPBridge(hub_dir=args.hub_dir, config_path=args.config)

    if args.list_tools:
        print(json.dumps(bridge._tools, ensure_ascii=False, indent=2))
    else:
        bridge.serve()
