"""hal/nic_emu.py — USCS 网络传输抽象层 (v1.0)

提供 NodeTransport：跨节点进程快照传输的抽象接口。

支持传输协议：
  - "http"    : 基于 HTTP POST/GET 的 RESTful 传输
  - "stdio"   : 基于标准输入/输出的本地进程传输（调试用）
  - "local"   : 单节点自传输（无需网络）

复用模式：
  - _load_config() 配置加载 (hal/llm_router.py)
"""

from __future__ import annotations

import json
import os
import time
from typing import Any, Optional

import yaml


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────


def _load_config(path: str = "config.yaml") -> dict:
    """复用 hal/llm_router.py 的配置加载模式。"""
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    return {}


# ──────────────────────────────────────────────────────────────────────────────
# NodeTransport
# ──────────────────────────────────────────────────────────────────────────────


class NodeTransport:
    """
    跨节点网络传输抽象层。

    支持三种传输模式：
      - "local" : 单节点自传输（ProcessSnapshot 直接传递，不经过网络）
      - "http"  : HTTP 传输（生产用）
      - "stdio" : 标准 I/O 传输（调试用）

    Attributes:
        mode   : 传输模式
        nodes  : 已知节点列表 [{id, host, port}, ...]
    """

    def __init__(self, mode: str = "local", config_path: str = "config.yaml"):
        self.mode = mode
        cfg = _load_config(config_path)
        migration_cfg = cfg.get("migration", {})
        self.nodes: list[dict] = migration_cfg.get("nodes", [])
        self._node_map: dict[str, dict] = {n["id"]: n for n in self.nodes}

    def send(self, snapshot: Any, target_node: str) -> bool:
        """
        发送 ProcessSnapshot 到目标节点。

        Args:
            snapshot     : ProcessSnapshot 实例
            target_node  : 目标节点 ID

        Returns:
            是否发送成功
        """
        if self.mode == "local":
            return self._send_local(snapshot, target_node)

        elif self.mode == "http":
            return self._send_http(snapshot, target_node)

        elif self.mode == "stdio":
            return self._send_stdio(snapshot, target_node)

        else:
            raise ValueError(f"Unknown transport mode: {self.mode}")

    def recv(self, source_node: str) -> Optional[dict]:
        """
        接收来自源节点的 ProcessSnapshot。

        Args:
            source_node : 源节点 ID

        Returns:
            ProcessSnapshot 的 dict 表示，或 None（接收失败）
        """
        if self.mode == "local":
            return self._recv_local(source_node)

        elif self.mode == "http":
            return self._recv_http(source_node)

        elif self.mode == "stdio":
            return self._recv_stdio(source_node)

        else:
            raise ValueError(f"Unknown transport mode: {self.mode}")

    # ── 传输实现 ──

    def _send_local(self, snapshot: Any, target_node: str) -> bool:
        """本地传输：直接缓存到 _recv_buffer。"""
        if not hasattr(self, "_recv_buffer"):
            self._recv_buffer: dict[str, Any] = {}
        snapshot_json = snapshot.to_json() if hasattr(snapshot, "to_json") else json.dumps(snapshot)
        self._recv_buffer[target_node] = json.loads(snapshot_json)
        return True

    def _send_http(self, snapshot: Any, target_node: str) -> bool:
        """HTTP 传输：POST snapshot 到目标节点的 /migration/import 端点。"""
        node_info = self._node_map.get(target_node)
        if node_info is None:
            return False

        url = f"http://{node_info['host']}:{node_info['port']}/migration/import"
        payload = (
            snapshot.to_json() if hasattr(snapshot, "to_json") else json.dumps(snapshot)
        )

        try:
            import urllib.request
            req = urllib.request.Request(
                url,
                data=payload.encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            urllib.request.urlopen(req, timeout=30)
            return True
        except Exception:
            return False

    def _send_stdio(self, snapshot: Any, target_node: str) -> bool:
        """Stdio 传输：将快照写入 stdout（调试用）。"""
        payload = (
            snapshot.to_json() if hasattr(snapshot, "to_json") else json.dumps(snapshot)
        )
        print(json.dumps({"type": "migration_snapshot", "target": target_node, "payload": json.loads(payload)}))
        return True

    def _recv_local(self, source_node: str) -> Optional[dict]:
        """本地接收：从 _recv_buffer 读取。"""
        if hasattr(self, "_recv_buffer"):
            return self._recv_buffer.pop(source_node, None)
        return None

    def _recv_http(self, source_node: str) -> Optional[dict]:
        """HTTP 接收：GET source_node 的 /migration/export/{pid} 端点。"""
        # HTTP recv 由目标节点 API 端点处理，
        # 这里是客户端发起 GET 的场景。
        return None  # 被动接收由 API server 处理

    def _recv_stdio(self, source_node: str) -> Optional[dict]:
        """Stdio 接收：从 stdin 读取（调试用）。"""
        result = input()
        try:
            return json.loads(result)
        except json.JSONDecodeError:
            return None

    # ── 节点管理 ──

    def add_node(self, node_id: str, host: str, port: int) -> None:
        """动态添加节点。"""
        node_info = {"id": node_id, "host": host, "port": port}
        self.nodes.append(node_info)
        self._node_map[node_id] = node_info

    def remove_node(self, node_id: str) -> None:
        """动态移除节点。"""
        self.nodes = [n for n in self.nodes if n["id"] != node_id]
        self._node_map.pop(node_id, None)

    def list_nodes(self) -> list[dict]:
        """列出所有已知节点。"""
        return self.nodes
