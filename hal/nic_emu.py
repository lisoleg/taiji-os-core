"""
hal/nic_emu.py - 网络传输抽象层 (NIC Emulation)

提供跨节点网络传输抽象，支持 HTTP 模式进程快照传输。
用于 USCS 内核子系统的跨节点迁移功能。

配置从 config.yaml 的 migration.nodes 段读取，与 llm_router.py 的 _load_config() 模式一致。
"""

import json
import os
import socket
import urllib.error
import urllib.request
from collections import deque
from typing import Optional

import yaml


def _load_config(path: str = "config.yaml") -> dict:
    """
    从 YAML 文件加载配置（与 llm_router.py 一致的加载模式）。

    Args:
        path: 配置文件路径，默认为 config.yaml

    Returns:
        配置字典；文件不存在时返回空字典
    """
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    return {}


class TransportError(Exception):
    """
    传输异常，包含 target_node 和 reason 字段。

    Attributes:
        target_node: 目标节点 ID
        reason: 错误原因描述
    """

    def __init__(self, target_node: str, reason: str):
        self.target_node = target_node
        self.reason = reason
        super().__init__(f"TransportError(node={target_node}, reason={reason})")


class NodeTransport:
    """
    跨节点网络传输抽象类/实现。

    支持 HTTP 模式，通过标准库 urllib.request + json 发送和接收
    ProcessSnapshot JSON。配置从 config.yaml 的 migration.nodes 段读取。

    Attributes:
        node_id: 本节点 ID
        nodes_config: 节点配置字典，格式为 {node_id: {"host": ..., "port": ...}}
        timeout: 传输超时（秒），默认 30
        _recv_queue: 接收队列（用于存储接收到的快照 JSON）
    """

    def __init__(
        self,
        node_id: str,
        nodes_config: Optional[list[dict]] = None,
        config_path: str = "config.yaml",
    ):
        """
        初始化 NodeTransport。

        从 nodes_config 或 config.yaml 的 migration.nodes 段加载对端节点信息。

        Args:
            node_id: 本节点 ID（对应 migration.node_id）
            nodes_config: 节点配置列表，格式为
                [{"id": ..., "host": ..., "port": ...}, ...]
                如果为 None，则从 config_path 的 migration.nodes 加载
            config_path: 配置文件路径（仅当 nodes_config 为 None 时使用）
        """
        self.node_id = node_id
        self.timeout = 30
        self._recv_queue: deque[str] = deque()

        if nodes_config is not None:
            self.nodes_config = {n["id"]: n for n in nodes_config}
        else:
            cfg = _load_config(config_path)
            migration_cfg = cfg.get("migration", {})
            nodes_list = migration_cfg.get("nodes", [])
            self.nodes_config = {n["id"]: n for n in nodes_list}

    def _resolve_node(self, target_node: str) -> dict:
        """
        从配置中解析目标节点的 host 和 port。

        Args:
            target_node: 目标节点 ID

        Returns:
            包含 host 和 port 的字典，例如 {"host": "127.0.0.1", "port": 8001}

        Raises:
            TransportError: 目标节点不存在于配置中
        """
        if target_node not in self.nodes_config:
            raise TransportError(target_node, "node not found in config")
        return self.nodes_config[target_node]

    def check_node_alive(self, target_node: str) -> bool:
        """
        检查目标节点是否可达。

        向 http://{host}:{port}/health 发送 HEAD 请求，
        状态码 200 视为节点存活。

        Args:
            target_node: 目标节点 ID

        Returns:
            True 如果节点可达（HTTP 200），False 否则
        """
        node = self._resolve_node(target_node)
        url = f"http://{node['host']}:{node['port']}/health"

        req = urllib.request.Request(url, method="HEAD")
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                return resp.status == 200
        except (urllib.error.URLError, urllib.error.HTTPError, socket.timeout):
            return False

    def send(self, snapshot_json: str, target_node: str) -> dict:
        """
        发送 ProcessSnapshot JSON 到目标节点，返回目标节点的响应。

        向 http://{host}:{port}/migration/import 发起 POST 请求，
        请求体为 snapshot_json。

        Args:
            snapshot_json: ProcessSnapshot 的 JSON 字符串
            target_node: 目标节点 ID

        Returns:
            目标节点返回的响应（解析为 dict）

        Raises:
            TransportError: 发送失败（网络错误、超时、HTTP 错误等）
        """
        node = self._resolve_node(target_node)
        url = f"http://{node['host']}:{node['port']}/migration/import"

        data = snapshot_json.encode("utf-8")
        req = urllib.request.Request(
            url,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                resp_body = resp.read().decode("utf-8")
                return json.loads(resp_body)
        except urllib.error.HTTPError as e:
            raise TransportError(
                target_node, f"HTTP {e.code}: {e.reason}"
            ) from e
        except urllib.error.URLError as e:
            raise TransportError(
                target_node, f"URL error: {e.reason}"
            ) from e
        except socket.timeout:
            raise TransportError(target_node, "timeout") from None
        except Exception as e:
            raise TransportError(
                target_node, f"unexpected error: {e}"
            ) from e

    def recv(self) -> str:
        """
        接收来自其他节点的 ProcessSnapshot JSON。

        从内部接收队列中获取最近接收到的快照 JSON。
        实际数据由 api/server.py 的 POST /migration/import 端点通过
        _on_receive() 方法推入队列。

        Returns:
            接收到的 ProcessSnapshot JSON 字符串

        Raises:
            TransportError: 接收队列为空（无可用数据）
        """
        if not self._recv_queue:
            raise TransportError("", "receive queue empty, no data available")
        return self._recv_queue.popleft()

    def _on_receive(self, snapshot_json: str) -> None:
        """
        内部方法：由 API 服务器调用，将接收到的快照 JSON 推入接收队列。

        当本地节点的 /migration/import 端点收到 POST 请求时，
        api/server.py 应调用此方法将数据存入队列，供 recv() 读取。

        Args:
            snapshot_json: 接收到的 ProcessSnapshot JSON 字符串
        """
        self._recv_queue.append(snapshot_json)

    def list_nodes(self) -> list[str]:
        """
        返回可用节点列表（不含本节点）。

        Returns:
            可用节点 ID 列表
        """
        return [nid for nid in self.nodes_config if nid != self.node_id]
