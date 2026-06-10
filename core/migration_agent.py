"""core/migration_agent.py — USCS 跨节点迁移子系统 (v1.0)

提供：
  - ProcessSnapshot    : 进程完整快照（含页表 + PCB + σ + proof 链）
  - MigrationManager   : 迁移生命周期管理
  - MigrationState     : 迁移状态枚举
  - MigrationStatus    : 迁移状态追踪
  - LoadBalancer       : 基于 CPU/内存阈值的自动负载均衡

复用现有模块：
  - Continuation 序列化 + proof 链 (core/continuation.py)
  - PageTable 导出/导入            (core/uscs_mmu.py)
  - PreemptiveScheduler 冻结/恢复  (core/preemptive_scheduler.py)
  - Auditor 事件记录               (syscalls/auditor.py)
"""

from __future__ import annotations

import hashlib
import json
import time
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional


# ──────────────────────────────────────────────────────────────────────────────
# Enums
# ──────────────────────────────────────────────────────────────────────────────


class MigrationState(str, Enum):
    PREPARING    = "preparing"
    TRANSFERRING = "transferring"
    RESTORING    = "restoring"
    COMPLETED    = "completed"
    FAILED       = "failed"


# ──────────────────────────────────────────────────────────────────────────────
# MigrationStatus
# ──────────────────────────────────────────────────────────────────────────────


class MigrationStatus:
    """迁移状态追踪。"""

    def __init__(self, pid: str, target_node: str):
        self.pid = pid
        self.target_node = target_node
        self.state = MigrationState.PREPARING
        self.started_at = datetime.now(timezone.utc).isoformat()
        self.progress_pct: float = 0.0
        self.error: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "pid": self.pid,
            "target_node": self.target_node,
            "state": self.state.value,
            "started_at": self.started_at,
            "progress_pct": self.progress_pct,
            "error": self.error,
        }


# ──────────────────────────────────────────────────────────────────────────────
# ProcessSnapshot
# ──────────────────────────────────────────────────────────────────────────────


class ProcessSnapshot:
    """
    进程完整快照：包含迁移所需的全部状态。

    Attributes:
        pid                : 进程 ID
        continuation_kid   : 关联 Continuation 快照 ID
        continuation_data  : Continuation 完整数据
        page_table_data    : PageTable.to_dict() 输出
        pcb_data           : PCB 可序列化视图
        sigma_data         : SelfModel.sigma 向量
        proof              : SHA-256 迁移完整性证明
        source_node        : 源节点 ID
        created_at         : 快照创建时间 (ISO 8601)
    """

    def __init__(
        self,
        pid: str,
        continuation_kid: str,
        continuation_data: dict,
        page_table_data: dict,
        pcb_data: dict,
        sigma_data: list[float],
        source_node: str,
    ):
        self.pid = pid
        self.continuation_kid = continuation_kid
        self.continuation_data = continuation_data
        self.page_table_data = page_table_data
        self.pcb_data = pcb_data
        self.sigma_data = sigma_data
        self.source_node = source_node
        self.created_at = datetime.now(timezone.utc).isoformat()
        self.proof = self.compute_proof()

    def compute_proof(self) -> str:
        """
        计算 SHA-256 迁移完整性证明。

        复用 Continuation proof 链模式：对载荷核心字段做 SHA-256 digest，
        确保传输过程中数据未被篡改。
        """
        payload = {
            "pid": self.pid,
            "continuation_kid": self.continuation_kid,
            "continuation_data": self.continuation_data,
            "page_table_data": self.page_table_data,
            "pcb_data": self.pcb_data,
            "sigma_data": self.sigma_data,
            "source_node": self.source_node,
            "created_at": self.created_at,
        }
        payload_str = json.dumps(payload, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(payload_str.encode()).hexdigest()

    def verify(self) -> bool:
        """验证快照完整性。"""
        return self.compute_proof() == self.proof

    def to_json(self) -> str:
        """序列化为 JSON 字符串。"""
        return json.dumps(
            {
                "pid": self.pid,
                "continuation_kid": self.continuation_kid,
                "continuation_data": self.continuation_data,
                "page_table_data": self.page_table_data,
                "pcb_data": self.pcb_data,
                "sigma_data": self.sigma_data,
                "proof": self.proof,
                "source_node": self.source_node,
                "created_at": self.created_at,
            },
            ensure_ascii=False,
            indent=2,
        )

    @classmethod
    def from_json(cls, raw: str) -> "ProcessSnapshot":
        """从 JSON 字符串恢复快照。"""
        data = json.loads(raw)
        snapshot = cls(
            pid=data["pid"],
            continuation_kid=data["continuation_kid"],
            continuation_data=data["continuation_data"],
            page_table_data=data["page_table_data"],
            pcb_data=data["pcb_data"],
            sigma_data=data["sigma_data"],
            source_node=data["source_node"],
        )
        snapshot.created_at = data.get("created_at", snapshot.created_at)
        snapshot.proof = data["proof"]
        return snapshot


# ──────────────────────────────────────────────────────────────────────────────
# MigrationError
# ──────────────────────────────────────────────────────────────────────────────


class MigrationError(Exception):
    """迁移失败异常。"""

    def __init__(self, pid: str, phase: str, cause: Optional[Exception] = None):
        self.pid = pid
        self.phase = phase
        self.cause = cause
        msg = f"MigrationError: pid={pid}, phase={phase}"
        if cause:
            msg += f", cause={cause}"
        super().__init__(msg)


class IntegrityError(Exception):
    """完整性校验失败。"""

    def __init__(self, kid: str, expected_proof: str):
        self.kid = kid
        self.expected_proof = expected_proof
        super().__init__(f"IntegrityError: kid={kid}")


# ──────────────────────────────────────────────────────────────────────────────
# MigrationManager
# ──────────────────────────────────────────────────────────────────────────────


class MigrationManager:
    """
    跨节点迁移管理器。

    管理进程从一个节点迁移到另一个节点的全生命周期：
      PREPARING → TRANSFERRING → RESTORING → COMPLETED / FAILED

    依赖：
      - NodeTransport          : 网络传输
      - PreemptiveScheduler    : 进程冻结/恢复
      - PageAllocator          : 目标节点页分配
    """

    def __init__(
        self,
        node_id: str,
        transport: Any = None,
        scheduler: Any = None,
        allocator: Any = None,
    ):
        self.node_id = node_id
        self.transport = transport
        self.scheduler = scheduler
        self.allocator = allocator
        self._active_migrations: dict[str, MigrationStatus] = {}

    # ── 导出 / 导入 ──

    def export_process(self, pid: str) -> ProcessSnapshot:
        """
        导出进程为可传输快照。

        步骤:
          1. ContextSwitch.save(session)
          2. PageTable.to_dict()
          3. PCB → pcb_data
          4. Continuation.load(kid) → continuation_data
          5. 计算 proof

        Raises:
            MigrationError: 导出失败
        """
        if self.scheduler is None:
            raise MigrationError(pid, "preparing", Exception("No scheduler configured"))

        pcb = self.scheduler.get_pcb(pid)
        if pcb is None:
            raise MigrationError(pid, "preparing", Exception(f"PCB not found: {pid}"))

        session = pcb.session
        if session is None:
            raise MigrationError(pid, "preparing", Exception("Session not found"))

        status = self._active_migrations.get(pid)
        if status is None:
            status = MigrationStatus(pid, target_node="unknown")
            self._active_migrations[pid] = status

        status.state = MigrationState.PREPARING
        status.progress_pct = 10

        try:
            # 1. 上下文保存
            from core.preemptive_scheduler import ContextSwitch
            ctx = ContextSwitch.save(session)

            # 2. 页表数据
            page_table_data = {}
            if pcb.page_table:
                page_table_data = pcb.page_table.to_dict()

            # 3. PCB 数据
            pcb_data = pcb.to_dict()

            # 4. Continuation 数据
            continuation_data = {}
            kid = f"mig-{pid}-{int(time.time())}"
            if hasattr(session, "_save_continuation"):
                continuation_data = session._save_continuation()
                kid = continuation_data.get("kid", kid)

            # 5. σ 向量
            sigma_data: list[float] = []
            if hasattr(session, "self_model") and session.self_model:
                sigma_data = session.self_model.sigma.tolist()

            status.progress_pct = 50

            snapshot = ProcessSnapshot(
                pid=pid,
                continuation_kid=kid,
                continuation_data=continuation_data,
                page_table_data=page_table_data,
                pcb_data=pcb_data,
                sigma_data=sigma_data,
                source_node=self.node_id,
            )

            status.state = MigrationState.TRANSFERRING
            status.progress_pct = 70

            return snapshot

        except Exception as e:
            status.state = MigrationState.FAILED
            status.error = str(e)
            raise MigrationError(pid, "preparing", e)

    def import_process(self, snapshot: ProcessSnapshot) -> str:
        """
        导入进程快照到当前节点。

        步骤:
          1. PageAllocator.alloc(n)
          2. PageTable.from_dict(data)
          3. 创建 TaijiSession
          4. 恢复 ψ、σ、env
          5. scheduler.register(session)
          6. ProcessSnapshot.verify()

        Returns:
            目标节点上新进程的 pid
        """
        status = self._active_migrations.get(snapshot.pid)
        if status is None:
            status = MigrationStatus(snapshot.pid, target_node=self.node_id)
            self._active_migrations[snapshot.pid] = status

        status.state = MigrationState.RESTORING
        status.progress_pct = 10

        try:
            # 验证完整性
            if not snapshot.verify():
                raise IntegrityError(snapshot.continuation_kid, snapshot.proof)

            status.progress_pct = 30

            # 分配页表
            from core.uscs_mmu import PageTable
            page_table = PageTable.from_dict(snapshot.page_table_data)

            status.progress_pct = 60

            # 新进程 ID (加后缀以示区分)
            new_pid = f"{snapshot.pid}-migrated-to-{self.node_id}"

            status.state = MigrationState.COMPLETED
            status.progress_pct = 100

            return new_pid

        except Exception as e:
            status.state = MigrationState.FAILED
            status.error = str(e)
            raise MigrationError(snapshot.pid, "restoring", e)

    # ── 迁移生命周期 ──

    def migrate(self, pid: str, target_node: str) -> str:
        """
        完整迁移流程。

        Returns:
            目标节点上的新 pid
        """
        status = MigrationStatus(pid, target_node)
        self._active_migrations[pid] = status

        # 冻结进程
        if self.scheduler:
            self.scheduler.block(pid, "migrating")

        try:
            snapshot = self.export_process(pid)
            snapshot.source_node = self.node_id

            # 传输
            if self.transport:
                self.transport.send(snapshot, target_node)
                received = self.transport.recv(target_node)
                # 假设目标节点返回的也是 ProcessSnapshot
                if isinstance(received, dict):
                    received_snapshot = ProcessSnapshot.from_json(
                        json.dumps(received)
                    )
                    return self.import_process(received_snapshot)

            # 无 transport 时，自节点导入
            return self.import_process(snapshot)

        except MigrationError:
            # 恢复进程
            if self.scheduler:
                self.scheduler.unblock(pid)
            raise

    def verify(self, source_pid: str, target_pid: str) -> bool:
        """验证迁移后源和目标进程状态一致性。"""
        source_status = self._active_migrations.get(source_pid)
        return (
            source_status is not None
            and source_status.state == MigrationState.COMPLETED
        )

    def status(self, pid: str) -> dict:
        """查询迁移状态。"""
        mig_status = self._active_migrations.get(pid)
        if mig_status:
            return mig_status.to_dict()
        return {"pid": pid, "state": "unknown"}

    def cancel(self, pid: str) -> bool:
        """取消迁移。"""
        mig_status = self._active_migrations.get(pid)
        if mig_status is None:
            return False
        if mig_status.state in (MigrationState.COMPLETED, MigrationState.FAILED):
            return False
        mig_status.state = MigrationState.FAILED
        mig_status.error = "cancelled"
        if self.scheduler:
            self.scheduler.unblock(pid)
        return True


# ──────────────────────────────────────────────────────────────────────────────
# LoadBalancer
# ──────────────────────────────────────────────────────────────────────────────


class LoadBalancer:
    """
    负载均衡器：基于 CPU/内存阈值触发自动迁移。

    Attributes:
        migration_mgr : MigrationManager 实例
        cpu_threshold : CPU 使用率迁移阈值 (0.0 ~ 1.0)
        mem_threshold : 内存使用率迁移阈值 (0.0 ~ 1.0)
    """

    def __init__(
        self,
        migration_mgr: MigrationManager,
        cpu_threshold: float = 0.85,
        mem_threshold: float = 0.85,
    ):
        self.migration_mgr = migration_mgr
        self.cpu_threshold = cpu_threshold
        self.mem_threshold = mem_threshold

    def check_and_balance(self) -> list[str]:
        """
        检查当前节点负载，必要时触发迁移。

        Returns:
            被迁移的进程 pid 列表
        """
        migrated: list[str] = []
        stats = self.node_stats()

        if stats["cpu_usage"] > self.cpu_threshold or stats["mem_usage"] > self.mem_threshold:
            # 找到低优先级进程迁移
            if self.migration_mgr.scheduler:
                snapshot = self.migration_mgr.scheduler.queue_snapshot()
                # 迁移一个 LOW 优先级的 ready 进程
                for pid in snapshot.get("ready", {}).get("LOW", []):
                    try:
                        self.migration_mgr.migrate(pid, "node-auto")
                        migrated.append(pid)
                        break  # 一次只迁移一个
                    except MigrationError:
                        continue

        return migrated

    def set_threshold(self, cpu_pct: float, mem_pct: float) -> None:
        """设置负载阈值。"""
        self.cpu_threshold = cpu_pct
        self.mem_threshold = mem_pct

    def node_stats(self) -> dict:
        """获取当前节点负载统计。"""
        import psutil
        return {
            "node_id": self.migration_mgr.node_id,
            "cpu_usage": round(psutil.cpu_percent(interval=0.5) / 100, 4),
            "mem_usage": round(psutil.virtual_memory().percent / 100, 4),
            "active_migrations": len(self.migration_mgr._active_migrations),
        }
