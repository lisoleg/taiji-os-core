"""core/migration_agent.py — 跨节点迁移代理模块

实现 AGI 进程在集群节点间的完整迁移能力，包含：
  - ProcessSnapshot: 进程完整状态快照，SHA-256 完整性证明
  - MigrationManager: 迁移控制器，管理导出/导入/传输全链路
  - LoadBalancer: 负载均衡器，基于 CPU/内存阈值触发自动迁移
  - 异常体系: IntegrityError / MigrationError（继承 USCSError）

复用模式：
  - ProcessSnapshot.verify() 复用 Continuation.verify() 的 proof 链验证模式
  - MigrationManager.export_process() 冻结进程的方式复用 TaijiSession._save_continuation()
  - MigrationManager.import_process() 恢复进程的方式复用 TaijiSession.resume()
"""
from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from enum import Enum
from typing import TYPE_CHECKING, Optional

import numpy as np

from core.uscs_mmu import USCSError

if TYPE_CHECKING:
    from core.preemptive_scheduler import PreemptiveScheduler
    from core.uscs_mmu import PageAllocator


# ---------------------------------------------------------------------------
# 异常体系
# ---------------------------------------------------------------------------


class IntegrityError(USCSError):
    """完整性校验失败异常。

    Attributes:
        kid: Continuation ID
        expected_proof: 期望的 proof 值
    """

    def __init__(self, kid: str, expected_proof: str = ""):
        self.kid = kid
        self.expected_proof = expected_proof
        msg = f"IntegrityError: integrity check failed for kid={kid!r}"
        if expected_proof:
            msg += f", expected_proof={expected_proof[:16]}..."
        super().__init__(msg)


class MigrationError(USCSError):
    """迁移失败异常。

    Attributes:
        pid: 进程 ID
        phase: 失败阶段 — "preparing" | "transferring" | "restoring"
        cause: 原始异常
    """

    def __init__(self, pid: str, phase: str, cause: Exception):
        self.pid = pid
        self.phase = phase
        self.cause = cause
        super().__init__(
            f"MigrationError(pid={pid!r}, phase={phase!r}): {cause}"
        )


# ---------------------------------------------------------------------------
# 枚举
# ---------------------------------------------------------------------------


class MigrationState(Enum):
    """迁移状态枚举。"""
    PREPARING = "preparing"
    TRANSFERRING = "transferring"
    RESTORING = "restoring"
    COMPLETED = "completed"
    FAILED = "failed"


# ---------------------------------------------------------------------------
# MigrationStatus — 迁移状态追踪
# ---------------------------------------------------------------------------


@dataclass
class MigrationStatus:
    """迁移状态追踪数据类。

    Attributes:
        pid: 进程 ID
        target_node: 目标节点 ID
        state: 当前迁移状态
        started_at: 迁移开始时间 (ISO 8601)
        progress_pct: 进度百分比 0.0 ~ 100.0
        error: 错误信息（仅在 FAILED 状态下有值）
    """
    pid: str
    target_node: str
    state: MigrationState
    started_at: str
    progress_pct: float = 0.0
    error: Optional[str] = None


# ---------------------------------------------------------------------------
# ProcessSnapshot — 进程完整状态快照
# ---------------------------------------------------------------------------


@dataclass
class ProcessSnapshot:
    """进程完整状态快照，用于跨节点迁移。

    包含进程运行所需的全部状态：Continuation 快照、页表、PCB、
    自我模型 σ 向量，以及 SHA-256 完整性证明。

    Attributes:
        pid: 进程 ID
        continuation_kid: 关联的 Continuation 快照 ID
        continuation_data: Continuation 完整数据
        page_table_data: PageTable.to_dict() 结果
        pcb_data: PCB 可序列化视图
        sigma_data: SelfModel.sigma 向量的 float 列表
        proof: SHA-256 迁移完整性证明
        source_node: 源节点 ID
        created_at: 快照创建时间 (ISO 8601)
    """
    pid: str
    continuation_kid: str
    continuation_data: dict
    page_table_data: dict
    pcb_data: dict
    sigma_data: list
    proof: str
    source_node: str
    created_at: str

    def compute_proof(self) -> str:
        """计算快照的 SHA-256 完整性证明。

        proof = SHA-256(all_fields_except_proof)，序列化时排序键以确保确定性。

        Returns:
            64 字符 hex 哈希字符串
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
        raw = json.dumps(payload, ensure_ascii=False, sort_keys=True)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def verify(self) -> bool:
        """验证快照完整性。

        重新计算 proof 并与存储的 proof 比对，
        复用 Continuation.verify() 的 SHA-256 proof 验证模式。

        Returns:
            True 表示快照完整未被篡改，False 表示验证失败
        """
        recomputed = self.compute_proof()
        return recomputed == self.proof

    def to_json(self) -> str:
        """将快照序列化为 JSON 字符串。

        使用 json.dumps + dataclasses.asdict，保持与网络传输兼容。

        Returns:
            JSON 字符串
        """
        return json.dumps(
            asdict(self), ensure_ascii=False, sort_keys=True, indent=2
        )

    @classmethod
    def from_json(cls, raw: str) -> "ProcessSnapshot":
        """从 JSON 字符串反序列化快照。

        注意：反序列化后的 proof 为存储值，调用方应通过 verify() 校验。

        Args:
            raw: JSON 字符串

        Returns:
            重建的 ProcessSnapshot 实例
        """
        data = json.loads(raw)
        return cls(
            pid=data["pid"],
            continuation_kid=data["continuation_kid"],
            continuation_data=data["continuation_data"],
            page_table_data=data["page_table_data"],
            pcb_data=data["pcb_data"],
            sigma_data=data["sigma_data"],
            proof=data["proof"],
            source_node=data["source_node"],
            created_at=data["created_at"],
        )


# ---------------------------------------------------------------------------
# MigrationManager — 迁移管理器
# ---------------------------------------------------------------------------


class MigrationManager:
    """跨节点迁移管理器。

    管理进程从导出到导入的完整迁移生命周期，包括：
      - 冻结源进程并导出完整状态快照
      - 在目标节点重建进程状态
      - 迁移完整性验证
      - 迁移状态追踪与取消

    Attributes:
        node_id: 本节点 ID
        transport: 节点间传输层接口（NodeTransport）
        scheduler: 抢占调度器实例（PreemptiveScheduler）
        allocator: 物理页分配器实例（PageAllocator）
        llm_router: LLM 路由实例（用于重建 session）
        _active_migrations: 活跃迁移追踪字典
    """

    def __init__(
        self,
        node_id: str,
        transport,
        scheduler: "PreemptiveScheduler",
        allocator: "PageAllocator",
        llm_router=None,
    ):
        """初始化迁移管理器。

        Args:
            node_id: 本节点 ID
            transport: 节点间传输层实例（支持 send/recv 接口）
            scheduler: 抢占调度器实例
            allocator: 物理页分配器实例
            llm_router: LLM 路由实例，用于在目标节点重建 TaijiSession
        """
        self.node_id = node_id
        self.transport = transport
        self.scheduler = scheduler
        self.allocator = allocator
        self.llm_router = llm_router
        self._active_migrations: dict[str, MigrationStatus] = {}

    def export_process(self, pid: str) -> ProcessSnapshot:
        """导出进程的完整状态快照。

        执行步骤:
            a. 冻结进程（scheduler.block(pid, "migrating")）
            b. ContextSwitch.save(session)
            c. PageTable.to_dict()
            d. PCB 序列化
            e. Continuation.load(kid)
            f. 打包为 ProcessSnapshot，compute_proof()

        Args:
            pid: 要导出的进程 ID

        Returns:
            ProcessSnapshot 完整状态快照

        Raises:
            MigrationError: 导出过程中发生异常
            ScheduleError: 进程不存在或状态不合法
        """
        try:
            # ---- 步骤 a: 冻结进程 ----
            pcb = self.scheduler.get_pcb(pid)
            if pcb is None:
                raise MigrationError(
                    pid, "preparing", ValueError(f"Process {pid!r} not found")
                )
            if pcb.session is None:
                raise MigrationError(
                    pid, "preparing", ValueError(f"Process {pid!r} has no session")
                )

            self.scheduler.block(pid, "migrating")
            self._update_status(
                pid, MigrationState.PREPARING, progress_pct=10.0
            )

            # ---- 步骤 b: ContextSwitch.save(session) ----
            from core.preemptive_scheduler import ContextSwitch

            ctx_snapshot = ContextSwitch.save(pcb.session)
            self._update_status(pid, progress_pct=30.0)

            # ---- 步骤 c: PageTable.to_dict() ----
            page_table_data = {}
            if pcb.page_table is not None:
                page_table_data = pcb.page_table.to_dict()
            self._update_status(pid, progress_pct=50.0)

            # ---- 步骤 d: PCB 序列化 ----
            pcb_data = pcb.to_dict()
            self._update_status(pid, progress_pct=60.0)

            # ---- 步骤 e: Continuation 数据获取 ----
            # 先调用 _save_continuation 获取最新的 kid
            session = pcb.session
            env_dict = session.env.to_dict() if session.env else {}
            reason = "migration"
            k = session._save_continuation(env_dict, reason)
            continuation_kid = k.kid

            # 加载 Continuation 完整数据
            from core.continuation import Continuation

            c = Continuation.load(continuation_kid, session.snapshot_dir)
            continuation_data = {
                "kid": c.kid,
                "sid": c.sid,
                "psi": c.psi.tolist(),
                "env": c.env,
                "reason": c.reason,
                "ts": c.ts,
                "parent_kid": c.parent_kid,
                "payload_hash": c.payload_hash,
                "proof": c.proof,
            }
            self._update_status(pid, progress_pct=80.0)

            # ---- 步骤 f: 打包 + 计算 proof ----
            snapshot = ProcessSnapshot(
                pid=pid,
                continuation_kid=continuation_kid,
                continuation_data=continuation_data,
                page_table_data=page_table_data,
                pcb_data=pcb_data,
                sigma_data=ctx_snapshot.get("sigma", []),
                proof="",  # 先占位，下面 compute_proof() 填充
                source_node=self.node_id,
                created_at=datetime.now(timezone.utc).isoformat(),
            )
            snapshot.proof = snapshot.compute_proof()

            self._update_status(pid, MigrationState.PREPARING, progress_pct=100.0)
            return snapshot

        except MigrationError:
            raise
        except Exception as e:
            raise MigrationError(pid, "preparing", e)

    def import_process(self, snapshot: ProcessSnapshot, new_pid: Optional[str] = None) -> str:
        """从快照导入/重建进程。

        执行步骤:
            a. verify() 校验完整性
            b. PageAllocator.alloc(n) 分配物理页
            c. PageTable.from_dict(data) 重建页表
            d. 重建 TaijiSession
            e. 恢复 ψ、σ、env
            f. scheduler.register(session, priority)
            g. scheduler.unblock(new_pid)
            h. 返回新 pid

        Args:
            snapshot: 要导入的进程快照
            new_pid: 可选的新进程 ID，默认使用原始 pid

        Returns:
            新进程的 pid

        Raises:
            IntegrityError: 完整性校验失败
            MigrationError: 恢复过程中发生异常
        """
        pid = new_pid or snapshot.pid

        try:
            # ---- 步骤 a: 完整性校验 ----
            if not snapshot.verify():
                raise IntegrityError(
                    snapshot.continuation_kid,
                    expected_proof=snapshot.proof,
                )

            self._update_status(
                pid, MigrationState.RESTORING, progress_pct=10.0
            )

            # ---- 步骤 b: 分配物理页 ----
            page_table_data = snapshot.page_table_data
            n_pages = len(page_table_data.get("entries", []))
            if n_pages > 0:
                self.allocator.alloc(n_pages)
            self._update_status(pid, progress_pct=30.0)

            # ---- 步骤 c: 重建页表 ----
            from core.uscs_mmu import PageTable

            new_page_table = PageTable.from_dict(page_table_data) if page_table_data else PageTable(pid=pid)
            new_page_table.pid = pid
            self._update_status(pid, progress_pct=50.0)

            # ---- 步骤 d: 重建 TaijiSession ----
            if self.llm_router is None:
                raise MigrationError(
                    pid, "restoring",
                    ValueError("llm_router is required to rebuild TaijiSession")
                )

            from core.session import TaijiSession

            session = TaijiSession(
                sid=f"session-{pid}",
                llm_router=self.llm_router,
                snapshot_dir="snapshots",
            )
            self._update_status(pid, progress_pct=70.0)

            # ---- 步骤 e: 恢复 ψ、σ、env ----
            cont = snapshot.continuation_data
            # 恢复 ψ 向量
            session.w.psi = np.array(cont.get("psi", []))
            # 恢复 σ 向量
            session.self_model.sigma = np.array(snapshot.sigma_data)
            # 恢复 ClosureEnv
            from core.closure_env import ClosureEnv

            session.env = ClosureEnv.from_dict(cont.get("env", {}))
            self._update_status(pid, progress_pct=85.0)

            # ---- 步骤 f: 注册到调度器 ----
            from core.preemptive_scheduler import Priority

            pcb_data = snapshot.pcb_data
            priority_name = pcb_data.get("priority", "MEDIUM")
            priority = Priority[priority_name] if priority_name in Priority.__members__ else Priority.MEDIUM

            session.page_table = new_page_table
            self.scheduler.register(session, priority)
            # 更新 pcb 的 pid（registration 使用 sid 生成，这里手动覆盖）
            if session.pcb:
                session.pcb.pid = pid

            self._update_status(pid, progress_pct=95.0)

            # ---- 步骤 g: 解阻塞 ----
            self.scheduler.unblock(pid)

            # ---- 步骤 h: 返回新 pid ----
            self._update_status(
                pid, MigrationState.COMPLETED, progress_pct=100.0
            )
            return pid

        except (IntegrityError, MigrationError):
            raise
        except Exception as e:
            raise MigrationError(pid, "restoring", e)

    def migrate(self, pid: str, target_node: str) -> str:
        """将进程迁移到目标节点。

        完整迁移流程:
            1. export_process(pid) 导出快照
            2. transport.send(snapshot, target_node) 传输
            3. 目标节点 import_process(snapshot) 重建
            4. 源节点 unregister(pid) 清理

        Args:
            pid: 要迁移的进程 ID
            target_node: 目标节点 ID

        Returns:
            目标节点上新建进程的 pid

        Raises:
            MigrationError: 迁移失败（重试后仍失败）
        """
        # 初始化迁移状态
        self._active_migrations[pid] = MigrationStatus(
            pid=pid,
            target_node=target_node,
            state=MigrationState.PREPARING,
            started_at=datetime.now(timezone.utc).isoformat(),
            progress_pct=0.0,
        )

        try:
            # Step 1: 导出
            self._update_status(pid, MigrationState.PREPARING, progress_pct=10.0)
            snapshot = self.export_process(pid)
            self._update_status(pid, progress_pct=40.0)

            # Step 2: 传输
            self._update_status(
                pid, MigrationState.TRANSFERRING, progress_pct=50.0
            )
            try:
                raw = snapshot.to_json()
                self.transport.send(raw, target_node)
            except Exception as e:
                raise MigrationError(pid, "transferring", e)
            self._update_status(pid, progress_pct=70.0)

            # Step 3: 目标节点重建（本地模拟：直接调用 import_process）
            self._update_status(
                pid, MigrationState.RESTORING, progress_pct=80.0
            )
            new_pid = self.import_process(snapshot)
            self._update_status(pid, progress_pct=90.0)

            # Step 4: 源节点清理
            self.scheduler.unregister(pid)
            self._update_status(
                pid, MigrationState.COMPLETED, progress_pct=100.0
            )

            return new_pid

        except (IntegrityError, MigrationError):
            self._update_status(
                pid, MigrationState.FAILED,
                error="migration failed, see traceback for details"
            )

            # 迁移失败：尝试恢复源进程
            try:
                if pid in self.scheduler.pcb_map:
                    self.scheduler.unblock(pid)
            except Exception:
                pass
            raise
        except Exception as e:
            self._update_status(
                pid, MigrationState.FAILED,
                error=str(e)
            )
            raise MigrationError(pid, "preparing", e)

    def verify(self, source_pid: str, target_pid: str) -> bool:
        """验证源进程和目标进程状态一致性。

        对比两个进程的 Continuation proof、ψ 向量和 σ 向量。

        Args:
            source_pid: 源进程 ID
            target_pid: 目标进程 ID

        Returns:
            True 表示状态一致，False 表示不一致
        """
        src_pcb = self.scheduler.get_pcb(source_pid)
        tgt_pcb = self.scheduler.get_pcb(target_pid)

        if src_pcb is None or tgt_pcb is None:
            return False

        if src_pcb.session is None or tgt_pcb.session is None:
            return False

        src_session = src_pcb.session
        tgt_session = tgt_pcb.session

        try:
            # 比较 ψ 向量
            if not np.array_equal(src_session.w.psi, tgt_session.w.psi):
                return False

            # 比较 σ 向量
            if not np.array_equal(
                src_session.self_model.sigma,
                tgt_session.self_model.sigma,
            ):
                return False

            # 比较 Continuation 快照 key
            if src_session._last_kid != tgt_session._last_kid:
                return False

            return True
        except Exception:
            return False

    def status(self, pid: str) -> dict:
        """查询指定进程的迁移状态。

        Args:
            pid: 进程 ID

        Returns:
            包含迁移状态信息的字典，若不存在则返回 None 状态
        """
        ms = self._active_migrations.get(pid)
        if ms is None:
            return {"pid": pid, "state": None, "message": "no active migration"}
        return {
            "pid": ms.pid,
            "target_node": ms.target_node,
            "state": ms.state.value,
            "started_at": ms.started_at,
            "progress_pct": ms.progress_pct,
            "error": ms.error,
        }

    def cancel(self, pid: str) -> bool:
        """取消进行中的迁移。

        取消迁移并将源进程恢复为就绪状态。

        Args:
            pid: 进程 ID

        Returns:
            True 表示取消成功，False 表示不存在活跃迁移
        """
        if pid not in self._active_migrations:
            return False

        ms = self._active_migrations[pid]
        if ms.state in (MigrationState.COMPLETED, MigrationState.FAILED):
            return False

        ms.state = MigrationState.FAILED
        ms.error = "cancelled by user"

        # 尝试恢复源进程
        try:
            if pid in self.scheduler.pcb_map:
                self.scheduler.unblock(pid)
        except Exception:
            pass

        return True

    # ---- 内部辅助 ----
    def _update_status(
        self,
        pid: str,
        state: Optional[MigrationState] = None,
        progress_pct: Optional[float] = None,
        error: Optional[str] = None,
    ) -> None:
        """更新活跃迁移的状态（内部方法）。"""
        if pid not in self._active_migrations:
            self._active_migrations[pid] = MigrationStatus(
                pid=pid,
                target_node="",
                state=MigrationState.PREPARING,
                started_at=datetime.now(timezone.utc).isoformat(),
            )
        ms = self._active_migrations[pid]
        if state is not None:
            ms.state = state
        if progress_pct is not None:
            ms.progress_pct = progress_pct
        if error is not None:
            ms.error = error


# ---------------------------------------------------------------------------
# LoadBalancer — 负载均衡器
# ---------------------------------------------------------------------------


class LoadBalancer:
    """负载均衡器，基于 CPU/内存使用率触发自动进程迁移。

    周期性检查节点负载，当超过阈值时自动选择可迁移进程
    并调用 MigrationManager.migrate() 分发到低负载节点。

    Attributes:
        migration_mgr: MigrationManager 实例
        cpu_threshold: CPU 使用率迁移阈值 (0.0 ~ 1.0)
        mem_threshold: 内存使用率迁移阈值 (0.0 ~ 1.0)
    """

    def __init__(
        self,
        migration_mgr: MigrationManager,
        cpu_threshold: float = 0.85,
        mem_threshold: float = 0.85,
    ):
        """初始化负载均衡器。

        Args:
            migration_mgr: MigrationManager 实例
            cpu_threshold: CPU 使用率阈值，默认 0.85
            mem_threshold: 内存使用率阈值，默认 0.85
        """
        self.migration_mgr = migration_mgr
        self.cpu_threshold = cpu_threshold
        self.mem_threshold = mem_threshold

    def check_and_balance(self) -> list[str]:
        """检查节点负载，必要时触发迁移。

        通过 transport 获取集群各节点状态，对比本节点负载。
        如果本节点超过阈值，选出低负载目标节点并迁移进程。

        Returns:
            被迁移的进程 pid 列表
        """
        migrated: list[str] = []

        stats = self.node_stats()
        cpu_pct = stats.get("cpu_used_percent", 0.0)
        mem_pct = stats.get("mem_used_percent", 0.0)

        # 检查是否超过阈值
        if cpu_pct < self.cpu_threshold and mem_pct < self.mem_threshold:
            return migrated

        # 获取集群节点状态（通过 transport）
        try:
            cluster_stats = self.migration_mgr.transport.get_cluster_stats()
        except Exception:
            # 无法获取集群状态，跳过
            return migrated

        # 找出低负载目标节点
        target_nodes = []
        for node in cluster_stats:
            node_cpu = node.get("cpu_used_percent", 1.0)
            node_mem = node.get("mem_used_percent", 1.0)
            if node_cpu < self.cpu_threshold and node_mem < self.mem_threshold:
                target_nodes.append((node["node_id"], node_cpu))

        if not target_nodes:
            return migrated

        # 按负载升序排序，优先选最空闲的节点
        target_nodes.sort(key=lambda x: x[1])

        # 从调度器中选可迁移的进程
        scheduler = self.migration_mgr.scheduler
        queue_snap = scheduler.queue_snapshot()

        # 优先选择低优先级的就绪进程
        migratable_pids = []
        for p_name in ("LOW", "MEDIUM", "HIGH"):
            for pid in queue_snap.get("ready", {}).get(p_name, []):
                pcb = scheduler.get_pcb(pid)
                if pcb and not pcb.wait_reason:
                    migratable_pids.append(pid)

        # 执行迁移
        for pid in migratable_pids[: len(target_nodes)]:
            target = target_nodes[migratable_pids.index(pid)]
            try:
                self.migration_mgr.migrate(pid, target[0])
                migrated.append(pid)
            except Exception:
                continue

        return migrated

    def set_threshold(self, cpu_pct: float, mem_pct: float) -> None:
        """设置 CPU 和内存使用率迁移阈值。

        Args:
            cpu_pct: CPU 使用率阈值 (0.0 ~ 1.0)
            mem_pct: 内存使用率阈值 (0.0 ~ 1.0)
        """
        self.cpu_threshold = cpu_pct
        self.mem_threshold = mem_pct

    def node_stats(self) -> dict:
        """获取本节点资源使用统计。

        通过 allocator 和 scheduler 计算当前节点状态。

        Returns:
            包含节点统计信息的字典
        """
        allocator = self.migration_mgr.allocator
        scheduler = self.migration_mgr.scheduler

        # 内存使用率
        mem_used = allocator.used()
        mem_total = allocator.total()
        mem_used_percent = mem_used / max(mem_total, 1)

        # CPU 使用率（基于活跃进程数估算）
        sched_stats = scheduler.stats()
        queue_snap = scheduler.queue_snapshot()
        ready_count = sum(len(v) for v in queue_snap.get("ready", {}).values())
        if sched_stats.get("current_pid") is not None:
            ready_count += 1

        cpu_used_percent = min(
            1.0,
            ready_count / max(sched_stats.get("total_processes", 1), 1),
        )

        return {
            "node_id": self.migration_mgr.node_id,
            "cpu_used_percent": cpu_used_percent,
            "mem_used_percent": mem_used_percent,
            "mem_used_pages": mem_used,
            "mem_total_pages": mem_total,
            "mem_available_pages": allocator.available(),
            "total_processes": sched_stats.get("total_processes", 0),
            "ready_count": ready_count,
            "blocked_count": sched_stats.get("blocked_count", 0),
            "waiting_count": sched_stats.get("waiting_count", 0),
        }
