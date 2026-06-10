"""
core/preemptive_scheduler.py — 抢占调度器模块

实现基于优先级的抢占式进程调度器，支持多级就绪队列、时间片轮转、
上下文切换、进程状态管理等功能。

本模块与 PhiScheduler 互不干扰——PhiScheduler 在 GAN 内部做语义门控，
PreemptiveScheduler 在更高层做进程调度。

复用说明：
  - ContextSwitch.save() 序列化 ψ 向量的方式复用 Continuation.__init__ 中 psi.tolist() 模式
  - ContextSwitch.restore() 恢复 ClosureEnv 的方式复用 ClosureEnv.from_dict()
  - ContextSwitch.restore() 恢复 WorldModel ψ 的方式复用 TaijiSession.resume() 中的逻辑
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Optional
from collections import deque


# ------------------------------------------------------------------
# 异常类
# ------------------------------------------------------------------


class ScheduleError(Exception):
    """
    调度异常基类。

    属性:
        pid: 进程 ID（如果有的话）
        reason: 异常原因描述
    """

    def __init__(self, reason: str = "", pid: Optional[str] = None):
        self.pid = pid
        self.reason = reason
        msg = f"ScheduleError: {reason}"
        if pid:
            msg += f" (pid={pid})"
        super().__init__(msg)


# ------------------------------------------------------------------
# 枚举定义
# ------------------------------------------------------------------


class ProcessState(Enum):
    """进程状态枚举。"""
    RUNNING = "running"
    READY = "ready"
    WAITING = "waiting"
    BLOCKED = "blocked"


class Priority(Enum):
    """进程优先级枚举（数值越小优先级越高）。"""
    HIGH = 0
    MEDIUM = 1
    LOW = 2


# ------------------------------------------------------------------
# PCB（进程控制块）
# ------------------------------------------------------------------


@dataclass
class PCB:
    """
    进程控制块（Process Control Block）。

    属性:
        pid: 进程唯一标识符
        priority: 进程优先级（Priority 枚举）
        state: 当前进程状态（ProcessState 枚举）
        ticks_remaining: 剩余时间片（tick 数）
        ticks_total: 本轮分配的时间片总量（tick 数）
        page_table: 页表对象（可选，由 uscs_mmu 模块提供）
        session: 关联的 TaijiSession 对象（可选）
        cpu_time_ms: 累计 CPU 时间（毫秒）
        switch_count: 上下文切换次数
        wait_reason: WAITING/BLOCKED 状态的原因描述
        created_at: 创建时间（ISO 8601 格式字符串）
        snapshot: 上下文切换时保存的快照（字典）
        wait_start_time: 开始等待的时间戳（用于超时计算）
    """

    pid: str
    priority: Priority
    state: ProcessState = ProcessState.READY
    ticks_remaining: int = 10  # 默认时间片
    ticks_total: int = 10
    page_table: Optional["PageTable"] = None
    session: Optional["TaijiSession"] = None
    cpu_time_ms: int = 0
    switch_count: int = 0
    wait_reason: str = ""
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    snapshot: Optional[dict] = field(default=None, repr=False)
    wait_start_time: Optional[float] = field(default=None, repr=False)

    def to_dict(self) -> dict:
        """将 PCB 序列化为字典（用于调试和迁移）。"""
        return {
            "pid": self.pid,
            "priority": self.priority.name,
            "state": self.state.value,
            "ticks_remaining": self.ticks_remaining,
            "ticks_total": self.ticks_total,
            "cpu_time_ms": self.cpu_time_ms,
            "switch_count": self.switch_count,
            "wait_reason": self.wait_reason,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "PCB":
        """从字典恢复 PCB（不包含 session 和 page_table）。"""
        pcb = cls(
            pid=data["pid"],
            priority=Priority[data["priority"]],
            state=ProcessState(data["state"]),
            ticks_remaining=data["ticks_remaining"],
            ticks_total=data["ticks_total"],
            cpu_time_ms=data["cpu_time_ms"],
            switch_count=data["switch_count"],
            wait_reason=data.get("wait_reason", ""),
            created_at=data.get("created_at", ""),
        )
        return pcb


# ------------------------------------------------------------------
# ContextSwitch（上下文切换）
# ------------------------------------------------------------------


class ContextSwitch:
    """
    上下文切换工具类（静态方法）。

    负责保存和恢复 TaijiSession 的执行状态，
    包括 ψ 向量、ClosureEnv 环境和 SelfModel 的 σ 向量。
    """

    @staticmethod
    def save(session: "TaijiSession") -> dict:
        """
        保存会话的当前状态快照。

        参数:
            session: 要保存状态的 TaijiSession 实例

        返回:
            包含序列化状态的字典，格式：
            {
                "psi": list[float],           # WorldModel ψ 向量
                "env": dict,                   # ClosureEnv 序列化结果
                "sigma": list[float],          # SelfModel σ 向量
            }

        注意:
            序列化 ψ 的方式与 Continuation._save() 中 psi.tolist() 一致。
        """
        snapshot = {}

        # 保存 ψ 向量（使用 .tolist() 匹配 Continuation 的序列化方式）
        if hasattr(session, "w") and session.w is not None:
            psi = getattr(session.w, "psi", None)
            if psi is not None:
                snapshot["psi"] = psi.tolist()

        # 保存 ClosureEnv（使用 .to_dict()）
        if hasattr(session, "env") and session.env is not None:
            snapshot["env"] = session.env.to_dict()

        # 保存 SelfModel 的 σ 向量（使用 .tolist()）
        if hasattr(session, "self_model") and session.self_model is not None:
            sigma = getattr(session.self_model, "sigma", None)
            if sigma is not None:
                snapshot["sigma"] = sigma.tolist()

        return snapshot

    @staticmethod
    def restore(session: "TaijiSession", snapshot: dict) -> None:
        """
        从快照恢复会话状态。

        参数:
            session: 要恢复状态的 TaijiSession 实例
            snapshot: 之前由 save() 方法生成的快照字典

        注意:
            - 恢复 ψ 的方式复用 TaijiSession.resume() 中的逻辑
            - 恢复 ClosureEnv 的方式复用 ClosureEnv.from_dict()
        """
        import numpy as np

        # 恢复 ψ 向量
        if "psi" in snapshot and hasattr(session, "w") and session.w is not None:
            session.w.psi = np.array(snapshot["psi"])

        # 恢复 ClosureEnv（使用 from_dict()）
        if "env" in snapshot and hasattr(session, "env"):
            from core.closure_env import ClosureEnv

            session.env = ClosureEnv.from_dict(snapshot["env"])

        # 恢复 SelfModel 的 σ 向量
        if (
            "sigma" in snapshot
            and hasattr(session, "self_model")
            and session.self_model is not None
        ):
            session.self_model.sigma = np.array(snapshot["sigma"])


# ------------------------------------------------------------------
# PreemptiveScheduler（抢占调度器）
# ------------------------------------------------------------------


class PreemptiveScheduler:
    """
    基于优先级的抢占式进程调度器。

    特性:
        - 三级就绪队列（HIGH / MEDIUM / LOW）
        - 时间片轮转 + 优先级抢占
        - 等待队列（支持超时自动唤醒）
        - 阻塞队列

    属性:
        tick_interval_ms: tick 间隔（毫秒）
        default_time_slice: 默认时间片（tick 数）
        ready_queues: 就绪队列字典，键为 Priority 枚举，值为 PCB 双端队列
        waiting_queue: 等待队列（PCB 双端队列）
        blocked_queue: 阻塞队列（PCB 双端队列）
        current: 当前运行的 PCB（可选）
        pcb_map: pid 到 PCB 的映射字典
        _tick_count: 全局 tick 计数器
        max_waiting_timeout_ms: 等待队列最大超时（毫秒）
    """

    def __init__(
        self,
        tick_interval_ms: int = 100,
        default_time_slice: int = 10,
        max_waiting_timeout_ms: int = 30000,
    ):
        """
        初始化抢占调度器。

        参数:
            tick_interval_ms: tick 间隔（毫秒），默认 100ms
            default_time_slice: 默认时间片（tick 数），默认 10 个 tick
            max_waiting_timeout_ms: 等待超时（毫秒），默认 30000ms
        """
        self.tick_interval_ms = tick_interval_ms
        self._default_time_slice = default_time_slice
        self.max_waiting_timeout_ms = max_waiting_timeout_ms

        # 三级就绪队列
        self.ready_queues: dict = {
            Priority.HIGH: deque(),
            Priority.MEDIUM: deque(),
            Priority.LOW: deque(),
        }

        # 等待队列和阻塞队列
        self.waiting_queue: deque = deque()
        self.blocked_queue: deque = deque()

        # 当前运行进程
        self.current: Optional[PCB] = None

        # pid → PCB 映射
        self.pcb_map: dict = {}

        # 全局 tick 计数器
        self._tick_count: int = 0

        # 统计信息
        self._total_switches: int = 0
        self._total_ticks: int = 0

    def register(self, session: "TaijiSession", priority: Priority) -> PCB:
        """
        注册一个新会话到调度器。

        创建 PageTable 和 PCB，并回填 session.page_table 和 session.pcb。

        参数:
            session: 要注册的 TaijiSession 实例
            priority: 进程优先级

        返回:
            新创建的 PCB 对象

        异常:
            ScheduleError: 如果会话已经注册过
        """
        # 检查是否已经注册
        if hasattr(session, "pcb") and session.pcb is not None:
            raise ScheduleError(
                f"Session already registered with pid={session.pcb.pid}",
                session.pcb.pid,
            )

        # 生成 pid（使用 session 的 sid 或生成新的）
        pid = getattr(session, "sid", f"proc_{len(self.pcb_map)}")

        # 创建 PCB
        pcb = PCB(
            pid=pid,
            priority=priority,
            state=ProcessState.READY,
            ticks_remaining=self._default_time_slice,
            ticks_total=self._default_time_slice,
        )

        # 尝试创建 PageTable（从 uscs_mmu 导入）
        try:
            from core.uscs_mmu import PageTable

            page_table = PageTable(pid=pid)
            pcb.page_table = page_table
        except ImportError:
            # uscs_mmu 尚未实现，跳过 PageTable 创建
            page_table = None

        # 关联 session 和 pcb
        pcb.session = session
        session.page_table = page_table  # 回填
        session.pcb = pcb  # 回填

        # 加入就绪队列
        self.ready_queues[priority].append(pcb)
        self.pcb_map[pid] = pcb

        return pcb

    def unregister(self, pid: str) -> None:
        """
        注销一个进程。

        参数:
            pid: 要注销的进程 ID

        异常:
            ScheduleError: 如果进程不存在
        """
        if pid not in self.pcb_map:
            raise ScheduleError(f"Process {pid} not found", pid)

        pcb = self.pcb_map[pid]

        # 如果当前运行的就是该进程，清空 current
        if self.current is not None and self.current.pid == pid:
            self.current = None

        # 从就绪队列中移除
        for priority in Priority:
            queue = self.ready_queues[priority]
            self.ready_queues[priority] = deque(
                [p for p in queue if p.pid != pid]
            )

        # 从等待队列中移除
        self.waiting_queue = deque(
            [p for p in self.waiting_queue if p.pid != pid]
        )

        # 从阻塞队列中移除
        self.blocked_queue = deque(
            [p for p in self.blocked_queue if p.pid != pid]
        )

        # 从映射中删除
        del self.pcb_map[pid]

        # 清理 session 的引用
        if pcb.session is not None:
            if hasattr(pcb.session, "pcb"):
                pcb.session.pcb = None
            if hasattr(pcb.session, "page_table"):
                pcb.session.page_table = None

    def tick(self) -> Optional[str]:
        """
        执行一次调度 tick（核心调度逻辑）。

        调度流程:
            a. 检查当前进程时间片是否耗尽
            b. 检查高优先级就绪队列是否有进程（抢占条件）
            c. 检查 waiting_queue 中超时的进程
            d. 如需切换，执行 ContextSwitch
            e. 选择下一个进程（HIGH→MEDIUM→LOW 优先级顺序）
            f. 返回被调度到的 pid（或 None）

        返回:
            被调度到的进程 pid，如果没有可调度进程则返回 None
        """
        self._tick_count += 1
        self._total_ticks += 1

        # ---- 步骤 a: 检查当前进程时间片是否耗尽 ----
        need_switch = False
        if self.current is not None:
            self.current.ticks_remaining -= 1
            self.current.cpu_time_ms += self.tick_interval_ms

            if self.current.ticks_remaining <= 0:
                # 时间片耗尽，放回就绪队列末尾
                need_switch = True
                self._requeue_current()

        # ---- 步骤 b: 检查高优先级抢占条件 ----
        if not need_switch and self.current is not None:
            # 如果当前进程不是最高优先级，且有高优先级进程就绪，则抢占
            current_priority = self.current.priority
            for priority in Priority:
                if priority.value < current_priority.value:
                    if len(self.ready_queues[priority]) > 0:
                        need_switch = True
                        self._requeue_current()
                        break

        # ---- 步骤 c: 检查 waiting_queue 中超时的进程 ----
        self._check_waiting_queue()

        # ---- 步骤 d & e: 如果需要切换，执行上下文切换并选择下一个进程 ----
        if need_switch or self.current is None:
            # 执行上下文切换：保存当前进程状态
            if self.current is not None:
                self._context_switch_out(self.current)

            # 选择下一个进程
            next_pcb = self._select_next_process()

            if next_pcb is not None:
                # 执行上下文切换：恢复下一个进程状态
                self._context_switch_in(next_pcb)
                self.current = next_pcb
                self.current.state = ProcessState.RUNNING
                self.current.switch_count += 1
                self._total_switches += 1

                # 重置时间片
                self.current.ticks_remaining = self.current.ticks_total

                return self.current.pid
            else:
                self.current = None
                return None

        # 当前进程继续运行
        return self.current.pid if self.current is not None else None

    def _requeue_current(self) -> None:
        """将当前进程重新放入就绪队列末尾。"""
        if self.current is not None:
            self.current.state = ProcessState.READY
            self.ready_queues[self.current.priority].append(self.current)

    def _check_waiting_queue(self) -> None:
        """
        检查等待队列中超时的进程，将其移回就绪队列。

        等待超过 max_waiting_timeout_ms 的进程会被自动唤醒。
        """
        ready_list = []
        current_time = time.time() * 1000  # 转换为毫秒

        remaining = deque()
        while self.waiting_queue:
            pcb = self.waiting_queue.popleft()
            if pcb.wait_start_time is not None:
                elapsed = current_time - pcb.wait_start_time
                if elapsed >= self.max_waiting_timeout_ms:
                    # 超时，唤醒
                    pcb.state = ProcessState.READY
                    pcb.wait_reason = ""
                    pcb.wait_start_time = None
                    ready_list.append(pcb)
                else:
                    remaining.append(pcb)
            else:
                # 没有设置等待时间，立即唤醒
                pcb.state = ProcessState.READY
                pcb.wait_reason = ""
                ready_list.append(pcb)

        # 将未超时的进程放回等待队列
        self.waiting_queue = remaining

        # 将唤醒的进程加入就绪队列
        for pcb in ready_list:
            self.ready_queues[pcb.priority].append(pcb)

    def _select_next_process(self) -> Optional[PCB]:
        """
        按照 HIGH → MEDIUM → LOW 的优先级顺序选择下一个进程。

        返回:
            选中的 PCB，如果没有就绪进程则返回 None
        """
        for priority in Priority:
            queue = self.ready_queues[priority]
            if len(queue) > 0:
                pcb = queue.popleft()
                return pcb
        return None

    def _context_switch_out(self, pcb: PCB) -> None:
        """
        执行上下文切换：保存进程状态到 PCB.snapshot。

        参数:
            pcb: 要保存状态的 PCB
        """
        if pcb.session is not None:
            try:
                pcb.snapshot = ContextSwitch.save(pcb.session)
            except Exception:
                pass  # 保存失败不阻止切换

    def _context_switch_in(self, pcb: PCB) -> None:
        """
        执行上下文切换：从 PCB.snapshot 恢复进程状态。

        参数:
            pcb: 要恢复状态的 PCB
        """
        if pcb.session is not None and pcb.snapshot is not None:
            try:
                ContextSwitch.restore(pcb.session, pcb.snapshot)
            except Exception:
                pass

    def yield_cpu(self, pid: str) -> None:
        """
        进程主动让出 CPU。

        参数:
            pid: 要让出 CPU 的进程 ID

        异常:
            ScheduleError: 如果进程不存在或不在运行态
        """
        if pid not in self.pcb_map:
            raise ScheduleError(f"Process {pid} not found", pid)

        pcb = self.pcb_map[pid]

        if self.current is None or self.current.pid != pid:
            raise ScheduleError(f"Process {pid} is not currently running", pid)

        # 将当前进程放回就绪队列
        pcb.state = ProcessState.READY
        self.ready_queues[pcb.priority].append(pcb)
        self.current = None

    def block(self, pid: str, reason: str = "") -> None:
        """
        阻塞一个进程。

        参数:
            pid: 要阻塞的进程 ID
            reason: 阻塞原因

        异常:
            ScheduleError: 如果进程不存在
        """
        if pid not in self.pcb_map:
            raise ScheduleError(f"Process {pid} not found", pid)

        pcb = self.pcb_map[pid]

        # 如果已经在阻塞状态，直接返回
        if pcb.state == ProcessState.BLOCKED:
            return

        original_state = pcb.state  # 保存原始状态
        pcb.state = ProcessState.BLOCKED
        pcb.wait_reason = reason

        # 从可能的位置移除
        if self.current is not None and self.current.pid == pid:
            # 正在运行，清空 current
            self.current = None
        elif original_state == ProcessState.READY:
            # 从就绪队列移除
            priority = pcb.priority
            self.ready_queues[priority] = deque(
                [p for p in self.ready_queues[priority] if p.pid != pid]
            )
        elif original_state == ProcessState.WAITING:
            # 从等待队列移除
            self.waiting_queue = deque(
                [p for p in self.waiting_queue if p.pid != pid]
            )

        self.blocked_queue.append(pcb)

    def unblock(self, pid: str) -> None:
        """
        解除对一个进程的阻塞。

        参数:
            pid: 要解除阻塞的进程 ID

        异常:
            ScheduleError: 如果进程不存在或不在阻塞态
        """
        if pid not in self.pcb_map:
            raise ScheduleError(f"Process {pid} not found", pid)

        pcb = self.pcb_map[pid]

        if pcb.state != ProcessState.BLOCKED:
            raise ScheduleError(f"Process {pid} is not blocked", pid)

        # 从阻塞队列中移除
        self.blocked_queue = deque(
            [p for p in self.blocked_queue if p.pid != pid]
        )

        # 放回就绪队列
        pcb.state = ProcessState.READY
        pcb.wait_reason = ""
        self.ready_queues[pcb.priority].append(pcb)

    def set_priority(self, pid: str, priority: Priority) -> None:
        """
        设置进程的优先级。

        参数:
            pid: 进程 ID
            priority: 新的优先级

        异常:
            ScheduleError: 如果进程不存在
        """
        if pid not in self.pcb_map:
            raise ScheduleError(f"Process {pid} not found", pid)

        pcb = self.pcb_map[pid]
        old_priority = pcb.priority
        pcb.priority = priority

        # 如果进程在旧优先级的就绪队列中，需要移到新优先级队列
        if pcb.state == ProcessState.READY:
            # 从旧队列移除
            self.ready_queues[old_priority] = deque(
                [p for p in self.ready_queues[old_priority] if p.pid != pid]
            )
            # 加入新队列
            self.ready_queues[priority].append(pcb)

    def get_pcb(self, pid: str) -> Optional[PCB]:
        """
        获取指定进程的 PCB。

        参数:
            pid: 进程 ID

        返回:
            PCB 对象，如果不存在则返回 None
        """
        return self.pcb_map.get(pid)

    def stats(self) -> dict:
        """
        获取调度器统计信息。

        返回:
            包含统计信息的字典
        """
        ready_counts = {p.name: len(self.ready_queues[p]) for p in Priority}

        return {
            "tick_count": self._tick_count,
            "total_ticks": self._total_ticks,
            "total_switches": self._total_switches,
            "current_pid": self.current.pid if self.current is not None else None,
            "ready_counts": ready_counts,
            "waiting_count": len(self.waiting_queue),
            "blocked_count": len(self.blocked_queue),
            "total_processes": len(self.pcb_map),
        }

    def queue_snapshot(self) -> dict:
        """
        获取各队列的快照（用于调试）。

        返回:
            包含各队列中进程 pid 列表的字典
        """
        snapshot = {
            "ready": {
                p.name: [pcb.pid for pcb in self.ready_queues[p]]
                for p in Priority
            },
            "waiting": [pcb.pid for pcb in self.waiting_queue],
            "blocked": [pcb.pid for pcb in self.blocked_queue],
            "current": self.current.pid if self.current is not None else None,
        }
        return snapshot
