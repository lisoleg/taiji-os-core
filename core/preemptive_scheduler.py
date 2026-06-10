"""core/preemptive_scheduler.py — USCS 抢占调度子系统 (v1.0)

提供：
  - ProcessState / Priority 枚举
  - PCB                  : 进程控制块
  - PreemptiveScheduler  : 优先级抢占调度器
  - ContextSwitch        : 上下文保存/恢复（静态方法）

复用现有模块：
  - ClosureEnv.from_dict()    (core/closure_env.py)
  - TaijiSession resume() 模式 (core/session.py)
  - Continuation 序列化       (core/continuation.py)

与 PhiScheduler 关系：
  PhiScheduler 在 GAN 内部做语义门控，PreemptiveScheduler 在更高层做进程调度，
  二者互不替代。
"""

from __future__ import annotations

import json
import time
from collections import deque
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional


# ──────────────────────────────────────────────────────────────────────────────
# Enums
# ──────────────────────────────────────────────────────────────────────────────


class ProcessState(str, Enum):
    RUNNING = "running"
    READY   = "ready"
    WAITING = "waiting"
    BLOCKED = "blocked"


class Priority(int, Enum):
    HIGH   = 0
    MEDIUM = 1
    LOW    = 2


# ──────────────────────────────────────────────────────────────────────────────
# PCB
# ──────────────────────────────────────────────────────────────────────────────


class PCB:
    """
    进程控制块 (Process Control Block)。

    Attributes:
        pid             : 进程 ID
        priority        : 进程优先级
        state           : 当前状态
        ticks_remaining : 剩余时间片 (tick 数)
        ticks_total     : 本轮分配的时间片总量
        page_table      : 关联的 PageTable（延迟导入避免循环依赖）
        session         : 关联的 TaijiSession
        cpu_time_ms     : 累计 CPU 时间 (ms)
        switch_count    : 上下文切换次数
        wait_reason     : WAITING / BLOCKED 原因
        created_at      : 创建时间 (ISO 8601)
    """

    def __init__(
        self,
        pid: str,
        priority: Priority = Priority.MEDIUM,
        ticks_total: int = 10,
        page_table: Any = None,
        session: Any = None,
    ):
        self.pid = pid
        self.priority = priority
        self.state = ProcessState.READY
        self.ticks_remaining = ticks_total
        self.ticks_total = ticks_total
        self.page_table = page_table
        self.session = session
        self.cpu_time_ms: int = 0
        self.switch_count: int = 0
        self.wait_reason: str = ""
        self.created_at = datetime.now(timezone.utc).isoformat()

    def reset_ticks(self) -> None:
        """重置时间片。"""
        self.ticks_remaining = self.ticks_total

    def consume_tick(self) -> bool:
        """
        消耗一个 time tick。
        
        Returns:
            True 如果时间片耗尽
        """
        self.ticks_remaining -= 1
        return self.ticks_remaining <= 0

    def to_dict(self) -> dict:
        """PCB 可序列化视图。"""
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


# ──────────────────────────────────────────────────────────────────────────────
# PreemptiveScheduler
# ──────────────────────────────────────────────────────────────────────────────


class PreemptiveScheduler:
    """
    优先级抢占调度器。

    按 HIGH → MEDIUM → LOW 顺序轮转就绪队列。
    每 tick_interval_ms 触发一次 tick()，检查是否需要抢占当前进程。

    与 PhiScheduler 关系：
      PhiScheduler 在 CarbonSiliconGAN 内部做语义门控（Φ 值阈值），
      PreemptiveScheduler 在更高层做进程级 CPU 调度，二者互不干扰。
    """

    def __init__(self, tick_interval_ms: int = 100):
        self.tick_interval_ms = tick_interval_ms
        self.ready_queues: dict[Priority, deque[PCB]] = {
            Priority.HIGH:   deque(),
            Priority.MEDIUM: deque(),
            Priority.LOW:    deque(),
        }
        self.waiting_queue: deque[PCB] = deque()
        self.blocked_queue: deque[PCB] = deque()
        self.current: Optional[PCB] = None
        self.pcb_map: dict[str, PCB] = {}
        self._tick_count: int = 0
        self._stats: dict = {
            "total_switches": 0,
            "total_ticks": 0,
            "preemptions": 0,
        }

    # ── 进程生命周期 ──

    def register(self, session: Any, priority: Priority = Priority.MEDIUM) -> PCB:
        """
        注册一个 TaijiSession 为调度进程。

        Args:
            session  : TaijiSession 实例
            priority : 进程优先级
        Returns:
            新创建的 PCB
        """
        pid = getattr(session, "sid", f"proc-{len(self.pcb_map)}")
        pcb = PCB(pid=pid, priority=priority, session=session)
        self.pcb_map[pid] = pcb
        self.ready_queues[priority].append(pcb)
        return pcb

    def unregister(self, pid: str) -> None:
        """注销进程。"""
        pcb = self.pcb_map.pop(pid, None)
        if pcb is None:
            return
        # 从所有队列中移除
        for q in self.ready_queues.values():
            if pcb in q:
                q.remove(pcb)
        if pcb in self.waiting_queue:
            self.waiting_queue.remove(pcb)
        if pcb in self.blocked_queue:
            self.blocked_queue.remove(pcb)
        if self.current and self.current.pid == pid:
            self.current = None

    # ── 调度核心 ──

    def tick(self) -> Optional[str]:
        """
        执行一个调度 tick。

        Returns:
            被调度到的 pid，或 None（无进程可调度）
        """
        self._tick_count += 1
        self._stats["total_ticks"] += 1

        # 1. 检查当前进程时间片是否耗尽
        if self.current and self.current.state == ProcessState.RUNNING:
            if self.current.consume_tick():
                # 时间片耗尽 → 抢占
                self._stats["preemptions"] += 1
                self._enqueue_ready(self.current)
                self.current.state = ProcessState.READY
                self.current = None

        # 2. 检查是否有更高优先级进程就绪
        if self.current:
            next_pcb = self._select_next()
            if next_pcb and next_pcb.priority < self.current.priority:
                # 优先级抢占
                self._stats["preemptions"] += 1
                self._enqueue_ready(self.current)
                self.current.state = ProcessState.READY
                self.current = next_pcb
                self.current.state = ProcessState.RUNNING
                self._stats["total_switches"] += 1
                self.current.switch_count += 1
                return self.current.pid

        # 3. 如果当前无进程运行，选择下一个
        if self.current is None:
            next_pcb = self._select_next()
            if next_pcb:
                self.current = next_pcb
                self.current.state = ProcessState.RUNNING
                self._stats["total_switches"] += 1
                self.current.switch_count += 1
                return self.current.pid

        return None

    def yield_cpu(self, pid: str) -> None:
        """当前进程主动让出 CPU。"""
        pcb = self.pcb_map.get(pid)
        if pcb and pcb.state == ProcessState.RUNNING:
            self._enqueue_ready(pcb)
            pcb.state = ProcessState.READY
            if self.current and self.current.pid == pid:
                self.current = None

    # ── 阻塞 / 唤醒 ──

    def block(self, pid: str, reason: str = "") -> None:
        """阻塞进程（如等待 I/O、迁移期间）。"""
        pcb = self.pcb_map.get(pid)
        if pcb is None:
            return
        pcb.state = ProcessState.BLOCKED
        pcb.wait_reason = reason
        self.blocked_queue.append(pcb)
        if self.current and self.current.pid == pid:
            self.current = None

    def unblock(self, pid: str) -> None:
        """解除进程阻塞，放回就绪队列。"""
        pcb = self.pcb_map.get(pid)
        if pcb is None:
            return
        if pcb in self.blocked_queue:
            self.blocked_queue.remove(pcb)
        elif pcb in self.waiting_queue:
            self.waiting_queue.remove(pcb)
        pcb.state = ProcessState.READY
        pcb.wait_reason = ""
        self._enqueue_ready(pcb)

    # ── 管理接口 ──

    def set_priority(self, pid: str, priority: Priority) -> None:
        """动态修改进程优先级。"""
        pcb = self.pcb_map.get(pid)
        if pcb:
            old_priority = pcb.priority
            pcb.priority = priority
            # 从旧队列移除，加入新队列
            if pcb in self.ready_queues[old_priority]:
                self.ready_queues[old_priority].remove(pcb)
                self._enqueue_ready(pcb)

    def get_pcb(self, pid: str) -> Optional[PCB]:
        """获取进程 PCB。"""
        return self.pcb_map.get(pid)

    def stats(self) -> dict:
        """返回调度统计。"""
        return {
            **self._stats,
            "current_pid": self.current.pid if self.current else None,
            "total_processes": len(self.pcb_map),
            "ready_count": sum(len(q) for q in self.ready_queues.values()),
            "waiting_count": len(self.waiting_queue),
            "blocked_count": len(self.blocked_queue),
        }

    def queue_snapshot(self) -> dict:
        """各队列快照（调试用）。"""
        return {
            "ready": {
                p.name: [pcb.pid for pcb in q]
                for p, q in self.ready_queues.items()
            },
            "waiting": [pcb.pid for pcb in self.waiting_queue],
            "blocked": [pcb.pid for pcb in self.blocked_queue],
            "current": self.current.pid if self.current else None,
        }

    # ── 内部方法 ──

    def _select_next(self) -> Optional[PCB]:
        """从就绪队列按优先级选择下一个进程。"""
        for priority in (Priority.HIGH, Priority.MEDIUM, Priority.LOW):
            q = self.ready_queues[priority]
            if q:
                return q.popleft()
        return None

    def _enqueue_ready(self, pcb: PCB) -> None:
        """将 PCB 放入对应优先级的就绪队列。"""
        pcb.reset_ticks()
        self.ready_queues[pcb.priority].append(pcb)


# ──────────────────────────────────────────────────────────────────────────────
# ContextSwitch
# ──────────────────────────────────────────────────────────────────────────────


class ContextSwitch:
    """
    上下文切换：保存/恢复 TaijiSession 的运行态。

    复用现有模块：
      - ψ 序列化   : Continuation.__init__ 中的 psi.tolist() 模式
      - env 恢复    : ClosureEnv.from_dict()
      - ψ 恢复      : TaijiSession.resume() 中的 self.w.psi = k.psi
    """

    @staticmethod
    def save(session: Any) -> dict:
        """
        保存当前进程上下文。

        Returns:
            包含 ψ、σ、env 的 dict
        """
        snapshot: dict[str, Any] = {}
        snapshot["sid"] = session.sid

        # ψ 向量 (复用 Continuation psi.tolist() 模式)
        if hasattr(session, "w") and session.w and hasattr(session.w, "psi"):
            snapshot["psi"] = session.w.psi.tolist()

        # σ 向量
        if hasattr(session, "self_model") and session.self_model:
            snapshot["sigma"] = session.self_model.sigma.tolist()

        # env 环境 (复用 ClosureEnv.to_dict)
        if hasattr(session, "env") and session.env:
            snapshot["env"] = session.env.to_dict()

        snapshot["ts"] = datetime.now(timezone.utc).isoformat()
        return snapshot

    @staticmethod
    def restore(session: Any, snapshot: dict) -> None:
        """
        恢复进程上下文到 session。

        Args:
            session  : TaijiSession 实例
            snapshot : ContextSwitch.save() 的输出
        """
        import numpy as np
        from core.closure_env import ClosureEnv

        # 恢复 ψ (复用 TaijiSession.resume() 模式)
        if "psi" in snapshot:
            session.w.psi = np.array(snapshot["psi"])

        # 恢复 σ
        if "sigma" in snapshot and hasattr(session, "self_model") and session.self_model:
            session.self_model.sigma = np.array(snapshot["sigma"])

        # 恢复 env (复用 ClosureEnv.from_dict)
        if "env" in snapshot:
            session.env = ClosureEnv.from_dict(snapshot["env"])
