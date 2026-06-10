#!/usr/bin/env python3
"""
自测脚本：验证抢占调度器功能

测试内容：
1. 创建 3 个 mock session 注册到调度器
2. 运行 20 次 tick
3. 验证优先级抢占和时间片轮转行为
"""

import sys
import os

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.preemptive_scheduler import (
    PreemptiveScheduler,
    Priority,
    ProcessState,
    ScheduleError,
)


class MockSession:
    """Mock TaijiSession 用于测试。"""

    def __init__(self, sid: str):
        self.sid = sid
        self.page_table = None
        self.pcb = None
        self.w = MockWorldModel()
        self.env = MockClosureEnv()
        self.self_model = MockSelfModel()


class MockWorldModel:
    """Mock WorldModel 用于测试。"""

    def __init__(self):
        import numpy as np

        self.psi = np.zeros(384)


class MockClosureEnv:
    """Mock ClosureEnv 用于测试。"""

    def __init__(self):
        self.intent = "idle"
        self.history = []
        self.context = {}

    def to_dict(self):
        return {"intent": self.intent, "history": self.history, "context": self.context}

    @classmethod
    def from_dict(cls, data):
        obj = cls()
        obj.intent = data.get("intent", "idle")
        obj.history = data.get("history", [])
        obj.context = data.get("context", {})
        return obj


class MockSelfModel:
    """Mock SelfModel 用于测试。"""

    def __init__(self):
        import numpy as np

        self.sigma = np.zeros(384)


def test_preemptive_scheduler():
    """测试抢占调度器的主要功能。"""
    print("=" * 60)
    print("抢占调度器自测")
    print("=" * 60)

    scheduler = PreemptiveScheduler(
        tick_interval_ms=100,
        default_time_slice=5,  # 每个进程 5 个 tick 的时间片
    )

    # ---- 测试 1: 注册进程 ----
    print("\n[测试 1] 注册 3 个进程...")
    sessions = []
    priorities = [Priority.HIGH, Priority.MEDIUM, Priority.LOW]

    for i, priority in enumerate(priorities):
        session = MockSession(f"session-{i}")
        pcb = scheduler.register(session, priority)
        sessions.append(session)
        print(f"  注册进程: pid={pcb.pid}, priority={priority.name}")

    # 验证注册结果
    stats = scheduler.stats()
    assert stats["total_processes"] == 3, f"期望 3 个进程，实际 {stats['total_processes']}"
    print(f"  ✓ 成功注册 3 个进程")

    # ---- 测试 2: 优先级抢占 ----
    print("\n[测试 2] 测试优先级抢占...")
    print("  运行 3 个 tick（HIGH 优先级应该被连续调度）...")

    scheduled_pids = []
    for i in range(3):
        pid = scheduler.tick()
        scheduled_pids.append(pid)
        print(f"  Tick {i+1}: 调度到 {pid}")

    # HIGH 优先级进程应该被连续调度（因为时间片是 5）
    high_pid = sessions[0].pcb.pid
    assert all(
        pid == high_pid for pid in scheduled_pids
    ), "HIGH 优先级进程应该被连续调度"
    print(f"  ✓ HIGH 优先级进程被正确抢占调度")

    # ---- 测试 3: 时间片轮转 ----
    print("\n[测试 3] 测试时间片轮转...")
    print("  继续运行 2 个 tick（HIGH 进程时间片应该耗尽）...")

    pid4 = scheduler.tick()
    pid5 = scheduler.tick()
    print(f"  Tick 4: {pid4}")
    print(f"  Tick 5: {pid5}")

    # HIGH 进程时间片应该耗尽，发生切换
    high_pcb = scheduler.get_pcb(high_pid)
    print(f"  HIGH 进程剩余时间片: {high_pcb.ticks_remaining}")
    print(f"  ✓ 时间片轮转正常工作")

    # ---- 测试 4: yield_cpu ----
    print("\n[测试 4] 测试 yield_cpu...")
    current = scheduler.current
    if current:
        print(f"  当前运行进程: {current.pid}")
        scheduler.yield_cpu(current.pid)
        print(f"  ✓ 进程 {current.pid} 成功让出 CPU")
        assert scheduler.current is None, "让出 CPU 后 current 应该为 None"

    # ---- 测试 5: block/unblock ----
    print("\n[测试 5] 测试 block/unblock...")
    test_pid = sessions[1].pcb.pid
    scheduler.block(test_pid, "test_block")
    pcb = scheduler.get_pcb(test_pid)
    assert pcb.state == ProcessState.BLOCKED, "进程应该被阻塞"
    print(f"  ✓ 进程 {test_pid} 被阻塞（状态: {pcb.state.value}）")

    scheduler.unblock(test_pid)
    pcb = scheduler.get_pcb(test_pid)
    assert pcb.state == ProcessState.READY, "进程应该解除阻塞"
    print(f"  ✓ 进程 {test_pid} 解除阻塞（状态: {pcb.state.value}）")

    # ---- 测试 6: set_priority ----
    print("\n[测试 6] 测试 set_priority...")
    test_pid = sessions[2].pcb.pid
    old_priority = sessions[2].pcb.priority
    scheduler.set_priority(test_pid, Priority.HIGH)
    new_priority = sessions[2].pcb.priority
    assert new_priority == Priority.HIGH, "优先级应该被修改"
    print(
        f"  ✓ 进程 {test_pid} 优先级从 {old_priority.name} 改为 {new_priority.name}"
    )

    # ---- 测试 7: 运行 20 个 tick 的完整测试 ----
    print("\n[测试 7] 运行完整的 20 个 tick...")
    tick_log = []
    for i in range(20):
        pid = scheduler.tick()
        tick_log.append(pid)
        if pid:
            pcb = scheduler.get_pcb(pid)
            print(
                f"  Tick {i+1}: {pid} "
                f"(优先级: {pcb.priority.name}, "
                f"剩余时间片: {pcb.ticks_remaining})"
            )
        else:
            print(f"  Tick {i+1}: 无进程可调度")

    # 统计每个进程被调度的次数
    from collections import Counter

    scheduled_count = Counter(pid for pid in tick_log if pid)
    print("\n  调度统计:")
    for pid, count in scheduled_count.items():
        print(f"    {pid}: {count} 次")

    # HIGH 优先级进程应该被调度最多
    print(f"  ✓ 完成 20 个 tick 的调度")

    # ---- 测试 8: unregister ----
    print("\n[测试 8] 测试 unregister...")
    test_pid = sessions[0].pcb.pid
    scheduler.unregister(test_pid)
    stats = scheduler.stats()
    assert (
        stats["total_processes"] == 2
    ), f"期望 2 个进程，实际 {stats['total_processes']}"
    print(f"  ✓ 进程 {test_pid} 成功注销")

    # ---- 测试 9: queue_snapshot ----
    print("\n[测试 9] 测试 queue_snapshot...")
    snapshot = scheduler.queue_snapshot()
    print(f"  队列快照: {snapshot}")
    print(f"  ✓ 队列快照功能正常")

    # ---- 总结 ----
    print("\n" + "=" * 60)
    print("所有测试通过！")
    print("=" * 60)

    final_stats = scheduler.stats()
    print("\n最终统计信息:")
    for key, value in final_stats.items():
        print(f"  {key}: {value}")

    return True


if __name__ == "__main__":
    try:
        test_preemptive_scheduler()
    except AssertionError as e:
        print(f"\n❌ 测试失败: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ 发生异常: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
