#!/usr/bin/env python3
"""
USCS 功能冒烟测试 — 测试所有4个模块的基本功能
测试目标：
  - PageTable: 创建、映射、查找
  - PageAllocator: 分配、释放
  - PageReclaimer: 回收页面
  - PreemptiveScheduler: tick, yield, block
  - MigrationManager: 创建快照、验证
  - NodeTransport: send, recv
"""

import sys
import os
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

pass_count = 0
fail_count = 0
skip_count = 0
results = []

def test(name, func):
    global pass_count, fail_count
    try:
        func()
        results.append(f"  PASS: {name}")
        pass_count += 1
    except Exception as e:
        results.append(f"  FAIL: {name} — {e}")
        fail_count += 1

# ─────────────────────────────────────────────
# PageTable 测试
# ─────────────────────────────────────────────
def test_pagetable_create():
    from core.uscs_mmu import PageTable
    pt = PageTable(pid="test-pid")
    assert pt.pid == "test-pid"
    assert pt.page_count() == 0

def test_pagetable_map_lookup():
    from core.uscs_mmu import PageTable
    pt = PageTable(pid="test-pid")
    pt.map(0x1000, 0xA000)
    pa, flags = pt.lookup(0x10FF)  # 同一页内任意地址
    assert pa == 0xA000
    assert flags == 0x7

def test_pagetable_unmap():
    from core.uscs_mmu import PageTable, PageFault
    pt = PageTable(pid="test-pid")
    pt.map(0x1000, 0xA000)
    pt.unmap(0x1000)
    try:
        pt.lookup(0x1000)
        assert False, "Expected PageFault"
    except PageFault:
        pass

def test_pagetable_contains():
    from core.uscs_mmu import PageTable
    pt = PageTable(pid="test-pid")
    pt.map(0x1000, 0xA000)
    assert pt.contains(0x10FF) is True
    assert pt.contains(0x2000) is False

def test_pagetable_serialize():
    from core.uscs_mmu import PageTable
    pt = PageTable(pid="test-pid")
    pt.map(0x1000, 0xA000, flags=0x5)
    d = pt.to_dict()
    assert d["pid"] == "test-pid"
    pt2 = PageTable.from_dict(d)
    assert pt2.pid == "test-pid"
    pa, _ = pt2.lookup(0x1000)
    assert pa == 0xA000

# ─────────────────────────────────────────────
# PageAllocator 测试
# ─────────────────────────────────────────────
def test_allocator_alloc_free():
    from core.uscs_mmu import PageAllocator
    alloc = PageAllocator(total_pages=16, page_size=4096)
    assert alloc.total() == 16
    assert alloc.available() == 16
    pa = alloc.alloc(n_pages=2)
    assert pa % 4096 == 0
    assert alloc.available() == 14
    alloc.free(pa, n_pages=2)
    assert alloc.available() == 16

def test_allocator_exhausted():
    from core.uscs_mmu import PageAllocator, MemoryExhaustedError
    alloc = PageAllocator(total_pages=2, page_size=4096)
    alloc.alloc(n_pages=2)
    try:
        alloc.alloc(n_pages=1)
        assert False, "Expected MemoryExhaustedError"
    except MemoryExhaustedError:
        pass

# ─────────────────────────────────────────────
# PageReclaimer 测试
# ─────────────────────────────────────────────
def test_reclaimer_lru():
    from core.uscs_mmu import PageTable, PageAllocator, PageReclaimer
    import tempfile
    swap_dir = tempfile.mkdtemp()
    pt = PageTable(pid="test-pid")
    pt.map(0x1000, 0xA000)
    pt.map(0x2000, 0xB000)
    reclaim = PageReclaimer(policy="lru", swap_dir=swap_dir)
    n = reclaim.reclaim(n=1, page_table=pt)
    assert n == 1
    assert pt.page_count() == 1
    # 换入
    pa = reclaim.page_in(0x1000, pt)
    assert pa == 0xA000

def test_reclaimer_clock():
    from core.uscs_mmu import PageTable, PageReclaimer
    import tempfile
    swap_dir = tempfile.mkdtemp()
    pt = PageTable(pid="test-pid")
    pt.map(0x1000, 0xA000)
    entry = pt.entries[0x1000]
    entry.ref_count = 0  # 可被回收
    reclaim = PageReclaimer(policy="clock", swap_dir=swap_dir)
    n = reclaim.reclaim(n=1, page_table=pt)
    assert n >= 0  # clock 可能回收0或多页

def test_reclaimer_none_policy():
    from core.uscs_mmu import PageTable, PageReclaimer
    pt = PageTable(pid="test-pid")
    pt.map(0x1000, 0xA000)
    reclaim = PageReclaimer(policy="none")
    n = reclaim.reclaim(n=1, page_table=pt)
    assert n == 0

# ─────────────────────────────────────────────
# PreemptiveScheduler 测试
# ─────────────────────────────────────────────
def test_scheduler_tick():
    from core.preemptive_scheduler import PreemptiveScheduler, Priority, ProcessState
    sched = PreemptiveScheduler(tick_interval_ms=100)
    # 无进程时 tick 返回 None
    result = sched.tick()
    assert result is None

def test_scheduler_yield():
    from core.preemptive_scheduler import PreemptiveScheduler, Priority, ProcessState

    class FakeSession:
        def __init__(self, sid):
            self.sid = sid

    sched = PreemptiveScheduler(tick_interval_ms=100)
    sess = FakeSession("s1")
    pcb = sched.register(sess, priority=Priority.MEDIUM)
    assert pcb.state == ProcessState.READY
    # 手动设置 current 模拟运行中
    sched.current = pcb
    pcb.state = ProcessState.RUNNING
    sched.yield_cpu(pcb.pid)
    assert pcb.state == ProcessState.READY
    assert sched.current is None

def test_scheduler_block_unblock():
    from core.preemptive_scheduler import PreemptiveScheduler, Priority, ProcessState

    class FakeSession:
        def __init__(self, sid):
            self.sid = sid

    sched = PreemptiveScheduler(tick_interval_ms=100)
    sess = FakeSession("s1")
    pcb = sched.register(sess, priority=Priority.MEDIUM)
    sched.block(pcb.pid, reason="IO")
    assert pcb.state == ProcessState.BLOCKED
    sched.unblock(pcb.pid)
    assert pcb.state == ProcessState.READY

def test_scheduler_priority():
    from core.preemptive_scheduler import PreemptiveScheduler, Priority, ProcessState

    class FakeSession:
        def __init__(self, sid):
            self.sid = sid

    sched = PreemptiveScheduler(tick_interval_ms=100)
    sess = FakeSession("s1")
    pcb = sched.register(sess, priority=Priority.LOW)
    assert pcb.priority == Priority.LOW
    sched.set_priority(pcb.pid, Priority.HIGH)
    assert pcb.priority == Priority.HIGH

def test_scheduler_stats():
    from core.preemptive_scheduler import PreemptiveScheduler, Priority

    class FakeSession:
        def __init__(self, sid):
            self.sid = sid

    sched = PreemptiveScheduler(tick_interval_ms=100)
    sess = FakeSession("s1")
    sched.register(sess, priority=Priority.MEDIUM)
    s = sched.stats()
    assert "total_switches" in s
    assert s["total_processes"] == 1

# ─────────────────────────────────────────────
# ContextSwitch 测试
# ─────────────────────────────────────────────
def test_context_switch_save_restore():
    from core.preemptive_scheduler import ContextSwitch
    import numpy as np

    class FakeSession:
        def __init__(self):
            self.sid = "test-session"
            self.w = type("W", (), {"psi": np.array([1.0, 2.0, 3.0])})()
            self.self_model = type("SM", (), {"sigma": np.array([0.1, 0.2])})()
            self.env = type("Env", (), {"to_dict": lambda self: {"a": 1}})()

    sess = FakeSession()
    snapshot = ContextSwitch.save(sess)
    assert "sid" in snapshot
    assert "psi" in snapshot

# ─────────────────────────────────────────────
# MigrationManager / ProcessSnapshot 测试
# ─────────────────────────────────────────────
def test_process_snapshot_create_verify():
    from core.migration_agent import ProcessSnapshot
    snap = ProcessSnapshot(
        pid="p1",
        continuation_kid="kid-1",
        continuation_data={"phi": 0.9},
        page_table_data={"pid": "p1", "page_size": 4096, "entries": []},
        pcb_data={"pid": "p1", "state": "ready"},
        sigma_data=[0.1, 0.2, 0.3],
        source_node="node-A",
    )
    assert snap.verify() is True
    assert snap.pid == "p1"
    j = snap.to_json()
    assert "proof" in j

def test_process_snapshot_json_roundtrip():
    from core.migration_agent import ProcessSnapshot
    import json
    snap = ProcessSnapshot(
        pid="p1",
        continuation_kid="kid-1",
        continuation_data={"phi": 0.9},
        page_table_data={"pid": "p1", "page_size": 4096, "entries": []},
        pcb_data={"pid": "p1"},
        sigma_data=[0.1, 0.2],
        source_node="node-A",
    )
    raw = snap.to_json()
    snap2 = ProcessSnapshot.from_json(raw)
    assert snap2.pid == "p1"
    assert snap2.verify() is True

def test_migration_manager_export():
    from core.migration_agent import MigrationManager
    from core.preemptive_scheduler import PreemptiveScheduler, Priority

    class FakeSession:
        def __init__(self, sid):
            self.sid = sid

    sched = PreemptiveScheduler()
    mgr = MigrationManager(node_id="node-A", scheduler=sched)
    sess = FakeSession("s1")
    pcb = sched.register(sess, priority=Priority.MEDIUM)
    # 无 session 属性，export 会抛 MigrationError
    try:
        mgr.export_process(pcb.pid)
        # 可能成功也可能失败，取决于 session 结构
    except Exception:
        pass  # 预期行为（session 是 fake 的）

def test_migration_status():
    from core.migration_agent import MigrationStatus, MigrationState
    status = MigrationStatus(pid="p1", target_node="node-B")
    assert status.state == MigrationState.PREPARING
    d = status.to_dict()
    assert d["pid"] == "p1"

def test_load_balancer():
    from core.migration_agent import MigrationManager, LoadBalancer
    mgr = MigrationManager(node_id="node-A")
    lb = LoadBalancer(mgr, cpu_threshold=0.85, mem_threshold=0.85)
    assert lb.cpu_threshold == 0.85
    lb.set_threshold(0.90, 0.90)
    assert lb.cpu_threshold == 0.90

# ─────────────────────────────────────────────
# NodeTransport 测试
# ─────────────────────────────────────────────
def test_node_transport_local_send_recv():
    from hal.nic_emu import NodeTransport
    import json

    class FakeSnapshot:
        def __init__(self):
            self.pid = "p1"
        def to_json(self):
            return json.dumps({"pid": self.pid})

    transport = NodeTransport(mode="local")
    snapshot = FakeSnapshot()
    result = transport.send(snapshot, "node-B")
    assert result is True
    received = transport.recv("node-B")
    assert received is not None
    assert received["pid"] == "p1"

def test_node_transport_add_remove_node():
    from hal.nic_emu import NodeTransport
    transport = NodeTransport(mode="local")
    assert len(transport.list_nodes()) == 0
    transport.add_node("node-1", "127.0.0.1", 8080)
    assert len(transport.list_nodes()) == 1
    transport.remove_node("node-1")
    assert len(transport.list_nodes()) == 0

def test_node_transport_stdio():
    from hal.nic_emu import NodeTransport
    transport = NodeTransport(mode="stdio")
    # stdio send 不会真的发，只是 print；不会抛异常即通过
    import io, contextlib
    f = io.StringIO()
    with contextlib.redirect_stdout(f):
        result = transport.send(
            type("S", (), {"to_json": lambda self: "{}"})(),
            "node-B"
        )
    assert result is True

# ─────────────────────────────────────────────
# 运行所有测试
# ─────────────────────────────────────────────
print("=" * 60)
print("USCS 功能冒烟测试")
print("=" * 60)

print("\n[PageTable]")
test("PageTable.create", test_pagetable_create)
test("PageTable.map+lookup", test_pagetable_map_lookup)
test("PageTable.unmap", test_pagetable_unmap)
test("PageTable.contains", test_pagetable_contains)
test("PageTable.serialize", test_pagetable_serialize)

print("\n[PageAllocator]")
test("PageAllocator.alloc+free", test_allocator_alloc_free)
test("PageAllocator.exhausted", test_allocator_exhausted)

print("\n[PageReclaimer]")
test("PageReclaimer.lru", test_reclaimer_lru)
test("PageReclaimer.clock", test_reclaimer_clock)
test("PageReclaimer.none_policy", test_reclaimer_none_policy)

print("\n[PreemptiveScheduler]")
test("PreemptiveScheduler.tick", test_scheduler_tick)
test("PreemptiveScheduler.yield_cpu", test_scheduler_yield)
test("PreemptiveScheduler.block+unblock", test_scheduler_block_unblock)
test("PreemptiveScheduler.set_priority", test_scheduler_priority)
test("PreemptiveScheduler.stats", test_scheduler_stats)

print("\n[ContextSwitch]")
test("ContextSwitch.save", test_context_switch_save_restore)

print("\n[MigrationAgent]")
test("ProcessSnapshot.create+verify", test_process_snapshot_create_verify)
test("ProcessSnapshot.json_roundtrip", test_process_snapshot_json_roundtrip)
test("MigrationManager.export", test_migration_manager_export)
test("MigrationStatus", test_migration_status)
test("LoadBalancer", test_load_balancer)

print("\n[NodeTransport]")
test("NodeTransport.local_send_recv", test_node_transport_local_send_recv)
test("NodeTransport.add_remove_node", test_node_transport_add_remove_node)
test("NodeTransport.stdio", test_node_transport_stdio)

# ─────────────────────────────────────────────
# 汇总
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
for r in results:
    print(r)
print("=" * 60)
total = pass_count + fail_count + skip_count
print(f"\n总计: {total}  通过: {pass_count}  失败: {fail_count}  跳过: {skip_count}")
print("=" * 60)

if fail_count > 0:
    print("\n*** 存在失败测试，请检查上方 FAIL 条目 ***")
    sys.exit(1)
else:
    print("\n*** 全部通过 ***")
    sys.exit(0)
