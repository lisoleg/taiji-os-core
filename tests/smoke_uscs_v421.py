"""USCS v4.2.1 功能冒烟测试 — 匹配实际 API"""
import sys
sys.path.insert(0, '.')

from core.uscs_mmu import PageTable, PageAllocator, PageReclaimer, PageFault, MemoryExhaustedError
from core.preemptive_scheduler import PreemptiveScheduler, ProcessState, Priority, ContextSwitch, PCB
from core.migration_agent import ProcessSnapshot, MigrationManager, MigrationState, LoadBalancer
from hal.nic_emu import NodeTransport

def test_pagetable():
    """PageTable: 创建、映射、查找"""
    pt = PageTable(pid="proc-1")
    va, pa = 0x1000, 0xA000
    pt.map(va, pa)
    found_pa, flags = pt.lookup(va)
    assert found_pa == pa, f"lookup: expected {pa:#x}, got {found_pa:#x}"
    assert flags == 0x7, f"flags: expected 0x7, got {flags:#x}"
    assert pt.contains(va)
    assert not pt.contains(0xDEAD)
    # 未映射地址应抛 PageFault
    try:
        pt.lookup(0xDEAD)
        assert False, "Expected PageFault"
    except PageFault:
        pass
    # 权限检查 (READ=0x1, WRITE=0x2, EXEC=0x4)
    from core.uscs_mmu import PageEntry
    pt.validate_access(va, PageEntry.FLAG_READ)       # 读权限 OK
    pt.validate_access(va, PageEntry.FLAG_READ | PageEntry.FLAG_WRITE)  # 读写 OK
    # 创建一个只读页测试权限拒绝
    pt.map(0x3000, 0xC000, flags=PageEntry.FLAG_READ)  # 只读
    try:
        pt.validate_access(0x3000, PageEntry.FLAG_WRITE)  # 需要写权限但只有读
        assert False, "Expected PageFault for permission violation"
    except PageFault as e:
        assert e.access_type == "permission"
    # 序列化往返
    d = pt.to_dict()
    pt2 = PageTable.from_dict(d)
    assert pt2.page_count() == pt.page_count()
    print("  PageTable: PASS")

def test_pageallocator():
    """PageAllocator: 分配、释放"""
    alloc = PageAllocator(total_pages=64)
    assert alloc.available() == 64
    pa1 = alloc.alloc(1)
    pa2 = alloc.alloc(2)
    assert alloc.available() == 61
    alloc.free(pa1, 1)
    alloc.free(pa2, 2)
    assert alloc.available() == 64
    # 耗尽后应抛异常
    for _ in range(64):
        alloc.alloc(1)
    assert alloc.available() == 0
    try:
        alloc.alloc(1)
        assert False, "Expected MemoryExhaustedError"
    except MemoryExhaustedError:
        pass
    print("  PageAllocator: PASS")

def test_pagereclaimer():
    """PageReclaimer: LRU 策略回收 + swap"""
    import os
    pt = PageTable(pid="proc-reclaim")
    for i in range(10):
        pt.map(va=i * 0x1000, pa=i * 0x1000)
    assert pt.page_count() == 10

    reclaimer = PageReclaimer(policy="lru", swap_dir="tests/_swap_test")
    reclaimed = reclaimer.reclaim(n=3, page_table=pt)
    assert reclaimed == 3
    assert pt.page_count() == 7
    # 验证 swap 文件存在
    assert os.path.exists("tests/_swap_test")
    # 换入测试
    va = 0x0000  # 第一个被 swap out
    # page_in 需要 swap 文件存在
    # Actually reclaim puts pages into swap, but reclaim va=0 was first to go (oldest)
    # Let's test page_in for the first reclaimed page
    # Which are the 3 oldest? They are 0x0000, 0x1000, 0x2000 (sorted by last_access_ts increasing)
    pa = reclaimer.page_in(va=0x0000, page_table=pt)
    assert pa is not None
    # Cleanup
    import shutil
    shutil.rmtree("tests/_swap_test", ignore_errors=True)
    print("  PageReclaimer: PASS")

def test_scheduler():
    """PreemptiveScheduler: register, tick, block, unblock"""
    # PreemptiveScheduler.register() needs a session-like object
    class MockSession:
        def __init__(self, sid):
            self.sid = sid
    sched = PreemptiveScheduler(tick_interval_ms=10)
    s1 = MockSession("p1")
    s2 = MockSession("p2")
    pcb1 = sched.register(s1, priority=Priority.HIGH)
    pcb2 = sched.register(s2, priority=Priority.LOW)
    assert pcb1.pid == "p1"
    assert pcb2.pid == "p2"
    # tick: 应调度 HIGH 优先级的 p1
    pid = sched.tick()
    assert pid == "p1"
    assert sched.current.pid == "p1"
    # block p1
    sched.block("p1", "waiting for I/O")
    assert pcb1.state == ProcessState.BLOCKED
    # next tick 应调度 p2
    pid = sched.tick()
    assert pid == "p2"
    # unblock p1
    sched.unblock("p1")
    assert pcb1.state == ProcessState.READY
    # 检查统计
    stats = sched.stats()
    assert stats["total_processes"] == 2
    print("  PreemptiveScheduler: PASS")

def test_contextswitch():
    """ContextSwitch: 静态方法存在性"""
    assert hasattr(ContextSwitch, "save")
    assert hasattr(ContextSwitch, "restore")
    print("  ContextSwitch: PASS")

def test_migration():
    """MigrationManager + ProcessSnapshot: 创建、验证、序列化"""
    snapshot = ProcessSnapshot(
        pid="proc-42",
        continuation_kid="cont-001",
        continuation_data={"psi": [0.1, 0.2], "version": 1},
        page_table_data={"pid": "proc-42", "page_size": 4096, "entries": []},
        pcb_data={"pid": "proc-42", "priority": "MEDIUM", "state": "running"},
        sigma_data=[0.5, 0.6, 0.7],
        source_node="node-a",
    )
    assert snapshot.pid == "proc-42"
    assert snapshot.verify()  # 自证明应通过
    # to_json / from_json 往返
    raw = snapshot.to_json()
    snapshot2 = ProcessSnapshot.from_json(raw)
    assert snapshot2.pid == snapshot.pid
    assert snapshot2.proof == snapshot.proof

    # MigrationManager
    mgr = MigrationManager(node_id="node-0")
    status = mgr.status("proc-42")
    assert status["state"] == "unknown"
    assert mgr.cancel("proc-42") is False  # no active migration

    print("  MigrationManager: PASS")

def test_nic():
    """NodeTransport: local 模式 send/recv"""
    transport = NodeTransport(mode="local")
    # Need a ProcessSnapshot to send (transport.send expects a snapshot with to_json())
    snapshot = ProcessSnapshot(
        pid="test-proc",
        continuation_kid="k-1",
        continuation_data={},
        page_table_data={},
        pcb_data={},
        sigma_data=[],
        source_node="node-a",
    )
    ok = transport.send(snapshot, target_node="node-b")
    assert ok
    received = transport.recv(source_node="node-b")
    assert received is not None
    assert received["pid"] == "test-proc"

    # 节点管理
    transport.add_node("node-x", "10.0.0.1", 8080)
    nodes = transport.list_nodes()
    assert any(n["id"] == "node-x" for n in nodes)
    transport.remove_node("node-x")

    print("  NodeTransport: PASS")

if __name__ == "__main__":
    print("=== USCS v4.2.1 Smoke Tests ===")
    test_pagetable()
    test_pageallocator()
    test_pagereclaimer()
    test_scheduler()
    test_contextswitch()
    test_migration()
    test_nic()
    print("ALL USCS SMOKE TESTS PASSED")
