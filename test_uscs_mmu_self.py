"""自测脚本 — 验证 uscs_mmu.py 核心功能 (alloc/free/map/lookup/reclaim)"""
import os
import sys
import shutil
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.uscs_mmu import (
    PageFault,
    PageEntry,
    PageTable,
    PageAllocator,
    PageReclaimer,
    MemoryExhaustedError,
    USCSError,
)

PASS = 0
FAIL = 0


def check(name, condition, detail=""):
    global PASS, FAIL
    if condition:
        PASS += 1
        print(f"  [PASS] {name}")
    else:
        FAIL += 1
        print(f"  [FAIL] {name}  {detail}")


def test_page_table():
    print("\n=== PageTable ===")
    pt = PageTable(pid="test-proc", page_size=4096)

    # map + lookup
    pt.map(0x1000, 0xA000)
    pa, flags = pt.lookup(0x1000)
    check("map + lookup", pa == 0xA000 and flags == 0x7)

    # contains + page_count
    check("contains", pt.contains(0x1000))
    check("page_count", pt.page_count() == 1)
    check("not contains", not pt.contains(0x2000))

    # PageFault on unmapped va
    try:
        pt.lookup(0xDEAD)
        check("PageFault on unmapped", False, "no exception raised")
    except PageFault as e:
        check("PageFault on unmapped", e.va == 0xDEAD and e.access_type == "not_mapped")

    # validate_access
    pt.map(0x2000, 0xB000, flags=0x5)  # R+X, no W
    check("validate r on RWX", pt.validate_access(0x1000, "r"))
    check("validate w on R-X", not pt.validate_access(0x2000, "w"))
    check("validate x on R-X", pt.validate_access(0x2000, "x"))

    # unmap
    pt.unmap(0x1000)
    check("unmap", not pt.contains(0x1000))
    check("page_count after unmap", pt.page_count() == 1)

    # to_dict / from_dict roundtrip
    pt2 = PageTable(pid="rt-proc")
    pt2.map(0x0, 0x100, flags=0x3)
    pt2.map(0x1000, 0x200, flags=0x7)
    d = pt2.to_dict()
    check("to_dict has payload_hash", "payload_hash" in d.get("metadata", {}))
    pt3 = PageTable.from_dict(d)
    check("from_dict pid", pt3.pid == "rt-proc")
    check("from_dict page_count", pt3.page_count() == 2)
    pa3, fl3 = pt3.lookup(0x0)
    check("from_dict entry data", pa3 == 0x100 and fl3 == 0x3)

    # integrity check failure
    d_bad = dict(d)
    d_bad["entries"][0]["pa"] = 0x999  # tamper
    try:
        PageTable.from_dict(d_bad)
        check("integrity check rejects tampered data", False)
    except USCSError:
        check("integrity check rejects tampered data", True)


def test_page_allocator():
    print("\n=== PageAllocator ===")
    alloc = PageAllocator(total_pages=100)

    # basic alloc
    pa1 = alloc.alloc(5)
    check("alloc returns non-negative", pa1 >= 0)
    check("available after alloc", alloc.available() == 95)
    check("used after alloc", alloc.used() == 5)

    # second alloc should be contiguous
    pa2 = alloc.alloc(3)
    check("contiguous alloc", pa2 == pa1 + 5)
    check("total unchanged", alloc.total() == 100)

    # free
    alloc.free(pa1, 5)
    check("available after free", alloc.available() == 97)

    # alloc again from freed region
    pa3 = alloc.alloc(2)
    check("alloc from freed region", pa3 == pa1)

    # MemoryExhaustedError
    alloc2 = PageAllocator(total_pages=3)
    alloc2.alloc(3)
    try:
        alloc2.alloc(1)
        check("MemoryExhaustedError", False)
    except MemoryExhaustedError as e:
        check("MemoryExhaustedError", e.requested == 1 and e.available == 0)

    # read/write page
    alloc3 = PageAllocator(total_pages=10)
    pa = alloc3.alloc(1)
    check("write_page", alloc3.write_page(pa, b"hello"))
    data = alloc3.read_page(pa)
    check("read_page", data is not None and data[:5] == b"hello")

    # allocation order tracking
    order = alloc3.get_allocation_order()
    check("allocation order", len(order) > 0)


def test_page_reclaimer_lru():
    print("\n=== PageReclaimer (LRU) ===")
    swap_dir = os.path.join(os.path.dirname(__file__), "swap_test_lru")
    if os.path.exists(swap_dir):
        shutil.rmtree(swap_dir)

    alloc = PageAllocator(total_pages=10)
    reclaimer = PageReclaimer(policy="lru", swap_dir=swap_dir, allocator=alloc)
    pt = PageTable(pid="lru-proc")

    # 分配并映射 5 个页
    for i in range(5):
        pa = alloc.alloc(1)
        pt.map(i * 0x1000, pa)

    check("before reclaim: used", alloc.used() == 5)
    check("before reclaim: pages", pt.page_count() == 5)

    # 给前两页较新的访问时间
    time.sleep(0.01)
    pt.lookup(0x0000)  # 最近访问
    pt.lookup(0x1000)  # 最近访问

    # 回收 2 页（应换出最久未访问的）
    reclaimed = reclaimer.reclaim(2, pt)
    check("reclaimed count", reclaimed == 2)
    check("after reclaim: used", alloc.used() == 3)
    check("after reclaim: pages", pt.page_count() == 3)

    # 验证 swap 文件存在
    swap_files = [f for f in os.listdir(swap_dir) if f.endswith(".json")]
    check("swap files created", len(swap_files) == 2)

    # page_in
    # 先找出哪个 va 被换出了
    swapped_vas = [int(f.replace(".json", "")) for f in swap_files]
    va_in = swapped_vas[0]
    pa_new = reclaimer.page_in(va_in, pt)
    check("page_in returns pa", pa_new >= 0)
    check("after page_in: pages", pt.page_count() == 4)
    check("after page_in: mapped", pt.contains(va_in))

    # swap 文件应被删除
    swap_files_after = [f for f in os.listdir(swap_dir) if f.endswith(".json")]
    check("swap file removed after page_in", len(swap_files_after) == 1)

    # policy_info
    info = reclaimer.policy_info()
    check("policy_info", info["policy"] == "lru")

    # cleanup
    shutil.rmtree(swap_dir, ignore_errors=True)


def test_page_reclaimer_clock():
    print("\n=== PageReclaimer (Clock) ===")
    swap_dir = os.path.join(os.path.dirname(__file__), "swap_test_clock")
    if os.path.exists(swap_dir):
        shutil.rmtree(swap_dir)

    alloc = PageAllocator(total_pages=10)
    reclaimer = PageReclaimer(policy="clock", swap_dir=swap_dir, allocator=alloc)
    pt = PageTable(pid="clock-proc")

    # 分配并映射 5 个页
    for i in range(5):
        pa = alloc.alloc(1)
        pt.map(i * 0x1000, pa)

    # 手动设置旧的访问时间让它们可被回收
    for i in range(5):
        pt._entries[i * 0x1000].last_access_ts = time.time() - 10

    reclaimed = reclaimer.reclaim(2, pt)
    check("clock reclaimed", reclaimed == 2)
    check("clock after reclaim: pages", pt.page_count() == 3)

    # policy_info
    info = reclaimer.policy_info()
    check("clock policy_info", info["policy"] == "clock")

    # cleanup
    shutil.rmtree(swap_dir, ignore_errors=True)


def test_exceptions():
    print("\n=== Exceptions ===")
    # USCSError hierarchy
    pf = PageFault(0x1000, "p1", "not_mapped")
    check("PageFault is USCSError", isinstance(pf, USCSError))

    me = MemoryExhaustedError(10, 5)
    check("MemoryExhaustedError is USCSError", isinstance(me, USCSError))

    check("PageFault fields", pf.va == 0x1000 and pf.pid == "p1" and pf.access_type == "not_mapped")
    check("MemoryExhaustedError fields", me.requested == 10 and me.available == 5)


if __name__ == "__main__":
    test_page_table()
    test_page_allocator()
    test_page_reclaimer_lru()
    test_page_reclaimer_clock()
    test_exceptions()
    print(f"\n{'='*40}")
    print(f"Results: {PASS} passed, {FAIL} failed")
    if FAIL > 0:
        sys.exit(1)
