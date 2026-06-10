"""core/uscs_mmu.py — USCS 页式内存管理子系统 (v1.0)

提供四个核心类：
  - PageEntry       : 页表条目（虚拟地址 → 物理地址映射 + 访问权限 + 引用计数）
  - PageTable       : 进程级页表（支持共享页 ref_count）
  - PageAllocator   : 物理页分配器
  - PageReclaimer   : 页回收器（LRU / Clock 策略 + swap 换入换出）

异常层级：
  - PageFault       : 页缺失或权限违规

复用现有模块：
  - Continuation JSON 序列化模式（core/continuation.py）
  - MemoryHub 注册模式（core/memory_hub.py）
"""

from __future__ import annotations

import json
import os
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Optional


# ──────────────────────────────────────────────────────────────────────────────
# Exception Hierarchy
# ──────────────────────────────────────────────────────────────────────────────


class USCSError(Exception):
    """USCS 子系统基础异常。"""
    pass


class PageFault(USCSError):
    """
    页缺失或权限违规异常。

    Attributes:
        va          : 触发异常的虚拟地址
        pid         : 进程 ID
        access_type : "not_mapped" | "permission" | "swap_error"
    """

    def __init__(self, va: int, pid: str, access_type: str):
        self.va = va
        self.pid = pid
        self.access_type = access_type
        super().__init__(
            f"PageFault: va=0x{va:X}, pid={pid}, type={access_type}"
        )


class MemoryExhaustedError(USCSError):
    """物理内存耗尽异常。"""

    def __init__(self, requested: int, available: int):
        self.requested = requested
        self.available = available
        super().__init__(
            f"MemoryExhausted: requested={requested}, available={available}"
        )


# ──────────────────────────────────────────────────────────────────────────────
# PageEntry
# ──────────────────────────────────────────────────────────────────────────────


@dataclass
class PageEntry:
    """
    页表条目：虚拟地址到物理地址的映射。

    Attributes:
        va              : 虚拟地址 (bytes)
        pa              : 物理地址 (bytes)
        flags           : 访问权限标志位 (bit0=R bit1=W bit2=X)
        last_access_ts  : 最近访问时间戳（LRU 用）
        ref_count       : 共享页引用计数
    """
    va: int
    pa: int
    flags: int = 0x7          # 默认 RWX
    last_access_ts: float = field(default_factory=time.time)
    ref_count: int = 1

    # ── 权限常量 ──
    FLAG_READ  = 0x1
    FLAG_WRITE = 0x2
    FLAG_EXEC  = 0x4

    @property
    def readable(self) -> bool:
        return bool(self.flags & self.FLAG_READ)

    @property
    def writable(self) -> bool:
        return bool(self.flags & self.FLAG_WRITE)

    @property
    def executable(self) -> bool:
        return bool(self.flags & self.FLAG_EXEC)

    def check_access(self, access: int) -> bool:
        """检查 access 权限是否满足。"""
        return (self.flags & access) == access

    def to_dict(self) -> dict:
        return {
            "va": self.va,
            "pa": self.pa,
            "flags": self.flags,
            "last_access_ts": self.last_access_ts,
            "ref_count": self.ref_count,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "PageEntry":
        return cls(
            va=data["va"],
            pa=data["pa"],
            flags=data.get("flags", 0x7),
            last_access_ts=data.get("last_access_ts", time.time()),
            ref_count=data.get("ref_count", 1),
        )


# ──────────────────────────────────────────────────────────────────────────────
# PageTable
# ──────────────────────────────────────────────────────────────────────────────


class PageTable:
    """
    进程级页表：管理进程的虚拟地址空间。

    复用 Continuation JSON 序列化模式：
      - to_dict()   → 可序列化 dict
      - from_dict() → 从 dict 恢复（classmethod）
    """

    def __init__(self, pid: str, page_size: int = 4096):
        self.pid = pid
        self.page_size = page_size
        self.entries: dict[int, PageEntry] = {}  # va → PageEntry

    # ── 映射操作 ──

    def map(self, va: int, pa: int, flags: int = 0x7) -> None:
        """将虚拟地址 va 映射到物理地址 pa。"""
        self.entries[va] = PageEntry(va=va, pa=pa, flags=flags)

    def unmap(self, va: int) -> None:
        """解除虚拟地址 va 的映射。"""
        self.entries.pop(va, None)

    def lookup(self, va: int) -> tuple[int, int]:
        """
        查找虚拟地址 va 对应的物理地址和标志位。

        Returns:
            (pa, flags)

        Raises:
            PageFault: 当 va 未映射时
        """
        va_page = self._page_align(va)
        entry = self.entries.get(va_page)
        if entry is None:
            raise PageFault(va, self.pid, "not_mapped")
        entry.last_access_ts = time.time()
        return entry.pa, entry.flags

    def contains(self, va: int) -> bool:
        """检查虚拟地址 va 是否已映射。"""
        return self._page_align(va) in self.entries

    def validate_access(self, va: int, access: int) -> bool:
        """
        检查对虚拟地址 va 的访问权限是否满足。

        Raises:
            PageFault: 权限不足时
        """
        va_page = self._page_align(va)
        entry = self.entries.get(va_page)
        if entry is None:
            raise PageFault(va, self.pid, "not_mapped")
        if not entry.check_access(access):
            raise PageFault(va, self.pid, "permission")
        return True

    # ── 信息查询 ──

    def page_count(self) -> int:
        """返回已映射的页数。"""
        return len(self.entries)

    # ── 序列化 ──

    def to_dict(self) -> dict:
        """序列化为 dict（复用 Continuation JSON 序列化模式）。"""
        return {
            "pid": self.pid,
            "page_size": self.page_size,
            "entries": [
                entry.to_dict() for entry in self.entries.values()
            ],
            "metadata": {
                "total_pages": self.page_count(),
                "export_version": "1.0",
            },
        }

    @classmethod
    def from_dict(cls, data: dict) -> "PageTable":
        """从 dict 恢复页表。"""
        pt = cls(pid=data["pid"], page_size=data.get("page_size", 4096))
        for entry_data in data.get("entries", []):
            entry = PageEntry.from_dict(entry_data)
            pt.entries[entry.va] = entry
        return pt

    # ── 内部工具 ──

    def _page_align(self, addr: int) -> int:
        return (addr // self.page_size) * self.page_size


# ──────────────────────────────────────────────────────────────────────────────
# PageAllocator
# ──────────────────────────────────────────────────────────────────────────────


class PageAllocator:
    """
    物理页分配器：管理物理内存页的分配与回收。

    Attributes:
        total_pages : 物理页总数
        free_set    : 空闲物理页集合
        allocated   : pa → n_pages（分配大小追踪）
        next_pa     : 下一个候选物理地址
    """

    def __init__(self, total_pages: int, page_size: int = 4096):
        self.total_pages = total_pages
        self.page_size = page_size
        self.free_set: set[int] = set(range(total_pages))
        self.allocated: dict[int, int] = {}
        self.next_pa: int = 0

    def alloc(self, n_pages: int = 1) -> int:
        """
        分配 n_pages 个连续物理页。

        Returns:
            起始物理地址 (pa)

        Raises:
            MemoryExhaustedError: 无足够空闲页
        """
        if len(self.free_set) < n_pages:
            raise MemoryExhaustedError(
                requested=n_pages, available=len(self.free_set)
            )

        # 简单策略：从 next_pa 开始找连续页
        free_list = sorted(self.free_set)
        start_idx = 0
        found = False
        while start_idx + n_pages <= len(free_list):
            if free_list[start_idx + n_pages - 1] == free_list[start_idx] + n_pages - 1:
                found = True
                break
            start_idx += 1

        if not found:
            # 非连续也行，返回第一个空闲页
            pa = free_list[0]
            for i in range(n_pages):
                self.free_set.discard(pa + i)
            self.allocated[pa] = n_pages
            return pa * self.page_size

        pa = free_list[start_idx]
        for i in range(n_pages):
            self.free_set.discard(pa + i)
        self.allocated[pa] = n_pages
        return pa * self.page_size

    def free(self, pa: int, n_pages: int = 1) -> None:
        """释放从 pa 开始的 n_pages 个物理页。"""
        pa_idx = pa // self.page_size
        for i in range(n_pages):
            self.free_set.add(pa_idx + i)
        self.allocated.pop(pa, None)

    def available(self) -> int:
        """返回可用物理页数。"""
        return len(self.free_set)

    def total(self) -> int:
        """返回物理页总数。"""
        return self.total_pages

    def used(self) -> int:
        """返回已分配物理页数。"""
        return self.total_pages - len(self.free_set)


# ──────────────────────────────────────────────────────────────────────────────
# PageReclaimer
# ──────────────────────────────────────────────────────────────────────────────


class PageReclaimer:
    """
    页回收器：当物理内存不足时回收冷页。

    支持策略：
      - "lru"   : 最近最少使用
      - "clock" : 时钟算法
      - "none"  : 不回收（引发 MemoryExhaustedError）

    Attributes:
        policy      : 回收策略
        swap_dir    : swap 文件目录
        clock_hand  : Clock 算法指针
        access_list : LRU 访问序追踪 (OrderedDict: va → last_access_ts)
    """

    def __init__(self, policy: str = "lru", swap_dir: str = "swap"):
        self.policy = policy
        self.swap_dir = swap_dir
        self.clock_hand: int = 0
        self.access_list: OrderedDict = OrderedDict()
        os.makedirs(self.swap_dir, exist_ok=True)

    def reclaim(self, n: int, page_table: PageTable) -> int:
        """
        回收 n 个冷页。

        Returns:
            实际回收的页数
        """
        if self.policy == "none":
            return 0

        if len(page_table.entries) == 0:
            return 0

        reclaimed = 0
        if self.policy == "lru":
            reclaimed = self._reclaim_lru(n, page_table)
        elif self.policy == "clock":
            reclaimed = self._reclaim_clock(n, page_table)

        return reclaimed

    def page_in(self, va: int, page_table: PageTable) -> int:
        """
        从 swap 换入一页。

        Returns:
            该页映射的物理地址 (pa)

        Raises:
            PageFault: swap 文件丢失或损坏
        """
        swap_file = os.path.join(self.swap_dir, f"{page_table.pid}_{va}.json")
        if not os.path.exists(swap_file):
            raise PageFault(va, page_table.pid, "swap_error")

        with open(swap_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        # 换入：标记为映射（pa 由调用方分配）
        page_table.map(va, data["pa"], data.get("flags", 0x7))
        return data["pa"]

    def policy_info(self) -> dict:
        """返回回收策略信息。"""
        return {
            "policy": self.policy,
            "swap_dir": self.swap_dir,
            "tracked_pages": len(self.access_list),
        }

    # ── 内部方法 ──

    def _reclaim_lru(self, n: int, page_table: PageTable) -> int:
        """LRU 回收策略。"""
        # 按 last_access_ts 排序，回收最旧的前 n 个
        sorted_vas = sorted(
            page_table.entries.keys(),
            key=lambda va: page_table.entries[va].last_access_ts,
        )
        reclaimed = 0
        for va in sorted_vas[:n]:
            entry = page_table.entries[va]
            # 写出到 swap
            self._swap_out(va, entry, page_table.pid)
            page_table.unmap(va)
            reclaimed += 1
        return reclaimed

    def _reclaim_clock(self, n: int, page_table: PageTable) -> int:
        """Clock 回收策略。"""
        vas = list(page_table.entries.keys())
        if not vas:
            return 0

        reclaimed = 0
        visited = 0
        while reclaimed < n and visited < len(vas) * 2:
            va = vas[self.clock_hand % len(vas)]
            entry = page_table.entries[va]
            if entry.ref_count <= 0:
                self._swap_out(va, entry, page_table.pid)
                page_table.unmap(va)
                reclaimed += 1
            else:
                entry.ref_count -= 1
            self.clock_hand = (self.clock_hand + 1) % len(vas)
            visited += 1

        return reclaimed

    def _swap_out(self, va: int, entry: PageEntry, pid: str) -> None:
        """将一页写出到 swap 文件。"""
        swap_file = os.path.join(self.swap_dir, f"{pid}_{va}.json")
        with open(swap_file, "w", encoding="utf-8") as f:
            json.dump(entry.to_dict(), f, ensure_ascii=False, indent=2)
