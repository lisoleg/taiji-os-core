"""core/uscs_mmu.py — USCS 页式内存管理模块

提供进程级虚拟内存隔离与物理页池管理，包含：
  - PageTable: 每进程页表，va→pa 映射与访问权限检查
  - PageAllocator: 全局物理页分配器，OrderedDict 追踪分配顺序
  - PageReclaimer: 页面置换（LRU/Clock），swap 文件复用 Continuation 的
    JSON 序列化模式（json.dump/load + SHA-256 校验）
  - 异常体系: USCSError → PageFault / MemoryExhaustedError
"""
from __future__ import annotations

import json
import hashlib
import os
import time
from collections import OrderedDict
from dataclasses import dataclass, field, asdict
from typing import Optional


# ---------------------------------------------------------------------------
# 异常体系
# ---------------------------------------------------------------------------

class USCSError(Exception):
    """USCS 内核基础异常。"""


class PageFault(USCSError):
    """页缺失 / 权限违规异常。

    Attributes:
        va: 触发异常的虚拟地址
        pid: 进程 ID
        access_type: 异常类型 — "not_mapped" | "permission" | "swap_error"
    """

    def __init__(self, va: int, pid: str, access_type: str):
        self.va = va
        self.pid = pid
        self.access_type = access_type
        super().__init__(
            f"PageFault(va=0x{va:X}, pid={pid!r}, access_type={access_type!r})"
        )


class MemoryExhaustedError(USCSError):
    """物理内存耗尽异常。

    Attributes:
        requested: 请求分配的页数
        available: 当前可用页数
    """

    def __init__(self, requested: int, available: int):
        self.requested = requested
        self.available = available
        super().__init__(
            f"MemoryExhaustedError: requested {requested} pages, "
            f"only {available} available"
        )


# ---------------------------------------------------------------------------
# PageEntry — 页表项
# ---------------------------------------------------------------------------

@dataclass
class PageEntry:
    """页表项数据类。

    Attributes:
        va: 虚拟地址
        pa: 物理地址
        flags: 访问标志位 (bit0=R, bit1=W, bit2=X)
        last_access_ts: 最后访问时间戳 (time.time())
        ref_count: 共享页引用计数
    """

    va: int
    pa: int
    flags: int = 0x7
    last_access_ts: float = field(default_factory=time.time)
    ref_count: int = 1


# ---------------------------------------------------------------------------
# PageTable — 每进程页表
# ---------------------------------------------------------------------------

class PageTable:
    """每进程页表，维护 va → PageEntry 映射。

    支持映射、解除映射、查找、权限校验、序列化/反序列化。
    序列化模式复用 Continuation 的 JSON + SHA-256 校验模式。
    """

    def __init__(self, pid: str, page_size: int = 4096):
        """初始化页表。

        Args:
            pid: 所属进程 ID
            page_size: 页大小 (bytes)，默认 4096
        """
        self.pid = pid
        self.page_size = page_size
        self._entries: dict[int, PageEntry] = {}

    # ---- 核心操作 ----

    def map(self, va: int, pa: int, flags: int = 0x7) -> None:
        """建立虚拟地址到物理地址的映射。

        Args:
            va: 虚拟地址
            pa: 物理地址
            flags: 访问标志位 (bit0=R, bit1=W, bit2=X)，默认 0x7 (RWX)
        """
        self._entries[va] = PageEntry(va=va, pa=pa, flags=flags)

    def unmap(self, va: int) -> None:
        """解除虚拟地址映射。若 va 不存在则静默忽略。

        Args:
            va: 要解除映射的虚拟地址
        """
        self._entries.pop(va, None)

    def lookup(self, va: int) -> tuple[int, int]:
        """查找虚拟地址对应的物理地址和标志位。

        Args:
            va: 虚拟地址

        Returns:
            (pa, flags) 元组

        Raises:
            PageFault: 当 va 未映射时，access_type="not_mapped"
        """
        entry = self._entries.get(va)
        if entry is None:
            raise PageFault(va, self.pid, "not_mapped")
        entry.last_access_ts = time.time()
        return (entry.pa, entry.flags)

    def validate_access(self, va: int, access: str) -> bool:
        """检查对指定虚拟地址的访问权限。

        Args:
            va: 虚拟地址
            access: 访问类型 — "r" (读) / "w" (写) / "x" (执行)

        Returns:
            True 表示权限允许，False 表示权限不足

        Raises:
            PageFault: 当 va 未映射时
        """
        _, flags = self.lookup(va)
        bit_map = {"r": 0, "w": 1, "x": 2}
        bit = bit_map.get(access)
        if bit is None:
            return False
        return bool(flags & (1 << bit))

    # ---- 查询 ----

    def page_count(self) -> int:
        """返回已映射的页数。"""
        return len(self._entries)

    def contains(self, va: int) -> bool:
        """检查虚拟地址是否已映射。"""
        return va in self._entries

    def get_entry(self, va: int) -> Optional[PageEntry]:
        """获取页表项（不更新访问时间），若不存在返回 None。"""
        return self._entries.get(va)

    # ---- 序列化（复用 Continuation JSON + SHA-256 模式） ----

    def to_dict(self) -> dict:
        """将页表序列化为字典，包含 SHA-256 校验。

        序列化格式遵循 arch-uscs-kernel.md §4.3 规范，
        复用 Continuation._save() 的 JSON 序列化模式。
        """
        entries_data = []
        for va in sorted(self._entries):
            e = self._entries[va]
            entries_data.append({
                "va": e.va,
                "pa": e.pa,
                "flags": e.flags,
                "last_access_ts": e.last_access_ts,
                "ref_count": e.ref_count,
            })

        payload = {
            "pid": self.pid,
            "page_size": self.page_size,
            "entries": entries_data,
            "metadata": {
                "total_pages": len(entries_data),
                "export_version": "1.0",
            },
        }

        # SHA-256 校验 — 复用 Continuation 的 payload_hash 模式
        raw = json.dumps(payload, ensure_ascii=False, sort_keys=True)
        payload["metadata"]["payload_hash"] = hashlib.sha256(
            raw.encode("utf-8")
        ).hexdigest()
        return payload

    @classmethod
    def from_dict(cls, data: dict) -> "PageTable":
        """从字典反序列化页表，并校验 SHA-256 完整性。

        Args:
            data: to_dict() 输出的字典

        Returns:
            重建的 PageTable 实例

        Raises:
            USCSError: SHA-256 校验失败时
        """
        stored_hash = data.get("metadata", {}).get("payload_hash", "")
        verify_data = {
            k: v for k, v in data.items()
            if k != "metadata"
        }
        verify_data["metadata"] = {
            k: v for k, v in data.get("metadata", {}).items()
            if k != "payload_hash"
        }
        raw = json.dumps(verify_data, ensure_ascii=False, sort_keys=True)
        expected_hash = hashlib.sha256(raw.encode("utf-8")).hexdigest()
        if stored_hash and stored_hash != expected_hash:
            raise USCSError(
                f"PageTable integrity check failed: "
                f"expected {expected_hash[:16]}..., "
                f"got {stored_hash[:16]}..."
            )

        pt = cls(pid=data["pid"], page_size=data.get("page_size", 4096))
        for e_data in data.get("entries", []):
            entry = PageEntry(
                va=e_data["va"],
                pa=e_data["pa"],
                flags=e_data.get("flags", 0x7),
                last_access_ts=e_data.get("last_access_ts", time.time()),
                ref_count=e_data.get("ref_count", 1),
            )
            pt._entries[entry.va] = entry
        return pt

    def __repr__(self) -> str:
        return (
            f"<PageTable pid={self.pid!r} pages={self.page_count()}>"
        )


# ---------------------------------------------------------------------------
# PageAllocator — 全局物理页分配器
# ---------------------------------------------------------------------------

class PageAllocator:
    """全局单例物理页分配器，管理物理页池。

    使用 OrderedDict 追踪分配顺序以支持碎片整理。
    物理内存使用 dict[int, bytearray] 模拟（key=pa, value=4KB bytearray）。
    """

    def __init__(self, total_pages: int, page_size: int = 4096):
        """初始化物理页分配器。

        Args:
            total_pages: 物理页总数
            page_size: 页大小 (bytes)，默认 4096
        """
        self.page_size = page_size
        self._total_pages = total_pages
        # 空闲页集合
        self._free_set: set[int] = set(range(total_pages))
        # pa → n_pages，OrderedDict 保持分配顺序（碎片整理用）
        self._allocated: OrderedDict[int, int] = OrderedDict()
        # 模拟物理内存
        self._physical_memory: dict[int, bytearray] = {}
        # 下一个候选物理页号
        self._next_pa: int = 0

    def alloc(self, n_pages: int) -> int:
        """分配连续物理页。

        Args:
            n_pages: 请求分配的页数

        Returns:
            起始物理页号 (pa)

        Raises:
            MemoryExhaustedError: 物理内存不足时
        """
        if n_pages <= 0:
            raise ValueError("n_pages must be positive")

        # 尝试找到连续的空闲页
        start_pa = self._find_contiguous(n_pages)
        if start_pa is None:
            raise MemoryExhaustedError(n_pages, len(self._free_set))

        # 标记分配
        for i in range(n_pages):
            pa = start_pa + i
            self._free_set.discard(pa)
            self._physical_memory[pa] = bytearray(self.page_size)

        self._allocated[start_pa] = n_pages
        self._next_pa = start_pa + n_pages
        return start_pa

    def free(self, pa: int, n_pages: int) -> None:
        """释放之前分配的物理页。

        Args:
            pa: 起始物理页号
            n_pages: 要释放的页数
        """
        for i in range(n_pages):
            page = pa + i
            self._free_set.add(page)
            self._physical_memory.pop(page, None)
        # 从分配追踪中移除
        self._allocated.pop(pa, None)

    def available(self) -> int:
        """返回可用物理页数。"""
        return len(self._free_set)

    def total(self) -> int:
        """返回物理页总数。"""
        return self._total_pages

    def used(self) -> int:
        """返回已使用物理页数。"""
        return self._total_pages - len(self._free_set)

    def read_page(self, pa: int) -> Optional[bytearray]:
        """读取物理页内容（模拟内存读取）。

        Args:
            pa: 物理页号

        Returns:
            页内容的 bytearray 副本，若未分配返回 None
        """
        data = self._physical_memory.get(pa)
        return bytearray(data) if data is not None else None

    def write_page(self, pa: int, data: bytes, offset: int = 0) -> bool:
        """向物理页写入数据（模拟内存写入）。

        Args:
            pa: 物理页号
            data: 要写入的数据
            offset: 页内偏移量

        Returns:
            True 写入成功，False 页未分配或越界
        """
        page = self._physical_memory.get(pa)
        if page is None:
            return False
        end = offset + len(data)
        if end > self.page_size:
            return False
        page[offset:end] = data
        return True

    def get_allocation_order(self) -> list[tuple[int, int]]:
        """返回分配顺序列表 [(pa, n_pages), ...]，供碎片整理使用。"""
        return list(self._allocated.items())

    def _find_contiguous(self, n_pages: int) -> Optional[int]:
        """在空闲页中寻找 n_pages 个连续页，返回起始 pa。"""
        if len(self._free_set) < n_pages:
            return None

        # 简单策略：从头扫描空闲页
        sorted_free = sorted(self._free_set)
        count = 1
        for i in range(1, len(sorted_free)):
            if sorted_free[i] == sorted_free[i - 1] + 1:
                count += 1
                if count >= n_pages:
                    return sorted_free[i] - n_pages + 1
            else:
                count = 1
        # n_pages == 1 的边界情况
        if n_pages == 1 and sorted_free:
            return sorted_free[0]
        return None

    def __repr__(self) -> str:
        return (
            f"<PageAllocator total={self._total_pages} "
            f"used={self.used()} free={self.available()}>"
        )


# ---------------------------------------------------------------------------
# PageReclaimer — 页面置换算法
# ---------------------------------------------------------------------------

class PageReclaimer:
    """页面置换算法，支持 LRU 和 Clock 两种策略。

    swap 文件持久化复用 Continuation 的 JSON 序列化模式
    （json.dump/load + SHA-256 校验）。
    """

    def __init__(
        self,
        policy: str = "lru",
        swap_dir: str = "swap",
        page_size: int = 4096,
        allocator: Optional[PageAllocator] = None,
    ):
        """初始化页面置换器。

        Args:
            policy: 置换策略 — "lru" 或 "clock"
            swap_dir: swap 文件目录
            page_size: 页大小 (bytes)
            allocator: 物理页分配器引用（page_in 时使用）
        """
        if policy not in ("lru", "clock"):
            raise ValueError(f"Unsupported policy: {policy!r}, use 'lru' or 'clock'")
        self.policy = policy
        self.swap_dir = swap_dir
        self.page_size = page_size
        self._allocator = allocator

        # LRU 访问序追踪：va → last_access_ts
        self._access_list: OrderedDict[int, float] = OrderedDict()
        # Clock 算法指针
        self._clock_hand: int = 0
        self._clock_vas: list[int] = []  # 有序的 va 列表

        os.makedirs(swap_dir, exist_ok=True)

    # ---- 回收 ----

    def reclaim(self, n: int, page_table: PageTable) -> int:
        """从页表中回收 n 个冷页，写入 swap 文件。

        Args:
            n: 期望回收的页数
            page_table: 目标页表

        Returns:
            实际换出的页数
        """
        if self.policy == "lru":
            return self._reclaim_lru(n, page_table)
        elif self.policy == "clock":
            return self._reclaim_clock(n, page_table)
        return 0

    def _reclaim_lru(self, n: int, page_table: PageTable) -> int:
        """LRU 策略：按访问时间从旧到新换出。"""
        # 收集所有页表项并按访问时间排序
        entries = list(page_table._entries.values())
        entries.sort(key=lambda e: e.last_access_ts)

        reclaimed = 0
        for entry in entries:
            if reclaimed >= n:
                break
            if entry.ref_count > 1:
                continue  # 共享页不换出
            # 写入 swap
            self._swap_out(entry, page_table)
            # 解除映射并释放物理页
            page_table.unmap(entry.va)
            if self._allocator:
                self._allocator.free(entry.pa, 1)
            self._access_list.pop(entry.va, None)
            reclaimed += 1
        return reclaimed

    def _reclaim_clock(self, n: int, page_table: PageTable) -> int:
        """Clock 策略：环形扫描，访问位为 0 的页换出。"""
        vas = sorted(page_table._entries.keys())
        if not vas:
            return 0

        reclaimed = 0
        attempts = 0
        max_attempts = len(vas) * 2  # 最多扫两圈

        while reclaimed < n and attempts < max_attempts:
            va = vas[self._clock_hand % len(vas)]
            entry = page_table._entries.get(va)
            if entry is None:
                self._clock_hand = (self._clock_hand + 1) % len(vas)
                attempts += 1
                continue

            # 检查"访问位"——通过比较 last_access_ts 判断近期是否被访问
            now = time.time()
            is_recent = (now - entry.last_access_ts) < 1.0  # 1秒内算"刚访问"

            if is_recent:
                # 跳过，清除访问位（等下次扫描再判断）
                pass
            elif entry.ref_count > 1:
                # 共享页不换出
                pass
            else:
                # 换出
                self._swap_out(entry, page_table)
                page_table.unmap(entry.va)
                if self._allocator:
                    self._allocator.free(entry.pa, 1)
                reclaimed += 1
                # 更新 vas 列表
                vas = sorted(page_table._entries.keys())
                if not vas:
                    break

            self._clock_hand = (self._clock_hand + 1) % max(len(vas), 1)
            attempts += 1

        return reclaimed

    # ---- 换入 ----

    def page_in(self, va: int, page_table: PageTable) -> int:
        """从 swap 文件换入页面。

        Args:
            va: 虚拟地址
            page_table: 目标页表

        Returns:
            新分配的物理地址 (pa)

        Raises:
            PageFault: swap 文件不存在或损坏时，access_type="swap_error"
        """
        swap_path = os.path.join(self.swap_dir, f"{va}.json")
        if not os.path.exists(swap_path):
            raise PageFault(va, page_table.pid, "swap_error")

        try:
            with open(swap_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            raise PageFault(va, page_table.pid, "swap_error") from e

        # SHA-256 校验 — 复用 Continuation 的完整性验证模式
        stored_hash = data.get("payload_hash", "")
        verify_data = {k: v for k, v in data.items() if k != "payload_hash"}
        raw = json.dumps(verify_data, ensure_ascii=False, sort_keys=True)
        expected_hash = hashlib.sha256(raw.encode("utf-8")).hexdigest()
        if stored_hash and stored_hash != expected_hash:
            raise PageFault(va, page_table.pid, "swap_error")

        # 分配新物理页
        if self._allocator is None:
            raise USCSError("PageReclaimer has no allocator reference for page_in")
        pa = self._allocator.alloc(1)

        # 恢复页表映射
        flags = data.get("flags", 0x7)
        page_table.map(va, pa, flags)

        # 恢复物理内存内容（如有）
        page_data_hex = data.get("page_data")
        if page_data_hex and self._allocator:
            page_bytes = bytes.fromhex(page_data_hex)
            self._allocator.write_page(pa, page_bytes)

        # 删除 swap 文件
        try:
            os.remove(swap_path)
        except OSError:
            pass

        return pa

    # ---- Swap I/O ----

    def _swap_out(self, entry: PageEntry, page_table: PageTable) -> None:
        """将页表项写入 swap 文件，复用 Continuation._save() 的 JSON 序列化模式。"""
        # 读取物理页内容
        page_data = None
        if self._allocator:
            raw = self._allocator.read_page(entry.pa)
            if raw is not None:
                page_data = raw.hex()

        payload = {
            "va": entry.va,
            "pa": entry.pa,
            "flags": entry.flags,
            "last_access_ts": entry.last_access_ts,
            "ref_count": entry.ref_count,
            "pid": page_table.pid,
            "page_data": page_data,
        }

        # SHA-256 校验 — 复用 Continuation 的 payload_hash 模式
        raw_json = json.dumps(payload, ensure_ascii=False, sort_keys=True)
        payload["payload_hash"] = hashlib.sha256(
            raw_json.encode("utf-8")
        ).hexdigest()

        swap_path = os.path.join(self.swap_dir, f"{entry.va}.json")
        with open(swap_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

    # ---- 查询 ----

    def policy_info(self) -> dict:
        """返回当前置换策略信息。"""
        return {
            "policy": self.policy,
            "swap_dir": self.swap_dir,
            "clock_hand": self._clock_hand if self.policy == "clock" else None,
            "tracked_pages": len(self._access_list) if self.policy == "lru" else None,
        }

    def __repr__(self) -> str:
        return (
            f"<PageReclaimer policy={self.policy!r} swap_dir={self.swap_dir!r}>"
        )
