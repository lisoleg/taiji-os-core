# USCS 内核子系统 — 增量需求文档 (PRD)

> **文档版本**: v1.0  
> **所属项目**: taiji-os-core (Taiji OS — FlowForge Core)  
> **当前基线**: v2.3.0  
> **目标版本**: v2.4.0  
> **作者**: 许清楚 (Xu)  
> **日期**: 2026-06-10  
> **状态**: ✅ 已完成

---

## 1. 产品目标

本次增量开发为 taiji-os-core 引入三个内核级子系统，将项目从"单 AGI 进程推演引擎"升级为"具备操作系统内核特征的多进程协作平台"。核心目标如下：

1. **构建虚拟内存抽象层**：以 4KB 页为粒度管理 AGI 进程的语义记忆空间，提供页分配/释放/回收机制，使多进程间的记忆数据获得隔离与保护。
2. **实现多任务抢占调度**：支持多 TaijiSession 并发运行，基于时间片的抢占调度保障高优先级任务的计算资源，避免低优先级任务长期占用。
3. **支持跨节点热迁移**：使 AGI 进程可在不同计算节点间无缝迁移，利用现有 Continuation 快照机制实现序列化、传输、恢复全链路。
4. **兼容现有架构**：三个子系统以增量方式嵌入现有模块结构（core/hal/syscalls/api），不破坏已有 Φ 调度器、碳硅 GAN 推演、Walrus Memory 等核心功能。
5. **为分布式部署奠基**：抢占调度 + 页式内存 + 跨节点迁移三者协同，构成 taiji-os-core 走向多节点分布式 AGI 操作系统的基础设施。

---

## 2. 用户故事

### 2.1 USCS 页式内存管理

| # | 作为... | 我想要... | 以便... |
|---|---------|-----------|---------|
| U1 | AGI 进程开发者 | 创建 TaijiSession 时自动分配隔离的虚拟地址空间，进程间无法越界访问对方的内存 | 多个 AGI 进程可以安全地共享同一宿主机，不会因内存越界导致数据污染或崩溃 |
| U2 | 系统运维工程师 | 在物理内存不足时，系统能自动将冷页面换出到磁盘（swap），并在需要时换入 | 即使承载的 AGI 进程数超过物理内存容量，系统仍可稳定运行而不 OOM |
| U3 | 内核开发者 | 调用统一的页表 API 来查询/修改指定进程的虚拟地址到物理地址映射 | 方便调试内存布局、检查页面权限，以及实现高级内存管理策略（如共享页、写时复制） |

### 2.2 抢占调度

| # | 作为... | 我想要... | 以便... |
|---|---------|-----------|---------|
| U4 | 平台管理员 | 为不同 AGI 进程设置优先级（HIGH/MEDIUM/LOW），高优先级进程能抢占低优先级进程的 CPU 时间 | 关键业务（如实时推理）不会被批处理任务阻塞，保障整体服务质量 |
| U5 | AGI 进程开发者 | 进程被调度器挂起时，其执行上下文（WorldModel ψ、SelfModel σ、ClosureEnv）能被完整保存和恢复 | 调度器切换进程后，AGI 进程的推演状态不发生丢失或错乱 |
| U6 | 系统运维工程师 | 查看当前所有进程的调度队列（就绪/等待/阻塞）和 CPU 时间分配统计 | 及时发现资源争抢热点，辅助容量规划与性能优化 |

### 2.3 跨节点迁移

| # | 作为... | 我想要... | 以便... |
|---|---------|-----------|---------|
| U7 | 平台管理员 | 当某个计算节点负载过高时，将部分 AGI 进程热迁移到空闲节点 | 实现集群负载均衡，避免单点过载导致的响应延迟 |
| U8 | AGI 进程开发者 | 迁移过程中 AGI 进程的 Continuation proof 链不断裂，迁移后一致性校验自动通过 | 确保迁移后的进程状态与源节点完全一致，不会产生"分裂进程" |
| U9 | 系统运维工程师 | 迁移过程对上层 API 调用者透明，迁移期间体验为短暂暂停而非失败 | 迁移操作不会中断用户正在进行的 AGI 推演任务，提升可用性 |

---

## 3. 需求池

### P0 — 必须交付（MVP）

| ID | 子系统 | 需求描述 |
|----|--------|----------|
| M01 | USCS 页式 | 实现页表数据结构 `PageTable`，支持 4KB 页粒度的虚拟地址到物理地址映射 |
| M02 | USCS 页式 | 实现页分配器 `PageAllocator`：`alloc(n_pages)` 返回连续虚拟地址范围 |
| M03 | USCS 页式 | 实现页释放器 `PageFree(free(va, n_pages))`，归还页到空闲池 |
| M04 | USCS 页式 | 实现页权限检查（读/写/执行标志位），非法访问抛出 `PageFault` |
| M05 | USCS 页式 | 与 `MemoryHub` 集成：每个 `TaijiSession` 拥有独立的页表实例 |
| S01 | 抢占调度 | 实现进程控制块 `PCB`：持有 pid、优先级、状态（RUNNING/READY/WAITING/BLOCKED）、时间片剩余 |
| S02 | 抢占调度 | 实现三级优先级队列（HIGH/MEDIUM/LOW），每级独立就绪队列 |
| S03 | 抢占调度 | 实现调度器 `PreemptiveScheduler`：tick 驱动的抢占，每次 tick 检查是否需要上下文切换 |
| S04 | 抢占调度 | 实现上下文保存/恢复 `ContextSwitch.save(pid)` / `.restore(pid)`，封装 TaijiSession 的 ψ、σ、env 序列化 |
| S05 | 抢占调度 | 对接现有 `TaijiSession.run()`，使 session 执行受调度器管控 |
| N01 | 跨节点迁移 | 实现迁移管理器 `MigrationManager`：支持源节点导出、目标节点导入 |
| N02 | 跨节点迁移 | 进程状态网络序列化（JSON + Base64 编码 ψ 向量），基于现有 Continuation 快照格式扩展 |
| N03 | 跨节点迁移 | 迁移后一致性验证：复用现有 `Continuation.verify()` 校验 proof 链 |
| N04 | 跨节点迁移 | 目标节点恢复后自动重建页表和调度上下文 |

### P1 — 重要增强

| ID | 子系统 | 需求描述 |
|----|--------|----------|
| M06 | USCS 页式 | 实现 LRU 页面置换算法：物理内存不足时，换出最久未访问页到 swap 文件 |
| M07 | USCS 页式 | 实现 Clock 替换算法的可选实现，通过 `config.yaml` 的 `uscs.reclaim_policy` 切换 |
| M08 | USCS 页式 | 支持共享页：多个进程映射同一物理页（用于跨进程 Continuation 共享） |
| M09 | USCS 页式 | 页表导出/导入（JSON 格式），支持跨节点迁移时页表随进程迁移 |
| S06 | 抢占调度 | 等待队列 `wait_queue` 和阻塞队列 `blocked_queue` 管理：进程可主动 yield 或被 I/O 阻塞 |
| S07 | 抢占调度 | 调度统计与 API 查询：`scheduler.stats()` 返回每进程 CPU 时间、切换次数、队列长度 |
| S08 | 抢占调度 | 实时调度参数调整：运行时修改进程优先级和时间片长度 |
| N05 | 跨节点迁移 | 自动负载均衡触发迁移：`LoadBalancer` 监控节点 CPU/内存使用率，超阈值自动发起迁移 |
| N06 | 跨节点迁移 | 故障转移：检测到节点心跳超时，自动将故障节点上的进程迁移到健康节点 |
| N07 | 跨节点迁移 | 迁移进度 API：通过 `/migration/{kid}/status` 查询迁移状态（准备中/传输中/恢复中/完成） |

### P2 — 远期规划

| ID | 子系统 | 需求描述 |
|----|--------|----------|
| M10 | USCS 页式 | 写时复制（Copy-on-Write）：fork 子进程时共享父进程页表，写入时才实际复制 |
| M11 | USCS 页式 | 大页支持（2MB/1GB Huge Page），降低大内存 AGI 进程的页表层级开销 |
| S09 | 抢占调度 | 多核亲和性调度：感知多节点 CPU 拓扑，优先在同 NUMA 节点内调度 |
| S10 | 抢占调度 | 组调度（Group Scheduling）：一组关联的 AGI 进程视为调度单元，同时切换 |
| N08 | 跨节点迁移 | 增量迁移：仅传输自上次迁移后的脏页和状态差异，减少网络开销 |
| N09 | 跨节点迁移 | 预拷贝（Pre-Copy）迁移策略：先传输内存页，在迁移窗口期只传输最终脏页 |

---

## 4. 模块交互概览

### 4.1 整体架构图

```
┌─────────────────────────────────────────────────────────────────────┐
│                         api/server.py                               │
│   POST /run   GET /status   POST /resume   GET /migration/{kid}     │
└─────────────────────────────────┬───────────────────────────────────┘
                                  │
┌─────────────────────────────────▼───────────────────────────────────┐
│                      syscalls/ (系统调用层)                          │
│  ┌──────────┐  ┌───────────┐  ┌──────────┐  ┌──────────────────┐  │
│  │ planner  │  │ executor  │  │ auditor  │  │ mcp_bridge       │  │
│  └──────────┘  └───────────┘  └──────────┘  └──────────────────┘  │
└─────────────────────────────────┬───────────────────────────────────┘
                                  │
┌─────────────────────────────────▼───────────────────────────────────┐
│                          core/ (内核层)  ★ 本次增量 ★               │
│                                                                     │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │  ★ uscs_mmu.py (USCS 页式内存管理)                           │  │
│  │  ├── PageTable       : 虚拟→物理地址映射 (4KB 页)              │  │
│  │  ├── PageAllocator   : 页分配 / 释放 / 回收                    │  │
│  │  └── PageReclaimer   : LRU / Clock 页面置换                    │  │
│  └──────────────────┬───────────────────────────────────────────┘  │
│                     │ 页表绑定每进程                                 │
│  ┌──────────────────▼───────────────────────────────────────────┐  │
│  │  ★ preemptive_scheduler.py (抢占调度)                         │  │
│  │  ├── PCB             : 进程控制块 (pid/priority/state/ticks)  │  │
│  │  ├── Scheduler       : tick 驱动抢占 + 多级队列               │  │
│  │  └── ContextSwitch   : 保存/恢复 TaijiSession 完整状态         │  │
│  └──────────────────┬───────────────────────────────────────────┘  │
│                     │ 调度 TaijiSession                              │
│  ┌──────────────────▼───────────────────────────────────────────┐  │
│  │  ★ migration_agent.py (跨节点迁移) ▸ 复用 ◇ 快照机制          │  │
│  │  ├── MigrationManager : 导出→传输→导入→验证                   │  │
│  │  └── LoadBalancer     : 负载感知迁移触发                       │  │
│  └──────────────────────────────────────────────────────────────┘  │
│                                                                     │
│  ┌──────────────┐  ┌───────────────┐  ┌────────────────────────┐   │
│  │ world_model  │  │ self_model    │  │ closure_env            │   │
│  │ (ψ 状态向量) │  │ (σ 自我表示)  │  │ (intent/history)       │   │
│  └──────────────┘  └───────────────┘  └────────────────────────┘   │
│  ┌──────────────────┐  ┌───────────────────────────────────────┐   │
│  │ carbon_silicon_  │  │ continuation.py                       │   │
│  │ gan (Φ 门控 GAN) │  │ (快照序列化 ◇ 证明链 ◇ 持久化)       │   │
│  └──────────────────┘  └───────────────────────────────────────┘   │
│  ┌──────────────────┐  ┌──────────────────┐  ┌────────────────┐   │
│  │ session.py       │  │ phi_scheduler.py │  │ memory_hub.py  │   │
│  │ (TaijiSession)   │  │ (Φ 语义门控)     │  │ (Walrus 记忆)  │   │
│  └──────────────────┘  └──────────────────┘  └────────────────┘   │
└─────────────────────────────────┬───────────────────────────────────┘
                                  │
┌─────────────────────────────────▼───────────────────────────────────┐
│                      hal/ (硬件抽象层)                               │
│  ┌──────────────────┐  ┌───────────────────────────────────────┐   │
│  │ llm_router.py    │  │  ★ hal/nic_emu.py (网络接口模拟)      │   │
│  │ (LLM API 路由)   │  │    用于跨节点迁移的网络传输抽象        │   │
│  └──────────────────┘  └───────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────┘
```

### 4.2 子系统交互关系

```
                    ┌──────────────────┐
                    │  Preemptive      │
                    │  Scheduler       │
                    │  (调度控制面)     │
                    └───┬──────────┬───┘
                调度      │          │  迁移触发
                抢占      │          │
           ┌──────────────▼──┐  ┌────▼──────────────────┐
           │  TaijiSession   │  │  MigrationAgent       │
           │  (AGI 进程)     │  │  (跨节点迁移)          │
           │                 │  │                       │
           │  • 持有 PCB     │  │  • 导出进程完整状态    │
           │  • 持有 PageTable│  │  • 网络传输            │
           │  • 持有 ψ/σ/env │  │  • 目标节点恢复        │
           └──────┬──────────┘  └────┬──────────────────┘
                  │ 页操作            │ 序列化
                  │                   │
    ┌─────────────▼───────────────────▼──────────────────┐
    │           USCS MMU (页式内存管理)                   │
    │                                                     │
    │  PageTable ←→ PageAllocator ←→ PageReclaimer        │
    │       │              │              │               │
    │       └──────────────┼──────────────┘               │
    │                      │                              │
    │              snapshots/ (物理页持久化)               │
    └─────────────────────────────────────────────────────┘
```

### 4.3 与现有模块的关键集成点

| 现有模块 | 集成方式 | 影响范围 |
|----------|----------|----------|
| `core/session.py` (TaijiSession) | 构造函数新增 `page_table` 和 `pcb` 参数；`run()` 方法由调度器统一驱动 | 中等修改 |
| `core/continuation.py` | 复用快照序列化格式；跨节点迁移时 Continuation 作为传输载体 | 新增扩展字段 |
| `core/memory_hub.py` | 页表注册到 MemoryHub，支持跨会话页共享 | 新增方法 |
| `core/phi_scheduler.py` | 保持不变；抢占调度器在更高层运行，不替代 Φ 门控 | 无修改 |
| `syscalls/auditor.py` | 调度事件、迁移事件写入审计日志 | 新增事件类型 |
| `api/server.py` | 新增端点：`/scheduler/stats`、`/migration/{kid}/status` | 新增路由 |
| `config.yaml` | 新增 `uscs`、`scheduler`、`migration` 配置段 | 新增配置 |

---

## 5. 接口边界

### 5.1 USCS 页式内存管理 — `core/uscs_mmu.py`

```python
class PageTable:
    """每进程一个实例，维护虚拟地址→物理地址映射"""

    def __init__(self, pid: str, page_size: int = 4096):
        """初始化空页表"""

    def map(self, va: int, pa: int, flags: int = 0x7) -> None:
        """建立 va → pa 映射，flags 位掩码: bit0=读 bit1=写 bit2=执行"""

    def unmap(self, va: int) -> None:
        """解除 va 映射"""

    def lookup(self, va: int) -> tuple[int, int]:
        """返回 (pa, flags)，页缺失时抛出 PageFault"""

    def to_dict(self) -> dict:
        """导出页表为可序列化字典（迁移用）"""

    @classmethod
    def from_dict(cls, data: dict) -> "PageTable":
        """从字典恢复页表（迁移用）"""


class PageAllocator:
    """全局单例，管理物理页分配与回收"""

    def __init__(self, total_pages: int = 2**20):
        """初始化空闲页池 (默认 4GB 总物理内存)"""

    def alloc(self, n_pages: int) -> int:
        """分配 n 个连续物理页，返回起始物理地址"""

    def free(self, pa: int, n_pages: int) -> None:
        """释放物理页"""

    def available(self) -> int:
        """返回空闲页数"""


class PageReclaimer:
    """页面置换算法：LRU / Clock"""

    def __init__(self, policy: str = "lru"):
        """policy: "lru" | "clock" | "none" """

    def reclaim(self, n: int) -> int:
        """换出 n 个页面到 swap，返回实际换出数量"""

    def page_in(self, va: int) -> int:
        """从 swap 换入指定页面"""
```

### 5.2 抢占调度 — `core/preemptive_scheduler.py`

```python
class ProcessState(Enum):
    RUNNING = "running"
    READY = "ready"
    WAITING = "waiting"
    BLOCKED = "blocked"

class Priority(Enum):
    HIGH = 0
    MEDIUM = 1
    LOW = 2

@dataclass
class PCB:
    """进程控制块"""
    pid: str
    priority: Priority
    state: ProcessState
    ticks_remaining: int        # 剩余时间片
    page_table: PageTable       # 绑定页表
    session: TaijiSession       # 关联的 AGI 进程

class PreemptiveScheduler:
    """tick 驱动的抢占调度器"""

    def __init__(self, tick_interval_ms: int = 100):
        """初始化调度器，tick_interval_ms 为时间片长度"""

    def register(self, session: TaijiSession, priority: Priority) -> PCB:
        """注册 TaijiSession 为可调度进程，返回 PCB"""

    def tick(self) -> Optional[str]:
        """一次调度 tick：检查抢占条件、执行上下文切换、返回切换到的 pid"""

    def yield_cpu(self, pid: str) -> None:
        """进程主动放弃 CPU，进入 READY 队列"""

    def block(self, pid: str, reason: str) -> None:
        """阻塞进程（如等待 I/O），进入 BLOCKED 队列"""

    def unblock(self, pid: str) -> None:
        """解除阻塞，进入 READY 队列"""

    def set_priority(self, pid: str, priority: Priority) -> None:
        """运行时调整优先级"""

    def stats(self) -> dict:
        """返回调度统计：各队列长度、每进程 CPU 时间分布"""


class ContextSwitch:
    """上下文切换：保存/恢复 TaijiSession 完整状态"""

    @staticmethod
    def save(session: TaijiSession) -> dict:
        """保存会话状态为可序列化字典 (ψ, σ, env, 页表引用)"""

    @staticmethod
    def restore(session: TaijiSession, snapshot: dict) -> None:
        """从快照恢复会话状态"""
```

### 5.3 跨节点迁移 — `core/migration_agent.py`

```python
class MigrationManager:
    """跨节点迁移管理器"""

    def __init__(self, node_id: str, transport: "NodeTransport"):
        """初始化迁移管理器"""

    def export_process(self, pid: str) -> "ProcessSnapshot":
        """冻结进程、序列化完整状态（页表 + continuation + PCB），导出为 ProcessSnapshot"""

    def import_process(self, snapshot: "ProcessSnapshot") -> str:
        """从快照恢复进程，重建页表/调度上下文/conty，返回新 pid"""

    def migrate(self, pid: str, target_node: str) -> str:
        """发起迁移：源节点导出 → 网络传输 → 目标节点恢复，返回目标 pid"""

    def verify(self, source_pid: str, target_pid: str) -> bool:
        """迁移后一致性验证"""

    def status(self, pid: str) -> dict:
        """查询迁移进度"""

@dataclass
class ProcessSnapshot:
    """进程完整快照 — 跨节点传输的原子单元"""
    pid: str
    continuation_kid: str       # Continuation ID
    continuation_data: dict     # Continuation 完整数据
    page_table_data: dict       # 页表序列化
    pcb_data: dict              # PCB 序列化
    proof: str                  # SHA-256 迁移完整性证明
    created_at: str             # ISO 8601 时间戳


class LoadBalancer:
    """负载均衡器"""

    def __init__(self, migration_mgr: MigrationManager):
        """绑定迁移管理器"""

    def check_and_balance(self) -> list[str]:
        """检查本节点负载，超过阈值自动发起迁移，返回被迁移的 pid 列表"""

    def set_threshold(self, cpu_pct: float, mem_pct: float) -> None:
        """设置迁移触发阈值"""
```

### 5.4 新增 API 端点

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/scheduler/stats` | 调度器统计信息（队列长度、CPU 分配） |
| GET | `/scheduler/queues` | 各队列进程列表 |
| PUT | `/scheduler/priority/{pid}` | 调整进程优先级 |
| POST | `/migration/migrate/{pid}` | 发起进程迁移（请求体含 target_node） |
| GET | `/migration/{pid}/status` | 查询迁移状态 |
| GET | `/node/stats` | 本节点资源使用统计（CPU/内存/进程数） |

---

## 6. 待确认问题

| # | 问题 | 影响范围 | 建议方向 |
|---|------|----------|----------|
| Q1 | **调度粒度选择**：抢占调度是以 `TaijiSession` 为粒度，还是以更细的 `plan → execute` 步骤为粒度？若以步骤为粒度，需要将现有 `Planner` 输出作为独立任务入队。 | 调度器设计 | 建议先以 Session 为粒度实现 MVP（P0），步骤级调度作为 P1 选项 |
| Q2 | **Physics vs. Simulation**：页式内存管理在纯 Python 中是完全模拟还是需要对接 OS 级 `mmap`？若为模拟，LRU 页面置换的 swap 文件仅用于概念验证，还是需要真实磁盘 I/O？ | USCS MMU 实现策略 | 建议 MVP 阶段使用 Python dict 模拟物理内存 + 文件 swap；后续可选对接 `mmap` |
| Q3 | **节点发现协议**：跨节点迁移需要节点间互相发现和通信。是复用现有的 HTTP API（通过 config 配置对端地址），还是需要引入服务发现机制（如 Consul / Kubernetes label）？ | Migration 网络层 | 建议 MVP 通过 `config.yaml` 静态配置对端节点列表；动态发现纳入 P2 |
| Q4 | **迁移暂停时间目标**：跨节点热迁移的暂停时间窗口允许多长？现有的 `test_migration.py` 目标为 <1s，但那是本地加载。网络传输 + 远程恢复的时间预期需要明确。 | Migration SLA | 需确认——本地恢复 <1s、同集群内 <5s、跨地域 <30s？ |
| Q5 | **与现有 `PhiScheduler` 的命名冲突**：现有 `core/phi_scheduler.py` 是 Φ 语义门控调度器，新增 `core/preemptive_scheduler.py` 是进程调度器。两个 "scheduler" 是否会造成团队混淆？是否需要重命名现有模块？ | 代码可维护性 | 建议保留 `phi_scheduler.py` 不变（已形成文档体系），新增的用 `preemptive_scheduler.py` 明确区分。也可在 P2 将现有 `phi_scheduler` 重命名为 `flow_gate.py` |

---

## 附录 A：文件变更清单（预期）

```
taiji-os-core/
├── core/
│   ├── uscs_mmu.py              # ★ 新增：USCS 页式内存管理
│   ├── preemptive_scheduler.py  # ★ 新增：抢占调度器
│   ├── migration_agent.py       # ★ 新增：跨节点迁移代理
│   └── session.py               #  修改：集成 PCB + PageTable
├── hal/
│   └── nic_emu.py               # ★ 新增：网络传输抽象（迁移用）
├── api/
│   └── server.py                #  修改：新增调度/迁移 API 端点
├── syscalls/
│   └── auditor.py               #  修改：新增事件类型
├── tests/
│   ├── test_uscs_mmu.py         # ★ 新增：页式管理测试
│   ├── test_preemptive.py       # ★ 新增：调度器测试
│   └── test_migration.py        #  修改：扩展跨节点测试用例
├── config.yaml                  #  修改：新增 uscs/scheduler/migration 配置段
└── docs/
    └── prd-uscs-kernel.md       # ★ 本文件
```

---

## 附录 B：术语表

| 术语 | 全称 | 说明 |
|------|------|------|
| USCS | User-Space Cooperative Scheduling | 用户态协同调度框架，本项目的调度理论基础 |
| ψ (psi) | World State Vector | WorldModel 维护的全局语义状态向量 |
| σ (sigma) | Self Representation | SelfModel 维护的 AGI 进程自我表示向量 |
| Φ (phi) | Semantic Consistency | ψ 与候选向量之间的余弦相似度度量 |
| Continuation (k) | 快照/续态 | AGI 进程的可序列化状态快照，支持 SHA-256 proof 链 |
| PCB | Process Control Block | 进程控制块，持有调度所需的进程元数据 |
| MMU | Memory Management Unit | 内存管理单元，本项目中为纯软件实现 |
| Page Fault | 页缺失 | 访问未映射虚拟地址时触发的异常 |
