# USCS 内核子系统 — 增量架构设计文档

> **文档版本**: v1.0  
> **所属项目**: taiji-os-core (Taiji OS — FlowForge Core)  
> **基线版本**: v2.3.0 → v2.4.0  
> **作者**: 李架构 (Li)  
> **日期**: 2026-06-10  
> **状态**: ✅ 已完成

---

## CHANGE_LOG

| 版本 | 日期 | 变更内容 |
|------|------|----------|
| v0.1-draft | 2026-06-10 | 初始架构设计，覆盖 USCS 页式内存 / 抢占调度 / 跨节点迁移 |
| v1.0 | 2026-06-10 | 定稿：USCS 内核实现完成，TruthfulQA 外部基准集成，回归测试 5/5 通过 |

---

## 1. 现有架构分析

taiji-os-core v2.3.0 采用四层架构：**HAL**（`hal/llm_router.py`）→ **Core**（`core/`）→ **Syscalls**（`syscalls/`）→ **API**（`api/server.py`）。核心数据流为：用户输入 → `TaijiSession.run()` → `CarbonSiliconGAN.step()` → G-Core（LLM 生成）→ D-Core（SelfConsistencyLoop 矛盾检测 + `PhiScheduler` Φ 门控）→ 更新 WorldModel ψ 或保存 `Continuation` 快照。关键类型关系：`TaijiSession` 持有 `WorldModel`（ψ）、`SelfModel`（σ）、`ClosureEnv`（intent/history）、`CarbonSiliconGAN`（含 `PhiScheduler`）；可选 `MemoryHub` 实现跨会话记忆共享。`Continuation` 以 SHA-256 proof 链保证快照完整性，`MemoryHub` 管理多会话注册与跨会话搜索。当前为单进程模型——一个 `TaijiSession` 实例独占运行，无内存隔离、无调度、无迁移能力。

---

## 2. 增量架构设计

### 2.1 三个子系统类设计

#### 2.1.1 USCS 页式内存管理 — `core/uscs_mmu.py`

```
┌─────────────────────────────────────────────────────────────────┐
│  PageTable                                                      │
│  ─────────                                                      │
│  - pid: str                                                     │
│  - page_size: int = 4096                                        │
│  - entries: dict[int, PageEntry]    # va → PageEntry            │
│  ─────────                                                      │
│  + map(va: int, pa: int, flags: int = 0x7) → None              │
│  + unmap(va: int) → None                                        │
│  + lookup(va: int) → tuple[int, int]     # (pa, flags)         │
│  + to_dict() → dict                                             │
│  + from_dict(data: dict) → PageTable  <<classmethod>>           │
│  + page_count() → int                                           │
│  + contains(va: int) → bool                                     │
│  + validate_access(va: int, access: int) → bool                 │
├─────────────────────────────────────────────────────────────────┤
│  PageEntry                                                      │
│  ─────────                                                      │
│  + va: int                                                      │
│  + pa: int                                                      │
│  + flags: int          # bit0=R bit1=W bit2=X                  │
│  + last_access_ts: float                                       │
│  + ref_count: int = 1       # 共享页引用计数                    │
├─────────────────────────────────────────────────────────────────┤
│  PageAllocator                                                  │
│  ─────────────                                                  │
│  - total_pages: int                                             │
│  - free_set: set[int]         # 空闲物理页集合                  │
│  - allocated: dict[int, int]  # pa → n_pages (分配大小追踪)     │
│  - next_pa: int               # 下一个候选物理地址              │
│  ─────────────                                                  │
│  + alloc(n_pages: int) → int           # 返回起始 pa           │
│  + free(pa: int, n_pages: int) → None                           │
│  + available() → int                                            │
│  + total() → int                                                │
│  + used() → int                                                 │
├─────────────────────────────────────────────────────────────────┤
│  PageReclaimer                                                  │
│  ──────────────                                                 │
│  - policy: str = "lru"         # "lru" | "clock" | "none"      │
│  - swap_dir: str                                                │
│  - clock_hand: int = 0          # Clock 算法指针                │
│  - access_list: OrderedDict     # LRU 访问序追踪                │
│  ──────────────                                                 │
│  + reclaim(n: int, page_table: PageTable) → int                │
│  + page_in(va: int, page_table: PageTable) → int              │
│  + policy_info() → dict                                         │
├─────────────────────────────────────────────────────────────────┤
│  PageFault(Exception)                                           │
│  ─────────────────                                              │
│  + va: int                                                      │
│  + pid: str                                                     │
│  + access_type: str        # "read" | "write" | "exec"         │
└─────────────────────────────────────────────────────────────────┘
```

**复用说明**：
- `PageTable.to_dict()` / `from_dict()` 复用 `Continuation` 的 JSON 序列化模式（`core/continuation.py:109-144`）
- `PageReclaimer` 的 swap 持久化复用 `snapshot_dir` 目录模式（`Continuation._save()` 在 `core/continuation.py:109-125`）
- 页表注册到 `MemoryHub` 的方式复用 `MemoryHub.register()` 模式（`core/memory_hub.py:48-58`）

#### 2.1.2 抢占调度 — `core/preemptive_scheduler.py`

```
┌─────────────────────────────────────────────────────────────────┐
│  ProcessState(Enum)                                             │
│  RUNNING = "running"                                            │
│  READY = "ready"                                                │
│  WAITING = "waiting"                                            │
│  BLOCKED = "blocked"                                            │
├─────────────────────────────────────────────────────────────────┤
│  Priority(Enum)                                                 │
│  HIGH = 0                                                       │
│  MEDIUM = 1                                                     │
│  LOW = 2                                                        │
├─────────────────────────────────────────────────────────────────┤
│  PCB                                                            │
│  ───                                                            │
│  + pid: str                                                     │
│  + priority: Priority                                           │
│  + state: ProcessState                                          │
│  + ticks_remaining: int                                         │
│  + ticks_total: int              # 本轮分配的时间片总量          │
│  + page_table: PageTable                                        │
│  + session: TaijiSession                                        │
│  + cpu_time_ms: int = 0           # 累计 CPU 时间              │
│  + switch_count: int = 0          # 上下文切换次数              │
│  + wait_reason: str = ""          # WAITING/BLOCKED 原因       │
│  + created_at: str                # ISO 8601                   │
├─────────────────────────────────────────────────────────────────┤
│  PreemptiveScheduler                                            │
│  ──────────────────                                              │
│  - tick_interval_ms: int = 100                                  │
│  - ready_queues: dict[Priority, deque[PCB]]                     │
│  - waiting_queue: deque[PCB]                                    │
│  - blocked_queue: deque[PCB]                                    │
│  - current: Optional[PCB]                                       │
│  - pcb_map: dict[str, PCB]         # pid → PCB                 │
│  - _tick_count: int = 0                                         │
│  ──────────────────                                              │
│  + register(session: TaijiSession, priority: Priority) → PCB   │
│  + unregister(pid: str) → None                                  │
│  + tick() → Optional[str]            # 返回被调度到的 pid       │
│  + yield_cpu(pid: str) → None                                   │
│  + block(pid: str, reason: str) → None                          │
│  + unblock(pid: str) → None                                     │
│  + set_priority(pid: str, priority: Priority) → None           │
│  + get_pcb(pid: str) → Optional[PCB]                           │
│  + stats() → dict                                               │
│  + queue_snapshot() → dict         # 各队列快照（调试用）       │
├─────────────────────────────────────────────────────────────────┤
│  ContextSwitch                                                  │
│  ────────────                                                   │
│  + save(session: TaijiSession) → dict         <<staticmethod>> │
│  + restore(session: TaijiSession, snapshot: dict) → None      │
│                    <<staticmethod>>                              │
└─────────────────────────────────────────────────────────────────┘
```

**复用说明**：
- `ContextSwitch.save()` 序列化 ψ 向量的方式复用 `Continuation.__init__` 中 `psi.tolist()` 模式（`core/continuation.py:72`）
- `ContextSwitch.restore()` 恢复 `ClosureEnv` 的方式复用 `ClosureEnv.from_dict()`（`core/closure_env.py:27-31`）
- `ContextSwitch.restore()` 恢复 WorldModel ψ 的方式复用 `TaijiSession.resume()` 中的 `self.w.psi = k.psi`（`core/session.py:194`）
- `PreemptiveScheduler` 与 `PhiScheduler` 互不干扰——`PhiScheduler` 在 GAN 内部做语义门控（`core/phi_scheduler.py:71-95`），`PreemptiveScheduler` 在更高层做进程调度

#### 2.1.3 跨节点迁移 — `core/migration_agent.py`

```
┌─────────────────────────────────────────────────────────────────┐
│  ProcessSnapshot                                                │
│  ──────────────                                                 │
│  + pid: str                                                     │
│  + continuation_kid: str                                        │
│  + continuation_data: dict     # Continuation 完整数据          │
│  + page_table_data: dict       # PageTable.to_dict() 结果      │
│  + pcb_data: dict              # PCB 可序列化视图               │
│  + sigma_data: list[float]     # SelfModel.sigma.tolist()      │
│  + proof: str                  # SHA-256 迁移完整性证明        │
│  + source_node: str                                            │
│  + created_at: str             # ISO 8601                      │
│  ──────────────                                                 │
│  + compute_proof() → str                                        │
│  + verify() → bool                                             │
│  + to_json() → str                                             │
│  + from_json(raw: str) → ProcessSnapshot  <<classmethod>>      │
├─────────────────────────────────────────────────────────────────┤
│  MigrationManager                                               │
│  ─────────────────                                              │
│  - node_id: str                                                 │
│  - transport: NodeTransport                                     │
│  - scheduler: PreemptiveScheduler                               │
│  - allocator: PageAllocator                                     │
│  - _active_migrations: dict[str, MigrationStatus]              │
│  ─────────────────                                              │
│  + export_process(pid: str) → ProcessSnapshot                  │
│  + import_process(snapshot: ProcessSnapshot) → str            │
│  + migrate(pid: str, target_node: str) → str                  │
│  + verify(source_pid: str, target_pid: str) → bool            │
│  + status(pid: str) → dict                                     │
│  + cancel(pid: str) → bool                                     │
├─────────────────────────────────────────────────────────────────┤
│  MigrationState(Enum)                                           │
│  PREPARING = "preparing"                                        │
│  TRANSFERRING = "transferring"                                  │
│  RESTORING = "restoring"                                        │
│  COMPLETED = "completed"                                       │
│  FAILED = "failed"                                              │
├─────────────────────────────────────────────────────────────────┤
│  MigrationStatus                                                │
│  ──────────────                                                 │
│  + pid: str                                                     │
│  + target_node: str                                             │
│  + state: MigrationState                                        │
│  + started_at: str                                              │
│  + progress_pct: float = 0.0                                    │
│  + error: Optional[str]                                         │
├─────────────────────────────────────────────────────────────────┤
│  LoadBalancer                                                   │
│  ─────────────                                                  │
│  - migration_mgr: MigrationManager                              │
│  - cpu_threshold: float = 0.85                                  │
│  - mem_threshold: float = 0.85                                  │
│  ─────────────                                                  │
│  + check_and_balance() → list[str]                             │
│  + set_threshold(cpu_pct: float, mem_pct: float) → None       │
│  + node_stats() → dict                                         │
└─────────────────────────────────────────────────────────────────┘
```

**复用说明**：
- `ProcessSnapshot.continuation_data` 直接复用 `Continuation` 的序列化格式（`core/continuation.py:109-125`）
- `ProcessSnapshot.verify()` 复用 `Continuation.verify()` 的 SHA-256 proof 验证模式（`core/continuation.py:88-103`）
- `MigrationManager.export_process()` 冻结进程的方式复用 `TaijiSession._save_continuation()` 的保存逻辑（`core/session.py:140-174`）
- `MigrationManager.import_process()` 恢复进程的方式复用 `TaijiSession.resume()` 的恢复逻辑（`core/session.py:191-200`）

---

### 2.2 数据流设计

#### 2.2.1 Tick 循环

```
           ┌──────────────────────────────────────────────┐
           │              Timer / asyncio loop             │
           │         (每 tick_interval_ms 触发一次)       │
           └──────────────────┬───────────────────────────┘
                              │
                              ▼
                   ┌─────────────────────┐
                   │  scheduler.tick()   │
                   └──────────┬──────────┘
                              │
               ┌──────────────┼──────────────┐
               │              │              │
               ▼              ▼              ▼
        ┌──────────┐   ┌──────────┐   ┌──────────┐
        │ 检查当前 │   │ 检查高优 │   │ 检查等待 │
        │ 进程时间 │   │ 先级就绪 │   │ 队列可唤 │
        │ 片是否耗 │   │ 队列是否 │   │ 醒的进程 │
        │ 尽       │   │ 有进程   │   │          │
        └─────┬────┘   └─────┬────┘   └─────┬────┘
              │              │              │
              │    ┌─────────┘              │
              │    │ 需要抢占?              │
              ▼    ▼                        │
        ┌──────────────┐   否              │
        │ 是: 执行     │──────┐            │
        │ ContextSwitch│      │            │
        │ .save(当前)  │      │            │
        └──────┬───────┘      │            │
               │              │            │
               ▼              ▼            ▼
        ┌──────────────────────────────────────┐
        │   选择下一个进程（优先级队列轮转）    │
        │   HIGH → MEDIUM → LOW                │
        └──────────────┬───────────────────────┘
                       │
                       ▼
              ┌──────────────────┐
              │ ContextSwitch    │
              │ .restore(下一个) │
              └────────┬─────────┘
                       │
                       ▼
              ┌──────────────────┐
              │ 更新 PCB 状态    │
              │ 记录 cpu_time    │
              │ Auditor.log()    │
              └──────────────────┘
```

#### 2.2.2 内存分配路径

```
  TaijiSession.__init__()
         │
         ▼
  PageAllocator.alloc(n_pages)
         │
         ├─ 空闲页足够? ──→ 返回起始物理地址 pa
         │       │
         │       ▼ 否
         │  PageReclaimer.reclaim(n, page_table)
         │       │
         │       ├─ LRU/Clock 选出冷页
         │       ├─ 写冷页数据到 swap_dir/{va}.json
         │       ├─ PageTable.unmap(va)
         │       └─ PageAllocator.free(pa, 1)
         │              │
         │              ▼
         │       重新 alloc(n_pages) → pa
         │
         ▼
  PageTable.map(va, pa, flags)
         │
         ├─ 写入 entries[va] = PageEntry(va, pa, flags)
         └─ 更新 last_access_ts

  ────────── 页面访问路径 ──────────

  session.w.psi 访问 (或任何语义记忆操作)
         │
         ▼
  PageTable.lookup(va)
         │
         ├─ va 在 entries 中? ──→ 检查 flags 权限
         │       │                     │
         │       │              权限足够? ──→ 返回 (pa, flags)
         │       │                     │
         │       │              权限不足? ──→ 抛出 PageFault(va, "permission")
         │       │
         │       ▼ 否 (页缺失)
         │  PageReclaimer.page_in(va, page_table)
         │       │
         │       ├─ 从 swap_dir/{va}.json 读取
         │       ├─ PageAllocator.alloc(1) → 新 pa
         │       └─ PageTable.map(va, 新 pa, flags)
         │              │
         │              ▼
         │       返回 (新 pa, flags)
         │
         ▼
  抛出 PageFault(va, "not_mapped")
```

#### 2.2.3 迁移全链路

```
  源节点                                    目标节点
  ──────                                    ──────

  scheduler.block(pid, "migrating")
         │
         ▼
  MigrationManager.export_process(pid)
    ├─ ContextSwitch.save(session)
    │     ├─ ψ = session.w.psi.tolist()
    │     ├─ σ = session.self_model.sigma.tolist()
    │     └─ env = session.env.to_dict()
    ├─ PageTable.to_dict()
    ├─ PCB → pcb_data dict
    └─ Continuation.load(kid) → continuation_data
         │
         ▼
  ProcessSnapshot(pid, ..., proof)
         │
         ▼
  NodeTransport.send(snapshot)
    ─────────────────────────────→  NodeTransport.recv()
                                           │
                                           ▼
                                   MigrationManager.import_process(snapshot)
                                     ├─ PageAllocator.alloc(n)
                                     ├─ PageTable.from_dict(data)
                                     ├─ TaijiSession(sid, llm_router)
                                     ├─ session.w.psi = np.array(ψ)
                                     ├─ session.self_model.sigma = np.array(σ)
                                     ├─ session.env = ClosureEnv.from_dict(env)
                                     ├─ scheduler.register(session, priority)
                                     └─ ProcessSnapshot.verify()
                                           │
                                           ▼
                                   scheduler.unblock(new_pid)
                                           │
                                           ▼
                                   迁移完成, 目标 pid 报告给源节点

  源节点收到确认:
    scheduler.unregister(pid)
    Auditor.log("migration_complete", ...)
```

---

### 2.3 状态机设计

#### 2.3.1 PCB 状态转换图

```
                          register()
                              │
                              ▼
                    ┌─────────────────┐
                    │     READY       │◄──────────────────┐
                    └────────┬────────┘                   │
                             │ tick() 选中               │
                             ▼                            │
                    ┌─────────────────┐                   │
                    │    RUNNING      │──── yield_cpu() ──┘
                    └──┬─────┬────────┘                   │
                       │     │                            │
          时间片耗尽   │     │ block(reason)              │
          tick() 抢占  │     │                            │
                       │     ▼                            │
                       │  ┌─────────────────┐             │
                       │  │    BLOCKED      │── unblock()─┘
                       │  └─────────────────┘
                       │
                       │    block(reason)
                       │        │
                       │        ▼
                       │  ┌─────────────────┐
                       │  │    WAITING      │── wake() ────┘
                       │  └─────────────────┘
                       │        │
                       │  迁移冻结
                       │        │
                       ▼        ▼
               返回 READY 队列

  特殊路径:
    BLOCKED ──迁移──→ BLOCKED (迁移期间保持阻塞)
    WAITING ──超时──→ READY (超时自动唤醒)
    任何状态 ──unregister──→ 终止
```

#### 2.3.2 迁移状态机

```
  ┌───────────┐   export_process()   ┌──────────────┐
  │  IDLE     │────────────────────→│  PREPARING   │
  └───────────┘                      └──────┬───────┘
                                            │
                               序列化完成    │
                                            ▼
                                     ┌──────────────┐
                                     │ TRANSFERRING │
                                     └──────┬───────┘
                                            │
                               传输完成      │
                                            ▼
                                     ┌──────────────┐
                                     │  RESTORING   │
                                     └──────┬───────┘
                                       ┌────┴────┐
                              验证通过  │         │  验证失败/异常
                                       ▼         ▼
                               ┌──────────┐  ┌──────────┐
                               │COMPLETED │  │  FAILED  │
                               └──────────┘  └────┬─────┘
                                                  │
                                        cancel()  │ 重试 ──→ PREPARING
                                                  │
                                                  ▼
                                              IDLE
```

---

## 3. 集成策略

### 3.1 最小化修改方案

| 文件 | 修改类型 | 具体变更 | 影响函数 |
|------|----------|----------|----------|
| `core/session.py` | 修改 | `__init__` 新增 `page_table`、`pcb` 可选参数；新增 `_scheduler` 引用 | `__init__`, `run()` |
| `core/continuation.py` | 扩展 | 新增 `migration_meta` 可选字段（迁移来源节点、时间戳） | `_save()`, `load()` |
| `core/memory_hub.py` | 扩展 | 新增 `register_page_table()` 方法，支持页表注册与跨会话页共享 | 新增方法 |
| `syscalls/auditor.py` | 扩展 | 新增事件类型：`schedule_tick`, `context_switch`, `page_fault`, `migration_start`, `migration_complete` | `log()` |
| `api/server.py` | 修改 | 新增 6 个 API 端点（见 §5.4）；`kernel` dict 新增 `scheduler`、`migration_mgr` 实例 | 新增路由 |
| `config.yaml` | 修改 | 新增 `uscs`、`scheduler`、`migration` 三个配置段 | 无函数影响 |
| `core/__init__.py` | 修改 | 导出新增模块 | 无函数影响 |

**不修改的模块**：
- `core/phi_scheduler.py` — Φ 门控调度器独立运行，不被抢占调度器替代
- `core/carbon_silicon_gan.py` — GAN 推演逻辑不变
- `core/world_model.py` / `core/self_model.py` / `core/closure_env.py` — 核心数据类型不变
- `core/self_consistency_loop.py` — D-Core 语义检测不变
- `syscalls/planner.py` / `syscalls/executor.py` — 系统调用层不变
- `hal/llm_router.py` — HAL 层不变

### 3.2 TaijiSession 与 PCB/PageTable 绑定

```python
# core/session.py 修改方案

class TaijiSession:
    def __init__(
        self,
        sid: str,
        llm_router,
        snapshot_dir: str = "snapshots",
        mode: str = "text",
        headless: bool = True,
        memory_hub=None,
        # ★ 新增参数 ★
        page_table: Optional["PageTable"] = None,
        pcb: Optional["PCB"] = None,
        scheduler: Optional["PreemptiveScheduler"] = None,
    ):
        # ... 现有逻辑不变 ...
        self.page_table = page_table  # 由 PreemptiveScheduler.register() 创建
        self.pcb = pcb                # 由 PreemptiveScheduler.register() 创建
        self._scheduler = scheduler   # 调度器引用，用于 yield_cpu

    @property
    def pid(self) -> Optional[str]:
        """进程 ID（与 PCB 绑定后才有效）"""
        return self.pcb.pid if self.pcb else None
```

绑定时机：`PreemptiveScheduler.register(session, priority)` 内部创建 `PageTable` 和 `PCB`，然后回填 `session.page_table` 和 `session.pcb`。

### 3.3 调度器接管 session.run()

```python
# 方案：调度器不替换 run()，而是在外层控制调用时机

# api/server.py 修改方案
kernel = {
    "sessions": {},
    "llm": LLMRouter(),
    "scheduler": PreemptiveScheduler(tick_interval_ms=100),  # ★ 新增
}

@app.post("/run")
async def run(req: RunRequest):
    sid = req.sid
    if sid not in kernel["sessions"]:
        sess = TaijiSession(sid, kernel["llm"])
        pcb = kernel["scheduler"].register(sess, Priority.MEDIUM)  # ★ 注册
        kernel["sessions"][sid] = sess
    sess = kernel["sessions"][sid]
    # 调度器管控：仅当进程处于 RUNNING 状态时才执行
    if sess.pcb and sess.pcb.state != ProcessState.RUNNING:
        kernel["scheduler"].unblock(sess.pcb.pid)
    out = sess.run(req.cmd)
    # 执行后主动 yield
    if sess.pcb:
        kernel["scheduler"].yield_cpu(sess.pcb.pid)
    return {"sid": sid, "output": out}
```

**设计决策**：调度器在"外层"控制 `run()` 的调用时机，而非修改 `run()` 内部逻辑。这确保了 `TaijiSession.run()` 的向后兼容性——无调度器时仍可直接调用。

### 3.4 Continuation 扩展以支持迁移

```python
# core/continuation.py 修改方案

# _save() 新增可选字段
def _save(self):
    # ... 现有字段不变 ...
    payload = {
        # ... 现有字段 ...
        # ★ 新增迁移元数据 ★
        "migration_meta": getattr(self, "_migration_meta", None),
    }

# load() 新增解析
@classmethod
def load(cls, kid: str, snapshot_dir: str = "snapshots") -> "Continuation":
    # ... 现有逻辑 ...
    obj._migration_meta = data.get("migration_meta")  # ★ 新增
    return obj
```

迁移时，`MigrationManager.export_process()` 先调用 `session._save_continuation()` 创建一个带 `migration_meta` 的 Continuation，再将完整状态打包为 `ProcessSnapshot`。

---

## 4. 数据模型

### 4.1 配置 Schema

```yaml
# ===== 新增配置段 =====

# USCS 页式内存管理
uscs:
  page_size: 4096                  # 页大小 (bytes)
  total_physical_pages: 1048576   # 物理页总数 (默认 4GB)
  reclaim_policy: "lru"           # "lru" | "clock" | "none"
  swap_dir: "swap"                 # swap 文件目录
  shared_pages_enabled: false      # P1: 是否启用共享页
  huge_page_size: 2097152          # P2: 大页大小 (2MB)

# 抢占调度
scheduler:
  enabled: true                    # 是否启用抢占调度
  tick_interval_ms: 100            # tick 间隔 (毫秒)
  default_priority: "MEDIUM"       # 新进程默认优先级: HIGH | MEDIUM | LOW
  default_time_slice: 10           # 默认时间片 (tick 数)
  max_waiting_timeout_ms: 30000   # 等待队列超时 (毫秒)
  stats_log_interval_ms: 60000    # 调度统计日志间隔

# 跨节点迁移
migration:
  enabled: true                    # 是否启用迁移
  node_id: "node-001"             # 本节点 ID
  transport: "http"               # "http" | "stdio"
  nodes:                          # 集群节点列表 (静态配置)
    - id: "node-001"
      host: "127.0.0.1"
      port: 8000
    - id: "node-002"
      host: "127.0.0.1"
      port: 8001
  auto_balance: false             # P1: 是否启用自动负载均衡
  cpu_threshold: 0.85             # P1: CPU 使用率迁移阈值
  mem_threshold: 0.85             # P1: 内存使用率迁移阈值
  retry_count: 3                  # 迁移重试次数
  retry_delay_ms: 1000            # 迁移重试间隔

# ===== 现有配置段保持不变 =====
# llm, fallback, embedding, taiji, dcore, browser, memory, mcp
```

### 4.2 ProcessSnapshot JSON Schema

```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "title": "ProcessSnapshot",
  "type": "object",
  "required": ["pid", "continuation_kid", "continuation_data", "page_table_data", "pcb_data", "sigma_data", "proof", "source_node", "created_at"],
  "properties": {
    "pid": {
      "type": "string",
      "description": "进程 ID"
    },
    "continuation_kid": {
      "type": "string",
      "description": "关联的 Continuation 快照 ID"
    },
    "continuation_data": {
      "type": "object",
      "description": "Continuation 完整数据，与 continuation.py _save() 格式一致",
      "properties": {
        "kid": {"type": "string"},
        "sid": {"type": "string"},
        "psi": {
          "type": "array",
          "items": {"type": "number"},
          "description": "Base64 编码的 ψ 向量（迁移优化）或原始 float 数组"
        },
        "env": {"type": "object"},
        "reason": {"type": "string"},
        "ts": {"type": "string"},
        "parent_kid": {"type": ["string", "null"]},
        "payload_hash": {"type": "string"},
        "proof": {"type": "string"},
        "migration_meta": {
          "type": ["object", "null"],
          "properties": {
            "source_node": {"type": "string"},
            "migrated_at": {"type": "string"}
          }
        }
      }
    },
    "page_table_data": {
      "type": "object",
      "description": "PageTable.to_dict() 输出",
      "properties": {
        "pid": {"type": "string"},
        "page_size": {"type": "integer"},
        "entries": {
          "type": "array",
          "items": {
            "type": "object",
            "properties": {
              "va": {"type": "integer"},
              "pa": {"type": "integer"},
              "flags": {"type": "integer"},
              "last_access_ts": {"type": "number"}
            }
          }
        }
      }
    },
    "pcb_data": {
      "type": "object",
      "description": "PCB 可序列化视图",
      "properties": {
        "pid": {"type": "string"},
        "priority": {"type": "string", "enum": ["HIGH", "MEDIUM", "LOW"]},
        "state": {"type": "string", "enum": ["running", "ready", "waiting", "blocked"]},
        "ticks_remaining": {"type": "integer"},
        "ticks_total": {"type": "integer"},
        "cpu_time_ms": {"type": "integer"},
        "switch_count": {"type": "integer"},
        "created_at": {"type": "string"}
      }
    },
    "sigma_data": {
      "type": "array",
      "items": {"type": "number"},
      "description": "SelfModel.sigma 向量的 float 列表"
    },
    "proof": {
      "type": "string",
      "description": "SHA-256(ProcessSnapshot 载荷) 迁移完整性证明"
    },
    "source_node": {
      "type": "string",
      "description": "源节点 ID"
    },
    "created_at": {
      "type": "string",
      "format": "date-time",
      "description": "快照创建时间 (ISO 8601)"
    }
  }
}
```

### 4.3 页表序列化格式

```json
{
  "pid": "session-abc123",
  "page_size": 4096,
  "entries": [
    {
      "va": 0,
      "pa": 65536,
      "flags": 7,
      "last_access_ts": 1718012345.678
    },
    {
      "va": 4096,
      "pa": 131072,
      "flags": 5,
      "last_access_ts": 1718012346.123
    }
  ],
  "metadata": {
    "total_pages": 2,
    "export_version": "1.0",
    "source_node": "node-001"
  }
}
```

---

## 5. 线程与并发模型

### 5.1 整体策略

本项目为纯 Python 实现，采用 **asyncio 事件循环 + 可选 threading** 的混合模型：

| 子系统 | 并发策略 | 原因 |
|--------|----------|------|
| PreemptiveScheduler.tick() | asyncio 定时器 (`loop.call_later`) | tick 频率高（100ms），asyncio 足够，避免线程切换开销 |
| PageAllocator | asyncio Lock | 物理页分配是临界区，需互斥 |
| PageReclaimer | asyncio Lock + 文件 I/O 用 `loop.run_in_executor` | swap 文件写入不阻塞事件循环 |
| MigrationManager | asyncio Task | 迁移是长时间操作，用 Task 管理生命周期 |
| NodeTransport (HTTP) | `aiohttp` / `httpx` async | 网络传输天然异步 |
| LoadBalancer | asyncio 定时器 | 周期性检查，非阻塞 |

### 5.2 并发安全要点

```
┌─────────────────────────────────────────────────────┐
│                    asyncio Event Loop                 │
│                                                      │
│  ┌──────────────┐  ┌──────────────┐  ┌────────────┐ │
│  │ Scheduler    │  │ Migration    │  │ API Server │ │
│  │ tick timer   │  │ Manager Task │  │ (FastAPI)  │ │
│  └──────┬───────┘  └──────┬───────┘  └─────┬──────┘ │
│         │                 │                │         │
│         ▼                 ▼                ▼         │
│  ┌──────────────────────────────────────────────┐   │
│  │              asyncio.Lock (全局)              │   │
│  │   保护: PageAllocator + PageTable 修改操作     │   │
│  └──────────────────────────────────────────────┘   │
│                                                      │
│  ┌──────────────────────────────────────────────┐   │
│  │     run_in_executor (ThreadPoolExecutor)      │   │
│  │     用于: swap 文件 I/O, 大向量序列化           │   │
│  └──────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────┘
```

**关键约束**：
- `PageAllocator.alloc()` / `free()` 在 asyncio.Lock 保护下执行，保证物理页分配的原子性
- `PreemptiveScheduler.tick()` 在事件循环主线程中执行，不需要额外锁
- `ContextSwitch.save()` / `restore()` 操作 `session.w.psi` 等 numpy 数组——在单线程事件循环中无竞争
- 迁移期间进程处于 BLOCKED 状态，`tick()` 不会调度该进程，避免 save/restore 竞争

---

## 6. 错误处理策略

### 6.1 异常类型体系

```
USCSError                          # 基础异常
├── PageFault                      # 页缺失 / 权限违规
│   ├── va: int                    # 触发异常的虚拟地址
│   ├── pid: str                   # 进程 ID
│   └── access_type: str           # "not_mapped" | "permission" | "swap_error"
│
├── MigrationError                 # 迁移失败
│   ├── pid: str
│   ├── phase: str                 # "preparing" | "transferring" | "restoring"
│   └── cause: Exception          # 原始异常
│
├── ScheduleError                  # 调度异常
│   ├── pid: str
│   └── reason: str
│
├── MemoryExhaustedError           # 物理内存耗尽
│   ├── requested: int
│   └── available: int
│
└── IntegrityError                 # 完整性校验失败
    ├── kid: str                   # Continuation ID
    └── expected_proof: str
```

### 6.2 各异常处理策略

| 异常 | 触发场景 | 处理方式 |
|------|----------|----------|
| `PageFault(va, pid, "not_mapped")` | 访问未映射的虚拟地址 | 1. 尝试 `PageReclaimer.page_in()` 2. 若 swap 中无此页，记录审计日志并通知进程（通过 `ClosureEnv.set_intent("page_fault")`） |
| `PageFault(va, pid, "permission")` | 对只读页执行写操作 | 记录审计日志，拒绝操作，进程保持 RUNNING（不中断） |
| `PageFault(va, pid, "swap_error")` | swap 文件损坏或丢失 | 记录审计日志，标记进程为 BLOCKED(reason="corrupted_swap")，等待人工干预 |
| `MigrationError(pid, "preparing", e)` | 序列化进程状态失败 | 中止迁移，进程恢复为 READY 状态，审计记录错误 |
| `MigrationError(pid, "transferring", e)` | 网络传输失败 | 重试最多 `migration.retry_count` 次；全部失败后源进程恢复为 READY |
| `MigrationError(pid, "restoring", e)` | 目标节点恢复失败 | 目标节点清理部分恢复的状态，通知源节点，源进程恢复为 READY |
| `IntegrityError` | `ProcessSnapshot.verify()` 或 `Continuation.verify()` 失败 | 迁移标记为 FAILED，进程不恢复运行，审计日志记录 proof 不匹配细节 |
| `MemoryExhaustedError` | `PageAllocator.alloc()` 无空闲页且 `PageReclaimer.reclaim()` 无法回收 | 1. 强制回收低优先级进程的页面 2. 若仍不足，阻塞新分配请求并记录警告 |
| `ScheduleError(pid, "deadlock")` | 所有就绪进程互相等待 | 记录审计日志，强制一个进程超时回到 READY（最少等待者优先） |

---

## 7. 测试策略建议

### 7.1 单元测试

| 测试文件 | 覆盖范围 | 关键测试用例 |
|----------|----------|--------------|
| `tests/test_uscs_mmu.py` | PageTable, PageAllocator, PageReclaimer | 页映射/解除映射；连续分配与释放；PageFault 权限检查；LRU/Clock 回收；页表序列化往返 |
| `tests/test_preemptive.py` | PCB, PreemptiveScheduler, ContextSwitch | 优先级抢占；时间片耗尽切换；yield/block/unblock；上下文保存恢复一致性；空队列 tick |
| `tests/test_migration_agent.py` | ProcessSnapshot, MigrationManager | 快照序列化/反序列化；proof 计算与验证；export → import 往返；迁移中异常处理 |

### 7.2 集成测试

| 场景 | 验证点 |
|------|--------|
| Session 生命周期 + 调度器 | 创建 Session → register → tick 调度 → yield → run → unregister |
| 页分配 + Continuation 保存 | Session 创建时分配页 → run() 产生 Continuation → 页表与快照一致性 |
| 迁移全链路（单节点模拟） | export → serialize → deserialize → import → verify，proof 链不断裂 |
| API 端点集成 | POST /run 自动注册调度器；GET /scheduler/stats 返回正确队列状态；POST /migration/migrate/{pid} 完成迁移 |
| 与现有 MemoryHub 协同 | 页表注册到 Hub；跨会话搜索不影响页表隔离 |

### 7.3 压力测试

| 场景 | 参数 | 预期行为 |
|------|------|----------|
| 大量进程并发 | 100+ TaijiSession 同时注册，HIGH/MEDIUM/LOW 各 1/3 | 调度器在有限 tick 内公平分配 CPU 时间；无死锁 |
| 物理内存不足 | 总页数设为 100，创建 20 个进程各需 10 页 | PageReclaimer 自动回收冷页；swap 正确换入换出；无 OOM |
| 连续迁移 | 短时间内对 10 个进程发起迁移 | 迁移队列正确排序；无 proof 链断裂；目标节点恢复后状态一致 |
| 长时间运行 | 10000 次 tick 循环 + 随机 block/unblock | 无内存泄漏；调度统计正确；审计日志完整 |

---

## 附录 A：新增文件清单

```
taiji-os-core/
├── core/
│   ├── uscs_mmu.py              # ★ 新增：USCS 页式内存管理
│   ├── preemptive_scheduler.py  # ★ 新增：抢占调度器
│   ├── migration_agent.py       # ★ 新增：跨节点迁移代理
│   └── session.py               # 修改：集成 PCB + PageTable
├── hal/
│   └── nic_emu.py               # ★ 新增：网络传输抽象
├── api/
│   └── server.py                # 修改：新增调度/迁移 API 端点
├── syscalls/
│   └── auditor.py               # 修改：新增事件类型
├── tests/
│   ├── test_uscs_mmu.py         # ★ 新增
│   ├── test_preemptive.py       # ★ 新增
│   └── test_migration.py        # ★ 新增
├── config.yaml                  # 修改：新增配置段
└── docs/
    └── arch-uscs-kernel.md      # ★ 本文件
```

## 附录 B：与现有模块的接口映射

| 新模块 | 依赖的现有模块 | 依赖方式 |
|--------|----------------|----------|
| `uscs_mmu.py` | `continuation.py` | 复用 JSON 序列化模式 |
| `uscs_mmu.py` | `memory_hub.py` | `register_page_table()` 新增方法 |
| `preemptive_scheduler.py` | `session.py` | `ContextSwitch` 操作 session 属性 |
| `preemptive_scheduler.py` | `closure_env.py` | `ClosureEnv.from_dict()` 恢复环境 |
| `preemptive_scheduler.py` | `auditor.py` | 记录调度事件 |
| `migration_agent.py` | `continuation.py` | 复用快照序列化 + proof 链 |
| `migration_agent.py` | `uscs_mmu.py` | 页表导出/导入 |
| `migration_agent.py` | `preemptive_scheduler.py` | 冻结/恢复 PCB |
| `migration_agent.py` | `auditor.py` | 记录迁移事件 |
| `nic_emu.py` | `llm_router.py` | 复用 `_load_config()` 配置加载模式 |
| `server.py` | `preemptive_scheduler.py` | 新增 API 路由 |
| `server.py` | `migration_agent.py` | 新增 API 路由 |

## 附录 C：术语对照

| 术语 | 现有实现 | 新增/扩展 |
|------|----------|-----------|
| 调度 | `PhiScheduler` — 语义门控 | `PreemptiveScheduler` — 进程调度（不同层级，互不替代） |
| 快照 | `Continuation` — 单进程推演快照 | `ProcessSnapshot` — 完整进程快照（含页表 + PCB + σ） |
| 内存 | `MemoryHub` — 跨会话记忆共享 | `PageTable` + `PageAllocator` — 进程虚拟内存隔离 |
| 传输 | 无 | `NodeTransport` — 跨节点网络传输抽象 |
| 审计 | `Auditor` — 事件日志 | 扩展事件类型（调度/迁移/页缺失） |
