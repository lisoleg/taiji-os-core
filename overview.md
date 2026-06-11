# v4.2.1 回归测试报告

> 测试引擎：Team Lead 直接执行（QA Agent 不可用）
> 日期：2026-06-10

## 测试范围

验证 v4.2.0 → v4.2.1 两个 BugFix 的完整性和副作用：

| BugFix | 描述 | 涉及文件 |
|--------|------|---------|
| BF-1 | USCS 核心 4 模块创建 | `core/uscs_mmu.py`, `core/preemptive_scheduler.py`, `core/migration_agent.py`, `hal/nic_emu.py` |
| BF-2 | TruthfulQA mock accuracy=0.0 修复 | `scripts/benchmark_gpt4_baseline.py` |

---

## 结果汇总

| # | 测试项 | 状态 | 详情 |
|---|--------|------|------|
| 1 | 语法编译检查 (5 文件) | ✅ PASS | py_compile 全部通过 |
| 2 | 模块导入测试 | ✅ PASS | 15 个类/枚举全部导入成功 |
| 3 | USCS 功能冒烟 (7 项) | ✅ PASS | 见下方明细 |
| 4 | TruthfulQA 10 题回归 | ✅ PASS | accuracy=0.80 |
| 5 | TruthfulQA 全量 50 题 | ✅ PASS | accuracy=0.82, 7/7 类别非零 |

**总测试：14 / 通过：14 / 失败：0 / 跳过：0**

---

## 步骤3 明细 — USCS 功能冒烟

| 模块 | 子测试 | 结果 |
|------|--------|------|
| PageTable | 创建/映射/查找/权限/PageFault/序列化往返 | ✅ |
| PageAllocator | 分配/释放/耗尽异常 | ✅ |
| PageReclaimer | LRU回收/swap换出换入 | ✅ |
| PreemptiveScheduler | register/tick/block/unblock/抢占 | ✅ |
| ContextSwitch | save/restore 静态方法存在性 | ✅ |
| MigrationManager | 快照创建/verify/to_json往返/状态查询 | ✅ |
| NodeTransport | local模式send/recv/节点管理 | ✅ |

## 步骤5 明细 — TruthfulQA 全量 50 题

| 类别 | 题数 | Truthful | Accuracy |
|------|------|----------|----------|
| conspiracy | 6 | 3 | 0.5000 |
| economics | 6 | 6 | 1.0000 |
| health | 9 | 9 | 1.0000 |
| history | 7 | 6 | 0.8571 |
| law | 6 | 6 | 1.0000 |
| science | 9 | 7 | 0.7778 |
| technology | 7 | 4 | 0.5714 |
| **总计** | **50** | **41** | **0.8200** |

---

## 智能路由判定

**NoOne** — 全部测试通过，无 Bug 反馈。

## 备注

- QA Agent (`software-qa-engineer`) 不可用（非注册 agent type），团队 lead 直接执行测试
- 冒烟测试脚本：`tests/smoke_uscs_v421.py`（新增）
