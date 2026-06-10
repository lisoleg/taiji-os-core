# 太极OS (Taiji OS) — FlowForge Core

![Version](https://img.shields.io/badge/version-v4.2.0-blue)
![Tests](https://img.shields.io/badge/tests-65%20passed-green)
![License](https://img.shields.io/badge/license-MIT-orange)
![Python](https://img.shields.io/badge/python-3.10+-blueviolet)

**太极OS**是一个基于**连续性（Continuation）**的AGI Agent运行时（AGI Agent Runtime），集成了**Walrus Memory共享记忆**、**MCP原生桥接**、**硅基代理治理（三旋治理 + 五层次穿透架构）**。

---

## 📦 核心特性

| 特性 | 说明 | 状态 |
|------|------|------|
| **Continuation v2** | AGI进程可序列化快照，含SHA-256 proof链 + parent_kid记忆图谱 | ✅ 生产就绪 |
| **Φ语义连贯性门控** | 余弦相似度门控（static 0.65 / adaptive 滑动窗口），低于阈值触发Continuation | ✅ 生产就绪 |
| **自洽性推演循环** | G-Core(LLM生成) + D-Core(语义矛盾检测 + 关键词回退)双核推演 | ✅ 生产就绪 |
| **Walrus Memory** | 跨会话共享记忆空间，proof链完整性验证，MCP原生支持 | ✅ 生产就绪 |
| **MCP Bridge** | stdio JSON-RPC协议，6个工具暴露，对接Claude Desktop等 | ✅ 生产就绪 |
| **浏览器云脑** | Playwright自动化，WebWorldModel向量化，8种浏览器动作 | ✅ 生产就绪 |
| **硅基代理治理** | 三旋治理(情治/理治/法治) + 五层次穿透架构(L1-L5) | ✅ 生产就绪 |
| **AIC凭证系统** | W3C VC扩展，M175锚定，同伦类哈希H_h，归责真空定理 | ✅ 生产就绪 |
| **GCD归约算子** | Pre/Post约束校验，消除小龙虾死锁 | ✅ 生产就绪 |
| **约柜合约** | M175封印 + M106验收 + M178罚没，模拟区块链智能合约 | ✅ 生产就绪 |
| **确权仪式** | Plan→Consult→Ratify三阶段，消除事后推诿 | ✅ 生产就绪 |
| **OPC注册表** | 人人即法人，责任节点注册，AIC生命周期管理 | ✅ 生产就绪 |
| **USCS 页式内存** | PageTable/PageAllocator/PageReclaimer，4KB页粒度虚拟内存 | ✅ 生产就绪 |
| **抢占调度** | PreemptiveScheduler 多级优先级队列 + ContextSwitch | ✅ 生产就绪 |
| **跨节点迁移** | MigrationManager 进程热迁移 + SHA-256 完整性证明 | ✅ 生产就绪 |
| **TruthfulQA 基准** | 50题7类别外部基准，GPT-4 vs DeepSeek 对比评测 | ✅ 生产就绪 |

---

## 🏗️ 架构全景

```
太极OS v4.1 架构:
┌─────────────────────────────────────────┐
│ L1 流贯 (Ftel)      意图捕获 + φ 度量   │
│ L2 代数壳 (M175)    AIC 锚定 + 归责校验  │
│ L3 拓扑流贯 (GCD)   约束校验 + 执行流    │
│ L4 IDO/ICE           M106验收/M178罚没   │
│ L5 现象渲染          交付物输出          │
├─────────────────────────────────────────┤
│ Walrus Memory        proof链 + 共享记忆   │
│ MCP Bridge           stdio JSON-RPC      │
│ OPC Registry         责任节点 + AIC管理   │
│ RatifyRitual         Plan→Consult→Ratify │
│ SelfConsistencyLoop  G-Core + D-Core     │
│ WorldModel           DeepSeek Embedding  │
│ BrowserExecutor      Playwright 云脑     │
└─────────────────────────────────────────┘
```

---

## 🚀 快速开始

### 环境要求

- Python 3.10+
- DeepSeek API Key（必需，用于Embedding和推理）
- Claude API Key（可选，用于fallback）

### 安装

```bash
git clone https://github.com/lisoleg/taiji-os-core.git
cd taiji-os-core

# 创建虚拟环境（推荐）
python3 -m venv venv
source venv/bin/activate  # Linux/Mac
# 或 venv\Scripts\activate  # Windows

# 安装依赖
pip install -r requirements.txt

# 安装浏览器驱动（浏览器云脑模式需要）
playwright install chromium
```

### 配置

复制 `config.yaml`，修改API Keys：

```bash
# 设置环境变量
export DEEPEEK_API_KEY="sk-..."
export CLAUDE_API_KEY="sk-..."  # 可选
```

### 运行

#### 文本模式（默认）

```bash
# 单次执行
python cli.py --sid alice "设计芯片"

# 交互模式
python cli.py --sid alice
>>> 设计芯片
>>> 继续优化

# 恢复Continuation
python cli.py --continue <kid>
```

#### 浏览器云脑模式

```bash
# 单次执行
python cli.py --web "搜索 太极OS"

# 显示浏览器窗口
python cli.py --web --no-headless "打开 https://github.com"

# 交互模式
python cli.py --web --sid bob
```

#### API服务模式

```bash
uvicorn api.server:app --host 0.0.0.0 --port 8000
```

---

## 📚 文档

| 文档 | 说明 |
|------|------|
| [使用文档](docs/USER_GUIDE.md) | 安装、配置、CLI使用、Python API、故障排除 |
| [技术报告](docs/TECHNICAL_REPORT.md) | 设计与实现技术报告，含摘要、架构、测试验证 |
| [架构文档](docs/ARCHITECTURE.md) | 模块依赖图、数据流、五层次穿透架构详解 |
| [OSDI 论文](docs/osdi_paper_cn.md) | USCS统一语义-计算状态页式管理 (中文论文) |
| [USCS 架构设计](docs/arch-uscs-kernel.md) | USCS 页式内存/抢占调度/跨节点迁移架构 |
| [USCS 需求文档](docs/prd-uscs-kernel.md) | USCS 内核子系统增量需求 |

---

## 🧪 测试

### 运行全量测试

```bash
pytest tests/ -v
# 预期：65 passed, 2 skipped, 0 failed
```

### 运行特定测试

```bash
# Walrus Memory 测试
pytest tests/test_walrus_memory.py -v

# 硅基代理治理测试
pytest tests/test_silicon_governance.py -v

# 浏览器云脑测试
pytest tests/test_web_session.py -v

# HDR（幻觉拦截率）测试
pytest tests/test_hdr.py -v

# SCS（世界一致性）测试
pytest tests/test_scs.py -v

# TruthfulQA 外部基准
python scripts/benchmark_gpt4_baseline.py --mock --sample 50
python scripts/benchmark_compare.py --sample 50
python scripts/benchmark_hdr.py --sample 50
```

---

## 🔧 CLI用法

```bash
# 文本模式交互
python cli.py --sid mySession

# Web模式交互
python cli.py --web --sid mySession

# 单次执行
python cli.py --sid alice "设计芯片"

# 恢复Continuation
python cli.py --sid alice --continue <kid>

# 查看session状态
python cli.py --sid alice --status
```

---

## 🌐 API端点

| 方法 | 路径 | 说明 |
|------|------|------|
| GET  | `/health` | 健康检查 |
| POST | `/run` | 执行指令 |
| GET  | `/session/{sid}/status` | 查看session状态 |
| POST | `/session/{sid}/resume/{kid}` | 恢复Continuation |
| WS   | `/ws` | WebSocket实时通信 |

---

## 🔌 MCP Bridge

太极OS可作为MCP Server运行，对接Claude Desktop等MCP客户端：

```bash
python -m syscalls.mcp_bridge
```

暴露的MCP工具：
- `taiji.run(query)` — 执行一轮推演
- `taiji.status(sid)` — 查询会话状态
- `taiji.resume(kid)` — 从Continuation恢复
- `taiji.memory_search(q)` — 搜索共享记忆
- `taiji.verify(mid)` — 验证记忆完整性
- `taiji.list_sessions()` — 列出已注册会话

---

## 🏛️ 硅基代理治理

### 三旋治理

| 治理维度 | 说明 |
|---------|------|
| **情治 (Consensus)** | 激活主体责任意识，上线前强制主体认领 |
| **理治 (Cryptography)** | 密码学锚定身份契约，强制AIC凭证 + M175封印 |
| **法治 (Statute)** | 行为归责，强制GCD约束，错误时自动M178罚没 |

### 五层次穿透架构

| 层次 | 名称 | 功能 |
|------|------|------|
| L1 | 流贯 (Ftel) | 意图捕获 + φ度量 |
| L2 | 代数壳 (M175) | AIC锚定 + 归责校验 |
| L3 | 拓扑流贯 (GCD) | 约束校验 + 执行流 |
| L4 | IDO/ICE | M106验收/M178罚没 |
| L5 | 现象渲染 | 交付物 + 审计追踪 |

---

## 📁 项目结构

```
taiji-os-core/
├── core/                     # 核心模块
│   ├── session.py           # TaijiSession v4 主控
│   ├── continuation.py      # Continuation v2 (proof链)
│   ├── world_model.py      # WorldModel (DeepSeek Embedding)
│   ├── web_world_model.py  # WebWorldModel (浏览器向量化)
│   ├── self_consistency_loop.py # SelfConsistencyLoop (v4.1 前称 CarbonSiliconGAN)
│   ├── closure_env.py      # ClosureEnv
│   ├── self_model.py       # SelfModel
│   ├── phi_scheduler.py    # Φ-Scheduler
│   ├── memory_hub.py      # Walrus Memory Hub
│   ├── aic.py             # AIC凭证系统
│   ├── gcd_engine.py      # GCD归约算子
│   ├── ark_covenant.py    # 约柜合约
│   ├── tri_spin_governor.py   # 三旋治理
│   ├── ratify_ritual.py   # 确权仪式
│   ├── five_layer_architecture.py  # 五层次穿透架构
│   ├── uscs_mmu.py         # USCS 页式内存管理
│   ├── preemptive_scheduler.py  # 抢占调度器
│   └── migration_agent.py  # 跨节点迁移代理
├── syscalls/               # 系统调用层
│   ├── executor.py         # 文本执行器
│   ├── browser_executor.py # 浏览器执行器
│   ├── planner.py          # 文本规划器
│   ├── web_planner.py      # Web规划器
│   ├── mcp_bridge.py      # MCP Bridge
│   ├── opc_registry.py    # OPC注册表
│   └── auditor.py         # 审计器
├── scripts/                # 评测脚本
│   ├── benchmark_gpt4_baseline.py  # GPT-4 零样本 baseline
│   ├── benchmark_compare.py        # DeepSeek vs GPT-4 对比
│   ├── benchmark_hdr.py            # HDR 在 TruthfulQA 上验证
│   └── fetch_truthfulqa.py         # TruthfulQA 数据抓取
├── data/test_sets/        # 测试数据集
│   ├── truthfulqa_subset.json      # TruthfulQA 50题子集
│   ├── hdr_contradictions.json     # HDR 矛盾正例
│   └── hdr_consistent.json         # HDR 一致性负例
├── hal/                    # HAL层
│   └── llm_router.py     # LLM Router
├── api/                    # API服务层
│   └── server.py          # FastAPI服务
├── tests/                  # 测试套件
│   ├── test_walrus_memory.py       # Walrus Memory测试
│   ├── test_silicon_governance.py # 硅基代理治理测试
│   ├── test_web_session.py         # 浏览器云脑测试
│   ├── test_hdr.py                 # HDR测试
│   └── test_scs.py               # SCS测试
├── cli.py                  # CLI客户端
├── config.yaml             # 配置文件
├── requirements.txt        # Python依赖
└── docs/                 # 文档
    ├── USER_GUIDE.md      # 使用文档
    ├── TECHNICAL_REPORT.md  # 技术报告
    └── ARCHITECTURE.md      # 架构文档
```

---

## 📝 更新日志

### v4.2.0 (2026-06-10)

**新增**：
- USCS 页式内存管理（PageTable/PageAllocator/PageReclaimer，4KB 页粒度）
- 抢占调度器（PreemptiveScheduler 多级优先级队列 + ContextSwitch）
- 跨节点迁移代理（MigrationManager + LoadBalancer）
- TruthfulQA 50题外部基准（7类别，GPT-4 vs DeepSeek 对比评测框架）
- OSDI 论文新增 §4.7 TruthfulQA 外部基准验证章节

**Bug 修复**（4/4，回归测试 5/5 PASS）：
- BUG-1: `[ERROR]` 答案标记为 untruthful
- BUG-2: 添加 `--mock` 离线模式参数
- BUG-3: DeepSeek 离线模式自动回退 mock
- BUG-4: 字段名统一为 `correct_answers`/`incorrect_answers`

**文档**：
- 新增 `docs/osdi_paper_cn.md`（USCS 统一语义-计算状态论文）
- 新增 `docs/arch-uscs-kernel.md`（USCS 内核架构设计）
- 新增 `docs/prd-uscs-kernel.md`（USCS 内核需求文档）

### v4.0.0 (2026-06-04)

**新增**：
- 硅基代理治理体系（三旋治理 + 五层次穿透架构）
- AIC凭证系统（W3C VC扩展 + M175锚定 + 同伦类哈希）
- GCD归约算子（Pre/Post约束校验）
- 约柜合约（M175封印 + M106验收 + M178罚没）
- 确权仪式（Plan→Consult→Ratify）
- OPC注册表（人人即法人）

**升级**：
- `session.py` → v4: 集成三旋治理 + 五层次管道
- `config.yaml`: 新增governance/aic/gcd/ark/ratify/opc配置段

**测试**：
- 新增 `test_silicon_governance.py`: 45个测试
- 全量：65 passed, 2 skipped, 0回归

### v3.0.0 (2026-06-03)

**新增**：
- Walrus Memory集成（可移植记忆 + 完整性证明 + MCP桥接）
- 浏览器云脑模式（Playwright + WebWorldModel）
- MCP Bridge（stdio JSON-RPC）

**升级**：
- `continuation.py` → v2: SHA-256 proof链 + parent_kid
- `session.py` → v3: MemoryHub集成
- `config.yaml`: 新增memory/mcp/browser配置段

### v2.3.0 (2026-06-01)

**初始版本**：
- Continuation机制（AGI进程可序列化快照）
- Φ-Scheduler（幻觉控制）
- SelfConsistencyLoop（自洽性推演循环，v4.1 前称 CarbonSiliconGAN）
- Docker/systemd部署

---

## 🤝 贡献

欢迎提交Issue和Pull Request！

---

## 📄 许可证

MIT License

---

## 🔗 相关链接

- **GitHub仓库**: https://github.com/lisoleg/taiji-os-core
- **技术报告**: [docs/TECHNICAL_REPORT.md](docs/TECHNICAL_REPORT.md)
- **使用文档**: [docs/USER_GUIDE.md](docs/USER_GUIDE.md)
- **架构文档**: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)

---

**Built with ❤️ by [lisoleg](https://github.com/lisoleg)**
