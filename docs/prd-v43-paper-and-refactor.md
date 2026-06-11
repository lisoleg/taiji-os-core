# 太极OS v4.3 — 论文完善 + 代码重构优化 PRD

> 版本：v1.0  
> 日期：2026-06-11  
> 作者：许清楚（Xu），太极OS 产品经理  
> 状态：待评审

---

## 1. 产品目标

**v4.3 交付一份可投稿的 OSDI 论文（补充真实语义嵌入对比实验 + 图表 + 参考文献）和一个接口统一、文档完备的代码库。**

---

## 2. 用户故事

| # | 角色 | 故事 | 验收标准 |
|---|------|------|----------|
| US-1 | OSDI 审稿人 | 作为审稿人，我需要在 §4 看到哈希嵌入（Φ≈0）与真实语义嵌入（DeepSeek）的对照数据，以判断 Φ 门控的判别力是否依赖于嵌入质量 | §4.1-4.5 每节有 DeepSeek 对照列 |
| US-2 | OSDI 审稿人 | 作为审稿人，我需要在 E7 看到 TruthfulQA 上 GPT-4 零样本 vs DeepSeek Self-Consistency 的实际对比，而非"框架就绪"状态 | E7 有真实 API 跑出的 Accuracy/F1 对比表 |
| US-3 | OSDI 审稿人 | 作为审稿人，我需要消融对比柱状图、Φ 分布图、SCS 对比率可视化，以便直观理解实验结果 | ≥3 张图表（E1 消融柱状图、E4 Φ 分布、SCS 对比率） |
| US-4 | 代码维护者 | 作为贡献者，我需要核心模块有统一的接口签名和类型注解，以便理解数据流和扩展功能 | `core/` 10+ 模块有 `__init__` 类型签名 |
| US-5 | 新用户 | 作为新用户，我需要 README 反映当前版本（v4.2.1）状态，`overview.md` 是真正的架构概览而非测试报告 | README badge 更新，overview.md 重写为架构概览 |

---

## 3. 需求池

### 3.1 P0 — 论文（B线）

| 编号 | 需求 | 涉及章节 | 当前状态 | 目标状态 | 工作量 |
|------|------|----------|----------|----------|--------|
| B-P0-1 | **E1-E5 真实语义嵌入对照数据** | §4.1-4.5 | 全部基于哈希嵌入（Φ≈0），每条标注"确定性哈希嵌入模式下的基线测量" | 每节表格增加 DeepSeek API 对照列，展示真实 Φ 值下的 Acc/Prec/Rec/F1 | 🔴 大（需要 DeepSeek API key） |
| B-P0-2 | **E7 TruthfulQA 实际对比结果** | §4.7 | 框架就绪但 mock accuracy=0，"仅需接入真实 API 密钥即可产出有意义的语义对比" | 接入 DeepSeek API 跑出 GPT-4 baseline vs DeepSeek Self-Consistency 的 Accuracy/F1/领域覆盖对比表 | 🔴 大（需要 DeepSeek API key） |
| B-P0-3 | **消融实验对比柱状图** | §4.1-4.3 | 无图 | E1 柱状图（关键词 F1=0.035 vs D-Core 扩展关键词 F1=0.868 vs DeepSeek 语义 TBD）；E2-E3 阈值/adaptive 对比子图 | 🟡 中 |
| B-P0-4 | **Φ 分布直方图** | §4.4 | 无图 | 稳定序列 Φ 分布（期望集中在 >0.5）vs 漂移序列 Φ 分布（期望分散/低值），双直方图并排 | 🟡 中 |
| B-P0-5 | **SCS 对比率可视化** | §4.4 | 仅有表 | CV 对比率 21.27× 的箱线图或散点图（稳定 vs 漂移 × 漂移子类型） | 🟡 中 |
| B-P0-6 | **§6.2 路线图更新** | §6.2 | v4.2.0 内容，标注"本文已完成"的仍是旧内容 | 更新为：v4.2.0 TruthfulQA 框架已完成、v4.3 真实 API 对比已完成、下一站 SWE-bench/GAIA | 🟢 小 |
| B-P0-7 | **参考文献扩充** | 参考文献 | 仅 4 条 | 补充：Lin et al. ACL 2022 (TruthfulQA)、Li et al. EMNLP 2023 (HaluEval)、Kwon et al. SOSP 2023 (PagedAttention)、Strata 2024、DeepSeek 技术报告（如可用） | 🟢 小 |

**注意**：B-P0-1 和 B-P0-2 依赖 DeepSeek API key。如果用户当前没有 key，需评估替代方案：
- 方案 A：用户申请 DeepSeek API key 后批量运行
- 方案 B：使用其他免费嵌入 API（如 text-embedding-3-small）做近似对比，论文中注明限制
- 方案 C：当前阶段仅完善论文框架，实验数据标注"待 API key"，先把图表/参考文献/路线图完成

### 3.2 P0 — 代码（C线）

| 编号 | 需求 | 涉及文件 | 当前状态 | 目标状态 | 工作量 |
|------|------|----------|----------|----------|--------|
| C-P0-1 | **README 版本更新** | `README.md` | badge 显示 `v4.2.0`，changelog 最新条目 v4.2.0 | badge 更新为 `v4.2.1`，新增 v4.2.1 changelog 条目（BF-1 USCS 核心 4 模块创建、BF-2 TruthfulQA mock accuracy 修复） | 🟢 小 |
| C-P0-2 | **`overview.md` 重写** | `overview.md` | 当前是 v4.2.1 回归测试报告 | 重写为真正的项目架构概览：系统分层图、模块关系、数据流、五层次架构、USCS 内核 | 🟡 中 |
| C-P0-3 | **核心模块接口统一** | `core/` 10+ 文件 | 部分模块缺少类型注解和 docstring | 为 `session.py`, `continuation.py`, `world_model.py`, `self_consistency_loop.py`, `phi_scheduler.py`, `uscs_mmu.py`, `preemptive_scheduler.py`, `migration_agent.py`, `closure_env.py` 添加 `from __future__ import annotations` + 关键函数类型签名 + 模块级 docstring | 🟡 中 |
| C-P0-4 | **HAL 层接口规范** | `hal/llm_router.py`, `hal/nic_emu.py` | llm_router 接口可能不统一，nic_emu 新模块 | 统一 LLM Router 的 `route()` 返回值类型（目前可能返回 str / dict 混合）；为 nic_emu 补充 docstring | 🟢 小 |
| C-P0-5 | **syscalls 层文档补充** | `syscalls/` | `executor.py`, `browser_executor.py`, `mcp_bridge.py` 等缺少模块 docstring | 每个 syscall 模块添加模块级 docstring 说明职责 | 🟢 小 |

### 3.3 P1 — 次要改进

| 编号 | 需求 | 说明 | 工作量 |
|------|------|------|--------|
| B-P1-1 | **§3 实验设计补充假设检验细节** | 当前缺效应量（Cohen's d）计算细节和统计显著性（p-value）报告方法 | 🟢 小 |
| B-P1-2 | **§5 相关工作补充 2024-2025 最新工作** | 补充 Megatron-LM、JAX/PAX 等系统、以及最近 LLM agent 的 OS 抽象相关论文 | 🟡 中 |
| C-P1-1 | **`config.yaml` 无 API key 友好的默认配置** | 添加 `phi_mode: hash` 作为安全默认值，`api_mode: offline` 为 mock 模式，减小新用户入门摩擦 | 🟢 小 |
| C-P1-2 | **测试文档补充** | 在 `tests/` 目录添加 README 说明测试结构、各文件覆盖范围、运行方式 | 🟢 小 |
| C-P1-3 | **`scripts/` 文档补充** | 为 4 个 benchmark 脚本添加文件头 docstring 说明用途和参数 | 🟢 小 |

### 3.4 P2 — 锦上添花

| 编号 | 需求 | 说明 | 工作量 |
|------|------|------|--------|
| B-P2-1 | **论文英文版撰写/机翻** | 当前只有中文版，投稿需要英文版（ACL/EMNLP/OSDI 均为英文会议） | 🔴 大 |
| B-P2-2 | **论文图美观升级** | 当前图表为纯文字表格，升级为 matplotlib/seaborn 专业图表风格 | 🟡 中 |
| C-P2-1 | **`.gitignore` 校验** | 确保 `*.pyc`, `__pycache__`, `venv/`, `.env`, `checkpoints/` 被忽略 | 🟢 小 |
| C-P2-2 | **`requirements.txt` 版本锁定** | 当前无版本号，锁定主依赖版本避免环境漂移 | 🟢 小 |

---

## 4. 待确认问题

| # | 问题 | 背景 | 建议 |
|---|------|------|------|
| Q-1 | **DeepSeek API key 可用性** | B-P0-1 和 B-P0-2 都需要真实 API 运行实验，这是论文的全部新增实验价值所在。没有 API key 则 P0 论文需求几乎全部无法推进 | 优先确认用户是否已有 key 或愿意申请（DeepSeek API 注册即送额度）。如有 key，优先跑 E7 TruthfulQA（50 题数据量小，快）；再跑 E1-E5 对照；如果暂时没有，先完成 B-P0-3/4/5/6/7 + C 线全部需求 |
| Q-2 | **论文投稿目标会议确认** | §6.3 提出 ACL/EMNLP 短文先投，但 PRD 标题为"OSDI 论文"。ACL 短文（6-8 页）和 OSDI（12-14 页）的篇幅和内容要求差异巨大。当前论文 7 章约 ~300 行，可能只够 ACL 短文篇幅 | 建议明确：如果目标 ACL 短文则当前长度合适，仅需补充实验数据；如果目标 OSDI 则需要扩展到 12-14 页纸版，包含更多 §6.1 的 USCS 页式管理系统实现细节 |
| Q-3 | **代码重构深度边界** | C-P0-3 要求"模块级 docstring + 关键函数类型签名"。是否需要更深层的重构（如类方法重命名、配置项统一、错误处理统一）？ | 建议 v4.3 仅做浅层统一（docstring + 类型注解），不触及逻辑重构，避免引入 bug |
| Q-4 | **`overview.md` 与 `ARCHITECTURE.md` 的定位边界** | 当前 `docs/ARCHITECTURE.md` 已存在，如果 `overview.md` 也写架构概览，两者功能可能重叠 | 建议 `overview.md` 放项目根目录，作新用户第一眼入口（100-200 行），`ARCHITECTURE.md` 放 docs/ 作为详细技术文档 |
| Q-5 | **论文作者署名/致谢** | 当前论文无作者信息 | 确认作者列表、单位、致谢（如 DeepSeek API 使用致谢），补充到论文头部 |

---

## 5. 里程碑建议

| 阶段 | 内容 | 依赖 |
|------|------|------|
| **M1 — 低垂果实周** | C-P0-1/2/4/5 + B-P0-6/7 + C-P1-1/2/3 + C-P2 | 无 API 依赖，可立即启动 |
| **M2 — 图表周** | B-P0-3/4/5（3 张图 + 视觉美化） | 需要实验原始数据（已有，来自 mock 运行） |
| **M3 — 实验周** | B-P0-1/2（DeepSeek API 对照实验） | **需要 API key（Q-1）** |
| **M4 — 代码深度周** | C-P0-3（类型注解 + docstring） | M1 完成后 |
| **M5 — 终审** | Q-1~Q-5 全部确认、PRD 终审 | 以上全部 |

---

## 附录 A：已知限制

1. **API 依赖瓶颈**：论文新增实验价值 80% 依赖 DeepSeek API key，没有 key 则 v4.3 对论文的提升仅限于图表 + 排版
2. **英文版缺失**：当前论文仅中文，投稿 ACL/EMNLP/OSDI 均需英文版，翻译工作量较大
3. **代码库不熟悉**：PRD 作者未逐文件阅读 `core/` 所有模块，C-P0-3 的类型注解需求基于经验推断，实际文件可能需要调整
4. **mock 数据的局限性**：当前所有 E1-E6 实验数据基于哈希嵌入（Φ≈0），这些"负结果"反而成为最有价值的发现——但它们不能替代真实语义嵌入的正面结果
