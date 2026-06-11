# 太极OS v4.3 增量架构设计

> 版本：v1.0
> 日期：2026-06-11
> 作者：高见远（Gao），架构师
> 状态：待评审

---

## 1. 实现方案

### 1.1 代码线 C-P0（无 API 依赖）

#### C-P0-1：README.md 版本更新

**现状：** badge 显示 `v4.2.0`，changelog 最新条目为 v4.2.0，测试数显示 `65 passed`（实际应有 84 passed）。

**方案：**
- 将所有 `v4.2.0` badge 替换为 `v4.2.1`
- 在 changelog 顶部新增 v4.2.1 条目，描述 BF-1（USCS 核心 4 模块创建）和 BF-2（TruthfulQA mock accuracy 修复）
- 更新测试数：`65 passed` → `84 passed`
- 更新架构图标题 `太极OS v4.1 架构` → `太极OS v4.2 架构`

**修改文件：** `README.md`（仅修改，不重写）

---

#### C-P0-2：overview.md 重写为架构概览

**现状：** 当前 `overview.md` 是 v4.2.1 回归测试报告，与项目根目录入口文件的定位不符。

**方案：** 重写为 100-200 行架构概览，包含以下章节：

```
# 太极OS 架构概览

## 项目定位
（三段：AGI Agent Runtime、Continuation 机制、五层次穿透架构）

## 核心概念图
（ASCII art：L1-L5 五层次 + Walrus Memory + MCP Bridge）

## 模块架构
（目录树 + 每层职责 1 句话）

## 数据流
（用户输入 → LLM Router → G-Core/D-Core → Φ门控 → ψ更新 → 输出）

## 快速开始
（5 行：安装 → 配置 → 运行）
```

**修改文件：** `overview.md`（整篇重写）

---

#### C-P0-3：核心模块类型注解 + docstring

**现状分析（逐文件）：**

| 文件 | 模块 docstring | 类 docstring | 方法 docstring | `from __future__ import annotations` | 类型注解问题 |
|------|------|------|------|------|------|
| `core/session.py` | ✅ | ✅ | ✅（部分） | ✅ | `llm_router` 参数无类型 |
| `core/continuation.py` | ✅ | ✅ | ✅ | ❌（用 `typing.Tuple`） | 较好 |
| `core/world_model.py` | **❌ 无** | **❌ 字符串非 docstring** | 仅有 `encode`/`update`/`phi` | ❌ | 构造函数参数无类型注解 |
| `core/self_consistency_loop.py` | ✅ | ✅（字符串非 `"""`） | ✅ | ✅ | `step()` 返回 `tuple` 未标注 |
| `core/phi_scheduler.py` | ✅ | ✅ | ✅ | ✅ | 较好 |
| `core/uscs_mmu.py` | ✅ | ✅ | ✅ | ✅ | 较好 |
| `core/preemptive_scheduler.py` | ✅ | ✅ | ✅ | ✅ | `register()` 的 `session: Any` 建议改为 `TaijiSession` 前向引用 |
| `core/migration_agent.py` | ✅ | ✅ | ✅ | ✅ | `export_process()` 异常描述不完整 |
| `core/closure_env.py` | **❌ 无** | ✅（字符串非 `"""`） | `push`/`set_intent` **无 docstring** | ❌ | 构造函数 `intent` 无类型 |

**方案：**

1. **添加 `from __future__ import annotations`**（所有尚未加的 `core/` 文件）
2. **补充模块级 docstring**（在文件头部 `"""..."""` 格式）：
   - `world_model.py`：描述 WorldModel 职责、嵌入策略（在线 API vs 离线哈希）、ψ 更新语义
   - `closure_env.py`：描述 ClosureEnv 职责、intent/history/context 字段含义
3. **将类体中的普通字符串改为 `"""` docstring**（`self_consistency_loop.py`、`closure_env.py`）
4. **补充函数/方法类型注解**：
   - `session.py`：`__init__(self, sid: str, llm_router: LLMRouter, ...)`
   - `world_model.py`：`__init__(self, dim: int = 1536, config_path: str = "config.yaml") -> None`
   - `self_consistency_loop.py`：`step(self, env: dict, user_input: str) -> tuple[str | None, str]`
   - `closure_env.py`：`__init__(self, intent: str = "idle") -> None`，`push(self, role: str, content: str) -> None`
5. **补充缺失的方法 docstring**（`closure_env.py` 的 `push`、`set_intent`）

**修改文件：** 以下 `core/` 文件：
- `core/session.py`
- `core/continuation.py`
- `core/world_model.py`（主要改动）
- `core/self_consistency_loop.py`
- `core/phi_scheduler.py`
- `core/uscs_mmu.py`
- `core/preemptive_scheduler.py`
- `core/migration_agent.py`
- `core/closure_env.py`（主要改动）

---

#### C-P0-4：HAL 层接口规范

**现状分析：**

`hal/llm_router.py`：
- **无模块 docstring**
- **无类 docstring**（仅有普通字符串）
- `complete(self, prompt: str) -> str`：**返回值类型标注了，但参数 `prompt` 有类型，`max_retry` 逻辑内隐**
- `_call_primary`/`_call_fallback`：无 docstring，无返回类型
- 返回值：正常返回 `str`，异常返回 `"[LLMRouter Error] ..."`——**返回类型不统一（应定义为 `str` 或抛异常）**

`hal/nic_emu.py`：
- 模块 docstring ✅
- 类 docstring ✅
- 方法 docstring ✅（Google 风格）
- **但 `send()` 的 `snapshot: Any` 类型太宽泛**——建议定义为 `ProcessSnapshot | dict`

**方案：**

1. `hal/llm_router.py`：
   - 添加模块 docstring（描述 HAL 层定位、LLM Router 职责、fallback 策略）
   - 将类体普通字符串改为 `"""` docstring
   - `complete()` 添加可选异常抛出模式（或明确返回 `str` 含错误信息）
   - 为 `_call_primary`、`_call_fallback`、`_mock_response` 添加 docstring 和返回类型
2. `hal/nic_emu.py`：
   - `NodeTransport.send()` 的 `snapshot: Any` 改为 `snapshot: "ProcessSnapshot | dict"`
   - 补充 `_send_http` 中 `__future__ annotations` 前向引用说明

**修改文件：**
- `hal/llm_router.py`（主要改动）
- `hal/nic_emu.py`（小改动）

---

#### C-P0-5：syscalls 层 docstring 补充

**现状分析：**

| 文件 | 模块 docstring | 类 docstring | 方法 docstring |
|------|------|------|------|
| `syscalls/executor.py` | **❌ 仅有注释** | ✅ | 部分 |
| `syscalls/browser_executor.py` | 待确认 | 待确认 | 待确认 |
| `syscalls/planner.py` | 待确认 | 待确认 | 待确认 |
| `syscalls/web_planner.py` | 待确认 | 待确认 | 待确认 |
| `syscalls/mcp_bridge.py` | 待确认 | 待确认 | 待确认 |
| `syscalls/opc_registry.py` | 待确认 | 待确认 | 待确认 |
| `syscalls/auditor.py` | 待确认 | 待确认 | 待确认 |

**方案：**
- 为每个 `syscalls/*.py` 添加模块级 docstring（1-3 行，说明该模块职责）
- 如类 docstring 缺失则补充
- 保持风格与 `syscalls/executor.py` 现有风格一致（已有类 docstring）

**修改文件：** `syscalls/` 下所有 `.py` 文件（具体以实际读取为准）

---

### 1.2 论文线 B-P0（图表 + 文案）

#### B-P0-3：消融实验对比柱状图

**数据来源：** `scripts/ablation.py`（需确认是否有 E1 消融数据导出逻辑）

**方案：**
- 使用 `matplotlib` 绘制 E1 消融对比柱状图
- 子图 1：关键词 F1=0.035 vs D-Core 扩展关键词 F1=0.868
- 子图 2：E2 阈值对比（0.50/0.65/0.80）
- 子图 3：E3 adaptive vs static
- 输出：`docs/figures/ablation_comparison.png`（300 DPI）

**新建文件：** `scripts/gen_figures.py`（图表生成脚本）

---

#### B-P0-4：Φ 分布直方图

**数据来源：** `scripts/benchmark_compare.py` 或 `scripts/benchmark_hdr.py` 运行后产生的 Φ 值日志

**方案：**
- 稳定序列 Φ 分布（期望集中在 >0.5）——直方图
- 漂移序列 Φ 分布（期望分散/低值）——直方图
- 双直方图并排（`plt.subplots(1, 2)`）
- 输出：`docs/figures/phi_distribution.png`

**新建文件：** 复用 `scripts/gen_figures.py`

---

#### B-P0-5：SCS 对比率可视化

**数据来源：** `tests/test_scs.py` 或 `scripts/benchmark_compare.py`

**方案：**
- 箱线图：稳定 vs 漂移 × 漂移子类型（语义漂移/时间漂移/混合漂移）
- 或散点图：x=序列长度，y=SCS 对比率，颜色区分序列类型
- CV 对比率 21.27× 在图中标出（红色虚线）
- 输出：`docs/figures/scs_comparison.png`

**新建文件：** 复用 `scripts/gen_figures.py`

---

#### B-P0-6：§6.2 路线图更新

**修改文件：** `docs/osdi_paper_cn.md`（§6.2 路线图章节）

**方案：**
- 将 v4.2.0 TruthfulQA 框架已完成 标注为 ✅
- 新增 v4.3 真实 API 对比已完成（若 B-P0-1/2 完成）
- 下一站：SWE-bench / GAIA（标注为 🔲 规划中）

---

#### B-P0-7：参考文献扩充

**修改文件：** `docs/osdi_paper_cn.md`（参考文献章节）

**方案：** 补充以下条目：
1. Lin et al. ACL 2022 — TruthfulQA 原始论文
2. Li et al. EMNLP 2023 — HaluEval
3. Kwon et al. SOSP 2023 — PagedAttention (vLLM)
4. Strata 2024 — 多代理 OS 抽象
5. DeepSeek 技术报告（如可用）

---

### 1.3 P1/P2 需求

#### C-P1-1：`config.yaml` 无 API key 友好默认配置

**方案：**
- 新增 `taiji.phi_mode: "hash"` 作为默认值（当前默认依赖 API）
- 新增 `taiji.api_mode: "offline"` 为 mock 模式
- 在 README 中说明：`DEEPSEEK_API_KEY` 为空时自动使用 hash 模式

**修改文件：** `config.yaml`

---

#### C-P1-2：`tests/` 目录 README

**方案：** 新建 `tests/README.md`，说明：
- 测试结构（每个测试文件覆盖哪些模块）
- 运行方式（`pytest tests/ -v`）
- mock 模式说明（`--mock`）

**新建文件：** `tests/README.md`

---

#### C-P1-3：`scripts/` 文件头 docstring

**方案：** 为以下脚本添加文件头 docstring（5-10 行）：
- `scripts/benchmark_gpt4_baseline.py`
- `scripts/benchmark_compare.py`
- `scripts/benchmark_hdr.py`
- `scripts/fetch_truthfulqa.py`
- `scripts/ablation.py`
- `scripts/gen_figures.py`（新建）

---

#### C-P2-1：`.gitignore` 校验

**方案：** 确保以下被忽略：
```
__pycache__/
*.pyc
venv/
.env
checkpoints/
snapshots/
memory_store/
swap/
*.json      # snapshot JSON（可选，或仅忽略 snapshots/）
```

**修改文件：** `.gitignore`

---

#### C-P2-2：`requirements.txt` 版本锁定

**方案：** 将 `requirements.txt` 从无序列表改为 pin 版本：
```
openai==1.x.x
pyyaml==6.x
numpy==1.x
matplotlib>=3.5
pytest>=7.0
playwright>=1.30
```

**修改文件：** `requirements.txt`

---

## 2. 文件清单

| 编号 | 文件路径 | 操作类型 | 说明 |
|------|----------|----------|------|
| 1 | `README.md` | 修改 | 版本 badge + changelog |
| 2 | `overview.md` | 重写 | 测试报告 → 架构概览 |
| 3 | `core/world_model.py` | 修改 | 模块/类 docstring + 类型注解 |
| 4 | `core/closure_env.py` | 修改 | 模块 docstring + 方法类型注解 |
| 5 | `core/session.py` | 修改 | `llm_router` 参数类型注解 |
| 6 | `core/self_consistency_loop.py` | 修改 | 类 docstring 格式修正 |
| 7 | `core/continuation.py` | 修改 | 添加 `from __future__ import annotations` |
| 8 | `core/phi_scheduler.py` | 修改 | 小幅补全类型注解 |
| 9 | `core/uscs_mmu.py` | 修改 | 小幅补全类型注解 |
| 10 | `core/preemptive_scheduler.py` | 修改 | `session` 参数类型前向引用 |
| 11 | `core/migration_agent.py` | 修改 | 小幅补全 docstring |
| 12 | `hal/llm_router.py` | 修改 | 模块/类 docstring + 接口规范 |
| 13 | `hal/nic_emu.py` | 修改 | `snapshot` 参数类型收窄 |
| 14 | `syscalls/executor.py` | 修改 | 模块 docstring 格式修正 |
| 15 | `syscalls/*.py`（其余） | 修改 | 模块 docstring 补充 |
| 16 | `config.yaml` | 修改 | 无 API key 友好默认配置 |
| 17 | `tests/README.md` | **新建** | 测试结构说明 |
| 18 | `scripts/gen_figures.py` | **新建** | 图表生成脚本（B-P0-3/4/5） |
| 19 | `.gitignore` | 修改 | 校验忽略规则 |
| 20 | `requirements.txt` | 修改 | 版本锁定 |
| 21 | `docs/osdi_paper_cn.md` | 修改 | B-P0-6/7 论文更新 |
| 22 | `docs/figures/ablation_comparison.png` | **新建** | 消融对比柱状图 |
| 23 | `docs/figures/phi_distribution.png` | **新建** | Φ 分布直方图 |
| 24 | `docs/figures/scs_comparison.png` | **新建** | SCS 对比率可视化 |

---

## 3. 任务列表

> 依赖关系：`→` 表示"依赖于"

| 任务编号 | 优先级 | 复杂度 | 依赖 | 任务描述 |
|----------|--------|--------|------|----------|
| T-01 | P0 | 🟢 小 | 无 | C-P0-1：`README.md` 版本更新（v4.2.0 → v4.2.1） |
| T-02 | P0 | 🟡 中 | 无 | C-P0-2：`overview.md` 重写为架构概览 |
| T-03 | P0 | 🟡 中 | T-01 | C-P0-3：`core/world_model.py` 补充 docstring + 类型注解 |
| T-04 | P0 | 🟢 小 | T-01 | C-P0-3：`core/closure_env.py` 补充 docstring + 类型注解 |
| T-05 | P0 | 🟢 小 | T-01 | C-P0-4：`hal/llm_router.py` 补充 docstring + 接口规范 |
| T-06 | P0 | 🟢 小 | T-01 | C-P0-5：`syscalls/executor.py` 补充模块 docstring |
| T-07 | P0 | 🟡 中 | T-03, T-04 | C-P0-3：其余 `core/` 文件补全类型注解（批量） |
| T-08 | P0 | 🟢 小 | T-05 | C-P0-4：`hal/nic_emu.py` 参数类型收窄 |
| T-09 | P0 | 🟢 小 | T-06 | C-P0-5：其余 `syscalls/*.py` 补充模块 docstring（批量） |
| T-10 | P0 | 🟡 中 | 无 | B-P0-6：`docs/osdi_paper_cn.md` §6.2 路线图更新 |
| T-11 | P0 | 🟢 小 | 无 | B-P0-7：`docs/osdi_paper_cn.md` 参考文献扩充 |
| T-12 | P1 | 🟢 小 | 无 | C-P1-1：`config.yaml` 无 API key 友好默认配置 |
| T-13 | P1 | 🟢 小 | 无 | C-P1-2：新建 `tests/README.md` |
| T-14 | P1 | 🟢 小 | 无 | C-P1-3：`scripts/*.py` 文件头 docstring 补充 |
| T-15 | P2 | 🟢 小 | 无 | C-P2-1：`.gitignore` 校验 |
| T-16 | P2 | 🟢 小 | 无 | C-P2-2：`requirements.txt` 版本锁定 |
| T-17 | P1 | 🟡 中 | 无 | B-P0-3/4/5：新建 `scripts/gen_figures.py` + 生成 3 张图 |
| T-18 | P0 | 🔴 大 | Q-1 确认 | B-P0-1/2：E1-E5 真实语义嵌入对照 + E7 TruthfulQA 实际对比（需 API key） |

**说明：**
- T-03/T-04 可并行（不同文件）
- T-07 依赖 T-03/T-04 完成（确保风格一致）
- T-17（图表）可在无 API key 时先用 mock 数据生成框架图
- T-18（真实实验）依赖 Q-1（DeepSeek API key）确认

---

## 4. 框架/工具选型

### 4.1 图表库

**推荐：`matplotlib` + `seaborn`**

| 库 | 用途 | 理由 |
|-----|------|------|
| `matplotlib` | 核心绘图、导出 PNG | 零依赖、精细控制、论文级输出 |
| `seaborn` | 统计图表（箱线图、直方图） | 默认样式美观、适合论文 |

**不推荐 `plotly`：** 论文需要静态高清 PNG，`plotly` 主打交互式 HTML，增加依赖且不适合打印。

**决策：** `matplotlib` + `seaborn`，在 `scripts/gen_figures.py` 中统一封装。

---

### 4.2 Docstring 风格

**推荐：Google 风格**

理由：
1. 项目现有 `hal/nic_emu.py` 已使用 Google 风格（`Args:`/`Returns:`/`Raises:`）
2. Google 风格在 VS Code 中渲染效果好（Python Docstring Generator 插件默认）
3. NumPy 风格过于冗长，Sphinx 风格不适合纯 Python 项目

**规范模板：**

```python
def function_name(param1: str, param2: int = 10) -> bool:
    """函数功能一句话描述。

    详细描述（可选，1-3 段）。

    Args:
        param1: 参数1 说明。
        param2: 参数2 说明，默认值 10。

    Returns:
        返回说明。

    Raises:
        ValueError: 当 xxx 时抛出。
    """
```

---

### 4.3 类型注解风格

**规范：**
1. 所有 public 方法必须有参数类型注解 + 返回类型注解
2. private 方法（`_开头`）可省略，但建议标注
3. 使用 `from __future__ import annotations`（允许前向引用如 `TaijiSession`）
4. 避免 `Any`，用 `Union` 或具体类型替代；确实无法确定的用 `Any` 并加 `# type: ignore`
5. 集合类型用 lowercase：`list[str]`、`dict[str, int]`（Python 3.9+）

**示例：**

```python
from __future__ import annotations

class TaijiSession:
    def __init__(self, sid: str, llm_router: LLMRouter, ...) -> None:
        ...

    def run(self, user_input: str) -> str:
        ...

    def search_memory(self, query: str) -> list[dict]:
        ...
```

---

## 5. 共享知识（跨文件约定）

### 5.1 类型注解规范

| 场景 | 约定 |
|------|------|
| 前向引用（未定义类） | `from __future__ import annotations` + 直接用类名 |
| 可选参数 | `str | None = None`（Python 3.10+）或 `Optional[str] = None` |
| 集合类型 | `list[str]`、`dict[str, int]`（不用 `List`、`Dict`） |
| 无法确定的类型 | `Any`（需加注释说明原因） |
| 回调函数 | `Callable[[str], None]` |

### 5.2 Docstring 模板

**模块级 docstring（每个 `.py` 文件头部）：**

```python
"""module_name — 一句话描述

详细描述（可选，1-3 段）。
提及依赖的模块、设计模式、注意事项。
"""
```

**类 docstring：**

```python
class ClassName:
    """一句话描述。

    详细描述（可选）。

    Attributes:
        attr1: 属性1 说明。
        attr2: 属性2 说明。
    """
```

**方法 docstring：**

```python
def method(self, param: str) -> bool:
    """一句话描述。

    Args:
        param: 参数说明。

    Returns:
        返回说明。

    Raises:
        ExceptionType: 触发条件。
    """
```

### 5.3 命名约定

| 类型 | 约定 | 示例 |
|------|------|------|
| 类 | `PascalCase` | `TaijiSession` |
| 函数/方法 | `snake_case` | `run_structured` |
| 常量 | `UPPER_SNAKE` | `DEFAULT_THRESHOLD` |
| 私有属性/方法 | `_leading_underscore` | `_save_continuation` |
| 类型别名 | `PascalCase` | `ProcessSnapshotDict` |

### 5.4 `from __future__ import annotations` 使用规则

- **所有 `core/` 和 `hal/` 文件必须加**（目前约一半有）
- `syscalls/` 可选（依赖复杂时用）
- 加了之后，参数类型注解中可直接用类名，无需加引号

---

## 6. 待明确事项（需用户确认）

| 编号 | 问题 | 背景 | 建议 |
|------|------|------|------|
| Q-1 | **DeepSeek API key 可用性** | B-P0-1 和 B-P0-2 都需要真实 API 运行实验 | 优先确认用户是否已有 key 或愿意申请。如有 key，优先跑 E7（数据量小）；如无，先完成 B-P0-3/4/5 + C 线全部需求 |
| Q-2 | **论文投稿目标会议** | PRD 标题为"OSDI 论文"，但 §6.3 提出 ACL/EMNLP 短文 | 建议明确：如目标 ACL 短文则当前长度合适；如目标 OSDI 则需扩展到 12-14 页 |
| Q-3 | **代码重构深度边界** | C-P0-3 要求"类型注解 + docstring"，是否做更深层的重构？ | 建议 v4.3 仅做浅层统一，不触及逻辑重构，避免引入 bug |
| Q-4 | **`overview.md` 与 `ARCHITECTURE.md` 的定位边界** | 当前 `docs/ARCHITECTURE.md` 已存在 | 建议 `overview.md` 放项目根目录作新用户入口（100-200 行），`ARCHITECTURE.md` 放 docs/ 作详细技术文档 |
| Q-5 | **论文作者署名/致谢** | 当前论文无作者信息 | 确认作者列表、单位、致谢，补充到论文头部 |

---

## 附录 A：任务分组建议（里程碑）

| 里程碑 | 内容 | 预计耗时 | 依赖 |
|--------|------|----------|------|
| **M1 — 低垂果实周** | T-01/02/05/06/10/11/12/13/14/15/16 | 2-3 天 | 无 API 依赖 |
| **M2 — 图表周** | T-17（B-P0-3/4/5） | 2-3 天 | 需要实验数据（mock 可先出框架图） |
| **M3 — 实验周** | T-18（B-P0-1/2） | 1-2 天 | **需要 API key（Q-1）** |
| **M4 — 代码深度周** | T-03/04/07/08/09 | 3-4 天 | M1 完成后 |
| **M5 — 终审** | Q-1~Q-5 全部确认、PRD 终审 | 1 天 | 以上全部 |

---

## 附录 B：参考文献（待扩充到论文）

1. Lin et al. ACL 2022 — TruthfulQA: Measuring How Models Mimic Human Falsehoods
2. Li et al. EMNLP 2023 — HaluEval: A Large-Scale Hallucination Evaluation Benchmark
3. Kwon et al. SOSP 2023 — PagedAttention in vLLM: Efficient Memory Management for LLM Serving
4. Strata 2024 — Strata: A Multi-Agent OS Abstraction for LLM Workflows
5. DeepSeek AI — DeepSeek Technical Report (if available)
6. Ouyang et al. NeurIPS 2022 — Training Language Models to Follow Instructions with Human Feedback（RLHF 基准）
