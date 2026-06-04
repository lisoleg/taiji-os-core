# 太极OS 使用文档

> **Taiji OS User Guide** — 安装、配置、使用、故障排除完整指南

---

## 目录

1. [安装指南](#1-安装指南)
2. [配置说明](#2-配置说明)
3. [CLI使用](#3-cli使用)
4. [Python API](#4-python-api)
5. [工作模式](#5-工作模式)
6. [Walrus Memory](#6-walrus-memory)
7. [MCP Bridge](#7-mcp-bridge)
8. [硅基代理治理](#8-硅基代理治理)
9. [故障排除](#9-故障排除)

---

## 1. 安装指南

### 1.1 环境要求

| 依赖 | 版本 | 必需 |
|------|------|------|
| Python | 3.10+ | ✅ |
| DeepSeek API Key | - | ✅ |
| Claude API Key | - | ⚠️ 可选（fallback）|
| Playwright | 最新 | ⚠️ 仅Web模式需要 |

### 1.2 安装步骤

```bash
# 1. 克隆仓库
git clone https://github.com/lisoleg/taiji-os-core.git
cd taiji-os-core

# 2. 创建虚拟环境（推荐）
python3 -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate     # Windows

# 3. 安装Python依赖
pip install -r requirements.txt

# 4. 安装浏览器驱动（仅Web模式需要）
playwright install chromium
```

### 1.3 验证安装

```bash
# 运行测试验证安装
pytest tests/test_hdr.py -v
# 预期：PASS
```

---

## 2. 配置说明

### 2.1 配置文件

主配置文件：`config.yaml`

### 2.2 API Keys配置

```yaml
# 设置环境变量（推荐）
export DEEPEEK_API_KEY="sk-..."
export CLAUDE_API_KEY="sk-..."  # 可选

# 或直接修改config.yaml
llm:
  provider: deepseek
  api_key: sk-...  # 直接填写（不推荐，会提交到Git）
```

### 2.3 各模块配置

#### LLM配置

```yaml
llm:
  provider: deepseek
  base_url: https://api.deepseek.com/v1
  api_key: ${DEEPSEEK_API_KEY}
  model: deepseek-reasoner
  temperature: 0.2

fallback:
  enabled: true
  provider: claude
  api_key: ${CLAUDE_API_KEY}
  model: claude-sonnet-4-20250514
```

#### Embedding配置

```yaml
embedding:
  provider: deepseek
  base_url: https://api.deepseek.com/v1
  api_key: ${DEEPSEEK_API_KEY}
  model: deepseek-embedding
```

#### 太极OS核心配置

```yaml
taiji:
  phi_threshold: 0.65   # Φ阈值（低于此值触发Continuation）
  max_retry: 3          # 最大重试次数
  snapshot_dir: snapshots # Continuation快照目录
```

#### 浏览器云脑配置

```yaml
browser:
  mode: text          # "text" | "web"
  headless: true      # Web模式下浏览器是否无头启动
  default_engine: baidu   # 默认搜索引擎
  timeout_ms: 30000   # 页面加载超时（毫秒）
```

#### Walrus Memory配置

```yaml
memory:
  enabled: true
  store_dir: memory_store    # 记忆文件存储目录
  auto_register: true        # 自动注册session到MemoryHub
```

#### MCP Bridge配置

```yaml
mcp:
  enabled: true
  transport: stdio           # MCP传输协议: stdio | sse
  tools:
    - taiji.run
    - taiji.status
    - taiji.resume
    - taiji.memory_search
    - taiji.verify
    - taiji.list_sessions
```

#### 硅基代理治理配置

```yaml
# 三旋治理
governance:
  enabled: true
  mode: tri-spin             # tri-spin | basic | none

  consensus:
    require_acknowledge: true
    recursive_check: true
    max_acknowledge_retry: 3

  cryptography:
    require_aic: true
    require_seal: true
    signature_algo: sha256with_ecdsa

  statute:
    require_gcd: true
    auto_complete: false
    auto_slash_on_error: true

# AIC凭证系统
aic:
  credential_type: AgentIdentityCredential
  standard: w3c_vc_v1.1_extended

# GCD归约算子
gcd:
  mode: strict               # strict | loose | audit
  builtin_rules:
    - browser.navigate
    - browser.click
    - browser.type
    - shell.exec
    - file.read
    - file.write
    - api.call
  custom_rules_dir: gcd_rules/

# 约柜合约
ark:
  chain: simulated
  pos:
    finality_depth: 12
    block_time_ms: 2000
  covenant:
    auto_seal: true
    auto_settle: false

# 确权仪式
ratify:
  phases:
    - plan
    - consult
    - ratify
  require_consult_approval: true
  auto_ratify: false

# OPC注册表
opc:
  default_asset_pool: 0.0
  max_agents_per_rn: 100
  personality:
    auto_grant: false
```

---

## 3. CLI使用

### 3.1 基本用法

```bash
python cli.py [选项] [指令]
```

### 3.2 选项说明

| 选项 | 说明 | 示例 |
|------|------|------|
| `cmd` | 要执行的指令 | `python cli.py "设计芯片"` |
| `--sid` | Session ID | `python cli.py --sid alice "设计芯片"` |
| `--continue <kid>` | 从Continuation ID恢复 | `python cli.py --continue abc12345` |
| `--status` | 查看session状态 | `python cli.py --status` |
| `--web` | 启用浏览器云脑模式 | `python cli.py --web "搜索太极OS"` |
| `--no-headless` | Web模式显示浏览器窗口 | `python cli.py --web --no-headless` |

### 3.3 使用示例

#### 文本模式

```bash
# 单次执行
python cli.py --sid alice "设计芯片"

# 交互模式
python cli.py --sid alice
>>> 设计芯片
>>> 继续优化

# 恢复Continuation
python cli.py --sid alice --continue abc12345

# 查看状态
python cli.py --sid alice --status
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

---

## 4. Python API

### 4.1 基本使用

```python
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from hal.llm_router import LLMRouter
from core.session import TaijiSession

# 初始化
llm = LLMRouter()
session = TaijiSession("mySession", llm, mode="text")

# 执行指令
result = session.run("设计芯片")
print(result)

# 查看状态
status = session.status()
print(status)
```

### 4.2 Web模式

```python
from hal.llm_router import LLMRouter
from core.session import TaijiSession

llm = LLMRouter()
session = TaijiSession("mySession", llm, mode="web", headless=True)

result = session.run("搜索 太极OS")
print(result)
```

### 4.3 Walrus Memory集成

```python
from core.memory_hub import MemoryHub
from core.session import TaijiSession

memory_hub = MemoryHub(store_dir="memory_store")
session = TaijiSession("mySession", llm, memory_hub=memory_hub)

# 搜索记忆
results = session.search_memory("芯片设计")
print(results)

# 验证完整性
is_valid = session.verify_integrity()
print(f"Integrity: {is_valid}")
```

### 4.4 硅基代理治理

```python
from core.session import TaijiSession

session = TaijiSession(
    sid="myAgent",
    llm_router=llm,
    governance="tri-spin",
    agent_spec="Agent行为规范文本",
    owner_did="did:opc:owner123",
    escrow_tokens=100.0,
)

# 执行（自动穿越五层次穿透架构）
result = session.run("设计芯片")
print(result)

# 查看治理状态
status = session.status()
print(status["governance_report"])
```

---

## 5. 工作模式

### 5.1 文本模式（默认）

- 使用`WorldModel`（DeepSeek Embedding向量化）
- 使用`Executor`（文本执行器）
- 使用`Planner`（文本规划器）

适用场景：纯文本推演、代码生成、文档撰写

### 5.2 浏览器云脑模式

- 使用`WebWorldModel`（DOM+URL+截图向量化）
- 使用`PlaywrightExecutor`（浏览器自动化）
- 使用`WebPlanner`（Web规划器）

适用场景：网页搜索、表单填写、数据抓取、Web自动化

---

## 6. Walrus Memory

### 6.1 概念

Walrus Memory是太极OS的**跨会话共享记忆空间**，概念映射：

| Walrus概念 | 太极OS实现 |
|-----------|-------------|
| Portable Memory | Continuation v2 proof链 |
| Integrity Proofs | SHA-256 verify()验证 |
| Shared Memory | MemoryHub跨session空间 |
| MCP Native | mcp_bridge.py stdio JSON-RPC |

### 6.2 使用

```python
from core.memory_hub import MemoryHub

hub = MemoryHub(store_dir="memory_store")

# 注册session
hub.register("session1")

# 存储continuation（自动生成proof链）
hub.store(continuation_data)

# 搜索记忆
results = hub.search("芯片设计", top_k=5)

# 验证完整性
is_valid = hub.verify_all()
```

### 6.3 MCP工具

```bash
# 启动MCP Bridge
python -m syscalls.mcp_bridge

# 然后Claude Desktop等MCP客户端可调用：
# - taiji.memory_search(q)
# - taiji.verify(mid)
```

---

## 7. MCP Bridge

### 7.1 启动

```bash
python -m syscalls.mcp_bridge
```

### 7.2 暴露的工具

| 工具 | 说明 |
|------|------|
| `taiji.run(query)` | 执行一轮推演 |
| `taiji.status(sid)` | 查询会话状态 |
| `taiji.resume(kid)` | 从Continuation恢复 |
| `taiji.memory_search(q)` | 搜索共享记忆 |
| `taiji.verify(mid)` | 验证记忆完整性 |
| `taiji.list_sessions()` | 列出已注册会话 |

### 7.3 Claude Desktop配置

在`claude_desktop_config.json`中添加：

```json
{
  "mcpServers": {
    "taiji-os": {
      "command": "python",
      "args": ["-m", "syscalls.mcp_bridge"],
      "cwd": "/path/to/taiji-os-core"
    }
  }
}
```

---

## 8. 硅基代理治理

### 8.1 三旋治理

| 维度 | 说明 | 配置 |
|------|------|------|
| **情治 (Consensus)** | 激活主体责任意识 | `governance.consensus.*` |
| **理治 (Cryptography)** | 密码学锚定身份契约 | `governance.cryptography.*` |
| **法治 (Statute)** | 行为归责 | `governance.statute.*` |

### 8.2 五层次穿透架构

| 层次 | 名称 | 功能 |
|------|------|------|
| L1 | 流贯 (Ftel) | 意图捕获 + φ度量 |
| L2 | 代数壳 (M175) | AIC锚定 + 归责校验 |
| L3 | 拓扑流贯 (GCD) | 约束校验 + 执行流 |
| L4 | IDO/ICE | M106验收/M178罚没 |
| L5 | 现象渲染 | 交付物 + 审计追踪 |

### 8.3 AIC凭证

```python
from core.aic import ACIssuer, AgentIdentityCredential

issuer = ACIssuer()
credential = issuer.issue(
    agent_name="myAgent",
    owner_did="did:opc:owner123",
    capabilities=["text_generate", "web_browse"],
    pi_spec="Agent行为规范",
)

# 验证凭证
is_valid = issuer.verify(credential)
```

### 8.4 约柜合约

```python
from core.ark_covenant import ArkCovenant

ark = ArkCovenant(chain="simulated")

# 部署合约（自动M175封印）
spec_hash = ark.deploy("myAgent", "行为规范文本", escrow_tokens=100.0)

# 验收（M106）
result = ark.settle(spec_hash, delivery_hash, signer_did="did:opc:owner123")

# 罚没（M178）
result = ark.slash(spec_hash, penalty=50.0, signer_did="did:opc:owner123")
```

### 8.5 确权仪式

```python
from core.ratify_ritual import RatifyRitual, AgentSpec

ritual = RatifyRitual()

spec = AgentSpec(
    agent_name="myAgent",
    owner_did="did:opc:owner123",
    purpose="Taiji Session",
    capabilities=["text_generate", "web_browse"],
    boundaries=["no_system_modify", "no_sensitive_read"],
    constraints=["obey_gcd", "respect_ark"],
)

# Plan阶段
ritual.plan(spec)

# Consult阶段（模拟法务评审）
consultants = [{"name": "Legal Expert", "role": "legal"}]
ritual.consult(consultants)

# Ratify阶段（数字签名封印）
ritual.ratify(signature="digital_signature_here")
```

### 8.6 GCD归约算子

```python
from core.gcd_engine import GCDEngine, GcdRule

gcd = GCDEngine(mode="strict")

# 添加约束规则
rule = GcdRule(
    tool_name="browser.navigate",
    condition={"url": ".*"},
    pre_condition=lambda args: args.get("url", "").startswith("http"),
    post_condition=lambda result: result is not None,
    description="URL必须以http/https开头",
)
gcd.add_rule(rule)

# 校验执行流
is_valid, blocked_rule = gcd.verify_execution(execution_plan)
```

---

## 9. 故障排除

### 9.1 DeepSeek API连接失败

**症状**：`APIConnectionError`

**解决**：
1. 检查API Key是否正确
2. 检查网络连接
3. 检查`base_url`是否正确

### 9.2 浏览器云脑模式失败

**症状**：`PlaywrightException`

**解决**：
1. 安装浏览器驱动：`playwright install chromium`
2. 检查`config.yaml`中`browser.headless`设置
3. 查看`browser`依赖是否已安装：`pip show playwright`

### 9.3 Continuation恢复失败

**症状**：`ContinuationNotFound`

**解决**：
1. 检查`snapshot_dir`是否存在
2. 检查kid是否正确
3. 使用`python cli.py --status`查看可用Continuation

### 9.4 测试失败

**症状**：`pytest`运行失败

**解决**：
1. 检查是否安装所有依赖：`pip install -r requirements.txt`
2. 检查Python版本：需要3.10+
3. 查看具体错误信息：`pytest tests/ -v --tb=long`

### 9.5 硅基代理治理阻塞

**症状**：`Governance BLOCKED`

**解决**：
1. 检查是否完成确权仪式：`ritual.status()`
2. 检查AIC凭证是否有效：`aic.verify()`
3. 检查GCD规则是否过于严格：修改`config.yaml`中`gcd.mode`为`loose`

---

## 附录：完整配置示例

见`config.yaml`文件。

---

**文档版本**: v1.0.0 (2026-06-04)
