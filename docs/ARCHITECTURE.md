# 太极OS (Taiji OS) 架构文档

> **Taiji OS Architecture** — Module Dependencies, Data Flow, Five-Layer Penetration

---

## 目录

1. [架构概览](#1-架构概览)
2. [模块依赖图](#2-模块依赖图)
3. [数据流](#3-数据流)
4. [五层次穿透架构详解](#4-五层次穿透架构详解)
5. [三旋治理流程](#5-三旋治理流程)
6. [Walrus Memory proof链](#6-walrus-memory-proof链)
7. [MCP Bridge协议](#7-mcp-bridge协议)

---

## 1. 架构概览

太极OS采用**分层架构** + **事件驱动**设计：

```
┌──────────────────────────────────────────────────────────┐
│                    用户接口层                          │
│  cli.py / api/server.py / syscalls/mcp_bridge.py     │
└────────────────────┬─────────────────────────────────────┘
                     │
┌────────────────────▼─────────────────────────────────────┐
│                 核心控制层                                │
│  core/session.py (TaijiSession v4)                     │
│  - 五层次穿透架构 (L1-L5)                             │
│  - 三旋治理 (TriSpinGovernor)                         │
│  - 确权仪式 (RatifyRitual)                            │
└────────────────────┬─────────────────────────────────────┘
                     │
        ┌────────────┼────────────┐
        │            │            │
┌───────▼─────┐ ┌──▼──────┐ ┌──▼──────┐
│ 世界模型层    │ │ 推演层    │ │ 记忆层    │
│ WorldModel   │ │ GAN      │ │ MemoryHub │
│ WebWorldModel│ │          │ │          │
└───────┬─────┘ └──┬──────┘ └──┬──────┘
        │            │            │
┌───────▼────────────▼────────────▼─────┐
│              执行层                     │
│  Executor / PlaywrightExecutor         │
│  Planner / WebPlanner                 │
└───────┬───────────────────────────────┘
        │
┌───────▼───────────────────────────────┐
│          HAL层 (LLM Router)           │
│  hal/llm_router.py                   │
└───────────────────────────────────────┘
```

---

## 2. 模块依赖图

### 2.1 核心模块依赖

```
TaijiSession (core/session.py)
├── WorldModel (core/world_model.py)
│   └── DeepSeek Embedding API
├── WebWorldModel (core/web_world_model.py)
│   └── Playwright
├── CarbonSiliconGAN (core/carbon_silicon_gan.py)
│   ├── LLMRouter (hal/llm_router.py)
│   └── WorldModel (core/world_model.py)
├── Continuation (core/continuation.py)
│   └── numpy
├── MemoryHub (core/memory_hub.py)
│   └── Continuation (core/continuation.py)
├── TriSpinGovernor (core/tri_spin_governor.py)
│   ├── ACIssuer (core/aic.py)
│   ├── ArkCovenant (core/ark_covenant.py)
│   └── GCDEngine (core/gcd_engine.py)
├── FiveLayerPipeline (core/five_layer_architecture.py)
│   └── TriSpinGovernor (core/tri_spin_governor.py)
├── RatifyRitual (core/ratify_ritual.py)
│   └── ACIssuer (core/aic.py)
├── LLMRouter (hal/llm_router.py)
│   └── DeepSeek API / Claude API
├── Executor (syscalls/executor.py)
├── PlaywrightExecutor (syscalls/browser_executor.py)
│   └── Playwright
├── Planner (syscalls/planner.py)
├── WebPlanner (syscalls/web_planner.py)
└── MCPBridge (syscalls/mcp_bridge.py)
    ├── TaijiSession (core/session.py)
    └── MemoryHub (core/memory_hub.py)
```

### 2.2 测试模块依赖

```
tests/
├── test_hdr.py                 → Continuation, PhiScheduler
├── test_scs.py                 → WorldModel, Continuation
├── test_migration.py           → Continuation
├── test_csg.py                 → CarbonSiliconGAN
├── test_walrus_memory.py       → Continuation v2, MemoryHub
├── test_silicon_governance.py → AIC, GCD, Ark, TriSpin, Ratify, Pipeline
├── test_web_session.py         → WebWorldModel, PlaywrightExecutor
└── test_swebench.py            → WebPlanner
```

---

## 3. 数据流

### 3.1 文本推演数据流

```
用户输入
    │
    ▼
TaijiSession.run()
    │
    ├─→ ClosureEnv.push("user", input)
    │
    ├─→ WorldModel.update(psi, input)
    │
    ├─→ CarbonSiliconGAN.step(env, input)
    │   ├─→ G-Core: LLM生成候选响应
    │   └─→ D-Core: 矛盾检测 + Φ-Scheduler过滤
    │
    ├─→ PhiScheduler.check(phi)
    │   ├─→ phi >= threshold → 返回响应
    │   └─→ phi < threshold  → _save_continuation()
    │       │
    │       └─→ Continuation.save()
    │           └─→ 快照到 snapshots/*.json
    │
    └─→ ClosureEnv.push("assistant", output)
```

### 3.2 Web推演数据流

```
用户输入
    │
    ▼
TaijiSession.run()
    │
    ├─→ PlaywrightExecutor.execute("navigate", url)
    │   └─→ Playwright自动化
    │
    ├─→ WebWorldModel.observe_page()
    │   ├─→ DOM向量化
    │   ├─→ URL向量化
    │   └─→ 截图向量化
    │
    ├─→ WebPlanner.plan(intent, page_state)
    │   └─→ 规则层 + LLM层规划
    │
    └─→ 返回执行结果
```

### 3.3 Walrus Memory数据流

```
Continuation保存
    │
    ▼
TaijiSession._save_continuation()
    │
    ├─→ Continuation.save()
    │   └─→ 快照到 snapshots/*.json (含proof链)
    │
    └─→ MemoryHub.store(continuation)
        ├─→ 注册到 shared_memory/
        ├─→ 生成proof链
        └─→ 写入磁盘
```

### 3.4 五层次穿透数据流

```
TaijiSession.run()
    │
    ▼
FiveLayerPipeline.execute()
    │
    ├─→ L1: FtelIntentCapture.capture()
    │   └─→ 意图捕获 + φ度量
    │
    ├─→ L2: M175Shell.anchor()
    │   ├─→ AIC锚定
    │   └─→ 归责校验
    │
    ├─→ L3: GCDTopologyFlow.verify()
    │   ├─→ Pre条件校验
    │   └─→ Post条件校验
    │
    ├─→ L4: IDOICEAdjudicator.adjudicate()
    │   ├─→ M106验收
    │   └─→ M178罚没
    │
    └─→ L5: PhenomenonRenderer.render()
        └─→ 交付物 + 审计追踪
```

---

## 4. 五层次穿透架构详解

### 4.1 L1: 流贯层 (Ftel)

**文件**: `core/five_layer_architecture.py` → `FtelIntentCapture`

**功能**：
- 意图捕获
- φ度量（余弦相似度）

**关键代码**：
```python
class FtelIntentCapture:
    def capture(self, intent: str, world_model: WorldModel) -> LayerResult:
        # 计算φ（余弦相似度）
        psi = world_model.get_psi()
        candidate_psi = self._encode_intent(intent)
        phi = self._cosine_similarity(psi, candidate_psi)

        return LayerResult(
            layer_name="L1_Ftel",
            status="passed" if phi >= 0.65 else "warning",
            output={"phi": phi, "intent": intent},
            next_layer_input={"intent": intent, "phi": phi},
        )
```

### 4.2 L2: 代数壳层 (M175)

**文件**: `core/five_layer_architecture.py` → `M175Shell`

**功能**：
- AIC锚定
- 归责校验

**关键代码**：
```python
class M175Shell:
    def anchor(self, agent_name: str, owner_did: str, spec_text: str) -> LayerResult:
        # M175锚定
        issuer = ACIssuer()
        credential = issuer.issue(
            agent_name=agent_name,
            owner_did=owner_did,
            capabilities=[],
            pi_spec=spec_text,
        )

        # 归责校验
        vacuum_risk = self._compute_vacuum_risk(credential)

        return LayerResult(
            layer_name="L2_M175",
            status="passed" if vacuum_risk == 0.0 else "blocked",
            output={"credential": credential, "vacuum_risk": vacuum_risk},
            next_layer_input={"credential": credential},
        )
```

### 4.3 L3: 拓扑流贯层 (GCD)

**文件**: `core/five_layer_architecture.py` → `GCDTopologyFlow`

**功能**：
- Pre条件校验
- Post条件校验

**关键代码**：
```python
class GCDTopologyFlow:
    def verify(self, execution_plan: list, gcd_engine: GCDEngine) -> LayerResult:
        # GCD校验
        is_valid, blocked_rule = gcd_engine.verify_execution(execution_plan)

        return LayerResult(
            layer_name="L3_GCD",
            status="passed" if is_valid else "blocked",
            output={"blocked_rule": blocked_rule} if not is_valid else {},
            next_layer_input={"execution_plan": execution_plan if is_valid else []},
        )
```

### 4.4 L4: IDO/ICE层

**文件**: `core/five_layer_architecture.py` → `IDOICEAdjudicator`

**功能**：
- M106验收
- M178罚没

**关键代码**：
```python
class IDOICEAdjudicator:
    def adjudicate(self, delivery: dict, ark: ArkCovenant, signer_did: str) -> LayerResult:
        # M106验收
        result = ark.settle(
            spec_hash=delivery.get("spec_hash"),
            delivery_hash=delivery.get("hash"),
            signer_did=signer_did,
        )

        return LayerResult(
            layer_name="L4_IDO_ICE",
            status="passed" if result["status"] == "settled" else "blocked",
            output=result,
            next_layer_input={"delivery": delivery},
        )
```

### 4.5 L5: 现象渲染层

**文件**: `core/five_layer_architecture.py` → `PhenomenonRenderer`

**功能**：
- 交付物输出
- 审计追踪

**关键代码**：
```python
class PhenomenonRenderer:
    def render(self, delivery: dict, session_id: str) -> LayerResult:
        # 渲染交付物
        output = self._format_output(delivery)

        # 审计追踪
        audit_log = {
            "session_id": session_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "delivery": delivery,
        }
        self._write_audit_log(audit_log)

        return LayerResult(
            layer_name="L5_Render",
            status="passed",
            output=output,
            next_layer_input={},
        )
```

---

## 5. 三旋治理流程

### 5.1 流程图

```
代理上线
    │
    ▼
┌───────────────────────────────────┐
│ Phase 1: 确权仪式 (RatifyRitual) │
│  - Plan: 明确边界                │
│  - Consult: 法务评审            │
│  - Ratify: 数字签名封印         │
└───────────────┬───────────────────┘
                │
                ▼
┌───────────────────────────────────┐
│ Phase 2: AIC凭证签发             │
│  - M175锚定                     │
│  - 同伦类哈希 H_h              │
│  - DID责任绑定                  │
└───────────────┬───────────────────┘
                │
                ▼
┌───────────────────────────────────┐
│ Phase 3: 三旋治理激活           │
│  - 情治: 主体认领              │
│  - 理治: 密码学锚定            │
│  - 法治: 行为归责              │
└───────────────┬───────────────────┘
                │
                ▼
┌───────────────────────────────────┐
│ Phase 4: 五层次穿透执行         │
│  L1 → L2 → L3 → L4 → L5      │
└───────────────┬───────────────────┘
                │
                ▼
┌───────────────────────────────────┐
│ Phase 5: 约柜合约结算           │
│  - M106验收 (成功)             │
│  - M178罚没 (失败)             │
└───────────────────────────────────┘
```

### 5.2 情治 (Consensus) 详解

**目标**：激活主体责任意识

**流程**：
1. 代理上线前，强制主体认领
2. 认识递归迭代验证（`recursive_check`）
3. 最多重试3次（`max_acknowledge_retry`）

**代码路径**：`core/tri_spin_governor.py` → `TriSpinGovernor.consensus_verify()`

### 5.3 理治 (Cryptography) 详解

**目标**：密码学锚定身份契约

**流程**：
1. 验证AIC凭证有效性
2. 验证M175封印
3. SHA-256 with ECDSA签名

**代码路径**：`core/tri_spin_governor.py` → `TriSpinGovernor.cryptography_anchor()`

### 5.4 法治 (Statute) 详解

**目标**：行为归责

**流程**：
1. 强制GCD约束
2. 执行后自动M106验收
3. 错误时自动M178罚没

**代码路径**：`core/tri_spin_governor.py` → `TriSpinGovernor.statute_complete()`

---

## 6. Walrus Memory proof链

### 6.1 proof链结构

```
Genesis (创世快照)
    │
    ▼
快照 #1: proof = SHA-256("" + data_1)
    │
    ▼
快照 #2: proof = SHA-256(proof_1 + data_2)
    │
    ▼
快照 #3: proof = SHA-256(proof_2 + data_3)
    │
    ▼
...
```

### 6.2 proof链验证

```python
# 验证单个快照
is_valid = Continuation.verify(kid="abc12345", snapshot_dir="snapshots")

# 验证所有快照
hub = MemoryHub(store_dir="memory_store")
all_valid = hub.verify_all()
```

### 6.3 篡改检测

```python
# 模拟篡改
snapshot_path = "snapshots/abc12345.json"
with open(snapshot_path, "r") as f:
    data = json.load(f)
data["env"]["history"].append({"role": "fake", "content": "Tampered!"})
with open(snapshot_path, "w") as f:
    json.dump(data, f)

# 验证（应返回False）
is_valid = Continuation.verify(kid="abc12345")
assert not is_valid  # ✅ 篡改被检测！
```

---

## 7. MCP Bridge协议

### 7.1 JSON-RPC协议

MCP Bridge使用**stdio JSON-RPC**协议：

**请求格式**：
```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "method": "tools/call",
  "params": {
    "name": "taiji.run",
    "arguments": {
      "query": "设计芯片"
    }
  }
}
```

**响应格式**：
```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "result": {
    "content": [
      {
        "type": "text",
        "text": "芯片设计方案..."
      }
    ]
  }
}
```

### 7.2 工具定义

| 工具名称 | 说明 | 参数 |
|-----------|------|------|
| `taiji.run` | 执行一轮推演 | `query: str` |
| `taiji.status` | 查询会话状态 | `sid: str` |
| `taiji.resume` | 从Continuation恢复 | `kid: str` |
| `taiji.memory_search` | 搜索共享记忆 | `q: str, top_k: int = 5` |
| `taiji.verify` | 验证记忆完整性 | `mid: str` |
| `taiji.list_sessions` | 列出已注册会话 | 无 |

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

## 附录：配置参考

见`config.yaml`。

---

**文档版本**: v1.0.0 (2026-06-04)
