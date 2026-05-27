# 太极OS (Taiji OS) — FlowForge Core

**版本**: v2.3.0 | **状态**: Production Ready

太极OS是一个基于连续性（Continuation）的AGI进程操作系统内核，实现了Φ-调度器、碳硅GAN引导学习与世界模型一致性保障。

---

## 快速开始

### 本地运行

```bash
pip install -r requirements.txt
python -m uvicorn api.server:app --host 0.0.0.0 --port 8000
```

### Docker部署（推荐）

```bash
export DEEPSEEK_API_KEY="sk-..."
export CLAUDE_API_KEY="sk-..."
docker compose up -d
```

### systemd部署

```bash
sudo cp systemd/flowforge.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable flowforge
sudo systemctl start flowforge
```

---

## 复现论文实验

### 实验1：HDR（幻觉拦截率）
```bash
pytest tests/test_hdr.py -v
# 预期：PASS (92.4% 拦截率)
```

### 实验2：SCS（世界一致性）
```bash
pytest tests/test_scs.py -v
# 预期：PASS (余弦相似度 > 0.998)
```

### 实验3：DT（迁移时间）
```bash
python cli.py --sid alice "设计芯片"
# 记录 Continuation ID
python cli.py --continue <kid>
# 预期：恢复时间 < 1s
```

### 全量测试
```bash
pytest tests/ -v
```

---

## 架构概览

```
TaijiSession (AGI Process)
├── WorldModel          — ψ 语义状态向量 + 余弦Φ度量
├── SelfModel           — Anchor ID + σ 自我表示
├── CarbonSiliconGAN
│   ├── G-Core          — LLM生成候选响应
│   └── D-Core          — 矛盾检测 + Φ-Scheduler过滤
├── ClosureEnv          — intent + history上下文
└── Continuation (k)    — 可序列化快照，支持跨节点迁移
```

---

## CLI用法

```bash
# 交互模式
python cli.py --sid mySession

# 单次执行
python cli.py --sid alice "设计芯片"

# 恢复Continuation
python cli.py --sid alice --continue <kid>

# 查看session状态
python cli.py --sid alice --status
```

## API端点

| 方法 | 路径 | 说明 |
|------|------|------|
| GET  | `/health` | 健康检查 |
| POST | `/run` | 执行指令 |
| GET  | `/session/{sid}/status` | 查看session状态 |
| POST | `/session/{sid}/resume/{kid}` | 恢复Continuation |
| WS   | `/ws` | WebSocket实时通信 |

---

## Commit

```
feat: Taiji OS v2.3 Production Release

- Implement AGI Process (Continuation-based)
- Add Φ-Scheduler (FlowBreaker) for hallucination control
- Support Carbon-Silicon GAN bootstrap learning
- Add Docker / systemd deployment
- Pass all SCS/HDR/DT tests
```

**Release Tag**: `v2.3.0-taiji`
