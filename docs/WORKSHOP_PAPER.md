# Taiji OS: An AGI Agent Runtime with Continuation-based Process Model

> **Workshop Paper (4 pages)** — Target: HotOS Workshop / EuroSys Workshop / MLSys Demo Track

---

## Abstract

Taiji OS is an **AGI Agent Runtime** that introduces a **Continuation-based process model**
for long-running AI agents. Unlike stateless API wrappers, Taiji OS treats each agent session
as a serializable computational process with snapshot/restore, integrity proofs, and cross-session
memory sharing. The system supports three governance layers — consensus, cryptography, and
statute — forming a five-layer penetration architecture. We present the design motivation,
engineering implementation, and open challenges.

**Key contributions:**
1. Continuation v2 with SHA-256 proof chains for agent state portability
2. Walrus Memory: shared memory space with batch integrity verification
3. Silicon Agent Governance: a five-layer architecture for agent accountability
4. MCP-native bridge exposing agent lifecycle as standard tools

---

## 1. Introduction

Modern AI agent frameworks (LangChain, AutoGPT, CrewAI) are predominantly **stateless**
— each interaction starts fresh, with no persistent compute state, no integrity guarantees,
and no accountability for agent actions. This statelessness creates three problems:

1. **No recoverability**: Agent reasoning cannot be suspended and resumed without replaying the full history.
2. **No memory integrity**: Cross-session memory has no cryptographic proof that it hasn't been tampered with.
3. **No accountability**: When an agent makes an error, there is no clear attribution of responsibility.

Taiji OS addresses these by treating each agent session as a **Continuation** — a concept
borrowed from programming language theory (call/cc) and operating system process models.
Each Continuation captures not just the conversation history, but the full semantic state
(ψ vector), environment context, and a cryptographic proof linking it to prior states.

### Design Motivation (not "Formal Proofs")

The current implementation is a **design exploration**, not a formally verified system.
Our contributions are engineering: demonstrating that OS-inspired abstractions
(processes, snapshots, integrity proofs) can meaningfully improve agent reliability.
Where prior drafts claimed "formal proofs" or "theorems," this paper replaces them with
"design arguments" and "implemented mechanisms."

---

## 2. System Design

### 2.1 Continuation v2: Portable Agent State

Each Continuation is a JSON snapshot containing:
```
{kid, sid, psi[N], env: dict, reason: str, proof: SHA-256, parent_kid: str|None}
```

The `proof` field forms a hash chain: `proof_i = SHA-256(proof_{i-1} || data_i)`.
This enables integrity verification across an agent's entire lifecycle.

The `parent_kid` field creates a memory graph, enabling traversal of an agent's
decision history.

**Current scope:** Single-node snapshot/restore. Distributed migration is design discussion
(not implemented).

### 2.2 Self-Consistency Loop (formerly "CarbonSiliconGAN")

The core reasoning loop has two components:

- **G-Core (Generator):** LLM generates a candidate response.
- **D-Core (Discriminator):** Performs semantic contradiction detection using a
  zero-shot prompt (`VERDICT: CONTRADICTION/CONSISTENT`), with keyword fallback.
  This is a **semantic consistency check**, not an adversarial GAN training loop.

The Φ-gate then checks cosine similarity of the candidate's embedding against the
world model's current ψ vector. A sliding-window adaptive threshold mode
(`PhiScheduler(mode="adaptive")`) adjusts the gate based on semantic volatility.

### 2.3 Walrus Memory: Shared Cross-Session Memory

Inspired by the Walrus Memory concept (portable memory + integrity proofs + shared
spaces), MemoryHub provides:

- **register(sid):** Register a new session.
- **store(sid, record):** Store a memory record with proof chain.
- **search(query):** Keyword-based search across all sessions.
- **verify_all():** Batch integrity verification of all records.

### 2.4 Five-Layer Penetration Architecture

Inspired by the "Silicon Agent Governance" framework, the five layers are:

```
L1 流贯 (Flow)          Intent capture + semantic embedding
L2 代数壳 (Algebraic)   Agent Identity Credential (AIC) anchoring
L3 拓扑流贯 (Topology)  GCD constraint validation (Pre/Post)
L4 IDO/ICE (Adjudication) Covenant settlement (accept/slash)
L5 渲染 (Rendering)     Deliverable output + audit trail
```

### 2.5 MCP Bridge

Standard MCP stdio JSON-RPC bridge exposing 6 tools:
`taiji.run`, `taiji.status`, `taiji.resume`, `taiji.memory_search`,
`taiji.verify`, `taiji.list_sessions`.

---

## 3. Implementation

| Module | LOC | Description |
|--------|-----|-------------|
| `core/continuation.py` | 80 | Continuation v2 with proof chains |
| `core/self_consistency_loop.py` | 140 | Semantic contradiction detection |
| `core/phi_scheduler.py` | 110 | Static/adaptive coherence gating |
| `core/world_model.py` | 120 | DeepSeek Embedding API + hash fallback |
| `core/memory_hub.py` | 150 | Cross-session shared memory |
| `core/session.py` | 420 | Agent process lifecycle |
| `syscalls/mcp_bridge.py` | 130 | MCP stdio bridge |
| Governance modules | 800 | AIC, GCD, Ark Covenant, Tri-spin, etc. |

**Dependencies:** `numpy`, `openai`, `pyyaml`. No torch/transformers (zero heavy ML deps).

---

## 4. Evaluation

### 4.1 Standardized Benchmarks (v4.1)

We provide reproducible benchmark scripts:

- `scripts/benchmark_hdr.py`: Hallucination Detection Rate (HDR) with 25 standardized
  contradictory/consistent input pairs. Offline mode supported (no API needed).
- `scripts/benchmark_scs.py`: Semantic Coherence Score (SCS) with 25 consistency/drift pairs.

CI pipeline: Weekly GitHub Actions run (`benchmark.yml`).

**Current numbers:** [Pending standardized evaluation. Prior claims of 92.4% HDR, 0.998 SCS,
and 0.8s migration time were aspirational — v4.1 replaces them with reproducible benchmarks.]

### 4.2 Test Coverage

- 65 unit tests, 2 skipped (Playwright-dependent) — 97% pass rate
- 45 governance tests (AIC verification, GCD blocking, covenant integrity, accountability)
- 8 Walrus Memory tests (proof chains, tamper detection, cross-session search)

---

## 5. Related Work

| System | State Model | Integrity Proofs | Agent Governance |
|--------|------------|-----------------|------------------|
| LangChain | Stateless | None | None |
| AutoGPT | File-based memory | None | None |
| CrewAI | In-memory only | None | None |
| Anthropic MCP | Tool protocol only | None | None |
| **Taiji OS** | **Continuation + proof chain** | **SHA-256 chain** | **5-layer architecture** |

The MCP Bridge is a unique differentiator — Taiji OS is the first agent runtime to
expose its full lifecycle (run/resume/verify) as standard MCP tools, enabling
integration with Claude Desktop and other MCP-compatible clients.

---

## 6. Limitations and Future Work

1. **Benchmark validation:** HDR/SCS numbers need standardized reproduction.
2. **No distributed migration:** Continuation is single-node only.
3. **No KV Cache persistence:** Resume requires full re-prefill (vLLM integration planned).
4. **Governance simulation:** Ark Covenant and OPC registry use mock blockchains.
5. **Scalability:** MemoryHub is file-based; needs distributed backend for production.

---

## 7. Conclusion

Taiji OS demonstrates that **OS-inspired abstractions** (processes, snapshots, integrity proofs)
can bring meaningful reliability improvements to AI agent systems. The v4.1 release makes
the codebase honest: removing unsubstantiated claims, adding reproducible benchmarks, and
renaming modules to reflect what they actually do.

**Repository:** [github.com/lisoleg/taiji-os-core](https://github.com/lisoleg/taiji-os-core)
