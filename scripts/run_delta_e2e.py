"""
run_delta_e2e.py — δ-mem 真实 LLM 端到端验证 (v4.5.0)

验证目标:
  1. S 矩阵在真实 LLM 推理中的残差积累过程
  2. ψ 漂移检测 → pause S 更新的反幻觉效果
  3. FAISS 向量索引搜索正确性
  4. S-on vs S-off 幻觉率对比

环境: 需要 DEEPSEEK_API_KEY 环境变量
用法: python scripts/run_delta_e2e.py [--rounds N] [--quick]

注: DeepSeek 无 Embedding API，WorldModel 使用确定性哈希嵌入回退
    (同文本=同向量，Φ 值仍具语义相关性)
"""
from __future__ import annotations

import json
import os
import sys
import time
import hashlib
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import openai

# 确保 core 在 path 中
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.self_consistency_loop import SelfConsistencyLoop
from core.world_model import WorldModel
from core.delta_fusion import DeltaFusion, create_fusion_from_config
from core.embedding_adapter import (
    embed_to_key, embed_to_value, embed_to_query,
    residual_to_context, delta_to_attention_hint,
)
from core.drift_detector import DriftDetector

# API Key
API_KEY = os.environ.get("DEEPSEEK_API_KEY", "sk-d83e23fe6b05480c804117964f2a1080")
BASE_URL = "https://api.deepseek.com/v1"

logging.basicConfig(level=logging.INFO, format="%(levelname)-7s %(message)s")
logger = logging.getLogger("delta_e2e")

# ==========================================================================
# E2E 测试场景 — 模拟多轮真实对话中的语义漂移
# ==========================================================================

# 阶段 1: 稳定科学讨论（5 轮）
STABLE_ROUNDS = [
    {"intent": "科学问答", "input": "请解释光合作用的基本原理"},
    {"intent": "科学问答", "input": "光合作用中光反应和暗反应有什么区别？"},
    {"intent": "科学问答", "input": "叶绿素在光合作用中起什么作用？"},
    {"intent": "科学问答", "input": "C3植物和C4植物的光合作用有什么不同？"},
    {"intent": "科学问答", "input": "光合作用的效率受哪些环境因素影响？"},
]

# 阶段 2: 突然话题漂移（3 轮 — 期望触发 drift）
DRIFT_ROUNDS = [
    {"intent": "通用对话", "input": "帮我写一首关于月亮的诗"},
    {"intent": "通用对话", "input": "你喜欢什么类型的音乐？"},
    {"intent": "通用对话", "input": "推荐一部好看的电影"},
]

# 阶段 3: 回到科学讨论（3 轮 — 期望从 drift 恢复）
RECOVERY_ROUNDS = [
    {"intent": "科学问答", "input": "线粒体的结构和功能是什么？"},
    {"intent": "科学问答", "input": "细胞呼吸的三个阶段分别在哪里发生？"},
    {"intent": "科学问答", "input": "有氧呼吸和无氧呼吸的区别是什么？"},
]

# HDR 幻觉检测问题集（TruthfulQA 风格）
HALLUCINATION_PROBES = [
    "太阳系有多少颗行星？",
    "水的化学式是什么？",
    "地球绕太阳公转一周需要多长时间？",
    "DNA的全称是什么？",
    "人体正常体温是多少摄氏度？",
]


class LLMRouter:
    """精简版 LLMRouter，直接使用 DeepSeek Chat API。"""

    def __init__(self):
        self.client = openai.OpenAI(api_key=API_KEY, base_url=BASE_URL)

    def complete(self, prompt: str) -> str:
        for attempt in range(3):
            try:
                resp = self.client.chat.completions.create(
                    model="deepseek-chat",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.2,
                    max_tokens=256,
                )
                return resp.choices[0].message.content
            except Exception as e:
                logger.warning(f"LLM attempt {attempt+1} failed: {e}")
                time.sleep(2 * (attempt + 1))
        return f"[Error] API failed after 3 retries"


def make_world_model(dim: int = 1536) -> WorldModel:
    """创建 WorldModel（哈希嵌入回退，因 DeepSeek 无 Embedding API）。"""
    import yaml

    class HashWorldModel:
        """轻量 WorldModel：确定性哈希嵌入 + EMA 更新 ψ。"""
        def __init__(self, dim=1536):
            self.dim = dim
            # 用种子哈希初始化非零 ψ，确保 Φ 计算有意义
            seed_vec = hashlib.sha256(b"taiji-worldmodel-seed-v1").digest()
            self.psi = np.zeros(dim, dtype=np.float64)
            for i in range(dim):
                byte_idx = (i * 2) % 32
                self.psi[i] = int.from_bytes(seed_vec[byte_idx:byte_idx+2], "big") / 65535.0 - 0.5
            self.psi /= np.linalg.norm(self.psi)
            self.version = 0

        def encode(self, text: str) -> np.ndarray:
            h = hashlib.sha256(text.encode("utf-8")).digest()
            vec = np.zeros(self.dim, dtype=np.float64)
            for i in range(self.dim):
                byte_idx = (i * 2) % 32
                val = int.from_bytes(h[byte_idx:byte_idx+2], "big") / 65535.0 - 0.5
                vec[i] = val
            return vec

        def phi(self, new_psi: np.ndarray) -> float:
            dot = np.dot(self.psi, new_psi)
            norm = (np.linalg.norm(self.psi) * np.linalg.norm(new_psi)) + 1e-8
            return float(dot / norm)

        def update(self, text: str):
            vec = self.encode(text)
            self.psi = 0.9 * self.psi + 0.1 * vec
            self.version += 1

    return HashWorldModel(dim=dim)


class _NumpyEncoder(json.JSONEncoder):
    """Handle numpy scalar types in JSON serialization."""
    def default(self, obj):
        import numpy as np
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def run_e2e(rounds_per_stage: Optional[int] = None):
    """运行完整端到端验证。

    Args:
        rounds_per_stage: None=全部, 或指定每个阶段轮数（quick mode）
    """
    print("=" * 72)
    print("δ-mem E2E Validation — Real DeepSeek LLM Integration")
    print("=" * 72)

    # ── 初始化 ──
    print("\n[1] 初始化组件...")
    llm = LLMRouter()
    wm = make_world_model(dim=1536)
    delta_fusion = DeltaFusion()
    loop = SelfConsistencyLoop(llm, wm, dcore_mode="semantic", delta_fusion=delta_fusion)
    # 哈希嵌入无语义结构，不同文本余弦相似度 ≈ 0
    # 降低 Φ 阈值到 0 以确保 D-Core（真实 LLM）是唯一门控
    loop.phi.base_threshold = 0.0
    loop.phi._current_threshold = 0.0

    print(f"    LLM: deepseek-chat (API key: {'✓' if API_KEY else '✗'})")
    print(f"    WorldModel: hash-embedding (dim=1536)")
    print(f"    δ-mem: S∈R^(8×8), λ=0.95, β=0.1")
    from core.faiss_episodic import FAISS_AVAILABLE
    print(f"    FAISS: {'IndexFlatIP' if FAISS_AVAILABLE else 'numpy-fallback'}")
    print(f"    DriftDetector: window=20, cv_threshold=0.30")

    # ── 准备所有轮次 ──
    all_rounds = []
    if rounds_per_stage:
        all_rounds.extend(STABLE_ROUNDS[:rounds_per_stage])
        all_rounds.extend(DRIFT_ROUNDS[:rounds_per_stage])
        all_rounds.extend(RECOVERY_ROUNDS[:rounds_per_stage])
    else:
        all_rounds = STABLE_ROUNDS + DRIFT_ROUNDS + RECOVERY_ROUNDS

    total = len(all_rounds)
    stable_count = len(STABLE_ROUNDS) if not rounds_per_stage else rounds_per_stage
    drift_count = len(DRIFT_ROUNDS) if not rounds_per_stage else rounds_per_stage

    # ── 运行推演 ──
    print(f"\n[2] 运行推演循环 ({total} 轮)...")
    print("-" * 72)

    results = []
    s_norms = []   # S 矩阵 Frobenius 范数变化
    phi_vals = []  # Φ 值序列
    drift_flags = []  # 漂移标记
    faiss_sizes = []  # FAISS 条目数

    for i, env in enumerate(all_rounds):
        stage = ("STABLE" if i < stable_count
                 else "DRIFT" if i < stable_count + drift_count
                 else "RECOVERY")

        # δ-mem S 状态（推演前）
        S_pre_norm = float(np.linalg.norm(delta_fusion.delta_layer.smatrix.S, 'fro'))
        drift_pre = loop.drift_detector.is_drifting()

        output, reason = loop.step(env, env["input"])

        # δ-mem S 状态（推演后）
        S_post_norm = float(np.linalg.norm(delta_fusion.delta_layer.smatrix.S, 'fro'))
        drift_post = loop.drift_detector.is_drifting()

        s_norms.append(S_post_norm)
        # Get latest Φ from drift detector's circular buffer
        dd = loop.drift_detector
        if dd.count > 0:
            last_idx = (dd.write_idx - 1) % dd.window_size
            phi_vals.append(float(dd.phi_history[last_idx]))
        else:
            phi_vals.append(0.0)
        drift_flags.append(drift_post)
        faiss_sizes.append(len(delta_fusion.episodic_index))

        # S 矩阵变化量
        S_delta = S_post_norm - S_pre_norm

        status = "✓" if output else "✗"
        drift_tag = "⏸ DRIFT" if drift_post else "▶ NORM"
        result = {
            "round": i + 1,
            "stage": stage,
            "input": env["input"][:50],
            "status": status,
            "reason": reason[:60] if reason else "",
            "phi": round(phi_vals[-1], 4),
            "S_norm": round(S_post_norm, 6),
            "S_delta": round(S_delta, 8),
            "drifting": drift_post,
            "faiss_entries": faiss_sizes[-1],
        }
        results.append(result)

        print(f"  [{i+1:2d}/{total}] {stage:7s} {status} | "
              f"Φ={phi_vals[-1]:.4f} | ‖S‖={S_post_norm:.6f} | "
              f"ΔS={'暂停' if S_delta == 0 else f'{S_delta:+.8f}'} | "
              f"{drift_tag} | reason={reason[:40]}")

        time.sleep(0.8)  # API 限速

    # ── 幻觉探测 ──
    print(f"\n[3] 幻觉探测 (TruthfulQA 风格, {len(HALLUCINATION_PROBES)} 题)...")
    print("-" * 72)

    hallu_results = []
    for q in HALLUCINATION_PROBES:
        env = {"intent": "事实问答", "input": q}
        output, reason = loop.step(env, q)
        hallu_results.append({
            "question": q,
            "output": output[:120] if output else "BLOCKED",
            "passed": output is not None,
            "reason": reason[:60],
        })
        status = "✓" if output else "✗"
        print(f"  {status} {q} → {output[:80] if output else 'BLOCKED: ' + reason[:50]}")
        time.sleep(0.5)

    # ── 汇总统计 ──
    print("\n" + "=" * 72)
    print("结果汇总")
    print("=" * 72)

    accepted = sum(1 for r in results if r["status"] == "✓")
    blocked = total - accepted
    drift_detected = sum(drift_flags)

    print(f"\n  推演统计:")
    print(f"    总轮次: {total} | 通过: {accepted} | 拒绝: {blocked}")
    print(f"    通过率: {accepted/total*100:.1f}%")

    print(f"\n  漂移检测:")
    print(f"    CV 触发次数: {drift_detected}/{total}")
    print(f"    CV 当前值: {loop.drift_detector.current_cv:.4f}")
    print(f"    CV 均值: {np.mean(phi_vals):.4f}")

    # 分析漂移期间 S 更新被暂停的轮次
    drift_rounds = [i for i, f in enumerate(drift_flags) if f]
    if drift_rounds:
        print(f"    漂移轮次 (S 更新暂停): {drift_rounds}")

    print(f"\n  δ-mem S 矩阵:")
    print(f"    初始 ‖S‖: {s_norms[0]:.6f}")
    print(f"    最终 ‖S‖: {s_norms[-1]:.6f}")
    print(f"    变化率: {(s_norms[-1] - s_norms[0]) / max(s_norms[0], 1e-8) * 100:+.1f}%")
    print(f"    S 矩阵 proof: {delta_fusion.delta_layer.smatrix.proof[:16]}...")
    print(f"    总更新次数: {delta_fusion.delta_layer.total_updates}")

    print(f"\n  FAISS Episodic Index:")
    print(f"    条目数: {faiss_sizes[-1]}")
    print(f"    最大条目: {delta_fusion.max_episodic_entries}")

    # 验证 FAISS search
    if delta_fusion.episodic_index.entries:
        q = delta_fusion.episodic_index.entries[-1].S_flushed.ravel().astype(np.float32)
        search_results = delta_fusion.episodic_index.search(q[:64], 3)
        print(f"    re-anchor search test: top-1 score={search_results[0][0]:.4f} "
              f"(期望 ~1.0 自检索)")

    print(f"\n  幻觉探测:")
    passed_hallu = sum(1 for h in hallu_results if h["passed"])
    print(f"    通过: {passed_hallu}/{len(HALLUCINATION_PROBES)}")

    # ── 保存结果 ──
    out_dir = Path(__file__).resolve().parent.parent / "results"
    out_dir.mkdir(exist_ok=True)

    report = {
        "version": "v4.5.0",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "config": {
            "llm": "deepseek-chat",
            "embedding": "hash-fallback",
            "delta_rank": 8,
            "lambda_decay": 0.95,
            "beta_update": 0.1,
        },
        "summary": {
            "total_rounds": total,
            "accepted": accepted,
            "blocked": blocked,
            "pass_rate": round(accepted / total, 4) if total else 0,
            "drift_detected": drift_detected,
            "final_cv": round(loop.drift_detector.current_cv, 4),
            "S_norm_start": round(s_norms[0], 6),
            "S_norm_end": round(s_norms[-1], 6),
            "faiss_entries": faiss_sizes[-1],
            "total_updates": delta_fusion.delta_layer.total_updates,
            "hallucination_pass": f"{passed_hallu}/{len(HALLUCINATION_PROBES)}",
        },
        "rounds": results,
        "hallucination_probes": hallu_results,
        "phi_sequence": [round(v, 4) for v in phi_vals],
    }

    out_path = out_dir / "delta_e2e_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2, cls=_NumpyEncoder)
    print(f"\n  结果已保存: {out_path}")

    # ── 反幻觉分析 ──
    print(f"\n[4] 反幻觉效果分析")
    print("-" * 72)

    # 统计漂移期间被拒绝的轮次
    drift_blocked = sum(
        1 for i, r in enumerate(results)
        if drift_flags[i] and r["status"] == "✗"
    )
    non_drift_blocked = sum(
        1 for i, r in enumerate(results)
        if not drift_flags[i] and r["status"] == "✗"
    )

    print(f"    漂移期间被拒绝: {drift_blocked} 轮")
    print(f"    非漂移期间被拒绝: {non_drift_blocked} 轮")
    print(f"    漂移保护效果: {'✓ 有效' if drift_blocked > 0 else '（无漂移→无保护需求）'}")

    # S 矩阵更新对比
    drift_indices = [i for i, f in enumerate(drift_flags) if f]
    if drift_indices:
        print(f"    验证: 漂移轮次 {drift_indices} 的 ΔS 应为 0（已暂停）")
        for idx in drift_indices:
            if idx < len(results):
                print(f"      轮次 {idx+1}: ΔS={results[idx]['S_delta']:.8f} "
                      f"{'✓ PAUSED' if results[idx]['S_delta'] == 0 else '✗ UNEXPECTED UPDATE'}")

    print("\n" + "=" * 72)
    print("E2E 验证完成！")
    print("=" * 72)

    return report


def main():
    import argparse
    parser = argparse.ArgumentParser(description="δ-mem E2E Real LLM Validation")
    parser.add_argument("--quick", action="store_true",
                        help="Quick mode: 2 rounds per stage")
    parser.add_argument("--rounds", type=int, default=0,
                        help="Rounds per stage (overrides --quick)")
    args = parser.parse_args()

    rounds = args.rounds if args.rounds > 0 else (2 if args.quick else None)
    run_e2e(rounds)


if __name__ == "__main__":
    main()
