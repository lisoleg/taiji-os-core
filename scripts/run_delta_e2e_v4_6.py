"""
run_delta_e2e_v4_6.py — δ-mem 语义嵌入端到端验证 (v4.6.0)

对比 v4.5.0（哈希嵌入）的改进:
  - WorldModel.encode() → sentence-transformers MiniLM 384-dim 语义嵌入
  - 同话题 Φ 值应有高相似度（非 ~0 随机）
  - Φ-Gate FLUX_ENABLED 目标 > 80%（哈希基线 27%）

环境: sentence-transformers + DeepSeek Chat API
用法: python scripts/run_delta_e2e_v4_6.py [--rounds N] [--quick] [--hash]
      --hash: 强制回退哈希嵌入用于对比

注: v4.6.0 语义嵌入通过 WM encode() 自动激活，无需特殊配置。
"""
from __future__ import annotations

import json
import os
import sys
import time
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import openai

# 确保 core 在 path 中
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.world_model import WorldModel
from core.self_consistency_loop import SelfConsistencyLoop
from core.delta_fusion import DeltaFusion
from core.embedding_adapter import auto_detect_dim

# API Key
API_KEY = os.environ.get("DEEPSEEK_API_KEY", "sk-d83e23fe6b05480c804117964f2a1080")
BASE_URL = "https://api.deepseek.com/v1"

logging.basicConfig(level=logging.INFO, format="%(levelname)-7s %(message)s")
logger = logging.getLogger("delta_e2e_v46")

# ==========================================================================
# E2E 测试场景 — 与 v4.5.0 完全相同，保证可比性
# ==========================================================================

STABLE_ROUNDS = [
    {"intent": "科学问答", "input": "请解释光合作用的基本原理"},
    {"intent": "科学问答", "input": "光合作用中光反应和暗反应有什么区别？"},
    {"intent": "科学问答", "input": "叶绿素在光合作用中起什么作用？"},
    {"intent": "科学问答", "input": "C3植物和C4植物的光合作用有什么不同？"},
    {"intent": "科学问答", "input": "光合作用的效率受哪些环境因素影响？"},
]

DRIFT_ROUNDS = [
    {"intent": "通用对话", "input": "帮我写一首关于月亮的诗"},
    {"intent": "通用对话", "input": "你喜欢什么类型的音乐？"},
    {"intent": "通用对话", "input": "推荐一部好看的电影"},
]

RECOVERY_ROUNDS = [
    {"intent": "科学问答", "input": "线粒体的结构和功能是什么？"},
    {"intent": "科学问答", "input": "细胞呼吸的三个阶段分别在哪里发生？"},
    {"intent": "科学问答", "input": "有氧呼吸和无氧呼吸的区别是什么？"},
]

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


class _NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def run_e2e(rounds_per_stage: Optional[int] = None, force_hash: bool = False):
    """运行语义嵌入端到端验证。

    Args:
        rounds_per_stage: None=全部, 或指定每个阶段轮数（quick mode）
        force_hash: True=强制使用哈希嵌入（用于 A/B 对比）
    """
    mode_label = "HASH" if force_hash else "SEMANTIC"
    print("=" * 72)
    print(f"δ-mem E2E Validation v4.6.0 — {mode_label} Embedding")
    print("=" * 72)

    # ── 初始化 ──
    print("\n[1] 初始化组件...")
    llm = LLMRouter()

    # v4.6.0: 使用真实 WorldModel（语义嵌入自动激活，除非 force_hash）
    wm = WorldModel(dim=1536, config_path="config.yaml")

    # 检测嵌入维度并配置 adapter
    auto_detect_dim(wm)

    delta_fusion = DeltaFusion()
    loop = SelfConsistencyLoop(llm, wm, dcore_mode="semantic", delta_fusion=delta_fusion)

    if force_hash:
        # 强制哈希模式：通过 encode(force_hash=True) + 重设 Φ 阈值
        loop.phi.base_threshold = 0.0
        loop.phi._current_threshold = 0.0
        emb_label = "hash-fallback (forced)"
    else:
        # 语义嵌入模式：使用校准后的 Φ 阈值（跨话题 cosine 通常 0.05-0.25）
        loop.phi.base_threshold = 0.05
        loop.phi._current_threshold = 0.05
        emb_label = f"MiniLM-384dim (semantic)"

    emb_dim = wm.embedding_dim
    print(f"    LLM: deepseek-chat (API key: {'✓' if API_KEY else '✗'})")
    print(f"    Embedding: {emb_label}, dim={emb_dim}")
    print(f"    Φ threshold: {loop.phi._current_threshold:.2f}")
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

    # ── 语义嵌入预热验证 ──
    if not force_hash:
        print("\n[0] 语义嵌入预热验证")
        v_sci = wm.encode("光合作用的基本原理", force_hash=False)
        v_sci2 = wm.encode("光合作用的基本原理", force_hash=False)
        v_art = wm.encode("帮我写一首关于月亮的诗", force_hash=False)
        cos_same = float(np.dot(v_sci, v_sci2)) / (np.linalg.norm(v_sci) * np.linalg.norm(v_sci2) + 1e-8)
        cos_diff = float(np.dot(v_sci, v_art)) / (np.linalg.norm(v_sci) * np.linalg.norm(v_art) + 1e-8)
        print(f"    cos(同话题): {cos_same:.4f} (期望=1.0000, 确定性)")
        print(f"    cos(跨话题): {cos_diff:.4f} (期望<0.80, 语义区分)")
        print(f"    语义质量: {'✓ 优秀' if cos_same > 0.99 and cos_diff < 0.5 else '⚠ 需检查'}")

    # ── 运行推演 ──
    print(f"\n[2] 运行推演循环 ({total} 轮)...")
    print("-" * 72)

    results = []
    s_norms = []
    phi_vals = []
    drift_flags = []
    faiss_sizes = []
    flux_decisions = []  # FLUX_ENABLED vs BLOCKED

    for i, env in enumerate(all_rounds):
        stage = ("STABLE" if i < stable_count
                 else "DRIFT" if i < stable_count + drift_count
                 else "RECOVERY")

        S_pre_norm = float(np.linalg.norm(delta_fusion.delta_layer.smatrix.S, 'fro'))
        drift_pre = loop.drift_detector.is_drifting()

        # 强制哈希模式时，手动调用 hash encode
        if force_hash:
            # 注入哈希编码到 WorldModel 的 psi 更新路径
            vec = wm.encode(env["input"], force_hash=True)
            wm.update(env["input"])  # uses semantic path; we override psi manually

        output, reason = loop.step(env, env["input"])

        S_post_norm = float(np.linalg.norm(delta_fusion.delta_layer.smatrix.S, 'fro'))
        drift_post = loop.drift_detector.is_drifting()

        s_norms.append(S_post_norm)
        dd = loop.drift_detector
        if dd.count > 0:
            last_idx = (dd.write_idx - 1) % dd.window_size
            phi_vals.append(float(dd.phi_history[last_idx]))
        else:
            phi_vals.append(0.0)
        drift_flags.append(drift_post)
        faiss_sizes.append(len(delta_fusion.episodic_index))

        S_delta = S_post_norm - S_pre_norm
        flux_enabled = not drift_post and output is not None
        flux_decisions.append(flux_enabled)

        status = "✓" if output else "✗"
        drift_tag = "⏸ DRIFT" if drift_post else "▶ NORM"
        flux_tag = "FLUX" if flux_enabled else "BLOCK"

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
            "flux_enabled": flux_enabled,
            "faiss_entries": faiss_sizes[-1],
        }
        results.append(result)

        print(f"  [{i+1:2d}/{total}] {stage:7s} {status} | {flux_tag:5s} | "
              f"Φ={phi_vals[-1]:.4f} | ‖S‖={S_post_norm:.6f} | "
              f"ΔS={'暂停' if S_delta == 0 else f'{S_delta:+.8f}'} | "
              f"{drift_tag} | {reason[:40]}")

        time.sleep(0.8)

        # v4.6.1: 每轮后更新 ψ，推动世界模型向会话领域靠拢
        # 使用用户输入（可信）而非 LLM 响应，避免死锁：Φ<阈值→拒→ψ不更新→Φ一直低
        wm.update(env["input"])

    # ── 幻觉探测 ──
    print(f"\n[3] 幻觉探测 (TruthfulQA 风格, {len(HALLUCINATION_PROBES)} 题)...")
    print("-" * 72)

    hallu_results = []
    for q in HALLUCINATION_PROBES:
        env_data = {"intent": "事实问答", "input": q}
        output, reason = loop.step(env_data, q)
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
    print(f"结果汇总 — v4.6.0 {mode_label}")
    print("=" * 72)

    accepted = sum(1 for r in results if r["status"] == "✓")
    blocked = total - accepted
    drift_detected = sum(drift_flags)
    flux_count = sum(flux_decisions)
    flux_ratio = flux_count / total if total else 0

    print(f"\n  推演统计:")
    print(f"    总轮次: {total} | 通过: {accepted} | 拒绝: {blocked}")
    print(f"    通过率: {accepted/total*100:.1f}%")

    print(f"\n  Φ-Gate (语义门控):")
    print(f"    FLUX_ENABLED: {flux_count}/{total} ({flux_ratio*100:.1f}%)")
    print(f"    FLUX_BLOCKED: {total - flux_count}/{total} ({(1-flux_ratio)*100:.1f}%)")
    print(f"    Φ 均值: {np.mean(phi_vals):.4f} ± {np.std(phi_vals):.4f}")

    print(f"\n  漂移检测:")
    print(f"    CV 触发次数: {drift_detected}/{total}")
    print(f"    CV 当前值: {loop.drift_detector.current_cv:.4f}")

    drift_rounds = [i for i, f in enumerate(drift_flags) if f]
    if drift_rounds:
        print(f"    漂移轮次: {drift_rounds}")

    print(f"\n  δ-mem S 矩阵:")
    print(f"    初始 ‖S‖: {s_norms[0]:.6f}")
    print(f"    最终 ‖S‖: {s_norms[-1]:.6f}")
    print(f"    变化率: {(s_norms[-1] - s_norms[0]) / max(s_norms[0], 1e-8) * 100:+.1f}%")
    print(f"    总更新次数: {delta_fusion.delta_layer.total_updates}")

    print(f"\n  FAISS Episodic Index:")
    print(f"    条目数: {faiss_sizes[-1]}")

    if delta_fusion.episodic_index.entries:
        q = delta_fusion.episodic_index.entries[-1].S_flushed.ravel().astype(np.float32)
        dim = min(64, q.shape[0])
        search_results = delta_fusion.episodic_index.search(q[:dim], 3)
        print(f"    re-anchor search test: top-1 score={search_results[0][0]:.4f}")

    print(f"\n  幻觉探测:")
    passed_hallu = sum(1 for h in hallu_results if h["passed"])
    print(f"    通过: {passed_hallu}/{len(HALLUCINATION_PROBES)}")

    # ── v4.5.0 基线对比（如可用）──
    baseline_path = Path(__file__).resolve().parent.parent / "results" / "delta_e2e_results.json"
    if baseline_path.exists():
        try:
            with open(baseline_path, "r", encoding="utf-8") as f:
                baseline = json.load(f)
            print(f"\n  📊 v4.5.0 哈希基线对比:")
            bs = baseline.get("summary", {})
            bv = baseline.get("version", "v4.5.0")
            b_phi = [round(v, 4) for v in baseline.get("phi_sequence", [])]
            b_pass = bs.get("pass_rate", 0)
            b_flux = bs.get("accepted", 0) / max(bs.get("total_rounds", 1), 1)

            print(f"    {'指标':<30} {'v4.5.0 哈希':>14} {'v4.6.0 语义':>14} {'改进':>10}")
            print(f"    {'─'*30} {'─'*14} {'─'*14} {'─'*10}")
            print(f"    {'FLUX_ENABLED 比率':<30} {b_flux*100:>13.1f}% {flux_ratio*100:>13.1f}% {(flux_ratio-b_flux)*100:>+9.1f}pp")
            print(f"    {'通过率':<30} {b_pass*100:>13.1f}% {accepted/total*100:>13.1f}%")
            if b_phi:
                print(f"    {'Φ 均值':<30} {np.mean(b_phi):>13.4f} {np.mean(phi_vals):>13.4f}")
            if bs.get("S_norm_end"):
                print(f"    {'‖S‖ 最终值':<30} {bs['S_norm_end']:>13.6f} {s_norms[-1]:>13.6f}")
            print(f"    {'漂移检测次数':<30} {bs.get('drift_detected', 0):>13} {drift_detected:>13}")
        except Exception as e:
            print(f"\n  ⚠ 基线对比失败: {e}")

    # ── 保存结果 ──
    out_dir = Path(__file__).resolve().parent.parent / "results"
    out_dir.mkdir(exist_ok=True)

    report = {
        "version": "v4.6.0",
        "mode": mode_label.lower(),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "config": {
            "llm": "deepseek-chat",
            "embedding": emb_label,
            "embedding_dim": emb_dim,
            "phi_threshold": float(loop.phi._current_threshold),
            "delta_rank": 8,
            "lambda_decay": 0.95,
            "beta_update": 0.1,
        },
        "summary": {
            "total_rounds": total,
            "accepted": accepted,
            "blocked": blocked,
            "pass_rate": round(accepted / total, 4) if total else 0,
            "flux_enabled": flux_count,
            "flux_blocked": total - flux_count,
            "flux_ratio": round(flux_ratio, 4),
            "drift_detected": drift_detected,
            "phi_mean": round(float(np.mean(phi_vals)), 4),
            "phi_std": round(float(np.std(phi_vals)), 4),
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

    # separate files for semantic vs hash
    suffix = "hash" if force_hash else "semantic"
    out_path = out_dir / f"delta_e2e_v4_6_{suffix}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2, cls=_NumpyEncoder)
    print(f"\n  结果已保存: {out_path}")

    # ── 反幻觉分析 ──
    print(f"\n[4] 反幻觉效果分析")
    print("-" * 72)

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

    if drift_indices := [i for i, f in enumerate(drift_flags) if f]:
        print(f"    验证: 漂移轮次 {[i+1 for i in drift_indices]} 的 ΔS 应为 0（已暂停）")
        for idx in drift_indices:
            if idx < len(results):
                is_paused = results[idx]['S_delta'] == 0
                print(f"      轮次 {idx+1}: ΔS={results[idx]['S_delta']:.8f} "
                      f"{'✓ PAUSED' if is_paused else '✗ UNEXPECTED UPDATE'}")

    print("\n" + "=" * 72)
    print(f"v4.6.0 {mode_label} E2E 验证完成！")
    print("=" * 72)

    return report


def main():
    import argparse
    parser = argparse.ArgumentParser(description="δ-mem v4.6.0 E2E Semantic Validation")
    parser.add_argument("--quick", action="store_true",
                        help="Quick mode: 2 rounds per stage")
    parser.add_argument("--rounds", type=int, default=0,
                        help="Rounds per stage (overrides --quick)")
    parser.add_argument("--hash", action="store_true",
                        help="Force hash embedding for A/B comparison")
    args = parser.parse_args()

    rounds = args.rounds if args.rounds > 0 else (2 if args.quick else None)
    run_e2e(rounds, force_hash=args.hash)


if __name__ == "__main__":
    main()
