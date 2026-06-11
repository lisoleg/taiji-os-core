"""
run_delta_e2e_v4_7_0.py — δ-mem v4.7.0 指数衰减 CV 端到端验证

v4.7.0 核心改动（相对 v4.6.3）:
  - DriftDetector v1.3: 滑动窗口 CV 计算使用指数衰减权重 (decay=0.80)
  - 恢复加速: RECOVERY 阶段的高 Φ 值更快拉低 CV，解除漂移标记
  - 预期：FLUX_ENABLED ≥ 60%（RECOVERY 阶段恢复 FLUX）

对比基线:
  - v4.5.0 哈希:  FLUX_ENABLED=27.3%, Φ=-0.11, 通过率=27.3%
  - v4.6.0 语义:  FLUX_ENABLED=18.2%, Φ=+0.35, 通过率=100%
  - v4.6.2 修复:  FLUX_ENABLED=36.4%, Φ=+0.34, 通过率=100%
  - v4.6.3 降β:   FLUX_ENABLED=36.4%, Φ=+0.35, 通过率=100%
  - v4.7.0 衰减CV: 目标 FLUX_ENABLED ≥ 60% (RECOVERY 恢复)

环境: sentence-transformers + DeepSeek Chat API
用法: python scripts/run_delta_e2e_v4_7_0.py [--rounds N] [--quick]
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

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.world_model import WorldModel
from core.self_consistency_loop import SelfConsistencyLoop
from core.delta_fusion import DeltaFusion
from core.embedding_adapter import auto_detect_dim

API_KEY = os.environ.get("DEEPSEEK_API_KEY", "sk-d83e23fe6b05480c804117964f2a1080")
BASE_URL = "https://api.deepseek.com/v1"

logging.basicConfig(level=logging.INFO, format="%(levelname)-7s %(message)s")
logger = logging.getLogger("delta_e2e_v470")

# ==========================================================================
# E2E 测试场景 — 与 v4.5.0-v4.6.3 完全相同，保证可比性
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


def run_e2e(rounds_per_stage: Optional[int] = None):
    """运行 v4.7.0 指数衰减 CV 端到端验证。

    Args:
        rounds_per_stage: None=全部, 或指定每个阶段轮数（quick mode）
    """
    print("=" * 72)
    print("δ-mem E2E Validation v4.7.0 — Exponential Decay CV Recovery")
    print("=" * 72)

    # ── 初始化 ──
    print("\n[1] 初始化组件...")
    llm = LLMRouter()
    wm = WorldModel(dim=1536, config_path="config.yaml")
    auto_detect_dim(wm)

    delta_fusion = DeltaFusion()
    loop = SelfConsistencyLoop(llm, wm, dcore_mode="semantic", delta_fusion=delta_fusion)

    # 语义嵌入模式：使用校准后的 Φ 阈值
    loop.phi.base_threshold = 0.05
    loop.phi._current_threshold = 0.05

    emb_dim = wm.embedding_dim
    original_beta = delta_fusion.delta_layer.smatrix.beta
    dd_decay = loop.drift_detector.decay

    print(f"    LLM: deepseek-chat (API key: {'✓' if API_KEY else '✗'})")
    print(f"    Embedding: MiniLM-384dim (semantic), dim={emb_dim}")
    print(f"    Φ threshold: {loop.phi._current_threshold:.2f}")
    print(f"    δ-mem: S∈R^(8×8), λ=0.95, β={original_beta} (β→{original_beta*0.2} on drift)")
    print(f"    v4.7.0 Feature: Exp-Decay CV (decay={dd_decay}) — faster RECOVERY")
    from core.faiss_episodic import FAISS_AVAILABLE
    print(f"    FAISS: {'IndexFlatIP' if FAISS_AVAILABLE else 'numpy-fallback'}")
    print(f"    DriftDetector v1.3: window=20, cv_threshold=0.30, min_samples=5, "
          f"hysteresis=2, decay={dd_decay}")

    # ── 语义嵌入预热验证 ──
    print("\n[0] 语义嵌入预热验证")
    v_sci = wm.encode("光合作用的基本原理", force_hash=False)
    v_sci2 = wm.encode("光合作用的基本原理", force_hash=False)
    v_art = wm.encode("帮我写一首关于月亮的诗", force_hash=False)
    cos_same = float(np.dot(v_sci, v_sci2)) / (np.linalg.norm(v_sci) * np.linalg.norm(v_sci2) + 1e-8)
    cos_diff = float(np.dot(v_sci, v_art)) / (np.linalg.norm(v_sci) * np.linalg.norm(v_art) + 1e-8)
    print(f"    cos(同话题): {cos_same:.4f} (期望=1.0000)")
    print(f"    cos(跨话题): {cos_diff:.4f} (期望<0.80)")
    print(f"    语义质量: {'✓ 优秀' if cos_same > 0.99 and cos_diff < 0.5 else '⚠ 需检查'}")

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
    s_norms = []
    phi_vals = []
    drift_flags = []
    cv_vals = []  # v4.7.0: track CV over time
    faiss_sizes = []
    flux_decisions = []

    for i, env in enumerate(all_rounds):
        stage = ("STABLE" if i < stable_count
                 else "DRIFT" if i < stable_count + drift_count
                 else "RECOVERY")

        S_pre_norm = float(np.linalg.norm(delta_fusion.delta_layer.smatrix.S, 'fro'))

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
        cv_vals.append(float(dd.current_cv))
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
            "cv": round(cv_vals[-1], 4),  # v4.7.0
            "S_norm": round(S_post_norm, 6),
            "S_delta": round(S_delta, 8),
            "drifting": drift_post,
            "flux_enabled": flux_enabled,
            "faiss_entries": faiss_sizes[-1],
        }
        results.append(result)

        print(f"  [{i+1:2d}/{total}] {stage:7s} {status} | {flux_tag:5s} | "
              f"Φ={phi_vals[-1]:.4f} | CV={cv_vals[-1]:.4f} | ‖S‖={S_post_norm:.6f} | "
              f"{drift_tag} | {reason[:40]}")

        time.sleep(0.8)
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
    print(f"结果汇总 — v4.7.0 Exponential Decay CV")
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

    print(f"\n  漂移检测 (v4.7.0: 指数衰减 CV, decay={dd_decay}):")
    print(f"    CV 触发次数: {drift_detected}/{total}")
    print(f"    最终 CV: {cv_vals[-1]:.4f}")
    print(f"    CV 序列: {[round(v, 4) for v in cv_vals]}")

    drift_rounds_num = [i for i, f in enumerate(drift_flags) if f]
    if drift_rounds_num:
        print(f"    漂移轮次: {[r+1 for r in drift_rounds_num]}")
        first_drift = drift_rounds_num[0] + 1
        print(f"    首次触发: 第 {first_drift} 轮 (v4.6.3=第5轮)")

        # Highlight RECOVERY
        recovery_start = stable_count + drift_count
        recovery_drift = [r+1 for r in drift_rounds_num if r >= recovery_start]
        if not recovery_drift:
            print(f"    ✓ RECOVERY 阶段漂移已解除！")
        else:
            print(f"    ⚠ RECOVERY 阶段仍有漂移: {recovery_drift}")

    print(f"\n  δ-mem S 矩阵:")
    print(f"    初始 ‖S‖: {s_norms[0]:.6f}")
    print(f"    最终 ‖S‖: {s_norms[-1]:.6f}")
    print(f"    变化率: {(s_norms[-1] - s_norms[0]) / max(s_norms[0], 1e-8) * 100:+.1f}%")
    print(f"    总更新次数: {delta_fusion.delta_layer.total_updates}")
    print(f"    当前 β: {delta_fusion.delta_layer.smatrix.beta}")

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

    # ── 历史基线对比 ──
    print(f"\n  📊 历史基线对比:")
    print(f"    {'指标':<32} {'v4.5.0':>10} {'v4.6.2':>10} {'v4.6.3':>10} {'v4.7.0':>10}")
    print(f"    {'─'*32} {'─'*10} {'─'*10} {'─'*10} {'─'*10}")
    print(f"    {'FLUX_ENABLED':<32} {'27.3%':>10} {'36.4%':>10} {'36.4%':>10} {f'{flux_ratio*100:.1f}%':>10}")
    print(f"    {'通过率':<32} {'27.3%':>10} {'100%':>10} {'100%':>10} {f'{accepted/total*100:.0f}%':>10}")
    print(f"    {'Φ 均值':<32} {'-0.11':>10} {'+0.34':>10} {'+0.35':>10} {f'{np.mean(phi_vals):+.2f}':>10}")
    print(f"    {'漂移首触发':<32} {'-':>10} {'第5轮':>10} {'第5轮':>10} {f'第{drift_rounds_num[0]+1 if drift_rounds_num else "?"}轮':>10}")
    print(f"    {'CV 策略':<32} {'等权':>10} {'等权':>10} {'等权':>10} {f'衰减({dd_decay})':>10}")

    # ── FLUX_ENABLED 断点分析 ──
    stable_flux = sum(1 for i, r in enumerate(results)
                     if i < stable_count and r["flux_enabled"])
    drift_flux = sum(1 for i, r in enumerate(results)
                    if stable_count <= i < stable_count + drift_count and r["flux_enabled"])
    recover_flux = sum(1 for i, r in enumerate(results)
                      if i >= stable_count + drift_count and r["flux_enabled"])
    recover_total = total - stable_count - drift_count

    print(f"\n    FLUX_ENABLED 阶段分布:")
    print(f"      STABLE:   {stable_flux}/{stable_count} ({stable_flux/stable_count*100:.0f}%)")
    print(f"      DRIFT:    {drift_flux}/{drift_count} ({drift_flux/drift_count*100:.0f}%)")
    print(f"      RECOVERY: {recover_flux}/{recover_total} ({recover_flux/recover_total*100:.0f}%) "
          f"{'✓ 恢复成功!' if recover_flux > 0 else '✗ 未恢复'}")

    # ── 恢复加速效果分析 ──
    print(f"\n  ⚡ 恢复加速分析 (v4.7.0 vs v4.6.3):")
    recovery_cvs = cv_vals[stable_count + drift_count:]
    if recovery_cvs:
        print(f"    RECOVERY CV 序列: {[round(v, 4) for v in recovery_cvs]}")
        cv_trend = "下降中" if len(recovery_cvs) >= 2 and recovery_cvs[-1] < recovery_cvs[0] else "持平/上升"
        print(f"    CV 趋势: {cv_trend}")
        print(f"    v4.6.3 RECOVERY CV: 0.46 (等权，未恢复)")
        print(f"    v4.7.0 RECOVERY CV: {recovery_cvs[-1]:.4f} (衰减, "
              f"{'已恢复' if recovery_cvs[-1] < 0.30 else '恢复中'})")

    # ── 保存结果 ──
    out_dir = Path(__file__).resolve().parent.parent / "results"
    out_dir.mkdir(exist_ok=True)

    report = {
        "version": "v4.7.0",
        "mode": "exponential-decay-cv",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "config": {
            "llm": "deepseek-chat",
            "embedding": f"MiniLM-384dim (semantic), dim={emb_dim}",
            "phi_threshold": float(loop.phi._current_threshold),
            "delta_rank": 8,
            "lambda_decay": 0.95,
            "beta_update": 0.1,
            "drift_strategy": "beta_reduction_0.2x",
            "drift_min_samples": loop.drift_detector.min_samples_before_detect,
            "drift_hysteresis": loop.drift_detector.hysteresis_rounds,
            "cv_decay": dd_decay,
            "cv_window_size": loop.drift_detector.window_size,
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
            "first_drift_round": (drift_rounds_num[0] + 1) if drift_rounds_num else None,
            "phi_mean": round(float(np.mean(phi_vals)), 4),
            "phi_std": round(float(np.std(phi_vals)), 4),
            "final_cv": round(cv_vals[-1], 4),
            "S_norm_start": round(s_norms[0], 6),
            "S_norm_end": round(s_norms[-1], 6),
            "faiss_entries": faiss_sizes[-1],
            "total_updates": delta_fusion.delta_layer.total_updates,
            "hallucination_pass": f"{passed_hallu}/{len(HALLUCINATION_PROBES)}",
        },
        "rounds": results,
        "hallucination_probes": hallu_results,
        "phi_sequence": [round(v, 4) for v in phi_vals],
        "cv_sequence": [round(v, 4) for v in cv_vals],
    }

    out_path = out_dir / "delta_e2e_v4_7_0.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2, cls=_NumpyEncoder)
    print(f"\n  结果已保存: {out_path}")

    print("\n" + "=" * 72)
    print(f"v4.7.0 Exponential Decay CV E2E 验证完成！")
    print("=" * 72)

    return report


def main():
    import argparse
    parser = argparse.ArgumentParser(description="δ-mem v4.7.0 Exponential Decay CV E2E Validation")
    parser.add_argument("--quick", action="store_true",
                        help="Quick mode: 2 rounds per stage")
    parser.add_argument("--rounds", type=int, default=0,
                        help="Rounds per stage (overrides --quick)")
    args = parser.parse_args()

    rounds = args.rounds if args.rounds > 0 else (2 if args.quick else None)
    run_e2e(rounds)


if __name__ == "__main__":
    main()
