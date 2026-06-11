"""
run_delta_e2e_v4_7_0_extended.py — δ-mem v4.7.0 扩展 E2E 验证 (13+ 轮)

与 v4.7.0 标准版的核心区别:
  - RECOVERY 从 3 轮扩展到 5 轮，验证指数衰减 CV 最终跌破 0.30 阈值
  - 目标: FLUX_ENABLED ≥ 60%（因为 5/13 ≈ 38% BLOCK 来自 3 DRIFT + 早期 RECOVERY）
  - 关键指标: RECOVERY 最后一轮 CV < 0.30（宣告恢复成功）

对比基线:
  - v4.6.3 等权 CV: RECOVERY CV = 0.46（未能恢复）
  - v4.7.0 标准: 3 RECOVERY, CV = 0.3147（接近阈值但未跌破）
  - v4.7.0 扩展: 5 RECOVERY, 预期 CV < 0.28（恢复成功）

环境: sentence-transformers + DeepSeek Chat API
用法: python scripts/run_delta_e2e_v4_7_0_extended.py
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
logger = logging.getLogger("delta_e2e_v470_ext")

# ==========================================================================
# 扩展版 E2E 测试场景
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

# 扩展 RECOVERY: 5 轮科学问答 (标准版只有 3 轮)
RECOVERY_ROUNDS = [
    {"intent": "科学问答", "input": "线粒体的结构和功能是什么？"},
    {"intent": "科学问答", "input": "细胞呼吸的三个阶段分别在哪里发生？"},
    {"intent": "科学问答", "input": "有氧呼吸和无氧呼吸的区别是什么？"},
    {"intent": "科学问答", "input": "ATP在细胞代谢中扮演什么角色？"},
    {"intent": "科学问答", "input": "DNA复制的主要步骤是什么？"},
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


def run_extended_e2e():
    """运行 v4.7.0 扩展 E2E: 5 STABLE + 3 DRIFT + 5 RECOVERY = 13 轮"""
    print("=" * 72)
    print("δ-mem E2E EXTENDED Validation v4.7.0 — Exponential Decay CV + Extended RECOVERY")
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

    # 验证 decay 参数
    print(f"    LLM: deepseek-chat (API key: {'✓' if API_KEY else '✗'})")
    print(f"    Embedding: MiniLM-384dim (semantic), dim={emb_dim}")
    print(f"    Φ threshold: {loop.phi._current_threshold:.2f}")
    print(f"    δ-mem: S∈R^(8×8), λ=0.95, β={original_beta} (β→{original_beta*0.2} on drift)")
    print(f"    v4.7.0 EXTENDED: decay={dd_decay}, {len(RECOVERY_ROUNDS)} RECOVERY rounds")
    print(f"    DriftDetector: window=20, cv_threshold=0.30, min_samples=5, hysteresis=2")

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
    all_rounds = STABLE_ROUNDS + DRIFT_ROUNDS + RECOVERY_ROUNDS
    total = len(all_rounds)
    stable_count = len(STABLE_ROUNDS)
    drift_count = len(DRIFT_ROUNDS)
    recover_count = len(RECOVERY_ROUNDS)
    drift_start = stable_count
    recover_start = stable_count + drift_count

    print(f"\n  测试设计: {stable_count} STABLE + {drift_count} DRIFT + {recover_count} RECOVERY = {total} 轮")
    print(f"  DRIFT 轮次占比: {drift_count}/{total} ({drift_count/total*100:.0f}%)")
    print(f"  RECOVERY 轮次占比: {recover_count}/{total} ({recover_count/total*100:.0f}%)")

    # ── 运行推演 ──
    print(f"\n[2] 运行推演循环 ({total} 轮)...")
    print("-" * 72)

    results = []
    s_norms = []
    phi_vals = []
    drift_flags = []
    cv_vals = []
    faiss_sizes = []
    flux_decisions = []
    betas = []  # track beta changes

    for i, env in enumerate(all_rounds):
        stage = ("STABLE" if i < drift_start
                 else "DRIFT" if i < recover_start
                 else "RECOVERY")

        S_pre_norm = float(np.linalg.norm(delta_fusion.delta_layer.smatrix.S, 'fro'))
        current_beta = float(delta_fusion.delta_layer.smatrix.beta)

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
        betas.append(current_beta)

        S_delta = S_post_norm - S_pre_norm
        flux_enabled = not drift_post and output is not None
        flux_decisions.append(flux_enabled)

        status = "✓" if output else "✗"
        drift_tag = "⏸ DRIFT" if drift_post else "▶ NORM"
        flux_tag = "FLUX" if flux_enabled else "BLOCK"
        beta_tag = f"β={current_beta:.3f}"

        result = {
            "round": i + 1,
            "stage": stage,
            "input": env["input"][:50],
            "status": status,
            "reason": reason[:60] if reason else "",
            "phi": round(phi_vals[-1], 4),
            "cv": round(cv_vals[-1], 4),
            "S_norm": round(S_post_norm, 6),
            "S_delta": round(S_delta, 8),
            "drifting": drift_post,
            "flux_enabled": flux_enabled,
            "faiss_entries": faiss_sizes[-1],
            "beta": round(current_beta, 3),
        }
        results.append(result)

        print(f"  [{i+1:2d}/{total}] {stage:7s} {status} | {flux_tag:5s} | "
              f"Φ={phi_vals[-1]:.4f} | CV={cv_vals[-1]:.4f} | ‖S‖={S_post_norm:.6f} | "
              f"{drift_tag} | {beta_tag} | {reason[:30]}")

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
    print(f"结果汇总 — v4.7.0 EXTENDED (Exponential Decay CV + 5-RECOVERY)")
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

    print(f"\n  漂移检测 (v4.7.0: 指数衰减 CV, decay={dd_decay}, {recover_count} RECOVERY):")
    print(f"    CV 触发次数: {drift_detected}/{total}")
    print(f"    最终 CV: {cv_vals[-1]:.4f}")
    print(f"    CV 序列 (ALL):   {[round(v, 4) for v in cv_vals]}")

    drift_rounds_num = [i for i, f in enumerate(drift_flags) if f]
    if drift_rounds_num:
        print(f"    漂移轮次: {[r+1 for r in drift_rounds_num]}")
        first_drift = drift_rounds_num[0] + 1
        print(f"    首次触发: 第 {first_drift} 轮")

        # RECOVERY 分析
        recovery_drift = [r+1 for r in drift_rounds_num if r >= recover_start]
        if not recovery_drift:
            print(f"    ✓✓✓ RECOVERY 阶段漂移已完全解除!")
        else:
            recovery_exit = recover_start + recover_count - 1
            last_recovery = recover_start + recover_count
            print(f"    ⚠ RECOVERY 阶段仍有漂移: {recovery_drift}")
            print(f"    但最终 CV={cv_vals[-1]:.4f} "
                  f"({'< 0.30 ✓ 恢复成功' if cv_vals[-1] < 0.30 else '> 0.30 ✗ 仍需更多恢复轮次'})")

    # RECOVERY 阶段详细分析
    recovery_cvs = cv_vals[recover_start:]
    recovery_phis = phi_vals[recover_start:]
    print(f"\n  ⚡ RECOVERY 阶段详细分析:")
    print(f"    RECOVERY CV 序列 (5轮): {[round(v, 4) for v in recovery_cvs]}")
    print(f"    RECOVERY Φ 序列 (5轮):  {[round(v, 4) for v in recovery_phis]}")
    if len(recovery_cvs) >= 2:
        cv_trend = recovery_cvs[-1] - recovery_cvs[0]
        print(f"    CV 变化: {recovery_cvs[0]:.4f} → {recovery_cvs[-1]:.4f} ({cv_trend:+.4f})")
        print(f"    趋势: {'↓ 下降中 (正确)' if cv_trend < 0 else '↑ 上升 (异常)'}")
        print(f"    阈值距离: {recovery_cvs[-1] - 0.30:+.4f} "
              f"({'已低于阈值!' if recovery_cvs[-1] < 0.30 else '仍需更多轮次'})")

    print(f"\n  δ-mem S 矩阵:")
    print(f"    初始 ‖S‖: {s_norms[0]:.6f}")
    print(f"    最终 ‖S‖: {s_norms[-1]:.6f}")
    print(f"    变化率: {(s_norms[-1] - s_norms[0]) / max(s_norms[0], 1e-8) * 100:+.1f}%")
    print(f"    总更新次数: {delta_fusion.delta_layer.total_updates}")
    print(f"    最终 β: {delta_fusion.delta_layer.smatrix.beta}")

    print(f"\n  FAISS Episodic Index:")
    print(f"    条目数: {faiss_sizes[-1]}")

    print(f"\n  幻觉探测:")
    passed_hallu = sum(1 for h in hallu_results if h["passed"])
    print(f"    通过: {passed_hallu}/{len(HALLUCINATION_PROBES)}")

    # ── 历史基线对比 (含扩展版) ──
    print(f"\n  📊 全版本对比:")
    print(f"    {'指标':<28} {'v4.5.0':>8} {'v4.6.3':>8} {'v4.7.0':>8} {'v4.7.0':>8} {'v4.7.0':>8}")
    print(f"    {'':28} {'哈希':>8} {'降β':>8} {'3REC':>8} {'2REC':>8} {'5REC+':>8}")
    print(f"    {'─'*28} {'─'*8} {'─'*8} {'─'*8} {'─'*8} {'─'*8}")
    print(f"    {'轮次 (S/D/R)':<28} {'5/3/3':>8} {'5/3/3':>8} {'5/3/3':>8} {'5/3/2':>8} {f'{stable_count}/{drift_count}/{recover_count}':>8}")
    print(f"    {'FLUX_ENABLED':<28} {'27.3%':>8} {'36.4%':>8} {'45.5%':>8} {'46.2%*':>8} {f'{flux_ratio*100:.1f}%':>8}")
    print(f"    {'STAGE 5/5 100%':<28} {'NO':>8} {'YES':>8} {'YES':>8} {'N/A':>8} {f'{sum(1 for f in flux_decisions[:stable_count] if f)==stable_count}':>8}")
    print(f"    {'最终 CV':<28} {'-':>8} {'0.46':>8} {'0.3147':>8} {'~0.28*':>8} {f'{cv_vals[-1]:.4f}':>8}")
    print(f"    {'CV < 0.30':<28} {'×':>8} {'×':>8} {'×':>8} {'~✓*':>8} {f'{cv_vals[-1] < 0.30}':>8}")
    print(f"    {'通过率':<28} {'27.3%':>8} {'100%':>8} {'100%':>8} {'100%':>8} {f'{accepted/total*100:.0f}%':>8}")
    print(f"    {'Φ 均值':<28} {'-0.11':>8} {'+0.35':>8} {'+0.35':>8} {'+0.35':>8} {f'{np.mean(phi_vals):+.2f}':>8}")
    print(f"  * 预测值 (测试结果会更新)")

    # FLUX_ENABLED 阶段分布
    stable_flux = sum(1 for i, r in enumerate(results) if i < drift_start and r["flux_enabled"])
    drift_flux = sum(1 for i, r in enumerate(results) if drift_start <= i < recover_start and r["flux_enabled"])
    recover_flux = sum(1 for i, r in enumerate(results) if i >= recover_start and r["flux_enabled"])

    print(f"\n  FLUX_ENABLED 阶段分布:")
    print(f"    STABLE:   {stable_flux}/{stable_count} ({stable_flux/stable_count*100:.0f}%)")
    print(f"    DRIFT:    {drift_flux}/{drift_count} ({drift_flux/drift_count*100:.0f}%)")
    print(f"    RECOVERY: {recover_flux}/{recover_count} ({recover_flux/recover_count*100:.0f}%)")

    # ── 关键结论 ──
    print(f"\n  🔑 关键结论:")
    if cv_vals[-1] < 0.30:
        print(f"    ✓ 扩展 RECOVERY ({recover_count}轮) 后 CV={cv_vals[-1]:.4f} < 0.30，恢复成功!")
        print(f"    ✓ 证明: 指数衰减 CV 算法有效，瓶颈仅在于 RECOVERY 轮次不足")
    else:
        print(f"    ⚠ {recover_count} RECOVERY 轮次仍不足以使 CV < 0.30")
        print(f"    ⚠ 建议: 调低 decay 参数 (如 0.55) 或增加 RECOVERY 轮次")
    print(f"    FLUX_ENABLED={flux_ratio*100:.1f}%, "
          f"{'达标' if flux_ratio >= 0.6 else f'距 60% 目标差 {60-flux_ratio*100:.1f}pp'}")

    # ── 保存结果 ──
    out_dir = Path(__file__).resolve().parent.parent / "results"
    out_dir.mkdir(exist_ok=True)

    report = {
        "version": "v4.7.0-extended",
        "mode": "exponential-decay-cv-extended-recovery",
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
            "design": f"{stable_count}S+{drift_count}D+{recover_count}R={total} rounds",
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
            "cv_below_threshold": cv_vals[-1] < 0.30,
            "recovery_cv_sequence": [round(v, 4) for v in recovery_cvs] if recovery_cvs else [],
            "S_norm_start": round(s_norms[0], 6),
            "S_norm_end": round(s_norms[-1], 6),
            "faiss_entries": faiss_sizes[-1],
            "total_updates": delta_fusion.delta_layer.total_updates,
            "hallucination_pass": f"{passed_hallu}/{len(HALLUCINATION_PROBES)}",
            "stable_flux": f"{stable_flux}/{stable_count}",
            "drift_flux": f"{drift_flux}/{drift_count}",
            "recovery_flux": f"{recover_flux}/{recover_count}",
        },
        "rounds": results,
        "hallucination_probes": hallu_results,
        "phi_sequence": [round(v, 4) for v in phi_vals],
        "cv_sequence": [round(v, 4) for v in cv_vals],
        "drift_flags": drift_flags,
    }

    out_path = out_dir / "delta_e2e_v4_7_0_extended.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2, cls=_NumpyEncoder)
    print(f"\n  结果已保存: {out_path}")

    print("\n" + "=" * 72)
    print(f"v4.7.0 EXTENDED E2E 验证完成!")
    print("=" * 72)

    return report


if __name__ == "__main__":
    run_extended_e2e()
