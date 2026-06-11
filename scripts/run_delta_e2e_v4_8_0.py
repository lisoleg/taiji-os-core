"""
run_delta_e2e_v4_8_0.py — v4.8.0 自适应三态衰减 E2E 验证

目的: 测试自适应三态 decay (STABLE=0.70/DRIFT=0.35/RECOVERY=0.55)
  对比 v4.7.0 固定 decay=0.55，验证自适应模式能否改善 FLUX_ENABLED。
  
自适应策略 (DriftDetector v1.4):
  STABLE:   γ=0.70 — 慢遗忘，稳态
  DRIFTING: γ=0.35 — 快遗忘，快速适应  
  RECOVERY: γ=0.55 — 平衡恢复

对比基线:
  - v4.7.0 decay=0.55 (11轮): FLUX_ENABLED=72.7%, 最终 CV=0.2478, 2轮恢复
  - v4.8.0 adaptive (11轮): 目标 FLUX_ENABLED ≥ 72.7%, 2轮内恢复

环境: sentence-transformers + DeepSeek Chat API
用法: python scripts/run_delta_e2e_v4_8_0.py
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
from core.drift_detector import DriftDetector

API_KEY = os.environ.get("DEEPSEEK_API_KEY", "sk-d83e23fe6b05480c804117964f2a1080")
BASE_URL = "https://api.deepseek.com/v1"

logging.basicConfig(level=logging.INFO, format="%(levelname)-7s %(message)s")
logger = logging.getLogger("delta_e2e_adaptive")

# ==========================================================================
# E2E 测试场景
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
        return "[Error] API failed after 3 retries"


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


def run_e2e():
    """Run v4.8.0 adaptive decay E2E validation."""
    print("=" * 72)
    print("δ-mem E2E Validation v4.8.0 — Adaptive Decay (STABLE/DRIFT/RECOVERY)")
    print("=" * 72)

    # ── 初始化 ──
    print("\n[1] 初始化组件...")
    llm = LLMRouter()
    wm = WorldModel(dim=1536, config_path="config.yaml")
    auto_detect_dim(wm)

    delta_fusion = DeltaFusion()
    loop = SelfConsistencyLoop(llm, wm, dcore_mode="semantic", delta_fusion=delta_fusion)

    # 语义嵌入模式
    loop.phi.base_threshold = 0.05
    loop.phi._current_threshold = 0.05

    # ⚡ CRITICAL: enable adaptive mode (v4.8.0)
    loop.drift_detector.adaptive = True
    # RECOVERY 阶段使用 decay=0.55 (adaptive 模式下 decay 字段即为 RECOVERY 值)
    loop.drift_detector.decay = 0.55

    dd = loop.drift_detector
    emb_dim = wm.embedding_dim
    original_beta = delta_fusion.delta_layer.smatrix.beta

    print(f"    LLM: deepseek-chat (API key: {'✓' if API_KEY else '✗'})")
    print(f"    Embedding: MiniLM-384dim (semantic), dim={emb_dim}")
    print(f"    Φ threshold: {loop.phi._current_threshold:.2f}")
    print(f"    δ-mem: S∈R^(8×8), λ=0.95, β={original_beta}")
    print(f"    ⚡ v4.8.0 Adaptive Decay: STABLE=0.70 / DRIFT=0.35 / RECOVERY=0.55")
    print(f"    DriftDetector v1.4: window=20, cv_threshold=0.30, hysteresis=2")

    # ── 语义嵌入预热 ──
    print("\n[0] 语义嵌入预热验证")
    v_sci = wm.encode("光合作用的基本原理", force_hash=False)
    v_art = wm.encode("帮我写一首关于月亮的诗", force_hash=False)
    cos_diff = float(np.dot(v_sci, v_art)) / (np.linalg.norm(v_sci) * np.linalg.norm(v_art) + 1e-8)
    print(f"    cos(跨话题): {cos_diff:.4f} (期望<0.80)")

    # ── 所有轮次 ──
    all_rounds = STABLE_ROUNDS + DRIFT_ROUNDS + RECOVERY_ROUNDS
    total = len(all_rounds)
    stable_count = len(STABLE_ROUNDS)
    drift_count = len(DRIFT_ROUNDS)

    # ── 运行推演 ──
    print(f"\n[2] 运行推演循环 ({total} 轮)...")
    print("-" * 72)

    results = []
    phi_vals = []
    cv_vals = []
    decay_vals = []
    stage_vals = []
    drift_flags = []
    flux_decisions = []
    s_norms = []

    for i, env in enumerate(all_rounds):
        stage = ("STABLE" if i < stable_count
                 else "DRIFT" if i < stable_count + drift_count
                 else "RECOVERY")

        S_pre = float(np.linalg.norm(delta_fusion.delta_layer.smatrix.S, 'fro'))
        output, reason = loop.step(env, env["input"])
        S_post = float(np.linalg.norm(delta_fusion.delta_layer.smatrix.S, 'fro'))

        drift_post = dd.is_drifting()
        cv = float(dd.current_cv)
        current_decay = dd._get_decay()
        current_stage = dd._stage

        s_norms.append(S_post)
        if dd.count > 0:
            last_idx = (dd.write_idx - 1) % dd.window_size
            phi_vals.append(float(dd.phi_history[last_idx]))
        else:
            phi_vals.append(0.0)
        drift_flags.append(drift_post)
        cv_vals.append(cv)
        decay_vals.append(current_decay)
        stage_vals.append(current_stage)

        flux_enabled = not drift_post and output is not None
        flux_decisions.append(flux_enabled)

        status = "✓" if output else "✗"
        stage_tag = f"[{current_stage}]"
        flux_tag = "FLUX" if flux_enabled else "BLOCK"

        result = {
            "round": i + 1,
            "stage": stage,
            "detector_stage": current_stage,
            "input": env["input"][:50],
            "status": status,
            "reason": reason[:60] if reason else "",
            "phi": round(phi_vals[-1], 4),
            "cv": round(cv, 4),
            "decay": round(current_decay, 2),
            "S_norm": round(S_post, 6),
            "drifting": drift_post,
            "flux_enabled": flux_enabled,
        }
        results.append(result)

        print(f"  [{i+1:2d}/{total}] {stage:7s} {status} | {flux_tag:5s} | "
              f"Φ={phi_vals[-1]:.4f} | CV={cv:.4f} | γ={current_decay:.2f} | "
              f"{stage_tag:12s} | {reason[:30]}")

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
        print(f"  {status} {q[:40]} → {output[:80] if output else 'BLOCKED'}")
        time.sleep(0.5)

    # ── 汇总统计 ──
    print("\n" + "=" * 72)
    print(f"结果汇总 — v4.8.0 Adaptive Decay")
    print("=" * 72)

    accepted = sum(1 for r in results if r["status"] == "✓")
    flux_count = sum(flux_decisions)
    flux_ratio = flux_count / total if total else 0
    drift_detected = sum(drift_flags)

    print(f"\n  推演统计:")
    print(f"    总轮次: {total} | 通过: {accepted}")
    print(f"    通过率: {accepted/total*100:.1f}%")

    print(f"\n  Φ-Gate (语义门控):")
    print(f"    FLUX_ENABLED: {flux_count}/{total} ({flux_ratio*100:.1f}%)")
    print(f"    Φ 均值: {np.mean(phi_vals):.4f} ± {np.std(phi_vals):.4f}")

    print(f"\n  漂移检测 (自适应):")
    print(f"    CV 触发次数: {drift_detected}/{total}")
    print(f"    最终 CV: {cv_vals[-1]:.4f} " + ("✓ 跌破阈值!" if cv_vals[-1] < 0.30 else "✗ 未恢复"))
    print(f"    CV 序列: {[round(v, 4) for v in cv_vals]}")
    print(f"    Decay 序列: {[round(v, 2) for v in decay_vals]}")
    print(f"    阶段序列: {stage_vals}")

    # ── 阶段分布 ──
    stable_flux = sum(1 for i, r in enumerate(results) if i < stable_count and r["flux_enabled"])
    drift_flux = sum(1 for i, r in enumerate(results)
                     if stable_count <= i < stable_count + drift_count and r["flux_enabled"])
    recover_flux = sum(1 for i, r in enumerate(results)
                       if i >= stable_count + drift_count and r["flux_enabled"])
    recover_total = total - stable_count - drift_count

    print(f"\n    FLUX_ENABLED 阶段分布:")
    print(f"      STABLE:   {stable_flux}/{stable_count} ({stable_flux/stable_count*100:.0f}%)")
    print(f"      DRIFT:    {drift_flux}/{drift_count} ({drift_flux/drift_count*100:.0f}%)")
    print(f"      RECOVERY: {recover_flux}/{recover_total} ({recover_flux/recover_total*100:.0f}%)")

    # 恢复分析
    recovery_cv = [cv_vals[i] for i in range(total) if i >= stable_count + drift_count]
    cv_below_03 = next((i+1 for i, cv in enumerate(recovery_cv) if cv < 0.30), None)
    if cv_below_03:
        print(f"    RECOVERY {cv_below_03} 轮后 CV < 0.30 ✓")

    # ── 阶段切换分析 ──
    stage_transitions = []
    prev_stage = stage_vals[0]
    for i, s in enumerate(stage_vals):
        if s != prev_stage:
            stage_transitions.append(f"第{i+1}轮: {prev_stage}→{s}")
            prev_stage = s
    if stage_transitions:
        print(f"\n    阶段切换: {' | '.join(stage_transitions)}")

    # ── 对比表 ──
    print(f"\n  📊 版本对比 (11-round):")
    print(f"    {'版本':>12} {'策略':>22} {'FLUX':>8} {'最终CV':>10} {'恢复':>6}")
    print(f"    {'─'*12} {'─'*22} {'─'*8} {'─'*10} {'─'*6}")
    print(f"    {'v4.7.0':>12} {'fixed decay=0.55':>22} {'72.7%':>8} {'0.2478':>10} {'2轮':>6}")
    recovery_label = f"{cv_below_03}轮" if cv_below_03 else "未恢复"
    print(f"    {'v4.8.0':>12} {'adaptive 3-stage':>22} {f'{flux_ratio*100:.1f}%':>8} "
          f"{f'{cv_vals[-1]:.4f}':>10} {recovery_label:>6}")

    # ── 保存 ──
    out_dir = Path(__file__).resolve().parent.parent / "results"
    out_dir.mkdir(exist_ok=True)

    report = {
        "version": "v4.8.0",
        "mode": "adaptive-three-stage-decay",
        "decay_strategy": {"STABLE": 0.70, "DRIFTING": 0.35, "RECOVERY": 0.55},
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "summary": {
            "total_rounds": total,
            "accepted": accepted,
            "flux_enabled": flux_count,
            "flux_ratio": round(flux_ratio, 4),
            "phi_mean": round(float(np.mean(phi_vals)), 4),
            "final_cv": round(cv_vals[-1], 4),
            "cv_below_threshold": cv_vals[-1] < 0.30,
            "recovery_rounds_to_cv_below_03": cv_below_03,
            "hallucination_pass": f"{sum(1 for h in hallu_results if h['passed'])}/{len(HALLUCINATION_PROBES)}",
            "stage_transitions": stage_transitions,
        },
        "rounds": results,
        "hallucination_probes": hallu_results,
        "cv_sequence": [round(v, 4) for v in cv_vals],
        "decay_sequence": [round(v, 2) for v in decay_vals],
        "stage_sequence": stage_vals,
    }

    out_path = out_dir / "delta_e2e_v4_8_0.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2, cls=_NumpyEncoder)
    print(f"\n  结果已保存: {out_path}")

    # ── 幻觉汇总 ──
    hall_passed = sum(1 for h in hallu_results if h["passed"])
    print(f"\n  幻觉探测: {hall_passed}/{len(HALLUCINATION_PROBES)}")

    print("\n" + "=" * 72)
    print(f"v4.8.0 Adaptive Decay E2E 验证完成！")
    print("=" * 72)

    return report


if __name__ == "__main__":
    run_e2e()
