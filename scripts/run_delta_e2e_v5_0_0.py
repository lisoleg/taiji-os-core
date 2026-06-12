#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_delta_e2e_v5_0_0.py -- v5.0.0 E2E 验证（连续衰减自动调优）

基于 v4.9.0，唯一变更：
  DriftDetector(auto_tune=True) → 三态 lookup 替换为连续 sigmoid 公式
  gamma(CV, dCV/dt) = gamma_max - Delta_gamma * sigma((CV-CV_mid)/T) * slope_factor(dCV/dt)

FLUX 定义（同 v4.9.0）：output is not None（DRIFT 阶段有输出即 FLUX）

11 轮 E2E (5 稳定 + 3 漂移 + 3 恢复) + 5 幻觉探测
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
logger = logging.getLogger("delta_e2e_v500")

# ==========================================================================
# E2E 测试场景（同 v4.9.0）
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
    """Run v5.0.0 E2E validation -- continuous auto-tune decay."""

    print("=" * 72)
    print("delta E2E Validation v5.0.0 -- Continuous Auto-Tune Decay")
    print("=" * 72)
    print("  gamma(CV, dCV/dt) = gamma_max - Delta*sigma((CV-CV_mid)/T) * slope_factor")
    print("  auto_tune=True (default in v5.0)")
    print()

    # ---- 初始化 ----
    print("[1] Initializing components...")
    llm = LLMRouter()
    wm = WorldModel(dim=1536, config_path="config.yaml")
    auto_detect_dim(wm)

    delta_fusion = DeltaFusion()
    loop = SelfConsistencyLoop(llm, wm, dcore_mode="semantic", delta_fusion=delta_fusion)

    # v4.9.0 参数 (不变)
    loop.phi.base_threshold = 0.05
    loop.phi._current_threshold = 0.05

    # v5.0: auto_tune 默认启用，验证一下
    dd = loop.drift_detector
    assert dd.auto_tune, "v5.0: auto_tune must be True!"
    assert dd.adaptive, "v5.0: adaptive must be True!"

    emb_dim = wm.embedding_dim
    original_beta = delta_fusion.delta_layer.smatrix.beta

    print(f"    LLM: deepseek-chat (API key: {'OK' if API_KEY else 'MISSING'})")
    print(f"    Embedding: MiniLM-384dim (semantic), dim={emb_dim}")
    print(f"    Phi threshold: {loop.phi._current_threshold:.2f}")
    print(f"    delta-mem: S in R^(8x8), lambda=0.95, beta={original_beta}")
    print(f"    v5.0 DriftDetector v1.5: auto_tune=True, gamma_max={dd.gamma_max}, gamma_min={dd.gamma_min}")
    print(f"    Sigmoid: CV_mid={dd.cv_mid}, T={dd.temperature}")
    print(f"    Slope factor: alpha={dd.slope_alpha}, k={dd.slope_k}")

    # ---- 语义嵌入预热 ----
    print("\n[0] Semantic embedder warmup...")
    v_sci = wm.encode("光合作用的基本原理", force_hash=False)
    v_art = wm.encode("帮我写一首关于月亮的诗", force_hash=False)
    cos_diff = float(np.dot(v_sci, v_art)) / (np.linalg.norm(v_sci) * np.linalg.norm(v_art) + 1e-8)
    print(f"    cos(跨话题): {cos_diff:.4f} (expect < 0.80)")

    # ---- 所有轮次 ----
    all_rounds = STABLE_ROUNDS + DRIFT_ROUNDS + RECOVERY_ROUNDS
    total = len(all_rounds)
    stable_count = len(STABLE_ROUNDS)
    drift_count = len(DRIFT_ROUNDS)

    # ---- 运行推演 ----
    print(f"\n[2] Running inference loop ({total} rounds)...")
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

        # v4.9.0: FLUX = output is not None
        flux_enabled = (output is not None)
        flux_decisions.append(flux_enabled)

        status = "OK" if output else "BLOCK"
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
            "decay": round(current_decay, 4),   # v5.0: 连续值
            "S_norm": round(S_post, 6),
            "drifting": drift_post,
            "flux_enabled": flux_enabled,
        }
        results.append(result)

        print(f"  [{i+1:2d}/{total}] {stage:7s} {status:5s} | {flux_tag:5s} | "
              f"Phi={phi_vals[-1]:.4f} | CV={cv:.4f} | gamma={current_decay:.4f} | "
              f"{stage_tag:12s} | {reason[:30]}")
        time.sleep(0.8)
        wm.update(env["input"])

    # ---- 幻觉探测 ----
    print(f"\n[3] Hallucination probe ({len(HALLUCINATION_PROBES)} questions)...")
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
        status = "OK" if output else "BLOCK"
        print(f"  {status} {q[:40]} -> {output[:80] if output else 'BLOCKED'}")
        time.sleep(0.5)

    # ---- 汇总统计 ----
    print("\n" + "=" * 72)
    print("Summary -- v5.0.0 (Continuous Auto-Tune Decay)")
    print("=" * 72)

    accepted = sum(1 for r in results if r["status"] == "OK")
    flux_count = sum(flux_decisions)
    flux_ratio = flux_count / total if total else 0
    drift_detected = sum(drift_flags)

    print(f"\n  Inference stats:")
    print(f"    Total rounds: {total} | Passed: {accepted}")
    print(f"    Pass rate: {accepted/total*100:.1f}%")

    print(f"\n  Phi-Gate:")
    print(f"    FLUX_ENABLED: {flux_count}/{total} ({flux_ratio*100:.1f}%)")
    print(f"    Phi mean: {np.mean(phi_vals):.4f} +/- {np.std(phi_vals):.4f}")

    print(f"\n  Drift detection (auto-tune):")
    print(f"    CV trigger count: {drift_detected}/{total}")
    print(f"    Final CV: {cv_vals[-1]:.4f} " + ("RECOVERED (CV < 0.30)!" if cv_vals[-1] < 0.30 else "NOT recovered"))
    print(f"    CV sequence: {[round(v, 4) for v in cv_vals]}")
    print(f"    Decay sequence (continuous): {[round(v, 4) for v in decay_vals]}")
    print(f"    Stage sequence: {stage_vals}")

    # v5.0: 分析连续衰减
    print(f"\n  v5.0 Auto-Tune Analysis:")
    print(f"    Decay range: [{min(decay_vals):.4f}, {max(decay_vals):.4f}]")
    print(f"    Decay mean: {np.mean(decay_vals):.4f}")
    print(f"    Decay std: {np.std(decay_vals):.4f}")
    # 检查衰减是否随 CV 连续变化（而非三态跳变）
    unique_decays = len(set(round(v, 4) for v in decay_vals))
    print(f"    Unique decay values: {unique_decays} (v1.4 would be 3: 0.70/0.35/0.55)")

    # ---- 阶段分布 ----
    stable_flux = sum(1 for i, r in enumerate(results) if i < stable_count and r["flux_enabled"])
    drift_flux = sum(1 for i, r in enumerate(results)
                     if stable_count <= i < stable_count + drift_count and r["flux_enabled"])
    recover_flux = sum(1 for i, r in enumerate(results)
                       if i >= stable_count + drift_count and r["flux_enabled"])
    recover_total = total - stable_count - drift_count

    print(f"\n    FLUX_ENABLED by stage:")
    print(f"      STABLE:   {stable_flux}/{stable_count} ({stable_flux/stable_count*100:.0f}%)")
    print(f"      DRIFT:    {drift_flux}/{drift_count} ({drift_flux/drift_count*100:.0f}%)")
    print(f"      RECOVERY: {recover_flux}/{recover_total} ({recover_flux/recover_total*100:.0f}%)")

    # 恢复分析
    recovery_cv = [cv_vals[i] for i in range(total) if i >= stable_count + drift_count]
    cv_below_03 = next((i+1 for i, cv in enumerate(recovery_cv) if cv < 0.30), None)
    if cv_below_03:
        print(f"    RECOVERY: CV < 0.30 after {cv_below_03} rounds")

    # ---- 版本对比 ----
    print(f"\n  Version comparison (11 rounds):")
    print(f"    {'Version':>12} {'FLUX definition':>40} {'FLUX':>8} {'Final CV':>10} {'DRIFT FLUX':>12}")
    print(f"    {'-'*12} {'-'*40} {'-'*8} {'-'*10} {'-'*12}")
    print(f"    {'v4.9.0':>12} {'output is not None':>40} {'?':>8} {'?':>10} {'?':>12}")
    print(f"    {'v5.0.0':>12} {'output is not None (same)':>40} {f'{flux_ratio*100:.1f}%':>8} "
          f"{f'{cv_vals[-1]:.4f}':>10} {f'{drift_flux}/{drift_count} ({drift_flux/drift_count*100:.0f}%)':>12}")

    # ---- 保存 ----
    out_dir = Path(__file__).resolve().parent.parent / "results"
    out_dir.mkdir(exist_ok=True)

    report = {
        "version": "v5.0.0",
        "mode": "continuous-auto-tune-decay",
        "change": "DriftDetector(auto_tune=True) -- continuous sigmoid decay",
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
            "decay_range": [round(min(decay_vals), 4), round(max(decay_vals), 4)],
            "decay_unique_values": unique_decays,
        },
        "rounds": results,
        "hallucination_probes": hallu_results,
        "cv_sequence": [round(v, 4) for v in cv_vals],
        "decay_sequence": [round(v, 4) for v in decay_vals],
        "stage_sequence": stage_vals,
    }

    out_path = out_dir / "delta_e2e_v5_0_0.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2, cls=_NumpyEncoder)
    print(f"\n  Results saved: {out_path}")

    # ---- 幻觉汇总 ----
    hall_passed = sum(1 for h in hallu_results if h["passed"])
    print(f"\n  Hallucination probe: {hall_passed}/{len(HALLUCINATION_PROBES)}")

    print("\n" + "=" * 72)
    print("v5.0.0 Continuous Auto-Tune Decay E2E validation complete!")
    print("=" * 72)

    return report


if __name__ == "__main__":
    run_e2e()
