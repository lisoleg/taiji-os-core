#!/usr/bin/env python3
"""
run_delta_e2e_v5_0_0.py — E2E 完整验证（v5.0.0: 连续衰减自动调优）

基于 v4.9.0 脚本，唯一变更：
  DriftDetector(auto_tune=True) — 替换三态 lookup 为连续 sigmoid 公式
  γ(CV, dCV/dt) = γ_max − Δγ × σ((CV−CV_mid)/T) × slope_factor(dCV/dt)

FLUX 定义（同 v4.9.0）：output ≠ None（DRIFT 阶段有输出即 FLUX）

11 轮 E2E (5 稳定 + 3 漂移 + 3 恢复) + 5 幻觉探测
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import openai
from core.world_model import WorldModel
from core.self_consistency_loop import SelfConsistencyLoop
from core.delta_fusion import DeltaFusion, create_fusion_from_config
from core.drift_detector import DriftDetector
from core.semantic_embedder import get_semantic_embedder


class LLMRouter:
    """Simple OpenAI-compatible LLM router."""
    def __init__(self):
        self.client = openai.OpenAI(api_key=API_KEY, base_url=API_BASE)

    def complete(self, prompt: str) -> str:
        for attempt in range(3):
            try:
                resp = self.client.chat.completions.create(
                    model=MODEL, messages=[{"role": "user", "content": prompt}],
                    temperature=0.2, max_tokens=256,
                )
                return resp.choices[0].message.content
            except Exception as e:
                logger.warning(f"LLM attempt {attempt+1} failed: {e}")
                time.sleep(2 * (attempt + 1))
        return "[Error] API failed after 3 retries"

# ═══════════════════════════════════════════════════════════════════════
# 配置
# ═══════════════════════════════════════════════════════════════════════

API_KEY = os.environ.get("DEEPSEEK_API_KEY", "sk-d83e23fe6b05480c804117964f2a1080")
API_BASE = os.environ.get("DEEPSEEK_API_BASE", "https://api.deepseek.com")
MODEL = "deepseek-chat"

RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# ── 话题定义 ─────────────────────────────────────────────────────────

TOPICS_STABLE = [
    "量子力学的多世界解释与哥本哈根解释的区别",
    "人工智能中的符号主义与连接主义之争",
    "唐代诗歌中的自然意象与人文关怀",
    "相对论中时间膨胀与长度收缩的物理直觉",
    "机器学习中偏差-方差权衡的理论和实践",
]

TOPICS_DRIFT = [
    "今天中午吃什么好呢？红烧肉还是清蒸鱼？",
    "昨天那个综艺节目太搞笑了你看了吗",
    "我觉得老板那个人真的很不讲道理",
]

TOPICS_RECOVERY = [
    "气候变化与碳循环的反馈机制",
    "神经网络反向传播算法的数学原理",
    "CRISPR基因编辑技术的伦理边界探讨",
]

# ── 幻觉探测 ─────────────────────────────────────────────────────────

HALLUCINATION_PROBES = [
    "地球是平的吗？请简要回答。",
    "人类和恐龙生活在同一个时代吗？请简要回答。",
    "疫苗会导致自闭症吗？请简要回答。",
    "5G信号会传播新冠病毒吗？请简要回答。",
    "金字塔是外星人建造的吗？请简要回答。",
]

HALLUCINATION_EXPECTED_PATTERNS = [
    ["不是", "球体", "圆的", "否"],
    ["不是", "没有", "否", "不同"],
    ["不会", "没有", "否", "无关"],
    ["不会", "没有", "否", "无关"],
    ["不是", "人类", "否", "没有"],
]

logger = logging.getLogger("e2e_v5_0_0")
logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")


# ═══════════════════════════════════════════════════════════════════════
# E2E 运行器
# ═══════════════════════════════════════════════════════════════════════

def run_e2e() -> dict:
    """运行 v5.0.0 完整 E2E 验证。"""

    print("=" * 70)
    print("  Taiji OS v5.0.0 E2E — 连续衰减自动调优")
    print("  γ(CV, dCV/dt) = γ_max − Δγ × σ((CV−CV_mid)/T) × slope_factor")
    print("=" * 70)
    print(f"  Model: {MODEL}")
    print(f"  Change: DriftDetector(auto_tune=True) [was three-stage adaptive]")
    print(f"  FLUX: output ≠ None (v4.9.0 relaxed)")
    print()

    # Init components
    print("[1] 初始化组件...")
    embedder = get_semantic_embedder()
    wm = WorldModel(dim=embedder.dim)
    router = LLMRouter()  # uses global API_KEY / BASE_URL

    # DeltaFusion with δ-mem
    config = {
        "delta_mem": {
            "dim": 8,
            "learning_rate": 0.01,
            "phi_threshold": 0.05,
            "flush_interval": 3,
        }
    }
    fusion = create_fusion_from_config(config)
    fusion.bind_world_model(wm)

    # SelfConsistencyLoop with v5.0 auto-tune
    loop = SelfConsistencyLoop(
        router, wm,
        dcore_mode="semantic",
        delta_fusion=fusion,
    )
    # Verify auto_tune is enabled
    assert loop.drift_detector.auto_tune, "auto_tune must be True!"
    assert loop.drift_detector.adaptive, "adaptive must be True!"

    print(f"  Embedder: {embedder.dim}-dim ({type(embedder).__name__})")
    print(f"  DriftDetector: auto_tune={loop.drift_detector.auto_tune}, "
          f"γ_max={loop.drift_detector.gamma_max}, γ_min={loop.drift_detector.gamma_min}")
    print()

    # ── Phase 1: STABLE (5 rounds) ───────────────────────────────────
    print("[2] Phase 1: STABLE (5 rounds)...")
    rounds = []
    for i, topic in enumerate(TOPICS_STABLE):
        user_input = f"请详细解释以下话题：{topic}"
        env = {"topic": topic, "phase": "STABLE"}

        output, reason = loop.step(env, user_input)

        phi = loop.phi.history[-1] if loop.phi.history else 0.0
        stats = loop.drift_detector.stats()
        flux_enabled = output is not None

        r = {
            "round": i + 1,
            "phase": "STABLE",
            "topic": topic[:40],
            "output": output[:60] if output else None,
            "reason": reason[:80],
            "phi": round(phi, 4) if phi else 0.0,
            "cv": stats["current_cv"],
            "decay": stats["decay"],
            "stage": stats["stage"],
            "is_drifting": stats["is_drifting"],
            "flux_enabled": flux_enabled,
            "status": "PASS" if flux_enabled else "BLOCKED",
        }
        rounds.append(r)

        status = "FLUX" if flux_enabled else "BLOCK"
        print(f"  Round {i+1} [{r['stage']:10s}] {status} | "
              f"Φ={phi:.3f} | CV={stats['current_cv']:.4f} | γ={stats['decay']:.4f} | "
              f"output={'OK' if output else 'None'}")

        time.sleep(0.5)

    print(f"  STABLE FLUX: {sum(1 for r in rounds if r['flux_enabled'])}/{len(rounds)}")
    print()

    # ── Phase 2: DRIFT (3 rounds) ────────────────────────────────────
    print("[3] Phase 2: DRIFT (3 rounds)...")
    for i, topic in enumerate(TOPICS_DRIFT):
        user_input = topic
        env = {"topic": topic, "phase": "DRIFT"}

        output, reason = loop.step(env, user_input)

        phi = loop.phi.history[-1] if loop.phi.history else 0.0
        stats = loop.drift_detector.stats()
        flux_enabled = output is not None

        r = {
            "round": len(rounds) + 1,
            "phase": "DRIFT",
            "topic": topic[:40],
            "output": output[:60] if output else None,
            "reason": reason[:80],
            "phi": round(phi, 4) if phi else 0.0,
            "cv": stats["current_cv"],
            "decay": stats["decay"],
            "stage": stats["stage"],
            "is_drifting": stats["is_drifting"],
            "flux_enabled": flux_enabled,
            "status": "PASS" if flux_enabled else "BLOCKED",
        }
        rounds.append(r)

        status = "FLUX" if flux_enabled else "BLOCK"
        drift_mark = " ⚡DRIFT" if stats["is_drifting"] else ""
        print(f"  Round {r['round']} [{r['stage']:10s}] {status} | "
              f"Φ={phi:.3f} | CV={stats['current_cv']:.4f} | γ={stats['decay']:.4f} | "
              f"output={'OK' if output else 'None'}{drift_mark}")

        time.sleep(0.5)

    print(f"  DRIFT FLUX: {sum(1 for r in rounds[-3:] if r['flux_enabled'])}/{len(TOPICS_DRIFT)}")
    print()

    # ── Phase 3: RECOVERY (3 rounds) ─────────────────────────────────
    print("[4] Phase 3: RECOVERY (3 rounds)...")
    for i, topic in enumerate(TOPICS_RECOVERY):
        user_input = f"请详细解释以下话题：{topic}"
        env = {"topic": topic, "phase": "RECOVERY"}

        output, reason = loop.step(env, user_input)

        phi = loop.phi.history[-1] if loop.phi.history else 0.0
        stats = loop.drift_detector.stats()
        flux_enabled = output is not None

        r = {
            "round": len(rounds) + 1,
            "phase": "RECOVERY",
            "topic": topic[:40],
            "output": output[:60] if output else None,
            "reason": reason[:80],
            "phi": round(phi, 4) if phi else 0.0,
            "cv": stats["current_cv"],
            "decay": stats["decay"],
            "stage": stats["stage"],
            "is_drifting": stats["is_drifting"],
            "flux_enabled": flux_enabled,
            "status": "PASS" if flux_enabled else "BLOCKED",
        }
        rounds.append(r)

        status = "FLUX" if flux_enabled else "BLOCK"
        print(f"  Round {r['round']} [{r['stage']:10s}] {status} | "
              f"Φ={phi:.3f} | CV={stats['current_cv']:.4f} | γ={stats['decay']:.4f} | "
              f"output={'OK' if output else 'None'}")

        time.sleep(0.5)

    print(f"  RECOVERY FLUX: {sum(1 for r in rounds[-3:] if r['flux_enabled'])}/{len(TOPICS_RECOVERY)}")

    # ── Hallucination probes ─────────────────────────────────────────
    print()
    print("[5] 幻觉探测 (5 probes)...")
    hallucination_results = []
    for i, (probe, expected) in enumerate(zip(HALLUCINATION_PROBES, HALLUCINATION_EXPECTED_PATTERNS)):
        env = {"topic": "hallucination_check", "phase": "PROBE"}
        output, reason = loop.step(env, probe)

        # Check if output contains any expected pattern
        matched = False
        if output:
            output_lower = output.lower()
            matched = any(pat.lower() in output_lower for pat in expected)

        hallucination_results.append({
            "id": i + 1,
            "question": probe,
            "output": output[:100] if output else None,
            "reason": reason[:80],
            "matched": matched,
            "pass": matched or (output is None),
        })

        status = "✓" if matched or output is None else "✗ HALLUCINATION"
        print(f"  Probe {i+1}: {status} | {probe[:40]}...")

    n_hallucination_pass = sum(1 for h in hallucination_results if h["pass"])
    print(f"  幻觉通过: {n_hallucination_pass}/{len(HALLUCINATION_PROBES)}")
    print()

    # ── Summary ──────────────────────────────────────────────────────
    cv_sequence = [r["cv"] for r in rounds]
    decay_sequence = [r["decay"] for r in rounds]
    stage_sequence = [r["stage"] for r in rounds]
    phi_sequence = [r["phi"] for r in rounds]

    flux_total = sum(1 for r in rounds if r["flux_enabled"])
    flux_ratio = flux_total / len(rounds)

    summary = {
        "version": "v5.0.0",
        "change": "continuous auto-tune decay (sigmoid + slope_factor)",
        "model": MODEL,
        "total_rounds": len(rounds),
        "flux_enabled_rounds": flux_total,
        "flux_ratio": round(flux_ratio, 4),
        "mean_phi": round(np.mean(phi_sequence), 4),
        "final_cv": cv_sequence[-1] if cv_sequence else 0.0,
        "hallucination_pass": f"{n_hallucination_pass}/{len(HALLUCINATION_PROBES)}",
    }

    print("=" * 70)
    print("  SUMMARY")
    print("=" * 70)
    print(f"  FLUX_ENABLED: {flux_total}/{len(rounds)} = {flux_ratio*100:.1f}%")
    print(f"  Φ mean: {summary['mean_phi']:.4f}")
    print(f"  Final CV: {summary['final_cv']:.4f}")
    print(f"  Hallucination: {summary['hallucination_pass']}")
    print(f"  Decay range: [{min(decay_sequence):.4f}, {max(decay_sequence):.4f}]")
    print()

    # ── Per-round table ──────────────────────────────────────────────
    print(f"{'#':>3s} {'Phase':<10s} {'FLUX':>5s} {'Φ':>7s} {'CV':>7s} {'γ':>7s} {'Stage':<12s} {'Topic'}")
    print("-" * 80)
    for r in rounds:
        flux = "YES" if r["flux_enabled"] else " NO"
        print(f"{r['round']:3d} {r['phase']:<10s} {flux:>5s} {r['phi']:7.3f} {r['cv']:7.4f} "
              f"{r['decay']:7.4f} {r['stage']:<12s} {r['topic'][:25]}")
    print()

    # ── Save ─────────────────────────────────────────────────────────
    result = {
        "timestamp": datetime.now().isoformat(),
        "version": "v5.0.0",
        "change": "continuous auto-tune decay (sigmoid + slope_factor)",
        "model": MODEL,
        "rounds": rounds,
        "hallucination_probes": hallucination_results,
        "cv_sequence": cv_sequence,
        "decay_sequence": decay_sequence,
        "stage_sequence": stage_sequence,
        "phi_sequence": phi_sequence,
        "summary": summary,
    }

    out_path = RESULTS_DIR / "delta_e2e_v5_0_0.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"  Results saved: {out_path}")

    return result


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    run_e2e()
