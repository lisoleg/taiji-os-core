#!/usr/bin/env python3
"""
benchmark_scs.py — SCS (Semantic Consistency Score) 标准化 Benchmark

评估 ψ 的稳定性（稳定序列）和漂移检测灵敏度（漂移序列）。

指标:
  - 稳定序列 CV (变异系数, 越低越好): 衡量 ψ 在一致陈述下的波动
  - 漂移序列 CV (越高越好): 衡量 ψ 对语义漂移的响应
  - 对比率 (drift_CV / stable_CV): 综合指标, 越高越好

用法:
  python scripts/benchmark_scs.py --offline
"""

import json, os, sys, argparse
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.world_model import WorldModel
from core.phi_scheduler import PhiScheduler

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "test_sets")
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")

def load_dataset(name):
    with open(os.path.join(DATA_DIR, name), "r", encoding="utf-8") as f:
        return json.load(f)

def compute_sequence_cv(wm, statements):
    """计算一个陈述序列的 ψ CV。"""
    phi_values = []
    for stmt in statements:
        vec = wm.encode(stmt)
        phi_val = wm.phi(vec) if wm.version > 0 else 1.0
        phi_values.append(phi_val)
        wm.update(stmt)

    vals = np.array(phi_values[1:])  # 跳过第一个（无参照）
    if len(vals) < 2:
        return 0.0
    mu = np.mean(vals)
    sigma = np.std(vals)
    return float(sigma / (mu + 1e-8))

def run_benchmark(args):
    stable = load_dataset("scs_stable.json")
    drift = load_dataset("scs_drift.json")

    print(f"Loaded {len(stable['entries'])} stable + {len(drift['entries'])} drift sequences")

    results = {"config": {"phi_mode": args.phi_mode}, "stable": [], "drift": [], "summary": {}}

    # 稳定序列
    stable_cvs = []
    for entry in stable["entries"]:
        wm = WorldModel()
        cv = compute_sequence_cv(wm, entry["statements"])
        stable_cvs.append(cv)
        results["stable"].append({
            "id": entry["id"], "topic": entry["topic"],
            "n_statements": len(entry["statements"]), "cv": round(cv, 6),
        })

    # 漂移序列
    drift_cvs = []
    for entry in drift["entries"]:
        wm = WorldModel()
        cv = compute_sequence_cv(wm, entry["statements"])
        drift_cvs.append(cv)
        results["drift"].append({
            "id": entry["id"], "topic": entry["topic"],
            "n_statements": len(entry["statements"]), "cv": round(cv, 6),
            "drift_type": entry.get("drift_type", "unknown"),
        })

    # 汇总
    results["summary"] = {
        "stable_mean_cv": round(float(np.mean(stable_cvs)), 6),
        "stable_median_cv": round(float(np.median(stable_cvs)), 6),
        "stable_std_cv": round(float(np.std(stable_cvs)), 6),
        "drift_mean_cv": round(float(np.mean(drift_cvs)), 6),
        "drift_median_cv": round(float(np.median(drift_cvs)), 6),
        "drift_std_cv": round(float(np.std(drift_cvs)), 6),
        "contrast_ratio": round(float(np.mean(drift_cvs) / (np.mean(stable_cvs) + 1e-8)), 3),
        "effect_size": round(float(
            (np.mean(drift_cvs) - np.mean(stable_cvs)) / (np.std(stable_cvs + drift_cvs) + 1e-8)
        ), 3),
    }

    return results


def print_report(results):
    s = results["summary"]
    print("\n" + "=" * 60)
    print("  SCS BENCHMARK REPORT")
    print("=" * 60)
    print(f"\n  Stable Sequences ({len(results['stable'])}):")
    print(f"    Mean CV:  {s['stable_mean_cv']:.6f}")
    print(f"    Median CV:{s['stable_median_cv']:.6f}")
    print(f"    Std CV:   {s['stable_std_cv']:.6f}")
    print(f"\n  Drift Sequences ({len(results['drift'])}):")
    print(f"    Mean CV:  {s['drift_mean_cv']:.6f}")
    print(f"    Median CV:{s['drift_median_cv']:.6f}")
    print(f"    Std CV:   {s['drift_std_cv']:.6f}")
    print(f"\n  Composite:")
    print(f"    Contrast Ratio: {s['contrast_ratio']}")
    print(f"    Effect Size:    {s['effect_size']}")

    # 分类分析
    print(f"\n  By Drift Type:")
    for dt in ["topic_transition", "incremental_contradiction"]:
        cvs = [e["cv"] for e in results["drift"] if e["drift_type"] == dt]
        if cvs:
            print(f"    {dt:30s} n={len(cvs)} mean_cv={np.mean(cvs):.6f}")


def main():
    parser = argparse.ArgumentParser(description="SCS Benchmark")
    parser.add_argument("--offline", action="store_true", default=True)
    parser.add_argument("--phi-mode", default="static", choices=["static","adaptive"])
    parser.add_argument("--output", default="")
    args = parser.parse_args()

    results = run_benchmark(args)
    print_report(results)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out = args.output or os.path.join(OUTPUT_DIR, "scs_results.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\nResults saved to {out}")


if __name__ == "__main__":
    main()
