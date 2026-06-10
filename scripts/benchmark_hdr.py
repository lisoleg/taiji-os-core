#!/usr/bin/env python3
"""
benchmark_hdr.py — HDR (Hallucination Detection Rate) 标准化 Benchmark

评估 Φ 门控对矛盾检测的性能。支持两种模式:
  --offline : 使用确定性哈希嵌入（无需 API key，可复现）
  --online  : 使用 DeepSeek API 语义检测

指标:
  - Accuracy, Precision, Recall, F1
  - 按矛盾类型分组的性能分析
  - 阈值扫描曲线数据

用法:
  python scripts/benchmark_hdr.py --offline
  python scripts/benchmark_hdr.py --online --threshold 0.65
  python scripts/benchmark_hdr.py --sweep  # 阈值扫描 0.3~0.95
  python scripts/benchmark_hdr.py --dataset truthfulqa --online  # TruthfulQA 评估
"""

import json
import os
import sys
import time
import argparse
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from core.world_model import WorldModel
from core.phi_scheduler import PhiScheduler
from core.self_consistency_loop import SelfConsistencyLoop

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "test_sets")
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")

def load_dataset(name: str) -> dict:
    with open(os.path.join(DATA_DIR, name), "r", encoding="utf-8") as f:
        return json.load(f)

def compute_metrics(y_true, y_pred):
    tp = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 1)
    fp = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 1)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 0)
    tn = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 0)
    acc = (tp + tn) / max(len(y_true), 1)
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-8)
    return {"accuracy": round(acc,4), "precision": round(prec,4),
            "recall": round(rec,4), "f1": round(f1,4),
            "tp": tp, "fp": fp, "fn": fn, "tn": tn, "total": len(y_true)}

def run_benchmark(args):
    """运行完整 HDR benchmark。"""
    # 加载数据集
    contradictions = load_dataset("hdr_contradictions.json")
    consistent = load_dataset("hdr_consistent.json")

    print(f"Loaded {len(contradictions['entries'])} contradictions + "
          f"{len(consistent['entries'])} consistent pairs")

    # 初始化组件
    wm = WorldModel()
    phi = PhiScheduler(threshold=args.threshold, mode=args.phi_mode)
    dcore = SelfConsistencyLoop(online=args.online)

    # 建立 ψ 基准（用一致性数据"训练"世界模型）
    print("Warming up world model with consistent data...")
    for entry in consistent["entries"]:
        wm.update(entry["statement_a"])
        wm.update(entry["statement_b"])

    results = {
        "config": {
            "threshold": phi.threshold, "phi_mode": phi.mode,
            "dcore_mode": dcore.mode, "online": args.online,
        },
        "overall": {},
        "by_category": {},
        "per_sample": [],
    }

    y_true, y_pred = [], []
    cat_stats = defaultdict(lambda: {"y_true": [], "y_pred": []})

    # 测试矛盾正例
    for entry in contradictions["entries"]:
        wm.update(entry["statement_a"])
        vec = wm.encode(entry["statement_b"])
        ok, phi_val = phi.check(wm, vec)
        is_contra = not ok

        # 也跑语义检测
        sem_contra, _, method = dcore.detect_contradiction(
            entry["statement_a"], entry["statement_b"]
        )

        y_true.append(1)  # 真值: 矛盾
        y_pred.append(1 if is_contra else 0)

        cat_stats[entry["category"]]["y_true"].append(1)
        cat_stats[entry["category"]]["y_pred"].append(1 if is_contra else 0)

        results["per_sample"].append({
            "id": entry["id"], "category": entry["category"],
            "label": "contradiction", "phi": round(phi_val, 4),
            "predicted": 1 if is_contra else 0,
            "semantic_method": method,
        })

    # 测试一致性负例
    for entry in consistent["entries"]:
        wm.update(entry["statement_a"])
        vec = wm.encode(entry["statement_b"])
        ok, phi_val = phi.check(wm, vec)
        is_contra = not ok

        y_true.append(0)  # 真值: 一致
        y_pred.append(1 if is_contra else 0)

        results["per_sample"].append({
            "id": entry["id"], "category": "consistent",
            "label": "consistent", "phi": round(phi_val, 4),
            "predicted": 1 if is_contra else 0,
            "semantic_method": "n/a",
        })

    # 计算指标
    results["overall"] = compute_metrics(y_true, y_pred)
    for cat, stats in cat_stats.items():
        results["by_category"][cat] = compute_metrics(stats["y_true"], stats["y_pred"])

    results["phi_stats"] = {
        "contradiction_mean": round(float(np.mean([
            s["phi"] for s in results["per_sample"] if s["label"] == "contradiction"
        ])), 4),
        "consistent_mean": round(float(np.mean([
            s["phi"] for s in results["per_sample"] if s["label"] == "consistent"
        ])), 4),
    }

    return results


def run_truthfulqa_benchmark(args):
    """运行 TruthfulQA 数据集矛盾检测评估。

    评估逻辑:
      - correct vs incorrect → 矛盾对 (label=1)
      - correct vs correct (不同表述) → 一致对 (label=0)
    """
    tqa = load_dataset("truthfulqa_subset.json")
    entries = tqa["entries"]
    print(f"Loaded TruthfulQA subset: {len(entries)} entries")

    # 初始化组件
    wm = WorldModel()
    phi = PhiScheduler(threshold=args.threshold, mode=args.phi_mode)
    dcore = SelfConsistencyLoop(online=args.online)

    # 用所有正确答案预热世界模型
    print("Warming up world model with correct answers...")
    for entry in entries:
        for ans in entry["correct_answers"]:
            wm.update(ans)

    results = {
        "config": {
            "threshold": phi.threshold, "phi_mode": phi.mode,
            "dcore_mode": dcore.mode, "online": args.online,
            "dataset": "truthfulqa_subset",
            "source": tqa.get("metadata", {}).get("source", "unknown"),
        },
        "overall": {},
        "contradiction_pairs": {},
        "consistent_pairs": {},
        "per_sample": [],
    }

    y_true, y_pred = [], []
    c_y_true, c_y_pred = [], []  # 矛盾对
    s_y_true, s_y_pred = [], []  # 一致对

    for entry in entries:
        question = entry["question"]
        correct_answers = entry["correct_answers"]
        incorrect_answers = entry["incorrect_answers"]

        # 矛盾对: 正确答案 vs 错误答案
        for correct_ans in correct_answers:
            for incorrect_ans in incorrect_answers:
                wm.update(correct_ans)
                vec = wm.encode(incorrect_ans)
                ok, phi_val = phi.check(wm, vec)
                is_contra = not ok

                y_true.append(1)
                y_pred.append(1 if is_contra else 0)
                c_y_true.append(1)
                c_y_pred.append(1 if is_contra else 0)

                results["per_sample"].append({
                    "id": entry["id"],
                    "type": "correct_vs_incorrect",
                    "label": "contradiction",
                    "phi": round(phi_val, 4),
                    "predicted": 1 if is_contra else 0,
                })

        # 一致对: 正确答案 vs 正确答案 (不同表述)
        if len(correct_answers) >= 2:
            for i in range(len(correct_answers)):
                for j in range(i + 1, len(correct_answers)):
                    wm.update(correct_answers[i])
                    vec = wm.encode(correct_answers[j])
                    ok, phi_val = phi.check(wm, vec)
                    is_contra = not ok

                    y_true.append(0)
                    y_pred.append(1 if is_contra else 0)
                    s_y_true.append(0)
                    s_y_pred.append(1 if is_contra else 0)

                    results["per_sample"].append({
                        "id": entry["id"],
                        "type": "correct_vs_correct",
                        "label": "consistent",
                        "phi": round(phi_val, 4),
                        "predicted": 1 if is_contra else 0,
                    })

    # 计算指标
    results["overall"] = compute_metrics(y_true, y_pred)
    if c_y_true:
        results["contradiction_pairs"] = compute_metrics(c_y_true, c_y_pred)
    if s_y_true:
        results["consistent_pairs"] = compute_metrics(s_y_true, s_y_pred)

    if results["per_sample"]:
        contra_phis = [s["phi"] for s in results["per_sample"] if s["label"] == "contradiction"]
        cons_phis = [s["phi"] for s in results["per_sample"] if s["label"] == "consistent"]
        results["phi_stats"] = {
            "contradiction_mean": round(float(np.mean(contra_phis)) if contra_phis else 0, 4),
            "consistent_mean": round(float(np.mean(cons_phis)) if cons_phis else 0, 4),
        }

    return results


def print_truthfulqa_report(results):
    """打印 TruthfulQA 评估报告。"""
    print("\n" + "=" * 60)
    print("  TRUTHFULQA BENCHMARK REPORT")
    print("=" * 60)
    print(f"  Dataset: {results['config'].get('dataset', 'unknown')}")
    print(f"  Source:  {results['config'].get('source', 'unknown')}")
    print(f"  Config: threshold={results['config']['threshold']}, "
          f"phi_mode={results['config']['phi_mode']}, "
          f"dcore={results['config']['dcore_mode']}")

    print(f"\n  Overall:")
    m = results["overall"]
    print(f"    Accuracy:  {m['accuracy']:.4f}")
    print(f"    Precision: {m['precision']:.4f}")
    print(f"    Recall:    {m['recall']:.4f}")
    print(f"    F1 Score:  {m['f1']:.4f}")
    print(f"    TP={m['tp']} FP={m['fp']} FN={m['fn']} TN={m['tn']}")

    if "phi_stats" in results:
        print(f"\n  Φ Statistics:")
        ps = results["phi_stats"]
        print(f"    μ(contradiction): {ps['contradiction_mean']:.4f}")
        print(f"    μ(consistent):    {ps['consistent_mean']:.4f}")
        if ps['consistent_mean'] and ps['contradiction_mean']:
            print(f"    Δ:                {ps['consistent_mean'] - ps['contradiction_mean']:.4f}")

    if "contradiction_pairs" in results and results["contradiction_pairs"].get("total", 0) > 0:
        print(f"\n  Contradiction Pairs (correct vs incorrect):")
        cp = results["contradiction_pairs"]
        print(f"    Accuracy:  {cp['accuracy']:.4f}")
        print(f"    Precision: {cp['precision']:.4f}")
        print(f"    Recall:    {cp['recall']:.4f}")
        print(f"    F1 Score:  {cp['f1']:.4f}")

    if "consistent_pairs" in results and results["consistent_pairs"].get("total", 0) > 0:
        print(f"\n  Consistent Pairs (correct vs correct):")
        sp = results["consistent_pairs"]
        print(f"    Accuracy:  {sp['accuracy']:.4f}")
        print(f"    Precision: {sp['precision']:.4f}")
        print(f"    Recall:    {sp['recall']:.4f}")
        print(f"    F1 Score:  {sp['f1']:.4f}")


def run_threshold_sweep(args):
    """扫描阈值 0.30 ~ 0.95，生成性能曲线。"""
    sweep_results = []
    for t in np.arange(0.30, 0.96, 0.05):
        t = round(t, 2)
        args.threshold = t
        r = run_benchmark(args)
        sweep_results.append({
            "threshold": t,
            "accuracy": r["overall"]["accuracy"],
            "precision": r["overall"]["precision"],
            "recall": r["overall"]["recall"],
            "f1": r["overall"]["f1"],
        })
        print(f"  threshold={t:.2f}  acc={r['overall']['accuracy']:.3f}  "
              f"f1={r['overall']['f1']:.3f}  prec={r['overall']['precision']:.3f}  "
              f"rec={r['overall']['recall']:.3f}")
    return sweep_results


def print_report(results):
    """打印人类可读的报告。"""
    print("\n" + "=" * 60)
    print("  HDR BENCHMARK REPORT")
    print("=" * 60)
    print(f"  Config: threshold={results['config']['threshold']}, "
          f"phi_mode={results['config']['phi_mode']}, "
          f"dcore={results['config']['dcore_mode']}")
    print(f"\n  Overall:")
    m = results["overall"]
    print(f"    Accuracy:  {m['accuracy']:.4f}")
    print(f"    Precision: {m['precision']:.4f}")
    print(f"    Recall:    {m['recall']:.4f}")
    print(f"    F1 Score:  {m['f1']:.4f}")
    print(f"    TP={m['tp']} FP={m['fp']} FN={m['fn']} TN={m['tn']}")

    print(f"\n  Φ Statistics:")
    ps = results["phi_stats"]
    print(f"    μ(contradiction): {ps['contradiction_mean']:.4f}")
    print(f"    μ(consistent):    {ps['consistent_mean']:.4f}")
    print(f"    Δ:                {ps['consistent_mean'] - ps['contradiction_mean']:.4f}")

    print(f"\n  By Category:")
    for cat, m in sorted(results["by_category"].items()):
        print(f"    {cat:20s}  acc={m['accuracy']:.3f}  f1={m['f1']:.3f}  "
              f"prec={m['precision']:.3f}  rec={m['recall']:.3f}")


def main():
    parser = argparse.ArgumentParser(description="HDR Benchmark")
    parser.add_argument("--offline", action="store_true", help="Use offline hash embeddings")
    parser.add_argument("--online", action="store_true", help="Use DeepSeek API")
    parser.add_argument("--threshold", type=float, default=0.65, help="Phi threshold")
    parser.add_argument("--phi-mode", default="static", choices=["static","adaptive"])
    parser.add_argument("--sweep", action="store_true", help="Run threshold sweep")
    parser.add_argument("--dataset", default="hdr", choices=["hdr","truthfulqa"],
                        help="Dataset to benchmark (default: hdr)")
    parser.add_argument("--output", default="", help="Output JSON file path")
    args = parser.parse_args()

    is_truthfulqa = args.dataset == "truthfulqa"

    if args.sweep:
        if is_truthfulqa:
            print("Running TruthfulQA threshold sweep (0.30 ~ 0.95)...")
            sweep_results = []
            for t in np.arange(0.30, 0.96, 0.05):
                t = round(t, 2)
                args.threshold = t
                r = run_truthfulqa_benchmark(args)
                sweep_results.append({
                    "threshold": t,
                    "accuracy": r["overall"]["accuracy"],
                    "precision": r["overall"]["precision"],
                    "recall": r["overall"]["recall"],
                    "f1": r["overall"]["f1"],
                })
                print(f"  threshold={t:.2f}  acc={r['overall']['accuracy']:.3f}  "
                      f"f1={r['overall']['f1']:.3f}  prec={r['overall']['precision']:.3f}  "
                      f"rec={r['overall']['recall']:.3f}")
            os.makedirs(OUTPUT_DIR, exist_ok=True)
            out_file = args.output or os.path.join(OUTPUT_DIR, "hdr_truthfulqa_sweep.json")
        else:
            print("Running threshold sweep (0.30 ~ 0.95)...")
            sweep = run_threshold_sweep(args)
            os.makedirs(OUTPUT_DIR, exist_ok=True)
            out_file = args.output or os.path.join(OUTPUT_DIR, "hdr_sweep.json")
            with open(out_file, "w", encoding="utf-8") as f:
                json.dump({"sweep_results": sweep}, f, ensure_ascii=False, indent=2)
            print(f"\nSweep results saved to {out_file}")

            # 输出 LaTeX 表格
            print("\n% LaTeX table:")
            print("\\begin{tabular}{lcccc}")
            print("\\toprule")
            print("Threshold & Accuracy & Precision & Recall & F1 \\\\")
            print("\\midrule")
            for r in sweep:
                print(f"{r['threshold']:.2f} & {r['accuracy']:.3f} & "
                      f"{r['precision']:.3f} & {r['recall']:.3f} & {r['f1']:.3f} \\\\")
            print("\\bottomrule")
            print("\\end{tabular}")
            return

        with open(out_file, "w", encoding="utf-8") as f:
            json.dump({"sweep_results": sweep_results}, f, ensure_ascii=False, indent=2)
        print(f"\nSweep results saved to {out_file}")
    elif is_truthfulqa:
        print(f"Running TruthfulQA benchmark (threshold={args.threshold}, mode={args.phi_mode})...")
        results = run_truthfulqa_benchmark(args)
        print_truthfulqa_report(results)

        os.makedirs(OUTPUT_DIR, exist_ok=True)
        out_file = args.output or os.path.join(OUTPUT_DIR, "hdr_truthfulqa_results.json")
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"\nDetailed results saved to {out_file}")
    else:
        print(f"Running HDR benchmark (threshold={args.threshold}, mode={args.phi_mode})...")
        results = run_benchmark(args)
        print_report(results)

        os.makedirs(OUTPUT_DIR, exist_ok=True)
        out_file = args.output or os.path.join(OUTPUT_DIR, "hdr_results.json")
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"\nDetailed results saved to {out_file}")


if __name__ == "__main__":
    main()
