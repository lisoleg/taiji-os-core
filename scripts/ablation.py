#!/usr/bin/env python3
"""
ablation.py — Φ 门控消融实验矩阵

七组消融实验:
  1. D-Core: 语义检测 vs 关键词匹配
  2. Φ 阈值扫描: 0.30 ~ 0.95, step 0.05
  3. Adaptive vs Static Φ
  4. 嵌入对比: DeepSeek API vs 确定性哈希
  5. ψ EMA 衰减率扫描
  6. 矛盾类型消融（每种类型单独评估）
  7. GPT-4 Baseline vs D-Core 对比

输出:
  - results/ablation_report.json — 机器可读完整结果
  - LaTeX 表格 — 可直接复制到论文

用法:
  python scripts/ablation.py --offline
  python scripts/ablation.py --offline --quick  # 快速模式(仅核心消融)
  python scripts/ablation.py --online --gpt4     # 在线模式 + GPT-4 对比
  python scripts/ablation.py --online --gpt4 --latex  # 输出 LaTeX
"""

import json, os, sys, argparse, time
from collections import defaultdict
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.world_model import WorldModel
from core.phi_scheduler import PhiScheduler
from core.self_consistency_loop import SelfConsistencyLoop

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "test_sets")
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")

def load_json(n):
    with open(os.path.join(DATA_DIR, n), "r", encoding="utf-8") as f:
        return json.load(f)

def metrics(y_true, y_pred):
    tp = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 1)
    fp = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 1)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 0)
    tn = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 0)
    acc = (tp + tn) / max(len(y_true), 1)
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-8)
    return {"accuracy": round(acc,4), "precision": round(prec,4),
            "recall": round(rec,4), "f1": round(f1,4), "total": len(y_true)}

# ============ Experiment 1: D-Core semantic vs keyword ============

def ablation_dcore():
    print("\n[1/6] D-Core: Semantic vs Keyword")
    contradictions = load_json("hdr_contradictions.json")
    consistent = load_json("hdr_consistent.json")

    # Keyword method
    kw_y_true, kw_y_pred = [], []
    for e in contradictions["entries"]:
        combined = e["statement_a"] + " " + e["statement_b"]
        kw_y_true.append(1)
        kw_y_pred.append(1 if any(
            kw in combined for kw in ["矛盾","不一致","冲突","相反","contradiction","inconsistent"]
        ) else 0)
    for e in consistent["entries"]:
        combined = e["statement_a"] + " " + e["statement_b"]
        kw_y_true.append(0)
        kw_y_pred.append(1 if any(
            kw in combined for kw in ["矛盾","不一致","冲突","相反","contradiction","inconsistent"]
        ) else 0)
    kw_result = metrics(kw_y_true, kw_y_pred)

    # Semantic method (offline = keyword fallback, but with more keywords)
    sem = SelfConsistencyLoop(online=False)
    sem_y_true, sem_y_pred = [], []
    for e in contradictions["entries"]:
        is_c, _, method = sem.detect_contradiction(e["statement_a"], e["statement_b"])
        sem_y_true.append(1)
        sem_y_pred.append(1 if is_c else 0)
    for e in consistent["entries"]:
        is_c, _, method = sem.detect_contradiction(e["statement_a"], e["statement_b"])
        sem_y_true.append(0)
        sem_y_pred.append(1 if is_c else 0)
    sem_result = metrics(sem_y_true, sem_y_pred)

    return {
        "keyword": kw_result,
        "semantic_offline": sem_result,
        "delta_f1": round(sem_result["f1"] - kw_result["f1"], 4),
    }


# ============ Experiment 2: Φ threshold sweep ============

def ablation_threshold_sweep():
    print("\n[2/6] Φ Threshold Sweep (0.30 ~ 0.95)")
    contradictions = load_json("hdr_contradictions.json")
    consistent = load_json("hdr_consistent.json")

    sweep = []
    for t in np.arange(0.30, 0.96, 0.05):
        t = round(t, 2)
        phi = PhiScheduler(threshold=t)
        y_true, y_pred = [], []

        for e in contradictions["entries"]:
            wm = WorldModel()
            wm.update(e["statement_a"])
            ok, _ = phi.check(wm, wm.encode(e["statement_b"]))
            y_true.append(1); y_pred.append(1 if not ok else 0)
        for e in consistent["entries"]:
            wm = WorldModel()
            wm.update(e["statement_a"])
            ok, _ = phi.check(wm, wm.encode(e["statement_b"]))
            y_true.append(0); y_pred.append(1 if not ok else 0)

        m = metrics(y_true, y_pred)
        sweep.append({"threshold": t, **m})

    # Find optimal
    best = max(sweep, key=lambda x: x["f1"])
    return {"sweep": sweep, "best_threshold": best["threshold"], "best_f1": best["f1"]}


# ============ Experiment 3: Adaptive vs Static ============

def ablation_adaptive_vs_static():
    print("\n[3/6] Adaptive vs Static Φ")
    contradictions = load_json("hdr_contradictions.json")
    consistent = load_json("hdr_consistent.json")

    results = {}
    for mode in ["static", "adaptive"]:
        phi = PhiScheduler(mode=mode)
        y_true, y_pred = [], []

        for e in contradictions["entries"]:
            wm = WorldModel()
            wm.update(e["statement_a"])
            ok, _ = phi.check(wm, wm.encode(e["statement_b"]))
            y_true.append(1); y_pred.append(1 if not ok else 0)
        for e in consistent["entries"]:
            wm = WorldModel()
            wm.update(e["statement_a"])
            ok, _ = phi.check(wm, wm.encode(e["statement_b"]))
            y_true.append(0); y_pred.append(1 if not ok else 0)

        results[mode] = {
            "metrics": metrics(y_true, y_pred),
            "phi_stats": phi.stats(),
        }

    return results


# ============ Experiment 4: Embedding comparison ============

def ablation_embedding_comparison():
    print("\n[4/6] Embedding: DeepSeek API vs Hash")

    # Hash embeddings (always available)
    contradictions = load_json("hdr_contradictions.json")
    consistent = load_json("hdr_consistent.json")

    phi = PhiScheduler()
    y_true, y_pred = [], []

    for e in contradictions["entries"]:
        wm = WorldModel()
        wm.update(e["statement_a"])
        ok, _ = phi.check(wm, wm.encode(e["statement_b"]))
        y_true.append(1); y_pred.append(1 if not ok else 0)
    for e in consistent["entries"]:
        wm = WorldModel()
        wm.update(e["statement_a"])
        ok, _ = phi.check(wm, wm.encode(e["statement_b"]))
        y_true.append(0); y_pred.append(1 if not ok else 0)

    hash_result = metrics(y_true, y_pred)

    return {
        "hash_embedding": hash_result,
        "api_embedding": "requires DEEPSEEK_API_KEY",
        "note": "确定性哈希嵌入结果; DeepSeek API 嵌入需要有效的 API key",
    }


# ============ Experiment 5: EMA decay rate sweep ============

def ablation_ema_sweep():
    print("\n[5/6] ψ EMA Decay Rate Sweep")
    contradictions = load_json("hdr_contradictions.json")
    consistent = load_json("hdr_consistent.json")

    sweep = []
    for decay in [0.7, 0.8, 0.9, 0.95, 0.99]:
        phi = PhiScheduler()
        y_true, y_pred = [], []

        for e in contradictions["entries"]:
            wm = WorldModel()
            wm._decay = decay
            wm.update(e["statement_a"])
            ok, _ = phi.check(wm, wm.encode(e["statement_b"]))
            y_true.append(1); y_pred.append(1 if not ok else 0)
        for e in consistent["entries"]:
            wm = WorldModel()
            wm._decay = decay
            wm.update(e["statement_a"])
            ok, _ = phi.check(wm, wm.encode(e["statement_b"]))
            y_true.append(0); y_pred.append(1 if not ok else 0)

        sweep.append({"decay": decay, **metrics(y_true, y_pred)})

    best = max(sweep, key=lambda x: x["f1"])
    return {"sweep": sweep, "best_decay": best["decay"], "best_f1": best["f1"]}


# ============ Experiment 6: Per-category ablation ============

def ablation_per_category():
    print("\n[6/6] Per-Category Ablation")
    contradictions = load_json("hdr_contradictions.json")
    consistent = load_json("hdr_consistent.json")

    phi = PhiScheduler()
    cat_stats = defaultdict(lambda: {"y_true": [], "y_pred": []})

    for e in contradictions["entries"]:
        wm = WorldModel()
        wm.update(e["statement_a"])
        ok, _ = phi.check(wm, wm.encode(e["statement_b"]))
        cat_stats[e["category"]]["y_true"].append(1)
        cat_stats[e["category"]]["y_pred"].append(1 if not ok else 0)

    results = {}
    for cat, stats in sorted(cat_stats.items()):
        results[cat] = metrics(stats["y_true"], stats["y_pred"])

    return results


# ============ Experiment 7: GPT-4 Baseline vs D-Core ============

def ablation_gpt4_vs_dcore(online=False):
    """E7: GPT-4o 零样本 vs D-Core (DeepSeek) 矛盾检测对比。"""
    print("\n[7/7] GPT-4 Baseline vs D-Core")

    # 检查 GPT-4 可用性
    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        print("  [SKIP] OPENAI_API_KEY 未设置，跳过 GPT-4 baseline 实验")
        return {"status": "skipped", "reason": "OPENAI_API_KEY not set"}

    try:
        from core.gpt4_baseline import GPT4ContradictionDetector
        gpt4 = GPT4ContradictionDetector()
    except Exception as e:
        print(f"  [SKIP] GPT-4 初始化失败: {e}")
        return {"status": "skipped", "reason": f"GPT4 init failed: {e}"}

    if not gpt4.is_available:
        print("  [SKIP] GPT-4 不可用")
        return {"status": "skipped", "reason": "GPT4 not available"}

    # 加载数据
    contradictions = load_json("hdr_contradictions.json")
    consistent = load_json("hdr_consistent.json")

    # D-Core 在线检测
    dcore = SelfConsistencyLoop(online=online)
    gpt4_y_true, gpt4_y_pred = [], []
    dcore_y_true, dcore_y_pred = [], []
    latencies = {"gpt4": [], "dcore": []}

    # 矛盾对
    for e in contradictions["entries"]:
        # GPT-4
        t0 = time.perf_counter()
        r = gpt4.detect(e["statement_a"], e["statement_b"])
        gpt4_lat = (time.perf_counter() - t0) * 1000
        latencies["gpt4"].append(gpt4_lat)
        gpt4_y_true.append(1)
        gpt4_y_pred.append(1 if r["is_contradiction"] else 0)

        # D-Core
        t0 = time.perf_counter()
        is_c, _, _ = dcore.detect_contradiction(e["statement_a"], e["statement_b"])
        dcore_lat = (time.perf_counter() - t0) * 1000
        latencies["dcore"].append(dcore_lat)
        dcore_y_true.append(1)
        dcore_y_pred.append(1 if is_c else 0)

    # 一致对
    for e in consistent["entries"]:
        # GPT-4
        t0 = time.perf_counter()
        r = gpt4.detect(e["statement_a"], e["statement_b"])
        gpt4_lat = (time.perf_counter() - t0) * 1000
        latencies["gpt4"].append(gpt4_lat)
        gpt4_y_true.append(0)
        gpt4_y_pred.append(1 if r["is_contradiction"] else 0)

        # D-Core
        t0 = time.perf_counter()
        is_c, _, _ = dcore.detect_contradiction(e["statement_a"], e["statement_b"])
        dcore_lat = (time.perf_counter() - t0) * 1000
        latencies["dcore"].append(dcore_lat)
        dcore_y_true.append(0)
        dcore_y_pred.append(1 if is_c else 0)

    gpt4_result = metrics(gpt4_y_true, gpt4_y_pred)
    dcore_result = metrics(dcore_y_true, dcore_y_pred)

    gpt4_result["avg_latency_ms"] = round(float(np.mean(latencies["gpt4"])), 1)
    dcore_result["avg_latency_ms"] = round(float(np.mean(latencies["dcore"])), 1)

    return {
        "gpt4_zero_shot": gpt4_result,
        "dcore": dcore_result,
        "delta_f1": round(gpt4_result["f1"] - dcore_result["f1"], 4),
        "delta_accuracy": round(gpt4_result["accuracy"] - dcore_result["accuracy"], 4),
        "status": "ok",
    }


# ============ Main ============

def run_all(args):
    report = {"config": {"mode": "offline" if args.offline else "online",
                          "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S")}}

    report["1_dcore_semantic_vs_keyword"] = ablation_dcore()
    report["2_threshold_sweep"] = ablation_threshold_sweep()
    report["3_adaptive_vs_static"] = ablation_adaptive_vs_static()

    if not args.quick:
        report["4_embedding_comparison"] = ablation_embedding_comparison()
        report["5_ema_sweep"] = ablation_ema_sweep()
        report["6_per_category"] = ablation_per_category()

    # E7: GPT-4 Baseline vs D-Core (仅在 --gpt4 时运行)
    if getattr(args, "gpt4", False):
        report["7_gpt4_vs_dcore"] = ablation_gpt4_vs_dcore(online=not args.offline)

    return report


def print_latex(report):
    """输出可复制的 LaTeX 表格。"""
    print("\n" + "=" * 60)
    print("  LATEX TABLES FOR PAPER")
    print("=" * 60)

    # Table 1: D-Core Ablation
    if "1_dcore_semantic_vs_keyword" in report:
        r = report["1_dcore_semantic_vs_keyword"]
        print("\n% Table: D-Core Semantic vs Keyword")
        print("\\begin{tabular}{lcccc}")
        print("\\toprule")
        print("Method & Accuracy & Precision & Recall & F1 \\\\")
        print("\\midrule")
        for name in ["keyword", "semantic_offline"]:
            m = r[name]
            print(f"{name} & {m['accuracy']:.3f} & {m['precision']:.3f} & "
                  f"{m['recall']:.3f} & {m['f1']:.3f} \\\\")
        print("\\bottomrule")
        print("\\end{tabular}")

    # Table 2: Threshold Sweep (abbreviated)
    if "2_threshold_sweep" in report:
        r = report["2_threshold_sweep"]
        print("\n% Table: Φ Threshold Sweep (Selected)")
        print("\\begin{tabular}{lcccc}")
        print("\\toprule")
        print("$\\Phi$ Threshold & Accuracy & Precision & Recall & F1 \\\\")
        print("\\midrule")
        for s in r["sweep"][::2]:  # every other
            print(f"{s['threshold']:.2f} & {s['accuracy']:.3f} & "
                  f"{s['precision']:.3f} & {s['recall']:.3f} & {s['f1']:.3f} \\\\")
        print("\\bottomrule")
        print("\\end{tabular}")
        print(f"\n% Best: threshold={r['best_threshold']}, F1={r['best_f1']}")

    # Table 3: Per-Category
    if "6_per_category" in report:
        r = report["6_per_category"]
        print("\n% Table: Per-Category Performance")
        print("\\begin{tabular}{lcccc}")
        print("\\toprule")
        print("Category & Accuracy & Precision & Recall & F1 \\\\")
        print("\\midrule")
        for cat, m in sorted(r.items()):
            print(f"{cat} & {m['accuracy']:.3f} & {m['precision']:.3f} & "
                  f"{m['recall']:.3f} & {m['f1']:.3f} \\\\")
        print("\\bottomrule")
        print("\\end{tabular}")

    # Table 4: GPT-4 vs D-Core
    if "7_gpt4_vs_dcore" in report:
        r = report["7_gpt4_vs_dcore"]
        if r.get("status") == "ok":
            print("\n% Table: GPT-4 Baseline vs D-Core")
            print("\\begin{tabular}{lccccc}")
            print("\\toprule")
            print("Method & Accuracy & Precision & Recall & F1 & Latency (ms) \\\\")
            print("\\midrule")
            for method in ["dcore", "gpt4_zero_shot"]:
                m = r[method]
                lat = m.get("avg_latency_ms", "N/A")
                print(f"{method} & {m['accuracy']:.3f} & {m['precision']:.3f} & "
                      f"{m['recall']:.3f} & {m['f1']:.3f} & {lat} \\\\")
            print("\\bottomrule")
            print("\\end{tabular}")
            print(f"\n% Delta F1: {r['delta_f1']}, Delta Acc: {r['delta_accuracy']}")


def main():
    parser = argparse.ArgumentParser(description="Φ Gating Ablation Experiments")
    parser.add_argument("--offline", action="store_true", default=True)
    parser.add_argument("--online", action="store_true", help="Online mode (use API)")
    parser.add_argument("--quick", action="store_true", help="Quick mode (core ablation only)")
    parser.add_argument("--gpt4", action="store_true", help="Include E7: GPT-4 vs D-Core comparison")
    parser.add_argument("--output", default="")
    parser.add_argument("--latex", action="store_true", help="Output LaTeX tables")
    args = parser.parse_args()

    # --online flag 覆盖默认 offline
    if args.online:
        args.offline = False

    total_exps = 4 if args.quick else 7
    print("=" * 60)
    print(f"  Φ GATING ABLATION EXPERIMENTS ({total_exps} experiments)")
    print("=" * 60)

    report = run_all(args)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out = args.output or os.path.join(OUTPUT_DIR, "ablation_report.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\nFull report saved to {out}")

    # Summary
    print("\n=== ABLATION SUMMARY ===")
    if "1_dcore_semantic_vs_keyword" in report:
        r1 = report["1_dcore_semantic_vs_keyword"]
        print(f"D-Core: keyword F1={r1['keyword']['f1']}, semantic F1={r1['semantic_offline']['f1']}, Δ={r1['delta_f1']}")
    if "2_threshold_sweep" in report:
        r2 = report["2_threshold_sweep"]
        print(f"Threshold: best={r2['best_threshold']} (F1={r2['best_f1']})")
    if "3_adaptive_vs_static" in report:
        r3 = report["3_adaptive_vs_static"]
        for mode in ["static","adaptive"]:
            m = r3[mode]["metrics"]
            print(f"Φ {mode}: acc={m['accuracy']}, f1={m['f1']}")
    if "7_gpt4_vs_dcore" in report:
        r7 = report["7_gpt4_vs_dcore"]
        if r7.get("status") == "ok":
            dcore_m = r7["dcore"]
            gpt4_m = r7["gpt4_zero_shot"]
            print(f"E7 GPT-4 vs D-Core:")
            print(f"  D-Core:  acc={dcore_m['accuracy']}, f1={dcore_m['f1']}, "
                  f"latency={dcore_m.get('avg_latency_ms','N/A')}ms")
            print(f"  GPT-4:   acc={gpt4_m['accuracy']}, f1={gpt4_m['f1']}, "
                  f"latency={gpt4_m.get('avg_latency_ms','N/A')}ms")
            print(f"  ΔF1={r7['delta_f1']}, ΔAcc={r7['delta_accuracy']}")
        else:
            print(f"E7 GPT-4: {r7.get('reason', 'skipped')}")
    elif getattr(args, "gpt4", False):
        print("E7 GPT-4: experiment failed")

    if args.latex:
        print_latex(report)


if __name__ == "__main__":
    main()
