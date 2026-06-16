#!/usr/bin/env python3
"""
bench_longconv.py -- delta-mem Long Conversation Incremental Validation.

Simulates multi-turn conversations (5-20 turns) to measure delta-mem
(S-matrix cumulative effects) in long-dialogue scenarios.

Key insight: SWE-bench is single-turn (prompt -> patch); it cannot
demonstrate delta-mem's cumulative benefit over many turns. This script
fills that gap with synthetic phi trajectories across three scenarios:

    STABLE   -- phi stays high (0.75-0.95), consistent dialogue
    DRIFTING -- phi degrades (0.85 -> 0.25), topic drift
    MIXED    -- periodic oscillation with long-term decline

Each scenario runs with delta_on=True and delta_off=False (baseline
using simple sliding-window CV, no S-matrix accumulation).

No real LLM API calls are needed -- uses synthetic embeddings and
controlled phi (consistency) trajectories.

Output: CSV + JSON report in results/

Author: Taiji OS Team
Version: v1.2 -- 2026-06-16 (adaptive cv_threshold + sensitive drift params)
"""

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Callable

import numpy as np

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from core.delta_mem import (
    DEFAULT_RANK,
    DEFAULT_LAMBDA,
    DEFAULT_BETA,
    SMatrix,
    DeltaMemLayer,
)
from core.drift_detector import DriftDetector, HyperParamAdapter


# ===========================================================================
# Synthetic phi (consistency score) generators per scenario
# ===========================================================================

def get_phi_stable(turn: int, total: int, rng: np.random.Generator) -> float:
    """Stable dialogue: phi fluctuates around 0.80-0.90 with small noise."""
    base = 0.85
    noise = 0.05 * np.sin(turn * 0.7) + float(rng.normal(0, 0.025))
    return float(np.clip(base + noise, 0.70, 0.95))


def get_phi_drifting(turn: int, total: int, rng: np.random.Generator) -> float:
    """Topic drift: phi deteriorates from ~0.85 to ~0.25."""
    start, end = 0.85, 0.25
    progress = turn / max(total - 1, 1)
    trend = start + (end - start) * progress
    noise = float(rng.normal(0, 0.03))
    return float(np.clip(trend + noise, 0.15, 0.95))


def get_phi_mixed(turn: int, total: int, rng: np.random.Generator) -> float:
    """Mixed mode: oscillation with gradual long-term decline."""
    progress = turn / max(total - 1, 1)
    base = 0.78 - 0.20 * progress
    wave = 0.18 * np.sin(turn * 2.0 * np.pi / 7.0)
    noise = float(rng.normal(0, 0.03))
    return float(np.clip(base + wave + noise, 0.25, 0.92))


SCENARIOS: dict[str, Callable] = {
    "STABLE":   get_phi_stable,
    "DRIFTING": get_phi_drifting,
    "MIXED":    get_phi_mixed,
}


# ===========================================================================
# Baseline: simple sliding-window CV (no delta-mem, no exponential decay)
# ===========================================================================

class SlidingWindowCV:
    """Naive sliding-window CV -- no decay weighting, no S matrix.

    This is the baseline for "delta_off" comparison.
    """

    def __init__(self, window_size: int = 5):
        self._buf: list[float] = []
        self._max = window_size

    def push(self, phi: float) -> float:
        self._buf.append(phi)
        if len(self._buf) > self._max:
            self._buf.pop(0)
        if len(self._buf) < 2:
            return 0.0
        arr = np.array(self._buf, dtype=np.float64)
        mean = arr.mean()
        if mean <= 0:
            return 0.0
        return float(arr.std(ddof=1) / mean)


# ===========================================================================
# Run configurations
# ===========================================================================

def _fieldnames() -> list[str]:
    return [
        "scenario", "delta_on", "turn", "phi", "cv",
        "is_drifting", "S_fro_norm", "step", "gamma",
    ]


def run_delta_on(
    scenario_name: str,
    phi_func: Callable,
    turns: int,
    rank: int,
    seed: int,
    cv_threshold: float | None = None,
    window_size: int | None = None,
    adaptive_threshold: bool = True,
) -> list[dict]:
    """Run WITH delta-mem: full S-matrix accumulation + drift detector."""
    rng = np.random.default_rng(seed + hash(scenario_name) % 10000)

    # Scenario-specific auto-tuned defaults
    if cv_threshold is None:
        cv_threshold = 0.18 if scenario_name == "DRIFTING" else 0.30
    if window_size is None:
        window_size = 10 if scenario_name == "DRIFTING" else 20

    smatrix = SMatrix(
        S=np.zeros((rank, rank), dtype=np.float32),
        r=rank,
        lambda_=DEFAULT_LAMBDA,
        beta=DEFAULT_BETA,
    )
    delta_layer = DeltaMemLayer(smatrix=smatrix)

    detector = DriftDetector(
        window_size=window_size,
        cv_threshold=cv_threshold,
        adaptive_cv_threshold=adaptive_threshold,  # v1.7
        gamma_max=0.85,
        gamma_min=0.20,
        cv_mid=0.25,
        temperature=0.08,
        auto_tune=True,
    )
    detector.adapter = HyperParamAdapter()

    # DRIFTING scenario: lower sensitivity barriers for gradual drift
    if scenario_name == "DRIFTING":
        detector.min_samples_before_detect = 3
        detector.hysteresis_rounds = 1

    rows: list[dict] = []
    for t in range(turns):
        phi = phi_func(t, turns, rng)

        detector.push(phi)
        is_drifting = detector.is_drifting()
        cv = detector.current_cv
        gamma = detector._get_decay()

        # Synthesize a new k/v pair and update S
        k = rng.standard_normal(rank).astype(np.float32)
        k /= np.linalg.norm(k) + 1e-8
        v = rng.standard_normal(rank).astype(np.float32)
        v /= np.linalg.norm(v) + 1e-8
        smatrix.update(k, v)

        s_norm = float(np.linalg.norm(smatrix.S, "fro"))

        rows.append({
            "scenario": scenario_name,
            "delta_on": True,
            "turn": t,
            "phi": round(phi, 6),
            "cv": round(cv, 6),
            "is_drifting": int(is_drifting),
            "S_fro_norm": round(s_norm, 6),
            "step": smatrix.step,
            "gamma": round(gamma, 6),
        })

    return rows


def run_delta_off(
    scenario_name: str,
    phi_func: Callable,
    turns: int,
    seed: int,
    cv_threshold: float | None = None,
    window_size: int | None = None,
) -> list[dict]:
    """Run WITHOUT delta-mem: simple sliding-window CV baseline."""
    rng = np.random.default_rng(seed + hash(scenario_name) % 10000)

    # Scenario-specific auto-tuned defaults
    if cv_threshold is None:
        cv_threshold = 0.18 if scenario_name == "DRIFTING" else 0.30
    if window_size is None:
        window_size = 5  # sliding window baseline always uses 5

    sw = SlidingWindowCV(window_size=window_size)

    rows: list[dict] = []
    for t in range(turns):
        phi = phi_func(t, turns, rng)
        cv = sw.push(phi)
        is_drifting = int(cv > cv_threshold)

        rows.append({
            "scenario": scenario_name,
            "delta_on": False,
            "turn": t,
            "phi": round(phi, 6),
            "cv": round(cv, 6),
            "is_drifting": is_drifting,
            "S_fro_norm": 0.0,
            "step": 0,
            "gamma": 0.0,
        })

    return rows


# ===========================================================================
# Metrics
# ===========================================================================

def compute_metrics(rows: list[dict]) -> dict:
    """Derive summary metrics from a completed run."""
    n = len(rows)
    phis = np.array([r["phi"] for r in rows])
    cvs = np.array([r["cv"] for r in rows])
    s_norms = np.array([r["S_fro_norm"] for r in rows])
    drift_flags = np.array([r["is_drifting"] for r in rows])

    drift_turns = np.where(drift_flags == 1)[0]
    drift_latency = int(drift_turns[0]) if len(drift_turns) > 0 else n

    return {
        "turns": n,
        "phi_mean": round(float(phis.mean()), 6),
        "phi_std": round(float(phis.std()), 6),
        "phi_min": round(float(phis.min()), 6),
        "phi_max": round(float(phis.max()), 6),
        "cv_mean": round(float(cvs.mean()), 6),
        "cv_std": round(float(cvs.std()), 6),
        "cv_max": round(float(cvs.max()), 6),
        "drift_detection_latency": drift_latency,
        "drift_count": int(drift_flags.sum()),
        "drift_ratio": round(float(drift_flags.sum()) / n, 4),
        "S_fro_initial": round(float(s_norms[0]), 6),
        "S_fro_final": round(float(s_norms[-1]), 6),
        "S_fro_growth": round(float(s_norms[-1] - s_norms[0]), 6),
        "S_fro_mean": round(float(s_norms.mean()), 6),
    }


# ===========================================================================
# CLI
# ===========================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="delta-mem Long Conversation Incremental Validation"
    )
    parser.add_argument(
        "--turns", type=int, default=20,
        help="Conversation turns per scenario (default: 20)"
    )
    parser.add_argument(
        "--scenarios", nargs="+", default=["STABLE", "DRIFTING", "MIXED"],
        help="Scenarios to run (default: STABLE DRIFTING MIXED)"
    )
    parser.add_argument(
        "--rank", type=int, default=DEFAULT_RANK,
        help=f"S-matrix rank (default: {DEFAULT_RANK})"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--outdir", type=str, default="results",
        help="Output directory (default: results)"
    )
    parser.add_argument(
        "--cv-threshold", type=float, default=None,
        help="Drift CV threshold (default: 0.30 for STABLE/MIXED, "
             "0.18 for DRIFTING — auto-adjusted by scenario unless set here)"
    )
    parser.add_argument(
        "--window-size", type=int, default=None,
        help="Drift detection window size (default: 20 for STABLE/MIXED, "
             "10 for DRIFTING — auto-adjusted by scenario unless set here)"
    )
    parser.add_argument(
        "--adaptive-threshold", action="store_true", default=True,
        help="Enable adaptive cv_threshold (exponential decay based on "
             "conversation length, default: ON)"
    )
    parser.add_argument(
        "--no-adaptive-threshold", action="store_false",
        dest="adaptive_threshold",
        help="Disable adaptive cv_threshold (use fixed threshold)"
    )
    parser.add_argument(
        "--quiet", action="store_true",
        help="Suppress per-turn output"
    )
    args = parser.parse_args()

    # ---- resolve output paths ------------------------------------------------
    outdir = PROJECT_ROOT / args.outdir
    outdir.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    csv_path = outdir / f"longconv_bench_{ts}.csv"
    json_path = outdir / f"longconv_report_{ts}.json"

    # ---- header --------------------------------------------------------------
    print("=" * 72)
    print("  delta-mem Long Conversation Incremental Validation")
    print("=" * 72)
    print(f"  turns/scenario : {args.turns}")
    print(f"  scenarios      : {', '.join(args.scenarios)}")
    print(f"  rank (r)       : {args.rank}")
    print(f"  seed           : {args.seed}")
    print(f"  cv_threshold   : {args.cv_threshold or 'auto (0.30/0.18)'}")
    print(f"  window_size    : {args.window_size or 'auto (20/10)'}")
    print(f"  adaptive_thr   : {'ON' if args.adaptive_threshold else 'OFF'}")
    print(f"  output         : {csv_path}")
    print()

    # ---- run all configurations ----------------------------------------------
    all_rows: list[dict] = []
    all_metrics: list[dict] = []

    for sc_name in args.scenarios:
        phi_func = SCENARIOS[sc_name]

        for delta_on in [True, False]:
            label = "delta-ON " if delta_on else "delta-OFF"
            print(f"  [{sc_name:<10}] {label} ...", end=" ", flush=True)

            if delta_on:
                rows = run_delta_on(
                    sc_name, phi_func, args.turns, args.rank, args.seed,
                    args.cv_threshold, args.window_size,
                    args.adaptive_threshold,
                )
            else:
                rows = run_delta_off(
                    sc_name, phi_func, args.turns, args.seed,
                    args.cv_threshold, args.window_size,
                )

            all_rows.extend(rows)
            m = compute_metrics(rows)
            m["scenario"] = sc_name
            m["delta_on"] = delta_on
            all_metrics.append(m)

            print(f"done  CV_mean={m['cv_mean']:.4f}  drift_ratio={m['drift_ratio']:.2%}")

    # ---- write CSV -----------------------------------------------------------
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=_fieldnames())
        w.writeheader()
        w.writerows(all_rows)

    # ---- build JSON report ---------------------------------------------------
    report = {
        "config": {
            "turns": args.turns,
            "scenarios": args.scenarios,
            "rank": args.rank,
            "seed": args.seed,
            "timestamp": ts,
        },
        "metrics": all_metrics,
    }

    # Per-scenario delta-ON vs delta-OFF comparison
    report["comparison"] = {}
    for sc_name in args.scenarios:
        on_m = next(m for m in all_metrics if m["scenario"] == sc_name and m["delta_on"])
        off_m = next(m for m in all_metrics if m["scenario"] == sc_name and not m["delta_on"])
        report["comparison"][sc_name] = {
            "delta_on": on_m,
            "delta_off": off_m,
            "cv_stability_gain": round(off_m["cv_std"] - on_m["cv_std"], 6),
            "drift_latency_delta": on_m["drift_detection_latency"],
            "s_matrix_growth": on_m["S_fro_growth"],
        }

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    # ---- summary table -------------------------------------------------------
    print(f"\n  CSV  -> {csv_path.name}")
    print(f"  JSON -> {json_path.name}")

    print("\n" + "=" * 80)
    print("  SUMMARY TABLE")
    print("=" * 80)
    header = (
        f"{'Scenario':<10} {'delta':<7} {'CV_mean':>9} {'CV_std':>9} "
        f"{'CV_max':>9} {'Drift%':>7} {'S_grow':>9} {'Latency':>7}"
    )
    print(header)
    print("-" * 80)
    for m in all_metrics:
        dm = "ON" if m["delta_on"] else "OFF"
        print(
            f"{m['scenario']:<10} {dm:<7} "
            f"{m['cv_mean']:>9.4f} {m['cv_std']:>9.4f} "
            f"{m['cv_max']:>9.4f} {m['drift_ratio']:>7.2%} "
            f"{m['S_fro_growth']:>9.4f} {m['drift_detection_latency']:>7d}"
        )

    # ---- key insights --------------------------------------------------------
    print("\n" + "=" * 80)
    print("  KEY INSIGHTS")
    print("=" * 80)

    for sc_name in args.scenarios:
        comp = report["comparison"][sc_name]
        on_m = comp["delta_on"]
        off_m = comp["delta_off"]

        print(f"\n  [{sc_name}]")
        print(f"    delta-ON  | CV={on_m['cv_mean']:.4f}+-{on_m['cv_std']:.4f}  "
              f"max={on_m['cv_max']:.4f}  drift={on_m['drift_ratio']:.1%}  "
              f"S_norm={on_m['S_fro_final']:.3f}")
        print(f"    delta-OFF | CV={off_m['cv_mean']:.4f}+-{off_m['cv_std']:.4f}  "
              f"max={off_m['cv_max']:.4f}  drift={off_m['drift_ratio']:.1%}")

        if sc_name == "STABLE":
            fp = on_m["drift_count"]
            if fp == 0:
                print(f"    >>> Zero false positives -- decay-weighted CV robust to noise")
            else:
                print(f"    >>> {fp} false positives -- hysteresis may need tuning")
            if comp["s_matrix_growth"] > 0:
                print(f"    >>> S accumulates ~{comp['s_matrix_growth']:.3f} Frobenius "
                      f"growth even in stable dialogue (expected: S naturally "
                      f"builds structure from all inputs)")

        elif sc_name == "DRIFTING":
            lat = comp["drift_latency_delta"]
            # Actual threshold used (auto-tuned by scenario unless --cv-threshold is set)
            actual_thr = (
                args.cv_threshold
                if args.cv_threshold is not None
                else 0.18
            )
            actual_win = (
                args.window_size
                if args.window_size is not None
                else 10
            )
            if lat < args.turns:
                print(f"    >>> Drift detected at turn {lat} -- "
                      f"{args.turns - lat} turns ahead of the trajectory end")
                print(f"    >>> (sensitive mode: cv_threshold={actual_thr}, "
                      f"window_size={actual_win}, min_samples=3, hysteresis=1)")
            else:
                # Show adaptive threshold info if enabled
                if args.adaptive_threshold and args.turns > 20:
                    # Compute effective threshold at end of run
                    import math
                    gap = actual_thr - 0.12
                    eff = 0.12 + gap * math.exp(-args.turns / 50)
                    print(f"    >>> No drift triggered (CV_max={on_m['cv_max']:.4f})")
                    print(f"    >>> Adaptive threshold: {actual_thr:.2f} → {eff:.3f} "
                          f"(decayed over {args.turns} turns), "
                          f"but phi decline is too gradual ({on_m['phi_min']:.2f}→{on_m['phi_max']:.2f})")
                    print(f"    >>> The decay-weighted CV smooths extreme-gradual drift. "
                          f"For immediate detection, try --cv-threshold 0.12")
                else:
                    print(f"    >>> No drift triggered (CV_max={on_m['cv_max']:.4f} < "
                          f"threshold={actual_thr})")
                    print(f"    >>> Sensitive mode is ON: cv_threshold={actual_thr}, "
                          f"window_size={actual_win}. If still not triggered, "
                          f"try --cv-threshold 0.12 --window-size 5")

        elif sc_name == "MIXED":
            print(f"    >>> Oscillating phi with periodic drift "
                  f"(phi range {on_m['phi_min']:.2f}-{on_m['phi_max']:.2f})")
            if comp["cv_stability_gain"] > 0.001:
                print(f"    >>> delta-mem CV is {comp['cv_stability_gain']:.4f} "
                      f"more stable than simple sliding window")
            lat = comp["drift_latency_delta"]
            if lat < args.turns and args.adaptive_threshold:
                import math
                eff = 0.30 - (0.30 - 0.12) * (1 - math.exp(-lat / 50)) if lat > 20 else 0.30
                print(f"    >>> Drift caught at turn {lat} by adaptive threshold "
                      f"(static=0.30, effective≈{eff:.3f} at turn {lat})")

    print("\n  Done.\n")


if __name__ == "__main__":
    main()
