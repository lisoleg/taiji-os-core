#!/usr/bin/env python3
"""scripts/benchmark_scs.py — 世界一致性 (SCS) 标准化 Benchmark

用法:
    python scripts/benchmark_scs.py                    # 离线模式（默认）
    python scripts/benchmark_scs.py --online            # 在线模式（需要 API Key）
    python scripts/benchmark_scs.py --output results/   # 指定输出目录

v4.1: 基于标准化测试集 data/test_sets/ 计算 SCS。
SCS = mean(语义连贯性指标Φ值) ，衡量世界模型的状态稳定性。
漂移测试：主题跨度越大，Φ值越低。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

# 将项目根目录加入 sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from core.session import TaijiSession  # noqa: E402
from hal.llm_router import LLMRouter   # noqa: E402

# 测试数据路径
DATA_DIR = PROJECT_ROOT / "data" / "test_sets"
CONSISTENT_FILE = DATA_DIR / "scs_consistent.json"
DRIFT_FILE = DATA_DIR / "scs_drift.json"


def load_test_cases(filepath: Path) -> list[dict]:
    """加载测试用例。"""
    if not filepath.exists():
        print(f"ERROR: 测试数据文件不存在: {filepath}")
        sys.exit(1)
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def run_scs_benchmark(online: bool = False) -> dict:
    """运行 SCS Benchmark。

    参数:
        online: 是否使用在线 API（需要 DEEPSEEK_API_KEY）

    返回:
        {"scs_consistent": float, "scs_drift": float, "phi_consistent": [...], ...}
    """
    consistent_cases = load_test_cases(CONSISTENT_FILE)
    drift_cases = load_test_cases(DRIFT_FILE)

    print(f"加载一致性输入: {len(consistent_cases)} 条")
    print(f"加载漂移输入: {len(drift_cases)} 条")
    print(f"模式: {'在线 (API)' if online else '离线 (Mock)'}")
    print("-" * 60)

    # --- 一致性测试（同一主题，Φ 应稳定） ---
    sess = TaijiSession("bench_scs_consistent", LLMRouter(), mode="text")
    phi_consistent = []

    for i, case in enumerate(consistent_cases, 1):
        start = time.time()
        result = sess.run_structured(case["input"])
        elapsed = time.time() - start
        phi_consistent.append(result.phi_value)
        status = "Consistent ✓" if result.accepted else f"Drifted ✗ (Φ={result.phi_value:.3f})"
        print(f"  [{i:2d}/{len(consistent_cases)}] {status} ({elapsed:.2f}s) | {case.get('category', '')}")

    # --- 漂移测试（主题切换，Φ 应下降） ---
    sess2 = TaijiSession("bench_scs_drift", LLMRouter(), mode="text")
    phi_drift = []

    for i, case in enumerate(drift_cases, 1):
        start = time.time()
        result = sess2.run_structured(case["input"])
        elapsed = time.time() - start
        phi_drift.append(result.phi_value)
        status = "Drifted ✓" if not result.accepted else f"Consistent ✗ (Φ={result.phi_value:.3f})"
        print(f"  [{i:2d}/{len(drift_cases)}] {status} ({elapsed:.2f}s) | {case.get('category', '')}")

    # 计算指标
    scs_consistent = sum(phi_consistent) / len(phi_consistent) if phi_consistent else 0.0
    scs_drift = sum(phi_drift) / len(phi_drift) if phi_drift else 0.0

    return {
        "benchmark": "SCS",
        "scs_consistent": round(scs_consistent, 4),
        "scs_drift": round(scs_drift, 4),
        "scs_delta": round(scs_consistent - scs_drift, 4),
        "phi_consistent_stats": {
            "mean": round(scs_consistent, 4),
            "values": [round(v, 4) for v in phi_consistent],
        },
        "phi_drift_stats": {
            "mean": round(scs_drift, 4),
            "values": [round(v, 4) for v in phi_drift],
        },
        "n_consistent": len(consistent_cases),
        "n_drift": len(drift_cases),
        "mode": "online" if online else "offline",
    }


def main():
    parser = argparse.ArgumentParser(description="SCS Benchmark")
    parser.add_argument(
        "--online", action="store_true",
        help="使用在线 DeepSeek API（需要 DEEPSEEK_API_KEY）"
    )
    parser.add_argument(
        "--output", default="benchmark_results",
        help="输出目录（默认: benchmark_results）"
    )
    args = parser.parse_args()

    # 确保测试数据存在
    if not CONSISTENT_FILE.exists() or not DRIFT_FILE.exists():
        print("ERROR: 测试数据不存在，请先运行 data/test_sets/ 准备测试集")
        sys.exit(1)

    results = run_scs_benchmark(online=args.online)

    # 保存结果
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "scs.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print("-" * 60)
    print(f"SCS (一致性) = {results['scs_consistent']:.4f}  ({results['n_consistent']} 条)")
    print(f"SCS (漂移)   = {results['scs_drift']:.4f}  ({results['n_drift']} 条)")
    print(f"ΔSCS         = {results['scs_delta']:.4f}")
    print(f"结果已保存: {output_file}")


if __name__ == "__main__":
    main()
