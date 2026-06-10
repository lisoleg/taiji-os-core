#!/usr/bin/env python3
"""scripts/benchmark_hdr.py — 幻觉拦截率 (HDR) 标准化 Benchmark

用法:
    python scripts/benchmark_hdr.py                    # 离线模式（默认）
    python scripts/benchmark_hdr.py --online            # 在线模式（需要 API Key）
    python scripts/benchmark_hdr.py --output results/   # 指定输出目录

v4.1: 基于标准化测试集 data/test_sets/ 计算 HDR。
HDR = 被拦截的矛盾输入数 / 矛盾输入总数
FPR = 被误拦截的非矛盾输入数 / 非矛盾输入总数
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
NEGATIVE_FILE = DATA_DIR / "hdr_negative.json"
POSITIVE_FILE = DATA_DIR / "hdr_positive.json"


def load_test_cases(filepath: Path) -> list[dict]:
    """加载测试用例。"""
    if not filepath.exists():
        print(f"ERROR: 测试数据文件不存在: {filepath}")
        sys.exit(1)
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def is_blocked(output: str) -> bool:
    """判断输出是否被拦截（Continuation 保存 或 GCD 阻断）。"""
    return (
        "Continuation Saved" in output
        or "BLOCKED" in output
        or "D-Core" in output
    )


def run_hdr_benchmark(online: bool = False) -> dict:
    """运行 HDR Benchmark。

    参数:
        online: 是否使用在线 API（需要 DEEPSEEK_API_KEY）

    返回:
        {"hdr": float, "fpr": float, "intercepted": int, "false_positives": int, ...}
    """
    negative_cases = load_test_cases(NEGATIVE_FILE)
    positive_cases = load_test_cases(POSITIVE_FILE)

    print(f"加载矛盾输入: {len(negative_cases)} 条")
    print(f"加载非矛盾输入: {len(positive_cases)} 条")
    print(f"模式: {'在线 (API)' if online else '离线 (Mock)'}")
    print("-" * 60)

    # 创建 Session
    sess = TaijiSession("bench_hdr", LLMRouter(), mode="text")

    # --- 矛盾输入（期望被拦截） ---
    intercepted = 0
    for i, case in enumerate(negative_cases, 1):
        start = time.time()
        out = sess.run(case["input"])
        elapsed = time.time() - start
        blocked = is_blocked(out)
        if blocked:
            intercepted += 1
        status = "BLOCKED ✓" if blocked else "PASSED ✗ (should block)"
        print(f"  [{i:2d}/{len(negative_cases)}] {status} ({elapsed:.2f}s) | {case['category']}")

    # --- 非矛盾输入（期望通过） ---
    false_positives = 0
    for i, case in enumerate(positive_cases, 1):
        start = time.time()
        out = sess.run(case["input"])
        elapsed = time.time() - start
        blocked = is_blocked(out)
        if blocked:
            false_positives += 1
        status = "PASSED ✓" if not blocked else "BLOCKED ✗ (should pass)"
        print(f"  [{i:2d}/{len(positive_cases)}] {status} ({elapsed:.2f}s) | {case['category']}")

    # 计算指标
    hdr = intercepted / len(negative_cases) if negative_cases else 0.0
    fpr = false_positives / len(positive_cases) if positive_cases else 0.0

    return {
        "benchmark": "HDR",
        "hdr": round(hdr, 4),
        "fpr": round(fpr, 4),
        "intercepted": intercepted,
        "false_positives": false_positives,
        "n_contradictory": len(negative_cases),
        "n_consistent": len(positive_cases),
        "mode": "online" if online else "offline",
    }


def main():
    parser = argparse.ArgumentParser(description="HDR Benchmark")
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
    if not NEGATIVE_FILE.exists() or not POSITIVE_FILE.exists():
        print("ERROR: 测试数据不存在，请先运行 data/test_sets/ 准备测试集")
        sys.exit(1)

    results = run_hdr_benchmark(online=args.online)

    # 保存结果
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "hdr.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print("-" * 60)
    print(f"HDR = {results['hdr']:.1%}  ({results['intercepted']}/{results['n_contradictory']})")
    print(f"FPR = {results['fpr']:.1%}  ({results['false_positives']}/{results['n_consistent']})")
    print(f"结果已保存: {output_file}")


if __name__ == "__main__":
    main()
