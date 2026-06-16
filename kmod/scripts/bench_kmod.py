#!/usr/bin/env python3
"""
bench_kmod.py — 太极OS内核模块 vs 纯 Python 性能基准测试

对比:
  - 纯 Python (numpy) S 矩阵 Delta Rule
  - 内核模块 ioctl S 矩阵 Delta Rule (需要 sudo insmod)

测试维度:
  - 不同迭代次数: 100, 500, 1000, 5000, 10000
  - S 矩阵更新 (ingest) + 查询 (query) + 漂移检测 (push_phi)

输出: CSV 格式结果 + 控制台汇总

用法:
  # 仅测试 Python 基准（无需内核模块）
  python3 scripts/bench_kmod.py

  # 同时测试内核模块（需要先 insmod）
  sudo insmod taiji_os_kmod.ko && sudo chmod 666 /dev/taiji_os
  python3 scripts/bench_kmod.py

  # 自定义迭代次数
  python3 scripts/bench_kmod.py --iters 100 500 1000 5000
"""

import argparse
import csv
import sys
import os
import time
import struct
import numpy as np

# 添加父目录到 path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

RANK = 8


# ── 纯 Python S 矩阵实现 ──────────────────────────────────

class PurePythonSMatrix:
    """纯 Python/numpy 实现的 S 矩阵 Delta Rule"""

    def __init__(self, rank=8, lambda_=0.95, beta=0.1):
        self.rank = rank
        self.S = np.zeros((rank, rank), dtype=np.float32)
        self.lambda_ = lambda_
        self.beta = beta
        self.step = 0

    def update(self, k, v):
        """Delta Rule: S_t = λ·S_{t-1} + β·(v - S·k)·k^T"""
        k = np.asarray(k, dtype=np.float32).ravel()[:self.rank]
        v = np.asarray(v, dtype=np.float32).ravel()[:self.rank]
        Sk = self.S @ k
        error = v - Sk
        delta = self.beta * np.outer(error, k)
        self.S = self.lambda_ * self.S + delta
        self.step += 1

    def query(self, q):
        """残差查询: r = S·q"""
        q = np.asarray(q, dtype=np.float32).ravel()[:self.rank]
        return self.S @ q


class PurePythonDriftDetector:
    """纯 Python 漂移检测器（简化版，仅用于基准测试）"""

    def __init__(self, window_size=20, cv_threshold=0.30):
        self.window_size = window_size
        self.cv_threshold = cv_threshold
        self.phi_history = []
        self.current_cv = 0.0
        self.is_drifting = False

    def push(self, phi_value):
        self.phi_history.append(phi_value)
        if len(self.phi_history) > self.window_size:
            self.phi_history = self.phi_history[-self.window_size:]
        if len(self.phi_history) >= 5:
            arr = np.array(self.phi_history)
            mean = np.mean(arr)
            std = np.std(arr)
            self.current_cv = std / abs(mean) if abs(mean) > 1e-8 else 0.0
            self.is_drifting = self.current_cv > self.cv_threshold


# ── 基准测试函数 ──────────────────────────────────────────

def bench_python_s_update(n_iters, rank=8):
    """基准: 纯 Python S 矩阵更新"""
    sm = PurePythonSMatrix(rank)
    k = np.random.randn(rank).astype(np.float32)
    v = np.random.randn(rank).astype(np.float32)

    # Warmup
    for _ in range(10):
        sm.update(k, v)

    t0 = time.perf_counter()
    for _ in range(n_iters):
        sm.update(k, v)
    elapsed = time.perf_counter() - t0
    return elapsed, n_iters / elapsed


def bench_python_s_query(n_iters, rank=8):
    """基准: 纯 Python S 矩阵查询"""
    sm = PurePythonSMatrix(rank)
    k = np.random.randn(rank).astype(np.float32)
    v = np.random.randn(rank).astype(np.float32)
    for _ in range(100):  # 预填充
        sm.update(k, v)
    q = np.random.randn(rank).astype(np.float32)

    # Warmup
    for _ in range(10):
        sm.query(q)

    t0 = time.perf_counter()
    for _ in range(n_iters):
        sm.query(q)
    elapsed = time.perf_counter() - t0
    return elapsed, n_iters / elapsed


def bench_python_drift(n_iters, rank=8):
    """基准: 纯 Python 漂移检测"""
    dd = PurePythonDriftDetector()

    # Warmup
    for i in range(10):
        dd.push(0.5 + 0.1 * np.random.randn())

    t0 = time.perf_counter()
    for i in range(n_iters):
        phi_val = 0.5 + 0.1 * np.random.randn()
        dd.push(phi_val)
    elapsed = time.perf_counter() - t0
    return elapsed, n_iters / elapsed


def bench_kernel_s_update(client, n_iters, rank=8):
    """基准: 内核模块 S 矩阵更新"""
    import fcntl
    k = np.random.randn(rank).astype(np.float32)
    v = np.random.randn(rank).astype(np.float32)

    # Warmup
    for _ in range(10):
        client.ingest(k, v)

    t0 = time.perf_counter()
    for _ in range(n_iters):
        client.ingest(k, v)
    elapsed = time.perf_counter() - t0
    return elapsed, n_iters / elapsed


def bench_kernel_s_query(client, n_iters, rank=8):
    """基准: 内核模块 S 矩阵查询"""
    k = np.random.randn(rank).astype(np.float32)
    v = np.random.randn(rank).astype(np.float32)
    for _ in range(100):
        client.ingest(k, v)
    q = np.random.randn(rank).astype(np.float32)

    # Warmup
    for _ in range(10):
        client.query(q)

    t0 = time.perf_counter()
    for _ in range(n_iters):
        client.query(q)
    elapsed = time.perf_counter() - t0
    return elapsed, n_iters / elapsed


def bench_kernel_drift(client, n_iters):
    """基准: 内核模块漂移检测"""
    # Warmup
    for _ in range(10):
        client.push_phi(0.5)

    t0 = time.perf_counter()
    for i in range(n_iters):
        phi_val = 0.5 + 0.1 * np.random.randn()
        client.push_phi(phi_val)
    elapsed = time.perf_counter() - t0
    return elapsed, n_iters / elapsed


# ── 主函数 ──────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="太极OS内核模块性能基准测试")
    parser.add_argument("--iters", nargs='+', type=int,
                        default=[100, 500, 1000, 5000, 10000],
                        help="测试迭代次数列表")
    parser.add_argument("--output", type=str, default=None,
                        help="CSV 输出文件路径")
    parser.add_argument("--no-kernel", action="store_true",
                        help="跳过内核模块测试")
    args = parser.parse_args()

    # 检测内核模块
    have_kernel = not args.no_kernel and os.path.exists("/dev/taiji_os")

    results = []

    print("=" * 70)
    print("太极OS S 矩阵性能基准测试")
    print("=" * 70)
    print(f"Python: {sys.version.split()[0]}")
    print(f"NumPy:  {np.__version__}")
    print(f"内核模块: {'可用' if have_kernel else '不可用'}")
    print(f"测试迭代: {args.iters}")
    print()

    for n in args.iters:
        print(f"── 迭代 {n:>6} ──")

        # ── 纯 Python ──
        py_upd_t, py_upd_ops = bench_python_s_update(n)
        py_qry_t, py_qry_ops = bench_python_s_query(n)
        py_drft_t, py_drft_ops = bench_python_drift(n)

        row = {
            "iters": n,
            "py_update_s": f"{py_upd_t:.6f}",
            "py_update_ops": f"{py_upd_ops:.0f}",
            "py_query_s": f"{py_qry_t:.6f}",
            "py_query_ops": f"{py_qry_ops:.0f}",
            "py_drift_s": f"{py_drft_t:.6f}",
            "py_drift_ops": f"{py_drft_ops:.0f}",
        }

        print(f"  Python update: {py_upd_t:.6f}s  ({py_upd_ops:>8.0f} ops/s)")
        print(f"  Python query:  {py_qry_t:.6f}s  ({py_qry_ops:>8.0f} ops/s)")
        print(f"  Python drift:  {py_drft_t:.6f}s  ({py_drft_ops:>8.0f} ops/s)")

        # ── 内核模块 ──
        if have_kernel:
            try:
                from python.taiji_os_kmod import TaijiOSClient
            except ImportError:
                sys.path.insert(0, os.path.dirname(__file__))
                from python.taiji_os_kmod import TaijiOSClient

            client = TaijiOSClient()
            client.init()

            k_upd_t, k_upd_ops = bench_kernel_s_update(client, n)
            k_qry_t, k_qry_ops = bench_kernel_s_query(client, n)
            k_drft_t, k_drft_ops = bench_kernel_drift(client, n)

            client.close()

            row.update({
                "kmod_update_s": f"{k_upd_t:.6f}",
                "kmod_update_ops": f"{k_upd_ops:.0f}",
                "kmod_query_s": f"{k_qry_t:.6f}",
                "kmod_query_ops": f"{k_qry_ops:.0f}",
                "kmod_drift_s": f"{k_drft_t:.6f}",
                "kmod_drift_ops": f"{k_drft_ops:.0f}",
                "speedup_update": f"{py_upd_ops/k_upd_ops:.2f}",
                "speedup_query": f"{py_qry_ops/k_qry_ops:.2f}",
                "speedup_drift": f"{py_drft_ops/k_drft_ops:.2f}",
            })

            print(f"  Kernel update: {k_upd_t:.6f}s  ({k_upd_ops:>8.0f} ops/s)  "
                  f"加速: {py_upd_ops/k_upd_ops:.2f}x")
            print(f"  Kernel query:  {k_qry_t:.6f}s  ({k_qry_ops:>8.0f} ops/s)  "
                  f"加速: {py_qry_ops/k_qry_ops:.2f}x")
            print(f"  Kernel drift:  {k_drft_t:.6f}s  ({k_drft_ops:>8.0f} ops/s)  "
                  f"加速: {py_drft_ops/k_drft_ops:.2f}x")
        else:
            print("  Kernel: 跳过（模块不可用）")

        print()
        results.append(row)

    # CSV 输出
    if args.output or True:  # 默认输出到 stdout
        fieldnames = list(results[0].keys())
        writer = csv.DictWriter(sys.stdout, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

        if args.output:
            with open(args.output, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(results)
            print(f"\n结果已保存到: {args.output}")


if __name__ == "__main__":
    main()
