#!/usr/bin/env python3
"""
taiji_os_kmod.py — 太极OS内核模块 Python 封装库 (v1.1)

API 兼容 core/delta_mem.py 的 DeltaMemLayer 类。
如果内核模块已加载，使用 /dev/taiji_os ioctl 接口；
否则自动降级到纯 Python 实现。

用法:
    import taiji_os_kmod
    client = taiji_os_kmod.TaijiOSClient()
    client.ingest(key_vec, value_vec)
    residual = client.query(query_vec)
    phi_info = client.push_phi(0.85)
"""

import os
import sys
import struct
import fcntl
import numpy as np

__all__ = ["TaijiOSClient", "HAVE_KERNEL_MODULE", "DeltaMemLayerKernel"]

# ── ioctl 命令号（与内核模块 taiji_os_ioctl.h 同步）─────────
# x86_64 架构下的 ioctl 编码。如需交叉编译，需要重新计算。

_IOC_NRBITS   = 8
_IOC_TYPEBITS = 8
_IOC_SIZEBITS = 14
_IOC_DIRBITS  = 2

_IOC_NRSHIFT   = 0
_IOC_TYPESHIFT = 8
_IOC_SIZESHIFT = 16
_IOC_DIRSHIFT   = 30

_IOC_NONE   = 0
_IOC_WRITE  = 1
_IOC_READ   = 2

def _IOC(dir_, type_, nr, size):
    return ((dir_ << _IOC_DIRSHIFT) |
            ((type_ & 0xFF) << _IOC_TYPESHIFT) |
            ((nr & 0xFF) << _IOC_NRSHIFT) |
            ((size & 0x3FFF) << _IOC_SIZESHIFT))

def _IOW(type_, nr, size):
    return _IOC(_IOC_WRITE, ord(type_) if isinstance(type_, str) else type_, nr, size)

def _IOR(type_, nr, size):
    return _IOC(_IOC_READ, ord(type_) if isinstance(type_, str) else type_, nr, size)

def _IO(type_, nr):
    return _IOC(_IOC_NONE, ord(type_) if isinstance(type_, str) else type_, nr, 0)

# 幻数
TAJI_IOC_MAGIC = 'T'  # ASCII 84

RANK = 8

# ── C 结构体布局（x86_64，含 padding）──────────────────────
#
# taiji_config:  9×float + 2×uint8 + 2×pad + uint32 = 44 bytes
#   Format: "9fBB2xI"
#
# taiji_update_arg: 8×float + 8×float = 64 bytes
#   Format: "8f8f"
#
# taiji_query_arg: 8×float + 8×float = 64 bytes
#   Format: "8f8f"
#
# taiji_read_arg: 8×float + 8×float + 8×float + float = 100 bytes
#   Format: "8f8f8ff"
#
# taiji_flush_arg: 64×float + uint64 + uint32 + 4×pad = 272 bytes
#   Format: "64fQI4x"
#
# taiji_s_matrix_arg: 64×float + 2×float + uint64 + 17×char + 7×pad = 296 bytes
#   Format: "64f2fQ17s7x"
#
# taiji_push_phi_arg: float + uint8 + 3×pad + 2×float = 16 bytes
#   Format: "fB3xff"
#
# taiji_drift_info: uint8 + 7×pad + 3×float + 2×uint32 = 32 bytes
#   Format: "B7x3f2I"
#   Wait, let's compute:
#   uint8_t is_drifting;   // 0
#   // pad 3 bytes          // 1-3
#   float current_cv;      // 4-7
#   float current_gamma;   // 8-11
#   float last_phi;        // 12-15
#   uint32_t drift_counter; // 16-19
#   uint32_t total_phi_pushes; // 20-23
#   Total: 24 bytes
#   Format: "B3x3f2I"
#
# taiji_params: 5×float + uint8 + 3×pad = 24 bytes
#   Format: "5fB3x"
#
# taiji_stats: 3×uint64 + 2×float + 2×uint32 = 40 bytes
#   Format: "3Q2f2I"

# ── 结构体大小常量 ──────────────────────────────────────────

CONFIG_SIZE = struct.calcsize("9fBB2xI")      # 44
UPDATE_ARG_SIZE = struct.calcsize("8f8f")      # 64
QUERY_ARG_SIZE = struct.calcsize("8f8f")        # 64
READ_ARG_SIZE = struct.calcsize("8f8f8ff")     # 100
FLUSH_ARG_SIZE = struct.calcsize("64fQI4x")    # 272
SMATRIX_ARG_SIZE = struct.calcsize("64f2fQ17s7x") # 296
PUSH_PHI_ARG_SIZE = struct.calcsize("fB3xff")  # 16
DRIFT_INFO_SIZE = struct.calcsize("B3x3f2I")   # 24
PARAMS_SIZE = struct.calcsize("5fB3x")         # 24
STATS_SIZE = struct.calcsize("3Q2f2I")          # 40

# ── ioctl 命令 ──────────────────────────────────────────────

TAJI_INIT       = _IOW(TAJI_IOC_MAGIC, 1,  CONFIG_SIZE)
TAJI_RESET      = _IO(TAJI_IOC_MAGIC, 2)

TAJI_S_UPDATE   = _IOW(TAJI_IOC_MAGIC, 10, UPDATE_ARG_SIZE)
TAJI_S_QUERY    = _IOR(TAJI_IOC_MAGIC, 11, QUERY_ARG_SIZE)
TAJI_S_READ     = _IOR(TAJI_IOC_MAGIC, 12, READ_ARG_SIZE)
TAJI_S_FLUSH    = _IOR(TAJI_IOC_MAGIC, 13, FLUSH_ARG_SIZE)
TAJI_S_GET      = _IOR(TAJI_IOC_MAGIC, 14, SMATRIX_ARG_SIZE)

TAJI_PUSH_PHI   = _IOW(TAJI_IOC_MAGIC, 20, PUSH_PHI_ARG_SIZE)
TAJI_GET_DRIFT  = _IOR(TAJI_IOC_MAGIC, 21, DRIFT_INFO_SIZE)

TAJI_SET_PARAMS = _IOW(TAJI_IOC_MAGIC, 30, PARAMS_SIZE)
TAJI_GET_PARAMS = _IOR(TAJI_IOC_MAGIC, 31, PARAMS_SIZE)

TAJI_GET_STATS  = _IOR(TAJI_IOC_MAGIC, 40, STATS_SIZE)

TAJI_BATCH_UPDATE = _IOW(TAJI_IOC_MAGIC, 50, struct.calcsize("I4xQQ"))  # count + pad + 2 ptrs = 24 bytes

# ── 内核模块可用性检测 ────────────────────────────────────────

HAVE_KERNEL_MODULE = False
_DEVICE_PATH = "/dev/taiji_os"

def check_kernel_module():
    """检测内核模块是否可用"""
    global HAVE_KERNEL_MODULE
    if os.path.exists(_DEVICE_PATH):
        try:
            fd = os.open(_DEVICE_PATH, os.O_RDWR)
            os.close(fd)
            HAVE_KERNEL_MODULE = True
            return True
        except (OSError, PermissionError):
            pass
    HAVE_KERNEL_MODULE = False
    return False

check_kernel_module()


# ── 客户端类 ────────────────────────────────────────────────────

class TaijiOSClient:
    """
    太极OS内核模块客户端 (v1.1)。

    用法:
        client = TaijiOSClient()
        client.init(lambda_=0.95, beta=0.1)
        client.ingest(k, v)
        r = client.query(q)
        phi_info = client.push_phi(0.85)
        stats = client.get_stats()
        client.close()
    """

    def __init__(self, device=_DEVICE_PATH):
        if not HAVE_KERNEL_MODULE:
            raise OSError(
                f"内核模块不可用（{device} 不存在）。\n"
                f"请先运行: sudo insmod taiji_os_kmod.ko && sudo chmod 666 {device}"
            )
        self.fd = os.open(device, os.O_RDWR)

    def init(self, lambda_=0.95, beta=0.1, cv_threshold=0.30,
             gamma_max=0.85, gamma_min=0.20, cv_mid=0.25,
             temperature=0.08, slope_alpha=0.15, slope_k=20.0,
             auto_tune=1, hyper_adapt=0, window_size=20):
        """初始化内核模块配置

        struct taiji_config 布局 (x86_64, 44 bytes):
          9×float (36) + 2×uint8 (2) + 2×pad (2) + uint32 (4) = 44
          Format: "9fBB2xI"
        """
        cfg = struct.pack("9fBB2xI",
                          lambda_, beta, cv_threshold, gamma_max, gamma_min,
                          cv_mid, temperature, slope_alpha, slope_k,
                          auto_tune, hyper_adapt, window_size)
        fcntl.ioctl(self.fd, TAJI_INIT, cfg)

    def ingest(self, key_vec, value_vec):
        """通过 ioctl TAJI_S_UPDATE 更新 S 矩阵 (Delta Rule)。

        Args:
            key_vec:   (8,) float32 — k 向量
            value_vec: (8,) float32 — v 向量
        """
        k = np.asarray(key_vec, dtype=np.float32).ravel()
        v = np.asarray(value_vec, dtype=np.float32).ravel()
        if len(k) < RANK:
            k = np.pad(k, (0, RANK - len(k)))
        if len(v) < RANK:
            v = np.pad(v, (0, RANK - len(v)))
        arg = struct.pack("8f8f", *(k[:RANK].tolist() + v[:RANK].tolist()))
        fcntl.ioctl(self.fd, TAJI_S_UPDATE, arg)

    def query(self, q):
        """通过 ioctl TAJI_S_QUERY 查询残差: r = S·q

        Returns:
            (8,) float32 残差向量
        """
        q = np.asarray(q, dtype=np.float32).ravel()
        if len(q) < RANK:
            q = np.pad(q, (0, RANK - len(q)))
        arg = bytearray(QUERY_ARG_SIZE)
        struct.pack_into("8f", arg, 0, *q[:RANK])
        fcntl.ioctl(self.fd, TAJI_S_QUERY, arg)
        result = struct.unpack_from("8f", arg, RANK * 4)
        return np.array(result, dtype=np.float32)

    def read_attention_delta(self, q, k):
        """通过 ioctl TAJI_S_READ 计算注意力修正 Δ

        struct taiji_read_arg: query[8] + key[8] + result[8] + scale = 100 bytes

        Returns:
            (result[8], scale) — 注意力修正向量和缩放因子
        """
        q = np.asarray(q, dtype=np.float32).ravel()[:RANK]
        k = np.asarray(k, dtype=np.float32).ravel()[:RANK]
        arg = bytearray(READ_ARG_SIZE)
        struct.pack_into("8f", arg, 0, *q)
        struct.pack_into("8f", arg, RANK * 4, *k)
        fcntl.ioctl(self.fd, TAJI_S_READ, arg)
        result = struct.unpack_from("8f", arg, RANK * 4 * 2)
        scale = struct.unpack_from("f", arg, RANK * 4 * 3)[0]
        return np.array(result, dtype=np.float32), scale

    def push_phi(self, phi_value):
        """推送 Φ 值到内核漂移检测器。

        struct taiji_push_phi_arg (16 bytes, x86_64):
          float phi_value     @ 0-3    (input)
          uint8_t is_drifting @ 4      (output)
          3 bytes padding     @ 5-7
          float current_cv    @ 8-11   (output)
          float current_gamma @ 12-15  (output)

        Returns:
            dict: {is_drifting: bool, current_cv: float, current_gamma: float}
        """
        # 输入: phi_value, is_drifting=0, pad, current_cv=0.0, current_gamma=0.0
        arg = struct.pack("fB3xff", phi_value, 0, 0.0, 0.0)
        fcntl.ioctl(self.fd, TAJI_PUSH_PHI, arg)
        # 解析返回
        phi_val, is_drift, _pad, cv, gamma = struct.unpack("fB3xff", arg)
        return {
            "is_drifting": bool(is_drift),
            "current_cv": float(cv),
            "current_gamma": float(gamma),
        }

    def get_drift(self):
        """获取漂移检测器状态。

        struct taiji_drift_info (24 bytes, x86_64):
          uint8_t is_drifting   @ 0
          3 bytes padding       @ 1-3
          float current_cv      @ 4-7
          float current_gamma   @ 8-11
          float last_phi        @ 12-15
          uint32_t drift_counter       @ 16-19
          uint32_t total_phi_pushes    @ 20-23

        Returns:
            dict with drift info
        """
        buf = bytearray(DRIFT_INFO_SIZE)
        fcntl.ioctl(self.fd, TAJI_GET_DRIFT, buf)
        is_drift, _pad, cv, gamma, last_phi, drift_cnt, total_pushes = \
            struct.unpack("B3x3f2I", buf)
        return {
            "is_drifting": bool(is_drift),
            "current_cv": float(cv),
            "current_gamma": float(gamma),
            "last_phi": float(last_phi),
            "drift_counter": int(drift_cnt),
            "total_phi_pushes": int(total_pushes),
        }

    def set_params(self, cv_threshold=0.30, gamma_max=0.85, gamma_min=0.20,
                   cv_mid=0.25, temperature=0.08, auto_tune=1):
        """设置漂移检测器参数。

        struct taiji_params (24 bytes, x86_64):
          5×float (20) + uint8 (1) + 3×pad (3) = 24
          Format: "5fB3x"
        """
        arg = struct.pack("5fB3x", cv_threshold, gamma_max, gamma_min,
                          cv_mid, temperature, auto_tune)
        fcntl.ioctl(self.fd, TAJI_SET_PARAMS, arg)

    def get_params(self):
        """获取漂移检测器参数。"""
        buf = bytearray(PARAMS_SIZE)
        fcntl.ioctl(self.fd, TAJI_GET_PARAMS, buf)
        cv_thresh, g_max, g_min, cv_mid, temp, auto_tune = struct.unpack("5fB3x", buf)
        return {
            "cv_threshold": float(cv_thresh),
            "gamma_max": float(g_max),
            "gamma_min": float(g_min),
            "cv_mid": float(cv_mid),
            "temperature": float(temp),
            "auto_tune": bool(auto_tune),
        }

    def s_flush(self):
        """刷新 S 矩阵，返回快照。

        struct taiji_flush_arg (272 bytes, x86_64):
          64×float (256) + uint64 (8) + uint32 (4) + 4×pad (4) = 272
          Format: "64fQI4x"

        Returns:
            dict with S_snapshot, step, flushed_count
        """
        buf = bytearray(FLUSH_ARG_SIZE)
        fcntl.ioctl(self.fd, TAJI_S_FLUSH, buf)
        snapshot_flat = struct.unpack_from("64f", buf, 0)
        step = struct.unpack_from("Q", buf, 64 * 4)[0]
        flushed = struct.unpack_from("I", buf, 64 * 4 + 8)[0]
        snapshot = np.array(snapshot_flat, dtype=np.float32).reshape(RANK, RANK)
        return {"S_snapshot": snapshot, "step": int(step), "flushed_count": int(flushed)}

    def s_get(self):
        """读取完整 S 矩阵状态。

        struct taiji_s_matrix_arg (296 bytes, x86_64):
          64×float (256) + 2×float (8) + uint64 (8) + 17×char (17) + 7×pad (7) = 296

        Returns:
            dict with S, lambda, beta, step, proof
        """
        buf = bytearray(SMATRIX_ARG_SIZE)
        fcntl.ioctl(self.fd, TAJI_S_GET, buf)
        S_flat = struct.unpack_from("64f", buf, 0)
        lam, beta = struct.unpack_from("2f", buf, 64 * 4)
        step = struct.unpack_from("Q", buf, 64 * 4 + 8)[0]
        proof_raw = struct.unpack_from("17s", buf, 64 * 4 + 16)[0]
        proof = proof_raw.rstrip(b'\x00').decode('ascii', errors='replace')
        S = np.array(S_flat, dtype=np.float32).reshape(RANK, RANK)
        return {"S": S, "lambda": float(lam), "beta": float(beta),
                "step": int(step), "proof": proof}

    def get_stats(self):
        """获取统计信息。

        struct taiji_stats (40 bytes, x86_64):
          3×uint64 (24) + 2×float (8) + 2×uint32 (8) = 40
          Format: "3Q2f2I"

        Returns:
            dict with stats
        """
        buf = bytearray(STATS_SIZE)
        fcntl.ioctl(self.fd, TAJI_GET_STATS, buf)
        total_upd, total_q, drift_ev, cv, gamma, s_step, phi_len = \
            struct.unpack("3Q2f2I", buf)
        return {
            "total_updates": int(total_upd),
            "total_queries": int(total_q),
            "drift_events": int(drift_ev),
            "current_cv": float(cv),
            "current_gamma": float(gamma),
            "s_step": int(s_step),
            "phi_history_len": int(phi_len),
        }

    def reset(self):
        """重置 session"""
        fcntl.ioctl(self.fd, TAJI_RESET)

    def close(self):
        """关闭文件描述符"""
        if hasattr(self, 'fd') and self.fd >= 0:
            os.close(self.fd)
            self.fd = -1

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


# ── 与原 DeltaMemLayer API 兼容的适配层 ──────────────────────

try:
    from core.delta_mem import DeltaMemLayer, SMatrix
    _HAVE_PYTHON_FALLBACK = True
except ImportError:
    _HAVE_PYTHON_FALLBACK = False


class DeltaMemLayerKernel:
    """
    如果内核模块可用，使用内核态 S 矩阵运算；
    否则自动降级到纯 Python DeltaMemLayer。

    用法与 core/delta_mem.DeltaMemLayer 完全兼容。
    """

    def __init__(self, rank=8, use_kernel=None):
        self.rank = rank
        self._use_kernel = (use_kernel if use_kernel is not None
                           else HAVE_KERNEL_MODULE)

        if self._use_kernel:
            try:
                self._kclient = TaijiOSClient()
                self._kclient.init()
            except OSError:
                self._use_kernel = False
                if _HAVE_PYTHON_FALLBACK:
                    self._py_layer = DeltaMemLayer.create_default(rank)
                else:
                    raise
        else:
            if _HAVE_PYTHON_FALLBACK:
                self._py_layer = DeltaMemLayer.create_default(rank)
            else:
                raise RuntimeError("既无内核模块，也无 Python 回退")

    @property
    def using_kernel(self):
        """是否正在使用内核模块"""
        return self._use_kernel

    def ingest(self, key_vec, value_vec):
        """S 矩阵 Delta Rule 更新"""
        if self._use_kernel:
            self._kclient.ingest(key_vec, value_vec)
        else:
            self._py_layer.ingest(key_vec, value_vec)
        return self

    def query(self, q):
        """查询残差: r = S·q"""
        if self._use_kernel:
            return self._kclient.query(q)
        else:
            return self._py_layer.query(q)

    def query_attention_delta(self, q, k):
        """计算注意力修正 Δ"""
        if self._use_kernel:
            result, scale = self._kclient.read_attention_delta(q, k)
            return result
        else:
            return self._py_layer.query_attention_delta(q, k)

    def push_phi(self, phi_value):
        """推送 Φ 值到漂移检测器（仅内核模块模式有效）

        Returns:
            dict: {is_drifting, current_cv, current_gamma} 或 None（纯 Python 模式）
        """
        if self._use_kernel:
            return self._kclient.push_phi(phi_value)
        return None

    def get_drift_info(self):
        """获取漂移检测器状态"""
        if self._use_kernel:
            return self._kclient.get_drift()
        return None

    def set_params(self, **kwargs):
        """设置漂移检测器参数"""
        if self._use_kernel:
            self._kclient.set_params(**kwargs)

    def get_params(self):
        """获取漂移检测器参数"""
        if self._use_kernel:
            return self._kclient.get_params()
        return None

    def s_flush(self):
        """刷新 S 矩阵"""
        if self._use_kernel:
            return self._kclient.s_flush()
        return None

    def s_get(self):
        """读取完整 S 矩阵"""
        if self._use_kernel:
            return self._kclient.s_get()
        return None

    def get_stats(self):
        """获取统计信息"""
        if self._use_kernel:
            return self._kclient.get_stats()
        return None

    def to_dict(self):
        if not self._use_kernel and _HAVE_PYTHON_FALLBACK:
            return self._py_layer.to_dict()
        raise NotImplementedError("内核模块暂不支持序列化")

    @classmethod
    def from_dict(cls, data):
        if _HAVE_PYTHON_FALLBACK:
            return DeltaMemLayer.from_dict(data)
        raise NotImplementedError()

    def close(self):
        if self._use_kernel:
            self._kclient.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


# ── CLI 测试 ────────────────────────────────────────────────────

def main():
    import argparse
    import time
    parser = argparse.ArgumentParser(description="太极OS内核模块测试")
    parser.add_argument("--test", action="store_true", help="运行基础功能测试")
    parser.add_argument("--bench", type=int, default=1000, help="性能测试迭代次数")
    args = parser.parse_args()

    if args.test:
        print("=== 太极OS内核模块功能测试 ===")
        print(f"内核模块可用: {HAVE_KERNEL_MODULE}")

        if not HAVE_KERNEL_MODULE:
            print("⚠ 内核模块不可用，请先安装:")
            print("  cd kmod && make install")
            return

        client = TaijiOSClient()
        print(f"✓ 设备打开成功 (fd={client.fd})")

        # 初始化
        client.init()
        print("✓ TAJI_INIT ioctl 成功")

        # 测试 S 矩阵更新
        k = np.random.randn(RANK).astype(np.float32)
        v = np.random.randn(RANK).astype(np.float32)
        client.ingest(k, v)
        print("✓ S_UPDATE ioctl 成功")

        # 测试查询
        q = np.random.randn(RANK).astype(np.float32)
        r = client.query(q)
        print(f"✓ S_QUERY ioctl 成功，结果范数: {np.linalg.norm(r):.4f}")

        # 测试 push_phi
        phi_info = client.push_phi(0.85)
        print(f"✓ PUSH_PHI ioctl 成功: {phi_info}")

        # 统计信息
        stats = client.get_stats()
        print(f"✓ GET_STATS: {stats}")

        client.close()
        print("✓ 所有测试通过")

    elif args.bench > 0:
        print(f"=== 性能测试 ({args.bench} 次迭代) ===")
        if not HAVE_KERNEL_MODULE:
            print("⚠ 内核模块不可用，仅测试 Python 回退模式")
            from core.delta_mem import SMatrix
            s = SMatrix()
            k = np.random.randn(RANK).astype(np.float32)
            v = np.random.randn(RANK).astype(np.float32)
            t0 = time.perf_counter()
            for _ in range(args.bench):
                s.update(k, v)
            py_time = time.perf_counter() - t0
            print(f"Python: {py_time:.4f}s, {args.bench/py_time:.0f} ops/s")
        else:
            client = TaijiOSClient()
            client.init()
            k = np.random.randn(RANK).astype(np.float32)
            v = np.random.randn(RANK).astype(np.float32)
            t0 = time.perf_counter()
            for _ in range(args.bench):
                client.ingest(k, v)
            k_time = time.perf_counter() - t0
            print(f"Kernel: {k_time:.4f}s, {args.bench/k_time:.0f} ops/s")
            client.close()


if __name__ == "__main__":
    main()
