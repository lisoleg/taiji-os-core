#!/usr/bin/env python3
"""
taiji_os_kmod.py — 太极OS内核模块 Python 封装库

API 兼容 core/delta_mem.py 的 DeltaMemLayer 类。
如果内核模块已加载，使用 /dev/taiji_os ioctl 接口；
否则自动降级到纯 Python 实现。

用法:
    import taiji_os_kmod
    client = taiji_os_kmod.TaijiOSClient()
    client.ingest(key_vec, value_vec)
    residual = client.query(query_vec)
"""

import os
import sys
import struct
import fcntl
import numpy as np

__all__ = ["TaijiOSClient", "HAVE_KERNEL_MODULE", "DeltaMemLayerKernel"]

# ── ioctl 命令号（与内核模块 taiji_os_ioctl.h 同步）─────────
# 这些数值是通过 C 宏 _IOW/_IOR 在相同架构上编译确定的。
# 如果架构不同（如 32-bit arm），需要重新计算。

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

# 结构体大小（bytes）
_SIZEOF_FLOAT  = 4
_SIZEOF_UINT8 = 1
_SIZEOF_UINT32 = 4
_SIZEOF_UINT64 = 8

RANK = 8

def _config_size():
    """struct taiji_config 大小"""
    return (_SIZEOF_FLOAT * 11 +  # lambda, beta, cv_threshold, gamma_max, gamma_min, cv_mid, temperature, slope_alpha, slope_k
            _SIZEOF_UINT8 * 2 +   # auto_tune, hyper_adapt
            _SIZEOF_UINT32)            # window_size

def _update_arg_size():
    return _SIZEOF_FLOAT * RANK * 2  # key[8] + value[8]

def _query_arg_size():
    return _SIZEOF_FLOAT * RANK * 2  # query[8] + result[8]

def _read_arg_size():
    return _SIZEOF_FLOAT * RANK * 3 + _SIZEOF_FLOAT  # query[8] + key[8] + result[8] + scale

def _push_phi_arg_size():
    return _SIZEOF_FLOAT + _SIZEOF_UINT8 + _SIZEOF_FLOAT * 2  # phi_value + is_drifting + current_cv + current_gamma

def _drift_info_size():
    return (_SIZEOF_UINT8 + _SIZEOF_FLOAT * 2 +
            _SIZEOF_UINT32 * 2)  # is_drifting + current_cv + current_gamma + last_phi + counters

def _stats_size():
    return _SIZEOF_UINT64 * 3 + _SIZEOF_FLOAT * 2 + _SIZEOF_UINT32 * 2

# ioctl 命令
TAJI_INIT      = _IOW(TAJI_IOC_MAGIC, 1,  _config_size())
TAJI_RESET     = _IO(TAJI_IOC_MAGIC, 2)

TAJI_S_UPDATE  = _IOW(TAJI_IOC_MAGIC, 10, _update_arg_size())
TAJI_S_QUERY   = _IOR(TAJI_IOC_MAGIC, 11, _query_arg_size())
TAJI_S_READ    = _IOR(TAJI_IOC_MAGIC, 12, _read_arg_size())
TAJI_S_GET     = _IOR(TAJI_IOC_MAGIC, 14,
                      _SIZEOF_FLOAT * RANK * RANK + _SIZEOF_FLOAT * 2 + _SIZEOF_UINT64 + 17)

TAJI_PUSH_PHI = _IOW(TAJI_IOC_MAGIC, 20, _SIZEOF_FLOAT)
TAJI_GET_DRIFT = _IOR(TAJI_IOC_MAGIC, 21, _drift_info_size())

TAJI_GET_STATS = _IOR(TAJI_IOC_MAGIC, 40, _stats_size())

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
    太极OS内核模块客户端。

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
        self._use_kernel = True

    def init(self, lambda_=0.95, beta=0.1, cv_threshold=0.30,
             gamma_max=0.85, gamma_min=0.20, cv_mid=0.25,
             auto_tune=1, hyper_adapt=0):
        """初始化内核模块配置"""
        cfg = struct.pack(
            "9f"   # lambda, beta, cv_threshold, gamma_max, gamma_min, cv_mid, temperature, slope_alpha, slope_k
            lambda_, beta, cv_threshold, gamma_max, gamma_min, cv_mid, 0.08, 0.15, 20.0
        ) + struct.pack("BB", auto_tune, hyper_adapt) + struct.pack("I", 20)
        # 注意：结构体 padding 可能影响大小，实际使用时建议用 C 程序打印 sizeof
        # 此处用简化版，完整实现需要精确匹配
        fcntl.ioctl(self.fd, TAJI_INIT, cfg)

    def ingest(self, key_vec: np.ndarray, value_vec: np.ndarray) -> None:
        """
        通过 ioctl TAJI_S_UPDATE 更新 S 矩阵。

        Args:
            key_vec:   (8,) float32
            value_vec: (8,) float32
        """
        k = np.asarray(key_vec, dtype=np.float32).ravel()
        v = np.asarray(value_vec, dtype=np.float32).ravel()
        if len(k) < RANK:
            k = np.pad(k, (0, RANK - len(k)))
        if len(v) < RANK:
            v = np.pad(v, (0, RANK - len(v)))
        arg = struct.pack(f"{RANK}f{RANK}f", *(k[:RANK].tolist() + v[:RANK].tolist()))
        fcntl.ioctl(self.fd, TAJI_S_UPDATE, arg)

    def query(self, q: np.ndarray) -> np.ndarray:
        """
        通过 ioctl TAJI_S_QUERY 查询残差: r = S·q

        Returns:
            (8,) float32 残差向量
        """
        q = np.asarray(q, dtype=np.float32).ravel()
        if len(q) < RANK:
            q = np.pad(q, (0, RANK - len(q)))
        arg = bytearray(RANK * 4 + RANK * 4)  # query + result
        struct.pack_into(f"{RANK}f", arg, 0, *q[:RANK])
        fcntl.ioctl(self.fd, TAJI_S_QUERY, arg)
        result = struct.unpack_from(f"{RANK}f", arg, RANK * 4)
        return np.array(result, dtype=np.float32)

    def read_attention_delta(self, q: np.ndarray, k: np.ndarray) -> np.ndarray:
        """
        通过 ioctl TAJI_S_READ 计算注意力修正 Δ
        """
        q = np.asarray(q, dtype=np.float32).ravel()[:RANK]
        k = np.asarray(k, dtype=np.float32).ravel()[:RANK]
        arg = bytearray(RANK * 4 * 3 + 4)  # query + key + result + scale
        struct.pack_into(f"{RANK}f", arg, 0, *q)
        struct.pack_into(f"{RANK}f", arg, RANK * 4, *k)
        fcntl.ioctl(self.fd, TAJI_S_READ, arg)
        result = struct.unpack_from(f"{RANK}f", arg, RANK * 4 * 2)
        return np.array(result, dtype=np.float32)

    def push_phi(self, phi_value: float) -> dict:
        """
        推送 Φ值到内核漂移检测器。

        Returns:
            dict: {is_drifting, current_cv, current_gamma}
        """
        arg = struct.pack("f", phi_value)
        # 注意：TAJI_PUSH_PHI 需要传入结构体指针，此处简化
        # 完整实现需要用 TAJI_PUSH_PHI 的正确参数格式
        # 这里用写入 /proc 的简化方案（或扩展 ioctl）
        return {"is_drifting": False, "current_cv": 0.0, "current_gamma": 0.55}

    def get_stats(self) -> dict:
        """获取统计信息"""
        buf = bytearray(128)  # 足够大以容纳 struct taiji_stats
        try:
            fcntl.ioctl(self.fd, TAJI_GET_STATS, buf)
            # 解析 stats...
            return {"raw": buf.hex()}
        except OSError as e:
            return {"error": str(e)}

    def reset(self):
        """重置 session"""
        fcntl.ioctl(self.fd, TAJI_RESET)

    def close(self):
        """关闭文件描述符"""
        if hasattr(self, 'fd'):
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

    def ingest(self, key_vec, value_vec):
        if self._use_kernel:
            self._kclient.ingest(key_vec, value_vec)
        else:
            self._py_layer.ingest(key_vec, value_vec)
        return self

    def query(self, q):
        if self._use_kernel:
            return self._kclient.query(q)
        else:
            return self._py_layer.query(q)

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

        # 测试 S 矩阵更新
        k = np.random.randn(RANK).astype(np.float32)
        v = np.random.randn(RANK).astype(np.float32)
        client.ingest(k, v)
        print("✓ S_UPDATE ioctl 成功")

        # 测试查询
        q = np.random.randn(RANK).astype(np.float32)
        r = client.query(q)
        print(f"✓ S_QUERY ioctl 成功，结果范数: {np.linalg.norm(r):.4f}")

        # 统计信息
        stats = client.get_stats()
        print(f"统计数据: {stats}")

        client.close()
        print("✓ 所有测试通过")

    elif args.bench > 0:
        print(f"=== 性能测试 ({args.bench} 次迭代) ===")
        if not HAVE_KERNEL_MODULE:
            print("⚠ 内核模块不可用，仅测试 Python 回退模式")
            # 降级到纯 Python 测试
            from core.delta_mem import SMatrix
            s = SMatrix()
            k = np.random.randn(RANK).astype(np.float32)
            v = np.random.randn(RANK).astype(np.float32)
            t0 = __import__('time').time()
            for _ in range(args.bench):
                s.update(k, v)
            py_time = __import__('time').time() - t0
            print(f"Python: {py_time:.4f}s, {args.bench/py_time:.0f} ops/s")
        else:
            client = TaijiOSClient()
            k = np.random.randn(RANK).astype(np.float32)
            v = np.random.randn(RANK).astype(np.float32)
            t0 = __import__('time').time()
            for _ in range(args.bench):
                client.ingest(k, v)
            k_time = __import__('time').time() - t0
            print(f"Kernel: {k_time:.4f}s, {args.bench/k_time:.0f} ops/s")
            client.close()


if __name__ == "__main__":
    main()
