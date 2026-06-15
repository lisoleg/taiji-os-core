# 太极OS Linux内核模块 — 架构设计文档

**版本**: v1.0-draft (2026-06-15)  
**目标**: 将太极OS核心δ-mem算法封装为可加载内核模块（.ko）  
**作者**: 太极OS团队

---

## 1. 设计目标与范围

### 1.1 目标
- 将δ-mem核心计算（S矩阵Delta Rule、漂移检测、Φ门控）移入内核态，消除Python调用开销
- 保留LLM API调用、语义嵌入、FASS索引在用户态（内核态无法做HTTP/IPC）
- 通过字符设备 + ioctl接口实现用户态/内核态数据交换
- 完全兼容现有Python API（`core/delta_mem.py`、`core/drift_detector.py`）

### 1.2 范围分界

| 组件 | 位置 | 理由 |
|------|------|------|
| S矩阵Delta Rule (8×8) | **内核态** | 高频小矩阵运算，内核态省IPC开销 |
| DriftDetector + HyperParamAdapter | **内核态** | 纯数值计算，无I/O |
| Φ门控（余弦相似度） | **内核态** | 向量运算，可offload |
| ψ向量（WorldModel） | **用户态** | 1536维，编码依赖外部API |
| LLM API调用 | **用户态** | 内核态无法发起HTTPS请求 |
| sentence-transformers嵌入 | **用户态** | 需要Python/torch运行时 |
| FASS episodic索引 | **用户态** | 用户态库，内核态无法链接 |
| 快照/换出 | **用户态** | 文件I/O |

---

## 2. 内核模块整体架构

```
┌─────────────────────────────────────────────────────────────┐
│                   用户态 (userspace)                      │
│                                                           │
│  Python应用层                                           │
│  ┌─────────────────────────────────────────────────┐      │
│  │ taiji_os_kmod.py (Python封装库)               │      │
│  │  - TaijiOSKmodClient 类                       │      │
│  │  - 兼容原 delta_mem.py API                    │      │
│  │  - open/ioctl/close 封装                      │      │
│  └──────────────────┬──────────────────────────┘      │
│                       │ ioctl(/dev/taiji_os)            │
│  ┌──────────────────▼──────────────────────────┐      │
│  │ DeepSeek API │ sentence-transformers │ FASS  │      │
│  └─────────────────────────────────────────────┘      │
└───────────────────────────┬─────────────────────────────┘
                            │ ioctl copy_from_user/copy_to_user
┌───────────────────────────▼─────────────────────────────┐
│                   内核态 (kernelspace)                    │
│                                                           │
│  taiji_os_kmod.ko (字符设备驱动)                        │
│  ┌─────────────────────────────────────────────────┐    │
│  │ 字符设备操作 (open/release/ioctl)             │    │
│  ├─────────────────────────────────────────────────┤    │
│  │ struct taiji_session {                          │    │
│  │   S_matrix[8][8]  (r=8, 64 floats)       │    │
│  │   phi_history[128] (滑动窗口)                 │    │
│  │   cv, gamma, is_drifting                     │    │
│  │   hyper_adapter_state                          │    │
│  │ } (每open分配一个session)                     │    │
│  ├─────────────────────────────────────────────────┤    │
│  │ S矩阵操作 (delta_rule_update, read, query)   │    │
│  │ DriftDetector (CV计算, sigmoid auto-tune)    │    │
│  │ HyperParamAdapter (每N轮自适应)              │    │
│  │ Φ计算 (余弦相似度, psi来自用户态输入)      │    │
│  ├─────────────────────────────────────────────────┤    │
│  │ /proc/taiji_os/stats (统计信息)             │    │
│  └─────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────┘
```

---

## 3. 数据结构设计

### 3.1 S矩阵 (内核态)

```c
/* taiji_os_kmod.h */

#define TAJI_RANK 8
#define TAJI_PHI_HIST_LEN 128
#define TAJI_CV_WIN 20

struct taiji_s_matrix {
    float S[TAJI_RANK][TAJI_RANK];  /* 8×8 = 64 floats = 256 bytes */
    float lambda;                       /* 衰减因子 λ (default 0.95) */
    float beta;                         /* 更新强度 β (default 0.1) */
    uint64_t step;                     /* Delta Rule步数 */
    char proof[17];                   /* SHA-256前16字符 (正确性校验) */
};

/* Delta Rule: S_t = λ·S_{t-1} + β·(v - S·k)·k^T */
void taiji_s_update(struct taiji_s_matrix *s,
                   const float *k,   /* key, 8-dim */
                   const float *v);  /* value, 8-dim */

/* 读取残差: r = S·q */
void taiji_s_read(const struct taiji_s_matrix *s,
                 const float *q,   /* query, 8-dim */
                 float *result);   /* output, 8-dim */

/* attention_delta: Δ = (S·q) * clamp(k^T·q) */
float taiji_s_attention_delta(const struct taiji_s_matrix *s,
                              const float *q,
                              const float *k,
                              float *result);
```

### 3.2 漂移检测器

```c
struct taiji_drift_detector {
    /* 配置 */
    uint32_t window_size;           /* CV滑动窗口 (default 20) */
    float cv_threshold;             /* CV阈值 (default 0.30) */
    float gamma_max;                /* (default 0.85) */
    float gamma_min;                /* (default 0.20) */
    float cv_mid;                  /* sigmoid中点 (default 0.25) */
    float temperature;               /* sigmoid陡度 (default 0.08) */
    float slope_alpha;              /* (default 0.15) */
    float slope_k;                  /* (default 20.0) */
    uint32_t min_samples;
    uint32_t hysteresis_rounds;

    /* 状态 */
    float phi_history[TAJI_PHI_HIST_LEN];  /* 环形缓冲区 */
    uint32_t write_idx;
    uint32_t count;
    float current_cv;
    float _prev_cv;
    uint8_t is_drifting;
    uint32_t drift_counter;          /* 施密特触发器 */

    /* 连续sigmoid自动调优 */
    uint8_t auto_tune;
};

/* 推送phi值，返回是否漂移 */
int taiji_drift_push(struct taiji_drift_detector *dd, float phi_val);

/* 计算当前CV (指数衰减加权) */
float taiji_drift_compute_cv(struct taiji_drift_detector *dd);

/* 连续sigmoid: 计算自适应gamma */
float taiji_drift_compute_gamma(struct taiji_drift_detector *dd);
```

### 3.3 HyperParamAdapter

```c
struct taiji_hyper_adapter {
    float cv_history[200];          /* CV历史 (滚动) */
    uint32_t cv_hist_len;
    uint32_t adaptation_interval;     /* 默认20 */
    float cv_mid_quantile;          /* 默认0.60 */
    uint8_t needs_adapt;            /* 标志位 */

    /* 输出（由adapt()填写） */
    float new_gamma_max;
    float new_gamma_min;
    float new_cv_mid;
    uint8_t adapted;
};

void taiji_hyper_push(struct taiji_hyper_adapter *ha, float cv);
int taiji_hyper_should_adapt(struct taiji_hyper_adapter *ha, uint32_t step);
void taiji_hyper_adapt(struct taiji_hyper_adapter *ha,
                       struct taiji_drift_detector *dd);
```

### 3.4 Session (每openfh分配一个)

```c
struct taiji_session {
    struct taiji_s_matrix s_matrix;
    struct taiji_drift_detector drift;
    struct taiji_hyper_adapter hyper;

    /* 统计 */
    uint64_t total_updates;
    uint64_t total_queries;
    uint64_t drift_events;

    struct list_head list;    /* 全局session链表 */
    uint32_t session_id;
};
```

---

## 4. ioctl接口设计

### 4.1 ioctl命令定义

```c
/* taiji_os_ioctl.h — 用户态/内核态共享 */

#define TAJI_IOC_MAGIC 'T'  /* 幻数 */

/* 初始化/配置 */
#define TAJI_INIT      _IOW(TAJI_IOC_MAGIC, 1, struct taiji_config)
#define TAJI_RESET     _IO(TAJI_IOC_MAGIC, 2)

/* S矩阵操作 */
#define TAJI_S_UPDATE  _IOW(TAJI_IOC_MAGIC, 10, struct taiji_update_arg)
#define TAJI_S_QUERY   _IOR(TAJI_IOC_MAGIC, 11, struct taiji_query_arg)
#define TAJI_S_READ    _IOR(TAJI_IOC_MAGIC, 12, struct taiji_read_arg)
#define TAJI_S_FLUSH    _IOR(TAJI_IOC_MAGIC, 13, struct taiji_flush_arg)

/* 漂移检测 */
#define TAJI_PUSH_PHI _IOW(TAJI_IOC_MAGIC, 20, float)  /* 推送phi值 */
#define TAJI_GET_DRIFT _IOR(TAJI_IOC_MAGIC, 21, struct taiji_drift_info)

/* 参数调整 */
#define TAJI_SET_PARAMS _IOW(TAJI_IOC_MAGIC, 30, struct taiji_params)
#define TAJI_GET_PARAMS _IOR(TAJI_IOC_MAGIC, 31, struct taiji_params)

/* 统计 */
#define TAJI_GET_STATS  _IOR(TAJI_IOC_MAGIC, 40, struct taiji_stats)

/* 幻数检查 */
#define TAJI_IOC_MAXNR 40
```

### 4.2 ioctl参数结构体

```c
struct taiji_config {
    float lambda;       /* default 0.95 */
    float beta;         /* default 0.1 */
    float cv_threshold;  /* default 0.30 */
    float gamma_max;    /* default 0.85 */
    float gamma_min;    /* default 0.20 */
    float cv_mid;       /* default 0.25 */
    uint8_t auto_tune; /* default 1 */
    uint8_t hyper_adapt; /* default 0 */
};

struct taiji_update_arg {
    float key[8];     /* k向量 (8-dim) */
    float value[8];    /* v向量 (8-dim) */
};

struct taiji_query_arg {
    float query[8];    /* q向量 */
    float result[8];    /* 返回: S·q */
};

struct taiji_read_arg {
    float query[8];
    float key[8];
    float result[8];
    float scale;        /* 返回: k^T·q的clipped值 */
};

struct taiji_drift_info {
    uint8_t is_drifting;
    float current_cv;
    float gamma;
    float last_phi;
};

struct taiji_stats {
    uint64_t total_updates;
    uint64_t total_queries;
    uint64_t drift_events;
    float current_cv;
    float current_gamma;
    uint32_t s_step;
};
```

---

## 5. 字符设备接口

### 5.1 设备文件
- **设备路径**: `/dev/taiji_os`
- **主设备号**: 动态分配 (alloc_chrdev_region)
- **次设备号**: 0 (单设备)

### 5.2 文件操作

```c
static const struct file_operations taiji_fops = {
    .owner = THIS_MODULE,
    .open = taiji_open,      /* 分配session */
    .release = taiji_release, /* 释放session */
    .unlocked_ioctl = taiji_ioctl,
    .compat_ioctl = taiji_ioctl,  /* 32-bit on 64-bit kernel */
    .read = taiji_read,      /* 可选: 二进制状态转储 */
    .write = taiji_write,    /* 可选: 批量key/value注入 */
};
```

### 5.3 open/release语义
- `open("/dev/taiji_os")`: 分配`struct taiji_session`，初始化S矩阵为零，返回fd
- `close(fd)`: 释放session，从全局链表中移除
- 支持多session并发（每个fd独立状态）

---

## 6. /proc接口

### 6.1 /proc/taiji_os/
```
/proc/taiji_os/
├── stats          # 全局统计 (总session数、总更新次数)
├── sessions/N    # 第N个session的详细状态
└── version       # 模块版本字符串
```

### 6.2 格式示例
```
$ cat /proc/taiji_os/stats
version: taiji_os_kmod v1.0
sessions: 3
total_updates: 14832
total_queries: 20391
drift_events: 127
```

---

## 7. 用户态Python封装库

### 7.1 API设计目标
完全兼容现有`core/delta_mem.py`的`DeltaMemLayer`类：

```python
# taiji_os_kmod.py

class TaijiOSKmodClient:
    """内核模块客户端，API兼容DeltaMemLayer"""
    
    def __init__(self, device="/dev/taiji_os"):
        self.fd = os.open(device, os.O_RDWR)
        # 初始化配置
        self._init_config()
    
    def ingest(self, key_vec: np.ndarray, value_vec: np.ndarray) -> None:
        """通过ioctl TAJI_S_UPDATE更新S矩阵"""
        arg = struct.pack('8f8f', *key_vec, *value_vec)
        fcntl.ioctl(self.fd, TAJI_S_UPDATE, arg)
    
    def query(self, q: np.ndarray) -> np.ndarray:
        """通过ioctl TAJI_S_QUERY查询残差"""
        arg = struct.pack('8f', *q)
        result_buf = bytearray(8 * 4)  # 8 floats
        fcntl.ioctl(self.fd, TAJI_S_QUERY, arg + result_buf)
        return np.frombuffer(result_buf, dtype=np.float32)
    
    def push_phi(self, phi_val: float) -> bool:
        """推送phi值到内核漂移检测器"""
        fcntl.ioctl(self.fd, TAJI_PUSH_PHI, struct.pack('f', phi_val))
        # 读取漂移状态
        ...
    
    def get_stats(self) -> dict:
        """获取统计信息"""
        ...
    
    def close(self):
        os.close(self.fd)
```

### 7.2 与原Python API的适配层

```python
# core/delta_mem_kernel.py — 用内核模块替换原DeltaMemLayer

try:
    import taiji_os_kmod
    _HAVE_KMOD = True
except (ImportError, OSError):
    _HAVE_KMOD = False

class DeltaMemLayerKernel(DeltaMemLayer):
    """如果内核模块可用，自动使用；否则降级到纯Python"""
    
    def __init__(self, *args, **kwargs):
        if _HAVE_KMOD:
            self._kclient = taiji_os_kmod.TaijiOSKmodClient()
        else:
            super().__init__(*args, **kwargs)
            self._use_kernel = False
    
    def ingest(self, key_vec, value_vec):
        if self._use_kernel:
            self._kclient.ingest(key_vec, value_vec)
        else:
            super().ingest(key_vec, value_vec)
    # ... 其他方法类似
```

---

## 8. 构建系统

### 8.1 Makefile (Kbuild)

```makefile
# kmod/Makefile
obj-m += taiji_os_kmod.o

KDIR ?= /lib/modules/$(shell uname -r)/build
PWD  := $(shell pwd)

all:
	$(MAKE) -C $(KDIR) M=$(PWD) modules

clean:
	$(MAKE) -C $(KDIR) M=$(PWD) clean

test: all
	sudo insmod taiji_os_kmod.ko
	sudo chmod 666 /dev/taiji_os
	python3 tests/test_kmod.py
	sudo rmmod taiji_os_kmod
```

### 8.2 安装脚本

```bash
#!/bin/bash
# install.sh
set -e
make clean
make
sudo insmod taiji_os_kmod.ko
sudo mknod /dev/taiji_os c $(cat /sys/module/taiji_os_kmod/parameters/major) 0
sudo chmod 666 /dev/taiji_os
echo "taiji_os_kmod installed. Device: /dev/taiji_os"
```

---

## 9. 测试策略

### 9.1 内核模块单元测试（用户态触发）

```python
# tests/test_kmod.py

def test_s_matrix_delta_rule():
    """验证内核态S矩阵Delta Rule与Python原版一致性"""
    client = TaijiOSKmodClient()
    k = np.random.randn(8).astype(np.float32)
    v = np.random.randn(8).astype(np.float32)
    
    # 内核态更新
    client.ingest(k, v)
    
    # 用户态原版计算
    s_py = SMatrix()
    s_py.update(k, v)
    
    # 读取内核态S矩阵
    s_ker = client.get_s_matrix()
    
    # 比对
    assert np.allclose(s_ker, s_py.S, atol=1e-5), "S matrix mismatch!"

def test_drift_detector():
    """验证漂移检测与Python原版一致性"""
    ...

def test_phi_computation():
    """验证Φ计算正确性"""
    ...
```

### 9.2 性能测试

```python
def benchmark_kernel_vs_python():
    """对比内核模块 vs 纯Python的性能"""
    N = 10000
    
    # 内核模块
    client = TaijiOSKmodClient()
    t0 = time.time()
    for _ in range(N):
        client.ingest(k, v)
    kernel_time = time.time() - t0
    
    # 纯Python
    s = SMatrix()
    t0 = time.time()
    for _ in range(N):
        s.update(k, v)
    python_time = time.time() - t0
    
    print(f"Kernel: {kernel_time:.4f}s, Python: {python_time:.4f}s")
    print(f"Speedup: {python_time/kernel_time:.2f}x")
```

---

## 10. 移植检查清单

- [ ] `drift_detector.py`: DriftDetector + HyperParamAdapter → C结构体+函数
- [ ] `delta_mem.py`: SMatrix Delta Rule → C内核函数
- [ ] `world_model.py`: phi()余弦相似度 → C函数（psi向量从用户态传入）
- [ ] `self_consistency_loop.py`: 主循环保留用户态，调用ioctl
- [ ] `config.yaml`: 配置参数 → `taiji_config`结构体 + module_param
- [ ] 浮点支持: 使用`kernel_fpu_begin()`/`kernel_fpu_end()`包裹浮点运算
- [ ] 内存分配: 使用`kmalloc(GFP_KERNEL)`分配session，不能用malloc
- [ ] 并发安全: 使用`mutex_lock`/`mutex_unlock`保护session操作
- [ ] /proc接口: 实现`proc_ops`回调
- [ ] Python封装: 实现`taiji_os_kmod.py`
- [ ] 测试: 实现`tests/test_kmod.py`

---

## 11. 风险与限制

| 风险 | 影响 | 缓解措施 |
|------|------|----------|
| 浮点在内核态支持不完整 | 部分架构可能无法编译 | 使用kernel_fpu_* API；提供纯定点算术回退 |
| ioctl copy_from/to_user开销 | 每次调用~1μs | 批量操作接口（write批量key/value） |
| 内核模块崩溃导致系统宕机 | 高 | 充分测试；提供降级到纯Python的路径 |
| 不同内核版本API变化 | 中 | 使用Kbuild版本检测；提供向后兼容层 |
| session内存泄漏 | 中 | 使用kmalloc跟踪；release强制释放 |

---

## 12. 后续扩展

1. **Netlink接口**: 异步通知用户态漂移事件（而不是轮询）
2. **mmap支持**: 将S矩阵映射到用户态地址空间，零拷贝读取
3. **cgroup集成**: 按cgroup分配δ-mem配额
4. **eBPF集成**: 用eBPF程序在内核网络栈中直接调用δ-mem
5. **USCS-MMU内核化**: 将页表管理、`page_fault`处理移植为真实内核模块

---

## 附录: 文件清单

```
kmod/
├── taiji_os_kmod.h        # 内核头文件 (数据结构、函数声明)
├── taiji_os_ioctl.h       # ioctl定义 (用户态/内核态共享)
├── taiji_os_kmod.c        # 主模块实现 (~800行)
├── Makefile               # Kbuild
├── install.sh             # 安装脚本
├── uninstall.sh           # 卸载脚本
├── python/
│   ├── taiji_os_kmod.py  # Python封装库
│   └── delta_mem_kernel.py # 与原API适配层
└── tests/
    ├── test_kmod.py      # 功能测试
    └── benchmark.py      # 性能对比
```
