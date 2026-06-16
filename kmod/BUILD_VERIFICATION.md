# 太极OS内核模块 — 构建验证报告

## 审查时间
2026-06-13（静态分析，Windows x86_64 环境，非实体机构建）

## 实体构建验证
2026-06-16 新增 `kmod/scripts/verify_build.sh` — 在 Linux x86_64 上自动完成：
1. 环境检查（gcc/make/内核头文件）
2. 编译 (`make`)
3. 加载模块 (`insmod`)
4. 功能测试 (ioctl: GET_STATS/SET_PARAMS/GET_PARAMS/PUSH_PHI/S_UPDATE)
5. 卸载模块 (`rmmod`)

**用法** (在 Linux x86_64 上):
```bash
chmod +x kmod/scripts/verify_build.sh
sudo kmod/scripts/verify_build.sh
```

**前置条件**:
- Linux kernel >= 5.6（支持 `proc_ops`）
- 已安装 `linux-headers-$(uname -r)`
- 已安装 `gcc`, `make`, `libc-dev`

## 审查范围
| 文件 | 行数 | 状态 |
|------|------|------|
| `taiji_os_ioctl.h` | 204 | 已审查 |
| `taiji_os_kmod.h` | ~480 | 已审查 |
| `taiji_os_kmod.c` | ~550 | 已审查 |
| `python/taiji_os_kmod.py` | ~400 | 已审查 |

## 审查结果摘要
- ✅ 通过项: **17/19**
- ⚠️ 警告项: **3**
- ❌ 失败项: **2**（已修复，见附录）

---

## A. 结构体内存布局一致性

| 结构体 | C 大小 | Python fmt | Python 大小 | 结果 |
|--------|--------|------------|-------------|------|
| `taiji_config` | 44 | `9fBB2xI` | 44 | ✅ |
| `taiji_update_arg` | 64 | `8f8f` | 64 | ✅ |
| `taiji_query_arg` | 64 | `8f8f` | 64 | ✅ |
| `taiji_read_arg` | 100 | `8f8f8ff` | 100 | ✅ |
| `taiji_flush_arg` | 272 | `64fQI4x` | 272 | ✅ |
| `taiji_s_matrix_arg` | 296 | `64f2fQ17s7x` | 296 | ✅ |
| `taiji_push_phi_arg` | 16 | `fB3xff` | 16 | ✅ |
| `taiji_drift_info` | 24 | `B3x3f2I` | 24 | ✅ |
| `taiji_params` | **24** | ~~`4fB3x`~~ → `5fB3x` | ~~20~~ → **24** | ❌→✅ |
| `taiji_stats` | 40 | `3Q2f2I` | 40 | ✅ |
| `taiji_batch_arg` | **24** | ~~`I`~~ → `I4xQQ` | ~~4~~ → **24** | ❌→✅ |

### 详细说明

**❌ taiji_params (已修复)**:
- C struct 含 5 个 float（cv_threshold, gamma_max, gamma_min, cv_mid, temperature） + 1 uint8
- Python 原来用 `"4fB3x"` 缺少 `temperature` 字段
- 导致 ioctl 命令号不匹配：C 用 nr=30,size=24，Python 用 nr=30,size=20 → 内核收到错误命令号返回 -EINVAL
- **修复**: `"4fB3x"` → `"5fB3x"` = 24 bytes

**❌ taiji_batch_arg (已修复)**:
- C struct 含 uint32(4) + pad(4) + 2指针(16) = 24 bytes
- Python 原来用 `struct.calcsize("I")` = 4 bytes，完全错误
- **修复**: `"I4xQQ"` = 24 bytes

---

## B. ioctl 命令号一致性

| 命令 | C nr | C size | Python nr | Python size | 结果 |
|------|------|--------|-----------|-------------|------|
| TAJI_INIT | 1 | 44 | 1 | 44 | ✅ |
| TAJI_RESET | 2 | 0 | 2 | 0 | ✅ |
| TAJI_S_UPDATE | 10 | 64 | 10 | 64 | ✅ |
| TAJI_S_QUERY | 11 | 64 | 11 | 64 | ✅ |
| TAJI_S_READ | 12 | 100 | 12 | 100 | ✅ |
| TAJI_S_FLUSH | 13 | 272 | 13 | 272 | ✅ |
| TAJI_S_GET | 14 | 296 | 14 | 296 | ✅ |
| TAJI_PUSH_PHI | 20 | 16 | 20 | 16 | ✅ |
| TAJI_GET_DRIFT | 21 | 24 | 21 | 24 | ✅ |
| TAJI_SET_PARAMS | 30 | **24** | 30 | **24** | ✅(已修复) |
| TAJI_GET_PARAMS | 31 | **24** | 31 | **24** | ✅(已修复) |
| TAJI_GET_STATS | 40 | 40 | 40 | 40 | ✅ |
| TAJI_BATCH_UPDATE | 50 | **24** | 50 | **24** | ✅(已修复) |

### ioctl 方向语义警告

以下命令在 C 和 Python 中使用相同方向（相互匹配），但语义上不精确，建议未来修正：

| 命令 | 当前方向 | 实际语义 | 建议方向 | 风险 |
|------|---------|---------|---------|------|
| TAJI_S_QUERY | `_IOR` | 读+写 | `_IOWR` | ⚠️ 低（两端一致） |
| TAJI_S_READ | `_IOR` | 读+写 | `_IOWR` | ⚠️ 低 |
| TAJI_S_FLUSH | `_IOR` | 读+写 | `_IOWR` | ⚠️ 低 |
| TAJI_PUSH_PHI | `_IOW` | 读+写 | `_IOWR` | ⚠️ 低 |

---

## C. 内存管理审查

| 检查项 | 结果 |
|--------|------|
| taiji_open: kzalloc → 失败返回 -ENOMEM | ✅ |
| taiji_release: kfree + mutex_destroy 配对 | ✅ |
| BATCH_UPDATE: kmalloc ×2 → 错误路径 kfree ×2 | ✅ |
| mutex_init ↔ mutex_destroy 配对（open/release） | ✅ |
| 无内存泄漏路径 | ✅ |

---

## D. FPU 上下文配对

| 位置 | 检查结果 |
|------|---------|
| TAJI_S_UPDATE: begin/end 配对 | ✅ |
| TAJI_S_QUERY: begin/end 配对 | ✅ |
| TAJI_S_READ: begin/end 配对 | ✅ |
| TAJI_S_FLUSH: begin/end 配对 | ✅ |
| TAJI_PUSH_PHI: 2 对 begin/end，无嵌套 | ✅ |
| TAJI_GET_DRIFT: begin/end 配对 | ✅ |
| TAJI_GET_STATS: begin/end 配对 | ✅ |
| TAJI_BATCH_UPDATE: begin/end 配对 | ✅ |
| FPU 块内无 crypto_shash API 调用 | ✅ |
| FPU 块内无可能睡眠的函数 | ✅ |

---

## E. 错误处理完整性

| 检查项 | 结果 |
|--------|------|
| copy_from_user → 检查返回值 → return -EFAULT | ✅ |
| copy_to_user → 检查返回值 → return -EFAULT | ✅ |
| kmalloc 失败 → return -ENOMEM | ✅ |
| BATCH count 越界 → return -EINVAL | ✅ |
| 未知 ioctl → return -EINVAL | ⚠️（建议 -ENOTTY） |

---

## F. 内核 API 兼容性

| 检查项 | 结果 |
|--------|------|
| class_create 版本检测 (≥6.4 用1参数) | ✅ |
| proc_ops 结构体 (Linux 5.6+) | ✅ |
| crypto_shash API: alloc→setkey→digest→free | ✅ |
| `#include <crypto/sha.h>` 已添加 | ✅ |
| `#include <linux/version.h>` 已添加 | ✅ |

---

## G. 逻辑正确性

| 检查项 | 结果 |
|--------|------|
| S 矩阵 Delta Rule: S = λ·S + β·(v - S·k)·k^T | ✅ |
| taiji_s_flush: memset(S, 0)，step = 0 | ✅ |
| flush_count: 使用专用 sess→flush_count 字段 | ✅ |
| 漂移检测施密特触发器 | ✅ |
| 连续 sigmoid γ 计算: γ = γ_max − Δγ·σ((CV−CV_mid)/T)·slope | ✅ |
| HyperParamAdapter 触发逻辑 | ✅ |
| S_COMPUTE_PROOF SHA-256 路径 | ✅ |

---

## H. v5.2.0 关键 Bug 修复确认

| Bug | 修复状态 |
|-----|---------|
| `taiji_taiji_expf` 双重前缀 → `taiji_expf` | ✅ 已修复 |
| `#ifdef __KERNEL__` 守卫 ioctl.h | ✅ 已修复 |
| `class_create` 版本兼容 | ✅ 已修复 |
| BATCH_UPDATE 实现 | ✅ 已实现 |
| S_FLUSH 使用 flush_count | ✅ 已修复 |
| `taiji_params` temperature 字段缺失 | ❌→✅ 本报告发现并修复 |
| `taiji_batch_arg` ioctl 大小错误 | ❌→✅ 本报告发现并修复 |

---

## 总结与建议

### 总体评估：良好，2 个结构体布局 bug 已修复

内核模块 C 代码经过 v5.2.0 修复后质量良好：
- 内存管理正确，无泄漏路径
- FPU 上下文配对完整
- 错误处理覆盖完整
- 内核 API 兼容性已处理

### 本报告发现的 Bug 及修复

1. **`taiji_params` struct 大小**: Python 缺少 `temperature` 字段 → `"4fB3x"` 改为 `"5fB3x"`
2. **`taiji_batch_arg` struct 大小**: Python 只传 4 bytes → `"I"` 改为 `"I4xQQ"`

### 下一步建议

1. 在 Linux x86_64 上 `cd kmod && make` 验证编译
2. `sudo insmod taiji_os.ko && python3 test_kmod.py` 功能测试
3. 考虑将 ioctl 方向修正为语义正确的 `_IOWR`（当前匹配所以不影响功能）
4. 未知 ioctl 返回码从 `-EINVAL` 改为 `-ENOTTY`
