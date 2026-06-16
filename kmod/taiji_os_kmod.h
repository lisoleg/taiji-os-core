/* taiji_os_kmod.h — 太极OS内核模块专用头文件
 *
 * 包含：数据结构定义、内部函数声明、宏定义
 * 用户态程序不应包含本文件。
 */

#ifndef TAJI_OS_KMOD_H
#define TAJI_OS_KMOD_H

#include <linux/module.h>
#include <linux/kernel.h>
#include <linux/fs.h>
#include <linux/cdev.h>
#include <linux/uaccess.h>
#include <linux/slab.h>
#include <linux/mutex.h>
#include <linux/proc_fs.h>
#include <linux/seq_file.h>
#include <linux/string.h>
#include <linux/crypto.h>
#include <crypto/hash.h>
#include <crypto/sha.h>       /* SHA256_DIGEST_SIZE */
#include <linux/version.h>    /* KERNEL_VERSION for class_create compat */
#include <asm/fpu/api.h>   /* kernel_fpu_begin/end (x86_64) */

#include "taiji_os_ioctl.h"
/* ── 内核态数学函数（无 libm 依赖）────────────────────── */

/**
 * taiji_expf — 简单 e^x 近似（Taylor 级数，|x| < 5）
 * 在内核态使用，不依赖 libm
 */
static inline float taiji_expf(float x)
{
    if (x > 5.0f)  return 148.4132f;
    if (x < -5.0f) return 0.0f;
    /* Taylor: 1 + x + x²/2! + x³/3! + x⁴/4! */
    float x2 = x * x;
    float x3 = x2 * x;
    float x4 = x2 * x2;
    return 1.0f + x + x2/2.0f + x3/6.0f + x4/24.0f;
}

/**
 * taiji_tanhf — 双曲正切近似
 */
static inline float taiji_tanhf(float x)
{
    float e2x = taiji_expf(2.0f * x);
    return (e2x - 1.0f) / (e2x + 1.0f);
}

/**
 * taiji_sqrtf — 牛顿法开平方
 */
static inline float taiji_sqrtf(float x)
{
    float guess = x * 0.5f + 0.5f;
    int i;
    if (x < 1e-8f) return 0.0f;
    for (i = 0; i < 10; i++)
        guess = 0.5f * (guess + x / guess);
    return guess;
}

/**
 * taiji_fabsf — 浮点绝对值
 */
static inline float taiji_fabsf(float x)
{
    return (x < 0.0f) ? -x : x;
}



/* ── 模块信息 ─────────────────────────────────────────── */
#define MODULE_NAME "taiji_os"
#define TAJI_VERSION "v1.0"

/* ── S 矩阵（内核态实现）─────────────────────────────── */

struct taiji_s_matrix {
    float S[TAJI_RANK][TAJI_RANK];  /* 8×8 = 64 floats = 256B */
    float lambda;                         /* 衰减因子 λ */
    float beta;                           /* 更新强度 β */
    uint64_t step;                       /* Delta Rule 步数 */
    char proof[17];                      /* SHA-256 前16字符 */
};

/**
 * taiji_s_update — Delta Rule: S_t = λ·S_{t-1} + β·(v - S·k)·k^T
 * 必须在 kernel_fpu_begin/end 之间调用！
 */
static inline void taiji_s_update(struct taiji_s_matrix *s,
                                const float *k,
                                const float *v)
{
    float Sk[TAJI_RANK];
    float error[TAJI_RANK];
    int i, j;

    /* Sk = S·k */
    for (i = 0; i < TAJI_RANK; i++) {
        Sk[i] = 0.0f;
        for (j = 0; j < TAJI_RANK; j++)
            Sk[i] += s->S[i][j] * k[j];
    }

    /* error = v - Sk */
    for (i = 0; i < TAJI_RANK; i++)
        error[i] = v[i] - Sk[i];

    /* delta = β·error·k^T */
    /* S = λ·S + delta */
    for (i = 0; i < TAJI_RANK; i++) {
        for (j = 0; j < TAJI_RANK; j++) {
            float delta = s->beta * error[i] * k[j];
            s->S[i][j] = s->lambda * s->S[i][j] + delta;
        }
    }

    s->step++;
    /* proof 在 push_phi 时批量更新，避免每次 FPU 外调用 crypto API */
}

/**
 * taiji_s_read — 读取残差: r = S·q
 */
static inline void taiji_s_read(const struct taiji_s_matrix *s,
                               const float *q,
                               float *result)
{
    int i, j;
    for (i = 0; i < TAJI_RANK; i++) {
        result[i] = 0.0f;
        for (j = 0; j < TAJI_RANK; j++)
            result[i] += s->S[i][j] * q[j];
    }
}

/**
 * taiji_s_attention_delta — 计算注意力修正 Δ
 * Δ = (S·q) * clamp(k^T·q, 0, 1)
 */
static inline void taiji_s_attention_delta(const struct taiji_s_matrix *s,
                                          const float *q,
                                          const float *k,
                                          float *result)
{
    float residual[TAJI_RANK];
    float dot_kq = 0.0f;
    float scale;
    int i;

    taiji_s_read(s, q, residual);

    for (i = 0; i < TAJI_RANK; i++)
        dot_kq += k[i] * q[i];
    scale = (dot_kq < 0.0f) ? 0.0f : (dot_kq > 1.0f ? 1.0f : dot_kq);

    for (i = 0; i < TAJI_RANK; i++)
        result[i] = residual[i] * scale;
}

/**
 * taiji_s_flush — 刷新 S 矩阵，返回快照后软重置
 */
static inline void taiji_s_flush(struct taiji_s_matrix *s,
                                float S_snapshot[TAJI_RANK][TAJI_RANK])
{
    int i, j;
    for (i = 0; i < TAJI_RANK; i++)
        for (j = 0; j < TAJI_RANK; j++) {
            S_snapshot[i][j] = s->S[i][j];
            s->S[i][j] *= 0.1f;  /* 软重置 */
        }
    s->step = 0;
}

/* ── 漂移检测器 ─────────────────────────────────────────── */

struct taiji_drift_detector {
    /* 配置 */
    uint32_t window_size;
    float cv_threshold;
    float gamma_max;
    float gamma_min;
    float cv_mid;
    float temperature;
    float slope_alpha;
    float slope_k;
    uint32_t min_samples_before_detect;
    uint32_t hysteresis_rounds;

    /* 状态 */
    float phi_history[TAJI_PHI_HIST_LEN];
    uint32_t write_idx;
    uint32_t count;
    float current_cv;
    float _prev_cv;
    uint8_t is_drifting;
    uint32_t drift_counter;
    uint8_t auto_tune;
};

/**
 * taiji_drift_init — 初始化漂移检测器
 */
static inline void taiji_drift_init(struct taiji_drift_detector *dd,
                                   const struct taiji_config *cfg)
{
    memset(dd, 0, sizeof(*dd));
    dd->window_size = cfg->window_size ?: TAJI_CV_WIN;
    dd->cv_threshold = cfg->cv_threshold;
    dd->gamma_max    = cfg->gamma_max;
    dd->gamma_min    = cfg->gamma_min;
    dd->cv_mid       = cfg->cv_mid;
    dd->temperature   = cfg->temperature;
    dd->slope_alpha  = cfg->slope_alpha;
    dd->slope_k     = cfg->slope_k;
    dd->min_samples_before_detect = 5;
    dd->hysteresis_rounds = 2;
    dd->auto_tune = cfg->auto_tune;
}

/**
 * taiji_sigmoid — 标准化 sigmoid 函数
 */
static inline float taiji_sigmoid(float x)
{
    /* 防止 exp 溢出 */
    if (x > 50.0f)  return 1.0f;
    if (x < -50.0f) return 0.0f;
    return 1.0f / (1.0f + taiji_expf(-x));
}

/**
 * taiji_drift_compute_weighted_cv — 计算指数衰减加权 CV
 *
 * 权重: w[i] = decay^(n-1-i)，最近的 Φ 权重最大
 * CV = std(weighted) / mean(weighted)
 */
static inline float taiji_drift_compute_weighted_cv(struct taiji_drift_detector *dd)
{
    float weights[TAJI_PHI_HIST_LEN];
    float weighted_mean = 0.0f, weighted_var = 0.0f;
    float decay = 0.55f;  /* 基础衰减，由 auto_tune 覆盖 */
    float sum_w = 0.0f;
    float sum_wx = 0.0f;
    float sum_wx2 = 0.0f;
    uint32_t n = dd->count;
    uint32_t i, idx;

    if (n < 2)
        return 0.0f;

    /* 使用权重：最近的最大 */
    for (i = 0; i < n; i++) {
        int pi = (dd->write_idx - 1 - i + TAJI_PHI_HIST_LEN) % TAJI_PHI_HIST_LEN;
        /* 简化：等权重（内核态避免浮点幂运算） */
        weights[pi] = 1.0f;
    }

    for (i = 0; i < n; i++) {
        idx = (dd->write_idx - n + i + TAJI_PHI_HIST_LEN) % TAJI_PHI_HIST_LEN;
        sum_w  += weights[idx];
        sum_wx += weights[idx] * dd->phi_history[idx];
    }
    if (sum_w < 1e-8f)
        return 0.0f;
    weighted_mean = sum_wx / sum_w;

    for (i = 0; i < n; i++) {
        float diff;
        idx = (dd->write_idx - n + i + TAJI_PHI_HIST_LEN) % TAJI_PHI_HIST_LEN;
        diff = dd->phi_history[idx] - weighted_mean;
        sum_wx2 += weights[idx] * diff * diff;
    }
    weighted_var = sum_wx2 / sum_w;

    {
        float weighted_std = taiji_sqrtf(weighted_var);
        float cv = (taiji_fabsf(weighted_mean) < 1e-8f) ? 0.0f : (weighted_std / weighted_mean);
        dd->current_cv = cv;
        return cv;
    }
}

/**
 * taiji_drift_compute_gamma — 连续 sigmoid 自动调优
 *
 * γ(CV, dCV/dt) = γ_max − Δγ × σ((CV−CV_mid)/T) × slope_factor
 */
static inline float taiji_drift_compute_gamma(struct taiji_drift_detector *dd)
{
    float cv = dd->current_cv;
    float x, sig, slope_factor, gamma;

    if (!dd->auto_tune)
        return 0.55f;  /* 固定衰减 */

    x = (cv - dd->cv_mid) / dd->temperature;
    sig = taiji_sigmoid(x);

    {
        float dcv_dt = cv - dd->_prev_cv;
        float tanh_val = taiji_tanhf(dd->slope_k * dcv_dt);
        slope_factor = 1.0f - dd->slope_alpha * tanh_val;
    }

    gamma = dd->gamma_max - (dd->gamma_max - dd->gamma_min) * sig * slope_factor;
    gamma = (gamma < dd->gamma_min) ? dd->gamma_min :
             (gamma > dd->gamma_max) ? dd->gamma_max : gamma;

    dd->_prev_cv = cv;
    return gamma;
}

/**
 * taiji_drift_push — 推送 Φ 值，返回是否确认漂移
 *
 * 返回值: 0 = 无漂移, 1 = 漂移确认
 */
static inline int taiji_drift_push(struct taiji_drift_detector *dd, float phi_val)
{
    float cv;
    int is_drift;

    /* 推入历史（环形缓冲区）*/
    dd->phi_history[dd->write_idx] = phi_val;
    dd->write_idx = (dd->write_idx + 1) % TAJI_PHI_HIST_LEN;
    if (dd->count < TAJI_PHI_HIST_LEN)
        dd->count++;

    if (dd->count < dd->min_samples_before_detect)
        return 0;

    cv = taiji_drift_compute_weighted_cv(dd);
    is_drift = (cv > dd->cv_threshold) ? 1 : 0;

    /* 施密特触发器：连续 hysteresis_rounds 次才确认 */
    if (is_drift) {
        dd->drift_counter++;
        if (dd->drift_counter >= dd->hysteresis_rounds) {
            dd->is_drifting = 1;
            return 1;
        }
    } else {
        dd->drift_counter = 0;
        if (dd->is_drifting && cv < dd->cv_threshold * 0.8f) {
            dd->is_drifting = 0;
            dd->drift_counter = 0;
        }
    }

    /* 计算自适应 gamma（无论是否漂移）*/
    taiji_drift_compute_gamma(dd);

    return 0;
}

/* ── HyperParamAdapter ─────────────────────────────────────── */

struct taiji_hyper_adapter {
    float cv_history[TAJI_HYPER_HIST];
    uint32_t cv_idx;
    uint32_t cv_count;
    uint32_t adaptation_interval;
    float cv_mid_quantile;
    uint8_t needs_adapt;

    /* 输出 */
    float new_gamma_max;
    float new_gamma_min;
    float new_cv_mid;
    uint8_t adapted;
};

static inline void taiji_hyper_init(struct taiji_hyper_adapter *ha,
                                    const struct taiji_config *cfg)
{
    memset(ha, 0, sizeof(*ha));
    ha->adaptation_interval = 20;
    ha->cv_mid_quantile = 0.60f;
}

static inline void taiji_hyper_push(struct taiji_hyper_adapter *ha, float cv)
{
    ha->cv_history[ha->cv_idx] = cv;
    ha->cv_idx = (ha->cv_idx + 1) % TAJI_HYPER_HIST;
    if (ha->cv_count < TAJI_HYPER_HIST)
        ha->cv_count++;
}

/**
 * taiji_hyper_should_adapt — 每 adaptation_interval 轮触发一次
 */
static inline int taiji_hyper_should_adapt(struct taiji_hyper_adapter *ha,
                                            uint64_t step)
{
    if (ha->cv_count < ha->adaptation_interval)
        return 0;
    return (step % ha->adaptation_interval == 0) ? 1 : 0;
}

/**
 * 简化分位数计算（内核态避免完整排序）
 * 使用 O(n) 选择算法（快速选择）的简化版：采样估计
 */
static inline float taiji_approx_quantile(float *arr, uint32_t n, float q)
{
    /* 简化：返回均值附近的估计值 */
    float sum = 0.0f;
    uint32_t i;
    for (i = 0; i < n; i++)
        sum += arr[i];
    return sum / n;  /* 实际应实现完整分位数，此处为原型 */
}

static inline void taiji_hyper_adapt(struct taiji_hyper_adapter *ha,
                                     struct taiji_drift_detector *dd)
{
    float cv_mean;
    uint32_t n = ha->cv_count;

    if (n < ha->adaptation_interval)
        return;

    cv_mean = taiji_approx_quantile(ha->cv_history, n, ha->cv_mid_quantile);
    dd->cv_mid = cv_mean;
    dd->gamma_max = 0.70f + 0.25f * (1.0f - cv_mean);  /* 稳定时 gamma_max 更高 */
    if (dd->gamma_max > 0.95f) dd->gamma_max = 0.95f;
    if (dd->gamma_max < 0.70f) dd->gamma_max = 0.70f;

    /* gamma_min 三档 */
    if (cv_mean < 0.15f)
        dd->gamma_min = 0.10f;
    else if (cv_mean < 0.30f)
        dd->gamma_min = 0.20f;
    else
        dd->gamma_min = 0.35f;

    ha->adapted = 1;
}

/* ── Session 结构 ────────────────────────────────────────── */

struct taiji_session {
    struct taiji_s_matrix s_matrix;
    struct taiji_drift_detector drift;
    struct taiji_hyper_adapter hyper;

    /* 统计 */
    uint64_t total_updates;
    uint64_t total_queries;
    uint64_t drift_events;
    uint32_t flush_count;     /* 累计 S 矩阵刷新次数 */

    struct mutex lock;       /* 并发保护 */
    struct list_head list;    /* 全局链表 */
    uint32_t session_id;
};

/* ── 全局状态 ───────────────────────────────────────────── */

extern struct taiji_config global_config;
extern struct list_head taiji_sessions;
extern struct mutex taiji_sessions_lock;
extern uint32_t taiji_session_counter;

extern struct proc_dir_entry *taiji_proc_dir;

/* ── 函数声明 ─────────────────────────────────────────── */

long taiji_ioctl(struct file *file, unsigned int cmd, unsigned long arg);
int taiji_open(struct inode *inode, struct file *file);
int taiji_release(struct inode *inode, struct file *file);

int taiji_proc_init(void);
void taiji_proc_cleanup(void);

#endif /* TAJI_OS_KMOD_H */
