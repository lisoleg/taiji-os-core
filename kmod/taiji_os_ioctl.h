/* taiji_os_ioctl.h — 用户态/内核态共享 ioctl 定义
 *
 * 太极OS内核模块 — S矩阵Delta Rule + 漂移检测器
 * 版本: v1.0 (2026-06-15)
 *
 * 编译时无需依赖内核头文件（用户态程序只包含本文件）。
 * 内核态模块同时include本文件和内核头文件。
 */

#ifndef TAJI_OS_IOCTL_H
#define TAJI_OS_IOCTL_H

#ifdef __KERNEL__
#include <linux/types.h>   /* 内核态 */
#else
#include <stdint.h>        /* 用户态 */
typedef uint32_t __u32;
typedef uint64_t __u64;
typedef uint8_t  __u8;
#endif

/* ── 幻数 & ioctl 命令 ─────────────────────────────── */

#define TAJI_IOC_MAGIC  'T'   /* 幻数：ASCII 'T' = 84 */

/* 初始化 / 配置 */
#define TAJI_INIT       _IOW(TAJI_IOC_MAGIC, 1,  struct taiji_config)
#define TAJI_RESET      _IO(TAJI_IOC_MAGIC, 2)

/* S 矩阵操作 */
#define TAJI_S_UPDATE   _IOW(TAJI_IOC_MAGIC, 10, struct taiji_update_arg)
#define TAJI_S_QUERY    _IOR(TAJI_IOC_MAGIC, 11, struct taiji_query_arg)
#define TAJI_S_READ     _IOR(TAJI_IOC_MAGIC, 12, struct taiji_read_arg)
#define TAJI_S_FLUSH    _IOR(TAJI_IOC_MAGIC, 13, struct taiji_flush_arg)
#define TAJI_S_GET      _IOR(TAJI_IOC_MAGIC, 14, struct taiji_s_matrix_arg)

/* 漂移检测 */
#define TAJI_PUSH_PHI  _IOW(TAJI_IOC_MAGIC, 20, struct taiji_push_phi_arg)
#define TAJI_GET_DRIFT  _IOR(TAJI_IOC_MAGIC, 21, struct taiji_drift_info)

/* 参数调整 */
#define TAJI_SET_PARAMS _IOW(TAJI_IOC_MAGIC, 30, struct taiji_params)
#define TAJI_GET_PARAMS _IOR(TAJI_IOC_MAGIC, 31, struct taiji_params)

/* 统计信息 */
#define TAJI_GET_STATS  _IOR(TAJI_IOC_MAGIC, 40, struct taiji_stats)

/* 批量操作 */
#define TAJI_BATCH_UPDATE _IOW(TAJI_IOC_MAGIC, 50, struct taiji_batch_arg)

#define TAJI_IOC_MAXNR  50

/* ── 常量 ─────────────────────────────────────────────── */

#define TAJI_RANK         8    /* S 矩阵秩 = 8 (8×8 = 64 floats) */
#define TAJI_PHI_HIST_LEN 128  /* Φ 历史环形缓冲区长度 */
#define TAJI_CV_WIN       20   /* CV 滑动窗口（默认）*/
#define TAJI_HYPER_HIST  200  /* HyperParamAdapter CV 历史长度 */

/* ── 数据结构（ioctl 参数）───────────────────────────── */

/**
 * taiji_config — 模块初始化配置
 */
struct taiji_config {
    float lambda;          /* 衰减因子 λ (default 0.95) */
    float beta;            /* 更新强度 β (default 0.1) */
    float cv_threshold;    /* CV 漂移阈值 (default 0.30) */
    float gamma_max;       /* sigmoid 上界 (default 0.85) */
    float gamma_min;       /* sigmoid 下界 (default 0.20) */
    float cv_mid;         /* sigmoid 中点 (default 0.25) */
    float temperature;      /* sigmoid 陡度 (default 0.08) */
    float slope_alpha;     /* 斜率调整上限 (default 0.15) */
    float slope_k;         /* 斜率灵敏度 (default 20.0) */
    uint8_t auto_tune;    /* 启用连续 sigmoid (default 1) */
    uint8_t hyper_adapt;  /* 启用 HyperParamAdapter (default 0) */
    uint32_t window_size;  /* CV 滑动窗口 (default 20) */
};

/**
 * taiji_update_arg — S_UPDATE 参数
 * 用户态传入 key[8] + value[8]，内核执行 Delta Rule
 */
struct taiji_update_arg {
    float key[TAJI_RANK];    /* k 向量 (8-dim float32) */
    float value[TAJI_RANK];  /* v 向量 (8-dim float32) */
};

/**
 * taiji_query_arg — S_QUERY 参数
 * 传入 query[8]，返回 result[8] = S·q
 */
struct taiji_query_arg {
    float query[TAJI_RANK];
    float result[TAJI_RANK];
};

/**
 * taiji_read_arg — S_READ (attention_delta) 参数
 * 传入 query[8] + key[8]，返回 result[8] + scale
 */
struct taiji_read_arg {
    float query[TAJI_RANK];
    float key[TAJI_RANK];
    float result[TAJI_RANK];
    float scale;       /* 输出: clamp(k^T·q, 0, 1) */
};

/**
 * taiji_flush_arg — S_FLUSH 参数
 * 返回刷新前的 S 矩阵快照
 */
struct taiji_flush_arg {
    float S_snapshot[TAJI_RANK][TAJI_RANK];  /* 刷新前的 S */
    uint64_t step;                                   /* 当前步数 */
    uint32_t flushed_count;                         /* 累计刷新次数 */
};

/**
 * taiji_s_matrix_arg — S_GET 参数
 * 读取完整 S 矩阵状态
 */
struct taiji_s_matrix_arg {
    float S[TAJI_RANK][TAJI_RANK];
    float lambda;
    float beta;
    uint64_t step;
    char proof[17];    /* SHA-256 前 16 字符 */
};

/**
 * taiji_push_phi_arg — PUSH_PHI 参数
 */
struct taiji_push_phi_arg {
    float phi_value;          /* 当前 Φ 值 */
    uint8_t is_drifting;    /* 输出: 漂移状态 */
    float current_cv;        /* 输出: 当前 CV */
    float current_gamma;     /* 输出: 当前 γ */
};

/**
 * taiji_drift_info — GET_DRIFT 返回
 */
struct taiji_drift_info {
    uint8_t is_drifting;
    float current_cv;
    float current_gamma;
    float last_phi;
    uint32_t drift_counter;
    uint32_t total_phi_pushes;
};

/**
 * taiji_params — SET_PARAMS / GET_PARAMS 参数
 */
struct taiji_params {
    float cv_threshold;
    float gamma_max;
    float gamma_min;
    float cv_mid;
    float temperature;
    uint8_t auto_tune;
};

/**
 * taiji_stats — GET_STATS 返回
 */
struct taiji_stats {
    uint64_t total_updates;    /* S 矩阵总更新次数 */
    uint64_t total_queries;    /* 总查询次数 */
    uint64_t drift_events;      /* 漂移事件次数 */
    float current_cv;          /* 当前 CV */
    float current_gamma;       /* 当前 γ */
    uint32_t s_step;           /* S 矩阵当前步数 */
    uint32_t phi_history_len;   /* Φ 历史有效长度 */
};

/**
 * taiji_batch_arg — BATCH_UPDATE 参数
 * 一次 ioctl 调用更新多个 (k, v) 对
 */
struct taiji_batch_arg {
    uint32_t count;           /* 批量大小 (max 64) */
    float *keys;              /* 用户态指针: count×8 floats */
    float *values;            /* 用户态指针: count×8 floats */
};

/* ── 默认配置 ─────────────────────────────────────────── */
#define TAJI_CONFIG_DEFAULT {               \
    .lambda       = 0.95f,               \
    .beta         = 0.10f,               \
    .cv_threshold = 0.30f,               \
    .gamma_max    = 0.85f,               \
    .gamma_min    = 0.20f,               \
    .cv_mid       = 0.25f,               \
    .temperature   = 0.08f,               \
    .slope_alpha  = 0.15f,               \
    .slope_k      = 20.0f,                \
    .auto_tune    = 1,                    \
    .hyper_adapt  = 0,                    \
    .window_size   = TAJI_CV_WIN,         \
}

#endif /* TAJI_OS_IOCTL_H */
