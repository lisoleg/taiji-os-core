/* taiji_os_kmod.c — 太极OS内核模块主文件
 *
 * 实现：S矩阵Delta Rule + 漂移检测器 + 字符设备 + /proc接口
 * 版本: v1.0 (2026-06-15)
 *
 * 编译:
 *   cd kmod && make
 *
 * 使用:
 *   sudo insmod taiji_os_kmod.ko
 *   sudo chmod 666 /dev/taiji_os
 *   python3 python/taiji_os_kmod.py
 *   sudo rmmod taiji_os_kmod
 */

#define KBUILD_MODNAME "taiji_os"

#include "taiji_os_kmod.h"

MODULE_LICENSE("GPL");
MODULE_AUTHOR("Taiji OS Team (Zhang Feng, Li Zonghai)");
MODULE_DESCRIPTION("Taiji OS Kernel Module: S-Matrix Delta Rule + Drift Detector");
MODULE_VERSION(TAJI_VERSION);

/* ── 全局状态 ─────────────────────────────────────── */

static struct taiji_config global_config = TAJI_CONFIG_DEFAULT;
static struct list_head taiji_sessions;
static DEFINE_MUTEX(taiji_sessions_lock);
static uint32_t taiji_session_counter = 0;

static int taiji_major = 0;  /* 动态分配主设备号 */
static struct class *taiji_class = NULL;
static struct device *taiji_device = NULL;
static struct cdev taiji_cdev;

struct proc_dir_entry *taiji_proc_dir = NULL;

/* ── 辅助函数 ─────────────────────────────────────── */

/**
 * taiji_s_compute_proof — 计算 S 矩阵完整性哈希
 *
 * 必须在 kernel_fpu_begin/end 外调用（crypto API 可能睡眠）
 */
static void taiji_s_compute_proof(struct taiji_s_matrix *s, char *proof)
{
    struct crypto_shash *tfm;
    struct shash_desc *desc;
    uint8_t hash[SHA256_DIGEST_SIZE];
    int i, j, err;

    tfm = crypto_alloc_shash("sha256", 0, 0);
    if (IS_ERR(tfm)) {
        pr_warn(MODULE_NAME ": sha256 alloc failed\n");
        strcpy(proof, "0000000000000000");
        return;
    }

    desc = kmalloc(sizeof(*desc) + crypto_shash_descsize(tfm), GFP_KERNEL);
    if (!desc) {
        crypto_free_shash(tfm);
        strcpy(proof, "0000000000000000");
        return;
    }
    desc->tfm = tfm;

    crypto_shash_init(desc);
    for (i = 0; i < TAJI_RANK; i++)
        for (j = 0; j < TAJI_RANK; j++) {
            uint32_t v;
            memcpy(&v, &s->S[i][j], 4);
            crypto_shash_update(desc, (uint8_t *)&v, 4);
        }
    {
        uint64_t step_le = cpu_to_be64(s->step);
        crypto_shash_update(desc, (uint8_t *)&step_le, 8);
    }
    crypto_shash_final(desc, hash);

    for (i = 0; i < 16; i++)
        sprintf(proof + 2 * i, "%02x", hash[i]);
    proof[16] = '\0';

    kfree(desc);
    crypto_free_shash(tfm);
}

/**
 * taiji_ioctl_dispatcher — ioctl 命令分发器
 */
static long taiji_ioctl_impl(struct file *file, unsigned int cmd, void __user *arg)
{
    struct taiji_session *sess = file->private_data;
    int ret = 0;

    if (!sess)
        return -EINVAL;

    mutex_lock(&sess->lock);

    switch (cmd) {

    /* ── 初始化 / 配置 ──────────────────────────── */
    case TAJI_INIT: {
        struct taiji_config cfg;
        if (copy_from_user(&cfg, arg, sizeof(cfg)))
            { ret = -EFAULT; goto out; }
        taiji_drift_init(&sess->drift, &cfg);
        taiji_hyper_init(&sess->hyper, &cfg);
        memcpy(&global_config, &cfg, sizeof(cfg));
        pr_info(MODULE_NAME ": session %u initialized\n", sess->session_id);
        break;
    }
    case TAJI_RESET: {
        memset(&sess->s_matrix.S[0][0], 0, sizeof(float) * TAJI_RANK * TAJI_RANK);
        sess->s_matrix.step = 0;
        sess->drift.count = 0;
        sess->drift.write_idx = 0;
        sess->drift.is_drifting = 0;
        sess->drift.drift_counter = 0;
        sess->total_updates = 0;
        sess->total_queries = 0;
        sess->drift_events = 0;
        pr_info(MODULE_NAME ": session %u reset\n", sess->session_id);
        break;
    }

    /* ── S 矩阵操作 ────────────────────────────────── */
    case TAJI_S_UPDATE: {
        struct taiji_update_arg uarg;
        if (copy_from_user(&uarg, arg, sizeof(uarg)))
            { ret = -EFAULT; goto out; }
        kernel_fpu_begin();
        taiji_s_update(&sess->s_matrix, uarg.key, uarg.value);
        kernel_fpu_end();
        sess->total_updates++;
        break;
    }
    case TAJI_S_QUERY: {
        struct taiji_query_arg qarg;
        if (copy_from_user(&qarg, arg, sizeof(qarg)))
            { ret = -EFAULT; goto out; }
        kernel_fpu_begin();
        taiji_s_read(&sess->s_matrix, qarg.query, qarg.result);
        kernel_fpu_end();
        if (copy_to_user(((struct taiji_query_arg __user *)arg)->result,
                         qarg.result, sizeof(qarg.result)))
            { ret = -EFAULT; goto out; }
        sess->total_queries++;
        break;
    }
    case TAJI_S_READ: {
        struct taiji_read_arg rarg;
        if (copy_from_user(&rarg, arg, sizeof(rarg)))
            { ret = -EFAULT; goto out; }
        kernel_fpu_begin();
        taiji_s_attention_delta(&sess->s_matrix, rarg.query, rarg.key, rarg.result);
        kernel_fpu_end();
        /* 计算 scale = clamp(k^T·q) */
        {
            float dot = 0.0f;
            int i;
            for (i = 0; i < TAJI_RANK; i++)
                dot += rarg.key[i] * rarg.query[i];
            rarg.scale = (dot < 0.0f) ? 0.0f : (dot > 1.0f ? 1.0f : dot);
        }
        if (copy_to_user(arg, &rarg, sizeof(rarg)))
            { ret = -EFAULT; goto out; }
        sess->total_queries++;
        break;
    }
    case TAJI_S_FLUSH: {
        struct taiji_flush_arg farg;
        memset(&farg, 0, sizeof(farg));
        kernel_fpu_begin();
        taiji_s_flush(&sess->s_matrix, farg.S_snapshot);
        kernel_fpu_end();
        farg.step = sess->s_matrix.step;         /* flush 后 step 已归 0 */
        farg.flushed_count = ++sess->flush_count; /* 使用专用 flush_count 字段 */
        if (copy_to_user(arg, &farg, sizeof(farg)))
            { ret = -EFAULT; goto out; }
        break;
    }
    case TAJI_S_GET: {
        struct taiji_s_matrix_arg sarg;
        kernel_fpu_begin();
        memcpy(sarg.S, sess->s_matrix.S, sizeof(float) * TAJI_RANK * TAJI_RANK);
        kernel_fpu_end();
        sarg.lambda = sess->s_matrix.lambda;
        sarg.beta   = sess->s_matrix.beta;
        sarg.step   = sess->s_matrix.step;
        taiji_s_compute_proof(&sess->s_matrix, sarg.proof);
        if (copy_to_user(arg, &sarg, sizeof(sarg)))
            { ret = -EFAULT; goto out; }
        break;
    }

    /* ── 漂移检测 ──────────────────────────────────── */
    case TAJI_PUSH_PHI: {
        struct taiji_push_phi_arg parg;
        int was_drifting;
        if (copy_from_user(&parg, arg, sizeof(parg)))
            { ret = -EFAULT; goto out; }

        kernel_fpu_begin();
        was_drifting = sess->drift.is_drifting;
        taiji_drift_push(&sess->drift, parg.phi_value);
        kernel_fpu_end();

        parg.is_drifting  = sess->drift.is_drifting;
        parg.current_cv    = sess->drift.current_cv;
        kernel_fpu_begin();
        parg.current_gamma = taiji_drift_compute_gamma(&sess->drift);
        kernel_fpu_end();

        if (sess->drift.is_drifting && !was_drifting)
            sess->drift_events++;

        /* HyperParamAdapter */
        if (global_config.hyper_adapt) {
            taiji_hyper_push(&sess->hyper, sess->drift.current_cv);
            if (taiji_hyper_should_adapt(&sess->hyper, sess->total_updates))
                taiji_hyper_adapt(&sess->hyper, &sess->drift);
        }

        if (copy_to_user(arg, &parg, sizeof(parg)))
            { ret = -EFAULT; goto out; }
        break;
    }
    case TAJI_GET_DRIFT: {
        struct taiji_drift_info dinfo;
        dinfo.is_drifting = sess->drift.is_drifting;
        dinfo.current_cv = sess->drift.current_cv;
        kernel_fpu_begin();
        dinfo.current_gamma = taiji_drift_compute_gamma(&sess->drift);
        kernel_fpu_end();
        dinfo.last_phi = (sess->drift.count > 0) ?
            sess->drift.phi_history[(sess->drift.write_idx - 1 + TAJI_PHI_HIST_LEN) % TAJI_PHI_HIST_LEN] : 0.0f;
        dinfo.drift_counter = sess->drift.drift_counter;
        dinfo.total_phi_pushes = sess->drift.count;
        if (copy_to_user(arg, &dinfo, sizeof(dinfo)))
            { ret = -EFAULT; goto out; }
        break;
    }

    /* ── 参数调整 ──────────────────────────────────── */
    case TAJI_SET_PARAMS: {
        struct taiji_params params;
        if (copy_from_user(&params, arg, sizeof(params)))
            { ret = -EFAULT; goto out; }
        sess->drift.cv_threshold = params.cv_threshold;
        sess->drift.gamma_max    = params.gamma_max;
        sess->drift.gamma_min    = params.gamma_min;
        sess->drift.cv_mid       = params.cv_mid;
        sess->drift.temperature   = params.temperature;
        sess->drift.auto_tune    = params.auto_tune;
        break;
    }
    case TAJI_GET_PARAMS: {
        struct taiji_params params;
        params.cv_threshold = sess->drift.cv_threshold;
        params.gamma_max    = sess->drift.gamma_max;
        params.gamma_min    = sess->drift.gamma_min;
        params.cv_mid       = sess->drift.cv_mid;
        params.temperature   = sess->drift.temperature;
        params.auto_tune    = sess->drift.auto_tune;
        if (copy_to_user(arg, &params, sizeof(params)))
            { ret = -EFAULT; goto out; }
        break;
    }

    /* ── 统计信息 ──────────────────────────────────── */
    case TAJI_GET_STATS: {
        struct taiji_stats stats;
        stats.total_updates   = sess->total_updates;
        stats.total_queries   = sess->total_queries;
        stats.drift_events     = sess->drift_events;
        stats.current_cv      = sess->drift.current_cv;
        kernel_fpu_begin();
        stats.current_gamma    = taiji_drift_compute_gamma(&sess->drift);
        kernel_fpu_end();
        stats.s_step          = sess->s_matrix.step;
        stats.phi_history_len = sess->drift.count;
        if (copy_to_user(arg, &stats, sizeof(stats)))
            { ret = -EFAULT; goto out; }
        break;
    }

    /* ── 批量操作 ──────────────────────────────────── */
    case TAJI_BATCH_UPDATE: {
        struct taiji_batch_arg barg;
        float *kbuf = NULL, *vbuf = NULL;
        uint32_t i;

        if (copy_from_user(&barg, arg, sizeof(barg)))
            { ret = -EFAULT; goto out; }
        if (barg.count == 0 || barg.count > 64)
            { ret = -EINVAL; goto out; }

        kbuf = kmalloc(sizeof(float) * TAJI_RANK * barg.count, GFP_KERNEL);
        vbuf = kmalloc(sizeof(float) * TAJI_RANK * barg.count, GFP_KERNEL);
        if (!kbuf || !vbuf) {
            ret = -ENOMEM;
            goto batch_cleanup;
        }

        if (copy_from_user(kbuf, barg.keys,
                           sizeof(float) * TAJI_RANK * barg.count))
            { ret = -EFAULT; goto batch_cleanup; }
        if (copy_from_user(vbuf, barg.values,
                           sizeof(float) * TAJI_RANK * barg.count))
            { ret = -EFAULT; goto batch_cleanup; }

        kernel_fpu_begin();
        for (i = 0; i < barg.count; i++) {
            taiji_s_update(&sess->s_matrix,
                           &kbuf[i * TAJI_RANK],
                           &vbuf[i * TAJI_RANK]);
        }
        kernel_fpu_end();
        sess->total_updates += barg.count;

batch_cleanup:
        if (kbuf) kfree(kbuf);
        if (vbuf) kfree(vbuf);
        break;
    }

    default:
        ret = -ENOTTY;  /* 不支持的 ioctl 命令 */
        break;
    }

out:
    mutex_unlock(&sess->lock);
    return ret;
}

#ifdef CONFIG_COMPAT
static long taiji_compat_ioctl(struct file *file, unsigned int cmd, unsigned long arg)
{
    return taiji_ioctl_impl(file, cmd, compat_ptr(arg));
}
#endif

static long taiji_unlocked_ioctl(struct file *file, unsigned int cmd, unsigned long arg)
{
    return taiji_ioctl_impl(file, cmd, (void __user *)arg);
}

/* ── 文件操作 ─────────────────────────────────────── */

static int taiji_open(struct inode *inode, struct file *file)
{
    struct taiji_session *sess;

    sess = kmalloc(sizeof(*sess), GFP_KERNEL);
    if (!sess)
        return -ENOMEM;

    memset(sess, 0, sizeof(*sess));
    sess->s_matrix.lambda = global_config.lambda;
    sess->s_matrix.beta   = global_config.beta;
    taiji_drift_init(&sess->drift, &global_config);
    taiji_hyper_init(&sess->hyper, &global_config);
    mutex_init(&sess->lock);

    mutex_lock(&taiji_sessions_lock);
    sess->session_id = ++taiji_session_counter;
    list_add_tail(&sess->list, &taiji_sessions);
    mutex_unlock(&taiji_sessions_lock);

    file->private_data = sess;
    pr_info(MODULE_NAME ": session %u opened (fd=%p)\n",
            sess->session_id, file);
    return 0;
}

static int taiji_release(struct inode *inode, struct file *file)
{
    struct taiji_session *sess = file->private_data;
    if (sess) {
        pr_info(MODULE_NAME ": session %u closed (updates=%llu)\n",
                sess->session_id, sess->total_updates);
        mutex_lock(&taiji_sessions_lock);
        list_del(&sess->list);
        mutex_unlock(&taiji_sessions_lock);
        mutex_destroy(&sess->lock);
        kfree(sess);
        file->private_data = NULL;
    }
    return 0;
}

static const struct file_operations taiji_fops = {
    .owner = THIS_MODULE,
    .open = taiji_open,
    .release = taiji_release,
    .unlocked_ioctl = taiji_unlocked_ioctl,
#ifdef CONFIG_COMPAT
    .compat_ioctl = taiji_compat_ioctl,
#endif
    .llseek = no_llseek,
};

/* ── /proc 接口 ───────────────────────────────────── */

static int taiji_proc_stats_show(struct seq_file *m, void *v)
{
    struct taiji_session *sess;
    uint32_t n_sessions = 0;
    uint64_t total_updates = 0;

    mutex_lock(&taiji_sessions_lock);
    list_for_each_entry(sess, &taiji_sessions, list) {
        n_sessions++;
        total_updates += sess->total_updates;
    }
    mutex_unlock(&taiji_sessions_lock);

    seq_printf(m, "version: " MODULE_NAME " " TAJI_VERSION "\n");
    seq_printf(m, "sessions: %u\n", n_sessions);
    seq_printf(m, "total_updates: %llu\n", total_updates);
    seq_printf(m, "config: lambda=%.3f beta=%.3f cv_thresh=%.3f\n",
               global_config.lambda, global_config.beta, global_config.cv_threshold);
    return 0;
}

static int taiji_proc_stats_open(struct inode *inode, struct file *file)
{
    return single_open(file, taiji_proc_stats_show, NULL);
}

static const struct proc_ops taiji_proc_stats_ops = {
    .proc_open = taiji_proc_stats_open,
    .proc_read = seq_read,
    .proc_lseek = seq_lseek,
    .proc_release = single_release,
};

static int taiji_proc_sessions_show(struct seq_file *m, void *v)
{
    struct taiji_session *sess;
    mutex_lock(&taiji_sessions_lock);
    list_for_each_entry(sess, &taiji_sessions, list) {
        seq_printf(m, "session_id=%u updates=%llu queries=%llu cv=%.4f drifting=%u\n",
                   sess->session_id,
                   sess->total_updates,
                   sess->total_queries,
                   sess->drift.current_cv,
                   sess->drift.is_drifting);
    }
    mutex_unlock(&taiji_sessions_lock);
    return 0;
}

static int taiji_proc_sessions_open(struct inode *inode, struct file *file)
{
    return single_open(file, taiji_proc_sessions_show, NULL);
}

static const struct proc_ops taiji_proc_sessions_ops = {
    .proc_open = taiji_proc_sessions_open,
    .proc_read = seq_read,
    .proc_lseek = seq_lseek,
    .proc_release = single_release,
};

static int taiji_proc_init(void)
{
    taiji_proc_dir = proc_mkdir(MODULE_NAME, NULL);
    if (!taiji_proc_dir)
        return -ENOMEM;

    if (!proc_create("stats", 0444, taiji_proc_dir, &taiji_proc_stats_ops))
        goto fail;

    if (!proc_create("sessions", 0444, taiji_proc_dir, &taiji_proc_sessions_ops))
        goto fail;

    return 0;
fail:
    remove_proc_subtree(MODULE_NAME, NULL);
    taiji_proc_dir = NULL;
    return -ENOMEM;
}

static void taiji_proc_cleanup(void)
{
    remove_proc_subtree(MODULE_NAME, NULL);
    taiji_proc_dir = NULL;
}

/* ── 模块初始化 / 退出 ────────────────────────────── */

static int __init taiji_init(void)
{
    dev_t dev;
    int ret;

    pr_info(MODULE_NAME ": loading version " TAJI_VERSION "\n");

    /* 初始化全局链表 */
    INIT_LIST_HEAD(&taiji_sessions);
    mutex_init(&taiji_sessions_lock);

    /* 动态分配字符设备号 */
    ret = alloc_chrdev_region(&dev, 0, 1, MODULE_NAME);
    if (ret < 0) {
        pr_err(MODULE_NAME ": alloc_chrdev_region failed (%d)\n", ret);
        return ret;
    }
    taiji_major = MAJOR(dev);
    pr_info(MODULE_NAME ": major=%d\n", taiji_major);

    /* 初始化 cdev */
    cdev_init(&taiji_cdev, &taiji_fops);
    taiji_cdev.owner = THIS_MODULE;
    ret = cdev_add(&taiji_cdev, dev, 1);
    if (ret < 0) {
        pr_err(MODULE_NAME ": cdev_add failed (%d)\n", ret);
        unregister_chrdev_region(dev, 1);
        return ret;
    }

    /* 创建 /sys/class/taiji_os 用于 udev 自动创建设备文件 */
#if LINUX_VERSION_CODE >= KERNEL_VERSION(6,4,0)
    taiji_class = class_create(MODULE_NAME);
#else
    taiji_class = class_create(THIS_MODULE, MODULE_NAME);
#endif
    if (IS_ERR(taiji_class)) {
        pr_warn(MODULE_NAME ": class_create failed, manual mknod needed\n");
    } else {
        taiji_device = device_create(taiji_class, NULL, dev, NULL, MODULE_NAME);
        if (IS_ERR(taiji_device))
            pr_warn(MODULE_NAME ": device_create failed\n");
    }

    /* 创建 /proc/taiji_os/ */
    ret = taiji_proc_init();
    if (ret < 0)
        pr_warn(MODULE_NAME ": /proc init failed (non-fatal)\n");

    pr_info(MODULE_NAME ": loaded successfully\n");
    return 0;
}

static void __exit taiji_exit(void)
{
    dev_t dev = MKDEV(taiji_major, 0);
    struct taiji_session *sess, *tmp;

    pr_info(MODULE_NAME ": unloading\n");

    /* 清理 /proc */
    taiji_proc_cleanup();

    /* 销毁所有活跃 session */
    mutex_lock(&taiji_sessions_lock);
    list_for_each_entry_safe(sess, tmp, &taiji_sessions, list) {
        pr_warn(MODULE_NAME ": session %u still open, force-closing\n", sess->session_id);
        list_del(&sess->list);
        kfree(sess);
    }
    mutex_unlock(&taiji_sessions_lock);

    /* 销毁字符设备 */
    if (taiji_device)
        device_destroy(taiji_class, dev);
    if (taiji_class)
        class_destroy(taiji_class);

    cdev_del(&taiji_cdev);
    unregister_chrdev_region(dev, 1);

    pr_info(MODULE_NAME ": unloaded\n");
}

module_init(taiji_init);
module_exit(taiji_exit);
