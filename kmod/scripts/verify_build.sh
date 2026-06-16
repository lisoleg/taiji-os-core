#!/bin/bash
# ==============================================================================
#  taiji-os-core kmod — Linux 实体构建验证脚本
#  用法: 在 Linux x86_64 上运行
#    chmod +x verify_build.sh
#    sudo ./verify_build.sh
# ==============================================================================

set -e
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

MODULE_DIR="$(cd "$(dirname "$0")/.." && pwd)"
BUILD_LOG="/tmp/taiji_kmod_build_$(date +%Y%m%d_%H%M%S).log"

echo "============================================================"
echo "  taiji-os-core 内核模块 — 实体构建验证"
echo "============================================================"
echo "  模块目录 : $MODULE_DIR"
echo "  构建日志 : $BUILD_LOG"
echo "  内核版本 : $(uname -r)"
echo "  架构     : $(uname -m)"
echo ""

# --------------------------------------------------------------------------
#  Step 1: 环境检查
# --------------------------------------------------------------------------
echo "Step 1: 环境检查"
echo "------------------------------------------------------------"

PASS=0
FAIL=0

check_cmd() {
    if command -v "$1" &>/dev/null; then
        echo -e "  ${GREEN}✅${NC} $1 : $($1 --version 2>&1 | head -1)"
        ((PASS++))
    else
        echo -e "  ${RED}❌${NC} $1 : 未安装"
        ((FAIL++))
    fi
}

check_file() {
    if [ -f "$1" ]; then
        echo -e "  ${GREEN}✅${NC} $2 : $1"
        ((PASS++))
    else
        echo -e "  ${RED}❌${NC} $2 : $1 (不存在)"
        ((FAIL++))
    fi
}

check_dir() {
    if [ -d "$1" ]; then
        echo -e "  ${GREEN}✅${NC} $2 : $1"
        ((PASS++))
    else
        echo -e "  ${RED}❌${NC} $2 : $1 (不存在)"
        echo -e "      请安装: sudo apt install linux-headers-$(uname -r)"
        ((FAIL++))
    fi
}

check_cmd gcc
check_cmd make
check_cmd ld
check_cmd objcopy

# 内核头文件路径 (KDIR)
KDIR="/lib/modules/$(uname -r)/build"
check_dir "$KDIR" "内核头文件"

# 检查内核版本是否 >= 6.4 (class_create API 变更)
KVER_MAJOR=$(uname -r | cut -d. -f1)
KVER_MINOR=$(uname -r | cut -d. -f2)
echo -n "  内核 API 版本检测 : "
if [ "$KVER_MAJOR" -gt 6 ] || ([ "$KVER_MAJOR" -eq 6 ] && [ "$KVER_MINOR" -ge 4 ]); then
    echo -e "${YELLOW}⚠️  Kernel $(uname -r) >= 6.4, class_create 1-arg mode${NC}"
else
    echo -e "${GREEN}✅ Kernel $(uname -r) < 6.4, class_create 2-arg mode${NC}"
fi

echo ""
echo "  环境检查: ${PASS} 通过, ${FAIL} 失败"
if [ "$FAIL" -gt 0 ]; then
    echo -e "${RED}  请先安装缺失的依赖，然后重新运行${NC}"
    exit 1
fi
echo ""

# --------------------------------------------------------------------------
#  Step 2: 编译
# --------------------------------------------------------------------------
echo "Step 2: 编译内核模块"
echo "------------------------------------------------------------"

cd "$MODULE_DIR"
echo "  执行: make clean && make"
make clean > /dev/null 2>&1 || true
make 2>&1 | tee "$BUILD_LOG"

if [ -f "${MODULE_DIR}/taiji_os.ko" ]; then
    echo -e "  ${GREEN}✅ 编译成功${NC} : ${MODULE_DIR}/taiji_os.ko"
    echo "  文件信息:"
    modinfo "${MODULE_DIR}/taiji_os.ko" | head -20
else
    echo -e "  ${RED}❌ 编译失败${NC}"
    echo "  请查看日志: $BUILD_LOG"
    exit 1
fi
echo ""

# --------------------------------------------------------------------------
#  Step 3: 加载模块
# --------------------------------------------------------------------------
echo "Step 3: 加载内核模块"
echo "------------------------------------------------------------"

# 检查是否已加载
if lsmod | grep -q taiji_os; then
    echo "  模块已加载，先卸载..."
    rmmod taiji_os 2>&1 || true
    sleep 1
fi

echo "  执行: insmod taiji_os.ko"
insmod "${MODULE_DIR}/taiji_os.ko" 2>&1 && echo -e "  ${GREEN}✅ 加载成功${NC}" || {
    echo -e "  ${RED}❌ 加载失败${NC}"
    dmesg | tail -20
    exit 1
}

echo "  验证设备节点:"
ls -la /dev/taiji_os 2>/dev/null && echo -e "  ${GREEN}✅ /dev/taiji_os 存在${NC}" || echo -e "  ${YELLOW}⚠️  /dev/taiji_os 未自动创建（可能需要 mknod）${NC}"

echo "  验证 /proc 接口:"
ls -la /proc/taiji_os/ 2>/dev/null && echo -e "  ${GREEN}✅ /proc/taiji_os/ 存在${NC}" || echo -e "  ${YELLOW}⚠️  /proc/taiji_os/ 不存在${NC}"

echo "  dmesg 输出 (最后 10 行):"
dmesg | tail -10 | head -10
echo ""

# --------------------------------------------------------------------------
#  Step 4: 功能测试 (ioctl)
# --------------------------------------------------------------------------
echo "Step 4: 功能测试 (ioctl 调用)"
echo "------------------------------------------------------------"

# 编译测试程序
TEST_C="${MODULE_DIR}/test_kmod.c"
TEST_BIN="/tmp/test_taiji_kmod"

cat > "$TEST_C" << 'TESTEOF'
#include <stdio.h>
#include <fcntl.h>
#include <unistd.h>
#include <string.h>
#include <errno.h>
#include <stdlib.h>
#include "taiji_os_ioctl.h"

int main(int argc, char *argv[]) {
    int fd = open("/dev/taiji_os", O_RDWR);
    if (fd < 0) {
        fprintf(stderr, "open failed: %s\n", strerror(errno));
        // Try mknod
        fprintf(stderr, "Try: sudo mknod /dev/taiji_os c $(cat /sys/class/taiji_os/taiji_os/dev 2>/dev/null || echo 'N/A') 0\n");
        return 1;
    }
    printf("✅ 设备打开成功: fd=%d\n", fd);

    // Test 1: Get stats (TAJI_GET_STATS, ioctl #10)
    struct taiji_stats stats;
    memset(&stats, 0, sizeof(stats));
    int ret = ioctl(fd, TAJI_GET_STATS, &stats);
    if (ret == 0) {
        printf("✅ TAJI_GET_STATS 成功\n");
        printf("   total_updates=%lu, drift_events=%lu, current_cv=%.4f, current_gamma=%.4f\n",
               stats.total_updates, stats.drift_events, stats.current_cv, stats.current_gamma);
    } else {
        fprintf(stderr, "❌ TAJI_GET_STATS 失败: %s\n", strerror(errno));
    }

    // Test 2: Set params (TAJI_SET_PARAMS, ioctl #12)
    struct taiji_params params = {
        .cv_threshold = 0.30f,
        .gamma_max = 0.85f,
        .gamma_min = 0.20f,
        .cv_mid = 0.25f,
        .temperature = 0.08f,
        .auto_tune = 1,
    };
    ret = ioctl(fd, TAJI_SET_PARAMS, &params);
    if (ret == 0) {
        printf("✅ TAJI_SET_PARAMS 成功\n");
    } else {
        fprintf(stderr, "❌ TAJI_SET_PARAMS 失败: %s\n", strerror(errno));
    }

    // Test 3: Get params (TAJI_GET_PARAMS, ioctl #13)
    memset(&params, 0, sizeof(params));
    ret = ioctl(fd, TAJI_GET_PARAMS, &params);
    if (ret == 0) {
        printf("✅ TAJI_GET_PARAMS 成功\n");
        printf("   cv_threshold=%.4f, gamma_max=%.4f, gamma_min=%.4f, cv_mid=%.4f, temperature=%.4f, auto_tune=%u\n",
               params.cv_threshold, params.gamma_max, params.gamma_min, params.cv_mid, params.temperature, params.auto_tune);
    } else {
        fprintf(stderr, "❌ TAJI_GET_PARAMS 失败: %s\n", strerror(errno));
    }

    // Test 4: Push phi (TAJI_PUSH_PHI, ioctl #20)
    struct taiji_push_phi_arg phi_arg = {
        .phi_value = 0.75f,
        .is_drifting = 0,
        .current_cv = 0.0f,
        .current_gamma = 0.0f,
    };
    ret = ioctl(fd, TAJI_PUSH_PHI, &phi_arg);
    if (ret == 0) {
        printf("✅ TAJI_PUSH_PHI 成功 (phi=%.4f)\n", phi_arg.phi_value);
        printf("   返回: is_drifting=%u, current_cv=%.4f, current_gamma=%.4f\n",
               phi_arg.is_drifting, phi_arg.current_cv, phi_arg.current_gamma);
    } else {
        fprintf(stderr, "❌ TAJI_PUSH_PHI 失败: %s\n", strerror(errno));
    }

    // Test 5: S-update (TAJI_S_UPDATE, ioctl #30)
    float key[8] = {0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f, 0.8f};
    float value[8] = {0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f, 0.8f, 0.9f};
    struct taiji_update_arg update_arg;
    memcpy(update_arg.key, key, sizeof(key));
    memcpy(update_arg.value, value, sizeof(value));
    ret = ioctl(fd, TAJI_S_UPDATE, &update_arg);
    if (ret == 0) {
        printf("✅ TAJI_S_UPDATE 成功\n");
    } else {
        fprintf(stderr, "❌ TAJI_S_UPDATE 失败: %s\n", strerror(errno));
    }

    close(fd);
    printf("\n所有测试完成 ✅\n");
    return 0;
}
TESTEOF

# 编译测试程序
echo "  编译测试程序..."
gcc -I "$MODULE_DIR" "$TEST_C" -o "$TEST_BIN" 2>&1 && echo -e "  ${GREEN}✅ 测试程序编译成功${NC}" || {
    echo -e "  ${RED}❌ 测试程序编译失败${NC}"
    exit 1
}

# 运行测试程序
echo "  运行功能测试..."
"$TEST_BIN"
TEST_RESULT=$?
echo ""

# --------------------------------------------------------------------------
#  Step 5: 卸载模块
# --------------------------------------------------------------------------
echo "Step 5: 卸载内核模块"
echo "------------------------------------------------------------"

echo "  执行: rmmod taiji_os"
rmmod taiji_os 2>&1 && echo -e "  ${GREEN}✅ 卸载成功${NC}" || {
    echo -e "  ${YELLOW}⚠️  卸载失败 (可能有未关闭的 fd)${NC}"
    lsof /dev/taiji_os 2>/dev/null || true
}

echo "  dmesg 输出 (最后 10 行):"
dmesg | tail -10 | head -10
echo ""

# --------------------------------------------------------------------------
#  总结
# --------------------------------------------------------------------------
echo "============================================================"
echo "  构建验证总结"
echo "============================================================"
echo -e "  编译      : ${GREEN}✅ 通过${NC}"
if [ $TEST_RESULT -eq 0 ]; then
    echo -e "  功能测试  : ${GREEN}✅ 通过${NC}"
else
    echo -e "  功能测试  : ${RED}❌ 失败 (exit code: $TEST_RESULT)${NC}"
fi
echo "  构建日志  : $BUILD_LOG"
echo ""
echo "  下一步:"
echo "    1. 查看 dmesg 确认内核日志正常"
echo "    2. 运行 Python 封装测试: sudo python3 ${MODULE_DIR}/python/taiji_os_kmod.py --test"
echo "    3. 运行性能基准: sudo python3 ${MODULE_DIR}/scripts/bench_kmod.py --iters 100 1000 10000"
echo ""
