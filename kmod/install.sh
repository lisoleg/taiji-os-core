#!/bin/bash
# install.sh — 太极OS内核模块安装脚本
# 用法: sudo bash kmod/install.sh

set -e

MODNAME="taiji_os"
MODDIR="$(cd "$(dirname "$0")" && pwd)"
MODKO="$MODDIR/taiji_os_kmod.ko"

echo "=== 太极OS内核模块安装 ==="
echo "模块目录: $MODDIR"

# 1. 检查是否已加载
if lsmod | grep -q "^$MODNAME "; then
    echo "⚠ 模块已加载，先卸载..."
    rmmod "$MODNAME" 2>/dev/null || true
fi

# 2. 构建
echo "→ 构建内核模块..."
cd "$MODDIR"
make clean
make
if [ ! -f "$MODKO" ]; then
    echo "✗ 构建失败: $MODKO 不存在"
    exit 1
fi
echo "✓ 构建成功: $MODKO"

# 3. 加载模块
echo "→ 加载内核模块..."
insmod "$MODKO"
sleep 0.5

# 4. 获取主设备号
MAJOR=$(dmesg | grep "$MODNAME.*major" | tail -1 | grep -oP 'major=\K[0-9]+' || echo "")
if [ -z "$MAJOR" ]; then
    # 尝试从 /sys/module/ 读取
    if [ -f "/sys/module/$MODNAME/parameters/major" ]; then
        MAJOR=$(cat "/sys/module/$MODNAME/parameters/major" 2>/dev/null || echo "240")
    else
        MAJOR=240  # 回退到动态分配的常见值
    fi
fi
echo "  主设备号: $MAJOR"

# 5. 创建设备文件
if [ ! -e "/dev/$MODNAME" ]; then
    echo "→ 创建设备文件 /dev/$MODNAME..."
    mknod "/dev/$MODNAME" c "$MAJOR" 0
fi
chmod 666 "/dev/$MODNAME"
echo "✓ 设备文件: /dev/$MODNAME (权限 666)"

# 6. 检查 /proc 接口
if [ -d "/proc/$MODNAME" ]; then
    echo "✓ /proc/$MODNAME/ 可用"
    cat "/proc/$MODNAME/stats"
else
    echo "⚠ /proc/$MODNAME/ 不可用（非致命）"
fi

# 7. 检查 dmesg
echo "--- dmesg 输出 ---"
dmesg | grep "$MODNAME" | tail -10

echo ""
echo "=== 安装完成 ==="
echo "测试: python3 $MODDIR/python/taiji_os_kmod.py --test"
echo "卸载: sudo bash $MODDIR/uninstall.sh"
