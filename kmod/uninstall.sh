#!/bin/bash
# uninstall.sh — 太极OS内核模块卸载脚本
# 用法: sudo bash kmod/uninstall.sh

set -e

MODNAME="taiji_os"

echo "=== 太极OS内核模块卸载 ==="

# 1. 卸载模块
if lsmod | grep -q "^$MODNAME "; then
    echo "→ 卸载内核模块..."
    rmmod "$MODNAME" 2>/dev/null || true
    echo "✓ 模块已卸载"
else
    echo "⠠ 模块未加载，跳过"
fi

# 2. 删除设备文件
if [ -e "/dev/$MODNAME" ]; then
    echo "→ 删除设备文件..."
    rm -f "/dev/$MODNAME"
    echo "✓ 设备文件已删除"
fi

# 3. 清理构建产物
MODDIR="$(cd "$(dirname "$0")" && pwd)"
echo "→ 清理构建产物..."
cd "$MODDIR"
make clean 2>/dev/null || true
echo "✓ 构建产物已清理"

echo ""
echo "=== 卸载完成 ==="
