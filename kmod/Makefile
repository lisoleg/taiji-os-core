# kmod/Makefile — 太极OS内核模块构建
#
# 用法:
#   make           # 构建 taiji_os_kmod.ko
#   make test     # 安装 → 运行测试 → 卸载
#   make clean    # 清理构建产物
#   make install  # 仅安装（不测试）
#   make uninstall # 卸载

# ── 内核构建目录 ──────────────────────────────────────
# 自动检测当前内核的构建目录
KDIR ?= /lib/modules/$(shell uname -r)/build
PWD  := $(shell pwd)

# ── 模块名 ────────────────────────────────────────────
obj-m += taiji_os_kmod.o

# ── 编译选项 ─────────────────────────────────────────
# -DDEBUG: 启用 pr_debug() 输出
# -Werror: 将警告视为错误（开发时用）
EXTRA_CFLAGS := -DDEBUG -Werror -Wno-error=deprecated-declarations

# ── 默认目标 ──────────────────────────────────────────

all: modules

modules:
	$(MAKE) -C $(KDIR) M=$(PWD) modules

clean:
	$(MAKE) -C $(KDIR) M=$(PWD) clean
	rm -f *.o.d *.mod.c .*.cmd modules.order Module.symvers

# ── 安装 / 卸载 ─────────────────────────────────────

install: all
	@echo "→ 安装 taiji_os_kmod.ko..."
	@sudo insmod taiji_os_kmod.ko || echo "  警告: insmod 失败（可能已加载）"
	@sleep 0.5
	@if [ -e /sys/module/taiji_os_kmod/parameters/major ]; then \
		MAJOR=$$(cat /sys/module/taiji_os_kmod/parameters/major 2>/dev/null || \
		       dmesg | grep "taiji_os.*major" | tail -1 | awk '{print $$NF}'); \
		if [ -z "$$MAJOR" ]; then \
			MAJOR=$$(ls -l /dev/taiji_os 2>/dev/null | awk '{print $$5}' | cut -d, -f1); \
		fi; \
		echo "  主设备号: $$MAJOR"; \
	else \
		MAJOR=$$(dmesg | grep "taiji_os.*major" | tail -1 | grep -oP 'major=\K[0-9]+'); \
		echo "  主设备号（dmesg): $$MAJOR"; \
	fi
	@if [ ! -e /dev/taiji_os ]; then \
		sudo mknod /dev/taiji_os c $$(dmesg | grep "taiji_os.*major" | tail -1 | grep -oP 'major=\K[0-9]+' || echo 240) 0; \
		echo "  创建设备文件 /dev/taiji_os"; \
	fi
	@sudo chmod 666 /dev/taiji_os
	@echo "✓ 安装完成"
	@sudo dmesg | grep "taiji_os" | tail -5

uninstall:
	@echo "→ 卸载 taiji_os_kmod..."
	@sudo rmmod taiji_os_kmod || echo "  警告: rmmod 失败（可能未加载）"
	@sudo rm -f /dev/taiji_os
	@echo "✓ 卸载完成"

# ── 测试 ─────────────────────────────────────────────

test: install
	@echo "→ 运行内核模块测试..."
	@if [ -e python/taiji_os_kmod.py ]; then \
		sudo python3 python/taiji_os_kmod.py --test; \
	else \
		echo "  未找到 python/taiji_os_kmod.py，跳过"; \
	fi
	@if [ -e tests/test_kmod.py ]; then \
		sudo python3 tests/test_kmod.py; \
	else \
		echo "  未找到 tests/test_kmod.py，跳过"; \
	fi
	@$(MAKE) uninstall

# ── 快捷目标 ──────────────────────────────────────────

reload: uninstall install

status:
	@echo "=== taiji_os_kmod 状态 ==="
	@lsmod | grep taiji_os || echo "  模块未加载"
	@echo "--- /dev/taiji_os ---"
	@ls -la /dev/taiji_os 2>/dev/null || echo "  设备文件不存在"
	@echo "--- /proc/taiji_os/ ---"
	@cat /proc/taiji_os/stats 2>/dev/null || echo "  /proc 接口不可用"
	@dmesg | grep "taiji_os" | tail -10

# ── 交叉编译（示例）────────────────────────────────
# ARCH=arm64 KDIR=/path/to/arm64/kernel make

.PHONY: all modules clean install uninstall test reload status
