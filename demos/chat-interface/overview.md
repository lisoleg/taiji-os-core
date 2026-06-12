# 太极OS 聊天演示界面 — 项目概览

## 交付物

单文件 HTML 演示界面 (`demos/chat-interface/index.html`)，零构建工具依赖，CDN 加载 Chart.js。

## 功能特性

| 功能 | 说明 |
|------|------|
| 左侧聊天面板 | ChatGPT 风格消息气泡，用户蓝色靠右，AI 灰色靠左 |
| 暂停/恢复按钮 | ⏸ 暂停（红色脉冲动画）/ ▶ 恢复 |
| 自动演示模式 | 每 3.5 秒推进一轮，共 13 轮完整对话 |
| 手动控制 | 下一轮按钮、空格键暂停、Ctrl+A 自动播放、→ 键推进 |
| Φ-Gate 门控 | 环形进度条，颜色绿→黄→红，PASS/BLOCK 状态 |
| CV 漂移检测 | Chart.js 折线图，20 轮历史 + 阈值线(0.30) |
| δ-mem S 矩阵 | 8×8 热力图，蓝(-1)→白(0)→红(+1) |
| 漂移阶段标签 | STABLE(绿) / DRIFTING(红) / RECOVERY(橙) |
| D-Core 幻觉检测 | 最近 5 轮 Pass/Fail 日志 |
| FLUX 有效吞吐 | 累计统计 |
| 手动输入 | 输入框支持自由对话 |

## 技术方案

- **纯 HTML + CSS + JS**，无 React/Vite/npm
- Chart.js v4.4.0 CDN 绘图
- 深色主题（ChatGPT Dark 风格）
- 毛玻璃半透明卡片 + 动画过渡
- 响应式设计（移动端上下布局）

## 预置演示脚本 (13 轮)

```
R1-5:  STABLE (机器学习主题, CV上升 0.05→0.19)
R6-8:  DRIFTING (量子/区块链/分布式系统, CV 0.31→0.44, decay 0.52→0.22)
R9:    幻觉检测 (CAP定理, Φ=0.88, S暂停)
R10-11: RECOVERY (回到ML, 含一次幻觉)
R12-13: STABLE (完全恢复, CV 0.16, decay 0.85)
```

## 启动方式

1. 浏览器直接打开 `demos/chat-interface/index.html`（需要网络加载 Chart.js CDN）
2. 或本地服务器：`python -m http.server 8000` 然后访问 `http://localhost:8000`

## 键盘快捷键

| 键 | 功能 |
|-----|------|
| 空格 | 暂停/恢复 |
| → 右键 | 下一轮 |
| Ctrl+A | 自动播放/停止 |
| Enter | 发送输入消息 |
