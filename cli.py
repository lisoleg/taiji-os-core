#!/usr/bin/env python3
"""
cli.py — Taiji OS 命令行客户端

用法（文本模式）:
  python cli.py --sid alice "设计芯片"
  python cli.py --continue <kid>
  python cli.py --status

用法（浏览器云脑模式）:
  python cli.py --web "搜索 太极OS"
  python cli.py --web --sid bob "打开 https://github.com"
  python cli.py --web "帮我订明天上海到北京的高铁"
"""

import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from hal.llm_router import LLMRouter
from core.session import TaijiSession

_sessions: dict = {}
_llm = LLMRouter()


def get_session(sid: str, mode: str = "text", headless: bool = True) -> TaijiSession:
    key = f"{sid}:{mode}"
    if key not in _sessions:
        _sessions[key] = TaijiSession(sid, _llm, mode=mode, headless=headless)
    return _sessions[key]


def main():
    parser = argparse.ArgumentParser(description="Taiji OS CLI · 云脑外壳")
    parser.add_argument("cmd", nargs="?", help="要执行的指令")
    parser.add_argument("--sid", default="default", help="Session ID")
    parser.add_argument("--continue", dest="kid", default=None, help="从 Continuation ID 恢复")
    parser.add_argument("--status", action="store_true", help="查看 session 状态")
    parser.add_argument("--web", action="store_true", help="启用浏览器云脑模式")
    parser.add_argument("--no-headless", action="store_true", help="Web 模式显示浏览器窗口")
    args = parser.parse_args()

    mode = "web" if args.web else "text"
    headless = not args.no_headless
    sess = get_session(args.sid, mode=mode, headless=headless)

    if args.kid:
        sess.resume(args.kid)
        print(f"[✓] 已从 Continuation {args.kid} 恢复 session={args.sid}")

    if args.status:
        import json
        print(json.dumps(sess.status(), ensure_ascii=False, indent=2))
        return

    if args.cmd:
        out = sess.run(args.cmd)
        print(out)
    else:
        mode_label = "🌐 浏览器云脑" if mode == "web" else "💬 文本推演"
        print(f"Taiji OS v3.0 · {mode_label} · session={args.sid} · 输入 exit 退出")
        while True:
            try:
                line = input(">>> ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if line.lower() in ("exit", "quit"):
                break
            if not line:
                continue
            print(sess.run(line))

    if mode == "web":
        sess.close()


if __name__ == "__main__":
    main()
