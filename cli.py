#!/usr/bin/env python3
"""
cli.py — Taiji OS 命令行客户端
用法:
  python cli.py --sid alice "设计芯片"
  python cli.py --continue <kid>
"""

import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from hal.llm_router import LLMRouter
from core.session import TaijiSession

_sessions: dict = {}
_llm = LLMRouter()


def get_session(sid: str) -> TaijiSession:
    if sid not in _sessions:
        _sessions[sid] = TaijiSession(sid, _llm)
    return _sessions[sid]


def main():
    parser = argparse.ArgumentParser(description="Taiji OS CLI")
    parser.add_argument("cmd", nargs="?", help="要执行的指令")
    parser.add_argument("--sid", default="default", help="Session ID")
    parser.add_argument("--continue", dest="kid", default=None, help="从 Continuation ID 恢复")
    parser.add_argument("--status", action="store_true", help="查看 session 状态")
    args = parser.parse_args()

    sess = get_session(args.sid)

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
        # 交互模式
        print(f"Taiji OS v2.3 · session={args.sid} · 输入 exit 退出")
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


if __name__ == "__main__":
    main()
