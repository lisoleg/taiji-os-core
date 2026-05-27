from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import json
import os
import sys

# 确保项目根目录在 Python 路径中
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hal.llm_router import LLMRouter
from core.session import TaijiSession

app = FastAPI(title="Taiji OS API", version="2.3.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

kernel = {"sessions": {}, "llm": LLMRouter()}


class RunRequest(BaseModel):
    sid: str = "default"
    cmd: str


@app.get("/health")
async def health():
    return {"status": "ok", "version": "2.3.0", "sessions": len(kernel["sessions"])}


@app.post("/run")
async def run(req: RunRequest):
    sid = req.sid
    if sid not in kernel["sessions"]:
        kernel["sessions"][sid] = TaijiSession(sid, kernel["llm"])
    sess = kernel["sessions"][sid]
    out = sess.run(req.cmd)
    return {"sid": sid, "output": out}


@app.get("/session/{sid}/status")
async def session_status(sid: str):
    if sid not in kernel["sessions"]:
        raise HTTPException(status_code=404, detail=f"Session '{sid}' not found")
    return kernel["sessions"][sid].status()


@app.post("/session/{sid}/resume/{kid}")
async def resume_session(sid: str, kid: str):
    if sid not in kernel["sessions"]:
        kernel["sessions"][sid] = TaijiSession(sid, kernel["llm"])
    kernel["sessions"][sid].resume(kid)
    return {"sid": sid, "resumed_from": kid, "status": "ok"}


@app.websocket("/ws")
async def ws(ws: WebSocket):
    await ws.accept()
    try:
        while True:
            msg = await ws.receive_json()
            sid = msg.get("sid", "default")
            if sid not in kernel["sessions"]:
                kernel["sessions"][sid] = TaijiSession(sid, kernel["llm"])
            sess = kernel["sessions"][sid]
            out = sess.run(msg["cmd"])
            await ws.send_text(out)
    except WebSocketDisconnect:
        pass
    except Exception as e:
        await ws.send_text(f"[Error] {e}")
