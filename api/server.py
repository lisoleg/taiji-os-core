from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import json
import os
import sys

# 确保项目根目录在 Python 路径中
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hal.llm_router import LLMRouter
from hal.nic_emu import NodeTransport

from core.session import TaijiSession
from core.uscs_mmu import PageAllocator, PageReclaimer
from core.preemptive_scheduler import (
    PreemptiveScheduler,
    Priority,
    ProcessState,
)
from core.migration_agent import MigrationManager, LoadBalancer

import yaml


def _load_config(path: str = "config.yaml") -> dict:
    """加载 YAML 配置文件。"""
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    return {}


app = FastAPI(title="Taiji OS API", version="3.0.0-uscs-kernel")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

uscs_config = _load_config()
kernel = {
    "sessions": {},
    "llm": LLMRouter(),
    # USCS 内存管理
    "allocator": PageAllocator(
        total_pages=uscs_config.get("uscs", {}).get("total_physical_pages", 1048576),
        page_size=uscs_config.get("uscs", {}).get("page_size", 4096),
    ),
    "reclaimer": PageReclaimer(
        policy=uscs_config.get("uscs", {}).get("reclaim_policy", "lru"),
        swap_dir=uscs_config.get("uscs", {}).get("swap_dir", "swap"),
    ),
    # 抢占调度器
    "scheduler": PreemptiveScheduler(
        tick_interval_ms=uscs_config.get("scheduler", {}).get("tick_interval_ms", 100),
        default_time_slice=uscs_config.get("scheduler", {}).get("default_time_slice", 10),
        max_waiting_timeout_ms=uscs_config.get("scheduler", {}).get("max_waiting_timeout_ms", 30000),
    ),
    # 跨节点迁移
    "transport": NodeTransport(
        node_id=uscs_config.get("migration", {}).get("node_id", "node-001"),
        config_path="config.yaml",
    ),
    "migration_mgr": None,  # 延迟初始化（需要 scheduler + allocator + transport）
}
# 初始化迁移管理器
kernel["migration_mgr"] = MigrationManager(
    node_id=uscs_config.get("migration", {}).get("node_id", "node-001"),
    transport=kernel["transport"],
    scheduler=kernel["scheduler"],
    allocator=kernel["allocator"],
    llm_router=kernel["llm"],
)
# 负载均衡器
kernel["balancer"] = LoadBalancer(
    migration_mgr=kernel["migration_mgr"],
)


class RunRequest(BaseModel):
    sid: str = "default"
    cmd: str


@app.get("/health")
async def health():
    return {"status": "ok", "version": "3.0.0-uscs-kernel", "sessions": len(kernel["sessions"])}


@app.post("/run")
async def run(req: RunRequest):
    sid = req.sid
    if sid not in kernel["sessions"]:
        sess = TaijiSession(sid, kernel["llm"])
        # 注册到抢占调度器
        priority_name = uscs_config.get("scheduler", {}).get("default_priority", "MEDIUM")
        pcb = kernel["scheduler"].register(sess, Priority[priority_name])
        kernel["sessions"][sid] = sess
    sess = kernel["sessions"][sid]
    # 调度器管控：阻塞进程需要先唤醒
    if sess.pcb and sess.pcb.state == ProcessState.BLOCKED:
        kernel["scheduler"].unblock(sess.pcb.pid)
    out = sess.run(req.cmd)
    # 执行后主动 yield
    if sess.pcb:
        kernel["scheduler"].yield_cpu(sess.pcb.pid)
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


# ═══════════════════════════════════════════════════════════════
# USCS 内核 API 端点
# ═══════════════════════════════════════════════════════════════

# ---- 调度器端点 ----


@app.get("/scheduler/stats")
async def scheduler_stats():
    """获取调度器统计信息。"""
    return kernel["scheduler"].stats()


@app.get("/scheduler/queues")
async def scheduler_queues():
    """获取各队列快照（调试用）。"""
    return kernel["scheduler"].queue_snapshot()


@app.get("/scheduler/pcb/{pid}")
async def scheduler_pcb(pid: str):
    """获取指定进程的 PCB 状态。"""
    pcb = kernel["scheduler"].get_pcb(pid)
    if pcb is None:
        raise HTTPException(status_code=404, detail=f"PCB for pid='{pid}' not found")
    return pcb.to_dict()


@app.post("/scheduler/tick")
async def scheduler_tick():
    """手动触发一次调度 tick。"""
    scheduled_pid = kernel["scheduler"].tick()
    return {"tick": kernel["scheduler"]._tick_count, "scheduled_pid": scheduled_pid}


# ---- 迁移端点 ----


@app.post("/migration/migrate/{pid}")
async def migration_migrate(pid: str, target_node: str = "node-002"):
    """将进程 pid 迁移到目标节点。"""
    try:
        new_pid = kernel["migration_mgr"].migrate(pid, target_node)
        return {"pid": pid, "new_pid": new_pid, "target_node": target_node, "status": "completed"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Migration failed: {e}")


@app.get("/migration/status/{pid}")
async def migration_status(pid: str):
    """查询进程迁移状态。"""
    return kernel["migration_mgr"].status(pid)


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
