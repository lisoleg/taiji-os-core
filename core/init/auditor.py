"""core/init/auditor.py — USCS 内核审计器：资源记账 + 计费模型"""
import json
import os
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


class DataAuditor:
    """
    USCS 内核审计器。

    功能：
    - 记录所有会话事件（调度、分配、回收、迁移）到 JSONL 审计日志
    - 按会话生成资源账单（页面-滴答、迁移次数、总费用）
    - 支持事后回溯与合规审查
    """

    # 默认计费率（单位：微积分/单位）
    DEFAULT_RATES: Dict[str, float] = {
        "page_alloc": 0.001,       # 每页每次分配
        "page_reclaim": 0.0005,    # 每页每次回收
        "cpu_tick": 0.01,          # 每时钟滴答
        "migration": 50.0,         # 每次迁移
        "session_init": 1.0,       # 每次会话初始化
    }

    def __init__(
        self,
        audit_dir: str = "audit",
        rates: Optional[Dict[str, float]] = None,
    ):
        self.audit_dir = audit_dir
        self.rates = rates or dict(self.DEFAULT_RATES)
        os.makedirs(audit_dir, exist_ok=True)

    # ── 审计日志 ──────────────────────────────────────────────

    def log(self, event_type: str, sid: str, data: Dict[str, Any]) -> str:
        """追加一条审计记录，返回记录 ID。"""
        ts = datetime.now(timezone.utc).isoformat()
        record_id = f"{sid}-{ts}"
        record = {
            "id": record_id,
            "ts": ts,
            "event_type": event_type,
            "sid": sid,
            "data": data,
        }
        log_file = os.path.join(self.audit_dir, f"{sid}.jsonl")
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        return record_id

    def read(self, sid: str) -> List[Dict[str, Any]]:
        """读取指定 session 的全部审计记录。"""
        log_file = os.path.join(self.audit_dir, f"{sid}.jsonl")
        if not os.path.exists(log_file):
            return []
        records: List[Dict[str, Any]] = []
        with open(log_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        return records

    # ── 资源记账 ──────────────────────────────────────────────

    def audit_session(
        self,
        sid: str,
        event_type: str,
        **kwargs: Any,
    ) -> str:
        """记录一次会话级审计事件（resource accounting）。"""
        return self.log(event_type, sid, kwargs)

    def _aggregate(self, sid: str) -> Dict[str, float]:
        """聚合 session 的资源用量。"""
        records = self.read(sid)
        usage: Dict[str, float] = defaultdict(float)
        for rec in records:
            et = rec.get("event_type", "")
            data = rec.get("data", {})
            if et == "page_alloc":
                usage["pages_allocated"] += data.get("count", 0)
            elif et == "page_reclaim":
                usage["pages_reclaimed"] += data.get("count", 0)
            elif et == "cpu_tick":
                usage["cpu_ticks"] += data.get("ticks", 1)
            elif et == "migration":
                usage["migrations"] += 1
            elif et == "session_init":
                usage["sessions"] += 1
        return dict(usage)

    # ── 计费模型 ──────────────────────────────────────────────

    def generate_bill(self, sid: str) -> Dict[str, Any]:
        """为指定 session 生成计费账单。"""
        usage = self._aggregate(sid)
        items: List[Dict[str, Any]] = []
        total = 0.0

        for resource, amount in usage.items():
            rate_key = {
                "pages_allocated": "page_alloc",
                "pages_reclaimed": "page_reclaim",
                "cpu_ticks": "cpu_tick",
                "migrations": "migration",
                "sessions": "session_init",
            }.get(resource)
            if rate_key is None:
                continue
            rate = self.rates.get(rate_key, 0.0)
            cost = round(amount * rate, 6)
            total += cost
            items.append({
                "resource": resource,
                "amount": amount,
                "rate": rate,
                "cost": cost,
            })

        return {
            "sid": sid,
            "total_cost": round(total, 6),
            "currency": "u-credit",  # 微积分
            "items": items,
            "generated_at": datetime.now(timezone.utc).isoformat(),
        }

    # ── 汇总 ──────────────────────────────────────────────────

    def summary(self) -> Dict[str, Any]:
        """全部 session 的汇总统计。"""
        sessions: Dict[str, int] = {}
        total_events = 0
        for fname in os.listdir(self.audit_dir):
            if fname.endswith(".jsonl"):
                sid = fname[:-6]
                records = self.read(sid)
                sessions[sid] = len(records)
                total_events += len(records)

        return {
            "total_sessions": len(sessions),
            "total_events": total_events,
            "sessions": sessions,
        }
