"""syscalls/auditor.py — 太极OS 系统调用层：审计器"""
import json
import os
from datetime import datetime
from typing import Any, Dict


class Auditor:
    """
    记录所有系统调用和 GAN 步骤到审计日志，支持事后回溯与合规审查。
    """

    def __init__(self, audit_dir: str = "audit"):
        self.audit_dir = audit_dir
        os.makedirs(audit_dir, exist_ok=True)

    def log(self, event_type: str, sid: str, data: Dict[str, Any]) -> str:
        """追加一条审计记录，返回记录 ID。"""
        ts = datetime.utcnow().isoformat()
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

    def read(self, sid: str) -> list:
        """读取指定 session 的全部审计记录。"""
        log_file = os.path.join(self.audit_dir, f"{sid}.jsonl")
        if not os.path.exists(log_file):
            return []
        records = []
        with open(log_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        records.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
        return records
