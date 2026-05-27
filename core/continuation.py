import uuid
import json
import os
import numpy as np
from datetime import datetime


class Continuation:
    """
    Continuation (k): AGI 进程的可序列化快照。
    保存 ψ 向量、环境状态、中断原因，支持跨节点迁移恢复。
    """

    def __init__(self, sid: str, psi: np.ndarray, env: dict, reason: str,
                 snapshot_dir: str = "snapshots"):
        self.kid = str(uuid.uuid4())[:8]
        self.sid = sid
        self.psi = psi
        self.env = env
        self.reason = reason
        self.ts = datetime.utcnow().isoformat()
        self.snapshot_dir = snapshot_dir
        self._save()

    def _save(self):
        os.makedirs(self.snapshot_dir, exist_ok=True)
        path = os.path.join(self.snapshot_dir, f"{self.kid}.json")
        payload = {
            "kid": self.kid,
            "sid": self.sid,
            "psi": self.psi.tolist(),
            "env": self.env,
            "reason": self.reason,
            "ts": self.ts,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, kid: str, snapshot_dir: str = "snapshots") -> "Continuation":
        path = os.path.join(snapshot_dir, f"{kid}.json")
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        obj = object.__new__(cls)
        obj.kid = data["kid"]
        obj.sid = data["sid"]
        obj.psi = np.array(data["psi"])
        obj.env = data["env"]
        obj.reason = data["reason"]
        obj.ts = data["ts"]
        obj.snapshot_dir = snapshot_dir
        return obj

    def __repr__(self):
        return f"<Continuation kid={self.kid} sid={self.sid} reason={self.reason}>"
