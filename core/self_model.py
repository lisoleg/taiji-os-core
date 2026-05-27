import numpy as np


class SelfModel:
    """
    SelfModel: AGI 进程的自我锚定模型。
    维护不变的 Anchor ID 与可进化的自我表示向量 σ（sigma）。
    """

    def __init__(self, sid: str, dim: int = 384):
        self.sid = sid
        self.anchor_id = sid  # 不可变锚点
        self.sigma = np.zeros(dim)
        self.dim = dim
        self.version = 0

    def update(self, feedback_vec: np.ndarray) -> None:
        """根据外部反馈更新自我表示（EMA）。"""
        self.sigma = 0.95 * self.sigma + 0.05 * feedback_vec
        self.version += 1

    def identity(self) -> dict:
        return {
            "anchor_id": self.anchor_id,
            "version": self.version,
            "sigma_norm": float(np.linalg.norm(self.sigma)),
        }
