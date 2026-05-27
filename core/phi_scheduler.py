import numpy as np
from core.world_model import WorldModel


class PhiScheduler:
    """
    Φ-Scheduler / FlowBreaker:
    检查候选语义向量是否与当前世界模型保持足够的一致性。
    低于阈值时触发 FlowBreaker，拒绝本次更新。
    """

    def __init__(self, threshold: float = 0.65):
        self.threshold = threshold

    def check(self, w: WorldModel, new_psi: np.ndarray):
        """
        返回 (ok: bool, phi_val: float)
        ok=True  → 通过一致性检验
        ok=False → Φ 值过低，触发 FlowBreaker
        """
        phi_val = w.phi(new_psi)
        if phi_val < self.threshold:
            return False, phi_val
        return True, phi_val
