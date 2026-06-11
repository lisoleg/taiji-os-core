"""
Enhanced Φ-Gate with δ-mem S flush decision logic.

Extends the PhiScheduler to add δ-mem flush awareness:
  - When Φ is high (strong consistency), flush S to Episodic Memory
  - When Φ is low (drift detected), prevent S corruption by pausing updates
  - Maintain flush history for audit

This is a lightweight wrapper — the heavy lifting is in DeltaFusion.flush_if_needed().

Author: Taiji OS Team (Zhang Feng, Li Zonghai)
Version: v1.0 — Prototype (2026-06-11)
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class DeltaPhiGate:
    """Enhanced Φ-Gate with δ-mem integration.

    Wraps the existing PhiScheduler and adds:
      - S flush decision on high Φ
      - S update pausing on low Φ (drift protection)
      - Flush audit trail

    Usage:
        gate = DeltaPhiGate(phi_scheduler, delta_fusion)
        ok, phi_val, flushed = gate.check(world_model, new_psi, sid)
    """

    phi_scheduler: object = None           # Original PhiScheduler instance
    delta_fusion: object = None            # DeltaFusion instance
    flush_history: list = field(default_factory=list)   # List of flush events
    pause_on_drift: bool = True            # Pause S updates when Φ < threshold
    is_paused: bool = False
    total_flushes: int = 0
    total_blocks: int = 0

    def check(self, world_model, new_psi, sid: str = "unknown") -> tuple:
        """Φ-Gate check with δ-mem flush decision.

        Args:
            world_model: WorldModel instance (has .phi() method).
            new_psi: New ψ vector to check.
            sid: Session ID for episodic entries.

        Returns:
            (accepted: bool, phi_value: float, flushed_entry: Optional[EpisodicMemoryEntry])
        """
        # 1. Run original Φ check
        if self.phi_scheduler is not None:
            ok, phi_val = self.phi_scheduler.check(world_model, new_psi)
        else:
            phi_val = float(world_model.phi(new_psi))
            ok = phi_val >= 0.65

        flushed = None

        # 2. Handle drift: pause S updates if Φ too low
        if self.pause_on_drift and not ok:
            self.is_paused = True
            self.total_blocks += 1
            return ok, phi_val, flushed

        # 3. Handle high consistency: flush S to Episodic Memory
        if self.delta_fusion is not None and ok:
            self.is_paused = False
            flushed = self.delta_fusion.flush_if_needed(phi_val, sid)
            if flushed is not None:
                self.total_flushes += 1
                self.flush_history.append({
                    "eid": flushed.eid,
                    "phi": phi_val,
                    "timestamp": flushed.timestamp,
                })
                # Keep only last 100 flush records
                if len(self.flush_history) > 100:
                    self.flush_history = self.flush_history[-100:]

        return ok, phi_val, flushed

    def should_ingest(self) -> bool:
        """Check if δ-mem should accept new (k, v) pairs.

        Returns False when Φ is too low (drift detected, updates paused).
        """
        if not self.pause_on_drift:
            return True
        return not self.is_paused

    def stats(self) -> dict:
        """Return gate statistics for monitoring."""
        return {
            "total_flushes": self.total_flushes,
            "total_blocks": self.total_blocks,
            "is_paused": self.is_paused,
            "flush_history_len": len(self.flush_history),
            "last_flush": self.flush_history[-1] if self.flush_history else None,
        }
