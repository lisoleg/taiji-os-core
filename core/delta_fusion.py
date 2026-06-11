"""
δ-mem ↔ Taiji OS Fusion Bridge (L1 ↔ L2).

Implements the fusion architecture described in OSDI paper §5.7:

  L1 (δ-mem Hot Cache)  ────Φ-Gate───▶  L2 (Taiji OS Cold Storage)
  S ∈ R^(8×8)                          ψ + Episodic Memory (Walrus)
  64 floats                            Continuation Snapshot

Key integration points:
  1. Φ-Gate flush:  When Φ > Φ_flush, flush S to Episodic Memory
  2. Continuation serialization:  S matrix inside Continuation snapshot
  3. Re-anchor:  On resume, align S with the restored ψ
  4. Episodic Index:  Store flushed S states in Walrus MemoryHub

Author: Taiji OS Team (Zhang Feng, Li Zonghai)
Version: v1.0 — Prototype (2026-06-11)
"""

import hashlib
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

from .delta_mem import DeltaMemLayer, SMatrix, DEFAULT_RANK, project_to_srank


# ────────────────────────────────────────────────────────────────────────────
# EpisodicMemoryEntry — A single flushed S-state in L2 cold storage
# ────────────────────────────────────────────────────────────────────────────

@dataclass
class EpisodicMemoryEntry:
    """A single entry in the Episodic Memory (L2 cold storage).

    Created when the Φ-Gate decides to flush the hot cache (S matrix)
    to persistent storage.

    Attributes:
        eid: Unique entry ID.
        sid: Source session ID.
        S_flushed: The S matrix state at flush time (r×r numpy array).
        phi_value: Φ consistency score at flush time.
        timestamp: ISO 8601 timestamp of flush.
        proof: SHA-256 integrity hash.
    """

    eid: str
    sid: str
    S_flushed: np.ndarray
    phi_value: float
    timestamp: str
    proof: str = ""

    def __post_init__(self):
        if not self.proof:
            self.proof = self._compute_proof()

    def _compute_proof(self) -> str:
        data = (
            self.eid.encode()
            + self.S_flushed.tobytes()
            + struct_pack_float(self.phi_value)
            + self.timestamp.encode()
        )
        return hashlib.sha256(data).hexdigest()[:16]

    def to_dict(self) -> dict:
        return {
            "eid": self.eid,
            "sid": self.sid,
            "S_flushed": self.S_flushed.tolist(),
            "phi_value": self.phi_value,
            "timestamp": self.timestamp,
            "proof": self.proof,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "EpisodicMemoryEntry":
        return cls(
            eid=data["eid"],
            sid=data["sid"],
            S_flushed=np.array(data["S_flushed"], dtype=np.float32),
            phi_value=data["phi_value"],
            timestamp=data["timestamp"],
            proof=data.get("proof", ""),
        )


def struct_pack_float(f: float) -> bytes:
    """Pack a float into bytes for proof computation."""
    import struct
    return struct.pack("<d", f)


# ────────────────────────────────────────────────────────────────────────────
# DeltaFusion — the L1 ↔ L2 bridge
# ────────────────────────────────────────────────────────────────────────────

@dataclass
class DeltaFusion:
    """δ-mem ↔ Taiji OS fusion bridge.

    Manages the full L1/L2 lifecycle:
      - δ-mem ingest and query (L1)
      - Φ-Gate controlled flush to Episodic Memory (L1 → L2)
      - S matrix serialization into Continuation snapshots
      - Re-anchor on session resume

    Usage:
        fusion = DeltaFusion(delta_layer, world_model, memory_hub)
        fusion.ingest(key_vec, value_vec)
        fusion.flush_if_needed(phi_value)    # Called by Φ-Gate
        snapshot_dict = fusion.serialize()    # Into Continuation
        fusion.re_anchor(psi)                 # On resume
    """

    delta_layer: DeltaMemLayer = field(default_factory=DeltaMemLayer.create_default)
    episodic_entries: list = field(default_factory=list)     # EpisodicMemoryEntry[]
    flush_threshold: float = 0.85           # Φ > this → flush
    flush_enabled: bool = True
    max_episodic_entries: int = 1000        # Limit L2 entries
    _world_model = None                      # Weak ref to WorldModel (set later)
    _memory_hub = None                       # Weak ref to MemoryHub (set later)

    # ── L1 Operations ────────────────────────────────────────────────────

    def ingest(self, key_vec: np.ndarray, value_vec: np.ndarray) -> "DeltaFusion":
        """Ingest a (k, v) pair into the δ-mem hot cache."""
        kr = project_to_srank(key_vec, self.delta_layer.smatrix.r)
        vr = project_to_srank(value_vec, self.delta_layer.smatrix.r)
        self.delta_layer.ingest(kr, vr)
        return self

    def query(self, q: np.ndarray) -> np.ndarray:
        """Read residual signal for query q from δ-mem."""
        qr = project_to_srank(q, self.delta_layer.smatrix.r)
        return self.delta_layer.query(qr)

    def query_attention_delta(self, q: np.ndarray, k: np.ndarray) -> np.ndarray:
        """Compute attention correction Δ from δ-mem for (q, k)."""
        qr = project_to_srank(q, self.delta_layer.smatrix.r)
        kr = project_to_srank(k, self.delta_layer.smatrix.r)
        return self.delta_layer.correct_attention(qr, kr)

    # ── Φ-Gate Flush (L1 → L2) ──────────────────────────────────────────

    def flush_if_needed(self, phi_value: float, sid: str = "unknown") -> Optional[EpisodicMemoryEntry]:
        """Check Φ-Gate condition and flush S to Episodic Memory if warranted.

        Called by the Φ-Gate after each consistency check.

        Args:
            phi_value: Current Φ (cosine similarity) score.
            sid: Session ID for the episodic entry.

        Returns:
            EpisodicMemoryEntry if a flush occurred, None otherwise.
        """
        if not self.flush_enabled:
            return None

        should_flush = (
            phi_value >= self.flush_threshold
            and self.delta_layer.is_dirty_since_last_flush()
        )
        if not should_flush:
            return None

        return self._do_flush(phi_value, sid)

    def _do_flush(self, phi_value: float, sid: str) -> EpisodicMemoryEntry:
        """Execute the flush: copy S → Episodic Entry, soft-reset S."""
        S_state = self.delta_layer.flush()         # Returns S copy + soft reset

        eid = f"epi-{int(time.time() * 1000)}-{self.delta_layer.flushed_count:04d}"
        entry = EpisodicMemoryEntry(
            eid=eid,
            sid=sid,
            S_flushed=S_state,
            phi_value=phi_value,
            timestamp=time.strftime("%Y-%m-%dT%H:%M:%S"),
        )
        self.episodic_entries.append(entry)

        # Prune old entries if over limit
        if len(self.episodic_entries) > self.max_episodic_entries:
            self.episodic_entries = self.episodic_entries[-self.max_episodic_entries:]

        # If MemoryHub is available, store the entry there too
        if self._memory_hub is not None and hasattr(self._memory_hub, "store"):
            try:
                self._memory_hub.store(
                    sid=sid,
                    data={
                        "type": "delta_mem_flush",
                        "eid": eid,
                        "phi": phi_value,
                        "S_shape": list(S_state.shape),
                        "S_sample": S_state[0, :4].tolist(),
                    },
                )
            except Exception:
                pass  # Non-critical: MemoryHub write failure is OK

        return entry

    # ── Continuation Serialization ───────────────────────────────────────

    def serialize_s(self) -> dict:
        """Serialize δ-mem S state for inclusion in a Continuation snapshot.

        Returns a JSON-safe dict that can be stored alongside ψ, env, etc.
        """
        return {
            "delta_mem": self.delta_layer.to_dict(),
            "episodic_count": len(self.episodic_entries),
            "last_flush_eid": self.episodic_entries[-1].eid if self.episodic_entries else None,
        }

    def deserialize_s(self, s_data: dict) -> None:
        """Restore δ-mem S state from a Continuation snapshot."""
        if "delta_mem" in s_data:
            self.delta_layer = DeltaMemLayer.from_dict(s_data["delta_mem"])

    # ── Re-anchor ────────────────────────────────────────────────────────

    def re_anchor(self, psi: np.ndarray, top_k: int = 3) -> list:
        """Re-anchor δ-mem S matrix against restored ψ on session resume.

        On resume, the S matrix may be stale relative to the new ψ context.
        Re-anchoring adjusts S by replaying the top-k most recent episodic
        entries that are most similar to the current ψ.

        Args:
            psi: Current ψ semantic vector (post-resume).
            top_k: Number of recent episodic entries to replay.

        Returns:
            List of replayed entry IDs.
        """
        if not self.episodic_entries:
            return []

        # Score entries by Φ similarity to current ψ
        psi_flat = psi.ravel()
        scored = []
        for e in self.episodic_entries:
            S_flat = e.S_flushed.ravel()
            # Simple cosine similarity proxy in flattened space
            norm_psi = np.linalg.norm(psi_flat)
            norm_S = np.linalg.norm(S_flat)
            if norm_psi > 0 and norm_S > 0:
                score = float(np.dot(psi_flat[:8], S_flat[:8]) / (norm_psi * norm_S + 1e-8))
            else:
                score = 0.0
            scored.append((score, e))

        # Sort by similarity (descending), take top_k
        scored.sort(key=lambda x: x[0], reverse=True)
        top_entries = scored[:top_k]

        # Replay top entries into S (lightweight re-absorption)
        replayed = []
        for score, entry in top_entries:
            if score > 0.5:  # Only replay highly relevant entries
                # Absorb: S += α * entry.S_flushed (soft blend)
                alpha = 0.3 * score
                self.delta_layer.smatrix.S += alpha * entry.S_flushed
                replayed.append(entry.eid)

        return replayed

    # ── Full State ───────────────────────────────────────────────────────

    def to_dict(self) -> dict:
        """Full serialization of the fusion bridge state."""
        return {
            "delta_layer": self.delta_layer.to_dict(),
            "episodic_entries": [e.to_dict() for e in self.episodic_entries[-50:]],  # Last 50
            "flush_threshold": self.flush_threshold,
            "flush_enabled": self.flush_enabled,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "DeltaFusion":
        """Deserialize the full fusion bridge state."""
        fusion = cls(
            delta_layer=DeltaMemLayer.from_dict(data["delta_layer"]),
            flush_threshold=data.get("flush_threshold", 0.85),
            flush_enabled=data.get("flush_enabled", True),
        )
        if "episodic_entries" in data:
            fusion.episodic_entries = [
                EpisodicMemoryEntry.from_dict(e) for e in data["episodic_entries"]
            ]
        return fusion

    # ── Bind to Taiji OS components ──────────────────────────────────────

    def bind_world_model(self, wm) -> None:
        """Bind to the Taiji OS WorldModel for ψ access."""
        self._world_model = wm

    def bind_memory_hub(self, mh) -> None:
        """Bind to the Taiji OS MemoryHub (Walrus) for persistent storage."""
        self._memory_hub = mh


# ────────────────────────────────────────────────────────────────────────────
# Utility: Create fusion layer from config
# ────────────────────────────────────────────────────────────────────────────

def create_fusion_from_config(config: dict) -> DeltaFusion:
    """Create a DeltaFusion instance from a configuration dict.

    Expected config keys (under 'delta_mem' section):
        enabled: bool
        rank: int (default 8)
        lambda_decay: float (default 0.95)
        beta_update: float (default 0.1)
        flush_enabled: bool (default True)
        flush_threshold: float (default 0.85)

    Args:
        config: Full Taiji OS config dict.

    Returns:
        Configured DeltaFusion instance.
    """
    dc = config.get("delta_mem", {})
    if not dc.get("enabled", True):
        return DeltaFusion(flush_enabled=False)

    rank = dc.get("rank", DEFAULT_RANK)
    layer = DeltaMemLayer(
        smatrix=SMatrix(
            S=np.zeros((rank, rank), dtype=np.float32),
            r=rank,
            lambda_=dc.get("lambda_decay", DEFAULT_LAMBDA),
            beta=dc.get("beta_update", 0.1),
        )
    )
    return DeltaFusion(
        delta_layer=layer,
        flush_threshold=dc.get("flush_threshold", 0.85),
        flush_enabled=dc.get("flush_enabled", True),
    )
