"""
faiss_episodic.py — FAISS Vector Index for Episodic Memory (L2 Cold Storage).

Replaces the O(n) linear scan over JSON-serialized episodic entries with
a FAISS IVFFlat / IndexFlatIP index for efficient cosine similarity search.

Fallback: when faiss is not installed, falls back to a numpy-based linear
scan with equivalent semantics (graceful degradation, not a crash).

Key operations:
  - add(entry): Insert an S_flushed vector into the index.
  - search(query_vec, k): Return top-k similar entries by cosine similarity.
  - rebuild_index(): Train IVF centroids after sufficient entries accumulate.
  - save_index/load_index: Persist FAISS index to disk.

Author: Taiji OS Team
Version: v1.0 — 2026-06-11
"""

from __future__ import annotations

import hashlib
import os
import struct
from pathlib import Path
from typing import Optional

import numpy as np

# ── Optional FAISS import ─────────────────────────────────────────────────

FAISS_AVAILABLE = False
try:
    import faiss
    FAISS_AVAILABLE = True
except Exception:
    # faiss may fail for various reasons: not installed, numpy version
    # mismatch (e.g. numpy 2.x with faiss compiled for numpy 1.x), etc.
    # Graceful fallback: numpy-based linear scan.
    pass


# ── Episodic Memory Entry (re-import compatible) ──────────────────────────


class FAISSEpisodicIndex:
    """FAISS-backed episodic memory index for fast vector similarity search.

    Maintains a FAISS inner-product index over flattened S matrices
    (8x8 = 64-dim) for efficient nearest-neighbor retrieval during
    session resume / re-anchor operations.

    Attributes:
        dim: Vector dimension (rank * rank = 64 for r=8).
        index_type: FAISS index type ("FlatIP", "IVFFlat", "IVFPQ").
        nlist: Number of IVF clusters (IVFFlat/IVFPQ only).
        entries: List of EpisodicMemoryEntry metadata (parallel to index).
        index: The FAISS index object (or None if fallback mode).
        trained: Whether IVF centroids have been trained.
        _pending_vectors: Accumulated vectors before index training.
    """

    def __init__(
        self,
        dim: int = 64,
        index_type: str = "FlatIP",
        nlist: int = 10,
    ):
        self.dim = dim
        self.index_type = index_type
        self.nlist = nlist
        self.entries: list = []  # EpisodicMemoryEntry objects
        self.index = None
        self.trained = False
        self._pending_vectors: list = []  # Accumulate for IVF training

        if FAISS_AVAILABLE:
            self._init_faiss_index()
        else:
            self.index = None  # Fallback to numpy linear scan

    def _init_faiss_index(self) -> None:
        """Initialize the FAISS index based on index_type."""
        if self.index_type == "FlatIP":
            self.index = faiss.IndexFlatIP(self.dim)
            self.trained = True  # FlatIP needs no training

        elif self.index_type == "IVFFlat":
            quantizer = faiss.IndexFlatIP(self.dim)
            self.index = faiss.IndexIVFFlat(quantizer, self.dim, self.nlist)
            self.trained = False

        elif self.index_type == "IVFPQ":
            quantizer = faiss.IndexFlatIP(self.dim)
            # IVFPQ with M=8 sub-quantizers, 8 bits each
            self.index = faiss.IndexIVFPQ(
                quantizer, self.dim, self.nlist, 8, 8
            )
            self.trained = False

        else:
            self.index = faiss.IndexFlatIP(self.dim)
            self.trained = True

    # ── Core operations ────────────────────────────────────────────────

    def add(self, entry) -> None:
        """Add an episodic memory entry to the index.

        Args:
            entry: EpisodicMemoryEntry with S_flushed matrix (r×r).
        """
        # Flatten S matrix to 64-dim vector
        vec = np.asarray(entry.S_flushed, dtype=np.float32).ravel()[:self.dim]
        # Pad if too short
        if len(vec) < self.dim:
            vec = np.pad(vec, (0, self.dim - len(vec)))

        self.entries.append(entry)

        if FAISS_AVAILABLE and self.index is not None:
            if self.trained:
                self.index.add(vec.reshape(1, -1))
            else:
                # Accumulate for IVF training
                self._pending_vectors.append(vec)
                # Train when we have enough samples (need at least nlist vectors)
                if len(self._pending_vectors) >= max(self.nlist * 4, 20):
                    self._train_and_add()

    def _train_and_add(self) -> None:
        """Train IVF centroids and add all pending vectors to the index."""
        if not FAISS_AVAILABLE or self.index is None or self.trained:
            return

        vectors = np.array(self._pending_vectors, dtype=np.float32)
        if len(vectors) < self.nlist:
            return  # Not enough vectors to train

        try:
            self.index.train(vectors)
            self.trained = True
            # Add all pending vectors
            self.index.add(vectors)
            self._pending_vectors.clear()

        except Exception:
            # Training failed → fall back to FlatIP
            self.index = faiss.IndexFlatIP(self.dim)
            self.trained = True
            self.index.add(vectors)
            self._pending_vectors.clear()

    def rebuild_index(self) -> None:
        """Force rebuild of the FAISS index from all current entries.

        Useful after deserialization to reconstruct the search index.
        """
        if not FAISS_AVAILABLE:
            return

        self._init_faiss_index()

        all_vectors = []
        for entry in self.entries:
            vec = np.asarray(entry.S_flushed, dtype=np.float32).ravel()[:self.dim]
            if len(vec) < self.dim:
                vec = np.pad(vec, (0, self.dim - len(vec)))
            all_vectors.append(vec)

        if not all_vectors:
            return

        vectors = np.array(all_vectors, dtype=np.float32)

        if self.index_type != "FlatIP" and not self.trained:
            if len(vectors) >= self.nlist:
                try:
                    self.index.train(vectors)
                    self.trained = True
                except Exception:
                    self.index = faiss.IndexFlatIP(self.dim)
                    self.trained = True

        self.index.add(vectors)

    def search(
        self, query_vec: np.ndarray, k: int = 3
    ) -> list[tuple[float, any]]:
        """Search for top-k similar entries by cosine similarity.

        Args:
            query_vec: Query vector (same dim as index).
            k: Number of results to return.

        Returns:
            List of (score, entry) tuples, sorted by descending similarity.
        """
        query = np.asarray(query_vec, dtype=np.float32).ravel()[:self.dim]

        if FAISS_AVAILABLE and self.index is not None and self.index.ntotal > 0:
            # FAISS path: FlatIP returns inner product, normalize to cosine-like score
            actual_k = min(k, self.index.ntotal)
            distances, indices = self.index.search(
                query.reshape(1, -1), actual_k
            )
            query_norm = float(np.linalg.norm(query))
            results = []
            for dist, idx in zip(distances[0], indices[0]):
                if 0 <= idx < len(self.entries):
                    # Normalize inner product to cosine-like score in [0, 1]
                    vec = np.asarray(
                        self.entries[idx].S_flushed, dtype=np.float32
                    ).ravel()[:self.dim]
                    vec_norm = float(np.linalg.norm(vec))
                    if query_norm > 1e-8 and vec_norm > 1e-8:
                        score = max(0.0, min(1.0, float(dist) / (query_norm * vec_norm)))
                    else:
                        score = 0.0
                    results.append((score, self.entries[idx]))
            return results

        else:
            # NumPy fallback: linear scan
            return self._numpy_search(query, k)

    def _numpy_search(
        self, query: np.ndarray, k: int
    ) -> list[tuple[float, any]]:
        """Linear scan fallback using numpy."""
        if not self.entries:
            return []

        query_norm = np.linalg.norm(query)
        if query_norm < 1e-8:
            return [(0.0, self.entries[0])] if self.entries else []

        scores = []
        for entry in self.entries:
            vec = np.asarray(entry.S_flushed, dtype=np.float32).ravel()[:self.dim]
            if len(vec) < self.dim:
                vec = np.pad(vec, (0, self.dim - len(vec)))
            vec_norm = np.linalg.norm(vec)
            if vec_norm < 1e-8:
                scores.append(0.0)
            else:
                score = float(np.dot(query, vec) / (query_norm * vec_norm))
                scores.append(max(0.0, min(1.0, score)))

        # Sort by score descending
        indices = np.argsort(scores)[::-1][:k]
        results = []
        for idx in indices:
            results.append((scores[idx], self.entries[idx]))
        return results

    # ── Serialization ──────────────────────────────────────────────────

    def save_index(self, path: str) -> None:
        """Save the FAISS index to a binary file.

        Args:
            path: File path for the index (e.g., 'episodic.faiss').
        """
        if FAISS_AVAILABLE and self.index is not None and self.index.ntotal > 0:
            # Flush pending vectors first
            if self._pending_vectors:
                self._train_and_add()
            faiss.write_index(self.index, path)

    def load_index(self, path: str) -> None:
        """Load a FAISS index from a binary file.

        Args:
            path: File path for the index.
        """
        if FAISS_AVAILABLE and os.path.exists(path):
            self.index = faiss.read_index(path)
            self.trained = True

    def to_dict(self) -> dict:
        """Serialize entry metadata (not FAISS index) for JSON persistence.

        The FAISS binary index is saved separately via save_index/load_index.
        """
        return {
            "entries": [
                e.to_dict() if hasattr(e, "to_dict") else e
                for e in self.entries
            ],
            "dim": self.dim,
            "index_type": self.index_type,
            "nlist": self.nlist,
            "faiss_available": FAISS_AVAILABLE,
            "ntotal": self.index.ntotal if (FAISS_AVAILABLE and self.index) else len(self.entries),
        }

    @classmethod
    def from_dict(cls, data: dict, entries_class=None) -> "FAISSEpisodicIndex":
        """Deserialize from a dict, optionally reconstructing entries.

        Args:
            data: Dict from to_dict().
            entries_class: Class with from_dict() to reconstruct entries
                           (e.g. EpisodicMemoryEntry).

        Returns:
            Reconstructed FAISSEpisodicIndex.
        """
        idx = cls(
            dim=data.get("dim", 64),
            index_type=data.get("index_type", "FlatIP"),
            nlist=data.get("nlist", 10),
        )

        if "entries" in data and entries_class is not None:
            for e_data in data["entries"]:
                if isinstance(e_data, dict) and hasattr(entries_class, "from_dict"):
                    idx.entries.append(entries_class.from_dict(e_data))

        # Rebuild FAISS index from loaded entries
        if FAISS_AVAILABLE and idx.entries:
            idx.rebuild_index()

        return idx

    # ── Utilities ──────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self.entries)

    def __bool__(self) -> bool:
        return len(self.entries) > 0
