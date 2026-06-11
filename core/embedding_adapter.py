"""
embedding_adapter.py — Bridge between WorldModel embeddings (semantic/API/hash)
and δ-mem S-matrix (8-dim) for real LLM integration.

Converts WorldModel.encode() outputs into δ-mem k/q/v vectors,
and translates δ-mem residual signals into injectable prompt context.

v1.1 — Dynamic embedding dimension detection from WorldModel.
        Supports: MiniLM (384-dim), DeepSeek API (1536-dim), hash (any).

Author: Taiji OS Team
Version: v1.1 — 2026-06-11
"""

from __future__ import annotations

import numpy as np

from core.delta_mem import project_to_srank, DEFAULT_RANK


# Cached embedding dimension; auto-detected from WorldModel
_EMBEDDING_DIM: int = 384  # Default to MiniLM dim


def get_embedding_dim() -> int:
    """Get the current embedding dimension (auto-detected or default)."""
    return _EMBEDDING_DIM


def set_embedding_dim(dim: int) -> None:
    """Override the embedding dimension.

    Args:
        dim: New embedding dimension (e.g., 384 for MiniLM, 1536 for DeepSeek).
    """
    global _EMBEDDING_DIM
    _EMBEDDING_DIM = dim


def auto_detect_dim(wm) -> int:
    """Auto-detect embedding dimension from a WorldModel instance.

    Args:
        wm: WorldModel instance with possible embedding_dim property.

    Returns:
        Detected dimension (384, 1536, or fallback).
    """
    global _EMBEDDING_DIM
    if hasattr(wm, "embedding_dim"):
        _EMBEDDING_DIM = wm.embedding_dim
    return _EMBEDDING_DIM


def embed_to_key(text: str, wm, target_rank: int = DEFAULT_RANK) -> np.ndarray:
    """Encode prompt text as a δ-mem key vector (8-dim).

    Uses WorldModel.encode() for semantic embedding, then projects
    down to S-matrix rank via deterministic project_to_srank().

    Args:
        text: The text to encode as a key (typically the prompt).
        wm: WorldModel instance with encode(text) method.
        target_rank: S-matrix rank (default 8).

    Returns:
        numpy array of shape (target_rank,), float32.
    """
    vec = wm.encode(text)
    return project_to_srank(vec, target_rank)


def embed_to_value(text: str, wm, target_rank: int = DEFAULT_RANK) -> np.ndarray:
    """Encode response text as a δ-mem value vector (8-dim).

    Same encoding path as embed_to_key, but conceptually used for the
    value side of the (k, v) association pair. Could use a different
    projection strategy in production (e.g. separate learned layer).

    Args:
        text: The text to encode as a value (typically the LLM response).
        wm: WorldModel instance.
        target_rank: S-matrix rank (default 8).

    Returns:
        numpy array of shape (target_rank,), float32.
    """
    vec = wm.encode(text)
    return project_to_srank(vec, target_rank)


def embed_to_query(text: str, wm, target_rank: int = DEFAULT_RANK) -> np.ndarray:
    """Encode query text as a δ-mem query vector (8-dim).

    Used to read residual signals from the S matrix before an LLM call.
    Unlike key/value encoding, this includes a small noise injection
    to break symmetry when query == key.

    Args:
        text: The text to encode as a query.
        wm: WorldModel instance.
        target_rank: S-matrix rank (default 8).

    Returns:
        numpy array of shape (target_rank,), float32.
    """
    vec = wm.encode(text)
    q = project_to_srank(vec, target_rank)
    # Tiny noise to break degeneracy when q == k
    noise = np.random.default_rng(
        abs(hash(text[:50])) % (2**31)
    ).normal(0, 1e-4, target_rank).astype(np.float32)
    return q + noise


def residual_to_context(delta_vec: np.ndarray, threshold: float = 0.05) -> str:
    """Translate a δ-mem residual vector into an injectable natural language
    context string for the prompt.

    The residual signal encodes patterns from prior interactions that
    are relevant to the current query. We translate it into a compact
    hint that the LLM can use without overwhelming the prompt.

    Args:
        delta_vec: δ-mem residual vector (rank-dim), e.g. S @ q.
        threshold: Minimum norm to generate a context string.

    Returns:
        Natural language context string, or empty string if below threshold.
    """
    norm = float(np.linalg.norm(delta_vec))

    if norm < threshold:
        return ""

    # Extract dominant signals from the residual
    # Each dimension contributes a directional hint
    dims = len(delta_vec)
    # Find the strongest 3 dimensions
    abs_vals = np.abs(delta_vec)
    top_indices = np.argsort(abs_vals)[-3:][::-1]

    hints = []
    for idx in top_indices:
        val = delta_vec[idx]
        strength = abs(val)
        direction = "positive" if val > 0 else "negative"

        if strength > 0.5:
            weight = "strong"
        elif strength > 0.2:
            weight = "moderate"
        else:
            weight = "subtle"

        hints.append(f"channel_{idx}: {weight} {direction} signal")

    # Build context string
    overall = (
        "strong prior knowledge indicates relevance"
        if norm > 1.0
        else "some prior context may be relevant"
    )

    return f"{overall} ({', '.join(hints)})"


def delta_to_attention_hint(
    attention_delta: np.ndarray, threshold: float = 0.01
) -> str:
    """Translate an attention correction Δ into a prompt hint for the LLM.

    Used for D-Core detection: when prior patterns suggest the candidate
    should be scrutinized more carefully (high attention delta), we hint
    the LLM to be more critical.

    Args:
        attention_delta: δ-mem attention correction Δ = (S·q) * clamp(k^T·q).
        threshold: Minimum norm to generate a hint.

    Returns:
        Hint string for injection into the detection prompt.
    """
    norm = float(np.linalg.norm(attention_delta))

    if norm < threshold:
        return ""

    if norm > 0.5:
        return (
            "prior memory strongly suggests verifying consistency "
            "with recent interactions"
        )
    elif norm > 0.1:
        return "prior memory suggests checking against earlier context"
    else:
        return "minor prior signal detected; no urgent action needed"
