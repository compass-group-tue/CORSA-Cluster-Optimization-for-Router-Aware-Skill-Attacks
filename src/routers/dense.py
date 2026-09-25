"""Shared one-stage dense router used by hosted embedding backends."""
from __future__ import annotations

from typing import Protocol

import numpy as np

from .base import Router


class Embedder(Protocol):
    model_id: str

    def embed_documents(self, skills: list[dict]) -> np.ndarray: ...
    def embed_query(self, query: str) -> np.ndarray: ...


def _normalized(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr[None, :]
    if not np.isfinite(arr).all():
        raise ValueError("embedder returned NaN or infinite values")
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    if np.any(norms == 0):
        raise ValueError("embedder returned a zero vector")
    return arr / norms


class DenseEmbeddingRouter(Router):
    label = "dense_embedding"

    def __init__(self, embedder: Embedder, label: str):
        self.embedder = embedder
        self.label = label
        self._pool: list[dict] = []
        self._pool_embs: np.ndarray | None = None

    def provenance(self) -> dict:
        data = {
            "router": self.label,
            "kind": "embedding_only",
            "encoder_model": self.embedder.model_id,
            "reranker_model": None,
        }
        extra = getattr(self.embedder, "provenance", None)
        if callable(extra):
            data.update(extra())
        return data

    def prepare_pool(self, pool: list[dict]) -> None:
        self._pool = list(pool)
        if not self._pool:
            self._pool_embs = np.empty((0, 0), dtype=np.float32)
            return
        self._pool_embs = _normalized(self.embedder.embed_documents(self._pool))
        if len(self._pool_embs) != len(self._pool):
            raise ValueError("embedder returned the wrong number of pool vectors")

    def _scores(self, query: str, our_skill: dict | None = None) -> np.ndarray:
        assert self._pool_embs is not None, "call prepare_pool() first"
        q = _normalized(self.embedder.embed_query(query))[0]
        matrix = self._pool_embs
        if our_skill is not None:
            ours = _normalized(self.embedder.embed_documents([our_skill]))
            if len(matrix) == 0:
                matrix = np.empty((0, ours.shape[1]), dtype=np.float32)
            if ours.shape[1] != matrix.shape[1]:
                raise ValueError("query/document embedding dimensions do not match")
            matrix = np.concatenate([matrix, ours], axis=0)
        if matrix.shape[1] != len(q):
            raise ValueError("query/document embedding dimensions do not match")
        return matrix @ q

    def rank(self, query: str, our_skill: dict) -> int | None:
        scores = self._scores(query, our_skill)
        # Stable sort makes ties deterministic: the inserted attacker skill is last.
        order = np.argsort(-scores, kind="stable")
        pos = np.flatnonzero(order == len(scores) - 1)
        return int(pos[0]) if len(pos) else None

    def retrieval_similarity(self, query: str, our_skill: dict) -> float:
        scores = self._scores(query, our_skill)
        return float(scores[-1] - np.max(scores[:-1], initial=-np.inf))

    def top1_pool_skill(self, query: str) -> dict | None:
        if not self._pool:
            return None
        scores = self._scores(query)
        return self._pool[int(np.argmax(scores))]
