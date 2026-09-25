"""Shared semantic fields and persistent content-addressed embedding cache."""
import hashlib
import json
import os
from pathlib import Path
from src.infra.paths import ROUTER_CACHE_DIR

import numpy as np

from .gemini import _SQLiteEmbeddingCache

QUERY_INSTRUCTION = "Given a user task, retrieve the skill document that is most useful for completing it."


def document_text(skill):
    return f"{skill.get('name') or ''} | {(skill.get('description') or '')[:500]} | {(skill.get('body') or '')[:8000]}"


def resolve_device(device=None):
    value = device or os.environ.get("ROUTER_DEVICE")
    return None if not value or value == "auto" else value


def resolve_torch_dtype(torch, dtype=None):
    """Resolve the explicitly configured inference dtype without loading torch early."""
    value = (dtype or os.environ.get("ROUTER_TORCH_DTYPE") or "float32").lower()
    choices = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    if value not in choices:
        raise ValueError(
            "ROUTER_TORCH_DTYPE must be float32, float16, or bfloat16"
        )
    return choices[value]


def validate_vectors(values, count, dimension=None):
    values = np.asarray(values, dtype=np.float32)
    if values.ndim != 2 or values.shape[0] != count or values.shape[1] == 0:
        raise ValueError("embedder returned an invalid vector shape")
    if dimension is not None and values.shape[1] != dimension:
        raise ValueError(f"expected embedding dimension {dimension}, got {values.shape[1]}")
    if not np.isfinite(values).all() or np.any(np.linalg.norm(values, axis=1) == 0):
        raise ValueError("embedder returned non-finite or zero vectors")
    return values


class CachedEmbedder:
    def _init_cache(self, cache_dir, identity, batch_size):
        if batch_size < 1:
            raise ValueError("embedding batch size must be positive")
        self.batch_size = batch_size
        self._identity = json.dumps(identity, sort_keys=True)
        root = Path(cache_dir or os.environ.get("ROUTER_CACHE_DIR", ROUTER_CACHE_DIR))
        self.cache = _SQLiteEmbeddingCache(root / "native_embeddings.sqlite3")

    def _embed_cached(self, texts, role):
        if not texts:
            return np.empty((0, self.dimension or 0), dtype=np.float32)
        keys = [hashlib.sha256(f"v1\0{self._identity}\0{role}\0{text}".encode()).hexdigest() for text in texts]
        found = {key: self.cache.get(key) for key in dict.fromkeys(keys)}
        missing = list({key: text for key, text in zip(keys, texts) if found[key] is None}.items())
        for start in range(0, len(missing), self.batch_size):
            batch = missing[start:start + self.batch_size]
            vectors = validate_vectors(self._encode([text for _, text in batch], role), len(batch), self.dimension)
            for (key, _), vector in zip(batch, vectors):
                self.cache.put(key, vector)
                found[key] = vector
        return validate_vectors(np.stack([found[key] for key in keys]), len(keys), self.dimension)

    def embed_documents(self, skills):
        return self._embed_cached([document_text(skill) for skill in skills], "document")

    def embed_query(self, query):
        return self._embed_cached([query or ""], "query")[0]
