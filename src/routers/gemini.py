"""Gemini Embedding 2 adapter for one-stage skill routing.

Gemini Embedding 2 aggregates multiple inputs into one vector, so this adapter
intentionally sends one skill per request and caches every vector by content
hash. A shared cache makes repeated GEPA jobs reuse the immutable benign pool.
"""
from __future__ import annotations

import hashlib
import os
import sqlite3
import time
from pathlib import Path
from src.infra.paths import ROUTER_CACHE_DIR

import numpy as np


class _SQLiteEmbeddingCache:
    def __init__(self, path: Path):
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as db:
            db.execute(
                "CREATE TABLE IF NOT EXISTS embeddings "
                "(cache_key TEXT PRIMARY KEY, dim INTEGER NOT NULL, vector BLOB NOT NULL)"
            )

    def _connect(self):
        return sqlite3.connect(self.path, timeout=120)

    def get(self, key: str) -> np.ndarray | None:
        with self._connect() as db:
            row = db.execute(
                "SELECT dim, vector FROM embeddings WHERE cache_key = ?", (key,)
            ).fetchone()
        if row is None:
            return None
        dim, raw = row
        value = np.frombuffer(raw, dtype=np.float32).copy()
        if len(value) != dim:
            raise ValueError(f"corrupt embedding cache entry {key}")
        return value

    def put(self, key: str, value: np.ndarray) -> None:
        arr = np.asarray(value, dtype=np.float32)
        with self._connect() as db:
            db.execute(
                "INSERT OR IGNORE INTO embeddings(cache_key, dim, vector) VALUES (?, ?, ?)",
                (key, len(arr), arr.tobytes()),
            )


class GeminiEmbedding2Embedder:
    """Text retrieval formatting recommended for Gemini Embedding 2."""

    def __init__(
        self,
        model_id: str = "gemini-embedding-2",
        output_dimensionality: int = 768,
        cache_dir: str | Path | None = None,
        max_attempts: int = 6,
    ):
        if not 128 <= output_dimensionality <= 3072:
            raise ValueError("Gemini output dimensionality must be in [128, 3072]")
        self.model_id = model_id
        self.output_dimensionality = output_dimensionality
        cache_root = Path(cache_dir or os.environ.get(
            "ROUTER_CACHE_DIR", ROUTER_CACHE_DIR
        ))
        self.cache = _SQLiteEmbeddingCache(
            cache_root / f"{model_id.replace('/', '__')}_{output_dimensionality}.sqlite3"
        )
        self.max_attempts = max_attempts
        self._client = None

    def provenance(self) -> dict:
        return {
            "embedding_dimension": self.output_dimensionality,
            "query_format": "task: search result | query: {query}",
            "document_format": "title: {name} | text: {description} | {body}",
            "cache": str(self.cache.path),
        }

    @staticmethod
    def _document_text(skill: dict) -> str:
        name = (skill.get("name") or "")
        description = (skill.get("description") or "")[:500]
        body = (skill.get("body") or "")[:8000]
        return f"title: {name or 'none'} | text: {description} | {body}"

    @staticmethod
    def _query_text(query: str) -> str:
        return f"task: search result | query: {(query or '')[:2000]}"

    def _cache_key(self, role: str, text: str) -> str:
        raw = f"v1\0{self.model_id}\0{self.output_dimensionality}\0{role}\0{text}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def _get_client(self):
        if self._client is None:
            try:
                from google import genai
            except ImportError as exc:
                raise RuntimeError(
                    "Gemini routing requires `pip install google-genai`"
                ) from exc
            if not (os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")):
                raise RuntimeError(
                    "Gemini routing requires GEMINI_API_KEY or GOOGLE_API_KEY"
                )
            self._client = genai.Client()
        return self._client

    def _embed(self, role: str, text: str) -> np.ndarray:
        key = self._cache_key(role, text)
        cached = self.cache.get(key)
        if cached is not None:
            return cached

        # Several cluster jobs may prepare the same immutable pool concurrently.
        # Serialize cache misses to avoid duplicate requests across jobs.
        import fcntl
        lock_path = self.cache.path.with_suffix(self.cache.path.suffix + ".lock")
        with lock_path.open("a+b") as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            cached = self.cache.get(key)
            if cached is not None:
                return cached
            try:
                from google.genai import types
            except ImportError as exc:
                raise RuntimeError(
                    "Gemini routing requires `pip install google-genai`"
                ) from exc

            last_error = None
            for attempt in range(self.max_attempts):
                try:
                    result = self._get_client().models.embed_content(
                        model=self.model_id,
                        contents=text,
                        config=types.EmbedContentConfig(
                            output_dimensionality=self.output_dimensionality
                        ),
                    )
                    if len(result.embeddings or []) != 1:
                        raise ValueError("Gemini returned an unexpected embedding count")
                    value = np.asarray(result.embeddings[0].values, dtype=np.float32)
                    if len(value) != self.output_dimensionality:
                        raise ValueError(
                            f"Gemini returned dimension {len(value)}, expected "
                            f"{self.output_dimensionality}"
                        )
                    self.cache.put(key, value)
                    return value
                except Exception as exc:  # SDK errors vary by transport/version
                    last_error = exc
                    if attempt + 1 == self.max_attempts:
                        break
                    time.sleep(min(2 ** attempt, 30))
            raise RuntimeError(f"Gemini embedding failed after retries: {last_error}")

    def embed_documents(self, skills: list[dict]) -> np.ndarray:
        return np.stack([
            self._embed("document", self._document_text(skill)) for skill in skills
        ])

    def embed_query(self, query: str) -> np.ndarray:
        return self._embed("query", self._query_text(query))
