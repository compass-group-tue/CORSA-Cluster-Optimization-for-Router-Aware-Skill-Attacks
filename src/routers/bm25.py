"""BM25 sparse-retrieval router.

Provides a lexical baseline for comparison with semantic retrieval.
"""
from __future__ import annotations

import math
import re
from collections import Counter

from .base import Router


_WORD_RE = re.compile(r"[A-Za-z][A-Za-z0-9_-]{1,}")


def tokenize(text: str) -> list[str]:
    return [w.lower() for w in _WORD_RE.findall(text or "")]


def format_skill(skill: dict, desc_max: int = 500, body_max: int = 8000) -> str:
    name = skill.get("name", "") or ""
    desc = (skill.get("description") or "")[:desc_max]
    body = (skill.get("body") or "")[:body_max]
    return f"{name} | {desc} | {body}"


class _BM25Index:
    def __init__(self, docs: list[str], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        toks = [tokenize(d) for d in docs]
        self.doc_lens = [len(t) for t in toks]
        self.avgdl = (sum(self.doc_lens) / max(len(self.doc_lens), 1)) or 1.0
        df = Counter()
        for t in toks:
            for w in set(t):
                df[w] += 1
        self.N = len(docs)
        self.idf = {
            w: math.log(1 + (self.N - c + 0.5) / (c + 0.5))
            for w, c in df.items()
        }
        self.tf = [Counter(t) for t in toks]

    def score(self, q: list[str]) -> list[float]:
        out = [0.0] * self.N
        for i in range(self.N):
            tf, dl = self.tf[i], self.doc_lens[i]
            s = 0.0
            for w in q:
                if w not in tf:
                    continue
                f = tf[w]
                idf = self.idf.get(w, 0.0)
                denom = f + self.k1 * (1 - self.b + self.b * dl / self.avgdl)
                s += idf * (f * (self.k1 + 1) / denom)
            out[i] = s
        return out


class BM25Router(Router):
    label = "bm25"

    def __init__(self):
        self._pool: list[dict] = []

    def prepare_pool(self, pool: list[dict]) -> None:
        self._pool = list(pool)

    def provenance(self) -> dict:
        return {
            "router": self.label,
            "kind": "sparse_retrieval",
            "encoder_model": "bm25(k1=1.5,b=0.75)",
            "reranker_model": None,
        }

    def retrieval_similarity(self, query: str, our_skill: dict) -> float:
        """BM25 score margin over the best benign skill for GEPA Stage A."""
        docs = [format_skill(s) for s in self._pool] + [format_skill(our_skill)]
        scores = _BM25Index(docs).score(tokenize(query))
        our_score = scores[-1]
        best_pool = max(scores[:-1], default=0.0)
        return float(our_score - best_pool)

    def top1_pool_skill(self, query: str) -> dict | None:
        if not self._pool:
            return None
        scores = _BM25Index([format_skill(s) for s in self._pool]).score(tokenize(query))
        best = max(range(len(scores)), key=lambda i: scores[i])
        return self._pool[best]

    def rank(self, query: str, our_skill: dict) -> int | None:
        # Rebuild the index because the candidate changes across GEPA rounds.
        docs = [format_skill(s) for s in self._pool] + [format_skill(our_skill)]
        ids = [s["skill_id"] for s in self._pool] + [our_skill["skill_id"]]
        idx = _BM25Index(docs)
        scores = idx.score(tokenize(query))
        order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        for r, i in enumerate(order):
            if ids[i] == our_skill["skill_id"]:
                return r
        return None
