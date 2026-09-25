"""Encoder-only router.

Uses the same encoder as full SkillRouter without reranking, enabling
comparison of cosine-similarity retrieval with the complete pipeline.
"""
from __future__ import annotations

import torch

from ._skillrouter_client import SkillRouterClient
from .base import Router


class EncoderOnlyRouter(Router):
    label = "encoder_only"

    def __init__(self, client: SkillRouterClient | None = None, label: str | None = None):
        self.client = client or SkillRouterClient(load_reranker=False)
        if label:
            self.label = label
        self._pool: list[dict] = []
        self._pool_embs: torch.Tensor | None = None

    def provenance(self) -> dict:
        return {
            "router": self.label,
            "kind": "embedding_only",
            "encoder_model": self.client.encoder_path,
            "reranker_model": None,
        }

    def retrieval_similarity(self, query: str, our_skill: dict) -> float:
        """Top-1 score margin used by retrieval-only GEPA Stage A."""
        import torch.nn.functional as _F
        assert self._pool_embs is not None, "call prepare_pool() first"
        our_emb = _F.normalize(
            self.client.embed_texts([self.client.format_skill_text(our_skill)]), dim=-1)
        q_emb = _F.normalize(
            self.client.embed_texts([self.client.format_query_text(query)]), dim=-1)
        pool_sims = (q_emb @ _F.normalize(self._pool_embs, dim=-1).T).squeeze(0)
        our_sim = float((q_emb @ our_emb.T).squeeze().item())
        return our_sim - float(pool_sims.max().item())

    def top1_pool_skill(self, query: str) -> dict | None:
        assert self._pool_embs is not None, "call prepare_pool() first"
        if not self._pool:
            return None
        q_emb = self.client.embed_texts([self.client.format_query_text(query)])
        scores = (q_emb @ self._pool_embs.T).squeeze(0)
        return self._pool[int(torch.argmax(scores).item())]

    def prepare_pool(self, pool: list[dict]) -> None:
        self._pool = list(pool)
        texts = [self.client.format_skill_text(s) for s in pool]
        self._pool_embs = self.client.embed_texts(texts)

    def rank(self, query: str, our_skill: dict) -> int | None:
        assert self._pool_embs is not None, "call prepare_pool() first"
        our_emb = self.client.embed_texts([self.client.format_skill_text(our_skill)])
        q_emb = self.client.embed_texts([self.client.format_query_text(query)])
        all_embs = torch.cat([self._pool_embs, our_emb], dim=0)
        ids = [s["skill_id"] for s in self._pool] + [our_skill["skill_id"]]
        sims = (q_emb @ all_embs.T).squeeze(0)
        order = torch.argsort(sims, descending=True).tolist()
        for r, i in enumerate(order):
            if ids[i] == our_skill["skill_id"]:
                return r
        return None
