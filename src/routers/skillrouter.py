"""Full SkillRouter: encoder retrieval followed by reranking.

Retrieves top-K candidates by cosine similarity and reranks using causal-LM
yes/no scores. Each rank() call inserts the candidate without re-embedding
the immutable competitive pool.
"""
from __future__ import annotations

import torch

from ._skillrouter_client import SkillRouterClient
from .base import Router


class SkillRouterFull(Router):
    label = "full_skillrouter"

    def __init__(
        self,
        client: SkillRouterClient | None = None,
        retrieval_top_k: int = 20,
        label: str | None = None,
    ):
        self.client = client or SkillRouterClient()
        self.retrieval_top_k = retrieval_top_k
        if label:
            self.label = label
        self._pool: list[dict] = []
        self._pool_embs: torch.Tensor | None = None

    def provenance(self) -> dict:
        return {
            "router": self.label,
            "kind": "retrieve_then_rerank",
            "encoder_model": self.client.encoder_path,
            "reranker_model": self.client.reranker_path,
            "retrieval_top_k": self.retrieval_top_k,
        }

    def prepare_pool(self, pool: list[dict]) -> None:
        self._pool = list(pool)
        texts = [self.client.format_skill_text(s) for s in pool]
        self._pool_embs = self.client.embed_texts(texts)

    def retrieval_similarity(self, query: str, our_skill: dict) -> float:
        """Return candidate similarity minus the K-th pool similarity.

        A positive margin places the candidate inside the encoder retrieval
        window. This continuous signal is defined even when rank() returns
        None, providing Stage-A feedback for candidates outside the window.
        """
        import torch.nn.functional as _F
        our_emb = _F.normalize(
            self.client.embed_texts([self.client.format_skill_text(our_skill)]), dim=-1)
        q_emb = _F.normalize(
            self.client.embed_texts([self.client.format_query_text(query)]), dim=-1)
        our_sim = float((q_emb @ our_emb.T).squeeze().item())
        if self._pool_embs is None:
            return our_sim
        pool_sims = (q_emb @ _F.normalize(self._pool_embs, dim=-1).T).squeeze(0)
        k = min(self.retrieval_top_k, pool_sims.shape[0])
        kth = float(torch.topk(pool_sims, k).values[-1].item())
        return our_sim - kth

    def top1_pool_skill(self, query: str) -> dict | None:
        """Return the highest-ranked pool skill without inserting a candidate."""
        assert self._pool_embs is not None, "call prepare_pool() first"
        q_emb = self.client.embed_texts([self.client.format_query_text(query)])
        sims = (q_emb @ self._pool_embs.T).squeeze(0)
        k = min(self.retrieval_top_k, sims.shape[0])
        _, idx = torch.topk(sims, k)
        cands = [self._pool[int(i)] for i in idx]
        scores = self.client.rerank(query, cands)
        ranked = sorted(zip(cands, scores), key=lambda x: x[1], reverse=True)
        return ranked[0][0] if ranked else None

    def ranked_pool(self, query: str, k: int = 10) -> list[str]:
        """Return up to k pool skill IDs after retrieval and reranking.

        No candidate is inserted. The encoder selects retrieval_top_k skills
        (default 20), and the reranker orders that window. Ground-truth skills
        outside the retrieval window cannot appear in the returned list.
        """
        assert self._pool_embs is not None, "call prepare_pool() first"
        q_emb = self.client.embed_texts([self.client.format_query_text(query)])
        sims = (q_emb @ self._pool_embs.T).squeeze(0)
        kk = min(self.retrieval_top_k, sims.shape[0])
        _, idx = torch.topk(sims, kk)
        cands = [self._pool[int(i)] for i in idx]
        scores = self.client.rerank(query, cands)
        ranked = sorted(zip(cands, scores), key=lambda x: x[1], reverse=True)
        return [c["skill_id"] for c, _ in ranked[:k]]

    def rank(self, query: str, our_skill: dict) -> int | None:
        assert self._pool_embs is not None, "call prepare_pool() first"
        our_emb = self.client.embed_texts([self.client.format_skill_text(our_skill)])
        q_emb = self.client.embed_texts([self.client.format_query_text(query)])
        all_embs = torch.cat([self._pool_embs, our_emb], dim=0)
        ids = [s["skill_id"] for s in self._pool] + [our_skill["skill_id"]]
        sims = (q_emb @ all_embs.T).squeeze(0)

        k = min(self.retrieval_top_k, sims.shape[0])
        _, idx = torch.topk(sims, k)
        top_ids = [ids[int(i)] for i in idx]
        # Rerank the retrieved top-K
        cand_map = {s["skill_id"]: s for s in self._pool}
        cand_map[our_skill["skill_id"]] = our_skill
        candidates = [cand_map[sid] for sid in top_ids]
        scores = self.client.rerank(query, candidates)
        ranked = sorted(zip(top_ids, scores), key=lambda x: x[1], reverse=True)
        for r, (sid, _) in enumerate(ranked):
            if sid == our_skill["skill_id"]:
                return r
        return None  # our_skill fell outside the top-K retrieval window
