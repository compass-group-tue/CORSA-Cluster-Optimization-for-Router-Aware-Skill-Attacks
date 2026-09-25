"""Compose the unchanged SkillRouter retrieval stage with a native reranker."""
import hashlib
import json

import numpy as np

from .dense import _normalized
from .skillrouter import SkillRouterFull


class _ComposedClient:
    def __init__(self, retriever, reranker):
        self.retriever = retriever
        self.reranker = reranker
        self.encoder_path = retriever.encoder_path
        self.reranker_path = reranker.model_id
        self.candidate_ids = []
        self.candidate_hash = None

    def __getattr__(self, name):
        return getattr(self.retriever, name)

    def rerank(self, query, candidates):
        self.candidate_ids = [c["skill_id"] for c in candidates]
        self.candidate_hash = hashlib.sha256(json.dumps(self.candidate_ids, ensure_ascii=False).encode()).hexdigest()
        scores = np.asarray(self.reranker.rerank(query, candidates), dtype=np.float64)
        if scores.shape != (len(candidates),) or not np.isfinite(scores).all():
            raise ValueError("reranker must return exactly one finite score per candidate")
        return scores.tolist()


class RetrieveThenRerankRouter(SkillRouterFull):
    def __init__(self, retriever, reranker, *, retrieval_top_k=20, label="retrieve_then_rerank"):
        if retrieval_top_k < 1:
            raise ValueError("retrieval_top_k must be positive")
        self.retriever = retriever
        self.reranker = reranker
        super().__init__(client=_ComposedClient(retriever, reranker),
                         retrieval_top_k=retrieval_top_k, label=label)

    def provenance(self):
        result = super().provenance()
        method = getattr(self.reranker, "provenance", None)
        if callable(method):
            result.update(method())
        result["retrieval_implementation"] = "SkillRouterFull"
        return result

    @property
    def candidate_set_hash(self):
        return self.client.candidate_hash

    @property
    def candidate_ids(self):
        return list(self.client.candidate_ids)

    def top1_pool_skill(self, query):
        if self._pool_embs is None:
            raise AssertionError("call prepare_pool() first")
        return super().top1_pool_skill(query) if self._pool else None


class DenseRetrieveThenRerankRouter:
    """Top-k dense retrieval followed by a native reranker."""

    def __init__(self, embedder, reranker, *, retrieval_top_k=20,
                 label="dense_retrieve_then_rerank"):
        if retrieval_top_k < 1:
            raise ValueError("retrieval_top_k must be positive")
        self.embedder = embedder
        self.reranker = reranker
        self.retrieval_top_k = retrieval_top_k
        self.label = label
        self._pool = []
        self._pool_embs = None
        self._candidate_ids = []
        self._candidate_hash = None

    def provenance(self):
        result = {
            "router": self.label,
            "kind": "retrieve_then_rerank",
            "encoder_model": self.embedder.model_id,
            "reranker_model": self.reranker.model_id,
            "retrieval_top_k": self.retrieval_top_k,
            "retrieval_implementation": "dense_cosine",
        }
        for component in (self.embedder, self.reranker):
            method = getattr(component, "provenance", None)
            if callable(method):
                result.update(method())
        return result

    def prepare_pool(self, pool):
        self._pool = list(pool)
        if not self._pool:
            self._pool_embs = np.empty((0, 0), dtype=np.float32)
            return
        self._pool_embs = _normalized(self.embedder.embed_documents(self._pool))
        if len(self._pool_embs) != len(self._pool):
            raise ValueError("embedder returned the wrong number of pool vectors")

    def _query_and_ours(self, query, our_skill):
        q = _normalized(self.embedder.embed_query(query))[0]
        ours = _normalized(self.embedder.embed_documents([our_skill]))[0]
        if self._pool_embs.shape[1] not in (0, len(q)) or len(ours) != len(q):
            raise ValueError("query/document embedding dimensions do not match")
        return q, ours

    def retrieval_similarity(self, query, our_skill):
        assert self._pool_embs is not None, "call prepare_pool() first"
        q, ours = self._query_and_ours(query, our_skill)
        our_score = float(ours @ q)
        if not self._pool:
            return our_score
        pool_scores = self._pool_embs @ q
        k = min(self.retrieval_top_k, len(pool_scores))
        cutoff_index = len(pool_scores) - k
        cutoff = float(np.partition(pool_scores, cutoff_index)[cutoff_index])
        return our_score - cutoff

    def _rerank(self, query, candidates):
        self._candidate_ids = [candidate["skill_id"] for candidate in candidates]
        self._candidate_hash = hashlib.sha256(
            json.dumps(self._candidate_ids, ensure_ascii=False).encode()
        ).hexdigest()
        scores = np.asarray(self.reranker.rerank(query, candidates), dtype=np.float64)
        if scores.shape != (len(candidates),) or not np.isfinite(scores).all():
            raise ValueError("reranker must return exactly one finite score per candidate")
        return scores

    def rank(self, query, our_skill):
        assert self._pool_embs is not None, "call prepare_pool() first"
        q, ours = self._query_and_ours(query, our_skill)
        matrix = ours[None, :] if not self._pool else np.concatenate(
            [self._pool_embs, ours[None, :]], axis=0
        )
        order = np.argsort(-(matrix @ q), kind="stable")
        k = min(self.retrieval_top_k, len(order))
        selected = order[:k]
        attacker_index = len(matrix) - 1
        if attacker_index not in selected:
            return None
        all_skills = self._pool + [our_skill]
        candidates = [all_skills[int(index)] for index in selected]
        reranked = np.argsort(-self._rerank(query, candidates), kind="stable")
        for rank, candidate_index in enumerate(reranked):
            if candidates[int(candidate_index)]["skill_id"] == our_skill["skill_id"]:
                return rank
        return None

    def top1_pool_skill(self, query):
        assert self._pool_embs is not None, "call prepare_pool() first"
        if not self._pool:
            return None
        q = _normalized(self.embedder.embed_query(query))[0]
        order = np.argsort(-(self._pool_embs @ q), kind="stable")
        candidates = [self._pool[int(index)] for index in order[:self.retrieval_top_k]]
        reranked = np.argsort(-self._rerank(query, candidates), kind="stable")
        return candidates[int(reranked[0])]

    @property
    def candidate_set_hash(self):
        return self._candidate_hash

    @property
    def candidate_ids(self):
        return list(self._candidate_ids)
