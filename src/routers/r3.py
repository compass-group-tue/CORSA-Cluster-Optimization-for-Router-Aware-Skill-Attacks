"""Official R3-Skill two-stage router (R3-Embedding + R3-Reranker)."""
from __future__ import annotations

import os

import numpy as np

from ._embedding_utils import resolve_device
from .skillrouter import SkillRouterFull


R3_ENCODER = "tencent/R3-embedding-0.6b"
R3_RERANKER = "tencent/R3-rerank-0.6b"
R3_EMBEDDING_INSTRUCTION = (
    "Instruct: Given a user request, retrieve the agent skill that solves it.\n"
    "Query: "
)
R3_RERANKER_INSTRUCTION = (
    "Given a user request, retrieve the agent skill that solves it."
)


def _skill_text(skill: dict) -> str:
    """Serialize skills exactly as recommended by the released R3 models."""
    return (
        f"{skill.get('name') or ''} | "
        f"{skill.get('description') or ''} | "
        f"{skill.get('body') or ''}"
    )


class R3SkillClient:
    """Duck-typed client consumed by the pipeline's two-stage router."""

    def __init__(
        self,
        encoder_path: str = R3_ENCODER,
        reranker_path: str = R3_RERANKER,
        *,
        device: str | None = None,
        encoder_max_length: int = 4096,
        reranker_max_length: int = 4096,
        body_max_tokens: int = 4096,
        encoder_batch_size: int | None = None,
        reranker_batch_size: int | None = None,
        revision: str | None = None,
    ):
        if encoder_max_length < 1 or reranker_max_length < 1 or body_max_tokens < 1:
            raise ValueError("R3 token limits must be positive")
        self.encoder_path = encoder_path
        self.reranker_path = reranker_path
        self.device = resolve_device(device)
        self.encoder_max_length = encoder_max_length
        self.reranker_max_length = reranker_max_length
        self.body_max_tokens = body_max_tokens
        default_batch = int(os.environ.get("ROUTER_BATCH_SIZE") or "8")
        self.encoder_batch_size = encoder_batch_size or default_batch
        self.reranker_batch_size = reranker_batch_size or default_batch
        if self.encoder_batch_size < 1 or self.reranker_batch_size < 1:
            raise ValueError("R3 batch sizes must be positive")
        self.revision = revision or os.environ.get("ROUTER_MODEL_REVISION") or "main"
        self.embedding_model = None
        self.reranker_model = None

    def _load_embedding(self) -> None:
        if self.embedding_model is not None:
            return
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError(
                "R3 routing requires sentence-transformers>=5.3"
            ) from exc
        self.embedding_model = SentenceTransformer(
            self.encoder_path,
            trust_remote_code=True,
            revision=self.revision,
            device=self.device,
        )
        self.embedding_model.max_seq_length = self.encoder_max_length

    def _load_reranker(self) -> None:
        if self.reranker_model is not None:
            return
        try:
            from sentence_transformers import CrossEncoder
        except ImportError as exc:
            raise RuntimeError(
                "R3 routing requires sentence-transformers>=5.3"
            ) from exc
        self.reranker_model = CrossEncoder(
            self.reranker_path,
            trust_remote_code=True,
            revision=self.revision,
            device=self.device,
        )
        self.reranker_model.max_length = self.reranker_max_length

    def format_skill_text(self, skill: dict) -> str:
        return _skill_text(skill)

    def format_query_text(self, task_text: str) -> str:
        return R3_EMBEDDING_INSTRUCTION + (task_text or "")

    def embed_texts(self, texts: list[str]):
        self._load_embedding()
        return self.embedding_model.encode(
            texts,
            batch_size=self.encoder_batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_tensor=True,
        )

    def _truncate_body(self, text: str) -> str:
        parts = text.split(" | ", 2)
        if len(parts) < 3:
            return text
        token_ids = self.reranker_model.tokenizer.encode(
            parts[2], add_special_tokens=False
        )
        if len(token_ids) <= self.body_max_tokens:
            return text
        body = self.reranker_model.tokenizer.decode(
            token_ids[: self.body_max_tokens], skip_special_tokens=True
        )
        return f"{parts[0]} | {parts[1]} | {body}"

    def rerank(
        self,
        query_text: str,
        candidates: list[dict],
        prompt_format: str = "flat-full",
    ) -> list[float]:
        del prompt_format  # R3 has one released, training-consistent format.
        if not candidates:
            return []
        self._load_reranker()
        pairs = [
            (query_text, self._truncate_body(self.format_skill_text(candidate)))
            for candidate in candidates
        ]
        scores = self.reranker_model.predict(
            pairs,
            batch_size=self.reranker_batch_size,
            prompt=R3_RERANKER_INSTRUCTION,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        return np.asarray(scores, dtype=np.float32).reshape(-1).tolist()


class R3SkillRouter(SkillRouterFull):
    """R3's released bi-encoder recall and cross-encoder reranking pipeline."""

    def provenance(self) -> dict:
        result = super().provenance()
        result.update(
            {
                "paper": "arXiv:2606.03565",
                "implementation": "sentence-transformers",
                "encoder_revision": self.client.revision,
                "embedding_instruction": R3_EMBEDDING_INSTRUCTION,
                "reranker_instruction": R3_RERANKER_INSTRUCTION,
                "skill_format": "name | description | body",
                "encoder_max_length": self.client.encoder_max_length,
                "reranker_max_length": self.client.reranker_max_length,
                "body_max_tokens": self.client.body_max_tokens,
            }
        )
        return result
