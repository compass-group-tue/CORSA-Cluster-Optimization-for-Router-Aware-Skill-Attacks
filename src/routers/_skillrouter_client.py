"""Thin wrapper around SkillRouter's open-model pipeline.

Loads the two 0.6B HuggingFace checkpoints
  - pipizhao/SkillRouter-Embedding-0.6B  (encoder)
  - pipizhao/SkillRouter-Reranker-0.6B   (reranker)

and exposes:
  - embed_texts(texts) -> torch.Tensor
  - rerank(query, candidates) -> list[float]
  - format_skill_text(skill) -> str  (matches paper's flat-full format)
  - format_query_text(text) -> str

EncoderOnlyRouter and FullSkillRouter accept this client, allowing callers
to reuse loaded models when sharing an instance.
"""
from __future__ import annotations

import importlib.util
import torch

from ..infra.paths import SKILLROUTER_REPO


# SkillRouter's helpers live at <repo>/src/common.py. Our project ALSO has a
# top-level `src/` package, so `from src.common import ...` collides.
# Load their file by absolute path via importlib to avoid the collision.
def _load_skillrouter_common():
    common_path = SKILLROUTER_REPO / "src" / "common.py"
    spec = importlib.util.spec_from_file_location(
        "_skillrouter_common", str(common_path)
    )
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


_sr = None


def _common():
    """Load the external SkillRouter helpers only when a neural router is built."""
    global _sr
    if _sr is None:
        _sr = _load_skillrouter_common()
    return _sr


class SkillRouterClient:
    def __init__(
        self,
        encoder_path: str = "pipizhao/SkillRouter-Embedding-0.6B",
        reranker_path: str = "pipizhao/SkillRouter-Reranker-0.6B",
        encoder_max_length: int = 4096,
        reranker_max_length: int = 4096,
        encoder_batch_size: int = 32,
        reranker_batch_size: int = 8,
        load_reranker: bool = True,
    ):
        sr = _common()
        self.encoder_path = encoder_path
        self.reranker_path = reranker_path if load_reranker else None
        self.device = sr.get_device()
        self.encoder_max_length = encoder_max_length
        self.reranker_max_length = reranker_max_length
        self.encoder_batch_size = encoder_batch_size
        self.reranker_batch_size = reranker_batch_size

        self._sr = sr
        self.emb_model, self.emb_tok = sr.load_embedding_model(encoder_path)
        self.emb_model.to(self.device).eval()

        self.rr_model = self.rr_tok = None
        self._rr_prefix = self._rr_suffix = None
        self._rr_yes_id = self._rr_no_id = None
        if load_reranker:
            self.rr_model, self.rr_tok = sr.load_reranker_model(reranker_path)
            self.rr_model.to(self.device).eval()
            self._rr_prefix, self._rr_suffix = sr.get_reranker_template_tokens(self.rr_tok)
            self._rr_yes_id = self.rr_tok.convert_tokens_to_ids("yes")
            self._rr_no_id = self.rr_tok.convert_tokens_to_ids("no")

    def format_skill_text(self, skill: dict) -> str:
        return self._sr.format_skill(skill, desc_max=500, body_max=8000)

    def format_query_text(self, task_text: str) -> str:
        return self._sr.format_query(task_text, max_len=2000)

    def embed_texts(self, texts: list[str]) -> torch.Tensor:
        return self._sr.encode_texts(
            self.emb_model,
            self.emb_tok,
            texts,
            self.encoder_max_length,
            self.encoder_batch_size,
            self.device,
        )

    def rerank(
        self,
        query_text: str,
        candidates: list[dict],
        prompt_format: str = "flat-full",
    ) -> list[float]:
        if self.rr_model is None or self.rr_tok is None:
            raise RuntimeError("rerank() called on an encoder-only SkillRouterClient")
        texts = [
            self._sr.format_rerank_prompt(
                c["name"],
                c.get("description", "") or "",
                c.get("body", "") or "",
                query_text,
                prompt_format=prompt_format,
            )
            for c in candidates
        ]
        tokenized = [
            self._sr.tokenize_reranker_text(
                t, self.rr_tok, self._rr_prefix, self._rr_suffix, self.reranker_max_length
            )
            for t in texts
        ]
        scores: list[float] = []
        pad_id = self.rr_tok.pad_token_id if self.rr_tok.pad_token_id is not None else 0
        for i in range(0, len(tokenized), self.reranker_batch_size):
            batch = tokenized[i : i + self.reranker_batch_size]
            max_len = max(len(x) for x in batch)
            padded, masks = [], []
            for ids in batch:
                pad_len = max_len - len(ids)
                padded.append([pad_id] * pad_len + ids)
                masks.append([0] * pad_len + [1] * len(ids))
            input_ids = torch.tensor(padded, dtype=torch.long, device=self.device)
            mask = torch.tensor(masks, dtype=torch.long, device=self.device)
            with torch.no_grad():
                logits = self.rr_model(input_ids=input_ids, attention_mask=mask).logits[:, -1, :]
                s = (logits[:, self._rr_yes_id] - logits[:, self._rr_no_id]).float().cpu().tolist()
            scores.extend(s)
        return scores
