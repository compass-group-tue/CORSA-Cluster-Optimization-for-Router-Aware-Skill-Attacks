"""Qwen causal yes/no scoring and BGE cross-encoder scoring."""
import os
from pathlib import Path

import numpy as np

from ._embedding_utils import (
    QUERY_INSTRUCTION,
    document_text,
    resolve_device,
    resolve_torch_dtype,
)


class Qwen3Reranker:
    PREFIX = '<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be "yes" or "no".<|im_end|>\n<|im_start|>user\n'
    SUFFIX = '<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n'

    def __init__(self, model_id="Qwen/Qwen3-Reranker-0.6B", *, revision=None,
                 batch_size=None, device=None, max_length=4096, dtype=None):
        self.model_id = model_id
        self.revision = (
            revision
            or os.environ.get("ROUTER_RERANKER_REVISION")
            or os.environ.get("ROUTER_MODEL_REVISION")
            or "main"
        )
        self.batch_size = batch_size if batch_size is not None else int(os.environ.get("ROUTER_BATCH_SIZE") or "8")
        if self.batch_size < 1:
            raise ValueError("reranker batch size must be positive")
        self.device = resolve_device(device)
        self.max_length = max_length
        self.dtype = dtype or os.environ.get("ROUTER_TORCH_DTYPE") or "float32"
        self.model = self.tokenizer = None

    def provenance(self):
        return {"reranker_revision": self.revision, "reranker_instruction": QUERY_INSTRUCTION,
                "reranker_max_length": self.max_length, "reranker_precision": self.dtype,
                "reranker_score": "yes_minus_no_logit"}

    def _load(self):
        if self.model is not None:
            return
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError("Qwen3 reranking requires torch and transformers>=4.51.0") from exc
        self.device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id, revision=self.revision, padding_side="left")
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            revision=self.revision,
            torch_dtype=resolve_torch_dtype(torch, self.dtype),
        ).to(self.device).eval()

    def rerank(self, query, candidates):
        if not candidates:
            return []
        self._load()
        import torch
        prefix = self.tokenizer.encode(self.PREFIX, add_special_tokens=False)
        suffix = self.tokenizer.encode(self.SUFFIX, add_special_tokens=False)
        budget = self.max_length - len(prefix) - len(suffix)
        if budget < 1:
            raise ValueError("reranker max_length is too small for the Qwen template")
        yes = self.tokenizer.convert_tokens_to_ids("yes")
        no = self.tokenizer.convert_tokens_to_ids("no")
        scores = []
        for start in range(0, len(candidates), self.batch_size):
            texts = [f"<Instruct>: {QUERY_INSTRUCTION}\n<Query>: {query}\n<Document>: {document_text(c)}"
                     for c in candidates[start:start + self.batch_size]]
            encoded = self.tokenizer(texts, padding=False, truncation=True, max_length=budget, add_special_tokens=False)
            ids = [prefix + row + suffix for row in encoded["input_ids"]]
            batch = self.tokenizer.pad({"input_ids": ids, "attention_mask": [[1] * len(row) for row in ids]}, padding=True, return_tensors="pt").to(self.device)
            with torch.inference_mode():
                logits = self.model(**batch, use_cache=False).logits[:, -1, :]
                scores.extend((logits[:, yes] - logits[:, no]).float().cpu().tolist())
        return scores


class BGEReranker(Qwen3Reranker):
    def __init__(self, model_id="BAAI/bge-reranker-v2-m3", **kwargs):
        super().__init__(model_id, **kwargs)

    def provenance(self):
        return {"reranker_revision": self.revision, "reranker_instruction": None,
                "reranker_max_length": self.max_length, "reranker_precision": "float32",
                "reranker_score": "cross_encoder_logit"}

    def _load(self):
        if self.model is not None:
            return
        try:
            from FlagEmbedding import FlagReranker
            from huggingface_hub import snapshot_download
        except ImportError as exc:
            raise RuntimeError("BGE reranking requires `pip install FlagEmbedding`") from exc
        path = self.model_id if Path(self.model_id).is_dir() else snapshot_download(self.model_id, revision=self.revision)
        self.model = FlagReranker(path, use_fp16=False, **({"devices": self.device} if self.device else {}))

    def rerank(self, query, candidates):
        if not candidates:
            return []
        self._load()
        scores = self.model.compute_score([[query, document_text(c)] for c in candidates],
            batch_size=self.batch_size, max_length=self.max_length, normalize=False)
        return np.asarray(scores, dtype=np.float32).reshape(-1).tolist()
