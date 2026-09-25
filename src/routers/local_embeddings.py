"""Native Qwen3 last-token pooling and BGE-M3 dense inference."""
import os
from pathlib import Path

from ._embedding_utils import (
    CachedEmbedder,
    QUERY_INSTRUCTION,
    resolve_device,
    resolve_torch_dtype,
)


class Qwen3Embedder(CachedEmbedder):
    def __init__(self, model_id="Qwen/Qwen3-Embedding-0.6B", *, cache_dir=None,
                 revision=None, batch_size=None, device=None, max_length=4096,
                 dtype=None):
        self.model_id = model_id
        self.revision = (
            revision
            or os.environ.get("ROUTER_ENCODER_REVISION")
            or os.environ.get("ROUTER_MODEL_REVISION")
            or "main"
        )
        self.device = resolve_device(device)
        self.max_length = max_length
        self.dtype = dtype or os.environ.get("ROUTER_TORCH_DTYPE") or "float32"
        self.dimension = None
        self.model = self.tokenizer = None
        self._init_cache(cache_dir, {"adapter": "qwen3-last-token", "model": model_id,
            "revision": self.revision, "max_length": max_length, "instruction": QUERY_INSTRUCTION},
            batch_size if batch_size is not None else int(os.environ.get("ROUTER_BATCH_SIZE") or "8"))

    def provenance(self):
        return {"encoder_revision": self.revision, "query_instruction": QUERY_INSTRUCTION,
                "pooling": "last_non_padding_token", "max_length": self.max_length,
                "precision": self.dtype, "cache": str(self.cache.path)}

    def _load(self):
        if self.model is not None:
            return
        try:
            import torch
            from transformers import AutoModel, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError("Qwen3 embeddings require torch and transformers>=4.51.0") from exc
        self.device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id, revision=self.revision, padding_side="left")
        self.model = AutoModel.from_pretrained(
            self.model_id,
            revision=self.revision,
            torch_dtype=resolve_torch_dtype(torch, self.dtype),
        ).to(self.device).eval()

    @staticmethod
    def last_token_pool(hidden, attention_mask):
        import torch
        positions = torch.arange(attention_mask.shape[1], device=attention_mask.device)
        last = (positions[None, :] * attention_mask).max(dim=1).values
        return hidden[torch.arange(hidden.shape[0], device=hidden.device), last]

    def _encode(self, texts, role):
        self._load()
        import torch
        if role == "query":
            texts = [f"Instruct: {QUERY_INSTRUCTION}\nQuery: {text}" for text in texts]
        batch = self.tokenizer(texts, padding=True, truncation=True, max_length=self.max_length, return_tensors="pt").to(self.device)
        with torch.inference_mode():
            return self.last_token_pool(self.model(**batch).last_hidden_state, batch["attention_mask"]).float().cpu().numpy()


class BGEM3Embedder(CachedEmbedder):
    def __init__(self, model_id="BAAI/bge-m3", *, cache_dir=None,
                 revision=None, batch_size=None, device=None, max_length=4096):
        self.model_id = model_id
        self.revision = revision or os.environ.get("ROUTER_MODEL_REVISION") or "main"
        self.device = resolve_device(device)
        self.max_length = max_length
        self.dimension = None
        self.model = None
        self._init_cache(cache_dir, {"adapter": "bge-m3-dense", "model": model_id,
            "revision": self.revision, "max_length": max_length},
            batch_size if batch_size is not None else int(os.environ.get("ROUTER_BATCH_SIZE") or "8"))

    def provenance(self):
        return {"encoder_revision": self.revision, "query_instruction": None,
                "embedding_mode": "dense", "max_length": self.max_length,
                "precision": "float32", "cache": str(self.cache.path)}

    def _load(self):
        if self.model is not None:
            return
        try:
            from FlagEmbedding import BGEM3FlagModel
            from huggingface_hub import snapshot_download
        except ImportError as exc:
            raise RuntimeError("BGE-M3 routing requires `pip install FlagEmbedding`") from exc
        path = self.model_id if Path(self.model_id).is_dir() else snapshot_download(self.model_id, revision=self.revision)
        self.model = BGEM3FlagModel(path, use_fp16=False, **({"devices": self.device} if self.device else {}))

    def _encode(self, texts, role):
        self._load()
        return self.model.encode(texts, batch_size=self.batch_size, max_length=self.max_length,
            return_dense=True, return_sparse=False, return_colbert_vecs=False)["dense_vecs"]
