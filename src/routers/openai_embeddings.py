"""Batched OpenAI embeddings with explicit, independent Azure configuration."""
import os
from urllib.parse import urlparse

from ._embedding_utils import CachedEmbedder


class OpenAIEmbedder(CachedEmbedder):
    def __init__(self, model_id="text-embedding-3-large", *, dimensions=None,
                 cache_dir=None, batch_size=None, max_length=4096, client=None):
        if dimensions is not None and not 1 <= dimensions <= 3072:
            raise ValueError("OpenAI embedding dimensions must be in [1, 3072]")
        self.model_id = model_id
        self.dimensions = dimensions
        bare_model_id = model_id.rsplit("/", 1)[-1]
        native_dimensions = {
            "text-embedding-3-large": 3072,
            "text-embedding-3-small": 1536,
            "text-embedding-ada-002": 1536,
        }
        self.dimension = dimensions or native_dimensions.get(bare_model_id)
        self.max_length = max_length
        self._client = client
        self._tokenizer = None
        self.endpoint = os.environ.get("AZURE_OPENAI_EMBEDDING_ENDPOINT", "")
        self.deployment = os.environ.get("AZURE_OPENAI_EMBEDDING_DEPLOYMENT", "")
        self.base_url = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
        if bool(self.endpoint) != bool(self.deployment):
            raise RuntimeError("Set both AZURE_OPENAI_EMBEDDING_ENDPOINT and AZURE_OPENAI_EMBEDDING_DEPLOYMENT")
        host = (urlparse(self.base_url).hostname or "").lower()
        self.provider = (
            "azure" if self.endpoint
            else "openrouter" if host == "openrouter.ai" or host.endswith(".openrouter.ai")
            else "openai"
        )
        self._init_cache(cache_dir, {"adapter": "openai", "model": model_id,
            "dimensions": dimensions, "max_length": max_length,
            "endpoint": self.endpoint or self.base_url,
            "deployment": self.deployment},
            batch_size if batch_size is not None else int(os.environ.get("ROUTER_BATCH_SIZE") or "32"))
        if self.batch_size > 64:
            raise ValueError("OpenAI embedding batch size must not exceed 64 at the 4096-token ceiling")

    def provenance(self):
        return {"provider": self.provider,
                "embedding_dimension": self.dimension, "embedding_deployment": self.deployment or None,
                "api_base": self.endpoint or self.base_url,
                "max_length": self.max_length, "query_instruction": None,
                "cache": str(self.cache.path)}

    def _get_client(self):
        if self._client is None:
            try:
                from openai import OpenAI, AzureOpenAI
            except ImportError as exc:
                raise RuntimeError("OpenAI embeddings require `pip install openai`") from exc
            kwargs = {"max_retries": int(os.environ.get("OPENAI_MAX_RETRIES", "8")),
                      "timeout": float(os.environ.get("OPENAI_TIMEOUT_S", "120"))}
            if self.endpoint:
                key = (os.environ.get("AZURE_OPENAI_API_KEY") or os.environ.get("AZURE_OPENAI_KEY")
                       or os.environ.get("AZURE_ACCESS_KEY"))
                if not key:
                    raise RuntimeError("Azure embeddings require AZURE_OPENAI_API_KEY, AZURE_OPENAI_KEY or AZURE_ACCESS_KEY")
                self._client = AzureOpenAI(api_key=key, azure_endpoint=self.endpoint,
                    api_version=os.environ.get("AZURE_OPENAI_API_VERSION", "2024-02-01"), **kwargs)
            else:
                if self.provider == "openrouter":
                    key = (os.environ.get("OPEN_ROUTER_KEY")
                           or os.environ.get("OPENROUTER_API_KEY")
                           or os.environ.get("OPENAI_API_KEY"))
                    if not key:
                        raise RuntimeError(
                            "OpenRouter embeddings require OPEN_ROUTER_KEY, "
                            "OPENROUTER_API_KEY or OPENAI_API_KEY"
                        )
                else:
                    key = os.environ.get("OPENAI_API_KEY")
                    if not key:
                        raise RuntimeError("OpenAI embeddings require OPENAI_API_KEY")
                self._client = OpenAI(api_key=key, base_url=self.base_url, **kwargs)
        return self._client

    def _encode(self, texts, role):
        if self._tokenizer is None:
            try:
                import tiktoken
            except ImportError as exc:
                raise RuntimeError("OpenAI embedding token limits require `pip install tiktoken`") from exc
            self._tokenizer = tiktoken.get_encoding("cl100k_base")
        inputs = [self._tokenizer.encode(text, disallowed_special=())[:self.max_length] for text in texts]
        if any(not tokens for tokens in inputs):
            raise ValueError("OpenAI embedding input must not be empty")
        kwargs = {"model": self.deployment or self.model_id, "input": inputs, "encoding_format": "float"}
        if self.dimensions is not None:
            kwargs["dimensions"] = self.dimensions
        response = self._get_client().embeddings.create(**kwargs)
        data = sorted(response.data, key=lambda item: item.index)
        if [item.index for item in data] != list(range(len(texts))):
            raise ValueError("OpenAI returned missing or duplicate embedding indices")
        return [item.embedding for item in data]
