"""Single router factory shared by every optimization/evaluation entry point."""
from __future__ import annotations

import os
from pathlib import Path
from src.infra.paths import ROUTER_CACHE_DIR


ROUTER_IDS = (
    "bm25",
    "encoder_only",
    "full_skillrouter",
    "skillrouter_8b",
    "gemini_embedding_2",
    "qwen3_embedding",
    "qwen3_8b_pair",
    "bge_m3_dense",
    "openai_embedding_3_large",
    "skillrouter_qwen3_reranker",
    "skillrouter_bge_reranker",
    "r3_skill_06",
)

SR_06_ENCODER = "pipizhao/SkillRouter-Embedding-0.6B"
SR_06_RERANKER = "pipizhao/SkillRouter-Reranker-0.6B"


def add_router_arguments(parser, default: str = "full_skillrouter") -> None:
    parser.add_argument("--router", choices=ROUTER_IDS, default=default)
    parser.add_argument("--router_encoder_model", default="")
    parser.add_argument("--router_reranker_model", default="")
    parser.add_argument("--router_retrieval_top_k", type=int, default=20)
    parser.add_argument("--router_embedding_dim", type=int, default=0,
                        help="0 uses model defaults (Gemini: 768, OpenAI large: 3072)")
    parser.add_argument("--router_cache_dir", default=os.environ.get(
        "ROUTER_CACHE_DIR", str(ROUTER_CACHE_DIR)
    ))


def build_router(
    name: str,
    *,
    encoder_model: str = "",
    reranker_model: str = "",
    retrieval_top_k: int = 20,
    embedding_dim: int = 0,
    cache_dir: str | Path | None = None,
):
    if name == "bm25":
        from .bm25 import BM25Router
        return BM25Router()

    if name == "gemini_embedding_2":
        from .dense import DenseEmbeddingRouter
        from .gemini import GeminiEmbedding2Embedder
        embedder = GeminiEmbedding2Embedder(
            model_id=encoder_model or "gemini-embedding-2",
            output_dimensionality=embedding_dim or 768,
            cache_dir=cache_dir,
        )
        return DenseEmbeddingRouter(embedder, label=name)

    if name in ("qwen3_embedding", "bge_m3_dense", "openai_embedding_3_large"):
        from .dense import DenseEmbeddingRouter
        if name == "openai_embedding_3_large":
            from .openai_embeddings import OpenAIEmbedder
            embedder = OpenAIEmbedder(encoder_model or "text-embedding-3-large",
                                     dimensions=embedding_dim or None, cache_dir=cache_dir)
        else:
            from .local_embeddings import Qwen3Embedder, BGEM3Embedder
            cls, model = ((Qwen3Embedder, "Qwen/Qwen3-Embedding-0.6B") if name == "qwen3_embedding"
                          else (BGEM3Embedder, "BAAI/bge-m3"))
            embedder = cls(encoder_model or model, cache_dir=cache_dir)
        return DenseEmbeddingRouter(embedder, label=name)

    if name in ("skillrouter_qwen3_reranker", "skillrouter_bge_reranker"):
        from ._skillrouter_client import SkillRouterClient
        from .native_rerankers import Qwen3Reranker, BGEReranker
        from .retrieve_then_rerank import RetrieveThenRerankRouter
        cls, model = ((Qwen3Reranker, "Qwen/Qwen3-Reranker-0.6B") if name == "skillrouter_qwen3_reranker"
                      else (BGEReranker, "BAAI/bge-reranker-v2-m3"))
        return RetrieveThenRerankRouter(
            SkillRouterClient(encoder_path=encoder_model or SR_06_ENCODER, load_reranker=False),
            cls(reranker_model or model), retrieval_top_k=retrieval_top_k, label=name)

    if name == "qwen3_8b_pair":
        from .local_embeddings import Qwen3Embedder
        from .native_rerankers import Qwen3Reranker
        from .retrieve_then_rerank import DenseRetrieveThenRerankRouter
        return DenseRetrieveThenRerankRouter(
            Qwen3Embedder(
                encoder_model or "Qwen/Qwen3-Embedding-8B",
                cache_dir=cache_dir,
            ),
            Qwen3Reranker(reranker_model or "Qwen/Qwen3-Reranker-8B"),
            retrieval_top_k=retrieval_top_k,
            label=name,
        )

    if name == "r3_skill_06":
        from .r3 import R3_ENCODER, R3_RERANKER, R3SkillClient, R3SkillRouter
        client = R3SkillClient(
            encoder_path=encoder_model or R3_ENCODER,
            reranker_path=reranker_model or R3_RERANKER,
        )
        return R3SkillRouter(
            client=client, retrieval_top_k=retrieval_top_k, label=name
        )

    from ._skillrouter_client import SkillRouterClient
    if name == "encoder_only":
        from .encoder_only import EncoderOnlyRouter
        client = SkillRouterClient(
            encoder_path=encoder_model or SR_06_ENCODER,
            load_reranker=False,
        )
        return EncoderOnlyRouter(client=client)

    if name in ("full_skillrouter", "skillrouter_8b"):
        from .skillrouter import SkillRouterFull
        if name == "skillrouter_8b" and not (encoder_model and reranker_model):
            raise RuntimeError(
                "skillrouter_8b requires the exact accessible checkpoints via "
                "--router_encoder_model and --router_reranker_model"
            )
        default_encoder = SR_06_ENCODER
        default_reranker = SR_06_RERANKER
        client = SkillRouterClient(
            encoder_path=encoder_model or default_encoder,
            reranker_path=reranker_model or default_reranker,
        )
        return SkillRouterFull(
            client=client, retrieval_top_k=retrieval_top_k, label=name
        )

    raise ValueError(f"unknown router {name!r}; choose one of {ROUTER_IDS}")


def build_router_from_args(args):
    return build_router(
        args.router,
        encoder_model=args.router_encoder_model,
        reranker_model=args.router_reranker_model,
        retrieval_top_k=args.router_retrieval_top_k,
        embedding_dim=args.router_embedding_dim,
        cache_dir=args.router_cache_dir,
    )


def get_provenance(router) -> dict:
    method = getattr(router, "provenance", None)
    if callable(method):
        return method()
    return {"router": router.label, "kind": type(router).__name__}
