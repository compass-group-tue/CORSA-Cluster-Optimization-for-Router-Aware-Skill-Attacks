"""Abstract router interface.

Router adapters implement the same two-method contract:
  - prepare_pool(pool): cache features for a competitive skill pool.
  - rank(query, our_skill) -> int | None: return the zero-based candidate
    rank, or None when it falls outside a retrieve-then-rerank window.

Victim scaffolds such as Claude Code or Codex CLI are distinct from this
retrieval interface and require experiment-provided runtime adapters.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

OUR_SKILL_ID = "__attacker_skill__"


class Router(ABC):
    label: str  # short slug used in logs / result tables

    @abstractmethod
    def prepare_pool(self, pool: list[dict]) -> None:
        """Ingest the benign skill pool once. Subsequent rank() calls reuse
        whatever cached state this build produced (embeddings, BM25 index)."""
        ...

    @abstractmethod
    def rank(self, query: str, our_skill: dict) -> int | None:
        """Rank our_skill among (prepared pool + our_skill). Return 0 for
        top-1. None if our_skill fell outside the router's retrieval window."""
        ...
