"""Load and subsample SkillRouter's eval_core benchmark.

Retain sampled tasks' ground-truth skills in the competitive pool. Dataset
sizes and pool composition are determined from the loaded benchmark files.
"""
from __future__ import annotations

import gzip
import json
import os
import random
from pathlib import Path

from src.infra.paths import SKILLROUTER_REPO
from src.infra.artifacts import fingerprint

SR_DATA_ROOT = SKILLROUTER_REPO / "data/eval_core"


def _read_jsonl_or_shards(path: Path) -> list[dict]:
    """`path` is either a single .jsonl file OR a directory of .jsonl.gz shards."""
    if path.is_file():
        return [json.loads(line) for line in path.read_text().splitlines() if line]
    if path.is_dir():
        recs: list[dict] = []
        for shard in sorted(path.glob("*.jsonl.gz")):
            with gzip.open(shard, "rt", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if line:
                        recs.append(json.loads(line))
        return recs
    raise FileNotFoundError(f"{path} not found")


def load_core_tasks() -> tuple[list[dict], dict]:
    tasks = _read_jsonl_or_shards(SR_DATA_ROOT / "tasks.jsonl")
    relevance = json.loads((SR_DATA_ROOT / "relevance.json").read_text())
    core = [t for t in tasks if relevance.get(t["task_id"], {}).get("task_type") != "generic_only"]
    return core, relevance


def load_easy_pool() -> list[dict]:
    return _read_jsonl_or_shards(SR_DATA_ROOT / "easy")


def load_hard_pool() -> list[dict]:
    """Load the external hard pool; validate actual distractor coverage at runtime."""
    return _read_jsonl_or_shards(SR_DATA_ROOT / "hard")


def subsample(
    n_tasks: int,
    m_pool: int,
    seed: int = 20260714,
    task_split: str = "mixed",
    tier: str | None = None,
) -> tuple[list[dict], dict, list[dict]]:
    """Return (tasks_subset, relevance_subset, pool_subset).

    Guarantees every task's ground-truth skills survive in pool_subset.

    task_split:
      "mixed"       — no filtering (default).
      "single_only" — keep queries with exactly one core ground-truth skill.
      "multi_only"  — keep queries with 2+ core ground-truth skills.
    """
    # Explicit CLI arguments take precedence over the optional library default.
    tier = (tier or os.environ.get("SKILL_TIER", "easy")).lower()
    validate_tier(tier)
    tasks, relevance = load_core_tasks()
    full_pool = load_hard_pool() if tier == "hard" else load_easy_pool()
    pool_by_id = {s["skill_id"]: s for s in full_pool}

    # Apply task_split filter BEFORE subsampling so N_TASKS lands within
    # the requested class.
    if task_split in ("single_only", "multi_only"):
        # Split on core_gt_ids, following SkillRouter's definition. gt_skill_ids
        # also includes auxiliary skills with lower relevance grades.
        def _gt_count(tid: str) -> int:
            rel = relevance.get(tid, {})
            core = rel.get("core_gt_ids")
            if core is None:
                core = rel.get("gt_skill_ids", [])
            return len(core or [])
        if task_split == "single_only":
            tasks = [t for t in tasks if _gt_count(t["task_id"]) == 1]
        else:
            tasks = [t for t in tasks if _gt_count(t["task_id"]) > 1]

    rng = random.Random(seed)
    ids = [t["task_id"] for t in tasks]
    rng.shuffle(ids)
    picked_ids = ids[:n_tasks]
    picked_tasks = [t for t in tasks if t["task_id"] in set(picked_ids)]

    needed: set[str] = set()
    for t in picked_tasks:
        for sid in relevance.get(t["task_id"], {}).get("gt_skill_ids", []) or []:
            if sid in pool_by_id:
                needed.add(sid)
    # Retain all loaded distractors rather than assuming a fixed external count.
    if tier == "hard":
        for s in full_pool:
            if s.get("source") == "distractor":
                needed.add(s["skill_id"])

    remaining = [sid for sid in pool_by_id if sid not in needed]
    rng.shuffle(remaining)
    extra = remaining[: max(0, m_pool - len(needed))]

    pool_ids = sorted(needed) + extra
    rng.shuffle(pool_ids)
    picked_pool = [pool_by_id[sid] for sid in pool_ids]

    picked_rel = {t["task_id"]: relevance[t["task_id"]] for t in picked_tasks}
    return picked_tasks, picked_rel, picked_pool


def validate_tier(tier):
    if tier not in {"easy", "hard"}:
        raise ValueError("tier must be easy or hard")
    return tier


def pool_statistics(tasks, relevance, pool, tier):
    validate_tier(tier)
    ids = {s["skill_id"] for s in pool}
    task_required = {sid for t in tasks for sid in (relevance.get(t["task_id"], {}).get("gt_skill_ids") or [])}
    required = {sid for rel in relevance.values() for sid in (rel.get("gt_skill_ids") or [])}
    missing = required - ids
    if missing:
        raise ValueError(f"Required benchmark skills absent from pool: {sorted(missing)}")
    distractors = sum(s.get("source") == "distractor" for s in pool)
    if tier == "hard" and not distractors:
        raise ValueError("Hard tier loaded no distractor skills")
    return dict(n_tasks=len(tasks), n_benign_skills=len(pool) - distractors,
                n_required_skills=len(required), n_task_required_skills=len(task_required),
                n_distractor_skills=distractors,
                n_pool_skills=len(pool), n_total_candidates=len(pool) + 1,
                pool_sha256=fingerprint(pool),
                tier=tier)
