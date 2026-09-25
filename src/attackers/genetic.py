"""Shared CORSA candidate identity and explicit scoring."""
from copy import deepcopy
import math
from src.infra.artifacts import skill_hash
from src.payloads.prerequisite import attach_prerequisite

_skill_hash = skill_hash


def replace_fields(candidate, updates, allowed):
    if set(updates) - set(allowed):
        raise ValueError('Mutation attempted to change a frozen field')
    return {**deepcopy(candidate), **updates}


def feedback_success(row):
    return bool(row['hit_at_1']) if row['metric_version'] == 'corsa-retrieval-v1' else row['asr'] == 1


def score_retrieval(candidate, task, router, *, cluster, provenance, hit_bonus=1.0):
    rendered = attach_prerequisite(candidate)
    rank = router.rank(task['instruction_text'], rendered)
    if rank is not None and (type(rank) is not int or rank < 0):
        raise ValueError('Router rank must be nonnegative or None')
    hit = int(rank == 0)
    margin = float(router.retrieval_similarity(task['instruction_text'], rendered))
    if not math.isfinite(margin) or not math.isfinite(hit_bonus):
        raise ValueError('Retrieval reward components must be finite')
    return dict(task_id=task['task_id'], cluster=cluster, skill_sha256=skill_hash(candidate),
                metric_version='corsa-retrieval-v1', provenance=provenance,
                router_rank=rank, hit_at_1=hit, exec=None, asr=None,
                execution_state='not_attempted', utility=None, utility_state='unavailable',
                naturalism=None, naturalism_state='unavailable', retrieval_margin=margin,
                optimization_reward=margin + hit_bonus * hit)
