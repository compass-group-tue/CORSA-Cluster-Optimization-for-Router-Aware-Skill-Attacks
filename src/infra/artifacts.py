"""Versioned skill artifacts shared by curriculum and frozen evaluation."""
from __future__ import annotations
import hashlib
import json
import math
from pathlib import Path
from src.exec_env.contracts import ConfigurationError
from src.routers.base import OUR_SKILL_ID

SKILL_FIELDS = {"skill_id", "name", "description", "body", "payload_framing"}


def fingerprint(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    allow_nan=False).encode()).hexdigest()


def validate_skill(skill: dict) -> dict:
    if not isinstance(skill, dict) or set(skill) - SKILL_FIELDS:
        raise ConfigurationError("Skill must contain only public text fields")
    for key in ("name", "description", "body"):
        if not isinstance(skill.get(key), str):
            raise ConfigurationError(f"Skill requires string {key}")
    if not skill["name"].strip():
        raise ConfigurationError("Skill name cannot be empty")
    if "payload_framing" in skill and not isinstance(skill["payload_framing"], str):
        raise ConfigurationError("payload_framing must be a string")
    return {"payload_framing": "", **skill, "skill_id": OUR_SKILL_ID}


def skill_hash(skill: dict) -> str:
    return fingerprint(validate_skill(skill))


def make_winner(*, stage, skill, cluster, router, config, task_ids,
                source_run, score, parent=None):
    artifact = dict(schema_version=1, kind="corsa_winner", stage=stage,
                    skill=validate_skill(skill), skill_sha256=skill_hash(skill),
                    cluster=cluster, router=router, configuration=config,
                    task_ids=list(task_ids), source_run=str(source_run),
                    score=score, parent=parent)
    artifact["artifact_sha256"] = fingerprint(artifact)
    validate_winner(artifact)
    return artifact


def validate_winner(value, *, stage=None, cluster=None):
    if not isinstance(value, dict) or value.get("schema_version") != 1 or value.get("kind") != "corsa_winner":
        raise ConfigurationError("Expected schema_version=1 corsa_winner artifact")
    if value.get("stage") not in {"A", "B"} or (stage and value["stage"] != stage):
        raise ConfigurationError(f"Expected Stage-{stage or 'A/B'} winner")
    if not isinstance(value.get("cluster"), str) or not value["cluster"] or (cluster and value["cluster"] != cluster):
        raise ConfigurationError("Winner cluster does not match requested cluster")
    if not isinstance(value.get("router"), dict) or not value["router"].get("router"):
        raise ConfigurationError("Winner requires router provenance")
    if not isinstance(value.get("configuration"), dict) or type(value["configuration"].get("seed")) is not int:
        raise ConfigurationError("Winner requires configuration and random seed")
    if not value.get("source_run") or not isinstance(value.get("task_ids"), list) or not value["task_ids"]:
        raise ConfigurationError("Winner requires source_run and task_ids")
    if any(not isinstance(t, str) or not t for t in value["task_ids"]) or len(set(value["task_ids"])) != len(value["task_ids"]):
        raise ConfigurationError("Winner task IDs must be unique nonempty strings")
    score = value.get("score", {})
    for key in ("hit_at_1", "optimization_reward"):
        v = score.get(key)
        if type(v) not in (int, float) or not math.isfinite(v):
            raise ConfigurationError(f"Winner requires finite score.{key}")
    if not 0 <= score["hit_at_1"] <= 1:
        raise ConfigurationError("Winner Hit@1 must be in [0,1]")
    if value.get("skill_sha256") != skill_hash(value.get("skill")):
        raise ConfigurationError("Winner skill hash mismatch")
    unsigned = {k: v for k, v in value.items() if k != "artifact_sha256"}
    if value.get("artifact_sha256") != fingerprint(unsigned):
        raise ConfigurationError("Winner provenance hash mismatch")
    if value["stage"] == "B":
        parent = value.get("parent")
        if not isinstance(parent, dict) or not all(parent.get(k) for k in (
            "source_run", "artifact_sha256", "skill_sha256", "stage_a_score"
        )):
            raise ConfigurationError("Stage-B winner requires complete Stage-A parent provenance")
    return value


def load_winner(path, **requirements):
    return validate_winner(json.loads(Path(path).read_text()), **requirements)


def stage_b_seed(path, cluster):
    artifact = load_winner(path, stage="A", cluster=cluster)
    return dict(artifact["skill"]), artifact
