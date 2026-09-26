"""Central path configuration."""
from __future__ import annotations

import os
from pathlib import Path

# Repository root = two levels up from this file (src/infra/paths.py -> repo).
# Override with CORSA_ROOT if you run the code from elsewhere.
PROJECT_ROOT = Path(os.environ.get("CORSA_ROOT", str(Path(__file__).resolve().parents[2])))
REPO_ROOT = PROJECT_ROOT

# External SkillRouter and SkillsBench checkouts, configured by environment
# variables. See README: Installation and configuration.
SKILLROUTER_REPO = Path(os.environ.get("SKILLROUTER_REPO", str(PROJECT_ROOT / "benchmarks/skillrouter_repo")))
SKILLSBENCH_REPO = Path(os.environ.get("SKILLSBENCH_REPO", str(PROJECT_ROOT / "benchmarks/skillsbench_repo")))

# Per-run scratch location; this path does not provide execution isolation.
# CORSA_SCRATCH overrides the local, git-ignored default.
REAL_EXEC_SCRATCH = Path(os.environ.get("CORSA_SCRATCH", str(PROJECT_ROOT / ".corsa_scratch")))


def skillsbench_task_dir(task_id: str) -> Path:
    d = SKILLSBENCH_REPO / "tasks" / task_id
    if not d.is_dir():
        d = SKILLSBENCH_REPO / "tasks-extra" / task_id
    if not d.is_dir():
        raise FileNotFoundError(f"No SkillsBench task dir for task_id={task_id!r}")
    return d


# Default pinned task categorization (task_id -> cluster). Entry points may
# explicitly select an alternative categorization file.
TASK_CATEGORIZATION_DIR = PROJECT_ROOT / "data/task_categorization"


def default_categories_json() -> Path:
    """Path to the canonical task_categories.json (the common source of truth)."""
    return TASK_CATEGORIZATION_DIR / "task_categories.json"


def resolve_categories_json(out_dir: str | os.PathLike | None = None) -> Path:
    """Prefer a per-run section_2_2_categorization/task_categories.json under
    out_dir; fall back to the canonical common file. Raises if neither exists."""
    if out_dir:
        per_run = Path(out_dir) / "section_2_2_categorization" / "task_categories.json"
        if per_run.is_file():
            return per_run
    common = default_categories_json()
    if common.is_file():
        return common
    raise FileNotFoundError(f"No task_categories.json in {out_dir} nor at {common}")


# HuggingFace cache (persistent so model downloads survive restarts).
HF_HOME = Path(os.environ.get("HF_HOME", str(Path.home() / ".cache/huggingface")))


def load_openai_key() -> str:
    """Return the API key from the OPENAI_API_KEY environment variable."""
    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("Set the OPENAI_API_KEY environment variable.")
    return key

ROUTER_CACHE_DIR = Path(os.environ.get("ROUTER_CACHE_DIR", str(HF_HOME / "corsa_routers")))
