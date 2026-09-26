"""Group SkillsBench application domains into eight task clusters.

Outputs:
  task_categories.json         task ID to category mapping
  super_category_labels.json   category labels, descriptions and domains
  per_cluster_counts.json      task counts by category
  domains_grouping.json        raw LLM JSON output
  pipeline.log                 progress log
  raw_prompt.txt               categorization prompt
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

from src.infra.paths import PROJECT_ROOT as PROJECT

from src.infra.paths import load_openai_key  # noqa: E402
from src.benchmarks.skillrouter_bench import load_core_tasks  # noqa: E402


FIXED_SUPER_CATEGORIES = [
    {"label": "scientific-research",
     "description": "physics / chemistry / biology / astronomy / earth-science research"},
    {"label": "coding-and-devops",
     "description": "programming languages, compilers, build systems, code translation, code repro"},
    {"label": "data-science-and-analytics",
     "description": "data cleaning, statistics, ML implementation, forecasting, analysis"},
    {"label": "document-and-office",
     "description": "docx / pptx / xlsx / pdf editing, latex, spreadsheets, document workflows"},
    {"label": "media-and-signals",
     "description": "audio, video, 3D graphics, image processing, speech synthesis"},
    {"label": "security-and-privacy",
     "description": "security testing, CTF, credential handling, privacy analysis"},
    {"label": "engineering-and-simulation",
     "description": "control systems, manufacturing, quantum simulation, formal methods, hardware"},
    {"label": "general-tooling-and-web",
     "description": "web performance, BGP routing, file management, scheduling, general utilities"},
]

CATEGORIZE_SYSTEM = (
    "You are labelling every application-domain of a coding-agent benchmark. "
    "For each domain you MUST pick exactly ONE of the fixed 8 super-category "
    "labels I provide. Output ONLY strict JSON."
)

CATEGORIZE_USER_TMPL = """\
The 8 fixed super-category labels you must choose from:
{fixed_labels}

Below are the coding-agent domains you need to label, each with 1-2 example
task descriptions so you can see what each domain covers.

DOMAINS TO LABEL:
{domains_block}

For each domain, pick exactly ONE super-category label from the list above.
Domain names must be COPIED EXACTLY (case- and character-preserving).

Return JSON with this exact shape (a flat mapping):
{{
  "assignments": {{
    "<domain_a>": "<super_category_label>",
    "<domain_b>": "<super_category_label>",
    ...
  }}
}}

Every one of the {n_domains} domains listed above must appear as a key.
Return JSON only. No commentary, no code fences.
"""


def _build_domains_block(tasks: list[dict], max_examples: int = 2, snippet_len: int = 140) -> str:
    """Group tasks by domain and format a compact per-domain listing."""
    by_dom: dict[str, list[str]] = {}
    for t in tasks:
        dom = t.get("domain", "unknown") or "unknown"
        instr = (t.get("instruction_text") or "").replace("\n", " ")[:snippet_len]
        by_dom.setdefault(dom, []).append(instr)
    lines: list[str] = []
    for dom in sorted(by_dom):
        lines.append(f"- {dom}:")
        for ex in by_dom[dom][:max_examples]:
            lines.append(f"    * {ex}")
    return "\n".join(lines)


def _mutator_call(client, model: str, system: str, user: str) -> dict:
    kwargs = dict(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        response_format={"type": "json_object"},
    )
    if model.startswith(("gpt-5", "o1", "o3")):
        kwargs["max_completion_tokens"] = 4000
    else:
        kwargs["temperature"] = 0.0
        kwargs["max_tokens"] = 4000
    resp = client.chat.completions.create(**kwargs)
    return json.loads((resp.choices[0].message.content or "{}").strip())


_FALLBACK_KEYWORDS = [
    ("security-and-privacy", ("secur", "ctf", "privacy", "credential")),
    ("coding-and-devops", ("code", "coding", "compil", "erlang", "devops",
                            "parallel", "translation", "reproduction",
                            "flink", "legacy", "bugfix")),
    ("data-science-and-analytics", ("data", "statistic", "ml", "nlp",
                                     "analysis", "forecast", "analytics",
                                     "econom", "finance", "healthcare")),
    ("document-and-office", ("document", "spreadsheet", "latex", "office",
                              "editing")),
    ("media-and-signals", ("audio", "video", "image", "media", "speech",
                            "3d", "graphics", "multimodal", "visual",
                            "dubbing", "subtitle")),
    ("scientific-research", ("research", "seismolog", "astronom", "chem",
                              "materials", "quantum", "biolog", "physics",
                              "geo", "environment")),
    ("engineering-and-simulation", ("control", "manufactur", "simulation",
                                     "formal", "traffic", "bgp", "pedestrian",
                                     "energ", "erp", "hardware", "system")),
    ("general-tooling-and-web", ("web", "file", "management", "search",
                                   "travel", "games", "game", "scheduling",
                                   "visualization")),
]


def _fallback_bucket(domain: str) -> str:
    """Deterministic assignment for domains the LLM missed."""
    d = domain.lower()
    for label, keys in _FALLBACK_KEYWORDS:
        for k in keys:
            if k in d:
                return label
    return "general-tooling-and-web"


def _validate_assignments(assignments: dict, all_domains: set[str],
                          valid_labels: set[str]) -> tuple[bool, str]:
    """Validate a flat {domain: super_category} mapping."""
    if not isinstance(assignments, dict):
        return False, "'assignments' missing or not a dict"
    missing = sorted(all_domains - set(assignments))
    extra = sorted(set(assignments) - all_domains)
    if missing:
        return False, f"missing domains: {missing[:5]}..."
    if extra:
        return False, f"unknown domains: {extra[:5]}..."
    bad_labels = [f"{d}={lbl}" for d, lbl in assignments.items()
                  if lbl not in valid_labels]
    if bad_labels:
        return False, f"invalid super-category label(s): {bad_labels[:3]}..."
    return True, "ok"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", required=True,
                    help="Where to write task_categories.json etc.")
    ap.add_argument("--model", default="gpt-5-2025-08-07")
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    log = (out / "pipeline.log").open("w", buffering=1)

    def say(msg: str) -> None:
        stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        line = f"[{stamp}] {msg}"
        log.write(line + "\n")
        print(line, flush=True)

    say("=== Section 2.2 — 55 domains → 8 super-categories ===")

    tasks, relevance = load_core_tasks()
    say(f"loaded {len(tasks)} core tasks from SkillsBench")

    domains = Counter(t.get("domain", "unknown") for t in tasks)
    say(f"unique domains present: {len(domains)}")
    say(f"top 5 by count: {domains.most_common(5)}")

    domains_block = _build_domains_block(tasks)
    fixed_labels_block = "\n".join(
        f"- {s['label']}: {s['description']}" for s in FIXED_SUPER_CATEGORIES
    )
    user_prompt = CATEGORIZE_USER_TMPL.format(
        fixed_labels=fixed_labels_block,
        domains_block=domains_block,
        n_domains=len(domains),
    )
    (out / "raw_prompt.txt").write_text(
        CATEGORIZE_SYSTEM + "\n\n---\n\n" + user_prompt
    )

    from src.infra.llm_client import make_openai_client
    oa = make_openai_client()
    valid_labels = {s["label"] for s in FIXED_SUPER_CATEGORIES}

    say(f"calling {args.model} to assign each of the {len(domains)} "
        "domains to one of 8 fixed super-categories...")
    best_assignments: dict = {}
    for attempt in range(3):
        try:
            raw = _mutator_call(oa, args.model, CATEGORIZE_SYSTEM, user_prompt)
            got = raw.get("assignments") or {}
            # Merge with previous best (keep the largest set of valid entries)
            for d, lbl in got.items():
                if d in set(domains) and lbl in valid_labels:
                    best_assignments[d] = lbl
            missing = sorted(set(domains) - set(best_assignments))
            say(f"attempt {attempt+1}: got {len(best_assignments)}/{len(domains)} "
                f"valid; still missing {len(missing)}")
            if not missing:
                break
        except Exception as e:
            say(f"attempt {attempt+1}: LLM call failed: {e!s}")
        time.sleep(1)

    # Deterministic fallback for any domain the LLM still missed after retries.
    missing = sorted(set(domains) - set(best_assignments))
    if missing:
        say(f"deterministic fallback for {len(missing)} missing domain(s): {missing}")
        for d in missing:
            best_assignments[d] = _fallback_bucket(d)

    assignments = best_assignments
    ok, msg = _validate_assignments(assignments, set(domains), valid_labels)
    if not ok:
        say(f"FATAL: still invalid after fallback: {msg}")
        raise SystemExit(2)
    say(f"final: {msg}")

    # Build the "super_categories" section from the flat assignments
    per_label: dict[str, list[str]] = {s["label"]: [] for s in FIXED_SUPER_CATEGORIES}
    for dom, lbl in assignments.items():
        per_label[lbl].append(dom)
    mapping = {
        "super_categories": [
            {"label": s["label"],
             "description": s["description"],
             "domains": sorted(per_label[s["label"]])}
            for s in FIXED_SUPER_CATEGORIES
        ],
        "raw_assignments": assignments,
    }
    (out / "domains_grouping.json").write_text(
        json.dumps(mapping, indent=2, ensure_ascii=False)
    )

    task_categories: dict[str, str] = {}
    for t in tasks:
        dom = t.get("domain", "unknown") or "unknown"
        task_categories[t["task_id"]] = assignments.get(dom, "uncategorized")

    per_cluster = Counter(task_categories.values())
    (out / "task_categories.json").write_text(
        json.dumps(task_categories, indent=2, ensure_ascii=False)
    )
    (out / "super_category_labels.json").write_text(
        json.dumps(mapping["super_categories"], indent=2, ensure_ascii=False)
    )
    (out / "per_cluster_counts.json").write_text(
        json.dumps(dict(sorted(per_cluster.items(), key=lambda kv: -kv[1])),
                   indent=2, ensure_ascii=False)
    )

    say("=== Per-cluster counts ===")
    for name, cnt in sorted(per_cluster.items(), key=lambda kv: -kv[1]):
        say(f"  {name:30s}  {cnt} tasks")
    say(f"total tasks categorized: {sum(per_cluster.values())} (should be {len(tasks)})")
    say("wrote: task_categories.json + super_category_labels.json + per_cluster_counts.json + domains_grouping.json")
    log.close()


if __name__ == "__main__":
    main()
