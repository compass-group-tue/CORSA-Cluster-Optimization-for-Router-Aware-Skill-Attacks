"""Canonical cluster selection and explicit held-out environment mapping."""
import json
from pathlib import Path
from src.infra.paths import default_categories_json

CLUSTERS = ('coding-and-devops', 'data-science-and-analytics', 'engineering-and-simulation',
            'media-and-signals', 'general-tooling-and-web', 'scientific-research',
            'security-and-privacy', 'document-and-office')


def load_categories(path=None):
    categories = json.loads(Path(path or default_categories_json()).read_text())
    if not isinstance(categories, dict) or set(categories.values()) - set(CLUSTERS):
        raise ValueError('Invalid task categorization')
    return categories


def select_cluster(tasks, cluster, categories):
    if cluster not in CLUSTERS:
        raise ValueError(f'Unknown cluster: {cluster}')
    selected = [t for t in tasks if categories.get(t['task_id']) == cluster]
    if not selected:
        raise ValueError(f'No tasks loaded for cluster {cluster}')
    return selected


def heldout_tasks(root, cluster, kind, environment_map=None):
    if cluster not in CLUSTERS or kind not in {'paraphrase', 'synthetic'}:
        raise ValueError('Invalid held-out cluster/set')
    rows = [json.loads(line) for line in (Path(root) / cluster / f'{kind}.jsonl').read_text().splitlines() if line.strip()]
    if not rows:
        raise ValueError('Held-out dataset must contain at least one task')
    mapping = environment_map if environment_map is not None else {}
    if not isinstance(mapping, dict) or any(not isinstance(k, str) or not k or
            not isinstance(v, str) or not v.strip() for k, v in mapping.items()):
        raise ValueError('Environment mapping must map task IDs to nonempty environment strings')
    seen = set()
    for row in rows:
        if not isinstance(row, dict) or any(not isinstance(row.get(k), str) or not row[k].strip()
                for k in ('task_id', 'instruction_text')):
            raise ValueError('Held-out task requires nonempty task_id and instruction_text strings')
        if row['task_id'] in seen:
            raise ValueError('Duplicate held-out task_id')
        seen.add(row['task_id'])
        if kind == 'paraphrase' and (not isinstance(row.get('src'), str) or not row['src'].strip()):
            raise ValueError('Paraphrase task requires a source task ID')
    for row in rows:
        environment = mapping.get(row['task_id']) or (row.get('src') if kind == 'paraphrase' else None)
        if not environment:
            raise ValueError(f"Explicit environment mapping required for {row['task_id']}; no cluster-task substitution is allowed")
        row['environment_id'] = environment
    return rows
