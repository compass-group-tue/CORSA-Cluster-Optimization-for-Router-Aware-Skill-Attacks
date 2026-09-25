"""Frozen retrieval matrices using CORSA artifacts, scoring and aggregation."""
from copy import deepcopy
import csv
import json
from pathlib import Path

from src.attackers.genetic import score_retrieval
from src.eval.frozen import frozen_plan
from src.exec_env.contracts import ConfigurationError
from src.infra.artifacts import fingerprint
from src.reporting.tables import aggregate
from src.routers.registry import get_provenance


def validate_comparability(sources, tasks, categories):
    if not sources:
        raise ConfigurationError('At least one source skill set is required')
    scope = {}
    for task in tasks:
        scope.setdefault(categories.get(task['task_id']), set()).add(task['task_id'])
    plans = {}
    for source, artifacts in sources.items():
        if not isinstance(source, str) or not source.strip():
            raise ConfigurationError('Source identifiers must be nonempty strings')
        plan, by_cluster = frozen_plan(artifacts, tasks, categories, {})
        if len({fingerprint(a['router']) for a in artifacts}) != 1:
            raise ConfigurationError(f'Source {source}: mixed source router configurations')
        declared = {cl: set(a['task_ids']) for cl, a in by_cluster.items()}
        if declared != scope:
            raise ConfigurationError(f'Source {source}: winner task scope differs from evaluation scope')
        plans[source] = (plan, by_cluster)
    return plans


def evaluate_router_transfer(*, sources, targets, tasks, categories, pool, out_dir, condition=None):
    # Own immutable snapshots; routers receive disposable copies of skills/pool.
    sources, tasks, pool = deepcopy(sources), deepcopy(tasks), deepcopy(pool)
    plans = validate_comparability(sources, tasks, categories)
    if not targets:
        raise ConfigurationError('At least one target router is required')
    plan = dict(schema_version=1, mode='frozen_retrieval', optimization_enabled=False,
                sources={s: p for s, (p, _) in plans.items()},
                source_router_provenance={s: {a['cluster']: a['router'] for a in arts}
                                          for s, arts in sources.items()},
                target_routers={name: get_provenance(r) for name, r in targets.items()},
                pool_sha256=fingerprint(pool), condition=condition or {})
    digest = fingerprint(plan)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=False)
    (out / 'config.json').write_text(json.dumps(plan, indent=2))
    cells = {s: {} for s in sources}
    with (out / 'rows.jsonl').open('w') as sink:
        for target, router in targets.items():
            router.prepare_pool(deepcopy(pool))
            for source, (_, skills) in plans.items():
                provenance = dict(condition_sha256=digest, source=source, target=target,
                                  execution='retrieval_only')
                rows = []
                for task in tasks:
                    cl = categories[task['task_id']]
                    # Reuse canonical Hit@1 scoring, but never report training reward.
                    row = score_retrieval(deepcopy(skills[cl]['skill']), deepcopy(task), router,
                                          cluster=cl, provenance=provenance)
                    row['optimization_reward'] = None
                    row.update(source=source, target=target,
                               artifact_sha256=skills[cl]['artifact_sha256'])
                    rows.append(row)
                    sink.write(json.dumps(row) + '\n')
                cells[source][target] = aggregate(rows, sorted(skills))
    result = dict(schema_version=1, mode='frozen_retrieval', optimization_enabled=False,
                  condition_sha256=digest, cells=cells,
                  task_ids=[t['task_id'] for t in tasks])
    (out / 'summary.json').write_text(json.dumps(result, indent=2))
    write_matrices(out, cells, list(targets))
    return result


def write_matrices(out, cells, targets):
    for averaging in ('macro', 'micro'):
        with (out / f'hit_at_1_{averaging}.csv').open('w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['source', *targets])
            for source, row in cells.items():
                writer.writerow([source, *[row[t][averaging]['hit_at_1'] for t in targets]])
    # JSON quoting prevents arbitrary identifiers from becoming Markdown markup.
    def label(value):
        return json.dumps(value, ensure_ascii=True).replace('|', '&#124;').replace('<', '&lt;')
    lines = ['# Frozen router transfer', '', 'Macro Hit@1 (equal weight per declared cluster).', '',
             '| Source | ' + ' | '.join(label(t) for t in targets) + ' |',
             '|---|' + '---:|' * len(targets)]
    for source, row in cells.items():
        lines.append('| ' + label(source) + ' | ' + ' | '.join(
            f"{row[t]['macro']['hit_at_1']:.6f}" for t in targets) + ' |')
    (out / 'matrix.md').write_text('\n'.join(lines) + '\n')
