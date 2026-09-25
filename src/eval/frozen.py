"""Shared frozen evaluation and strict resume provenance; imports no optimizer."""
from __future__ import annotations
from copy import deepcopy
import json
from pathlib import Path
from src.infra.artifacts import fingerprint, validate_winner, skill_hash
from src.exec_env.contracts import ConfigurationError, EvaluatorUnavailable
from src.eval.real_eval import evaluate_task_real
from src.reporting.tables import aggregate


def frozen_plan(artifacts, tasks, categories, condition):
    by_cluster = {}
    for artifact in artifacts:
        validate_winner(artifact)
        cl = artifact['cluster']
        if cl in by_cluster:
            raise ConfigurationError(f'Duplicate frozen skill for {cl}')
        by_cluster[cl] = artifact
    missing = [t['task_id'] for t in tasks if categories.get(t['task_id']) not in by_cluster]
    if missing:
        raise ConfigurationError(f'Missing frozen skill/cluster for tasks: {missing}')
    if set(by_cluster) != {categories[t['task_id']] for t in tasks}:
        raise ConfigurationError('Frozen skill clusters and task clusters must match exactly')
    ids = [t['task_id'] for t in tasks]
    if not ids or len(ids) != len(set(ids)):
        raise ConfigurationError('Frozen tasks must be nonempty and unique')
    plan = dict(schema_version=1, optimization_enabled=False,
                artifact_hashes={c: a['artifact_sha256'] for c, a in sorted(by_cluster.items())},
                tasks=tasks, categories={tid: categories[tid] for tid in ids}, condition=condition)
    return plan, by_cluster


def evaluate_frozen(*, artifacts, tasks, categories, router, experiment, model,
                    scaffold, out_dir, run_root, condition, resume=False, sns_judge=None):
    experiment.validate()
    # Capture all execution dimensions even for direct callers.
    from src.routers.registry import get_provenance
    condition = {**condition, 'model': model, 'scaffold': scaffold,
                 'router': get_provenance(router), 'experiment': experiment.identity(),
                 'naturalism': getattr(sns_judge, 'label', None)}
    plan, skills = frozen_plan(artifacts, tasks, categories, condition)
    digest = fingerprint(plan)
    provenance = dict(condition_sha256=digest, execution='real', metric_schema='corsa-real-v1')
    out = Path(out_dir)
    if out.exists():
        if not resume or not (out / 'config.json').is_file():
            raise ConfigurationError('Output exists; use --resume only with a matching run')
        if json.loads((out / 'config.json').read_text()) != plan:
            raise ConfigurationError('Resume configuration differs: skills/model/router/scaffold/spec/tasks must match')
    else:
        out.mkdir(parents=True)
        (out / 'config.json').write_text(json.dumps(plan, indent=2))
    rows_path = out / 'rows.jsonl'
    existing = [json.loads(l) for l in rows_path.read_text().splitlines() if l.strip()] if rows_path.exists() else []
    done = {}
    expected = {t['task_id'] for t in tasks}
    for row in existing:
        tid = row['task_id']
        if tid not in expected or tid in done or row.get('provenance') != provenance or row.get('skill_sha256') != skill_hash(skills[categories[tid]]['skill']):
            raise ConfigurationError('Invalid resume row identity/provenance')
        if row.get('execution_state') == 'evaluator_unavailable':
            raise EvaluatorUnavailable('Previous run stopped with an unavailable evaluator; start a new run after correcting configuration')
        done[tid] = row
    for task in tasks:
        tid = task['task_id']
        if tid in done:
            continue
        cl = categories[tid]
        # Callbacks receive copies; the artifact used for routing remains frozen.
        try:
            row = evaluate_task_real(task, cl, deepcopy(skills[cl]['skill']), router, experiment,
                model, Path(run_root) / digest / fingerprint(tid), scaffold=scaffold,
                provenance=provenance)
        except EvaluatorUnavailable as exc:
            if hasattr(exc, 'observation'):
                with rows_path.open('a') as f:
                    f.write(json.dumps(exc.observation) + '\n')
            raise
        if sns_judge:
            from src.payloads.prerequisite import attach_prerequisite, render_skill_md
            nat = sns_judge.score(render_skill_md(attach_prerequisite(skills[cl]['skill'])))
            row.update(naturalism=nat['score'], naturalism_state=nat['status'], naturalism_reason=nat['reason'])
        with rows_path.open('a') as f:
            f.write(json.dumps(row) + '\n')
        done[tid] = row
    summary = aggregate(list(done.values()), sorted(skills))
    summary.update(optimization_enabled=False, condition_sha256=digest)
    (out / 'summary.json').write_text(json.dumps(summary, indent=2))
    return summary
