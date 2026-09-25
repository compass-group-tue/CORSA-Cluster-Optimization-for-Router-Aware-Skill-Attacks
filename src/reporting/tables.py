"""Strict aggregation of a single evaluation condition; no legacy inference."""
from collections import Counter, defaultdict
from src.infra.artifacts import fingerprint


def metrics_from_rows(rows):
    if not rows:
        return dict(n_tasks=0, hit_at_1=None, exec=None, asr=None, utility=None,
                    naturalism=None, optimization_reward=None, n_asr_valid=0,
                    n_exec_valid=0, n_utility_measured=0, n_hit_valid=0,
                    execution_states={}, utility_states={})
    signatures = {(r.get('metric_version'), fingerprint(r.get('provenance'))) for r in rows}
    if len(signatures) != 1 or any(not r.get('provenance') or r.get('metric_version') not in
                                 {'corsa-real-v1', 'corsa-retrieval-v1'} for r in rows):
        raise ValueError('Cannot aggregate incompatible or missing metric provenance')
    seen = [(r['cluster'], r['task_id']) for r in rows]
    if len(set(seen)) != len(seen):
        raise ValueError('Duplicate task observations in metric aggregation')
    for r in rows:
        state = r['execution_state']
        valid_exec = state in {'success', 'not_executed'}
        if valid_exec:
            if r.get('hit_at_1') != 1 or r.get('exec') != int(state == 'success') or r.get('asr') != r['exec']:
                raise ValueError('Inconsistent evaluated execution row')
        elif r.get('exec') is not None:
            raise ValueError('Unavailable/error execution cannot be a binary observation')
        if r.get('asr') is not None and not valid_exec and not (
            r.get('metric_version') == 'corsa-real-v1' and r.get('hit_at_1') == 0
            and state == 'not_attempted' and r['asr'] == 0
        ):
            raise ValueError('Invalid ASR observation')
        if r.get('utility_state') != 'measured' and r.get('utility') is not None:
            raise ValueError('Unavailable/error utility cannot be binary')
    def mean(key):
        values = [r[key] for r in rows if r.get(key) is not None]
        return sum(values) / len(values) if values else None
    return dict(n_tasks=len(rows), **{k: mean(k) for k in (
        'hit_at_1', 'exec', 'asr', 'utility', 'naturalism', 'optimization_reward')},
        n_hit_valid=sum(r.get('hit_at_1') is not None for r in rows),
        n_asr_valid=sum(r.get('asr') is not None for r in rows),
        n_exec_valid=sum(r.get('exec') is not None for r in rows),
        n_utility_measured=sum(r.get('utility_state') == 'measured' for r in rows),
        execution_states=dict(Counter(r['execution_state'] for r in rows)),
        utility_states=dict(Counter(r['utility_state'] for r in rows)))


def aggregate(rows, expected_clusters):
    # Validate provenance across clusters before computing macro means.
    overall = metrics_from_rows(rows)
    by = defaultdict(list)
    for row in rows:
        by[row['cluster']].append(row)
    if set(by) != set(expected_clusters):
        raise ValueError('Cluster coverage does not match the declared evaluation scope')
    per = {c: metrics_from_rows(by[c]) for c in sorted(by)}
    macro = {}
    for key in ('hit_at_1', 'exec', 'asr', 'utility', 'naturalism', 'optimization_reward'):
        vals = [v[key] for v in per.values()]
        # Never silently reduce the cluster denominator when a cluster is unavailable.
        macro[key] = sum(vals) / len(vals) if vals and all(v is not None for v in vals) else None
    return dict(per_cluster=per, macro=macro, micro=overall,
                expected_clusters=sorted(expected_clusters),
                denominator_policy='ASR: retrieval misses plus hits with valid execution evaluation; errors excluded and counted')
