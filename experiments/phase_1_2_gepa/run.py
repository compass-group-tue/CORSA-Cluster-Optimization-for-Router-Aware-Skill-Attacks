"""Portable CORSA curriculum: explicit Stage A and Stage B invocations."""
from __future__ import annotations
import argparse
import json
import math
from pathlib import Path
from src.benchmarks.clusters import CLUSTERS, load_categories, select_cluster
from src.benchmarks.skillrouter_bench import subsample, pool_statistics
from src.exec_env.experiment import load_experiment
from src.exec_env.contracts import ConfigurationError, EvaluatorUnavailable
from src.infra.artifacts import validate_skill, make_winner, stage_b_seed, fingerprint
from src.infra.paths import PROJECT_ROOT, REAL_EXEC_SCRATCH
from src.routers.registry import add_router_arguments, build_router_from_args, get_provenance


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage', choices=['A', 'B'], required=True)
    p.add_argument('--cluster', choices=CLUSTERS, required=True)
    p.add_argument('--categories-json')
    p.add_argument('--seed-skill', help='Stage A: exact JSON file with name, description, body')
    p.add_argument('--stage-a-winner', help='Stage B: versioned stage_a_winner.json')
    p.add_argument('--experiment', help='Stage B: runtime/evaluator configuration JSON')
    p.add_argument('--model', default='gpt-4.1', help='Victim model for the configured runtime')
    p.add_argument('--scaffold', default='experiment-defined')
    p.add_argument('--mutator-model', default='gpt-4.1')
    p.add_argument('--seed', type=int, default=20260714)
    p.add_argument('--n-rounds', type=int, default=15)
    p.add_argument('--minibatch-size', type=int, default=0)
    p.add_argument('--editable-fields', default='name,description,body')
    p.add_argument('--hit-bonus', type=float, default=1.0)
    p.add_argument('--pool-size', type=int, default=2000)
    p.add_argument('--tier', choices=['easy', 'hard'], default='easy')
    p.add_argument('--naturalism-model', default='', help='Optional reported SNS; never execution scoring')
    p.add_argument('--out-dir', required=True)
    p.add_argument('--scratch-dir', default=str(REAL_EXEC_SCRATCH))
    add_router_arguments(p)
    return p.parse_args(argv)


def preflight(args):
    experiment = load_experiment(args.experiment) if args.stage == 'B' else None
    if args.stage == 'A':
        if not args.seed_skill or args.stage_a_winner or args.experiment:
            raise ConfigurationError('Stage A requires --seed-skill; Stage-B inputs are not accepted')
        skill = validate_skill(json.loads(Path(args.seed_skill).read_text()))
        parent = None
    else:
        if not args.stage_a_winner or args.seed_skill:
            raise ConfigurationError('Stage B requires --stage-a-winner and does not accept --seed-skill')
        skill, parent = stage_b_seed(args.stage_a_winner, args.cluster)
    if args.n_rounds < 0 or args.minibatch_size < 0 or args.pool_size < 1 or not math.isfinite(args.hit_bonus):
        raise ConfigurationError('Invalid round/minibatch/pool count')
    fields = args.editable_fields.split(',')
    if not fields or set(fields) - {'name', 'description', 'body', 'payload_framing'}:
        raise ConfigurationError('Invalid editable fields')
    return skill, parent, experiment


def main(argv=None):
    args = parse_args(argv)
    initial, parent, experiment = preflight(args)  # before benchmark/model/API work
    categories = load_categories(args.categories_json)
    tasks, relevance, pool = subsample(999999, args.pool_size, args.seed, tier=args.tier)
    tasks = [{**t, 'environment_id': t['task_id']} for t in select_cluster(tasks, args.cluster, categories)]
    stats = pool_statistics(tasks, relevance, pool, args.tier)
    if parent and parent['task_ids'] != [t['task_id'] for t in tasks]:
        raise ConfigurationError('Stage B task IDs differ from the Stage-A winner')
    router = build_router_from_args(args)
    router_info = get_provenance(router)
    if parent and parent['router'] != router_info:
        raise ConfigurationError('Stage B must use the Stage-A router configuration; use frozen evaluation for transfer')
    if parent and parent['configuration'].get('pool', {}).get('pool_sha256') != stats['pool_sha256']:
        raise ConfigurationError('Stage B must use the Stage-A benchmark pool')
    router.prepare_pool(pool)
    config = {**vars(args), 'task_ids': [t['task_id'] for t in tasks],
              'pool': stats, 'router_provenance': router_info,
              'initial_skill_sha256': fingerprint(initial),
              'experiment': experiment.identity() if experiment else None,
              'parent_artifact_sha256': parent['artifact_sha256'] if parent else None}
    provenance = dict(condition_sha256=fingerprint(config), stage=args.stage,
                      execution='real' if experiment else 'retrieval_only',
                      naturalism_model=args.naturalism_model or None)
    out = Path(args.out_dir).resolve()
    out.mkdir(parents=True, exist_ok=False)
    (out / 'config.json').write_text(json.dumps(config, indent=2))
    from src.infra.llm_client import make_attacker_client, make_openai_client
    from src.infra.logger import PipelineLogger
    from src.attackers.genetic import score_retrieval
    from src.attackers.gepa_cluster_pareto import GepaClusterParetoAttacker
    from src.eval.real_eval import evaluate_task_real
    from src.reporting.tables import aggregate
    from src.payloads.prerequisite import attach_prerequisite, render_skill_md
    log = PipelineLogger(out / 'pipeline.log')
    sns = None
    if args.naturalism_model:
        from src.judges.sns_judge import SNSJudge
        sns = SNSJudge(make_openai_client(), model=args.naturalism_model)
    scored = 0
    def score(skill, task):
        nonlocal scored
        if args.stage == 'A':
            row = score_retrieval(skill, task, router, cluster=args.cluster,
                                  provenance=provenance, hit_bonus=args.hit_bonus)
        else:
            run_root = Path(args.scratch_dir) / fingerprint(config) / str(scored)
            try:
                row = evaluate_task_real(task, args.cluster, skill, router, experiment,
                                         args.model, run_root, scaffold=args.scaffold,
                                         provenance=provenance, measure_utility=False)
            except EvaluatorUnavailable as exc:
                if hasattr(exc, 'observation'):
                    with (out / 'observations.jsonl').open('a') as f:
                        f.write(json.dumps(exc.observation) + '\n')
                raise
            row['optimization_reward'] = row['asr']
        if sns:
            result = sns.score(render_skill_md(attach_prerequisite(skill)))
            row.update(naturalism=result['score'], naturalism_state=result['status'],
                       naturalism_reason=result['reason'])
        scored += 1
        with (out / 'observations.jsonl').open('a') as f:
            f.write(json.dumps(row) + '\n')
        if row['optimization_reward'] is None:
            raise RuntimeError(f"Optimization stopped: {row['execution_state']}; see observations.jsonl")
        return row
    try:
        optimizer = GepaClusterParetoAttacker(openai_client=make_attacker_client(), tasks=tasks,
            initial_skill=initial, score_candidate_task=score, out_dir=out, logger=log,
            n_rounds=args.n_rounds, mutator_model=args.mutator_model, rng_seed=args.seed,
            cluster_name=args.cluster, minibatch_size=args.minibatch_size,
            editable_fields=args.editable_fields.split(','))
        winner, rows = optimizer.optimize()
        summary = aggregate(rows, [args.cluster])
        artifact = make_winner(stage=args.stage, skill=winner, cluster=args.cluster,
            router=router_info, config=config, task_ids=config['task_ids'], source_run=str(out),
            score=summary['micro'], parent=(dict(source_run=parent['source_run'],
                artifact_sha256=parent['artifact_sha256'], skill_sha256=parent['skill_sha256'],
                stage_a_score=parent['score']) if parent else None))
        (out / f'stage_{args.stage.lower()}_winner.json').write_text(json.dumps(artifact, indent=2))
        (out / 'summary.json').write_text(json.dumps(summary, indent=2))
        (out / 'winner_observations.json').write_text(json.dumps(rows, indent=2))
    finally:
        log.close()
    return 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except ConfigurationError as exc:
        raise SystemExit(f'{getattr(exc, "state", "configuration_error")}: {exc}')
