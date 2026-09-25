"""Evaluate frozen CORSA artifacts, including model/router/scaffold/task transfer."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from src.exec_env.experiment import load_experiment
from src.exec_env.contracts import ConfigurationError
from src.infra.artifacts import load_winner, fingerprint
from src.infra.paths import PROJECT_ROOT, REAL_EXEC_SCRATCH
from src.benchmarks.clusters import load_categories, select_cluster, heldout_tasks
from src.benchmarks.skillrouter_bench import subsample, pool_statistics
from src.routers.registry import add_router_arguments, build_router_from_args, get_provenance
from src.eval.frozen import evaluate_frozen


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--winner', action='append', required=True, help='Exact versioned winner artifact; repeat for multiple clusters')
    p.add_argument('--experiment', required=True)
    p.add_argument('--model', required=True)
    p.add_argument('--scaffold', required=True)
    p.add_argument('--categories-json')
    p.add_argument('--task-set', choices=['original', 'paraphrase', 'synthetic'], default='original')
    p.add_argument('--heldout-dir', help='Required for paraphrase/synthetic: researcher-generated data root')
    p.add_argument('--environment-map', help='JSON task_id -> runtime environment ID; required for synthetic')
    p.add_argument('--pool-size', type=int, default=2000)
    p.add_argument('--tier', choices=['easy', 'hard'], default='easy')
    p.add_argument('--seed', type=int, default=20260714)
    p.add_argument('--naturalism-model', default='')
    p.add_argument('--out-dir', required=True)
    p.add_argument('--scratch-dir', default=str(REAL_EXEC_SCRATCH))
    p.add_argument('--resume', action='store_true')
    add_router_arguments(p)
    args = p.parse_args(argv)
    if args.task_set != 'original' and not args.heldout_dir:
        p.error('--heldout-dir is required for paraphrase/synthetic evaluation')
    return args


def main(argv=None):
    args = parse_args(argv)
    experiment = load_experiment(args.experiment)  # fail before model/pool loading
    artifacts = [load_winner(p) for p in args.winner]
    clusters = [a['cluster'] for a in artifacts]
    if len(set(clusters)) != len(clusters):
        raise ConfigurationError('Supply one frozen artifact per cluster')
    categories = load_categories(args.categories_json)
    mapping = json.loads(Path(args.environment_map).read_text()) if args.environment_map else {}
    selected = []
    if args.task_set != 'original':
        for cl in clusters:
            held = heldout_tasks(args.heldout_dir, cl, args.task_set, mapping)
            selected.extend(held)
            categories.update({t['task_id']: cl for t in held})
    tasks, relevance, pool = subsample(999999, args.pool_size, args.seed, tier=args.tier)
    original = [t for cl in clusters for t in select_cluster(tasks, cl, categories)]
    if args.task_set == 'original':
        selected = [{**t, 'environment_id': t['task_id']} for t in original]
    stats = pool_statistics(original, relevance, pool, args.tier)
    stats.update(n_tasks=len(selected), required_skills_scope='source benchmark tasks',
                 n_source_tasks=len(original))
    router = build_router_from_args(args)
    router.prepare_pool(pool)
    condition = dict(router=get_provenance(router), pool=stats, model=args.model,
                     scaffold=args.scaffold, experiment=experiment.identity(),
                     task_set=args.task_set, seed=args.seed, environment_map=mapping)
    sns = None
    if args.naturalism_model:
        from src.infra.llm_client import make_openai_client
        from src.judges.sns_judge import SNSJudge
        sns = SNSJudge(make_openai_client(), model=args.naturalism_model)
    evaluate_frozen(artifacts=artifacts, tasks=selected, categories=categories,
        router=router, experiment=experiment, model=args.model, scaffold=args.scaffold,
        out_dir=args.out_dir, run_root=args.scratch_dir, condition=condition,
        resume=args.resume, sns_judge=sns)
    return 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except ConfigurationError as exc:
        raise SystemExit(f'{getattr(exc, "state", "configuration_error")}: {exc}')
