"""Frozen retrieval-only transfer; real execution uses experiments.real_eval_ourmethod.run."""
import argparse
import json
from pathlib import Path
from src.benchmarks.clusters import load_categories, select_cluster
from src.benchmarks.skillrouter_bench import subsample, pool_statistics
from src.eval.router_transfer import evaluate_router_transfer, validate_comparability
from src.infra.artifacts import load_winner
from src.exec_env.contracts import ConfigurationError
from src.routers.registry import ROUTER_IDS, add_router_arguments, build_router


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument('--winner', action='append', help='One source set; repeat for clusters')
    inputs.add_argument('--sources', help='JSON object: source identifier -> list of winner paths, relative to manifest')
    parser.add_argument('--target-router', action='append', choices=ROUTER_IDS, required=True)
    parser.add_argument('--categories-json')
    parser.add_argument('--tier', choices=['easy', 'hard'], default='easy')
    parser.add_argument('--pool-size', type=int, default=2000)
    parser.add_argument('--seed', type=int, default=20260714)
    parser.add_argument('--out-dir', required=True)
    add_router_arguments(parser)
    # Matrix targets are selected by --target-router, never the single-router flag.
    parser.set_defaults(router=None)
    args = parser.parse_args(argv)
    if args.router is not None:
        parser.error('Use --target-router for retrieval matrices; --router belongs to experiments.real_eval_ourmethod.run')
    return args


def main(argv=None):
    args = parse_args(argv)
    if len(set(args.target_router)) != len(args.target_router):
        raise ConfigurationError('Duplicate target router')
    if args.pool_size < 1:
        raise ConfigurationError('Pool size must be positive')
    if args.sources:
        manifest = Path(args.sources)
        raw = json.loads(manifest.read_text())
        if not isinstance(raw, dict) or not raw or any(
                not isinstance(v, list) or not v or any(not isinstance(p, str) for p in v)
                for v in raw.values()):
            raise ConfigurationError('Sources must map source IDs to nonempty lists of winner paths')
        sources = {s: [load_winner(manifest.parent / p) for p in paths] for s, paths in raw.items()}
    else:
        sources = {'source': [load_winner(p) for p in args.winner]}
    categories = load_categories(args.categories_json)
    tasks, relevance, pool = subsample(999999, args.pool_size, args.seed, tier=args.tier)
    clusters = sorted({a['cluster'] for arts in sources.values() for a in arts})
    tasks = [t for cl in clusters for t in select_cluster(tasks, cl, categories)]
    validate_comparability(sources, tasks, categories)
    stats = pool_statistics(tasks, relevance, pool, args.tier)
    targets = {name: build_router(name, encoder_model=args.router_encoder_model,
        reranker_model=args.router_reranker_model, retrieval_top_k=args.router_retrieval_top_k,
        embedding_dim=args.router_embedding_dim, cache_dir=args.router_cache_dir)
        for name in args.target_router}
    return evaluate_router_transfer(sources=sources, targets=targets, tasks=tasks,
        categories=categories, pool=pool, out_dir=args.out_dir,
        condition={'seed': args.seed, 'pool': stats})


if __name__ == '__main__':
    main()
