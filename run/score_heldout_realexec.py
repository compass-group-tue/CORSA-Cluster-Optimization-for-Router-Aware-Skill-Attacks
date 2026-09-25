"""Held-out evaluation entry point; use --task-set paraphrase or synthetic.

Invoke as python -m run.score_heldout_realexec from the repository root.
"""
from experiments.real_eval_ourmethod.run import main

if __name__ == '__main__':
    raise SystemExit(main())
