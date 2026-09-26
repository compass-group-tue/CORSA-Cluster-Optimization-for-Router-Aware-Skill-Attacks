# CORSA — Cluster Optimization for Router-Aware Skill Attacks

Official implementation of **“Surviving the Router: Optimizing Skill
Injections for Retrieval and Execution.”**

Previous skill-injection evaluations largely study execution after an injected
skill is already available to the agent. In multi-skill environments, that skill
must first compete with benign skills for retrieval. End-to-end success therefore
requires the injected skill to survive retrieval, be selected, and subsequently
induce payload execution. CORSA explicitly optimizes this competitive retrieval
stage across clusters of related tasks.

**Stage A — Retrieval Optimization** uses router feedback and GEPA to improve one
candidate skill’s Hit@1 across a task cluster. **Stage B — End-to-End Optimization**
starts from the Stage-A winner and continues GEPA optimization using
`Hit(t, S) × Exec(t, S)`: retrieval and execution must succeed on the same task.

```text
Related Task Cluster
        ↓
Stage A: Retrieval Optimization
        ↓
Stage-A Winner
        ↓
Stage B: End-to-End Optimization
        ↓
Stage-B Winner
    Hit@1 × Exec
```

## Installation and Configuration

Use Python 3.10+ and run commands from the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

export OPENAI_API_KEY=YOUR_API_KEY
export SKILLROUTER_REPO=/path/to/skillrouter
export SKILLSBENCH_REPO=/path/to/skillsbench
```

Configure external benchmark repositories, model access, and experiment-specific
dependencies separately. Azure OpenAI is also supported through
`AZURE_OPENAI_ENDPOINT`, `AZURE_OPENAI_API_KEY` (or `AZURE_OPENAI_KEY`), and optional
`AZURE_OPENAI_API_VERSION`; model names must match Azure deployments.

Real-execution experiments require an `experiment.json` specifying an
experiment-provided isolated runtime and execution evaluator. See the
[execution contract](docs/execution_contract.md) for configuration.

## Router and Benchmark

The main experiment uses the **full SkillRouter pipeline**:

- Encoder: `pipizhao/SkillRouter-Embedding-0.6B`
- Reranker: `pipizhao/SkillRouter-Reranker-0.6B`
- Default retrieval window: **20**

The encoder retrieves candidates and the reranker produces the final ranking.
Hit@1 is measured from that final ranking. Transfer experiments also support
BM25 (`bm25`), OAI-Emb-3L (`openai_embedding_3_large`), R3-0.6B (`r3_skill_06`),
and Qwen3-8B (`qwen3_8b_pair`).

SkillRouter’s external benchmark checkout must provide
`data/eval_core/{tasks.jsonl,relevance.json,easy/,hard/}` and `src/common.py`.
Select `--tier easy` or `--tier hard`; Hard includes topically related distractor
skills for a more challenging retrieval setting.

## Task Clusters and Pinned Split

The 75 benchmark tasks are organized into eight semantic clusters. CORSA optimizes
one skill across each cluster. The fixed task-to-cluster split used in the reported
experiments is [task_categories.json](data/task_categorization/task_categories.json).

| Cluster | Tasks |
|---|---:|
| coding-and-devops | 9 |
| data-science-and-analytics | 13 |
| document-and-office | 3 |
| engineering-and-simulation | 12 |
| general-tooling-and-web | 9 |
| media-and-signals | 11 |
| scientific-research | 12 |
| security-and-privacy | 6 |

The clustering implementation is included:

```bash
python -m experiments.section_2_2_categorize.categorize \
  --model YOUR_CATEGORIZATION_MODEL \
  --out_dir results/categorization
```

Regenerated assignments may differ. Use the provided mapping to reproduce the
reported split, or select another mapping with `--categories-json`.

## Stage A — Retrieval Optimization

One candidate competes against the benign skill pool across the selected cluster.
GEPA uses retrieval feedback to improve its ranking. Supply `seed.json` with
`name`, `description`, and `body` fields; `payload_framing` is optional.

```bash
python -m experiments.phase_1_2_gepa.run \
  --stage A \
  --cluster coding-and-devops \
  --seed-skill seed.json \
  --router full_skillrouter \
  --n-rounds 15 \
  --seed 20260714 \
  --out-dir results/stage-a
```

The selected cluster uses the pinned mapping. Stage A optimizes retrieval only
and writes `results/stage-a/stage_a_winner.json`, which seeds Stage B.

## Stage B — End-to-End Optimization

Stage B initializes from the Stage-A winner and continues optimization through
real victim execution with objective `Hit(t, S) × Exec(t, S)`: the skill must be
selected and its associated payload executed on the same task.

```bash
python -m experiments.phase_1_2_gepa.run \
  --stage B \
  --cluster coding-and-devops \
  --stage-a-winner results/stage-a/stage_a_winner.json \
  --experiment experiment.json \
  --model YOUR_VICTIM_MODEL \
  --scaffold YOUR_SCAFFOLD \
  --router full_skillrouter \
  --seed 20260714 \
  --n-rounds 15 \
  --out-dir results/stage-b
```

The Stage-B winner is the final optimized CORSA skill, saved to
`results/stage-b/stage_b_winner.json`.

## Frozen Evaluation and Transfer

Hold the Stage-B winner fixed for downstream evaluation across victim models,
scaffolds, routers, and benchmark settings, without further optimization:

```bash
python -m experiments.real_eval_ourmethod.run \
  --winner results/stage-b/stage_b_winner.json \
  --experiment experiment.json \
  --model YOUR_VICTIM_MODEL \
  --scaffold YOUR_SCAFFOLD \
  --router full_skillrouter \
  --out-dir results/frozen
```

Change `--model`, `--scaffold`, `--router`, or `--tier` while keeping the optimized
skill fixed. The configured runtime must support the selected model and scaffold.
Utility is measured during frozen evaluation when a task verifier is configured.

## Router Transfer

Evaluate the same frozen skill across target routers without re-optimization:

```bash
python -m experiments.router_transfer.run \
  --winner results/stage-b/stage_b_winner.json \
  --target-router bm25 \
  --target-router full_skillrouter \
  --tier hard \
  --out-dir results/router-transfer
```

This produces retrieval/Hit@1 transfer results. Multiple frozen source sets can
form source-to-target matrices using `--sources`. Execution-enabled router
transfer uses `python -m experiments.router_transfer.run_real` with the same
arguments as frozen evaluation above.

## Defense Evaluation

CORSA includes pre-retrieval package-scanning evaluation with Cisco Skill Scanner
and NVIDIA SkillSpector. It reports malicious-package recall and benign-package
false-positive rate (FPR) for researcher-supplied packages.

```bash
python -m experiments.defenses.run --help
```

## Generalization

Frozen optimized skills are evaluated on **paraphrase tasks** (alternative wording
of an original task) and **synthetic tasks** (new tasks from the same semantic
cluster), without further optimization.

Generate tasks:

```bash
MODEL=YOUR_GENERATOR_MODEL \
N_PER_CLUSTER=9 \
python -m run.gen_heldout_tasks
```

Evaluate paraphrases, replacing `<DATE>` with the generated directory’s date
in `YYYYMMDD` format:

```bash
python -m run.score_heldout_realexec \
  --winner results/stage-b/stage_b_winner.json \
  --experiment experiment.json \
  --model YOUR_VICTIM_MODEL \
  --scaffold YOUR_SCAFFOLD \
  --task-set paraphrase \
  --heldout-dir "results_heldout_<DATE>" \
  --out-dir results/paraphrase
```

Use `--task-set synthetic` for synthetic evaluation and supply
`--environment-map environments.json` to map task IDs to prepared runtime environments.

## Metrics

- **Hit@1** — whether the injected skill ranks first.
- **Exec** — whether the associated payload is executed.
- **ASR** — successful retrieval and execution on the same task.
- **Utility** — whether the original user task is completed.
- **Naturalism** — how naturally the injected content fits within the skill.

Optimization rewards are kept separate from reported evaluation metrics.

## Validation

```bash
python -B -m unittest discover -s tests -v
python -B -m experiments.phase_1_2_gepa.run --help
python -B -m experiments.real_eval_ourmethod.run --help
python -B -m experiments.router_transfer.run --help
python -B -m experiments.defenses.run --help
```

## Responsible Release

This repository provides the CORSA optimization, routing, and evaluation pipeline.
Operational attack payloads and generated experimental artifacts are not included
in this release.

## Citation

**“Surviving the Router: Optimizing Skill Injections for Retrieval and Execution.”**
Citation details forthcoming.
