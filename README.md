# CORSA — Cluster Optimization for Router-Aware Skill Attacks

Official implementation of **“Surviving the Router: Optimizing Skill
Injections for Retrieval and Execution.”**

CORSA studies whether an injected skill survives both skill retrieval and subsequent
victim execution. One candidate is optimized across a cluster of related tasks. CORSA explicitly
optimizes competitive retrieval: a skill must survive routing against benign
skills before its execution can contribute to end-to-end success. Stage A uses
retrieval feedback in GEPA search; Stage B continues that search from the Stage-A
winner with the valid per-task objective `Hit(t, S) × Exec(t, S)`.

```text
Stage A: retrieval optimization with the real router
    ↓
versioned Stage-A winner
    ↓
Stage B: real victim execution through an experiment-provided isolated runtime
    ↓
runtime evidence
    ↓
experiment-provided ExecutionEvaluator
    ↓
Hit@1 / Exec / ASR, separate from optimization reward
```

This sanitized release intentionally excludes the original operational payload/helper
implementations, generated research artifacts and the private execution evaluator. Execution
success is experiment-specific and must be implemented through the public evaluator
interface. The release alone does **not** reproduce the paper’s original ASR numbers.

## What is supported

- Cluster-aware Pareto/reflective retrieval optimization (Stage A).
- Stage-B architecture using real execution, requiring a supplied runtime/evaluator.
- Frozen model, router, scaffold and held-out task evaluation using the same contracts.
- Easy and Hard benchmark pools, including actual distractor-count reporting.
- Optional real task-verifier utility during frozen evaluation; optional LLM naturalism.
- Explicit errors and unavailable metrics. No plan-based ASR or host-execution fallback.

**Runtime limitation:** this repository does not bundle a validated isolated victim
runtime. Its runtime protocol is an integration boundary, not a security boundary.
The experimenter must supply and validate an isolated runtime that launches the real
victim, provisions the correct task environment, contains its tool execution and
collects evidence. An adapter declaring `isolated = True` is a required assertion,
not proof of isolation. CORSA does not certify third-party adapters.

## Repository

```text
src/
  attackers/       # cluster search and explicit retrieval scoring
  benchmarks/      # cluster selection, pools, held-out environment mapping
  routers/         # full SkillRouter and alternative router adapters
  exec_env/        # evidence/evaluator/runtime contracts and experiment loading
  victims/         # real victim delegation to the configured runtime
  eval/            # real per-task scoring and frozen replay
  judges/          # optional naturalism judgment
  payloads/        # text rendering only; no bundled framing or resources
  reporting/      # strict metric aggregation
  infra/          # paths, artifact schemas, clients and logging
  defenses/       # package scanners/metrics and separate defensive utilities
experiments/
  phase_1_2_gepa/run.py       # explicit --stage A or B
  real_eval_ourmethod/run.py # frozen evaluation and all transfer axes
  router_transfer/           # frozen retrieval matrices and separate real evaluation
  defenses/                 # labeled pre-retrieval package scanning
  section_2_2_categorize/    # optional categorization utility
run/                         # optional generation and held-out evaluation entry points
data/task_categorization/    # pinned public split: 75 task assignments
docs/                       # execution contract
tests/                      # benign, dependency-light contract tests
```

## Installation and configuration

Python 3.10+ is required. From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export OPENAI_API_KEY=...
export SKILLROUTER_REPO=/your/checkouts/skillrouter
export SKILLSBENCH_REPO=/your/checkouts/skillsbench
```

External benchmark repositories and their data are not vendored. Obtain them
separately through their public distributions. SkillRouter must provide
`src/common.py` and `data/eval_core/{tasks.jsonl,relevance.json,easy/,hard/}`.
The configured real runtime is responsible for SkillsBench task images, inputs and
dependencies; pulling only a Dockerfile’s base image is not equivalent to building
the task environment. Record the external repository/data revisions in your experiment.
No automatic benchmark downloads are performed by the CORSA entry points.

Instead of direct OpenAI access, set `AZURE_OPENAI_ENDPOINT`, `AZURE_OPENAI_API_KEY`
(or `AZURE_OPENAI_KEY`) and optionally `AZURE_OPENAI_API_VERSION`. Azure model values
must name your deployments. No institution-specific credential files are read.
`ATTACKER_API_BASE` and `ATTACKER_API_KEY` optionally configure the mutator separately.

`CORSA_ROOT` overrides the auto-detected repository root. `CORSA_SCRATCH` defaults to
`.corsa_scratch/`; `HF_HOME` controls model caching. Explicit `--out-dir` and optional
`--scratch-dir` select run locations. Experiment adapters may require additional
packages and system tools; these are not installed by CORSA. Gemini routing requires
optional `google-genai`. The optional SkillsBench utility adapter requires Apptainer
and an already prepared image with Python/pytest and task dependencies.

## Router and benchmark

The default is the **full** SkillRouter pipeline:

- Encoder: `pipizhao/SkillRouter-Embedding-0.6B`.
- Reranker: `pipizhao/SkillRouter-Reranker-0.6B`.
- Default retrieval window: 20 (`--router_retrieval_top_k`).
- Hit@1 is based on the final reranking, not encoder ranking alone.

`src/routers/registry.py` lists alternative `--router` values. Checkpoint and cache
options use the underscore spellings shown by `--help`.

`--tier easy|hard` is explicit; invalid values fail. Hard sampling retains every
loaded distractor and fails if none are present. Configurations record actual task,
ordinary benign-skill, required-skill, distractor, pool and total-candidate counts.
Required skills overlap with the pool; these counts are not additive.
Both all retained benchmark-required skills and selected-task required skills are reported. The total
candidate count is the pool plus one candidate. No fixed distractor count is assumed.
The pool may exceed the requested size to retain required skills and distractors.
Pool content is fingerprinted and its ordering is seeded from a canonical ordering.

## Task clustering and pinned split

`data/task_categorization/task_categories.json` is public experiment configuration:
it pins the 75 benchmark tasks to the eight clusters used in the reported experiments:

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

The categorization implementation groups public benchmark tasks by domain, supplies
example task descriptions to an LLM, validates assignments to the eight fixed labels,
and uses deterministic keyword fallback for missing assignments. To generate your
own mapping without replacing the pinned split:

```bash
python -m experiments.section_2_2_categorize.categorize \
  --model YOUR_CATEGORIZATION_MODEL --out_dir results/categorization
```

This writes the mapping, domain grouping, counts, labels and prompt/log output under
`results/categorization/`. Regeneration does not guarantee the identical split.
Use `--categories-json results/categorization/task_categories.json` to select an
alternative mapping for optimization/evaluation explicitly.

## Stage A

Supply one JSON seed containing `name`, `description` and `body`; optional
`payload_framing` is text supplied by your experiment. CORSA inserts no placeholder
framing and creates no implicit resources. A directory of arbitrary JSON files is
not accepted.

```bash
python -m experiments.phase_1_2_gepa.run \
  --stage A --cluster coding-and-devops \
  --seed-skill seed.json --router full_skillrouter \
  --n-rounds 15 --seed 20260714 --out-dir results/stage-a
```

Selecting a cluster automatically uses `data/task_categorization/task_categories.json`.
Override it with `--categories-json`. Every available benchmark task in that cluster
is selected; effective task IDs are recorded. Stage A has no victim execution.

Reward is the router’s retrieval margin plus `--hit-bonus × Hit@1` (default 1.0).
The margin is a shaping term and reward is not ASR. Search uses per-task leaders,
round-robin field mutation and eligible field merges; full-cluster score improvement
controls acceptance. The RNG receives `--seed`; API-generated mutations are not
claimed deterministic. Output: `results/stage-a/stage_a_winner.json`.

## Stage B

First provide an experiment JSON and importable runtime/evaluator factories. The
full contract is in [docs/execution_contract.md](docs/execution_contract.md).
There is deliberately no working attack configuration bundled with the release.

```bash
python -m experiments.phase_1_2_gepa.run \
  --stage B --cluster coding-and-devops \
  --stage-a-winner results/stage-a/stage_a_winner.json \
  --experiment experiment.json --model YOUR_VICTIM_MODEL --scaffold YOUR_SCAFFOLD \
  --router full_skillrouter --seed 20260714 --n-rounds 15 \
  --out-dir results/stage-b
```

Stage B validates the Stage-A schema, content/provenance hashes, cluster, task IDs,
router and pool. Its initial candidate is the Stage-A winner. Stage-B reward is
valid per-task end-to-end success; no execution prediction is used. On an invalid
observation, it saves the observation and stops rather than optimizing a fake zero.
Output: `results/stage-b/stage_b_winner.json`.

`--editable-fields payload_framing` enables framing-only mutation while preserving
all other fields. The complete candidate, including framing, participates in cache
identity. No resources or evaluation specifications are mutated by the optimizer.

Training does **not** evaluate real user utility. Frozen evaluation supports it.
`--naturalism-model YOUR_JUDGE_MODEL` optionally reports naturalism independently;
it never supplies execution success or changes the optimization reward.

## Frozen evaluation and transfer

```bash
python -m experiments.real_eval_ourmethod.run \
  --winner results/stage-b/stage_b_winner.json \
  --experiment experiment.json --model YOUR_VICTIM_MODEL --scaffold YOUR_SCAFFOLD \
  --router full_skillrouter --out-dir results/frozen
```

Repeat `--winner` for additional clusters. Each cluster must have exactly one
versioned winner; no private artifacts are shipped. Frozen evaluation never calls
an optimizer and never changes the skill. The optional framing is rendered once
for both routing and execution from the unchanged artifact representation.

- **Model transfer:** change `--model` to a model supported by your runtime.
- **Router transfer:** change `--router`, or use `python -m experiments.router_transfer.run_real`
  with the same flags. Retrieval-only matrices have a separate CLI below.
- **Scaffold transfer:** change `--scaffold` to one supported and isolated by your runtime.
- **Hard routing:** add `--tier hard`.
- **Resume:** add `--resume` only for the same condition. Skills, tasks, environments,
  pool, model, router, scaffold, experiment specification, adapter source hashes and
  resource hashes must match. Existing error rows are retained; a new experiment
  requires a new output directory. No best-of-retries success selection occurs.

Outputs are `config.json`, evidence-bearing `rows.jsonl` and `summary.json`.
A run fingerprint prevents accidental mixing; it is not a cryptographic attestation
of adapter behavior. Evidence may be sensitive: use the documented ignored output directories and
review any custom output location before publication.

## Frozen router-transfer matrices (retrieval only)

The existing router implementations are reused: `bm25`,
`openai_embedding_3_large`, `full_skillrouter` (SR encoder + reranker),
`r3_skill_06` (R3 encoder + reranker), and `qwen3_8b_pair` (Qwen3 8B encoder +
reranker). A frozen winner is never re-optimized for a target router.

```bash
python -m experiments.router_transfer.run \
  --winner results/stage-b/stage_b_winner.json \
  --target-router bm25 --target-router full_skillrouter \
  --tier hard --out-dir results/router-transfer
```

This requires no victim, runtime, or ExecutionEvaluator. Repeat `--winner` for
clusters in one source set. For multiple source sets, replace `--winner` with
`--sources sources.json`: a JSON object mapping source-set identifiers to lists
of versioned winner paths, resolved relative to that manifest. Source router
provenance is read from the winners, not inferred from the source-set label.
All source sets must cover exactly the same clusters and task IDs; each winner's
task scope must match the selected public benchmark tasks. All cells share one
fixed competitive pool. Target configuration overrides apply to every target;
omit checkpoint overrides to use each registered router's native pair.

Outputs include `config.json`, `rows.jsonl`, `summary.json`,
`hit_at_1_macro.csv`, `hit_at_1_micro.csv`, and `matrix.md`. Macro Hit@1 weights
clusters equally; micro Hit@1 weights tasks equally. Router errors stop the run
without a completed summary. Source execution observations are never used.
The implementation reuses retrieval scoring, including its margin calculation,
but clears optimization reward in reported rows. No ASR is measured in this mode.
Runs require a fresh output directory; matrix resume is not implemented.

For execution-enabled transfer, use the existing separate path:

```bash
python -m experiments.router_transfer.run_real \
  --winner results/stage-b/stage_b_winner.json --router bm25 \
  --experiment experiment.json --model YOUR_VICTIM_MODEL --scaffold YOUR_SCAFFOLD \
  --out-dir results/router-transfer-real
```

This delegates to the shared frozen real evaluator and retains its runtime,
ExecutionEvaluator, error-state and ASR contracts.

## Pre-retrieval package detection

This separate experiment scans complete researcher-supplied packages before
retrieval. It does not invoke a victim or compute execution metrics. Neither
malicious packages nor saved scanner results are included in the release.

Provide a JSON manifest containing a nonempty list of records with exactly:
`sample_id` (unique string), `package_path` (relative to the manifest or absolute),
and `label` (`malicious` or `benign`). Labels are ground truth for aggregation;
they are never supplied to scanner adapters. Each package must contain a top-level
`SKILL.md`. Nested packages, symlinks and non-regular objects are rejected.
The complete package is hashed and copied to temporary storage; its copied hash
must match before scanning. No package files are generated by CORSA.

Install scanners separately in your chosen environment. CORSA does not install
or alter them. Executables resolve through PATH or `--scanner-bin`.
The adapters require Cisco Skill Scanner **2.1.0** or NVIDIA SkillSpector **2.11.2**.
They record expected source revisions, configuration and observed tool version;
a matching version string alone does not attest to the installed source revision.

```bash
python -m experiments.defenses.run \
  --manifest packages.json --scanner cisco --profile static \
  --out-dir results/cisco-static

python -m experiments.defenses.run \
  --manifest packages.json --scanner skillspector --profile static \
  --out-dir results/skillspector-static
```

Cisco uses its native scan with `--use-behavioral`. LLM mode additionally uses
`--use-llm --llm-provider openai-compatible`. A package is flagged only if at least
one finding is HIGH or CRITICAL. Cisco LLM validation checks analyzer presence and
positive token usage; it does not prove completion of every intended LLM call.

SkillSpector static mode uses `--no-llm`. LLM mode omits that flag and validates
LLM availability, non-degraded call accounting and provider token usage. Its rule
is **risk_score > 50**; exactly 50 is benign. Static scanning retains native
network/OSV behavior and is not an offline guarantee. A scanner completion field
refers to completion of scanning, not execution of the supplied package.

Retained LLM profiles are `qwen3.8-27b` and `gpt-5.4`. Configure a compatible
backend URL ending in `/v1` and a credential environment variable:

```bash
python -m experiments.defenses.run \
  --manifest packages.json --scanner skillspector --profile gpt-5.4 \
  --base-url "$SCANNER_BASE_URL" --api-key-env OPENAI_API_KEY \
  --out-dir results/skillspector-llm
```

The Qwen profile selects `Qwen/Qwen3.8-27B`, with context/output limits
65,536/16,384; the GPT profile selects `openai/gpt-5.4`, with limits
1,050,000/128,000 and low reasoning effort. These are experiment profile settings,
not claims about every backend's capabilities. SkillSpector receives a temporary
model registry. Cisco uses a 4,096-token output setting. Credentials are not saved.

Qwen uses a local compatibility transport to set `enable_thinking=false`, as
required by the retained profile. For Cisco backends rejecting uniqueness in the
strict evidence-ID schema, opt in with `--schema-compat`: it removes only
`uniqueItems` on `evidence_ids` array properties. Other schema constraints and
message content remain unchanged. It is not enabled for GPT by default; backend
requirements have not been live-tested. Transport supports non-streaming chat
completions only. It is a protocol adapter, not an isolation boundary.

**Invalid scanner outputs are not flagged and remain in denominators.** There is
one scan attempt per package, with no retry-until-valid policy:

- Recall = flagged malicious packages / all malicious packages evaluated.
- FPR = flagged benign packages / all benign packages evaluated.
- Invalid malicious results are false negatives; invalid benign results are not
  false positives. Valid/invalid counts, coverage and invalid rate are separate.
- A class with no evaluated packages has a null rate, not zero.

Malformed reports, scanner failures, timeouts and tool-version failures produce
invalid rows. Invalid configuration or malformed input packages fail preflight.
This denominator policy is specific to package detection and differs from the
execution-evaluation policy below. Results are written to `scanner_rows.jsonl`
and `summary.json`, with a run configuration. Use ignored `results/` locations.
The existing CoT monitor and instruction-level defense remain separate utilities.

## Metrics and errors

| Field | Meaning and denominator |
|---|---|
| `hit_at_1` | Final rank is zero; rate over successfully routed tasks |
| `exec` | 1 for evaluated success, 0 for evaluated non-execution; otherwise null |
| `asr` | Hit AND valid execution success on the same task; retrieval misses are known zero |
| `utility` | Boolean only when a real verifier measured it; otherwise null |
| `naturalism` | Optional SNS judgment, not execution evidence |
| `optimization_reward` | Optimization objective, always separate from metrics |

ASR denominator = retrieval misses + hits with valid execution evaluations. Router,
runtime and evaluator failures are excluded and counted explicitly. Exec rate is
conditional on valid evaluated hit attempts. Coverage and all execution-state counts
are reported; an incomplete/error-bearing run must not be presented as complete
benchmark performance. A missing evaluator fails before model loading or execution.

Execution states: `success`, `not_executed`, `execution_error`, `evaluator_unavailable`,
`evaluation_error`, `not_attempted`, and `router_error`. Missing evaluator configuration
is a preflight failure with no attempted task rows. Utility states: `measured`,
`unavailable`, `verifier_error`. Missing utility is never zero. On retrieval misses,
the victim is not run, so utility of the alternative benign routed skill is unknown.

Summaries include per-cluster metrics, macro means over the declared cluster scope,
and separately labeled micro means. If a cluster has no valid observations for a
metric, that macro metric is null rather than silently averaging fewer clusters.
Only a declared eight-cluster run is an eight-cluster macro result. Aggregation
rejects missing/incompatible provenance and legacy metrics.

## Generate and evaluate generalization tasks

Generated task datasets are not shipped. The generation code supports:

- **Paraphrase:** alternate wording of a source task while requesting preservation
  of its goal and requirements; `src` retains the source environment identity.
- **Synthetic:** new tasks in the same semantic cluster, using public benchmark
  tasks as examples. Each task needs a correctly provisioned environment and inputs.

Generate your own data with the pinned cluster split:

```bash
MODEL=YOUR_GENERATOR_MODEL N_PER_CLUSTER=9 CLUSTERS=coding-and-devops \
  python -m run.gen_heldout_tasks
```

Omit `CLUSTERS` to generate for every available cluster. The script writes both
`paraphrase.jsonl` and `synthetic.jsonl` under
`results_heldout_<YYYYMMDD>/<cluster>/`. `N_PER_CLUSTER` is a requested count, not
a guarantee: inspect actual outputs and generation failures. Generation uses API
access and is not deterministic. The current paraphrase generator truncates source
instructions to 1,500 characters; validate preservation of the full task before
using generated wording. Generated synthetic text alone is not an executable task
environment. Generated outputs are intentionally not version-controlled.

Evaluate a frozen winner against an explicit researcher-generated data directory:

```bash
python -m run.score_heldout_realexec \
  --winner results/stage-b/stage_b_winner.json \
  --experiment experiment.json --model YOUR_VICTIM_MODEL --scaffold YOUR_SCAFFOLD \
  --task-set paraphrase --heldout-dir "results_heldout_<YYYYMMDD>" \
  --out-dir results/paraphrase
```

Replace `results_heldout_<YYYYMMDD>` with your actual output directory (including
in the shell command). Supply additional `--winner` arguments for additional
clusters. For synthetic evaluation, use `--task-set synthetic` and
`--environment-map environments.json`, mapping each selected task ID to its
correctly provisioned runtime environment. Missing mappings fail; arbitrary
cluster-task substitution is prohibited. Evaluation never re-optimizes the skill.
`--heldout-dir` is required for either generated task set; there is no bundled-data
default. Each JSONL record requires unique `task_id` and nonempty `instruction_text`;
paraphrases also require a nonempty `src` source task ID.

## Runtime outputs and release limits

Use `results/` for optimization, categorization and frozen evaluation. Generation
uses `results_heldout_<YYYYMMDD>/`; runtime scratch defaults to `.corsa_scratch/`.
These directories are created when needed and ignored by Git, alongside named
winner/candidate/observation products. No empty output placeholders are shipped.
For custom output locations, verify Git exclusions yourself: arbitrary adapter
outputs cannot be covered automatically. Generic configuration files remain trackable.

Optimization emits complete versioned winners, per-round candidates/scores,
observations and summaries. Frozen evaluation emits configuration, evidence-bearing
rows and summaries. These are researcher-generated runtime products, not public
research artifacts bundled with this release.

Naturalism is reported separately from optimization reward. The library retains
isolation and context-fit scoring; the CLI uses isolation scoring. The primary
objective is retrieval reward in Stage A and valid Hit × Exec in Stage B;
stealthiness-reward ablations are not implemented.
The supplied runtime/evaluator, external benchmark revisions, task environments,
model access and researcher-generated artifacts remain necessary for execution
experiments; this release alone cannot reproduce the original ASR results.

## Validation

```bash
python -B -m unittest discover -s tests -v
python -B -m experiments.phase_1_2_gepa.run --help
python -B -m experiments.real_eval_ourmethod.run --help
```

Tests use only benign in-memory runtime doubles and temporary files. They neither
launch victim agents nor demonstrate isolation of a production runtime. Neural
router inference, external task environments and user-supplied adapters require
separate validation with your configured dependencies and infrastructure.
