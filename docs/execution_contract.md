# Execution integration contract

`src/exec_env/contracts.py` defines `ExecutionRuntime`, `ExecutionEvidence`,
`ExecutionEvaluator`, `ExecutionResult` and optional `UtilityEvaluator`/`UtilityResult`.
No private success detector or operational resource is supplied.

## Experiment configuration

An experimenter supplies JSON with this shape (adapter names below are illustrative,
not installed modules):

```json
{
  "schema_version": 1,
  "id": "my-controlled-experiment",
  "implementation_revision": "my-adapters-and-environment-revision",
  "runtime": "my_experiment.runtime:Runtime",
  "runtime_config": {},
  "evaluator": "my_experiment.evaluation:Evaluator",
  "evaluation_spec": {},
  "resources": {},
  "required_resources": []
}
```

Factories are imported and called with no arguments. They must come from inspectable
Python source. These are trusted experimenter-supplied code, not skill-controlled
configuration. Never load an experiment file from an untrusted skill.

`resources` maps safe relative resource names to existing files (paths relative to
the experiment JSON or absolute paths). CORSA reads bytes and records their hashes;
the runtime materializes them inside its isolated environment. Null/missing file paths
and absent declared required resources are configuration errors. An experiment with
no resources explicitly uses `{}` and `[]`; no fake executable is generated.

`implementation_revision` must identify transitive adapter code, runtime/image versions,
benchmark versions and any behavior not captured by the direct factory source hashes.
Changing adapters, resource bytes or configuration changes the resume fingerprint.
Do not place credentials in this JSON: configurations are persisted. Use environment
variables or your runtime’s credential provisioning instead.

## Runtime

A runtime must expose `isolated = True`, `validate(config)` and:

```python
execute(*, task, skill, resources, run_dir, model, scaffold, config) -> ExecutionEvidence
```

`validate` must reject incomplete/unsupported isolation configuration. The runtime must
validate the requested model/scaffold/environment, provision actual task inputs and
run the real victim with contained tool execution. `task["environment_id"]` identifies
the environment; original tasks use their task ID and paraphrases their source ID.
The skill contains text already rendered with optional framing exactly once.
Resources are immutable byte values indexed by relative paths.

There is no default host launcher. `run_dir` is an output location, not containment.
Declaring isolation is not evidence of isolation; deployment validation remains the
experimenter’s responsibility. Do not pass a planning or simulated backend as a real
runtime. Tests use explicitly labeled doubles solely to validate software contracts.

Evidence contains status, return code, errors, commands/tool calls, stdout/stderr,
filesystem observations, structured events and other metadata. Status `completed`
means the execution attempt completed, not that the experiment objective succeeded.
A nonzero return code or `execution_error` is not an observed attack failure.
Evidence must be JSON-serializable and must not expose runtime credentials.

## Execution evaluation

```python
evaluate(evidence, evaluation_spec) -> ExecutionResult
```

The evaluator derives its decision from actual runtime observations.
`ExecutionResult("success", reason)` and `ExecutionResult("not_executed", reason)`
are the only valid binary observations. Raw booleans, malformed responses and exceptions
are evaluation errors, not zero. An evaluator becoming unavailable raises
`EvaluatorUnavailable`; this aborts scoring rather than producing an ASR number.

The evaluator never receives a proposed plan as evidence of execution. Configuration
is checked before expensive work. Stage-B scoring stops on invalid observations;
frozen replay preserves invalid rows and reports coverage and error counts.

## Utility

Optional `utility_evaluator: "module:Factory"` supplies:

```python
evaluate(evidence, task, evaluation_spec) -> UtilityResult
```

Use a real task verifier. States are measured (boolean completion), unavailable
(null) and verifier_error (null). Stage-B training does not invoke utility evaluation.
Frozen evaluation does so after a completed victim run, even when execution evaluation
fails. Utility remains unavailable when no verifier is supplied or retrieval misses.

`src.exec_env.harness:SkillsBenchUtilityEvaluator` is an optional Apptainer verifier
adapter. It expects `evidence.metadata["skillsbench_verifier"]` with `image`,
`work_root`, `verifier_dir`, and optional `python`. The runtime must provide a complete
prepared image and arrange task outputs for the verifier’s expected `/root` paths.
Other benchmark layouts require a custom utility adapter. Pytest code 0 is pass,
code 1 is a measured test failure, and other codes/timeouts are verifier errors.
This utility adapter does not isolate the victim and has not been validated here
against external task images.

## Winner artifact

`stage_a_winner.json` and `stage_b_winner.json` use:

- `schema_version: 1`, `kind: "corsa_winner"`, `stage: "A" | "B"`.
- `skill`: name, description, body, optional payload_framing and canonical skill ID.
- `skill_sha256`: canonical content hash including framing.
- `cluster`, ordered `task_ids`, `router` provenance, `configuration` including seed,
  effective models, pool fingerprint and scoring settings.
- `source_run`: source output directory.
- `score`: separately named Hit@1, execution/ASR/utility/naturalism and optimization reward.
- `parent`: null for A; Stage-A source run, artifact hash, skill hash and score for B.
- `artifact_sha256`: hash of the entire artifact excluding this hash field.

Hashes detect accidental content/provenance mismatches; they do not authenticate an
untrusted artifact. Stage B accepts an exact artifact file and verifies stage, cluster,
task IDs, router and pool before optimization. No arbitrary JSON-directory scanning occurs.
