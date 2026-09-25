"""Frozen real execution through experiment-supplied runtime and evaluator."""
from __future__ import annotations
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
from src.exec_env.contracts import (ExecutionEvidence, ExecutionResult, UtilityResult,
                                    EvaluatorUnavailable, ConfigurationError)
from src.infra.artifacts import skill_hash
from src.payloads.prerequisite import attach_prerequisite

METRIC_VERSION = 'corsa-real-v1'


def evaluate_task_real(task, cluster, skill, router, experiment, model,
                       run_root, *, scaffold, provenance, measure_utility=True):
    # Required configuration must propagate, never become a measured negative.
    if experiment is None:
        raise EvaluatorUnavailable('Execution evaluator required for Stage-B / real-ASR evaluation.')
    experiment.validate()
    original_hash = skill_hash(skill)
    row = dict(task_id=task['task_id'], cluster=cluster, skill_sha256=original_hash,
               metric_version=METRIC_VERSION, provenance=deepcopy(provenance),
               router_rank=None, hit_at_1=None, exec=None, asr=None,
               execution_state='not_attempted', error=None, evidence=None,
               utility=None, utility_state='unavailable', utility_reason='not measured',
               naturalism=None, naturalism_state='unavailable', optimization_reward=None)
    routed = attach_prerequisite(skill)
    try:
        rank = router.rank(task['instruction_text'], deepcopy(routed))
        if rank is not None and (type(rank) is not int or rank < 0):
            raise ValueError('Router rank must be a nonnegative integer or None')
    except Exception as exc:
        row.update(execution_state='router_error', error=f'{type(exc).__name__}: {exc}')
        return row
    row.update(router_rank=rank, hit_at_1=int(rank == 0))
    if rank != 0:
        row.update(asr=0, utility_reason='victim not run after retrieval miss')
        return row
    try:
        from src.victims.real_exec_victim import RealExecVictim
        evidence = RealExecVictim(experiment, model, scaffold).execute_task(
            task=task, skill=routed, run_dir=Path(run_root))
        row['evidence'] = evidence.as_dict()
        if (evidence.status != 'completed' or evidence.error or
                evidence.returncode not in (None, 0)):
            row.update(execution_state='execution_error', error=evidence.error or 'Runtime did not complete successfully')
            return row
    except EvaluatorUnavailable as exc:
        row.update(execution_state='evaluator_unavailable', error=str(exc))
        exc.observation = row
        raise
    except Exception as exc:
        row.update(execution_state='execution_error', error=f'{type(exc).__name__}: {exc}')
        return row
    try:
        verdict = experiment.evaluator.evaluate(deepcopy(evidence), deepcopy(experiment.specification))
        if not isinstance(verdict, ExecutionResult):
            raise TypeError('Evaluator must return ExecutionResult')
        row.update(execution_state=verdict.status, exec=int(verdict.status == 'success'),
                   asr=int(verdict.status == 'success'), evaluation_reason=verdict.reason)
    except EvaluatorUnavailable as exc:
        row.update(execution_state='evaluator_unavailable', error=str(exc))
        exc.observation = row
        raise
    except Exception as exc:
        row.update(execution_state='evaluation_error', error=f'{type(exc).__name__}: {exc}')
    if measure_utility and experiment.utility_evaluator is not None:
        try:
            utility = experiment.utility_evaluator.evaluate(
                deepcopy(evidence), deepcopy(task), deepcopy(experiment.specification))
            if not isinstance(utility, UtilityResult):
                raise TypeError('Utility evaluator must return UtilityResult')
        except Exception as exc:
            utility = UtilityResult('verifier_error', reason=f'{type(exc).__name__}: {exc}')
        row.update(utility_state=utility.status, utility=utility.completed,
                   utility_reason=utility.reason)
    if skill_hash(skill) != original_hash:
        raise RuntimeError('Frozen skill changed during evaluation')
    return row
