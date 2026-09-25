"""Public execution contracts. No experiment-specific success detector is supplied."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Protocol


class ConfigurationError(ValueError):
    pass


class EvaluatorUnavailable(ConfigurationError):
    state = "evaluator_unavailable"


@dataclass
class ExecutionEvidence:
    """Runtime observations, not an agent's prediction of future execution.

    status is completed or execution_error. A completed run may still evaluate
    to not_executed. Plugins must report tool/runtime failures honestly.
    """
    status: str
    commands: list[str] = field(default_factory=list)
    tool_calls: list[dict] = field(default_factory=list)
    stdout: str = ""
    stderr: str = ""
    filesystem_observations: list[dict] = field(default_factory=list)
    events: list[dict] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    returncode: int | None = None
    error: str | None = None

    def as_dict(self):
        return asdict(self)


@dataclass(frozen=True)
class ExecutionResult:
    """Only these two evaluated outcomes may become binary Exec observations."""
    status: str  # success | not_executed
    reason: str = ""

    def __post_init__(self):
        if self.status not in {"success", "not_executed"}:
            raise ValueError("ExecutionResult must be success or not_executed")


@dataclass(frozen=True)
class UtilityResult:
    status: str  # measured | unavailable | verifier_error
    completed: bool | None = None
    reason: str = ""

    def __post_init__(self):
        if self.status not in {"measured", "unavailable", "verifier_error"}:
            raise ValueError("Invalid utility status")
        if (self.status == "measured" and type(self.completed) is not bool) or (
            self.status != "measured" and self.completed is not None
        ):
            raise ValueError("Only measured utility has a boolean completed value")


class ExecutionEvaluator(Protocol):
    def evaluate(self, evidence: ExecutionEvidence, evaluation_spec: dict) -> ExecutionResult: ...


class ExecutionRuntime(Protocol):
    """Experiment-provided isolation boundary; CORSA does not certify it."""
    isolated: bool

    def validate(self, config: dict) -> None: ...

    def execute(self, *, task: dict, skill: dict, resources: dict[str, bytes],
                run_dir: Path, model: str, scaffold: str,
                config: dict) -> ExecutionEvidence: ...


class UtilityEvaluator(Protocol):
    def evaluate(self, evidence: ExecutionEvidence, task: dict,
                 evaluation_spec: dict) -> UtilityResult: ...
