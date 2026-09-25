"""Load explicitly supplied experiment adapters and immutable resource bytes."""
from __future__ import annotations
import importlib
import inspect
import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from src.infra.artifacts import fingerprint
from .contracts import ConfigurationError, EvaluatorUnavailable


def _adapter(reference):
    if not isinstance(reference, str) or ':' not in reference:
        raise ConfigurationError("Adapter must be an importable module:factory")
    module, attr = reference.split(':', 1)
    try:
        factory = getattr(importlib.import_module(module), attr)
        obj = factory()
    except Exception as exc:
        raise ConfigurationError(f"Cannot initialize adapter {reference}: {exc}") from exc
    source = inspect.getsourcefile(factory)
    if not source or not Path(source).is_file():
        raise ConfigurationError("Adapter factory must have inspectable Python source")
    return obj, fingerprint(Path(source).read_text())


@dataclass
class Experiment:
    evaluator: object
    runtime: object
    specification: dict
    resources: dict[str, bytes] = field(default_factory=dict)
    runtime_config: dict = field(default_factory=dict)
    provenance: dict = field(default_factory=dict)
    utility_evaluator: object | None = None

    def identity(self):
        """Snapshot effective settings, including changes by programmatic callers."""
        return dict(declared=self.provenance, evaluation_spec=self.specification,
                    runtime_config=self.runtime_config,
                    evaluator_type=f'{type(self.evaluator).__module__}:{type(self.evaluator).__qualname__}',
                    runtime_type=f'{type(self.runtime).__module__}:{type(self.runtime).__qualname__}',
                    utility_type=f'{type(self.utility_evaluator).__module__}:{type(self.utility_evaluator).__qualname__}',
                    resource_sha256={k: hashlib.sha256(v).hexdigest() for k, v in self.resources.items()})

    def validate(self):
        if self.evaluator is None or not callable(getattr(self.evaluator, 'evaluate', None)):
            raise EvaluatorUnavailable("Execution evaluator required for Stage-B / real-ASR evaluation.")
        if self.runtime is None or getattr(self.runtime, 'isolated', False) is not True:
            raise ConfigurationError("An experiment-provided isolated execution runtime is required; no host fallback exists.")
        if not callable(getattr(self.runtime, 'execute', None)) or not callable(getattr(self.runtime, 'validate', None)):
            raise ConfigurationError("Runtime must implement validate(config) and execute(...)")
        self.runtime.validate(self.runtime_config)
        if not self.provenance:
            raise ConfigurationError("Experiment provenance is required")


def load_experiment(path):
    if not path:
        raise EvaluatorUnavailable("Execution evaluator required for Stage-B / real-ASR evaluation. Supply --experiment.")
    path = Path(path).resolve()
    spec = json.loads(path.read_text())
    if not isinstance(spec, dict):
        raise ConfigurationError('Experiment JSON must be an object')
    if not spec.get('evaluator'):
        raise EvaluatorUnavailable("Execution evaluator required for Stage-B / real-ASR evaluation.")
    if spec.get('schema_version') != 1 or not spec.get('id') or not spec.get('implementation_revision'):
        raise ConfigurationError("Experiment requires schema_version=1, id and implementation_revision")
    if not isinstance(spec.get('evaluation_spec'), dict) or not isinstance(spec.get('runtime_config', {}), dict):
        raise ConfigurationError("evaluation_spec and runtime_config must be objects")
    resource_paths = spec.get('resources', {})
    required = spec.get('required_resources', [])
    if not isinstance(resource_paths, dict) or not isinstance(required, list):
        raise ConfigurationError("resources must be an object; required_resources must be a list")
    resources = {}
    for name, source in resource_paths.items():
        rel = PurePosixPath(name)
        if not name or rel.is_absolute() or '..' in rel.parts or '\\' in name or name == '.':
            raise ConfigurationError(f"Resource destination must be relative: {name!r}")
        if not isinstance(source, str) or not source:
            raise ConfigurationError(f"Resource {name!r} requires a file path")
        p = Path(source)
        p = p if p.is_absolute() else path.parent / p
        if not p.is_file():
            raise ConfigurationError(f"Required resource does not exist: {name!r}")
        resources[name] = p.read_bytes()
    if any(not isinstance(name, str) for name in required):
        raise ConfigurationError('Required resource names must be strings')
    if any(name not in resources for name in required):
        raise ConfigurationError("A required experiment resource is absent")
    evaluator, evaluator_hash = _adapter(spec['evaluator'])
    runtime, runtime_hash = _adapter(spec.get('runtime'))
    utility, utility_hash = (None, None)
    if spec.get('utility_evaluator'):
        utility, utility_hash = _adapter(spec['utility_evaluator'])
        if not callable(getattr(utility, 'evaluate', None)):
            raise ConfigurationError("Utility evaluator must implement evaluate")
    provenance = dict(configuration=spec, evaluator_source_sha256=evaluator_hash,
                      runtime_source_sha256=runtime_hash, utility_source_sha256=utility_hash,
                      resource_sha256={k: hashlib.sha256(v).hexdigest() for k, v in resources.items()})
    experiment = Experiment(evaluator, runtime, spec['evaluation_spec'], resources,
                            spec.get('runtime_config', {}), provenance, utility)
    experiment.validate()
    return experiment
