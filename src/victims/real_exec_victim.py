"""Victim adapter: no host process launcher and no implicit execution backend."""
from copy import deepcopy
from src.exec_env.contracts import ExecutionEvidence


class RealExecVictim:
    def __init__(self, experiment, model, scaffold):
        experiment.validate()
        self.experiment = experiment
        self.model = model
        self.scaffold = scaffold

    def execute_task(self, *, task, skill, run_dir):
        evidence = self.experiment.runtime.execute(
            task=deepcopy(task), skill=deepcopy(skill),
            resources=dict(self.experiment.resources), run_dir=run_dir,
            model=self.model, scaffold=self.scaffold,
            config=deepcopy(self.experiment.runtime_config))
        if not isinstance(evidence, ExecutionEvidence):
            raise TypeError('Runtime must return ExecutionEvidence')
        return evidence
