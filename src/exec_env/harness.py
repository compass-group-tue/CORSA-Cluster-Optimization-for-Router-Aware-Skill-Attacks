"""Optional SkillsBench utility verifier for evidence from a configured runtime.

This verifier is a subprocess inside an experiment-provided Apptainer image.
It does not launch or isolate the victim. The runtime must prepare the complete
benchmark environment and supply the paths below in evidence.metadata.
"""
from pathlib import Path
import subprocess
from .contracts import UtilityResult


class SkillsBenchUtilityEvaluator:
    def evaluate(self, evidence, task, evaluation_spec):
        meta = evidence.metadata.get('skillsbench_verifier')
        if not meta:
            return UtilityResult('unavailable', reason='Runtime did not supply SkillsBench verifier metadata')
        try:
            paths = {k: Path(meta[k]).resolve() for k in ('image', 'work_root', 'verifier_dir')}
            if not paths['image'].is_file() or not paths['work_root'].is_dir() or not (
                paths['verifier_dir'] / 'test_outputs.py').is_file():
                raise ValueError('Verifier image/workspace/test_outputs.py unavailable')
            interpreter = meta.get('python', 'python3')
            proc = subprocess.run([
                'apptainer', 'exec', '--containall', '--cleanenv', '--no-home',
                '--bind', f"{paths['work_root']}:/root",
                '--bind', f"{paths['verifier_dir']}:/verifier:ro",
                str(paths['image']), interpreter, '-m', 'pytest',
                '/verifier/test_outputs.py', '-rA'], capture_output=True, text=True,
                timeout=int(evaluation_spec.get('verifier_timeout', 300)))
            # pytest: 0 pass, 1 failed tests; collection/config/runtime codes are errors.
            if proc.returncode not in (0, 1):
                return UtilityResult('verifier_error', reason=(proc.stdout + proc.stderr)[-4000:])
            return UtilityResult('measured', proc.returncode == 0, (proc.stdout + proc.stderr)[-4000:])
        except Exception as exc:
            return UtilityResult('verifier_error', reason=f'{type(exc).__name__}: {exc}')
