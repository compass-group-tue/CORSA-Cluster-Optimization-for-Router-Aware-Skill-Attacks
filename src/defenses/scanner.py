"""Shared native scanner transport/configuration, unrelated to victim execution."""
from contextlib import nullcontext
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
from urllib.parse import urlsplit
from .packages import copy_full_package
from src.infra.artifacts import fingerprint
from .scanner_transport import compatibility_endpoint

PROFILES = {'qwen3.8-27b': 'Qwen/Qwen3.8-27B', 'gpt-5.4': 'openai/gpt-5.4'}


@dataclass(frozen=True)
class ScannerConfig:
    scanner: str
    profile: str = 'static'
    executable: str | None = None
    base_url: str | None = None
    api_key_env: str = 'OPENAI_API_KEY'
    timeout: float = 900
    schema_compat: bool = False

    def validate(self):
        if self.scanner not in ('cisco', 'skillspector'):
            raise ValueError('Unknown scanner')
        if self.profile not in ('static', *PROFILES):
            raise ValueError('Unknown scanner backend profile')
        if not math.isfinite(self.timeout) or self.timeout <= 0:
            raise ValueError('Scanner timeout must be finite and positive')
        if self.profile != 'static':
            url = urlsplit(self.base_url or '')
            if url.scheme not in ('http', 'https') or not url.netloc or url.username or url.password or url.query or url.fragment or not url.path.rstrip('/').endswith('/v1'):
                raise ValueError('LLM mode requires a credential-free HTTP(S) --base-url ending in /v1')
            if not os.environ.get(self.api_key_env):
                raise ValueError(f'Set backend credential environment variable {self.api_key_env}')
        if self.schema_compat and (self.scanner != 'cisco' or self.profile == 'static'):
            raise ValueError('Schema compatibility is only available for Cisco LLM profiles')

    def provenance(self):
        return dict(scanner=self.scanner, profile=self.profile, model=PROFILES.get(self.profile),
                    timeout=self.timeout, package_scope='full_package', attempts=1,
                    backend_sha256=fingerprint(self.base_url) if self.profile != 'static' else None,
                    executable=self.executable,
                    reasoning_effort='low' if self.profile == 'gpt-5.4' or (self.scanner == 'cisco' and self.profile != 'static') else None,
                    thinking_disabled=self.profile == 'qwen3.8-27b',
                    evidence_ids_schema_compat=self.schema_compat)


def scanner_env(config, endpoint, temporary):
    env = {k: v for k, v in os.environ.items()
           if not any(token in k.upper() for token in ('API_KEY', 'TOKEN', 'SECRET', 'PASSWORD'))
           and not k.upper().startswith(('SKILLSPECTOR_', 'SKILL_SCANNER_LLM_'))
           and k.upper() != 'OPENAI_BASE_URL' and k != config.api_key_env}
    if config.profile == 'static':
        return env
    key = os.environ[config.api_key_env]
    model = PROFILES[config.profile]
    if config.scanner == 'cisco':
        env.update(SKILL_SCANNER_LLM_PROVIDER='openai-compatible',
                   SKILL_SCANNER_LLM_MODEL=('openai/' + model if config.profile == 'gpt-5.4' else model),
                   SKILL_SCANNER_LLM_BASE_URL=endpoint, SKILL_SCANNER_LLM_API_KEY=key,
                   SKILL_SCANNER_LLM_REASONING_EFFORT='low', SKILL_SCANNER_LLM_MAX_TOKENS='4096')
    else:
        qwen = config.profile == 'qwen3.8-27b'
        registry = Path(temporary) / 'model_registry.yaml'
        registry.write_text('models:\n  ' + json.dumps(model) + ':\n    context_length: ' +
                            str(65536 if qwen else 1050000) + '\n    max_output_tokens: ' +
                            str(16384 if qwen else 128000) + '\n')
        env.update(SKILLSPECTOR_PROVIDER='openai', SKILLSPECTOR_MODEL=model,
                   SKILLSPECTOR_MODEL_REGISTRY=str(registry), OPENAI_API_KEY=key,
                   OPENAI_BASE_URL=endpoint)
        if not qwen: env['SKILLSPECTOR_REASONING_EFFORT'] = 'low'
    env.update(NO_PROXY='127.0.0.1,localhost', no_proxy='127.0.0.1,localhost')
    return env


def run_scan(package, expected_hash, config, *, version, command_builder, normalize, provenance):
    config.validate()  # configuration errors are not attempted scanner outputs
    result = dict(status='invalid', verdict=None, score=None, error=None,
                  package_sha256=expected_hash, config={**config.provenance(), **provenance},
                  tool_version=None)
    executable = config.executable or shutil.which('skill-scanner' if config.scanner == 'cisco' else 'skillspector')
    try:
        if not executable: raise ValueError('Scanner executable not found; configure PATH or --scanner-bin')
        if Path(executable).is_file():
            executable = str(Path(executable).resolve())
        check = subprocess.run([executable, '--version'], capture_output=True, text=True, timeout=config.timeout)
        versions = re.findall(r'(?<!\d)\d+\.\d+\.\d+(?!\d)', check.stdout + check.stderr)
        if check.returncode != 0 or set(versions) != {version}:
            raise ValueError(f'Expected scanner version {version}')
        result['tool_version'] = version
        result['executable_sha256'] = fingerprint(Path(executable).read_bytes().hex()) if Path(executable).is_file() else None
        with tempfile.TemporaryDirectory(prefix='corsa-scan-') as temporary:
            target, files = copy_full_package(package, Path(temporary) / 'package', expected_hash)
            report_path = Path(temporary) / 'report.json'
            transport = (compatibility_endpoint(config.base_url,
                disable_thinking=config.profile == 'qwen3.8-27b',
                evidence_ids_compat=config.schema_compat, timeout=config.timeout)
                if config.profile == 'qwen3.8-27b' or config.schema_compat else nullcontext(config.base_url))
            with transport as endpoint:
                proc = subprocess.run(command_builder(executable, target, report_path, config),
                    cwd=temporary, env=scanner_env(config, endpoint, temporary),
                    capture_output=True, text=True, timeout=config.timeout)
            report = json.loads(report_path.read_text())
            if not isinstance(report, dict): raise ValueError('Scanner report must be an object')
            score, verdict = normalize(report, config)
            expected_rc = (1 if verdict == 'malicious' else 0) if config.scanner == 'skillspector' else 0
            if proc.returncode != expected_rc: raise ValueError('Scanner exit code inconsistent with valid report')
            result.update(status='valid', score=score, verdict=verdict, package_files=files,
                          findings=report.get('findings', report.get('issues', [])))
    except subprocess.TimeoutExpired:
        result.update(status='invalid', error='scanner_timeout')
    except Exception as exc:
        # Do not persist raw scanner logs or backend exceptions containing secrets.
        result.update(status='invalid', error=type(exc).__name__)
    return result
