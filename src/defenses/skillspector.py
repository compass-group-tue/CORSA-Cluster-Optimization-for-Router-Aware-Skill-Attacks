"""NVIDIA complete-package risk classification, separate from victim execution."""
import math
from .scanner import run_scan

VERSION = '2.11.2'


def command(executable, package, output, config):
    args = [executable, 'scan', str(package)]
    if config.profile == 'static': args.append('--no-llm')
    return args + ['--format', 'json', '--output', str(output)]


def normalize_report(report, config):
    if report.get('execution_successful') is not True:
        raise ValueError('Scanner did not complete successfully')
    metadata = report.get('metadata')
    llm = config.profile != 'static'
    if not isinstance(metadata, dict) or metadata.get('llm_requested') is not llm:
        raise ValueError('Requested scanner mode not established')
    if llm:
        attempted, succeeded = metadata.get('llm_calls_attempted'), metadata.get('llm_calls_succeeded')
        usage = metadata.get('inference_usage')
        if (metadata.get('llm_available') is not True or metadata.get('llm_degraded') is True or
            type(attempted) is not int or attempted <= 0 or type(succeeded) is not int or succeeded != attempted or
            not isinstance(usage, list) or not usage or any(not isinstance(u, dict) or
                u.get('usage_source') != 'provider_response' or type(u.get('total_tokens')) is not int or
                u['total_tokens'] <= 0 for u in usage)):
            raise ValueError('Incomplete scanner LLM coverage/usage')
    risk = report.get('risk_assessment')
    score = risk.get('score') if isinstance(risk, dict) else None
    if type(score) not in (int, float) or not math.isfinite(score) or not 0 <= score <= 100:
        raise ValueError('Risk score must be finite and within [0,100]')
    if not isinstance(report.get('issues'), list): raise ValueError('Issues must be a list')
    return float(score), 'malicious' if score > 50 else 'benign'


def scan_package(package, expected_hash, config):
    if config.scanner != 'skillspector': raise ValueError('Expected SkillSpector configuration')
    return run_scan(package, expected_hash, config, version=VERSION, command_builder=command,
        normalize=normalize_report, provenance=dict(version_required=VERSION,
        source_commit='69dcdfb74487d361ba4c811d088cfdea2ff3a9dc',
        rule='risk_score > 50', network_enabled=True, osv_live_enabled=True))
