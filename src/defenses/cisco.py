"""Cisco complete-package classification: HIGH or CRITICAL flags a package."""
from .scanner import run_scan

VERSION = '2.1.0'
SEVERITIES = {'NONE': 0, 'SAFE': 0, 'INFO': 1, 'LOW': 2, 'MEDIUM': 3, 'HIGH': 4, 'CRITICAL': 5}
CORE_ANALYZERS = {'static_analyzer', 'bytecode', 'pipeline', 'correlation', 'behavioral_analyzer'}


def command(executable, package, output, config):
    args = [executable, 'scan', str(package), '--use-behavioral']
    if config.profile != 'static': args += ['--use-llm', '--llm-provider', 'openai-compatible']
    return args + ['--format', 'json', '--output-json', str(output), '--compact']


def normalize_report(report, config):
    used = report.get('analyzers_used')
    if report.get('analyzers_failed') or not isinstance(used, list) or not CORE_ANALYZERS <= set(used):
        raise ValueError('Missing or failed Cisco analyzers')
    usage = report.get('llm_usage')
    if config.profile == 'static':
        if 'llm_analyzer' in used or usage: raise ValueError('Unexpected LLM use in static condition')
    elif ('llm_analyzer' not in used or not isinstance(usage, dict) or
          type(usage.get('total_tokens')) is not int or usage['total_tokens'] <= 0):
        raise ValueError('Cisco LLM presence/token usage not established')
    findings = report.get('findings')
    if not isinstance(findings, list): raise ValueError('Findings must be a list')
    values = []
    for finding in findings:
        if not isinstance(finding, dict): raise ValueError('Finding must be an object')
        severity = str(finding.get('severity', '')).upper()
        if severity not in SEVERITIES: raise ValueError('Invalid severity')
        values.append(SEVERITIES[severity])
    score = max(values, default=0)
    return score, 'malicious' if score >= 4 else 'benign'


def scan_package(package, expected_hash, config):
    if config.scanner != 'cisco': raise ValueError('Expected Cisco configuration')
    return run_scan(package, expected_hash, config, version=VERSION, command_builder=command,
        normalize=normalize_report, provenance=dict(version_required=VERSION,
        source_commit='a24df340ca6056a6446a239f4a7b114b11c6073a',
        rule='any HIGH or CRITICAL', behavioral=True))
