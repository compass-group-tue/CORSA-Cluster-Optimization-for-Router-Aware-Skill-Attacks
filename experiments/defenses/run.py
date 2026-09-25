"""Pre-retrieval labeled package scanning; no routing or victim execution."""
import argparse
import json
from pathlib import Path
from src.defenses import cisco, skillspector
from src.defenses.scanner import ScannerConfig, PROFILES
from src.defenses.packages import package_hash
from src.defenses.metrics import aggregate_scans
from src.infra.artifacts import fingerprint


def load_manifest(path):
    path = Path(path)
    records = json.loads(path.read_text())
    if not isinstance(records, list) or not records: raise ValueError('Manifest must be a nonempty JSON list')
    samples, seen = [], set()
    for row in records:
        if not isinstance(row, dict) or set(row) != {'sample_id', 'package_path', 'label'}:
            raise ValueError('Manifest records require sample_id, package_path, label only')
        if any(not isinstance(row[k], str) or not row[k].strip() for k in row):
            raise ValueError('Manifest fields must be nonempty strings')
        if row['label'] not in ('malicious', 'benign') or row['sample_id'] in seen:
            raise ValueError('Invalid label or duplicate sample ID')
        seen.add(row['sample_id'])
        package = path.parent / row['package_path']
        digest = package_hash(package)  # reject invalid packages before invoking any scanner
        samples.append({**row, 'package_path': str(package.absolute()), 'package_sha256': digest})
    return samples


def evaluate_packages(samples, config, out_dir):
    config.validate()
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=False)
    plan = dict(schema_version=1, samples=samples, scanner=config.provenance())
    plan['condition_sha256'] = fingerprint(plan)
    (out / 'config.json').write_text(json.dumps(plan, indent=2))
    adapter = cisco if config.scanner == 'cisco' else skillspector
    rows = []
    with (out / 'scanner_rows.jsonl').open('w') as sink:
        for sample in samples:
            # Ground truth never enters the scanner adapter.
            result = adapter.scan_package(sample['package_path'], sample['package_sha256'], config)
            row = {**result, 'sample_id': sample['sample_id'], 'label': sample['label'],
                   'condition_sha256': plan['condition_sha256']}
            rows.append(row); sink.write(json.dumps(row) + '\n')
    summary = aggregate_scans(rows)
    (out / 'summary.json').write_text(json.dumps(summary, indent=2))
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--scanner', choices=['cisco','skillspector'], required=True)
    parser.add_argument('--profile', choices=['static', *PROFILES], default='static')
    parser.add_argument('--scanner-bin')
    parser.add_argument('--base-url', help='OpenAI-compatible endpoint including /v1')
    parser.add_argument('--api-key-env', default='OPENAI_API_KEY')
    parser.add_argument('--timeout', type=float, default=900)
    parser.add_argument('--schema-compat', action='store_true', help='Cisco only: relax evidence_ids array uniqueness')
    parser.add_argument('--out-dir', required=True)
    args = parser.parse_args(argv)
    config = ScannerConfig(args.scanner, args.profile, args.scanner_bin, args.base_url,
                           args.api_key_env, args.timeout, args.schema_compat)
    config.validate()
    return evaluate_packages(load_manifest(args.manifest), config, args.out_dir)


if __name__ == '__main__':
    main()
