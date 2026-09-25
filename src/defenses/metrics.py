"""Labeled package detection: invalid scanner outputs remain not flagged."""
from src.infra.artifacts import fingerprint


def aggregate_scans(rows):
    if not rows: raise ValueError('No packages evaluated')
    if len({r['sample_id'] for r in rows}) != len(rows): raise ValueError('Duplicate samples')
    if len({fingerprint(r['config']) for r in rows}) != 1: raise ValueError('Mixed scanner configurations')
    for row in rows:
        if row['label'] not in ('malicious', 'benign'): raise ValueError('Invalid ground-truth label')
        if row['status'] not in ('valid', 'invalid'): raise ValueError('Unknown scanner status')
        if row['status'] == 'valid' and row['verdict'] not in ('malicious', 'benign'):
            raise ValueError('Valid output requires a verdict')
    groups = {}
    for label in ('malicious', 'benign'):
        subset = [r for r in rows if r['label'] == label]
        flagged = sum(r['status'] == 'valid' and r['verdict'] == 'malicious' for r in subset)
        valid = sum(r['status'] == 'valid' for r in subset)
        groups[label] = dict(n_evaluated=len(subset), n_valid=valid, n_invalid=len(subset)-valid,
                             n_flagged=flagged, rate=flagged/len(subset) if subset else None)
    valid = sum(r['status'] == 'valid' for r in rows)
    return dict(recall=groups['malicious']['rate'], fpr=groups['benign']['rate'],
                n_evaluated=len(rows), n_valid=valid, n_invalid=len(rows)-valid,
                coverage=valid/len(rows), invalid_rate=1-valid/len(rows), by_label=groups,
                false_negatives=groups['malicious']['n_evaluated']-groups['malicious']['n_flagged'],
                policy='Invalid scanner outputs are not flagged and remain in denominators; one attempt')
