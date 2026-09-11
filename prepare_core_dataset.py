"""Prepare an unbalanced, provenance-preserving core-feature view; never train."""
import argparse
import hashlib
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

CORE = ['dwell_time_ms', 'blockage_mean', 'blockage_sd', 'auc']
KEY = ['recording_id', 'event_id']


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def prepare(upstream, output_root):
    upstream = Path(upstream).resolve()
    parent = json.loads((upstream / 'run_manifest.json').read_text())
    if parent['status'] != 'completed':
        raise ValueError('Upstream extraction is not completed')
    names = ['abf_event_features.csv', 'abf_event_qc.csv',
             'duplicate_annotations.csv', 'annotation_alignment.csv',
             'recording_qc_summary.csv', 'configuration.json']
    for name in names:
        if sha256(upstream / name) != parent['generated_files'][name]['sha256']:
            raise ValueError(f'Upstream checksum mismatch: {name}')
    # Read source strings to preserve every exported feature and identifier exactly.
    source = pd.read_csv(upstream / names[0], dtype=str, keep_default_na=False)
    qc = pd.read_csv(upstream / names[1], dtype=str, keep_default_na=False)
    accepted = qc[qc.extraction_qc_pass.eq('True')]
    if source.duplicated(KEY).any() or qc.duplicated(KEY).any():
        raise ValueError('Duplicate provenance key')
    if set(map(tuple, source[KEY].values)) != set(map(tuple, accepted[KEY].values)):
        raise ValueError('Feature/QC accepted IDs disagree')
    if source.groupby('protein').size().to_dict() != {'F8': 153, 'HSA': 12735}:
        raise ValueError('Unexpected cohort; review before preparing a new dataset')
    boundary = ['abf_path', 'sweep_index', 'event_start_index', 'event_end_index']
    duplicates = source[source.duplicated(boundary, keep=False)]
    if not duplicates.empty:
        raise ValueError('Accepted events contain repeated ABF sample boundaries')
    values = source[CORE].apply(pd.to_numeric, errors='raise')
    if not np.isfinite(values.to_numpy()).all():
        raise ValueError('Core features contain missing or infinite values; no imputation authorized')
    if (values.dwell_time_ms <= 0).any() or (values[CORE[1:]] < 0).any().any():
        raise ValueError('Invalid core-feature signs')
    duration = (pd.to_numeric(source.abf_event_end_time_s)
                - pd.to_numeric(source.abf_event_start_time_s)) * 1000
    if not np.allclose(duration, values.dwell_time_ms, atol=1e-7, rtol=1e-7):
        raise ValueError('Duration/boundary mismatch')
    sample_count = pd.to_numeric(source.event_end_index) - pd.to_numeric(source.event_start_index)
    if not sample_count.eq(pd.to_numeric(source.n_event_points)).all() or (sample_count <= 0).any():
        raise ValueError('Invalid event sample counts')
    from event_extraction import MIN_BASELINE_POINTS
    if sha256(Path(__file__).with_name('event_extraction.py')) != parent['code_version']['event_extraction.py']:
        raise ValueError('Extraction code changed since upstream; review baseline contract')
    if (pd.to_numeric(source.actual_baseline_points) < MIN_BASELINE_POINTS).any():
        raise ValueError('Accepted event violates upstream minimum baseline points')
    # Explicit metadata allowlist: never rely on automatic numeric-column selection for X.
    provenance = list(source.columns[:30]) + [
        'shape_qc_pass', 'shape_qc_reason', 'baseline_mean', 'baseline_sd',
        'n_event_points', 'requested_baseline_points', 'actual_baseline_points',
        'event_start_index', 'event_end_index']
    table = source[provenance + CORE].merge(
        accepted[KEY + ['extraction_qc_pass', 'extraction_qc_reason']],
        on=KEY, how='left', validate='one_to_one', sort=False)
    assert table[CORE].equals(source[CORE])
    now = datetime.now(timezone.utc)
    experiment = 'nanopore_core_natural_' + now.strftime('%Y%m%dT%H%M%S%fZ')
    output = Path(output_root).resolve() / experiment / 'core_dataset_natural'
    output.mkdir(parents=True, exist_ok=False)
    manifest = {
        'status': 'running', 'created_at': now.isoformat(), 'experiment_id': experiment,
        'dataset_version': parent['dataset_version'] + '_core_natural_v1',
        'upstream_dataset_version': parent['dataset_version'],
        'upstream_experiment_id': parent['experiment_id'], 'upstream_output_path': str(upstream),
        'upstream_manifest_sha256': sha256(upstream / 'run_manifest.json'),
        'source_paths': [str(upstream / n) for n in names] + parent['source_paths'],
        'source_checksums': {str(upstream / n): sha256(upstream / n) for n in names},
        'upstream_raw_checksums': parent['source_checksums'],
        'configuration': {'features': CORE, 'balance_policy': 'natural_unbalanced',
                          'shape_filter': False, 'imputation': None, 'outlier_removal': None,
                          'scaling': None, 'training': False, 'split': None,
                          'extreme_review': 'Per recording minimum and maximum, retain all ties; descriptive only'},
        'random_seeds': {'used': False}, 'feature_list': CORE,
        'code_config_version': {'script_sha256': sha256(__file__),
                                'upstream_code': parent['code_version'],
                                'upstream_configuration': parent['configuration']},
        'environment': {'python': sys.version, 'numpy': np.__version__, 'pandas': pd.__version__},
        'limitations': ['No new waveform QC or waveform identity comparison across different boundaries.',
                        'Raw input checksums inherited, not recomputed in this preparation stage.',
                        'HSA has one recording: independent both-class grouped evaluation is not feasible.',
                        'No balancing, split, imputation, scaling, fitting or performance evaluation.']}
    def save_manifest():
        (output / 'run_manifest.json').write_text(json.dumps(manifest, indent=2, allow_nan=False) + '\n')
    save_manifest()
    table.to_csv(output / 'core_features_natural.csv', index=False)
    table[table.protein.eq('F8')].to_csv(output / 'f8_core_features_natural.csv', index=False)
    for name in names[1:]:
        shutil.copyfile(upstream / name, output / ('upstream_' + name))
    duplicates.to_csv(output / 'accepted_duplicate_boundaries.csv', index=False)
    summaries, ranges, extremes = [], [], []
    for (protein, recording), group in table.groupby(['protein', 'recording_id']):
        summaries.append({'protein': protein, 'recording_id': recording,
                          'events': len(group), 'shape_qc_pass': int(group.shape_qc_pass.eq('True').sum()),
                          'baseline_shorter_than_requested': int((pd.to_numeric(group.actual_baseline_points) < pd.to_numeric(group.requested_baseline_points)).sum()),
                          'minimum_actual_baseline_points': int(pd.to_numeric(group.actual_baseline_points).min())})
        for feature in CORE:
            numeric = pd.to_numeric(group[feature])
            ranges.append({'protein': protein, 'recording_id': recording, 'feature': feature,
                           'count': len(group), 'missing': int(numeric.isna().sum()),
                           'infinite': int(np.isinf(numeric).sum()), 'min': float(numeric.min()),
                           'median': float(numeric.median()), 'max': float(numeric.max())})
            for side, value in [('minimum', numeric.min()), ('maximum', numeric.max())]:
                for _, event in group[numeric.eq(value)].iterrows():
                    extremes.append({'protein': protein, 'recording_id': recording,
                                     'event_id': event.event_id, 'feature': feature,
                                     'extreme': side, 'value': float(value)})
    pd.DataFrame(summaries).to_csv(output / 'cohort_summary.csv', index=False)
    pd.DataFrame(ranges).to_csv(output / 'core_feature_audit.csv', index=False)
    pd.DataFrame(extremes).to_csv(output / 'extreme_values_review.csv', index=False)
    reread = pd.read_csv(output / 'core_features_natural.csv', dtype=str, keep_default_na=False)
    pd.testing.assert_frame_equal(reread, table)
    assert len(pd.read_csv(output / 'f8_core_features_natural.csv')) == 153
    for name in names[1:]:
        assert sha256(output / ('upstream_' + name)) == sha256(upstream / name)
    (output / 'README.md').write_text(
        '# Natural core-feature dataset\n\n'
        'Model feature allowlist: ' + ', '.join(CORE) + '. All other columns are metadata/QC, not X.\n\n'
        'Units: dwell_time_ms = ms; blockage_mean = percent; blockage_sd = percentage points; auc = percent·ms.\n\n'
        'All 153 F8 and 12,735 HSA extraction-accepted events retained. Shape QC does not filter rows. '
        'Core values copied unchanged, no balancing, outlier removal, imputation or training. '
        'The F8-only file is a subset of the combined file; never concatenate them.\n\n'
        'Full upstream QC (including rejected rows), duplicates and boundary-validation tables are preserved. '
        'CSV boundary validation is not waveform alignment. Extreme values are descriptive min/max, not rejection thresholds.\n\n'
        'Group any future split by at least recording_id. HSA has only one recording, '
        'so independent both-class evaluation cannot yet be performed.\n')
    manifest.update(status='completed', finished_at=datetime.now(timezone.utc).isoformat(),
                    row_counts={'combined': len(table), 'F8': 153, 'HSA': 12735,
                                'upstream_qc': len(qc), 'upstream_rejected': len(qc)-len(accepted)},
                    recording_counts={'F8': 3, 'HSA': 1},
                    verification={'core_values_equal_upstream': True, 'csv_roundtrip_equal': True,
                                  'accepted_duplicate_boundaries': 0, 'nonfinite_core_values': 0},
                    generated_files={p.name: {'path': str(p), 'sha256': sha256(p)}
                                     for p in output.iterdir() if p.name != 'run_manifest.json'})
    manifest['manifest_path'] = str(output / 'run_manifest.json')
    save_manifest()
    print(json.dumps({'output': str(output), 'cohort': summaries, 'ranges': ranges}, indent=2))
    return output


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--upstream', required=True)
    parser.add_argument('--output-root', required=True)
    args = parser.parse_args()
    prepare(args.upstream, args.output_root)
