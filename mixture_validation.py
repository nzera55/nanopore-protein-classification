"""Fail-closed validation for locked external mixture inference."""
import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pyabf

MODEL_SET_GROUPS = (
    {'set_c_core_shape', 'set_d_core_segments', 'set_e_segment_only'},
    {'set_f_core_segment5', 'set_g_segment5_only'},
)
# Backward-compatible name for the original locked C/D/E model group.
SETS = MODEL_SET_GROUPS[0]
SEEDS = {11, 42}


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def validate_models(models):
    require(not models.duplicated(['feature_set', 'seed']).any(), 'Duplicate feature_set/seed')
    model_sets = set(models.feature_set)
    require(model_sets in MODEL_SET_GROUPS,
            'Models must contain exactly sets C/D/E or sets F/G')
    require(set(zip(models.feature_set, models.seed)) ==
            {(feature_set, seed) for feature_set in model_sets for seed in SEEDS},
            'Models must contain exactly every configured feature set x seeds 11/42')
    for r in models.itertuples():
        require(sha256(r.model_path) == r.model_sha256, f'Model checksum mismatch: {r.model_path}')
        require((r.n_hsa, r.n_f8, r.n_total) == (432, 432, 864), 'Training counts mismatch')


def validate_predictions(predictions, event_ids, feature_sets=None):
    require(not predictions.duplicated(['event_id', 'feature_set', 'seed']).any(), 'Duplicate prediction key')
    feature_sets = set(predictions.feature_set) if feature_sets is None else set(feature_sets)
    require(feature_sets in MODEL_SET_GROUPS, 'Unexpected prediction feature-set group')
    expected = pd.MultiIndex.from_product([event_ids, sorted(feature_sets), sorted(SEEDS)])
    actual = pd.MultiIndex.from_frame(predictions[['event_id', 'feature_set', 'seed']])
    require(len(actual) == len(expected) and len(expected.difference(actual)) == 0, 'Incomplete prediction coverage')
    require(np.isfinite(predictions.probability_f8).all() and predictions.probability_f8.between(0, 1).all(),
            'Invalid probability')


def audit_inputs(feature_table, model_run, events, models):
    validate_models(models)
    extraction = feature_table.parent
    em = json.loads((extraction / 'run_manifest.json').read_text())
    augmentation_manifest = None
    if 'source_extraction_run' in em:
        augmentation_manifest = extraction / 'run_manifest.json'
        extraction = Path(em['source_extraction_run'])
        em = json.loads((extraction / 'run_manifest.json').read_text())
    tm = json.loads((model_run / 'run_manifest.json').read_text())
    manifest_path = Path(em['configuration']['manifest_path'])
    require(sha256(manifest_path) == em['manifest_sha256'], 'Extraction manifest checksum mismatch')
    manifest = pd.read_csv(manifest_path, dtype={'recording_id': str})
    expected = {'26709002': ('20260709_F8-Alb_1-1_-100mV.csv', .5),
                '26709003': ('20260709_F8-Alb_19-1_-100mV.csv', .95),
                '26714010': ('mixture19_1_at300nM.csv', .05)}
    require(set(manifest.recording_id) == set(expected) and len(manifest) == 3, 'Mixture mapping mismatch')
    root = Path(em['configuration']['data_root'])
    qc = pd.read_csv(extraction / 'abf_event_qc.csv', dtype={'recording_id': str})
    require(qc.event_id.is_unique and events.event_id.notna().all() and events.event_id.is_unique,
            'Missing or duplicate event IDs')
    require(set(events.event_id) == set(qc.loc[qc.extraction_qc_pass.eq(True), 'event_id']), 'Accepted/QC mismatch')
    require(qc.loc[~qc.extraction_qc_pass.eq(True), 'qc_reason'].notna().all(), 'Rejected event missing QC reason')
    selection = pd.read_csv(model_run / 'training_event_selection.csv', dtype={'recording_id': str})
    require(set(selection.protein) == {'HSA', 'F8'}, 'Non-pure training selection')
    require(not set(selection.event_id) & set(qc.event_id), 'Mixture event in training')
    require(not set(selection.recording_id) & set(manifest.recording_id), 'Mixture recording in training')
    require(not selection.duplicated(['seed', 'event_id']).any(), 'Duplicate training event within seed')
    require(selection.groupby(['seed', 'protein']).size().to_dict() ==
            {(s, c): 432 for s in SEEDS for c in ['HSA', 'F8']}, 'Training selection count mismatch')
    pure_path = Path(tm['feature_table'])
    require(sha256(pure_path) == tm['feature_table_sha256'], 'Pure feature checksum mismatch')
    require(sha256(tm['extraction_manifest']) == tm['extraction_manifest_sha256'], 'Pure extraction checksum mismatch')
    pure = pd.read_csv(pure_path, dtype={'recording_id': str}, low_memory=False)
    require(not pure.protein.eq('MIXTURE').any(), 'Mixture in training source')
    lookup = pure.set_index('event_id')
    require(lookup.index.is_unique and set(selection.event_id) <= set(lookup.index), 'Training source event mismatch')
    require((lookup.loc[selection.event_id, 'protein'].to_numpy() == selection.protein.to_numpy()).all(), 'Training labels mismatch')
    files = [feature_table, extraction / 'run_manifest.json', extraction / 'abf_event_qc.csv',
             extraction / 'raw_event_waveform_index.csv', extraction / 'annotation_alignment.csv',
             manifest_path, model_run / 'trained_models.csv', model_run / 'training_event_selection.csv',
             model_run / 'run_manifest.json', pure_path, Path(tm['extraction_manifest'])]
    files += [Path(x) for x in models.model_path]
    if augmentation_manifest is not None:
        files.append(augmentation_manifest)
    checks = []
    for r in manifest.itertuples():
        csv = (root / r.annotation_path).resolve(); abf_path = (root / r.abf_path).resolve()
        name, fraction = expected[r.recording_id]
        require(csv.name == name and abf_path.stem == r.recording_id, 'CSV/ABF mapping mismatch')
        require(r.f8_fraction == fraction and r.hsa_fraction == 1-fraction or
                np.isclose(r.f8_fraction, fraction) and np.isclose(r.hsa_fraction, 1-fraction), 'Nominal ratio mismatch')
        require(r.pore_id == 'mixture_pore_01' and r.voltage_mV == -100 and r.concentration_nM == 300,
                'Experimental conditions mismatch')
        require(r.event_time_unit == 'ms' and r.time_offset_s == 0 and r.sweep_index == 0, 'Time configuration mismatch')
        with csv.open() as f:
            embedded = re.search(r'File:\s*([^,\s]+)', f.readline())
        embedded_id = embedded.group(1) if embedded else ''
        mapping_qc = ('embedded_id_matches' if embedded_id == r.recording_id else
                      'embedded_id_conflicts_with_user_confirmed_manifest' if embedded_id else 'embedded_id_missing')
        abf = pyabf.ABF(str(abf_path), loadData=False)
        duration = abf.sweepPointCount / abf.dataRate
        q = qc[qc.recording_id == r.recording_id]
        outside = (q.source_event_start < 0) | (q.source_event_end / 1000 > duration)
        require(not (outside & q.extraction_qc_pass.eq(True)).any(), 'Accepted annotation outside ABF')
        e = events[events.recording_id == r.recording_id]
        require(e.abf_path.eq(str(abf_path)).all() and e.annotation_path.eq(str(csv)).all(), 'Feature source mapping mismatch')
        require(e.mixture_id.eq(r.mixture_id).all() and np.isclose(e.f8_fraction, fraction).all(), 'Feature ratio mismatch')
        require(((e.event_start_index >= 0) & (e.event_end_index <= abf.sweepPointCount) &
                 (e.event_end_index > e.event_start_index)).all(), 'Invalid extracted sample boundaries')
        checks.append(dict(recording_id=r.recording_id, mixture_id=r.mixture_id, duration_s=duration,
                           embedded_csv_recording_id=embedded_id, mapping_qc_reason=mapping_qc,
                           annotated=len(q), accepted=len(e), rejected=len(q)-len(e),
                           annotations_outside_abf=int(outside.sum()), shape_qc_failed=int((~e.shape_qc_pass).sum()),
                           reconstructed=int(e.event_boundary_reconstructed.sum())))
        files += [csv, abf_path]
    return {'input_sha256': {str(p.resolve()): sha256(p) for p in files},
            'checks': checks, 'training_overlap_events': 0, 'training_overlap_recordings': 0,
            'alignment_limit': 'User-confirmed current manifest mapping used. Bounds and sampled waveform alignment checks support extraction, but do not independently prove annotation-to-waveform identity.'}, qc
