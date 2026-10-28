"""Preserve all F8 annotations and assess baseline recovery without changing features."""
import csv
import gc
import json
import platform
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyabf

import event_extraction as ee
from project_config import sha256_file

PROJECT = Path(__file__).resolve().parent
ROOT = PROJECT.parents[2]
UPSTREAM = ROOT / 'analysis/runs/nanopore_context_fix_20260908_133737_133225/event_extraction'


def key(recording, sweep, start, end):
    return (str(recording), int(sweep), round(float(start), 12), round(float(end), 12))


def candidate_window(intervals, anchor, points, direction, n_samples):
    """Nearest full window outside all valid annotated sample intervals."""
    if direction == 'pre':
        right = anchor
        for start, end in reversed(intervals):
            if start >= right:
                continue
            if end <= right - points:
                break
            right = min(right, start)
        return (right - points, right) if right >= points else None
    left = anchor
    for start, end in intervals:
        if end <= left:
            continue
        if start >= left + points:
            break
        left = max(left, end)
    return (left, left + points) if left + points <= n_samples else None


def main():
    manifest_path = ROOT / 'recordings.csv'
    manifest = ee.load_recording_manifest(manifest_path)
    manifest = manifest[manifest.protein.eq('F8')]
    ee.validate_recording_manifest(manifest)
    upstream_manifest = json.loads((UPSTREAM / 'run_manifest.json').read_text())
    f = pd.read_csv(UPSTREAM / 'abf_event_features.csv')
    f = f[f.protein.eq('F8')]
    q = pd.read_csv(UPSTREAM / 'abf_event_qc.csv').fillna({'qc_reason': ''})
    q = q[q.protein.eq('F8')].set_index('event_id')
    experiment = 'f8_annotation_coverage_' + datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    out = ROOT / 'analysis/runs' / experiment / 'annotation_coverage_and_recovery'
    out.mkdir(parents=True, exist_ok=False)
    paths = [manifest_path, UPSTREAM/'abf_event_features.csv', UPSTREAM/'abf_event_qc.csv', UPSTREAM/'run_manifest.json',
             *map(Path, manifest.annotation_path), *map(Path, manifest.abf_path)]
    hashes = {str(p): sha256_file(p) for p in paths}
    for p in paths[:3]:
        expected = upstream_manifest.get('source_checksums', {}).get(str(p))
        if expected:
            assert hashes[str(p)] == expected, f'Input changed: {p}'
    for p in list(map(Path, manifest.annotation_path)) + list(map(Path, manifest.abf_path)):
        assert hashes[str(p)] == upstream_manifest['source_checksums'][str(p)], str(p)
    run = dict(status='running', created_at=datetime.now().astimezone().isoformat(),
               experiment_id=experiment, agent_stage='annotation_coverage_and_recovery',
               dataset_version='v6_f8_all_annotations_audit_v1',
               upstream_dataset_version=upstream_manifest['dataset_version'],
               upstream_experiment_id=upstream_manifest['experiment_id'], upstream_output_path=str(UPSTREAM),
               source_paths=list(hashes), source_checksums=hashes,
               configuration=dict(output_directory=str(out), balance_policy='natural_unbalanced',
                   feature_policy='link accepted v6 features by physical boundary; unavailable remain missing',
                   baseline_candidate_policy='nearest full 1 ms interval before/after event excluding all valid annotations; diagnostic only',
                   maximum_candidate_distance_ms=100.0, baseline_rule_changed=False),
               random_seeds={'used': False},
               code_version={str(p):sha256_file(p) for p in [Path(__file__), PROJECT/'event_extraction.py', PROJECT/'feature_engineering.py']},
               environment=dict(python=platform.python_version(), numpy=np.__version__, pandas=pd.__version__, pyabf=pyabf.__version__))
    def save_run():
        (out/'run_manifest.json').write_text(json.dumps(run, indent=2), encoding='utf-8')
    save_run()
    print('OUTPUT', out, flush=True)
    try:
        feature_by_key = {key(r.recording_id,r.sweep_index,r.abf_event_start_time_s,r.abf_event_end_time_s): r._asdict() for r in f.itertuples(index=False)}
        assert len(feature_by_key) == len(f) == 153
        features = [c for c in f.columns if c not in q.columns and c != 'event_id']
        run['feature_list'] = features
        rows, physical, raw_lookup = [], {}, {}
        for record in manifest.itertuples(index=False):
            with open(record.annotation_path, encoding='utf-8-sig', newline='') as handle:
                raw = list(csv.DictReader(handle))
            events = ee.load_annotation_events(record)
            assert len(events) == len(raw)
            for i, (event, source) in enumerate(zip(events,raw)):
                eid = f'{record.annotation_id}:{i}'
                raw_lookup[eid] = source
                row = q.loc[eid].to_dict()
                assert int(row['source_row_number']) == i+2
                row.update(event_id=eid, original_csv_row_json=json.dumps(source,ensure_ascii=False),
                           feature_available=False, feature_source_event_id='', physical_event_id='',
                           baseline_recovery_applied=False, representation='annotation_row')
                k = None
                if not event['_load_qc_reason']:
                    s = ee.apply_time_offset(event['eventStartTime'],record.time_offset_s)
                    e = ee.apply_time_offset(event['eventEndTime'],record.time_offset_s)
                    k = key(record.recording_id,record.sweep_index,s,e)
                    if k not in physical:
                        physical[k] = dict(physical_event_id=f'{record.recording_id}:sweep{record.sweep_index}:boundary{len(physical)+1}',
                            canonical_annotation_id=eid, recording_id=str(record.recording_id),sweep_index=int(record.sweep_index),
                            abf_path=record.abf_path, start_s=s,end_s=e, annotation_row_count=0)
                    physical[k]['annotation_row_count'] += 1
                    row['physical_event_id'] = physical[k]['physical_event_id']
                if k in feature_by_key:
                    linked = feature_by_key[k]
                    row.update({c:linked[c] for c in features})
                    row.update(feature_available=True, feature_source_event_id=linked['event_id'],
                               feature_status='accepted_original' if row['qc_pass'] else 'duplicate_linked_features')
                else:
                    row['feature_status'] = 'invalid_boundary' if k is None else 'baseline_unavailable'
                rows.append(row)
        all_rows = pd.DataFrame(rows)
        assert len(all_rows)==345 and all_rows.event_id.nunique()==345 and len(physical)==169
        assert all_rows.feature_available.sum()==306
        assert set(all_rows.event_id)==set(q.index)
        all_rows.to_csv(out/'f8_all_annotations.csv', index=False)
        all_rows[all_rows.feature_status.eq('invalid_boundary')].to_csv(out/'f8_invalid_annotations.csv',index=False)
        pd.DataFrame(physical.values()).to_csv(out/'f8_physical_event_index.csv', index=False)
        diagnostics=[]
        for record in manifest.itertuples(index=False):
            print('Reviewing waveform', record.recording_id, flush=True)
            data=ee.load_abf_signal(record.abf_path,int(record.sweep_index))
            t,y=data['time_s'],data['signal']
            points=int(round(.001/data['sampling_period_s']))
            group=sorted([v for v in physical.values() if v['recording_id']==str(record.recording_id)],key=lambda v:(v['start_s'],v['end_s']))
            for v in group:
                v['start_index'],v['end_index']=ee.get_event_indices(t,v['start_s'],v['end_s'])
            intervals=[(v['start_index'],v['end_index']) for v in group]
            plot_entries=[]
            for pos,v in enumerate(group):
                eid=v['canonical_annotation_id']
                if q.loc[eid,'qc_reason']!='Insufficient baseline after excluding the preceding event':
                    continue
                start,end=v['start_index'],v['end_index']
                previous_end=max([b for _,b in intervals[:pos]],default=0)
                available=max(0,start-max(start-points,previous_end,0))
                assert available<3
                d=dict(**v,existing_baseline_points=available,n_event_points=end-start,
                       waveform_finite=bool(np.isfinite(y[start:end]).all()),recovered=False,
                       baseline_feature_values_generated=False)
                windows={}
                for side,anchor in [('pre',start),('post',end)]:
                    w=candidate_window(intervals,anchor,points,side,len(t))
                    if w:
                        a,b=w
                        assert all(b<=s or a>=e for s,e in intervals)
                        distance=(start-b if side=='pre' else a-end)*data['sampling_period_s']*1000
                        d.update({f'{side}_candidate_start_index':a,f'{side}_candidate_end_index':b,
                            f'{side}_candidate_distance_ms':distance,f'{side}_candidate_mean_pA':float(np.mean(y[a:b])),
                            f'{side}_candidate_sd_pA':float(np.std(y[a:b],ddof=1)),
                            f'{side}_candidate_within_100ms':bool(distance<=100)})
                        windows[side]=w
                d['post_minus_pre_candidate_mean_pA']=d.get('post_candidate_mean_pA',np.nan)-d.get('pre_candidate_mean_pA',np.nan)
                diagnostics.append(d)
                left=max(0,start-5*points);right=min(len(t),end+5*points)
                name=eid.replace(':','_')
                np.savez_compressed(out/f'{name}_waveform.npz', sample_indices=np.arange(left,right),
                    time_relative_ms=(t[left:right]-v['start_s'])*1000,signal_pA=y[left:right],
                    event_start_index=start,event_end_index=end,
                    **{f'{side}_candidate_signal_pA':y[a:b] for side,(a,b) in windows.items()})
                plot_entries.append((v,d,left,right))
            for page in range(0,len(plot_entries),4):
                entries=plot_entries[page:page+4]
                fig,axes=plt.subplots(len(entries),1,figsize=(12,3.3*len(entries)),squeeze=False,layout='constrained')
                for ax,(v,d,left,right) in zip(axes[:,0],entries):
                    ax.plot((t[left:right]-v['start_s'])*1000,y[left:right],lw=.7)
                    ax.axvspan(0,(v['end_s']-v['start_s'])*1000,color='orange',alpha=.25)
                    ax.axhline(d.get('pre_candidate_mean_pA',np.nan),color='green',ls='--',label='candidate pre mean')
                    ax.axhline(d.get('post_candidate_mean_pA',np.nan),color='purple',ls=':',label='candidate post mean')
                    ax.set(title=f"{v['canonical_annotation_id']} | existing baseline {d['existing_baseline_points']} samples",xlabel='Time from annotated start (ms)',ylabel='Current (pA)')
                    ax.legend(fontsize=8,loc='best')
                fig.savefig(out/f'recovery_waveforms_{record.recording_id}_{page//4+1}.png',dpi=120)
                plt.close(fig)
            del data,t,y
            gc.collect()
        audit=pd.DataFrame(diagnostics)
        assert len(audit)==16 and audit.canonical_annotation_id.nunique()==16
        audit.to_csv(out/'f8_baseline_recovery_candidates.csv',index=False)
        summary=all_rows.groupby(['recording_id','feature_status']).size().rename('annotation_rows').reset_index()
        summary.to_csv(out/'f8_coverage_summary.csv',index=False)
        assert all(sha256_file(p)==h for p,h in hashes.items())
        run.update(status='completed', finished_at=datetime.now().astimezone().isoformat(),
                   row_counts=dict(annotation_rows=len(all_rows),unique_valid_boundaries=len(physical),
                       independent_feature_events=len(f),annotation_rows_with_linked_features=int(all_rows.feature_available.sum()),
                       annotation_rows_without_features=int((~all_rows.feature_available).sum()),recovery_candidates=len(audit)),
                   recording_counts={'F8':3},validation='Coverage 345 rows; 169 physical boundaries; 153 accepted sources; 16 waveform reviews; no candidate overlap with valid annotations; input hashes unchanged',
                   generated_files={p.name:{'path':str(p),'sha256':sha256_file(p)} for p in out.iterdir() if p.name!='run_manifest.json'})
        save_run()
        print(summary.to_string(index=False),flush=True)
        print(audit[['canonical_annotation_id','existing_baseline_points','pre_candidate_distance_ms','post_candidate_distance_ms','post_minus_pre_candidate_mean_pA']].to_string(index=False),flush=True)
    except BaseException as exc:
        run.update(status='failed',error=repr(exc));save_run();raise


if __name__=='__main__':
    main()
