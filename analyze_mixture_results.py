"""Descriptive external-mixture analysis; never calibrate using mixtures."""
import argparse
import itertools
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mixture_validation import sha256, require


def build_review(events, threshold, margin=.1):
    rows = []
    for event_id, group in events.groupby('event_id'):
        row = dict(event_id=event_id, mixture_id=group.mixture_id.iloc[0],
                   recording_id=group.recording_id.iloc[0],
                   near_threshold=bool((abs(group.probability_f8_mean-threshold) <= margin).any()),
                   seed_disagreement=bool((~group.seed_agreement).any()),
                   feature_set_disagreement=group.predicted_class.nunique() > 1)
        for r in group.itertuples():
            row[f'{r.feature_set}_p_f8'] = r.probability_f8_mean
            row[f'{r.feature_set}_seed_agreement'] = r.seed_agreement
        row['review_required'] = any(row[k] for k in ['near_threshold', 'seed_disagreement', 'feature_set_disagreement'])
        row['review_reason'] = ';'.join(k for k in ['near_threshold', 'seed_disagreement', 'feature_set_disagreement'] if row[k])
        rows.append(row)
    return pd.DataFrame(rows)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--inference-run', required=True)
    parser.add_argument('--extraction-run', required=True)
    parser.add_argument('--resume', action='store_true', help='Regenerate derived analysis for this run')
    args = parser.parse_args(argv)
    inference = Path(args.inference_run).resolve(); extraction = Path(args.extraction_run).resolve()
    root = inference.parents[2]; run_id = inference.name
    viz = root/'visualization'/'runs'/run_id
    wave = root/'waveform_review'/'runs'/run_id
    report_dir = root/'reports'/'runs'/run_id
    for p in [viz, wave, report_dir]: p.mkdir(parents=True, exist_ok=args.resume)
    manifest = json.loads((inference/'run_manifest.json').read_text())
    threshold = manifest['threshold']
    e = pd.read_csv(inference/'ensemble_event_predictions.csv', dtype={'recording_id': str})
    summary = pd.read_csv(inference/'mixture_prediction_summary.csv')
    features = pd.read_csv(extraction/'abf_event_features.csv', dtype={'recording_id': str})
    index = pd.read_csv(extraction/'raw_event_waveform_index.csv', dtype={'recording_id': str})
    require(index.event_id.is_unique, 'Duplicate waveform ID')
    review = build_review(e, threshold)
    review = review.merge(index, on=['event_id', 'recording_id'], how='left', validate='one_to_one')
    require(review.archive_file.notna().all(), 'Missing waveform provenance')
    review['archive_file'] = review.archive_file.map(lambda p: str((extraction/p).resolve()))
    review = review.merge(features[['event_id', 'shape_qc_pass', 'shape_qc_reason', 'event_boundary_reconstructed']], on='event_id', validate='one_to_one')
    review['visual_review_status'] = 'not_selected_for_montage'
    # Deterministic montage: up to four flagged events per mixture, seed disagreement first.
    selected = review[review.review_required].sort_values(
        ['mixture_id','seed_disagreement','feature_set_disagreement','event_id'],
        ascending=[True,False,False,True]).groupby('mixture_id').head(4).event_id.tolist()
    review.loc[review.event_id.isin(selected), 'visual_review_status'] = 'montage_generated_not_expert_labeled'
    review.to_csv(wave/'all_event_review_flags.csv', index=False)
    review[review.review_required].to_csv(wave/'uncertain_event_review.csv', index=False)
    rows = []
    for (mix, fs), group in e.groupby(['mixture_id','feature_set']):
        for t in np.arange(1,10)/10:
            n = int((group.probability_f8_mean >= t).sum())
            rows.append(dict(mixture_id=mix, feature_set=fs, threshold=t, n_events=len(group),
                             predicted_f8_count=n, predicted_hsa_count=len(group)-n, predicted_f8_fraction=n/len(group)))
    sensitivity = pd.DataFrame(rows); sensitivity.to_csv(inference/'threshold_sensitivity.csv',index=False)
    agreement = []
    for mix, group in [('ALL',e), *list(e.groupby('mixture_id'))]:
        wide = group.pivot(index='event_id',columns='feature_set',values='predicted_class')
        for a,b in itertools.combinations(wide.columns,2):
            agreement.append(dict(mixture_id=mix,feature_set_a=a,feature_set_b=b,n_shared_events=len(wide),
                                  agreement_fraction=float(wide[a].eq(wide[b]).mean())))
    agreement = pd.DataFrame(agreement); agreement.to_csv(inference/'feature_set_agreement.csv',index=False)
    odds = summary[['mixture_id','feature_set','nominal_f8_fraction','predicted_f8_fraction']].copy()
    odds['apparent_predicted_to_nominal_odds_ratio'] = ((summary.predicted_f8_events+.5)/(summary.predicted_hsa_events+.5))/(summary.nominal_f8_fraction/summary.nominal_hsa_fraction)
    odds['interpretation'] = 'Descriptive only; combines classifier, capture, annotation and QC effects; not detection efficiency.'
    odds.to_csv(inference/'descriptive_apparent_odds.csv',index=False)
    mixtures = summary.sort_values('nominal_f8_fraction').mixture_id.unique(); sets = sorted(e.feature_set.unique())
    for filename in ['nominal_vs_predicted_f8_fraction.png','mixture_probability_distributions.png']:
        (viz/filename).write_bytes((inference/filename).read_bytes())
    fig,axs=plt.subplots(1,3,figsize=(14,4),layout='constrained')
    for ax,mix in zip(axs,mixtures):
        for fs in sets:
            g=sensitivity[(sensitivity.mixture_id==mix)&(sensitivity.feature_set==fs)]
            ax.plot(g.threshold,g.predicted_f8_fraction,'o-',label=fs.split('_')[1].upper())
        ax.axvline(threshold,color='gray',ls='--'); ax.set(title=mix,xlabel='Threshold',ylabel='Predicted F8 event fraction',ylim=(-.03,1.03)); ax.legend(title='Set')
    fig.savefig(viz/'threshold_sensitivity.png',dpi=180); plt.close(fig)
    fig,axs=plt.subplots(1,3,figsize=(14,4),layout='constrained')
    for ax,mix in zip(axs,mixtures):
        for fs in sets:
            x=np.sort(e.loc[(e.mixture_id==mix)&(e.feature_set==fs),'probability_f8_mean'])
            ax.step(x,np.arange(1,len(x)+1)/len(x),where='post',label=fs.split('_')[1].upper())
        ax.axvline(threshold,color='gray',ls='--'); ax.set(title=mix,xlabel='Ensemble P(F8)',ylabel='Cumulative event fraction',xlim=(0,1)); ax.legend(title='Set')
    fig.savefig(viz/'probability_ecdf.png',dpi=180); plt.close(fig)
    nrows=max(1, (len(selected)+3)//4)
    fig,axs=plt.subplots(nrows,4,figsize=(16,3.4*nrows),layout='constrained',squeeze=False)
    hashes={}; waveform_checks=[]
    # Verify every flagged archived waveform against its indexed ABF samples.
    import pyabf
    for rid,group in review[review.review_required].groupby('recording_id'):
        abf=pyabf.ABF(group.abf_path.iloc[0]); abf.setSweep(int(group.sweep_index.iloc[0]))
        for r in group.itertuples():
            archive=Path(r.archive_file)
            if str(archive) not in hashes: hashes[str(archive)]=sha256(archive)
            with np.load(archive,allow_pickle=False) as z:
                signal=z['raw_signal'][int(r.waveform_value_start):int(r.waveform_value_stop)]
                raw=abf.sweepY[int(r.abf_event_start_index):int(r.abf_event_end_index)]
                require(np.array_equal(signal,raw),f'Archived waveform mismatch: {r.event_id}')
                waveform_checks.append(dict(event_id=r.event_id,raw_abf_match=True,n_samples=len(signal)))
    for ax,event_id in zip(axs.flat,selected):
        r=review.set_index('event_id').loc[event_id]
        with np.load(r.archive_file,allow_pickle=False) as z:
            start,stop=int(r.waveform_value_start),int(r.waveform_value_stop)
            signal=z['raw_signal'][start:stop]; times=(z['raw_time_s'][start:stop]-z['raw_time_s'][start])*1000
            baseline=z['baseline_mean'][int(r.waveform_index_in_archive)]
        ax.plot(times,signal,lw=.8,color='#253746'); ax.axhline(baseline,ls='--',color='gray',lw=.8)
        probs=' '.join(f"{fs.split('_')[1].upper()}={r[fs+'_p_f8']:.2f}" for fs in sets)
        ax.set(title=f'{event_id}\n{probs}',xlabel='Time from event start (ms)',ylabel='Raw current (pA)')
        ax.title.set_fontsize(8)
    for ax in list(axs.flat)[len(selected):]: ax.axis('off')
    fig.savefig(wave/'uncertain_waveform_montage.png',dpi=160); plt.close(fig)
    pd.DataFrame(waveform_checks).to_csv(wave/'waveform_integrity.csv',index=False)
    qc=pd.read_csv(inference/'recording_qc_summary.csv')
    def table(df):
        def cell(x):
            return str(x).replace('|', '/').replace('\n', ' ')
        lines=['| '+' | '.join(map(cell,df.columns))+' |', '| '+' | '.join(['---']*len(df.columns))+' |']
        lines += ['| '+' | '.join(map(cell,row))+' |' for row in df.itertuples(index=False,name=None)]
        return '\n'.join(lines)
    report=f'''# External mixture testing report

## 1. Dataset and conditions
HSA:F8 = 1:1, 1:19 and 19:1; 300 nM total; -100 mV; shared mixture_pore_01.
The user-confirmed corrected manifest is used. CSV 1:19 embeds File: 26714000, conflicting with its assigned 26709003 ABF; CSV 1:1 has no embedded ABF ID. These source metadata anomalies are logged, with no edits to raw inputs. Manifest checksum and annotation/sample boundaries are checked, but these checks cannot independently establish annotation-to-waveform identity. Confirm export provenance before biological interpretation.

## 2. Extraction / QC
{table(qc)}

Rejected reasons are recorded in qc_reason_summary.csv. Accepted means extraction accepted, not all shape features passed QC. Missing shape features are retained and processed by the already-fitted pure-training median imputer; no mixture imputation is fitted. Feature missingness is reported separately. Reconstructed boundaries are retained and flagged, with no additional exclusions.

## 3. Models and feature sets
Locked Random Forest pipelines; seeds 11 and 42; 432 HSA + 432 F8 events per seed. C: core + shape; D: core + 10 normalized blockage segments; E: segments only. Feature names/order, artifact checksums, class mapping, seed and training-source provenance checked. No mixture event or recording overlaps training selection. No training, calibration or parameter selection was performed here. Ensemble is the arithmetic mean of the two seeds; F8 if P(F8) >= {threshold}.

## 4. Mixture results
{table(summary[['mixture_id','feature_set','n_events','nominal_f8_fraction','predicted_f8_fraction','predicted_hsa_fraction','seed_agreement_fraction']])}

Full counts and probability mean, median, Q25/Q75 are in mixture_prediction_summary.csv. Mixture 19:1 has only 49 accepted events; each event changes its fraction by about 2.04 percentage points.

## 5. Seed and feature-set agreement
{table(agreement)}

There are {int(review.seed_disagreement.sum())} events with seed disagreement, {int(review.feature_set_disagreement.sum())} with feature-set disagreement and {int(review.near_threshold.sum())} near the fixed threshold (absolute distance <= 0.10 in any set). Threshold sensitivity evaluates 0.1 through 0.9 descriptively; no threshold is selected to match nominal ratios.

## 6. Detection bias
Predicted event fraction is not molar concentration. Descriptive predicted-to-nominal odds combine classifier error, capture, annotation and QC effects. This dataset cannot identify those effects separately. No detection-efficiency fit or mixture-based calibration is performed. The nominal diagonal is a visual reference only, not an expected accuracy target.

## 7. Waveform review
All {int(review.review_required.sum())} flagged events have indexed raw waveform provenance and exact archive-to-ABF sample checks. The montage displays up to four flagged events per recording, prioritizing seed and feature-set disagreement. It shows signed raw current with the extraction baseline. Generation and integrity verification do not constitute expert biological event labeling. The 19:1 mixture has no events meeting these uncertainty/disagreement criteria; its absence from the montage is explicit rather than a lost selection. Full flags are in all_event_review_flags.csv; uncertain_event_review.csv contains every flagged event, including those outside the montage.

## 8. Limitations
One shared pore controls pore variation within mixtures but does not remove temporal drift or training-to-mixture pore differences. Training metadata reports one HSA recording and three F8 recordings sharing a pore. Missing shape features and annotation/QC selection may shift predictions. Events within recordings are not independent biological replicates. RF probabilities are uncalibrated scores; two-seed agreement does not measure accuracy. No event-level truth exists: no confusion matrix is produced.

## 9. Next experiments
Use independent replicate pores and repeated pure controls under matching conditions; randomize ratio order to examine drift. Measure observation time and detection/annotation yield. Any classifier calibration must use held-out pure-protein validation only. A separate controlled calibration experiment is required before concentration estimation; these mixtures remain external tests.

Output root: {root}
Inference: {inference}
Visualization: {viz}
Waveform review: {wave}
'''
    (report_dir/'REPORT.md').write_text(report)
    manifest.update(analysis_code_sha256={str(Path(__file__).resolve()):sha256(__file__)},
                    waveform_archive_sha256=hashes, uncertainty_margin=.1,
                    analysis_counts=dict(flagged=int(review.review_required.sum()),montage=len(selected),raw_verified=len(waveform_checks)),
                    output_directories=[str(p) for p in [inference,viz,wave,report_dir]])
    # Recheck inputs after analysis and reopen every deliverable.
    for p,h in manifest['validation']['input_sha256'].items(): require(sha256(p)==h,f'Input changed during run: {p}')
    from PIL import Image
    outputs={}
    for directory in [inference,viz,wave,report_dir]:
        for p in directory.iterdir():
            if not p.is_file() or p.name=='run_manifest.json': continue
            if p.suffix=='.csv': pd.read_csv(p)
            elif p.suffix=='.png':
                with Image.open(p) as im: im.verify()
            elif p.suffix=='.md': p.read_text()
            outputs[str(p)]=sha256(p)
    manifest['output_sha256']=outputs
    for p in [inference/'run_manifest.json',report_dir/'run_manifest.json']:
        p.write_text(json.dumps(manifest,indent=2)); json.loads(p.read_text())
    print(report_dir)

if __name__=='__main__': main()
