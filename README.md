# Nanopore Protein Classification

Reproducible Python pipeline for extracting solid-state nanopore blockade events
from ABF recordings, engineering waveform-derived features, visualizing HSA and
F8 event distributions, exploring PCA/UMAP representations, and evaluating
protein-classification models.

The scientific motivation follows the senior project *AI-assisted Analysis of
Nanopore Current Blockades for Single Molecule Kidney Disease Biomarker
Identification*. The project studies ionic-current blockade patterns associated
with Human Serum Albumin (HSA) and Fragment Factor VIII (F8). This repository is
the updated code pipeline; the original report's numerical results and the V9
results must be treated as separate experimental versions.

## Pipeline

1. Pair each annotation table with its ABF recording through a manifest.
2. Validate event boundaries, time units, sweep indices, and recording offsets.
3. Extract each accepted event and its pre-event baseline from the raw ABF trace.
4. Save provenance-preserving raw waveforms and event-level QC records.
5. Calculate scalar, shape, and compact temporal-segmentation features.
6. Generate distribution, scatter, correlation, PCA, and UMAP analyses.
7. Evaluate KNN, Random Forest, and XGBoost with hyperparameter tuning and
   repeated balanced-training views.

## Current feature contract

Scalar and shape features include dwell time, annotation-supplied blockage
percentage, ABF-derived mean and standard deviation of blockage, AUC, skewness,
kurtosis, FWHM, rise/fall times, and left/right slopes.

Compact temporal segmentation divides every event into ten equal normalized-time
intervals and exports only the mean ABF-derived blockage for each interval:
`segment_01_blockage_mean` through `segment_10_blockage_mean`. Every interval
requires at least three raw samples; otherwise all ten segment values remain
missing and the reason is recorded in QC.

## Repository layout

```text
event_extraction.py       annotation/ABF loading, event extraction, provenance, QC
feature_engineering.py    numerical feature definitions
visualization.py          distributions, scatter plots, correlations, summaries
unsupervised_analysis.py  PCA, UMAP, and exploratory clustering diagnostics
exploratory_ml.py         KNN, Random Forest, and XGBoost evaluation
project_config.py         path resolution, versioned outputs, run manifests
config/                   public example configurations
tests/                    unit and pipeline tests
```

## Installation

Python 3.11 or later is recommended.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Data manifest

Raw recordings and annotations are intentionally not included. Create a local
`recordings.csv` using `recordings.example.csv` as a template. The manifest is
the authority that pairs annotations with recordings.

Required fields are:

```text
protein, concentration_nM, annotation_id, recording_id, annotation_path,
annotation_format, abf_path, sweep_index, event_time_unit, time_reference,
time_offset_s, enabled
```

The current tabular workflow expects event boundaries in `Event start` and
`Event end`. Exactly equal boundaries may be reconstructed symmetrically around
the reported time only when a finite positive source `Dwell time(ms)` exists.
Original and corrected boundaries are both retained.

## Run the pipeline

Copy the public example config to a local config and adjust only local paths and
experiment settings.

```bash
cp config/experiment.example.json config/experiment.local.json
python event_extraction.py --config config/experiment.local.json
python visualization.py --config config/experiment.local.json \
  --extraction-run-directory /path/to/completed/extraction/run
```

PCA and UMAP:

```bash
python unsupervised_analysis.py \
  --input /path/to/abf_event_features.csv \
  --output /path/to/pca_umap_run \
  --feature-set all_calculated_with_segments
```

Model evaluation:

```bash
cp config/exploratory_ml.example.json config/exploratory_ml.local.json
python exploratory_ml.py --config config/exploratory_ml.local.json
```

## Reproducibility and scientific limitations

- Every completed stage writes to a new run directory and records input paths,
  hashes, parameters, package versions, random seeds, and row counts.
- Rejected events remain in machine-readable QC tables with explicit reasons.
- Protein labels never determine event boundaries or segmentation.
- Event-level train/test splits can leak recording-specific information.
- With only one independent HSA recording, protein and recording effects cannot
  be separated and model scores are exploratory rather than estimates of
  biological generalization.
- Raw ABF files, annotation tables, extracted waveforms, generated results, and
  trained model files are excluded from this public repository.

## Tests

```bash
python -m pytest -q
```

## Data availability

Experimental recordings and annotations are not distributed in this repository.
Users must supply authorized local data and update the manifest paths.

## License

No open-source license has been selected yet. Copyright remains with the
repository owner unless a license is added.

