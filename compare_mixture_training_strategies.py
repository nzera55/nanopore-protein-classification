"""Compare locked holdout-trained and full-432 models on external mixtures."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SETS = {
    "set_c_core_shape": "C",
    "set_d_core_segments": "D",
    "set_e_segment_only": "E",
}
SEEDS = (11, 42)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def summarize(events, threshold):
    ensemble = (events.groupby([
        "event_id", "recording_id", "mixture_id", "hsa_to_f8_ratio",
        "nominal_hsa_fraction", "nominal_f8_fraction", "feature_set",
        "training_strategy"], as_index=False)
        .agg(probability_f8_mean=("probability_f8", "mean"),
             probability_f8_sd=("probability_f8", "std"),
             n_seeds=("seed", "nunique")))
    require(ensemble.n_seeds.eq(2).all(), "Every ensemble requires seeds 11 and 42")
    ensemble["predicted_class"] = np.where(
        ensemble.probability_f8_mean >= threshold, "F8", "HSA")
    summary = (ensemble.groupby([
        "mixture_id", "hsa_to_f8_ratio", "nominal_hsa_fraction",
        "nominal_f8_fraction", "feature_set", "training_strategy"],
        as_index=False)
        .agg(n_events=("event_id", "size"),
             predicted_f8_events=("predicted_class", lambda x: int(x.eq("F8").sum())),
             mean_probability_f8=("probability_f8_mean", "mean"),
             median_probability_f8=("probability_f8_mean", "median"),
             q25_probability_f8=("probability_f8_mean", lambda x: x.quantile(.25)),
             q75_probability_f8=("probability_f8_mean", lambda x: x.quantile(.75))))
    summary["predicted_hsa_events"] = summary.n_events-summary.predicted_f8_events
    summary["predicted_f8_fraction"] = summary.predicted_f8_events/summary.n_events
    summary["predicted_hsa_fraction"] = 1-summary.predicted_f8_fraction
    return ensemble, summary


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature-table", required=True)
    parser.add_argument("--full-inference-run", required=True)
    parser.add_argument("--exploratory-run", required=True)
    parser.add_argument("--set-e-run", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--threshold", type=float, default=.5)
    args = parser.parse_args(argv)
    require(0 < args.threshold < 1, "Threshold must be between zero and one")
    feature_path = Path(args.feature_table).resolve()
    full_run = Path(args.full_inference_run).resolve()
    exploratory = Path(args.exploratory_run).resolve()
    set_e_run = Path(args.set_e_run).resolve()
    events = pd.read_csv(feature_path, dtype={"recording_id": str}, low_memory=False)
    require(events.event_id.is_unique, "Mixture event IDs must be unique")

    # The wrapper class must be importable under the module name stored in joblib.
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import machine_learning  # noqa: F401

    sources = []
    predictions = []
    for feature_set in SETS:
        base = set_e_run if feature_set == "set_e_segment_only" else exploratory
        for seed in SEEDS:
            model_path = base / (
                f"model_E1_seed_11_{feature_set}_balanced_training_"
                f"{seed}_random_forest.joblib")
            require(model_path.is_file(), f"Missing locked holdout model: {model_path}")
            model = joblib.load(model_path)
            features = list(model.feature_names_in_)
            require(set(features) <= set(events.columns), "Mixture feature mismatch")
            require(list(model.classes_) == [0, 1], "Expected classes 0=HSA, 1=F8")
            counts = model.named_steps["model"].fit_class_counts_
            require(counts == {"0": 346, "1": 346}, "Unexpected holdout training counts")
            probability = model.predict_proba(events[features])[:, 1]
            sources.append({
                "training_strategy": "holdout_then_balance_346_each",
                "feature_set": feature_set, "seed": seed,
                "n_hsa_train": 346, "n_f8_train": 346,
                "model_path": str(model_path), "model_sha256": sha256(model_path),
                "features": json.dumps(features),
            })
            for row, value in zip(events.itertuples(index=False), probability):
                predictions.append({
                    "event_id": row.event_id, "recording_id": row.recording_id,
                    "mixture_id": row.mixture_id,
                    "hsa_to_f8_ratio": row.hsa_to_f8_ratio,
                    "nominal_hsa_fraction": row.hsa_fraction,
                    "nominal_f8_fraction": row.f8_fraction,
                    "feature_set": feature_set, "seed": seed,
                    "training_strategy": "holdout_then_balance_346_each",
                    "probability_f8": float(value),
                })

    holdout_predictions = pd.DataFrame(predictions)
    require(len(holdout_predictions) == len(events)*len(SETS)*len(SEEDS),
            "Incomplete holdout-model prediction coverage")
    require(not holdout_predictions.duplicated(
        ["event_id", "feature_set", "seed"]).any(), "Duplicate predictions")
    full = pd.read_csv(full_run/"per_seed_predictions.csv", dtype={"recording_id": str})
    full = full.rename(columns={"nominal_hsa_fraction": "nominal_hsa_fraction",
                                "nominal_f8_fraction": "nominal_f8_fraction"})
    full["training_strategy"] = "full_balance_432_each"
    keep = list(holdout_predictions.columns)
    require(set(keep) <= set(full.columns), "Full-model prediction schema mismatch")
    combined = pd.concat([holdout_predictions, full[keep]], ignore_index=True)
    ensemble, summary = summarize(combined, args.threshold)

    wide = summary.pivot(index=["mixture_id", "feature_set"],
                         columns="training_strategy",
                         values="predicted_f8_fraction").reset_index()
    wide["difference_holdout_minus_full"] = (
        wide["holdout_then_balance_346_each"]-wide["full_balance_432_each"])
    wide = wide.merge(summary[["mixture_id", "feature_set", "nominal_f8_fraction"]]
                      .drop_duplicates(), on=["mixture_id", "feature_set"])

    run_id = datetime.now().strftime("compare_training_strategies_%Y%m%d_%H%M%S")
    output = Path(args.output_root).resolve()/run_id
    output.mkdir(parents=True, exist_ok=False)
    pd.DataFrame(sources).to_csv(output/"model_sources.csv", index=False)
    combined.to_csv(output/"per_seed_predictions.csv", index=False)
    ensemble.to_csv(output/"ensemble_event_predictions.csv", index=False)
    summary.to_csv(output/"training_strategy_prediction_summary.csv", index=False)
    wide.to_csv(output/"training_strategy_comparison.csv", index=False)

    mixtures = (summary[["mixture_id", "nominal_f8_fraction"]]
                .drop_duplicates().sort_values("nominal_f8_fraction").mixture_id.tolist())
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharey=True, constrained_layout=True)
    labels = {"holdout_then_balance_346_each": "Holdout then 346:346",
              "full_balance_432_each": "Full 432:432"}
    colors = {"holdout_then_balance_346_each": "#4C78A8",
              "full_balance_432_each": "#F58518"}
    for ax, mixture in zip(axes, mixtures):
        subset = summary[summary.mixture_id.eq(mixture)]
        x = np.arange(3)
        for offset, strategy in zip([-.18, .18], labels):
            values = (subset[subset.training_strategy.eq(strategy)]
                      .set_index("feature_set").loc[list(SETS), "predicted_f8_fraction"])
            ax.bar(x+offset, values, .36, label=labels[strategy], color=colors[strategy])
        nominal = subset.nominal_f8_fraction.iloc[0]
        ax.axhline(nominal, color="black", ls="--", lw=1, label="Nominal molar fraction")
        ax.set(title=mixture, xticks=x, xticklabels=list(SETS.values()),
               xlabel="Feature set", ylim=(0, 1.03))
    axes[0].set_ylabel("Predicted F8 event fraction")
    axes[-1].legend(fontsize=8, loc="lower left")
    fig.savefig(output/"training_strategy_comparison.png", dpi=180)
    plt.close(fig)

    manifest = {
        "run_type": "external_mixture_training_strategy_comparison",
        "created_at": datetime.now().astimezone().isoformat(),
        "threshold": args.threshold,
        "mixture_used_for_training_or_selection": False,
        "feature_table": str(feature_path),
        "feature_table_sha256": sha256(feature_path),
        "full_inference_run": str(full_run),
        "exploratory_holdout_run": str(exploratory),
        "set_e_holdout_run": str(set_e_run),
        "strategies": {
            "holdout_then_balance_346_each": {
                "split": "80% train pool / 20% natural imbalanced holdout, stratified by event",
                "model_training": "346 HSA + 346 F8 events per seed",
                "holdout": "2548 HSA + 86 F8 events; not used for mixture inference",
                "limitation": "event-level split has recording leakage",
            },
            "full_balance_432_each": {
                "model_training": "432 HSA + all 432 F8 events per seed",
                "limitation": "no separate pure holdout for these final fitted artifacts",
            },
        },
        "interpretation_limit": (
            "Predicted event fractions are not calibrated molar concentration estimates; "
            "differences also reflect training sample size and fitted hyperparameters."),
        "counts": {"accepted_mixture_events": len(events),
                   "per_seed_predictions": len(combined),
                   "ensemble_predictions": len(ensemble)},
    }
    for path in output.iterdir():
        if path.is_file():
            if path.suffix == ".csv": pd.read_csv(path)
            manifest.setdefault("output_sha256", {})[str(path)] = sha256(path)
    (output/"run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
