"""Apply locked balanced pure-protein models to extracted mixture events."""
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mixture_validation import audit_inputs, validate_predictions, require, sha256


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature-table", required=True)
    parser.add_argument("--model-run", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args(argv)

    feature_table = Path(args.feature_table).resolve()
    model_run = Path(args.model_run).resolve()
    output_root = Path(args.output_root).resolve()
    if not 0.0 < args.threshold < 1.0:
        raise ValueError("threshold must be between 0 and 1")

    events = pd.read_csv(feature_table, dtype={"recording_id": str}, low_memory=False)
    models = pd.read_csv(model_run / "trained_models.csv")
    required_metadata = {"event_id", "recording_id", "mixture_id", "hsa_fraction", "f8_fraction"}
    missing_metadata = sorted(required_metadata - set(events.columns))
    if missing_metadata:
        raise ValueError(f"Mixture feature table is missing metadata: {missing_metadata}")
    if events["event_id"].duplicated().any():
        raise ValueError("Mixture feature table contains duplicated event_id values")

    audit, qc = audit_inputs(feature_table, model_run, events, models)
    prediction_rows = []
    for row in models.itertuples(index=False):
        features = json.loads(row.features)
        missing_features = sorted(set(features) - set(events.columns))
        if missing_features:
            raise ValueError(
                f"{row.feature_set} is missing features: {missing_features}"
            )
        model = joblib.load(row.model_path)
        if list(model.feature_names_in_) != features:
            raise ValueError(f"Feature order mismatch for {row.model_path}")
        require(list(model.classes_) == [0, 1], "Expected classes 0=HSA, 1=F8")
        require(model.named_steps['model'].random_state == int(row.seed), "Model seed mismatch")
        probability_f8 = model.predict_proba(events[features])[:, 1]
        for event, probability in zip(events.itertuples(index=False), probability_f8):
            prediction_rows.append({
                "event_id": event.event_id,
                "recording_id": event.recording_id,
                "mixture_id": event.mixture_id,
                "hsa_to_f8_ratio": getattr(event, "hsa_to_f8_ratio", ""),
                "nominal_hsa_fraction": event.hsa_fraction,
                "nominal_f8_fraction": event.f8_fraction,
                "pore_id": getattr(event, "pore_id", ""),
                "feature_set": row.feature_set,
                "seed": int(row.seed),
                "probability_f8": float(probability),
            })

    per_seed = pd.DataFrame(prediction_rows)
    validate_predictions(
        per_seed,
        events.event_id.tolist(),
        feature_sets=set(models.feature_set),
    )
    keys = [
        "event_id", "recording_id", "mixture_id", "hsa_to_f8_ratio",
        "nominal_hsa_fraction", "nominal_f8_fraction", "pore_id", "feature_set",
    ]
    ensemble = (
        per_seed.groupby(keys, as_index=False, dropna=False)
        .agg(
            probability_f8_mean=("probability_f8", "mean"),
            probability_f8_sd=("probability_f8", "std"),
            n_models=("seed", "nunique"),
        )
    )
    ensemble["predicted_class"] = np.where(
        ensemble["probability_f8_mean"] >= args.threshold, "F8", "HSA"
    )
    ensemble["seed_agreement"] = (
        per_seed.assign(
            predicted_seed=lambda frame: np.where(
                frame.probability_f8 >= args.threshold, "F8", "HSA"
            )
        )
        .groupby(keys, dropna=False)["predicted_seed"]
        .nunique()
        .eq(1)
        .to_numpy()
    )

    require(ensemble.n_models.eq(2).all(), "Ensemble requires two seeds")
    summary = (
        ensemble.groupby(
            ["mixture_id", "hsa_to_f8_ratio", "nominal_hsa_fraction",
             "nominal_f8_fraction", "pore_id", "feature_set"],
            as_index=False,
            dropna=False,
        )
        .agg(
            n_events=("event_id", "size"),
            predicted_f8_events=("predicted_class", lambda x: int((x == "F8").sum())),
            predicted_hsa_events=("predicted_class", lambda x: int((x == "HSA").sum())),
            mean_probability_f8=("probability_f8_mean", "mean"),
            median_probability_f8=("probability_f8_mean", "median"),
            q25_probability_f8=("probability_f8_mean", lambda x: x.quantile(0.25)),
            q75_probability_f8=("probability_f8_mean", lambda x: x.quantile(0.75)),
            seed_agreement_fraction=("seed_agreement", "mean"),
        )
    )
    summary["predicted_f8_fraction"] = (
        summary["predicted_f8_events"] / summary["n_events"]
    )
    summary["predicted_hsa_fraction"] = (
        summary["predicted_hsa_events"] / summary["n_events"]
    )

    agreement_wide = ensemble.pivot(
        index="event_id", columns="feature_set", values="predicted_class"
    )
    agreement = pd.DataFrame([
        {
            "feature_set_a": left,
            "feature_set_b": right,
            "n_shared_events": int(agreement_wide[[left, right]].dropna().shape[0]),
            "agreement_fraction": float(
                (agreement_wide[left] == agreement_wide[right]).mean()
            ),
        }
        for i, left in enumerate(agreement_wide.columns)
        for right in agreement_wide.columns[i + 1:]
    ])

    run_id = datetime.now().strftime("mixture_inference_%Y%m%d_%H%M%S")
    output = output_root / run_id
    output.mkdir(parents=True, exist_ok=False)
    pd.DataFrame(audit['checks']).to_csv(output / 'recording_qc_summary.csv', index=False)
    qc.assign(qc_reason=qc.qc_reason.fillna('accepted')).groupby(
        ['recording_id', 'extraction_qc_pass', 'qc_reason'], dropna=False
    ).size().rename('n_events').reset_index().to_csv(output / 'qc_reason_summary.csv', index=False)
    missingness = []
    for feature_set, columns in models.groupby('feature_set').features.first().items():
        for mixture, group in events.groupby('mixture_id'):
            for column in json.loads(columns):
                missingness.append(dict(mixture_id=mixture, feature_set=feature_set, feature=column,
                                        n_missing=int(group[column].isna().sum()), n_events=len(group)))
    pd.DataFrame(missingness).to_csv(output / 'feature_missingness.csv', index=False)
    per_seed.to_csv(output / "per_seed_predictions.csv", index=False)
    ensemble.to_csv(output / "ensemble_event_predictions.csv", index=False)
    summary.to_csv(output / "mixture_prediction_summary.csv", index=False)
    agreement.to_csv(output / "feature_set_agreement.csv", index=False)

    ordered_mixtures = (
        summary[["mixture_id", "nominal_f8_fraction"]]
        .drop_duplicates()
        .sort_values("nominal_f8_fraction")
    )
    mixture_order = ordered_mixtures["mixture_id"].tolist()
    feature_order = sorted(summary["feature_set"].unique())
    x = np.arange(len(mixture_order))
    width = 0.24
    fig, ax = plt.subplots(figsize=(9, 5), constrained_layout=True)
    for index, feature_set in enumerate(feature_order):
        values = (
            summary[summary.feature_set == feature_set]
            .set_index("mixture_id")
            .loc[mixture_order, "predicted_f8_fraction"]
        )
        ax.bar(x + (index - 1) * width, values, width, label=feature_set)
    nominal = ordered_mixtures.set_index("mixture_id").loc[
        mixture_order, "nominal_f8_fraction"
    ]
    ax.scatter(x, nominal, marker="D", s=70, color="black", label="Nominal F8 fraction")
    ax.set_xticks(x, mixture_order)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("F8 fraction")
    ax.set_title("Nominal mixture ratio vs event-level predicted F8 fraction")
    ax.legend(fontsize=8)
    fig.savefig(output / "nominal_vs_predicted_f8_fraction.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, len(feature_order), figsize=(14, 4),
                             sharey=True, constrained_layout=True)
    for ax, feature_set in zip(axes, feature_order):
        subset = ensemble[ensemble.feature_set == feature_set]
        groups = [
            subset.loc[subset.mixture_id == mixture, "probability_f8_mean"].to_numpy()
            for mixture in mixture_order
        ]
        ax.boxplot(groups, tick_labels=mixture_order, showfliers=False)
        ax.axhline(args.threshold, color="red", linestyle="--", linewidth=1)
        ax.set_title(feature_set)
        ax.tick_params(axis="x", rotation=25)
        ax.set_xlabel("Mixture")
    axes[0].set_ylabel("Ensemble P(F8)")
    fig.suptitle("Mixture event probability distributions")
    fig.savefig(output / "mixture_probability_distributions.png", dpi=180)
    plt.close(fig)
    (output / "run_manifest.json").write_text(json.dumps({
        "run_type": "mixture_inference",
        "validation": audit,
        "event_counts": {"accepted": len(events), "per_seed": len(per_seed), "ensemble": len(ensemble)},
        "code_sha256": {str(Path(__file__).resolve()): sha256(__file__),
                         str(Path(__file__).with_name('mixture_validation.py').resolve()): sha256(Path(__file__).with_name('mixture_validation.py'))},
        "created_at": datetime.now().astimezone().isoformat(),
        "feature_table": str(feature_table),
        "model_run": str(model_run),
        "threshold": args.threshold,
        "feature_sets": sorted(models.feature_set.unique().tolist()),
        "seeds": sorted(models.seed.astype(int).unique().tolist()),
        "interpretation_limit": (
            "Predicted event fractions are classification outputs and are not "
            "calibrated estimates of molar concentration."
        ),
    }, indent=2), encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
