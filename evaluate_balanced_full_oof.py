"""Five-fold out-of-fold evaluation on full 432:432 balanced datasets."""
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import balanced_accuracy_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler

from train_mixture_models import load_config


LABELS = ["HSA", "F8"]


def save_matrix(path, counts, title):
    normalized = counts / counts.sum(axis=1, keepdims=True)
    fig, axes = plt.subplots(1, 2, figsize=(9, 4), constrained_layout=True)
    for ax, matrix, subtitle, fmt in (
        (axes[0], counts, "Counts", ".1f"),
        (axes[1], normalized, "Row-normalized", ".3f"),
    ):
        image = ax.imshow(matrix, cmap="Blues", vmin=0)
        for row in range(2):
            for column in range(2):
                ax.text(column, row, format(matrix[row, column], fmt),
                        ha="center", va="center")
        ax.set_xticks([0, 1], LABELS)
        ax.set_yticks([0, 1], LABELS)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_title(subtitle)
        fig.colorbar(image, ax=ax, fraction=0.046)
    fig.suptitle(title)
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return normalized


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--folds", type=int, default=5)
    args = parser.parse_args(argv)
    config = load_config(args.config)
    data = pd.read_csv(config["feature_table"], dtype={"recording_id": str}, low_memory=False)
    hsa = data[data.protein == config["negative_label"]].copy()
    f8 = data[data.protein == config["positive_label"]].copy()
    if len(hsa) < len(f8):
        raise ValueError("HSA must contain at least as many events as F8")

    run_id = datetime.now().strftime("balanced_432_each_5fold_oof_2seed_%Y%m%d_%H%M%S")
    output = Path(config["output_root"]).parents[1] / "model_evaluation" / "runs" / run_id
    matrices_directory = output / "confusion_matrices"
    matrices_directory.mkdir(parents=True, exist_ok=False)

    metrics = []
    selections = []
    predictions = []
    matrices = {}
    for seed in config["seeds"]:
        balanced = pd.concat([
            hsa.sample(n=len(f8), replace=False, random_state=seed),
            f8,
        ], ignore_index=True).sample(frac=1.0, random_state=seed).reset_index(drop=True)
        target = (balanced.protein == config["positive_label"]).astype(int)
        for row in balanced.itertuples(index=False):
            selections.append({
                "seed": seed, "event_id": row.event_id,
                "recording_id": row.recording_id, "protein": row.protein,
            })

        cv = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=seed)
        fold_by_position = np.empty(len(balanced), dtype=int)
        for fold, (_, test_positions) in enumerate(cv.split(balanced, target)):
            fold_by_position[test_positions] = fold

        for feature_set, features in config["feature_sets"].items():
            parameters = config["random_forest_parameters"][feature_set][str(seed)]
            pipeline = Pipeline([
                ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
                ("scaler", RobustScaler()),
                ("model", RandomForestClassifier(
                    random_state=seed, n_jobs=1, **parameters
                )),
            ])
            predicted = cross_val_predict(
                clone(pipeline), balanced[features], target, cv=cv,
                method="predict", n_jobs=1,
            )
            counts = confusion_matrix(target, predicted, labels=[0, 1])
            normalized = save_matrix(
                matrices_directory / f"{feature_set}_seed_{seed}.png",
                counts.astype(float),
                f"{feature_set} | seed {seed} | {args.folds}-fold out-of-fold | HSA=F8=432",
            )
            matrices.setdefault(feature_set, []).append(counts.astype(float))
            tn, fp, fn, tp = counts.ravel()
            metrics.append({
                "feature_set": feature_set, "seed": seed, "folds": args.folds,
                "n_hsa": 432, "n_f8": 432,
                "tn": tn, "fp": fp, "fn": fn, "tp": tp,
                "specificity_hsa": normalized[0, 0],
                "sensitivity_f8": normalized[1, 1],
                "balanced_accuracy": balanced_accuracy_score(target, predicted),
            })
            for position, row in enumerate(balanced.itertuples(index=False)):
                predictions.append({
                    "feature_set": feature_set, "seed": seed,
                    "fold": int(fold_by_position[position]),
                    "event_id": row.event_id, "recording_id": row.recording_id,
                    "actual": row.protein,
                    "predicted": LABELS[int(predicted[position])],
                })

    mean_rows = []
    for feature_set, values in matrices.items():
        mean_counts = np.mean(values, axis=0)
        normalized = save_matrix(
            matrices_directory / f"{feature_set}_mean_across_2_seeds.png",
            mean_counts,
            f"{feature_set} | mean across seeds 11 and 42 | 5-fold OOF | HSA=F8=432",
        )
        tn, fp, fn, tp = mean_counts.ravel()
        mean_rows.append({
            "feature_set": feature_set, "n_seeds": 2, "folds": args.folds,
            "n_hsa_per_seed": 432, "n_f8_per_seed": 432,
            "mean_tn": tn, "mean_fp": fp, "mean_fn": fn, "mean_tp": tp,
            "mean_specificity_hsa": normalized[0, 0],
            "mean_sensitivity_f8": normalized[1, 1],
            "mean_balanced_accuracy": normalized.diagonal().mean(),
        })
        pd.DataFrame(mean_counts, index=LABELS, columns=LABELS).rename_axis(
            "actual"
        ).to_csv(matrices_directory / f"{feature_set}_mean_counts.csv")
        pd.DataFrame(normalized, index=LABELS, columns=LABELS).rename_axis(
            "actual"
        ).to_csv(matrices_directory / f"{feature_set}_mean_normalized.csv")

    pd.DataFrame(selections).to_csv(output / "balanced_dataset_selection.csv", index=False)
    pd.DataFrame(predictions).to_csv(output / "out_of_fold_predictions.csv", index=False)
    pd.DataFrame(metrics).to_csv(output / "per_seed_metrics.csv", index=False)
    pd.DataFrame(mean_rows).to_csv(output / "two_seed_mean_metrics.csv", index=False)
    (output / "run_manifest.json").write_text(json.dumps({
        "run_type": "balanced_full_dataset_five_fold_out_of_fold_evaluation",
        "created_at": datetime.now().astimezone().isoformat(),
        "feature_table": config["feature_table"],
        "seeds": config["seeds"], "folds": args.folds,
        "n_hsa_per_seed": 432, "n_f8_per_seed": 432,
        "balance_rule": "sample HSA without replacement to equal all 432 F8 events",
        "evaluation_rule": "each confusion-matrix prediction is out-of-fold",
        "generalization_claim_allowed": False,
        "limitation": "Folds contain events from the same physical recordings; recording leakage remains."
    }, indent=2), encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
