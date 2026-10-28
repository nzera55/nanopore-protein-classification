"""Evaluate locked balanced Random Forest settings on one untouched holdout set."""
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
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import balanced_accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler

from train_mixture_models import load_config


def save_matrix(path, counts, title):
    normalized = counts / counts.sum(axis=1, keepdims=True)
    labels = ["HSA", "F8"]
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
        ax.set_xticks([0, 1], labels)
        ax.set_yticks([0, 1], labels)
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
    parser.add_argument("--test-fraction", type=float, default=0.2)
    parser.add_argument("--split-seed", type=int, default=11)
    args = parser.parse_args(argv)
    config = load_config(args.config)

    data = pd.read_csv(config["feature_table"], dtype={"recording_id": str}, low_memory=False)
    data = data[data.protein.isin([config["negative_label"], config["positive_label"]])].copy()
    data["target"] = (data.protein == config["positive_label"]).astype(int)
    train_index, test_index = train_test_split(
        np.arange(len(data)), test_size=args.test_fraction,
        random_state=args.split_seed, stratify=data["target"],
    )
    train_pool = data.iloc[train_index].copy()
    test = data.iloc[test_index].copy()

    run_id = datetime.now().strftime("balanced_2seed_holdout_%Y%m%d_%H%M%S")
    output = Path(config["output_root"]).parents[1] / "model_evaluation" / "runs" / run_id
    matrix_directory = output / "confusion_matrices"
    matrix_directory.mkdir(parents=True, exist_ok=False)

    split_rows = []
    for role, subset in (("train_pool", train_pool), ("test", test)):
        for row in subset.itertuples(index=False):
            split_rows.append({
                "role": role, "event_id": row.event_id,
                "recording_id": row.recording_id, "protein": row.protein,
            })

    metrics = []
    selections = []
    matrices = {}
    for seed in config["seeds"]:
        negative = train_pool[train_pool.protein == config["negative_label"]]
        positive = train_pool[train_pool.protein == config["positive_label"]]
        sampled_negative = negative.sample(n=len(positive), replace=False, random_state=seed)
        training = pd.concat([sampled_negative, positive], ignore_index=True)
        training = training.sample(frac=1.0, random_state=seed).reset_index(drop=True)
        for row in training.itertuples(index=False):
            selections.append({
                "seed": seed, "event_id": row.event_id,
                "recording_id": row.recording_id, "protein": row.protein,
            })

        for feature_set, features in config["feature_sets"].items():
            parameters = config["random_forest_parameters"][feature_set][str(seed)]
            pipeline = Pipeline([
                ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
                ("scaler", RobustScaler()),
                ("model", RandomForestClassifier(
                    random_state=seed, n_jobs=1, **parameters
                )),
            ])
            pipeline.fit(training[features],
                         (training.protein == config["positive_label"]).astype(int))
            prediction = pipeline.predict(test[features])
            counts = confusion_matrix(test.target, prediction, labels=[0, 1])
            normalized = save_matrix(
                matrix_directory / f"{feature_set}_seed_{seed}.png", counts,
                f"{feature_set} | balanced training seed {seed} | untouched test",
            )
            matrices.setdefault(feature_set, []).append(counts.astype(float))
            tn, fp, fn, tp = counts.ravel()
            metrics.append({
                "feature_set": feature_set, "seed": seed,
                "balanced_accuracy": balanced_accuracy_score(test.target, prediction),
                "tn": tn, "fp": fp, "fn": fn, "tp": tp,
                "sensitivity_f8": tp / (tp + fn),
                "specificity_hsa": tn / (tn + fp),
                "n_train_hsa": int((training.protein == config["negative_label"]).sum()),
                "n_train_f8": int((training.protein == config["positive_label"]).sum()),
                "n_test_hsa": int((test.protein == config["negative_label"]).sum()),
                "n_test_f8": int((test.protein == config["positive_label"]).sum()),
            })

    mean_rows = []
    for feature_set, values in matrices.items():
        mean_counts = np.mean(values, axis=0)
        normalized = save_matrix(
            matrix_directory / f"{feature_set}_mean_across_2_seeds.png",
            mean_counts,
            f"{feature_set} | mean across seeds 11 and 42 | same untouched test",
        )
        tn, fp, fn, tp = mean_counts.ravel()
        mean_rows.append({
            "feature_set": feature_set, "n_seeds": 2,
            "mean_tn": tn, "mean_fp": fp, "mean_fn": fn, "mean_tp": tp,
            "mean_hsa_correct_fraction": normalized[0, 0],
            "mean_f8_correct_fraction": normalized[1, 1],
            "mean_balanced_accuracy": (normalized[0, 0] + normalized[1, 1]) / 2,
        })
        pd.DataFrame(mean_counts, index=["HSA", "F8"], columns=["HSA", "F8"]).rename_axis(
            "actual"
        ).to_csv(matrix_directory / f"{feature_set}_mean_counts.csv")
        pd.DataFrame(normalized, index=["HSA", "F8"], columns=["HSA", "F8"]).rename_axis(
            "actual"
        ).to_csv(matrix_directory / f"{feature_set}_mean_normalized.csv")

    pd.DataFrame(split_rows).to_csv(output / "split_assignments.csv", index=False)
    pd.DataFrame(selections).to_csv(output / "balanced_training_selection.csv", index=False)
    pd.DataFrame(metrics).to_csv(output / "per_seed_metrics.csv", index=False)
    pd.DataFrame(mean_rows).to_csv(output / "two_seed_mean_metrics.csv", index=False)
    (output / "run_manifest.json").write_text(json.dumps({
        "run_type": "balanced_two_seed_holdout_evaluation",
        "created_at": datetime.now().astimezone().isoformat(),
        "feature_table": config["feature_table"],
        "seeds": config["seeds"],
        "split_seed": args.split_seed,
        "test_fraction": args.test_fraction,
        "training_balance_rule": "downsample HSA to F8 count after splitting",
        "test_balance_rule": "untouched natural held-out distribution",
        "same_test_events_across_seeds": True,
        "generalization_claim_allowed": False,
        "limitation": "Event-level split contains recording leakage because HSA has one recording."
    }, indent=2), encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
