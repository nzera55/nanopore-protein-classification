"""Explicitly limited exploratory HSA/F8 experiments.

E1 intentionally permits recording leakage for software/exploratory analysis.
E2 holds out one F8 recording at a time but cannot evaluate independent HSA.
Neither experiment estimates biological generalization.
"""
from __future__ import annotations

import argparse
import json
import platform
import time
from datetime import datetime
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (average_precision_score, balanced_accuracy_score,
                             confusion_matrix, f1_score, precision_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from xgboost import XGBClassifier

from machine_learning import BalancedFitClassifier, audit_dataset, load_config, sha256

SCOPE_E1 = "exploratory_with_recording_leakage"
SCOPE_E2 = "limited_f8_robustness_no_independent_hsa_evaluation"


def dependency_status():
    try:
        import xgboost
        return {"xgboost_available": True, "xgboost_blocker": None,
                "xgboost_version": xgboost.__version__, "xgboost_path": xgboost.__file__}
    except ImportError as exc:
        return {"xgboost_available": False,
                "xgboost_blocker": f"XGBoost unavailable; not installed automatically: {exc}"}


def exploratory_split(data, seed, test_fraction):
    train_idx, test_idx = train_test_split(
        np.arange(len(data)), test_size=test_fraction, random_state=seed,
        stratify=data["protein"])
    return np.asarray(train_idx), np.asarray(test_idx)


def leave_one_f8_recording_out(data, held_out, positive_label="F8"):
    test = (data["protein"] == positive_label) & (data["recording_id"].astype(str) == str(held_out))
    if not test.any():
        raise ValueError(f"No F8 rows for held-out recording {held_out}")
    train = ~test
    return np.flatnonzero(train), np.flatnonzero(test)


def model_spaces(seed, balanced):
    models = {
        "knn": (KNeighborsClassifier(), {"model__n_neighbors": [5, 15],
                                         "model__weights": ["uniform", "distance"],
                                         "model__p": [1, 2]}),
        "random_forest": (RandomForestClassifier(random_state=seed, n_jobs=1),
                          {"model__n_estimators": [100],
                           "model__max_depth": [None, 8],
                           "model__min_samples_leaf": [1, 5]}),
        "xgboost": (XGBClassifier(random_state=seed, n_jobs=1, eval_metric="logloss"),
                     {"model__n_estimators": [100], "model__max_depth": [3, 6],
                      "model__learning_rate": [0.05, 0.15],
                      "model__subsample": [0.8], "model__colsample_bytree": [0.8]}),
    }
    if balanced:
        return {name: (BalancedFitClassifier(est, random_state=seed),
                       {key.replace("model__", "model__estimator__"): value for key, value in grid.items()})
                for name, (est, grid) in models.items()}
    return models


def build_pipeline(estimator):
    return Pipeline([("imputer", SimpleImputer(strategy="median", add_indicator=True)),
                     ("scaler", RobustScaler()), ("model", estimator)])


def feature_sets(config, columns):
    sets = {name: list(values) for name, values in config["feature_sets"].items()}
    if not sets:
        raise ValueError("At least one feature set is required")
    forbidden = {"segmentation_qc_pass", "segmentation_qc_reason", "minimum_segment_points",
                 "insufficient_segment_count"}
    for name, values in sets.items():
        missing = sorted(set(values) - set(columns))
        leaked = sorted(set(values) & forbidden)
        if missing or leaked or len(values) != len(set(values)):
            raise ValueError(f"Invalid {name} feature set; missing={missing}, forbidden={leaked}")
    return sets


def evaluate(y, pred, score):
    labels = np.unique(y); both = len(labels) == 2
    cm = confusion_matrix(y, pred, labels=[0, 1]); tn, fp, fn, tp = cm.ravel()
    return {"balanced_accuracy": balanced_accuracy_score(y, pred) if both else np.nan,
            "roc_auc": roc_auc_score(y, score) if both else np.nan,
            "pr_auc": average_precision_score(y, score) if both else np.nan,
            "sensitivity": recall_score(y, pred, zero_division=0),
            "specificity": tn / (tn + fp) if tn + fp else np.nan,
            "precision": precision_score(y, pred, zero_division=0),
            "f1": f1_score(y, pred, zero_division=0),
            "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}


def _score(estimator, X):
    return estimator.predict_proba(X)[:, 1]


def confusion_arrays(y, pred):
    counts = confusion_matrix(y, pred, labels=[0, 1]).astype(int)
    denominators = counts.sum(axis=1, keepdims=True)
    normalized = np.divide(counts, denominators, out=np.full(counts.shape, np.nan, float),
                           where=denominators != 0)
    return counts, normalized


def save_confusion_artifacts(root, stem, y, pred, title):
    root.mkdir(parents=True, exist_ok=True)
    counts, normalized = confusion_arrays(y, pred)
    labels = ["HSA", "F8"]
    pd.DataFrame(counts, index=labels, columns=labels).rename_axis("actual").to_csv(root / f"{stem}_counts.csv")
    pd.DataFrame(normalized, index=labels, columns=labels).rename_axis("actual").to_csv(root / f"{stem}_normalized.csv")
    fig, axes = plt.subplots(1, 2, figsize=(9, 4), constrained_layout=True)
    for ax, matrix, subtitle, fmt in ((axes[0], counts, "Counts", "d"),
                                       (axes[1], normalized, "Row-normalized", ".3f")):
        image = ax.imshow(np.nan_to_num(matrix), cmap="Blues", vmin=0)
        for i in range(2):
            for j in range(2):
                value = matrix[i, j]
                text = "NA" if not np.isfinite(value) else format(value, fmt)
                ax.text(j, i, text, ha="center", va="center")
        ax.set_xticks([0, 1], labels); ax.set_yticks([0, 1], labels)
        ax.set_xlabel("Predicted"); ax.set_ylabel("Actual"); ax.set_title(subtitle)
        fig.colorbar(image, ax=ax, fraction=.046)
    fig.suptitle(f"{title} (n={len(y)})")
    fig.savefig(root / f"{stem}.png", dpi=160); plt.close(fig)
    return counts, normalized


def run_one(data, train_idx, test_idx, config, experiment, split_id, scope, output, all_feature_sets):
    train, test = data.iloc[train_idx].copy(), data.iloc[test_idx].copy()
    rows, cv_rows, best_rows, recording_rows, confusion_rows = [], [], [], [], []
    for feature_set, features in all_feature_sets.items():
      for arm, balanced in (("natural", False), ("balanced_training", True)):
        seeds = config["seeds"] if balanced else [config["seeds"][0]]
        for seed in seeds:
          cv = StratifiedKFold(n_splits=config["exploratory_cv_splits"], shuffle=True, random_state=seed)
          for model_name, (model, grid) in model_spaces(seed, balanced).items():
                started = time.perf_counter()
                search = GridSearchCV(build_pipeline(model), grid, scoring="balanced_accuracy", cv=cv,
                                      n_jobs=1, return_train_score=False, error_score="raise")
                search.fit(train[features], train["target"])
                for fold in range(config["exploratory_cv_splits"]):
                    cv_rows.append({"experiment": experiment, "scope": scope, "split_id": split_id,
                                    "feature_set": feature_set, "arm": arm, "seed": seed, "model": model_name, "fold": fold,
                                    "balanced_accuracy": search.cv_results_[f"split{fold}_test_score"][search.best_index_]})
                pred = search.best_estimator_.predict(test[features]); score = _score(search.best_estimator_, test[features])
                result = evaluate(test.target.to_numpy(), pred, score)
                result.update({"experiment": experiment, "scope": scope, "split_id": split_id,
                               "feature_set": feature_set, "arm": arm, "seed": seed, "model": model_name,
                               "runtime_seconds": time.perf_counter() - started,
                               "n_train": len(train), "n_test": len(test)})
                rows.append(result)
                stem = f"{experiment}_{split_id}_{feature_set}_{arm}_{seed}_{model_name}"
                counts, _ = save_confusion_artifacts(output / "confusion_matrices", stem,
                    test.target.to_numpy(), pred, f"{scope} | {arm} | {model_name} | seed {seed}")
                confusion_rows.append({"experiment": experiment, "scope": scope, "split_id": split_id,
                    "feature_set": feature_set, "arm": arm, "seed": seed, "model": model_name, "n_test": len(test),
                    "tn": int(counts[0, 0]), "fp": int(counts[0, 1]),
                    "fn": int(counts[1, 0]), "tp": int(counts[1, 1])})
                for recording_id, indices in test.groupby("recording_id").groups.items():
                    positions = test.index.get_indexer(indices)
                    record_result = evaluate(test.loc[indices, "target"].to_numpy(),
                                             pred[positions], score[positions])
                    record_result.update({"experiment": experiment, "scope": scope,
                                          "split_id": split_id, "arm": arm, "seed": seed,
                                          "feature_set": feature_set, "model": model_name, "recording_id": recording_id,
                                          "protein": test.loc[indices, "protein"].iloc[0],
                                          "n_events": len(indices)})
                    recording_rows.append(record_result)
                best_rows.append({"experiment": experiment, "scope": scope, "split_id": split_id,
                                  "arm": arm, "seed": seed, "model": model_name,
                                  "feature_set": feature_set,
                                  "best_cv_balanced_accuracy": search.best_score_,
                                  "best_params": json.dumps({k: str(v) for k, v in search.best_params_.items()})})
                joblib.dump(search.best_estimator_, output / f"model_{experiment}_{split_id}_{feature_set}_{arm}_{seed}_{model_name}.joblib")
    return rows, cv_rows, best_rows, recording_rows, confusion_rows


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(Path(__file__).parent / "config/exploratory_ml.json"))
    args = parser.parse_args(argv); config = load_config(args.config)
    data = pd.read_csv(config["feature_table"], dtype={"recording_id": str}, low_memory=False)
    duplicates = pd.read_csv(config["duplicate_table"])
    sets = feature_sets(config, data.columns)
    audit_config = dict(config)
    audit_config["features"] = list(dict.fromkeys(
        feature for values in sets.values() for feature in values
    ))
    audit = audit_dataset(data, audit_config, duplicates)
    data["target"] = (data.protein == config["positive_label"]).astype(int)
    run_id = datetime.now().strftime("nanopore_exploratory_ml_%Y%m%d_%H%M%S")
    output = Path(config["output_root"]) / run_id; output.mkdir(parents=True, exist_ok=False)
    dep = dependency_status()
    warnings = [
        "E1 is exploratory_with_recording_leakage: events from the same recording occur in train and test.",
        "E2 tests held-out F8 recordings only; HSA has no independent test recording.",
        "Inner CV is event-stratified and retains recording dependence because HSA has only one physical recording.",
        "Neither E1 nor E2 estimates biological generalization.",
    ]
    if dep["xgboost_blocker"]: warnings.append(dep["xgboost_blocker"])
    all_rows, all_cv, all_best, all_recording_rows, all_confusions, assignments = [], [], [], [], [], []
    tr, te = exploratory_split(data, config["seeds"][0], config["exploratory_test_fraction"])
    for role, idx in (("train", tr), ("test", te)):
        assignments.extend({"experiment": "E1", "scope": SCOPE_E1, "split_id": "seed_11", "role": role,
                            "event_id": data.iloc[i].event_id, "recording_id": data.iloc[i].recording_id,
                            "protein": data.iloc[i].protein} for i in idx)
    values = run_one(data, tr, te, config, "E1", "seed_11", SCOPE_E1, output, sets)
    for target, source in zip((all_rows, all_cv, all_best, all_recording_rows, all_confusions), values): target.extend(source)
    for held_out in sorted(data.loc[data.protein == config["positive_label"], "recording_id"].unique()):
        tr, te = leave_one_f8_recording_out(data, held_out, config["positive_label"])
        for role, idx in (("train", tr), ("test", te)):
            assignments.extend({"experiment": "E2", "scope": SCOPE_E2, "split_id": f"held_out_{held_out}", "role": role,
                                "event_id": data.iloc[i].event_id, "recording_id": data.iloc[i].recording_id,
                                "protein": data.iloc[i].protein} for i in idx)
        values = run_one(data, tr, te, config, "E2", f"held_out_{held_out}", SCOPE_E2, output, sets)
        for target, source in zip((all_rows, all_cv, all_best, all_recording_rows, all_confusions), values): target.extend(source)
    pd.DataFrame(assignments).to_csv(output / "split_assignments.csv", index=False)
    pd.DataFrame(all_rows).to_csv(output / "test_metrics.csv", index=False)
    pd.DataFrame(all_cv).to_csv(output / "cv_distribution.csv", index=False)
    pd.DataFrame(all_best).to_csv(output / "best_parameters.csv", index=False)
    pd.DataFrame(all_recording_rows).to_csv(output / "per_recording_metrics.csv", index=False)
    confusion_frame = pd.DataFrame(all_confusions)
    confusion_frame.to_csv(output / "confusion_matrix_index.csv", index=False)
    aggregate = (confusion_frame[confusion_frame.arm == "balanced_training"]
                 .groupby(["experiment", "scope", "split_id", "feature_set", "model"], as_index=False)
                 .agg(n_seeds=("seed", "nunique"), mean_tn=("tn", "mean"), mean_fp=("fp", "mean"),
                      mean_fn=("fn", "mean"), mean_tp=("tp", "mean")))
    aggregate.to_csv(output / "balanced_seed_mean_confusion_matrices.csv", index=False)
    aggregate_root = output / "confusion_matrices" / "balanced_seed_means"
    aggregate_root.mkdir(parents=True, exist_ok=True)
    for row in aggregate.itertuples(index=False):
        mean_counts = np.array([[row.mean_tn, row.mean_fp], [row.mean_fn, row.mean_tp]], float)
        denominators = mean_counts.sum(axis=1, keepdims=True)
        mean_normalized = np.divide(mean_counts, denominators,
            out=np.full((2, 2), np.nan), where=denominators != 0)
        stem = f"{row.experiment}_{row.split_id}_{row.feature_set}_{row.model}_mean_across_{row.n_seeds}_seeds"
        labels = ["HSA", "F8"]
        pd.DataFrame(mean_counts, index=labels, columns=labels).rename_axis("actual").to_csv(aggregate_root / f"{stem}_mean_counts.csv")
        pd.DataFrame(mean_normalized, index=labels, columns=labels).rename_axis("actual").to_csv(aggregate_root / f"{stem}_mean_normalized.csv")
        fig, axes = plt.subplots(1, 2, figsize=(9, 4), constrained_layout=True)
        for ax, matrix, subtitle in ((axes[0], mean_counts, "Mean counts"),
                                     (axes[1], mean_normalized, "Normalized mean counts")):
            ax.imshow(np.nan_to_num(matrix), cmap="Blues", vmin=0)
            for i in range(2):
                for j in range(2):
                    ax.text(j, i, "NA" if not np.isfinite(matrix[i, j]) else f"{matrix[i, j]:.3f}", ha="center", va="center")
            ax.set_xticks([0, 1], labels); ax.set_yticks([0, 1], labels)
            ax.set_xlabel("Predicted"); ax.set_ylabel("Actual"); ax.set_title(subtitle)
        fig.suptitle(f"Balanced training mean across {row.n_seeds} seeds; repeated same test set")
        fig.savefig(aggregate_root / f"{stem}.png", dpi=160); plt.close(fig)
    (output / "feature_sets.json").write_text(json.dumps(sets, indent=2))
    missingness = []
    for set_name, features in sets.items():
        for protein, subset in data.groupby("protein"):
            for feature in features:
                missingness.append({"feature_set": set_name, "protein": protein, "feature": feature,
                                    "n_missing": int(subset[feature].isna().sum()),
                                    "fraction_missing": float(subset[feature].isna().mean())})
    pd.DataFrame(missingness).to_csv(output / "feature_missingness.csv", index=False)
    (output / "search_space.json").write_text(json.dumps({"models": ["KNN", "Random Forest", "XGBoost"],
        "xgboost": dep, "balanced_seeds": config["seeds"]}, indent=2))
    (output / "warnings.json").write_text(json.dumps(warnings, indent=2))
    manifest = {"run_type": "exploratory_supervised_ml", "created_at": datetime.now().astimezone().isoformat(),
                "evaluation_scopes": {"E1": SCOPE_E1, "E2": SCOPE_E2}, "generalization_claim_allowed": False,
                "python_version": platform.python_version(), "feature_table": config["feature_table"],
                "feature_table_sha256": sha256(config["feature_table"]), "extraction_manifest": config["extraction_manifest"],
                "extraction_manifest_sha256": sha256(config["extraction_manifest"]), "dataset_audit": audit,
                "dependencies": dep, "seeds": config["seeds"], "warnings": warnings,
                "test_used_for_tuning": False,
                "confusion_matrix_policy": "Per-seed test matrices are primary; balanced_seed_mean_confusion_matrices.csv averages cell counts across seeds and does not pool repeated predictions."}
    (output / "run_manifest.json").write_text(json.dumps(manifest, indent=2))
    metrics_frame = pd.DataFrame(all_rows)
    e1 = metrics_frame[metrics_frame.experiment == "E1"]
    e2 = (metrics_frame[metrics_frame.experiment == "E2"]
          .groupby(["feature_set", "arm", "model"], as_index=False)[["sensitivity", "precision", "f1"]].mean())
    report = ["# Exploratory ML — NOT A GENERALIZATION ESTIMATE", "", *[f"- {w}" for w in warnings], "",
              "E1 uses an event-level stratified split and therefore has recording leakage by design.",
              "E2 holds out each F8 recording, but all HSA events remain in training and no independent HSA test exists.",
              "Natural and training-only balanced arms are reported separately; test rows were never balanced.", "",
              "## E1 event-level metrics (recording leakage)", "",
              "```text", e1[["feature_set", "arm", "seed", "model", "balanced_accuracy", "roc_auc", "pr_auc", "sensitivity", "specificity", "precision", "f1"]].to_string(index=False), "```", "",
              "## E2 mean held-out-F8 metrics", "",
              "Balanced accuracy, ROC AUC, PR AUC, and specificity are undefined because every E2 test fold contains F8 only.", "",
              "```text", e2.to_string(index=False), "```", "",
              "Full fold, seed, confusion-matrix, and per-recording results are stored in the CSV artifacts."]
    (output / "REPORT.md").write_text("\n".join(report) + "\n")
    print(output)


if __name__ == "__main__": main()
