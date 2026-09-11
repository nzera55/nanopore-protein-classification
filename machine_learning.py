"""Leakage-safe grouped HSA/F8 classification experiments.

Real evaluation fails closed when either class lacks enough independent physical
recordings.  ``--synthetic`` exercises the complete workflow without presenting
synthetic scores as biological generalization evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.decomposition import PCA
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, balanced_accuracy_score,
                             confusion_matrix, f1_score, precision_score,
                             recall_score, roc_auc_score, make_scorer)
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.svm import SVC


if __name__ == "__main__":
    # Keep custom estimators importable when artifacts are created via CLI.
    sys.modules.setdefault("machine_learning", sys.modules[__name__])

LABEL_COLUMN = "protein"
FORBIDDEN_FEATURE_TOKENS = (
    "protein", "label", "class", "recording", "annotation", "event_id",
    "path", "reason", "qc", "source_", "index", "sampling_rate", "baseline",
)


class BalancedFitClassifier(ClassifierMixin, BaseEstimator):
    """Undersample only inside ``fit``; validation/test rows remain untouched."""

    def __init__(self, estimator=None, random_state=0):
        self.estimator = estimator
        self.random_state = random_state

    def fit(self, X, y):
        y_array = np.asarray(y)
        classes, counts = np.unique(y_array, return_counts=True)
        if len(classes) != 2:
            raise ValueError("BalancedFitClassifier requires two classes")
        target = int(counts.min())
        rng = np.random.default_rng(self.random_state)
        selected = np.concatenate([
            rng.choice(np.flatnonzero(y_array == label), target, replace=False)
            for label in classes
        ])
        rng.shuffle(selected)
        self.sampled_indices_ = selected.copy()
        base = self.estimator if self.estimator is not None else LogisticRegression(max_iter=3000)
        self.estimator_ = clone(base).fit(np.asarray(X)[selected], y_array[selected])
        self.classes_ = self.estimator_.classes_
        self.fit_class_counts_ = {str(k): target for k in classes}
        return self

    def predict(self, X):
        return self.estimator_.predict(X)

    def predict_proba(self, X):
        return self.estimator_.predict_proba(X)


BalancedFitClassifier.__module__ = "machine_learning"

def load_config(path):
    path = Path(path).resolve()
    config = json.loads(path.read_text(encoding="utf-8"))
    for key in ("feature_table", "extraction_manifest", "duplicate_table", "output_root"):
        config[key] = str((path.parent / config[key]).resolve())
    config["config_path"] = str(path)
    return config


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_features(data, features):
    missing = sorted(set(features) - set(data.columns))
    forbidden = sorted(f for f in features if any(t in f.lower() for t in FORBIDDEN_FEATURE_TOKENS))
    if missing or forbidden or len(features) != len(set(features)):
        raise ValueError(f"Invalid feature contract; missing={missing}, forbidden={forbidden}")
    nonnumeric = [f for f in features if not pd.api.types.is_numeric_dtype(data[f])]
    if nonnumeric:
        raise ValueError(f"Features must be numeric: {nonnumeric}")


def audit_dataset(data, config, duplicate_table=None):
    features = config["features"]
    group = config["group_column"]
    validate_features(data, features)
    if data[group].isna().any() or data[LABEL_COLUMN].isna().any():
        raise ValueError("Labels and physical recording IDs must be complete")
    expected_labels = {config["negative_label"], config["positive_label"]}
    observed_labels = set(data[LABEL_COLUMN].unique())
    if observed_labels != expected_labels:
        raise ValueError(f"Expected labels {sorted(expected_labels)}; found {sorted(observed_labels)}")
    mixed = data.groupby(group)[LABEL_COLUMN].nunique()
    if (mixed > 1).any():
        raise ValueError(f"Recording IDs have mixed labels: {mixed[mixed > 1].index.tolist()}")
    required_event_columns = ["event_id", "sweep_index", "event_start_index", "event_end_index"]
    missing_event_columns = sorted(set(required_event_columns) - set(data.columns))
    if missing_event_columns:
        raise ValueError(f"Missing event provenance columns: {missing_event_columns}")
    event_duplicate_mask = data.duplicated([group, "sweep_index", "event_start_index", "event_end_index"], keep=False)
    duplicate_event_id_mask = data.duplicated("event_id", keep=False)
    exact_feature_duplicates = data.duplicated(features, keep=False)
    higher = [c for c in config["higher_level_group_candidates"] if c in data.columns]
    higher_level_audit = {}
    for column in higher:
        missing_count = int(data[column].isna().sum())
        recording_to_group = data[[group, column]].drop_duplicates().groupby(group)[column].nunique(dropna=False)
        higher_level_audit[column] = {
            "n_unique": int(data[column].nunique(dropna=True)),
            "n_missing_events": missing_count,
            "recordings_with_multiple_values": recording_to_group[recording_to_group > 1].index.astype(str).tolist(),
        }
    counts = data.groupby([LABEL_COLUMN, group]).size().rename("n_events").reset_index()
    recording_counts = counts.groupby(LABEL_COLUMN)[group].nunique().to_dict()
    required = int(config["minimum_recordings_per_class_for_three_way_split"])
    ready = all(recording_counts.get(label, 0) >= required for label in
                (config["negative_label"], config["positive_label"]))
    return {
        "n_events": int(len(data)), "n_features": len(features),
        "event_counts_by_class": data[LABEL_COLUMN].value_counts().to_dict(),
        "recording_counts_by_class": recording_counts,
        "events_by_recording": counts.to_dict("records"),
        "higher_level_groups_available": higher,
        "higher_level_groups_missing": [c for c in config["higher_level_group_candidates"] if c not in data],
        "accepted_duplicate_boundary_rows": int(event_duplicate_mask.sum()),
        "duplicate_event_id_rows": int(duplicate_event_id_mask.sum()),
        "accepted_exact_feature_duplicate_rows": int(exact_feature_duplicates.sum()),
        "rejected_duplicate_annotation_rows": int(len(duplicate_table)) if duplicate_table is not None else None,
        "real_grouped_evaluation_ready": ready,
        "higher_level_group_audit": higher_level_audit,
        "blocking_reason": None if ready else
            f"Need >= {required} independent recordings per class for train/validation/test; found {recording_counts}",
    }


def make_group_partitions(data, group_column, seed, test_fraction=.2, validation_fraction=.2):
    """Deterministically allocate whole recordings, stratified by recording label."""
    table = data[[group_column, LABEL_COLUMN]].drop_duplicates()
    if table[group_column].duplicated().any():
        raise ValueError("A physical recording maps to multiple labels")
    rng = np.random.default_rng(seed)
    assignment = {}
    for _, rows in table.groupby(LABEL_COLUMN, sort=True):
        groups = rows[group_column].astype(str).to_numpy().copy(); rng.shuffle(groups)
        if len(groups) < 3:
            raise ValueError("At least three independent recordings per class are required")
        n_test = max(1, int(round(len(groups) * test_fraction)))
        n_val = max(1, int(round(len(groups) * validation_fraction)))
        if n_test + n_val >= len(groups): n_test = n_val = 1
        assignment.update({g: "test" for g in groups[:n_test]})
        assignment.update({g: "validation" for g in groups[n_test:n_test+n_val]})
        assignment.update({g: "train" for g in groups[n_test+n_val:]})
    partitions = data[group_column].astype(str).map(assignment)
    assert_group_disjoint(data.assign(partition=partitions), group_column)
    return partitions


def assert_group_disjoint(assignments, group_column="recording_id"):
    cross = assignments.groupby(group_column)["partition"].nunique()
    if (cross > 1).any():
        raise AssertionError(f"Groups cross partitions: {cross[cross > 1].index.tolist()}")
    sets = {p: set(assignments.loc[assignments.partition == p, group_column].astype(str))
            for p in ("train", "validation", "test")}
    if any(sets[a] & sets[b] for a, b in (("train", "validation"), ("train", "test"), ("validation", "test"))):
        raise AssertionError("Group sets overlap")


def score_vector(estimator, X):
    if hasattr(estimator, "predict_proba"):
        return estimator.predict_proba(X)[:, 1]
    return estimator.decision_function(X)


def metrics(y_true, y_pred, y_score):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1]); tn, fp, fn, tp = cm.ravel()
    both = len(np.unique(y_true)) == 2
    return {
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred) if both else np.nan,
        "roc_auc": roc_auc_score(y_true, y_score) if both else np.nan,
        "pr_auc": average_precision_score(y_true, y_score) if both else np.nan,
        "sensitivity_recall": recall_score(y_true, y_pred, zero_division=0),
        "specificity": tn / (tn + fp) if tn + fp else np.nan,
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


def recording_metrics(frame, predictions, scores, group_column):
    rows = []
    for recording, indices in frame.groupby(group_column).groups.items():
        y = frame.loc[indices, "target"].to_numpy()
        pred = predictions[frame.index.get_indexer(indices)]
        score = scores[frame.index.get_indexer(indices)]
        row = metrics(y, pred, score)
        row.update({group_column: recording, "protein": frame.loc[indices, LABEL_COLUMN].iloc[0], "n_events": len(indices)})
        rows.append(row)
    return pd.DataFrame(rows)


def recording_bootstrap_ci(frame, predictions, scores, group_column, n_bootstraps, seed):
    """Stratified cluster bootstrap; sample recordings, then retain all their events."""
    rng = np.random.default_rng(seed); rows = []
    groups_by_class = {
        label: part[group_column].unique()
        for label, part in frame.groupby("target", sort=True)
    }
    by_group = {g: np.flatnonzero(frame[group_column].to_numpy() == g)
                for g in frame[group_column].unique()}
    for _ in range(n_bootstraps):
        sampled = np.concatenate([
            rng.choice(groups, len(groups), replace=True)
            for groups in groups_by_class.values()
        ])
        idx = np.concatenate([by_group[g] for g in sampled])
        rows.append(metrics(frame.target.to_numpy()[idx], predictions[idx], scores[idx]))
    result = []
    for name in ("balanced_accuracy", "roc_auc", "pr_auc", "sensitivity_recall", "specificity", "precision", "f1"):
        values = np.asarray([r[name] for r in rows], float); values = values[np.isfinite(values)]
        result.append({"metric": name, "lower_95": np.quantile(values, .025) if len(values) else np.nan,
                       "upper_95": np.quantile(values, .975) if len(values) else np.nan,
                       "method": "protein-stratified cluster bootstrap over physical recordings",
                       "n_bootstraps": int(n_bootstraps)})
    return pd.DataFrame(result)


def model_spaces(seed, balanced):
    models = {
        "dummy": (DummyClassifier(strategy="prior"), {"model__strategy": ["prior"]}),
        "logistic_regression": (LogisticRegression(max_iter=3000, random_state=seed), {"model__C": [.1, 1., 10.]}),
        "random_forest": (RandomForestClassifier(random_state=seed, n_jobs=1), {"model__n_estimators": [100], "model__max_depth": [None, 6], "model__min_samples_leaf": [1, 5]}),
        "svm": (SVC(probability=True, random_state=seed), {"model__C": [.5, 2.], "model__kernel": ["linear", "rbf"], "model__gamma": ["scale"]}),
    }
    output = {}
    for name, (estimator, grid) in models.items():
        if balanced:
            estimator = BalancedFitClassifier(estimator, random_state=seed)
            grid = {k.replace("model__", "model__estimator__"): v for k, v in grid.items()}
        output[name] = (estimator, grid)
    return output


def build_pipeline(estimator, n_features):
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
        ("scaler", RobustScaler()),
        ("select", SelectKBest(f_classif, k="all")),
        ("pca", "passthrough"),
        ("model", estimator),
    ])


def hyperparameter_search_space(n_features):
    """Human-readable, JSON-safe record of every tuned dimension."""
    return {
        "preprocessing": {
            "imputer": ["median_with_missing_indicators"],
            "scaler": ["RobustScaler"],
            "select__k": ["all", max(3, n_features // 2)],
            "pca": ["passthrough", "PCA(n_components=0.95, svd_solver=full)"],
        },
        "dummy": {"strategy": ["prior"]},
        "logistic_regression": {"C": [.1, 1., 10.]},
        "random_forest": {"n_estimators": [100], "max_depth": [None, 6],
                          "min_samples_leaf": [1, 5]},
        "svm": {"C": [.5, 2.], "kernel": ["linear", "rbf"], "gamma": ["scale"]},
        "A_natural_class_weight": {"models": ["logistic_regression", "random_forest", "svm"],
                                   "class_weight": "balanced"},
        "B_balanced_training": {"method": "random undersampling inside each estimator.fit",
                                "seeds": "config.seeds"},
    }


def synthetic_dataset(config, seed=2026):
    rng = np.random.default_rng(seed); rows = []
    n_rec = int(config["synthetic_recordings_per_class"]); n_events = int(config["synthetic_events_per_recording"])
    for target, protein in enumerate((config["negative_label"], config["positive_label"])):
        for rec in range(n_rec):
            recording_shift = rng.normal(0, .25, len(config["features"]))
            for event in range(n_events + (20 if target == 0 else 0)):
                values = rng.normal(target * .55, 1, len(config["features"])) + recording_shift
                row = dict(zip(config["features"], values))
                row.update({LABEL_COLUMN: protein, "recording_id": f"synthetic_{protein}_{rec}",
                            "event_id": f"synthetic_{protein}_{rec}:{event}", "sweep_index": 0,
                            "event_start_index": event * 10, "event_end_index": event * 10 + 5})
                rows.append(row)
    data = pd.DataFrame(rows)
    data.loc[data.index[::37], config["features"][2]] = np.nan
    return data


def run_experiments(data, config, output):
    group = config["group_column"]; features = config["features"]
    partitions = make_group_partitions(data, group, config["seeds"][0], config["outer_test_fraction"], config["validation_fraction"])
    data = data.copy(); data["partition"] = partitions; data["target"] = (data[LABEL_COLUMN] == config["positive_label"]).astype(int)
    data[[LABEL_COLUMN, group, "event_id", "partition", "target"]].to_csv(output / "split_assignments.csv", index=False)
    train_val = data[data.partition != "test"].copy(); test = data[data.partition == "test"].copy()
    # Validation groups participate only in grouped model selection, never preprocessing outside folds.
    X = train_val[features]; y = train_val.target; groups = train_val[group]
    min_groups = train_val.groupby(LABEL_COLUMN)[group].nunique().min()
    n_splits = min(int(config["inner_cv_splits"]), int(min_groups))
    if n_splits < 2:
        raise ValueError("Grouped tuning requires at least two development recordings per class")
    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=config["seeds"][0])
    cv_splits = list(cv.split(X, y, groups))
    fold_assignments = []
    for fold, (fit_idx, validation_idx) in enumerate(cv_splits):
        for role, indices in (("train", fit_idx), ("validation", validation_idx)):
            for recording_id in sorted(train_val.iloc[indices][group].astype(str).unique()):
                fold_assignments.append({"fold": fold, group: recording_id, "role": role})
    pd.DataFrame(fold_assignments).to_csv(output / "grouped_cv_fold_assignments.csv", index=False)
    scoring = {
        "balanced_accuracy": "balanced_accuracy",
        "roc_auc": "roc_auc",
        "pr_auc": "average_precision",
        "sensitivity_recall": "recall",
        "specificity": make_scorer(recall_score, pos_label=0, zero_division=0),
        "precision": make_scorer(precision_score, zero_division=0),
        "f1": make_scorer(f1_score, zero_division=0),
    }
    cv_rows = []; candidates = []
    for experiment, balanced in (("A_natural", False), ("A_natural_class_weight", False), ("B_balanced_training", True)):
        seeds = config["seeds"] if balanced else [config["seeds"][0]]
        for seed in seeds:
            for name, (estimator, grid) in model_spaces(seed, balanced).items():
                if experiment == "A_natural_class_weight":
                    if name not in ("logistic_regression", "random_forest", "svm"): continue
                    estimator.set_params(class_weight="balanced")
                pipe = build_pipeline(estimator, len(features))
                grid = {**grid, "select__k": ["all", max(3, len(features)//2)], "pca": ["passthrough", PCA(n_components=.95, svd_solver="full")]}
                search = GridSearchCV(pipe, grid, scoring=scoring, cv=cv_splits,
                                      n_jobs=1, refit=config["selection_metric"],
                                      return_train_score=False, error_score="raise")
                search.fit(X, y, groups=groups)
                for split in range(n_splits):
                    row = {"experiment": experiment, "seed": seed, "model": name, "fold": split}
                    for metric_name in scoring:
                        row[metric_name] = search.cv_results_[f"split{split}_test_{metric_name}"][search.best_index_]
                    cv_rows.append(row)
                candidates.append({"experiment": experiment, "seed": seed, "model": name,
                                   "cv_mean": search.best_score_, "best_params": search.best_params_, "estimator": search.best_estimator_})
    cv_frame = pd.DataFrame(cv_rows)
    cv_frame.to_csv(output / "grouped_cv_distribution.csv", index=False)
    metric_columns = list(scoring)
    seed_means = (cv_frame[cv_frame.experiment == "B_balanced_training"]
                  .groupby(["model", "seed"], as_index=False)[metric_columns].mean())
    variance_rows = []
    for model, model_rows in seed_means.groupby("model"):
        for metric_name in metric_columns:
            values = model_rows[metric_name]
            variance_rows.append({"model": model, "metric": metric_name,
                                  "mean_across_seeds": values.mean(),
                                  "std_across_seeds": values.std(ddof=1),
                                  "min_across_seeds": values.min(),
                                  "max_across_seeds": values.max(),
                                  "n_seeds": len(values)})
    pd.DataFrame(variance_rows).to_csv(output / "balanced_seed_variance.csv", index=False)
    serializable = [{k: (v if k != "best_params" else {pk: str(pv) for pk, pv in v.items()})
                     for k, v in c.items() if k != "estimator"} for c in candidates]
    pd.DataFrame(serializable).to_json(output / "candidate_models.json", orient="records", indent=2)
    locked = []
    for experiment in ("A_natural", "A_natural_class_weight", "B_balanced_training"):
        pool = [c for c in candidates if c["experiment"] == experiment]
        winner = sorted(pool, key=lambda c: (-c["cv_mean"], c["model"], c["seed"]))[0]
        locked.append(winner)
    (output / "locked_models.json").write_text(json.dumps([{k: ({pk: str(pv) for pk,pv in v.items()} if k == "best_params" else v)
        for k,v in item.items() if k != "estimator"} for item in locked], indent=2), encoding="utf-8")
    # All experiment definitions are locked above before this single evaluation phase.
    result_rows = []
    for winner in locked:
        pred = winner["estimator"].predict(test[features]); score = score_vector(winner["estimator"], test[features])
        row = metrics(test.target, pred, score); row.update({k: winner[k] for k in ("experiment", "seed", "model")}); result_rows.append(row)
        stem = winner["experiment"]
        recording_metrics(test, pred, score, group).to_csv(output / f"per_recording_{stem}.csv", index=False)
        recording_bootstrap_ci(test, pred, score, group, config["confidence_interval_bootstraps"], winner["seed"]).to_csv(output / f"recording_aware_ci_{stem}.csv", index=False)
        joblib.dump(winner["estimator"], output / f"model_{stem}.joblib")
    pd.DataFrame(result_rows).to_csv(output / "locked_test_metrics.csv", index=False)
    return data, locked


def write_report(output, audit, synthetic, config, real_audit=None):
    status = "SYNTHETIC PIPELINE TEST ONLY" if synthetic else "REAL GROUPED EVALUATION"
    locked_metrics = pd.read_csv(output / "locked_test_metrics.csv")
    cv_frame = pd.read_csv(output / "grouped_cv_distribution.csv")
    cv_summary = cv_frame.groupby(["experiment", "model"])["balanced_accuracy"].agg(["mean", "std"]).reset_index()
    lines = [f"# HSA/F8 supervised classification report — {status}", "", "## 1. Dataset and grouping audit", "",
             f"- Events: {audit['n_events']}; recordings by class: `{audit['recording_counts_by_class']}`.",
             f"- Grouping key: `{config['group_column']}`. Higher-level fields available: `{audit['higher_level_groups_available']}`.",
             "", "## 2. Leakage checks", "", f"- Accepted duplicate-boundary rows: {audit['accepted_duplicate_boundary_rows']}.",
             f"- Exact feature duplicate rows (audit flag, not automatically deleted): {audit['accepted_exact_feature_duplicate_rows']}.",
             f"- Duplicate event-ID rows: {audit['duplicate_event_id_rows']}.",
             f"- Rejected duplicate annotations: {audit['rejected_duplicate_annotation_rows']}.",
             "- Labels, identifiers, paths, source fields, QC fields, indices, sampling-rate, and baseline fields are excluded from X.",
             "- Imputation, robust scaling, SelectKBest, optional PCA, balancing, and model fitting occur inside training folds.",
             "", "## 3. Split strategy", "",
             "- Whole physical recordings are assigned to train/validation/test; automated assertions reject overlap.",
             "- Inner tuning uses shuffled StratifiedGroupKFold on development recordings.",
             "", "## 4. Experiment A results", "", "Natural and class-weight variants are reported separately below (synthetic software test only):"]
    for row in locked_metrics[locked_metrics.experiment.str.startswith("A_")].itertuples():
        lines.append(f"- `{row.experiment}` locked `{row.model}`: balanced accuracy={row.balanced_accuracy:.3f}, ROC AUC={row.roc_auc:.3f}, PR AUC={row.pr_auc:.3f}, recall={row.sensitivity_recall:.3f}, specificity={row.specificity:.3f}, precision={row.precision:.3f}, F1={row.f1:.3f}; confusion matrix TN/FP/FN/TP={row.tn}/{row.fp}/{row.fn}/{row.tp}.")
    lines += ["", "## 5. Experiment B results", "", "HSA/F8 undersampling happens only in estimator.fit within each fold. All configured seeds are retained to quantify sampling variance.",
             "Seed-level mean/std/min/max are in `balanced_seed_variance.csv`.",
             f"- Locked synthetic result: balanced accuracy={locked_metrics.loc[locked_metrics.experiment == 'B_balanced_training', 'balanced_accuracy'].iloc[0]:.3f}; this is not biological evidence.",
             "", "## 6. Hyperparameter search space and selection rule", "",
             "- Models: Dummy, Logistic Regression, Random Forest, SVM; full grids are recorded in `hyperparameter_search_space.json`, and selected candidates in `candidate_models.json`.",
             f"- Selection: highest mean grouped-CV `{config['selection_metric']}`, deterministic model/seed tie-break; locks written before test access.",
             "", "## 7. Grouped CV distribution", "", "Fold-level values for all requested scalar metrics are in `grouped_cv_distribution.csv`."]
    for row in cv_summary.itertuples():
        lines.append(f"- `{row.experiment}` / `{row.model}`: balanced accuracy mean={row.mean:.3f}, SD={row.std:.3f} across recorded fold/seed results.")
    lines += [
             "", "## 8. Locked test metrics", "", "One locked test evaluation is in `locked_test_metrics.csv`.",
             "", "## 9. Per-recording behavior", "", "Per-recording tables and cluster-bootstrap confidence intervals are saved per experiment.",
             "", "## 10. Limitations and mixture-test readiness", ""]
    if synthetic:
        lines += ["- All model scores in this run are synthetic software-validation results, not biological generalization performance.",
                  f"- The real-data audit remains blocked with recording counts `{real_audit['recording_counts_by_class'] if real_audit else 'see real_dataset_audit.json'}`.",
                  "- Real grouped evaluation is blocked: HSA has one physical recording, so an independent HSA train/test split is impossible.",
                  "- Collect at least 2 additional independent HSA recordings to reach the mechanical minimum of 3 per class; 5–10+ recordings per class across pores/days/batches is recommended for useful uncertainty estimates.",
                  "- Add pore/day/batch/session metadata and keep those dependencies disjoint before a mixture test.",
                  "- The pipeline is software-ready, but biological mixture-test readiness is blocked pending independent recordings and a locked pure-protein evaluation."]
    else:
        lines += ["- Mixture testing should begin only after the locked pure-protein grouped evaluation meets preregistered criteria."]
    (output / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_blocked_report(output, audit, config):
    lines = ["# HSA/F8 real grouped evaluation — BLOCKED", "",
             "## 1. Dataset and grouping audit", "",
             f"- Events by class: `{audit['event_counts_by_class']}`.",
             f"- Physical recordings by class: `{audit['recording_counts_by_class']}`.",
             f"- Higher-level groups available: `{audit['higher_level_groups_available']}`; missing: `{audit['higher_level_groups_missing']}`.",
             "", "## 2. Leakage checks", "",
             f"- Accepted duplicate-boundary rows: {audit['accepted_duplicate_boundary_rows']}; exact-feature duplicate rows: {audit['accepted_exact_feature_duplicate_rows']}.",
             f"- Duplicate event-ID rows: {audit['duplicate_event_id_rows']}.",
             f"- Rejected duplicate annotations: {audit['rejected_duplicate_annotation_rows']}.",
             "- Explicit features exclude labels, identifiers, paths, source/audit fields, QC fields, indices, sampling rate, and baseline fields.",
             "", "## 3. Split strategy", "",
             "- No real split was made and no model was fit. Splitting HSA events from one ABF would leak recording identity.",
             "", "## 4. Experiment A results", "", "- Not run on real data: blocked before fitting.",
             "", "## 5. Experiment B results", "", "- Not run on real data: undersampling cannot create independent recordings.",
             "", "## 6. Hyperparameter search space and selection rule", "",
             "- Configured models: Dummy, Logistic Regression, Random Forest, SVM; preprocessing includes SelectKBest and optional PCA.",
             f"- Selection metric is `{config['selection_metric']}` in grouped CV; lock must precede a single test access.",
             "", "## 7. Grouped CV distribution", "", "- Unavailable for real data.",
             "", "## 8. Locked test metrics", "", "- None; real test data were neither created nor accessed.",
             "", "## 9. Per-recording behavior", "", "- HSA=1 and F8=3 physical recordings; generalization cannot be estimated.",
             "", "## 10. Limitations and mixture-test readiness", "",
             "- Collect at least 2 additional independent HSA recordings for the mechanical minimum of 3 per class.",
             "- Prefer 5–10+ recordings per class across independent pores, days, batches, and sessions for useful uncertainty estimates.",
             "- Real evaluation and mixture testing remain blocked until those recordings are available."]
    (output / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv=None):
    parser = argparse.ArgumentParser(); parser.add_argument("--config", default=str(Path(__file__).parent / "config/supervised_experiment.json")); parser.add_argument("--synthetic", action="store_true")
    args = parser.parse_args(argv); config = load_config(args.config)
    real = pd.read_csv(config["feature_table"], dtype={config["group_column"]: str}); duplicates = pd.read_csv(config["duplicate_table"])
    real_audit = audit_dataset(real, config, duplicates)
    run_id = datetime.now().strftime("nanopore_supervised_%Y%m%d_%H%M%S") + ("_synthetic" if args.synthetic else "")
    output = Path(config["output_root"]) / run_id; output.mkdir(parents=True, exist_ok=False)
    (output / "real_dataset_audit.json").write_text(json.dumps(real_audit, indent=2), encoding="utf-8")
    metadata_columns = [column for column in real.columns if column not in config["features"]]
    schema = {"shape": [int(real.shape[0]), int(real.shape[1])],
              "columns": [{"name": column, "dtype": str(real[column].dtype),
                           "missing": int(real[column].isna().sum()),
                           "role": "feature" if column in config["features"] else "metadata_or_audit"}
                          for column in real.columns]}
    (output / "dataset_schema.json").write_text(json.dumps(schema, indent=2), encoding="utf-8")
    (output / "feature_list.json").write_text(json.dumps(config["features"], indent=2), encoding="utf-8")
    (output / "metadata_columns.json").write_text(json.dumps(metadata_columns, indent=2), encoding="utf-8")
    (output / "resolved_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    (output / "hyperparameter_search_space.json").write_text(
        json.dumps(hyperparameter_search_space(len(config["features"])), indent=2), encoding="utf-8")
    if not real_audit["real_grouped_evaluation_ready"] and not args.synthetic:
        manifest = {"run_type": "supervised_classification", "evaluation_scope": "real_blocked",
                    "created_at": datetime.now().astimezone().isoformat(), "python_version": platform.python_version(),
                    "config": config, "feature_table_sha256": sha256(config["feature_table"]),
                    "extraction_manifest_sha256": sha256(config["extraction_manifest"]),
                    "seeds": config["seeds"], "test_access_count": 0, "real_dataset_blocked": True}
        (output / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        write_blocked_report(output, real_audit, config)
        raise SystemExit(f"BLOCKED: {real_audit['blocking_reason']}; audit saved to {output}")
    data = synthetic_dataset(config) if args.synthetic else real
    audit = audit_dataset(data, config, duplicates if not args.synthetic else None)
    assignments, locked = run_experiments(data, config, output)
    split_summary = assignments.groupby(["partition", LABEL_COLUMN])[config["group_column"]].agg(n_events="size", n_recordings="nunique").reset_index()
    split_summary.to_csv(output / "split_summary.csv", index=False)
    manifest = {"run_type": "supervised_classification", "evaluation_scope": "synthetic_only" if args.synthetic else "real",
                "created_at": datetime.now().astimezone().isoformat(), "python_version": platform.python_version(),
                "config": config, "feature_table_sha256": sha256(config["feature_table"]),
                "extraction_manifest_sha256": sha256(config["extraction_manifest"]), "seeds": config["seeds"],
                "test_access_count": 1, "real_dataset_blocked": not real_audit["real_grouped_evaluation_ready"]}
    (output / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    write_report(output, audit, args.synthetic, config, real_audit=real_audit)
    print(output)


if __name__ == "__main__":
    main()
