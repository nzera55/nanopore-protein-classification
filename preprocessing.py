"""Reusable preprocessing for exploratory nanopore embeddings.

Protein labels and provenance are retained in metadata and are never included
in the numerical feature matrix.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler, StandardScaler


FEATURE_SETS = {
    "all_features": [
        "dwell_time_ms", "blockage_percent", "blockage_mean", "blockage_sd",
        "auc", "skewness", "kurtosis", "fwhm_width", "fwhm_drop",
        "rise_time", "fall_time", "right_slope", "left_slope",
    ],
    "waveform_only": [
        "blockage_mean", "blockage_sd", "auc", "skewness", "kurtosis",
        "fwhm_width", "fwhm_drop", "rise_time", "fall_time",
        "right_slope", "left_slope",
    ],
    "shape_only": [
        "fwhm_width", "rise_time", "fall_time", "right_slope", "left_slope",
        "skewness", "kurtosis",
    ],
}

METADATA_COLUMNS = {
    "protein", "recording_id", "event_id", "json_path", "abf_path",
    "source_json", "source_abf", "sweep_index", "sweep_number",
}
POSITIVE_SKEW_CANDIDATES = {
    "blockage_mean", "blockage_sd", "auc", "fwhm_width",
    "fwhm_drop", "rise_time", "fall_time",
}


@dataclass
class PreparedFeatureMatrix:
    matrix: np.ndarray
    metadata: pd.DataFrame
    feature_names: list[str]
    transformed_data: pd.DataFrame
    transformations: dict
    imputation_parameters: pd.DataFrame
    scaling_parameters: pd.DataFrame
    missingness: pd.DataFrame


def select_feature_set(name):
    """Return a copy of a named, definition-safe feature list."""
    if name not in FEATURE_SETS:
        raise ValueError(f"Unknown feature set {name!r}; choose {sorted(FEATURE_SETS)}")
    features = list(FEATURE_SETS[name])
    _validate_no_definition_duplicates(features)
    return features


def _validate_no_definition_duplicates(features):
    forbidden_pairs = [
        {"dwell_time_ms", "log_dwell_time"},
        {"right_slope", "right_slope_magnitude"},
        {"left_slope", "left_slope_magnitude"},
    ]
    for pair in forbidden_pairs:
        if pair.issubset(features):
            raise ValueError(f"Definitionally duplicate features selected: {sorted(pair)}")


def validate_feature_columns(data, features):
    """Reject missing, duplicated, metadata, target, and nonnumeric columns."""
    if len(features) != len(set(features)):
        raise ValueError("Feature list contains duplicate columns")
    _validate_no_definition_duplicates(features)
    leaked = sorted(set(features) & METADATA_COLUMNS)
    if leaked:
        raise ValueError(f"Metadata/target columns cannot be features: {leaked}")
    missing = [feature for feature in features if feature not in data.columns]
    # log dwell is intentionally derived from dwell_time_ms.
    if "log_dwell_time" in missing and "dwell_time_ms" in data.columns:
        missing.remove("log_dwell_time")
    if missing:
        raise ValueError(f"Feature columns missing from input: {missing}")
    nonnumeric = [f for f in features if f in data and not pd.api.types.is_numeric_dtype(data[f])]
    if nonnumeric:
        raise TypeError(f"Feature columns must be numeric: {nonnumeric}")


def replace_infinite_values(frame):
    """Replace positive and negative infinity with missing values."""
    return frame.replace([np.inf, -np.inf], np.nan)


def transform_skewed_features(frame, skew_threshold=1.0):
    """Apply log1p only to nonnegative candidate features with right skew."""
    transformed = frame.copy()
    decisions = {}
    for feature in frame.columns:
        series = frame[feature].dropna()
        apply_log = (
            feature in POSITIVE_SKEW_CANDIDATES
            and not series.empty
            and bool((series >= 0).all())
            and float(series.skew()) > skew_threshold
        )
        if apply_log:
            transformed[feature] = np.log1p(transformed[feature])
            decisions[feature] = "log1p"
        else:
            decisions[feature] = "none"
    return transformed, decisions


def handle_missing_values(frame):
    """Median-impute features and return fitted values."""
    all_missing = frame.columns[frame.isna().all()].tolist()
    if all_missing:
        raise ValueError(f"Cannot median-impute entirely missing features: {all_missing}")
    imputer = SimpleImputer(strategy="median")
    values = imputer.fit_transform(frame)
    parameters = pd.DataFrame({"feature": frame.columns, "median": imputer.statistics_})
    return values, imputer, parameters


def scale_features(values, feature_names, scaler="standard"):
    """Scale an imputed matrix and return fitted parameters."""
    if scaler == "robust":
        fitted = RobustScaler().fit(values)
        parameters = pd.DataFrame({
            "feature": feature_names, "center": fitted.center_, "scale": fitted.scale_,
        })
    elif scaler == "standard":
        fitted = StandardScaler().fit(values)
        parameters = pd.DataFrame({
            "feature": feature_names, "center": fitted.mean_, "scale": fitted.scale_,
        })
    else:
        raise ValueError("scaler must be 'robust' or 'standard'")
    return fitted.transform(values), fitted, parameters


def summarize_missingness(data, features):
    """Report missingness by feature, protein, and recording."""
    rows = []
    group_columns = [c for c in ("protein", "recording_id") if c in data]
    for keys, group in data.groupby(group_columns, dropna=False) if group_columns else [((), data)]:
        keys = keys if isinstance(keys, tuple) else (keys,)
        identity = dict(zip(group_columns, keys))
        for feature in features:
            source = "dwell_time_ms" if feature == "log_dwell_time" else feature
            count = int(group[source].replace([np.inf, -np.inf], np.nan).isna().sum())
            rows.append({**identity, "feature": feature, "n_rows": len(group),
                         "n_missing": count, "missing_fraction": count / len(group)})
    return pd.DataFrame(rows)


def prepare_feature_matrix(data, feature_set="all_features", scaler="standard",
                           transform_skewed=True, metadata_columns=None):
    """Prepare an exploratory full-dataset matrix plus aligned provenance."""
    features = select_feature_set(feature_set)
    validate_feature_columns(data, features)
    working = pd.DataFrame(index=data.index)
    for feature in features:
        if feature == "log_dwell_time":
            dwell = pd.to_numeric(data["dwell_time_ms"], errors="coerce")
            working[feature] = np.where(dwell > 0, np.log10(dwell), np.nan)
        else:
            working[feature] = data[feature]
    working = replace_infinite_values(working)
    missingness = summarize_missingness(data, features)
    if transform_skewed:
        working, transformations = transform_skewed_features(working)
    else:
        transformations = {feature: "none" for feature in features}
    imputed, _, imputation_parameters = handle_missing_values(working)
    matrix, _, scaling_parameters = scale_features(imputed, features, scaler)
    requested_metadata = metadata_columns or sorted(METADATA_COLUMNS & set(data.columns))
    required = [c for c in ("protein", "recording_id", "event_id") if c in data]
    metadata_names = list(dict.fromkeys(required + list(requested_metadata)))
    metadata = data.loc[:, metadata_names].reset_index(drop=True)
    return PreparedFeatureMatrix(
        matrix=matrix, metadata=metadata, feature_names=features,
        transformed_data=working.reset_index(drop=True), transformations=transformations,
        imputation_parameters=imputation_parameters,
        scaling_parameters=scaling_parameters, missingness=missingness,
    )


def high_correlation_pairs(frame, threshold=0.90):
    """Return absolute Spearman correlations above threshold."""
    correlation = frame.corr(method="spearman").abs()
    rows = []
    for i, left in enumerate(correlation.columns):
        for right in correlation.columns[i + 1:]:
            value = correlation.loc[left, right]
            if pd.notna(value) and value >= threshold:
                rows.append({"feature_1": left, "feature_2": right,
                             "absolute_spearman": float(value)})
    return pd.DataFrame(rows).sort_values("absolute_spearman", ascending=False) if rows else pd.DataFrame(
        columns=["feature_1", "feature_2", "absolute_spearman"])
