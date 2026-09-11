import numpy as np
import pandas as pd
import pytest

from preprocessing import (FEATURE_SETS, handle_missing_values,
                           prepare_feature_matrix, select_feature_set,
                           validate_feature_columns)


def feature_data(n=12):
    x = np.arange(1, n + 1, dtype=float)
    return pd.DataFrame({
        "protein": ["HSA"] * (n // 2) + ["F8"] * (n - n // 2),
        "recording_id": ["h1"] * (n // 2) + ["f1"] * (n - n // 2),
        "event_id": [f"e{i}" for i in range(n)], "dwell_time_ms": x,
        "blockage_percent": x + 1, "blockage_mean": x + 2,
        "blockage_sd": x / 5, "auc": x * 2, "skewness": x / 10,
        "kurtosis": x / 20, "fwhm_width": x / 3, "fwhm_drop": x / 4,
        "rise_time": x / 6, "fall_time": x / 7,
        "right_slope": -x, "left_slope": x,
    })


def test_feature_sets_are_safe_and_expected():
    assert set(select_feature_set("shape_only")) == {
        "fwhm_width", "rise_time", "fall_time", "right_slope", "left_slope",
        "skewness", "kurtosis"}
    assert "blockage_percent" not in select_feature_set("waveform_only")
    assert "dwell_time_ms" in select_feature_set("all_features")
    assert "log_dwell_time" not in select_feature_set("all_features")
    for features in FEATURE_SETS.values():
        assert not {"dwell_time_ms", "log_dwell_time"}.issubset(features)


def test_metadata_and_target_cannot_leak():
    with pytest.raises(ValueError, match="Metadata/target"):
        validate_feature_columns(feature_data(), ["protein", "auc"])


def test_median_imputation_and_scaling_shape():
    frame = pd.DataFrame({"a": [1., np.nan, 9.], "b": [2., 4., 6.]})
    values, _, parameters = handle_missing_values(frame)
    assert values[1, 0] == 5.
    assert parameters.loc[parameters.feature == "a", "median"].item() == 5.
    data = feature_data(); data.loc[0, "auc"] = np.nan
    prepared = prepare_feature_matrix(data)
    assert prepared.matrix.shape == (len(data), len(prepared.feature_names))
    assert not np.isnan(prepared.matrix).any()
    assert set(prepared.metadata) >= {"protein", "recording_id", "event_id"}


def test_default_scaler_is_standard():
    data = feature_data()
    prepared = prepare_feature_matrix(data)
    feature = prepared.feature_names.index("blockage_percent")
    assert np.isclose(prepared.matrix[:, feature].mean(), 0.0)
    assert np.isclose(prepared.matrix[:, feature].std(ddof=0), 1.0)
