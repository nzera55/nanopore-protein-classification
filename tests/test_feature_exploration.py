import numpy as np
import pandas as pd
import pytest

from feature_exploration import (
    ALL_FEATURES,
    SEGMENT_FEATURES,
    load_and_validate,
    prepare_embedding_matrix,
)


def valid_table():
    data = pd.DataFrame({
        "protein": ["HSA", "F8", "F8"],
        "recording_id": ["h1", "f1", "f2"],
        "event_id": ["h:1", "f:1", "f:2"],
    })
    for index, feature in enumerate(ALL_FEATURES):
        data[feature] = [index + 1.0, index + 2.0, index + 3.0]
    return data


def test_compact_analysis_contains_exactly_ten_segment_means():
    assert SEGMENT_FEATURES == [
        f"segment_{number:02d}_blockage_mean" for number in range(1, 11)
    ]
    assert len(ALL_FEATURES) == 23


def test_validation_rejects_legacy_segment_feature(tmp_path):
    data = valid_table()
    data["segment_01_slope"] = 1.0
    path = tmp_path / "features.csv"
    data.to_csv(path, index=False)
    with pytest.raises(ValueError, match="non-compact"):
        load_and_validate(path)


def test_embedding_preprocessing_preserves_rows_and_handles_missing():
    numeric = valid_table()[ALL_FEATURES].copy()
    numeric.loc[0, "fwhm_width"] = np.nan
    matrix, colors, parameters = prepare_embedding_matrix(numeric)
    assert matrix.shape == numeric.shape
    assert colors.shape == numeric.shape
    assert np.isfinite(matrix).all()
    assert parameters.loc[
        parameters.feature.eq("fwhm_width"), "n_missing"
    ].item() == 1
