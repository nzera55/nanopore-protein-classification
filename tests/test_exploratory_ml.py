import numpy as np
import pandas as pd
import json
from pathlib import Path

from exploratory_ml import (SCOPE_E1, build_pipeline, confusion_arrays,
                            dependency_status, exploratory_split,
                            feature_sets, leave_one_f8_recording_out, model_spaces,
                            save_confusion_artifacts)


def frame():
    return pd.DataFrame({"protein": ["HSA"] * 20 + ["F8"] * 12,
                         "recording_id": ["h1"] * 20 + ["f1"] * 6 + ["f2"] * 6})


def test_e1_is_explicitly_labeled_and_stratified_but_leaky():
    data = frame(); train, test = exploratory_split(data, 11, .25)
    assert SCOPE_E1 == "exploratory_with_recording_leakage"
    assert set(data.iloc[train].protein) == {"HSA", "F8"}
    assert set(data.iloc[test].protein) == {"HSA", "F8"}
    assert "h1" in set(data.iloc[train].recording_id) & set(data.iloc[test].recording_id)


def test_e2_holds_out_only_requested_f8_recording():
    data = frame(); train, test = leave_one_f8_recording_out(data, "f1")
    assert set(data.iloc[test].recording_id) == {"f1"}
    assert set(data.iloc[test].protein) == {"F8"}
    assert "f1" not in set(data.iloc[train].recording_id)
    assert "h1" in set(data.iloc[train].recording_id)


def test_balancing_is_wrapped_inside_model_fit():
    spaces = model_spaces(42, balanced=True)
    assert set(spaces) == {"knn", "random_forest", "xgboost"}
    for estimator, _ in spaces.values():
        assert estimator.__class__.__name__ == "BalancedFitClassifier"


def test_knn_pipeline_has_imputation_and_scaling_before_model():
    estimator, _ = model_spaces(11, balanced=False)["knn"]
    pipeline = build_pipeline(estimator)
    assert list(pipeline.named_steps) == ["imputer", "scaler", "model"]


def test_xgboost_dependency_records_version_and_path():
    status = dependency_status()
    assert status["xgboost_available"]
    assert status["xgboost_version"] == "3.4.1"
    assert status["xgboost_path"].endswith("xgboost/__init__.py")


def test_confusion_outputs_are_fixed_two_by_two(tmp_path):
    counts, normalized = save_confusion_artifacts(
        tmp_path, "one_class", np.ones(3, dtype=int), np.array([1, 0, 1]), "E2 test")
    assert counts.shape == normalized.shape == (2, 2)
    assert np.isnan(normalized[0]).all()
    assert (tmp_path / "one_class_counts.csv").exists()
    assert (tmp_path / "one_class_normalized.csv").exists()
    assert (tmp_path / "one_class.png").exists()


def test_feature_sets_exclude_segmentation_qc_and_have_expected_membership():
    compact = ["dwell_time_ms", "blockage_mean"]
    segments = [f"segment_{i}" for i in range(10)]
    config = {"feature_sets": {"compact": compact, "core_segments": compact + segments}}
    result = feature_sets(config, compact + segments + ["segmentation_qc_pass"])
    assert result["compact"] == compact
    assert "segmentation_qc_pass" not in result["core_segments"]


def test_feature_sets_reject_qc_predictor():
    segment = ["segment_01_blockage_mean", "segmentation_qc_pass"]
    config = {"feature_sets": {"core_segments": segment}}
    import pytest
    with pytest.raises(ValueError, match="forbidden"):
        feature_sets(config, segment)


def test_v9_segments_only_feature_set_contains_exact_compact_schema():
    path = Path(__file__).resolve().parents[1] / "config" / "exploratory_ml_feature_sets_v9.json"
    configured = json.loads(path.read_text())["feature_sets"]["set_e_segments_only"]
    assert configured == [f"segment_{index:02d}_blockage_mean" for index in range(1, 11)]
