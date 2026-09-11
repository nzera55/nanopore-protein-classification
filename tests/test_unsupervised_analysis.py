import json

import numpy as np
import pandas as pd
import pytest

from project_config import require_new_output_directory
from unsupervised_analysis import (run_pca, run_umap,
                                   FEATURE_SETS, prepare,
                                   assert_feature_contract, recording_aware_sample,
                                   domain_audit, normalize_qc_flag,
                                   save_feature_color_views,
                                   validate_extraction_provenance)


def test_pca_rows_and_variance():
    rng = np.random.default_rng(4)
    matrix = rng.normal(size=(20, 5))
    metadata = pd.DataFrame({"protein": ["HSA"] * 10 + ["F8"] * 10,
                             "recording_id": ["a"] * 10 + ["b"] * 10,
                             "event_id": range(20)})
    _, coordinates, variance, _, _, _ = run_pca(matrix, metadata, list("abcde"))
    assert len(coordinates) == len(metadata)
    assert variance.explained_variance_ratio.between(0, 1).all()


def test_umap_is_reproducible_with_same_seed():
    rng = np.random.default_rng(5)
    matrix = rng.normal(size=(20, 4))
    metadata = pd.DataFrame({"protein": ["HSA"] * 10 + ["F8"] * 10,
                             "recording_id": ["a"] * 10 + ["b"] * 10,
                             "event_id": range(20)})
    _, first = run_umap(matrix, metadata, 42, 5, .1)
    _, second = run_umap(matrix, metadata, 42, 5, .1)
    np.testing.assert_allclose(first[["UMAP1", "UMAP2"]], second[["UMAP1", "UMAP2"]])


def test_existing_output_is_rejected(tmp_path):
    path = tmp_path / "existing"; path.mkdir()
    with pytest.raises(FileExistsError): require_new_output_directory(path)


def test_manifest_and_features_must_share_run(tmp_path):
    left = tmp_path / "left"; right = tmp_path / "right"
    left.mkdir(); right.mkdir()
    manifest = left / "run_manifest.json"
    manifest.write_text(json.dumps({"run_type": "event_extraction"}))
    features = right / "abf_event_features.csv"; features.write_text("x\n1\n")
    with pytest.raises(ValueError, match="same extraction run"):
        validate_extraction_provenance(left, manifest, features)


def test_all_calculated_feature_selection_has_no_metadata():
    expected = {
        "dwell_time_ms", "blockage_percent", "blockage_mean", "blockage_sd",
        "auc", "skewness", "kurtosis", "fwhm_width", "fwhm_drop",
        "rise_time", "fall_time", "left_slope", "right_slope",
    }
    assert set(FEATURE_SETS["all_calculated_features"]) == expected
    assert not expected & {"protein", "recording_id", "event_id"}


def test_prepare_uses_median_robust_scaler_and_preserves_rows():
    data = pd.DataFrame({"a": [1., np.nan, 9.], "b": [2., 4., 6.]})
    numeric, matrix, parameters = prepare(data, ["a", "b"])
    assert len(numeric) == len(matrix) == len(data)
    assert np.isfinite(matrix).all()
    assert parameters.loc[parameters.feature == "a", "median"].item() == 5.
    assert set(parameters) >= {"robust_center", "robust_scale_iqr"}


def test_pca_3d_schema_and_color_by_feature_output(tmp_path):
    rng = np.random.default_rng(9)
    matrix = rng.normal(size=(20, 4))
    metadata = pd.DataFrame({"protein": ["HSA"] * 10 + ["F8"] * 10,
                             "recording_id": ["h1"] * 10 + ["f1"] * 10,
                             "event_id": range(20)})
    _, scores, variance, _, _, _ = run_pca(matrix, metadata, list("abcd"))
    assert {"PC1", "PC2", "PC3"} <= set(scores)
    colors = pd.DataFrame(matrix, columns=list("abcd"))
    save_feature_color_views(scores, colors, list("abcd"), tmp_path, "PCA_test",
                             ("PC1", "PC2", "PC3"),
                             {f"PC{i+1}": value for i, value in enumerate(variance.explained_variance_ratio)})
    outputs = list((tmp_path / "color_by_feature").glob("*.png"))
    assert len(outputs) == 8
    assert all(path.stat().st_size > 0 for path in outputs)


def test_qc_strings_are_parsed_as_booleans():
    parsed = normalize_qc_flag(pd.Series(["True", "False", 1, 0, None]), "qc")
    assert parsed.tolist() == [True, False, True, False, False]


def test_domain_audit_flags_invalid_time_and_slope_values():
    data = pd.DataFrame({"dwell_time_ms": [1., 0.], "fwhm_width": [2., -1.],
                         "left_slope": [1., -1.], "right_slope": [-1., 1.]})
    audit = domain_audit(data, data.columns)
    counts = audit.set_index("check").n_flagged.to_dict()
    assert counts["dwell_time_ms_nonpositive"] == 1
    assert counts["fwhm_width_exceeds_dwell_time"] == 1
    assert counts["left_slope_nonpositive"] == 1
    assert counts["right_slope_nonnegative"] == 1


def test_feature_contract_rejects_labels_paths_and_qc():
    data = pd.DataFrame({"blockage_mean": [1.], "protein": ["HSA"],
                         "abf_path": ["x"], "shape_qc_pass": [True]})
    assert_feature_contract(data, ["blockage_mean"])
    for leaked in ("protein", "abf_path", "shape_qc_pass"):
        with pytest.raises(ValueError, match="Forbidden"):
            assert_feature_contract(data, [leaked])


def test_balanced_sample_preserves_recording_strata_and_equal_proteins():
    data = pd.DataFrame({"protein": ["HSA"] * 8 + ["F8"] * 4,
                         "recording_id": ["h1"] * 8 + ["f1"] * 2 + ["f2"] * 2,
                         "event_id": range(12)})
    sampled = recording_aware_sample(data, 42, balanced=True)
    assert sampled.protein.value_counts().nunique() == 1
    assert set(sampled[sampled.protein == "F8"].recording_id) == {"f1", "f2"}
