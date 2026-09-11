import numpy as np
import pandas as pd

from feature_exploration import ALL_FEATURES
from feature_set_exploration import FEATURE_SETS, feature_set_coverage


def test_feature_sets_match_predeclared_design():
    assert [len(FEATURE_SETS[name]) for name in FEATURE_SETS] == [4, 5, 9, 15, 10]
    assert len(set(FEATURE_SETS["set_d_core_segments"])) == 15
    assert FEATURE_SETS["set_e_segments_only"] == [
        f"segment_{index:02d}_blockage_mean" for index in range(1, 11)
    ]
    assert set().union(*map(set, FEATURE_SETS.values())) <= set(ALL_FEATURES)


def test_coverage_reports_complete_rows_by_protein():
    data = pd.DataFrame({"protein": ["HSA", "F8"]})
    numeric = pd.DataFrame({feature: [1.0, 2.0] for feature in ALL_FEATURES})
    numeric.loc[1, "segment_01_blockage_mean"] = np.nan
    coverage = feature_set_coverage(data, numeric)
    core_f8 = coverage.query("feature_set == 'set_b_core' and protein == 'F8'").iloc[0]
    segment_f8 = coverage.query("feature_set == 'set_d_core_segments' and protein == 'F8'").iloc[0]
    segments_only_f8 = coverage.query(
        "feature_set == 'set_e_segments_only' and protein == 'F8'"
    ).iloc[0]
    assert core_f8.n_complete_events == 1
    assert segment_f8.n_complete_events == 0
    assert segment_f8.n_events_requiring_imputation == 1
    assert segments_only_f8.n_complete_events == 0
    assert segments_only_f8.n_events_requiring_imputation == 1
