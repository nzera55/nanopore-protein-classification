import numpy as np
import pandas as pd
import unittest
from sklearn.linear_model import LogisticRegression

from machine_learning import (BalancedFitClassifier, assert_group_disjoint,
                              audit_dataset, build_pipeline, make_group_partitions, metrics,
                              model_spaces,
                              recording_bootstrap_ci, synthetic_dataset)


def config():
    return {"features": ["dwell_time_ms", "blockage_mean", "auc"], "group_column": "recording_id",
            "higher_level_group_candidates": ["pore_id", "batch_id"], "negative_label": "HSA", "positive_label": "F8",
            "minimum_recordings_per_class_for_three_way_split": 3,
            "synthetic_recordings_per_class": 6, "synthetic_events_per_recording": 12}


class SupervisedLearningTests(unittest.TestCase):
    def test_real_constraint_blocks_one_hsa_recording(self):
        data = synthetic_dataset(config())
        data = data[~((data.protein == "HSA") & (data.recording_id != "synthetic_HSA_0"))]
        audit = audit_dataset(data, config())
        self.assertFalse(audit["real_grouped_evaluation_ready"])
        self.assertEqual(audit["recording_counts_by_class"]["HSA"], 1)

    def test_group_partitions_are_disjoint_and_have_both_classes(self):
        data = synthetic_dataset(config())
        data["partition"] = make_group_partitions(data, "recording_id", 11)
        assert_group_disjoint(data)
        for partition in ("train", "validation", "test"):
            self.assertEqual(set(data.loc[data.partition == partition, "protein"]), {"HSA", "F8"})

    def test_overlap_assertion_fails_closed(self):
        frame = pd.DataFrame({"recording_id": ["r1", "r1"], "partition": ["train", "test"]})
        with self.assertRaises(AssertionError): assert_group_disjoint(frame)

    def test_balancing_occurs_only_during_fit(self):
        X = np.arange(24).reshape(12, 2); y = np.array([0] * 9 + [1] * 3)
        model = BalancedFitClassifier(LogisticRegression(), random_state=2).fit(X, y)
        self.assertEqual(model.fit_class_counts_, {"0": 3, "1": 3})
        self.assertEqual(len(model.predict(X)), 12)
        self.assertEqual(BalancedFitClassifier.__module__, "machine_learning")

    def test_pipeline_order_and_supported_models(self):
        spaces = model_spaces(seed=11, balanced=False)
        self.assertEqual(set(spaces), {"dummy", "logistic_regression", "random_forest", "svm"})
        pipeline = build_pipeline(spaces["logistic_regression"][0], n_features=3)
        self.assertEqual(list(pipeline.named_steps), ["imputer", "scaler", "select", "pca", "model"])
        self.assertEqual(pipeline.named_steps["pca"], "passthrough")

    def test_forbidden_feature_is_rejected(self):
        data = synthetic_dataset(config())
        bad = config(); bad["features"] = ["recording_id"]
        with self.assertRaises(ValueError): audit_dataset(data, bad)

    def test_duplicate_event_ids_are_audited(self):
        data = synthetic_dataset(config())
        data.loc[data.index[1], "event_id"] = data.loc[data.index[0], "event_id"]
        audit = audit_dataset(data, config())
        self.assertEqual(audit["duplicate_event_id_rows"], 2)

    def test_all_required_metrics_and_recording_bootstrap(self):
        data = synthetic_dataset(config())
        data["target"] = (data.protein == "F8").astype(int)
        prediction = data.target.to_numpy()
        scores = prediction.astype(float)
        result = metrics(data.target, prediction, scores)
        self.assertTrue({"balanced_accuracy", "roc_auc", "pr_auc", "sensitivity_recall",
                         "specificity", "precision", "f1", "tn", "fp", "fn", "tp"} <= set(result))
        ci = recording_bootstrap_ci(data, prediction, scores, "recording_id", 20, 4)
        self.assertEqual(set(ci.metric), {"balanced_accuracy", "roc_auc", "pr_auc",
                                         "sensitivity_recall", "specificity", "precision", "f1"})


if __name__ == "__main__":
    unittest.main()
