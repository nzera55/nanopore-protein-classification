import unittest

import numpy as np

import feature_engineering as fe


class FeatureEngineeringTests(unittest.TestCase):
    def setUp(self):
        fe.configure_feature_engineering(
            shape_smoothing_window_ms=0.05,
            minimum_shape_peak_snr=3.0,
        )

    def test_edge_padding_preserves_constant_trace(self):
        values = np.full(9, 7.0)
        result = fe.smooth_trace(values, window_points=5)
        np.testing.assert_allclose(result, values)

    def test_triangular_event_has_complete_shape_features(self):
        time_s = np.arange(41, dtype=float) * 10e-6
        trace = np.r_[
            np.zeros(10),
            np.arange(1, 11),
            np.arange(9, 0, -1),
            np.zeros(12),
        ].astype(float)
        result = fe.extract_shape_features(
            time_s,
            trace,
            baseline_noise_pct=0.2,
        )

        self.assertTrue(result["shape_qc_pass"])
        self.assertEqual(result["peak_count"], 1)
        self.assertGreater(result["peak_snr"], 3.0)
        self.assertTrue(np.isfinite(result["fwhm_width"]))
        self.assertGreater(result["left_slope"], 0)
        self.assertLess(result["right_slope"], 0)

    def test_low_snr_event_keeps_diagnostics_but_not_shape(self):
        time_s = np.arange(41, dtype=float) * 10e-6
        trace = np.r_[
            np.zeros(10),
            np.arange(1, 11),
            np.arange(9, 0, -1),
            np.zeros(12),
        ].astype(float)
        result = fe.extract_shape_features(
            time_s,
            trace,
            baseline_noise_pct=10.0,
        )

        self.assertFalse(result["shape_qc_pass"])
        self.assertEqual(result["shape_qc_reason"], "low_peak_snr")
        self.assertTrue(np.isnan(result["fwhm_width"]))
        self.assertTrue(np.isfinite(result["peak_blockage"]))

    def test_invalid_timestamps_fail_shape_qc(self):
        time_s = np.array([0.0, 0.1, 0.1, 0.2, 0.3])
        trace = np.array([0.0, 1.0, 3.0, 1.0, 0.0])
        result = fe.extract_shape_features(
            time_s,
            trace,
            baseline_noise_pct=0.1,
        )
        self.assertEqual(
            result["shape_qc_reason"],
            "invalid_event_timestamps",
        )

    def test_calculate_event_features_schema(self):
        time_s = np.arange(41, dtype=float) * 10e-6
        blockage = np.r_[
            np.zeros(10),
            np.arange(1, 11),
            np.arange(9, 0, -1),
            np.zeros(12),
        ].astype(float)
        baseline = np.array([99.8, 100.0, 100.2, 99.9, 100.1])
        result = fe.calculate_event_features(
            event_time_s=time_s,
            event_signal=100.0 - blockage,
            baseline_signal=baseline,
            start_time_s=0.0,
            end_time_s=0.00041,
            annotated_blockage_percent=10.0,
        )

        required = {
            "dwell_time_ms",
            "blockage_percent",
            "blockage_mean",
            "blockage_sd",
            "auc",
            "skewness",
            "kurtosis",
            "fwhm_width",
            "fwhm_drop",
            "rise_time",
            "fall_time",
            "right_slope",
            "left_slope",
            "peak_blockage",
            "peak_index",
            "peak_count",
            "peak_snr",
            "shape_qc_pass",
            "shape_qc_reason",
            "baseline_mean",
            "baseline_sd",
            "n_event_points",
            "segment_01_blockage_mean",
            "segment_10_blockage_mean",
            "segmentation_qc_pass",
            "segmentation_qc_reason",
            "minimum_segment_points",
            "insufficient_segment_count",
        }
        self.assertTrue(required.issubset(result))

    def test_shape_context_recovers_crossings_outside_annotation(self):
        time_s = np.arange(61, dtype=float) * 10e-6
        trace = np.r_[
            np.zeros(10),
            np.arange(1, 11),
            np.arange(9, 0, -1),
            np.zeros(32),
        ].astype(float)
        result = fe.extract_shape_features(
            time_s,
            trace,
            baseline_noise_pct=0.2,
            peak_search_start_index=15,
            peak_search_end_index=25,
        )
        self.assertTrue(result["rise_qc_pass"])
        self.assertTrue(result["fall_qc_pass"])
        self.assertTrue(result["fwhm_qc_pass"])

    def test_rectangular_event_has_uniform_segment_means_only(self):
        time_s = np.linspace(0.0, 0.001, 1001)
        blockage = np.full_like(time_s, 20.0)
        result = fe.calculate_segment_features(
            time_s,
            blockage,
            start_time_s=0.0,
            end_time_s=0.001,
        )

        self.assertTrue(result["segmentation_qc_pass"])
        self.assertEqual(result["insufficient_segment_count"], 0)
        for segment_number in range(1, 11):
            prefix = f"segment_{segment_number:02d}"
            self.assertAlmostEqual(
                result[f"{prefix}_blockage_mean"], 20.0
            )
        segmentation_outputs = {
            name for name in result
            if name.startswith("segment_") and "qc" not in name
        }
        self.assertEqual(
            segmentation_outputs,
            {f"segment_{number:02d}_blockage_mean" for number in range(1, 11)},
        )

    def test_short_event_fails_segmentation_without_partial_features(self):
        time_s = np.linspace(0.0, 0.001, 21)
        blockage = np.linspace(1.0, 20.0, len(time_s))
        result = fe.calculate_segment_features(
            time_s,
            blockage,
            start_time_s=0.0,
            end_time_s=0.001,
        )

        self.assertFalse(result["segmentation_qc_pass"])
        self.assertEqual(
            result["segmentation_qc_reason"],
            "insufficient_segmentation_points",
        )
        self.assertLess(result["minimum_segment_points"], 3)
        self.assertGreater(result["insufficient_segment_count"], 0)
        for segment_number in range(1, 11):
            prefix = f"segment_{segment_number:02d}"
            self.assertTrue(np.isnan(result[f"{prefix}_blockage_mean"]))

    def test_segment_offset_and_minimum_points(self):
        for offset in (0.0, 1000.0):
            edges = offset + 0.001 * np.linspace(0, 1, 11)
            time = np.concatenate([
                [edges[i], edges[i] + 0.00002, edges[i] + 0.00004]
                for i in range(10)
            ])
            values = np.repeat(np.arange(1., 11.), 3)
            result = fe.calculate_segment_features(time, values, offset, offset + 0.001)
            self.assertTrue(result['segmentation_qc_pass'])
            self.assertEqual(result['minimum_segment_points'], 3)
            for i in range(1, 11):
                self.assertAlmostEqual(result[f'segment_{i:02d}_blockage_mean'], i)

    def test_zero_blockage_has_valid_zero_segment_means(self):
        time = np.linspace(0., .001, 101)
        result = fe.calculate_segment_features(time, np.zeros(101), 0., .001)
        self.assertTrue(result['segmentation_qc_pass'])
        for number in range(1, 11):
            self.assertEqual(result[f'segment_{number:02d}_blockage_mean'], 0.0)

    def test_overflow_cannot_pass_segmentation(self):
        time = np.linspace(0., .001, 101)
        with np.errstate(over='ignore', invalid='ignore'):
            result = fe.calculate_segment_features(time, np.full(101, 1e308), 0., .001)
        self.assertFalse(result['segmentation_qc_pass'])
        self.assertEqual(result['segmentation_qc_reason'], 'nonfinite_segmentation_feature')

    def test_segment_boundaries_assign_each_sample_once(self):
        time_s = np.linspace(0.0, 0.001, 101)
        blockage = np.arange(101, dtype=float) + 1.0
        result = fe.calculate_segment_features(
            time_s,
            blockage,
            start_time_s=0.0,
            end_time_s=0.001,
        )

        self.assertTrue(result["segmentation_qc_pass"])
        self.assertAlmostEqual(result["segment_01_blockage_mean"], 5.5)
        self.assertAlmostEqual(result["segment_02_blockage_mean"], 15.5)
        self.assertAlmostEqual(result["segment_10_blockage_mean"], 96.0)


if __name__ == "__main__":
    unittest.main()
