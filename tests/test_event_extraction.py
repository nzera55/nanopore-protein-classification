import tempfile
import unittest
from unittest.mock import patch
import contextlib
import io
from pathlib import Path

import numpy as np
import pandas as pd

import event_extraction as ee
from annotation_alignment import validate_recording_alignment
from project_config import load_project_config


class EventExtractionTests(unittest.TestCase):
    def test_duplicate_does_not_change_features_or_cross_next_event(self):
        time = np.arange(2000) * 1e-5
        signal = -1000 + np.sin(np.arange(2000))
        signal[500:551] += 300
        signal[551:571] += np.linspace(300, 0, 20)
        data = dict(time_s=time, signal=signal, sampling_period_s=1e-5,
                    sampling_rate_hz=100000., sweep_count=1)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'events.csv'
            manifest = pd.DataFrame([dict(protein='F8', recording_id='r',
                annotation_id='a', annotation_path=str(path), annotation_format='csv',
                json_path='', abf_path='synthetic.abf', sweep_index=0,
                event_time_unit='ms', time_offset_s=0.)])
            results = []
            for duplicate in (False, True):
                path.write_text('Event start,Event end,%blockage\n5,5.5,30\n'
                                + ('5,5.5,30\n' if duplicate else '')
                                + '5.8,6.2,10\n')
                with patch.object(ee, 'load_abf_signal', return_value=data), contextlib.redirect_stdout(io.StringIO()):
                    f, q, _ = ee.process_abf_group('synthetic.abf', 0, manifest, 1.)
                results.append(f.iloc[0])
                self.assertEqual(int(q.qc_reason.eq('duplicate_annotation').sum()), int(duplicate))
            for column in results[0].index:
                if column.startswith('source_'):
                    continue
                a, b = results[0][column], results[1][column]
                if pd.isna(a) and pd.isna(b):
                    continue
                self.assertEqual(a, b, column)
            self.assertTrue(results[1].shape_qc_pass)
            self.assertGreater(results[1].actual_shape_context_post_points, 0)
            self.assertEqual(results[1].shape_end_index, np.searchsorted(time, .0058))

    def test_project_configuration_resolves_manifest(self):
        config = load_project_config()
        self.assertTrue(Path(config["manifest_path"]).is_file())
        self.assertEqual(
            config["dataset_version"],
            "v9_centered_zero_duration_reconstruction",
        )

    def test_event_indices_include_annotated_end_sample(self):
        time_s = np.arange(20, dtype=float) * 0.001
        start, end = ee.get_event_indices(time_s, 0.005, 0.010)
        self.assertEqual(start, 5)
        self.assertEqual(end, 11)

    def test_legacy_manifest_is_normalized_to_annotation_columns(self):
        with tempfile.TemporaryDirectory() as directory:
            legacy_path = Path(directory) / "legacy_recordings.csv"
            legacy_path.write_text(
                "protein,recording_id,json_path,abf_path,sweep_index,time_offset_s\n"
                "HSA,HSA_01,example.json,example.abf,0,0.0\n",
                encoding="utf-8",
            )
            manifest = ee.load_recording_manifest(legacy_path)
        self.assertIn("annotation_path", manifest.columns)
        self.assertTrue((manifest["annotation_format"] == "json").all())
        self.assertTrue((manifest["event_time_unit"] == "s").all())

    def test_current_manifest_resolves_and_validates_tabular_annotations(self):
        manifest = ee.load_recording_manifest(
            Path(__file__).resolve().parents[4] / "recordings.csv"
        )
        ee.validate_recording_manifest(manifest)
        self.assertEqual(len(manifest), 4)
        self.assertTrue((manifest["annotation_format"] == "csv").all())
        self.assertEqual(manifest["recording_id"].nunique(), 4)

    def test_csv_loader_preserves_invalid_rows_with_specific_qc(self):
        content = (
            "Event start,Event end,Amplitude,Amp S.D.,Dwell time(ms),"
            "%blockage,Level 0\n"
            "10,12,5,1,2,10,-50\n"
            "20,20,4,1,0,8,-50\n"
            ",,,,,,\n"
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "events.csv"
            path.write_text(content, encoding="utf-8")
            events = ee.load_csv_events(path, event_time_unit="ms")

        self.assertEqual(len(events), 3)
        self.assertAlmostEqual(events[0]["eventStartTime"], 0.010)
        self.assertAlmostEqual(events[0]["eventEndTime"], 0.012)
        self.assertEqual(events[0]["_source_row_number"], 2)
        self.assertEqual(events[1]["_load_qc_reason"], "nonpositive_event_duration")
        self.assertEqual(events[2]["_load_qc_reason"], "missing_event_start")

    def test_equal_boundary_is_reconstructed_symmetrically_from_dwell_time(self):
        content = (
            "Event start,Event end,Dwell time(ms),%blockage\n"
            "20,20,0.2,8\n"
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "events.csv"
            path.write_text(content, encoding="utf-8")
            event = ee.load_csv_events(path, event_time_unit="ms")[0]

        self.assertAlmostEqual(event["eventStartTime"], 0.0199)
        self.assertAlmostEqual(event["eventEndTime"], 0.0201)
        self.assertEqual(event["_source_event_start"], "20")
        self.assertEqual(event["_source_event_end"], "20")
        self.assertTrue(event["_event_boundary_reconstructed"])
        self.assertEqual(
            event["_boundary_reconstruction_method"],
            "centered_equal_boundary_using_source_dwell_time",
        )
        self.assertEqual(event["_load_qc_reason"], "")

    def test_excel_loader_matches_csv_time_and_source_fields(self):
        table = pd.DataFrame({
            "Event start": [1.0, ""],
            "Event end": [1.5, 2.0],
            "Amplitude": [12.0, 3.0],
            "Amp S.D.": [0.5, 0.2],
            "Dwell time(ms)": [0.5, 0.4],
            "%blockage": [20.0, 10.0],
            "Level 0": [-100.0, -100.0],
        })
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "events.xlsx"
            table.to_excel(path, index=False)
            events = ee.load_excel_events(path, "ms")
        self.assertAlmostEqual(events[0]["eventStartTime"], 0.001)
        self.assertAlmostEqual(events[0]["eventEndTime"], 0.0015)
        self.assertEqual(events[0]["_annotation_format"], "excel")
        self.assertEqual(events[1]["_load_qc_reason"], "missing_event_start")

    def test_raw_waveform_archive_round_trips_exact_abf_samples(self):
        abf = {
            "time_s": np.arange(20, dtype=float) * 0.001,
            "signal": np.arange(20, dtype=float) + 100.0,
            "sampling_period_s": 0.001,
            "sampling_rate_hz": 1000.0,
            "sweep_count": 1,
            "signal_unit": "pA",
        }
        features = pd.DataFrame({
            "protein": ["HSA", "HSA"],
            "recording_id": ["r1", "r1"],
            "annotation_id": ["a1", "a1"],
            "event_id": ["a1:0", "a1:1"],
            "abf_path": ["synthetic.abf", "synthetic.abf"],
            "sweep_index": [0, 0],
            "event_start_index": [2, 10],
            "event_end_index": [5, 14],
            "n_event_points": [3, 4],
            "sampling_rate_hz": [1000.0, 1000.0],
            "abf_event_start_time_s": [0.002, 0.010],
            "abf_event_end_time_s": [0.005, 0.014],
            "baseline_mean": [100.0, 100.0],
        })
        with tempfile.TemporaryDirectory() as directory, patch.object(
            ee, "load_abf_signal", return_value=abf
        ):
            paths = ee.save_raw_event_waveforms(features, directory)
            index = pd.read_csv(paths["raw_waveform_index_path"])
            with np.load(paths["raw_waveform_path"], allow_pickle=False) as saved:
                np.testing.assert_array_equal(
                    saved["raw_signal"], [102, 103, 104, 110, 111, 112, 113]
                )
                np.testing.assert_allclose(
                    saved["raw_time_s"], [0.002, 0.003, 0.004, 0.010, 0.011, 0.012, 0.013]
                )
                np.testing.assert_array_equal(saved["offsets"], [0, 3, 7])
                np.testing.assert_array_equal(saved["baseline_mean"], [100.0, 100.0])
                np.testing.assert_array_equal(saved["event_start_time_s"], [0.002, 0.010])
                np.testing.assert_array_equal(saved["event_end_time_s"], [0.005, 0.014])
        self.assertEqual(index.event_id.tolist(), ["a1:0", "a1:1"])
        self.assertEqual(index.signal_unit.tolist(), ["pA", "pA"])

    def test_process_single_event_preserves_extraction_metadata(self):
        time_s = np.arange(100, dtype=float) * 10e-6
        signal = np.full(100, 100.0)
        blockage = np.r_[
            np.zeros(10),
            np.arange(1, 11),
            np.arange(9, 0, -1),
            np.zeros(12),
        ].astype(float)
        signal[30:71] = 100.0 - blockage
        event = {
            "eventStartTime": float(time_s[30]),
            "eventEndTime": float(time_s[70]),
            "blockagePercent": 10.0,
        }
        result = ee.process_single_event(
            event,
            time_s,
            signal,
            baseline_points=10,
        )

        self.assertEqual(result["event_start_index"], 30)
        self.assertEqual(result["event_end_index"], 71)
        self.assertEqual(result["actual_baseline_points"], 10)
        self.assertIn("shape_qc_pass", result)

    def test_process_single_event_applies_time_offset_only_to_abf_mapping(self):
        time_s = np.arange(200, dtype=float) * 0.001
        signal = np.full(200, 100.0)
        signal[60:71] = 90.0
        event = {
            "eventStartTime": 0.010,
            "eventEndTime": 0.020,
            "blockagePercent": 10.0,
        }
        result = ee.process_single_event(
            event, time_s, signal, baseline_points=10, time_offset_s=0.050
        )
        self.assertEqual(result["event_start_index"], 60)
        self.assertEqual(result["event_end_index"], 71)
        self.assertAlmostEqual(result["event_start_time_s"], 0.010)
        self.assertAlmostEqual(result["abf_event_start_time_s"], 0.060)
        self.assertAlmostEqual(result["dwell_time_ms"], 10.0)

    def test_shape_context_does_not_change_annotated_event_features(self):
        time_s = np.arange(100, dtype=float) * 10e-6
        signal = np.full(100, 100.0)
        signal[25:35] = 100.0 - np.arange(1, 11)
        signal[35:45] = 100.0 - np.arange(10, 0, -1)
        event = {
            "eventStartTime": float(time_s[30]),
            "eventEndTime": float(time_s[39]),
            "blockagePercent": 10.0,
        }
        without_context = ee.process_single_event(
            event, time_s, signal, baseline_points=10
        )
        with_context = ee.process_single_event(
            event, time_s, signal, baseline_points=10,
            shape_context_points=10,
        )
        self.assertAlmostEqual(
            without_context["blockage_mean"],
            with_context["blockage_mean"],
        )
        self.assertAlmostEqual(
            without_context["dwell_time_ms"],
            with_context["dwell_time_ms"],
        )
        self.assertTrue(with_context["rise_qc_pass"])
        self.assertTrue(with_context["fall_qc_pass"])

    def test_alignment_validation_accepts_configured_offset(self):
        time_s = np.arange(1000, dtype=float) * 0.001
        signal = np.sin(np.arange(1000, dtype=float) * 0.1)
        event = {
            "timestamp": time_s[100:180].tolist(),
            "signal": signal[150:230].tolist(),
        }
        rows = validate_recording_alignment(
            [event], time_s, signal, time_offset_s=0.050,
            recording_id="synthetic", sample_count=1,
        )
        self.assertTrue(rows[0]["alignment_pass"])
        self.assertAlmostEqual(rows[0]["observed_time_offset_s"], 0.050)

    def test_alignment_validation_rejects_wrong_offset(self):
        time_s = np.arange(1000, dtype=float) * 0.001
        rng = np.random.default_rng(42)
        signal = rng.normal(size=1000)
        event = {
            "timestamp": time_s[100:180].tolist(),
            "signal": signal[150:230].tolist(),
        }
        with self.assertRaisesRegex(ValueError, "alignment failed"):
            validate_recording_alignment(
                [event], time_s, signal, time_offset_s=0.0,
                recording_id="synthetic", sample_count=1,
            )

    def test_recording_qc_summary_separates_extraction_and_shape_qc(self):
        features = pd.DataFrame({
            "protein": ["HSA", "HSA"],
            "recording_id": ["HSA_01", "HSA_01"],
            "shape_qc_pass": [True, False],
            "peak_snr": [5.0, 2.0],
        })
        qc = pd.DataFrame({
            "protein": ["HSA", "HSA", "HSA"],
            "recording_id": ["HSA_01"] * 3,
            "qc_pass": [True, True, False],
            "qc_reason": ["", "", "ignored"],
        })
        summary = ee.build_recording_qc_summary(features, qc).iloc[0]
        self.assertEqual(summary["n_annotated"], 3)
        self.assertEqual(summary["n_accepted"], 2)
        self.assertEqual(summary["n_ignored"], 1)
        self.assertEqual(summary["shape_qc_pass_count"], 1)
        self.assertAlmostEqual(summary["shape_qc_pass_rate"], 0.5)

    def test_save_results_writes_complete_schema(self):
        features = pd.DataFrame({
            "protein": ["F8"],
            "recording_id": ["F8_01"],
            "shape_qc_pass": [True],
            "peak_snr": [5.0],
        })
        qc = pd.DataFrame({
            "protein": ["F8"],
            "recording_id": ["F8_01"],
            "qc_pass": [True],
            "qc_reason": [""],
        })

        with tempfile.TemporaryDirectory() as directory:
            alignment = pd.DataFrame({
                "recording_id": ["F8_01"],
                "alignment_pass": [True],
            })
            paths = ee.save_results(features, qc, directory, alignment=alignment)
            qc_saved = pd.read_csv(paths["qc_path"])
            self.assertIn("extraction_qc_pass", qc_saved)
            self.assertIn("extraction_qc_reason", qc_saved)
            self.assertTrue(paths["recording_summary_path"].is_file())
            self.assertTrue(paths["alignment_path"].is_file())
            recording_directory = Path(directory) / "recordings" / "F8_01"
            self.assertTrue((recording_directory / "event_features.csv").is_file())
            self.assertTrue((recording_directory / "event_qc.csv").is_file())
            self.assertTrue(
                (recording_directory / "annotation_alignment.csv").is_file()
            )


if __name__ == "__main__":
    unittest.main()
