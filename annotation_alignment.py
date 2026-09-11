"""Validation of JSON-embedded waveforms against their source ABF signal."""

import numpy as np


def _sample_event_indices(event_count, sample_count):
    """Return deterministic, evenly spaced event indices."""

    if event_count <= 0:
        return []
    count = min(int(sample_count), event_count)
    return sorted(set(np.linspace(0, event_count - 1, count, dtype=int)))


def compare_embedded_event(
    event,
    abf_time_s,
    abf_signal,
    time_offset_s,
    max_points=1000,
):
    """Compare one JSON waveform with ABF samples at the corrected time."""

    json_time_s = np.asarray(event.get("timestamp", []), dtype=float)
    json_signal = np.asarray(event.get("signal", []), dtype=float)
    if len(json_time_s) < 3 or len(json_time_s) != len(json_signal):
        raise ValueError("JSON event lacks an aligned timestamp/signal waveform")
    if not np.all(np.isfinite(json_time_s)) or not np.all(np.isfinite(json_signal)):
        raise ValueError("JSON embedded waveform contains nonfinite values")

    corrected_start_s = float(json_time_s[0]) + float(time_offset_s)
    base_index = int(np.searchsorted(abf_time_s, corrected_start_s, side="left"))
    candidates = range(max(base_index - 2, 0), min(base_index + 3, len(abf_signal)))
    best = None

    for start_index in candidates:
        point_count = min(int(max_points), len(json_signal), len(abf_signal) - start_index)
        if point_count < 3:
            continue
        observed = np.asarray(abf_signal[start_index:start_index + point_count], dtype=float)
        expected = json_signal[:point_count]
        correlation = float(np.corrcoef(observed, expected)[0, 1])
        scale = float(np.std(expected))
        normalized_rmse = float(
            np.sqrt(np.mean((observed - expected) ** 2)) / scale
        ) if scale > 0 else float(np.max(np.abs(observed - expected)))
        score = (correlation, -normalized_rmse)
        if best is None or score > best[0]:
            best = (score, start_index, point_count, correlation, normalized_rmse)

    if best is None:
        raise ValueError("JSON embedded waveform lies outside the ABF signal")

    _, start_index, point_count, correlation, normalized_rmse = best
    return {
        "json_waveform_start_s": float(json_time_s[0]),
        "matched_abf_start_s": float(abf_time_s[start_index]),
        "configured_time_offset_s": float(time_offset_s),
        "observed_time_offset_s": float(abf_time_s[start_index] - json_time_s[0]),
        "alignment_points": int(point_count),
        "alignment_correlation": correlation,
        "alignment_normalized_rmse": normalized_rmse,
    }


def validate_recording_alignment(
    events,
    abf_time_s,
    abf_signal,
    time_offset_s,
    recording_id,
    sample_count=3,
    minimum_correlation=0.999,
    maximum_normalized_rmse=0.01,
):
    """Validate representative JSON waveforms and return audit rows."""

    rows = []
    for event_index in _sample_event_indices(len(events), sample_count):
        result = compare_embedded_event(
            events[event_index], abf_time_s, abf_signal, time_offset_s
        )
        passed = (
            result["alignment_correlation"] >= float(minimum_correlation)
            and result["alignment_normalized_rmse"] <= float(maximum_normalized_rmse)
        )
        row = {
            "recording_id": recording_id,
            "source_event_id": int(event_index),
            **result,
            "alignment_pass": bool(passed),
        }
        rows.append(row)
        if not passed:
            raise ValueError(
                f"JSON/ABF alignment failed for {recording_id}, event {event_index}: "
                f"correlation={result['alignment_correlation']:.6f}, "
                f"normalized_rmse={result['alignment_normalized_rmse']:.6g}"
            )
    return rows
