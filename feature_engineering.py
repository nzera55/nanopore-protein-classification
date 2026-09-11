"""Numerical feature engineering for annotated nanopore events.

This module is intentionally independent of JSON, ABF, manifests, and file
paths. It receives already-extracted NumPy arrays and returns scalar features
and shape-quality diagnostics.
"""

import numpy as np

from scipy.signal import find_peaks
from scipy.stats import kurtosis, skew


SHAPE_SMOOTHING_WINDOW_MS = 0.05
MIN_SHAPE_PEAK_SNR = 3.0
MIN_SHAPE_POINTS = 5
MIN_MOMENT_POINTS = 8
N_TIME_SEGMENTS = 10
MIN_SEGMENT_POINTS = 3


def configure_feature_engineering(
    shape_smoothing_window_ms,
    minimum_shape_peak_snr,
):
    """Apply approved feature parameters from the run configuration."""

    global SHAPE_SMOOTHING_WINDOW_MS
    global MIN_SHAPE_PEAK_SNR

    shape_smoothing_window_ms = float(
        shape_smoothing_window_ms
    )
    minimum_shape_peak_snr = float(
        minimum_shape_peak_snr
    )

    if shape_smoothing_window_ms <= 0:
        raise ValueError(
            "shape_smoothing_window_ms must be positive"
        )

    if minimum_shape_peak_snr <= 0:
        raise ValueError(
            "minimum_shape_peak_snr must be positive"
        )

    SHAPE_SMOOTHING_WINDOW_MS = shape_smoothing_window_ms
    MIN_SHAPE_PEAK_SNR = minimum_shape_peak_snr


def interpolate_crossing(time_ms, values, index_1, index_2, target):
    """Linearly interpolate a threshold crossing between two samples."""

    time_1 = time_ms[index_1]
    time_2 = time_ms[index_2]
    value_1 = values[index_1]
    value_2 = values[index_2]

    if np.isclose(value_1, value_2):
        return (time_1 + time_2) / 2.0

    fraction = (target - value_1) / (value_2 - value_1)
    return time_1 + fraction * (time_2 - time_1)


def find_left_crossing(time_ms, values, peak_index, target):
    """Find the closest upward crossing to the left of a peak."""

    crossing_indices = []

    for index in range(peak_index):
        if values[index] <= target and values[index + 1] > target:
            crossing_indices.append(index)

    if not crossing_indices:
        return np.nan

    index_1 = crossing_indices[-1]
    return interpolate_crossing(
        time_ms,
        values,
        index_1,
        index_1 + 1,
        target,
    )


def find_right_crossing(time_ms, values, peak_index, target):
    """Find the first downward crossing to the right of a peak."""

    for index in range(peak_index, len(values) - 1):
        if values[index] >= target and values[index + 1] < target:
            return interpolate_crossing(
                time_ms,
                values,
                index,
                index + 1,
                target,
            )

    return np.nan


def smooth_trace(values, window_points=5):
    """Apply a centered moving average using edge-value padding."""

    values = np.asarray(values, dtype=float)
    window_points = int(window_points)

    if window_points <= 1:
        return values.copy()

    window_points = min(window_points, len(values))

    if window_points % 2 == 0:
        window_points -= 1

    if window_points <= 1:
        return values.copy()

    kernel = np.ones(window_points, dtype=float) / window_points
    pad_width = window_points // 2
    padded_values = np.pad(values, pad_width=pad_width, mode="edge")

    return np.convolve(padded_values, kernel, mode="valid")


def _empty_segmentation_result(reason, segment_counts=None):
    """Return failed segmentation features and useful point diagnostics."""

    if segment_counts is None:
        minimum_segment_points = 0
        insufficient_segment_count = N_TIME_SEGMENTS
    else:
        segment_counts = np.asarray(segment_counts, dtype=int)
        minimum_segment_points = int(np.min(segment_counts))
        insufficient_segment_count = int(
            np.sum(segment_counts < MIN_SEGMENT_POINTS)
        )

    result = {}
    for segment_number in range(1, N_TIME_SEGMENTS + 1):
        prefix = f"segment_{segment_number:02d}"
        result[f"{prefix}_blockage_mean"] = np.nan

    result.update({
        "segmentation_qc_pass": False,
        "segmentation_qc_reason": reason,
        "minimum_segment_points": minimum_segment_points,
        "insufficient_segment_count": insufficient_segment_count,
    })
    return result


def calculate_segment_features(
    event_time_s,
    blockage_trace_pct,
    start_time_s,
    end_time_s,
):
    """Calculate mean blockage in ten equal normalized-time segments.

    Raw ABF samples are assigned to half-open normalized-time segments, with
    the annotated right boundary included in the final segment. Each segment
    must contain at least three observed ABF samples. No interpolation,
    smoothing, slope, dispersion, AUC, or cross-segment summary is used.
    """

    event_time_s = np.asarray(event_time_s, dtype=float)
    blockage_trace_pct = np.asarray(blockage_trace_pct, dtype=float)

    if len(event_time_s) != len(blockage_trace_pct):
        return _empty_segmentation_result(
            "segmentation_time_signal_length_mismatch"
        )

    if (
        not np.all(np.isfinite(event_time_s))
        or not np.all(np.isfinite(blockage_trace_pct))
    ):
        return _empty_segmentation_result(
            "nonfinite_segmentation_feature"
        )

    time_differences_s = np.diff(event_time_s)
    if np.any(time_differences_s <= 0):
        return _empty_segmentation_result(
            "invalid_segmentation_timestamps"
        )

    start_time_s = float(start_time_s)
    end_time_s = float(end_time_s)
    duration_s = end_time_s - start_time_s
    if (
        not np.isfinite(start_time_s)
        or not np.isfinite(end_time_s)
        or duration_s <= 0
    ):
        return _empty_segmentation_result(
            "invalid_segmentation_boundaries"
        )

    normalized_time = (event_time_s - start_time_s) / duration_s
    if np.any(normalized_time < 0) or np.any(normalized_time > 1):
        return _empty_segmentation_result(
            "samples_outside_segmentation_boundaries"
        )

    # Compare timestamps directly with representable event boundaries. Dividing
    # by duration and flooring can move an exact boundary sample to the left.
    segment_edges = np.linspace(0.0, 1.0, N_TIME_SEGMENTS + 1)
    absolute_edges = start_time_s + duration_s * segment_edges
    segment_indices = np.searchsorted(
        absolute_edges[1:-1], event_time_s, side="right"
    )
    segment_counts = np.bincount(
        segment_indices,
        minlength=N_TIME_SEGMENTS,
    )
    if np.any(segment_counts < MIN_SEGMENT_POINTS):
        return _empty_segmentation_result(
            "insufficient_segmentation_points",
            segment_counts=segment_counts,
        )

    result = {}
    for segment_index in range(N_TIME_SEGMENTS):
        mask = segment_indices == segment_index
        segment_values = blockage_trace_pct[mask]
        prefix = f"segment_{segment_index + 1:02d}"
        result[f"{prefix}_blockage_mean"] = float(np.mean(segment_values))

    if not np.all(np.isfinite(list(result.values()))):
        return _empty_segmentation_result(
            "nonfinite_segmentation_feature", segment_counts=segment_counts
        )
    result.update({
        "segmentation_qc_pass": True,
        "segmentation_qc_reason": "",
        "minimum_segment_points": int(np.min(segment_counts)),
        "insufficient_segment_count": 0,
    })
    return result


def _empty_shape_result(reason):
    """Return a complete failed shape-feature record."""

    return {
        "fwhm_width": np.nan,
        "fwhm_drop": np.nan,
        "rise_time": np.nan,
        "fall_time": np.nan,
        "right_slope": np.nan,
        "left_slope": np.nan,
        "peak_blockage": np.nan,
        "peak_index": np.nan,
        "peak_count": 0,
        "peak_snr": np.nan,
        "shape_qc_pass": False,
        "shape_qc_reason": reason,
        "fwhm_qc_pass": False,
        "rise_qc_pass": False,
        "fall_qc_pass": False,
    }


def _smoothing_window_points(sample_period_ms, trace_length):
    """Convert the configured smoothing duration to a valid odd window."""

    window_points = max(
        int(round(SHAPE_SMOOTHING_WINDOW_MS / sample_period_ms)),
        1,
    )

    if window_points % 2 == 0:
        window_points += 1

    window_points = min(window_points, trace_length)

    if window_points % 2 == 0:
        window_points -= 1

    return max(window_points, 1)


def extract_shape_features(
    event_time_s,
    blockage_trace_pct,
    baseline_noise_pct,
    peak_search_start_index=0,
    peak_search_end_index=None,
):
    """Calculate shape features using context around an annotated event.

    The dominant peak is selected only inside the annotated event, while
    threshold crossings may be found in the supplied pre/post-event context.
    """

    event_time_s = np.asarray(event_time_s, dtype=float)
    blockage_trace_pct = np.asarray(blockage_trace_pct, dtype=float)

    if len(event_time_s) != len(blockage_trace_pct):
        return _empty_shape_result("time_signal_length_mismatch")

    if len(blockage_trace_pct) < MIN_SHAPE_POINTS:
        return _empty_shape_result("insufficient_shape_points")

    if not np.all(np.isfinite(blockage_trace_pct)):
        return _empty_shape_result("nonfinite_blockage_trace")

    time_ms = (event_time_s - event_time_s[0]) * 1000.0
    time_differences_ms = np.diff(time_ms)

    if (
        not np.all(np.isfinite(time_differences_ms))
        or np.any(time_differences_ms <= 0)
    ):
        return _empty_shape_result("invalid_event_timestamps")

    sample_period_ms = float(np.median(time_differences_ms))
    window_points = _smoothing_window_points(
        sample_period_ms,
        len(blockage_trace_pct),
    )
    smoothed_trace_pct = smooth_trace(blockage_trace_pct, window_points)

    if peak_search_end_index is None:
        peak_search_end_index = len(smoothed_trace_pct)

    peak_search_start_index = int(peak_search_start_index)
    peak_search_end_index = int(peak_search_end_index)
    if not (
        0 <= peak_search_start_index
        < peak_search_end_index
        <= len(smoothed_trace_pct)
    ):
        return _empty_shape_result("invalid_peak_search_bounds")

    peak_index = peak_search_start_index + int(np.argmax(
        smoothed_trace_pct[
            peak_search_start_index:peak_search_end_index
        ]
    ))
    peak_blockage_pct = float(smoothed_trace_pct[peak_index])
    result = _empty_shape_result("")
    result.update({
        "peak_blockage": peak_blockage_pct,
        "peak_index": peak_index,
    })

    if not np.isfinite(peak_blockage_pct) or peak_blockage_pct <= 0:
        result["shape_qc_reason"] = "nonpositive_peak_blockage"
        return result

    baseline_noise_pct = float(baseline_noise_pct)

    if np.isfinite(baseline_noise_pct) and baseline_noise_pct > 0:
        peak_snr = peak_blockage_pct / baseline_noise_pct
        minimum_prominence_pct = MIN_SHAPE_PEAK_SNR * baseline_noise_pct
    elif baseline_noise_pct == 0:
        peak_snr = np.inf
        minimum_prominence_pct = 0.0
    else:
        peak_snr = np.nan
        minimum_prominence_pct = None

    if minimum_prominence_pct is None:
        peak_indices = np.asarray([], dtype=int)
    else:
        all_peak_indices, _ = find_peaks(
            smoothed_trace_pct,
            prominence=minimum_prominence_pct,
        )
        peak_indices = all_peak_indices[
            (all_peak_indices >= peak_search_start_index)
            & (all_peak_indices < peak_search_end_index)
        ]
    result.update({
        "peak_count": int(len(peak_indices)),
        "peak_snr": peak_snr,
    })

    if np.isnan(peak_snr):
        result["shape_qc_reason"] = "baseline_noise_unavailable"
        return result

    if peak_snr < MIN_SHAPE_PEAK_SNR:
        result["shape_qc_reason"] = "low_peak_snr"
        return result

    level_10 = 0.10 * peak_blockage_pct
    level_50 = 0.50 * peak_blockage_pct
    level_90 = 0.90 * peak_blockage_pct

    left_10 = find_left_crossing(
        time_ms, smoothed_trace_pct, peak_index, level_10
    )
    left_50 = find_left_crossing(
        time_ms, smoothed_trace_pct, peak_index, level_50
    )
    left_90 = find_left_crossing(
        time_ms, smoothed_trace_pct, peak_index, level_90
    )
    right_90 = find_right_crossing(
        time_ms, smoothed_trace_pct, peak_index, level_90
    )
    right_50 = find_right_crossing(
        time_ms, smoothed_trace_pct, peak_index, level_50
    )
    right_10 = find_right_crossing(
        time_ms, smoothed_trace_pct, peak_index, level_10
    )

    if np.isfinite(left_50) and np.isfinite(right_50) and right_50 > left_50:
        fwhm_width_ms = right_50 - left_50
    else:
        fwhm_width_ms = np.nan

    if np.isfinite(left_10) and np.isfinite(left_90) and left_90 > left_10:
        rise_time_ms = left_90 - left_10
        left_slope = (level_90 - level_10) / rise_time_ms
    else:
        rise_time_ms = np.nan
        left_slope = np.nan

    if np.isfinite(right_90) and np.isfinite(right_10) and right_10 > right_90:
        fall_time_ms = right_10 - right_90
        right_slope = (level_10 - level_90) / fall_time_ms
    else:
        fall_time_ms = np.nan
        right_slope = np.nan

    result.update({
        "fwhm_width": fwhm_width_ms,
        "fwhm_drop": level_50,
        "rise_time": rise_time_ms,
        "fall_time": fall_time_ms,
        "right_slope": right_slope,
        "left_slope": left_slope,
        "fwhm_qc_pass": bool(np.isfinite(fwhm_width_ms)),
        "rise_qc_pass": bool(
            np.isfinite(rise_time_ms) and np.isfinite(left_slope)
        ),
        "fall_qc_pass": bool(
            np.isfinite(fall_time_ms) and np.isfinite(right_slope)
        ),
    })

    required_shape_features = [
        "fwhm_width",
        "fwhm_drop",
        "rise_time",
        "fall_time",
        "right_slope",
        "left_slope",
    ]
    missing = [
        name
        for name in required_shape_features
        if not np.isfinite(result[name])
    ]

    if len(peak_indices) == 0:
        result["shape_qc_reason"] = "no_prominent_local_peak"
    elif missing:
        result["shape_qc_reason"] = (
            "missing_threshold_crossing:" + ",".join(missing)
        )
    else:
        result["shape_qc_pass"] = True
        result["shape_qc_reason"] = ""

    return result


def calculate_event_features(
    event_time_s,
    event_signal,
    baseline_signal,
    start_time_s,
    end_time_s,
    annotated_blockage_percent=np.nan,
    shape_time_s=None,
    shape_signal=None,
    shape_baseline_signal=None,
    shape_event_start_index=0,
    shape_event_end_index=None,
):
    """Calculate every scalar feature for one extracted event."""

    event_time_s = np.asarray(event_time_s, dtype=float)
    event_signal = np.asarray(event_signal, dtype=float)
    baseline_signal = np.asarray(baseline_signal, dtype=float)

    if len(event_time_s) != len(event_signal):
        raise ValueError("Event time and signal lengths differ")

    if len(event_signal) < 3:
        raise ValueError("Event signal is too short")

    if len(baseline_signal) < 3:
        raise ValueError("Baseline signal is too short")

    if not np.all(np.isfinite(event_time_s)):
        raise ValueError("Event timestamps contain NaN or infinity")

    if not np.all(np.isfinite(event_signal)):
        raise ValueError("Event signal contains NaN or infinity")

    if not np.all(np.isfinite(baseline_signal)):
        raise ValueError("Baseline signal contains NaN or infinity")

    baseline_mean = float(np.mean(baseline_signal))
    baseline_sd = float(np.std(baseline_signal, ddof=1))

    if np.isclose(baseline_mean, 0):
        raise ValueError("Baseline mean is close to zero")

    blockage_trace_pct = (
        np.abs(baseline_mean - event_signal)
        / np.abs(baseline_mean)
        * 100.0
    )
    dwell_time_ms = (float(end_time_s) - float(start_time_s)) * 1000.0
    blockage_mean_pct = float(np.mean(blockage_trace_pct))
    blockage_sd_pct = float(np.std(blockage_trace_pct, ddof=1))
    relative_time_ms = (event_time_s - event_time_s[0]) * 1000.0
    auc_pct_ms = float(np.trapezoid(blockage_trace_pct, relative_time_ms))
    segment_features = calculate_segment_features(
        event_time_s=event_time_s,
        blockage_trace_pct=blockage_trace_pct,
        start_time_s=start_time_s,
        end_time_s=end_time_s,
    )

    if len(blockage_trace_pct) >= MIN_MOMENT_POINTS:
        event_skewness = float(skew(blockage_trace_pct, bias=False))
        event_kurtosis = float(
            kurtosis(blockage_trace_pct, fisher=True, bias=False)
        )
    else:
        event_skewness = np.nan
        event_kurtosis = np.nan

    if shape_time_s is None:
        shape_time_s = event_time_s
    if shape_signal is None:
        shape_signal = event_signal

    shape_time_s = np.asarray(shape_time_s, dtype=float)
    shape_signal = np.asarray(shape_signal, dtype=float)
    if len(shape_time_s) != len(shape_signal):
        raise ValueError("Shape time and signal lengths differ")

    if shape_baseline_signal is None:
        shape_baseline_signal = baseline_signal
    shape_baseline_signal = np.asarray(shape_baseline_signal, dtype=float)
    if len(shape_baseline_signal) >= 3:
        shape_baseline_mean = float(np.mean(shape_baseline_signal))
        shape_baseline_sd = float(np.std(shape_baseline_signal, ddof=1))
    else:
        shape_baseline_mean = baseline_mean
        shape_baseline_sd = baseline_sd

    shape_blockage_trace_pct = (
        np.abs(shape_baseline_mean - shape_signal)
        / np.abs(shape_baseline_mean)
        * 100.0
    )
    shape_features = extract_shape_features(
        event_time_s=shape_time_s,
        blockage_trace_pct=shape_blockage_trace_pct,
        baseline_noise_pct=(
            shape_baseline_sd / np.abs(shape_baseline_mean) * 100.0
        ),
        peak_search_start_index=shape_event_start_index,
        peak_search_end_index=shape_event_end_index,
    )

    return {
        "dwell_time_ms": dwell_time_ms,
        "blockage_percent": annotated_blockage_percent,
        "blockage_mean": blockage_mean_pct,
        "blockage_sd": blockage_sd_pct,
        "auc": auc_pct_ms,
        "skewness": event_skewness,
        "kurtosis": event_kurtosis,
        **segment_features,
        **shape_features,
        "baseline_mean": baseline_mean,
        "baseline_sd": baseline_sd,
        "n_event_points": len(event_signal),
    }
