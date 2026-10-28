import argparse
import csv
import json
from decimal import Decimal
from pathlib import Path

import numpy as np
import pandas as pd
import pyabf

from annotation_alignment import validate_recording_alignment
from feature_engineering import (
    calculate_event_features,
    configure_feature_engineering,
)
from project_config import (
    DEFAULT_CONFIG_PATH,
    load_project_config,
    make_run_id,
    require_new_output_directory,
    sha256_file,
    write_run_manifest,
)


MIN_BASELINE_POINTS = 3
MIN_EVENT_POINTS = 3


def apply_time_offset(time_s, time_offset_s):
    """Apply a recording offset without sample-boundary float drift."""

    return float(Decimal(str(time_s)) + Decimal(str(time_offset_s)))


# =========================================================
# Manifest loading and validation
# =========================================================

def load_recording_manifest(
    manifest_path,
):
    """
    Load a recording manifest and resolve annotation/ABF paths
    relative to the directory containing the CSV.
    """

    manifest_path = Path(
        manifest_path
    ).resolve()

    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"Manifest not found: {manifest_path}"
        )

    manifest = pd.read_csv(
        manifest_path
    )

    if "annotation_path" not in manifest.columns:
        if "json_path" not in manifest.columns:
            raise ValueError(
                "Manifest requires annotation_path or legacy json_path"
            )
        manifest["annotation_path"] = manifest["json_path"]
        manifest["annotation_format"] = "json"

    defaults = {
        "annotation_id": manifest.get("recording_id"),
        "annotation_format": "json",
        "event_time_unit": "s",
        "time_reference": "recording_start",
        "enabled": True,
    }
    for column, default in defaults.items():
        if column not in manifest.columns:
            manifest[column] = default

    required_columns = [
        "protein",
        "annotation_id",
        "recording_id",
        "annotation_path",
        "annotation_format",
        "abf_path",
        "sweep_index",
        "event_time_unit",
        "time_reference",
        "time_offset_s",
        "enabled",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in manifest.columns
    ]

    if missing_columns:
        raise ValueError(
            "Manifest is missing columns: "
            f"{missing_columns}"
        )

    data_root = manifest_path.parent

    def resolve_data_path(value):
        path = Path(str(value))

        if not path.is_absolute():
            path = data_root / path

        return str(path.resolve())

    manifest["annotation_path"] = (
        manifest["annotation_path"]
        .map(resolve_data_path)
    )

    # Keep the legacy provenance column while JSON support is retained.
    manifest["json_path"] = np.where(
        manifest["annotation_format"].astype(str).str.lower().eq("json"),
        manifest["annotation_path"],
        "",
    )

    manifest["abf_path"] = (
        manifest["abf_path"]
        .map(resolve_data_path)
    )

    manifest["sweep_index"] = (
        pd.to_numeric(
            manifest["sweep_index"],
            errors="raise",
        )
        .astype(int)
    )

    manifest["time_offset_s"] = pd.to_numeric(
        manifest["time_offset_s"], errors="raise"
    ).astype(float)

    manifest["protein"] = (
        manifest["protein"]
        .astype(str)
        .str.strip()
    )

    manifest["recording_id"] = (
        manifest["recording_id"]
        .astype(str)
        .str.strip()
    )

    for column in (
        "annotation_id",
        "annotation_format",
        "event_time_unit",
        "time_reference",
    ):
        manifest[column] = manifest[column].astype(str).str.strip()

    enabled_text = manifest["enabled"].astype(str).str.strip().str.lower()
    enabled_map = {
        "true": True, "1": True, "yes": True,
        "false": False, "0": False, "no": False,
    }
    invalid_enabled = sorted(set(enabled_text) - set(enabled_map))
    if invalid_enabled:
        raise ValueError(f"Invalid enabled values: {invalid_enabled}")
    manifest["enabled"] = enabled_text.map(enabled_map)
    manifest = manifest.loc[manifest["enabled"]].reset_index(drop=True)

    return manifest


def validate_recording_manifest(
    manifest,
):
    """Validate manifest values and file paths."""

    errors = []

    valid_proteins = {
        "HSA",
        "F8",
        "MIXTURE",
    }

    invalid_proteins = (
        set(manifest["protein"])
        - valid_proteins
    )

    if invalid_proteins:
        errors.append(
            "Invalid protein labels: "
            f"{sorted(invalid_proteins)}"
        )

    duplicated_annotation_ids = (
        manifest.loc[
            manifest["annotation_id"]
            .duplicated(keep=False),
            "annotation_id",
        ]
        .unique()
        .tolist()
    )

    if duplicated_annotation_ids:
        errors.append(
            "Duplicated annotation_id values: "
            f"{duplicated_annotation_ids}"
        )

    for row in manifest.itertuples(
        index=False
    ):
        annotation_path = Path(row.annotation_path)
        abf_path = Path(row.abf_path)

        if not row.recording_id:
            errors.append(
                "Empty recording_id found"
            )

        if not row.annotation_id:
            errors.append("Empty annotation_id found")

        if not annotation_path.is_file():
            errors.append(
                f"Missing annotation for "
                f"{row.recording_id}: "
                f"{annotation_path}"
            )

        if not abf_path.is_file():
            errors.append(
                f"Missing ABF for "
                f"{row.recording_id}: "
                f"{abf_path}"
            )

        if row.sweep_index < 0:
            errors.append(
                f"Negative sweep_index for "
                f"{row.recording_id}"
            )

        if not np.isfinite(row.time_offset_s):
            errors.append(
                f"Nonfinite time_offset_s for {row.recording_id}"
            )

        if row.annotation_format.lower() not in {
            "json", "csv", "excel", "xlsx", "xls"
        }:
            errors.append(
                f"Unsupported annotation_format for {row.annotation_id}: "
                f"{row.annotation_format}"
            )

        if row.event_time_unit.lower() not in {"s", "ms"}:
            errors.append(
                f"Unsupported event_time_unit for {row.annotation_id}: "
                f"{row.event_time_unit}"
            )

        if row.time_reference.lower() != "recording_start":
            errors.append(
                f"Unsupported time_reference for {row.annotation_id}: "
                f"{row.time_reference}"
            )

    for recording_id, group in manifest.groupby("recording_id", sort=False):
        consistency = group[["protein", "abf_path", "sweep_index"]].drop_duplicates()
        if len(consistency) > 1:
            errors.append(
                "Inconsistent physical recording mapping for "
                f"recording_id {recording_id}"
            )

    if errors:
        raise ValueError(
            "Manifest validation failed:\n"
            + "\n".join(errors)
        )


# =========================================================
# Annotation and ABF loaders
# =========================================================

def load_json_events(
    json_path,
):
    """Load pre-annotated events from one JSON file."""

    json_path = Path(json_path)

    try:
        with json_path.open(
            "r",
            encoding="utf-8",
        ) as file:
            data = json.load(file)

    except json.JSONDecodeError as error:
        raise ValueError(
            f"Invalid JSON in {json_path}: "
            f"{error}"
        ) from error

    if not isinstance(data, dict):
        raise ValueError(
            f"Invalid JSON structure: {json_path}"
        )

    if not isinstance(
        data.get("events"),
        list,
    ):
        raise ValueError(
            f"Missing events list: {json_path}"
        )

    return data["events"]


def _normalize_tabular_events(rows, fieldnames, event_time_unit, source_format):
    """Normalize CSV/Excel rows to the shared event annotation schema."""

    unit = str(event_time_unit).strip().lower()
    factors = {"s": 1.0, "ms": 0.001}
    if unit not in factors:
        raise ValueError(f"Unsupported event_time_unit: {event_time_unit}")

    required = {"Event start", "Event end"}
    available_fields = [] if fieldnames is None else list(fieldnames)
    missing = sorted(required - set(available_fields))
    if missing:
        raise ValueError(f"{source_format.upper()} annotation is missing columns: {missing}")

    events = []
    for source_row_number, row in enumerate(rows, start=2):
        normalized_row = {
            str(key).strip().casefold(): value
            for key, value in row.items()
        }

        def source_number(column):
            value = str(
                normalized_row.get(str(column).strip().casefold()) or ""
            ).strip()
            try:
                number = float(value)
            except ValueError:
                return np.nan
            return number if np.isfinite(number) else np.nan

        raw_start = str(row.get("Event start") or "").strip()
        raw_end = str(row.get("Event end") or "").strip()
        source_dwell_time_ms = source_number("Dwell time(ms)")
        reason = ""
        start = end = np.nan
        boundary_reconstructed = False
        boundary_reconstruction_method = ""

        if not raw_start:
            reason = "missing_event_start"
        elif not raw_end:
            reason = "missing_event_end"
        else:
            try:
                start = float(raw_start)
                end = float(raw_end)
            except ValueError:
                reason = "nonnumeric_event_time"
            else:
                if not np.isfinite(start) or not np.isfinite(end):
                    reason = "nonfinite_event_time"
                elif end < start:
                    reason = "nonpositive_event_duration"
                elif end == start:
                    if source_dwell_time_ms > 0:
                        dwell_in_source_unit = (
                            source_dwell_time_ms
                            if unit == "ms"
                            else source_dwell_time_ms / 1000.0
                        )
                        half_dwell = dwell_in_source_unit / 2.0
                        start -= half_dwell
                        end += half_dwell
                        boundary_reconstructed = True
                        boundary_reconstruction_method = (
                            "centered_equal_boundary_using_source_dwell_time"
                        )
                    else:
                        reason = "nonpositive_event_duration"

        events.append({
            "eventStartTime": start * factors[unit],
            "eventEndTime": end * factors[unit],
            "blockagePercent": source_number("%blockage"),
            "_annotation_format": source_format,
            "_load_qc_reason": reason,
            "_source_row_number": source_row_number,
            "_source_event_start": raw_start,
            "_source_event_end": raw_end,
            "_source_event_time_unit": unit,
            "_event_boundary_reconstructed": boundary_reconstructed,
            "_boundary_reconstruction_method": boundary_reconstruction_method,
            "_source_amplitude": source_number("Amplitude"),
            "_source_amplitude_sd": source_number("Amp S.D."),
            "_source_dwell_time_ms": source_dwell_time_ms,
            "_source_blockage_percent": source_number("%blockage"),
            "_source_baseline": source_number("Level 0"),
        })

    return events


def load_csv_events(csv_path, event_time_unit="ms"):
    """Load CSV event annotations and normalize times to seconds."""

    csv_path = Path(csv_path)
    with csv_path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        return _normalize_tabular_events(
            reader, reader.fieldnames, event_time_unit, "csv"
        )


def load_excel_events(excel_path, event_time_unit="ms"):
    """Load XLSX/XLS event annotations and normalize times to seconds."""

    excel_path = Path(excel_path)
    table = pd.read_excel(excel_path, dtype=str, keep_default_na=False)
    return _normalize_tabular_events(
        table.to_dict(orient="records"), table.columns, event_time_unit, "excel"
    )


def load_annotation_events(row):
    """Dispatch an annotation manifest row to its source-specific loader."""

    annotation_format = str(row.annotation_format).strip().lower()
    if annotation_format == "json":
        events = load_json_events(row.annotation_path)
        for source_row_number, event in enumerate(events):
            event.setdefault("_annotation_format", "json")
            event.setdefault("_load_qc_reason", "")
            event.setdefault("_source_row_number", source_row_number)
        return events
    if annotation_format == "csv":
        return load_csv_events(row.annotation_path, row.event_time_unit)
    if annotation_format in {"excel", "xlsx", "xls"}:
        return load_excel_events(row.annotation_path, row.event_time_unit)
    raise ValueError(f"Unsupported annotation_format: {row.annotation_format}")


def load_abf_signal(
    abf_path,
    sweep_index=0,
):
    """Load one sweep from an ABF recording."""

    abf_path = Path(abf_path)

    abf = pyabf.ABF(
        str(abf_path)
    )

    if sweep_index < 0:
        raise ValueError(
            "sweep_index cannot be negative"
        )

    if sweep_index >= abf.sweepCount:
        raise ValueError(
            f"Invalid sweep {sweep_index} "
            f"for {abf_path}. "
            f"ABF contains "
            f"{abf.sweepCount} sweeps."
        )

    abf.setSweep(
        sweep_index
    )

    time_s = np.asarray(
        abf.sweepX,
        dtype=float,
    )

    signal = np.asarray(
        abf.sweepY,
        dtype=float,
    )

    if time_s.ndim != 1:
        raise ValueError(
            "ABF time must be one-dimensional"
        )

    if signal.ndim != 1:
        raise ValueError(
            "ABF signal must be one-dimensional"
        )

    if len(time_s) != len(signal):
        raise ValueError(
            "ABF time and signal lengths differ"
        )

    if len(time_s) < 3:
        raise ValueError(
            "ABF sweep contains too few samples"
        )

    if not np.all(
        np.isfinite(time_s)
    ):
        raise ValueError(
            "ABF timestamps contain "
            "NaN or infinity"
        )

    if not np.all(
        np.isfinite(signal)
    ):
        raise ValueError(
            "ABF signal contains "
            "NaN or infinity"
        )

    # Calculate once per ABF/sweep.
    time_differences_s = np.diff(
        time_s
    )

    if np.any(
        time_differences_s <= 0
    ):
        raise ValueError(
            "ABF timestamps are not "
            "strictly increasing"
        )

    sampling_period_s = float(
        np.median(
            time_differences_s
        )
    )

    if (
        not np.isfinite(
            sampling_period_s
        )
        or sampling_period_s <= 0
    ):
        raise ValueError(
            "Invalid ABF sampling period"
        )

    sampling_rate_hz = (
        1.0 / sampling_period_s
    )

    return {
        "time_s":
            time_s,

        "signal":
            signal,

        "sampling_period_s":
            sampling_period_s,

        "sampling_rate_hz":
            sampling_rate_hz,

        "sweep_count":
            int(abf.sweepCount),
        "signal_unit":
            str(abf.sweepUnitsY),
    }

# =========================================================
# Event positioning and baseline extraction
# =========================================================

def get_event_indices(
    time_s,
    start_time_s,
    end_time_s,
):
    """
    Convert annotation event times into ABF sample indices.
    """

    start_time_s = float(
        start_time_s
    )

    end_time_s = float(
        end_time_s
    )

    if not np.isfinite(start_time_s):
        raise ValueError(
            "eventStartTime is not finite"
        )

    if not np.isfinite(end_time_s):
        raise ValueError(
            "eventEndTime is not finite"
        )

    if end_time_s <= start_time_s:
        raise ValueError(
            "eventEndTime must be greater "
            "than eventStartTime"
        )

    if start_time_s > time_s[-1]:
        raise ValueError(
            "Event starts after ABF sweep ends"
        )

    if end_time_s < time_s[0]:
        raise ValueError(
            "Event ends before ABF sweep starts"
        )

    start_index = int(
        np.searchsorted(
            time_s,
            start_time_s,
            side="left",
        )
    )

    end_index = int(
        np.searchsorted(
            time_s,
            end_time_s,
            side="right",
        )
    )

    start_index = max(
        start_index,
        0,
    )

    end_index = min(
        end_index,
        len(time_s),
    )

    if (
        end_index - start_index
        < MIN_EVENT_POINTS
    ):
        raise ValueError(
            "Event contains fewer than "
            f"{MIN_EVENT_POINTS} samples"
        )

    return start_index, end_index


def extract_event_and_baseline(
    time_s,
    signal,
    start_time_s,
    end_time_s,
    baseline_points,
    prev_event_end_index=0,
    next_event_start_index=None,
    shape_context_points=0,
):
    """
    Extract an event and its local pre-event baseline.

    The baseline cannot overlap the preceding
    annotated event.
    """

    start_index, end_index = (
        get_event_indices(
            time_s=time_s,
            start_time_s=start_time_s,
            end_time_s=end_time_s,
        )
    )

    requested_baseline_start = (
        start_index - baseline_points
    )

    baseline_start_index = max(
        requested_baseline_start,
        prev_event_end_index,
        0,
    )

    baseline_end_index = start_index

    baseline_time_s = time_s[
        baseline_start_index:
        baseline_end_index
    ]

    baseline_signal = signal[
        baseline_start_index:
        baseline_end_index
    ]

    event_time_s = time_s[
        start_index:end_index
    ]

    event_signal = signal[
        start_index:end_index
    ]

    shape_start_index = max(
        start_index - int(shape_context_points),
        prev_event_end_index,
        0,
    )
    right_limit = (
        len(time_s)
        if next_event_start_index is None
        else max(int(next_event_start_index), end_index)
    )
    shape_end_index = min(
        end_index + int(shape_context_points),
        right_limit,
        len(time_s),
    )
    shape_time_s = time_s[shape_start_index:shape_end_index]
    shape_signal = signal[shape_start_index:shape_end_index]
    shape_baseline_end_index = shape_start_index
    shape_baseline_start_index = max(
        shape_baseline_end_index - baseline_points,
        prev_event_end_index,
        0,
    )
    shape_baseline_signal = signal[
        shape_baseline_start_index:shape_baseline_end_index
    ]

    if (
        len(baseline_signal)
        < MIN_BASELINE_POINTS
    ):
        raise ValueError(
            "Insufficient baseline after "
            "excluding the preceding event"
        )

    if (
        len(event_signal)
        < MIN_EVENT_POINTS
    ):
        raise ValueError(
            "Event signal is too short"
        )

    return {
        "event_time_s":
            event_time_s,
        "event_signal":
            event_signal,
        "shape_time_s": shape_time_s,
        "shape_signal": shape_signal,
        "shape_baseline_signal": shape_baseline_signal,
        "shape_start_index": shape_start_index,
        "shape_end_index": shape_end_index,
        "shape_event_start_index": start_index - shape_start_index,
        "shape_event_end_index": end_index - shape_start_index,
        "actual_shape_context_pre_points": start_index - shape_start_index,
        "actual_shape_context_post_points": shape_end_index - end_index,
        "actual_shape_baseline_points": len(shape_baseline_signal),
        "baseline_time_s":
            baseline_time_s,
        "baseline_signal":
            baseline_signal,
        "start_index":
            start_index,
        "end_index":
            end_index,
        "baseline_start_index":
            baseline_start_index,
        "baseline_end_index":
            baseline_end_index,
        "requested_baseline_points":
            baseline_points,
        "actual_baseline_points":
            len(baseline_signal),
    }


# =========================================================
# Feature calculation bridge
# =========================================================

def process_single_event(
    event,
    time_s,
    signal,
    baseline_points,
    prev_event_end_index=0,
    time_offset_s=0.0,
    next_event_start_index=None,
    shape_context_points=0,
):
    """Extract one event and delegate numerical features."""

    required_fields = [
        "eventStartTime",
        "eventEndTime",
    ]

    missing_fields = [
        field
        for field in required_fields
        if field not in event
    ]

    if missing_fields:
        raise ValueError(
            f"Event missing fields: {missing_fields}"
        )

    annotation_start_time_s = float(
        event["eventStartTime"]
    )
    annotation_end_time_s = float(
        event["eventEndTime"]
    )
    time_offset_s = float(time_offset_s)
    start_time_s = apply_time_offset(annotation_start_time_s, time_offset_s)
    end_time_s = apply_time_offset(annotation_end_time_s, time_offset_s)

    segments = extract_event_and_baseline(
        time_s=time_s,
        signal=signal,
        start_time_s=start_time_s,
        end_time_s=end_time_s,
        baseline_points=baseline_points,
        prev_event_end_index=(
            prev_event_end_index
        ),
        next_event_start_index=next_event_start_index,
        shape_context_points=shape_context_points,
    )

    features = calculate_event_features(
        event_time_s=segments["event_time_s"],
        event_signal=segments["event_signal"],
        baseline_signal=(
            segments["baseline_signal"]
        ),
        start_time_s=start_time_s,
        end_time_s=end_time_s,
        annotated_blockage_percent=event.get(
            "blockagePercent",
            np.nan,
        ),
        shape_time_s=segments["shape_time_s"],
        shape_signal=segments["shape_signal"],
        shape_baseline_signal=segments["shape_baseline_signal"],
        shape_event_start_index=segments["shape_event_start_index"],
        shape_event_end_index=segments["shape_event_end_index"],
    )

    return {
        "event_start_time_s": annotation_start_time_s,
        "event_end_time_s": annotation_end_time_s,
        "annotation_event_start_time_s": annotation_start_time_s,
        "annotation_event_end_time_s": annotation_end_time_s,
        "json_event_start_time_s": (
            annotation_start_time_s
            if event.get("_annotation_format", "json") == "json"
            else np.nan
        ),
        "json_event_end_time_s": (
            annotation_end_time_s
            if event.get("_annotation_format", "json") == "json"
            else np.nan
        ),
        "time_offset_s": time_offset_s,
        "abf_event_start_time_s": start_time_s,
        "abf_event_end_time_s": end_time_s,
        **features,
        "requested_baseline_points": segments[
            "requested_baseline_points"
        ],
        "actual_baseline_points": segments[
            "actual_baseline_points"
        ],
        "event_start_index": segments[
            "start_index"
        ],
        "event_end_index": segments[
            "end_index"
        ],
        "shape_start_index": segments["shape_start_index"],
        "shape_end_index": segments["shape_end_index"],
        "actual_shape_context_pre_points": segments[
            "actual_shape_context_pre_points"
        ],
        "actual_shape_context_post_points": segments[
            "actual_shape_context_post_points"
        ],
        "actual_shape_baseline_points": segments[
            "actual_shape_baseline_points"
        ],
    }


# =========================================================
# Collect events sharing one ABF/sweep
# =========================================================

def collect_events_for_abf_group(
    manifest_group,
):
    """
    Collect events from every annotation referencing the
    same ABF and sweep, then sort chronologically.
    """

    collected_events = []

    for row in manifest_group.itertuples(
        index=False
    ):
        events = load_annotation_events(row)

        for source_event_id, event in enumerate(
            events
        ):
            collected_events.append({
                "protein":
                    row.protein,

                "recording_id":
                    row.recording_id,

                "annotation_id":
                    row.annotation_id,

                "annotation_path":
                    row.annotation_path,

                "annotation_format":
                    row.annotation_format,

                "json_path":
                    row.json_path,

                "source_event_id":
                    source_event_id,

                "event":
                    event,

                "time_offset_s":
                    float(row.time_offset_s),

                "experiment_metadata": {
                    "concentration_nM": getattr(row, "concentration_nM", np.nan),
                    "mixture_id": getattr(row, "mixture_id", ""),
                    "hsa_to_f8_ratio": getattr(row, "hsa_to_f8_ratio", ""),
                    "hsa_fraction": getattr(row, "hsa_fraction", np.nan),
                    "f8_fraction": getattr(row, "f8_fraction", np.nan),
                    "voltage_mV": getattr(row, "voltage_mV", np.nan),
                    "pore_id": getattr(row, "pore_id", ""),
                },
            })

    def event_start_time(item):
        event = item["event"]

        try:
            return apply_time_offset(
                event["eventStartTime"], item["time_offset_s"]
            )
        except (
            KeyError,
            TypeError,
            ValueError,
        ):
            return np.inf

    collected_events.sort(
        key=event_start_time
    )

    return collected_events


# =========================================================
# Process one ABF/sweep group
# =========================================================

def process_abf_group(
    abf_path,
    sweep_index,
    manifest_group,
    baseline_window_ms,
    shape_context_window_ms=1.0,
    alignment_sample_events=3,
    alignment_minimum_correlation=0.999,
    alignment_maximum_normalized_rmse=0.01,
    raw_waveform_output_directory=None,
    raw_waveform_group_number=None,
):
    """
    Process every annotation associated
    with one ABF/sweep.
    """

    print("  Loading ABF...")

    abf_data = load_abf_signal(
        abf_path=abf_path,
        sweep_index=sweep_index,
    )

    time_s = abf_data["time_s"]
    signal = abf_data["signal"]

    sampling_period_s = (
        abf_data["sampling_period_s"]
    )

    sampling_rate_hz = (
        abf_data["sampling_rate_hz"]
    )

    # Convert baseline duration from milliseconds
    # to samples once per ABF/sweep.
    baseline_points = max(
        int(
            round(
                baseline_window_ms
                / 1000.0
                / sampling_period_s
            )
        ),
        MIN_BASELINE_POINTS,
    )
    shape_context_points = max(
        int(round(shape_context_window_ms / 1000.0 / sampling_period_s)),
        0,
    )

    print(
        "  ABF loaded:",
        len(signal),
        "samples",
    )

    print(
        "  Sampling rate:",
        f"{sampling_rate_hz:.2f}",
        "Hz",
    )

    print(
        "  Baseline window:",
        baseline_window_ms,
        "ms",
    )

    print(
        "  Baseline points:",
        baseline_points,
    )

    print(
        "  Loading annotation events..."
    )

    alignment_rows = []
    for row in manifest_group.itertuples(index=False):
        alignment_events = load_annotation_events(row)
        if row.annotation_format.lower() == "json":
            recording_rows = validate_recording_alignment(
                events=alignment_events,
                abf_time_s=time_s,
                abf_signal=signal,
                time_offset_s=float(row.time_offset_s),
                recording_id=row.recording_id,
                sample_count=alignment_sample_events,
                minimum_correlation=alignment_minimum_correlation,
                maximum_normalized_rmse=alignment_maximum_normalized_rmse,
            )
        else:
            valid_events = [
                event for event in alignment_events
                if not event.get("_load_qc_reason")
            ]
            recording_rows = []
            for source_index in np.linspace(
                0,
                max(len(valid_events) - 1, 0),
                min(alignment_sample_events, len(valid_events)),
                dtype=int,
            ) if valid_events else []:
                event = valid_events[int(source_index)]
                start_s = apply_time_offset(
                    event["eventStartTime"], row.time_offset_s
                )
                end_s = apply_time_offset(
                    event["eventEndTime"], row.time_offset_s
                )
                in_bounds = bool(
                    time_s[0] <= start_s < end_s <= time_s[-1] + sampling_period_s
                )
                recording_rows.append({
                    "recording_id": row.recording_id,
                    "source_event_id": int(source_index),
                    "source_row_number": event.get("_source_row_number"),
                    "alignment_method": "boundary_and_duration_only",
                    "annotation_event_start_s": float(event["eventStartTime"]),
                    "annotation_event_end_s": float(event["eventEndTime"]),
                    "event_boundary_reconstructed": bool(
                        event.get("_event_boundary_reconstructed", False)
                    ),
                    "boundary_reconstruction_method": event.get(
                        "_boundary_reconstruction_method", ""
                    ),
                    "configured_time_offset_s": float(row.time_offset_s),
                    "alignment_pass": in_bounds,
                })
                if not in_bounds:
                    raise ValueError(
                        f"CSV/ABF boundary validation failed for "
                        f"{row.recording_id}, row {event.get('_source_row_number')}"
                    )
        for alignment_row in recording_rows:
            alignment_rows.append({
                "protein": row.protein,
                "annotation_id": row.annotation_id,
                "annotation_path": str(row.annotation_path),
                "annotation_format": row.annotation_format,
                "json_path": str(row.json_path),
                "abf_path": str(row.abf_path),
                "sweep_index": int(row.sweep_index),
                **alignment_row,
            })
        print(
            f"  Annotation validation completed: {row.recording_id}; "
            f"offset={float(row.time_offset_s):.8f} s; "
            f"sampled_events={len(recording_rows)}"
        )

    collected_events = (
        collect_events_for_abf_group(
            manifest_group
        )
    )

    total_events = len(
        collected_events
    )

    print(
        "  Annotation rows loaded:",
        total_events,
    )

    feature_rows = []
    qc_rows = []

    prev_event_end_index = 0

    # Detect repeated event boundaries from
    # multiple annotation files using the same ABF.
    seen_event_boundaries = {}

    for event_number, item in enumerate(
        collected_events,
        start=1,
    ):
        if (
            event_number == 1
            or event_number % 100 == 0
            or event_number == total_events
        ):
            print(
                f"  Event "
                f"{event_number}/{total_events}"
            )

        protein = item["protein"]

        recording_id = (
            item["recording_id"]
        )

        json_path = item["json_path"]
        annotation_id = item["annotation_id"]
        annotation_path = item["annotation_path"]
        annotation_format = item["annotation_format"]

        source_event_id = (
            item["source_event_id"]
        )

        event = item["event"]
        time_offset_s = float(item["time_offset_s"])
        experiment_metadata = item["experiment_metadata"]

        metadata = {
            "protein":
                protein,

            "recording_id":
                recording_id,

            "annotation_id":
                annotation_id,

            "annotation_path":
                str(annotation_path),

            "annotation_format":
                annotation_format,

            "json_path":
                str(json_path),

            "abf_path":
                str(abf_path),

            "sweep_index":
                sweep_index,

            "event_id":
                f"{annotation_id}:{source_event_id}",

            "source_event_id":
                source_event_id,

            "source_row_number":
                event.get("_source_row_number"),

            "source_event_start":
                event.get("_source_event_start", event.get("eventStartTime")),

            "source_event_end":
                event.get("_source_event_end", event.get("eventEndTime")),

            "source_event_time_unit":
                event.get("_source_event_time_unit", "s"),

            "event_boundary_reconstructed":
                bool(event.get("_event_boundary_reconstructed", False)),

            "boundary_reconstruction_method":
                event.get("_boundary_reconstruction_method", ""),

            "source_amplitude":
                event.get("_source_amplitude", np.nan),

            "source_amplitude_sd":
                event.get("_source_amplitude_sd", np.nan),

            "source_dwell_time_ms":
                event.get("_source_dwell_time_ms", np.nan),

            "source_blockage_percent":
                event.get("_source_blockage_percent", event.get("blockagePercent", np.nan)),

            "source_baseline":
                event.get("_source_baseline", np.nan),

            **experiment_metadata,

            "time_offset_s":
                time_offset_s,
        }

        if event.get("_load_qc_reason"):
            qc_rows.append({
                **metadata,
                "qc_pass": False,
                "qc_reason": event["_load_qc_reason"],
            })
            continue

        # Convert annotated event times into
        # ABF indices.
        try:
            json_start_time_s = float(
                event["eventStartTime"]
            )

            json_end_time_s = float(
                event["eventEndTime"]
            )
            start_time_s = apply_time_offset(json_start_time_s, time_offset_s)
            end_time_s = apply_time_offset(json_end_time_s, time_offset_s)

            start_index, end_index = (
                get_event_indices(
                    time_s=time_s,
                    start_time_s=start_time_s,
                    end_time_s=end_time_s,
                )
            )

            next_event_start_index = None
            # Duplicate annotations describe this same physical interval and
            # must not truncate its recovery context. Invalid times cannot
            # define a neighbouring interval; ignored valid events still do.
            current_boundary = (round(start_time_s, 12), round(end_time_s, 12))
            for next_position in range(event_number, total_events):
                next_item = collected_events[next_position]
                if next_item["event"].get("_load_qc_reason"):
                    continue
                try:
                    next_start_time_s = apply_time_offset(
                        next_item["event"]["eventStartTime"],
                        float(next_item["time_offset_s"]),
                    )
                    next_end_time_s = apply_time_offset(
                        next_item["event"]["eventEndTime"],
                        float(next_item["time_offset_s"]),
                    )
                    if not (np.isfinite(next_start_time_s)
                            and np.isfinite(next_end_time_s)
                            and next_end_time_s > next_start_time_s):
                        continue
                    if (round(next_start_time_s, 12), round(next_end_time_s, 12)) == current_boundary:
                        continue
                    next_event_start_index = int(np.searchsorted(
                        time_s, next_start_time_s, side="left"
                    ))
                except (KeyError, TypeError, ValueError):
                    continue
                break

        except (
            KeyError,
            TypeError,
            ValueError,
        ) as error:
            qc_rows.append({
                **metadata,

                "qc_pass":
                    False,

                "qc_reason":
                    str(error),
            })
            continue

        # Use rounded times to avoid insignificant
        # floating-point representation differences.
        event_boundary_key = (
            round(start_time_s, 12),
            round(end_time_s, 12),
        )

        if (
            event_boundary_key
            in seen_event_boundaries
        ):
            previous = seen_event_boundaries[
                event_boundary_key
            ]

            if previous["protein"] != protein:
                raise ValueError(
                    "Conflicting protein labels for the same "
                    "ABF/sweep event boundary: "
                    f"{previous['protein']} vs {protein}; "
                    f"boundary={event_boundary_key}; "
                    f"ABF={abf_path}; sweep={sweep_index}"
                )

            qc_rows.append({
                **metadata,

                "qc_pass":
                    False,

                "qc_reason":
                    "duplicate_annotation",
            })
            continue

        seen_event_boundaries[event_boundary_key] = {
            "protein": protein,
            "recording_id": recording_id,
            "annotation_path": str(annotation_path),
        }

        # Skip manually ignored events, but prevent
        # the ignored interval from being used as the
        # baseline for the following event.
        if bool(
            event.get(
                "ignored",
                False,
            )
        ):
            qc_rows.append({
                **metadata,

                "qc_pass":
                    False,

                "qc_reason":
                    "ignored",
            })

            prev_event_end_index = max(
                prev_event_end_index,
                end_index,
            )

            continue

        try:
            # Important: pass baseline_points,
            # not baseline_window_ms.
            features = process_single_event(
                event=event,
                time_s=time_s,
                signal=signal,
                baseline_points=(
                    baseline_points
                ),
                prev_event_end_index=(
                    prev_event_end_index
                ),
                time_offset_s=time_offset_s,
                next_event_start_index=next_event_start_index,
                shape_context_points=shape_context_points,
            )

            feature_rows.append({
                **metadata,

                "sampling_rate_hz":
                    sampling_rate_hz,

                "sweep_count":
                    abf_data[
                        "sweep_count"
                    ],

                **features,
            })

            qc_rows.append({
                **metadata,

                "qc_pass":
                    True,

                "qc_reason":
                    "",
            })

        except (
            KeyError,
            TypeError,
            ValueError,
        ) as error:
            qc_rows.append({
                **metadata,

                "qc_pass":
                    False,

                "qc_reason":
                    str(error),
            })

        finally:
            # Do not allow an annotated event interval
            # to become baseline for the next event.
            prev_event_end_index = max(
                prev_event_end_index,
                end_index,
            )

    feature_df = pd.DataFrame(
        feature_rows
    )

    qc_df = pd.DataFrame(
        qc_rows
    )

    if raw_waveform_output_directory is not None and not feature_df.empty:
        save_loaded_abf_event_waveforms(
            feature_df,
            abf_data,
            raw_waveform_output_directory,
            raw_waveform_group_number,
        )

    print(
        "  Accepted events:",
        len(feature_df),
    )

    if not qc_df.empty:
        failed_count = int(
            (~qc_df["qc_pass"]).sum()
        )

        print(
            "  Rejected/ignored events:",
            failed_count,
        )

    return (
        feature_df,
        qc_df,
        pd.DataFrame(alignment_rows),
    )

# =========================================================
# Process all ABF/sweep groups
# =========================================================

def process_all_recordings(
    manifest,
    baseline_window_ms,
    shape_context_window_ms=1.0,
    alignment_sample_events=3,
    alignment_minimum_correlation=0.999,
    alignment_maximum_normalized_rmse=0.01,
    raw_waveform_output_directory=None,
):
    """Process every ABF/sweep group in manifest."""

    feature_frames = []
    qc_frames = []
    alignment_frames = []

    group_columns = [
        "abf_path",
        "sweep_index",
    ]

    grouped_manifest = manifest.groupby(
        group_columns,
        sort=False,
        dropna=False,
    )

    for group_number, (
        (abf_path, sweep_index),
        manifest_group,
    ) in enumerate(grouped_manifest):
        print(
            "\nProcessing:"
            f"\n  ABF: {abf_path}"
            f"\n  sweep: {sweep_index}"
            f"\n  Annotation files: "
            f"{len(manifest_group)}"
        )

        features, qc, alignment = process_abf_group(
            abf_path=abf_path,
            sweep_index=int(
                sweep_index
            ),
            manifest_group=(
                manifest_group
            ),
            baseline_window_ms=(
                baseline_window_ms
            ),
            shape_context_window_ms=shape_context_window_ms,
            alignment_sample_events=alignment_sample_events,
            alignment_minimum_correlation=alignment_minimum_correlation,
            alignment_maximum_normalized_rmse=alignment_maximum_normalized_rmse,
            raw_waveform_output_directory=raw_waveform_output_directory,
            raw_waveform_group_number=group_number,
        )

        if not features.empty:
            feature_frames.append(
                features
            )

        if not qc.empty:
            qc_frames.append(qc)

        if not alignment.empty:
            alignment_frames.append(alignment)

    if not feature_frames:
        raise RuntimeError(
            "No events passed feature extraction"
        )

    combined_features = pd.concat(
        feature_frames,
        ignore_index=True,
    )

    if qc_frames:
        combined_qc = pd.concat(
            qc_frames,
            ignore_index=True,
        )
    else:
        combined_qc = pd.DataFrame()

    combined_alignment = pd.concat(
        alignment_frames, ignore_index=True
    ) if alignment_frames else pd.DataFrame()

    return combined_features, combined_qc, combined_alignment

# =========================================================
# Save output
# =========================================================

def build_recording_qc_summary(features, qc):
    """Summarize extraction and shape QC for every recording."""

    keys = ["protein", "recording_id"]
    groups = set()

    if not features.empty:
        groups.update(
            map(tuple, features[keys].drop_duplicates().to_numpy())
        )

    if not qc.empty:
        groups.update(
            map(tuple, qc[keys].drop_duplicates().to_numpy())
        )

    rows = []

    for protein, recording_id in sorted(groups):
        feature_group = features[
            (features["protein"] == protein)
            & (features["recording_id"] == recording_id)
        ]
        qc_group = qc[
            (qc["protein"] == protein)
            & (qc["recording_id"] == recording_id)
        ]

        reasons = qc_group.loc[
            ~qc_group["qc_pass"],
            "qc_reason",
        ]
        n_annotated = len(qc_group)
        n_accepted = int(qc_group["qc_pass"].sum())
        n_ignored = int((reasons == "ignored").sum())
        n_duplicate = int(
            reasons.isin([
                "duplicate_annotation",
                "duplicate_annotation_label_conflict",
            ]).sum()
        )
        n_extraction_failed = int(
            (~qc_group["qc_pass"]).sum()
            - n_ignored
            - n_duplicate
        )

        if "shape_qc_pass" in feature_group:
            shape_pass_count = int(
                feature_group["shape_qc_pass"].fillna(False).sum()
            )
        else:
            shape_pass_count = 0

        peak_snr = pd.to_numeric(
            feature_group.get("peak_snr", pd.Series(dtype=float)),
            errors="coerce",
        ).replace([np.inf, -np.inf], np.nan)

        rows.append({
            "protein": protein,
            "recording_id": recording_id,
            "n_annotated": n_annotated,
            "n_ignored": n_ignored,
            "n_duplicate": n_duplicate,
            "n_extraction_failed": n_extraction_failed,
            "n_accepted": n_accepted,
            "acceptance_rate": (
                n_accepted / n_annotated
                if n_annotated
                else np.nan
            ),
            "shape_qc_pass_count": shape_pass_count,
            "shape_qc_pass_rate": (
                shape_pass_count / len(feature_group)
                if len(feature_group)
                else np.nan
            ),
            "median_peak_snr": (
                float(peak_snr.median())
                if not peak_snr.dropna().empty
                else np.nan
            ),
        })

    return pd.DataFrame(rows)


def save_loaded_abf_event_waveforms(
    features,
    abf_data,
    output_directory,
    group_number,
):
    """Save event slices while their source ABF sweep is already loaded."""

    output_directory = Path(output_directory)
    recording_ids = features["recording_id"].astype(str).unique()
    if len(recording_ids) != 1:
        raise ValueError(
            "Each raw waveform archive must contain exactly one recording_id"
        )
    recording_id = recording_ids[0]
    if Path(recording_id).name != recording_id or recording_id in {".", ".."}:
        raise ValueError(f"Unsafe recording_id for output path: {recording_id}")
    recording_directory = output_directory / "recordings" / recording_id
    archive_directory = recording_directory / "raw_event_waveforms"
    archive_directory.mkdir(parents=True, exist_ok=True)
    archive_name = f"waveform_group_{int(group_number):03d}.npz"
    archive_path = archive_directory / archive_name
    if archive_path.exists():
        raise FileExistsError(f"Raw waveform archive already exists: {archive_path}")

    full_signal = abf_data["signal"]
    full_time_s = abf_data["time_s"]
    signal_parts, time_parts, offsets, index_rows = [], [], [0], []
    starts, ends, baselines = [], [], []
    for waveform_index, row in enumerate(features.itertuples(index=False)):
        start, stop = int(row.event_start_index), int(row.event_end_index)
        if not (0 <= start < stop <= len(full_signal)):
            raise ValueError(f"Invalid waveform indices for {row.event_id}")
        signal = np.asarray(full_signal[start:stop], dtype=np.float64)
        time_s = np.asarray(full_time_s[start:stop], dtype=np.float64)
        if len(signal) != int(row.n_event_points) or len(signal) != len(time_s):
            raise ValueError(f"Waveform length mismatch for {row.event_id}")
        if not np.all(np.isfinite(signal)) or not np.all(np.isfinite(time_s)):
            raise ValueError(f"Nonfinite raw waveform for {row.event_id}")
        value_start = offsets[-1]
        value_stop = value_start + len(signal)
        signal_parts.append(signal)
        time_parts.append(time_s)
        offsets.append(value_stop)
        starts.append(float(row.abf_event_start_time_s))
        ends.append(float(row.abf_event_end_time_s))
        baselines.append(float(row.baseline_mean))
        index_rows.append({
            "archive_file": str(
                Path("recordings") / recording_id / "raw_event_waveforms" / archive_name
            ),
            "waveform_index_in_archive": waveform_index,
            "protein": row.protein,
            "recording_id": row.recording_id,
            "annotation_id": row.annotation_id,
            "event_id": row.event_id,
            "abf_path": row.abf_path,
            "sweep_index": int(row.sweep_index),
            "abf_event_start_index": start,
            "abf_event_end_index": stop,
            "waveform_value_start": value_start,
            "waveform_value_stop": value_stop,
            "n_samples": len(signal),
            "sampling_rate_hz": float(row.sampling_rate_hz),
            "signal_unit": abf_data["signal_unit"],
        })

    raw_signal = np.concatenate(signal_parts)
    raw_time_s = np.concatenate(time_parts)
    offsets_array = np.asarray(offsets, dtype=np.int64)
    np.savez(
        archive_path,
        raw_signal=raw_signal,
        raw_time_s=raw_time_s,
        offsets=offsets_array,
        event_start_time_s=np.asarray(starts, dtype=np.float64),
        event_end_time_s=np.asarray(ends, dtype=np.float64),
        baseline_mean=np.asarray(baselines, dtype=np.float64),
    )
    with np.load(archive_path, allow_pickle=False) as saved:
        if not np.array_equal(saved["raw_signal"], raw_signal):
            raise RuntimeError("Raw signal changed during archive round-trip")
        if not np.array_equal(saved["raw_time_s"], raw_time_s):
            raise RuntimeError("Raw timestamps changed during archive round-trip")
        if not np.array_equal(saved["offsets"], offsets_array):
            raise RuntimeError("Raw waveform offsets changed during archive round-trip")

    index_path = output_directory / "raw_event_waveform_index.csv"
    index = pd.DataFrame(index_rows)
    index.to_csv(
        index_path,
        mode="a" if index_path.exists() else "w",
        header=not index_path.exists(),
        index=False,
    )
    recording_index_path = recording_directory / "raw_event_waveform_index.csv"
    index.to_csv(
        recording_index_path,
        mode="a" if recording_index_path.exists() else "w",
        header=not recording_index_path.exists(),
        index=False,
    )
    return archive_path, index_path


def save_raw_event_waveforms(features, output_directory):
    """Save accepted raw ABF event samples with a provenance index.

    Variable-length waveforms are stored as one concatenated float64 array.
    ``waveform_value_start`` and ``waveform_value_stop`` in the CSV index map
    every event to its slice without requiring pickle/object arrays.
    """

    required = {
        "protein", "recording_id", "annotation_id", "event_id", "abf_path",
        "sweep_index", "event_start_index", "event_end_index",
        "n_event_points", "sampling_rate_hz", "abf_event_start_time_s",
        "abf_event_end_time_s", "baseline_mean",
    }
    missing = sorted(required - set(features.columns))
    if missing:
        raise ValueError(f"Cannot save raw waveforms; missing columns: {missing}")
    if features.empty:
        raise ValueError("Cannot save raw waveforms for an empty feature table")

    signal_parts = []
    time_parts = []
    offsets = [0]
    event_start_times_s = []
    event_end_times_s = []
    baseline_means = []
    index_rows = []
    waveform_number = 0
    grouped = features.groupby(["abf_path", "sweep_index"], sort=False)
    for (abf_path, sweep_index), group in grouped:
        abf_data = load_abf_signal(abf_path, int(sweep_index))
        full_signal = abf_data["signal"]
        full_time_s = abf_data["time_s"]
        for row in group.itertuples(index=False):
            start = int(row.event_start_index)
            stop = int(row.event_end_index)
            if not (0 <= start < stop <= len(full_signal)):
                raise ValueError(f"Invalid waveform indices for {row.event_id}")
            waveform = np.asarray(full_signal[start:stop], dtype=np.float64)
            waveform_time_s = np.asarray(full_time_s[start:stop], dtype=np.float64)
            if len(waveform) != int(row.n_event_points):
                raise ValueError(f"Waveform length mismatch for {row.event_id}")
            if not np.all(np.isfinite(waveform)):
                raise ValueError(f"Nonfinite raw waveform for {row.event_id}")

            value_start = offsets[-1]
            value_stop = value_start + len(waveform)
            signal_parts.append(waveform)
            time_parts.append(waveform_time_s)
            offsets.append(value_stop)
            event_start_times_s.append(float(row.abf_event_start_time_s))
            event_end_times_s.append(float(row.abf_event_end_time_s))
            baseline_means.append(float(row.baseline_mean))
            index_rows.append({
                "waveform_index": waveform_number,
                "protein": row.protein,
                "recording_id": row.recording_id,
                "annotation_id": row.annotation_id,
                "event_id": row.event_id,
                "abf_path": str(abf_path),
                "sweep_index": int(sweep_index),
                "abf_event_start_index": start,
                "abf_event_end_index": stop,
                "waveform_value_start": value_start,
                "waveform_value_stop": value_stop,
                "n_samples": len(waveform),
                "sampling_rate_hz": float(row.sampling_rate_hz),
                "sample_period_s": float(abf_data["sampling_period_s"]),
                "event_start_time_s": float(row.abf_event_start_time_s),
                "event_end_time_s": float(row.abf_event_end_time_s),
                "signal_unit": abf_data["signal_unit"],
            })
            waveform_number += 1

    raw_signal = np.concatenate(signal_parts)
    raw_time_s = np.concatenate(time_parts)
    offsets_array = np.asarray(offsets, dtype=np.int64)
    if (len(offsets_array) != len(features) + 1
            or offsets_array[-1] != len(raw_signal)
            or len(raw_time_s) != len(raw_signal)):
        raise RuntimeError("Raw waveform offset validation failed")

    output_directory = Path(output_directory)
    waveform_path = output_directory / "raw_event_waveforms.npz"
    waveform_index_path = output_directory / "raw_event_waveform_index.csv"
    np.savez_compressed(
        waveform_path,
        raw_signal=raw_signal,
        raw_time_s=raw_time_s,
        offsets=offsets_array,
        event_start_time_s=np.asarray(event_start_times_s, dtype=np.float64),
        event_end_time_s=np.asarray(event_end_times_s, dtype=np.float64),
        baseline_mean=np.asarray(baseline_means, dtype=np.float64),
    )
    waveform_index = pd.DataFrame(index_rows)
    waveform_index.to_csv(waveform_index_path, index=False)

    with np.load(waveform_path, allow_pickle=False) as saved:
        if not np.array_equal(saved["raw_signal"], raw_signal):
            raise RuntimeError("Saved raw waveform values changed during round-trip")
        if not np.array_equal(saved["raw_time_s"], raw_time_s):
            raise RuntimeError("Saved raw waveform timestamps changed during round-trip")
        if not np.array_equal(saved["offsets"], offsets_array):
            raise RuntimeError("Saved raw waveform offsets changed during round-trip")
        if len(saved["event_start_time_s"]) != len(features):
            raise RuntimeError("Saved raw waveform boundaries have wrong length")
    if waveform_index["event_id"].duplicated().any():
        raise RuntimeError("Raw waveform index contains duplicate event IDs")

    return {
        "raw_waveform_path": waveform_path,
        "raw_waveform_index_path": waveform_index_path,
    }


def save_results(
    features,
    qc,
    output_directory,
    alignment=None,
):
    """Save extracted features and QC results."""

    output_directory = Path(
        output_directory
    )

    if not output_directory.is_dir():
        raise FileNotFoundError(
            "Run output directory must be created before saving: "
            f"{output_directory}"
        )

    feature_output_path = (
        output_directory
        / "abf_event_features.csv"
    )

    qc_output_path = (
        output_directory
        / "abf_event_qc.csv"
    )

    duplicate_output_path = (
        output_directory
        / "duplicate_annotations.csv"
    )

    recording_summary_path = (
        output_directory
        / "recording_qc_summary.csv"
    )

    alignment_path = output_directory / "annotation_alignment.csv"

    features.to_csv(
        feature_output_path,
        index=False,
    )

    qc_for_output = qc.copy()
    qc_for_output["extraction_qc_pass"] = (
        qc_for_output["qc_pass"]
    )
    qc_for_output["extraction_qc_reason"] = (
        qc_for_output["qc_reason"]
    )

    qc_for_output.to_csv(
        qc_output_path,
        index=False,
    )

    if not qc.empty:
        duplicates = qc[
            qc["qc_reason"]
            == "duplicate_annotation"
        ].copy()
    else:
        duplicates = pd.DataFrame()

    duplicates.to_csv(
        duplicate_output_path,
        index=False,
    )

    recording_summary = build_recording_qc_summary(
        features,
        qc,
    )
    recording_summary.to_csv(
        recording_summary_path,
        index=False,
    )

    (alignment if alignment is not None else pd.DataFrame()).to_csv(
        alignment_path, index=False
    )

    alignment_for_output = alignment if alignment is not None else pd.DataFrame()
    recording_ids = sorted(
        set(features.get("recording_id", pd.Series(dtype=str)).dropna().astype(str))
        | set(qc_for_output.get("recording_id", pd.Series(dtype=str)).dropna().astype(str))
        | set(alignment_for_output.get("recording_id", pd.Series(dtype=str)).dropna().astype(str))
    )
    for recording_id in recording_ids:
        if Path(recording_id).name != recording_id or recording_id in {".", ".."}:
            raise ValueError(f"Unsafe recording_id for output path: {recording_id}")
        recording_directory = output_directory / "recordings" / recording_id
        recording_directory.mkdir(parents=True, exist_ok=True)
        recording_features = features[
            features["recording_id"].astype(str) == recording_id
        ]
        recording_qc = qc_for_output[
            qc_for_output["recording_id"].astype(str) == recording_id
        ]
        recording_duplicates = recording_qc[
            recording_qc["qc_reason"] == "duplicate_annotation"
        ]
        recording_alignment = (
            alignment_for_output[
                alignment_for_output["recording_id"].astype(str) == recording_id
            ]
            if "recording_id" in alignment_for_output
            else pd.DataFrame()
        )
        recording_features.to_csv(
            recording_directory / "event_features.csv", index=False
        )
        recording_qc.to_csv(recording_directory / "event_qc.csv", index=False)
        recording_duplicates.to_csv(
            recording_directory / "duplicate_annotations.csv", index=False
        )
        recording_alignment.to_csv(
            recording_directory / "annotation_alignment.csv", index=False
        )

    return {
        "feature_path":
            feature_output_path,
        "qc_path":
            qc_output_path,
        "duplicate_path":
            duplicate_output_path,
        "recording_summary_path":
            recording_summary_path,
        "alignment_path":
            alignment_path,
    }

# =========================================================
# Main entry point
# =========================================================

def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Extract versioned nanopore event features."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
    )
    parser.add_argument(
        "--run-id",
        type=str,
        default=None,
    )
    return parser.parse_args()


def main(args=None):
    if args is None:
        args = parse_arguments()

    config = load_project_config(args.config)
    run_id = args.run_id or make_run_id(
        config["dataset_version"],
        prefix="extract",
    )
    output_directory = require_new_output_directory(
        Path(config["extraction_output_root"])
        / run_id
    )

    configure_feature_engineering(
        shape_smoothing_window_ms=(
            config["shape_smoothing_window_ms"]
        ),
        minimum_shape_peak_snr=(
            config["minimum_shape_peak_snr"]
        ),
    )

    manifest = load_recording_manifest(
        config["manifest_path"]
    )

    validate_recording_manifest(
        manifest
    )

    features, qc, alignment = (
        process_all_recordings(
            manifest=manifest,
            baseline_window_ms=(
                float(config["baseline_window_ms"])
            ),
            shape_context_window_ms=float(config["shape_context_window_ms"]),
            alignment_sample_events=int(config["alignment_sample_events"]),
            alignment_minimum_correlation=float(
                config["alignment_minimum_correlation"]
            ),
            alignment_maximum_normalized_rmse=float(
                config["alignment_maximum_normalized_rmse"]
            ),
            raw_waveform_output_directory=output_directory,
        )
    )

    # -----------------------------------------
    # Check missing feature values
    # -----------------------------------------

    feature_columns = [
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
    ]

    print("\nMissing feature values:")

    print(
        features[
            feature_columns
        ]
        .isna()
        .sum()
    )

    # -----------------------------------------
    # Save results
    # -----------------------------------------

    output_paths = save_results(
        features=features,
        qc=qc,
        output_directory=(
            output_directory
        ),
        alignment=alignment,
    )
    raw_waveform_index_path = output_directory / "raw_event_waveform_index.csv"
    raw_waveform_directory = output_directory / "recordings"
    if (
        not raw_waveform_index_path.is_file()
        or not raw_waveform_directory.is_dir()
        or not any(raw_waveform_directory.glob("*/raw_event_waveforms/*.npz"))
    ):
        raise RuntimeError("Raw waveform artifacts were not created")
    output_paths.update({
        "raw_waveform_index_path": raw_waveform_index_path,
        "raw_waveform_directory": raw_waveform_directory,
    })

    run_manifest_path = write_run_manifest(
        output_directory / "run_manifest.json",
        {
            "run_type": "event_extraction",
            "run_id": run_id,
            "dataset_version": config["dataset_version"],
            "feature_definition_version": (
                config["feature_definition_version"]
            ),
            "configuration": config,
            "manifest_sha256": sha256_file(
                config["manifest_path"]
            ),
            "n_manifest_rows": len(manifest),
            "n_accepted_events": len(features),
            "n_qc_rows": len(qc),
            "n_alignment_checks": len(alignment),
            "recording_time_offsets_s": {
                row.recording_id: float(row.time_offset_s)
                for row in manifest.itertuples(index=False)
            },
            "output_files": {
                name: str(path)
                for name, path in output_paths.items()
            },
        },
    )
    output_paths["run_manifest_path"] = run_manifest_path

    print("\nAccepted events:")

    print(
        features
        .groupby(
            [
                "protein",
                "recording_id",
            ]
        )
        .size()
    )

    if not qc.empty:
        print("\nQC summary:")

        print(
            qc
            .groupby(
                [
                    "protein",
                    "qc_pass",
                ]
            )
            .size()
        )

        failed_qc = qc[
            ~qc["qc_pass"]
        ]

        if not failed_qc.empty:
            print(
                "\nQC failure reasons:"
            )

            print(
                failed_qc[
                    "qc_reason"
                ]
                .value_counts()
            )

    print("\nOutput files:")

    for name, path in output_paths.items():
        print(
            f"  {name}: {path}"
        )


if __name__ == "__main__":
    main()
