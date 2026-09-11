"""Shared configuration and run-provenance helpers."""

import hashlib
import json
import platform
from datetime import datetime
from pathlib import Path


DEFAULT_CONFIG_PATH = (
    Path(__file__).resolve().parent
    / "config"
    / "experiment.json"
)


def load_project_config(config_path=DEFAULT_CONFIG_PATH):
    """Load JSON configuration and resolve paths relative to the config."""

    config_path = Path(config_path).resolve()

    if not config_path.is_file():
        raise FileNotFoundError(
            f"Configuration not found: {config_path}"
        )

    with config_path.open("r", encoding="utf-8") as file:
        config = json.load(file)

    required = [
        "data_root",
        "manifest_path",
        "extraction_output_root",
        "visualization_output_root",
        "dataset_version",
        "feature_definition_version",
        "baseline_window_ms",
        "shape_smoothing_window_ms",
        "shape_context_window_ms",
        "minimum_shape_peak_snr",
        "alignment_sample_events",
        "alignment_minimum_correlation",
        "alignment_maximum_normalized_rmse",
        "protein_order",
        "random_seed",
    ]
    missing = [key for key in required if key not in config]

    if missing:
        raise ValueError(
            f"Configuration is missing keys: {missing}"
        )

    data_root = (
        config_path.parent / config["data_root"]
    ).resolve()
    resolved = dict(config)
    resolved["config_path"] = str(config_path)
    resolved["data_root"] = str(data_root)
    resolved["manifest_path"] = str(
        (data_root / config["manifest_path"]).resolve()
    )
    resolved["extraction_output_root"] = str(
        (data_root / config["extraction_output_root"]).resolve()
    )
    resolved["visualization_output_root"] = str(
        (data_root / config["visualization_output_root"]).resolve()
    )

    if float(resolved["baseline_window_ms"]) <= 0:
        raise ValueError("baseline_window_ms must be positive")

    if float(resolved["shape_smoothing_window_ms"]) <= 0:
        raise ValueError("shape_smoothing_window_ms must be positive")

    if float(resolved["shape_context_window_ms"]) < 0:
        raise ValueError("shape_context_window_ms cannot be negative")

    if float(resolved["minimum_shape_peak_snr"]) <= 0:
        raise ValueError("minimum_shape_peak_snr must be positive")

    if int(resolved["alignment_sample_events"]) <= 0:
        raise ValueError("alignment_sample_events must be positive")

    correlation = float(resolved["alignment_minimum_correlation"])
    if not 0 < correlation <= 1:
        raise ValueError("alignment_minimum_correlation must be in (0, 1]")

    if float(resolved["alignment_maximum_normalized_rmse"]) < 0:
        raise ValueError("alignment_maximum_normalized_rmse cannot be negative")

    return resolved


def make_run_id(dataset_version, prefix="run"):
    """Create a sortable run identifier."""

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{prefix}_{timestamp}_{dataset_version}"


def require_new_output_directory(path):
    """Create a new run directory and refuse any existing destination."""

    path = Path(path)

    if path.exists():
        raise FileExistsError(
            f"Run output already exists; choose a new run ID: {path}"
        )

    path.mkdir(parents=True, exist_ok=False)
    return path


def sha256_file(path):
    """Return the SHA-256 checksum of a file."""

    digest = hashlib.sha256()

    with Path(path).open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def write_run_manifest(path, payload):
    """Write a JSON run manifest with runtime metadata."""

    manifest = {
        "created_at": datetime.now().astimezone().isoformat(),
        "python_version": platform.python_version(),
        **payload,
    }

    path = Path(path)

    with path.open("x", encoding="utf-8") as file:
        json.dump(manifest, file, indent=2, sort_keys=True)

    return path
