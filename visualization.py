import argparse
import json
from pathlib import Path
from itertools import combinations
import matplotlib

# Save figures without opening GUI windows.
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import (
    mannwhitneyu,
    pointbiserialr,
    rankdata,
)

from project_config import (
    DEFAULT_CONFIG_PATH,
    load_project_config,
    make_run_id,
    require_new_output_directory,
    sha256_file,
    write_run_manifest,
)

# =========================================================
# Configuration
# =========================================================

FEATURE_PATH = None
QC_PATH = None
INPUT_RUN_MANIFEST_PATH = None
OUTPUT_DIRECTORY = None

HISTOGRAM_DIRECTORY = None
VIOLIN_DIRECTORY = None
SCATTER_DIRECTORY = None
CORRELATION_DIRECTORY = None
SUMMARY_DIRECTORY = None
SINGLE_FEATURE_DIRECTORY = None

SOURCE_FEATURE_COLUMNS = [
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

VISUALIZATION_FEATURE_COLUMNS = [
    "dwell_time_ms",
    "log_dwell_time",
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
    "right_slope_magnitude",
    "left_slope_magnitude",
]

FEATURE_LABELS = {
    "dwell_time_ms":
        "Dwell time (ms)",

    "log_dwell_time":
        "log10(Dwell time [ms])",

    "blockage_percent":
        "Blockage percent (%)",

    "blockage_mean":
        "Blockage mean (%)",

    "blockage_sd":
        "Blockage SD (%)",

    "auc":
        "Area under the curve (%·ms)",

    "skewness":
        "Skewness",

    "kurtosis":
        "Kurtosis",

    "fwhm_width":
        "FWHM width (ms)",

    "fwhm_drop":
        "Half-maximum blockage (%)",

    "rise_time":
        "Rise time (ms)",

    "fall_time":
        "Fall time (ms)",

    "right_slope":
        "Right slope (%/ms)",

    "left_slope":
        "Left slope (%/ms)",

    "right_slope_magnitude":
        "Right slope magnitude (%/ms)",

    "left_slope_magnitude":
        "Left slope magnitude (%/ms)",
}

# Positive and often right-skewed features.
LOG_SCALE_FEATURES = {
    "dwell_time_ms",
    "auc",
    "fwhm_width",
    "rise_time",
    "fall_time",
}

# Histogram-specific scaling. These features are positive and
# strongly right-skewed, so logarithmically spaced bins make
# their central distributions visible without deleting values.
HISTOGRAM_LOG_SCALE_FEATURES = (
    LOG_SCALE_FEATURES
    | {
        "blockage_mean",
        "blockage_sd",
        "fwhm_drop",
        "right_slope_magnitude",
        "left_slope_magnitude",
    }
)

# Fisher kurtosis can be negative, so regular log scaling is
# invalid. Use a central percentile view for its histogram.
HISTOGRAM_PERCENTILE_FEATURES = {
    "kurtosis",
}

# Extra log scaling used only on scatter-plot axes.
SCATTER_LOG_SCALE_FEATURES = (
    LOG_SCALE_FEATURES
    | {
        "blockage_mean",
        "blockage_sd",
        "fwhm_drop",
        "right_slope_magnitude",
        "left_slope_magnitude",
    }
)

# Fisher kurtosis may be negative, so a regular log axis
# cannot represent it without discarding valid events.
SCATTER_SYMLOG_SCALE_FEATURES = {
    "kurtosis",
}

# Preserve full-range plots for QC and create a second,
# central-range view that is easier to interpret.
PLOT_LOWER_PERCENTILE = 1.0
PLOT_UPPER_PERCENTILE = 99.0

PROTEIN_ORDER = [
    "HSA",
    "F8",
]

PROTEIN_COLORS = {
    "HSA": "tab:blue",
    "F8": "tab:orange",
}

EXCLUDED_SCATTER_PAIRS = {
    frozenset({
        "dwell_time_ms",
        "log_dwell_time",
    }),
}

def create_scatter_pairs():
    """
    Create every unique feature pair, except
    dwell_time_ms versus log_dwell_time.
    """

    scatter_pairs = []

    for x_feature, y_feature in combinations(
        VISUALIZATION_FEATURE_COLUMNS,
        2,
    ):
        pair = frozenset({
            x_feature,
            y_feature,
        })

        if pair in EXCLUDED_SCATTER_PAIRS:
            continue

        scatter_pairs.append(
            (
                x_feature,
                y_feature,
            )
        )

    return scatter_pairs


ALL_SCATTER_PAIRS = create_scatter_pairs()

# A compact diagnostic set avoids treating 90 pairwise plots as 90
# independent findings. The complete matrix remains available by config.
CURATED_SCATTER_PAIRS = [
    ("dwell_time_ms", "blockage_percent"),
    ("log_dwell_time", "blockage_mean"),
    ("blockage_percent", "blockage_mean"),
    ("dwell_time_ms", "auc"),
    ("dwell_time_ms", "fwhm_width"),
    ("rise_time", "fall_time"),
    ("rise_time", "left_slope_magnitude"),
    ("fall_time", "right_slope_magnitude"),
    ("skewness", "kurtosis"),
    ("skewness", "blockage_percent"),
    ("blockage_mean", "blockage_sd"),
]


def configure_output_paths(output_directory):
    """Set run-specific output paths used by plotting functions."""

    global OUTPUT_DIRECTORY
    global HISTOGRAM_DIRECTORY, VIOLIN_DIRECTORY, SCATTER_DIRECTORY
    global CORRELATION_DIRECTORY, SUMMARY_DIRECTORY
    global SINGLE_FEATURE_DIRECTORY

    OUTPUT_DIRECTORY = Path(output_directory)
    HISTOGRAM_DIRECTORY = OUTPUT_DIRECTORY / "histograms"
    VIOLIN_DIRECTORY = OUTPUT_DIRECTORY / "violin_plots"
    SCATTER_DIRECTORY = OUTPUT_DIRECTORY / "scatter_plots"
    CORRELATION_DIRECTORY = OUTPUT_DIRECTORY / "correlations"
    SUMMARY_DIRECTORY = OUTPUT_DIRECTORY / "summaries"
    SINGLE_FEATURE_DIRECTORY = OUTPUT_DIRECTORY / "single_feature_association"


# =========================================================
# General utilities
# =========================================================

def create_output_directories():
    """Create every visualization output folder."""

    directories = [
        OUTPUT_DIRECTORY,
        HISTOGRAM_DIRECTORY,
        VIOLIN_DIRECTORY,
        SCATTER_DIRECTORY,
        CORRELATION_DIRECTORY,
        SUMMARY_DIRECTORY,
        SINGLE_FEATURE_DIRECTORY,
    ]

    for directory in directories:
        directory.mkdir(
            parents=True,
            exist_ok=True,
        )


def configure_plot_style():
    """Set a consistent figure style."""

    sns.set_theme(
        style="whitegrid",
        context="notebook",
    )

    plt.rcParams.update({
        "figure.dpi": 120,
        "savefig.dpi": 300,
        "axes.titleweight": "bold",
        "axes.labelsize": 11,
        "axes.titlesize": 13,
        "legend.frameon": True,
    })


def load_feature_data(
    feature_path,
):
    """Load and validate the extracted feature table."""

    feature_path = Path(
        feature_path
    )

    if not feature_path.is_file():
        raise FileNotFoundError(
            f"Feature file not found: "
            f"{feature_path}"
        )

    data = pd.read_csv(
        feature_path
    )

    required_columns = [
        "protein",
        "recording_id",
        "abf_path",
        *SOURCE_FEATURE_COLUMNS,
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in data.columns
    ]

    if missing_columns:
        raise ValueError(
            "Feature table is missing columns: "
            f"{missing_columns}"
        )

    if data.empty:
        raise ValueError(
            "Feature table contains no events"
        )

    data = data.copy()

    # Replace infinities with NaN so plots can
    # handle missing values consistently.
    data[SOURCE_FEATURE_COLUMNS] = (
        data[SOURCE_FEATURE_COLUMNS]
        .replace(
            [np.inf, -np.inf],
            np.nan,
        )
    )

    data["protein"] = (
        data["protein"]
        .astype(str)
        .str.strip()
    )

    unexpected = sorted(set(data["protein"]) - set(PROTEIN_ORDER))
    if unexpected:
        raise ValueError(f"Unexpected protein labels: {unexpected}")

    return data


def load_qc_data(qc_path):
    """Load event-level extraction QC produced by the same run."""

    qc = pd.read_csv(qc_path)
    required = {"protein", "recording_id", "extraction_qc_pass", "extraction_qc_reason"}
    missing = sorted(required - set(qc.columns))
    if missing:
        raise ValueError(f"QC table is missing columns: {missing}")
    return qc

def save_figure(
    fig,
    output_path,
):
    """Save and close one Matplotlib figure."""

    fig.tight_layout()

    fig.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)


def get_valid_feature_data(
    data,
    feature,
    positive_only=False,
):
    """Return protein and one finite feature."""

    plot_data = (
        data[
            [
                "protein",
                feature,
            ]
        ]
        .dropna()
        .copy()
    )

    if positive_only:
        plot_data = plot_data[
            plot_data[feature] > 0
        ]

    return plot_data


def get_percentile_limits(
    data,
    feature,
    lower=PLOT_LOWER_PERCENTILE,
    upper=PLOT_UPPER_PERCENTILE,
):
    """Return robust plotting limits without changing source data."""

    values = (
        data[feature]
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
        .astype(float)
    )

    if feature in LOG_SCALE_FEATURES:
        values = values[values > 0]

    if values.empty:
        return None

    lower_limit = float(np.percentile(values, lower))
    upper_limit = float(np.percentile(values, upper))

    if (
        not np.isfinite(lower_limit)
        or not np.isfinite(upper_limit)
        or lower_limit >= upper_limit
    ):
        return None

    return lower_limit, upper_limit


def get_valid_pair_data(
    data,
    x_feature,
    y_feature,
):
    """Return finite data for one feature pair."""

    plot_data = (
        data[
            [
                "protein",
                x_feature,
                y_feature,
            ]
        ]
        .dropna()
        .copy()
    )

    if x_feature in SCATTER_LOG_SCALE_FEATURES:
        plot_data = plot_data[
            plot_data[x_feature] > 0
        ]

    if y_feature in SCATTER_LOG_SCALE_FEATURES:
        plot_data = plot_data[
            plot_data[y_feature] > 0
        ]

    return plot_data

# =========================================================
# Dataset summaries
# =========================================================

def save_numeric_summary(data):
    summary = (
        data
        .groupby("protein")[
            VISUALIZATION_FEATURE_COLUMNS
        ]
        .describe()
    )

    summary.to_csv(
        SUMMARY_DIRECTORY
        / "feature_summary_by_protein.csv"
    )


def save_missing_summary(data):
    missing_count = (
        data[VISUALIZATION_FEATURE_COLUMNS]
        .isna()
        .sum()
    )

    missing_percent = (
        data[VISUALIZATION_FEATURE_COLUMNS]
        .isna()
        .mean()
        * 100.0
    )

    summary = pd.DataFrame({
        "missing_count": missing_count,
        "missing_percent": missing_percent,
    })

    summary.to_csv(
        SUMMARY_DIRECTORY
        / "missing_values.csv"
    )

    return summary


def save_recording_level_summaries(data, qc):
    """Save sample-size, acceptance, shape-QC and missingness by recording."""

    accepted = (
        data.groupby(["protein", "recording_id"])
        .size().rename("n_accepted").reset_index()
    )
    annotated = (
        qc.groupby(["protein", "recording_id"])
        .size().rename("n_annotated").reset_index()
    )
    counts = annotated.merge(accepted, how="left", on=["protein", "recording_id"])
    counts["n_accepted"] = counts["n_accepted"].fillna(0).astype(int)
    counts["acceptance_rate"] = counts["n_accepted"] / counts["n_annotated"]
    counts.to_csv(SUMMARY_DIRECTORY / "event_counts_by_recording.csv", index=False)

    missing = (
        data.groupby(["protein", "recording_id"])[VISUALIZATION_FEATURE_COLUMNS]
        .apply(lambda frame: frame.isna().mean() * 100.0)
        .reset_index()
    )
    missing.to_csv(SUMMARY_DIRECTORY / "missing_percent_by_recording.csv", index=False)

    shape = (
        data.groupby(["protein", "recording_id"])
        .agg(
            n_accepted=("recording_id", "size"),
            n_shape_qc_pass=("shape_qc_pass", "sum"),
            median_peak_snr=("peak_snr", "median"),
        ).reset_index()
    )
    shape["shape_qc_pass_rate"] = shape["n_shape_qc_pass"] / shape["n_accepted"]
    shape.to_csv(SUMMARY_DIRECTORY / "shape_qc_by_recording.csv", index=False)

    medians = (
        data.groupby(["protein", "recording_id"])[VISUALIZATION_FEATURE_COLUMNS]
        .median().reset_index()
    )
    medians.to_csv(SUMMARY_DIRECTORY / "feature_medians_by_recording.csv", index=False)

    status = {
        "recordings_by_protein": data.groupby("protein")["recording_id"].nunique().to_dict(),
        "recording_grouped_classifier_valid": bool(
            data.groupby("protein")["recording_id"].nunique().min() >= 2
        ),
        "note": "At least two independent recordings per protein are required; more are recommended.",
    }
    with (SUMMARY_DIRECTORY / "classifier_readiness.json").open("x", encoding="utf-8") as handle:
        json.dump(status, handle, indent=2, ensure_ascii=False)

    return counts, missing, shape, medians


def plot_recording_diagnostics(counts, missing, shape):
    """Plot recording-level counts, QC rates and feature missingness."""

    counts = counts.copy()
    counts["label"] = counts["protein"] + ": " + counts["recording_id"]
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(data=counts, x="n_accepted", y="label", hue="protein",
                palette=PROTEIN_COLORS, dodge=False, ax=ax)
    ax.set_title("Accepted events by independent recording")
    ax.set_xlabel("Accepted events")
    ax.set_ylabel("Recording")
    save_figure(fig, SUMMARY_DIRECTORY / "accepted_events_by_recording.png")

    rate_data = counts.merge(
        shape[["protein", "recording_id", "shape_qc_pass_rate"]],
        on=["protein", "recording_id"], how="left",
    )
    rate_long = rate_data.melt(
        id_vars=["protein", "recording_id", "label"],
        value_vars=["acceptance_rate", "shape_qc_pass_rate"],
        var_name="QC stage", value_name="rate",
    )
    fig, ax = plt.subplots(figsize=(10, 7))
    sns.barplot(data=rate_long, x="rate", y="label", hue="QC stage", ax=ax)
    ax.set_xlim(0, 1)
    ax.set_title("Extraction acceptance and shape-QC rates")
    ax.set_xlabel("Fraction")
    ax.set_ylabel("Recording")
    save_figure(fig, SUMMARY_DIRECTORY / "qc_rates_by_recording.png")

    heat = missing.set_index(["protein", "recording_id"])[VISUALIZATION_FEATURE_COLUMNS]
    fig, ax = plt.subplots(figsize=(14, max(5, 0.45 * len(heat))))
    sns.heatmap(heat, cmap="magma_r", vmin=0, vmax=100,
                cbar_kws={"label": "Missing (%)"}, ax=ax)
    ax.set_title("Feature missingness by recording")
    save_figure(fig, SUMMARY_DIRECTORY / "missingness_by_recording.png")


def save_and_plot_blockage_consistency(data):
    """Compare JSON blockage annotation with waveform-derived mean blockage."""

    columns = ["protein", "recording_id", "event_id", "blockage_percent", "blockage_mean"]
    comparison = data[columns].dropna().copy()
    comparison["computed_minus_annotation"] = (
        comparison["blockage_mean"] - comparison["blockage_percent"]
    )
    comparison["absolute_difference"] = comparison["computed_minus_annotation"].abs()
    comparison.sort_values("absolute_difference", ascending=False).to_csv(
        SUMMARY_DIRECTORY / "annotation_vs_computed_blockage.csv", index=False
    )

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.scatterplot(data=comparison, x="blockage_percent", y="blockage_mean",
                    hue="protein", palette=PROTEIN_COLORS, alpha=0.4, s=20, ax=ax)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_title("Annotation vs waveform-derived blockage (QC)")
    ax.set_xlabel("JSON blockage percent (%)")
    ax.set_ylabel("Waveform mean blockage (%)")
    save_figure(fig, SUMMARY_DIRECTORY / "annotation_vs_computed_blockage.png")


def save_extreme_feature_events(data):
    """Save events requiring waveform and baseline QC."""

    diagnostic_columns = [
        "protein",
        "recording_id",
        "event_id",
        "baseline_mean",
        "baseline_sd",
        "blockage_percent",
        "blockage_mean",
        "fwhm_drop",
        "auc",
    ]

    diagnostic_columns = [
        column
        for column in diagnostic_columns
        if column in data.columns
    ]

    extreme_mask = (
        (data["fwhm_drop"] > 100)
        | (data["blockage_mean"] > 100)
    )

    extreme_events = (
        data.loc[extreme_mask, diagnostic_columns]
        .sort_values(
            ["fwhm_drop", "blockage_mean"],
            ascending=False,
        )
    )

    extreme_events.to_csv(
        SUMMARY_DIRECTORY
        / "extreme_blockage_events.csv",
        index=False,
    )

    print(
        "\nEvents requiring blockage QC:",
        len(extreme_events),
    )

    return extreme_events


def plot_event_counts(
    data,
):
    """Plot accepted event counts by protein."""

    event_counts = (
        data["protein"]
        .value_counts()
        .reindex(PROTEIN_ORDER)
        .dropna()
    )

    fig, ax = plt.subplots(
        figsize=(6, 5)
    )

    colors = [
        PROTEIN_COLORS.get(protein)
        for protein in event_counts.index
    ]

    bars = ax.bar(
        event_counts.index,
        event_counts.values,
        color=colors,
        alpha=0.8,
    )

    for bar, count in zip(
        bars,
        event_counts.values,
    ):
        ax.text(
            bar.get_x()
            + bar.get_width() / 2,
            bar.get_height(),
            f"{int(count)}",
            ha="center",
            va="bottom",
        )

    ax.set_xlabel("Protein")
    ax.set_ylabel("Accepted events")
    ax.set_title(
        "Accepted event counts"
    )

    save_figure(
        fig,
        SUMMARY_DIRECTORY
        / "accepted_event_counts.png",
    )


def plot_missing_values(
    missing_summary,
):
    """Plot missing percentage for each feature."""

    plot_data = (
        missing_summary
        .reset_index()
        .rename(
            columns={
                "index": "feature",
            }
        )
    )

    fig, ax = plt.subplots(
        figsize=(10, 6)
    )

    sns.barplot(
        data=plot_data,
        x="missing_percent",
        y="feature",
        color="tab:red",
        alpha=0.7,
        ax=ax,
    )

    ax.set_xlabel("Missing values (%)")
    ax.set_ylabel("Feature")
    ax.set_title(
        "Missing feature values"
    )

    save_figure(
        fig,
        SUMMARY_DIRECTORY
        / "missing_values.png",
    )


# =========================================================
# Histograms
# =========================================================

def plot_feature_histogram(
    data,
    feature,
):
    """
    Plot normalized HSA/F8 histograms.

    Right-skewed positive features use
    logarithmically spaced bins.
    """

    use_log_scale = (
        feature in HISTOGRAM_LOG_SCALE_FEATURES
    )

    plot_data = get_valid_feature_data(
        data=data,
        feature=feature,
        positive_only=use_log_scale,
    )

    if plot_data.empty:
        print(
            f"Skipping histogram for "
            f"{feature}: no valid values"
        )
        return

    use_percentile_range = (
        feature in HISTOGRAM_PERCENTILE_FEATURES
    )

    if use_percentile_range:
        limits = get_percentile_limits(
            data=plot_data,
            feature=feature,
        )

        if limits is not None:
            lower_limit, upper_limit = limits
            plot_data = plot_data[
                plot_data[feature].between(
                    lower_limit,
                    upper_limit,
                    inclusive="both",
                )
            ]

    if plot_data.empty:
        print(
            f"Skipping histogram for "
            f"{feature}: no values in plotting range"
        )
        return

    all_values = (
        plot_data[feature]
        .to_numpy(dtype=float)
    )

    if use_log_scale:
        minimum_value = np.min(
            all_values
        )

        maximum_value = np.max(
            all_values
        )

        bins = np.logspace(
            np.log10(minimum_value),
            np.log10(maximum_value),
            41,
        )

        print(
            f"{feature}: log bins from "
            f"{minimum_value:.6g} to "
            f"{maximum_value:.6g}"
        )
    else:
        bins = np.linspace(
            np.min(all_values),
            np.max(all_values),
            41,
        )

    fig, ax = plt.subplots(
        figsize=(8, 5)
    )

    # Use Matplotlib directly so both proteins
    # use exactly the same bin edges.
    for protein in PROTEIN_ORDER:
        values = (
            plot_data.loc[
                plot_data["protein"]
                == protein,
                feature,
            ]
            .to_numpy(dtype=float)
        )

        if len(values) == 0:
            continue

        ax.hist(
            values,
            bins=bins,
            density=True,
            histtype="stepfilled",
            alpha=0.30,
            linewidth=1.5,
            color=PROTEIN_COLORS[protein],
            edgecolor=PROTEIN_COLORS[
                protein
            ],
            label=(
                f"{protein} "
                f"(n={len(values)})"
            ),
        )

    if use_log_scale:
        ax.set_xscale("log")

    ax.set_xlabel(
        FEATURE_LABELS[feature]
    )

    ax.set_ylabel(
        "Density"
    )

    title_suffix = (
        " (central 1–99%)"
        if use_percentile_range
        else ""
    )

    ax.set_title(
        f"Distribution of "
        f"{FEATURE_LABELS[feature]}"
        f"{title_suffix}"
    )

    ax.legend(
        title="Protein"
    )

    if use_log_scale:
        output_name = (
            f"hist_{feature}_logbins.png"
        )
    elif use_percentile_range:
        output_name = (
            f"hist_{feature}_central.png"
        )
    else:
        output_name = (
            f"hist_{feature}.png"
        )

    save_figure(
        fig,
        HISTOGRAM_DIRECTORY
        / output_name,
    )


def plot_all_histograms(data):
    for feature in VISUALIZATION_FEATURE_COLUMNS:
        plot_feature_histogram(
            data=data,
            feature=feature,
        )


# =========================================================
# Violin plots
# =========================================================

def plot_feature_violin(
    data,
    feature,
    zoom=False,
):
    """Plot one HSA/F8 violin plot."""

    use_log_scale = feature in HISTOGRAM_LOG_SCALE_FEATURES

    plot_data = get_valid_feature_data(
        data=data,
        feature=feature,
        positive_only=use_log_scale,
    )

    if plot_data.empty:
        print(
            f"Skipping violin for "
            f"{feature}: no valid values"
        )
        return

    original_count = len(plot_data)

    if zoom:
        limits = get_percentile_limits(
            data=plot_data,
            feature=feature,
        )

        if limits is not None:
            lower_limit, upper_limit = limits
            plot_data = plot_data[
                plot_data[feature].between(
                    lower_limit,
                    upper_limit,
                    inclusive="both",
                )
            ]

    if plot_data.empty:
        return

    fig, ax = plt.subplots(
        figsize=(7, 5)
    )
    
    sns.violinplot(
        data=plot_data,
        x="protein",
        y=feature,
        hue="protein",
        order=PROTEIN_ORDER,
        hue_order=PROTEIN_ORDER,
        palette=PROTEIN_COLORS,
        inner="quartile",
        cut=0,
        density_norm="width",
        legend=False,
        ax=ax,
    )

    if use_log_scale:
        ax.set_yscale("log")

    ax.set_xlabel("Protein")

    ax.set_ylabel(
        FEATURE_LABELS[feature]
    )

    range_label = (
        "central 1–99%"
        if zoom
        else "full range"
    )

    ax.set_title(
        f"{FEATURE_LABELS[feature]}: "
        f"HSA vs F8 ({range_label})"
    )

    output_suffix = "zoomed" if zoom else "full"

    save_figure(
        fig,
        VIOLIN_DIRECTORY
        / f"violin_{feature}_{output_suffix}.png",
    )

    if zoom:
        print(
            f"{feature}: central violin used "
            f"{len(plot_data)}/{original_count} events"
        )


def plot_all_violin_plots(data):
    for feature in VISUALIZATION_FEATURE_COLUMNS:
        plot_feature_violin(
            data=data,
            feature=feature,
            zoom=False,
        )

        plot_feature_violin(
            data=data,
            feature=feature,
            zoom=True,
        )

        
# =========================================================
# Scatter plots
# =========================================================

def plot_feature_scatter(
    data,
    x_feature,
    y_feature,
    apply_automatic_axis_scales=True,
):
    """Plot full and central-range views of one feature pair together."""

    plot_data = get_valid_pair_data(
        data=data,
        x_feature=x_feature,
        y_feature=y_feature,
    )

    if plot_data.empty:
        print(
            "Skipping scatter: "
            f"{x_feature} vs {y_feature}"
        )
        return

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    for ax, range_label in zip(axes, ["Full range", "Central 1–99%"]):
        sns.scatterplot(
            data=plot_data,
            x=x_feature,
            y=y_feature,
            hue="protein",
            hue_order=PROTEIN_ORDER,
            palette=PROTEIN_COLORS,
            alpha=0.40,
            s=20,
            edgecolor=None,
            ax=ax,
        )

        if apply_automatic_axis_scales and x_feature in SCATTER_LOG_SCALE_FEATURES:
            ax.set_xscale("log")
        if apply_automatic_axis_scales and y_feature in SCATTER_LOG_SCALE_FEATURES:
            ax.set_yscale("log")
        if apply_automatic_axis_scales and x_feature in SCATTER_SYMLOG_SCALE_FEATURES:
            ax.set_xscale("symlog", linthresh=1.0)
        if apply_automatic_axis_scales and y_feature in SCATTER_SYMLOG_SCALE_FEATURES:
            ax.set_yscale("symlog", linthresh=1.0)

        ax.set_xlabel(FEATURE_LABELS[x_feature])
        ax.set_ylabel(FEATURE_LABELS[y_feature])
        ax.set_title(f"{range_label} (n={len(plot_data):,})")

    x_limits = get_percentile_limits(plot_data, x_feature)
    y_limits = get_percentile_limits(plot_data, y_feature)
    if x_limits is not None:
        axes[1].set_xlim(*x_limits)
    if y_limits is not None:
        axes[1].set_ylim(*y_limits)

    axes[0].legend(title="Protein")
    if axes[1].get_legend() is not None:
        axes[1].get_legend().remove()

    fig.suptitle(
        f"{FEATURE_LABELS[x_feature]} vs {FEATURE_LABELS[y_feature]}",
        fontsize=15,
        fontweight="bold",
    )

    save_figure(
        fig,
        SCATTER_DIRECTORY
        / (
            f"scatter_{x_feature}"
            f"_vs_{y_feature}"
            "_full_and_zoomed.png"
        ),
    )

def plot_all_scatter_plots(
    data,
    generate_all=False,
):
    """Plot curated diagnostics, or every pair when explicitly requested."""

    scatter_pairs = ALL_SCATTER_PAIRS if generate_all else CURATED_SCATTER_PAIRS
    for (
        x_feature,
        y_feature,
    ) in scatter_pairs:
        plot_feature_scatter(
            data=data,
            x_feature=x_feature,
            y_feature=y_feature,
        )


def plot_recording_scatter_panels(data, output_directory):
    """Show curated feature pairs with recording identity made explicit."""

    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)
    recording_order = sorted(data["recording_id"].astype(str).unique())
    palette = dict(zip(recording_order, sns.color_palette("tab10", len(recording_order))))

    for x_feature, y_feature in CURATED_SCATTER_PAIRS:
        plot_data = get_valid_pair_data(data, x_feature, y_feature).copy()
        if plot_data.empty:
            continue
        plot_data["recording_id"] = data.loc[plot_data.index, "recording_id"].astype(str)

        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        sns.scatterplot(
            data=plot_data, x=x_feature, y=y_feature, hue="protein",
            hue_order=PROTEIN_ORDER, palette=PROTEIN_COLORS, alpha=0.40,
            s=20, edgecolor=None, ax=axes[0],
        )
        sns.scatterplot(
            data=plot_data, x=x_feature, y=y_feature, hue="recording_id",
            hue_order=recording_order, palette=palette, alpha=0.45,
            s=20, edgecolor=None, ax=axes[1],
        )
        for ax in axes:
            if x_feature in SCATTER_LOG_SCALE_FEATURES:
                ax.set_xscale("log")
            if y_feature in SCATTER_LOG_SCALE_FEATURES:
                ax.set_yscale("log")
            if x_feature in SCATTER_SYMLOG_SCALE_FEATURES:
                ax.set_xscale("symlog", linthresh=1.0)
            if y_feature in SCATTER_SYMLOG_SCALE_FEATURES:
                ax.set_yscale("symlog", linthresh=1.0)
            ax.set_xlabel(FEATURE_LABELS[x_feature])
            ax.set_ylabel(FEATURE_LABELS[y_feature])
        axes[0].set_title("Colored by protein")
        axes[1].set_title("Colored by recording")
        fig.suptitle(
            f"Protein and recording views: {FEATURE_LABELS[x_feature]} vs "
            f"{FEATURE_LABELS[y_feature]}",
            fontsize=14, fontweight="bold",
        )
        save_figure(
            fig,
            output_directory / f"recording_scatter_{x_feature}_vs_{y_feature}.png",
        )

# =========================================================
# Correlation heatmaps
# =========================================================

def plot_correlation_heatmap(
    data,
    method,
    suffix,
    title,
):
    """Plot a feature-correlation heatmap."""

    correlation = (
        data[VISUALIZATION_FEATURE_COLUMNS]
        .corr(
            method=method,
            min_periods=3,
        )
    )   

    fig, ax = plt.subplots(
        figsize=(13, 11)
    )

    sns.heatmap(
        correlation,
        cmap="coolwarm",
        center=0,
        vmin=-1,
        vmax=1,
        square=True,
        annot=True,
        fmt=".2f",
        linewidths=0.4,
        cbar_kws={
            "label": (
                f"{method.title()} "
                "correlation"
            ),
        },
        ax=ax,
    )

    ax.set_title(title)

    save_figure(
        fig,
        CORRELATION_DIRECTORY
        / (
            f"correlation_"
            f"{method}_{suffix}.png"
        ),
    )

    valid = data[VISUALIZATION_FEATURE_COLUMNS].notna().astype(int)
    pair_counts = valid.T.dot(valid)
    pair_counts.to_csv(
        CORRELATION_DIRECTORY / f"pairwise_counts_{suffix}.csv"
    )
    if method == "spearman":
        fig, ax = plt.subplots(figsize=(13, 11))
        sns.heatmap(pair_counts, cmap="Blues", annot=True, fmt="d",
                    square=True, linewidths=0.4, ax=ax)
        ax.set_title(f"Pairwise valid event counts: {title.split(':')[-1].strip()}")
        save_figure(fig, CORRELATION_DIRECTORY / f"pairwise_counts_{suffix}.png")


def plot_all_correlations(
    data,
):
    """
    Plot Pearson and Spearman correlation for
    all data and each protein separately.
    """

    for method in [
        "pearson",
        "spearman",
    ]:
        plot_correlation_heatmap(
            data=data,
            method=method,
            suffix="all",
            title=(
                f"{method.title()} correlation: "
                "all accepted events"
            ),
        )

        for protein in PROTEIN_ORDER:
            protein_data = data[
                data["protein"] == protein
            ]

            if protein_data.empty:
                continue

            plot_correlation_heatmap(
                data=protein_data,
                method=method,
                suffix=protein.lower(),
                title=(
                    f"{method.title()} correlation: "
                    f"{protein}"
                ),
            )

def add_derived_features(
    data,
):
    """
    Add derived features used only for
    visualization or later modeling.
    """

    data = data.copy()

    data["log_dwell_time"] = np.nan

    valid_mask = (
        data["dwell_time_ms"].notna()
        & np.isfinite(
            data["dwell_time_ms"]
        )
        & (
            data["dwell_time_ms"] > 0
        )
    )

    data.loc[
        valid_mask,
        "log_dwell_time",
    ] = np.log10(
        data.loc[
            valid_mask,
            "dwell_time_ms",
        ]
    )

    # Plot the magnitude in the original %/ms unit. Scatter
    # axes apply logarithmic scaling without changing values.
    data["right_slope_magnitude"] = (
        data["right_slope"].abs()
    )

    data["left_slope_magnitude"] = (
        data["left_slope"].abs()
    )

    return data


# =========================================================
# Association between one feature and protein class
# =========================================================

def benjamini_hochberg_adjustment(
    p_values,
):
    """Adjust multiple p-values using Benjamini-Hochberg FDR."""

    p_values = np.asarray(
        p_values,
        dtype=float,
    )

    adjusted = np.full(
        p_values.shape,
        np.nan,
        dtype=float,
    )

    finite_indices = np.flatnonzero(
        np.isfinite(p_values)
    )

    if finite_indices.size == 0:
        return adjusted

    finite_p = p_values[finite_indices]
    order = np.argsort(finite_p)
    ranked_p = finite_p[order]
    n_tests = len(ranked_p)

    ranked_adjusted = (
        ranked_p
        * n_tests
        / np.arange(1, n_tests + 1)
    )

    ranked_adjusted = np.minimum.accumulate(
        ranked_adjusted[::-1]
    )[::-1]

    ranked_adjusted = np.clip(
        ranked_adjusted,
        0.0,
        1.0,
    )

    finite_adjusted = np.empty_like(
        ranked_adjusted
    )
    finite_adjusted[order] = ranked_adjusted
    adjusted[finite_indices] = finite_adjusted

    return adjusted


def calculate_single_feature_association(
    data,
):
    """
    Measure each feature's association with protein class.

    Target coding is HSA=0 and F8=1. Positive correlation
    and Cliff's delta therefore mean that F8 tends to have
    larger feature values than HSA.
    """

    target_map = {
        "HSA": 0,
        "F8": 1,
    }

    results = []

    for feature in VISUALIZATION_FEATURE_COLUMNS:
        subset = (
            data[["protein", feature]]
            .replace([np.inf, -np.inf], np.nan)
            .dropna()
        )

        subset = subset[
            subset["protein"].isin(target_map)
        ].copy()

        subset["target"] = (
            subset["protein"]
            .map(target_map)
            .astype(int)
        )

        hsa = subset.loc[
            subset["protein"] == "HSA",
            feature,
        ].to_numpy(dtype=float)

        f8 = subset.loc[
            subset["protein"] == "F8",
            feature,
        ].to_numpy(dtype=float)

        if (
            len(hsa) < 3
            or len(f8) < 3
            or subset[feature].nunique() < 2
        ):
            continue

        correlation, correlation_p = pointbiserialr(
            subset["target"],
            subset[feature],
        )

        mann_whitney = mannwhitneyu(
            f8,
            hsa,
            alternative="two-sided",
        )

        # Rank-based AUC, with F8 as the positive class.
        all_values = np.concatenate([hsa, f8])
        ranks = rankdata(all_values, method="average")
        f8_rank_sum = ranks[len(hsa):].sum()
        raw_auc = (
            f8_rank_sum
            - len(f8) * (len(f8) + 1) / 2
        ) / (len(f8) * len(hsa))

        # For two independent groups, Cliff's delta equals
        # 2*AUC-1 when F8 is compared with HSA.
        cliffs_delta = 2.0 * raw_auc - 1.0

        results.append({
            "feature": feature,
            "n_hsa": len(hsa),
            "n_f8": len(f8),
            "hsa_median": np.median(hsa),
            "f8_median": np.median(f8),
            "point_biserial_r": correlation,
            "point_biserial_p": correlation_p,
            "raw_auc_f8_positive": raw_auc,
            "separation_auc": max(raw_auc, 1.0 - raw_auc),
            "cliffs_delta_f8_vs_hsa": cliffs_delta,
            "mann_whitney_p": mann_whitney.pvalue,
        })

    results = pd.DataFrame(results)

    if results.empty:
        return results

    results["point_biserial_p_fdr"] = (
        benjamini_hochberg_adjustment(
            results["point_biserial_p"]
        )
    )

    results["mann_whitney_p_fdr"] = (
        benjamini_hochberg_adjustment(
            results["mann_whitney_p"]
        )
    )

    return results.sort_values(
        "separation_auc",
        ascending=False,
    ).reset_index(drop=True)


def save_single_feature_association(
    data,
):
    """Calculate and save one-feature-versus-class statistics."""

    results = calculate_single_feature_association(
        data
    )

    output_path = (
        SINGLE_FEATURE_DIRECTORY
        / "single_feature_class_association.csv"
    )

    results.to_csv(
        output_path,
        index=False,
    )

    return results, output_path


def plot_single_feature_association(
    results,
):
    """Plot exploratory feature-to-class association summaries."""

    if results.empty:
        return

    plot_data = results.copy()
    plot_data["feature_label"] = plot_data["feature"].map(
        FEATURE_LABELS
    ).fillna(plot_data["feature"])

    specifications = [
        {
            "column": "point_biserial_r",
            "filename": "point_biserial_correlation.png",
            "xlabel": "Point-biserial correlation (HSA=0, F8=1)",
            "title": "Single-feature association with protein class",
            "reference": 0.0,
            "limits": (-1.0, 1.0),
            "signed": True,
        },
        {
            "column": "separation_auc",
            "filename": "single_feature_separation_auc.png",
            "xlabel": "In-sample separation AUC",
            "title": "Single-feature HSA–F8 separation",
            "reference": 0.5,
            "limits": (0.45, 1.0),
            "signed": False,
        },
        {
            "column": "cliffs_delta_f8_vs_hsa",
            "filename": "cliffs_delta.png",
            "xlabel": "Cliff's delta (F8 versus HSA)",
            "title": "Single-feature distribution effect size",
            "reference": 0.0,
            "limits": (-1.0, 1.0),
            "signed": True,
        },
    ]

    for specification in specifications:
        column = specification["column"]
        ordered = plot_data.sort_values(
            column,
            ascending=True,
        )

        if specification["signed"]:
            colors = [
                PROTEIN_COLORS["F8"]
                if value > 0
                else PROTEIN_COLORS["HSA"]
                for value in ordered[column]
            ]
        else:
            colors = "tab:purple"

        figure_height = max(
            6.0,
            0.48 * len(ordered),
        )
        fig, ax = plt.subplots(
            figsize=(10, figure_height)
        )

        bars = ax.barh(
            ordered["feature_label"],
            ordered[column],
            color=colors,
            alpha=0.82,
        )

        ax.axvline(
            specification["reference"],
            color="black",
            linestyle="--",
            linewidth=1.2,
        )
        ax.set_xlim(*specification["limits"])
        ax.set_xlabel(specification["xlabel"])
        ax.set_ylabel("Feature")
        ax.set_title(
            specification["title"]
            + " (exploratory; accepted events)"
        )

        ax.bar_label(
            bars,
            labels=[
                f"{value:.3f}"
                for value in ordered[column]
            ],
            padding=3,
            fontsize=8,
        )

        if specification["signed"]:
            ax.text(
                0.01,
                0.01,
                "Negative: HSA tends higher   |   Positive: F8 tends higher",
                transform=ax.transAxes,
                fontsize=8,
                va="bottom",
            )
        else:
            ax.text(
                0.01,
                0.01,
                "Dashed line = chance level; not held-out model performance",
                transform=ax.transAxes,
                fontsize=8,
                va="bottom",
            )

        save_figure(
            fig,
            SINGLE_FEATURE_DIRECTORY
            / specification["filename"],
        )


def create_balanced_analysis_view(data, seed):
    """Return an event-balanced copy; never mutate or write the master table."""

    class_counts = data["protein"].value_counts().reindex(PROTEIN_ORDER)
    if class_counts.isna().any() or (class_counts <= 0).any():
        raise ValueError("Both HSA and F8 are required for balanced visualization")
    target = int(class_counts.min())
    samples = []
    for protein in PROTEIN_ORDER:
        class_data = data[data["protein"] == protein]
        samples.append(class_data.sample(n=target, replace=False, random_state=seed))
    return (
        pd.concat(samples, axis=0)
        .sample(frac=1.0, random_state=seed)
        .reset_index(drop=True)
    )


def calculate_distribution_overlap(data):
    """Estimate HSA/F8 overlap using shared finite-value histogram bins."""

    rows = []
    for feature in VISUALIZATION_FEATURE_COLUMNS:
        subset = data[["protein", feature]].replace([np.inf, -np.inf], np.nan).dropna()
        hsa = subset.loc[subset["protein"] == "HSA", feature].to_numpy(float)
        f8 = subset.loc[subset["protein"] == "F8", feature].to_numpy(float)
        if len(hsa) < 3 or len(f8) < 3:
            continue
        pooled = np.concatenate([hsa, f8])
        edges = np.histogram_bin_edges(pooled, bins="fd")
        if len(edges) < 3:
            edges = np.linspace(float(np.min(pooled)), float(np.max(pooled)), 11)
        h_hist, _ = np.histogram(hsa, bins=edges, density=False)
        f_hist, _ = np.histogram(f8, bins=edges, density=False)
        h_prob = h_hist / h_hist.sum()
        f_prob = f_hist / f_hist.sum()
        rows.append({
            "feature": feature,
            "n_hsa": len(hsa),
            "n_f8": len(f8),
            "histogram_overlap_coefficient": float(np.minimum(h_prob, f_prob).sum()),
            "shared_observed_range_low": max(float(np.min(hsa)), float(np.min(f8))),
            "shared_observed_range_high": min(float(np.max(hsa)), float(np.max(f8))),
        })
    return pd.DataFrame(rows).sort_values("histogram_overlap_coefficient")


def save_general_extreme_event_table(data, output_path):
    """Flag feature-wise 0.1/99.9 percentile events without excluding them."""

    identity = [
        column for column in [
            "protein", "recording_id", "annotation_id", "event_id",
            "source_row_number", "abf_path", "event_start_time_s",
        ] if column in data.columns
    ]
    frames = []
    for feature in SOURCE_FEATURE_COLUMNS:
        finite = data[feature].replace([np.inf, -np.inf], np.nan).dropna()
        if len(finite) < 20:
            continue
        low, high = np.percentile(finite, [0.1, 99.9])
        mask = data[feature].le(low) | data[feature].ge(high)
        flagged = data.loc[mask, identity + [feature]].copy()
        flagged = flagged.rename(columns={feature: "value"})
        flagged["feature"] = feature
        flagged["tail"] = np.where(flagged["value"] <= low, "low", "high")
        flagged["threshold_low_0.1pct"] = low
        flagged["threshold_high_99.9pct"] = high
        frames.append(flagged)
    result = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    result.to_csv(output_path, index=False)
    return result


def calculate_recording_confounding_report(data):
    """Compare overall class effects with each F8 recording versus HSA."""

    overall = calculate_single_feature_association(data).set_index("feature")
    f8_recordings = sorted(
        data.loc[data["protein"] == "F8", "recording_id"].astype(str).unique()
    )
    rows = []
    hsa = data[data["protein"] == "HSA"]
    for recording_id in f8_recordings:
        comparison = pd.concat([
            hsa,
            data[(data["protein"] == "F8") &
                 (data["recording_id"].astype(str) == recording_id)],
        ], ignore_index=True)
        result = calculate_single_feature_association(comparison)
        for row in result.itertuples(index=False):
            rows.append({
                "feature": row.feature,
                "f8_recording_id": recording_id,
                "n_hsa_events": row.n_hsa,
                "n_f8_events": row.n_f8,
                "cliffs_delta_f8_vs_hsa": row.cliffs_delta_f8_vs_hsa,
                "separation_auc": row.separation_auc,
                "mann_whitney_p_fdr_within_comparison": row.mann_whitney_p_fdr,
                "overall_cliffs_delta": overall.loc[row.feature, "cliffs_delta_f8_vs_hsa"],
            })
    detail = pd.DataFrame(rows)
    if detail.empty:
        return detail, pd.DataFrame()
    summary = (
        detail.groupby("feature")
        .agg(
            n_f8_recordings=("f8_recording_id", "nunique"),
            min_recording_delta=("cliffs_delta_f8_vs_hsa", "min"),
            max_recording_delta=("cliffs_delta_f8_vs_hsa", "max"),
            median_recording_delta=("cliffs_delta_f8_vs_hsa", "median"),
            min_recording_separation_auc=("separation_auc", "min"),
            max_recording_separation_auc=("separation_auc", "max"),
        ).reset_index()
    )
    overall_delta = overall["cliffs_delta_f8_vs_hsa"].rename("overall_cliffs_delta")
    summary = summary.merge(overall_delta, left_on="feature", right_index=True, how="left")
    summary["direction_consistent_across_f8_recordings"] = (
        (summary["min_recording_delta"] >= 0) | (summary["max_recording_delta"] <= 0)
    )
    return detail, summary.sort_values("min_recording_separation_auc", ascending=False)


def run_visualization_suite(data, qc=None, include_recording_qc=False):
    """Create one complete natural or balanced visualization suite."""

    save_numeric_summary(data)
    missing_summary = save_missing_summary(data)
    plot_missing_values(missing_summary)
    plot_event_counts(data)
    association_results, _ = save_single_feature_association(data)
    plot_single_feature_association(association_results)
    calculate_distribution_overlap(data).to_csv(
        SUMMARY_DIRECTORY / "hsa_f8_distribution_overlap.csv", index=False
    )
    plot_all_histograms(data)
    plot_all_violin_plots(data)
    plot_all_scatter_plots(data, generate_all=False)
    plot_all_correlations(data)
    if include_recording_qc and qc is not None:
        counts, recording_missing, shape_summary, _ = save_recording_level_summaries(data, qc)
        plot_recording_diagnostics(counts, recording_missing, shape_summary)
        save_and_plot_blockage_consistency(data)
        save_extreme_feature_events(data)
        plot_recording_scatter_panels(data, SCATTER_DIRECTORY / "recording_colored")
    return association_results


# =========================================================
# Main
# =========================================================

def parse_arguments():
    parser = argparse.ArgumentParser(description="Create versioned nanopore QC visualizations")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--extraction-run-directory", type=Path)
    parser.add_argument("--run-id")
    parser.add_argument(
        "--balanced-seeds", type=int, nargs="+", default=[11, 42, 73, 101, 202],
        help="Predeclared random seeds for balanced analysis views",
    )
    return parser.parse_args()


def resolve_extraction_directory(config, requested):
    if requested is not None:
        return requested.resolve()
    candidates = [path for path in Path(config["extraction_output_root"]).iterdir()
                  if path.is_dir() and (path / "run_manifest.json").is_file()]
    if not candidates:
        raise FileNotFoundError("No completed extraction run was found")
    return max(candidates, key=lambda path: path.stat().st_mtime)


def main():
    global PLOT_LOWER_PERCENTILE, PLOT_UPPER_PERCENTILE

    arguments = parse_arguments()
    config = load_project_config(arguments.config)
    extraction_directory = resolve_extraction_directory(
        config, arguments.extraction_run_directory
    )
    feature_path = extraction_directory / "abf_event_features.csv"
    qc_path = extraction_directory / "abf_event_qc.csv"
    extraction_manifest_path = extraction_directory / "run_manifest.json"

    run_id = arguments.run_id or make_run_id(
        config["dataset_version"], prefix="visualization"
    )
    output_directory = require_new_output_directory(
        Path(config["visualization_output_root"]) / run_id
    )
    PLOT_LOWER_PERCENTILE = float(config["plot_lower_percentile"])
    PLOT_UPPER_PERCENTILE = float(config["plot_upper_percentile"])
    configure_plot_style()

    data = load_feature_data(
        feature_path
    )
    qc = load_qc_data(qc_path)

    data = add_derived_features(
        data
    )

    print(
        "Feature table loaded:",
        len(data),
        "events",
    )

    print("\nEvents by protein:")

    print(
        data["protein"]
        .value_counts()
    )

    natural_directory = output_directory / "natural_imbalance"
    configure_output_paths(natural_directory)
    create_output_directories()
    print("\nCreating natural-imbalance analysis...")
    association_results = run_visualization_suite(
        data, qc=qc, include_recording_qc=True
    )
    association_path = SINGLE_FEATURE_DIRECTORY / "single_feature_class_association.csv"
    save_general_extreme_event_table(
        data, SUMMARY_DIRECTORY / "extreme_events_all_features.csv"
    )
    recording_detail, recording_summary = calculate_recording_confounding_report(data)
    recording_detail.to_csv(
        SUMMARY_DIRECTORY / "recording_confounding_feature_detail.csv", index=False
    )
    recording_summary.to_csv(
        SUMMARY_DIRECTORY / "recording_confounding_summary.csv", index=False
    )

    print(
        association_results[
            [
                "feature",
                "point_biserial_r",
                "separation_auc",
                "cliffs_delta_f8_vs_hsa",
                "mann_whitney_p_fdr",
            ]
        ].to_string(index=False)
    )

    print(
        "Single-feature results saved to:",
        association_path,
    )

    balanced_root = output_directory / "balanced_multi_seed"
    stability_frames = []
    selection_frames = []
    print("\nCreating balanced multi-seed analysis views...")
    for seed in arguments.balanced_seeds:
        balanced = create_balanced_analysis_view(data, seed)
        seed_directory = balanced_root / f"seed_{seed}"
        configure_output_paths(seed_directory)
        create_output_directories()
        seed_results = run_visualization_suite(balanced)
        seed_results.insert(0, "seed", seed)
        stability_frames.append(seed_results)
        selected = (
            balanced.groupby(["protein", "recording_id"])
            .size().rename("n_selected").reset_index()
        )
        selected.insert(0, "seed", seed)
        selection_frames.append(selected)

    balanced_root.mkdir(parents=True, exist_ok=True)
    stability = pd.concat(stability_frames, ignore_index=True)
    selections = pd.concat(selection_frames, ignore_index=True)
    stability.to_csv(balanced_root / "balanced_association_by_seed.csv", index=False)
    selections.to_csv(balanced_root / "balanced_recording_selection_by_seed.csv", index=False)
    stability_summary = (
        stability.groupby("feature")
        .agg(
            n_seeds=("seed", "nunique"),
            median_cliffs_delta=("cliffs_delta_f8_vs_hsa", "median"),
            min_cliffs_delta=("cliffs_delta_f8_vs_hsa", "min"),
            max_cliffs_delta=("cliffs_delta_f8_vs_hsa", "max"),
            median_separation_auc=("separation_auc", "median"),
            min_separation_auc=("separation_auc", "min"),
            max_separation_auc=("separation_auc", "max"),
            fdr_significant_seed_fraction=(
                "mann_whitney_p_fdr", lambda values: float((values < 0.05).mean())
            ),
        ).reset_index()
    )
    stability_summary["effect_direction_stable"] = (
        (stability_summary["min_cliffs_delta"] >= 0)
        | (stability_summary["max_cliffs_delta"] <= 0)
    )
    stability_summary.sort_values(
        "min_separation_auc", ascending=False
    ).to_csv(balanced_root / "balanced_stability_summary.csv", index=False)

    print(
        "\nVisualization completed."
    )

    manifest_path = write_run_manifest(
        output_directory / "run_manifest.json",
        {
            "run_id": run_id,
            "stage": "visualization",
            "dataset_version": config["dataset_version"],
            "feature_definition_version": config["feature_definition_version"],
            "input_extraction_run": str(extraction_directory),
            "input_feature_sha256": sha256_file(feature_path),
            "input_qc_sha256": sha256_file(qc_path),
            "input_extraction_manifest": str(extraction_manifest_path),
            "n_events": int(len(data)),
            "recordings_by_protein": data.groupby("protein")["recording_id"].nunique().to_dict(),
            "natural_class_event_counts": data["protein"].value_counts().to_dict(),
            "balanced_target_per_class": int(data["protein"].value_counts().min()),
            "balanced_random_seeds": arguments.balanced_seeds,
            "balanced_sampling": "Independent within-class event sampling without replacement; master dataset unchanged",
            "recording_confounding_limitation": (
                "HSA has one recording, so protein and recording effects cannot be separated"
            ),
            "generate_all_scatter_plots": False,
            "output_directory": str(output_directory),
        },
    )

    print(
        "Figures saved to:",
        output_directory,
    )
    print("Run manifest:", manifest_path)

if __name__ == "__main__":
    main()
