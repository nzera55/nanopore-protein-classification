"""Add V9 segmentation-feature distributions to an existing visualization run."""

import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


SEGMENT_COLUMNS = [f"segment_{index:02d}_blockage_mean" for index in range(1, 11)]
SEEDS = [11, 42, 73, 101, 202]
PROTEIN_ORDER = ["HSA", "F8"]
COLORS = {"HSA": "tab:blue", "F8": "tab:orange"}


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature-path", type=Path, required=True)
    parser.add_argument("--visualization-run", type=Path, required=True)
    return parser.parse_args()


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save_figure(fig, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def balanced_view(data, seed):
    target = int(data["protein"].value_counts().min())
    views = []
    for protein in PROTEIN_ORDER:
        group = data[data["protein"] == protein]
        views.append(group.sample(n=target, replace=False, random_state=seed))
    return pd.concat(views).sample(frac=1.0, random_state=seed).reset_index(drop=True)


def calculate_limits(data):
    limits = {}
    for column in SEGMENT_COLUMNS:
        values = data[column].replace([np.inf, -np.inf], np.nan).dropna()
        limits[column] = {
            "full_low": float(values.min()),
            "full_high": float(values.max()),
            "central_low": float(values.quantile(0.01)),
            "central_high": float(values.quantile(0.99)),
        }
    return limits


def plot_histogram_grid(data, limits, output_path, central, title):
    fig, axes = plt.subplots(2, 5, figsize=(20, 8), sharey=False)
    for index, (ax, column) in enumerate(zip(axes.flat, SEGMENT_COLUMNS), start=1):
        bound = limits[column]
        low = bound["central_low"] if central else bound["full_low"]
        high = bound["central_high"] if central else bound["full_high"]
        bins = np.linspace(low, high, 41)
        for protein in PROTEIN_ORDER:
            values = data.loc[data["protein"] == protein, column].dropna()
            if central:
                values = values[values.between(low, high, inclusive="both")]
            ax.hist(
                values, bins=bins, density=True, histtype="stepfilled",
                alpha=0.28, linewidth=1.2, color=COLORS[protein],
                edgecolor=COLORS[protein], label=f"{protein} (n={len(values)})",
            )
        ax.set_title(f"Segment {index}")
        ax.set_xlabel("Mean blockage (%)")
        ax.set_ylabel("Density")
        ax.legend(fontsize=7)
    fig.suptitle(title, fontsize=16, fontweight="bold")
    save_figure(fig, output_path)


def long_segment_data(data):
    return data[["protein", "recording_id", *SEGMENT_COLUMNS]].melt(
        id_vars=["protein", "recording_id"], var_name="feature", value_name="blockage_mean"
    ).assign(
        segment=lambda frame: frame["feature"].str.extract(r"segment_(\d+)")[0].astype(int)
    )


def plot_violin_grid(data, limits, output_path, title):
    fig, axes = plt.subplots(2, 5, figsize=(20, 9))
    for index, (ax, column) in enumerate(zip(axes.flat, SEGMENT_COLUMNS), start=1):
        low, high = limits[column]["central_low"], limits[column]["central_high"]
        subset = data[["protein", column]].dropna()
        sns.violinplot(
            data=subset, x="protein", y=column, hue="protein", order=PROTEIN_ORDER,
            hue_order=PROTEIN_ORDER, palette=COLORS, cut=0, inner="quartile",
            density_norm="width", legend=False, ax=ax,
        )
        ax.set_ylim(low, high)
        ax.set_title(f"Segment {index}")
        ax.set_xlabel("")
        ax.set_ylabel("Mean blockage (%)")
    fig.suptitle(title + " (central 1–99% display)", fontsize=16, fontweight="bold")
    save_figure(fig, output_path)


def save_summary(data, path, view, seed=None):
    rows = []
    for index, column in enumerate(SEGMENT_COLUMNS, start=1):
        for protein in PROTEIN_ORDER:
            values = data.loc[data["protein"] == protein, column].dropna()
            rows.append({
                "view": view, "seed": seed, "protein": protein, "segment": index,
                "feature": column, "n_events_total": int((data["protein"] == protein).sum()),
                "n_valid": int(len(values)), "n_missing": int((data.loc[data["protein"] == protein, column]).isna().sum()),
                "median": float(values.median()), "q25": float(values.quantile(0.25)),
                "q75": float(values.quantile(0.75)), "mean": float(values.mean()),
                "sd": float(values.std(ddof=1)),
            })
    result = pd.DataFrame(rows)
    result.to_csv(path, index=False)
    return result


def plot_recording_medians(data, output_path):
    long = long_segment_data(data).dropna()
    medians = (
        long.groupby(["protein", "recording_id", "segment"])["blockage_mean"]
        .median().reset_index()
    )
    fig, axes = plt.subplots(1, 2, figsize=(15, 5), sharey=True)
    palette = dict(zip(
        sorted(medians["recording_id"].astype(str).unique()),
        sns.color_palette("tab10", medians["recording_id"].nunique()),
    ))
    for ax, protein in zip(axes, PROTEIN_ORDER):
        subset = medians[medians["protein"] == protein]
        for recording, values in subset.groupby("recording_id"):
            ax.plot(values["segment"], values["blockage_mean"], marker="o",
                    label=str(recording), color=palette[str(recording)])
        ax.set_title(protein)
        ax.set_xticks(range(1, 11))
        ax.set_xlabel("Segment")
        ax.legend(title="Recording")
    axes[0].set_ylabel("Median segment blockage mean (%)")
    fig.suptitle("Recording-level segmentation profiles", fontweight="bold")
    save_figure(fig, output_path)


def plot_seed_stability(summaries, output_path):
    balanced = summaries[summaries["view"] == "balanced"]
    fig, axes = plt.subplots(1, 2, figsize=(15, 5), sharey=True)
    for ax, protein in zip(axes, PROTEIN_ORDER):
        subset = balanced[balanced["protein"] == protein]
        for seed, values in subset.groupby("seed"):
            ax.plot(values["segment"], values["median"], alpha=0.55, linewidth=1,
                    label=f"seed {int(seed)}")
        aggregate = subset.groupby("segment")["median"].median()
        ax.plot(aggregate.index, aggregate.values, color=COLORS[protein], marker="o",
                linewidth=2.5, label="median across seeds")
        ax.set_title(protein)
        ax.set_xticks(range(1, 11))
        ax.set_xlabel("Segment")
        ax.legend(fontsize=8)
    axes[0].set_ylabel("Median blockage mean (%)")
    fig.suptitle("Balanced-view segmentation profile stability across seeds", fontweight="bold")
    save_figure(fig, output_path)


def main():
    args = parse_arguments()
    feature_path = args.feature_path.resolve()
    run_path = args.visualization_run.resolve()
    output = run_path / "segmentation_distributions"
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {output}")
    output.mkdir(parents=True)

    data = pd.read_csv(feature_path, low_memory=False)
    required = {"protein", "recording_id", *SEGMENT_COLUMNS}
    missing = sorted(required - set(data.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    if set(data["protein"]) != set(PROTEIN_ORDER):
        raise ValueError("Expected exactly HSA and F8 labels")
    data[SEGMENT_COLUMNS] = data[SEGMENT_COLUMNS].replace([np.inf, -np.inf], np.nan)
    limits = calculate_limits(data)
    sns.set_theme(style="whitegrid", context="notebook")

    natural_dir = output / "natural_imbalance"
    natural_dir.mkdir()
    plot_histogram_grid(data, limits, natural_dir / "segment_distributions_full.png", False,
                        "Segmentation feature distributions: natural imbalance (full range)")
    plot_histogram_grid(data, limits, natural_dir / "segment_distributions_central.png", True,
                        "Segmentation feature distributions: natural imbalance (central 1–99%)")
    plot_violin_grid(data, limits, natural_dir / "segment_violins_central.png",
                     "Segmentation feature distributions: natural imbalance")
    plot_recording_medians(data, natural_dir / "recording_segment_medians.png")
    summaries = [save_summary(data, natural_dir / "segment_distribution_summary.csv", "natural")]

    balanced_root = output / "balanced_multi_seed"
    selections = []
    for seed in SEEDS:
        view = balanced_view(data, seed)
        seed_dir = balanced_root / f"seed_{seed}"
        seed_dir.mkdir(parents=True)
        plot_histogram_grid(view, limits, seed_dir / "segment_distributions_full.png", False,
                            f"Balanced segmentation distributions: seed {seed} (full range)")
        plot_histogram_grid(view, limits, seed_dir / "segment_distributions_central.png", True,
                            f"Balanced segmentation distributions: seed {seed} (central 1–99%)")
        plot_violin_grid(view, limits, seed_dir / "segment_violins_central.png",
                         f"Balanced segmentation distributions: seed {seed}")
        summaries.append(save_summary(
            view, seed_dir / "segment_distribution_summary.csv", "balanced", seed
        ))
        selected = view.groupby(["protein", "recording_id"]).size().rename("n_selected").reset_index()
        selected.insert(0, "seed", seed)
        selections.append(selected)

    all_summaries = pd.concat(summaries, ignore_index=True)
    all_summaries.to_csv(output / "all_distribution_summaries.csv", index=False)
    pd.concat(selections, ignore_index=True).to_csv(
        balanced_root / "balanced_recording_selection_by_seed.csv", index=False
    )
    plot_seed_stability(all_summaries, balanced_root / "balanced_segment_profile_stability.png")
    pd.DataFrame([
        {"feature": column, **limits[column]} for column in SEGMENT_COLUMNS
    ]).to_csv(output / "shared_axis_limits.csv", index=False)

    manifest = {
        "run_id": "segmentation_distributions_v9_20260911",
        "created_at": datetime.now().astimezone().isoformat(),
        "stage": "segmentation_distribution_visualization_addition",
        "parent_visualization_run": str(run_path),
        "input_feature_path": str(feature_path),
        "input_feature_sha256": sha256_file(feature_path),
        "dataset_version": "v9_centered_zero_duration_reconstruction",
        "feature_definition_version": "2.1.0",
        "segment_features": SEGMENT_COLUMNS,
        "natural_event_counts": data["protein"].value_counts().to_dict(),
        "balanced_target_per_class": int(data["protein"].value_counts().min()),
        "balanced_random_seeds": SEEDS,
        "balanced_sampling": "Independent within-class event sampling without replacement; master dataset unchanged",
        "axis_policy": "Natural-data full range and pooled 1st-99th percentile limits shared across natural and all balanced seeds",
        "recording_confounding_limitation": "HSA has one recording; protein and recording effects cannot be separated",
    }
    with (output / "run_manifest.json").open("x", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, ensure_ascii=False)
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
