"""Focused natural-data visualization, correlation, PCA, and UMAP workflow."""

import argparse
import json
import os
import platform
from datetime import datetime
from importlib.metadata import version
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/nanopore_matplotlib_cache")
os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/nanopore_numba_cache")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler

from project_config import require_new_output_directory, sha256_file


BASE_FEATURES = [
    "dwell_time_ms", "blockage_percent", "blockage_mean", "blockage_sd",
    "auc", "skewness", "kurtosis", "fwhm_width", "fwhm_drop",
    "rise_time", "fall_time", "left_slope", "right_slope",
]
SEGMENT_FEATURES = [f"segment_{number:02d}_blockage_mean" for number in range(1, 11)]
ALL_FEATURES = BASE_FEATURES + SEGMENT_FEATURES
MAIN_DISTRIBUTION_FEATURES = [
    "dwell_time_ms", "blockage_mean", "blockage_sd", "auc", "fwhm_width",
]


def load_and_validate(feature_path):
    feature_path = Path(feature_path).resolve()
    data = pd.read_csv(feature_path)
    required = {"protein", "recording_id", "event_id", *ALL_FEATURES}
    missing = sorted(required - set(data.columns))
    if missing:
        raise ValueError(f"Feature table is missing columns: {missing}")
    if data.empty:
        raise ValueError("Feature table contains no accepted events")
    if data[["protein", "recording_id", "event_id"]].isna().any().any():
        raise ValueError("Required provenance contains missing values")
    if data.duplicated(["recording_id", "event_id"]).any():
        raise ValueError("Duplicate accepted event identity")
    unexpected = sorted(set(data.protein.astype(str)) - {"HSA", "F8"})
    if unexpected:
        raise ValueError(f"Unexpected protein labels: {unexpected}")
    legacy_segments = sorted(
        column for column in data
        if column.startswith("segment_")
        and column not in SEGMENT_FEATURES
        and column not in {"segmentation_qc_pass", "segmentation_qc_reason"}
        and column not in {"minimum_segment_points", "insufficient_segment_count"}
    )
    if legacy_segments:
        raise ValueError(
            "Input contains non-compact segmentation features: "
            f"{legacy_segments[:5]}"
        )
    numeric = data[ALL_FEATURES].apply(pd.to_numeric, errors="coerce")
    numeric = numeric.replace([np.inf, -np.inf], np.nan)
    return data, numeric


def save_distributions(data, numeric, output):
    output.mkdir()
    plot_data = pd.concat([data[["protein", "recording_id"]], numeric], axis=1)
    for feature in MAIN_DISTRIBUTION_FEATURES:
        values = plot_data[["protein", feature]].dropna()
        central_parts = []
        for _, group in values.groupby("protein", sort=False):
            low, high = group[feature].quantile([0.01, 0.99])
            central_parts.append(group[group[feature].between(low, high)])
        central = pd.concat(central_parts, ignore_index=True)

        fig, ax = plt.subplots(figsize=(8, 5.5))
        for protein, color in (("HSA", "tab:blue"), ("F8", "tab:orange")):
            group = central.loc[central.protein == protein, feature]
            ax.hist(group, bins=50, density=True, histtype="stepfilled",
                    linewidth=1.5, alpha=.35,
                    label=f"{protein} (n={len(group):,})", color=color)
        ax.set(xlabel=feature, ylabel="Density",
               title=f"{feature}: central 1–99% distribution")
        ax.legend()
        fig.tight_layout()
        fig.savefig(output / f"distribution_{feature}.png", dpi=200)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(7, 5.5))
        sns.violinplot(data=central, x="protein", y=feature, order=["HSA", "F8"],
                       inner="quartile", cut=0, ax=ax)
        ax.set_title(f"{feature}: central 1–99%")
        fig.tight_layout()
        fig.savefig(output / f"violin_{feature}.png", dpi=200)
        plt.close(fig)


def save_scatter_plots(data, numeric, output):
    """Save pairwise central-range scatter plots for the five main features."""
    output.mkdir(exist_ok=True)
    plot_data = pd.concat([data[["protein", "recording_id"]], numeric], axis=1)
    for first_index, x_feature in enumerate(MAIN_DISTRIBUTION_FEATURES):
        for y_feature in MAIN_DISTRIBUTION_FEATURES[first_index + 1:]:
            values = plot_data[["protein", x_feature, y_feature]].dropna()
            central_parts = []
            for _, group in values.groupby("protein", sort=False):
                x_low, x_high = group[x_feature].quantile([.01, .99])
                y_low, y_high = group[y_feature].quantile([.01, .99])
                central_parts.append(group[
                    group[x_feature].between(x_low, x_high)
                    & group[y_feature].between(y_low, y_high)
                ])
            central = pd.concat(central_parts, ignore_index=True)
            fig, ax = plt.subplots(figsize=(7.5, 6))
            sns.scatterplot(
                data=central, x=x_feature, y=y_feature, hue="protein",
                hue_order=["HSA", "F8"], palette={"HSA": "tab:blue", "F8": "tab:orange"},
                s=13, alpha=.4, linewidth=0, ax=ax,
            )
            ax.set_title(f"{x_feature} vs {y_feature} — within-protein central 1–99%")
            ax.legend(markerscale=2)
            fig.tight_layout()
            fig.savefig(output / f"scatter_{x_feature}_vs_{y_feature}.png", dpi=200)
            plt.close(fig)

def save_correlations(data, numeric, output):
    output.mkdir()
    membership = {"all": pd.Series(True, index=data.index)}
    membership.update({protein.lower(): data.protein.eq(protein) for protein in ("HSA", "F8")})
    for subset, mask in membership.items():
        values = numeric.loc[mask]
        valid = values.notna().astype(int)
        valid.T.dot(valid).to_csv(output / f"pairwise_valid_counts_{subset}.csv")
        for method in ("pearson", "spearman"):
            correlation = values.corr(method=method, min_periods=3)
            correlation.to_csv(output / f"{method}_correlation_{subset}.csv")
            fig, ax = plt.subplots(figsize=(22, 20))
            sns.heatmap(correlation, cmap="coolwarm", center=0, vmin=-1, vmax=1,
                        square=True, linewidths=.25, ax=ax, annot=True, fmt=".2f",
                        annot_kws={"fontsize": 5.5},
                        cbar_kws={"label": f"{method.title()} correlation"})
            ax.set_title(f"{method.title()} correlation — {subset}")
            fig.tight_layout()
            fig.savefig(output / f"{method}_correlation_{subset}.png", dpi=200)
            plt.close(fig)

            segment_correlation = correlation.loc[SEGMENT_FEATURES, SEGMENT_FEATURES]
            segment_correlation.to_csv(
                output / f"{method}_correlation_segments_{subset}.csv"
            )
            fig, ax = plt.subplots(figsize=(11, 9.5))
            sns.heatmap(
                segment_correlation,
                cmap="coolwarm",
                center=0,
                vmin=-1,
                vmax=1,
                square=True,
                linewidths=.4,
                annot=True,
                fmt=".2f",
                annot_kws={"fontsize": 8},
                ax=ax,
                cbar_kws={"label": f"{method.title()} correlation"},
            )
            ax.set_title(f"{method.title()} correlation — segmentation — {subset}")
            fig.tight_layout()
            fig.savefig(
                output / f"{method}_correlation_segments_{subset}.png", dpi=200
            )
            plt.close(fig)

            base_correlation = correlation.loc[BASE_FEATURES, BASE_FEATURES]
            base_correlation.to_csv(
                output / f"{method}_correlation_existing_{subset}.csv"
            )
            fig, ax = plt.subplots(figsize=(14, 12))
            sns.heatmap(
                base_correlation, cmap="coolwarm", center=0, vmin=-1, vmax=1,
                square=True, linewidths=.35, annot=True, fmt=".2f",
                annot_kws={"fontsize": 7}, ax=ax,
                cbar_kws={"label": f"{method.title()} correlation"},
            )
            ax.set_title(f"{method.title()} correlation — existing features — {subset}")
            fig.tight_layout()
            fig.savefig(
                output / f"{method}_correlation_existing_{subset}.png", dpi=200
            )
            plt.close(fig)


def prepare_embedding_matrix(numeric):
    features = list(numeric.columns)
    imputer = SimpleImputer(strategy="median")
    imputed = imputer.fit_transform(numeric)
    scaler = RobustScaler()
    matrix = scaler.fit_transform(imputed)
    parameters = pd.DataFrame({
        "feature": features,
        "median_imputation": imputer.statistics_,
        "robust_center": scaler.center_,
        "robust_scale_iqr": scaler.scale_,
        "n_missing": numeric.isna().sum().to_numpy(),
    })
    color_values = pd.DataFrame(imputed, columns=features, index=numeric.index)
    return matrix, color_values, parameters


def central_limits(values):
    low, high = np.quantile(values, [0.01, 0.99])
    padding = (high - low) * .03
    if not np.isfinite(padding) or padding == 0:
        padding = 1.0
    return low - padding, high + padding


def save_embedding_view(coordinates, x, y, color, output, title, categorical=False):
    for central in (False, True):
        fig, ax = plt.subplots(figsize=(8, 6.5))
        if categorical:
            sns.scatterplot(data=coordinates, x=x, y=y, hue=color, s=12,
                            alpha=.45, linewidth=0, ax=ax)
            ax.legend(markerscale=2, loc="best")
        else:
            values = pd.to_numeric(coordinates[color], errors="coerce")
            finite = values[np.isfinite(values)]
            low, high = np.quantile(finite, [.01, .99])
            points = ax.scatter(coordinates[x], coordinates[y], c=values,
                                cmap="viridis", vmin=low, vmax=high, s=9,
                                alpha=.45, linewidths=0, rasterized=True)
            fig.colorbar(points, ax=ax, label=color)
        if central:
            ax.set_xlim(central_limits(coordinates[x]))
            ax.set_ylim(central_limits(coordinates[y]))
        suffix = "central 1–99%" if central else "full"
        ax.set_title(f"{title} — {suffix}")
        fig.tight_layout()
        fig.savefig(output / f"{title.lower().replace(' ', '_')}_{color}_{'central' if central else 'full'}.png",
                    dpi=200, bbox_inches="tight")
        plt.close(fig)


def save_plain_embedding(coordinates, x, y, output, title):
    for central in (False, True):
        fig, ax = plt.subplots(figsize=(8, 6.5))
        ax.scatter(coordinates[x], coordinates[y], color="0.25", s=9,
                   alpha=.35, linewidths=0, rasterized=True)
        if central:
            ax.set_xlim(central_limits(coordinates[x]))
            ax.set_ylim(central_limits(coordinates[y]))
        suffix = "central 1–99%" if central else "full"
        ax.set(xlabel=x, ylabel=y, title=f"{title} — no color — {suffix}")
        fig.tight_layout()
        fig.savefig(output / f"{title.lower().replace(' ', '_')}_plain_{'central' if central else 'full'}.png",
                    dpi=200, bbox_inches="tight")
        plt.close(fig)


def save_pca(data, matrix, color_values, output):
    output.mkdir()
    pca = PCA().fit(matrix)
    transformed = pca.transform(matrix)
    scores = data[["protein", "recording_id", "event_id"]].copy()
    scores[["PC1", "PC2", "PC3"]] = transformed[:, :3]
    scores.to_csv(output / "pca_scores.csv", index=False)
    variance = pd.DataFrame({
        "component": np.arange(1, len(ALL_FEATURES) + 1),
        "explained_variance_ratio": pca.explained_variance_ratio_,
        "cumulative_explained_variance": np.cumsum(pca.explained_variance_ratio_),
    })
    variance.to_csv(output / "explained_variance.csv", index=False)
    pd.DataFrame(pca.components_.T, index=ALL_FEATURES,
                 columns=[f"PC{i + 1}" for i in range(len(ALL_FEATURES))]).to_csv(
                     output / "pca_loadings.csv")
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(variance.component, variance.explained_variance_ratio, "o-")
    ax.set(xlabel="Principal component", ylabel="Explained variance ratio",
           title="PCA scree plot")
    fig.tight_layout(); fig.savefig(output / "pca_scree.png", dpi=200); plt.close(fig)

    colored = pd.concat([scores, color_values.reset_index(drop=True)], axis=1)
    save_plain_embedding(scores, "PC1", "PC2", output, "PCA PC1 PC2")
    fig = plt.figure(figsize=(9, 7.5))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(scores.PC1, scores.PC2, scores.PC3, color="#2C7FB8", s=7,
               alpha=.25, linewidths=0, rasterized=True)
    ax.set(xlabel="PC1", ylabel="PC2", zlabel="PC3", title="PCA PC1–PC3 — no color")
    fig.tight_layout()
    fig.savefig(output / "pca_pc1_pc2_pc3_plain.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    fig = plt.figure(figsize=(9, 7.5))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(scores.PC1, scores.PC2, scores.PC3, color="#2C7FB8", s=7,
               alpha=.25, linewidths=0, rasterized=True)
    ax.set_xlim(central_limits(scores.PC1))
    ax.set_ylim(central_limits(scores.PC2))
    ax.set_zlim(central_limits(scores.PC3))
    ax.set(xlabel="PC1", ylabel="PC2", zlabel="PC3",
           title="PCA PC1–PC3 — no color — central 1–99%")
    fig.tight_layout()
    fig.savefig(output / "pca_pc1_pc2_pc3_plain_central.png", dpi=200,
                bbox_inches="tight")
    plt.close(fig)
    save_embedding_view(colored, "PC1", "PC2", "protein", output, "PCA PC1 PC2", True)
    save_embedding_view(colored, "PC1", "PC2", "recording_id", output, "PCA PC1 PC2", True)
    for feature in ALL_FEATURES:
        save_embedding_view(colored, "PC1", "PC2", feature, output, "PCA PC1 PC2")
    return variance


def save_umap(data, matrix, color_values, output, seed, n_neighbors, min_dist):
    from umap import UMAP
    output.mkdir()
    model = UMAP(n_components=2, n_neighbors=n_neighbors, min_dist=min_dist,
                 metric="euclidean", random_state=seed, transform_seed=seed)
    embedding = model.fit_transform(matrix)
    coordinates = data[["protein", "recording_id", "event_id"]].copy()
    coordinates[["UMAP1", "UMAP2"]] = embedding
    coordinates.to_csv(output / "umap_coordinates.csv", index=False)
    colored = pd.concat([coordinates, color_values.reset_index(drop=True)], axis=1)
    save_plain_embedding(coordinates, "UMAP1", "UMAP2", output, "UMAP")
    save_embedding_view(colored, "UMAP1", "UMAP2", "protein", output, "UMAP", True)
    save_embedding_view(colored, "UMAP1", "UMAP2", "recording_id", output, "UMAP", True)
    for feature in ALL_FEATURES:
        save_embedding_view(colored, "UMAP1", "UMAP2", feature, output, "UMAP")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--umap-neighbors", type=int, default=30)
    parser.add_argument("--umap-min-dist", type=float, default=.1)
    args = parser.parse_args(argv)

    source = args.input.resolve()
    extraction_manifest_path = source.parent / "run_manifest.json"
    if not extraction_manifest_path.is_file():
        raise FileNotFoundError("Extraction run manifest is required")
    extraction_manifest = json.loads(extraction_manifest_path.read_text())
    declared = extraction_manifest.get("output_files", {}).get("feature_path")
    if declared and Path(declared).resolve() != source:
        raise ValueError("Feature table does not match its extraction manifest")

    output = require_new_output_directory(args.output.resolve())
    distributions = output / "distributions"
    scatter_directory = output / "scatter_plots"
    correlations = output / "correlations"
    pca_directory = output / "pca"
    umap_directory = output / "umap"
    data, numeric = load_and_validate(source)
    pd.DataFrame({"feature": ALL_FEATURES,
                  "n_missing": numeric.isna().sum().to_numpy(),
                  "missing_fraction": numeric.isna().mean().to_numpy(),
                  "minimum": numeric.min().to_numpy(),
                  "median": numeric.median().to_numpy(),
                  "maximum": numeric.max().to_numpy()}).to_csv(
                      output / "feature_quality_summary.csv", index=False)
    data.groupby(["protein", "recording_id"], as_index=False).size().rename(
        columns={"size": "n_events"}).to_csv(output / "event_counts.csv", index=False)
    save_distributions(data, numeric, distributions)
    save_scatter_plots(data, numeric, scatter_directory)
    save_correlations(data, numeric, correlations)
    matrix, color_values, parameters = prepare_embedding_matrix(numeric)
    parameters.to_csv(output / "embedding_preprocessing.csv", index=False)
    variance = save_pca(data, matrix, color_values, pca_directory)
    save_umap(data, matrix, color_values, umap_directory, args.random_seed,
              args.umap_neighbors, args.umap_min_dist)

    report = (
        "# Feature exploration\n\n"
        f"Input accepted events: {len(data):,}; HSA={int((data.protein == 'HSA').sum()):,}; "
        f"F8={int((data.protein == 'F8').sum()):,}.\n\n"
        f"Analyzed {len(ALL_FEATURES)} features: 13 existing features and 10 compact "
        "segment blockage means. Distribution plots use five predeclared main features. "
        "Pearson and Spearman correlations cover every analyzed feature. PCA and UMAP "
        "use median imputation followed by RobustScaler fitted to the full exploratory "
        "dataset. Feature-color plots use original units after median imputation.\n\n"
        f"PC1 explains {variance.explained_variance_ratio.iloc[0]:.2%}; PC1–PC2 explain "
        f"{variance.explained_variance_ratio.iloc[:2].sum():.2%}.\n\n"
        "This is descriptive exploratory analysis. No train/test split, balancing, "
        "classifier fitting, or performance evaluation was performed. Protein remains "
        "confounded with recording because HSA has one recording.\n"
    )
    (output / "REPORT.md").write_text(report, encoding="utf-8")

    generated = {}
    for path in sorted(p for p in output.rglob("*") if p.is_file()):
        generated[str(path.relative_to(output))] = {
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
    manifest = {
        "status": "completed",
        "run_type": "feature_exploration_no_training",
        "created_at": datetime.now().astimezone().isoformat(),
        "input_path": str(source),
        "input_sha256": sha256_file(source),
        "extraction_manifest_path": str(extraction_manifest_path),
        "extraction_manifest_sha256": sha256_file(extraction_manifest_path),
        "dataset_version": extraction_manifest.get("dataset_version"),
        "feature_definition_version": extraction_manifest.get("feature_definition_version"),
        "features": ALL_FEATURES,
        "main_distribution_features": MAIN_DISTRIBUTION_FEATURES,
        "scatter_features": MAIN_DISTRIBUTION_FEATURES,
        "preprocessing": ["replace_inf_with_nan", "median_imputation", "RobustScaler"],
        "fit_scope": "full_natural_exploratory_dataset",
        "umap": {"n_neighbors": args.umap_neighbors, "min_dist": args.umap_min_dist,
                 "metric": "euclidean", "random_seed": args.random_seed},
        "training": False,
        "events_by_protein": data.protein.value_counts().to_dict(),
        "recordings_by_protein": data.groupby("protein").recording_id.nunique().to_dict(),
        "environment": {"python": platform.python_version(),
                        **{name: version(name) for name in (
                            "numpy", "pandas", "scikit-learn", "umap-learn",
                            "matplotlib", "seaborn")}},
        "generated_files": generated,
    }
    (output / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
