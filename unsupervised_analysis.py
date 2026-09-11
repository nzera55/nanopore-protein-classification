"""Unified PCA/UMAP analysis for time-aligned nanopore feature tables."""

import argparse
import json
import os
import platform
from itertools import cycle
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
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.cluster import AgglomerativeClustering, DBSCAN, KMeans
from sklearn.mixture import GaussianMixture
from sklearn.impute import SimpleImputer
from sklearn.manifold import trustworthiness
from sklearn.metrics import (adjusted_rand_score, calinski_harabasz_score,
                             davies_bouldin_score, pairwise_distances,
                             silhouette_score)
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import RobustScaler

from project_config import require_new_output_directory, sha256_file


def validate_extraction_provenance(run, manifest_path, feature_path):
    """Validate that a manifest and feature table belong to one extraction run."""
    if manifest_path.parent != run or feature_path.parent != run:
        raise ValueError("Manifest and feature CSV must come from the same extraction run")
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if payload.get("run_type") not in (None, "event_extraction"):
        raise ValueError("Input manifest is not an event-extraction manifest")
    declared = payload.get("output_files", {}).get("feature_path")
    if declared and Path(declared).resolve() != feature_path.resolve():
        raise ValueError("Feature CSV does not match the extraction manifest")
    return payload


def run_pca(matrix, metadata, feature_names):
    """Fit PCA while retaining the legacy reusable API."""
    n_components = min(matrix.shape)
    model = PCA(n_components=n_components).fit(matrix)
    transformed = model.transform(matrix)
    coordinates = metadata.copy()
    for index in range(min(5, transformed.shape[1])):
        coordinates[f"PC{index + 1}"] = transformed[:, index]
    variance = pd.DataFrame({
        "component": np.arange(1, n_components + 1),
        "explained_variance_ratio": model.explained_variance_ratio_,
        "cumulative_explained_variance": np.cumsum(model.explained_variance_ratio_),
    })
    loadings = pd.DataFrame(model.components_.T, index=feature_names,
                            columns=[f"PC{i + 1}" for i in range(n_components)])
    contributions = loadings.iloc[:, :5].pow(2)
    contributions = contributions.div(contributions.sum(axis=0), axis=1)
    thresholds = {str(int(t * 100)): int(np.searchsorted(
        variance.cumulative_explained_variance, t) + 1) for t in (.8, .9, .95)}
    return model, coordinates, variance, loadings, contributions, thresholds


def run_umap(matrix, metadata, seed, n_neighbors, min_dist):
    """Fit a reproducible two-dimensional UMAP embedding."""
    from umap import UMAP
    if len(matrix) < 3:
        raise ValueError("UMAP requires at least three events")
    model = UMAP(n_components=2, n_neighbors=min(n_neighbors, len(matrix) - 1),
                 min_dist=min_dist, metric="euclidean", random_state=seed,
                 transform_seed=seed)
    embedding = model.fit_transform(matrix)
    coordinates = metadata.copy()
    coordinates[["UMAP1", "UMAP2"]] = embedding
    return model, coordinates


def _central_limits(series, lower=.01, upper=.99):
    low, high = pd.Series(series).quantile([lower, upper])
    padding = (high - low) * .03
    if not np.isfinite(padding) or padding == 0:
        padding = 1.
    return float(low - padding), float(high + padding)


def save_feature_color_views(coordinates, color_values, feature_names, directory,
                             prefix, axes, explained_variance=None,
                             elevation=24, azimuth=-58):
    """Save full/central embedding views colored in original feature units."""
    color_directory = directory / "color_by_feature"
    color_directory.mkdir(exist_ok=True)
    is_3d = len(axes) == 3
    for index, feature in enumerate(feature_names):
        values = pd.to_numeric(color_values[feature], errors="coerce").to_numpy(dtype=float)
        finite = values[np.isfinite(values)]
        color_min, color_max = np.quantile(finite, [.01, .99]) if finite.size else (None, None)
        if not np.isfinite(color_min) or not np.isfinite(color_max) or color_min == color_max:
            color_min = color_max = None
        for central in (False, True):
            fig = plt.figure(figsize=(9, 7))
            ax = fig.add_subplot(111, projection="3d") if is_3d else fig.add_subplot(111)
            kwargs = dict(c=values, cmap="viridis", vmin=color_min, vmax=color_max,
                          s=10, alpha=.45, linewidths=0, rasterized=True)
            if is_3d:
                points = ax.scatter(coordinates[axes[0]], coordinates[axes[1]],
                                    coordinates[axes[2]], **kwargs)
                ax.set(xlabel=axes[0], ylabel=axes[1], zlabel=axes[2])
                ax.view_init(elev=elevation, azim=azimuth)
                if central:
                    ax.set_xlim(_central_limits(coordinates[axes[0]]))
                    ax.set_ylim(_central_limits(coordinates[axes[1]]))
                    ax.set_zlim(_central_limits(coordinates[axes[2]]))
            else:
                points = ax.scatter(coordinates[axes[0]], coordinates[axes[1]], **kwargs)
                ax.set(xlabel=axes[0], ylabel=axes[1])
                if central:
                    ax.set_xlim(_central_limits(coordinates[axes[0]]))
                    ax.set_ylim(_central_limits(coordinates[axes[1]]))
            view = "central" if central else "full"
            suffix = " (central 1–99% view)" if central else ""
            variance_text = ""
            if explained_variance:
                variance_text = " | " + ", ".join(
                    f"{axis}={explained_variance[axis]:.1%}" for axis in axes)
            ax.set_title(f"{prefix} colored by {feature}{variance_text} | "
                         f"n={len(coordinates)}{suffix}")
            colorbar = fig.colorbar(points, ax=ax, pad=.1 if is_3d else .02)
            colorbar.set_label(feature)
            fig.tight_layout()
            axes_name = "_".join(axis.lower() for axis in axes)
            fig.savefig(color_directory / f"{prefix.lower()}_{axes_name}_by_{feature}_{view}.png",
                        dpi=180, bbox_inches="tight")
            plt.close(fig)


FEATURE_SETS = {
    "all_calculated_features": [
        "dwell_time_ms", "blockage_percent", "blockage_mean", "blockage_sd",
        "auc", "skewness", "kurtosis", "fwhm_width", "fwhm_drop",
        "rise_time", "fall_time", "left_slope", "right_slope",
    ],
    "set_a_blockage_mean": ["blockage_mean", "dwell_time_ms", "fwhm_width", "blockage_sd", "skewness", "kurtosis"],
    "set_b_blockage_percent": ["blockage_percent", "dwell_time_ms", "fwhm_width", "blockage_sd", "skewness", "kurtosis"],
    "set_c_blockage_mean_auc": ["blockage_mean", "dwell_time_ms", "fwhm_width", "blockage_sd", "skewness", "kurtosis", "auc"],
    "combined_blockage_3d": ["blockage_percent", "dwell_time_ms", "fwhm_width",
                              "blockage_sd", "skewness", "kurtosis", "blockage_mean"],
    "all_calculated_with_segments": [
        "dwell_time_ms", "blockage_percent", "blockage_mean", "blockage_sd",
        "auc", "skewness", "kurtosis", "fwhm_width", "fwhm_drop",
        "rise_time", "fall_time", "left_slope", "right_slope",
        *[f"segment_{index:02d}_blockage_mean" for index in range(1, 11)],
    ],
}
UMAP_CONFIGS = ((15, 0.1), (30, 0.1), (50, 0.3))
EXCLUDED_SHAPE_FEATURES = ("rise_time", "fall_time", "left_slope", "right_slope",
                           "left_slope_magnitude", "right_slope_magnitude")

IDENTITY_METADATA_COLUMNS = (
    "protein", "recording_id", "annotation_id", "event_id", "source_event_id",
)
CONFOUNDING_METADATA_COLUMNS = (
    "sampling_rate_hz", "baseline_mean", "baseline_sd", "peak_snr",
    "shape_qc_pass", "shape_qc_reason", "fwhm_qc_pass", "rise_qc_pass",
    "fall_qc_pass", "annotation_format", "sweep_index",
)
FORBIDDEN_FEATURE_TOKENS = (
    "protein", "label", "class", "recording", "annotation", "event_id",
    "path", "reason", "qc", "source_", "index", "sampling_rate", "baseline",
)


def assert_feature_contract(data, features):
    """Fail closed if an identifier, label, path, QC, or derived label enters X."""
    missing = sorted(set(features) - set(data.columns))
    if missing:
        raise ValueError(f"Missing features: {missing}")
    forbidden = sorted(feature for feature in features
                       if any(token in feature.lower() for token in FORBIDDEN_FEATURE_TOKENS))
    if forbidden:
        raise ValueError(f"Forbidden metadata/label columns in features: {forbidden}")
    if len(features) != len(set(features)):
        raise ValueError("Feature list contains duplicates")


def internal_metrics(matrix, labels, max_events=5000, seed=42):
    """Compute internal indices on a bounded reproducible sample."""
    labels = np.asarray(labels)
    valid = labels != -1
    matrix, labels = matrix[valid], labels[valid]
    unique = np.unique(labels)
    if len(unique) < 2 or len(unique) >= len(labels):
        return {"silhouette": np.nan, "davies_bouldin": np.nan,
                "calinski_harabasz": np.nan, "n_metric_events": len(labels)}
    if len(labels) > max_events:
        indices = np.random.default_rng(seed).choice(len(labels), max_events, replace=False)
        matrix, labels = matrix[indices], labels[indices]
    return {"silhouette": float(silhouette_score(matrix, labels)),
            "davies_bouldin": float(davies_bouldin_score(matrix, labels)),
            "calinski_harabasz": float(calinski_harabasz_score(matrix, labels)),
            "n_metric_events": len(labels)}


def choose_k_without_labels(metric_table):
    """Rank k using internal indices and an inertia elbow proxy only."""
    table = metric_table.copy()
    table["silhouette_rank"] = table.silhouette.rank(ascending=False, method="min")
    table["davies_bouldin_rank"] = table.davies_bouldin.rank(ascending=True, method="min")
    table["calinski_harabasz_rank"] = table.calinski_harabasz.rank(ascending=False, method="min")
    table["inertia_gain"] = -table.inertia.diff()
    table["elbow_rank"] = table.inertia_gain.rank(ascending=False, method="min").fillna(len(table))
    table["internal_rank_sum"] = table[["silhouette_rank", "davies_bouldin_rank",
                                        "calinski_harabasz_rank", "elbow_rank"]].sum(axis=1)
    return int(table.sort_values(["internal_rank_sum", "k"]).iloc[0].k), table


def recording_aware_sample(data, seed, balanced=False):
    """Sample within protein/recording strata; balance proteins only when requested."""
    rng = np.random.default_rng(seed)
    if not balanced:
        return data.copy().reset_index(drop=True)
    counts = data.protein.value_counts()
    target = int(counts.min())
    pieces = []
    for protein, protein_frame in data.groupby("protein", sort=True):
        rec_counts = protein_frame.recording_id.value_counts()
        raw = target * rec_counts / rec_counts.sum()
        allocation = np.floor(raw).astype(int)
        for recording in (raw - allocation).sort_values(ascending=False).index[:target-allocation.sum()]:
            allocation.loc[recording] += 1
        for recording, n_rows in allocation.items():
            group = protein_frame[protein_frame.recording_id == recording]
            indices = rng.choice(group.index.to_numpy(), size=int(n_rows), replace=False)
            pieces.append(data.loc[indices])
    return pd.concat(pieces).sample(frac=1, random_state=seed).reset_index(drop=True)


def _composition(assignments, column):
    if column not in assignments:
        return pd.DataFrame()
    return assignments.groupby(["cluster", column], dropna=False).size().rename("n_events").reset_index()


def run_cluster_experiment(data, output, features, seeds):
    """Run natural and balanced exploratory clustering with post-hoc audits."""
    assert_feature_contract(data, features)
    metadata_columns = [c for c in (*IDENTITY_METADATA_COLUMNS, *CONFOUNDING_METADATA_COLUMNS,
                                    "abf_path", "annotation_path") if c in data]
    specification = {"feature_columns": features, "metadata_columns": metadata_columns,
                     "explicitly_excluded_from_X": sorted(set(data.columns) - set(features)),
                     "transformations": "none", "missing_values": "median imputation",
                     "scaling": "RobustScaler", "outlier_policy": "retain; audit robust z > 5",
                     "fit_scope": "each exploratory scenario independently; no classifier use"}
    (output / "preprocessing_specification.json").write_text(json.dumps(specification, indent=2))
    summary_rows, stability_rows = [], []
    reference_labels = {}
    for scenario in ("natural", "balanced"):
        scenario_seeds = [seeds[0]] if scenario == "natural" else seeds
        for seed in scenario_seeds:
            subset = recording_aware_sample(data, seed, balanced=scenario == "balanced")
            run_dir = output / scenario / f"seed_{seed}"; run_dir.mkdir(parents=True)
            numeric, matrix, params = prepare(subset, features)
            params.to_csv(run_dir / "preprocessing_parameters.csv", index=False)
            robust_z = np.abs(matrix)
            pd.DataFrame({"feature": features, "n_missing": numeric.isna().sum().values,
                          "missing_fraction": numeric.isna().mean().values,
                          "n_robust_abs_gt_5": (robust_z > 5).sum(axis=0)}).to_csv(
                              run_dir / "feature_quality.csv", index=False)
            pca, coordinates, variance, loadings, _, thresholds = run_pca(
                matrix, subset[metadata_columns].reset_index(drop=True), features)
            n_pc = min(max(thresholds["90"], 2), matrix.shape[1])
            representation = pca.transform(matrix)[:, :n_pc]
            variance.to_csv(run_dir / "pca_explained_variance.csv", index=False)
            loadings.to_csv(run_dir / "pca_loadings.csv")
            coordinates.to_csv(run_dir / "pca_scores.csv", index=False)
            plot_dir = run_dir / "plots"; plot_dir.mkdir()
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.plot(variance.component, variance.explained_variance_ratio, "o-", label="individual")
            ax.plot(variance.component, variance.cumulative_explained_variance, "s--", label="cumulative")
            ax.axhline(.90, color="grey", linewidth=1, linestyle=":")
            ax.set(xlabel="Principal component", ylabel="Variance ratio",
                   title=f"PCA scree — {scenario}, seed {seed}")
            ax.legend(); fig.tight_layout()
            fig.savefig(plot_dir / "pca_scree.png", dpi=180, bbox_inches="tight"); plt.close(fig)
            k_rows = []
            for k in range(2, min(9, len(subset))):
                model = KMeans(n_clusters=k, n_init=20, random_state=seed).fit(representation)
                k_rows.append({"k": k, "inertia": float(model.inertia_),
                               **internal_metrics(representation, model.labels_, seed=seed)})
            selected_k, k_table = choose_k_without_labels(pd.DataFrame(k_rows))
            k_table.to_csv(run_dir / "k_selection_internal_metrics.csv", index=False)
            models = {
                "kmeans": KMeans(n_clusters=selected_k, n_init=50, random_state=seed).fit(representation),
                "gmm": GaussianMixture(n_components=selected_k, covariance_type="full",
                                       n_init=5, random_state=seed).fit(representation),
            }
            labels_by_method = {"kmeans": models["kmeans"].labels_,
                                "gmm": models["gmm"].predict(representation)}
            if len(subset) <= 2000:
                labels_by_method["hierarchical_ward"] = AgglomerativeClustering(
                    n_clusters=selected_k, linkage="ward").fit_predict(representation)
            neighbor_k = min(10, len(subset) - 1)
            distances = NearestNeighbors(n_neighbors=neighbor_k).fit(representation).kneighbors()[0][:, -1]
            eps = float(np.quantile(distances, .90))
            labels_by_method["dbscan"] = DBSCAN(eps=eps, min_samples=neighbor_k).fit_predict(representation)
            assignment = subset[metadata_columns].reset_index(drop=True).copy()
            for method, labels in labels_by_method.items():
                metrics = internal_metrics(representation, labels, seed=seed)
                summary_rows.append({"scenario": scenario, "seed": seed, "method": method,
                                     "selected_k_internal_only": selected_k,
                                     "n_clusters_excluding_noise": len(set(labels) - {-1}),
                                     "noise_fraction": float(np.mean(labels == -1)), **metrics})
                assignment[f"cluster_{method}"] = labels
            assignment["cluster"] = labels_by_method["kmeans"]
            for baseline_column in ("baseline_mean", "baseline_sd"):
                if baseline_column in assignment:
                    assignment[f"{baseline_column}_quartile"] = pd.qcut(
                        assignment[baseline_column], 4, duplicates="drop").astype(str)
            assignment.to_csv(run_dir / "cluster_assignments.csv", index=False)
            plotted = coordinates.copy()
            plotted["cluster"] = assignment.cluster.map(lambda value: f"cluster_{value}")
            for x, y in (("PC1", "PC2"), ("PC1", "PC3")):
                for color in ("cluster", "protein", "recording_id"):
                    save_scatter(plotted, x, y, color,
                                 plot_dir / f"{x.lower()}_{y.lower()}_by_{color}.png",
                                 f"PCA {x}–{y} by {color} — {scenario}, seed {seed}")
                    save_scatter(plotted, x, y, color,
                                 plot_dir / f"{x.lower()}_{y.lower()}_by_{color}_zoom_p01_p99.png",
                                 f"PCA {x}–{y} by {color} — {scenario}, seed {seed}",
                                 central=True)
            for color in ("cluster", "protein", "recording_id"):
                save_pca_3d(plotted, color, plot_dir / f"pca_3d_by_{color}.png",
                            f"PCA 3D by {color} — {scenario}, seed {seed}")
                save_pca_3d(plotted, color,
                            plot_dir / f"pca_3d_by_{color}_zoom_p01_p99.png",
                            f"PCA 3D by {color} — {scenario}, seed {seed}", central=True)
            feature_color_values = numeric.fillna(numeric.median())
            save_feature_color_views(
                plotted, feature_color_values, features, plot_dir,
                f"PCA3D_{scenario}_seed_{seed}", ("PC1", "PC2", "PC3"),
                {f"PC{i + 1}": value for i, value in enumerate(pca.explained_variance_ratio_)})
            for column in ("protein", "recording_id", "sampling_rate_hz", "shape_qc_pass",
                           "shape_qc_reason", "baseline_mean_quartile", "baseline_sd_quartile"):
                composition = _composition(assignment, column)
                if not composition.empty:
                    composition.to_csv(run_dir / f"composition_by_{column}.csv", index=False)
            profiles = numeric.assign(cluster=assignment.cluster).groupby("cluster")[features].median()
            profiles.to_csv(run_dir / "cluster_profiles_median_original_units.csv")
            representative_rows = []
            labels = labels_by_method["kmeans"]
            for cluster in sorted(set(labels)):
                members = np.flatnonzero(labels == cluster)
                center = models["kmeans"].cluster_centers_[cluster]
                nearest = members[np.argsort(np.linalg.norm(representation[members] - center, axis=1))[:5]]
                for rank, index in enumerate(nearest, 1):
                    representative_rows.append({"cluster": cluster, "rank": rank,
                                                **assignment.iloc[index][metadata_columns].to_dict()})
            pd.DataFrame(representative_rows).to_csv(run_dir / "representative_events.csv", index=False)
            key = scenario
            if key not in reference_labels:
                reference_labels[key] = (assignment.event_id.astype(str) + "|" + assignment.recording_id.astype(str), labels)
            else:
                ids0, lab0 = reference_labels[key]
                ids1 = assignment.event_id.astype(str) + "|" + assignment.recording_id.astype(str)
                common = pd.Index(ids0).intersection(pd.Index(ids1))
                map0 = dict(zip(ids0, lab0)); map1 = dict(zip(ids1, labels))
                stability_rows.append({"scenario": scenario, "comparison": f"{seeds[0]}_vs_{seed}",
                                       "n_common_events": len(common), "adjusted_rand_index":
                                       adjusted_rand_score([map0[x] for x in common], [map1[x] for x in common])})
    pd.DataFrame(summary_rows).to_csv(output / "internal_validation_metrics.csv", index=False)
    natural = pd.read_csv(output / "natural" / f"seed_{seeds[0]}" / "cluster_assignments.csv")
    natural_ids = natural.event_id.astype(str) + "|" + natural.recording_id.astype(str)
    natural_labels = dict(zip(natural_ids, natural.cluster))
    natural_k = int(natural.cluster.nunique())
    # Leave-one-recording-out sensitivity is the strongest feasible recording-aware
    # check here; a protein-stratified recording bootstrap is not identifiable when
    # a protein has only one recording.
    for omitted in sorted(data.recording_id.unique()):
        reduced = data[data.recording_id != omitted].reset_index(drop=True)
        if reduced.protein.nunique() < data.protein.nunique():
            stability_rows.append({"scenario": "natural", "comparison": f"omit_{omitted}",
                                   "n_common_events": len(reduced), "adjusted_rand_index": np.nan,
                                   "note": "not identifiable: omission removes an entire protein"})
            continue
        _, reduced_matrix, _ = prepare(reduced, features)
        reduced_pca = PCA().fit(reduced_matrix)
        n_pc = min(max(int(np.searchsorted(np.cumsum(reduced_pca.explained_variance_ratio_), .9) + 1), 2),
                   reduced_matrix.shape[1])
        reduced_repr = reduced_pca.transform(reduced_matrix)[:, :n_pc]
        reduced_labels = KMeans(n_clusters=natural_k, n_init=50, random_state=seeds[0]).fit_predict(reduced_repr)
        reduced_ids = reduced.event_id.astype(str) + "|" + reduced.recording_id.astype(str)
        stability_rows.append({"scenario": "natural", "comparison": f"omit_{omitted}",
                               "n_common_events": len(reduced), "adjusted_rand_index":
                               adjusted_rand_score([natural_labels[x] for x in reduced_ids], reduced_labels),
                               "note": "leave-one-recording-out refit"})
    pd.DataFrame(stability_rows).to_csv(output / "cluster_stability.csv", index=False)
    comparisons = []
    for seed in seeds:
        balanced = pd.read_csv(output / "balanced" / f"seed_{seed}" / "cluster_assignments.csv")
        comparisons.append({"balanced_seed": seed, "n_events_natural": len(natural),
                            "n_events_balanced": len(balanced),
                            "natural_clusters": natural.cluster.nunique(),
                            "balanced_clusters": balanced.cluster.nunique()})
    pd.DataFrame(comparisons).to_csv(output / "natural_vs_balanced.csv", index=False)
    comparison_dir = output / "comparison_plots"; comparison_dir.mkdir()
    panels = [("natural", seeds[0])] + [("balanced", seed) for seed in seeds]
    for color in ("protein", "cluster"):
        for zoom in (False, True):
            fig, axes = plt.subplots(2, 3, figsize=(18, 11), squeeze=False)
            for ax, (scenario, seed) in zip(axes.flat, panels):
                run_dir = output / scenario / f"seed_{seed}"
                scores = pd.read_csv(run_dir / "pca_scores.csv")
                assignments = pd.read_csv(run_dir / "cluster_assignments.csv")
                hue = (assignments.cluster.map(lambda value: f"cluster_{value}")
                       if color == "cluster" else scores[color].astype(str))
                sns.scatterplot(x=scores.PC1, y=scores.PC2, hue=hue, s=14, alpha=.55,
                                linewidth=0, ax=ax, legend="brief")
                if zoom:
                    ax.set_xlim(central_limits(scores.PC1)); ax.set_ylim(central_limits(scores.PC2))
                ax.set_title(f"{scenario}, seed {seed} (independent PCA fit)")
                ax.legend(fontsize="x-small", loc="best")
            for ax in axes.flat[len(panels):]:
                ax.axis("off")
            suffix = " — zoom 1–99%" if zoom else ""
            fig.suptitle(f"Natural vs balanced PCA PC1–PC2 by {color}{suffix}", fontsize=15)
            fig.tight_layout()
            filename = f"natural_vs_balanced_pc1_pc2_by_{color}"
            if zoom:
                filename += "_zoom_p01_p99"
            fig.savefig(comparison_dir / f"{filename}.png", dpi=180, bbox_inches="tight")
            plt.close(fig)
    fig, ax = plt.subplots(figsize=(9, 6))
    for scenario, seed in panels:
        variance = pd.read_csv(output / scenario / f"seed_{seed}" / "pca_explained_variance.csv")
        ax.plot(variance.component, variance.cumulative_explained_variance,
                marker="o", markersize=3, label=f"{scenario} seed {seed}")
    ax.axhline(.90, color="black", linewidth=1, linestyle=":")
    ax.set(xlabel="Principal component", ylabel="Cumulative explained variance",
           title="PCA variance: natural vs balanced")
    ax.legend(fontsize="small"); fig.tight_layout()
    fig.savefig(comparison_dir / "natural_vs_balanced_cumulative_variance.png",
                dpi=180, bbox_inches="tight"); plt.close(fig)


def central_limits(values):
    low, high = np.nanquantile(values, [0.01, 0.99])
    pad = max((high - low) * 0.03, 1e-9)
    return low - pad, high + pad


def save_scatter(frame, x, y, hue, path, title, central=False, style=None):
    fig, ax = plt.subplots(figsize=(9, 7))
    sns.scatterplot(data=frame, x=x, y=y, hue=hue, style=style, s=20, alpha=.58,
                    linewidth=0, ax=ax)
    if central:
        ax.set_xlim(central_limits(frame[x])); ax.set_ylim(central_limits(frame[y]))
        title += " (central 1–99% view)"
    ax.set_title(title)
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize="small")
    fig.tight_layout(); fig.savefig(path, dpi=180, bbox_inches="tight"); plt.close(fig)


def save_centroids(frame, x, y, path, title):
    centroids = frame.groupby(["protein", "recording_id"], as_index=False)[[x, y]].median()
    fig, ax = plt.subplots(figsize=(9, 7))
    sns.scatterplot(data=centroids, x=x, y=y, hue="protein", style="recording_id",
                    s=120, ax=ax)
    for row in centroids.itertuples():
        ax.annotate(row.recording_id, (getattr(row, x), getattr(row, y)), fontsize=7,
                    xytext=(3, 3), textcoords="offset points")
    ax.set_title(title + "; one centroid per recording")
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize="small")
    fig.tight_layout(); fig.savefig(path, dpi=180, bbox_inches="tight"); plt.close(fig)
    return centroids


def save_pca_3d(frame, color, path, title, central=False,
                elevation=24, azimuth=-58):
    """Save a PC1-PC3 projection without changing the fitted PCA."""
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    labels = frame[color].astype(str)
    categories = labels.drop_duplicates().tolist()
    palette = ({"HSA": "tab:blue", "F8": "tab:orange"} if color == "protein"
               else dict(zip(categories, sns.color_palette("husl", len(categories)))))
    for category in categories:
        group = frame[labels == category]
        ax.scatter(group.PC1, group.PC2, group.PC3, s=10, alpha=.4,
                   linewidth=0, label=category, color=palette.get(category, "grey"))
    if central:
        ax.set_xlim(central_limits(frame.PC1)); ax.set_ylim(central_limits(frame.PC2))
        ax.set_zlim(central_limits(frame.PC3)); title += " (central 1–99% view)"
    ax.view_init(elev=elevation, azim=azimuth)
    ax.set(xlabel="PC1", ylabel="PC2", zlabel="PC3", title=title)
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize="small")
    fig.tight_layout(); fig.savefig(path, dpi=180, bbox_inches="tight"); plt.close(fig)


def save_pca_3d_protein_recording(frame, path, title, central=False,
                                  elevation=24, azimuth=-58):
    """Color by protein and use marker identity for recording in 3D."""
    fig = plt.figure(figsize=(11, 8))
    ax = fig.add_subplot(111, projection="3d")
    markers = cycle(("o", "X", "s", "P", "D", "^", "v", "<", ">", "*", "h", "p", "8", "d"))
    marker_map = {recording: next(markers) for recording in frame.recording_id.unique()}
    colors = {"HSA": "tab:blue", "F8": "tab:orange"}
    for (protein, recording), group in frame.groupby(["protein", "recording_id"]):
        ax.scatter(group.PC1, group.PC2, group.PC3, s=12, alpha=.42, linewidth=0,
                   color=colors.get(protein, "grey"), marker=marker_map[recording],
                   label=f"{protein}:{recording}")
    if central:
        ax.set_xlim(central_limits(frame.PC1)); ax.set_ylim(central_limits(frame.PC2))
        ax.set_zlim(central_limits(frame.PC3)); title += " (central 1–99% view)"
    ax.view_init(elev=elevation, azim=azimuth)
    ax.set(xlabel="PC1", ylabel="PC2", zlabel="PC3", title=title)
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize="x-small")
    fig.tight_layout(); fig.savefig(path, dpi=180, bbox_inches="tight"); plt.close(fig)


def centroid_distance_table(centroids, axes):
    """Return all pairwise Euclidean distances between recording centroids."""
    rows = []
    for i, left in centroids.iterrows():
        for _, right in centroids.iloc[i + 1:].iterrows():
            rows.append({"protein_1": left.protein, "recording_1": left.recording_id,
                         "protein_2": right.protein, "recording_2": right.recording_id,
                         "distance": float(np.linalg.norm(left[list(axes)] - right[list(axes)]))})
    return pd.DataFrame(rows).sort_values("distance", ascending=False)


def diagnostics(frame, x, y, seed=42, max_events=5000, max_distance_events=2000):
    """Compute bounded-memory exploratory diagnostics on a reproducible sample."""
    if len(frame) > max_events:
        frame = frame.sample(n=max_events, random_state=seed)
    values = frame[[x, y]].to_numpy()
    distance_frame = (frame.sample(n=max_distance_events, random_state=seed)
                      if len(frame) > max_distance_events else frame)
    distance_values = distance_frame[[x, y]].to_numpy()
    distances = pairwise_distances(distance_values)
    upper = np.triu(np.ones(distances.shape, dtype=bool), 1)
    recording = distance_frame.recording_id.to_numpy()
    protein = distance_frame.protein.to_numpy()
    same_recording = recording[:, None] == recording
    same_protein = protein[:, None] == protein
    def mean(mask):
        selected = distances[upper & mask]
        return float(selected.mean()) if selected.size else np.nan
    return {
        "n_events_silhouette": len(frame),
        "n_events_distance": len(distance_frame),
        "silhouette_protein": float(silhouette_score(values, frame.protein)),
        "silhouette_recording": float(silhouette_score(values, frame.recording_id)),
        "within_recording_mean_distance": mean(same_recording),
        "between_recording_mean_distance": mean(~same_recording),
        "within_protein_mean_distance": mean(same_protein),
        "between_protein_mean_distance": mean(~same_protein),
    }


def prepare(data, features):
    numeric = data[features].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    imputer = SimpleImputer(strategy="median")
    imputed = imputer.fit_transform(numeric)
    scaler = RobustScaler()
    matrix = scaler.fit_transform(imputed)
    if not np.isfinite(matrix).all():
        raise ValueError("Preprocessed matrix contains non-finite values")
    params = pd.DataFrame({"feature": features, "median": imputer.statistics_,
                           "robust_center": scaler.center_, "robust_scale_iqr": scaler.scale_})
    return numeric, matrix, params


def normalize_qc_flag(series, column):
    """Parse common CSV boolean representations and reject unknown values."""
    mapping = {True: True, False: False, 1: True, 0: False,
               "true": True, "false": False, "1": True, "0": False,
               "yes": True, "no": False}
    parsed = series.map(lambda value: False if pd.isna(value)
                        else mapping.get(value if isinstance(value, (bool, int))
                                         else str(value).strip().lower()))
    if parsed.isna().any():
        invalid = sorted(series[parsed.isna()].astype(str).unique().tolist())
        raise ValueError(f"Invalid boolean values in {column}: {invalid}")
    return parsed.astype(bool)


def domain_audit(data, features):
    """Report physically invalid or internally inconsistent feature values."""
    numeric = data[list(features)].apply(pd.to_numeric, errors="coerce")
    checks = []
    def add(name, mask):
        checks.append({"check": name, "n_flagged": int(mask.fillna(False).sum())})
    for feature in ("dwell_time_ms", "fwhm_width", "fwhm_drop", "rise_time", "fall_time"):
        if feature in numeric:
            add(f"{feature}_negative", numeric[feature] < 0)
    if "dwell_time_ms" in numeric:
        add("dwell_time_ms_nonpositive", numeric["dwell_time_ms"] <= 0)
    if {"fwhm_width", "dwell_time_ms"} <= set(numeric):
        add("fwhm_width_exceeds_dwell_time", numeric.fwhm_width > numeric.dwell_time_ms)
    if "left_slope" in numeric:
        add("left_slope_nonpositive", numeric.left_slope <= 0)
    if "right_slope" in numeric:
        add("right_slope_nonnegative", numeric.right_slope >= 0)
    return pd.DataFrame(checks)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--analysis-profile", choices=["v3-aligned"],
                        default="v3-aligned", help=argparse.SUPPRESS)
    parser.add_argument("--feature-set", choices=["all", *FEATURE_SETS],
                        default="all_calculated_features")
    parser.add_argument("--experiment-mode", choices=["embedding", "cluster-comparison"],
                        default="embedding")
    parser.add_argument("--balanced-seeds", default="11,42,73,202,991",
                        help="Comma-separated seeds for balanced resampling")
    args = parser.parse_args(argv)
    source = Path(args.input).resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Input feature CSV does not exist: {source}")
    extraction_manifest_path = source.parent / "run_manifest.json"
    extraction_manifest = None
    if extraction_manifest_path.is_file():
        extraction_manifest = validate_extraction_provenance(
            source.parent, extraction_manifest_path, source)
    requested_output = Path(args.output).resolve()
    if requested_output.exists():
        requested_output = requested_output.with_name(
            f"{requested_output.name}_{datetime.now():%Y%m%d_%H%M%S}")
    output = require_new_output_directory(requested_output)
    data = pd.read_csv(source)
    required_metadata = {"protein", "recording_id", "event_id"}
    missing_metadata = required_metadata - set(data.columns)
    if missing_metadata: raise ValueError(f"Missing metadata: {sorted(missing_metadata)}")
    selected_feature_sets = (FEATURE_SETS if args.feature_set == "all"
                             else {args.feature_set: FEATURE_SETS[args.feature_set]})
    required_features = set().union(*selected_feature_sets.values())
    missing_features = required_features - set(data.columns)
    if missing_features: raise ValueError(f"Missing features: {sorted(missing_features)}")
    if data[["protein", "recording_id", "event_id"]].isna().any().any():
        raise ValueError("Required metadata contains missing values")
    identity_duplicates = int(data.duplicated(["protein", "recording_id", "event_id"]).sum())
    exact_duplicates = int(data.duplicated().sum())
    if identity_duplicates:
        raise ValueError(f"Duplicate event identities found: {identity_duplicates}")

    if args.experiment_mode == "cluster-comparison":
        if args.feature_set == "all":
            raise ValueError("cluster-comparison requires one explicit feature set")
        features = selected_feature_sets[args.feature_set]
        seeds = [int(value.strip()) for value in args.balanced_seeds.split(",") if value.strip()]
        if not seeds:
            raise ValueError("At least one balanced seed is required")
        run_cluster_experiment(data, output, features, seeds)
        upstream_experiment_id = (extraction_manifest or {}).get("experiment_id")
        upstream_output_path = str(source.parent)
        report_text = (
            "# Unsupervised clustering run\n\n"
            f"Input accepted events: {len(data):,}. "
            f"Protein counts: {data.protein.value_counts().to_dict()}. "
            f"Recording counts: {data.groupby('protein').recording_id.nunique().to_dict()}.\n\n"
            "Protein labels, recording identifiers, paths, QC fields/reasons, baselines, "
            "sampling rate, and label-derived columns were excluded from X. Labels were "
            "used only for post-hoc composition. Cluster count was selected without labels.\n\n"
            "Natural and balanced outputs are separated. Because at least one protein has "
            "only one recording, protein and recording effects are not independently "
            "identifiable; event-level balancing cannot remove this confounding. Clusters "
            "must not be described as protein identities. Any classifier must refit all "
            "preprocessing and representation steps using training data only.\n"
        )
        (output / "REPORT.md").write_text(report_text, encoding="utf-8")
        generated_files = {}
        for path in sorted(p for p in output.rglob("*") if p.is_file()):
            relative = str(path.relative_to(output))
            generated_files[relative] = {
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            if path.suffix == ".csv":
                generated_files[relative]["rows"] = max(
                    sum(1 for _ in path.open(encoding="utf-8")) - 1, 0)
        experiment_id = output.parent.name
        agent_stage = output.name
        manifest = {
            "run_type": "unsupervised_cluster_comparison",
            "experiment_id": experiment_id,
            "agent_stage": agent_stage,
            "created_at": datetime.now().astimezone().isoformat(),
            "dataset_version": (extraction_manifest or {}).get("dataset_version"),
            "upstream": {"experiment_id": upstream_experiment_id,
                         "agent_stage": (extraction_manifest or {}).get("agent_stage", "event_extraction"),
                         "output_path": upstream_output_path,
                         "manifest_path": str(extraction_manifest_path)},
            "source_paths": {"accepted_event_feature_table": str(source),
                             "upstream_output": upstream_output_path,
                             "upstream_manifest": str(extraction_manifest_path)},
            "configuration": {
                "feature_set": args.feature_set,
                "label_policy": "protein excluded from X; post-hoc composition only",
                "cluster_count_policy": "internal metrics and elbow only; labels prohibited",
                "scenarios": {"natural": "all accepted events",
                              "balanced": "equal HSA/F8 counts, allocated within recordings"},
                "preprocessing": ["replace_inf_with_nan", "median_imputation", "RobustScaler"],
                "hierarchical_policy": "Ward clustering only when scenario n<=2000",
                "fit_scope": "exploratory full scenario; not valid for classifier reuse",
            },
            "random_seeds": {"natural": [seeds[0]], "balanced": seeds},
            "feature_list": features,
            "row_counts": {"input_accepted_events": len(data),
                           "natural_events": len(data),
                           "balanced_events_per_seed": int(2 * data.protein.value_counts().min())},
            "recording_counts": {"total": int(data.recording_id.nunique()),
                                 "by_protein": data.groupby("protein").recording_id.nunique().to_dict()},
            "code_version": {"unsupervised_analysis.py_sha256": sha256_file(Path(__file__).resolve()),
                             "preprocessing.py_sha256": sha256_file(Path(__file__).with_name("preprocessing.py"))},
            "config_version": {"upstream_manifest_sha256": (sha256_file(extraction_manifest_path)
                                                               if extraction_manifest_path.is_file() else None),
                               "feature_definition_version": (extraction_manifest or {}).get("feature_definition_version")},
            "generated_files": generated_files,
        }
        if extraction_manifest is not None:
            manifest["extraction_manifest_path"] = str(extraction_manifest_path)
            manifest["extraction_manifest_sha256"] = sha256_file(extraction_manifest_path)
            manifest["dataset_version"] = extraction_manifest.get("dataset_version")
            manifest["feature_definition_version"] = extraction_manifest.get("feature_definition_version")
        (output / "run_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True))
        print(output)
        return

    pre = output / "preprocessing"; summaries = output / "summaries"
    pre.mkdir(); summaries.mkdir()
    schema = pd.DataFrame({"column": data.columns, "dtype": data.dtypes.astype(str),
                           "n_missing": data.isna().sum().values,
                           "n_unique": [data[c].nunique(dropna=False) for c in data]})
    schema.to_csv(pre / "schema_audit.csv", index=False)
    data.groupby(["protein", "recording_id"], as_index=False).size().rename(
        columns={"size": "n_events"}).to_csv(summaries / "event_counts_by_recording.csv", index=False)
    data.groupby("protein", as_index=False).size().rename(
        columns={"size": "n_events"}).to_csv(summaries / "class_distribution.csv", index=False)
    pd.DataFrame([{"n_exact_duplicate_rows": exact_duplicates,
                   "n_duplicate_event_identities": identity_duplicates}]).to_csv(
                       summaries / "duplicate_audit.csv", index=False)
    qc_columns = [c for c in ("fwhm_qc_pass", "rise_qc_pass", "fall_qc_pass", "shape_qc_pass") if c in data]
    qc_rows = []
    for (protein, recording), group in data.groupby(["protein", "recording_id"]):
        for column in qc_columns:
            passed = normalize_qc_flag(group[column], column)
            qc_rows.append({"protein": protein, "recording_id": recording, "qc_flag": column,
                            "n_events": len(group), "n_pass": int(passed.sum()),
                            "n_fail": int((~passed).sum())})
    pd.DataFrame(qc_rows).to_csv(summaries / "shape_qc_by_recording.csv", index=False)
    audit_features = sorted((required_features | set(EXCLUDED_SHAPE_FEATURES)) & set(data.columns))
    audit_numeric = data[audit_features].apply(pd.to_numeric, errors="coerce")
    pd.DataFrame({"feature": audit_features, "n_missing": audit_numeric.isna().sum(),
                  "n_pos_inf": np.isposinf(audit_numeric).sum(), "n_neg_inf": np.isneginf(audit_numeric).sum(),
                  "min": audit_numeric.replace([np.inf, -np.inf], np.nan).min(),
                  "median": audit_numeric.replace([np.inf, -np.inf], np.nan).median(),
                  "p99": audit_numeric.replace([np.inf, -np.inf], np.nan).quantile(.99),
                  "max": audit_numeric.replace([np.inf, -np.inf], np.nan).max()}).to_csv(
                      pre / "numeric_quality_audit.csv", index=False)
    miss = []
    for (protein, recording), group in data.groupby(["protein", "recording_id"], dropna=False):
        for feature in audit_features:
            values = pd.to_numeric(group[feature], errors="coerce").replace([np.inf, -np.inf], np.nan)
            miss.append({"protein": protein, "recording_id": recording, "feature": feature,
                         "n_events": len(group), "n_missing": int(values.isna().sum()),
                         "missing_fraction": float(values.isna().mean())})
    pd.DataFrame(miss).to_csv(pre / "missingness_by_protein_recording.csv", index=False)
    domain_audit(data, audit_features).to_csv(pre / "domain_audit.csv", index=False)

    all_diagnostics, all_centroids, redundancy = [], [], []
    seeds = [args.random_seed, args.random_seed + 1, args.random_seed + 2]
    metadata = data[["protein", "recording_id", "event_id"]].reset_index(drop=True)
    for set_name, features in selected_feature_sets.items():
        set_dir = output / set_name; pca_dir = set_dir / "pca"; umap_dir = set_dir / "umap"
        pca_dir.mkdir(parents=True); umap_dir.mkdir()
        numeric, matrix, params = prepare(data, features)
        color_values = numeric.fillna(numeric.median())
        params.to_csv(pre / f"preprocessing_parameters_{set_name}.csv", index=False)
        corr = numeric.corr(method="spearman")
        pearson = numeric.corr(method="pearson")
        valid_counts = numeric.notna().astype(int).T.dot(numeric.notna().astype(int))
        corr.to_csv(pre / f"spearman_correlation_{set_name}.csv")
        pearson.to_csv(pre / f"pearson_correlation_{set_name}.csv")
        valid_counts.to_csv(pre / f"pairwise_valid_counts_{set_name}.csv")
        for i, left in enumerate(features):
            for right in features[i + 1:]:
                redundancy.append({"feature_set": set_name, "feature_1": left, "feature_2": right,
                                   "spearman": corr.loc[left, right], "pearson": pearson.loc[left, right],
                                   "n_valid": int(valid_counts.loc[left, right]),
                                   "abs_spearman_ge_0_80": bool(abs(corr.loc[left, right]) >= .80)})

        pca = PCA().fit(matrix); score_values = pca.transform(matrix)
        scores = metadata.copy()
        for i in range(min(3, score_values.shape[1])): scores[f"PC{i+1}"] = score_values[:, i]
        variance = pd.DataFrame({"component": np.arange(1, len(features)+1),
                                 "explained_variance_ratio": pca.explained_variance_ratio_,
                                 "cumulative_explained_variance": np.cumsum(pca.explained_variance_ratio_)})
        loadings = pd.DataFrame(pca.components_.T, index=features,
                                columns=[f"PC{i+1}" for i in range(len(features))])
        scores.to_csv(pca_dir / "pca_scores.csv", index=False); variance.to_csv(pca_dir / "explained_variance.csv", index=False)
        loadings.to_csv(pca_dir / "pca_loadings.csv")
        contributions = loadings.iloc[:, :5].pow(2)
        contributions = contributions.div(contributions.sum(axis=0), axis=1)
        contributions.to_csv(pca_dir / "feature_contributions_pc1_pc5.csv")
        thresholds = {str(int(t * 100)): int(np.searchsorted(
            variance.cumulative_explained_variance, t) + 1) for t in (.8, .9, .95)}
        (pca_dir / "variance_threshold_components.json").write_text(json.dumps(thresholds, indent=2))
        fig, ax = plt.subplots(); ax.plot(variance.component, variance.explained_variance_ratio, "o-")
        ax.set(xlabel="Principal component", ylabel="Explained variance ratio", title=f"PCA scree: {set_name}")
        fig.tight_layout(); fig.savefig(pca_dir / "scree_plot.png", dpi=180); plt.close(fig)
        for color in ("protein", "recording_id"):
            save_pca_3d(scores, color, pca_dir / f"pca_3d_by_{color}_full.png",
                        f"PCA 3D: {set_name}")
            save_pca_3d(scores, color, pca_dir / f"pca_3d_by_{color}_central.png",
                        f"PCA 3D: {set_name}", central=True)
        for central in (False, True):
            save_pca_3d_protein_recording(
                scores, pca_dir / f"pca_3d_by_protein_recording_{'central' if central else 'full'}.png",
                f"PCA 3D: {set_name}", central=central)
        variance_map = {f"PC{i+1}": value for i, value in enumerate(pca.explained_variance_ratio_)}
        save_feature_color_views(scores, color_values, features, pca_dir, f"PCA_{set_name}", ("PC1", "PC2"), variance_map)
        save_feature_color_views(scores, color_values, features, pca_dir, f"PCA_{set_name}", ("PC1", "PC3"), variance_map)
        save_feature_color_views(scores, color_values, features, pca_dir, f"PCA_{set_name}", ("PC1", "PC2", "PC3"), variance_map)
        for x, y in (("PC1", "PC2"), ("PC1", "PC3"), ("PC2", "PC3")):
            for central in (False, True):
                suffix = "central" if central else "full"
                save_scatter(scores, x, y, "protein", pca_dir / f"{x.lower()}_{y.lower()}_by_protein_{suffix}.png",
                             f"PCA {x}–{y}: {set_name}", central)
                save_scatter(scores, x, y, "recording_id", pca_dir / f"{x.lower()}_{y.lower()}_by_recording_{suffix}.png",
                             f"PCA {x}–{y}: {set_name}", central)
                save_scatter(scores, x, y, "protein", pca_dir / f"{x.lower()}_{y.lower()}_by_protein_recording_{suffix}.png",
                             f"PCA {x}–{y}: {set_name}", central, "recording_id")
            centroids = save_centroids(scores, x, y, pca_dir / f"recording_centroids_{x.lower()}_{y.lower()}.png",
                                       f"PCA {x}–{y}: {set_name}")
            centroids.assign(feature_set=set_name, method=f"PCA_{x}_{y}").to_csv(
                pca_dir / f"recording_centroids_{x.lower()}_{y.lower()}.csv", index=False)
            all_centroids.append(centroids.assign(feature_set=set_name, method=f"PCA_{x}_{y}"))
            centroid_distance_table(centroids, (x, y)).to_csv(
                pca_dir / f"recording_centroid_distances_{x.lower()}_{y.lower()}.csv", index=False)
        centroid3d = scores.groupby(["protein", "recording_id"], as_index=False)[["PC1", "PC2", "PC3"]].median()
        centroid3d.assign(feature_set=set_name, method="PCA_PC1_PC2_PC3").to_csv(
            pca_dir / "recording_centroids_pc1_pc2_pc3.csv", index=False)
        centroid_distance_table(centroid3d, ("PC1", "PC2", "PC3")).to_csv(
            pca_dir / "recording_centroid_distances_pc1_pc2_pc3.csv", index=False)
        all_diagnostics.append({"feature_set": set_name, "method": "PCA_PC1_PC2",
                                **diagnostics(scores, "PC1", "PC2", args.random_seed)})

        from umap import UMAP
        embeddings = {}
        for neighbors, min_dist in UMAP_CONFIGS:
            for seed in seeds:
                label = f"nn{neighbors}_md{min_dist}_seed{seed}"
                embedding = UMAP(n_components=2, n_neighbors=neighbors, min_dist=min_dist,
                                 metric="euclidean", random_state=seed, transform_seed=seed).fit_transform(matrix)
                coordinates = metadata.copy(); coordinates[["UMAP1", "UMAP2"]] = embedding
                coordinates.to_csv(umap_dir / f"umap_{label}.csv", index=False)
                save_feature_color_views(coordinates, color_values, features, umap_dir,
                                         f"UMAP_{label}", ("UMAP1", "UMAP2"))
                for central in (False, True):
                    suffix = "central" if central else "full"
                    save_scatter(coordinates, "UMAP1", "UMAP2", "protein",
                                 umap_dir / f"umap_{label}_by_protein_{suffix}.png",
                                 f"UMAP {label}: {set_name}", central)
                    save_scatter(coordinates, "UMAP1", "UMAP2", "recording_id",
                                 umap_dir / f"umap_{label}_by_recording_{suffix}.png",
                                 f"UMAP {label}: {set_name}", central)
                    save_scatter(coordinates, "UMAP1", "UMAP2", "protein",
                                 umap_dir / f"umap_{label}_by_protein_recording_{suffix}.png",
                                 f"UMAP {label}: {set_name}", central, "recording_id")
                row = {"feature_set": set_name, "method": "UMAP", "n_neighbors": neighbors,
                       "min_dist": min_dist, "seed": seed, "metric": "euclidean",
                       "trustworthiness_k10": float(trustworthiness(matrix, embedding, n_neighbors=10)),
                       **diagnostics(coordinates, "UMAP1", "UMAP2", seed)}
                embeddings[(neighbors, min_dist, seed)] = embedding
                all_diagnostics.append(row)
                centroids = save_centroids(coordinates, "UMAP1", "UMAP2",
                    umap_dir / f"recording_centroids_{label}.png", f"UMAP {label}: {set_name}")
                centroids.assign(feature_set=set_name, method=f"UMAP_{label}").to_csv(
                    umap_dir / f"recording_centroids_{label}.csv", index=False)
                all_centroids.append(centroids.assign(feature_set=set_name, method=f"UMAP_{label}"))
        for neighbors, min_dist in UMAP_CONFIGS:
            for i, seed1 in enumerate(seeds):
                for seed2 in seeds[i+1:]:
                    sample_size = min(len(matrix), 2000)
                    sample_indices = np.random.default_rng(args.random_seed).choice(
                        len(matrix), size=sample_size, replace=False)
                    d1 = pairwise_distances(embeddings[(neighbors, min_dist, seed1)][sample_indices])
                    d2 = pairwise_distances(embeddings[(neighbors, min_dist, seed2)][sample_indices])
                    upper = np.triu_indices_from(d1, 1)
                    rho = spearmanr(d1[upper], d2[upper]).statistic
                    all_diagnostics.append({"feature_set": set_name, "method": "UMAP_seed_stability",
                        "n_neighbors": neighbors, "min_dist": min_dist, "seed": f"{seed1}_vs_{seed2}",
                        "n_events_distance_stability": sample_size,
                        "pairwise_distance_spearman": float(rho)})

    pd.DataFrame(all_diagnostics).to_csv(summaries / "embedding_diagnostics.csv", index=False)
    pd.concat(all_centroids, ignore_index=True).to_csv(summaries / "recording_centroids.csv", index=False)
    pd.DataFrame(redundancy).sort_values("spearman", key=lambda x: x.abs(), ascending=False).to_csv(
        summaries / "feature_redundancy.csv", index=False)
    manifest = {
        "run_type": "unsupervised_exploratory_v3_aligned", "created_at": datetime.now().astimezone().isoformat(),
        "input_path": str(source), "input_sha256": sha256_file(source), "code_file": str(Path(__file__).resolve()),
        "code_sha256": sha256_file(Path(__file__).resolve()), "feature_sets": selected_feature_sets,
        "excluded_shape_features": [feature for feature in EXCLUDED_SHAPE_FEATURES
                                    if feature not in required_features],
        "preprocessing": ["replace_inf_with_nan", "median_imputation", "RobustScaler"],
        "fit_scope": "exploratory_full_dataset",
        "diagnostic_sampling": {"silhouette_max_events": 5000,
                                "pairwise_distance_max_events": 2000},
        "feature_color_values": "original_units_after_median_imputation",
        "pca": {"svd_solver": "auto",
            "three_dimensional_view": {"elevation": 24, "azimuth": -58}},
        "umap_configurations": [{"n_neighbors": n, "min_dist": d, "metric": "euclidean"} for n, d in UMAP_CONFIGS],
        "random_seeds": seeds, "n_events_input": len(data), "n_events_used": len(data), "n_events_removed": 0,
        "recordings_by_protein": data.groupby("protein").recording_id.nunique().to_dict(),
        "events_by_protein": data.protein.value_counts().to_dict(), "python_version": platform.python_version(),
        "package_versions": {name: version(name) for name in ("numpy", "pandas", "scipy", "scikit-learn", "umap-learn", "matplotlib", "seaborn")},
    }
    if extraction_manifest is not None:
        manifest["extraction_manifest_path"] = str(extraction_manifest_path)
        manifest["extraction_manifest_sha256"] = sha256_file(extraction_manifest_path)
        manifest["dataset_version"] = extraction_manifest.get("dataset_version")
        manifest["feature_definition_version"] = extraction_manifest.get("feature_definition_version")
    (output / "run_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True))
    files = [p for p in output.rglob("*") if p.is_file()]
    (summaries / "output_inventory.csv").write_text("path,size_bytes\n" + "\n".join(f"{p},{p.stat().st_size}" for p in files) + "\n")
    print(output)


if __name__ == "__main__":
    main()
