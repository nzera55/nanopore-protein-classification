"""Compare predeclared compact feature sets with PCA and UMAP; never train."""

import argparse
import json
import os
from datetime import datetime
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/nanopore_matplotlib_cache")
os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/nanopore_numba_cache")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from feature_exploration import (
    SEGMENT_FEATURES,
    load_and_validate,
    prepare_embedding_matrix,
    save_embedding_view,
)
from project_config import require_new_output_directory, sha256_file


FEATURE_SETS = {
    "set_a_compact": [
        "dwell_time_ms", "blockage_mean", "blockage_sd", "fwhm_width",
    ],
    "set_b_core": [
        "dwell_time_ms", "blockage_mean", "blockage_sd", "auc", "fwhm_width",
    ],
    "set_c_core_shape": [
        "dwell_time_ms", "blockage_mean", "blockage_sd", "auc", "fwhm_width",
        "skewness", "kurtosis", "rise_time", "fall_time",
    ],
    "set_d_core_segments": [
        "dwell_time_ms", "blockage_mean", "blockage_sd", "auc", "fwhm_width",
        *SEGMENT_FEATURES,
    ],
    "set_e_segments_only": [*SEGMENT_FEATURES],
}


def feature_set_coverage(data, numeric):
    rows = []
    for name, features in FEATURE_SETS.items():
        complete = numeric[features].notna().all(axis=1)
        for protein in ("all", "HSA", "F8"):
            mask = pd.Series(True, index=data.index) if protein == "all" else data.protein.eq(protein)
            denominator = int(mask.sum())
            count = int((complete & mask).sum())
            rows.append({
                "feature_set": name,
                "protein": protein,
                "n_features": len(features),
                "n_events": denominator,
                "n_complete_events": count,
                "complete_fraction": count / denominator if denominator else np.nan,
                "n_events_requiring_imputation": denominator - count,
            })
    return pd.DataFrame(rows)


def plot_variance(variance, path, name):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(np.arange(1, len(variance) + 1), variance, "o-")
    ax.set(xlabel="Principal component", ylabel="Explained variance ratio",
           title=f"PCA explained variance — {name}")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def analyze_set(data, numeric, name, features, output, seed, neighbors, min_dist):
    from umap import UMAP

    directory = output / name
    directory.mkdir()
    matrix, colors, preprocessing = prepare_embedding_matrix(numeric[features])
    preprocessing.to_csv(directory / "preprocessing.csv", index=False)

    pca = PCA().fit(matrix)
    transformed = pca.transform(matrix)
    metadata = data[["protein", "recording_id", "event_id"]].reset_index(drop=True)
    pca_scores = metadata.copy()
    pca_scores[["PC1", "PC2"]] = transformed[:, :2]
    pca_scores.to_csv(directory / "pca_scores.csv", index=False)
    loadings = pd.DataFrame(
        pca.components_.T,
        index=features,
        columns=[f"PC{i + 1}" for i in range(len(features))],
    )
    loadings.to_csv(directory / "pca_loadings.csv")
    variance = pd.DataFrame({
        "component": np.arange(1, len(features) + 1),
        "explained_variance_ratio": pca.explained_variance_ratio_,
        "cumulative_explained_variance": np.cumsum(pca.explained_variance_ratio_),
    })
    variance.to_csv(directory / "pca_explained_variance.csv", index=False)
    plot_variance(pca.explained_variance_ratio_, directory / "pca_scree.png", name)

    colored_pca = pd.concat([pca_scores, colors.reset_index(drop=True)], axis=1)
    save_embedding_view(colored_pca, "PC1", "PC2", "protein", directory, f"{name} PCA", True)
    save_embedding_view(colored_pca, "PC1", "PC2", "recording_id", directory, f"{name} PCA", True)
    for feature in features:
        save_embedding_view(colored_pca, "PC1", "PC2", feature, directory, f"{name} PCA")

    umap = UMAP(n_components=2, n_neighbors=neighbors, min_dist=min_dist,
                metric="euclidean", random_state=seed, transform_seed=seed)
    embedding = umap.fit_transform(matrix)
    umap_scores = metadata.copy()
    umap_scores[["UMAP1", "UMAP2"]] = embedding
    umap_scores.to_csv(directory / "umap_coordinates.csv", index=False)
    colored_umap = pd.concat([umap_scores, colors.reset_index(drop=True)], axis=1)
    save_embedding_view(colored_umap, "UMAP1", "UMAP2", "protein", directory, f"{name} UMAP", True)
    save_embedding_view(colored_umap, "UMAP1", "UMAP2", "recording_id", directory, f"{name} UMAP", True)
    for feature in features:
        save_embedding_view(colored_umap, "UMAP1", "UMAP2", feature, directory, f"{name} UMAP")

    return {
        "feature_set": name,
        "n_features": len(features),
        "pc1_explained_variance": float(pca.explained_variance_ratio_[0]),
        "pc1_pc2_explained_variance": float(pca.explained_variance_ratio_[:2].sum()),
        "features": features,
    }


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
    extraction_manifest = json.loads(extraction_manifest_path.read_text())
    if Path(extraction_manifest["output_files"]["feature_path"]).resolve() != source:
        raise ValueError("Feature table does not match its extraction manifest")

    output = require_new_output_directory(args.output.resolve())
    data, numeric = load_and_validate(source)
    coverage = feature_set_coverage(data, numeric)
    coverage.to_csv(output / "feature_set_coverage.csv", index=False)
    summaries = [
        analyze_set(data, numeric, name, features, output, args.random_seed,
                    args.umap_neighbors, args.umap_min_dist)
        for name, features in FEATURE_SETS.items()
    ]
    pd.DataFrame([{k: v for k, v in row.items() if k != "features"} for row in summaries]).to_csv(
        output / "pca_summary.csv", index=False
    )

    report_lines = [
        "# Feature-set exploration\n",
        f"Natural dataset: {len(data):,} events; HSA={(data.protein == 'HSA').sum():,}; "
        f"F8={(data.protein == 'F8').sum():,}. No balancing or model training was performed.\n",
        "Each feature set was independently median-imputed and RobustScaler-transformed "
        "on the full descriptive dataset before PCA and UMAP.\n",
    ]
    for row in summaries:
        report_lines.append(
            f"- {row['feature_set']}: {row['n_features']} features; PC1="
            f"{row['pc1_explained_variance']:.2%}; PC1+PC2="
            f"{row['pc1_pc2_explained_variance']:.2%}."
        )
    report_lines.append(
        "\nThese embeddings are exploratory and do not measure classification performance. "
        "Protein remains confounded with recording because HSA has one recording.\n"
    )
    (output / "REPORT.md").write_text("\n".join(report_lines), encoding="utf-8")

    manifest = {
        "status": "completed",
        "run_type": "feature_set_exploration_no_training",
        "created_at": datetime.now().astimezone().isoformat(),
        "input_path": str(source),
        "input_sha256": sha256_file(source),
        "extraction_manifest_path": str(extraction_manifest_path),
        "dataset_version": extraction_manifest.get("dataset_version"),
        "feature_definition_version": extraction_manifest.get("feature_definition_version"),
        "feature_sets": FEATURE_SETS,
        "preprocessing": ["replace_inf_with_nan", "median_imputation", "RobustScaler"],
        "fit_scope": "full_natural_exploratory_dataset_separately_per_feature_set",
        "umap": {"random_seed": args.random_seed, "n_neighbors": args.umap_neighbors,
                 "min_dist": args.umap_min_dist, "metric": "euclidean"},
        "training": False,
        "summaries": summaries,
    }
    (output / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
