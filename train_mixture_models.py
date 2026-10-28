"""Train final balanced pure-protein models for unlabeled mixture inference."""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
from datetime import datetime
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_config(path):
    path = Path(path).resolve()
    config = json.loads(path.read_text(encoding="utf-8"))
    root = path.parent
    resolved = dict(config)
    for key in ("feature_table", "extraction_manifest", "output_root"):
        resolved[key] = str((root / config[key]).resolve())
    resolved["config_path"] = str(path)
    return resolved


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-id", default=None)
    args = parser.parse_args(argv)
    config = load_config(args.config)

    data = pd.read_csv(
        config["feature_table"],
        dtype={"recording_id": str},
        low_memory=False,
    )
    negative = data[data["protein"] == config["negative_label"]].copy()
    positive = data[data["protein"] == config["positive_label"]].copy()
    if negative.empty or positive.empty:
        raise ValueError("Both configured pure-protein classes are required")
    if len(negative) < len(positive):
        raise ValueError("HSA has fewer events than F8; configured downsampling is invalid")

    all_features = {
        feature
        for values in config["feature_sets"].values()
        for feature in values
    }
    missing = sorted(all_features - set(data.columns))
    if missing:
        raise ValueError(f"Feature table is missing columns: {missing}")

    run_id = args.run_id or datetime.now().strftime(
        "balanced_pure_training_2seed_%Y%m%d_%H%M%S"
    )
    output = Path(config["output_root"]) / run_id
    output.mkdir(parents=True, exist_ok=False)

    selection_rows = []
    model_rows = []
    for seed in config["seeds"]:
        sampled_negative = negative.sample(
            n=len(positive), replace=False, random_state=seed
        )
        training = pd.concat([sampled_negative, positive], ignore_index=True)
        training = training.sample(frac=1.0, random_state=seed).reset_index(drop=True)
        training["target"] = (
            training["protein"] == config["positive_label"]
        ).astype(int)

        for row in training.itertuples(index=False):
            selection_rows.append({
                "seed": seed,
                "event_id": row.event_id,
                "recording_id": row.recording_id,
                "protein": row.protein,
            })

        for feature_set, features in config["feature_sets"].items():
            parameters = config["random_forest_parameters"][feature_set][str(seed)]
            estimator = RandomForestClassifier(
                random_state=seed,
                n_jobs=1,
                **parameters,
            )
            pipeline = Pipeline([
                ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
                ("scaler", RobustScaler()),
                ("model", estimator),
            ])
            pipeline.fit(training[features], training["target"])
            model_path = output / f"model_{feature_set}_balanced_seed_{seed}.joblib"
            joblib.dump(pipeline, model_path)
            model_rows.append({
                "feature_set": feature_set,
                "seed": seed,
                "model_family": config["model_family"],
                "n_hsa": int((training.protein == config["negative_label"]).sum()),
                "n_f8": int((training.protein == config["positive_label"]).sum()),
                "n_total": len(training),
                "features": json.dumps(features),
                "parameters": json.dumps(parameters, sort_keys=True),
                "model_path": str(model_path.resolve()),
                "model_sha256": sha256(model_path),
            })

    pd.DataFrame(selection_rows).to_csv(
        output / "training_event_selection.csv", index=False
    )
    pd.DataFrame(model_rows).to_csv(output / "trained_models.csv", index=False)
    manifest = {
        "run_type": "balanced_pure_protein_training_for_mixture_inference",
        "created_at": datetime.now().astimezone().isoformat(),
        "python_version": platform.python_version(),
        "config_path": config["config_path"],
        "feature_table": config["feature_table"],
        "feature_table_sha256": sha256(config["feature_table"]),
        "extraction_manifest": config["extraction_manifest"],
        "extraction_manifest_sha256": sha256(config["extraction_manifest"]),
        "seeds": config["seeds"],
        "sampling_rule": "all F8 plus equal-size HSA sample without replacement per seed",
        "test_data_used_for_training": False,
        "mixture_data_used_for_training": False,
        "class_mapping": {"0": config["negative_label"], "1": config["positive_label"]},
        "n_available": {
            config["negative_label"]: len(negative),
            config["positive_label"]: len(positive),
        },
        "n_per_class_per_seed": len(positive),
        "feature_sets": config["feature_sets"],
        "limitations": [
            "HSA comes from one physical recording.",
            "The three F8 recordings come from the same pore.",
            "Mixture event probabilities are not calibrated concentration estimates."
        ],
    }
    (output / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(output)


if __name__ == "__main__":
    main()
