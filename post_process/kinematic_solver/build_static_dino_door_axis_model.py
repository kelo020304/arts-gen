"""Build the train-only static DINO door hinge-family proposal model."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .benchmark_kin_agent import _find_gt_part
from .sdk.static_dino_features import pool_static_part_dino_feature


VIEW_SETS = ((0, 3, 6, 9), (0, 4, 7, 11), (1, 4, 7, 10), (0, 3, 8, 11))


def build_static_dino_door_axis_model(
    predictions_path: Path,
    split_path: Path,
    data_root: Path,
    output_path: Path,
) -> dict:
    prediction_bytes = Path(predictions_path).read_bytes()
    split_bytes = Path(split_path).read_bytes()
    predictions = json.loads(prediction_bytes)
    split = json.loads(split_bytes)
    train = {
        f"{item['dataset_id']}::{item['obj_id']}"
        for item in split.get("train_ids") or []
    }
    rows = []
    for raw in predictions.get("predictions") or []:
        if raw.get("dataset") != "realappliance" or raw.get("category") != "door":
            continue
        object_id = str(raw["object_id"])
        if f"realappliance::{object_id}" not in train:
            continue
        part_info = json.loads((
            Path(data_root) / "realappliance" / "reconstruction" / "part_info"
            / object_id / "part_info.json"
        ).read_text(encoding="utf-8"))
        _key, gt = _find_gt_part(part_info["parts"], str(raw["label"]))
        x, y, z = (float(value) for value in gt["joint_params"][:3])
        target = int(np.argmax(np.abs(np.asarray([x, z, -y], dtype=np.float64))))
        render_root = Path(data_root) / "realappliance" / "renders" / object_id
        for view_set in VIEW_SETS:
            feature = pool_static_part_dino_feature(
                render_root, str(raw["label"]), view_indices=view_set,
            )
            if feature is not None:
                rows.append((feature.feature, target, object_id))
    features = np.asarray([row[0] for row in rows], dtype=np.float64)
    targets = np.asarray([row[1] for row in rows], dtype=np.int64)
    groups = np.asarray([row[2] for row in rows])
    model = make_pipeline(
        StandardScaler(),
        PCA(4, whiten=True, random_state=0),
        LogisticRegression(C=1.0, max_iter=1000, class_weight="balanced"),
    )
    probabilities = cross_val_predict(
        model, features, targets, groups=groups,
        cv=GroupKFold(min(5, len(set(groups)))), method="predict_proba",
    )
    classes = np.unique(targets)
    predictions_cv = classes[np.argmax(probabilities, axis=1)]
    confidence = np.max(probabilities, axis=1)
    threshold_audit = []
    for threshold in np.arange(0.50, 0.901, 0.025):
        selected = confidence >= threshold
        precision = float(np.mean(predictions_cv[selected] == targets[selected])) if selected.any() else 1.0
        coverage = float(np.mean(selected))
        threshold_audit.append({
            "threshold": float(threshold), "precision": precision, "coverage": coverage,
        })
    eligible = [row for row in threshold_audit if row["precision"] >= 0.90]
    chosen = max(eligible, key=lambda row: (row["coverage"], row["threshold"])) if eligible else {
        "threshold": 0.90, "precision": 0.0, "coverage": 0.0,
    }
    model.fit(features, targets)
    metadata = {
        "format": "arts_gen_kin_static_dino_door_axis_v1",
        "training_contract": (
            "RealAppliance door samples from canonical train_ids only; cached DINO patch tokens are "
            "pooled inside four selected part boxes and differenced from whole-object tokens"
        ),
        "training_rows": len(rows),
        "training_unique_objects": len(set(groups)),
        "view_sets": [list(values) for values in VIEW_SETS],
        "group_cv_accuracy": float(np.mean(predictions_cv == targets)),
        "confidence_threshold": float(chosen["threshold"]),
        "threshold_cv_precision": float(chosen["precision"]),
        "threshold_cv_coverage": float(chosen["coverage"]),
        "threshold_audit": threshold_audit,
        "split_sha256": hashlib.sha256(split_bytes).hexdigest(),
        "base_predictions_sha256": hashlib.sha256(prediction_bytes).hexdigest(),
        "classes": [int(value) for value in model.classes_],
    }
    artifact = {**metadata, "model": model}
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, output_path)
    output_path.with_suffix(".json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8",
    )
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build_static_dino_door_axis_model(
        args.predictions, args.split, args.data_root, args.output,
    ), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
