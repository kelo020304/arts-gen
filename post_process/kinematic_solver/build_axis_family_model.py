"""Fit a compact axis-family model from official training objects only."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re

import joblib
import numpy as np
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler

from .sdk import KinematicCandidate, estimate_motion_from_render_states, load_obj_points
from .sdk.axis_family_model import axis_family_numeric_features
from .benchmark_kin_agent import benchmark_points_in_delivery_frame


def _normalize_label(value: str) -> str:
    return re.sub(r"[^0-9a-z\u4e00-\u9fff]+", "", Path(value).stem.lower())


def build_axis_family_model(
    predictions_path: Path,
    split_path: Path,
    data_root: Path,
    output_path: Path,
) -> dict:
    prediction_bytes = Path(predictions_path).read_bytes()
    split_bytes = Path(split_path).read_bytes()
    frozen = json.loads(prediction_bytes)
    split = json.loads(split_bytes)
    train = {
        f"{item['dataset_id']}::{item['obj_id']}"
        for item in split.get("train_ids") or []
    }
    rows = []
    for raw in frozen.get("predictions") or []:
        if raw.get("dataset") != "realappliance" or raw.get("category") != "knob":
            continue
        object_key = f"{raw['dataset']}::{raw['object_id']}"
        if object_key not in train:
            continue
        candidate_payload = dict(raw["candidate"])
        candidate_payload["axis_world"] = tuple(candidate_payload["axis_world"])
        candidate_payload["origin_world"] = tuple(candidate_payload["origin_world"])
        candidate = KinematicCandidate(**candidate_payload)
        observation = estimate_motion_from_render_states(
            Path(data_root) / raw["dataset"] / "renders" / str(raw["object_id"]),
            str(raw["label"]), "revolute",
        )
        numeric = axis_family_numeric_features(
            benchmark_points_in_delivery_frame(
                load_obj_points(Path(raw["body_mesh"])), str(raw["dataset"]),
            ),
            benchmark_points_in_delivery_frame(
                load_obj_points(Path(raw["moving_mesh"])), str(raw["dataset"]),
            ),
            candidate,
            list(raw.get("trace") or []),
            observation,
        )
        part_info = json.loads((
            Path(data_root) / raw["dataset"] / "reconstruction" / "part_info"
            / str(raw["object_id"]) / "part_info.json"
        ).read_text(encoding="utf-8"))
        wanted = _normalize_label(str(raw["label"]))
        matches = [part for label, part in part_info["parts"].items() if _normalize_label(label) == wanted]
        if len(matches) != 1:
            continue
        source_axis = np.asarray(matches[0]["joint_params"][:3], dtype=np.float64)
        axis = np.abs(np.asarray([source_axis[0], source_axis[2], -source_axis[1]], dtype=np.float64))
        rows.append((str(raw["label"]), numeric, int(np.argmax(axis)), str(raw["object_id"])))
    labels = [row[0] for row in rows]
    numeric = np.asarray([row[1] for row in rows], dtype=np.float64)
    targets = np.asarray([row[2] for row in rows], dtype=np.int64)
    groups = np.asarray([row[3] for row in rows])
    vectorizer = TfidfVectorizer(analyzer="char", ngram_range=(1, 4), min_df=2, sublinear_tf=True)
    text_features = vectorizer.fit_transform(labels)
    scaler = StandardScaler()
    numeric_features = scaler.fit_transform(numeric)
    features = hstack((text_features, csr_matrix(numeric_features)))
    classifier = LogisticRegression(C=0.05, max_iter=500, class_weight="balanced")
    folds = min(5, len(set(groups)))
    cv_predictions = cross_val_predict(
        classifier, features, targets, groups=groups, cv=GroupKFold(folds),
    )
    classifier.fit(features, targets)
    metadata = {
        "format": "arts_gen_kin_axis_family_lr_v2",
        "training_contract": (
            "realappliance knob samples from canonical split train_ids only; source axes are mapped "
            "through the inverse delivery-root rotation into decoded coordinates; model artifact contains "
            "vectorizer/scaler/coefficients and no per-object GT rows"
        ),
        "training_samples": len(rows),
        "training_unique_objects": len(set(groups)),
        "group_cv_folds": folds,
        "group_cv_accuracy": float(accuracy_score(targets, cv_predictions)),
        "split_sha256": hashlib.sha256(split_bytes).hexdigest(),
        "base_predictions_sha256": hashlib.sha256(prediction_bytes).hexdigest(),
        "numeric_feature_count": int(numeric.shape[1]),
        "classes": [int(value) for value in classifier.classes_],
    }
    artifact = {
        **metadata,
        "vectorizer": vectorizer,
        "scaler": scaler,
        "classifier": classifier,
    }
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
    print(json.dumps(build_axis_family_model(
        args.predictions, args.split, args.data_root, args.output,
    ), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
