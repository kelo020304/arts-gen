"""Fit a static visual-relation axis proposal model from official train objects."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler

from .benchmark_kin_agent import (
    _find_gt_part,
    benchmark_points_in_delivery_frame,
    mechanical_category,
)
from .sdk import KinematicCandidate, estimate_static_part_observation, load_obj_points
from .sdk.axis_family_model import axis_family_numeric_features


def _decoded_axis_family(dataset: str, raw_axis) -> int:
    x, y, z = (float(value) for value in raw_axis)
    if dataset == "realappliance":
        axis = np.asarray([x, z, -y], dtype=np.float64)
    elif dataset == "physx-0511-drawer-door":
        axis = np.asarray([x, -z, y], dtype=np.float64)
    else:
        axis = np.asarray([x, y, z], dtype=np.float64)
    return int(np.argmax(np.abs(axis)))


def _select_threshold(categories, targets, base_families, probabilities, classes):
    result = {}
    predictions = classes[np.argmax(probabilities, axis=1)]
    confidence = np.max(probabilities, axis=1)
    for category in sorted(set(categories)):
        indices = np.asarray([index for index, value in enumerate(categories) if value == category])
        best = None
        for threshold in np.arange(0.45, 0.951, 0.025):
            final = np.asarray(base_families, dtype=np.int64).copy()
            selected = indices[confidence[indices] >= threshold]
            final[selected] = predictions[selected]
            accuracy = float(np.mean(final[indices] == np.asarray(targets)[indices]))
            proposal_precision = float(np.mean(predictions[selected] == np.asarray(targets)[selected])) if len(selected) else 1.0
            score = (accuracy, proposal_precision, float(threshold))
            if best is None or score > best[0]:
                best = (score, float(threshold), int(len(selected)), accuracy, proposal_precision)
        result[category] = {
            "threshold": best[1], "selected": best[2],
            "final_accuracy": best[3], "proposal_precision": best[4],
        }
    return result


def build_static_axis_family_model(
    predictions_path: Path,
    split_path: Path,
    data_root: Path,
    output_path: Path,
) -> dict:
    prediction_bytes = Path(predictions_path).read_bytes()
    split_bytes = Path(split_path).read_bytes()
    frozen = json.loads(prediction_bytes)
    split = json.loads(split_bytes)
    train = {f"{item['dataset_id']}::{item['obj_id']}" for item in split.get("train_ids") or []}
    rows = []
    body_cache = {}
    moving_cache = {}
    for raw in frozen.get("predictions") or []:
        dataset = str(raw.get("dataset") or "")
        category = str(raw.get("category") or mechanical_category(str(raw.get("label") or "")) or "")
        if dataset != "realappliance" or category not in {"door", "lid", "knob"}:
            continue
        if f"{dataset}::{raw['object_id']}" not in train:
            continue
        observation = estimate_static_part_observation(
            Path(data_root) / dataset / "renders" / str(raw["object_id"]),
            str(raw["label"]),
        )
        if observation is None:
            continue
        candidate_payload = dict(raw["candidate"])
        candidate_payload["axis_world"] = tuple(candidate_payload["axis_world"])
        candidate_payload["origin_world"] = tuple(candidate_payload["origin_world"])
        candidate = KinematicCandidate(**candidate_payload)
        body_key = str(raw["body_mesh"])
        moving_key = str(raw["moving_mesh"])
        if body_key not in body_cache:
            body_cache[body_key] = benchmark_points_in_delivery_frame(
                load_obj_points(Path(body_key)), dataset,
            )
        if moving_key not in moving_cache:
            moving_cache[moving_key] = benchmark_points_in_delivery_frame(
                load_obj_points(Path(moving_key)), dataset,
            )
        numeric = axis_family_numeric_features(
            body_cache[body_key], moving_cache[moving_key], candidate,
            list(raw.get("trace") or []), observation,
        )
        part_info = json.loads((
            Path(data_root) / dataset / "reconstruction" / "part_info"
            / str(raw["object_id"]) / "part_info.json"
        ).read_text(encoding="utf-8"))
        _key, gt = _find_gt_part(part_info["parts"], str(raw["label"]))
        rows.append({
            "text": f"{category} {raw['label']}", "category": category,
            "numeric": numeric, "target": _decoded_axis_family(dataset, gt["joint_params"][:3]),
            "base_family": int(np.argmax(np.abs(np.asarray(candidate.axis_world)))),
            "object_id": str(raw["object_id"]),
        })
    labels = [row["text"] for row in rows]
    numeric = np.asarray([row["numeric"] for row in rows], dtype=np.float64)
    targets = np.asarray([row["target"] for row in rows], dtype=np.int64)
    groups = np.asarray([row["object_id"] for row in rows])
    vectorizer = TfidfVectorizer(analyzer="char", ngram_range=(1, 4), min_df=2, sublinear_tf=True)
    text_features = vectorizer.fit_transform(labels)
    scaler = StandardScaler()
    numeric_features = scaler.fit_transform(numeric)
    features = hstack((text_features, csr_matrix(numeric_features)))
    classifier = LogisticRegression(C=0.05, max_iter=1000, class_weight="balanced")
    folds = min(5, len(set(groups)))
    cv_probabilities = cross_val_predict(
        classifier, features, targets, groups=groups, cv=GroupKFold(folds), method="predict_proba",
    )
    classes = np.unique(targets)
    threshold_audit = _select_threshold(
        [row["category"] for row in rows], targets,
        [row["base_family"] for row in rows], cv_probabilities, classes,
    )
    classifier.fit(features, targets)
    category_thresholds = {
        key: float(value["threshold"]) for key, value in threshold_audit.items()
    }
    metadata = {
        "format": "arts_gen_kin_static_axis_family_lr_v1",
        "training_contract": (
            "RealAppliance door/lid/knob samples from canonical train_ids only; features use decoded "
            "SLat geometry plus one articulated state of calibrated bbox/camera observations"
        ),
        "training_samples": len(rows),
        "training_unique_objects": len(set(groups)),
        "group_cv_folds": folds,
        "category_thresholds": category_thresholds,
        "category_cv_audit": threshold_audit,
        "split_sha256": hashlib.sha256(split_bytes).hexdigest(),
        "base_predictions_sha256": hashlib.sha256(prediction_bytes).hexdigest(),
        "numeric_feature_count": int(numeric.shape[1]),
        "classes": [int(value) for value in classifier.classes_],
    }
    artifact = {
        **metadata, "vectorizer": vectorizer, "scaler": scaler, "classifier": classifier,
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
    print(json.dumps(build_static_axis_family_model(
        args.predictions, args.split, args.data_root, args.output,
    ), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
