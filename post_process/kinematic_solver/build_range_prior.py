"""Build frozen signed range priors and train-only observation calibrators."""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import re

import numpy as np

from .benchmark_kin_agent import mechanical_category
from .sdk.range_prior import RANGE_CALIBRATOR_FEATURES, range_calibration_features


SIGNED_INTERVAL_CELLS = {
    ("phyx-verse", "knob", "revolute"),
    ("realappliance", "knob", "revolute"),
    ("phyx-verse", "lid", "revolute"),
    ("realappliance", "lid", "revolute"),
}
PHYSX_0511 = "physx-0511-drawer-door"
PHYSX_0511_DIR = "PhysX-Mobility-full-4view-0511"


def resolve_dataset_root(data_root: Path, dataset: str) -> Path:
    """Resolve canonical profiles, including the nested PhysX-0511 layout."""
    root = Path(data_root)
    if dataset != PHYSX_0511:
        return root / dataset
    candidates = (
        root / PHYSX_0511_DIR / PHYSX_0511_DIR,
        root / PHYSX_0511_DIR,
        root / PHYSX_0511,
        root,
    )
    for candidate in candidates:
        if (candidate / "reconstruction" / "part_info").is_dir():
            return candidate
    return candidates[0]


def build_range_prior(
    split_path: Path,
    data_root: Path,
    output_path: Path,
    *,
    calibration_predictions: Path | None = None,
    calibration_metrics: Path | None = None,
) -> dict:
    split_bytes = Path(split_path).read_bytes()
    split = json.loads(split_bytes)
    train_objects = sorted({
        (str(item["dataset_id"]), str(item["obj_id"]))
        for item in split.get("train_ids") or []
    })
    train_lookup = set(train_objects)
    per_cell_object: dict[
        tuple[str, str, str], dict[str, list[tuple[float, float]]]
    ] = defaultdict(lambda: defaultdict(list))
    missing = 0
    accepted_parts = 0
    for dataset, object_id in train_objects:
        part_info = (
            resolve_dataset_root(data_root, dataset)
            / "reconstruction" / "part_info" / object_id / "part_info.json"
        )
        if not part_info.is_file():
            missing += 1
            continue
        parts = json.loads(part_info.read_text(encoding="utf-8")).get("parts") or {}
        for label, part in parts.items():
            joint_type = str(part.get("joint") or "").lower()
            params = part.get("joint_params") or []
            category = mechanical_category(str(label), str(part.get("type") or ""))
            if category is None or joint_type not in {"prismatic", "revolute"} or len(params) < 8:
                continue
            lower, upper = _canonical_decoded_interval(dataset, params[:3], params[6], params[7])
            span = upper - lower
            if not np.isfinite(span) or span <= 1e-6:
                continue
            per_cell_object[(dataset, category, joint_type)][object_id].append((lower, upper))
            accepted_parts += 1
    cells = {}
    for key, by_object in sorted(per_cell_object.items()):
        object_values = np.asarray([
            np.median(np.asarray(values, dtype=np.float64), axis=0)
            for values in by_object.values() if values
        ], dtype=np.float64)
        if not len(object_values):
            continue
        lower_values = object_values[:, 0]
        upper_values = object_values[:, 1]
        spans = upper_values - lower_values
        dataset, category, joint_type = key
        runtime_strategy = "signed_interval" if key in SIGNED_INTERVAL_CELLS else "none"
        if category == "door":
            runtime_strategy = "span_envelope"
        if key == (PHYSX_0511, "drawer", "prismatic"):
            runtime_strategy = "ridge_observed_span"
        cells["|".join(key)] = {
            "dataset": dataset,
            "category": category,
            "joint_type": joint_type,
            "unit": "asset_unit" if joint_type == "prismatic" else "rad",
            "scale_mode": "normalized_asset_units" if joint_type == "prismatic" else "radians",
            "canonical_frame": "decoded_dominant_axis_positive",
            "n_objects": int(len(object_values)),
            **_span_quantiles(spans),
            "lower": _named_quantiles(lower_values),
            "upper": _named_quantiles(upper_values),
            "runtime_strategy": runtime_strategy,
            "usable": bool(len(object_values) >= 8),
        }

    calibrators = {}
    calibration_audit = None
    if calibration_predictions is not None:
        key = (PHYSX_0511, "drawer", "prismatic")
        calibrator, calibration_audit = _build_physx_0511_drawer_calibrator(
            Path(calibration_predictions), Path(data_root), train_lookup,
            metrics_path=Path(calibration_metrics) if calibration_metrics is not None else None,
        )
        if calibrator is not None:
            calibrators["|".join(key)] = calibrator

    payload = {
        "format": "arts_gen_kin_range_prior_v3",
        "training_contract": (
            "object-balanced signed endpoint quantiles from canonical split train_ids; "
            "optional Ridge calibration uses only frozen decoded geometry and bbox/camera "
            "trajectory features at runtime; artifact contains no per-object rows"
        ),
        "split_sha256": hashlib.sha256(split_bytes).hexdigest(),
        "train_object_count": len(train_objects),
        "objects_missing_part_info": missing,
        "accepted_training_parts": accepted_parts,
        "minimum_objects_per_usable_cell": 8,
        "cells": cells,
        "calibrators": calibrators,
        "calibration_audit": calibration_audit,
    }
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return payload


def _build_physx_0511_drawer_calibrator(
    predictions_path: Path,
    data_root: Path,
    train_lookup: set[tuple[str, str]],
    *,
    metrics_path: Path | None = None,
) -> tuple[dict | None, dict]:
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import GroupKFold
    from sklearn.preprocessing import StandardScaler

    frozen = json.loads(predictions_path.read_text(encoding="utf-8"))
    metric_diagonals = {}
    if metrics_path is not None:
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        metric_diagonals = {
            str(row["sample_id"]): float(row["object_diagonal"])
            for row in metrics.get("samples") or []
        }
    rows = []
    cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for row in frozen.get("predictions") or []:
        dataset = str(row.get("dataset") or "")
        object_id = str(row.get("object_id") or "")
        category = str(row.get("category") or mechanical_category(str(row.get("label") or "")) or "")
        candidate = row.get("candidate") or {}
        if (
            dataset != PHYSX_0511
            or category != "drawer"
            or str(candidate.get("joint_type") or "") != "prismatic"
            or (dataset, object_id) not in train_lookup
        ):
            continue
        gt_path = (
            resolve_dataset_root(data_root, dataset)
            / "reconstruction" / "part_info" / object_id / "part_info.json"
        )
        parts = json.loads(gt_path.read_text(encoding="utf-8")).get("parts") or {}
        gt = _find_exact_part(parts, str(row.get("label") or ""))
        params = gt.get("joint_params") or []
        if str(gt.get("joint") or "") != "prismatic" or len(params) < 8:
            continue
        lower, upper = _canonical_decoded_interval(dataset, params[:3], params[6], params[7])
        object_diagonal = metric_diagonals.get(str(row.get("sample_id") or ""))
        if object_diagonal is None:
            body_bounds = _obj_bounds(Path(str(row["body_mesh"])), cache)
            moving_bounds = _obj_bounds(Path(str(row["moving_mesh"])), cache)
            object_diagonal = float(np.linalg.norm(
                np.maximum(body_bounds[1], moving_bounds[1])
                - np.minimum(body_bounds[0], moving_bounds[0])
            ))
        observation = row.get("motion_observation") or {}
        features = range_calibration_features(
            candidate.get("signals") or {},
            observation.get("diagnostics") or {},
            object_diagonal,
        )
        rows.append((object_id, features, upper - lower))

    groups = np.asarray([row[0] for row in rows])
    features = np.asarray([row[1] for row in rows], dtype=np.float64)
    targets = np.asarray([row[2] for row in rows], dtype=np.float64)
    unique_groups = sorted(set(groups.tolist()))
    audit = {
        "source_predictions_sha256": hashlib.sha256(predictions_path.read_bytes()).hexdigest(),
        "decoded_scale_cache_sha256": (
            hashlib.sha256(metrics_path.read_bytes()).hexdigest() if metrics_path is not None else None
        ),
        "accepted_parts": int(len(rows)),
        "train_objects": int(len(unique_groups)),
        "group_cv_folds": min(5, len(unique_groups)),
    }
    if len(rows) < 8 or len(unique_groups) < 5:
        audit["usable"] = False
        return None, audit

    fold_count = min(5, len(unique_groups))
    residuals = []
    for train_index, valid_index in GroupKFold(n_splits=fold_count).split(features, targets, groups):
        scaler = StandardScaler().fit(features[train_index])
        model = Ridge(alpha=10.0).fit(scaler.transform(features[train_index]), targets[train_index])
        prediction = model.predict(scaler.transform(features[valid_index]))
        residuals.extend(np.abs(prediction - targets[valid_index]).tolist())
    scaler = StandardScaler().fit(features)
    model = Ridge(alpha=10.0).fit(scaler.transform(features), targets)
    q50, q80, q90, q95 = np.quantile(np.asarray(residuals), [0.50, 0.80, 0.90, 0.95])
    target_q05, target_q95 = np.quantile(targets, [0.05, 0.95])
    calibrator = {
        "format": "ridge_standardized_v1",
        "training_inputs": "decoded geometry scale plus bbox/camera trajectory diagnostics",
        "feature_names": list(RANGE_CALIBRATOR_FEATURES),
        "ridge_alpha": 10.0,
        "scaler_mean": scaler.mean_.tolist(),
        "scaler_scale": scaler.scale_.tolist(),
        "coefficients": model.coef_.tolist(),
        "intercept": float(model.intercept_),
        "prediction_clip": [float(max(0.0, target_q05)), float(target_q95)],
        "interval_anchor": "upper_zero",
        "absolute_span_residual_quantiles": {
            "q50": float(q50), "q80": float(q80),
            "q90": float(q90), "q95": float(q95),
        },
        "train_parts": int(len(rows)),
        "train_objects": int(len(unique_groups)),
        "group_cv_folds": int(fold_count),
        "group_cv_span_mae": float(np.mean(residuals)),
        "group_cv_endpoint_mae": float(0.5 * np.mean(residuals)),
    }
    audit.update({"usable": True, **{
        key: calibrator[key]
        for key in ("group_cv_span_mae", "group_cv_endpoint_mae")
    }})
    return calibrator, audit


def _canonical_decoded_interval(dataset: str, axis_values, lower_value, upper_value):
    axis = np.asarray(axis_values, dtype=np.float64)
    if dataset == PHYSX_0511:
        x, y, z = axis
        axis = np.asarray([x, -z, y], dtype=np.float64)
    elif dataset == "realappliance":
        x, y, z = axis
        axis = np.asarray([x, z, -y], dtype=np.float64)
    norm = float(np.linalg.norm(axis))
    if norm <= 1e-12:
        raise ValueError("zero-length training joint axis")
    axis /= norm
    lower, upper = float(lower_value), float(upper_value)
    if axis[int(np.argmax(np.abs(axis)))] < 0.0:
        lower, upper = -upper, -lower
    return lower, upper


def _named_quantiles(values: np.ndarray) -> dict:
    q10, q25, q50, q75, q90 = np.quantile(values, [0.10, 0.25, 0.50, 0.75, 0.90])
    return {
        "q10": float(q10), "q25": float(q25), "q50": float(q50),
        "q75": float(q75), "q90": float(q90),
    }


def _span_quantiles(values: np.ndarray) -> dict:
    result = _named_quantiles(values)
    result["mad"] = float(np.median(np.abs(values - np.median(values))))
    return result


def _normalize_label(value: str) -> str:
    value = Path(value).stem.lower()
    value = re.sub(r"^part_\d+_", "", value)
    return re.sub(r"[^0-9a-z\u4e00-\u9fff]+", "", value)


def _find_exact_part(parts: dict, label: str) -> dict:
    wanted = _normalize_label(label)
    matches = [part for key, part in parts.items() if _normalize_label(key) == wanted]
    if len(matches) != 1:
        raise ValueError(f"expected one GT training part for {label!r}, found {len(matches)}")
    return matches[0]


def _obj_bounds(
    path: Path,
    cache: dict[str, tuple[np.ndarray, np.ndarray]],
) -> tuple[np.ndarray, np.ndarray]:
    key = str(path.resolve())
    if key in cache:
        return cache[key]
    minimum = np.full(3, np.inf, dtype=np.float64)
    maximum = np.full(3, -np.inf, dtype=np.float64)
    with path.open(encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            if not line.startswith("v "):
                continue
            fields = line.split()
            point = np.asarray(fields[1:4], dtype=np.float64)
            minimum = np.minimum(minimum, point)
            maximum = np.maximum(maximum, point)
    if not np.all(np.isfinite(minimum)):
        raise ValueError(f"decoded OBJ has no vertices: {path}")
    cache[key] = (minimum, maximum)
    return cache[key]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--calibration-predictions", type=Path)
    parser.add_argument("--calibration-metrics", type=Path)
    args = parser.parse_args()
    payload = build_range_prior(
        args.split, args.data_root, args.output,
        calibration_predictions=args.calibration_predictions,
        calibration_metrics=args.calibration_metrics,
    )
    print(json.dumps({
        "format": payload["format"],
        "train_object_count": payload["train_object_count"],
        "accepted_training_parts": payload["accepted_training_parts"],
        "objects_missing_part_info": payload["objects_missing_part_info"],
        "cells": len(payload["cells"]),
        "calibrators": len(payload["calibrators"]),
    }, indent=2))


if __name__ == "__main__":
    main()
