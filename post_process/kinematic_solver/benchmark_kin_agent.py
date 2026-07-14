"""Two-phase, benchmark-only evaluation of Kin Agent on decoded SLat meshes.

The ``infer`` command never opens dataset annotations.  The ``evaluate``
command consumes its frozen prediction JSON and only then loads GT joints.
This separation makes accidental GT leakage straightforward to audit.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import csv
from functools import lru_cache
import json
import math
from pathlib import Path
import re
import time
from typing import Iterable

import numpy as np

from .run_kin_agent import _semantic_part_category, _semantic_type_hint
from .run_kin_agent_bundle import (
    _apply_motion_observation,
    _apply_range_prior,
    _select_motion_observation,
)
from .sdk import (
    KinematicAgentConfig,
    KinematicAgentResult,
    KinematicCandidate,
    apply_axis_family_reranker,
    apply_phyx_door_contact_axis_critic,
    apply_phyx_knob_thin_axis_critic,
    apply_static_axis_family_reranker,
    apply_static_dino_door_axis_reranker,
    estimate_motion_hypotheses_from_render_states,
    estimate_static_part_observation,
    infer_kinematics,
    load_obj_points,
    load_range_prior,
    pool_static_part_dino_feature,
)


BENCHMARK_CATEGORIES = ("drawer", "door", "knob", "lid")
_BUTTON_TOKENS = ("button", "按钮", "按键")


def benchmark_points_in_delivery_frame(points: np.ndarray, dataset: str) -> np.ndarray:
    """Normalize legacy benchmark meshes to the current decoded delivery frame.

    The frozen ``0707-ra`` benchmark OBJs predate the root-body direction
    correction and have RealAppliance coordinates baked into the annotation
    frame. Current delivery keeps those vertices unbaked and applies Rx(+90)
    on the MJCF/USD root, so inference must first apply the inverse rotation.
    Other frozen benchmark profiles already use their decoded local frame.
    """
    values = np.asarray(points, dtype=np.float64).reshape((-1, 3))
    if str(dataset).lower() != "realappliance":
        return values
    return values[:, (0, 2, 1)] * np.asarray([1.0, 1.0, -1.0], dtype=np.float64)


def _normalize_label(value: str) -> str:
    value = Path(value).stem.lower()
    value = re.sub(r"^part_\d+_", "", value)
    return re.sub(r"[^0-9a-z\u4e00-\u9fff]+", "", value)


def _decoded_dir(decoded_root: Path, sample: dict) -> Path:
    dataset = sample["dataset"]
    object_id = sample["object_id"]
    angle = int(sample.get("angle", 0))
    name = f"{dataset}__{object_id}__angle_{angle:02d}__mujoco"
    directory = Path(decoded_root) / name / "assets"
    if not directory.is_dir():
        raise FileNotFoundError(f"decoded assets not found: {directory}")
    return directory


def _resolve_part(assets: Path, label: str) -> Path:
    wanted = _normalize_label(label)
    matches = [path for path in assets.glob("part_*.obj") if _normalize_label(path.name) == wanted]
    if len(matches) != 1:
        raise ValueError(f"expected one decoded part for {label!r} in {assets}, found {matches}")
    return matches[0]


def mechanical_category(label: str, part_type: str = "") -> str | None:
    """Map visible semantics to the four benchmark categories without GT joints."""
    text = f"{label} {part_type}".lower()
    blocked = any(token in text for token in _BUTTON_TOKENS)
    if any(token in text for token in ("knob", "dial", "旋钮")):
        return "knob"
    if any(token in text for token in ("drawer", "抽屉")):
        return "drawer"
    if not blocked and any(token in text for token in ("door", "门")):
        return "door"
    if not blocked and any(token in text for token in ("lid", "cover", "盖")):
        return "lid"
    return None


def build_expanded_manifest(
    decoded_roots: dict[str, Path],
    data_root: Path,
    output_path: Path,
) -> dict:
    """Discover every exact-match drawer/door/knob/lid part at one angle/object.

    Dataset annotations are used only to define the evaluation cohort.  Joint
    type, axis, origin and limits are deliberately omitted from the manifest.
    """
    samples: list[dict] = []
    audit: dict[str, dict] = {}
    for dataset, decoded_root in sorted(decoded_roots.items()):
        pattern = re.compile(rf"{re.escape(dataset)}__(.+)__angle_(\d+)__mujoco$")
        available: dict[str, list[tuple[int, Path]]] = {}
        for assets in Path(decoded_root).glob(f"{dataset}__*__angle_*__mujoco/assets"):
            match = pattern.match(assets.parent.name)
            if match:
                available.setdefault(match.group(1), []).append((int(match.group(2)), assets))
        dataset_samples: list[dict] = []
        skipped_missing_gt = 0
        skipped_non_target = 0
        for object_id, choices in sorted(available.items()):
            # Prefer the rest observation where available, otherwise use the
            # lowest decoded angle.  Every object contributes only one pose.
            angle, assets = min(choices, key=lambda item: (item[0] != 0, item[0]))
            gt_path = Path(data_root) / dataset / "reconstruction" / "part_info" / object_id / "part_info.json"
            if not gt_path.is_file():
                skipped_missing_gt += 1
                continue
            parts = json.loads(gt_path.read_text(encoding="utf-8")).get("parts", {})
            for label, part in parts.items():
                category = mechanical_category(str(label), str(part.get("type", "")))
                if category is None or str(part.get("joint", "")).lower() not in {"prismatic", "revolute"}:
                    skipped_non_target += 1
                    continue
                decoded_part = _resolve_part(assets, str(label))
                dataset_samples.append({
                    "sample_id": f"{dataset}:{object_id}:{label}",
                    "dataset": dataset,
                    "object_id": object_id,
                    "angle": angle,
                    "label": str(label),
                    "category": category,
                    "decoded_part_file": decoded_part.name,
                })
        samples.extend(dataset_samples)
        audit[dataset] = {
            "decoded_unique_objects": len(available),
            "selected_samples": len(dataset_samples),
            "selected_unique_objects": len({row["object_id"] for row in dataset_samples}),
            "unique_objects_by_category": {
                category: len({row["object_id"] for row in dataset_samples if row["category"] == category})
                for category in BENCHMARK_CATEGORIES
            },
            "samples_by_category": {
                category: sum(row["category"] == category for row in dataset_samples)
                for category in BENCHMARK_CATEGORIES
            },
            "missing_part_info_objects": skipped_missing_gt,
            "non_target_parts_skipped": skipped_non_target,
        }
    payload = {
        "format": "arts_gen_kin_agent_benchmark_manifest_v2",
        "selection_contract": (
            "one decoded pose per object (angle 0 preferred); all exact label-matched moving "
            "drawer/door/knob/lid parts; no joint type/axis/origin/range fields"
        ),
        "decoded_roots": {key: str(Path(value)) for key, value in sorted(decoded_roots.items())},
        "audit": audit,
        "samples": samples,
    }
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return payload


def build_blind_manifest(
    decoded_roots: dict[str, Path],
    output_path: Path,
    *,
    excluded_object_keys: set[str] | None = None,
) -> dict:
    """Freeze a decoded-only cohort before any annotation file is opened."""
    excluded = excluded_object_keys or set()
    samples = []
    audit = {}
    for dataset, decoded_root in sorted(decoded_roots.items()):
        pattern = re.compile(rf"{re.escape(dataset)}__(.+)__angle_(\d+)__mujoco$")
        available: dict[str, list[tuple[int, Path]]] = {}
        for assets in Path(decoded_root).glob(f"{dataset}__*__angle_*__mujoco/assets"):
            match = pattern.match(assets.parent.name)
            if match:
                available.setdefault(match.group(1), []).append((int(match.group(2)), assets))
        dataset_rows = []
        for object_id, choices in sorted(available.items()):
            if f"{dataset}::{object_id}" in excluded:
                continue
            angle, assets = min(choices, key=lambda item: (item[0] != 0, item[0]))
            if not (assets / "body_without_parts.obj").is_file():
                continue
            for part_path in sorted(assets.glob("part_*.obj")):
                match = re.match(r"^part_\d+_(.+)\.obj$", part_path.name)
                if not match:
                    continue
                label = match.group(1)
                category = mechanical_category(label)
                if category is None:
                    continue
                dataset_rows.append({
                    "sample_id": f"{dataset}:{object_id}:{label}",
                    "dataset": dataset,
                    "object_id": object_id,
                    "angle": angle,
                    "label": label,
                    "category": category,
                    "decoded_part_file": part_path.name,
                })
        samples.extend(dataset_rows)
        audit[dataset] = {
            "decoded_unique_objects": len(available),
            "excluded_objects": sum(f"{dataset}::{object_id}" in excluded for object_id in available),
            "selected_samples": len(dataset_rows),
            "selected_unique_objects": len({row["object_id"] for row in dataset_rows}),
            "samples_by_category": {
                category: sum(row["category"] == category for row in dataset_rows)
                for category in BENCHMARK_CATEGORIES
            },
        }
    payload = {
        "format": "arts_gen_kin_agent_blind_manifest_v1",
        "selection_contract": (
            "decoded OBJ filenames and body presence only; no part_info, joint_transforms, "
            "source USD, joint type, axis, origin or range opened before freeze"
        ),
        "decoded_roots": {key: str(Path(value)) for key, value in sorted(decoded_roots.items())},
        "excluded_object_keys": sorted(excluded),
        "audit": audit,
        "samples": samples,
    }
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return payload


def infer_manifest(
    manifest_path: Path,
    output_path: Path,
    *,
    max_iterations: int = 7,
) -> dict:
    """Run GT-free inference and freeze predictions before evaluation."""
    if not 1 <= max_iterations < 10:
        raise ValueError("max_iterations must be in [1, 9]")
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    roots = {key: Path(value) for key, value in manifest["decoded_roots"].items()}
    rows = []
    body_points_cache: dict[str, np.ndarray] = {}
    started = time.perf_counter()
    for sample in manifest["samples"]:
        dataset = str(sample["dataset"])
        assets = _decoded_dir(roots[dataset], sample)
        body_path = assets / "body_without_parts.obj"
        part_path = _resolve_part(assets, str(sample["label"]))
        hint = _semantic_type_hint(str(sample["label"]))
        body_key = str(body_path.resolve())
        if body_key not in body_points_cache:
            body_points_cache[body_key] = benchmark_points_in_delivery_frame(
                load_obj_points(body_path), dataset,
            )
        part_started = time.perf_counter()
        result = infer_kinematics(
            benchmark_points_in_delivery_frame(load_obj_points(part_path), dataset),
            body_points_cache[body_key],
            config=KinematicAgentConfig(max_iterations=max_iterations),
            joint_type_hint=hint,
            part_category=_semantic_part_category(str(sample["label"])),
            dataset_profile=dataset,
        )
        rows.append({
            "sample_id": sample.get("sample_id") or f"{dataset}:{sample['object_id']}:{sample['label']}",
            "dataset": dataset,
            "object_id": str(sample["object_id"]),
            "angle": int(sample.get("angle", 0)),
            "category": sample.get("category"),
            "label": str(sample["label"]),
            "body_mesh": str(body_path),
            "moving_mesh": str(part_path),
            "semantic_type_hint": hint,
            "ra_prismatic_export_local_z": dataset == "realappliance",
            "benchmark_mesh_frame": (
                "legacy_baked_annotation_normalized_to_delivery"
                if dataset == "realappliance" else "decoded_delivery"
            ),
            "iterations": result.iterations,
            "runtime_seconds": time.perf_counter() - part_started,
            "candidate": asdict(result.candidate),
            "trace": result.trace,
        })
    payload = {
        "format": "arts_gen_kin_agent_benchmark_predictions_v1",
        "input_contract": (
            "decoded SLat meshes only; legacy baked RealAppliance benchmark vertices are normalized "
            "to the current unbaked delivery frame; annotations are not opened during inference"
        ),
        "manifest": str(Path(manifest_path).resolve()),
        "max_iterations": max_iterations,
        "runtime_seconds": time.perf_counter() - started,
        "predictions": rows,
    }
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return payload


def refine_predictions_with_observations(
    predictions_path: Path,
    output_path: Path,
    observation_roots: dict[str, Path],
    static_observation_roots: dict[str, Path] | None = None,
    static_view_indices: dict[str, tuple[int, ...] | list[int]] | None = None,
) -> dict:
    """Apply legal multi-state observations and frozen priors to predictions."""
    frozen = json.loads(Path(predictions_path).read_text(encoding="utf-8"))
    static_observation_roots = static_observation_roots or {}
    static_view_indices = static_view_indices or {}
    rows = []
    mesh_points_cache: dict[str, np.ndarray] = {}
    started = time.perf_counter()
    for raw in frozen["predictions"]:
        row = dict(raw)
        candidate_payload = dict(row["candidate"])
        candidate_payload["axis_world"] = tuple(candidate_payload["axis_world"])
        candidate_payload["origin_world"] = tuple(candidate_payload["origin_world"])
        result = KinematicAgentResult(
            candidate=KinematicCandidate(**candidate_payload),
            iterations=int(row["iterations"]),
            trace=list(row.get("trace") or []),
        )
        category = str(row.get("category") or _semantic_part_category(str(row["label"])) or "")
        observation = None
        static_observation = None
        motion_hypotheses = {}
        render_base = observation_roots.get(str(row["dataset"]))
        if render_base is not None:
            motion_hypotheses = estimate_motion_hypotheses_from_render_states(
                Path(render_base) / str(row["object_id"]),
                str(row["label"]),
            )
            observation = _select_motion_observation(
                result, motion_hypotheses, part_category=category,
            )
        static_render_base = static_observation_roots.get(str(row["dataset"]))
        if observation is None and static_render_base is not None:
            static_observation = estimate_static_part_observation(
                Path(static_render_base) / str(row["object_id"]),
                str(row["label"]),
            )
        if observation is not None and category != "knob":
            result = _apply_motion_observation(
                result, observation, max_iterations=int(frozen.get("max_iterations", 7)),
                part_category=category,
            )
        axis_family_model = None
        static_axis_family_model = None
        static_dino_door_axis_model = None
        phyx_thin_axis_critic = None
        phyx_door_contact_axis_critic = None
        if str(row["dataset"]) == "realappliance" and category == "knob":
            body_key = str(Path(row["body_mesh"]).resolve())
            moving_key = str(Path(row["moving_mesh"]).resolve())
            if body_key not in mesh_points_cache:
                mesh_points_cache[body_key] = benchmark_points_in_delivery_frame(
                    load_obj_points(Path(body_key)), str(row["dataset"]),
                )
            if moving_key not in mesh_points_cache:
                mesh_points_cache[moving_key] = benchmark_points_in_delivery_frame(
                    load_obj_points(Path(moving_key)), str(row["dataset"]),
                )
            result, axis_family_model = apply_axis_family_reranker(
                result,
                label=str(row["label"]),
                body_points=mesh_points_cache[body_key],
                moving_points=mesh_points_cache[moving_key],
                observation=observation or static_observation,
                max_iterations=int(frozen.get("max_iterations", 7)),
            )
            if axis_family_model is None:
                result, phyx_thin_axis_critic = apply_phyx_knob_thin_axis_critic(
                    result,
                    dataset_id=str(row["dataset"]),
                    part_category=category,
                    moving_points=mesh_points_cache[moving_key],
                    max_iterations=int(frozen.get("max_iterations", 7)),
                    min_confidence=0.95,
                    max_score_drop=0.15,
                    allowed_dataset_ids=("realappliance",),
                )
        elif (
            str(row["dataset"]) == "realappliance"
            and category == "door"
            and observation is None
            and static_render_base is not None
        ):
            dino_feature = pool_static_part_dino_feature(
                Path(static_render_base) / str(row["object_id"]),
                str(row["label"]),
                view_indices=static_view_indices.get("realappliance", (0, 3, 8, 11)),
            )
            result, static_dino_door_axis_model = apply_static_dino_door_axis_reranker(
                result,
                dino_feature=dino_feature,
                max_iterations=int(frozen.get("max_iterations", 7)),
            )
        elif (
            str(row["dataset"]) == "realappliance"
            and category == "lid"
            and observation is None
        ):
            body_key = str(Path(row["body_mesh"]).resolve())
            moving_key = str(Path(row["moving_mesh"]).resolve())
            if body_key not in mesh_points_cache:
                mesh_points_cache[body_key] = benchmark_points_in_delivery_frame(
                    load_obj_points(Path(body_key)), str(row["dataset"]),
                )
            if moving_key not in mesh_points_cache:
                mesh_points_cache[moving_key] = benchmark_points_in_delivery_frame(
                    load_obj_points(Path(moving_key)), str(row["dataset"]),
                )
            result, static_axis_family_model = apply_static_axis_family_reranker(
                result,
                label=str(row["label"]),
                category=category,
                body_points=mesh_points_cache[body_key],
                moving_points=mesh_points_cache[moving_key],
                static_observation=static_observation,
                max_iterations=int(frozen.get("max_iterations", 7)),
            )
        elif str(row["dataset"]) == "phyx-verse" and category == "knob":
            moving_key = str(Path(row["moving_mesh"]).resolve())
            if moving_key not in mesh_points_cache:
                mesh_points_cache[moving_key] = load_obj_points(Path(moving_key))
            result, phyx_thin_axis_critic = apply_phyx_knob_thin_axis_critic(
                result,
                dataset_id=str(row["dataset"]),
                part_category=category,
                moving_points=mesh_points_cache[moving_key],
                max_iterations=int(frozen.get("max_iterations", 7)),
            )
        elif (
            str(row["dataset"]) == "phyx-verse"
            and category == "door"
            and observation is None
        ):
            body_key = str(Path(row["body_mesh"]).resolve())
            moving_key = str(Path(row["moving_mesh"]).resolve())
            if body_key not in mesh_points_cache:
                mesh_points_cache[body_key] = load_obj_points(Path(body_key))
            if moving_key not in mesh_points_cache:
                mesh_points_cache[moving_key] = load_obj_points(Path(moving_key))
            result, phyx_door_contact_axis_critic = apply_phyx_door_contact_axis_critic(
                result,
                dataset_id=str(row["dataset"]),
                part_category=category,
                body_points=mesh_points_cache[body_key],
                moving_points=mesh_points_cache[moving_key],
                max_iterations=int(frozen.get("max_iterations", 7)),
            )
        prior = load_range_prior(str(row["dataset"]), category, result.candidate.joint_type)
        range_calibration = None
        if prior is not None:
            result = _apply_range_prior(
                result, prior, max_iterations=int(frozen.get("max_iterations", 7)),
                object_diagonal=_decoded_object_diagonal(row),
                observation_diagnostics=(
                    observation.diagnostics if observation is not None else None
                ),
            )
            range_calibration = next((
                trace_row.get("range_calibration")
                for trace_row in reversed(result.trace)
                if isinstance(trace_row.get("range_calibration"), dict)
                and trace_row["range_calibration"].get("applied")
            ), None)
        row.update({
            "iterations": result.iterations,
            "candidate": asdict(result.candidate),
            "trace": result.trace,
            "motion_observation": observation.to_dict() if observation is not None else None,
            "static_observation": static_observation.to_dict() if static_observation is not None else None,
            "motion_observation_hypotheses": {
                key: value.to_dict() for key, value in motion_hypotheses.items()
            },
            "axis_family_model": axis_family_model,
            "static_axis_family_model": static_axis_family_model,
            "static_dino_door_axis_model": static_dino_door_axis_model,
            "phyx_knob_thin_axis_critic": phyx_thin_axis_critic,
            "phyx_door_contact_axis_critic": phyx_door_contact_axis_critic,
            "range_prior": prior.to_dict() if prior is not None else None,
            "range_calibration": range_calibration,
        })
        rows.append(row)
    payload = {
        "format": "arts_gen_kin_agent_benchmark_predictions_v2",
        "input_contract": (
            "frozen decoded-mesh predictions plus calibrated multi-state 2D boxes/cameras and "
            "a canonical-train-only range prior; no joint annotations are opened"
        ),
        "base_predictions": str(Path(predictions_path).resolve()),
        "observation_roots": {key: str(Path(value).resolve()) for key, value in observation_roots.items()},
        "static_observation_roots": {
            key: str(Path(value).resolve()) for key, value in static_observation_roots.items()
        },
        "static_view_indices": {
            key: [int(value) for value in values] for key, values in static_view_indices.items()
        },
        "max_iterations": int(frozen.get("max_iterations", 7)),
        "runtime_seconds": time.perf_counter() - started,
        "predictions": rows,
    }
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return payload


def axis_angular_error_deg(predicted_local: Iterable[float], gt_world: Iterable[float]) -> float:
    pred = _unit(predicted_local)
    gt = _unit(gt_world)
    return math.degrees(math.acos(float(np.clip(abs(pred @ gt), -1.0, 1.0))))


def origin_line_distance(
    predicted_origin_local: Iterable[float],
    predicted_axis_local: Iterable[float],
    gt_origin_world: Iterable[float],
    gt_axis_world: Iterable[float],
) -> float:
    p0 = np.asarray(list(predicted_origin_local), dtype=np.float64)
    p1 = np.asarray(list(gt_origin_world), dtype=np.float64)
    a0 = _unit(predicted_axis_local)
    a1 = _unit(gt_axis_world)
    normal = np.cross(a0, a1)
    norm = float(np.linalg.norm(normal))
    delta = p1 - p0
    if norm < 1e-8:
        return float(np.linalg.norm(np.cross(delta, a0)))
    return abs(float(delta @ normal)) / norm


def origin_gt_axis_perpendicular_offset(
    predicted_origin: Iterable[float],
    gt_origin: Iterable[float],
    gt_axis: Iterable[float],
) -> float:
    """Origin offset after removing the unidentifiable coordinate along GT axis."""
    delta = np.asarray(list(predicted_origin), dtype=np.float64) - np.asarray(list(gt_origin), dtype=np.float64)
    axis = _unit(gt_axis)
    perpendicular = delta - axis * float(delta @ axis)
    return float(np.linalg.norm(perpendicular))


def aligned_range_endpoint_errors(candidate: dict, gt_params: list[float]) -> tuple[float, float, float]:
    pred_axis = _unit(candidate["axis_world"])
    gt_axis = _unit(gt_params[:3])
    lower, upper = float(candidate["lower"]), float(candidate["upper"])
    if float(pred_axis @ gt_axis) < 0.0:
        lower, upper = -upper, -lower
    lower_error = abs(lower - float(gt_params[6]))
    upper_error = abs(upper - float(gt_params[7]))
    return lower_error, upper_error, 0.5 * (lower_error + upper_error)


def prediction_in_annotation_frame(candidate: dict, dataset: str) -> dict:
    """Map decoded/render-frame predictions into each dataset annotation frame.

    PhysX-0511 inherits the project's exported-OBJ Y-up coordinates, while
    ``part_info`` and SAM3D source annotations are Z-up, so evaluation applies
    ``(x, y, z) -> (x, z, -y)``. RealAppliance delivery puts Rx(+90deg) on the
    MJCF/USD root object; evaluation applies that same root transform,
    ``(x, y, z) -> (x, -z, y)``. Inference and delivered joints remain in the
    decoded mesh-local frame.
    """
    result = dict(candidate)
    dataset = str(dataset).lower()
    if dataset not in {"physx-0511-drawer-door", "realappliance"}:
        return result

    def decoded_to_annotation(raw: Iterable[float]) -> list[float]:
        x, y, z = (float(value) for value in raw)
        return [x, z, -y]

    if dataset == "realappliance":
        def decoded_to_annotation(raw: Iterable[float]) -> list[float]:
            x, y, z = (float(value) for value in raw)
            return [x, -z, y]

    result["axis_world"] = decoded_to_annotation(candidate["axis_world"])
    result["origin_world"] = decoded_to_annotation(candidate["origin_world"])
    return result


def _unit(values: Iterable[float]) -> np.ndarray:
    result = np.asarray(list(values), dtype=np.float64)
    norm = float(np.linalg.norm(result))
    if norm <= 1e-12:
        raise ValueError("zero-length axis")
    return result / norm


@lru_cache(maxsize=None)
def _obj_bounds(path_value: str) -> tuple[np.ndarray, np.ndarray]:
    minimum = np.full(3, np.inf, dtype=np.float64)
    maximum = np.full(3, -np.inf, dtype=np.float64)
    found = False
    with Path(path_value).open(encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            if not line.startswith("v "):
                continue
            fields = line.split()
            if len(fields) < 4:
                continue
            point = np.asarray([float(fields[1]), float(fields[2]), float(fields[3])], dtype=np.float64)
            minimum = np.minimum(minimum, point)
            maximum = np.maximum(maximum, point)
            found = True
    if not found:
        raise ValueError(f"decoded OBJ has no vertices: {path_value}")
    return minimum, maximum


def _decoded_object_diagonal(row: dict) -> float:
    body_min, body_max = _obj_bounds(str(row["body_mesh"]))
    moving_min, moving_max = _obj_bounds(str(row["moving_mesh"]))
    return float(np.linalg.norm(
        np.maximum(body_max, moving_max) - np.minimum(body_min, moving_min)
    ))


def _find_gt_part(parts: dict, label: str) -> tuple[str, dict]:
    wanted = _normalize_label(label)
    matches = [(key, value) for key, value in parts.items() if _normalize_label(key) == wanted]
    if len(matches) != 1:
        raise ValueError(f"expected one GT part for {label!r}, found {[key for key, _ in matches]}")
    return matches[0]


def _rotation_axis(matrix: np.ndarray) -> np.ndarray | None:
    rotation = np.asarray(matrix, dtype=np.float64)[:3, :3]
    angle = math.acos(float(np.clip((np.trace(rotation) - 1.0) * 0.5, -1.0, 1.0)))
    if angle < 1e-5:
        return None
    if abs(math.sin(angle)) > 1e-5:
        return _unit((
            rotation[2, 1] - rotation[1, 2],
            rotation[0, 2] - rotation[2, 0],
            rotation[1, 0] - rotation[0, 1],
        ))
    values, vectors = np.linalg.eig(rotation)
    index = int(np.argmin(np.abs(values - 1.0)))
    axis = np.real(vectors[:, index])
    return _unit(axis) if float(np.linalg.norm(axis)) > 1e-8 else None


def annotation_frame_axis_error_deg(data_root: Path, dataset: str, object_id: str, gt: dict) -> float | None:
    """Cross-check part_info axes against independently stored pose transforms."""
    path = Path(data_root) / dataset / "joint_transforms" / f"{object_id}.json"
    if not path.is_file():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    raw_label = str(gt.get("raw_label", gt.get("part_index", "")))
    observed: list[np.ndarray] = []
    for record in (payload.get("angles") or {}).values():
        transform = (record.get("part_transforms") or {}).get(raw_label)
        if transform is None:
            continue
        matrix = np.asarray(transform, dtype=np.float64)
        if matrix.shape != (4, 4):
            continue
        if str(gt.get("joint", "")).lower() == "prismatic":
            vector = matrix[:3, 3]
            if float(np.linalg.norm(vector)) > 1e-7:
                observed.append(_unit(vector))
        else:
            axis = _rotation_axis(matrix)
            if axis is not None:
                observed.append(axis)
    if not observed:
        return None
    gt_axis = _unit(gt["joint_params"][:3])
    return float(np.median([
        math.degrees(math.acos(float(np.clip(abs(axis @ gt_axis), -1.0, 1.0))))
        for axis in observed
    ]))


def evaluate_predictions(
    predictions_path: Path,
    data_root: Path,
    output_dir: Path,
    split_path: Path | None = None,
) -> dict:
    """Load GT only after inference has produced an immutable prediction file."""
    frozen = json.loads(Path(predictions_path).read_text(encoding="utf-8"))
    split_lookup: dict[str, str] = {}
    if split_path is not None:
        split_payload = json.loads(Path(split_path).read_text(encoding="utf-8"))
        for split_name, key in (("train", "train_ids"), ("heldout", "heldout_ids")):
            for item in split_payload.get(key) or []:
                split_lookup[f"{item['dataset_id']}::{item['obj_id']}"] = split_name
    rows = []
    for pred in frozen["predictions"]:
        gt_path = Path(data_root) / pred["dataset"] / "reconstruction" / "part_info" / pred["object_id"] / "part_info.json"
        gt_data = json.loads(gt_path.read_text(encoding="utf-8"))
        gt_key, gt = _find_gt_part(gt_data["parts"], pred["label"])
        gt_type = str(gt["joint"]).lower()
        if gt_type not in {"prismatic", "revolute"}:
            raise ValueError(f"unsupported benchmark GT joint {gt_type!r}: {gt_path}:{gt_key}")
        params = [float(value) for value in gt["joint_params"]]
        candidate = pred["candidate"]
        metric_candidate = prediction_in_annotation_frame(candidate, str(pred["dataset"]))
        type_correct = candidate["joint_type"] == gt_type
        axis_error = axis_angular_error_deg(metric_candidate["axis_world"], params[:3])
        if type_correct:
            lower_error, upper_error, endpoint_error = aligned_range_endpoint_errors(metric_candidate, params)
        else:
            lower_error = upper_error = endpoint_error = None
        body_min, body_max = _obj_bounds(str(pred["body_mesh"]))
        moving_min, moving_max = _obj_bounds(str(pred["moving_mesh"]))
        object_diagonal = float(np.linalg.norm(
            np.maximum(body_max, moving_max) - np.minimum(body_min, moving_min)
        ))
        infinite_line_distance = (
            origin_line_distance(
                metric_candidate["origin_world"], metric_candidate["axis_world"], params[3:6], params[:3]
            )
            if gt_type == "revolute" and type_correct else None
        )
        origin_offset = (
            origin_gt_axis_perpendicular_offset(metric_candidate["origin_world"], params[3:6], params[:3])
            if gt_type == "revolute" and type_correct and axis_error <= 15.0 else None
        )
        row = {
            **{key: pred.get(key) for key in ("sample_id", "dataset", "object_id", "angle", "category", "label")},
            "gt_key": gt_key,
            "official_split": split_lookup.get(f"{pred['dataset']}::{pred['object_id']}", "unmapped"),
            "predicted_type": candidate["joint_type"],
            "gt_type": gt_type,
            "type_correct": type_correct,
            "predicted_axis_decoded_frame": candidate["axis_world"],
            "predicted_axis_annotation_frame": metric_candidate["axis_world"],
            "axis_angular_error_deg": axis_error,
            "origin_gt_axis_perpendicular_offset": origin_offset,
            "origin_gt_axis_perpendicular_offset_normalized": (
                origin_offset / object_diagonal if origin_offset is not None and object_diagonal > 1e-12 else None
            ),
            "origin_infinite_line_distance_secondary": infinite_line_distance,
            "object_diagonal": object_diagonal,
            "range_lower_error": lower_error,
            "range_upper_error": upper_error,
            "range_endpoint_error": endpoint_error,
            "range_unit": "m" if gt_type == "prismatic" else "rad",
            "iterations": int(pred["iterations"]),
            "runtime_seconds": float(pred["runtime_seconds"]),
            "score": float(candidate["score"]),
            "annotation_frame_axis_error_deg": annotation_frame_axis_error_deg(
                Path(data_root), pred["dataset"], pred["object_id"], gt
            ),
        }
        rows.append(row)
    summary = _summarize(rows)
    summary["refinement_runtime_seconds"] = float(frozen.get("runtime_seconds", 0.0))
    base_prediction_path = frozen.get("base_predictions")
    if base_prediction_path and Path(str(base_prediction_path)).is_file():
        base_payload = json.loads(Path(str(base_prediction_path)).read_text(encoding="utf-8"))
        summary["base_inference_runtime_seconds"] = float(base_payload.get("runtime_seconds", 0.0))
        summary["total_inference_runtime_seconds"] = (
            summary["base_inference_runtime_seconds"] + summary["refinement_runtime_seconds"]
        )
    payload = {
        "format": "arts_gen_kin_agent_benchmark_metrics_v1",
        "prediction_file": str(Path(predictions_path).resolve()),
        "gt_contract": "GT loaded in evaluator only, after predictions were frozen",
        "metric_frame": (
            "legacy baked RealAppliance benchmark vertices are first normalized to the current unbaked "
            "delivery frame, then decoded predictions are mapped into part_info coordinates with the root "
            "Rx(+90deg); PhysX-0511 uses its fixed decoded-to-annotation transform and PhyX is identity; "
            "annotation axes are independently cross-checked against joint_transforms"
        ),
        "summary": summary,
        "samples": rows,
    }
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "metrics.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    _write_csv(output_dir / "metrics.csv", rows)
    (output_dir / "summary.md").write_text(_summary_markdown(summary, rows), encoding="utf-8")
    _write_summary_plot(output_dir / "benchmark_summary.png", summary, rows)
    return payload


def _summarize(rows: list[dict]) -> dict:
    def mean(key: str, subset: list[dict]) -> float | None:
        values = [float(row[key]) for row in subset if row.get(key) is not None]
        return float(np.mean(values)) if values else None

    def median(key: str, subset: list[dict]) -> float | None:
        values = [float(row[key]) for row in subset if row.get(key) is not None]
        return float(np.median(values)) if values else None

    result = {
        "samples": len(rows),
        "type_accuracy": float(np.mean([row["type_correct"] for row in rows])),
        "axis_angular_error_deg_mean": mean("axis_angular_error_deg", rows),
        "axis_angular_error_deg_median": float(np.median([row["axis_angular_error_deg"] for row in rows])),
        "type_correct_axis_error_deg_mean": mean("axis_angular_error_deg", [row for row in rows if row["type_correct"]]),
        "type_correct_axis_error_deg_median": float(np.median([
            row["axis_angular_error_deg"] for row in rows if row["type_correct"]
        ])),
        "revolute_origin_gt_axis_perpendicular_offset_mean": mean("origin_gt_axis_perpendicular_offset", rows),
        "revolute_origin_gt_axis_perpendicular_offset_normalized_mean": mean(
            "origin_gt_axis_perpendicular_offset_normalized", rows
        ),
        "revolute_origin_audited_samples": sum(
            row.get("origin_gt_axis_perpendicular_offset") is not None for row in rows
        ),
        "iterations_mean": mean("iterations", rows),
        "iterations_max": max(row["iterations"] for row in rows),
        "runtime_seconds_mean": mean("runtime_seconds", rows),
        "runtime_seconds_total": sum(row["runtime_seconds"] for row in rows),
        "annotation_frame_axis_error_deg_median": median("annotation_frame_axis_error_deg", rows),
        "annotation_frame_axis_error_deg_mean": mean("annotation_frame_axis_error_deg", rows),
        "annotation_frame_axis_error_deg_max": max(
            (row["annotation_frame_axis_error_deg"] for row in rows if row.get("annotation_frame_axis_error_deg") is not None),
            default=None,
        ),
        "annotation_frame_axis_audited_samples": sum(row.get("annotation_frame_axis_error_deg") is not None for row in rows),
    }
    for kind in ("prismatic", "revolute"):
        subset = [row for row in rows if row["gt_type"] == kind]
        correct_subset = [row for row in subset if row["type_correct"]]
        result[f"{kind}_samples"] = len(subset)
        result[f"{kind}_type_correct_samples"] = len(correct_subset)
        result[f"{kind}_range_endpoint_error_mean"] = mean("range_endpoint_error", correct_subset)
    for dataset in sorted({row["dataset"] for row in rows}):
        subset = [row for row in rows if row["dataset"] == dataset]
        result[f"{dataset}_type_accuracy"] = float(np.mean([row["type_correct"] for row in subset]))
        result[f"{dataset}_axis_error_median_deg"] = float(np.median([row["axis_angular_error_deg"] for row in subset]))
    result["by_category"] = {}
    for category in sorted({str(row["category"]) for row in rows}):
        subset = [row for row in rows if str(row["category"]) == category]
        result["by_category"][category] = {
            "samples": len(subset),
            "unique_objects": len({(row.get("dataset"), row.get("object_id")) for row in subset}),
            "type_accuracy": float(np.mean([row["type_correct"] for row in subset])),
            "axis_error_mean_deg": mean("axis_angular_error_deg", subset),
            "axis_error_median_deg": float(np.median([row["axis_angular_error_deg"] for row in subset])),
            "axis_error_p90_deg": float(np.quantile([row["axis_angular_error_deg"] for row in subset], 0.9)),
            "axis_outliers_over_30_deg": sum(row["axis_angular_error_deg"] > 30.0 for row in subset),
            "axis_outliers_over_60_deg": sum(row["axis_angular_error_deg"] > 60.0 for row in subset),
            "type_correct_axis_error_median_deg": (
                float(np.median([row["axis_angular_error_deg"] for row in subset if row["type_correct"]]))
                if any(row["type_correct"] for row in subset) else None
            ),
            "range_endpoint_error_mean": mean("range_endpoint_error", subset),
            "origin_gt_axis_perpendicular_offset_normalized_mean": mean(
                "origin_gt_axis_perpendicular_offset_normalized", subset
            ),
        }
    result["by_dataset_category"] = {}
    for dataset in sorted({row["dataset"] for row in rows}):
        result["by_dataset_category"][dataset] = {}
        for category in sorted({str(row["category"]) for row in rows if row["dataset"] == dataset}):
            subset = [
                row for row in rows
                if row["dataset"] == dataset and str(row["category"]) == category
            ]
            result["by_dataset_category"][dataset][category] = {
                "samples": len(subset),
                "unique_objects": len({row.get("object_id") for row in subset}),
                "type_accuracy": float(np.mean([row["type_correct"] for row in subset])),
                "axis_error_median_deg": median("axis_angular_error_deg", subset),
                "axis_error_p90_deg": float(np.quantile([row["axis_angular_error_deg"] for row in subset], 0.9)),
                "axis_outliers_over_30_deg": sum(row["axis_angular_error_deg"] > 30.0 for row in subset),
            }
    result["type_mismatches"] = [
        {
            "sample_id": row.get("sample_id"),
            "dataset": row.get("dataset"),
            "category": row.get("category"),
            "label": row.get("label"),
            "gt_type": row.get("gt_type"),
            "predicted_type": row.get("predicted_type"),
        }
        for row in rows if not row["type_correct"]
    ]
    result["axis_outliers"] = [
        {
            "sample_id": row.get("sample_id"),
            "category": row.get("category"),
            "gt_type": row.get("gt_type"),
            "predicted_type": row.get("predicted_type"),
            "axis_error_deg": row["axis_angular_error_deg"],
        }
        for row in sorted(rows, key=lambda item: item["axis_angular_error_deg"], reverse=True)[:20]
    ]
    if any(row.get("official_split") for row in rows):
        result["by_official_split"] = {}
        for split_name in sorted({str(row.get("official_split")) for row in rows}):
            subset = [row for row in rows if str(row.get("official_split")) == split_name]
            result["by_official_split"][split_name] = {
                "samples": len(subset),
                "unique_objects": len({(row.get("dataset"), row.get("object_id")) for row in subset}),
                "type_accuracy": float(np.mean([row["type_correct"] for row in subset])),
                "axis_error_mean_deg": mean("axis_angular_error_deg", subset),
                "axis_error_median_deg": median("axis_angular_error_deg", subset),
                "axis_outliers_over_30_deg": sum(row["axis_angular_error_deg"] > 30.0 for row in subset),
                "range_endpoint_error_mean": mean("range_endpoint_error", subset),
            }
    return result


def _write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_summary_plot(path: Path, summary: dict, rows: list[dict]) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    categories = list(summary["by_category"])
    means = [summary["by_category"][key]["axis_error_mean_deg"] for key in categories]
    medians = [summary["by_category"][key]["axis_error_median_deg"] for key in categories]
    p90 = [summary["by_category"][key]["axis_error_p90_deg"] for key in categories]
    ranges = [summary["by_category"][key]["range_endpoint_error_mean"] or 0.0 for key in categories]
    figure, axes = plt.subplots(1, 3, figsize=(14, 4.4), constrained_layout=True)
    x = np.arange(len(categories))
    axes[0].bar(x - 0.25, means, width=0.25, label="mean", color="#146c94")
    axes[0].bar(x, medians, width=0.25, label="median", color="#2a7f62")
    axes[0].bar(x + 0.25, p90, width=0.25, label="p90", color="#b45f06")
    axes[0].set_title("Axis angular error")
    axes[0].set_ylabel("degrees")
    axes[0].set_xticks(x, categories)
    axes[0].legend(frameon=False)
    axes[0].grid(axis="y", alpha=0.25)
    axes[1].bar(x, ranges, color="#146c94")
    axes[1].set_title("Range endpoint error")
    axes[1].set_xticks(x, categories)
    axes[1].grid(axis="y", alpha=0.25)
    split_metrics = summary.get("by_official_split") or {}
    split_names = list(split_metrics)
    split_means = [split_metrics[key]["axis_error_mean_deg"] for key in split_names]
    split_outliers = [
        100.0 * split_metrics[key]["axis_outliers_over_30_deg"] / max(split_metrics[key]["samples"], 1)
        for key in split_names
    ]
    sx = np.arange(len(split_names))
    axes[2].bar(sx - 0.18, split_means, width=0.36, label="axis mean", color="#6f4e7c")
    axes[2].bar(sx + 0.18, split_outliers, width=0.36, label=">30 deg rate (%)", color="#c44e52")
    axes[2].set_title("Official object split")
    axes[2].set_xticks(sx, split_names)
    axes[2].legend(frameon=False)
    axes[2].grid(axis="y", alpha=0.25)
    figure.suptitle(
        f"Kin Agent: {len(rows)} parts | type {summary['type_accuracy']:.1%} | "
        f"axis mean {summary['axis_angular_error_deg_mean']:.2f} deg",
        fontsize=12,
    )
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _summary_markdown(summary: dict, rows: list[dict]) -> str:
    lines = [
        "# Kin Agent decoded-mesh benchmark", "",
        f"- Samples: {summary['samples']}",
        f"- Type accuracy: {summary['type_accuracy']:.3f}",
        f"- Axis error median / mean: {summary['axis_angular_error_deg_median']:.2f} / {summary['axis_angular_error_deg_mean']:.2f} deg",
        f"- Type-correct axis error median / mean: {summary['type_correct_axis_error_deg_median']:.2f} / {summary['type_correct_axis_error_deg_mean']:.2f} deg",
        f"- Revolute origin GT-axis perpendicular offset mean: {summary['revolute_origin_gt_axis_perpendicular_offset_mean']:.4f}",
        f"- Revolute normalized origin offset mean: {summary['revolute_origin_gt_axis_perpendicular_offset_normalized_mean']:.4f}",
        f"- Revolute origin samples (type correct, axis <= 15 deg): {summary['revolute_origin_audited_samples']}",
        f"- Prismatic range endpoint error mean: {summary['prismatic_range_endpoint_error_mean']:.4f} normalized asset units",
        f"- Revolute range endpoint error mean: {summary['revolute_range_endpoint_error_mean']:.4f} rad",
        f"- Iterations mean / max: {summary['iterations_mean']:.2f} / {summary['iterations_max']}",
        f"- Runtime total: {summary['runtime_seconds_total']:.2f} s", "",
        f"- Annotation frame audit samples: {summary['annotation_frame_axis_audited_samples']}",
        f"- Annotation frame axis error median / max: {summary['annotation_frame_axis_error_deg_median']:.8f} / "
        f"{summary['annotation_frame_axis_error_deg_max']:.8f} deg", "",
        "## Category metrics", "",
        "| category | samples / objects | type acc | axis median / p90 | >30 / >60 | range mean | origin norm mean |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for category, metrics in summary["by_category"].items():
        range_mean = "-" if metrics["range_endpoint_error_mean"] is None else f"{metrics['range_endpoint_error_mean']:.4f}"
        origin_mean = (
            "-" if metrics["origin_gt_axis_perpendicular_offset_normalized_mean"] is None
            else f"{metrics['origin_gt_axis_perpendicular_offset_normalized_mean']:.4f}"
        )
        lines.append(
            f"| {category} | {metrics['samples']} / {metrics['unique_objects']} | {metrics['type_accuracy']:.3f} | "
            f"{metrics['axis_error_median_deg']:.2f} / {metrics['axis_error_p90_deg']:.2f} | "
            f"{metrics['axis_outliers_over_30_deg']} / {metrics['axis_outliers_over_60_deg']} | "
            f"{range_mean} | {origin_mean} |"
        )
    lines.extend([
        "", "## Per-sample metrics", "",
        "| sample | GT / pred | axis deg | origin | range | iter | sec |", "|---|---|---:|---:|---:|---:|---:|",
    ])
    for row in rows:
        origin = (
            "-" if row["origin_gt_axis_perpendicular_offset"] is None
            else f"{row['origin_gt_axis_perpendicular_offset']:.4f}"
        )
        range_value = "-" if row["range_endpoint_error"] is None else f"{row['range_endpoint_error']:.4f} {row['range_unit']}"
        lines.append(
            f"| {row['sample_id']} | {row['gt_type']} / {row['predicted_type']} | "
            f"{row['axis_angular_error_deg']:.2f} | {origin} | {range_value} | "
            f"{row['iterations']} | {row['runtime_seconds']:.2f} |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    build_parser = subparsers.add_parser("build-manifest", help="discover expanded exact-match benchmark cohort")
    build_parser.add_argument("--decoded-root", action="append", required=True, metavar="DATASET=PATH")
    build_parser.add_argument("--data-root", type=Path, required=True)
    build_parser.add_argument("--output", type=Path, required=True)
    blind_parser = subparsers.add_parser(
        "build-blind-manifest", help="freeze a decoded-filename cohort without opening annotations"
    )
    blind_parser.add_argument("--decoded-root", action="append", required=True, metavar="DATASET=PATH")
    blind_parser.add_argument("--exclude-manifest", type=Path)
    blind_parser.add_argument("--output", type=Path, required=True)
    infer_parser = subparsers.add_parser("infer", help="GT-free decoded-mesh inference")
    infer_parser.add_argument("--manifest", type=Path, required=True)
    infer_parser.add_argument("--output", type=Path, required=True)
    infer_parser.add_argument("--max-iterations", type=int, default=7)
    refine_parser = subparsers.add_parser(
        "refine-observations", help="refine frozen predictions with calibrated multi-state observations"
    )
    refine_parser.add_argument("--predictions", type=Path, required=True)
    refine_parser.add_argument("--observation-root", action="append", default=[], metavar="DATASET=PATH")
    refine_parser.add_argument("--static-observation-root", action="append", default=[], metavar="DATASET=PATH")
    refine_parser.add_argument("--static-view-indices", action="append", default=[], metavar="DATASET=0,3,8,11")
    refine_parser.add_argument("--output", type=Path, required=True)
    eval_parser = subparsers.add_parser("evaluate", help="benchmark-only GT metrics")
    eval_parser.add_argument("--predictions", type=Path, required=True)
    eval_parser.add_argument("--data-root", type=Path, required=True)
    eval_parser.add_argument("--output-dir", type=Path, required=True)
    eval_parser.add_argument("--split", type=Path)
    args = parser.parse_args()
    if args.command in {"build-manifest", "build-blind-manifest"}:
        roots = {}
        for raw in args.decoded_root:
            dataset, separator, value = raw.partition("=")
            if not separator or not dataset or not value:
                parser.error(f"invalid --decoded-root {raw!r}; expected DATASET=PATH")
            roots[dataset] = Path(value)
        if args.command == "build-manifest":
            result = build_expanded_manifest(roots, args.data_root, args.output)
        else:
            excluded = set()
            if args.exclude_manifest:
                previous = json.loads(args.exclude_manifest.read_text(encoding="utf-8"))
                excluded = {
                    f"{row['dataset']}::{row['object_id']}" for row in previous.get("samples") or []
                }
            result = build_blind_manifest(roots, args.output, excluded_object_keys=excluded)
        display = {"samples": len(result["samples"]), "audit": result["audit"]}
        print(json.dumps(display, indent=2, ensure_ascii=False))
        return
    if args.command == "infer":
        result = infer_manifest(args.manifest, args.output, max_iterations=args.max_iterations)
    elif args.command == "refine-observations":
        roots = {}
        for raw in args.observation_root:
            dataset, separator, value = raw.partition("=")
            if not separator or not dataset or not value:
                parser.error(f"invalid --observation-root {raw!r}; expected DATASET=PATH")
            roots[dataset] = Path(value)
        static_roots = {}
        for raw in args.static_observation_root:
            dataset, separator, value = raw.partition("=")
            if not separator or not dataset or not value:
                parser.error(f"invalid --static-observation-root {raw!r}; expected DATASET=PATH")
            static_roots[dataset] = Path(value)
        view_indices = {}
        for raw in args.static_view_indices:
            dataset, separator, value = raw.partition("=")
            if not separator or not dataset or not value:
                parser.error(f"invalid --static-view-indices {raw!r}; expected DATASET=0,3,8,11")
            view_indices[dataset] = tuple(
                int(item) for item in value.split(",") if item.strip()
            )
        result = refine_predictions_with_observations(
            args.predictions, args.output, roots, static_roots, view_indices,
        )
    else:
        result = evaluate_predictions(args.predictions, args.data_root, args.output_dir, args.split)
    display = result["summary"] if "summary" in result else {"predictions": len(result["predictions"])}
    print(json.dumps(display, indent=2))


if __name__ == "__main__":
    main()
