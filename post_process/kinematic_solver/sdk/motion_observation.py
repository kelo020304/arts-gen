"""Kinematic evidence from calibrated, multi-state 2D observations.

This module intentionally accepts only rendered 2D boxes and camera poses.  It
never opens joint annotations, source USD files, or reconstructed GT meshes.
The boxes define a coarse visual hull for each observed articulation state;
the resulting 3D centroid trajectory supplies an independent axis/origin
critic for the decoded-mesh solver.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import re
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class MotionObservationEstimate:
    joint_type: str
    axis_world: tuple[float, float, float]
    origin_world: tuple[float, float, float]
    observed_span: float
    confidence: float
    state_count: int
    trajectory_points: tuple[tuple[float, float, float], ...]
    diagnostics: dict[str, float]
    input_files: tuple[str, ...]

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class StaticPartObservationEstimate:
    center_world: tuple[float, float, float]
    support: float
    view_count: int
    state_index: int
    trajectory_points: tuple[tuple[float, float, float], ...]
    diagnostics: dict[str, float]
    input_files: tuple[str, ...]

    def to_dict(self) -> dict:
        return asdict(self)


def estimate_static_part_observation(
    render_root: Path,
    label: str,
    *,
    state_index: int = 0,
) -> StaticPartObservationEstimate | None:
    """Triangulate one part center from one articulated state and multiple views."""
    render_root = Path(render_root)
    _assert_safe_observation_root(render_root)
    state_dir = render_root / f"angle_{int(state_index)}"
    if not state_dir.is_dir():
        state_dirs = sorted(
            (path for path in render_root.glob("angle_*") if path.is_dir()),
            key=_state_sort_key,
        )
        if not state_dirs:
            return None
        state_dir = state_dirs[0]
        match = re.search(r"(\d+)$", state_dir.name)
        state_index = int(match.group(1)) if match else 0
    camera_path = state_dir / "camera_transforms.json"
    boxes_path = state_dir / "bbox_gt.json"
    if not camera_path.is_file() or not boxes_path.is_file():
        return None
    cameras = json.loads(camera_path.read_text(encoding="utf-8"))
    boxes = json.loads(boxes_path.read_text(encoding="utf-8"))
    part_key = _match_part_key(boxes.get("parts") or {}, label)
    if part_key is None:
        return None
    views = boxes["parts"][part_key].get("views") or {}
    center, support = _bbox_cone_center(cameras, views)
    if center is None:
        return None
    center = _to_source_frame(center, cameras, render_root)
    view_count = sum(
        1
        for frame in cameras.get("frames") or []
        if (views.get(str(frame.get("view_index"))) or {}).get("bbox")
    )
    center_tuple = tuple(float(value) for value in center)
    return StaticPartObservationEstimate(
        center_world=center_tuple,
        support=float(support),
        view_count=int(view_count),
        state_index=int(state_index),
        trajectory_points=(center_tuple,),
        diagnostics={
            "static_visual_hull_support": float(support),
            "static_view_count": float(view_count),
            "static_state_index": float(state_index),
        },
        input_files=(str(camera_path.resolve()), str(boxes_path.resolve())),
    )


def estimate_motion_from_render_states(
    render_root: Path,
    label: str,
    joint_type: str,
    *,
    grid_resolution: int = 48,
    top_views: int = 5,
) -> MotionObservationEstimate | None:
    """Estimate motion from calibrated boxes across distinct object states."""
    if joint_type not in {"prismatic", "revolute"}:
        raise ValueError("joint_type must be prismatic or revolute")
    return estimate_motion_hypotheses_from_render_states(
        render_root,
        label,
        grid_resolution=grid_resolution,
        top_views=top_views,
    ).get(joint_type)


def estimate_motion_hypotheses_from_render_states(
    render_root: Path,
    label: str,
    *,
    grid_resolution: int = 48,
    top_views: int = 5,
) -> dict[str, MotionObservationEstimate]:
    """Fit line and circle hypotheses to one shared set of render observations."""
    render_root = Path(render_root)
    _assert_safe_observation_root(render_root)
    state_dirs = sorted(
        (path for path in render_root.glob("angle_*") if path.is_dir()),
        key=_state_sort_key,
    )
    if len(state_dirs) < 3:
        return {}
    grid_resolution = int(np.clip(grid_resolution, 24, 96))
    top_views = int(np.clip(top_views, 3, 8))

    observations: list[tuple[np.ndarray, float]] = []
    input_files: list[str] = []
    for state_dir in state_dirs:
        camera_path = state_dir / "camera_transforms.json"
        boxes_path = state_dir / "bbox_gt.json"
        if not camera_path.is_file() or not boxes_path.is_file():
            continue
        cameras = json.loads(camera_path.read_text(encoding="utf-8"))
        boxes = json.loads(boxes_path.read_text(encoding="utf-8"))
        part_key = _match_part_key(boxes.get("parts") or {}, label)
        if part_key is None:
            continue
        centroid, support = _bbox_cone_center(
            cameras,
            boxes["parts"][part_key].get("views") or {},
        )
        if centroid is None:
            continue
        centroid = _to_source_frame(centroid, cameras, render_root)
        observations.append((centroid, support))
        input_files.extend((str(camera_path.resolve()), str(boxes_path.resolve())))
    if len(observations) < 3:
        return {}

    points = np.asarray([row[0] for row in observations], dtype=np.float64)
    supports = np.asarray([row[1] for row in observations], dtype=np.float64)
    result = {}
    for joint_type in ("prismatic", "revolute"):
        estimate = fit_motion_trajectory(points, joint_type)
        if estimate is None:
            continue
        diagnostics = {
            **estimate[4],
            "visual_hull_support_mean": float(np.mean(supports)),
            "visual_hull_support_min": float(np.min(supports)),
            "grid_resolution": float(grid_resolution),
            "top_views": float(top_views),
        }
        result[joint_type] = MotionObservationEstimate(
            joint_type=joint_type,
            axis_world=tuple(float(value) for value in estimate[0]),
            origin_world=tuple(float(value) for value in estimate[1]),
            observed_span=float(estimate[2]),
            confidence=float(estimate[3]),
            state_count=len(points),
            trajectory_points=tuple(tuple(float(value) for value in point) for point in points),
            diagnostics=diagnostics,
            input_files=tuple(dict.fromkeys(input_files)),
        )
    return result


def fit_motion_trajectory(
    points: Iterable[Iterable[float]],
    joint_type: str,
) -> tuple[np.ndarray, np.ndarray, float, float, dict[str, float]] | None:
    """Fit a line or circle trajectory without consulting joint parameters."""
    values = np.asarray(list(points), dtype=np.float64).reshape((-1, 3))
    values = values[np.isfinite(values).all(axis=1)]
    if len(values) < 3:
        return None
    centered = values - values.mean(axis=0)
    _, singular, vh = np.linalg.svd(centered, full_matrices=False)
    scale = max(float(singular[0]), 1e-12)
    secondary_ratio = float(singular[1] / scale)
    tertiary_ratio = float(singular[2] / max(singular[1], 1e-12))

    if joint_type == "prismatic":
        axis = _cardinal_axis(vh[0])
        projected = values @ axis
        relative_q = projected - projected[0]
        observed_lower = float(np.min(relative_q))
        observed_upper = float(np.max(relative_q))
        span = observed_upper - observed_lower
        linearity = 1.0 - min(1.0, float(singular[1] / scale))
        spread = min(1.0, span / 0.08)
        observable = float(singular[1] / scale) < 0.35
        # Translation direction remains identifiable from the dominant state
        # displacement even when bbox-center noise inflates the second mode.
        confidence = 0.90 if observable else 0.60
        diagnostics = {
            "trajectory_linearity": linearity,
            "trajectory_planarity": 0.0,
            "trajectory_spread": spread,
            "trajectory_singular_0": float(singular[0]),
            "trajectory_singular_1": float(singular[1]),
            "trajectory_singular_2": float(singular[2]),
            "trajectory_secondary_ratio": secondary_ratio,
            "trajectory_tertiary_ratio": tertiary_ratio,
            "trajectory_observable": 1.0 if observable else 0.0,
            "observed_lower": observed_lower,
            "observed_upper": observed_upper,
        }
        return axis, values.mean(axis=0), span, confidence, diagnostics

    if joint_type != "revolute":
        raise ValueError("joint_type must be prismatic or revolute")
    raw_axis = _canonicalize_axis(vh[-1], snap=False)
    axis = raw_axis
    basis_x = vh[0]
    basis_y = np.cross(raw_axis, basis_x)
    basis_y /= max(float(np.linalg.norm(basis_y)), 1e-12)
    coordinates = np.column_stack((centered @ basis_x, centered @ basis_y))
    design = np.column_stack((coordinates[:, 0], coordinates[:, 1], np.ones(len(coordinates))))
    target = -(coordinates[:, 0] ** 2 + coordinates[:, 1] ** 2)
    coefficients, *_ = np.linalg.lstsq(design, target, rcond=None)
    circle_center_2d = -0.5 * coefficients[:2]
    radius_squared = float(circle_center_2d @ circle_center_2d - coefficients[2])
    radius = math.sqrt(max(radius_squared, 0.0))
    origin = values.mean(axis=0) + basis_x * circle_center_2d[0] + basis_y * circle_center_2d[1]
    relative = coordinates - circle_center_2d
    angles = np.mod(np.arctan2(relative[:, 1], relative[:, 0]), 2.0 * math.pi)
    relative_angles = np.angle(np.exp(1j * (angles - angles[0])))
    observed_lower = float(np.min(relative_angles))
    observed_upper = float(np.max(relative_angles))
    span = observed_upper - observed_lower
    radial_residual = float(np.mean(np.abs(np.linalg.norm(relative, axis=1) - radius)))
    planarity_ratio = float(singular[2] / max(singular[1], 1e-12))
    planarity = 1.0 - min(1.0, planarity_ratio)
    curvature = min(1.0, float(singular[1] / scale) * 5.0)
    circle_fit = math.exp(-radial_residual / max(radius * 0.25, 1e-4))
    spread = min(1.0, span / 0.35)
    displacement = float(np.linalg.norm(np.ptp(values, axis=0)))
    strong = radius > 5e-3 and displacement > 1e-2 and singular[1] > 2e-3 and planarity_ratio <= 0.20
    moderate = radius > 5e-3 and displacement > 1e-2 and singular[1] > 2e-3 and planarity_ratio <= 0.30
    if strong:
        axis = _cardinal_axis(raw_axis)
        confidence = 0.90
    elif moderate:
        confidence = 0.40
    else:
        confidence = 0.15
    diagnostics = {
        "trajectory_linearity": 0.0,
        "trajectory_planarity": planarity,
        "trajectory_planarity_ratio": planarity_ratio,
        "trajectory_curvature": curvature,
        "trajectory_circle_fit": circle_fit,
        "trajectory_circle_radius": radius,
        "trajectory_circle_residual": radial_residual,
        "trajectory_spread": spread,
        "trajectory_singular_0": float(singular[0]),
        "trajectory_singular_1": float(singular[1]),
        "trajectory_singular_2": float(singular[2]),
        "trajectory_secondary_ratio": secondary_ratio,
        "trajectory_tertiary_ratio": tertiary_ratio,
        "trajectory_observable": 1.0 if strong else 0.5 if moderate else 0.0,
        "observed_lower": observed_lower,
        "observed_upper": observed_upper,
    }
    return axis, origin, span, confidence, diagnostics


def _bbox_cone_center(
    cameras: dict,
    views: dict,
) -> tuple[np.ndarray | None, float]:
    rays = []
    resolution = int(cameras.get("resolution") or 512)
    focal = 0.5 * resolution / math.tan(math.radians(float(cameras.get("fov_deg") or 40.0)) / 2.0)
    for frame in cameras.get("frames") or []:
        raw_box = (views.get(str(frame.get("view_index"))) or {}).get("bbox")
        if not raw_box or len(raw_box) != 4:
            continue
        box = np.asarray(raw_box, dtype=np.float64)
        if not np.isfinite(box).all() or box[2] <= box[0] or box[3] <= box[1]:
            continue
        u = 0.5 * (box[0] + box[2]) * (resolution - 1) / 1000.0
        v = 0.5 * (box[1] + box[3]) * (resolution - 1) / 1000.0
        ray_camera = np.asarray([
            (u - resolution / 2.0) / focal,
            -(v - resolution / 2.0) / focal,
            -1.0,
        ], dtype=np.float64)
        ray_camera /= max(float(np.linalg.norm(ray_camera)), 1e-12)
        transform = np.asarray(frame["transform_matrix"], dtype=np.float64)
        direction = transform[:3, :3] @ ray_camera
        direction /= max(float(np.linalg.norm(direction)), 1e-12)
        rays.append((transform[:3, 3], direction))
    if len(rays) < 3:
        return None, 0.0
    matrix = np.zeros((3, 3), dtype=np.float64)
    target = np.zeros(3, dtype=np.float64)
    for origin, direction in rays:
        projector = np.eye(3) - np.outer(direction, direction)
        matrix += projector
        target += projector @ origin
    center = np.linalg.lstsq(matrix, target, rcond=None)[0]
    residual = float(np.mean([
        np.linalg.norm(np.cross(center - origin, direction)) for origin, direction in rays
    ]))
    return center, max(0.0, 1.0 - residual)


def _to_source_frame(point: np.ndarray, cameras: dict, render_root: Path) -> np.ndarray:
    scale = float(cameras.get("scale") or 1.0)
    offset = np.asarray(cameras.get("offset") or [0.0, 0.0, 0.0], dtype=np.float64)
    result = (np.asarray(point, dtype=np.float64) - offset) / max(scale, 1e-12)
    normalized_root = str(render_root).lower()
    if "phyx-verse" in normalized_root or "realappliance" in normalized_root:
        result = np.asarray([result[0], result[2], -result[1]], dtype=np.float64)
    return result


def _match_part_key(parts: dict, label: str) -> str | None:
    wanted = _normalize_label(label)
    matches = [key for key in parts if _normalize_label(key) == wanted]
    return matches[0] if len(matches) == 1 else None


def _normalize_label(value: str) -> str:
    value = Path(str(value)).stem.lower()
    value = re.sub(r"^part_\d+_", "", value)
    return re.sub(r"[^0-9a-z\u4e00-\u9fff]+", "", value)


def _state_sort_key(path: Path) -> tuple[int, str]:
    match = re.search(r"(\d+)$", path.name)
    return (int(match.group(1)) if match else 10**9, path.name)


def _assert_safe_observation_root(path: Path) -> None:
    normalized = str(Path(path).resolve()).replace("\\", "/").lower()
    forbidden = (
        "/reconstruction/part_info/",
        "/joint_transforms/",
        "/source/model/",
        "/raw/partseg/",
    )
    if any(token in normalized for token in forbidden) or "/renders/" not in normalized:
        raise ValueError(
            "motion observations must come from renders/<object> and may not use joint/GT asset paths"
        )


def _cardinal_axis(axis: Iterable[float]) -> np.ndarray:
    value = _canonicalize_axis(axis, snap=False)
    snapped = np.zeros(3, dtype=np.float64)
    index = int(np.argmax(np.abs(value)))
    snapped[index] = 1.0 if value[index] >= 0.0 else -1.0
    return snapped


def _canonicalize_axis(axis: Iterable[float], *, snap: bool = True) -> np.ndarray:
    value = np.asarray(list(axis), dtype=np.float64)
    norm = float(np.linalg.norm(value))
    if norm <= 1e-12:
        raise ValueError("axis must be non-zero")
    value /= norm
    index = int(np.argmax(np.abs(value)))
    if value[index] < 0.0:
        value = -value
    # Dataset/world axes are overwhelmingly canonical.  The observation is
    # used as a family selector when it lands within 20 degrees of one.
    if snap and abs(float(value[index])) >= math.cos(math.radians(20.0)):
        snapped = np.zeros(3, dtype=np.float64)
        snapped[index] = 1.0
        return snapped
    return value
