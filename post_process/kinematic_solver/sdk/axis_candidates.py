"""Infer signed axis-action candidates from local part geometry."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from .motion_search import AXIS_ACTIONS
from .schemas import EstimateContext
from .coordinate_frame import context_uses_canonical_frame, source_to_canonical_points


@dataclass(frozen=True)
class AxisCandidate:
    axis_label: str
    axis_world: tuple[float, float, float]
    score: float
    reason: str


def with_axis_candidate_evidence(
    ctx: EstimateContext,
    *,
    converter_output_root: Path,
) -> EstimateContext:
    """Return a context whose evidence includes geometry-derived axis candidates."""
    evidence = deepcopy(ctx.evidence)
    for joint_name, joint in ctx.joints.items():
        joint_evidence = evidence.setdefault(joint_name, {})
        candidates = infer_axis_candidates_for_joint(
            ctx.object_id,
            joint_name,
            joint,
            converter_output_root=converter_output_root,
            labels=joint_evidence.get("labels", []),
            transform_source_frame=context_uses_canonical_frame(ctx),
        )
        if not candidates:
            continue
        joint_evidence["axis_candidates"] = [asdict(candidate) for candidate in candidates]
        joint_evidence["recommended_axis_label"] = candidates[0].axis_label
        joint_evidence["recommended_axis_world"] = list(candidates[0].axis_world)
        type_warning = _joint_type_warning(
            joint,
            converter_output_root=converter_output_root,
            object_id=ctx.object_id,
            transform_source_frame=context_uses_canonical_frame(ctx),
        )
        if type_warning:
            joint_evidence["joint_type_warning"] = type_warning
    return EstimateContext(
        object_id=ctx.object_id,
        joints=ctx.joints,
        evidence=evidence,
    )


def infer_axis_candidates_for_joint(
    object_id: str,
    joint_name: str,
    joint: dict,
    *,
    converter_output_root: Path,
    labels: list[str] | tuple[str, ...] | None = None,
    transform_source_frame: bool = False,
) -> list[AxisCandidate]:
    """Infer likely signed axis actions for a joint from its moving-part geometry."""
    moving_parts = list(joint.get("moving_parts") or [joint_name])
    vertices = []
    obj_dir = converter_output_root / f"raw/partseg/{object_id}/objs"
    for part in moving_parts:
        obj_path = obj_dir / f"{part}.obj"
        if not obj_path.is_file():
            continue
        vertices.append(_load_obj_vertices(obj_path, transform_source_frame=transform_source_frame))
    if not vertices:
        return []
    points = np.concatenate(vertices, axis=0)
    if points.shape[0] < 3:
        return []
    body_points = _load_body_vertices(obj_dir, transform_source_frame=transform_source_frame)

    authored_candidates = _authored_axis_candidates(joint)

    if joint.get("type") == "revolute":
        if body_points is not None:
            mount = _revolute_contact_region_axis_candidate(points, body_points)
            if mount is None:
                mount = _revolute_surface_normal_candidate(
                    obj_dir / "body.obj",
                    moving_points=points,
                    transform_source_frame=transform_source_frame,
                )
            if mount is None:
                mount = _revolute_mount_axis_candidate(points, body_points)
            if mount is not None:
                preferred = np.asarray(mount.axis_world, dtype=np.float64)
                reason = mount.reason
                primary_label = mount.axis_label
                primary_score = mount.score
                primary_candidates: list[AxisCandidate] = []
            else:
                return authored_candidates
        else:
            return authored_candidates
    elif joint.get("type") == "prismatic":
        if body_points is not None:
            rest_face = _best_prismatic_rest_face_exit_candidate(points, body_points)
            if rest_face is not None:
                preferred = np.asarray(rest_face.axis_world, dtype=np.float64)
                reason = rest_face.reason
                primary_label = rest_face.axis_label
                primary_score = rest_face.score
                primary_candidates = []
            else:
                return authored_candidates
        else:
            return authored_candidates
        if preferred is None:
            return authored_candidates
    else:
        return []
    pca_candidate = AxisCandidate(
        axis_label=primary_label,
        axis_world=tuple(float(value) for value in preferred),
        score=primary_score,
        reason=reason,
    )
    signed_candidates = _rank_signed_axis_actions(preferred, reason=f"{reason} signed basis")
    return _merge_axis_candidates([pca_candidate], primary_candidates, authored_candidates, signed_candidates)


def _revolute_contact_region_axis_candidate(
    moving_points: np.ndarray,
    body_points: np.ndarray,
) -> AxisCandidate | None:
    """Pick rotation axis from the contact region between moving part and body.

    Contact region = moving-part vertices within eps of nearest body vertex.
    PCA on that subset distinguishes:
    - Line-shaped contact (hinge): largest PC = hinge line direction
    - Plane/ring contact (knob, dial on a panel): smallest PC = panel normal
    """
    if moving_points.shape[0] < 8 or body_points.shape[0] < 3:
        return None
    try:
        from scipy.spatial import cKDTree
    except ImportError:
        return None
    extent_norm = _aabb_extent_norm(moving_points)
    if extent_norm <= 1e-9:
        return None
    eps = max(1e-6, 0.02 * extent_norm)
    tree = cKDTree(body_points)
    distances, _ = tree.query(moving_points, k=1)
    contact_pts = moving_points[distances < eps]
    if contact_pts.shape[0] < 8:
        return None
    centered = contact_pts - contact_pts.mean(axis=0, keepdims=True)
    cov = np.cov(centered.T)
    values, vectors = np.linalg.eigh(cov)
    order = np.argsort(values)[::-1]
    l1 = max(float(values[order[0]]), 0.0)
    l2 = max(float(values[order[1]]), 0.0)
    l3 = max(float(values[order[2]]), 0.0)
    if l1 <= 1e-12:
        return None
    line_ratio = l1 / max(l2, 1e-9 * l1)
    plane_ratio = l2 / max(l3, 1e-9 * l2) if l2 > 1e-12 else 0.0
    if line_ratio >= 3.0:
        axis = vectors[:, int(order[0])]
        shape = "line"
        score = line_ratio
    elif plane_ratio >= 3.0:
        axis = vectors[:, int(order[2])]
        shape = "plane"
        score = plane_ratio
    else:
        return None
    axis = _canonical_axis_sign(axis)
    return AxisCandidate(
        axis_label="contact_pca",
        axis_world=tuple(float(value) for value in axis),
        score=float(score),
        reason=f"revolute contact-region PCA ({shape}-shaped)",
    )


def _revolute_mount_axis_candidate(
    moving_points: np.ndarray,
    body_points: np.ndarray,
) -> AxisCandidate | None:
    moving_center = (moving_points.min(axis=0) + moving_points.max(axis=0)) * 0.5
    body_center = (body_points.min(axis=0) + body_points.max(axis=0)) * 0.5
    half_extents = np.maximum((body_points.max(axis=0) - body_points.min(axis=0)) * 0.5, 1e-9)
    normalized_offset = (moving_center - body_center) / half_extents
    index = int(np.argmax(np.abs(normalized_offset)))
    if abs(float(normalized_offset[index])) <= 0.20:
        return None
    axis = [0.0, 0.0, 0.0]
    axis[index] = 1.0 if normalized_offset[index] >= 0.0 else -1.0
    labels = ("X", "Y", "Z")
    label = ("+" if axis[index] > 0.0 else "-") + labels[index]
    return AxisCandidate(
        axis_label=label,
        axis_world=tuple(axis),
        score=abs(float(normalized_offset[index])),
        reason="revolute Articraft-style mount axis",
    )


def _revolute_surface_normal_candidate(
    body_obj_path: Path,
    *,
    moving_points: np.ndarray,
    transform_source_frame: bool,
) -> AxisCandidate | None:
    mesh = _load_obj_mesh(body_obj_path, transform_source_frame=transform_source_frame)
    if mesh is None:
        return None
    vertices, faces = mesh
    if vertices.shape[0] < 3 or not faces:
        return None
    moving_center = moving_points.mean(axis=0)
    best: tuple[float, np.ndarray] | None = None
    for face in faces:
        tri = vertices[list(face)]
        normal = np.cross(tri[1] - tri[0], tri[2] - tri[0])
        norm = float(np.linalg.norm(normal))
        if norm <= 1e-12:
            continue
        normal = normal / norm
        center = tri.mean(axis=0)
        to_moving = moving_center - center
        distance = float(np.linalg.norm(to_moving))
        if distance <= 1e-12:
            continue
        if float(normal @ to_moving) < 0.0:
            normal = -normal
        alignment = float(normal @ (to_moving / distance))
        if alignment < 0.35:
            continue
        score = alignment / max(distance, 1e-6)
        if best is None or score > best[0]:
            best = (score, normal)
    if best is None:
        return None
    axis = best[1] / max(float(np.linalg.norm(best[1])), 1e-12)
    return AxisCandidate(
        axis_label="surface_normal",
        axis_world=tuple(float(value) for value in axis),
        score=float(best[0]),
        reason="revolute Articraft-style nearby body surface normal",
    )


def _joint_type_warning(
    joint: dict,
    *,
    converter_output_root: Path,
    object_id: str,
    transform_source_frame: bool,
) -> str | None:
    if joint.get("type") != "revolute":
        return None
    moving_parts = list(joint.get("moving_parts") or [])
    if not moving_parts:
        return None
    obj_dir = converter_output_root / f"raw/partseg/{object_id}/objs"
    moving_vertices = []
    for part in moving_parts:
        path = obj_dir / f"{part}.obj"
        if path.is_file():
            moving_vertices.append(_load_obj_vertices(path, transform_source_frame=transform_source_frame))
    body_points = _load_body_vertices(obj_dir, transform_source_frame=transform_source_frame)
    if not moving_vertices or body_points is None:
        return None
    moving_points = np.concatenate(moving_vertices, axis=0)
    rest_face = _best_prismatic_rest_face_exit_candidate(moving_points, body_points)
    if rest_face is None:
        return None
    return (
        "VLM/context marks this joint as revolute, but rest-pose geometry has a "
        f"clear prismatic exit face {rest_face.axis_label} with non-motion overlap. "
        "If this part is a drawer/slider, set type='prismatic' in the initial JSON."
    )


def _load_obj_vertices(obj_path: Path, *, transform_source_frame: bool = False) -> np.ndarray:
    vertices = []
    for line in obj_path.read_text(errors="ignore").splitlines():
        if not line.startswith("v "):
            continue
        parts = line.split()
        if len(parts) >= 4:
            vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
    points = np.asarray(vertices, dtype=np.float64)
    return source_to_canonical_points(points) if transform_source_frame else points


def _load_obj_mesh(
    obj_path: Path,
    *,
    transform_source_frame: bool = False,
) -> tuple[np.ndarray, list[tuple[int, int, int]]] | None:
    if not obj_path.is_file():
        return None
    vertices = []
    faces: list[tuple[int, int, int]] = []
    for line in obj_path.read_text(errors="ignore").splitlines():
        if line.startswith("v "):
            parts = line.split()
            if len(parts) >= 4:
                vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
        elif line.startswith("f "):
            raw = line.split()[1:]
            indices = []
            for token in raw:
                try:
                    indices.append(int(token.split("/")[0]) - 1)
                except ValueError:
                    continue
            if len(indices) >= 3:
                first = indices[0]
                for offset in range(1, len(indices) - 1):
                    faces.append((first, indices[offset], indices[offset + 1]))
    points = np.asarray(vertices, dtype=np.float64)
    if points.size == 0:
        return None
    if transform_source_frame:
        points = source_to_canonical_points(points)
    return points, faces


def _load_body_vertices(obj_dir: Path, *, transform_source_frame: bool = False) -> np.ndarray | None:
    body_path = obj_dir / "body.obj"
    if not body_path.is_file():
        return None
    vertices = _load_obj_vertices(body_path, transform_source_frame=transform_source_frame)
    return vertices if vertices.shape[0] >= 3 else None


def _thin_axis(points: np.ndarray) -> np.ndarray | None:
    centered = points - points.mean(axis=0, keepdims=True)
    cov = np.cov(centered.T)
    values, vectors = np.linalg.eigh(cov)
    order = np.argsort(values)
    thin_value = float(values[order[0]])
    next_value = float(values[order[1]])
    if next_value <= 1e-12 or thin_value / next_value > 0.35:
        return None
    axis = vectors[:, int(order[0])]
    return _canonical_axis_sign(axis)


def _long_axis(points: np.ndarray) -> np.ndarray:
    centered = points - points.mean(axis=0, keepdims=True)
    cov = np.cov(centered.T)
    values, vectors = np.linalg.eigh(cov)
    axis = vectors[:, int(np.argmax(values))]
    return _canonical_axis_sign(axis)


def _pca_axes(points: np.ndarray) -> list[np.ndarray]:
    centered = points - points.mean(axis=0, keepdims=True)
    cov = np.cov(centered.T)
    _values, vectors = np.linalg.eigh(cov)
    return [
        vectors[:, index] / max(float(np.linalg.norm(vectors[:, index])), 1e-12)
        for index in range(3)
    ]


def _best_prismatic_pca_axis_by_overlap_progress(
    moving_points: np.ndarray,
    body_points: np.ndarray,
) -> np.ndarray | None:
    values = np.linalg.eigvalsh(np.cov((moving_points - moving_points.mean(axis=0, keepdims=True)).T))
    finite_values = [float(value) for value in values if np.isfinite(value) and value > 1e-12]
    if len(finite_values) < 3 or max(finite_values) / min(finite_values) < 1.2:
        return None
    base_overlap = _aabb_intersection_volume(moving_points, body_points)
    axes = _pca_axes(moving_points)
    if base_overlap <= 1e-12:
        outward = moving_points.mean(axis=0) - body_points.mean(axis=0)
        best = max(axes, key=lambda axis: abs(float(axis @ outward)))
        return _orient_axis_toward_centroid_outward(best, moving_points, body_points)

    probe = min(0.03, max(0.02, _aabb_extent_norm(moving_points) * 0.15))
    best_axis = axes[0]
    best_overlap = float("inf")
    for axis in axes:
        for sign in (1.0, -1.0):
            signed_axis = axis * sign
            overlap = _aabb_intersection_volume(
                moving_points + signed_axis * probe,
                body_points,
            )
            if overlap < best_overlap:
                best_overlap = overlap
                best_axis = signed_axis
    return best_axis / max(float(np.linalg.norm(best_axis)), 1e-12)


def _best_prismatic_rest_face_exit_candidate(
    moving_points: np.ndarray,
    body_points: np.ndarray,
) -> AxisCandidate | None:
    """Choose the signed face normal that the moving part already exits at rest.

    This follows the same relation-driven idea as Articraft's exact checks: a
    slider's initial pose should reveal which body face it exits, while the two
    non-motion axes should still have projected overlap/containment.
    """
    moving_lo = moving_points.min(axis=0)
    moving_hi = moving_points.max(axis=0)
    body_lo = body_points.min(axis=0)
    body_hi = body_points.max(axis=0)
    moving_extent = np.maximum(moving_hi - moving_lo, 1e-9)
    body_extent = np.maximum(body_hi - body_lo, 1e-9)
    labels = ("X", "Y", "Z")
    best: AxisCandidate | None = None
    for axis_index in range(3):
        for sign in (1.0, -1.0):
            exposure = (
                moving_hi[axis_index] - body_hi[axis_index]
                if sign > 0.0
                else body_lo[axis_index] - moving_lo[axis_index]
            )
            if exposure <= 1e-5:
                continue
            overlap_ratios = []
            within_margins = []
            for other_index in range(3):
                if other_index == axis_index:
                    continue
                overlap = min(moving_hi[other_index], body_hi[other_index]) - max(
                    moving_lo[other_index],
                    body_lo[other_index],
                )
                overlap_ratio = max(0.0, float(overlap)) / min(
                    moving_extent[other_index],
                    body_extent[other_index],
                )
                overlap_ratios.append(overlap_ratio)
                within_margins.append(
                    min(
                        moving_lo[other_index] - body_lo[other_index],
                        body_hi[other_index] - moving_hi[other_index],
                    )
                )
            min_overlap_ratio = min(overlap_ratios) if overlap_ratios else 0.0
            if min_overlap_ratio < 0.45:
                continue
            axis = [0.0, 0.0, 0.0]
            axis[axis_index] = sign
            label = ("+" if sign > 0.0 else "-") + labels[axis_index]
            score = float(exposure) * min_overlap_ratio
            if any(margin >= -0.002 for margin in within_margins):
                score *= 1.25
            candidate = AxisCandidate(
                axis_label=label,
                axis_world=tuple(axis),
                score=score,
                reason="prismatic Articraft-style rest-face exit axis",
            )
            if best is None or candidate.score > best.score:
                best = candidate
    return best


def _authored_unit_axis(joint: dict) -> np.ndarray | None:
    raw_axis = joint.get("axis_world")
    if not raw_axis or len(raw_axis) != 3:
        return None
    axis = np.asarray([float(value) for value in raw_axis], dtype=np.float64)
    norm = float(np.linalg.norm(axis))
    if norm <= 1e-12:
        return None
    return axis / norm


def _orient_axis_toward_centroid_outward(
    axis: np.ndarray,
    moving_points: np.ndarray,
    body_points: np.ndarray,
) -> np.ndarray:
    axis = axis / max(float(np.linalg.norm(axis)), 1e-12)
    outward = moving_points.mean(axis=0) - body_points.mean(axis=0)
    if float(outward @ axis) < 0.0:
        return -axis
    return axis


def _clockwise_axis_from_outside(
    axis: np.ndarray,
    moving_points: np.ndarray,
    body_points: np.ndarray,
) -> np.ndarray:
    outward_axis = _orient_axis_toward_centroid_outward(axis, moving_points, body_points)
    return -outward_axis


def _canonical_axis_sign(axis: np.ndarray) -> np.ndarray:
    axis = axis / max(float(np.linalg.norm(axis)), 1e-12)
    idx = int(np.argmax(np.abs(axis)))
    if axis[idx] < 0.0:
        axis = -axis
    return axis


def _aabb_intersection_volume(left: np.ndarray, right: np.ndarray) -> float:
    left_lo = left.min(axis=0)
    left_hi = left.max(axis=0)
    right_lo = right.min(axis=0)
    right_hi = right.max(axis=0)
    extents = np.maximum(0.0, np.minimum(left_hi, right_hi) - np.maximum(left_lo, right_lo))
    return float(np.prod(extents))


def _aabb_extent_norm(points: np.ndarray) -> float:
    extents = points.max(axis=0) - points.min(axis=0)
    return float(np.linalg.norm(extents))


def _rank_signed_axis_actions(axis: np.ndarray, *, reason: str) -> list[AxisCandidate]:
    ranked = []
    for action in AXIS_ACTIONS:
        action_axis = np.asarray(action.axis_world, dtype=np.float64)
        score = float(axis @ action_axis)
        ranked.append(AxisCandidate(
            axis_label=action.label,
            axis_world=action.axis_world,
            score=score,
            reason=reason,
        ))
    return sorted(ranked, key=lambda candidate: candidate.score, reverse=True)


def _authored_axis_candidates(joint: dict) -> list[AxisCandidate]:
    raw_axis = joint.get("axis_world")
    if not raw_axis or len(raw_axis) != 3:
        return []
    axis = np.asarray([float(value) for value in raw_axis], dtype=np.float64)
    norm = float(np.linalg.norm(axis))
    if norm <= 1e-12:
        return []
    ranked = _rank_signed_axis_actions(axis / norm, reason="authored signed joint axis")
    return ranked[:1]


def _labels_describe_rotary_control(labels: list[str] | tuple[str, ...] | None) -> bool:
    text = " ".join(str(label).lower() for label in (labels or []))
    return any(
        cue in text
        for cue in (
            "knob",
            "dial",
            "temperature",
            "timer",
            "control",
            "rotary",
            "旋钮",
            "表盘",
        )
    )


def _labels_describe_pull_out_drawer(labels: list[str] | tuple[str, ...] | None) -> bool:
    text = " ".join(str(label).lower() for label in (labels or []))
    return any(
        cue in text
        for cue in (
            "pull-out",
            "pull out",
            "drawer",
            "pan",
            "tray",
            "basket",
            "bin",
            "fryer basket",
            "air fryer",
            "抽屉",
            "炸篮",
            "炸桶",
        )
    )


def _merge_axis_candidates(*groups: list[AxisCandidate]) -> list[AxisCandidate]:
    merged: list[AxisCandidate] = []
    seen: set[str] = set()
    for group in groups:
        for candidate in group:
            if candidate.axis_label in seen:
                continue
            seen.add(candidate.axis_label)
            merged.append(candidate)
    return merged
