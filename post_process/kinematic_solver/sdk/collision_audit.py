"""Decoded-mesh collision audit for kinematic bundle delivery.

The audit is deliberately separate from simulator collision proxies.  MuJoCo
convexifies a single concave mesh, so its contact set is not evidence that the
decoded visual surfaces intersect.  Open3D supplies a bounded swept broad
phase and Manifold confirms overlap volume when both inputs are closed solids.
"""

from __future__ import annotations

from dataclasses import dataclass
import gc
import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .kin_agent import KinematicCandidate


AUDIT_VERSION = "decoded_collision_audit_v2"


@dataclass(frozen=True)
class DecodedCollisionAuditConfig:
    max_surface_points: int = 12000
    min_q_samples: int = 9
    max_q_samples: int = 65
    displacement_step_ratio: float = 0.005
    signed_depth_tolerance_ratio: float = 0.001
    broad_excess_fraction: float = 0.002
    exact_excess_fraction: float = 0.002
    exact_absolute_fraction: float = 0.005
    max_narrow_samples: int = 3


def audit_decoded_bundle_collisions(
    body_mesh: Path,
    parts: list[dict[str, Any]],
    *,
    config: DecodedCollisionAuditConfig | None = None,
) -> dict[str, Any]:
    """Audit body sweeps and pairwise moving-part interference.

    ``parts`` must contain ``label``, ``mesh`` and ``candidate``.  All paths
    are decoded delivery assets; the function has no source/GT lookup path.
    """
    cfg = config or DecodedCollisionAuditConfig()
    body = _load_mesh(body_mesh)
    loaded_parts = [
        {
            **part,
            "_mesh": _load_mesh(Path(part["mesh"])),
        }
        for part in parts
    ]
    body_audits = []
    for part in loaded_parts:
        body_audits.append(audit_joint_collision(
            body, part["_mesh"], part["candidate"], config=cfg,
        ))
        # Manifold may retain sizeable temporary buffers until collection.
        # Release them between decoded parts to keep an interactive bundle run
        # bounded on high-resolution SLat meshes.
        gc.collect()
    pairwise = _audit_pairwise(loaded_parts, config=cfg)
    by_label: dict[str, list[dict[str, Any]]] = {
        str(part.get("label") or part.get("body_name")): [] for part in loaded_parts
    }
    for row in pairwise:
        by_label[row["part_a"]].append(row)
        by_label[row["part_b"]].append(row)
    per_joint = []
    for part, audit in zip(loaded_parts, body_audits, strict=True):
        label = str(part.get("label") or part.get("body_name"))
        attached = by_label[label]
        pair_review = any(item["requires_review"] for item in attached)
        per_joint.append({
            **audit,
            "label": label,
            "pairwise_interference": attached,
            "requires_review": bool(audit["requires_review"] or pair_review),
        })
    return {
        "version": AUDIT_VERSION,
        "evidence_scope": "decoded_slat_meshes_only",
        "per_joint": per_joint,
        "pairwise": pairwise,
        "requires_review": any(item["requires_review"] for item in per_joint),
    }


def audit_joint_collision(
    body_mesh: Any,
    moving_mesh: Any,
    candidate: KinematicCandidate,
    *,
    config: DecodedCollisionAuditConfig | None = None,
) -> dict[str, Any]:
    """Audit one decoded moving mesh over the candidate interval."""
    cfg = config or DecodedCollisionAuditConfig()
    body = _load_mesh(body_mesh)
    moving = _load_mesh(moving_mesh)
    mesh_state = _mesh_state(body, moving)
    if not mesh_state["has_triangles"]:
        return _unavailable_report(mesh_state, "decoded mesh has no triangle faces")
    diagonal = _combined_diagonal(body, moving)
    q_values = _adaptive_q_values(candidate, moving, diagonal, cfg)
    try:
        broad = _open3d_sweep(body, moving, candidate, q_values, diagonal, cfg)
    except (ImportError, ModuleNotFoundError) as exc:
        return _unavailable_report(mesh_state, f"Open3D unavailable: {exc}", q_values=q_values)
    except Exception as exc:
        return _unavailable_report(
            mesh_state, f"Open3D broad phase failed: {type(exc).__name__}: {exc}", q_values=q_values,
        )

    exact_capable = bool(mesh_state["body_is_volume"] and mesh_state["moving_is_volume"])
    narrow = _narrow_phase(body, moving, candidate, broad, cfg) if exact_capable else {
        "method": "not_run_non_watertight",
        "exact": False,
        "samples": [],
        "error": "Manifold exact volume requires both decoded meshes to be closed volumes",
    }
    exact_samples = list(narrow.get("samples") or [])
    exact_invalid = [row for row in exact_samples if row["invalid"]]
    broad_invalid = [row for row in broad if row["invalid"]]
    if exact_capable and narrow.get("exact"):
        invalid = bool(exact_invalid)
        status = "collision" if invalid else "clear"
        method = "open3d_adaptive_sweep+manifold_narrow_phase"
        confidence = "high" if not narrow.get("error") else "medium"
        requires_review = bool(invalid or narrow.get("error"))
    else:
        invalid = bool(broad_invalid)
        status = "approximate_collision" if invalid else "approximate_unverified"
        method = "open3d_adaptive_sweep_approximate"
        confidence = "low"
        requires_review = True
    first_invalid = min(
        (row["q"] for row in (exact_invalid if exact_capable and narrow.get("exact") else broad_invalid)),
        default=None,
        key=lambda value: abs(float(value)),
    )
    if invalid and candidate.joint_type == "revolute":
        recommended_actions = ["revise_hinge_origin", "shrink_range_if_no_clear_origin"]
    elif invalid:
        recommended_actions = ["verify_signed_axis", "shrink_range_or_repair_segmentation_boundary"]
    elif requires_review:
        recommended_actions = ["review_non_watertight_collision_geometry"]
    else:
        recommended_actions = []
    return {
        "version": AUDIT_VERSION,
        "status": status,
        "method": method,
        "confidence": confidence,
        "mesh_state": mesh_state,
        "q_sample_count": len(q_values),
        "q_interval": [float(candidate.lower), float(candidate.upper)],
        "q_samples": broad,
        "baseline_inside_fraction": _sample_at_zero(broad).get("inside_fraction", 0.0),
        "max_inside_fraction": max((row["inside_fraction"] for row in broad), default=0.0),
        "first_invalid_q": first_invalid,
        "narrow_phase": narrow,
        "collision_detected": invalid,
        "requires_review": requires_review,
        "recommended_actions": recommended_actions,
    }


def _audit_pairwise(parts: list[dict[str, Any]], *, config: DecodedCollisionAuditConfig) -> list[dict[str, Any]]:
    rows = []
    for index, first in enumerate(parts):
        for second in parts[index + 1 :]:
            rows.append(_audit_part_pair(first, second, config=config))
    return rows


def _audit_part_pair(first: dict[str, Any], second: dict[str, Any], *, config: DecodedCollisionAuditConfig) -> dict[str, Any]:
    import trimesh

    mesh_a, mesh_b = first["_mesh"], second["_mesh"]
    label_a = str(first.get("label") or first.get("body_name"))
    label_b = str(second.get("label") or second.get("body_name"))
    state = _mesh_state(mesh_a, mesh_b)
    if not state["has_triangles"]:
        return {
            "part_a": label_a, "part_b": label_b, "status": "unavailable",
            "method": "none", "samples": [], "collision_detected": False,
            "requires_review": True, "reason": "decoded mesh has no triangle faces",
        }
    states_a = _three_states(first["candidate"])
    states_b = _three_states(second["candidate"])
    exact_capable = bool(state["body_is_volume"] and state["moving_is_volume"])
    sample_count = max(800, min(config.max_surface_points // 3, 4000))
    points_a, _ = trimesh.sample.sample_surface(mesh_a, sample_count, seed=11)
    points_b, _ = trimesh.sample.sample_surface(mesh_b, sample_count, seed=17)
    tolerance = _combined_diagonal(mesh_a, mesh_b) * 0.002
    samples = []
    error = None
    for q_a in states_a:
        transformed_a = _transform_points(points_a, first["candidate"], q_a)
        for q_b in states_b:
            transformed_b = _transform_points(points_b, second["candidate"], q_b)
            try:
                near_fraction, minimum_distance = _surface_proximity(
                    transformed_a, transformed_b, tolerance=tolerance,
                )
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
                near_fraction, minimum_distance = None, None
            samples.append({
                "q_a": float(q_a), "q_b": float(q_b),
                "near_fraction": near_fraction,
                "minimum_distance": minimum_distance,
            })
    baseline = min(samples, key=lambda row: abs(row["q_a"]) + abs(row["q_b"]))
    baseline_fraction = float(baseline["near_fraction"] or 0.0)
    for row in samples:
        row["excess_near_fraction"] = max(0.0, float(row["near_fraction"] or 0.0) - baseline_fraction)
        row["invalid"] = bool(row["near_fraction"] is not None and row["excess_near_fraction"] > 0.01)
    flagged = [row for row in samples if row["invalid"]]
    exact_rows = []
    if exact_capable and flagged and _manifold_available():
        for row in sorted(flagged, key=lambda item: item["excess_near_fraction"], reverse=True)[:2]:
            transformed_a = _transformed_mesh(mesh_a, first["candidate"], row["q_a"])
            transformed_b = _transformed_mesh(mesh_b, second["candidate"], row["q_b"])
            exact = _intersection_fraction(transformed_a, transformed_b)
            exact_rows.append({**row, **exact, "invalid": exact["overlap_fraction"] > config.exact_absolute_fraction})
    collision = any(row["invalid"] for row in exact_rows) if exact_rows else bool(flagged)
    exact = bool(exact_rows)
    approximate = not exact_capable or (bool(flagged) and not exact)
    return {
        "part_a": label_a, "part_b": label_b,
        "status": "collision" if collision and exact else "approximate_collision" if collision else "approximate_unverified" if approximate else "clear_broad_phase",
        "method": "surface_proximity_3x3+manifold" if exact else "surface_proximity_3x3",
        "samples": samples, "exact_samples": exact_rows,
        "baseline_near_fraction": baseline_fraction,
        "collision_detected": collision,
        "requires_review": bool(collision or approximate or error),
        "error": error,
    }


def _open3d_sweep(body, moving, candidate, q_values, diagonal, cfg):
    import open3d as o3d
    import trimesh

    scene = _open3d_scene(body)
    count = min(cfg.max_surface_points, max(1000, len(moving.faces) // 2))
    points, _ = trimesh.sample.sample_surface(moving, count, seed=0)
    tolerance = max(diagonal * cfg.signed_depth_tolerance_ratio, 1e-6)
    raw = []
    for q_value in q_values:
        transformed = _transform_points(points, candidate, float(q_value))
        signed = np.asarray(scene.compute_signed_distance(
            o3d.core.Tensor(transformed.astype(np.float32)), nsamples=5,
        ).numpy(), dtype=np.float64)
        depth = np.maximum(-signed - tolerance, 0.0)
        raw.append({
            "q": float(q_value),
            "inside_fraction": float(np.mean(signed < -tolerance)),
            "max_depth": float(np.max(depth, initial=0.0)),
            "p95_depth": float(np.quantile(depth, 0.95)) if len(depth) else 0.0,
        })
    baseline = _sample_at_zero(raw)["inside_fraction"]
    threshold = baseline + cfg.broad_excess_fraction
    return [
        {**row, "excess_inside_fraction": max(0.0, row["inside_fraction"] - baseline), "invalid": bool(row["inside_fraction"] > threshold)}
        for row in raw
    ]


def _narrow_phase(body, moving, candidate, broad, cfg):
    if not _manifold_available():
        return {"method": "unavailable", "exact": False, "samples": [], "error": "manifold3d unavailable"}
    baseline_row = _sample_at_zero(broad)
    ranked = sorted(broad, key=lambda row: (row["invalid"], row["inside_fraction"]), reverse=True)
    requested = [0.0]
    requested.extend(float(row["q"]) for row in ranked if row["invalid"])
    requested.extend([float(candidate.lower), float(candidate.upper), float(baseline_row["q"])])
    q_values = _ordered_unique(requested)[: cfg.max_narrow_samples]
    samples = []
    error = None
    moving_volume = abs(float(moving.volume))
    for q_value in q_values:
        try:
            transformed = _transformed_mesh(moving, candidate, q_value)
            intersection = _intersection_fraction(body, transformed, denominator=moving_volume)
            samples.append({"q": q_value, **intersection})
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            break
    baseline = min(samples, key=lambda row: abs(row["q"])).get("overlap_fraction", 0.0) if samples else 0.0
    threshold = max(cfg.exact_absolute_fraction, baseline + cfg.exact_excess_fraction)
    samples = [
        {**row, "excess_overlap_fraction": max(0.0, row["overlap_fraction"] - baseline), "invalid": bool(row["overlap_fraction"] > threshold)}
        for row in samples
    ]
    return {
        "method": "trimesh_manifold_intersection_volume",
        "exact": bool(samples) and error is None,
        "baseline_overlap_fraction": baseline,
        "invalid_threshold": threshold,
        "samples": samples,
        "error": error,
    }


def _intersection_fraction(first, second, *, denominator: float | None = None):
    import trimesh

    result = trimesh.boolean.intersection([first, second], engine="manifold")
    overlap_volume = 0.0 if result is None else abs(float(result.volume))
    base = denominator if denominator is not None else min(abs(float(first.volume)), abs(float(second.volume)))
    return {
        "overlap_volume": overlap_volume,
        "overlap_fraction": overlap_volume / max(base, 1e-12),
    }


def _surface_proximity(first: np.ndarray, second: np.ndarray, *, tolerance: float) -> tuple[float, float]:
    from scipy.spatial import cKDTree

    distance_ab, _ = cKDTree(second).query(first, k=1, workers=1)
    distance_ba, _ = cKDTree(first).query(second, k=1, workers=1)
    distances = np.concatenate([distance_ab, distance_ba])
    return float(np.mean(distances < tolerance)), float(np.min(distances, initial=float("inf")))


def _open3d_scene(mesh):
    import open3d as o3d

    legacy = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(np.asarray(mesh.vertices, dtype=np.float64)),
        o3d.utility.Vector3iVector(np.asarray(mesh.faces, dtype=np.int32)),
    )
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(legacy))
    return scene


def _adaptive_q_values(candidate, moving, diagonal, cfg):
    span = abs(float(candidate.upper) - float(candidate.lower))
    target = max(diagonal * cfg.displacement_step_ratio, 1e-5)
    if candidate.joint_type == "prismatic":
        max_displacement = span
    else:
        axis = _unit(candidate.axis_world)
        origin = np.asarray(candidate.origin_world, dtype=np.float64)
        relative = np.asarray(moving.vertices, dtype=np.float64) - origin
        radius = float(np.max(np.linalg.norm(relative - np.outer(relative @ axis, axis), axis=1), initial=0.0))
        max_displacement = radius * span
    count = int(math.ceil(max_displacement / target)) + 1
    count = int(np.clip(count, cfg.min_q_samples, cfg.max_q_samples))
    values = np.linspace(float(candidate.lower), float(candidate.upper), count)
    return _unique_values([*values.tolist(), 0.0])


def _three_states(candidate):
    return _unique_values([
        float(candidate.lower),
        0.5 * (candidate.lower + candidate.upper),
        float(candidate.upper),
        0.0,
    ])


def _transformed_mesh(mesh, candidate, q_value):
    import trimesh

    return trimesh.Trimesh(
        vertices=_transform_points(np.asarray(mesh.vertices), candidate, q_value),
        faces=np.asarray(mesh.faces), process=False,
    )


def _transform_points(points, candidate, q_value):
    axis = _unit(candidate.axis_world)
    if candidate.joint_type == "prismatic":
        return np.asarray(points, dtype=np.float64) + axis * q_value
    origin = np.asarray(candidate.origin_world, dtype=np.float64)
    relative = np.asarray(points, dtype=np.float64) - origin
    cosine, sine = math.cos(q_value), math.sin(q_value)
    return origin + relative * cosine + np.cross(axis, relative) * sine + np.outer(relative @ axis, axis) * (1.0 - cosine)


def _load_mesh(value):
    import trimesh

    if hasattr(value, "vertices") and hasattr(value, "faces"):
        return value
    path = Path(value)
    loaded = trimesh.load(path, force="scene", process=False)
    if not isinstance(loaded, trimesh.Scene):
        return loaded
    triangle_meshes = [
        geometry for geometry in loaded.geometry.values()
        if isinstance(geometry, trimesh.Trimesh) and len(geometry.faces)
    ]
    if triangle_meshes:
        return loaded.to_mesh()
    # Some unit fixtures and diagnostic OBJ files intentionally contain only
    # vertices.  Trimesh represents those as a point cloud which cannot be
    # passed through Scene.to_mesh(), so preserve them as an explicit
    # no-triangle mesh and let the audit return an unavailable/review result.
    vertices = []
    if path.suffix.lower() == ".obj":
        for line in path.read_text(errors="ignore").splitlines():
            if line.startswith("v "):
                fields = line.split()
                if len(fields) >= 4:
                    vertices.append([float(fields[1]), float(fields[2]), float(fields[3])])
    return trimesh.Trimesh(
        vertices=np.asarray(vertices, dtype=np.float64).reshape((-1, 3)),
        faces=np.empty((0, 3), dtype=np.int64),
        process=False,
    )


def _mesh_state(first, second):
    first_faces = len(getattr(first, "faces", ()))
    second_faces = len(getattr(second, "faces", ()))
    return {
        "has_triangles": bool(first_faces and second_faces),
        "body_face_count": int(first_faces),
        "moving_face_count": int(second_faces),
        "body_watertight": bool(first.is_watertight) if first_faces else False,
        "moving_watertight": bool(second.is_watertight) if second_faces else False,
        "body_is_volume": bool(first.is_volume) if first_faces else False,
        "moving_is_volume": bool(second.is_volume) if second_faces else False,
    }


def _combined_diagonal(first, second):
    low = np.minimum(np.asarray(first.bounds)[0], np.asarray(second.bounds)[0])
    high = np.maximum(np.asarray(first.bounds)[1], np.asarray(second.bounds)[1])
    return max(float(np.linalg.norm(high - low)), 1e-6)


def _sample_at_zero(samples):
    return min(samples, key=lambda row: abs(float(row["q"])))


def _unique_values(values: Iterable[float]):
    result = []
    for value in sorted((float(item) for item in values)):
        if not result or abs(value - result[-1]) > 1e-10:
            result.append(value)
    return result


def _ordered_unique(values: Iterable[float]):
    result = []
    for item in values:
        value = float(item)
        if not any(abs(value - previous) <= 1e-10 for previous in result):
            result.append(value)
    return result


def _unit(values):
    vector = np.asarray(values, dtype=np.float64)
    norm = float(np.linalg.norm(vector))
    if norm <= 1e-12:
        raise ValueError("candidate axis has zero length")
    return vector / norm


def _manifold_available():
    try:
        import manifold3d  # noqa: F401
    except (ImportError, ModuleNotFoundError):
        return False
    return True


def _unavailable_report(mesh_state, reason, *, q_values=()):
    return {
        "version": AUDIT_VERSION,
        "status": "unavailable",
        "method": "none",
        "confidence": "none",
        "mesh_state": mesh_state,
        "q_sample_count": len(q_values),
        "q_samples": [],
        "narrow_phase": {"method": "not_run", "exact": False, "samples": []},
        "collision_detected": False,
        "requires_review": True,
        "reason": reason,
    }
