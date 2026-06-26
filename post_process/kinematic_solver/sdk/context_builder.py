"""Build non-cheating context for estimate_limit.py."""

from __future__ import annotations

import json
from pathlib import Path

from post_process.kinematic_solver.utils.motion_direction_prior import load_part_semantics

from .coordinate_frame import with_canonical_coordinate_frame
from .schemas import EstimateContext


def build_context_from_roots(
    *,
    object_id: str,
    converter_output_root: Path,
    source_root: Path,
) -> EstimateContext:
    oracle_path = converter_output_root / f"raw/vlm_oracle/{object_id}.json"
    if oracle_path.is_file():
        oracle = json.loads(oracle_path.read_text())
    else:
        oracle = {"object_id": object_id, "joints": {}}
    part_semantics = load_part_semantics(source_root, object_id)
    evidence = _partseg_evidence(converter_output_root, object_id)
    for joint_name, joint in oracle.get("joints", {}).items():
        labels = []
        for part in joint.get("moving_parts", []):
            labels.extend(part_semantics.get(str(part), []))
        evidence[joint_name] = {"labels": labels}
    return with_canonical_coordinate_frame(EstimateContext(
        object_id=object_id,
        joints=dict(oracle.get("joints", {})),
        evidence=evidence,
    ))


def _partseg_evidence(converter_output_root: Path, object_id: str) -> dict:
    obj_dir = converter_output_root / f"raw/partseg/{object_id}/objs"
    obj_paths = sorted(
        path for path in obj_dir.glob("*.obj")
        if path.stem == "body" or path.stem.startswith("part_")
    )
    if not obj_paths:
        return {}
    return {
        "__available_parts__": [path.stem for path in obj_paths],
        "__part_centers__": {
            path.stem: _obj_vertex_center(path)
            for path in obj_paths
        },
    }


def _obj_vertex_center(path: Path) -> list[float]:
    vertices = []
    for line in path.read_text(errors="ignore").splitlines():
        if not line.startswith("v "):
            continue
        parts = line.split()
        if len(parts) < 4:
            continue
        try:
            vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
        except ValueError:
            continue
    if not vertices:
        return [0.0, 0.0, 0.0]
    count = float(len(vertices))
    return [
        sum(vertex[index] for vertex in vertices) / count
        for index in range(3)
    ]
