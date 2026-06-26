"""Phase-0 data preparation for V1 KinematicSolver."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import defaultdict, deque
from pathlib import Path
from typing import Any

import numpy as np

from .config import (
    V1_COACD_RUN_PARAMS,
    V1_TEN_IDS,
    V1_VHACD_CACHE_METADATA,
    V1DatasetRoots,
)
from .usd_limit_reader import read_limits_from_source_usd

_AXIS_TO_VECTOR = {
    "X": [1.0, 0.0, 0.0],
    "Y": [0.0, 1.0, 0.0],
    "Z": [0.0, 0.0, 1.0],
}
_OBJ_FILE_RE = re.compile(r"^(body|part_\d+)$")


def list_obj_groups(obj_dir: Path) -> set[str]:
    """Return the expected baked OBJ group stems for one object."""
    return {p.stem for p in obj_dir.glob("*.obj") if _OBJ_FILE_RE.match(p.stem)}


def _path_basename(path: Any) -> str:
    return str(path).rstrip("/").split("/")[-1]


def _rel_target(prim, name: str):
    rel = prim.GetRelationship(name)
    targets = rel.GetTargets() if rel else []
    return targets[0] if targets else None


def _attr_value(prim, name: str, default=None):
    attr = prim.GetAttribute(name)
    if not attr:
        return default
    value = attr.Get()
    return default if value is None else value


def _fixed_children(stage) -> dict[str, set[str]]:
    children: dict[str, set[str]] = defaultdict(set)
    meters_per_unit = float(stage.GetMetadata("metersPerUnit") or 1.0)
    for prim in stage.Traverse():
        if prim.GetTypeName() != "PhysicsFixedJoint":
            continue
        body0 = _rel_target(prim, "physics:body0")
        body1 = _rel_target(prim, "physics:body1")
        if body0 and body1:
            children[_path_basename(body0)].add(_path_basename(body1))
    return children


def _closure(seed: str, graph: dict[str, set[str]]) -> set[str]:
    seen: set[str] = set()
    q = deque([seed])
    while q:
        cur = q.popleft()
        if cur in seen:
            continue
        seen.add(cur)
        q.extend(sorted(graph.get(cur, ())))
    return seen


def parse_usd_joints(source_usd: Path, object_id: str, obj_stems: set[str]) -> tuple[dict, dict]:
    """Parse authored USD joints into V1 oracle joints and GT limits."""
    from pxr import Usd

    stage = Usd.Stage.Open(str(source_usd))
    if stage is None:
        raise FileNotFoundError(f"failed to open USD stage: {source_usd}")

    fixed_graph = _fixed_children(stage)
    joints: dict[str, dict] = {}
    limits: dict[str, dict] = {}
    joint_prim_paths: dict[str, str] = {}
    body0_paths: dict[str, str | None] = {}
    body1_paths: dict[str, str | None] = {}
    meters_per_unit = float(stage.GetMetadata("metersPerUnit") or 1.0)

    for prim in stage.Traverse():
        type_name = prim.GetTypeName()
        if type_name not in {"PhysicsPrismaticJoint", "PhysicsRevoluteJoint"}:
            continue

        joint_name = prim.GetName()
        joint_type = "prismatic" if type_name == "PhysicsPrismaticJoint" else "revolute"
        axis_name = str(_attr_value(prim, "physics:axis", "X"))
        axis_world = _AXIS_TO_VECTOR.get(axis_name, [1.0, 0.0, 0.0])
        origin = _attr_value(prim, "physics:localPos0", (0.0, 0.0, 0.0))
        origin_world = [float(x) for x in origin]
        lower, upper = read_limits_from_source_usd(
            prim,
            joint_type,
            meters_per_unit=meters_per_unit,
        )

        body0 = _rel_target(prim, "physics:body0")
        body1 = _rel_target(prim, "physics:body1")
        body0_name = _path_basename(body0) if body0 else "body"
        body1_name = _path_basename(body1) if body1 else joint_name
        moving = sorted(_closure(body1_name, fixed_graph) & obj_stems)
        if not moving and body1_name in obj_stems:
            moving = [body1_name]
        static = sorted(obj_stems - set(moving))

        joints[joint_name] = {
            "object_id": object_id,
            "joint_name": joint_name,
            "joint_path": str(prim.GetPath()),
            "type": joint_type,
            "canonical_unit": "meters" if joint_type == "prismatic" else "radians",
            "axis_world": axis_world,
            "origin_world": origin_world,
            "moving_parts": moving,
            "static_parts": static,
            "body0_path": str(body0) if body0 else None,
            "child_body_path": str(body1) if body1 else None,
            "body0_link_name": body0_name,
        }
        limits[joint_name] = {
            "lower": lower,
            "upper": upper,
            "type": joint_type,
            "canonical_unit": "meters" if joint_type == "prismatic" else "radians",
        }
        joint_prim_paths[joint_name] = str(prim.GetPath())
        body0_paths[joint_name] = str(body0) if body0 else None
        body1_paths[joint_name] = str(body1) if body1 else None

    stage_metadata = {
        "object_id": object_id,
        "source_id": object_id.removeprefix("ra_"),
        "meters_per_unit": meters_per_unit,
        "stage_up_axis": str(stage.GetMetadata("upAxis") or "Z"),
        "joint_prim_paths": joint_prim_paths,
        "body0_paths": body0_paths,
        "child_body_paths": body1_paths,
    }
    return {"object_id": object_id, "joints": joints}, {
        "object_id": object_id,
        "limits": limits,
        "stage_metadata": stage_metadata,
    }


def write_default_artifacts(roots: V1DatasetRoots, object_id: str) -> None:
    """Write vlm_oracle, gt_limits, and stage_metadata for one object."""
    co = roots.converter_output_root
    source_usd = roots.aligned_usd_for(object_id)
    obj_dir = co / f"raw/partseg/{object_id}/objs"
    obj_stems = list_obj_groups(obj_dir)
    oracle, gt_payload = parse_usd_joints(source_usd, object_id, obj_stems)

    for subdir in ("vlm_oracle", "gt_limits", "stage_metadata"):
        (co / f"raw/{subdir}").mkdir(parents=True, exist_ok=True)
    (co / f"raw/vlm_oracle/{object_id}.json").write_text(
        json.dumps(oracle, indent=2)
    )
    (co / f"raw/gt_limits/{object_id}.json").write_text(
        json.dumps({
            "object_id": object_id,
            "limits": gt_payload["limits"],
        }, indent=2)
    )
    (co / f"raw/stage_metadata/{object_id}.json").write_text(
        json.dumps(gt_payload["stage_metadata"], indent=2)
    )


def _mesh_to_json(vertices, faces) -> dict:
    return {
        "vertices": np.asarray(vertices, dtype=float).tolist(),
        "faces": np.asarray(faces, dtype=int).tolist(),
    }


def cook_vhacd_for(
    object_id: str,
    part_obj_path: Path,
    out_dir: Path,
    coacd_run_params: dict,
    vhacd_cache_metadata: dict,
) -> Path:
    """Cook one OBJ into a deterministic JSON convex decomposition cache."""
    import coacd
    import trimesh

    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"{part_obj_path.stem}.json"
    source_sha = hashlib.sha256(part_obj_path.read_bytes()).hexdigest()
    if out_file.is_file():
        cached = json.loads(out_file.read_text())
        if (
            cached.get("source_sha256") == source_sha
            and cached.get("coacd_run_params") == coacd_run_params
            and cached.get("vhacd_cache_metadata") == vhacd_cache_metadata
        ):
            return out_file

    mesh = trimesh.load(part_obj_path, force="mesh", process=False)
    if mesh.vertices.size == 0 or mesh.faces.size == 0:
        raise ValueError(f"empty mesh cannot be decomposed: {part_obj_path}")
    coacd_mesh = coacd.Mesh(np.asarray(mesh.vertices), np.asarray(mesh.faces))
    hulls = []
    for i, (vertices, faces) in enumerate(coacd.run_coacd(coacd_mesh, **coacd_run_params)):
        hull = _mesh_to_json(vertices, faces)
        hull["hull_index"] = i
        hulls.append(hull)

    payload = {
        "object_id": object_id,
        "part_name": part_obj_path.stem,
        "source_obj": str(part_obj_path),
        "source_sha256": source_sha,
        "vhacd_cache_metadata": vhacd_cache_metadata,
        "coacd_run_params": coacd_run_params,
        "frame": "world_baked",
        "hulls": hulls,
        "n_hulls": len(hulls),
    }
    out_file.write_text(json.dumps(payload))
    return out_file


def run_default(roots: V1DatasetRoots, ids: list[str]) -> None:
    for object_id in ids:
        write_default_artifacts(roots, object_id)
        print(f"[OK] default artifacts {object_id}")


def run_vhacd(roots: V1DatasetRoots, ids: list[str]) -> None:
    for object_id in ids:
        obj_dir = roots.converter_output_root / f"raw/partseg/{object_id}/objs"
        out_dir = roots.converter_output_root / f"raw/vhacd/{object_id}"
        for obj in sorted(obj_dir.glob("*.obj")):
            if not _OBJ_FILE_RE.match(obj.stem):
                continue
            cook_vhacd_for(
                object_id=object_id,
                part_obj_path=obj,
                out_dir=out_dir,
                coacd_run_params=dict(V1_COACD_RUN_PARAMS),
                vhacd_cache_metadata=dict(V1_VHACD_CACHE_METADATA),
            )
        print(f"[OK] vhacd cache {object_id}")


def main() -> None:
    parser = argparse.ArgumentParser(description="V1 KinematicSolver data prep")
    parser.add_argument("--stage", choices=["default", "vhacd"], required=True)
    parser.add_argument("--converter-output-root", type=Path, required=True)
    parser.add_argument("--source-root", type=Path)
    parser.add_argument("--object-ids", default=",".join(V1_TEN_IDS))
    args = parser.parse_args()

    roots = V1DatasetRoots(
        converter_output_root=args.converter_output_root,
        source_root=args.source_root or V1DatasetRoots().source_root,
    )
    ids = [s.strip() for s in args.object_ids.split(",") if s.strip()]

    if args.stage == "default":
        run_default(roots, ids)
    elif args.stage == "vhacd":
        run_vhacd(roots, ids)


if __name__ == "__main__":
    main()
