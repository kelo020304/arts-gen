#!/usr/bin/env python3
"""Repair RealAppliance door/lid targets whose panel voxels live in child parts.

The RealAppliance source grouping sometimes stores one logical movable part as:

* a parent door/lid/glass target containing only the frame or ring; and
* one fixed child group containing the middle glass/panel voxels.

This script merges selected child panel voxels into the parent target voxel file,
removes the child ``ind_*.npy`` file to avoid duplicate supervision, and rewrites
``surface.npy`` as the union of the remaining part voxel files.

It is dry-run by default. Pass ``--apply`` to modify ``voxel_expanded``.
"""
from __future__ import annotations

import argparse
import json
import shutil
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


DEFAULT_DATA_ROOT = Path("/robot/data-lab/jzh/art-gen/data/realappliance")
DEFAULT_AUDIT_ROOT = (
    DEFAULT_DATA_ROOT / "manifests" / "door_lid_panel_audit_20260618"
)
DEFAULT_REPAIR_ROOT = (
    DEFAULT_DATA_ROOT / "manifests" / "door_lid_panel_repair_20260618"
)
DEFAULT_CONNECTIVITY_BACKUP_ROOT = (
    DEFAULT_DATA_ROOT
    / "manifests"
    / "connectivity_repair_20260618"
    / "backup"
)
DEFAULT_RESOLUTION = "64"
STRUCTURAL_CHILD_MIN_AVG_VOXELS = 80.0
STRUCTURAL_CHILD_MIN_AREA_RATIO = 0.09


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def coords_to_set(coords: np.ndarray) -> set[tuple[int, int, int]]:
    if coords.size == 0:
        return set()
    arr = np.asarray(coords, dtype=np.int64)
    if arr.ndim != 2 or arr.shape[1] != 3:
        raise ValueError(f"expected coordinate array [N,3], got {arr.shape}")
    return {tuple(map(int, row)) for row in arr.tolist()}


def set_to_coords(voxels: set[tuple[int, int, int]]) -> np.ndarray:
    if not voxels:
        return np.empty((0, 3), dtype=np.int64)
    return np.asarray(sorted(voxels), dtype=np.int64)


def load_voxels(path: Path) -> set[tuple[int, int, int]]:
    return coords_to_set(np.load(path, allow_pickle=False))


def save_voxels(path: Path, voxels: set[tuple[int, int, int]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, set_to_coords(voxels))


def voxel_dir(data_root: Path, object_id: str, angle: str, resolution: str) -> Path:
    return data_root / "reconstruction" / "voxel_expanded" / object_id / angle / resolution


def ind_name(part_name: str) -> str:
    return f"ind_{part_name}.npy"


def source_candidates(
    *,
    data_root: Path,
    repair_root: Path,
    connectivity_backup_root: Path,
    object_id: str,
    angle: str,
    resolution: str,
    part_name: str,
) -> list[tuple[str, Path]]:
    rel = Path("reconstruction") / "voxel_expanded" / object_id / angle / resolution / ind_name(part_name)
    return [
        ("structural_child_patch_backup", repair_root / "backup_current_before_structural_child_patch" / rel),
        ("connectivity_repair_20260618_backup", connectivity_backup_root / rel),
        ("panel_repair_current_backup", repair_root / "backup_current_before_panel_repair" / rel),
        ("live", data_root / rel),
    ]


def is_anonymous_structural_child(rec: dict[str, Any]) -> bool:
    child_type = str(rec.get("child_type", ""))
    return (
        str(rec.get("child_joint_type")) == "E"
        and child_type.startswith("part_")
        and not bool(rec.get("control_name"))
        and float(rec.get("avg_child_voxels") or 0.0) >= STRUCTURAL_CHILD_MIN_AVG_VOXELS
        and float(rec.get("area_ratio_2d") or 0.0) >= STRUCTURAL_CHILD_MIN_AREA_RATIO
    )


def load_best_source(
    *,
    data_root: Path,
    repair_root: Path,
    connectivity_backup_root: Path,
    object_id: str,
    angle: str,
    resolution: str,
    part_name: str,
) -> tuple[set[tuple[int, int, int]], str, str]:
    for source_kind, path in source_candidates(
        data_root=data_root,
        repair_root=repair_root,
        connectivity_backup_root=connectivity_backup_root,
        object_id=object_id,
        angle=angle,
        resolution=resolution,
        part_name=part_name,
    ):
        if path.is_file():
            return load_voxels(path), source_kind, str(path)
    raise FileNotFoundError(f"no voxel source found for {object_id}/{angle}/{part_name}")


def discover_targets(audit_root: Path) -> list[dict[str, Any]]:
    """Load the reviewed parent/child panel candidates.

    The candidate file was produced by the audit pass and intentionally encodes
    the human-reviewed scope: parent target plus fixed child panel part. The
    repair also absorbs anonymous fixed structural child parts, such as door
    handles, when they are attached to the same door/lid parent group. It avoids
    broad name-only heuristics that could accidentally absorb buttons or knobs.
    """

    path = audit_root / "likely_panel_child_candidates.json"
    records = load_json(path)
    grouped: dict[tuple[str, str], set[str]] = defaultdict(set)
    for rec in records:
        if not rec.get("likely_panel_child", True):
            continue
        object_id = str(rec["object_id"])
        target_part = str(rec["target_part"])
        child_part = str(rec["child_part"])
        grouped[(object_id, target_part)].add(child_part)

    structural_path = audit_root / "non_panel_group_children.json"
    if structural_path.is_file():
        structural_records = load_json(structural_path)
        for rec in structural_records:
            if not is_anonymous_structural_child(rec):
                continue
            object_id = str(rec["object_id"])
            target_part = str(rec["target_part"])
            child_part = str(rec["child_part"])
            grouped[(object_id, target_part)].add(child_part)

    return [
        {"object_id": object_id, "target_part": target_part, "child_parts": sorted(child_parts)}
        for (object_id, target_part), child_parts in sorted(grouped.items())
    ]


def build_preview_label_merges(data_root: Path, targets: list[dict[str, Any]]) -> dict[str, Any]:
    merges: dict[str, dict[str, Any]] = {}
    for target in targets:
        object_id = target["object_id"]
        part_info_path = data_root / "reconstruction" / "part_info" / object_id / "part_info.json"
        if not part_info_path.is_file():
            continue
        part_info = load_json(part_info_path)
        parts = part_info.get("parts") or {}
        parent = parts.get(target["target_part"])
        if not isinstance(parent, dict):
            continue
        child_labels: list[int] = []
        child_parts: list[str] = []
        for child_part in target["child_parts"]:
            child = parts.get(child_part)
            if not isinstance(child, dict) or child.get("label") is None:
                continue
            child_parts.append(child_part)
            child_labels.append(int(child["label"]))
        if not child_labels:
            continue
        merges.setdefault(object_id, {})[target["target_part"]] = {
            "parent_label": int(parent["label"]),
            "child_parts": child_parts,
            "child_labels": sorted(set(child_labels)),
        }
    return {
        "created_utc": utc_now(),
        "description": (
            "Preview-only label merge for RealAppliance repaired door/lid targets. "
            "Overlay masks should color selected fixed child panel/handle labels as the parent target."
        ),
        "merges": merges,
    }


def angle_dirs_for_object(data_root: Path, object_id: str, resolution: str) -> list[Path]:
    root = data_root / "reconstruction" / "voxel_expanded" / object_id
    if not root.is_dir():
        raise FileNotFoundError(f"missing voxel object directory: {root}")
    return [
        path for path in sorted(root.glob("angle_*"))
        if (path / resolution).is_dir()
    ]


def copy_once(src: Path, dst: Path) -> None:
    if dst.exists():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def backup_live_files(
    *,
    backup_root: Path,
    data_root: Path,
    object_id: str,
    angle: str,
    resolution: str,
    part_names: list[str],
) -> None:
    live_dir = voxel_dir(data_root, object_id, angle, resolution)
    rel_dir = Path("reconstruction") / "voxel_expanded" / object_id / angle / resolution
    for name in [ind_name(part_name) for part_name in part_names] + ["surface.npy"]:
        src = live_dir / name
        if src.is_file():
            copy_once(src, backup_root / rel_dir / name)


def recompute_surface(part_dir: Path) -> tuple[set[tuple[int, int, int]], int]:
    surface: set[tuple[int, int, int]] = set()
    part_count = 0
    for path in sorted(part_dir.glob("ind_*.npy")):
        surface |= load_voxels(path)
        part_count += 1
    return surface, part_count


def validate_live_target(
    *,
    data_root: Path,
    repair_root: Path,
    connectivity_backup_root: Path,
    object_id: str,
    angle: str,
    resolution: str,
    target_part: str,
    child_parts: list[str],
) -> dict[str, Any]:
    parent_pre, parent_kind, _ = load_best_source(
        data_root=data_root,
        repair_root=repair_root,
        connectivity_backup_root=connectivity_backup_root,
        object_id=object_id,
        angle=angle,
        resolution=resolution,
        part_name=target_part,
    )
    child_union: set[tuple[int, int, int]] = set()
    child_counts: dict[str, int] = {}
    child_kinds: dict[str, str] = {}
    for child_part in child_parts:
        child_voxels, child_kind, _ = load_best_source(
            data_root=data_root,
            repair_root=repair_root,
            connectivity_backup_root=connectivity_backup_root,
            object_id=object_id,
            angle=angle,
            resolution=resolution,
            part_name=child_part,
        )
        child_union |= child_voxels
        child_counts[child_part] = len(child_voxels)
        child_kinds[child_part] = child_kind

    expected_parent = parent_pre | child_union
    part_dir = voxel_dir(data_root, object_id, angle, resolution)
    live_parent_path = part_dir / ind_name(target_part)
    live_parent = load_voxels(live_parent_path) if live_parent_path.is_file() else set()
    child_files_present = [
        child_part for child_part in child_parts if (part_dir / ind_name(child_part)).exists()
    ]
    surface_path = part_dir / "surface.npy"
    live_surface = load_voxels(surface_path) if surface_path.is_file() else set()
    recomputed_surface, part_file_count = recompute_surface(part_dir)

    return {
        "object_id": object_id,
        "angle": angle,
        "target_part": target_part,
        "child_parts": child_parts,
        "pre_parent_voxels": len(parent_pre),
        "child_union_voxels": len(child_union),
        "after_parent_voxels": len(live_parent),
        "added_to_parent": len(expected_parent - parent_pre),
        "parent_pre_source": parent_kind,
        "child_pre_sources": child_kinds,
        "child_counts": child_counts,
        "parent_matches_expected": live_parent == expected_parent,
        "child_files_present": child_files_present,
        "surface_matches_part_union": live_surface == recomputed_surface,
        "surface_voxels": len(live_surface),
        "part_file_count": part_file_count,
    }


def apply_target(
    *,
    data_root: Path,
    repair_root: Path,
    connectivity_backup_root: Path,
    backup_root: Path,
    object_id: str,
    angle: str,
    resolution: str,
    target_part: str,
    child_parts: list[str],
    apply: bool,
) -> dict[str, Any]:
    parent_pre, parent_kind, parent_path = load_best_source(
        data_root=data_root,
        repair_root=repair_root,
        connectivity_backup_root=connectivity_backup_root,
        object_id=object_id,
        angle=angle,
        resolution=resolution,
        part_name=target_part,
    )
    child_union: set[tuple[int, int, int]] = set()
    child_sources: dict[str, dict[str, Any]] = {}
    for child_part in child_parts:
        child_voxels, child_kind, child_path = load_best_source(
            data_root=data_root,
            repair_root=repair_root,
            connectivity_backup_root=connectivity_backup_root,
            object_id=object_id,
            angle=angle,
            resolution=resolution,
            part_name=child_part,
        )
        child_union |= child_voxels
        child_sources[child_part] = {
            "source_kind": child_kind,
            "path": child_path,
            "voxels": len(child_voxels),
        }

    expected_parent = parent_pre | child_union
    part_dir = voxel_dir(data_root, object_id, angle, resolution)
    live_parent_path = part_dir / ind_name(target_part)
    previous_parent = load_voxels(live_parent_path) if live_parent_path.is_file() else set()
    child_files_present = [
        child_part for child_part in child_parts if (part_dir / ind_name(child_part)).exists()
    ]

    if apply:
        backup_live_files(
            backup_root=backup_root,
            data_root=data_root,
            object_id=object_id,
            angle=angle,
            resolution=resolution,
            part_names=[target_part, *child_parts],
        )
        save_voxels(live_parent_path, expected_parent)
        for child_part in child_parts:
            child_path = part_dir / ind_name(child_part)
            if child_path.exists():
                child_path.unlink()
        surface, part_file_count = recompute_surface(part_dir)
        save_voxels(part_dir / "surface.npy", surface)
    else:
        surface, part_file_count = recompute_surface(part_dir)

    return {
        "object_id": object_id,
        "angle": angle,
        "target_part": target_part,
        "child_parts": child_parts,
        "parent_source": parent_path,
        "parent_source_kind": parent_kind,
        "child_sources": child_sources,
        "pre_parent_voxels": len(parent_pre),
        "child_union_voxels": len(child_union),
        "expected_parent_voxels": len(expected_parent),
        "previous_parent_voxels": len(previous_parent),
        "added_vs_previous": len(expected_parent - previous_parent),
        "removed_vs_previous": len(previous_parent - expected_parent),
        "child_files_present_before": child_files_present,
        "surface_voxels": len(surface),
        "part_file_count": part_file_count,
        "applied": bool(apply),
    }


def run(args: argparse.Namespace) -> int:
    data_root = args.data_root
    audit_root = args.audit_root
    repair_root = args.repair_root
    connectivity_backup_root = args.connectivity_backup_root
    resolution = str(args.resolution)
    report_root = args.report_root or (
        data_root / "manifests" / f"door_lid_panel_repair_replay_{utc_now().replace(':', '')}"
    )
    backup_root = args.backup_root or (
        report_root / "backup_current_before_structural_child_patch"
    )

    targets = discover_targets(audit_root)
    if args.object_ids:
        object_filter = {item.strip() for item in args.object_ids.split(",") if item.strip()}
        targets = [target for target in targets if target["object_id"] in object_filter]

    records: list[dict[str, Any]] = []
    for target in targets:
        object_id = target["object_id"]
        target_part = target["target_part"]
        child_parts = target["child_parts"]
        for angle_dir in angle_dirs_for_object(data_root, object_id, resolution):
            records.append(
                apply_target(
                    data_root=data_root,
                    repair_root=repair_root,
                    connectivity_backup_root=connectivity_backup_root,
                    backup_root=backup_root,
                    object_id=object_id,
                    angle=angle_dir.name,
                    resolution=resolution,
                    target_part=target_part,
                    child_parts=child_parts,
                    apply=args.apply,
                )
            )

    validation: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    for rec in records:
        check = validate_live_target(
            data_root=data_root,
            repair_root=repair_root,
            connectivity_backup_root=connectivity_backup_root,
            object_id=rec["object_id"],
            angle=rec["angle"],
            resolution=resolution,
            target_part=rec["target_part"],
            child_parts=rec["child_parts"],
        )
        validation.append(check)
        if not check["parent_matches_expected"]:
            errors.append({**check, "reason": "parent != pre_parent union child_union"})
        if check["child_files_present"]:
            errors.append({**check, "reason": "child ind files still present"})
        if not check["surface_matches_part_union"]:
            errors.append({**check, "reason": "surface != union(ind_*.npy)"})

    by_target: dict[tuple[str, str], dict[str, Any]] = {}
    for check in validation:
        key = (check["object_id"], check["target_part"])
        acc = by_target.setdefault(
            key,
            {
                "object_id": check["object_id"],
                "target_part": check["target_part"],
                "views": 0,
                "pre_parent": 0,
                "child_union": 0,
                "after": 0,
                "added": 0,
            },
        )
        acc["views"] += 1
        acc["pre_parent"] += check["pre_parent_voxels"]
        acc["child_union"] += check["child_union_voxels"]
        acc["after"] += check["after_parent_voxels"]
        acc["added"] += check["added_to_parent"]

    summary = {
        "created_utc": utc_now(),
        "mode": "apply" if args.apply else "dry_run",
        "data_root": str(data_root),
        "audit_root": str(audit_root),
        "repair_root": str(repair_root),
        "connectivity_backup_root": str(connectivity_backup_root),
        "backup_root": str(backup_root) if args.apply else None,
        "target_count": len(targets),
        "panel_child_count": sum(len(target["child_parts"]) for target in targets),
        "object_count": len({target["object_id"] for target in targets}),
        "modified_part_angle_records": len(records) if args.apply else 0,
        "checked_part_angle_records": len(records),
        "errors": errors,
        "packed_dirs_requiring_rebuild": [
            str(data_root.parent / "part_promptable_seg_packed_v4")
        ],
        "policy": (
            "Merge selected fixed panel/handle child voxels into the parent door/lid/glass "
            "target, delete selected child ind files, and recompute surface.npy as "
            "the union of current ind_*.npy files. Source priority is "
            "structural child patch backup, connectivity repair backup, panel repair backup, then live data."
        ),
        "by_target": list(by_target.values()),
    }

    write_json(report_root / "repair_records.json", records)
    write_json(report_root / "validation.json", {"errors": errors, "stats": validation})
    write_json(report_root / "summary.json", summary)
    preview_merges = build_preview_label_merges(data_root, targets)
    write_json(report_root / "preview_label_merges.json", preview_merges)
    if args.apply:
        write_json(repair_root / "preview_label_merges.json", preview_merges)

    print(f"mode={summary['mode']}")
    print(f"targets={summary['target_count']} child_parts={summary['panel_child_count']} records={len(records)}")
    print(f"errors={len(errors)}")
    print(f"report={report_root}")
    if not args.apply:
        print("dry-run only; pass --apply to modify voxel_expanded")
    return 1 if errors else 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--audit-root", type=Path, default=DEFAULT_AUDIT_ROOT)
    parser.add_argument("--repair-root", type=Path, default=DEFAULT_REPAIR_ROOT)
    parser.add_argument(
        "--connectivity-backup-root",
        type=Path,
        default=DEFAULT_CONNECTIVITY_BACKUP_ROOT,
    )
    parser.add_argument("--resolution", default=DEFAULT_RESOLUTION)
    parser.add_argument("--object-ids", help="Optional comma-separated object ids.")
    parser.add_argument("--report-root", type=Path)
    parser.add_argument("--backup-root", type=Path)
    parser.add_argument("--apply", action="store_true", help="Modify voxel_expanded.")
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
