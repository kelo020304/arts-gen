#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib
import sys
from dataclasses import dataclass
from multiprocessing import Pool
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "pipeline"))

from config_loader import load_config


voxelize = importlib.import_module("03_voxelize")
load_part_specs = voxelize.load_part_specs
render_surface_voxel = voxelize.render_surface_voxel
render_per_part_voxel = voxelize.render_per_part_voxel


@dataclass(frozen=True)
class WorkItem:
    reconstruction_dir: str
    part_info_dir: str
    object_id: str
    num_angles: int
    resolution: int


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Regenerate voxel visualization PNGs from existing .npy voxel data.",
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to the dataset toolkit YAML config.",
    )
    parser.add_argument(
        "--object-ids",
        help="Optional comma-separated object ID subset, e.g. 100064,100283.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Multiprocessing worker count.",
    )
    return parser.parse_args(argv)


def _parse_object_ids(raw_value: str) -> list[str]:
    object_ids = [item.strip() for item in raw_value.split(",")]
    if not object_ids or any(not item for item in object_ids):
        raise ValueError("--object-ids must be a comma-separated list of non-empty IDs")
    if len(object_ids) != len(set(object_ids)):
        raise ValueError("--object-ids contains duplicate IDs")
    return object_ids


def resolve_object_ids(config: Any, object_ids_arg: str | None) -> list[str]:
    available_object_ids = config.list_object_ids()
    if object_ids_arg is None:
        return available_object_ids

    requested_object_ids = _parse_object_ids(object_ids_arg)
    available_set = set(available_object_ids)
    missing = [object_id for object_id in requested_object_ids if object_id not in available_set]
    if missing:
        raise FileNotFoundError(
            "Missing finaljson for requested object IDs: " + ", ".join(missing)
        )
    return requested_object_ids


def build_work_items(config: Any, object_ids: list[str]) -> list[WorkItem]:
    return [
        WorkItem(
            reconstruction_dir=config.reconstruction_dir,
            part_info_dir=config.part_info_dir,
            object_id=object_id,
            num_angles=config.get_num_angles(object_id),
            resolution=config.voxel.resolution,
        )
        for object_id in object_ids
    ]


def _load_part_indices(voxel_dir: Path) -> dict[str, np.ndarray]:
    per_part_indices: dict[str, np.ndarray] = {}
    for path in sorted(voxel_dir.glob("ind_*.npy"), key=lambda item: item.name):
        part_name = path.name[len("ind_") : -len(".npy")]
        per_part_indices[part_name] = np.load(path)
    return per_part_indices


def _filter_part_specs(part_specs: list[Any], per_part_indices: dict[str, np.ndarray]) -> list[Any]:
    part_names = set(per_part_indices)
    known_names = {spec.canonical_name for spec in part_specs}
    unknown_names = sorted(part_names - known_names)
    if unknown_names:
        raise KeyError("ind_*.npy has part names not present in part_info: " + ", ".join(unknown_names))
    return [spec for spec in part_specs if spec.canonical_name in part_names]


def regenerate_angle(
    *,
    voxel_dir: Path,
    part_specs: list[Any],
    object_id: str,
    angle_idx: int,
    resolution: int,
) -> None:
    if not voxel_dir.is_dir():
        raise FileNotFoundError(f"voxel_dir does not exist: {voxel_dir}")

    surface_path = voxel_dir / "surface.npy"
    if not surface_path.is_file():
        raise FileNotFoundError(f"surface.npy does not exist: {surface_path}")

    surface_indices = np.load(surface_path)
    per_part_indices = _load_part_indices(voxel_dir)
    existing_part_specs = _filter_part_specs(part_specs, per_part_indices)

    viz_dir = voxel_dir / "viz"
    viz_dir.mkdir(parents=True, exist_ok=True)  # D-03: defense-in-depth + explicit intent
    render_surface_voxel(
        allind=surface_indices,
        out_path=viz_dir / "surface_voxel.png",
        object_id=object_id,
        angle_idx=angle_idx,
        resolution=resolution,
    )
    render_per_part_voxel(
        part_specs=existing_part_specs,
        per_part_indices=per_part_indices,
        out_path=viz_dir / "per_part_voxel.png",
        object_id=object_id,
        angle_idx=angle_idx,
        resolution=resolution,
    )


def process_object(work_item: WorkItem) -> dict[str, Any]:
    object_id = work_item.object_id
    reconstruction_dir = Path(work_item.reconstruction_dir)
    part_info_path = Path(work_item.part_info_dir) / object_id / "part_info.json"
    voxel_root = reconstruction_dir / "voxel_expanded" / object_id

    result: dict[str, Any] = {
        "object_id": object_id,
        "angles_regenerated": 0,
        "status": "done",
        "error": None,
    }

    try:
        part_specs = load_part_specs(part_info_path, object_id)
        for angle_idx in range(work_item.num_angles):
            voxel_dir = voxel_root / f"angle_{angle_idx}" / str(work_item.resolution)
            regenerate_angle(
                voxel_dir=voxel_dir,
                part_specs=part_specs,
                object_id=object_id,
                angle_idx=angle_idx,
                resolution=work_item.resolution,
            )
            result["angles_regenerated"] += 1
    except Exception as exc:
        result["status"] = "error"
        result["error"] = f"{type(exc).__name__}: {exc}"

    return result


def format_progress_message(index: int, total: int, result: dict[str, Any]) -> str:
    object_id = result["object_id"]
    regenerated = result["angles_regenerated"]
    if result["status"] == "done":
        return f"[{index}/{total}] {object_id} done ({regenerated} angles regenerated)"
    return f"[{index}/{total}] {object_id} error ({regenerated} angles regenerated): {result['error']}"


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.workers < 1:
        raise ValueError("--workers must be >= 1")

    config = load_config(args.config)
    object_ids = resolve_object_ids(config, args.object_ids)
    work_items = build_work_items(config, object_ids)

    print(
        f"[INFO] Regenerating voxel viz for {len(work_items)} objects "
        f"at {config.voxel.resolution}^3 with {args.workers} worker(s)"
    )

    results: list[dict[str, Any]] = []
    if args.workers == 1:
        for index, item in enumerate(work_items, start=1):
            result = process_object(item)
            results.append(result)
            print(format_progress_message(index, len(work_items), result), flush=True)
    else:
        with Pool(processes=args.workers) as pool:
            for index, result in enumerate(pool.imap_unordered(process_object, work_items), start=1):
                results.append(result)
                print(format_progress_message(index, len(work_items), result), flush=True)

    errors = [result for result in results if result["status"] != "done"]
    regenerated = sum(result["angles_regenerated"] for result in results)

    print(f"[DONE] Angles regenerated: {regenerated}")
    if errors:
        print(f"[ERROR] Objects failed: {len(errors)}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
