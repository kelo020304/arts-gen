#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
import shutil
from pathlib import Path

from config_loader import load_config


ANGLE_DIR_RE = re.compile(r"angle_(\d+)$")
TARGET_SUBDIRS = (
    Path("renders"),
    Path("reconstruction/voxel_expanded"),
    Path("reconstruction/part_labels"),
    Path("reconstruction/ss_latents_expanded"),
    Path("reconstruction/dinov2_tokens"),
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Remove extra angle directories for static objects, keeping only angle_0.",
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to the dataset toolkit YAML config.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print directories that would be deleted without removing them.",
    )
    return parser.parse_args(argv)


def _log(message: str, dry_run: bool) -> None:
    prefix = "[DRY RUN] " if dry_run else ""
    print(f"{prefix}{message}")


def _list_angle_indices_to_delete(
    renders_object_dir: Path,
    num_angles: int,
) -> list[int]:
    if not renders_object_dir.is_dir():
        return []

    angle_indices: list[int] = []
    for path in sorted(renders_object_dir.iterdir()):
        if not path.is_dir():
            continue
        match = ANGLE_DIR_RE.fullmatch(path.name)
        if match is None:
            continue
        angle_idx = int(match.group(1))
        if angle_idx >= num_angles:
            raise ValueError(
                f"Unexpected angle directory outside configured range [0, {num_angles - 1}]: {path}"
            )
        if angle_idx != 0:
            angle_indices.append(angle_idx)
    return angle_indices


def _delete_object_angle_dirs(
    data_root: Path,
    object_id: str,
    angle_indices: list[int],
    dry_run: bool,
) -> int:
    deleted_dir_count = 0

    for angle_idx in angle_indices:
        angle_dir_name = f"angle_{angle_idx}"
        for subdir in TARGET_SUBDIRS:
            target_dir = data_root / subdir / object_id / angle_dir_name
            if not target_dir.is_dir():
                continue
            _log(f"delete {target_dir}", dry_run=dry_run)
            if not dry_run:
                shutil.rmtree(target_dir)
            deleted_dir_count += 1

    return deleted_dir_count


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = load_config(args.config)

    static_objects = config.joint_transform.static_objects
    if not static_objects:
        _log("No static objects in config; nothing to clean up.", dry_run=args.dry_run)
        return 0

    num_angles = config.joint_transform.num_angles
    data_root = Path(config.data_root)
    total_objects = len(static_objects)
    total_deleted_dirs = 0

    for index, object_id in enumerate(static_objects, start=1):
        renders_object_dir = data_root / "renders" / object_id
        angle_indices = _list_angle_indices_to_delete(
            renders_object_dir=renders_object_dir,
            num_angles=num_angles,
        )
        deleted_dirs = _delete_object_angle_dirs(
            data_root=data_root,
            object_id=object_id,
            angle_indices=angle_indices,
            dry_run=args.dry_run,
        )
        total_deleted_dirs += deleted_dirs
        _log(
            f"[{index}/{total_objects}] {object_id} done (deleted {deleted_dirs} dirs)",
            dry_run=args.dry_run,
        )

    action = "Would delete" if args.dry_run else "Deleted"
    _log(
        f"{action} {total_deleted_dirs} dirs across {total_objects} static objects "
        "(directory count only; size not computed).",
        dry_run=args.dry_run,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
