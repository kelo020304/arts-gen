#!/usr/bin/env python3
"""Stage 05 render orchestrator for dataset_toolkits.

This file is the single user-facing render entrypoint.  The concrete render
sets stay in separate scripts so each Blender path remains small and debuggable.
Default behavior runs the current mainline render set: part_complete.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "utils"))

from config_loader import load_config  # noqa: E402


REPO_ROOT = Path(__file__).resolve().parent.parent
PIPELINE_ROOT = Path(__file__).resolve().parent


@dataclass(frozen=True)
class RenderSet:
    name: str
    script: str
    requires_manifest: bool
    supports_angle_ids: bool
    supports_part_keys: bool
    supports_num_views: bool
    default_num_views: int | None
    description: str


RENDER_SETS: dict[str, RenderSet] = {
    "full_object_all_views": RenderSet(
        name="full_object_all_views",
        script="05_render_full_object_all_views.py",
        requires_manifest=False,
        supports_angle_ids=True,
        supports_part_keys=False,
        supports_num_views=True,
        default_num_views=150,
        description="Full assembled object all-view RGB render set.",
    ),
    "valid_parts_all_views": RenderSet(
        name="valid_parts_all_views",
        script="05_render_valid_parts_all_views.py",
        requires_manifest=True,
        supports_angle_ids=True,
        supports_part_keys=True,
        supports_num_views=True,
        default_num_views=150,
        description="One 150-view RGB render set per valid movable target part.",
    ),
    "part_complete": RenderSet(
        name="part_complete",
        script="05_render_part_complete_rgb_mask.py",
        requires_manifest=True,
        supports_angle_ids=True,
        supports_part_keys=True,
        supports_num_views=True,
        default_num_views=16,
        description="16-view full-object RGB with valid-part masks and remaining mask.",
    ),
}
RENDER_SET_ALIASES = {
    "full150": "full_object_all_views",
    "full_object_150": "full_object_all_views",
    "parts150": "valid_parts_all_views",
    "valid_part_150": "valid_parts_all_views",
    "part_complete_16": "part_complete",
}


def _parse_csv(raw: str | None) -> list[str]:
    if raw is None:
        return []
    items = [item.strip() for item in raw.split(",") if item.strip()]
    if not items:
        raise ValueError("comma-separated arguments must contain at least one value")
    if len(items) != len(set(items)):
        raise ValueError(f"duplicate values in comma-separated argument: {raw}")
    return items


def _normalize_render_sets(items: list[str]) -> list[str]:
    normalized = [RENDER_SET_ALIASES.get(item, item) for item in items]
    if len(normalized) != len(set(normalized)):
        raise ValueError(f"duplicate render sets after alias normalization: {items}")
    return normalized


def _load_geometry_manifest(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    if not path.is_file():
        raise FileNotFoundError(f"Missing geometry manifest: {path}")
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise TypeError(f"geometry manifest must be an object: {path}")
    return payload


def _summarize_geometry_manifest(path: Path | None, payload: dict[str, Any] | None) -> str | None:
    if path is None or payload is None:
        return None
    summary = payload.get("summary", {}) if isinstance(payload.get("summary"), dict) else {}
    return (
        f"geometry_manifest={path} objects={summary.get('total_objects', '?')} "
        f"angles={summary.get('total_angles', '?')} valid_targets={summary.get('total_valid_target_parts', '?')}"
    )


def build_command(args: argparse.Namespace, render_set: RenderSet) -> list[str]:
    cmd = [sys.executable, str(PIPELINE_ROOT / render_set.script), "--config", args.config]
    if args.object_ids:
        cmd.extend(["--object-ids", args.object_ids])
    if args.workers is not None:
        cmd.extend(["--workers", str(args.workers)])
    if render_set.requires_manifest and args.manifest:
        cmd.extend(["--manifest", args.manifest])
    if render_set.supports_angle_ids and args.angle_ids:
        cmd.extend(["--angle-ids", args.angle_ids])
    if render_set.supports_part_keys and args.part_keys:
        cmd.extend(["--part-keys", args.part_keys])
    if render_set.supports_num_views:
        num_views = args.num_views if args.num_views is not None else render_set.default_num_views
        if num_views is not None:
            cmd.extend(["--num-views", str(num_views)])
    if args.force:
        cmd.append("--force")
    if args.dry_run:
        cmd.append("--dry-run")
    return cmd


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run or dry-run named Stage 05 render sets.")
    parser.add_argument("--config", required=True, help="Path to dataset toolkit YAML config.")
    parser.add_argument(
        "--sets",
        default="part_complete",
        help=(
            "Comma-separated render sets. Canonical: "
            + ", ".join(sorted(RENDER_SETS))
            + ". Aliases: "
            + ", ".join(sorted(RENDER_SET_ALIASES))
            + ". Default: part_complete."
        ),
    )
    parser.add_argument("--object-ids", help="Optional comma-separated object ID subset.")
    parser.add_argument("--angle-ids", help="Optional comma-separated angle subset for render sets that support it.")
    parser.add_argument("--part-keys", help="Optional comma-separated part key subset for part render sets.")
    parser.add_argument("--manifest", help="Delivery manifest path for manifest-gated render sets.")
    parser.add_argument("--geometry-manifest", help="Optional geometry manifest path for audit/dry-run reporting.")
    parser.add_argument("--workers", type=int, default=1, help="Worker count passed to child render scripts.")
    parser.add_argument("--num-views", type=int, help="Override views for all render sets that support --num-views.")
    parser.add_argument("--force", action="store_true", help="Pass --force to render sets that support it.")
    parser.add_argument("--dry-run", action="store_true", help="Print planned commands and pass child --dry-run.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = load_config(args.config)
    requested = _normalize_render_sets(_parse_csv(args.sets))
    unknown = sorted(set(requested) - set(RENDER_SETS))
    if unknown:
        raise ValueError(f"Unknown render sets: {', '.join(unknown)}")

    if args.manifest is None:
        args.manifest = str(Path(config.data_root) / "manifests" / f"{config.dataset_name}.json")

    geometry_manifest_path = Path(args.geometry_manifest) if args.geometry_manifest else None
    geometry_manifest = _load_geometry_manifest(geometry_manifest_path)
    geometry_summary = _summarize_geometry_manifest(geometry_manifest_path, geometry_manifest)
    if geometry_summary:
        print(f"[render] {geometry_summary}")

    for set_name in requested:
        render_set = RENDER_SETS[set_name]
        if args.angle_ids and not render_set.supports_angle_ids:
            print(f"[render] note: {set_name} ignores --angle-ids (child script does not support it)")
        if args.part_keys and not render_set.supports_part_keys:
            print(f"[render] note: {set_name} ignores --part-keys")
        cmd = build_command(args, render_set)
        print(f"[render] set={set_name}: {render_set.description}", flush=True)
        print("[render] command: " + " ".join(cmd), flush=True)
        subprocess.run(cmd, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
