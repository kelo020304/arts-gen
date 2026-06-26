#!/usr/bin/env python3
"""Preflight file-integrity check for single-view Part SS latent flow data.

This is an existence checker by default: it validates that manifest rows point
to the files the single-view trainer will need, without loading large arrays.
Use ``--check-shapes`` only for small samples because DINO token NPZ files are
large.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


REQUIRED_ROW_FIELDS = ("object_id", "angle_idx", "sample_id")


def _read_jsonl(path: Path) -> Iterable[tuple[int, dict[str, Any]]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                yield line_no, json.loads(line)
            except json.JSONDecodeError as exc:
                yield line_no, {"__json_error__": str(exc)}


def _resolve(data_root: Path, rel_or_abs: str) -> Path:
    path = Path(rel_or_abs)
    return path if path.is_absolute() else data_root / path


def _target_names(rec: dict[str, Any]) -> list[str]:
    names = rec.get("target_part_names")
    if names is not None:
        return [str(name) for name in names]
    return [str(part.get("name")) for part in rec.get("target_parts") or []]


def _view_indices(rec: dict[str, Any]) -> list[int]:
    if rec.get("view_indices") is not None:
        return [int(v) for v in rec["view_indices"]]
    if rec.get("view_idx") is not None:
        return [int(rec["view_idx"])]
    return []


def _dino_tokens_rel(rec: dict[str, Any], obj_id: str, angle_idx: int) -> str:
    paths = dict(rec.get("paths") or {})
    dinov2 = dict(rec.get("dinov2") or {})
    return str(
        paths.get("dinov2_tokens")
        or rec.get("feature_path")
        or dinov2.get("tokens_path")
        or f"reconstruction/dinov2_tokens/{obj_id}/angle_{angle_idx}/part_complete/tokens.npz"
    )


def _iter_required_files(rec: dict[str, Any]) -> Iterable[tuple[str, str, str | None]]:
    obj_id = str(rec["object_id"])
    angle_idx = int(rec["angle_idx"])
    paths = dict(rec.get("paths") or {})

    yield (
        "global_latent",
        str(
            paths.get("overall_latent")
            or f"reconstruction/ss_latents_expanded/{obj_id}/angle_{angle_idx}/latent.npz"
        ),
        None,
    )
    yield (
        "surface_voxel",
        str(
            paths.get("overall_surface")
            or f"reconstruction/voxel_expanded/{obj_id}/angle_{angle_idx}/64/surface.npy"
        ),
        None,
    )
    yield ("dino_tokens", _dino_tokens_rel(rec, obj_id, angle_idx), None)

    mask_paths = rec.get("mask_paths")
    if mask_paths is not None:
        for mask_path in mask_paths:
            yield ("mask", str(mask_path), None)
    else:
        for view_idx in _view_indices(rec):
            yield (
                "mask",
                f"renders/{obj_id}/angle_{angle_idx}/mask/mask_{view_idx}.npy",
                None,
            )

    part_by_name = {str(part.get("name")): part for part in rec.get("target_parts") or []}
    for part_name in _target_names(rec):
        part = dict(part_by_name.get(part_name) or {})
        part_paths = dict(part.get("paths") or {})
        yield (
            "part_latent",
            str(
                part_paths.get("part_latent")
                or f"reconstruction/ss_latents_per_part/{obj_id}/angle_{angle_idx}/{part_name}.npy"
            ),
            part_name,
        )
        yield (
            "part_voxel",
            str(
                part_paths.get("part_voxel")
                or f"reconstruction/voxel_expanded/{obj_id}/angle_{angle_idx}/64/ind_{part_name}.npy"
            ),
            part_name,
        )


def _iter_image_files(rec: dict[str, Any]) -> Iterable[tuple[str, str, str | None]]:
    for image_path in rec.get("image_paths") or []:
        yield ("image", str(image_path), None)


def _shape_issue(path: Path, kind: str) -> str | None:
    import numpy as np

    try:
        if kind == "global_latent":
            with np.load(path, allow_pickle=False) as data:
                if "mean" not in data.files:
                    return "global_latent missing key 'mean'"
                arr = data["mean"]
        elif kind == "dino_tokens":
            with np.load(path, allow_pickle=False) as data:
                if "tokens" not in data.files:
                    return "dino_tokens missing key 'tokens'"
                arr = data["tokens"]
                if arr.ndim != 3 or arr.shape[-1] != 1024:
                    return f"dino_tokens expected [V,T,1024], got {tuple(arr.shape)}"
                return None
        else:
            arr = np.load(path, allow_pickle=False)
    except Exception as exc:  # noqa: BLE001 - report exact data-load failure.
        return f"load failed: {exc}"

    if kind in ("global_latent", "part_latent") and tuple(arr.shape) != (8, 16, 16, 16):
        return f"{kind} expected (8,16,16,16), got {tuple(arr.shape)}"
    if kind in ("surface_voxel", "part_voxel", "mask") and arr.size == 0:
        return f"{kind} is empty"
    if kind in ("surface_voxel", "part_voxel") and (arr.ndim != 2 or arr.shape[-1] != 3):
        return f"{kind} expected [N,3], got {tuple(arr.shape)}"
    return None


def _make_issue(
    *,
    line_no: int,
    rec: dict[str, Any],
    kind: str,
    path: str | None,
    reason: str,
    part_name: str | None = None,
) -> dict[str, Any]:
    return {
        "line_no": line_no,
        "sample_id": rec.get("sample_id"),
        "object_id": rec.get("object_id"),
        "angle_idx": rec.get("angle_idx"),
        "view_idx": rec.get("view_idx"),
        "view_indices": rec.get("view_indices"),
        "part_name": part_name,
        "kind": kind,
        "path": path,
        "reason": reason,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check single-view Part SS latent flow manifest file integrity."
    )
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--manifest-path", required=True, type=Path)
    parser.add_argument("--report-jsonl", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=0, help="Check only the first N non-empty rows.")
    parser.add_argument("--progress-every", type=int, default=10000)
    parser.add_argument("--max-target-parts", type=int, default=20)
    parser.add_argument(
        "--skip-over-max-parts",
        action="store_true",
        help="Skip rows with K > --max-target-parts, matching current training filter.",
    )
    parser.add_argument(
        "--check-images",
        action="store_true",
        help="Also check image_paths. Off by default because training does not read RGB files.",
    )
    parser.add_argument(
        "--check-shapes",
        action="store_true",
        help="Load arrays and validate basic shapes. Slow for full datasets with DINO tokens.",
    )
    parser.add_argument("--max-issues-print", type=int, default=20)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    data_root = args.data_root
    manifest_abs = args.manifest_path
    if not manifest_abs.is_absolute():
        manifest_abs = data_root / manifest_abs
    if not manifest_abs.is_file():
        print(f"ERROR: manifest not found: {manifest_abs}", file=sys.stderr)
        return 2

    exists_cache: dict[Path, bool] = {}
    kind_counts: Counter[str] = Counter()
    missing_counts: Counter[str] = Counter()
    shape_counts: Counter[str] = Counter()
    schema_counts: Counter[str] = Counter()
    seen_k: Counter[str] = Counter()
    issues_printed = 0
    checked_rows = 0
    skipped_over_k = 0
    total_parts = 0

    report_handle = None
    if args.report_jsonl is not None:
        args.report_jsonl.parent.mkdir(parents=True, exist_ok=True)
        report_handle = args.report_jsonl.open("w", encoding="utf-8")

    try:
        for line_no, rec in _read_jsonl(manifest_abs):
            if args.limit and checked_rows >= args.limit:
                break
            checked_rows += 1
            if "__json_error__" in rec:
                issue = _make_issue(
                    line_no=line_no,
                    rec=rec,
                    kind="json",
                    path=str(manifest_abs),
                    reason=rec["__json_error__"],
                )
                schema_counts["json"] += 1
                if report_handle:
                    report_handle.write(json.dumps(issue, ensure_ascii=False) + "\n")
                continue

            missing_fields = [field for field in REQUIRED_ROW_FIELDS if field not in rec]
            if missing_fields:
                schema_counts["required_fields"] += 1
                issue = _make_issue(
                    line_no=line_no,
                    rec=rec,
                    kind="schema",
                    path=str(manifest_abs),
                    reason=f"missing fields: {missing_fields}",
                )
                if report_handle:
                    report_handle.write(json.dumps(issue, ensure_ascii=False) + "\n")
                continue

            k = len(_target_names(rec))
            total_parts += k
            if k <= 1:
                seen_k["1"] += 1
            elif k <= 2:
                seen_k["2"] += 1
            elif k <= 4:
                seen_k["3-4"] += 1
            elif k <= 8:
                seen_k["5-8"] += 1
            elif k <= 16:
                seen_k["9-16"] += 1
            elif k <= args.max_target_parts:
                seen_k[f"17-{args.max_target_parts}"] += 1
            else:
                seen_k[f">{args.max_target_parts}"] += 1

            if args.skip_over_max_parts and args.max_target_parts > 0 and k > args.max_target_parts:
                skipped_over_k += 1
                continue

            required = list(_iter_required_files(rec))
            if args.check_images:
                required.extend(_iter_image_files(rec))

            for kind, rel_path, part_name in required:
                kind_counts[kind] += 1
                path = _resolve(data_root, rel_path)
                exists = exists_cache.get(path)
                if exists is None:
                    exists = path.is_file()
                    exists_cache[path] = exists
                if not exists:
                    missing_counts[kind] += 1
                    issue = _make_issue(
                        line_no=line_no,
                        rec=rec,
                        kind=kind,
                        path=str(path),
                        reason="missing",
                        part_name=part_name,
                    )
                    if issues_printed < args.max_issues_print:
                        print(
                            "[missing] "
                            f"line={line_no} sample={rec.get('sample_id')} "
                            f"kind={kind} part={part_name} path={path}"
                        )
                        issues_printed += 1
                    if report_handle:
                        report_handle.write(json.dumps(issue, ensure_ascii=False) + "\n")
                    continue

                if args.check_shapes:
                    shape_issue = _shape_issue(path, kind)
                    if shape_issue:
                        shape_counts[kind] += 1
                        issue = _make_issue(
                            line_no=line_no,
                            rec=rec,
                            kind=kind,
                            path=str(path),
                            reason=shape_issue,
                            part_name=part_name,
                        )
                        if issues_printed < args.max_issues_print:
                            print(
                                "[shape] "
                                f"line={line_no} sample={rec.get('sample_id')} "
                                f"kind={kind} part={part_name} reason={shape_issue}"
                            )
                            issues_printed += 1
                        if report_handle:
                            report_handle.write(json.dumps(issue, ensure_ascii=False) + "\n")

            if args.progress_every > 0 and checked_rows % args.progress_every == 0:
                print(
                    f"[progress] rows={checked_rows} missing={sum(missing_counts.values())} "
                    f"shape_issues={sum(shape_counts.values())}"
                )
    finally:
        if report_handle:
            report_handle.close()

    total_missing = sum(missing_counts.values())
    total_shape_issues = sum(shape_counts.values())
    total_schema_issues = sum(schema_counts.values())

    print("\n[summary]")
    print(f"manifest: {manifest_abs}")
    print(f"data_root: {data_root}")
    print(f"rows_checked: {checked_rows}")
    print(f"target_parts_seen: {total_parts}")
    print(f"skipped_over_max_parts: {skipped_over_k}")
    print(f"unique_paths_checked: {len(exists_cache)}")
    print(f"schema_issues: {total_schema_issues}")
    print(f"missing_files: {total_missing}")
    print(f"shape_issues: {total_shape_issues}")

    print("\n[K histogram]")
    for key in ("1", "2", "3-4", "5-8", "9-16", f"17-{args.max_target_parts}", f">{args.max_target_parts}"):
        print(f"{key:>8}: {seen_k[key]}")

    print("\n[checked files by kind]")
    for key in sorted(kind_counts):
        print(f"{key:>14}: checked={kind_counts[key]} missing={missing_counts[key]} shape_issues={shape_counts[key]}")

    if schema_counts:
        print("\n[schema issues]")
        for key, count in schema_counts.most_common():
            print(f"{key:>14}: {count}")

    if args.report_jsonl:
        print(f"\nreport_jsonl: {args.report_jsonl}")

    return 1 if total_missing or total_shape_issues or total_schema_issues else 0


if __name__ == "__main__":
    raise SystemExit(main())
