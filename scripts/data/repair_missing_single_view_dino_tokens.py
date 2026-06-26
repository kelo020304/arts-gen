#!/usr/bin/env python3
"""Regenerate missing single-view part_complete DINO token files.

Input is the JSONL report produced by
``scripts/data/check_part_ss_single_view_integrity.py``. The script selects
missing ``kind=dino_tokens`` records, de-duplicates them by object/angle, and
directly extracts DINO tokens from ``renders/.../part_complete/rgb``. It does
not load the full dataset toolkit config, so repair does not require
``raw/finaljson`` or a valid Blender path.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any


def _read_missing_pairs(report_jsonl: Path) -> list[tuple[str, int]]:
    pairs: set[tuple[str, int]] = set()
    with report_jsonl.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            rec = json.loads(line)
            if rec.get("kind") != "dino_tokens" or rec.get("reason") != "missing":
                continue
            object_id = rec.get("object_id")
            angle_idx = rec.get("angle_idx")
            if object_id is None or angle_idx is None:
                raise ValueError(
                    f"{report_jsonl}:{line_no} missing object_id/angle_idx in dino record"
                )
            pairs.add((str(object_id), int(angle_idx)))
    return sorted(pairs, key=lambda item: (item[0], item[1]))


def _load_feature_module(feature_script: Path) -> Any:
    spec = importlib.util.spec_from_file_location("part_ss_repair_extract_feature", feature_script)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import feature script: {feature_script}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _view_paths(data_root: Path, object_id: str, angle_idx: int) -> tuple[Path, ...]:
    rgb_dir = data_root / "renders" / object_id / f"angle_{angle_idx}" / "part_complete" / "rgb"
    return tuple(rgb_dir / f"view_{idx}.png" for idx in range(16))


def _output_path(data_root: Path, object_id: str, angle_idx: int) -> Path:
    return (
        data_root
        / "reconstruction"
        / "dinov2_tokens"
        / object_id
        / f"angle_{angle_idx}"
        / "part_complete"
        / "tokens.npz"
    )


def _meta_path(tokens_path: Path) -> Path:
    return tokens_path.parent / f"{tokens_path.stem}_npz_meta.json"


def _write_meta(
    *,
    object_id: str,
    angle_idx: int,
    view_paths: tuple[Path, ...],
    output_path: Path,
    tokens_shape: tuple[int, ...],
) -> None:
    meta = {
        "schema_version": "v1-dinov2-render-tokens",
        "render_set": "part_complete",
        "object_id": object_id,
        "angle": f"angle_{angle_idx}",
        "part_key": None,
        "source_dir": str(view_paths[0].parent),
        "source_images": [str(path) for path in view_paths],
        "tokens_path": str(output_path),
        "tokens_shape": list(tokens_shape),
        "model_tokens": {
            "input_resolution": 518,
            "token_count": 1370,
            "token_dim": 1024,
        },
    }
    _meta_path(output_path).write_text(
        json.dumps(meta, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Regenerate missing part_complete DINO token files from an integrity report."
    )
    parser.add_argument("--report-jsonl", required=True, type=Path)
    parser.add_argument(
        "--toolkit-root",
        type=Path,
        default=Path("submodules/dataset_toolkits_single_image"),
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/PhysX-Mobility.yaml"),
        help="Deprecated; ignored by direct repair mode.",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=None,
        required=True,
        help="Dataset root containing renders/ and reconstruction/.",
    )
    parser.add_argument("--model", default="dinov2_vitl14_reg")
    parser.add_argument("--dinov2-repo", type=Path, default=Path("pretrained/dinov2"))
    parser.add_argument("--torch-hub-dir", type=Path, default=Path("pretrained/torch_hub"))
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--limit", type=int, default=0, help="Repair only first N unique object/angle pairs.")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--keep-going", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    toolkit_root = args.toolkit_root.resolve()
    repo_root = toolkit_root.parent.parent
    feature_script = toolkit_root / "pipeline" / "06_extract_feature.py"
    data_root = args.data_root.resolve()
    dinov2_repo = args.dinov2_repo if args.dinov2_repo.is_absolute() else repo_root / args.dinov2_repo
    torch_hub_dir = args.torch_hub_dir if args.torch_hub_dir.is_absolute() else repo_root / args.torch_hub_dir

    if not args.report_jsonl.is_file():
        print(f"ERROR: report_jsonl not found: {args.report_jsonl}", file=sys.stderr)
        return 2
    if not feature_script.is_file():
        print(f"ERROR: feature script not found: {feature_script}", file=sys.stderr)
        return 2
    if not data_root.is_dir():
        print(f"ERROR: data_root not found: {data_root}", file=sys.stderr)
        return 2

    pairs = _read_missing_pairs(args.report_jsonl)
    if args.limit:
        pairs = pairs[: args.limit]

    print(f"[repair] unique missing dino object/angle pairs: {len(pairs)}")
    for object_id, angle_idx in pairs[:20]:
        print(f"  {object_id} angle_{angle_idx}")
    if len(pairs) > 20:
        print(f"  ... {len(pairs) - 20} more")
    if not pairs:
        return 0

    pending: list[tuple[str, int, tuple[Path, ...], Path]] = []
    failures: list[tuple[str, int, str]] = []
    for object_id, angle_idx in pairs:
        paths = _view_paths(data_root, object_id, angle_idx)
        missing_views = [path for path in paths if not path.is_file()]
        output = _output_path(data_root, object_id, angle_idx)
        if output.is_file() and not args.overwrite:
            continue
        if missing_views:
            failures.append(
                (
                    object_id,
                    angle_idx,
                    "missing RGB views: " + ", ".join(str(path) for path in missing_views[:4]),
                )
            )
            if not args.keep_going:
                break
            continue
        pending.append((object_id, angle_idx, paths, output))

    print(f"[repair] pending token files: {len(pending)}")
    for index, (object_id, angle_idx, paths, output) in enumerate(pending, 1):
        print(
            f"[{index}/{len(pending)}] {object_id}/angle_{angle_idx}: "
            f"{paths[0].parent} -> {output}",
            flush=True,
        )
        if args.dry_run:
            continue

    if args.dry_run:
        if failures:
            print("\n[repair preflight failures]", file=sys.stderr)
            for object_id, angle_idx, reason in failures:
                print(f"  object_id={object_id} angle_{angle_idx}: {reason}", file=sys.stderr)
            return 1
        print("[repair] done")
        return 0

    if pending:
        feature_module = _load_feature_module(feature_script)
        transform = feature_module._build_transform()
        import torch
        import numpy as np

        device = torch.device(args.device)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("--device cuda requested, but torch.cuda.is_available() is false")
        model = feature_module._load_model(
            args.model,
            str(dinov2_repo),
            str(torch_hub_dir),
            device,
        )

        for index, (object_id, angle_idx, paths, output) in enumerate(pending, 1):
            print(
                f"[encode {index}/{len(pending)}] {object_id}/angle_{angle_idx} "
                f"({len(paths)} views)",
                flush=True,
            )
            try:
                tokens = feature_module._encode_views(
                    model,
                    paths,
                    transform,
                    device,
                    args.batch_size,
                )
                output.parent.mkdir(parents=True, exist_ok=True)
                np.savez(output, tokens=tokens)
                _write_meta(
                    object_id=object_id,
                    angle_idx=angle_idx,
                    view_paths=paths,
                    output_path=output,
                    tokens_shape=tuple(int(dim) for dim in tokens.shape),
                )
            except Exception as exc:  # noqa: BLE001 - continue mode needs exact failures.
                failures.append((object_id, angle_idx, str(exc)))
                if not args.keep_going:
                    break

    if failures:
        print("\n[repair failures]", file=sys.stderr)
        for object_id, angle_idx, reason in failures:
            print(f"  object_id={object_id} angle_{angle_idx}: {reason}", file=sys.stderr)
        return 1

    print("[repair] done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
