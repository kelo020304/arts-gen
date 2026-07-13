#!/usr/bin/env python3
"""Extract DINOv2 token features from current render assets.

Default Step 06 follows the current mainline render product:

    renders/<object_id>/angle_<i>/part_complete/rgb/view_0.png ... view_15.png

and writes:

    reconstruction/dinov2_tokens/<object_id>/angle_<i>/part_complete/tokens.npz

Optional 150-view render sets are supported explicitly and use sibling branches:

    reconstruction/dinov2_tokens/<object_id>/angle_<i>/full_object/tokens.npz
    reconstruction/dinov2_tokens/<object_id>/angle_<i>/valid_parts/<part_key>/tokens.npz

They are not produced by Step 05 unless requested.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "utils"))

from config_loader import load_config


ANGLE_DIR_RE = re.compile(r"angle_(\d+)$")
VIEW_FILE_RE = re.compile(r"view_(\d+)\.png$")
INDEXED_FILE_RE = re.compile(r"(\d+)\.png$")

DINO_INPUT_RESOLUTION = 518
PATCH_TOKEN_COUNT = 1369
TOKEN_COUNT = 1 + PATCH_TOKEN_COUNT
TOKEN_DIM = 1024
DEFAULT_BATCH_SIZE = 16
DEFAULT_SET = "part_complete"
DINO_CHECKPOINT_BY_MODEL = {
    "dinov2_vitl14_reg": "dinov2_vitl14_reg4_pretrain.pth",
}

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


@dataclass(frozen=True)
class RenderSetSpec:
    name: str
    input_subdir: str
    output_subdir: str | None
    expected_views: int
    file_pattern: re.Pattern[str]
    rgb_leaf: str | None = None
    per_part: bool = False


RENDER_SETS: dict[str, RenderSetSpec] = {
    "part_complete": RenderSetSpec(
        name="part_complete",
        input_subdir="part_complete",
        rgb_leaf="rgb",
        output_subdir="part_complete",
        expected_views=16,
        file_pattern=VIEW_FILE_RE,
    ),
    # Historical quadrant layout, retained only for explicit back-compat checks.
    "legacy_quadrant": RenderSetSpec(
        name="legacy_quadrant",
        input_subdir="",
        rgb_leaf="rgb",
        output_subdir="legacy_quadrant",
        expected_views=12,
        file_pattern=VIEW_FILE_RE,
    ),
    # Optional 150-view branches. Step 05 does not run these by default.
    "full_object_all_views": RenderSetSpec(
        name="full_object_all_views",
        input_subdir="render_full_obj_all_view",
        rgb_leaf=None,
        output_subdir="full_object",
        expected_views=150,
        file_pattern=INDEXED_FILE_RE,
    ),
    "valid_parts_all_views": RenderSetSpec(
        name="valid_parts_all_views",
        input_subdir="render_part_all_view",
        rgb_leaf=None,
        output_subdir="valid_parts",
        expected_views=150,
        file_pattern=INDEXED_FILE_RE,
        per_part=True,
    ),
}

ALIASES = {
    "part_complete_16": "part_complete",
    "full150": "full_object_all_views",
    "full_object_150": "full_object_all_views",
    "parts150": "valid_parts_all_views",
    "valid_part_150": "valid_parts_all_views",
    "legacy": "legacy_quadrant",
    "quadrant": "legacy_quadrant",
}


@dataclass(frozen=True)
class FeatureJob:
    render_set: str
    object_id: str
    angle_name: str
    source_dir: Path
    output_path: Path
    view_paths: tuple[Path, ...]
    part_key: str | None = None


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract DINOv2 token features from rendered RGB images. Default: part_complete 16-view RGB."
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to the dataset toolkit YAML config.",
    )
    parser.add_argument(
        "--object-ids",
        help="Optional comma-separated object ID subset, e.g. 100064,100283",
    )
    parser.add_argument(
        "--sets",
        default=DEFAULT_SET,
        help=(
            "Comma-separated render sets. Canonical: part_complete, full_object_all_views, "
            "valid_parts_all_views, legacy_quadrant. Aliases: part_complete_16, full150, "
            "parts150, legacy. Default: part_complete."
        ),
    )
    parser.add_argument(
        "--angle-ids",
        help="Optional comma-separated angle subset, e.g. 0,3,7.",
    )
    parser.add_argument(
        "--part-keys",
        help="Optional comma-separated part key subset for valid_parts_all_views.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help=f"Image batch size for DINOv2 forward passes. Default: {DEFAULT_BATCH_SIZE}.",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help="Torch device for DINOv2 extraction. Default is strict cuda; pass cpu explicitly for debugging.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Regenerate existing token files.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Enumerate jobs without loading DINOv2 or writing token files.",
    )
    return parser.parse_args(argv)


def _parse_csv(raw_value: str) -> list[str]:
    values = [item.strip() for item in raw_value.split(",")]
    if not values or any(not item for item in values):
        raise ValueError("comma-separated value must contain non-empty items")
    if len(values) != len(set(values)):
        raise ValueError("comma-separated value contains duplicates")
    return values


def _parse_sets(raw_value: str) -> list[str]:
    parsed: list[str] = []
    for item in _parse_csv(raw_value):
        canonical = ALIASES.get(item, item)
        if canonical not in RENDER_SETS:
            valid = sorted(set(RENDER_SETS) | set(ALIASES))
            raise ValueError(f"Unknown --sets item {item!r}. Valid values: {valid}")
        if canonical not in parsed:
            parsed.append(canonical)
    return parsed


def _parse_angle_ids(raw_value: str | None) -> set[int] | None:
    if raw_value is None:
        return None
    angle_ids: set[int] = set()
    for item in _parse_csv(raw_value):
        if not item.isdigit():
            raise ValueError(f"--angle-ids must contain non-negative integers, got {item!r}")
        angle_ids.add(int(item))
    return angle_ids


def _resolve_object_ids(config: Any, object_ids_arg: str | None) -> list[str]:
    available_object_ids = config.list_object_ids()
    if object_ids_arg is None:
        return available_object_ids

    requested_object_ids = _parse_csv(object_ids_arg)
    available_set = set(available_object_ids)
    missing_object_ids = [
        object_id for object_id in requested_object_ids if object_id not in available_set
    ]
    if missing_object_ids:
        missing_text = ", ".join(missing_object_ids)
        raise ValueError(f"Unknown object IDs in --object-ids: {missing_text}")
    return requested_object_ids


def _list_angle_dirs(
    object_dir: Path,
    expected_num_angles: int,
    *,
    require_all: bool = False,
) -> list[Path]:
    if not object_dir.is_dir():
        if not require_all:
            return []
        raise FileNotFoundError(f"Object render directory not found: {object_dir}")

    angle_dirs_by_idx: dict[int, Path] = {}
    for path in object_dir.iterdir():
        if not path.is_dir():
            continue
        match = ANGLE_DIR_RE.fullmatch(path.name)
        if match is None:
            continue
        angle_idx = int(match.group(1))
        if angle_idx in angle_dirs_by_idx:
            raise ValueError(f"Duplicate angle directory index {angle_idx} in: {object_dir}")
        angle_dirs_by_idx[angle_idx] = path

    expected_indices = list(range(expected_num_angles))
    missing_indices = [idx for idx in expected_indices if idx not in angle_dirs_by_idx]
    if require_all and missing_indices:
        missing_text = ", ".join(f"angle_{idx}" for idx in missing_indices)
        raise FileNotFoundError(f"Missing angle directories in {object_dir}: {missing_text}")

    unexpected_indices = sorted(set(angle_dirs_by_idx) - set(expected_indices))
    if unexpected_indices:
        unexpected_text = ", ".join(f"angle_{idx}" for idx in unexpected_indices)
        raise ValueError(f"Unexpected angle directories in {object_dir}: {unexpected_text}")

    return [angle_dirs_by_idx[idx] for idx in expected_indices if idx in angle_dirs_by_idx]


def _list_view_paths(rgb_dir: Path, expected_num_views: int, pattern: re.Pattern[str]) -> list[Path]:
    if not rgb_dir.is_dir():
        raise FileNotFoundError(f"RGB directory not found: {rgb_dir}")

    view_paths_by_idx: dict[int, Path] = {}
    for path in rgb_dir.iterdir():
        if not path.is_file():
            continue
        match = pattern.fullmatch(path.name)
        if match is None:
            continue
        view_idx = int(match.group(1))
        if view_idx in view_paths_by_idx:
            raise ValueError(f"Duplicate RGB view index {view_idx} in: {rgb_dir}")
        view_paths_by_idx[view_idx] = path

    expected_range = set(range(expected_num_views))
    missing_indices = [idx for idx in range(expected_num_views) if idx not in view_paths_by_idx]
    if missing_indices:
        if pattern is VIEW_FILE_RE:
            missing_text = ", ".join(f"view_{idx}.png" for idx in missing_indices)
        else:
            missing_text = ", ".join(f"{idx:03d}.png" for idx in missing_indices)
        raise FileNotFoundError(f"Missing RGB files in {rgb_dir}: {missing_text}")

    unexpected_indices = sorted(set(view_paths_by_idx) - expected_range)
    if unexpected_indices:
        unexpected_text = ", ".join(str(idx) for idx in unexpected_indices)
        raise ValueError(f"Unexpected RGB view indices in {rgb_dir}: {unexpected_text}")

    return [view_paths_by_idx[idx] for idx in range(expected_num_views)]


def _source_dir(angle_dir: Path, spec: RenderSetSpec) -> Path:
    base = angle_dir / spec.input_subdir if spec.input_subdir else angle_dir
    return base / spec.rgb_leaf if spec.rgb_leaf else base


def _output_path(output_root: Path, spec: RenderSetSpec, object_id: str, angle_name: str, part_key: str | None = None) -> Path:
    parts = [object_id, angle_name, spec.output_subdir or spec.name]
    if part_key is not None:
        parts.append(part_key)
    return output_root.joinpath(*parts) / "tokens.npz"


def _build_jobs_for_angle(
    *,
    output_root: Path,
    spec: RenderSetSpec,
    object_id: str,
    angle_dir: Path,
    part_keys: set[str] | None,
) -> list[FeatureJob]:
    if not spec.per_part:
        source_dir = _source_dir(angle_dir, spec)
        view_paths = _list_view_paths(source_dir, spec.expected_views, spec.file_pattern)
        return [
            FeatureJob(
                render_set=spec.name,
                object_id=object_id,
                angle_name=angle_dir.name,
                source_dir=source_dir,
                output_path=_output_path(output_root, spec, object_id, angle_dir.name),
                view_paths=tuple(view_paths),
            )
        ]

    parts_root = angle_dir / spec.input_subdir
    if not parts_root.is_dir():
        raise FileNotFoundError(f"Part render directory not found: {parts_root}")
    jobs: list[FeatureJob] = []
    for part_dir in sorted(path for path in parts_root.iterdir() if path.is_dir()):
        part_key = part_dir.name
        if part_keys is not None and part_key not in part_keys:
            continue
        view_paths = _list_view_paths(part_dir, spec.expected_views, spec.file_pattern)
        jobs.append(
            FeatureJob(
                render_set=spec.name,
                object_id=object_id,
                angle_name=angle_dir.name,
                part_key=part_key,
                source_dir=part_dir,
                output_path=_output_path(output_root, spec, object_id, angle_dir.name, part_key),
                view_paths=tuple(view_paths),
            )
        )
    if part_keys is not None:
        found = {job.part_key for job in jobs}
        missing = sorted(part_keys - {key for key in found if key is not None})
        if missing:
            raise FileNotFoundError(f"Requested --part-keys not found under {parts_root}: {missing}")
    return jobs


def _build_transform() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize(
                (DINO_INPUT_RESOLUTION, DINO_INPUT_RESOLUTION),
                interpolation=transforms.InterpolationMode.BICUBIC,
            ),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def _load_image_as_white_rgb(image_path: Path, transform: transforms.Compose) -> torch.Tensor:
    with Image.open(image_path) as image:
        has_alpha = image.mode in {"RGBA", "LA"} or (
            image.mode == "P" and "transparency" in image.info
        )
        if has_alpha:
            image = image.convert("RGBA")
            white_background = Image.new("RGBA", image.size, (255, 255, 255, 255))
            image = Image.alpha_composite(white_background, image).convert("RGB")
        else:
            image = image.convert("RGB")
    return transform(image)


def _require_dir(path: str | Path, label: str) -> Path:
    resolved = Path(path).expanduser()
    if not resolved.is_absolute():
        raise ValueError(f"{label} must be an absolute local path, got: {path}")
    if not resolved.is_dir():
        raise FileNotFoundError(f"{label} directory does not exist: {resolved}")
    return resolved


def _require_dinov2_checkpoint(torch_hub_dir: Path, model_name: str) -> Path:
    if model_name not in DINO_CHECKPOINT_BY_MODEL:
        raise ValueError(
            f"Unsupported DINOv2 model {model_name!r}. "
            f"Supported models: {sorted(DINO_CHECKPOINT_BY_MODEL)}"
        )
    checkpoint = torch_hub_dir / "checkpoints" / DINO_CHECKPOINT_BY_MODEL[model_name]
    if not checkpoint.is_file():
        raise FileNotFoundError(
            f"DINOv2 checkpoint missing for feature.model={model_name!r}: {checkpoint}"
        )
    return checkpoint


def _load_model(model_name: str, dinov2_repo: str, torch_hub_dir: str, device: torch.device) -> torch.nn.Module:
    repo_path = _require_dir(dinov2_repo, "feature.dinov2_repo")
    hub_dir = _require_dir(torch_hub_dir, "feature.torch_hub_dir")
    _require_dinov2_checkpoint(hub_dir, model_name)
    torch.hub.set_dir(str(hub_dir))
    model = torch.hub.load(str(repo_path), model_name, source="local")
    model.eval()
    model.to(device)
    return model


def _encode_views(
    model: torch.nn.Module,
    view_paths: tuple[Path, ...],
    transform: transforms.Compose,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    if batch_size < 1:
        raise ValueError("--batch-size must be >= 1")

    chunks: list[np.ndarray] = []
    for start in range(0, len(view_paths), batch_size):
        batch_paths = view_paths[start : start + batch_size]
        image_tensors = [_load_image_as_white_rgb(path, transform) for path in batch_paths]
        batch = torch.stack(image_tensors, dim=0).to(device)

        with torch.inference_mode():
            features = model.forward_features(batch)
            patch_tokens = features["x_norm_patchtokens"]
            cls_token = features["x_norm_clstoken"].unsqueeze(1)
            tokens = torch.cat([cls_token, patch_tokens], dim=1)

        expected_tail = (len(batch_paths), TOKEN_COUNT, TOKEN_DIM)
        if tuple(tokens.shape) != expected_tail:
            raise ValueError(
                f"Unexpected token batch shape {tuple(tokens.shape)}, expected {expected_tail}"
            )
        chunks.append(tokens.cpu().to(torch.float32).numpy())

    encoded = np.concatenate(chunks, axis=0)
    expected_shape = (len(view_paths), TOKEN_COUNT, TOKEN_DIM)
    if tuple(encoded.shape) != expected_shape:
        raise ValueError(f"Unexpected token shape {encoded.shape}, expected {expected_shape}")
    return encoded


def _write_meta(job: FeatureJob, tokens_shape: tuple[int, ...]) -> None:
    meta = {
        "schema_version": "v1-dinov2-render-tokens",
        "render_set": job.render_set,
        "object_id": job.object_id,
        "angle": job.angle_name,
        "part_key": job.part_key,
        "source_dir": str(job.source_dir),
        "source_images": [str(path) for path in job.view_paths],
        "tokens_path": str(job.output_path),
        "tokens_shape": list(tokens_shape),
        "model_tokens": {
            "input_resolution": DINO_INPUT_RESOLUTION,
            "token_count": TOKEN_COUNT,
            "token_dim": TOKEN_DIM,
        },
    }
    meta_path = job.output_path.parent / f"{job.output_path.stem}_npz_meta.json"
    meta_path.write_text(
        json.dumps(meta, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = load_config(args.config)
    render_sets = _parse_sets(args.sets)
    requested_angle_ids = _parse_angle_ids(args.angle_ids)
    part_keys = set(_parse_csv(args.part_keys)) if args.part_keys else None

    renders_dir = Path(config.renders_dir)
    if not renders_dir.is_dir():
        raise FileNotFoundError(f"renders directory not found: {renders_dir}")

    output_root = Path(config.reconstruction_dir) / "dinov2_tokens"
    object_ids = _resolve_object_ids(config, args.object_ids)

    jobs: list[FeatureJob] = []
    skipped_no_render_dir = 0
    skipped_missing_angle_dirs = 0
    for object_id in object_ids:
        object_render_dir = renders_dir / object_id
        object_num_angles = config.get_num_angles(object_id)
        if not object_render_dir.is_dir() and requested_angle_ids is None:
            skipped_no_render_dir += 1
        angle_dirs = _list_angle_dirs(
            object_render_dir,
            object_num_angles,
            require_all=requested_angle_ids is not None,
        )
        if requested_angle_ids is None:
            skipped_missing_angle_dirs += object_num_angles - len(angle_dirs)
        if requested_angle_ids is not None:
            by_index = {int(ANGLE_DIR_RE.fullmatch(path.name).group(1)): path for path in angle_dirs}
            missing_angles = sorted(requested_angle_ids - set(by_index))
            if missing_angles:
                raise ValueError(f"Object {object_id} does not have requested --angle-ids: {missing_angles}")
            angle_dirs = [by_index[idx] for idx in sorted(requested_angle_ids)]
        for angle_dir in angle_dirs:
            for render_set in render_sets:
                spec = RENDER_SETS[render_set]
                jobs.extend(
                    _build_jobs_for_angle(
                        output_root=output_root,
                        spec=spec,
                        object_id=object_id,
                        angle_dir=angle_dir,
                        part_keys=part_keys if spec.per_part else None,
                    )
                )

    pending_jobs = [job for job in jobs if args.overwrite or not job.output_path.is_file()]
    skipped_existing = len(jobs) - len(pending_jobs)
    print(
        f"Prepared DINOv2 jobs: total={len(jobs)} pending={len(pending_jobs)} "
        f"skipped_existing={skipped_existing} "
        f"skipped_no_render_dir={skipped_no_render_dir} "
        f"skipped_missing_angle_dirs={skipped_missing_angle_dirs} "
        f"sets={','.join(render_sets)}",
        flush=True,
    )
    if args.dry_run:
        for job in pending_jobs[:20]:
            part = f" part={job.part_key}" if job.part_key else ""
            print(
                f"[dry-run] {job.render_set} {job.object_id}/{job.angle_name}{part}: "
                f"{len(job.view_paths)} views -> {job.output_path}",
                flush=True,
            )
        if len(pending_jobs) > 20:
            print(f"[dry-run] ... {len(pending_jobs) - 20} more jobs", flush=True)
        return 0

    transform = _build_transform()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested, but torch.cuda.is_available() is false")
    model = _load_model(
        config.feature.model,
        config.feature.dinov2_repo,
        config.feature.torch_hub_dir,
        device,
    )

    for job_index, job in enumerate(pending_jobs, start=1):
        part = f" part={job.part_key}" if job.part_key else ""
        print(
            f"[{job_index}/{len(pending_jobs)}] {job.render_set} "
            f"{job.object_id}/{job.angle_name}{part} ({len(job.view_paths)} views)",
            flush=True,
        )
        tokens = _encode_views(model, job.view_paths, transform, device, args.batch_size)
        job.output_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(job.output_path, tokens=tokens)
        _write_meta(job, tuple(int(dim) for dim in tokens.shape))

    print("DINOv2 feature extraction done", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
