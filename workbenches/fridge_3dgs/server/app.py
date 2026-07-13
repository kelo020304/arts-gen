#!/usr/bin/env python3
from __future__ import annotations

import base64
import io
import json
import os
import signal
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from pathlib import Path
from typing import Any

import numpy as np
import requests
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, ImageOps
from pydantic import BaseModel, Field


REPO_ROOT = Path(__file__).resolve().parents[3]
WORKBENCH_ROOT = Path(__file__).resolve().parents[1]
STATIC_ROOT = WORKBENCH_ROOT / "static"
DATA_ROOT = Path(os.environ.get("ARTS_GEN_DATA_ROOT", "/robot/data-lab/jzh/art-gen"))
WORK_ROOT = Path(
    os.environ.get(
        "FRIDGE_3DGS_WORK_ROOT",
        str(DATA_ROOT / "workbench/fridge_3dgs"),
    )
)
SESSION_ID = os.environ.get("FRIDGE_3DGS_SESSION", "fridge_point_cloud")
POINT_CLOUD = Path(os.environ.get("FRIDGE_3DGS_POINT_CLOUD", str(REPO_ROOT / "data/point_cloud.ply")))
SAM3_ROOT = Path(os.environ.get("SAM3_ROOT", str(DATA_ROOT / "local-deps/sam3")))
DEFAULT_SAM3_PT = DATA_ROOT / "weights/sam3/sam3.pt"
DEFAULT_SAM3_SAFETENSORS = DATA_ROOT / "weights/sam3/sam3_1038lab_f055b060_sam3.safetensors"
SAM3_CKPT = Path(os.environ.get("SAM3_CKPT", str(DEFAULT_SAM3_PT if DEFAULT_SAM3_PT.is_file() else DEFAULT_SAM3_SAFETENSORS)))
SAM3_PORT = int(os.environ.get("FRIDGE_3DGS_SAM3_PORT", "8787"))
DEFAULT_SS_FLOW_CKPT = DATA_ROOT / "ckpt/tre-ss-flow/tre-ss-concat-0616-1/ckpts/denoiser_ema0.999_step0012500.pt"
DEFAULT_PART_SEG_CKPT = DATA_ROOT / "ckpt/part-prompt-seg/part-prompt-seg-L-0709-1-joint/ckpts/step_100000.pt"
DEFAULT_DATASET_CONFIG = DATA_ROOT / "data/part_promptable_seg_manifests/v6/split_official_verse_realappliance_0511dd_v6.json"
DATASET_CONFIG = Path(os.environ.get("FRIDGE_3DGS_DATA_CONFIG", str(DEFAULT_DATASET_CONFIG)))
LEGACY_PART_SEG_CKPT = DATA_ROOT / "ckpts/part-prompt-seg/part_promptable_seg_full_S_0618-1/ckpts/step_100000.pt"
TRELLIS_CKPT_ROOT = DATA_ROOT / "pretrained/TRELLIS-image-large/ckpts"
TRELLIS_THIRD_PARTY_CKPT_ROOT = DATA_ROOT / "third-party-weights/trellis/pretrained/TRELLIS-image-large/ckpts"
DEFAULT_SS_DECODER_CKPT = (
    TRELLIS_THIRD_PARTY_CKPT_ROOT / "ss_dec_conv3d_16l8_fp16.safetensors"
    if (TRELLIS_THIRD_PARTY_CKPT_ROOT / "ss_dec_conv3d_16l8_fp16.safetensors").is_file()
    else TRELLIS_CKPT_ROOT / "ss_dec_conv3d_16l8_fp16.safetensors"
)
DEFAULT_SLAT_FLOW_CKPT = TRELLIS_CKPT_ROOT / "slat_flow_img_dit_L_64l8p2_fp16.safetensors"
DEFAULT_SLAT_MESH_DECODER_CKPT = TRELLIS_CKPT_ROOT / "slat_dec_mesh_swin8_B_64l8m256c_fp16.safetensors"
DEFAULT_SLAT_GAUSSIAN_DECODER_CKPT = TRELLIS_CKPT_ROOT / "slat_dec_gs_swin8_B_64l8gs32_fp16.safetensors"
DINO_RESOLUTION = 518
DINO_EXPECTED_TOKENS = 1374
DINO_EXPECTED_CHANNELS = 1024


app = FastAPI(title="Fridge Multiview Asset Extraction Workbench")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
app.mount("/static", StaticFiles(directory=STATIC_ROOT), name="static")

SAM3_PROCESS: subprocess.Popen[str] | None = None
JOBS: dict[str, subprocess.Popen[str]] = {}
JOB_META: dict[str, dict[str, Any]] = {}
SESSION_LOCK = threading.RLock()


class LabelSpec(BaseModel):
    id: int
    name: str
    color: str | None = None


class SaveViewRequest(BaseModel):
    view_index: int = Field(ge=0, le=3)
    image_data_url: str
    camera: dict[str, Any] = Field(default_factory=dict)
    name: str | None = None


class ImportViewSpec(BaseModel):
    view_index: int = Field(ge=0, le=3)
    image_data_url: str
    name: str | None = None
    original_name: str | None = None
    source_view_id: int | None = None


class ImportViewsRequest(BaseModel):
    views: list[ImportViewSpec] = Field(default_factory=list)
    replace_existing: bool = False


class DatasetLoadRequest(BaseModel):
    object_id: str
    angle_idx: int = 0
    view_count: int = Field(default=4, ge=1, le=4)
    replace_existing: bool = False


class SaveMaskRequest(BaseModel):
    view_index: int = Field(ge=0, le=3)
    mask_data_url: str
    labels: list[LabelSpec] = Field(default_factory=list)


class FinalizeRequest(BaseModel):
    labels: list[LabelSpec] = Field(default_factory=list)


class Sam3StartRequest(BaseModel):
    port: int = SAM3_PORT
    device: str = "cuda"
    confidence_threshold: float = 0.5


class Sam3TextRequest(BaseModel):
    view_index: int = Field(ge=0, le=3)
    prompt: str
    confidence_threshold: float = 0.5


class Sam3PointRequest(BaseModel):
    view_index: int = Field(ge=0, le=3)
    points: list[list[float]]
    point_labels: list[int]
    multimask_output: bool = True


class ReconstructRequest(BaseModel):
    stage: str = "all"
    quick_steps: bool = True
    ss_flow_ckpt: str | None = None
    part_seg_ckpt: str | None = None
    ss_decoder_ckpt: str | None = None
    slat_flow_ckpt: str | None = None
    slat_mesh_decoder_ckpt: str | None = None
    slat_gaussian_decoder_ckpt: str | None = None
    ss_steps: int = 20
    slat_steps: int = 25


def session_dir() -> Path:
    path = WORK_ROOT / SESSION_ID
    path.mkdir(parents=True, exist_ok=True)
    for sub in ("rgb", "mask", "mask_preview", "camera", "dino_input", "dino_tokens", "sam3", "reconstruct", "model_input"):
        (path / sub).mkdir(parents=True, exist_ok=True)
    return path


def _json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(payload, indent=2, ensure_ascii=False) + "\n"
    with tempfile.NamedTemporaryFile(
        mode="w",
        prefix=f".{path.name}-",
        suffix=".tmp",
        dir=path.parent,
        encoding="utf-8",
        delete=False,
    ) as handle:
        staged_path = Path(handle.name)
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())
    try:
        os.replace(staged_path, path)
    finally:
        staged_path.unlink(missing_ok=True)


def _read_json(path: Path, default: Any) -> Any:
    if not path.is_file():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def _output_url(path: Path) -> str | None:
    if not path.is_file():
        return None
    return f"/outputs/{path.relative_to(WORK_ROOT)}"


def _decode_data_url(data_url: str) -> bytes:
    if "," not in data_url:
        raise HTTPException(status_code=400, detail="expected data URL")
    _header, encoded = data_url.split(",", 1)
    try:
        return base64.b64decode(encoded)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"invalid base64 payload: {exc}") from exc


def _data_url_to_image(data_url: str) -> Image.Image:
    payload = _decode_data_url(data_url)
    try:
        with Image.open(io.BytesIO(payload)) as image:
            return ImageOps.exif_transpose(image).convert("RGBA")
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"invalid image payload: {exc}") from exc


def _safe_output_path(path_text: str) -> Path:
    root = WORK_ROOT.resolve()
    path = (WORK_ROOT / path_text).resolve()
    if path != root and root not in path.parents:
        raise HTTPException(status_code=403, detail="path escapes work root")
    if not path.is_file():
        raise HTTPException(status_code=404, detail=str(path))
    return path


def _view_paths(root: Path) -> tuple[list[Path], list[Path]]:
    images = [root / "rgb" / f"view_{idx}.png" for idx in range(4)]
    masks = [root / "mask" / f"mask_{idx}.npy" for idx in range(4)]
    return images, masks


def _physical_view_indices(root: Path) -> list[int]:
    images, masks = _view_paths(root)
    image_indices = {idx for idx, path in enumerate(images) if path.is_file()}
    mask_indices = {idx for idx, path in enumerate(masks) if path.is_file()}
    if image_indices != mask_indices:
        missing_masks = sorted(image_indices - mask_indices)
        missing_images = sorted(mask_indices - image_indices)
        raise ValueError(f"RGB/mask view mismatch: missing_masks={missing_masks}, missing_images={missing_images}")
    return sorted(image_indices)


def _normalized_model_inputs(root: Path) -> dict[str, Any]:
    try:
        physical_indices = _physical_view_indices(root)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if not physical_indices:
        raise HTTPException(status_code=400, detail="expected 1-4 complete RGB/mask views, got 0")

    model_root = root / "model_input"
    rgb_root = model_root / "rgb"
    mask_root = model_root / "mask"
    rgb_root.mkdir(parents=True, exist_ok=True)
    mask_root.mkdir(parents=True, exist_ok=True)
    mapping = [physical_indices[slot % len(physical_indices)] for slot in range(4)]
    replacements: list[tuple[Path, Path]] = []
    with tempfile.TemporaryDirectory(prefix=".normalize-model-input-", dir=root) as staged_text:
        staged_root = Path(staged_text)
        for model_slot, physical_index in enumerate(mapping):
            staged_rgb = staged_root / f"view_{model_slot}.png"
            staged_mask = staged_root / f"mask_{model_slot}.npy"
            shutil.copyfile(root / "rgb" / f"view_{physical_index}.png", staged_rgb)
            shutil.copyfile(root / "mask" / f"mask_{physical_index}.npy", staged_mask)
            replacements.extend([
                (staged_rgb, rgb_root / f"view_{model_slot}.png"),
                (staged_mask, mask_root / f"mask_{model_slot}.npy"),
            ])
        payload = {
            "strategy": "cycle_physical_views_in_ascending_index_order",
            "physical_view_count": len(physical_indices),
            "physical_view_indices": physical_indices,
            "model_slot_count": 4,
            "model_slot_to_physical_view": mapping,
            "images": [str(rgb_root / f"view_{idx}.png") for idx in range(4)],
            "masks": [str(mask_root / f"mask_{idx}.npy") for idx in range(4)],
            "created_unix": time.time(),
        }
        staged_meta = staged_root / "mapping.json"
        _json(staged_meta, payload)
        replacements.append((staged_meta, model_root / "mapping.json"))
        _transactional_replace(root, replacements, [])
    return payload


def _view_state(root: Path) -> list[dict[str, Any]]:
    images, masks = _view_paths(root)
    out = []
    for idx, (image, mask) in enumerate(zip(images, masks)):
        mask_png = root / "mask" / f"mask_{idx}.png"
        preview = root / "mask_preview" / f"mask_{idx}.png"
        camera = root / "camera" / f"view_{idx}.json"
        out.append(
            {
                "view_index": idx,
                "image": str(image),
                "image_url": _output_url(image),
                "image_mtime": str(image.stat().st_mtime_ns) if image.is_file() else None,
                "mask": str(mask),
                "mask_exists": mask.is_file(),
                "mask_mtime": str(mask.stat().st_mtime_ns) if mask.is_file() else None,
                "mask_png_url": _output_url(mask_png),
                "mask_png_mtime": str(mask_png.stat().st_mtime_ns) if mask_png.is_file() else None,
                "mask_preview_url": _output_url(preview),
                "mask_preview_mtime": str(preview.stat().st_mtime_ns) if preview.is_file() else None,
                "camera": _read_json(camera, None),
            }
        )
    return out


def _input_source_state(root: Path) -> dict[str, Any]:
    entries: list[dict[str, Any]] = []
    sources: set[str] = set()
    for idx in range(4):
        image = root / "rgb" / f"view_{idx}.png"
        if not image.is_file():
            continue
        camera = _read_json(root / "camera" / f"view_{idx}.json", {})
        source = str(camera.get("source") or ("3dgs_capture" if camera else "unknown"))
        sources.add(source)
        entries.append(
            {
                "view_index": idx,
                "name": camera.get("name") or f"view_{idx}",
                "source": source,
                "original_name": camera.get("original_name"),
                "source_view_id": camera.get("source_view_id"),
            }
        )
    if not sources:
        source_type = "none"
    elif len(sources) == 1:
        source_type = next(iter(sources))
    else:
        source_type = "mixed"
    return {"type": source_type, "views": entries}


def _view_derivative_paths(root: Path, view_indices: list[int]) -> list[Path]:
    paths: list[Path] = []
    for idx in sorted(set(int(value) for value in view_indices)):
        paths.extend(
            [
                root / "mask" / f"mask_{idx}.npy",
                root / "mask" / f"mask_{idx}.png",
                root / "mask_preview" / f"mask_{idx}.png",
                root / "dino_input" / f"view_{idx}.png",
            ]
        )
        paths.extend(sorted((root / "sam3").glob(f"*_view_{idx}.json")))
    paths.extend([
        root / "dino_tokens" / "tokens.npz",
        root / "manifest.json",
        root / "dataset.json",
        root / "model_input" / "mapping.json",
        *list((root / "model_input").glob("rgb/*.png")),
        *list((root / "model_input").glob("mask/*.npy")),
    ])
    return paths


def _mask_derivative_paths(root: Path, view_index: int) -> list[Path]:
    return [
        root / "dino_input" / f"view_{int(view_index)}.png",
        root / "dino_tokens" / "tokens.npz",
        root / "manifest.json",
        root / "model_input" / "mapping.json",
        *list((root / "model_input").glob("rgb/*.png")),
        *list((root / "model_input").glob("mask/*.npy")),
    ]


def _transactional_replace(
    root: Path,
    replacements: list[tuple[Path, Path]],
    removals: list[Path],
) -> list[str]:
    replacement_targets = [target for _staged, target in replacements]
    targets = list(dict.fromkeys([*replacement_targets, *removals]))
    existing_removals = [path for path in dict.fromkeys(removals) if path.is_file()]
    with tempfile.TemporaryDirectory(prefix=".transaction-backup-", dir=root) as backup_text:
        backup_root = Path(backup_text)
        backups: list[tuple[Path, Path]] = []
        published: list[Path] = []
        try:
            for target in targets:
                if not target.is_file():
                    continue
                backup = backup_root / target.relative_to(root)
                backup.parent.mkdir(parents=True, exist_ok=True)
                os.replace(target, backup)
                backups.append((backup, target))
            for staged, target in replacements:
                target.parent.mkdir(parents=True, exist_ok=True)
                os.replace(staged, target)
                published.append(target)
        except Exception:
            for target in published:
                target.unlink(missing_ok=True)
            for backup, target in reversed(backups):
                target.parent.mkdir(parents=True, exist_ok=True)
                os.replace(backup, target)
            raise
    return [str(path.relative_to(root)) for path in existing_removals]


def _ensure_no_active_jobs() -> None:
    running = [job_id for job_id, process in JOBS.items() if process.poll() is None]
    if running:
        raise HTTPException(status_code=409, detail={"error": "reconstruct job is running", "job_ids": running})


def _labels_payload(labels: list[LabelSpec]) -> list[dict[str, Any]]:
    seen: set[int] = set()
    out = []
    for item in labels:
        if int(item.id) <= 0 or int(item.id) in seen:
            continue
        seen.add(int(item.id))
        name = (item.name or f"part_{int(item.id):02d}").strip() or f"part_{int(item.id):02d}"
        out.append({"id": int(item.id), "name": name, "color": item.color})
    return out


def _part_info(labels: list[LabelSpec]) -> dict[str, Any]:
    parts: dict[str, Any] = {}
    for item in _labels_payload(labels):
        name = str(item["name"]).strip() or f"part_{int(item['id']):02d}"
        slug = "".join(ch.lower() if ch.isalnum() else "_" for ch in name).strip("_") or f"part_{int(item['id']):02d}"
        low = name.lower()
        joint = "revolute" if any(term in low for term in ("door", "hinge", "lid")) else "fixed"
        parts[f"{slug}_{int(item['id']):02d}"] = {
            "label": int(item["id"]),
            "type": name,
            "joint": joint,
            "workbench_source": "fridge_multiview_manual_or_sam3_mask",
        }
    return {
        "format": "arts_gen_workbench_part_info_v1",
        "object": "fridge_3dgs",
        "parts": parts,
        "joint_note": "hinge/open-door viewer is intentionally interface-only in phase 1",
    }


def _colorize_mask(mask: np.ndarray, labels: list[LabelSpec]) -> Image.Image:
    h, w = mask.shape
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    palette = {
        int(item.id): item.color or "#00a3ff"
        for item in labels
        if int(item.id) > 0
    }
    fallback = ["#00a3ff", "#ffb000", "#00b875", "#ff5c7a", "#8f6fff", "#00c2c7"]
    for label_id in sorted(v for v in np.unique(mask).tolist() if int(v) > 0):
        color = palette.get(int(label_id), fallback[(int(label_id) - 1) % len(fallback)])
        color = color.lstrip("#")
        if len(color) != 6:
            color = "00a3ff"
        rgb = [int(color[i : i + 2], 16) for i in (0, 2, 4)]
        sel = mask == int(label_id)
        rgba[sel, :3] = rgb
        rgba[sel, 3] = 180
    return Image.fromarray(rgba, mode="RGBA")


def _rgba_with_mask_alpha(image: Image.Image, mask: np.ndarray) -> Image.Image:
    alpha = Image.fromarray((np.asarray(mask) > 0).astype(np.uint8) * 255, mode="L")
    rgba = image.convert("RGB").convert("RGBA")
    rgba.putalpha(alpha)
    return rgba


def _preprocess_dino_input(image: Image.Image, mask: np.ndarray) -> tuple[Image.Image, dict[str, Any]]:
    rgba_img = _rgba_with_mask_alpha(image, mask)
    rgba = np.asarray(rgba_img)
    alpha = rgba[:, :, 3]
    foreground = np.argwhere(alpha > 0.8 * 255)
    if foreground.shape[0] == 0:
        raise ValueError("empty foreground alpha from mask > 0")
    y0, x0 = foreground.min(axis=0)
    y1, x1 = foreground.max(axis=0)
    center = ((float(x0) + float(x1)) / 2.0, (float(y0) + float(y1)) / 2.0)
    size = int(max(int(x1) - int(x0), int(y1) - int(y0)) * 1.2)
    if size <= 0:
        raise ValueError(f"invalid foreground bbox {(int(x0), int(y0), int(x1), int(y1))}")
    crop_bbox = (
        center[0] - size // 2,
        center[1] - size // 2,
        center[0] + size // 2,
        center[1] + size // 2,
    )
    cropped = rgba_img.crop(crop_bbox).resize((DINO_RESOLUTION, DINO_RESOLUTION), Image.Resampling.LANCZOS)
    cropped_rgba = np.asarray(cropped).astype(np.float32) / 255.0
    rgb = cropped_rgba[:, :, :3] * cropped_rgba[:, :, 3:4]
    preview = Image.fromarray((rgb * 255.0).clip(0, 255).astype(np.uint8), mode="RGB")
    stats = {
        "input_size": [int(image.width), int(image.height)],
        "dino_size": [DINO_RESOLUTION, DINO_RESOLUTION],
        "alpha_bbox_xyxy": [int(x0), int(y0), int(x1), int(y1)],
        "crop_bbox_xyxy": [round(float(v), 3) for v in crop_bbox],
        "foreground_pixels": int(foreground.shape[0]),
        "foreground_fraction": round(float(foreground.shape[0]) / float(mask.size), 6),
        "unique_labels": sorted(int(v) for v in np.unique(mask).tolist()),
    }
    return preview, stats


def _ssflow_input_state(root: Path) -> dict[str, Any]:
    images, masks = _view_paths(root)
    entries: list[dict[str, Any]] = []
    ok = True
    try:
        physical_indices = _physical_view_indices(root)
    except ValueError:
        physical_indices = []
        ok = False
    if not physical_indices:
        ok = False
    for idx in physical_indices:
        image_path, mask_path = images[idx], masks[idx]
        entry: dict[str, Any] = {"view_index": idx, "name": f"view_{idx}"}
        if not image_path.is_file():
            ok = False
            entry.update({"ok": False, "error": f"missing image {image_path.name}"})
            entries.append(entry)
            continue
        if not mask_path.is_file():
            ok = False
            entry.update({"ok": False, "error": f"missing mask {mask_path.name}"})
            entries.append(entry)
            continue
        try:
            with Image.open(image_path) as source_image:
                image = source_image.convert("RGB")
            mask = np.load(mask_path)
            if mask.ndim != 2 or not np.issubdtype(mask.dtype, np.integer):
                raise ValueError(f"mask must be [H,W] integer, got {mask.shape} {mask.dtype}")
            if image.size != (int(mask.shape[1]), int(mask.shape[0])):
                raise ValueError(f"image size {image.size} != mask shape {mask.shape}")
            preview, stats = _preprocess_dino_input(image, mask)
            preview_path = root / "dino_input" / f"view_{idx}.png"
            preview.save(preview_path)
            entry.update({
                "ok": True,
                "preview": str(preview_path),
                "preview_url": _output_url(preview_path),
                "preview_mtime": str(preview_path.stat().st_mtime_ns),
                **stats,
            })
        except Exception as exc:
            ok = False
            entry.update({"ok": False, "error": str(exc)})
        entries.append(entry)
    return {
        "ok": bool(ok),
        "expected_token_shape": [4, DINO_EXPECTED_TOKENS, DINO_EXPECTED_CHANNELS],
        "physical_view_count": len(physical_indices),
        "model_slot_to_physical_view": [
            physical_indices[idx % len(physical_indices)] for idx in range(4)
        ] if physical_indices else [],
        "preprocess": "RGBA alpha from mask>0 -> foreground crop x1.2 -> 518 RGB premultiplied on black -> DINOv2 x_prenorm layer_norm",
        "views": entries,
    }


def _validate_contract(root: Path, declared_label_ids: set[int] | None = None) -> dict[str, Any]:
    images, masks = _view_paths(root)
    try:
        physical_indices = _physical_view_indices(root)
    except ValueError as exc:
        return {"ok": False, "error": str(exc)}
    if not physical_indices:
        return {"ok": False, "error": "expected 1-4 complete RGB/mask views, got 0"}
    image_sizes: list[tuple[int, int]] = []
    mask_shapes: list[tuple[int, int]] = []
    positive_labels: set[int] = set()
    for idx in physical_indices:
        image_path, mask_path = images[idx], masks[idx]
        with Image.open(image_path) as image:
            image_size = tuple(map(int, image.size))
        mask = np.load(mask_path)
        if mask.ndim != 2 or not np.issubdtype(mask.dtype, np.integer):
            return {"ok": False, "error": f"{mask_path} must be [H,W] integer, got {mask.shape} {mask.dtype}"}
        if image_size != (int(mask.shape[1]), int(mask.shape[0])):
            return {"ok": False, "error": f"{image_path.name} size {image_size} != {mask_path.name} shape {mask.shape}"}
        image_sizes.append(image_size)
        mask_shapes.append(tuple(map(int, mask.shape)))
        positive_labels.update(int(v) for v in np.unique(mask).tolist() if int(v) > 0)
    if not positive_labels:
        return {"ok": False, "error": "masks contain no positive labels"}
    declared_labels: set[int] = set(declared_label_ids or set())
    if declared_label_ids is None:
        for item in _read_json(root / "labels.json", []):
            if not isinstance(item, dict):
                return {"ok": False, "error": "labels.json must contain label objects"}
            try:
                label_id = int(item.get("id", 0))
            except (TypeError, ValueError):
                return {"ok": False, "error": f"invalid label id in labels.json: {item.get('id')}"}
            if label_id > 0:
                declared_labels.add(label_id)
    undeclared = sorted(positive_labels - declared_labels)
    if undeclared:
        return {"ok": False, "error": f"mask labels are not declared in labels.json: {undeclared}"}
    return {
        "ok": True,
        "image_count": len(physical_indices),
        "mask_count": len(physical_indices),
        "physical_view_indices": physical_indices,
        "model_slot_count": 4,
        "image_sizes": image_sizes,
        "mask_shapes": mask_shapes,
        "positive_labels": sorted(positive_labels),
    }


def _ckpt_config(req: ReconstructRequest, out_dir: Path) -> dict[str, Any]:
    return {
        "ss_flow_ckpt": req.ss_flow_ckpt or str(DEFAULT_SS_FLOW_CKPT),
        "part_seg_ckpt": req.part_seg_ckpt or str(DEFAULT_PART_SEG_CKPT),
        "ss_decoder_ckpt": req.ss_decoder_ckpt or str(DEFAULT_SS_DECODER_CKPT),
        "slat_flow_ckpt": req.slat_flow_ckpt or str(DEFAULT_SLAT_FLOW_CKPT),
        "slat_mesh_decoder_ckpt": req.slat_mesh_decoder_ckpt or str(DEFAULT_SLAT_MESH_DECODER_CKPT),
        "slat_gaussian_decoder_ckpt": req.slat_gaussian_decoder_ckpt or str(DEFAULT_SLAT_GAUSSIAN_DECODER_CKPT),
        "ss_steps": 2 if req.quick_steps else int(req.ss_steps),
        "slat_steps": 2 if req.quick_steps else int(req.slat_steps),
        "output_dir": str(out_dir),
    }


def _dataset_sources() -> list[dict[str, Any]]:
    if not DATASET_CONFIG.is_file():
        raise HTTPException(status_code=503, detail=f"dataset config not found: {DATASET_CONFIG}")
    payload = _read_json(DATASET_CONFIG, {})
    if "data_root" in payload and ("manifest_path" in payload or "manifest_paths" in payload):
        datasets = [{"dataset_id": payload.get("dataset_id", "default"), **payload}]
    else:
        datasets = list(payload.get("datasets") or [])
    sources: list[dict[str, Any]] = []
    for dataset in datasets:
        data_root = Path(str(dataset.get("data_root", "")))
        manifest_values = dataset.get("manifest_paths") or [dataset.get("manifest_path")]
        for manifest_value in manifest_values:
            if not manifest_value:
                continue
            manifest = Path(str(manifest_value))
            if not manifest.is_absolute():
                manifest = data_root / manifest
            sources.append({
                "dataset_id": str(dataset.get("dataset_id") or "default"),
                "data_root": data_root,
                "manifest_path": manifest,
                "mask_subdir": str(dataset.get("mask_subdir") or "renders"),
            })
    if not sources:
        raise HTTPException(status_code=503, detail=f"dataset config has no manifests: {DATASET_CONFIG}")
    sources.sort(key=lambda item: ("realappliance" not in item["dataset_id"].lower()))
    return sources


def _dataset_modules():
    trellis_root = REPO_ROOT / "TRELLIS-arts"
    if str(trellis_root) not in sys.path:
        sys.path.insert(0, str(trellis_root))
    from part_ss_eval_platform import infer_runs
    from trellis.datasets.arts.part_ss_latent_flow import PartSSLatentFlowDataset

    return infer_runs, PartSSLatentFlowDataset


def _find_dataset_record(object_id: str, angle_idx: int) -> tuple[dict[str, Any], dict[str, Any]]:
    wanted_object = str(object_id)
    wanted_angle = int(angle_idx)
    for source in _dataset_sources():
        manifest = source["manifest_path"]
        if not manifest.is_file():
            continue
        with manifest.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                record = json.loads(line)
                record_object = str(record.get("object_id") or record.get("obj_id") or "")
                if record_object == wanted_object and int(record.get("angle_idx", 0)) == wanted_angle:
                    return source, record
    raise HTTPException(status_code=404, detail=f"dataset object not found: {object_id} angle {angle_idx}")


def _dataset_view_paths(source: dict[str, Any], record: dict[str, Any]) -> tuple[list[Path], list[Path]]:
    _infer_runs, dataset_class = _dataset_modules()
    resolver = dataset_class.__new__(dataset_class)
    resolver.data_root = Path(source["data_root"])
    resolver.mask_root = resolver.data_root / str(source.get("mask_subdir") or "renders")
    sample = {
        "obj_id": str(record.get("object_id") or record.get("obj_id")),
        "angle_idx": int(record.get("angle_idx", 0)),
        "view_indices": [int(value) for value in record.get("view_indices") or []],
        "image_paths": list(record.get("image_paths") or []),
    }
    return resolver._iter_rgb_paths(sample), resolver._iter_mask_paths(sample)


def _remap_dataset_masks(record: dict[str, Any], masks: list[np.ndarray]) -> tuple[list[np.ndarray], list[dict[str, Any]]]:
    """Apply the manifest's target-only local label contract.

    Raw dataset masks may contain fixed body segments and child labels.  EE
    prompt masks must contain only target moving groups as stable local 1..K
    labels; every other raw label is background.
    """
    remap = {int(key): int(value) for key, value in dict(record.get("label_remap") or {}).items()}
    if not remap:
        for index, part in enumerate(record.get("target_parts") or [], start=1):
            local_label = int(part.get("local_label", index))
            values = part.get("prompt_original_labels") or part.get("merged_original_labels")
            if values is None:
                values = [part.get("original_label")]
            if not isinstance(values, (list, tuple, set)):
                values = [values]
            for value in values:
                if value is not None:
                    remap[int(value)] = local_label
    if not remap:
        raise HTTPException(status_code=400, detail="dataset sample has no target label_remap")

    remapped: list[np.ndarray] = []
    for raw in masks:
        local = np.zeros(raw.shape, dtype=np.int32)
        for original_label, local_label in remap.items():
            local[raw == int(original_label)] = int(local_label)
        remapped.append(local)

    names_by_local = {
        int(key): str(value)
        for key, value in dict(record.get("local_label_to_component") or {}).items()
    }
    target_names = [str(value) for value in record.get("target_part_names") or []]
    for index, name in enumerate(target_names, start=1):
        names_by_local.setdefault(index, name)
    label_ids = sorted(set(remap.values()))
    labels = [
        {"id": label_id, "name": names_by_local.get(label_id, f"part_{label_id}"), "color": None}
        for label_id in label_ids
    ]
    return remapped, labels


def _ckpt_status(config: dict[str, Any]) -> dict[str, Any]:
    keys = [
        "ss_flow_ckpt",
        "part_seg_ckpt",
        "ss_decoder_ckpt",
        "slat_flow_ckpt",
        "slat_mesh_decoder_ckpt",
        "slat_gaussian_decoder_ckpt",
    ]
    return {
        key: {"path": str(config[key]), "exists": Path(str(config[key])).expanduser().is_file()}
        for key in keys
    }


def _sam3_health(port: int = SAM3_PORT) -> dict[str, Any]:
    try:
        res = requests.get(f"http://127.0.0.1:{port}/health", timeout=5.0)
        if res.ok:
            return {"running": True, **res.json()}
        return {"running": False, "status_code": res.status_code}
    except Exception as exc:
        return {"running": False, "error": str(exc)}


def _terminate_sam3_process() -> None:
    global SAM3_PROCESS
    if SAM3_PROCESS and SAM3_PROCESS.poll() is None:
        SAM3_PROCESS.send_signal(signal.SIGTERM)
        try:
            SAM3_PROCESS.wait(timeout=5)
        except subprocess.TimeoutExpired:
            SAM3_PROCESS.kill()
            SAM3_PROCESS.wait(timeout=5)
    SAM3_PROCESS = None


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return (STATIC_ROOT / "index.html").read_text(encoding="utf-8")


@app.get("/viewer", response_class=HTMLResponse)
def viewer() -> str:
    return (STATIC_ROOT / "viewer.html").read_text(encoding="utf-8")


@app.get("/assets/point_cloud.ply")
def point_cloud() -> FileResponse:
    if not POINT_CLOUD.is_file():
        raise HTTPException(status_code=404, detail=str(POINT_CLOUD))
    return FileResponse(POINT_CLOUD)


@app.get("/outputs/{path:path}")
def outputs(path: str) -> FileResponse:
    return FileResponse(_safe_output_path(path))


@app.get("/api/config")
def config() -> dict[str, Any]:
    root = session_dir()
    cfg = _ckpt_config(ReconstructRequest(), root / "reconstruct/latest")
    with SESSION_LOCK:
        contract = _validate_contract(root)
    return {
        "repo_root": str(REPO_ROOT),
        "data_root": str(DATA_ROOT),
        "work_root": str(WORK_ROOT),
        "session_id": SESSION_ID,
        "session_dir": str(root),
        "point_cloud": {"path": str(POINT_CLOUD), "exists": POINT_CLOUD.is_file(), "url": "/assets/point_cloud.ply"},
        "view_sources": {"3dgs_capture": POINT_CLOUD.is_file(), "direct_upload": True},
        "sam3": {
            "root": str(SAM3_ROOT),
            "checkpoint": str(SAM3_CKPT),
            "checkpoint_exists": SAM3_CKPT.is_file(),
            "port": SAM3_PORT,
            "health": _sam3_health(),
        },
        "ckpts": _ckpt_status(cfg),
        "legacy_part_seg_ckpt": {"path": str(LEGACY_PART_SEG_CKPT), "exists": LEGACY_PART_SEG_CKPT.is_file()},
        "contract": contract,
    }


@app.get("/api/session")
def session() -> dict[str, Any]:
    root = session_dir()
    with SESSION_LOCK:
        return {
            "session_dir": str(root),
            "manifest": _read_json(root / "manifest.json", None),
            "dataset": _read_json(root / "dataset.json", None),
            "labels": _read_json(root / "labels.json", []),
            "input_source": _input_source_state(root),
            "views": _view_state(root),
            "ssflow_inputs": _ssflow_input_state(root),
            "contract": _validate_contract(root),
        }


@app.get("/api/dataset/objects")
def dataset_objects(limit: int = 5000) -> dict[str, Any]:
    if limit < 1 or limit > 50000:
        raise HTTPException(status_code=400, detail="limit must be between 1 and 50000")
    infer_runs, _dataset_class = _dataset_modules()
    objects: dict[tuple[str, str], dict[str, Any]] = {}
    for source in _dataset_sources():
        if len(objects) >= limit:
            break
        config = {"data_root": str(source["data_root"]), "manifest_path": str(source["manifest_path"])}
        try:
            entries = infer_runs.list_objects(config, limit=limit)
        except FileNotFoundError:
            continue
        for entry in entries:
            key = (source["dataset_id"], str(entry["object_id"]))
            current = objects.get(key)
            if current is None:
                current = {**entry, "dataset_id": source["dataset_id"]}
                objects[key] = current
            else:
                current["angles"] = sorted(set(current.get("angles", [])) | set(entry.get("angles", [])))
                current["target_part_names"] = sorted(
                    set(current.get("target_part_names", [])) | set(entry.get("target_part_names", []))
                )
            if len(objects) >= limit:
                break
    return {
        "ok": True,
        "config": str(DATASET_CONFIG),
        "objects": list(objects.values())[:limit],
    }


@app.post("/api/dataset/load")
def dataset_load(req: DatasetLoadRequest) -> dict[str, Any]:
    source, record = _find_dataset_record(req.object_id, req.angle_idx)
    rgb_paths, mask_paths = _dataset_view_paths(source, record)
    available = min(len(rgb_paths), len(mask_paths))
    if available < req.view_count:
        raise HTTPException(status_code=400, detail=f"dataset sample has {available} views, requested {req.view_count}")
    selected_rgb = rgb_paths[: req.view_count]
    selected_masks = mask_paths[: req.view_count]
    missing = [str(path) for path in [*selected_rgb, *selected_masks] if not path.is_file()]
    if missing:
        raise HTTPException(status_code=404, detail={"error": "dataset view files missing", "missing": missing})

    images: list[Image.Image] = []
    masks: list[np.ndarray] = []
    for rgb_path, mask_path in zip(selected_rgb, selected_masks):
        with Image.open(rgb_path) as image:
            rgb = ImageOps.exif_transpose(image).convert("RGB")
        mask = np.asarray(np.load(mask_path))
        if mask.ndim != 2 or not np.issubdtype(mask.dtype, np.integer):
            raise HTTPException(status_code=400, detail=f"dataset mask must be [H,W] integer: {mask_path} {mask.shape} {mask.dtype}")
        if rgb.size != (int(mask.shape[1]), int(mask.shape[0])):
            raise HTTPException(status_code=400, detail=f"dataset RGB/mask size mismatch: {rgb_path} {rgb.size}, {mask_path} {mask.shape}")
        images.append(rgb)
        masks.append(mask.astype(np.int32, copy=False))

    masks, labels = _remap_dataset_masks(record, masks)
    label_specs = [LabelSpec(**item) for item in labels]
    root = session_dir()
    metadata = {
        "source": "dataset",
        "dataset_id": source["dataset_id"],
        "dataset_config": str(DATASET_CONFIG),
        "manifest_path": str(source["manifest_path"]),
        "object_id": str(req.object_id),
        "angle_idx": int(req.angle_idx),
        "physical_view_count": int(req.view_count),
        "source_view_indices": [int(value) for value in (record.get("view_indices") or [])[: req.view_count]],
        "target_part_names": list(record.get("target_part_names") or []),
        "loaded_unix": time.time(),
    }
    with tempfile.TemporaryDirectory(prefix=".dataset-load-", dir=root) as staged_text:
        staged_root = Path(staged_text)
        replacements: list[tuple[Path, Path]] = []
        for idx, (image, mask) in enumerate(zip(images, masks)):
            staged_image = staged_root / f"view_{idx}.png"
            staged_mask = staged_root / f"mask_{idx}.npy"
            staged_preview = staged_root / f"preview_{idx}.png"
            staged_camera = staged_root / f"camera_{idx}.json"
            image.save(staged_image, format="PNG")
            np.save(staged_mask, mask)
            _colorize_mask(mask, label_specs).save(staged_preview)
            target_image = root / "rgb" / f"view_{idx}.png"
            _json(staged_camera, {
                "view_index": idx,
                "name": f"dataset_view_{metadata['source_view_indices'][idx]}",
                "source": "dataset",
                "source_view_id": metadata["source_view_indices"][idx],
                "image": str(target_image),
                "size": [image.width, image.height],
                "camera": {},
                "dataset": metadata,
                "saved_unix": time.time(),
            })
            replacements.extend([
                (staged_image, target_image),
                (staged_mask, root / "mask" / f"mask_{idx}.npy"),
                (staged_preview, root / "mask_preview" / f"mask_{idx}.png"),
                (staged_camera, root / "camera" / f"view_{idx}.json"),
            ])
        staged_labels = staged_root / "labels.json"
        staged_metadata = staged_root / "dataset.json"
        _json(staged_labels, labels)
        _json(staged_metadata, metadata)
        replacements.extend([(staged_labels, root / "labels.json"), (staged_metadata, root / "dataset.json")])
        with SESSION_LOCK:
            _ensure_no_active_jobs()
            existing = any((root / "rgb" / f"view_{idx}.png").is_file() for idx in range(4))
            if existing and not req.replace_existing:
                raise HTTPException(status_code=409, detail="session already has RGB views; set replace_existing=true after explicit confirmation")
            invalidated = _transactional_replace(
                root,
                replacements,
                [
                    *_view_derivative_paths(root, [0, 1, 2, 3]),
                    *[root / "rgb" / f"view_{idx}.png" for idx in range(req.view_count, 4)],
                    *[root / "mask" / f"mask_{idx}.npy" for idx in range(req.view_count, 4)],
                    *[root / "mask" / f"mask_{idx}.png" for idx in range(4)],
                    *[root / "mask_preview" / f"mask_{idx}.png" for idx in range(req.view_count, 4)],
                    *[root / "camera" / f"view_{idx}.json" for idx in range(req.view_count, 4)],
                    root / "part_info.json",
                    root / "model_input" / "mapping.json",
                    *list((root / "model_input").glob("rgb/*.png")),
                    *list((root / "model_input").glob("mask/*.npy")),
                ],
            )
    return {
        "ok": True,
        "dataset": metadata,
        "labels": labels,
        "views": _view_state(root),
        "input_source": _input_source_state(root),
        "contract": _validate_contract(root),
        "invalidated": invalidated,
    }


@app.post("/api/ssflow/dino/check")
def ssflow_dino_check() -> dict[str, Any]:
    root = session_dir()
    with SESSION_LOCK:
        _ensure_no_active_jobs()
        return _ssflow_dino_check(root)


def _ssflow_dino_check(root: Path) -> dict[str, Any]:
    contract = _validate_contract(root)
    if not contract.get("ok"):
        raise HTTPException(status_code=400, detail=contract)
    model_inputs = _normalized_model_inputs(root)
    images = [Path(path) for path in model_inputs["images"]]
    masks = [Path(path) for path in model_inputs["masks"]]
    rgba_images = []
    for image_path, mask_path in zip(images, masks):
        with Image.open(image_path) as source_image:
            image = source_image.convert("RGB")
        mask = np.load(mask_path)
        rgba_images.append(_rgba_with_mask_alpha(image, mask))
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    trellis_root = REPO_ROOT / "TRELLIS-arts"
    if str(trellis_root) not in sys.path:
        sys.path.insert(0, str(trellis_root))
    try:
        import torch
        import inference

        tokens = inference._images_to_tokens(rgba_images).detach().float().cpu()
        shape = list(tokens.shape)
        finite = bool(torch.isfinite(tokens).all().item())
        if shape != [4, DINO_EXPECTED_TOKENS, DINO_EXPECTED_CHANNELS]:
            raise ValueError(f"DINO token shape {shape} != {[4, DINO_EXPECTED_TOKENS, DINO_EXPECTED_CHANNELS]}")
        if not finite:
            raise ValueError("DINO tokens contain NaN/Inf")
        out_path = root / "dino_tokens" / "tokens.npz"
        np.savez(out_path, tokens=tokens.numpy().astype(np.float32, copy=False))
        return {
            "ok": True,
            "tokens": str(out_path),
            "shape": shape,
            "dtype": str(tokens.numpy().dtype),
            "finite": finite,
            "mean": float(tokens.mean().item()),
            "std": float(tokens.std().item()),
            "preprocess": _ssflow_input_state(root)["preprocess"],
        }
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"DINO check failed: {exc}") from exc


@app.post("/api/views")
def save_view(req: SaveViewRequest) -> dict[str, Any]:
    root = session_dir()
    image = _data_url_to_image(req.image_data_url).convert("RGB")
    path = root / "rgb" / f"view_{req.view_index}.png"
    camera_payload = {
        "view_index": int(req.view_index),
        "name": req.name or f"view_{req.view_index}",
        "source": "3dgs_capture",
        "image": str(path),
        "size": [image.width, image.height],
        "camera": req.camera,
        "saved_unix": time.time(),
    }
    camera_path = root / "camera" / f"view_{req.view_index}.json"
    with tempfile.TemporaryDirectory(prefix=".save-view-", dir=root) as staged_text:
        staged_root = Path(staged_text)
        staged_image = staged_root / "view.png"
        staged_camera = staged_root / "camera.json"
        image.save(staged_image, format="PNG")
        _json(staged_camera, camera_payload)
        with SESSION_LOCK:
            _ensure_no_active_jobs()
            invalidated = _transactional_replace(
                root,
                [(staged_image, path), (staged_camera, camera_path)],
                _view_derivative_paths(root, [req.view_index]),
            )
    return {
        "ok": True,
        "path": str(path),
        "url": f"/outputs/{path.relative_to(WORK_ROOT)}",
        "camera": camera_payload,
        "invalidated": invalidated,
    }


@app.post("/api/views/import")
def import_views(req: ImportViewsRequest) -> dict[str, Any]:
    if not 1 <= len(req.views) <= 4:
        raise HTTPException(status_code=400, detail=f"expected 1-4 images, got {len(req.views)}")
    by_index = {int(item.view_index): item for item in req.views}
    expected_indices = list(range(len(req.views)))
    if sorted(by_index) != expected_indices or len(by_index) != len(req.views):
        raise HTTPException(status_code=400, detail=f"view_index values must be unique and exactly {expected_indices}")

    decoded = {idx: _data_url_to_image(by_index[idx].image_data_url).convert("RGB") for idx in expected_indices}
    root = session_dir()
    with tempfile.TemporaryDirectory(prefix=".import-views-", dir=root) as staged_text:
        staged_root = Path(staged_text)
        replacements: list[tuple[Path, Path]] = []
        for idx in expected_indices:
            item = by_index[idx]
            image = decoded[idx]
            target = root / "rgb" / f"view_{idx}.png"
            staged_image = staged_root / f"view_{idx}.png"
            image.save(staged_image, format="PNG")
            replacements.append((staged_image, target))
            payload = {
                "view_index": idx,
                "name": item.name or f"view_{idx}",
                "source": "direct_upload",
                "original_name": item.original_name,
                "source_view_id": item.source_view_id,
                "image": str(target),
                "size": [image.width, image.height],
                "camera": {},
                "saved_unix": time.time(),
            }
            staged_camera = staged_root / f"camera_{idx}.json"
            _json(staged_camera, payload)
            replacements.append((staged_camera, root / "camera" / f"view_{idx}.json"))

        with SESSION_LOCK:
            _ensure_no_active_jobs()
            existing = any((root / "rgb" / f"view_{idx}.png").is_file() for idx in range(4))
            if existing and not req.replace_existing:
                raise HTTPException(
                    status_code=409,
                    detail="session already has RGB views; set replace_existing=true after explicit confirmation",
                )
            invalidated = _transactional_replace(
                root,
                replacements,
                [
                    *_view_derivative_paths(root, [0, 1, 2, 3]),
                    *[root / "rgb" / f"view_{idx}.png" for idx in range(len(req.views), 4)],
                    *[root / "camera" / f"view_{idx}.json" for idx in range(len(req.views), 4)],
                    root / "model_input" / "mapping.json",
                    *list((root / "model_input").glob("rgb/*.png")),
                    *list((root / "model_input").glob("mask/*.npy")),
                ],
            )
            return {
                "ok": True,
                "views": _view_state(root),
                "input_source": _input_source_state(root),
                "invalidated": invalidated,
            }


@app.post("/api/masks")
def save_mask(req: SaveMaskRequest) -> dict[str, Any]:
    root = session_dir()
    rgba = _data_url_to_image(req.mask_data_url)
    arr = np.asarray(rgba)
    if arr.ndim != 3 or arr.shape[2] < 1:
        raise HTTPException(status_code=400, detail=f"mask image must be RGBA/RGB, got {arr.shape}")
    mask = arr[..., 0].astype(np.int32)
    labels = _labels_payload(req.labels)
    label_specs = [LabelSpec(**item) for item in labels]
    declared_labels = {int(item["id"]) for item in labels}
    image_path = root / "rgb" / f"view_{req.view_index}.png"
    npy_path = root / "mask" / f"mask_{req.view_index}.npy"
    preview_path = root / "mask_preview" / f"mask_{req.view_index}.png"

    with SESSION_LOCK:
        _ensure_no_active_jobs()
        if not image_path.is_file():
            raise HTTPException(status_code=400, detail=f"save view_{req.view_index}.png first")
        with Image.open(image_path) as image:
            image_size = image.size
        if image_size != (int(mask.shape[1]), int(mask.shape[0])):
            raise HTTPException(status_code=400, detail=f"mask shape {mask.shape} does not match image size {image_size}")

        masks_by_view: dict[int, np.ndarray] = {int(req.view_index): mask}
        positive_labels: set[int] = {int(value) for value in np.unique(mask).tolist() if int(value) > 0}
        for idx in range(4):
            if idx == int(req.view_index):
                continue
            other_path = root / "mask" / f"mask_{idx}.npy"
            if not other_path.is_file():
                continue
            other_mask = np.load(other_path)
            if other_mask.ndim != 2 or not np.issubdtype(other_mask.dtype, np.integer):
                raise HTTPException(status_code=400, detail=f"invalid existing mask {other_path}: {other_mask.shape} {other_mask.dtype}")
            masks_by_view[idx] = other_mask
            positive_labels.update(int(value) for value in np.unique(other_mask).tolist() if int(value) > 0)
        undeclared = sorted(positive_labels - declared_labels)
        if undeclared:
            raise HTTPException(status_code=400, detail=f"mask labels are not declared in labels: {undeclared}")

        with tempfile.TemporaryDirectory(prefix=".save-mask-", dir=root) as staged_text:
            staged_root = Path(staged_text)
            replacements: list[tuple[Path, Path]] = []
            staged_npy = staged_root / f"mask_{req.view_index}.npy"
            staged_png = staged_root / f"mask_{req.view_index}.png"
            np.save(staged_npy, mask.astype(np.int32, copy=False))
            Image.fromarray(mask.clip(0, 255).astype(np.uint8), mode="L").save(staged_png)
            replacements.extend(
                [
                    (staged_npy, npy_path),
                    (staged_png, root / "mask" / f"mask_{req.view_index}.png"),
                ]
            )
            for idx, view_mask in masks_by_view.items():
                staged_preview = staged_root / f"preview_{idx}.png"
                _colorize_mask(view_mask, label_specs).save(staged_preview)
                replacements.append((staged_preview, root / "mask_preview" / f"mask_{idx}.png"))
            staged_labels = staged_root / "labels.json"
            staged_part_info = staged_root / "part_info.json"
            _json(staged_labels, labels)
            _json(staged_part_info, _part_info(label_specs))
            replacements.extend(
                [
                    (staged_labels, root / "labels.json"),
                    (staged_part_info, root / "part_info.json"),
                ]
            )
            invalidated = _transactional_replace(
                root,
                replacements,
                _mask_derivative_paths(root, req.view_index),
            )
        return {
            "ok": True,
            "mask": str(npy_path),
            "preview": str(preview_path),
            "preview_url": f"/outputs/{preview_path.relative_to(WORK_ROOT)}",
            "labels": labels,
            "unique_labels": sorted(int(v) for v in np.unique(mask).tolist()),
            "invalidated": invalidated,
            "contract": _validate_contract(root),
        }


@app.get("/api/masks/{view_index}")
def load_mask(view_index: int) -> dict[str, Any]:
    if view_index < 0 or view_index > 3:
        raise HTTPException(status_code=400, detail="view_index must be 0..3")
    root = session_dir()
    with SESSION_LOCK:
        mask_path = root / "mask" / f"mask_{view_index}.npy"
        if not mask_path.is_file():
            raise HTTPException(status_code=404, detail=str(mask_path))
        mask = np.load(mask_path)
        if mask.ndim != 2 or not np.issubdtype(mask.dtype, np.integer):
            raise HTTPException(status_code=400, detail=f"invalid mask {mask.shape} {mask.dtype}")
        buf = io.BytesIO()
        Image.fromarray(mask.clip(0, 255).astype(np.uint8), mode="L").save(buf, format="PNG")
        encoded = base64.b64encode(buf.getvalue()).decode("ascii")
        return {
            "ok": True,
            "view_index": view_index,
            "width": int(mask.shape[1]),
            "height": int(mask.shape[0]),
            "unique_labels": sorted(int(v) for v in np.unique(mask).tolist()),
            "mask_data_url": f"data:image/png;base64,{encoded}",
        }


@app.post("/api/export/finalize")
def finalize(req: FinalizeRequest) -> dict[str, Any]:
    root = session_dir()
    with SESSION_LOCK:
        _ensure_no_active_jobs()
        return _finalize(req, root)


def _finalize(req: FinalizeRequest, root: Path) -> dict[str, Any]:
    labels = _labels_payload(req.labels)
    label_specs = [LabelSpec(**item) for item in labels]
    declared_labels = {int(item["id"]) for item in labels}
    masks_by_view: dict[int, np.ndarray] = {}
    positive_labels: set[int] = set()
    for idx in range(4):
        mask_path = root / "mask" / f"mask_{idx}.npy"
        if not mask_path.is_file():
            continue
        mask = np.load(mask_path)
        if mask.ndim != 2 or not np.issubdtype(mask.dtype, np.integer):
            raise HTTPException(status_code=400, detail=f"invalid mask {mask_path}: {mask.shape} {mask.dtype}")
        masks_by_view[idx] = mask
        positive_labels.update(int(value) for value in np.unique(mask).tolist() if int(value) > 0)
    undeclared = sorted(positive_labels - declared_labels)
    if undeclared:
        raise HTTPException(status_code=400, detail=f"mask labels are not declared in labels: {undeclared}")

    part_info = _part_info(label_specs)
    contract = _validate_contract(root, declared_label_ids=declared_labels)
    if not contract.get("ok"):
        raise HTTPException(status_code=400, detail=contract)
    model_inputs = _normalized_model_inputs(root)
    images, masks = _view_paths(root)
    physical_indices = list(contract["physical_view_indices"])
    input_source = _input_source_state(root)
    uses_3dgs = any(item.get("source") == "3dgs_capture" for item in input_source["views"])
    manifest = {
        "format": "fridge_3dgs_asset_extraction_workbench_v1",
        "created_unix": time.time(),
        "object": "fridge",
        "input_source": input_source,
        "source_3dgs": str(POINT_CLOUD) if uses_3dgs else None,
        "view_count": len(physical_indices),
        "images": [str(images[idx]) for idx in physical_indices],
        "masks": [str(masks[idx]) for idx in physical_indices],
        "model_inputs": model_inputs,
        "part_info": str(root / "part_info.json"),
        "labels": labels,
        "mask_contract": "[H,W] int32 .npy; 0=background; positive labels are stable cross-view part ids",
        "contract": contract,
        "hinge_open_viewer": {"phase": "interface_only", "blocked": False},
    }
    with tempfile.TemporaryDirectory(prefix=".finalize-", dir=root) as staged_text:
        staged_root = Path(staged_text)
        replacements: list[tuple[Path, Path]] = []
        staged_labels = staged_root / "labels.json"
        staged_part_info = staged_root / "part_info.json"
        staged_manifest = staged_root / "manifest.json"
        _json(staged_labels, labels)
        _json(staged_part_info, part_info)
        _json(staged_manifest, manifest)
        replacements.extend(
            [
                (staged_labels, root / "labels.json"),
                (staged_part_info, root / "part_info.json"),
                (staged_manifest, root / "manifest.json"),
            ]
        )
        for idx, mask in masks_by_view.items():
            staged_preview = staged_root / f"preview_{idx}.png"
            _colorize_mask(mask, label_specs).save(staged_preview)
            replacements.append((staged_preview, root / "mask_preview" / f"mask_{idx}.png"))
        _transactional_replace(root, replacements, [])
    return {"ok": bool(contract.get("ok")), "manifest": manifest}


@app.post("/api/sam3/start")
def sam3_start(req: Sam3StartRequest) -> dict[str, Any]:
    global SAM3_PROCESS
    if SAM3_PROCESS and SAM3_PROCESS.poll() is None:
        health = _sam3_health(req.port)
        if health.get("import_ok") is not False:
            return {"ok": True, "already_running": True, "health": health}
        _terminate_sam3_process()
    if not SAM3_ROOT.is_dir():
        raise HTTPException(status_code=400, detail=f"SAM3 root not found: {SAM3_ROOT}")
    if not SAM3_CKPT.is_file():
        raise HTTPException(status_code=400, detail=f"SAM3 checkpoint not found: {SAM3_CKPT}")
    root = session_dir()
    log_path = root / "sam3/server.log"
    env = dict(os.environ)
    env["PYTHONPATH"] = f"{SAM3_ROOT}:{env.get('PYTHONPATH', '')}"
    cmd = [
        sys.executable,
        str(WORKBENCH_ROOT / "server/sam3_server.py"),
        "--host",
        "127.0.0.1",
        "--port",
        str(req.port),
        "--sam3-root",
        str(SAM3_ROOT),
        "--checkpoint",
        str(SAM3_CKPT),
        "--device",
        req.device,
        "--confidence-threshold",
        str(req.confidence_threshold),
    ]
    handle = log_path.open("a", encoding="utf-8")
    handle.write(f"\n[start] {' '.join(cmd)}\n")
    handle.flush()
    SAM3_PROCESS = subprocess.Popen(cmd, stdout=handle, stderr=subprocess.STDOUT, text=True, cwd=str(REPO_ROOT), env=env)
    time.sleep(0.5)
    return {"ok": True, "pid": SAM3_PROCESS.pid, "log": str(log_path), "health": _sam3_health(req.port)}


@app.post("/api/sam3/stop")
def sam3_stop() -> dict[str, Any]:
    _terminate_sam3_process()
    return {"ok": True}


@app.get("/api/sam3/status")
def sam3_status() -> dict[str, Any]:
    root = session_dir()
    log_path = root / "sam3/server.log"
    tail = ""
    if log_path.is_file():
        tail = "\n".join(log_path.read_text(encoding="utf-8", errors="replace").splitlines()[-60:])
    return {"process_running": bool(SAM3_PROCESS and SAM3_PROCESS.poll() is None), "health": _sam3_health(), "log_tail": tail}


@app.post("/api/sam3/text")
def sam3_text(req: Sam3TextRequest) -> dict[str, Any]:
    root = session_dir()
    with SESSION_LOCK:
        return _sam3_text(req, root)


def _sam3_text(req: Sam3TextRequest, root: Path) -> dict[str, Any]:
    image_path = root / "rgb" / f"view_{req.view_index}.png"
    if not image_path.is_file():
        raise HTTPException(status_code=400, detail=f"save view_{req.view_index}.png first")
    try:
        res = requests.post(
            f"http://127.0.0.1:{SAM3_PORT}/segment_text",
            json={"image_path": str(image_path), "prompt": req.prompt, "confidence_threshold": req.confidence_threshold},
            timeout=120,
        )
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"SAM3 server unavailable: {exc}") from exc
    if not res.ok:
        raise HTTPException(status_code=res.status_code, detail=res.text)
    payload = res.json()
    _json(root / "sam3" / f"text_view_{req.view_index}.json", payload)
    return payload


@app.post("/api/sam3/points")
def sam3_points(req: Sam3PointRequest) -> dict[str, Any]:
    root = session_dir()
    with SESSION_LOCK:
        return _sam3_points(req, root)


def _sam3_points(req: Sam3PointRequest, root: Path) -> dict[str, Any]:
    image_path = root / "rgb" / f"view_{req.view_index}.png"
    if not image_path.is_file():
        raise HTTPException(status_code=400, detail=f"save view_{req.view_index}.png first")
    try:
        res = requests.post(
            f"http://127.0.0.1:{SAM3_PORT}/segment_points",
            json={
                "image_path": str(image_path),
                "points": req.points,
                "point_labels": req.point_labels,
                "multimask_output": req.multimask_output,
            },
            timeout=120,
        )
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"SAM3 server unavailable: {exc}") from exc
    if not res.ok:
        raise HTTPException(status_code=res.status_code, detail=res.text)
    payload = res.json()
    _json(root / "sam3" / f"points_view_{req.view_index}.json", payload)
    return payload


@app.post("/api/reconstruct/start")
def reconstruct_start(req: ReconstructRequest) -> dict[str, Any]:
    root = session_dir()
    with SESSION_LOCK:
        _ensure_no_active_jobs()
        return _reconstruct_start(req, root)


def _reconstruct_start(req: ReconstructRequest, root: Path) -> dict[str, Any]:
    stage = str(req.stage or "all").strip() or "all"
    allowed_stages = {"all", "ssflow_decoder", "partseg"}
    if stage not in allowed_stages:
        raise HTTPException(status_code=400, detail=f"stage must be one of {sorted(allowed_stages)}")
    contract = _validate_contract(root)
    if not contract.get("ok"):
        raise HTTPException(status_code=400, detail=contract)
    model_inputs = _normalized_model_inputs(root)
    out_dir = root / "reconstruct" / f"{stage}_{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}"
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = _ckpt_config(req, out_dir)
    ckpt_status = _ckpt_status(cfg)
    missing = {key: item for key, item in ckpt_status.items() if not item["exists"]}
    if missing:
        raise HTTPException(status_code=400, detail={"error": "missing ckpt", "missing": missing, "ckpts": ckpt_status})
    cfg_path = out_dir / "ckpt_config.json"
    _json(cfg_path, cfg)
    images = [Path(path) for path in model_inputs["images"]]
    masks = [Path(path) for path in model_inputs["masks"]]
    cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts/inference/reconstruct.py"),
        "--images",
        *[str(path) for path in images],
        "--masks",
        *[str(path) for path in masks],
        "--part-info",
        str(root / "part_info.json"),
        "--ckpt-config-json",
        str(cfg_path),
        "--out-dir",
        str(out_dir),
    ]
    if req.quick_steps:
        cmd.append("--quick-steps")
    log_path = out_dir / "run.log"
    handle = log_path.open("w", encoding="utf-8")
    handle.write(f"[cmd] {' '.join(cmd)}\n")
    handle.flush()
    env = dict(os.environ)
    env.setdefault("SPCONV_ALGO", "native")
    env.setdefault("ATTN_BACKEND", "sdpa")
    env.setdefault("SPARSE_ATTN_BACKEND", "sdpa")
    proc = subprocess.Popen(cmd, stdout=handle, stderr=subprocess.STDOUT, text=True, cwd=str(REPO_ROOT), env=env)
    job_id = uuid.uuid4().hex[:12]
    JOBS[job_id] = proc
    JOB_META[job_id] = {
        "job_id": job_id,
        "pid": proc.pid,
        "out_dir": str(out_dir),
        "log": str(log_path),
        "cmd": cmd,
        "stage": stage,
        "started_unix": time.time(),
        "quick_steps": bool(req.quick_steps),
        "model_inputs": model_inputs,
    }
    _json(root / "reconstruct/latest_job.json", JOB_META[job_id])
    _json(root / f"reconstruct/latest_{stage}_job.json", JOB_META[job_id])
    return {"ok": True, **JOB_META[job_id], "ckpts": ckpt_status}


@app.get("/api/reconstruct/status/{job_id}")
def reconstruct_status(job_id: str) -> dict[str, Any]:
    meta = JOB_META.get(job_id)
    if not meta:
        raise HTTPException(status_code=404, detail=job_id)
    proc = JOBS.get(job_id)
    return_code = None if proc is None else proc.poll()
    log_path = Path(meta["log"])
    tail = ""
    if log_path.is_file():
        tail = "\n".join(log_path.read_text(encoding="utf-8", errors="replace").splitlines()[-120:])
    out_dir = Path(meta["out_dir"])
    summary = _read_json(out_dir / "summary.json", None)
    files = []
    if out_dir.is_dir():
        for path in sorted(out_dir.rglob("*")):
            if path.is_file():
                files.append({
                    "path": str(path),
                    "rel": str(path.relative_to(WORK_ROOT)),
                    "url": f"/outputs/{path.relative_to(WORK_ROOT)}",
                    "size": path.stat().st_size,
                })
    return {"ok": True, **meta, "running": return_code is None, "return_code": return_code, "log_tail": tail, "summary": summary, "files": files}


@app.get("/health")
def health() -> dict[str, Any]:
    return {"ok": True, "session": str(session_dir())}


@app.exception_handler(HTTPException)
def http_exception_handler(_request, exc: HTTPException) -> JSONResponse:
    return JSONResponse(status_code=exc.status_code, content={"ok": False, "detail": exc.detail})


@app.exception_handler(Exception)
def exception_handler(_request, exc: Exception) -> JSONResponse:
    return JSONResponse(status_code=500, content={"ok": False, "detail": repr(exc)})


def main() -> None:
    import uvicorn

    host = os.environ.get("FRIDGE_3DGS_HOST", "0.0.0.0")
    port = int(os.environ.get("FRIDGE_3DGS_PORT", "7865"))
    uvicorn.run(app, host=host, port=port, log_level="info")


if __name__ == "__main__":
    main()
