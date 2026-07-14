#!/usr/bin/env python3
from __future__ import annotations

import base64
import io
import json
import os
import re
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
from pydantic import BaseModel, ConfigDict, Field


REPO_ROOT = Path(__file__).resolve().parents[3]
WORKBENCH_ROOT = Path(__file__).resolve().parents[1]
STATIC_ROOT = WORKBENCH_ROOT / "static"
KIN_SHARED_ROOT = REPO_ROOT / "post_process/utils/shared_libs"
DATA_ROOT = Path(os.environ.get("ARTS_GEN_DATA_ROOT", "/robot/data-lab/jzh/art-gen"))
WORK_ROOT = Path(
    os.environ.get(
        "FRIDGE_3DGS_WORK_ROOT",
        str(DATA_ROOT / "workbench/fridge_3dgs"),
    )
)
DEFAULT_SESSION_ID = os.environ.get("FRIDGE_3DGS_SESSION", "fridge_point_cloud")
EE_EVAL_DIRNAME = "ee-eval"
ACTIVE_RUN_FILE = ".active_run.json"
RUN_ID_PATTERN = re.compile(r"^[A-Za-z0-9](?:[A-Za-z0-9._-]{0,63})$")


def _persisted_session_id() -> str:
    marker = WORK_ROOT / EE_EVAL_DIRNAME / ACTIVE_RUN_FILE
    try:
        payload = json.loads(marker.read_text(encoding="utf-8"))
        run_id = str(payload.get("run_id", ""))
        if RUN_ID_PATTERN.fullmatch(run_id) and run_id not in {".", ".."}:
            return run_id
    except (FileNotFoundError, OSError, ValueError, TypeError, json.JSONDecodeError):
        pass
    return DEFAULT_SESSION_ID


SESSION_ID = _persisted_session_id()
POINT_CLOUD = Path(os.environ.get("FRIDGE_3DGS_POINT_CLOUD", str(REPO_ROOT / "data/point_cloud.ply")))
SAM3_ROOT = Path(os.environ.get("SAM3_ROOT", str(DATA_ROOT / "local-deps/sam3")))
DEFAULT_SAM3_PT = DATA_ROOT / "weights/sam3/sam3.pt"
DEFAULT_SAM3_SAFETENSORS = DATA_ROOT / "weights/sam3/sam3_1038lab_f055b060_sam3.safetensors"
SAM3_CKPT = Path(os.environ.get("SAM3_CKPT", str(DEFAULT_SAM3_PT if DEFAULT_SAM3_PT.is_file() else DEFAULT_SAM3_SAFETENSORS)))
SAM3_PORT = int(os.environ.get("FRIDGE_3DGS_SAM3_PORT", "8787"))
DEFAULT_SS_FLOW_CKPT = DATA_ROOT / "ckpt/tre-ss-flow/tre-ss-concat-0616-1/ckpts/denoiser_ema0.999_step0012500.pt"
PART_SEG_CKPT_ROOT = DATA_ROOT / "ckpt/part-prompt-seg"
DEFAULT_PART_SEG_RUN = "part-prompt-seg-L-0709-1-joint"
DEFAULT_PART_SEG_CKPT = PART_SEG_CKPT_ROOT / DEFAULT_PART_SEG_RUN / "ckpts/latest.pt"
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


app = FastAPI(title="Arts-Gen EE Eval Workbench")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
app.mount("/static", StaticFiles(directory=STATIC_ROOT), name="static")
app.mount("/kin-shared", StaticFiles(directory=KIN_SHARED_ROOT), name="kin-shared")

SAM3_PROCESS: subprocess.Popen[str] | None = None
JOBS: dict[str, subprocess.Popen[str]] = {}
JOB_META: dict[str, dict[str, Any]] = {}
KIN_JOBS: dict[str, subprocess.Popen[str]] = {}
KIN_JOB_META: dict[str, dict[str, Any]] = {}
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
    dataset_id: str | None = None
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
    model_config = ConfigDict(extra="forbid")

    stage: str = "dino_ss_flow"
    force: bool = False
    quick_steps: bool = False
    ss_flow_ckpt: str | None = None
    part_seg_run_id: str | None = None
    ss_decoder_ckpt: str | None = None
    slat_flow_ckpt: str | None = None
    slat_mesh_decoder_ckpt: str | None = None
    slat_gaussian_decoder_ckpt: str | None = None
    ss_steps: int = 20
    slat_steps: int = 25

class KinAgentRequest(BaseModel):
    max_iterations: int = Field(default=7, ge=1, le=9)
    use_dataset_motion_states: bool = False


class RunCreateRequest(BaseModel):
    run_id: str
    select: bool = True


class RunSelectRequest(BaseModel):
    run_id: str
    create: bool = False


def _ee_eval_root() -> Path:
    return WORK_ROOT / EE_EVAL_DIRNAME


def _validate_run_id(run_id: str) -> str:
    normalized = str(run_id).strip()
    if normalized in {".", ".."} or not RUN_ID_PATTERN.fullmatch(normalized):
        raise HTTPException(
            status_code=400,
            detail="run_id must be 1-64 characters using only letters, digits, '.', '_' or '-', and start with a letter or digit",
        )
    return normalized


def _ensure_session_layout(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    for sub in ("rgb", "mask", "object_mask", "mask_preview", "camera", "dino_input", "dino_tokens", "sam3", "reconstruct", "model_input"):
        (path / sub).mkdir(parents=True, exist_ok=True)
    return path


def session_dir() -> Path:
    run_id = _validate_run_id(SESSION_ID)
    target = _ee_eval_root() / run_id
    legacy = WORK_ROOT / run_id
    if not target.exists() and legacy.is_dir() and legacy != _ee_eval_root():
        target.parent.mkdir(parents=True, exist_ok=True)
        os.replace(legacy, target)
    return _ensure_session_layout(target)


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
    object_mask_root = model_root / "object_mask"
    rgb_root.mkdir(parents=True, exist_ok=True)
    mask_root.mkdir(parents=True, exist_ok=True)
    object_mask_root.mkdir(parents=True, exist_ok=True)
    mapping = [physical_indices[slot % len(physical_indices)] for slot in range(4)]
    replacements: list[tuple[Path, Path]] = []
    with tempfile.TemporaryDirectory(prefix=".normalize-model-input-", dir=root) as staged_text:
        staged_root = Path(staged_text)
        for model_slot, physical_index in enumerate(mapping):
            staged_rgb = staged_root / f"view_{model_slot}.png"
            staged_mask = staged_root / f"mask_{model_slot}.npy"
            staged_object_mask = staged_root / f"object_mask_{model_slot}.npy"
            shutil.copyfile(root / "rgb" / f"view_{physical_index}.png", staged_rgb)
            shutil.copyfile(root / "mask" / f"mask_{physical_index}.npy", staged_mask)
            source_object_mask = root / "object_mask" / f"mask_{physical_index}.npy"
            if source_object_mask.is_file():
                shutil.copyfile(source_object_mask, staged_object_mask)
            else:
                part_mask = np.load(staged_mask)
                np.save(staged_object_mask, (part_mask > 0).astype(np.uint8))
            replacements.extend([
                (staged_rgb, rgb_root / f"view_{model_slot}.png"),
                (staged_mask, mask_root / f"mask_{model_slot}.npy"),
                (staged_object_mask, object_mask_root / f"mask_{model_slot}.npy"),
            ])
        dataset = _read_json(root / "dataset.json", {}) or {}
        token_cache_path = dataset.get("token_cache_path")
        if not token_cache_path and dataset.get("source") == "dataset":
            source, record = _find_dataset_record(
                str(dataset.get("object_id")), int(dataset.get("angle_idx", 0)), str(dataset.get("dataset_id"))
            )
            resolved_cache = _dataset_token_cache(source, record)
            token_cache_path = str(resolved_cache) if resolved_cache is not None else None
        normalized_tokens: Path | None = None
        if token_cache_path and Path(str(token_cache_path)).is_file():
            with np.load(Path(str(token_cache_path)), allow_pickle=False) as token_payload:
                all_tokens = np.asarray(token_payload["tokens"])
            source_views = [int(value) for value in dataset.get("source_view_indices") or []]
            selected_views = [source_views[index] for index in mapping]
            selected_tokens = np.ascontiguousarray(all_tokens[np.asarray(selected_views, dtype=np.int64)])
            if selected_tokens.shape != (4, DINO_EXPECTED_TOKENS, DINO_EXPECTED_CHANNELS):
                raise HTTPException(status_code=400, detail=f"canonical token selection has invalid shape {selected_tokens.shape}")
            staged_tokens = staged_root / "canonical_tokens.npz"
            np.savez_compressed(staged_tokens, tokens=selected_tokens.astype(np.float32, copy=False))
            normalized_tokens = model_root / "canonical_tokens.npz"
            replacements.append((staged_tokens, normalized_tokens))
        payload = {
            "strategy": "cycle_physical_views_in_ascending_index_order",
            "physical_view_count": len(physical_indices),
            "physical_view_indices": physical_indices,
            "model_slot_count": 4,
            "model_slot_to_physical_view": mapping,
            "images": [str(rgb_root / f"view_{idx}.png") for idx in range(4)],
            "masks": [str(mask_root / f"mask_{idx}.npy") for idx in range(4)],
            "object_masks": [str(object_mask_root / f"mask_{idx}.npy") for idx in range(4)],
            "cond_tokens": str(normalized_tokens) if normalized_tokens is not None else None,
            "token_source": "canonical_dataset_cache" if normalized_tokens is not None else "dino_from_object_foreground",
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
                root / "object_mask" / f"mask_{idx}.npy",
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
        *list((root / "model_input").glob("object_mask/*.npy")),
        root / "model_input" / "canonical_tokens.npz",
        *list((root / "reconstruct" / "pipeline").rglob("*")),
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
        *list((root / "model_input").glob("object_mask/*.npy")),
        root / "model_input" / "canonical_tokens.npz",
        *list((root / "reconstruct" / "pipeline").rglob("*")),
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
    running = [
        job_id
        for jobs in (JOBS, KIN_JOBS)
        for job_id, process in jobs.items()
        if process.poll() is None
    ]
    if running:
        raise HTTPException(status_code=409, detail={"error": "reconstruct job is running", "job_ids": running})


def _run_state(run_id: str, path: Path) -> dict[str, Any]:
    dataset = _read_json(path / "dataset.json", None)
    manifest = _read_json(path / "manifest.json", None)
    latest_job = _read_json(path / "reconstruct" / "latest_job.json", None)
    try:
        modified_unix = path.stat().st_mtime
    except OSError:
        modified_unix = None
    return {
        "run_id": run_id,
        "id": run_id,
        "active": run_id == SESSION_ID,
        "path": str(path),
        "dataset": dataset,
        "object_id": dataset.get("object_id") if isinstance(dataset, dict) else None,
        "has_inputs": any((path / "rgb" / f"view_{index}.png").is_file() for index in range(4)),
        "has_manifest": isinstance(manifest, dict),
        "latest_job": latest_job,
        "modified_unix": modified_unix,
    }


def _select_run(run_id: str) -> dict[str, Any]:
    global SESSION_ID
    run_id = _validate_run_id(run_id)
    root = _ee_eval_root() / run_id
    if not root.is_dir() or root.is_symlink():
        raise HTTPException(status_code=404, detail=f"ee-eval run does not exist: {run_id}")
    if run_id != SESSION_ID:
        _ensure_no_active_jobs()
        SESSION_ID = run_id
        _json(_ee_eval_root() / ACTIVE_RUN_FILE, {"run_id": run_id, "selected_unix": time.time()})
    return _run_state(run_id, _ensure_session_layout(root))


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
        if any(term in low for term in ("drawer", "slide", "tray", "pull_out")):
            joint = "prismatic"
        elif any(term in low for term in ("door", "hinge", "lid", "dial", "knob")):
            joint = "revolute"
        else:
            joint = "unknown"
        parts[f"{slug}_{int(item['id']):02d}"] = {
            "label": int(item["id"]),
            "type": name,
            "joint": joint,
            "workbench_source": "dataset_mask_or_wild_sam3_points",
        }
    return {
        "format": "arts_gen_workbench_part_info_v1",
        "object": "ee_eval_session",
        "parts": parts,
        "joint_note": "hinge/open-door viewer is intentionally interface-only in phase 1",
    }


def _ensure_part_info(root: Path) -> Path:
    path = root / "part_info.json"
    if path.is_file():
        return path
    labels = _read_json(root / "labels.json", [])
    if not isinstance(labels, list):
        raise HTTPException(status_code=400, detail="labels.json must contain a list")
    try:
        label_specs = [LabelSpec(**item) for item in labels]
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"invalid labels.json: {exc}") from exc
    if not _labels_payload(label_specs):
        raise HTTPException(status_code=400, detail="cannot reconstruct without positive part labels")
    _json(path, _part_info(label_specs))
    return path


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
        image_path, mask_path = images[idx], root / "object_mask" / f"mask_{idx}.npy"
        if not mask_path.is_file():
            mask_path = masks[idx]
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
        "preprocess": "whole-object foreground (dataset RGBA alpha/raw mask; wild part-mask union fallback) -> crop x1.2 -> 518 RGB premultiplied on black -> DINOv2 x_prenorm layer_norm",
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
    part_seg_ckpt = str(_resolve_part_seg_run_id(req.part_seg_run_id or DEFAULT_PART_SEG_RUN))
    return {
        "ss_flow_ckpt": req.ss_flow_ckpt or str(DEFAULT_SS_FLOW_CKPT),
        "part_seg_ckpt": part_seg_ckpt,
        "ss_decoder_ckpt": req.ss_decoder_ckpt or str(DEFAULT_SS_DECODER_CKPT),
        "slat_flow_ckpt": req.slat_flow_ckpt or str(DEFAULT_SLAT_FLOW_CKPT),
        "slat_mesh_decoder_ckpt": req.slat_mesh_decoder_ckpt or str(DEFAULT_SLAT_MESH_DECODER_CKPT),
        "slat_gaussian_decoder_ckpt": req.slat_gaussian_decoder_ckpt or str(DEFAULT_SLAT_GAUSSIAN_DECODER_CKPT),
        "ss_steps": 2 if req.quick_steps else int(req.ss_steps),
        "ss_cfg_strength": 7.5,
        "ss_fusion_mode": "concat",
        "ss_seed": 20260713,
        "slat_steps": 2 if req.quick_steps else int(req.slat_steps),
        "output_dir": str(out_dir),
    }


def _resolve_part_seg_run_id(run_id: str) -> Path:
    value = str(run_id or "").strip()
    relative = Path(value)
    if not value or relative.is_absolute() or len(relative.parts) != 1 or value in {".", ".."}:
        raise HTTPException(status_code=400, detail="invalid Part Prompt Seg training folder")
    root = PART_SEG_CKPT_ROOT.resolve()
    run_root = (root / relative).resolve()
    if not run_root.is_relative_to(root) or not run_root.is_dir() or run_root.is_symlink():
        raise HTTPException(status_code=404, detail=f"Part Prompt Seg training folder not found: {value}")
    ckpt_root = run_root / "ckpts"
    latest = ckpt_root / "latest.pt"
    if latest.is_file():
        resolved_latest = latest.resolve()
        if resolved_latest.is_relative_to(ckpt_root.resolve()) and resolved_latest.is_file():
            return latest
    raise HTTPException(
        status_code=404,
        detail=f"Part Prompt Seg training folder has no valid ckpts/latest.pt: {value}",
    )


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


def _find_dataset_record(
    object_id: str,
    angle_idx: int,
    dataset_id: str | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    wanted_object = str(object_id)
    wanted_angle = int(angle_idx)
    for source in _dataset_sources():
        if dataset_id and str(source["dataset_id"]) != str(dataset_id):
            continue
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


def _dataset_token_cache(source: dict[str, Any], record: dict[str, Any]) -> Path | None:
    data_root = Path(source["data_root"])
    object_id = str(record.get("object_id") or record.get("obj_id"))
    angle_idx = int(record.get("angle_idx", 0))
    explicit = dict(record.get("paths") or {}).get("dinov2_tokens")
    candidates = []
    if explicit:
        explicit_path = Path(str(explicit))
        candidates.append(explicit_path if explicit_path.is_absolute() else data_root / explicit_path)
    candidates.extend(
        data_root / "reconstruction" / subdir / object_id / f"angle_{angle_idx}" / "tokens.npz"
        for subdir in (
            "dinov2_tokens_official_prenorm1374",
            "dinov2_tokens",
            "dinov2_tokens_prenorm",
        )
    )
    for path in candidates:
        if path.is_file():
            return path.resolve()
    return None


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
        "active_run": SESSION_ID,
        "run_root": str(root),
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


@app.get("/api/checkpoints/part-prompt-seg")
def part_prompt_seg_checkpoints() -> dict[str, Any]:
    root = PART_SEG_CKPT_ROOT.resolve()
    runs: list[dict[str, Any]] = []
    if root.is_dir():
        for run_root in root.iterdir():
            ckpt_root = run_root / "ckpts"
            if not run_root.is_dir() or run_root.is_symlink() or not ckpt_root.is_dir():
                continue
            try:
                checkpoint = _resolve_part_seg_run_id(run_root.name)
            except HTTPException:
                continue
            stat = checkpoint.stat()
            runs.append({
                "id": run_root.name,
                "run_name": run_root.name,
                "path": str(run_root),
                "checkpoint_path": str(checkpoint),
                "checkpoint_filename": checkpoint.name,
                "uses_latest": True,
                "size_bytes": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
                "recommended": run_root.name == DEFAULT_PART_SEG_RUN,
                "label": run_root.name,
            })
    runs.sort(key=lambda item: (not item["recommended"], -int(item["mtime_ns"]), item["run_name"]))
    default_id = DEFAULT_PART_SEG_RUN if any(item["id"] == DEFAULT_PART_SEG_RUN for item in runs) else (runs[0]["id"] if runs else None)
    selected_id = None
    selected_config = _read_json(
        session_dir() / "reconstruct" / "pipeline" / "part_prompt_seg" / "ckpt_config.json",
        {},
    )
    selected_path_value = selected_config.get("part_seg_ckpt") if isinstance(selected_config, dict) else None
    if selected_path_value:
        selected_path = Path(str(selected_path_value)).expanduser().resolve()
        if selected_path.is_relative_to(root):
            relative = selected_path.relative_to(root)
            if len(relative.parts) >= 3 and relative.parts[1] == "ckpts":
                selected_id = relative.parts[0]
    return {
        "ok": True,
        "root": str(root),
        "default_id": default_id,
        "selected_id": selected_id,
        "default_path": str(_resolve_part_seg_run_id(default_id)) if default_id else None,
        "runs": runs,
    }


@app.get("/api/ee-eval/runs")
def ee_eval_runs() -> dict[str, Any]:
    root = _ee_eval_root()
    root.mkdir(parents=True, exist_ok=True)
    runs = [
        _run_state(path.name, path)
        for path in sorted(root.iterdir(), key=lambda item: item.name)
        if path.is_dir() and not path.is_symlink() and RUN_ID_PATTERN.fullmatch(path.name)
    ]
    return {
        "ok": True,
        "root": str(root),
        "active_run_id": SESSION_ID,
        "runs": runs,
    }


@app.get("/api/runs")
def runs() -> dict[str, Any]:
    payload = ee_eval_runs()
    return {
        "ok": True,
        "root": payload["root"],
        "active_run": payload["active_run_id"],
        "runs": payload["runs"],
    }


@app.post("/api/ee-eval/runs")
def ee_eval_run_create(req: RunCreateRequest) -> dict[str, Any]:
    run_id = _validate_run_id(req.run_id)
    root = _ee_eval_root() / run_id
    with SESSION_LOCK:
        created = not root.exists()
        if root.exists() and (not root.is_dir() or root.is_symlink()):
            raise HTTPException(status_code=409, detail=f"run path is not a regular directory: {run_id}")
        _ensure_session_layout(root)
        selected = _select_run(run_id) if req.select else _run_state(run_id, root)
    return {"ok": True, "created": created, "run": selected, "active_run_id": SESSION_ID}


@app.post("/api/ee-eval/runs/select")
def ee_eval_run_select(req: RunSelectRequest) -> dict[str, Any]:
    with SESSION_LOCK:
        run_id = _validate_run_id(req.run_id)
        root = _ee_eval_root() / run_id
        if req.create and not root.exists():
            _ensure_session_layout(root)
        selected = _select_run(req.run_id)
    return {"ok": True, "run": selected, "active_run_id": SESSION_ID}


@app.post("/api/runs/select")
def run_select(req: RunSelectRequest) -> dict[str, Any]:
    payload = ee_eval_run_select(req)
    return {"ok": True, "active_run": payload["active_run_id"], "run": payload["run"]}


@app.get("/api/session")
def session() -> dict[str, Any]:
    root = session_dir()
    with SESSION_LOCK:
        return {
            "session_dir": str(root),
            "active_run": SESSION_ID,
            "run_root": str(root),
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
    source, record = _find_dataset_record(req.object_id, req.angle_idx, req.dataset_id)
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
    object_masks: list[np.ndarray] = []
    for rgb_path, mask_path in zip(selected_rgb, selected_masks):
        with Image.open(rgb_path) as image:
            source_image = ImageOps.exif_transpose(image)
            rgba = source_image.convert("RGBA")
            rgb = source_image.convert("RGB")
        mask = np.asarray(np.load(mask_path))
        if mask.ndim != 2 or not np.issubdtype(mask.dtype, np.integer):
            raise HTTPException(status_code=400, detail=f"dataset mask must be [H,W] integer: {mask_path} {mask.shape} {mask.dtype}")
        if rgb.size != (int(mask.shape[1]), int(mask.shape[0])):
            raise HTTPException(status_code=400, detail=f"dataset RGB/mask size mismatch: {rgb_path} {rgb.size}, {mask_path} {mask.shape}")
        images.append(rgb)
        masks.append(mask.astype(np.int32, copy=False))
        alpha = np.asarray(rgba.getchannel("A"), dtype=np.uint8)
        if int(alpha.min()) < 255:
            object_mask = alpha > 0
            object_mask_source = "source_rgba_alpha"
        else:
            object_mask = mask > 0
            object_mask_source = "raw_mask_positive"
        if not bool(object_mask.any()):
            raise HTTPException(status_code=400, detail=f"empty whole-object foreground: {rgb_path}")
        object_masks.append(object_mask.astype(np.uint8))

    masks, labels = _remap_dataset_masks(record, masks)
    label_specs = [LabelSpec(**item) for item in labels]
    root = session_dir()
    token_cache = _dataset_token_cache(source, record)
    metadata = {
        "source": "dataset",
        "dataset_id": source["dataset_id"],
        "dataset_config": str(DATASET_CONFIG),
        "manifest_path": str(source["manifest_path"]),
        "object_id": str(req.object_id),
        "angle_idx": int(req.angle_idx),
        "physical_view_count": int(req.view_count),
        "source_view_indices": [int(value) for value in (record.get("view_indices") or [])[: req.view_count]],
        "object_mask_contract": "source RGBA alpha when non-opaque, otherwise raw dataset mask > 0",
        "object_mask_source": object_mask_source,
        "token_cache_path": str(token_cache) if token_cache is not None else None,
        "target_part_names": list(record.get("target_part_names") or []),
        "loaded_unix": time.time(),
    }
    with tempfile.TemporaryDirectory(prefix=".dataset-load-", dir=root) as staged_text:
        staged_root = Path(staged_text)
        replacements: list[tuple[Path, Path]] = []
        for idx, (image, mask, object_mask) in enumerate(zip(images, masks, object_masks)):
            staged_image = staged_root / f"view_{idx}.png"
            staged_mask = staged_root / f"mask_{idx}.npy"
            staged_preview = staged_root / f"preview_{idx}.png"
            staged_object_mask = staged_root / f"object_mask_{idx}.npy"
            staged_camera = staged_root / f"camera_{idx}.json"
            image.save(staged_image, format="PNG")
            np.save(staged_mask, mask)
            np.save(staged_object_mask, object_mask)
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
                (staged_object_mask, root / "object_mask" / f"mask_{idx}.npy"),
                (staged_preview, root / "mask_preview" / f"mask_{idx}.png"),
                (staged_camera, root / "camera" / f"view_{idx}.json"),
            ])
        staged_labels = staged_root / "labels.json"
        staged_part_info = staged_root / "part_info.json"
        staged_metadata = staged_root / "dataset.json"
        _json(staged_labels, labels)
        _json(staged_part_info, _part_info(label_specs))
        _json(staged_metadata, metadata)
        replacements.extend([
            (staged_labels, root / "labels.json"),
            (staged_part_info, root / "part_info.json"),
            (staged_metadata, root / "dataset.json"),
        ])
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
                    *[root / "object_mask" / f"mask_{idx}.npy" for idx in range(req.view_count, 4)],
                    *[root / "mask" / f"mask_{idx}.png" for idx in range(4)],
                    *[root / "mask_preview" / f"mask_{idx}.png" for idx in range(req.view_count, 4)],
                    *[root / "camera" / f"view_{idx}.json" for idx in range(req.view_count, 4)],
                    root / "model_input" / "mapping.json",
                    *list((root / "model_input").glob("rgb/*.png")),
                    *list((root / "model_input").glob("mask/*.npy")),
                    *list((root / "model_input").glob("object_mask/*.npy")),
                    root / "model_input" / "canonical_tokens.npz",
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
    masks = [Path(path) for path in model_inputs["object_masks"]]
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

        cached_tokens = model_inputs.get("cond_tokens")
        if cached_tokens:
            with np.load(cached_tokens, allow_pickle=False) as payload:
                tokens = torch.from_numpy(np.asarray(payload["tokens"], dtype=np.float32))
        else:
            tokens = inference._images_to_tokens(rgba_images).detach().float().cpu()
        shape = list(tokens.shape)
        finite = bool(torch.isfinite(tokens).all().item())
        if shape != [4, DINO_EXPECTED_TOKENS, DINO_EXPECTED_CHANNELS]:
            raise ValueError(f"DINO token shape {shape} != {[4, DINO_EXPECTED_TOKENS, DINO_EXPECTED_CHANNELS]}")
        if not finite:
            raise ValueError("DINO tokens contain NaN/Inf")
        out_path = root / "dino_tokens" / "tokens.npz"
        np.savez(out_path, tokens=tokens.numpy().astype(np.float32, copy=False))
        from scripts.inference.reconstruct_stages import _save_token_visualizations

        visualization = _save_token_visualizations(tokens.numpy(), root / "dino_tokens")["token_visualization"]
        return {
            "ok": True,
            "tokens": str(out_path),
            "shape": shape,
            "dtype": str(tokens.numpy().dtype),
            "finite": finite,
            "mean": float(tokens.mean().item()),
            "std": float(tokens.std().item()),
            "visualization": {
                **visualization,
                "input_view_urls": [
                    _output_url(root / "dino_input" / f"view_{index}.png") for index in range(4)
                ],
                "pca_url": _output_url(root / "dino_tokens" / "token_pca.png"),
                "norm_url": _output_url(root / "dino_tokens" / "token_norm.png"),
                "pca_view_urls": [
                    _output_url(root / "dino_tokens" / f"token_pca_view_{index}.png") for index in range(4)
                ],
                "norm_view_urls": [
                    _output_url(root / "dino_tokens" / f"token_norm_view_{index}.png") for index in range(4)
                ],
            },
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
    dataset_meta = _read_json(root / "dataset.json", None)
    uses_3dgs = any(item.get("source") == "3dgs_capture" for item in input_source["views"])
    manifest = {
        "format": "arts_gen_ee_eval_workbench_v1",
        "created_unix": time.time(),
        "object": (dataset_meta or {}).get("object_id") or SESSION_ID,
        "dataset": dataset_meta,
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
        "kin_agent_handoff": {
            "ready": bool(contract.get("ok")),
            "part_info": str(root / "part_info.json"),
            "input_manifest": str(root / "manifest.json"),
            "reconstruction_root": str(root / "reconstruct"),
        },
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


def _pipeline_stage_status(root: Path, stage: str) -> dict[str, Any]:
    stage_root = root / "reconstruct" / "pipeline" / stage
    payload = _read_json(stage_root / "status.json", {"stage": stage, "state": "not_started", "progress": 0})
    payload = dict(payload)
    artifacts = payload.get("artifacts")
    if isinstance(artifacts, dict):
        payload["artifact_urls"] = {
            key: (
                [_output_url(stage_root / value) for value in values]
                if isinstance(values, list)
                else _output_url(stage_root / values)
            )
            for key, values in artifacts.items()
        }
    if stage == "dino_ss_flow":
        pca_views = [stage_root / f"token_pca_view_{index}.png" for index in range(4)]
        rgb_views = [root / "dino_input" / f"view_{index}.png" for index in range(4)]
        if all(path.is_file() for path in pca_views):
            payload.setdefault("artifact_urls", {})["pca_views"] = [_output_url(path) for path in pca_views]
        if all(path.is_file() for path in rgb_views):
            payload.setdefault("artifact_urls", {})["rgb_views"] = [_output_url(path) for path in rgb_views]
    payload["files"] = [
        {
            "rel": str(path.relative_to(stage_root)),
            "url": _output_url(path),
            "size": path.stat().st_size,
            "mtime_ns": path.stat().st_mtime_ns,
        }
        for path in sorted(stage_root.rglob("*"))
        if path.is_file() and path.name not in {"run.log", "ckpt_config.json", "status.json"}
    ] if stage_root.is_dir() else []
    return payload


@app.get("/api/reconstruct/pipeline")
def reconstruct_pipeline() -> dict[str, Any]:
    root = session_dir()
    stages = ("dino_ss_flow", "ss_decode", "part_prompt_seg", "slat_decode")
    active_jobs = [
        dict(meta)
        for job_id, meta in JOB_META.items()
        if JOBS.get(job_id) is not None and JOBS[job_id].poll() is None
    ]
    return {
        "ok": True,
        "pipeline_root": str(root / "reconstruct" / "pipeline"),
        "stages": {stage: _pipeline_stage_status(root, stage) for stage in stages},
        "active_jobs": active_jobs,
    }


@app.get("/api/reconstruct/voxel/ss_decode")
def reconstruct_ss_decode_voxel() -> dict[str, Any]:
    root = session_dir()
    status = _pipeline_stage_status(root, "ss_decode")
    if status.get("state") not in {"complete", "cached"}:
        raise HTTPException(status_code=409, detail="SS Decoder has not completed for the current inputs")
    path = root / "reconstruct" / "pipeline" / "ss_decode" / "whole_coords.npy"
    if not path.is_file():
        raise HTTPException(status_code=404, detail="SS Decoder voxel output is not available")
    coords = np.load(path, allow_pickle=False)
    if coords.ndim != 2 or coords.shape[1] != 3:
        raise HTTPException(status_code=500, detail=f"invalid whole_coords shape {coords.shape}")
    coords = np.asarray(coords, dtype=np.int32)
    total = int(coords.shape[0])
    max_display = 75000
    stride = max(1, int(np.ceil(total / max_display)))
    display = coords[::stride]
    bounds = {
        "min": coords.min(axis=0).tolist() if total else [0, 0, 0],
        "max": coords.max(axis=0).tolist() if total else [0, 0, 0],
    }
    return {
        "ok": True,
        "resolution": 64,
        "voxel_count": total,
        "display_count": int(display.shape[0]),
        "display_stride": stride,
        "bounds": bounds,
        "coords": display.tolist(),
    }


@app.get("/api/reconstruct/voxel/part_prompt_seg")
def reconstruct_part_prompt_seg_voxel() -> dict[str, Any]:
    root = session_dir()
    status = _pipeline_stage_status(root, "part_prompt_seg")
    if status.get("state") not in {"complete", "cached"}:
        raise HTTPException(status_code=409, detail="Part Prompt Seg has not completed for the current inputs")

    ss_path = root / "reconstruct" / "pipeline" / "ss_decode" / "whole_coords.npy"
    stage_root = root / "reconstruct" / "pipeline" / "part_prompt_seg"
    parts_path = stage_root / "part_coords.npz"
    if not ss_path.is_file() or not parts_path.is_file():
        raise HTTPException(status_code=404, detail="Part Prompt Seg voxel output is not available")

    whole = np.asarray(np.load(ss_path, allow_pickle=False), dtype=np.int32)
    if whole.ndim != 2 or whole.shape[1] != 3:
        raise HTTPException(status_code=500, detail=f"invalid whole_coords shape {whole.shape}")
    metadata = _read_json(stage_root / "metadata.json", {}) or {}
    part_names = metadata.get("part_names") if isinstance(metadata.get("part_names"), dict) else {}
    configured_labels = {
        int(item["id"]): item
        for item in (_read_json(root / "labels.json", []) or [])
        if isinstance(item, dict) and str(item.get("id", "")).lstrip("-").isdigit()
    }
    palette = ["#146c94", "#b45f06", "#2a7f62", "#9a6fb0", "#ca9834", "#4f83b8"]

    resolution = 64
    whole_keys = (
        whole[:, 0].astype(np.int64) * resolution * resolution
        + whole[:, 1].astype(np.int64) * resolution
        + whole[:, 2].astype(np.int64)
    )
    raw_parts: list[tuple[int, np.ndarray, np.ndarray]] = []
    with np.load(parts_path, allow_pickle=False) as payload:
        part_ids = [int(value) for value in metadata.get("part_ids", [])]
        if not part_ids:
            part_ids = sorted(int(key) for key in payload.files if int(key) >= 0)
        for part_id in part_ids:
            key = str(part_id)
            coords = np.asarray(payload[key], dtype=np.int32).reshape(-1, 3) if key in payload else np.empty((0, 3), dtype=np.int32)
            keys = np.empty((0,), dtype=np.int64)
            if coords.size:
                coords = np.unique(coords, axis=0)
                keys = (
                    coords[:, 0].astype(np.int64) * resolution * resolution
                    + coords[:, 1].astype(np.int64) * resolution
                    + coords[:, 2].astype(np.int64)
                )
                inside = np.isin(keys, whole_keys)
                coords, keys = coords[inside], keys[inside]
            raw_parts.append((part_id, coords, keys))

    all_part_keys = np.concatenate([keys for _, _, keys in raw_parts]) if raw_parts else np.empty((0,), dtype=np.int64)
    if all_part_keys.size:
        unique_part_keys, key_counts = np.unique(all_part_keys, return_counts=True)
        conflict_keys = unique_part_keys[key_counts > 1]
    else:
        conflict_keys = np.empty((0,), dtype=np.int64)
    assigned_keys: list[np.ndarray] = []
    layers: list[dict[str, Any]] = []
    for index, (part_id, coords, keys) in enumerate(raw_parts):
        if conflict_keys.size and keys.size:
            keep = ~np.isin(keys, conflict_keys)
            coords, keys = coords[keep], keys[keep]
        if keys.size:
            assigned_keys.append(keys)
        configured = configured_labels.get(part_id, {})
        layers.append({
            "id": f"part-{part_id}",
            "part_id": part_id,
            "label": str(configured.get("name") or part_names.get(str(part_id)) or f"part_{part_id:02d}"),
            "kind": "part",
            "color": str(configured.get("color") or palette[index % len(palette)]),
            "voxel_count": int(coords.shape[0]),
            "visible": True,
            "coords": coords.tolist(),
        })

    if conflict_keys.size:
        conflict_coords = whole[np.isin(whole_keys, conflict_keys)]
        assigned_keys.append(conflict_keys)
        layers.append({
            "id": "conflicts",
            "part_id": -2,
            "label": "label conflicts",
            "kind": "conflict",
            "color": "#d43f3a",
            "voxel_count": int(conflict_coords.shape[0]),
            "visible": True,
            "coords": conflict_coords.tolist(),
        })

    claimed = np.unique(np.concatenate(assigned_keys)) if assigned_keys else np.empty((0,), dtype=np.int64)
    body = whole[~np.isin(whole_keys, claimed)]
    layers.insert(0, {
        "id": "body",
        "part_id": -1,
        "label": "body residual",
        "kind": "body",
        "color": "#929995",
        "voxel_count": int(body.shape[0]),
        "visible": True,
        "coords": body.tolist(),
    })
    total = int(sum(layer["voxel_count"] for layer in layers))
    return {
        "ok": True,
        "resolution": resolution,
        "voxel_count": total,
        "whole_voxel_count": int(whole.shape[0]),
        "overlap_voxel_count": int(conflict_keys.shape[0]),
        "layers": layers,
    }


@app.get("/api/reconstruct/viewer-manifest")
def reconstruct_viewer_manifest() -> dict[str, Any]:
    root = session_dir()
    stage_root = root / "reconstruct" / "pipeline" / "slat_decode"
    summary = _read_json(stage_root / "summary.json", None)
    if not isinstance(summary, dict):
        raise HTTPException(status_code=404, detail="slat_decode summary is not available")

    overall_assets = summary.get("overall_assets") or {}
    overall_mesh = stage_root / "overall" / str(overall_assets.get("mesh", "overall.glb"))
    overall_gaussian = stage_root / "overall" / str(overall_assets.get("gaussian", "overall.ply"))
    palette = ["#42c6ab", "#df8b47", "#75a7e8", "#d477aa", "#b0d063", "#9a82dd"]
    components = []
    for index, part in enumerate(summary.get("parts") or []):
        mesh_path = Path(str(part.get("mesh_path"))) if part.get("mesh_path") else None
        gaussian_path = Path(str(part.get("gaussian_path"))) if part.get("gaussian_path") else None
        components.append({
            "id": str(part.get("part_id", index + 1)),
            "label": str(part.get("label") or f"Part {index + 1}"),
            "kind": str(part.get("kind") or "part"),
            "mesh_url": _output_url(mesh_path) if mesh_path else None,
            "gaussian_url": _output_url(gaussian_path) if gaussian_path else None,
            "color": palette[index % len(palette)],
            "visible": False,
            "voxel_count": int(part.get("voxel_count", 0)),
        })
    dataset = _read_json(root / "dataset.json", {}) or {}
    return {
        "ok": True,
        "viewer": {
            "title": f"{dataset.get('object_id') or SESSION_ID} decoded components",
            "overall": {
                "id": "overall",
                "label": "Complete",
                "kind": "overall",
                "mesh_url": _output_url(overall_mesh),
                "gaussian_url": _output_url(overall_gaussian),
                "visible": True,
            },
            "components": components,
        },
    }


def _reconstruct_start(req: ReconstructRequest, root: Path) -> dict[str, Any]:
    stage = str(req.stage or "dino_ss_flow").strip() or "dino_ss_flow"
    aliases = {"ssflow_decoder": "dino_ss_flow", "partseg": "part_prompt_seg", "decoder": "slat_decode"}
    stage = aliases.get(stage, stage)
    allowed_stages = {"dino_ss_flow", "ss_decode", "part_prompt_seg", "slat_decode"}
    if stage not in allowed_stages:
        raise HTTPException(status_code=400, detail=f"stage must be one of {sorted(allowed_stages)}")
    contract = _validate_contract(root)
    if not contract.get("ok"):
        raise HTTPException(status_code=400, detail=contract)
    part_info_path = _ensure_part_info(root)
    model_inputs = _normalized_model_inputs(root)
    pipeline_root = root / "reconstruct" / "pipeline"
    out_dir = pipeline_root / stage
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = _ckpt_config(req, out_dir)
    ckpt_status = _ckpt_status(cfg)
    required_ckpts = {
        "dino_ss_flow": {"ss_flow_ckpt"},
        "ss_decode": {"ss_decoder_ckpt"},
        "part_prompt_seg": {"part_seg_ckpt", "ss_decoder_ckpt"},
        "slat_decode": {"slat_flow_ckpt", "slat_mesh_decoder_ckpt", "slat_gaussian_decoder_ckpt"},
    }[stage]
    missing = {key: item for key, item in ckpt_status.items() if key in required_ckpts and not item["exists"]}
    if missing:
        raise HTTPException(status_code=400, detail={"error": "missing ckpt", "missing": missing, "ckpts": ckpt_status})
    cfg_path = out_dir / "ckpt_config.json"
    _json(cfg_path, cfg)
    images = [Path(path) for path in model_inputs["images"]]
    masks = [Path(path) for path in model_inputs["masks"]]
    object_masks = [Path(path) for path in model_inputs["object_masks"]]
    cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts/inference/reconstruct_stages.py"),
        "--stage",
        stage,
        "--pipeline-root",
        str(pipeline_root),
        "--images",
        *[str(path) for path in images],
        "--masks",
        *[str(path) for path in masks],
        "--object-masks",
        *[str(path) for path in object_masks],
        "--part-info",
        str(part_info_path),
        "--ckpt-config-json",
        str(cfg_path),
    ]
    if model_inputs.get("cond_tokens"):
        cmd.extend(["--cond-tokens", str(model_inputs["cond_tokens"])])
    if req.force:
        cmd.append("--force")
    log_path = out_dir / "run.log"
    handle = log_path.open("w", encoding="utf-8")
    handle.write(f"[cmd] {' '.join(cmd)}\n")
    handle.flush()
    env = dict(os.environ)
    env.setdefault("SPCONV_ALGO", "native")
    env.setdefault("ATTN_BACKEND", "sdpa")
    env.setdefault("SPARSE_ATTN_BACKEND", "sdpa")
    bundled_utils3d = REPO_ROOT / "sam3d_cu118_deps" / "utils3d"
    if bundled_utils3d.is_dir():
        existing_pythonpath = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = str(bundled_utils3d) + (os.pathsep + existing_pythonpath if existing_pythonpath else "")
    proc = subprocess.Popen(cmd, stdout=handle, stderr=subprocess.STDOUT, text=True, cwd=str(REPO_ROOT), env=env)
    handle.close()
    job_id = uuid.uuid4().hex[:12]
    JOBS[job_id] = proc
    JOB_META[job_id] = {
        "job_id": job_id,
        "pid": proc.pid,
        "out_dir": str(out_dir),
        "log": str(log_path),
        "cmd": cmd,
        "stage": stage,
        "run_id": SESSION_ID,
        "session_dir": str(root),
        "pipeline_root": str(pipeline_root),
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
    job_root = Path(str(meta.get("session_dir") or session_dir()))
    stage_progress = _pipeline_stage_status(job_root, str(meta["stage"]))
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
    pipeline = {}
    pipeline_root = Path(meta.get("pipeline_root", out_dir.parent))
    for stage in ("dino_ss_flow", "ss_decode", "part_prompt_seg", "slat_decode"):
        pipeline[stage] = _pipeline_stage_status(job_root, stage)
    return {
        "ok": True, **meta, "running": return_code is None, "return_code": return_code,
        "progress": stage_progress, "pipeline": pipeline, "log_tail": tail, "summary": summary, "files": files,
    }


def _kin_agent_root(root: Path) -> Path:
    return root / "kin_agent"


def _kin_motion_observation_root(root: Path) -> Path | None:
    metadata = _read_json(root / "dataset.json", {}) or {}
    if str(metadata.get("source") or "").lower() != "dataset":
        return None
    dataset_id = str(metadata.get("dataset_id") or "").strip()
    object_id = str(metadata.get("object_id") or "").strip()
    if not dataset_id or not object_id:
        return None
    candidate = DATA_ROOT / "data" / dataset_id / "renders" / object_id
    return candidate if candidate.is_dir() else None


def _kin_result_payload(root: Path) -> dict[str, Any] | None:
    kin_root = _kin_agent_root(root)
    result_path = kin_root / "kinematic_result.json"
    summary_path = root / "reconstruct" / "pipeline" / "slat_decode" / "summary.json"
    slat_status = _pipeline_stage_status(root, "slat_decode")
    if slat_status.get("state") not in {"complete", "cached"} or not summary_path.is_file():
        return None
    payload = _read_json(result_path, None)
    if not isinstance(payload, dict):
        return None
    try:
        result_summary = Path(str(payload.get("summary_path", ""))).expanduser().resolve()
        if result_summary != summary_path.resolve() or result_path.stat().st_mtime_ns < summary_path.stat().st_mtime_ns:
            return None
    except OSError:
        return None
    input_files = payload.get("input_files")
    if isinstance(input_files, list):
        for item in input_files:
            if not isinstance(item, dict):
                return None
            try:
                path = Path(str(item["path"])).expanduser().resolve()
                stat = path.stat()
            except (KeyError, OSError):
                return None
            if stat.st_size != int(item.get("size", -1)) or stat.st_mtime_ns != int(item.get("mtime_ns", -1)):
                return None
    result = dict(payload)
    body_path = Path(str(result.get("body_source_mesh", "")))
    result["body_mesh_url"] = _output_url(body_path) if body_path.is_file() and body_path.is_relative_to(WORK_ROOT) else None
    parts = []
    for raw in result.get("parts") or []:
        item = dict(raw)
        source_path = Path(str(item.get("source_mesh", "")))
        item["mesh_url"] = _output_url(source_path) if source_path.is_file() and source_path.is_relative_to(WORK_ROOT) else None
        parts.append(item)
    result["parts"] = parts
    for key in ("xml_path", "usd_path", "collision_audit_path"):
        path = Path(str(result.get(key, "")))
        result[f"{key[:-5]}_url"] = _output_url(path) if path.is_file() and path.is_relative_to(WORK_ROOT) else None
    validation = dict(result.get("validation") or {})
    for key in ("report_path", "image_path"):
        path = Path(str(validation.get(key, "")))
        validation[f"{key[:-5]}_url"] = _output_url(path) if path.is_file() and path.is_relative_to(WORK_ROOT) else None
    result["validation"] = validation
    return result


@app.get("/api/kin-agent/config")
def kin_agent_config() -> dict[str, Any]:
    root = session_dir()
    summary_path = root / "reconstruct" / "pipeline" / "slat_decode" / "summary.json"
    summary = _read_json(summary_path, None)
    slat_status = _pipeline_stage_status(root, "slat_decode")
    parts = []
    if isinstance(summary, dict):
        parts = [
            {
                "part_id": item.get("part_id"), "label": item.get("label"),
                "kind": item.get("kind"), "mesh_path": item.get("mesh_path"),
            }
            for item in summary.get("parts") or []
        ]
    body_ready = any(item.get("kind") == "body" and Path(str(item.get("mesh_path", ""))).is_file() for item in parts)
    moving_ready = any(item.get("kind") != "body" and Path(str(item.get("mesh_path", ""))).is_file() for item in parts)
    motion_observation_root = _kin_motion_observation_root(root)
    return {
        "ok": True,
        "ready": slat_status.get("state") in {"complete", "cached"} and summary_path.is_file() and body_ready and moving_ready,
        "max_iterations": 9,
        "default_iterations": 7,
        "dataset_motion_states_available": motion_observation_root is not None,
        "motion_observation_root": str(motion_observation_root) if motion_observation_root else None,
        "summary_path": str(summary_path),
        "parts": parts,
        "status": _read_json(_kin_agent_root(root) / "status.json", {"state": "not_started", "progress": 0}),
        "result": _kin_result_payload(root),
    }


@app.post("/api/kin-agent/start")
def kin_agent_start(req: KinAgentRequest) -> dict[str, Any]:
    root = session_dir()
    with SESSION_LOCK:
        _ensure_no_active_jobs()
        summary_path = root / "reconstruct" / "pipeline" / "slat_decode" / "summary.json"
        slat_status = _pipeline_stage_status(root, "slat_decode")
        if slat_status.get("state") not in {"complete", "cached"} or not summary_path.is_file():
            raise HTTPException(status_code=409, detail="Run Mesh + GS Decode before Kin Agent")
        out_dir = _kin_agent_root(root)
        dataset = _read_json(root / "dataset.json", {}) or {}
        dataset_id = str(dataset.get("dataset_id") or "").strip()
        motion_observation_root = (
            _kin_motion_observation_root(root) if req.use_dataset_motion_states else None
        )
        static_observation_root = _kin_motion_observation_root(root)
        motion_observation_value = str(motion_observation_root.resolve()) if motion_observation_root else ""
        static_observation_value = str(static_observation_root.resolve()) if static_observation_root else ""
        static_view_indices = [
            int(value) for value in dataset.get("source_view_indices") or (0, 3, 8, 11)
        ]
        cached = _kin_result_payload(root)
        if (
            cached is not None
            and cached.get("format") == "arts_gen_kin_agent_v17"
            and (cached.get("collision_audit") or {}).get("version") == "decoded_collision_audit_v2"
            and Path(str(cached.get("collision_audit_path") or "")).is_file()
            and int(cached.get("max_iterations", -1)) == int(req.max_iterations)
            and str(cached.get("dataset_id") or "") == dataset_id
            and str(cached.get("motion_observation_root") or "") == motion_observation_value
            and str(cached.get("static_observation_root") or "") == static_observation_value
            and list(cached.get("static_view_indices") or [])
            == static_view_indices
        ):
            return {
                "ok": True, "cached": True, "job_id": None,
                "run_id": SESSION_ID, "session_dir": str(root),
                "out_dir": str(_kin_agent_root(root)), "max_iterations": int(req.max_iterations),
                "result": cached,
            }
        if out_dir.exists():
            shutil.rmtree(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        cmd = [
            sys.executable, "-m", "post_process.kinematic_solver.run_kin_agent_bundle",
            "--summary-json", str(summary_path), "--out-dir", str(out_dir),
            "--max-iterations", str(int(req.max_iterations)),
        ]
        if dataset_id:
            cmd.extend(["--dataset-id", dataset_id])
        if motion_observation_root is not None:
            cmd.extend(["--motion-observation-root", str(motion_observation_root)])
        if static_observation_root is not None:
            cmd.extend(["--static-observation-root", str(static_observation_root)])
            cmd.extend(["--static-view-indices", ",".join(str(value) for value in static_view_indices)])
        log_path = out_dir / "run.log"
        handle = log_path.open("w", encoding="utf-8")
        handle.write(f"[cmd] {' '.join(cmd)}\n")
        handle.flush()
        proc = subprocess.Popen(
            cmd, stdout=handle, stderr=subprocess.STDOUT, text=True,
            cwd=str(REPO_ROOT), env=dict(os.environ),
        )
        handle.close()
        job_id = uuid.uuid4().hex[:12]
        KIN_JOBS[job_id] = proc
        meta = {
            "job_id": job_id, "pid": proc.pid, "run_id": SESSION_ID,
            "session_dir": str(root), "out_dir": str(out_dir), "log": str(log_path),
            "cmd": cmd, "max_iterations": int(req.max_iterations), "started_unix": time.time(),
            "use_dataset_motion_states": bool(req.use_dataset_motion_states),
        }
        KIN_JOB_META[job_id] = meta
        _json(root / "latest_kin_agent_job.json", meta)
        return {"ok": True, **meta}


@app.get("/api/kin-agent/status/{job_id}")
def kin_agent_status(job_id: str) -> dict[str, Any]:
    meta = KIN_JOB_META.get(job_id)
    if not meta:
        raise HTTPException(status_code=404, detail=job_id)
    proc = KIN_JOBS[job_id]
    return_code = proc.poll()
    out_dir = Path(meta["out_dir"])
    log_path = Path(meta["log"])
    log_tail = ""
    if log_path.is_file():
        log_tail = "\n".join(log_path.read_text(encoding="utf-8", errors="replace").splitlines()[-100:])
    files = [
        {
            "rel": str(path.relative_to(out_dir)), "url": _output_url(path), "size": path.stat().st_size,
        }
        for path in sorted(out_dir.rglob("*"))
        if path.is_file() and path.name != "run.log"
    ]
    return {
        "ok": True, **meta, "running": return_code is None, "return_code": return_code,
        "progress": _read_json(out_dir / "status.json", {"state": "starting", "progress": 1}),
        "result": _kin_result_payload(Path(meta["session_dir"])), "log_tail": log_tail, "files": files,
    }


@app.get("/api/kin-agent/result")
def kin_agent_result() -> dict[str, Any]:
    payload = _kin_result_payload(session_dir())
    if payload is None:
        raise HTTPException(status_code=404, detail="Kin Agent result is not available")
    return {"ok": True, **payload}


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
