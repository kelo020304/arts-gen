"""FastAPI app for Stage A — image + mask → surface voxel."""
from __future__ import annotations

import json
import os
from io import BytesIO
from pathlib import Path
from typing import Optional

import numpy as np
from fastapi import File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse

from shared.backend import (
    JobDirs,
    compute_sha,
    create_app,
    read_image_rgb,
    read_mask_bool,
    read_upload_capped,
)
from surface_voxel import SurfaceVoxelPipeline

DATA_DIR = Path(os.environ.get("WEB_DATA_DIR", "./data")).resolve()
CONFIG_PATH = Path(os.environ["SAM3D_CONFIG_PATH"]).resolve()
FRONTEND = Path(__file__).parent.parent / "frontend"

PIPELINE: SurfaceVoxelPipeline | None = None

PREVIEW_CAP = 30000


def _health_extras() -> dict:
    return {
        "stage": "surface_voxel",
        "model_loaded": PIPELINE is not None,
        "config_path": str(CONFIG_PATH),
    }


app = create_app(
    title="Stage A · Surface Voxel",
    data_dir=DATA_DIR,
    health_extras=_health_extras,
    app_frontend_dir=FRONTEND,
)


@app.on_event("startup")
async def _load_model() -> None:
    global PIPELINE
    PIPELINE = SurfaceVoxelPipeline(CONFIG_PATH, device="cuda")


@app.on_event("shutdown")
async def _unload_model() -> None:
    global PIPELINE
    if PIPELINE is not None:
        PIPELINE.unload()
        PIPELINE = None


@app.get("/")
async def index() -> FileResponse:
    return FileResponse(FRONTEND / "index.html")


@app.post("/api/run")
async def run(
    image: UploadFile = File(...),
    mask: UploadFile = File(...),
    seed: int = Form(42),
    keep_layout_aux: bool = Form(True),
    gt_voxel_npy: Optional[UploadFile] = File(None),
) -> dict:
    if PIPELINE is None:
        raise HTTPException(503, "model still loading")

    image_bytes = await read_upload_capped(image)
    mask_bytes = await read_upload_capped(mask)
    # gt_voxel does NOT participate in the cache sha: inference is independent
    # of GT. Reading the cached surface.npy + recomputing IoU is cheap.
    sha = compute_sha(
        {
            "stage": "A",
            "image": image_bytes,
            "mask": mask_bytes,
            "seed": seed,
            "keep_layout_aux": int(keep_layout_aux),
        }
    )
    job = JobDirs(DATA_DIR, sha)
    out = job.ensure_output()

    if (out / "surface.npy").exists() and (out / "pose.json").exists():
        _ensure_preview(out, seed)
    else:
        inp = job.ensure_input()
        image_suffix = Path(image.filename or "image.png").suffix or ".png"
        mask_suffix = Path(mask.filename or "mask.png").suffix or ".png"
        (inp / f"image{image_suffix}").write_bytes(image_bytes)
        (inp / f"mask{mask_suffix}").write_bytes(mask_bytes)

        img_np = read_image_rgb(BytesIO(image_bytes))
        mask_np = read_mask_bool(BytesIO(mask_bytes))

        voxel = PIPELINE(img_np, mask_np, seed=seed, keep_layout_aux=keep_layout_aux)
        voxel.save(out)
        _write_preview(out, seed)

    response = _build_response(sha, out)

    if gt_voxel_npy is not None:
        gt_bytes = await read_upload_capped(gt_voxel_npy)
        try:
            gt_arr = np.load(BytesIO(gt_bytes), allow_pickle=False)
        except (OSError, ValueError) as exc:
            raise HTTPException(400, f"gt_voxel_npy not a valid .npy: {exc}") from exc
        if gt_arr.ndim != 2 or gt_arr.shape[1] != 3:
            raise HTTPException(
                400, f"gt_voxel_npy must be (N, 3), got shape {gt_arr.shape}"
            )
        gt_arr = gt_arr.astype(np.int64, copy=False)
        if gt_arr.size and (gt_arr.min() < 0 or gt_arr.max() > 63):
            raise HTTPException(
                400,
                f"gt_voxel_npy values must lie in [0, 63], got [{gt_arr.min()}, {gt_arr.max()}]",
            )
        pred = np.load(out / "surface.npy")
        comparison = _compute_comparison(pred, gt_arr, seed)
        response["compare"] = comparison

    return response


def _write_preview(out: Path, seed: int) -> None:
    """Write deterministic downsampled coords preview JSON for Three.js."""
    coords = np.load(out / "surface.npy")
    n = int(coords.shape[0])
    if n > PREVIEW_CAP:
        rng = np.random.default_rng(seed)
        idx = rng.choice(n, size=PREVIEW_CAP, replace=False)
        sampled = coords[idx]
    else:
        sampled = coords
    payload = {
        "coords": sampled.astype(int).tolist(),
        "grid_size": 64,
        "count": int(sampled.shape[0]),
    }
    (out / "coords_preview.json").write_text(json.dumps(payload, separators=(",", ":")))


def _ensure_preview(out: Path, seed: int) -> None:
    if not (out / "coords_preview.json").exists():
        _write_preview(out, seed)


def _build_response(sha: str, out: Path) -> dict:
    coords = np.load(out / "surface.npy")
    pose = json.loads((out / "pose.json").read_text())
    files = {
        "surface_npy": f"/api/jobs/{sha}/surface.npy",
        "pose_json": f"/api/jobs/{sha}/pose.json",
    }
    if (out / "pointmap_unnorm.npy").exists():
        files["pointmap_unnorm_npy"] = f"/api/jobs/{sha}/pointmap_unnorm.npy"
    return {
        "sha": sha,
        "files": files,
        "coords_count": int(coords.shape[0]),
        "coords_preview_url": f"/api/jobs/{sha}/coords_preview.json",
        "pose": pose,
    }


def _compute_comparison(pred: np.ndarray, gt: np.ndarray, seed: int) -> dict:
    """Pred vs GT IoU + per-bucket downsampled coords for the 3-color preview."""
    pred_set = {tuple(int(v) for v in row) for row in pred.tolist()}
    gt_set   = {tuple(int(v) for v in row) for row in gt.tolist()}
    tp = pred_set & gt_set
    fp = pred_set - gt_set
    fn = gt_set - pred_set
    union = pred_set | gt_set
    iou = len(tp) / max(1, len(union))

    rng = np.random.default_rng(seed)
    cap_per_layer = PREVIEW_CAP * 2 // 3

    def take(s: set) -> list[list[int]]:
        items = [list(t) for t in s]
        if len(items) > cap_per_layer:
            idx = rng.choice(len(items), size=cap_per_layer, replace=False)
            items = [items[i] for i in idx]
        return items

    return {
        "iou": float(iou),
        "pred_count": len(pred_set),
        "gt_count": len(gt_set),
        "tp_count": len(tp),
        "fp_count": len(fp),
        "fn_count": len(fn),
        "grid_size": 64,
        "layers": {
            "tp": {"coords": take(tp), "count": len(tp), "color": "#22c55e"},
            "fp": {"coords": take(fp), "count": len(fp), "color": "#ef4444"},
            "fn": {"coords": take(fn), "count": len(fn), "color": "#3b82f6"},
        },
    }
