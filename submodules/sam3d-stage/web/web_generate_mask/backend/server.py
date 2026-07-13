"""FastAPI app for interactive box-prompted segmentation (SAM 3-backed)."""
from __future__ import annotations

import json
import os
from io import BytesIO
from pathlib import Path

from fastapi import File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from PIL import Image
from pydantic import BaseModel
from starlette.responses import StreamingResponse

from shared.backend import (
    JobDirs,
    compute_sha,
    create_app,
    read_upload_capped,
)
from mask import BoxPrompt, MaskOutput, MaskPipeline, SessionManager

DATA_DIR = Path(os.environ.get("WEB_DATA_DIR", "./data")).resolve()
CKPT_PATH = Path(os.environ["SAM3_CKPT_PATH"]).resolve()
FRONTEND = Path(__file__).parent.parent / "frontend"
MAX_IMAGE_DIM = 1024  # resize uploaded images so SAM 3 runs fast and consistent

PIPELINE: MaskPipeline | None = None
SESSIONS = SessionManager()


def _health_extras() -> dict:
    return {
        "stage": "generate_mask",
        "model_loaded": PIPELINE is not None,
        "ckpt_path": str(CKPT_PATH),
        "active_sessions": len(SESSIONS),
    }


app = create_app(
    title="Generate Mask · SAM 3",
    data_dir=DATA_DIR,
    health_extras=_health_extras,
    app_frontend_dir=FRONTEND,
)


@app.on_event("startup")
async def _load_model() -> None:
    global PIPELINE
    PIPELINE = MaskPipeline(CKPT_PATH, device="cuda")


@app.on_event("shutdown")
async def _unload_model() -> None:
    global PIPELINE
    if PIPELINE is not None:
        PIPELINE.unload()
        PIPELINE = None


@app.get("/")
async def index() -> FileResponse:
    return FileResponse(FRONTEND / "index.html")


def _points_to_pixels(pts: list[list[float]], img_w: int, img_h: int) -> list[tuple[float, float]]:
    out: list[tuple[float, float]] = []
    for p in pts:
        if len(p) != 2:
            continue
        out.append((float(p[0]) * img_w, float(p[1]) * img_h))
    return out


def _points_sha_key(pts: list[list[float]]) -> str:
    # Order-insensitive, fixed precision so cache hits are stable.
    keys = sorted((round(p[0], 6), round(p[1], 6)) for p in pts if len(p) == 2)
    return ";".join(f"{x:.6f},{y:.6f}" for x, y in keys)


class BoxBody(BaseModel):
    cx: float
    cy: float
    w: float
    h: float
    # Each element is [nx, ny] in normalized [0, 1] image space.
    # SAM 3 labels: pos = include this region, neg = exclude this region.
    pos_points: list[list[float]] = []
    neg_points: list[list[float]] = []

    def as_prompt(self) -> BoxPrompt:
        return BoxPrompt(cx=self.cx, cy=self.cy, w=self.w, h=self.h)


@app.post("/api/upload")
async def upload(image: UploadFile = File(...)) -> dict:
    if PIPELINE is None:
        raise HTTPException(503, "model still loading")

    image_bytes = await read_upload_capped(image)
    try:
        pil = Image.open(BytesIO(image_bytes)).convert("RGB")
    except (Image.UnidentifiedImageError, OSError) as exc:
        raise HTTPException(415, f"unsupported or corrupt image: {exc}") from exc
    pil.thumbnail((MAX_IMAGE_DIM, MAX_IMAGE_DIM))
    # Re-serialize the resized image so the sha hashes exactly what SAM 3 saw.
    buf = BytesIO()
    pil.save(buf, format="PNG")
    resized_bytes = buf.getvalue()

    state = PIPELINE.embed(pil)
    sid = SESSIONS.create(pil, resized_bytes, state)

    return {
        "session_id": sid,
        "image_url": f"/api/sessions/{sid}/image",
        "width": pil.width,
        "height": pil.height,
    }


@app.get("/api/sessions/{sid}/image")
async def session_image(sid: str) -> StreamingResponse:
    entry = SESSIONS.get(sid)
    if entry is None:
        raise HTTPException(404, "session not found")
    buf = BytesIO()
    entry.image.save(buf, format="PNG")
    buf.seek(0)
    return StreamingResponse(buf, media_type="image/png")


@app.post("/api/sessions/{sid}/predict")
async def predict(sid: str, box: BoxBody) -> dict:
    if PIPELINE is None:
        raise HTTPException(503, "model still loading")
    entry = SESSIONS.get(sid)
    if entry is None:
        raise HTTPException(404, "session not found or expired")

    prompt = box.as_prompt()
    x0, y0, x1, y1 = prompt.to_xyxy_pixels(entry.width, entry.height)
    pos_px = _points_to_pixels(box.pos_points, entry.width, entry.height)
    neg_px = _points_to_pixels(box.neg_points, entry.width, entry.height)
    out: MaskOutput = PIPELINE.predict_box(
        entry.state, x0, y0, x1, y1,
        pos_points=pos_px or None,
        neg_points=neg_px or None,
    )
    return {
        "mask_png_base64": out.to_base64_png(),
        "score": out.score,
    }


@app.post("/api/sessions/{sid}/save")
async def save(sid: str, box: BoxBody) -> dict:
    if PIPELINE is None:
        raise HTTPException(503, "model still loading")
    entry = SESSIONS.get(sid)
    if entry is None:
        raise HTTPException(404, "session not found or expired")

    prompt = box.as_prompt()
    sha = compute_sha({
        "stage": "generate_mask",
        "image": entry.image_bytes,
        "box": f"{prompt.cx:.6f},{prompt.cy:.6f},{prompt.w:.6f},{prompt.h:.6f}",
        "pos_points": _points_sha_key(box.pos_points),
        "neg_points": _points_sha_key(box.neg_points),
    })
    job = JobDirs(DATA_DIR, sha)
    out_dir = job.ensure_output()
    inp_dir = job.ensure_input()

    # Always write input image (cheap, ~50 KB resized PNG; ensures cache dirs
    # are self-contained even when /save hits a previously-cached output).
    if not (inp_dir / "image.png").exists():
        (inp_dir / "image.png").write_bytes(entry.image_bytes)

    if (out_dir / "mask.png").exists():
        return _build_response(sha, out_dir)

    x0, y0, x1, y1 = prompt.to_xyxy_pixels(entry.width, entry.height)
    pos_px = _points_to_pixels(box.pos_points, entry.width, entry.height)
    neg_px = _points_to_pixels(box.neg_points, entry.width, entry.height)
    mask_out: MaskOutput = PIPELINE.predict_box(
        entry.state, x0, y0, x1, y1,
        pos_points=pos_px or None,
        neg_points=neg_px or None,
    )
    mask_out.save_png(out_dir / "mask.png")
    (out_dir / "prompt.json").write_text(
        json.dumps({
            "box": prompt.as_list(),
            "pos_points": box.pos_points,
            "neg_points": box.neg_points,
            "score": float(mask_out.score),
        })
    )

    return _build_response(sha, out_dir, mask_out)


@app.delete("/api/sessions/{sid}", status_code=204)
async def delete_session(sid: str) -> None:
    SESSIONS.delete(sid)


def _build_response(
    sha: str, out_dir: Path, mask_out: MaskOutput | None = None
) -> dict:
    files = {"mask_png": f"/api/jobs/{sha}/mask.png"}
    if (out_dir / "prompt.json").exists():
        files["prompt_json"] = f"/api/jobs/{sha}/prompt.json"

    if mask_out is not None:
        score = float(mask_out.score)
    else:
        # Cache hit: recover score from prompt.json so the response shape is stable.
        score = None
        pj = out_dir / "prompt.json"
        if pj.exists():
            try:
                score = float(json.loads(pj.read_text()).get("score"))
            except (ValueError, TypeError, json.JSONDecodeError):
                score = None
    return {"sha": sha, "files": files, "score": score}
