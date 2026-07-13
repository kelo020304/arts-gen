"""FastAPI app for Stage B - voxel + image + mask -> splat + mesh."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from fastapi import File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse

from io import BytesIO

from shared.backend import (
    JobDirs,
    compute_sha,
    create_app,
    read_image_rgb,
    read_mask_bool,
    read_upload_capped,
)
from texture import TexturePipeline, align_sam3d_outputs

DATA_DIR = Path(os.environ.get("WEB_DATA_DIR", "./data")).resolve()
CONFIG_PATH = Path(os.environ["SAM3D_CONFIG_PATH"]).resolve()
FRONTEND = Path(__file__).parent.parent / "frontend"

PIPELINE: TexturePipeline | None = None


def _health_extras() -> dict:
    return {
        "stage": "texture",
        "model_loaded": PIPELINE is not None,
        "config_path": str(CONFIG_PATH),
    }


app = create_app(
    title="Stage B - Texture",
    data_dir=DATA_DIR,
    health_extras=_health_extras,
    app_frontend_dir=FRONTEND,
)


@app.on_event("startup")
async def _load_model() -> None:
    global PIPELINE
    PIPELINE = TexturePipeline(
        CONFIG_PATH,
        device="cuda",
        load_mesh_decoder=True,
        load_gs4_decoder=False,
    )


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
    surface_npy: UploadFile = File(...),
    image: UploadFile = File(...),
    mask: UploadFile = File(...),
    pose_json: Optional[UploadFile] = File(None),
    pointmap_unnorm_npy: Optional[UploadFile] = File(None),
    seed: Optional[int] = Form(None),
    formats: str = Form("gaussian,mesh"),
    with_layout_postprocess: bool = Form(False),
) -> dict:
    if PIPELINE is None:
        raise HTTPException(503, "model still loading")

    surface_bytes = await read_upload_capped(surface_npy)
    image_bytes = await read_upload_capped(image)
    mask_bytes = await read_upload_capped(mask)
    pose_bytes = await read_upload_capped(pose_json) if pose_json else b""
    pm_bytes = await read_upload_capped(pointmap_unnorm_npy) if pointmap_unnorm_npy else b""

    if not pose_bytes:
        # Synthesize a minimal pose.json. Stage B only reads `seed` from it
        # (overridable via the seed form field). The pose/intrinsics fields
        # are placeholders; intrinsics is only consulted when layout-postopt is on.
        synth_seed = seed if seed is not None else 42
        pose_bytes = (
            '{"rotation":[[[1.0,0.0,0.0,0.0]]],'
            '"translation":[[0.0,0.0,0.0]],'
            '"scale":[[1.0,1.0,1.0]],'
            '"intrinsics":[[1.0,0.0,0.0],[0.0,1.0,0.0],[0.0,0.0,1.0]],'
            '"downsample_factor":1,'
            f'"seed":{int(synth_seed)}'
            '}'
        ).encode("utf-8")

    fmt_tuple = tuple(f.strip() for f in formats.split(",") if f.strip())
    if not fmt_tuple:
        raise HTTPException(400, "at least one format required (gaussian|mesh|gaussian_4)")

    if with_layout_postprocess and not pm_bytes:
        raise HTTPException(
            400, "layout postprocess requires pointmap_unnorm.npy (upload it)"
        )

    sha = compute_sha({
        # "B_a2" = Stage B, alignment v2 (rotate splat Z-up -> Y-up to match
        # the already-Y-up mesh.glb). Bumping invalidates pre-alignment caches
        # and the brief v1 attempt that double-rotated the mesh.
        "stage": "B_a2",
        "surface": surface_bytes,
        "pose": pose_bytes,
        "image": image_bytes,
        "mask": mask_bytes,
        "pm": pm_bytes,
        "seed": seed if seed is not None else -1,
        "formats": ",".join(sorted(fmt_tuple)),
        "postopt": int(with_layout_postprocess),
    })
    job = JobDirs(DATA_DIR, sha)
    out = job.ensure_output()

    expected = []
    if "gaussian" in fmt_tuple or "gaussian_4" in fmt_tuple:
        expected.append(out / "splat.ply")
    if "mesh" in fmt_tuple:
        expected.append(out / "mesh.glb")
    if expected and all(p.exists() for p in expected):
        return _build_response(sha, out, fmt_tuple)

    inp = job.ensure_input()
    (inp / "surface.npy").write_bytes(surface_bytes)
    (inp / "pose.json").write_bytes(pose_bytes)
    image_path = (inp / "image").with_suffix(Path(image.filename or "image.png").suffix or ".png")
    image_path.write_bytes(image_bytes)
    mask_path = (inp / "mask").with_suffix(Path(mask.filename or "mask.png").suffix or ".png")
    mask_path.write_bytes(mask_bytes)
    if pm_bytes:
        (inp / "pointmap_unnorm.npy").write_bytes(pm_bytes)

    img_np = read_image_rgb(BytesIO(image_bytes))
    mask_np = read_mask_bool(BytesIO(mask_bytes))

    appearance = PIPELINE(
        inp,
        img_np,
        mask_np,
        seed=seed,
        formats=fmt_tuple,
        with_layout_postprocess=with_layout_postprocess,
    )
    save_mesh = "mesh" in fmt_tuple
    appearance.save(out, save_mesh=save_mesh)

    # Bring the just-saved splat.ply into the same Y-up frame as the GLB.
    # (Safe to call on fresh outputs; calling twice would double-rotate, but
    # the cache-hit branch above short-circuits before we get here.)
    align_sam3d_outputs(out)

    # Persist num_gaussians sidecar so cache hits can re-report it.
    if appearance.num_gaussians is not None:
        (out / "meta.json").write_text(
            f'{{"num_gaussians": {int(appearance.num_gaussians)}}}'
        )

    return _build_response(sha, out, fmt_tuple, appearance)


def _build_response(
    sha: str,
    out: Path,
    fmt_tuple: tuple[str, ...],
    appearance=None,
) -> dict:
    files: dict[str, str] = {}
    if (out / "splat.ply").exists():
        files["splat_ply"] = f"/api/jobs/{sha}/splat.ply"
    if (out / "mesh.glb").exists():
        files["mesh_glb"] = f"/api/jobs/{sha}/mesh.glb"

    if appearance is not None:
        num_gaussians = appearance.num_gaussians
    elif (out / "meta.json").exists():
        import json
        num_gaussians = json.loads((out / "meta.json").read_text()).get("num_gaussians")
    else:
        num_gaussians = None

    return {
        "sha": sha,
        "files": files,
        "num_gaussians": num_gaussians,
        "formats": list(fmt_tuple),
    }
