#!/usr/bin/env python3
"""SLat encoder -> decoder textured round-trip on one dataset voxel/image case."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import torch
import trimesh
import trimesh.visual
from PIL import Image, ImageDraw
from safetensors.torch import load_file


matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from mpl_toolkits.mplot3d.art3d import Poly3DCollection  # noqa: E402


REPO_ROOT = Path(__file__).resolve().parents[2]
TRELLIS_ROOT = REPO_ROOT / "TRELLIS-arts"
for item in (str(REPO_ROOT), str(TRELLIS_ROOT)):
    if item not in sys.path:
        sys.path.insert(0, item)

os.environ.setdefault("SPCONV_ALGO", "native")
os.environ.setdefault("ATTN_BACKEND", "sdpa")
os.environ.setdefault("SPARSE_ATTN_BACKEND", "sdpa")

import inference  # noqa: E402
from scripts.tools.roundtrip.trellis_full_voxel_mesh_roundtrip import (  # noqa: E402
    load_camera_matrices,
    load_surface,
    make_sparse,
    project_features,
    require_dir,
    require_file,
)


def _restore_trellis_renderer_package() -> None:
    """inference.py registers light trellis stubs; official bake needs real renderers."""
    for name in list(sys.modules):
        if name == "trellis.renderers" or name.startswith("trellis.renderers."):
            sys.modules.pop(name, None)
    import trellis.renderers  # noqa: F401


_restore_trellis_renderer_package()
from trellis.utils import postprocessing_utils, render_utils  # noqa: E402


DEFAULT_DATA_ROOT = Path("/mnt/robot-data-lab/jzh/art-gen/data/phyx-verse")
DEFAULT_OUT_DIR = Path("/mnt/robot-data-lab/jzh/art-gen-output/EE-eval/0617-encoder-test")
DEFAULT_OBJECT_ID = "004d1e9e13934e319094151a4fad823f"
DEFAULT_ENCODER_CKPT = REPO_ROOT / "pretrained/TRELLIS-image-large/ckpts/slat_enc_swin8_B_64l8_fp16.safetensors"
DEFAULT_MESH_DECODER_CKPT = (
    REPO_ROOT / "pretrained/TRELLIS-image-large/ckpts/slat_dec_mesh_swin8_B_64l8m256c_fp16.safetensors"
)
DEFAULT_GAUSSIAN_DECODER_CKPT = (
    REPO_ROOT / "pretrained/TRELLIS-image-large/ckpts/slat_dec_gs_swin8_B_64l8gs32_fp16.safetensors"
)
Z_UP_TO_Y_UP = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=np.float32)
EXPECTED_PATCH_GRID = 37
EXPECTED_FEATURE_DIM = 1024


def _axis_transform(vertices: np.ndarray, axis_mode: str) -> np.ndarray:
    if axis_mode == "zup_yfront":
        return vertices
    if axis_mode == "yup_zfront":
        return vertices @ Z_UP_TO_Y_UP
    raise ValueError(f"unknown axis_mode={axis_mode!r}")


def _load_patchtokens(token_path: Path, view_indices: list[int]) -> torch.Tensor:
    with np.load(require_file(token_path, "DINOv2 tokens")) as data:
        if "patchtokens" in data.files:
            patch = data["patchtokens"]
            if patch.ndim != 4 or patch.shape[1:] != (EXPECTED_FEATURE_DIM, EXPECTED_PATCH_GRID, EXPECTED_PATCH_GRID):
                raise ValueError(f"{token_path}: bad patchtokens shape {patch.shape}")
        elif "tokens" in data.files:
            tokens = data["tokens"]
            if tokens.ndim != 3 or tokens.shape[0] != 12 or tokens.shape[-1] != EXPECTED_FEATURE_DIM:
                raise ValueError(f"{token_path}: bad tokens shape {tokens.shape}")
            patch_start = tokens.shape[1] - EXPECTED_PATCH_GRID * EXPECTED_PATCH_GRID
            if patch_start < 1:
                raise ValueError(f"{token_path}: cannot infer patch tokens from shape {tokens.shape}")
            patch = tokens[:, patch_start:, :].transpose(0, 2, 1).reshape(
                tokens.shape[0],
                EXPECTED_FEATURE_DIM,
                EXPECTED_PATCH_GRID,
                EXPECTED_PATCH_GRID,
            )
        else:
            raise KeyError(f"{token_path}: expected key 'patchtokens' or 'tokens', got {data.files}")
    selected = np.asarray(view_indices, dtype=np.int64)
    if selected.ndim != 1 or len(selected) == 0:
        raise ValueError("view_indices must be non-empty")
    if int(selected.min()) < 0 or int(selected.max()) >= patch.shape[0]:
        raise ValueError(f"{token_path}: view_indices out of range for shape {patch.shape}: {view_indices}")
    return torch.from_numpy(patch[selected].astype(np.float32, copy=False)).cuda()


def _load_encoder(ckpt: Path):
    from trellis.models.structured_latent_vae.encoder import SLatEncoder

    ckpt = require_file(ckpt, "SLat encoder ckpt")
    cfg_path = require_file(ckpt.with_suffix(".json"), "SLat encoder config")
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    if cfg.get("name") != "SLatEncoder":
        raise ValueError(f"{cfg_path}: expected SLatEncoder, got {cfg.get('name')!r}")
    model = SLatEncoder(**cfg["args"]).cuda().eval()
    model.load_state_dict(load_file(str(ckpt), device="cuda"), strict=True)
    for param in model.parameters():
        param.requires_grad_(False)
    print(f"[ckpt] encoder={ckpt}", flush=True)
    return model


def _mesh_arrays(mesh: Any) -> tuple[np.ndarray, np.ndarray]:
    vertices = mesh.vertices.detach().float().cpu().numpy() if torch.is_tensor(mesh.vertices) else np.asarray(mesh.vertices)
    faces = mesh.faces.detach().long().cpu().numpy() if torch.is_tensor(mesh.faces) else np.asarray(mesh.faces)
    return vertices.astype(np.float32, copy=False).reshape(-1, 3), faces.astype(np.int64, copy=False).reshape(-1, 3)


def _official_fast_textured_mesh(
    gaussian: Any,
    mesh: Any,
    *,
    simplify: float,
    texture_size: int,
    bake_resolution: int,
    bake_views: int,
    bake_mode: str,
    axis_mode: str,
    verbose: bool,
) -> trimesh.Trimesh:
    vertices, faces = _mesh_arrays(mesh)
    vertices, faces = postprocessing_utils.postprocess_mesh(
        vertices,
        faces,
        simplify=float(simplify) > 0,
        simplify_ratio=float(simplify),
        fill_holes=False,
        verbose=verbose,
    )
    vertices, faces, uvs = postprocessing_utils.parametrize_mesh(vertices, faces)
    observations, extrinsics, intrinsics = render_utils.render_multiview(
        gaussian,
        resolution=int(bake_resolution),
        nviews=int(bake_views),
    )
    masks = [np.any(observation > 0, axis=-1) for observation in observations]
    texture = postprocessing_utils.bake_texture(
        vertices,
        faces,
        uvs,
        observations,
        masks,
        [extr.detach().cpu().numpy() for extr in extrinsics],
        [intr.detach().cpu().numpy() for intr in intrinsics],
        texture_size=int(texture_size),
        mode=str(bake_mode),
        lambda_tv=0.01,
        verbose=verbose,
    )
    material = trimesh.visual.material.PBRMaterial(
        roughnessFactor=1.0,
        baseColorTexture=Image.fromarray(texture),
        baseColorFactor=np.array([255, 255, 255, 255], dtype=np.uint8),
    )
    return trimesh.Trimesh(
        vertices=_axis_transform(vertices, axis_mode),
        faces=faces,
        visual=trimesh.visual.TextureVisuals(uv=uvs, material=material),
        process=False,
    )


def _load_textured_triangles(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    loaded = trimesh.load(str(path), force="scene", process=False)
    geometries = [loaded] if isinstance(loaded, trimesh.Trimesh) else list(loaded.geometry.values())
    tris: list[np.ndarray] = []
    colors: list[np.ndarray] = []
    vertices_all: list[np.ndarray] = []
    for geom in geometries:
        if not isinstance(geom, trimesh.Trimesh) or len(geom.vertices) == 0 or len(geom.faces) == 0:
            continue
        vertices = np.asarray(geom.vertices, dtype=np.float32)
        faces = np.asarray(geom.faces, dtype=np.int64)
        visual = getattr(geom, "visual", None)
        material = getattr(visual, "material", None)
        texture = getattr(material, "baseColorTexture", None) or getattr(material, "image", None)
        uv = getattr(visual, "uv", None)
        if texture is not None and uv is not None and len(uv) == len(vertices):
            image = np.asarray(texture.convert("RGB") if hasattr(texture, "convert") else Image.fromarray(np.asarray(texture)).convert("RGB"))
            h, w = image.shape[:2]
            face_uv = np.asarray(uv, dtype=np.float32)[faces].mean(axis=1)
            px = np.clip((face_uv[:, 0] * (w - 1)).round().astype(np.int64), 0, w - 1)
            py = np.clip(((1.0 - face_uv[:, 1]) * (h - 1)).round().astype(np.int64), 0, h - 1)
            face_colors = image[py, px].astype(np.float32) / 255.0
        else:
            face_colors = np.full((len(faces), 3), 0.72, dtype=np.float32)
        tris.append(vertices[faces])
        colors.append(face_colors)
        vertices_all.append(vertices)
    if not tris:
        raise ValueError(f"{path}: no mesh triangles")
    return np.concatenate(tris, axis=0), np.concatenate(colors, axis=0), np.concatenate(vertices_all, axis=0)


def _render_textured_glb(path: Path, out_path: Path, *, title: str, resolution: int, azim: float, elev: float) -> Path:
    tri, face_colors, vertices = _load_textured_triangles(path)
    max_faces = 80000
    if len(tri) > max_faces:
        rng = np.random.default_rng(0)
        keep = np.sort(rng.choice(len(tri), max_faces, replace=False))
        tri = tri[keep]
        face_colors = face_colors[keep]
    lo = vertices.min(axis=0)
    hi = vertices.max(axis=0)
    center = (lo + hi) / 2.0
    span = max(float(np.max(hi - lo)), 1e-3)

    fig = plt.figure(figsize=(resolution / 100.0, resolution / 100.0), dpi=100)
    ax = fig.add_subplot(111, projection="3d")
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    ax.set_box_aspect((1, 1, 1))
    ax.view_init(elev=float(elev), azim=float(azim))
    collection = Poly3DCollection(
        tri,
        facecolors=np.clip(face_colors, 0, 1),
        edgecolors=(0.03, 0.03, 0.03, 0.04),
        linewidths=0.006,
    )
    collection.set_zsort("average")
    ax.add_collection3d(collection)
    ax.set_xlim(center[0] - span / 2.0, center[0] + span / 2.0)
    ax.set_ylim(center[1] - span / 2.0, center[1] + span / 2.0)
    ax.set_zlim(center[2] - span / 2.0, center[2] + span / 2.0)
    ax.set_axis_off()
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    fig.canvas.draw()
    arr = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()
    plt.close(fig)
    image = Image.fromarray(arr).convert("RGB")
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, image.width, 27), fill=(255, 255, 255))
    draw.text((6, 7), title[:56], fill=(20, 20, 20))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(out_path)
    return out_path


def _make_input_panel(data_root: Path, object_id: str, angle_idx: int, view_indices: list[int], out_path: Path) -> Path:
    thumb = 220
    label_h = 26
    pad = 8
    canvas = Image.new("RGB", (len(view_indices) * thumb + (len(view_indices) + 1) * pad, thumb + label_h + 2 * pad), (245, 245, 245))
    draw = ImageDraw.Draw(canvas)
    for idx, view_idx in enumerate(view_indices):
        src = data_root / "renders" / object_id / f"angle_{angle_idx}" / "rgb" / f"view_{view_idx}.png"
        image = Image.open(require_file(src, f"rgb view {view_idx}")).convert("RGB")
        image.thumbnail((thumb, thumb), Image.Resampling.LANCZOS)
        tile = Image.new("RGB", (thumb, thumb), (255, 255, 255))
        tile.paste(image, ((thumb - image.width) // 2, (thumb - image.height) // 2))
        x = pad + idx * (thumb + pad)
        draw.text((x + 6, pad + 5), f"view_{view_idx}", fill=(20, 20, 20))
        canvas.paste(tile, (x, pad + label_h))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)
    return out_path


def _texture_stats(path: Path) -> dict[str, Any]:
    loaded = trimesh.load(str(path), force="scene", process=False)
    geometries = [loaded] if isinstance(loaded, trimesh.Trimesh) else list(loaded.geometry.values())
    pixels: list[np.ndarray] = []
    for geom in geometries:
        material = getattr(getattr(geom, "visual", None), "material", None)
        texture = getattr(material, "baseColorTexture", None) or getattr(material, "image", None)
        if texture is not None:
            image = np.asarray(texture.convert("RGB") if hasattr(texture, "convert") else Image.fromarray(np.asarray(texture)).convert("RGB"))
            pixels.append(image.reshape(-1, 3))
    if not pixels:
        return {"has_texture": False}
    arr = np.concatenate(pixels, axis=0)
    sample = arr[:: max(1, len(arr) // 10000)]
    return {
        "has_texture": True,
        "mean": [float(x) for x in arr.mean(axis=0).tolist()],
        "std": [float(x) for x in arr.std(axis=0).tolist()],
        "unique_sample": int(len(np.unique(sample, axis=0))),
    }


def run(args: argparse.Namespace) -> None:
    data_root = require_dir(args.data_root, "data root")
    view_indices = list(args.view_indices)
    view_tag = "views_" + "_".join(str(x) for x in view_indices)
    bake_tag = f"bake_{args.bake_mode}_tex{int(args.texture_size)}"
    case_dir = args.out_dir.resolve() / f"{args.object_id}-{int(args.angle_idx)}" / view_tag / args.axis_mode / bake_tag
    case_dir.mkdir(parents=True, exist_ok=True)
    for path, label in (
        (args.encoder_ckpt, "SLat encoder ckpt"),
        (args.mesh_decoder_ckpt, "SLat mesh decoder ckpt"),
        (args.gaussian_decoder_ckpt, "SLat gaussian decoder ckpt"),
    ):
        require_file(path, label)
        require_file(path.with_suffix(".json"), f"{label} config")

    coords = load_surface(data_root, args.object_id, int(args.angle_idx))
    token_path = data_root / "reconstruction" / "dinov2_tokens" / args.object_id / f"angle_{int(args.angle_idx)}" / "tokens.npz"
    camera_path = data_root / "renders" / args.object_id / f"angle_{int(args.angle_idx)}" / "camera_transforms.json"
    patchtokens = _load_patchtokens(token_path, view_indices)
    extrinsics, intrinsics = load_camera_matrices(camera_path, view_indices)
    feats = project_features(coords, patchtokens, extrinsics, intrinsics)

    encoder = _load_encoder(args.encoder_ckpt)
    sparse = make_sparse(coords, feats)
    with torch.no_grad():
        slat = encoder(sparse, sample_posterior=False)
    if not torch.isfinite(slat.feats).all():
        raise RuntimeError("encoder returned NaN/Inf SLat feats")

    np.savez_compressed(
        case_dir / "encoder_slat_raw.npz",
        coords=slat.coords.detach().cpu().numpy().astype(np.int32),
        feats=slat.feats.detach().float().cpu().numpy().astype(np.float32),
        normalized=np.array(False),
        view_indices=np.asarray(view_indices, dtype=np.int32),
    )

    decoded = inference.decode_slat_assets(
        slat,
        gaussian_decoder_ckpt=str(args.gaussian_decoder_ckpt.resolve()),
        mesh_decoder_ckpt=str(args.mesh_decoder_ckpt.resolve()),
        slat_is_normalized=False,
    )
    gaussian = decoded.get("gaussian")
    mesh = decoded.get("mesh")
    if gaussian is None:
        raise RuntimeError("gaussian decoder returned None")
    if mesh is None or not getattr(mesh, "success", True):
        raise RuntimeError("mesh decoder failed")

    textured = _official_fast_textured_mesh(
        gaussian,
        mesh,
        simplify=float(args.mesh_simplify),
        texture_size=int(args.texture_size),
        bake_resolution=int(args.bake_resolution),
        bake_views=int(args.bake_views),
        bake_mode=str(args.bake_mode),
        axis_mode=str(args.axis_mode),
        verbose=bool(args.verbose_bake),
    )
    glb_path = case_dir / "encoder_textured.glb"
    textured.export(str(glb_path))
    render_path = _render_textured_glb(
        glb_path,
        case_dir / "encoder_textured_preview.png",
        title="SLat encoder -> decoder textured mesh",
        resolution=int(args.render_resolution),
        azim=float(args.view_azim),
        elev=float(args.view_elev),
    )
    input_panel = _make_input_panel(data_root, args.object_id, int(args.angle_idx), view_indices, case_dir / "input_views.png")
    stats = _texture_stats(glb_path)
    report = {
        "status": "done",
        "object_id": args.object_id,
        "angle_idx": int(args.angle_idx),
        "data_root": str(data_root),
        "view_indices": view_indices,
        "surface_voxels": int(coords.shape[0]),
        "projected_feature_shape": list(feats.shape),
        "slat_rows": int(slat.feats.shape[0]),
        "slat_feat_range": [float(slat.feats.min().item()), float(slat.feats.max().item())],
        "slat_is_normalized": False,
        "encoder_ckpt": str(args.encoder_ckpt.resolve()),
        "mesh_decoder_ckpt": str(args.mesh_decoder_ckpt.resolve()),
        "gaussian_decoder_ckpt": str(args.gaussian_decoder_ckpt.resolve()),
        "token_path": str(token_path.resolve()),
        "camera_path": str(camera_path.resolve()),
        "textured_glb": str(glb_path.resolve()),
        "preview": str(render_path.resolve()),
        "input_views": str(input_panel.resolve()),
        "texture_bake_backend": f"TRELLIS postprocessing_utils.bake_texture(mode={args.bake_mode},nvdiffrast)",
        "axis_mode": args.axis_mode,
        "texture_stats": stats,
    }
    (case_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SLat encoder-decoder textured round-trip diagnostic.")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--object-id", default=DEFAULT_OBJECT_ID)
    parser.add_argument("--angle-idx", type=int, default=0)
    parser.add_argument("--view-indices", type=int, nargs="+", default=[0, 4, 6, 11])
    parser.add_argument("--encoder-ckpt", type=Path, default=DEFAULT_ENCODER_CKPT)
    parser.add_argument("--mesh-decoder-ckpt", type=Path, default=DEFAULT_MESH_DECODER_CKPT)
    parser.add_argument("--gaussian-decoder-ckpt", type=Path, default=DEFAULT_GAUSSIAN_DECODER_CKPT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--texture-size", type=int, default=512)
    parser.add_argument("--bake-resolution", type=int, default=512)
    parser.add_argument("--bake-views", type=int, default=24)
    parser.add_argument("--bake-mode", choices=("fast", "opt"), default="fast")
    parser.add_argument("--axis-mode", choices=("zup_yfront", "yup_zfront"), default="zup_yfront")
    parser.add_argument("--mesh-simplify", type=float, default=0.88)
    parser.add_argument("--render-resolution", type=int, default=520)
    parser.add_argument("--view-azim", type=float, default=-90.0)
    parser.add_argument("--view-elev", type=float, default=24.0)
    parser.add_argument("--verbose-bake", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
