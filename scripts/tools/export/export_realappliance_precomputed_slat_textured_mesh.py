#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import trimesh
import utils3d
import nvdiffrast.torch as dr
from PIL import Image, ImageDraw


REPO_ROOT = Path(__file__).resolve().parents[2]
TRELLIS_ROOT = REPO_ROOT / "TRELLIS-arts"
for item in (str(REPO_ROOT), str(TRELLIS_ROOT)):
    if item not in sys.path:
        sys.path.insert(0, item)

os.environ.setdefault("SPCONV_ALGO", "native")
os.environ.setdefault("ATTN_BACKEND", "sdpa")
os.environ.setdefault("SPARSE_ATTN_BACKEND", "sdpa")

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import inference  # noqa: E402
from scripts.tools.roundtrip.trellis_full_voxel_mesh_roundtrip import load_camera_matrices  # noqa: E402
from trellis.modules.sparse import SparseTensor  # noqa: E402


def _restore_trellis_renderer_package() -> None:
    for name in list(sys.modules):
        if name == "trellis.renderers" or name.startswith("trellis.renderers."):
            sys.modules.pop(name, None)
    import trellis.renderers  # noqa: F401


_restore_trellis_renderer_package()
from trellis.utils import postprocessing_utils, render_utils  # noqa: E402
from trellis.renderers.gaussian_render import GaussianRenderer  # noqa: E402
from trellis.renderers.mesh_renderer import intrinsics_to_projection  # noqa: E402


DEFAULT_DATA_ROOT = Path("/mnt/robot-data-lab/jzh/art-gen/data/realappliance")
DEFAULT_OUT_DIR = Path("/mnt/robot-data-lab/jzh/art-gen-output/EE-eval/0617-realappliance-textured-bake-fixed")
DEFAULT_GAUSSIAN_DECODER = REPO_ROOT / "pretrained/TRELLIS-image-large/ckpts/slat_dec_gs_swin8_B_64l8gs32_fp16.safetensors"
DEFAULT_MESH_DECODER = REPO_ROOT / "pretrained/TRELLIS-image-large/ckpts/slat_dec_mesh_swin8_B_64l8m256c_fp16.safetensors"


def _load_latent(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        if set(data.files) != {"coords", "feats"}:
            raise ValueError(f"{path}: expected keys coords/feats, got {sorted(data.files)}")
        coords = np.asarray(data["coords"])
        feats = np.asarray(data["feats"])
    return (
        np.ascontiguousarray(coords.astype(np.int32, copy=False)),
        np.ascontiguousarray(feats.astype(np.float32, copy=False)),
    )


def _make_sparse(coords_np: np.ndarray, feats_np: np.ndarray) -> SparseTensor:
    coords = torch.from_numpy(coords_np).to(device="cuda", dtype=torch.int32)
    feats = torch.from_numpy(feats_np).to(device="cuda", dtype=torch.float32)
    batch = torch.zeros((coords.shape[0], 1), dtype=torch.int32, device=coords.device)
    return SparseTensor(coords=torch.cat([batch, coords], dim=1), feats=feats)


def _make_renderer(resolution: int, bg_color: tuple[int, int, int]) -> GaussianRenderer:
    renderer = GaussianRenderer()
    renderer.rendering_options.resolution = int(resolution)
    renderer.rendering_options.near = 0.8
    renderer.rendering_options.far = 1.6
    renderer.rendering_options.bg_color = bg_color
    renderer.rendering_options.ssaa = 1
    renderer.pipe.kernel_size = 0.1
    return renderer


@torch.no_grad()
def _render_gaussian_view(gaussian: Any, extrinsic: torch.Tensor, intrinsic: torch.Tensor, out_png: Path, resolution: int) -> Path:
    renderer = _make_renderer(resolution, (1, 1, 1))
    color = renderer.render(gaussian, extrinsic, intrinsic)["color"].detach().float().cpu().clamp(0, 1)
    arr = (color.permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr, mode="RGB").save(out_png)
    return out_png


def _mesh_arrays(mesh: Any) -> tuple[np.ndarray, np.ndarray]:
    vertices = mesh.vertices.detach().float().cpu().numpy() if torch.is_tensor(mesh.vertices) else np.asarray(mesh.vertices, dtype=np.float32)
    faces = mesh.faces.detach().long().cpu().numpy() if torch.is_tensor(mesh.faces) else np.asarray(mesh.faces, dtype=np.int64)
    return vertices.reshape(-1, 3), faces.reshape(-1, 3)


def _render_multiview_with_alpha(
    gaussian: Any,
    *,
    resolution: int,
    nviews: int,
    extrinsics: torch.Tensor | None,
    intrinsics: torch.Tensor | None,
    alpha_threshold: float,
    unpremultiply: bool,
) -> tuple[list[np.ndarray], list[np.ndarray], torch.Tensor, torch.Tensor, dict[str, Any]]:
    if extrinsics is None or intrinsics is None:
        r = 2
        fov = 40
        cams = [render_utils.sphere_hammersley_sequence(i, nviews) for i in range(nviews)]
        yaws = [cam[0] for cam in cams]
        pitchs = [cam[1] for cam in cams]
        extrinsics, intrinsics = render_utils.yaw_pitch_r_fov_to_extrinsics_intrinsics(yaws, pitchs, r, fov)
        camera_source = "sphere_hammersley"
    else:
        camera_source = "dataset"
    renderer_black = _make_renderer(resolution, (0, 0, 0))
    renderer_white = _make_renderer(resolution, (1, 1, 1))
    observations: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    alpha_fracs: list[float] = []
    alpha_means: list[float] = []
    for extrinsic, intrinsic in zip(extrinsics, intrinsics):
        black = renderer_black.render(gaussian, extrinsic, intrinsic)["color"].detach().float().cpu().clamp(0, 1)
        white = renderer_white.render(gaussian, extrinsic, intrinsic)["color"].detach().float().cpu().clamp(0, 1)
        black_np = black.permute(1, 2, 0).numpy()
        white_np = white.permute(1, 2, 0).numpy()
        alpha = np.clip(1.0 - (white_np - black_np).mean(axis=-1), 0.0, 1.0)
        if unpremultiply:
            color = black_np / np.maximum(alpha[..., None], 1e-4)
        else:
            color = black_np
        mask = alpha > float(alpha_threshold)
        alpha_fracs.append(float(mask.mean()))
        alpha_means.append(float(alpha[mask].mean()) if np.any(mask) else 0.0)
        observations.append(np.clip(color * 255.0, 0, 255).astype(np.uint8))
        masks.append(mask)
    stats = {
        "camera_source": camera_source,
        "num_observations": int(len(extrinsics)),
        "alpha_threshold": float(alpha_threshold),
        "unpremultiply": bool(unpremultiply),
        "alpha_mask_fraction_minmax": [float(min(alpha_fracs)), float(max(alpha_fracs))],
        "alpha_visible_mean_minmax": [float(min(alpha_means)), float(max(alpha_means))],
    }
    if isinstance(extrinsics, torch.Tensor):
        extrinsics_tensor = extrinsics
    else:
        extrinsics_tensor = torch.stack(extrinsics)
    if isinstance(intrinsics, torch.Tensor):
        intrinsics_tensor = intrinsics
    else:
        intrinsics_tensor = torch.stack(intrinsics)
    return observations, masks, extrinsics_tensor, intrinsics_tensor, stats


def _textured_mesh_from_decoded(
    gaussian: Any,
    mesh: Any,
    *,
    simplify: float,
    texture_size: int,
    bake_resolution: int,
    bake_views: int,
    bake_extrinsics: torch.Tensor | None,
    bake_intrinsics: torch.Tensor | None,
    bake_mode: str,
    alpha_threshold: float,
    unpremultiply: bool,
    fill_holes: bool,
    fill_holes_num_views: int,
    verbose: bool,
) -> tuple[trimesh.Trimesh, dict[str, Any]]:
    vertices, faces = _mesh_arrays(mesh)
    stats: dict[str, Any] = {
        "mesh_vertices_before": int(vertices.shape[0]),
        "mesh_faces_before": int(faces.shape[0]),
    }
    vertices, faces = postprocessing_utils.postprocess_mesh(
        vertices,
        faces,
        simplify=float(simplify) > 0,
        simplify_ratio=float(simplify),
        fill_holes=bool(fill_holes),
        fill_holes_num_views=int(fill_holes_num_views),
        verbose=verbose,
    )
    stats.update(
        {
            "mesh_vertices_post": int(vertices.shape[0]),
            "mesh_faces_post": int(faces.shape[0]),
        }
    )
    vertices, faces, uvs = postprocessing_utils.parametrize_mesh(vertices, faces)
    stats.update(
        {
            "uv_vertices": int(vertices.shape[0]),
            "uv_faces": int(faces.shape[0]),
            "uv_min": [float(x) for x in uvs.min(axis=0).tolist()],
            "uv_max": [float(x) for x in uvs.max(axis=0).tolist()],
        }
    )
    observations, masks, extrinsics, intrinsics, observation_stats = _render_multiview_with_alpha(
        gaussian,
        resolution=int(bake_resolution),
        nviews=int(bake_views),
        extrinsics=bake_extrinsics,
        intrinsics=bake_intrinsics,
        alpha_threshold=float(alpha_threshold),
        unpremultiply=bool(unpremultiply),
    )
    stats.update(observation_stats)
    mask_fracs = [float(mask.mean()) for mask in masks]
    stats["bake_mask_fraction_minmax"] = [float(min(mask_fracs)), float(max(mask_fracs))]
    texture = postprocessing_utils.bake_texture(
        vertices,
        faces,
        uvs,
        observations,
        masks,
        [extrinsics[i].detach().cpu().numpy() for i in range(len(extrinsics))],
        [intrinsics[i].detach().cpu().numpy() for i in range(len(intrinsics))],
        texture_size=int(texture_size),
        mode=str(bake_mode),
        lambda_tv=0.01,
        verbose=verbose,
    )
    stats["texture_mean"] = [float(x) for x in texture.reshape(-1, 3).mean(axis=0).tolist()]
    stats["texture_std"] = [float(x) for x in texture.reshape(-1, 3).std(axis=0).tolist()]
    material = trimesh.visual.material.PBRMaterial(
        roughnessFactor=1.0,
        baseColorTexture=Image.fromarray(texture),
        baseColorFactor=np.array([255, 255, 255, 255], dtype=np.uint8),
    )
    textured = trimesh.Trimesh(
        vertices=vertices,
        faces=faces,
        visual=trimesh.visual.TextureVisuals(uv=uvs, material=material),
        process=False,
    )
    return textured, stats


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
        material = getattr(getattr(geom, "visual", None), "material", None)
        texture = getattr(material, "baseColorTexture", None) or getattr(material, "image", None)
        uv = getattr(getattr(geom, "visual", None), "uv", None)
        if texture is None or uv is None:
            face_colors = np.full((faces.shape[0], 3), 0.75, dtype=np.float32)
        else:
            image = np.asarray(texture.convert("RGB") if hasattr(texture, "convert") else Image.fromarray(np.asarray(texture)).convert("RGB"))
            h, w = image.shape[:2]
            face_uv = np.asarray(uv, dtype=np.float32)[faces].mean(axis=1)
            px = np.clip((face_uv[:, 0] * (w - 1)).round().astype(np.int64), 0, w - 1)
            py = np.clip(((1.0 - face_uv[:, 1]) * (h - 1)).round().astype(np.int64), 0, h - 1)
            face_colors = image[py, px].astype(np.float32) / 255.0
        tris.append(vertices[faces])
        colors.append(face_colors)
        vertices_all.append(vertices)
    if not tris:
        raise ValueError(f"{path}: no textured triangle geometry")
    return np.concatenate(tris, axis=0), np.concatenate(colors, axis=0), np.concatenate(vertices_all, axis=0)


def _render_textured_preview(path: Path, out_png: Path, *, resolution: int, azim: float, elev: float, title: str) -> Path:
    tri, face_colors, vertices = _load_textured_triangles(path)
    if tri.shape[0] > 80000:
        rng = np.random.default_rng(123)
        keep = np.sort(rng.choice(tri.shape[0], 80000, replace=False))
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
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    collection3d = Poly3DCollection(
        tri,
        facecolors=np.clip(face_colors, 0, 1),
        edgecolors=(0.02, 0.02, 0.02, 0.03),
        linewidths=0.004,
    )
    collection3d.set_zsort("average")
    ax.add_collection3d(collection3d)
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
    draw.rectangle((0, 0, image.width, 30), fill=(255, 255, 255))
    draw.text((8, 9), title[:64], fill=(20, 20, 20))
    out_png.parent.mkdir(parents=True, exist_ok=True)
    image.save(out_png)
    return out_png


def _make_compare_panel(gaussian_png: Path, mesh_png: Path, out_png: Path) -> Path:
    images = [("Gaussian direct", gaussian_png), ("nvdiffrast baked mesh", mesh_png)]
    tile = 512
    canvas = Image.new("RGB", (tile * 2, tile + 32), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    for idx, (label, path) in enumerate(images):
        img = Image.open(path).convert("RGB")
        img.thumbnail((tile, tile), Image.Resampling.LANCZOS)
        x = idx * tile
        draw.rectangle((x, 0, x + tile, 32), fill=(0, 0, 0))
        draw.text((x + 8, 10), label, fill=(255, 255, 255))
        canvas.paste(img, (x + (tile - img.width) // 2, 32 + (tile - img.height) // 2))
    out_png.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_png)
    return out_png


def _render_textured_nvdiffrast_view(
    mesh: trimesh.Trimesh,
    extrinsic: torch.Tensor,
    intrinsic: torch.Tensor,
    out_png: Path,
    *,
    resolution: int,
) -> Path:
    vertices = torch.tensor(np.asarray(mesh.vertices, dtype=np.float32), device="cuda")
    faces = torch.tensor(np.asarray(mesh.faces, dtype=np.int32), device="cuda")
    uv_np = np.asarray(mesh.visual.uv, dtype=np.float32)
    if uv_np.shape[0] != vertices.shape[0]:
        raise ValueError(f"texture uv/vertex count mismatch: uv={uv_np.shape} vertices={tuple(vertices.shape)}")
    uvs = torch.tensor(uv_np, device="cuda")
    material = getattr(mesh.visual, "material", None)
    texture = getattr(material, "baseColorTexture", None) or getattr(material, "image", None)
    if texture is None:
        raise ValueError("textured mesh has no base color texture")
    image = texture.convert("RGB") if hasattr(texture, "convert") else Image.fromarray(np.asarray(texture)).convert("RGB")
    tex_np = np.asarray(image, dtype=np.float32) / 255.0
    # bake_texture returns a top-left-origin image; dr.texture samples with UV's
    # bottom-left convention, while the screen output is already in the same
    # orientation as GaussianRenderer. Flip only the texture, not the render.
    tex_tensor = torch.tensor(tex_np[::-1].copy(), device="cuda", dtype=torch.float32)[None]

    context = dr.RasterizeCudaContext(device="cuda")
    projection = intrinsics_to_projection(intrinsic, 0.1, 10.0)
    vertices_h = torch.cat([vertices[None], torch.ones((1, vertices.shape[0], 1), device="cuda")], dim=-1)
    vertices_clip = torch.bmm(vertices_h, (projection @ extrinsic).T[None])
    rast, rast_db = dr.rasterize(context, vertices_clip, faces, resolution=[int(resolution), int(resolution)], grad_db=True)
    uv_map, uv_dr = dr.interpolate(uvs[None], rast, faces, rast_db, diff_attrs="all")
    render_t = dr.texture(tex_tensor, uv_map, uv_dr)[0]
    render_t = dr.antialias(render_t[None], rast, vertices_clip, faces)[0]
    render = render_t.detach().cpu().numpy()
    mask = rast[0, ..., 3].detach().cpu().numpy() > 0
    out = np.ones((int(resolution), int(resolution), 3), dtype=np.float32)
    out[mask] = render[mask]
    out_png.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.clip(out * 255.0, 0, 255).astype(np.uint8)).save(out_png)
    return out_png


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Bake precomputed raw SLat Gaussian appearance to mesh with nvdiffrast.")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--object-id", default="039")
    parser.add_argument("--angle", type=int, default=1)
    parser.add_argument("--render-view", type=int, default=1)
    parser.add_argument("--component", default="overall")
    parser.add_argument("--gaussian-decoder-ckpt", type=Path, default=DEFAULT_GAUSSIAN_DECODER)
    parser.add_argument("--mesh-decoder-ckpt", type=Path, default=DEFAULT_MESH_DECODER)
    parser.add_argument("--texture-size", type=int, default=1024)
    parser.add_argument("--bake-resolution", type=int, default=768)
    parser.add_argument("--bake-views", type=int, default=64)
    parser.add_argument("--bake-camera-source", choices=("sphere", "dataset"), default="sphere")
    parser.add_argument("--bake-view-indices", default=None)
    parser.add_argument("--bake-mode", choices=("fast", "opt"), default="fast")
    parser.add_argument("--mesh-simplify", type=float, default=0.75)
    parser.add_argument("--no-fill-holes", action="store_true")
    parser.add_argument("--fill-holes-num-views", type=int, default=1000)
    parser.add_argument("--alpha-threshold", type=float, default=0.2)
    parser.add_argument("--premultiplied-observation", action="store_true")
    parser.add_argument("--preview-resolution", type=int, default=640)
    parser.add_argument("--view-azim", type=float, default=-90.0)
    parser.add_argument("--view-elev", type=float, default=24.0)
    parser.add_argument("--verbose-bake", action="store_true")
    return parser.parse_args()


def _parse_view_indices(value: str | None, *, fallback: list[int]) -> list[int]:
    if value is None or str(value).strip() == "":
        return fallback
    return [int(item) for item in str(value).replace(":", ",").split(",") if item.strip()]


def main() -> int:
    args = parse_args()
    started = time.time()
    inst = f"{args.object_id}_angle_{int(args.angle)}"
    inst_root = args.data_root / "part_synthesis_slat" / args.object_id[:2] / inst
    latent_path = inst_root / str(args.component) / "latent.npz"
    if str(args.component) == "overall":
        latent_path = inst_root / "overall" / "latent.npz"
    coords_np, feats_np = _load_latent(latent_path)
    slat = _make_sparse(coords_np, feats_np)
    decoded = inference.decode_slat_assets(
        slat,
        gaussian_decoder_ckpt=str(args.gaussian_decoder_ckpt.resolve()),
        mesh_decoder_ckpt=str(args.mesh_decoder_ckpt.resolve()),
        slat_is_normalized=False,
    )
    gaussian = decoded["gaussian"]
    mesh = decoded["mesh"]
    if mesh is None or not getattr(mesh, "success", True):
        raise RuntimeError(f"{args.component}: mesh decoder failed")
    case_dir = args.out_dir / f"{args.object_id}-{int(args.angle)}" / str(args.component)
    case_dir.mkdir(parents=True, exist_ok=True)
    extrinsics, intrinsics = load_camera_matrices(
        args.data_root / "renders" / args.object_id / f"angle_{int(args.angle)}" / "camera_transforms.json",
        [int(args.render_view)],
    )
    gaussian_png = _render_gaussian_view(
        gaussian,
        extrinsics[0],
        intrinsics[0],
        case_dir / "gaussian_direct.png",
        int(args.preview_resolution),
    )
    bake_extrinsics = None
    bake_intrinsics = None
    bake_view_indices = None
    if str(args.bake_camera_source) == "dataset":
        bake_view_indices = _parse_view_indices(args.bake_view_indices, fallback=[int(args.render_view)])
        bake_extrinsics, bake_intrinsics = load_camera_matrices(
            args.data_root / "renders" / args.object_id / f"angle_{int(args.angle)}" / "camera_transforms.json",
            bake_view_indices,
        )
    textured, bake_stats = _textured_mesh_from_decoded(
        gaussian,
        mesh,
        simplify=float(args.mesh_simplify),
        texture_size=int(args.texture_size),
        bake_resolution=int(args.bake_resolution),
        bake_views=int(args.bake_views),
        bake_extrinsics=bake_extrinsics,
        bake_intrinsics=bake_intrinsics,
        bake_mode=str(args.bake_mode),
        alpha_threshold=float(args.alpha_threshold),
        unpremultiply=not bool(args.premultiplied_observation),
        fill_holes=not bool(args.no_fill_holes),
        fill_holes_num_views=int(args.fill_holes_num_views),
        verbose=bool(args.verbose_bake),
    )
    glb_path = case_dir / "textured_mesh.glb"
    textured.export(str(glb_path))
    textured_nvdiffrast_png = _render_textured_nvdiffrast_view(
        textured,
        extrinsics[0],
        intrinsics[0],
        case_dir / "textured_mesh_nvdiffrast_render.png",
        resolution=int(args.preview_resolution),
    )
    mesh_png = _render_textured_preview(
        glb_path,
        case_dir / "textured_mesh_preview.png",
        resolution=int(args.preview_resolution),
        azim=float(args.view_azim),
        elev=float(args.view_elev),
        title=f"{args.component} textured mesh",
    )
    panel = _make_compare_panel(gaussian_png, textured_nvdiffrast_png, case_dir / "compare_gaussian_vs_baked.png")
    component_note = None
    if str(args.component) == "part_12_0":
        component_note = "internal/low-observation part; bake cannot recover unseen texture detail, default/inpainted material is expected"
    report = {
        "status": "done",
        "object_id": args.object_id,
        "angle": int(args.angle),
        "component": str(args.component),
        "component_note": component_note,
        "case_dir": str(case_dir.resolve()),
        "latent_path": str(latent_path.resolve()),
        "coords": int(coords_np.shape[0]),
        "gaussian_direct": str(gaussian_png.resolve()),
        "textured_mesh": str(glb_path.resolve()),
        "textured_preview": str(mesh_png.resolve()),
        "textured_nvdiffrast_render": str(textured_nvdiffrast_png.resolve()),
        "comparison": str(panel.resolve()),
        "bake_backend": "TRELLIS postprocessing_utils.bake_texture via nvdiffrast",
        "bake_fixes": [
            "fast bake now uses the current view mask instead of masks[0]",
            "bake masks are estimated from black/white Gaussian renders so black object surfaces are not dropped",
            "source is dataset part_synthesis_slat raw SLat with slat_is_normalized=False, not image SLat flow",
        ],
        "settings": {
            "texture_size": int(args.texture_size),
            "bake_resolution": int(args.bake_resolution),
            "bake_views": int(args.bake_views),
            "bake_camera_source": str(args.bake_camera_source),
            "bake_view_indices": bake_view_indices,
            "bake_mode": str(args.bake_mode),
            "mesh_simplify": float(args.mesh_simplify),
            "fill_holes": not bool(args.no_fill_holes),
            "fill_holes_num_views": int(args.fill_holes_num_views),
            "alpha_threshold": float(args.alpha_threshold),
            "unpremultiply_observation": not bool(args.premultiplied_observation),
        },
        "bake_stats": bake_stats,
        "seconds": round(time.time() - started, 3),
    }
    (case_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
