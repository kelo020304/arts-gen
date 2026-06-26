#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import matplotlib
import numpy as np
import torch
import trimesh
import trimesh.visual
from PIL import Image, ImageDraw


matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from mpl_toolkits.mplot3d.art3d import Poly3DCollection  # noqa: E402


REPO_ROOT = Path(__file__).resolve().parents[2]
TRELLIS_ROOT = REPO_ROOT / "TRELLIS-arts"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(TRELLIS_ROOT) not in sys.path:
    sys.path.insert(0, str(TRELLIS_ROOT))

os.environ.setdefault("SPCONV_ALGO", "native")
os.environ.setdefault("ATTN_BACKEND", "sdpa")
os.environ.setdefault("SPARSE_ATTN_BACKEND", "sdpa")

import inference  # noqa: E402
from part_ss_eval_platform.eval_0617_1 import (  # noqa: E402
    DEFAULT_DATA_CONFIG,
    DEFAULT_OUT_DIR,
    DEFAULT_PART_SEG_CKPT,
    DEFAULT_SLAT_FLOW_CKPT,
    DEFAULT_SLAT_MESH_DECODER_CKPT,
    DEFAULT_SPLIT_JSON_0617,
    DEFAULT_SS_DECODER_CKPT,
    DEFAULT_SS_FLOW_CKPT,
    _command_for_sample,
    _find_dataset_sample,
    _load_datasets,
    _run_dir_for_sample,
    _sample_data_config_path,
)
from part_ss_eval_platform.eval_real_0615 import (  # noqa: E402
    _execute,
    _load_coords,
)


def _restore_trellis_renderer_package() -> None:
    """Load TRELLIS renderers as the real package; inference.py registers a light stub."""
    for name in list(sys.modules):
        if name == "trellis.renderers" or name.startswith("trellis.renderers."):
            sys.modules.pop(name, None)
    import trellis.renderers  # noqa: F401


_restore_trellis_renderer_package()
from trellis.utils import postprocessing_utils, render_utils  # noqa: E402


DEFAULT_OBJECT_ID = "004d1e9e13934e319094151a4fad823f"
DEFAULT_DATASET_ID = "phyx-verse"
DEFAULT_ANGLE = 0
DEFAULT_OUT_DIR_TEST = Path("/mnt/robot-data-lab/jzh/art-gen-output/EE-eval/0617-test")
DEFAULT_GAUSSIAN_DECODER_CKPT = (
    REPO_ROOT / "pretrained/TRELLIS-image-large/ckpts/slat_dec_gs_swin8_B_64l8gs32_fp16.safetensors"
)
Z_UP_TO_Y_UP = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=np.float32)


def _axis_transform(vertices: np.ndarray, axis_mode: str) -> np.ndarray:
    if axis_mode == "zup_yfront":
        return vertices
    if axis_mode == "yup_zfront":
        return vertices @ Z_UP_TO_Y_UP
    raise ValueError(f"unknown axis_mode={axis_mode!r}")


def _safe_name(value: str, max_len: int = 80) -> str:
    out = "".join(ch if ch.isalnum() or ch in ("-", "_", ".") else "_" for ch in str(value)).strip("_")
    return (out or "part")[:max_len]


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _load_slat_cond_tokens_for_views(
    ds: Any,
    sample: dict[str, Any],
    view_indices: list[int],
) -> tuple[torch.Tensor, dict[str, Any]]:
    data_root = Path(ds.data_root)
    token_candidates = [
        data_root / ds.recon_subdir / "dinov2_tokens" / str(sample["obj_id"]) / f"angle_{int(sample['angle_idx'])}" / "tokens.npz",
        data_root / ds.recon_subdir / "dinov2_tokens_prenorm" / str(sample["obj_id"]) / f"angle_{int(sample['angle_idx'])}" / "tokens.npz",
        data_root / ds.recon_subdir / "dinov2_tokens_official_prenorm1374" / str(sample["obj_id"]) / f"angle_{int(sample['angle_idx'])}" / "tokens.npz",
    ]
    token_path = next((path for path in token_candidates if path.is_file()), token_candidates[0])
    if not token_path.is_file():
        raise FileNotFoundError(f"TRELLIS SLat DINO tokens not found: {token_path}")
    with np.load(token_path, allow_pickle=False) as data:
        if "tokens" not in data.files:
            raise KeyError(f"{token_path} expected key 'tokens', got {data.files}")
        tokens = np.asarray(data["tokens"], dtype=np.float32)
    if tokens.ndim != 3 or tokens.shape[-1] != 1024:
        raise ValueError(f"{token_path} expected [V,T,1024], got {tokens.shape}")
    picked_views = [int(v) for v in view_indices]
    if not picked_views:
        raise ValueError("slat view list must be non-empty")
    if max(picked_views) >= tokens.shape[0] or min(picked_views) < 0:
        raise ValueError(f"{token_path} has {tokens.shape[0]} views, cannot select {picked_views}")
    picked = torch.from_numpy(np.ascontiguousarray(tokens[picked_views])).float()
    picked = torch.nn.functional.layer_norm(picked, picked.shape[-1:])
    meta = {
        "token_path": str(token_path.resolve()),
        "available_token_shape": list(tokens.shape),
        "view_indices": picked_views,
        "picked_token_shape": list(picked.shape),
        "flow_input_shape": [int(np.prod(picked.shape[:2])), int(picked.shape[-1])],
        "contract": "SLat appearance flow uses only these selected view tokens, flattened by inference.run_slat_flow_from_tokens.",
    }
    return picked, meta


def _progress(path: Path, rec: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def _find_sample(ds: Any, object_id: str, angle: int, dataset_id: str) -> SimpleNamespace:
    for row in ds.samples:
        if str(row["obj_id"]) == object_id and int(row["angle_idx"]) == int(angle):
            return SimpleNamespace(
                split=str(row.get("split", "train")),
                dataset_id=dataset_id,
                obj_id=object_id,
                angle_idx=int(angle),
                data_root=str(ds.data_root),
                manifest_path=str(ds.manifest_path),
            )
    raise KeyError(f"{dataset_id}::{object_id} angle={angle} not found")


def _ensure_ss_and_part(args: argparse.Namespace, ds: Any, sample: SimpleNamespace, ds_sample: dict[str, Any]) -> Path:
    run_dir = _run_dir_for_sample(args.out_dir, sample)
    progress_path = Path(args.out_dir) / "progress_textured.jsonl"
    expected_parts = len(ds_sample["parts"])
    ss_done = (run_dir / "ss_latent.npy").is_file() and (run_dir / "voxel.npz").is_file()
    part_done = len(list((run_dir / "parts").glob("part_*_voxel.npz"))) >= expected_parts
    local_args = argparse.Namespace(**vars(args))
    local_args.data_config = str(_sample_data_config_path(Path(args.out_dir), sample, ds))

    for stage, done in (("ss", ss_done), ("part", part_done)):
        if done and not args.force_stage:
            _progress(progress_path, {"stage": stage, "status": "skipped", "run_dir": str(run_dir)})
            continue
        spec = _command_for_sample(Path(args.out_dir), sample, local_args, stage, ds)
        rec = _execute(
            spec,
            gpu=str(args.gpu),
            progress_path=progress_path,
            label=f"0617-test/{stage}/{sample.dataset_id}/{sample.obj_id}/{int(sample.angle_idx)}",
        )
        rec["status"] = "done"
        _progress(progress_path, rec)
    if not (run_dir / "voxel.npz").is_file():
        raise FileNotFoundError(f"missing whole voxel after ss stage: {run_dir / 'voxel.npz'}")
    if len(list((run_dir / "parts").glob("part_*_voxel.npz"))) < expected_parts:
        raise FileNotFoundError(f"missing part voxels after part stage: {run_dir / 'parts'}")
    return run_dir


def _mesh_arrays(mesh: Any) -> tuple[np.ndarray, np.ndarray]:
    vertices = mesh.vertices.detach().float().cpu().numpy() if torch.is_tensor(mesh.vertices) else np.asarray(mesh.vertices, dtype=np.float32)
    faces = mesh.faces.detach().long().cpu().numpy() if torch.is_tensor(mesh.faces) else np.asarray(mesh.faces, dtype=np.int64)
    return vertices.reshape(-1, 3), faces.reshape(-1, 3)


def _fast_textured_mesh(
    gaussian: Any,
    mesh: Any,
    *,
    simplify: float,
    texture_size: int,
    bake_resolution: int,
    bake_views: int,
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
    extrinsics_np = [extrinsics[i].detach().cpu().numpy() for i in range(len(extrinsics))]
    intrinsics_np = [intrinsics[i].detach().cpu().numpy() for i in range(len(intrinsics))]
    texture = postprocessing_utils.bake_texture(
        vertices,
        faces,
        uvs,
        observations,
        masks,
        extrinsics_np,
        intrinsics_np,
        texture_size=int(texture_size),
        mode="fast",
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


def _sparse_subset_from_coords(slat: Any, coords: np.ndarray, label: str) -> tuple[Any, int]:
    from trellis.modules.sparse import SparseTensor

    coords_np = np.asarray(coords, dtype=np.int64).reshape(-1, 3)
    if coords_np.size == 0:
        raise ValueError(f"{label}: empty part coords")
    slat_coords = slat.coords
    slat_feats = slat.feats
    spatial = slat_coords[:, 1:].detach().long().cpu().numpy()
    resolution = int(max(int(spatial.max(initial=0)) + 1, int(coords_np.max(initial=0)) + 1, 64))
    slat_keys = (
        spatial[:, 0].astype(np.int64) * resolution * resolution
        + spatial[:, 1].astype(np.int64) * resolution
        + spatial[:, 2].astype(np.int64)
    )
    part_keys = (
        coords_np[:, 0].astype(np.int64) * resolution * resolution
        + coords_np[:, 1].astype(np.int64) * resolution
        + coords_np[:, 2].astype(np.int64)
    )
    part_key_set = set(int(x) for x in part_keys.tolist())
    keep_np = np.fromiter((int(k) in part_key_set for k in slat_keys.tolist()), dtype=bool, count=len(slat_keys))
    keep = torch.from_numpy(keep_np).to(device=slat_feats.device)
    matched = int(keep_np.sum())
    if matched == 0:
        raise ValueError(f"{label}: no part coords matched whole SLat coords ({len(coords_np)} requested)")
    sub_feats = slat_feats[keep].contiguous()
    sub_coords = slat_coords[keep].contiguous()
    return SparseTensor(feats=sub_feats, coords=sub_coords), matched


def _decode_and_export_slat(
    *,
    slat: Any,
    coords_count: int,
    label: str,
    out_glb: Path,
    args: argparse.Namespace,
    matched_coords: int | None = None,
    slat_source: str,
) -> dict[str, Any]:
    started = time.time()
    decoded = inference.decode_slat_assets(
        slat,
        gaussian_decoder_ckpt=str(Path(args.slat_gaussian_decoder_ckpt).resolve()),
        mesh_decoder_ckpt=str(Path(args.slat_mesh_decoder_ckpt).resolve()),
        slat_is_normalized=True,
    )
    gaussian = decoded.get("gaussian")
    mesh = decoded.get("mesh")
    if gaussian is None:
        raise RuntimeError(f"{label}: gaussian decoder returned None")
    if mesh is None or not getattr(mesh, "success", True):
        raise RuntimeError(f"{label}: mesh decoder failed")
    textured = _fast_textured_mesh(
        gaussian,
        mesh,
        simplify=float(args.mesh_simplify),
        texture_size=int(args.texture_size),
        bake_resolution=int(args.bake_resolution),
        bake_views=int(args.bake_views),
        axis_mode=str(args.axis_mode),
        verbose=bool(args.verbose_bake),
    )
    out_glb.parent.mkdir(parents=True, exist_ok=True)
    textured.export(str(out_glb))
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return {
        "label": label,
        "mesh": str(out_glb.resolve()),
        "coords": int(coords_count),
        "matched_coords": None if matched_coords is None else int(matched_coords),
        "slat_source": slat_source,
        "texture_bake_backend": "trellis.postprocessing_utils.bake_texture(mode=fast,nvdiffrast)",
        "seconds": round(time.time() - started, 3),
        "texture_size": int(args.texture_size),
        "bake_resolution": int(args.bake_resolution),
        "bake_views": int(args.bake_views),
        "mesh_simplify": float(args.mesh_simplify),
        "axis_mode": str(args.axis_mode),
    }


def _component_record(
    *,
    label: str,
    out_glb: Path,
    coords_count: int,
    matched_coords: int | None,
    slat_source: str,
    args: argparse.Namespace,
    reused: bool = False,
) -> dict[str, Any]:
    rec = {
        "label": label,
        "mesh": str(out_glb.resolve()),
        "coords": int(coords_count),
        "matched_coords": None if matched_coords is None else int(matched_coords),
        "slat_source": slat_source,
        "texture_bake_backend": "trellis.postprocessing_utils.bake_texture(mode=fast,nvdiffrast)",
        "texture_size": int(args.texture_size),
        "bake_resolution": int(args.bake_resolution),
        "bake_views": int(args.bake_views),
        "mesh_simplify": float(args.mesh_simplify),
    }
    if reused:
        rec["reused"] = True
    return rec


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
        face_colors = None
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
        if face_colors is None:
            vc = getattr(visual, "vertex_colors", None)
            if vc is not None and len(vc) == len(vertices):
                face_colors = np.asarray(vc, dtype=np.float32)[faces, :3].mean(axis=1) / 255.0
            else:
                face_colors = np.full((len(faces), 3), 0.72, dtype=np.float32)
        tris.append(vertices[faces])
        colors.append(face_colors)
        vertices_all.append(vertices)
    if not tris:
        raise ValueError(f"{path}: no textured triangle geometry")
    return np.concatenate(tris, axis=0), np.concatenate(colors, axis=0), np.concatenate(vertices_all, axis=0)


def _render_textured_glb(path: Path, *, title: str, resolution: int, azim: float, elev: float, max_faces: int) -> Image.Image:
    tri, face_colors, vertices = _load_textured_triangles(path)
    if len(tri) > int(max_faces):
        rng = np.random.default_rng(abs(hash(str(path))) % (2**32))
        keep = np.sort(rng.choice(len(tri), int(max_faces), replace=False))
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
    return image


def _make_overview(case_dir: Path, records: list[dict[str, Any]], args: argparse.Namespace) -> Path:
    tile = int(args.tile_resolution)
    pad = 8
    cols = min(int(args.panel_cols), len(records))
    rows = int(np.ceil(len(records) / cols))
    canvas = Image.new("RGB", (cols * tile + (cols + 1) * pad, rows * tile + (rows + 1) * pad), (245, 245, 245))
    for idx, rec in enumerate(records):
        image = _render_textured_glb(
            Path(rec["mesh"]),
            title=str(rec["label"]),
            resolution=tile,
            azim=float(args.view_azim),
            elev=float(args.view_elev),
            max_faces=int(args.preview_max_faces),
        )
        x = pad + (idx % cols) * (tile + pad)
        y = pad + (idx // cols) * (tile + pad)
        canvas.paste(image, (x, y))
    out = case_dir / "overview_mesh.png"
    canvas.save(out)
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="0617-test: true textured overall and per-part SLat mesh export.")
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR_TEST)
    p.add_argument("--data-config", type=Path, default=DEFAULT_DATA_CONFIG)
    p.add_argument("--split-json", type=Path, default=DEFAULT_SPLIT_JSON_0617)
    p.add_argument("--dataset-id", default=DEFAULT_DATASET_ID)
    p.add_argument("--object-id", default=DEFAULT_OBJECT_ID)
    p.add_argument("--angle", type=int, default=DEFAULT_ANGLE)
    p.add_argument("--part-seg-ckpt", type=Path, default=DEFAULT_PART_SEG_CKPT)
    p.add_argument("--ss-flow-ckpt", type=Path, default=DEFAULT_SS_FLOW_CKPT)
    p.add_argument("--ss-decoder-ckpt", type=Path, default=DEFAULT_SS_DECODER_CKPT)
    p.add_argument("--slat-flow-ckpt", type=Path, default=DEFAULT_SLAT_FLOW_CKPT)
    p.add_argument("--slat-mesh-decoder-ckpt", type=Path, default=DEFAULT_SLAT_MESH_DECODER_CKPT)
    p.add_argument("--slat-gaussian-decoder-ckpt", type=Path, default=DEFAULT_GAUSSIAN_DECODER_CKPT)
    p.add_argument("--gpu", default="0")
    p.add_argument("--slat-steps", type=int, default=25)
    p.add_argument("--slat-seed", type=int, default=42)
    p.add_argument("--mesh-simplify", type=float, default=0.88)
    p.add_argument("--texture-size", type=int, default=512)
    p.add_argument("--bake-resolution", type=int, default=512)
    p.add_argument("--bake-views", type=int, default=24)
    p.add_argument("--tile-resolution", type=int, default=360)
    p.add_argument("--panel-cols", type=int, default=3)
    p.add_argument("--view-azim", type=float, default=-90.0)
    p.add_argument("--view-elev", type=float, default=24.0)
    p.add_argument("--axis-mode", choices=("zup_yfront", "yup_zfront"), default="zup_yfront")
    p.add_argument("--slat-view-indices", type=int, nargs="+", default=[0])
    p.add_argument("--preview-max-faces", type=int, default=60000)
    p.add_argument("--force-stage", action="store_true")
    p.add_argument("--force-export", action="store_true")
    p.add_argument("--verbose-bake", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    args.out_dir = Path(args.out_dir).resolve()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for attr in (
        "data_config",
        "split_json",
        "part_seg_ckpt",
        "ss_flow_ckpt",
        "ss_decoder_ckpt",
        "slat_flow_ckpt",
        "slat_mesh_decoder_ckpt",
        "slat_gaussian_decoder_ckpt",
    ):
        path = Path(getattr(args, attr)).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"{attr} not found: {path}")
        setattr(args, attr, path)

    datasets, dataset_meta = _load_datasets(args)
    if args.dataset_id not in datasets:
        raise KeyError(f"dataset_id={args.dataset_id!r} not found; available={sorted(datasets)}")
    ds = datasets[args.dataset_id]
    sample = _find_sample(ds, args.object_id, int(args.angle), args.dataset_id)
    ds_sample = _find_dataset_sample(ds, sample)
    run_dir = _ensure_ss_and_part(args, ds, sample, ds_sample)

    case_dir = args.out_dir / sample.split / args.dataset_id / f"{sample.obj_id}-{int(sample.angle_idx)}"
    case_dir.mkdir(parents=True, exist_ok=True)
    cond_tokens, slat_cond_meta = _load_slat_cond_tokens_for_views(ds, ds_sample, list(args.slat_view_indices))
    print(
        f"[0617-test] SLat cond views={slat_cond_meta['view_indices']} "
        f"picked_shape={slat_cond_meta['picked_token_shape']} "
        f"flow_input_shape={slat_cond_meta['flow_input_shape']} token_path={slat_cond_meta['token_path']}",
        flush=True,
    )
    records: list[dict[str, Any]] = []

    components: list[tuple[str, Path, str]] = [
        ("overall", run_dir / "voxel.npz", "overall_textured.glb"),
    ]
    for out_idx, part in enumerate(ds_sample["parts"]):
        label = f"part_{out_idx:02d}_{_safe_name(str(part['part_name']))}"
        components.append(
            (
                label,
                run_dir / "parts" / f"part_{out_idx:02d}_voxel.npz",
                f"{label}_textured.glb",
            )
        )

    overall_coords = _load_coords(run_dir / "voxel.npz")
    overall_coords_t = torch.from_numpy(np.ascontiguousarray(overall_coords.astype(np.int64, copy=False))).long()
    print(
        f"[0617-test] SLat flow ONCE overall coords={overall_coords.shape[0]} "
        f"seed={int(args.slat_seed)}",
        flush=True,
    )
    overall_slat = inference.run_slat_flow_from_tokens(
        cond_tokens,
        overall_coords_t,
        str(Path(args.slat_flow_ckpt).resolve()),
        num_steps=int(args.slat_steps),
        seed=int(args.slat_seed),
    )
    slat_flow_calls = 1

    for label, coords_path, filename in components:
        out_glb = case_dir / filename
        coords = _load_coords(coords_path)
        if label == "overall":
            component_slat = overall_slat
            matched_coords = int(coords.shape[0])
            slat_source = "whole_slat_flow_once"
        else:
            component_slat, matched_coords = _sparse_subset_from_coords(overall_slat, coords, label)
            slat_source = "subset_from_whole_slat_by_coords"
        if out_glb.is_file() and not args.force_export:
            rec = _component_record(
                label=label,
                out_glb=out_glb,
                coords_count=int(coords.shape[0]),
                matched_coords=matched_coords,
                slat_source=slat_source,
                args=args,
                reused=True,
            )
            rec["coords_path"] = str(coords_path.resolve())
            records.append(rec)
            continue
        print(
            f"[0617-test] decode+bake {label} coords={coords.shape[0]} "
            f"matched={matched_coords} source={slat_source} -> {out_glb}",
            flush=True,
        )
        rec = _decode_and_export_slat(
            slat=component_slat,
            coords_count=int(coords.shape[0]),
            label=label,
            out_glb=out_glb,
            args=args,
            matched_coords=matched_coords,
            slat_source=slat_source,
        )
        rec["coords_path"] = str(coords_path.resolve())
        records.append(rec)

    overview = _make_overview(case_dir, records, args)
    summary = {
        "status": "done",
        "dataset_id": args.dataset_id,
        "obj_id": args.object_id,
        "angle": int(args.angle),
        "case_dir": str(case_dir.resolve()),
        "run_dir": str(run_dir.resolve()),
        "overview": str(overview.resolve()),
        "components": records,
        "slat_flow_calls": slat_flow_calls,
        "slat_condition": slat_cond_meta,
        "axis_mode": str(args.axis_mode),
        "preview_view": {
            "azim": float(args.view_azim),
            "elev": float(args.view_elev),
            "front_convention": "-Y forward when axis_mode=zup_yfront and preview azim=-90",
        },
        "slat_part_rule": "SLat flow is run once on whole voxel coords; each part SparseTensor is sliced from whole SLat by matching coords[:, 1:].",
        "texture_bake_backend": "TRELLIS official postprocessing_utils: postprocess_mesh -> parametrize_mesh -> render_multiview -> bake_texture(mode=fast) using nvdiffrast.",
        "checkpoints": {
            "part_seg": str(args.part_seg_ckpt.resolve()),
            "ss_flow": str(args.ss_flow_ckpt.resolve()),
            "slat_flow": str(args.slat_flow_ckpt.resolve()),
            "slat_mesh_decoder": str(args.slat_mesh_decoder_ckpt.resolve()),
            "slat_gaussian_decoder": str(args.slat_gaussian_decoder_ckpt.resolve()),
        },
        "datasets": dataset_meta,
        "note": "Textured GLBs use TRELLIS official nvdiffrast texture baking from the whole-object SLat appearance; not point/voxel colorization and not the old projection fallback.",
    }
    _write_json(case_dir / "summary.json", summary)
    print(f"[0617-test] done -> {case_dir}", flush=True)
    print(f"[0617-test] overview -> {overview}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
