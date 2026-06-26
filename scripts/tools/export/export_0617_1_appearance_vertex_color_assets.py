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
from PIL import Image, ImageDraw
from scipy.spatial import cKDTree


REPO_ROOT = Path(__file__).resolve().parents[2]
TRELLIS_ROOT = REPO_ROOT / "TRELLIS-arts"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(TRELLIS_ROOT) not in sys.path:
    sys.path.insert(0, str(TRELLIS_ROOT))

os.environ.setdefault("SPCONV_ALGO", "native")
os.environ.setdefault("ATTN_BACKEND", "sdpa")
os.environ.setdefault("SPARSE_ATTN_BACKEND", "sdpa")
os.environ.setdefault("OPEN3D_CPU_RENDERING", "true")

import inference  # noqa: E402
from part_ss_eval_platform.eval_0617_1 import _load_coords  # noqa: E402
from scripts.tools.export.export_0617_1_part_mesh_assets import (  # noqa: E402
    DEFAULT_EVAL_DIR,
    DEFAULT_SLAT_FLOW_CKPT,
    _render_colored_mesh_matplotlib,
)
from scripts.tools.render.render_glb_open3d_preview import render as render_open3d  # noqa: E402
from trellis.renderers.sh_utils import SH2RGB  # noqa: E402
from trellis.utils.arts.slat_asset_writer import _SAM3D_Z_UP_TO_Y_UP  # noqa: E402


DEFAULT_PART_MESH_ROOT = DEFAULT_EVAL_DIR / "part_mesh_assets"
DEFAULT_OUT_SUBDIR = "part_mesh_appearance_assets"
DEFAULT_GAUSSIAN_DECODER_CKPT = (
    REPO_ROOT / "pretrained/TRELLIS-image-large/ckpts/slat_dec_gs_swin8_B_64l8gs32_fp16.safetensors"
)


def _load_scene_mesh(path: Path) -> trimesh.Trimesh:
    loaded = trimesh.load(str(path), force="scene", process=False)
    if isinstance(loaded, trimesh.Trimesh):
        return loaded
    meshes = [
        geom
        for geom in loaded.geometry.values()
        if isinstance(geom, trimesh.Trimesh) and len(geom.vertices) and len(geom.faces)
    ]
    if not meshes:
        raise ValueError(f"{path}: no mesh geometry")
    return trimesh.util.concatenate(meshes)


def _gaussian_rgb_xyz(gaussian: Any) -> tuple[np.ndarray, np.ndarray]:
    xyz = gaussian.get_xyz.detach().float().cpu().numpy()
    xyz = xyz @ _SAM3D_Z_UP_TO_Y_UP
    features = gaussian._features_dc.detach().float().cpu()
    rgb = SH2RGB(features.reshape(-1, 3)).numpy()
    rgb = np.clip(rgb, 0.0, 1.0)
    opacity = gaussian.get_opacity.detach().float().cpu().numpy().reshape(-1)
    keep = opacity > 0.02
    if keep.any():
        xyz = xyz[keep]
        rgb = rgb[keep]
    return xyz.astype(np.float32, copy=False), rgb.astype(np.float32, copy=False)


def _transfer_gaussian_colors(mesh: trimesh.Trimesh, gaussian: Any, *, k: int) -> tuple[trimesh.Trimesh, dict[str, Any]]:
    xyz, rgb = _gaussian_rgb_xyz(gaussian)
    if len(xyz) == 0:
        raise RuntimeError("gaussian has no visible color samples")

    vertices = np.asarray(mesh.vertices, dtype=np.float32)
    tree = cKDTree(xyz)
    k_eff = max(1, min(int(k), len(xyz)))
    dist, idx = tree.query(vertices, k=k_eff, workers=-1)
    if k_eff == 1:
        colors = rgb[np.asarray(idx)]
    else:
        dist = np.asarray(dist, dtype=np.float32)
        idx = np.asarray(idx)
        weights = 1.0 / np.maximum(dist, 1e-4)
        weights = weights / np.maximum(weights.sum(axis=1, keepdims=True), 1e-8)
        colors = (rgb[idx] * weights[..., None]).sum(axis=1)

    out = mesh.copy()
    rgba = np.concatenate(
        [np.clip(colors * 255.0, 0, 255).astype(np.uint8), np.full((len(colors), 1), 255, dtype=np.uint8)],
        axis=1,
    )
    out.visual.vertex_colors = rgba
    stats = {
        "gaussian_count": int(len(xyz)),
        "mesh_vertices": int(len(vertices)),
        "k": int(k_eff),
        "rgb_mean": [float(x) for x in colors.mean(axis=0).tolist()],
        "rgb_min": [float(x) for x in colors.min(axis=0).tolist()],
        "rgb_max": [float(x) for x in colors.max(axis=0).tolist()],
        "rgb_std": [float(x) for x in colors.std(axis=0).tolist()],
    }
    return out, stats


def _decode_gaussian(coords_path: Path, cond_tokens_path: Path, args: argparse.Namespace, seed: int) -> Any:
    coords = _load_coords(coords_path).astype(np.int64, copy=False)
    tokens = torch.from_numpy(np.load(cond_tokens_path)).float()
    if tokens.dim() != 2 or tokens.shape[-1] != 1024:
        raise ValueError(f"expected cond tokens [T,1024], got {tuple(tokens.shape)} from {cond_tokens_path}")
    coords_t = torch.from_numpy(np.ascontiguousarray(coords)).long()
    slat = inference.run_slat_flow_from_tokens(
        tokens,
        coords_t,
        str(Path(args.slat_flow_ckpt).resolve()),
        num_steps=int(args.slat_steps),
        seed=int(seed),
    )
    decoded = inference.decode_slat_assets(
        slat,
        gaussian_decoder_ckpt=str(Path(args.slat_gaussian_decoder_ckpt).resolve()),
        slat_is_normalized=True,
    )
    gaussian = decoded.get("gaussian")
    if gaussian is None:
        raise RuntimeError(f"gaussian decoder returned None for {coords_path}")
    return gaussian


def _export_colored_mesh(
    *,
    src_mesh: Path,
    coords_path: Path,
    cond_tokens_path: Path,
    out_dir: Path,
    label: str,
    seed: int,
    args: argparse.Namespace,
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    glb_path = out_dir / f"{label}.glb"
    meta_path = out_dir / f"{label}.json"
    iso_png = out_dir / "renders" / f"{label}_iso.png"
    if glb_path.is_file() and meta_path.is_file() and iso_png.is_file() and not args.force:
        return json.loads(meta_path.read_text(encoding="utf-8"))

    started = time.time()
    gaussian = _decode_gaussian(coords_path, cond_tokens_path, args, seed)
    mesh = _load_scene_mesh(src_mesh)
    colored, stats = _transfer_gaussian_colors(mesh, gaussian, k=int(args.nearest_k))
    colored.export(str(glb_path))
    renders = render_open3d(
        glb_path,
        out_dir=out_dir / "renders",
        views={"iso": (315.0, 24.0), "front": (270.0, 8.0), "side": (0.0, 8.0)},
        resolution=int(args.render_resolution),
        use_vertex_colors=True,
    )
    rec = {
        "label": label,
        "source_mesh": str(src_mesh.resolve()),
        "source_coords": str(coords_path.resolve()),
        "seed": int(seed),
        "mesh": str(glb_path.resolve()),
        "renders": [str(p.resolve()) for p in renders],
        "color_transfer": stats,
        "seconds": round(time.time() - started, 3),
    }
    meta_path.write_text(json.dumps(rec, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return rec


def _assemble(part_records: list[dict[str, Any]], out_dir: Path, args: argparse.Namespace) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    glb_path = out_dir / "assembled_parts_appearance.glb"
    iso_png = out_dir / "renders" / "assembled_parts_appearance_iso.png"
    if glb_path.is_file() and iso_png.is_file() and not args.force:
        return {
            "mesh": str(glb_path.resolve()),
            "renders": [str(p.resolve()) for p in sorted((out_dir / "renders").glob("assembled_parts_appearance_*.png"))],
        }
    meshes = [_load_scene_mesh(Path(rec["mesh"])) for rec in part_records]
    assembled = trimesh.util.concatenate(meshes)
    assembled.export(str(glb_path))
    renders = _render_colored_mesh_matplotlib(
        glb_path,
        out_dir=out_dir / "renders",
        stem="assembled_parts_appearance",
        resolution=int(args.render_resolution),
    )
    return {"mesh": str(glb_path.resolve()), "renders": [str(p.resolve()) for p in renders]}


def _make_panel(case_dir: Path, overall: dict[str, Any], parts: list[dict[str, Any]], assembled: dict[str, Any]) -> Path:
    def iso(record: dict[str, Any]) -> Path:
        for raw in record.get("renders", []):
            p = Path(raw)
            if p.name.endswith("_iso.png"):
                return p
        raise RuntimeError(f"missing iso render in {record.get('label', 'record')}")

    entries: list[tuple[str, Path]] = [("overall_appearance", iso(overall)), ("assembled_appearance", iso(assembled))]
    entries.extend((str(rec["label"]), iso(rec)) for rec in parts)
    thumb = 256
    label_h = 28
    pad = 8
    cols = min(5, len(entries))
    rows = int(np.ceil(len(entries) / cols))
    canvas = Image.new("RGB", (cols * thumb + (cols + 1) * pad, rows * (thumb + label_h) + (rows + 1) * pad), (245, 245, 245))
    draw = ImageDraw.Draw(canvas)
    for idx, (label, path) in enumerate(entries):
        image = Image.open(path).convert("RGB")
        image.thumbnail((thumb, thumb), Image.Resampling.LANCZOS)
        tile = Image.new("RGB", (thumb, thumb), (255, 255, 255))
        tile.paste(image, ((thumb - image.width) // 2, (thumb - image.height) // 2))
        x = pad + (idx % cols) * (thumb + pad)
        y = pad + (idx // cols) * (thumb + label_h + pad)
        draw.text((x + 6, y + 6), label[:42], fill=(20, 20, 20))
        canvas.paste(tile, (x, y + label_h))
    out = case_dir / "panel_appearance_parts.png"
    canvas.save(out)
    return out


def _summary_paths(root: Path) -> list[Path]:
    return sorted(root.glob("*/*/*/part_mesh_summary.json"))


def _progress(path: Path, rec: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def _export_case(summary_path: Path, out_root: Path, args: argparse.Namespace, progress_path: Path) -> dict[str, Any]:
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    sample_rel = Path(summary["split"]) / summary["dataset_id"] / f"{summary['obj_id']}-{int(summary['angle'])}"
    case_out = out_root / sample_rel
    done_path = case_out / "appearance_mesh_summary.json"
    if done_path.is_file() and not args.force:
        return json.loads(done_path.read_text(encoding="utf-8"))

    run_dir = Path(summary["run_dir"])
    cond_tokens_path = run_dir / "ss_latent.npy"
    if not cond_tokens_path.is_file():
        raise FileNotFoundError(f"missing cond token cache: {cond_tokens_path}")

    started = time.time()
    overall = _export_colored_mesh(
        src_mesh=Path(summary["overall"]["mesh"]),
        coords_path=run_dir / "voxel.npz",
        cond_tokens_path=cond_tokens_path,
        out_dir=case_out / "overall",
        label="overall_appearance",
        seed=int(summary["overall"]["seed"]),
        args=args,
    )
    part_records: list[dict[str, Any]] = []
    for part in summary["parts"]:
        rec = _export_colored_mesh(
            src_mesh=Path(part["mesh"]),
            coords_path=Path(part["source_voxel"]),
            cond_tokens_path=cond_tokens_path,
            out_dir=case_out / "parts" / str(part["label"]),
            label=f"{part['label']}_appearance",
            seed=int(part["seed"]),
            args=args,
        )
        rec.update(
            {
                "part_index": int(part["part_index"]),
                "part_name": str(part["part_name"]),
            }
        )
        part_records.append(rec)

    assembled = _assemble(part_records, case_out / "assembled_parts", args)
    panel = _make_panel(case_out, overall, part_records, assembled)
    out = {
        "status": "done",
        "split": summary["split"],
        "dataset_id": summary["dataset_id"],
        "obj_id": summary["obj_id"],
        "angle": int(summary["angle"]),
        "source_summary": str(summary_path.resolve()),
        "case_dir": str(case_out.resolve()),
        "overall": overall,
        "parts": part_records,
        "assembled_parts": assembled,
        "panel": str(panel.resolve()),
        "seconds": round(time.time() - started, 3),
    }
    case_out.mkdir(parents=True, exist_ok=True)
    done_path.write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    _progress(progress_path, out)
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Transfer TRELLIS Gaussian appearance colors onto 0617-1 per-part meshes.")
    p.add_argument("--part-mesh-root", type=Path, default=DEFAULT_PART_MESH_ROOT)
    p.add_argument("--out-dir", type=Path, default=None)
    p.add_argument("--slat-flow-ckpt", type=Path, default=DEFAULT_SLAT_FLOW_CKPT)
    p.add_argument("--slat-gaussian-decoder-ckpt", type=Path, default=DEFAULT_GAUSSIAN_DECODER_CKPT)
    p.add_argument("--gpu", default="0")
    p.add_argument("--shard-id", type=int, default=0)
    p.add_argument("--shard-count", type=int, default=1)
    p.add_argument("--max-samples", type=int, default=0)
    p.add_argument("--slat-steps", type=int, default=25)
    p.add_argument("--nearest-k", type=int, default=4)
    p.add_argument("--render-resolution", type=int, default=384)
    p.add_argument("--force", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    part_mesh_root = args.part_mesh_root.resolve()
    out_root = args.out_dir.resolve() if args.out_dir else part_mesh_root.parent / DEFAULT_OUT_SUBDIR
    out_root.mkdir(parents=True, exist_ok=True)

    summaries = _summary_paths(part_mesh_root)
    selected = [(idx, p) for idx, p in enumerate(summaries, 1) if (idx - 1) % int(args.shard_count) == int(args.shard_id)]
    if int(args.max_samples) > 0:
        selected = selected[: int(args.max_samples)]

    progress_path = out_root / f"progress_appearance_shard_{int(args.shard_id):02d}.jsonl"
    run_meta = {
        "part_mesh_root": str(part_mesh_root),
        "out_root": str(out_root),
        "total_available_summaries": len(summaries),
        "num_samples_this_shard": len(selected),
        "shard_id": int(args.shard_id),
        "shard_count": int(args.shard_count),
        "slat_flow_ckpt": str(Path(args.slat_flow_ckpt).resolve()),
        "slat_gaussian_decoder_ckpt": str(Path(args.slat_gaussian_decoder_ckpt).resolve()),
        "slat_steps": int(args.slat_steps),
        "nearest_k": int(args.nearest_k),
        "render_resolution": int(args.render_resolution),
    }
    (out_root / f"run_meta_appearance_shard_{int(args.shard_id):02d}.json").write_text(
        json.dumps(run_meta, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        f"[appearance] shard {args.shard_id}/{args.shard_count} samples={len(selected)} out={out_root}",
        flush=True,
    )

    failures = 0
    for local_idx, (global_idx, summary_path) in enumerate(selected, 1):
        row = json.loads(summary_path.read_text(encoding="utf-8"))
        start = {
            "status": "started",
            "local_idx": int(local_idx),
            "global_idx": int(global_idx),
            "split": row["split"],
            "dataset_id": row["dataset_id"],
            "obj_id": row["obj_id"],
            "angle": int(row["angle"]),
            "summary_path": str(summary_path.resolve()),
        }
        _progress(progress_path, start)
        print(
            f"[appearance] {local_idx}/{len(selected)} {row['split']} {row['dataset_id']}::{row['obj_id']} angle={int(row['angle'])}",
            flush=True,
        )
        try:
            _export_case(summary_path, out_root, args, progress_path)
        except Exception as exc:
            failures += 1
            rec = {**start, "status": "failed", "error": f"{type(exc).__name__}: {exc}"}
            _progress(progress_path, rec)
            print(f"[appearance] failed {row['obj_id']}: {type(exc).__name__}: {exc}", flush=True)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
