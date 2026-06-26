#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import torch
import trimesh
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
os.environ.setdefault("OPEN3D_CPU_RENDERING", "true")

import inference  # noqa: E402
from part_ss_eval_platform.eval_0617_1 import (  # noqa: E402
    DEFAULT_DATA_CONFIG,
    DEFAULT_SLAT_FLOW_CKPT,
    DEFAULT_SLAT_MESH_DECODER_CKPT,
    DEFAULT_SPLIT_JSON_0617,
    _dataset_for_sample,
    _find_dataset_sample,
    _load_coords,
    _load_datasets,
    _run_dir_for_sample,
    load_or_build_selection,
)
from scripts.tools.export.export_0617_1_part_mesh_assets import (  # noqa: E402
    OVERALL_COLOR,
    PART_COLORS,
    _load_part_items,
    _load_tokens,
    _safe_name,
    _shard,
    _tint_mesh,
)
from trellis.utils.arts.slat_asset_writer import save_decoded_slat_assets  # noqa: E402


DEFAULT_EVAL_DIR = Path("/mnt/robot-data-lab/jzh/art-gen-output/EE-eval/0617-1")
DEFAULT_REUSE_ROOT = DEFAULT_EVAL_DIR / "part_mesh_assets"
DEFAULT_OUT_SUBDIR = "part_mesh_flat"


def _sample_key(sample: Any) -> str:
    return f"{sample.split}/{sample.dataset_id}/{sample.obj_id}/{int(sample.angle_idx)}"


def _case_dir(out_root: Path, sample: Any) -> Path:
    return out_root / sample.split / sample.dataset_id / f"{sample.obj_id}-{int(sample.angle_idx)}"


def _reuse_summary_path(reuse_root: Path, sample: Any) -> Path:
    return reuse_root / sample.split / sample.dataset_id / f"{sample.obj_id}-{int(sample.angle_idx)}" / "part_mesh_summary.json"


def _link_or_copy(src: Path, dst: Path, *, force: bool) -> None:
    src = Path(src)
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        if not force:
            return
        dst.unlink()
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def _load_mesh(path: Path) -> trimesh.Trimesh:
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


def _mesh_facecolors(mesh: trimesh.Trimesh) -> np.ndarray:
    colors = getattr(mesh.visual, "vertex_colors", None)
    vertices = np.asarray(mesh.vertices)
    faces = np.asarray(mesh.faces)
    if colors is None or len(colors) != len(vertices):
        vertex_colors = np.full((len(vertices), 4), 210, dtype=np.uint8)
        vertex_colors[:, 3] = 255
    else:
        vertex_colors = np.asarray(colors, dtype=np.uint8)
        if vertex_colors.shape[1] == 3:
            alpha = np.full((len(vertex_colors), 1), 255, dtype=np.uint8)
            vertex_colors = np.concatenate([vertex_colors, alpha], axis=1)
    return vertex_colors[faces][:, :, :4].mean(axis=1) / 255.0


def _render_mesh_cell(
    glb_path: Path,
    *,
    title: str,
    resolution: int,
    azim: float,
    elev: float,
    max_faces: int,
) -> Image.Image:
    mesh = _load_mesh(glb_path)
    vertices = np.asarray(mesh.vertices, dtype=np.float32)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    facecolors = _mesh_facecolors(mesh)
    if len(faces) > int(max_faces):
        rng = np.random.default_rng(abs(hash(str(glb_path))) % (2**32))
        keep = np.sort(rng.choice(len(faces), size=int(max_faces), replace=False))
        faces = faces[keep]
        facecolors = facecolors[keep]

    tri = vertices[faces]
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
        facecolors=facecolors,
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
    draw.rectangle((0, 0, image.width, 25), fill=(255, 255, 255))
    draw.text((6, 6), title[:48], fill=(20, 20, 20))
    return image


def _make_overview(
    case_dir: Path,
    *,
    assembled_glb: Path,
    part_glbs: list[tuple[str, Path]],
    args: argparse.Namespace,
) -> Path:
    entries: list[tuple[str, Path]] = [("overall", assembled_glb)]
    entries.extend(part_glbs)
    tile = int(args.tile_resolution)
    pad = 8
    cols = min(int(args.panel_cols), max(1, len(entries)))
    rows = int(np.ceil(len(entries) / cols))
    canvas = Image.new("RGB", (cols * tile + (cols + 1) * pad, rows * tile + (rows + 1) * pad), (245, 245, 245))
    for idx, (title, glb) in enumerate(entries):
        image = _render_mesh_cell(
            glb,
            title=title,
            resolution=tile,
            azim=float(args.view_azim),
            elev=float(args.view_elev),
            max_faces=int(args.preview_max_faces),
        )
        x = pad + (idx % cols) * (tile + pad)
        y = pad + (idx // cols) * (tile + pad)
        canvas.paste(image, (x, y))
    out = case_dir / "overview_parts.png"
    canvas.save(out)
    return out


def _assemble_parts(part_glbs: list[tuple[str, Path]], out_path: Path, *, force: bool) -> None:
    if out_path.is_file() and not force:
        return
    meshes = [_load_mesh(path) for _, path in part_glbs]
    if not meshes:
        raise RuntimeError("cannot assemble zero part meshes")
    assembled = trimesh.util.concatenate(meshes)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    assembled.export(str(out_path))


def _decode_mesh_only(
    *,
    coords: np.ndarray,
    cond_tokens: torch.Tensor,
    label: str,
    color: tuple[int, int, int, int],
    out_path: Path,
    seed: int,
    args: argparse.Namespace,
) -> dict[str, Any]:
    if out_path.is_file() and not args.force:
        return {"label": label, "seed": int(seed), "mesh": str(out_path.resolve()), "reused": True}
    coords_t = torch.from_numpy(np.ascontiguousarray(coords.astype(np.int64, copy=False))).long()
    last_error: str | None = None
    for attempt in range(int(args.mesh_retry_count) + 1):
        actual_seed = int(seed) + attempt * 100003
        try:
            slat = inference.run_slat_flow_from_tokens(
                cond_tokens,
                coords_t,
                str(Path(args.slat_flow_ckpt).resolve()),
                num_steps=int(args.slat_steps),
                seed=actual_seed,
            )
            decoded = inference.decode_slat_assets(
                slat,
                mesh_decoder_ckpt=str(Path(args.slat_mesh_decoder_ckpt).resolve()),
                slat_is_normalized=True,
            )
            mesh = decoded.get("mesh")
            if mesh is None:
                raise RuntimeError("mesh decoder returned None")
            if not getattr(mesh, "success", True):
                raise RuntimeError("mesh decoder success=False")
            _tint_mesh(decoded, color)
            save_decoded_slat_assets(decoded, out_path.parent, mesh_name=out_path.name)
            return {
                "label": label,
                "seed": actual_seed,
                "mesh": str(out_path.resolve()),
                "coords": int(coords.shape[0]),
                "retry_attempt": int(attempt),
            }
        except Exception as exc:
            last_error = f"{type(exc).__name__}: {exc}"
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    raise RuntimeError(f"{label}: failed after retries: {last_error}")


def _flatten_reuse(summary_path: Path, case_dir: Path, args: argparse.Namespace) -> dict[str, Any]:
    src = json.loads(summary_path.read_text(encoding="utf-8"))
    case_dir.mkdir(parents=True, exist_ok=True)
    overall_dst = case_dir / "overall.glb"
    _link_or_copy(Path(src["overall"]["mesh"]), overall_dst, force=bool(args.force))

    part_records: list[dict[str, Any]] = []
    part_glbs: list[tuple[str, Path]] = []
    for part in src["parts"]:
        label = str(part["label"])
        dst = case_dir / f"{label}.glb"
        _link_or_copy(Path(part["mesh"]), dst, force=bool(args.force))
        part_glbs.append((label, dst))
        part_records.append(
            {
                "label": label,
                "part_index": int(part["part_index"]),
                "part_name": str(part["part_name"]),
                "mesh": str(dst.resolve()),
                "source_voxel": str(part.get("source_voxel", "")),
                "mesh_vertices": int(part.get("mesh_vertices", 0)),
                "mesh_faces": int(part.get("mesh_faces", 0)),
            }
        )

    assembled_dst = case_dir / "assembled_parts.glb"
    src_assembled = Path(src.get("assembled_parts", {}).get("mesh", ""))
    if src_assembled.is_file():
        _link_or_copy(src_assembled, assembled_dst, force=bool(args.force))
    else:
        _assemble_parts(part_glbs, assembled_dst, force=bool(args.force))

    overview = _make_overview(case_dir, assembled_glb=assembled_dst, part_glbs=part_glbs, args=args)
    return {
        "status": "done",
        "source": "reused_part_mesh_assets",
        "source_summary": str(summary_path.resolve()),
        "split": src["split"],
        "dataset_id": src["dataset_id"],
        "obj_id": src["obj_id"],
        "angle": int(src["angle"]),
        "overall": str(overall_dst.resolve()),
        "assembled_parts": str(assembled_dst.resolve()),
        "parts": part_records,
        "overview": str(overview.resolve()),
    }


def _export_fresh(global_idx: int, sample: Any, datasets: dict[str, Any], eval_dir: Path, case_dir: Path, args: argparse.Namespace) -> dict[str, Any]:
    ds = _dataset_for_sample(datasets, sample)
    ds_sample = _find_dataset_sample(ds, sample)
    run_dir = _run_dir_for_sample(eval_dir, sample)
    cond_tokens = _load_tokens(ds, ds_sample)
    whole_path = run_dir / "voxel.npz"
    if not whole_path.is_file():
        raise FileNotFoundError(f"missing whole voxel: {whole_path}")
    part_items = _load_part_items(run_dir, ds_sample)
    if not part_items:
        raise FileNotFoundError(f"no part voxel files under {run_dir / 'parts'}")

    case_dir.mkdir(parents=True, exist_ok=True)
    overall_seed = int(args.slat_seed) + int(global_idx) * 1009
    overall_rec = _decode_mesh_only(
        coords=_load_coords(whole_path).astype(np.int64, copy=False),
        cond_tokens=cond_tokens,
        label="overall",
        color=OVERALL_COLOR,
        out_path=case_dir / "overall.glb",
        seed=overall_seed,
        args=args,
    )

    part_records: list[dict[str, Any]] = []
    part_glbs: list[tuple[str, Path]] = []
    for part in part_items:
        label = f"part_{int(part['index']):02d}_{_safe_name(part['name'])}"
        seed = int(args.slat_seed) + int(global_idx) * 1009 + (int(part["index"]) + 1) * 9176
        dst = case_dir / f"{label}.glb"
        rec = _decode_mesh_only(
            coords=part["coords"],
            cond_tokens=cond_tokens,
            label=label,
            color=PART_COLORS[int(part["index"]) % len(PART_COLORS)],
            out_path=dst,
            seed=seed,
            args=args,
        )
        rec.update(
            {
                "part_index": int(part["index"]),
                "part_name": str(part["name"]),
                "source_voxel": str(Path(part["source_path"]).resolve()),
            }
        )
        part_records.append(rec)
        part_glbs.append((label, dst))

    assembled_dst = case_dir / "assembled_parts.glb"
    _assemble_parts(part_glbs, assembled_dst, force=bool(args.force))
    overview = _make_overview(case_dir, assembled_glb=assembled_dst, part_glbs=part_glbs, args=args)
    return {
        "status": "done",
        "source": "fresh_decode",
        "split": sample.split,
        "dataset_id": sample.dataset_id,
        "obj_id": sample.obj_id,
        "angle": int(sample.angle_idx),
        "run_dir": str(run_dir.resolve()),
        "overall": overall_rec["mesh"],
        "assembled_parts": str(assembled_dst.resolve()),
        "parts": part_records,
        "overview": str(overview.resolve()),
    }


def _progress(path: Path, rec: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Flat per-object part mesh export with one overview PNG per object.")
    p.add_argument("--eval-dir", type=Path, default=DEFAULT_EVAL_DIR)
    p.add_argument("--reuse-root", type=Path, default=DEFAULT_REUSE_ROOT)
    p.add_argument("--out-dir", type=Path, default=None)
    p.add_argument("--data-config", type=Path, default=DEFAULT_DATA_CONFIG)
    p.add_argument("--split-json", type=Path, default=DEFAULT_SPLIT_JSON_0617)
    p.add_argument("--slat-flow-ckpt", type=Path, default=DEFAULT_SLAT_FLOW_CKPT)
    p.add_argument("--slat-mesh-decoder-ckpt", type=Path, default=DEFAULT_SLAT_MESH_DECODER_CKPT)
    p.add_argument("--train-count", type=int, default=85)
    p.add_argument("--held-count", type=int, default=43)
    p.add_argument("--gpu", default="0")
    p.add_argument("--shard-id", type=int, default=0)
    p.add_argument("--shard-count", type=int, default=1)
    p.add_argument("--max-samples", type=int, default=0)
    p.add_argument("--slat-steps", type=int, default=25)
    p.add_argument("--slat-seed", type=int, default=42)
    p.add_argument("--mesh-retry-count", type=int, default=2)
    p.add_argument("--tile-resolution", type=int, default=320)
    p.add_argument("--panel-cols", type=int, default=5)
    p.add_argument("--view-azim", type=float, default=-45.0)
    p.add_argument("--view-elev", type=float, default=24.0)
    p.add_argument("--preview-max-faces", type=int, default=45000)
    p.add_argument("--force", action="store_true")
    p.add_argument("--overwrite-selection", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    eval_dir = args.eval_dir.resolve()
    reuse_root = args.reuse_root.resolve()
    out_root = args.out_dir.resolve() if args.out_dir else eval_dir / DEFAULT_OUT_SUBDIR
    out_root.mkdir(parents=True, exist_ok=True)

    selection_args = argparse.Namespace(**vars(args))
    selection_args.out_dir = str(eval_dir)
    samples = load_or_build_selection(selection_args)
    selected = _shard(samples, int(args.shard_id), int(args.shard_count))
    if int(args.max_samples) > 0:
        selected = selected[: int(args.max_samples)]

    datasets = None
    progress_path = out_root / f"progress_flat_shard_{int(args.shard_id):02d}.jsonl"
    run_meta = {
        "eval_dir": str(eval_dir),
        "reuse_root": str(reuse_root),
        "out_root": str(out_root),
        "num_samples_total": len(samples),
        "num_samples_this_shard": len(selected),
        "shard_id": int(args.shard_id),
        "shard_count": int(args.shard_count),
        "view_azim": float(args.view_azim),
        "view_elev": float(args.view_elev),
        "flat_contract": [
            "overall.glb",
            "assembled_parts.glb",
            "part_XX_name.glb",
            "overview_parts.png",
            "summary.json",
        ],
    }
    (out_root / f"run_meta_flat_shard_{int(args.shard_id):02d}.json").write_text(
        json.dumps(run_meta, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"[flat] shard {args.shard_id}/{args.shard_count} samples={len(selected)} out={out_root}", flush=True)

    failures = 0
    for local_idx, (global_idx, sample) in enumerate(selected, 1):
        case_dir = _case_dir(out_root, sample)
        summary_path = case_dir / "summary.json"
        start = {
            "status": "started",
            "local_idx": int(local_idx),
            "global_idx": int(global_idx),
            "sample_key": _sample_key(sample),
            "case_dir": str(case_dir.resolve()),
        }
        _progress(progress_path, start)
        print(
            f"[flat] {local_idx}/{len(selected)} global={global_idx}/{len(samples)} "
            f"{sample.split} {sample.dataset_id}::{sample.obj_id} angle={int(sample.angle_idx)}",
            flush=True,
        )
        try:
            if summary_path.is_file() and not args.force:
                out = json.loads(summary_path.read_text(encoding="utf-8"))
            else:
                reused = _reuse_summary_path(reuse_root, sample)
                if reused.is_file():
                    out = _flatten_reuse(reused, case_dir, args)
                else:
                    if datasets is None:
                        datasets, _dataset_meta = _load_datasets(selection_args)
                    out = _export_fresh(global_idx, sample, datasets, eval_dir, case_dir, args)
                summary_path.write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            _progress(progress_path, {**start, "status": "done", "overview": out.get("overview")})
        except Exception as exc:
            failures += 1
            rec = {**start, "status": "failed", "error": f"{type(exc).__name__}: {exc}"}
            _progress(progress_path, rec)
            print(f"[flat] failed {sample.obj_id}: {type(exc).__name__}: {exc}", flush=True)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
