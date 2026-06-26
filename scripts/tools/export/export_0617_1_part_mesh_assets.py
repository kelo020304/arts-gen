#!/usr/bin/env python3
from __future__ import annotations

import argparse
import colorsys
import json
import os
import sys
import time
from collections import OrderedDict
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageDraw


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
    _load_datasets,
    _load_coords,
    _run_dir_for_sample,
    load_or_build_selection,
)
from part_ss_eval_platform.eval_real_0615 import _load_slat_cond_tokens  # noqa: E402
from scripts.tools.render.render_glb_open3d_preview import render as render_open3d  # noqa: E402
from trellis.utils.arts.slat_asset_writer import save_decoded_slat_assets  # noqa: E402


DEFAULT_EVAL_DIR = Path("/mnt/robot-data-lab/jzh/art-gen-output/EE-eval/0617-1")
DEFAULT_OUT_SUBDIR = "part_mesh_assets"


def _palette(n: int = 128) -> list[tuple[int, int, int, int]]:
    colors: list[tuple[int, int, int, int]] = []
    golden = 0.381966011
    for idx in range(n):
        hue = (idx * golden) % 1.0
        sat = 0.78 if idx % 3 != 1 else 0.92
        val = 0.92 if idx % 3 != 2 else 0.76
        rgb = colorsys.hsv_to_rgb(hue, sat, val)
        colors.append(tuple(int(round(x * 255)) for x in rgb) + (255,))
    return colors


PART_COLORS = _palette()
OVERALL_COLOR = (210, 210, 210, 255)


def _sample_ns(row: dict[str, Any]) -> SimpleNamespace:
    return SimpleNamespace(
        split=str(row["split"]),
        dataset_id=str(row.get("dataset_id", "")),
        obj_id=str(row["obj_id"]),
        angle_idx=int(row["angle_idx"]),
        data_root=str(row.get("data_root", "")),
        manifest_path=str(row.get("manifest_path", "")),
    )


def _sample_key(sample: SimpleNamespace) -> str:
    return f"{sample.split}/{sample.dataset_id}/{sample.obj_id}/{int(sample.angle_idx)}"


def _safe_name(value: str, max_len: int = 80) -> str:
    keep = []
    for ch in str(value):
        if ch.isalnum() or ch in ("-", "_", "."):
            keep.append(ch)
        else:
            keep.append("_")
    out = "".join(keep).strip("_")
    return (out or "part")[:max_len]


def _read_selection(args: argparse.Namespace) -> list[SimpleNamespace]:
    samples = load_or_build_selection(args)
    return samples


def _shard(samples: list[SimpleNamespace], shard_id: int, shard_count: int) -> list[tuple[int, SimpleNamespace]]:
    return [(idx, sample) for idx, sample in enumerate(samples, 1) if (idx - 1) % shard_count == shard_id]


def _load_tokens(ds: Any, ds_sample: dict[str, Any]) -> torch.Tensor:
    tokens = _load_slat_cond_tokens(ds, ds_sample)
    if tokens.dim() != 2 or tokens.shape[-1] != 1024:
        raise ValueError(f"expected flattened cond tokens [T,1024], got {tuple(tokens.shape)}")
    return tokens


def _load_part_items(run_dir: Path, ds_sample: dict[str, Any]) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for part_idx, part in enumerate(ds_sample["parts"]):
        path = run_dir / "parts" / f"part_{part_idx:02d}_voxel.npz"
        if not path.is_file():
            continue
        coords = _load_coords(path).astype(np.int64, copy=False)
        if coords.size == 0:
            continue
        items.append(
            {
                "index": int(part_idx),
                "name": str(part.get("part_name", f"part_{part_idx:02d}")),
                "coords": coords,
                "source_path": path,
            }
        )
    return items


def _tint_mesh(decoded: dict[str, Any], rgba: tuple[int, int, int, int]) -> None:
    mesh = decoded.get("mesh")
    if mesh is None:
        return
    vertices = getattr(mesh, "vertices", None)
    if vertices is None:
        return
    n = int(vertices.shape[0]) if hasattr(vertices, "shape") else len(vertices)
    rgb = np.asarray(rgba[:3], dtype=np.float32) / 255.0
    attrs = np.tile(rgb[None, :], (n, 1)).astype(np.float32)
    setattr(mesh, "vertex_attrs", attrs)


def _mesh_counts(mesh: Any) -> tuple[int, int]:
    vertices = getattr(mesh, "vertices", None)
    faces = getattr(mesh, "faces", None)
    nv = int(vertices.shape[0]) if vertices is not None and hasattr(vertices, "shape") else 0
    nf = int(faces.shape[0]) if faces is not None and hasattr(faces, "shape") else 0
    return nv, nf


def _decode_export_mesh(
    *,
    coords: np.ndarray,
    cond_tokens: torch.Tensor,
    label: str,
    color: tuple[int, int, int, int],
    out_dir: Path,
    args: argparse.Namespace,
    seed_offset: int,
) -> dict[str, Any]:
    mesh_path = out_dir / f"{label}.glb"
    render_dir = out_dir / "renders"
    iso_png = render_dir / f"{label}_iso.png"
    meta_path = out_dir / f"{label}.json"
    if mesh_path.is_file() and iso_png.is_file() and meta_path.is_file() and not args.force:
        return json.loads(meta_path.read_text(encoding="utf-8"))

    started = time.time()
    coords_t = torch.from_numpy(np.ascontiguousarray(coords.astype(np.int64, copy=False))).long()
    seed = int(args.slat_seed) + int(seed_offset)
    slat = inference.run_slat_flow_from_tokens(
        cond_tokens,
        coords_t,
        str(Path(args.slat_flow_ckpt).resolve()),
        num_steps=int(args.slat_steps),
        seed=seed,
    )
    decoded = inference.decode_slat_assets(
        slat,
        mesh_decoder_ckpt=str(Path(args.slat_mesh_decoder_ckpt).resolve()),
        slat_is_normalized=True,
    )
    mesh = decoded.get("mesh")
    if mesh is None:
        raise RuntimeError(f"{label}: mesh decoder returned None")
    if not getattr(mesh, "success", True):
        raise RuntimeError(f"{label}: mesh decoder success=False")
    _tint_mesh(decoded, color)
    assets = save_decoded_slat_assets(decoded, out_dir, mesh_name=f"{label}.glb")
    renders = render_open3d(
        mesh_path,
        out_dir=render_dir,
        views={"iso": (315.0, 24.0), "front": (270.0, 8.0), "side": (0.0, 8.0)},
        resolution=int(args.render_resolution),
        use_vertex_colors=True,
    )
    nv, nf = _mesh_counts(mesh)
    rec = {
        "label": label,
        "coords": int(coords.shape[0]),
        "seed": seed,
        "color_rgba": list(color),
        "mesh": str((out_dir / assets["mesh"]).resolve()),
        "renders": [str(path.resolve()) for path in renders],
        "mesh_vertices": nv,
        "mesh_faces": nf,
        "seconds": round(time.time() - started, 3),
    }
    meta_path.write_text(json.dumps(rec, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return rec


def _load_colored_mesh(path: Path, trimesh):
    loaded = trimesh.load(str(path), force="scene", process=False)
    if isinstance(loaded, trimesh.Trimesh):
        return loaded
    meshes = []
    for geom in loaded.geometry.values():
        if isinstance(geom, trimesh.Trimesh) and len(geom.vertices) and len(geom.faces):
            meshes.append(geom)
    if not meshes:
        raise ValueError(f"{path}: no mesh geometry")
    return trimesh.util.concatenate(meshes)


def _render_colored_mesh_matplotlib(
    glb_path: Path,
    *,
    out_dir: Path,
    stem: str,
    resolution: int,
) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    import trimesh

    mesh = _load_colored_mesh(glb_path, trimesh)
    vertices = np.asarray(mesh.vertices, dtype=np.float32)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    colors = getattr(mesh.visual, "vertex_colors", None)
    if colors is None or len(colors) != len(vertices):
        vertex_colors = np.full((len(vertices), 4), 210, dtype=np.uint8)
        vertex_colors[:, 3] = 255
    else:
        vertex_colors = np.asarray(colors, dtype=np.uint8)
        if vertex_colors.shape[1] == 3:
            alpha = np.full((vertex_colors.shape[0], 1), 255, dtype=np.uint8)
            vertex_colors = np.concatenate([vertex_colors, alpha], axis=1)

    lo = vertices.min(axis=0)
    hi = vertices.max(axis=0)
    center = (lo + hi) / 2.0
    span = max(float(np.max(hi - lo)), 1e-3)
    facecolors = vertex_colors[faces][:, :, :4].mean(axis=1) / 255.0
    tri = vertices[faces]

    views = {"iso": (24.0, -45.0), "front": (8.0, -90.0), "side": (8.0, 0.0)}
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for name, (elev, azim) in views.items():
        fig = plt.figure(figsize=(resolution / 100.0, resolution / 100.0), dpi=100)
        ax = fig.add_subplot(111, projection="3d")
        fig.patch.set_facecolor("white")
        ax.set_facecolor("white")
        ax.set_box_aspect((1, 1, 1))
        ax.view_init(elev=elev, azim=azim)
        collection = Poly3DCollection(
            tri,
            facecolors=facecolors,
            edgecolors=(0.03, 0.03, 0.03, 0.05),
            linewidths=0.01,
        )
        collection.set_zsort("average")
        ax.add_collection3d(collection)
        ax.set_xlim(center[0] - span / 2.0, center[0] + span / 2.0)
        ax.set_ylim(center[1] - span / 2.0, center[1] + span / 2.0)
        ax.set_zlim(center[2] - span / 2.0, center[2] + span / 2.0)
        ax.set_axis_off()
        fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
        out = out_dir / f"{stem}_{name}.png"
        fig.savefig(out, facecolor="white", pad_inches=0)
        plt.close(fig)
        written.append(out)
    return written


def _assemble_parts(part_records: list[dict[str, Any]], out_dir: Path, args: argparse.Namespace) -> dict[str, Any]:
    import trimesh

    out_dir.mkdir(parents=True, exist_ok=True)
    glb_path = out_dir / "assembled_parts.glb"
    iso_png = out_dir / "renders" / "assembled_parts_iso.png"
    if glb_path.is_file() and iso_png.is_file() and not args.force:
        return {
            "mesh": str(glb_path.resolve()),
            "renders": [str(p.resolve()) for p in sorted((out_dir / "renders").glob("assembled_parts_*.png"))],
        }
    meshes = [_load_colored_mesh(Path(rec["mesh"]), trimesh) for rec in part_records]
    if not meshes:
        raise RuntimeError("cannot assemble empty part mesh list")
    assembled = trimesh.util.concatenate(meshes)
    assembled.export(str(glb_path))
    renders = _render_colored_mesh_matplotlib(
        glb_path,
        out_dir=out_dir / "renders",
        resolution=int(args.render_resolution),
        stem="assembled_parts",
    )
    return {"mesh": str(glb_path.resolve()), "renders": [str(p.resolve()) for p in renders]}


def _make_panel(case_dir: Path, overall: dict[str, Any], parts: list[dict[str, Any]], assembled: dict[str, Any]) -> Path:
    def iso(record: dict[str, Any]) -> Path:
        for raw in record.get("renders", []):
            p = Path(raw)
            if p.name.endswith("_iso.png"):
                return p
        raise RuntimeError(f"missing iso render in {record.get('label', 'record')}")

    entries: list[tuple[str, Path]] = [("overall", iso(overall)), ("assembled_parts", iso(assembled))]
    for rec in parts:
        entries.append((str(rec["label"]), iso(rec)))

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
    out = case_dir / "panel_parts.png"
    canvas.save(out)
    return out


def _progress(path: Path, rec: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def _case_dir(out_root: Path, sample: SimpleNamespace) -> Path:
    return out_root / sample.split / sample.dataset_id / f"{sample.obj_id}-{int(sample.angle_idx)}"


def export_sample(
    *,
    global_idx: int,
    sample: SimpleNamespace,
    datasets: dict[str, Any],
    eval_dir: Path,
    out_root: Path,
    args: argparse.Namespace,
    progress_path: Path,
) -> dict[str, Any]:
    ds = _dataset_for_sample(datasets, sample)
    ds_sample = _find_dataset_sample(ds, sample)
    run_dir = _run_dir_for_sample(eval_dir, sample)
    whole_path = run_dir / "voxel.npz"
    if not whole_path.is_file():
        raise FileNotFoundError(f"missing whole voxel: {whole_path}")
    part_items = _load_part_items(run_dir, ds_sample)
    if not part_items:
        raise FileNotFoundError(f"no part voxel files under {run_dir / 'parts'}")
    cond_tokens = _load_tokens(ds, ds_sample)
    case_dir = _case_dir(out_root, sample)
    case_dir.mkdir(parents=True, exist_ok=True)

    started = time.time()
    whole_coords = _load_coords(whole_path).astype(np.int64, copy=False)
    overall = _decode_export_mesh(
        coords=whole_coords,
        cond_tokens=cond_tokens,
        label="overall",
        color=OVERALL_COLOR,
        out_dir=case_dir / "overall",
        args=args,
        seed_offset=global_idx * 1009,
    )

    part_records: list[dict[str, Any]] = []
    for part in part_items:
        label = f"part_{part['index']:02d}_{_safe_name(part['name'])}"
        color = PART_COLORS[int(part["index"]) % len(PART_COLORS)]
        rec = _decode_export_mesh(
            coords=part["coords"],
            cond_tokens=cond_tokens,
            label=label,
            color=color,
            out_dir=case_dir / "parts" / label,
            args=args,
            seed_offset=global_idx * 1009 + (int(part["index"]) + 1) * 9176,
        )
        rec.update(
            {
                "part_index": int(part["index"]),
                "part_name": str(part["name"]),
                "source_voxel": str(Path(part["source_path"]).resolve()),
            }
        )
        part_records.append(rec)

    assembled = _assemble_parts(part_records, case_dir / "assembled_parts", args)
    panel = _make_panel(case_dir, overall, part_records, assembled)
    summary = {
        "status": "done",
        "global_idx": int(global_idx),
        "split": sample.split,
        "dataset_id": sample.dataset_id,
        "obj_id": sample.obj_id,
        "angle": int(sample.angle_idx),
        "run_dir": str(run_dir.resolve()),
        "case_dir": str(case_dir.resolve()),
        "overall": overall,
        "parts": part_records,
        "assembled_parts": assembled,
        "panel": str(panel.resolve()),
        "seconds": round(time.time() - started, 3),
    }
    (case_dir / "part_mesh_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    _progress(progress_path, summary)
    return summary


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Export true per-part SLat meshes for EE-eval 0617-1.")
    p.add_argument("--eval-dir", type=Path, default=DEFAULT_EVAL_DIR)
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
    p.add_argument("--slat-steps", type=int, default=25)
    p.add_argument("--slat-seed", type=int, default=42)
    p.add_argument("--render-resolution", type=int, default=512)
    p.add_argument("--max-samples", type=int, default=0)
    p.add_argument("--force", action="store_true")
    p.add_argument("--overwrite-selection", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    eval_dir = args.eval_dir.resolve()
    if args.out_dir is None:
        out_root = eval_dir / DEFAULT_OUT_SUBDIR
    else:
        out_root = args.out_dir.resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    # load_or_build_selection expects args.out_dir to point at the eval directory.
    selection_args = argparse.Namespace(**vars(args))
    selection_args.out_dir = str(eval_dir)
    samples = _read_selection(selection_args)
    selected = _shard(samples, int(args.shard_id), int(args.shard_count))
    if int(args.max_samples) > 0:
        selected = selected[: int(args.max_samples)]

    datasets, dataset_meta = _load_datasets(selection_args)
    progress_path = out_root / f"progress_part_mesh_shard_{int(args.shard_id):02d}.jsonl"
    run_meta = {
        "eval_dir": str(eval_dir),
        "out_root": str(out_root),
        "num_samples_total": len(samples),
        "num_samples_this_shard": len(selected),
        "shard_id": int(args.shard_id),
        "shard_count": int(args.shard_count),
        "datasets": dataset_meta,
        "slat_flow_ckpt": str(Path(args.slat_flow_ckpt).resolve()),
        "slat_mesh_decoder_ckpt": str(Path(args.slat_mesh_decoder_ckpt).resolve()),
        "slat_steps": int(args.slat_steps),
        "slat_seed": int(args.slat_seed),
        "render_resolution": int(args.render_resolution),
    }
    (out_root / f"run_meta_shard_{int(args.shard_id):02d}.json").write_text(
        json.dumps(run_meta, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        f"[part_mesh] shard {args.shard_id}/{args.shard_count} samples={len(selected)} "
        f"out={out_root}",
        flush=True,
    )

    failures = 0
    for local_idx, (global_idx, sample) in enumerate(selected, 1):
        start_rec = {
            "status": "started",
            "local_idx": local_idx,
            "global_idx": int(global_idx),
            "split": sample.split,
            "dataset_id": sample.dataset_id,
            "obj_id": sample.obj_id,
            "angle": int(sample.angle_idx),
            "sample_key": _sample_key(sample),
        }
        _progress(progress_path, start_rec)
        print(
            f"[part_mesh] {local_idx}/{len(selected)} global={global_idx}/{len(samples)} "
            f"{sample.split} {sample.dataset_id}::{sample.obj_id} angle={int(sample.angle_idx)}",
            flush=True,
        )
        try:
            export_sample(
                global_idx=global_idx,
                sample=sample,
                datasets=datasets,
                eval_dir=eval_dir,
                out_root=out_root,
                args=args,
                progress_path=progress_path,
            )
        except Exception as exc:
            failures += 1
            rec = {
                **start_rec,
                "status": "failed",
                "error": f"{type(exc).__name__}: {exc}",
            }
            _progress(progress_path, rec)
            print(f"[part_mesh] failed {sample.obj_id}: {type(exc).__name__}: {exc}", flush=True)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
