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

import numpy as np
import torch
from PIL import Image, ImageDraw


REPO_ROOT = Path(__file__).resolve().parents[2]
TRELLIS_ROOT = REPO_ROOT / "TRELLIS-arts"
for item in (str(REPO_ROOT), str(TRELLIS_ROOT)):
    if item not in sys.path:
        sys.path.insert(0, item)

os.environ.setdefault("SPCONV_ALGO", "native")
os.environ.setdefault("ATTN_BACKEND", "sdpa")
os.environ.setdefault("SPARSE_ATTN_BACKEND", "sdpa")

import inference  # noqa: E402
from part_ss_eval_platform.eval_0617_1 import (  # noqa: E402
    DEFAULT_DATA_CONFIG,
    DEFAULT_PART_SEG_CKPT,
    DEFAULT_SLAT_FLOW_CKPT,
    DEFAULT_SS_DECODER_CKPT,
    DEFAULT_SS_FLOW_CKPT,
    _command_for_sample,
    _find_dataset_sample,
    _load_datasets,
    _run_dir_for_sample,
    _sample_data_config_path,
)
from part_ss_eval_platform.eval_real_0615 import _execute, _load_coords  # noqa: E402
from scripts.tools.export.export_0617_test_textured_part_mesh import (  # noqa: E402
    _load_slat_cond_tokens_for_views,
    _safe_name,
    _sparse_subset_from_coords,
)
from scripts.tools.roundtrip.trellis_full_voxel_mesh_roundtrip import load_camera_matrices  # noqa: E402


def _restore_trellis_renderer_package() -> None:
    for name in list(sys.modules):
        if name == "trellis.renderers" or name.startswith("trellis.renderers."):
            sys.modules.pop(name, None)
    import trellis.renderers  # noqa: F401


_restore_trellis_renderer_package()
from trellis.renderers.gaussian_render import GaussianRenderer  # noqa: E402


DEFAULT_OUT_DIR = Path("/mnt/robot-data-lab/jzh/art-gen-output/EE-eval/0617-realappliance-gaussian")
DEFAULT_SPLIT_JSON = Path(
    "/mnt/robot-data-lab/jzh/art-gen-output/part_promptable_seg/manifests/split_official_verse_realappliance_v3.json"
)
DEFAULT_GAUSSIAN_DECODER_CKPT = (
    REPO_ROOT / "pretrained/TRELLIS-image-large/ckpts/slat_dec_gs_swin8_B_64l8gs32_fp16.safetensors"
)


def _find_sample(ds: Any, object_id: str, angle: int, dataset_id: str) -> SimpleNamespace:
    for row in ds.samples:
        if str(row["obj_id"]) == object_id and int(row["angle_idx"]) == int(angle):
            return SimpleNamespace(
                split=str(row.get("split", "train")),
                dataset_id=dataset_id,
                obj_id=object_id,
                angle_idx=int(angle),
                data_root=str(row.get("_eval_data_root") or ds.data_root),
                manifest_path=str(row.get("_eval_manifest_path") or ds.manifest_path),
            )
    raise KeyError(f"{dataset_id}::{object_id} angle={angle} not found")


def _progress(path: Path, rec: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(rec, ensure_ascii=False) + "\n")


def _ensure_ss_and_part(args: argparse.Namespace, ds: Any, sample: SimpleNamespace, ds_sample: dict[str, Any]) -> Path:
    run_dir = _run_dir_for_sample(args.out_dir, sample)
    expected_parts = len(ds_sample["parts"])
    progress_path = args.out_dir / "progress_gaussian.jsonl"
    ss_done = (run_dir / "ss_latent.npy").is_file() and (run_dir / "voxel.npz").is_file()
    part_done = len(list((run_dir / "parts").glob("part_*_voxel.npz"))) >= expected_parts
    local_args = argparse.Namespace(**vars(args))
    local_args.data_config = str(_sample_data_config_path(args.out_dir, sample, ds))

    for stage, done in (("ss", ss_done), ("part", part_done)):
        if done and not args.force_stage:
            _progress(progress_path, {"stage": stage, "status": "skipped", "run_dir": str(run_dir)})
            continue
        spec = _command_for_sample(args.out_dir, sample, local_args, stage, ds)
        rec = _execute(
            spec,
            gpu=str(args.gpu),
            progress_path=progress_path,
            label=f"realappliance-gaussian/{stage}/{sample.dataset_id}/{sample.obj_id}/{int(sample.angle_idx)}",
        )
        rec["status"] = "done"
        _progress(progress_path, rec)
    if not (run_dir / "voxel.npz").is_file():
        raise FileNotFoundError(f"missing whole voxel after ss stage: {run_dir / 'voxel.npz'}")
    if len(list((run_dir / "parts").glob("part_*_voxel.npz"))) < expected_parts:
        raise FileNotFoundError(f"missing part voxels after part stage: {run_dir / 'parts'}")
    return run_dir


def _make_renderer(resolution: int, kernel_size: float) -> GaussianRenderer:
    renderer = GaussianRenderer()
    renderer.rendering_options.resolution = int(resolution)
    renderer.rendering_options.near = 0.8
    renderer.rendering_options.far = 1.6
    renderer.rendering_options.bg_color = (0, 0, 0)
    renderer.rendering_options.ssaa = 1
    renderer.pipe.kernel_size = float(kernel_size)
    return renderer


@torch.no_grad()
def _decode_render_gaussian(
    *,
    slat: Any,
    label: str,
    out_png: Path,
    args: argparse.Namespace,
    extrinsic: torch.Tensor,
    intrinsic: torch.Tensor,
) -> dict[str, Any]:
    started = time.time()
    decoded = inference.decode_slat_assets(
        slat,
        gaussian_decoder_ckpt=str(args.slat_gaussian_decoder_ckpt.resolve()),
        slat_is_normalized=True,
    )
    gaussian = decoded.get("gaussian")
    if gaussian is None:
        raise RuntimeError(f"{label}: gaussian decoder returned None")
    renderer = _make_renderer(int(args.render_resolution), float(args.gaussian_kernel_size))
    color = renderer.render(gaussian, extrinsic, intrinsic)["color"].detach().float().cpu().clamp(0, 1)
    arr = (color.permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)
    image = Image.fromarray(arr).convert("RGB")
    out_png.parent.mkdir(parents=True, exist_ok=True)
    image.save(out_png)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return {
        "label": label,
        "png": str(out_png.resolve()),
        "gaussians": int(gaussian.get_xyz.shape[0]),
        "seconds": round(time.time() - started, 3),
        "render_backend": "TRELLIS GaussianRenderer direct render; no nvdiffrast bake",
    }


def _labeled_tile(path: Path, label: str, size: int) -> Image.Image:
    image = Image.open(path).convert("RGB")
    image.thumbnail((size, size), Image.Resampling.LANCZOS)
    tile = Image.new("RGB", (size, size + 28), (255, 255, 255))
    draw = ImageDraw.Draw(tile)
    draw.rectangle((0, 0, size, 28), fill=(0, 0, 0))
    draw.text((6, 8), label[:52], fill=(255, 255, 255))
    tile.paste(image, ((size - image.width) // 2, 28 + (size - image.height) // 2))
    return tile


def _make_overview(case_dir: Path, records: list[dict[str, Any]], tile_size: int, cols: int) -> Path:
    cols = max(1, min(int(cols), len(records)))
    rows = int(np.ceil(len(records) / cols))
    canvas = Image.new("RGB", (cols * tile_size, rows * (tile_size + 28)), (255, 255, 255))
    for idx, rec in enumerate(records):
        tile = _labeled_tile(Path(rec["png"]), str(rec["label"]), tile_size)
        canvas.paste(tile, ((idx % cols) * tile_size, (idx // cols) * (tile_size + 28)))
    out = case_dir / "overview_gaussian.png"
    canvas.save(out)
    return out


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RealAppliance whole-once SLat, per-part Gaussian direct renders.")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--data-config", type=Path, default=DEFAULT_DATA_CONFIG)
    parser.add_argument("--split-json", type=Path, default=DEFAULT_SPLIT_JSON)
    parser.add_argument("--dataset-id", default="realappliance")
    parser.add_argument("--object-id", default="039")
    parser.add_argument("--angle", type=int, default=0)
    parser.add_argument("--part-seg-ckpt", type=Path, default=DEFAULT_PART_SEG_CKPT)
    parser.add_argument("--ss-flow-ckpt", type=Path, default=DEFAULT_SS_FLOW_CKPT)
    parser.add_argument("--ss-decoder-ckpt", type=Path, default=DEFAULT_SS_DECODER_CKPT)
    parser.add_argument("--slat-flow-ckpt", type=Path, default=DEFAULT_SLAT_FLOW_CKPT)
    parser.add_argument("--slat-gaussian-decoder-ckpt", type=Path, default=DEFAULT_GAUSSIAN_DECODER_CKPT)
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--slat-steps", type=int, default=25)
    parser.add_argument("--slat-seed", type=int, default=42)
    parser.add_argument("--slat-view-indices", type=int, nargs="+", default=[0])
    parser.add_argument("--render-view", type=int, default=0)
    parser.add_argument("--render-resolution", type=int, default=512)
    parser.add_argument("--gaussian-kernel-size", type=float, default=0.1)
    parser.add_argument("--tile-size", type=int, default=320)
    parser.add_argument("--panel-cols", type=int, default=4)
    parser.add_argument("--force-stage", action="store_true")
    parser.add_argument("--force-export", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    args.out_dir = args.out_dir.resolve()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for attr in (
        "data_config",
        "split_json",
        "part_seg_ckpt",
        "ss_flow_ckpt",
        "ss_decoder_ckpt",
        "slat_flow_ckpt",
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
    overall_coords = _load_coords(run_dir / "voxel.npz")
    overall_coords_t = torch.from_numpy(np.ascontiguousarray(overall_coords.astype(np.int64, copy=False))).long()
    print(
        f"[gaussian] SLat flow ONCE whole object coords={overall_coords.shape[0]} "
        f"views={slat_cond_meta['view_indices']} seed={int(args.slat_seed)}",
        flush=True,
    )
    overall_slat = inference.run_slat_flow_from_tokens(
        cond_tokens,
        overall_coords_t,
        str(args.slat_flow_ckpt.resolve()),
        num_steps=int(args.slat_steps),
        seed=int(args.slat_seed),
    )
    slat_flow_calls = 1

    camera_path = Path(ds.data_root) / ds.recon_subdir / ".."
    data_root = Path(ds.data_root)
    extrinsics, intrinsics = load_camera_matrices(
        data_root / "renders" / sample.obj_id / f"angle_{int(sample.angle_idx)}" / "camera_transforms.json",
        [int(args.render_view)],
    )
    extrinsic = extrinsics[0]
    intrinsic = intrinsics[0]

    components: list[tuple[str, Path, str, Any, int | None, str]] = [
        ("overall", run_dir / "voxel.npz", "overall.png", overall_slat, int(overall_coords.shape[0]), "whole_slat_flow_once"),
    ]
    for out_idx, part in enumerate(ds_sample["parts"]):
        label = f"part_{out_idx:02d}_{_safe_name(str(part['part_name']))}"
        coords_path = run_dir / "parts" / f"part_{out_idx:02d}_voxel.npz"
        coords = _load_coords(coords_path)
        part_slat, matched = _sparse_subset_from_coords(overall_slat, coords, label)
        components.append(
            (label, coords_path, f"part_{out_idx}.png", part_slat, matched, "subset_from_whole_slat_by_coords")
        )

    records: list[dict[str, Any]] = []
    for label, coords_path, filename, slat, matched_coords, slat_source in components:
        out_png = case_dir / filename
        coords_count = int(_load_coords(coords_path).shape[0])
        if out_png.is_file() and not args.force_export:
            rec = {
                "label": label,
                "png": str(out_png.resolve()),
                "coords": coords_count,
                "matched_coords": matched_coords,
                "slat_source": slat_source,
                "render_backend": "TRELLIS GaussianRenderer direct render; no nvdiffrast bake",
                "reused": True,
            }
            records.append(rec)
            continue
        print(
            f"[gaussian] decode+render {label} coords={coords_count} "
            f"matched={matched_coords} source={slat_source} -> {out_png}",
            flush=True,
        )
        rec = _decode_render_gaussian(
            slat=slat,
            label=label,
            out_png=out_png,
            args=args,
            extrinsic=extrinsic,
            intrinsic=intrinsic,
        )
        rec.update(
            {
                "coords": coords_count,
                "matched_coords": matched_coords,
                "coords_path": str(coords_path.resolve()),
                "slat_source": slat_source,
            }
        )
        records.append(rec)

    overview = _make_overview(case_dir, records, int(args.tile_size), int(args.panel_cols))
    summary = {
        "status": "done",
        "dataset_id": args.dataset_id,
        "obj_id": sample.obj_id,
        "angle": int(sample.angle_idx),
        "case_dir": str(case_dir.resolve()),
        "run_dir": str(run_dir.resolve()),
        "overview": str(overview.resolve()),
        "components": records,
        "part_names": [str(part["part_name"]) for part in ds_sample["parts"]],
        "slat_flow_calls": slat_flow_calls,
        "slat_part_rule": "SLat flow is run once on whole voxel coords; each part SparseTensor is sliced from the whole SLat by matching coords[:, 1:].",
        "render_backend": "TRELLIS GaussianRenderer direct render only; nvdiffrast texture bake is not used.",
        "slat_condition": slat_cond_meta,
        "render_view": int(args.render_view),
        "checkpoints": {
            "part_seg": str(args.part_seg_ckpt.resolve()),
            "ss_flow": str(args.ss_flow_ckpt.resolve()),
            "slat_flow": str(args.slat_flow_ckpt.resolve()),
            "slat_gaussian_decoder": str(args.slat_gaussian_decoder_ckpt.resolve()),
        },
        "datasets": dataset_meta,
    }
    _write_json(case_dir / "summary.json", summary)
    print(f"[gaussian] done -> {case_dir}", flush=True)
    print(f"[gaussian] overview -> {overview}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
