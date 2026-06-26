#!/usr/bin/env python3
"""Export Part SS Latent Flow test examples for downstream decoding.

This standalone entry runs the same checkpoint sampling path as eval/decode,
then writes one directory per dataset example containing:

  - RGB images used by the condition
  - DINO condition tensors used by the network
  - GT part SS latents
  - predicted part SS latents
  - decoded predicted voxel coordinates
  - optional per-part SLat, mesh, and Gaussian assets
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import train_arts  # noqa: F401  # Registers lightweight trellis package stubs.
import torch
from torch.utils.data import DataLoader, Subset

from eval_part_ss_latent_flow import _apply_ckpt_latent_norm, _checkpoint_path, _cfg_dict, _dataset_cls, _load_model
from inference import decode_slat_assets, run_slat_flow_from_tokens
from trellis.trainers.arts.part_ss_latent_flow import (
    _object_id_filter_indices,
    _sample_indices_for_eval,
    _setup_rng,
    _to_device,
)
from trellis.trainers.arts.part_ss_latent_flow_eval import (
    decode_ss_latent_to_coords,
    load_ss_decoder,
)
from trellis.trainers.arts.part_ss_latent_flow_losses import (
    build_part_ss_sampler_kwargs,
    sample_part_ss_latent,
)
from trellis.utils.arts.config_utils import load_config


def _safe_name(value: str, max_len: int = 120) -> str:
    value = re.sub(r"[^A-Za-z0-9_.=-]+", "_", str(value)).strip("_")
    return (value or "sample")[:max_len]


def _rooted(dataset: Any, rel_or_abs: str) -> Path:
    path = Path(rel_or_abs)
    return path if path.is_absolute() else Path(dataset.data_root) / path


def _resolved_view_paths(
    dataset: Any,
    sample_meta: dict[str, Any],
    *,
    resolver_name: str,
    manifest_field: str,
) -> list[Path]:
    resolver = getattr(dataset, resolver_name, None)
    if callable(resolver):
        return [Path(path) for path in resolver(sample_meta)]
    return [_rooted(dataset, str(path)) for path in sample_meta.get(manifest_field, [])]


def _copy_view_files(
    paths: list[Path],
    sample_meta: dict[str, Any],
    out_dir: Path,
    folder_name: str,
) -> list[str]:
    dst_dir = out_dir / folder_name
    dst_dir.mkdir(parents=True, exist_ok=True)
    view_indices = list(sample_meta.get("view_indices", []))
    copied: list[str] = []
    for view_pos, src in enumerate(paths):
        if not src.is_file():
            copied.append(f"missing:{src}")
            continue
        view_idx = view_indices[view_pos] if view_pos < len(view_indices) else view_pos
        view_name = _safe_name(str(view_idx), max_len=32)
        dst = dst_dir / f"viewpos_{view_pos:02d}_view_{view_name}_{src.name}"
        shutil.copy2(src, dst)
        copied.append(str(dst.relative_to(out_dir)))
    return copied


def _rgb_view_paths(dataset: Any, sample_meta: dict[str, Any]) -> list[Path]:
    return _resolved_view_paths(
        dataset,
        sample_meta,
        resolver_name="_iter_rgb_paths",
        manifest_field="image_paths",
    )


def _mask_view_paths(dataset: Any, sample_meta: dict[str, Any]) -> list[Path]:
    return _resolved_view_paths(
        dataset,
        sample_meta,
        resolver_name="_iter_mask_paths",
        manifest_field="mask_paths",
    )


def _copy_rgb_views(dataset: Any, sample_meta: dict[str, Any], out_dir: Path) -> list[str]:
    return _copy_view_files(_rgb_view_paths(dataset, sample_meta), sample_meta, out_dir, "rgb")


def _copy_mask_views(dataset: Any, sample_meta: dict[str, Any], out_dir: Path) -> list[str]:
    return _copy_view_files(_mask_view_paths(dataset, sample_meta), sample_meta, out_dir, "mask")


def _stringify_paths(paths: list[Path]) -> list[str]:
    return [str(path) for path in paths]


def _save_latent(path: Path, tensor: torch.Tensor) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, tensor.detach().float().cpu().numpy().astype(np.float32, copy=False))


def _save_coords(path: Path, coords: torch.Tensor) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, coords.detach().cpu().long().numpy().astype(np.int64, copy=False))


def _slat_part_seed(base_seed: int, dataset_index: int, part_index: int) -> int:
    return (int(base_seed) + int(dataset_index) * 1_000_003 + int(part_index) * 9_176) % (2**63 - 1)


def _save_slat_tensor_payload(path: Path, slat: Any, *, is_normalized: bool = True) -> None:
    """Persist SLat as plain tensors so downstream loads do not pickle SparseTensor internals."""
    coords = getattr(slat, "coords", None)
    feats = getattr(slat, "feats", None)
    if coords is None or feats is None:
        raise TypeError(f"SLat object must expose coords and feats tensors, got {type(slat).__name__}")

    payload: dict[str, Any] = {
        "format": "trellis_sparse_tensor_v1",
        "coords": coords.detach().cpu(),
        "feats": feats.detach().cpu(),
        "is_normalized": bool(is_normalized),
        "metadata": {
            "reconstruct": "trellis.modules.sparse.basic.SparseTensor(feats=feats, coords=coords)",
        },
    }
    shape = getattr(slat, "shape", None)
    if shape is not None:
        payload["shape"] = tuple(int(dim) for dim in shape)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def _save_decoded_slat_assets(decoded: dict[str, Any], asset_dir: Path, example_dir: Path) -> dict[str, str]:
    from trellis.utils.arts.slat_asset_writer import save_decoded_slat_assets

    rec = save_decoded_slat_assets(decoded, asset_dir, mesh_name="mesh.obj", gaussian_name="gaussians.ply")
    out: dict[str, str] = {}
    if "gaussian" in rec:
        out["pred_gaussian"] = str((asset_dir / rec["gaussian"]).relative_to(example_dir))
    if "mesh" in rec:
        out["pred_mesh"] = str((asset_dir / rec["mesh"]).relative_to(example_dir))
    return out


def _write_part_slat_assets(
    *,
    pred_coords: torch.Tensor,
    cond_tokens: torch.Tensor,
    example_dir: Path,
    stem: str,
    slat_cfg: dict[str, Any],
    slat_seed: int | None = None,
) -> dict[str, Any]:
    if pred_coords.shape[0] == 0:
        if slat_cfg["empty_policy"] == "error":
            raise ValueError(f"cannot export SLat assets for empty predicted voxel part: {stem}")
        return {
            "slat_status": "skipped_empty_voxel",
            "slat_skip_reason": "pred_voxel_count=0",
        }

    slat = run_slat_flow_from_tokens(
        cond_tokens,
        pred_coords,
        slat_cfg["flow_ckpt"],
        num_steps=int(slat_cfg["num_steps"]),
        seed=slat_seed,
    )
    slat_dir = example_dir / "pred_slat"
    slat_dir.mkdir(parents=True, exist_ok=True)
    slat_path = slat_dir / f"{stem}.pt"
    _save_slat_tensor_payload(slat_path, slat, is_normalized=True)

    decoded = decode_slat_assets(
        slat,
        gaussian_decoder_ckpt=slat_cfg["gs_decoder_ckpt"],
        mesh_decoder_ckpt=slat_cfg["mesh_decoder_ckpt"],
    )
    asset_dir = example_dir / "pred_assets" / stem
    record = _save_decoded_slat_assets(decoded, asset_dir, example_dir)
    record.update({
        "slat_status": "generated",
        "pred_slat": str(slat_path.relative_to(example_dir)),
        "pred_asset_dir": str(asset_dir.relative_to(example_dir)),
    })
    if slat_seed is not None:
        record["slat_seed"] = int(slat_seed)
    return record


def _write_example(
    *,
    dataset: Any,
    sample_meta: dict[str, Any],
    batch: dict[str, Any],
    pred: torch.Tensor,
    decoder: Any,
    threshold: float,
    example_dir: Path,
    dataset_index: int,
    slat_cfg: dict[str, Any] | None = None,
) -> dict[str, Any]:
    example_dir.mkdir(parents=True, exist_ok=True)
    valid_k = int(batch["part_valid"][0].sum().item())
    target_names = list(batch["target_part_names"][0])
    target_slots = batch["target_slots"][0].detach().cpu().long().tolist()

    rgb_paths = _rgb_view_paths(dataset, sample_meta)
    mask_paths = _mask_view_paths(dataset, sample_meta)
    rgb_files = _copy_view_files(rgb_paths, sample_meta, example_dir, "rgb")
    mask_files = _copy_view_files(mask_paths, sample_meta, example_dir, "mask")
    np.savez_compressed(
        example_dir / "dino_condition.npz",
        cond=batch["cond"][0].detach().float().cpu().numpy().astype(np.float32, copy=False),
        mask_token_labels=batch["mask_token_labels"][0].detach().cpu().long().numpy(),
        view_indices=np.asarray(batch["view_indices"][0], dtype=np.int64),
        target_slots=np.asarray(target_slots[:valid_k], dtype=np.int64),
        target_part_names=np.asarray(target_names[:valid_k], dtype=str),
    )
    if "part_token_weights" in batch:
        np.savez_compressed(
            example_dir / "part_token_weights.npz",
            weights=batch["part_token_weights"][0, :valid_k].detach().float().cpu().numpy().astype(np.float32, copy=False),
            target_part_names=np.asarray(target_names[:valid_k], dtype=str),
        )

    gt_dir = example_dir / "gt_ss_latents"
    pred_dir = example_dir / "pred_ss_latents"
    voxel_dir = example_dir / "pred_voxels"
    cond_tokens = batch["cond"][0].detach().float().cpu() if slat_cfg is not None else None
    part_records = []
    for part_idx in range(valid_k):
        part_name = _safe_name(target_names[part_idx])
        stem = f"{part_idx:02d}_{part_name}"
        gt_path = gt_dir / f"{stem}.npy"
        pred_path = pred_dir / f"{stem}.npy"
        voxel_path = voxel_dir / f"{stem}.npy"
        _save_latent(gt_path, batch["x_1_parts"][0, part_idx])
        _save_latent(pred_path, pred[0, part_idx])
        pred_coords = decode_ss_latent_to_coords(
            decoder,
            pred[0, part_idx].detach().float().cpu(),
            threshold=threshold,
        )
        _save_coords(voxel_path, pred_coords)
        part_records.append(
            {
                "part_index": part_idx,
                "part_name": target_names[part_idx],
                "target_slot": int(target_slots[part_idx]),
                "gt_ss_latent": str(gt_path.relative_to(example_dir)),
                "pred_ss_latent": str(pred_path.relative_to(example_dir)),
                "pred_voxel": str(voxel_path.relative_to(example_dir)),
                "pred_voxel_count": int(pred_coords.shape[0]),
            }
        )
        if slat_cfg is not None:
            slat_seed = None
            if "seed" in slat_cfg:
                slat_seed = _slat_part_seed(int(slat_cfg["seed"]), dataset_index, part_idx)
            part_records[-1].update(
                _write_part_slat_assets(
                    pred_coords=pred_coords,
                    cond_tokens=cond_tokens,
                    example_dir=example_dir,
                    stem=stem,
                    slat_cfg=slat_cfg,
                    slat_seed=slat_seed,
                )
            )

    tokens_rel = sample_meta.get("tokens_rel")
    meta = {
        "dataset_index": int(dataset_index),
        "sample_id": batch["sample_id"][0],
        "obj_id": batch["obj_id"][0],
        "angle_idx": int(batch["angle_idx"][0]),
        "view_indices": list(batch["view_indices"][0]),
        "image_paths": list(sample_meta.get("image_paths", [])),
        "mask_paths": list(sample_meta.get("mask_paths", [])),
        "source_rgb_paths": _stringify_paths(rgb_paths),
        "source_mask_paths": _stringify_paths(mask_paths),
        "rgb_files": rgb_files,
        "mask_files": mask_files,
        "source_dino_tokens": str(_rooted(dataset, str(tokens_rel))) if tokens_rel else None,
        "dino_condition": "dino_condition.npz",
        "part_token_weights": "part_token_weights.npz" if "part_token_weights" in batch else None,
        "threshold": float(threshold),
        "parts": part_records,
    }
    (example_dir / "meta.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return meta


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export Part SS Latent Flow checkpoint predictions as per-example files."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--ckpt", default=None)
    parser.add_argument("--load-dir", default=None)
    parser.add_argument("--step", type=int, default=None)
    parser.add_argument("--export-root", required=True)
    parser.add_argument("--max-samples", type=int, default=4)
    parser.add_argument("--sample-mode", choices=("first", "spread"), default="first")
    parser.add_argument(
        "--object-ids",
        default=None,
        help="Optional comma-separated object IDs. Sampling is applied after filtering to these objects.",
    )
    parser.add_argument("--num-steps", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--export-slat-assets", action="store_true")
    parser.add_argument("--slat-flow-ckpt", default=None)
    parser.add_argument("--slat-gs-decoder-ckpt", default=None)
    parser.add_argument("--slat-mesh-decoder-ckpt", default=None)
    parser.add_argument("--slat-num-steps", type=int, default=None)
    parser.add_argument("--slat-seed", type=int, default=None)
    parser.add_argument("--slat-empty-policy", choices=("skip", "error"), default="skip")
    parser.add_argument("overrides", nargs="*")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cfg = load_config(args.config, overrides=args.overrides)
    stage = str(cfg.get("stage"))
    data_cfg = _cfg_dict(cfg.data)
    flow_cfg = _cfg_dict(cfg.flow)
    eval_cfg = _cfg_dict(cfg.eval)
    if args.num_steps is not None:
        flow_cfg["num_steps"] = int(args.num_steps)

    seed = int(getattr(cfg.training, "seed", 42)) if "training" in cfg else 42
    _setup_rng(seed)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    dataset = _dataset_cls(stage)(data_cfg)
    candidate_indices = _object_id_filter_indices(dataset, args.object_ids)
    sample_indices = _sample_indices_for_eval(
        len(dataset),
        int(args.max_samples),
        args.sample_mode,
        candidate_indices,
    )
    eval_dataset = Subset(dataset, sample_indices)
    loader = DataLoader(eval_dataset, batch_size=1, shuffle=False, num_workers=0, collate_fn=dataset.collate_fn)
    ckpt_path = _checkpoint_path(args)
    model, ckpt_step, ckpt_cfg = _load_model(cfg, ckpt_path, device)
    _apply_ckpt_latent_norm(flow_cfg, ckpt_cfg)
    decoder = load_ss_decoder(eval_cfg["ss_decoder_ckpt"])
    threshold = float(eval_cfg.get("decode_threshold", 0.0))
    slat_cfg = None
    if args.export_slat_assets:
        missing = [
            name
            for name, value in (
                ("--slat-flow-ckpt", args.slat_flow_ckpt),
                ("--slat-gs-decoder-ckpt", args.slat_gs_decoder_ckpt),
                ("--slat-mesh-decoder-ckpt", args.slat_mesh_decoder_ckpt),
            )
            if not value
        ]
        if missing:
            raise ValueError("--export-slat-assets requires " + ", ".join(missing))
        slat_cfg = {
            "flow_ckpt": args.slat_flow_ckpt,
            "gs_decoder_ckpt": args.slat_gs_decoder_ckpt,
            "mesh_decoder_ckpt": args.slat_mesh_decoder_ckpt,
            "num_steps": int(args.slat_num_steps if args.slat_num_steps is not None else flow_cfg.get("num_steps", 20)),
            "seed": int(args.slat_seed if args.slat_seed is not None else seed),
            "empty_policy": args.slat_empty_policy,
        }
    export_root = Path(args.export_root) / f"step_{ckpt_step}"
    export_root.mkdir(parents=True, exist_ok=True)

    print("============================================================", flush=True)
    print("Part SS Latent Flow Example Export", flush=True)
    print(f"  stage:       {stage}", flush=True)
    print(f"  checkpoint:  {ckpt_path}", flush=True)
    print(f"  step:        {ckpt_step}", flush=True)
    print(f"  samples:     {len(sample_indices)} mode={args.sample_mode}", flush=True)
    print(f"  object_ids:  {args.object_ids or '<none>'}", flush=True)
    print(f"  slat_assets: {'on' if slat_cfg is not None else 'off'}", flush=True)
    if slat_cfg is not None:
        print(f"  slat_steps:  {slat_cfg['num_steps']}", flush=True)
        print(f"  slat_flow:   {slat_cfg['flow_ckpt']}", flush=True)
        print(f"  slat_gs:     {slat_cfg['gs_decoder_ckpt']}", flush=True)
        print(f"  slat_mesh:   {slat_cfg['mesh_decoder_ckpt']}", flush=True)
        print(f"  slat_seed:   {slat_cfg['seed']}", flush=True)
        print(f"  empty_policy:{slat_cfg['empty_policy']}", flush=True)
    print(f"  export_root: {export_root}", flush=True)
    print("============================================================", flush=True)

    rows = []
    for idx, batch in enumerate(loader):
        source_idx = sample_indices[idx]
        sample_meta = dataset.samples[source_idx]
        batch = _to_device(batch, device)
        pred = sample_part_ss_latent(
            model,
            z_global=batch["z_global"],
            cond=batch["cond"],
            mask_token_labels=batch["mask_token_labels"],
            part_valid=batch["part_valid"],
            target_slots=batch["target_slots"],
            part_token_weights=batch.get("part_token_weights"),
            num_steps=int(flow_cfg.get("num_steps", 20)),
            noise_scale=float(flow_cfg.get("noise_scale", 1.0)),
            latent_scale=float(flow_cfg.get("latent_scale", 1.0)),
            **build_part_ss_sampler_kwargs(model, flow_cfg),
        )
        sample_id = _safe_name(str(batch["sample_id"][0]))
        example_dir = export_root / f"example_{idx:06d}_{sample_id}"
        print(
            f"[{idx + 1}/{len(sample_indices)}] dataset_idx={source_idx} "
            f"sample={batch['sample_id'][0]} -> {example_dir}",
            flush=True,
        )
        rows.append(
            _write_example(
                dataset=dataset,
                sample_meta=sample_meta,
                batch=batch,
                pred=pred,
                decoder=decoder,
                threshold=threshold,
                example_dir=example_dir,
                dataset_index=source_idx,
                slat_cfg=slat_cfg,
            )
        )

    (export_root / "index.json").write_text(
        json.dumps({"stage": stage, "checkpoint_step": ckpt_step, "examples": rows}, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"[done] {export_root / 'index.json'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
