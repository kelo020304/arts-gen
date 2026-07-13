#!/usr/bin/env python3
"""Render fixed Track1 PartMasked decoder snapshots from saved checkpoints."""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
TRELLIS_ROOT = REPO_ROOT / "TRELLIS-arts"
if str(TRELLIS_ROOT) not in sys.path:
    sys.path.insert(0, str(TRELLIS_ROOT))
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from track1_online_render import (  # noqa: E402
    DEFAULT_CACHE_MANIFEST,
    DEFAULT_DECODER_CKPT,
    OnlineRenderLossProbe,
    PartMaskedOnlineRenderDataset,
    build_partmasked_decoder_from_pretrained,
)
from trellis.modules.sparse import SparseTensor  # noqa: E402
from trellis.representations import MeshExtractResult  # noqa: E402


DEFAULT_CKPT_DIR = Path("/robot/data-lab/jzh/art-gen/ckpts/slat-dec-part/overfit-0703")
DEFAULT_SAMPLE_INDICES = [2, 3, 14, 16, 20, 6]


def seed_all(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed) % (2**32))
    torch.manual_seed(int(seed))
    torch.cuda.manual_seed_all(int(seed))


def subset_collate(dataset: PartMaskedOnlineRenderDataset, indices: list[int]) -> dict[str, Any]:
    return dataset.collate_fn([dataset[int(idx)] for idx in indices])


def _snapshot_mask_modulation(snapshot: Path) -> str:
    payload = torch.load(snapshot, map_location="cpu", weights_only=False)
    args = payload.get("args") or {}
    return str(args.get("mask_modulation", "none"))


def _snapshot_arg(snapshot: Path, key: str, default: Any) -> Any:
    payload = torch.load(snapshot, map_location="cpu", weights_only=False)
    args = payload.get("args") or {}
    return args.get(key, default)


def load_decoder_from_snapshot(base_ckpt: Path, snapshot: Path, *, device: torch.device) -> torch.nn.Module:
    mask_modulation = _snapshot_mask_modulation(snapshot)
    decoder = build_partmasked_decoder_from_pretrained(
        base_ckpt,
        device=device,
        train=False,
        mask_modulation=mask_modulation,
    )
    payload = torch.load(snapshot, map_location=device, weights_only=False)
    state = payload.get("model")
    if not isinstance(state, dict):
        raise ValueError(f"{snapshot}: missing model state dict")
    missing, unexpected = decoder.load_state_dict(state, strict=True)
    if missing or unexpected:
        raise RuntimeError(f"{snapshot}: decoder load mismatch missing={missing} unexpected={unexpected}")
    decoder.eval()
    return decoder


def tensor_image_chw(x: torch.Tensor) -> np.ndarray:
    arr = x.detach().float().cpu()
    if arr.ndim == 2:
        arr = arr.unsqueeze(0).repeat(3, 1, 1)
    if arr.ndim == 3 and arr.shape[0] == 1:
        arr = arr.repeat(3, 1, 1)
    if arr.ndim != 3:
        raise ValueError(f"cannot convert tensor image shape {tuple(arr.shape)}")
    if arr.shape[0] == 3:
        arr = arr.permute(1, 2, 0)
    return arr.numpy().clip(0, 1)


def normal_to_rgb(x: torch.Tensor) -> np.ndarray:
    arr = x.detach().float().cpu()
    if arr.ndim == 3 and arr.shape[0] == 3:
        arr = arr.permute(1, 2, 0)
    return ((arr.numpy() + 1.0) * 0.5).clip(0, 1)


def surface_voxel_count(rep: MeshExtractResult, *, resolution: int = 64) -> int:
    if not bool(getattr(rep, "success", False)):
        return 0
    vertices = getattr(rep, "vertices", None)
    faces = getattr(rep, "faces", None)
    if vertices is None or faces is None or vertices.numel() == 0 or faces.numel() == 0:
        return 0
    verts = vertices.detach().float()
    face_idx = faces.detach().long()
    centroids = verts[face_idx].mean(dim=1)
    pts = torch.cat([verts, centroids], dim=0)
    q = torch.floor((pts + 0.5) * float(resolution)).long().clamp(0, int(resolution) - 1)
    keys = q[:, 0] * int(resolution) * int(resolution) + q[:, 1] * int(resolution) + q[:, 2]
    return int(torch.unique(keys).numel())


def render_checkpoint(
    *,
    snapshot: Path,
    prepared_items: list[tuple[dict[str, Any], dict[str, Any]]],
    probe: OnlineRenderLossProbe,
    base_decoder_ckpt: Path,
    device: torch.device,
    out_png: Path,
) -> list[dict[str, Any]]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    decoder = load_decoder_from_snapshot(base_decoder_ckpt, snapshot, device=device)
    probe.training_models["decoder"] = decoder
    probe.models = probe.training_models
    step_value = int(torch.load(snapshot, map_location="cpu", weights_only=False).get("step", -1))
    rendered: list[dict[str, Any]] = []
    for batch, prepared in prepared_items:
        latents: SparseTensor = prepared["latents"]
        probe.renderer.rendering_options.resolution = int(prepared["image"].shape[-1])
        with torch.no_grad():
            reps = decoder(latents)
            gt_meshes = [
                MeshExtractResult(item["vertices"].to(device=device), item["faces"].to(device=device))
                for item in prepared["mesh"]
            ]
            target = probe._render_batch(gt_meshes, prepared["extrinsics"], prepared["intrinsics"], return_types=["mask", "normal"])
            pred = probe._render_batch(reps, prepared["extrinsics"], prepared["intrinsics"], return_types=["mask", "normal"])
            target["normal"] = probe._flip_normal(target["normal"], prepared["extrinsics"], prepared["intrinsics"])
            pred["normal"] = probe._flip_normal(pred["normal"], prepared["extrinsics"], prepared["intrinsics"])
        rendered.append({"batch": batch, "prepared": prepared, "reps": reps, "target": target, "pred": pred})

    rows: list[dict[str, Any]] = []
    n = len(rendered)
    fig, axes = plt.subplots(n, 3, figsize=(9, 3 * n), squeeze=False)
    for i, item in enumerate(rendered):
        meta = item["batch"]["sample_meta"][0]
        reps = item["reps"]
        target = item["target"]
        pred = item["pred"]
        gt_mask = target["mask"][0, 0] if target["mask"].ndim == 4 else target["mask"][0]
        pred_mask = pred["mask"][0, 0] if pred["mask"].ndim == 4 else pred["mask"][0]
        gt_area = float(gt_mask.detach().float().sum().cpu().item())
        pred_area = float(pred_mask.detach().float().sum().cpu().item())
        mask_l1 = float(torch.mean(torch.abs(pred_mask.detach().float() - gt_mask.detach().float())).cpu().item())
        success = bool(getattr(reps[0], "success", False))
        faces = int(getattr(getattr(reps[0], "faces", None), "shape", [0])[0]) if success else 0
        surf_vox = surface_voxel_count(reps[0], resolution=64)
        rows.append(
            {
                "step": step_value,
                "tag": meta.get("tag"),
                "obj_id": meta.get("obj_id"),
                "component_name": meta.get("component_name"),
                "mask_mode": meta.get("mask_mode"),
                "mask_voxels_on_slat": int(meta.get("mask_voxels_on_slat", 0)),
                "success": success,
                "faces": faces,
                "pred_surface_voxels64": surf_vox,
                "gt_mask_area": gt_area,
                "pred_mask_area": pred_area,
                "pred_over_gt_area": pred_area / max(gt_area, 1.0),
                "mask_l1": mask_l1,
            }
        )
        axes[i, 0].imshow(tensor_image_chw(gt_mask))
        axes[i, 0].set_title(f"GT {meta.get('component_name')}\\n{meta.get('mask_mode')}")
        axes[i, 1].imshow(tensor_image_chw(pred_mask))
        axes[i, 1].set_title(f"Pred mask area {pred_area / max(gt_area, 1.0):.2f}x")
        axes[i, 2].imshow(normal_to_rgb(pred["normal"][0]))
        axes[i, 2].set_title(f"Pred normal success={success}")
        for j in range(3):
            axes[i, j].axis("off")
    fig.suptitle(f"Track1 snapshot {snapshot.stem}")
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=140)
    plt.close(fig)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=DEFAULT_CACHE_MANIFEST)
    parser.add_argument("--base-decoder-ckpt", type=Path, default=DEFAULT_DECODER_CKPT)
    parser.add_argument("--ckpt-dir", type=Path, default=DEFAULT_CKPT_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_CKPT_DIR / "track1_snapshots")
    parser.add_argument("--steps", type=int, nargs="+", default=[500, 1000, 1500, 2000, 2500, 3000])
    parser.add_argument("--sample-indices", type=int, nargs="+", default=DEFAULT_SAMPLE_INDICES)
    parser.add_argument("--mask-profile", choices=["gt", "front_only"], default="gt")
    parser.add_argument("--latent-input-mode", choices=["auto", "whole", "expanded_subset"], default="auto")
    parser.add_argument("--subset-dilation", type=int, default=-1)
    parser.add_argument("--resolution", type=int, default=256)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260703)
    args = parser.parse_args()

    device = torch.device(f"cuda:{int(args.gpu)}")
    torch.cuda.set_device(device)
    seed_all(int(args.seed))
    degrade_prob = 1.0 if args.mask_profile == "front_only" else 0.0
    first_snapshot = args.ckpt_dir / f"step_{int(args.steps[0]):07d}.pt"
    latent_input_mode = str(args.latent_input_mode)
    if latent_input_mode == "auto":
        latent_input_mode = str(_snapshot_arg(first_snapshot, "latent_input_mode", "whole")) if first_snapshot.is_file() else "whole"
    subset_dilation = int(args.subset_dilation)
    if subset_dilation < 0:
        subset_dilation = int(_snapshot_arg(first_snapshot, "subset_dilation", 1)) if first_snapshot.is_file() else 1
    dataset = PartMaskedOnlineRenderDataset(
        args.manifest,
        resolution=int(args.resolution),
        include_body=True,
        normalize_gt_mesh=True,
        mask_degrade_prob=degrade_prob,
        front_only_prob=1.0,
        latent_input_mode=latent_input_mode,
        subset_dilation=subset_dilation,
    )
    seed_all(int(args.seed))
    batches = []
    for sample_index in [int(i) for i in args.sample_indices]:
        seed_all(int(args.seed) + sample_index)
        batches.append(subset_collate(dataset, [sample_index]))
    base_mask_modulation = _snapshot_mask_modulation(first_snapshot) if first_snapshot.is_file() else "none"
    base_decoder = build_partmasked_decoder_from_pretrained(
        args.base_decoder_ckpt,
        device=device,
        train=False,
        mask_modulation=base_mask_modulation,
    )
    probe = OnlineRenderLossProbe(
        base_decoder,
        device=device,
        render_resolution=int(args.resolution),
        lambda_tsdf=0.01,
        lambda_ssim=0.2,
        lambda_lpips=0.0,
    )
    prepared_items = []
    for sample_index, batch in zip([int(i) for i in args.sample_indices], batches):
        seed_all(int(args.seed) + sample_index)
        prepared_items.append((batch, probe.prepare_batch(batch, device=device)))

    all_rows: list[dict[str, Any]] = []
    for step in args.steps:
        snapshot = args.ckpt_dir / f"step_{int(step):07d}.pt"
        if not snapshot.is_file():
            raise FileNotFoundError(f"snapshot missing: {snapshot}")
        out_png = args.out_dir / args.mask_profile / f"snapshot_step_{int(step):07d}.png"
        rows = render_checkpoint(
            snapshot=snapshot,
            prepared_items=prepared_items,
            probe=probe,
            base_decoder_ckpt=args.base_decoder_ckpt,
            device=device,
            out_png=out_png,
        )
        for row in rows:
            row["png"] = str(out_png)
            row["mask_profile"] = args.mask_profile
        all_rows.extend(rows)
        print(json.dumps({"step": int(step), "png": str(out_png), "rows": rows}, sort_keys=True), flush=True)

    csv_path = args.out_dir / args.mask_profile / "snapshot_metrics.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(all_rows[0].keys()))
        writer.writeheader()
        writer.writerows(all_rows)
    (args.out_dir / args.mask_profile / "snapshot_samples.json").write_text(
        json.dumps([batch["sample_meta"][0] for batch in batches], ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"csv": str(csv_path), "out_dir": str(args.out_dir / args.mask_profile)}, sort_keys=True), flush=True)


if __name__ == "__main__":
    os.environ.setdefault("SPCONV_ALGO", "native")
    os.environ.setdefault("ATTN_BACKEND", "sdpa")
    os.environ.setdefault("SPARSE_ATTN_BACKEND", "sdpa")
    main()
