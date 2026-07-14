#!/usr/bin/env python3
"""3D voxel visualizations for Route-V predictions."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader


PROJECT_ROOT = Path(__file__).resolve().parents[2]
TRELLIS_PATH = PROJECT_ROOT / "TRELLIS-arts"
if str(TRELLIS_PATH) not in sys.path:
    sys.path.insert(0, str(TRELLIS_PATH))
DEV_PATH = PROJECT_ROOT / "scripts" / "dev"
if str(DEV_PATH) not in sys.path:
    sys.path.insert(0, str(DEV_PATH))

from scripts.train.part_promptable_seg.part_promptable_seg_utils import (  # noqa: E402
    PromptablePartDataset,
    collate_promptable_parts,
    decode_metrics_for_batch,
    enumerate_part_rows,
    latent_support_mask,
    load_ss_decoder,
    make_base_dataset,
    mask_morphology,
)
from scripts.train.part_promptable_seg.train_part_promptable_seg import decode_full_occ, voxel_decode_metrics_from_forward  # noqa: E402
from trellis.models.part_seg.promptable_latent_seg import (  # noqa: E402
    PromptablePartLatentSegNet,
    joint_local_depth_from_ckpt,
    joint_local_mode_from_ckpt,
    semantic_classes_from_ckpt,
    voxel_embedding_dim_from_ckpt,
)


DEFAULT_CKPT = Path("/mnt/robot-data-lab/jzh/art-gen-output/debug/part_promptable_seg_route_v_nocap_16samples/ckpts/latest.pt")
DEFAULT_SELECTION = Path("/mnt/robot-data-lab/jzh/art-gen-output/debug/part_promptable_seg_debug_selections/d2_route_v_16samples.json")
DEFAULT_OUT = Path("/mnt/robot-data-lab/jzh/art-gen-output/debug/part_promptable_seg_voxel_vis_16_step5000")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", type=Path, default=DEFAULT_CKPT)
    parser.add_argument("--selection-json", type=Path, default=DEFAULT_SELECTION)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--target-obj", default="100283")
    parser.add_argument("--target-angle", type=int, default=0)
    parser.add_argument("--target-part", default="door_0")
    parser.add_argument("--support-multiplier", type=float, default=4.0)
    return parser.parse_args()


def load_model(ckpt_path: Path, device: torch.device) -> tuple[PromptablePartLatentSegNet, torch.Tensor, dict[str, Any]]:
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    args = ckpt.get("args", {})
    state = ckpt["model"]
    stem_channels = int(state["stem.weight"].shape[1])
    mask_encoder = str(args.get("mask_encoder", "cnn_grid"))
    model = PromptablePartLatentSegNet(
        dim=int(args.get("dim", 256)),
        depth=int(args.get("depth", 6)),
        head_depth=int(args.get("head_depth", 2)),
        heads=int(args.get("heads", 8)),
        use_xyz=stem_channels == 11,
        use_voxel_head=True,
        voxel_depth=int(args.get("voxel_depth", 3)),
        refine_mode=str(args.get("refine_mode", "token")),
        spconv_depth=int(args.get("spconv_depth", 4)),
        mask_encoder=mask_encoder,
        point_k_boundary=int(args.get("point_k_boundary", 32)),
        point_k_interior=int(args.get("point_k_interior", 32)),
        point_resample_points=bool(args.get("point_resample_points", False)),
        semantic_classes=semantic_classes_from_ckpt(ckpt),
        voxel_embedding_dim=voxel_embedding_dim_from_ckpt(ckpt),
        use_body_prompt=bool(args.get("joint_seg", False)) or "body_prompt" in state,
        joint_local_mode=joint_local_mode_from_ckpt(ckpt),
        joint_local_depth=joint_local_depth_from_ckpt(ckpt),
    ).to(device)
    model.load_state_dict(state, strict=True)
    model.eval()
    return model, ckpt["empty_code"].float(), args


def coord_keys(coords: np.ndarray) -> set[int]:
    coords = np.asarray(coords, dtype=np.int64).reshape(-1, 3)
    if coords.size == 0:
        return set()
    keys = coords[:, 0] * 4096 + coords[:, 1] * 64 + coords[:, 2]
    return {int(v) for v in keys}


def keys_to_coords(keys: set[int]) -> np.ndarray:
    if not keys:
        return np.empty((0, 3), dtype=np.int64)
    arr = np.asarray(sorted(keys), dtype=np.int64)
    x = arr // 4096
    rem = arr % 4096
    y = rem // 64
    z = rem % 64
    return np.stack([x, y, z], axis=1)


def scatter3d(ax, coords: np.ndarray, *, color, label: str, size: float = 5.0, alpha: float = 0.7) -> None:
    if coords is None or len(coords) == 0:
        return
    ax.scatter(coords[:, 0], coords[:, 1], coords[:, 2], c=[color], s=size, alpha=alpha, marker="s", linewidths=0, label=label, depthshade=False)


def coords_to_bool_grid(coords: np.ndarray) -> np.ndarray:
    grid = np.zeros((64, 64, 64), dtype=bool)
    coords = np.asarray(coords, dtype=np.int64).reshape(-1, 3)
    if coords.size:
        coords = np.clip(coords, 0, 63)
        grid[coords[:, 0], coords[:, 1], coords[:, 2]] = True
    return grid


def draw_voxel_grid(ax, filled: np.ndarray, facecolors: np.ndarray, *, title: str) -> None:
    ax.voxels(filled, facecolors=facecolors, edgecolor=(0.08, 0.08, 0.08, 0.10), linewidth=0.08, shade=False)
    setup_3d(ax, title)


def make_rgba(color, alpha: float = 0.72) -> np.ndarray:
    rgba = np.asarray(color, dtype=np.float32)
    if rgba.shape[0] == 3:
        rgba = np.concatenate([rgba, np.asarray([1.0], dtype=np.float32)])
    rgba[3] = float(alpha)
    return rgba


def setup_3d(ax, title: str) -> None:
    ax.set_title(title, fontsize=9)
    ax.set_xlim(0, 64)
    ax.set_ylim(0, 64)
    ax.set_zlim(0, 64)
    ax.set_box_aspect((1, 1, 1))
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.view_init(elev=24, azim=-58)
    ax.tick_params(labelsize=6, pad=0)


def save_error_figure(gt: np.ndarray, pred: np.ndarray, out_path: Path, title: str) -> dict[str, int]:
    gt_set = coord_keys(gt)
    pred_set = coord_keys(pred)
    tp = keys_to_coords(gt_set & pred_set)
    fn = keys_to_coords(gt_set - pred_set)
    fp = keys_to_coords(pred_set - gt_set)
    fig = plt.figure(figsize=(16, 8), dpi=180)
    for i, (elev, azim, name) in enumerate([(24, -58, "iso"), (90, -90, "top"), (0, -90, "front"), (0, 0, "side")]):
        ax = fig.add_subplot(2, 2, i + 1, projection="3d")
        filled = np.zeros((64, 64, 64), dtype=bool)
        colors = np.zeros((64, 64, 64, 4), dtype=np.float32)
        for coords, color in ((tp, (0.17, 0.62, 0.17, 0.48)), (fn, (0.84, 0.15, 0.16, 0.95)), (fp, (0.12, 0.47, 0.71, 0.82))):
            grid = coords_to_bool_grid(coords)
            filled |= grid
            colors[grid] = color
        draw_voxel_grid(ax, filled, colors, title=f"{name}: green TP, red FN, blue FP")
        ax.view_init(elev=elev, azim=azim)
        handles = [
            plt.Line2D([0], [0], marker="s", color="w", label=f"TP {len(tp)}", markerfacecolor="#2ca02c", markersize=8),
            plt.Line2D([0], [0], marker="s", color="w", label=f"FN {len(fn)}", markerfacecolor="#d62728", markersize=8),
            plt.Line2D([0], [0], marker="s", color="w", label=f"FP {len(fp)}", markerfacecolor="#1f77b4", markersize=8),
        ]
        ax.legend(handles=handles, fontsize=7, loc="upper right")
    fig.suptitle(title, fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out_path)
    plt.close(fig)
    return {"tp": int(len(tp)), "fn": int(len(fn)), "fp": int(len(fp)), "gt": int(len(gt_set)), "pred": int(len(pred_set))}


def save_parts_figure(parts: list[dict[str, Any]], key: str, out_path: Path, title: str) -> None:
    cmap = plt.get_cmap("tab20")
    fig = plt.figure(figsize=(12, 10), dpi=180)
    ax = fig.add_subplot(1, 1, 1, projection="3d")
    filled = np.zeros((64, 64, 64), dtype=bool)
    colors = np.zeros((64, 64, 64, 4), dtype=np.float32)
    handles = []
    for idx, item in enumerate(parts):
        coords = item[key]
        color = cmap(idx % 20)
        grid = coords_to_bool_grid(coords)
        filled |= grid
        colors[grid] = make_rgba(color, 0.72)
        handles.append(plt.Line2D([0], [0], marker="s", color="w", label=f"{idx}:{item['part_name']} ({len(coords)})", markerfacecolor=color, markersize=7))
    draw_voxel_grid(ax, filled, colors, title=title)
    ax.legend(handles=handles, fontsize=6, loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0.0)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def save_mask_overlay(base_ds, sample: dict[str, Any], part: dict[str, Any], label: int, out_path: Path) -> list[dict[str, Any]]:
    mask_paths = list(base_ds._iter_mask_paths(sample))
    rows = []
    fig, axes = plt.subplots(2, 4, figsize=(16, 8), dpi=160)
    for i, (mask_path, image_path) in enumerate(zip(mask_paths, sample["image_paths"])):
        mask = np.load(mask_path)
        binary = mask == int(label)
        ys, xs = np.nonzero(binary)
        if len(xs):
            bbox = [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]
            centroid = [float(xs.mean()), float(ys.mean())]
        else:
            bbox = [None, None, None, None]
            centroid = [None, None]
        img = np.asarray(Image.open(image_path).convert("RGB").resize((mask.shape[1], mask.shape[0])))
        overlay = img.astype(np.float32)
        overlay[binary] = overlay[binary] * 0.35 + np.array([255, 40, 30], dtype=np.float32) * 0.65
        axes[0, i].imshow(overlay.astype(np.uint8))
        axes[0, i].axis("off")
        axes[0, i].set_title(f"view {sample['view_indices'][i]} pixels={int(binary.sum())}")
        axes[1, i].imshow(binary, cmap="gray")
        axes[1, i].axis("off")
        axes[1, i].set_title(f"bbox={bbox}")
        rows.append({"view": int(sample["view_indices"][i]), "pixels": int(binary.sum()), "bbox_xyxy": bbox, "centroid_xy": centroid})
    fig.suptitle(f"{sample['obj_id']} angle_{sample['angle_idx']} {part['part_name']} label={label}", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out_path)
    plt.close(fig)
    return rows


@torch.no_grad()
def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() or not args.device.startswith("cuda") else "cpu")
    base_ds = make_base_dataset()
    all_rows = enumerate_part_rows(base_ds)
    specs = json.loads(args.selection_json.read_text(encoding="utf-8"))
    rows = []
    for spec in specs:
        matches = [
            row
            for row in all_rows
            if row.obj_id == str(spec["obj_id"])
            and row.angle_idx == int(spec["angle_idx"])
            and row.part_name == str(spec["part_name"])
        ]
        if len(matches) != 1:
            raise RuntimeError(f"selection matched {len(matches)} rows: {spec}")
        rows.append(matches[0])
    ds = PromptablePartDataset(base_ds, rows, mask_size=512)
    model, empty_code, ckpt_args = load_model(args.ckpt, device)
    decoder = load_ss_decoder(device=device, fp32=True)

    parts_out = []
    target_detail = None
    for idx in range(len(ds)):
        batch = collate_promptable_parts([ds[idx]])
        z_global = batch["z_global"].to(device=device, dtype=torch.float32)
        masks2d = batch["masks2d"].to(device=device, dtype=torch.float32)
        latent_gt = batch["latent_gt"].to(device=device, dtype=torch.float32)
        raw_m = batch["m_gt"].to(device=device, dtype=torch.float32)
        support_m, _noise, _thr = latent_support_mask(latent_gt, empty_code.to(device), raw_m, multiplier=float(args.support_multiplier))
        full_occ = decode_full_occ(decoder, z_global, threshold=0.0)
        out_gt = model.forward_voxels(z_global, masks2d, mask_morphology(support_m, "dilate"), full_occ, max_voxels_per_sample=0)
        pred_m = (out_gt["m_logit"].sigmoid() > 0.5).float().view(support_m.shape)
        out_pred = model.forward_voxels(z_global, masks2d, mask_morphology(pred_m, "dilate"), full_occ, max_voxels_per_sample=0)
        metrics = voxel_decode_metrics_from_forward(out_pred["voxel_logits"], out_pred["voxel_coords"], batch["raw_coords"])[0]
        pred_coords = out_pred["voxel_coords"][0][out_pred["voxel_logits"][0, : out_pred["voxel_coords"][0].shape[0]].float().sigmoid() > 0.5].detach().cpu().numpy()
        gt_coords = np.asarray(batch["raw_coords"][0], dtype=np.int64)
        item = {
            "obj_id": batch["obj_id"][0],
            "angle_idx": int(batch["angle_idx"][0]),
            "part_name": batch["part_name"][0],
            "raw_count": int(batch["raw_count"][0].item()),
            "gt_coords": gt_coords,
            "pred_coords": pred_coords,
            "metrics": metrics,
        }
        parts_out.append(item)
        if item["obj_id"] == str(args.target_obj) and item["angle_idx"] == int(args.target_angle) and item["part_name"] == str(args.target_part):
            target_detail = item

    if target_detail is None:
        raise RuntimeError("target part not found in selection")

    err_counts = save_error_figure(
        target_detail["gt_coords"],
        target_detail["pred_coords"],
        args.out_dir / "door_0_pred_vs_gt_error_3d.png",
        f"{args.target_obj} angle_{args.target_angle} {args.target_part} pred vs GT IoU={target_detail['metrics']['decode_iou']:.4f}",
    )
    save_parts_figure(parts_out, "gt_coords", args.out_dir / "all_parts_gt_3d_colored.png", "16-sample GT voxels, one color per part")
    save_parts_figure(parts_out, "pred_coords", args.out_dir / "all_parts_pred_3d_colored.png", "16-sample predicted voxels, one color per part")
    target_obj_gt_parts = [
        {
            "part_name": row.part_name,
            "gt_coords": base_ds._load_raw_ind_coords(base_ds.samples[row.sample_idx], base_ds.samples[row.sample_idx]["parts"][row.part_idx]),
        }
        for row in rows
        if row.obj_id == str(args.target_obj) and row.angle_idx == int(args.target_angle)
    ]
    target_obj_pred_parts = [
        {"part_name": p["part_name"], "pred_coords": p["pred_coords"]}
        for p in parts_out
        if p["obj_id"] == str(args.target_obj) and p["angle_idx"] == int(args.target_angle)
    ]
    save_parts_figure(
        target_obj_gt_parts,
        "gt_coords",
        args.out_dir / f"{args.target_obj}_all_parts_gt_3d_colored.png",
        f"{args.target_obj} angle_{args.target_angle} GT voxels, one color per part",
    )
    save_parts_figure(
        target_obj_pred_parts,
        "pred_coords",
        args.out_dir / f"{args.target_obj}_trained_parts_pred_3d_colored.png",
        f"{args.target_obj} angle_{args.target_angle} predicted voxels for trained parts",
    )

    target_row = [r for r in rows if r.obj_id == str(args.target_obj) and r.angle_idx == int(args.target_angle) and r.part_name == str(args.target_part)][0]
    sample = base_ds.samples[target_row.sample_idx]
    part = sample["parts"][target_row.part_idx]
    label = int(base_ds._part_original_label(sample, part))
    mask_rows = save_mask_overlay(base_ds, sample, part, label, args.out_dir / "door_0_mask_overlays.png")

    serializable_parts = [
        {
            "obj_id": p["obj_id"],
            "angle_idx": p["angle_idx"],
            "part_name": p["part_name"],
            "raw_count": p["raw_count"],
            "pred_count": int(len(p["pred_coords"])),
            "iou": float(p["metrics"]["decode_iou"]),
            "precision": float(p["metrics"]["decode_precision"]),
            "recall": float(p["metrics"]["decode_recall"]),
        }
        for p in parts_out
    ]
    meta = {
        "ckpt": str(args.ckpt),
        "ckpt_args": {k: str(v) for k, v in ckpt_args.items()},
        "target": {
            "obj_id": args.target_obj,
            "angle_idx": int(args.target_angle),
            "part_name": args.target_part,
            "raw_count": int(target_detail["raw_count"]),
            "metrics": target_detail["metrics"],
            "error_counts": err_counts,
            "label": label,
            "view_indices": [int(v) for v in sample["view_indices"]],
            "mask_views": mask_rows,
        },
        "parts": serializable_parts,
        "outputs": {
            "door_error_3d": str(args.out_dir / "door_0_pred_vs_gt_error_3d.png"),
            "gt_all_parts_3d": str(args.out_dir / "all_parts_gt_3d_colored.png"),
            "pred_all_parts_3d": str(args.out_dir / "all_parts_pred_3d_colored.png"),
            "target_obj_gt_all_parts_3d": str(args.out_dir / f"{args.target_obj}_all_parts_gt_3d_colored.png"),
            "target_obj_pred_trained_parts_3d": str(args.out_dir / f"{args.target_obj}_trained_parts_pred_3d_colored.png"),
            "door_mask_overlays": str(args.out_dir / "door_0_mask_overlays.png"),
        },
    }
    (args.out_dir / "metadata.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(meta["target"], indent=2, ensure_ascii=False), flush=True)
    print(json.dumps(meta["outputs"], indent=2, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
