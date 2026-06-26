"""Evaluation and inspection helpers for part SS latent flow."""

from __future__ import annotations

import functools
import os
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
import torch.nn.functional as F

from trellis.trainers.arts.part_ss_latent_flow_losses import k_bucket_name


__all__ = [
    "coords_iou",
    "compute_decode_metrics",
    "decode_ss_latent_to_coords",
    "decode_ss_latent_to_coords_with_stats",
    "load_ss_decoder",
    "part_assignment_iou_matrix",
    "summarize_assignment_matrix",
    "summarize_bucketed_part_metrics",
    "write_part_ss_inspection",
    "write_part_ss_inspection_sample",
]


_PART_COLORS = (
    "#d62728",
    "#1f77b4",
    "#2ca02c",
    "#ff7f0e",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#17becf",
)


def _visual_voxel_block_size() -> int:
    value = int(os.environ.get("PART_SS_VIS_VOXEL_BLOCK", "2"))
    if value <= 0 or 64 % value != 0:
        raise ValueError(f"PART_SS_VIS_VOXEL_BLOCK must be a positive divisor of 64, got {value}")
    return value


def _coords_set(coords: torch.Tensor | np.ndarray) -> set[tuple[int, int, int]]:
    if isinstance(coords, torch.Tensor):
        arr = coords.detach().cpu().long().numpy()
    else:
        arr = np.asarray(coords, dtype=np.int64)
    if arr.size == 0:
        return set()
    if arr.ndim != 2 or arr.shape[1] != 3:
        raise ValueError(f"coords expected [N,3], got {arr.shape}")
    return set(map(tuple, arr.tolist()))


def coords_iou(pred: torch.Tensor | np.ndarray, gt: torch.Tensor | np.ndarray) -> Dict[str, float]:
    pred_set = _coords_set(pred)
    gt_set = _coords_set(gt)
    inter = len(pred_set & gt_set)
    union = len(pred_set | gt_set)
    return {
        "iou": float(inter / union) if union else 1.0,
        "precision": float(inter / len(pred_set)) if pred_set else 0.0,
        "recall": float(inter / len(gt_set)) if gt_set else 0.0,
        "pred_count": len(pred_set),
        "gt_count": len(gt_set),
    }


def compute_decode_metrics(
    *,
    pred_coords: torch.Tensor | np.ndarray,
    gt_decode_coords: torch.Tensor | np.ndarray,
    raw_ind_coords: torch.Tensor | np.ndarray,
) -> Dict[str, float]:
    pred_vs_gt = coords_iou(pred_coords, gt_decode_coords)
    pred_vs_raw = coords_iou(pred_coords, raw_ind_coords)
    return {
        "decode_iou_pred_vs_gt_decode": pred_vs_gt["iou"],
        "decode_iou_pred_vs_raw_ind": pred_vs_raw["iou"],
        "decode_precision_pred_vs_raw_ind": pred_vs_raw["precision"],
        "decode_recall_pred_vs_raw_ind": pred_vs_raw["recall"],
        "pred_count": pred_vs_raw["pred_count"],
        "gt_decode_count": pred_vs_gt["gt_count"],
        "raw_ind_count": pred_vs_raw["gt_count"],
    }


def summarize_bucketed_part_metrics(
    rows: list[Dict[str, Any]],
    size_boundaries: tuple[float, float] = (500.0, 3000.0),
) -> Dict[str, float]:
    small_hi, medium_hi = (float(size_boundaries[0]), float(size_boundaries[1]))
    if not small_hi < medium_hi:
        raise ValueError(f"size_boundaries must be increasing, got {size_boundaries}")
    size_groups: Dict[str, list[Dict[str, Any]]] = {"small": [], "medium": [], "large": []}
    k_groups: Dict[str, list[Dict[str, Any]]] = {
        "k_1_2": [],
        "k_3_5": [],
        "k_6_10": [],
        "k_11_15": [],
        "k_16_plus": [],
    }
    for row in rows:
        count = float(row.get("part_raw_voxel_count", row.get("raw_ind_count", 0.0)))
        size_name = "small" if count < small_hi else "medium" if count < medium_hi else "large"
        size_groups[size_name].append(row)
        k_groups[k_bucket_name(int(row.get("object_part_count", 0)))].append(row)

    def _mean(group: list[Dict[str, Any]], key: str) -> float:
        return float(np.mean([float(row.get(key, 0.0)) for row in group])) if group else float("nan")

    summary: Dict[str, float] = {}
    for name, group in size_groups.items():
        summary[f"iou_size_{name}"] = _mean(group, "decode_iou_pred_vs_raw_ind")
        summary[f"recall_size_{name}"] = _mean(group, "decode_recall_pred_vs_raw_ind")
        summary[f"cos_size_{name}"] = _mean(group, "latent_cos")
    for name, group in k_groups.items():
        summary[f"iou_{name}"] = _mean(group, "decode_iou_pred_vs_raw_ind")
        summary[f"recall_{name}"] = _mean(group, "decode_recall_pred_vs_raw_ind")
        summary[f"cos_{name}"] = _mean(group, "latent_cos")
    return summary


def part_assignment_iou_matrix(
    pred_coords_list: List[torch.Tensor | np.ndarray],
    raw_coords_list: List[torch.Tensor | np.ndarray],
) -> torch.Tensor:
    if len(pred_coords_list) != len(raw_coords_list):
        raise ValueError(
            f"pred/raw list lengths must match, got {len(pred_coords_list)} and {len(raw_coords_list)}"
        )
    K = len(pred_coords_list)
    matrix = torch.zeros((K, K), dtype=torch.float32)
    for pred_idx, pred_coords in enumerate(pred_coords_list):
        for raw_idx, raw_coords in enumerate(raw_coords_list):
            matrix[pred_idx, raw_idx] = float(coords_iou(pred_coords, raw_coords)["iou"])
    return matrix


def summarize_assignment_matrix(matrix: torch.Tensor | np.ndarray) -> Dict[str, float]:
    mat = torch.as_tensor(matrix, dtype=torch.float32)
    if mat.dim() != 2 or mat.shape[0] != mat.shape[1]:
        raise ValueError(f"assignment matrix must be square [K,K], got {tuple(mat.shape)}")
    if mat.numel() == 0:
        return {"assignment_diag_iou": 0.0, "assignment_offdiag_max": 0.0}
    diag = torch.diag(mat)
    if mat.shape[0] <= 1:
        offdiag_max = torch.tensor(0.0, dtype=mat.dtype)
    else:
        offdiag = mat.masked_select(~torch.eye(mat.shape[0], dtype=torch.bool, device=mat.device))
        offdiag_max = offdiag.max() if offdiag.numel() else torch.tensor(0.0, dtype=mat.dtype)
    return {
        "assignment_diag_iou": float(diag.mean().item()),
        "assignment_offdiag_max": float(offdiag_max.item()),
    }


def _logit_stats(logits: torch.Tensor, thresholds: tuple[float, ...]) -> Dict[str, Any]:
    flat = logits.detach().float().reshape(-1).cpu()
    return {
        "logit_min": float(flat.min().item()),
        "logit_max": float(flat.max().item()),
        "logit_mean": float(flat.mean().item()),
        "logit_std": float(flat.std(unbiased=False).item()),
        "logit_p95": float(torch.quantile(flat, 0.95).item()),
        "logit_p99": float(torch.quantile(flat, 0.99).item()),
        "counts": {float(thr): int((flat > float(thr)).sum().item()) for thr in thresholds},
    }


@torch.no_grad()
def decode_ss_latent_to_coords_with_stats(
    decoder,
    z: torch.Tensor,
    threshold: float = 0.0,
    debug_thresholds: tuple[float, ...] | list[float] = (0.0, -0.25, -0.5, -1.0),
) -> tuple[torch.Tensor, Dict[str, Any]]:
    if z.dim() != 4:
        raise ValueError(f"z must be [C,16,16,16], got {tuple(z.shape)}")
    thresholds = tuple(float(x) for x in debug_thresholds)
    if float(threshold) not in thresholds:
        thresholds = (float(threshold),) + thresholds
    device = next(decoder.parameters()).device
    dtype = next(decoder.parameters()).dtype
    z_in = z.unsqueeze(0).to(device=device, dtype=dtype)
    logits = decoder(z_in)
    logits_3d = logits[0, 0].float()
    occ = logits_3d > float(threshold)
    return torch.nonzero(occ, as_tuple=False).long().cpu(), _logit_stats(logits_3d, thresholds)


@torch.no_grad()
def decode_ss_latent_to_coords(decoder, z: torch.Tensor, threshold: float = 0.0) -> torch.Tensor:
    coords, _stats = decode_ss_latent_to_coords_with_stats(decoder, z, threshold=threshold)
    return coords


@functools.lru_cache(maxsize=2)
def load_ss_decoder(ckpt_path: str | os.PathLike[str]):
    """Load SparseStructureDecoder from json+safetensors files."""
    import json
    from safetensors.torch import load_file
    from trellis.models.sparse_structure_vae import SparseStructureDecoder

    ckpt = Path(ckpt_path)
    if not ckpt.is_absolute():
        ckpt = Path.cwd() / ckpt
    base = ckpt.with_suffix("")
    config_path = base.with_suffix(".json")
    weights_path = ckpt if ckpt.suffix == ".safetensors" else base.with_suffix(".safetensors")
    if not config_path.is_file():
        raise FileNotFoundError(f"SS decoder config not found: {config_path}")
    if not weights_path.is_file():
        raise FileNotFoundError(f"SS decoder weights not found: {weights_path}")
    with config_path.open("r", encoding="utf-8") as f:
        config = json.load(f)
    decoder = SparseStructureDecoder(**config["args"]).cuda().eval()
    decoder.load_state_dict(load_file(str(weights_path)), strict=False)
    for p in decoder.parameters():
        p.requires_grad_(False)
    return decoder


def _coords_array(coords: np.ndarray) -> np.ndarray:
    arr = np.asarray(coords, dtype=np.int64)
    if arr.size == 0:
        return np.zeros((0, 3), dtype=np.int64)
    if arr.ndim != 2 or arr.shape[1] != 3:
        raise ValueError(f"coords expected [N,3], got {arr.shape}")
    return arr


def _coords_diff(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    left_set = set(map(tuple, _coords_array(left).tolist()))
    right_set = set(map(tuple, _coords_array(right).tolist()))
    diff = sorted(left_set - right_set)
    return np.asarray(diff, dtype=np.int64).reshape(-1, 3) if diff else np.zeros((0, 3), dtype=np.int64)


def _coords_union(coords_list: list[np.ndarray]) -> np.ndarray:
    merged: set[tuple[int, int, int]] = set()
    for coords in coords_list:
        merged.update(map(tuple, _coords_array(coords).tolist()))
    if not merged:
        return np.zeros((0, 3), dtype=np.int64)
    return np.asarray(sorted(merged), dtype=np.int64).reshape(-1, 3)


def _downsample(coords: np.ndarray, max_points: int = 9000) -> np.ndarray:
    arr = _coords_array(coords)
    if arr.shape[0] <= max_points:
        return arr
    idx = np.linspace(0, arr.shape[0] - 1, max_points, dtype=np.int64)
    return arr[idx]


def _setup_3d_axis(ax, title: str) -> None:
    ax.set_title(title, fontsize=9)
    ax.set_xlim(0, 63)
    ax.set_ylim(0, 63)
    ax.set_zlim(0, 63)
    ax.set_xlabel("x", labelpad=-5, fontsize=7)
    ax.set_ylabel("y", labelpad=-5, fontsize=7)
    ax.set_zlabel("z", labelpad=-5, fontsize=7)
    ax.tick_params(axis="both", which="major", labelsize=6, pad=-2)
    ax.view_init(elev=24, azim=-58)
    ax.set_box_aspect((1, 1, 1))
    ax.grid(True, linewidth=0.25, alpha=0.55)


def _scatter_coords(ax, coords: np.ndarray, color: str, label: str, alpha: float = 0.82) -> None:
    arr = _downsample(coords)
    if arr.size == 0:
        ax.text2D(0.42, 0.50, "empty", transform=ax.transAxes, color="#b00020", fontsize=10)
        return
    ax.scatter(arr[:, 0], arr[:, 1], arr[:, 2], s=5, c=color, alpha=alpha, depthshade=False, label=label)


def _scatter_overlay(ax, first: np.ndarray, second: np.ndarray, first_label: str, second_label: str) -> None:
    _scatter_coords(ax, second, "#1f77b4", second_label, alpha=0.55)
    _scatter_coords(ax, first, "#d62728", first_label, alpha=0.80)
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(loc="upper right", fontsize=6, frameon=True)


def _coords_to_mask(coords: np.ndarray, resolution: int = 64, block_size: int = 1) -> np.ndarray:
    arr = _coords_array(coords)
    display_resolution = resolution // block_size
    mask = np.zeros((display_resolution, display_resolution, display_resolution), dtype=bool)
    if arr.size == 0:
        return mask
    valid = np.all((arr >= 0) & (arr < resolution), axis=1)
    arr = arr[valid]
    if block_size > 1 and arr.size:
        arr = arr // block_size
    if arr.size:
        mask[arr[:, 0], arr[:, 1], arr[:, 2]] = True
    return mask


def _rgba(color: str, alpha: float) -> tuple[float, float, float, float]:
    import matplotlib.colors as mcolors

    r, g, b = mcolors.to_rgb(color)
    return (float(r), float(g), float(b), float(alpha))


def _setup_voxel_axis(ax, title: str, display_resolution: int = 64, block_size: int = 1) -> None:
    _setup_3d_axis(ax, title)
    ax.set_xlim(0, display_resolution)
    ax.set_ylim(0, display_resolution)
    ax.set_zlim(0, display_resolution)
    ticks = np.linspace(0, display_resolution, 5, dtype=int)
    labels = [str(int(t * block_size)) for t in ticks]
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_zticks(ticks)
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    ax.set_zticklabels(labels)


def _plot_voxel_layers(ax, layers: list[dict[str, Any]], title: str) -> None:
    block_size = _visual_voxel_block_size()
    display_resolution = 64 // block_size
    filled = np.zeros((display_resolution, display_resolution, display_resolution), dtype=bool)
    facecolors = np.zeros((display_resolution, display_resolution, display_resolution, 4), dtype=np.float32)
    for layer in layers:
        mask = _coords_to_mask(
            layer.get("coords", np.zeros((0, 3), dtype=np.int64)),
            block_size=block_size,
        )
        if not mask.any():
            continue
        filled[mask] = True
        facecolors[mask] = _rgba(str(layer.get("color", "#9e9e9e")), float(layer.get("alpha", 0.8)))
    if filled.any():
        ax.voxels(
            filled,
            facecolors=facecolors,
            edgecolors=(0.0, 0.0, 0.0, 0.36),
            linewidth=0.28,
            shade=True,
        )
    else:
        ax.text2D(0.42, 0.50, "empty", transform=ax.transAxes, color="#b00020", fontsize=10)
    _setup_voxel_axis(ax, f"{title}\nvoxel block={block_size}^3", display_resolution, block_size)


def _part_layers(parts: list[dict[str, Any]], coord_key: str, alpha: float = 0.86) -> list[dict[str, Any]]:
    layers = []
    for idx, part in enumerate(parts):
        layers.append({
            "coords": part.get(coord_key, np.zeros((0, 3), dtype=np.int64)),
            "color": _PART_COLORS[idx % len(_PART_COLORS)],
            "alpha": alpha,
            "label": str(part.get("name", f"part_{idx}")),
        })
    return layers


def _complete_object_layers(
    object_panel: Dict[str, Any],
    coord_key: str,
    *,
    body_alpha: float = 0.18,
    part_alpha: float = 0.82,
) -> list[dict[str, Any]]:
    surface_coords = _coords_array(object_panel.get("surface_coords", np.zeros((0, 3), dtype=np.int64)))
    parts = list(object_panel.get("parts", []))
    raw_part_union = _coords_union([
        part.get("raw_coords", np.zeros((0, 3), dtype=np.int64)) for part in parts
    ])
    residual_body = _coords_diff(surface_coords, raw_part_union)
    body_layer = {
        "coords": residual_body,
        "color": "#bdbdbd",
        "alpha": body_alpha,
        "label": "GT body context (not counted)",
    }
    return [body_layer, *_part_layers(parts, coord_key, alpha=part_alpha)]


def _complete_coords(object_panel: Dict[str, Any], coord_key: str) -> np.ndarray:
    layers = _complete_object_layers(object_panel, coord_key, body_alpha=1.0, part_alpha=1.0)
    return _coords_union([layer["coords"] for layer in layers])


def _target_parts_coords(object_panel: Dict[str, Any], coord_key: str) -> np.ndarray:
    return _coords_union([
        part.get(coord_key, np.zeros((0, 3), dtype=np.int64))
        for part in object_panel.get("parts", [])
    ])


def _populate_complete_object_metrics(object_panel: Dict[str, Any]) -> None:
    gt_target_parts = _target_parts_coords(object_panel, "raw_coords")
    pred_target_parts = _target_parts_coords(object_panel, "pred_coords")
    target_iou = coords_iou(pred_target_parts, gt_target_parts)
    object_panel["target_parts_iou_pred_vs_gt"] = target_iou["iou"]
    object_panel["target_parts_precision_pred_vs_gt"] = target_iou["precision"]
    object_panel["target_parts_recall_pred_vs_gt"] = target_iou["recall"]
    object_panel["pred_target_parts_count"] = target_iou["pred_count"]
    object_panel["gt_target_parts_count"] = target_iou["gt_count"]
    surface_coords = _coords_array(object_panel.get("surface_coords", np.zeros((0, 3), dtype=np.int64)))
    object_panel["body_context_count"] = int(_coords_diff(surface_coords, gt_target_parts).shape[0])

    part_metrics = []
    for part in object_panel.get("parts", []):
        metrics = coords_iou(
            part.get("pred_coords", np.zeros((0, 3), dtype=np.int64)),
            part.get("raw_coords", np.zeros((0, 3), dtype=np.int64)),
        )
        part_metrics.append({
            "name": str(part.get("name", "part")),
            "iou": metrics["iou"],
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "pred_count": metrics["pred_count"],
            "gt_count": metrics["gt_count"],
        })
    object_panel["part_metrics"] = part_metrics


def _matching_object_panel(
    row: Dict[str, Any],
    object_panels: list[Dict[str, Any]],
) -> Dict[str, Any] | None:
    for object_panel in object_panels:
        common_keys = [key for key in ("dataset_index", "sample_id", "obj_id") if key in row and key in object_panel]
        if common_keys and all(row[key] == object_panel[key] for key in common_keys):
            return object_panel
    return None


def _read_rgb_image(image_path: Path) -> np.ndarray:
    from PIL import Image

    with Image.open(image_path) as image:
        if image.mode in {"RGBA", "LA"} or ("transparency" in image.info):
            image = image.convert("RGBA")
            background = Image.new("RGBA", image.size, (255, 255, 255, 255))
            image = Image.alpha_composite(background, image).convert("RGB")
        else:
            image = image.convert("RGB")
        return np.asarray(image)


def _plot_image_cell(ax, paths: list[Path], index: int, title: str, plot_fn) -> None:
    ax.axis("off")
    if index >= len(paths):
        ax.set_title(title, fontsize=8)
        ax.text(0.5, 0.5, "missing", ha="center", va="center", fontsize=8, color="#b00020")
        return
    plot_fn(ax, paths[index], title)


def _plot_image_collection(fig, cell, paths: list[Path], title: str, plot_fn) -> None:
    if len(paths) <= 1:
        ax = fig.add_subplot(cell)
        _plot_image_cell(ax, paths, 0, title, plot_fn)
        return
    subgrid = cell.subgridspec(2, 2, wspace=0.03, hspace=0.12)
    for idx in range(4):
        ax = fig.add_subplot(subgrid[idx // 2, idx % 2])
        _plot_image_cell(ax, paths, idx, f"{title} {idx}", plot_fn)


def _plot_mask_path_axis(ax, mask_path: Path, title: str) -> None:
    _plot_label_mask_axis(ax, mask_path, title)


def _make_object_panel_png(object_panel: Dict[str, Any], out_path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    obj_id = object_panel.get("obj_id", "object")
    parts = list(object_panel.get("parts", []))
    rgb_paths = _image_paths_for_panel(object_panel, "rgb_views", max_views=4)
    mask_paths = _image_paths_for_panel(object_panel, "mask_views", max_views=4)
    global_coords = _coords_array(object_panel.get("global_coords", np.zeros((0, 3), dtype=np.int64)))
    _populate_complete_object_metrics(object_panel)
    target_iou = {
        "iou": float(object_panel["target_parts_iou_pred_vs_gt"]),
        "precision": float(object_panel["target_parts_precision_pred_vs_gt"]),
        "recall": float(object_panel["target_parts_recall_pred_vs_gt"]),
        "pred_count": int(object_panel["pred_target_parts_count"]),
        "gt_count": int(object_panel["gt_target_parts_count"]),
    }

    fig = plt.figure(figsize=(26, 6.8), dpi=140)
    fig.suptitle(
        f"{obj_id} target-parts inspection | parts={len(parts)} | "
        f"target parts IoU(pred vs GT parts)={target_iou['iou']:.4f} "
        f"P={target_iou['precision']:.4f} R={target_iou['recall']:.4f} "
        f"pred={target_iou['pred_count']} gt={target_iou['gt_count']} "
        f"| gray body is GT context, not counted",
        fontsize=12,
        y=0.97,
    )
    grid = fig.add_gridspec(
        nrows=1,
        ncols=5,
        left=0.02,
        right=0.99,
        top=0.88,
        bottom=0.16,
        wspace=0.06,
        width_ratios=[1.05, 1.05, 1.0, 1.18, 1.18],
    )

    _plot_image_collection(fig, grid[0, 0], rgb_paths, "input RGB", _plot_rgb_axis)
    _plot_image_collection(fig, grid[0, 1], mask_paths, "input mask", _plot_mask_path_axis)

    ax = fig.add_subplot(grid[0, 2], projection="3d")
    _plot_voxel_layers(
        ax,
        [{"coords": global_coords, "color": "#9e9e9e", "alpha": 0.28, "label": "input global"}],
        "input voxel\n(decoded z_global)",
    )

    ax = fig.add_subplot(grid[0, 3], projection="3d")
    _plot_voxel_layers(
        ax,
        _complete_object_layers(object_panel, "raw_coords", body_alpha=0.16, part_alpha=0.88),
        "GT target parts + body context\nsame part colors; body not counted",
    )

    ax = fig.add_subplot(grid[0, 4], projection="3d")
    _plot_voxel_layers(
        ax,
        _complete_object_layers(object_panel, "pred_coords", body_alpha=0.16, part_alpha=0.88),
        f"decoded pred target parts + body context\nparts IoU={target_iou['iou']:.4f}",
    )

    legend_items = [Patch(facecolor="#bdbdbd", alpha=0.35, label="GT body context (not counted)")]
    for idx, part in enumerate(parts):
        legend_items.append(Patch(facecolor=_PART_COLORS[idx % len(_PART_COLORS)], alpha=0.80, label=str(part.get("name", f"part_{idx}"))))
    fig.legend(handles=legend_items, loc="lower center", ncol=min(6, max(1, len(legend_items))), fontsize=7, frameon=True)
    fig.savefig(out_path)
    plt.close(fig)


def _print_progress(progress_prefix: str | None, message: str) -> None:
    if progress_prefix:
        print(f"{progress_prefix} {message}", flush=True)


def _should_report_progress(index_1based: int, total: int, every: int) -> bool:
    return index_1based == 1 or index_1based == total or index_1based % every == 0


def _format_threshold_counts(counts: Dict[float, int] | None) -> str:
    if not counts:
        return ""
    return " ".join(f"@{float(thr):g}:{int(count)}" for thr, count in sorted(counts.items(), reverse=True))


def _image_paths_for_panel(panel: Dict[str, Any], key: str, max_views: int = 4) -> list[Path]:
    paths = []
    for raw_path in panel.get(key, [])[:max_views]:
        paths.append(Path(str(raw_path)))
    return paths


def _plot_rgb_axis(ax, image_path: Path | None, title: str) -> None:
    ax.axis("off")
    ax.set_title(title, fontsize=9)
    if image_path is None:
        ax.text(0.5, 0.5, "missing RGB", ha="center", va="center", fontsize=9, color="#b00020")
        return
    if not image_path.is_file():
        ax.text(
            0.5,
            0.5,
            f"missing RGB\n{image_path}",
            ha="center",
            va="center",
            fontsize=7,
            color="#b00020",
            wrap=True,
        )
        return
    ax.imshow(_read_rgb_image(image_path))


def _plot_label_mask_axis(ax, mask_path: Path | None, title: str) -> None:
    ax.axis("off")
    ax.set_title(title, fontsize=9)
    if mask_path is None:
        ax.text(0.5, 0.5, "missing mask", ha="center", va="center", fontsize=9, color="#b00020")
        return
    if not mask_path.is_file():
        ax.text(
            0.5,
            0.5,
            f"missing mask\n{mask_path}",
            ha="center",
            va="center",
            fontsize=7,
            color="#b00020",
            wrap=True,
        )
        return
    try:
        mask = np.asarray(np.load(mask_path))
    except Exception as exc:  # noqa: BLE001 - inspection should render the failure visibly.
        ax.text(0.5, 0.5, f"mask load failed\n{exc}", ha="center", va="center", fontsize=8, color="#b00020")
        return
    if mask.ndim != 2:
        ax.text(0.5, 0.5, f"mask shape {mask.shape}", ha="center", va="center", fontsize=8, color="#b00020")
        return
    masked = np.ma.masked_where(mask == 0, mask)
    ax.imshow(mask == 0, cmap="gray", vmin=0, vmax=1)
    ax.imshow(masked, cmap="tab20", interpolation="nearest")
    labels = np.unique(mask)
    labels = labels[labels != 0]
    preview = ",".join(str(int(x)) for x in labels[:8])
    if labels.size > 8:
        preview += ",..."
    ax.text(
        0.02,
        0.98,
        f"labels: {preview or 'none'}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=7,
        color="white",
        bbox={"facecolor": "black", "alpha": 0.45, "pad": 2, "edgecolor": "none"},
    )


def _make_panel_png(
    row: Dict[str, Any],
    panel: Dict[str, Any],
    out_path: Path,
    complete_object_panel: Dict[str, Any] | None = None,
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pred = _coords_array(panel.get("pred_coords", np.zeros((0, 3), dtype=np.int64)))
    global_coords = _coords_array(panel.get("global_coords", np.zeros((0, 3), dtype=np.int64)))
    raw = _coords_array(panel.get("raw_gt_coords", np.zeros((0, 3), dtype=np.int64)))

    rgb_paths = _image_paths_for_panel(panel, "rgb_views", max_views=4)
    mask_paths = _image_paths_for_panel(panel, "mask_views", max_views=4)
    view_count = max(1, min(4, max(len(rgb_paths), len(mask_paths))))
    fig = plt.figure(figsize=(18, 13.2), dpi=140)
    fig.suptitle(
        f"{row.get('obj_id')} | {row.get('target_part_name')} | "
        f"latent_mse={float(row.get('latent_mse', 0.0)):.5f} "
        f"latent_l1={float(row.get('latent_l1', 0.0)):.5f} "
        f"raw_iou={float(row.get('decode_iou_pred_vs_raw_ind', 0.0)):.5f} "
        f"mse/zero={float(row.get('mse_vs_zero', 0.0)):.3f} "
        f"pred_logit_max={float(row.get('pred_logit_max', 0.0)):.2f}",
        fontsize=12,
        y=0.98,
    )
    pred_counts = _format_threshold_counts(row.get("pred_count_at_thresholds"))

    image_grid = fig.add_gridspec(
        nrows=3,
        ncols=4,
        height_ratios=[1.0, 1.0, 2.4],
        left=0.03,
        right=0.985,
        top=0.93,
        bottom=0.04,
        wspace=0.08,
        hspace=0.20,
    )
    for view_idx in range(4):
        rgb_ax = fig.add_subplot(image_grid[0, view_idx])
        mask_ax = fig.add_subplot(image_grid[1, view_idx])
        if view_idx < view_count:
            _plot_rgb_axis(
                rgb_ax,
                rgb_paths[view_idx] if view_idx < len(rgb_paths) else None,
                f"input RGB view {view_idx}",
            )
            _plot_label_mask_axis(
                mask_ax,
                mask_paths[view_idx] if view_idx < len(mask_paths) else None,
                f"label mask view {view_idx}",
            )
        else:
            rgb_ax.axis("off")
            mask_ax.axis("off")

    specs = [
        (
            "input global voxel\n(decoded z_global condition)",
            [{"coords": global_coords, "color": "#9e9e9e", "alpha": 0.24}],
        ),
        (
            f"pred part latent decode\n{pred_counts}",
            [{"coords": pred, "color": "#d62728", "alpha": 0.82}],
        ),
        (
            "GT target voxel\nraw ind target",
            [{"coords": raw, "color": "#1f77b4", "alpha": 0.82}],
        ),
        (
            f"pred(red) vs raw(blue)\npred={len(pred)} raw={len(raw)}",
            [
                {"coords": raw, "color": "#1f77b4", "alpha": 0.50},
                {"coords": pred, "color": "#d62728", "alpha": 0.78},
            ],
        ),
    ]
    for idx, (title, layers) in enumerate(specs, start=1):
        ax = fig.add_subplot(image_grid[2, idx - 1], projection="3d")
        _plot_voxel_layers(ax, layers, title)

    fig.savefig(out_path)
    plt.close(fig)


def write_part_ss_inspection_sample(
    root: str | os.PathLike[str],
    step: int,
    rows: List[Dict[str, Any]],
    panels: List[Dict[str, Any]],
    object_panel: Dict[str, Any] | None = None,
    progress_prefix: str | None = None,
    sample_index: int | None = None,
    total_samples: int | None = None,
) -> Path:
    out_dir = Path(root) / f"step_{int(step):06d}"
    out_dir.mkdir(parents=True, exist_ok=True)
    sample_label = ""
    if sample_index is not None and total_samples is not None:
        sample_label = f" sample {int(sample_index)}/{int(total_samples)}"
    _print_progress(
        progress_prefix,
        f"writing inspection PNGs{sample_label} to {out_dir} "
        f"(part_panels={len(panels)}, complete_voxel_panels={1 if object_panel else 0})",
    )
    if object_panel:
        png_name = object_panel.get("png_name") or f"{object_panel.get('obj_id', 'object')}_complete_voxels.png"
        object_panel["png_name"] = png_name
        _print_progress(
            progress_prefix,
            f"complete voxel panel{sample_label}: "
            f"obj={object_panel.get('obj_id')} parts={int(object_panel.get('part_count', len(object_panel.get('parts', []))))}",
        )
        _make_object_panel_png(object_panel, out_dir / png_name)
    total_panels = min(len(rows), len(panels))
    for idx, (row, panel) in enumerate(zip(rows, panels), start=1):
        png_name = row.get("png_name") or f"{row['obj_id']}_{row['target_part_name']}.png"
        row["png_name"] = png_name
        if _should_report_progress(idx, total_panels, every=10):
            _print_progress(
                progress_prefix,
                f"part voxel panel {idx}/{total_panels}: obj={row.get('obj_id')} part={row.get('target_part_name')}",
            )
        row_object_panel = (
            object_panel
            if object_panel is not None and _matching_object_panel(row, [object_panel]) is not None
            else None
        )
        _make_panel_png(row, panel, out_dir / png_name, complete_object_panel=row_object_panel)
    return out_dir


def write_part_ss_inspection(
    root: str | os.PathLike[str],
    step: int,
    rows: List[Dict[str, Any]],
    panels: List[Dict[str, Any]],
    object_panels: List[Dict[str, Any]] | None = None,
    progress_prefix: str | None = None,
    write_images: bool = True,
) -> Path:
    out_dir = Path(root) / f"step_{int(step):06d}"
    out_dir.mkdir(parents=True, exist_ok=True)
    object_panels = object_panels or []
    if write_images:
        _print_progress(
            progress_prefix,
            f"writing inspection PNGs to {out_dir} "
            f"(part_panels={len(panels)}, complete_voxel_panels={len(object_panels)})",
        )
        for idx, object_panel in enumerate(object_panels):
            png_name = object_panel.get("png_name") or f"{idx:03d}_{object_panel.get('obj_id', 'object')}_complete_voxels.png"
            object_panel["png_name"] = png_name
            _print_progress(
                progress_prefix,
                f"complete voxel panel {idx + 1}/{len(object_panels)}: "
                f"obj={object_panel.get('obj_id')} parts={int(object_panel.get('part_count', len(object_panel.get('parts', []))))}",
            )
            _make_object_panel_png(object_panel, out_dir / png_name)
        total_panels = min(len(rows), len(panels))
        for idx, (row, panel) in enumerate(zip(rows, panels), start=1):
            png_name = row.get("png_name") or f"{row['obj_id']}_{row['target_part_name']}.png"
            row["png_name"] = png_name
            if _should_report_progress(idx, total_panels, every=10):
                _print_progress(
                    progress_prefix,
                    f"part voxel panel {idx}/{total_panels}: obj={row.get('obj_id')} part={row.get('target_part_name')}",
                )
            _make_panel_png(
                row,
                panel,
                out_dir / png_name,
                complete_object_panel=_matching_object_panel(row, object_panels),
            )
    else:
        _print_progress(
            progress_prefix,
            f"writing inspection index to {out_dir} "
            f"(part_panels={len(panels)}, complete_voxel_panels={len(object_panels)})",
        )

    index_path = out_dir / "index.txt"
    lines = [
        f"# Part SS Latent Flow Inspection - step {int(step)}",
        "",
        "| obj_id | part_idx | part | latent_mse | zero_mse | mse/zero | latent_l1 | raw_iou | assign_diag | assign_offdiag | gt_decode_iou | pred_count | raw_count | pred_logit_max | gt_logit_max | pred_counts | png |",
        "|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row.get('obj_id')} | {int(row.get('part_index', 0))} | {row.get('target_part_name')} | "
            f"{float(row.get('latent_mse', 0.0)):.6f} | "
            f"{float(row.get('zero_mse', 0.0)):.6f} | "
            f"{float(row.get('mse_vs_zero', 0.0)):.3f} | "
            f"{float(row.get('latent_l1', 0.0)):.6f} | "
            f"{float(row.get('decode_iou_pred_vs_raw_ind', 0.0)):.6f} | "
            f"{float(row.get('assignment_diag_iou', 0.0)):.6f} | "
            f"{float(row.get('assignment_offdiag_max', 0.0)):.6f} | "
            f"{float(row.get('decode_iou_pred_vs_gt_decode', 0.0)):.6f} | "
            f"{int(row.get('pred_count', 0))} | {int(row.get('raw_ind_count', 0))} | "
            f"{float(row.get('pred_logit_max', 0.0)):.3f} | "
            f"{float(row.get('gt_logit_max', 0.0)):.3f} | "
            f"{_format_threshold_counts(row.get('pred_count_at_thresholds'))} | "
            f"[png]({row.get('png_name')}) |"
        )
    if object_panels:
        lines.extend([
            "",
            "## Target Parts Voxel Panels",
            "",
            "| obj_id | parts | target_parts_iou | precision | recall | pred_target_count | gt_target_count | body_context_count | png |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---|",
        ])
        for object_panel in object_panels:
            if "target_parts_iou_pred_vs_gt" not in object_panel:
                _populate_complete_object_metrics(object_panel)
            lines.append(
                f"| {object_panel.get('obj_id')} | {int(object_panel.get('part_count', len(object_panel.get('parts', []))))} | "
                f"{float(object_panel.get('target_parts_iou_pred_vs_gt', 0.0)):.6f} | "
                f"{float(object_panel.get('target_parts_precision_pred_vs_gt', 0.0)):.6f} | "
                f"{float(object_panel.get('target_parts_recall_pred_vs_gt', 0.0)):.6f} | "
                f"{int(object_panel.get('pred_target_parts_count', 0))} | "
                f"{int(object_panel.get('gt_target_parts_count', 0))} | "
                f"{int(object_panel.get('body_context_count', 0))} | "
                f"[png]({object_panel.get('png_name')}) |"
            )
        part_metric_rows = []
        for object_panel in object_panels:
            if "part_metrics" not in object_panel:
                _populate_complete_object_metrics(object_panel)
            for metric in object_panel.get("part_metrics", []):
                part_metric_rows.append((object_panel, metric))
        if part_metric_rows:
            lines.extend([
                "",
                "## Complete Object Part Metrics",
                "",
                "| obj_id | part | part_iou | precision | recall | pred_count | gt_count |",
                "|---|---|---:|---:|---:|---:|---:|",
            ])
            for object_panel, metric in part_metric_rows:
                lines.append(
                    f"| {object_panel.get('obj_id')} | {metric.get('name')} | "
                    f"{float(metric.get('iou', 0.0)):.6f} | "
                    f"{float(metric.get('precision', 0.0)):.6f} | "
                    f"{float(metric.get('recall', 0.0)):.6f} | "
                    f"{int(metric.get('pred_count', 0))} | "
                    f"{int(metric.get('gt_count', 0))} |"
                )
    index_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    _print_progress(progress_prefix, f"wrote inspection index: {index_path}")
    return index_path
