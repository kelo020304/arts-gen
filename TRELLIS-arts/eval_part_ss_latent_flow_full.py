#!/usr/bin/env python3
"""Full/subset evaluation report for Part SS Latent Flow checkpoints."""

from __future__ import annotations

import argparse
import json
import math
import random
import re
from pathlib import Path
from typing import Any

import numpy as np
import train_arts  # noqa: F401  # Registers lightweight trellis package stubs.
import torch
from torch.utils.data import DataLoader, Subset

from eval_part_ss_latent_flow import _apply_ckpt_latent_norm, _checkpoint_path, _cfg_dict, _dataset_cls, _load_model
from part_ss_full_eval_report import (
    K_BUCKETS,
    SIZE_BUCKETS,
    build_object_rows,
    enrich_part_rows,
    select_visualization_examples,
    summarize_tables,
    write_markdown_report,
)
from trellis.trainers.arts.part_ss_latent_flow import (
    _latent_metric_values,
    _object_id_filter_indices,
    _sample_indices_for_eval,
    _setup_rng,
    _to_device,
)
from trellis.trainers.arts.part_ss_latent_flow_eval import (
    _make_object_panel_png,
    compute_decode_metrics,
    coords_iou,
    decode_ss_latent_to_coords,
    decode_ss_latent_to_coords_with_stats,
    load_ss_decoder,
    part_assignment_iou_matrix,
    summarize_assignment_matrix,
)
from trellis.trainers.arts.part_ss_latent_flow_losses import (
    build_part_ss_sampler_kwargs,
    sample_part_ss_latent,
)
from trellis.utils.arts.config_utils import load_config


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, default=_json_default) + "\n")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _persist_metric_outputs(
    report_root: Path,
    *,
    part_rows: list[dict[str, Any]],
    object_rows: list[dict[str, Any]],
    summary: dict[str, Any],
) -> None:
    _write_jsonl(report_root / "part_metrics.jsonl", part_rows)
    _write_jsonl(report_root / "object_metrics.jsonl", object_rows)
    (report_root / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, default=_json_default) + "\n",
        encoding="utf-8",
    )


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, default=_json_default) + "\n", encoding="utf-8")


def _metric_outputs_complete(report_root: Path) -> bool:
    required = [
        report_root / "part_metrics.jsonl",
        report_root / "object_metrics.jsonl",
        report_root / "summary.json",
    ]
    if not all(path.is_file() for path in required):
        return False
    try:
        json.loads((report_root / "summary.json").read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return False
    return True


def _shard_sample_indices(indices: list[int], *, num_shards: int, shard_index: int) -> list[int]:
    n = int(num_shards)
    i = int(shard_index)
    if n < 1:
        raise ValueError(f"num_shards must be >= 1, got {num_shards}")
    if i < 0 or i >= n:
        raise ValueError(f"shard_index must be in [0, {n}), got {shard_index}")
    return list(indices)[i::n]


def _shard_report_root(step_root: Path, *, num_shards: int, shard_index: int) -> Path:
    n = int(num_shards)
    i = int(shard_index)
    if n <= 1:
        return step_root
    return step_root / "shards" / f"shard_{i:05d}_of_{n:05d}"


def _eval_sample_seed(base_seed: int, dataset_index: int) -> int:
    return (int(base_seed) + int(dataset_index) * 1_000_003) % (2**63 - 1)


def _set_eval_sample_seed(base_seed: int, dataset_index: int) -> int:
    seed = _eval_sample_seed(base_seed, dataset_index)
    random.seed(seed)
    np.random.seed(seed % (2**32 - 1))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    return seed


def _make_eval_initial_noise(
    batch: dict[str, Any],
    *,
    dataset_indices: list[int],
    base_seed: int,
) -> torch.Tensor:
    z_global = batch["z_global"]
    part_valid = batch["part_valid"]
    shape = (int(z_global.shape[0]), int(part_valid.shape[1])) + tuple(z_global.shape[1:])
    noise = torch.empty(shape, device=z_global.device, dtype=z_global.dtype)
    for row, dataset_index in enumerate(dataset_indices):
        seed = _eval_sample_seed(base_seed, int(dataset_index))
        try:
            generator = torch.Generator(device=z_global.device)
        except TypeError:
            generator = torch.Generator()
        generator.manual_seed(seed)
        noise[row] = torch.randn(
            shape[1:],
            generator=generator,
            device=z_global.device,
            dtype=z_global.dtype,
        )
    return noise


def _interior_recall(pred_coords, raw_ind_coords) -> tuple[float, int]:
    """Recall on the part's INTERIOR voxels (those whose 6 face-neighbors are all
    occupied in GT, i.e. fully buried / never visible). Directly measures amodal
    completion of the hidden inside. NaN when the part has no interior voxels."""
    gt = np.asarray(raw_ind_coords.cpu() if hasattr(raw_ind_coords, "cpu") else raw_ind_coords, dtype=np.int64)
    if gt.ndim != 2 or gt.shape[0] == 0:
        return float("nan"), 0
    gt_set = set(map(tuple, gt.tolist()))
    neigh = ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1))
    interior = [
        v for v in gt_set
        if all((v[0] + dx, v[1] + dy, v[2] + dz) in gt_set for dx, dy, dz in neigh)
    ]
    if not interior:
        return float("nan"), 0
    pred = np.asarray(pred_coords.cpu() if hasattr(pred_coords, "cpu") else pred_coords, dtype=np.int64)
    pred_set = set(map(tuple, pred.tolist())) if pred.size else set()
    hit = sum(1 for v in interior if v in pred_set)
    return float(hit / len(interior)), len(interior)


def _decode_sample_from_prediction(
    *,
    dataset,
    decoder,
    sample_meta: dict[str, Any],
    batch: dict[str, Any],
    pred: torch.Tensor,
    row_index: int,
    dataset_index: int,
    threshold: float,
    debug_thresholds: tuple[float, ...],
    include_panel_arrays: bool,
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any] | None]:
    row = int(row_index)
    obj_id = batch["obj_id"][row]
    valid_k = int(batch["part_valid"][row].sum().item())
    x_1 = batch["x_1_parts"]
    global_coords = decode_ss_latent_to_coords(decoder, batch["z_global"][row].detach().float().cpu(), threshold=threshold)
    rows: list[dict[str, Any]] = []
    pred_coords_list: list[torch.Tensor] = []
    raw_coords_list: list[torch.Tensor] = []
    object_parts: list[dict[str, Any]] = []
    for part_idx in range(valid_k):
        part_name = batch["target_part_names"][row][part_idx]
        pred_coords, pred_stats = decode_ss_latent_to_coords_with_stats(
            decoder,
            pred[row, part_idx].detach().float().cpu(),
            threshold=threshold,
            debug_thresholds=debug_thresholds,
        )
        gt_coords, gt_stats = decode_ss_latent_to_coords_with_stats(
            decoder,
            x_1[row, part_idx].detach().float().cpu(),
            threshold=threshold,
            debug_thresholds=debug_thresholds,
        )
        raw_ind_coords = batch["raw_ind_coords"][row][part_idx].detach().cpu()
        latent_metrics = _latent_metric_values(
            pred[row, part_idx].detach().unsqueeze(0),
            x_1[row, part_idx].detach().unsqueeze(0),
        )
        metrics = compute_decode_metrics(
            pred_coords=pred_coords,
            gt_decode_coords=gt_coords,
            raw_ind_coords=raw_ind_coords,
        )
        interior_recall, interior_count = _interior_recall(pred_coords, raw_ind_coords)
        row_metrics = {
            "obj_id": obj_id,
            "sample_id": batch["sample_id"][row],
            "dataset_index": int(dataset_index),
            "part_index": int(part_idx),
            "object_part_count": int(valid_k),
            "target_part_name": str(part_name),
            "target_slot": int(batch["target_slots"][row, part_idx].item()),
            "part_raw_voxel_count": int(raw_ind_coords.shape[0]),
            "interior_recall": interior_recall,
            "interior_voxel_count": interior_count,
            **latent_metrics,
            **metrics,
            "pred_logit_max": float(pred_stats["logit_max"]),
            "pred_logit_mean": float(pred_stats["logit_mean"]),
            "pred_logit_p99": float(pred_stats["logit_p99"]),
            "gt_logit_max": float(gt_stats["logit_max"]),
            "gt_logit_mean": float(gt_stats["logit_mean"]),
            "pred_count_at_thresholds": pred_stats["counts"],
            "gt_count_at_thresholds": gt_stats["counts"],
        }
        rows.append(row_metrics)
        pred_coords_list.append(pred_coords)
        raw_coords_list.append(raw_ind_coords)
        if include_panel_arrays:
            object_parts.append({
                "name": str(part_name),
                "pred_coords": pred_coords.numpy(),
                "gt_coords": gt_coords.numpy(),
                "raw_coords": raw_ind_coords.numpy(),
            })

    assignment = summarize_assignment_matrix(part_assignment_iou_matrix(pred_coords_list, raw_coords_list))
    for row_metrics in rows:
        row_metrics.update(assignment)

    target_iou = coords_iou(
        _coords_union(pred_coords_list),
        _coords_union(raw_coords_list),
    )
    object_metric_panel = {
        "obj_id": obj_id,
        "sample_id": batch["sample_id"][row],
        "dataset_index": int(dataset_index),
        "part_count": int(valid_k),
        "target_parts_iou_pred_vs_gt": target_iou["iou"],
        "target_parts_precision_pred_vs_gt": target_iou["precision"],
        "target_parts_recall_pred_vs_gt": target_iou["recall"],
        "pred_target_parts_count": target_iou["pred_count"],
        "gt_target_parts_count": target_iou["gt_count"],
    }
    object_panel = None
    if include_panel_arrays:
        object_panel = {
            **object_metric_panel,
            "surface_coords": batch["raw_surface_coords"][row].detach().cpu().numpy(),
            "global_coords": global_coords.numpy(),
            "rgb_views": [str(path) for path in dataset._iter_rgb_paths(sample_meta)],
            "mask_views": [str(path) for path in dataset._iter_mask_paths(sample_meta)],
            "parts": object_parts,
        }
    return rows, object_metric_panel, object_panel


@torch.no_grad()
def _decode_batch_samples(
    *,
    model,
    dataset,
    decoder,
    sample_metas: list[dict[str, Any]],
    batch: dict[str, Any],
    dataset_indices: list[int],
    device: torch.device,
    flow_cfg: dict[str, Any],
    threshold: float,
    debug_thresholds: tuple[float, ...],
    include_panel_arrays: bool,
    base_seed: int | None = None,
) -> list[tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any] | None]]:
    batch = _to_device(batch, device)
    if len(sample_metas) != len(dataset_indices):
        raise ValueError(f"sample_metas length {len(sample_metas)} does not match dataset_indices length {len(dataset_indices)}")
    if int(batch["z_global"].shape[0]) != len(dataset_indices):
        raise ValueError(f"batch size {int(batch['z_global'].shape[0])} does not match dataset_indices length {len(dataset_indices)}")
    initial_noise = None
    if base_seed is not None:
        initial_noise = _make_eval_initial_noise(batch, dataset_indices=dataset_indices, base_seed=base_seed)
    pred = sample_part_ss_latent(
        model,
        z_global=batch["z_global"],
        cond=batch["cond"],
        mask_token_labels=batch["mask_token_labels"],
        part_valid=batch["part_valid"],
        target_slots=batch["target_slots"],
        part_token_weights=batch.get("part_token_weights"),
        initial_noise=initial_noise,
        num_steps=int(flow_cfg.get("num_steps", 20)),
        noise_scale=float(flow_cfg.get("noise_scale", 1.0)),
        latent_scale=float(flow_cfg.get("latent_scale", 1.0)),
        **build_part_ss_sampler_kwargs(model, flow_cfg),
    )
    decoded = []
    for row, (sample_meta, dataset_index) in enumerate(zip(sample_metas, dataset_indices)):
        decoded.append(_decode_sample_from_prediction(
            dataset=dataset,
            decoder=decoder,
            sample_meta=sample_meta,
            batch=batch,
            pred=pred,
            row_index=row,
            dataset_index=int(dataset_index),
            threshold=threshold,
            debug_thresholds=debug_thresholds,
            include_panel_arrays=include_panel_arrays,
        ))
    return decoded


@torch.no_grad()
def _decode_one_sample(
    *,
    model,
    dataset,
    decoder,
    sample_meta: dict[str, Any],
    batch: dict[str, Any],
    dataset_index: int,
    device: torch.device,
    flow_cfg: dict[str, Any],
    threshold: float,
    debug_thresholds: tuple[float, ...],
    include_panel_arrays: bool,
    base_seed: int | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any] | None]:
    return _decode_batch_samples(
        model=model,
        dataset=dataset,
        decoder=decoder,
        sample_metas=[sample_meta],
        batch=batch,
        dataset_indices=[int(dataset_index)],
        device=device,
        flow_cfg=flow_cfg,
        threshold=threshold,
        debug_thresholds=debug_thresholds,
        include_panel_arrays=include_panel_arrays,
        base_seed=base_seed,
    )[0]

def _safe_filename_component(value: Any) -> str:
    text = str(value)
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", text).strip("._")
    return text or "unknown"


def _coords_union(coords_list: list[torch.Tensor | np.ndarray]) -> np.ndarray:
    all_coords: set[tuple[int, int, int]] = set()
    for coords in coords_list:
        arr = coords.detach().cpu().long().numpy() if isinstance(coords, torch.Tensor) else np.asarray(coords, dtype=np.int64)
        if arr.size == 0:
            continue
        if arr.ndim != 2 or arr.shape[1] != 3:
            raise ValueError(f"coords expected [N,3], got {arr.shape}")
        all_coords.update(map(tuple, arr.tolist()))
    if not all_coords:
        return np.zeros((0, 3), dtype=np.int64)
    return np.asarray(sorted(all_coords), dtype=np.int64)


def _select_sample_indices(dataset, *, max_samples: int, sample_mode: str, object_ids: str | None) -> list[int]:
    candidate_indices = _object_id_filter_indices(dataset, object_ids)
    if sample_mode == "all" or int(max_samples) < 0:
        if candidate_indices is not None:
            return list(candidate_indices)
        return list(range(len(dataset)))
    if sample_mode not in {"first", "spread"}:
        raise ValueError(f"unknown sample_mode={sample_mode!r}; expected first, spread, or all")
    return _sample_indices_for_eval(len(dataset), int(max_samples), sample_mode, candidate_indices)

def _plot_bar(path: Path, labels: list[str], values: list[float], *, title: str, ylabel: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7.5, 4.2), dpi=150)
    ax.bar(labels, values, color="#4c78a8")
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_ylim(0.0, 1.0)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _plot_heatmap(path: Path, matrix: list[list[float]], *, title: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.asarray(matrix, dtype=np.float32)
    fig, ax = plt.subplots(figsize=(8.4, 4.2), dpi=150)
    im = ax.imshow(arr, vmin=0.0, vmax=1.0, cmap="viridis")
    ax.set_xticks(range(len(K_BUCKETS)), K_BUCKETS, rotation=25, ha="right")
    ax.set_yticks(range(len(SIZE_BUCKETS)), SIZE_BUCKETS)
    ax.set_title(title)
    for y in range(arr.shape[0]):
        for x in range(arr.shape[1]):
            value = arr[y, x]
            text = "nan" if math.isnan(float(value)) else f"{float(value):.2f}"
            ax.text(x, y, text, ha="center", va="center", color="white", fontsize=8)
    fig.colorbar(im, ax=ax, label="IoU / recall")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _plot_scatter(path: Path, x: list[float], y: list[float], *, title: str, xlabel: str, ylabel: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6.8, 4.4), dpi=150)
    ax.scatter(x, y, s=10, alpha=0.55, color="#d95f02")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_ylim(0.0, 1.0)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _write_plots(report_root: Path, summary: dict[str, Any], part_rows: list[dict[str, Any]], object_rows: list[dict[str, Any]]) -> list[Path]:
    plots_dir = report_root / "plots"
    plots: list[Path] = []
    iou_size = plots_dir / "iou_by_size.png"
    _plot_bar(iou_size, list(SIZE_BUCKETS), [summary["by_size"][name]["iou_mean"] for name in SIZE_BUCKETS], title="Part IoU by size", ylabel="IoU")
    plots.append(iou_size.relative_to(report_root))

    recall_size = plots_dir / "recall_by_size.png"
    _plot_bar(recall_size, list(SIZE_BUCKETS), [summary["by_size"][name]["recall_mean"] for name in SIZE_BUCKETS], title="Part recall by size", ylabel="Recall")
    plots.append(recall_size.relative_to(report_root))

    iou_k = plots_dir / "iou_by_k.png"
    _plot_bar(iou_k, list(K_BUCKETS), [summary["by_k"][name]["iou_mean"] for name in K_BUCKETS], title="Part IoU by K bucket", ylabel="IoU")
    plots.append(iou_k.relative_to(report_root))

    heatmap = plots_dir / "iou_size_x_k_heatmap.png"
    _plot_heatmap(
        heatmap,
        [[summary["by_size_x_k"][size][k]["iou_mean"] for k in K_BUCKETS] for size in SIZE_BUCKETS],
        title="Part IoU by size x K bucket",
    )
    plots.append(heatmap.relative_to(report_root))

    recall_heatmap = plots_dir / "recall_size_x_k_heatmap.png"
    _plot_heatmap(
        recall_heatmap,
        [[summary["by_size_x_k"][size][k]["recall_mean"] for k in K_BUCKETS] for size in SIZE_BUCKETS],
        title="Part recall by size x K bucket",
    )
    plots.append(recall_heatmap.relative_to(report_root))

    if part_rows:
        size_scatter = plots_dir / "iou_vs_log_part_size.png"
        _plot_scatter(
            size_scatter,
            [math.log10(max(1, int(row.get("part_raw_voxel_count", 0)))) for row in part_rows],
            [float(row.get("part_iou", row.get("decode_iou_pred_vs_raw_ind", 0.0))) for row in part_rows],
            title="Part IoU vs log10(raw voxel count)",
            xlabel="log10(raw voxel count)",
            ylabel="part IoU",
        )
        plots.append(size_scatter.relative_to(report_root))

    if object_rows:
        object_k = plots_dir / "object_iou_vs_k.png"
        _plot_scatter(
            object_k,
            [float(row.get("object_part_count", 0)) for row in object_rows],
            [float(row.get("target_parts_iou_pred_vs_gt", 0.0)) for row in object_rows],
            title="Target-parts IoU vs K",
            xlabel="K target parts",
            ylabel="target-parts IoU",
        )
        plots.append(object_k.relative_to(report_root))
        mix = plots_dir / "object_iou_vs_size_mix_ratio.png"
        _plot_scatter(
            mix,
            [math.log10(max(1.0, float(row.get("size_mix_ratio", 1.0)))) for row in object_rows],
            [float(row.get("target_parts_iou_pred_vs_gt", 0.0)) for row in object_rows],
            title="Target-parts IoU vs size mix ratio",
            xlabel="log10(max part voxels / min part voxels)",
            ylabel="target-parts IoU",
        )
        plots.append(mix.relative_to(report_root))
    return plots


def _render_selected_examples(
    *,
    selected: list[dict[str, Any]],
    model,
    dataset,
    decoder,
    device: torch.device,
    flow_cfg: dict[str, Any],
    report_root: Path,
    threshold: float,
    debug_thresholds: tuple[float, ...],
    base_seed: int,
) -> list[dict[str, Any]]:
    if not selected:
        return selected
    indices = sorted({int(item["dataset_index"]) for item in selected})
    subset = Subset(dataset, indices)
    loader = DataLoader(subset, batch_size=1, shuffle=False, num_workers=0, collate_fn=dataset.collate_fn)
    rendered_by_index: dict[int, Path] = {}
    for idx, batch in enumerate(loader):
        dataset_index = indices[idx]
        sample_meta = dataset.samples[dataset_index]
        _rows, _object_metric_panel, object_panel = _decode_one_sample(
            model=model,
            dataset=dataset,
            decoder=decoder,
            sample_meta=sample_meta,
            batch=batch,
            dataset_index=dataset_index,
            device=device,
            flow_cfg=flow_cfg,
            threshold=threshold,
            debug_thresholds=debug_thresholds,
            include_panel_arrays=True,
            base_seed=base_seed,
        )
        if object_panel is None:
            continue
        obj_name = _safe_filename_component(object_panel.get("obj_id"))
        out_rel = Path("voxel_examples") / "shared" / f"{dataset_index:06d}_{obj_name}_target_parts.png"
        out_abs = report_root / out_rel
        out_abs.parent.mkdir(parents=True, exist_ok=True)
        _make_object_panel_png(object_panel, out_abs)
        rendered_by_index[dataset_index] = out_rel
    out = []
    for item in selected:
        copy = dict(item)
        png = rendered_by_index.get(int(copy["dataset_index"]))
        if png is not None:
            copy["png"] = str(png)
        out.append(copy)
    return out


def _finalize_full_eval_outputs(
    *,
    report_root: Path,
    part_rows: list[dict[str, Any]],
    object_rows: list[dict[str, Any]],
    summary: dict[str, Any],
    model,
    dataset,
    decoder,
    device: torch.device,
    flow_cfg: dict[str, Any],
    threshold: float,
    debug_thresholds: tuple[float, ...],
    base_seed: int,
    write_voxel_examples: int,
    ckpt_step: int,
    stage: str,
) -> Path:
    _persist_metric_outputs(report_root, part_rows=part_rows, object_rows=object_rows, summary=summary)
    selected = select_visualization_examples(part_rows, object_rows, limit_per_group=int(write_voxel_examples))
    selected = _render_selected_examples(
        selected=selected,
        model=model,
        dataset=dataset,
        decoder=decoder,
        device=device,
        flow_cfg=flow_cfg,
        report_root=report_root,
        threshold=threshold,
        debug_thresholds=debug_thresholds,
        base_seed=base_seed,
    )
    _write_json(report_root / "selected_examples.json", selected)
    plots = _write_plots(report_root, summary, part_rows, object_rows)
    return write_markdown_report(
        report_root / "report.md",
        summary=summary,
        part_rows=part_rows,
        object_rows=object_rows,
        selected_examples=selected,
        plots=plots,
        step=ckpt_step,
        stage=stage,
    )


def _merge_shard_outputs(
    *,
    report_root: Path,
    step: int,
    num_shards: int,
    stage: str,
    size_boundaries: tuple[float, float],
    write_voxel_examples: int,
) -> Path:
    step_root = report_root / f"step_{int(step):06d}"

    part_rows: list[dict[str, Any]] = []
    object_rows: list[dict[str, Any]] = []
    selected_png_by_group_index: dict[tuple[str, int], str] = {}
    selected_png_by_index: dict[int, str] = {}
    for shard_index in range(int(num_shards)):
        shard_root = _shard_report_root(step_root, num_shards=int(num_shards), shard_index=shard_index)
        if not _metric_outputs_complete(shard_root):
            raise FileNotFoundError(f"shard metrics are incomplete, cannot merge: {shard_root}")
        part_rows.extend(_read_jsonl(shard_root / "part_metrics.jsonl"))
        object_rows.extend(_read_jsonl(shard_root / "object_metrics.jsonl"))
        selected_path = shard_root / "selected_examples.json"
        if selected_path.is_file():
            prefix = Path("shards") / shard_root.name
            for item in json.loads(selected_path.read_text(encoding="utf-8")):
                if not item.get("png"):
                    continue
                dataset_index = int(item["dataset_index"])
                png = str(prefix / str(item["png"])).replace("\\", "/")
                selected_png_by_index.setdefault(dataset_index, png)
                selected_png_by_group_index.setdefault((str(item.get("group", "")), dataset_index), png)

    part_rows = enrich_part_rows(part_rows, size_boundaries=size_boundaries)
    object_rows = build_object_rows(part_rows, object_rows)
    summary = summarize_tables(part_rows, object_rows)
    selected_examples = select_visualization_examples(
        part_rows,
        object_rows,
        limit_per_group=int(write_voxel_examples),
    )
    for item in selected_examples:
        dataset_index = int(item["dataset_index"])
        png = selected_png_by_group_index.get((str(item.get("group", "")), dataset_index))
        if png is None:
            png = selected_png_by_index.get(dataset_index)
        if png is not None:
            item["png"] = png
    _persist_metric_outputs(step_root, part_rows=part_rows, object_rows=object_rows, summary=summary)
    _write_json(step_root / "selected_examples.json", selected_examples)
    plots = _write_plots(step_root, summary, part_rows, object_rows)
    return write_markdown_report(
        step_root / "report.md",
        summary=summary,
        part_rows=part_rows,
        object_rows=object_rows,
        selected_examples=selected_examples,
        plots=plots,
        step=int(step),
        stage=stage,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Part SS Latent Flow full/subset eval with Markdown tables.")
    parser.add_argument("--config", required=True, help="Part SS Latent Flow YAML config")
    parser.add_argument("--ckpt", default=None, help="Direct checkpoint path")
    parser.add_argument("--load-dir", default=None, help="Run directory or ckpts directory")
    parser.add_argument("--step", type=int, default=None, help="Checkpoint step used with --load-dir")
    parser.add_argument("--report-root", required=True, help="Directory for full eval metrics/report")
    parser.add_argument("--max-samples", type=int, default=-1, help="-1 means all selected samples")
    parser.add_argument("--batch-size", type=int, default=1, help="Per-process eval batch size")
    parser.add_argument("--sample-mode", choices=("first", "spread", "all"), default="all")
    parser.add_argument("--object-ids", default=None, help="Optional comma-separated object IDs")
    parser.add_argument("--num-steps", type=int, default=None, help="Override flow.num_steps")
    parser.add_argument("--device", default=None, help="cuda, cuda:0, or cpu")
    parser.add_argument("--write-voxel-examples", type=int, default=20, help="Worst/stratified voxel panels per group; 0 disables")
    parser.add_argument("--size-bucket-boundaries", nargs=2, type=float, default=(500.0, 3000.0))
    parser.add_argument("--num-shards", type=int, default=1, help="Number of eval shards")
    parser.add_argument("--shard-index", type=int, default=0, help="This process shard index")
    parser.add_argument("--skip-existing", action="store_true", help="Skip if this shard/report already has primary metric outputs")
    parser.add_argument("--merge-shards-only", action="store_true", help="Merge existing shard outputs without running model inference")
    parser.add_argument("--merge-step", type=int, default=None, help="Checkpoint step to merge when --merge-shards-only is set")
    parser.add_argument("overrides", nargs="*", help="OmegaConf dotlist overrides")
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
    if int(args.batch_size) < 1:
        raise ValueError(f"--batch-size must be >= 1, got {args.batch_size}")
    seed = int(getattr(cfg.training, "seed", 42)) if "training" in cfg else 42
    _setup_rng(seed)

    size_boundaries = tuple(float(x) for x in args.size_bucket_boundaries)
    if args.merge_shards_only:
        merge_step = args.merge_step if args.merge_step is not None else args.step
        if merge_step is None:
            raise ValueError("--merge-shards-only requires --merge-step or --step")
        report_path = _merge_shard_outputs(
            report_root=Path(args.report_root),
            step=int(merge_step),
            num_shards=int(args.num_shards),
            stage=stage,
            size_boundaries=size_boundaries,
            write_voxel_examples=int(args.write_voxel_examples),
        )
        print(f"[full_eval:merge] wrote report: {report_path}", flush=True)
        return 0

    if args.skip_existing and args.step is not None:
        expected_step_root = Path(args.report_root) / f"step_{int(args.step):06d}"
        expected_report_root = _shard_report_root(
            expected_step_root,
            num_shards=int(args.num_shards),
            shard_index=int(args.shard_index),
        )
        if _metric_outputs_complete(expected_report_root):
            print(f"[full_eval] skip existing metrics before model load: {expected_report_root}", flush=True)
            return 0

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    dataset = _dataset_cls(stage)(data_cfg)
    sample_indices = _select_sample_indices(
        dataset,
        max_samples=int(args.max_samples),
        sample_mode=str(args.sample_mode),
        object_ids=args.object_ids,
    )
    sample_indices = _shard_sample_indices(
        sample_indices,
        num_shards=int(args.num_shards),
        shard_index=int(args.shard_index),
    )
    ckpt_path = _checkpoint_path(args)
    model, ckpt_step, ckpt_cfg = _load_model(cfg, ckpt_path, device)
    _apply_ckpt_latent_norm(flow_cfg, ckpt_cfg)
    decoder = load_ss_decoder(eval_cfg["ss_decoder_ckpt"])
    threshold = float(eval_cfg.get("decode_threshold", 0.0))
    debug_thresholds = tuple(float(x) for x in eval_cfg.get("debug_thresholds", [0.0, -0.25, -0.5, -1.0]))
    step_root = Path(args.report_root) / f"step_{int(ckpt_step):06d}"
    report_root = _shard_report_root(
        step_root,
        num_shards=int(args.num_shards),
        shard_index=int(args.shard_index),
    )
    report_root.mkdir(parents=True, exist_ok=True)
    if args.skip_existing and _metric_outputs_complete(report_root):
        print(f"[full_eval] skip existing metrics: {report_root}", flush=True)
        return 0

    print("============================================================", flush=True)
    print("Part SS Latent Flow Full Eval", flush=True)
    print(f"  stage:           {stage}", flush=True)
    print(f"  checkpoint:      {ckpt_path}", flush=True)
    print(f"  checkpoint step: {ckpt_step}", flush=True)
    print(f"  device:          {device}", flush=True)
    print(f"  samples:         {len(sample_indices)}", flush=True)
    print(f"  batch_size:      {int(args.batch_size)}", flush=True)
    print(f"  shard:           {int(args.shard_index)}/{int(args.num_shards)}", flush=True)
    print(f"  sample_mode:     {args.sample_mode}", flush=True)
    print(f"  object_ids:      {args.object_ids or '<none>'}", flush=True)
    print(f"  report_root:     {report_root}", flush=True)
    print("============================================================", flush=True)

    subset = Subset(dataset, sample_indices)
    loader = DataLoader(subset, batch_size=int(args.batch_size), shuffle=False, num_workers=0, collate_fn=dataset.collate_fn)
    part_rows: list[dict[str, Any]] = []
    object_metric_panels: list[dict[str, Any]] = []
    processed = 0
    for batch in loader:
        batch_count = len(batch["obj_id"])
        batch_indices = sample_indices[processed:processed + batch_count]
        sample_metas = [dataset.samples[dataset_index] for dataset_index in batch_indices]
        print(
            f"[full_eval] samples {processed + 1}-{processed + batch_count}/{len(sample_indices)} "
            f"dataset_idx={batch_indices}",
            flush=True,
        )
        decoded = _decode_batch_samples(
            model=model,
            dataset=dataset,
            decoder=decoder,
            sample_metas=sample_metas,
            batch=batch,
            dataset_indices=batch_indices,
            device=device,
            flow_cfg=flow_cfg,
            threshold=threshold,
            debug_thresholds=debug_thresholds,
            include_panel_arrays=False,
            base_seed=seed,
        )
        for rows, object_metric_panel, _object_panel in decoded:
            part_rows.extend(rows)
            object_metric_panels.append(object_metric_panel)
        processed += batch_count

    part_rows = enrich_part_rows(part_rows, size_boundaries=size_boundaries)
    object_rows = build_object_rows(part_rows, object_metric_panels)
    summary = summarize_tables(part_rows, object_rows)
    report_path = _finalize_full_eval_outputs(
        report_root=report_root,
        part_rows=part_rows,
        object_rows=object_rows,
        summary=summary,
        model=model,
        dataset=dataset,
        decoder=decoder,
        device=device,
        flow_cfg=flow_cfg,
        threshold=threshold,
        debug_thresholds=debug_thresholds,
        base_seed=seed,
        write_voxel_examples=int(args.write_voxel_examples),
        ckpt_step=ckpt_step,
        stage=stage,
    )
    print(f"[full_eval] wrote report: {report_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
