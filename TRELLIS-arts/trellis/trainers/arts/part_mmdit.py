"""Trainer entry for PartMMDiT."""

from __future__ import annotations

import json
import math
import random
import time
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data import Subset
from torch.utils.data.distributed import DistributedSampler

from trellis.datasets.arts.part_mmdit import PartMMDiTDataset
from trellis.models.part_flow import PartMMDiTModel
from trellis.trainers.arts.part_mmdit_losses import rectified_flow_loss
from trellis.trainers.arts.part_ss_latent_flow_eval import (
    coords_iou,
    decode_ss_latent_to_coords,
    load_ss_decoder,
    part_assignment_iou_matrix,
    summarize_assignment_matrix,
)
from trellis.trainers.arts.part_ss_latent_flow import (
    _load_resume_checkpoint,
    _save_ckpt,
    _wrap_ddp_model,
)
from trellis.trainers.arts.part_ss_latent_flow_losses import sample_flow_timesteps
from trellis.utils.arts.config_utils import config_to_dict, load_config
from trellis.utils.arts.ddp_utils import setup_ddp


def _setup_rng(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _cfg_dict(cfg) -> dict:
    return config_to_dict(cfg) if not isinstance(cfg, dict) else dict(cfg)


def _to_device(batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    out = {}
    for key, value in batch.items():
        out[key] = value.to(device, non_blocking=True) if isinstance(value, torch.Tensor) else value
    return out


def _assert_no_native_fp16_trainable_params(model) -> None:
    raw_model = model.module if isinstance(model, DDP) else model
    fp16_params = [
        name
        for name, param in raw_model.named_parameters()
        if param.requires_grad and param.dtype == torch.float16
    ]
    if fp16_params:
        preview = ", ".join(fp16_params[:5])
        if len(fp16_params) > 5:
            preview += f", ... ({len(fp16_params)} total)"
        raise RuntimeError(
            "part_mmdit trainer expects FP32 trainable parameters. "
            "Use training.fp16=true for AMP autocast, but keep model.use_fp16=false. "
            f"Found native FP16 trainable parameters: {preview}"
        )


def _part_weight_kwargs(loss_cfg: dict) -> dict:
    return {
        "mode": str(loss_cfg.get("part_weight_mode", "none")),
        "alpha": float(loss_cfg.get("part_weight_alpha", 0.5)),
        "min_w": float(loss_cfg.get("part_weight_min", 0.5)),
        "max_w": float(loss_cfg.get("part_weight_max", 3.0)),
        "ref_mode": str(loss_cfg.get("part_weight_ref_mode", "median")),
        "normalize_per_object": bool(
            loss_cfg.get("normalize_part_weights_per_object", True)
        ),
    }


def _object_weight_kwargs(loss_cfg: dict) -> dict:
    return {
        "mode": str(loss_cfg.get("object_weight_mode", "none")),
        "k_ref": loss_cfg.get("object_weight_k_ref"),
        "min_w": float(loss_cfg.get("object_weight_min", 0.75)),
        "max_w": float(loss_cfg.get("object_weight_max", 2.0)),
    }


def _loss_kwargs(flow_cfg: dict, loss_cfg: dict) -> dict:
    fg_cfg = loss_cfg.get("foreground_weight", {})
    return {
        "latent_scale": float(flow_cfg.get("latent_scale", 1.0)),
        "cfg_dropout_name": float(loss_cfg.get("cfg_dropout_name", 0.1)),
        "cfg_dropout_anchor": float(loss_cfg.get("cfg_dropout_anchor", 0.1)),
        "foreground_weight": {
            "enabled": bool(fg_cfg.get("enabled", True)),
            "bg_weight": float(fg_cfg.get("bg_weight", 0.1)),
        },
        "part_weight_kwargs": _part_weight_kwargs(loss_cfg),
        "object_balanced": bool(loss_cfg.get("object_balanced", False)),
        "object_weight_kwargs": _object_weight_kwargs(loss_cfg),
    }


def _apply_timestep_shift(
    t: torch.Tensor,
    *,
    shift: float,
    t_min: float = 0.0,
    t_max: float = 1.0,
) -> torch.Tensor:
    """Apply RF dynamic timestep shift while preserving endpoints."""

    shift = float(shift)
    if shift < 0:
        raise ValueError(f"timestep shift must be >= 0, got {shift}")
    if shift == 0.0 or shift == 1.0:
        return t
    t_min = float(t_min)
    t_max = float(t_max)
    if not t_min < t_max:
        raise ValueError(f"t_min must be < t_max, got t_min={t_min} t_max={t_max}")
    t01 = ((t - t_min) / (t_max - t_min)).clamp(0.0, 1.0)
    shifted = (shift * t01) / (1.0 + (shift - 1.0) * t01)
    return shifted * (t_max - t_min) + t_min


def _dynamic_timestep_shift(model, flow_cfg: dict) -> float:
    if not bool(flow_cfg.get("dynamic_timestep_shift", False)):
        shift = float(flow_cfg.get("timestep_shift", 0.0))
        if shift < 0:
            raise ValueError(f"timestep_shift must be >= 0, got {shift}")
        return shift
    if "timestep_shift" in flow_cfg:
        shift = float(flow_cfg["timestep_shift"])
    else:
        raw_model = model.module if isinstance(model, DDP) else model
        token_count = (int(raw_model.resolution) // int(raw_model.patch_size)) ** 3
        shift = math.sqrt(
            token_count * int(raw_model.model_channels)
            / float(flow_cfg.get("timestep_shift_base", 4096.0))
        )
    if shift < 0:
        raise ValueError(f"dynamic timestep shift must be >= 0, got {shift}")
    return shift


def _sampler_timestep_shift(model, flow_cfg: dict) -> float:
    if "runtime_timestep_shift" in flow_cfg:
        shift = float(flow_cfg["runtime_timestep_shift"])
        if shift < 0:
            raise ValueError(f"runtime_timestep_shift must be >= 0, got {shift}")
        return shift
    return _dynamic_timestep_shift(model, flow_cfg)


def _safe_mean(values: list[float]) -> float:
    return float(np.mean(values)) if values else float("nan")


def _metric_str(value: float, precision: int = 4) -> str:
    value = float(value)
    if not math.isfinite(value):
        return "n/a"
    return f"{value:.{int(precision)}f}"


def _obj_id_set(dataset) -> set[str]:
    return {str(sample["obj_id"]) for sample in dataset.samples}


def _build_eval_dataset(
    eval_cfg: dict,
    *,
    dataset_cls: type,
    train_dataset,
):
    eval_data_cfg = eval_cfg.get("data")
    if not eval_data_cfg:
        return train_dataset
    eval_dataset = dataset_cls(dict(eval_data_cfg))
    overlap = sorted(_obj_id_set(train_dataset) & _obj_id_set(eval_dataset))
    if overlap:
        raise ValueError(
            "train/val obj_id overlap: "
            f"{overlap[:20]}{' ...' if len(overlap) > 20 else ''}"
        )
    return eval_dataset


def _eval_metric_line(step: int, metrics: Dict[str, float]) -> str:
    return (
        f"| {step} | "
        f"{metrics['train_latent_cos']:.4f} | "
        f"{metrics['train_latent_rel_l2']:.4f} | "
        f"{metrics['latent_cos']:.4f} | "
        f"{metrics['latent_rel_l2']:.4f} | "
        f"{metrics['target_iou']:.4f} | "
        f"{metrics['part_iou']:.4f} | "
        f"{metrics['recall']:.4f} | "
        f"{metrics['precision']:.4f} | "
        f"{metrics['small_recall']:.4f} | "
        f"{metrics['large_recall']:.4f} | "
        f"{metrics['count_error']:.4f} | "
        f"{metrics['offdiag']:.4f} |"
    )


def _append_eval_metrics_markdown(path: str | Path, *, step: int, metrics: Dict[str, float]) -> None:
    md_path = Path(path)
    md_path.parent.mkdir(parents=True, exist_ok=True)
    if not md_path.exists():
        md_path.write_text("# PartMMDiT Code Update Log\n", encoding="utf-8")
    existing = md_path.read_text(encoding="utf-8")
    header = (
        "\n## Train+Val Latent Metrics Curve\n\n"
        "| step | train_latent_cos | train_latent_rel_l2 | val_latent_cos | val_latent_rel_l2 | target_iou | part_iou | recall | precision | small_recall | large_recall | count_error | offdiag |\n"
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n"
    )
    with md_path.open("a", encoding="utf-8") as update_file:
        if "## Train+Val Latent Metrics Curve" not in existing:
            update_file.write(header)
        update_file.write(_eval_metric_line(step, metrics) + "\n")


def compute_part_mmdit_eval_metrics(
    pred_coords_by_object: list[list[torch.Tensor]],
    raw_coords_by_object: list[list[torch.Tensor]],
    part_raw_voxel_counts: torch.Tensor,
    part_valid: torch.Tensor,
    *,
    size_boundaries: tuple[float, float] = (500.0, 3000.0),
) -> Dict[str, float]:
    """Summarize decoded PartMMDiT predictions against raw per-part voxels."""

    if len(pred_coords_by_object) != len(raw_coords_by_object):
        raise ValueError("pred/raw object lists must have the same length")
    small_hi, medium_hi = float(size_boundaries[0]), float(size_boundaries[1])
    if not small_hi < medium_hi:
        raise ValueError(f"size_boundaries must be increasing, got {size_boundaries}")
    valid = part_valid.bool()
    counts = part_raw_voxel_counts.to(dtype=torch.float32)
    if counts.shape != valid.shape:
        raise ValueError(f"counts shape {tuple(counts.shape)} must match valid {tuple(valid.shape)}")

    per_part_iou = []
    per_part_recall = []
    per_part_precision = []
    small_recall = []
    medium_recall = []
    large_recall = []
    target_iou = []
    target_recall = []
    target_precision = []
    count_error = []
    offdiag = []
    total_pred = 0
    total_gt = 0
    for obj_idx, (pred_list, raw_list) in enumerate(zip(pred_coords_by_object, raw_coords_by_object)):
        valid_k = int(valid[obj_idx].sum().item())
        pred_valid = pred_list[:valid_k]
        raw_valid = raw_list[:valid_k]
        if len(pred_valid) != valid_k or len(raw_valid) != valid_k:
            raise ValueError(
                f"object {obj_idx} expected {valid_k} coord lists, got "
                f"pred={len(pred_valid)} raw={len(raw_valid)}"
            )
        pred_union = []
        raw_union = []
        for part_idx, (pred_coords, raw_coords) in enumerate(zip(pred_valid, raw_valid)):
            metric = coords_iou(pred_coords, raw_coords)
            per_part_iou.append(metric["iou"])
            per_part_recall.append(metric["recall"])
            per_part_precision.append(metric["precision"])
            part_count = float(counts[obj_idx, part_idx].item())
            if part_count < small_hi:
                small_recall.append(metric["recall"])
            elif part_count < medium_hi:
                medium_recall.append(metric["recall"])
            else:
                large_recall.append(metric["recall"])
            pred_union.append(pred_coords)
            raw_union.append(raw_coords)
            total_pred += int(metric["pred_count"])
            total_gt += int(metric["gt_count"])

        pred_cat = torch.cat(pred_union, dim=0) if pred_union else torch.zeros((0, 3), dtype=torch.long)
        raw_cat = torch.cat(raw_union, dim=0) if raw_union else torch.zeros((0, 3), dtype=torch.long)
        target_metric = coords_iou(pred_cat, raw_cat)
        target_iou.append(target_metric["iou"])
        target_recall.append(target_metric["recall"])
        target_precision.append(target_metric["precision"])
        count_error.append(
            abs(float(target_metric["pred_count"]) - float(target_metric["gt_count"]))
            / max(1.0, float(target_metric["gt_count"]))
        )
        offdiag.append(summarize_assignment_matrix(part_assignment_iou_matrix(pred_valid, raw_valid))["assignment_offdiag_max"])

    return {
        "target_iou": _safe_mean(target_iou),
        "part_iou": _safe_mean(per_part_iou),
        "recall": _safe_mean(target_recall),
        "precision": _safe_mean(target_precision),
        "part_recall": _safe_mean(per_part_recall),
        "part_precision": _safe_mean(per_part_precision),
        "small_recall": _safe_mean(small_recall),
        "medium_recall": _safe_mean(medium_recall),
        "large_recall": _safe_mean(large_recall),
        "target_recall": _safe_mean(target_recall),
        "target_precision": _safe_mean(target_precision),
        "count_error": _safe_mean(count_error),
        "offdiag": _safe_mean(offdiag),
        "pred_count": float(total_pred),
        "gt_count": float(total_gt),
    }


def compute_part_mmdit_latent_alignment(
    pred: torch.Tensor,
    target: torch.Tensor,
    part_valid: torch.Tensor,
    *,
    eps: float = 1.0e-8,
) -> Dict[str, float]:
    """Cosine and relative L2 over valid decoded-scale part latents."""
    if pred.shape != target.shape:
        raise ValueError(f"pred shape {tuple(pred.shape)} must match target {tuple(target.shape)}")
    if part_valid.shape != pred.shape[:2]:
        raise ValueError(f"part_valid shape {tuple(part_valid.shape)} must match pred[:2] {tuple(pred.shape[:2])}")
    valid = part_valid.bool().to(device=pred.device)
    if not bool(valid.any()):
        return {"latent_cos": math.nan, "latent_rel_l2": math.nan}
    pred_flat = pred[valid].detach().float().flatten(start_dim=1)
    target_flat = target[valid].detach().float().flatten(start_dim=1)
    dot = (pred_flat * target_flat).sum(dim=1)
    denom = pred_flat.norm(dim=1) * target_flat.norm(dim=1)
    cos = dot / denom.clamp_min(float(eps))
    rel_l2 = (pred_flat - target_flat).norm(dim=1) / target_flat.norm(dim=1).clamp_min(float(eps))
    return {
        "latent_cos": float(cos.mean().item()),
        "latent_rel_l2": float(rel_l2.mean().item()),
    }


def _mean_tensor(values: list[float]) -> float:
    return float(np.mean(values)) if values else float("nan")


@torch.no_grad()
def sample_part_mmdit_latent_cond_only_scaled(
    model,
    *,
    z_global: torch.Tensor,
    cond: torch.Tensor,
    name_tokens: torch.Tensor,
    name_mask: torch.Tensor,
    anchor: torch.Tensor,
    anchor_valid: torch.Tensor,
    part_valid: torch.Tensor,
    initial_noise: torch.Tensor | None = None,
    num_steps: int = 20,
    noise_scale: float = 1.0,
    timestep_shift: float = 0.0,
) -> torch.Tensor:
    """Clean conditional RF sampler that returns train-space scaled latents."""

    if int(num_steps) <= 0:
        raise ValueError(f"num_steps must be > 0, got {num_steps}")
    batch_size, part_count = part_valid.shape
    expected_shape = (batch_size, part_count) + tuple(z_global.shape[1:])
    if initial_noise is None:
        x = torch.randn(expected_shape, device=z_global.device, dtype=z_global.dtype)
    else:
        if tuple(initial_noise.shape) != expected_shape:
            raise ValueError(f"initial_noise shape {tuple(initial_noise.shape)} != {expected_shape}")
        x = initial_noise.to(device=z_global.device, dtype=z_global.dtype)
    valid_view = part_valid.to(device=z_global.device, dtype=z_global.dtype).view(
        batch_size,
        part_count,
        1,
        1,
        1,
        1,
    )
    x = x * float(noise_scale) * valid_view
    base_grid = torch.linspace(
        0.0,
        1.0,
        int(num_steps) + 1,
        device=z_global.device,
        dtype=z_global.dtype,
    )
    t_grid = _apply_timestep_shift(base_grid, shift=float(timestep_shift))
    drop_none = torch.zeros_like(part_valid.bool())
    for step_idx in range(int(num_steps)):
        t_value = t_grid[step_idx]
        dt = t_grid[step_idx + 1] - t_value
        t = torch.full((batch_size,), t_value, device=z_global.device, dtype=z_global.dtype)
        v = model(
            x,
            t,
            z_global,
            cond,
            name_tokens,
            name_mask,
            anchor,
            anchor_valid,
            part_valid,
            drop_name=drop_none,
            drop_anchor=drop_none,
        )
        x = (x + v * dt) * valid_view
    return x * valid_view


def _per_part_cos(pred: torch.Tensor, target: torch.Tensor, part_valid: torch.Tensor) -> torch.Tensor:
    if pred.shape != target.shape:
        raise ValueError(f"pred shape {tuple(pred.shape)} must match target {tuple(target.shape)}")
    flat_pred = pred.detach().float().flatten(start_dim=2)
    flat_target = target.detach().float().flatten(start_dim=2)
    dot = (flat_pred * flat_target).sum(dim=2)
    denom = flat_pred.norm(dim=2) * flat_target.norm(dim=2)
    cos = dot / denom.clamp_min(1.0e-8)
    return torch.where(
        part_valid.bool(),
        cos,
        torch.full_like(cos, float("nan")),
    )


def _select_group_cos(
    cos: torch.Tensor,
    batch: dict,
    *,
    group: str,
) -> float:
    values = []
    for row, part_names in enumerate(batch["target_part_names"]):
        valid_k = int(batch["part_valid"][row].sum().item())
        if group == "single":
            if valid_k == 1:
                values.append(float(cos[row, 0].item()))
        elif group == "multi":
            if valid_k > 1:
                values.extend(float(cos[row, part_idx].item()) for part_idx in range(valid_k))
        elif group == "buttons":
            for part_idx, part_name in enumerate(part_names[:valid_k]):
                if str(part_name).startswith("button"):
                    values.append(float(cos[row, part_idx].item()))
        else:
            raise ValueError(f"unknown cos group={group!r}")
    return _mean_tensor(values)


def _find_named_part(batch: dict, *, obj_id: str, part_name: str) -> tuple[int, int] | None:
    for row, current_obj_id in enumerate(batch["obj_id"]):
        if str(current_obj_id) != str(obj_id):
            continue
        valid_k = int(batch["part_valid"][row].sum().item())
        names = [str(name) for name in batch["target_part_names"][row][:valid_k]]
        if part_name in names:
            return row, names.index(part_name)
    return None


def _velocity_rel_l2(pred: torch.Tensor, target: torch.Tensor) -> float:
    pred_flat = pred.detach().float().reshape(-1)
    target_flat = target.detach().float().reshape(-1)
    return float(((pred_flat - target_flat).norm() / target_flat.norm().clamp_min(1.0e-8)).item())


def _velocity_rel_l2_by_part(
    pred: torch.Tensor,
    target: torch.Tensor,
    part_valid: torch.Tensor,
) -> torch.Tensor:
    if pred.shape != target.shape:
        raise ValueError(f"pred shape {tuple(pred.shape)} must match target {tuple(target.shape)}")
    if part_valid.shape != pred.shape[:2]:
        raise ValueError(f"part_valid shape {tuple(part_valid.shape)} must match pred[:2] {tuple(pred.shape[:2])}")
    pred_flat = pred.detach().float().flatten(start_dim=2)
    target_flat = target.detach().float().flatten(start_dim=2)
    rel_l2 = (pred_flat - target_flat).norm(dim=2) / target_flat.norm(dim=2).clamp_min(1.0e-8)
    return rel_l2[part_valid.bool()]


def _bucket_name(count: float, size_boundaries: tuple[float, float]) -> str:
    small_hi, medium_hi = float(size_boundaries[0]), float(size_boundaries[1])
    if not small_hi < medium_hi:
        raise ValueError(f"size_boundaries must be increasing, got {size_boundaries}")
    if float(count) < small_hi:
        return "small"
    if float(count) < medium_hi:
        return "medium"
    return "large"


def _empty_bucket_accum() -> dict[str, dict[str, list[float] | int]]:
    return {
        bucket: {
            "n_parts": 0,
            "cos": [],
            "part_iou": [],
            "recall": [],
            "vel_err_t0.02": [],
        }
        for bucket in ("small", "medium", "large")
    }


def _finalize_bucket_metrics(
    accum: dict[str, dict[str, list[float] | int]],
    *,
    prefix: str,
) -> Dict[str, float]:
    metrics: Dict[str, float] = {}
    for bucket, values in accum.items():
        metrics[f"{prefix}_{bucket}_n_parts"] = float(values["n_parts"])
        for key in ("cos", "part_iou", "recall", "vel_err_t0.02"):
            metrics[f"{prefix}_{bucket}_{key}"] = _safe_mean(values[key])
    return metrics


@torch.no_grad()
def _smoke_velocity_sweep(
    model,
    batch: dict,
    *,
    latent_scale: float,
    t_values: list[float],
    noise: torch.Tensor,
) -> dict[str, dict[float, float]]:
    raw_model = model.module if isinstance(model, DDP) else model
    raw_model.eval()
    x1_scaled = batch["x_1_parts"] * float(latent_scale)
    v_target = x1_scaled - noise
    drop_none = torch.zeros_like(batch["part_valid"].bool())
    targets = {
        "button_0": _find_named_part(batch, obj_id="100058", part_name="button_(top_handle)_0"),
        "lid_0": _find_named_part(batch, obj_id="100015", part_name="lid_0"),
    }
    metrics: dict[str, dict[float, float]] = {}
    for label, location in targets.items():
        metrics[label] = {float(t_value): float("nan") for t_value in t_values}
        if location is None:
            continue
        row, part_idx = location
        for t_value in t_values:
            t = torch.full(
                (batch["x_1_parts"].shape[0],),
                float(t_value),
                device=batch["x_1_parts"].device,
                dtype=batch["x_1_parts"].dtype,
            )
            tt = t.view(-1, 1, 1, 1, 1, 1)
            x_t = (1.0 - tt) * noise + tt * x1_scaled
            v_pred = raw_model(
                x_t,
                t,
                batch["z_global"],
                batch["cond"],
                batch["name_tokens"],
                batch["name_mask"],
                batch["anchor"],
                batch["anchor_valid"],
                batch["part_valid"],
                drop_name=drop_none,
                drop_anchor=drop_none,
            )
            metrics[label][float(t_value)] = _velocity_rel_l2(
                v_pred[row, part_idx],
                v_target[row, part_idx],
            )
    raw_model.train()
    return metrics


@torch.no_grad()
def _eval_smoke_curve(model, dataset, device, flow_cfg: dict, eval_cfg: dict) -> Dict[str, float]:
    raw_model = model.module if isinstance(model, DDP) else model
    raw_model.eval()
    max_samples = int(eval_cfg.get("max_samples", len(dataset)))
    loader = DataLoader(
        Subset(dataset, list(range(min(max_samples, len(dataset))))),
        batch_size=int(eval_cfg.get("batch_size", max_samples)),
        shuffle=False,
        num_workers=0,
        collate_fn=dataset.collate_fn,
    )
    batches = [_to_device(batch, device) for batch in loader]
    cos_single = []
    cos_multi = []
    cos_buttons = []
    sweep_accum: dict[str, dict[float, list[float]]] = {
        "button_0": {},
        "lid_0": {},
    }
    t_values = [float(x) for x in eval_cfg.get("velocity_t_values", [0.02, 0.1, 0.3, 0.5, 0.9])]
    generator = torch.Generator(device=device)
    generator.manual_seed(int(eval_cfg.get("noise_seed", 12345)))
    for batch in batches:
        noise = torch.randn(
            batch["x_1_parts"].shape,
            generator=generator,
            device=device,
            dtype=batch["x_1_parts"].dtype,
        )
        pred_scaled = sample_part_mmdit_latent_cond_only_scaled(
            raw_model,
            z_global=batch["z_global"],
            cond=batch["cond"],
            name_tokens=batch["name_tokens"],
            name_mask=batch["name_mask"],
            anchor=batch["anchor"],
            anchor_valid=batch["anchor_valid"],
            part_valid=batch["part_valid"],
            initial_noise=noise,
            num_steps=int(flow_cfg.get("num_steps", 20)),
            noise_scale=float(flow_cfg.get("noise_scale", 1.0)),
            timestep_shift=_sampler_timestep_shift(raw_model, flow_cfg),
        )
        x1_scaled = batch["x_1_parts"] * float(flow_cfg.get("latent_scale", 8.0))
        cos = _per_part_cos(pred_scaled, x1_scaled, batch["part_valid"])
        cos_single.append(_select_group_cos(cos, batch, group="single"))
        cos_multi.append(_select_group_cos(cos, batch, group="multi"))
        cos_buttons.append(_select_group_cos(cos, batch, group="buttons"))
        sweep = _smoke_velocity_sweep(
            raw_model,
            batch,
            latent_scale=float(flow_cfg.get("latent_scale", 8.0)),
            t_values=t_values,
            noise=noise,
        )
        for label, by_t in sweep.items():
            for t_value, value in by_t.items():
                if math.isnan(value):
                    continue
                sweep_accum.setdefault(label, {}).setdefault(t_value, []).append(value)
    out = {
        "cos_single": _safe_mean([v for v in cos_single if not math.isnan(v)]),
        "cos_multi": _safe_mean([v for v in cos_multi if not math.isnan(v)]),
        "cos_buttons": _safe_mean([v for v in cos_buttons if not math.isnan(v)]),
    }
    for label, by_t in sweep_accum.items():
        for t_value in t_values:
            out[f"vel_err_{label}_t{t_value:g}"] = _safe_mean(by_t.get(t_value, []))
    raw_model.train()
    return out


def _eval_sample_count(dataset, eval_cfg: dict, key: str) -> int:
    value = int(eval_cfg.get(key, eval_cfg.get("max_samples", len(dataset))))
    value = min(value, len(dataset))
    if value <= 0:
        raise ValueError(f"eval sample count for {key!r} must be > 0, got {value}")
    return value


def _eval_sample_indices(
    dataset,
    eval_cfg: dict,
    max_samples_key: str,
    *,
    prefer_obj_ids_key: str | None = None,
) -> list[int]:
    max_samples = _eval_sample_count(dataset, eval_cfg, max_samples_key)
    prefer_obj_ids = []
    if prefer_obj_ids_key is not None:
        prefer_obj_ids = [str(obj_id) for obj_id in eval_cfg.get(prefer_obj_ids_key, [])]
    if len(prefer_obj_ids) > max_samples:
        raise ValueError(
            f"eval.{prefer_obj_ids_key} has {len(prefer_obj_ids)} ids but "
            f"{max_samples_key}={max_samples}; increase the eval sample count"
        )
    first_idx_by_obj_id = {}
    for idx, sample in enumerate(dataset.samples):
        obj_id = str(sample["obj_id"])
        if obj_id not in first_idx_by_obj_id:
            first_idx_by_obj_id[obj_id] = idx

    selected = []
    selected_set = set()
    for obj_id in prefer_obj_ids:
        if obj_id not in first_idx_by_obj_id:
            raise ValueError(
                f"eval.{prefer_obj_ids_key} obj_id={obj_id!r} is not present in "
                f"the selected eval split"
            )
        idx = first_idx_by_obj_id[obj_id]
        if idx not in selected_set:
            selected.append(idx)
            selected_set.add(idx)
    for idx in range(len(dataset)):
        if len(selected) >= max_samples:
            break
        if idx in selected_set:
            continue
        selected.append(idx)
        selected_set.add(idx)
    return selected


@torch.no_grad()
def _eval_latent_group_cos(
    model,
    dataset,
    device,
    flow_cfg: dict,
    eval_cfg: dict,
    *,
    max_samples_key: str = "max_samples",
    prefer_obj_ids_key: str | None = None,
) -> Dict[str, float]:
    raw_model = model.module if isinstance(model, DDP) else model
    raw_model.eval()
    sample_indices = _eval_sample_indices(
        dataset,
        eval_cfg,
        max_samples_key,
        prefer_obj_ids_key=prefer_obj_ids_key,
    )
    loader = DataLoader(
        Subset(dataset, sample_indices),
        batch_size=int(eval_cfg.get("batch_size", 1)),
        shuffle=False,
        num_workers=0,
        collate_fn=dataset.collate_fn,
    )
    generator = torch.Generator(device=device)
    generator.manual_seed(int(eval_cfg.get("noise_seed", 12345)))
    cos_single = []
    cos_multi = []
    cos_buttons = []
    for batch in loader:
        batch = _to_device(batch, device)
        noise = torch.randn(
            batch["x_1_parts"].shape,
            generator=generator,
            device=device,
            dtype=batch["x_1_parts"].dtype,
        )
        pred_scaled = sample_part_mmdit_latent_cond_only_scaled(
            raw_model,
            z_global=batch["z_global"],
            cond=batch["cond"],
            name_tokens=batch["name_tokens"],
            name_mask=batch["name_mask"],
            anchor=batch["anchor"],
            anchor_valid=batch["anchor_valid"],
            part_valid=batch["part_valid"],
            initial_noise=noise,
            num_steps=int(flow_cfg.get("num_steps", 20)),
            noise_scale=float(flow_cfg.get("noise_scale", 1.0)),
            timestep_shift=_sampler_timestep_shift(raw_model, flow_cfg),
        )
        x1_scaled = batch["x_1_parts"] * float(flow_cfg.get("latent_scale", 8.0))
        cos = _per_part_cos(pred_scaled, x1_scaled, batch["part_valid"])
        cos_single.append(_select_group_cos(cos, batch, group="single"))
        cos_multi.append(_select_group_cos(cos, batch, group="multi"))
        cos_buttons.append(_select_group_cos(cos, batch, group="buttons"))
    raw_model.train()
    return {
        "cos_single": _safe_mean([value for value in cos_single if not math.isnan(value)]),
        "cos_multi": _safe_mean([value for value in cos_multi if not math.isnan(value)]),
        "cos_buttons": _safe_mean([value for value in cos_buttons if not math.isnan(value)]),
    }


@torch.no_grad()
def _eval_velocity_sweep_mean(
    model,
    dataset,
    device,
    flow_cfg: dict,
    eval_cfg: dict,
    *,
    max_samples_key: str = "velocity_max_samples",
    prefer_obj_ids_key: str | None = None,
) -> Dict[str, float]:
    raw_model = model.module if isinstance(model, DDP) else model
    raw_model.eval()
    sample_indices = _eval_sample_indices(
        dataset,
        eval_cfg,
        max_samples_key,
        prefer_obj_ids_key=prefer_obj_ids_key,
    )
    loader = DataLoader(
        Subset(dataset, sample_indices),
        batch_size=int(eval_cfg.get("velocity_batch_size", eval_cfg.get("batch_size", 1))),
        shuffle=False,
        num_workers=0,
        collate_fn=dataset.collate_fn,
    )
    t_values = [float(x) for x in eval_cfg.get("velocity_t_values", [0.02, 0.1, 0.3, 0.5, 0.9])]
    accum: dict[float, list[float]] = {t_value: [] for t_value in t_values}
    generator = torch.Generator(device=device)
    generator.manual_seed(int(eval_cfg.get("noise_seed", 12345)))
    for batch in loader:
        batch = _to_device(batch, device)
        noise = torch.randn(
            batch["x_1_parts"].shape,
            generator=generator,
            device=device,
            dtype=batch["x_1_parts"].dtype,
        )
        x1_scaled = batch["x_1_parts"] * float(flow_cfg.get("latent_scale", 8.0))
        v_target = x1_scaled - noise
        drop_none = torch.zeros_like(batch["part_valid"].bool())
        for t_value in t_values:
            t = torch.full(
                (batch["x_1_parts"].shape[0],),
                float(t_value),
                device=batch["x_1_parts"].device,
                dtype=batch["x_1_parts"].dtype,
            )
            tt = t.view(-1, 1, 1, 1, 1, 1)
            x_t = (1.0 - tt) * noise + tt * x1_scaled
            v_pred = raw_model(
                x_t,
                t,
                batch["z_global"],
                batch["cond"],
                batch["name_tokens"],
                batch["name_mask"],
                batch["anchor"],
                batch["anchor_valid"],
                batch["part_valid"],
                drop_name=drop_none,
                drop_anchor=drop_none,
            )
            accum[t_value].extend(
                float(value) for value in _velocity_rel_l2_by_part(
                    v_pred,
                    v_target,
                    batch["part_valid"],
                ).detach().cpu().tolist()
            )
    raw_model.train()
    return {f"vel_err_t{t_value:g}": _safe_mean(accum[t_value]) for t_value in t_values}


@torch.no_grad()
def _eval_bucket_metrics_for_dataset(
    model,
    dataset,
    device,
    flow_cfg: dict,
    eval_cfg: dict,
    *,
    max_samples_key: str,
    prefer_obj_ids_key: str | None = None,
    prefix: str,
) -> Dict[str, float]:
    raw_model = model.module if isinstance(model, DDP) else model
    raw_model.eval()
    decoder = load_ss_decoder(eval_cfg["ss_decoder_ckpt"])
    size_boundaries = tuple(
        float(x) for x in eval_cfg.get("size_bucket_boundaries", [500.0, 3000.0])
    )
    sample_indices = _eval_sample_indices(
        dataset,
        eval_cfg,
        max_samples_key,
        prefer_obj_ids_key=prefer_obj_ids_key,
    )
    loader = DataLoader(
        Subset(dataset, sample_indices),
        batch_size=int(eval_cfg.get("batch_size", 1)),
        shuffle=False,
        num_workers=0,
        collate_fn=dataset.collate_fn,
    )
    accum = _empty_bucket_accum()
    generator = torch.Generator(device=device)
    generator.manual_seed(int(eval_cfg.get("noise_seed", 12345)))
    for batch in loader:
        batch = _to_device(batch, device)
        noise = torch.randn(
            batch["x_1_parts"].shape,
            generator=generator,
            device=device,
            dtype=batch["x_1_parts"].dtype,
        )
        pred_scaled = sample_part_mmdit_latent_cond_only_scaled(
            raw_model,
            z_global=batch["z_global"],
            cond=batch["cond"],
            name_tokens=batch["name_tokens"],
            name_mask=batch["name_mask"],
            anchor=batch["anchor"],
            anchor_valid=batch["anchor_valid"],
            part_valid=batch["part_valid"],
            initial_noise=noise,
            num_steps=int(flow_cfg.get("num_steps", 20)),
            noise_scale=float(flow_cfg.get("noise_scale", 1.0)),
            timestep_shift=_sampler_timestep_shift(raw_model, flow_cfg),
        )
        x1_scaled = batch["x_1_parts"] * float(flow_cfg.get("latent_scale", 8.0))
        cos = _per_part_cos(pred_scaled, x1_scaled, batch["part_valid"])

        t_value = 0.02
        t = torch.full(
            (batch["x_1_parts"].shape[0],),
            t_value,
            device=batch["x_1_parts"].device,
            dtype=batch["x_1_parts"].dtype,
        )
        tt = t.view(-1, 1, 1, 1, 1, 1)
        x_t = (1.0 - tt) * noise + tt * x1_scaled
        drop_none = torch.zeros_like(batch["part_valid"].bool())
        v_pred = raw_model(
            x_t,
            t,
            batch["z_global"],
            batch["cond"],
            batch["name_tokens"],
            batch["name_mask"],
            batch["anchor"],
            batch["anchor_valid"],
            batch["part_valid"],
            drop_name=drop_none,
            drop_anchor=drop_none,
        )
        v_target = x1_scaled - noise
        vel_err = torch.full(
            batch["part_valid"].shape,
            float("nan"),
            device=batch["x_1_parts"].device,
            dtype=torch.float32,
        )
        valid_mask = batch["part_valid"].bool()
        pred_flat = v_pred.detach().float().flatten(start_dim=2)
        target_flat = v_target.detach().float().flatten(start_dim=2)
        vel_all = (
            (pred_flat - target_flat).norm(dim=2)
            / target_flat.norm(dim=2).clamp_min(1.0e-8)
        )
        vel_err[valid_mask] = vel_all[valid_mask]

        pred_raw = pred_scaled / float(flow_cfg.get("latent_scale", 8.0))
        for row in range(batch["x_1_parts"].shape[0]):
            valid_k = int(batch["part_valid"][row].sum().item())
            for part_idx in range(valid_k):
                count = float(batch["part_raw_voxel_counts"][row, part_idx].item())
                bucket = _bucket_name(count, size_boundaries)
                pred_coords = decode_ss_latent_to_coords(
                    decoder,
                    pred_raw[row, part_idx].detach().float().cpu(),
                    threshold=float(eval_cfg.get("decode_threshold", 0.0)),
                )
                raw_coords = batch["raw_ind_coords"][row][part_idx].detach().cpu()
                decoded = coords_iou(pred_coords, raw_coords)
                accum[bucket]["n_parts"] += 1
                accum[bucket]["cos"].append(float(cos[row, part_idx].item()))
                accum[bucket]["part_iou"].append(float(decoded["iou"]))
                accum[bucket]["recall"].append(float(decoded["recall"]))
                accum[bucket]["vel_err_t0.02"].append(float(vel_err[row, part_idx].item()))
    raw_model.train()
    return _finalize_bucket_metrics(accum, prefix=prefix)


@torch.no_grad()
def _eval_bucket_metrics(
    model,
    train_dataset,
    eval_dataset,
    device,
    flow_cfg: dict,
    eval_cfg: dict,
) -> Dict[str, float]:
    metrics = {}
    metrics.update(
        _eval_bucket_metrics_for_dataset(
            model,
            train_dataset,
            device,
            flow_cfg,
            eval_cfg,
            max_samples_key="train_max_samples",
            prefer_obj_ids_key="train_prefer_obj_ids",
            prefix="train",
        )
    )
    metrics.update(
        _eval_bucket_metrics_for_dataset(
            model,
            eval_dataset,
            device,
            flow_cfg,
            eval_cfg,
            max_samples_key="max_samples",
            prefer_obj_ids_key="prefer_obj_ids",
            prefix="val",
        )
    )
    return metrics


def _bucket_eval_log_lines(
    *,
    step: int,
    wall: float,
    loss_value: float,
    lr: float,
    grad_norm: float,
    metrics: Dict[str, float],
) -> list[str]:
    lines = [
        (
            f"  [EVAL_BUCKET step={step} wall={wall:.1f}s] "
            f"loss={loss_value:.4f} lr={lr:.2e} gradnorm={grad_norm:.4f}"
        )
    ]
    for split in ("train", "val"):
        for bucket in ("small", "medium", "large"):
            lines.append(
                f"  [EVAL_BUCKET step={step} split={split} bucket={bucket}] "
                f"n_parts={int(metrics[f'{split}_{bucket}_n_parts'])} "
                f"cos={_metric_str(metrics[f'{split}_{bucket}_cos'])} "
                f"part_iou={_metric_str(metrics[f'{split}_{bucket}_part_iou'])} "
                f"recall={_metric_str(metrics[f'{split}_{bucket}_recall'])} "
                f"vel_err_t0.02={_metric_str(metrics[f'{split}_{bucket}_vel_err_t0.02'])}"
            )
    return lines


def _sample_eval_part_latent_raw(
    raw_model,
    batch: dict,
    flow_cfg: dict,
    eval_cfg: dict,
) -> torch.Tensor:
    if bool(eval_cfg.get("cond_only", False)):
        pred_scaled = sample_part_mmdit_latent_cond_only_scaled(
            raw_model,
            z_global=batch["z_global"],
            cond=batch["cond"],
            name_tokens=batch["name_tokens"],
            name_mask=batch["name_mask"],
            anchor=batch["anchor"],
            anchor_valid=batch["anchor_valid"],
            part_valid=batch["part_valid"],
            num_steps=int(flow_cfg.get("num_steps", 20)),
            noise_scale=float(flow_cfg.get("noise_scale", 1.0)),
            timestep_shift=_sampler_timestep_shift(raw_model, flow_cfg),
        )
        return pred_scaled / float(flow_cfg.get("latent_scale", 8.0))
    return sample_part_mmdit_latent(
        raw_model,
        z_global=batch["z_global"],
        cond=batch["cond"],
        name_tokens=batch["name_tokens"],
        name_mask=batch["name_mask"],
        anchor=batch["anchor"],
        anchor_valid=batch["anchor_valid"],
        part_valid=batch["part_valid"],
        num_steps=int(flow_cfg.get("num_steps", 20)),
        noise_scale=float(flow_cfg.get("noise_scale", 1.0)),
        latent_scale=float(flow_cfg.get("latent_scale", 8.0)),
        s_name=float(flow_cfg.get("s_name", 1.0)),
        s_anchor=float(flow_cfg.get("s_anchor", 1.0)),
        timestep_shift=_sampler_timestep_shift(raw_model, flow_cfg),
    )


@torch.no_grad()
def sample_part_mmdit_latent(
    model,
    *,
    z_global: torch.Tensor,
    cond: torch.Tensor,
    name_tokens: torch.Tensor,
    name_mask: torch.Tensor,
    anchor: torch.Tensor,
    anchor_valid: torch.Tensor,
    part_valid: torch.Tensor,
    initial_noise: torch.Tensor | None = None,
    num_steps: int = 20,
    noise_scale: float = 1.0,
    latent_scale: float = 8.0,
    s_name: float = 1.0,
    s_anchor: float = 1.0,
    timestep_shift: float = 0.0,
) -> torch.Tensor:
    if int(num_steps) <= 0:
        raise ValueError(f"num_steps must be > 0, got {num_steps}")
    if float(latent_scale) <= 0:
        raise ValueError(f"latent_scale must be > 0, got {latent_scale}")
    batch_size, part_count = part_valid.shape
    expected_shape = (batch_size, part_count) + tuple(z_global.shape[1:])
    if initial_noise is None:
        x = torch.randn(expected_shape, device=z_global.device, dtype=z_global.dtype)
    else:
        if tuple(initial_noise.shape) != expected_shape:
            raise ValueError(f"initial_noise shape {tuple(initial_noise.shape)} != {expected_shape}")
        x = initial_noise.to(device=z_global.device, dtype=z_global.dtype)
    valid_view = part_valid.to(device=z_global.device, dtype=z_global.dtype).view(
        batch_size,
        part_count,
        1,
        1,
        1,
        1,
    )
    x = x * float(noise_scale) * valid_view
    base_grid = torch.linspace(
        0.0,
        1.0,
        int(num_steps) + 1,
        device=z_global.device,
        dtype=z_global.dtype,
    )
    t_grid = _apply_timestep_shift(base_grid, shift=float(timestep_shift))
    drop_all = part_valid.bool()
    drop_none = torch.zeros_like(drop_all)
    for step_idx in range(int(num_steps)):
        t_value = t_grid[step_idx]
        dt = t_grid[step_idx + 1] - t_value
        t = torch.full((batch_size,), t_value, device=z_global.device, dtype=z_global.dtype)
        v_null = model(
            x,
            t,
            z_global,
            cond,
            name_tokens,
            name_mask,
            anchor,
            anchor_valid,
            part_valid,
            drop_name=drop_all,
            drop_anchor=drop_all,
        )
        v_name = model(
            x,
            t,
            z_global,
            cond,
            name_tokens,
            name_mask,
            anchor,
            anchor_valid,
            part_valid,
            drop_name=drop_none,
            drop_anchor=drop_all,
        )
        v_anchor = model(
            x,
            t,
            z_global,
            cond,
            name_tokens,
            name_mask,
            anchor,
            anchor_valid,
            part_valid,
            drop_name=drop_all,
            drop_anchor=drop_none,
        )
        v = v_null + float(s_name) * (v_name - v_null) + float(s_anchor) * (v_anchor - v_null)
        x = (x + v * dt) * valid_view
    return (x / float(latent_scale)) * valid_view


@torch.no_grad()
def _eval_latent_alignment(model, dataset, device, flow_cfg: dict, eval_cfg: dict) -> Dict[str, float]:
    raw_model = model.module if isinstance(model, DDP) else model
    raw_model.eval()
    sample_indices = _eval_sample_indices(
        dataset,
        eval_cfg,
        "max_samples",
        prefer_obj_ids_key="prefer_obj_ids",
    )
    loader = DataLoader(
        Subset(dataset, sample_indices),
        batch_size=1,
        shuffle=False,
        num_workers=0,
        collate_fn=dataset.collate_fn,
    )
    latent_cos = []
    latent_rel_l2 = []
    for batch in loader:
        batch = _to_device(batch, device)
        pred = _sample_eval_part_latent_raw(
            raw_model,
            batch,
            flow_cfg,
            eval_cfg,
        )
        latent_metrics = compute_part_mmdit_latent_alignment(
            pred,
            batch["x_1_parts"],
            batch["part_valid"],
        )
        latent_cos.append(latent_metrics["latent_cos"])
        latent_rel_l2.append(latent_metrics["latent_rel_l2"])
    raw_model.train()
    return {
        "latent_cos": _safe_mean(latent_cos),
        "latent_rel_l2": _safe_mean(latent_rel_l2),
    }


@torch.no_grad()
def _eval_decode_metrics(model, dataset, device, flow_cfg: dict, eval_cfg: dict) -> Dict[str, float]:
    raw_model = model.module if isinstance(model, DDP) else model
    raw_model.eval()
    decoder = load_ss_decoder(eval_cfg["ss_decoder_ckpt"])
    sample_indices = _eval_sample_indices(
        dataset,
        eval_cfg,
        "max_samples",
        prefer_obj_ids_key="prefer_obj_ids",
    )
    loader = DataLoader(
        Subset(dataset, sample_indices),
        batch_size=1,
        shuffle=False,
        num_workers=0,
        collate_fn=dataset.collate_fn,
    )
    pred_by_object = []
    raw_by_object = []
    counts = []
    valid = []
    latent_cos = []
    latent_rel_l2 = []
    for batch in loader:
        batch = _to_device(batch, device)
        pred = _sample_eval_part_latent_raw(
            raw_model,
            batch,
            flow_cfg,
            eval_cfg,
        )
        latent_metrics = compute_part_mmdit_latent_alignment(
            pred,
            batch["x_1_parts"],
            batch["part_valid"],
        )
        latent_cos.append(latent_metrics["latent_cos"])
        latent_rel_l2.append(latent_metrics["latent_rel_l2"])
        valid_k = int(batch["part_valid"][0].sum().item())
        obj_pred = []
        obj_raw = []
        for part_idx in range(valid_k):
            obj_pred.append(
                decode_ss_latent_to_coords(
                    decoder,
                    pred[0, part_idx].detach().float().cpu(),
                    threshold=float(eval_cfg.get("decode_threshold", 0.0)),
                )
            )
            obj_raw.append(batch["raw_ind_coords"][0][part_idx].detach().cpu())
        pred_by_object.append(obj_pred)
        raw_by_object.append(obj_raw)
        counts.append(batch["part_raw_voxel_counts"][0].detach().cpu())
        valid.append(batch["part_valid"][0].detach().cpu())
    max_parts = max(int(item.numel()) for item in counts)
    count_pad = torch.zeros((len(counts), max_parts), dtype=torch.float32)
    valid_pad = torch.zeros((len(valid), max_parts), dtype=torch.bool)
    for row, (count_row, valid_row) in enumerate(zip(counts, valid)):
        part_count = int(count_row.numel())
        count_pad[row, :part_count] = count_row.float()
        valid_pad[row, :part_count] = valid_row.bool()
    metrics = compute_part_mmdit_eval_metrics(
        pred_by_object,
        raw_by_object,
        count_pad,
        valid_pad,
        size_boundaries=tuple(float(x) for x in eval_cfg.get("size_bucket_boundaries", [500.0, 3000.0])),
    )
    metrics.update(
        {
            "latent_cos": _safe_mean(latent_cos),
            "latent_rel_l2": _safe_mean(latent_rel_l2),
        }
    )
    raw_model.train()
    return metrics


def _empty_decode_metrics() -> Dict[str, float]:
    return {
        "target_iou": math.nan,
        "part_iou": math.nan,
        "recall": math.nan,
        "precision": math.nan,
        "part_recall": math.nan,
        "part_precision": math.nan,
        "small_recall": math.nan,
        "medium_recall": math.nan,
        "large_recall": math.nan,
        "target_recall": math.nan,
        "target_precision": math.nan,
        "count_error": math.nan,
        "offdiag": math.nan,
        "pred_count": math.nan,
        "gt_count": math.nan,
    }


@torch.no_grad()
def _eval_queue_metrics(
    model,
    train_dataset,
    eval_dataset,
    device,
    flow_cfg: dict,
    eval_cfg: dict,
) -> Dict[str, float]:
    decode_enabled = bool(eval_cfg.get("decode", True))
    train_cos = _eval_latent_group_cos(
        model,
        train_dataset,
        device,
        flow_cfg,
        eval_cfg,
        max_samples_key="train_max_samples",
        prefer_obj_ids_key="train_prefer_obj_ids",
    )
    val_cos = _eval_latent_group_cos(
        model,
        eval_dataset,
        device,
        flow_cfg,
        eval_cfg,
        max_samples_key="max_samples",
        prefer_obj_ids_key="prefer_obj_ids",
    )
    if decode_enabled:
        val_decode = _eval_decode_metrics(model, eval_dataset, device, flow_cfg, eval_cfg)
    else:
        val_decode = _empty_decode_metrics()
    vel_err = _eval_velocity_sweep_mean(
        model,
        eval_dataset,
        device,
        flow_cfg,
        eval_cfg,
        max_samples_key="velocity_max_samples",
        prefer_obj_ids_key="prefer_obj_ids",
    )
    return {
        "train_cos_single": train_cos["cos_single"],
        "train_cos_multi": train_cos["cos_multi"],
        "train_cos_buttons": train_cos["cos_buttons"],
        "val_cos_single": val_cos["cos_single"],
        "val_cos_multi": val_cos["cos_multi"],
        "val_cos_buttons": val_cos["cos_buttons"],
        "small_recall": val_decode["small_recall"],
        "part_iou": val_decode["part_iou"],
        "target_iou": val_decode["target_iou"],
        **vel_err,
    }


def _eval_interval_steps(eval_cfg: dict) -> int:
    if "eval_every" in eval_cfg and "fixed_every" in eval_cfg:
        eval_every = int(eval_cfg["eval_every"])
        fixed_every = int(eval_cfg["fixed_every"])
        if eval_every != fixed_every:
            raise ValueError(
                f"eval.eval_every ({eval_every}) and eval.fixed_every ({fixed_every}) disagree"
            )
        return eval_every
    return int(eval_cfg.get("eval_every", eval_cfg.get("fixed_every", 0)))


def _queue_eval_log_line(
    *,
    step: int,
    wall: float,
    loss_value: float,
    lr: float,
    grad_norm: float,
    metrics: Dict[str, float],
    t_values: list[float],
) -> str:
    vel_parts = " ".join(
        f"t{t_value:g}={_metric_str(metrics[f'vel_err_t{t_value:g}'])}"
        for t_value in t_values
    )
    return (
        f"  [EVAL step={step} wall={wall:.1f}s] "
        f"loss={loss_value:.4f} lr={lr:.2e} gradnorm={grad_norm:.4f} | "
        f"TRAIN cos_single={_metric_str(metrics['train_cos_single'])} "
        f"cos_multi={_metric_str(metrics['train_cos_multi'])} "
        f"cos_buttons={_metric_str(metrics['train_cos_buttons'])} | "
        f"VAL cos_single={_metric_str(metrics['val_cos_single'])} "
        f"cos_multi={_metric_str(metrics['val_cos_multi'])} "
        f"cos_buttons={_metric_str(metrics['val_cos_buttons'])} "
        f"small_recall={_metric_str(metrics['small_recall'])} "
        f"part_iou={_metric_str(metrics['part_iou'])} "
        f"target_iou={_metric_str(metrics['target_iou'])} | "
        f"VEL_ERR {vel_parts}"
    )


@torch.no_grad()
def _eval_fixed_metrics(model, train_dataset, eval_dataset, device, flow_cfg: dict, eval_cfg: dict) -> Dict[str, float]:
    decode_enabled = bool(eval_cfg.get("decode", True))
    if decode_enabled:
        val_metrics = _eval_decode_metrics(model, eval_dataset, device, flow_cfg, eval_cfg)
    else:
        val_metrics = _empty_decode_metrics()
        val_metrics.update(_eval_latent_alignment(model, eval_dataset, device, flow_cfg, eval_cfg))

    train_eval_cfg = dict(eval_cfg)
    train_eval_cfg["max_samples"] = int(eval_cfg.get("train_max_samples", eval_cfg.get("max_samples", 4)))
    train_latent = _eval_latent_alignment(model, train_dataset, device, flow_cfg, train_eval_cfg)
    val_metrics["train_latent_cos"] = train_latent["latent_cos"]
    val_metrics["train_latent_rel_l2"] = train_latent["latent_rel_l2"]
    return val_metrics


def train(config, dataset_cls: type = PartMMDiTDataset) -> None:
    cfg = config
    rank, local_rank, world_size = setup_ddp()
    is_distributed = world_size > 1
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    seed = int(getattr(cfg.training, "seed", 42))
    _setup_rng(seed + rank)

    data_cfg = _cfg_dict(cfg.data)
    model_cfg = _cfg_dict(cfg.model)
    flow_cfg = _cfg_dict(cfg.flow)
    loss_cfg = _cfg_dict(cfg.loss) if "loss" in cfg else {}
    training_cfg = _cfg_dict(cfg.training)
    eval_cfg = _cfg_dict(cfg.eval) if "eval" in cfg else {}

    dataset = dataset_cls(data_cfg)
    eval_dataset = _build_eval_dataset(
        eval_cfg,
        dataset_cls=dataset_cls,
        train_dataset=dataset,
    )
    sampler = DistributedSampler(dataset, shuffle=True) if is_distributed else None
    loader = DataLoader(
        dataset,
        batch_size=int(training_cfg.get("batch_size", 1)),
        shuffle=sampler is None,
        sampler=sampler,
        num_workers=int(training_cfg.get("num_workers", 0)),
        pin_memory=torch.cuda.is_available(),
        collate_fn=dataset.collate_fn,
        drop_last=False,
    )

    model = PartMMDiTModel(**model_cfg).to(device)
    if is_distributed:
        model = _wrap_ddp_model(model, local_rank)
    raw_model = model.module if isinstance(model, DDP) else model
    _assert_no_native_fp16_trainable_params(model)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training_cfg.get("lr", 1.0e-4)),
        weight_decay=float(training_cfg.get("weight_decay", 0.01)),
    )
    max_steps = int(training_cfg.get("max_steps", 3000))
    warmup_steps = max(1, int(training_cfg.get("warmup_steps", 50)))

    def lr_lambda(step):
        step = max(1, step)
        if step <= warmup_steps:
            return step / float(warmup_steps)
        progress = min(1.0, (step - warmup_steps) / float(max(1, max_steps - warmup_steps)))
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
    start_step = _load_resume_checkpoint(model, optimizer, scheduler, training_cfg, device, rank)
    loss_kwargs = _loss_kwargs(flow_cfg, loss_cfg)

    output_dir = Path(training_cfg.get("output_dir", "runs/part_mmdit"))
    if rank == 0:
        output_dir.mkdir(parents=True, exist_ok=True)
        with (output_dir / "config.json").open("w", encoding="utf-8") as config_file:
            json.dump(config_to_dict(cfg), config_file, indent=2)
        n_params = sum(param.numel() for param in raw_model.parameters())
        total_parts = sum(len(sample["parts"]) for sample in dataset.samples)
        eval_total_parts = sum(len(sample["parts"]) for sample in eval_dataset.samples)
        print("\n[PartMMDiT Train]")
        print(f"  device={device} world_size={world_size}")
        print(
            f"  train dataset: {len(dataset)} object samples / {total_parts} target parts "
            f"| batch/gpu={training_cfg.get('batch_size', 1)}"
        )
        if eval_dataset is dataset:
            print("  eval dataset: train dataset (no eval.data configured)")
        else:
            eval_obj_ids = sorted(_obj_id_set(eval_dataset))
            print(
                f"  val dataset: {len(eval_dataset)} object samples / "
                f"{eval_total_parts} target parts / {len(eval_obj_ids)} obj_ids"
            )
            print(f"  val obj_ids: {eval_obj_ids}")
            with (output_dir / "val_obj_ids.json").open("w", encoding="utf-8") as val_file:
                json.dump(eval_obj_ids, val_file, indent=2)
        print(f"  model: PartMMDiTModel ({n_params:,} params)")
        print(
            "  conditioning: CLIP name tokens + anchor tokens via dual-stream joint attention; "
            "global/image tokens are shared cross-attn memory"
        )
        print(f"  cross_part_layers={model_cfg.get('cross_part_layers', [3, 6, 9])}")
        timestep_shift = _dynamic_timestep_shift(raw_model, flow_cfg)
        print(
            "  flow: "
            f"t_schedule={flow_cfg.get('t_schedule', 'uniform')} "
            f"logit_mean={float(flow_cfg.get('t_logit_normal_mean', 0.0)):.2f} "
            f"logit_std={float(flow_cfg.get('t_logit_normal_std', 1.0)):.2f} "
            f"num_steps={int(flow_cfg.get('num_steps', 20))} "
            f"timestep_shift={timestep_shift:.4f}"
        )
        print(f"  loss: {loss_cfg}")
        print(f"  starting training: {max_steps} steps")
        if start_step > 0:
            print(f"  resume: continuing from step {start_step}")

    use_fp16 = bool(training_cfg.get("fp16", False)) and torch.cuda.is_available()
    scaler = torch.cuda.amp.GradScaler(enabled=use_fp16) if torch.cuda.is_available() else None
    grad_clip = float(training_cfg.get("grad_clip", 1.0))
    log_every = int(training_cfg.get("log_every", 10))
    ckpt_every = int(training_cfg.get("checkpoint_every", 500))
    fixed_every = _eval_interval_steps(eval_cfg)
    metrics_markdown_path = eval_cfg.get("metrics_markdown_path")
    smoke_metrics_enabled = bool(eval_cfg.get("smoke_metrics", False))
    queue_metrics_enabled = bool(eval_cfg.get("queue_metrics", False))
    bucket_metrics_enabled = bool(eval_cfg.get("bucket_metrics", False))
    enabled_eval_modes = [
        name
        for name, enabled in (
            ("smoke_metrics", smoke_metrics_enabled),
            ("queue_metrics", queue_metrics_enabled),
            ("bucket_metrics", bucket_metrics_enabled),
        )
        if enabled
    ]
    if len(enabled_eval_modes) > 1:
        raise ValueError(f"eval metric modes are mutually exclusive: {enabled_eval_modes}")
    start_time = time.time()
    last_grad_norm = float("nan")

    model.train()
    epoch = 0
    if sampler is not None:
        sampler.set_epoch(epoch)
    data_iter = iter(loader)
    for step in range(start_step + 1, max_steps + 1):
        try:
            batch = next(data_iter)
        except StopIteration:
            epoch += 1
            if sampler is not None:
                sampler.set_epoch(epoch)
            data_iter = iter(loader)
            batch = next(data_iter)

        batch = _to_device(batch, device)
        optimizer.zero_grad(set_to_none=True)
        t = sample_flow_timesteps(
            batch["x_1_parts"].shape[0],
            device=device,
            dtype=batch["x_1_parts"].dtype,
            t_min=float(flow_cfg.get("t_min", 0.0)),
            t_max=float(flow_cfg.get("t_max", 1.0)),
            t_schedule=str(flow_cfg.get("t_schedule", "uniform")),
            t_logit_normal_mean=float(flow_cfg.get("t_logit_normal_mean", 0.0)),
            t_logit_normal_std=float(flow_cfg.get("t_logit_normal_std", 1.0)),
        )
        t = _apply_timestep_shift(
            t,
            shift=_dynamic_timestep_shift(raw_model, flow_cfg),
            t_min=float(flow_cfg.get("t_min", 0.0)),
            t_max=float(flow_cfg.get("t_max", 1.0)),
        )

        optimizer_step_ran = False
        if torch.cuda.is_available():
            with torch.cuda.amp.autocast(enabled=use_fp16, cache_enabled=False):
                loss = rectified_flow_loss(
                    model,
                    batch["x_1_parts"],
                    t,
                    z_global=batch["z_global"],
                    cond=batch["cond"],
                    name_tokens=batch["name_tokens"],
                    name_mask=batch["name_mask"],
                    anchor=batch["anchor"],
                    anchor_valid=batch["anchor_valid"],
                    part_valid=batch["part_valid"],
                    part_raw_voxel_counts=batch["part_raw_voxel_counts"],
                    part_fg_mask=batch["part_fg_mask"],
                    **loss_kwargs,
            )
            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite PartMMDiT loss at step {step}: {float(loss.item())}")
            if scaler is not None and scaler.is_enabled():
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                last_grad_norm = float(grad_norm.item())
                old_scale = float(scaler.get_scale())
                if not torch.isfinite(grad_norm) and rank == 0:
                    print(
                        f"  [AMP_OVERFLOW step={step}] gradnorm={last_grad_norm} "
                        f"loss={float(loss.item()):.4f} grad_scale={old_scale:.1f}",
                        flush=True,
                    )
                scaler.step(optimizer)
                scaler.update()
                new_scale = float(scaler.get_scale())
                optimizer_step_ran = new_scale >= old_scale
                if not optimizer_step_ran and rank == 0:
                    print(
                        f"  [AMP_OVERFLOW step={step}] optimizer step skipped; "
                        f"grad_scale {old_scale:.1f}->{new_scale:.1f}",
                        flush=True,
                    )
            else:
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                if not torch.isfinite(grad_norm):
                    raise RuntimeError(f"non-finite PartMMDiT gradnorm at step {step}: {float(grad_norm.item())}")
                last_grad_norm = float(grad_norm.item())
                optimizer.step()
                optimizer_step_ran = True
        else:
            loss = rectified_flow_loss(
                model,
                batch["x_1_parts"],
                t,
                z_global=batch["z_global"],
                cond=batch["cond"],
                name_tokens=batch["name_tokens"],
                name_mask=batch["name_mask"],
                anchor=batch["anchor"],
                anchor_valid=batch["anchor_valid"],
                part_valid=batch["part_valid"],
                part_raw_voxel_counts=batch["part_raw_voxel_counts"],
                part_fg_mask=batch["part_fg_mask"],
                **loss_kwargs,
            )
            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite PartMMDiT loss at step {step}: {float(loss.item())}")
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            if not torch.isfinite(grad_norm):
                raise RuntimeError(f"non-finite PartMMDiT gradnorm at step {step}: {float(grad_norm.item())}")
            last_grad_norm = float(grad_norm.item())
            optimizer.step()
            optimizer_step_ran = True
        if optimizer_step_ran:
            scheduler.step()

        if rank == 0 and (step == 1 or step % log_every == 0):
            valid_parts = int(batch["part_valid"].sum().item())
            print(
                f"  step {step:6d}/{max_steps} | loss={float(loss.item()):.4f} "
                f"parts={valid_parts} t_mean={float(t.mean().item()):.2f} "
                f"lr={scheduler.get_last_lr()[0]:.2e}"
            )

        if ckpt_every > 0 and step % ckpt_every == 0:
            if dist.is_initialized():
                dist.barrier()
            _save_ckpt(step, model, optimizer, scheduler, cfg, output_dir, rank)
            if dist.is_initialized():
                dist.barrier()

        if fixed_every > 0 and step % fixed_every == 0:
            if dist.is_initialized():
                dist.barrier()
            metrics = None
            if rank == 0:
                if smoke_metrics_enabled:
                    metrics = _eval_smoke_curve(model, eval_dataset, device, flow_cfg, eval_cfg)
                elif queue_metrics_enabled:
                    metrics = _eval_queue_metrics(model, dataset, eval_dataset, device, flow_cfg, eval_cfg)
                elif bucket_metrics_enabled:
                    metrics = _eval_bucket_metrics(model, dataset, eval_dataset, device, flow_cfg, eval_cfg)
                else:
                    metrics = _eval_fixed_metrics(model, dataset, eval_dataset, device, flow_cfg, eval_cfg)
            if rank == 0:
                if smoke_metrics_enabled:
                    wall = time.time() - start_time
                    print(
                        f"  [EVAL step={step} wall={wall:.1f}s] "
                        f"loss={float(loss.item()):.4f} "
                        f"lr={scheduler.get_last_lr()[0]:.2e} "
                        f"gradnorm={last_grad_norm:.4f} | "
                        f"cos_single={metrics['cos_single']:.4f} "
                        f"cos_multi={metrics['cos_multi']:.4f} "
                        f"cos_buttons={metrics['cos_buttons']:.4f} | "
                        f"VEL_ERR button_0 "
                        f"t0.02={metrics['vel_err_button_0_t0.02']:.4f} "
                        f"t0.1={metrics['vel_err_button_0_t0.1']:.4f} "
                        f"t0.3={metrics['vel_err_button_0_t0.3']:.4f} "
                        f"t0.5={metrics['vel_err_button_0_t0.5']:.4f} "
                        f"t0.9={metrics['vel_err_button_0_t0.9']:.4f} | "
                        f"VEL_ERR lid_0 "
                        f"t0.02={metrics['vel_err_lid_0_t0.02']:.4f} "
                        f"t0.1={metrics['vel_err_lid_0_t0.1']:.4f} "
                        f"t0.3={metrics['vel_err_lid_0_t0.3']:.4f} "
                        f"t0.5={metrics['vel_err_lid_0_t0.5']:.4f} "
                        f"t0.9={metrics['vel_err_lid_0_t0.9']:.4f}"
                    )
                elif queue_metrics_enabled:
                    print(
                        _queue_eval_log_line(
                            step=step,
                            wall=time.time() - start_time,
                            loss_value=float(loss.item()),
                            lr=scheduler.get_last_lr()[0],
                            grad_norm=last_grad_norm,
                            metrics=metrics,
                            t_values=[
                                float(x)
                                for x in eval_cfg.get(
                                    "velocity_t_values",
                                    [0.02, 0.1, 0.3, 0.5, 0.9],
                                )
                            ],
                        )
                    )
                elif bucket_metrics_enabled:
                    for line in _bucket_eval_log_lines(
                        step=step,
                        wall=time.time() - start_time,
                        loss_value=float(loss.item()),
                        lr=scheduler.get_last_lr()[0],
                        grad_norm=last_grad_norm,
                        metrics=metrics,
                    ):
                        print(line)
                else:
                    print(
                        f"  [EVAL_FIXED @ {step}] "
                        f"train_latent_cos={metrics['train_latent_cos']:.4f} "
                        f"train_latent_rel_l2={metrics['train_latent_rel_l2']:.4f} "
                        f"val_latent_cos={metrics['latent_cos']:.4f} "
                        f"val_latent_rel_l2={metrics['latent_rel_l2']:.4f} "
                        f"target_iou={metrics['target_iou']:.4f} "
                        f"part_iou={metrics['part_iou']:.4f} "
                        f"recall={metrics['recall']:.4f} "
                        f"precision={metrics['precision']:.4f} "
                        f"small_recall={metrics['small_recall']:.4f} "
                        f"large_recall={metrics['large_recall']:.4f} "
                        f"count_error={metrics['count_error']:.4f} "
                        f"offdiag={metrics['offdiag']:.4f}"
                    )
                if metrics_markdown_path and not smoke_metrics_enabled and not queue_metrics_enabled:
                    _append_eval_metrics_markdown(
                        metrics_markdown_path,
                        step=step,
                        metrics=metrics,
                    )
            if dist.is_initialized():
                dist.barrier()

    if dist.is_initialized():
        dist.barrier()
    _save_ckpt(max_steps, model, optimizer, scheduler, cfg, output_dir, rank)
    if dist.is_initialized():
        dist.barrier()


def main():
    import argparse

    parser = argparse.ArgumentParser(description="PartMMDiT trainer")
    parser.add_argument("--config", required=True)
    parser.add_argument("overrides", nargs="*")
    args = parser.parse_args()
    cfg = load_config(args.config, overrides=args.overrides)
    train(cfg)


if __name__ == "__main__":
    main()
