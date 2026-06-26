"""Trainer entry for part_ss_latent_flow."""

from __future__ import annotations

import json
import math
import os
import random
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Subset
from torch.utils.data.distributed import DistributedSampler

from trellis.datasets.arts.part_ss_latent_flow import PartSSLatentFlowDataset
from trellis.models.part_flow import PartSSLatentFlowModel
from trellis.trainers.arts.part_ss_latent_flow_eval import (
    _print_progress,
    compute_decode_metrics,
    decode_ss_latent_to_coords,
    decode_ss_latent_to_coords_with_stats,
    load_ss_decoder,
    part_assignment_iou_matrix,
    summarize_assignment_matrix,
    summarize_bucketed_part_metrics,
    write_part_ss_inspection,
    write_part_ss_inspection_sample,
)
from trellis.trainers.arts.part_ss_latent_flow_losses import (
    PartSSLatentRFLoss,
    sample_part_ss_latent,
)
from trellis.utils.arts.config_utils import config_to_dict, load_config
from trellis.utils.arts.ddp_utils import setup_ddp


def _ddp_active() -> bool:
    return dist.is_initialized() and dist.get_world_size() > 1


def _run_rank0_only_work(label: str, rank: int, work):
    """Run long rank0-only work while keeping DDP ranks aligned."""
    if _ddp_active():
        dist.barrier()
    result = work() if rank == 0 else None
    if _ddp_active():
        dist.barrier()
    return result


def _wrap_ddp_model(model, local_rank: int):
    return DDP(
        model,
        device_ids=[local_rank],
        output_device=local_rank,
        broadcast_buffers=False,
    )


def _setup_rng(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _to_device(batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    out = {}
    for key, value in batch.items():
        out[key] = value.to(device, non_blocking=True) if isinstance(value, torch.Tensor) else value
    return out


def _cfg_dict(cfg) -> dict:
    return config_to_dict(cfg) if not isinstance(cfg, dict) else dict(cfg)


def _resolve_object_weight_k_ref(dataset, loss_cfg: Dict[str, Any], *, rank: int, is_distributed: bool) -> float | None:
    mode = str(loss_cfg.get("object_weight_mode", "none"))
    if mode == "none":
        return None
    if mode != "sqrt_k":
        raise ValueError(f"unknown object_weight_mode={mode!r}")

    source = str(loss_cfg.get("object_weight_k_ref_source", "dataset_median"))
    if source == "fixed":
        value = loss_cfg.get("object_weight_k_ref")
        if value is None or float(value) <= 0:
            raise ValueError("object_weight_k_ref_source=fixed requires object_weight_k_ref > 0")
        return float(value)
    if source != "dataset_median":
        raise ValueError(f"unknown object_weight_k_ref_source={source!r}")

    counts = sorted(float(len(sample["parts"])) for sample in dataset.samples)
    if not counts:
        raise ValueError("cannot resolve object_weight_k_ref from empty dataset")
    mid = len(counts) // 2
    value = counts[mid] if len(counts) % 2 else 0.5 * (counts[mid - 1] + counts[mid])
    if is_distributed and dist.is_available() and dist.is_initialized():
        payload = [float(value)]
        dist.broadcast_object_list(payload, src=0)
        value = float(payload[0])
    return float(value)


def _compute_latent_stats(dataset, latent_channels: int, max_samples: int) -> tuple[list[float], list[float]]:
    """Compute per-channel latent mean/std over valid parts of a dataset subset.

    Iterates up to ``max_samples`` objects, accumulating per-channel sum and
    sum-of-squares over every voxel of every valid part latent. Returns two
    length-``latent_channels`` lists (mean, std). Raises if no voxels are seen
    so the caller cannot silently fall back to garbage statistics.
    """
    count = len(dataset) if max_samples <= 0 else min(int(max_samples), len(dataset))
    if count <= 0:
        raise ValueError("cannot compute latent stats from an empty dataset")
    sum_c = torch.zeros(latent_channels, dtype=torch.float64)
    sum_sq_c = torch.zeros(latent_channels, dtype=torch.float64)
    total_voxels = 0
    for idx in range(count):
        sample = dataset[idx]
        x_1_parts = sample["x_1_parts"]  # [K, C, R, R, R]
        part_valid = sample["part_valid"].bool()  # [K]
        if x_1_parts.shape[1] != latent_channels:
            raise ValueError(
                f"latent channel mismatch: dataset latent has {x_1_parts.shape[1]} channels, "
                f"model.latent_channels={latent_channels}"
            )
        valid = x_1_parts[part_valid].double()  # [Kv, C, R, R, R]
        if valid.numel() == 0:
            continue
        # Reduce over part + spatial dims, keep channel dim.
        flat = valid.permute(1, 0, 2, 3, 4).reshape(latent_channels, -1)
        sum_c += flat.sum(dim=1)
        sum_sq_c += (flat * flat).sum(dim=1)
        total_voxels += flat.shape[1]
    if total_voxels == 0:
        raise ValueError("latent stats sample contained no valid part voxels")
    mean = sum_c / total_voxels
    var = sum_sq_c / total_voxels - mean * mean
    var = torch.clamp(var, min=1.0e-12)
    std = torch.sqrt(var)
    return mean.float().tolist(), std.float().tolist()


def _resolve_latent_stats(
    dataset,
    *,
    flow_cfg: Dict[str, Any],
    model_cfg: Dict[str, Any],
    rank: int,
    is_distributed: bool,
) -> None:
    """Resolve per-channel latent_mean/std for per_channel normalization in place.

    Mutates ``flow_cfg['latent_mean']`` / ``flow_cfg['latent_std']`` so the loss
    build and every eval sampler call (all of which read ``flow_cfg``) pick up the
    same statistics. Resolution order:
      1. Both already provided in config -> validate + keep.
      2. ``flow.latent_stats_path`` JSON ({"latent_mean": [...], "latent_std": [...]}).
      3. Compute over a dataset subset on rank0, broadcast to all ranks.
    No-op when latent_norm_mode != 'per_channel'.
    """
    mode = str(flow_cfg.get("latent_norm_mode", "scalar"))
    if mode != "per_channel":
        return
    latent_channels = int(model_cfg.get("latent_channels", 8))

    def _validate(name: str, values) -> list[float]:
        seq = list(values)
        if len(seq) != latent_channels:
            raise ValueError(
                f"{name} must have {latent_channels} elements (one per latent channel), got {len(seq)}"
            )
        return [float(v) for v in seq]

    mean = flow_cfg.get("latent_mean")
    std = flow_cfg.get("latent_std")
    if mean is not None and std is not None:
        flow_cfg["latent_mean"] = _validate("latent_mean", mean)
        flow_cfg["latent_std"] = _validate("latent_std", std)
        if rank == 0:
            print(f"  [LATENT_STATS] using config-provided per-channel stats ({latent_channels} channels)")
        return

    stats_path = flow_cfg.get("latent_stats_path")
    if stats_path is not None:
        path = Path(stats_path)
        if not path.is_file():
            raise FileNotFoundError(f"flow.latent_stats_path not found: {path}")
        with path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        flow_cfg["latent_mean"] = _validate("latent_mean", payload["latent_mean"])
        flow_cfg["latent_std"] = _validate("latent_std", payload["latent_std"])
        if rank == 0:
            print(f"  [LATENT_STATS] loaded per-channel stats from {path}")
        return

    max_samples = int(flow_cfg.get("latent_stats_max_samples", 256))
    if rank == 0:
        print(f"  [LATENT_STATS] computing per-channel stats over up to {max_samples} objects ...")
        mean_list, std_list = _compute_latent_stats(dataset, latent_channels, max_samples)
        print(f"  [LATENT_STATS] mean={['%.4f' % v for v in mean_list]}")
        print(f"  [LATENT_STATS] std ={['%.4f' % v for v in std_list]}")
    else:
        mean_list, std_list = None, None
    if is_distributed and dist.is_available() and dist.is_initialized():
        payload = [mean_list, std_list]
        dist.broadcast_object_list(payload, src=0)
        mean_list, std_list = payload[0], payload[1]
    flow_cfg["latent_mean"] = _validate("latent_mean", mean_list)
    flow_cfg["latent_std"] = _validate("latent_std", std_list)


def _build_part_ss_latent_rf_loss(
    *,
    loss_cfg: Dict[str, Any],
    flow_cfg: Dict[str, Any],
    resolved_object_weight_k_ref: float | None,
    model_cfg: Dict[str, Any] | None = None,
):
    model_cfg = model_cfg or {}
    criterion_kwargs = dict(
        t_min=float(flow_cfg.get("t_min", 0.0)),
        t_max=float(flow_cfg.get("t_max", 1.0)),
        noise_scale=float(flow_cfg.get("noise_scale", 1.0)),
        latent_scale=float(flow_cfg.get("latent_scale", 1.0)),
        part_weight_mode=str(loss_cfg.get("part_weight_mode", "none")),
        part_weight_ref_mode=str(loss_cfg.get("part_weight_ref_mode", "median")),
        part_weight_alpha=float(loss_cfg.get("part_weight_alpha", 0.5)),
        part_weight_min=float(loss_cfg.get("part_weight_min", 0.5)),
        part_weight_max=float(loss_cfg.get("part_weight_max", 3.0)),
        normalize_part_weights_per_object=bool(loss_cfg.get("normalize_part_weights_per_object", True)),
        size_bucket_boundaries=tuple(float(x) for x in loss_cfg.get("size_bucket_boundaries", [500.0, 3000.0])),
        object_balanced=bool(loss_cfg.get("object_balanced", False)),
        object_weight_mode=str(loss_cfg.get("object_weight_mode", "none")),
        object_weight_k_ref=resolved_object_weight_k_ref,
        object_weight_min=float(loss_cfg.get("object_weight_min", 0.75)),
        object_weight_max=float(loss_cfg.get("object_weight_max", 2.0)),
        relative_endpoint_weight=float(loss_cfg.get("relative_endpoint_weight", 0.0)),
        relative_endpoint_eps=float(loss_cfg.get("relative_endpoint_eps", 1.0e-6)),
        # Fix 4: DeltaFM velocity-contrastive identity term (default on).
        velocity_contrastive_weight=float(loss_cfg.get("velocity_contrastive_weight", 0.05)),
        velocity_contrastive_lambda=float(loss_cfg.get("velocity_contrastive_lambda", 0.05)),
        # Legacy endpoint-based identity contrastive (ablation only).
        identity_contrastive_weight=float(loss_cfg.get("identity_contrastive_weight", 0.0)),
        identity_contrastive_temperature=float(loss_cfg.get("identity_contrastive_temperature", 0.1)),
        identity_contrastive_eps=float(loss_cfg.get("identity_contrastive_eps", 1.0e-6)),
        # Fix 5: continuous-time sampling schedule.
        t_schedule=str(flow_cfg.get("t_schedule", "logit_normal")),
        t_logit_normal_mean=float(flow_cfg.get("t_logit_normal_mean", 0.0)),
        t_logit_normal_std=float(flow_cfg.get("t_logit_normal_std", 1.0)),
        # Fix 6: per-channel latent normalization.
        latent_norm_mode=str(flow_cfg.get("latent_norm_mode", "scalar")),
        latent_channels=int(model_cfg.get("latent_channels", 8)),
        latent_mean=flow_cfg.get("latent_mean"),
        latent_std=flow_cfg.get("latent_std"),
        # Fix 3: per-object slot<->part shuffle.
        part_shuffle=bool(loss_cfg.get("part_shuffle", False)),
        # Model-side self-conditioning + CFG training support.
        self_conditioning=bool(model_cfg.get("self_conditioning", False)),
        self_conditioning_prob=float(loss_cfg.get("self_conditioning_prob", 0.5)),
        cfg_dropout_prob=float(loss_cfg.get("cfg_dropout_prob", 0.0)),
        foreground_weight=dict(loss_cfg.get("foreground_weight", {})),
    )
    return PartSSLatentRFLoss(**criterion_kwargs)


def _assert_no_native_fp16_trainable_params(model) -> None:
    raw_model = model.module if isinstance(model, DDP) else model
    fp16_params = [
        name for name, param in raw_model.named_parameters()
        if param.requires_grad and param.dtype == torch.float16
    ]
    if fp16_params:
        preview = ", ".join(fp16_params[:5])
        if len(fp16_params) > 5:
            preview += f", ... ({len(fp16_params)} total)"
        raise RuntimeError(
            "part_ss_latent_flow trainer expects FP32 trainable parameters. "
            "Use training.fp16=true for AMP autocast, but keep model.use_fp16=false. "
            f"Found native FP16 trainable parameters: {preview}"
        )


def _safe_ratio(value: float, baseline: float) -> float:
    if baseline == 0.0:
        return 0.0 if value == 0.0 else math.inf
    return float(value / baseline)


def _object_id_filter_indices(dataset, object_ids: str | list[str] | tuple[str, ...] | None) -> list[int] | None:
    if object_ids is None:
        return None
    if isinstance(object_ids, str):
        wanted = [item.strip() for item in object_ids.split(",") if item.strip()]
    else:
        wanted = [str(item).strip() for item in object_ids if str(item).strip()]
    if not wanted:
        return None
    wanted_set = set(wanted)
    samples = getattr(dataset, "samples", None)
    if samples is None:
        raise ValueError("object_id filtering requires dataset.samples")
    indices = [
        idx
        for idx, sample in enumerate(samples)
        if str(sample.get("obj_id", sample.get("object_id", ""))) in wanted_set
    ]
    if not indices:
        raise ValueError(f"object_id filter matched 0 samples: {wanted}")
    return indices


def _sample_indices_for_eval(
    dataset_len: int,
    max_samples: int | None,
    sample_mode: str = "first",
    candidate_indices: list[int] | None = None,
) -> list[int]:
    base_indices = list(range(dataset_len)) if candidate_indices is None else list(candidate_indices)
    if not base_indices:
        return []
    if max_samples is None or int(max_samples) >= len(base_indices):
        return base_indices
    count = max(0, int(max_samples))
    if count == 0:
        return []
    if sample_mode == "first":
        return base_indices[:count]
    if sample_mode == "spread":
        if count == 1:
            return [base_indices[0]]
        return [base_indices[round(i * (len(base_indices) - 1) / (count - 1))] for i in range(count)]
    raise ValueError(f"unknown eval sample_mode={sample_mode!r}; expected 'first' or 'spread'")


def _latent_metric_values(pred: torch.Tensor, target: torch.Tensor) -> Dict[str, float]:
    if pred.numel() == 0:
        return {
            "latent_mse": math.nan,
            "latent_l1": math.nan,
            "latent_cos": math.nan,
            "zero_mse": math.nan,
            "zero_l1": math.nan,
            "mse_vs_zero": math.nan,
            "l1_vs_zero": math.nan,
        }
    latent_mse = torch.mean((pred - target) ** 2).item()
    latent_l1 = torch.mean(torch.abs(pred - target)).item()
    pred_flat = pred.detach().float().reshape(pred.shape[0], -1)
    target_flat = target.detach().float().reshape(target.shape[0], -1)
    dot = (pred_flat * target_flat).sum(dim=1)
    denom = pred_flat.norm(dim=1) * target_flat.norm(dim=1)
    latent_cos = (dot / denom.clamp_min(1.0e-8)).mean().item()
    zero_mse = torch.mean(target ** 2).item()
    zero_l1 = torch.mean(torch.abs(target)).item()
    return {
        "latent_mse": float(latent_mse),
        "latent_l1": float(latent_l1),
        "latent_cos": float(latent_cos),
        "zero_mse": float(zero_mse),
        "zero_l1": float(zero_l1),
        "mse_vs_zero": _safe_ratio(float(latent_mse), float(zero_mse)),
        "l1_vs_zero": _safe_ratio(float(latent_l1), float(zero_l1)),
    }


def _save_ckpt(step, model, optimizer, scheduler, cfg, output_dir: Path, rank: int) -> None:
    if rank != 0:
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = output_dir / "ckpts"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    raw_model = model.module if isinstance(model, DDP) else model
    path = ckpt_dir / f"step_{int(step)}.pt"
    torch.save({
        "step": int(step),
        "model": raw_model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "config": config_to_dict(cfg),
    }, path)
    print(f"  [CKPT] saved: {path}")


def _resolve_resume_checkpoint(load_dir: str | os.PathLike[str], resume_step: int) -> Path:
    load_dir = Path(load_dir)
    filename = f"step_{int(resume_step)}.pt"
    candidates = (
        [load_dir / filename, load_dir.parent / "ckpts" / filename]
        if load_dir.name == "ckpts"
        else [load_dir / "ckpts" / filename, load_dir / filename]
    )
    for path in candidates:
        if path.is_file():
            return path
    return candidates[0]


def _compatible_model_state_dict(model, state_dict: dict) -> dict:
    """Drop opt-in camera-pose weights when loading legacy/no-pose configs."""
    model_keys = set((model.module if isinstance(model, DDP) else model).state_dict().keys())
    filtered = dict(state_dict)
    for key in ("backbone.view_pose_proj.weight", "backbone.view_pose_proj.bias"):
        if key in filtered and key not in model_keys:
            filtered.pop(key)
    return filtered


def _load_resume_checkpoint(model, optimizer, scheduler, training_cfg: dict, device: torch.device, rank: int) -> int:
    load_dir = training_cfg.get("load_dir")
    resume_step = training_cfg.get("resume_step")
    if load_dir is None and resume_step is None:
        return 0
    if load_dir is None or resume_step is None:
        raise ValueError("resume requires both training.load_dir and training.resume_step")

    ckpt_path = _resolve_resume_checkpoint(load_dir, int(resume_step))
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"checkpoint not found: {ckpt_path}")

    raw_model = model.module if isinstance(model, DDP) else model
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    raw_model.load_state_dict(_compatible_model_state_dict(raw_model, ckpt["model"]))
    if bool(training_cfg.get("resume_weights_only", False)):
        if rank == 0:
            print(f"  [RESUME] loaded model weights only: {ckpt_path} (optimizer/scheduler reset)")
        return 0
    optimizer.load_state_dict(ckpt["optimizer"])
    if scheduler is not None and ckpt.get("scheduler") is not None:
        scheduler.load_state_dict(ckpt["scheduler"])
    start_step = int(ckpt.get("step", resume_step))
    if start_step != int(resume_step):
        raise ValueError(
            f"checkpoint step mismatch: requested {int(resume_step)}, "
            f"file contains {start_step}"
        )
    if rank == 0:
        print(f"  [RESUME] loaded checkpoint: {ckpt_path} (step {start_step})")
    return start_step


@torch.no_grad()
def _eval_latent(
    model,
    dataset,
    device,
    flow_cfg,
    max_samples: int | None = None,
    progress_prefix: str | None = None,
    sample_mode: str = "first",
    object_ids: str | list[str] | tuple[str, ...] | None = None,
) -> Dict[str, float]:
    raw_model = model.module if isinstance(model, DDP) else model
    raw_model.eval()
    candidate_indices = _object_id_filter_indices(dataset, object_ids)
    sample_indices = _sample_indices_for_eval(len(dataset), max_samples, sample_mode, candidate_indices)
    eval_dataset = Subset(dataset, sample_indices)
    loader = DataLoader(eval_dataset, batch_size=1, shuffle=False, num_workers=0, collate_fn=dataset.collate_fn)
    total_samples = len(sample_indices)
    _print_progress(progress_prefix, f"latent eval progress: 0/{total_samples} samples")
    total_objects = 0
    total_parts = 0
    mse_sum = 0.0
    l1_sum = 0.0
    zero_mse_sum = 0.0
    zero_l1_sum = 0.0
    latent_cos_sum = 0.0
    for idx, batch in enumerate(loader):
        source_idx = sample_indices[idx]
        sample_meta = dataset.samples[source_idx]
        batch = _to_device(batch, device)
        obj_id = batch["obj_id"][0]
        valid_k = int(batch["part_valid"][0].sum().item())
        _print_progress(
            progress_prefix,
            f"latent sample {idx + 1}/{total_samples}: dataset_idx={source_idx} obj={obj_id} parts={valid_k}",
        )
        pred = sample_part_ss_latent(
            raw_model,
            z_global=batch["z_global"],
            cond=batch["cond"],
            mask_token_labels=batch["mask_token_labels"],
            part_valid=batch["part_valid"],
            target_slots=batch["target_slots"],
            part_token_weights=batch.get("part_token_weights"),
            num_steps=int(flow_cfg.get("num_steps", 20)),
            noise_scale=float(flow_cfg.get("noise_scale", 1.0)),
            latent_scale=float(flow_cfg.get("latent_scale", 1.0)),
            latent_norm_mode=str(flow_cfg.get("latent_norm_mode", "scalar")),
            latent_mean=flow_cfg.get("latent_mean"),
            latent_std=flow_cfg.get("latent_std"),
            self_conditioning=bool(getattr(raw_model, "self_conditioning", False)),
            cfg_scale=float(flow_cfg.get("cfg_scale", 1.0)),
        )
        x_1 = batch["x_1_parts"]
        valid = batch["part_valid"].bool()
        part_count = int(valid.sum().item())
        metrics = _latent_metric_values(pred[valid], x_1[valid])
        mse_sum += metrics["latent_mse"] * part_count
        l1_sum += metrics["latent_l1"] * part_count
        latent_cos_sum += metrics["latent_cos"] * part_count
        zero_mse_sum += metrics["zero_mse"] * part_count
        zero_l1_sum += metrics["zero_l1"] * part_count
        total_parts += part_count
        total_objects += int(batch["z_global"].shape[0])
        _print_progress(
            progress_prefix,
            f"finished latent sample {idx + 1}/{total_samples}: dataset_idx={source_idx} obj={obj_id}",
        )
    raw_model.train()
    if total_parts == 0:
        return {
            "samples": 0,
            "objects": 0,
            "parts": 0,
            "latent_mse": math.nan,
            "latent_l1": math.nan,
            "latent_cos": math.nan,
            "zero_mse": math.nan,
            "zero_l1": math.nan,
            "mse_vs_zero": math.nan,
            "l1_vs_zero": math.nan,
        }
    latent_mse = mse_sum / total_parts
    latent_l1 = l1_sum / total_parts
    latent_cos = latent_cos_sum / total_parts
    zero_mse = zero_mse_sum / total_parts
    zero_l1 = zero_l1_sum / total_parts
    return {
        "samples": total_samples,
        "objects": total_objects,
        "parts": total_parts,
        "latent_mse": latent_mse,
        "latent_l1": latent_l1,
        "latent_cos": latent_cos,
        "zero_mse": zero_mse,
        "zero_l1": zero_l1,
        "mse_vs_zero": _safe_ratio(latent_mse, zero_mse),
        "l1_vs_zero": _safe_ratio(latent_l1, zero_l1),
    }


def _inspection_summary(rows: list[Dict[str, Any]]) -> Dict[str, float]:
    if not rows:
        out = {
            "samples": 0,
            "pred_nonempty": 0,
            "pred_count_mean": math.nan,
            "pred_logit_max_mean": math.nan,
            "gt_logit_max_mean": math.nan,
            "mse_vs_zero_mean": math.nan,
            "latent_cos_mean": math.nan,
        }
        out.update(summarize_bucketed_part_metrics(rows))
        return out
    out = {
        "samples": len(rows),
        "pred_nonempty": sum(1 for row in rows if int(row.get("pred_count", 0)) > 0),
        "pred_count_mean": float(np.mean([float(row.get("pred_count", 0)) for row in rows])),
        "pred_logit_max_mean": float(np.mean([float(row.get("pred_logit_max", 0.0)) for row in rows])),
        "gt_logit_max_mean": float(np.mean([float(row.get("gt_logit_max", 0.0)) for row in rows])),
        "mse_vs_zero_mean": float(np.mean([float(row.get("mse_vs_zero", 0.0)) for row in rows])),
        "latent_cos_mean": float(np.mean([float(row.get("latent_cos", 0.0)) for row in rows])),
        "assignment_diag_iou_mean": float(np.mean([float(row.get("assignment_diag_iou", 0.0)) for row in rows])),
        "assignment_offdiag_max_mean": float(np.mean([float(row.get("assignment_offdiag_max", 0.0)) for row in rows])),
    }
    out.update(summarize_bucketed_part_metrics(rows))
    return out


@torch.no_grad()
def _eval_decode_inspection(model, dataset, device, flow_cfg, eval_cfg, step: int) -> tuple[Path, Dict[str, float]]:
    raw_model = model.module if isinstance(model, DDP) else model
    raw_model.eval()
    decoder = load_ss_decoder(eval_cfg["ss_decoder_ckpt"])
    max_samples = int(eval_cfg.get("decode_max_samples", 20))
    sample_mode = str(eval_cfg.get("sample_mode", "first"))
    candidate_indices = _object_id_filter_indices(dataset, eval_cfg.get("object_ids"))
    sample_indices = _sample_indices_for_eval(len(dataset), max_samples, sample_mode, candidate_indices)
    eval_dataset = Subset(dataset, sample_indices)
    loader = DataLoader(eval_dataset, batch_size=1, shuffle=False, num_workers=0, collate_fn=dataset.collate_fn)
    threshold = float(eval_cfg.get("decode_threshold", 0.0))
    debug_thresholds = tuple(float(x) for x in eval_cfg.get("debug_thresholds", [0.0, -0.25, -0.5, -1.0]))
    rows = []
    panels = []
    object_panels = []
    progress_prefix = f"  [INSPECT @ {step}]"
    total_samples = len(sample_indices)
    _print_progress(progress_prefix, f"sampling/decode progress: 0/{total_samples} samples")
    for idx, batch in enumerate(loader):
        source_idx = sample_indices[idx]
        sample_meta = dataset.samples[source_idx]
        batch = _to_device(batch, device)
        obj_id = batch["obj_id"][0]
        valid_k = int(batch["part_valid"][0].sum().item())
        _print_progress(
            progress_prefix,
            f"sampling sample {idx + 1}/{total_samples}: dataset_idx={source_idx} obj={obj_id} parts={valid_k}",
        )
        pred = sample_part_ss_latent(
            raw_model,
            z_global=batch["z_global"],
            cond=batch["cond"],
            mask_token_labels=batch["mask_token_labels"],
            part_valid=batch["part_valid"],
            target_slots=batch["target_slots"],
            part_token_weights=batch.get("part_token_weights"),
            num_steps=int(flow_cfg.get("num_steps", 20)),
            noise_scale=float(flow_cfg.get("noise_scale", 1.0)),
            latent_scale=float(flow_cfg.get("latent_scale", 1.0)),
            latent_norm_mode=str(flow_cfg.get("latent_norm_mode", "scalar")),
            latent_mean=flow_cfg.get("latent_mean"),
            latent_std=flow_cfg.get("latent_std"),
            self_conditioning=bool(getattr(raw_model, "self_conditioning", False)),
            cfg_scale=float(flow_cfg.get("cfg_scale", 1.0)),
        )
        x_1 = batch["x_1_parts"]
        z_global = batch["z_global"]
        global_coords = decode_ss_latent_to_coords(decoder, z_global[0].detach().float().cpu(), threshold=threshold)
        object_rows = []
        sample_panels = []
        pred_coords_list = []
        raw_coords_list = []
        object_parts = []
        complete_png_name = f"{idx:03d}_{obj_id}_complete_voxels.png"
        for part_idx in range(valid_k):
            part_name = batch["target_part_names"][0][part_idx]
            pred_coords, pred_stats = decode_ss_latent_to_coords_with_stats(
                decoder,
                pred[0, part_idx].detach().float().cpu(),
                threshold=threshold,
                debug_thresholds=debug_thresholds,
            )
            gt_coords, gt_stats = decode_ss_latent_to_coords_with_stats(
                decoder,
                x_1[0, part_idx].detach().float().cpu(),
                threshold=threshold,
                debug_thresholds=debug_thresholds,
            )
            raw_ind_coords = batch["raw_ind_coords"][0][part_idx].detach().cpu()
            latent_metrics = _latent_metric_values(
                pred[0, part_idx].detach().unsqueeze(0),
                x_1[0, part_idx].detach().unsqueeze(0),
            )
            metrics = compute_decode_metrics(
                pred_coords=pred_coords,
                gt_decode_coords=gt_coords,
                raw_ind_coords=raw_ind_coords,
            )
            row = {
                "obj_id": obj_id,
                "sample_id": batch["sample_id"][0],
                "dataset_index": source_idx,
                "part_index": part_idx,
                "object_part_count": valid_k,
                "target_part_name": part_name,
                "target_slot": int(batch["target_slots"][0, part_idx].item()),
                "part_raw_voxel_count": int(raw_ind_coords.shape[0]),
                **latent_metrics,
                **metrics,
                "pred_logit_max": pred_stats["logit_max"],
                "pred_logit_mean": pred_stats["logit_mean"],
                "pred_logit_p99": pred_stats["logit_p99"],
                "pred_count_at_thresholds": pred_stats["counts"],
                "gt_logit_max": gt_stats["logit_max"],
                "gt_logit_mean": gt_stats["logit_mean"],
                "gt_count_at_thresholds": gt_stats["counts"],
                "png_name": complete_png_name,
            }
            object_rows.append(row)
            panel = {
                "obj_id": obj_id,
                "target_part_name": part_name,
                "pred_coords": pred_coords.numpy(),
                "gt_coords": gt_coords.numpy(),
                "global_coords": global_coords.numpy(),
                "raw_gt_coords": raw_ind_coords.numpy(),
                "rgb_views": [
                    str(path)
                    for path in dataset._iter_rgb_paths(sample_meta)
                ],
                "mask_views": [
                    str(path)
                    for path in dataset._iter_mask_paths(sample_meta)
                ],
            }
            panels.append(panel)
            sample_panels.append(panel)
            pred_coords_list.append(pred_coords)
            raw_coords_list.append(raw_ind_coords)
            object_parts.append({
                "name": part_name,
                "pred_coords": pred_coords.numpy(),
                "gt_coords": gt_coords.numpy(),
                "raw_coords": raw_ind_coords.numpy(),
            })
        assignment = summarize_assignment_matrix(part_assignment_iou_matrix(pred_coords_list, raw_coords_list))
        for row in object_rows:
            row.update(assignment)
        rows.extend(object_rows)
        object_panel = None
        if bool(eval_cfg.get("complete_object_voxels", True)):
            object_panel = {
                "obj_id": obj_id,
                "sample_id": batch["sample_id"][0],
                "dataset_index": source_idx,
                "part_count": valid_k,
                "surface_coords": batch["raw_surface_coords"][0].detach().cpu().numpy(),
                "global_coords": global_coords.numpy(),
                "rgb_views": [
                    str(path)
                    for path in dataset._iter_rgb_paths(sample_meta)
                ],
                "mask_views": [
                    str(path)
                    for path in dataset._iter_mask_paths(sample_meta)
                ],
                "parts": object_parts,
                "png_name": complete_png_name,
            }
            object_panels.append(object_panel)
        write_part_ss_inspection_sample(
            eval_cfg["inspection_root"],
            step,
            [],
            [],
            object_panel=object_panel,
            progress_prefix=progress_prefix,
            sample_index=idx + 1,
            total_samples=total_samples,
        )
        _print_progress(
            progress_prefix,
            f"finished sample {idx + 1}/{total_samples}: dataset_idx={source_idx} obj={obj_id} accumulated_part_panels={len(panels)}",
        )
    index_path = write_part_ss_inspection(
        eval_cfg["inspection_root"],
        step,
        rows,
        panels,
        object_panels=object_panels,
        progress_prefix=progress_prefix,
        write_images=False,
    )
    summary = _inspection_summary(rows)
    raw_model.train()
    return index_path, summary


def train(config, dataset_cls: type = PartSSLatentFlowDataset) -> None:
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
    resolved_object_weight_k_ref = _resolve_object_weight_k_ref(
        dataset,
        loss_cfg,
        rank=rank,
        is_distributed=is_distributed,
    )
    # Resolve per-channel latent normalization stats (no-op unless
    # flow.latent_norm_mode == 'per_channel'). Mutates flow_cfg in place so the
    # loss build and all eval sampler calls share the same statistics.
    _resolve_latent_stats(
        dataset,
        flow_cfg=flow_cfg,
        model_cfg=model_cfg,
        rank=rank,
        is_distributed=is_distributed,
    )
    # Persist the resolved per-channel stats back into the OmegaConf cfg so that
    # _save_ckpt's config_to_dict(cfg) carries them. Without this the stats live
    # only in the local flow_cfg dict and downstream samplers loading the ckpt
    # (inference / diagnose --from-ckpt-config) would denormalize with the wrong
    # (scalar) transform -> garbage decoded voxels.
    if flow_cfg.get("latent_mean") is not None and flow_cfg.get("latent_std") is not None:
        cfg.flow.latent_mean = list(flow_cfg["latent_mean"])
        cfg.flow.latent_std = list(flow_cfg["latent_std"])
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

    model = PartSSLatentFlowModel(**model_cfg).to(device)
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
    criterion = _build_part_ss_latent_rf_loss(
        loss_cfg=loss_cfg,
        flow_cfg=flow_cfg,
        resolved_object_weight_k_ref=resolved_object_weight_k_ref,
        model_cfg=model_cfg,
    )
    start_step = _load_resume_checkpoint(model, optimizer, scheduler, training_cfg, device, rank)

    output_dir = Path(training_cfg.get("output_dir", "runs/part_ss_latent_flow"))
    if rank == 0:
        output_dir.mkdir(parents=True, exist_ok=True)
        with (output_dir / "config.json").open("w", encoding="utf-8") as f:
            json.dump(config_to_dict(cfg), f, indent=2)
        n_params = sum(p.numel() for p in raw_model.parameters())
        total_parts = sum(len(sample["parts"]) for sample in dataset.samples)
        print("\n[Part SS Latent Flow Train]")
        print(f"  device={device} world_size={world_size}")
        print(f"  dataset: {len(dataset)} object samples / {total_parts} target parts | batch/gpu={training_cfg.get('batch_size', 1)}")
        print(f"  model: PartSSLatentFlowModel ({n_params:,} params)")
        print(
            "  binding-fix flags: "
            f"cross_part_attention={bool(model_cfg.get('cross_part_attention', False))} "
            f"token_identity_embedding={bool(model_cfg.get('token_identity_embedding', False))} "
            f"self_conditioning={bool(model_cfg.get('self_conditioning', False))} "
            f"classifier_free_guidance={bool(model_cfg.get('classifier_free_guidance', False))} "
            f"soft_role_marking={bool(model_cfg.get('soft_role_marking', False))} "
            f"summary_cross_part_attention={bool(model_cfg.get('summary_cross_part_attention', False))}"
            f"(m={int(model_cfg.get('n_summary_tokens', 64))}) "
            f"mask_attention_bias={bool(dict(model_cfg.get('mask_attention_bias', {})).get('enabled', False))}"
        )
        print(
            "  loss-fix flags: "
            f"part_shuffle={bool(loss_cfg.get('part_shuffle', False))} "
            f"velocity_contrastive_weight={float(loss_cfg.get('velocity_contrastive_weight', 0.05))} "
            f"t_schedule={flow_cfg.get('t_schedule', 'logit_normal')} "
            f"latent_norm_mode={flow_cfg.get('latent_norm_mode', 'scalar')} "
            f"cfg_dropout_prob={float(loss_cfg.get('cfg_dropout_prob', 0.0))} "
            f"self_conditioning_prob={float(loss_cfg.get('self_conditioning_prob', 0.5))}"
        )
        print(f"  loss: {loss_cfg}")
        if resolved_object_weight_k_ref is not None:
            print(f"  resolved_object_weight_k_ref={resolved_object_weight_k_ref:.3f}")
        print(f"  starting training: {max_steps} steps")
        if start_step > 0:
            print(f"  resume: continuing from step {start_step}")
        print("  [METRIC TARGETS] per-step fields and where each should head:")
        print("    loss / mse / mse_unw : total + RF velocity MSE        -> DECREASE, stay finite (NaN => lower lr)")
        print("    latent_l1            : L1(endpoint, GT latent)         -> DECREASE")
        print("    rel                  : relative-endpoint (small-part completeness) -> DECREASE")
        print("    vc                   : DeltaFM velocity-contrastive loss -> DECREASE")
        print("    vc_acc               : BINDING acc (pred velocity nearest to OWN target) -> RISE -> 1.0 (chance ~1/K)")
        print("    sc                   : self-conditioning fired this step (0/1) -> ~50% of steps =1 (sanity, no target)")
        print("    cfg_drop             : #parts with cond nulled this step -> ~10% rate (sanity, no target)")
        print("    wmax / owmax         : part / object loss-weight maxima -> bounded by clamps (sanity)")
        print("    t_mean               : mean sampled timestep            -> ~0.5 for logit_normal(0,1) (sanity)")
        print("    id / id_acc          : legacy endpoint contrastive (OFF) -> stays 0 / nan")
        print("    parts                : #valid target parts in batch     -> (sanity)")
        print("    >> REAL binding metric is OFFLINE: diagnose --mode pred -> diag IoU UP / off-diag DOWN")
        print("       (baseline before fixes: diag 0.13 / off-diag 0.44)")

    use_fp16 = bool(training_cfg.get("fp16", False)) and torch.cuda.is_available()
    scaler = torch.cuda.amp.GradScaler(enabled=use_fp16) if torch.cuda.is_available() else None
    grad_clip = float(training_cfg.get("grad_clip", 1.0))
    log_every = int(training_cfg.get("log_every", 10))
    ckpt_every = int(training_cfg.get("checkpoint_every", 500))
    fixed_every = int(getattr(cfg.eval, "fixed_every", 0)) if "eval" in cfg else 0
    all_every = int(getattr(cfg.eval, "all_every", 0)) if "eval" in cfg else 0
    decode_every = int(getattr(cfg.eval, "decode_every", 0)) if "eval" in cfg else 0

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
        if torch.cuda.is_available():
            # cache_enabled=False is REQUIRED with gradient checkpointing here.
            # The loss runs a no_grad self-conditioning pass before the grad pass;
            # with autocast weight-caching ON, that first pass populates the fp16
            # weight cache so the grad forward records cache HITS (no cast nodes),
            # but checkpoint recompute (fresh autocast in backward) records cache
            # MISSES (cast nodes) -> mismatched saved-tensor counts ->
            # CheckpointError. Disabling the cache makes forward and recompute
            # cast-identical (verified: gradients bit-identical to non-checkpointed).
            with torch.cuda.amp.autocast(enabled=use_fp16, cache_enabled=False):
                loss, metrics = criterion(model, batch)
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()
        else:
            loss, metrics = criterion(model, batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
        scheduler.step()

        if rank == 0 and (step == 1 or step % log_every == 0):
            print(
                f"  step {step:6d}/{max_steps} | loss={float(loss.item()):.4f} "
                f"mse={metrics['mse']:.4f} latent_l1={metrics['latent_l1']:.4f} "
                f"mse_unw={metrics.get('mse_unweighted', metrics['mse']):.4f} "
                f"wmax={metrics.get('part_weight_max', 1.0):.2f} "
                f"owmax={metrics.get('object_weight_max', 1.0):.2f} "
                f"rel={metrics.get('relative_endpoint_loss', 0.0):.3f} "
                f"vc={metrics.get('velocity_contrastive_loss', 0.0):.3f} "
                f"vc_acc={metrics.get('velocity_contrastive_acc', float('nan')):.2f}(->1) "
                f"id={metrics.get('identity_contrastive_loss', 0.0):.3f} "
                f"id_acc={metrics.get('identity_contrastive_acc', float('nan')):.2f} "
                f"sc={int(metrics.get('self_cond_active', 0))} "
                f"cfg_drop={int(metrics.get('cfg_dropped_parts', 0))} "
                f"parts={int(metrics.get('parts', 0))} "
                f"t_mean={metrics['t_mean']:.2f} | lr={scheduler.get_last_lr()[0]:.2e}"
            )

        if fixed_every > 0 and step % fixed_every == 0:
            ev = _run_rank0_only_work(
                "eval_fixed",
                rank,
                lambda: _eval_latent(model, dataset, device, flow_cfg, max_samples=1),
            )
            if rank == 0:
                print(
                    f"  [EVAL_FIXED @ {step}] objects={ev['objects']} parts={ev['parts']} "
                    f"latent_mse={ev['latent_mse']:.6f} latent_l1={ev['latent_l1']:.6f} "
                    f"zero_mse={ev['zero_mse']:.6f} mse/zero={ev['mse_vs_zero']:.3f} "
                    f"zero_l1={ev['zero_l1']:.6f} l1/zero={ev['l1_vs_zero']:.3f}"
                )
        if all_every > 0 and step % all_every == 0:
            ev = _run_rank0_only_work(
                "eval_all",
                rank,
                lambda: _eval_latent(model, dataset, device, flow_cfg, max_samples=None),
            )
            if rank == 0:
                print(
                    f"  [EVAL_ALL @ {step}] objects={ev['objects']} parts={ev['parts']} "
                    f"latent_mse={ev['latent_mse']:.6f} latent_l1={ev['latent_l1']:.6f} "
                    f"zero_mse={ev['zero_mse']:.6f} mse/zero={ev['mse_vs_zero']:.3f} "
                    f"zero_l1={ev['zero_l1']:.6f} l1/zero={ev['l1_vs_zero']:.3f}"
                )
        if decode_every > 0 and step % decode_every == 0:
            result = _run_rank0_only_work(
                "eval_decode",
                rank,
                lambda: _eval_decode_inspection(model, dataset, device, flow_cfg, eval_cfg, step),
            )
            if rank == 0:
                index_path, summary = result
                print(
                    f"  [INSPECT @ {step}] {index_path} "
                    f"pred_nonempty={int(summary['pred_nonempty'])}/{int(summary['samples'])} "
                    f"pred_count_mean={summary['pred_count_mean']:.1f} "
                    f"pred_logit_max_mean={summary['pred_logit_max_mean']:.2f} "
                    f"gt_logit_max_mean={summary['gt_logit_max_mean']:.2f} "
                    f"mse/zero_mean={summary['mse_vs_zero_mean']:.3f} "
                    f"assign_diag={summary['assignment_diag_iou_mean']:.3f} "
                    f"assign_offdiag={summary['assignment_offdiag_max_mean']:.3f}"
                )

        if ckpt_every > 0 and step % ckpt_every == 0:
            _run_rank0_only_work(
                "checkpoint",
                rank,
                lambda: _save_ckpt(step, model, optimizer, scheduler, cfg, output_dir, rank),
            )

    _run_rank0_only_work(
        "final_checkpoint",
        rank,
        lambda: _save_ckpt(max_steps, model, optimizer, scheduler, cfg, output_dir, rank),
    )
    if dist.is_initialized():
        dist.barrier()


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Part SS latent flow trainer")
    parser.add_argument("--config", required=True)
    parser.add_argument("overrides", nargs="*")
    args = parser.parse_args()
    cfg = load_config(args.config, overrides=args.overrides)
    train(cfg)


if __name__ == "__main__":
    main()
