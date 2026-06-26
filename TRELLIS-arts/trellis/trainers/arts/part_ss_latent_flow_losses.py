"""Continuous Rectified Flow loss and sampler for part SS latent flow."""

from __future__ import annotations

import math
from typing import Any, Dict, Tuple

import torch
import torch.nn.functional as F


__all__ = [
    "PartSSLatentRFLoss",
    "build_part_ss_sampler_kwargs",
    "compute_object_loss_weights",
    "compute_part_loss_weights",
    "foreground_weighted_part_mse",
    "k_bucket_name",
    "sample_flow_timesteps",
    "sample_part_ss_latent",
    "size_bucket_masks",
]


def build_part_ss_sampler_kwargs(model, flow_cfg) -> dict:
    """Single source of truth for the latent-norm / self-cond / CFG sampler kwargs.

    Every ``sample_part_ss_latent`` call site (trainer eval, inference, full-eval,
    export, diagnose) must build these from the SAME (model, flow_cfg) so sampling
    matches how the checkpoint was trained. ``flow_cfg`` is a plain dict
    (``config_to_dict(cfg.flow)``). Fails loudly when per-channel norm is selected
    without the stats, instead of silently mis-denormalizing (CLAUDE.md: no silent
    fallback). The per-channel stats are persisted into ``ckpt['config']['flow']``
    by the trainer, so ``--from-ckpt-config`` resolves them automatically.
    """
    latent_norm_mode = str(flow_cfg.get("latent_norm_mode", "scalar"))
    latent_mean = flow_cfg.get("latent_mean")
    latent_std = flow_cfg.get("latent_std")
    if latent_norm_mode == "per_channel" and (latent_mean is None or latent_std is None):
        raise ValueError(
            "latent_norm_mode='per_channel' requires flow.latent_mean and flow.latent_std, "
            "but they are None. For a trained checkpoint pass --from-ckpt-config (the trainer "
            "saves the stats into ckpt['config']['flow']), or set flow.latent_stats_path."
        )
    return dict(
        latent_norm_mode=latent_norm_mode,
        latent_mean=latent_mean,
        latent_std=latent_std,
        self_conditioning=bool(getattr(model, "self_conditioning", False)),
        cfg_scale=float(flow_cfg.get("cfg_scale", 1.0)),
    )


_EPS = 1.0e-6


def sample_flow_timesteps(
    batch_size: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
    t_min: float = 0.0,
    t_max: float = 1.0,
    t_schedule: str = "logit_normal",
    t_logit_normal_mean: float = 0.0,
    t_logit_normal_std: float = 1.0,
) -> torch.Tensor:
    """Sample continuous RF timesteps from the configured training schedule."""
    batch_size = int(batch_size)
    if batch_size <= 0:
        raise ValueError(f"batch_size must be > 0, got {batch_size}")
    t_min = float(t_min)
    t_max = float(t_max)
    if not t_min < t_max:
        raise ValueError(f"t_min must be < t_max, got t_min={t_min} t_max={t_max}")
    t_schedule = str(t_schedule)
    if t_schedule not in ("uniform", "logit_normal"):
        raise ValueError(f"unknown t_schedule={t_schedule!r}; expected 'uniform' or 'logit_normal'")
    t_logit_normal_std = float(t_logit_normal_std)
    if t_logit_normal_std <= 0:
        raise ValueError(f"t_logit_normal_std must be > 0, got {t_logit_normal_std}")

    if t_schedule == "logit_normal":
        z = torch.randn(batch_size, device=device, dtype=dtype)
        t01 = torch.sigmoid(z * t_logit_normal_std + float(t_logit_normal_mean))
    else:
        t01 = torch.rand(batch_size, device=device, dtype=dtype)
    return t01 * (t_max - t_min) + t_min


def _masked_positive_median(values: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    refs = []
    positive_valid = valid & (values > 0)
    for row, mask in zip(values, positive_valid):
        selected = row[mask]
        if selected.numel() == 0:
            refs.append(torch.ones((), device=values.device, dtype=values.dtype))
        else:
            refs.append(selected.float().median().to(device=values.device, dtype=values.dtype))
    return torch.stack(refs)


def compute_part_loss_weights(
    part_raw_voxel_counts: torch.Tensor,
    part_valid: torch.Tensor,
    *,
    mode: str,
    alpha: float,
    min_w: float,
    max_w: float,
    ref_mode: str,
    normalize_per_object: bool,
) -> tuple[torch.Tensor, Dict[str, Any]]:
    valid = part_valid.bool()
    counts = part_raw_voxel_counts.to(device=valid.device, dtype=torch.float32)
    if counts.shape != valid.shape:
        raise ValueError(f"part_raw_voxel_counts shape {tuple(counts.shape)} must match part_valid {tuple(valid.shape)}")
    if mode == "none":
        return valid.float(), {"part_count_zero": 0}
    if mode != "raw_voxel_count":
        raise ValueError(f"unknown part_weight_mode={mode!r}")
    if ref_mode != "median":
        raise ValueError(f"unknown part_weight_ref_mode={ref_mode!r}")
    if float(min_w) <= 0 or float(max_w) < float(min_w):
        raise ValueError(f"invalid part weight clamp min_w={min_w} max_w={max_w}")
    if bool((valid.sum(dim=1) <= 0).any()):
        raise ValueError("part_valid contains an object with zero valid parts")

    zero_valid = valid & (counts <= 0)
    safe_counts = counts.clamp_min(1.0)
    refs = _masked_positive_median(counts, valid).view(-1, 1).clamp_min(1.0)
    raw = (refs / safe_counts).pow(float(alpha)).clamp(float(min_w), float(max_w))
    raw = torch.where(zero_valid, torch.ones_like(raw), raw)
    raw = raw * valid.float()
    if normalize_per_object:
        denom = raw.sum(dim=1, keepdim=True).clamp_min(_EPS)
        k_valid = valid.float().sum(dim=1, keepdim=True).clamp_min(1.0)
        raw = raw * (k_valid / denom)
    return raw, {"part_count_zero": int(zero_valid.sum().item())}


def compute_object_loss_weights(
    part_valid: torch.Tensor,
    *,
    mode: str,
    k_ref: float | None,
    min_w: float,
    max_w: float,
) -> torch.Tensor:
    valid = part_valid.bool()
    k_valid = valid.sum(dim=1).float()
    if bool((k_valid <= 0).any()):
        raise ValueError("K_valid must be > 0 for every object")
    if mode == "none":
        return torch.ones_like(k_valid)
    if mode != "sqrt_k":
        raise ValueError(f"unknown object_weight_mode={mode!r}")
    if k_ref is None or float(k_ref) <= 0:
        raise ValueError(f"object_weight_k_ref must be > 0 for sqrt_k, got {k_ref}")
    if float(min_w) <= 0 or float(max_w) < float(min_w):
        raise ValueError(f"invalid object weight clamp min_w={min_w} max_w={max_w}")
    return torch.sqrt(k_valid / float(k_ref)).clamp(float(min_w), float(max_w))


def size_bucket_masks(
    counts: torch.Tensor,
    valid: torch.Tensor,
    boundaries: tuple[float, float],
) -> Dict[str, torch.Tensor]:
    small_hi, medium_hi = (float(boundaries[0]), float(boundaries[1]))
    if not small_hi < medium_hi:
        raise ValueError(f"size_bucket_boundaries must be increasing, got {boundaries}")
    counts = counts.to(device=valid.device, dtype=torch.float32)
    valid = valid.bool()
    return {
        "small": valid & (counts < small_hi),
        "medium": valid & (counts >= small_hi) & (counts < medium_hi),
        "large": valid & (counts >= medium_hi),
    }


def _raw_coords_to_part_fg_mask(
    coords: torch.Tensor,
    *,
    raw_resolution: int = 64,
    latent_resolution: int = 16,
    dilate: int = 0,
) -> torch.Tensor:
    if int(raw_resolution) % int(latent_resolution) != 0:
        raise ValueError(
            f"raw_resolution={raw_resolution} must be divisible by "
            f"latent_resolution={latent_resolution}"
        )
    coords = torch.as_tensor(coords, dtype=torch.long)
    if coords.dim() != 2 or coords.shape[1] != 3:
        raise ValueError(f"coords expected [N,3], got {tuple(coords.shape)}")
    mask = torch.zeros(
        (int(latent_resolution), int(latent_resolution), int(latent_resolution)),
        dtype=torch.bool,
    )
    if coords.numel() == 0:
        return mask
    if int(coords.min().item()) < 0 or int(coords.max().item()) >= int(raw_resolution):
        raise ValueError(
            f"raw coords must be in [0,{int(raw_resolution) - 1}], "
            f"got min={int(coords.min().item())} max={int(coords.max().item())}"
        )
    stride = int(raw_resolution) // int(latent_resolution)
    latent_coords = torch.div(coords, stride, rounding_mode="floor")
    mask[latent_coords[:, 0], latent_coords[:, 1], latent_coords[:, 2]] = True
    dilate = int(dilate)
    if dilate < 0:
        raise ValueError(f"dilate must be >= 0, got {dilate}")
    if dilate:
        pooled = F.max_pool3d(
            mask.float().view(1, 1, *mask.shape),
            kernel_size=2 * dilate + 1,
            stride=1,
            padding=dilate,
        )
        mask = pooled[0, 0].bool()
    return mask


def foreground_weighted_part_mse(
    sq_error: torch.Tensor,
    part_fg_mask: torch.Tensor | None,
    *,
    enabled: bool = True,
    bg_weight: float = 0.1,
    eps: float = _EPS,
) -> torch.Tensor:
    """Reduce [B,K,C,R,R,R] squared error to per-part MSE with optional fg weights."""

    if sq_error.dim() != 6:
        raise ValueError(f"sq_error expected [B,K,C,R,R,R], got {tuple(sq_error.shape)}")
    if not enabled:
        return sq_error.flatten(start_dim=2).mean(dim=2)
    bg_weight = float(bg_weight)
    if bg_weight <= 0:
        raise ValueError(f"foreground_weight.bg_weight must be > 0, got {bg_weight}")
    if part_fg_mask is None:
        raise KeyError("part_fg_mask is required when foreground_weight.enabled=true")
    expected = sq_error.shape[:2] + sq_error.shape[3:]
    if tuple(part_fg_mask.shape) != tuple(expected):
        raise ValueError(f"part_fg_mask expected {tuple(expected)}, got {tuple(part_fg_mask.shape)}")
    weights = torch.where(
        part_fg_mask.to(device=sq_error.device).bool(),
        torch.ones((), device=sq_error.device, dtype=sq_error.dtype),
        torch.full((), bg_weight, device=sq_error.device, dtype=sq_error.dtype),
    ).unsqueeze(2).expand_as(sq_error)
    weighted_sum = (sq_error * weights).sum(dim=(2, 3, 4, 5))
    denom = weights.sum(dim=(2, 3, 4, 5)).clamp_min(float(eps))
    return weighted_sum / denom


def k_bucket_name(k_valid: int) -> str:
    k_valid = int(k_valid)
    if k_valid <= 2:
        return "k_1_2"
    if k_valid <= 5:
        return "k_3_5"
    if k_valid <= 10:
        return "k_6_10"
    if k_valid <= 15:
        return "k_11_15"
    return "k_16_plus"


def _normalize_latent_stats(
    stats: torch.Tensor | list | tuple | None,
    *,
    name: str,
    latent_channels: int,
) -> torch.Tensor | None:
    """Normalize a per-channel latent mean/std spec to a [C,1,1,1] tensor."""
    if stats is None:
        return None
    tensor = stats if isinstance(stats, torch.Tensor) else torch.as_tensor(stats, dtype=torch.float32)
    tensor = tensor.float()
    if tensor.numel() != latent_channels:
        raise ValueError(
            f"{name} must have {latent_channels} elements (one per latent channel), "
            f"got {tuple(tensor.shape)}"
        )
    return tensor.reshape(latent_channels, 1, 1, 1)


def _broadcast_latent_stats(stats: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Broadcast a [C,1,1,1] stat tensor against an [..., C, R, R, R] latent tensor."""
    view = (1,) * (x.dim() - 4) + tuple(stats.shape)
    return stats.to(device=x.device, dtype=x.dtype).view(view)


class PartSSLatentRFLoss:
    def __init__(
        self,
        t_min: float = 0.0,
        t_max: float = 1.0,
        noise_scale: float = 1.0,
        latent_scale: float = 1.0,
        *,
        part_weight_mode: str = "none",
        part_weight_ref_mode: str = "median",
        part_weight_alpha: float = 0.5,
        part_weight_min: float = 0.5,
        part_weight_max: float = 3.0,
        normalize_part_weights_per_object: bool = True,
        size_bucket_boundaries: tuple[float, float] = (500.0, 3000.0),
        object_balanced: bool = False,
        object_weight_mode: str = "none",
        object_weight_k_ref: float | None = None,
        object_weight_min: float = 0.75,
        object_weight_max: float = 2.0,
        relative_endpoint_weight: float = 0.0,
        relative_endpoint_eps: float = 1.0e-6,
        # Fix 4: DeltaFM velocity-contrastive identity term (default on).
        velocity_contrastive_weight: float = 0.05,
        velocity_contrastive_lambda: float = 0.05,
        # Legacy endpoint-based identity contrastive, kept for ablation only.
        identity_contrastive_weight: float = 0.0,
        identity_contrastive_temperature: float = 0.1,
        identity_contrastive_eps: float = 1.0e-6,
        # Fix 5: continuous-time sampling schedule.
        t_schedule: str = "logit_normal",
        t_logit_normal_mean: float = 0.0,
        t_logit_normal_std: float = 1.0,
        # Fix 6: per-channel latent normalization (alternative to scalar latent_scale).
        latent_norm_mode: str = "scalar",
        latent_channels: int = 8,
        latent_mean: torch.Tensor | list | tuple | None = None,
        latent_std: torch.Tensor | list | tuple | None = None,
        # Fix 3: per-object slot<->part shuffle to defeat fixed-slot memorization.
        part_shuffle: bool = False,
        # Model-side self-conditioning double-pass (training-time).
        self_conditioning: bool = False,
        self_conditioning_prob: float = 0.5,
        # Classifier-free-guidance per-part condition dropout (training-time).
        cfg_dropout_prob: float = 0.0,
        foreground_weight: Dict[str, Any] | None = None,
    ):
        self.t_min = float(t_min)
        self.t_max = float(t_max)
        self.noise_scale = float(noise_scale)
        self.latent_scale = float(latent_scale)
        self.part_weight_mode = str(part_weight_mode)
        self.part_weight_ref_mode = str(part_weight_ref_mode)
        self.part_weight_alpha = float(part_weight_alpha)
        self.part_weight_min = float(part_weight_min)
        self.part_weight_max = float(part_weight_max)
        self.normalize_part_weights_per_object = bool(normalize_part_weights_per_object)
        self.size_bucket_boundaries = (float(size_bucket_boundaries[0]), float(size_bucket_boundaries[1]))
        self.object_balanced = bool(object_balanced)
        self.object_weight_mode = str(object_weight_mode)
        self.object_weight_k_ref = None if object_weight_k_ref is None else float(object_weight_k_ref)
        self.object_weight_min = float(object_weight_min)
        self.object_weight_max = float(object_weight_max)
        self.relative_endpoint_weight = float(relative_endpoint_weight)
        self.relative_endpoint_eps = float(relative_endpoint_eps)
        self.velocity_contrastive_weight = float(velocity_contrastive_weight)
        self.velocity_contrastive_lambda = float(velocity_contrastive_lambda)
        self.identity_contrastive_weight = float(identity_contrastive_weight)
        self.identity_contrastive_temperature = float(identity_contrastive_temperature)
        self.identity_contrastive_eps = float(identity_contrastive_eps)
        self.t_schedule = str(t_schedule)
        self.t_logit_normal_mean = float(t_logit_normal_mean)
        self.t_logit_normal_std = float(t_logit_normal_std)
        self.latent_norm_mode = str(latent_norm_mode)
        self.latent_channels = int(latent_channels)
        self.latent_mean = _normalize_latent_stats(latent_mean, name="latent_mean", latent_channels=self.latent_channels)
        self.latent_std = _normalize_latent_stats(latent_std, name="latent_std", latent_channels=self.latent_channels)
        self.part_shuffle = bool(part_shuffle)
        self.self_conditioning = bool(self_conditioning)
        self.self_conditioning_prob = float(self_conditioning_prob)
        self.cfg_dropout_prob = float(cfg_dropout_prob)
        foreground_weight = foreground_weight or {"enabled": True, "bg_weight": 0.1, "dilate": 0}
        self.foreground_weight_enabled = bool(foreground_weight.get("enabled", True))
        self.foreground_bg_weight = float(foreground_weight.get("bg_weight", 0.1))
        self.foreground_dilate = int(foreground_weight.get("dilate", 0))

        if self.latent_scale <= 0:
            raise ValueError(f"latent_scale must be > 0, got {latent_scale}")
        if self.relative_endpoint_weight < 0:
            raise ValueError(f"relative_endpoint_weight must be >= 0, got {relative_endpoint_weight}")
        if self.relative_endpoint_eps <= 0:
            raise ValueError(f"relative_endpoint_eps must be > 0, got {relative_endpoint_eps}")
        if self.velocity_contrastive_weight < 0:
            raise ValueError(f"velocity_contrastive_weight must be >= 0, got {velocity_contrastive_weight}")
        if self.velocity_contrastive_lambda < 0:
            raise ValueError(f"velocity_contrastive_lambda must be >= 0, got {velocity_contrastive_lambda}")
        if self.identity_contrastive_weight < 0:
            raise ValueError(f"identity_contrastive_weight must be >= 0, got {identity_contrastive_weight}")
        if self.identity_contrastive_temperature <= 0:
            raise ValueError(
                f"identity_contrastive_temperature must be > 0, got {identity_contrastive_temperature}"
            )
        if self.identity_contrastive_eps <= 0:
            raise ValueError(f"identity_contrastive_eps must be > 0, got {identity_contrastive_eps}")
        if self.t_schedule not in ("uniform", "logit_normal"):
            raise ValueError(f"unknown t_schedule={t_schedule!r}; expected 'uniform' or 'logit_normal'")
        if self.t_logit_normal_std <= 0:
            raise ValueError(f"t_logit_normal_std must be > 0, got {t_logit_normal_std}")
        if self.latent_norm_mode not in ("scalar", "per_channel"):
            raise ValueError(f"unknown latent_norm_mode={latent_norm_mode!r}; expected 'scalar' or 'per_channel'")
        if self.latent_channels <= 0:
            raise ValueError(f"latent_channels must be > 0, got {latent_channels}")
        if self.latent_norm_mode == "per_channel":
            if self.latent_mean is None or self.latent_std is None:
                raise ValueError("latent_norm_mode='per_channel' requires both latent_mean and latent_std")
            if bool((self.latent_std <= 0).any()):
                raise ValueError("latent_std must be > 0 for every channel")
        if not 0.0 <= self.self_conditioning_prob <= 1.0:
            raise ValueError(f"self_conditioning_prob must be in [0,1], got {self_conditioning_prob}")
        if not 0.0 <= self.cfg_dropout_prob <= 1.0:
            raise ValueError(f"cfg_dropout_prob must be in [0,1], got {cfg_dropout_prob}")
        if self.foreground_bg_weight <= 0:
            raise ValueError(f"foreground_weight.bg_weight must be > 0, got {self.foreground_bg_weight}")
        if self.foreground_dilate < 0:
            raise ValueError(f"foreground_weight.dilate must be >= 0, got {self.foreground_dilate}")

    # ------------------------------------------------------------------
    # Latent (de)normalization (Fix 6)
    # ------------------------------------------------------------------
    def _normalize_latent(self, x_raw: torch.Tensor) -> torch.Tensor:
        """Map raw decoder-scale latents into the RF training space."""
        if self.latent_norm_mode == "per_channel":
            mean = _broadcast_latent_stats(self.latent_mean, x_raw)
            std = _broadcast_latent_stats(self.latent_std, x_raw)
            return (x_raw - mean) / std
        return x_raw * self.latent_scale

    def _denormalize_latent(self, x_norm: torch.Tensor) -> torch.Tensor:
        """Map an RF-space latent back to raw decoder scale."""
        if self.latent_norm_mode == "per_channel":
            mean = _broadcast_latent_stats(self.latent_mean, x_norm)
            std = _broadcast_latent_stats(self.latent_std, x_norm)
            return x_norm * std + mean
        return x_norm / self.latent_scale

    # ------------------------------------------------------------------
    # Continuous-time sampling (Fix 5)
    # ------------------------------------------------------------------
    def _sample_t(self, B: int, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        return sample_flow_timesteps(
            B,
            device=device,
            dtype=dtype,
            t_min=self.t_min,
            t_max=self.t_max,
            t_schedule=self.t_schedule,
            t_logit_normal_mean=self.t_logit_normal_mean,
            t_logit_normal_std=self.t_logit_normal_std,
        )

    def _aggregate_part_loss(
        self,
        per_part_loss: torch.Tensor,
        part_valid: torch.Tensor,
        part_weights: torch.Tensor,
        object_weights: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        valid_f = part_valid.to(device=per_part_loss.device, dtype=per_part_loss.dtype)
        part_weights = part_weights.to(device=per_part_loss.device, dtype=per_part_loss.dtype)
        weighted_terms = per_part_loss * part_weights * valid_f
        per_object_denom = (part_weights * valid_f).sum(dim=1).clamp_min(_EPS)
        per_object_loss = weighted_terms.sum(dim=1) / per_object_denom
        if self.object_balanced:
            object_weights = object_weights.to(device=per_part_loss.device, dtype=per_part_loss.dtype)
            loss = (per_object_loss * object_weights).sum() / object_weights.sum().clamp_min(_EPS)
        else:
            loss = weighted_terms.sum() / (part_weights * valid_f).sum().clamp_min(_EPS)
        return loss, per_object_loss

    def _part_fg_mask_from_batch(
        self,
        batch: Dict[str, Any],
        part_valid: torch.Tensor,
        *,
        device: torch.device,
    ) -> torch.Tensor | None:
        if not self.foreground_weight_enabled:
            return None
        if "part_fg_mask" in batch:
            mask = batch["part_fg_mask"]
            return mask.to(device=device, dtype=torch.bool)
        if "raw_ind_coords" not in batch:
            raise KeyError("batch missing 'raw_ind_coords' for foreground_weight.enabled=true")
        raw_ind_coords = batch["raw_ind_coords"]
        batch_size, part_count = part_valid.shape
        rows = []
        for row in range(batch_size):
            if row >= len(raw_ind_coords):
                raise ValueError(
                    f"raw_ind_coords has {len(raw_ind_coords)} rows, expected {batch_size}"
                )
            row_masks = []
            for part_idx in range(part_count):
                if bool(part_valid[row, part_idx]):
                    if part_idx >= len(raw_ind_coords[row]):
                        raise ValueError(
                            f"raw_ind_coords[{row}] has {len(raw_ind_coords[row])} parts, "
                            f"expected at least {part_idx + 1}"
                        )
                    row_masks.append(
                        _raw_coords_to_part_fg_mask(
                            raw_ind_coords[row][part_idx],
                            raw_resolution=64,
                            latent_resolution=16,
                            dilate=self.foreground_dilate,
                        )
                    )
                else:
                    row_masks.append(torch.zeros((16, 16, 16), dtype=torch.bool))
            rows.append(torch.stack(row_masks, dim=0))
        return torch.stack(rows, dim=0).to(device=device, dtype=torch.bool)

    # ------------------------------------------------------------------
    # Fix 3: per-object slot<->part shuffle
    # ------------------------------------------------------------------
    def _shuffle_parts(
        self,
        x_1_raw: torch.Tensor,
        target_slots: torch.Tensor,
        part_token_weights: torch.Tensor | None,
        part_valid: torch.Tensor,
        part_raw_voxel_counts: torch.Tensor | None,
        part_fg_mask: torch.Tensor | None,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor | None,
    ]:
        """Independently permute the (target_slot <-> GT part) binding per object.

        Permuting x_1_parts, target_slots, part_token_weights (and the raw voxel
        counts used for part weighting) by the SAME per-object permutation over
        the valid positions keeps every (latent, slot id, condition tokens, weight)
        tuple intact while breaking any fixed slot-index -> part identity mapping.
        Invalid (padded) positions are left untouched.
        """
        x_1 = x_1_raw.clone()
        slots = target_slots.clone()
        weights = None if part_token_weights is None else part_token_weights.clone()
        counts = None if part_raw_voxel_counts is None else part_raw_voxel_counts.clone()
        fg_mask = None if part_fg_mask is None else part_fg_mask.clone()
        B, K = part_valid.shape
        for b in range(B):
            valid_idx = torch.nonzero(part_valid[b], as_tuple=False).flatten()
            if valid_idx.numel() < 2:
                continue
            perm = valid_idx[torch.randperm(valid_idx.numel(), device=valid_idx.device)]
            x_1[b, valid_idx] = x_1_raw[b, perm]
            slots[b, valid_idx] = target_slots[b, perm]
            if weights is not None:
                weights[b, valid_idx] = part_token_weights[b, perm]
            if counts is not None:
                counts[b, valid_idx] = part_raw_voxel_counts[b, perm]
            if fg_mask is not None:
                fg_mask[b, valid_idx] = part_fg_mask[b, perm]
        return x_1, slots, weights, counts, fg_mask

    # ------------------------------------------------------------------
    # Fix 4: DeltaFM velocity-contrastive identity loss
    # ------------------------------------------------------------------
    def _velocity_contrastive_loss(
        self,
        v_pred: torch.Tensor,
        v_target: torch.Tensor,
        part_valid: torch.Tensor,
    ) -> tuple[torch.Tensor, Dict[str, float]]:
        """Push each part's predicted velocity toward its own target velocity and
        away from sibling parts' target velocities (same-object negatives).

        loss_i = ||v_i - v_target_i||^2 - lambda * mean_{j!=i} ||v_i - v_target_j||^2

        Operates purely on the model's predicted velocity (no GT latent leakage
        into the prediction path) and only on objects with >= 2 valid parts.
        """
        lam = float(self.velocity_contrastive_lambda)
        losses = []
        accs = []
        margins = []
        for batch_idx in range(v_pred.shape[0]):
            valid = part_valid[batch_idx].bool()
            k_valid = int(valid.sum().item())
            if k_valid < 2:
                continue
            pred = v_pred[batch_idx, valid].flatten(start_dim=1).float()
            target = v_target[batch_idx, valid].flatten(start_dim=1).float()
            dim = float(pred.shape[1])
            # Pairwise squared L2 (mean over latent dim) between predicted velocity
            # i and target velocity j: ||p_i||^2 + ||q_j||^2 - 2 p_i . q_j.
            pred_norm = pred.pow(2).mean(dim=1, keepdim=True)  # [K,1]
            target_norm = target.pow(2).mean(dim=1).view(1, -1)  # [1,K]
            cross = pred @ target.t() / dim  # [K,K]
            dist = (pred_norm + target_norm - 2.0 * cross).clamp_min(0.0)  # [K,K]
            pos = dist.diagonal()
            eye = torch.eye(k_valid, device=v_pred.device, dtype=torch.bool)
            neg = dist.masked_fill(eye, 0.0).sum(dim=1) / float(k_valid - 1)
            losses.append((pos - lam * neg).mean())
            with torch.no_grad():
                accs.append((dist.argmin(dim=1) == torch.arange(k_valid, device=v_pred.device)).float().mean())
                offdiag = dist.masked_fill(eye, float("inf"))
                margins.append((offdiag.min(dim=1).values - pos).mean())
        if not losses:
            zero = v_pred.new_zeros(())
            return zero, {"objects": 0.0, "acc": math.nan, "margin": math.nan}
        loss = torch.stack(losses).mean()
        with torch.no_grad():
            acc = torch.stack(accs).mean().item()
            margin = torch.stack(margins).mean().item()
        return loss, {"objects": float(len(losses)), "acc": float(acc), "margin": float(margin)}

    # ------------------------------------------------------------------
    # Legacy endpoint-based identity contrastive (ablation only)
    # ------------------------------------------------------------------
    def _identity_contrastive_loss(
        self,
        endpoint_raw: torch.Tensor,
        x_1_raw: torch.Tensor,
        part_valid: torch.Tensor,
    ) -> tuple[torch.Tensor, Dict[str, float]]:
        losses = []
        accs = []
        margins = []
        eye_cache: dict[int, torch.Tensor] = {}
        for batch_idx in range(endpoint_raw.shape[0]):
            valid = part_valid[batch_idx].bool()
            k_valid = int(valid.sum().item())
            if k_valid < 2:
                continue
            pred = endpoint_raw[batch_idx, valid].flatten(start_dim=1).float()
            target = x_1_raw[batch_idx, valid].flatten(start_dim=1).float()
            dim = float(pred.shape[1])
            pred_norm = pred.pow(2).mean(dim=1, keepdim=True)
            target_norm = target.pow(2).mean(dim=1).view(1, -1)
            cross = pred @ target.t() / dim
            dist = (pred_norm + target_norm - 2.0 * cross).clamp_min(0.0)
            relative_dist = dist / target_norm.clamp_min(self.identity_contrastive_eps)
            logits = -relative_dist / self.identity_contrastive_temperature
            targets = torch.arange(k_valid, device=endpoint_raw.device)
            row_loss = F.cross_entropy(logits, targets)
            col_loss = F.cross_entropy(logits.t(), targets)
            losses.append(0.5 * (row_loss + col_loss))
            with torch.no_grad():
                row_acc = (relative_dist.argmin(dim=1) == targets).float().mean()
                col_acc = (relative_dist.argmin(dim=0) == targets).float().mean()
                accs.append(0.5 * (row_acc + col_acc))
                if k_valid not in eye_cache:
                    eye_cache[k_valid] = torch.eye(k_valid, device=endpoint_raw.device, dtype=torch.bool)
                offdiag = relative_dist.masked_fill(eye_cache[k_valid], float("inf"))
                margins.append((offdiag.min(dim=1).values - relative_dist.diag()).mean())
        if not losses:
            zero = endpoint_raw.new_zeros(())
            return zero, {"objects": 0.0, "acc": math.nan, "margin": math.nan}
        loss = torch.stack(losses).mean()
        with torch.no_grad():
            acc = torch.stack(accs).mean().item()
            margin = torch.stack(margins).mean().item()
        return loss, {"objects": float(len(losses)), "acc": float(acc), "margin": float(margin)}

    def _sample_cfg_drop_mask(
        self,
        part_valid: torch.Tensor,
    ) -> torch.Tensor | None:
        """Per-part Bernoulli CFG dropout mask over valid parts ([B,K] bool)."""
        if self.cfg_dropout_prob <= 0:
            return None
        drop = torch.rand(part_valid.shape, device=part_valid.device) < self.cfg_dropout_prob
        return drop & part_valid

    def __call__(self, model, batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, float]]:
        x_1_raw = batch["x_1_parts"]
        z_global = batch["z_global"]
        cond = batch["cond"]
        mask_token_labels = batch["mask_token_labels"]
        part_valid = batch["part_valid"].bool()
        target_slots = batch["target_slots"]
        if not bool(part_valid.any()):
            raise ValueError("part_valid contains no valid target parts")

        part_token_weights = batch.get("part_token_weights")
        part_raw_voxel_counts = batch.get("part_raw_voxel_counts")
        part_fg_mask = self._part_fg_mask_from_batch(batch, part_valid, device=x_1_raw.device)

        # Fix 3: shuffle the slot<->part binding per object before building the
        # RF target so identity cannot collapse onto a fixed slot index.
        if self.part_shuffle:
            x_1_raw, target_slots, part_token_weights, part_raw_voxel_counts, part_fg_mask = self._shuffle_parts(
                x_1_raw,
                target_slots,
                part_token_weights,
                part_valid,
                part_raw_voxel_counts,
                part_fg_mask,
            )

        x_1 = self._normalize_latent(x_1_raw)

        B = x_1.shape[0]
        device = x_1.device
        noise = batch.get("debug_noise")
        if noise is None:
            noise = torch.randn_like(x_1) * self.noise_scale
        else:
            noise = noise.to(device=device, dtype=x_1.dtype)

        t = batch.get("debug_t")
        if t is None:
            t = self._sample_t(B, device=device, dtype=x_1.dtype)
        else:
            t = t.to(device=device, dtype=x_1.dtype)

        view_shape = (B,) + (1,) * (x_1.dim() - 1)
        t_view = t.view(view_shape)
        x_t = (1.0 - t_view) * noise + t_view * x_1
        v_target = x_1 - noise

        base_model_kwargs: Dict[str, Any] = {}
        if part_token_weights is not None:
            base_model_kwargs["part_token_weights"] = part_token_weights

        # Classifier-free-guidance training dropout: null per-part condition while
        # keeping the global z tokens (handled model-side via drop_part_cond).
        drop_mask = self._sample_cfg_drop_mask(part_valid)
        cfg_dropped_parts = int(drop_mask.sum().item()) if drop_mask is not None else 0

        # Self-conditioning double-pass: with probability self_conditioning_prob,
        # run a stop-grad pass to estimate x0_hat and feed it back as x_self_cond.
        x_self_cond = None
        self_cond_active = False
        if self.self_conditioning and float(torch.rand(())) < self.self_conditioning_prob:
            self_cond_active = True
            with torch.no_grad():
                first_kwargs = dict(base_model_kwargs)
                first_kwargs["x_self_cond"] = torch.zeros_like(x_t)
                if drop_mask is not None:
                    first_kwargs["drop_part_cond"] = drop_mask
                v_first = model(
                    x_t, t, z_global, cond, mask_token_labels, part_valid, target_slots, **first_kwargs
                )
                # Endpoint estimate x0_hat in RF space, fed back as previous-x0_hat.
                x_self_cond = (x_t + (1.0 - t_view) * v_first).detach()

        model_kwargs = dict(base_model_kwargs)
        if self.self_conditioning:
            model_kwargs["x_self_cond"] = x_self_cond if x_self_cond is not None else torch.zeros_like(x_t)
        if drop_mask is not None:
            model_kwargs["drop_part_cond"] = drop_mask

        v_pred = model(x_t, t, z_global, cond, mask_token_labels, part_valid, target_slots, **model_kwargs)
        diff2 = (v_pred - v_target).pow(2)
        per_part_mse = foreground_weighted_part_mse(
            diff2,
            part_fg_mask,
            enabled=self.foreground_weight_enabled,
            bg_weight=self.foreground_bg_weight,
        ).masked_fill(~part_valid, 0.0)
        if self.part_weight_mode == "raw_voxel_count":
            if part_raw_voxel_counts is None:
                raise KeyError("batch missing 'part_raw_voxel_counts' for raw_voxel_count weighting")
            part_weights, weight_stats = compute_part_loss_weights(
                part_raw_voxel_counts,
                part_valid,
                mode=self.part_weight_mode,
                alpha=self.part_weight_alpha,
                min_w=self.part_weight_min,
                max_w=self.part_weight_max,
                ref_mode=self.part_weight_ref_mode,
                normalize_per_object=self.normalize_part_weights_per_object,
            )
            part_weights = part_weights.to(device=device, dtype=per_part_mse.dtype)
        else:
            part_weights, weight_stats = compute_part_loss_weights(
                torch.zeros_like(part_valid, dtype=torch.float32),
                part_valid,
                mode="none",
                alpha=self.part_weight_alpha,
                min_w=self.part_weight_min,
                max_w=self.part_weight_max,
                ref_mode=self.part_weight_ref_mode,
                normalize_per_object=self.normalize_part_weights_per_object,
            )
            part_weights = part_weights.to(device=device, dtype=per_part_mse.dtype)

        if self.object_balanced:
            object_weights = compute_object_loss_weights(
                part_valid,
                mode=self.object_weight_mode,
                k_ref=self.object_weight_k_ref,
                min_w=self.object_weight_min,
                max_w=self.object_weight_max,
            ).to(device=device, dtype=per_part_mse.dtype)
        else:
            object_weights = torch.ones(B, device=device, dtype=per_part_mse.dtype)
        base_loss, per_object_loss = self._aggregate_part_loss(per_part_mse, part_valid, part_weights, object_weights)
        endpoint = x_t + (1.0 - t_view) * v_pred
        endpoint_raw = self._denormalize_latent(endpoint)

        if self.relative_endpoint_weight > 0:
            endpoint_diff2 = (endpoint_raw - x_1_raw).pow(2)
            per_part_endpoint_mse = endpoint_diff2.flatten(start_dim=2).mean(dim=2)
            per_part_zero_mse = x_1_raw.pow(2).flatten(start_dim=2).mean(dim=2).clamp_min(self.relative_endpoint_eps)
            per_part_relative = per_part_endpoint_mse / per_part_zero_mse
            relative_endpoint_loss, _relative_per_object = self._aggregate_part_loss(
                per_part_relative,
                part_valid,
                part_weights,
                object_weights,
            )
        else:
            relative_endpoint_loss = base_loss.new_zeros(())

        if self.velocity_contrastive_weight > 0:
            velocity_contrastive_loss, velocity_stats = self._velocity_contrastive_loss(
                v_pred,
                v_target,
                part_valid,
            )
        else:
            velocity_contrastive_loss = base_loss.new_zeros(())
            velocity_stats = {"objects": 0.0, "acc": math.nan, "margin": math.nan}

        if self.identity_contrastive_weight > 0:
            identity_contrastive_loss, identity_stats = self._identity_contrastive_loss(
                endpoint_raw,
                x_1_raw,
                part_valid,
            )
        else:
            identity_contrastive_loss = base_loss.new_zeros(())
            identity_stats = {"objects": 0.0, "acc": math.nan, "margin": math.nan}

        loss = (
            base_loss
            + self.relative_endpoint_weight * relative_endpoint_loss
            + self.velocity_contrastive_weight * velocity_contrastive_loss
            + self.identity_contrastive_weight * identity_contrastive_loss
        )

        with torch.no_grad():
            latent_l1 = F.l1_loss(endpoint_raw[part_valid], x_1_raw[part_valid])
            latent_mse = F.mse_loss(endpoint_raw[part_valid], x_1_raw[part_valid])
            mse_unweighted = per_part_mse[part_valid].mean()
            valid_weights = part_weights[part_valid]
        metrics = {
            "loss_total": float(loss.detach().item()),
            "mse": float(base_loss.detach().item()),
            "mse_unweighted": float(mse_unweighted.detach().item()),
            "mse_weighted": float(base_loss.detach().item()),
            "relative_endpoint_loss": float(relative_endpoint_loss.detach().item()),
            "relative_endpoint_weighted": float(
                (self.relative_endpoint_weight * relative_endpoint_loss).detach().item()
            ),
            "velocity_contrastive_loss": float(velocity_contrastive_loss.detach().item()),
            "velocity_contrastive_weighted": float(
                (self.velocity_contrastive_weight * velocity_contrastive_loss).detach().item()
            ),
            "velocity_contrastive_objects": int(velocity_stats["objects"]),
            "velocity_contrastive_acc": float(velocity_stats["acc"]),
            "velocity_contrastive_margin": float(velocity_stats["margin"]),
            "identity_contrastive_loss": float(identity_contrastive_loss.detach().item()),
            "identity_contrastive_weighted": float(
                (self.identity_contrastive_weight * identity_contrastive_loss).detach().item()
            ),
            "identity_contrastive_objects": int(identity_stats["objects"]),
            "identity_contrastive_acc": float(identity_stats["acc"]),
            "identity_contrastive_margin": float(identity_stats["margin"]),
            "latent_mse": float(latent_mse.detach().item()),
            "latent_l1": float(latent_l1.detach().item()),
            "t_mean": float(t.detach().float().mean().item()),
            "parts": int(part_valid.sum().item()),
            "part_weight_mean": float(valid_weights.detach().float().mean().item()),
            "part_weight_max": float(valid_weights.detach().float().max().item()),
            "object_weight_mean": float(object_weights.detach().float().mean().item()),
            "object_weight_max": float(object_weights.detach().float().max().item()),
            "part_count_zero": int(weight_stats.get("part_count_zero", 0)),
            "self_cond_active": int(self_cond_active),
            "cfg_dropped_parts": int(cfg_dropped_parts),
        }
        counts = part_raw_voxel_counts
        if counts is not None:
            counts = counts.to(device=device, dtype=per_part_mse.dtype)
            for name, mask in size_bucket_masks(counts, part_valid, self.size_bucket_boundaries).items():
                metrics[f"part_count_{name}"] = int(mask.sum().item())
                metrics[f"mse_size_{name}"] = (
                    float(per_part_mse[mask].detach().mean().item()) if bool(mask.any()) else math.nan
                )
        k_valid = part_valid.sum(dim=1).detach().cpu().tolist()
        for bucket in ("k_1_2", "k_3_5", "k_6_10", "k_11_15", "k_16_plus"):
            obj_mask = torch.tensor(
                [k_bucket_name(int(k)) == bucket for k in k_valid],
                device=device,
                dtype=torch.bool,
            )
            metrics[f"mse_{bucket}"] = (
                float(per_object_loss[obj_mask].detach().mean().item()) if bool(obj_mask.any()) else math.nan
            )
        return loss, metrics


@torch.no_grad()
def sample_part_ss_latent(
    model,
    *,
    z_global: torch.Tensor,
    cond: torch.Tensor,
    mask_token_labels: torch.Tensor,
    part_valid: torch.Tensor,
    target_slots: torch.Tensor,
    part_token_weights: torch.Tensor | None = None,
    initial_noise: torch.Tensor | None = None,
    num_steps: int = 20,
    noise_scale: float = 1.0,
    latent_scale: float = 1.0,
    latent_norm_mode: str = "scalar",
    latent_mean: torch.Tensor | list | tuple | None = None,
    latent_std: torch.Tensor | list | tuple | None = None,
    self_conditioning: bool = False,
    cfg_scale: float = 1.0,
) -> torch.Tensor:
    if int(num_steps) <= 0:
        raise ValueError(f"num_steps must be > 0, got {num_steps}")
    cfg_scale = float(cfg_scale)
    if cfg_scale < 0:
        raise ValueError(f"cfg_scale must be >= 0, got {cfg_scale}")
    latent_norm_mode = str(latent_norm_mode)
    if latent_norm_mode not in ("scalar", "per_channel"):
        raise ValueError(f"unknown latent_norm_mode={latent_norm_mode!r}; expected 'scalar' or 'per_channel'")
    latent_scale = float(latent_scale)
    if latent_norm_mode == "scalar" and latent_scale <= 0:
        raise ValueError(f"latent_scale must be > 0, got {latent_scale}")
    if part_valid.dim() != 2:
        raise ValueError(f"part_valid must be [B,K], got {tuple(part_valid.shape)}")
    B = z_global.shape[0]
    if part_valid.shape[0] != B:
        raise ValueError(f"part_valid batch {part_valid.shape[0]} does not match z_global batch {B}")
    if target_slots.shape != part_valid.shape:
        raise ValueError(
            f"target_slots shape {tuple(target_slots.shape)} does not match "
            f"part_valid shape {tuple(part_valid.shape)}"
        )
    latent_channels = int(z_global.shape[1])
    mean_t = _normalize_latent_stats(latent_mean, name="latent_mean", latent_channels=latent_channels)
    std_t = _normalize_latent_stats(latent_std, name="latent_std", latent_channels=latent_channels)
    if latent_norm_mode == "per_channel":
        if mean_t is None or std_t is None:
            raise ValueError("latent_norm_mode='per_channel' requires both latent_mean and latent_std")
        if bool((std_t <= 0).any()):
            raise ValueError("latent_std must be > 0 for every channel")
    K = part_valid.shape[1]
    expected_shape = (B, K) + tuple(z_global.shape[1:])
    if initial_noise is None:
        x = torch.randn(
            expected_shape,
            device=z_global.device,
            dtype=z_global.dtype,
        )
    else:
        if tuple(initial_noise.shape) != expected_shape:
            raise ValueError(f"initial_noise shape {tuple(initial_noise.shape)} does not match expected {expected_shape}")
        x = initial_noise.to(device=z_global.device, dtype=z_global.dtype)
    x = x * float(noise_scale)
    valid_view = part_valid.to(device=z_global.device, dtype=z_global.dtype).view(B, K, 1, 1, 1, 1)
    x = x * valid_view
    dt = 1.0 / float(num_steps)
    # Self-conditioning carries the previous step's x0_hat estimate back in; it
    # starts as zeros (the model trains with a 0 self-cond on the no-double-pass
    # branch). CFG combines the conditional + null-condition velocities.
    use_self_cond = bool(self_conditioning)
    use_cfg = cfg_scale != 1.0
    x_self_cond = torch.zeros_like(x) if use_self_cond else None
    base_kwargs = {}
    if part_token_weights is not None:
        base_kwargs["part_token_weights"] = part_token_weights
    for i in range(int(num_steps)):
        t = torch.full((B,), i * dt, device=z_global.device, dtype=z_global.dtype)
        model_kwargs = dict(base_kwargs)
        if use_self_cond:
            model_kwargs["x_self_cond"] = x_self_cond
        v_cond = model(x, t, z_global, cond, mask_token_labels, part_valid, target_slots, **model_kwargs)
        if use_cfg:
            v_uncond = model(
                x, t, z_global, cond, mask_token_labels, part_valid, target_slots,
                drop_part_cond=True, **model_kwargs,
            )
            v = v_uncond + cfg_scale * (v_cond - v_uncond)
        else:
            v = v_cond
        if use_self_cond:
            # Endpoint estimate x0_hat in RF space (matches the training
            # double-pass in PartSSLatentRFLoss.__call__), fed back next step.
            t_view = t.view(B, 1, 1, 1, 1, 1)
            x_self_cond = ((x + (1.0 - t_view) * v) * valid_view).detach()
        x = (x + v * dt) * valid_view
    if latent_norm_mode == "per_channel":
        mean = _broadcast_latent_stats(mean_t, x)
        std = _broadcast_latent_stats(std_t, x)
        return (x * std + mean) * valid_view
    return (x / latent_scale) * valid_view
