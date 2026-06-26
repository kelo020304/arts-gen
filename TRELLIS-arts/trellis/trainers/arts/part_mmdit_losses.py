"""Pure rectified-flow velocity MSE for PartMMDiT."""

from __future__ import annotations

import torch

from trellis.trainers.arts.part_ss_latent_flow_losses import (
    compute_object_loss_weights,
    compute_part_loss_weights,
)


__all__ = [
    "foreground_weighted_part_mse",
    "rectified_flow_loss",
]


def foreground_weighted_part_mse(
    sq_error: torch.Tensor,
    part_fg_mask: torch.Tensor | None,
    *,
    enabled: bool = True,
    bg_weight: float = 0.1,
    eps: float = 1.0e-6,
) -> torch.Tensor:
    """Reduce squared velocity error to per-part MSE with optional foreground weights."""

    if sq_error.dim() != 6:
        raise ValueError(f"sq_error expected [B,K,C,R,R,R], got {tuple(sq_error.shape)}")
    if not enabled:
        return sq_error.mean(dim=(2, 3, 4, 5))
    bg_weight = float(bg_weight)
    if bg_weight <= 0.0:
        raise ValueError(f"foreground_weight.bg_weight must be > 0, got {bg_weight}")
    if part_fg_mask is None:
        raise KeyError("part_fg_mask is required when foreground_weight.enabled=true")
    expected_shape = sq_error.shape[:2] + sq_error.shape[3:]
    if tuple(part_fg_mask.shape) != tuple(expected_shape):
        raise ValueError(
            f"part_fg_mask expected {tuple(expected_shape)}, got {tuple(part_fg_mask.shape)}"
        )
    weights = torch.where(
        part_fg_mask.to(device=sq_error.device).bool(),
        torch.ones((), device=sq_error.device, dtype=sq_error.dtype),
        torch.full((), bg_weight, device=sq_error.device, dtype=sq_error.dtype),
    ).unsqueeze(2)
    weights = weights.expand_as(sq_error)
    weighted_sum = (sq_error * weights).sum(dim=(2, 3, 4, 5))
    denom = weights.sum(dim=(2, 3, 4, 5)).clamp_min(float(eps))
    return weighted_sum / denom


def rectified_flow_loss(
    model,
    x_1: torch.Tensor,
    t: torch.Tensor,
    *,
    z_global: torch.Tensor,
    cond: torch.Tensor,
    name_tokens: torch.Tensor,
    name_mask: torch.Tensor,
    anchor: torch.Tensor,
    anchor_valid: torch.Tensor,
    part_valid: torch.Tensor,
    part_raw_voxel_counts: torch.Tensor,
    latent_scale: float,
    part_fg_mask: torch.Tensor | None = None,
    cfg_dropout_name: float = 0.1,
    cfg_dropout_anchor: float = 0.1,
    foreground_weight: dict | None = None,
    part_weight_kwargs: dict | None = None,
    object_balanced: bool = False,
    object_weight_kwargs: dict | None = None,
) -> torch.Tensor:
    """Compute PartMMDiT's pure RF velocity MSE.

    The global latent and image tokens are never dropped. Name and anchor CFG
    dropout are independent Bernoulli masks over valid part slots.
    """

    if x_1.dim() != 6:
        raise ValueError(f"x_1 expected [B,K,C,R,R,R], got {tuple(x_1.shape)}")
    batch_size, part_count = x_1.shape[:2]
    if tuple(part_valid.shape) != (batch_size, part_count):
        raise ValueError(
            f"part_valid expected {(batch_size, part_count)}, got {tuple(part_valid.shape)}"
        )
    if float(latent_scale) == 0.0:
        raise ValueError("latent_scale must be non-zero")

    x_1 = x_1 * float(latent_scale)
    x_0 = torch.randn_like(x_1)
    tt = t.view(batch_size, 1, 1, 1, 1, 1)
    x_t = (1.0 - tt) * x_0 + tt * x_1
    v_target = x_1 - x_0

    valid = part_valid.bool()
    drop_name = (torch.rand(batch_size, part_count, device=x_1.device) < float(cfg_dropout_name))
    drop_anchor = (
        torch.rand(batch_size, part_count, device=x_1.device) < float(cfg_dropout_anchor)
    )
    drop_name = drop_name & valid
    drop_anchor = drop_anchor & valid

    v_pred = model(
        x_t,
        t,
        z_global,
        cond,
        name_tokens,
        name_mask,
        anchor,
        anchor_valid,
        valid,
        drop_name=drop_name,
        drop_anchor=drop_anchor,
    )
    foreground_weight = foreground_weight or {"enabled": True, "bg_weight": 0.1}
    sq_error = foreground_weighted_part_mse(
        (v_pred - v_target) ** 2,
        part_fg_mask,
        enabled=bool(foreground_weight.get("enabled", True)),
        bg_weight=float(foreground_weight.get("bg_weight", 0.1)),
    )

    part_weight_kwargs = part_weight_kwargs or dict(
        mode="none",
        alpha=0.5,
        min_w=0.5,
        max_w=3.0,
        ref_mode="median",
        normalize_per_object=True,
    )
    part_weights, _stats = compute_part_loss_weights(
        part_raw_voxel_counts,
        valid,
        **part_weight_kwargs,
    )
    weights = part_weights * valid.float()

    if object_balanced:
        object_weight_kwargs = object_weight_kwargs or dict(
            mode="none",
            k_ref=None,
            min_w=0.75,
            max_w=2.0,
        )
        object_weights = compute_object_loss_weights(valid, **object_weight_kwargs)
        weights = weights * object_weights.view(batch_size, 1)

    return (sq_error * weights).sum() / weights.sum().clamp_min(1.0e-6)
