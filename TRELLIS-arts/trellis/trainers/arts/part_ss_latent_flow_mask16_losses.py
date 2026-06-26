"""RF loss and sampler for the mask16-conditioned part SS latent flow."""

from __future__ import annotations

import math
from typing import Any, Dict, Tuple

import torch
import torch.nn.functional as F

from .part_ss_latent_flow_losses import (
    PartSSLatentRFLoss,
    _broadcast_latent_stats,
    _normalize_latent_stats,
    compute_object_loss_weights,
    compute_part_loss_weights,
    foreground_weighted_part_mse,
    k_bucket_name,
    size_bucket_masks,
)


__all__ = ["PartSSLatentMask16RFLoss", "sample_part_ss_latent_mask16"]


class PartSSLatentMask16RFLoss(PartSSLatentRFLoss):
    """Same RF objective as the 0526 loss, with part_mask16 passed to the model."""

    @staticmethod
    def _shuffle_parts_with_mask16(
        x_1_raw: torch.Tensor,
        target_slots: torch.Tensor,
        part_token_weights: torch.Tensor | None,
        part_valid: torch.Tensor,
        part_raw_voxel_counts: torch.Tensor | None,
        part_fg_mask: torch.Tensor | None,
        part_mask16: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor,
    ]:
        x_1 = x_1_raw.clone()
        slots = target_slots.clone()
        weights = None if part_token_weights is None else part_token_weights.clone()
        counts = None if part_raw_voxel_counts is None else part_raw_voxel_counts.clone()
        fg_mask = None if part_fg_mask is None else part_fg_mask.clone()
        mask16 = part_mask16.clone()
        B, _K = part_valid.shape
        for b in range(B):
            valid_idx = torch.nonzero(part_valid[b], as_tuple=False).flatten()
            if valid_idx.numel() < 2:
                continue
            perm = valid_idx[torch.randperm(valid_idx.numel(), device=valid_idx.device)]
            x_1[b, valid_idx] = x_1_raw[b, perm]
            slots[b, valid_idx] = target_slots[b, perm]
            mask16[b, valid_idx] = part_mask16[b, perm]
            if weights is not None:
                weights[b, valid_idx] = part_token_weights[b, perm]
            if counts is not None:
                counts[b, valid_idx] = part_raw_voxel_counts[b, perm]
            if fg_mask is not None:
                fg_mask[b, valid_idx] = part_fg_mask[b, perm]
        return x_1, slots, weights, counts, fg_mask, mask16

    def __call__(self, model, batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, float]]:
        x_1_raw = batch["x_1_parts"]
        z_global = batch["z_global"]
        cond = batch["cond"]
        mask_token_labels = batch["mask_token_labels"]
        part_valid = batch["part_valid"].bool()
        target_slots = batch["target_slots"]
        if not bool(part_valid.any()):
            raise ValueError("part_valid contains no valid target parts")

        part_mask16 = batch.get("part_mask16")
        if part_mask16 is None:
            part_mask16 = batch.get("part_mask16_gt")
        if part_mask16 is None:
            raise KeyError("batch missing part_mask16 or part_mask16_gt")
        part_mask16 = part_mask16.to(device=x_1_raw.device, dtype=x_1_raw.dtype)

        part_token_weights = batch.get("part_token_weights")
        part_raw_voxel_counts = batch.get("part_raw_voxel_counts")
        part_fg_mask = self._part_fg_mask_from_batch(batch, part_valid, device=x_1_raw.device)

        if self.part_shuffle:
            x_1_raw, target_slots, part_token_weights, part_raw_voxel_counts, part_fg_mask, part_mask16 = (
                self._shuffle_parts_with_mask16(
                x_1_raw,
                target_slots,
                part_token_weights,
                part_valid,
                part_raw_voxel_counts,
                part_fg_mask,
                part_mask16,
            )
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

        base_model_kwargs: Dict[str, Any] = {"part_mask16": part_mask16}
        if part_token_weights is not None:
            base_model_kwargs["part_token_weights"] = part_token_weights

        drop_mask = self._sample_cfg_drop_mask(part_valid)
        cfg_dropped_parts = int(drop_mask.sum().item()) if drop_mask is not None else 0

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
            "mask16_condition_mean": float(part_mask16[part_valid].detach().float().mean().item()),
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
def sample_part_ss_latent_mask16(
    model,
    *,
    z_global: torch.Tensor,
    cond: torch.Tensor,
    mask_token_labels: torch.Tensor,
    part_valid: torch.Tensor,
    target_slots: torch.Tensor,
    part_mask16: torch.Tensor,
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
    use_self_cond = bool(self_conditioning)
    use_cfg = cfg_scale != 1.0
    x_self_cond = torch.zeros_like(x) if use_self_cond else None
    base_kwargs = {"part_mask16": part_mask16.to(device=z_global.device, dtype=z_global.dtype)}
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
            t_view = t.view(B, 1, 1, 1, 1, 1)
            x_self_cond = ((x + (1.0 - t_view) * v) * valid_view).detach()
        x = (x + v * dt) * valid_view
    if latent_norm_mode == "per_channel":
        mean = _broadcast_latent_stats(mean_t, x)
        std = _broadcast_latent_stats(std_t, x)
        return (x * std + mean) * valid_view
    return (x / latent_scale) * valid_view
