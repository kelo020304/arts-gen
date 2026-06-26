"""Part SS latent flow with explicit 16^3 part-mask conditioning.

This file is intentionally separate from ``part_ss_latent_flow.py`` so the 0526
model/config path remains unchanged.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .part_ss_latent_flow import PartSSLatentFlowModel


__all__ = ["PartSSLatentFlowMask16Model"]


class PartSSLatentFlowMask16Model(PartSSLatentFlowModel):
    """Generate part SS latents conditioned on z_global and a target mask16.

    The mask is concatenated as a 3D channel next to the RF state:

        [x_t_part, target_mask16, optional z_global, optional x_self_cond]

    The model still predicts only the denoising velocity for ``x_t_part``.
    """

    def __init__(self, *args, mask16_condition: dict | None = None, **kwargs):
        self.mask16_condition_cfg = dict(mask16_condition or {})
        self.use_mask16_condition = bool(self.mask16_condition_cfg.get("enabled", True))
        self.mask16_binary_threshold = self.mask16_condition_cfg.get("binary_threshold", None)
        self.mask16_channels = 1 if self.use_mask16_condition else 0
        super().__init__(*args, **kwargs)
        if self.use_mask16_condition:
            self._expand_backbone_input_for_mask16()

    def _expand_backbone_input_for_mask16(self) -> None:
        old = self.backbone.input_layer
        old_in = int(self.backbone.in_channels)
        new_in = old_in + self.mask16_channels
        self.backbone.in_channels = new_in
        self.backbone.input_layer = nn.Linear(
            new_in * self.patch_size ** 3,
            self.model_channels,
            bias=old.bias is not None,
        )
        nn.init.xavier_uniform_(self.backbone.input_layer.weight)
        if self.backbone.input_layer.bias is not None:
            nn.init.zeros_(self.backbone.input_layer.bias)
        with torch.no_grad():
            cols = old.weight.shape[1]
            self.backbone.input_layer.weight[:, :cols].copy_(old.weight)
            if old.bias is not None:
                self.backbone.input_layer.bias.copy_(old.bias)

    def _normalize_part_mask16(
        self,
        part_mask16: torch.Tensor | None,
        *,
        x_t_parts: torch.Tensor,
    ) -> torch.Tensor | None:
        if not self.use_mask16_condition:
            return None
        if part_mask16 is None:
            raise ValueError("part_mask16 is required when mask16_condition.enabled=true")
        B, K, _C, R, H, W = x_t_parts.shape
        if part_mask16.dim() == 5:
            if tuple(part_mask16.shape) != (B, K, R, H, W):
                raise ValueError(
                    f"part_mask16 shape {tuple(part_mask16.shape)} must be {(B, K, R, H, W)}"
                )
            mask = part_mask16.unsqueeze(2)
        elif part_mask16.dim() == 6:
            if tuple(part_mask16.shape) != (B, K, 1, R, H, W):
                raise ValueError(
                    f"part_mask16 shape {tuple(part_mask16.shape)} must be {(B, K, 1, R, H, W)}"
                )
            mask = part_mask16
        else:
            raise ValueError(f"part_mask16 must be [B,K,R,R,R] or [B,K,1,R,R,R], got {tuple(part_mask16.shape)}")
        mask = mask.to(device=x_t_parts.device, dtype=x_t_parts.dtype)
        if self.mask16_binary_threshold is not None:
            mask = (mask > float(self.mask16_binary_threshold)).to(dtype=x_t_parts.dtype)
        return mask

    def _assemble_part_input(
        self,
        x_valid: torch.Tensor,
        z_valid: torch.Tensor,
        x_self_cond_valid: torch.Tensor | None,
        mask16_valid: torch.Tensor | None = None,
    ) -> torch.Tensor:
        parts = [x_valid]
        if self.use_mask16_condition:
            if mask16_valid is None:
                raise ValueError("mask16_valid is required when mask16_condition.enabled=true")
            if mask16_valid.dim() == 4:
                mask16_valid = mask16_valid.unsqueeze(1)
            if mask16_valid.shape[:2] != (x_valid.shape[0], 1) or mask16_valid.shape[2:] != x_valid.shape[2:]:
                raise ValueError(
                    f"mask16_valid shape {tuple(mask16_valid.shape)} does not match x_valid {tuple(x_valid.shape)}"
                )
            parts.append(mask16_valid.to(device=x_valid.device, dtype=x_valid.dtype))
        if self.concat_global:
            parts.append(z_valid)
        if self.self_conditioning:
            if x_self_cond_valid is None:
                x_self_cond_valid = torch.zeros_like(x_valid)
            parts.append(x_self_cond_valid)
        return torch.cat(parts, dim=1) if len(parts) > 1 else parts[0]

    def forward(
        self,
        x_t_parts: torch.Tensor,
        t: torch.Tensor,
        z_global: torch.Tensor,
        cond: torch.Tensor,
        mask_token_labels: torch.Tensor,
        part_valid: torch.Tensor,
        target_slots: torch.Tensor,
        part_token_weights: torch.Tensor | None = None,
        x_self_cond: torch.Tensor | None = None,
        drop_part_cond: bool | torch.Tensor | None = None,
        part_mask16: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if x_t_parts.dim() != 6:
            raise ValueError(f"x_t_parts must be [B,K,C,R,R,R], got {tuple(x_t_parts.shape)}")
        B, K, C, R, H, W = x_t_parts.shape
        if z_global.shape != (B, C, R, H, W):
            raise ValueError(
                f"z_global shape {tuple(z_global.shape)} does not match "
                f"x_t_parts object shape {(B, C, R, H, W)}"
            )
        if t.dim() != 1 or t.shape[0] != B:
            raise ValueError(f"t must be [B], got {tuple(t.shape)}")
        if part_valid.shape != (B, K):
            raise ValueError(f"part_valid must be {(B, K)}, got {tuple(part_valid.shape)}")
        if target_slots.shape != (B, K):
            raise ValueError(f"target_slots must be {(B, K)}, got {tuple(target_slots.shape)}")
        if x_self_cond is not None:
            if not self.self_conditioning:
                raise ValueError("x_self_cond provided but model.self_conditioning=false")
            if x_self_cond.shape != x_t_parts.shape:
                raise ValueError(
                    f"x_self_cond shape {tuple(x_self_cond.shape)} must match "
                    f"x_t_parts {tuple(x_t_parts.shape)}"
                )

        part_valid = part_valid.bool()
        if not bool(part_valid.any(dim=1).all()):
            raise ValueError("each object must contain at least one valid target part")
        valid_idx = part_valid.nonzero(as_tuple=False)
        if valid_idx.numel() == 0:
            raise ValueError("part_valid contains no valid target parts")
        part_mask16 = self._normalize_part_mask16(part_mask16, x_t_parts=x_t_parts)

        cond_tokens, part_queries = self.build_condition_tokens(
            cond,
            mask_token_labels,
            part_valid,
            target_slots,
            part_token_weights=part_token_weights,
        )
        global_tokens = self._encode_global_condition(z_global)
        drop_mask = self._normalize_drop_part_cond(drop_part_cond, part_valid)
        slot_ids = target_slots.clamp(0, self.part_label_vocab_size - 1)
        out = torch.zeros_like(x_t_parts)

        if self.cross_part_attention:
            if self.joint_flat_batch:
                return self._forward_joint_flat(
                    x_t_parts=x_t_parts,
                    t=t,
                    z_global=z_global,
                    cond_tokens=cond_tokens,
                    mask_token_labels=mask_token_labels,
                    part_queries=part_queries,
                    part_valid=part_valid,
                    target_slots=target_slots,
                    slot_ids=slot_ids,
                    global_tokens=global_tokens,
                    x_self_cond=x_self_cond,
                    drop_mask=drop_mask,
                    part_token_weights=part_token_weights,
                    part_mask16=part_mask16,
                    out=out,
                )
            return self._forward_joint(
                x_t_parts=x_t_parts,
                t=t,
                z_global=z_global,
                cond_tokens=cond_tokens,
                mask_token_labels=mask_token_labels,
                part_queries=part_queries,
                part_valid=part_valid,
                target_slots=target_slots,
                slot_ids=slot_ids,
                global_tokens=global_tokens,
                x_self_cond=x_self_cond,
                drop_mask=drop_mask,
                part_token_weights=part_token_weights,
                part_mask16=part_mask16,
                out=out,
            )
        return self._forward_independent(
            x_t_parts=x_t_parts,
            t=t,
            z_global=z_global,
            cond_tokens=cond_tokens,
            mask_token_labels=mask_token_labels,
            part_queries=part_queries,
            part_valid=part_valid,
            target_slots=target_slots,
            slot_ids=slot_ids,
            global_tokens=global_tokens,
            valid_idx=valid_idx,
            x_self_cond=x_self_cond,
            drop_mask=drop_mask,
            part_token_weights=part_token_weights,
            part_mask16=part_mask16,
            out=out,
        )

    def _forward_independent(
        self,
        *,
        x_t_parts,
        t,
        z_global,
        cond_tokens,
        mask_token_labels,
        part_queries,
        part_valid,
        target_slots,
        slot_ids,
        global_tokens,
        valid_idx,
        x_self_cond,
        drop_mask,
        part_token_weights,
        part_mask16,
        out,
    ) -> torch.Tensor:
        batch_idx = valid_idx[:, 0]
        part_idx = valid_idx[:, 1]
        chunk_size = self.max_part_forward_batch or int(valid_idx.shape[0])
        for start in range(0, int(valid_idx.shape[0]), chunk_size):
            end = min(start + chunk_size, int(valid_idx.shape[0]))
            chunk_batch_idx = batch_idx[start:end]
            chunk_part_idx = part_idx[start:end]
            x_valid = x_t_parts[chunk_batch_idx, chunk_part_idx]
            z_valid = z_global[chunk_batch_idx]
            sc_valid = x_self_cond[chunk_batch_idx, chunk_part_idx] if x_self_cond is not None else None
            mask_valid = part_mask16[chunk_batch_idx, chunk_part_idx] if part_mask16 is not None else None
            x_in = self._assemble_part_input(x_valid, z_valid, sc_valid, mask_valid)
            t_valid = t[chunk_batch_idx]
            chunk_drop = drop_mask[chunk_batch_idx, chunk_part_idx] if drop_mask is not None else None
            cond_memory = self._build_cond_memory_batch(
                cond_tokens,
                mask_token_labels,
                part_queries,
                part_valid,
                target_slots,
                global_tokens,
                chunk_batch_idx,
                chunk_part_idx,
                drop_part_cond=chunk_drop,
                part_token_weights=part_token_weights,
            )
            attn_bias = self._build_mask_attention_bias_batch(
                cond_tokens=cond_tokens,
                part_queries=part_queries,
                global_tokens=global_tokens,
                batch_idx=chunk_batch_idx,
                part_idx=chunk_part_idx,
                part_token_weights=part_token_weights,
            )
            if self.token_identity_embedding:
                pred_valid = self._backbone_with_identity(
                    x_in, t_valid, cond_memory, slot_ids[chunk_batch_idx, chunk_part_idx], attn_bias=attn_bias
                )
            elif attn_bias is None:
                pred_valid = self.backbone(x_in, t_valid, cond_memory)
            else:
                pred_valid = self.backbone(x_in, t_valid, cond_memory, attn_bias=attn_bias)
            out[chunk_batch_idx, chunk_part_idx] = pred_valid.to(dtype=out.dtype)
        return out

    def _forward_joint(
        self,
        *,
        x_t_parts,
        t,
        z_global,
        cond_tokens,
        mask_token_labels,
        part_queries,
        part_valid,
        target_slots,
        slot_ids,
        global_tokens,
        x_self_cond,
        drop_mask,
        part_token_weights,
        part_mask16,
        out,
    ) -> torch.Tensor:
        B = x_t_parts.shape[0]
        for b in range(B):
            valid_b = torch.nonzero(part_valid[b], as_tuple=False).flatten()
            if valid_b.numel() == 0:
                raise ValueError(f"object {b} has zero valid parts in joint forward")
            x_valid = x_t_parts[b, valid_b]
            z_valid = z_global[b : b + 1].expand(valid_b.numel(), -1, -1, -1, -1)
            sc_valid = x_self_cond[b, valid_b] if x_self_cond is not None else None
            mask_valid = part_mask16[b, valid_b] if part_mask16 is not None else None
            x_in = self._assemble_part_input(x_valid, z_valid, sc_valid, mask_valid)
            chunk_batch_idx = torch.full_like(valid_b, b)
            chunk_drop = drop_mask[b, valid_b] if drop_mask is not None else None
            cond_memory = self._build_cond_memory_batch(
                cond_tokens,
                mask_token_labels,
                part_queries,
                part_valid,
                target_slots,
                global_tokens,
                chunk_batch_idx,
                valid_b,
                drop_part_cond=chunk_drop,
                part_token_weights=part_token_weights,
            )
            attn_bias = self._build_mask_attention_bias_batch(
                cond_tokens=cond_tokens,
                part_queries=part_queries,
                global_tokens=global_tokens,
                batch_idx=chunk_batch_idx,
                part_idx=valid_b,
                part_token_weights=part_token_weights,
            )
            identity_slot_ids = slot_ids[b, valid_b] if self.token_identity_embedding else None
            summary_slot_ids = slot_ids[b, valid_b] if self.summary_cross_part_attention else None
            pred_valid = self._forward_object_joint(
                x_in,
                t[b],
                cond_memory,
                attn_bias=attn_bias,
                identity_slot_ids=identity_slot_ids,
                summary_slot_ids=summary_slot_ids,
            )
            out[b, valid_b] = pred_valid.to(dtype=out.dtype)
        return out

    def _forward_joint_flat(
        self,
        *,
        x_t_parts,
        t,
        z_global,
        cond_tokens,
        mask_token_labels,
        part_queries,
        part_valid,
        target_slots,
        slot_ids,
        global_tokens,
        x_self_cond,
        drop_mask,
        part_token_weights,
        part_mask16,
        out,
    ) -> torch.Tensor:
        valid_idx = part_valid.nonzero(as_tuple=False)
        batch_idx = valid_idx[:, 0]
        part_idx = valid_idx[:, 1]
        x_valid = x_t_parts[batch_idx, part_idx]
        z_valid = z_global[batch_idx]
        sc_valid = x_self_cond[batch_idx, part_idx] if x_self_cond is not None else None
        mask_valid = part_mask16[batch_idx, part_idx] if part_mask16 is not None else None
        x_in = self._assemble_part_input(x_valid, z_valid, sc_valid, mask_valid)
        t_valid = t[batch_idx]
        chunk_drop = drop_mask[batch_idx, part_idx] if drop_mask is not None else None
        cond_memory = self._build_cond_memory_batch(
            cond_tokens,
            mask_token_labels,
            part_queries,
            part_valid,
            target_slots,
            global_tokens,
            batch_idx,
            part_idx,
            drop_part_cond=chunk_drop,
            part_token_weights=part_token_weights,
        )
        attn_bias = self._build_mask_attention_bias_batch(
            cond_tokens=cond_tokens,
            part_queries=part_queries,
            global_tokens=global_tokens,
            batch_idx=batch_idx,
            part_idx=part_idx,
            part_token_weights=part_token_weights,
        )
        identity_slot_ids = slot_ids[batch_idx, part_idx] if self.token_identity_embedding else None
        summary_slot_ids = slot_ids[batch_idx, part_idx] if self.summary_cross_part_attention else None
        groups = self._object_groups(batch_idx)
        pred = self._run_flat_backbone(
            x_in, t_valid, cond_memory, attn_bias, groups, identity_slot_ids, summary_slot_ids
        )
        out[batch_idx, part_idx] = pred.to(dtype=out.dtype)
        return out
