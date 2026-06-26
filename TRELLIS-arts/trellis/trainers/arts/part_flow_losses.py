"""Generic flow-matching loss + sampler driven by a :class:`BaseCategoricalFlowBridge`.

Bridge-agnostic: works with any active bridge in
``trellis.models.part_flow.bridges`` — currently :class:`FisherBridge` and
:class:`GumbelSoftmaxBridge`. Default loss is endpoint CE, which is exactly
paper Eq. 11 for Gumbel-Softmax FM and the standard classifier-FM loss for
Fisher.

Workflow (per training step):
    1. Sample t ~ Uniform(0, t_max) per sample.
    2. Build ``x_1 = one_hot(per_voxel_labels, k_max)`` on valid dims only.
    3. Call ``bridge.sample_conditional_path(x_1, t, voxel_layout, num_parts)``
       to get ``x_t``.
    4. Call ``model(..., is_on_surface)`` to get
       endpoint logits.
    5. Apply weighted focal endpoint CE outside the bridge.

Sampling (inference): Euler ODE from t=0 to t=t_max, bridge-driven ``step``.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from trellis.models.part_flow.bridges import (
    BaseCategoricalFlowBridge,
    _build_part_valid_mask,
    _expand_valid_per_voxel,
)


__all__ = [
    'FlowMatchingLoss',
    'flow_sample',
    'weighted_focal_endpoint_ce',
]


def _build_one_hot_padded(
    per_voxel_labels: torch.Tensor,
    k_max: int,
    valid_per_voxel: torch.Tensor,
    dtype: torch.dtype = torch.float32,
    ignore_index: int = -1,
) -> torch.Tensor:
    """Per-voxel labels -> ``[N_total, k_max]`` one-hot on valid dims.

    Ignore-index rows get a temporary empty target for path sampling; their
    endpoint loss is masked out by ``weighted_focal_endpoint_ce``.
    """
    labels = per_voxel_labels.long()
    safe_labels = labels.masked_fill(labels == ignore_index, 0)
    safe_labels = safe_labels.clamp(min=0, max=k_max - 1)
    one_hot = F.one_hot(safe_labels, num_classes=k_max).to(dtype)
    # Defensive: if a label ever points into padding, zero it out.
    one_hot = one_hot * valid_per_voxel.to(dtype)
    return one_hot


def _class_weights(
    k_max: int,
    empty_weight: float,
    part_weight: float,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Build class weights where simplex slot 0 is empty."""
    weights = torch.full((k_max,), float(part_weight), device=device, dtype=dtype)
    weights[0] = float(empty_weight)
    return weights


def weighted_focal_endpoint_ce(
    endpoint_logits: torch.Tensor,
    x_1_idx: torch.Tensor,
    valid_per_voxel: torch.Tensor,
    class_weights: torch.Tensor,
    gamma: float = 2.0,
    ignore_index: int = -1,
    reduction: str = 'class_balanced',
    balance_group_ids: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Per-class weighted focal CE on endpoint logits.

    Padding slots are masked before log-softmax. Rows whose label is
    ``ignore_index`` are excluded from the mean.

    reduction:
        ``voxel_mean`` averages over supervised voxels, preserving the legacy
        dense-grid behavior.
        ``class_balanced`` first averages each present sample-local class, then
        averages those class losses. This prevents the 96%+ empty voxels in a
        dense 64^3 grid from hiding target-slot failures during overfit runs.
    """
    assert reduction in ('voxel_mean', 'class_balanced'), (
        f'unknown endpoint CE reduction {reduction!r}'
    )
    assert endpoint_logits.dim() == 2
    N, K = endpoint_logits.shape
    assert x_1_idx.shape == (N,)
    assert valid_per_voxel.shape == (N, K)
    assert class_weights.shape == (K,)
    if balance_group_ids is not None:
        assert balance_group_ids.shape == (N,), (
            f'balance_group_ids shape {tuple(balance_group_ids.shape)} '
            f'mismatch, expected [{N}]'
        )

    supervised = x_1_idx != ignore_index
    assert supervised.any(), 'weighted_focal_endpoint_ce received no supervised voxels'

    logits = endpoint_logits[supervised]
    labels = x_1_idx[supervised].clamp(min=0, max=K - 1).long()
    valid = valid_per_voxel[supervised]

    masked = logits.masked_fill(~valid, -1e4)
    log_probs = F.log_softmax(masked, dim=-1)
    log_pt = log_probs.gather(1, labels.unsqueeze(-1)).squeeze(-1)
    pt = log_pt.exp()
    weights = class_weights[labels].to(log_pt.dtype)
    loss_per_voxel = -weights * (1.0 - pt).pow(float(gamma)) * log_pt
    if reduction == 'voxel_mean':
        return loss_per_voxel.mean()

    if balance_group_ids is None:
        group_ids = labels
    else:
        group_ids = balance_group_ids[supervised].to(labels.device).long()
    group_losses = [
        loss_per_voxel[group_ids == group_id].mean()
        for group_id in torch.unique(group_ids, sorted=True)
    ]
    return torch.stack(group_losses).mean()


class FlowMatchingLoss(nn.Module):
    """Generic flow-matching loss for variable-K categorical part prediction.

    Args:
        bridge: concrete flow bridge instance (FisherBridge or
            GumbelSoftmaxBridge). Owns k_max, t_max, and path/step logic.
        parameterization: 'endpoint_logits' (default) — model outputs logits
            for p(x_1 | x_t), loss is masked CE. Future: 'velocity_mse' for
            direct velocity regression (bridge-specific, not default).
    """

    def __init__(
        self,
        bridge: BaseCategoricalFlowBridge,
        parameterization: str = 'endpoint_logits',
        empty_weight: float = 0.05,
        part_weight: float = 1.0,
        focal_gamma: float = 2.0,
        ignore_index: int = -1,
        reduction: str = 'class_balanced',
    ):
        super().__init__()
        assert parameterization in ('endpoint_logits',), \
            f'Unknown parameterization {parameterization}. ' \
            f'Only endpoint_logits is supported in this revision.'
        self.bridge = bridge
        self.parameterization = parameterization
        self.empty_weight = float(empty_weight)
        self.part_weight = float(part_weight)
        self.focal_gamma = float(focal_gamma)
        self.ignore_index = int(ignore_index)
        assert reduction in ('voxel_mean', 'class_balanced'), (
            f'unknown FlowMatchingLoss reduction {reduction!r}'
        )
        self.reduction = reduction

    def forward(
        self, model: nn.Module, batch: Dict[str, Any],
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute loss and metrics.

        Expected batch keys:
            coords              [N_total, 4] int32
            cond                [B, V*T, cond_dim]
            mask_token_labels   [B, V*T] int64
            per_voxel_labels    [N_total] int64 in {-1, 0..K_b}
            is_on_surface       [N_total] int64 in {0,1}
            voxel_layout        List[slice]  (per-sample row range in N_total)
            num_parts           List[int]    (per-sample K_b+1)
        """
        coords: torch.Tensor = batch['coords']
        cond: torch.Tensor = batch['cond']
        mask_token_labels: torch.Tensor = batch['mask_token_labels']
        per_voxel_labels: torch.Tensor = batch['per_voxel_labels']
        is_on_surface: torch.Tensor = batch['is_on_surface']
        voxel_layout: List[slice] = batch['voxel_layout']
        num_parts: List[int] = list(batch['num_parts'])
        B = cond.shape[0]
        device = cond.device
        k_max = self.bridge.k_max

        # --- Build valid masks (batch-level and per-voxel) ---
        part_valid = _build_part_valid_mask(num_parts, k_max, device)  # [B, k_max]
        valid_per_voxel = _expand_valid_per_voxel(part_valid, voxel_layout, coords.shape[0])

        # --- Build x_1 (one-hot on valid dims) ---
        labels_dev = per_voxel_labels.long().to(device)
        x_1 = _build_one_hot_padded(
            labels_dev, k_max, valid_per_voxel, ignore_index=self.ignore_index,
        ).to(device)

        # --- Sample t per sample, broadcast to voxels ---
        t_per_sample = torch.rand(B, device=device) * self.bridge.t_max
        batch_idx = coords[:, 0].long().to(device)
        t_per_voxel = t_per_sample[batch_idx]

        # --- Sample x_t along conditional path ---
        x_t = self.bridge.sample_conditional_path(
            x_1, t_per_voxel, voxel_layout, num_parts,
        )

        # --- Model forward (endpoint parameterization) ---
        out = model(
            x_t, t_per_sample, coords, cond, mask_token_labels, num_parts,
            is_on_surface,
        )
        endpoint_logits: torch.Tensor = out['endpoint_logits']

        # --- Loss ---
        class_weights = _class_weights(
            k_max, self.empty_weight, self.part_weight, device,
            dtype=endpoint_logits.dtype,
        )
        loss = weighted_focal_endpoint_ce(
            endpoint_logits, labels_dev, valid_per_voxel, class_weights,
            gamma=self.focal_gamma, ignore_index=self.ignore_index,
            reduction=self.reduction,
            balance_group_ids=batch_idx * k_max + labels_dev.clamp(min=0),
        )

        with torch.no_grad():
            supervised = labels_dev != self.ignore_index
            masked_logits = endpoint_logits.masked_fill(~valid_per_voxel, -1e4)
            probs = F.softmax(masked_logits[supervised], dim=-1)
            labels_sup = labels_dev[supervised]
            gt_prob = probs.gather(1, labels_sup.unsqueeze(-1)).squeeze(-1)
            pred = probs.argmax(dim=-1)
            body_slots = torch.tensor(num_parts, device=device).long()[batch_idx[supervised]] - 1
            empty_mask = labels_sup == 0
            body_mask = labels_sup == body_slots
            target_mask = (labels_sup > 0) & (labels_sup < body_slots)
            part_mask = labels_sup > 0
            metrics = {
                'loss': float(loss.item()),
                'gt_prob_mean': float(gt_prob.mean().item()),
                'endpoint_acc': float((pred == labels_sup).float().mean().item()),
                'empty_acc': float((pred[empty_mask] == 0).float().mean().item()) if empty_mask.any() else 0.0,
                'part_acc': float((pred[part_mask] == labels_sup[part_mask]).float().mean().item()) if part_mask.any() else 0.0,
                'target_acc': float((pred[target_mask] == labels_sup[target_mask]).float().mean().item()) if target_mask.any() else 0.0,
                'body_acc': float((pred[body_mask] == labels_sup[body_mask]).float().mean().item()) if body_mask.any() else 0.0,
                'empty_frac': float(empty_mask.float().mean().item()),
                'target_frac': float(target_mask.float().mean().item()),
                'body_frac': float(body_mask.float().mean().item()),
                'empty_weight': self.empty_weight,
                'part_weight': self.part_weight,
                'focal_gamma': self.focal_gamma,
                'ignore_frac': float((~supervised).float().mean().item()),
            }

        # Additional diagnostics
        with torch.no_grad():
            metrics['t_mean'] = t_per_sample.mean().item()
            metrics['x_t_simplex_err'] = (
                (x_t.sum(dim=-1) - valid_per_voxel.any(dim=-1).to(x_t.dtype)).abs().mean().item()
            )
        return loss, metrics


# --------------------------------------------------------------------------- #
# Sampler                                                                     #
# --------------------------------------------------------------------------- #


def _model_endpoint_logits_chunked(
    model: nn.Module,
    x_t: torch.Tensor,
    t_batch: torch.Tensor,
    coords: torch.Tensor,
    cond: torch.Tensor,
    mask_token_labels: torch.Tensor,
    num_parts: List[int],
    is_on_surface: torch.Tensor,
) -> torch.Tensor:
    """Call model in eval/inference voxel chunks.

    Training loss intentionally does not use this helper; it keeps the full
    dense forward path so supervision and gradient behavior stay simple.
    """
    chunk_size = int(getattr(model, 'voxel_chunk_size', 0) or 0)
    N_total = coords.shape[0]
    if chunk_size <= 0 or chunk_size >= N_total:
        return model(
            x_t, t_batch, coords, cond, mask_token_labels, num_parts,
            is_on_surface,
        )['endpoint_logits']

    assert cond.shape[0] == 1 and len(num_parts) == 1, (
        'chunked flow_sample currently supports eval/inference batch size 1; '
        'training forward remains full dense and unchunked'
    )
    assert torch.all(coords[:, 0] == 0), (
        'chunked flow_sample expects a single packed sample with batch_idx=0'
    )
    logits_chunks = []
    for start in range(0, N_total, chunk_size):
        end = min(start + chunk_size, N_total)
        out = model(
            x_t[start:end],
            t_batch,
            coords[start:end],
            cond,
            mask_token_labels,
            num_parts,
            is_on_surface[start:end],
        )
        logits_chunks.append(out['endpoint_logits'])
    return torch.cat(logits_chunks, dim=0)


@torch.no_grad()
def flow_sample(
    model: nn.Module,
    bridge: BaseCategoricalFlowBridge,
    coords: torch.Tensor,
    cond: torch.Tensor,
    mask_token_labels: torch.Tensor,
    voxel_layout: List[slice],
    num_parts: List[int],
    is_on_surface: torch.Tensor,
    num_steps: int = 20,
    solver: str = 'euler',
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Generic Euler ODE sampler.

    Args:
        model: trained PartFlowPredictor.
        bridge: the flow bridge used for training.
        coords, cond, mask_token_labels, voxel_layout, num_parts: same layout
            as training batch.
        num_steps: ODE integration steps.
        solver: 'euler' (default) or 'heun' (2nd-order predictor-corrector).

    Returns:
        labels: [N_total] int64, argmax over valid dims.
        soft:   [N_total, k_max] final soft probabilities (0 on padding).
    """
    assert solver in ('euler', 'heun'), f'Unsupported solver {solver}'
    device = cond.device
    k_max = bridge.k_max
    B = cond.shape[0]
    n_per = [sl.stop - sl.start for sl in voxel_layout]
    N_total = sum(n_per)
    assert coords.shape[0] == N_total

    # Source sample
    x_t = bridge.sample_source(num_parts, n_per, device)

    # Build valid mask for argmax later
    part_valid = _build_part_valid_mask(num_parts, k_max, device)
    valid_per_voxel = _expand_valid_per_voxel(part_valid, voxel_layout, N_total)

    t_grid = torch.linspace(0.0, bridge.t_max, num_steps + 1, device=device)
    for i in range(num_steps):
        t_curr = t_grid[i]
        t_next = t_grid[i + 1]
        dt = (t_next - t_curr).item()
        t_batch = torch.full((B,), t_curr.item(), device=device)

        endpoint_logits = _model_endpoint_logits_chunked(
            model, x_t, t_batch, coords, cond, mask_token_labels,
            num_parts, is_on_surface,
        )
        # endpoint probs (masked softmax — padding -> 0)
        endpoint_probs = F.softmax(endpoint_logits, dim=-1)
        endpoint_probs = endpoint_probs * valid_per_voxel.to(endpoint_probs.dtype)
        # Renormalize to sum=1 on valid dims (softmax over masked -inf should
        # already give this, but guard against tiny float drift)
        endpoint_probs = endpoint_probs / endpoint_probs.sum(
            dim=-1, keepdim=True,
        ).clamp(min=1e-8)

        if solver == 'euler':
            x_t = bridge.step(
                x_t, endpoint_probs, t_curr.item(), dt, voxel_layout, num_parts,
            )
        else:  # heun
            # Predict with Euler, then re-evaluate at midpoint for correction
            x_mid = bridge.step(
                x_t, endpoint_probs, t_curr.item(), dt, voxel_layout, num_parts,
            )
            endpoint_logits_mid = _model_endpoint_logits_chunked(
                model, x_mid, torch.full((B,), t_next.item(), device=device),
                coords, cond, mask_token_labels, num_parts, is_on_surface,
            )
            ep_mid = F.softmax(endpoint_logits_mid, dim=-1) * valid_per_voxel.to(x_mid.dtype)
            ep_mid = ep_mid / ep_mid.sum(dim=-1, keepdim=True).clamp(min=1e-8)
            # Average the two endpoint probs and take a single full step
            ep_avg = 0.5 * (endpoint_probs + ep_mid)
            x_t = bridge.step(
                x_t, ep_avg, t_curr.item(), dt, voxel_layout, num_parts,
            )

    # Argmax on valid dims; masked_fill makes padding -inf-like
    logits_like = x_t.masked_fill(~valid_per_voxel, -1.0)
    labels = logits_like.argmax(dim=-1).long()
    return labels, x_t
