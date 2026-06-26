"""
Part Predictor loss functions: Hungarian matching + focal + dice + class CE + decode-aware Chamfer.

Components:
    HungarianMatcher: optimal bipartite matching (IoU + dice + class cost, Mask2Former weights)
    PartPredictorLoss: combined loss with configurable weights
    DecodeAwareLoss: frozen decoder per-part Chamfer loss (Plan 04-03)
    chamfer_distance: symmetric L1 Chamfer distance (differentiable)

Convention: All mask tensors use [K, N] layout (K=queries, N=voxels).

Training strategy (D-15):
    - Stage 1: sigmoid focal + dice + class CE (+ aux intermediate layer loss)
    - Stage 2: + decode-aware Chamfer loss (DecodeAwareLoss, YAML toggle)

V3 changes (query utilization fix, Mask2Former Section 3.3):
    - Mask loss: BCE → sigmoid focal loss (α=0.25, γ=2.0) — downweights easy voxels
    - Hungarian cost: IoU-only → IoU + dice + class (weights 5:5:2) — shape-sensitive matching
"""

from typing import Dict, List, Optional, Tuple

import json
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment


__all__ = ['HungarianMatcher', 'PartPredictorLoss', 'DecodeAwareLoss', 'chamfer_distance']


class HungarianMatcher:
    """Optimal bipartite matching between predicted and GT part masks.

    Uses IoU + dice + class cost matrix (Mask2Former Section 3.3 defaults)
    and scipy.optimize.linear_sum_assignment for O(n^3) Hungarian algorithm.

    Cost weights (Mask2Former defaults):
        cost_mask_weight=5.0 (IoU), cost_dice_weight=5.0, cost_class_weight=2.0

    Usage:
        matcher = HungarianMatcher()
        pred_idx, gt_idx = matcher(pred_masks, gt_masks)
        # pred_idx[i] is matched to gt_idx[i]
    """

    def __init__(
        self,
        cost_mask_weight: float = 5.0,
        cost_dice_weight: float = 5.0,
        cost_class_weight: float = 2.0,
    ):
        self.cost_mask_weight = cost_mask_weight
        self.cost_dice_weight = cost_dice_weight
        self.cost_class_weight = cost_class_weight

    @torch.no_grad()
    def __call__(
        self,
        pred_masks: torch.Tensor,
        gt_masks: torch.Tensor,
        pred_classes: Optional[torch.Tensor] = None,
        gt_classes: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute optimal matching between predictions and GT.

        Args:
            pred_masks: [K, N] predicted soft masks (after softmax).
            gt_masks: [M, N] GT binary masks (one-hot per part).
            pred_classes: [K, C+1] class logits (optional).
            gt_classes: [M] GT class indices (optional).

        Returns:
            (pred_idx, gt_idx): matched index tensors, each of length min(K, M).
        """
        K, N = pred_masks.shape
        M = gt_masks.shape[0]

        # intersection[k, m] = sum_n(pred[k,n] * gt[m,n])
        intersection = torch.einsum('kn,mn->km', pred_masks, gt_masks)  # [K, M]
        pred_area = pred_masks.sum(dim=1, keepdim=True)  # [K, 1]
        gt_area = gt_masks.sum(dim=1, keepdim=True).T     # [1, M]

        # IoU cost: negative IoU (minimize cost = maximize IoU)
        union = pred_area + gt_area - intersection         # [K, M]
        iou_cost = -(intersection / (union + 1e-6))        # [K, M]

        # Dice cost: 1 - 2*inter / (sum_pred + sum_gt)
        # Complements IoU — more sensitive to shape overlap, less biased by area
        dice_den = pred_area + gt_area                     # [K, M]
        dice_cost = 1.0 - 2.0 * intersection / (dice_den + 1e-6)  # [K, M]

        cost = (
            self.cost_mask_weight * iou_cost
            + self.cost_dice_weight * dice_cost
        )

        # Optional class cost
        if pred_classes is not None and gt_classes is not None:
            # CE cost: -log(P(correct class)) for each (pred_k, gt_m) pair
            pred_probs = F.softmax(pred_classes, dim=-1)  # [K, C+1]
            class_cost = -pred_probs[:, gt_classes]        # [K, M]
            cost = cost + self.cost_class_weight * class_cost

        # Scipy Hungarian algorithm
        cost_np = cost.cpu().numpy()
        row_ind, col_ind = linear_sum_assignment(cost_np)

        device = pred_masks.device
        return (
            torch.tensor(row_ind, dtype=torch.int64, device=device),
            torch.tensor(col_ind, dtype=torch.int64, device=device),
        )


class PartPredictorLoss(nn.Module):
    """Combined loss for Part Predictor training.

    Components:
        1. Sigmoid focal loss (on logits, matched pairs) — replaces BCE
        2. Dice loss (on soft masks, matched pairs)
        3. Class cross-entropy (matched pairs)
        4. Auxiliary intermediate layer loss (Mask2Former-style)

    Args:
        mask_weight: Weight for mask focal loss.
        dice_weight: Weight for dice loss.
        cls_weight: Weight for classification CE loss.
        aux_weight: Weight for auxiliary intermediate layer losses.
        focal_alpha: Focal loss alpha (class balance weight). Default 0.25.
        focal_gamma: Focal loss gamma (hard example focus). Default 2.0.
    """

    def __init__(
        self,
        mask_weight: float = 1.0,
        dice_weight: float = 2.0,
        cls_weight: float = 1.0,
        aux_weight: float = 1.0,
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0,
        decode_aware_cfg: Optional[dict] = None,
    ):
        super().__init__()
        self.mask_weight = mask_weight
        self.dice_weight = dice_weight
        self.cls_weight = cls_weight
        self.aux_weight = aux_weight
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        self.matcher = HungarianMatcher()

        # Decode-aware loss (optional, Plan 04-03)
        if decode_aware_cfg and decode_aware_cfg.get('enabled'):
            self.decode_aware = DecodeAwareLoss(
                decoder_ckpt=decode_aware_cfg['decoder_ckpt'],
                weight=decode_aware_cfg.get('weight', 0.5),
            )
        else:
            self.decode_aware = None

    def forward(
        self,
        pred,
        gt_labels,
        gt_type_ids,
        num_parts,
        z_slat_st=None,
        gt_points_per_part: Optional[List[torch.Tensor]] = None,
        dense_to_slat_idx=None,
        voxel_layout: Optional[List[slice]] = None,
    ) -> Dict[str, torch.Tensor]:
        """Compute matched losses. Supports B=1 (legacy dict) and B>=1 (list).

        Multi-batch args:
            pred: List[Dict] of length B (each dict keys: mask_logits, soft_masks,
                class_logits, query_embs), or a single dict for B=1 compat.
            gt_labels: [N_total] int64 packed labels (requires voxel_layout) OR
                List[LongTensor[N_b]] OR single [N] tensor.
            gt_type_ids: List[LongTensor[M_b]] OR single tensor for B=1.
            num_parts: List[int] OR int for B=1.
            z_slat_st: List[SparseTensor] OR single SparseTensor OR None.
            gt_points_per_part: List[List[Tensor]] OR List[Tensor] OR None.
            dense_to_slat_idx: List[LongTensor] OR single tensor OR None.
            voxel_layout: per-sample slice list (required when gt_labels is packed).

        Returns dict with keys averaged across batch:
            loss, mask_ce, dice, cls_ce (+ decode_aware_loss, chamfer_mean if enabled).
        """
        # ---- Normalize to List[...] ----
        if isinstance(pred, dict):
            pred_list = [pred]
            gt_labels_list = [gt_labels]
            gt_type_ids_list = [gt_type_ids]
            num_parts_list = [num_parts] if isinstance(num_parts, int) else list(num_parts)
            z_slat_list = [z_slat_st]
            gt_pts_list = [gt_points_per_part]
            d2s_list = [dense_to_slat_idx]
        else:
            pred_list = list(pred)
            B = len(pred_list)
            # gt_labels: packed [N_total] + voxel_layout, or list
            if isinstance(gt_labels, list):
                gt_labels_list = gt_labels
            elif voxel_layout is not None:
                gt_labels_list = [gt_labels[sl] for sl in voxel_layout]
            else:
                assert B == 1, "gt_labels must be list or packed+voxel_layout for B>1"
                gt_labels_list = [gt_labels]
            gt_type_ids_list = gt_type_ids if isinstance(gt_type_ids, list) else [gt_type_ids]
            num_parts_list = list(num_parts) if not isinstance(num_parts, int) else [num_parts]
            if z_slat_st is None or isinstance(z_slat_st, list):
                z_slat_list = z_slat_st if isinstance(z_slat_st, list) else [None] * B
            else:
                z_slat_list = [z_slat_st]
            if gt_points_per_part is None:
                gt_pts_list = [None] * B
            elif len(gt_points_per_part) == B and (len(gt_points_per_part) == 0 or isinstance(gt_points_per_part[0], list)):
                gt_pts_list = gt_points_per_part
            else:
                gt_pts_list = [gt_points_per_part]
            if dense_to_slat_idx is None:
                d2s_list = [None] * B
            elif isinstance(dense_to_slat_idx, list):
                d2s_list = dense_to_slat_idx
            else:
                d2s_list = [dense_to_slat_idx]

        B = len(pred_list)
        per_sample_results: List[Dict[str, torch.Tensor]] = []
        for b in range(B):
            per_sample_results.append(self._forward_single(
                pred_list[b],
                gt_labels_list[b],
                gt_type_ids_list[b],
                int(num_parts_list[b]),
                z_slat_list[b],
                gt_pts_list[b],
                d2s_list[b],
            ))

        # Aggregate: mean over samples for each scalar key, sum for others.
        device = pred_list[0]['mask_logits'].device
        keys = [k for k in per_sample_results[0].keys() if isinstance(per_sample_results[0][k], torch.Tensor)]
        agg: Dict[str, torch.Tensor] = {}
        for k in keys:
            vals = [r[k] for r in per_sample_results if k in r]
            if len(vals) == 0:
                continue
            agg[k] = torch.stack(vals).mean()

        # Attach per-sample chamfer lists if decode-aware
        if 'chamfer_per_part' in per_sample_results[0]:
            agg['chamfer_per_part'] = [r.get('chamfer_per_part', []) for r in per_sample_results]

        return agg

    def _sigmoid_focal_loss(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """Sigmoid focal loss (Lin et al., ICCV 2017).

        Downweights easy-to-classify voxels so the model focuses on hard
        boundary/small-part voxels. Replaces plain BCE for mask supervision.

        Args:
            logits: [*, N] raw logits (before sigmoid).
            targets: [*, N] binary targets {0, 1}.

        Returns:
            Scalar mean focal loss.
        """
        prob = logits.sigmoid()
        ce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        p_t = prob * targets + (1 - prob) * (1 - targets)
        focal_term = (1 - p_t).pow(self.focal_gamma)
        alpha_term = self.focal_alpha * targets + (1 - self.focal_alpha) * (1 - targets)
        return (alpha_term * focal_term * ce_loss).mean()

    def _forward_single(
        self,
        pred: Dict[str, torch.Tensor],
        gt_labels: torch.Tensor,
        gt_type_ids: torch.Tensor,
        num_parts: int,
        z_slat_st=None,
        gt_points_per_part: Optional[List[torch.Tensor]] = None,
        dense_to_slat_idx: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        device = pred['mask_logits'].device
        K, N = pred['mask_logits'].shape
        M = num_parts

        # Construct GT binary masks: [M, N]
        gt_masks = torch.zeros(M, N, device=device, dtype=torch.float32)
        for m in range(M):
            gt_masks[m] = (gt_labels == m).float()

        # Hungarian matching
        pred_idx, gt_idx = self.matcher(
            pred['soft_masks'], gt_masks,
            pred['class_logits'], gt_type_ids,
        )
        num_matched = len(pred_idx)

        if num_matched == 0:
            zero = torch.tensor(0.0, device=device, requires_grad=True)
            return {'loss': zero, 'mask_ce': zero, 'dice': zero, 'cls_ce': zero}

        # --- 1. Sigmoid focal loss (replaces BCE, Mask2Former Section 3.3) ---
        matched_logits = pred['mask_logits'][pred_idx]   # [num_matched, N]
        matched_gt = gt_masks[gt_idx]                     # [num_matched, N]
        mask_ce = self._sigmoid_focal_loss(matched_logits, matched_gt)

        # --- 2. Dice loss ---
        matched_soft = pred['soft_masks'][pred_idx]  # [num_matched, N]
        # Per-match dice, then average
        intersection = (matched_soft * matched_gt).sum(dim=1)       # [num_matched]
        cardinality = matched_soft.sum(dim=1) + matched_gt.sum(dim=1)  # [num_matched]
        dice_per_match = 2.0 * intersection / (cardinality + 1e-6)  # [num_matched]
        dice_loss = 1.0 - dice_per_match.mean()

        # --- 3. Classification cross-entropy ---
        matched_class_logits = pred['class_logits'][pred_idx]  # [num_matched, C+1]
        matched_gt_classes = gt_type_ids[gt_idx]                # [num_matched]
        cls_ce = F.cross_entropy(matched_class_logits, matched_gt_classes)

        # --- Total loss (basic) ---
        total_loss = (
            self.mask_weight * mask_ce
            + self.dice_weight * dice_loss
            + self.cls_weight * cls_ce
        )

        result = {
            'loss': total_loss,
            'mask_ce': mask_ce,
            'dice': dice_loss,
            'cls_ce': cls_ce,
        }

        # --- 4. Auxiliary loss from intermediate decoder layers (Mask2Former-style) ---
        aux_preds = pred.get('aux_outputs', [])
        if aux_preds and self.aux_weight > 0:
            aux_mask_ce_sum = torch.tensor(0.0, device=device)
            aux_dice_sum = torch.tensor(0.0, device=device)
            aux_cls_ce_sum = torch.tensor(0.0, device=device)
            num_aux = len(aux_preds)
            for aux_pred in aux_preds:
                aux_pred_idx, aux_gt_idx = self.matcher(
                    aux_pred['soft_masks'], gt_masks,
                    aux_pred['class_logits'], gt_type_ids,
                )
                if len(aux_pred_idx) == 0:
                    continue
                # Mask focal loss (same as main layer)
                aux_mask_ce_sum = aux_mask_ce_sum + self._sigmoid_focal_loss(
                    aux_pred['mask_logits'][aux_pred_idx],
                    gt_masks[aux_gt_idx],
                )
                # Dice
                aux_soft = aux_pred['soft_masks'][aux_pred_idx]
                aux_gt = gt_masks[aux_gt_idx]
                aux_inter = (aux_soft * aux_gt).sum(dim=1)
                aux_card = aux_soft.sum(dim=1) + aux_gt.sum(dim=1)
                aux_dice_sum = aux_dice_sum + (1.0 - (2.0 * aux_inter / (aux_card + 1e-6)).mean())
                # Class CE
                aux_cls_ce_sum = aux_cls_ce_sum + F.cross_entropy(
                    aux_pred['class_logits'][aux_pred_idx],
                    gt_type_ids[aux_gt_idx],
                )

            # Average over intermediate layers, then weight
            aux_mask_ce_avg = aux_mask_ce_sum / num_aux
            aux_dice_avg = aux_dice_sum / num_aux
            aux_cls_ce_avg = aux_cls_ce_sum / num_aux
            aux_total = (
                self.mask_weight * aux_mask_ce_avg
                + self.dice_weight * aux_dice_avg
                + self.cls_weight * aux_cls_ce_avg
            )
            total_loss = total_loss + self.aux_weight * aux_total
            result['loss'] = total_loss
            result['aux_mask_ce'] = aux_mask_ce_avg
            result['aux_dice'] = aux_dice_avg
            result['aux_cls_ce'] = aux_cls_ce_avg

        # --- Decode-aware loss (optional, Plan 04-03) ---
        if self.decode_aware is not None and z_slat_st is not None and gt_points_per_part is not None:
            da_dict = self.decode_aware(
                pred['soft_masks'], z_slat_st, gt_points_per_part,
                dense_to_slat_idx=dense_to_slat_idx,
            )
            total_loss = total_loss + da_dict['decode_aware_loss']
            result['loss'] = total_loss
            result.update(da_dict)

        return result


# ==============================================================================
# Decode-aware loss (Plan 04-03)
# ==============================================================================

def chamfer_distance(pred_pts: torch.Tensor, gt_pts: torch.Tensor) -> torch.Tensor:
    """Symmetric L1 Chamfer distance between two point clouds.

    Differentiable w.r.t. pred_pts (gradients flow back through Gaussian means).

    Args:
        pred_pts: [M, 3] predicted points (e.g. Gaussian._xyz means).
        gt_pts: [P, 3] ground-truth points (e.g. 64-cube voxel centers).

    Returns:
        Scalar: pred_to_gt_mean + gt_to_pred_mean (L1 Chamfer).
    """
    # Pairwise L2 distances: [M, P]
    dist = torch.cdist(pred_pts.unsqueeze(0), gt_pts.unsqueeze(0)).squeeze(0)  # [M, P]

    # Pred -> GT: for each pred point, min distance to any GT point
    pred_to_gt = dist.min(dim=1)[0].mean()

    # GT -> Pred: for each GT point, min distance to any pred point
    gt_to_pred = dist.min(dim=0)[0].mean()

    return pred_to_gt + gt_to_pred


class DecodeAwareLoss(nn.Module):
    """Decode-aware loss: frozen decoder per-part Chamfer (D-13, D-14, D-16).

    Loads a frozen SLatGaussianDecoder, splits z_slat by soft masks,
    decodes each part subset, computes Chamfer distance against GT point clouds.
    Gradients flow through the frozen decoder to soft_masks -> Part Predictor.

    Pitfall 1: NO torch.no_grad() around decoder call (gradients must flow through).
    Pitfall 5: Cast weighted feats to half() for FP16 decoder compatibility.
    Pitfall 6: Sequential per-part decode with torch.cuda.empty_cache() between parts.

    Args:
        decoder_ckpt: Base path to decoder checkpoint (without .json/.safetensors extension).
        weight: Loss weight multiplier (lambda_dec).
        device: Device for decoder.
    """

    def __init__(
        self,
        decoder_ckpt: str,
        weight: float = 0.5,
        device: str = 'cuda',
    ):
        super().__init__()
        self.weight = weight

        # Load frozen decoder
        self.decoder = self._load_frozen_decoder(decoder_ckpt, device)

    def _load_frozen_decoder(self, ckpt_base: str, device: str):
        """Load SLatGaussianDecoder with frozen parameters."""
        from safetensors.torch import load_file
        from trellis.models.structured_latent_vae.decoder_gs import SLatGaussianDecoder

        config_path = f'{ckpt_base}.json'
        weights_path = f'{ckpt_base}.safetensors'

        with open(config_path, 'r') as f:
            config = json.load(f)

        decoder = SLatGaussianDecoder(**config['args'])
        state_dict = load_file(weights_path)
        decoder.load_state_dict(state_dict, strict=True)
        decoder.to(device).eval()
        decoder.requires_grad_(False)

        print(f'[DecodeAwareLoss] Loaded frozen decoder: {config["name"]}')
        return decoder

    def forward(
        self,
        soft_masks: torch.Tensor,
        z_slat_st,
        gt_points_per_part: List[torch.Tensor],
        dense_to_slat_idx: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Compute per-part Chamfer loss through frozen decoder.

        Gradient chain: Chamfer -> Gaussian._xyz -> decoder feats ->
            soft_mask * z_slat -> soft_mask -> Part Predictor parameters.

        Args:
            soft_masks: [K, N_dense] from Part Predictor (after softmax, differentiable).
            z_slat_st: SparseTensor with z_slat features [N_slat, C] (treated as constant data).
            gt_points_per_part: List of K tensors, each [P_k, 3] float GT point clouds.
            dense_to_slat_idx: [N_dense] long, maps each dense voxel to its SLat
                index (or -1 if unmatched). Required because 64-cube active voxels
                and SLat active voxels are NOT the same subset (Review Round 1 #1).

        Returns:
            dict with: decode_aware_loss, chamfer_mean, chamfer_per_part.
        """
        from trellis.utils.sparse_subset import sparse_soft_split

        # Project soft_masks [K, N_dense] -> [K, N_slat] via alignment mapping
        # Only matched voxels contribute; unmatched dense voxels are dropped.
        N_slat = z_slat_st.feats.shape[0]
        K = soft_masks.shape[0]

        if dense_to_slat_idx is not None:
            matched_mask = (dense_to_slat_idx >= 0)  # [N_dense]
            matched_dense_idx = torch.where(matched_mask)[0]  # indices into dense
            matched_slat_idx = dense_to_slat_idx[matched_mask]  # indices into slat

            # Build projected soft_masks [K, N_slat] — default ZERO for unmatched.
            # Unmatched SLat voxels get weight 0 for ALL parts, so they are
            # effectively excluded from per-part decode (Review Round 2 fix:
            # 1/K uniform was wrong — it leaked shared geometry into every part).
            soft_masks_slat = torch.zeros(
                K, N_slat,
                device=soft_masks.device, dtype=soft_masks.dtype,
            )
            # Only matched SLat voxels get the predicted part assignment
            soft_masks_slat[:, matched_slat_idx] = soft_masks[:, matched_dense_idx]
        else:
            # Fallback: assume N_dense == N_slat and same order (legacy, unsafe)
            soft_masks_slat = soft_masks

        # Split z_slat by projected soft masks -> K weighted SparseTensors (D-13)
        weighted_sts = sparse_soft_split(z_slat_st, soft_masks_slat, batch_idx=0)

        K = len(weighted_sts)
        chamfer_losses = []
        device = soft_masks.device

        for k in range(K):
            weighted_st_k = weighted_sts[k]

            # FP16 dtype handling (Pitfall 5): frozen decoder uses convert_to_fp16
            weighted_st_k = weighted_st_k.replace(weighted_st_k.feats.half())

            # Decode through frozen decoder — NO torch.no_grad() (Pitfall 1)
            # Gradients flow through decoder to soft_masks
            gaussians_k = self.decoder(weighted_st_k)

            # Get Gaussian means as point cloud [M_k * 32, 3], differentiable
            pred_pts_k = gaussians_k[0].get_xyz  # [M_k*32, 3]

            # Get GT points for this part
            if k < len(gt_points_per_part):
                gt_pts_k = gt_points_per_part[k].to(device).float()

                # Cast pred back to float for stable Chamfer computation
                cd_k = chamfer_distance(pred_pts_k.float(), gt_pts_k)
                chamfer_losses.append(cd_k)

            # Memory cleanup between parts (Pitfall 6)
            del weighted_st_k, gaussians_k, pred_pts_k
            torch.cuda.empty_cache()

        if len(chamfer_losses) == 0:
            zero = torch.tensor(0.0, device=device, requires_grad=True)
            return {
                'decode_aware_loss': zero,
                'chamfer_mean': zero,
                'chamfer_per_part': [],
            }

        # Mean over K parts (D-14)
        chamfer_mean = torch.stack(chamfer_losses).mean()

        return {
            'decode_aware_loss': self.weight * chamfer_mean,
            'chamfer_mean': chamfer_mean,
            'chamfer_per_part': [cd.item() for cd in chamfer_losses],
        }
