"""Confidence-gated spatial refinement for joint part voxel labels."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn.functional as F


def _neighbor_offsets(neighborhood: int) -> list[tuple[int, int, int, float]]:
    if int(neighborhood) not in (6, 18, 26):
        raise ValueError(f"neighborhood must be 6, 18, or 26, got {neighborhood}")
    offsets: list[tuple[int, int, int, float]] = []
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            for dz in (-1, 0, 1):
                if dx == 0 and dy == 0 and dz == 0:
                    continue
                nonzero = int(dx != 0) + int(dy != 0) + int(dz != 0)
                if int(neighborhood) == 6 and nonzero != 1:
                    continue
                if int(neighborhood) == 18 and nonzero > 2:
                    continue
                if (dx, dy, dz) <= (0, 0, 0):
                    continue
                distance = math.sqrt(float(dx * dx + dy * dy + dz * dz))
                offsets.append((dx, dy, dz, 1.0 / distance))
    return offsets


def joint_neighbor_pairs(
    coords: torch.Tensor,
    *,
    neighborhood: int = 6,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return each occupied-grid neighbor pair once, in original coord order."""
    coords = torch.as_tensor(coords, dtype=torch.long)
    if coords.dim() != 2 or coords.shape[1] != 3:
        raise ValueError(f"coords expected [S,3], got {tuple(coords.shape)}")
    if coords.numel() == 0:
        empty = torch.empty((0,), dtype=torch.long, device=coords.device)
        return empty, empty, torch.empty((0,), dtype=torch.float32, device=coords.device)
    if int(coords.min().item()) < 0 or int(coords.max().item()) >= 64:
        raise ValueError("joint voxel coords must be inside [0,64)")

    keys = coords[:, 0] * 4096 + coords[:, 1] * 64 + coords[:, 2]
    order = torch.argsort(keys)
    sorted_keys = keys[order]
    sorted_coords = coords[order]
    sorted_ids = torch.arange(coords.shape[0], dtype=torch.long, device=coords.device)
    left: list[torch.Tensor] = []
    right: list[torch.Tensor] = []
    weights: list[torch.Tensor] = []
    for dx, dy, dz, pair_weight in _neighbor_offsets(int(neighborhood)):
        delta = coords.new_tensor((dx, dy, dz)).view(1, 3)
        query = sorted_coords + delta
        in_bounds = ((query >= 0) & (query < 64)).all(dim=1)
        if not bool(in_bounds.any().item()):
            continue
        source = sorted_ids[in_bounds]
        query = query[in_bounds]
        query_keys = query[:, 0] * 4096 + query[:, 1] * 64 + query[:, 2]
        pos = torch.searchsorted(sorted_keys, query_keys)
        present = pos < sorted_keys.shape[0]
        if not bool(present.any().item()):
            continue
        source = source[present]
        query_keys = query_keys[present]
        pos = pos[present]
        matched = sorted_keys[pos] == query_keys
        if not bool(matched.any().item()):
            continue
        a = order[source[matched]]
        b = order[pos[matched]]
        left.append(a)
        right.append(b)
        weights.append(torch.full((a.shape[0],), float(pair_weight), dtype=torch.float32, device=coords.device))
    if not left:
        empty = torch.empty((0,), dtype=torch.long, device=coords.device)
        return empty, empty, torch.empty((0,), dtype=torch.float32, device=coords.device)
    return torch.cat(left), torch.cat(right), torch.cat(weights)


@torch.no_grad()
def refine_joint_labels(
    logits: torch.Tensor,
    coords: torch.Tensor,
    *,
    enabled: bool = True,
    iterations: int = 1,
    pairwise_weight: float = 3.0,
    margin_threshold: float = 0.0,
    margin_quantile: float = 0.01,
    neighborhood: int = 6,
    min_vote_gain: float = 0.0,
    preserve_small_classes: int = 32,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Refine only low-confidence interface voxels while preserving unary cores.

    The confidence gate uses the top-2 softmax probability margin. Neighbor
    support is normalized by occupied-grid degree, so the pairwise strength is
    stable across surface thickness and neighborhood choices.
    """
    logits = torch.as_tensor(logits).float()
    coords = torch.as_tensor(coords, device=logits.device, dtype=torch.long)
    if logits.dim() != 2:
        raise ValueError(f"logits expected [S,K], got {tuple(logits.shape)}")
    if coords.shape != (logits.shape[0], 3):
        raise ValueError(f"coords shape {tuple(coords.shape)} must match logits {tuple(logits.shape)}")
    if logits.shape[1] <= 0:
        raise ValueError("joint logits must contain at least one class")
    if int(iterations) < 0:
        raise ValueError(f"iterations must be >= 0, got {iterations}")
    if float(pairwise_weight) < 0.0:
        raise ValueError(f"pairwise_weight must be >= 0, got {pairwise_weight}")
    if not 0.0 <= float(margin_threshold) <= 1.0:
        raise ValueError(f"margin_threshold must be in [0,1], got {margin_threshold}")
    if not 0.0 <= float(margin_quantile) <= 1.0:
        raise ValueError(f"margin_quantile must be in [0,1], got {margin_quantile}")
    if int(preserve_small_classes) < 0:
        raise ValueError(f"preserve_small_classes must be >= 0, got {preserve_small_classes}")

    probabilities = F.softmax(logits, dim=1)
    raw_labels = probabilities.argmax(dim=1)
    if logits.shape[1] > 1:
        top2 = torch.topk(probabilities, k=2, dim=1).values
        margins = top2[:, 0] - top2[:, 1]
    else:
        margins = probabilities.new_ones((probabilities.shape[0],))
    raw_top2 = torch.topk(logits, k=min(2, logits.shape[1]), dim=1).values
    raw_margins = (
        raw_top2[:, 0] - raw_top2[:, 1]
        if logits.shape[1] > 1
        else logits.new_full((logits.shape[0],), float("inf"))
    )
    ambiguous = margins < float(margin_threshold)
    quantile_threshold = None
    if float(margin_quantile) > 0.0 and raw_margins.numel() > 0:
        quantile_threshold = torch.quantile(raw_margins, float(margin_quantile))
        ambiguous |= raw_margins <= quantile_threshold
    before_counts = torch.bincount(raw_labels, minlength=logits.shape[1])
    small_locked = torch.zeros_like(ambiguous)
    if int(preserve_small_classes) > 0:
        small_classes = (before_counts > 0) & (before_counts <= int(preserve_small_classes))
        small_locked = small_classes[raw_labels]
        ambiguous &= ~small_locked
    pair_a, pair_b, pair_weights = joint_neighbor_pairs(coords, neighborhood=int(neighborhood))
    record: dict[str, Any] = {
        "enabled": bool(enabled),
        "iterations_requested": int(iterations),
        "iterations_run": 0,
        "pairwise_weight": float(pairwise_weight),
        "margin_threshold": float(margin_threshold),
        "margin_quantile": float(margin_quantile),
        "raw_margin_quantile_threshold": (
            None if quantile_threshold is None else float(quantile_threshold.detach().item())
        ),
        "min_vote_gain": float(min_vote_gain),
        "preserve_small_classes": int(preserve_small_classes),
        "neighborhood": int(neighborhood),
        "voxel_count": int(logits.shape[0]),
        "class_count": int(logits.shape[1]),
        "neighbor_pairs": int(pair_a.numel()),
        "ambiguous_voxels": int(ambiguous.sum().item()),
        "small_class_locked_voxels": int(small_locked.sum().item()),
        "interface_candidates": 0,
        "changed_voxels": 0,
        "class_counts_before": torch.bincount(raw_labels, minlength=logits.shape[1]).cpu().tolist(),
        "class_counts_after": torch.bincount(raw_labels, minlength=logits.shape[1]).cpu().tolist(),
    }
    if (
        not bool(enabled)
        or int(iterations) <= 0
        or float(pairwise_weight) <= 0.0
        or logits.shape[1] <= 1
        or pair_a.numel() == 0
        or not bool(ambiguous.any().item())
    ):
        return raw_labels, record

    unary = F.log_softmax(logits, dim=1)
    current = raw_labels.clone()
    interface_union = torch.zeros_like(ambiguous)
    for iteration in range(int(iterations)):
        votes = probabilities.new_zeros(probabilities.shape)
        degree = probabilities.new_zeros((probabilities.shape[0],))
        votes.index_put_((pair_a, current[pair_b]), pair_weights, accumulate=True)
        votes.index_put_((pair_b, current[pair_a]), pair_weights, accumulate=True)
        degree.index_add_(0, pair_a, pair_weights)
        degree.index_add_(0, pair_b, pair_weights)
        support = votes / degree.clamp_min(1.0).unsqueeze(1)

        interface = torch.zeros_like(ambiguous)
        disagreement = current[pair_a] != current[pair_b]
        if bool(disagreement.any().item()):
            interface[pair_a[disagreement]] = True
            interface[pair_b[disagreement]] = True
        interface_union |= interface
        proposed = (unary + float(pairwise_weight) * support).argmax(dim=1)
        row_ids = torch.arange(current.shape[0], device=current.device)
        vote_gain = support[row_ids, proposed] - support[row_ids, current]
        update = (
            ambiguous
            & interface
            & (proposed != current)
            & (vote_gain >= float(min_vote_gain))
        )
        if not bool(update.any().item()):
            record["iterations_run"] = int(iteration + 1)
            break
        current[update] = proposed[update]
        record["iterations_run"] = int(iteration + 1)

    # Never erase a class that had a prediction before refinement. Restore its
    # strongest unary seed if the local optimizer removed the entire class.
    after_counts = torch.bincount(current, minlength=logits.shape[1])
    for class_idx in range(int(logits.shape[1])):
        if int(before_counts[class_idx].item()) <= 0 or int(after_counts[class_idx].item()) > 0:
            continue
        donor_has_spare = after_counts[current] > 1
        preferred = torch.nonzero((raw_labels == class_idx) & donor_has_spare, as_tuple=False).flatten()
        candidates = preferred
        if candidates.numel() == 0:
            candidates = torch.nonzero(donor_has_spare, as_tuple=False).flatten()
        if candidates.numel() == 0:
            continue
        seed = candidates[probabilities[candidates, class_idx].argmax()]
        donor_class = int(current[seed].item())
        current[seed] = int(class_idx)
        after_counts[donor_class] -= 1
        after_counts[class_idx] += 1

    record["interface_candidates"] = int(interface_union.sum().item())
    record["changed_voxels"] = int((current != raw_labels).sum().item())
    record["class_counts_after"] = torch.bincount(current, minlength=logits.shape[1]).cpu().tolist()
    return current, record


def save_joint_partition(
    path: str | Path,
    *,
    coords: torch.Tensor,
    logits: torch.Tensor,
    labels: torch.Tensor,
    class_names: Sequence[str],
    refinement: dict[str, Any],
) -> Path:
    """Persist soft joint ownership for offline boundary ablations."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    logits_f = torch.as_tensor(logits).detach().float().cpu()
    raw_labels = logits_f.argmax(dim=1)
    probabilities = F.softmax(logits_f, dim=1)
    if logits_f.shape[1] > 1:
        top2 = torch.topk(probabilities, k=2, dim=1)
        margins = top2.values[:, 0] - top2.values[:, 1]
        second_labels = top2.indices[:, 1]
    else:
        margins = torch.ones((logits_f.shape[0],), dtype=torch.float32)
        second_labels = torch.zeros((logits_f.shape[0],), dtype=torch.long)
    np.savez_compressed(
        path,
        coords=torch.as_tensor(coords).detach().short().cpu().numpy(),
        logits=logits_f.half().numpy(),
        labels_raw=raw_labels.short().numpy(),
        labels_refined=torch.as_tensor(labels).detach().short().cpu().numpy(),
        second_labels=second_labels.short().numpy(),
        probability_margin=margins.half().numpy(),
        class_names=np.asarray([str(name) for name in class_names], dtype=np.str_),
        refinement_json=np.asarray(json.dumps(refinement, ensure_ascii=True, sort_keys=True), dtype=np.str_),
    )
    return path
