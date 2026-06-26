"""Shared mask utilities for Part Flow training."""

from __future__ import annotations

import torch
import torch.nn.functional as F


__all__ = ['patch_aggregate_foreground_wins']


def patch_aggregate_foreground_wins(
    mask_2d: torch.Tensor,
    grid: int = 37,
    patch: int = 14,
    min_fg: int = 3,
) -> torch.Tensor:
    """Downsample a 2D part mask by per-patch foreground voting.

    Args:
        mask_2d: ``[H, W]`` integer tensor. ``0`` is background.
        grid: output grid side.
        patch: input patch side.
        min_fg: minimum non-background pixels required for a foreground label.

    Returns:
        ``[grid, grid]`` int64 tensor. Each output cell is the non-background
        mode label inside that patch, or 0 when foreground count is below
        ``min_fg``.
    """
    assert mask_2d.dim() == 2, f'expected [H,W], got {tuple(mask_2d.shape)}'
    mask_2d = mask_2d.long()

    target = grid * patch
    H, W = mask_2d.shape
    pad_h = max(0, target - H)
    pad_w = max(0, target - W)
    if pad_h or pad_w:
        mask_2d = F.pad(mask_2d, (0, pad_w, 0, pad_h), value=0)
    mask_2d = mask_2d[:target, :target]

    patches = mask_2d.view(grid, patch, grid, patch)
    patches = patches.permute(0, 2, 1, 3).contiguous().view(grid * grid, patch * patch)

    num_classes = max(int(patches.max().item()) + 1, 2)
    hist = F.one_hot(patches, num_classes=num_classes).sum(dim=1)
    fg_hist = hist[:, 1:]
    fg_count = fg_hist.sum(dim=1)
    fg_label = fg_hist.argmax(dim=1) + 1

    out = torch.where(
        fg_count >= int(min_fg),
        fg_label,
        torch.zeros_like(fg_label),
    )
    return out.view(grid, grid).long()
