"""
SparseTensor mask-based slicing utilities.

Provides two functions for extracting subsets of SparseTensor:
1. sparse_subset: boolean mask -> subset SparseTensor (hard split, for P0 / inference)
2. sparse_soft_split: soft masks -> list of weighted SparseTensors (differentiable, for decode-aware loss)

Usage:
    from trellis.utils.sparse_subset import sparse_subset, sparse_soft_split

    # Hard subset (GT part label or argmax mask)
    part_st = sparse_subset(z_slat, mask_bool, batch_idx=0)

    # Soft split (differentiable, K weighted copies)
    part_sts = sparse_soft_split(z_slat, soft_masks, batch_idx=0)

Notes:
    - SparseTensor coords format: [N, 4] where column 0 = batch_idx
    - _spatial_cache is NOT preserved (new topology invalidates cached convolution indices)
    - Designed for batch_size=1 (RTX 4090 constraint), but batch_idx arg allows extension
"""

from typing import List

import torch

from ..modules.sparse import SparseTensor


__all__ = ['sparse_subset', 'sparse_soft_split']


def sparse_subset(
    st: SparseTensor,
    mask: torch.Tensor,
    batch_idx: int = 0,
) -> SparseTensor:
    """Extract subset of a single-batch SparseTensor using boolean mask.

    Args:
        st: SparseTensor with at least one batch element.
        mask: Boolean tensor [N] where N = number of voxels in the
              specified batch element. True = keep voxel.
        batch_idx: Which batch element to subset (default 0).

    Returns:
        New SparseTensor with only masked voxels, batch_idx reset to 0.
        The _spatial_cache is NOT copied (new topology).
    """
    layout_slice = st.layout[batch_idx]
    feats_batch = st.feats[layout_slice]       # [N, C]
    coords_batch = st.coords[layout_slice]     # [N, 4]

    subset_feats = feats_batch[mask]            # [M, C]
    subset_coords = coords_batch[mask].clone()  # [M, 4]
    subset_coords[:, 0] = 0                     # Reset batch index

    return SparseTensor(feats=subset_feats, coords=subset_coords)


def sparse_soft_split(
    st: SparseTensor,
    soft_masks: torch.Tensor,
    batch_idx: int = 0,
) -> List[SparseTensor]:
    """Create K weighted copies of a SparseTensor using soft masks.

    For decode-aware loss (D-13): z_slat_k = z_slat * mask_k.
    All voxels are preserved in each copy; features are weighted by the
    soft mask so gradients flow back through the mask prediction head.

    Args:
        st: SparseTensor with at least one batch element.
        soft_masks: Float tensor [K, N] where K = number of parts,
                    N = number of voxels. Typically softmax output
                    (sum over K dim = 1 for each voxel).
        batch_idx: Which batch element to split (default 0).

    Returns:
        List of K SparseTensors, each with same coords but weighted feats.
        feats_k = original_feats * soft_masks[k].unsqueeze(-1)  # [N, C]
    """
    layout_slice = st.layout[batch_idx]
    feats_batch = st.feats[layout_slice]       # [N, C]
    coords_batch = st.coords[layout_slice]     # [N, 4]

    # Reset batch index once (shared across all K copies)
    coords_reset = coords_batch.clone()
    coords_reset[:, 0] = 0

    K = soft_masks.shape[0]
    result = []
    for k in range(K):
        weights = soft_masks[k].unsqueeze(-1)  # [N, 1]
        weighted_feats = feats_batch * weights  # [N, C]
        result.append(
            SparseTensor(feats=weighted_feats, coords=coords_reset.clone())
        )

    return result
