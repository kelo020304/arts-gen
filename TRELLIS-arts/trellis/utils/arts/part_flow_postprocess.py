"""Optional hard-label cleanup + dual-output writers for Part Flow inference.

Migrated from scripts/inference/export_part_flow.py during Phase 9 09-10
hard cut. Consumed by:
  - TRELLIS-arts/inference.py::run_part_flow (when postprocess=True)
  - TRELLIS-arts/tests/arts/part_flow/test_postprocess.py
  - TRELLIS-arts/tests/arts/part_flow/test_output_contract.py

Behavior is byte-identical to the original — same morphological closing,
tiny-blob filter, and dual-output (soft_probs.npz + hard_labels.npy) format.
"""
from __future__ import annotations

from pathlib import Path
from typing import Tuple, Union

import numpy as np
from scipy import ndimage


def apply_postprocess(
    label_volume: np.ndarray,
    enabled: bool = False,
    tiny_blob_min_voxels: int = 5,
    morph_close_kernel: int = 3,
) -> np.ndarray:
    """Optional hard-label cleanup for inference-time export.

    Args:
        label_volume: ``[64, 64, 64]`` int64 — 0 = empty, 1..K = part slots.
        enabled: when False, returns a fresh copy of label_volume unchanged.
        tiny_blob_min_voxels: connected components with < this many voxels
            are dropped.
        morph_close_kernel: cubic kernel size for binary closing per part.
            Set to 1 to skip closing.

    Returns:
        ``[64, 64, 64]`` int64 with the same label range as the input.
    """
    assert label_volume.shape == (64, 64, 64), label_volume.shape
    assert label_volume.dtype == np.int64, label_volume.dtype
    if not enabled:
        return label_volume.copy()

    result = np.zeros_like(label_volume)
    labels = [int(v) for v in np.unique(label_volume).tolist() if int(v) > 0]
    structure = np.ones((morph_close_kernel,) * 3, dtype=bool)

    for label in labels:
        mask = label_volume == label
        if morph_close_kernel > 1:
            mask = ndimage.binary_closing(mask, structure=structure)
        cc, n_cc = ndimage.label(mask)
        if n_cc == 0:
            continue
        counts = np.bincount(cc.reshape(-1))
        keep_ids = np.nonzero(counts >= int(tiny_blob_min_voxels))[0]
        keep_ids = keep_ids[keep_ids != 0]
        if len(keep_ids) == 0:
            continue
        keep = np.isin(cc, keep_ids)
        result[(result == 0) & keep] = label
    return result.astype(np.int64, copy=False)


def write_dual_output(
    soft_probs: np.ndarray,
    hard_labels: np.ndarray,
    output_dir: Union[str, Path],
) -> Tuple[Path, Path]:
    """Write ``soft_probs.npz`` (key='probs') and ``hard_labels.npy``.

    D-28/D-30 dual-output contract.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    assert soft_probs.shape[:3] == (64, 64, 64), soft_probs.shape
    assert hard_labels.shape == (64, 64, 64), hard_labels.shape
    assert soft_probs.dtype == np.float16, soft_probs.dtype
    assert hard_labels.dtype == np.int64, hard_labels.dtype

    soft_path = output_dir / "soft_probs.npz"
    hard_path = output_dir / "hard_labels.npy"
    np.savez_compressed(soft_path, probs=soft_probs)
    np.save(hard_path, hard_labels)
    return soft_path, hard_path


__all__ = ["apply_postprocess", "write_dual_output"]
