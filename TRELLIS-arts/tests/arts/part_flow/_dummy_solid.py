"""Dummy solid-voxel + surface-index generators for Phase 8 tests.

Generated during Plan 08-01 so downstream code-first tasks can run in CI
before the real solid voxel data arrives from the data team.

Contract (matches Phase 8 expectations):
  - Solid volume: shape [R, R, R], dtype int64, 0=empty and 1..K=parts
  - Surface indices: [M, 3] int32, outer-shell voxels of the solid
"""

from __future__ import annotations

import numpy as np


def make_dummy_solid(
    resolution: int = 64,
    num_parts: int = 5,           # K_b real parts (excluding empty)
    empty_ratio: float = 0.95,    # ~95% voxels empty
    seed: int = 42,
) -> np.ndarray:
    """Return [resolution,resolution,resolution] int64 array.

    0 = empty, 1..num_parts = part labels. Dense 5% occupied as blobby parts.
    """
    rng = np.random.default_rng(seed)
    vol = np.zeros((resolution,) * 3, dtype=np.int64)
    total_voxels = resolution ** 3
    target_fg = int(total_voxels * (1.0 - empty_ratio))
    # Sample num_parts blob centers and fill a small sphere around each
    centers = rng.integers(8, resolution - 8, size=(num_parts, 3))
    per_part_budget = target_fg // num_parts
    for pid in range(1, num_parts + 1):
        cx, cy, cz = centers[pid - 1]
        # Expand radius until we cover per_part_budget voxels
        for r in range(1, resolution):
            x, y, z = np.ogrid[:resolution, :resolution, :resolution]
            mask = (x - cx) ** 2 + (y - cy) ** 2 + (z - cz) ** 2 <= r * r
            if mask.sum() >= per_part_budget:
                vol[mask & (vol == 0)] = pid
                break
    return vol


def make_dummy_surface_indices(solid_volume: np.ndarray) -> np.ndarray:
    """Return [M,3] int32 surface voxel indices (outer shell of solid parts).

    Simulates what ``allind.npy`` looks like for the dummy solid.
    """
    import scipy.ndimage as ndi
    mask = solid_volume > 0
    eroded = ndi.binary_erosion(mask, iterations=1)
    surface = mask & (~eroded)
    idx = np.array(np.nonzero(surface)).T.astype(np.int32)  # [M, 3]
    return idx
