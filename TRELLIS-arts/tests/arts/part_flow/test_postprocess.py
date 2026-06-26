"""D-29: postprocess tiny-blob filtering and morphological closing.

Phase 9 09-10 update: apply_postprocess migrated from
scripts/inference/export_part_flow.py (deleted) to
trellis.utils.arts.part_flow_postprocess.
"""

import numpy as np

from trellis.utils.arts.part_flow_postprocess import apply_postprocess


def test_tiny_blob_dropped():
    vol = np.zeros((64, 64, 64), dtype=np.int64)
    vol[10, 10, 10] = 1
    out = apply_postprocess(vol, enabled=True, tiny_blob_min_voxels=5, morph_close_kernel=1)
    assert out.sum() == 0


def test_morph_close_fills_hole():
    vol = np.zeros((64, 64, 64), dtype=np.int64)
    vol[20:25, 20:25, 20:25] = 2
    vol[22, 22, 22] = 0
    out = apply_postprocess(vol, enabled=True, tiny_blob_min_voxels=5, morph_close_kernel=3)
    assert out[22, 22, 22] == 2


def test_disabled_is_identity():
    vol = np.zeros((64, 64, 64), dtype=np.int64)
    vol[1, 2, 3] = 4
    out = apply_postprocess(vol, enabled=False)
    np.testing.assert_array_equal(out, vol)
