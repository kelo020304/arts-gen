"""D-28/D-30: dual-export shape, dtype, and slicing contract.

Phase 9 09-10 update: source migrated from scripts/inference/export_part_flow.py
(deleted) to trellis.utils.arts.part_flow_postprocess. The export_batch helper
was inlined into pipeline/02_part_flow.py and is no longer a public symbol;
this test now only checks the shape/dtype/key contract of write_dual_output
and apply_postprocess.
"""

import numpy as np

from trellis.utils.arts.part_flow_postprocess import (
    apply_postprocess,
    write_dual_output,
)


def test_soft_file_key_is_probs(tmp_path):
    soft = np.zeros((64, 64, 64, 8), dtype=np.float16)
    hard = np.zeros((64, 64, 64), dtype=np.int64)
    soft_path, _ = write_dual_output(soft, hard, tmp_path)
    data = np.load(soft_path)
    assert set(data.files) == {'probs'}
    assert data['probs'].shape == soft.shape
    assert data['probs'].dtype == np.float16


def test_hard_file_shape_and_dtype(tmp_path):
    soft = np.zeros((64, 64, 64, 4), dtype=np.float16)
    hard = np.ones((64, 64, 64), dtype=np.int64)
    _, hard_path = write_dual_output(soft, hard, tmp_path)
    loaded = np.load(hard_path)
    assert loaded.shape == (64, 64, 64)
    assert loaded.dtype == np.int64


def test_hard_label_slicing_per_D30():
    labels = np.zeros((64, 64, 64), dtype=np.int64)
    labels[1:3, 2:4, 3:5] = 2
    mask = labels == 2
    assert mask.sum() == 8
    assert not (labels[mask] != 2).any()


def test_export_module_exports_symbols():
    # Both helpers must be importable callables (apply_postprocess +
    # write_dual_output). export_batch is no longer a public symbol —
    # batched inference is orchestrated by pipeline/02_part_flow.py.
    assert callable(apply_postprocess)
    assert callable(write_dual_output)
