import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[4] / "TRELLIS-arts"))
from inference_pipeline import part_prompt_seg_stage
from inference_pipeline import voxel_io


def test_model_args_from_promptable_ckpt_detects_route_and_semantic_classes():
    ckpt = {
        "args": {
            "route": "voxel",
            "dim": 384,
            "depth": 8,
            "head_depth": 2,
            "heads": 8,
            "voxel_depth": 3,
            "mask_encoder": "fg_points",
            "point_k_boundary": 32,
            "point_k_interior": 32,
            "point_resample_points": True,
        },
        "model": {"semantic_head.weight": torch.zeros(17, 384)},
    }

    args = part_prompt_seg_stage._model_args_from_ckpt(ckpt)

    assert args["dim"] == 384
    assert args["depth"] == 8
    assert args["use_voxel_head"] is True
    assert args["mask_encoder"] == "fg_points"
    assert args["point_resample_points"] is True
    assert args["semantic_classes"] == 17


def test_load_voxel_reads_part_npz_metadata(tmp_path):
    path = tmp_path / "voxel.npz"
    np.savez_compressed(
        path,
        coords=np.array([[1, 2, 3]], np.int32),
        resolution=np.int32(64),
        source="trellis_ss_flow",
    )

    voxel = voxel_io.load_voxel(path)

    assert voxel["coords"].shape == (1, 3)
    assert voxel["resolution"] == 64
    assert voxel["source"] == "trellis_ss_flow"
