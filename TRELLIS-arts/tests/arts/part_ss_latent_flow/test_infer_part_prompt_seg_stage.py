import sys
from pathlib import Path

import numpy as np
import pytest
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


@pytest.mark.parametrize(
    ("route", "joint_seg", "candidate_mode", "refine", "save_logits"),
    [
        ("voxel", False, "proposal", True, False),
        ("voxel", False, "proposal", False, True),
        ("voxel", False, "full_occ", False, False),
        ("latent", False, "proposal", True, False),
    ],
)
def test_joint_partition_options_require_joint_voxel_checkpoint(
    route,
    joint_seg,
    candidate_mode,
    refine,
    save_logits,
):
    with pytest.raises(ValueError, match="args.joint_seg=true"):
        part_prompt_seg_stage._validate_joint_partition_request(
            route=route,
            joint_seg=joint_seg,
            candidate_mode=candidate_mode,
            refine=refine,
            save_logits=save_logits,
        )


def test_joint_partition_defaults_remain_valid_for_legacy_checkpoint():
    part_prompt_seg_stage._validate_joint_partition_request(
        route="voxel",
        joint_seg=False,
        candidate_mode="proposal",
        refine=False,
        save_logits=False,
    )


def test_joint_partition_options_are_valid_for_joint_voxel_checkpoint():
    part_prompt_seg_stage._validate_joint_partition_request(
        route="voxel",
        joint_seg=True,
        candidate_mode="full_occ",
        refine=True,
        save_logits=True,
    )
