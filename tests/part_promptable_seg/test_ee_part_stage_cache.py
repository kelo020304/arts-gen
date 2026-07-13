from __future__ import annotations

import json
from pathlib import Path

from scripts.eval.tasks.ee_part_stage_cache import (
    PART_STAGE_SIGNATURE_NAME,
    build_part_stage_signature,
    clear_part_stage_outputs,
    part_stage_cache_status,
)


def _touch(path: Path, payload: bytes = b"x") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        path.write_bytes(payload)
    return path


def _signature(tmp_path: Path, **overrides):
    values = {
        "part_seg_ckpt": _touch(tmp_path / "joint.pt"),
        "ss_latent_path": _touch(tmp_path / "ss_latent.npy"),
        "voxel_path": _touch(tmp_path / "voxel.npz"),
        "expected_parts": 2,
        "joint_candidate_mode": "proposal",
        "joint_refine": False,
        "joint_refine_iters": 1,
        "joint_refine_pairwise": 3.0,
        "joint_refine_margin": 0.0,
        "joint_refine_margin_quantile": 0.01,
        "joint_refine_neighborhood": 6,
        "joint_refine_min_vote_gain": 0.0,
        "joint_refine_preserve_small_classes": 32,
        "joint_save_logits": False,
        "seed": None,
    }
    values.update(overrides)
    return build_part_stage_signature(**values)


def test_part_stage_signature_changes_with_refiner_contract(tmp_path: Path) -> None:
    raw = _signature(tmp_path)
    refined = _signature(tmp_path, joint_refine=True)
    full_occ = _signature(tmp_path, joint_candidate_mode="full_occ")

    assert raw != refined
    assert raw != full_occ
    assert raw["joint_partition"]["refine"] is False


def test_part_stage_cache_requires_matching_signature_and_logits(tmp_path: Path) -> None:
    parts_dir = tmp_path / "parts"
    _touch(parts_dir / "part_00_voxel.npz")
    _touch(parts_dir / "part_01_voxel.npz")
    signature = _signature(tmp_path, joint_save_logits=True)
    (parts_dir / PART_STAGE_SIGNATURE_NAME).write_text(json.dumps(signature), encoding="utf-8")

    ok, reason = part_stage_cache_status(parts_dir, expected_parts=2, expected_signature=signature)
    assert not ok
    assert reason == "joint_partition.npz was requested but is missing"

    _touch(parts_dir / "joint_partition.npz")
    ok, reason = part_stage_cache_status(parts_dir, expected_parts=2, expected_signature=signature)
    assert ok
    assert reason == "signature and outputs match"

    changed = _signature(tmp_path, joint_save_logits=True, joint_refine=True)
    ok, reason = part_stage_cache_status(parts_dir, expected_parts=2, expected_signature=changed)
    assert not ok
    assert reason == "part stage signature mismatch"


def test_clear_part_stage_outputs_keeps_unrelated_files(tmp_path: Path) -> None:
    parts_dir = tmp_path / "parts"
    owned = [
        _touch(parts_dir / "part_00_voxel.npz"),
        _touch(parts_dir / "part_body_voxel.npz"),
        _touch(parts_dir / "part_00_meta.json"),
        _touch(parts_dir / "joint_partition.npz"),
        _touch(parts_dir / PART_STAGE_SIGNATURE_NAME),
    ]
    keep = _touch(parts_dir / "overall.glb")

    removed = clear_part_stage_outputs(parts_dir)

    assert set(removed) == {path.name for path in owned}
    assert all(not path.exists() for path in owned)
    assert keep.is_file()
