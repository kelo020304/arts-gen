from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.eval.post.joint_boundary_diagnostics import (
    IMPROVED_COLOR,
    REGRESSED_COLOR,
    _projection_frame,
    _project_pixels,
    _render_projection_panel,
    run_joint_boundary_diagnostics,
)


def _save_coords(path: Path, coords: list[list[int]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, np.asarray(coords, dtype=np.int32))
    return path


def test_joint_boundary_metrics_use_full_grid_and_ignore_multi_claim(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    _save_coords(data_root / "surface.npy", [[x, 0, 0] for x in range(5)])
    _save_coords(data_root / "part_a.npy", [[1, 0, 0], [3, 0, 0]])
    _save_coords(data_root / "part_b.npy", [[2, 0, 0], [3, 0, 0]])

    run_dir = tmp_path / "run"
    parts_dir = run_dir / "parts"
    parts_dir.mkdir(parents=True)
    # Deliberately unsorted: logits and labels must retain artifact row order.
    coords = np.asarray([[2, 0, 0], [0, 0, 0], [3, 0, 0], [1, 0, 0]], dtype=np.int16)
    logits = np.asarray(
        [
            [0.0, 0.20, 0.10],  # raw A, low raw-logit gap; refined B (improved)
            [4.0, 0.00, 0.00],  # raw body
            [0.0, 0.00, 0.05],  # raw B, GT multi-claim; refined A (neutral)
            [0.0, 3.00, 0.00],  # raw A; refined B (regressed)
        ],
        dtype=np.float32,
    )
    labels_raw = np.asarray([1, 0, 2, 1], dtype=np.int16)
    labels_refined = np.asarray([2, 0, 1, 2], dtype=np.int16)
    refinement = {
        "enabled": True,
        "candidate_mode": "proposal",
        "margin_quantile": 0.01,
        "margin_threshold": 0.0,
        "raw_margin_quantile_threshold": 0.11,
    }
    joint_path = parts_dir / "joint_partition.npz"
    np.savez_compressed(
        joint_path,
        coords=coords,
        logits=logits.astype(np.float16),
        labels_raw=labels_raw,
        labels_refined=labels_refined,
        # This is intentionally bogus: diagnostics must recompute raw-logit gaps.
        probability_margin=np.ones((4,), dtype=np.float16),
        class_names=np.asarray(["part_body", "part_a", "part_b"]),
        refinement_json=np.asarray(json.dumps(refinement)),
    )
    np.savez_compressed(
        run_dir / "voxel.npz",
        coords=np.asarray([[0, 0, 0], [1, 0, 0], [2, 0, 0], [3, 0, 0], [5, 0, 0]], dtype=np.int16),
    )
    sample = {
        "_eval_dataset_id": "synthetic",
        "obj_id": "object-1",
        "angle_idx": 0,
        "surface_rel": "surface.npy",
        "parts": [
            {"part_name": "part_a", "raw_ind_rel": "part_a.npy"},
            {"part_name": "part_b", "raw_ind_rel": "part_b.npy"},
        ],
    }
    output_png = tmp_path / "diagnostic.png"
    output_json = tmp_path / "diagnostic.json"

    metrics = run_joint_boundary_diagnostics(
        joint_path,
        data_root=data_root,
        ds_sample=sample,
        output_png=output_png,
        output_json=output_json,
    )

    assert metrics["candidate_mode"] == "proposal"
    assert metrics["dataset_id"] == "synthetic"
    assert metrics["sample"]["dataset_id"] == "synthetic"
    assert metrics["gt"]["multi_claim_ignore_voxels"] == 1
    assert metrics["raw"]["per_class"]["part_a"]["gt_voxels"] == 1
    assert metrics["raw"]["per_class"]["part_b"]["gt_voxels"] == 1
    assert metrics["raw"]["per_class"]["part_b"]["pred_voxels"] == 0

    # Missing x=4 is a full-grid false negative, not silently dropped.
    assert metrics["candidate"]["recall"] == 0.75
    assert metrics["candidate"]["gt_whole_recall"] == 0.8
    assert metrics["whole"]["iou"] == 4 / 6
    assert metrics["whole"]["candidate_fallback"] is False
    assert metrics["raw"]["mean_iou"] == 1 / 3
    assert metrics["raw"]["part_mean_iou"] == 0.25

    assert metrics["interface"]["pairs"] == 2
    assert metrics["interface"]["pair_coverage"] == 1.0
    assert metrics["raw"]["boundary_error"] == pytest.approx(1 / 3)
    assert metrics["raw"]["boundary_candidate_coverage"] == 1.0
    assert metrics["raw"]["boundary_error_covered"] == pytest.approx(1 / 3)
    assert metrics["raw"]["cross_label_same_pred_rate"] == 0.5
    assert metrics["raw"]["predicted_interface_pairs"] == 2
    assert metrics["raw"]["predicted_to_gt_interface_ratio"] == 1.0

    assert metrics["changed"] == {
        "voxels": 3,
        "improved": 1,
        "regressed": 1,
        "neutral": 1,
        "net_improved": 0,
        "ignored_gt": 1,
    }
    assert metrics["low_margin"]["raw_logit_gap_threshold"] == 0.11
    assert metrics["low_margin"]["raw_logit_gap_threshold_source"] == "refinement_json"
    assert metrics["low_margin"]["voxels"] == 2
    assert metrics["low_margin"]["low_margin_voxels"] == 2
    assert metrics["artifact_checks"]["raw_label_logit_argmax_mismatch"] == 0

    raw_components = metrics["raw"]["components"]
    assert raw_components["part_tiny_components_le_8"] == 2
    assert raw_components["part_tiny_voxels_le_8"] == 3
    assert raw_components["per_class"]["part_a"]["tiny_components_le_8"] == 1

    assert output_png.is_file()
    assert output_json.is_file()
    from PIL import Image

    with Image.open(output_png) as image:
        assert image.size == (1720, 540)
        pixels = np.asarray(image.convert("RGB"), dtype=np.uint8).reshape(-1, 3)
    assert bool(np.all(pixels == np.asarray(IMPROVED_COLOR), axis=1).any())
    assert bool(np.all(pixels == np.asarray(REGRESSED_COLOR), axis=1).any())
    assert json.loads(output_json.read_text(encoding="utf-8"))["changed"]["neutral"] == 1


def test_projection_draws_internal_markers_after_base_voxels() -> None:
    coords = np.asarray([[10, 10, 10]], dtype=np.int32)
    image = _render_projection_panel(
        title="marker ordering",
        subtitle=(),
        base_layers=[(coords, (1, 2, 3), 15)],
        marker_layers=[(coords, IMPROVED_COLOR, 5)],
        frame_coords=coords,
        width=200,
        height=200,
    )
    frame = _projection_frame(coords, 200, 200)
    px, py, _depth = _project_pixels(coords, frame)
    center = (int(round(float(px[0]))), int(round(float(py[0]))))
    assert image.getpixel(center) == IMPROVED_COLOR
