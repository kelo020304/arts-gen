from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.eval.post.aggregate_joint_boundary_ab import aggregate_joint_boundary_ab


def _write_report(
    root: Path,
    *,
    object_id: str,
    mode: str,
    raw_miou: float,
    refined_miou: float,
    raw_boundary: float,
    refined_boundary: float,
    improved: int,
    regressed: int,
    direct_schema: bool = False,
) -> None:
    prefix = f"RealAppliance__{object_id}__angle_00"
    png_path = root / f"{prefix}__joint_boundary.png"
    png_path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (96, 72), (80, 130, 180)).save(png_path)
    if direct_schema:
        payload = {
            "joint_partition_path": str(root / object_id / "parts" / "joint_partition.npz"),
            "candidate_mode": mode,
            "sample": {"dataset_id": "RealAppliance", "object_id": object_id, "angle": 0},
            "class_names": ["body", "drawer"],
            "gt": {"multi_claim_ignore_voxels": 7},
            "candidate": {"recall": 1.0},
            "whole": {"iou": 0.91},
            "interface": {"pair_coverage": 0.98},
            "raw": {
                "mean_iou": raw_miou,
                "part_mean_iou": raw_miou - 0.04,
                "boundary_error": raw_boundary,
                "boundary_error_covered": raw_boundary * 0.5,
                "cross_label_same_pred_rate": 0.34,
                "predicted_to_gt_interface_ratio": 0.17,
            },
            "refined": {
                "mean_iou": refined_miou,
                "part_mean_iou": refined_miou - 0.04,
                "boundary_error": refined_boundary,
                "boundary_error_covered": refined_boundary * 0.5,
                "cross_label_same_pred_rate": 0.33,
                "predicted_to_gt_interface_ratio": 0.18,
            },
            "changed": {
                "voxels": improved + regressed,
                "improved": improved,
                "regressed": regressed,
                "neutral": 1,
            },
            "low_margin": {"low_margin_voxels": 12, "changed_voxels": 4},
            "artifacts": {"png": str(png_path.resolve())},
        }
    else:
        payload = {
            "dataset_id": "RealAppliance",
            "object_id": object_id,
            "angle": 0,
            "candidate_mode": mode,
            "stage": "partseg",
            "artifact_path": str(root / object_id / "parts" / "joint_partition.npz"),
            "png_path": png_path.name,
            "metrics": {
                "shared": {
                    "candidate_recall": 0.96,
                    "whole_iou": 0.91,
                    "interface_ratio": 0.18,
                },
                "raw": {
                    "mIoU": raw_miou,
                    "part_mIoU": raw_miou - 0.04,
                    "boundary_error": raw_boundary,
                    "boundary_error_covered": raw_boundary * 0.5,
                    "cross_same": 0.34,
                },
                "refined": {
                    "mIoU": refined_miou,
                    "part_mIoU": refined_miou - 0.04,
                    "boundary_error": refined_boundary,
                    "boundary_error_covered": refined_boundary * 0.5,
                    "cross_same": 0.33,
                },
                "delta": {},
            },
            "diagnostics": {
                "changed_voxels": improved + regressed,
                "improved": improved,
                "regressed": regressed,
                "neutral": 1,
                "low_margin_voxels": 12,
                "gt_overlap_voxels": 7,
                "class_names": ["body", "drawer"],
            },
        }
    (root / f"{prefix}__joint_boundary.json").write_text(
        json.dumps(payload),
        encoding="utf-8",
    )


def test_aggregate_joint_boundary_ab_writes_metrics_plot_and_object_panels(tmp_path: Path) -> None:
    proposal = tmp_path / "proposal"
    full_occ = tmp_path / "full_occ"
    for object_id, offset in (("003", 0.0), ("017", 0.02)):
        _write_report(
            proposal,
            object_id=object_id,
            mode="proposal",
            raw_miou=0.50 + offset,
            refined_miou=0.52 + offset,
            raw_boundary=0.40,
            refined_boundary=0.36,
            improved=8,
            regressed=2,
        )
        _write_report(
            full_occ,
            object_id=object_id,
            mode="full_occ",
            raw_miou=0.54 + offset,
            refined_miou=0.55 + offset,
            raw_boundary=0.38,
            refined_boundary=0.37,
            improved=5,
            regressed=3,
            direct_schema=True,
        )

    out_dir = tmp_path / "aggregate"
    result = aggregate_joint_boundary_ab(
        {"proposal": proposal, "full_occ": full_occ},
        out_dir,
        panel_width=220,
        panel_height=180,
    )

    assert result["report_count"] == 4
    assert result["unique_object_count"] == 2
    assert result["variants"]["proposal"]["refined"]["mIoU"]["mean"] == 0.53
    assert result["variants"]["proposal"]["diagnostics"]["improved"]["sum"] == 16.0
    assert result["pairwise"][0]["common_object_count"] == 2
    assert result["pairwise"][0]["right_better_object_count"]["mIoU"] == 2

    with (out_dir / "metrics_per_object.csv").open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 4
    assert {row["candidate_mode"] for row in rows} == {"proposal", "full_occ"}
    assert all(row["mIoU"] and row["boundary_error"] for row in rows)
    assert all(row["boundary_error_covered"] for row in rows)

    per_object = json.loads((out_dir / "metrics_per_object.json").read_text(encoding="utf-8"))
    assert {row["boundary_error_outcome"] for row in per_object} == {"improved"}
    direct_rows = [row for row in per_object if row["variant"] == "full_occ"]
    assert {row["object_key"] for row in direct_rows} == {
        "RealAppliance__003__angle_00",
        "RealAppliance__017__angle_00",
    }
    assert all(row["candidate_recall"] == 1.0 for row in direct_rows)
    assert all(row["whole_iou"] == 0.91 for row in direct_rows)
    assert all(row["interface_ratio"] == 0.18 for row in direct_rows)
    assert all(row["changed"] == 8.0 for row in direct_rows)
    assert all(row["low_margin_voxels"] == 12.0 for row in direct_rows)
    assert all(row["gt_overlap_voxels"] == 7.0 for row in direct_rows)
    assert all(row["png_exists"] for row in direct_rows)
    assert (out_dir / "aggregate.json").is_file()
    assert (out_dir / "aggregate.png").stat().st_size > 0
    assert (out_dir / "per_object_metrics__proposal.png").stat().st_size > 0
    assert (out_dir / "per_object_metrics__full_occ.png").stat().st_size > 0
    panels = sorted((out_dir / "object_boundary_panels").glob("*.png"))
    assert len(panels) == 2
    with Image.open(panels[0]) as panel:
        assert panel.size == (440, 218)
