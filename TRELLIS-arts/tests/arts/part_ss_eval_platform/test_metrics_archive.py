import json
from pathlib import Path

import pytest

from part_ss_eval_platform.archive import ExperimentArchive, safe_artifact_path
from part_ss_eval_platform.metrics import load_eval_metrics


def _write_json(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def _write_jsonl(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )


def _make_eval_run(root: Path) -> Path:
    run = root / "part_ss_latent_flow" / "eval_a"
    report = run / "full_eval" / "step_000200"
    _write_json(
        report / "summary.json",
        {
            "overall": {
                "parts": 3,
                "objects": 2,
                "part_iou_mean": 0.5,
                "target_parts_iou_mean": 0.7,
            },
            "by_size": {},
        },
    )
    _write_jsonl(
        report / "part_metrics.jsonl",
        [
            {
                "obj_id": "1",
                "dataset_index": 1,
                "target_part_name": "small_handle",
                "object_part_count": 7,
                "part_raw_voxel_count": 100,
                "raw_ind_count": 100,
                "part_iou": 0.2,
                "part_recall": 0.25,
                "part_precision": 0.5,
                "pred_count": 0,
                "assignment_offdiag_max": 0.30,
                "assignment_diag_iou": 0.10,
            },
            {
                "obj_id": "1",
                "dataset_index": 1,
                "target_part_name": "large_door",
                "object_part_count": 7,
                "part_raw_voxel_count": 5000,
                "raw_ind_count": 5000,
                "part_iou": 0.8,
                "part_recall": 0.9,
                "part_precision": 0.75,
                "pred_count": 4500,
                "assignment_offdiag_max": 0.10,
                "assignment_diag_iou": 0.85,
            },
            {
                "obj_id": "2",
                "dataset_index": 2,
                "target_part_name": "medium_drawer",
                "object_part_count": 2,
                "part_raw_voxel_count": 1000,
                "raw_ind_count": 1000,
                "part_iou": 0.5,
                "part_recall": 0.6,
                "part_precision": 0.4,
                "pred_count": 1200,
                "assignment_offdiag_max": 0.20,
                "assignment_diag_iou": 0.60,
            },
        ],
    )
    _write_jsonl(
        report / "object_metrics.jsonl",
        [
            {
                "obj_id": "1",
                "dataset_index": 1,
                "object_part_count": 7,
                "target_parts_iou_pred_vs_gt": 0.55,
                "target_parts_precision_pred_vs_gt": 0.70,
                "target_parts_recall_pred_vs_gt": 0.65,
                "part_iou_min": 0.2,
                "size_mix_ratio": 50.0,
            },
            {
                "obj_id": "2",
                "dataset_index": 2,
                "object_part_count": 2,
                "target_parts_iou_pred_vs_gt": 0.85,
                "target_parts_precision_pred_vs_gt": 0.80,
                "target_parts_recall_pred_vs_gt": 0.90,
                "part_iou_min": 0.5,
                "size_mix_ratio": 1.0,
            },
        ],
    )
    _write_json(
        report / "selected_examples.json",
        [
            {
                "group": "worst_small_parts",
                "obj_id": "1",
                "dataset_index": 1,
                "label": "small_handle",
                "metric": 0.2,
                "png": "voxel_examples/shared/000001_target_parts.png",
            }
        ],
    )
    (report / "plots").mkdir(parents=True)
    (report / "plots" / "iou_by_size.png").write_bytes(b"png")
    (report / "voxel_examples" / "shared").mkdir(parents=True)
    (report / "voxel_examples" / "shared" / "000001_target_parts.png").write_bytes(b"png")
    (run / "full_eval.log").write_text("done\n", encoding="utf-8")
    return run


def _make_test_run(root: Path) -> Path:
    run = root / "part_ss_latent_flow" / "test_b"
    _write_json(
        run / "examples" / "index.json",
        {
            "examples": [
                {
                    "example_id": "000001",
                    "obj_id": "1",
                    "parts": [{"name": "door", "pred_slat": "pred_slat/00_door.pt"}],
                }
            ]
        },
    )
    (run / "test_export.log").write_text("export done\n", encoding="utf-8")
    return run


def test_load_eval_metrics_computes_research_metrics(tmp_path):
    report = _make_eval_run(tmp_path) / "full_eval" / "step_000200"

    metrics = load_eval_metrics(report)

    assert metrics["task_kind"] == "eval"
    assert metrics["overall"]["target_iou"]["value"] == pytest.approx(0.7)
    assert metrics["overall"]["part_iou"]["value"] == pytest.approx(0.5)
    mean_precision = (0.5 + 0.75 + 0.4) / 3
    mean_recall = (0.25 + 0.9 + 0.6) / 3
    assert metrics["overall"]["f1"]["value"] == pytest.approx(
        2 * mean_precision * mean_recall / (mean_precision + mean_recall)
    )
    assert metrics["overall"]["count_error"]["value"] == pytest.approx((1.0 + 0.1 + 0.2) / 3)
    assert metrics["overall"]["empty_rate"]["value"] == pytest.approx(1 / 3)
    assert metrics["overall"]["confusion"]["value"] == pytest.approx(0.2)
    assert metrics["focused"]["small_recall"]["value"] == pytest.approx(0.25)
    assert metrics["focused"]["small_empty_rate"]["value"] == pytest.approx(1.0)
    assert metrics["focused"]["multi_target_iou"]["value"] == pytest.approx(0.55)
    assert metrics["focused"]["multi_worst_part_iou"]["value"] == pytest.approx(0.2)
    assert metrics["focused"]["size_gap_target_iou"]["value"] == pytest.approx(0.55)
    assert metrics["size_buckets"]["small"]["definition"] == "raw_count < 500"
    assert metrics["metric_definitions"]["target_iou"]["formula"]


def test_load_eval_metrics_exposes_binding_diag(tmp_path):
    """The diagnose-style binding signal (assignment diag/off-diag) must be a
    first-class platform metric, with a small-part breakout (diag UP / off-diag
    DOWN is how binding correctness is judged, especially in the small bucket)."""
    report = _make_eval_run(tmp_path) / "full_eval" / "step_000200"

    metrics = load_eval_metrics(report)

    # diag IoU (higher = correct part claims its own region)
    assert metrics["overall"]["binding_diag"]["value"] == pytest.approx((0.10 + 0.85 + 0.60) / 3)
    # off-diag (confusion) still exposed, unchanged
    assert metrics["overall"]["confusion"]["value"] == pytest.approx((0.30 + 0.10 + 0.20) / 3)
    # small-bucket breakout = only the small_handle part (raw_count 100 < 500)
    assert metrics["focused"]["small_binding_diag"]["value"] == pytest.approx(0.10)
    assert metrics["focused"]["small_confusion"]["value"] == pytest.approx(0.30)
    # metric metadata present for the UI
    assert metrics["metric_definitions"]["binding_diag"]["formula"]
    assert metrics["metric_definitions"]["small_binding_diag"]["meaning"]
    assert metrics["metric_definitions"]["small_confusion"]["meaning"]


def test_archive_scans_eval_and_test_runs_and_exposes_artifacts(tmp_path):
    eval_run = _make_eval_run(tmp_path)
    test_run = _make_test_run(tmp_path)

    archive = ExperimentArchive([tmp_path])
    experiments = archive.list_experiments()
    by_name = {item["name"]: item for item in experiments}

    assert by_name["eval_a"]["kind"] == "eval"
    assert by_name["eval_a"]["report_dir"].endswith("full_eval/step_000200")
    assert by_name["test_b"]["kind"] == "test"
    assert by_name["test_b"]["export_dir"].endswith("examples")

    detail = archive.get_experiment(by_name["eval_a"]["id"])
    assert detail["metrics"]["overall"]["target_iou"]["value"] == pytest.approx(0.7)
    assert detail["artifacts"]["plots"][0]["path"] == "plots/iou_by_size.png"
    assert detail["examples"][0]["png"] == "voxel_examples/shared/000001_target_parts.png"

    assert safe_artifact_path(eval_run, "full_eval.log") == eval_run / "full_eval.log"
    with pytest.raises(ValueError, match="outside experiment root"):
        safe_artifact_path(eval_run, "../secret.txt")
    with pytest.raises(KeyError):
        archive.get_experiment("missing")


def test_archive_reads_test_export_summary(tmp_path):
    _make_test_run(tmp_path)

    archive = ExperimentArchive([tmp_path])
    test_exp = next(item for item in archive.list_experiments() if item["kind"] == "test")
    detail = archive.get_experiment(test_exp["id"])

    assert detail["metrics"]["task_kind"] == "test"
    assert detail["metrics"]["overall"]["examples"]["value"] == 1
    assert detail["metrics"]["overall"]["parts"]["value"] == 1
    assert detail["artifacts"]["export_root"] == "examples"


def test_archive_returns_diagnostics_for_legacy_eval_missing_metric_files(tmp_path):
    run = tmp_path / "part_ss_latent_flow" / "legacy_eval_with_old_schema"
    report = run / "full_eval" / "step_000300"
    _write_json(
        report / "summary.json",
        {"overall": {"parts": 12, "objects": 4, "target_parts_iou_mean": 0.42}},
    )

    archive = ExperimentArchive([tmp_path])
    experiment = next(item for item in archive.list_experiments() if item["name"] == run.name)
    detail = archive.get_experiment(experiment["id"])

    diagnostics = detail["metrics"]["diagnostics"]
    assert detail["name"] == "legacy_eval_with_old_schema"
    assert detail["metrics"]["summary"]["overall"]["target_parts_iou_mean"] == pytest.approx(0.42)
    assert diagnostics["status"] == "incomplete"
    assert diagnostics["missing"] == ["part_metrics.jsonl", "object_metrics.jsonl"]
