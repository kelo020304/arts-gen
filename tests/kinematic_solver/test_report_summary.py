import json

from post_process.kinematic_solver.utils.report_summary import summarize_rows, write_report_summary


def test_report_summary_counts_statuses_and_success_rates(tmp_path):
    rows = [
        {
            "status": "ok",
            "type": "prismatic",
            "status_upper": "ok",
            "status_lower": "ok",
            "success_upper": True,
            "success_lower": False,
            "iou_range": 0.5,
        },
        {
            "status": "partial",
            "type": "prismatic",
            "status_upper": "initial_collision",
            "status_lower": "ok",
            "success_upper": None,
            "success_lower": True,
            "iou_range": None,
        },
    ]

    summary = summarize_rows(rows)
    out = write_report_summary(tmp_path, rows)

    assert summary["n_total"] == 2
    assert summary["n_ok"] == 1
    assert summary["succ_upper_all"] == "1/1"
    assert summary["succ_lower_all"] == "1/2"
    assert json.loads((tmp_path / "report_summary.json").read_text())["n_total"] == 2
    assert "V1 KinematicSolver internal report" in out.read_text()
