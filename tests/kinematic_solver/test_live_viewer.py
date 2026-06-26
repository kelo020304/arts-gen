import json

from post_process.kinematic_solver.sdk.live_viewer import LiveViewer
from post_process.kinematic_solver.sdk.schemas import CandidateReport, LimitEstimate


def test_live_viewer_writes_event_stream_without_legacy_html(tmp_path):
    viewer = LiveViewer(tmp_path)

    viewer.prepare()
    viewer.emit("run_started", iteration=0, status="running", phase="start")
    viewer.emit_report(
        "validation_finished",
        CandidateReport(
            passed=True,
            estimates=[
                LimitEstimate(
                    joint_name="part_02",
                    lower=0.0,
                    upper=0.15,
                    confidence=0.7,
                    reason="drawer opens in positive direction",
                )
            ],
        ),
        iteration=1,
        phase="validation",
    )

    events = [
        json.loads(line)
        for line in (tmp_path / "agent_events.jsonl").read_text().splitlines()
    ]
    assert [event["event"] for event in events] == [
        "run_started",
        "validation_finished",
    ]
    assert all(event["run_id"] == viewer.run_id for event in events)
    assert events[0]["iteration"] == 0
    assert events[1]["iteration"] == 1
    assert events[1]["status"] == "passed"
    assert events[1]["estimates"][0]["joint_name"] == "part_02"
    assert events[1]["estimates"][0]["upper"] == 0.15

    assert not (tmp_path / "agent_live.html").exists()


def test_live_viewer_prepare_resets_frontend_state(tmp_path):
    stale_state = tmp_path / "frontend_state.json"
    stale_state.parent.mkdir(parents=True, exist_ok=True)
    stale_state.write_text(json.dumps({
        "latest_iteration": 9,
        "latest_preview": {"asset_name": "stale"},
        "iterations": [{"iteration": 9}],
    }))

    viewer = LiveViewer(tmp_path, run_id="fresh_run")
    viewer.prepare()

    state = json.loads(stale_state.read_text())
    assert state == {
        "iterations": [],
        "latest_iteration": 0,
        "latest_preview": None,
        "run_id": "fresh_run",
    }
