import json
import os
import subprocess
import sys

import pytest

from post_process.kinematic_solver.estimate_limit import _clear_run_artifacts, estimate_limits
from post_process.kinematic_solver.sdk import EstimateContext


def test_clear_run_artifacts_removes_only_generated_outputs(tmp_path):
    (tmp_path / "candidate_report.json").write_text("old report")
    (tmp_path / "predictions.jsonl").write_text("old predictions")
    (tmp_path / "notes.txt").write_text("keep me")
    viz_dir = tmp_path / "ra_063" / "agent_viz" / "part_02"
    viz_dir.mkdir(parents=True)
    (viz_dir / "step_viewer.html").write_text("old viewer")
    sibling = tmp_path / "ra_063" / "manual_note.txt"
    sibling.write_text("keep this too")

    removed = _clear_run_artifacts(tmp_path, object_id="ra_063")

    assert sorted(removed) == [
        "candidate_report.json",
        "predictions.jsonl",
        "ra_063/agent_viz",
    ]
    assert not (tmp_path / "candidate_report.json").exists()
    assert not (tmp_path / "predictions.jsonl").exists()
    assert not (tmp_path / "ra_063" / "agent_viz").exists()
    assert (tmp_path / "notes.txt").read_text() == "keep me"
    assert sibling.read_text() == "keep this too"


def test_estimate_limit_entrypoint_compiles_candidate_and_writes_predictions(tmp_path):
    candidate = tmp_path / "estimate_limit_copy.py"
    candidate.write_text(
        "from post_process.kinematic_solver.sdk import LimitEstimate\n\n"
        "# >>> USER_CODE_START\n"
        "def estimate_limits(ctx):\n"
        "    return [LimitEstimate(joint_name=name, lower=0.0, upper=0.15)\n"
        "            for name in sorted(ctx.joints)]\n"
        "# >>> USER_CODE_END\n"
    )
    context = tmp_path / "context.json"
    context.write_text(json.dumps({
        "object_id": "ra_063",
        "joints": {
            "part_02": {"type": "prismatic", "axis_world": [0, 1, 0]},
        },
        "evidence": {
            "part_02": {"labels": ["pull-out drawer pan"]},
        },
    }))
    out_dir = tmp_path / "out"

    result = subprocess.run(
        [
            sys.executable,
            "-m", "post_process.kinematic_solver.estimate_limit",
            "--candidate-path", str(candidate),
            "--context-json", str(context),
            "--out-dir", str(out_dir),
        ],
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    assert result.returncode == 0, result.stderr
    report = json.loads((out_dir / "candidate_report.json").read_text())
    assert report["passed"] is True
    pred = json.loads((out_dir / "predictions.jsonl").read_text().strip())
    assert pred["object_id"] == "ra_063"
    assert pred["joint_name"] == "part_02"
    assert pred["predicted_lower"] == 0.0
    assert pred["predicted_upper"] == 0.15


def test_estimate_limit_entrypoint_writes_axis_override_predictions(tmp_path):
    candidate = tmp_path / "estimate_limit_axis.py"
    candidate.write_text(
        "from post_process.kinematic_solver.sdk import LimitEstimate\n\n"
        "# >>> USER_CODE_START\n"
        "def estimate_limits(ctx):\n"
        "    return [LimitEstimate(joint_name='part_02', lower=0.0, upper=0.15,\n"
        "                          axis_world=[0.0, -1.0, 0.0], axis_label='-Y')]\n"
        "# >>> USER_CODE_END\n"
    )
    context = tmp_path / "context.json"
    context.write_text(json.dumps({
        "object_id": "ra_063",
        "joints": {
            "part_02": {"type": "prismatic", "axis_world": [0, 1, 0]},
        },
        "evidence": {},
    }))
    out_dir = tmp_path / "out_axis"

    result = subprocess.run(
        [
            sys.executable,
            "-m", "post_process.kinematic_solver.estimate_limit",
            "--candidate-path", str(candidate),
            "--context-json", str(context),
            "--out-dir", str(out_dir),
        ],
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    assert result.returncode == 0, result.stderr
    pred = json.loads((out_dir / "predictions.jsonl").read_text().strip())
    assert pred["predicted_axis_world"] == [0.0, -1.0, 0.0]
    assert pred["predicted_axis_label"] == "-Y"


def test_estimate_limit_entrypoint_defaults_to_maintained_file(tmp_path):
    context = tmp_path / "context.json"
    context.write_text(json.dumps({
        "object_id": "ra_063",
        "joints": {
            "part_02": {
                "type": "prismatic",
                "canonical_unit": "meters",
                "axis_world": [0, 1, 0],
                "moving_parts": ["part_02"],
            },
        },
        "evidence": {
            "part_02": {"labels": ["pull-out drawer pan"]},
        },
    }))
    out_dir = tmp_path / "out_default"

    result = subprocess.run(
        [
            sys.executable,
            "-m", "post_process.kinematic_solver.estimate_limit",
            "--context-json", str(context),
            "--out-dir", str(out_dir),
        ],
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    assert result.returncode == 0, result.stderr
    pred = json.loads((out_dir / "predictions.jsonl").read_text().strip())
    assert pred["joint_name"] == "part_02"
    assert pred["predicted_lower"] == 0.0
    assert 0.0 < pred["predicted_upper"] <= 0.5


def test_estimate_limits_uses_previous_candidate_axis_from_feedback_for_next_axis_step():
    ctx = EstimateContext(
        object_id="ra_063",
        joints={
            "part_00": {
                "type": "revolute",
                "axis_world": [0.0, 0.0, 1.0],
            },
        },
        evidence={
            "part_00": {
                "labels": ["knob"],
                "recommended_axis_world": [0.0, 0.0, 1.0],
                "initial_estimate": {
                    "axis_world": [0.0, 0.0, 1.0],
                    "lower": 0.0,
                    "upper": 0.0,
                },
            },
            "__last_feedback__": (
                "<compile_signals>\n"
                "[failure][qc][QC_JOINT_AXIS] joint=part_00: "
                "part_00: candidate axis +Z deviates from rotary control geometry axis +Z by 9.9 degrees; "
                "target_axis_world=[0.25881905, 0.0, 0.96592583]; "
                "candidate_axis_world=[0.08715574, 0.0, 0.99619470]; "
                "max_angle_degrees=5; switch axis or rotate toward the SDK motion target before marking this joint correct\n"
                "</compile_signals>"
            ),
        },
    )

    estimate = estimate_limits(ctx)[0]

    assert estimate.joint_name == "part_00"
    assert estimate.axis_world == pytest.approx(
        [0.17364818, 0.0, 0.98480775],
        abs=1e-6,
    )
    assert "previous candidate_axis_world" in estimate.reason


def test_print_api_settings_uses_articraft_env_without_printing_keys(tmp_path):
    env = {
        **os.environ,
        "OPENROUTER_API_KEY": "secret-should-not-print",
        "OPENROUTER_BASE_URL": "https://api-router.evad.mioffice.cn/v1",
        "ARTICRAFT_MODEL": "gpt-5.5",
        "ARTICRAFT_THINKING_LEVEL": "high",
    }

    result = subprocess.run(
        [
            sys.executable,
            "-m", "post_process.kinematic_solver.estimate_limit",
            "--print-api-settings",
            "--out-dir", str(tmp_path / "unused"),
        ],
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env,
    )

    assert result.returncode == 0, result.stderr
    assert "secret-should-not-print" not in result.stdout
    payload = json.loads(result.stdout)
    assert payload["provider"] == "openrouter"
    assert payload["model"] == "gpt-5.5"
    assert payload["thinking_level"] == "high"
    assert payload["openrouter_base_url"] == "https://api-router.evad.mioffice.cn/v1"


def test_estimate_limit_entrypoint_writes_live_viewer_events(tmp_path):
    context = tmp_path / "context.json"
    context.write_text(json.dumps({
        "object_id": "ra_063",
        "joints": {
            "part_02": {
                "type": "prismatic",
                "canonical_unit": "meters",
                "axis_world": [0, 1, 0],
            },
        },
        "evidence": {
            "part_02": {"labels": ["pull-out drawer pan"]},
        },
    }))
    out_dir = tmp_path / "out_live"

    result = subprocess.run(
        [
            sys.executable,
            "-m", "post_process.kinematic_solver.estimate_limit",
            "--context-json", str(context),
            "--out-dir", str(out_dir),
            "--live-viewer",
            "--no-live-server",
        ],
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    assert result.returncode == 0, result.stderr
    assert not (out_dir / "agent_live.html").exists()
    assert (out_dir / "agent_events.jsonl").is_file()
    assert (out_dir / "frontend_state.json").is_file()
    events = [
        json.loads(line)
        for line in (out_dir / "agent_events.jsonl").read_text().splitlines()
    ]
    event_names = [event["event"] for event in events]
    assert event_names == [
        "run_started",
        "context_started",
        "context_finished",
        "agent_iteration_started",
        "compile_started",
        "compile_finished",
        "validation_finished",
        "predictions_written",
        "run_finished",
    ]
    assert events[0]["details"]["mode"] == "single-check"
    assert events[3]["iteration"] == 1
    assert events[-1]["status"] == "passed"
    validation = next(event for event in events if event["event"] == "validation_finished")
    assert validation["estimates"][0]["joint_name"] == "part_02"


def test_estimate_limit_entrypoint_visualizes_vlm_initial_json(tmp_path):
    candidate = tmp_path / "estimate_limit_initial.py"
    candidate.write_text(
        "from post_process.kinematic_solver.sdk import LimitEstimate\n\n"
        "# >>> USER_CODE_START\n"
        "def estimate_limits(ctx):\n"
        "    estimates = []\n"
        "    for name in sorted(ctx.joints):\n"
        "        init = ctx.evidence[name]['initial_estimate']\n"
        "        estimates.append(LimitEstimate(\n"
        "            joint_name=name,\n"
        "            lower=init['lower'],\n"
        "            upper=init['upper'],\n"
        "            axis_world=init['axis_world'],\n"
        "        ))\n"
        "    return estimates\n"
        "# >>> USER_CODE_END\n"
    )
    converter_root = tmp_path / "converter"
    obj_dir = converter_root / "raw/partseg/ra_063/objs"
    obj_dir.mkdir(parents=True)
    for name in ["body", "part_00", "part_01", "part_02"]:
        (obj_dir / f"{name}.obj").write_text(
            "\n".join([
                "v 0 0 0",
                "v 0.01 0 0",
                "v 0 0.01 0",
                "v 0 0 0.01",
                "f 1 2 3",
            ])
        )
    context = tmp_path / "context.json"
    context.write_text(json.dumps({
        "object_id": "ra_063",
        "joints": {
            "part_00": {
                "type": "revolute",
                "origin_world": [0, 0, 0],
                "axis_world": [0, 1, 0],
                "moving_parts": ["part_00"],
                "static_parts": ["body", "part_01", "part_02"],
            },
            "part_01": {
                "type": "revolute",
                "origin_world": [0, 0, 0],
                "axis_world": [0, 1, 0],
                "moving_parts": ["part_01"],
                "static_parts": ["body", "part_00", "part_02"],
            },
            "part_02": {
                "type": "prismatic",
                "origin_world": [0, 0, 0],
                "axis_world": [0, 1, 0],
                "moving_parts": ["part_02"],
                "static_parts": ["body", "part_00", "part_01"],
            },
        },
        "evidence": {
            "part_00": {"labels": ["temperature knob"]},
            "part_01": {"labels": ["timer knob"]},
            "part_02": {"labels": ["pull-out drawer pan"]},
        },
    }))
    initial_json = tmp_path / "vlm_initial.json"
    initial_json.write_text(json.dumps({
        "object_id": "ra_063",
        "initial_joints": {
            "part_00": {"type": "revolute", "axis": [0, 0, 1], "limit": None, "parent": "body"},
            "part_01": {"type": "revolute", "axis": [1, 0, 0], "limit": [-360, 360], "parent": "body"},
            "part_02": {"type": "prismatic", "axis": [1, 0, 0], "limit": [0, 30], "parent": "body"},
        },
    }))
    out_dir = tmp_path / "out_initial"

    result = subprocess.run(
        [
            sys.executable,
            "-m", "post_process.kinematic_solver.estimate_limit",
            "--candidate-path", str(candidate),
            "--context-json", str(context),
            "--converter-output-root", str(converter_root),
            "--initial-joints-json", str(initial_json),
            "--skip-motion-validation",
            "--out-dir", str(out_dir),
            "--live-viewer",
            "--no-live-server",
        ],
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    assert result.returncode == 0, result.stderr
    state = json.loads((out_dir / "frontend_state.json").read_text())
    initial_preview = next(item for item in state["iterations"] if item.get("preview_kind") == "vlm_initial")
    estimates = {item["joint_name"]: item for item in initial_preview["estimates"]}
    assert estimates["part_00"]["axis_world"] == pytest.approx(
        [0.5773502691896258, 0.5773502691896256, 0.5773502691896258]
    )
    assert estimates["part_00"]["lower"] == 0.0
    assert estimates["part_00"]["upper"] == 0.0
    assert estimates["part_01"]["axis_world"] == pytest.approx(
        [0.5773502691896258, 0.5773502691896256, 0.5773502691896258]
    )
    assert estimates["part_01"]["lower"] == pytest.approx(-6.283185307179586)
    assert estimates["part_01"]["upper"] == pytest.approx(6.283185307179586)
    assert estimates["part_02"]["axis_world"] == pytest.approx(
        [0.5773502691896258, 0.5773502691896256, 0.5773502691896258]
    )
    assert estimates["part_02"]["upper"] == pytest.approx(0.03)


def test_estimate_limit_roots_can_define_joints_from_initial_json_without_oracle(tmp_path):
    candidate = tmp_path / "estimate_limit_initial_only.py"
    candidate.write_text(
        "from post_process.kinematic_solver.sdk import LimitEstimate\n\n"
        "# >>> USER_CODE_START\n"
        "def estimate_limits(ctx):\n"
        "    return [LimitEstimate(joint_name=name,\n"
        "                          lower=ctx.evidence[name]['initial_estimate']['lower'],\n"
        "                          upper=ctx.evidence[name]['initial_estimate']['upper'],\n"
        "                          axis_world=ctx.evidence[name]['initial_estimate']['axis_world'])\n"
        "            for name in sorted(ctx.joints)]\n"
        "# >>> USER_CODE_END\n"
    )
    converter_root = tmp_path / "converter"
    obj_dir = converter_root / "raw/partseg/ra_036/objs"
    obj_dir.mkdir(parents=True)
    for name in ["body", "part_05", "part_07"]:
        (obj_dir / f"{name}.obj").write_text(
            "\n".join([
                "v 0 0 0",
                "v 0.01 0 0",
                "v 0 0.01 0",
                "f 1 2 3",
            ]),
            encoding="utf-8",
        )
    source_root = tmp_path / "source"
    (source_root / "source/model/036").mkdir(parents=True)
    initial_json = tmp_path / "vlm_initial.json"
    initial_json.write_text(json.dumps({
        "object_id": "ra_036",
        "initial_joints": {
            "part_05": {"type": "revolute", "axis": [0, 0, 1], "limit": [-360, 360], "parent": "body"},
            "part_07": {"type": "prismatic", "axis": [1, 0, 0], "limit": [0, 30], "parent": "body"},
        },
    }))
    out_dir = tmp_path / "out_initial_only"

    result = subprocess.run(
        [
            sys.executable,
            "-m", "post_process.kinematic_solver.estimate_limit",
            "--candidate-path", str(candidate),
            "--object-id", "ra_036",
            "--converter-output-root", str(converter_root),
            "--source-root", str(source_root),
            "--initial-joints-json", str(initial_json),
            "--skip-motion-validation",
            "--out-dir", str(out_dir),
        ],
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    assert result.returncode == 0, result.stderr
    predictions = [
        json.loads(line)
        for line in (out_dir / "predictions.jsonl").read_text().splitlines()
    ]
    assert [item["joint_name"] for item in predictions] == ["part_05", "part_07"]
