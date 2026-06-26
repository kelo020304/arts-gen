import json

import pytest

from post_process.kinematic_solver.sdk import (
    CandidateCompileError,
    EstimateContext,
    extract_editable_code,
    compile_candidate_report,
    replace_editable_code,
)


def _scaffold(body: str = "") -> str:
    return (
        "from post_process.kinematic_solver.sdk import LimitEstimate\n\n"
        "# >>> USER_CODE_START\n"
        f"{body}"
        "# >>> USER_CODE_END\n"
    )


def test_editable_region_replacement_cannot_modify_scaffold_imports():
    full_code = _scaffold("def estimate_limits(ctx):\n    return []\n")

    updated = replace_editable_code(
        full_code,
        "def estimate_limits(ctx):\n    return [LimitEstimate('j', 0.0, 1.0)]\n",
    )

    assert updated.startswith("from post_process.kinematic_solver.sdk import LimitEstimate")
    assert "return [LimitEstimate('j', 0.0, 1.0)]" in extract_editable_code(updated)
    assert "# >>> USER_CODE_START" in updated
    assert "# >>> USER_CODE_END" in updated


def test_compile_candidate_requires_estimate_limits(tmp_path):
    candidate = tmp_path / "estimate_limit_copy.py"
    candidate.write_text(_scaffold("def helper():\n    return 1\n"))
    ctx = EstimateContext(object_id="ra_test", joints={}, evidence={})

    with pytest.raises(CandidateCompileError, match="estimate_limits"):
        compile_candidate_report(candidate, ctx)


def test_compile_candidate_runs_single_file_and_validates_limit_schema(tmp_path):
    candidate = tmp_path / "estimate_limit_copy.py"
    candidate.write_text(_scaffold(
        "def estimate_limits(ctx):\n"
        "    joint = ctx.joints['part_02']\n"
        "    assert joint['type'] == 'prismatic'\n"
        "    return [LimitEstimate(joint_name='part_02', lower=0.0, upper=0.15)]\n"
    ))
    ctx = EstimateContext(
        object_id="ra_063",
        joints={"part_02": {"type": "prismatic"}},
        evidence={"part_02": {"labels": ["pull-out drawer pan"]}},
    )

    report = compile_candidate_report(candidate, ctx)

    assert report.passed is True
    assert len(report.estimates) == 1
    assert report.estimates[0].joint_name == "part_02"
    assert report.estimates[0].lower == pytest.approx(0.0)
    assert report.estimates[0].upper == pytest.approx(0.15)
    payload = json.loads(report.to_json())
    assert payload["estimates"][0]["upper"] == pytest.approx(0.15)


def test_compile_candidate_does_not_use_geometry_candidate_as_axis_pass_gate(tmp_path):
    candidate = tmp_path / "estimate_limit_copy.py"
    candidate.write_text(_scaffold(
        "def estimate_limits(ctx):\n"
        "    return [LimitEstimate(joint_name='part_00', lower=0.0, upper=1.0,\n"
        "                          axis_world=[0.0, -1.0, 0.0], axis_label='-Y')]\n"
    ))
    ctx = EstimateContext(
        object_id="ra_063",
        joints={
            "part_00": {
                "type": "revolute",
                "axis_world": [0.0, 1.0, 0.0],
            },
        },
        evidence={"part_00": {"labels": ["temperature knob"]}},
    )
    ctx.evidence["part_00"]["axis_candidates"] = [
        {"axis_label": "+Z", "axis_world": [0.0, 0.0, 1.0], "score": 1.0, "reason": "thin-axis"}
    ]

    report = compile_candidate_report(candidate, ctx)

    assert report.passed is True


def test_compile_candidate_allows_drawer_signed_axis_override(tmp_path):
    candidate = tmp_path / "estimate_limit_copy.py"
    candidate.write_text(_scaffold(
        "def estimate_limits(ctx):\n"
        "    return [LimitEstimate(joint_name='part_02', lower=0.0, upper=0.16,\n"
        "                          axis_world=[0.0, -1.0, 0.0], axis_label='-Y')]\n"
    ))
    ctx = EstimateContext(
        object_id="ra_063",
        joints={
            "part_02": {
                "type": "prismatic",
                "axis_world": [0.0, 1.0, 0.0],
            },
        },
        evidence={"part_02": {"labels": ["pull-out drawer pan"]}},
    )

    report = compile_candidate_report(candidate, ctx)

    assert report.passed is True


def test_compile_candidate_leaves_drawer_axis_judgment_to_motion_search(tmp_path):
    candidate = tmp_path / "estimate_limit_copy.py"
    candidate.write_text(_scaffold(
        "def estimate_limits(ctx):\n"
        "    return [LimitEstimate(joint_name='part_02', lower=0.0, upper=0.16,\n"
        "                          axis_world=[0.0, 1.0, 0.0], axis_label='+Y')]\n"
    ))
    ctx = EstimateContext(
        object_id="ra_063",
        joints={
            "part_02": {
                "type": "prismatic",
                "axis_world": [0.0, 1.0, 0.0],
            },
        },
        evidence={
            "part_02": {
                "labels": ["pull-out drawer pan"],
                "axis_candidates": [
                    {
                        "axis_label": "-Y",
                        "axis_world": [0.0, -1.0, 0.0],
                        "score": 1.0,
                        "reason": "prismatic drawer centroid-outward axis",
                    }
                ],
            }
        },
    )

    report = compile_candidate_report(candidate, ctx)

    assert report.passed is True
