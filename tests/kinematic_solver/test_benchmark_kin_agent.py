import json

import numpy as np
import pytest

from post_process.kinematic_solver.benchmark_kin_agent import (
    _find_gt_part,
    _resolve_part,
    _summarize,
    aligned_range_endpoint_errors,
    annotation_frame_axis_error_deg,
    axis_angular_error_deg,
    benchmark_points_in_delivery_frame,
    build_blind_manifest,
    mechanical_category,
    origin_gt_axis_perpendicular_offset,
    origin_line_distance,
    prediction_in_annotation_frame,
    refine_predictions_with_observations,
)


def test_axis_metric_compares_decoded_canonical_frame():
    assert axis_angular_error_deg((0, 0, 1), (0, 0, 1)) == 0.0


def test_axis_error_is_direction_invariant():
    assert axis_angular_error_deg((1, 0, 0), (-1, 0, 0)) == 0.0


def test_origin_metric_compares_infinite_hinge_lines():
    assert np.isclose(origin_line_distance((0, 0, 0), (1, 0, 0), (8, 0.25, 0), (1, 0, 0)), 0.25)


def test_primary_origin_metric_uses_gt_axis_perpendicular_plane():
    # Offset along the hinge axis is unidentifiable and must not count.
    assert np.isclose(origin_gt_axis_perpendicular_offset((8, 0.25, 0), (0, 0, 0), (1, 0, 0)), 0.25)


def test_range_endpoints_flip_when_axis_sign_is_opposite():
    candidate = {"axis_world": [-1, 0, 0], "lower": 0.0, "upper": 0.4}
    lower, upper, mean = aligned_range_endpoint_errors(candidate, [1, 0, 0, 0, 0, 0, -0.4, 0.0])
    assert np.allclose((lower, upper, mean), (0.0, 0.0, 0.0))


def test_physx_0511_prediction_is_mapped_back_to_y_up_annotation_frame():
    candidate = {"axis_world": [0, -1, 0], "origin_world": [1, 2, 3]}
    mapped = prediction_in_annotation_frame(candidate, "physx-0511-drawer-door")
    assert mapped["axis_world"] == [0.0, 0.0, 1.0]
    assert mapped["origin_world"] == [1.0, 3.0, -2.0]
    assert candidate["axis_world"] == [0, -1, 0]


def test_other_dataset_prediction_frame_is_identity():
    candidate = {"axis_world": [0, -1, 0], "origin_world": [1, 2, 3]}
    assert prediction_in_annotation_frame(candidate, "phyx-verse") == candidate


def test_realappliance_prediction_applies_delivery_root_rotation():
    candidate = {"axis_world": [0, 0, -1], "origin_world": [1, 2, 3]}
    mapped = prediction_in_annotation_frame(candidate, "realappliance")
    assert mapped["axis_world"] == [0.0, 1.0, 0.0]
    assert mapped["origin_world"] == [1.0, -3.0, 2.0]


def test_legacy_realappliance_benchmark_points_are_unbaked_for_delivery():
    points = np.asarray([[1.0, 2.0, 3.0], [-4.0, -5.0, -6.0]])
    decoded = benchmark_points_in_delivery_frame(points, "realappliance")
    assert decoded.tolist() == [[1.0, 3.0, -2.0], [-4.0, -6.0, 5.0]]
    assert prediction_in_annotation_frame(
        {"axis_world": decoded[0], "origin_world": decoded[1]}, "realappliance",
    ) == {"axis_world": [1.0, 2.0, 3.0], "origin_world": [-4.0, -5.0, -6.0]}


def test_other_benchmark_points_are_already_in_delivery_frame():
    points = np.asarray([[1.0, 2.0, 3.0]])
    assert benchmark_points_in_delivery_frame(points, "phyx-verse") is not points
    assert benchmark_points_in_delivery_frame(points, "phyx-verse").tolist() == points.tolist()


def test_summary_excludes_type_mismatch_from_range_error():
    common = {
        "dataset": "sample", "category": "drawer", "gt_type": "prismatic",
        "axis_angular_error_deg": 10.0, "origin_line_distance": None,
        "origin_line_distance_normalized": None, "iterations": 2, "runtime_seconds": 0.1,
    }
    summary = _summarize([
        {**common, "type_correct": True, "range_endpoint_error": 0.2},
        {**common, "type_correct": False, "range_endpoint_error": None},
    ])
    assert summary["prismatic_type_correct_samples"] == 1
    assert summary["prismatic_range_endpoint_error_mean"] == 0.2


@pytest.mark.parametrize(("label", "expected"), [
    ("middle_drawer_0", "drawer"),
    ("微波炉门_0", "door"),
    ("时间旋钮_0", "knob"),
    ("glass_lid_0", "lid"),
    ("open_door_button_0", None),
    ("开盖按钮_0", None),
])
def test_mechanical_category_avoids_button_false_positives(label, expected):
    assert mechanical_category(label) == expected


def test_decoded_part_resolution_never_uses_substring_fallback(tmp_path):
    (tmp_path / "part_00_glass_door_0.obj").touch()
    with pytest.raises(ValueError):
        _resolve_part(tmp_path, "door_0")


def test_gt_part_resolution_never_uses_substring_fallback():
    with pytest.raises(ValueError):
        _find_gt_part({"glass_door_0": {}}, "door_0")


def test_blind_manifest_uses_decoded_filenames_only(tmp_path, monkeypatch):
    decoded_root = tmp_path / "decoded"
    assets = decoded_root / "phyx-verse__fresh-object__angle_00__mujoco" / "assets"
    assets.mkdir(parents=True)
    (assets / "body_without_parts.obj").touch()
    (assets / "part_00_left_door_0.obj").touch()
    (assets / "part_01_open_door_button_0.obj").touch()
    output = tmp_path / "blind.json"

    original_read_text = type(output).read_text

    def reject_annotation_reads(path, *args, **kwargs):
        assert "part_info" not in str(path)
        return original_read_text(path, *args, **kwargs)

    monkeypatch.setattr(type(output), "read_text", reject_annotation_reads)
    result = build_blind_manifest(
        {"phyx-verse": decoded_root},
        output,
        excluded_object_keys={"phyx-verse::old-object"},
    )

    assert [row["label"] for row in result["samples"]] == ["left_door_0"]
    assert result["samples"][0]["decoded_part_file"] == "part_00_left_door_0.obj"
    assert result["selection_contract"].startswith("decoded OBJ filenames")


def test_annotation_frame_audit_matches_prismatic_transform(tmp_path):
    path = tmp_path / "sample" / "joint_transforms"
    path.mkdir(parents=True)
    (path / "obj.json").write_text(
        '{"angles":{"1":{"part_transforms":{"3":'
        '[[1,0,0,0],[0,1,0,-0.2],[0,0,1,0],[0,0,0,1]]}}}}'
    )
    gt = {"raw_label": 3, "joint": "prismatic", "joint_params": [0, -1, 0, 0, 0, 0, 0, 0.2]}
    assert annotation_frame_axis_error_deg(tmp_path, "sample", "obj", gt) == 0.0


def test_benchmark_refine_applies_phyx_knob_thin_axis_critic(tmp_path):
    moving = tmp_path / "moving.obj"
    moving.write_text("".join(
        f"v {x} {y} {z}\n"
        for x in (-1.0, 0.0, 1.0)
        for y in (-1.0, 0.0, 1.0)
        for z in (-0.04, 0.04)
    ))
    body = tmp_path / "body.obj"
    body.write_text("v 0 0 0\nv 1 0 0\nv 0 1 0\nv 0 0 1\n")
    incumbent = {
        "joint_type": "revolute", "axis_world": [1.0, 0.0, 0.0],
        "origin_world": [0.0, 0.0, 0.0], "lower": -1.0, "upper": 1.0,
        "score": 0.95, "signals": {"axis_confidence": 0.7}, "reason": "fixture",
    }
    z_proposal = {
        **incumbent, "axis_world": [0.0, 0.0, 1.0],
        "origin_world": [0.1, 0.2, 0.3], "score": 0.82,
    }
    frozen = tmp_path / "predictions.json"
    frozen.write_text(json.dumps({
        "format": "fixture", "max_iterations": 7, "predictions": [{
            "sample_id": "fixture", "dataset": "phyx-verse", "object_id": "object",
            "angle": 0, "category": "knob", "label": "control_knob_0",
            "body_mesh": str(body), "moving_mesh": str(moving), "iterations": 1,
            "candidate": incumbent,
            "trace": [{"iteration": 1, "selected": incumbent, "alternatives": [incumbent, z_proposal]}],
        }],
    }))

    payload = refine_predictions_with_observations(frozen, tmp_path / "refined.json", {})
    row = payload["predictions"][0]

    assert row["candidate"]["axis_world"] == pytest.approx((0.0, 0.0, 1.0))
    assert row["phyx_knob_thin_axis_critic"]["used"] is True
    assert row["iterations"] < 10


def test_benchmark_refine_applies_static_phyx_door_contact_axis_critic(tmp_path):
    moving = tmp_path / "moving.obj"
    moving.write_text("".join(
        f"v {x} {y} {z}\n"
        for x in (-0.05, 0.0, 0.05)
        for y in (-0.4, 0.0, 0.4)
        for z in np.linspace(-1.0, 1.0, 21)
    ))
    body = tmp_path / "body.obj"
    body.write_text("".join(
        f"v {x} {y} {z}\n"
        for x in (-0.16, -0.07)
        for y in (-0.06, 0.0, 0.06)
        for z in np.linspace(-1.05, 1.05, 21)
    ))
    incumbent = {
        "joint_type": "revolute", "axis_world": [0.0, 1.0, 0.0],
        "origin_world": [0.0, 0.0, 0.0], "lower": 0.0, "upper": 0.5,
        "score": 0.95, "signals": {"axis_confidence": 0.9}, "reason": "fixture",
    }
    z_proposal = {
        **incumbent, "axis_world": [0.0, 0.0, 1.0],
        "origin_world": [0.1, 0.2, 0.3], "score": 0.82,
    }
    frozen = tmp_path / "predictions.json"
    frozen.write_text(json.dumps({
        "format": "fixture", "max_iterations": 7, "predictions": [{
            "sample_id": "fixture", "dataset": "phyx-verse", "object_id": "object",
            "angle": 0, "category": "door", "label": "oven_door_0",
            "body_mesh": str(body), "moving_mesh": str(moving), "iterations": 1,
            "candidate": incumbent,
            "trace": [{"iteration": 1, "selected": incumbent, "alternatives": [incumbent, z_proposal]}],
        }],
    }))

    payload = refine_predictions_with_observations(frozen, tmp_path / "refined.json", {})
    row = payload["predictions"][0]

    assert row["candidate"]["axis_world"] == pytest.approx((0.0, 0.0, 1.0))
    assert row["phyx_door_contact_axis_critic"]["used"] is True
    assert row["iterations"] < 10
