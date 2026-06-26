from pathlib import Path

import pytest

from post_process.kinematic_solver.sdk.axis_candidates import (
    infer_axis_candidates_for_joint,
    with_axis_candidate_evidence,
)
from post_process.kinematic_solver.sdk.schemas import EstimateContext


def _write_box_obj(path: Path, *, extent: tuple[float, float, float]) -> None:
    hx, hy, hz = (value * 0.5 for value in extent)
    vertices = [
        (-hx, -hy, -hz),
        (hx, -hy, -hz),
        (hx, hy, -hz),
        (-hx, hy, -hz),
        (-hx, -hy, hz),
        (hx, -hy, hz),
        (hx, hy, hz),
        (-hx, hy, hz),
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(f"v {x} {y} {z}" for x, y, z in vertices))


def _write_slanted_panel_obj(path: Path, *, normal: tuple[float, float, float]) -> None:
    import numpy as np

    n = np.asarray(normal, dtype=float)
    n = n / np.linalg.norm(n)
    tangent = np.asarray([0.0, 1.0, 0.0], dtype=float)
    bitangent = np.cross(n, tangent)
    bitangent = bitangent / np.linalg.norm(bitangent)
    center = np.zeros(3, dtype=float)
    points = [
        center - tangent * 0.12 - bitangent * 0.12,
        center + tangent * 0.12 - bitangent * 0.12,
        center + tangent * 0.12 + bitangent * 0.12,
        center - tangent * 0.12 + bitangent * 0.12,
    ]
    back = [point - n * 0.08 for point in points]
    vertices = points + back
    faces = [
        (1, 2, 3),
        (1, 3, 4),
        (5, 7, 6),
        (5, 8, 7),
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(
            [*(f"v {x} {y} {z}" for x, y, z in vertices), *(f"f {a} {b} {c}" for a, b, c in faces)]
        )
    )


def test_revolute_axis_candidates_use_authored_axis_without_relation_evidence(tmp_path):
    _write_box_obj(
        tmp_path / "raw/partseg/ra_063/objs/part_00.obj",
        extent=(0.04, 0.04, 0.01),
    )
    obj_path = tmp_path / "raw/partseg/ra_063/objs/part_00.obj"
    shifted = []
    for line in obj_path.read_text().splitlines():
        _, x, y, z = line.split()
        shifted.append(f"v {x} {y} {float(z) + 0.16}")
    obj_path.write_text("\n".join(shifted))

    candidates = infer_axis_candidates_for_joint(
        "ra_063",
        "part_00",
        {"type": "revolute", "moving_parts": ["part_00"], "axis_world": [0.0, 1.0, 0.0]},
        converter_output_root=tmp_path,
    )

    assert candidates[0].axis_label == "+Y"
    assert candidates[0].reason == "authored signed joint axis"


def test_axis_candidate_evidence_is_added_to_context(tmp_path):
    _write_box_obj(
        tmp_path / "raw/partseg/ra_063/objs/part_00.obj",
        extent=(0.04, 0.04, 0.01),
    )
    ctx = EstimateContext(
        object_id="ra_063",
        joints={"part_00": {"type": "revolute", "moving_parts": ["part_00"]}},
        evidence={"part_00": {"labels": ["temperature knob"]}},
    )

    updated = with_axis_candidate_evidence(ctx, converter_output_root=tmp_path)

    assert "recommended_axis_label" not in updated.evidence["part_00"]
    assert "axis_candidates" not in updated.evidence["part_00"]


def test_rotary_control_prefers_geometry_axis_over_authored_axis(tmp_path):
    _write_box_obj(
        tmp_path / "raw/partseg/ra_063/objs/body.obj",
        extent=(0.20, 0.20, 0.20),
    )
    _write_box_obj(
        tmp_path / "raw/partseg/ra_063/objs/part_00.obj",
        extent=(0.04, 0.04, 0.01),
    )
    # Move the knob vertices above the body center so its outward normal is +Z.
    obj_path = tmp_path / "raw/partseg/ra_063/objs/part_00.obj"
    shifted = []
    for line in obj_path.read_text().splitlines():
        _, x, y, z = line.split()
        shifted.append(f"v {x} {y} {float(z) + 0.16}")
    obj_path.write_text("\n".join(shifted))

    candidates = infer_axis_candidates_for_joint(
        "ra_063",
        "part_00",
        {
            "type": "revolute",
            "moving_parts": ["part_00"],
            "axis_world": [0.0, 1.0, 0.0],
        },
        converter_output_root=tmp_path,
        labels=["temperature knob"],
    )

    assert candidates[0].axis_label == "+Z"
    assert candidates[0].reason == "revolute Articraft-style mount axis"
    assert any(candidate.axis_label == "+Y" for candidate in candidates)


def test_revolute_axis_candidates_use_nearby_body_surface_normal_not_pca(tmp_path):
    normal = (0.9660607, 0.0, 0.2583151)
    _write_slanted_panel_obj(
        tmp_path / "raw/partseg/ra_036/objs/body.obj",
        normal=normal,
    )
    _write_box_obj(
        tmp_path / "raw/partseg/ra_036/objs/part_03.obj",
        extent=(0.02, 0.03, 0.03),
    )
    obj_path = tmp_path / "raw/partseg/ra_036/objs/part_03.obj"
    shifted = []
    for line in obj_path.read_text().splitlines():
        _, x, y, z = line.split()
        shifted.append(f"v {float(x) + normal[0] * 0.05} {y} {float(z) + normal[2] * 0.05}")
    obj_path.write_text("\n".join(shifted))

    candidates = infer_axis_candidates_for_joint(
        "ra_036",
        "part_03",
        {
            "type": "revolute",
            "moving_parts": ["part_03"],
            "axis_world": [0.0, 0.0, 1.0],
        },
        converter_output_root=tmp_path,
        labels=["knob"],
    )

    assert candidates[0].axis_label == "surface_normal"
    assert candidates[0].axis_world == pytest.approx(normal, abs=1e-5)
    assert candidates[0].reason == "revolute Articraft-style nearby body surface normal"


def test_rotary_control_uses_geometry_axis_when_authored_axis_is_missing(tmp_path):
    _write_box_obj(
        tmp_path / "raw/partseg/ra_063/objs/body.obj",
        extent=(0.20, 0.20, 0.20),
    )
    _write_box_obj(
        tmp_path / "raw/partseg/ra_063/objs/part_00.obj",
        extent=(0.04, 0.04, 0.01),
    )
    obj_path = tmp_path / "raw/partseg/ra_063/objs/part_00.obj"
    shifted = []
    for line in obj_path.read_text().splitlines():
        _, x, y, z = line.split()
        shifted.append(f"v {x} {y} {float(z) + 0.16}")
    obj_path.write_text("\n".join(shifted))

    candidates = infer_axis_candidates_for_joint(
        "ra_063",
        "part_00",
        {
            "type": "revolute",
            "moving_parts": ["part_00"],
        },
        converter_output_root=tmp_path,
        labels=["temperature knob"],
    )

    assert candidates[0].axis_label == "+Z"
    assert candidates[0].axis_world == pytest.approx((0.0, 0.0, 1.0), abs=1e-9)


def test_pull_out_drawer_axis_prefers_centroid_outward_direction(tmp_path):
    _write_box_obj(
        tmp_path / "raw/partseg/ra_063/objs/body.obj",
        extent=(0.20, 0.20, 0.20),
    )
    _write_box_obj(
        tmp_path / "raw/partseg/ra_063/objs/part_02.obj",
        extent=(0.08, 0.12, 0.04),
    )
    # Move the drawer/pan in front of the body along -Y.
    obj_path = tmp_path / "raw/partseg/ra_063/objs/part_02.obj"
    shifted = []
    for line in obj_path.read_text().splitlines():
        _, x, y, z = line.split()
        shifted.append(f"v {x} {float(y) - 0.18} {z}")
    obj_path.write_text("\n".join(shifted))

    candidates = infer_axis_candidates_for_joint(
        "ra_063",
        "part_02",
        {
            "type": "prismatic",
            "moving_parts": ["part_02"],
            "axis_world": [0.0, 1.0, 0.0],
        },
        converter_output_root=tmp_path,
        labels=["pull-out drawer pan"],
    )

    assert candidates[0].axis_label == "-Y"
    assert candidates[0].axis_world == pytest.approx((0.0, -1.0, 0.0), abs=1e-9)
    assert candidates[0].reason == "prismatic Articraft-style rest-face exit axis"


def test_axis_candidate_evidence_transforms_source_y_front_frame_to_ros_frame(tmp_path):
    _write_box_obj(
        tmp_path / "raw/partseg/ra_063/objs/body.obj",
        extent=(0.20, 0.20, 0.20),
    )
    _write_box_obj(
        tmp_path / "raw/partseg/ra_063/objs/part_02.obj",
        extent=(0.08, 0.12, 0.04),
    )
    obj_path = tmp_path / "raw/partseg/ra_063/objs/part_02.obj"
    shifted = []
    for line in obj_path.read_text().splitlines():
        _, x, y, z = line.split()
        # Source asset frame pulls out toward -Y; canonical ROS frame should see +X.
        shifted.append(f"v {x} {float(y) - 0.18} {z}")
    obj_path.write_text("\n".join(shifted))
    ctx = EstimateContext(
        object_id="ra_063",
        joints={
            "part_02": {
                "type": "prismatic",
                "moving_parts": ["part_02"],
                "axis_world": [1.0, 0.0, 0.0],
            }
        },
        evidence={
            "__coordinate_frame__": {
                "name": "ros_x_front_y_left_z_up",
                "source": "source_neg_y_axis_front_x_axis_right_z_up",
            },
            "part_02": {"labels": ["pull-out drawer pan"]},
        },
    )

    updated = with_axis_candidate_evidence(ctx, converter_output_root=tmp_path)

    assert updated.evidence["part_02"]["recommended_axis_label"] == "+X"
    assert updated.evidence["part_02"]["recommended_axis_world"] == pytest.approx([1.0, 0.0, 0.0], abs=1e-9)


def test_prismatic_axis_candidates_prefer_rest_pose_face_exit_over_slanted_pca(tmp_path):
    _write_box_obj(
        tmp_path / "raw/partseg/ra_036/objs/body.obj",
        extent=(0.2295, 0.235, 0.306),
    )
    _write_box_obj(
        tmp_path / "raw/partseg/ra_036/objs/part_07.obj",
        extent=(0.2662, 0.20, 0.1415),
    )
    body_path = tmp_path / "raw/partseg/ra_036/objs/body.obj"
    shifted_body = []
    for line in body_path.read_text().splitlines():
        _, x, y, z = line.split()
        shifted_body.append(f"v {float(x) - 0.03645} {y} {z}")
    body_path.write_text("\n".join(shifted_body))
    part_path = tmp_path / "raw/partseg/ra_036/objs/part_07.obj"
    shifted_part = []
    for line in part_path.read_text().splitlines():
        _, x, y, z = line.split()
        shifted_part.append(f"v {float(x) + 0.01693} {y} {float(z) - 0.05853}")
    part_path.write_text("\n".join(shifted_part))

    candidates = infer_axis_candidates_for_joint(
        "ra_036",
        "part_07",
        {
            "type": "prismatic",
            "moving_parts": ["part_07"],
            "axis_world": [0.0, 0.0, 1.0],
        },
        converter_output_root=tmp_path,
        labels=["drawer", "basket"],
    )

    assert candidates[0].axis_label == "+X"
    assert candidates[0].axis_world == pytest.approx((1.0, 0.0, 0.0), abs=1e-9)
    assert candidates[0].reason == "prismatic Articraft-style rest-face exit axis"
