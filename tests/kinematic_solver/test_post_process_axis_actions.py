from pathlib import Path

from post_process.kinematic_solver.sdk.axis_candidates import (
    infer_axis_candidates_for_joint,
)


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


def test_revolute_axis_candidates_use_authored_axis_without_relation_evidence(tmp_path):
    _write_box_obj(
        tmp_path / "raw/partseg/ra_063/objs/part_00.obj",
        extent=(0.04, 0.04, 0.01),
    )

    candidates = infer_axis_candidates_for_joint(
        "ra_063",
        "part_00",
        {
            "type": "revolute",
            "moving_parts": ["part_00"],
            "axis_world": [0.0, 1.0, 0.0],
        },
        converter_output_root=tmp_path,
    )

    assert candidates[0].axis_label == "+Y"
    assert candidates[0].reason == "authored signed joint axis"


def test_revolute_knob_axis_candidates_prefer_geometry_before_authored_fallback(tmp_path):
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
            "axis_world": [0.0, 1.0, 0.0],
        },
        converter_output_root=tmp_path,
        labels=["temperature knob"],
    )

    labels = [candidate.axis_label for candidate in candidates[:4]]
    assert labels[0] == "+Z"
    assert "+Y" in labels
    assert candidates[0].reason == "revolute Articraft-style mount axis"


def test_revolute_axis_candidates_do_not_infer_axis_without_relation_or_authored_axis(tmp_path):
    _write_box_obj(
        tmp_path / "raw/partseg/ra_063/objs/part_00.obj",
        extent=(0.04, 0.04, 0.01),
    )

    candidates = infer_axis_candidates_for_joint(
        "ra_063",
        "part_00",
        {"type": "revolute", "moving_parts": ["part_00"]},
        converter_output_root=tmp_path,
    )

    assert candidates == []
