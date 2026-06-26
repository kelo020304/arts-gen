import json

from post_process.kinematic_solver.sdk import build_context_from_roots
from post_process.kinematic_solver.sdk.vlm_initial import load_vlm_initial_context


def _write_obj(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(
            [
                "v 1 2 3",
                "v 3 2 3",
                "v 1 4 3",
                "v 1 2 5",
                "f 1 2 3",
            ]
        ),
        encoding="utf-8",
    )


def test_build_context_from_roots_uses_oracle_and_part_semantics(tmp_path):
    converter = tmp_path / "converter"
    source = tmp_path / "source"
    (converter / "raw/vlm_oracle").mkdir(parents=True)
    (converter / "raw/vlm_oracle/ra_063.json").write_text(json.dumps({
        "object_id": "ra_063",
        "joints": {
            "part_02": {
                "type": "prismatic",
                "canonical_unit": "meters",
                "axis_world": [0, 1, 0],
                "moving_parts": ["part_02"],
            }
        },
    }))
    (source / "source/model/063").mkdir(parents=True)
    (source / "source/model/063/gt_part.json").write_text(json.dumps({
        "pull-out drawer pan": "part_02",
    }))

    ctx = build_context_from_roots(
        object_id="ra_063",
        converter_output_root=converter,
        source_root=source,
    )

    assert ctx.object_id == "ra_063"
    assert ctx.joints["part_02"]["type"] == "prismatic"
    assert ctx.evidence["part_02"]["labels"] == ["pull-out drawer pan"]


def test_build_context_from_roots_allows_missing_oracle_for_json_defined_joints(tmp_path):
    converter = tmp_path / "converter"
    source = tmp_path / "source"
    for name in ["body", "part_03", "part_05", "part_07"]:
        _write_obj(converter / f"raw/partseg/ra_036/objs/{name}.obj")

    ctx = build_context_from_roots(
        object_id="ra_036",
        converter_output_root=converter,
        source_root=source,
    )

    assert ctx.object_id == "ra_036"
    assert ctx.joints == {}
    assert ctx.evidence["__available_parts__"] == ["body", "part_03", "part_05", "part_07"]
    assert ctx.evidence["__part_centers__"]["part_05"] == [-2.5, -1.5, 3.5]


def test_load_vlm_initial_context_creates_missing_joints_from_available_parts(tmp_path):
    converter = tmp_path / "converter"
    source = tmp_path / "source"
    for name in ["body", "part_03", "part_05", "part_07"]:
        _write_obj(converter / f"raw/partseg/ra_036/objs/{name}.obj")
    ctx = build_context_from_roots(
        object_id="ra_036",
        converter_output_root=converter,
        source_root=source,
    )
    initial_json = tmp_path / "vlm_initial.json"
    initial_json.write_text(json.dumps({
        "object_id": "ra_036",
        "initial_joints": {
            "part_05": {
                "type": "revolute",
                "axis": [0, 0, 1],
                "limit": [-360, 360],
                "parent": "body",
            },
            "part_07": {
                "type": "prismatic",
                "axis": [1, 0, 0],
                "limit": [0, 30],
                "parent": "body",
            },
        },
    }))

    next_ctx, estimates = load_vlm_initial_context(ctx, initial_json)

    assert sorted(next_ctx.joints) == ["part_05", "part_07"]
    assert next_ctx.joints["part_05"]["moving_parts"] == ["part_05"]
    assert next_ctx.joints["part_05"]["static_parts"] == ["body", "part_03", "part_07"]
    assert next_ctx.joints["part_05"]["origin_world"] == [-2.5, -1.5, 3.5]
    assert next_ctx.joints["part_05"]["body0_link_name"] == "body"
    assert next_ctx.joints["part_07"]["canonical_unit"] == "meters"
    assert {estimate.joint_name for estimate in estimates} == {"part_05", "part_07"}
