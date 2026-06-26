"""Unit tests for scripts/tools/articulator/schema.py.

Each test constructs the smallest schema dict that should trigger one
specific failure mode, exercising one validator branch at a time.
"""
from __future__ import annotations

import copy

import pytest

from schema import SchemaError, joints_by_child, topo_sort, validate


def _minimal_part(pid: str = "p0", **overrides) -> dict:
    p = {
        "id": pid,
        "clusters": ["cluster_00"],
        "physics": "kinematic",
        "collision": {"approx": "sdf", "resolution": 64},
        "scale_xyz": [1, 1, 1],
    }
    p.update(overrides)
    return p


def _minimal_revolute(jid="j0", parent="p0", child="p1", **overrides) -> dict:
    j = {
        "id": jid,
        "parent": parent,
        "child": child,
        "type": "revolute",
        "axis_p0": [0, 0, 0],
        "axis_p1": [1, 0, 0],
        "lower": -90.0,
        "upper": 90.0,
    }
    j.update(overrides)
    return j


def _ok_schema(**overrides) -> dict:
    """A passing 2-part 1-joint schema; tests then mutate fields to break it."""
    s = {
        "version": 2,
        "parts": [_minimal_part("p0"), _minimal_part("p1", physics="dynamic")],
        "joints": [_minimal_revolute()],
        "external_meshes": [],
    }
    s.update(overrides)
    return s


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


def test_minimal_valid_schema_passes():
    validate(_ok_schema())


def test_free_joint_does_not_require_axis():
    s = _ok_schema()
    s["joints"][0] = {"id": "j0", "parent": "p0", "child": "p1", "type": "free"}
    validate(s)  # should not raise


def test_fixed_joint_does_not_require_axis():
    s = _ok_schema()
    s["joints"][0] = {"id": "j0", "parent": "p0", "child": "p1", "type": "fixed"}
    validate(s)


def test_empty_joints_list_is_ok_for_pure_free_bodies():
    s = _ok_schema()
    s["joints"] = []
    validate(s)


# ---------------------------------------------------------------------------
# Version + structure
# ---------------------------------------------------------------------------


def test_wrong_version_rejected():
    s = _ok_schema(version=1)
    with pytest.raises(SchemaError, match="schema version"):
        validate(s)


def test_missing_version_rejected():
    s = _ok_schema()
    del s["version"]
    with pytest.raises(SchemaError, match="schema version"):
        validate(s)


def test_empty_parts_rejected():
    s = _ok_schema()
    s["parts"] = []
    with pytest.raises(SchemaError, match="parts"):
        validate(s)


# ---------------------------------------------------------------------------
# Parts
# ---------------------------------------------------------------------------


def test_duplicate_part_id_rejected():
    s = _ok_schema()
    s["parts"][1]["id"] = "p0"  # duplicate
    s["joints"][0]["child"] = "p0"
    with pytest.raises(SchemaError, match="duplicate part ids"):
        validate(s)


def test_unknown_physics_rejected():
    s = _ok_schema()
    s["parts"][0]["physics"] = "magnetic"
    with pytest.raises(SchemaError, match="physics"):
        validate(s)


def test_unknown_collision_approx_rejected():
    s = _ok_schema()
    s["parts"][0]["collision"]["approx"] = "telekinesis"
    with pytest.raises(SchemaError, match="collision.approx"):
        validate(s)


def test_sdf_without_resolution_rejected():
    s = _ok_schema()
    del s["parts"][0]["collision"]["resolution"]
    with pytest.raises(SchemaError, match="resolution"):
        validate(s)


def test_invalid_scale_xyz_rejected():
    s = _ok_schema()
    s["parts"][0]["scale_xyz"] = [1, 1]  # wrong length
    with pytest.raises(SchemaError, match="scale_xyz"):
        validate(s)


# ---------------------------------------------------------------------------
# Joints
# ---------------------------------------------------------------------------


def test_joint_unknown_type_rejected():
    s = _ok_schema()
    s["joints"][0]["type"] = "wormhole"
    with pytest.raises(SchemaError, match="type"):
        validate(s)


def test_joint_parent_not_in_parts_rejected():
    s = _ok_schema()
    s["joints"][0]["parent"] = "ghost"
    with pytest.raises(SchemaError, match="parent"):
        validate(s)


def test_joint_child_not_in_parts_rejected():
    s = _ok_schema()
    s["joints"][0]["child"] = "ghost"
    with pytest.raises(SchemaError, match="child"):
        validate(s)


def test_joint_self_loop_rejected():
    s = _ok_schema()
    s["joints"][0]["child"] = "p0"  # parent == child
    with pytest.raises(SchemaError, match="same"):
        validate(s)


def test_revolute_missing_axis_rejected():
    s = _ok_schema()
    del s["joints"][0]["axis_p0"]
    with pytest.raises(SchemaError, match="axis_p0"):
        validate(s)


def test_prismatic_missing_axis_dir_rejected():
    s = _ok_schema()
    s["joints"][0] = {
        "id": "j0", "parent": "p0", "child": "p1", "type": "prismatic",
        "axis_origin": [0, 0, 0], "lower": 0, "upper": 0.05,
    }
    with pytest.raises(SchemaError, match="axis_dir"):
        validate(s)


def test_duplicate_joint_id_rejected():
    s = _ok_schema()
    s["parts"].append(_minimal_part("p2", physics="dynamic"))
    s["joints"].append(_minimal_revolute("j0", parent="p0", child="p2"))
    with pytest.raises(SchemaError, match="duplicate joint ids"):
        validate(s)


# ---------------------------------------------------------------------------
# Topology
# ---------------------------------------------------------------------------


def test_two_incoming_joints_to_same_child_rejected():
    s = _ok_schema()
    s["parts"].append(_minimal_part("p2"))
    # second joint targets p1 again
    s["joints"].append(_minimal_revolute("j1", parent="p2", child="p1"))
    with pytest.raises(SchemaError, match="multiple incoming"):
        validate(s)


def test_cycle_rejected():
    s = _ok_schema()
    s["parts"] = [_minimal_part("a"), _minimal_part("b", physics="dynamic"), _minimal_part("c", physics="dynamic")]
    s["joints"] = [
        _minimal_revolute("j0", parent="a", child="b"),
        _minimal_revolute("j1", parent="b", child="c"),
        _minimal_revolute("j2", parent="c", child="a"),
    ]
    with pytest.raises(SchemaError, match="cycle"):
        validate(s)


# ---------------------------------------------------------------------------
# External meshes
# ---------------------------------------------------------------------------


def test_external_mesh_attach_to_unknown_part_rejected():
    s = _ok_schema()
    s["external_meshes"] = [{
        "attach_to": "ghost",
        "glb": "x.glb",
        "transform": {"t": [0, 0, 0], "q_wxyz": [1, 0, 0, 0], "s": [1, 1, 1]},
    }]
    with pytest.raises(SchemaError, match="attach_to"):
        validate(s)


# ---------------------------------------------------------------------------
# topo_sort + joints_by_child
# ---------------------------------------------------------------------------


def test_topo_sort_root_before_child():
    s = _ok_schema()
    order = topo_sort(s)
    assert order.index("p0") < order.index("p1")


def test_topo_sort_chain_order():
    s = _ok_schema()
    s["parts"] = [_minimal_part("a"), _minimal_part("b", physics="dynamic"), _minimal_part("c", physics="dynamic")]
    s["joints"] = [
        _minimal_revolute("j0", parent="a", child="b"),
        _minimal_revolute("j1", parent="b", child="c"),
    ]
    validate(s)
    order = topo_sort(s)
    assert order == ["a", "b", "c"]


def test_joints_by_child_excludes_free():
    s = _ok_schema()
    s["parts"].append(_minimal_part("p2", physics="dynamic"))
    s["joints"].append({"id": "j_free", "parent": "p0", "child": "p2", "type": "free"})
    by_child = joints_by_child(s)
    assert "p1" in by_child
    assert "p2" not in by_child  # free joints excluded


# ---------------------------------------------------------------------------
# Schema v3: per-part sites[] (semantic AABB regions)
# ---------------------------------------------------------------------------


def _site(sid="screen_face", kind="screen",
          aabb_min=(-0.05, -0.05, 0.0), aabb_max=(0.05, 0.05, 0.005)) -> dict:
    return {"id": sid, "kind": kind, "aabb_min": list(aabb_min), "aabb_max": list(aabb_max)}


def test_v3_passes_with_empty_sites():
    s = _ok_schema(version=3)
    s["parts"][0]["sites"] = []
    validate(s)


def test_v3_passes_with_valid_sites():
    s = _ok_schema(version=3)
    s["parts"][0]["sites"] = [_site("a"), _site("b", kind="button")]
    validate(s)


def test_v2_still_accepted_after_v3_rollout():
    """Backward compat: a v2 file (no sites) must still validate so users
    don't have to migrate every old labels.json the moment v3 lands."""
    s = _ok_schema(version=2)
    validate(s)


def test_site_with_unknown_kind_rejected():
    s = _ok_schema(version=3)
    s["parts"][0]["sites"] = [_site(kind="bogus_kind")]
    with pytest.raises(SchemaError, match="sites\\[0\\].kind"):
        validate(s)


def test_site_aabb_min_must_be_le_max():
    s = _ok_schema(version=3)
    s["parts"][0]["sites"] = [_site(aabb_min=(0.1, 0, 0), aabb_max=(0, 0, 0))]
    with pytest.raises(SchemaError, match="aabb_min must be component-wise <="):
        validate(s)


def test_site_missing_aabb_rejected():
    s = _ok_schema(version=3)
    site = _site()
    del site["aabb_max"]
    s["parts"][0]["sites"] = [site]
    with pytest.raises(SchemaError, match="missing required field 'aabb_max'"):
        validate(s)


def test_duplicate_site_ids_within_part_rejected():
    s = _ok_schema(version=3)
    s["parts"][0]["sites"] = [_site("dup"), _site("dup", kind="button")]
    with pytest.raises(SchemaError, match="duplicate site ids"):
        validate(s)


def test_same_site_id_across_parts_is_ok():
    """Sites are scoped to their part — two parts can each have a 'screen'
    site without collision."""
    s = _ok_schema(version=3)
    s["parts"][0]["sites"] = [_site("screen")]
    s["parts"][1]["sites"] = [_site("screen")]
    validate(s)


# ---------------------------------------------------------------------------
# PR-C: site material_override (generic — allowed on any site kind)
# ---------------------------------------------------------------------------


def test_screen_site_with_valid_material_override_passes():
    s = _ok_schema(version=3)
    s["parts"][0]["sites"] = [{
        **_site(kind="screen"),
        "material_override": {"diffuseColor": [0.02, 0.02, 0.02], "roughness": 0.15, "metallic": 0.0},
    }]
    validate(s)


def test_material_override_allowed_on_any_site_kind():
    """material_override is generic — paint a button red, paint a handle
    rubber-black, etc. ``kind`` is just a semantic label."""
    for k in ("screen", "button", "camera", "handle", "custom"):
        s = _ok_schema(version=3)
        s["parts"][0]["sites"] = [{
            **_site(kind=k),
            "material_override": {"diffuseColor": [0.5, 0.1, 0.1]},
        }]
        validate(s)  # must not raise


def test_material_override_diffuse_must_be_3_floats_in_unit_range():
    s = _ok_schema(version=3)
    s["parts"][0]["sites"] = [{
        **_site(kind="screen"),
        "material_override": {"diffuseColor": [1.5, 0, 0]},  # >1
    }]
    with pytest.raises(SchemaError, match="diffuseColor"):
        validate(s)


def test_material_override_roughness_out_of_range_rejected():
    s = _ok_schema(version=3)
    s["parts"][0]["sites"] = [{
        **_site(kind="screen"),
        "material_override": {"diffuseColor": [0.1, 0.1, 0.1], "roughness": 2.0},
    }]
    with pytest.raises(SchemaError, match="roughness"):
        validate(s)
