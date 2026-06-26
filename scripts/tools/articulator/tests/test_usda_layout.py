"""Structural tests for the new USDA Visuals/Collisions/Joint layout (PR-A).

Drives ``build_usda()`` directly with synthetic part meshes — no Blender,
no GLB, no real fixture file is touched. The only on-disk asset read is
``tests/fixtures/fold_phone_labels.v2.json`` (a *backup copy* of the user's
fold-phone labels.json; user's live ``outputs/xiaomi_fold_4/`` is never
modified by these tests)."""
from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from build_usd import build_usda

FIXTURES = Path(__file__).parent / "fixtures"


# --------------------------------------------------------------------------
# Synthetic helpers — build a minimal valid v2 labels dict + tiny meshes
# --------------------------------------------------------------------------


def _unit_cube_mesh():
    """A 1m unit cube centred on the origin. Tuple shape matches what
    ``_read_part_npz`` produces: (verts, faces, uvs, fuvi)."""
    verts = [
        (0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 1.0, 0.0), (0.0, 1.0, 0.0),
        (0.0, 0.0, 1.0), (1.0, 0.0, 1.0), (1.0, 1.0, 1.0), (0.0, 1.0, 1.0),
    ]
    faces = [
        [0, 1, 2, 3], [4, 5, 6, 7],
        [0, 1, 5, 4], [2, 3, 7, 6],
        [1, 2, 6, 5], [0, 3, 7, 4],
    ]
    return (verts, faces, [], [])


def _z_tall_box_mesh():
    """A box whose longest dimension is Z in Blender/USD space."""
    verts = [
        (0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 2.0, 0.0), (0.0, 2.0, 0.0),
        (0.0, 0.0, 10.0), (1.0, 0.0, 10.0), (1.0, 2.0, 10.0), (0.0, 2.0, 10.0),
    ]
    faces = [
        [0, 1, 2, 3], [4, 5, 6, 7],
        [0, 1, 5, 4], [2, 3, 7, 6],
        [1, 2, 6, 5], [0, 3, 7, 4],
    ]
    return (verts, faces, [], [])


def _two_part_revolute_labels() -> dict:
    return {
        "version": 2,
        "device": "TestDevice",
        "physical_dims_mm": {"x": 100, "y": 100, "z": 100},
        "parts": [
            {
                "id": "part_a",
                "clusters": ["c0"],
                "physics": "kinematic",
                "collision": {"approx": "convexHull"},
                "scale_xyz": [1, 1, 1],
                "mass": 0.10,
            },
            {
                "id": "part_b",
                "clusters": ["c1"],
                "physics": "dynamic",
                "collision": {"approx": "sdf", "resolution": 64},
                "scale_xyz": [1, 1, 1],
                "mass": 0.05,
            },
        ],
        "joints": [
            {
                "id": "ab_hinge",
                "parent": "part_a",
                "child": "part_b",
                "type": "revolute",
                "axis_p0": [0.0, 0.0, 0.0],
                "axis_p1": [1.0, 0.0, 0.0],
                "lower": 0.0,
                "upper": 90.0,
                "drive": {"target": 0, "stiffness": 100, "damping": 10, "max_force": 50},
                "limit_hard": True,
            }
        ],
    }


def _free_part_labels() -> dict:
    return {
        "version": 2,
        "device": "FreeDevice",
        "physical_dims_mm": {"x": 100, "y": 100, "z": 100},
        "parts": [
            {
                "id": "loose_part",
                "clusters": ["c0"],
                "physics": "dynamic",
                "collision": {"approx": "convexHull"},
                "scale_xyz": [1, 1, 1],
            }
        ],
        "joints": [],
    }


def _build(labels: dict) -> str:
    meshes = {p["id"]: _unit_cube_mesh() for p in labels["parts"]}
    return build_usda(labels, meshes, with_ground=False)


def _body_span(usda: str, body_name: str) -> str:
    start = usda.index(f'def Xform "{body_name}"')
    brace = usda.index("{", start)
    depth = 0
    for i, ch in enumerate(usda[brace:], start=brace):
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return usda[start:i]
    raise AssertionError(f"{body_name} body close brace not found")


# --------------------------------------------------------------------------
# Structural assertions
# --------------------------------------------------------------------------


def test_stage_metadata_physics_and_ground_are_z_up():
    labels = _free_part_labels()
    meshes = {"loose_part": _unit_cube_mesh()}
    usda = build_usda(labels, meshes, with_ground=True)

    assert 'upAxis = "Z"' in usda
    assert 'vector3f physics:gravityDirection = (0, 0, -1)' in usda
    assert re.search(
        r'def Mesh "Ground".*?point3f\[\] points = \[\s*'
        r'\(-4, -4, 0\), \(4, -4, 0\), \(4, 4, 0\), \(-4, 4, 0\),\s*'
        r'\(-4, -4, -0\.1\), \(4, -4, -0\.1\), \(4, 4, -0\.1\), \(-4, 4, -0\.1\)',
        usda,
        re.DOTALL,
    )


def test_stage5_centers_xy_and_places_mesh_bottom_on_z_plane():
    labels = _free_part_labels()
    labels["physical_dims_mm"] = {"x": 100, "y": 100, "z": 100}
    meshes = {"loose_part": _z_tall_box_mesh()}
    usda = build_usda(labels, meshes, with_ground=False)
    body = _body_span(usda, "loose_part")

    assert (
        "float3[] extent = [(-0.005000, -0.010000, 0.000000), "
        "(0.005000, 0.010000, 0.100000)]"
    ) in body


def test_revolute_joint_schema_y_up_axis_is_emitted_as_usd_z_up():
    labels = _two_part_revolute_labels()
    labels["joints"][0]["axis_p0"] = [0.0, 0.0, 0.0]
    labels["joints"][0]["axis_p1"] = [0.0, 1.0, 0.0]
    usda = _build(labels)

    assert "point3f physics:localPos0 = (-0.050000, -0.050000, 0.000000)" in usda
    assert "quatf physics:localRot0 = (0.707107, 0.000000, -0.707107, 0.000000)" in usda


def test_site_schema_y_up_aabb_is_emitted_as_usd_z_up():
    labels = _free_part_labels()
    labels["version"] = 3
    labels["parts"][0]["sites"] = [
        {
            "id": "handle",
            "kind": "handle",
            "aabb_min": [0.0, 0.2, -0.4],
            "aabb_max": [1.0, 0.4, -0.2],
        }
    ]
    meshes = {"loose_part": _unit_cube_mesh()}
    usda = build_usda(labels, meshes, with_ground=False)

    assert "double3 xformOp:translate = (0.000000, -0.020000, 0.030000)" in usda
    assert "float3 xformOp:scale = (0.050000, 0.010000, 0.010000)" in usda


def test_part_emitted_as_xform_not_mesh():
    """Each rigid body must be an Xform (not a top-level Mesh) so that
    Visuals/Collisions can nest below it."""
    usda = _build(_two_part_revolute_labels())
    assert re.search(r'def Xform "part_a"', usda)
    assert re.search(r'def Xform "part_b"', usda)
    # Critical: there must NOT be a top-level Mesh prim with the part id.
    assert not re.search(r'^\s*def Mesh "part_a"', usda, re.MULTILINE)
    assert not re.search(r'^\s*def Mesh "part_b"', usda, re.MULTILINE)


def test_each_body_has_visuals_xform():
    usda = _build(_two_part_revolute_labels())
    # Two parts -> exactly two Visuals/ Xforms
    assert len(re.findall(r'def Xform "Visuals"', usda)) == 2


def test_each_body_has_collisions_xform_with_guide_purpose():
    usda = _build(_two_part_revolute_labels())
    # Each Collisions/ Xform must declare purpose="guide" as an attribute
    # inside the prim body. USDA does not allow uniform token attributes in
    # the prim metadata parentheses.
    matches = re.findall(
        r'def Xform "Collisions"\s*\{\s*\n\s*uniform token purpose = "guide"',
        usda
    )
    assert len(matches) == 2, usda
    assert not re.search(r'def Xform "Collisions"\s*\([^)]*uniform token purpose', usda)


def test_visual_mesh_carries_material_binding_only():
    """The Mesh inside Visuals/ has MaterialBindingAPI; collision stuff
    must NOT be on the visual mesh."""
    usda = _build(_two_part_revolute_labels())
    # Find the Visuals/mesh block for part_a and check its apiSchemas line
    m = re.search(
        r'def Xform "Visuals"\s*\{\s*\n\s*def Mesh "mesh"\s*\(\s*\n\s*'
        r'prepend apiSchemas = \[([^\]]+)\]',
        usda
    )
    assert m, "Visuals/mesh block not found"
    apis = m.group(1)
    assert '"MaterialBindingAPI"' in apis
    assert "PhysicsCollisionAPI" not in apis
    assert "PhysicsRigidBodyAPI" not in apis


def test_collision_mesh_carries_physics_apis_no_material():
    usda = _build(_two_part_revolute_labels())
    # collider_0 inside Collisions/
    m = re.search(
        r'def Xform "Collisions"[^\{]*\{\s*\n\s*uniform token purpose = "guide"\s*\n\s*'
        r'def Mesh "collider_0"\s*\(\s*\n\s*'
        r'prepend apiSchemas = \[([^\]]+)\]',
        usda
    )
    assert m, "Collisions/collider_0 block not found"
    apis = m.group(1)
    assert "PhysicsCollisionAPI" in apis
    assert "PhysicsMeshCollisionAPI" in apis
    assert "MaterialBindingAPI" not in apis


def test_body_xform_has_rigid_body_api():
    """RigidBodyAPI moves to the Xform (the rigid body), not the Mesh."""
    usda = _build(_two_part_revolute_labels())
    m = re.search(
        r'def Xform "part_a"\s*\(\s*\n\s*prepend apiSchemas = \[([^\]]+)\]',
        usda
    )
    assert m, "part_a body apiSchemas line not found"
    apis = m.group(1)
    assert "PhysicsRigidBodyAPI" in apis
    assert "PhysicsMassAPI" in apis


def test_kinematic_flag_on_body_xform_not_visual_mesh():
    usda = _build(_two_part_revolute_labels())
    # The kinematic flag must sit between the part_a body's opening brace
    # and its first nested Xform (Visuals/), not inside the visual mesh.
    body_re = re.compile(
        r'def Xform "part_a".*?\{(?P<inside>.*?)def Xform "Visuals"',
        re.DOTALL,
    )
    m = body_re.search(usda)
    assert m, "part_a body header not found"
    inside = m.group("inside")
    assert "physics:kinematicEnabled = 1" in inside, inside


def test_revolute_joint_nested_inside_child_body():
    """Joint must be a child of part_b's Xform, NOT a sibling."""
    usda = _build(_two_part_revolute_labels())
    # Find part_b block, joint should appear *before* part_b's closing brace.
    # We test this by ensuring the joint lives inside part_b's curly-brace span.
    pb_open = usda.index('def Xform "part_b"')
    joint_idx = usda.index('def PhysicsRevoluteJoint "ab_hinge"', pb_open)
    # Walk braces from part_b open to find its matching close.
    brace_open = usda.index("{", pb_open)
    depth = 0
    pb_close = None
    for i, ch in enumerate(usda[brace_open:], start=brace_open):
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                pb_close = i; break
    assert pb_close is not None
    assert brace_open < joint_idx < pb_close, (
        "PhysicsRevoluteJoint should be nested inside part_b's Xform"
    )


def test_joint_body_refs_point_to_xform_paths():
    """body0/body1 must reference the body Xform paths (e.g.
    /World/TestDevice/part_a), since RigidBodyAPI is on the Xform."""
    usda = _build(_two_part_revolute_labels())
    assert re.search(r'rel physics:body0 = </World/TestDevice/part_a>', usda)
    assert re.search(r'rel physics:body1 = </World/TestDevice/part_b>', usda)


def test_free_part_has_no_joint_block():
    """A part with no incoming joint must not get any joint prim emitted."""
    usda = _build(_free_part_labels())
    assert "PhysicsRevoluteJoint" not in usda
    assert "PhysicsFixedJoint" not in usda
    assert "PhysicsPrismaticJoint" not in usda
    # Body still emitted as Xform with V/C
    assert re.search(r'def Xform "loose_part"', usda)
    assert 'def Xform "Visuals"' in usda
    assert 'def Xform "Collisions"' in usda


def test_collision_approximation_on_collider_not_body():
    """`physics:approximation` and `sdfResolution` belong on the collider
    Mesh inside Collisions/, never on the body Xform or visual mesh."""
    usda = _build(_two_part_revolute_labels())
    # Locate part_b body span
    pb_open = usda.index('def Xform "part_b"')
    pb_brace = usda.index("{", pb_open)
    # Walk to its matching close
    depth = 0; pb_close = None
    for i, ch in enumerate(usda[pb_brace:], start=pb_brace):
        if ch == "{": depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0: pb_close = i; break
    body_text = usda[pb_open:pb_close]

    # The line `physics:approximation = "sdf"` must come *after* the
    # Collisions/ Xform header inside the body — i.e. nested deep,
    # not on the body Xform itself.
    coll_idx = body_text.index('def Xform "Collisions"')
    approx_idx = body_text.index('physics:approximation = "sdf"')
    assert approx_idx > coll_idx, "approximation must be inside Collisions/, not on body"
    assert "physxSDFMeshCollision:sdfResolution = 64" in body_text


# --------------------------------------------------------------------------
# Backup-fixture smoke (validates the user's REAL fold-phone labels still
# parses cleanly through the new emitter — without Blender, without writing
# anywhere near outputs/)
# --------------------------------------------------------------------------


def test_sites_block_omitted_when_no_sites():
    """A part with no sites must not get an empty Sites/ Xform."""
    usda = _build(_two_part_revolute_labels())
    assert 'def Xform "Sites"' not in usda


def test_sites_block_emitted_with_invisible_cubes():
    labels = _two_part_revolute_labels()
    labels["version"] = 3
    labels["parts"][0]["sites"] = [
        {
            "id": "screen_face",
            "kind": "screen",
            "aabb_min": [-0.4, 0.0, -0.3],
            "aabb_max": [0.4, 0.05, 0.3],
        },
        {
            "id": "power_button",
            "kind": "button",
            "aabb_min": [0.35, 0.02, 0.0],
            "aabb_max": [0.4, 0.04, 0.05],
        },
    ]
    meshes = {p["id"]: _unit_cube_mesh() for p in labels["parts"]}
    usda = build_usda(labels, meshes, with_ground=False)

    assert usda.count('def Xform "Sites"') == 1
    assert re.search(r'def Xform "Sites"\s*\{\s*\n\s*uniform token purpose = "guide"', usda)
    assert not re.search(r'def Xform "Sites"\s*\([^)]*uniform token purpose', usda)
    assert 'def Cube "screen_face"' in usda
    assert 'def Cube "power_button"' in usda
    assert usda.count('token visibility = "invisible"') >= 2
    assert 'userProperties:siteKind = "screen"' in usda
    assert 'userProperties:siteKind = "button"' in usda


def test_sites_xform_nested_inside_body_xform():
    labels = _two_part_revolute_labels()
    labels["version"] = 3
    labels["parts"][0]["sites"] = [
        {"id": "s0", "kind": "custom", "aabb_min": [0, 0, 0], "aabb_max": [0.1, 0.1, 0.1]}
    ]
    meshes = {p["id"]: _unit_cube_mesh() for p in labels["parts"]}
    usda = build_usda(labels, meshes, with_ground=False)
    # Sites/ must live inside part_a's Xform brace span
    pa_open = usda.index('def Xform "part_a"')
    sites_idx = usda.index('def Xform "Sites"', pa_open)
    pa_brace = usda.index("{", pa_open)
    depth = 0; pa_close = None
    for i, ch in enumerate(usda[pa_brace:], start=pa_brace):
        if ch == "{": depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0: pa_close = i; break
    assert pa_brace < sites_idx < pa_close


def test_site_with_material_override_emits_overlay_and_material():
    """A site with material_override -> a thin Cube overlay inside Visuals/,
    bound to a freshly defined Material under /World/Materials/. Works on
    any site kind (this test uses 'screen', the next uses 'button')."""
    labels = _two_part_revolute_labels()
    labels["version"] = 3
    labels["parts"][0]["sites"] = [
        {
            "id": "screen_face",
            "kind": "screen",
            "aabb_min": [-0.4, 0.04, -0.3],
            "aabb_max": [0.4, 0.05, 0.3],
            "material_override": {
                "diffuseColor": [0.02, 0.02, 0.02],
                "roughness": 0.15,
                "metallic": 0.0,
            },
        }
    ]
    meshes = {p["id"]: _unit_cube_mesh() for p in labels["parts"]}
    usda = build_usda(labels, meshes, with_ground=False)

    assert 'def Cube "screen_face_overlay"' in usda
    pa_open = usda.index('def Xform "part_a"')
    visuals_idx = usda.index('def Xform "Visuals"', pa_open)
    overlay_idx = usda.index('def Cube "screen_face_overlay"', pa_open)
    brace = usda.index("{", visuals_idx)
    depth = 0; vc_close = None
    for i, ch in enumerate(usda[brace:], start=brace):
        if ch == "{": depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0: vc_close = i; break
    assert visuals_idx < overlay_idx < vc_close, "site overlay must live inside Visuals/"
    assert 'def Material "part_a_screen_face_OverrideMat"' in usda
    assert 'rel material:binding = </World/Materials/part_a_screen_face_OverrideMat>' in usda
    assert 'color3f inputs:diffuseColor = (0.0200, 0.0200, 0.0200)' in usda


def test_button_site_with_override_also_gets_overlay():
    """Override is generic — a button-kind site with override should emit an
    overlay just like a screen-kind one."""
    labels = _two_part_revolute_labels()
    labels["version"] = 3
    labels["parts"][0]["sites"] = [
        {
            "id": "power_button",
            "kind": "button",
            "aabb_min": [0.35, 0.02, 0.0],
            "aabb_max": [0.4, 0.04, 0.05],
            "material_override": {"diffuseColor": [0.8, 0.1, 0.1]},
        }
    ]
    meshes = {p["id"]: _unit_cube_mesh() for p in labels["parts"]}
    usda = build_usda(labels, meshes, with_ground=False)
    assert 'def Cube "power_button_overlay"' in usda
    assert 'def Material "part_a_power_button_OverrideMat"' in usda
    assert 'color3f inputs:diffuseColor = (0.8000, 0.1000, 0.1000)' in usda


def test_site_without_override_emits_no_overlay():
    labels = _two_part_revolute_labels()
    labels["version"] = 3
    labels["parts"][0]["sites"] = [
        {"id": "screen_face", "kind": "screen",
         "aabb_min": [-0.4, 0.04, -0.3], "aabb_max": [0.4, 0.05, 0.3]}
    ]
    meshes = {p["id"]: _unit_cube_mesh() for p in labels["parts"]}
    usda = build_usda(labels, meshes, with_ground=False)
    assert 'def Cube "screen_face"' in usda          # invisible Sites/ cube
    assert "_overlay" not in usda                     # no Visuals/ overlay
    assert "OverrideMat" not in usda


def test_fold_phone_backup_fixture_emits_v_c_layout():
    """Drive the new emitter with the BACKED-UP fold-phone labels.
    The user's live ``outputs/xiaomi_fold_4/`` is not read or modified."""
    labels_path = FIXTURES / "fold_phone_labels.v2.json"
    assert labels_path.exists(), "fixture missing"
    labels = json.loads(labels_path.read_text())
    meshes = {p["id"]: _unit_cube_mesh() for p in labels["parts"]}
    usda = build_usda(labels, meshes, with_ground=False)
    assert 'def Xform "main_screen"' in usda
    assert 'def Xform "fold_screen"' in usda
    assert usda.count('def Xform "Visuals"') == 2
    assert usda.count('def Xform "Collisions"') == 2
    # Joint nested in fold_screen (the child)
    assert re.search(
        r'def Xform "fold_screen".*?def PhysicsRevoluteJoint "main_screen_fold_screen_joint"',
        usda, re.DOTALL,
    )
