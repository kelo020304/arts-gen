from io import BytesIO
import json
from pathlib import Path
import subprocess

import pytest

from post_process.kinematic_solver.sdk.mjcf_preview import write_iteration_mjcf_preview
from post_process.kinematic_solver.sdk.schemas import EstimateContext, LimitEstimate
from post_process.object_post_process.kinematic_workbench import _build_static_mesh_mjcf
from post_process.object_post_process.server import create_app
from post_process.object_post_process.web_editor import _build_parser


def _write_obj(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(
            [
                "v 0 0 0",
                "v 1 0 0",
                "v 0 1 0",
                "v 0 0 1",
                "f 1 2 3",
                "f 1 2 4",
                "f 1 3 4",
                "f 2 3 4",
            ]
        )
    )


def _write_offset_obj(path: Path, center: tuple[float, float, float]) -> None:
    cx, cy, cz = center
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(
            [
                f"v {cx - 0.5} {cy - 0.5} {cz - 0.5}",
                f"v {cx + 0.5} {cy - 0.5} {cz - 0.5}",
                f"v {cx - 0.5} {cy + 0.5} {cz - 0.5}",
                f"v {cx - 0.5} {cy - 0.5} {cz + 0.5}",
                "f 1 2 3",
                "f 1 2 4",
                "f 1 3 4",
                "f 2 3 4",
            ]
        )
    )


def _obj_center(path: Path) -> tuple[float, float, float]:
    vertices = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.startswith("v "):
            continue
        _, x, y, z = line.split()[:4]
        vertices.append((float(x), float(y), float(z)))
    assert vertices
    count = float(len(vertices))
    return tuple(sum(vertex[i] for vertex in vertices) / count for i in range(3))


def _write_vhacd_placeholder(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{}", encoding="utf-8")


def _save_payload_from_manifest(manifest: dict) -> dict:
    root_bodies = [
        body for body in manifest["bodies"]
        if body.get("parent") in {None, "", "world"}
    ]
    assert len(root_bodies) == 1
    return {
        "xml_path": manifest["xml_path"],
        "scene_graph": {
            "root_body": root_bodies[0]["name"],
            "bodies": [
                {
                    "id": body["id"],
                    "name": body["name"],
                    "parent": None if body.get("parent") in {None, "", "world"} else body["parent"],
                    "pos": body["pos"],
                    "quat": body["quat"],
                    "visual_geoms": body["visual_geoms"],
                    "collision_geoms": body["collision_geoms"],
                    "joint_ids": body.get("joint_ids", []),
                    "merged_from": body.get("merged_from", []),
                }
                for body in manifest["bodies"]
            ],
            "joints": [
                {
                    "id": joint["id"],
                    "name": joint["name"],
                    "body": joint["body"],
                    "type": joint["type"],
                    "anchor": joint["anchor"],
                    "axis": joint["axis"],
                    "range": joint["range"],
                    "default": joint["default"],
                    "order": index,
                }
                for index, joint in enumerate(manifest["joints"])
            ],
        },
    }


def test_post_process_server_serves_kinematic_run_state_and_manifest(tmp_path):
    converter_root = tmp_path / "converter"
    _write_obj(converter_root / "raw/partseg/ra_063/objs/body.obj")
    _write_obj(converter_root / "raw/partseg/ra_063/objs/part_02.obj")
    run_dir = tmp_path / "run"
    ctx = EstimateContext(
        object_id="ra_063",
        joints={
            "part_02": {
                "joint_name": "part_02",
                "type": "prismatic",
                "canonical_unit": "meters",
                "origin_world": [0.0, 0.0, 0.0],
                "axis_world": [0.0, 1.0, 0.0],
                "moving_parts": ["part_02"],
                "static_parts": ["body"],
            },
        },
        evidence={},
    )
    write_iteration_mjcf_preview(
        ctx,
        [LimitEstimate(joint_name="part_02", lower=0.0, upper=0.15)],
        converter_output_root=converter_root,
        run_dir=run_dir,
        iteration=1,
    )
    app = create_app(
        assets_root=tmp_path / "unused_assets",
        kinematic_run_dir=run_dir,
        kinematic_run_id="unit_run",
    )
    client = app.test_client()

    page = client.get("/kinematic-agent/unit_run")
    assert page.status_code == 200
    assert b"viewer-canvas" in page.data
    assert b"kinematic-agent-panel" in page.data
    assert b"kinematic-agent-progress-fill" in page.data
    assert b"kinematic-agent-progress-label" in page.data

    state = client.get("/api/kinematic-agent/unit_run/state")
    assert state.status_code == 200
    state_payload = state.get_json()
    assert state_payload["latest_iteration"] == 1
    assert state_payload["latest_preview"]["playback"]["mode"] == "sequential_full_range"

    manifest = client.get("/api/kinematic-agent/unit_run/manifest")
    assert manifest.status_code == 200
    manifest_payload = manifest.get_json()
    assert manifest_payload["status"] == "ok"
    assert manifest_payload["joints"][0]["name"] == "part_02"
    body = next(item for item in manifest_payload["bodies"] if item["name"] == "body")
    assert body["mesh_file"] == "assets/body.obj"
    assert manifest_payload["asset_base_url"].endswith("/mjcf/assets")

    old_frontend_mesh_url = (
        manifest_payload["asset_base_url"] + "/" + body["mesh_file"]
    )
    assert "/assets/assets/body.obj" in old_frontend_mesh_url
    assert client.get(old_frontend_mesh_url).status_code == 404

    normalized_mesh_url = (
        manifest_payload["asset_base_url"] + "/" + body["mesh_file"].removeprefix("assets/")
    )
    assert client.get(normalized_mesh_url).status_code == 200


def test_workbench_kinematic_run_serves_iteration_preview_meshes(tmp_path):
    converter_root = tmp_path / "converter"
    _write_obj(converter_root / "raw/partseg/ra_036/objs/body.obj")
    _write_obj(converter_root / "raw/partseg/ra_036/objs/part_00.obj")
    workbench_root = tmp_path / "workbench"
    run_dir = workbench_root / "runs" / "ra_036_agent_run_unit"
    ctx = EstimateContext(
        object_id="ra_036",
        joints={
            "part_00": {
                "joint_name": "part_00",
                "type": "revolute",
                "canonical_unit": "radians",
                "origin_world": [0.0, 0.0, 0.0],
                "axis_world": [0.0, 0.0, 1.0],
                "moving_parts": ["part_00"],
                "static_parts": ["body"],
            },
        },
        evidence={},
    )
    write_iteration_mjcf_preview(
        ctx,
        [LimitEstimate(joint_name="part_00", lower=-0.2, upper=0.2)],
        converter_output_root=converter_root,
        run_dir=run_dir,
        iteration=0,
    )
    app = create_app(
        assets_root=tmp_path / "unused_assets",
        kinematic_workbench_root=workbench_root,
    )
    client = app.test_client()

    manifest = client.get("/api/kinematic-agent/ra_036_agent_run_unit/manifest")

    assert manifest.status_code == 200
    manifest_payload = manifest.get_json()
    assert manifest_payload["status"] == "ok"
    body = next(item for item in manifest_payload["bodies"] if item["name"] == "body")
    part = next(item for item in manifest_payload["bodies"] if item["name"] == "part_00")
    body_mesh_url = (
        manifest_payload["asset_base_url"] + "/" + body["mesh_file"].removeprefix("assets/")
    )
    part_mesh_url = (
        manifest_payload["asset_base_url"] + "/" + part["mesh_file"].removeprefix("assets/")
    )
    assert client.get(body_mesh_url).status_code == 200
    assert client.get(part_mesh_url).status_code == 200


def test_post_process_agent_page_has_copyable_diagnostics_box(tmp_path):
    app = create_app(
        assets_root=tmp_path / "unused_assets",
        kinematic_run_dir=tmp_path / "run",
        kinematic_run_id="unit_run",
    )
    client = app.test_client()

    page = client.get("/kinematic-agent/unit_run")

    assert page.status_code == 200
    assert b"kinematic-agent-diagnostics-copy" in page.data
    assert b"kinematic-agent-copy-diagnostics" in page.data
    assert b"copyKinematicDiagnostics" in page.data
    assert b"stopKinematicPlaybackAtManualState" in page.data
    assert b"normalizeAssetPath" in page.data
    assert b"createFallbackMeshMaterial" in page.data
    assert b"forceOpaquePreviewMaterial" in page.data
    assert b"THREE.DoubleSide" in page.data
    assert b"kinematicMaxAgentIterations" in page.data
    assert b"set-global-orientation" in page.data
    assert b"applyWorkbenchGlobalOrientationToRoot" in page.data
    assert b"applyKinematicAgentOrientation" in page.data
    assert b"orientation_degrees" in page.data
    assert b"reversible attempt" not in page.data


def test_joint_editor_supports_merge_into_selected_target(tmp_path):
    app = create_app(
        assets_root=tmp_path / "unused_assets",
    )
    client = app.test_client()

    page = client.get("/object-post-process/")

    assert page.status_code == 200
    assert b"merge-target-select" in page.data
    assert b"Merge Into Target" in page.data
    assert b"handleMergeIntoTargetBody" in page.data


def test_web_editor_cli_accepts_kinematic_run_fixture_dir():
    args = _build_parser().parse_args([
        "--kinematic-run-dir",
        "kin_test/ra_063",
        "--kinematic-run-id",
        "ra_063",
        "--no-browser",
    ])

    assert str(args.kinematic_run_dir) == "kin_test/ra_063"
    assert args.kinematic_run_id == "ra_063"

    workbench_args = _build_parser().parse_args([
        "--kinematic-workbench",
        "--kinematic-workspace-root",
        "kin_test/workbench",
        "--no-browser",
    ])
    assert workbench_args.kinematic_workbench is True
    assert str(workbench_args.kinematic_workspace_root) == "kin_test/workbench"


def test_post_process_server_serves_standalone_kinematic_workbench(tmp_path):
    app = create_app(
        assets_root=tmp_path / "unused_assets",
        kinematic_workbench_root=tmp_path / "workbench",
    )
    client = app.test_client()

    page = client.get("/kinematic-workbench")

    assert page.status_code == 200
    assert b"kinematic-workbench" in page.data
    assert b"Asset Setup" in page.data
    assert b"Agent Run" in page.data
    assert b"source-path-input" in page.data
    assert b"source-folder-button" in page.data
    assert b"importUploadedAsset" in page.data
    assert b"autoImportSelectedFiles" in page.data
    assert b"automatic upload/import" in page.data
    assert b"agent-run-button" in page.data
    assert b"initial-json-editor" in page.data
    assert b"save-initial-json-button" in page.data
    assert b"load-template-json-button" in page.data
    assert b"saveManualInitialJson" in page.data
    assert b"loadInitialJsonTemplate" in page.data
    assert b"saveCurrentPreviewXmlIfPossible" in page.data
    assert b"handleSaveXml" in page.data
    assert b"syncing merged partseg and cooking VHACD" in page.data
    assert b"partseg_bootstrap=" in page.data
    assert b"vhacd_cache=" in page.data
    assert b"target_path: $('initial-json-input').value.trim()" in page.data
    assert b"generated_mesh_dir: workbenchState.generatedMeshDir" in page.data
    assert b"workbench_asset_name: workbenchState.assetName" in page.data
    assert b"orientation_degrees: getOrientation()" in page.data
    assert b"return saveManualInitialJson({ quiet: true });" in page.data
    assert b"normalized_json_text" in page.data
    assert b"buildInitialJsonTemplate" in page.data
    assert b"syncObjectDerivedPaths" in page.data
    assert b"syncInitialJsonObjectId" in page.data
    assert b"syncObjectFromViewerAsset" in page.data
    assert b"frame.contentWindow.location.pathname" in page.data
    assert b"defaultKinematicRoot" in page.data
    assert b"source/model" in page.data
    assert b"default_initial_joints_root" in page.data
    assert b"/vlm_initial.json" in page.data
    assert b"/run_1" in page.data
    assert b"DEFAULT_INITIAL_JSON_TEXT = JSON.stringify" not in page.data


def test_kinematic_workbench_config_uses_env_without_exposing_key(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "secret-test-key")
    monkeypatch.setenv("OPENROUTER_BASE_URL", "https://router.example/v1")
    monkeypatch.setenv("ARTICRAFT_MODEL", "gpt-test")
    monkeypatch.setenv("ARTICRAFT_THINKING_LEVEL", "medium")
    app = create_app(
        assets_root=tmp_path / "unused_assets",
        kinematic_workbench_root=tmp_path / "workbench",
    )
    client = app.test_client()

    response = client.get("/api/kinematic-workbench/config")

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["status"] == "ok"
    assert payload["openrouter_base_url"] == "https://router.example/v1"
    assert payload["model"] == "gpt-test"
    assert payload["thinking_level"] == "medium"
    assert payload["default_source_root"] == "data/RealAppliance"
    assert payload["default_model_root"].endswith("data/RealAppliance/source/model")
    assert payload["default_initial_joints_root"].endswith("workbench/initial_joints")
    assert Path(payload["project_root"]).name == "arts-reconstruction"
    assert payload["has_openrouter_api_key"] is True
    assert "secret-test-key" not in response.get_data(as_text=True)


def test_kinematic_workbench_imports_mjcf_bundle_for_preview(tmp_path):
    source_dir = tmp_path / "source"
    _write_obj(source_dir / "assets/body.obj")
    (source_dir / "toy.xml").write_text(
        "\n".join(
            [
                '<mujoco model="toy">',
                '  <compiler angle="radian" meshdir="."/>',
                '  <asset><mesh name="body_mesh" file="assets/body.obj"/></asset>',
                '  <worldbody>',
                '    <body name="body" pos="0 0 0">',
                '      <geom name="body_visual" type="mesh" mesh="body_mesh" group="2" contype="0" conaffinity="0"/>',
                '    </body>',
                '  </worldbody>',
                '</mujoco>',
            ]
        ),
        encoding="utf-8",
    )
    app = create_app(
        assets_root=tmp_path / "unused_assets",
        kinematic_workbench_root=tmp_path / "workbench",
    )
    client = app.test_client()

    response = client.post(
        "/api/kinematic-workbench/import",
        json={
            "source_path": str(source_dir / "toy.xml"),
            "object_id": "toy_asset",
            "xml_save_root": str(tmp_path / "xml_out"),
            "mesh_save_root": str(tmp_path / "mesh_out"),
            "orientation_degrees": {"roll": 0, "pitch": 0, "yaw": 0},
        },
    )

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["status"] == "ok"
    assert payload["asset_name"] == "toy_asset"
    assert payload["viewer_url"] == "/object-post-process/toy_asset"
    assert Path(payload["generated_xml_path"]).is_file()
    assert Path(payload["generated_mesh_dir"], "body.obj").is_file()
    manifest = client.get("/api/assets/toy_asset/preview-manifest")
    assert manifest.status_code == 200
    assert manifest.get_json()["status"] == "ok"


def test_kinematic_workbench_imports_urdf_bundle_for_preview(tmp_path):
    source_dir = tmp_path / "urdf_source"
    _write_obj(source_dir / "body.obj")
    _write_obj(source_dir / "drawer.obj")
    (source_dir / "drawer.urdf").write_text(
        "\n".join(
            [
                '<robot name="drawer">',
                '  <link name="body">',
                '    <visual><geometry><mesh filename="body.obj"/></geometry></visual>',
                '  </link>',
                '  <link name="drawer">',
                '    <visual><geometry><mesh filename="drawer.obj"/></geometry></visual>',
                '  </link>',
                '  <joint name="drawer_joint" type="prismatic">',
                '    <parent link="body"/>',
                '    <child link="drawer"/>',
                '    <origin xyz="0 0 0" rpy="0 0 0"/>',
                '    <axis xyz="1 0 0"/>',
                '    <limit lower="0" upper="0.12"/>',
                '  </joint>',
                '</robot>',
            ]
        ),
        encoding="utf-8",
    )
    app = create_app(
        assets_root=tmp_path / "unused_assets",
        kinematic_workbench_root=tmp_path / "workbench",
    )
    client = app.test_client()

    response = client.post(
        "/api/kinematic-workbench/import",
        json={
            "source_path": str(source_dir / "drawer.urdf"),
            "object_id": "drawer_asset",
            "xml_save_root": str(tmp_path / "xml_out"),
            "mesh_save_root": str(tmp_path / "mesh_out"),
        },
    )

    assert response.status_code == 200
    payload = response.get_json()
    assert Path(payload["generated_xml_path"]).is_file()
    assert Path(payload["generated_mesh_dir"], "drawer.obj").is_file()
    manifest = client.get("/api/assets/drawer_asset/preview-manifest").get_json()
    assert manifest["status"] == "ok"
    assert manifest["joints"][0]["name"] == "drawer_joint"
    assert manifest["joints"][0]["type"] == "slide"


def test_kinematic_workbench_imports_usd_bundle_for_preview(tmp_path):
    pytest.importorskip("pxr")
    from pxr import Gf, Usd, UsdGeom, Vt

    source_path = tmp_path / "toy.usda"
    stage = Usd.Stage.CreateNew(str(source_path))
    mesh = UsdGeom.Mesh.Define(stage, "/part_07")
    mesh.CreatePointsAttr(
        Vt.Vec3fArray(
            [
                Gf.Vec3f(0, 0, 0),
                Gf.Vec3f(1, 0, 0),
                Gf.Vec3f(0, 1, 0),
                Gf.Vec3f(0, 0, 1),
            ]
        )
    )
    mesh.CreateFaceVertexCountsAttr([3, 3, 3, 3])
    mesh.CreateFaceVertexIndicesAttr([0, 1, 2, 0, 1, 3, 0, 2, 3, 1, 2, 3])
    stage.GetRootLayer().Save()
    app = create_app(
        assets_root=tmp_path / "unused_assets",
        kinematic_workbench_root=tmp_path / "workbench",
    )
    client = app.test_client()

    response = client.post(
        "/api/kinematic-workbench/import",
        json={
            "source_path": str(source_path),
            "object_id": "usd_asset",
            "xml_save_root": str(tmp_path / "xml_out"),
            "mesh_save_root": str(tmp_path / "mesh_out"),
        },
    )

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["source_type"] == "usda"
    assert Path(payload["generated_xml_path"]).is_file()
    assert Path(payload["generated_mesh_dir"], "part_07.obj").is_file()
    manifest = client.get("/api/assets/usd_asset/preview-manifest").get_json()
    assert manifest["status"] == "ok"
    assert manifest["bodies"][0]["name"] == "root"
    assert {body["name"] for body in manifest["bodies"]} == {"root", "part_07"}
    save_response = client.post(
        "/api/assets/usd_asset/save-xml",
        json=_save_payload_from_manifest(manifest),
    )
    assert save_response.status_code == 200
    assert save_response.get_json()["status"] == "ok"


def test_static_usd_mjcf_uses_single_saveable_root_body():
    root = _build_static_mesh_mjcf(
        "usd_asset",
        [("part_07", "part_07.obj"), ("body", "body.obj")],
        preview=True,
    )

    worldbody = root.find("worldbody")
    assert worldbody is not None
    root_bodies = worldbody.findall("body")
    assert [body.get("name") for body in root_bodies] == ["root"]
    child_names = [body.get("name") for body in root_bodies[0].findall("body")]
    assert child_names == ["part_07", "body"]


def test_kinematic_workbench_saves_manual_initial_joints_json(tmp_path):
    app = create_app(
        assets_root=tmp_path / "unused_assets",
        kinematic_workbench_root=tmp_path / "workbench",
    )
    client = app.test_client()

    response = client.post(
        "/api/kinematic-workbench/initial-joints-json",
        json={
            "object_id": "ra_063",
            "json_text": json.dumps({
                "joints": [
                    {
                        "name": "part_02",
                        "type": "prismatic",
                        "axis": [1, 0, 0],
                        "limit": [0, 30],
                    }
                ]
            }),
        },
    )

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["status"] == "ok"
    saved_path = Path(payload["initial_joints_json"])
    assert saved_path.is_file()
    assert saved_path.name == "vlm_initial.json"
    assert json.loads(saved_path.read_text(encoding="utf-8"))["joints"][0]["name"] == "part_02"


def test_kinematic_workbench_saves_manual_initial_joints_json_to_requested_file(tmp_path):
    app = create_app(
        assets_root=tmp_path / "unused_assets",
        kinematic_workbench_root=tmp_path / "workbench",
    )
    client = app.test_client()
    requested_path = tmp_path / "custom_root" / "ra_036_init.json"

    response = client.post(
        "/api/kinematic-workbench/initial-joints-json",
        json={
            "object_id": "ra_036",
            "target_path": str(requested_path),
            "json_text": json.dumps({
                "object_id": "ra_036",
                "initial_joints": {
                    "part_07": {
                        "type": "prismatic",
                        "axis": [1, 0, 0],
                        "limit": [0, 30],
                    }
                },
            }),
        },
    )

    assert response.status_code == 200
    payload = response.get_json()
    assert Path(payload["initial_joints_json"]) == requested_path
    assert json.loads(requested_path.read_text(encoding="utf-8"))["object_id"] == "ra_036"


def test_kinematic_workbench_saves_manual_initial_joints_json_normalizes_object_id(tmp_path):
    app = create_app(
        assets_root=tmp_path / "unused_assets",
        kinematic_workbench_root=tmp_path / "workbench",
    )
    client = app.test_client()
    requested_path = tmp_path / "custom_root" / "ra_036.json"

    response = client.post(
        "/api/kinematic-workbench/initial-joints-json",
        json={
            "object_id": "ra_036",
            "target_path": str(requested_path),
            "json_text": json.dumps({
                "object_id": "ra_063",
                "initial_joints": {
                    "part_02": {
                        "type": "prismatic",
                        "axis": [1, 0, 0],
                        "limit": [0, 30],
                    }
                },
            }),
        },
    )

    assert response.status_code == 200
    payload = response.get_json()
    saved = json.loads(requested_path.read_text(encoding="utf-8"))
    assert saved["object_id"] == "ra_036"
    assert payload["object_id_normalized"] is True
    assert json.loads(payload["normalized_json_text"])["object_id"] == "ra_036"


def test_kinematic_workbench_saves_manual_initial_joints_json_to_requested_root(tmp_path):
    app = create_app(
        assets_root=tmp_path / "unused_assets",
        kinematic_workbench_root=tmp_path / "workbench",
    )
    client = app.test_client()
    requested_root = tmp_path / "custom_initial_json_root"

    response = client.post(
        "/api/kinematic-workbench/initial-joints-json",
        json={
            "object_id": "ra_036",
            "target_path": str(requested_root),
            "json_text": json.dumps({"object_id": "ra_036", "initial_joints": {}}),
        },
    )

    assert response.status_code == 200
    saved_path = Path(response.get_json()["initial_joints_json"])
    assert saved_path == requested_root / "ra_036" / "vlm_initial.json"
    assert saved_path.is_file()


def test_kinematic_workbench_initial_joints_template_uses_workbench_asset_bodies(tmp_path):
    asset_root = tmp_path / "workbench" / "object_assets" / "ra_036" / "mjcf"
    mesh_dir = asset_root / "assets"
    mesh_dir.mkdir(parents=True)
    _write_obj(mesh_dir / "body.obj")
    _write_obj(mesh_dir / "knob.obj")
    _write_obj(mesh_dir / "drawer.obj")
    (asset_root / "ra_036.xml").write_text(
        """<mujoco>
  <asset>
    <mesh name="body_mesh" file="assets/body.obj"/>
    <mesh name="knob_mesh" file="assets/knob.obj"/>
    <mesh name="drawer_mesh" file="assets/drawer.obj"/>
  </asset>
  <worldbody>
    <body name="body">
      <geom name="body" type="mesh" mesh="body_mesh" group="2" contype="0" conaffinity="0"/>
      <body name="knob"><geom name="knob" type="mesh" mesh="knob_mesh" group="2" contype="0" conaffinity="0"/></body>
      <body name="drawer"><geom name="drawer" type="mesh" mesh="drawer_mesh" group="2" contype="0" conaffinity="0"/></body>
    </body>
  </worldbody>
</mujoco>""",
        encoding="utf-8",
    )
    app = create_app(
        assets_root=tmp_path / "unused_assets",
        kinematic_workbench_root=tmp_path / "workbench",
    )
    client = app.test_client()

    response = client.post(
        "/api/kinematic-workbench/initial-joints-template",
        json={"object_id": "ra_036", "workbench_asset_name": "ra_036"},
    )

    assert response.status_code == 200
    payload = response.get_json()
    template = json.loads(payload["json_text"])
    assert template["object_id"] == "ra_036"
    assert sorted(template["initial_joints"]) == ["drawer", "knob"]
    assert template["initial_joints"]["drawer"]["moving_parts"] == ["drawer"]
    assert template["initial_joints"]["drawer"]["parent"] == "body"


def test_kinematic_workbench_rejects_invalid_manual_initial_json(tmp_path):
    app = create_app(
        assets_root=tmp_path / "unused_assets",
        kinematic_workbench_root=tmp_path / "workbench",
    )
    client = app.test_client()

    response = client.post(
        "/api/kinematic-workbench/initial-joints-json",
        json={"object_id": "ra_063", "json_text": "{bad json"},
    )

    assert response.status_code == 400
    assert "valid JSON" in response.get_json()["message"]


def test_kinematic_workbench_uploads_browser_selected_usd_for_preview(tmp_path):
    usda_text = "\n".join(
        [
            "#usda 1.0",
            'def Mesh "mesh_0"',
            "{",
            "    point3f[] points = [(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)]",
            "    int[] faceVertexCounts = [3, 3, 3, 3]",
            "    int[] faceVertexIndices = [0, 1, 2, 0, 1, 3, 0, 2, 3, 1, 2, 3]",
            "}",
        ]
    ).encode("utf-8")
    app = create_app(
        assets_root=tmp_path / "unused_assets",
        kinematic_workbench_root=tmp_path / "workbench",
    )
    client = app.test_client()

    response = client.post(
        "/api/kinematic-workbench/import-upload",
        data={
            "object_id": "uploaded_usd",
            "source_relative_path": "Aligned.usda",
            "xml_save_root": str(tmp_path / "xml_out"),
            "mesh_save_root": str(tmp_path / "mesh_out"),
            "files": (BytesIO(usda_text), "Aligned.usda"),
        },
        content_type="multipart/form-data",
    )

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["status"] == "ok"
    assert payload["source_type"] == "usda"
    assert Path(payload["upload_root"], "Aligned.usda").is_file()
    assert Path(payload["generated_xml_path"]).is_file()
    manifest = client.get("/api/assets/uploaded_usd/preview-manifest").get_json()
    assert manifest["status"] == "ok"


def test_kinematic_workbench_starts_agent_subprocess(tmp_path, monkeypatch):
    calls = []

    class DummyProcess:
        pid = 12345

    def fake_popen(cmd, cwd, env, stdout, stderr, start_new_session):
        calls.append({
            "cmd": cmd,
            "cwd": cwd,
            "env": env,
            "stderr": stderr,
            "start_new_session": start_new_session,
        })
        return DummyProcess()

    monkeypatch.setenv("OPENROUTER_API_KEY", "server-env-key")
    monkeypatch.setenv("OPENROUTER_BASE_URL", "https://router.example/v1")
    monkeypatch.setattr(subprocess, "Popen", fake_popen)
    converter_root = tmp_path / "converter"
    _write_obj(converter_root / "raw" / "partseg" / "ra_063" / "objs" / "body.obj")
    _write_vhacd_placeholder(converter_root / "raw" / "vhacd" / "ra_063" / "body.json")
    initial_json = tmp_path / "vlm_initial.json"
    initial_json.write_text(json.dumps({"object_id": "ra_063", "initial_joints": {}}), encoding="utf-8")
    app = create_app(
        assets_root=tmp_path / "unused_assets",
        kinematic_workbench_root=tmp_path / "workbench",
    )
    client = app.test_client()

    response = client.post(
        "/api/kinematic-workbench/run-agent",
            json={
                "object_id": "ra_063",
                "converter_output_root": str(converter_root),
                "source_root": str(tmp_path / "source"),
                "initial_joints_json": str(initial_json),
                "out_dir": str(tmp_path / "run"),
                "model": "gpt-test",
                "thinking_level": "high",
            "max_agent_iterations": 7,
            "api_heartbeat_seconds": 1,
            "orientation_degrees": {"roll": 0, "pitch": 0, "yaw": 90},
        },
    )

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["status"] == "ok"
    assert payload["run_id"] == "run"
    assert payload["viewer_url"] == "/kinematic-agent/run"
    assert payload["state_url"] == "/api/kinematic-agent/run/state"
    assert "server-env-key" not in response.get_data(as_text=True)
    assert calls
    cmd = calls[0]["cmd"]
    assert cmd[:3] == ["/home/mi/anaconda3/envs/env-isaacsim/bin/python", "-m", "post_process.kinematic_solver.estimate_limit"]
    assert "--agent-loop" in cmd
    assert "--live-viewer" in cmd
    assert "--no-live-server" in cmd
    assert cmd[cmd.index("--max-agent-iterations") + 1] == "7"
    assert calls[0]["env"]["OPENROUTER_API_KEY"] == "server-env-key"
    assert calls[0]["env"]["ARTICRAFT_MODEL"] == "gpt-test"
    assert calls[0]["env"]["ARTICRAFT_THINKING_LEVEL"] == "high"
    assert payload["initial_joints_json"] == str(initial_json)
    orientation_path = Path(payload["out_dir"]) / "workbench_orientation.json"
    orientation_payload = json.loads(orientation_path.read_text(encoding="utf-8"))
    assert orientation_payload["orientation_degrees"] == {"roll": 0.0, "pitch": 0.0, "yaw": 90.0}
    assert "viewer-only" in orientation_payload["note"]


def test_kinematic_agent_state_keeps_run_orientation_as_source_metadata(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "frontend_state.json").write_text(
        json.dumps({
            "latest_iteration": 0,
            "latest_preview": {"asset_name": "ks_test", "iteration": 0},
            "iterations": [{"asset_name": "ks_test", "iteration": 0}],
        }),
        encoding="utf-8",
    )
    (run_dir / "workbench_orientation.json").write_text(
        json.dumps({"orientation_degrees": {"roll": 0, "pitch": 0, "yaw": 90}}),
        encoding="utf-8",
    )
    app = create_app(
        assets_root=tmp_path / "unused_assets",
        kinematic_run_dir=run_dir,
        kinematic_run_id="unit_run",
    )
    client = app.test_client()

    response = client.get("/api/kinematic-agent/unit_run/state")

    assert response.status_code == 200
    payload = response.get_json()
    expected = {"roll": 0.0, "pitch": 0.0, "yaw": 90.0}
    assert payload["source_orientation_degrees"] == expected
    assert "orientation_degrees" not in payload["latest_preview"]
    assert "orientation_degrees" not in payload["iterations"][0]


def test_kinematic_workbench_run_agent_bootstraps_partseg_from_generated_mesh_dir(tmp_path, monkeypatch):
    calls = []

    class DummyProcess:
        pid = 22334

    def fake_popen(cmd, cwd, env, stdout, stderr, start_new_session):
        calls.append({"cmd": cmd, "env": env})
        return DummyProcess()

    generated_mesh_dir = tmp_path / "generated_mesh" / "ra_036" / "mjcf" / "assets"
    generated_mesh_dir.mkdir(parents=True)
    _write_obj(generated_mesh_dir / "body.obj")
    _write_obj(generated_mesh_dir / "part_05.obj")
    _write_obj(generated_mesh_dir / "part_07.obj")
    converter_root = tmp_path / "kin_036"
    _write_vhacd_placeholder(converter_root / "raw" / "vhacd" / "ra_036" / "body.json")
    _write_vhacd_placeholder(converter_root / "raw" / "vhacd" / "ra_036" / "part_05.json")
    _write_vhacd_placeholder(converter_root / "raw" / "vhacd" / "ra_036" / "part_07.json")
    initial_json = tmp_path / "vlm_initial.json"
    initial_json.write_text(json.dumps({"object_id": "ra_036", "initial_joints": {}}), encoding="utf-8")

    monkeypatch.setattr(subprocess, "Popen", fake_popen)
    app = create_app(
        assets_root=tmp_path / "unused_assets",
        kinematic_workbench_root=tmp_path / "workbench",
    )
    client = app.test_client()

    response = client.post(
        "/api/kinematic-workbench/run-agent",
        json={
                "object_id": "ra_036",
                "converter_output_root": str(converter_root),
                "source_root": str(tmp_path / "source"),
                "initial_joints_json": str(initial_json),
                "generated_mesh_dir": str(generated_mesh_dir),
                "out_dir": str(tmp_path / "run"),
            },
    )

    assert response.status_code == 200
    payload = response.get_json()
    copied_dir = converter_root / "raw/partseg/ra_036/objs"
    assert Path(copied_dir, "body.obj").is_file()
    assert Path(copied_dir, "part_05.obj").is_file()
    assert Path(copied_dir, "part_07.obj").is_file()
    assert payload["partseg_bootstrap"]["copied_count"] == 3
    assert calls


def test_kinematic_workbench_run_agent_cooks_missing_vhacd_before_agent(tmp_path, monkeypatch):
    calls = []

    class DummyProcess:
        pid = 33445

    def fake_run(cmd, cwd, stdout, stderr, text, check):
        calls.append({"kind": "run", "cmd": cmd, "cwd": cwd, "check": check})
        cache_dir = converter_root / "raw" / "vhacd" / "ra_036"
        cache_dir.mkdir(parents=True, exist_ok=True)
        (cache_dir / "body.json").write_text("{}", encoding="utf-8")
        (cache_dir / "part_07.json").write_text("{}", encoding="utf-8")
        return subprocess.CompletedProcess(cmd, 0, stdout="[OK] vhacd cache ra_036\n", stderr="")

    def fake_popen(cmd, cwd, env, stdout, stderr, start_new_session):
        calls.append({"kind": "popen", "cmd": cmd, "env": env})
        return DummyProcess()

    converter_root = tmp_path / "kin_036"
    _write_obj(converter_root / "raw" / "partseg" / "ra_036" / "objs" / "body.obj")
    _write_obj(converter_root / "raw" / "partseg" / "ra_036" / "objs" / "part_07.obj")
    initial_json = tmp_path / "vlm_initial.json"
    initial_json.write_text(json.dumps({"object_id": "ra_036", "initial_joints": {}}), encoding="utf-8")

    monkeypatch.setattr(subprocess, "run", fake_run)
    monkeypatch.setattr(subprocess, "Popen", fake_popen)
    app = create_app(
        assets_root=tmp_path / "unused_assets",
        kinematic_workbench_root=tmp_path / "workbench",
    )
    client = app.test_client()

    response = client.post(
        "/api/kinematic-workbench/run-agent",
        json={
            "object_id": "ra_036",
            "converter_output_root": str(converter_root),
            "source_root": str(tmp_path / "source"),
            "initial_joints_json": str(initial_json),
            "out_dir": str(tmp_path / "run"),
        },
    )

    assert response.status_code == 200
    payload = response.get_json()
    assert [call["kind"] for call in calls] == ["run", "popen"]
    cook_cmd = calls[0]["cmd"]
    assert cook_cmd[:3] == [
        "/home/mi/anaconda3/envs/env-isaacsim/bin/python",
        "-m",
        "post_process.kinematic_solver.utils.data_prep",
    ]
    assert "--stage" in cook_cmd
    assert cook_cmd[cook_cmd.index("--stage") + 1] == "vhacd"
    assert payload["vhacd_cache"]["status"] == "cooked"
    assert payload["vhacd_cache"]["missing_before"] == ["body.json", "part_07.json"]
    assert Path(payload["vhacd_cache"]["log_path"]).is_file()


def test_kinematic_workbench_run_agent_syncs_merged_workbench_asset_to_partseg(tmp_path, monkeypatch):
    calls = []

    class DummyProcess:
        pid = 44556

    def fake_run(cmd, cwd, stdout, stderr, text, check):
        calls.append({"kind": "run", "cmd": cmd})
        cache_dir = converter_root / "raw" / "vhacd" / "ra_036"
        cache_dir.mkdir(parents=True, exist_ok=True)
        (cache_dir / "body.json").write_text("{}", encoding="utf-8")
        (cache_dir / "drawer.json").write_text("{}", encoding="utf-8")
        return subprocess.CompletedProcess(cmd, 0, stdout="[OK] vhacd cache ra_036\n", stderr="")

    def fake_popen(cmd, cwd, env, stdout, stderr, start_new_session):
        calls.append({"kind": "popen", "cmd": cmd})
        return DummyProcess()

    workbench_root = tmp_path / "workbench"
    asset_root = workbench_root / "object_assets" / "ra_036" / "mjcf"
    mesh_dir = asset_root / "assets"
    mesh_dir.mkdir(parents=True)
    _write_offset_obj(mesh_dir / "body.obj", (0.0, 0.0, 0.0))
    _write_offset_obj(mesh_dir / "drawer.obj", (2.0, -1.0, 0.5))
    (asset_root / "ra_036.xml").write_text(
        """<mujoco>
  <asset>
    <mesh name="body_mesh" file="assets/body.obj"/>
    <mesh name="drawer_mesh" file="assets/drawer.obj"/>
  </asset>
  <worldbody>
    <body name="body">
      <geom name="body" type="mesh" mesh="body_mesh" group="2" contype="0" conaffinity="0"/>
      <body name="drawer"><geom name="drawer" type="mesh" mesh="drawer_mesh" group="2" contype="0" conaffinity="0"/></body>
    </body>
  </worldbody>
</mujoco>""",
        encoding="utf-8",
    )
    converter_root = tmp_path / "kin_036"
    stale_partseg = converter_root / "raw" / "partseg" / "ra_036" / "objs"
    _write_obj(stale_partseg / "body.obj")
    _write_obj(stale_partseg / "part_00.obj")
    _write_vhacd_placeholder(converter_root / "raw" / "vhacd" / "ra_036" / "part_00.json")
    initial_json = tmp_path / "vlm_initial.json"
    initial_json.write_text(json.dumps({"object_id": "ra_036", "initial_joints": {}}), encoding="utf-8")

    monkeypatch.setattr(subprocess, "run", fake_run)
    monkeypatch.setattr(subprocess, "Popen", fake_popen)
    app = create_app(
        assets_root=tmp_path / "unused_assets",
        kinematic_workbench_root=workbench_root,
    )
    client = app.test_client()

    response = client.post(
        "/api/kinematic-workbench/run-agent",
        json={
            "object_id": "ra_036",
            "workbench_asset_name": "ra_036",
            "converter_output_root": str(converter_root),
            "source_root": str(tmp_path / "source"),
            "initial_joints_json": str(initial_json),
            "out_dir": str(tmp_path / "run"),
        },
    )

    assert response.status_code == 200
    payload = response.get_json()
    synced_dir = converter_root / "raw" / "partseg" / "ra_036" / "objs"
    assert sorted(path.name for path in synced_dir.glob("*.obj")) == ["body.obj", "drawer.obj"]
    assert _obj_center(synced_dir / "drawer.obj") == _obj_center(mesh_dir / "drawer.obj")
    assert not (converter_root / "raw" / "vhacd" / "ra_036" / "part_00.json").exists()
    assert payload["partseg_bootstrap"]["status"] == "synced_from_workbench_asset"
    assert payload["partseg_bootstrap"]["copied_count"] == 2
    assert [call["kind"] for call in calls] == ["run", "popen"]


def test_kinematic_workbench_run_agent_rejects_missing_partseg_for_object_id(tmp_path, monkeypatch):
    calls = []

    def fake_popen(*args, **kwargs):
        calls.append((args, kwargs))
        raise AssertionError("subprocess should not start when object meshes are missing")

    other_obj_dir = tmp_path / "converter" / "raw" / "partseg" / "ra_036" / "objs"
    _write_obj(other_obj_dir / "body.obj")
    monkeypatch.setattr(subprocess, "Popen", fake_popen)
    app = create_app(
        assets_root=tmp_path / "unused_assets",
        kinematic_workbench_root=tmp_path / "workbench",
    )
    client = app.test_client()

    response = client.post(
        "/api/kinematic-workbench/run-agent",
        json={
            "object_id": "ra_063",
            "converter_output_root": str(tmp_path / "converter"),
            "source_root": str(tmp_path / "source"),
            "out_dir": str(tmp_path / "run"),
        },
    )

    assert response.status_code == 400
    assert "No mesh OBJ files found for object_id=ra_063" in response.get_json()["message"]
    assert "ra_036" in response.get_json()["message"]
    assert calls == []


def test_kinematic_workbench_run_agent_rejects_initial_json_object_mismatch(tmp_path, monkeypatch):
    calls = []

    def fake_popen(*args, **kwargs):
        calls.append((args, kwargs))
        raise AssertionError("subprocess should not start when initial JSON object_id mismatches")

    converter_root = tmp_path / "converter"
    _write_obj(converter_root / "raw" / "partseg" / "ra_036" / "objs" / "body.obj")
    initial_json = tmp_path / "ra_036.json"
    initial_json.write_text(json.dumps({"object_id": "ra_063", "initial_joints": {}}), encoding="utf-8")
    monkeypatch.setattr(subprocess, "Popen", fake_popen)
    app = create_app(
        assets_root=tmp_path / "unused_assets",
        kinematic_workbench_root=tmp_path / "workbench",
    )
    client = app.test_client()

    response = client.post(
        "/api/kinematic-workbench/run-agent",
        json={
            "object_id": "ra_036",
            "converter_output_root": str(converter_root),
            "source_root": str(tmp_path / "source"),
            "initial_joints_json": str(initial_json),
            "out_dir": str(tmp_path / "run"),
        },
    )

    assert response.status_code == 400
    assert "initial JSON object_id='ra_063' does not match Object ID 'ra_036'" in response.get_json()["message"]
    assert calls == []
