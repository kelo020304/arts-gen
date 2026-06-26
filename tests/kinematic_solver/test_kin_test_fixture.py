import json
from pathlib import Path

from post_process.object_post_process.mjcf_parser import generate_manifest


def test_ra063_kin_test_fixture_is_web_loadable():
    root = Path("kin_test/ra_063")
    initial_json = root / "vlm_initial.json"
    state_json = root / "frontend_state.json"

    assert initial_json.is_file()
    initial = json.loads(initial_json.read_text())
    assert initial["object_id"] == "ra_063"
    assert initial["initial_joints"]["part_02"]["limit"] == [0, 30]

    assert state_json.is_file()
    state = json.loads(state_json.read_text())
    latest = state["latest_preview"]
    asset_name = latest["asset_name"]
    asset_dirs = [
        path
        for path in (root / "object_assets").iterdir()
        if path.is_dir()
    ]
    assert [path.name for path in asset_dirs] == [asset_name]
    assert state["iterations"] == [latest]
    asset_dir = root / "object_assets" / asset_name
    assert (asset_dir / "mjcf" / f"{asset_name}.xml").is_file()
    assert (asset_dir / "mjcf" / "assets" / "body.obj").is_file()
    assert (asset_dir / "mjcf" / "assets" / "part_00.obj").is_file()
    assert (asset_dir / "mjcf" / "assets" / "part_01.obj").is_file()
    assert (asset_dir / "mjcf" / "assets" / "part_02.obj").is_file()

    manifest = generate_manifest(asset_name, root / "object_assets")
    assert manifest["status"] == "ok"
    assert {joint["name"] for joint in manifest["joints"]} == {"part_00", "part_01", "part_02"}
