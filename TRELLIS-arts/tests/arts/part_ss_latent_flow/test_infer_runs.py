import sys, json, numpy as np, pytest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[4] / "TRELLIS-arts"))
from part_ss_eval_platform import infer_runs

def _mk(base, obj, run, *, mode="A", view="four"):
    rd = base/obj/run; (rd/"parts").mkdir(parents=True)
    rd.joinpath("meta.json").write_text(json.dumps({"mode":mode,"view":view,"object_id":obj,"run_id":run,
        "stage_status":{"ss":"done","part":"pending","slat":"pending","assemble":"pending"}}))
    np.savez(rd/"voxel.npz", coords=np.zeros((1,3),np.int32))
    return rd

def test_list_runs(tmp_path):
    _mk(tmp_path,"o1","r1"); _mk(tmp_path,"o1","r2")
    runs = infer_runs.list_runs(tmp_path)
    assert {r["run_id"] for r in runs}=={"r1","r2"}
    assert next(r for r in runs if r["run_id"]=="r1")["stage_status"]["ss"]=="done"

def test_latest_run_filters_mode_and_view(tmp_path):
    _mk(tmp_path, "o1", "run_1", mode="A", view="four")
    _mk(tmp_path, "o1", "run_2", mode="B", view="four")
    _mk(tmp_path, "o1", "run_3", mode="A", view="single")

    assert infer_runs.latest_run(tmp_path, "o1")["run_id"] == "run_3"
    assert infer_runs.latest_run(tmp_path, "o1", mode="A", view="four")["run_id"] == "run_1"
    assert infer_runs.latest_run(tmp_path, "o1", mode="B", view="four")["run_id"] == "run_2"
    assert infer_runs.latest_run(tmp_path, "o1", mode="B", view="single") is None

def test_manifest(tmp_path):
    _mk(tmp_path,"o1","r1")
    m = infer_runs.read_manifest(tmp_path,"o1","r1")
    assert "voxel.npz" in m["artifacts"] and m["meta"]["mode"]=="A"

def test_artifact_escape(tmp_path):
    _mk(tmp_path,"o1","r1")
    with pytest.raises(ValueError):
        infer_runs.safe_run_artifact(tmp_path,"o1","r1","../../etc/passwd")

def test_artifact_ok(tmp_path):
    _mk(tmp_path,"o1","r1")
    assert infer_runs.safe_run_artifact(tmp_path,"o1","r1","voxel.npz").name=="voxel.npz"

def test_part_voxels_combined_includes_body_context_without_part_overlap(tmp_path):
    rd = _mk(tmp_path, "o1", "r1")
    np.savez(
        rd / "voxel.npz",
        coords=np.array([[0, 0, 0], [1, 1, 1], [2, 2, 2]], np.int32),
    )
    np.savez(
        rd / "parts" / "part_00_voxel.npz",
        coords=np.array([[1, 1, 1]], np.int32),
    )
    np.savez(
        rd / "parts" / "part_01_voxel.npz",
        coords=np.array([[3, 3, 3]], np.int32),
    )

    rows = np.frombuffer(
        infer_runs.part_voxels_combined(tmp_path, "o1", "r1"),
        dtype="<u2",
    ).reshape(-1, 4)

    assert rows.tolist() == [
        [0, 0, 0, infer_runs.BODY_VOXEL_LABEL],
        [2, 2, 2, infer_runs.BODY_VOXEL_LABEL],
        [1, 1, 1, 0],
        [3, 3, 3, 1],
    ]

def test_stage_outputs_detect_existing_artifacts(tmp_path):
    rd = _mk(tmp_path, "o1", "r1")
    (rd / "ss_latent.npy").write_bytes(b"latent")
    np.savez(rd / "parts" / "part_00_voxel.npz", coords=np.zeros((1, 3), np.int32))
    (rd / "parts" / "part_00.glb").write_bytes(b"glb")
    (rd / "parts" / "part_00.ply").write_bytes(b"ply")
    (rd / "complete.glb").write_bytes(b"complete")

    outputs = infer_runs.stage_outputs(tmp_path, "o1", "r1")

    assert outputs["ss"]["exists"] is True
    assert "voxel.npz" in outputs["ss"]["artifacts"]
    assert outputs["part"]["exists"] is True
    assert "parts/part_00_voxel.npz" in outputs["part"]["artifacts"]
    assert outputs["slat"]["exists"] is True
    assert "parts/part_00.glb" in outputs["slat"]["artifacts"]
    assert outputs["assemble"]["exists"] is True
    assert "complete.glb" in outputs["assemble"]["artifacts"]

def test_stage_outputs_detect_overall_slat_artifact(tmp_path):
    rd = _mk(tmp_path, "o1", "r1")
    (rd / "parts" / "overall.glb").write_bytes(b"glb")

    outputs = infer_runs.stage_outputs(tmp_path, "o1", "r1")
    labels = infer_runs.part_labels(tmp_path, "o1", "r1")

    assert outputs["slat"]["exists"] is True
    assert outputs["slat"]["artifacts"] == ["parts/overall.glb"]
    assert labels["components"][0]["stem"] == "overall"
