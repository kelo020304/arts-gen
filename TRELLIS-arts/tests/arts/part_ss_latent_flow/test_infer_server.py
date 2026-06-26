import sys, json, threading, urllib.request, numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[4] / "TRELLIS-arts"))
from part_ss_eval_platform.server import create_server

def _get(port, path):
    with urllib.request.urlopen(f"http://127.0.0.1:{port}{path}") as r: return json.loads(r.read())

def test_infer_roots_runs(tmp_path, monkeypatch):
    base = tmp_path/"inference"; rd = base/"rootA"/"o1"/"r1"/"parts"; rd.mkdir(parents=True)
    np.savez(rd/"part_00_voxel.npz", coords=np.zeros((1,3), np.int32))
    rd.parent.joinpath("meta.json").write_text(json.dumps({"mode":"A","view":"four","object_id":"o1",
        "run_id":"r1","stage_status":{"ss":"done","part":"pending","slat":"pending","assemble":"pending"}}))
    rd_b = base/"rootA"/"o1"/"r2"; rd_b.mkdir(parents=True)
    rd_b.joinpath("meta.json").write_text(json.dumps({"mode":"B","view":"four","object_id":"o1",
        "run_id":"r2","stage_status":{"ss":"done","part":"pending","slat":"pending","assemble":"pending"}}))
    monkeypatch.setenv("PART_SS_PLATFORM_INFER_BASE", str(base))
    httpd = create_server(host="127.0.0.1", port=0, roots=[str(tmp_path)], output_root=str(tmp_path))
    port = httpd.server_address[1]; threading.Thread(target=httpd.serve_forever, daemon=True).start()
    try:
        assert any(r["name"]=="rootA" for r in _get(port,"/api/infer/roots")["roots"])
        runs = _get(port, f"/api/infer/runs?root={base/'rootA'}")["runs"]
        assert runs[0]["object_id"]=="o1"
        latest_a = _get(port, f"/api/infer/latest_run?root={base/'rootA'}&object_id=o1&mode=A&view=four")["run"]
        latest_b = _get(port, f"/api/infer/latest_run?root={base/'rootA'}&object_id=o1&mode=B&view=four")["run"]
        assert latest_a["run_id"] == "r1"
        assert latest_b["run_id"] == "r2"
        outputs = _get(port, f"/api/infer/stage_outputs?root={base/'rootA'}&object_id=o1&run_id=r1")["outputs"]
        assert outputs["ss"]["exists"] is False
        assert outputs["part"]["exists"] is True
    finally: httpd.shutdown()
