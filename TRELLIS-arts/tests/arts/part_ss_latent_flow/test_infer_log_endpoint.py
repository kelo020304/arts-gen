import sys, json, threading, urllib.request, urllib.error
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[4] / "TRELLIS-arts"))
from part_ss_eval_platform.server import create_server


def _get(port, path):
    with urllib.request.urlopen(f"http://127.0.0.1:{port}{path}") as r:
        return r.status, json.loads(r.read())


def _status(port, path):
    """Return HTTP status for a request, tolerating 4xx error responses."""
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}{path}") as r:
            return r.status
    except urllib.error.HTTPError as exc:
        return exc.code


def test_infer_log_endpoint(tmp_path, monkeypatch):
    # Layout: <infer_base>/rootA/o1/r1/part.log  (a run with a stage log).
    base = tmp_path / "inference"
    root = base / "rootA"
    run_dir = root / "o1" / "r1"
    run_dir.mkdir(parents=True)
    run_dir.joinpath("part.log").write_text("part stage boom\ntraceback here\n")
    monkeypatch.setenv("PART_SS_PLATFORM_INFER_BASE", str(base))

    httpd = create_server(host="127.0.0.1", port=0, roots=[str(tmp_path)], output_root=str(tmp_path))
    port = httpd.server_address[1]
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    try:
        # existing part.log -> returns its text
        _, body = _get(port, f"/api/infer/log?root={root}&object_id=o1&run_id=r1&stage=part")
        assert body["stage"] == "part"
        assert "part stage boom" in body["log"]

        # missing log (slat.log absent) -> tolerant empty string
        _, body = _get(port, f"/api/infer/log?root={root}&object_id=o1&run_id=r1&stage=slat")
        assert body == {"stage": "slat", "log": ""}

        # bad stage -> 400
        assert _status(port, f"/api/infer/log?root={root}&object_id=o1&run_id=r1&stage=bogus") == 400

        # path escape (run_id climbs above root via ..) -> 400
        assert _status(port, f"/api/infer/log?root={root}&object_id=o1&run_id=../..&stage=part") == 400
    finally:
        httpd.shutdown()
