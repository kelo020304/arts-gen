import json
import threading
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from part_ss_eval_platform.server import create_server


def _write_json(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _make_eval_run(root: Path):
    report = root / "part_ss_latent_flow" / "eval_server" / "full_eval" / "step_000001"
    _write_json(report / "summary.json", {"overall": {"parts": 0, "objects": 0}})
    (report / "part_metrics.jsonl").write_text("", encoding="utf-8")
    (report / "object_metrics.jsonl").write_text("", encoding="utf-8")


def _get_json(base_url: str, path: str):
    with urlopen(base_url + path, timeout=5) as resp:
        return json.loads(resp.read().decode("utf-8"))


def test_server_summary_and_experiments_api(tmp_path):
    _make_eval_run(tmp_path)
    httpd = create_server(host="127.0.0.1", port=0, roots=[tmp_path], output_root=tmp_path)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        base = f"http://127.0.0.1:{httpd.server_address[1]}"
        summary = _get_json(base, "/api/summary")
        experiments = _get_json(base, "/api/experiments")

        assert summary["completed"] == 1
        assert summary["running"] == 0
        assert experiments["experiments"][0]["name"] == "eval_server"

        exp_id = experiments["experiments"][0]["id"]
        detail = _get_json(base, f"/api/experiments/{exp_id}")
        assert detail["experiment"]["id"] == exp_id
        assert detail["metrics"]["task_kind"] == "eval"
    finally:
        httpd.shutdown()
        httpd.server_close()


def test_server_returns_json_error_shape_for_missing_experiment(tmp_path):
    httpd = create_server(host="127.0.0.1", port=0, roots=[tmp_path], output_root=tmp_path)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        base = f"http://127.0.0.1:{httpd.server_address[1]}"
        try:
            _get_json(base, "/api/experiments/missing")
        except HTTPError as exc:
            payload = json.loads(exc.read().decode("utf-8"))
            assert exc.code == 404
            assert payload["error"]["code"] == "not_found"
        else:
            raise AssertionError("expected 404")
    finally:
        httpd.shutdown()
        httpd.server_close()


def test_server_rejects_invalid_job_request(tmp_path):
    httpd = create_server(host="127.0.0.1", port=0, roots=[tmp_path], output_root=tmp_path)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        base = f"http://127.0.0.1:{httpd.server_address[1]}"
        req = Request(
            base + "/api/jobs",
            data=json.dumps({"task_type": "eval", "view_mode": "four"}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            urlopen(req, timeout=5)
        except HTTPError as exc:
            payload = json.loads(exc.read().decode("utf-8"))
            assert exc.code == 400
            assert payload["error"]["code"] == "bad_request"
        else:
            raise AssertionError("expected 400")
    finally:
        httpd.shutdown()
        httpd.server_close()


def _get_bytes(base_url: str, path: str):
    with urlopen(base_url + path, timeout=5) as resp:
        return resp.status, resp.read()


def test_server_artifact_voxel_bin_falls_back_to_npz(tmp_path):
    """viewer 只会读 voxel.bin，但旧 run 可能只落了 voxel.npz。artifact 端点必须从
    同目录 voxel.npz 即时转换出 voxel.bin（200 + 正确字节），而不是 404。"""
    import numpy as np
    from inference_pipeline.voxel_io import save_voxel, voxel_bin_bytes

    run_dir = tmp_path / "102252" / "run_a"
    coords = np.array([[0, 1, 2], [63, 63, 63], [10, 20, 30]], dtype=np.int32)
    save_voxel(run_dir, coords, resolution=64, source="gt")
    (run_dir / "voxel.bin").unlink()  # 模拟旧 run：只剩 voxel.npz
    assert (run_dir / "voxel.npz").is_file() and not (run_dir / "voxel.bin").exists()

    httpd = create_server(host="127.0.0.1", port=0, roots=[tmp_path], output_root=tmp_path)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        base = f"http://127.0.0.1:{httpd.server_address[1]}"
        q = (f"/api/infer/artifact?root={tmp_path}"
             f"&object_id=102252&run_id=run_a&rel=voxel.bin")
        status, body = _get_bytes(base, q)
        assert status == 200
        assert body == voxel_bin_bytes(coords)  # 与磁盘 save_voxel 的 .bin 字节一致
    finally:
        httpd.shutdown()
        httpd.server_close()


def test_server_part_voxels_combines_labeled(tmp_path):
    """/api/infer/part_voxels 合并 parts/part_*_voxel.npz → LE uint16 [x,y,z,label]，
    label 按文件序（每 part 一色）。"""
    import numpy as np

    parts = tmp_path / "102252" / "run_p" / "parts"
    parts.mkdir(parents=True)
    np.savez_compressed(parts / "part_00_voxel.npz",
                        coords=np.array([[1, 2, 3]], np.int32), part_index=np.int32(0))
    np.savez_compressed(parts / "part_01_voxel.npz",
                        coords=np.array([[7, 8, 9], [10, 11, 12]], np.int32), part_index=np.int32(5))

    httpd = create_server(host="127.0.0.1", port=0, roots=[tmp_path], output_root=tmp_path)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        base = f"http://127.0.0.1:{httpd.server_address[1]}"
        status, body = _get_bytes(
            base, f"/api/infer/part_voxels?root={tmp_path}&object_id=102252&run_id=run_p")
        assert status == 200
        arr = np.frombuffer(body, dtype="<u2").reshape(-1, 4)
        # part_00 -> label 0, part_01 -> label 1 (file order, NOT part_index 5)
        assert list(arr[0]) == [1, 2, 3, 0]
        assert list(arr[1]) == [7, 8, 9, 1] and list(arr[2]) == [10, 11, 12, 1]
    finally:
        httpd.shutdown()
        httpd.server_close()


def test_server_rgb_allows_symlinked_render_root(tmp_path, monkeypatch):
    """开发机数据集是软链桥接的：data_root/renders 是指向真实单层目录的软链。
    越界校验必须用 <data_root>/<mask_subdir> 的 resolve 作前缀（与文件同样穿软链），
    否则合法 rgb 被误判越界返回 400。本测试复现该软链布局，断言 /api/infer/rgb 返回 200。"""
    from PIL import Image

    # 真实单层数据 + 一张 rgb
    real_renders = tmp_path / "real" / "renders"
    img_path = real_renders / "102252" / "angle_0" / "rgb" / "view_0.png"
    img_path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (8, 8), (123, 45, 67)).save(img_path)

    # data_root 里 renders 是指向真实目录的软链（resolve 文件会穿软链，resolve data_root 不会）
    data_root = tmp_path / "nest"
    data_root.mkdir(parents=True, exist_ok=True)
    (data_root / "renders").symlink_to(real_renders, target_is_directory=True)

    # 最小 data_config yaml（load_data_config 只要求 data: 段含 data_root）
    cfg_yaml = tmp_path / "dc.yaml"
    cfg_yaml.write_text(
        f"data:\n  data_root: {data_root}\n  mask_subdir: renders\n", encoding="utf-8")
    monkeypatch.setenv("PART_SS_PLATFORM_INFER_DATA_CONFIG", str(cfg_yaml))

    httpd = create_server(host="127.0.0.1", port=0, roots=[tmp_path], output_root=tmp_path)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        base = f"http://127.0.0.1:{httpd.server_address[1]}"
        status, body = _get_bytes(base, "/api/infer/rgb?object_id=102252&angle_idx=0&view=0")
        assert status == 200
        assert body[:8] == b"\x89PNG\r\n\x1a\n"  # 真发回了那张 png
    finally:
        httpd.shutdown()
        httpd.server_close()


def test_server_eval_options_lists_checkpoints(tmp_path):
    """The eval job form needs a checkpoint dropdown, so /api/eval/options must
    return the scanned checkpoints (same source as the inference dropdown)."""
    ckpt = tmp_path / "part_ss_latent_flow" / "run_x" / "ckpts" / "step_5000.pt"
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    ckpt.write_bytes(b"x")
    httpd = create_server(host="127.0.0.1", port=0, roots=[tmp_path], output_root=tmp_path)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        base = f"http://127.0.0.1:{httpd.server_address[1]}"
        options = _get_json(base, "/api/eval/options")
        assert isinstance(options["checkpoints"], list)
        paths = [c["path"] for c in options["checkpoints"]]
        assert any(p.replace("\\", "/").endswith("ckpts/step_5000.pt") for p in paths)
        assert all("path" in c and "label" in c for c in options["checkpoints"])
    finally:
        httpd.shutdown()
        httpd.server_close()

