import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[4] / "TRELLIS-arts"))
from part_ss_eval_platform.infer_jobs import InferJobRequest, InferStageExistsError, build_infer_command

def test_build_cmd(tmp_path):
    req = InferJobRequest(stage="part", object_id="o", root=str(tmp_path), run_id="r1",
        mode="B", view="four", data_config="/c.yaml", part_flow_ckpt="/p", gpu_ids="2")
    cmd = build_infer_command(req, repo_root=Path("/repo"))
    # single unified venv -> infer_stage launched with the platform's own interpreter
    assert cmd.args[:4]==[sys.executable,"scripts/inference/infer_stage.py","--stage","part"]
    assert "--view" in cmd.args and "four" in cmd.args
    assert "--decode-backend" in cmd.args and "trellis" in cmd.args
    assert "--ss-decoder-ckpt" not in cmd.args
    assert "--ss-flow-ckpt" not in cmd.args
    assert cmd.env["CUDA_VISIBLE_DEVICES"]=="2"
    assert "--device" not in cmd.args            # GPU 只经 env，不双重索引
    assert cmd.run_dir==str(tmp_path/"o-0"/"r1") and cmd.log_path.endswith("/part.log")

def test_build_cmd_forwards_ss_flow_ckpt(tmp_path):
    req = InferJobRequest(stage="ss", object_id="o", root=str(tmp_path), run_id="r1",
        mode="B", view="four", data_config="/c.yaml", ss_flow_ckpt="/ss/denoiser.pt",
        gpu_ids="3")
    cmd = build_infer_command(req, repo_root=Path("/repo"))
    assert "--ss-flow-ckpt" in cmd.args
    assert cmd.args[cmd.args.index("--ss-flow-ckpt") + 1] == "/ss/denoiser.pt"
    assert cmd.env["CUDA_VISIBLE_DEVICES"]=="3"

def test_build_cmd_forwards_slat_scope(tmp_path):
    req = InferJobRequest(stage="slat", object_id="o", root=str(tmp_path), run_id="r1",
        mode="B", view="four", data_config="/c.yaml", slat_scope="whole")
    cmd = build_infer_command(req, repo_root=Path("/repo"))
    assert "--slat-scope" in cmd.args
    assert cmd.args[cmd.args.index("--slat-scope") + 1] == "whole"

def test_build_cmd_forwards_promptable_seg_backend(tmp_path):
    req = InferJobRequest(stage="part", object_id="o", root=str(tmp_path), run_id="r1",
        mode="B", view="four", data_config="/c.yaml", part_backend="promptable_seg",
        part_seg_ckpt="/seg/latest.pt", ss_flow_ckpt="/ss/denoiser.pt")
    cmd = build_infer_command(req, repo_root=Path("/repo"))
    assert "--part-backend" in cmd.args
    assert cmd.args[cmd.args.index("--part-backend") + 1] == "promptable_seg"
    assert "--part-seg-ckpt" in cmd.args
    assert cmd.args[cmd.args.index("--part-seg-ckpt") + 1] == "/seg/latest.pt"
    assert "--part-flow-ckpt" not in cmd.args

def test_part_requires_part_flow_ckpt_only(tmp_path):
    import pytest
    with pytest.raises(ValueError):
        build_infer_command(InferJobRequest(stage="part", object_id="o", root=str(tmp_path),
            run_id="r", mode="A", view="four", data_config="/c.yaml"), repo_root=Path("/repo"))

    cmd = build_infer_command(InferJobRequest(stage="part", object_id="o", root=str(tmp_path),
        run_id="r", mode="B", view="four", data_config="/c.yaml", part_flow_ckpt="/p"), repo_root=Path("/repo"))
    assert "--part-flow-ckpt" in cmd.args
    assert "--ss-decoder-ckpt" not in cmd.args

def test_part_promptable_requires_part_seg_ckpt(tmp_path):
    import pytest
    with pytest.raises(ValueError):
        build_infer_command(InferJobRequest(stage="part", object_id="o", root=str(tmp_path),
            run_id="r", mode="A", view="four", data_config="/c.yaml",
            part_backend="promptable_seg"), repo_root=Path("/repo"))

def test_existing_stage_requires_overwrite(tmp_path):
    import pytest
    parts = tmp_path / "o" / "r" / "parts"
    parts.mkdir(parents=True)
    (parts / "part_00_voxel.npz").write_bytes(b"old")
    req = InferJobRequest(stage="part", object_id="o", root=str(tmp_path), run_id="r",
        mode="A", view="four", data_config="/c.yaml", part_flow_ckpt="/p")

    with pytest.raises(InferStageExistsError):
        build_infer_command(req, repo_root=Path("/repo"))

    cmd = build_infer_command(
        InferJobRequest(**{**req.__dict__, "overwrite": True}),
        repo_root=Path("/repo"),
    )
    assert "--stage" in cmd.args and "part" in cmd.args
