import os, subprocess, sys
import json
from pathlib import Path
from types import SimpleNamespace
REPO = Path(__file__).resolve().parents[4]
CLI = REPO/"scripts"/"inference"/"infer_stage.py"
PY = sys.executable

def test_help():
    r = subprocess.run([PY,str(CLI),"--help"], capture_output=True, text=True)
    assert r.returncode==0 and "--stage" in r.stdout

def test_part_gate_missing_ss_latent(tmp_path):
    env = {**os.environ, "INFER_DRY_RUN":"1"}
    r = subprocess.run([PY,str(CLI),"--stage","part","--object-id","o","--root",str(tmp_path),
        "--run-id","r","--mode","A","--view","four","--data-config","/x.yaml",
        "--part-flow-ckpt","/c","--ss-decoder-ckpt","/d"], capture_output=True, text=True, env=env)
    assert r.returncode==2 and "ss_latent.npy" in (r.stderr+r.stdout)

def test_slat_gate_missing_parts(tmp_path):
    env = {**os.environ, "INFER_DRY_RUN":"1"}
    r = subprocess.run([PY,str(CLI),"--stage","slat","--object-id","o","--root",str(tmp_path),
        "--run-id","r","--mode","A","--view","four","--data-config","/x.yaml",
        "--part-flow-ckpt","/c","--ss-decoder-ckpt","/d"], capture_output=True, text=True, env=env)
    assert r.returncode==2 and "part_" in (r.stderr+r.stdout)

def test_stage_launch_refreshes_meta_when_mode_changes(tmp_path):
    env = {**os.environ, "INFER_DRY_RUN":"1"}
    common = [PY, str(CLI), "--stage", "ss", "--object-id", "o", "--root", str(tmp_path),
              "--run-id", "r", "--view", "four", "--data-config", "/x.yaml",
              "--part-flow-ckpt", "/c", "--overwrite"]
    r1 = subprocess.run(common + ["--mode", "B"], capture_output=True, text=True, env=env)
    r2 = subprocess.run(common + ["--mode", "A"], capture_output=True, text=True, env=env)

    assert r1.returncode == 0, r1.stderr + r1.stdout
    assert r2.returncode == 0, r2.stderr + r2.stdout
    meta = json.loads((tmp_path / "o-0" / "r" / "meta.json").read_text())
    assert meta["mode"] == "A"

def test_mode_b_part_reencodes_voxel_and_uses_trellis_defaults(tmp_path, monkeypatch):
    import scripts.inference.infer_stage as infer_stage
    import inference_pipeline.part_flow_stage as part_flow_stage
    import inference_pipeline.ss_encode_stage as ss_encode_stage

    rd = tmp_path/"o"/"r"; rd.mkdir(parents=True)
    (rd/"ss_latent.npy").write_bytes(b"old-latent-placeholder")
    (rd/"voxel.npz").write_bytes(b"voxel-placeholder")
    encodes = []
    runs = []
    monkeypatch.setattr(infer_stage, "_load_dc", lambda a: {})
    monkeypatch.setattr(ss_encode_stage, "run", lambda run_dir, **kwargs: encodes.append((Path(run_dir), kwargs)) or {"latent_shape": (8,16,16,16)})
    monkeypatch.setattr(part_flow_stage, "run", lambda *args, **kwargs: runs.append(kwargs) or ["part_00_voxel.npz"])

    args = SimpleNamespace(
        stage="part",
        object_id="o",
        angle_idx=0,
        mode="B",
        part_flow_ckpt="/fake/part.pt",
        ss_decoder_ckpt="",
        ss_encoder_ckpt="/chosen/ss_enc.safetensors",
        decode_backend="trellis",
    )
    infer_stage._dispatch(args, rd, dry=False, view_mode="four")

    assert encodes == [(rd, {"encoder_ckpt": "/chosen/ss_enc.safetensors"})]
    assert len(runs) == 1
    assert runs[0]["decode_backend"] == "trellis"
    assert runs[0]["ss_decoder_ckpt"] == infer_stage.SS_DECODER_CKPT

def test_mode_b_part_keeps_trellis_ss_flow_latent(tmp_path, monkeypatch):
    import scripts.inference.infer_stage as infer_stage
    import inference_pipeline.part_flow_stage as part_flow_stage
    import inference_pipeline.ss_encode_stage as ss_encode_stage

    rd = tmp_path/"o"/"r"; rd.mkdir(parents=True)
    (rd/"ss_latent.npy").write_bytes(b"trellis-latent-placeholder")
    npz = rd/"voxel.npz"
    import numpy as np
    np.savez_compressed(npz, coords=np.zeros((1, 3), np.int32), source="trellis_ss_flow")
    encodes = []
    runs = []
    monkeypatch.setattr(infer_stage, "_load_dc", lambda a: {})
    monkeypatch.setattr(ss_encode_stage, "run", lambda *args, **kwargs: encodes.append(kwargs))
    monkeypatch.setattr(part_flow_stage, "run", lambda *args, **kwargs: runs.append(kwargs) or ["part_00_voxel.npz"])

    args = SimpleNamespace(
        stage="part",
        object_id="o",
        angle_idx=0,
        mode="B",
        ss_flow_ckpt="/fake/denoiser_step0050000.pt",
        part_flow_ckpt="/fake/part.pt",
        ss_decoder_ckpt="",
        ss_encoder_ckpt="/chosen/ss_enc.safetensors",
        decode_backend="trellis",
    )
    infer_stage._dispatch(args, rd, dry=False, view_mode="four")

    assert encodes == []
    assert len(runs) == 1

def test_part_promptable_seg_backend_dispatches(tmp_path, monkeypatch):
    import numpy as np
    import scripts.inference.infer_stage as infer_stage
    import inference_pipeline.part_prompt_seg_stage as part_prompt_seg_stage
    import inference_pipeline.part_flow_stage as part_flow_stage
    import inference_pipeline.ss_encode_stage as ss_encode_stage

    rd = tmp_path/"o"/"r"; rd.mkdir(parents=True)
    (rd/"ss_latent.npy").write_bytes(b"trellis-latent-placeholder")
    np.savez_compressed(rd/"voxel.npz", coords=np.zeros((1, 3), np.int32), source="trellis_ss_flow")
    seg_runs = []
    monkeypatch.setattr(infer_stage, "_load_dc", lambda a: {"data_root": "/data"})
    monkeypatch.setattr(ss_encode_stage, "run", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("re-encode must not run")))
    monkeypatch.setattr(part_flow_stage, "run", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("part_flow must not run")))
    monkeypatch.setattr(part_prompt_seg_stage, "run", lambda *args, **kwargs: seg_runs.append(kwargs) or ["part_00_voxel.npz"])

    args = SimpleNamespace(
        stage="part",
        object_id="o",
        angle_idx=0,
        mode="B",
        ss_flow_ckpt="/fake/denoiser_step0020000.pt",
        part_backend="promptable_seg",
        part_seg_ckpt="/fake/part_seg/latest.pt",
        part_flow_ckpt="",
        ss_decoder_ckpt="",
        ss_encoder_ckpt="/chosen/ss_enc.safetensors",
        decode_backend="trellis",
    )
    infer_stage._dispatch(args, rd, dry=False, view_mode="four")

    assert len(seg_runs) == 1
    assert seg_runs[0]["part_seg_ckpt"] == "/fake/part_seg/latest.pt"
    assert seg_runs[0]["ss_decoder_ckpt"] == infer_stage.SS_DECODER_CKPT
    assert seg_runs[0]["decode_backend"] == "trellis"

def test_mode_b_part_requires_voxel_for_trellis_reencode(tmp_path, monkeypatch):
    import pytest
    import scripts.inference.infer_stage as infer_stage
    import inference_pipeline.part_flow_stage as part_flow_stage

    rd = tmp_path/"o"/"r"; rd.mkdir(parents=True)
    (rd/"ss_latent.npy").write_bytes(b"sam3d-latent-placeholder")
    monkeypatch.setattr(part_flow_stage, "run", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("part flow must not run")))

    args = SimpleNamespace(
        stage="part",
        object_id="o",
        angle_idx=0,
        mode="B",
        part_flow_ckpt="/fake/part.pt",
        ss_decoder_ckpt="",
        ss_encoder_ckpt="/chosen/ss_enc.safetensors",
        decode_backend="trellis",
    )
    with pytest.raises(SystemExit) as exc:
        infer_stage._dispatch(args, rd, dry=False, view_mode="four")
    assert exc.value.code == 2

def test_sam3d_ss_stage_uses_pipeline_default_when_ss_flow_ckpt_empty(tmp_path, monkeypatch):
    import types
    import scripts.inference.infer_stage as infer_stage
    import inference_pipeline.inputs_materialize as inputs_materialize
    import inference_pipeline.ss_encode_stage as ss_encode_stage

    rd = tmp_path/"o"/"r"; rd.mkdir(parents=True)
    spawns = []
    encodes = []
    object_inputs = types.ModuleType("inference_pipeline.object_inputs")
    object_inputs.load_object_inputs = (
        lambda dc, *, object_id, angle_idx, view_mode: {"view_indices": [0]}
    )
    monkeypatch.setitem(sys.modules, "inference_pipeline.object_inputs", object_inputs)
    monkeypatch.setattr(infer_stage, "_load_dc", lambda a: {})
    monkeypatch.setattr(inputs_materialize, "materialize", lambda *args, **kwargs: {"rgb": "/rgb.png", "mask": "/mask.png"})
    monkeypatch.setattr(infer_stage, "_spawn_sam3d", lambda extra, **kwargs: spawns.append([str(x) for x in extra]))
    monkeypatch.setattr(ss_encode_stage, "run", lambda run_dir, **kwargs: encodes.append((Path(run_dir), kwargs)) or {"latent_shape": (8,16,16,16)})

    args = SimpleNamespace(
        stage="ss",
        object_id="o",
        angle_idx=0,
        mode="B",
        ss_flow_ckpt="",
        ss_encoder_ckpt="/chosen/ss_enc.safetensors",
    )
    infer_stage._dispatch(args, rd, dry=False, view_mode="four")

    assert len(spawns) == 1
    assert "--ss-generator-ckpt" not in spawns[0]
    assert encodes == [(rd, {"encoder_ckpt": "/chosen/ss_enc.safetensors"})]

def test_trellis_ss_stage_uses_ss_flow_ckpt_when_set(tmp_path, monkeypatch):
    import types
    import scripts.inference.infer_stage as infer_stage
    import inference_pipeline.inputs_materialize as inputs_materialize
    import inference_pipeline.ss_encode_stage as ss_encode_stage

    rd = tmp_path/"o"/"r"; rd.mkdir(parents=True)
    calls = []
    object_inputs = types.ModuleType("inference_pipeline.object_inputs")
    object_inputs.load_object_inputs = (
        lambda dc, *, object_id, angle_idx, view_mode: {"view_indices": [0, 1, 2, 3]}
    )
    monkeypatch.setitem(sys.modules, "inference_pipeline.object_inputs", object_inputs)
    monkeypatch.setattr(infer_stage, "_load_dc", lambda a: {"data_root": "/data"})
    monkeypatch.setattr(inputs_materialize, "materialize", lambda *args, **kwargs: {"rgb": "/rgb.png", "mask": "/mask.png"})
    monkeypatch.setattr(infer_stage, "_spawn_sam3d", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("sam3d must not run")))
    monkeypatch.setattr(ss_encode_stage, "run", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("re-encode must not run")))
    monkeypatch.setattr(
        infer_stage,
        "_run_trellis_ss_flow_stage",
        lambda a, run_dir, dc, item, mat: calls.append((a.ss_flow_ckpt, Path(run_dir), item["view_indices"], mat)),
    )

    args = SimpleNamespace(
        stage="ss",
        object_id="o",
        angle_idx=0,
        mode="B",
        ss_flow_ckpt="/fake/denoiser_step0050000.pt",
        ss_decoder_ckpt="/fake/ss_dec.safetensors",
        ss_encoder_ckpt="/chosen/ss_enc.safetensors",
    )
    infer_stage._dispatch(args, rd, dry=False, view_mode="four")

    assert calls == [(
        "/fake/denoiser_step0050000.pt",
        rd,
        [0, 1, 2, 3],
        {"rgb": "/rgb.png", "mask": "/mask.png"},
    )]

def test_sam3d_python_prefers_repo_venv_before_current_interpreter(tmp_path, monkeypatch):
    import scripts.inference.infer_stage as infer_stage

    fake_repo = tmp_path / "repo"
    cu118_py = fake_repo / "submodules/sam3d-stage/submodules/sam-3d-objects/.venv/sam3d-cu118/bin/python"
    cu121_py = fake_repo / "submodules/sam3d-stage/submodules/sam-3d-objects/.venv/sam3d/bin/python"
    cu118_py.parent.mkdir(parents=True)
    cu118_py.write_text("#!/usr/bin/env python\n")
    cu121_py.parent.mkdir(parents=True)
    cu121_py.write_text("#!/usr/bin/env python\n")

    monkeypatch.delenv("SAM3D_VENV_PYTHON", raising=False)
    monkeypatch.setattr(infer_stage, "REPO", fake_repo)

    assert infer_stage._sam3d_python_candidates()[0] == str(cu118_py)

    monkeypatch.setenv("SAM3D_VENV_PYTHON", "/custom/sam3d/python")
    assert infer_stage._sam3d_python_candidates() == ["/custom/sam3d/python"]

def test_slat_whole_scope_spawns_overall_voxel_decode(tmp_path, monkeypatch):
    import numpy as np
    import scripts.inference.infer_stage as infer_stage

    rd = tmp_path / "o" / "r"
    (rd / "parts").mkdir(parents=True)
    (rd / "input_rgb").mkdir()
    (rd / "input_rgb" / "view_0.png").write_bytes(b"png")
    (rd / "input_mask.png").write_bytes(b"png")
    np.savez(rd / "voxel.npz", coords=np.zeros((1, 3), np.int32))
    spawns = []
    monkeypatch.setattr(infer_stage, "_spawn_sam3d", lambda extra, **kwargs: spawns.append([str(x) for x in extra]))

    args = SimpleNamespace(stage="slat", slat_scope="whole")
    infer_stage._dispatch(args, rd, dry=False, view_mode="four")

    assert len(spawns) == 1
    assert "--whole-voxel" in spawns[0]
    assert str(rd / "voxel.npz") in spawns[0]
    assert "--parts-dir" not in spawns[0]
