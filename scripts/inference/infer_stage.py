#!/usr/bin/env python
from __future__ import annotations
import argparse, os, random, subprocess, sys, types
from pathlib import Path
import numpy as np
REPO = Path(__file__).resolve().parents[2]
TRELLIS_PATH = REPO / "TRELLIS-arts"
sys.path.insert(0, str(TRELLIS_PATH))

def _setup_trellis_imports() -> None:
    """Register trellis packages without executing trellis/__init__.py.

    The full TRELLIS package imports demo pipelines, which require rembg. The
    platform only needs datasets/models/samplers here, so mirror train/eval's
    lightweight package shell to keep inference jobs independent from rembg.
    """
    pkg = types.ModuleType("trellis")
    pkg.__path__ = [str(TRELLIS_PATH / "trellis")]
    pkg.__package__ = "trellis"
    sys.modules.setdefault("trellis", pkg)
    for sp in ("models", "modules", "trainers", "utils", "datasets", "pipelines", "renderers"):
        mod = types.ModuleType(f"trellis.{sp}")
        mod.__path__ = [str(TRELLIS_PATH / "trellis" / sp)]
        mod.__package__ = f"trellis.{sp}"
        sys.modules.setdefault(f"trellis.{sp}", mod)

_setup_trellis_imports()
os.environ.setdefault("TORCH_HOME", str(REPO / "submodules" / "TRELLIS.1"))
os.environ.setdefault("ATTN_BACKEND", "sdpa")
from inference_pipeline import transform_io  # noqa: E402

# The full sam3d image->surface/SLat glue needs sam3d_objects + pytorch3d.
# The standalone SS decoder glue only needs the sparse-structure VAE modules.
# Override SAM3D_VENV_PYTHON to force a specific cloud/dev interpreter.
def _sam3d_python_candidates() -> list[str]:
    override = os.environ.get("SAM3D_VENV_PYTHON")
    if override:
        return [override]
    base = REPO / "submodules/sam3d-stage/submodules/sam-3d-objects/.venv"
    candidates: list[str] = []
    for rel in ("sam3d-cu118/bin/python", "sam3d/bin/python"):
        py = base / rel
        if py.is_file():
            candidates.append(str(py))
    candidates.append(sys.executable)
    seen: set[str] = set()
    return [p for p in candidates if not (p in seen or seen.add(p))]

SAM3D_PIPELINE = os.environ.get("SAM3D_PIPELINE_YAML", "/robot/data-lab/jzh/art-gen/weights/pipeline.yaml")
# mode B：sam3d 出 surface voxel 后，用这个 TRELLIS SS encoder 把 voxel 重编码成 z_global
# （TRELLIS 空间，匹配 0526 等 TRELLIS part flow ckpt；sam3d 自己的 ss_latent 是 sam3d 空间）。
SS_ENCODER_CKPT = os.environ.get(
    "SS_ENCODER_CKPT", "pretrained/TRELLIS-image-large/ckpts/ss_enc_conv3d_16l8_fp16.safetensors")
SS_DECODER_CKPT = os.environ.get(
    "SS_DECODER_CKPT", "pretrained/TRELLIS-image-large/ckpts/ss_dec_conv3d_16l8_fp16.safetensors")
SS_GLUE = REPO/"submodules/sam3d-stage/infer_glue/ss_stage.py"
SLAT_GLUE = REPO/"submodules/sam3d-stage/infer_glue/slat_stage.py"
DECODE_GLUE = REPO/"submodules/sam3d-stage/infer_glue/decode_ss_glue.py"

def _die(msg, code=2): print(f"[infer_stage][ERROR] {msg}", file=sys.stderr); sys.exit(code)

def _seed_all(seed: int | None) -> None:
    if seed is None:
        return
    seed_i = int(seed)
    random.seed(seed_i)
    np.random.seed(seed_i % (2**32))
    try:
        import torch

        torch.manual_seed(seed_i)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed_i)
    except Exception:
        pass

def _run_dir(a):
    root = Path(a.root)
    object_id = str(a.object_id)
    run_id = str(a.run_id)
    angle_container = f"{object_id}-{int(a.angle_idx)}"
    candidates = [
        root / angle_container / run_id,
        root / angle_container / object_id / run_id,
        root / angle_container / angle_container / run_id,
        root / object_id / run_id,
        root / object_id / object_id / run_id,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    if root.is_dir():
        meta_paths = list(root.glob("*/*/meta.json")) + list(root.glob("*/*/*/meta.json"))
        for meta_path in meta_paths:
            try:
                meta = __import__("json").loads(meta_path.read_text())
            except (OSError, ValueError):
                continue
            if (
                str(meta.get("object_id")) == object_id
                and str(meta.get("run_id")) == run_id
                and int(meta.get("angle_idx", -1)) == int(a.angle_idx)
            ):
                return meta_path.parent
    return candidates[0]

def _unlink_if_exists(path: Path) -> None:
    if path.is_file() or path.is_symlink():
        path.unlink()

def _cleanup_stage_outputs(run_dir: Path, stage: str) -> None:
    """Remove only the artifacts owned by one stage before an explicit overwrite."""
    run_dir = Path(run_dir)
    parts = run_dir / "parts"
    if stage == "ss":
        for name in ("ss_latent.npy", "voxel.npz", "voxel.bin", "pose.json", "transform.json"):
            _unlink_if_exists(run_dir / name)
        if (run_dir / "input_rgb").is_dir():
            for child in (run_dir / "input_rgb").glob("*"):
                _unlink_if_exists(child)
        _unlink_if_exists(run_dir / "input_mask.png")
    elif stage == "part" and parts.is_dir():
        for pattern in ("part_*_latent.npy", "part_*_meta.json", "part_*_voxel.npz"):
            for path in parts.glob(pattern):
                _unlink_if_exists(path)
        _unlink_if_exists(parts / "joint_partition.npz")
    elif stage == "slat" and parts.is_dir():
        for pattern in ("overall.glb", "overall.ply", "body.glb", "body.ply", "body_voxel.npz", "part_*.glb", "part_*.ply"):
            for path in parts.glob(pattern):
                _unlink_if_exists(path)
    elif stage == "assemble":
        for name in ("complete.glb", "complete.ply"):
            _unlink_if_exists(run_dir / name)

def main():
    p = argparse.ArgumentParser(prog="infer_stage")
    p.add_argument("--stage", required=True, choices=["ss","part","slat","assemble"])
    p.add_argument("--object-id", required=True); p.add_argument("--root", required=True)
    p.add_argument("--run-id", required=True)
    p.add_argument("--mode", required=True, choices=["A","B"])
    p.add_argument("--view", required=True, choices=["single","four"])
    p.add_argument("--angle-idx", type=int, default=0)
    p.add_argument("--data-config", required=True); p.add_argument("--data-root", default="")
    p.add_argument("--part-flow-ckpt", default=""); p.add_argument("--part-seg-ckpt", default="")
    p.add_argument("--ss-decoder-ckpt", default=SS_DECODER_CKPT)
    p.add_argument("--ss-encoder-ckpt", default=SS_ENCODER_CKPT)
    p.add_argument("--ss-flow-ckpt", default=""); p.add_argument("--gpu", default="0")
    p.add_argument("--seed", type=int, default=None,
                   help="Optional deterministic seed for SS flow and promptable part segmentation setup.")
    p.add_argument("--part-backend", default="part_flow", choices=["part_flow", "promptable_seg"],
                   help="part 阶段后端：part_flow=扩散式 part latent flow；promptable_seg=part-promptable segmentation")
    p.add_argument("--decode-backend", default="trellis", choices=["sam3d", "trellis"],
                   help="part latent 解码后端：trellis=TRELLIS ss_dec_conv3d（TRELLIS SS 空间的 ckpt，如 0526）；sam3d=sam3d ss_decoder")
    p.add_argument("--part-joint-candidate-mode", default="proposal", choices=["proposal", "full_occ"])
    p.add_argument("--part-joint-refine", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--part-joint-refine-iters", type=int, default=1)
    p.add_argument("--part-joint-refine-pairwise", type=float, default=3.0)
    p.add_argument("--part-joint-refine-margin", type=float, default=0.0)
    p.add_argument("--part-joint-refine-margin-quantile", type=float, default=0.01)
    p.add_argument("--part-joint-refine-neighborhood", type=int, choices=[6, 18, 26], default=6)
    p.add_argument("--part-joint-refine-min-vote-gain", type=float, default=0.0)
    p.add_argument("--part-joint-refine-preserve-small-classes", type=int, default=32)
    p.add_argument("--part-joint-save-logits", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--slat-scope", default="parts", choices=["parts", "whole", "both"],
                   help="slat 阶段解码范围：parts=现有 body+parts；whole=整体 voxel+图；both=两者都写出")
    p.add_argument("--overwrite", action="store_true",
                   help="确认覆盖当前 run 下该 stage 的既有产物；只清理本 stage 输出")
    a = p.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = a.gpu       # glue 恒用 cuda:0
    _seed_all(a.seed)
    rd = _run_dir(a); rd.mkdir(parents=True, exist_ok=True)
    dry = os.environ.get("INFER_DRY_RUN") == "1"
    view_mode = a.view
    # Refresh meta on every stage launch. write_meta preserves stage_status for
    # the same run contract, and resets it when mode/view/object/angle changes.
    transform_io.write_meta(rd, mode=a.mode, view=a.view, object_id=a.object_id, run_id=a.run_id,
        ckpts={"part_flow": a.part_flow_ckpt, "ss_decoder": a.ss_decoder_ckpt,
               "ss_encoder": a.ss_encoder_ckpt, "ss_flow": a.ss_flow_ckpt,
               "part_seg": a.part_seg_ckpt},
        part_backend=a.part_backend,
        angle_idx=a.angle_idx)
    if a.overwrite:
        _cleanup_stage_outputs(rd, a.stage)
    transform_io.set_stage_status(rd, a.stage, "running")
    try:
        _dispatch(a, rd, dry, view_mode)
    except SystemExit:
        transform_io.set_stage_status(rd, a.stage, "failed"); raise
    except Exception as e:
        transform_io.set_stage_status(rd, a.stage, "failed"); _die(str(e), 1)
    transform_io.set_stage_status(rd, a.stage, "done")
    print(f"[infer_stage] stage={a.stage} done -> {rd}")

def _load_dc(a):
    from inference_pipeline.data_config_io import load_data_config
    return load_data_config(a.data_config, data_root_override=(a.data_root or None))

def _as_numpy(x):
    return x.detach().cpu().numpy() if hasattr(x, "detach") else np.asarray(x)

def _load_object_sample_meta(data_config: dict, *, object_id: str, angle_idx: int, view_mode: str) -> dict:
    from inference_pipeline.object_inputs import _dataset_for

    ds = _dataset_for(view_mode, data_config)
    for sample in ds.samples:
        if str(sample.get("obj_id", sample.get("object_id"))) == str(object_id) and int(sample["angle_idx"]) == int(angle_idx):
            return sample
    raise KeyError(f"manifest 中无 object_id={object_id} angle_idx={angle_idx}")

def _is_trellis_ss_flow_run(run_dir: Path, a) -> bool:
    if getattr(a, "ss_flow_ckpt", ""):
        return True
    voxel_path = Path(run_dir) / "voxel.npz"
    if not voxel_path.is_file():
        return False
    try:
        with np.load(voxel_path) as data:
            source = str(data.get("source", ""))
    except Exception:
        return False
    return source == "trellis_ss_flow"

def _run_trellis_ss_flow_stage(a, rd: Path, dc: dict, item: dict, mat: dict) -> dict:
    from inference import decode_ss, run_ss_flow_from_tokens
    from inference_pipeline import inputs_materialize
    from inference_pipeline.voxel_io import save_voxel

    view_indices = [int(v) for v in item.get("view_indices", [])]
    if not view_indices:
        raise ValueError("TRELLIS SS flow 需要至少一个 view_index")
    if len(view_indices) != 4:
        raise ValueError(f"TRELLIS SS multiflow 需要 4 个 view_index，got {view_indices}")
    # Materialize RGB/mask for downstream promptable part segmentation, but feed
    # SS-flow the same official prenorm DINO tokens used by training/eval.
    recon_root = Path(dc["data_root"]) / dc.get("recon_subdir", "reconstruction")
    token_candidates = [
        recon_root / "dinov2_tokens_official_prenorm1374" / str(a.object_id) / f"angle_{int(a.angle_idx)}" / "tokens.npz",
        recon_root / "dinov2_tokens_prenorm" / str(a.object_id) / f"angle_{int(a.angle_idx)}" / "tokens.npz",
        recon_root / "dinov2_tokens" / str(a.object_id) / f"angle_{int(a.angle_idx)}" / "tokens.npz",
    ]
    token_path = None
    all_tokens = None
    bad_tokens = []
    for candidate in token_candidates:
        if not candidate.is_file():
            continue
        with np.load(candidate, allow_pickle=False) as data:
            if "tokens" not in data.files:
                bad_tokens.append(f"{candidate}: keys={data.files}")
                continue
            tokens = np.asarray(data["tokens"], dtype=np.float32)
        if tokens.ndim == 3 and tokens.shape[1:] == (1374, 1024):
            token_path = candidate
            all_tokens = tokens
            break
        bad_tokens.append(f"{candidate}: shape={tokens.shape}")
    if token_path is None or all_tokens is None:
        detail = "; ".join(bad_tokens) if bad_tokens else "no candidate token files exist"
        raise FileNotFoundError(
            "TRELLIS SS-flow official DINO tokens [V,1374,1024] not found; "
            f"checked {token_candidates}; {detail}"
        )
    if min(view_indices) < 0 or max(view_indices) >= all_tokens.shape[0]:
        raise ValueError(f"{token_path} cannot select views {view_indices} from shape {all_tokens.shape}")
    cond_tokens = np.ascontiguousarray(all_tokens[view_indices])
    ss_decoder_ckpt = a.ss_decoder_ckpt or SS_DECODER_CKPT
    ss_fusion_mode = str(
        os.environ.get("SS_FLOW_FUSION_MODE")
        or ("concat" if "tre-ss-concat" in str(a.ss_flow_ckpt) else "multidiffusion")
    )
    ss_kwargs = {}
    if getattr(a, "seed", None) is not None:
        ss_kwargs["seed"] = int(a.seed)
    z_global = run_ss_flow_from_tokens(cond_tokens, a.ss_flow_ckpt, fusion_mode=ss_fusion_mode, **ss_kwargs)
    z_np = _as_numpy(z_global).astype(np.float32)
    if z_np.shape != (8, 16, 16, 16):
        raise ValueError(f"TRELLIS SS flow latent 形状异常 {z_np.shape}（期望 (8,16,16,16)）")
    np.save(rd / "ss_latent.npy", np.ascontiguousarray(z_np, np.float32))
    coords = _as_numpy(decode_ss(z_global, ss_decoder_ckpt, threshold=0.0)).astype(np.int32)
    save_voxel(rd, coords, resolution=64, source="trellis_ss_flow")
    print(
        "[infer_stage] ss(mode B): TRELLIS SS flow "
        f"fusion={ss_fusion_mode} seed={getattr(a, 'seed', None)} "
        f"views={view_indices} tokens={token_path} ckpt={a.ss_flow_ckpt} "
        f"-> latent={rd/'ss_latent.npy'} "
        f"voxel={rd/'voxel.npz'}"
    )
    return {
        "fusion_mode": ss_fusion_mode,
        "views": view_indices,
        "tokens": str(token_path),
        "tokens_shape": list(cond_tokens.shape),
        "cond_shape": [1, int(cond_tokens.shape[0] * cond_tokens.shape[1]), int(cond_tokens.shape[2])]
        if ss_fusion_mode == "concat"
        else list(cond_tokens.shape),
        "rgb": mat.get("rgb", ""),
        "mask": mat.get("mask", ""),
        "latent_shape": tuple(z_np.shape),
        "num_voxels": int(coords.shape[0]),
    }

def _dispatch(a, rd, dry, view_mode):
    if a.stage == "ss":
        if dry: print("[dry] ss"); return
        dc = _load_dc(a)
        from inference_pipeline import inputs_materialize
        item = _load_object_sample_meta(dc, object_id=a.object_id, angle_idx=a.angle_idx, view_mode=view_mode)
        mat = inputs_materialize.materialize(rd, dc, object_id=a.object_id, angle_idx=a.angle_idx,
                                             view_indices=item["view_indices"])
        if a.mode == "A":
            from inference_pipeline import ss_stage_local
            ss_stage_local.run_mode_a(dc, object_id=a.object_id, angle_idx=a.angle_idx,
                                      view_mode=view_mode, out_dir=rd,
                                      ss_decoder_ckpt=(a.ss_decoder_ckpt or SS_DECODER_CKPT))
        else:
            if a.ss_flow_ckpt:
                _run_trellis_ss_flow_stage(a, rd, dc, item, mat)
            else:
                # mode B：sam3d 出 surface voxel（voxel.npz）+ 它自己的 ss_latent（sam3d 空间）。
                ss_encoder_ckpt = a.ss_encoder_ckpt or SS_ENCODER_CKPT
                ss_args = [str(SS_GLUE), "--image", mat["rgb"], "--mask", mat["mask"],
                           "--config", SAM3D_PIPELINE,
                           "--out", str(rd), "--device", "cuda:0"]
                _spawn_sam3d(ss_args, runtime="full")
                # part flow（TRELLIS）吃的是 TRELLIS 空间的 z_global → 用 TRELLIS SS encoder
                # 把 sam3d 的 voxel 重编码、覆盖 ss_latent.npy（图→sam3d voxel→TRELLIS encode）。
                from inference_pipeline import ss_encode_stage
                info = ss_encode_stage.run(rd, encoder_ckpt=ss_encoder_ckpt)
                print(f"[infer_stage] ss(mode B): TRELLIS re-encode voxel -> z_global {info}")
    elif a.stage == "part":
        if not (rd/"ss_latent.npy").is_file():
            _die(f"part 阶段缺 ss_latent.npy（先跑 ss）：{rd/'ss_latent.npy'}", 2)
        if dry: print("[dry] part"); return
        if getattr(a, "part_backend", "part_flow") == "promptable_seg":
            if not getattr(a, "part_seg_ckpt", ""):
                _die("promptable_seg part 阶段缺 --part-seg-ckpt", 2)
        elif not getattr(a, "part_flow_ckpt", ""):
            _die("part_flow part 阶段缺 --part-flow-ckpt", 2)
        if getattr(a, "mode", "") == "B":
            if not (rd/"voxel.npz").is_file():
                _die(f"mode B part 阶段缺 voxel.npz，无法 TRELLIS re-encode：{rd/'voxel.npz'}", 2)
            if _is_trellis_ss_flow_run(rd, a):
                print("[infer_stage] part(mode B): using TRELLIS SS flow z_global; skip re-encode")
            else:
                # 防止旧 run 里残留 sam3d 空间的 ss_latent.npy：part flow 0526 吃 TRELLIS z_global。
                from inference_pipeline import ss_encode_stage
                info = ss_encode_stage.run(rd, encoder_ckpt=(a.ss_encoder_ckpt or SS_ENCODER_CKPT))
                print(f"[infer_stage] part(mode B): refreshed TRELLIS z_global from voxel {info}")
        if getattr(a, "part_backend", "part_flow") == "promptable_seg":
            from inference_pipeline import part_prompt_seg_stage
            # part-promptable seg 逐 target part 读取 2D part mask prompt，写出
            # parts/part_NN_voxel.npz，后续 slat/assemble 复用同一产物契约。
            part_prompt_seg_stage.run(
                rd,
                _load_dc(a),
                object_id=a.object_id,
                angle_idx=a.angle_idx,
                view_mode=view_mode,
                part_seg_ckpt=a.part_seg_ckpt,
                ss_decoder_ckpt=(a.ss_decoder_ckpt or SS_DECODER_CKPT),
                decode_backend=a.decode_backend,
                joint_candidate_mode=a.part_joint_candidate_mode,
                joint_refine=bool(a.part_joint_refine),
                joint_refine_iters=int(a.part_joint_refine_iters),
                joint_refine_pairwise=float(a.part_joint_refine_pairwise),
                joint_refine_margin=float(a.part_joint_refine_margin),
                joint_refine_margin_quantile=float(a.part_joint_refine_margin_quantile),
                joint_refine_neighborhood=int(a.part_joint_refine_neighborhood),
                joint_refine_min_vote_gain=float(a.part_joint_refine_min_vote_gain),
                joint_refine_preserve_small_classes=int(a.part_joint_refine_preserve_small_classes),
                joint_save_logits=bool(a.part_joint_save_logits),
            )
        else:
            from inference_pipeline import part_flow_stage
            # part flow (arts-gen/trellis) 预测 per-part latent → parts/part_NN_latent.npy (+ meta)。
            part_flow_stage.run(rd, _load_dc(a), object_id=a.object_id, angle_idx=a.angle_idx,
                                view_mode=view_mode, part_flow_ckpt=a.part_flow_ckpt,
                                ss_decoder_ckpt=(a.ss_decoder_ckpt or SS_DECODER_CKPT),
                                decode_backend=a.decode_backend)
        # sam3d 后端：latent 在 sam3d SS 空间 → 单独子进程跑 sam3d ss_decoder（VRAM 隔离）。
        # trellis 后端：part_flow_stage 已用 TRELLIS ss_dec_conv3d 解码并写出 part_NN_voxel.npz。
        if a.decode_backend == "sam3d" and getattr(a, "part_backend", "part_flow") == "part_flow":
            decode_args = [str(DECODE_GLUE), "--parts-dir", str(rd/"parts"),
                           "--config", SAM3D_PIPELINE, "--device", "cuda:0"]
            # The CLI default is the TRELLIS SS decoder because decode_backend
            # defaults to trellis. For sam3d decoding, leave the ckpt unspecified
            # so decode_ss_glue uses pipeline.yaml's ss_decoder.ckpt unless the
            # caller explicitly passes a different SAM3D decoder.
            if a.ss_decoder_ckpt and a.ss_decoder_ckpt != SS_DECODER_CKPT:
                decode_args += ["--ss-decoder-ckpt", a.ss_decoder_ckpt]
            _spawn_sam3d(decode_args, runtime="ss_decoder")
    elif a.stage == "slat":
        parts = rd/"parts"
        needs_parts = a.slat_scope in {"parts", "both"}
        needs_whole = a.slat_scope in {"whole", "both"}
        if needs_parts and not list(parts.glob("part_*_voxel.npz")):
            _die(f"slat 阶段缺 parts/part_*_voxel.npz（先跑 part）：{parts}", 2)
        if needs_whole and not (rd/"voxel.npz").is_file():
            _die(f"slat 阶段缺整体 voxel.npz（先跑 ss）：{rd/'voxel.npz'}", 2)
        rgbs = sorted((rd/"input_rgb").glob("view_*.png")) if (rd/"input_rgb").is_dir() else []
        if not rgbs or not (rd/"input_mask.png").is_file():
            _die(f"slat 阶段缺 input_rgb/ 或 input_mask.png（先跑 ss）：{rd}", 2)
        if dry: print("[dry] slat -> sam3d"); return
        base_args = [str(SLAT_GLUE), "--image", str(rgbs[0]),
                     "--mask", str(rd/"input_mask.png"), "--config", SAM3D_PIPELINE,
                     "--formats", "gaussian", "mesh", "--device", "cuda:0"]
        if needs_whole:
            _spawn_sam3d([*base_args, "--whole-voxel", str(rd/"voxel.npz"),
                          "--whole-stem", "overall", "--out", str(parts)],
                         runtime="full")
        if needs_parts:
            _spawn_sam3d([*base_args, "--parts-dir", str(parts), "--out", str(parts)],
                         runtime="full")
    elif a.stage == "assemble":
        parts = rd/"parts"
        glbs = sorted(p.name for p in parts.glob("part_*.glb")) if parts.is_dir() else []
        if (parts/"body.glb").is_file():
            glbs = ["body.glb", *glbs]
        elif not glbs and (parts/"overall.glb").is_file():
            glbs = ["overall.glb"]
        if not glbs:
            _die(f"assemble 阶段缺逐 part glb（先跑 slat）：{parts}", 2)
        if dry: print("[dry] assemble"); return
        from inference_pipeline.assemble import assemble_complete
        plys = [n[:-4]+".ply" for n in glbs if (parts/(n[:-4]+".ply")).is_file()]
        assemble_complete(rd, part_mesh_names=glbs,
                          part_gaussian_names=plys)

def _spawn_sam3d(extra, *, runtime: str = "full"):
    errors: list[str] = []
    for sam3d_py in _sam3d_python_candidates():
        if not Path(sam3d_py).is_file():
            errors.append(f"{sam3d_py}: python 不存在")
            continue
        ok, detail = _check_sam3d_runtime(sam3d_py, runtime)
        if ok:
            cmd = [sam3d_py, *extra]; print("[infer_stage] spawn:", " ".join(cmd))
            if subprocess.run(cmd).returncode != 0: _die("sam3d 子进程失败", 1)
            return
        errors.append(f"{sam3d_py}:\n{detail}")
    _die(
        f"找不到满足 runtime={runtime!r} 的 sam3d python。\n"
        "可设置 SAM3D_VENV_PYTHON 显式指定解释器。\n"
        + "\n\n".join(errors),
        3,
    )

def _check_sam3d_runtime(py: str, runtime: str) -> tuple[bool, str]:
    probes = {
        "full": (
            "import os; os.environ.setdefault('LIDRA_SKIP_INIT','true'); "
            "import sam3d_objects; "
            "from sam3d_objects.pipeline.inference_pipeline_pointmap import InferencePipelinePointMap"
        ),
        "ss_decoder": (
            "import os; os.environ.setdefault('LIDRA_SKIP_INIT','true'); "
            "os.environ.setdefault('ATTN_BACKEND','sdpa'); "
            "import numpy; import torch, hydra, omegaconf; import sam3d_objects; "
            "from sam3d_objects.model.backbone.tdfy_dit.models.sparse_structure_vae "
            "import SparseStructureDecoderTdfyWrapper, SparseStructureEncoderTdfyWrapper"
        ),
    }
    probe = probes.get(runtime)
    if probe is None:
        return False, f"unknown sam3d runtime: {runtime}"
    result = subprocess.run([py, "-c", probe], capture_output=True, text=True)
    if result.returncode != 0:
        detail = (result.stderr or result.stdout).strip().splitlines()[-8:]
        return False, "\n".join(detail)
    return True, ""

if __name__ == "__main__": main()
