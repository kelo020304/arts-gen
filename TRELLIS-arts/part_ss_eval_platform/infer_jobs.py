from __future__ import annotations
import sys
import json
from dataclasses import dataclass
from pathlib import Path
from .jobs import CommandSpec
from . import infer_runs
_STAGES = ("ss", "part", "slat", "assemble")


class InferStageExistsError(ValueError):
    """Raised when a stage already has artifacts and overwrite is not confirmed."""


@dataclass
class InferJobRequest:
    stage: str
    object_id: str
    root: str
    run_id: str
    mode: str
    view: str
    data_config: str
    angle_idx: int = 0
    data_root: str = ""
    part_flow_ckpt: str = ""
    part_seg_ckpt: str = ""
    ss_decoder_ckpt: str = ""
    ss_encoder_ckpt: str = ""
    ss_flow_ckpt: str = ""
    part_backend: str = "part_flow"
    # part 阶段逐 part latent 的解码后端：
    #   trellis → TRELLIS SparseStructureDecoder（ss_dec_conv3d）。0526 这类用 TRELLIS
    #             SS VAE 训练的 part flow ckpt 必须用它（latent 在 TRELLIS SS 空间）。
    #   sam3d   → sam3d 的 ss_decoder（用 sam3d SS VAE 训练的新模型）。
    decode_backend: str = "trellis"
    # slat 阶段实验开关：
    #   parts → 现有 body+parts SAM3D SLat；
    #   whole → 整体 voxel + 图直接 SAM3D SLat；
    #   both  → 同一 run 下同时写 overall 与 body/parts。
    slat_scope: str = "parts"
    gpu_ids: str = "0"
    sam3d_python: str = ""
    sam3d_pipeline_yaml: str = ""
    overwrite: bool = False

    @property
    def name(self):
        return f"{self.object_id}/{self.run_id}:{self.stage}"


def build_infer_command(req, *, repo_root: Path) -> CommandSpec:
    if req.stage not in _STAGES:
        raise ValueError(f"stage 必须 {_STAGES}")
    if req.part_backend not in ("part_flow", "promptable_seg"):
        raise ValueError("part_backend 必须是 part_flow 或 promptable_seg")
    if req.stage == "part" and req.part_backend == "part_flow" and not req.part_flow_ckpt:
        raise ValueError("part_flow part 阶段需要 --part-flow-ckpt")
    if req.stage == "part" and req.part_backend == "promptable_seg" and not req.part_seg_ckpt:
        raise ValueError("promptable_seg part 阶段需要 --part-seg-ckpt")
    gpu = (req.gpu_ids.split(",")[0] or "0")
    run_dir = str(infer_runs._run_dir(req.root, req.object_id, req.run_id, angle_idx=req.angle_idx))
    if Path(run_dir).is_dir() and not req.overwrite:
        outputs = infer_runs.stage_outputs(
            req.root, req.object_id, req.run_id, angle_idx=req.angle_idx
        )
        stage_status = ""
        meta_path = Path(run_dir) / "meta.json"
        if meta_path.is_file():
            try:
                stage_status = json.loads(meta_path.read_text()).get("stage_status", {}).get(req.stage, "")
            except (json.JSONDecodeError, OSError):
                stage_status = ""
        if outputs.get(req.stage, {}).get("exists") or stage_status in {"done", "running"}:
            artifacts = ", ".join(outputs.get(req.stage, {}).get("artifacts", [])[:5])
            detail = artifacts or f"meta.stage_status={stage_status}"
            raise InferStageExistsError(
                f"{req.stage} 阶段已有产物或状态：{detail}。请确认覆盖后重跑。"
            )
    args = [sys.executable, "scripts/inference/infer_stage.py", "--stage", req.stage,
            "--object-id", str(req.object_id), "--root", req.root, "--run-id", req.run_id,
            "--mode", req.mode, "--view", req.view, "--angle-idx", str(req.angle_idx),
            "--data-config", req.data_config, "--gpu", gpu]
    if req.data_root:
        args += ["--data-root", req.data_root]
    if req.part_flow_ckpt:
        args += ["--part-flow-ckpt", req.part_flow_ckpt]
    if req.part_seg_ckpt:
        args += ["--part-seg-ckpt", req.part_seg_ckpt]
    if req.ss_decoder_ckpt:
        args += ["--ss-decoder-ckpt", req.ss_decoder_ckpt]
    if req.ss_encoder_ckpt:
        args += ["--ss-encoder-ckpt", req.ss_encoder_ckpt]
    if req.ss_flow_ckpt:
        args += ["--ss-flow-ckpt", req.ss_flow_ckpt]
    if req.part_backend:
        args += ["--part-backend", req.part_backend]
    if req.decode_backend:
        args += ["--decode-backend", req.decode_backend]
    if req.slat_scope:
        args += ["--slat-scope", req.slat_scope]
    if req.overwrite:
        args += ["--overwrite"]
    env = {"CUDA_VISIBLE_DEVICES": gpu}
    if req.sam3d_python:
        env["SAM3D_VENV_PYTHON"] = req.sam3d_python
    if req.sam3d_pipeline_yaml:
        env["SAM3D_PIPELINE_YAML"] = req.sam3d_pipeline_yaml
    return CommandSpec(args=args, env=env, cwd=str(repo_root),
                       log_path=str(Path(run_dir) / f"{req.stage}.log"), run_dir=run_dir)
