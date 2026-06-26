from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

from .jobs import CommandSpec
from . import kinematic_runs


@dataclass
class KinematicJobRequest:
    source_root: str
    object_id: str
    source_run_id: str
    angle_idx: int = 0
    output_root: str = ""
    test_data_root: str = ""
    gpu_ids: str = "0"
    agent_loop: bool = False
    max_agent_iterations: int = 3
    skip_motion_validation: bool = True
    live_viewer: bool = False
    overwrite: bool = False

    @property
    def name(self) -> str:
        return f"kin:{self.object_id}-{self.angle_idx}/{self.source_run_id}"


def build_kinematic_command(req: KinematicJobRequest, *, repo_root: Path) -> CommandSpec:
    prepared = kinematic_runs.prepare_kinematic_input(
        source_root=req.source_root,
        object_id=req.object_id,
        source_run_id=req.source_run_id,
        angle_idx=req.angle_idx,
        out_root=req.output_root or None,
        test_root=req.test_data_root or None,
        overwrite=req.overwrite,
    )
    post_root = repo_root / "post_process"
    script = post_root / "kinematic_solver" / "estimate_limit.py"
    if not script.is_file():
        # part_ss_eval_platform lives in TRELLIS-arts but repo_root points at
        # /root/code/arts-gen. Keep the error explicit if the layout changes.
        raise FileNotFoundError(f"kinematic solver entrypoint not found: {script}")

    args = [
        sys.executable,
        str(script),
        "--context-json",
        str(prepared.context_json),
        "--initial-joints-json",
        str(prepared.initial_joints_json),
        "--out-dir",
        str(prepared.run_dir),
    ]
    # The default platform path treats dataset GT as the temporary VLM result.
    # estimate_limit.py overlays VLM axes first, but if converter assets are
    # passed it also derives geometry axis candidates and may replace those GT
    # axes. Only pass converter assets when a geometry-dependent feature needs
    # them.
    if req.live_viewer or not req.skip_motion_validation:
        args.extend(["--converter-output-root", str(prepared.converter_output_root)])
    if req.live_viewer:
        args.extend(["--live-viewer", "--no-live-server"])
    if req.skip_motion_validation:
        args.append("--skip-motion-validation")
    if req.agent_loop:
        args.extend(["--agent-loop", "--max-agent-iterations", str(int(req.max_agent_iterations))])

    gpu = (str(req.gpu_ids or "0").split(",")[0].strip() or "0")
    kinematic_runs.update_run_meta(
        prepared.run_dir,
        status="queued",
        agent_loop=bool(req.agent_loop),
        skip_motion_validation=bool(req.skip_motion_validation),
        live_viewer=bool(req.live_viewer),
    )
    return CommandSpec(
        args=args,
        env={"CUDA_VISIBLE_DEVICES": gpu, "PYTHONPATH": str(repo_root)},
        cwd=str(repo_root),
        log_path=str(prepared.run_dir / "agent_subprocess.log"),
        run_dir=str(prepared.run_dir),
    )
