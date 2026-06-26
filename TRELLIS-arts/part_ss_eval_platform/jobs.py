from __future__ import annotations

import os
import re
import signal
import subprocess
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


class ExperimentExistsError(ValueError):
    """Raised when an experiment name already has a run dir and overwrite is off.

    Distinct from a plain ValueError so the API can return a specific code and
    the UI can offer an overwrite choice instead of a hard failure.
    """


# Per-batch progress line printed by the eval, e.g. "[full_eval] samples 5-8/120".
_PROGRESS_RE = re.compile(r"samples\s+\d+-(\d+)/(\d+)")
# Startup banner line printed ONCE per run, e.g. "  samples:         120".
# Used to scope progress to the current run (the log is append-mode).
_RUN_START_RE = re.compile(r"\bsamples:\s+\d+")


def _tail_text(path: Path, max_bytes: int = 8192) -> str:
    if not path.is_file():
        return ""
    size = path.stat().st_size
    with path.open("rb") as fh:
        if size > max_bytes:
            fh.seek(size - max_bytes)
        return fh.read().decode("utf-8", errors="replace")


def _progress_from_logs(run_dir: str) -> dict[str, Any] | None:
    """Real progress from the eval log(s): last 'samples X-Y/N' line.

    Single-GPU writes full_eval.log; multi-shard writes one log per shard, so
    we sum done/total across shards to get overall progress. Returns None when
    no progress line exists yet (e.g. still loading the model, or test export).

    The eval log is APPEND-mode (`tee -a`), so it can contain finished previous
    runs. We therefore only read progress AFTER the current run's startup banner
    ("samples: N"); otherwise a stale "samples 1-1/1" reads as 100% while the new
    run is still initializing.
    """
    run_root = Path(run_dir)
    if not run_root.is_dir():
        return None
    shard_logs = sorted(run_root.glob("full_eval_shard_*_of_*.log"))
    logs = shard_logs if shard_logs else [run_root / "full_eval.log"]
    done = total = 0
    found = False
    for log_path in logs:
        text = _tail_text(log_path, 65536)
        starts = list(_RUN_START_RE.finditer(text))
        segment = text[starts[-1].start():] if starts else text
        matches = _PROGRESS_RE.findall(segment)
        if matches:
            found = True
            last_done, last_total = matches[-1]
            done += int(last_done)
            total += int(last_total)
    if not found or total <= 0:
        return None
    return {"done": done, "total": total, "fraction": min(1.0, done / total)}


@dataclass
class JobRequest:
    task_type: str
    view_mode: str
    experiment_name: str
    output_root: str
    checkpoint: str = ""
    load_dir: str = ""
    step: int | None = None
    gpu_ids: str = "0"
    max_samples: int = -1
    sample_mode: str = "all"
    object_ids: str = ""
    overrides: list[str] = field(default_factory=list)
    overwrite: bool = False


@dataclass
class CommandSpec:
    args: list[str]
    env: dict[str, str]
    cwd: str
    log_path: str
    run_dir: str


@dataclass
class RunningJob:
    id: str
    request: JobRequest
    command: CommandSpec
    process: subprocess.Popen
    started_at: float
    terminated: bool = False

    def to_dict(self) -> dict[str, Any]:
        code = self.process.poll()
        if code is None:
            status = "running"
        elif self.terminated:
            status = "terminated"
        else:
            status = "completed" if code == 0 else "failed"
        progress = _progress_from_logs(self.command.run_dir) if status == "running" else None
        return {
            "id": self.id,
            "name": self.request.experiment_name,
            "task_type": self.request.task_type,
            "view_mode": self.request.view_mode,
            "status": status,
            "returncode": code,
            "started_at": self.started_at,
            "log_path": self.command.log_path,
            "run_dir": self.command.run_dir,
            "gpu_ids": self.request.gpu_ids,
            "progress": progress,
        }


def _script_for(task_type: str, view_mode: str) -> tuple[str, dict[str, str]]:
    if task_type not in {"eval", "test"}:
        raise ValueError(f"task_type must be eval or test, got {task_type}")
    if view_mode not in {"single", "four"}:
        raise ValueError(f"view_mode must be single or four, got {view_mode}")
    raise RuntimeError(
        "Legacy part_ss_latent_flow train/eval launchers were removed from the "
        "open-source active tree. Use scripts/eval/run_ee_eval.bash for the "
        "current 0617 end-to-end eval path, or restore archived legacy launchers "
        "from scripts/_archive/2026-06-train-launchers if you intentionally need "
        "the old platform job mode."
    )


def _gpu_env(gpu_ids: str) -> dict[str, str]:
    ids = [item.strip() for item in str(gpu_ids or "0").split(",") if item.strip()]
    if not ids:
        ids = ["0"]
    return {"GPU_ID": ids[0], "GPU_IDS": ",".join(ids), "NUM_SHARDS": str(len(ids))}


def build_command(request: JobRequest, *, repo_root: Path) -> CommandSpec:
    if not request.experiment_name:
        raise ValueError("experiment_name is required")
    if not request.output_root:
        raise ValueError("output_root is required")
    if not request.checkpoint and not (request.load_dir and request.step is not None):
        raise ValueError("checkpoint or load_dir+step is required")

    script, script_env = _script_for(request.task_type, request.view_mode)
    run_dir = str(Path(request.output_root) / "part_ss_latent_flow" / request.experiment_name)
    # Reject duplicate experiment names unless the caller explicitly overwrites.
    if Path(run_dir).exists() and not request.overwrite:
        raise ExperimentExistsError(
            f"实验名已存在：{request.experiment_name}（{run_dir}）。"
            f"请改个名字，或确认覆盖重跑。"
        )
    log_name = "full_eval.log" if request.task_type == "eval" else "test_export.log"
    env = {
        **script_env,
        **_gpu_env(request.gpu_ids),
        "RUN_ID": request.experiment_name,
        "OUTPUT_ROOT": request.output_root,
        "RUN_DIR": run_dir,
        "MAX_SAMPLES": str(request.max_samples),
        "SAMPLE_MODE": request.sample_mode,
        # overwrite -> force a full recompute (the bash skips existing metrics by default).
        "SKIP_EXISTING": "0" if request.overwrite else "1",
    }
    if request.object_ids:
        env["OBJECT_IDS"] = request.object_ids
    if request.checkpoint:
        env["CHECKPOINT"] = request.checkpoint
    else:
        env["LOAD_DIR"] = request.load_dir
        env["STEP"] = str(request.step)
    if request.task_type == "test":
        env["RUN_EVAL_DECODE"] = "1"
        env["EXPORT_SLAT_ASSETS"] = "1"
    return CommandSpec(
        args=["bash", script, *request.overrides],
        env=env,
        cwd=str(repo_root),
        log_path=str(Path(run_dir) / log_name),
        run_dir=run_dir,
    )


class JobManager:
    def __init__(self, repo_root: Path):
        self.repo_root = repo_root
        self.jobs: dict[str, RunningJob] = {}

    def launch(self, request: JobRequest) -> RunningJob:
        command = build_command(request, repo_root=self.repo_root)
        Path(command.run_dir).mkdir(parents=True, exist_ok=True)
        env = os.environ.copy()
        env.update(command.env)
        # start_new_session=True puts the bash + its python eval child in their
        # OWN process group, so terminate() can kill the WHOLE tree (otherwise
        # killing the bash would orphan the GPU-holding python child).
        process = subprocess.Popen(command.args, cwd=command.cwd, env=env, start_new_session=True)
        job = RunningJob(
            id=uuid.uuid4().hex[:12],
            request=request,
            command=command,
            process=process,
            started_at=time.time(),
        )
        self.jobs[job.id] = job
        return job

    def launch_command(self, command: CommandSpec, *, name: str, kind: str = "infer") -> RunningJob:
        """泛化进程组启动：把任意 CommandSpec 作为 job 跑，复用 terminate/list（进程组杀树）。"""
        Path(command.run_dir).mkdir(parents=True, exist_ok=True)
        log_fh = open(command.log_path, "ab")
        env = os.environ.copy(); env.update(command.env)
        process = subprocess.Popen(command.args, cwd=command.cwd, env=env,
                                   stdout=log_fh, stderr=subprocess.STDOUT, start_new_session=True)
        req = JobRequest(task_type=kind, view_mode="", experiment_name=name,
                         output_root=command.run_dir, gpu_ids=env.get("CUDA_VISIBLE_DEVICES", "0"))
        job = RunningJob(id=uuid.uuid4().hex[:12], request=req, command=command,
                         process=process, started_at=time.time())
        self.jobs[job.id] = job
        return job

    def terminate(self, job_id: str) -> dict[str, Any]:
        job = self.jobs.get(job_id)
        if job is None:
            raise KeyError(job_id)
        if job.process.poll() is None:  # still running
            job.terminated = True
            try:
                pgid = os.getpgid(job.process.pid)
                os.killpg(pgid, signal.SIGTERM)
            except ProcessLookupError:
                pass  # already gone
            else:
                # give it a moment; escalate to SIGKILL if it ignores SIGTERM
                for _ in range(20):
                    if job.process.poll() is not None:
                        break
                    time.sleep(0.1)
                if job.process.poll() is None:
                    try:
                        os.killpg(pgid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
        return job.to_dict()

    def list_jobs(self) -> list[dict[str, Any]]:
        return [job.to_dict() for job in self.jobs.values()]

    def get(self, job_id: str) -> RunningJob:
        return self.jobs[job_id]
