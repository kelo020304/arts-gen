#!/usr/bin/env python3
"""
Render full-object TRELLIS-style multi-view RGB images for each object angle.

This is intentionally separate from the historical quadrant rendering layout:
- quadrant rendering produced fixed grouped RGB/mask views per angle;
- this script renders the complete assembled object from 150 Hammersley-sampled
  camera views by default, following TRELLIS dataset_toolkits/render.py.

Output layout:
    renders/<object_id>/angle_<i>/render_full_obj_all_view/
        000.png
        001.png
        ...
        149.png
        transforms.json
"""

from __future__ import annotations

import argparse
import atexit
import hashlib
import json
import math
import os
import signal
import subprocess
import sys
import tempfile
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any


SCRIPT_PATH = Path(__file__).resolve()
SCRIPT_DIR = SCRIPT_PATH.parent
DEFAULT_OUTPUT_SUBDIR = "render_full_obj_all_view"
DEFAULT_NUM_VIEWS = 150
DEFAULT_FOV_DEG = 40.0
DEFAULT_RADIUS = 2.0
DEFAULT_TIMEOUT_SECONDS = 7200
PR_SET_PDEATHSIG = 1

LIVE_PROCS: set[subprocess.Popen[Any]] = set()
SHUTDOWN_REQUESTED = False
SHUTDOWN_SIGNAL_COUNT = 0
LAST_SHUTDOWN_SIGNAL: int | None = None


@dataclass(frozen=True)
class JobSpec:
    object_id: str
    angle_idx: int
    finaljson_path: Path
    objs_dir: Path
    joint_transforms_path: Path
    output_dir: Path
    blender_binary: str
    resolution: int
    num_views: int
    fov_deg: float
    radius: float
    engine: str
    samples: int
    seed: int
    obj_up_axis: str

    @property
    def desc(self) -> str:
        return f"{self.object_id} angle_{self.angle_idx}"


@dataclass
class PreparedJob:
    spec: JobSpec
    command: list[str]
    log_path: Path
    transforms_json_path: Path
    views_json_path: Path
    start_time: float | None = None


def _require_mapping(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise TypeError(f"{name} must be a mapping, got {type(value).__name__}")
    return value


def _require_list(value: Any, name: str) -> list[Any]:
    if not isinstance(value, list):
        raise TypeError(f"{name} must be a list, got {type(value).__name__}")
    return value


def _require_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    return int(value)


def _validate_matrix4x4(value: Any, name: str) -> list[list[float]]:
    rows = _require_list(value, name)
    if len(rows) != 4:
        raise ValueError(f"{name} must contain 4 rows, got {len(rows)}")
    matrix: list[list[float]] = []
    for row_idx, row in enumerate(rows):
        row_values = _require_list(row, f"{name}[{row_idx}]")
        if len(row_values) != 4:
            raise ValueError(f"{name}[{row_idx}] must contain 4 values, got {len(row_values)}")
        matrix_row: list[float] = []
        for col_idx, item in enumerate(row_values):
            if isinstance(item, bool) or not isinstance(item, (int, float)):
                raise TypeError(f"{name}[{row_idx}][{col_idx}] must be numeric")
            matrix_row.append(float(item))
        matrix.append(matrix_row)
    return matrix


def radical_inverse(base: int, n: int) -> float:
    value = 0.0
    inv_base = 1.0 / base
    inv_base_n = inv_base
    while n > 0:
        digit = n % base
        value += digit * inv_base_n
        n //= base
        inv_base_n *= inv_base
    return value


def sphere_hammersley_sequence(n: int, num_samples: int, offset: tuple[float, float]) -> tuple[float, float]:
    """Match TRELLIS dataset_toolkits/utils.py sphere_hammersley_sequence."""
    u = n / num_samples
    v = radical_inverse(2, n)
    u += offset[0] / num_samples
    v += offset[1]
    u = 2 * u if u < 0.25 else 2 / 3 * u + 1 / 3
    theta = math.acos(1 - 2 * u) - math.pi / 2
    phi = v * 2 * math.pi
    return phi, theta


def deterministic_view_offset(object_id: str, angle_idx: int, seed: int) -> tuple[float, float]:
    """Stable replacement for TRELLIS' np.random.rand() offset."""
    digest = hashlib.sha256(f"{seed}:{object_id}:{angle_idx}".encode("utf-8")).digest()
    denom = float(1 << 64)
    return (
        int.from_bytes(digest[:8], "big") / denom,
        int.from_bytes(digest[8:16], "big") / denom,
    )


def build_trellis_views(
    *,
    object_id: str,
    angle_idx: int,
    num_views: int,
    radius: float,
    fov_deg: float,
    seed: int,
) -> list[dict[str, float]]:
    if num_views < 1:
        raise ValueError(f"num_views must be >= 1, got {num_views}")
    if radius <= 0:
        raise ValueError(f"radius must be > 0, got {radius}")
    if not 0 < fov_deg < 180:
        raise ValueError(f"fov_deg must be in (0, 180), got {fov_deg}")

    offset = deterministic_view_offset(object_id, angle_idx, seed)
    fov = math.radians(fov_deg)
    views = []
    for view_idx in range(num_views):
        yaw, pitch = sphere_hammersley_sequence(view_idx, num_views, offset)
        views.append(
            {
                "yaw": yaw,
                "pitch": pitch,
                "radius": radius,
                "fov": fov,
            }
        )
    return views


def _parse_csv(value: str, field_name: str) -> list[str]:
    items = [item.strip() for item in value.split(",")]
    if not items or any(not item for item in items):
        raise ValueError(f"{field_name} must be a comma-separated list of non-empty values")
    if len(items) != len(set(items)):
        raise ValueError(f"{field_name} contains duplicate values")
    return items


def _parse_angle_ids(raw_value: str | None) -> list[int] | None:
    if raw_value is None:
        return None
    angle_ids: list[int] = []
    for item in _parse_csv(raw_value, "--angle-ids"):
        try:
            angle_idx = int(item)
        except ValueError as exc:
            raise ValueError(f"--angle-ids values must be integers, got {item!r}") from exc
        if angle_idx < 0:
            raise ValueError(f"--angle-ids values must be >= 0, got {angle_idx}")
        angle_ids.append(angle_idx)
    return angle_ids


def _resolve_object_ids(config: Any, object_ids_arg: str | None) -> list[str]:
    available_object_ids = config.list_object_ids()
    if object_ids_arg is None:
        return available_object_ids

    requested_object_ids = _parse_csv(object_ids_arg, "--object-ids")
    available_set = set(available_object_ids)
    unknown_or_filtered = [
        object_id for object_id in requested_object_ids if object_id not in available_set
    ]
    if unknown_or_filtered:
        raise ValueError(
            "Unknown or filtered-out object IDs in --object-ids: "
            + ", ".join(unknown_or_filtered)
        )
    return requested_object_ids


def _load_angle_transforms(joint_transforms_path: Path, object_id: str, angle_idx: int) -> dict[str, Any]:
    with joint_transforms_path.open("r", encoding="utf-8") as handle:
        archive = _require_mapping(json.load(handle), f"joint_transforms[{joint_transforms_path}]")

    archive_object_id = str(archive.get("object_id"))
    if archive_object_id != object_id:
        raise ValueError(
            f"object_id mismatch in {joint_transforms_path}: expected {object_id}, got {archive_object_id}"
        )

    num_parts = _require_int(archive.get("num_parts"), "joint_transforms['num_parts']")
    angles = _require_mapping(archive.get("angles"), "joint_transforms['angles']")
    angle_key = str(angle_idx)
    if angle_key not in angles:
        raise KeyError(f"Missing angle '{angle_key}' in {joint_transforms_path}")

    angle_data = _require_mapping(angles[angle_key], f"joint_transforms['angles']['{angle_key}']")
    joint_states = _require_mapping(angle_data.get("joint_states"), f"angles['{angle_key}']['joint_states']")
    part_transforms = _require_mapping(
        angle_data.get("part_transforms"),
        f"angles['{angle_key}']['part_transforms']",
    )

    expected_keys = {str(part_idx) for part_idx in range(num_parts)}
    actual_keys = set(part_transforms)
    missing_keys = sorted(expected_keys - actual_keys)
    extra_keys = sorted(actual_keys - expected_keys)
    if missing_keys:
        raise KeyError(f"Missing part transforms for object {object_id} angle_{angle_idx}: {missing_keys}")
    if extra_keys:
        raise KeyError(f"Unexpected part transforms for object {object_id} angle_{angle_idx}: {extra_keys}")

    for part_idx in range(num_parts):
        _validate_matrix4x4(part_transforms[str(part_idx)], f"part_transforms['{part_idx}']")

    return {
        "object_id": object_id,
        "num_parts": num_parts,
        "angle_idx": angle_idx,
        "joint_states": joint_states,
        "part_transforms": part_transforms,
    }


def _write_temp_json(prefix: str, payload: Any, suffix: str = ".json") -> Path:
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        prefix=prefix,
        suffix=suffix,
        delete=False,
    ) as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")
        return Path(handle.name)


def _create_temp_log_path(prefix: str) -> Path:
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        prefix=prefix,
        suffix=".log",
        delete=False,
    ) as handle:
        return Path(handle.name)


def _output_is_complete(output_dir: Path, num_views: int) -> bool:
    transforms_path = output_dir / "transforms.json"
    if not transforms_path.is_file():
        return False
    return all((output_dir / f"{view_idx:03d}.png").is_file() for view_idx in range(num_views))


def build_jobs(
    config: Any,
    object_ids: list[str],
    *,
    angle_ids: list[int] | None,
    num_views: int,
    resolution: int | None,
    engine: str,
    samples: int,
    fov_deg: float,
    radius: float,
    seed: int,
    output_subdir: str,
    force: bool,
) -> tuple[list[JobSpec], int]:
    from config_loader import resolve_obj_up_axis

    finaljson_root = Path(config.finaljson_dir)
    objs_root = Path(config.partseg_dir)
    joint_root = Path(config.joint_transforms_dir)
    renders_root = Path(config.renders_dir)
    jobs: list[JobSpec] = []
    skipped_jobs = 0
    render_resolution = resolution if resolution is not None else int(config.render.resolution)

    for object_id in object_ids:
        finaljson_path = finaljson_root / f"{object_id}.json"
        if not finaljson_path.is_file():
            raise FileNotFoundError(f"Missing finaljson: {finaljson_path}")

        objs_dir = objs_root / object_id / "objs"
        if not objs_dir.is_dir():
            raise FileNotFoundError(f"Missing OBJ directory: {objs_dir}")

        joint_transforms_path = joint_root / f"{object_id}.json"
        if not joint_transforms_path.is_file():
            raise FileNotFoundError(f"Missing joint transforms: {joint_transforms_path}")

        object_num_angles = config.get_num_angles(object_id)
        selected_angles = angle_ids if angle_ids is not None else list(range(object_num_angles))
        invalid_angles = [angle_idx for angle_idx in selected_angles if angle_idx >= object_num_angles]
        if invalid_angles:
            raise ValueError(
                f"Object {object_id} has {object_num_angles} angle(s); invalid --angle-ids: {invalid_angles}"
            )

        obj_up_axis = resolve_obj_up_axis(config, finaljson_path)
        for angle_idx in selected_angles:
            output_dir = renders_root / object_id / f"angle_{angle_idx}" / output_subdir
            if not force and _output_is_complete(output_dir, num_views):
                skipped_jobs += 1
                continue

            jobs.append(
                JobSpec(
                    object_id=object_id,
                    angle_idx=angle_idx,
                    finaljson_path=finaljson_path,
                    objs_dir=objs_dir,
                    joint_transforms_path=joint_transforms_path,
                    output_dir=output_dir,
                    blender_binary=config.render.blender,
                    resolution=render_resolution,
                    num_views=num_views,
                    fov_deg=fov_deg,
                    radius=radius,
                    engine=engine,
                    samples=samples,
                    seed=seed,
                    obj_up_axis=obj_up_axis,
                )
            )

    return jobs, skipped_jobs


def prepare_job(spec: JobSpec) -> PreparedJob:
    spec.output_dir.mkdir(parents=True, exist_ok=True)
    angle_payload = _load_angle_transforms(spec.joint_transforms_path, spec.object_id, spec.angle_idx)
    views_payload = build_trellis_views(
        object_id=spec.object_id,
        angle_idx=spec.angle_idx,
        num_views=spec.num_views,
        radius=spec.radius,
        fov_deg=spec.fov_deg,
        seed=spec.seed,
    )
    transforms_json_path = _write_temp_json(
        prefix=f"render_full_{spec.object_id}_angle_{spec.angle_idx}_transforms_",
        payload=angle_payload,
    )
    views_json_path = _write_temp_json(
        prefix=f"render_full_{spec.object_id}_angle_{spec.angle_idx}_views_",
        payload=views_payload,
    )
    log_path = _create_temp_log_path(prefix=f"render_full_{spec.object_id}_angle_{spec.angle_idx}_")
    command = [
        spec.blender_binary,
        "--background",
        "--python",
        str(SCRIPT_PATH),
        "--",
        "--blender-worker",
        "--objs-dir",
        str(spec.objs_dir),
        "--finaljson",
        str(spec.finaljson_path),
        "--transforms-json",
        str(transforms_json_path),
        "--views-json",
        str(views_json_path),
        "--output-folder",
        str(spec.output_dir),
        "--resolution",
        str(spec.resolution),
        "--engine",
        spec.engine,
        "--samples",
        str(spec.samples),
        "--obj-up-axis",
        spec.obj_up_axis,
    ]
    return PreparedJob(
        spec=spec,
        command=command,
        log_path=log_path,
        transforms_json_path=transforms_json_path,
        views_json_path=views_json_path,
    )


def _cleanup_job_artifacts(job: PreparedJob) -> None:
    for path in (job.transforms_json_path, job.views_json_path, job.log_path):
        try:
            path.unlink()
        except FileNotFoundError:
            pass


def _register_proc(proc: subprocess.Popen[Any]) -> None:
    LIVE_PROCS.add(proc)


def _unregister_proc(proc: subprocess.Popen[Any]) -> None:
    LIVE_PROCS.discard(proc)


def _signal_proc_group(proc: subprocess.Popen[Any], sig: int) -> None:
    if proc.poll() is not None:
        return
    os.killpg(proc.pid, sig)


def _terminate_processes(procs: list[subprocess.Popen[Any]], swallow_process_lookup: bool) -> None:
    live = [proc for proc in procs if proc.poll() is None]
    if not live:
        return

    for proc in live:
        if swallow_process_lookup:
            try:
                _signal_proc_group(proc, signal.SIGTERM)
            except ProcessLookupError:
                pass
        else:
            _signal_proc_group(proc, signal.SIGTERM)

    time.sleep(2)

    for proc in live:
        if proc.poll() is None:
            if swallow_process_lookup:
                try:
                    _signal_proc_group(proc, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            else:
                _signal_proc_group(proc, signal.SIGKILL)


def cleanup() -> None:
    live = list(LIVE_PROCS)
    if not live:
        return
    _terminate_processes(live, swallow_process_lookup=True)
    for proc in live:
        proc.poll()
        if proc.returncode is not None:
            _unregister_proc(proc)


atexit.register(cleanup)


def _configure_child_process() -> None:
    if not sys.platform.startswith("linux"):
        return

    import ctypes

    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(PR_SET_PDEATHSIG, signal.SIGTERM) != 0:
        errno_value = ctypes.get_errno()
        raise OSError(errno_value, os.strerror(errno_value))


def _handle_shutdown_signal(signum: int, _frame: Any) -> None:
    global SHUTDOWN_REQUESTED, SHUTDOWN_SIGNAL_COUNT, LAST_SHUTDOWN_SIGNAL

    SHUTDOWN_REQUESTED = True
    SHUTDOWN_SIGNAL_COUNT += 1
    LAST_SHUTDOWN_SIGNAL = signum

    if SHUTDOWN_SIGNAL_COUNT >= 2:
        for proc in list(LIVE_PROCS):
            _signal_proc_group(proc, signal.SIGKILL)
        os._exit(1)


def _install_signal_handlers() -> dict[int, Any]:
    handled_signals = [signal.SIGINT, signal.SIGTERM]
    if hasattr(signal, "SIGHUP"):
        handled_signals.append(signal.SIGHUP)

    original_handlers: dict[int, Any] = {}
    for signum in handled_signals:
        original_handlers[signum] = signal.getsignal(signum)
        signal.signal(signum, _handle_shutdown_signal)
    return original_handlers


def _restore_signal_handlers(original_handlers: dict[int, Any]) -> None:
    for signum, handler in original_handlers.items():
        signal.signal(signum, handler)


def launch_blender(command: list[str], log_path: Path) -> subprocess.Popen[Any]:
    log_handle = log_path.open("w", encoding="utf-8")
    try:
        return subprocess.Popen(
            command,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            preexec_fn=_configure_child_process if sys.platform.startswith("linux") else None,
        )
    finally:
        log_handle.close()


def _read_log(log_path: Path) -> str:
    try:
        content = log_path.read_text(encoding="utf-8", errors="replace")
    except FileNotFoundError:
        return "<log missing>"
    return content if content.strip() else "<log empty>"


def _raise_blender_failure(job: PreparedJob, returncode: int) -> None:
    log_content = _read_log(job.log_path)
    _cleanup_job_artifacts(job)
    raise RuntimeError(
        f"Blender subprocess failed for {job.spec.desc} "
        f"(exit code {returncode}) with log:\n{log_content}"
    )


def _check_job_timeouts(running: dict[subprocess.Popen[Any], PreparedJob], timeout_seconds: int) -> None:
    now = time.monotonic()
    for proc, job in list(running.items()):
        if job.start_time is None:
            continue
        elapsed = now - job.start_time
        if elapsed <= timeout_seconds:
            continue

        _terminate_processes([proc], swallow_process_lookup=False)
        _unregister_proc(proc)
        del running[proc]
        log_content = _read_log(job.log_path)
        _cleanup_job_artifacts(job)
        raise RuntimeError(
            f"Blender subprocess timed out for {job.spec.desc} after {timeout_seconds}s with log:\n"
            f"{log_content}"
        )


def parse_driver_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render full-object 150-view TRELLIS-style RGB images for object angles."
    )
    parser.add_argument("--config", required=True, help="Path to dataset toolkit YAML config.")
    parser.add_argument("--object-ids", help="Optional comma-separated object ID subset.")
    parser.add_argument("--angle-ids", help="Optional comma-separated angle index subset, e.g. 0 or 0,1.")
    parser.add_argument("--workers", type=int, default=1, help="Parallel Blender subprocess workers.")
    parser.add_argument("--num-views", type=int, default=DEFAULT_NUM_VIEWS, help="Views per object angle.")
    parser.add_argument(
        "--resolution",
        type=int,
        default=None,
        help="Square render resolution; defaults to render.resolution from config.",
    )
    parser.add_argument(
        "--engine",
        default="BLENDER_EEVEE_NEXT",
        help="Blender render engine. Use CYCLES for stricter TRELLIS-style rendering.",
    )
    parser.add_argument("--samples", type=int, default=128, help="Cycles samples when --engine CYCLES is used.")
    parser.add_argument("--fov-deg", type=float, default=DEFAULT_FOV_DEG, help="Camera horizontal FOV in degrees.")
    parser.add_argument("--radius", type=float, default=DEFAULT_RADIUS, help="Camera orbit radius.")
    parser.add_argument("--seed", type=int, default=0, help="Seed for deterministic Hammersley view offsets.")
    parser.add_argument(
        "--output-subdir",
        default=DEFAULT_OUTPUT_SUBDIR,
        help="Subdirectory under renders/<object_id>/angle_<i>/ for these full-object views.",
    )
    parser.add_argument("--force", action="store_true", help="Re-render even when all expected outputs exist.")
    parser.add_argument("--dry-run", action="store_true", help="List jobs without launching Blender.")
    parser.add_argument(
        "--timeout-seconds",
        type=int,
        default=DEFAULT_TIMEOUT_SECONDS,
        help="Timeout per Blender job.",
    )
    return parser.parse_args(argv)


def main_driver(argv: list[str] | None = None) -> int:
    global SHUTDOWN_REQUESTED, SHUTDOWN_SIGNAL_COUNT, LAST_SHUTDOWN_SIGNAL

    sys.path.insert(0, str(SCRIPT_DIR.parent / "utils"))
    from config_loader import load_config

    args = parse_driver_args(argv)
    if args.workers <= 0:
        raise ValueError("--workers must be positive")
    if args.num_views <= 0:
        raise ValueError("--num-views must be positive")
    if args.samples <= 0:
        raise ValueError("--samples must be positive")
    if args.timeout_seconds <= 0:
        raise ValueError("--timeout-seconds must be positive")
    if "/" in args.output_subdir or args.output_subdir in {"", ".", ".."}:
        raise ValueError("--output-subdir must be a simple directory name")

    config = load_config(args.config)
    object_ids = _resolve_object_ids(config, args.object_ids)
    angle_ids = _parse_angle_ids(args.angle_ids)
    Path(config.renders_dir).mkdir(parents=True, exist_ok=True)

    SHUTDOWN_REQUESTED = False
    SHUTDOWN_SIGNAL_COUNT = 0
    LAST_SHUTDOWN_SIGNAL = None

    jobs, skipped_jobs = build_jobs(
        config,
        object_ids,
        angle_ids=angle_ids,
        num_views=args.num_views,
        resolution=args.resolution,
        engine=args.engine,
        samples=args.samples,
        fov_deg=args.fov_deg,
        radius=args.radius,
        seed=args.seed,
        output_subdir=args.output_subdir,
        force=args.force,
    )
    total_jobs = len(jobs)

    if args.dry_run:
        print(
            f"Dry run: objects={len(object_ids)} jobs={total_jobs} skipped={skipped_jobs} "
            f"views_per_job={args.num_views}",
            flush=True,
        )
        for job in jobs[:20]:
            print(f"  {job.desc} -> {job.output_dir}", flush=True)
        if len(jobs) > 20:
            print(f"  ... {len(jobs) - 20} more job(s)", flush=True)
        return 0

    if total_jobs == 0:
        print(f"No pending full-object render jobs. skipped={skipped_jobs}", flush=True)
        return 0

    completed_jobs = 0
    total_views_rendered = 0
    pending_jobs = deque(jobs)
    running: dict[subprocess.Popen[Any], PreparedJob] = {}
    original_handlers = _install_signal_handlers()

    try:
        while pending_jobs or running:
            if SHUTDOWN_REQUESTED:
                raise KeyboardInterrupt(f"received signal {LAST_SHUTDOWN_SIGNAL}")

            while len(running) < args.workers and pending_jobs and not SHUTDOWN_REQUESTED:
                spec = pending_jobs.popleft()
                prepared = prepare_job(spec)
                if SHUTDOWN_REQUESTED:
                    _cleanup_job_artifacts(prepared)
                    break

                try:
                    proc = launch_blender(prepared.command, prepared.log_path)
                except Exception:
                    _cleanup_job_artifacts(prepared)
                    raise

                prepared.start_time = time.monotonic()
                running[proc] = prepared
                _register_proc(proc)

            if SHUTDOWN_REQUESTED:
                raise KeyboardInterrupt(f"received signal {LAST_SHUTDOWN_SIGNAL}")

            finished: list[tuple[subprocess.Popen[Any], PreparedJob, int]] = []
            for proc, job in list(running.items()):
                returncode = proc.poll()
                if returncode is not None:
                    finished.append((proc, job, returncode))

            for proc, job, returncode in finished:
                _unregister_proc(proc)
                del running[proc]
                if returncode != 0:
                    _raise_blender_failure(job, returncode)

                completed_jobs += 1
                total_views_rendered += job.spec.num_views
                _cleanup_job_artifacts(job)
                print(
                    f"[{completed_jobs}/{total_jobs}] {job.spec.desc} full-object done "
                    f"({job.spec.num_views} views)",
                    flush=True,
                )

            _check_job_timeouts(running, args.timeout_seconds)

            if not finished and running and not SHUTDOWN_REQUESTED:
                time.sleep(0.5)

        if SHUTDOWN_REQUESTED:
            raise KeyboardInterrupt(f"received signal {LAST_SHUTDOWN_SIGNAL}")

        print(
            f"Summary: objects={len(object_ids)} jobs={total_jobs} skipped={skipped_jobs} "
            f"views_rendered={total_views_rendered}",
            flush=True,
        )
        return 0
    finally:
        cleanup()
        for job in running.values():
            _cleanup_job_artifacts(job)
        _restore_signal_handlers(original_handlers)


def parse_blender_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Blender worker for full-object TRELLIS-style multi-view rendering."
    )
    parser.add_argument("--blender-worker", action="store_true")
    parser.add_argument("--objs-dir", required=True)
    parser.add_argument("--finaljson", required=True)
    parser.add_argument("--transforms-json", required=True)
    parser.add_argument("--views-json", required=True)
    parser.add_argument("--output-folder", required=True)
    parser.add_argument("--resolution", type=int, required=True)
    parser.add_argument("--engine", required=True)
    parser.add_argument("--samples", type=int, required=True)
    parser.add_argument("--obj-up-axis", choices=("Y", "Z"), default="Y")
    if "--" not in argv:
        raise ValueError("Blender script arguments must be passed after '--'")
    args = parser.parse_args(argv[argv.index("--") + 1 :])
    if not args.blender_worker:
        raise ValueError("Missing --blender-worker")
    return args


def main_blender_worker(argv: list[str]) -> int:
    # Imports are intentionally local so the driver can run outside Blender.
    import bpy
    import numpy as np
    from mathutils import Matrix, Vector

    y_up_to_z_up = Matrix.Rotation(math.radians(90.0), 4, "X")
    z_up_to_y_up = y_up_to_z_up.inverted()

    def init_render(engine: str, resolution: int, samples: int) -> None:
        scene = bpy.context.scene
        scene.render.engine = engine
        scene.render.resolution_x = resolution
        scene.render.resolution_y = resolution
        scene.render.resolution_percentage = 100
        scene.render.image_settings.file_format = "PNG"
        scene.render.image_settings.color_mode = "RGBA"
        scene.render.film_transparent = True
        scene.display_settings.display_device = "sRGB"
        scene.view_settings.view_transform = "Filmic"
        scene.view_settings.look = "None"

        if engine == "CYCLES":
            scene.cycles.samples = samples
            scene.cycles.filter_type = "BOX"
            scene.cycles.filter_width = 1
            scene.cycles.diffuse_bounces = 1
            scene.cycles.glossy_bounces = 1
            scene.cycles.transparent_max_bounces = 3
            scene.cycles.transmission_bounces = 3
            scene.cycles.use_denoising = True
            try:
                scene.cycles.device = "GPU"
                prefs = bpy.context.preferences.addons["cycles"].preferences
                prefs.get_devices()
                prefs.compute_device_type = "CUDA"
            except Exception as exc:  # pragma: no cover - Blender runtime only
                print(f"[WARN] Could not enable CUDA cycles rendering: {exc}", flush=True)

    def init_scene() -> None:
        for obj in list(bpy.data.objects):
            bpy.data.objects.remove(obj, do_unlink=True)
        for material in list(bpy.data.materials):
            bpy.data.materials.remove(material, do_unlink=True)
        for texture in list(bpy.data.textures):
            bpy.data.textures.remove(texture, do_unlink=True)
        for image in list(bpy.data.images):
            bpy.data.images.remove(image, do_unlink=True)

    def init_camera():
        cam = bpy.data.objects.new("Camera", bpy.data.cameras.new("Camera"))
        bpy.context.collection.objects.link(cam)
        bpy.context.scene.camera = cam
        cam.data.sensor_height = cam.data.sensor_width = 32
        cam_constraint = cam.constraints.new(type="TRACK_TO")
        cam_constraint.track_axis = "TRACK_NEGATIVE_Z"
        cam_constraint.up_axis = "UP_Y"
        cam_empty = bpy.data.objects.new("Empty", None)
        cam_empty.location = (0, 0, 0)
        cam_empty.empty_display_size = 0
        cam_empty.hide_render = True
        bpy.context.scene.collection.objects.link(cam_empty)
        cam_constraint.target = cam_empty
        return cam

    def init_lighting() -> None:
        bpy.ops.object.select_all(action="DESELECT")
        bpy.ops.object.select_by_type(type="LIGHT")
        bpy.ops.object.delete()

        default_light = bpy.data.objects.new(
            "Default_Light",
            bpy.data.lights.new("Default_Light", type="POINT"),
        )
        bpy.context.collection.objects.link(default_light)
        default_light.data.energy = 1000
        default_light.location = (4, 1, 6)

        top_light = bpy.data.objects.new("Top_Light", bpy.data.lights.new("Top_Light", type="AREA"))
        bpy.context.collection.objects.link(top_light)
        top_light.data.energy = 10000
        top_light.location = (0, 0, 10)
        top_light.scale = (100, 100, 100)

        bottom_light = bpy.data.objects.new(
            "Bottom_Light",
            bpy.data.lights.new("Bottom_Light", type="AREA"),
        )
        bpy.context.collection.objects.link(bottom_light)
        bottom_light.data.energy = 1000
        bottom_light.location = (0, 0, -10)

    def scene_bbox() -> tuple[Any, Any]:
        bbox_min = (math.inf,) * 3
        bbox_max = (-math.inf,) * 3
        found = False
        for obj in bpy.context.scene.objects.values():
            if not isinstance(obj.data, bpy.types.Mesh):
                continue
            found = True
            for coord in obj.bound_box:
                coord = obj.matrix_world @ Vector(coord)
                bbox_min = tuple(min(x, y) for x, y in zip(bbox_min, coord))
                bbox_max = tuple(max(x, y) for x, y in zip(bbox_max, coord))
        if not found:
            raise RuntimeError("No mesh objects in scene")
        return Vector(bbox_min), Vector(bbox_max)

    def normalize_scene() -> tuple[float, Any]:
        scene_root_objects = [obj for obj in bpy.context.scene.objects.values() if not obj.parent]
        if len(scene_root_objects) > 1:
            scene = bpy.data.objects.new("ParentEmpty", None)
            bpy.context.scene.collection.objects.link(scene)
            scene.empty_display_size = 0
            scene.hide_render = True
            for obj in scene_root_objects:
                obj.parent = scene
        else:
            scene = scene_root_objects[0]

        bbox_min, bbox_max = scene_bbox()
        scale = 1.0 / max(bbox_max - bbox_min)
        scene.scale = scene.scale * scale

        bpy.context.view_layer.update()
        bbox_min, bbox_max = scene_bbox()
        offset = -(bbox_min + bbox_max) / 2
        scene.matrix_world.translation += offset
        bpy.ops.object.select_all(action="DESELECT")
        return scale, offset

    def get_transform_matrix(obj: Any) -> list[list[float]]:
        pos, rt, _ = obj.matrix_world.decompose()
        rt = rt.to_matrix()
        matrix = []
        for row_idx in range(3):
            row = [rt[row_idx][col_idx] for col_idx in range(3)]
            row.append(pos[row_idx])
            matrix.append(row)
        matrix.append([0.0, 0.0, 0.0, 1.0])
        return matrix

    def load_parts_with_transforms(
        objs_dir: str,
        finaljson_path: str,
        transforms_json_path: str,
        obj_up_axis: str,
    ) -> None:
        with open(finaljson_path, "r", encoding="utf-8") as handle:
            finaljson_data = _require_mapping(json.load(handle), f"finaljson[{finaljson_path}]")
        with open(transforms_json_path, "r", encoding="utf-8") as handle:
            transforms_data = _require_mapping(json.load(handle), f"transforms[{transforms_json_path}]")

        parts = _require_list(finaljson_data.get("parts"), "finaljson['parts']")
        part_transforms = _require_mapping(transforms_data.get("part_transforms"), "transforms['part_transforms']")

        imported_mesh_count = 0
        for part_idx, part_data in enumerate(parts):
            part_mapping = _require_mapping(part_data, f"finaljson['parts'][{part_idx}]")
            obj_names = _require_list(part_mapping.get("obj"), f"parts[{part_idx}]['obj']")
            if not obj_names:
                raise ValueError(f"parts[{part_idx}]['obj'] must not be empty")

            matrix_key = str(part_idx)
            if matrix_key not in part_transforms:
                raise KeyError(f"Missing transform for part {part_idx} in {transforms_json_path}")
            raw_matrix = _validate_matrix4x4(part_transforms[matrix_key], f"part_transforms['{matrix_key}']")
            if obj_up_axis == "Y":
                part_matrix = y_up_to_z_up @ Matrix(raw_matrix) @ z_up_to_y_up
                import_kwargs = {}
            else:
                part_matrix = Matrix(raw_matrix)
                import_kwargs = {"forward_axis": "Y", "up_axis": "Z"}

            for obj_name in obj_names:
                if not isinstance(obj_name, str) or not obj_name:
                    raise TypeError(f"parts[{part_idx}]['obj'] entries must be non-empty strings")
                obj_path = os.path.join(objs_dir, f"{obj_name}.obj")
                if not os.path.isfile(obj_path):
                    raise FileNotFoundError(f"Missing OBJ for part {part_idx}: {obj_path}")

                before_objects = set(bpy.data.objects.keys())
                bpy.ops.wm.obj_import(filepath=obj_path, **import_kwargs)
                after_objects = set(bpy.data.objects.keys())
                new_object_names = after_objects - before_objects

                mesh_objects = []
                for blender_name in new_object_names:
                    obj = bpy.data.objects[blender_name]
                    if obj.type != "MESH":
                        continue
                    obj.matrix_world = part_matrix @ obj.matrix_world
                    obj.name = f"part_{part_idx:04d}"
                    if obj.data is not None:
                        obj.data.name = f"part_{part_idx:04d}"
                    mesh_objects.append(obj)
                    imported_mesh_count += 1

                if not mesh_objects:
                    raise RuntimeError(f"OBJ import produced no mesh objects: {obj_path}")

        if imported_mesh_count == 0:
            raise RuntimeError("No mesh objects were imported from finaljson")

    args = parse_blender_args(argv)
    output_folder = os.path.abspath(args.output_folder)
    os.makedirs(output_folder, exist_ok=True)

    with open(args.views_json, "r", encoding="utf-8") as handle:
        views = _require_list(json.load(handle), f"views[{args.views_json}]")

    init_render(args.engine, args.resolution, args.samples)
    init_scene()
    load_parts_with_transforms(args.objs_dir, args.finaljson, args.transforms_json, args.obj_up_axis)
    scale, offset = normalize_scene()
    cam = init_camera()
    init_lighting()

    to_export: dict[str, Any] = {
        "aabb": [[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
        "scale": scale,
        "offset": [offset.x, offset.y, offset.z],
        "resolution": args.resolution,
        "engine": args.engine,
        "samples": args.samples,
        "frames": [],
    }

    for view_idx, raw_view in enumerate(views):
        view = _require_mapping(raw_view, f"views[{view_idx}]")
        yaw = float(view["yaw"])
        pitch = float(view["pitch"])
        radius = float(view["radius"])
        fov = float(view["fov"])
        cam.location = (
            radius * np.cos(yaw) * np.cos(pitch),
            radius * np.sin(yaw) * np.cos(pitch),
            radius * np.sin(pitch),
        )
        cam.data.lens = 16 / np.tan(fov / 2)

        bpy.context.scene.render.filepath = os.path.join(output_folder, f"{view_idx:03d}.png")
        bpy.ops.render.render(write_still=True)
        bpy.context.view_layer.update()

        to_export["frames"].append(
            {
                "file_path": f"{view_idx:03d}.png",
                "camera_angle_x": fov,
                "yaw": yaw,
                "pitch": pitch,
                "radius": radius,
                "transform_matrix": get_transform_matrix(cam),
            }
        )

    with open(os.path.join(output_folder, "transforms.json"), "w", encoding="utf-8") as handle:
        json.dump(to_export, handle, indent=2)
        handle.write("\n")

    return 0


if __name__ == "__main__":
    if "--blender-worker" in sys.argv:
        raise SystemExit(main_blender_worker(sys.argv))
    raise SystemExit(main_driver())
