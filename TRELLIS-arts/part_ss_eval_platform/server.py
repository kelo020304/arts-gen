from __future__ import annotations

import json
import math
import mimetypes
import os
import subprocess
import threading
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, unquote, urlparse

from .archive import ExperimentArchive, safe_artifact_path
from .jobs import ExperimentExistsError, JobManager, JobRequest
from . import infer_runs
from .infer_jobs import InferJobRequest, InferStageExistsError, build_infer_command
from . import kinematic_runs
from .kinematic_jobs import KinematicJobRequest, build_kinematic_command


PACKAGE_ROOT = Path(__file__).resolve().parent
STATIC_ROOT = PACKAGE_ROOT / "static"
REPO_ROOT = PACKAGE_ROOT.parents[1]


class PlatformState:
    def __init__(self, *, roots: list[Path], output_root: Path, repo_root: Path):
        self.roots = roots
        self.output_root = output_root
        self.repo_root = repo_root
        self.archive = ExperimentArchive(roots)
        self.jobs = JobManager(repo_root)
        # Inference产物根：默认同时扫描 inference/ 和 full-stage/。
        # full-stage 的目录形态同样是 <root>/<object_id>/<run_id>/，例如
        # /robot/data-lab/jzh/art-gen-output/full-stage/102252/102252/run_...
        infer_bases_env = os.environ.get("PART_SS_PLATFORM_INFER_BASES")
        infer_base_env = os.environ.get("PART_SS_PLATFORM_INFER_BASE")
        if infer_bases_env:
            self.infer_bases = [
                Path(item.strip()) for item in infer_bases_env.split(",") if item.strip()
            ]
        elif infer_base_env:
            self.infer_bases = [Path(infer_base_env)]
        else:
            self.infer_bases = [roots[0] / "inference", roots[0] / "full-stage"]
        # Backward-compatible alias for tests or callers that inspect state.
        self.infer_base = self.infer_bases[0]
        # eval data_config：供 /api/infer/inputs 复用 eval dataset 取数；载入失败留 None，
        # inputs/jobs 端点再按需报错（绝不静默假设），不阻塞 roots/runs/manifest/artifact。
        self.infer_data_config = None
        cfg_path = os.environ.get("PART_SS_PLATFORM_INFER_DATA_CONFIG") or (
            str(repo_root) + "/TRELLIS-arts/configs/arts/part_ss_latent_flow/part_ss_latent_flow.yaml"
        )
        data_root_override = os.environ.get("PART_SS_PLATFORM_INFER_DATA_ROOT") or None
        try:
            from inference_pipeline import data_config_io
            self.infer_data_config = data_config_io.load_data_config(
                cfg_path, data_root_override=data_root_override
            )
        except Exception:
            self.infer_data_config = None


def _json_sanitize(obj: Any) -> Any:
    """Replace NaN/Inf floats with None so the body is STANDARD JSON.

    Metrics legitimately contain NaN (empty size buckets, 0-part runs) and the
    raw summary.json embeds NaN too. Emitting the literal `NaN` (allow_nan=True)
    makes the browser's fetch().json() throw "Unexpected token N ... not valid
    JSON". The frontend already renders null as "-".
    """
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, dict):
        return {key: _json_sanitize(value) for key, value in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_sanitize(value) for value in obj]
    return obj


def _json_response(handler: BaseHTTPRequestHandler, payload: Any, status: int = 200) -> None:
    data = json.dumps(_json_sanitize(payload), ensure_ascii=False, allow_nan=False).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(data)))
    handler.end_headers()
    handler.wfile.write(data)


def _error(handler: BaseHTTPRequestHandler, status: int, code: str, message: str) -> None:
    _json_response(handler, {"error": {"code": code, "message": message}}, status)


def _read_body(handler: BaseHTTPRequestHandler) -> dict[str, Any]:
    length = int(handler.headers.get("Content-Length", "0") or "0")
    if length <= 0:
        return {}
    return json.loads(handler.rfile.read(length).decode("utf-8"))


def _tail(path: Path, max_bytes: int = 65536) -> str:
    if not path.is_file():
        return ""
    size = path.stat().st_size
    with path.open("rb") as fh:
        if size > max_bytes:
            fh.seek(size - max_bytes)
        return fh.read().decode("utf-8", errors="replace")


def _watch_kinematic_job(job, run_dir: str) -> None:
    """Persist terminal state so kinematic runs do not stay 'running' forever."""
    def wait_and_update() -> None:
        code = job.process.wait()
        if job.terminated:
            status = "terminated"
        else:
            status = "done" if code == 0 else "failed"
        kinematic_runs.update_run_meta(
            Path(run_dir),
            status=status,
            returncode=code,
            finished_at=time.time(),
        )

    threading.Thread(target=wait_and_update, daemon=True).start()


def _query_gpus() -> list[dict[str, Any]]:
    """Per-GPU memory via nvidia-smi. Empty list if it's missing/unavailable."""
    try:
        out = subprocess.run(
            ["nvidia-smi",
             "--query-gpu=index,memory.total,memory.used,memory.free,utilization.gpu",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        )
    except (FileNotFoundError, OSError, subprocess.SubprocessError):
        return []
    if out.returncode != 0:
        return []
    gpus: list[dict[str, Any]] = []
    for line in out.stdout.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 5:
            continue
        try:
            gpus.append({
                "index": int(parts[0]),
                "total_mb": int(parts[1]),
                "used_mb": int(parts[2]),
                "free_mb": int(parts[3]),
                "util": int(parts[4]),
            })
        except ValueError:
            continue
    return gpus


def _job_request_from_payload(payload: dict[str, Any], output_root: Path) -> JobRequest:
    # Trim leading/trailing whitespace on every text field — a stray space in
    # experiment_name otherwise creates a directory like "name /full_eval".
    def s(key: str, default: str = "") -> str:
        value = payload.get(key)
        return (str(value) if value is not None else default).strip()

    return JobRequest(
        task_type=s("task_type"),
        view_mode=s("view_mode"),
        experiment_name=s("experiment_name"),
        output_root=s("output_root") or str(output_root),
        checkpoint=s("checkpoint"),
        load_dir=s("load_dir"),
        step=int(payload["step"]) if str(payload.get("step", "")).strip() not in ("", "None") else None,
        gpu_ids=s("gpu_ids", "0") or "0",
        max_samples=int(payload.get("max_samples", -1)),
        sample_mode=s("sample_mode", "all") or "all",
        object_ids=s("object_ids"),
        overrides=[str(item).strip() for item in (payload.get("overrides") or []) if str(item).strip()],
        overwrite=bool(payload.get("overwrite")),
    )


def create_handler(state: PlatformState):
    class Handler(BaseHTTPRequestHandler):
        server_version = "PartSSEvalPlatform/0.1"

        def log_message(self, fmt: str, *args: Any) -> None:
            return

        def do_GET(self) -> None:
            parsed = urlparse(self.path)
            path = unquote(parsed.path)
            try:
                if path == "/api/summary":
                    self._api_summary()
                elif path == "/api/experiments":
                    _json_response(self, {"experiments": state.archive.list_experiments()})
                elif path.startswith("/api/experiments/"):
                    exp_id = path.rsplit("/", 1)[-1]
                    try:
                        detail = state.archive.get_experiment(exp_id)
                        _json_response(self, {"experiment": detail, **detail})
                    except KeyError:
                        _error(self, HTTPStatus.NOT_FOUND, "not_found", f"experiment not found: {exp_id}")
                elif path == "/api/jobs":
                    _json_response(self, {"jobs": state.jobs.list_jobs()})
                elif path == "/api/gpus":
                    _json_response(self, {"gpus": _query_gpus()})
                elif path.startswith("/api/jobs/") and path.endswith("/log"):
                    job_id = path.split("/")[3]
                    job = state.jobs.get(job_id)
                    _json_response(self, {"job_id": job_id, "log": _tail(Path(job.command.log_path))})
                elif path == "/api/infer/roots":
                    roots = []
                    seen = set()
                    for base in state.infer_bases:
                        base_path = Path(base)
                        if base_path.is_dir():
                            entry = {"name": base_path.name, "path": str(base_path.resolve())}
                            key = entry["path"]
                            if key not in seen:
                                seen.add(key)
                                roots.append(entry)
                        for entry in infer_runs.list_roots(base):
                            key = entry.get("path") or entry.get("name")
                            if key in seen:
                                continue
                            seen.add(key)
                            roots.append(entry)
                    _json_response(self, {"roots": roots})
                elif path == "/api/infer/runs":
                    root = self._qs(parsed, "root")
                    _json_response(self, {"runs": infer_runs.list_runs(root)})
                elif path == "/api/infer/objects":
                    if state.infer_data_config is None:
                        _error(self, HTTPStatus.INTERNAL_SERVER_ERROR, "no_data_config",
                               "infer data_config 未载入（检查 PART_SS_PLATFORM_INFER_DATA_CONFIG）")
                        return
                    limit = int(self._qs(parsed, "limit", "5000") or "5000")
                    _json_response(self, {"objects": infer_runs.list_objects(state.infer_data_config, limit=limit)})
                elif path == "/api/infer/latest_run":
                    root = self._qs(parsed, "root")
                    object_id = self._qs(parsed, "object_id")
                    mode = self._qs(parsed, "mode", "")
                    view = self._qs(parsed, "view", "")
                    angle_raw = self._qs(parsed, "angle_idx", "")
                    angle_idx = int(angle_raw) if str(angle_raw).strip() != "" else None
                    _json_response(self, {
                        "run": infer_runs.latest_run(
                            root, object_id, mode=mode, view=view, angle_idx=angle_idx
                        )
                    })
                elif path == "/api/infer/manifest":
                    root = self._qs(parsed, "root")
                    object_id = self._qs(parsed, "object_id")
                    run_id = self._qs(parsed, "run_id")
                    angle_idx = self._qs(parsed, "angle_idx", "")
                    try:
                        _json_response(self, infer_runs.read_manifest(root, object_id, run_id, angle_idx=angle_idx))
                    except KeyError:
                        _error(self, HTTPStatus.NOT_FOUND, "not_found",
                               f"run not found: {object_id}/{run_id}")
                elif path == "/api/infer/artifact":
                    root = self._qs(parsed, "root")
                    object_id = self._qs(parsed, "object_id")
                    run_id = self._qs(parsed, "run_id")
                    angle_idx = self._qs(parsed, "angle_idx", "")
                    rel = self._qs(parsed, "rel")
                    target = infer_runs.safe_run_artifact(root, object_id, run_id, rel, angle_idx=angle_idx)
                    if not target.is_file():
                        # voxel.bin 是 3D viewer 唯一能读的格式，但它是后期（9d17b51）才开始
                        # 写的；旧 run / 只落了 voxel.npz 的情况下，从同目录 voxel.npz 即时转
                        # 成 .bin 字节再发，而不是 404（npz 才是服务端/测试契约里的权威产物）。
                        if rel == "voxel.bin":
                            npz = target.with_name("voxel.npz")
                            if npz.is_file():
                                from inference_pipeline.voxel_io import npz_to_bin_bytes
                                self._send_bytes(npz_to_bin_bytes(npz), "application/octet-stream")
                                return
                        _error(self, HTTPStatus.NOT_FOUND, "not_found", f"artifact not found: {rel}")
                        return
                    self._send_file(target)
                elif path == "/api/infer/part_voxels":
                    root = self._qs(parsed, "root")
                    object_id = self._qs(parsed, "object_id")
                    run_id = self._qs(parsed, "run_id")
                    angle_idx = self._qs(parsed, "angle_idx", "")
                    try:
                        data = infer_runs.part_voxels_combined(root, object_id, run_id, angle_idx=angle_idx)
                    except KeyError as exc:
                        _error(self, HTTPStatus.NOT_FOUND, "not_found", str(exc))
                        return
                    self._send_bytes(data, "application/octet-stream")
                elif path == "/api/infer/stage_outputs":
                    root = self._qs(parsed, "root")
                    object_id = self._qs(parsed, "object_id")
                    run_id = self._qs(parsed, "run_id")
                    angle_idx = self._qs(parsed, "angle_idx", "")
                    try:
                        _json_response(self, {"outputs": infer_runs.stage_outputs(root, object_id, run_id, angle_idx=angle_idx)})
                    except KeyError:
                        _error(self, HTTPStatus.NOT_FOUND, "not_found",
                               f"run not found: {object_id}/{run_id}")
                elif path == "/api/infer/part_labels":
                    root = self._qs(parsed, "root")
                    object_id = self._qs(parsed, "object_id")
                    run_id = self._qs(parsed, "run_id")
                    angle_idx = self._qs(parsed, "angle_idx", "")
                    try:
                        _json_response(self, infer_runs.part_labels(root, object_id, run_id, angle_idx=angle_idx))
                    except KeyError:
                        _error(self, HTTPStatus.NOT_FOUND, "not_found",
                               f"run not found: {object_id}/{run_id}")
                elif path == "/api/infer/rgb":
                    self._api_infer_rgb(parsed)
                elif path == "/api/infer/log":
                    self._api_infer_log(parsed)
                elif path == "/api/infer/inputs":
                    object_id = self._qs(parsed, "object_id")
                    angle_idx = int(self._qs(parsed, "angle_idx", "0") or "0")
                    view_mode = self._qs(parsed, "view_mode", "four") or "four"
                    if state.infer_data_config is None:
                        _error(self, HTTPStatus.INTERNAL_SERVER_ERROR, "no_data_config",
                               "infer data_config 未载入（检查 PART_SS_PLATFORM_INFER_DATA_CONFIG）")
                        return
                    try:
                        _json_response(self, infer_runs.object_inputs_preview(
                            state, object_id, angle_idx=angle_idx, view_mode=view_mode))
                    except KeyError as exc:
                        _error(self, HTTPStatus.NOT_FOUND, "not_found", str(exc))
                elif path == "/api/infer/options":
                    _json_response(self, {
                        "configs": infer_runs.list_configs(state.repo_root),
                        "checkpoints": infer_runs.list_checkpoints(state.repo_root, state.roots),
                    })
                elif path == "/api/kin/runs":
                    root = self._qs(parsed, "root", "")
                    _json_response(self, {"runs": kinematic_runs.list_runs(root or None)})
                elif path == "/api/kin/log":
                    run_id = self._qs(parsed, "run_id")
                    root = self._qs(parsed, "root", "")
                    log_path = kinematic_runs.safe_kin_artifact(
                        run_id, "agent_subprocess.log", root=root or None
                    )
                    _json_response(self, {"run_id": run_id, "log": _tail(log_path)})
                elif path == "/api/kin/artifact":
                    run_id = self._qs(parsed, "run_id")
                    rel = self._qs(parsed, "rel")
                    root = self._qs(parsed, "root", "")
                    target = kinematic_runs.safe_kin_artifact(run_id, rel, root=root or None)
                    if not target.is_file():
                        _error(self, HTTPStatus.NOT_FOUND, "not_found", f"artifact not found: {rel}")
                        return
                    self._send_file(target)
                elif path == "/api/eval/options":
                    # Checkpoint dropdown for the eval job form (same scan as the
                    # inference dropdown; eval reads config from the ckpt itself).
                    _json_response(self, {
                        "checkpoints": infer_runs.list_checkpoints(state.repo_root, state.roots),
                    })
                elif path.startswith("/artifacts/"):
                    self._serve_artifact(path)
                else:
                    self._serve_static(parsed.path)
            except ValueError as exc:
                _error(self, HTTPStatus.BAD_REQUEST, "bad_request", str(exc))
            except Exception as exc:
                _error(self, HTTPStatus.INTERNAL_SERVER_ERROR, "internal_error", str(exc))

        def do_POST(self) -> None:
            path = unquote(urlparse(self.path).path)
            if path.startswith("/api/jobs/") and path.endswith("/terminate"):
                job_id = path.split("/")[3]
                try:
                    result = state.jobs.terminate(job_id)
                    _json_response(self, {"ok": True, "job": result})
                except KeyError:
                    _error(self, HTTPStatus.NOT_FOUND, "not_found", f"job not found: {job_id}")
                except Exception as exc:
                    _error(self, HTTPStatus.INTERNAL_SERVER_ERROR, "internal_error", str(exc))
                return
            if path == "/api/infer/jobs":
                try:
                    body = _read_body(self)
                    req = InferJobRequest(**body)
                    # 先确认 manifest 里有该 object（spec §9）：无则 400，绝不静默放过。
                    if state.infer_data_config is None:
                        _error(self, HTTPStatus.INTERNAL_SERVER_ERROR, "no_data_config",
                               "infer data_config 未载入（检查 PART_SS_PLATFORM_INFER_DATA_CONFIG）")
                        return
                    try:
                        infer_runs.object_inputs_preview(
                            state, req.object_id, angle_idx=req.angle_idx, view_mode=req.view)
                    except KeyError as exc:
                        _error(self, HTTPStatus.BAD_REQUEST, "bad_request",
                               f"object not in manifest: {req.object_id} ({exc})")
                        return
                    cmd = build_infer_command(req, repo_root=state.repo_root)
                    job = state.jobs.launch_command(cmd, name=req.name)
                    _json_response(self, {"job": job.to_dict()}, HTTPStatus.CREATED)
                except InferStageExistsError as exc:
                    _error(self, HTTPStatus.CONFLICT, "infer_stage_exists", str(exc))
                except (TypeError, ValueError, json.JSONDecodeError) as exc:
                    _error(self, HTTPStatus.BAD_REQUEST, "bad_request", str(exc))
                except Exception as exc:
                    _error(self, HTTPStatus.INTERNAL_SERVER_ERROR, "internal_error", str(exc))
                return
            if path == "/api/kin/jobs":
                try:
                    body = _read_body(self)
                    req = KinematicJobRequest(**body)
                    cmd = build_kinematic_command(req, repo_root=state.repo_root)
                    job = state.jobs.launch_command(cmd, name=req.name, kind="kinematic")
                    kinematic_runs.update_run_meta(Path(cmd.run_dir), status="running", job_id=job.id)
                    _watch_kinematic_job(job, cmd.run_dir)
                    _json_response(self, {"job": job.to_dict(), "run_id": Path(cmd.run_dir).name}, HTTPStatus.CREATED)
                except (TypeError, ValueError, KeyError, json.JSONDecodeError, FileNotFoundError) as exc:
                    _error(self, HTTPStatus.BAD_REQUEST, "bad_request", str(exc))
                except Exception as exc:
                    _error(self, HTTPStatus.INTERNAL_SERVER_ERROR, "internal_error", str(exc))
                return
            if path != "/api/jobs":
                _error(self, HTTPStatus.NOT_FOUND, "not_found", "unknown endpoint")
                return
            try:
                request = _job_request_from_payload(_read_body(self), state.output_root)
                job = state.jobs.launch(request)
                _json_response(self, {"job": job.to_dict()}, HTTPStatus.CREATED)
            except ExperimentExistsError as exc:
                # Distinct code so the UI can offer "overwrite?" instead of failing.
                _error(self, HTTPStatus.CONFLICT, "experiment_exists", str(exc))
            except (ValueError, KeyError, json.JSONDecodeError) as exc:
                _error(self, HTTPStatus.BAD_REQUEST, "bad_request", str(exc))

        def do_DELETE(self) -> None:
            parsed = urlparse(self.path)
            path = unquote(parsed.path)
            prefix = "/api/experiments/"
            exp_id = path[len(prefix):] if path.startswith(prefix) else ""
            if not exp_id or "/" in exp_id:
                _error(self, HTTPStatus.NOT_FOUND, "not_found", "unknown endpoint")
                return
            try:
                deleted = state.archive.delete_experiment(exp_id)
                _json_response(self, {"ok": True, "deleted": deleted})
            except KeyError:
                _error(self, HTTPStatus.NOT_FOUND, "not_found", f"experiment not found: {exp_id}")
            except ValueError as exc:
                _error(self, HTTPStatus.BAD_REQUEST, "bad_request", str(exc))
            except Exception as exc:
                _error(self, HTTPStatus.INTERNAL_SERVER_ERROR, "internal_error", str(exc))

        def _api_summary(self) -> None:
            jobs = state.jobs.list_jobs()
            running = sum(1 for job in jobs if job["status"] == "running")
            completed = len(state.archive.list_experiments()) + sum(1 for job in jobs if job["status"] == "completed")
            _json_response(self, {"running": running, "completed": completed})

        def _api_infer_rgb(self, parsed) -> None:
            """serve the input RGB png for (object_id, angle_idx, view).

            Resolve via inference_pipeline.inputs_materialize.resolve_rgb_path
            (it builds <data_root>/<mask_subdir>/<id>/angle_<a>/rgb/view_<v>.png
            and raises FileNotFoundError if missing). Confirm the resolved path is
            inside data_root before serving — never serve outside the dataset.
            """
            if state.infer_data_config is None:
                _error(self, HTTPStatus.INTERNAL_SERVER_ERROR, "no_data_config",
                       "infer data_config 未载入（检查 PART_SS_PLATFORM_INFER_DATA_CONFIG）")
                return
            object_id = self._qs(parsed, "object_id")
            angle_idx = self._qs(parsed, "angle_idx", "0") or "0"
            view = self._qs(parsed, "view")
            from inference_pipeline.inputs_materialize import resolve_rgb_path
            try:
                p = resolve_rgb_path(
                    state.infer_data_config,
                    object_id=object_id,
                    angle_idx=int(angle_idx),
                    view_index=int(view),
                )
            except (FileNotFoundError, KeyError) as exc:
                _error(self, HTTPStatus.NOT_FOUND, "not_found", str(exc))
                return
            except ValueError as exc:
                _error(self, HTTPStatus.BAD_REQUEST, "bad_request", str(exc))
                return
            # 越界校验必须与 resolve_rgb_path 的构造对称：它用 <data_root>/<mask_subdir>
            # 拼路径、只做 .is_file()（自动跟随软链）。开发机上数据集是软链桥接的，对“文件”
            # resolve() 会跟随软链到真实位置，而对未穿软链的 data_root 做 resolve() 落在另一处
            # → 合法图片被误判越界（HTTP 400 → 浏览器 <img> 显示裂图）。改成同样穿软链的
            # render 根（<data_root>/<mask_subdir> 的 resolve）作前缀比较；仍能挡住 ../ 逃逸。
            resolved = Path(p).resolve()
            dc = state.infer_data_config
            render_root = (Path(dc["data_root"]) / dc.get("mask_subdir", "renders")).resolve()
            if not (str(resolved) + os.sep).startswith(str(render_root) + os.sep):
                _error(self, HTTPStatus.BAD_REQUEST, "bad_request",
                       f"rgb 路径越界（renders 根之外）：{resolved}")
                return
            self._send_file(resolved)

        def _api_infer_log(self, parsed) -> None:
            """serve a stage's log text (<run_dir>/<stage>.log) for copyable display.

            stage ∈ {ss,part,slat,assemble}. Locate the run dir path-safely via
            infer_runs._run_dir (containment-checked), then _tail the stage log —
            _tail returns "" when the file doesn't exist yet (tolerant).
            """
            root = self._qs(parsed, "root")
            object_id = self._qs(parsed, "object_id")
            run_id = self._qs(parsed, "run_id")
            angle_idx = self._qs(parsed, "angle_idx", "")
            stage = self._qs(parsed, "stage")
            if stage not in ("ss", "part", "slat", "assemble"):
                _error(self, HTTPStatus.BAD_REQUEST, "bad_request", f"未知 stage：{stage}")
                return
            rd = infer_runs._run_dir(root, object_id, run_id, angle_idx=angle_idx)
            log_path = rd / f"{stage}.log"
            _json_response(self, {"stage": stage, "log": _tail(log_path)})

        def _qs(self, parsed, key: str, default: str | None = None) -> str:
            """解析 querystring 中的单值；缺必填（无 default）抛 ValueError → 上层 400。"""
            values = parse_qs(parsed.query).get(key)
            if values:
                return values[0]
            if default is not None:
                return default
            raise ValueError(f"缺少必填参数: {key}")

        def _serve_static(self, path: str) -> None:
            if path in {"", "/"}:
                relative = "index.html"
            else:
                relative = path.lstrip("/")
            target = (STATIC_ROOT / relative).resolve()
            try:
                target.relative_to(STATIC_ROOT.resolve())
            except ValueError:
                _error(self, HTTPStatus.BAD_REQUEST, "bad_request", "invalid static path")
                return
            if not target.is_file():
                target = STATIC_ROOT / "index.html"
            self._send_file(target)

        def _serve_artifact(self, path: str) -> None:
            _, _, rest = path.partition("/artifacts/")
            exp_id, _, rel = rest.partition("/")
            try:
                exp = state.archive.get_experiment(exp_id)
            except KeyError:
                _error(self, HTTPStatus.NOT_FOUND, "not_found", f"experiment not found: {exp_id}")
                return
            target = safe_artifact_path(Path(exp["root"]), rel)
            if not target.is_file():
                _error(self, HTTPStatus.NOT_FOUND, "not_found", f"artifact not found: {rel}")
                return
            self._send_file(target)

        def _send_file(self, path: Path) -> None:
            data = path.read_bytes()
            ctype = mimetypes.guess_type(str(path))[0] or "application/octet-stream"
            self._send_bytes(data, ctype)

        def _send_bytes(self, data: bytes, ctype: str) -> None:
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(data)))
            # Dev tool served from disk: never let the browser run a stale
            # cached app.js/styles.css. Always revalidate so edits take effect
            # on a normal refresh (no hard-reload needed).
            self.send_header("Cache-Control", "no-store, max-age=0")
            self.end_headers()
            self.wfile.write(data)

    return Handler


def create_server(*, host: str, port: int, roots: list[str | Path], output_root: str | Path) -> ThreadingHTTPServer:
    state = PlatformState(
        roots=[Path(root).expanduser() for root in roots],
        output_root=Path(output_root).expanduser(),
        repo_root=REPO_ROOT,
    )
    return ThreadingHTTPServer((host, int(port)), create_handler(state))


DEFAULT_ROOTS = "/robot/data-lab/jzh/art-gen-output,/robot/data-lab/arts-gen-data/output"


def main() -> None:
    # `os.environ.get(key, default)` returns "" when the var is SET but empty
    # (e.g. `PART_SS_PLATFORM_ROOTS=` in the shell), which would scan no roots
    # and show "未扫描到实验". `... or DEFAULT_ROOTS` falls back on empty too.
    roots_env = os.environ.get("PART_SS_PLATFORM_ROOTS") or DEFAULT_ROOTS
    roots = [item.strip() for item in roots_env.split(",") if item.strip()]
    if not roots:
        roots = [item.strip() for item in DEFAULT_ROOTS.split(",") if item.strip()]
    output_root = os.environ.get("PART_SS_PLATFORM_OUTPUT_ROOT") or roots[0]
    host = os.environ.get("PART_SS_PLATFORM_HOST") or "0.0.0.0"
    port = int(os.environ.get("PART_SS_PLATFORM_PORT") or "7861")
    httpd = create_server(host=host, port=port, roots=roots, output_root=output_root)
    print(f"[part_ss_eval_platform] serving on http://{host}:{port}", flush=True)
    print(f"[part_ss_eval_platform] scanning roots: {roots}", flush=True)
    httpd.serve_forever()


if __name__ == "__main__":
    main()
