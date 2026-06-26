"""Flask server for the MJCF joint editor web viewer."""

from __future__ import annotations

import json
import math
import os
import shutil
import subprocess
from datetime import datetime
from pathlib import Path, PurePosixPath
from typing import Any

from flask import Flask, jsonify, redirect, request, send_from_directory

from .kinematic_workbench import (
    WorkbenchError,
    import_asset_to_workbench,
    save_initial_joints_json,
    save_workbench_orientation,
)
from .mjcf_parser import generate_manifest, generate_manifest_from_xml_bytes
from .xml_saver import save_editor_state_to_xml


def create_app(
    assets_root: Path,
    default_asset: str | None = None,
    *,
    kinematic_run_dir: Path | None = None,
    kinematic_run_id: str | None = None,
    kinematic_workbench_root: Path | None = None,
) -> Flask:
    """Create and return the editor-only Flask application."""
    assets_root = Path(assets_root).resolve()
    kinematic_run_dir = Path(kinematic_run_dir).resolve() if kinematic_run_dir else None
    kinematic_assets_root = (
        kinematic_run_dir / "object_assets" if kinematic_run_dir is not None else None
    )
    kinematic_run_id = kinematic_run_id or (
        kinematic_run_dir.name if kinematic_run_dir is not None else None
    )
    kinematic_workbench_root = (
        Path(kinematic_workbench_root).resolve() if kinematic_workbench_root else None
    )
    kinematic_workbench_assets_root = (
        kinematic_workbench_root / "object_assets"
        if kinematic_workbench_root is not None
        else None
    )
    kinematic_workbench_runs: dict[str, Path] = {}
    app = Flask(__name__)

    # Resolve utility/frontend locations relative to the project root.
    gen_obj_root = Path(__file__).resolve().parents[1]
    shared_libs_dir = gen_obj_root / "utils" / "shared_libs"
    frontend_dir = gen_obj_root / "utils" / "frontend"

    # ------------------------------------------------------------------
    # Route 1: GET /health -- server readiness
    # ------------------------------------------------------------------
    @app.get("/health")
    def health():
        return jsonify({"status": "ok", "ready": True}), 200

    # ------------------------------------------------------------------
    # Route 2: GET / -- redirect to editor
    # ------------------------------------------------------------------
    @app.route("/")
    def index():
        if kinematic_workbench_root is not None:
            return redirect("/kinematic-workbench")
        if default_asset:
            return redirect(f"/object-post-process/{default_asset}")
        return redirect("/object-post-process/")

    # ------------------------------------------------------------------
    # Route 3: GET /object-post-process/ -- editor with no asset
    # ------------------------------------------------------------------
    @app.route("/object-post-process/")
    def editor_empty():
        return send_from_directory(str(frontend_dir), "mjcf_joint_editor.html")

    # ------------------------------------------------------------------
    # Route 4: GET /object-post-process/<asset> -- editor with asset
    # ------------------------------------------------------------------
    @app.route("/object-post-process/<asset>")
    def editor_with_asset(asset: str):
        return send_from_directory(str(frontend_dir), "mjcf_joint_editor.html")

    # ------------------------------------------------------------------
    # Route 4b: GET /kinematic-agent/<run_id> -- same viewer, agent mode
    # ------------------------------------------------------------------
    @app.route("/kinematic-agent/<run_id>")
    def kinematic_agent_viewer(run_id: str):
        if _resolve_kinematic_run_dir(
            run_id,
            kinematic_run_dir=kinematic_run_dir,
            kinematic_run_id=kinematic_run_id,
            kinematic_workbench_root=kinematic_workbench_root,
            kinematic_workbench_runs=kinematic_workbench_runs,
        ) is None:
            return jsonify(_error_response(f"Unknown kinematic run: {run_id}")), 404
        response = send_from_directory(str(frontend_dir), "mjcf_joint_editor.html")
        response.headers["Cache-Control"] = "no-store"
        return response

    # ------------------------------------------------------------------
    # Route 4c: GET /kinematic-workbench -- standalone local workbench
    # ------------------------------------------------------------------
    @app.route("/kinematic-workbench")
    def kinematic_workbench_viewer():
        if kinematic_workbench_root is None:
            return jsonify(_error_response("Kinematic workbench is not enabled")), 404
        response = send_from_directory(str(frontend_dir), "kinematic_workbench.html")
        response.headers["Cache-Control"] = "no-store"
        return response

    # ------------------------------------------------------------------
    # Route 5: GET /api/assets -- list available assets
    # ------------------------------------------------------------------
    @app.route("/api/assets")
    def list_assets():
        asset_names: set[str] = set()
        for root in (assets_root, kinematic_workbench_assets_root):
            if root is None or not root.is_dir():
                continue
            for child in sorted(root.iterdir()):
                if child.is_dir() and (child / "mjcf").is_dir():
                    asset_names.add(child.name)
        return jsonify(sorted(asset_names))

    # ------------------------------------------------------------------
    # Route 6: GET /api/assets/<asset>/preview-manifest
    # ------------------------------------------------------------------
    @app.route("/api/assets/<asset>/preview-manifest")
    def preview_manifest(asset: str):
        asset_root = _resolve_asset_root(
            asset,
            assets_root,
            kinematic_assets_root,
            kinematic_workbench_assets_root,
            kinematic_workbench_root=kinematic_workbench_root,
            kinematic_workbench_runs=kinematic_workbench_runs,
        )
        result = generate_manifest(asset, asset_root)
        print(f"[server] preview-manifest for {asset}: status={result.get('status')}", flush=True)
        return jsonify(result)

    # ------------------------------------------------------------------
    # Route 6b: GET /api/kinematic-agent/<run_id>/state
    # ------------------------------------------------------------------
    @app.route("/api/kinematic-agent/<run_id>/state")
    def kinematic_agent_state(run_id: str):
        resolved_run_dir = _resolve_kinematic_run_dir(
            run_id,
            kinematic_run_dir=kinematic_run_dir,
            kinematic_run_id=kinematic_run_id,
            kinematic_workbench_root=kinematic_workbench_root,
            kinematic_workbench_runs=kinematic_workbench_runs,
        )
        if resolved_run_dir is None:
            return jsonify(_error_response(f"Unknown kinematic run: {run_id}")), 404
        response = jsonify(_read_kinematic_frontend_state(resolved_run_dir))
        response.headers["Cache-Control"] = "no-store"
        return response

    # ------------------------------------------------------------------
    # Route 6c: GET /api/kinematic-agent/<run_id>/manifest
    # ------------------------------------------------------------------
    @app.route("/api/kinematic-agent/<run_id>/manifest")
    def kinematic_agent_manifest(run_id: str):
        resolved_run_dir = _resolve_kinematic_run_dir(
            run_id,
            kinematic_run_dir=kinematic_run_dir,
            kinematic_run_id=kinematic_run_id,
            kinematic_workbench_root=kinematic_workbench_root,
            kinematic_workbench_runs=kinematic_workbench_runs,
        )
        if resolved_run_dir is None:
            return jsonify(_error_response(f"Unknown kinematic run: {run_id}")), 404
        state = _read_kinematic_frontend_state(resolved_run_dir)
        latest = state.get("latest_preview")
        if not isinstance(latest, dict) or not latest.get("asset_name"):
            return jsonify(_error_response("No kinematic MJCF preview has been written yet")), 404
        result = generate_manifest(str(latest["asset_name"]), resolved_run_dir / "object_assets")
        response = jsonify(result)
        response.headers["Cache-Control"] = "no-store"
        return response

    # ------------------------------------------------------------------
    # Route 6d: Workbench config/import/run APIs
    # ------------------------------------------------------------------
    @app.route("/api/kinematic-workbench/config")
    def kinematic_workbench_config():
        if kinematic_workbench_root is None:
            return jsonify(_error_response("Kinematic workbench is not enabled")), 404
        project_root = Path(__file__).resolve().parents[2]
        payload = {
            "status": "ok",
            "project_root": str(project_root),
            "workspace_root": str(kinematic_workbench_root),
            "default_xml_save_root": str(kinematic_workbench_root / "generated_xml"),
            "default_mesh_save_root": str(kinematic_workbench_root / "generated_mesh"),
            "default_initial_joints_root": str(kinematic_workbench_root / "initial_joints"),
            "default_run_root": str(kinematic_workbench_root / "runs"),
            "default_source_root": "data/RealAppliance",
            "default_model_root": str(project_root / "data" / "RealAppliance" / "source" / "model"),
            "openrouter_base_url": os.environ.get("OPENROUTER_BASE_URL") or "",
            "model": os.environ.get("ARTICRAFT_MODEL") or "gpt-5.5",
            "thinking_level": os.environ.get("ARTICRAFT_THINKING_LEVEL") or "medium",
            "has_openrouter_api_key": bool(
                os.environ.get("OPENROUTER_API_KEY") or os.environ.get("OPENROUTER_API_KEYS")
            ),
        }
        response = jsonify(payload)
        response.headers["Cache-Control"] = "no-store"
        return response

    @app.route("/api/kinematic-workbench/import", methods=["POST"])
    def kinematic_workbench_import():
        if kinematic_workbench_root is None:
            return jsonify(_error_response("Kinematic workbench is not enabled")), 404
        if not request.is_json:
            return jsonify(_error_response("Content-Type must be application/json")), 400
        payload = request.get_json(silent=True)
        if not isinstance(payload, dict):
            return jsonify(_error_response("Malformed JSON body")), 400
        try:
            result = import_asset_to_workbench(payload, kinematic_workbench_root)
        except WorkbenchError as exc:
            return jsonify(_error_response(str(exc))), 400
        return jsonify(result), 200

    @app.route("/api/kinematic-workbench/import-upload", methods=["POST"])
    def kinematic_workbench_import_upload():
        if kinematic_workbench_root is None:
            return jsonify(_error_response("Kinematic workbench is not enabled")), 404
        try:
            result = _import_workbench_uploaded_asset(request, kinematic_workbench_root)
        except (ValueError, WorkbenchError) as exc:
            return jsonify(_error_response(str(exc))), 400
        return jsonify(result), 200

    @app.route("/api/kinematic-workbench/orientation", methods=["POST"])
    def kinematic_workbench_orientation():
        if kinematic_workbench_root is None:
            return jsonify(_error_response("Kinematic workbench is not enabled")), 404
        if not request.is_json:
            return jsonify(_error_response("Content-Type must be application/json")), 400
        payload = request.get_json(silent=True)
        if not isinstance(payload, dict):
            return jsonify(_error_response("Malformed JSON body")), 400
        try:
            result = save_workbench_orientation(
                asset_name=str(payload.get("asset_name") or ""),
                orientation_degrees=payload.get("orientation_degrees") or {},
                workbench_root=kinematic_workbench_root,
            )
        except WorkbenchError as exc:
            return jsonify(_error_response(str(exc))), 400
        return jsonify(result), 200

    @app.route("/api/kinematic-workbench/initial-joints-json", methods=["POST"])
    def kinematic_workbench_initial_joints_json():
        if kinematic_workbench_root is None:
            return jsonify(_error_response("Kinematic workbench is not enabled")), 404
        if not request.is_json:
            return jsonify(_error_response("Content-Type must be application/json")), 400
        payload = request.get_json(silent=True)
        if not isinstance(payload, dict):
            return jsonify(_error_response("Malformed JSON body")), 400
        try:
            result = save_initial_joints_json(payload, kinematic_workbench_root)
        except WorkbenchError as exc:
            return jsonify(_error_response(str(exc))), 400
        return jsonify(result), 200

    @app.route("/api/kinematic-workbench/initial-joints-template", methods=["POST"])
    def kinematic_workbench_initial_joints_template():
        if kinematic_workbench_root is None:
            return jsonify(_error_response("Kinematic workbench is not enabled")), 404
        if not request.is_json:
            return jsonify(_error_response("Content-Type must be application/json")), 400
        payload = request.get_json(silent=True)
        if not isinstance(payload, dict):
            return jsonify(_error_response("Malformed JSON body")), 400
        try:
            result = _build_initial_joints_template_response(
                payload,
                kinematic_workbench_root=kinematic_workbench_root,
            )
        except ValueError as exc:
            return jsonify(_error_response(str(exc))), 400
        return jsonify(result), 200

    @app.route("/api/kinematic-workbench/run-agent", methods=["POST"])
    def kinematic_workbench_run_agent():
        if kinematic_workbench_root is None:
            return jsonify(_error_response("Kinematic workbench is not enabled")), 404
        if not request.is_json:
            return jsonify(_error_response("Content-Type must be application/json")), 400
        payload = request.get_json(silent=True)
        if not isinstance(payload, dict):
            return jsonify(_error_response("Malformed JSON body")), 400
        try:
            result = _start_workbench_agent_run(
                payload,
                kinematic_workbench_root=kinematic_workbench_root,
                kinematic_workbench_runs=kinematic_workbench_runs,
            )
        except ValueError as exc:
            return jsonify(_error_response(str(exc))), 400
        return jsonify(result), 200

    # ------------------------------------------------------------------
    # Route 7: POST /api/assets/<asset>/preview-from-xml
    # ------------------------------------------------------------------
    @app.route("/api/assets/<asset>/preview-from-xml", methods=["POST"])
    def preview_from_xml(asset: str):
        if "file" not in request.files:
            return jsonify({"status": "error", "message": "No file uploaded"}), 400
        uploaded = request.files["file"]
        xml_bytes = uploaded.read()
        if not xml_bytes:
            return jsonify({"status": "error", "message": "Empty file"}), 400
        result = generate_manifest_from_xml_bytes(xml_bytes, asset, assets_root)
        print(f"[server] preview-from-xml for {asset}: status={result.get('status')}", flush=True)
        return jsonify(result)

    # ------------------------------------------------------------------
    # Route 8: POST /api/assets/<asset>/save-xml -- save editor state
    # ------------------------------------------------------------------
    @app.route("/api/assets/<asset>/save-xml", methods=["POST"])
    def save_xml(asset: str):
        asset_root = _resolve_asset_root(
            asset,
            assets_root,
            kinematic_assets_root,
            kinematic_workbench_assets_root,
            kinematic_workbench_root=kinematic_workbench_root,
            kinematic_workbench_runs=kinematic_workbench_runs,
        )
        asset_dir = asset_root / asset
        if not asset_dir.is_dir():
            return jsonify(_error_response(f"Asset does not exist: {asset}")), 400
        if not request.is_json:
            return jsonify(_error_response("Content-Type must be application/json")), 400

        raw_body = request.get_data(cache=True)
        if not raw_body:
            return jsonify(_error_response("Request body must be non-empty JSON")), 400

        request_json = request.get_json(silent=True)
        if request_json is None:
            return jsonify(_error_response("Malformed JSON body")), 400

        print(f"[server] save-xml start for {asset}", flush=True)
        try:
            result = save_editor_state_to_xml(asset, request_json, asset_root)
        except Exception as exc:
            print(f"[server] save-xml unexpected failure for {asset}: {exc}", flush=True)
            return jsonify(_error_response(f"Unexpected server error: {exc}")), 500

        if result.get("status") == "ok":
            print(f"[server] save-xml success for {asset}: xml_path={result.get('xml_path')}", flush=True)
            return jsonify(result), 200

        print(f"[server] save-xml failure for {asset}: {result.get('message')}", flush=True)
        return jsonify(result), 400

    # ------------------------------------------------------------------
    # Route 9: proxy Isaac Sim status + export
    # ------------------------------------------------------------------
    import os
    import urllib.request
    import urllib.error

    _isaac_port = int(os.environ.get("ISAAC_PORT", "8081"))

    @app.route("/api/isaac-status")
    def isaac_status():
        try:
            req = urllib.request.Request(f"http://127.0.0.1:{_isaac_port}/health")
            with urllib.request.urlopen(req, timeout=3) as resp:
                data = resp.read()
                return app.response_class(data, mimetype="application/json")
        except Exception:
            return jsonify({"ready": False, "message": f"Isaac Sim not reachable on port {_isaac_port}"}), 200

    @app.route("/api/assets/<asset>/export-usd", methods=["POST"])
    def export_usd(asset: str):
        try:
            import json as _json
            payload = _json.dumps({"asset_name": asset, "assets_root": str(assets_root)}).encode()
            req = urllib.request.Request(
                f"http://127.0.0.1:{_isaac_port}/export",
                data=payload,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=300) as resp:
                data = resp.read()
                return app.response_class(data, mimetype="application/json")
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", errors="replace")
            return app.response_class(body, status=e.code, mimetype="application/json")
        except Exception as exc:
            return jsonify({"status": "error", "message": str(exc)}), 503

    # ------------------------------------------------------------------
    # Route 10: GET /assets/<asset>/<path:filepath> -- serve asset files
    # ------------------------------------------------------------------
    @app.route("/assets/<asset>/<path:filepath>")
    def serve_asset_file(asset: str, filepath: str):
        base = _resolve_asset_root(
            asset,
            assets_root,
            kinematic_assets_root,
            kinematic_workbench_assets_root,
            kinematic_workbench_root=kinematic_workbench_root,
            kinematic_workbench_runs=kinematic_workbench_runs,
        ) / asset
        target = base / filepath
        if not target.is_file():
            return "Not found", 404
        try:
            target.resolve().relative_to(base.resolve())
        except ValueError:
            return "Forbidden", 403
        return send_from_directory(str(target.parent), target.name)

    # ------------------------------------------------------------------
    # Route 10: GET /shared_libs/<path:filename> -- serve JS libraries
    # ------------------------------------------------------------------
    @app.route("/shared_libs/<path:filename>")
    def serve_shared_libs(filename: str):
        return send_from_directory(str(shared_libs_dir), filename)

    return app


def _error_response(message: str) -> dict[str, Any]:
    return {"status": "error", "message": message, "details": None}


def _resolve_asset_root(
    asset: str,
    assets_root: Path,
    kinematic_assets_root: Path | None,
    kinematic_workbench_assets_root: Path | None = None,
    *,
    kinematic_workbench_root: Path | None = None,
    kinematic_workbench_runs: dict[str, Path] | None = None,
) -> Path:
    run_asset_root = _resolve_workbench_run_asset_root(
        asset,
        kinematic_workbench_root=kinematic_workbench_root,
        kinematic_workbench_runs=kinematic_workbench_runs,
    )
    if run_asset_root is not None:
        return run_asset_root
    if (
        kinematic_workbench_assets_root is not None
        and (kinematic_workbench_assets_root / asset).is_dir()
    ):
        return kinematic_workbench_assets_root
    if kinematic_assets_root is not None and (kinematic_assets_root / asset).is_dir():
        return kinematic_assets_root
    return assets_root


def _resolve_workbench_run_asset_root(
    asset: str,
    *,
    kinematic_workbench_root: Path | None,
    kinematic_workbench_runs: dict[str, Path] | None,
) -> Path | None:
    if kinematic_workbench_runs:
        for run_dir in reversed(list(kinematic_workbench_runs.values())):
            asset_root = run_dir / "object_assets"
            if (asset_root / asset).is_dir():
                return asset_root
    if kinematic_workbench_root is None:
        return None

    runs_root = kinematic_workbench_root / "runs"
    if not runs_root.is_dir():
        return None
    for candidate in sorted(runs_root.glob(f"*/object_assets/{asset}"), reverse=True):
        if candidate.is_dir():
            return candidate.parent
    return None


def _resolve_kinematic_run_dir(
    run_id: str,
    *,
    kinematic_run_dir: Path | None,
    kinematic_run_id: str | None,
    kinematic_workbench_root: Path | None,
    kinematic_workbench_runs: dict[str, Path],
) -> Path | None:
    if kinematic_run_dir is not None and kinematic_run_id == run_id:
        return kinematic_run_dir
    if run_id in kinematic_workbench_runs:
        return kinematic_workbench_runs[run_id]
    if kinematic_workbench_root is None:
        return None
    candidate = kinematic_workbench_root / "runs" / run_id
    if candidate.is_dir():
        kinematic_workbench_runs[run_id] = candidate.resolve()
        return kinematic_workbench_runs[run_id]
    return None


def _start_workbench_agent_run(
    payload: dict[str, Any],
    *,
    kinematic_workbench_root: Path,
    kinematic_workbench_runs: dict[str, Path],
) -> dict[str, Any]:
    object_id = _required_text(payload, "object_id")
    converter_output_root = _required_text(payload, "converter_output_root")
    source_root = _required_text(payload, "source_root")
    initial_joints_json = str(payload.get("initial_joints_json") or "").strip()
    max_iterations = _positive_int(payload.get("max_agent_iterations", 10), "max_agent_iterations")
    heartbeat_seconds = _positive_float(
        payload.get("api_heartbeat_seconds", 1.0),
        "api_heartbeat_seconds",
    )

    out_dir_raw = str(payload.get("out_dir") or "").strip()
    if out_dir_raw:
        out_dir = Path(out_dir_raw).expanduser()
    else:
        out_dir = kinematic_workbench_root / "runs" / _safe_run_id(object_id)
    out_dir = out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    run_id = out_dir.name
    kinematic_workbench_runs[run_id] = out_dir
    orientation_degrees = _payload_orientation(payload.get("orientation_degrees"))
    (out_dir / "workbench_orientation.json").write_text(
        json.dumps(
            {
                "orientation_degrees": orientation_degrees,
                "note": (
                    "Workbench orientation is a viewer-only import aid. "
                    "Agent MJCF previews are already written in canonical +X-front, +Y-left, +Z-up coordinates."
                ),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    python_bin = os.environ.get(
        "KINEMATIC_SOLVER_PYTHON",
        "/home/mi/anaconda3/envs/env-isaacsim/bin/python",
    )
    converter_output_path = Path(converter_output_root).expanduser()
    source_root_path = Path(source_root).expanduser()
    partseg_bootstrap = _bootstrap_partseg_from_workbench_asset(
        object_id=object_id,
        converter_output_root=converter_output_path,
        workbench_root=kinematic_workbench_root,
        workbench_asset_name=payload.get("workbench_asset_name"),
    )
    if partseg_bootstrap["status"] == "skipped":
        partseg_bootstrap = _bootstrap_partseg_from_generated_mesh_dir(
            object_id=object_id,
            converter_output_root=converter_output_path,
            generated_mesh_dir=payload.get("generated_mesh_dir"),
        )
    _validate_workbench_agent_inputs(
        object_id=object_id,
        converter_output_root=converter_output_path,
        initial_joints_json=initial_joints_json,
    )
    vhacd_cache = _ensure_workbench_vhacd_cache(
        object_id=object_id,
        converter_output_root=converter_output_path,
        source_root=source_root_path,
        out_dir=out_dir,
        python_bin=python_bin,
    )
    cmd = [
        python_bin,
        "-m",
        "post_process.kinematic_solver.estimate_limit",
        "--object-id",
        object_id,
        "--converter-output-root",
        converter_output_root,
        "--source-root",
        source_root,
        "--out-dir",
        str(out_dir),
        "--agent-loop",
        "--max-agent-iterations",
        str(max_iterations),
        "--api-heartbeat-seconds",
        str(heartbeat_seconds),
        "--live-viewer",
        "--no-live-server",
    ]
    if initial_joints_json:
        cmd.extend(["--initial-joints-json", initial_joints_json])

    env = os.environ.copy()
    model = str(payload.get("model") or env.get("ARTICRAFT_MODEL") or "gpt-5.5").strip()
    thinking_level = str(
        payload.get("thinking_level") or env.get("ARTICRAFT_THINKING_LEVEL") or "medium"
    ).strip()
    base_url = str(payload.get("openrouter_base_url") or env.get("OPENROUTER_BASE_URL") or "").strip()
    api_key = str(
        payload.get("openrouter_api_key")
        or payload.get("api_key")
        or env.get("OPENROUTER_API_KEY")
        or env.get("OPENROUTER_API_KEYS")
        or ""
    ).strip()
    if model:
        env["ARTICRAFT_MODEL"] = model
    if thinking_level:
        env["ARTICRAFT_THINKING_LEVEL"] = thinking_level
    if base_url:
        env["OPENROUTER_BASE_URL"] = base_url
    if api_key:
        env["OPENROUTER_API_KEY"] = api_key

    project_root = Path(__file__).resolve().parents[2]
    log_path = out_dir / "agent_subprocess.log"
    with log_path.open("ab") as log_file:
        process = subprocess.Popen(
            cmd,
            cwd=str(project_root),
            env=env,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )

    api_key_source = "request" if payload.get("openrouter_api_key") or payload.get("api_key") else (
        "environment" if api_key else "missing"
    )
    return {
        "status": "ok",
        "run_id": run_id,
        "pid": process.pid,
        "out_dir": str(out_dir),
        "log_path": str(log_path),
        "viewer_url": f"/kinematic-agent/{run_id}",
        "state_url": f"/api/kinematic-agent/{run_id}/state",
        "manifest_url": f"/api/kinematic-agent/{run_id}/manifest",
        "api_key_source": api_key_source,
        "initial_joints_json": initial_joints_json,
        "partseg_bootstrap": partseg_bootstrap,
        "vhacd_cache": vhacd_cache,
    }


def _validate_workbench_agent_inputs(
    *,
    object_id: str,
    converter_output_root: Path,
    initial_joints_json: str,
) -> None:
    obj_dir = converter_output_root / "raw" / "partseg" / object_id / "objs"
    obj_files = sorted(obj_dir.glob("*.obj")) if obj_dir.is_dir() else []
    if not obj_files:
        available = _available_partseg_object_ids(converter_output_root)
        suffix = f" Available partseg object ids: {', '.join(available)}." if available else ""
        raise ValueError(
            f"No mesh OBJ files found for object_id={object_id} at {obj_dir}.{suffix} "
            "Load/save the asset for this Object ID first, or fix the Object ID / converter root mismatch."
        )

    if not initial_joints_json:
        return
    initial_path = Path(initial_joints_json).expanduser()
    if not initial_path.is_file():
        raise ValueError(f"Initial joints JSON does not exist: {initial_path}")
    try:
        payload = json.loads(initial_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Initial joints JSON must be valid JSON: {exc.msg}") from exc
    if not isinstance(payload, dict):
        raise ValueError("Initial joints JSON root must be an object")
    json_object_id = payload.get("object_id")
    if json_object_id is not None and str(json_object_id) != object_id:
        raise ValueError(
            f"initial JSON object_id={json_object_id!r} does not match Object ID {object_id!r}"
        )


def _ensure_workbench_vhacd_cache(
    *,
    object_id: str,
    converter_output_root: Path,
    source_root: Path,
    out_dir: Path,
    python_bin: str,
) -> dict[str, Any]:
    missing_before = _missing_vhacd_cache_files(object_id, converter_output_root)
    cache_dir = converter_output_root / "raw" / "vhacd" / object_id
    if not missing_before:
        return {
            "status": "ready",
            "cache_dir": str(cache_dir),
            "missing_before": [],
        }

    project_root = Path(__file__).resolve().parents[2]
    log_path = out_dir / "vhacd_cook.log"
    cmd = [
        python_bin,
        "-m",
        "post_process.kinematic_solver.utils.data_prep",
        "--stage",
        "vhacd",
        "--converter-output-root",
        str(converter_output_root),
        "--source-root",
        str(source_root),
        "--object-ids",
        object_id,
    ]
    result = subprocess.run(
        cmd,
        cwd=str(project_root),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    log_path.write_text(
        "\n".join(
            [
                "$ " + " ".join(cmd),
                "",
                "[stdout]",
                result.stdout or "",
                "",
                "[stderr]",
                result.stderr or "",
            ]
        ),
        encoding="utf-8",
    )
    if result.returncode != 0:
        tail = (result.stderr or result.stdout or "").strip().splitlines()[-5:]
        detail = "\n".join(tail) if tail else "no subprocess output"
        raise ValueError(
            "VHACD cache cook failed before starting agent. "
            f"object_id={object_id}, log={log_path}. Last output: {detail}"
        )

    missing_after = _missing_vhacd_cache_files(object_id, converter_output_root)
    if missing_after:
        raise ValueError(
            "VHACD cache cook finished but required cache files are still missing. "
            f"object_id={object_id}, missing={missing_after}, log={log_path}"
        )

    return {
        "status": "cooked",
        "cache_dir": str(cache_dir),
        "missing_before": missing_before,
        "log_path": str(log_path),
    }


def _missing_vhacd_cache_files(object_id: str, converter_output_root: Path) -> list[str]:
    obj_dir = converter_output_root / "raw" / "partseg" / object_id / "objs"
    cache_dir = converter_output_root / "raw" / "vhacd" / object_id
    expected = [
        f"{obj_path.stem}.json"
        for obj_path in sorted(obj_dir.glob("*.obj"))
        if _is_solver_obj_stem(obj_path.stem)
    ]
    return [name for name in expected if not (cache_dir / name).is_file()]


def _is_solver_obj_stem(stem: str) -> bool:
    return stem == "body" or (stem.startswith("part_") and stem.removeprefix("part_").isdigit())


def _build_initial_joints_template_response(
    payload: dict[str, Any],
    *,
    kinematic_workbench_root: Path,
) -> dict[str, Any]:
    object_id = _required_text(payload, "object_id")
    asset_name = str(payload.get("workbench_asset_name") or object_id).strip()
    converter_root_raw = str(payload.get("converter_output_root") or "").strip()

    part_names: list[str]
    parents: dict[str, str]
    if asset_name and (kinematic_workbench_root / "object_assets" / asset_name).is_dir():
        manifest = generate_manifest(asset_name, kinematic_workbench_root / "object_assets")
        if manifest.get("status") != "ok":
            raise ValueError(str(manifest.get("message") or "Failed to read current workbench asset"))
        body_records = [
            body for body in manifest.get("bodies", [])
            if isinstance(body, dict) and body.get("name")
        ]
        part_names = [
            str(body["name"])
            for body in body_records
            if str(body["name"]) != "body" and body.get("parent") not in {None, "", "world"}
        ]
        parents = {
            str(body["name"]): str(body.get("parent") or "body")
            for body in body_records
        }
    elif converter_root_raw:
        obj_dir = Path(converter_root_raw).expanduser() / "raw" / "partseg" / object_id / "objs"
        part_names = [
            path.stem for path in sorted(obj_dir.glob("*.obj"))
            if path.stem != "body" and _is_solver_obj_stem(path.stem)
        ]
        parents = {name: "body" for name in part_names}
    else:
        part_names = []
        parents = {}

    template = {
        "object_id": object_id,
        "initial_joints": {
            name: {
                "type": "revolute",
                "axis": [0, 0, 1],
                "limit": None,
                "parent": parents.get(name, "body"),
                "moving_parts": [name],
            }
            for name in sorted(part_names)
        },
    }
    return {
        "status": "ok",
        "object_id": object_id,
        "part_names": sorted(part_names),
        "json_text": json.dumps(template, indent=2, ensure_ascii=False),
    }


def _available_partseg_object_ids(converter_output_root: Path) -> list[str]:
    root = converter_output_root / "raw" / "partseg"
    if not root.is_dir():
        return []
    return sorted(
        child.name for child in root.iterdir()
        if child.is_dir() and (child / "objs").is_dir()
    )


def _bootstrap_partseg_from_workbench_asset(
    *,
    object_id: str,
    converter_output_root: Path,
    workbench_root: Path,
    workbench_asset_name: Any,
) -> dict[str, Any]:
    asset_name = str(workbench_asset_name or "").strip()
    if not asset_name:
        return {"status": "skipped", "reason": "no_workbench_asset_name"}

    asset_root = workbench_root / "object_assets"
    asset_dir = asset_root / asset_name
    if not asset_dir.is_dir():
        return {
            "status": "skipped",
            "reason": "workbench_asset_missing",
            "asset_name": asset_name,
        }

    manifest = generate_manifest(asset_name, asset_root)
    if manifest.get("status") != "ok":
        raise ValueError(str(manifest.get("message") or "Failed to read current workbench asset"))

    target_dir = converter_output_root / "raw" / "partseg" / object_id / "objs"
    tmp_dir = target_dir.parent / f".{target_dir.name}.tmp"
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    body_world = _manifest_body_world_transforms(manifest)
    written: list[str] = []
    for body in manifest.get("bodies", []):
        if not isinstance(body, dict) or not body.get("name"):
            continue
        visual_geoms = [
            geom for geom in body.get("visual_geoms", [])
            if isinstance(geom, dict) and geom.get("mesh_file")
        ]
        if not visual_geoms:
            continue
        body_name = _safe_local_name(str(body["name"]))
        body_transform = body_world.get(str(body["name"]), _identity_transform())
        _write_manifest_body_obj(
            visual_geoms,
            body_transform=body_transform,
            asset_dir=asset_dir,
            out_path=tmp_dir / f"{body_name}.obj",
        )
        written.append(f"{body_name}.obj")

    if not written:
        shutil.rmtree(tmp_dir)
        return {
            "status": "skipped",
            "reason": "workbench_asset_has_no_visual_mesh_bodies",
            "asset_name": asset_name,
        }

    if target_dir.exists():
        shutil.rmtree(target_dir)
    tmp_dir.replace(target_dir)
    vhacd_dir = converter_output_root / "raw" / "vhacd" / object_id
    if vhacd_dir.exists():
        shutil.rmtree(vhacd_dir)
    return {
        "status": "synced_from_workbench_asset",
        "asset_name": asset_name,
        "target_dir": str(target_dir),
        "copied_count": len(written),
        "files": sorted(written),
        "vhacd_cache_invalidated": True,
    }


def _bootstrap_partseg_from_generated_mesh_dir(
    *,
    object_id: str,
    converter_output_root: Path,
    generated_mesh_dir: Any,
) -> dict[str, Any]:
    target_dir = converter_output_root / "raw" / "partseg" / object_id / "objs"
    existing_objs = sorted(target_dir.glob("*.obj")) if target_dir.is_dir() else []
    if existing_objs:
        return {
            "status": "skipped",
            "reason": "partseg_exists",
            "target_dir": str(target_dir),
            "existing_count": len(existing_objs),
        }

    raw_mesh_dir = str(generated_mesh_dir or "").strip()
    if not raw_mesh_dir:
        return {
            "status": "skipped",
            "reason": "no_generated_mesh_dir",
            "target_dir": str(target_dir),
        }

    source_dir = Path(raw_mesh_dir).expanduser()
    if not source_dir.is_dir():
        return {
            "status": "skipped",
            "reason": "generated_mesh_dir_missing",
            "source_dir": str(source_dir),
            "target_dir": str(target_dir),
        }

    obj_files = [
        path for path in sorted(source_dir.glob("*.obj"))
        if path.stem == "body" or path.stem.startswith("part_")
    ]
    if not obj_files:
        return {
            "status": "skipped",
            "reason": "no_body_or_part_obj",
            "source_dir": str(source_dir),
            "target_dir": str(target_dir),
        }

    target_dir.mkdir(parents=True, exist_ok=True)
    for obj_file in obj_files:
        shutil.copyfile(obj_file, target_dir / obj_file.name)
    return {
        "status": "copied",
        "source_dir": str(source_dir),
        "target_dir": str(target_dir),
        "copied_count": len(obj_files),
    }


def _manifest_body_world_transforms(manifest: dict[str, Any]) -> dict[str, tuple[list[list[float]], list[float]]]:
    bodies = {
        str(body["name"]): body
        for body in manifest.get("bodies", [])
        if isinstance(body, dict) and body.get("name")
    }
    cache: dict[str, tuple[list[list[float]], list[float]]] = {}

    def compute(name: str) -> tuple[list[list[float]], list[float]]:
        if name in cache:
            return cache[name]
        body = bodies[name]
        local = (_quat_to_matrix(body.get("quat", [1, 0, 0, 0])), _vec3(body.get("pos", [0, 0, 0])))
        parent = str(body.get("parent") or "world")
        if parent in bodies:
            result = _compose_transform(compute(parent), local)
        else:
            result = local
        cache[name] = result
        return result

    for body_name in bodies:
        compute(body_name)
    return cache


def _write_manifest_body_obj(
    geoms: list[dict[str, Any]],
    *,
    body_transform: tuple[list[list[float]], list[float]],
    asset_dir: Path,
    out_path: Path,
) -> None:
    vertices_out: list[list[float]] = []
    faces_out: list[list[int]] = []
    for geom in geoms:
        mesh_path = _resolve_manifest_mesh_path(asset_dir, str(geom["mesh_file"]))
        vertices, faces = _read_obj_vertices_faces(mesh_path)
        offset = len(vertices_out)
        vertices_out.extend(vertices)
        faces_out.extend([[index + offset for index in face] for face in faces])

    out_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"v {v[0]:.12g} {v[1]:.12g} {v[2]:.12g}\n" for v in vertices_out]
    lines.extend("f " + " ".join(str(index + 1) for index in face) + "\n" for face in faces_out)
    out_path.write_text("".join(lines), encoding="utf-8")


def _resolve_manifest_mesh_path(asset_dir: Path, mesh_file: str) -> Path:
    candidates = [
        asset_dir / "mjcf" / mesh_file,
        asset_dir / "xml" / mesh_file,
        asset_dir / "mjcf" / "assets" / Path(mesh_file).name,
        asset_dir / "xml" / "assets" / Path(mesh_file).name,
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise ValueError(f"Manifest mesh file not found for partseg sync: {mesh_file}")


def _read_obj_vertices_faces(path: Path) -> tuple[list[list[float]], list[list[int]]]:
    vertices: list[list[float]] = []
    faces: list[list[int]] = []
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        if line.startswith("v "):
            parts = line.split()
            if len(parts) >= 4:
                vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
        elif line.startswith("f "):
            indices = []
            for token in line.split()[1:]:
                raw_index = token.split("/")[0]
                if not raw_index:
                    continue
                index = int(raw_index)
                if index < 0:
                    index = len(vertices) + index + 1
                indices.append(index - 1)
            if len(indices) >= 3:
                faces.append(indices)
    if not vertices or not faces:
        raise ValueError(f"OBJ mesh must contain vertices and faces: {path}")
    return vertices, faces


def _identity_transform() -> tuple[list[list[float]], list[float]]:
    return ([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], [0.0, 0.0, 0.0])


def _compose_transform(
    first: tuple[list[list[float]], list[float]],
    second: tuple[list[list[float]], list[float]],
) -> tuple[list[list[float]], list[float]]:
    r1, t1 = first
    r2, t2 = second
    return (
        _matmul3(r1, r2),
        _vec_add(_matvec3(r1, t2), t1),
    )


def _apply_transform(transform: tuple[list[list[float]], list[float]], point: list[float]) -> list[float]:
    rotation, translation = transform
    return _vec_add(_matvec3(rotation, point), translation)


def _quat_to_matrix(raw_quat: Any) -> list[list[float]]:
    q = list(raw_quat) if isinstance(raw_quat, (list, tuple)) else [1.0, 0.0, 0.0, 0.0]
    if len(q) != 4:
        q = [1.0, 0.0, 0.0, 0.0]
    w, x, y, z = (float(q[0]), float(q[1]), float(q[2]), float(q[3]))
    norm = math.sqrt(w * w + x * x + y * y + z * z)
    if norm <= 1e-12:
        return _identity_transform()[0]
    w, x, y, z = w / norm, x / norm, y / norm, z / norm
    return [
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ]


def _vec3(raw: Any) -> list[float]:
    if not isinstance(raw, (list, tuple)) or len(raw) != 3:
        return [0.0, 0.0, 0.0]
    return [float(raw[0]), float(raw[1]), float(raw[2])]


def _matmul3(a: list[list[float]], b: list[list[float]]) -> list[list[float]]:
    return [
        [sum(a[i][k] * b[k][j] for k in range(3)) for j in range(3)]
        for i in range(3)
    ]


def _matvec3(matrix: list[list[float]], vector: list[float]) -> list[float]:
    return [sum(matrix[i][k] * vector[k] for k in range(3)) for i in range(3)]


def _vec_add(a: list[float], b: list[float]) -> list[float]:
    return [a[0] + b[0], a[1] + b[1], a[2] + b[2]]


def _import_workbench_uploaded_asset(upload_request, kinematic_workbench_root: Path) -> dict[str, Any]:
    uploaded_files = upload_request.files.getlist("files")
    if not uploaded_files:
        uploaded_files = upload_request.files.getlist("file")
    uploaded_files = [item for item in uploaded_files if item and item.filename]
    if not uploaded_files:
        raise ValueError("No uploaded source file or folder files were received")

    object_id = str(upload_request.form.get("object_id") or "").strip()
    if not object_id:
        first_name = _safe_upload_relative_path(uploaded_files[0].filename).name
        object_id = Path(first_name).stem or "uploaded_asset"
    safe_object = _safe_local_name(object_id)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    upload_root = kinematic_workbench_root / "uploads" / f"{safe_object}_{stamp}"
    upload_root.mkdir(parents=True, exist_ok=True)

    saved_rel_paths: list[PurePosixPath] = []
    for file_storage in uploaded_files:
        rel_path = _safe_upload_relative_path(file_storage.filename)
        target = upload_root.joinpath(*rel_path.parts)
        target.parent.mkdir(parents=True, exist_ok=True)
        file_storage.save(target)
        saved_rel_paths.append(rel_path)

    requested_source = str(upload_request.form.get("source_relative_path") or "").strip()
    if requested_source:
        source_rel = _safe_upload_relative_path(requested_source)
    else:
        source_rel = _first_supported_source_path(saved_rel_paths)
    source_path = upload_root.joinpath(*source_rel.parts)
    if not source_path.is_file():
        raise ValueError(f"Uploaded source file not found in bundle: {source_rel}")

    payload = {
        "source_path": str(source_path),
        "object_id": object_id,
        "xml_save_root": str(upload_request.form.get("xml_save_root") or ""),
        "mesh_save_root": str(upload_request.form.get("mesh_save_root") or ""),
        "orientation_degrees": _form_orientation(upload_request.form),
    }
    result = import_asset_to_workbench(payload, kinematic_workbench_root)
    result["upload_root"] = str(upload_root)
    result["uploaded_source_path"] = str(source_path)
    result["uploaded_file_count"] = len(saved_rel_paths)
    return result


def _safe_upload_relative_path(raw_path: str) -> PurePosixPath:
    normalized = str(raw_path or "").replace("\\", "/").strip()
    if not normalized:
        raise ValueError("Uploaded file path is empty")
    rel_path = PurePosixPath(normalized)
    if rel_path.is_absolute():
        raise ValueError(f"Uploaded file path must be relative: {raw_path}")
    if any(part in {"", ".", ".."} for part in rel_path.parts):
        raise ValueError(f"Unsafe uploaded file path: {raw_path}")
    return rel_path


def _first_supported_source_path(paths: list[PurePosixPath]) -> PurePosixPath:
    for path in paths:
        if path.suffix.lower() in {".usd", ".usda", ".usdc", ".urdf", ".xml", ".mjcf"}:
            return path
    raise ValueError("Uploaded bundle does not contain a USD, URDF, or MJCF/XML source")


def _safe_local_name(raw: str) -> str:
    safe = "".join(ch if ch.isalnum() or ch == "_" else "_" for ch in str(raw)).strip("_")
    if not safe:
        return "uploaded_asset"
    return safe


def _form_orientation(form) -> dict[str, float]:
    raw_json = str(form.get("orientation_degrees") or "").strip()
    if raw_json:
        try:
            parsed = json.loads(raw_json)
        except json.JSONDecodeError as exc:
            raise ValueError("orientation_degrees must be valid JSON") from exc
        if not isinstance(parsed, dict):
            raise ValueError("orientation_degrees must be a JSON object")
        return parsed
    return {
        "roll": float(form.get("roll", 0.0) or 0.0),
        "pitch": float(form.get("pitch", 0.0) or 0.0),
        "yaw": float(form.get("yaw", 0.0) or 0.0),
    }


def _required_text(payload: dict[str, Any], key: str) -> str:
    value = str(payload.get(key) or "").strip()
    if not value:
        raise ValueError(f"Missing required field: {key}")
    return value


def _positive_int(value: Any, label: str) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be an integer") from exc
    if parsed <= 0:
        raise ValueError(f"{label} must be positive")
    return parsed


def _positive_float(value: Any, label: str) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be a number") from exc
    if parsed <= 0:
        raise ValueError(f"{label} must be positive")
    return parsed


def _safe_run_id(object_id: str) -> str:
    import datetime as _datetime

    safe_object = "".join(ch if ch.isalnum() or ch == "_" else "_" for ch in object_id).strip("_")
    if not safe_object:
        safe_object = "run"
    stamp = _datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{safe_object}_agent_run_{stamp}"


def _read_kinematic_frontend_state(run_dir: Path) -> dict[str, Any]:
    state_path = run_dir / "frontend_state.json"
    if state_path.is_file():
        try:
            state = json.loads(state_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            state = {"status": "error", "message": "Malformed frontend_state.json"}
    else:
        state = {}
    events_path = run_dir / "agent_events.jsonl"
    events: list[dict[str, Any]] = []
    if events_path.is_file():
        for line in events_path.read_text(encoding="utf-8").splitlines()[-200:]:
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    latest_event = events[-1] if events else None
    latest_preview = state.get("latest_preview")
    iterations = state.get("iterations", [])
    source_orientation = _read_run_orientation(run_dir)
    return {
        "status": "ok",
        "run_dir": str(run_dir),
        "latest_iteration": state.get("latest_iteration", 0),
        "latest_preview": latest_preview,
        "iterations": iterations,
        "events": events,
        "latest_event": latest_event,
        **({"source_orientation_degrees": source_orientation} if source_orientation is not None else {}),
    }


def _read_run_orientation(run_dir: Path) -> dict[str, float] | None:
    orientation_path = run_dir / "workbench_orientation.json"
    if not orientation_path.is_file():
        return None
    try:
        payload = json.loads(orientation_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, dict):
        return None
    try:
        return _payload_orientation(payload.get("orientation_degrees"))
    except ValueError:
        return None


def _payload_orientation(raw: Any) -> dict[str, float]:
    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise ValueError("orientation_degrees must be a JSON object")
    orientation: dict[str, float] = {}
    for axis in ("roll", "pitch", "yaw"):
        try:
            orientation[axis] = float(raw.get(axis, 0.0) or 0.0)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"orientation_degrees.{axis} must be a number") from exc
    return orientation
