#!/usr/bin/env python3
"""Local Isaac Sim USD export service."""

from __future__ import annotations

import argparse
import atexit
import json
import os
import queue
import signal
import sys
import threading
import time
from pathlib import Path
from typing import Any

from flask import Flask, jsonify, request

_PENDING_STAGE_QUEUE: queue.Queue[str] = queue.Queue()


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Serve persistent Isaac Sim MJCF -> USD exports over localhost HTTP.")
    parser.add_argument("--host", default="127.0.0.1", help="Bind host. Must remain localhost.")
    parser.add_argument("--port", type=int, required=True, help="Bind port for the Isaac export service.")
    return parser


def _build_error_result(asset_name: str | None, assets_root: Path | None, message: str) -> dict[str, Any]:
    asset = asset_name or ""
    usd_path = ""
    xml_path = ""
    output_path = ""
    if asset:
        xml_path = f"mjcf/{asset}.xml"
        output_path = f"usd/{asset}.usd"
        if assets_root is not None:
            usd_path = str((assets_root / asset / "usd" / f"{asset}.usd").resolve())
    return {
        "status": "error",
        "asset": asset,
        "xml_path": xml_path,
        "output_path": output_path,
        "usd_path": usd_path,
        "message": message,
    }


def _parse_export_request(payload: Any) -> tuple[str, Path]:
    if not isinstance(payload, dict):
        raise ValueError("Request body must be a JSON object.")

    asset_name = payload.get("asset_name")
    assets_root = payload.get("assets_root")

    if not isinstance(asset_name, str) or not asset_name.strip():
        raise ValueError("`asset_name` must be a non-empty string.")
    if not isinstance(assets_root, str) or not assets_root.strip():
        raise ValueError("`assets_root` must be a non-empty string.")

    return asset_name.strip(), Path(assets_root).expanduser().resolve()


def _create_app(session) -> Flask:
    app = Flask(__name__)
    export_lock = threading.Lock()

    @app.get("/health")
    def health():
        return jsonify({"status": "ok", "ready": True}), 200

    @app.post("/export")
    def export():
        if not request.is_json:
            return jsonify(_build_error_result(None, None, "Content-Type must be application/json.")), 400

        payload = request.get_json(silent=True)
        try:
            asset_name, assets_root = _parse_export_request(payload)
        except ValueError as exc:
            return jsonify(_build_error_result(None, None, str(exc))), 400

        if not export_lock.acquire(blocking=False):
            return (
                jsonify(_build_error_result(asset_name, assets_root, "Isaac Sim USD export is already in progress.")),
                409,
            )

        try:
            result = session.export_asset_to_usd(asset_name, assets_root)
            if result.get("status") != "ok":
                return jsonify(result), 500

            _PENDING_STAGE_QUEUE.put(str(Path(result["usd_path"]).expanduser().resolve()))
            return jsonify(result), 200
        except Exception as exc:
            return jsonify(_build_error_result(asset_name, assets_root, str(exc))), 500
        finally:
            export_lock.release()

    return app


def main() -> int:
    args = _build_parser().parse_args()
    if args.host != "127.0.0.1":
        print("Error: isaac_export_service.py must bind to 127.0.0.1 only.", file=sys.stderr, flush=True)
        return 1

    script_dir = Path(__file__).resolve().parent
    gen_obj_root = script_dir.parent
    sys.path.insert(0, str(gen_obj_root))

    try:
        from utils.usd_exporter import IsaacUsdExportSession
    except Exception as exc:
        print(f"Error: failed to import persistent USD exporter: {exc}", file=sys.stderr, flush=True)
        return 1

    try:
        session = IsaacUsdExportSession(headless=False)
    except Exception as exc:
        print(f"Error: failed to initialize Isaac Sim export session: {exc}", file=sys.stderr, flush=True)
        return 1

    simulation_app = session.simulation_app
    cleanup_state = {"closed": False}

    def cleanup() -> None:
        if cleanup_state["closed"]:
            return
        cleanup_state["closed"] = True
        session.close()

    def _handle_shutdown(signum, frame) -> None:
        cleanup()
        raise SystemExit(0)

    atexit.register(cleanup)
    signal.signal(signal.SIGINT, _handle_shutdown)
    signal.signal(signal.SIGTERM, _handle_shutdown)

    app = _create_app(session)
    flask_thread = threading.Thread(
        target=lambda: app.run(host=args.host, port=args.port, debug=False, use_reloader=False),
        daemon=True,
    )
    flask_thread.start()
    print(f"[isaac-export-service] ready on http://{args.host}:{args.port}", flush=True)
    try:
        while simulation_app.is_running():
            simulation_app.update()
            try:
                usd_path = _PENDING_STAGE_QUEUE.get_nowait()
            except queue.Empty:
                continue

            try:
                import omni.usd

                usd_context = omni.usd.get_context()
                usd_context.open_stage(str(usd_path))
                print(f"[isaac-export-service] opened stage: {usd_path}", flush=True)
            except Exception as exc:
                print(f"[isaac-export-service] failed to open stage {usd_path}: {exc}", file=sys.stderr, flush=True)
    finally:
        cleanup()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
