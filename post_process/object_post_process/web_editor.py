#!/usr/bin/env python3
"""CLI: launch the object post-process Flask web service."""

from __future__ import annotations

import argparse
import os
import sys
import threading
import webbrowser
from pathlib import Path


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Launch object post-process web service")
    parser.add_argument(
        "--asset",
        default=None,
        help="Asset name under assets/object_assets/ (optional, opens asset picker if omitted)",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--no-browser", action="store_true", help="Don't auto-open browser")
    parser.add_argument(
        "--kinematic-run-dir",
        type=Path,
        default=None,
        help="Load an existing kinematic run directory containing frontend_state.json and object_assets/.",
    )
    parser.add_argument(
        "--kinematic-run-id",
        default=None,
        help="URL id for --kinematic-run-dir. Defaults to the run directory name.",
    )
    parser.add_argument(
        "--kinematic-workbench",
        action="store_true",
        help="Open the standalone KinematicSolver workbench.",
    )
    parser.add_argument(
        "--kinematic-workspace-root",
        type=Path,
        default=Path("kin_test/kinematic_workbench"),
        help="Workspace root for workbench imports, saved XML/meshes, and launched agent runs.",
    )
    return parser


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    gen_obj_root = script_dir.parent
    assets_root = gen_obj_root / "assets" / "object_assets"
    sys.path.insert(0, str(gen_obj_root))

    if args.asset:
        asset_dir = assets_root / args.asset
        if not asset_dir.is_dir():
            print(f"Error: asset directory not found: {asset_dir}", flush=True)
            return 1
        if not (asset_dir / "mjcf").is_dir():
            print(f"Error: mjcf directory not found: {asset_dir / 'mjcf'}", flush=True)
            return 1

    if args.kinematic_run_dir is not None:
        if not args.kinematic_run_dir.is_dir():
            print(f"Error: kinematic run directory not found: {args.kinematic_run_dir}", flush=True)
            return 1
        if not (args.kinematic_run_dir / "frontend_state.json").is_file():
            print(
                f"Error: frontend_state.json not found: {args.kinematic_run_dir / 'frontend_state.json'}",
                flush=True,
            )
            return 1
        if not (args.kinematic_run_dir / "object_assets").is_dir():
            print(
                f"Error: object_assets directory not found: {args.kinematic_run_dir / 'object_assets'}",
                flush=True,
            )
            return 1

    isaac_port_raw = os.environ.get("ISAAC_PORT", "8081")
    try:
        isaac_port = int(isaac_port_raw)
    except ValueError:
        print(f"Error: ISAAC_PORT must be an integer, got: {isaac_port_raw}", flush=True)
        return 1

    try:
        from object_post_process.server import create_app

        app = create_app(
            assets_root=assets_root,
            default_asset=args.asset,
            kinematic_run_dir=args.kinematic_run_dir,
            kinematic_run_id=args.kinematic_run_id,
            kinematic_workbench_root=(
                args.kinematic_workspace_root if args.kinematic_workbench else None
            ),
        )
    except Exception as exc:
        print(f"Error: failed to launch object post-process server: {exc}", flush=True)
        return 1

    if args.kinematic_workbench:
        url = f"http://{args.host}:{args.port}/kinematic-workbench"
    elif args.kinematic_run_dir is not None:
        run_id = args.kinematic_run_id or args.kinematic_run_dir.name
        url = f"http://{args.host}:{args.port}/kinematic-agent/{run_id}"
    elif args.asset:
        url = f"http://{args.host}:{args.port}/object-post-process/{args.asset}"
    else:
        url = f"http://{args.host}:{args.port}/object-post-process/"

    if not args.no_browser:
        threading.Timer(1.0, lambda: webbrowser.open(url)).start()

    print(f"Starting object post-process server at {url}", flush=True)
    app.run(host=args.host, port=args.port, debug=False, use_reloader=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
