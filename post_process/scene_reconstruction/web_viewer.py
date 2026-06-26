#!/usr/bin/env python3
"""CLI: launch the standalone 3DGS scene viewer Flask service."""

from __future__ import annotations

import argparse
import sys
import threading
import webbrowser
from pathlib import Path


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Launch scene reconstruction 3DGS viewer")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8083)
    parser.add_argument("--no-browser", action="store_true", help="Don't auto-open browser")
    return parser


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    gen_obj_root = script_dir.parent
    scene_assets_root = gen_obj_root / "assets" / "scene_assets"
    sys.path.insert(0, str(gen_obj_root))

    if not scene_assets_root.is_dir():
        print(f"Error: scene assets directory not found: {scene_assets_root}", flush=True)
        return 1

    try:
        from scene_reconstruction.server import create_app

        app = create_app(scene_assets_root=scene_assets_root)
    except Exception as exc:
        print(f"Error: failed to launch scene reconstruction server: {exc}", flush=True)
        return 1

    listen_url = f"http://{args.host}:{args.port}/scenes/"
    browser_host = "127.0.0.1" if args.host in {"127.0.0.1", "0.0.0.0"} else args.host
    browser_url = f"http://{browser_host}:{args.port}/scenes/"

    if not args.no_browser:
        threading.Timer(1.0, lambda: webbrowser.open(browser_url)).start()

    print(f"Starting scene reconstruction server at {listen_url}", flush=True)
    app.run(host=args.host, port=args.port, debug=False, use_reloader=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
