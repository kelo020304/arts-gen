#!/usr/bin/env python3
"""Articulator dev server.

Two responsibilities:
1. Static file serving from this directory (replaces ``python3 -m http.server``).
2. ``POST /api/preprocess?eps=...&min_verts=...`` — upload a raw GLB,
   run ``preprocess_glb.py`` (DBSCAN cluster split), return JSON with
   the resulting clean GLB path + log output. The clean GLB is written
   to ``./data/preprocessed.glb`` so the browser can fetch it via the
   existing static-file path.

Usage::

    bash scripts/tools/articulator/serve.sh
    # or directly:
    python scripts/tools/articulator/serve.py
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path
from urllib.parse import parse_qs, urlparse

ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parents[2]
PREPROCESS_SCRIPT = ROOT / "preprocess_glb.py"
BUILD_USD_SCRIPT = ROOT / "build_usd.py"
PREPROCESS_TIMEOUT_SEC = 600
BUILD_USD_TIMEOUT_SEC = 600
MAX_UPLOAD_BYTES = 500 * 1024 * 1024   # 500 MB hard cap


def _resolve(p: str) -> Path:
    """Resolve a user-supplied path. Absolute -> as-is; relative -> against
    the repo root (so 'outputs/x/y.glb' Just Works regardless of CWD)."""
    pp = Path(p).expanduser()
    return pp if pp.is_absolute() else (PROJECT_ROOT / pp).resolve()


class Handler(SimpleHTTPRequestHandler):
    def do_POST(self):  # noqa: N802 — http.server convention
        if self.path.startswith("/api/preprocess"):
            handler = self._handle_preprocess
        elif self.path.startswith("/api/build_usd"):
            handler = self._handle_build_usd
        else:
            self.send_error(404)
            return
        try:
            handler()
        except Exception as e:
            self._json_response({"ok": False, "log": f"[server-error] {e}"}, status=500)

    def _handle_preprocess(self):
        qs = parse_qs(urlparse(self.path).query)
        try:
            eps = float(qs.get("eps", ["0.10"])[0])
            min_verts = int(qs.get("min_verts", ["200"])[0])
        except ValueError as e:
            self._json_response({"ok": False, "log": f"[server-error] bad params: {e}", "preprocessed_path": None}, status=400)
            return

        size = int(self.headers.get("Content-Length", 0))
        if size <= 0 or size > MAX_UPLOAD_BYTES:
            self._json_response({"ok": False, "log": f"[server-error] bad upload size {size}", "preprocessed_path": None}, status=400)
            return

        body = self.rfile.read(size)
        with tempfile.TemporaryDirectory() as td:
            in_path = Path(td) / "input.glb"
            in_path.write_bytes(body)
            data_dir = ROOT / "data"
            data_dir.mkdir(parents=True, exist_ok=True)
            out_path = data_dir / "preprocessed.glb"

            cmd = [
                sys.executable, str(PREPROCESS_SCRIPT),
                "--in", str(in_path),
                "--out", str(out_path),
                "--eps", f"{eps}",
                "--min_verts", f"{min_verts}",
            ]
            print(f"[preprocess] {' '.join(cmd)}", flush=True)
            try:
                proc = subprocess.run(cmd, capture_output=True, text=True, timeout=PREPROCESS_TIMEOUT_SEC)
                log_text = (proc.stdout or "") + ("\n[stderr]\n" + proc.stderr if proc.stderr else "")
                ok = proc.returncode == 0 and out_path.exists()
            except subprocess.TimeoutExpired:
                ok = False
                log_text = f"[error] preprocess timed out (>{PREPROCESS_TIMEOUT_SEC}s)"

        self._json_response({
            "ok": ok,
            "log": log_text[-8000:],   # keep responses small
            "preprocessed_path": "./data/preprocessed.glb" if ok else None,
            "eps": eps,
            "min_verts": min_verts,
        })

    def _handle_build_usd(self):
        """POST /api/build_usd, body = JSON
              {labels: <full v2 schema>, clean_glb_path: "...", out_path: "..."}.
        Writes labels to a tmp file, runs build_usd.py, returns log + ok."""
        size = int(self.headers.get("Content-Length", 0))
        if size <= 0 or size > MAX_UPLOAD_BYTES:
            self._json_response({"ok": False, "log": f"[server-error] bad body size {size}"}, status=400)
            return
        body = json.loads(self.rfile.read(size))
        labels = body.get("labels")
        if not isinstance(labels, dict):
            self._json_response({"ok": False, "log": "[server-error] 'labels' missing"}, status=400)
            return
        clean_glb = _resolve(body.get("clean_glb_path", ""))
        out_path = _resolve(body.get("out_path", ""))
        if not clean_glb.exists():
            self._json_response({"ok": False, "log": f"[server-error] clean_glb not found: {clean_glb}"}, status=400)
            return
        out_path.parent.mkdir(parents=True, exist_ok=True)

        with tempfile.NamedTemporaryFile(mode="w", suffix=".labels.json", delete=False, dir=str(ROOT)) as f:
            json.dump(labels, f, indent=2)
            labels_tmp = Path(f.name)

        # Trace what the server actually received — helps debug
        # "I did box-split but the export ignores it" complaints.
        n_parts = len(labels.get("parts", []))
        n_joints = len(labels.get("joints", []))
        splits = list((labels.get("split_clusters") or {}).keys())
        print(f"[build_usd] received: {n_parts} parts, {n_joints} joints, "
              f"{len(splits)} splits {splits}", flush=True)

        cmd = [
            sys.executable, str(BUILD_USD_SCRIPT),
            "--clean_glb", str(clean_glb),
            "--labels", str(labels_tmp),
            "--out", str(out_path),
        ]
        texture_glb = body.get("texture_glb_path")
        if texture_glb:
            cmd.extend(["--texture_glb", str(_resolve(texture_glb))])
        print(f"[build_usd] {' '.join(cmd)}", flush=True)
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=BUILD_USD_TIMEOUT_SEC)
            log_text = (proc.stdout or "") + ("\n[stderr]\n" + proc.stderr if proc.stderr else "")
            ok = proc.returncode == 0 and out_path.exists()
        except subprocess.TimeoutExpired:
            ok = False
            log_text = f"[error] build_usd timed out (>{BUILD_USD_TIMEOUT_SEC}s)"
        finally:
            labels_tmp.unlink(missing_ok=True)

        self._json_response({
            "ok": ok,
            "log": log_text[-8000:],
            "out_path": str(out_path) if ok else None,
            "size_kb": round(out_path.stat().st_size / 1024, 1) if ok else None,
        })

    def _json_response(self, obj: dict, status: int = 200):
        body = json.dumps(obj).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    # Quieter access log
    def log_message(self, fmt: str, *args):
        if "/api/" in self.path or self.command == "POST":
            super().log_message(fmt, *args)


def main():
    if not PREPROCESS_SCRIPT.exists():
        print(f"[warning] {PREPROCESS_SCRIPT} not found — /api/preprocess will fail", file=sys.stderr)

    port = int(os.environ.get("PORT", 8000))
    os.chdir(ROOT)
    addr = ("127.0.0.1", port)
    print(f"[articulator] serving http://{addr[0]}:{addr[1]}/  (cwd: {ROOT})")
    HTTPServer(addr, Handler).serve_forever()


if __name__ == "__main__":
    main()
