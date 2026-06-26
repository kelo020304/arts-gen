from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path


def _query_gpus() -> list[dict[str, int]]:
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,memory.used,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
        timeout=5,
    )
    if result.returncode != 0:
        raise RuntimeError((result.stderr or result.stdout).strip())
    rows: list[dict[str, int]] = []
    for line in result.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 3:
            continue
        rows.append({"index": int(parts[0]), "used_mib": int(parts[1]), "util": int(parts[2])})
    return rows


def _append_log(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")


def _pick_gpu(rows: list[dict[str, int]], candidates: set[int], max_used_mib: int, max_util: int) -> int | None:
    for row in rows:
        if int(row["index"]) not in candidates:
            continue
        if int(row["used_mib"]) <= int(max_used_mib) and int(row["util"]) <= int(max_util):
            return int(row["index"])
    return None


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Wait until a GPU is idle, then run an eval command.")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--gpu-candidates", default="0,1")
    p.add_argument("--max-used-mib", type=int, default=8000)
    p.add_argument("--max-util", type=int, default=20)
    p.add_argument("--poll-sec", type=float, default=120.0)
    p.add_argument("--log", default="")
    p.add_argument("cmd", nargs=argparse.REMAINDER)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if args.cmd and args.cmd[0] == "--":
        args.cmd = args.cmd[1:]
    if not args.cmd:
        raise ValueError("missing command after --")
    out_dir = Path(args.out_dir)
    log_path = Path(args.log) if args.log else out_dir / "watcher.log"
    candidates = {int(x) for x in str(args.gpu_candidates).split(",") if x.strip()}
    _append_log(
        log_path,
        {
            "event": "watcher_start",
            "time": time.time(),
            "gpu_candidates": sorted(candidates),
            "max_used_mib": int(args.max_used_mib),
            "max_util": int(args.max_util),
            "poll_sec": float(args.poll_sec),
            "cmd_template": args.cmd,
        },
    )
    while True:
        try:
            rows = _query_gpus()
            picked = _pick_gpu(rows, candidates, int(args.max_used_mib), int(args.max_util))
        except Exception as exc:
            rows = []
            picked = None
            _append_log(log_path, {"event": "query_error", "time": time.time(), "error": f"{type(exc).__name__}: {exc}"})
        if picked is not None:
            cmd = [part.format(gpu=picked) for part in args.cmd]
            _append_log(log_path, {"event": "launch", "time": time.time(), "gpu": picked, "gpus": rows, "cmd": cmd})
            out_dir.mkdir(parents=True, exist_ok=True)
            with (out_dir / "run.log").open("a", encoding="utf-8") as log:
                log.write(f"\n[wait_for_free_gpu] launch gpu={picked} cmd={' '.join(cmd)}\n")
                log.flush()
                proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)
                code = proc.wait()
            _append_log(log_path, {"event": "finished", "time": time.time(), "gpu": picked, "returncode": int(code)})
            return int(code)
        _append_log(log_path, {"event": "wait", "time": time.time(), "gpus": rows})
        time.sleep(float(args.poll_sec))


if __name__ == "__main__":
    raise SystemExit(main())
