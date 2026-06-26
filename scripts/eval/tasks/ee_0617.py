#!/usr/bin/env python3
from __future__ import annotations

import json
import math
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[3]
EE_BATCH = REPO_ROOT / "scripts/eval/tasks/ee_0617_batch.py"


def _safe_float(value: Any, default: float = float("nan")) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _read_json(path: Path, default: Any = None) -> Any:
    if not path.is_file():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _fix_mount_path(value: str) -> str:
    return str(value).replace("/robot/data-lab", "/mnt/robot-data-lab")


def _resolve_manifest(data_root: str, value: str) -> str:
    path = Path(_fix_mount_path(value))
    if path.is_absolute():
        return str(path)
    return str(Path(data_root) / path)


def _materialize_split_from_packed_index(args: Any) -> Path:
    packed_index = Path(args.packed_index)
    packed = _read_json(packed_index, {})
    if not isinstance(packed, dict):
        raise ValueError(f"{packed_index}: expected JSON object")
    allowed = {item.strip() for item in str(args.allowed_datasets).split(",") if item.strip()}
    datasets = []
    for item in packed.get("datasets", []):
        dataset_id = str(item.get("dataset_id") or "")
        if dataset_id not in allowed:
            continue
        data_root = _fix_mount_path(str(item["data_root"]))
        raw_manifest = item.get("manifest_paths", item.get("manifest_path"))
        manifest_values = raw_manifest if isinstance(raw_manifest, list) else [raw_manifest]
        manifest_paths = [_resolve_manifest(data_root, str(path)) for path in manifest_values if path]
        missing = [path for path in manifest_paths if not Path(path).is_file()]
        if missing:
            raise FileNotFoundError(f"{dataset_id} missing manifest(s): {missing[:5]}")
        datasets.append({"dataset_id": dataset_id, "data_root": data_root, "manifest_paths": manifest_paths})
    if not datasets:
        raise RuntimeError(f"{packed_index}: no datasets matched allowed={sorted(allowed)}")
    out_path = Path(args.out_dir) / "_inputs" / "split_from_packed_index.json"
    _write_json(
        out_path,
        {
            "name": "run_eval_ee_0617_auto_split_from_packed_index",
            "packed_index": str(packed_index),
            "source_split_json": packed.get("split_json"),
            "datasets": datasets,
            "train_ids": [],
            "heldout_ids": [],
            "selection_policy": "dataset spec only; samples are selected deterministically from packed index",
        },
    )
    return out_path


def _ensure_split_json(args: Any) -> None:
    split_json = Path(args.split_json)
    if split_json.is_file():
        return
    if str(args.selection_mode) != "samples":
        raise FileNotFoundError(f"split json not found for selection-mode={args.selection_mode}: {split_json}")
    args.split_json = _materialize_split_from_packed_index(args)


def _read_progress(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.is_file():
        return rows
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            rows.append({"status": "malformed", "raw": line})
    return rows


def _command(args: Any) -> list[str]:
    cmd = [
        str(args.python),
        str(EE_BATCH.relative_to(REPO_ROOT)),
        "--out-dir",
        str(args.out_dir),
        "--data-config",
        str(args.data_config),
        "--split-json",
        str(args.split_json),
        "--part-seg-ckpt",
        str(args.part_seg_ckpt),
        "--ss-flow-ckpt",
        str(args.ss_flow_ckpt),
        "--ss-decoder-ckpt",
        str(args.ss_decoder_ckpt),
        "--slat-flow-ckpt",
        str(args.slat_flow_ckpt),
        "--slat-mesh-decoder-ckpt",
        str(args.slat_mesh_decoder_ckpt),
        "--slat-gaussian-decoder-ckpt",
        str(args.slat_gaussian_decoder_ckpt),
        "--limit",
        str(int(args.limit)),
        "--train-count",
        str(int(args.train_count)),
        "--held-count",
        str(int(args.held_count)),
        "--gpus",
        str(args.gpus),
        "--allowed-datasets",
        str(args.allowed_datasets),
        "--selection-mode",
        str(args.selection_mode),
        "--sample-selection-unit",
        str(args.sample_selection_unit),
        "--packed-index",
        str(args.packed_index),
        "--slat-steps",
        str(int(args.slat_steps)),
        "--slat-seed",
        str(int(args.slat_seed)),
        "--render-view",
        str(int(args.render_view)),
        "--resolution",
        str(int(args.resolution)),
        "--tile-size",
        str(int(args.tile_size)),
        "--panel-cols",
        str(int(args.panel_cols)),
        "--slat-token-source",
        str(args.slat_token_source),
    ]
    if args.export_mujoco:
        cmd.append("--export-mujoco")
    if args.force:
        cmd.append("--force")
    if args.force_stage:
        cmd.append("--force-stage")
    if args.force_export:
        cmd.append("--force-export")
    if args.overwrite_selection:
        cmd.append("--overwrite-selection")
    if args.dry_run:
        cmd.append("--dry-run")
    return cmd


def _summarize_outputs(args: Any, *, returncode: int, seconds: float, command: list[str]) -> dict[str, Any]:
    out_dir = Path(args.out_dir)
    progress = _read_progress(out_dir / "progress_batch.jsonl")
    run_config = _read_json(out_dir / "run_config.json", {}) or {}
    selection = _read_json(out_dir / "selection.json", {}) or {}
    summary_paths = sorted(out_dir.glob("*__summary.json"))
    summaries = []
    for path in summary_paths:
        payload = _read_json(path, {})
        if isinstance(payload, dict):
            summaries.append(payload)

    failed = [row for row in progress if row.get("status") == "failed"]
    done_rows = [row for row in progress if row.get("status") == "done"]
    skipped_rows = [row for row in progress if row.get("status") == "skipped"]
    flow_calls = [
        int(summary.get("slat_stage", {}).get("flow_calls", -1))
        for summary in summaries
        if isinstance(summary.get("slat_stage"), dict)
    ]
    components = [int(summary.get("component_count", 0)) for summary in summaries]
    gaussians_after: list[float] = []
    mesh_vertices: list[float] = []
    for summary in summaries:
        for comp in summary.get("components", []) or []:
            stats = comp.get("gs_preset") or {}
            gaussians_after.append(_safe_float(stats.get("gaussians_after")))
            mesh_vertices.append(_safe_float(comp.get("mesh_vertices")))
    gaussians_after = [v for v in gaussians_after if math.isfinite(v)]
    mesh_vertices = [v for v in mesh_vertices if math.isfinite(v)]
    selected_total = sum(len(selection.get("samples", {}).get(split, [])) for split in ("train", "held"))

    metrics = {
        "schema": "arts-gen.eval.ee_0617.v1",
        "status": "passed" if returncode == 0 and not failed else "failed",
        "out_dir": str(out_dir),
        "backend": "scripts/eval/tasks/ee_0617_batch.py",
        "command": command,
        "seconds": round(float(seconds), 3),
        "requested_limit": int(args.limit),
        "selected_total": int(selected_total),
        "summary_count": len(summaries),
        "done": len(done_rows),
        "skipped": len(skipped_rows),
        "failed": len(failed),
        "returncode": int(returncode),
        "run_config": {
            "status": run_config.get("status"),
            "done": run_config.get("done"),
            "skipped": run_config.get("skipped"),
            "failed": run_config.get("failed"),
            "gpus": run_config.get("gpus"),
            "selection_mode": run_config.get("selection_mode"),
            "sample_selection_unit": run_config.get("sample_selection_unit"),
        },
        "pipeline_contract": {
            "ss_flow_fusion_mode": "concat",
            "part_backend": "promptable_seg",
            "slat_token_source": str(args.slat_token_source),
            "slat_flow_calls_per_object": "one whole-object call; per-part SLat sliced by voxel coords",
            "gs_preset": run_config.get("gs_preset", {}),
        },
        "artifact_counts": {
            "gaussian_png": len(list(out_dir.glob("*__gaussian.png"))),
            "mesh_png": len(list(out_dir.glob("*__mesh.png"))),
            "diagnostic_png": len(list(out_dir.glob("*__diagnostic.png"))),
            "summary_json": len(summaries),
        },
        "key_fields": {
            "objects_complete": len(summaries),
            "slat_flow_calls_min": min(flow_calls) if flow_calls else None,
            "slat_flow_calls_max": max(flow_calls) if flow_calls else None,
            "components_mean": (sum(components) / len(components)) if components else None,
            "gaussians_after_mean": (sum(gaussians_after) / len(gaussians_after)) if gaussians_after else None,
            "mesh_vertices_mean": (sum(mesh_vertices) / len(mesh_vertices)) if mesh_vertices else None,
        },
        "failed_objects": failed,
    }
    _write_json(out_dir / "metrics.json", metrics)
    _write_json(out_dir / "failed_objects.json", failed)
    return metrics


def run(args: Any) -> int:
    if not EE_BATCH.is_file():
        raise FileNotFoundError(f"0617 EE runner is missing: {EE_BATCH}")
    args.out_dir = Path(args.out_dir).resolve()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    _ensure_split_json(args)
    cmd = _command(args)
    started = time.time()
    proc = subprocess.run(cmd, cwd=REPO_ROOT, text=True)
    metrics = _summarize_outputs(args, returncode=proc.returncode, seconds=time.time() - started, command=cmd)
    print(json.dumps(metrics, ensure_ascii=False, indent=2), flush=True)
    return int(proc.returncode) if proc.returncode != 0 else (0 if metrics["status"] == "passed" else 2)
