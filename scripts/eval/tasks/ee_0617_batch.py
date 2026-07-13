#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path
from queue import Queue
from types import SimpleNamespace
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[3]
TRELLIS_ROOT = REPO_ROOT / "TRELLIS-arts"
for item in (str(REPO_ROOT), str(TRELLIS_ROOT)):
    if item not in sys.path:
        sys.path.insert(0, item)

from part_ss_eval_platform.eval_0617_1 import (  # noqa: E402
    DEFAULT_DATA_CONFIG,
    DEFAULT_SLAT_FLOW_CKPT,
    DEFAULT_SLAT_MESH_DECODER_CKPT,
    DEFAULT_SS_DECODER_CKPT,
    _load_datasets,
    _normalize_sample,
    _sample_part_stats,
    load_or_build_selection,
)
from scripts.eval.tasks.ee_0617_single import (  # noqa: E402
    DEFAULT_GAUSSIAN_DECODER,
    DEFAULT_OUT_DIR,
    DEFAULT_PART_SEG_CKPT,
    DEFAULT_SPLIT_JSON,
    DEFAULT_SS_FLOW_CKPT,
    GS_PRESET,
)


PYTHON = Path("/opt/venvs/arts-gen/bin/python")
DEFAULT_PACKED_INDEX = Path("/robot/data-lab/jzh/art-gen/data/part_promptable_seg_packed_v6/index.json")


def _safe_name(value: str, max_len: int = 80) -> str:
    out = "".join(ch if ch.isalnum() or ch in ("-", "_", ".") else "_" for ch in str(value)).strip("_")
    return (out or "item")[:max_len]


def _prefix(dataset_id: str, object_id: str, angle: int) -> str:
    return f"{dataset_id}__{_safe_name(object_id)}__angle_{int(angle):02d}"


def _sample_payload(sample: Any) -> dict[str, Any]:
    return {
        "split": str(getattr(sample, "split", "")),
        "dataset_id": str(getattr(sample, "dataset_id", "")),
        "object_id": str(getattr(sample, "obj_id", "")),
        "angle": int(getattr(sample, "angle_idx", 0)),
        "part_count": int(getattr(sample, "part_count", 0)),
        "priority_bucket": str(getattr(sample, "priority_bucket", "")),
        "sample_bucket": str(getattr(sample, "sample_bucket", "")),
        "data_root": str(getattr(sample, "data_root", "")),
        "manifest_path": str(getattr(sample, "manifest_path", "")),
    }


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _load_or_build_sample_selection(args: argparse.Namespace) -> list[Any]:
    selection_path = args.out_dir / "selection.json"
    if selection_path.is_file() and not args.overwrite_selection:
        data = json.loads(selection_path.read_text(encoding="utf-8"))
        rows = []
        for split in ("train", "held"):
            for item in data.get("samples", {}).get(split, []):
                rows.append(_normalize_sample({**item, "split": split}))
        if rows:
            return rows

    datasets, dataset_meta = _load_datasets(args)
    wanted: list[tuple[str, str, int]] = []
    allowed_datasets = {item.strip() for item in str(args.allowed_datasets).split(",") if item.strip()}
    if getattr(args, "packed_index", None):
        packed = json.loads(Path(args.packed_index).read_text(encoding="utf-8"))
        if args.sample_selection_unit == "objects":
            seen_keys = set()
            for entry in packed.get("entries", []):
                dataset_id = str(entry.get("dataset_id", ""))
                if allowed_datasets and dataset_id not in allowed_datasets:
                    continue
                obj_id = str(entry["obj_id"])
                angle_idx = int(entry["angle_idx"])
                key = (dataset_id, obj_id)
                if key in seen_keys:
                    continue
                seen_keys.add(key)
                wanted.append((dataset_id, obj_id, angle_idx))
                if len(wanted) >= int(args.limit):
                    break
        else:
            object_order: list[tuple[str, str]] = []
            angles_by_object: dict[tuple[str, str], list[int]] = {}
            seen_pairs = set()
            for entry in packed.get("entries", []):
                dataset_id = str(entry.get("dataset_id", ""))
                if allowed_datasets and dataset_id not in allowed_datasets:
                    continue
                obj_id = str(entry["obj_id"])
                angle_idx = int(entry["angle_idx"])
                object_key = (dataset_id, obj_id)
                pair_key = (dataset_id, obj_id, angle_idx)
                if pair_key in seen_pairs:
                    continue
                seen_pairs.add(pair_key)
                if object_key not in angles_by_object:
                    object_order.append(object_key)
                    angles_by_object[object_key] = []
                angles_by_object[object_key].append(angle_idx)
            max_angles = max((len(angles) for angles in angles_by_object.values()), default=0)
            for offset in range(max_angles):
                for object_key in object_order:
                    angles = angles_by_object[object_key]
                    if offset >= len(angles):
                        continue
                    wanted.append((object_key[0], object_key[1], angles[offset]))
                    if len(wanted) >= int(args.limit):
                        break
                if len(wanted) >= int(args.limit):
                    break
    wanted_set = set(wanted)
    wanted_order = {key: idx for idx, key in enumerate(wanted)}
    rows: list[Any] = []
    sample_idx = 0
    for dataset_id, ds in sorted(datasets.items()):
        for sample in ds.samples:
            if wanted_set and (dataset_id, str(sample["obj_id"]), int(sample["angle_idx"])) not in wanted_set:
                continue
            stats = _sample_part_stats(ds, sample)
            split = "train" if sample_idx < int(args.train_count) else "held"
            sample_idx += 1
            rows.append(
                _normalize_sample(
                    {
                        "split": split,
                        "dataset_id": dataset_id,
                        "obj_id": str(sample["obj_id"]),
                        "angle_idx": int(sample["angle_idx"]),
                        "data_root": str(sample.get("_eval_data_root") or ds.data_root),
                        "manifest_path": str(sample.get("_eval_manifest_path") or ds.manifest_path),
                        "sample_bucket": str(stats["sample_bucket"]),
                        "priority_bucket": str(stats["priority_bucket"]),
                        "part_count": int(stats["part_count"]),
                        "min_raw_voxels": int(stats["min_raw_voxels"]),
                        "max_raw_voxels": int(stats["max_raw_voxels"]),
                        "has_button": bool(stats["has_button"]),
                        "has_large_keyword": bool(stats["has_large_keyword"]),
                        "selected_reason": "sample_mode_first_n",
                    }
                )
            )
    unique_rows: list[Any] = []
    seen_keys = set()
    for item in rows:
        key = (
            (str(getattr(item, "dataset_id", "")), str(getattr(item, "obj_id", "")))
            if args.sample_selection_unit == "objects"
            else (
                str(getattr(item, "dataset_id", "")),
                str(getattr(item, "obj_id", "")),
                int(getattr(item, "angle_idx", 0)),
            )
        )
        if key in seen_keys:
            continue
        seen_keys.add(key)
        unique_rows.append(item)
    rows = unique_rows
    rows.sort(
        key=lambda item: (
            wanted_order.get(
                (
                    str(getattr(item, "dataset_id", "")),
                    str(getattr(item, "obj_id", "")),
                    int(getattr(item, "angle_idx", 0)),
                ),
                0,
            ),
            str(getattr(item, "dataset_id", "")),
            str(getattr(item, "obj_id", "")),
            int(getattr(item, "angle_idx", 0)),
        )
    )
    rows = rows[: int(args.limit)]
    by_split: dict[str, list[dict[str, Any]]] = {"train": [], "held": []}
    for idx, item in enumerate(rows):
        split = "train" if idx < int(args.train_count) else "held"
        item.split = split
        by_split[split].append(
            {
                "split": split,
                "dataset_id": str(getattr(item, "dataset_id", "")),
                "object_key": f"{getattr(item, 'dataset_id', '')}::{item.obj_id}",
                "obj_id": str(item.obj_id),
                "angle_idx": int(item.angle_idx),
                "data_root": str(getattr(item, "data_root", "")),
                "manifest_path": str(getattr(item, "manifest_path", "")),
                "bucket": str(getattr(item, "bucket", "")),
                "sample_bucket": str(getattr(item, "sample_bucket", "")),
                "priority_bucket": str(getattr(item, "priority_bucket", "")),
                "part_count": int(getattr(item, "part_count", 0)),
                "min_raw_voxels": int(getattr(item, "min_raw_voxels", 0)),
                "max_raw_voxels": int(getattr(item, "max_raw_voxels", 0)),
                "has_button": bool(getattr(item, "has_button", False)),
                "has_large_keyword": bool(getattr(item, "has_large_keyword", False)),
                "selected_reason": str(getattr(item, "selected_reason", "")),
                "original_split": str(getattr(item, "original_split", split)),
            }
        )
    _write_json(
        selection_path,
        {
            "name": "0625-128ee-sample-mode",
            "split_json": str(args.split_json),
            "selection_policy": (
                "first N unique objects from v5 packed index; first valid angle per object only"
                if args.sample_selection_unit == "objects"
                else "round-robin unique obj-angle pairs from v5 packed index; covers all eligible objects before reusing objects"
            ),
            "sample_selection_unit": str(args.sample_selection_unit),
            "datasets": dataset_meta,
            "samples": by_split,
        },
    )
    return rows


def _append_jsonl(path: Path, payload: Any, lock: threading.Lock) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with lock:
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def _outputs(out_dir: Path, sample: dict[str, Any]) -> tuple[Path, Path, Path, Path]:
    prefix = _prefix(sample["dataset_id"], sample["object_id"], int(sample["angle"]))
    return (
        out_dir / f"{prefix}__gaussian.png",
        out_dir / f"{prefix}__mesh.png",
        out_dir / f"{prefix}__summary.json",
        out_dir / f"{prefix}__mujoco" / f"{prefix}.xml",
    )


def _is_done(
    out_dir: Path,
    sample: dict[str, Any],
    slat_token_source: str,
    *,
    export_mujoco: bool,
    part_cc_filter: bool,
    part_cc_min_component_voxels: int,
    part_cc_min_component_fraction: float,
    part_cc_max_component_distance: int,
    part_cc_max_large_component_distance: int | None,
    part_t0_filter: bool,
    part_t0_disable_cc: bool,
    part_joint_candidate_mode: str,
    part_joint_refine: bool,
    part_joint_refine_iters: int,
    part_joint_refine_pairwise: float,
    part_joint_refine_margin: float,
    part_joint_refine_margin_quantile: float,
    part_joint_refine_neighborhood: int,
    part_joint_refine_min_vote_gain: float,
    part_joint_refine_preserve_small_classes: int,
    part_joint_save_logits: bool,
) -> bool:
    gaussian_png, mesh_png, summary_path, mujoco_xml = _outputs(out_dir, sample)
    if not (gaussian_png.is_file() and mesh_png.is_file() and summary_path.is_file()):
        return False
    if part_joint_save_logits:
        prefix = _prefix(sample["dataset_id"], sample["object_id"], int(sample["angle"]))
        if not (
            (out_dir / f"{prefix}__joint_boundary.png").is_file()
            and (out_dir / f"{prefix}__joint_boundary.json").is_file()
        ):
            return False
    if export_mujoco and not mujoco_xml.is_file():
        return False
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except Exception:
        return False
    slat_stage = summary.get("slat_stage", {})
    cond = slat_stage.get("condition", {})
    labels = {str(comp.get("label", "")) for comp in summary.get("components", []) or []}
    expected_token_source = "live_official_trellis_rgba" if slat_token_source == "live" else "cache"
    mujoco_rd_ok = True
    if export_mujoco:
        mujoco_rd_ok = bool((summary.get("mujoco_rd_postprocess") or {}).get("enabled", False))
    joint = summary.get("part_joint_partition") or {}
    joint_boundary = summary.get("joint_boundary_diagnostics") or {}
    cc_filter = summary.get("part_cc_filter") or {}
    cc_enabled = bool(part_cc_filter or (part_t0_filter and not part_t0_disable_cc))
    cc_config_matches = not cc_enabled or (
        int(cc_filter.get("min_component_voxels", 32)) == int(part_cc_min_component_voxels)
        and float(cc_filter.get("min_component_fraction", 0.05)) == float(part_cc_min_component_fraction)
        and int(cc_filter.get("max_component_distance", 2)) == int(part_cc_max_component_distance)
        and cc_filter.get("max_large_component_distance") == part_cc_max_large_component_distance
    )
    return (
        summary.get("status") == "done"
        and summary.get("ss_stage", {}).get("fusion_mode") == "concat"
        and slat_stage.get("flow_calls") == 1
        and cond.get("token_source") == expected_token_source
        and "body_without_parts" in labels
        and mujoco_rd_ok
        and bool(cc_filter.get("enabled", False)) == cc_enabled
        and cc_config_matches
        and bool((summary.get("part_t0_filter") or {}).get("enabled", False)) == bool(part_t0_filter)
        and bool(((summary.get("part_t0_filter") or {}).get("cc") or {}).get("enabled", False))
        == bool(part_t0_filter and not part_t0_disable_cc)
        and str(joint.get("candidate_mode", "proposal")) == str(part_joint_candidate_mode)
        and bool(joint.get("refine", False)) == bool(part_joint_refine)
        and int(joint.get("refine_iters", 1)) == int(part_joint_refine_iters)
        and float(joint.get("refine_pairwise", 3.0)) == float(part_joint_refine_pairwise)
        and float(joint.get("refine_margin", 0.0)) == float(part_joint_refine_margin)
        and float(joint.get("refine_margin_quantile", 0.01)) == float(part_joint_refine_margin_quantile)
        and int(joint.get("refine_neighborhood", 6)) == int(part_joint_refine_neighborhood)
        and float(joint.get("refine_min_vote_gain", 0.0)) == float(part_joint_refine_min_vote_gain)
        and int(joint.get("refine_preserve_small_classes", 32))
        == int(part_joint_refine_preserve_small_classes)
        and bool(joint.get("save_logits", False)) == bool(part_joint_save_logits)
        and (not part_joint_save_logits or joint_boundary.get("status") != "error")
    )


def _command(args: argparse.Namespace, sample: dict[str, Any], gpu: str) -> list[str]:
    cmd = [
        str(args.python),
        "scripts/eval/tasks/ee_0617_single.py",
        "--out-dir",
        str(args.out_dir),
        "--data-config",
        str(args.data_config),
        "--split-json",
        str(args.split_json),
        "--dataset-id",
        sample["dataset_id"],
        "--object-id",
        sample["object_id"],
        "--angle",
        str(int(sample["angle"])),
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
        "--gpu",
        str(gpu),
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
    if args.seed is not None:
        cmd += ["--seed", str(int(args.seed))]
    if args.part_cc_filter:
        cmd.append("--part-cc-filter")
    cc_enabled = bool(args.part_cc_filter or (args.part_t0_filter and not args.part_t0_disable_cc))
    if cc_enabled:
        cmd += [
            "--part-cc-min-component-voxels",
            str(int(args.part_cc_min_component_voxels)),
            "--part-cc-min-component-fraction",
            str(float(args.part_cc_min_component_fraction)),
            "--part-cc-max-component-distance",
            str(int(args.part_cc_max_component_distance)),
        ]
    if args.part_cc_max_large_component_distance is not None:
        cmd += [
            "--part-cc-max-large-component-distance",
            str(int(args.part_cc_max_large_component_distance)),
        ]
    cmd += ["--part-joint-candidate-mode", str(args.part_joint_candidate_mode)]
    cmd += ["--part-joint-refine-iters", str(int(args.part_joint_refine_iters))]
    cmd += ["--part-joint-refine-pairwise", str(float(args.part_joint_refine_pairwise))]
    cmd += ["--part-joint-refine-margin", str(float(args.part_joint_refine_margin))]
    cmd += ["--part-joint-refine-margin-quantile", str(float(args.part_joint_refine_margin_quantile))]
    cmd += ["--part-joint-refine-neighborhood", str(int(args.part_joint_refine_neighborhood))]
    cmd += ["--part-joint-refine-min-vote-gain", str(float(args.part_joint_refine_min_vote_gain))]
    cmd += [
        "--part-joint-refine-preserve-small-classes",
        str(int(args.part_joint_refine_preserve_small_classes)),
    ]
    cmd.append("--part-joint-refine" if args.part_joint_refine else "--no-part-joint-refine")
    cmd.append("--part-joint-save-logits" if args.part_joint_save_logits else "--no-part-joint-save-logits")
    if args.part_t0_filter:
        cmd.append("--part-t0-filter")
        cmd += [
            "--part-t0-part-threshold",
            str(float(args.part_t0_part_threshold)),
            "--part-t0-margin-threshold",
            str(float(args.part_t0_margin_threshold)),
            "--part-t0-smooth-iters",
            str(int(args.part_t0_smooth_iters)),
        ]
        if args.part_t0_disable_cc:
            cmd.append("--part-t0-disable-cc")
    if args.export_mujoco:
        cmd.append("--export-mujoco")
    if args.export_usd:
        cmd.append("--export-usd")
    if args.mujoco_textured_assets:
        cmd.append("--mujoco-textured-assets")
        cmd += [
            "--mujoco-appearance-source",
            str(args.mujoco_appearance_source),
            "--mujoco-texture-size",
            str(int(args.mujoco_texture_size)),
            "--mujoco-texture-render-resolution",
            str(int(args.mujoco_texture_render_resolution)),
            "--mujoco-texture-nviews",
            str(int(args.mujoco_texture_nviews)),
            "--mujoco-texture-mode",
            str(args.mujoco_texture_mode),
        ]
    if args.fill_hidden_vertex_colors:
        cmd.append("--fill-hidden-vertex-colors")
        if args.hidden_color_fill_out_dir is not None:
            cmd += ["--hidden-color-fill-out-dir", str(args.hidden_color_fill_out_dir)]
        cmd += ["--hidden-color-fill-dark-threshold", str(float(args.hidden_color_fill_dark_threshold))]
    if args.force_stage:
        cmd.append("--force-stage")
    if args.force_export:
        cmd.append("--force-export")
    return cmd


def _worker(
    worker_id: int,
    gpu: str,
    queue: Queue[tuple[int, dict[str, Any]]],
    args: argparse.Namespace,
    progress_path: Path,
    progress_lock: threading.Lock,
    counters: dict[str, int],
    counters_lock: threading.Lock,
) -> None:
    log_dir = args.out_dir / "_batch_logs"
    while True:
        try:
            index, sample = queue.get_nowait()
        except Exception:
            return
        started = time.time()
        gaussian_png, mesh_png, summary_path, mujoco_xml = _outputs(args.out_dir, sample)
        rec_base = {
            "idx": int(index),
            "total": int(args.limit),
            "worker": int(worker_id),
            "gpu": str(gpu),
            **sample,
            "gaussian_png": str(gaussian_png),
            "mesh_png": str(mesh_png),
            "mujoco_xml": str(mujoco_xml),
            "summary": str(summary_path),
        }
        try:
            if _is_done(
                args.out_dir,
                sample,
                str(args.slat_token_source),
                export_mujoco=bool(args.export_mujoco),
                part_cc_filter=bool(args.part_cc_filter),
                part_cc_min_component_voxels=int(args.part_cc_min_component_voxels),
                part_cc_min_component_fraction=float(args.part_cc_min_component_fraction),
                part_cc_max_component_distance=int(args.part_cc_max_component_distance),
                part_cc_max_large_component_distance=args.part_cc_max_large_component_distance,
                part_t0_filter=bool(args.part_t0_filter),
                part_t0_disable_cc=bool(args.part_t0_disable_cc),
                part_joint_candidate_mode=str(args.part_joint_candidate_mode),
                part_joint_refine=bool(args.part_joint_refine),
                part_joint_refine_iters=int(args.part_joint_refine_iters),
                part_joint_refine_pairwise=float(args.part_joint_refine_pairwise),
                part_joint_refine_margin=float(args.part_joint_refine_margin),
                part_joint_refine_margin_quantile=float(args.part_joint_refine_margin_quantile),
                part_joint_refine_neighborhood=int(args.part_joint_refine_neighborhood),
                part_joint_refine_min_vote_gain=float(args.part_joint_refine_min_vote_gain),
                part_joint_refine_preserve_small_classes=int(args.part_joint_refine_preserve_small_classes),
                part_joint_save_logits=bool(args.part_joint_save_logits),
            ) and not args.force:
                rec = {**rec_base, "status": "skipped", "seconds": 0.0}
                _append_jsonl(progress_path, rec, progress_lock)
                with counters_lock:
                    counters["skipped"] += 1
                print(
                    f"[0617-128ee-batch] {index}/{args.limit} skip "
                    f"{sample['dataset_id']}::{sample['object_id']} angle={sample['angle']}",
                    flush=True,
                )
                continue

            _append_jsonl(progress_path, {**rec_base, "status": "started"}, progress_lock)
            print(
                f"[0617-128ee-batch] {index}/{args.limit} start gpu={gpu} "
                f"{sample['dataset_id']}::{sample['object_id']} angle={sample['angle']} parts={sample['part_count']}",
                flush=True,
            )
            log_path = log_dir / (
                f"{index:03d}__{sample['dataset_id']}__{_safe_name(sample['object_id'])}"
                f"__angle_{int(sample['angle']):02d}__gpu{gpu}.log"
            )
            log_path.parent.mkdir(parents=True, exist_ok=True)
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(gpu)
            env["SS_FLOW_FUSION_MODE"] = "concat"
            with log_path.open("a", encoding="utf-8") as log:
                log.write("\n[0617-128ee-batch] cmd=" + " ".join(_command(args, sample, gpu)) + "\n")
                log.flush()
                proc = subprocess.run(
                    _command(args, sample, gpu),
                    cwd=REPO_ROOT,
                    env=env,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    text=True,
                )
            seconds = round(time.time() - started, 3)
            if proc.returncode != 0:
                raise RuntimeError(f"returncode={proc.returncode} log={log_path}")
            rec = {
                **rec_base,
                "status": "done",
                "seconds": seconds,
                "log": str(log_path),
            }
            try:
                summary = json.loads(summary_path.read_text(encoding="utf-8"))
                rec["component_count"] = int(summary.get("component_count", 0))
                rec["slat_flow_calls"] = int(summary.get("slat_stage", {}).get("flow_calls", -1))
            except Exception:
                pass
            _append_jsonl(progress_path, rec, progress_lock)
            with counters_lock:
                counters["done"] += 1
            print(
                f"[0617-128ee-batch] {index}/{args.limit} done seconds={seconds} "
                f"{sample['dataset_id']}::{sample['object_id']}",
                flush=True,
            )
        except Exception as exc:
            seconds = round(time.time() - started, 3)
            rec = {
                **rec_base,
                "status": "failed",
                "seconds": seconds,
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
            _append_jsonl(progress_path, rec, progress_lock)
            with counters_lock:
                counters["failed"] += 1
            print(
                f"[0617-128ee-batch] {index}/{args.limit} failed {type(exc).__name__}: {exc}",
                flush=True,
            )
        finally:
            queue.task_done()


def _require_file(path: Path, label: str) -> Path:
    path = Path(path).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"{label} not found: {path}")
    return path


def _checkpoint_model_meta(path: Path) -> dict[str, Any]:
    try:
        import torch

        ckpt = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}
    args = ckpt.get("args") if isinstance(ckpt, dict) else None
    args = args if isinstance(args, dict) else {}
    return {
        "step": ckpt.get("step") if isinstance(ckpt, dict) else None,
        "dim": args.get("dim"),
        "depth": args.get("depth"),
        "route": args.get("route"),
        "joint_seg": bool(args.get("joint_seg", False)),
        "mask_encoder": args.get("mask_encoder"),
        "voxel_depth": args.get("voxel_depth"),
        "spconv_depth": args.get("spconv_depth"),
        "voxel_embedding_dim": args.get("voxel_embedding_dim"),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run 0617-128ee flat EE eval from images/tokens with 4-GPU workers.")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--data-config", type=Path, default=DEFAULT_DATA_CONFIG)
    parser.add_argument("--split-json", type=Path, default=DEFAULT_SPLIT_JSON)
    parser.add_argument("--part-seg-ckpt", type=Path, default=DEFAULT_PART_SEG_CKPT)
    parser.add_argument("--ss-flow-ckpt", type=Path, default=DEFAULT_SS_FLOW_CKPT)
    parser.add_argument("--ss-decoder-ckpt", type=Path, default=DEFAULT_SS_DECODER_CKPT)
    parser.add_argument("--slat-flow-ckpt", type=Path, default=DEFAULT_SLAT_FLOW_CKPT)
    parser.add_argument("--slat-mesh-decoder-ckpt", type=Path, default=DEFAULT_SLAT_MESH_DECODER_CKPT)
    parser.add_argument("--slat-gaussian-decoder-ckpt", type=Path, default=DEFAULT_GAUSSIAN_DECODER)
    parser.add_argument("--python", type=Path, default=PYTHON)
    parser.add_argument("--limit", type=int, default=128)
    parser.add_argument("--train-count", type=int, default=85)
    parser.add_argument("--held-count", type=int, default=43)
    parser.add_argument("--gpus", default="0,1,2,3")
    parser.add_argument("--allowed-datasets", default="phyx-verse,realappliance")
    parser.add_argument("--selection-mode", choices=("objects", "samples"), default="objects")
    parser.add_argument("--sample-selection-unit", choices=("objects", "pairs"), default="objects")
    parser.add_argument("--packed-index", type=Path, default=DEFAULT_PACKED_INDEX)
    parser.add_argument("--slat-steps", type=int, default=25)
    parser.add_argument("--slat-seed", type=int, default=42)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--render-view", type=int, default=0)
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--tile-size", type=int, default=240)
    parser.add_argument("--panel-cols", type=int, default=4)
    parser.add_argument("--export-mujoco", action="store_true")
    parser.add_argument("--export-usd", action="store_true")
    parser.add_argument("--mujoco-textured-assets", action="store_true")
    parser.add_argument(
        "--mujoco-appearance-source",
        choices=("obj-vertex-color", "mesh-vertex-texture", "gaussian-texture"),
        default="obj-vertex-color",
    )
    parser.add_argument("--mujoco-texture-size", type=int, default=512)
    parser.add_argument("--mujoco-texture-render-resolution", type=int, default=512)
    parser.add_argument("--mujoco-texture-nviews", type=int, default=30)
    parser.add_argument("--mujoco-texture-mode", choices=("fast", "opt"), default="fast")
    parser.add_argument("--fill-hidden-vertex-colors", action="store_true")
    parser.add_argument("--hidden-color-fill-out-dir", type=Path, default=None)
    parser.add_argument("--hidden-color-fill-dark-threshold", type=float, default=0.18)
    parser.add_argument(
        "--slat-token-source",
        choices=("live", "cache"),
        default="live",
        help="SLat flow condition source. Default live matches the accepted TRELLIS RGBA preprocessing path.",
    )
    parser.add_argument(
        "--part-cc-filter",
        action="store_true",
        help="Post-process predicted part voxels by moving small remote connected components back to body.",
    )
    parser.add_argument("--part-cc-min-component-voxels", type=int, default=32)
    parser.add_argument("--part-cc-min-component-fraction", type=float, default=0.05)
    parser.add_argument("--part-cc-max-component-distance", type=int, default=2)
    parser.add_argument("--part-cc-max-large-component-distance", type=int, default=None)
    parser.add_argument("--part-joint-candidate-mode", choices=("proposal", "full_occ"), default="proposal")
    parser.add_argument("--part-joint-refine", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--part-joint-refine-iters", type=int, default=1)
    parser.add_argument("--part-joint-refine-pairwise", type=float, default=3.0)
    parser.add_argument("--part-joint-refine-margin", type=float, default=0.0)
    parser.add_argument("--part-joint-refine-margin-quantile", type=float, default=0.01)
    parser.add_argument("--part-joint-refine-neighborhood", type=int, choices=(6, 18, 26), default=6)
    parser.add_argument("--part-joint-refine-min-vote-gain", type=float, default=0.0)
    parser.add_argument("--part-joint-refine-preserve-small-classes", type=int, default=32)
    parser.add_argument("--part-joint-save-logits", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--part-t0-filter",
        action="store_true",
        help="Enable T0 joint argmax + competition-band smoothing + CC part boundary postprocess.",
    )
    parser.add_argument("--part-t0-part-threshold", type=float, default=0.5)
    parser.add_argument("--part-t0-margin-threshold", type=float, default=0.35)
    parser.add_argument("--part-t0-smooth-iters", type=int, default=1)
    parser.add_argument("--part-t0-disable-cc", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--force-stage", action="store_true")
    parser.add_argument("--force-export", action="store_true")
    parser.add_argument("--overwrite-selection", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.out_dir = Path(args.out_dir).resolve()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for attr, label in (
        ("data_config", "data config"),
        ("split_json", "split json"),
        ("part_seg_ckpt", "part-seg ckpt"),
        ("ss_flow_ckpt", "SS-flow ckpt"),
        ("ss_decoder_ckpt", "SS decoder ckpt"),
        ("slat_flow_ckpt", "SLat flow ckpt"),
        ("slat_mesh_decoder_ckpt", "SLat mesh decoder ckpt"),
        ("slat_gaussian_decoder_ckpt", "SLat gaussian decoder ckpt"),
        ("python", "python executable"),
    ):
        setattr(args, attr, _require_file(Path(getattr(args, attr)), label))

    selection_args = SimpleNamespace(
        out_dir=str(args.out_dir),
        data_config=str(args.data_config),
        split_json=str(args.split_json),
        train_count=int(args.train_count),
        held_count=int(args.held_count),
        overwrite_selection=bool(args.overwrite_selection),
    )
    if args.selection_mode == "samples":
        samples = [_sample_payload(sample) for sample in _load_or_build_sample_selection(args)]
    else:
        samples = [_sample_payload(sample) for sample in load_or_build_selection(selection_args)]
    allowed = {item.strip() for item in str(args.allowed_datasets).split(",") if item.strip()}
    bad_datasets = sorted({sample["dataset_id"] for sample in samples if sample["dataset_id"] not in allowed})
    if bad_datasets:
        raise RuntimeError(f"selection contains disallowed dataset(s): {bad_datasets}; allowed={sorted(allowed)}")
    if len(samples) < int(args.limit):
        raise RuntimeError(f"selection has {len(samples)} samples, requested {args.limit}")
    samples = samples[: int(args.limit)]
    object_keys = [(sample["dataset_id"], sample["object_id"]) for sample in samples]
    pair_keys = [(sample["dataset_id"], sample["object_id"], int(sample["angle"])) for sample in samples]
    if len(set(pair_keys)) != len(pair_keys):
        duplicates = sorted({key for key in pair_keys if pair_keys.count(key) > 1})
        raise RuntimeError(f"selection contains repeated obj-angle pairs: {duplicates[:10]}")
    if args.sample_selection_unit == "objects" and len(set(object_keys)) != len(object_keys):
        duplicates = sorted({key for key in object_keys if object_keys.count(key) > 1})
        raise RuntimeError(f"selection contains repeated objects across angles: {duplicates[:10]}")
    args.limit = len(samples)
    gpu_ids = [item.strip() for item in str(args.gpus).split(",") if item.strip()]
    if not gpu_ids:
        raise ValueError("--gpus must contain at least one GPU id")

    run_config = {
        "status": "dry_run" if args.dry_run else "running",
        "out_dir": str(args.out_dir),
        "limit": int(args.limit),
        "gpus": gpu_ids,
        "allowed_datasets": sorted(allowed),
        "selection_mode": str(args.selection_mode),
        "sample_selection_unit": str(args.sample_selection_unit),
        "split_json": str(args.split_json),
        "data_config": str(args.data_config),
        "ss_stage": {
            "source": "input 4-view DINO tokens for SS stage",
            "fusion_mode": "concat",
            "ckpt": str(args.ss_flow_ckpt),
        },
        "part_stage": {
            "backend": "promptable_seg",
            "ckpt": str(args.part_seg_ckpt),
            "ckpt_model_meta": _checkpoint_model_meta(args.part_seg_ckpt),
        },
        "part_cc_filter": {
            "enabled": bool(args.part_cc_filter or (args.part_t0_filter and not args.part_t0_disable_cc)),
            "min_component_voxels": int(args.part_cc_min_component_voxels),
            "min_component_fraction": float(args.part_cc_min_component_fraction),
            "max_component_distance": int(args.part_cc_max_component_distance),
            "max_large_component_distance": args.part_cc_max_large_component_distance,
        },
        "part_t0_filter": {
            "enabled": bool(args.part_t0_filter),
            "part_threshold": float(args.part_t0_part_threshold),
            "margin_threshold": float(args.part_t0_margin_threshold),
            "smooth_iters": int(args.part_t0_smooth_iters),
            "includes_cc_filter": bool(args.part_t0_filter and not args.part_t0_disable_cc),
        },
        "part_joint_partition": {
            "candidate_mode": str(args.part_joint_candidate_mode),
            "refine": bool(args.part_joint_refine),
            "refine_iters": int(args.part_joint_refine_iters),
            "refine_pairwise": float(args.part_joint_refine_pairwise),
            "refine_margin": float(args.part_joint_refine_margin),
            "refine_margin_quantile": float(args.part_joint_refine_margin_quantile),
            "refine_neighborhood": int(args.part_joint_refine_neighborhood),
            "refine_min_vote_gain": float(args.part_joint_refine_min_vote_gain),
            "refine_preserve_small_classes": int(args.part_joint_refine_preserve_small_classes),
            "save_logits": bool(args.part_joint_save_logits),
        },
        "slat_stage": {
            "rule": "one whole-object SLat flow per object, parts sliced by voxel coords",
            "condition_source": str(args.slat_token_source),
            "condition_contract": (
                "live TRELLIS RGBA alpha crop + black premultiply + DINO x_prenorm layer_norm"
                if args.slat_token_source == "live"
                else "cached dataset tokens; diagnostic only for the accepted EE path"
            ),
            "flow_ckpt": str(args.slat_flow_ckpt),
            "steps": int(args.slat_steps),
            "seed": int(args.slat_seed),
        },
        "setup_seed": None if args.seed is None else int(args.seed),
        "outputs": (
            "flat directory: each object has one *_gaussian.png, one *_mesh.png, one *_summary.json"
            + (
                ", and one R-D MuJoCo *_mujoco/*.xml with root Z-forward quat, local +Z drawer slide, "
                "actuator/keyframe, no floor, no shell inertia, and optional drawer_glass mesh"
                if args.export_mujoco
                else ""
            )
            + (", and one *_usd/*.usda Isaac Sim scene with decoded vertex colors" if args.export_usd else "")
        ),
        "usd_export": {
            "enabled": bool(args.export_usd),
            "source": "decoded mesh vertices/faces/vertex_attrs",
        },
        "mujoco_textured_assets": {
            "enabled": bool(args.mujoco_textured_assets),
            "appearance_source": str(args.mujoco_appearance_source),
            "texture_size": int(args.mujoco_texture_size),
            "render_resolution": int(args.mujoco_texture_render_resolution),
            "nviews": int(args.mujoco_texture_nviews),
            "mode": str(args.mujoco_texture_mode),
        },
        "gs_preset": GS_PRESET,
        "selection": samples,
    }
    _write_json(args.out_dir / "run_config.json", run_config)
    if args.dry_run:
        print(f"[0617-128ee-batch] dry-run selected={len(samples)} out={args.out_dir}", flush=True)
        return 0

    queue: Queue[tuple[int, dict[str, Any]]] = Queue()
    for idx, sample in enumerate(samples, start=1):
        queue.put((idx, sample))
    progress_path = args.out_dir / "progress_batch.jsonl"
    progress_lock = threading.Lock()
    counters_lock = threading.Lock()
    counters = {"done": 0, "failed": 0, "skipped": 0}
    started = time.time()
    threads = [
        threading.Thread(
            target=_worker,
            args=(idx, gpu, queue, args, progress_path, progress_lock, counters, counters_lock),
            daemon=False,
        )
        for idx, gpu in enumerate(gpu_ids)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    run_config.update(
        {
            "status": "done" if counters["failed"] == 0 else "done_with_failures",
            "done": int(counters["done"]),
            "skipped": int(counters["skipped"]),
            "failed": int(counters["failed"]),
            "seconds": round(time.time() - started, 3),
        }
    )
    _write_json(args.out_dir / "run_config.json", run_config)
    print(
        f"[0617-128ee-batch] finished done={counters['done']} "
        f"skipped={counters['skipped']} failed={counters['failed']} out={args.out_dir}",
        flush=True,
    )
    return 0 if counters["failed"] == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
