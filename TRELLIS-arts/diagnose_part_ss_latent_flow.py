#!/usr/bin/env python3
"""Diagnostics for Part SS Latent Flow identity/completeness failures.

This standalone entry answers the four evidence questions that gate the fix
plan (run it on the box that has the dataset, i.e. the remote /robot/data-lab
machine). It writes one JSON report and prints a human summary.

Modes (``--mode``):

  ceiling     Model-free SS-DECODER CEILING. Decode each GT part latent
              (x_1_parts) with the frozen SS decoder at several thresholds and
              compare to the raw GT voxel occupancy (raw_ind). Tells you, per
              part-size bucket, whether the decoder can even reproduce a part
              from a PERFECT latent. No checkpoint needed.
                -> small-bucket recall@0 low  => decoder is the bottleneck (F7)
                -> recall jumps at thr<0       => threshold calibration (F4)

  pred        FLOW-vs-CEILING head to head (needs --ckpt). Sample the flow
              model, decode BOTH pred and GT latents at the same thresholds,
              report pred recall/precision/iou next to the ceiling on the SAME
              parts, plus per-object assignment diag/off-diag IoU (identity
              slot-swap), bucketed by part size.
                -> pred recall << ceiling recall => flow problem (F5/F3)
                -> off-diag high only in small bucket => identity masked by
                   aggregate object IoU (confirms the small-part story)

  ckpt-scale  Load a checkpoint and assert it embeds config.flow.latent_scale
              (and num_steps). Production inference.py reads these from the
              ckpt and SILENTLY falls back to latent_scale=1.0 if absent, an 8x
              decode-input scale break. One-line safety gate.

  all         ceiling + ckpt-scale + (pred if --ckpt given).

Examples:
  # Dataset-wide decoder ceiling (no model), 200 objects:
  python TRELLIS-arts/diagnose_part_ss_latent_flow.py \\
      --config TRELLIS-arts/configs/arts/part_ss_latent_flow/part_ss_latent_flow.yaml \\
      --mode ceiling --max-samples 200 --output /tmp/part_ss_ceiling.json

  # Flow-vs-ceiling on the two failing objects:
  python TRELLIS-arts/diagnose_part_ss_latent_flow.py \\
      --config TRELLIS-arts/configs/arts/part_ss_latent_flow/part_ss_latent_flow.yaml \\
      --ckpt .../ckpts/step_50000.pt --mode pred \\
      --object-ids 100283,101049,101106 --output /tmp/part_ss_pred.json
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List

import train_arts  # noqa: F401  # Registers lightweight trellis package stubs.
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, Subset

from eval_part_ss_latent_flow import _apply_ckpt_latent_norm, _cfg_dict, _checkpoint_path, _dataset_cls, _load_model
from trellis.trainers.arts.part_ss_latent_flow import (
    _object_id_filter_indices,
    _sample_indices_for_eval,
    _setup_rng,
    _to_device,
)
from trellis.trainers.arts.part_ss_latent_flow_eval import (
    coords_iou,
    load_ss_decoder,
    part_assignment_iou_matrix,
    summarize_assignment_matrix,
)
from trellis.trainers.arts.part_ss_latent_flow_losses import (
    build_part_ss_sampler_kwargs,
    sample_part_ss_latent,
)
from trellis.utils.arts.config_utils import load_config


def _bucket_name(count: float, boundaries: tuple[float, float]) -> str:
    small_hi, medium_hi = boundaries
    if count < small_hi:
        return "small"
    if count < medium_hi:
        return "medium"
    return "large"


@torch.no_grad()
def _decode_logits_volume(decoder, z: torch.Tensor) -> torch.Tensor:
    """Decode a single [8,16,16,16] latent into a [64,64,64] logit volume."""
    if tuple(z.shape) != (8, 16, 16, 16):
        raise ValueError(f"latent must be (8,16,16,16), got {tuple(z.shape)}")
    device = next(decoder.parameters()).device
    dtype = next(decoder.parameters()).dtype
    logits = decoder(z.unsqueeze(0).to(device=device, dtype=dtype))
    if logits.dim() != 5 or logits.shape[1] < 1:
        raise ValueError(f"SS decoder logits must be [N,C,D,H,W], got {tuple(logits.shape)}")
    if tuple(logits.shape[-3:]) != (64, 64, 64):
        raise ValueError(f"SS decoder logits spatial shape must be 64^3, got {tuple(logits.shape[-3:])}")
    return logits[0, 0].float().cpu()


def _metrics_at_thresholds(
    logit_volume: torch.Tensor,
    raw_coords: torch.Tensor,
    thresholds: tuple[float, ...],
) -> Dict[str, Dict[str, float]]:
    """recall/precision/iou of decode(latent)>thr vs raw GT coords, per threshold."""
    out: Dict[str, Dict[str, float]] = {}
    for thr in thresholds:
        coords = torch.nonzero(logit_volume > float(thr), as_tuple=False).long()
        m = coords_iou(coords, raw_coords)
        out[f"{thr:g}"] = {
            "recall": float(m["recall"]),
            "precision": float(m["precision"]),
            "iou": float(m["iou"]),
            "pred_count": int(m["pred_count"]),
        }
    return out


def _aggregate(rows: List[Dict[str, Any]], thresholds: tuple[float, ...]) -> Dict[str, Any]:
    """Mean recall/precision/iou per threshold, overall and per size bucket."""
    def _group(group_rows: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not group_rows:
            return {"parts": 0}
        agg: Dict[str, Any] = {"parts": len(group_rows)}
        for thr in thresholds:
            key = f"{thr:g}"
            agg[key] = {
                metric: float(
                    sum(r["thresholds"][key][metric] for r in group_rows) / len(group_rows)
                )
                for metric in ("recall", "precision", "iou", "pred_count")
            }
        # ceiling/pred extra fields if present
        for extra in ("ceiling_recall_0", "pred_recall_0", "raw_count"):
            vals = [r[extra] for r in group_rows if extra in r]
            if vals:
                agg[extra] = float(sum(vals) / len(vals))
        return agg

    buckets = {"small": [], "medium": [], "large": []}
    for r in rows:
        buckets[r["size_bucket"]].append(r)
    return {
        "overall": _group(rows),
        "by_size": {name: _group(group) for name, group in buckets.items()},
    }


def _run_ceiling(
    dataset,
    decoder,
    sample_indices: List[int],
    thresholds: tuple[float, ...],
    boundaries: tuple[float, float],
) -> Dict[str, Any]:
    rows: List[Dict[str, Any]] = []
    skipped_empty = 0
    total = len(sample_indices)
    for n, idx in enumerate(sample_indices, start=1):
        item = dataset[idx]
        obj_id = item["obj_id"]
        x_1 = item["x_1_parts"]  # [K,8,16,16,16]
        raw_list = item["raw_ind_coords"]  # list[K] of [N,3]
        names = item["target_part_names"]
        for k in range(int(x_1.shape[0])):
            raw = raw_list[k]
            raw_count = int(raw.shape[0])
            if raw_count == 0:
                skipped_empty += 1
                continue
            vol = _decode_logits_volume(decoder, x_1[k])
            thr_metrics = _metrics_at_thresholds(vol, raw, thresholds)
            rows.append({
                "obj_id": obj_id,
                "part_name": names[k],
                "raw_count": raw_count,
                "size_bucket": _bucket_name(raw_count, boundaries),
                "thresholds": thr_metrics,
                "ceiling_recall_0": thr_metrics[f"{thresholds[0]:g}"]["recall"],
            })
        if n == 1 or n == total or n % 25 == 0:
            print(f"[INFO][ceiling] {n}/{total} objects | parts so far={len(rows)}", flush=True)
    print(f"[INFO][ceiling] skipped {skipped_empty} parts with zero raw voxels", flush=True)
    return {"parts": len(rows), "skipped_empty": skipped_empty, "summary": _aggregate(rows, thresholds), "rows": rows}


def _run_pred(
    dataset,
    decoder,
    model,
    device,
    flow_cfg: Dict[str, Any],
    sample_indices: List[int],
    thresholds: tuple[float, ...],
    boundaries: tuple[float, float],
) -> Dict[str, Any]:
    loader = DataLoader(
        Subset(dataset, sample_indices),
        batch_size=1, shuffle=False, num_workers=0, collate_fn=dataset.collate_fn,
    )
    primary_thr = float(thresholds[0])
    part_rows: List[Dict[str, Any]] = []
    object_rows: List[Dict[str, Any]] = []
    total = len(sample_indices)
    for n, batch in enumerate(loader, start=1):
        batch = _to_device(batch, device)
        obj_id = batch["obj_id"][0]
        valid_k = int(batch["part_valid"][0].sum().item())
        pred = sample_part_ss_latent(
            model,
            z_global=batch["z_global"],
            cond=batch["cond"],
            mask_token_labels=batch["mask_token_labels"],
            part_valid=batch["part_valid"],
            target_slots=batch["target_slots"],
            part_token_weights=batch.get("part_token_weights"),
            num_steps=int(flow_cfg.get("num_steps", 20)),
            noise_scale=float(flow_cfg.get("noise_scale", 1.0)),
            latent_scale=float(flow_cfg.get("latent_scale", 1.0)),
            **build_part_ss_sampler_kwargs(model, flow_cfg),
        )
        x_1 = batch["x_1_parts"]
        names = batch["target_part_names"][0]
        pred_coords_list = []
        raw_coords_list = []
        for k in range(valid_k):
            raw = batch["raw_ind_coords"][0][k].detach().cpu()
            raw_count = int(raw.shape[0])
            pred_vol = _decode_logits_volume(decoder, pred[0, k].detach().float().cpu())
            gt_vol = _decode_logits_volume(decoder, x_1[0, k].detach().float().cpu())
            pred_coords_primary = torch.nonzero(pred_vol > primary_thr, as_tuple=False).long()
            pred_coords_list.append(pred_coords_primary)
            raw_coords_list.append(raw)
            if raw_count == 0:
                continue
            pred_metrics = _metrics_at_thresholds(pred_vol, raw, thresholds)
            ceil_metrics = _metrics_at_thresholds(gt_vol, raw, thresholds)
            part_rows.append({
                "obj_id": obj_id,
                "part_name": names[k],
                "raw_count": raw_count,
                "size_bucket": _bucket_name(raw_count, boundaries),
                "thresholds": pred_metrics,
                "pred_recall_0": pred_metrics[f"{primary_thr:g}"]["recall"],
                "ceiling_recall_0": ceil_metrics[f"{primary_thr:g}"]["recall"],
                "ceiling_thresholds": ceil_metrics,
            })
        assignment = summarize_assignment_matrix(
            part_assignment_iou_matrix(pred_coords_list, raw_coords_list)
        )
        per_part_recall = [r["pred_recall_0"] for r in part_rows if r["obj_id"] == obj_id]
        object_rows.append({
            "obj_id": obj_id,
            "K": valid_k,
            "assignment_diag_iou": float(assignment["assignment_diag_iou"]),
            "assignment_offdiag_max": float(assignment["assignment_offdiag_max"]),
            "min_part_recall_0": float(min(per_part_recall)) if per_part_recall else math.nan,
            "has_small_part": any(
                r["size_bucket"] == "small" for r in part_rows if r["obj_id"] == obj_id
            ),
        })
        print(
            f"[INFO][pred] {n}/{total} obj={obj_id} K={valid_k} "
            f"diag={assignment['assignment_diag_iou']:.3f} "
            f"offdiag={assignment['assignment_offdiag_max']:.3f}",
            flush=True,
        )
    return {
        "parts": len(part_rows),
        "objects": len(object_rows),
        "summary": _aggregate(part_rows, thresholds),
        "object_rows": object_rows,
        "rows": part_rows,
    }


def _run_ckpt_scale(cfg, ckpt_path: Path, device) -> Dict[str, Any]:
    yaml_latent_scale = float(_cfg_dict(cfg.flow).get("latent_scale", 1.0))
    yaml_num_steps = int(_cfg_dict(cfg.flow).get("num_steps", 20))
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    ckpt_cfg = ckpt.get("config") or {}
    ckpt_flow = ckpt_cfg.get("flow") if isinstance(ckpt_cfg, dict) else None
    result: Dict[str, Any] = {
        "ckpt_path": str(ckpt_path),
        "yaml_latent_scale": yaml_latent_scale,
        "yaml_num_steps": yaml_num_steps,
        "ckpt_has_config": bool(ckpt_cfg),
        "ckpt_has_flow": bool(ckpt_flow),
    }
    problems: List[str] = []
    if not ckpt_flow:
        problems.append(
            "ckpt has NO config.flow -> inference.py SILENTLY uses latent_scale=1.0 "
            f"(training was {yaml_latent_scale}) -> ~{yaml_latent_scale:g}x decode-input scale break"
        )
    else:
        result["ckpt_latent_scale"] = float(ckpt_flow.get("latent_scale", 1.0))
        result["ckpt_num_steps"] = int(ckpt_flow.get("num_steps", 20))
        if abs(result["ckpt_latent_scale"] - yaml_latent_scale) > 1e-9:
            problems.append(
                f"ckpt latent_scale={result['ckpt_latent_scale']} != yaml {yaml_latent_scale}"
            )
    result["problems"] = problems
    result["ok"] = not problems
    for p in problems:
        print(f"[ERROR][ckpt-scale] {p}", flush=True)
    if result["ok"]:
        print(
            f"[INFO][ckpt-scale] OK latent_scale={result.get('ckpt_latent_scale')} "
            f"num_steps={result.get('ckpt_num_steps')}",
            flush=True,
        )
    return result


def _print_summary(title: str, summary: Dict[str, Any], thresholds: tuple[float, ...]) -> None:
    print(f"\n===== {title} =====", flush=True)
    header = "bucket      parts  " + "  ".join(f"R@{t:g}".rjust(8) for t in thresholds)
    print(header, flush=True)
    for name in ("overall", "small", "medium", "large"):
        grp = summary["overall"] if name == "overall" else summary["by_size"][name]
        parts = int(grp.get("parts", 0))
        if parts == 0:
            print(f"{name:<10}  {parts:>5}   (none)", flush=True)
            continue
        cells = "  ".join(f"{grp[f'{t:g}']['recall']:.3f}".rjust(8) for t in thresholds)
        print(f"{name:<10}  {parts:>5}   {cells}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Part SS Latent Flow identity/completeness diagnostics.")
    parser.add_argument("--config", default=None, help="Part SS Latent Flow YAML config (omit if --from-ckpt-config)")
    parser.add_argument(
        "--from-ckpt-config",
        action="store_true",
        help="Build the config from the checkpoint's embedded config (guarantees model/data/scale match the run "
             "that produced the ckpt). Requires --ckpt/--load-dir. Overrides still apply on top.",
    )
    parser.add_argument("--mode", choices=("ceiling", "pred", "ckpt-scale", "all"), default="all")
    parser.add_argument("--ckpt", default=None, help="Checkpoint .pt file (needed for pred / ckpt-scale)")
    parser.add_argument("--load-dir", default=None)
    parser.add_argument("--step", type=int, default=None)
    parser.add_argument("--max-samples", type=int, default=100, help="Objects to scan (ceiling/pred)")
    parser.add_argument("--sample-mode", choices=("first", "spread"), default="spread")
    parser.add_argument("--object-ids", default=None, help="Comma-separated object IDs (applied before sampling)")
    parser.add_argument("--thresholds", default="0,-0.25,-0.5,-1.0", help="Comma-separated decode thresholds")
    parser.add_argument("--device", default=None)
    parser.add_argument("--output", default=None, help="Write the full JSON report here")
    parser.add_argument("overrides", nargs="*", help="OmegaConf dotlist overrides, e.g. data.data_root=/path")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.from_ckpt_config:
        ckpt_for_cfg = _checkpoint_path(args)
        _ck = torch.load(ckpt_for_cfg, map_location="cpu", weights_only=False)
        if not _ck.get("config"):
            raise ValueError(
                f"{ckpt_for_cfg} has no embedded 'config'; cannot use --from-ckpt-config "
                "(pass --config <yaml> with data overrides instead)"
            )
        cfg = OmegaConf.create(_ck["config"])
        print(f"[INFO] config source: embedded in ckpt {ckpt_for_cfg}", flush=True)
    else:
        if not args.config:
            raise ValueError("provide --config <yaml>, or --from-ckpt-config together with --ckpt/--load-dir")
        cfg = load_config(args.config)
        print(f"[INFO] config source: {args.config}", flush=True)
    if args.overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(args.overrides))
    stage = str(cfg.get("stage"))
    data_cfg = _cfg_dict(cfg.data)
    flow_cfg = _cfg_dict(cfg.flow)
    eval_cfg = _cfg_dict(cfg.eval)
    loss_cfg = _cfg_dict(cfg.loss) if "loss" in cfg else {}
    boundaries = tuple(float(x) for x in loss_cfg.get("size_bucket_boundaries", [500.0, 3000.0]))
    thresholds = tuple(float(x) for x in str(args.thresholds).split(","))

    seed = int(getattr(cfg.training, "seed", 42)) if "training" in cfg else 42
    _setup_rng(seed)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    needs_model = args.mode in ("pred",) or (args.mode == "all" and (args.ckpt or args.load_dir))
    needs_decoder = args.mode in ("ceiling", "pred", "all")
    needs_ckpt = args.mode in ("pred", "ckpt-scale") or (args.mode == "all" and (args.ckpt or args.load_dir))

    print("=" * 60, flush=True)
    print("Part SS Latent Flow Diagnostics", flush=True)
    print(f"  stage={stage} mode={args.mode} device={device}", flush=True)
    print(f"  size_bucket_boundaries={boundaries} thresholds={thresholds}", flush=True)
    print(f"  flow.latent_scale={flow_cfg.get('latent_scale')} num_steps={flow_cfg.get('num_steps')}", flush=True)
    print("=" * 60, flush=True)

    dataset = _dataset_cls(stage)(data_cfg)
    candidate_indices = _object_id_filter_indices(dataset, args.object_ids)
    sample_indices = _sample_indices_for_eval(
        len(dataset), int(args.max_samples), args.sample_mode, candidate_indices,
    )
    print(f"[INFO] selected {len(sample_indices)} / {len(dataset)} objects", flush=True)

    decoder = load_ss_decoder(eval_cfg["ss_decoder_ckpt"]) if needs_decoder else None

    report: Dict[str, Any] = {
        "stage": stage,
        "config": args.config if not args.from_ckpt_config else "ckpt-embedded",
        "thresholds": list(thresholds),
        "size_bucket_boundaries": list(boundaries),
        "num_objects_selected": len(sample_indices),
        "flow_latent_scale": float(flow_cfg.get("latent_scale", 1.0)),
        "flow_num_steps": int(flow_cfg.get("num_steps", 20)),
    }

    if needs_ckpt:
        ckpt_path = _checkpoint_path(args)
        report["ckpt_scale"] = _run_ckpt_scale(cfg, ckpt_path, device)

    if args.mode in ("ceiling", "all"):
        report["ceiling"] = _run_ceiling(dataset, decoder, sample_indices, thresholds, boundaries)
        _print_summary("CEILING recall (decode GT latent vs raw)", report["ceiling"]["summary"], thresholds)

    if needs_model:
        ckpt_path = _checkpoint_path(args)
        model, ckpt_step, ckpt_cfg = _load_model(cfg, ckpt_path, device)
        _apply_ckpt_latent_norm(flow_cfg, ckpt_cfg)
        report["ckpt_step"] = ckpt_step
        report["pred"] = _run_pred(
            dataset, decoder, model, device, flow_cfg, sample_indices, thresholds, boundaries,
        )
        _print_summary("PRED recall (decode flow sample vs raw)", report["pred"]["summary"], thresholds)
        offdiags = [o["assignment_offdiag_max"] for o in report["pred"]["object_rows"]]
        if offdiags:
            print(
                f"\n[INFO][pred] identity off-diag IoU: "
                f"mean={sum(offdiags)/len(offdiags):.3f} max={max(offdiags):.3f} "
                f"(>0.3 on {sum(1 for x in offdiags if x > 0.3)}/{len(offdiags)} objects)",
                flush=True,
            )
    elif args.mode == "all" and not (args.ckpt or args.load_dir):
        print("[WARN] mode=all but no --ckpt/--load-dir: skipping pred (needs a checkpoint)", flush=True)

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"\n[INFO] wrote report: {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
