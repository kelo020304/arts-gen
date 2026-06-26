#!/usr/bin/env python3
"""D1 diagnostics for Route-V promptable part segmentation.

Reports:
  - Head1 cell-IoU / voxel-IoU with GT candidate cells / e2e voxel-IoU.
  - Candidate voxel token counts before cap for GT and predicted masks.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader


PROJECT_ROOT = Path(__file__).resolve().parents[2]
TRELLIS_PATH = PROJECT_ROOT / "TRELLIS-arts"
if str(TRELLIS_PATH) not in sys.path:
    sys.path.insert(0, str(TRELLIS_PATH))
DEV_PATH = PROJECT_ROOT / "scripts" / "dev"
if str(DEV_PATH) not in sys.path:
    sys.path.insert(0, str(DEV_PATH))

from scripts.train.part_promptable_seg.part_promptable_seg_utils import (  # noqa: E402
    PromptablePartDataset,
    bucket_name,
    collate_promptable_parts,
    compute_empty_code,
    decode_metrics_for_batch,
    dense_occ_from_coords,
    enumerate_part_rows,
    format_table,
    latent_support_mask,
    load_ss_decoder,
    load_ss_encoder,
    make_base_dataset,
    mask_morphology,
    pick_gate1_rows,
    summarize_by_bucket,
)
from trellis.models.part_seg.promptable_latent_seg import (  # noqa: E402
    PromptablePartLatentSegNet,
    semantic_classes_from_ckpt,
    voxel_embedding_dim_from_ckpt,
)
from scripts.train.part_promptable_seg.train_part_promptable_seg import decode_full_occ, voxel_decode_metrics_from_forward  # noqa: E402


DEFAULT_CKPT = Path("/mnt/robot-data-lab/jzh/art-gen-output/debug/part_promptable_seg_gate1_route_v_voxel_cap8192/ckpts/latest.pt")
DEFAULT_OUT = Path("/mnt/robot-data-lab/jzh/art-gen-output/debug/part_promptable_seg_d1_route_v_step5000")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", type=Path, default=DEFAULT_CKPT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--support-multiplier", type=float, default=4.0)
    parser.add_argument("--voxel-max-tokens", type=int, default=8192)
    return parser.parse_args()


def append_code_update(text: str) -> None:
    path = TRELLIS_PATH / "code_update" / "part_promptable_seg.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write("\n\n")
        f.write(text.rstrip())
        f.write("\n")


def load_model(ckpt_path: Path, device: torch.device) -> tuple[PromptablePartLatentSegNet, torch.Tensor, dict[str, Any]]:
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    args = ckpt.get("args", {})
    model = PromptablePartLatentSegNet(
        dim=int(args.get("dim", 256)),
        depth=int(args.get("depth", 6)),
        head_depth=int(args.get("head_depth", 2)),
        heads=int(args.get("heads", 8)),
        use_voxel_head=True,
        semantic_classes=semantic_classes_from_ckpt(ckpt),
        voxel_embedding_dim=voxel_embedding_dim_from_ckpt(ckpt),
    ).to(device)
    model.load_state_dict(ckpt["model"], strict=True)
    model.eval()
    empty_code = ckpt["empty_code"].float()
    return model, empty_code, args


def jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    return value


def candidate_token_count(full_occ: torch.Tensor, cells16: torch.Tensor) -> torch.Tensor:
    cells64 = cells16.bool().unsqueeze(1)
    cells64 = cells64.repeat_interleave(4, dim=2).repeat_interleave(4, dim=3).repeat_interleave(4, dim=4)
    return ((full_occ > 0.5) & cells64).sum(dim=(1, 2, 3, 4)).long()


def summarize_counts(rows: list[dict[str, Any]], key: str, cap: int) -> dict[str, Any]:
    vals = np.asarray([int(r[key]) for r in rows], dtype=np.int64)
    truncated = vals > int(cap)
    return {
        "n": int(vals.size),
        "cap": int(cap),
        "truncated_n": int(truncated.sum()),
        "truncated_frac": float(truncated.mean()) if vals.size else float("nan"),
        "min": int(vals.min()) if vals.size else 0,
        "p50": float(np.percentile(vals, 50)) if vals.size else float("nan"),
        "p90": float(np.percentile(vals, 90)) if vals.size else float("nan"),
        "p95": float(np.percentile(vals, 95)) if vals.size else float("nan"),
        "max": int(vals.max()) if vals.size else 0,
    }


def count_table(rows: list[dict[str, Any]], cap: int) -> str:
    out = []
    for name in ("gt_token_count", "e2e_token_count"):
        item = summarize_counts(rows, name, cap)
        out.append(
            {
                "kind": name.replace("_token_count", ""),
                "n": item["n"],
                "cap": item["cap"],
                "trunc_n": item["truncated_n"],
                "trunc_%": f"{100.0 * item['truncated_frac']:.1f}",
                "p50": f"{item['p50']:.0f}",
                "p90": f"{item['p90']:.0f}",
                "p95": f"{item['p95']:.0f}",
                "max": item["max"],
            }
        )
    return format_table(out, ["kind", "n", "cap", "trunc_n", "trunc_%", "p50", "p90", "p95", "max"])


def metric_table(summary: dict[str, Any]) -> str:
    rows = []
    for bucket in ("tiny", "small", "medium", "large", "button", "all"):
        if bucket not in summary:
            continue
        item = summary[bucket]
        rows.append(
            {
                "bucket": bucket,
                "n": item["n"],
                "Head1_cell": f"{item.get('cell_iou', float('nan')):.4f}",
                "GTcand_voxel": f"{item.get('gt_candidate_voxel_iou', float('nan')):.4f}",
                "e2e_voxel": f"{item.get('e2e_voxel_iou', float('nan')):.4f}",
            }
        )
    return format_table(rows, ["bucket", "n", "Head1_cell", "GTcand_voxel", "e2e_voxel"])


@torch.no_grad()
def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() or not args.device.startswith("cuda") else "cpu")

    base_ds = make_base_dataset()
    all_rows = enumerate_part_rows(base_ds)
    selected, gate1_meta = pick_gate1_rows(all_rows)
    ds = PromptablePartDataset(base_ds, selected, mask_size=512)
    loader = DataLoader(ds, batch_size=int(args.batch_size), shuffle=False, num_workers=0, collate_fn=collate_promptable_parts)

    model, ckpt_empty_code, ckpt_args = load_model(args.ckpt, device)
    encoder = load_ss_encoder(device=device, fp32=True)
    decoder = load_ss_decoder(device=device, fp32=True)
    empty_code = ckpt_empty_code if ckpt_empty_code is not None else compute_empty_code(encoder, device=device)

    rows: list[dict[str, Any]] = []
    for batch in loader:
        z_global = batch["z_global"].to(device=device, dtype=torch.float32)
        masks2d = batch["masks2d"].to(device=device, dtype=torch.float32)
        latent_gt = batch["latent_gt"].to(device=device, dtype=torch.float32)
        raw_m = batch["m_gt"].to(device=device, dtype=torch.float32)
        support_m, _noise, _thr = latent_support_mask(latent_gt, empty_code.to(device), raw_m, multiplier=float(args.support_multiplier))
        full_occ = decode_full_occ(decoder, z_global, threshold=0.0)

        gt_cand = mask_morphology(support_m, "dilate")
        out_gt = model.forward_voxels(
            z_global,
            masks2d,
            gt_cand,
            full_occ,
            max_voxels_per_sample=int(args.voxel_max_tokens),
        )
        pred_m = (out_gt["m_logit"].sigmoid() > 0.5).float().view(support_m.shape)
        e2e_cand = mask_morphology(pred_m, "dilate")
        out_e2e = model.forward_voxels(
            z_global,
            masks2d,
            e2e_cand,
            full_occ,
            max_voxels_per_sample=int(args.voxel_max_tokens),
        )
        gt_metrics = voxel_decode_metrics_from_forward(out_gt["voxel_logits"], out_gt["voxel_coords"], batch["raw_coords"])
        e2e_metrics = voxel_decode_metrics_from_forward(out_e2e["voxel_logits"], out_e2e["voxel_coords"], batch["raw_coords"])

        gt_counts = candidate_token_count(full_occ, gt_cand).cpu().tolist()
        e2e_counts = candidate_token_count(full_occ, e2e_cand).cpu().tolist()
        m_flat = support_m.reshape(support_m.shape[0], -1).bool()
        p_flat = pred_m.reshape(pred_m.shape[0], -1).bool()
        for idx in range(z_global.shape[0]):
            inter = (p_flat[idx] & m_flat[idx]).sum().float()
            union = (p_flat[idx] | m_flat[idx]).sum().float()
            raw_count = int(batch["raw_count"][idx].item())
            rows.append(
                {
                    "obj_id": batch["obj_id"][idx],
                    "angle_idx": int(batch["angle_idx"][idx]),
                    "part_name": batch["part_name"][idx],
                    "raw_count": raw_count,
                    "bucket": bucket_name(raw_count),
                    "cell_iou": float((inter / union.clamp_min(1.0)).item()),
                    "cell_pred_count": int(p_flat[idx].sum().item()),
                    "cell_gt_count": int(m_flat[idx].sum().item()),
                    "gt_candidate_voxel_iou": float(gt_metrics[idx]["decode_iou"]),
                    "e2e_voxel_iou": float(e2e_metrics[idx]["decode_iou"]),
                    "gt_token_count": int(gt_counts[idx]),
                    "e2e_token_count": int(e2e_counts[idx]),
                    "gt_truncated": bool(int(gt_counts[idx]) > int(args.voxel_max_tokens)),
                    "e2e_truncated": bool(int(e2e_counts[idx]) > int(args.voxel_max_tokens)),
                    "gt_pred_count": int(gt_metrics[idx]["pred_count"]),
                    "e2e_pred_count": int(e2e_metrics[idx]["pred_count"]),
                }
            )

    summary = summarize_by_bucket(rows, ("cell_iou", "gt_candidate_voxel_iou", "e2e_voxel_iou"))
    summary["all"] = {
        "n": len(rows),
        "cell_iou": float(np.mean([r["cell_iou"] for r in rows])),
        "gt_candidate_voxel_iou": float(np.mean([r["gt_candidate_voxel_iou"] for r in rows])),
        "e2e_voxel_iou": float(np.mean([r["e2e_voxel_iou"] for r in rows])),
    }
    metrics = metric_table(summary)
    counts = count_table(rows, int(args.voxel_max_tokens))

    metadata = {
        "ckpt": str(args.ckpt),
        "ckpt_args": jsonable(ckpt_args),
        "gate1_selection": gate1_meta,
        "voxel_max_tokens": int(args.voxel_max_tokens),
        "note": "No view dropout or semantic auxiliary head exists in this minimal Route-V training path.",
    }
    (args.out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (args.out_dir / "rows.json").write_text(json.dumps(rows, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (args.out_dir / "metrics.txt").write_text(metrics + "\n", encoding="utf-8")
    (args.out_dir / "cap_counts.txt").write_text(counts + "\n", encoding="utf-8")
    append_code_update(
        f"# D1 Route-V Step5000 Diagnostics\n\n"
        f"out_dir: `{args.out_dir}`\n"
        f"ckpt: `{args.ckpt}`\n"
        f"minimal-config note: no view dropout or semantic auxiliary head exists in this path.\n\n"
        f"Metrics:\n\n```\n{metrics}\n```\n\n"
        f"Cap `{int(args.voxel_max_tokens)}` token counts:\n\n```\n{counts}\n```"
    )
    print("Metrics:\n" + metrics, flush=True)
    print("\nCap counts:\n" + counts, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
