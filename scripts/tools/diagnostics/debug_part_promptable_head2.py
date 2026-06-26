#!/usr/bin/env python3
"""Head2 assembly/decode checks for a single promptable part checkpoint."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
TRELLIS_PATH = PROJECT_ROOT / "TRELLIS-arts"
if str(TRELLIS_PATH) not in sys.path:
    sys.path.insert(0, str(TRELLIS_PATH))
DEV_PATH = PROJECT_ROOT / "scripts" / "dev"
if str(DEV_PATH) not in sys.path:
    sys.path.insert(0, str(DEV_PATH))

from scripts.train.part_promptable_seg.part_promptable_seg_utils import (  # noqa: E402
    PromptablePartDataset,
    coords_iou,
    decode_latents_to_coords,
    latent_support_mask,
    load_ss_decoder,
    make_base_dataset,
)
from trellis.models.part_seg.promptable_latent_seg import (  # noqa: E402
    PromptablePartLatentSegNet,
    semantic_classes_from_ckpt,
    voxel_embedding_dim_from_ckpt,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=Path, required=True)
    parser.add_argument("--obj-id", default="100283")
    parser.add_argument("--angle-idx", type=int, default=0)
    parser.add_argument("--part-name", default="button_0")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--support-multiplier", type=float, default=4.0)
    return parser.parse_args()


def load_model(ckpt, device):
    args = dict(ckpt.get("args", {}))
    state = ckpt["model"]
    model = PromptablePartLatentSegNet(
        dim=int(args.get("dim", 256)),
        depth=int(args.get("depth", 6)),
        head_depth=int(args.get("head_depth", 2)),
        heads=int(args.get("heads", 8)),
        use_xyz=int(state["stem.weight"].shape[1]) == 11,
        use_voxel_head=bool("voxel_out.weight" in state),
        semantic_classes=semantic_classes_from_ckpt(ckpt),
        voxel_embedding_dim=voxel_embedding_dim_from_ckpt(ckpt),
    ).to(device).eval()
    model.load_state_dict(state, strict=True)
    return model


@torch.no_grad()
def main() -> int:
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() or not args.device.startswith("cuda") else "cpu")
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    model = load_model(ckpt, device)
    decoder = load_ss_decoder(device=device, fp32=True)
    empty_code = ckpt["empty_code"].float().to(device)
    from scripts.train.part_promptable_seg.part_promptable_seg_utils import enumerate_part_rows

    base_ds = make_base_dataset()
    rows = [
        row
        for row in enumerate_part_rows(base_ds)
        if row.obj_id == str(args.obj_id) and row.angle_idx == int(args.angle_idx) and row.part_name == str(args.part_name)
    ]
    if len(rows) != 1:
        raise RuntimeError(f"expected one row, got {len(rows)}")
    ds = PromptablePartDataset(base_ds, rows, mask_size=512)
    item = ds[0]
    z_global = item["z_global"].unsqueeze(0).to(device)
    masks2d = item["masks2d"].unsqueeze(0).to(device)
    latent_gt = item["latent_gt"].unsqueeze(0).to(device)
    raw_m = item["m_gt"].unsqueeze(0).to(device)
    support, noise, threshold = latent_support_mask(latent_gt, empty_code, raw_m, multiplier=float(args.support_multiplier))

    oracle = support.unsqueeze(1) * latent_gt + (1.0 - support.unsqueeze(1)) * empty_code.view(1, 8, 16, 16, 16)
    out_gt = model(z_global, masks2d, empty_code, m_override=support)
    pred_support = (out_gt["m_logit"].sigmoid().view_as(support) > 0.5).float()
    out_pred = model(z_global, masks2d, empty_code, m_override=pred_support)

    latents = {
        "latent_gt": latent_gt,
        "oracle_support_gt": oracle,
        "model_gt_support": out_gt["part_latent"],
        "model_pred_support": out_pred["part_latent"],
        "empty": empty_code.view(1, 8, 16, 16, 16),
        "z_global": z_global,
    }
    coords = {name: decode_latents_to_coords(decoder, latent)[0] for name, latent in latents.items()}
    raw = item["raw_coords"]
    report = {
        "ckpt": str(args.ckpt),
        "obj_id": args.obj_id,
        "angle_idx": args.angle_idx,
        "part_name": args.part_name,
        "raw_count": int(raw.shape[0]),
        "support_count": int(support.sum().item()),
        "raw_cell_count": int(raw_m.sum().item()),
        "support_noise": float(noise.item()),
        "support_threshold": float(threshold.item()),
        "mask_pred_iou": float(((pred_support.bool() & support.bool()).sum().float() / (pred_support.bool() | support.bool()).sum().float().clamp_min(1)).item()),
        "latent_l1_vs_gt": {
            name: float((latent - latent_gt).abs().mean().item())
            for name, latent in latents.items()
        },
        "latent_l1_support_vs_gt": {
            name: float(((latent - latent_gt).abs() * support.unsqueeze(1)).sum().item() / support.sum().clamp_min(1).item() / 8.0)
            for name, latent in latents.items()
        },
        "decode": {
            name: {
                **coords_iou(coord, raw),
                "coord_count": int(coord.shape[0]),
            }
            for name, coord in coords.items()
        },
    }
    print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
