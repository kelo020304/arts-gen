#!/usr/bin/env python3
"""Pipeline step 02: global SS latent + part condition -> per-part SS latents."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "TRELLIS-arts"))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from inference import run_part_ss_latent_flow  # noqa: E402


def _load_ss_latent(path: str) -> torch.Tensor:
    data = np.load(path)
    if isinstance(data, np.lib.npyio.NpzFile):
        if "mean" in data.files:
            arr = data["mean"]
        elif "latent" in data.files:
            arr = data["latent"]
        else:
            raise KeyError(f"{path} must contain key 'mean' or 'latent'; found {data.files}")
    else:
        arr = data
    tensor = torch.from_numpy(np.asarray(arr)).float()
    if tensor.shape != (8, 16, 16, 16):
        raise ValueError(f"SS latent must be [8,16,16,16], got {tuple(tensor.shape)} from {path}")
    return tensor


def _load_cond_tokens(path: str) -> torch.Tensor:
    data = np.load(path)
    if isinstance(data, np.lib.npyio.NpzFile):
        if "tokens" not in data.files:
            raise KeyError(f"{path} must contain key 'tokens'; found {data.files}")
        arr = data["tokens"]
    else:
        arr = data
    tensor = torch.from_numpy(np.asarray(arr)).float()
    if tensor.dim() == 3:
        tensor = tensor.reshape(-1, tensor.shape[-1])
    if tensor.dim() != 2:
        raise ValueError(f"cond tokens must be [V*T,D] or [V,T,D], got {tuple(tensor.shape)}")
    return tensor


def _load_mask_token_labels(path: str) -> torch.Tensor:
    data = np.load(path)
    if isinstance(data, np.lib.npyio.NpzFile):
        if "mask_token_labels" not in data.files:
            raise KeyError(f"{path} must contain key 'mask_token_labels'; found {data.files}")
        arr = data["mask_token_labels"]
    else:
        arr = data
    tensor = torch.from_numpy(np.asarray(arr)).long()
    if tensor.dim() == 2:
        tensor = tensor.reshape(-1)
    if tensor.dim() != 1:
        raise ValueError(f"mask_token_labels must be [V*T] or [V,T], got {tuple(tensor.shape)}")
    return tensor


def _json_list(value: str, *, name: str):
    parsed = json.loads(value)
    if not isinstance(parsed, list):
        raise ValueError(f"{name} must be a JSON list, got {type(parsed).__name__}")
    return parsed


def main():
    ap = argparse.ArgumentParser(description="Part SS latent flow inference")
    ap.add_argument("--mode", default="part_ss_latent", choices=["part_ss_latent"])
    ap.add_argument("--ss-latent", required=True, help="global SS latent .npz/.npy")
    ap.add_argument("--cond-tokens", required=True, help="DINOv2 tokens .npz/.npy")
    ap.add_argument("--mask-token-labels", required=True, help="mask token labels .npy/.npz")
    ap.add_argument("--target-part-names", required=True, help='JSON list, e.g. ["wheel_0"]')
    ap.add_argument("--target-slots", required=True, help="JSON list, e.g. [1]")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--ss-decoder-ckpt", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--num-steps", type=int, default=None)
    ap.add_argument("--decode-threshold", type=float, default=0.0)
    args = ap.parse_args()

    z_global = _load_ss_latent(args.ss_latent)
    cond = _load_cond_tokens(args.cond_tokens)
    mask_token_labels = _load_mask_token_labels(args.mask_token_labels)
    if mask_token_labels.shape[0] != cond.shape[0]:
        raise ValueError(
            f"mask_token_labels has {mask_token_labels.shape[0]} entries, "
            f"but cond tokens have {cond.shape[0]}"
        )
    target_part_names = [str(x) for x in _json_list(args.target_part_names, name="target_part_names")]
    target_slots = [int(x) for x in _json_list(args.target_slots, name="target_slots")]

    result = run_part_ss_latent_flow(
        z_global,
        cond,
        args.ckpt,
        target_slots=target_slots,
        mask_token_labels=mask_token_labels,
        target_part_names=target_part_names,
        ss_decoder_ckpt=args.ss_decoder_ckpt,
        num_steps=args.num_steps,
        decode_threshold=args.decode_threshold,
    )

    out = Path(args.output_dir)
    latent_dir = out / "part_ss_latents"
    voxel_dir = out / "part_voxels"
    latent_dir.mkdir(parents=True, exist_ok=True)
    voxel_dir.mkdir(parents=True, exist_ok=True)
    summary = {"target_slots": result["target_slots"], "parts": {}}
    for part_name in target_part_names:
        np.save(latent_dir / f"{part_name}.npy", result["part_latents"][part_name])
        np.savez_compressed(voxel_dir / f"{part_name}.npz", coords=result["part_coords"][part_name])
        summary["parts"][part_name] = {
            "latent": str(latent_dir / f"{part_name}.npy"),
            "coords": str(voxel_dir / f"{part_name}.npz"),
            "num_coords": int(result["part_coords"][part_name].shape[0]),
        }
    (out / "part_flow_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[02_part_flow] saved part SS latents + decoded voxels -> {out}")


if __name__ == "__main__":
    main()
