#!/usr/bin/env python3
"""Pipeline step 01b: SS latent (.npz) -> sparse occupancy coords (.npz).

Thin orchestrator (D-23).
"""
import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "TRELLIS-arts"))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from inference import decode_ss  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description="SS decoder inference")
    ap.add_argument("--ss_latent", required=True, help="path to ss_latent.npz")
    ap.add_argument("--ckpt", required=True,
                    help="SparseStructureDecoder ckpt (.safetensors or basename)")
    ap.add_argument("--threshold", type=float, default=0.0)
    ap.add_argument("--output_dir", required=True)
    args = ap.parse_args()

    z_s = torch.from_numpy(np.load(args.ss_latent)["latent"]).float()  # [8,16,16,16]
    coords = decode_ss(z_s, args.ckpt, threshold=args.threshold)        # [N,3] cpu int64

    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / "occupancy.npz", coords=coords.numpy().astype(np.int64))
    print(f"[01_ss_decode] saved occupancy.npz (N={coords.shape[0]}) -> {out}")


if __name__ == "__main__":
    main()
