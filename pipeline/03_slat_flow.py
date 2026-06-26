#!/usr/bin/env python3
"""Pipeline step 03a: multi-view RGB + sparse coords -> SLat (.pt).

Thin orchestrator (D-23). Source: scripts/eval/stage4/infer.py:256-408 (D-29).
"""
import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "TRELLIS-arts"))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402
from inference import run_slat_flow  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description="SLat Flow inference")
    ap.add_argument("--images", required=True, nargs="+",
                    help="paths to multi-view RGB images (PNG/JPG)")
    ap.add_argument("--occupancy", required=True, help="path to occupancy.npz")
    ap.add_argument("--ckpt", required=True, help="SLat Flow DiT ckpt (.safetensors)")
    ap.add_argument("--num_steps", type=int, default=25)
    ap.add_argument("--output_dir", required=True)
    args = ap.parse_args()

    pil_images = [Image.open(p) for p in args.images]
    coords = torch.from_numpy(np.load(args.occupancy)["coords"]).long()  # [N,3]
    slat = run_slat_flow(pil_images, coords, args.ckpt, num_steps=args.num_steps)

    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    torch.save(slat, out / "slat.pt")
    print(f"[03_slat_flow] saved slat.pt -> {out}")


if __name__ == "__main__":
    main()
