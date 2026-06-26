#!/usr/bin/env python3
"""Pipeline step 01a: multi-view RGB images -> SS latent (.npz).

Thin orchestrator (D-23): only argparse + load images + call inference + save.
"""
import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "TRELLIS-arts"))

import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402
from inference import run_ss_flow  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description="SS Flow multi-view inference")
    ap.add_argument("--images", required=True, nargs="+",
                    help="paths to multi-view RGB images (PNG/JPG)")
    ap.add_argument("--ckpt", required=True, help="SS Flow DiT .safetensors")
    ap.add_argument("--num_steps", type=int, default=25)
    ap.add_argument("--output_dir", required=True)
    args = ap.parse_args()

    pil_images = [Image.open(p) for p in args.images]
    latent = run_ss_flow(pil_images, args.ckpt, num_steps=args.num_steps)  # [8,16,16,16] cpu

    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / "ss_latent.npz", latent=latent.numpy().astype(np.float32))
    print(f"[01_ss_flow_mv] saved ss_latent.npz -> {out}")


if __name__ == "__main__":
    main()
