"""Run Tencent Hunyuan3D-2 image-to-3D shape generation on a single photo.

Designed for the Xiaomi Buds 6 case reconstruction sanity check on a 16 GB
RTX A4000: uses the 0.6B ``Hunyuan3D-2mini`` variant (shape only, ~6 GB VRAM)
and skips the 1.3B texture pipeline.

Run::

    conda activate arts-gen
    python scripts/inference/hunyuan3d_image_to_mesh.py \\
        --image /home/mi/图片/1.jpg \\
        --output_dir outputs/xiaomi_buds6_hunyuan
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

# HF cache lives inside the project so we don't pollute ~/.cache
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
os.environ.setdefault("HF_HOME", str(PROJECT_ROOT / "pretrained" / "hf_cache"))
os.environ.setdefault("TRANSFORMERS_CACHE", str(PROJECT_ROOT / "pretrained" / "hf_cache"))

from PIL import Image  # noqa: E402

from hy3dgen.rembg import BackgroundRemover  # noqa: E402
from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True, help="input RGB image of the object")
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--model", default="tencent/Hunyuan3D-2mini",
                    help="HF repo id (default: Hunyuan3D-2mini, 0.6B, ~6GB VRAM)")
    ap.add_argument("--subfolder", default="hunyuan3d-dit-v2-mini",
                    help="HF subfolder (mini=hunyuan3d-dit-v2-mini, base=hunyuan3d-dit-v2-0)")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--steps", type=int, default=30,
                    help="flow-matching sampling steps (default 30)")
    ap.add_argument("--guidance", type=float, default=5.0,
                    help="classifier-free guidance scale")
    ap.add_argument("--octree_resolution", type=int, default=256,
                    help="output mesh octree resolution (256 = 6GB; 384/512 need more)")
    ap.add_argument("--num_inference_steps", type=int, default=None,
                    help="alias for --steps (some pipelines use this name)")
    args = ap.parse_args(argv)

    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[load] {args.model}")
    t0 = time.time()
    pipe = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(args.model, subfolder=args.subfolder)
    print(f"[load] {time.time() - t0:.1f}s")

    img = Image.open(args.image).convert("RGBA")
    if img.mode == "RGB" or img.getextrema()[3][0] == 255:
        print("[rembg] removing background")
        img = BackgroundRemover()(img.convert("RGB"))
        img.save(out_dir / "input_nobg.png")

    print(f"[shape] sampling {args.steps} steps  octree={args.octree_resolution}")
    t0 = time.time()
    import torch
    torch.manual_seed(args.seed)
    mesh = pipe(
        image=img,
        num_inference_steps=args.num_inference_steps or args.steps,
        guidance_scale=args.guidance,
        octree_resolution=args.octree_resolution,
    )[0]
    print(f"[shape] {time.time() - t0:.1f}s")

    out_glb = out_dir / "shape.glb"
    out_obj = out_dir / "shape.obj"
    mesh.export(str(out_glb))
    mesh.export(str(out_obj))
    print(f"[saved] {out_glb}")
    print(f"[saved] {out_obj}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
