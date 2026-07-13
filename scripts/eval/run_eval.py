#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.eval.tasks.ee_0617_batch import (  # noqa: E402
    DEFAULT_DATA_CONFIG,
    DEFAULT_GAUSSIAN_DECODER,
    DEFAULT_OUT_DIR,
    DEFAULT_PACKED_INDEX,
    DEFAULT_SLAT_FLOW_CKPT,
    DEFAULT_SLAT_MESH_DECODER_CKPT,
    DEFAULT_SPLIT_JSON,
    DEFAULT_SS_DECODER_CKPT,
    PYTHON,
)
from scripts.eval.tasks import ee_0617  # noqa: E402


DEFAULT_LOCAL_PART_SEG_CKPT = Path(
    "/robot/data-lab/jzh/art-gen/ckpts/part-prompt-seg/part_promptable_seg_full_S_0618-1/ckpts/step_100000.pt"
)
DEFAULT_LOCAL_SS_FLOW_CKPT = Path(
    "/mnt/robot-data-lab/jzh/art-gen/ckpt/tre-ss-flow/tre-ss-concat-0616-1/ckpts/denoiser_ema0.999_step0012500.pt"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Unified eval CLI for arts-gen.")
    sub = parser.add_subparsers(dest="task", required=True)

    ee = sub.add_parser(
        "ee_0617",
        help="Accepted 0617 EE pipeline: rendered views -> SS concat -> promptable part seg -> one whole SLat flow.",
    )
    ee.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    ee.add_argument("--data-config", type=Path, default=DEFAULT_DATA_CONFIG)
    ee.add_argument("--split-json", type=Path, default=DEFAULT_SPLIT_JSON)
    ee.add_argument("--part-seg-ckpt", type=Path, default=DEFAULT_LOCAL_PART_SEG_CKPT)
    ee.add_argument("--ss-flow-ckpt", type=Path, default=DEFAULT_LOCAL_SS_FLOW_CKPT)
    ee.add_argument("--ss-decoder-ckpt", type=Path, default=DEFAULT_SS_DECODER_CKPT)
    ee.add_argument("--slat-flow-ckpt", type=Path, default=DEFAULT_SLAT_FLOW_CKPT)
    ee.add_argument("--slat-mesh-decoder-ckpt", type=Path, default=DEFAULT_SLAT_MESH_DECODER_CKPT)
    ee.add_argument("--slat-gaussian-decoder-ckpt", type=Path, default=DEFAULT_GAUSSIAN_DECODER)
    ee.add_argument("--python", type=Path, default=PYTHON)
    ee.add_argument("--limit", type=int, default=128)
    ee.add_argument("--train-count", type=int, default=85)
    ee.add_argument("--held-count", type=int, default=43)
    ee.add_argument("--gpus", default="0,1,2,3")
    ee.add_argument("--allowed-datasets", default="phyx-verse,realappliance")
    ee.add_argument("--selection-mode", choices=("objects", "samples"), default="samples")
    ee.add_argument("--sample-selection-unit", choices=("objects", "pairs"), default="objects")
    ee.add_argument("--packed-index", type=Path, default=DEFAULT_PACKED_INDEX)
    ee.add_argument("--slat-steps", type=int, default=25)
    ee.add_argument("--slat-seed", type=int, default=42)
    ee.add_argument("--render-view", type=int, default=0)
    ee.add_argument("--resolution", type=int, default=512)
    ee.add_argument("--tile-size", type=int, default=240)
    ee.add_argument("--panel-cols", type=int, default=4)
    ee.add_argument("--export-mujoco", action="store_true")
    ee.add_argument("--export-usd", action="store_true")
    ee.add_argument("--mujoco-textured-assets", action="store_true")
    ee.add_argument(
        "--mujoco-appearance-source",
        choices=("obj-vertex-color", "mesh-vertex-texture", "gaussian-texture"),
        default="obj-vertex-color",
    )
    ee.add_argument("--mujoco-texture-size", type=int, default=512)
    ee.add_argument("--mujoco-texture-render-resolution", type=int, default=512)
    ee.add_argument("--mujoco-texture-nviews", type=int, default=30)
    ee.add_argument("--mujoco-texture-mode", choices=("fast", "opt"), default="fast")
    ee.add_argument("--fill-hidden-vertex-colors", action="store_true")
    ee.add_argument("--hidden-color-fill-out-dir", type=Path, default=None)
    ee.add_argument("--hidden-color-fill-dark-threshold", type=float, default=0.18)
    ee.add_argument(
        "--slat-token-source",
        choices=("live", "cache"),
        default="live",
        help="Use live for the accepted 0617 EE path; cache is diagnostic only.",
    )
    ee.add_argument(
        "--part-cc-filter",
        action="store_true",
        help="Post-process predicted part voxels by moving small remote connected components back to body.",
    )
    ee.add_argument("--part-cc-min-component-voxels", type=int, default=32)
    ee.add_argument("--part-cc-min-component-fraction", type=float, default=0.05)
    ee.add_argument("--part-cc-max-component-distance", type=int, default=2)
    ee.add_argument("--part-cc-max-large-component-distance", type=int, default=None)
    ee.add_argument("--part-joint-candidate-mode", choices=("proposal", "full_occ"), default="proposal")
    ee.add_argument("--part-joint-refine", action=argparse.BooleanOptionalAction, default=False)
    ee.add_argument("--part-joint-refine-iters", type=int, default=1)
    ee.add_argument("--part-joint-refine-pairwise", type=float, default=3.0)
    ee.add_argument("--part-joint-refine-margin", type=float, default=0.0)
    ee.add_argument("--part-joint-refine-margin-quantile", type=float, default=0.01)
    ee.add_argument("--part-joint-refine-neighborhood", type=int, choices=(6, 18, 26), default=6)
    ee.add_argument("--part-joint-refine-min-vote-gain", type=float, default=0.0)
    ee.add_argument("--part-joint-refine-preserve-small-classes", type=int, default=32)
    ee.add_argument("--part-joint-save-logits", action=argparse.BooleanOptionalAction, default=False)
    ee.add_argument(
        "--part-t0-filter",
        action="store_true",
        help="Enable T0 joint argmax + competition-band smoothing + CC part boundary postprocess.",
    )
    ee.add_argument("--part-t0-part-threshold", type=float, default=0.5)
    ee.add_argument("--part-t0-margin-threshold", type=float, default=0.35)
    ee.add_argument("--part-t0-smooth-iters", type=int, default=1)
    ee.add_argument("--part-t0-disable-cc", action="store_true")
    ee.add_argument("--force", action="store_true")
    ee.add_argument("--force-stage", action="store_true")
    ee.add_argument("--force-export", action="store_true")
    ee.add_argument("--overwrite-selection", action="store_true")
    ee.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    os.environ.setdefault("SS_FLOW_FUSION_MODE", "concat")
    args = parse_args()
    if args.task == "ee_0617":
        return ee_0617.run(args)
    raise SystemExit(f"unsupported task: {args.task}")


if __name__ == "__main__":
    raise SystemExit(main())
