#!/usr/bin/env python3
"""Unified arts training entry. Driven by YAML `stage:` field.

Usage:
    # Single GPU smoke test
    python TRELLIS-arts/train_arts.py \\
        --config TRELLIS-arts/configs/arts/ss_flow_art/smoke_test.yaml \\
        training.max_steps=5

    # Multi-GPU DDP (called by scripts/train/<stage>_train.bash via torchrun)
    torchrun --nproc_per_node=4 TRELLIS-arts/train_arts.py \\
        --config TRELLIS-arts/configs/arts/ss_flow_art/mv_4view.yaml

    # Resume
    python TRELLIS-arts/train_arts.py \\
        --config TRELLIS-arts/configs/arts/ss_flow_art/mv_4view.yaml \\
        --load-dir output/ss_flow_art_mv_4view --resume-step 50000

Design notes:
- Stage is selected by YAML's top-level `stage:` field (D-11), NOT a CLI flag.
  This guarantees launcher / CLI / resume see consistent dispatch and prevents
  users from "skipping" stages by accident.
- We avoid `import trellis` at the top level (CLAUDE.md Lessons Learned:
  "训练入口必须最小依赖"). Instead we register `trellis` and key subpackages
  via types.ModuleType so the eager imports in trellis/__init__.py
  (pipelines / renderers / rembg / torchvision) do not trigger.
- PROJECT_ROOT is computed as dirname(dirname(__file__)) — 1 level up because
  train_arts.py lives in TRELLIS-arts/, NOT 4 levels up like the legacy
  scripts/train/stage2/train.py (D-13).
"""
import argparse
import importlib
import os
import sys
import types

# ---- PROJECT_ROOT (D-13: 1 level up; train_arts.py lives in TRELLIS-arts/) ----
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRELLIS_PATH = os.path.join(PROJECT_ROOT, "TRELLIS-arts")
if TRELLIS_PATH not in sys.path:
    sys.path.insert(0, TRELLIS_PATH)

# ---- Minimal-deps trellis registration (CLAUDE.md Lessons Learned) ----
# Avoids triggering trellis/__init__.py which eagerly imports
# pipelines → rembg → torchvision (heavy, not needed for training).
_pkg = types.ModuleType("trellis")
_pkg.__path__ = [os.path.join(TRELLIS_PATH, "trellis")]
_pkg.__package__ = "trellis"
sys.modules.setdefault("trellis", _pkg)

# Register the 5 subpackages used by training (lazy: their submodules are
# loaded on demand via `from trellis.X.Y import Z`).
for sp in ("models", "modules", "trainers", "utils", "datasets"):
    m = types.ModuleType(f"trellis.{sp}")
    m.__path__ = [os.path.join(TRELLIS_PATH, "trellis", sp)]
    m.__package__ = f"trellis.{sp}"
    sys.modules.setdefault(f"trellis.{sp}", m)

# pipelines: register stub package (path is set so submodules like
# trellis.pipelines.samplers can still be imported), but DO NOT execute its
# __init__.py (which would pull TrellisImageTo3DPipeline → rembg).
_pipelines = types.ModuleType("trellis.pipelines")
_pipelines.__path__ = [os.path.join(TRELLIS_PATH, "trellis", "pipelines")]
_pipelines.__package__ = "trellis.pipelines"
sys.modules.setdefault("trellis.pipelines", _pipelines)

# representations: NOT stubbed — its __init__.py is light (4 internal imports)
# and SLat's render_utils does `from ..representations import Octree, Gaussian,
# MeshExtractResult` which needs those names. Stubbing breaks SLat training.
# Leave to normal Python import machinery via trellis.__path__.

# DINOv2 hub cache + default attention backend
os.environ.setdefault("TORCH_HOME", os.path.join(PROJECT_ROOT, "submodules", "TRELLIS.1"))
os.environ.setdefault("ATTN_BACKEND", "sdpa")
# Default the CUDA caching allocator to expandable segments. Must be set before
# the first CUDA allocation (before torch initializes the allocator), which is why
# it lives here at the entry top — this runs in every torchrun worker and in
# container/queue launches alike. setdefault keeps any explicit override.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

# ---- Imports (after stub registration) ----
from trellis.utils.arts.config_utils import load_config  # noqa: E402

# ---- Dispatch table (D-12) ----
_STAGE_DISPATCH = {
    "ss_flow_art":    "trellis.trainers.arts.ss_flow_art",
    "ss_flow_global_z": "trellis.trainers.arts.ss_flow_global_z",
    "slat_flow_art":  "trellis.trainers.arts.slat_flow_art",
    "part_ss_latent_flow": "trellis.trainers.arts.part_ss_latent_flow",
    "part_ss_latent_flow_mask16": "trellis.trainers.arts.part_ss_latent_flow_mask16",
    "part_ss_latent_flow_single_view": "trellis.trainers.arts.part_ss_latent_flow_single_view",
    "part_mmdit": "trellis.trainers.arts.part_mmdit",
    "part_predictor": "trellis.trainers.arts.part_predictor",
}


def main():
    parser = argparse.ArgumentParser(
        description="Unified arts training entry (D-10). Stage is selected by "
                    "YAML `stage:` field (D-11), not a CLI flag."
    )
    parser.add_argument("--config", required=True,
                        help="Path to YAML in TRELLIS-arts/configs/arts/")
    parser.add_argument("--load-dir", default=None,
                        help="Resume checkpoint directory")
    parser.add_argument("--resume-step", type=int, default=None,
                        help="Resume step (paired with --load-dir)")
    parser.add_argument("overrides", nargs="*",
                        help="OmegaConf dotlist overrides (e.g. training.lr=1e-5 "
                             "or training.max_steps=5 for smoke tests)")
    args = parser.parse_args()

    cfg = load_config(args.config, overrides=args.overrides)

    stage = cfg.get("stage")
    if stage not in _STAGE_DISPATCH:
        raise SystemExit(
            f"YAML 'stage:' must be one of {list(_STAGE_DISPATCH)}, got {stage!r}.\n"
            f"  Hint: every yaml in TRELLIS-arts/configs/arts/ inherits 'stage:' from\n"
            f"  base.yaml or its stage subdirectory's mid-level yaml. Check `_base_:`\n"
            f"  chain in: {args.config}"
        )

    # Inject resume args into cfg (each stage trainer reads cfg.training.load_dir
    # / resume_step; cfg.training is OmegaConf DictConfig, supports attribute
    # assignment).
    if args.load_dir is not None:
        cfg.training.load_dir = args.load_dir
    if args.resume_step is not None:
        cfg.training.resume_step = args.resume_step

    # Dispatch
    train_fn = importlib.import_module(_STAGE_DISPATCH[stage]).train
    train_fn(cfg)


if __name__ == "__main__":
    main()
