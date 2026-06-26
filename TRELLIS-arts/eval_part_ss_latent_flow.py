#!/usr/bin/env python3
"""Standalone checkpoint eval/decode inspection for Part SS Latent Flow."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import train_arts  # noqa: F401  # Registers lightweight trellis package stubs.
import torch

from trellis.datasets.arts.part_ss_latent_flow import PartSSLatentFlowDataset
from trellis.datasets.arts.part_ss_latent_flow_single_view import PartSSLatentFlowSingleViewDataset
from trellis.models.part_flow import PartSSLatentFlowModel
from trellis.trainers.arts.part_ss_latent_flow import (
    _compatible_model_state_dict,
    _eval_decode_inspection,
    _eval_latent,
    _resolve_resume_checkpoint,
    _setup_rng,
)
from trellis.utils.arts.config_utils import config_to_dict, load_config


def _cfg_dict(cfg: Any) -> dict:
    return config_to_dict(cfg) if not isinstance(cfg, dict) else dict(cfg)


def _dataset_cls(stage: str):
    if stage == "part_ss_latent_flow":
        return PartSSLatentFlowDataset
    if stage == "part_ss_latent_flow_single_view":
        return PartSSLatentFlowSingleViewDataset
    raise ValueError(
        "eval_part_ss_latent_flow only supports stage "
        f"'part_ss_latent_flow' or 'part_ss_latent_flow_single_view', got {stage!r}"
    )


def _checkpoint_path(args: argparse.Namespace) -> Path:
    if args.ckpt:
        path = Path(args.ckpt)
        if path.is_dir():
            raise FileNotFoundError(
                f"checkpoint must be a .pt FILE, but got a directory: {path}\n"
                f"  - pass the full file, e.g. {path}/ckpts/step_<N>.pt\n"
                f"  - or leave --ckpt empty and use --load-dir {path}/ckpts --step <N>"
            )
        if not path.is_file():
            raise FileNotFoundError(f"checkpoint not found: {path}")
        return path
    if args.load_dir is None or args.step is None:
        raise ValueError("provide either --ckpt PATH or both --load-dir DIR and --step N")
    path = _resolve_resume_checkpoint(args.load_dir, int(args.step))
    if not path.is_file():
        raise FileNotFoundError(f"checkpoint not found: {path}")
    return path


def _load_model(config, ckpt_path: Path, device: torch.device):
    """Build the model from the config EMBEDDED in the checkpoint when present.

    The trainer persists the full training config into ``ckpt['config']``. The
    model architecture MUST match the saved weights, so we build from
    ``ckpt['config']['model']`` rather than the eval YAML — otherwise evaluating a
    checkpoint trained with a different set of flags (e.g. summary-token off)
    fails strict load on the missing keys. Legacy checkpoints without an embedded
    config fall back to the passed config.

    Returns (model, step, ckpt_config_or_None); callers recover the latent-norm
    stats from the returned config via ``_apply_ckpt_latent_norm``.
    """
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    if "model" not in ckpt:
        raise KeyError(f"{ckpt_path} missing key 'model'")
    ckpt_config = ckpt.get("config")
    if isinstance(ckpt_config, dict) and isinstance(ckpt_config.get("model"), dict):
        model_kwargs = dict(ckpt_config["model"])
        eval_model_cfg = _cfg_dict(config.model)
        if bool(dict(eval_model_cfg.get("mask_attention_bias", {})).get("enabled", False)):
            model_kwargs["mask_attention_bias"] = dict(eval_model_cfg["mask_attention_bias"])
    else:
        model_kwargs = _cfg_dict(config.model)
        ckpt_config = None
    # Activation checkpointing is a training-time memory tradeoff. Eval runs under
    # no_grad, so keep the checkpoint architecture/flags but disable recompute.
    model_kwargs["use_checkpoint"] = False
    model = PartSSLatentFlowModel(**model_kwargs).to(device)
    model.load_state_dict(_compatible_model_state_dict(model, ckpt["model"]), strict=True)
    model.eval()
    return model, int(ckpt.get("step", 0)), ckpt_config


def _apply_ckpt_latent_norm(flow_cfg: dict, ckpt_config) -> None:
    """Pull the latent (de)normalization stats from the checkpoint into flow_cfg.

    How latents are normalized (mode + per-channel mean/std + scale) is fixed at
    training time; the sampler must use the SAME stats or it mis-denormalizes. The
    trainer persists them into ``ckpt['config']['flow']``. ``cfg_scale`` / num_steps
    stay eval-time choices (not copied). Mutates flow_cfg in place; no-op when the
    checkpoint has no embedded config (legacy)."""
    if not isinstance(ckpt_config, dict):
        return
    ck_flow = ckpt_config.get("flow")
    if not isinstance(ck_flow, dict):
        return
    for key in ("latent_norm_mode", "latent_mean", "latent_std", "latent_scale"):
        if key in ck_flow and ck_flow[key] is not None:
            flow_cfg[key] = ck_flow[key]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a short single-process latent eval + decode inspection from a Part SS checkpoint."
    )
    parser.add_argument("--config", required=True, help="Part SS Latent Flow YAML config")
    parser.add_argument("--ckpt", default=None, help="Direct checkpoint path, e.g. .../ckpts/step_50000.pt")
    parser.add_argument("--load-dir", default=None, help="Run directory or ckpts directory")
    parser.add_argument("--step", type=int, default=None, help="Checkpoint step used with --load-dir")
    parser.add_argument("--inspection-root", required=True, help="Directory where inspection step_<N>/ is written")
    parser.add_argument("--max-samples", type=int, default=4, help="Number of object samples to decode")
    parser.add_argument(
        "--sample-mode",
        choices=("first", "spread"),
        default="first",
        help="Which dataset samples to inspect: first N, or N samples evenly spread across the dataset.",
    )
    parser.add_argument(
        "--object-ids",
        default=None,
        help="Optional comma-separated object IDs. Sampling is applied after filtering to these objects.",
    )
    parser.add_argument("--num-steps", type=int, default=None, help="Override flow.num_steps for sampling")
    parser.add_argument("--device", default=None, help="cuda, cuda:0, or cpu. Default: cuda if available else cpu")
    parser.add_argument("overrides", nargs="*", help="OmegaConf dotlist overrides, e.g. data.data_root=/path")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cfg = load_config(args.config, overrides=args.overrides)
    stage = str(cfg.get("stage"))
    data_cfg = _cfg_dict(cfg.data)
    flow_cfg = _cfg_dict(cfg.flow)
    eval_cfg = _cfg_dict(cfg.eval)
    eval_cfg["inspection_root"] = args.inspection_root
    eval_cfg["decode_max_samples"] = int(args.max_samples)
    eval_cfg["sample_mode"] = args.sample_mode
    eval_cfg["object_ids"] = args.object_ids
    if args.num_steps is not None:
        flow_cfg["num_steps"] = int(args.num_steps)

    seed = int(getattr(cfg.training, "seed", 42)) if "training" in cfg else 42
    _setup_rng(seed)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    dataset = _dataset_cls(stage)(data_cfg)
    ckpt_path = _checkpoint_path(args)
    model, ckpt_step, ckpt_cfg = _load_model(cfg, ckpt_path, device)
    _apply_ckpt_latent_norm(flow_cfg, ckpt_cfg)

    print("============================================================", flush=True)
    print("Part SS Latent Flow Standalone Eval/Decode", flush=True)
    print(f"  stage:           {stage}", flush=True)
    print(f"  config:          {args.config}", flush=True)
    print(f"  checkpoint:      {ckpt_path}", flush=True)
    print(f"  checkpoint step: {ckpt_step}", flush=True)
    print(f"  device:          {device}", flush=True)
    print(f"  max_samples:     {int(args.max_samples)}", flush=True)
    print(f"  sample_mode:     {args.sample_mode}", flush=True)
    print(f"  object_ids:      {args.object_ids or '<none>'}", flush=True)
    print(f"  num_steps:       {int(flow_cfg.get('num_steps', 20))}", flush=True)
    print(f"  inspection_root: {args.inspection_root}", flush=True)
    print("============================================================", flush=True)

    latent = _eval_latent(
        model,
        dataset,
        device,
        flow_cfg,
        max_samples=int(args.max_samples),
        progress_prefix=f"  [LATENT @ {ckpt_step}]",
        sample_mode=args.sample_mode,
        object_ids=args.object_ids,
    )
    print("[latent]", json.dumps(latent, indent=2, ensure_ascii=False), flush=True)

    index_path, summary = _eval_decode_inspection(
        model,
        dataset,
        device,
        flow_cfg,
        eval_cfg,
        step=ckpt_step,
    )
    print("[decode]", json.dumps(summary, indent=2, ensure_ascii=False), flush=True)
    print(f"[inspection] {index_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
