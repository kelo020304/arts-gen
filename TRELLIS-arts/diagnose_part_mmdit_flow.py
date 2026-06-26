#!/usr/bin/env python3
"""Cheap PartMMDiT RF field diagnostics for a fixed object/part.

Runs without optimizer steps:
  1. Single-step endpoint scan at fixed t values.
  2. Multi-step sampler cosine / relative L2 at selected Euler step counts.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import types
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRELLIS_PATH = PROJECT_ROOT / "TRELLIS-arts"
if str(TRELLIS_PATH) not in sys.path:
    sys.path.insert(0, str(TRELLIS_PATH))

_pkg = types.ModuleType("trellis")
_pkg.__path__ = [str(TRELLIS_PATH / "trellis")]
_pkg.__package__ = "trellis"
sys.modules.setdefault("trellis", _pkg)
for _subpackage in ("models", "modules", "trainers", "utils", "datasets"):
    _module = types.ModuleType(f"trellis.{_subpackage}")
    _module.__path__ = [str(TRELLIS_PATH / "trellis" / _subpackage)]
    _module.__package__ = f"trellis.{_subpackage}"
    sys.modules.setdefault(f"trellis.{_subpackage}", _module)

os.environ.setdefault("ATTN_BACKEND", "sdpa")

from trellis.datasets.arts.part_mmdit import PartMMDiTDataset  # noqa: E402
from trellis.models.part_flow import PartMMDiTModel  # noqa: E402
from trellis.trainers.arts.part_mmdit import (  # noqa: E402
    compute_part_mmdit_latent_alignment,
    _dynamic_timestep_shift,
    sample_part_mmdit_latent,
)
from trellis.utils.arts.config_utils import config_to_dict, load_config  # noqa: E402


def _cfg_dict(cfg: Any) -> dict:
    return config_to_dict(cfg) if not isinstance(cfg, dict) else dict(cfg)


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _load_ckpt_config(ckpt_path: Path) -> dict:
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if "config" not in ckpt:
        raise KeyError(f"{ckpt_path} missing checkpoint key 'config'")
    if "model" not in ckpt:
        raise KeyError(f"{ckpt_path} missing checkpoint key 'model'")
    return ckpt


def _load_model(ckpt: dict, model_cfg: dict, device: torch.device) -> PartMMDiTModel:
    model = PartMMDiTModel(**model_cfg).to(device)
    missing, unexpected = model.load_state_dict(ckpt["model"], strict=False)
    if missing or unexpected:
        raise RuntimeError(
            f"checkpoint/model key mismatch: missing={missing[:10]} unexpected={unexpected[:10]}"
        )
    model.eval()
    return model


def _build_dataset(data_cfg: dict, obj_id: str) -> PartMMDiTDataset:
    data_cfg = dict(data_cfg)
    data_cfg["include_obj_ids"] = [str(obj_id)]
    data_cfg.pop("exclude_obj_ids", None)
    data_cfg["max_samples"] = 1
    return PartMMDiTDataset(data_cfg)


def _to_device(batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for key, value in batch.items():
        out[key] = value.to(device, non_blocking=True) if isinstance(value, torch.Tensor) else value
    return out


def _part_index(batch: Dict[str, Any], part_name: str) -> int:
    names = list(batch["target_part_names"][0])
    if part_name not in names:
        raise ValueError(f"part_name={part_name!r} not found in sample parts: {names}")
    return int(names.index(part_name))


def _rel_l2(pred: torch.Tensor, target: torch.Tensor, eps: float = 1.0e-8) -> float:
    pred_flat = pred.detach().float().reshape(-1)
    target_flat = target.detach().float().reshape(-1)
    return float(((pred_flat - target_flat).norm() / target_flat.norm().clamp_min(eps)).item())


def _cos(pred: torch.Tensor, target: torch.Tensor, eps: float = 1.0e-8) -> float:
    pred_flat = pred.detach().float().reshape(-1)
    target_flat = target.detach().float().reshape(-1)
    return float(
        ((pred_flat * target_flat).sum() / (pred_flat.norm() * target_flat.norm()).clamp_min(eps)).item()
    )


@torch.no_grad()
def run_diagnostic(
    *,
    ckpt_path: Path,
    config_path: Path | None,
    obj_id: str,
    part_name: str,
    t_values: list[float],
    sampler_steps: list[int],
    seed: int,
    noise_seed: int,
    device: torch.device,
) -> dict:
    ckpt = _load_ckpt_config(ckpt_path)
    if config_path is not None:
        cfg = load_config(str(config_path))
        cfg_dict = config_to_dict(cfg)
    else:
        cfg_dict = ckpt["config"]

    model_cfg = _cfg_dict(cfg_dict["model"])
    data_cfg = _cfg_dict(cfg_dict["data"])
    flow_cfg = _cfg_dict(cfg_dict["flow"])
    model = _load_model(ckpt, model_cfg, device)

    dataset = _build_dataset(data_cfg, obj_id)
    batch = dataset.collate_fn([dataset[0]])
    part_idx = _part_index(batch, part_name)
    batch = _to_device(batch, device)

    latent_scale = float(flow_cfg.get("latent_scale", 8.0))
    x1_raw = batch["x_1_parts"][:, part_idx : part_idx + 1]
    x1_train = x1_raw * latent_scale
    part_valid = batch["part_valid"][:, part_idx : part_idx + 1]
    anchor = batch["anchor"][:, part_idx : part_idx + 1]
    anchor_valid = batch["anchor_valid"][:, part_idx : part_idx + 1]
    name_tokens = batch["name_tokens"][:, part_idx : part_idx + 1]
    name_mask = batch["name_mask"][:, part_idx : part_idx + 1]

    generator = torch.Generator(device=device)
    generator.manual_seed(int(noise_seed))
    noise = torch.randn(x1_train.shape, generator=generator, device=device, dtype=x1_train.dtype)

    t_scan = []
    for t_value in t_values:
        t = torch.full((1,), float(t_value), device=device, dtype=x1_train.dtype)
        t_view = t.view(1, 1, 1, 1, 1, 1)
        x_t = (1.0 - t_view) * noise + t_view * x1_train
        v_pred = model(
            x_t,
            t,
            batch["z_global"],
            batch["cond"],
            name_tokens,
            name_mask,
            anchor,
            anchor_valid,
            part_valid,
            drop_name=torch.zeros_like(part_valid),
            drop_anchor=torch.zeros_like(part_valid),
        )
        x_hat_train = x_t + (1.0 - t_view) * v_pred
        t_scan.append(
            {
                "t": float(t_value),
                "rel_l2_train_space": _rel_l2(x_hat_train, x1_train),
                "cos_train_space": _cos(x_hat_train, x1_train),
            }
        )

    sampler_scan = []
    for steps in sampler_steps:
        generator.manual_seed(int(noise_seed))
        initial_noise = torch.randn(
            x1_train.shape,
            generator=generator,
            device=device,
            dtype=x1_train.dtype,
        )
        pred = sample_part_mmdit_latent(
            model,
            z_global=batch["z_global"],
            cond=batch["cond"],
            name_tokens=name_tokens,
            name_mask=name_mask,
            anchor=anchor,
            anchor_valid=anchor_valid,
            part_valid=part_valid,
            initial_noise=initial_noise,
            num_steps=int(steps),
            noise_scale=float(flow_cfg.get("noise_scale", 1.0)),
            latent_scale=latent_scale,
            s_name=float(flow_cfg.get("s_name", 1.0)),
            s_anchor=float(flow_cfg.get("s_anchor", 1.0)),
            timestep_shift=_dynamic_timestep_shift(model, flow_cfg),
        )
        latent_metrics = compute_part_mmdit_latent_alignment(pred, x1_raw, part_valid)
        sampler_scan.append(
            {
                "num_steps": int(steps),
                "cos_sample_x1": latent_metrics["latent_cos"],
                "rel_l2_sample_x1": latent_metrics["latent_rel_l2"],
            }
        )

    return {
        "checkpoint": str(ckpt_path),
        "checkpoint_step": int(ckpt.get("step", -1)),
        "obj_id": str(obj_id),
        "part_name": str(part_name),
        "part_idx": int(part_idx),
        "sample_id": batch["sample_id"][0],
        "target_part_names": list(batch["target_part_names"][0]),
        "latent_scale": latent_scale,
        "seed": int(seed),
        "noise_seed": int(noise_seed),
        "t_scan": t_scan,
        "sampler_scan": sampler_scan,
    }


def _write_markdown(path: Path, result: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as out:
        out.write("# PartMMDiT Flow Diagnostic\n\n")
        out.write(f"- checkpoint: `{result['checkpoint']}`\n")
        out.write(f"- checkpoint_step: `{result['checkpoint_step']}`\n")
        out.write(f"- obj_id/part: `{result['obj_id']}` / `{result['part_name']}`\n")
        out.write(f"- sample_id: `{result['sample_id']}`\n")
        out.write(f"- latent_scale: `{result['latent_scale']}`\n\n")
        out.write("## Single-Step T Scan\n\n")
        out.write("| t | rel_l2_train_space | cos_train_space |\n")
        out.write("|---:|---:|---:|\n")
        for row in result["t_scan"]:
            out.write(
                f"| {row['t']:.2f} | {row['rel_l2_train_space']:.6f} | "
                f"{row['cos_train_space']:.6f} |\n"
            )
        out.write("\n## Sampler Step Scan\n\n")
        out.write("| num_steps | cos_sample_x1 | rel_l2_sample_x1 |\n")
        out.write("|---:|---:|---:|\n")
        for row in result["sampler_scan"]:
            out.write(
                f"| {row['num_steps']} | {row['cos_sample_x1']:.6f} | "
                f"{row['rel_l2_sample_x1']:.6f} |\n"
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose PartMMDiT RF field quality")
    parser.add_argument(
        "--ckpt",
        default="/robot/data-lab/jzh/art-gen/outputs/part_mmdit_train_v1/ckpts/step_2400.pt",
        help="PartMMDiT checkpoint path",
    )
    parser.add_argument("--config", default=None, help="Optional YAML config; defaults to ckpt['config']")
    parser.add_argument("--obj-id", default="102276")
    parser.add_argument("--part-name", default="wheel_0")
    parser.add_argument("--t-values", nargs="+", type=float, default=[0.1, 0.3, 0.5, 0.7, 0.9])
    parser.add_argument("--sampler-steps", nargs="+", type=int, default=[20, 100, 250])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--noise-seed", type=int, default=1234)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--out-json",
        default="code_update/part_mmdit_flow_diagnostic_102276_wheel_0_step2400.json",
    )
    parser.add_argument(
        "--out-md",
        default="code_update/part_mmdit_flow_diagnostic_102276_wheel_0_step2400.md",
    )
    args = parser.parse_args()

    _set_seed(args.seed)
    result = run_diagnostic(
        ckpt_path=Path(args.ckpt),
        config_path=Path(args.config) if args.config else None,
        obj_id=str(args.obj_id),
        part_name=str(args.part_name),
        t_values=[float(t) for t in args.t_values],
        sampler_steps=[int(steps) for steps in args.sampler_steps],
        seed=int(args.seed),
        noise_seed=int(args.noise_seed),
        device=torch.device(args.device),
    )

    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(result, indent=2), encoding="utf-8")
    _write_markdown(Path(args.out_md), result)

    print(json.dumps(result, indent=2), flush=True)
    print(f"[done] wrote {out_json} and {args.out_md}", flush=True)


if __name__ == "__main__":
    main()
