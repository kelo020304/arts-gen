#!/usr/bin/env python3
"""Diagnose PartMMDiT sampler timestep shift without retraining."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import types
from pathlib import Path
from typing import Any

import torch


TRELLIS_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = TRELLIS_ROOT.parent
if str(TRELLIS_ROOT) not in sys.path:
    sys.path.insert(0, str(TRELLIS_ROOT))

trellis_pkg = types.ModuleType("trellis")
trellis_pkg.__path__ = [str(TRELLIS_ROOT / "trellis")]
trellis_pkg.__package__ = "trellis"
sys.modules.setdefault("trellis", trellis_pkg)
for subpackage in ("datasets", "models", "modules", "trainers", "utils"):
    module = types.ModuleType(f"trellis.{subpackage}")
    module.__path__ = [str(TRELLIS_ROOT / "trellis" / subpackage)]
    module.__package__ = f"trellis.{subpackage}"
    sys.modules.setdefault(f"trellis.{subpackage}", module)

os.environ.setdefault("ATTN_BACKEND", "sdpa")

from trellis.datasets.arts.part_mmdit import PartMMDiTDataset  # noqa: E402
from trellis.models.part_flow import PartMMDiTModel  # noqa: E402
from trellis.trainers.arts.part_mmdit import (  # noqa: E402
    _apply_timestep_shift,
    _dynamic_timestep_shift,
    _to_device,
)
from trellis.utils.arts.config_utils import config_to_dict, load_config  # noqa: E402


DEFAULT_CKPT = (
    "/robot/data-lab/jzh/art-gen/outputs/part_mmdit_v2_memory_gate_quick3/"
    "ckpts/step_2000.pt"
)
DEFAULT_OUT_JSON = (
    "/robot/data-lab/jzh/art-gen/outputs/part_mmdit_v2_memory_gate_quick3/"
    "shift_diagnostic_step2000.json"
)
DEFAULT_OUT_CSV = (
    "/robot/data-lab/jzh/art-gen/outputs/part_mmdit_v2_memory_gate_quick3/"
    "shift_diagnostic_step2000.csv"
)


def _cfg_dict(cfg: Any) -> dict:
    return config_to_dict(cfg) if not isinstance(cfg, dict) else dict(cfg)


def _load_model(ckpt_path: Path, device: torch.device) -> tuple[PartMMDiTModel, dict]:
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    if "config" not in ckpt or "model" not in ckpt:
        raise KeyError(f"{ckpt_path} expected keys 'config' and 'model'")
    model_cfg = _cfg_dict(ckpt["config"]["model"])
    model = PartMMDiTModel(**model_cfg).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, ckpt


def _build_dataset(config: dict) -> PartMMDiTDataset:
    data_cfg = _cfg_dict(config["data"])
    data_cfg["include_obj_ids"] = ["102276", "100015"]
    data_cfg.pop("exclude_obj_ids", None)
    data_cfg.pop("max_samples", None)
    dataset = PartMMDiTDataset(data_cfg)
    by_sample_id = {sample["sample_id"]: idx for idx, sample in enumerate(dataset.samples)}
    wanted = ["physx-mobility_102276_angle_0", "physx-mobility_100015_angle_0"]
    missing = [sample_id for sample_id in wanted if sample_id not in by_sample_id]
    if missing:
        raise RuntimeError(f"missing diagnostic samples: {missing}")
    return dataset, [dataset[by_sample_id[sample_id]] for sample_id in wanted]


def _cos(pred: torch.Tensor, target: torch.Tensor, eps: float = 1.0e-8) -> float:
    pred_flat = pred.detach().float().reshape(-1)
    target_flat = target.detach().float().reshape(-1)
    return float(
        ((pred_flat * target_flat).sum() / (pred_flat.norm() * target_flat.norm()).clamp_min(eps)).item()
    )


def _rel_l2(pred: torch.Tensor, target: torch.Tensor, eps: float = 1.0e-8) -> float:
    pred_flat = pred.detach().float().reshape(-1)
    target_flat = target.detach().float().reshape(-1)
    return float(((pred_flat - target_flat).norm() / target_flat.norm().clamp_min(eps)).item())


def _part_rows(batch: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for obj_row, obj_id in enumerate(batch["obj_id"]):
        valid_k = int(batch["part_valid"][obj_row].sum().item())
        for part_idx, part_name in enumerate(batch["target_part_names"][obj_row][:valid_k]):
            rows.append(
                {
                    "obj_row": int(obj_row),
                    "part_idx": int(part_idx),
                    "obj_id": str(obj_id),
                    "sample_id": str(batch["sample_id"][obj_row]),
                    "part_name": str(part_name),
                }
            )
    return rows


@torch.no_grad()
def _sample_cond_only_scaled(
    model: PartMMDiTModel,
    batch: dict[str, Any],
    *,
    initial_noise: torch.Tensor,
    latent_scale: float,
    num_steps: int,
    timestep_shift: float | None,
) -> torch.Tensor:
    batch_size, part_count = batch["part_valid"].shape
    x = initial_noise.to(device=batch["x_1_parts"].device, dtype=batch["x_1_parts"].dtype)
    valid_view = batch["part_valid"].to(dtype=x.dtype).view(batch_size, part_count, 1, 1, 1, 1)
    x = x * valid_view
    base_grid = torch.linspace(
        0.0,
        1.0,
        int(num_steps) + 1,
        device=x.device,
        dtype=x.dtype,
    )
    if timestep_shift is None or float(timestep_shift) == 0.0:
        t_grid = base_grid
    else:
        t_grid = _apply_timestep_shift(base_grid, shift=float(timestep_shift))
    drop_none = torch.zeros_like(batch["part_valid"].bool())
    for step_idx in range(int(num_steps)):
        t_value = t_grid[step_idx]
        dt = t_grid[step_idx + 1] - t_value
        t = torch.full((batch_size,), t_value, device=x.device, dtype=x.dtype)
        v = model(
            x,
            t,
            batch["z_global"],
            batch["cond"],
            batch["name_tokens"],
            batch["name_mask"],
            batch["anchor"],
            batch["anchor_valid"],
            batch["part_valid"],
            drop_name=drop_none,
            drop_anchor=drop_none,
        )
        x = (x + v * dt) * valid_view
    return x * valid_view


def _sample_metrics(
    pred_scaled: torch.Tensor,
    x1_scaled: torch.Tensor,
    part_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows = []
    for part in part_rows:
        obj_row = part["obj_row"]
        part_idx = part["part_idx"]
        pred = pred_scaled[obj_row, part_idx]
        target = x1_scaled[obj_row, part_idx]
        rows.append(
            {
                **part,
                "cos_sample_x1_scaled": _cos(pred, target),
                "rel_l2_sample_x1_scaled": _rel_l2(pred, target),
            }
        )
    return rows


@torch.no_grad()
def _t_sweep(
    model: PartMMDiTModel,
    batch: dict[str, Any],
    *,
    initial_noise: torch.Tensor,
    latent_scale: float,
    t_values: list[float],
    part_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    x1_scaled = batch["x_1_parts"] * float(latent_scale)
    v_target = x1_scaled - initial_noise
    out = []
    drop_none = torch.zeros_like(batch["part_valid"].bool())
    for t_value in t_values:
        t = torch.full(
            (batch["x_1_parts"].shape[0],),
            float(t_value),
            device=batch["x_1_parts"].device,
            dtype=batch["x_1_parts"].dtype,
        )
        tt = t.view(-1, 1, 1, 1, 1, 1)
        x_t = (1.0 - tt) * initial_noise + tt * x1_scaled
        v_pred = model(
            x_t,
            t,
            batch["z_global"],
            batch["cond"],
            batch["name_tokens"],
            batch["name_mask"],
            batch["anchor"],
            batch["anchor_valid"],
            batch["part_valid"],
            drop_name=drop_none,
            drop_anchor=drop_none,
        )
        for part in part_rows:
            obj_row = part["obj_row"]
            part_idx = part["part_idx"]
            out.append(
                {
                    **part,
                    "t": float(t_value),
                    "rel_l2_v": _rel_l2(
                        v_pred[obj_row, part_idx],
                        v_target[obj_row, part_idx],
                    ),
                }
            )
    return out


def _write_csv(path: Path, result: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["section", "case", "obj_id", "part_name", "t", "cos", "rel_l2"])
        for case in result["test_a"]:
            for row in case["parts"]:
                writer.writerow(
                    [
                        "A",
                        case["case"],
                        row["obj_id"],
                        row["part_name"],
                        "",
                        row["cos_sample_x1_scaled"],
                        row["rel_l2_sample_x1_scaled"],
                    ]
                )
        for row in result["test_b"]:
            writer.writerow(
                [
                    "B",
                    "t_sweep",
                    row["obj_id"],
                    row["part_name"],
                    row["t"],
                    "",
                    row["rel_l2_v"],
                ]
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", default=DEFAULT_CKPT)
    parser.add_argument("--out-json", default=DEFAULT_OUT_JSON)
    parser.add_argument("--out-csv", default=DEFAULT_OUT_CSV)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    ckpt_path = Path(args.ckpt)
    model, ckpt = _load_model(ckpt_path, device)
    config = ckpt["config"]
    flow_cfg = _cfg_dict(config["flow"])
    latent_scale = float(flow_cfg.get("latent_scale", 8.0))
    dynamic_shift = _dynamic_timestep_shift(model, flow_cfg)

    dataset, items = _build_dataset(config)
    batch = dataset.collate_fn(items)
    batch = _to_device(batch, device)
    part_rows = _part_rows(batch)

    generator = torch.Generator(device=device)
    generator.manual_seed(int(args.seed))
    initial_noise = torch.randn(
        batch["x_1_parts"].shape,
        generator=generator,
        device=device,
        dtype=batch["x_1_parts"].dtype,
    )
    x1_scaled = batch["x_1_parts"] * latent_scale

    test_a = []
    for case_name, shift, steps in (
        ("A1_dynamic_shift_10", dynamic_shift, 10),
        ("A2_uniform_shift0_20", 0.0, 20),
        ("A3_uniform_shift0_50", 0.0, 50),
    ):
        pred = _sample_cond_only_scaled(
            model,
            batch,
            initial_noise=initial_noise,
            latent_scale=latent_scale,
            num_steps=steps,
            timestep_shift=shift,
        )
        test_a.append(
            {
                "case": case_name,
                "timestep_shift": float(shift),
                "num_steps": int(steps),
                "parts": _sample_metrics(pred, x1_scaled, part_rows),
            }
        )

    test_b = _t_sweep(
        model,
        batch,
        initial_noise=initial_noise,
        latent_scale=latent_scale,
        t_values=[0.02, 0.1, 0.3, 0.5, 0.7, 0.9],
        part_rows=part_rows,
    )
    result = {
        "checkpoint": str(ckpt_path),
        "checkpoint_step": int(ckpt.get("step", -1)),
        "seed": int(args.seed),
        "latent_scale": latent_scale,
        "dynamic_shift": float(dynamic_shift),
        "samples": [
            {
                "obj_id": str(obj_id),
                "sample_id": str(sample_id),
                "part_names": list(part_names),
            }
            for obj_id, sample_id, part_names in zip(
                batch["obj_id"],
                batch["sample_id"],
                batch["target_part_names"],
            )
        ],
        "test_a": test_a,
        "test_b": test_b,
    }
    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(result, indent=2), encoding="utf-8")
    _write_csv(Path(args.out_csv), result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
