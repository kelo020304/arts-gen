#!/usr/bin/env python3
"""Run a small PartMMDiT v2 memory-gate diagnostic.

This is a diagnostic overfit run, not a production trainer. It uses the same
PartMMDiT dataset/model/loss/timestep contract as the trainer, but evaluates a
clean conditional sampler with name+anchor enabled in a single forward path.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import sys
import types
from pathlib import Path
from typing import Any

import numpy as np


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

os.environ.setdefault("TORCH_HOME", str(PROJECT_ROOT / "submodules" / "TRELLIS.1"))
os.environ.setdefault("ATTN_BACKEND", "sdpa")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch  # noqa: E402
import torch.distributed as dist  # noqa: E402
from torch.nn.parallel import DistributedDataParallel as DDP  # noqa: E402

from trellis.datasets.arts.part_mmdit import PartMMDiTDataset  # noqa: E402
from trellis.models.part_flow import PartMMDiTModel  # noqa: E402
from trellis.trainers.arts.part_mmdit import (  # noqa: E402
    _apply_timestep_shift,
    _dynamic_timestep_shift,
    _loss_kwargs,
    _sampler_timestep_shift,
    _to_device,
)
from trellis.trainers.arts.part_mmdit_losses import rectified_flow_loss  # noqa: E402
from trellis.trainers.arts.part_ss_latent_flow_losses import sample_flow_timesteps  # noqa: E402
from trellis.utils.arts.config_utils import config_to_dict, load_config  # noqa: E402


DATA_ROOT = (
    "/robot/data-lab/jzh/art-gen/data/PhysX-Mobility-full-4view-0511/"
    "PhysX-Mobility-full-4view-0511"
)
DEFAULT_CONFIG = TRELLIS_ROOT / "configs/arts/part_mmdit/medium_128_logit_normal.yaml"
DEFAULT_OUTPUT_DIR = (
    "/robot/data-lab/jzh/art-gen/outputs/part_mmdit_v2_memory_gate"
)
DEFAULT_CODE_UPDATE = TRELLIS_ROOT / "code_update/code_update_part_mmdit.md"

MEMORY_SAMPLES = [
    {
        "group": "multi",
        "obj_id": "102276",
        "sample_id": "physx-mobility_102276_angle_0",
        "angle_idx": 0,
    },
    {
        "group": "multi",
        "obj_id": "100194",
        "sample_id": "physx-mobility_100194_angle_0",
        "angle_idx": 0,
    },
    {
        "group": "multi",
        "obj_id": "100279",
        "sample_id": "physx-mobility_100279_angle_0",
        "angle_idx": 0,
    },
    {
        "group": "multi",
        "obj_id": "101564",
        "sample_id": "physx-mobility_101564_angle_0",
        "angle_idx": 0,
    },
    {
        "group": "multi",
        "obj_id": "100405",
        "sample_id": "physx-mobility_100405_angle_0",
        "angle_idx": 0,
    },
    {
        "group": "multi",
        "obj_id": "101591",
        "sample_id": "physx-mobility_101591_angle_0",
        "angle_idx": 0,
    },
    {
        "group": "single",
        "obj_id": "100015",
        "sample_id": "physx-mobility_100015_angle_0",
        "angle_idx": 0,
    },
    {
        "group": "single",
        "obj_id": "100021",
        "sample_id": "physx-mobility_100021_angle_0",
        "angle_idx": 0,
    },
    {
        "group": "single",
        "obj_id": "100033",
        "sample_id": "physx-mobility_100033_angle_0",
        "angle_idx": 0,
    },
    {
        "group": "single",
        "obj_id": "100038",
        "sample_id": "physx-mobility_100038_angle_0",
        "angle_idx": 0,
    },
]

QUICK3_MEMORY_SAMPLES = [
    {
        "group": "multi",
        "obj_id": "102276",
        "sample_id": "physx-mobility_102276_angle_0",
        "angle_idx": 0,
    },
    {
        "group": "multi",
        "obj_id": "100194",
        "sample_id": "physx-mobility_100194_angle_0",
        "angle_idx": 0,
    },
    {
        "group": "single",
        "obj_id": "100015",
        "sample_id": "physx-mobility_100015_angle_0",
        "angle_idx": 0,
    },
]


def _setup_dist() -> tuple[int, int, int]:
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size > 1:
        if not torch.cuda.is_available():
            raise RuntimeError("torchrun world_size > 1 requires CUDA")
        torch.cuda.set_device(local_rank)
        dist.init_process_group("nccl", rank=rank, world_size=world_size)
    elif torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    return rank, local_rank, world_size


def _cleanup_dist() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def _setup_rng(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _as_dict(cfg_node: Any) -> dict:
    return config_to_dict(cfg_node) if not isinstance(cfg_node, dict) else dict(cfg_node)


def _find_exact_samples(
    dataset: PartMMDiTDataset,
    memory_samples: list[dict[str, Any]],
) -> list[int]:
    by_sample_id = {sample["sample_id"]: idx for idx, sample in enumerate(dataset.samples)}
    indices = []
    missing = []
    for wanted in memory_samples:
        sample_id = wanted["sample_id"]
        idx = by_sample_id.get(sample_id)
        if idx is None:
            missing.append(sample_id)
            continue
        sample = dataset.samples[idx]
        if str(sample["obj_id"]) != wanted["obj_id"]:
            raise ValueError(
                f"{sample_id} expected obj_id={wanted['obj_id']}, got {sample['obj_id']}"
            )
        if int(sample["angle_idx"]) != int(wanted["angle_idx"]):
            raise ValueError(
                f"{sample_id} expected angle_idx={wanted['angle_idx']}, got {sample['angle_idx']}"
            )
        indices.append(idx)
    if missing:
        raise RuntimeError(f"memory samples missing from dataset: {missing}")
    if len(indices) != len(set(indices)):
        raise RuntimeError(f"duplicate dataset indices selected: {indices}")
    return indices


def _build_memory_batch(
    dataset: PartMMDiTDataset,
    memory_samples: list[dict[str, Any]],
    *,
    min_multi: int,
    min_single: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    indices = _find_exact_samples(dataset, memory_samples)
    items = [dataset[idx] for idx in indices]
    batch = dataset.collate_fn(items)
    metadata = []
    sample_group = {item["sample_id"]: item["group"] for item in memory_samples}
    for row, item in enumerate(items):
        valid_k = int(item["part_valid"].sum().item())
        counts = [int(x) for x in item["part_raw_voxel_counts"].tolist()]
        metadata.append(
            {
                "row": row,
                "obj_id": item["obj_id"],
                "sample_id": item["sample_id"],
                "angle_idx": int(item["angle_idx"]),
                "group": sample_group[item["sample_id"]],
                "part_count": valid_k,
                "part_names": list(item["target_part_names"]),
                "part_types": list(item["target_part_types"]),
                "part_raw_voxel_counts": counts,
                "has_small_lt500": any(count < 500 for count in counts),
            }
        )
    _validate_memory_selection(metadata, min_multi=min_multi, min_single=min_single)
    return batch, metadata


def _validate_memory_selection(
    metadata: list[dict[str, Any]],
    *,
    min_multi: int,
    min_single: int,
) -> None:
    obj_ids = {item["obj_id"] for item in metadata}
    if len(obj_ids) != len(metadata):
        raise RuntimeError(f"expected one sample per obj_id, got obj_ids={sorted(obj_ids)}")
    by_obj = {item["obj_id"]: item for item in metadata}
    if "102276" not in by_obj:
        raise RuntimeError("required obj_id=102276 is missing")
    hard = by_obj["102276"]
    buttons = [name for name in hard["part_names"] if name.startswith("button_")]
    if len(buttons) != 6 or "wheel_0" not in hard["part_names"]:
        raise RuntimeError(
            "102276 must contain 6 button_* parts and wheel_0, got "
            f"{hard['part_names']}"
        )
    multi = [item for item in metadata if item["group"] == "multi"]
    single = [item for item in metadata if item["group"] == "single"]
    if len(multi) < int(min_multi):
        raise RuntimeError(
            f"expected at least {int(min_multi)} multi-part objects, got {len(multi)}"
        )
    if len(single) < int(min_single):
        raise RuntimeError(
            f"expected at least {int(min_single)} single-part objects, got {len(single)}"
        )
    bad_multi = [
        item["obj_id"]
        for item in multi
        if int(item["part_count"]) < 6 or not bool(item["has_small_lt500"])
    ]
    if bad_multi:
        raise RuntimeError(
            "multi-part memory objects must have >=6 parts and raw voxel <500: "
            f"{bad_multi}"
        )
    bad_single = [item["obj_id"] for item in single if int(item["part_count"]) != 1]
    if bad_single:
        raise RuntimeError(f"single-part memory objects must have K=1: {bad_single}")


def _static_shard_batch(batch: dict[str, Any], metadata: list[dict[str, Any]], rank: int, world_size: int) -> dict[str, Any]:
    if world_size <= 1:
        return batch
    rows = [idx for idx in range(len(metadata)) if idx % world_size == rank]
    if not rows:
        raise RuntimeError(f"rank {rank} got empty static shard for {len(metadata)} samples")
    return _slice_batch_rows(batch, rows)


def _slice_batch_rows(batch: dict[str, Any], rows: list[int]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    row_tensor = torch.tensor(rows, dtype=torch.long)
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            out[key] = value.index_select(0, row_tensor)
        elif isinstance(value, list):
            out[key] = [value[idx] for idx in rows]
        else:
            out[key] = value
    return out


def _move_float_batch_dtype(batch: dict[str, Any], dtype: torch.dtype) -> dict[str, Any]:
    out = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor) and torch.is_floating_point(value):
            out[key] = value.to(dtype=dtype)
        else:
            out[key] = value
    return out


def _make_model_cfg(base_cfg: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    model_cfg = dict(base_cfg)
    if args.use_checkpoint is not None:
        model_cfg["use_checkpoint"] = bool(args.use_checkpoint)
    return model_cfg


def _make_flow_cfg(base_cfg: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    flow_cfg = dict(base_cfg)
    flow_cfg["latent_scale"] = float(args.latent_scale)
    flow_cfg["num_steps"] = int(args.num_steps)
    flow_cfg["s_name"] = 1.0
    flow_cfg["s_anchor"] = 1.0
    return flow_cfg


def _make_loss_cfg(base_cfg: dict[str, Any]) -> dict[str, Any]:
    loss_cfg = dict(base_cfg)
    return loss_cfg


def _lr_for_step(step: int, *, max_steps: int, warmup_steps: int) -> float:
    step = max(1, int(step))
    if step <= int(warmup_steps):
        return step / float(max(1, int(warmup_steps)))
    progress = min(1.0, (step - int(warmup_steps)) / float(max(1, int(max_steps) - int(warmup_steps))))
    return 0.5 * (1.0 + math.cos(math.pi * progress))


@torch.no_grad()
def _sample_cond_only_scaled(
    model: torch.nn.Module,
    *,
    z_global: torch.Tensor,
    cond: torch.Tensor,
    name_tokens: torch.Tensor,
    name_mask: torch.Tensor,
    anchor: torch.Tensor,
    anchor_valid: torch.Tensor,
    part_valid: torch.Tensor,
    initial_noise: torch.Tensor,
    num_steps: int,
    timestep_shift: float,
) -> torch.Tensor:
    if int(num_steps) <= 0:
        raise ValueError(f"num_steps must be > 0, got {num_steps}")
    raw_model = model.module if isinstance(model, DDP) else model
    raw_model.eval()
    batch_size, part_count = part_valid.shape
    expected_shape = tuple(part_valid.shape) + tuple(z_global.shape[1:])
    if tuple(initial_noise.shape) != expected_shape:
        raise ValueError(
            f"initial_noise shape {tuple(initial_noise.shape)} != {expected_shape}"
        )
    x = initial_noise.to(device=z_global.device, dtype=z_global.dtype)
    valid_view = part_valid.to(device=z_global.device, dtype=z_global.dtype).view(
        batch_size,
        part_count,
        1,
        1,
        1,
        1,
    )
    x = x * valid_view
    base_grid = torch.linspace(
        0.0,
        1.0,
        int(num_steps) + 1,
        device=z_global.device,
        dtype=z_global.dtype,
    )
    t_grid = _apply_timestep_shift(base_grid, shift=float(timestep_shift))
    drop_none = torch.zeros_like(part_valid.bool(), device=z_global.device)
    for step_idx in range(int(num_steps)):
        t_value = t_grid[step_idx]
        dt = t_grid[step_idx + 1] - t_value
        t = torch.full((batch_size,), t_value, device=z_global.device, dtype=z_global.dtype)
        v = raw_model(
            x,
            t,
            z_global,
            cond,
            name_tokens,
            name_mask,
            anchor,
            anchor_valid,
            part_valid,
            drop_name=drop_none,
            drop_anchor=drop_none,
        )
        x = (x + v * dt) * valid_view
    raw_model.train()
    return x * valid_view


def _per_part_alignment(
    pred_scaled: torch.Tensor,
    target_scaled: torch.Tensor,
    part_valid: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if pred_scaled.shape != target_scaled.shape:
        raise ValueError(
            f"pred shape {tuple(pred_scaled.shape)} != target {tuple(target_scaled.shape)}"
        )
    if tuple(part_valid.shape) != tuple(pred_scaled.shape[:2]):
        raise ValueError(
            f"part_valid shape {tuple(part_valid.shape)} != pred[:2] {tuple(pred_scaled.shape[:2])}"
        )
    pred_flat = pred_scaled.detach().float().flatten(start_dim=2)
    target_flat = target_scaled.detach().float().flatten(start_dim=2)
    dot = (pred_flat * target_flat).sum(dim=2)
    denom = pred_flat.norm(dim=2) * target_flat.norm(dim=2)
    cos = dot / denom.clamp_min(1.0e-8)
    rel_l2 = (pred_flat - target_flat).norm(dim=2) / target_flat.norm(dim=2).clamp_min(1.0e-8)
    valid = part_valid.bool()
    cos = torch.where(valid, cos, torch.full_like(cos, float("nan")))
    rel_l2 = torch.where(valid, rel_l2, torch.full_like(rel_l2, float("nan")))
    return cos.cpu(), rel_l2.cpu()


def _mean_selected(values: torch.Tensor, mask: torch.Tensor) -> float:
    selected = values[mask]
    if selected.numel() == 0:
        return float("nan")
    return float(selected.float().mean().item())


def _eval_memory_gate(
    model: torch.nn.Module,
    batch: dict[str, Any],
    metadata: list[dict[str, Any]],
    *,
    device: torch.device,
    flow_cfg: dict[str, Any],
    eval_noise: torch.Tensor,
) -> dict[str, float]:
    device_batch = _to_device(batch, device)
    pred_scaled = _sample_cond_only_scaled(
        model,
        z_global=device_batch["z_global"],
        cond=device_batch["cond"],
        name_tokens=device_batch["name_tokens"],
        name_mask=device_batch["name_mask"],
        anchor=device_batch["anchor"],
        anchor_valid=device_batch["anchor_valid"],
        part_valid=device_batch["part_valid"],
        initial_noise=eval_noise,
        num_steps=int(flow_cfg.get("num_steps", 20)),
        timestep_shift=_sampler_timestep_shift(
            model.module if isinstance(model, DDP) else model,
            flow_cfg,
        ),
    )
    target_scaled = device_batch["x_1_parts"] * float(flow_cfg["latent_scale"])
    cos, rel_l2 = _per_part_alignment(
        pred_scaled,
        target_scaled,
        device_batch["part_valid"],
    )

    single_mask = torch.zeros_like(cos, dtype=torch.bool)
    multi_mask = torch.zeros_like(cos, dtype=torch.bool)
    buttons_mask = torch.zeros_like(cos, dtype=torch.bool)
    part_valid_cpu = batch["part_valid"].bool().cpu()
    for row, item in enumerate(metadata):
        valid_k = int(part_valid_cpu[row].sum().item())
        if item["group"] == "single":
            single_mask[row, :valid_k] = True
        elif item["group"] == "multi":
            multi_mask[row, :valid_k] = True
        else:
            raise ValueError(f"unknown group={item['group']!r}")
        if item["obj_id"] == "102276":
            for part_idx, part_name in enumerate(item["part_names"]):
                if part_name.startswith("button_"):
                    buttons_mask[row, part_idx] = True

    return {
        "cos_single": _mean_selected(cos, single_mask),
        "cos_multi": _mean_selected(cos, multi_mask),
        "cos_102276_buttons": _mean_selected(cos, buttons_mask),
        "rel_l2_multi": _mean_selected(rel_l2, multi_mask),
    }


def _write_csv_row(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "step",
        "train_loss",
        "cos_single",
        "cos_multi",
        "cos_102276_buttons",
    ]
    exists = path.is_file()
    with path.open("a", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        writer.writerow({name: row.get(name) for name in fieldnames})


def _format_float(value: float) -> str:
    if isinstance(value, float) and math.isnan(value):
        return "nan"
    return f"{float(value):.6f}"


def _append_memory_log(
    path: Path,
    *,
    title: str,
    metadata: list[dict[str, Any]],
    output_dir: Path,
    args: argparse.Namespace,
    config: dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        path.write_text("# PartMMDiT Code Update Log\n", encoding="utf-8")
    lines = [
        "",
        f"## {title}",
        "",
        "目标：",
        f"- v2 PartMMDiT 对齐采样器 memory-gate 诊断；不启用 cos 贴 0 早停，目标跑满 {args.max_steps} step。",
        "- 采样评估使用单路 clean conditional：`drop_name=False`、`drop_anchor=False`，不做 CFG 外插。",
        "",
        "选用 memory samples：",
    ]
    for item in metadata:
        lines.append(
            "- "
            f"{item['obj_id']} / {item['sample_id']} / group={item['group']} / "
            f"K={item['part_count']} / counts={item['part_raw_voxel_counts']} / "
            f"parts={item['part_names']}"
        )
    lines.extend(
        [
            "",
            "实际配置：",
            f"- output_dir: `{output_dir}`",
            f"- max_steps: `{args.max_steps}`",
            f"- warmup_steps: `{args.warmup_steps}`",
            f"- lr: `{args.lr}`",
            f"- fp16: `{args.fp16}`",
            f"- checkpoint_every: `{args.checkpoint_every}`",
            f"- eval_every: `{args.eval_every}`",
            f"- batch: `{len(metadata)}` object samples / full memory set per optimizer step",
            f"- flow.num_steps: `{args.num_steps}`",
            f"- flow.latent_scale: `{args.latent_scale}`",
            f"- model.use_checkpoint: `{config['model'].get('use_checkpoint')}`",
            "",
        ]
    )
    with path.open("a", encoding="utf-8") as update_file:
        update_file.write("\n".join(lines) + "\n")


def _append_result_log(
    path: Path,
    *,
    title: str,
    rows: list[dict[str, Any]],
    final_checkpoint: Path,
    output_dir: Path,
) -> None:
    lines = [
        "",
        f"## {title}",
        "",
        f"- output_dir: `{output_dir}`",
        f"- final_checkpoint: `{final_checkpoint}`",
        "",
        "| step | train_loss | cos_single | cos_multi | cos_102276_buttons |",
        "|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {int(row['step'])} | "
            f"{_format_float(float(row['train_loss']))} | "
            f"{_format_float(float(row['cos_single']))} | "
            f"{_format_float(float(row['cos_multi']))} | "
            f"{_format_float(float(row['cos_102276_buttons']))} |"
        )
    with path.open("a", encoding="utf-8") as update_file:
        update_file.write("\n".join(lines) + "\n")


def _save_checkpoint(
    path: Path,
    *,
    step: int,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    config: dict[str, Any],
) -> None:
    raw_model = model.module if isinstance(model, DDP) else model
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "step": int(step),
            "model": raw_model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "config": config,
        },
        path,
    )


def _load_resume(
    ckpt_path: Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    device: torch.device,
) -> int:
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"resume checkpoint not found: {ckpt_path}")
    raw_model = model.module if isinstance(model, DDP) else model
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    raw_model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    if ckpt.get("scheduler") is not None:
        scheduler.load_state_dict(ckpt["scheduler"])
    return int(ckpt["step"])


def _read_existing_metrics(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows = []
    with path.open("r", newline="", encoding="utf-8") as csv_file:
        for row in csv.DictReader(csv_file):
            rows.append(
                {
                    "step": int(row["step"]),
                    "train_loss": float(row["train_loss"]),
                    "cos_single": float(row["cos_single"]),
                    "cos_multi": float(row["cos_multi"]),
                    "cos_102276_buttons": float(row["cos_102276_buttons"]),
                    "rel_l2_multi": float(row["rel_l2_multi"])
                    if "rel_l2_multi" in row and row["rel_l2_multi"] not in ("", None)
                    else float("nan"),
                }
            )
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--preset",
        choices=("full10", "quick3"),
        default="full10",
        help="full10 uses the original 10-object set; quick3 uses the faster 3-object set.",
    )
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--code-update", default=str(DEFAULT_CODE_UPDATE))
    parser.add_argument("--max-steps", type=int, default=5000)
    parser.add_argument("--warmup-steps", type=int, default=200)
    parser.add_argument("--lr", type=float, default=2.0e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--fp16", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--checkpoint-every", type=int, default=500)
    parser.add_argument("--eval-every", type=int, default=250)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-steps", type=int, default=20)
    parser.add_argument("--latent-scale", type=float, default=8.0)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--dry-run-steps", type=int, default=0)
    parser.add_argument(
        "--use-checkpoint",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Override model.use_checkpoint from the config.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    memory_samples = QUICK3_MEMORY_SAMPLES if args.preset == "quick3" else MEMORY_SAMPLES
    if args.max_steps <= 0:
        raise ValueError(f"max_steps must be > 0, got {args.max_steps}")
    if args.eval_every <= 0:
        raise ValueError(f"eval_every must be > 0, got {args.eval_every}")
    if args.checkpoint_every <= 0:
        raise ValueError(f"checkpoint_every must be > 0, got {args.checkpoint_every}")

    rank, local_rank, world_size = _setup_dist()
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    is_main = rank == 0
    _setup_rng(args.seed + rank)

    cfg = load_config(args.config)
    data_cfg = _as_dict(cfg.data)
    data_cfg["data_root"] = DATA_ROOT
    data_cfg["include_obj_ids"] = [item["obj_id"] for item in memory_samples]
    data_cfg.pop("exclude_obj_ids", None)
    data_cfg.pop("max_samples", None)

    model_cfg = _make_model_cfg(_as_dict(cfg.model), args)
    flow_cfg = _make_flow_cfg(_as_dict(cfg.flow), args)
    loss_cfg = _make_loss_cfg(_as_dict(cfg.loss) if "loss" in cfg else {})
    train_cfg = {
        "seed": args.seed,
        "max_steps": args.max_steps,
        "warmup_steps": args.warmup_steps,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "grad_clip": args.grad_clip,
        "fp16": args.fp16,
        "checkpoint_every": args.checkpoint_every,
        "eval_every": args.eval_every,
        "log_every": args.log_every,
        "output_dir": args.output_dir,
    }
    run_config = {
        "stage": "part_mmdit_memory_gate",
        "model": model_cfg,
        "data": data_cfg,
        "flow": flow_cfg,
        "loss": loss_cfg,
        "training": train_cfg,
        "memory_samples": memory_samples,
    }

    dataset = PartMMDiTDataset(data_cfg)
    full_batch_cpu, metadata = _build_memory_batch(
        dataset,
        memory_samples,
        min_multi=2 if args.preset == "quick3" else 3,
        min_single=1 if args.preset == "quick3" else 2,
    )
    shard_batch_cpu = _static_shard_batch(full_batch_cpu, metadata, rank, world_size)

    output_dir = Path(args.output_dir)
    metrics_csv = output_dir / "memory_gate_metrics.csv"
    metadata_json = output_dir / "memory_gate_samples.json"
    config_json = output_dir / "config.json"
    code_update = Path(args.code_update)
    if is_main:
        output_dir.mkdir(parents=True, exist_ok=True)
        config_json.write_text(json.dumps(run_config, indent=2), encoding="utf-8")
        metadata_json.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        if not metrics_csv.exists() or args.resume is None:
            if metrics_csv.exists():
                metrics_csv.unlink()
        _append_memory_log(
            code_update,
            title="Memory Gate Diagnostic Start",
            metadata=metadata,
            output_dir=output_dir,
            args=args,
            config=run_config,
        )
        print("[memory_gate] selected samples:")
        for item in metadata:
            print(
                "  "
                f"{item['obj_id']} {item['sample_id']} group={item['group']} "
                f"K={item['part_count']} counts={item['part_raw_voxel_counts']} "
                f"parts={item['part_names']}"
            )
    if dist.is_initialized():
        dist.barrier()

    model = PartMMDiTModel(**model_cfg).to(device)
    if world_size > 1:
        model = DDP(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            broadcast_buffers=False,
        )
    raw_model = model.module if isinstance(model, DDP) else model
    fp16_params = [
        name
        for name, param in raw_model.named_parameters()
        if param.requires_grad and param.dtype == torch.float16
    ]
    if fp16_params:
        raise RuntimeError(
            "memory gate expects FP32 trainable params with AMP, found native fp16: "
            f"{fp16_params[:10]}"
        )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: _lr_for_step(
            step,
            max_steps=int(args.max_steps),
            warmup_steps=int(args.warmup_steps),
        ),
    )
    start_step = 0
    if args.resume:
        start_step = _load_resume(
            Path(args.resume),
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
        )
        if is_main:
            print(f"[memory_gate] resumed from {args.resume} at step {start_step}")

    if torch.cuda.is_available():
        amp_scaler = torch.cuda.amp.GradScaler(enabled=bool(args.fp16))
    else:
        amp_scaler = None
    use_fp16 = bool(args.fp16) and torch.cuda.is_available()
    loss_kwargs = _loss_kwargs(flow_cfg, loss_cfg)
    if is_main:
        n_params = sum(param.numel() for param in raw_model.parameters())
        total_parts = int(full_batch_cpu["part_valid"].sum().item())
        print(
            "[memory_gate] "
            f"device={device} world_size={world_size} model_params={n_params:,} "
            f"batch_objects={len(metadata)} total_parts={total_parts}"
        )
        print(
            "[memory_gate] "
            f"max_steps={args.max_steps} warmup={args.warmup_steps} lr={args.lr} "
            f"checkpoint_every={args.checkpoint_every} eval_every={args.eval_every} "
            f"fp16={args.fp16} no_early_stop=True"
        )
        print(
            "[memory_gate] "
            f"t_schedule={flow_cfg.get('t_schedule')} "
            f"timestep_shift={_dynamic_timestep_shift(raw_model, flow_cfg):.4f} "
            f"sampler_num_steps={flow_cfg.get('num_steps')}"
        )

    eval_generator = torch.Generator(device="cpu")
    eval_generator.manual_seed(args.seed + 12345)
    eval_noise = torch.randn(
        tuple(full_batch_cpu["x_1_parts"].shape),
        generator=eval_generator,
        dtype=full_batch_cpu["x_1_parts"].dtype,
    )
    full_batch_cpu = _move_float_batch_dtype(full_batch_cpu, torch.float32)
    shard_batch_cpu = _move_float_batch_dtype(shard_batch_cpu, torch.float32)

    rows = _read_existing_metrics(metrics_csv) if is_main else []
    last_loss = float("nan")
    max_steps = int(args.dry_run_steps) if int(args.dry_run_steps) > 0 else int(args.max_steps)
    model.train()
    for step in range(start_step + 1, max_steps + 1):
        batch = _to_device(shard_batch_cpu, device)
        optimizer.zero_grad(set_to_none=True)
        t = sample_flow_timesteps(
            batch["x_1_parts"].shape[0],
            device=device,
            dtype=batch["x_1_parts"].dtype,
            t_min=float(flow_cfg.get("t_min", 0.0)),
            t_max=float(flow_cfg.get("t_max", 1.0)),
            t_schedule=str(flow_cfg.get("t_schedule", "logit_normal")),
            t_logit_normal_mean=float(flow_cfg.get("t_logit_normal_mean", 0.0)),
            t_logit_normal_std=float(flow_cfg.get("t_logit_normal_std", 1.0)),
        )
        t = _apply_timestep_shift(
            t,
            shift=_dynamic_timestep_shift(raw_model, flow_cfg),
            t_min=float(flow_cfg.get("t_min", 0.0)),
            t_max=float(flow_cfg.get("t_max", 1.0)),
        )

        if torch.cuda.is_available():
            with torch.cuda.amp.autocast(enabled=use_fp16, cache_enabled=False):
                loss = rectified_flow_loss(
                    model,
                    batch["x_1_parts"],
                    t,
                    z_global=batch["z_global"],
                    cond=batch["cond"],
                    name_tokens=batch["name_tokens"],
                    name_mask=batch["name_mask"],
                    anchor=batch["anchor"],
                    anchor_valid=batch["anchor_valid"],
                    part_valid=batch["part_valid"],
                    part_raw_voxel_counts=batch["part_raw_voxel_counts"],
                    **loss_kwargs,
                )
            if amp_scaler is not None:
                amp_scaler.scale(loss).backward()
                amp_scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(args.grad_clip))
                amp_scaler.step(optimizer)
                amp_scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(args.grad_clip))
                optimizer.step()
        else:
            loss = rectified_flow_loss(
                model,
                batch["x_1_parts"],
                t,
                z_global=batch["z_global"],
                cond=batch["cond"],
                name_tokens=batch["name_tokens"],
                name_mask=batch["name_mask"],
                anchor=batch["anchor"],
                anchor_valid=batch["anchor_valid"],
                part_valid=batch["part_valid"],
                part_raw_voxel_counts=batch["part_raw_voxel_counts"],
                **loss_kwargs,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(args.grad_clip))
            optimizer.step()
        scheduler.step()
        last_loss = float(loss.item())

        if is_main and (step == 1 or step % int(args.log_every) == 0):
            print(
                f"[memory_gate] step {step:5d}/{args.max_steps} "
                f"loss={last_loss:.6f} t_mean={float(t.mean().item()):.4f} "
                f"lr={scheduler.get_last_lr()[0]:.2e}"
            )

        if step % int(args.checkpoint_every) == 0:
            if dist.is_initialized():
                dist.barrier()
            if is_main:
                ckpt_path = output_dir / "ckpts" / f"step_{step}.pt"
                _save_checkpoint(
                    ckpt_path,
                    step=step,
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    config=run_config,
                )
                print(f"[memory_gate] saved checkpoint: {ckpt_path}")
            if dist.is_initialized():
                dist.barrier()

        if step % int(args.eval_every) == 0 or step == max_steps:
            if dist.is_initialized():
                dist.barrier()
            if is_main:
                metrics = _eval_memory_gate(
                    model,
                    full_batch_cpu,
                    metadata,
                    device=device,
                    flow_cfg=flow_cfg,
                    eval_noise=eval_noise,
                )
                row = {"step": step, "train_loss": last_loss, **metrics}
                rows.append(row)
                _write_csv_row(metrics_csv, row)
                print(
                    "[memory_gate_eval] "
                    f"step={step} train_loss={last_loss:.6f} "
                    f"cos_single={metrics['cos_single']:.6f} "
                    f"cos_multi={metrics['cos_multi']:.6f} "
                    f"cos_102276_buttons={metrics['cos_102276_buttons']:.6f}"
                )
            if dist.is_initialized():
                dist.barrier()

    if dist.is_initialized():
        dist.barrier()
    final_step = max_steps
    final_ckpt = output_dir / "ckpts" / f"step_{final_step}.pt"
    if is_main:
        _save_checkpoint(
            final_ckpt,
            step=final_step,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            config=run_config,
        )
        print(f"[memory_gate] final checkpoint: {final_ckpt}")
        if int(args.dry_run_steps) <= 0:
            _append_result_log(
                code_update,
                title="Memory Gate Diagnostic Result",
                rows=rows,
                final_checkpoint=final_ckpt,
                output_dir=output_dir,
            )
    if dist.is_initialized():
        dist.barrier()
    _cleanup_dist()


if __name__ == "__main__":
    main()
