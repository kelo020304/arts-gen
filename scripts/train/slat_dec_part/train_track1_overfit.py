#!/usr/bin/env python3
"""Track1 4-GPU overfit entry for PartMaskedSLatMeshDecoder.

Runs a compact DDP loop with online nvdiffrast GT mesh rendering supervision.
This is intentionally scoped to the Phase 2 overfit cache and 500-step gate.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

REPO_ROOT = Path(__file__).resolve().parents[3]
TRELLIS_ROOT = REPO_ROOT / "TRELLIS-arts"
if str(TRELLIS_ROOT) not in sys.path:
    sys.path.insert(0, str(TRELLIS_ROOT))

from track1_online_render import (  # noqa: E402
    DEFAULT_CACHE_MANIFEST,
    DEFAULT_DECODER_CKPT,
    OnlineRenderLossProbe,
    PartMaskedOnlineRenderDataset,
    build_partmasked_decoder_from_pretrained,
)


DEFAULT_CKPT_DIR = Path("/robot/data-lab/jzh/art-gen/ckpts/slat-dec-part/overfit-0703")


def is_dist() -> bool:
    return "RANK" in os.environ and "WORLD_SIZE" in os.environ


def setup_dist() -> tuple[int, int, int, torch.device]:
    if is_dist():
        dist.init_process_group(backend="nccl")
        rank = dist.get_rank()
        world = dist.get_world_size()
        local_rank = int(os.environ.get("LOCAL_RANK", rank % torch.cuda.device_count()))
    else:
        rank = 0
        world = 1
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    return rank, world, local_rank, torch.device("cuda", local_rank)


def cleanup_dist() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def unwrap(model: torch.nn.Module) -> torch.nn.Module:
    return model.module if isinstance(model, DDP) else model


def apply_trainable_mode(model: torch.nn.Module, mode: str, *, unfreeze_last_blocks: int = 0) -> None:
    base = unwrap(model)
    mode = str(mode)
    if mode == "all":
        for param in base.parameters():
            param.requires_grad_(True)
        return
    if mode == "mask_input_only":
        for param in base.parameters():
            param.requires_grad_(False)
        weight = base.input_layer.weight
        weight.requires_grad_(True)

        def _mask_grad(grad: torch.Tensor) -> torch.Tensor:
            masked = torch.zeros_like(grad)
            masked[:, -1:] = grad[:, -1:]
            return masked

        weight.register_hook(_mask_grad)
        if base.input_layer.bias is not None:
            base.input_layer.bias.requires_grad_(False)
        return
    if mode == "mask_modulation_only":
        for param in base.parameters():
            param.requires_grad_(False)
        weight = base.input_layer.weight
        weight.requires_grad_(True)

        def _mask_grad(grad: torch.Tensor) -> torch.Tensor:
            masked = torch.zeros_like(grad)
            masked[:, -1:] = grad[:, -1:]
            return masked

        weight.register_hook(_mask_grad)
        if base.input_layer.bias is not None:
            base.input_layer.bias.requires_grad_(False)
        modulations = getattr(base, "mask_block_modulations", None)
        output_mod = getattr(base, "mask_output_modulation", None)
        has_trainable_mod = False
        if modulations is not None and len(modulations) > 0:
            for param in modulations.parameters():
                param.requires_grad_(True)
            has_trainable_mod = True
        if output_mod is not None:
            for param in output_mod.parameters():
                param.requires_grad_(True)
            has_trainable_mod = True
        if not has_trainable_mod:
            raise ValueError("trainable_mode=mask_modulation_only requires a mask modulation module")
        return
    if mode == "mask_up_out_lastN":
        for param in base.parameters():
            param.requires_grad_(False)
        weight = base.input_layer.weight
        weight.requires_grad_(True)

        def _mask_grad(grad: torch.Tensor) -> torch.Tensor:
            masked = torch.zeros_like(grad)
            masked[:, -1:] = grad[:, -1:]
            return masked

        weight.register_hook(_mask_grad)
        if base.input_layer.bias is not None:
            base.input_layer.bias.requires_grad_(False)
        modulations = getattr(base, "mask_block_modulations", None)
        output_mod = getattr(base, "mask_output_modulation", None)
        if (modulations is None or len(modulations) == 0) and output_mod is None:
            raise ValueError("mask_up_out_lastN requires --mask-modulation with block and/or output modulation")
        if modulations is not None:
            for param in modulations.parameters():
                param.requires_grad_(True)
        if output_mod is not None:
            for param in output_mod.parameters():
                param.requires_grad_(True)
        for param in base.upsample.parameters():
            param.requires_grad_(True)
        for param in base.out_layer.parameters():
            param.requires_grad_(True)
        n = max(0, int(unfreeze_last_blocks))
        if n > 0:
            for block in list(base.blocks)[-n:]:
                for param in block.parameters():
                    param.requires_grad_(True)
        return
    raise ValueError(f"unsupported trainable_mode={mode!r}")


def make_ema(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {k: v.detach().float().cpu().clone() for k, v in unwrap(model).state_dict().items()}


def update_ema(ema: dict[str, torch.Tensor], model: torch.nn.Module, decay: float) -> None:
    state = unwrap(model).state_dict()
    for key, value in state.items():
        if key not in ema:
            ema[key] = value.detach().float().cpu().clone()
        elif torch.is_floating_point(value):
            ema[key].mul_(decay).add_(value.detach().float().cpu(), alpha=1.0 - decay)
        else:
            ema[key] = value.detach().cpu().clone()


def mask_channel_probe(model: torch.nn.Module) -> dict[str, float]:
    base = unwrap(model)
    weight = getattr(getattr(base, "input_layer", None), "weight", None)
    if weight is None or weight.ndim != 2 or weight.shape[1] < 2:
        return {
            "mask_input_weight_norm": float("nan"),
            "nonmask_input_weight_norm": float("nan"),
            "mask_input_grad_norm": float("nan"),
            "nonmask_input_grad_norm": float("nan"),
        }
    mask_weight = weight.detach()[:, -1:].float()
    nonmask_weight = weight.detach()[:, :-1].float()
    grad = weight.grad
    if grad is None:
        mask_grad_norm = float("nan")
        nonmask_grad_norm = float("nan")
    else:
        mask_grad_norm = float(grad.detach()[:, -1:].float().norm().cpu().item())
        nonmask_grad_norm = float(grad.detach()[:, :-1].float().norm().cpu().item())
    out = {
        "mask_input_weight_norm": float(mask_weight.norm().cpu().item()),
        "nonmask_input_weight_norm": float(nonmask_weight.norm().cpu().item()),
        "mask_input_grad_norm": mask_grad_norm,
        "nonmask_input_grad_norm": nonmask_grad_norm,
    }
    modulations = getattr(base, "mask_block_modulations", None)
    if modulations is not None and len(modulations) > 0:
        weights = []
        grads = []
        for module in modulations:
            weights.append(module.weight.detach().float().reshape(-1))
            if module.weight.grad is not None:
                grads.append(module.weight.grad.detach().float().reshape(-1))
            if module.bias is not None:
                weights.append(module.bias.detach().float().reshape(-1))
                if module.bias.grad is not None:
                    grads.append(module.bias.grad.detach().float().reshape(-1))
        out["mask_block_mod_weight_norm"] = float(torch.cat(weights).norm().cpu().item()) if weights else float("nan")
        out["mask_block_mod_grad_norm"] = float(torch.cat(grads).norm().cpu().item()) if grads else float("nan")
    else:
        out["mask_block_mod_weight_norm"] = float("nan")
        out["mask_block_mod_grad_norm"] = float("nan")
    output_mod = getattr(base, "mask_output_modulation", None)
    if output_mod is not None:
        weights = [output_mod.weight.detach().float().reshape(-1)]
        grads = []
        if output_mod.weight.grad is not None:
            grads.append(output_mod.weight.grad.detach().float().reshape(-1))
        if output_mod.bias is not None:
            weights.append(output_mod.bias.detach().float().reshape(-1))
            if output_mod.bias.grad is not None:
                grads.append(output_mod.bias.grad.detach().float().reshape(-1))
        out["mask_output_mod_weight_norm"] = float(torch.cat(weights).norm().cpu().item())
        out["mask_output_mod_grad_norm"] = float(torch.cat(grads).norm().cpu().item()) if grads else float("nan")
    else:
        out["mask_output_mod_weight_norm"] = float("nan")
        out["mask_output_mod_grad_norm"] = float("nan")
    return out


def zero_grad_anchor(model: torch.nn.Module) -> torch.Tensor:
    """Differentiable zero touching every trainable parameter for DDP-empty reps."""
    anchor: torch.Tensor | None = None
    for param in unwrap(model).parameters():
        if not param.requires_grad or param.numel() == 0:
            continue
        term = param.reshape(-1)[0].float() * 0.0
        anchor = term if anchor is None else anchor + term
    if anchor is None:
        raise RuntimeError("cannot build zero_grad_anchor: model has no trainable parameters")
    return anchor


def build_optimizer_param_groups(model: torch.nn.Module, args: argparse.Namespace) -> list[dict[str, Any]]:
    base = unwrap(model)
    base_lr = float(args.lr)
    if str(args.trainable_mode) == "mask_modulation_only":
        groups: list[dict[str, Any]] = []
        groups.append(
            {
                "name": "mask_input_weight",
                "params": [base.input_layer.weight],
                "lr": base_lr * float(args.mask_input_lr_scale),
            }
        )
        modulations = getattr(base, "mask_block_modulations", None)
        if modulations is None or len(modulations) == 0:
            mod_params = []
        else:
            mod_params = [param for param in modulations.parameters() if param.requires_grad]
        if mod_params:
            groups.append(
                {
                    "name": "mask_block_modulations",
                    "params": mod_params,
                    "lr": base_lr * float(args.mask_block_lr_scale),
                }
            )
        output_mod = getattr(base, "mask_output_modulation", None)
        output_params = [param for param in output_mod.parameters() if param.requires_grad] if output_mod is not None else []
        if output_params:
            groups.append(
                {
                    "name": "mask_output_modulation",
                    "params": output_params,
                    "lr": base_lr * float(args.mask_output_lr_scale),
                }
            )
        if len(groups) == 1:
            raise ValueError("mask_modulation_only found no trainable mask modulation parameters")
        return groups
    return [
        {
            "name": "trainable",
            "params": [param for param in base.parameters() if param.requires_grad],
            "lr": base_lr,
        }
    ]


def save_ckpt(path: Path, *, step: int, model: torch.nn.Module, optimizer: torch.optim.Optimizer, ema: dict[str, torch.Tensor], args: argparse.Namespace) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "step": int(step),
        "model": unwrap(model).state_dict(),
        "optimizer": optimizer.state_dict(),
        "ema": ema,
        "args": vars(args),
    }
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    tmp.replace(path)


def load_latest(ckpt_dir: Path, *, model: torch.nn.Module, optimizer: torch.optim.Optimizer, device: torch.device) -> tuple[int, dict[str, torch.Tensor]]:
    latest = ckpt_dir / "latest.pt"
    if not latest.is_file():
        return 0, make_ema(model)
    payload = torch.load(latest, map_location=device, weights_only=False)
    unwrap(model).load_state_dict(payload["model"], strict=True)
    optimizer.load_state_dict(payload["optimizer"])
    return int(payload["step"]), {k: v.detach().cpu().clone() for k, v in payload.get("ema", {}).items()}


def reduce_loss_dict(losses: dict[str, torch.Tensor], *, world: int) -> dict[str, float]:
    out: dict[str, float] = {}
    count_keys = {"tsdf_valid", "tsdf_skipped", "tsdf_total"}
    for key, value in losses.items():
        if torch.is_tensor(value):
            tensor = value.detach().float()
        else:
            tensor = torch.tensor(float(value), device="cuda")
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
            if key not in count_keys:
                tensor = tensor / float(world)
        out[key] = float(tensor.cpu().item())
    return out


def write_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def write_degradation_report(ckpt_dir: Path, rows: list[dict[str, Any]]) -> None:
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    json_path = ckpt_dir / "mask_degradation_report.json"
    json_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    md_lines = [
        "# Track1 mask degradation pool",
        "",
        "| tag | component | semantic | gt | front_only | erode | threshold | options | reason |",
        "|---|---|---|---:|---:|---:|---:|---|---|",
    ]
    for row in rows:
        options = ",".join(str(x) for x in row.get("degrade_options", [])) or "GT-only"
        md_lines.append(
            "| {tag} | {component_name} | {semantic_type} | {gt_voxels} | {front_only_voxels} | {erode_voxels} | {threshold} | {options} | {degrade_reason} |".format(
                **row,
                options=options,
            )
        )
    (ckpt_dir / "mask_degradation_report.md").write_text("\n".join(md_lines) + "\n", encoding="utf-8")


def train(args: argparse.Namespace) -> None:
    rank, world, local_rank, device = setup_dist()
    seed_all(int(args.seed) + rank)
    is_master = rank == 0
    if is_master:
        args.ckpt_dir.mkdir(parents=True, exist_ok=True)
        (args.ckpt_dir / "samples").mkdir(parents=True, exist_ok=True)

    dataset = PartMaskedOnlineRenderDataset(
        args.manifest,
        resolution=int(args.resolution),
        include_body=True,
        normalize_gt_mesh=True,
        mask_degrade_prob=float(args.mask_degrade_prob),
        front_only_prob=float(args.front_only_prob),
        latent_input_mode=str(args.latent_input_mode),
        subset_dilation=int(args.subset_dilation),
    )
    if is_master:
        write_degradation_report(args.ckpt_dir, dataset.degradation_report)
        print("[track1] mask degradation pool:", flush=True)
        for row in dataset.degradation_report:
            options = ",".join(str(x) for x in row.get("degrade_options", [])) or "GT-only"
            print(
                "[track1]   {tag}::{component_name} {gt_voxels}->{options} "
                "(front={front_only_voxels} erode={erode_voxels} threshold={threshold}; {degrade_reason})".format(
                    **row,
                    options=options,
                ),
                flush=True,
            )
    sampler = DistributedSampler(dataset, num_replicas=world, rank=rank, shuffle=True, drop_last=True) if world > 1 else None
    loader = DataLoader(
        dataset,
        batch_size=int(args.batch_size_per_gpu),
        shuffle=sampler is None,
        sampler=sampler,
        num_workers=int(args.num_workers),
        pin_memory=True,
        drop_last=True,
        collate_fn=dataset.collate_fn,
    )
    iterator = iter(loader)

    decoder = build_partmasked_decoder_from_pretrained(
        args.decoder_ckpt,
        device=device,
        train=True,
        mask_modulation=str(args.mask_modulation),
    )
    apply_trainable_mode(decoder, str(args.trainable_mode), unfreeze_last_blocks=int(args.unfreeze_last_blocks))
    model: torch.nn.Module = DDP(decoder, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False) if world > 1 else decoder
    optimizer_groups = build_optimizer_param_groups(model, args)
    optimizer = torch.optim.AdamW(
        optimizer_groups,
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
        eps=float(args.adam_eps),
    )
    start_step = 0
    ema = make_ema(model)
    if bool(args.resume):
        start_step, ema = load_latest(args.ckpt_dir, model=model, optimizer=optimizer, device=device)

    probe = OnlineRenderLossProbe(
        model,
        device=device,
        render_resolution=int(args.resolution),
        lambda_tsdf=float(args.lambda_tsdf),
        lambda_ssim=float(args.lambda_ssim),
        lambda_lpips=0.0,
    )
    model.train()
    if is_master:
        meta = {
            "started": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
            "world_size": world,
            "dataset_components": len(dataset),
            "batch_size_per_gpu": int(args.batch_size_per_gpu),
            "resolution": int(args.resolution),
            "mask_degrade_prob": float(args.mask_degrade_prob),
            "ckpt_dir": str(args.ckpt_dir),
            "start_step": start_step,
            "trainable_mode": str(args.trainable_mode),
            "mask_modulation": str(args.mask_modulation),
            "unfreeze_last_blocks": int(args.unfreeze_last_blocks),
            "latent_input_mode": str(args.latent_input_mode),
            "subset_dilation": int(args.subset_dilation),
            "optimizer_groups": [
                {
                    "name": str(group.get("name", f"group_{idx}")),
                    "lr": float(group.get("lr", args.lr)),
                    "num_params": int(sum(param.numel() for param in group.get("params", []))),
                }
                for idx, group in enumerate(optimizer_groups)
            ],
            "trainable_params": int(sum(p.numel() for p in unwrap(model).parameters() if p.requires_grad)),
            "total_params": int(sum(p.numel() for p in unwrap(model).parameters())),
        }
        (args.ckpt_dir / "run_meta.json").write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"[track1] start world={world} dataset={len(dataset)} start_step={start_step} max_steps={args.max_steps}", flush=True)

    consecutive_no_grad_skips = 0
    for step in range(start_step + 1, int(args.max_steps) + 1):
        if sampler is not None:
            sampler.set_epoch(step)
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            batch = next(iterator)
        optimizer.zero_grad(set_to_none=True)
        prepared = probe.prepare_batch(batch, device=device)
        # FlexiCubes mesh extraction currently mixes fp32 buffers with model
        # activations in scatter_reduce; external bf16 autocast causes dtype
        # mismatch. The decoder itself keeps its pretrained fp16 internal blocks.
        losses, status = probe.training_losses(**prepared)
        for status_key, status_value in status.items():
            if isinstance(status_value, (int, float)):
                losses[status_key] = torch.tensor(float(status_value), device=device)
        loss = losses["loss"]
        non_finite_terms = {
            key: (float(value.detach().float().cpu().item()) if torch.is_tensor(value) and value.numel() == 1 else str(value))
            for key, value in losses.items()
            if torch.is_tensor(value) and not torch.isfinite(value).all()
        }
        if non_finite_terms:
            debug = {
                "step": step,
                "non_finite_terms": non_finite_terms,
                "sample_meta": batch["sample_meta"],
                "mask_modes": [meta.get("mask_mode") for meta in batch["sample_meta"]],
            }
            if is_master:
                args.ckpt_dir.mkdir(parents=True, exist_ok=True)
                (args.ckpt_dir / f"nonfinite_step_{step:07d}.json").write_text(
                    json.dumps(debug, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8",
                )
            raise RuntimeError(f"non-finite Track1 terms at step {step}: {non_finite_terms}")
        raw_loss_requires_grad = bool(loss.requires_grad)
        no_grad_local = torch.tensor(0 if raw_loss_requires_grad else 1, device=device, dtype=torch.int32)
        if world > 1:
            torch.distributed.all_reduce(no_grad_local, op=torch.distributed.ReduceOp.SUM)
        if not raw_loss_requires_grad:
            loss = loss + zero_grad_anchor(model)
        if int(no_grad_local.item()) == world:
            consecutive_no_grad_skips += 1
            reduced = reduce_loss_dict(losses, world=world)
            if is_master:
                mode_counts = defaultdict(int)
                for meta in batch["sample_meta"]:
                    mode_counts[str(meta.get("mask_mode", "unknown"))] += 1
                row = {
                    "step": step,
                    "time": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
                    "losses": reduced,
                    "no_grad_loss_skipped": int(no_grad_local.item()),
                    "mask_modes": dict(mode_counts),
                    "sample_meta": batch["sample_meta"],
                }
                write_jsonl(args.ckpt_dir / "losses.jsonl", row)
                print(
                    f"[track1] step={step} skipped_no_grad ranks={int(no_grad_local.item())} "
                    f"rep_success={reduced.get('rep_success', float('nan')):.1f}/"
                    f"{reduced.get('rep_total', float('nan')):.1f} mask={dict(mode_counts)}",
                    flush=True,
                )
                if int(args.max_consecutive_no_grad_skips) > 0 and consecutive_no_grad_skips >= int(args.max_consecutive_no_grad_skips):
                    (args.ckpt_dir / f"no_grad_abort_step_{step:07d}.json").write_text(
                        json.dumps(row, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                        encoding="utf-8",
                    )
            if int(args.max_consecutive_no_grad_skips) > 0 and consecutive_no_grad_skips >= int(args.max_consecutive_no_grad_skips):
                raise RuntimeError(
                    f"Track1 no differentiable loss for {consecutive_no_grad_skips} consecutive steps "
                    f"(last step {step}, skipped_ranks={int(no_grad_local.item())})"
                )
            continue
        if int(no_grad_local.item()) > 0:
            consecutive_no_grad_skips = 0
            if is_master:
                print(
                    f"[track1] step={step} zero-anchor ranks={int(no_grad_local.item())}/{world}; "
                    "continuing with contributing ranks",
                    flush=True,
                )
        consecutive_no_grad_skips = 0
        loss.backward()
        mask_probe = mask_channel_probe(model)
        if float(args.grad_clip) > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(args.grad_clip))
        optimizer.step()
        after_step_probe = mask_channel_probe(model)
        mask_probe |= {
            "mask_input_weight_norm_after_step": after_step_probe["mask_input_weight_norm"],
            "mask_block_mod_weight_norm_after_step": after_step_probe["mask_block_mod_weight_norm"],
            "mask_output_mod_weight_norm_after_step": after_step_probe["mask_output_mod_weight_norm"],
        }
        update_ema(ema, model, float(args.ema_decay))
        reduced = reduce_loss_dict(losses, world=world)
        if is_master and (step == 1 or step % int(args.log_every) == 0 or step == int(args.max_steps)):
            mode_counts = defaultdict(int)
            for meta in batch["sample_meta"]:
                mode_counts[str(meta.get("mask_mode", "unknown"))] += 1
            row = {
                "step": step,
                "time": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
                "losses": reduced,
                "tsdf_skipped": int(round(reduced.get("tsdf_skipped", 0.0))),
                "tsdf_valid": int(round(reduced.get("tsdf_valid", 0.0))),
                "tsdf_total": int(round(reduced.get("tsdf_total", 0.0))),
                "zero_anchor_no_grad_ranks": int(no_grad_local.item()),
                "mask_modes": dict(mode_counts),
                "mask_channel_probe": mask_probe,
                "peak_mem_mb": float(torch.cuda.max_memory_allocated(device) / (1024.0 * 1024.0)),
            }
            write_jsonl(args.ckpt_dir / "losses.jsonl", row)
            print(
                f"[track1] step={step} loss={reduced.get('loss'):.6f} "
                f"tsdf_skipped={row['tsdf_skipped']}/{row['tsdf_total']} mask={dict(mode_counts)} "
                f"mask_w={mask_probe['mask_input_weight_norm_after_step']:.6f} "
                f"mask_g={mask_probe['mask_input_grad_norm']:.6f} "
                f"block_w={mask_probe['mask_block_mod_weight_norm_after_step']:.6f} "
                f"block_g={mask_probe['mask_block_mod_grad_norm']:.6f} "
                f"out_w={mask_probe['mask_output_mod_weight_norm_after_step']:.6f} "
                f"out_g={mask_probe['mask_output_mod_grad_norm']:.6f}",
                flush=True,
            )
        if is_master and (step % int(args.save_every) == 0 or step == int(args.max_steps)):
            save_ckpt(args.ckpt_dir / f"step_{step:07d}.pt", step=step, model=model, optimizer=optimizer, ema=ema, args=args)
            save_ckpt(args.ckpt_dir / "latest.pt", step=step, model=model, optimizer=optimizer, ema=ema, args=args)
        if is_master and (step % int(args.snapshot_every) == 0 or step == int(args.max_steps)):
            snapshot = {
                "step": step,
                "samples": batch["sample_meta"],
                "losses": reduced,
                "note": "numeric snapshot only; visual harness gate runs after 500-step smoke",
            }
            (args.ckpt_dir / "samples" / f"snapshot_step_{step:07d}.json").write_text(
                json.dumps(snapshot, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )

    if is_master:
        print(f"[track1] complete max_steps={args.max_steps}", flush=True)
    cleanup_dist()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=DEFAULT_CACHE_MANIFEST)
    parser.add_argument("--decoder-ckpt", type=Path, default=DEFAULT_DECODER_CKPT)
    parser.add_argument("--ckpt-dir", type=Path, default=DEFAULT_CKPT_DIR)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--batch-size-per-gpu", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1.0e-5)
    parser.add_argument("--mask-input-lr-scale", type=float, default=1.0)
    parser.add_argument("--mask-block-lr-scale", type=float, default=1.0)
    parser.add_argument("--mask-output-lr-scale", type=float, default=1.0)
    parser.add_argument("--adam-eps", type=float, default=1.0e-8)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--ema-decay", type=float, default=0.999)
    parser.add_argument("--lambda-tsdf", type=float, default=0.01)
    parser.add_argument("--lambda-ssim", type=float, default=0.2)
    parser.add_argument("--mask-degrade-prob", type=float, default=0.3)
    parser.add_argument("--front-only-prob", type=float, default=0.5)
    parser.add_argument("--trainable-mode", choices=["mask_input_only", "mask_modulation_only", "mask_up_out_lastN", "all"], default="mask_input_only")
    parser.add_argument(
        "--mask-modulation",
        choices=["none", "per_block_add", "output_feature_add", "per_block_add_output_feature_add"],
        default="none",
    )
    parser.add_argument("--unfreeze-last-blocks", type=int, default=0)
    parser.add_argument("--latent-input-mode", choices=["whole", "expanded_subset"], default="whole")
    parser.add_argument("--subset-dilation", type=int, default=1)
    parser.add_argument("--bf16", action="store_true", default=True)
    parser.add_argument("--no-bf16", dest="bf16", action="store_false")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--seed", type=int, default=20260703)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--save-every", type=int, default=100)
    parser.add_argument("--snapshot-every", type=int, default=100)
    parser.add_argument("--max-consecutive-no-grad-skips", type=int, default=20)
    train(parser.parse_args())


if __name__ == "__main__":
    os.environ.setdefault("SPCONV_ALGO", "native")
    main()
