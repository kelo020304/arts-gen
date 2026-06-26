"""Trainer entry for mask16-conditioned part SS latent flow.

Separate stage from ``part_ss_latent_flow`` so the 0526 path stays unchanged.
"""

from __future__ import annotations

import json
import math
import os
import random
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from trellis.datasets.arts.part_ss_latent_flow_mask16 import PartSSLatentFlowMask16Dataset
from trellis.models.part_flow.part_ss_latent_flow_mask16 import PartSSLatentFlowMask16Model
from trellis.trainers.arts.part_ss_latent_flow import (
    _assert_no_native_fp16_trainable_params,
    _cfg_dict,
    _load_resume_checkpoint,
    _resolve_latent_stats,
    _resolve_object_weight_k_ref,
    _run_rank0_only_work,
    _save_ckpt,
    _setup_rng,
    _to_device,
    _wrap_ddp_model,
)
from trellis.trainers.arts.part_ss_latent_flow_losses import build_part_ss_sampler_kwargs
from trellis.trainers.arts.part_ss_latent_flow_mask16_losses import PartSSLatentMask16RFLoss
from trellis.utils.arts.config_utils import config_to_dict
from trellis.utils.arts.ddp_utils import setup_ddp


def _build_part_ss_latent_mask16_rf_loss(
    *,
    loss_cfg: Dict[str, Any],
    flow_cfg: Dict[str, Any],
    resolved_object_weight_k_ref: float | None,
    model_cfg: Dict[str, Any] | None = None,
):
    model_cfg = model_cfg or {}
    criterion_kwargs = dict(
        t_min=float(flow_cfg.get("t_min", 0.0)),
        t_max=float(flow_cfg.get("t_max", 1.0)),
        noise_scale=float(flow_cfg.get("noise_scale", 1.0)),
        latent_scale=float(flow_cfg.get("latent_scale", 1.0)),
        part_weight_mode=str(loss_cfg.get("part_weight_mode", "none")),
        part_weight_ref_mode=str(loss_cfg.get("part_weight_ref_mode", "median")),
        part_weight_alpha=float(loss_cfg.get("part_weight_alpha", 0.5)),
        part_weight_min=float(loss_cfg.get("part_weight_min", 0.5)),
        part_weight_max=float(loss_cfg.get("part_weight_max", 3.0)),
        normalize_part_weights_per_object=bool(loss_cfg.get("normalize_part_weights_per_object", True)),
        size_bucket_boundaries=tuple(float(x) for x in loss_cfg.get("size_bucket_boundaries", [500.0, 3000.0])),
        object_balanced=bool(loss_cfg.get("object_balanced", False)),
        object_weight_mode=str(loss_cfg.get("object_weight_mode", "none")),
        object_weight_k_ref=resolved_object_weight_k_ref,
        object_weight_min=float(loss_cfg.get("object_weight_min", 0.75)),
        object_weight_max=float(loss_cfg.get("object_weight_max", 2.0)),
        relative_endpoint_weight=float(loss_cfg.get("relative_endpoint_weight", 0.0)),
        relative_endpoint_eps=float(loss_cfg.get("relative_endpoint_eps", 1.0e-6)),
        velocity_contrastive_weight=float(loss_cfg.get("velocity_contrastive_weight", 0.0)),
        velocity_contrastive_lambda=float(loss_cfg.get("velocity_contrastive_lambda", 0.05)),
        identity_contrastive_weight=float(loss_cfg.get("identity_contrastive_weight", 0.0)),
        identity_contrastive_temperature=float(loss_cfg.get("identity_contrastive_temperature", 0.1)),
        identity_contrastive_eps=float(loss_cfg.get("identity_contrastive_eps", 1.0e-6)),
        t_schedule=str(flow_cfg.get("t_schedule", "logit_normal")),
        t_logit_normal_mean=float(flow_cfg.get("t_logit_normal_mean", 0.0)),
        t_logit_normal_std=float(flow_cfg.get("t_logit_normal_std", 1.0)),
        latent_norm_mode=str(flow_cfg.get("latent_norm_mode", "scalar")),
        latent_channels=int(model_cfg.get("latent_channels", 8)),
        latent_mean=flow_cfg.get("latent_mean"),
        latent_std=flow_cfg.get("latent_std"),
        part_shuffle=bool(loss_cfg.get("part_shuffle", False)),
        self_conditioning=bool(model_cfg.get("self_conditioning", False)),
        self_conditioning_prob=float(loss_cfg.get("self_conditioning_prob", 0.5)),
        cfg_dropout_prob=float(loss_cfg.get("cfg_dropout_prob", 0.0)),
        foreground_weight=dict(loss_cfg.get("foreground_weight", {})),
    )
    return PartSSLatentMask16RFLoss(**criterion_kwargs)


def _compatible_mask16_state_dict(model, state_dict: dict) -> dict:
    raw_model = model.module if isinstance(model, DDP) else model
    model_state = raw_model.state_dict()
    model_keys = set(model_state.keys())
    filtered = dict(state_dict)
    for key in list(filtered):
        if key not in model_keys:
            filtered.pop(key)
            continue
        if tuple(filtered[key].shape) == tuple(model_state[key].shape):
            continue
        if key == "backbone.input_layer.weight" and filtered[key].dim() == 2:
            old = filtered[key]
            new = model_state[key].clone()
            cols = min(old.shape[1], new.shape[1])
            if old.shape[0] == new.shape[0] and cols > 0:
                new[:, :cols] = old[:, :cols]
                filtered[key] = new
                continue
        filtered.pop(key)
    return filtered


def _load_init_weights(model, path: str | os.PathLike[str] | None, device: torch.device, rank: int) -> None:
    if path is None:
        return
    ckpt_path = Path(path)
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"init checkpoint not found: {ckpt_path}")
    raw_model = model.module if isinstance(model, DDP) else model
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    state = ckpt.get("model", ckpt)
    missing, unexpected = raw_model.load_state_dict(_compatible_mask16_state_dict(raw_model, state), strict=False)
    if rank == 0:
        print(f"  [INIT] loaded compatible weights from {ckpt_path}")
        print(f"  [INIT] missing={len(missing)} unexpected={len(unexpected)}")
        if missing:
            print(f"  [INIT] missing preview: {missing[:8]}")
        if unexpected:
            print(f"  [INIT] unexpected preview: {unexpected[:8]}")


def train(config, dataset_cls: type = PartSSLatentFlowMask16Dataset) -> None:
    cfg = config
    rank, local_rank, world_size = setup_ddp()
    is_distributed = world_size > 1
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    seed = int(getattr(cfg.training, "seed", 42))
    _setup_rng(seed + rank)

    data_cfg = _cfg_dict(cfg.data)
    model_cfg = _cfg_dict(cfg.model)
    flow_cfg = _cfg_dict(cfg.flow)
    loss_cfg = _cfg_dict(cfg.loss) if "loss" in cfg else {}
    training_cfg = _cfg_dict(cfg.training)

    dataset = dataset_cls(data_cfg)
    resolved_object_weight_k_ref = _resolve_object_weight_k_ref(
        dataset,
        loss_cfg,
        rank=rank,
        is_distributed=is_distributed,
    )
    _resolve_latent_stats(
        dataset,
        flow_cfg=flow_cfg,
        model_cfg=model_cfg,
        rank=rank,
        is_distributed=is_distributed,
    )
    if flow_cfg.get("latent_mean") is not None and flow_cfg.get("latent_std") is not None:
        cfg.flow.latent_mean = list(flow_cfg["latent_mean"])
        cfg.flow.latent_std = list(flow_cfg["latent_std"])

    sampler = DistributedSampler(dataset, shuffle=True) if is_distributed else None
    loader = DataLoader(
        dataset,
        batch_size=int(training_cfg.get("batch_size", 1)),
        shuffle=sampler is None,
        sampler=sampler,
        num_workers=int(training_cfg.get("num_workers", 0)),
        pin_memory=torch.cuda.is_available(),
        collate_fn=dataset.collate_fn,
        drop_last=False,
    )

    model = PartSSLatentFlowMask16Model(**model_cfg).to(device)
    if is_distributed:
        model = _wrap_ddp_model(model, local_rank)
    raw_model = model.module if isinstance(model, DDP) else model
    _assert_no_native_fp16_trainable_params(model)

    _load_init_weights(model, training_cfg.get("init_weights"), device, rank)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training_cfg.get("lr", 1.0e-4)),
        weight_decay=float(training_cfg.get("weight_decay", 0.01)),
    )
    max_steps = int(training_cfg.get("max_steps", 3000))
    warmup_steps = max(1, int(training_cfg.get("warmup_steps", 50)))

    def lr_lambda(step):
        step = max(1, step)
        if step <= warmup_steps:
            return step / float(warmup_steps)
        progress = min(1.0, (step - warmup_steps) / float(max(1, max_steps - warmup_steps)))
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
    criterion = _build_part_ss_latent_mask16_rf_loss(
        loss_cfg=loss_cfg,
        flow_cfg=flow_cfg,
        resolved_object_weight_k_ref=resolved_object_weight_k_ref,
        model_cfg=model_cfg,
    )
    start_step = _load_resume_checkpoint(model, optimizer, scheduler, training_cfg, device, rank)

    output_dir = Path(training_cfg.get("output_dir", "runs/part_ss_latent_flow_mask16"))
    if rank == 0:
        output_dir.mkdir(parents=True, exist_ok=True)
        with (output_dir / "config.json").open("w", encoding="utf-8") as f:
            json.dump(config_to_dict(cfg), f, indent=2)
        n_params = sum(p.numel() for p in raw_model.parameters())
        total_parts = sum(len(sample["parts"]) for sample in dataset.samples)
        print("\n[Part SS Latent Flow Mask16 Train]")
        print(f"  device={device} world_size={world_size}")
        print(f"  dataset: {len(dataset)} object samples / {total_parts} target parts | batch/gpu={training_cfg.get('batch_size', 1)}")
        print(f"  model: PartSSLatentFlowMask16Model ({n_params:,} params)")
        print(f"  mask16_condition={model_cfg.get('mask16_condition', {'enabled': True})}")
        print(f"  loss: {loss_cfg}")
        print(f"  sampler_kwargs={build_part_ss_sampler_kwargs(raw_model, flow_cfg)}")
        print(f"  starting training: {max_steps} steps")

    use_fp16 = bool(training_cfg.get("fp16", False)) and torch.cuda.is_available()
    scaler = torch.cuda.amp.GradScaler(enabled=use_fp16) if torch.cuda.is_available() else None
    grad_clip = float(training_cfg.get("grad_clip", 1.0))
    log_every = int(training_cfg.get("log_every", 10))
    ckpt_every = int(training_cfg.get("checkpoint_every", 500))

    model.train()
    epoch = 0
    if sampler is not None:
        sampler.set_epoch(epoch)
    data_iter = iter(loader)
    for step in range(start_step + 1, max_steps + 1):
        try:
            batch = next(data_iter)
        except StopIteration:
            epoch += 1
            if sampler is not None:
                sampler.set_epoch(epoch)
            data_iter = iter(loader)
            batch = next(data_iter)
        batch = _to_device(batch, device)
        optimizer.zero_grad(set_to_none=True)
        if torch.cuda.is_available():
            with torch.cuda.amp.autocast(enabled=use_fp16, cache_enabled=False):
                loss, metrics = criterion(model, batch)
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()
        else:
            loss, metrics = criterion(model, batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
        scheduler.step()

        if rank == 0 and (step == 1 or step % log_every == 0):
            print(
                f"  step {step:6d}/{max_steps} | loss={float(loss.item()):.4f} "
                f"mse={metrics['mse']:.4f} latent_l1={metrics['latent_l1']:.4f} "
                f"mse_unw={metrics.get('mse_unweighted', metrics['mse']):.4f} "
                f"rel={metrics.get('relative_endpoint_loss', 0.0):.3f} "
                f"id={metrics.get('identity_contrastive_loss', 0.0):.3f} "
                f"id_acc={metrics.get('identity_contrastive_acc', float('nan')):.2f} "
                f"mask_mean={metrics.get('mask16_condition_mean', 0.0):.4f} "
                f"parts={int(metrics.get('parts', 0))} "
                f"t_mean={metrics['t_mean']:.2f} | lr={scheduler.get_last_lr()[0]:.2e}"
            )

        if ckpt_every > 0 and step % ckpt_every == 0:
            _run_rank0_only_work(
                "checkpoint",
                rank,
                lambda: _save_ckpt(step, model, optimizer, scheduler, cfg, output_dir, rank),
            )

    _run_rank0_only_work(
        "final_checkpoint",
        rank,
        lambda: _save_ckpt(max_steps, model, optimizer, scheduler, cfg, output_dir, rank),
    )
    if dist.is_initialized():
        dist.barrier()
