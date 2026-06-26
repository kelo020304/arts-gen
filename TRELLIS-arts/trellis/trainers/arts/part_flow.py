#!/usr/bin/env python3
"""Stage trainer: part_flow (Dense-64^3 Part Flow — Fisher / Gumbel-Softmax FM).

Migrated from scripts/train/part_flow/train_part_flow.py in Plan 09-03.
Removed sys.path / types.ModuleType bootstrap shim — minimal-deps registration
is performed once by TRELLIS-arts/train_arts.py.

Flow family is selected via YAML ``flow.type`` in {gumbel, fisher}; default is
``fisher`` for Phase 8 dense surface-to-solid completion. The model
(``PartFlowPredictor``) outputs endpoint logits; the bridge owns the path /
ODE step logic, while ``FlowMatchingLoss`` applies the weighted focal endpoint
objective.

Public API:
    train(config) -> None — Stage entry-point invoked by train_arts.py dispatch.
"""

import argparse
import contextlib
import copy
import math
import os
import random
import time

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP

# --- Project-root anchor for TORCH_HOME --------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(_HERE, "..", "..", "..", ".."))

os.environ.setdefault('TORCH_HOME', os.path.join(PROJECT_ROOT, 'submodules', 'TRELLIS.1'))
# Production dense-64^3 runs should choose ATTN_BACKEND explicitly via launcher
# (typically flash_attn on A100/H200). Do not silently force sdpa here.

# --- trellis-internal imports ------------------------------------------------
from trellis.utils.arts.config_utils import config_to_dict, load_config
from trellis.utils.arts.ddp_utils import setup_ddp as _setup_ddp_canonical
from trellis.datasets.arts.part_flow import PartFlowDataset
from trellis.trainers.arts.part_flow_losses import FlowMatchingLoss, flow_sample
from trellis.models.part_flow.bridges import build_bridge
from trellis.models.part_flow.part_flow_predictor import PartFlowPredictor


# ---------------------------------------------------------------------------
# DDP helpers
# ---------------------------------------------------------------------------

def setup_ddp():
    """Init DDP from torchrun env vars. Returns (rank, world_size, is_ddp).

    Wraps trellis.utils.arts.ddp_utils.setup_ddp with the legacy 3-tuple shape
    used by part_flow / part_predictor trainers (rank, world_size, is_ddp).
    """
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank, _local_rank, world_size = _setup_ddp_canonical()
        return rank, world_size, True
    return 0, 1, False


def cleanup_ddp():
    if dist.is_initialized():
        dist.destroy_process_group()


def is_main_process():
    return not dist.is_initialized() or dist.get_rank() == 0


class EMA:
    """Exponential moving average of model parameters."""

    def __init__(self, model: nn.Module, decay: float = 0.9999):
        self.decay = decay
        self.shadow = copy.deepcopy(model)
        self.shadow.eval()
        for p in self.shadow.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module):
        src_sd = model.state_dict()
        tgt_sd = self.shadow.state_dict()
        for k, v_tgt in tgt_sd.items():
            v_src = src_sd[k]
            if v_src.dtype.is_floating_point:
                v_tgt.mul_(self.decay).add_(v_src.detach(), alpha=1.0 - self.decay)
            else:
                v_tgt.copy_(v_src)

    def state_dict(self):
        return self.shadow.state_dict()

    def load_state_dict(self, sd):
        self.shadow.load_state_dict(sd)


def get_lr_scheduler(optimizer, warmup_steps: int, max_steps: int, min_lr_ratio: float = 0.1):
    def lr_lambda(step):
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
        progress = min(1.0, progress)
        cos = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cos
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ---------------------------------------------------------------------------
# Eval (generic: uses bridge + solver from config)
# ---------------------------------------------------------------------------

@torch.no_grad()
def eval_accuracy(model, bridge, eval_batch, num_steps: int, solver: str, device) -> dict:
    was_training = model.training
    model.eval()
    labels, soft = flow_sample(
        model, bridge,
        coords=eval_batch['coords'].to(device),
        cond=eval_batch['cond'].to(device),
        mask_token_labels=eval_batch['mask_token_labels'].to(device),
        voxel_layout=eval_batch['voxel_layout'],
        num_parts=eval_batch['num_parts'],
        is_on_surface=eval_batch['is_on_surface'].to(device),
        num_steps=num_steps,
        solver=solver,
    )
    gt = eval_batch['per_voxel_labels'].to(device).long()
    supervised = gt != -1
    assert supervised.any(), 'eval batch has no supervised voxels'
    acc = (labels[supervised] == gt[supervised]).float().mean().item()

    non_empty = supervised & (gt > 0)
    non_empty_acc = (
        (labels[non_empty] == gt[non_empty]).float().mean().item()
        if non_empty.any() else 0.0
    )

    ious = []
    non_empty_ious = []
    target_ious = []
    per_class_iou: dict = {}        # {sample_b: {c: iou}} for diagnostic print
    target_correct = 0
    target_total = 0
    body_correct = 0
    body_total = 0
    for b, (sl, K_b) in enumerate(zip(eval_batch['voxel_layout'], eval_batch['num_parts'])):
        gt_b = gt[sl]
        pred_b = labels[sl]
        keep = gt_b != -1
        gt_b = gt_b[keep]
        pred_b = pred_b[keep]
        per_class_iou[b] = {}
        body_slot = int(K_b) - 1
        target_mask = (gt_b > 0) & (gt_b < body_slot)
        body_mask = gt_b == body_slot
        if target_mask.any():
            target_correct += int((pred_b[target_mask] == gt_b[target_mask]).sum().item())
            target_total += int(target_mask.sum().item())
        if body_mask.any():
            body_correct += int((pred_b[body_mask] == gt_b[body_mask]).sum().item())
            body_total += int(body_mask.sum().item())
        for c in range(K_b):
            pm = pred_b == c
            gm = gt_b == c
            union = (pm | gm).sum().item()
            if union > 0:
                iou_c = (pm & gm).sum().item() / union
                ious.append(iou_c)
                if c > 0:
                    non_empty_ious.append(iou_c)
                if 0 < c < body_slot:
                    target_ious.append(iou_c)
                per_class_iou[b][c] = round(iou_c, 4)
            else:
                per_class_iou[b][c] = None  # class not present in this sample
    miou = float(np.mean(ious)) if ious else 0.0
    non_empty_miou = float(np.mean(non_empty_ious)) if non_empty_ious else 0.0
    target_miou = float(np.mean(target_ious)) if target_ious else 0.0
    target_acc = float(target_correct / target_total) if target_total > 0 else 0.0
    body_acc = float(body_correct / body_total) if body_total > 0 else 0.0

    # Diagnostic: prediction histogram (top 5 most-predicted slots) on the
    # whole eval batch + per-sample per-class IoU. Lets you see at a glance
    # whether the model has collapsed to "always predict slot 0 (empty)" —
    # which would show up as pred_top1=0 covering ~98% of voxels and per-class
    # IoU = {0: ~0.98, others: 0 or None}.
    pred_hist_full: dict = {}
    for v in labels.cpu().tolist():
        pred_hist_full[v] = pred_hist_full.get(v, 0) + 1
    pred_hist_top = sorted(pred_hist_full.items(), key=lambda kv: -kv[1])[:5]

    gt_hist_full: dict = {}
    for v in gt[supervised].cpu().tolist():
        gt_hist_full[v] = gt_hist_full.get(v, 0) + 1
    gt_hist_top = sorted(gt_hist_full.items(), key=lambda kv: -kv[1])[:5]

    pred_on_non_empty_hist: dict = {}
    for v in labels[non_empty].cpu().tolist():
        pred_on_non_empty_hist[v] = pred_on_non_empty_hist.get(v, 0) + 1
    pred_on_non_empty_top = sorted(
        pred_on_non_empty_hist.items(), key=lambda kv: -kv[1],
    )[:5]

    if was_training:
        model.train()
    return {
        'acc': acc,
        'non_empty_acc': non_empty_acc,
        'target_acc': target_acc,
        'body_acc': body_acc,
        'mIoU': miou,
        'non_empty_mIoU': non_empty_miou,
        'target_mIoU': target_miou,
        'num_ious': len(ious),
        'per_class_iou': per_class_iou,
        'gt_hist_top5': gt_hist_top,
        'pred_hist_top5': pred_hist_top,
        'pred_on_non_empty_top5': pred_on_non_empty_top,
    }


_EVAL_ALL_SCALAR_KEYS = (
    'acc',
    'non_empty_acc',
    'target_acc',
    'body_acc',
    'mIoU',
    'non_empty_mIoU',
    'target_mIoU',
)


def _mean_eval_scalars(rows):
    """Unweighted mean over per-sample eval rows."""
    assert rows, 'cannot summarize empty eval rows'
    return {
        key: float(np.mean([float(row[key]) for row in rows]))
        for key in _EVAL_ALL_SCALAR_KEYS
    }


@torch.no_grad()
def eval_all_samples(
    model,
    bridge,
    eval_loader,
    num_steps: int,
    solver: str,
    device,
    max_samples: int | None = None,
) -> dict:
    """Evaluate every sample from ``eval_loader`` and return mean + table rows.

    ``eval_loader`` must use batch_size=1 because eval/inference chunking is
    intentionally single-sample. The output is meant for overfit diagnostics:
    keep fixed-batch eval for a stable trend line, and use this all-sample pass
    to catch sample-specific failures.
    """
    rows = []
    for idx, batch in enumerate(eval_loader):
        if max_samples is not None and idx >= max_samples:
            break
        em = eval_accuracy(
            model, bridge, batch,
            num_steps=num_steps, solver=solver, device=device,
        )
        obj_id = batch.get('obj_id', ['<unknown>'])[0]
        sample_id = batch.get('sample_id', ['<unknown>'])[0]
        target_names = batch.get('target_part_names', [[]])[0]
        pred_top1 = em['pred_hist_top5'][0] if em['pred_hist_top5'] else None
        rows.append({
            **em,
            'obj_id': obj_id,
            'sample_id': sample_id,
            'target_part_names': target_names,
            'num_parts': int(batch['num_parts'][0]),
            'pred_top1': pred_top1,
        })
    assert rows, 'eval_all_samples received an empty eval_loader'
    return {
        'samples': len(rows),
        'mean': _mean_eval_scalars(rows),
        'rows': rows,
    }


# ---------------------------------------------------------------------------
# Stage entry-point (D-12 dispatch contract)
# ---------------------------------------------------------------------------

def train(config, *, load_dir: str = None, resume_step: int = None) -> None:
    """Stage entry-point invoked by TRELLIS-arts/train_arts.py.

    Args:
        config: OmegaConf DictConfig loaded from YAML.
        load_dir: optional checkpoint directory for resume.
        resume_step: optional step number for resume.
    """
    cfg = config
    rank, world_size, is_ddp = setup_ddp()
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    device = torch.device(f'cuda:{local_rank}') if torch.cuda.is_available() else torch.device('cpu')

    cfg_dict = config_to_dict(cfg)
    if (load_dir is None) != (resume_step is None):
        raise ValueError('load_dir and resume_step must be provided together')

    if is_main_process():
        print(f'\n[Part Flow Train] device: {device} | world_size: {world_size}')

    # Seed
    seed = int(cfg.training.get('seed', 42)) + rank
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

    # Flow config — fisher (default) | gumbel
    flow_cfg = dict(cfg_dict.get('flow', {}))
    flow_type = flow_cfg.pop('type', 'fisher')
    parameterization = flow_cfg.pop('parameterization', 'endpoint_logits')
    k_max = int(flow_cfg.pop('k_max', 128))
    solver = flow_cfg.pop('solver', 'euler')
    bridge_kwargs = {'k_max': k_max}
    for key in (
        't_max',
        'tau_max',
        'decay_rate',
        'noise_scale',
        'tau_min',
        'eps',
        'dirichlet_alpha',
    ):
        if key in flow_cfg:
            bridge_kwargs[key] = flow_cfg.pop(key)
    bridge = build_bridge(flow_type, **bridge_kwargs)
    if is_main_process():
        print(f'  flow: type={flow_type}, param={parameterization}, '
              f'k_max={k_max}, t_max={bridge.t_max}, solver={solver}')

    # Dataset
    data_cfg = dict(cfg_dict['data'])
    surface_cfg = dict(cfg_dict.get('surface', {}))
    if 'dropout_min' in surface_cfg:
        data_cfg['surface_dropout_min'] = surface_cfg['dropout_min']
    if 'dropout_max' in surface_cfg:
        data_cfg['surface_dropout_max'] = surface_cfg['dropout_max']
    data_cfg['max_k'] = k_max
    dataset = PartFlowDataset(data_cfg)
    assert len(dataset) > 0, 'Dataset empty — check data_root / part_info availability.'

    batch_size = int(cfg.training.get('batch_size', 1))
    assert batch_size <= 1 or len(dataset) >= batch_size, (
        f'Dataset has {len(dataset)} samples but batch_size={batch_size}.'
    )

    sampler = None
    shuffle = True
    if is_ddp:
        sampler = torch.utils.data.distributed.DistributedSampler(
            dataset, num_replicas=world_size, rank=rank, shuffle=True,
        )
        shuffle = False
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle,
        num_workers=int(cfg.training.get('num_workers', 4)),
        collate_fn=PartFlowDataset.collate_fn,
        drop_last=(batch_size > 1), sampler=sampler, pin_memory=True,
    )
    if is_main_process():
        total_bs = batch_size * world_size
        print(f'  dataset: {len(dataset)} samples | batch={batch_size}/gpu | '
              f'global_batch={total_bs}')

    eval_all_every = int(cfg.training.get('eval_all_every', 0))
    eval_all_max_samples_raw = cfg.training.get('eval_all_max_samples', None)
    eval_all_max_samples = (
        None
        if eval_all_max_samples_raw is None
        else int(eval_all_max_samples_raw)
    )
    eval_all_loader = None
    if is_main_process() and eval_all_every > 0:
        eval_data_cfg = dict(data_cfg)
        if eval_all_max_samples is not None:
            eval_data_cfg['num_samples'] = eval_all_max_samples
        # Full-dataset eval should be deterministic; the fixed training batch
        # still preserves one sampled dropout condition for trend tracking.
        eval_data_cfg['surface_dropout_min'] = 0.0
        eval_data_cfg['surface_dropout_max'] = 0.0
        eval_dataset = PartFlowDataset(eval_data_cfg)
        eval_all_loader = torch.utils.data.DataLoader(
            eval_dataset, batch_size=1, shuffle=False,
            num_workers=int(cfg.training.get('eval_all_num_workers', 1)),
            collate_fn=PartFlowDataset.collate_fn,
            drop_last=False, pin_memory=True,
        )
        print(f'  eval_all: {len(eval_dataset)} samples every {eval_all_every} steps '
              f'(surface_dropout=0.0)')

    # Model (variable-K)
    model_cfg = dict(cfg_dict['model'])
    model_cfg['k_max'] = k_max
    model = PartFlowPredictor(**model_cfg).to(device)
    if is_ddp:
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=False)
    raw_model = model.module if is_ddp else model
    if is_main_process():
        n_params = sum(p.numel() for p in raw_model.parameters())
        print(f'  model: PartFlowPredictor ({n_params:,} params) | '
              f'k_max={k_max} hidden_dim={model_cfg["hidden_dim"]}')

    # Loss
    loss_cfg = dict(cfg_dict.get('loss', {}))
    criterion = FlowMatchingLoss(
        bridge,
        parameterization=parameterization,
        empty_weight=float(loss_cfg.get('empty_weight', 0.05)),
        part_weight=float(loss_cfg.get('part_weight', 1.0)),
        focal_gamma=float(loss_cfg.get('focal_gamma', 2.0)),
        ignore_index=int(loss_cfg.get('ignore_index', -1)),
        reduction=str(loss_cfg.get('reduction', 'class_balanced')),
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg.training.lr),
        weight_decay=float(cfg.training.get('weight_decay', 0.01)),
    )
    scheduler = get_lr_scheduler(
        optimizer,
        warmup_steps=int(cfg.training.get('warmup_steps', 0)),
        max_steps=int(cfg.training.max_steps),
        min_lr_ratio=float(cfg.training.get('min_lr_ratio', 0.1)),
    )
    use_fp16 = bool(cfg.training.get('fp16', False)) and torch.cuda.is_available()
    scaler = torch.cuda.amp.GradScaler(enabled=use_fp16) if use_fp16 else None

    ema_decay = float(cfg.training.get('ema_decay', 0.0))
    ema = EMA(raw_model, decay=ema_decay) if ema_decay > 0 else None
    if ema and is_main_process():
        print(f'  EMA enabled (decay={ema_decay})')

    # Wandb
    wandb_run = None
    if is_main_process() and cfg_dict.get('wandb', {}).get('enabled', False):
        import wandb
        wandb_run = wandb.init(
            project=cfg.wandb.get('project', 'part_flow'),
            name=cfg.wandb.get('name', None),
            config=cfg_dict,
        )
        print(f'  wandb: {wandb_run.url}')

    # Resume
    start_step = 0
    if load_dir and resume_step is not None:
        ckpt_path = os.path.join(load_dir, f'step_{resume_step}.pt')
        assert os.path.isfile(ckpt_path), f'Resume checkpoint not found: {ckpt_path}'
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        raw_model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        if 'scheduler' in ckpt:
            scheduler.load_state_dict(ckpt['scheduler'])
        if ema and 'ema' in ckpt:
            ema.load_state_dict(ckpt['ema'])
        start_step = ckpt['step']
        if is_main_process():
            print(f'  resume: step {start_step}')

    output_dir = cfg.training.output_dir
    ckpt_dir = os.path.join(output_dir, 'ckpts')
    if is_main_process():
        os.makedirs(ckpt_dir, exist_ok=True)
    if is_ddp:
        dist.barrier()

    max_steps = int(cfg.training.max_steps)
    grad_clip = float(cfg.training.get('grad_clip', 1.0))
    log_every = int(cfg.training.get('log_every', 50))
    eval_every = int(cfg.training.get('eval_every', 2500))
    checkpoint_every = int(cfg.training.get('checkpoint_every', 5000))
    eval_ode_steps = int(cfg.training.get('eval_ode_steps', 20))
    grad_accum_steps = int(cfg.training.get('grad_accum_steps', 1))
    assert grad_accum_steps >= 1, f'grad_accum_steps must be >=1, got {grad_accum_steps}'

    eval_batch = None
    model.train()
    epoch = 0
    data_iter = None
    t_start = time.time()
    last_loss = None

    if is_main_process():
        print(f'\n[Part Flow Train] starting training: {max_steps} steps\n')

    for step in range(start_step + 1, max_steps + 1):
        optimizer.zero_grad(set_to_none=True)
        micro_metrics = []

        for micro_idx in range(grad_accum_steps):
            if data_iter is None:
                if sampler is not None:
                    sampler.set_epoch(epoch)
                data_iter = iter(dataloader)
            try:
                batch = next(data_iter)
            except StopIteration:
                epoch += 1
                if sampler is not None:
                    sampler.set_epoch(epoch)
                data_iter = iter(dataloader)
                batch = next(data_iter)

            if eval_batch is None:
                eval_batch = {
                    k: v for k, v in batch.items()
                    if k in ('coords', 'cond', 'mask_token_labels', 'per_voxel_labels',
                             'is_on_surface', 'voxel_layout', 'num_parts')
                }

            batch_dev = {
                'coords': batch['coords'].to(device),
                'cond': batch['cond'].to(device),
                'mask_token_labels': batch['mask_token_labels'].to(device),
                'per_voxel_labels': batch['per_voxel_labels'].to(device),
                'is_on_surface': batch['is_on_surface'].to(device),
                'voxel_layout': batch['voxel_layout'],
                'num_parts': batch['num_parts'],
            }

            sync_ctx = (
                model.no_sync()
                if is_ddp and micro_idx < grad_accum_steps - 1
                else contextlib.nullcontext()
            )
            with sync_ctx:
                with torch.cuda.amp.autocast(enabled=use_fp16):
                    loss, metrics = criterion(model, batch_dev)
                    loss_to_backprop = loss / grad_accum_steps

                if scaler is not None:
                    scaler.scale(loss_to_backprop).backward()
                else:
                    loss_to_backprop.backward()

            micro_metrics.append(metrics)

        if scaler is not None:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
        scheduler.step()
        if ema is not None:
            ema.update(raw_model)
        metrics = {
            key: float(np.mean([m[key] for m in micro_metrics]))
            for key in micro_metrics[0].keys()
        }
        last_loss = metrics['loss']

        if is_main_process() and (step % log_every == 0 or step == 1):
            lr = optimizer.param_groups[0]['lr']
            acc = metrics.get('endpoint_acc', 0.0)
            gt_p = metrics.get('gt_prob_mean', 0.0)
            print(f'  step {step:6d}/{max_steps} | loss={metrics["loss"]:.4f} '
                  f'ep_acc={acc:.3f} gt_p={gt_p:.3f} '
                  f'e/t/b_acc={metrics["empty_acc"]:.3f}/'
                  f'{metrics["target_acc"]:.3f}/{metrics["body_acc"]:.3f} '
                  f'e/t/b_frac={metrics["empty_frac"]:.3f}/'
                  f'{metrics["target_frac"]:.3f}/{metrics["body_frac"]:.3f} '
                  f't_mean={metrics["t_mean"]:.2f} | lr={lr:.2e}')
            if wandb_run is not None:
                import wandb
                wandb.log({**metrics, 'lr': lr}, step=step)

        if is_main_process() and eval_every > 0 and step % eval_every == 0:
            eval_model = ema.shadow.to(device) if ema else raw_model
            em = eval_accuracy(eval_model, bridge, eval_batch,
                               num_steps=eval_ode_steps, solver=solver, device=device)
            print(f'  [EVAL_FIXED @ {step}] acc={em["acc"]:.4f} '
                  f'non_empty_acc={em["non_empty_acc"]:.4f} '
                  f'target_acc={em["target_acc"]:.4f} body_acc={em["body_acc"]:.4f} '
                  f'mIoU={em["mIoU"]:.4f} non_empty_mIoU={em["non_empty_mIoU"]:.4f} '
                  f'target_mIoU={em["target_mIoU"]:.4f} '
                  f'({em["num_ious"]} per-class IoUs averaged)')
            # Diagnostic: per-class IoU per sample + global prediction histogram.
            # Reveals empty-collapse failure mode where pred=all 0 gives mIoU
            # ≈ acc/num_parts (slot-0 IoU full, all other slots zero).
            print(f'    per-class IoU per sample: {em["per_class_iou"]}')
            print(f'    gt histogram (top 5):    {em["gt_hist_top5"]}')
            print(f'    pred histogram (top 5):  {em["pred_hist_top5"]}')
            print(f'    pred on non-empty gt:    {em["pred_on_non_empty_top5"]}')
            if wandb_run is not None:
                import wandb
                wandb.log({
                    'eval/acc': em['acc'],
                    'eval/non_empty_acc': em['non_empty_acc'],
                    'eval/target_acc': em['target_acc'],
                    'eval/body_acc': em['body_acc'],
                    'eval/mIoU': em['mIoU'],
                    'eval/non_empty_mIoU': em['non_empty_mIoU'],
                    'eval/target_mIoU': em['target_mIoU'],
                }, step=step)

        if (
            is_main_process()
            and eval_all_loader is not None
            and eval_all_every > 0
            and step % eval_all_every == 0
        ):
            eval_model = ema.shadow.to(device) if ema else raw_model
            ea = eval_all_samples(
                eval_model, bridge, eval_all_loader,
                num_steps=eval_ode_steps, solver=solver, device=device,
            )
            mean = ea['mean']
            print(f'  [EVAL_ALL @ {step}] samples={ea["samples"]} '
                  f'acc={mean["acc"]:.4f} non_empty_acc={mean["non_empty_acc"]:.4f} '
                  f'target_acc={mean["target_acc"]:.4f} body_acc={mean["body_acc"]:.4f} '
                  f'mIoU={mean["mIoU"]:.4f} non_empty_mIoU={mean["non_empty_mIoU"]:.4f} '
                  f'target_mIoU={mean["target_mIoU"]:.4f}')
            print('    per-sample summary:')
            for i, row in enumerate(ea['rows']):
                pred_top = row['pred_top1']
                print(
                    f'      {i:02d} obj={row["obj_id"]} K={row["num_parts"]} '
                    f'target_mIoU={row["target_mIoU"]:.4f} '
                    f'non_empty_mIoU={row["non_empty_mIoU"]:.4f} '
                    f'target_acc={row["target_acc"]:.4f} '
                    f'body_acc={row["body_acc"]:.4f} pred_top1={pred_top}'
                )
            if wandb_run is not None:
                import wandb
                wandb.log({
                    'eval_all/acc': mean['acc'],
                    'eval_all/non_empty_acc': mean['non_empty_acc'],
                    'eval_all/target_acc': mean['target_acc'],
                    'eval_all/body_acc': mean['body_acc'],
                    'eval_all/mIoU': mean['mIoU'],
                    'eval_all/non_empty_mIoU': mean['non_empty_mIoU'],
                    'eval_all/target_mIoU': mean['target_mIoU'],
                }, step=step)

        if step % checkpoint_every == 0 and is_main_process():
            ckpt_path = os.path.join(ckpt_dir, f'step_{step}.pt')
            save_dict = {
                'model': raw_model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'step': step,
                'loss': last_loss,
                'flow_type': flow_type,
                'k_max': k_max,
                # Phase 9 fix (2026-04-26): bake YAML config into ckpt so
                # inference can rebuild PartFlowPredictor with the same dims
                # without needing the YAML on hand. Inference's
                # `_load_part_flow` reads ckpt['config'].
                'config': {
                    'flow': dict(cfg_dict.get('flow', {})),
                    'model': dict(cfg_dict.get('model', {})),
                    'training': {
                        'eval_ode_steps': int(cfg_dict.get('training', {}).get('eval_ode_steps', 10)),
                    },
                },
            }
            if ema is not None:
                save_dict['ema'] = ema.state_dict()
            torch.save(save_dict, ckpt_path)
            print(f'  [CKPT] saved: {ckpt_path}')

    if is_main_process():
        final_ckpt = os.path.join(ckpt_dir, f'step_{max_steps}.pt')
        save_dict = {
            'model': raw_model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'step': max_steps,
            'loss': last_loss,
            'flow_type': flow_type,
            'k_max': k_max,
            # Same config payload as the periodic ckpts above.
            'config': {
                'flow': dict(cfg_dict.get('flow', {})),
                'model': dict(cfg_dict.get('model', {})),
                'training': {
                    'eval_ode_steps': int(cfg_dict.get('training', {}).get('eval_ode_steps', 10)),
                },
            },
        }
        if ema is not None:
            save_dict['ema'] = ema.state_dict()
        torch.save(save_dict, final_ckpt)
        elapsed = time.time() - t_start
        print(f'\n[Part Flow Train] done: {max_steps} steps | '
              f'final loss={last_loss} | {elapsed:.1f}s')
    if wandb_run is not None:
        import wandb
        wandb.finish()
    cleanup_ddp()


# ---- CLI fallback ----------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description='Variable-K Part Flow training')
    p.add_argument('--config', type=str, required=True)
    p.add_argument('--load-dir', type=str, default=None)
    p.add_argument('--resume-step', type=int, default=None)
    p.add_argument('overrides', nargs='*', default=[])
    return p.parse_args()


def main():
    """CLI fallback for single-stage debugging."""
    args = parse_args()
    cfg = load_config(args.config, overrides=args.overrides or None)
    train(cfg, load_dir=args.load_dir, resume_step=args.resume_step)


if __name__ == '__main__':
    main()
