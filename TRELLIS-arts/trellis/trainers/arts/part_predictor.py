#!/usr/bin/env python3
"""Stage trainer: part_predictor (Hungarian matching + mask CE + dice + class CE).

Migrated from scripts/train/part_predictor/train.py in Plan 09-03 (Phase 9 refactor).
Removed sys.path / types.ModuleType bootstrap shim — minimal-deps registration
is performed once by TRELLIS-arts/train_arts.py.

Public API:
    train(config) -> None — Stage entry-point invoked by train_arts.py dispatch.
"""

import argparse
import math
import os
import random
import sys
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
os.environ.setdefault('ATTN_BACKEND', 'sdpa')

# --- trellis-internal imports ------------------------------------------------
from trellis.utils.arts.config_utils import load_config, config_to_dict
from trellis.utils.arts.ddp_utils import setup_ddp as _setup_ddp_canonical
from trellis.models.part_predictor.part_predictor import QueryPartPredictor
from trellis.models.part_predictor.losses import PartPredictorLoss
from trellis.datasets.arts.part_predictor import PartPredictorDataset


# ---------------------------------------------------------------------------
# DDP helpers
# ---------------------------------------------------------------------------

def setup_ddp():
    """Init DDP from torchrun env vars. Returns (rank, world_size, is_ddp)."""
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank, _local_rank, world_size = _setup_ddp_canonical()
        return rank, world_size, True
    return 0, 1, False


def cleanup_ddp():
    if dist.is_initialized():
        dist.destroy_process_group()


def is_main_process():
    return not dist.is_initialized() or dist.get_rank() == 0


def get_lr_scheduler(optimizer, warmup_steps: int, max_steps: int):
    def lr_lambda(step):
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        return 1.0
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


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
    device = torch.device(f'cuda:{local_rank}')

    cfg_dict = config_to_dict(cfg)

    if is_main_process():
        print(f'\n[Part Predictor Train] device: {device} | world_size: {world_size}')

    # Seed (per-rank for data shuffle diversity)
    seed = cfg.training.seed + rank
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

    # Dataset (propagate model.max_k as mask vocab upper bound)
    data_cfg = dict(cfg_dict['data'])
    data_cfg['max_k'] = int(cfg_dict['model']['max_k'])
    dataset = PartPredictorDataset(data_cfg)
    batch_size = int(cfg.training.get('batch_size', 1))

    sampler = None
    shuffle = True
    if is_ddp:
        sampler = torch.utils.data.distributed.DistributedSampler(
            dataset, num_replicas=world_size, rank=rank, shuffle=True,
        )
        shuffle = False

    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, num_workers=4,
        collate_fn=PartPredictorDataset.collate_fn,
        drop_last=True, sampler=sampler, pin_memory=True,
    )

    if is_main_process():
        total_bs = batch_size * world_size
        print(f'  dataset: {len(dataset)} samples | batch_size={batch_size}/gpu '
              f'| global_batch={total_bs}')

    if len(dataset) == 0:
        if is_main_process():
            print('[ERROR] Dataset is empty')
        cleanup_ddp()
        sys.exit(1)

    # Model
    model = QueryPartPredictor(**cfg_dict['model']).to(device)

    if is_ddp:
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=False)

    if is_main_process():
        raw_model = model.module if is_ddp else model
        num_params = sum(p.numel() for p in raw_model.parameters())
        print(f'  model: QueryPartPredictor ({num_params:,} params)')

    # Loss
    decode_aware_cfg = cfg_dict.get('decode_aware')
    criterion = PartPredictorLoss(
        **cfg_dict['loss'],
        decode_aware_cfg=decode_aware_cfg,
    )
    da_enabled = decode_aware_cfg and decode_aware_cfg.get('enabled', False)
    if is_main_process():
        if da_enabled:
            print(f'  decode-aware: ENABLED (weight={decode_aware_cfg.get("weight", 0.5)})')
        else:
            print(f'  decode-aware: disabled')

    # Optimizer + Scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.training.lr,
        weight_decay=cfg.training.weight_decay,
    )
    scheduler = get_lr_scheduler(
        optimizer,
        warmup_steps=cfg.training.get('warmup_steps', 0),
        max_steps=cfg.training.max_steps,
    )

    # Optional fp16
    use_fp16 = cfg.training.get('fp16', False)
    scaler = torch.cuda.amp.GradScaler(enabled=use_fp16) if use_fp16 else None

    # Optional Wandb (rank 0 only)
    wandb_run = None
    if is_main_process() and cfg_dict.get('wandb', {}).get('enabled', False):
        try:
            import wandb
            wandb_run = wandb.init(
                project=cfg.wandb.get('project', 'arts-part-predictor'),
                name=cfg.wandb.get('name', None),
                config=cfg_dict,
            )
            print(f'  wandb: {wandb_run.url}')
        except Exception as e:
            print(f'  [WARN] wandb 初始化失败: {e}')

    # Resume from checkpoint
    start_step = 0
    if load_dir and resume_step is not None:
        ckpt_path = os.path.join(load_dir, f'step_{resume_step}.pt')
        if os.path.isfile(ckpt_path):
            raw_model = model.module if is_ddp else model
            ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
            raw_model.load_state_dict(ckpt['model'])
            optimizer.load_state_dict(ckpt['optimizer'])
            if 'scheduler' in ckpt:
                scheduler.load_state_dict(ckpt['scheduler'])
            start_step = ckpt['step']
            if is_main_process():
                print(f'  resume: step {start_step}')
        else:
            if is_main_process():
                print(f'  [WARN] checkpoint not found: {ckpt_path}')

    # Output dir
    output_dir = cfg.training.output_dir
    ckpt_dir = os.path.join(output_dir, 'ckpts')
    if is_main_process():
        os.makedirs(ckpt_dir, exist_ok=True)
    if is_ddp:
        dist.barrier()

    # Training loop
    max_steps = cfg.training.max_steps
    grad_clip = cfg.training.grad_clip
    log_every = cfg.training.log_every
    checkpoint_every = cfg.training.checkpoint_every

    model.train()
    epoch = 0
    data_iter = None
    all_losses = []

    if is_main_process():
        print(f'\n[Part Predictor Train] starting training: {max_steps} steps')
    t_start = time.time()

    for step in range(start_step + 1, max_steps + 1):
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

        # Move packed batch tensors to device
        coords = batch['coords'].to(device)
        cond = batch['cond'].to(device)
        mask_token_labels = batch['mask_token_labels'].to(device)
        part_type_ids = [t.to(device) for t in batch['part_type_ids']]
        part_labels = batch['part_labels'].to(device)
        num_parts = list(batch['num_parts'])
        voxel_layout = batch['voxel_layout']
        B_cur = len(num_parts)

        # ---- Decode-aware (optional) ----
        z_slat_list = None
        gt_points_list = None
        d2s_list = None
        if da_enabled:
            gt_points_list = [
                [pts.to(device) for pts in gpp]
                for gpp in batch['gt_points_per_part']
            ]
            slat_coords_raw = batch.get('slat_coords')
            slat_feats_raw = batch.get('slat_feats')
            slat_layout = batch.get('slat_layout')
            if slat_coords_raw is not None and slat_feats_raw is not None and slat_layout is not None:
                from trellis.modules.sparse import SparseTensor
                sc_all = slat_coords_raw.to(device)
                sf_all = slat_feats_raw.to(device)
                z_slat_list = []
                d2s_list = []
                for b_idx in range(B_cur):
                    sl = slat_layout[b_idx]
                    sc_b_xyz = sc_all[sl, 1:4]
                    sf_b = sf_all[sl]
                    N_slat_b = sc_b_xyz.shape[0]
                    coords_4d = torch.cat([
                        torch.zeros(N_slat_b, 1, dtype=torch.int32, device=device),
                        sc_b_xyz.int(),
                    ], dim=1)
                    z_slat_list.append(SparseTensor(feats=sf_b, coords=coords_4d))
                    vs = voxel_layout[b_idx]
                    dense_xyz_b = coords[vs, 1:4].cpu().numpy()
                    sc_cpu = sc_b_xyz.cpu().numpy()
                    slat_map = {}
                    for j in range(N_slat_b):
                        slat_map[(int(sc_cpu[j, 0]), int(sc_cpu[j, 1]), int(sc_cpu[j, 2]))] = j
                    N_dense_b = dense_xyz_b.shape[0]
                    mapping = torch.full((N_dense_b,), -1, dtype=torch.long)
                    for i in range(N_dense_b):
                        key = (int(dense_xyz_b[i, 0]), int(dense_xyz_b[i, 1]), int(dense_xyz_b[i, 2]))
                        if key in slat_map:
                            mapping[i] = slat_map[key]
                    d2s_list.append(mapping.to(device))
                if is_main_process() and step == start_step + 1:
                    total_matched = sum((m >= 0).sum().item() for m in d2s_list)
                    total_dense = sum(m.numel() for m in d2s_list)
                    print(f'  [DA] dense↔slat alignment: {total_matched}/{total_dense} matched '
                          f'({100.0 * total_matched / max(1, total_dense):.1f}%)')
            else:
                if is_main_process() and step % 100 == 1:
                    print(f'  [WARN] step={step}: decode-aware ENABLED 但 batch 缺 SLat')

        # ---- Forward ----
        with torch.cuda.amp.autocast(enabled=use_fp16):
            pred = model(coords, cond, num_parts=num_parts, mask_token_labels=mask_token_labels)
            loss_dict = criterion(
                pred, part_labels, part_type_ids, num_parts,
                z_slat_st=z_slat_list,
                gt_points_per_part=gt_points_list,
                dense_to_slat_idx=d2s_list,
                voxel_layout=voxel_layout,
            )

        total_loss = loss_dict['loss']

        # ---- Backward ----
        optimizer.zero_grad()
        if scaler is not None:
            scaler.scale(total_loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

        scheduler.step()

        loss_val = total_loss.item()
        all_losses.append(loss_val)

        # ---- Log (rank 0) ----
        if is_main_process() and (step % log_every == 0 or step == 1):
            mask_ce_val = loss_dict['mask_ce'].item()
            dice_val = loss_dict['dice'].item()
            cls_ce_val = loss_dict['cls_ce'].item()
            lr_current = optimizer.param_groups[0]['lr']

            aux_str = ''
            aux_mce = loss_dict.get('aux_mask_ce')
            if aux_mce is not None:
                aux_str = f' aux_mce={aux_mce.item():.4f} aux_dice={loss_dict["aux_dice"].item():.4f}'

            da_loss_val = loss_dict.get('decode_aware_loss')
            chamfer_val = loss_dict.get('chamfer_mean')
            da_str = ''
            if da_loss_val is not None:
                da_str = f' da_loss={da_loss_val.item():.4f} chamfer={chamfer_val.item():.4f}'

            print(
                f'  step {step:6d}/{max_steps} | '
                f'loss={loss_val:.4f} mask_ce={mask_ce_val:.4f} '
                f'dice={dice_val:.4f} cls_ce={cls_ce_val:.4f}{aux_str}{da_str} | '
                f'lr={lr_current:.2e} K_mean={sum(num_parts)/len(num_parts):.1f} B={B_cur}x{world_size}'
            )

            if wandb_run is not None:
                try:
                    import wandb
                    log_dict = {
                        'loss': loss_val,
                        'mask_ce': mask_ce_val,
                        'dice': dice_val,
                        'cls_ce': cls_ce_val,
                        'lr': lr_current,
                        'num_parts_mean': sum(num_parts) / len(num_parts),
                        'batch_size': B_cur * world_size,
                    }
                    if aux_mce is not None:
                        log_dict['aux_mask_ce'] = aux_mce.item()
                        log_dict['aux_dice'] = loss_dict['aux_dice'].item()
                        log_dict['aux_cls_ce'] = loss_dict['aux_cls_ce'].item()
                    if da_loss_val is not None:
                        log_dict['decode_aware_loss'] = da_loss_val.item()
                        log_dict['chamfer_mean'] = chamfer_val.item()
                    wandb.log(log_dict, step=step)
                except Exception:
                    pass

        # ---- Checkpoint (rank 0) ----
        if step % checkpoint_every == 0 and is_main_process():
            raw_model = model.module if is_ddp else model
            ckpt_path = os.path.join(ckpt_dir, f'step_{step}.pt')
            torch.save({
                'model': raw_model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'step': step,
                'loss': loss_val,
            }, ckpt_path)
            print(f'  [CKPT] saved: {ckpt_path}')

    # ---- Final checkpoint ----
    if is_main_process():
        raw_model = model.module if is_ddp else model
        final_ckpt = os.path.join(ckpt_dir, f'step_{max_steps}.pt')
        torch.save({
            'model': raw_model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'step': max_steps,
            'loss': all_losses[-1] if all_losses else 0.0,
        }, final_ckpt)

        elapsed = time.time() - t_start
        print(f'\n[Part Predictor Train] training complete:')
        print(f'  total steps: {max_steps}')
        print(f'  final loss: {all_losses[-1]:.4f}' if all_losses else '  no loss recorded')
        print(f'  elapsed: {elapsed:.1f}s')
        print(f'  output: {output_dir}')

    if wandb_run is not None:
        try:
            import wandb
            wandb.finish()
        except Exception:
            pass

    cleanup_ddp()


# ---- CLI fallback ----------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description='Part Predictor training (Hungarian matching + mask CE + dice + class CE)'
    )
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--load-dir', type=str, default=None)
    parser.add_argument('--resume-step', type=int, default=None)
    parser.add_argument('overrides', nargs='*', default=[])
    return parser.parse_args()


def main():
    """CLI fallback for single-stage debugging."""
    args = parse_args()
    cfg = load_config(args.config, overrides=args.overrides if args.overrides else None)
    train(cfg, load_dir=args.load_dir, resume_step=args.resume_step)


if __name__ == '__main__':
    main()
