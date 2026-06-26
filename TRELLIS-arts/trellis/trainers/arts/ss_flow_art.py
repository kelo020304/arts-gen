#!/usr/bin/env python3
"""Stage trainer: ss_flow_art (multi-view conditioned Sparse Structure Flow Matching).

Migrated from scripts/train/stage2/train.py in Plan 09-03 (Phase 9 refactor).
The migration removed the sys.path / types.ModuleType bootstrap shim — the
trainer now lives inside the `trellis` package, so `from trellis.X import Y`
works directly without manual stubbing. The minimal-deps registration is
performed once by `TRELLIS-arts/train_arts.py` at process entry (CLAUDE.md
"Lessons Learned: 训练入口必须最小依赖").

Public API:
    train(config) -> None
        Stage entry-point invoked by TRELLIS-arts/train_arts.py dispatch.

CLI fallback (single-stage debugging):
    python -m trellis.trainers.arts.ss_flow_art \
        --config TRELLIS-arts/configs/arts/ss_flow_art/mv_4view.yaml
"""

import argparse
import hashlib
import json
import os
import random

import numpy as np
import torch
import torch.distributed as dist

# --- Project-root anchor for TORCH_HOME / pretrained ckpts -------------------
# This module lives at TRELLIS-arts/trellis/trainers/arts/ss_flow_art.py
# 4 levels up = repo root.
_HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(_HERE, "..", "..", "..", ".."))

os.environ.setdefault('TORCH_HOME', os.path.join(PROJECT_ROOT, 'submodules', 'TRELLIS.1'))
os.environ.setdefault('ATTN_BACKEND', 'sdpa')

# --- trellis-internal imports (resolve via real package; no stubs) -----------
from trellis.models.sparse_structure_flow import SparseStructureFlowModel
from trellis.trainers.flow_matching.flow_matching import (
    ImageConditionedFlowMatchingCFGTrainer,
)
from trellis.trainers.flow_matching.mixins.classifier_free_guidance import (
    ClassifierFreeGuidanceMixin,
)
from trellis.utils.arts.config_utils import load_config, config_to_dict
from trellis.utils.arts.lora_utils import apply_lora_to_model
from trellis.utils.arts.anchor_utils import L2SPAnchor
from trellis.utils.arts.ddp_utils import setup_ddp
from trellis.trainers.arts.mixins.wandb import WandbMixin
from trellis.datasets.arts.ss_flow_art import MvImageConditionedSLatDataset


# ---- Dynamic Trainer composition --------------------------------------------
class Stage2Trainer(WandbMixin, ImageConditionedFlowMatchingCFGTrainer):
    """Stage 2 (ss_flow_art) trainer — DENSE 16^3 SS Flow.

    MRO: WandbMixin -> ImageConditionedMixin -> ClassifierFreeGuidanceMixin
         -> FlowMatchingTrainer -> BasicTrainer -> Trainer

    Phase 9 fix (2026-04-26): parent was previously
    ImageConditionedSparseFlowMatchingCFGTrainer (sparse), but
    SparseStructureFlowModel.forward(x: torch.Tensor) is dense and
    encode_ss_latent_expanded.py writes dense `mean: [8,16,16,16]`. The
    sparse parent expected SparseTensor + `.feats` access — incompatible.
    Switching to dense ImageConditionedFlowMatchingCFGTrainer aligns the
    full stack: dataset (dense) → collate (stack) → trainer (dense MSE)
    → model (dense forward).

    Key overrides (CLAUDE.md "Lessons Learned: TRELLIS Trainer MRO 必须一次性审完"):
    - get_cond(): skip ImageConditionedMixin.encode_image() because the dataset
      provides pre-encoded DINOv2 tokens. Hand them straight to the CFG mixin.
    - snapshot_dataset(): no-op (data is sparse latent + tokens, not images).
    - snapshot(): skip init/intermediate snapshots (pre-encoded tokens cannot be
      visualized through ImageConditionedMixin).
    """

    def get_cond(self, cond, **kwargs):
        """Skip encode_image; tokens already pre-encoded by the dataset."""
        neg_cond = torch.zeros_like(cond)
        return ClassifierFreeGuidanceMixin.get_cond(self, cond, neg_cond=neg_cond, **kwargs)

    def snapshot_dataset(self, num_samples=100):
        """Skip dataset visualization (data is non-image)."""
        if self.is_master:
            print('[Stage2Trainer] snapshot_dataset 跳过（数据非图像格式）')

    def snapshot(self, suffix=None, num_samples=64, batch_size=4, verbose=False):
        """Skip model sampling snapshots (pre-encoded tokens are non-visualizable)."""
        if self.is_master:
            suffix = suffix or f'step{self.step:07d}'
            print(f'[Stage2Trainer] snapshot({suffix}) 跳过（预编码 tokens 无法可视化）')


# ---- Helpers ---------------------------------------------------------------

def ensure_manifest_split(cfg, rank: int = 0) -> None:
    """Auto-split manifest.json into manifest_train.json / manifest_val.json.

    Distributed-safe: only rank 0 performs IO; all ranks barrier before returning.
    Bucket = md5(obj_id) % 10000 / 10000; val_ratio = 0.1.
    """
    from tqdm import tqdm

    data_root = getattr(cfg.data, 'data_root', 'data/PhysX-Mobility')
    manifest_path_rel = getattr(cfg.data, 'manifest_path', None)
    if not manifest_path_rel:
        if dist.is_initialized():
            dist.barrier()
        return

    if os.path.isabs(manifest_path_rel):
        manifest_dir = os.path.dirname(manifest_path_rel)
    else:
        manifest_dir = os.path.dirname(
            os.path.join(PROJECT_ROOT, data_root, manifest_path_rel)
        )

    train_path = os.path.join(manifest_dir, 'manifest_train.json')
    val_path   = os.path.join(manifest_dir, 'manifest_val.json')
    src_path   = os.path.join(manifest_dir, 'manifest.json')

    if os.path.exists(train_path) and os.path.exists(val_path):
        if dist.is_initialized():
            dist.barrier()
        return

    if rank == 0:
        if not os.path.exists(src_path):
            print(f'[AUTO-SPLIT] 无法分割：找不到源 manifest: {src_path}')
        else:
            print(f'[AUTO-SPLIT] manifest_train.json / manifest_val.json 不存在')
            print(f'[AUTO-SPLIT] 正在从 {src_path} 分割...')

            with open(src_path, 'r') as f:
                data = json.load(f)

            if not isinstance(data, dict) or 'samples' not in data:
                print('[AUTO-SPLIT] ERROR: 不支持的 manifest 格式，期望 {"samples": [...]}，跳过')
            else:
                samples = data['samples']

                by_obj: dict = {}
                for s in samples:
                    by_obj.setdefault(str(s['object_id']), []).append(s)
                obj_ids = sorted(by_obj.keys())

                train_samples, val_samples = [], []
                train_obj_ids, val_obj_ids = [], []
                for obj_id in tqdm(obj_ids, desc='[AUTO-SPLIT] 分割 manifest', unit='obj'):
                    digest = hashlib.md5(obj_id.encode('utf-8')).hexdigest()
                    bucket = (int(digest[:16], 16) % 10000) / 10000.0
                    if bucket < 0.1:
                        val_samples.extend(by_obj[obj_id])
                        val_obj_ids.append(obj_id)
                    else:
                        train_samples.extend(by_obj[obj_id])
                        train_obj_ids.append(obj_id)

                train_samples.sort(key=lambda s: (s['object_id'], s['angle_idx']))
                val_samples.sort(key=lambda s: (s['object_id'], s['angle_idx']))

                with open(train_path, 'w') as f:
                    json.dump({'samples': train_samples}, f, indent=2, ensure_ascii=False)
                with open(val_path, 'w') as f:
                    json.dump({'samples': val_samples}, f, indent=2, ensure_ascii=False)

                print(f'[AUTO-SPLIT] train: {len(train_obj_ids)} obj / {len(train_samples)} samples')
                print(f'[AUTO-SPLIT] val:   {len(val_obj_ids)} obj / {len(val_samples)} samples')
                print(f'[AUTO-SPLIT] 已写入 {train_path}')
                print(f'[AUTO-SPLIT] 已写入 {val_path}')

    if dist.is_initialized():
        dist.barrier()


def setup_rng(seed: int = 42):
    """Set deterministic seeds for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


# ---- Stage entry-point (D-12 dispatch contract) ----------------------------

def train(config, *, load_dir: str = None, resume_step: int = None,
          dump_param_stats: bool = False) -> None:
    """Stage entry-point invoked by TRELLIS-arts/train_arts.py.

    Args:
        config: OmegaConf DictConfig already loaded from YAML.
        load_dir: optional checkpoint directory for resume.
        resume_step: optional step number for resume (paired with load_dir).
        dump_param_stats: if True, print parameter hashes before/after training
            (used by LoRA freeze verification harness).

    Returns:
        None.
    """
    cfg = config

    # ---- 1. Distributed init ----
    rank, local_rank, world_size = setup_ddp()
    is_distributed = world_size > 1
    setup_rng(seed=42 + rank)

    if rank == 0:
        print('\n[Stage2 Train] config loaded:')
        print(f'  distributed: {is_distributed} | world_size={world_size}')

    # ---- 2.5 Auto-split manifest if needed ----
    ensure_manifest_split(cfg, rank=rank)

    # ---- 3. Build Dataset ----
    data_cfg = config_to_dict(cfg.data)
    dataset = MvImageConditionedSLatDataset(data_cfg)

    # ---- 4. Build Model ----
    model_cfg = config_to_dict(cfg.model)
    model_cfg.pop('name', None)  # name is identifier only
    model_args = model_cfg.pop('args', model_cfg)
    model = SparseStructureFlowModel(**model_args).cuda()

    if rank == 0:
        num_params = sum(p.numel() for p in model.parameters())
        num_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f'\n[Stage2 Train] 模型: SparseStructureFlowModel')
        print(f'  总参数量: {num_params:,}')
        print(f'  可训练参数量: {num_trainable:,}')

    # ---- 4.5 Extract LoRA config (apply later, after pretrained load) ----
    lora_cfg = config_to_dict(cfg.lora) if 'lora' in cfg else {}
    lora_enabled = lora_cfg.get('enabled', False)

    # ---- 5. Training config ----
    training_cfg = config_to_dict(cfg.training)
    output_dir = training_cfg.pop('output_dir', 'output/ss_flow_art_default')
    pretrained_ckpt_original = training_cfg.get('pretrained_ckpt', None)
    pretrained_ckpt = training_cfg.pop('pretrained_ckpt', None)

    if rank == 0:
        os.makedirs(output_dir, exist_ok=True)

    # ---- 5.5 Pretrained / resume ordering ----
    is_resuming = load_dir is not None and resume_step is not None
    if is_resuming and pretrained_ckpt is not None:
        if rank == 0:
            print('[Stage2 Train] Resume 模式，跳过 pretrained_ckpt（避免覆盖 checkpoint 权重）')
        pretrained_ckpt = None

    pretrained_loaded = False
    if pretrained_ckpt is not None:
        ckpt_path = os.path.join(PROJECT_ROOT, pretrained_ckpt)
        if os.path.exists(ckpt_path):
            if ckpt_path.endswith('.safetensors'):
                try:
                    from safetensors.torch import load_file
                    state_dict = load_file(ckpt_path)
                    missing, unexpected = model.load_state_dict(state_dict, strict=False)
                    pretrained_loaded = True
                    if rank == 0:
                        print(f'[Stage2 Train] 从 safetensors 加载预训练权重: {ckpt_path}')
                        if missing:
                            print(f'  missing keys: {len(missing)}')
                        if unexpected:
                            print(f'  unexpected keys: {len(unexpected)}')
                except ImportError:
                    if rank == 0:
                        print(f'[ERROR] safetensors 未安装，无法加载 {ckpt_path}')
                        print(f'        请运行: pip install safetensors')
                        print(f'        模型将从随机初始化开始（未加载预训练权重）')
            else:
                ckpt_data = torch.load(ckpt_path, map_location='cuda', weights_only=True)
                missing, unexpected = model.load_state_dict(ckpt_data, strict=False)
                pretrained_loaded = True
                if rank == 0:
                    print(f'[Stage2 Train] 从 pt 加载预训练权重: {ckpt_path}')
                    if missing:
                        print(f'  missing keys: {len(missing)}')
                    if unexpected:
                        print(f'  unexpected keys: {len(unexpected)}')
            if rank == 0 and pretrained_loaded:
                print(f'[Stage2 Train] 预训练权重加载完成')
        else:
            if rank == 0:
                print(f'[WARN] 预训练权重不存在: {ckpt_path}，从随机初始化开始')

    # ---- 5.6 Apply LoRA (after pretrained load, before trainer ctor) ----
    if lora_enabled:
        model = apply_lora_to_model(model, lora_cfg)
        if rank == 0:
            trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
            total = sum(p.numel() for p in model.parameters())
            print(f'\n[Stage2 Train] LoRA 已启用:')
            print(f'  rank={lora_cfg.get("rank", 16)}, target={lora_cfg.get("target_modules", "all_attn")}')
            print(f'  可训练参数: {trainable:,} / {total:,} ({trainable/total*100:.2f}%)')

    # ---- 5.7 L2-SP Anchor ----
    anchor_cfg = config_to_dict(cfg.anchor) if 'anchor' in cfg else {}
    anchor_enabled = anchor_cfg.get('enabled', False)
    anchor = None
    if anchor_enabled:
        if pretrained_ckpt_original is None:
            if rank == 0:
                print('[Anchor] WARN: anchor.enabled=True 但 pretrained_ckpt 未配置，跳过 anchor 初始化')
        else:
            anchor_ckpt_path = (
                os.path.join(PROJECT_ROOT, pretrained_ckpt_original)
                if not os.path.isabs(pretrained_ckpt_original)
                else pretrained_ckpt_original
            )
            if not os.path.exists(anchor_ckpt_path):
                if rank == 0:
                    print(f'[Anchor] WARN: anchor source 不存在: {anchor_ckpt_path}，跳过 anchor 初始化')
            else:
                if anchor_ckpt_path.endswith('.safetensors'):
                    from safetensors.torch import load_file as load_safetensors
                    pretrained_state = load_safetensors(anchor_ckpt_path)
                else:
                    pretrained_state = torch.load(
                        anchor_ckpt_path,
                        map_location='cpu',
                        weights_only=True,
                    )
                anchor = L2SPAnchor.from_state_dict(
                    model,
                    pretrained_state,
                    lambda_=anchor_cfg.get('lambda', 1.0e-4),
                    target=anchor_cfg.get('target', 'trainable'),
                )
                anchor.attach()
                del pretrained_state
                if rank == 0:
                    print(f'[Anchor] L2-SP enabled, lambda={anchor.lambda_}, '
                          f'target={anchor.target}, source={anchor_ckpt_path}')

    # ---- 6. Build Trainer ----
    wandb_config = config_to_dict(cfg.wandb) if 'wandb' in cfg else None

    trainer = Stage2Trainer(
        models={'denoiser': model},
        dataset=dataset,
        output_dir=output_dir,
        load_dir=load_dir,
        step=resume_step,
        wandb_config=wandb_config,
        **training_cfg,
    )

    # ---- 7. Optional: param stats before training (LoRA freeze verification) ----
    param_snapshot_before = None
    if dump_param_stats and rank == 0:
        param_snapshot_before = {}
        for name, param in model.named_parameters():
            h = hashlib.md5(param.data.cpu().numpy().tobytes()).hexdigest()
            param_snapshot_before[name] = {
                'hash': h,
                'requires_grad': param.requires_grad,
                'numel': param.numel(),
            }
        total = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f'\n[PARAM_STATS_BEFORE] total={total} trainable={trainable} ratio={trainable/total*100:.4f}%')

    # ---- 8. Train ----
    if rank == 0:
        print('\n[Stage2 Train] 开始训练...')
    trainer.run()

    # ---- 9. Optional: param stats after training ----
    if dump_param_stats and rank == 0 and param_snapshot_before is not None:
        changed_params = []
        unchanged_params = []
        for name, param in model.named_parameters():
            h = hashlib.md5(param.data.cpu().numpy().tobytes()).hexdigest()
            if h != param_snapshot_before[name]['hash']:
                changed_params.append(name)
            else:
                unchanged_params.append(name)
        keep_patterns = lora_cfg.get('keep_trainable', []) or []
        lora_changed = [n for n in changed_params if 'lora_' in n]
        hybrid_changed = [
            n for n in changed_params
            if 'lora_' not in n and keep_patterns and any(pat in n for pat in keep_patterns)
        ]
        hybrid_changed_set = set(hybrid_changed)
        non_lora_changed = [
            n for n in changed_params
            if 'lora_' not in n and n not in hybrid_changed_set
        ]
        print(f'\n[PARAM_STATS_AFTER] changed={len(changed_params)} unchanged={len(unchanged_params)}')
        print(f'[PARAM_STATS_AFTER] lora_changed={len(lora_changed)} '
              f'hybrid_changed={len(hybrid_changed)} non_lora_changed={len(non_lora_changed)}')
        if hybrid_changed:
            print(f'[PARAM_STATS_AFTER] OK (hybrid): keep_trainable layers changed as expected: '
                  f'{hybrid_changed[:5]}')
        if non_lora_changed:
            print(f'[PARAM_STATS_AFTER] WARNING: 非 LoRA 且非 keep_trainable 参数发生变化: '
                  f'{non_lora_changed[:5]}')
        else:
            print(f'[PARAM_STATS_AFTER] OK: 非 LoRA 且非 keep_trainable 参数全部冻结')


# ---- CLI fallback ----------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description='Stage 2 (ss_flow_art) multi-view conditioned Sparse Structure Flow Matching training'
    )
    parser.add_argument('--config', type=str, required=True,
                        help='YAML 配置文件路径（支持 _base_ 继承）')
    parser.add_argument('--load-dir', type=str, default=None,
                        help='从此目录加载 checkpoint resume 训练')
    parser.add_argument('--resume-step', type=int, default=None,
                        help='resume 的 step 编号（需配合 --load-dir）')
    parser.add_argument('--dump-param-stats', action='store_true', default=False,
                        help='训练前后打印参数统计（用于 LoRA 冻结验证）')
    parser.add_argument('overrides', nargs='*', default=[],
                        help='OmegaConf 覆盖，格式: key=value')
    return parser.parse_args()


def main():
    """CLI fallback for single-stage debugging.

    Production training uses TRELLIS-arts/train_arts.py which calls train(config)
    directly via the dispatch table.
    """
    args = parse_args()
    cfg = load_config(args.config, overrides=args.overrides if args.overrides else None)
    train(
        cfg,
        load_dir=args.load_dir,
        resume_step=args.resume_step,
        dump_param_stats=args.dump_param_stats,
    )


if __name__ == '__main__':
    main()
