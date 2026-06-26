#!/usr/bin/env python3
"""Stage trainer: slat_flow_art (multi-view conditioned SLat Flow Matching).

Migrated from scripts/train/stage4/train.py in Plan 09-03 (Phase 9 refactor).
Removed sys.path / types.ModuleType bootstrap shim — minimal-deps registration
is now handled once at TRELLIS-arts/train_arts.py entry.

Key differences vs ss_flow_art trainer:
  - Base class: ImageConditionedSparseFlowMatchingCFGTrainer (Sparse, same)
  - Model:      ElasticSLatFlowModel (vs SparseStructureFlowModel)
  - Overrides BOTH get_cond AND get_inference_cond (snapshot path uses
    sampler.sample which calls get_inference_cond — must bypass DINOv2)
  - vis_cond returns {} (H-3 fix: pre-encoded tokens are not images)
  - run_snapshot: decoder-based snapshot via slat_render_utils

Public API:
    train(config) -> None — Stage entry-point invoked by train_arts.py dispatch.
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
_HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(_HERE, "..", "..", "..", ".."))

os.environ.setdefault('TORCH_HOME', os.path.join(PROJECT_ROOT, 'submodules', 'TRELLIS.1'))
os.environ.setdefault('ATTN_BACKEND', 'sdpa')

# --- trellis-internal imports ------------------------------------------------
from trellis.models.structured_latent_flow import ElasticSLatFlowModel
from trellis.trainers.flow_matching.sparse_flow_matching import (
    ImageConditionedSparseFlowMatchingCFGTrainer,
)
from trellis.trainers.flow_matching.mixins.classifier_free_guidance import (
    ClassifierFreeGuidanceMixin,
)
from trellis.utils.arts.config_utils import load_config, config_to_dict
from trellis.utils.arts.lora_utils import apply_lora_to_model
from trellis.utils.arts.anchor_utils import L2SPAnchor
from trellis.utils.arts.ddp_utils import setup_ddp
from trellis.trainers.arts.mixins.wandb import WandbMixin
from trellis.datasets.arts.slat_flow_art import MvImageConditionedSparseLatentDataset


# ---- Dynamic Trainer composition --------------------------------------------
class Stage4Trainer(WandbMixin, ImageConditionedSparseFlowMatchingCFGTrainer):
    """Stage 4 (slat_flow_art) trainer.

    MRO:
        WandbMixin -> ImageConditionedMixin -> ClassifierFreeGuidanceMixin
        -> SparseFlowMatchingTrainer -> FlowMatchingTrainer -> BasicTrainer -> Trainer

    Critical overrides (CLAUDE.md "Lessons Learned: TRELLIS Trainer MRO 必须一次性审完"):
      - get_cond: training_losses path (sparse_flow_matching.py:98), bypass DINOv2.
      - get_inference_cond: run_snapshot -> sampler.sample path; must override.
      - snapshot_dataset: no-op (data is sparse latent + tokens, not images).
      - snapshot: smoke-mode early exit; production calls super().snapshot.
      - vis_cond: returns {} (pre-encoded tokens are non-visualizable as images).
      - run_snapshot: decoder-based render via slat_render_utils.
    """

    def get_cond(self, cond, **kwargs):
        """Training condition: skip ImageConditionedMixin.encode_image."""
        neg_cond = torch.zeros_like(cond)
        return ClassifierFreeGuidanceMixin.get_cond(
            self, cond, neg_cond=neg_cond, **kwargs
        )

    def get_inference_cond(self, cond, **kwargs):
        """Inference condition: same as get_cond, skip DINOv2."""
        neg_cond = torch.zeros_like(cond)
        return ClassifierFreeGuidanceMixin.get_inference_cond(
            self, cond, neg_cond=neg_cond, **kwargs
        )

    def snapshot_dataset(self, num_samples=100):
        """Skip dataset visualization (data is sparse latent + tokens, not images)."""
        if self.is_master:
            print('[Stage4Trainer] snapshot_dataset skipped (pre-encoded tokens)')

    def snapshot(self, suffix=None, num_samples=64, batch_size=4, verbose=False):
        """Smoke-mode early exit; production runs run_snapshot.

        Smoke (i_sample >= max_steps and suffix != 'final'): skip to avoid
        50-step DDIM × N samples (slow). Final + production runs go through
        super().snapshot -> run_snapshot.
        """
        _smoke_mode = self.i_sample >= self.max_steps
        if _smoke_mode and suffix != 'final':
            if self.is_master:
                suffix_label = suffix or f'step{self.step:07d}'
                print(f'[Stage4Trainer] snapshot({suffix_label}) skipped '
                      f'(smoke mode: i_sample={self.i_sample} >= max_steps={self.max_steps})')
            return
        super().snapshot(suffix=suffix, num_samples=num_samples,
                         batch_size=batch_size, verbose=verbose)

    def vis_cond(self, cond=None, **kwargs):
        """H-3 fix: pre-encoded tokens are non-visualizable as images.

        BasicTrainer.run_snapshot calls cond_vis.append(self.vis_cond(**data));
        returning {} keeps the snapshot dict free of cond-visualization media.
        """
        return {}

    @torch.no_grad()
    def run_snapshot(self, num_samples, batch_size=1, verbose=False):
        """Decoder-based snapshot: sample -> decode -> Gaussian render -> wandb media.

        Pipeline:
          val_dataset (or train fallback) -> collate -> noise (same coords)
          -> sampler.sample(50 steps, cfg=3.0) -> z_slat (normalized)
          -> un_normalize_slat(z, mean, std)
          -> frozen SLatGaussianDecoder -> List[Gaussian]
          -> GaussianRenderer x 4 canonical views
          -> {'sample_rendered': {'value': [N,3,H,W], 'type': 'image'}}

        Decoder is loaded at start, deleted at end (~2GB VRAM) — same pattern
        as SLatVisMixin._loading_slat_dec / _delete_slat_dec.

        BasicTrainer.run() calls snapshot('init') before training; this method
        runs even in smoke mode. Graceful skip if decoder weights missing.
        """
        import copy
        from torch.utils.data import DataLoader
        from trellis.utils.arts.slat_render_utils import (
            load_slat_decoder,
            load_gaussian_renderer,
            get_canonical_cameras,
            un_normalize_slat,
            render_sample_to_views,
        )
        if not self.is_master:
            return {}

        # --- 0. Graceful skip if decoder weights are missing (smoke tests) ---
        _dec_ckpt = 'pretrained/ckpts/slat_dec_gs_swin8_B_64l8gs32_fp16'
        if not (os.path.exists(f'{_dec_ckpt}.json') and os.path.exists(f'{_dec_ckpt}.safetensors')):
            print(f'[Stage4Trainer.run_snapshot] decoder weights not found at {_dec_ckpt}, skipping snapshot')
            return {}

        # --- 1. Load decoder + renderer (ephemeral) ---
        decoder = load_slat_decoder()
        renderer = load_gaussian_renderer()
        extrinsics, intrinsics = get_canonical_cameras(num_views=4)

        # --- 2. Normalization tensors ---
        ds = getattr(self, 'val_dataset', None) or self.dataset
        if hasattr(ds, 'mean') and ds.mean is not None:
            mean_tensor = ds.mean.cuda().float()
            std_tensor = ds.std.cuda().float()
        else:
            from trellis.datasets.arts.slat_flow_art import DEFAULT_MEAN, DEFAULT_STD
            mean_tensor = torch.tensor(DEFAULT_MEAN, dtype=torch.float32).cuda()
            std_tensor = torch.tensor(DEFAULT_STD, dtype=torch.float32).cuda()

        # --- 3. Val dataset sampling ---
        val_ds = getattr(self, 'val_dataset', None)
        snapshot_dataset = val_ds if (val_ds is not None and len(val_ds) > 0) else self.dataset
        if len(snapshot_dataset) == 0:
            print('[Stage4Trainer.run_snapshot] snapshot_dataset is empty (no slat_latents_expanded?), skipping snapshot')
            return {}
        dl = DataLoader(
            copy.deepcopy(snapshot_dataset),
            batch_size=1,  # 固定 1：collate_fn packs SparseTensors; render is per-sample
            shuffle=True,
            num_workers=0,
            collate_fn=type(snapshot_dataset).collate_fn if hasattr(type(snapshot_dataset), 'collate_fn') else None,
        )

        # --- 4. Sample + Decode + Render ---
        sampler = self.get_sampler()
        rendered_views = []
        snapshot_dir = os.path.join(self.output_dir, 'snapshots', f'step_{self.step:07d}')
        os.makedirs(snapshot_dir, exist_ok=True)

        dl_iter = iter(dl)
        n_produced = 0
        for i in range(min(num_samples, len(snapshot_dataset))):
            try:
                data = next(dl_iter)
                # SparseTensor 不是 torch.Tensor 子类，需要单独处理 .cuda()
                for k, v in data.items():
                    if isinstance(v, torch.Tensor):
                        data[k] = v.cuda()
                    elif hasattr(v, 'cuda'):
                        data[k] = v.cuda()

                x_0 = data['x_0']
                cond = data['cond']
                noise = x_0.replace(torch.randn_like(x_0.feats))

                neg_cond = torch.zeros_like(cond)
                res = sampler.sample(
                    self.models['denoiser'],
                    noise=noise,
                    cond=cond,
                    neg_cond=neg_cond,
                    steps=50,
                    cfg_strength=3.0,
                    verbose=verbose,
                )
                z_slat = res.samples

                # Un-normalize before decoder (CRITICAL)
                z_slat_raw = un_normalize_slat(z_slat, mean_tensor, std_tensor)

                gaussians = decoder(z_slat_raw)
                gs = gaussians[0]
                views = render_sample_to_views(gs, extrinsics, intrinsics, renderer)
                rendered_views.append(views)
                n_produced += 1

                from torchvision.utils import save_image
                for v_idx in range(views.shape[0]):
                    save_image(
                        views[v_idx],
                        os.path.join(snapshot_dir, f'sample_{i:03d}_view{v_idx}.png'),
                    )
            except StopIteration:
                print(f'[Stage4Trainer.run_snapshot] DataLoader exhausted after {n_produced} samples')
                break
            except Exception as e:
                print(f'[Stage4Trainer.run_snapshot] sample {i} failed: {e}')
                continue

        # --- 5. Cleanup decoder to free VRAM ---
        del decoder
        del renderer
        torch.cuda.empty_cache()

        if rendered_views:
            stacked = torch.cat(rendered_views, dim=0)
        else:
            stacked = torch.zeros(1, 3, 512, 512, device='cuda')

        print(f'[Stage4Trainer.run_snapshot] step {self.step}: rendered {len(rendered_views)} samples')
        return {
            'sample_rendered': {'value': stacked, 'type': 'image'},
        }


# ---- Helpers ---------------------------------------------------------------

def ensure_manifest_split(cfg, rank: int = 0) -> None:
    """Auto-split manifest.json into manifest_train.json / manifest_val.json.

    Distributed-safe: only rank 0 performs IO; all ranks barrier before returning.
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
            print(f'[AUTO-SPLIT] cannot split: source manifest not found: {src_path}')
        else:
            print(f'[AUTO-SPLIT] manifest_train.json / manifest_val.json missing')
            print(f'[AUTO-SPLIT] splitting from: {src_path}')

            with open(src_path, 'r') as f:
                data = json.load(f)

            if not isinstance(data, dict) or 'samples' not in data:
                print('[AUTO-SPLIT] ERROR: unexpected format (expected {"samples": [...]}), skipping')
            else:
                samples = data['samples']

                by_obj: dict = {}
                for s in samples:
                    by_obj.setdefault(str(s['object_id']), []).append(s)
                obj_ids = sorted(by_obj.keys())

                train_samples, val_samples = [], []
                train_obj_ids, val_obj_ids = [], []
                for obj_id in tqdm(obj_ids, desc='[AUTO-SPLIT] splitting manifest', unit='obj'):
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
                print(f'[AUTO-SPLIT] wrote {train_path}')
                print(f'[AUTO-SPLIT] wrote {val_path}')

    if dist.is_initialized():
        dist.barrier()


def setup_rng(seed: int = 42):
    """Set deterministic seeds for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def _ensure_elastic_wired(model, memory_controller, rank):
    """LoRA × Elastic isinstance interaction mitigation.

    After peft.get_peft_model wraps ElasticSLatFlowModel in PeftModel,
    BasicTrainer.init_models_and_more's `isinstance(model, ElasticModuleMixin)`
    check returns False, so elastic controller is never registered to the base
    model. Without this fix, LoRA mode OOMs on 4090 where full mode does not.

    Always prints `[Stage4 Train] elastic_wired_check: wired=<bool>` for the
    smoke harness to grep one canonical pattern regardless of which branch ran.
    """
    unwrapped = model.get_base_model() if hasattr(model, 'get_base_model') else model

    if not isinstance(unwrapped, ElasticSLatFlowModel):
        if rank == 0:
            print(f'[Stage4 Train] elastic_wired_check: wired=False '
                  f'reason=not-elastic-class type={type(unwrapped).__name__}',
                  flush=True)
        return False

    if getattr(unwrapped, '_memory_controller', None) is not None:
        if rank == 0:
            print('[Stage4 Train] elastic_wired_check: wired=True '
                  'reason=already-registered', flush=True)
        return True

    if memory_controller is None:
        if rank == 0:
            print('[Stage4 Train] elastic_wired_check: wired=False '
                  'reason=no-controller-from-trainer', flush=True)
        return False

    if hasattr(unwrapped, 'register_memory_controller'):
        unwrapped.register_memory_controller(memory_controller)
        if rank == 0:
            print('[Stage4 Train] elastic_wired_check: wired=True '
                  'reason=manual-register', flush=True)
        return True

    if rank == 0:
        print('[Stage4 Train] elastic_wired_check: wired=False '
              'reason=no-register-method', flush=True)
    return False


def _print_parameter_freeze_summary(model, rank: int, label: str) -> None:
    if rank != 0:
        return
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = total - trainable
    print(
        f'[Stage4 Train] param_freeze_summary[{label}]: '
        f'trainable={trainable:,} frozen={frozen:,} total={total:,}',
        flush=True,
    )


# ---- Stage entry-point (D-12 dispatch contract) ----------------------------

def train(config, *, load_dir: str = None, resume_step: int = None,
          dump_param_stats: bool = False) -> None:
    """Stage entry-point invoked by TRELLIS-arts/train_arts.py.

    Args:
        config: OmegaConf DictConfig already loaded from YAML.
        load_dir: optional checkpoint directory for resume.
        resume_step: optional step number for resume.
        dump_param_stats: print param hashes before/after training (LoRA verify).
    """
    cfg = config

    # ---- 1. Distributed init ----
    rank, local_rank, world_size = setup_ddp()
    is_distributed = world_size > 1
    setup_rng(seed=42 + rank)

    if rank == 0:
        print('\n[Stage4 Train] config loaded:')
        print(f'  distributed: {is_distributed} | world_size={world_size}')

    # ---- 2.5 Auto-split manifest if needed ----
    ensure_manifest_split(cfg, rank=rank)

    # ---- 3. Build Dataset ----
    data_cfg = config_to_dict(cfg.data)
    dataset = MvImageConditionedSparseLatentDataset(data_cfg)

    # ---- 4. Build Model ----
    model_cfg = config_to_dict(cfg.model)
    model_cfg.pop('name', None)
    model_args = model_cfg.pop('args', model_cfg)
    model = ElasticSLatFlowModel(**model_args).cuda()

    if rank == 0:
        num_params = sum(p.numel() for p in model.parameters())
        num_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f'\n[Stage4 Train] model: ElasticSLatFlowModel')
        print(f'  total params: {num_params:,}')
        print(f'  trainable params: {num_trainable:,}')
    _print_parameter_freeze_summary(model, rank, 'initial')

    # ---- 4.5 Extract LoRA config ----
    lora_cfg = config_to_dict(cfg.lora) if 'lora' in cfg else {}
    lora_enabled = lora_cfg.get('enabled', False)

    # ---- 5. Training config ----
    training_cfg = config_to_dict(cfg.training)
    output_dir = training_cfg.pop('output_dir', 'output/slat_flow_art_default')
    pretrained_ckpt_original = training_cfg.get('pretrained_ckpt', None)
    pretrained_ckpt = training_cfg.pop('pretrained_ckpt', None)

    if rank == 0:
        os.makedirs(output_dir, exist_ok=True)

    # ---- 5.5 Resume detection ----
    is_resuming = load_dir is not None and resume_step is not None
    if is_resuming and pretrained_ckpt is not None:
        if rank == 0:
            print('[Stage4 Train] Resume mode, skipping pretrained_ckpt (avoid overwriting checkpoint weights)')
        pretrained_ckpt = None

    # ---- 5.6 Pretrained weight load ----
    pretrained_loaded = False
    if pretrained_ckpt is not None:
        ckpt_path = os.path.join(PROJECT_ROOT, pretrained_ckpt) \
            if not os.path.isabs(pretrained_ckpt) else pretrained_ckpt
        if os.path.exists(ckpt_path):
            if ckpt_path.endswith('.safetensors'):
                try:
                    from safetensors.torch import load_file
                    state_dict = load_file(ckpt_path)
                    missing, unexpected = model.load_state_dict(state_dict, strict=False)
                    if missing or unexpected:
                        raise RuntimeError(
                            f'[Stage4 Train] pretrained SLat flow checkpoint incompatible: '
                            f'missing={missing[:20]} unexpected={unexpected[:20]} '
                            f'(counts: missing={len(missing)} unexpected={len(unexpected)})'
                        )
                    pretrained_loaded = True
                    if rank == 0:
                        print(f'[Stage4 Train] loaded pretrained from safetensors: {ckpt_path}')
                        print(f'  missing keys: {len(missing)}')
                        print(f'  unexpected keys: {len(unexpected)}')
                except ImportError:
                    raise ImportError('[Stage4 Train] safetensors is required to load pretrained SLat flow weights')
            else:
                ckpt_data = torch.load(ckpt_path, map_location='cuda', weights_only=True)
                missing, unexpected = model.load_state_dict(ckpt_data, strict=False)
                if missing or unexpected:
                    raise RuntimeError(
                        f'[Stage4 Train] pretrained SLat flow checkpoint incompatible: '
                        f'missing={missing[:20]} unexpected={unexpected[:20]} '
                        f'(counts: missing={len(missing)} unexpected={len(unexpected)})'
                    )
                pretrained_loaded = True
                if rank == 0:
                    print(f'[Stage4 Train] loaded pretrained from pt: {ckpt_path}')
                    print(f'  missing keys: {len(missing)}')
                    print(f'  unexpected keys: {len(unexpected)}')
        else:
            raise FileNotFoundError(f'[Stage4 Train] pretrained weights not found: {ckpt_path}')

    # ---- 5.7 Apply LoRA ----
    if lora_enabled:
        model = apply_lora_to_model(model, lora_cfg)
        if rank == 0:
            trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
            total = sum(p.numel() for p in model.parameters())
            print(f'\n[Stage4 Train] LoRA enabled:')
            print(f'  rank={lora_cfg.get("rank", 16)}, target={lora_cfg.get("target_modules", "all_attn")}')
            print(f'  trainable params: {trainable:,} / {total:,} ({trainable/total*100:.2f}%)')
    _print_parameter_freeze_summary(model, rank, 'after_lora')

    # ---- 5.8 L2-SP Anchor ----
    anchor_cfg = config_to_dict(cfg.anchor) if 'anchor' in cfg else {}
    anchor_enabled = anchor_cfg.get('enabled', False)
    anchor = None
    if anchor_enabled:
        if pretrained_ckpt_original is None:
            if rank == 0:
                print('[Anchor] WARN: anchor.enabled=True but pretrained_ckpt not set, skipping')
        else:
            anchor_ckpt_path = (
                os.path.join(PROJECT_ROOT, pretrained_ckpt_original)
                if not os.path.isabs(pretrained_ckpt_original)
                else pretrained_ckpt_original
            )
            if not os.path.exists(anchor_ckpt_path):
                if rank == 0:
                    print(f'[Anchor] WARN: anchor source not found: {anchor_ckpt_path}, skipping')
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

    trainer = Stage4Trainer(
        models={'denoiser': model},
        dataset=dataset,
        output_dir=output_dir,
        load_dir=load_dir,
        step=resume_step,
        wandb_config=wandb_config,
        **training_cfg,
    )

    # ---- 6.5 Elastic + LoRA compatibility fix ----
    _elastic_wired = _ensure_elastic_wired(
        model,
        getattr(trainer, 'elastic_controller', None),
        rank,
    )
    if lora_enabled and not _elastic_wired and rank == 0:
        print('[WARN] LoRA mode: elastic wiring check returned wired=False -- '
              'LoRA will likely OOM on 4090. See _ensure_elastic_wired output above.')

    # ---- 6.6 Force SparseTensor backend init in main process ----
    _c = torch.zeros((1, 4), dtype=torch.int32)
    _f = torch.zeros((1, 8), dtype=torch.float32)
    from trellis.modules.sparse import SparseTensor as _SparseTensor
    _SparseTensor(coords=_c, feats=_f)
    del _SparseTensor, _c, _f

    # ---- 7. Optional: param stats before training ----
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
        print(f'\n[PARAM_STATS_BEFORE] total={total} trainable={trainable} '
              f'ratio={trainable/total*100:.4f}%')

    # ---- 8. Train ----
    if rank == 0:
        print('\n[Stage4 Train] starting training...')
    trainer.run()

    # M-5 fix: unconditional peak_mem_mb print
    if rank == 0:
        peak_mem = torch.cuda.max_memory_allocated() / 1024**2
        print(f'[Stage4 Train] peak_mem_mb={peak_mem:.1f}', flush=True)

    # ---- 9. Optional: param stats after training ----
    if dump_param_stats and rank == 0 and param_snapshot_before is not None:
        changed, unchanged = [], []
        for name, param in model.named_parameters():
            h = hashlib.md5(param.data.cpu().numpy().tobytes()).hexdigest()
            if h != param_snapshot_before[name]['hash']:
                changed.append(name)
            else:
                unchanged.append(name)
        keep_patterns = lora_cfg.get('keep_trainable', []) or []
        lora_changed = [n for n in changed if 'lora_' in n]
        hybrid_changed = [
            n for n in changed
            if 'lora_' not in n and keep_patterns and any(pat in n for pat in keep_patterns)
        ]
        hybrid_changed_set = set(hybrid_changed)
        non_lora_changed = [
            n for n in changed
            if 'lora_' not in n and n not in hybrid_changed_set
        ]
        print(f'\n[PARAM_STATS_AFTER] changed={len(changed)} unchanged={len(unchanged)}')
        print(f'[PARAM_STATS_AFTER] lora_changed={len(lora_changed)} '
              f'hybrid_changed={len(hybrid_changed)} non_lora_changed={len(non_lora_changed)}')
        if hybrid_changed:
            print(f'[PARAM_STATS_AFTER] OK (hybrid): keep_trainable layers changed as expected: '
                  f'{hybrid_changed[:5]}')
        if non_lora_changed:
            print(f'[PARAM_STATS_AFTER] WARNING: non-LoRA & non-keep_trainable params changed: '
                  f'{non_lora_changed[:5]}')
        else:
            print(f'[PARAM_STATS_AFTER] OK: all non-LoRA & non-keep_trainable params frozen')


# ---- CLI fallback ----------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description='Stage 4 (slat_flow_art) multi-view conditioned SLat Flow Matching training'
    )
    parser.add_argument('--config', type=str, required=True,
                        help='YAML config file path (supports _base_ inheritance)')
    parser.add_argument('--load-dir', type=str, default=None,
                        help='Load checkpoint from this directory for resume')
    parser.add_argument('--resume-step', type=int, default=None,
                        help='Resume step number (requires --load-dir)')
    parser.add_argument('--dump-param-stats', action='store_true', default=False,
                        help='Print param stats before/after training (for LoRA freeze verification)')
    parser.add_argument('overrides', nargs='*', default=[],
                        help='OmegaConf overrides, format: key=value')
    return parser.parse_args()


def main():
    """CLI fallback for single-stage debugging."""
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
