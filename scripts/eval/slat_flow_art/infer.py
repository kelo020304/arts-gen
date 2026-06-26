#!/usr/bin/env python3
"""Stage 4 (SLat Flow) 推理脚本：采样 SLat → decode → 渲染 → PSNR/SSIM + 渲染图。

不依赖 YAML config。模型架构使用硬编码默认值（与标准预训练 L 模型对齐）。

使用方式:
    # 单样本：推理 + 渲染可视化
    python scripts/eval/stage4/infer.py \\
        --ckpt pretrained/ckpts/slat_flow_img_dit_L_64l8p2_fp16.safetensors \\
        --decoder pretrained/ckpts/slat_dec_gs_swin8_B_64l8gs32_fp16 \\
        --data_root data/smoke_test \\
        --obj_id 100015 --angle_idx 0 \\
        --output output/eval_stage4/

    # 批量：遍历样本，输出 metrics.json
    python scripts/eval/stage4/infer.py \\
        --ckpt pretrained/ckpts/slat_flow_img_dit_L_64l8p2_fp16.safetensors \\
        --decoder pretrained/ckpts/slat_dec_gs_swin8_B_64l8gs32_fp16 \\
        --data_root data/smoke_test \\
        --manifest data/smoke_test/reconstruction/manifest.json \\
        --output output/eval_stage4/

数据路径约定:
    {data_root}/reconstruction/slat_latents_expanded/{obj_id}/angle_{N}/latent.npz  coords+feats
    {data_root}/reconstruction/dinov2_tokens/{obj_id}/angle_{N}/tokens.npz          tokens [V,T,D]
    {data_root}/reconstruction/renders/{obj_id}/angle_{N}/rgb/          (可选, GT 渲染对比用)
"""

import argparse
import json
import math
import os
import sys
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# 路径 + trellis 包注册（D-35：09-01~09-04 已把 trellis.* 子包搬入 TRELLIS-arts/）
# ---------------------------------------------------------------------------

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
TRELLIS_PATH = os.path.join(PROJECT_ROOT, 'TRELLIS-arts')

# eval 脚本是 standalone CLI（不在 trellis 包内），需把 TRELLIS-arts/ 加入 sys.path
# 让 `from trellis.X import Y` 能解析；不再用 types.ModuleType 黑魔法注册。
if TRELLIS_PATH not in sys.path:
    sys.path.insert(0, TRELLIS_PATH)

os.environ.setdefault('TORCH_HOME', os.path.join(PROJECT_ROOT, 'submodules', 'TRELLIS.1'))

# 所有 eval 脚本统一要求 trellis 环境（flash_attn 或 xformers）
import importlib as _importlib
_cur = os.environ.get('ATTN_BACKEND')
if _cur in (None, '', 'sdpa'):
    if _importlib.util.find_spec('flash_attn') is not None:
        os.environ['ATTN_BACKEND'] = 'flash_attn'
    elif _importlib.util.find_spec('xformers') is not None:
        os.environ['ATTN_BACKEND'] = 'xformers'
    else:
        raise RuntimeError(
            '所有 eval 脚本必须在 trellis conda 环境下运行 '
            '(需要 flash_attn 或 xformers)。请切换到 trellis 环境。'
        )

from trellis.models.structured_latent_flow import ElasticSLatFlowModel         # noqa: E402
from trellis.datasets.structured_latent import SLat                             # noqa: E402
from trellis.pipelines.samplers import FlowEulerCfgSampler                      # noqa: E402
from trellis.utils.arts.slat_render_utils import (                              # noqa: E402
    load_slat_decoder, load_gaussian_renderer, get_canonical_cameras,
    un_normalize_slat, render_sample_to_views,
)

# ---------------------------------------------------------------------------
# 硬编码默认值（与 slat_flow_img_dit_L_64l8p2_fp16 对齐）
# ---------------------------------------------------------------------------

DEFAULT_MODEL_ARGS = dict(
    resolution=64,
    in_channels=8,
    model_channels=1024,
    cond_channels=1024,
    out_channels=8,
    num_blocks=24,
    num_heads=16,
    num_head_channels=64,
    mlp_ratio=4,
    patch_size=2,
    num_io_res_blocks=2,
    io_block_channels=[128],
    pe_mode='ape',
    use_fp16=True,
    use_checkpoint=False,
    qk_rms_norm=True,
    qk_rms_norm_cross=False,
)

# SLat feats normalization constants (from configs/stage4/mv_4view.yaml)
DEFAULT_MEAN = torch.tensor([
    -2.1687545776367188, -0.004347046371549368, -0.13352349400520325,
    -0.08418072760105133, -0.5271206498146057,   0.7238689064979553,
    -1.1414450407028198,   1.2039363384246826,
], dtype=torch.float32)

DEFAULT_STD = torch.tensor([
     2.377650737762451,   2.386378288269043,  2.124418020248413,
     2.1748552322387695,  2.663944721221924,  2.371192216873169,
     2.6217446327209473,  2.684523105621338,
], dtype=torch.float32)

RECON_SUBDIR = 'reconstruction'
NUM_VIEWS = 4
NUM_STEPS = 25
CFG_STRENGTH = 3.0


# ---------------------------------------------------------------------------
# 模型加载
# ---------------------------------------------------------------------------

def build_model(ckpt_path: str) -> torch.nn.Module:
    model = ElasticSLatFlowModel(**DEFAULT_MODEL_ARGS).cuda()
    n = sum(p.numel() for p in model.parameters())
    print(f'[INFO] ElasticSLatFlowModel ({n:,} params)')

    ckpt_abs = ckpt_path if os.path.isabs(ckpt_path) else os.path.join(PROJECT_ROOT, ckpt_path)
    if not os.path.isfile(ckpt_abs):
        print(f'[ERROR] ckpt not found: {ckpt_abs}')
        sys.exit(1)

    if ckpt_abs.endswith('.safetensors'):
        from safetensors.torch import load_file
        sd = load_file(ckpt_abs)
    else:
        sd = torch.load(ckpt_abs, map_location='cuda', weights_only=True)

    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f'[INFO] loaded: missing={len(missing)} unexpected={len(unexpected)}')
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


# ---------------------------------------------------------------------------
# 数据加载
# ---------------------------------------------------------------------------

def load_sample(
    recon_root: str, obj_id: str, angle_idx: int, num_views: int = NUM_VIEWS,
) -> dict:
    """Load one SLat sample. Returns dict with keys: coords, feats (normalized), cond."""
    angle_dir = f'angle_{angle_idx}'

    # SLat latent (raw feats before normalization)
    latent_path = os.path.join(
        recon_root, 'slat_latents_expanded', obj_id, angle_dir, 'latent.npz',
    )
    if not os.path.isfile(latent_path):
        raise FileNotFoundError(latent_path)
    d = np.load(latent_path)
    coords = torch.from_numpy(d['coords']).int()   # [N, 3]
    feats_raw = torch.from_numpy(d['feats']).float()  # [N, 8]

    # Normalize feats: (feats - mean) / std
    mean = DEFAULT_MEAN.view(1, 8)
    std = DEFAULT_STD.view(1, 8)
    feats_norm = (feats_raw - mean) / std  # [N, 8]

    # DINOv2 tokens
    tokens_path = os.path.join(
        recon_root, 'dinov2_tokens', obj_id, angle_dir, 'tokens.npz',
    )
    if not os.path.isfile(tokens_path):
        raise FileNotFoundError(tokens_path)
    tokens = torch.from_numpy(np.load(tokens_path)['tokens']).float()  # [V, T, D]
    V_total, T, D = tokens.shape
    n = min(num_views, V_total)
    cond = tokens[:n].reshape(n * T, D)  # [V*T, D]

    return {'coords': coords, 'feats': feats_norm, 'cond': cond}


def load_gt_renders(
    recon_root: str, obj_id: str, angle_idx: int, num_views: int = NUM_VIEWS,
) -> Optional[torch.Tensor]:
    """Load GT Blender renders. Returns [V,3,H,W] in [0,1] or None."""
    from PIL import Image
    rgb_dir = os.path.join(recon_root, 'renders', obj_id, f'angle_{angle_idx}', 'rgb')
    if not os.path.isdir(rgb_dir):
        return None
    pngs = sorted(f for f in os.listdir(rgb_dir) if f.endswith('.png'))
    if len(pngs) < num_views:
        return None
    views = []
    for f in pngs[:num_views]:
        arr = np.asarray(Image.open(os.path.join(rgb_dir, f)).convert('RGB'), dtype=np.float32) / 255.0
        views.append(torch.from_numpy(arr).permute(2, 0, 1))
    return torch.stack(views).cuda()  # [V, 3, H, W]


def enumerate_samples(recon_root: str, manifest_path: Optional[str]) -> List[Tuple[str, int]]:
    if manifest_path and os.path.isfile(manifest_path):
        with open(manifest_path) as f:
            m = json.load(f)
        if isinstance(m, dict) and 'samples' in m:
            return [(str(s.get('object_id', s.get('id', ''))), s.get('angle_idx', 0))
                    for s in m['samples']]
        elif isinstance(m, list):
            return [(str(e.get('id', e.get('object_id', ''))), a)
                    for e in m for a in e.get('angles', [0])]

    slat_root = os.path.join(recon_root, 'slat_latents_expanded')
    if not os.path.isdir(slat_root):
        return []
    samples = []
    for obj_id in sorted(os.listdir(slat_root)):
        for angle_dir in sorted(os.listdir(os.path.join(slat_root, obj_id))):
            if not angle_dir.startswith('angle_'):
                continue
            angle_idx = int(angle_dir.split('_')[1])
            if os.path.isfile(os.path.join(slat_root, obj_id, angle_dir, 'latent.npz')):
                samples.append((obj_id, angle_idx))
    return samples


# ---------------------------------------------------------------------------
# 推理
# ---------------------------------------------------------------------------

@torch.no_grad()
def sample_one(
    model: torch.nn.Module,
    sample: dict,
    sampler: FlowEulerCfgSampler,
) -> 'SparseTensor':
    """Sample one SLat from noise, returns SparseTensor (normalized feats)."""
    packed = SLat.collate_fn([sample])
    x_0 = packed['x_0'].cuda()       # SparseTensor with GT coords
    cond = packed['cond'].cuda()      # [1, V*T, D]
    if cond.ndim == 2:
        cond = cond.unsqueeze(0)

    noise = x_0.replace(torch.randn_like(x_0.feats))
    neg_cond = torch.zeros_like(cond)

    result = sampler.sample(
        model, noise=noise, cond=cond, neg_cond=neg_cond,
        steps=NUM_STEPS, cfg_strength=CFG_STRENGTH, verbose=False,
    )
    return result.samples  # SparseTensor, normalized


def compute_psnr(pred: torch.Tensor, gt: torch.Tensor) -> float:
    pred = pred.clamp(0, 1)
    gt = gt.clamp(0, 1)
    try:
        from torchmetrics.image import PeakSignalNoiseRatio
        return float(PeakSignalNoiseRatio(data_range=1.0).to(pred.device)(pred, gt))
    except ImportError:
        mse = F.mse_loss(pred, gt).item()
        return -10.0 * math.log10(mse) if mse > 1e-10 else 99.0


def compute_ssim(pred: torch.Tensor, gt: torch.Tensor) -> float:
    pred = pred.clamp(0, 1)
    gt = gt.clamp(0, 1)
    try:
        from torchmetrics.image import StructuralSimilarityIndexMeasure
        return float(StructuralSimilarityIndexMeasure(data_range=1.0).to(pred.device)(pred, gt))
    except ImportError:
        return float('nan')


# ---------------------------------------------------------------------------
# 可视化（单样本模式）
# ---------------------------------------------------------------------------

def save_rendered_views(
    rendered: torch.Tensor,
    output_dir: str,
    obj_id: str,
    angle_idx: int,
    gt: Optional[torch.Tensor] = None,
) -> None:
    """Save rendered views as PNGs and optionally a side-by-side comparison."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from PIL import Image

    os.makedirs(output_dir, exist_ok=True)
    V = rendered.shape[0]

    n_cols = V
    n_rows = 2 if gt is not None else 1
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))
    if n_rows == 1:
        axes = [axes] if n_cols == 1 else list(axes)
        axes = [axes]

    for v in range(V):
        img = rendered[v].permute(1, 2, 0).cpu().clamp(0, 1).numpy()
        axes[0][v].imshow(img)
        axes[0][v].set_title(f'Pred view {v}', fontsize=9)
        axes[0][v].axis('off')
        if gt is not None:
            gt_img = gt[v].permute(1, 2, 0).cpu().clamp(0, 1).numpy()
            axes[1][v].imshow(gt_img)
            axes[1][v].set_title(f'GT view {v}', fontsize=9)
            axes[1][v].axis('off')

    fig.suptitle(f'{obj_id}/angle_{angle_idx}', fontsize=11)
    viz_path = os.path.join(output_dir, f'viz_{obj_id}_{angle_idx}.png')
    plt.savefig(viz_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'[INFO] viz → {viz_path}')


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description='Stage 4 SLat Flow 推理 + 渲染')
    p.add_argument('--ckpt', required=True, help='SLat Flow checkpoint (.safetensors 或 .pt)')
    p.add_argument('--decoder', default='pretrained/ckpts/slat_dec_gs_swin8_B_64l8gs32_fp16',
                   help='SLat Gaussian decoder basename（不含后缀）')
    p.add_argument('--data_root', required=True)
    p.add_argument('--recon_subdir', default=RECON_SUBDIR)
    p.add_argument('--obj_id', default=None)
    p.add_argument('--angle_idx', type=int, default=0)
    p.add_argument('--manifest', default=None)
    p.add_argument('--views', type=int, default=NUM_VIEWS)
    p.add_argument('--output', required=True)
    p.add_argument('--limit', type=int, default=-1)
    args = p.parse_args()
    if args.obj_id is None and args.manifest is None:
        p.error('必须提供 --obj_id 或 --manifest')
    if args.obj_id is not None and args.manifest is not None:
        p.error('--obj_id 和 --manifest 互斥')
    return args


def main() -> None:
    args = parse_args()
    recon_root = os.path.join(args.data_root, args.recon_subdir)
    os.makedirs(args.output, exist_ok=True)

    model = build_model(args.ckpt)
    mean_t = DEFAULT_MEAN.cuda()
    std_t = DEFAULT_STD.cuda()

    decoder_abs = args.decoder if os.path.isabs(args.decoder) else os.path.join(PROJECT_ROOT, args.decoder)
    decoder = load_slat_decoder(decoder_abs)
    renderer = load_gaussian_renderer()
    extrinsics, intrinsics = get_canonical_cameras(num_views=args.views)
    extrinsics = extrinsics.cuda()
    intrinsics = intrinsics.cuda()

    sampler = FlowEulerCfgSampler(sigma_min=1e-5)

    if args.obj_id is not None:
        samples = [(args.obj_id, args.angle_idx)]
        single_mode = True
    else:
        samples = enumerate_samples(recon_root, args.manifest)
        single_mode = False
        if args.limit > 0:
            samples = samples[:args.limit]
    print(f'[INFO] {len(samples)} sample(s) | views={args.views}')

    results = []
    for obj_id, angle_idx in samples:
        try:
            sample = load_sample(recon_root, obj_id, angle_idx, args.views)
        except FileNotFoundError as e:
            print(f'[WARN] missing data: {e}')
            continue

        z_slat_norm = sample_one(model, sample, sampler)
        z_slat_raw = un_normalize_slat(z_slat_norm, mean_t, std_t)
        gaussians = decoder(z_slat_raw)
        rendered = render_sample_to_views(gaussians[0], extrinsics, intrinsics, renderer)

        # Metrics vs GT
        gt = load_gt_renders(recon_root, obj_id, angle_idx, args.views)
        if gt is not None:
            if gt.shape != rendered.shape:
                gt = F.interpolate(gt, size=rendered.shape[-2:], mode='bilinear', align_corners=False)
            psnr = compute_psnr(rendered.unsqueeze(0), gt.unsqueeze(0))
            ssim = compute_ssim(rendered.unsqueeze(0), gt.unsqueeze(0))
        else:
            psnr, ssim = None, None
            if single_mode:
                print(f'[INFO] no GT renders found at {recon_root}/renders/{obj_id}/angle_{angle_idx}/rgb/')

        results.append({'obj_id': obj_id, 'angle_idx': angle_idx, 'psnr': psnr, 'ssim': ssim})
        psnr_str = f'{psnr:.2f}' if psnr is not None else 'N/A'
        print(f'[INFO] {obj_id}/angle_{angle_idx}: psnr={psnr_str}')

        if single_mode:
            save_rendered_views(rendered, args.output, obj_id, angle_idx, gt=gt)

    valid = [r for r in results if r['psnr'] is not None]
    mean_psnr = float(np.mean([r['psnr'] for r in valid])) if valid else None

    out = {
        'ckpt': args.ckpt, 'decoder': args.decoder, 'views': args.views,
        'n_samples': len(results), 'n_with_gt': len(valid),
        'mean_psnr': mean_psnr, 'samples': results,
    }
    metrics_path = os.path.join(args.output, 'metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(out, f, indent=2)
    psnr_str = f'{mean_psnr:.2f}' if mean_psnr is not None else 'N/A'
    print(f'[DONE] mean_psnr={psnr_str}  n={len(results)} → {metrics_path}')


if __name__ == '__main__':
    main()
