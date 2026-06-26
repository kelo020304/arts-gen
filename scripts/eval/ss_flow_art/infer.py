#!/usr/bin/env python3
"""Stage 2 (SS Flow) 推理脚本：采样 SS latent → Voxel IoU + 体素可视化。

不依赖 YAML config。模型架构使用硬编码默认值（与标准预训练 L 模型对齐）。

使用方式:
    # 单样本：推理 + 可视化
    python scripts/eval/stage2/infer.py \\
        --ckpt pretrained/ckpts/ss_flow_img_dit_L_16l8_fp16.safetensors \\
        --data_root data/smoke_test \\
        --obj_id 100015 --angle_idx 0 \\
        --output output/eval_stage2/

    # 批量：遍历所有样本，输出 metrics.json
    python scripts/eval/stage2/infer.py \\
        --ckpt pretrained/ckpts/ss_flow_img_dit_L_16l8_fp16.safetensors \\
        --data_root data/smoke_test \\
        --manifest data/smoke_test/reconstruction/manifest.json \\
        --output output/eval_stage2/

数据路径约定:
    {data_root}/reconstruction/ss_latents_expanded/{obj_id}/angle_{N}/latent.npz  key='mean' [8,16,16,16]
    {data_root}/reconstruction/dinov2_tokens/{obj_id}/angle_{N}/tokens.npz        key='tokens' [V,1370,1024]
"""

import argparse
import json
import os
import sys
from typing import List, Optional, Tuple

import numpy as np
import torch

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

from trellis.models.sparse_structure_flow import SparseStructureFlowModel   # noqa: E402
from trellis.pipelines.samplers.flow_euler import FlowEulerSampler           # noqa: E402

# ---------------------------------------------------------------------------
# 硬编码默认值（与 ss_flow_img_dit_L_16l8_fp16 对齐）
# ---------------------------------------------------------------------------

DEFAULT_MODEL_ARGS = dict(
    resolution=16,
    in_channels=8,
    model_channels=1024,
    cond_channels=1024,
    out_channels=8,
    num_blocks=24,
    num_heads=16,
    num_head_channels=64,
    mlp_ratio=4,
    patch_size=1,
    pe_mode='ape',
    use_fp16=True,
    use_checkpoint=False,
    share_mod=False,
    qk_rms_norm=False,
    qk_rms_norm_cross=False,
)

RECON_SUBDIR = 'reconstruction'
NUM_VIEWS = 4          # 取前 N 视角
NUM_STEPS = 25
IOU_THRESHOLD = 1e-3


# ---------------------------------------------------------------------------
# 模型加载
# ---------------------------------------------------------------------------

def build_model(ckpt_path: str) -> torch.nn.Module:
    model = SparseStructureFlowModel(**DEFAULT_MODEL_ARGS).cuda()
    n = sum(p.numel() for p in model.parameters())
    print(f'[INFO] SparseStructureFlowModel ({n:,} params)')

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
    return model


# ---------------------------------------------------------------------------
# 数据加载
# ---------------------------------------------------------------------------

def load_sample(
    recon_root: str, obj_id: str, angle_idx: int, num_views: int = NUM_VIEWS,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Load one sample. Returns (gt_latent [8,16,16,16], cond [1,V*T,1024])."""
    angle_dir = f'angle_{angle_idx}'

    # GT SS latent
    latent_path = os.path.join(recon_root, 'ss_latents_expanded', obj_id, angle_dir, 'latent.npz')
    if not os.path.isfile(latent_path):
        raise FileNotFoundError(latent_path)
    gt_latent = torch.from_numpy(np.load(latent_path)['mean']).float().cuda()  # [8,16,16,16]

    # DINOv2 tokens: take first num_views of V available, flatten
    tokens_path = os.path.join(recon_root, 'dinov2_tokens', obj_id, angle_dir, 'tokens.npz')
    if not os.path.isfile(tokens_path):
        raise FileNotFoundError(tokens_path)
    tokens = torch.from_numpy(np.load(tokens_path)['tokens']).float()  # [V_total, T, D]
    V_total, T, D = tokens.shape
    n = min(num_views, V_total)
    cond = tokens[:n].reshape(1, n * T, D).cuda()  # [1, V*T, 1024]

    return gt_latent, cond


def enumerate_samples(recon_root: str, manifest_path: Optional[str]) -> List[Tuple[str, int]]:
    """Return list of (obj_id, angle_idx) from manifest or directory scan."""
    if manifest_path and os.path.isfile(manifest_path):
        with open(manifest_path) as f:
            m = json.load(f)
        if isinstance(m, dict) and 'samples' in m:
            return [(str(s.get('object_id', s.get('id', ''))), s.get('angle_idx', 0))
                    for s in m['samples']]
        elif isinstance(m, list):
            samples = []
            for e in m:
                obj = str(e.get('id', e.get('object_id', '')))
                for a in e.get('angles', [0]):
                    samples.append((obj, a))
            return samples

    # Fallback: scan ss_latents_expanded/
    ss_root = os.path.join(recon_root, 'ss_latents_expanded')
    if not os.path.isdir(ss_root):
        return []
    samples = []
    for obj_id in sorted(os.listdir(ss_root)):
        obj_dir = os.path.join(ss_root, obj_id)
        if not os.path.isdir(obj_dir):
            continue
        for angle_dir in sorted(os.listdir(obj_dir)):
            if not angle_dir.startswith('angle_'):
                continue
            angle_idx = int(angle_dir.split('_')[1])
            latent_path = os.path.join(obj_dir, angle_dir, 'latent.npz')
            if os.path.isfile(latent_path):
                samples.append((obj_id, angle_idx))
    return samples


# ---------------------------------------------------------------------------
# 推理
# ---------------------------------------------------------------------------

@torch.no_grad()
def sample_one(
    model: torch.nn.Module,
    cond: torch.Tensor,
    sampler: FlowEulerSampler,
    latent_shape: tuple = (8, 16, 16, 16),
) -> torch.Tensor:
    """Sample one latent. Returns pred [8,16,16,16]."""
    noise = torch.randn(1, *latent_shape, device=cond.device, dtype=torch.float32)
    result = sampler.sample(model=model, noise=noise, cond=cond, steps=NUM_STEPS, verbose=False)
    return result.samples[0]  # [8,16,16,16]


def compute_voxel_iou(
    pred: torch.Tensor, gt: torch.Tensor, threshold: float = IOU_THRESHOLD,
) -> float:
    """Voxel IoU on [C,H,W,D] latent tensors (occupancy = channel abs sum > threshold)."""
    pred_occ = (pred.abs().sum(0) > threshold)
    gt_occ = (gt.abs().sum(0) > threshold)
    inter = (pred_occ & gt_occ).sum().item()
    union = (pred_occ | gt_occ).sum().item()
    return float(inter) / float(union) if union > 0 else 0.0


# ---------------------------------------------------------------------------
# 可视化
# ---------------------------------------------------------------------------

def visualize_occupancy(
    pred: torch.Tensor,
    gt: torch.Tensor,
    output_path: str,
    title: str = '',
    threshold: float = IOU_THRESHOLD,
) -> None:
    """Save side-by-side 3D voxel plot: pred (left) vs GT (right)."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    def occ_to_np(t: torch.Tensor) -> np.ndarray:
        return (t.cpu().abs().sum(0) > threshold).numpy()

    pred_occ = occ_to_np(pred)   # [16,16,16]
    gt_occ = occ_to_np(gt)

    fig = plt.figure(figsize=(14, 6))
    fig.suptitle(title, fontsize=12)

    for col, (occ, label) in enumerate([(pred_occ, 'Pred'), (gt_occ, 'GT')]):
        ax = fig.add_subplot(1, 2, col + 1, projection='3d')
        ax.voxels(occ, facecolors='#4363d8', edgecolors='#222222', linewidth=0.2, alpha=0.8)
        ax.set_title(label, fontsize=11)
        ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
        ax.view_init(elev=25, azim=135)
        ticks = range(0, 17, 4)
        ax.set_xticks(ticks); ax.set_yticks(ticks); ax.set_zticks(ticks)
        ax.tick_params(labelsize=7)

    os.makedirs(os.path.dirname(os.path.abspath(output_path)) or '.', exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description='Stage 2 SS Flow 推理 + Voxel IoU')
    p.add_argument('--ckpt', required=True, help='.safetensors 或 .pt checkpoint 路径')
    p.add_argument('--data_root', required=True, help='数据根目录（含 reconstruction/ 子目录）')
    p.add_argument('--recon_subdir', default=RECON_SUBDIR, help='重建数据子目录名（默认 reconstruction）')
    p.add_argument('--obj_id', default=None, help='单样本模式：物体 ID')
    p.add_argument('--angle_idx', type=int, default=0, help='单样本模式：关节角度索引')
    p.add_argument('--manifest', default=None, help='批量模式：manifest JSON 路径')
    p.add_argument('--views', type=int, default=NUM_VIEWS, help='条件视角数（默认 4）')
    p.add_argument('--output', required=True, help='输出目录')
    p.add_argument('--limit', type=int, default=-1, help='批量模式：最多处理 N 个样本（-1=全部）')
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
    sampler = FlowEulerSampler(sigma_min=1e-5)

    # Enumerate samples
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
            gt_latent, cond = load_sample(recon_root, obj_id, angle_idx, args.views)
        except FileNotFoundError as e:
            print(f'[WARN] missing data: {e}')
            continue

        pred = sample_one(model, cond, sampler)
        iou = compute_voxel_iou(pred, gt_latent)
        results.append({'obj_id': obj_id, 'angle_idx': angle_idx, 'voxel_iou': iou})
        print(f'[INFO] {obj_id}/angle_{angle_idx}: voxel_iou={iou:.4f}')

        if single_mode:
            viz_path = os.path.join(args.output, f'viz_{obj_id}_{angle_idx}.png')
            visualize_occupancy(
                pred, gt_latent, viz_path,
                title=f'{obj_id}/angle_{angle_idx}  IoU={iou:.4f}',
            )
            print(f'[INFO] viz → {viz_path}')

            pred_path = os.path.join(args.output, f'pred_{obj_id}_{angle_idx}.npz')
            np.savez_compressed(pred_path, pred=pred.cpu().numpy())
            print(f'[INFO] pred → {pred_path}')

    mean_iou = float(np.mean([r['voxel_iou'] for r in results])) if results else 0.0
    out = {
        'ckpt': args.ckpt, 'views': args.views,
        'n_samples': len(results), 'mean_voxel_iou': mean_iou,
        'samples': results,
    }
    metrics_path = os.path.join(args.output, 'metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(out, f, indent=2)
    print(f'[DONE] mean_iou={mean_iou:.4f}  n={len(results)} → {metrics_path}')


if __name__ == '__main__':
    main()
