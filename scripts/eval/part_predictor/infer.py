#!/usr/bin/env python3
"""Part Predictor 推理脚本：前向推理 → Part mIoU + 着色体素可视化。

不依赖 YAML config。模型架构使用硬编码默认值。

使用方式:
    # 单样本：推理 + 可视化（先跑几步训练得到 checkpoint）
    python scripts/train/part_predictor/train.py \\
        --config scripts/train/configs/part_predictor/smoke_test.yaml \\
        training.checkpoint_every=5

    python scripts/eval/part_predictor/infer.py \\
        --ckpt output/part_predictor_smoke/ckpts/step_5.pt \\
        --data_root data/smoke_test \\
        --obj_id 100015 --angle_idx 0 \\
        --output output/eval_pp/

    # 批量：遍历样本，输出 metrics.json
    python scripts/eval/part_predictor/infer.py \\
        --ckpt output/part_predictor_smoke/ckpts/step_5.pt \\
        --data_root data/smoke_test \\
        --manifest data/smoke_test/reconstruction/manifest.json \\
        --output output/eval_pp/

数据路径约定:
    {data_root}/reconstruction/part_labels/{obj_id}/angle_{N}/part_labels_64.npy  [64,64,64] int64
    {data_root}/reconstruction/dinov2_tokens/{obj_id}/angle_{N}/tokens.npz        key='tokens' [V,T,D]
    {data_root}/reconstruction/part_info/{obj_id}/part_info.json
    (Part Predictor 使用 coords + DINOv2 only — 不依赖 SS/SLat latent)

Checkpoint 格式（train.py 保存的）:
    {'model': state_dict, 'optimizer': ..., 'step': N, 'loss': X}
"""

import argparse
import json
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

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

from trellis.models.part_predictor.part_predictor import QueryPartPredictor  # noqa: E402

# ---------------------------------------------------------------------------
# 硬编码默认值（与 base.yaml 对齐）
# ---------------------------------------------------------------------------

DEFAULT_MODEL_ARGS = dict(
    num_layers=12,
    query_dim=1024,
    num_heads=16,
    num_part_types=32,
    max_k=128,      # Must match base.yaml model.max_k
    cond_dim=1024,
    dropout=0.0,    # 推理时关闭 dropout（model.eval() 也会关闭）
    fusion_mode='serial',  # Must match base.yaml model.fusion_mode for ckpt compat
    serial_order=['voxel', 'rgb', 'mask'],  # Only used when fusion_mode=='serial'; must match base.yaml
)

# Part type vocabulary（与 dataset.py 同步）
PART_TYPE_VOCAB = {
    'base_body': 0, 'button': 1, 'lid': 2, 'door': 3, 'drawer': 4,
    'handle': 5, 'knob': 6, 'lever': 7, 'wheel': 8, 'slider': 9,
    'hinge': 10, 'rotation_part': 11, 'switch': 12, 'dial': 13,
    'pedal': 14, 'shelf': 15, 'mirror': 16, 'screen': 17, 'keyboard': 18,
    'leg': 19, 'arm': 20, 'seat': 21, 'back': 22, 'top': 23,
    'bottom': 24, 'side': 25, 'front': 26, 'cover': 27, 'panel': 28,
    'frame': 29, 'rod': 30,
}
PART_TYPE_OTHER = 31

RECON_SUBDIR = 'reconstruction'
MASK_SUBDIR = None   # e.g. 'arts/renders' — set via --mask_subdir CLI
NUM_VIEWS = 4
DINO_PATCH_GRID = 37  # DINOv2-L/14: 518/14=37


# ---------------------------------------------------------------------------
# 模型加载
# ---------------------------------------------------------------------------

def build_model(ckpt_path: str) -> torch.nn.Module:
    model = QueryPartPredictor(**DEFAULT_MODEL_ARGS).cuda()
    n = sum(p.numel() for p in model.parameters())
    print(f'[INFO] QueryPartPredictor ({n:,} params)')

    ckpt_abs = ckpt_path if os.path.isabs(ckpt_path) else os.path.join(PROJECT_ROOT, ckpt_path)
    if not os.path.isfile(ckpt_abs):
        print(f'[ERROR] ckpt not found: {ckpt_abs}')
        sys.exit(1)

    raw = torch.load(ckpt_abs, map_location='cuda', weights_only=False)
    # train.py saves {'model': state_dict, ...}; fallback: raw is state_dict
    sd = raw['model'] if isinstance(raw, dict) and 'model' in raw else raw
    step = raw.get('step', '?') if isinstance(raw, dict) else '?'

    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing:
        print(f'[WARN] loaded step={step}: missing keys={missing} (V1 ckpt not compatible with V2)')
    else:
        print(f'[INFO] loaded step={step}: missing=0 unexpected={len(unexpected)}')
    model.eval()
    return model


# ---------------------------------------------------------------------------
# 数据加载
# ---------------------------------------------------------------------------

def _type_str_to_idx(type_str: str) -> int:
    return PART_TYPE_VOCAB.get(type_str.lower().strip(), PART_TYPE_OTHER)


def _compute_mask_token_labels(
    V: int, T: int,
    mask_dir: str,
    label_to_idx: dict,
    view_indices: list,
    max_k: int = 128,
) -> torch.Tensor:
    """Compute per-token mask labels for inference views.

    Downsamples 2D masks to DINOv2 patch grid, remaps labels to 0-indexed
    (0=bg, 1..K = part idx + 1). CLS token position = 0.

    Returns:
        mask_token_labels: [V, T] int64
    """
    mask_token_labels = torch.zeros(V, T, dtype=torch.int64)

    if not os.path.isdir(mask_dir):
        return mask_token_labels

    for v in view_indices:
        mp = os.path.join(mask_dir, f'mask_{v}.npy')
        if not os.path.isfile(mp):
            continue
        mask_2d = np.load(mp)  # [H, W] int32
        mask_t = torch.from_numpy(mask_2d.astype(np.float32))
        patch_mask = F.interpolate(
            mask_t.unsqueeze(0).unsqueeze(0),
            size=(DINO_PATCH_GRID, DINO_PATCH_GRID), mode='nearest',
        ).squeeze().long()  # [37, 37]

        remapped = torch.zeros_like(patch_mask)
        for orig_label, new_idx in label_to_idx.items():
            remapped[patch_mask == orig_label] = new_idx + 1
        remapped = remapped.clamp(max=max_k)

        mask_token_labels[v, 1:] = remapped.flatten()  # skip CLS at position 0

    return mask_token_labels


def load_sample(
    recon_root: str, obj_id: str, angle_idx: int, num_views: int = NUM_VIEWS,
    mask_root: Optional[str] = None,
) -> Dict[str, Any]:
    """Load one sample for part predictor V2 inference.

    Returns dict with:
        coords [N, 3] int32, cond [V*T, D] float32,
        mask_token_labels [V*T] int64,
        part_labels [N] int64 (0-indexed), part_type_ids [K] int64, num_parts int,
        label_to_part_name dict
    """
    angle_dir = f'angle_{angle_idx}'

    # Part labels [64,64,64] — defines active voxel positions
    labels_path = os.path.join(recon_root, 'part_labels', obj_id, angle_dir, 'part_labels_64.npy')
    if not os.path.isfile(labels_path):
        raise FileNotFoundError(labels_path)
    part_labels_dense = torch.from_numpy(np.load(labels_path).astype(np.int64))

    # Active mask: exclude background (0) and overlap/ignore (-1)
    active_mask = (part_labels_dense != 0) & (part_labels_dense != -1)
    active_indices = torch.nonzero(active_mask, as_tuple=False)  # [N, 3]
    coords = active_indices.int()
    x, y, z = active_indices[:, 0], active_indices[:, 1], active_indices[:, 2]
    raw_labels = part_labels_dense[x, y, z]  # [N] values 1, 2, ...

    # Part info
    info_path = os.path.join(recon_root, 'part_info', obj_id, 'part_info.json')
    if not os.path.isfile(info_path):
        raise FileNotFoundError(info_path)
    with open(info_path) as f:
        part_info = json.load(f)

    num_parts = part_info['num_parts']
    sorted_parts = sorted(part_info['parts'].values(), key=lambda p: p['label'])

    label_to_idx = {}
    part_type_ids_list = []
    label_to_part_name = {}
    for new_idx, part in enumerate(sorted_parts):
        orig_label = part['label']
        label_to_idx[orig_label] = new_idx
        part_type_ids_list.append(_type_str_to_idx(part['type']))
        label_to_part_name[orig_label] = list(part_info['parts'].keys())[
            list(v['label'] for v in part_info['parts'].values()).index(orig_label)
        ]

    part_labels_zero = torch.zeros_like(raw_labels)
    for orig, new in label_to_idx.items():
        part_labels_zero[raw_labels == orig] = new

    part_type_ids = torch.tensor(part_type_ids_list, dtype=torch.int64)

    # DINOv2 tokens — deterministic quadrant sampling, matching the training
    # distribution. Training randomly picks 1 view per quadrant from
    # [0,1,2][3,4,5][6,7,8][9,10,11]; inference uses the middle elevation of
    # each quadrant (fixed indices) so train/infer see the same 360° coverage.
    tokens_path = os.path.join(recon_root, 'dinov2_tokens', obj_id, angle_dir, 'tokens.npz')
    if not os.path.isfile(tokens_path):
        raise FileNotFoundError(tokens_path)
    tokens = torch.from_numpy(np.load(tokens_path)['tokens']).float()  # [V, T, D]
    V_total, T, D = tokens.shape

    QUADRANTS = [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]]
    num_quads = min(len(QUADRANTS), (V_total + 2) // 3)
    # Middle index of each quadrant — deterministic counterpart to training's
    # random.choice(quadrant). Falls back gracefully if V_total < 12.
    selected_views = [q[len(q) // 2] for q in QUADRANTS[:num_quads]
                      if q[len(q) // 2] < V_total]
    # Cap at num_views (usually 4) to honor the caller's request.
    selected_views = selected_views[:num_views]
    n = len(selected_views)

    cond = tokens[selected_views].reshape(n * T, D)  # [n*T, D]

    # Per-token mask labels for V2 mask-in-KV (0=bg, 1..K=part)
    mask_token_labels = torch.zeros(n * T, dtype=torch.int64)  # default: all bg
    if mask_root is not None:
        mask_dir = os.path.join(mask_root, obj_id, angle_dir, 'mask')
        # _compute_mask_token_labels writes into a [V, T] tensor using actual
        # view indices, so pass V_total and the selected indices, then gather.
        mtl_full = _compute_mask_token_labels(
            V_total, T, mask_dir, label_to_idx, selected_views,
        )
        mask_token_labels = mtl_full[selected_views].reshape(-1)  # [n*T]

    return {
        'coords': coords,
        'cond': cond,
        'mask_token_labels': mask_token_labels,
        'part_labels': part_labels_zero,
        'part_type_ids': part_type_ids,
        'num_parts': num_parts,
        'label_to_part_name': label_to_part_name,
    }


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

    part_info_dir = os.path.join(recon_root, 'part_info')
    if not os.path.isdir(part_info_dir):
        return []
    samples = []
    for obj_id in sorted(os.listdir(part_info_dir)):
        lbl_dir = os.path.join(recon_root, 'part_labels', obj_id)
        if not os.path.isdir(lbl_dir):
            continue
        for angle_dir in sorted(os.listdir(lbl_dir)):
            if not angle_dir.startswith('angle_'):
                continue
            angle_idx = int(angle_dir.split('_')[1])
            if os.path.isfile(os.path.join(lbl_dir, angle_dir, 'part_labels_64.npy')):
                samples.append((obj_id, angle_idx))
    return samples


# ---------------------------------------------------------------------------
# 推理 + 指标
# ---------------------------------------------------------------------------

@torch.no_grad()
def run_inference(model: torch.nn.Module, sample: Dict[str, Any]) -> Dict[str, torch.Tensor]:
    """Run QueryPartPredictor V2 forward.

    K learnable queries, mask info injected into cond KV via mask_token_labels.
    K comes from upstream VLM at inference time (here we use GT K as a proxy).
    """
    coords = sample['coords'].cuda()              # [N, 3]
    cond = sample['cond'].cuda().unsqueeze(0)     # [1, V*T, D]
    num_parts = int(sample['num_parts'])
    mtl = sample['mask_token_labels'].cuda().unsqueeze(0)  # [1, V*T]

    result = model(coords, cond, num_parts=num_parts, mask_token_labels=mtl)
    # Legacy path returns a single dict
    if isinstance(result, list):
        return result[0]
    return result


def hungarian_match_pred_to_gt(
    soft_masks: torch.Tensor,   # [K, N] pred prob over parts (per voxel)
    gt_labels: torch.Tensor,    # [N] 0-indexed GT part labels
    num_parts: int,             # GT K (= K_pred in current design)
) -> Dict[int, int]:
    """Hungarian match pred queries → GT parts via soft-IoU cost.

    Returns pred_to_gt: {pred_query_idx: gt_part_idx}. Since K_pred == K_gt in
    this design, matching is 1:1 and every query has a GT mapping.
    """
    from scipy.optimize import linear_sum_assignment
    K, N = soft_masks.shape
    gt_masks = torch.zeros(num_parts, N, device=soft_masks.device, dtype=soft_masks.dtype)
    for m in range(num_parts):
        gt_masks[m] = (gt_labels.to(soft_masks.device) == m).to(soft_masks.dtype)

    # Soft IoU cost: 1 - IoU (lower = better match)
    inter = soft_masks @ gt_masks.T                                   # [K, M]
    sum_pred = soft_masks.sum(dim=1, keepdim=True)                    # [K, 1]
    sum_gt = gt_masks.sum(dim=1, keepdim=True).T                      # [1, M]
    union = sum_pred + sum_gt - inter                                 # [K, M]
    iou = inter / (union + 1e-6)                                      # [K, M]
    cost = (-iou).cpu().float().numpy()

    row_idx, col_idx = linear_sum_assignment(cost)
    return {int(r): int(c) for r, c in zip(row_idx, col_idx)}


def compute_part_miou(
    pred_dict: Dict[str, torch.Tensor],
    gt_labels: torch.Tensor,
    num_parts: int,
) -> Tuple[float, List[float], Dict[int, int]]:
    """Compute Part mIoU via Hungarian matching.

    Queries are generic learnable fallbacks → query k does NOT correspond to GT
    part k. We first match pred queries → GT parts (soft-IoU Hungarian), then
    for each GT part compute IoU between its matched pred mask and GT mask.

    Args:
        pred_dict: output of model forward (has 'soft_masks' [K, N])
        gt_labels: [N] 0-indexed GT part labels
        num_parts: K (GT count; == K_pred in current design)

    Returns:
        (mean_iou, per_part_iou_list, pred_to_gt_mapping)
    """
    soft_masks = pred_dict['soft_masks']           # [K, N]
    hard_pred = soft_masks.argmax(dim=0).cpu()     # [N] query idx per voxel

    pred_to_gt = hungarian_match_pred_to_gt(soft_masks, gt_labels, num_parts)

    gt_labels_cpu = gt_labels.cpu()
    per_part_iou = [0.0] * num_parts
    for pred_q, gt_p in pred_to_gt.items():
        pred_mask = (hard_pred == pred_q)
        gt_mask = (gt_labels_cpu == gt_p)
        inter = (pred_mask & gt_mask).sum().item()
        union = (pred_mask | gt_mask).sum().item()
        per_part_iou[gt_p] = float(inter) / float(union) if union > 0 else 0.0

    return float(np.mean(per_part_iou)), per_part_iou, pred_to_gt


# ---------------------------------------------------------------------------
# 可视化（单样本模式）
# ---------------------------------------------------------------------------

DISTINCT_COLORS = [
    '#e6194B', '#3cb44b', '#4363d8', '#f58231', '#911eb4',
    '#42d4f4', '#f032e6', '#bfef45', '#fabed4', '#469990',
    '#dcbeff', '#9A6324', '#800000', '#aaffc3', '#808000',
    '#000075', '#a9a9a9', '#ffe119', '#ffd8b1', '#e6beff',
]


def _render_voxel_segmentation(
    coords: np.ndarray,
    labels: np.ndarray,
    title: str,
    output_path: str,
) -> None:
    """Render single full-size voxel segmentation plot matching reference style.

    Solid cubes, high-contrast palette, front-facing 3/4 view, legend with
    'part 1', 'part 2', ..., axis ticks every 4 from 0 to 64.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import to_rgba
    from matplotlib.patches import Patch

    unique_labels = sorted(set(int(l) for l in np.unique(labels) if l >= 0))
    label_to_color = {lbl: DISTINCT_COLORS[i % len(DISTINCT_COLORS)]
                      for i, lbl in enumerate(unique_labels)}

    filled = np.zeros((64, 64, 64), dtype=bool)
    facecolors = np.zeros((64, 64, 64, 4))
    for i in range(coords.shape[0]):
        x, y, z = int(coords[i, 0]), int(coords[i, 1]), int(coords[i, 2])
        lbl = int(labels[i])
        if lbl < 0:
            continue
        filled[x, y, z] = True
        facecolors[x, y, z] = to_rgba(label_to_color[lbl], alpha=0.9)

    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    ax.voxels(filled, facecolors=facecolors,
              edgecolors=facecolors * 0.7 + 0.3, linewidth=0.1)
    ax.set_title(title, fontsize=14, pad=15)
    ax.set_xlabel('X', fontsize=11)
    ax.set_ylabel('Y', fontsize=11)
    ax.set_zlabel('Z', fontsize=11)
    ax.view_init(elev=20, azim=-60)
    ticks = range(0, 65, 4)
    ax.set_xticks(ticks); ax.set_yticks(ticks); ax.set_zticks(ticks)
    ax.tick_params(labelsize=7)

    handles = [Patch(facecolor=label_to_color[lbl], label=f'part {lbl + 1}')
               for lbl in unique_labels]
    ax.legend(handles=handles, loc='upper right', fontsize=10)

    os.makedirs(os.path.dirname(os.path.abspath(output_path)) or '.', exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def save_visualization(
    sample: Dict[str, Any],
    pred_dict: Dict[str, torch.Tensor],
    pred_to_gt: Dict[int, int],
    output_dir: str,
    obj_id: str,
    angle_idx: int,
    mean_iou: float,
) -> List[str]:
    """Save GT + Pred part segmentation as separate full-size images.

    Returns list of saved paths.
    """
    coords_np = sample['coords'].numpy()
    gt_labels_np = sample['part_labels'].numpy()

    soft_masks = pred_dict['soft_masks'].cpu()
    pred_query_idx = soft_masks.argmax(dim=0).numpy()
    pred_labels_np = np.full_like(pred_query_idx, -1)
    for q, g in pred_to_gt.items():
        pred_labels_np[pred_query_idx == q] = g

    paths = []

    gt_path = os.path.join(output_dir, f'viz_{obj_id}_{angle_idx}_gt.png')
    _render_voxel_segmentation(
        coords_np, gt_labels_np,
        f'GT Part Segmentation: {obj_id} angle {angle_idx}',
        gt_path,
    )
    paths.append(gt_path)

    pred_path = os.path.join(output_dir, f'viz_{obj_id}_{angle_idx}_pred.png')
    _render_voxel_segmentation(
        coords_np, pred_labels_np,
        f'Pred Part Segmentation: {obj_id} angle {angle_idx}  mIoU={mean_iou:.4f}',
        pred_path,
    )
    paths.append(pred_path)

    return paths


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description='Part Predictor 推理 + Part mIoU')
    p.add_argument('--ckpt', required=True, help='checkpoint .pt 路径（train.py 保存的）')
    p.add_argument('--data_root', required=True)
    p.add_argument('--recon_subdir', default=RECON_SUBDIR)
    p.add_argument('--obj_id', default=None)
    p.add_argument('--angle_idx', type=int, default=0)
    p.add_argument('--manifest', default=None)
    p.add_argument('--views', type=int, default=NUM_VIEWS)
    p.add_argument('--mask_subdir', default=None,
                   help='2D mask 子目录（相对 data_root），如 arts/renders。不传则走 fallback_queries')
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
    mask_root = os.path.join(args.data_root, args.mask_subdir) if args.mask_subdir else None

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
            sample = load_sample(recon_root, obj_id, angle_idx, args.views, mask_root)
        except FileNotFoundError as e:
            print(f'[WARN] missing data: {e}')
            continue

        pred_dict = run_inference(model, sample)
        mean_iou, per_part_iou, pred_to_gt = compute_part_miou(
            pred_dict, sample['part_labels'], sample['num_parts'],
        )

        entry = {
            'obj_id': obj_id, 'angle_idx': angle_idx,
            'num_parts': sample['num_parts'],
            'part_miou': mean_iou,
            'per_part_iou': per_part_iou,
        }
        results.append(entry)
        print(f'[INFO] {obj_id}/angle_{angle_idx}: K={sample["num_parts"]} part_miou={mean_iou:.4f} '
              f'per_part={[f"{v:.3f}" for v in per_part_iou]}')

        if single_mode:
            viz_paths = save_visualization(
                sample, pred_dict, pred_to_gt, args.output, obj_id, angle_idx, mean_iou,
            )
            for vp in viz_paths:
                print(f'[INFO] viz → {vp}')

    mean_miou = float(np.mean([r['part_miou'] for r in results])) if results else 0.0
    out = {
        'ckpt': args.ckpt, 'views': args.views,
        'n_samples': len(results), 'mean_part_miou': mean_miou,
        'samples': results,
    }
    metrics_path = os.path.join(args.output, 'metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(out, f, indent=2)
    print(f'[DONE] mean_part_miou={mean_miou:.4f}  n={len(results)} → {metrics_path}')


if __name__ == '__main__':
    main()
