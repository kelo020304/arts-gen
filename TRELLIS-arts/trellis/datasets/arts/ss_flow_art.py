"""
Stage 2 多视角条件 SS Flow 数据集（dense 16³ structured-latent flow）。

从 arts/reconstruction/ 目录加载：
  - SS-VAE latent (dense `[8,16,16,16]` from key `mean`)
  - DINOv2 tokens（多视角 concat）
  - Part labels（可选，64³ 体素标签体积）

数据目录结构:
    {data_root}/arts/reconstruction/ss_latents_expanded/{obj_id}/angle_{angle_idx}/latent.npz
        (key='mean', shape=[8,16,16,16] float32 dense)
    {data_root}/arts/reconstruction/dinov2_tokens/{obj_id}/angle_{angle_idx}/tokens.npz
    {data_root}/arts/reconstruction/part_labels/{obj_id}/angle_{angle_idx}/part_labels_64.npy

History note (Phase 9 — 2026-04-26):
  Earlier code (scripts/train/stage2/dataset.py byte-faithfully migrated here)
  read `latent.npz['coords' / 'feats']` and built a SparseTensor — but the
  `encode_ss_latent_expanded.py` encoder always wrote `{mean}` (dense). The
  trainer parent (ImageConditionedSparseFlowMatchingCFGTrainer) was also
  sparse, while the model `SparseStructureFlowModel.forward(x: torch.Tensor)`
  is dense. The whole stack was end-to-end inconsistent and never actually
  trained successfully. Phase 9 09-10 phase-gate exposed this and we fixed
  three places in lockstep:
    1. Dataset returns `x_0: [8,16,16,16]` dense (this file)
    2. collate_fn stacks along batch dim (this file)
    3. Trainer parent → ImageConditionedFlowMatchingCFGTrainer (dense)
       (trellis/trainers/arts/ss_flow_art.py)

用法:
    from trellis.datasets.arts.ss_flow_art import MvImageConditionedSLatDataset
    ds = MvImageConditionedSLatDataset(config_to_dict(cfg.data))
"""

import os
import json
import random
from typing import Dict, List, Optional, Any

import numpy as np
import torch
from torch.utils.data import Dataset


class MvImageConditionedSLatDataset(Dataset):
    """多视角图像条件下的 SLat 数据集。

    加载 SS-VAE latent（稀疏结构）+ 多视角 DINOv2 tokens 条件，
    支持 view dropout 随机丢弃视角以增强泛化性。

    Args:
        data_config: 数据配置 dict，包含以下字段:
            - data_root (str): 数据根目录
            - manifest_path (str): manifest.json 相对于 data_root 的路径
            - num_views (int): 最大视角数，默认 4
            - min_views (int): 最小视角数（用于 view dropout），默认 1
            - view_dropout (bool): 是否启用 view dropout，默认 True
    """

    # 数据集 value_range，Trainer 的 visualize_sample 使用
    value_range = (0, 1)

    def __init__(self, data_config: dict):
        super().__init__()
        self.data_root = data_config['data_root']
        self.manifest_path = data_config['manifest_path']
        self.num_views = data_config.get('num_views', 4)
        self.min_views = data_config.get('min_views', 1)
        self.view_dropout = data_config.get('view_dropout', True)

        # 数据子目录
        recon_root = os.path.join(self.data_root, 'arts', 'reconstruction')
        self.latent_root = os.path.join(recon_root, 'ss_latents_expanded')
        self.token_root = os.path.join(recon_root, 'dinov2_tokens')
        self.label_root = os.path.join(recon_root, 'part_labels')

        # 可选：只使用指定物体（用于 smoke test 等小规模验证）
        test_obj_ids = data_config.get('test_obj_ids', None)

        # 加载 manifest 并展开为 (obj_id, angle_idx) 样本列表
        # 支持两种 manifest 格式:
        #   1. assembler 格式 (dict): {"samples": [{"object_id": ..., "angle_idx": ..., "complete": ...}]}
        #   2. 旧格式 (list): [{"id": ..., "angles": [...]}]
        manifest_abs = os.path.join(self.data_root, self.manifest_path)
        if not os.path.exists(manifest_abs):
            # manifest 不存在时，从 test_obj_ids + 目录枚举构建样本列表
            if test_obj_ids:
                self.samples = self._enumerate_from_dir(recon_root, test_obj_ids)
            else:
                raise FileNotFoundError(f"manifest 不存在: {manifest_abs}")
        else:
            with open(manifest_abs, 'r') as f:
                manifest = json.load(f)

            self.samples: List[tuple] = []
            if isinstance(manifest, dict) and 'samples' in manifest:
                for entry in manifest['samples']:
                    if entry.get('complete', False):
                        obj_id = str(entry['object_id'])
                        angle_idx = int(entry['angle_idx'])
                        self.samples.append((obj_id, angle_idx))
            elif isinstance(manifest, list):
                for entry in manifest:
                    obj_id = str(entry['id'])
                    for angle_idx in entry['angles']:
                        self.samples.append((obj_id, int(angle_idx)))
            else:
                raise ValueError(f"无法识别的 manifest 格式: {type(manifest)}")

        # 如果指定了 test_obj_ids，过滤样本
        if test_obj_ids and self.samples:
            test_set = set(str(x) for x in test_obj_ids)
            self.samples = [(oid, aidx) for oid, aidx in self.samples if oid in test_set]

        # 用于 load-balanced collate 的负载估计
        self.loads = [1] * len(self.samples)

        print(f'[MvImageConditionedSLatDataset] 加载 {len(self.samples)} 个样本 '
              f'(views={self.num_views}, dropout={self.view_dropout})')

    @staticmethod
    def _enumerate_from_dir(recon_root: str, obj_ids: list) -> list:
        """从目录结构枚举样本（manifest 不存在时的 fallback）。"""
        samples = []
        latent_root = os.path.join(recon_root, 'ss_latents_expanded')
        for obj_id in obj_ids:
            obj_dir = os.path.join(latent_root, str(obj_id))
            if not os.path.isdir(obj_dir):
                continue
            for entry in sorted(os.listdir(obj_dir)):
                if entry.startswith('angle_'):
                    try:
                        angle_idx = int(entry.split('_')[1])
                        samples.append((str(obj_id), angle_idx))
                    except (ValueError, IndexError):
                        pass
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """加载单个样本。

        Returns:
            dict，包含:
                - coords (Tensor [N, 3] int32): 稀疏体素坐标
                - feats (Tensor [N, C] float32): 稀疏体素特征
                - cond (Tensor [num_selected_views * T, D] float32): 多视角 DINOv2 tokens
        """
        try:
            return self._load_sample(idx)
        except Exception as e:
            # 加载失败时随机返回另一个样本（与 TRELLIS 原版行为一致）
            print(f'[WARN] 样本 {idx} 加载失败: {e}')
            return self.__getitem__(np.random.randint(0, len(self)))

    def _load_sample(self, idx: int) -> Dict[str, Any]:
        """实际加载逻辑。"""
        obj_id, angle_idx = self.samples[idx]
        angle_dir = f'angle_{angle_idx}'

        # ---- 1. 加载 SS-VAE latent (dense [8,16,16,16] from key 'mean') ----
        # encode_ss_latent_expanded.py writes np.savez_compressed(p, mean=latent)
        # where latent is the [C,16,16,16] feature volume. The Stage 2 SS Flow
        # model is dense; we feed the whole volume as x_0.
        latent_path = os.path.join(self.latent_root, obj_id, angle_dir, 'latent.npz')
        latent_data = np.load(latent_path)
        if 'mean' not in latent_data.files:
            raise KeyError(
                f"latent.npz at {latent_path} expected key 'mean' "
                f"(dense ss_latent), found keys: {list(latent_data.files)}"
            )
        latent = torch.from_numpy(np.asarray(latent_data['mean'])).float()  # [C,16,16,16]
        if latent.ndim != 4:
            raise ValueError(
                f"{latent_path}['mean'] expected ndim=4 [C,D,H,W], got shape {tuple(latent.shape)}"
            )

        # ---- 2. 加载多视角 DINOv2 tokens ----
        token_dir = os.path.join(self.token_root, obj_id, angle_dir)
        tokens_data = np.load(os.path.join(token_dir, 'tokens.npz'))
        # tokens.npz 中存储 tokens 字段，shape: [num_views, T, D]
        # 其中 T = 1 + num_patches (CLS + patch tokens) = 1370 for DINOv2-L/14 at 518px
        # D = 1024
        all_tokens = torch.tensor(tokens_data['tokens']).float()  # [V, T, D]
        total_views = all_tokens.shape[0]

        # view dropout: 随机选择 [min_views, num_views] 个视角
        if self.view_dropout and self.min_views < self.num_views:
            n_select = random.randint(self.min_views, min(self.num_views, total_views))
        else:
            n_select = min(self.num_views, total_views)

        # 随机选择视角索引
        view_indices = sorted(random.sample(range(total_views), n_select))
        selected_tokens = all_tokens[view_indices]  # [n_select, T, D]

        # 对缺失的视角用零填充，保持固定长度
        T, D = all_tokens.shape[1], all_tokens.shape[2]
        if n_select < self.num_views:
            pad_tokens = torch.zeros(self.num_views - n_select, T, D)
            selected_tokens = torch.cat([selected_tokens, pad_tokens], dim=0)

        # concat 所有视角 tokens: [num_views * T, D]
        cond = selected_tokens.reshape(-1, D)

        # Stage 2 SS Flow only uses (x_0, cond). Part labels are consumed by
        # part_predictor / part_ss_latent_flow tasks, not by SparseStructureFlowModel.
        # `training_losses(**batch) → model(x_t, t, cond, **kwargs)` propagates any
        # extra dict key as a model kwarg → TypeError. Drop them here.
        result = {
            'x_0': latent,    # dense [C, 16, 16, 16]
            'cond': cond,
        }
        return result

    @staticmethod
    def collate_fn(batch, split_size=None):
        """Dense batch collation for Stage 2 SS Flow.

        Each sample contributes:
          - 'x_0': dense [C, D, H, W] latent  (e.g. [8, 16, 16, 16])
          - 'cond': [V*T, D] DINOv2 tokens
          - 'part_labels' (optional): [64,64,64] int64

        Stacks tensors along a new batch dim. No SparseTensor.

        Args:
            batch: list of dicts from __getitem__
            split_size: optional, gradient-accumulation micro-batch size.
                When set, the batch is split into len(batch)//split_size packs
                so each forward pass uses a smaller effective batch.

        Returns:
            single pack dict (split_size=None) or list of pack dicts.
        """

        if split_size is None:
            group_idx = [list(range(len(batch)))]
        else:
            # Equal-size groups; for dense batches we don't need
            # load-balanced grouping (no variable-length feats).
            group_idx = [
                list(range(i, min(i + split_size, len(batch))))
                for i in range(0, len(batch), split_size)
            ]

        packs = []
        for group in group_idx:
            sub_batch = [batch[i] for i in group]
            pack = {}
            keys = list(sub_batch[0].keys())
            for k in keys:
                if isinstance(sub_batch[0][k], torch.Tensor):
                    pack[k] = torch.stack([b[k] for b in sub_batch])
                elif isinstance(sub_batch[0][k], list):
                    pack[k] = sum([b[k] for b in sub_batch], [])
                else:
                    pack[k] = [b[k] for b in sub_batch]
            packs.append(pack)

        if split_size is None:
            return packs[0]
        return packs
