"""
MvImageConditionedSparseLatentDataset -- Phase 3 Stage 4 SLat Flow dataset.

Reads sparse SLat VAE latents and multi-view DINOv2 tokens.  The legacy layout is
`reconstruction/slat_latents_expanded/{obj_id}/angle_{N}/latent.npz` plus
`reconstruction/dinov2_tokens/{obj_id}/angle_{N}/tokens.npz`.  For the corrected
whole-then-slice SLat finetune path, set `slat_layout=part_synthesis_overall`,
`slat_subdir=part_synthesis_slat`, and `token_subdir=dinov2_tokens_prenorm`.

Key differences vs Phase 2 MvImageConditionedSLatDataset (stage2/dataset.py):
  - Phase 2: dense latent [8,16,16,16] from ss_latents_expanded/ (key='mean')
  - Phase 3: sparse (coords[N,3] + feats[N,8]) from slat_latents_expanded/
  - Phase 2: no normalization in dataset; Phase 3: dataset-layer (feats - mean)/std
  - Phase 2: custom dense collate_fn; Phase 3: reuse SLat.collate_fn (layout + SparseTensor pack)
  - Phase 3 __init__ contains first-sample npz schema sanity check; fails loudly on contract drift

References:
  - 03-CONTEXT.md D-01, D-06, D-19, D-24
  - 03-RESEARCH.md section 3
  - TRELLIS SLat.collate_fn: trellis/datasets/structured_latent.py:161-203
"""

import json
import os
import random
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

# Stage 4 reuses TRELLIS canonical sparse collate (static method, layout cache logic intact)
from trellis.datasets.structured_latent import SLat


# TRELLIS canonical 8-dim normalization (source: slat_flow_img_dit_L_64l8p2_fp16.json)
# TODO(H200-run1): recompute on PhysX-Mobility slat_latents_expanded and override these defaults
DEFAULT_MEAN = [
    -2.1687545776367188, -0.004347046371549368, -0.13352349400520325,
    -0.08418072760105133, -0.5271206498146057,   0.7238689064979553,
    -1.1414450407028198,   1.2039363384246826,
]
DEFAULT_STD = [
     2.377650737762451,   2.386378288269043,  2.124418020248413,
     2.1748552322387695,  2.663944721221924,  2.371192216873169,
     2.6217446327209473,  2.684523105621338,
]


class MvImageConditionedSparseLatentDataset(Dataset):
    """SLat Flow multi-view conditioned dataset (Phase 3 Stage 4).

    __getitem__ output:
        {
            'coords': int32 Tensor [N, 3] -- xyz in 64^3 grid
            'feats':  float32 Tensor [N, 8] -- NORMALIZED SLat features
            'cond':   float32 Tensor [V*T, D] -- flattened multi-view DINOv2 tokens
        }

    Where V is padded to num_views for batching, T is usually 1374 for official
    prenorm DINOv2 tokens, D = 1024, and N <= max_num_voxels.

    collate_fn = SLat.collate_fn (staticmethod ref) handles SparseTensor pack + layout cache.
    """

    # REUSE -- do NOT rewrite! Layout cache + _shape setup + balanced grouping
    # all live in the original TRELLIS implementation.
    # CRITICAL: must be staticmethod so instance access returns unbound function.
    # A plain `collate_fn = SLat.collate_fn` stores the raw function in __dict__,
    # which Python's descriptor protocol then binds to `self` when accessed via
    # instance — causing `split_size` to receive the batch list as a positional arg
    # and "got multiple values" TypeError in DataLoader workers.
    collate_fn = staticmethod(SLat.collate_fn)

    # Trainer.visualize_sample compatibility
    value_range = (0, 1)

    def __init__(self, cfg: Dict[str, Any]):
        """Initialize dataset.

        cfg keys:
            data_root:       str (e.g. 'data/smoke_test')
            recon_subdir:    str (e.g. 'reconstruction')
            manifest_path:   str | None (relative to data_root or absolute; None -> fallback to test_obj_ids)
            manifest_format: str ('auto' | 'json' | 'jsonl')
            test_obj_ids:    list[str] | None (used when manifest_path is None)
            slat_subdir:     str (legacy default 'slat_latents_expanded')
            slat_root:       str | None (absolute or relative to data_root; overrides slat_subdir)
            slat_layout:     str ('legacy_angle' | 'part_synthesis_overall')
            token_subdir:    str (default 'dinov2_tokens')
            token_root:      str | None (absolute or relative to recon_root; overrides token_subdir)
            token_condition_norm: str ('none' | 'layer_norm')
            use_manifest_view_indices: bool (default False; True for part-completion jsonl)
            num_views:       int (default 4)
            min_views:       int (default 1)
            view_dropout:    bool (default True)
            max_num_voxels:  int (default 32768, D-19)
            normalization:   {'mean': list[8], 'std': list[8]} | None
        """
        super().__init__()
        self.data_root = os.path.abspath(os.path.expanduser(cfg['data_root']))
        self.recon_subdir = cfg.get('recon_subdir', 'reconstruction')
        self.manifest_path = cfg.get('manifest_path', None)
        self.manifest_format = str(cfg.get('manifest_format', 'auto')).lower()
        self.test_obj_ids = cfg.get('test_obj_ids', None)
        self.slat_subdir = cfg.get('slat_subdir', 'slat_latents_expanded')
        self.slat_layout = cfg.get('slat_layout', 'legacy_angle')
        self.token_subdir = cfg.get('token_subdir', 'dinov2_tokens')
        self.token_condition_norm = str(cfg.get('token_condition_norm', 'none')).lower()
        self.use_manifest_view_indices = bool(cfg.get('use_manifest_view_indices', False))
        self.num_views = int(cfg.get('num_views', 4))
        self.min_views = int(cfg.get('min_views', 1))
        self.view_dropout = bool(cfg.get('view_dropout', True))
        self.max_num_voxels = int(cfg.get('max_num_voxels', 32768))
        self.token_num_tokens = int(cfg.get('token_num_tokens', 0))
        self.token_dim = int(cfg.get('token_dim', 1024))
        if self.num_views < 1:
            raise ValueError(f'num_views must be >= 1, got {self.num_views}')
        if not (1 <= self.min_views <= self.num_views):
            raise ValueError(f'min_views must satisfy 1 <= min_views <= num_views, got {self.min_views}/{self.num_views}')
        if self.slat_layout not in {'legacy_angle', 'part_synthesis_overall'}:
            raise ValueError(f'Unsupported slat_layout={self.slat_layout!r}')
        if self.manifest_format not in {'auto', 'json', 'jsonl'}:
            raise ValueError(f'Unsupported manifest_format={self.manifest_format!r}')
        if self.token_condition_norm not in {'none', 'layer_norm'}:
            raise ValueError(f'Unsupported token_condition_norm={self.token_condition_norm!r}')

        # Normalization (CONTEXT D-06: dataset layer does (feats - mean) / std)
        norm_cfg = cfg.get('normalization', {}) or {}
        mean = norm_cfg.get('mean', DEFAULT_MEAN)
        std = norm_cfg.get('std', DEFAULT_STD)
        self.mean = torch.tensor(mean, dtype=torch.float32).view(1, 8)
        self.std = torch.tensor(std, dtype=torch.float32).view(1, 8)

        # Path roots
        self.recon_root = self._resolve_path(self.recon_subdir, base=self.data_root)
        self.slat_root = self._resolve_path(
            cfg.get('slat_root') or self.slat_subdir,
            base=self.data_root if self.slat_layout == 'part_synthesis_overall' else self.recon_root,
        )
        self.token_root = self._resolve_path(cfg.get('token_root') or self.token_subdir, base=self.recon_root)

        # Sample enumeration (reuse Phase 2 pattern)
        self.samples: List[Dict[str, Any]] = self._enumerate_samples()
        self.loads = [1] * len(self.samples)  # BalancedResumableSampler uniform load

        # CRITICAL: first-sample sanity check (RESEARCH section 3.8)
        if len(self.samples) > 0:
            self._validate_first_sample_schema()

        print(
            f'[MvImageConditionedSparseLatentDataset] loaded {len(self.samples)} samples '
            f'(slat_root={self.slat_root}, slat_layout={self.slat_layout}, '
            f'token_root={self.token_root}, num_views={self.num_views}, '
            f'token_condition_norm={self.token_condition_norm}, '
            f'min_views={self.min_views}, view_dropout={self.view_dropout}, '
            f'use_manifest_view_indices={self.use_manifest_view_indices}, '
            f'max_num_voxels={self.max_num_voxels})'
        )

    @staticmethod
    def _resolve_path(path: str, *, base: str) -> str:
        path = os.path.expanduser(str(path))
        if os.path.isabs(path):
            return os.path.abspath(path)
        return os.path.abspath(os.path.join(base, path))

    @staticmethod
    def _sample_record(obj_id: str, angle_idx: int, *, view_indices=None, sample_id: str = '') -> Dict[str, Any]:
        return {
            'object_id': str(obj_id),
            'angle_idx': int(angle_idx),
            'view_indices': [int(v) for v in view_indices] if view_indices is not None else None,
            'sample_id': str(sample_id) if sample_id else f'{obj_id}_angle_{angle_idx}',
        }

    def _manifest_abs_path(self) -> str:
        if self.manifest_path is None:
            raise ValueError('manifest_path is None')
        return self._resolve_path(self.manifest_path, base=self.data_root)

    def _load_manifest_samples(self, manifest_abs: str) -> List[Dict[str, Any]]:
        fmt = self.manifest_format
        if fmt == 'auto':
            suffix = Path(manifest_abs).suffix.lower()
            fmt = 'jsonl' if suffix == '.jsonl' else 'json'

        samples: List[Dict[str, Any]] = []
        if fmt == 'jsonl':
            with open(manifest_abs, 'r', encoding='utf-8') as f:
                for line_no, line in enumerate(f, start=1):
                    line = line.strip()
                    if not line:
                        continue
                    entry = json.loads(line)
                    if not isinstance(entry, dict):
                        raise ValueError(f'{manifest_abs}:{line_no}: expected JSON object')
                    if 'complete' in entry and not entry['complete']:
                        continue
                    if 'object_id' not in entry or 'angle_idx' not in entry:
                        raise ValueError(f'{manifest_abs}:{line_no}: missing object_id/angle_idx')
                    samples.append(self._sample_record(
                        entry['object_id'],
                        entry['angle_idx'],
                        view_indices=entry.get('view_indices'),
                        sample_id=entry.get('sample_id', ''),
                    ))
            return samples

        with open(manifest_abs, 'r', encoding='utf-8') as f:
            manifest = json.load(f)
        if not isinstance(manifest, dict) or 'samples' not in manifest:
            raise ValueError(f'{manifest_abs}: expected JSON manifest with top-level "samples"')
        for idx, entry in enumerate(manifest['samples']):
            if not isinstance(entry, dict):
                raise ValueError(f'{manifest_abs}: samples[{idx}] is not an object')
            if not entry.get('complete', False):
                continue
            if 'object_id' not in entry or 'angle_idx' not in entry:
                raise ValueError(f'{manifest_abs}: samples[{idx}] missing object_id/angle_idx')
            samples.append(self._sample_record(
                entry['object_id'],
                entry['angle_idx'],
                view_indices=entry.get('view_indices'),
                sample_id=entry.get('sample_id', ''),
            ))
        return samples

    def _enumerate_samples(self) -> List[Dict[str, Any]]:
        """Enumerate samples: manifest first, then test_obj_ids directory scan."""
        samples: List[Dict[str, Any]] = []

        if self.manifest_path:
            manifest_abs = self._manifest_abs_path()
            if not os.path.isfile(manifest_abs):
                raise FileNotFoundError(f'manifest_path not found: {manifest_abs}')
            samples = self._load_manifest_samples(manifest_abs)
            if not samples:
                raise RuntimeError(f'{manifest_abs}: produced 0 samples')
            return samples

        # Fallback: scan directories
        if self.test_obj_ids:
            for obj_id in self.test_obj_ids:
                obj_id = str(obj_id)
                if self.slat_layout == 'part_synthesis_overall':
                    obj_dir = os.path.join(self.slat_root, obj_id[:2])
                    prefix = f'{obj_id}_angle_'
                else:
                    obj_dir = os.path.join(self.slat_root, obj_id)
                    prefix = 'angle_'
                if not os.path.isdir(obj_dir):
                    continue
                for entry in sorted(os.listdir(obj_dir)):
                    if not entry.startswith(prefix):
                        continue
                    try:
                        angle_idx = int(entry.split('_')[-1])
                    except ValueError:
                        continue
                    samples.append(self._sample_record(obj_id, angle_idx))
            return samples

        raise FileNotFoundError(
            f'Neither manifest_path ({self.manifest_path}) nor test_obj_ids available; '
            f'cannot enumerate samples under {self.recon_root}'
        )

    def _latent_path(self, obj_id: str, angle_idx: int) -> str:
        if self.slat_layout == 'part_synthesis_overall':
            instance_id = f'{obj_id}_angle_{angle_idx}'
            return os.path.join(self.slat_root, obj_id[:2], instance_id, 'overall', 'latent.npz')
        return os.path.join(self.slat_root, obj_id, f'angle_{angle_idx}', 'latent.npz')

    def _token_path(self, obj_id: str, angle_idx: int) -> str:
        return os.path.join(self.token_root, obj_id, f'angle_{angle_idx}', 'tokens.npz')

    def _validate_first_sample_schema(self):
        """First-sample sanity check -- fails loudly to surface data-team contract drift."""
        sample = self.samples[0]
        obj_id = sample['object_id']
        angle_idx = sample['angle_idx']
        npz_path = self._latent_path(obj_id, angle_idx)
        token_path = self._token_path(obj_id, angle_idx)
        if not os.path.isfile(npz_path):
            raise FileNotFoundError(f'first sample latent not found: {npz_path}')
        if not os.path.isfile(token_path):
            raise FileNotFoundError(f'first sample DINO tokens not found: {token_path}')
        errors = []
        with np.load(npz_path) as data:
            if 'coords' not in data.files:
                errors.append(f"missing 'coords' key (got: {data.files})")
            if 'feats' not in data.files:
                errors.append(f"missing 'feats' key (got: {data.files})")
            if errors:
                raise ValueError(
                    f"[SCHEMA] {npz_path}:\n  " + "\n  ".join(errors) +
                    "\n  Expected: {'coords': int32 [N,3], 'feats': float32 [N,8]}"
                )

            c = data['coords']
            f = data['feats']
        if not np.issubdtype(c.dtype, np.integer):
            errors.append(f"coords dtype: expected integer dtype, got {c.dtype}")
        if c.ndim != 2 or c.shape[1] != 3:
            errors.append(f"coords shape: expected [N,3], got {c.shape}")
        elif c.size > 0 and (int(c.min()) < 0 or int(c.max()) >= 64):
            errors.append(f"coords range: expected [0,64), got min={int(c.min())} max={int(c.max())}")
        if f.dtype != np.float32:
            errors.append(f"feats dtype: expected float32, got {f.dtype}")
        if f.ndim != 2 or f.shape[1] != 8:
            errors.append(f"feats shape: expected [N,8], got {f.shape}")
        if c.shape[0] != f.shape[0]:
            errors.append(f"coords/feats N mismatch: {c.shape[0]} vs {f.shape[0]}")
        if errors:
            raise ValueError(
                f"[SCHEMA] {npz_path} contract violation:\n  " + "\n  ".join(errors)
            )

        with np.load(token_path) as token_data:
            if set(token_data.files) != {'tokens'}:
                errors.append(f"tokens keys: expected ['tokens'], got {sorted(token_data.files)}")
            else:
                tokens = token_data['tokens']
                if tokens.dtype != np.float32:
                    errors.append(f"tokens dtype: expected float32, got {tokens.dtype}")
                if tokens.ndim != 3:
                    errors.append(f"tokens shape: expected [V,T,D], got {tokens.shape}")
                else:
                    if self.token_num_tokens > 0 and tokens.shape[1] != self.token_num_tokens:
                        errors.append(f"tokens T: expected {self.token_num_tokens}, got {tokens.shape[1]}")
                    if tokens.shape[2] != self.token_dim:
                        errors.append(f"tokens D: expected {self.token_dim}, got {tokens.shape[2]}")
                    view_indices = sample.get('view_indices')
                    if self.use_manifest_view_indices:
                        if not view_indices:
                            errors.append('use_manifest_view_indices=True but first sample has no view_indices')
                        elif any(v < 0 or v >= tokens.shape[0] for v in view_indices):
                            errors.append(f'view_indices out of range for tokens V={tokens.shape[0]}: {view_indices}')
                    elif tokens.shape[0] < self.num_views:
                        errors.append(f'tokens V={tokens.shape[0]} < num_views={self.num_views}')
        if errors:
            raise ValueError(
                f"[SCHEMA] first sample contract violation:\n  " + "\n  ".join(errors)
            )

        print(f'[MvImageConditionedSparseLatentDataset] schema OK '
              f'(first sample: obj={obj_id} angle={angle_idx} N={c.shape[0]} voxels, '
              f'latent={npz_path}, tokens={token_path})')

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self._load_sample(idx)

    def _select_views(self, tokens: torch.Tensor, sample: Dict[str, Any]) -> torch.Tensor:
        V_total = tokens.shape[0]
        view_indices = sample.get('view_indices')
        if self.use_manifest_view_indices:
            if not view_indices:
                raise ValueError(f"{sample['sample_id']}: manifest view_indices required but missing")
            if len(view_indices) < self.min_views:
                raise ValueError(
                    f"{sample['sample_id']}: view_indices has {len(view_indices)} views, min_views={self.min_views}"
                )
            if any(v < 0 or v >= V_total for v in view_indices):
                raise ValueError(f"{sample['sample_id']}: view_indices out of [0,{V_total}): {view_indices}")
            base_indices = list(view_indices)
        else:
            if V_total < self.min_views:
                raise ValueError(f"{sample['sample_id']}: tokens has only {V_total} views, min_views={self.min_views}")
            base_indices = list(range(min(self.num_views, V_total)))

        if len(base_indices) > self.num_views:
            base_indices = base_indices[:self.num_views]

        if self.view_dropout and self.min_views < len(base_indices):
            n_keep = random.randint(self.min_views, len(base_indices))
            keep_positions = sorted(random.sample(range(len(base_indices)), n_keep))
            base_indices = [base_indices[pos] for pos in keep_positions]

        picked = tokens[torch.tensor(base_indices, dtype=torch.long)]
        T, D = picked.shape[1], picked.shape[2]
        if picked.shape[0] < self.num_views:
            pad = torch.zeros(self.num_views - picked.shape[0], T, D, dtype=picked.dtype)
            picked = torch.cat([picked, pad], dim=0)
        return picked

    def _load_sample(self, idx: int) -> Dict[str, Any]:
        sample = self.samples[idx]
        obj_id = sample['object_id']
        angle_idx = sample['angle_idx']

        # --- 1. Sparse latent ---
        npz_path = self._latent_path(obj_id, angle_idx)
        if not os.path.isfile(npz_path):
            raise FileNotFoundError(f"{sample['sample_id']}: latent not found: {npz_path}")
        with np.load(npz_path) as data:
            coords = torch.from_numpy(data['coords']).int()     # [N, 3]
            feats = torch.from_numpy(data['feats']).float()      # [N, 8] raw
        if coords.ndim != 2 or coords.shape[1] != 3:
            raise ValueError(f'{npz_path}: coords must be [N,3], got {tuple(coords.shape)}')
        if feats.ndim != 2 or feats.shape[1] != 8:
            raise ValueError(f'{npz_path}: feats must be [N,8], got {tuple(feats.shape)}')
        if coords.shape[0] != feats.shape[0]:
            raise ValueError(f'{npz_path}: coords/feats N mismatch: {coords.shape[0]} vs {feats.shape[0]}')

        # max_num_voxels guard (production safety -- smoke scale N << 32768)
        if coords.shape[0] > self.max_num_voxels:
            raise ValueError(
                f'{npz_path}: N={coords.shape[0]} > max_num_voxels={self.max_num_voxels}'
            )

        # Normalization (D-06)
        feats = (feats - self.mean) / self.std               # [N, 8] normalized

        # --- 2. Multi-view DINOv2 tokens ---
        token_path = self._token_path(obj_id, angle_idx)
        if not os.path.isfile(token_path):
            raise FileNotFoundError(f"{sample['sample_id']}: DINO tokens not found: {token_path}")
        with np.load(token_path) as token_data:
            if set(token_data.files) != {'tokens'}:
                raise ValueError(f"{token_path}: expected only 'tokens' key, got {sorted(token_data.files)}")
            tokens_np = token_data['tokens']
        if tokens_np.dtype != np.float32:
            raise ValueError(f'{token_path}: tokens dtype must be float32, got {tokens_np.dtype}')
        if tokens_np.ndim != 3:
            raise ValueError(f'{token_path}: tokens must be [V,T,D], got {tokens_np.shape}')
        if self.token_num_tokens > 0 and tokens_np.shape[1] != self.token_num_tokens:
            raise ValueError(f'{token_path}: tokens T must be {self.token_num_tokens}, got {tokens_np.shape[1]}')
        if tokens_np.shape[2] != self.token_dim:
            raise ValueError(f'{token_path}: tokens D must be {self.token_dim}, got {tokens_np.shape[2]}')
        tokens = torch.from_numpy(np.ascontiguousarray(tokens_np)).float()  # [V, T, D]
        if self.token_condition_norm == 'layer_norm':
            tokens = F.layer_norm(tokens, tokens.shape[-1:])
        tokens = self._select_views(tokens, sample)

        # Flatten to [num_views*T, D] for cross-attn
        T, D = tokens.shape[1], tokens.shape[2]
        cond = tokens.reshape(-1, D)            # [num_views*T, D]

        return {
            'coords': coords,
            'feats': feats,
            'cond': cond,
        }
