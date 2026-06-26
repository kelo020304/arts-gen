"""PartFlowDataset: dense-64^3 surface part labeling dataset (manifest-driven).

The Part Flow training task is **surface part labeling on a dense 64^3 grid**:
given a partial surface CONDITION + RGB cond + 2D mask part tokens, predict the
supervision slot of every surface voxel (with empty/0 elsewhere). Surface here
INCLUDES inner walls — e.g. drawer interiors that were missing from TRELLIS's
original outer-shell-only voxelization, but present in the data team's revised
``ind_<part>_*.npy`` and ``surface.npy`` outputs.

Spec: ``docs/specs/data/part_labels_surface_64.md`` (v2; renamed from v1
``part_labels_solid_64.md``). The contract is surface (含内壁), NOT a filled
solid volume.
Empty interior voxels are correctly labeled 0 in supervision and 0 in
is_on_surface — the model learns "where surface is" jointly with "what part each
surface voxel belongs to", with surface_dropout providing the completion signal.

Per-sample contract (after manifest-driven enumeration):

- ``coords``: dense ``[262144, 3]`` voxel coordinates (every (x,y,z) ∈ [0,64)³).
- ``is_on_surface``: ``[262144]`` int64 binary CONDITION from
  ``voxel_expanded/{obj}/angle_N/64/surface.npy``, with surface dropout applied.
- ``per_voxel_labels``: ``[262144]`` int64 supervision, online-synthesized from
  ``ind_<target>_*.npy`` (target slots) and ``surface − union(target ind)``
  (body slot). Values:
    - ``0`` = empty (off-surface or no part)
    - ``1..K_target`` = target slots (manifest order)
    - ``K_target + 1`` = body (all non-target part_info parts collapsed here)
    - ``-1`` = overlap ignore (multiple target ind voxels collide)
- ``num_parts``: ``K_target + 2`` (empty + targets + body).
- ``cond``: ``[V_sel * T, D]`` DINOv2 tokens for the manifest's ``view_indices``
  (deterministic 4 views, 1 per quadrant).
- ``mask_token_labels``: ``[V_sel * T]`` per-token supervision slot, remapped
  from raw mask labels via ``raw_to_slot`` (target raw → its slot, non-target
  raw → body slot, bg → 0).

The manifest_path config key is **required**. There is no directory-scan
fallback — target_part_names + view_indices are manifest-only contracts.
Missing masks, missing target ind files, unknown raw labels, and out-of-range
view_indices all fail loud.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional

import numpy as np
import torch

from trellis.utils.arts.mask_utils import patch_aggregate_foreground_wins
from trellis.datasets.arts.part_predictor import PartPredictorDataset


__all__ = ['PartFlowDataset', '_dense_coords']


_DENSE_COORDS_CACHE: Optional[torch.Tensor] = None


def _dense_coords(resolution: int = 64) -> torch.Tensor:
    """Return cached dense ``[resolution^3, 3]`` int32 XYZ grid."""
    global _DENSE_COORDS_CACHE
    assert resolution == 64, 'Phase 8 Part Flow is fixed at 64^3'
    if _DENSE_COORDS_CACHE is None:
        xs = torch.arange(resolution, dtype=torch.int32)
        gx, gy, gz = torch.meshgrid(xs, xs, xs, indexing='ij')
        _DENSE_COORDS_CACHE = torch.stack(
            [gx.reshape(-1), gy.reshape(-1), gz.reshape(-1)], dim=-1,
        )
    return _DENSE_COORDS_CACHE


class PartFlowDataset(PartPredictorDataset):
    """Dense 64^3 Part Flow training dataset.

    Args via ``data_config``:
        resolution: must be 64.
        surface_dropout_min/max: per-call dropout range for on-surface voxels.
        num_samples: optional cap for smoke runs.
    """

    def __init__(self, data_config: dict):
        super().__init__(data_config)
        self.resolution = int(data_config.get('resolution', 64))
        assert self.resolution == 64, 'Phase 8 Part Flow is fixed at 64^3'
        self.surface_dropout_min = float(data_config.get('surface_dropout_min', 0.05))
        self.surface_dropout_max = float(data_config.get('surface_dropout_max', 0.20))
        assert 0.0 <= self.surface_dropout_min <= self.surface_dropout_max < 1.0

        # When False (default), missing mask_root / mask_dir / per-view mask_*.npy
        # raises FileNotFoundError. Set True only for explicit no-mask ablation
        # runs where you accept that part_token pooling collapses to slot_emb /
        # empty_token fallbacks (model trains but VLM/mask conditioning is gone).
        self.allow_missing_masks = bool(data_config.get('allow_missing_masks', False))

        # Validate every manifest row's part_info upfront. Manifest is the
        # single source of truth for which samples enter training; if a
        # part_info.json is missing / empty / malformed, that's a data-contract
        # error on the manifest side and must fail loud. We do NOT silently
        # drop rows here — earlier revisions had a `num_parts <= 0: continue`
        # filter that violated the manifest-only principle.
        manifest_full = os.path.join(self.data_root, self.manifest_path)
        info_cache: Dict[str, int] = {}
        for meta in self.samples:
            obj_id = meta['obj_id']
            sample_id = meta.get('sample_id', '<unknown>')
            if obj_id in info_cache:
                continue
            info_path = os.path.join(
                self.recon_root, 'part_info', obj_id, 'part_info.json',
            )
            if not os.path.isfile(info_path):
                raise FileNotFoundError(
                    f'manifest row references missing part_info: '
                    f'sample_id={sample_id} obj_id={obj_id} '
                    f'expected={info_path} manifest={manifest_full}'
                )
            with open(info_path, 'r') as f:
                info = json.load(f)

            # num_parts: must be present, must coerce cleanly to a positive int.
            # Wrap int() to attach manifest context to the error — Python's
            # default ValueError says "invalid literal for int() with base 10"
            # which doesn't tell you which sample / manifest broke.
            if 'num_parts' not in info:
                raise KeyError(
                    f'part_info missing required key "num_parts": '
                    f'sample_id={sample_id} obj_id={obj_id} '
                    f'info_path={info_path} manifest={manifest_full}'
                )
            try:
                num_parts = int(info['num_parts'])
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f'part_info.num_parts must coerce to int: '
                    f'sample_id={sample_id} obj_id={obj_id} '
                    f'num_parts={info["num_parts"]!r} (type={type(info["num_parts"]).__name__}) '
                    f'info_path={info_path} manifest={manifest_full}'
                ) from exc
            if num_parts <= 0:
                raise ValueError(
                    f'part_info.num_parts must be >= 1: '
                    f'sample_id={sample_id} obj_id={obj_id} '
                    f'num_parts={num_parts} info_path={info_path} '
                    f'manifest={manifest_full}'
                )

            # parts: must be a dict (not list / scalar / None / missing).
            # Earlier check `if not parts_dict` only caught missing-or-empty
            # for dicts; a non-empty list would slip through and fail later in
            # __getitem__ at parts.values(), far from the data-contract error.
            if 'parts' not in info:
                raise KeyError(
                    f'part_info missing required key "parts": '
                    f'sample_id={sample_id} obj_id={obj_id} '
                    f'info_path={info_path} manifest={manifest_full}'
                )
            parts_dict = info['parts']
            if not isinstance(parts_dict, dict):
                raise TypeError(
                    f'part_info.parts must be a dict (str -> meta): '
                    f'sample_id={sample_id} obj_id={obj_id} '
                    f'got_type={type(parts_dict).__name__} '
                    f'info_path={info_path} manifest={manifest_full}'
                )
            if not parts_dict:
                raise ValueError(
                    f'part_info.parts is empty: '
                    f'sample_id={sample_id} obj_id={obj_id} '
                    f'info_path={info_path} manifest={manifest_full}'
                )

            info_cache[obj_id] = num_parts

        # Optional explicit debug cap (e.g. data.num_samples=4 in CLI override).
        # NOT a silent filter — user must opt in. For production training leave
        # it null so every manifest row reaches the DataLoader.
        num_samples_cap = data_config.get('num_samples', None)
        if num_samples_cap is not None:
            self.samples = self.samples[: int(num_samples_cap)]

        print(
            f'[PartFlowDataset/Phase8] {len(self.samples)} samples from manifest '
            f'(dense {self.resolution ** 3}-voxel enumeration)',
        )

    # ------------------------------------------------------------------ #
    # Sample enumeration (manifest-driven, jsonl)                         #
    # ------------------------------------------------------------------ #

    def _enumerate_samples(self) -> List[Dict[str, Any]]:
        """Manifest-driven enumeration. Reads jsonl, returns list of dicts.

        Each dict has:
            obj_id              : str
            angle_idx           : int
            sample_id           : str
            target_part_names   : list[str]   (manifest target_part_names, in slot order)
            view_indices        : list[int]   (4 view ids, 1 per quadrant)

        Phase 9 contract change vs parent (PartPredictorDataset): parent emits
        list[tuple[str, int]] discovered via part_labels/{oid}/angle_N/part_labels_64.npy
        existence (legacy part_predictor task). PartFlow uses surface.npy +
        per-target ind_*.npy, and target slot order comes from the manifest, so
        we can't fall back to directory scanning — manifest_path is required.
        """
        if not self.manifest_path:
            raise ValueError(
                'PartFlowDataset requires data_config["manifest_path"] (jsonl). '
                'Directory-scan fallback is removed because target_part_names + '
                'view_indices are manifest-only contracts.'
            )
        manifest_full = os.path.join(self.data_root, self.manifest_path)
        if not os.path.isfile(manifest_full):
            raise FileNotFoundError(f'Manifest not found: {manifest_full}')

        samples: List[Dict[str, Any]] = []
        with open(manifest_full, 'r') as f:
            for line_no, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                # Required fields, fail loudly on schema drift.
                for key in ('object_id', 'angle_idx', 'target_part_names',
                            'view_indices', 'sample_id'):
                    if key not in rec:
                        raise KeyError(
                            f'{manifest_full}:{line_no} missing field {key!r}'
                        )
                target_part_names = list(rec['target_part_names'])
                view_indices = [int(v) for v in rec['view_indices']]

                # Manifest field validation — fail loud, not at __getitem__ time.
                if not target_part_names:
                    raise ValueError(
                        f'{manifest_full}:{line_no} target_part_names is empty'
                    )
                if len(set(target_part_names)) != len(target_part_names):
                    raise ValueError(
                        f'{manifest_full}:{line_no} target_part_names contains duplicates: '
                        f'{target_part_names}'
                    )
                if len(view_indices) != 4:
                    raise ValueError(
                        f'{manifest_full}:{line_no} view_indices must have length 4, '
                        f'got {len(view_indices)}: {view_indices}'
                    )
                if len(set(view_indices)) != len(view_indices):
                    raise ValueError(
                        f'{manifest_full}:{line_no} view_indices has duplicates: {view_indices}'
                    )
                if any(v < 0 for v in view_indices):
                    raise ValueError(
                        f'{manifest_full}:{line_no} view_indices has negative values: {view_indices}'
                    )

                samples.append({
                    'obj_id': str(rec['object_id']),
                    'angle_idx': int(rec['angle_idx']),
                    'sample_id': str(rec['sample_id']),
                    'target_part_names': target_part_names,
                    'view_indices': view_indices,
                })
        if not samples:
            raise RuntimeError(f'Manifest {manifest_full} produced 0 samples')
        return samples

    # ------------------------------------------------------------------ #
    # Mask 2D -> patch grid (foreground-wins voting)                       #
    # ------------------------------------------------------------------ #

    def _downsample_mask(self, mask_2d_np: np.ndarray, patch_grid: int) -> torch.Tensor:
        """D-22/D-23 foreground-wins mask downsample (every non-zero raw label
        treated as foreground; majority-vote among foreground pixels per patch
        with min_fg=3 minimum coverage). Bg stays 0."""
        mask_t = torch.from_numpy(mask_2d_np.astype(np.int64))
        return patch_aggregate_foreground_wins(
            mask_t, grid=patch_grid, patch=14, min_fg=3,
        )

    @staticmethod
    def _apply_surface_dropout(
        is_on_surface: torch.Tensor,
        dropout_min: float,
        dropout_max: float,
    ) -> torch.Tensor:
        if dropout_max <= 0.0:
            return is_on_surface
        rate = float(np.random.uniform(dropout_min, dropout_max))
        surf_idx = torch.nonzero(is_on_surface, as_tuple=False).squeeze(-1)
        n_drop = int(len(surf_idx) * rate)
        if n_drop > 0:
            perm = torch.randperm(len(surf_idx))[:n_drop]
            is_on_surface = is_on_surface.clone()
            is_on_surface[surf_idx[perm]] = 0
        return is_on_surface

    # ------------------------------------------------------------------ #
    # Sample and collate                                                  #
    # ------------------------------------------------------------------ #

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        meta = self.samples[idx]
        obj_id = meta['obj_id']
        angle_idx = meta['angle_idx']
        angle_dir = f'angle_{angle_idx}'
        target_part_names: List[str] = meta['target_part_names']
        view_indices: List[int] = meta['view_indices']

        # ---- Load part_info ----
        info_path = os.path.join(self.recon_root, 'part_info', obj_id, 'part_info.json')
        with open(info_path, 'r') as f:
            part_info = json.load(f)
        K_real = int(part_info['num_parts'])
        parts_dict = part_info['parts']
        assert len(parts_dict) == K_real, (
            f'{obj_id}: part_info parts count {len(parts_dict)} != num_parts {K_real}'
        )

        # part_info raw labels MUST be contiguous 1..K_real (enforced by
        # 04_voxelize.load_part_specs:expected_label = part_index + 1). We rely
        # on this invariant when building raw_to_slot — fail loud if upstream
        # data drifts.
        observed_labels = sorted(int(p['label']) for p in parts_dict.values())
        if observed_labels != list(range(1, K_real + 1)):
            raise ValueError(
                f'{obj_id}: part_info raw labels must be contiguous 1..{K_real}, '
                f'got {observed_labels}'
            )

        # target_part_names from manifest must all exist in part_info.parts.
        unknown_targets = [n for n in target_part_names if n not in parts_dict]
        if unknown_targets:
            raise KeyError(
                f'{obj_id}: target_part_names {unknown_targets} not in '
                f'part_info["parts"] (known: {sorted(parts_dict.keys())})'
            )

        # ---- Build supervision slot mappings ----
        # slot 0 = empty / bg
        # slot 1..K_target = target parts (in manifest order)
        # slot K_target+1 = body (all non-target part_info parts collapsed here)
        target_to_slot: Dict[str, int] = {
            name: i + 1 for i, name in enumerate(target_part_names)
        }
        K_target = len(target_to_slot)
        assert K_target >= 1, f'{obj_id}: empty target_part_names'
        body_slot = K_target + 1
        num_parts_phase8 = K_target + 2

        # raw mask label (= part_info.parts[name].label) -> supervision slot
        raw_to_slot: Dict[int, int] = {0: 0}
        for name, pmeta in parts_dict.items():
            raw = int(pmeta['label'])
            if name in target_to_slot:
                raw_to_slot[raw] = target_to_slot[name]
            else:
                raw_to_slot[raw] = body_slot

        # ---- Load surface (CONDITION) and per-target ind (TARGET supervision) ----
        voxel_dir = os.path.join(
            self.recon_root, 'voxel_expanded', obj_id, angle_dir, '64',
        )
        surface_path = os.path.join(voxel_dir, 'surface.npy')
        if not os.path.isfile(surface_path):
            raise FileNotFoundError(
                f'{surface_path} missing — needed for is_on_surface CONDITION'
            )
        surface = np.load(surface_path)
        assert surface.ndim == 2 and surface.shape[1] == 3, \
            f'{surface_path} expected [M,3], got {surface.shape}'
        assert (surface >= 0).all() and (surface < self.resolution).all(), \
            f'{surface_path} has out-of-bounds indices'

        target_ind_arrays: List[np.ndarray] = []
        per_voxel_labels = np.zeros((self.resolution,) * 3, dtype=np.int64)
        overlap_mask = np.zeros_like(per_voxel_labels, dtype=bool)
        for name in target_part_names:
            ind_path = os.path.join(voxel_dir, f'ind_{name}.npy')
            if not os.path.isfile(ind_path):
                raise FileNotFoundError(
                    f'{ind_path} missing — required for target slot '
                    f'{target_to_slot[name]} (target_part {name})'
                )
            ind = np.load(ind_path)
            if ind.size == 0:
                continue
            assert ind.ndim == 2 and ind.shape[1] == 3, \
                f'{ind_path} expected [M,3], got {ind.shape}'
            assert (ind >= 0).all() and (ind < self.resolution).all(), \
                f'{ind_path} has out-of-bounds indices'
            slot = target_to_slot[name]
            existing = per_voxel_labels[ind[:, 0], ind[:, 1], ind[:, 2]] != 0
            overlap_mask[ind[existing, 0], ind[existing, 1], ind[existing, 2]] = True
            per_voxel_labels[ind[:, 0], ind[:, 1], ind[:, 2]] = slot
            target_ind_arrays.append(ind)

        # ---- Body slot voxels = surface − union(target ind) ----
        if target_ind_arrays:
            target_union = np.unique(np.vstack(target_ind_arrays), axis=0)
            target_set = set(map(tuple, target_union.tolist()))
            body_voxels = np.array(
                [p for p in surface.tolist() if tuple(p) not in target_set],
                dtype=np.int64,
            )
        else:
            body_voxels = surface.astype(np.int64).copy()
        if body_voxels.size:
            # By construction body voxels ⊆ surface, target ind ⊆ surface, and
            # body = surface − target_union, so they should not overlap. Guard
            # against contract drift loudly.
            existing = per_voxel_labels[
                body_voxels[:, 0], body_voxels[:, 1], body_voxels[:, 2]
            ]
            assert (existing == 0).all(), (
                f'{obj_id}/{angle_dir}: body voxel collides with target ind '
                f'(surface ⊉ union(target_ind))'
            )
            per_voxel_labels[
                body_voxels[:, 0], body_voxels[:, 1], body_voxels[:, 2]
            ] = body_slot
        per_voxel_labels[overlap_mask] = -1

        per_voxel_labels_t = torch.from_numpy(per_voxel_labels).reshape(-1).long()
        assert per_voxel_labels_t.min().item() >= -1
        assert per_voxel_labels_t.max().item() < num_parts_phase8, (
            f'{obj_id}/{angle_dir}: per_voxel_labels max '
            f'{per_voxel_labels_t.max().item()} >= num_parts_phase8 {num_parts_phase8}'
        )

        # ---- is_on_surface CONDITION (with surface dropout) ----
        is_on_surface_vol = np.zeros((self.resolution,) * 3, dtype=np.int64)
        is_on_surface_vol[surface[:, 0], surface[:, 1], surface[:, 2]] = 1
        is_on_surface = torch.from_numpy(is_on_surface_vol).reshape(-1).long()
        is_on_surface = self._apply_surface_dropout(
            is_on_surface, self.surface_dropout_min, self.surface_dropout_max,
        )

        # ---- DINOv2 cond (only manifest views) ----
        tokens_path = os.path.join(
            self.recon_root, 'dinov2_tokens', obj_id, angle_dir, 'tokens.npz',
        )
        tokens_data = np.load(tokens_path)
        tokens = torch.from_numpy(tokens_data['tokens']).float()  # [12, T, D]
        if tokens.dim() != 3:
            raise ValueError(
                f'{tokens_path} expected [V,T,D], got {tuple(tokens.shape)}'
            )
        V_total, T, D = tokens.shape
        max_view = max(view_indices)
        if max_view >= V_total:
            raise ValueError(
                f'{obj_id}/{angle_dir}: manifest view_indices max={max_view} '
                f'>= dinov2 V_total={V_total}'
            )
        tokens = tokens[view_indices]                          # [V_sel, T, D]
        cond = tokens.reshape(-1, D)                           # [V_sel*T, D]

        # ---- mask_token_labels (remap raw -> supervision slot, body merge) ----
        mask_token_labels = self._build_mask_token_labels_remapped(
            obj_id, angle_dir, view_indices, T, raw_to_slot,
        )                                                       # [V_sel, T]
        mask_token_labels = mask_token_labels.reshape(-1).long()

        # ---- Coords (dense 64^3) ----
        coords = _dense_coords(self.resolution).clone()         # [64^3, 3] int32

        return {
            'coords': coords,
            'per_voxel_labels': per_voxel_labels_t,
            'is_on_surface': is_on_surface,
            'cond': cond,
            'mask_token_labels': mask_token_labels,
            'num_parts': num_parts_phase8,
            'part_info': part_info,
            'obj_id': obj_id,
            'angle_idx': angle_idx,
            'sample_id': meta['sample_id'],
            'target_part_names': target_part_names,
            'view_indices': view_indices,
        }

    # ------------------------------------------------------------------ #
    # Mask token labels with body-slot remap                              #
    # ------------------------------------------------------------------ #

    def _build_mask_token_labels_remapped(
        self,
        obj_id: str,
        angle_dir: str,
        view_indices: List[int],
        T: int,
        raw_to_slot: Dict[int, int],
    ) -> torch.Tensor:
        """Build [V_sel, T] mask_token_labels, remapping raw labels to supervision slots.

        Convention preserved: index 0 = CLS token (always 0); indices 1..T-1 =
        patch tokens populated from foreground-wins downsampled mask.

        Fail-loud policy: in the default ``allow_missing_masks=False`` mode,
        ``mask_subdir`` must be configured, the per-object mask dir must exist,
        and every per-view ``mask_{v}.npy`` referenced by manifest view_indices
        must exist. Silent zeros would collapse part_token pooling to slot_emb /
        empty_token fallbacks and silently kill the VLM/mask conditioning signal
        — exactly the kind of bug-masking AGENTS.md forbids. Set
        ``data.allow_missing_masks: true`` only for explicit no-mask ablation.
        """
        V = len(view_indices)
        out = torch.zeros(V, T, dtype=torch.int64)

        if self.mask_root is None:
            if self.allow_missing_masks:
                return out
            raise ValueError(
                'PartFlowDataset: mask_subdir is required (mask_root is None). '
                'Set data.mask_subdir to the renders subdir, or explicitly opt '
                'into the no-mask ablation with data.allow_missing_masks: true.'
            )
        mask_dir = os.path.join(self.mask_root, obj_id, angle_dir, 'mask')
        if not os.path.isdir(mask_dir):
            if self.allow_missing_masks:
                return out
            raise FileNotFoundError(
                f'mask dir missing: {mask_dir}. Manifest specified view_indices '
                f'for {obj_id}/{angle_dir} but the mask directory is absent. Set '
                f'data.allow_missing_masks: true to accept all-zero mask tokens.'
            )

        patch_grid = int(round((T - 1) ** 0.5))

        for slot_idx, v in enumerate(view_indices):
            mask_path = os.path.join(mask_dir, f'mask_{v}.npy')
            if not os.path.isfile(mask_path):
                if self.allow_missing_masks:
                    continue
                raise FileNotFoundError(
                    f'{mask_path} missing — manifest view_{v} expects a mask file. '
                    f'Set data.allow_missing_masks: true to accept zero mask tokens '
                    f'for this view.'
                )
            mask_2d = np.load(mask_path)
            patch_mask = self._downsample_mask(mask_2d, patch_grid).long()  # [patch_grid, patch_grid]

            # Remap raw mask labels to supervision slots in one pass.
            remapped = torch.zeros_like(patch_mask)
            seen_raw = set(int(u) for u in torch.unique(patch_mask).tolist())
            for raw in seen_raw:
                if raw not in raw_to_slot:
                    raise KeyError(
                        f'{mask_path}: raw label {raw} not in part_info '
                        f'(known: {sorted(raw_to_slot.keys())})'
                    )
                remapped[patch_mask == raw] = raw_to_slot[raw]
            out[slot_idx, 1:] = remapped.flatten()

        return out

    @staticmethod
    def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        coords_list: List[torch.Tensor] = []
        voxel_layout: List[slice] = []
        start = 0

        for b_idx, sample in enumerate(batch):
            coords = sample['coords'].int()
            N_b = coords.shape[0]
            assert coords.shape == (64 ** 3, 3), f'batch[{b_idx}] coords {coords.shape}'
            bcol = torch.full((N_b, 1), b_idx, dtype=torch.int32)
            coords_list.append(torch.cat([bcol, coords], dim=1))
            voxel_layout.append(slice(start, start + N_b))
            start += N_b

        return {
            'coords': torch.cat(coords_list, dim=0),
            'per_voxel_labels': torch.cat(
                [sample['per_voxel_labels'].long() for sample in batch], dim=0,
            ),
            'is_on_surface': torch.cat(
                [sample['is_on_surface'].long() for sample in batch], dim=0,
            ),
            'cond': torch.stack([sample['cond'] for sample in batch], dim=0),
            'mask_token_labels': torch.stack(
                [sample['mask_token_labels'].long() for sample in batch], dim=0,
            ),
            'num_parts': [int(sample['num_parts']) for sample in batch],
            'voxel_layout': voxel_layout,
            'obj_id': [sample['obj_id'] for sample in batch],
            'angle_idx': [sample['angle_idx'] for sample in batch],
            'part_info': [sample['part_info'] for sample in batch],
            # Manifest-driven debug metadata (preserved through DataLoader for
            # per-sample logging / inspection / failure attribution). Hard-index
            # access — these keys are part of __getitem__'s contract; missing
            # them means the contract is broken and we want to fail loud here,
            # not silently insert None.
            'sample_id': [sample['sample_id'] for sample in batch],
            'target_part_names': [sample['target_part_names'] for sample in batch],
            'view_indices': [sample['view_indices'] for sample in batch],
        }
