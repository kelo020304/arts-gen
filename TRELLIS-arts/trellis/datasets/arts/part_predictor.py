"""
Part Predictor 数据集。

加载 part_labels + DINOv2 tokens + part_info：
    1. part_labels [64,64,64] -> active mask -> coords [N,3] + labels [N]
       (no per-voxel feats; model uses positional encoding of coords + DINOv2)
    2. DINOv2 tokens [V,T,D] -> view dropout -> flatten [V*T, D]
    3. part_info.json -> part_type_ids [K] + num_parts

独立 dataset 类（D-09），不继承 Phase 2 MvImageConditionedSLatDataset。

数据目录结构:
    {data_root}/{recon_subdir}/part_labels/{obj_id}/angle_{N}/part_labels_64.npy
    {data_root}/{recon_subdir}/dinov2_tokens/{obj_id}/angle_{N}/tokens.npz
    {data_root}/{recon_subdir}/part_info/{obj_id}/part_info.json
    (optional, decode-aware only) {data_root}/{recon_subdir}/{slat_dir}/{obj_id}/angle_{N}/latent.npz

Multi-batch (2026-04-15):
    Use PartPredictorDataset.collate_fn as DataLoader collate_fn to pack a list
    of per-sample dicts into a single batched dict (SLat-style packing):
        coords [N_total, 4] with col0=batch_idx,
        cond stacked [B, V*T, D], part_type_ids / num_parts / gt_points_per_part
        as Python lists, voxel_layout / query_layout as list[slice].
"""

import json
import os
import random
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


__all__ = ['PartPredictorDataset']


# Part type vocabulary (D-06): maps type string -> int index.
# Covers PartNet-Mobility common types + reserved "other" slot.
PART_TYPE_VOCAB = {
    'base_body': 0,
    'button': 1,
    'lid': 2,
    'door': 3,
    'drawer': 4,
    'handle': 5,
    'knob': 6,
    'lever': 7,
    'wheel': 8,
    'slider': 9,
    'hinge': 10,
    'rotation_part': 11,
    'switch': 12,
    'dial': 13,
    'pedal': 14,
    'shelf': 15,
    'mirror': 16,
    'screen': 17,
    'keyboard': 18,
    'leg': 19,
    'arm': 20,
    'seat': 21,
    'back': 22,
    'top': 23,
    'bottom': 24,
    'side': 25,
    'front': 26,
    'cover': 27,
    'panel': 28,
    'frame': 29,
    'rod': 30,
}
# Reserved index for unknown types
PART_TYPE_OTHER = 31


def _type_str_to_idx(type_str: str) -> int:
    """Map a part type string to vocabulary index.

    Strips trailing '_N' suffixes (e.g., 'lid_0' -> 'lid') before lookup.
    Unknown types map to PART_TYPE_OTHER (31).
    """
    # part_info keys are like 'lid_0', 'base_body_0'; type field is clean
    clean = type_str.lower().strip()
    return PART_TYPE_VOCAB.get(clean, PART_TYPE_OTHER)


class PartPredictorDataset(Dataset):
    """Part Predictor 训练/验证数据集。

    每个样本返回 dict:
        coords: [N, 3] int32  — 64-cube active voxel 坐标
        feats: [N, 8] float32 — trilinear 上采样后的 SS latent 特征
        part_labels: [N] int64 — 0-indexed part label (0..K-1)
        cond: [V*T, D] float32 — DINOv2 tokens (view dropout applied)
        part_type_ids: [K] int64 — part type vocabulary indices
        num_parts: int — K
        gt_points_per_part: List[Tensor] — per-part GT point clouds [P_k, 3]
        obj_id: str
        angle_idx: int
        slat_coords: [N_slat, 3] int or None — SLat voxel coords (decode-aware)
        slat_feats: [N_slat, 8] float32 or None — SLat voxel feats (decode-aware)

    Args:
        data_config: dict with keys:
            data_root, recon_subdir, manifest_path, test_obj_ids,
            num_angles, num_views, load_slat, slat_dir
    """

    def __init__(self, data_config: dict):
        super().__init__()
        self.data_root = data_config['data_root']
        self.recon_subdir = data_config.get('recon_subdir', 'reconstruction')
        self.manifest_path = data_config.get('manifest_path', None)
        self.test_obj_ids = data_config.get('test_obj_ids', None)
        self.num_angles = data_config.get('num_angles', 10)
        self.num_views = data_config.get('num_views', 4)
        self.load_slat = data_config.get('load_slat', False)
        self.slat_dir = data_config.get('slat_dir', 'slat_latents_expanded')
        # VLM mask directory: sibling of recon_subdir (e.g. "arts/renders")
        # At training time, uses GT rendered masks; at inference, VLM masks.
        self.mask_subdir = data_config.get('mask_subdir', None)
        # Max part id for mask_embed lookup — must match model.max_k in YAML so
        # high-id parts aren't silently folded. 128 matches base.yaml default.
        self.max_k = int(data_config.get('max_k', 128))

        self.recon_root = os.path.join(self.data_root, self.recon_subdir)
        if self.mask_subdir:
            self.mask_root = os.path.join(self.data_root, self.mask_subdir)
        else:
            self.mask_root = None
        self.samples = self._enumerate_samples()

    def _enumerate_samples(self) -> List[Tuple[str, int]]:
        """Build list of (obj_id, angle_idx) tuples.

        Priority:
            1. manifest_path (JSON) if provided and exists
            2. test_obj_ids whitelist + directory enumeration
            3. Full directory enumeration from part_info/
        """
        samples = []

        # Try manifest first (D-12: reuse Phase 2/3 split)
        if self.manifest_path:
            manifest_full = os.path.join(self.data_root, self.manifest_path)
            if os.path.isfile(manifest_full):
                with open(manifest_full, 'r') as f:
                    manifest = json.load(f)
                # Support assembler format: {"samples": [...]}
                if isinstance(manifest, dict) and 'samples' in manifest:
                    for s in manifest['samples']:
                        obj_id = str(s.get('object_id', s.get('id', '')))
                        angle_idx = s.get('angle_idx', 0)
                        samples.append((obj_id, angle_idx))
                # Support old format: [{"id": ..., "angles": [...]}]
                elif isinstance(manifest, list):
                    for entry in manifest:
                        obj_id = str(entry.get('id', entry.get('object_id', '')))
                        angles = entry.get('angles', list(range(self.num_angles)))
                        for a in angles:
                            samples.append((obj_id, a))
                if samples:
                    return samples

        # Fallback: enumerate from part_info/ directory
        part_info_dir = os.path.join(self.recon_root, 'part_info')
        if not os.path.isdir(part_info_dir):
            return samples

        obj_ids = sorted(os.listdir(part_info_dir))
        if self.test_obj_ids:
            obj_ids = [o for o in obj_ids if o in self.test_obj_ids]

        for obj_id in obj_ids:
            info_path = os.path.join(part_info_dir, obj_id, 'part_info.json')
            if not os.path.isfile(info_path):
                continue
            for angle_idx in range(self.num_angles):
                # Check if part_labels exists for this angle (the actual input)
                labels_path = os.path.join(
                    self.recon_root, 'part_labels', obj_id,
                    f'angle_{angle_idx}', 'part_labels_64.npy',
                )
                if os.path.isfile(labels_path):
                    samples.append((obj_id, angle_idx))

        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def _compute_mask_token_labels(
        self,
        V_total: int,
        T: int,
        obj_id: str,
        angle_dir: str,
        label_to_idx: dict,
        selected_views: list,
        max_k: int = 128,
    ) -> torch.Tensor:
        """Compute per-token mask labels for the selected views only.

        For each selected view, downsample the 2D mask [H,W] to the DINOv2
        patch grid (37×37) and remap original part labels to 0-indexed (1..K).
        CLS token position gets label 0 (background).

        Internally builds [V_total, T] then slices to [V_selected, T] to align
        with the gathered cond layout in __getitem__.

        Returns:
            mask_token_labels: [V_selected, T] int64, values in 0..K
                0 = background / CLS
                1..K = 0-indexed part id + 1
        """
        mask_token_labels = torch.zeros(V_total, T, dtype=torch.int64)

        if self.mask_root is None:
            return mask_token_labels[selected_views]

        mask_dir = os.path.join(self.mask_root, obj_id, angle_dir, 'mask')
        if not os.path.isdir(mask_dir):
            return mask_token_labels[selected_views]

        patch_grid = int(round((T - 1) ** 0.5))  # 37

        for v in selected_views:
            mask_path = os.path.join(mask_dir, f'mask_{v}.npy')
            if not os.path.isfile(mask_path):
                continue
            mask_2d = np.load(mask_path)  # [H, W] int32, 0=bg, label values from part_info

            # Downsample to patch grid via nearest interpolation
            mask_t = torch.from_numpy(mask_2d.astype(np.float32))
            patch_mask = F.interpolate(
                mask_t.unsqueeze(0).unsqueeze(0),
                size=(patch_grid, patch_grid), mode='nearest',
            ).squeeze().long()  # [37, 37] original label values

            # Remap original labels to 0-indexed: 0=bg, 1..K = part idx + 1
            remapped = torch.zeros_like(patch_mask)
            for orig_label, new_idx in label_to_idx.items():
                remapped[patch_mask == orig_label] = new_idx + 1  # +1 so 0 stays bg

            # Clamp to max_k
            remapped = remapped.clamp(max=max_k)

            # CLS token (position 0) = 0, spatial tokens (positions 1..T-1)
            mask_token_labels[v, 1:] = remapped.flatten()  # [1369]

        # Slice to selected views only, aligning with gathered cond layout
        return mask_token_labels[selected_views]  # [V_selected, T]

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        obj_id, angle_idx = self.samples[idx]
        angle_dir = f'angle_{angle_idx}'

        # --- 1. Load part labels (defines active voxel positions) ---
        labels_path = os.path.join(
            self.recon_root, 'part_labels', obj_id, angle_dir, 'part_labels_64.npy',
        )
        part_labels_dense = torch.from_numpy(
            np.load(labels_path).astype(np.int64),
        )  # [64, 64, 64]

        # --- 2. Active mask: exclude background (0) and overlap/ignore (-1) ---
        active_mask = (part_labels_dense != 0) & (part_labels_dense != -1)
        # [64, 64, 64] bool

        # Extract coords [N, 3], labels [N]. No per-voxel feats: model uses
        # positional encoding of coords + DINOv2 cross-attention only.
        active_indices = torch.nonzero(active_mask, as_tuple=False)  # [N, 3]
        coords = active_indices.int()  # [N, 3] int32

        x, y, z = active_indices[:, 0], active_indices[:, 1], active_indices[:, 2]
        raw_labels = part_labels_dense[x, y, z]  # [N] original label values (1, 2, ...)

        # --- 4. Load part_info.json ---
        info_path = os.path.join(
            self.recon_root, 'part_info', obj_id, 'part_info.json',
        )
        with open(info_path, 'r') as f:
            part_info = json.load(f)

        num_parts = part_info['num_parts']
        parts_dict = part_info['parts']  # {name: {label, type, ...}}

        # Build label -> (0-indexed, type_idx) mapping
        # Sort by original label value for deterministic ordering
        sorted_parts = sorted(
            parts_dict.values(), key=lambda p: p['label'],
        )
        label_to_idx = {}  # original label -> 0-indexed
        part_type_ids_list = []
        for new_idx, part in enumerate(sorted_parts):
            original_label = part['label']
            label_to_idx[original_label] = new_idx
            part_type_ids_list.append(_type_str_to_idx(part['type']))

        # Remap labels to contiguous 0-indexed
        part_labels = torch.zeros_like(raw_labels)
        for orig_label, new_idx in label_to_idx.items():
            part_labels[raw_labels == orig_label] = new_idx
        # [N] int64, values in 0..K-1

        part_type_ids = torch.tensor(part_type_ids_list, dtype=torch.int64)  # [K]

        # --- 5. Load DINOv2 tokens ---
        tokens_path = os.path.join(
            self.recon_root, 'dinov2_tokens', obj_id, angle_dir, 'tokens.npz',
        )
        tokens_data = np.load(tokens_path)
        tokens = torch.from_numpy(tokens_data['tokens']).float()  # [V, T, D]
        V_total, T, D = tokens.shape

        # View sampling: 12 views = 4 quadrants × 3 elevations.
        # Sample 1 view per quadrant for balanced 360° coverage.
        QUADRANTS = [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]]
        num_quads = min(len(QUADRANTS), (V_total + 2) // 3)  # handle V<12
        selected_views = sorted(
            random.choice(QUADRANTS[q]) for q in range(num_quads)
        )

        # Keep only the selected views (drop the other 8 entirely, no padding).
        # Previously the unselected views were zeroed but still flattened in —
        # 75% of cond tokens carried no signal, and the zero vectors still got
        # a non-zero cond_proj bias, polluting rgb cross-attn. This also
        # aligns train distribution with infer.py, which already slices.
        tokens = tokens[selected_views]  # [V_selected, T, D], e.g. [4, 1370, 1024]

        # Flatten to [V_selected*T, D]
        cond = tokens.reshape(-1, D)

        # --- 5b. Per-token mask labels for mask-in-KV ---
        # Downsample 2D masks to DINOv2 patch grid, remap to 0-indexed part ids.
        # Shape matches cond: [V_selected, T] then flatten to [V_selected*T].
        # Pass self.max_k so the clamp aligns with model.max_k in YAML
        # (previously hardcoded 100 while base.yaml already used 128).
        mask_token_labels = self._compute_mask_token_labels(
            V_total, T, obj_id, angle_dir, label_to_idx, selected_views,
            max_k=self.max_k,
        )
        # Flatten to [V_selected*T] matching cond layout
        mask_token_labels = mask_token_labels.reshape(-1)

        # --- 6. GT points per part (for decode-aware loss) ---
        gt_points_per_part = []
        for k in range(num_parts):
            pts = coords[part_labels == k].float() / 64.0  # [P_k, 3] normalized to [0,1]
            gt_points_per_part.append(pts)

        # --- 7. Optional SLat loading (D-01 data dependency, None-guard) ---
        slat_coords = None
        slat_feats = None
        if self.load_slat:
            slat_path = os.path.join(
                self.recon_root, self.slat_dir, obj_id, angle_dir, 'latent.npz',
            )
            if os.path.isfile(slat_path):
                slat_data = np.load(slat_path)
                slat_coords = torch.from_numpy(slat_data['coords']).int()  # [N_slat, 3]
                slat_feats = torch.from_numpy(slat_data['feats']).float()  # [N_slat, 8]

        return {
            'coords': coords,               # [N, 3] int32
            'part_labels': part_labels,      # [N] int64 (0-indexed)
            'cond': cond,                    # [V*T, D] float32
            'mask_token_labels': mask_token_labels,  # [V*T] int64 (0=bg, 1..K=part)
            'part_type_ids': part_type_ids,  # [K] int64
            'num_parts': num_parts,          # int
            'gt_points_per_part': gt_points_per_part,  # List[Tensor [P_k, 3]]
            'obj_id': obj_id,
            'angle_idx': angle_idx,
            'slat_coords': slat_coords,      # [N_slat, 3] or None
            'slat_feats': slat_feats,        # [N_slat, 8] or None
        }

    @staticmethod
    def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Pack per-sample dicts into a batched dict (SLat-style).

        Follows `trellis.datasets.structured_latent.SLat.collate_fn`:
        coords get a batch_idx column prepended (col0), feats/labels
        concat along N, cond stacked to [B, V*T, D], variable-K fields
        kept as Python lists. SLat optional fields packed the same way
        when present in every sample; otherwise returned as None.

        Output dict keys:
            coords:            [N_total, 4] int32 (col0=batch_idx)
            part_labels:       [N_total] int64 (0-indexed, per-sample)
            cond:              [B, V*T, D] float32
            mask_token_labels: [B, V*T] int64 (0=bg, 1..K=part idx+1)
            part_type_ids:     List[LongTensor[K_b]]
            num_parts:         List[int]
            gt_points_per_part: List[List[Tensor[P_k, 3]]]
            obj_id:            List[str]
            angle_idx:         List[int]
            voxel_layout:      List[slice]  (row ranges of coords per sample)
            query_layout:      List[slice]  (cumulative ranges over K_b)
            slat_coords:       [N_slat_total, 4] int32 or None
            slat_feats:        [N_slat_total, 8] float32 or None
            slat_layout:       List[slice] or None
        """
        B = len(batch)
        coords_list: List[torch.Tensor] = []
        labels_list: List[torch.Tensor] = []
        cond_list: List[torch.Tensor] = []
        voxel_layout: List[slice] = []
        query_layout: List[slice] = []

        v_start = 0
        q_start = 0
        for b_idx, s in enumerate(batch):
            N_b = s['coords'].shape[0]
            bcol = torch.full((N_b, 1), b_idx, dtype=torch.int32)
            coords_list.append(torch.cat([bcol, s['coords'].int()], dim=1))  # [N_b, 4]
            labels_list.append(s['part_labels'])
            cond_list.append(s['cond'])
            voxel_layout.append(slice(v_start, v_start + N_b))
            v_start += N_b

            K_b = int(s['num_parts'])
            query_layout.append(slice(q_start, q_start + K_b))
            q_start += K_b

        coords = torch.cat(coords_list, dim=0)
        part_labels = torch.cat(labels_list, dim=0)
        cond = torch.stack(cond_list, dim=0)  # [B, V*T, D]
        mask_token_labels = torch.stack([s['mask_token_labels'] for s in batch], dim=0)  # [B, V*T]

        part_type_ids = [s['part_type_ids'] for s in batch]
        num_parts = [int(s['num_parts']) for s in batch]
        gt_points_per_part = [s['gt_points_per_part'] for s in batch]
        obj_id = [s['obj_id'] for s in batch]
        angle_idx = [s['angle_idx'] for s in batch]

        # SLat (optional): only pack if every sample has it
        slat_coords = None
        slat_feats = None
        slat_layout: Optional[List[slice]] = None
        if all(s.get('slat_coords') is not None and s.get('slat_feats') is not None for s in batch):
            sc_list, sf_list = [], []
            s_start = 0
            slat_layout = []
            for b_idx, s in enumerate(batch):
                sc = s['slat_coords'].int()  # [N_slat_b, 3]
                sf = s['slat_feats']
                N_s = sc.shape[0]
                bcol = torch.full((N_s, 1), b_idx, dtype=torch.int32)
                sc_list.append(torch.cat([bcol, sc], dim=1))  # [N_slat_b, 4]
                sf_list.append(sf)
                slat_layout.append(slice(s_start, s_start + N_s))
                s_start += N_s
            slat_coords = torch.cat(sc_list, dim=0)
            slat_feats = torch.cat(sf_list, dim=0)

        return {
            'coords': coords,
            'part_labels': part_labels,
            'cond': cond,
            'mask_token_labels': mask_token_labels,
            'part_type_ids': part_type_ids,
            'num_parts': num_parts,
            'gt_points_per_part': gt_points_per_part,
            'obj_id': obj_id,
            'angle_idx': angle_idx,
            'voxel_layout': voxel_layout,
            'query_layout': query_layout,
            'slat_coords': slat_coords,
            'slat_feats': slat_feats,
            'slat_layout': slat_layout,
        }
