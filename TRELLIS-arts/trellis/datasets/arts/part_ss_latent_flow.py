"""Object-level joint part SS latent flow dataset.

This dataset is manifest-driven. Each manifest row describes one object/angle
with K target parts; the dataset keeps that row as one training sample:

    z_global SS latent + DINOv2 tokens + K target part mask queries
        -> K target part SS latents

No dense 64^3 empty/body categorical labels are produced here.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
from torch.utils.data import Dataset

from trellis.utils.arts.mask_utils import patch_aggregate_foreground_wins


__all__ = ["PartSSLatentFlowDataset"]


class PartSSLatentFlowDataset(Dataset):
    """Manifest-driven object-level joint part SS latent flow dataset."""

    def __init__(self, data_config: dict):
        super().__init__()
        self.data_root = Path(data_config["data_root"])
        self.recon_subdir = data_config.get("recon_subdir", "reconstruction")
        self.mask_subdir = data_config.get("mask_subdir", "renders")
        self.manifest_path = data_config["manifest_path"]
        self.num_views = int(data_config.get("num_views", 4))
        self.allow_missing_masks = bool(data_config.get("allow_missing_masks", False))
        self.require_part_token = bool(data_config.get("require_part_token", True))
        self.use_mask_overlap_pooling = bool(data_config.get("use_mask_overlap_pooling", False))
        self.mask_patch_grid = int(data_config.get("mask_patch_grid", 37))
        self.mask_overlap_patch_grid_h = int(
            data_config.get("mask_overlap_patch_grid_h", getattr(self, "mask_patch_grid", 37))
        )
        self.mask_overlap_patch_grid_w = int(
            data_config.get("mask_overlap_patch_grid_w", getattr(self, "mask_patch_grid", 37))
        )
        self.mask_overlap_patch_h = int(data_config.get("mask_overlap_patch_h", 14))
        self.mask_overlap_patch_w = int(data_config.get("mask_overlap_patch_w", 14))
        self.mask_overlap_patch_start_index = int(data_config.get("mask_overlap_patch_start_index", 1))
        self.filter_zero_mask_coverage = bool(
            data_config.get(
                "filter_zero_mask_coverage",
                self.require_part_token and not self.use_mask_overlap_pooling,
            )
        )
        self.zero_mask_coverage_report = data_config.get("zero_mask_coverage_report")
        self.include_obj_ids = self._normalize_obj_ids(data_config.get("include_obj_ids"))
        self.exclude_obj_ids = self._normalize_obj_ids(data_config.get("exclude_obj_ids"))

        self.recon_root = self.data_root / self.recon_subdir
        self.mask_root = self.data_root / self.mask_subdir
        self.samples = self._enumerate_part_samples()
        self.samples = self._filter_samples_by_obj_id(self.samples)
        if self.filter_zero_mask_coverage:
            self.samples = self._filter_samples_with_mask_coverage(self.samples)
        if not self.samples:
            raise RuntimeError("PartSSLatentFlowDataset produced 0 object samples")

        self.loads = [1] * len(self.samples)
        total_parts = sum(len(sample["parts"]) for sample in self.samples)
        print(
            f"[PartSSLatentFlowDataset] {len(self.samples)} object samples / "
            f"{total_parts} target parts "
            f"from manifest {self.manifest_path}"
        )

    def _manifest_abs(self) -> Path:
        path = Path(self.manifest_path)
        return path if path.is_absolute() else self.data_root / path

    @staticmethod
    def _normalize_obj_ids(value: Any) -> set[str] | None:
        if value is None:
            return None
        if isinstance(value, str):
            return {value}
        return {str(item) for item in value}

    def _filter_samples_by_obj_id(
        self,
        samples: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        if self.include_obj_ids is not None and self.exclude_obj_ids is not None:
            overlap = sorted(self.include_obj_ids & self.exclude_obj_ids)
            if overlap:
                raise ValueError(f"include_obj_ids and exclude_obj_ids overlap: {overlap}")
        filtered = []
        for sample in samples:
            obj_id = str(sample["obj_id"])
            if self.include_obj_ids is not None and obj_id not in self.include_obj_ids:
                continue
            if self.exclude_obj_ids is not None and obj_id in self.exclude_obj_ids:
                continue
            filtered.append(sample)
        return filtered

    def _enumerate_part_samples(self) -> List[Dict[str, Any]]:
        manifest_abs = self._manifest_abs()
        if not manifest_abs.is_file():
            raise FileNotFoundError(f"Manifest not found: {manifest_abs}")

        rows: List[Dict[str, Any]] = []
        with manifest_abs.open("r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, 1):
                if not line.strip():
                    continue
                rec = json.loads(line)
                for key in (
                    "object_id",
                    "angle_idx",
                    "sample_id",
                    "target_part_names",
                    "view_indices",
                ):
                    if key not in rec:
                        raise KeyError(f"{manifest_abs}:{line_no} missing field {key!r}")

                obj_id = str(rec["object_id"])
                angle_idx = int(rec["angle_idx"])
                target_names = [str(x) for x in rec["target_part_names"]]
                view_indices = [int(v) for v in rec["view_indices"]]
                if len(view_indices) != self.num_views:
                    raise ValueError(
                        f"{manifest_abs}:{line_no} expected {self.num_views} views, "
                        f"got {view_indices}"
                    )

                target_parts = rec.get("target_parts") or []
                part_by_name = {str(p.get("name")): p for p in target_parts}
                label_remap = {
                    int(k): int(v) for k, v in dict(rec.get("label_remap", {})).items()
                }
                if not label_remap:
                    for slot, name in enumerate(target_names, start=1):
                        part = part_by_name.get(name, {})
                        if "original_label" in part:
                            label_remap[int(part["original_label"])] = slot
                if not label_remap:
                    label_remap = {slot: slot for slot in range(1, len(target_names) + 1)}

                paths = dict(rec.get("paths", {}))
                parts = []
                for slot, part_name in enumerate(target_names, start=1):
                    part = part_by_name.get(part_name, {})
                    part_paths = dict(part.get("paths", {}))
                    target_slot = int(part.get("local_label", slot))
                    if "original_label" in part:
                        original_label = int(part["original_label"])
                        if original_label in label_remap:
                            target_slot = int(label_remap[original_label])
                    parts.append({
                        "part_name": part_name,
                        "target_slot": target_slot,
                        "target_part": part,
                        "z_part_rel": part_paths.get(
                            "part_latent",
                            f"{self.recon_subdir}/ss_latents_per_part/{obj_id}/angle_{angle_idx}/{part_name}.npy",
                        ),
                        "raw_ind_rel": part_paths.get(
                            "part_voxel",
                            f"{self.recon_subdir}/voxel_expanded/{obj_id}/angle_{angle_idx}/64/ind_{part_name}.npy",
                        ),
                    })
                rows.append({
                    "obj_id": obj_id,
                    "angle_idx": angle_idx,
                    "sample_id": str(rec["sample_id"]),
                    "category": str(rec.get("category", "")),
                    "name": str(rec.get("name", "")),
                    "source_line": rec.get("source_line"),
                    "task": str(rec.get("task", "")),
                    "target_part_names": target_names,
                    "parts": parts,
                    "label_remap": label_remap,
                    "view_indices": view_indices,
                    "image_paths": list(rec.get("image_paths", [])),
                    "manifest_paths": paths,
                    "z_global_rel": paths.get(
                        "overall_latent",
                        f"{self.recon_subdir}/ss_latents_expanded/{obj_id}/angle_{angle_idx}/latent.npz",
                    ),
                    "surface_rel": paths.get(
                        "overall_surface",
                        f"{self.recon_subdir}/voxel_expanded/{obj_id}/angle_{angle_idx}/64/surface.npy",
                    ),
                    "tokens_rel": paths.get(
                        "dinov2_tokens",
                        f"{self.recon_subdir}/dinov2_tokens/{obj_id}/angle_{angle_idx}/tokens.npz",
                    ),
                })
        return rows

    def __len__(self) -> int:
        return len(self.samples)

    def _rooted(self, rel_or_abs: str) -> Path:
        path = Path(rel_or_abs)
        return path if path.is_absolute() else self.data_root / path

    @staticmethod
    def _is_rank_zero() -> bool:
        return int(os.environ.get("RANK", "0")) == 0

    @staticmethod
    def _load_dense_latent(path: Path, *, obj_id: str, field: str, part_name: str | None = None) -> torch.Tensor:
        if not path.is_file():
            msg = f"{field} not found for obj_id={obj_id}: {path}"
            if part_name is not None:
                msg = f"{field} not found for obj_id={obj_id} part_name={part_name}: {path}"
            raise FileNotFoundError(msg)
        if path.suffix == ".npz":
            data = np.load(path)
            if "mean" not in data.files:
                raise KeyError(f"{path} expected key 'mean', found keys {data.files}")
            arr = data["mean"]
        else:
            arr = np.load(path)
        tensor = torch.from_numpy(np.asarray(arr)).float()
        if tuple(tensor.shape) != (8, 16, 16, 16):
            raise ValueError(f"{path} expected latent shape (8,16,16,16), got {tuple(tensor.shape)}")
        return tensor

    def _load_cond_tokens(self, sample: Dict[str, Any]) -> torch.Tensor:
        path = self._rooted(sample["tokens_rel"])
        if not path.is_file():
            raise FileNotFoundError(f"DINOv2 tokens not found for obj_id={sample['obj_id']}: {path}")
        data = np.load(path)
        if "tokens" not in data.files:
            raise KeyError(f"{path} expected key 'tokens', found keys {data.files}")
        tokens = torch.from_numpy(np.asarray(data["tokens"])).float()
        if tokens.dim() != 3:
            raise ValueError(f"{path} expected [V,T,D], got {tuple(tokens.shape)}")
        view_indices = sample["view_indices"]
        if max(view_indices) >= tokens.shape[0]:
            raise ValueError(
                f"{path} has {tokens.shape[0]} views, cannot select {view_indices}"
            )
        selected = tokens[view_indices]
        return selected.reshape(-1, selected.shape[-1])

    def _iter_mask_paths(self, sample: Dict[str, Any]) -> List[Path]:
        obj_id = sample["obj_id"]
        angle_dir = f"angle_{sample['angle_idx']}"
        return [
            self.mask_root / obj_id / angle_dir / "mask" / f"mask_{view_idx}.npy"
            for view_idx in sample["view_indices"]
        ]

    def _iter_rgb_paths(self, sample: Dict[str, Any]) -> List[Path]:
        obj_id = sample["obj_id"]
        angle_dir = f"angle_{sample['angle_idx']}"
        image_paths = list(sample.get("image_paths", []))
        out: List[Path] = []
        for row, view_idx in enumerate(sample["view_indices"]):
            candidates: List[Path] = []
            if row < len(image_paths):
                candidates.append(self._rooted(str(image_paths[row])))
            candidates.extend([
                self.mask_root / obj_id / angle_dir / "rgb" / f"view_{view_idx}.png",
                self.mask_root / obj_id / angle_dir / "rgb" / f"{view_idx:03d}.png",
                self.data_root / "renders" / obj_id / angle_dir / "rgb" / f"view_{view_idx}.png",
                self.data_root / "renders" / obj_id / angle_dir / "rgb" / f"{view_idx:03d}.png",
            ])
            out.append(next((path for path in candidates if path.is_file()), candidates[0]))
        return out

    @staticmethod
    def _pad_crop_mask(mask_2d: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
        h, w = mask_2d.shape
        pad_h = max(0, target_h - h)
        pad_w = max(0, target_w - w)
        if pad_h or pad_w:
            mask_2d = np.pad(mask_2d, ((0, pad_h), (0, pad_w)), mode="constant", constant_values=0)
        return mask_2d[:target_h, :target_w]

    def _part_original_label(self, sample: Dict[str, Any], part: Dict[str, Any]) -> int:
        target_part = dict(part.get("target_part", {}))
        if "original_label" in target_part:
            return int(target_part["original_label"])
        target_slot = int(part["target_slot"])
        matches = [
            int(orig_label)
            for orig_label, local_slot in sample["label_remap"].items()
            if int(local_slot) == target_slot
        ]
        if len(matches) == 1:
            return matches[0]
        raise KeyError(
            f"cannot resolve original_label for obj_id={sample['obj_id']} "
            f"part_name={part['part_name']} target_slot={target_slot}"
        )

    def _part_original_labels(self, sample: Dict[str, Any], part: Dict[str, Any]) -> List[int]:
        target_part = dict(part.get("target_part", {}))
        labels: List[int] = []
        seen: set[int] = set()

        def add(value: Any) -> None:
            if value is None:
                return
            if isinstance(value, (list, tuple, set)):
                for item in value:
                    add(item)
                return
            label = int(value)
            if label not in seen:
                labels.append(label)
                seen.add(label)

        add(target_part.get("original_label"))
        for key in ("prompt_original_labels", "merged_original_labels", "target_original_labels"):
            add(target_part.get(key))
            add(part.get(key))

        target_slot = int(part["target_slot"])
        for orig_label, local_slot in sample.get("label_remap", {}).items():
            if int(local_slot) == target_slot:
                add(orig_label)

        if not labels:
            labels.append(self._part_original_label(sample, part))
        return labels

    def _compute_part_token_weights(
        self,
        sample: Dict[str, Any],
        token_count_per_view: int,
    ) -> torch.Tensor:
        patch_grid_h = self.mask_overlap_patch_grid_h
        patch_grid_w = self.mask_overlap_patch_grid_w
        patch_h = self.mask_overlap_patch_h
        patch_w = self.mask_overlap_patch_w
        patch_start = self.mask_overlap_patch_start_index
        patch_count = patch_grid_h * patch_grid_w
        if patch_start + patch_count != token_count_per_view:
            raise ValueError(
                f"mask overlap token layout mismatch: patch_start={patch_start} "
                f"grid={patch_grid_h}x{patch_grid_w} token_count_per_view={token_count_per_view}"
            )
        if patch_grid_h <= 0 or patch_grid_w <= 0 or patch_h <= 0 or patch_w <= 0:
            raise ValueError(
                "mask overlap grid and patch sizes must be positive, got "
                f"grid={patch_grid_h}x{patch_grid_w} patch={patch_h}x{patch_w}"
            )

        parts = sample["parts"]
        weights = torch.zeros(
            (len(parts), len(sample["view_indices"]), token_count_per_view),
            dtype=torch.float32,
        )
        target_h = patch_grid_h * patch_h
        target_w = patch_grid_w * patch_w
        mask_paths = self._iter_mask_paths(sample)
        if len(mask_paths) != len(sample["view_indices"]):
            raise ValueError(
                f"len(mask_paths)={len(mask_paths)} does not match "
                f"len(view_indices)={len(sample['view_indices'])}"
            )
        original_label_sets = [self._part_original_labels(sample, part) for part in parts]

        for row, mask_path in enumerate(mask_paths):
            if not mask_path.is_file():
                if self.allow_missing_masks:
                    continue
                raise FileNotFoundError(
                    f"mask file not found for obj_id={sample['obj_id']}: {mask_path}"
                )
            mask_2d = np.asarray(np.load(mask_path))
            if mask_2d.ndim != 2:
                raise ValueError(f"{mask_path} expected [H,W] mask, got {mask_2d.shape}")
            mask_2d = self._pad_crop_mask(mask_2d, target_h, target_w)
            cells = mask_2d.reshape(patch_grid_h, patch_h, patch_grid_w, patch_w)
            cells = cells.transpose(0, 2, 1, 3).reshape(patch_count, patch_h * patch_w)
            for part_idx, labels in enumerate(original_label_sets):
                counts = np.isin(cells, np.asarray(labels, dtype=cells.dtype)).sum(axis=1).astype(np.float32, copy=False)
                weights[part_idx, row, patch_start:patch_start + patch_count] = torch.from_numpy(counts)

        flat = weights.reshape(len(parts), -1)
        denom = flat.sum(dim=-1, keepdim=True)
        if bool((denom <= 0).any()):
            missing = [
                parts[idx]["part_name"]
                for idx in torch.nonzero(
                    denom.squeeze(-1) <= 0,
                    as_tuple=False,
                ).flatten().tolist()
            ]
            raise ValueError(
                f"raw mask has zero overlap for obj_id={sample['obj_id']} "
                f"angle_idx={sample['angle_idx']} parts={missing}"
            )
        return flat / denom.clamp_min(1e-8)

    def _compute_mask_token_labels(
        self,
        sample: Dict[str, Any],
        token_count_per_view: int,
        *,
        validate_require_part_token: bool = True,
    ) -> torch.Tensor:
        view_indices = sample["view_indices"]
        labels = torch.zeros((len(view_indices), token_count_per_view), dtype=torch.long)
        if token_count_per_view < 2:
            raise ValueError(f"token_count_per_view must include CLS + patches, got {token_count_per_view}")

        patch_grid = int(round((token_count_per_view - 1) ** 0.5))
        if patch_grid * patch_grid != token_count_per_view - 1:
            raise ValueError(
                f"token_count_per_view={token_count_per_view} does not match CLS + square patch grid"
            )

        label_remap = sample["label_remap"]
        obj_id = sample["obj_id"]
        for row, mask_path in enumerate(self._iter_mask_paths(sample)):
            if not mask_path.is_file():
                if self.allow_missing_masks:
                    continue
                view_idx = view_indices[row]
                raise FileNotFoundError(f"mask file not found for obj_id={obj_id} view={view_idx}: {mask_path}")
            mask_2d = torch.from_numpy(np.asarray(np.load(mask_path))).long()
            patch_mask = patch_aggregate_foreground_wins(mask_2d, grid=patch_grid)
            remapped = torch.zeros_like(patch_mask)
            for orig_label, local_label in label_remap.items():
                remapped[patch_mask == int(orig_label)] = int(local_label)
            labels[row, 1:] = remapped.reshape(-1)

        flat = labels.reshape(-1)
        if self.require_part_token and validate_require_part_token:
            for part in sample["parts"]:
                target_slot = int(part["target_slot"])
                if not bool((flat == target_slot).any()):
                    raise ValueError(
                        f"target_slot={target_slot} has zero 2D mask token coverage for "
                        f"obj_id={obj_id} part_name={part['part_name']}"
                    )
        return flat

    def _zero_mask_coverage_records(self, sample: Dict[str, Any]) -> List[Dict[str, Any]]:
        token_count_per_view = self.mask_patch_grid * self.mask_patch_grid + 1
        labels = self._compute_mask_token_labels(
            sample,
            token_count_per_view,
            validate_require_part_token=False,
        )
        records = []
        for part in sample["parts"]:
            target_slot = int(part["target_slot"])
            if bool((labels == target_slot).any()):
                continue
            records.append({
                "part_name": part["part_name"],
                "target_slot": target_slot,
            })
        return records

    def _filter_samples_with_mask_coverage(
        self,
        samples: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        kept = []
        skipped = []
        for sample in samples:
            zero_parts = self._zero_mask_coverage_records(sample)
            if not zero_parts:
                kept.append(sample)
                continue
            skipped.append({
                "reason": "zero_2d_mask_token_coverage",
                "obj_id": sample["obj_id"],
                "angle_idx": int(sample["angle_idx"]),
                "sample_id": sample["sample_id"],
                "view_indices": list(sample["view_indices"]),
                "zero_parts": zero_parts,
            })

        if skipped and self._is_rank_zero():
            skipped_parts = sum(len(rec["zero_parts"]) for rec in skipped)
            print(
                f"[PartSSLatentFlowDataset] filtered {len(skipped)} object samples / "
                f"{skipped_parts} target parts with zero 2D mask token coverage"
            )
            if self.zero_mask_coverage_report:
                report_path = self._rooted(str(self.zero_mask_coverage_report))
                report_path.parent.mkdir(parents=True, exist_ok=True)
                report = {
                    "reason": "zero_2d_mask_token_coverage",
                    "input_samples": len(samples),
                    "kept_samples": len(kept),
                    "skipped_samples": len(skipped),
                    "skipped_parts": skipped_parts,
                    "mask_patch_grid": self.mask_patch_grid,
                    "manifest_path": str(self._manifest_abs()),
                    "records": skipped,
                }
                report_path.write_text(
                    json.dumps(report, indent=2, ensure_ascii=False) + "\n",
                    encoding="utf-8",
                )
                print(f"[PartSSLatentFlowDataset] zero-mask skip report: {report_path}")
        return kept

    def _load_raw_ind_coords(self, sample: Dict[str, Any], part: Dict[str, Any]) -> torch.Tensor:
        path = self._rooted(part["raw_ind_rel"])
        if not path.is_file():
            raise FileNotFoundError(
                f"raw ind voxel file not found for obj_id={sample['obj_id']} "
                f"part_name={part['part_name']}: {path}"
            )
        coords = torch.from_numpy(np.asarray(np.load(path))).long()
        if coords.dim() != 2 or coords.shape[1] != 3:
            raise ValueError(f"{path} expected [N,3] coords, got {tuple(coords.shape)}")
        return coords

    def _load_surface_coords(self, sample: Dict[str, Any]) -> torch.Tensor:
        path = self._rooted(sample["surface_rel"])
        if not path.is_file():
            raise FileNotFoundError(
                f"overall surface voxel file not found for obj_id={sample['obj_id']}: {path}"
            )
        coords = torch.from_numpy(np.asarray(np.load(path))).long()
        if coords.dim() != 2 or coords.shape[1] != 3:
            raise ValueError(f"{path} expected [N,3] coords, got {tuple(coords.shape)}")
        return coords

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self.samples[idx]
        obj_id = sample["obj_id"]

        z_global = self._load_dense_latent(
            self._rooted(sample["z_global_rel"]),
            obj_id=obj_id,
            field="z_global",
        )
        cond = self._load_cond_tokens(sample)
        token_count_per_view = cond.shape[0] // len(sample["view_indices"])
        mask_token_labels = self._compute_mask_token_labels(
            sample,
            token_count_per_view,
            validate_require_part_token=not self.use_mask_overlap_pooling,
        )
        if mask_token_labels.shape != (cond.shape[0],):
            raise ValueError(
                f"mask_token_labels shape {tuple(mask_token_labels.shape)} does not match "
                f"cond tokens {cond.shape[0]}"
            )
        part_token_weights = None
        if self.use_mask_overlap_pooling:
            part_token_weights = self._compute_part_token_weights(sample, token_count_per_view)
        x_1_parts = []
        raw_ind_coords = []
        target_slots = []
        target_part_names = []
        target_parts = []
        for part in sample["parts"]:
            part_name = part["part_name"]
            x_1_parts.append(self._load_dense_latent(
                self._rooted(part["z_part_rel"]),
                obj_id=obj_id,
                field="z_part",
                part_name=part_name,
            ))
            raw_ind_coords.append(self._load_raw_ind_coords(sample, part))
            target_slots.append(int(part["target_slot"]))
            target_part_names.append(part_name)
            target_parts.append(dict(part["target_part"]))
        part_raw_voxel_counts = torch.tensor(
            [int(coords.shape[0]) for coords in raw_ind_coords],
            dtype=torch.float32,
        )

        out = {
            "x_1_parts": torch.stack(x_1_parts, dim=0),
            "part_valid": torch.ones(len(x_1_parts), dtype=torch.bool),
            "part_raw_voxel_counts": part_raw_voxel_counts,
            "z_global": z_global,
            "cond": cond,
            "mask_token_labels": mask_token_labels.long(),
            "target_slots": torch.tensor(target_slots, dtype=torch.long),
            "target_part_names": target_part_names,
            "target_parts": target_parts,
            "raw_ind_coords": raw_ind_coords,
            "raw_surface_coords": self._load_surface_coords(sample),
            "obj_id": obj_id,
            "angle_idx": int(sample["angle_idx"]),
            "sample_id": sample["sample_id"],
            "view_indices": list(sample["view_indices"]),
            "image_paths": list(sample["image_paths"]),
        }
        if part_token_weights is not None:
            out["part_token_weights"] = part_token_weights
        return out

    @staticmethod
    def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        max_parts = max(int(sample["x_1_parts"].shape[0]) for sample in batch)
        latent_shape = batch[0]["x_1_parts"].shape[1:]
        x_1_parts = torch.zeros((len(batch), max_parts) + latent_shape, dtype=batch[0]["x_1_parts"].dtype)
        part_valid = torch.zeros((len(batch), max_parts), dtype=torch.bool)
        part_raw_voxel_counts = torch.zeros((len(batch), max_parts), dtype=torch.float32)
        target_slots = torch.zeros((len(batch), max_parts), dtype=torch.long)
        for row, sample in enumerate(batch):
            k = int(sample["x_1_parts"].shape[0])
            x_1_parts[row, :k] = sample["x_1_parts"]
            part_valid[row, :k] = sample["part_valid"]
            part_raw_voxel_counts[row, :k] = sample["part_raw_voxel_counts"].float()
            target_slots[row, :k] = sample["target_slots"]

        tensor_stack_keys = ("z_global", "cond", "mask_token_labels")
        out: Dict[str, Any] = {
            key: torch.stack([sample[key] for sample in batch], dim=0)
            for key in tensor_stack_keys
        }
        out["x_1_parts"] = x_1_parts
        out["part_valid"] = part_valid
        out["part_raw_voxel_counts"] = part_raw_voxel_counts
        out["target_slots"] = target_slots
        if "part_token_weights" in batch[0]:
            token_count = int(batch[0]["cond"].shape[0])
            part_token_weights = torch.zeros(
                (len(batch), max_parts, token_count),
                dtype=batch[0]["part_token_weights"].dtype,
            )
            for row, sample in enumerate(batch):
                k = int(sample["x_1_parts"].shape[0])
                part_token_weights[row, :k] = sample["part_token_weights"]
            out["part_token_weights"] = part_token_weights
        for key in (
            "target_part_names",
            "target_parts",
            "raw_ind_coords",
            "raw_surface_coords",
            "obj_id",
            "angle_idx",
            "sample_id",
            "view_indices",
            "image_paths",
        ):
            out[key] = [sample[key] for sample in batch]
        return out
