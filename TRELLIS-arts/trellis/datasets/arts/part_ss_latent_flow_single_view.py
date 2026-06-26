"""Single-view variant of PartSSLatentFlowDataset.

Consumes dataset_toolkits_single_image Part Completion manifests where each
row is one ``(object_id, angle_idx, view_idx)`` sample with K visible target
parts. Geometry loading, latent loading, and collate behavior stay inherited
from the 4-view dataset; this class only forces ``num_views=1`` and reads
per-row mask paths from the manifest when present.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch

from trellis.datasets.arts.part_ss_latent_flow import PartSSLatentFlowDataset
from trellis.utils.arts.mask_utils import patch_aggregate_foreground_wins


__all__ = ["PartSSLatentFlowSingleViewDataset"]


class PartSSLatentFlowSingleViewDataset(PartSSLatentFlowDataset):
    """Single-view dataset: one manifest row maps to one condition view."""

    def __init__(self, data_config: dict):
        forced = dict(data_config)
        configured = int(forced.get("num_views", 1))
        if configured != 1:
            raise ValueError(
                "PartSSLatentFlowSingleViewDataset requires num_views=1, "
                f"got {configured}"
            )
        forced["num_views"] = 1
        self.max_target_parts_per_sample = int(forced.get("max_target_parts_per_sample", 20))
        self.drop_over_max_parts = bool(forced.get("drop_over_max_parts", True))
        self.filter_missing_dino_tokens = bool(forced.get("filter_missing_dino_tokens", True))
        self.dropped_over_max_parts_samples = 0
        self.dropped_over_max_parts = 0
        self.dropped_missing_dino_token_samples = 0
        self.max_seen_target_parts = 0
        super().__init__(forced)

    def _enumerate_part_samples(self) -> List[Dict[str, Any]]:
        samples = super()._enumerate_part_samples()
        manifest_abs = self._manifest_abs()
        with manifest_abs.open("r", encoding="utf-8") as handle:
            rows = [json.loads(line) for line in handle if line.strip()]
        if len(rows) != len(samples):
            raise RuntimeError(
                f"manifest row count {len(rows)} differs from parent enumeration "
                f"{len(samples)}; cannot align mask_paths"
            )

        kept_samples: List[Dict[str, Any]] = []
        dropped_samples = 0
        dropped_parts = 0
        dropped_missing_dino = 0
        max_seen = 0
        for sample, rec in zip(samples, rows):
            part_count = len(sample["parts"])
            max_seen = max(max_seen, part_count)
            if (
                self.drop_over_max_parts
                and self.max_target_parts_per_sample > 0
                and part_count > self.max_target_parts_per_sample
            ):
                dropped_samples += 1
                dropped_parts += part_count
                continue

            if self.filter_missing_dino_tokens and not self._rooted(sample["tokens_rel"]).is_file():
                dropped_missing_dino += 1
                continue

            mask_paths = rec.get("mask_paths")
            if mask_paths is None:
                kept_samples.append(sample)
                continue
            if len(mask_paths) != 1:
                raise ValueError(
                    "single-view manifest requires len(mask_paths)==1, "
                    f"got {mask_paths} for sample {rec.get('sample_id')}"
                )
            sample["mask_paths"] = [str(mask_paths[0])]
            kept_samples.append(sample)

        self.dropped_over_max_parts_samples = dropped_samples
        self.dropped_over_max_parts = dropped_parts
        self.dropped_missing_dino_token_samples = dropped_missing_dino
        self.max_seen_target_parts = max_seen
        if dropped_samples and self._is_rank_zero():
            print(
                "[PartSSLatentFlowSingleViewDataset] dropped "
                f"{dropped_samples} samples / {dropped_parts} target parts with "
                f"K>{self.max_target_parts_per_sample}; max_seen_K={max_seen}"
            )
        if dropped_missing_dino and self._is_rank_zero():
            print(
                "[PartSSLatentFlowSingleViewDataset] dropped "
                f"{dropped_missing_dino} samples with missing DINO tokens"
            )
        return kept_samples

    def _iter_mask_paths(self, sample: Dict[str, Any]) -> List[Path]:
        if "mask_paths" not in sample:
            return super()._iter_mask_paths(sample)
        mask_paths = sample["mask_paths"]
        if len(mask_paths) != len(sample["view_indices"]):
            raise ValueError(
                f"single-view: len(mask_paths)={len(mask_paths)} "
                f"!= len(view_indices)={len(sample['view_indices'])}"
            )
        return [self._rooted(str(path)) for path in mask_paths]

    def _iter_rgb_paths(self, sample: Dict[str, Any]) -> List[Path]:
        image_paths = list(sample.get("image_paths", []))
        if len(image_paths) == len(sample["view_indices"]):
            rooted = [self._rooted(str(path)) for path in image_paths]
            if all(path.is_file() for path in rooted):
                return rooted
        obj_id = sample["obj_id"]
        angle_dir = f"angle_{sample['angle_idx']}"
        out: List[Path] = []
        for row, view_idx in enumerate(sample["view_indices"]):
            candidates: List[Path] = []
            if row < len(image_paths):
                candidates.append(self._rooted(str(image_paths[row])))
            candidates.extend([
                self.mask_root / obj_id / angle_dir / "part_complete" / "rgb" / f"view_{view_idx}.png",
                self.data_root / "renders" / obj_id / angle_dir / "part_complete" / "rgb" / f"view_{view_idx}.png",
            ])
            out.append(next((path for path in candidates if path.is_file()), candidates[0]))
        return out

    def _compute_mask_token_labels(
        self,
        sample: Dict[str, Any],
        token_count_per_view: int,
        *,
        validate_require_part_token: bool = True,
    ) -> torch.Tensor:
        if "mask_paths" not in sample:
            return super()._compute_mask_token_labels(
                sample,
                token_count_per_view,
                validate_require_part_token=validate_require_part_token,
            )

        view_indices = sample["view_indices"]
        mask_paths = sample["mask_paths"]
        if len(view_indices) != len(mask_paths):
            raise ValueError(
                f"single-view: len(view_indices)={len(view_indices)} "
                f"!= len(mask_paths)={len(mask_paths)}"
            )

        labels = torch.zeros((len(view_indices), token_count_per_view), dtype=torch.long)
        if token_count_per_view < 2:
            raise ValueError(
                "token_count_per_view must include CLS + patches, "
                f"got {token_count_per_view}"
            )
        patch_grid = int(round((token_count_per_view - 1) ** 0.5))
        if patch_grid * patch_grid != token_count_per_view - 1:
            raise ValueError(
                f"token_count_per_view={token_count_per_view} does not match "
                "CLS + square patch grid"
            )

        label_remap = sample["label_remap"]
        obj_id = sample["obj_id"]
        for row, (view_idx, rel_path) in enumerate(zip(view_indices, mask_paths)):
            mask_abs = self._rooted(rel_path)
            if not mask_abs.is_file():
                if self.allow_missing_masks:
                    continue
                raise FileNotFoundError(
                    f"mask file not found for obj_id={obj_id} view={view_idx}: {mask_abs}"
                )
            mask_2d = torch.from_numpy(np.asarray(np.load(mask_abs))).long()
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
