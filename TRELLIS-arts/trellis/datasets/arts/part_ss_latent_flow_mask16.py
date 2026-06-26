"""Part SS latent flow dataset with explicit mask16 conditioning."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch

from .part_ss_latent_flow import PartSSLatentFlowDataset


__all__ = ["PartSSLatentFlowMask16Dataset", "raw_coords_to_mask16"]


def raw_coords_to_mask16(
    coords: torch.Tensor,
    *,
    resolution: int = 64,
    latent_resolution: int = 16,
) -> torch.Tensor:
    coords = torch.as_tensor(coords, dtype=torch.long)
    mask = torch.zeros((latent_resolution, latent_resolution, latent_resolution), dtype=torch.float32)
    if coords.numel() == 0:
        return mask
    stride = resolution // latent_resolution
    latent = torch.div(coords.clamp(0, resolution - 1), stride, rounding_mode="floor")
    mask[latent[:, 0], latent[:, 1], latent[:, 2]] = 1.0
    return mask


class PartSSLatentFlowMask16Dataset(PartSSLatentFlowDataset):
    """Adds per-part 16^3 support masks to the existing part-flow dataset.

    ``part_mask16`` is the conditioning tensor used by the model. It can come
    from GT raw coords or from a Stage1 prediction export. ``part_mask16_gt`` is
    always kept for inspection and foreground weighting.
    """

    def __init__(self, data_config: dict):
        self.include_object_angles = self._normalize_object_angles(data_config.get("include_object_angles"))
        self.part_mask16_source = str(data_config.get("part_mask16_source", "gt")).lower()
        if self.part_mask16_source not in {"gt", "pred"}:
            raise ValueError("part_mask16_source must be 'gt' or 'pred'")
        self.pred_mask16_root = data_config.get("pred_mask16_root")
        self.pred_mask16_kind = str(data_config.get("pred_mask16_kind", "bin")).lower()
        if self.pred_mask16_kind not in {"bin", "prob"}:
            raise ValueError("pred_mask16_kind must be 'bin' or 'prob'")
        if self.part_mask16_source == "pred":
            if not self.pred_mask16_root:
                raise ValueError("pred_mask16_root is required when part_mask16_source='pred'")
            self.pred_mask16_root = Path(str(self.pred_mask16_root))
            if not self.pred_mask16_root.is_dir():
                raise FileNotFoundError(f"pred_mask16_root not found: {self.pred_mask16_root}")
        super().__init__(data_config)
        if self.include_object_angles is not None:
            before = len(self.samples)
            self.samples = [
                sample
                for sample in self.samples
                if (str(sample["obj_id"]), int(sample["angle_idx"])) in self.include_object_angles
            ]
            if not self.samples:
                raise RuntimeError("include_object_angles filtered dataset to 0 samples")
            print(
                f"[PartSSLatentFlowMask16Dataset] angle-filtered {before} -> {len(self.samples)} samples",
                flush=True,
            )
        print(
            f"[PartSSLatentFlowMask16Dataset] part_mask16_source={self.part_mask16_source} "
            f"pred_kind={self.pred_mask16_kind}",
            flush=True,
        )

    @staticmethod
    def _normalize_object_angles(value: Any) -> set[tuple[str, int]] | None:
        if value is None:
            return None
        out: set[tuple[str, int]] = set()
        for item in value:
            if isinstance(item, Mapping) or hasattr(item, "get"):
                obj_id = item.get("object_id", item.get("obj_id"))
                angle_idx = item.get("angle_idx")
            else:
                obj_id, angle_idx = item
            out.add((str(obj_id), int(angle_idx)))
        return out

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        out = super().__getitem__(idx)
        gt_mask16 = torch.stack(
            [raw_coords_to_mask16(coords) for coords in out["raw_ind_coords"]],
            dim=0,
        )
        out["part_mask16_gt"] = gt_mask16
        if self.part_mask16_source == "gt":
            out["part_mask16"] = gt_mask16
        else:
            out["part_mask16"] = torch.stack(
                [
                    self._load_pred_mask16(
                        obj_id=str(out["obj_id"]),
                        angle_idx=int(out["angle_idx"]),
                        part_name=str(part_name),
                    )
                    for part_name in out["target_part_names"]
                ],
                dim=0,
            )
        return out

    def _pred_mask16_path(self, *, obj_id: str, angle_idx: int, part_name: str) -> Path:
        assert self.pred_mask16_root is not None
        return Path(self.pred_mask16_root) / obj_id / f"angle_{angle_idx}" / f"{part_name}_{self.pred_mask16_kind}.npy"

    def _load_pred_mask16(self, *, obj_id: str, angle_idx: int, part_name: str) -> torch.Tensor:
        path = self._pred_mask16_path(obj_id=obj_id, angle_idx=angle_idx, part_name=part_name)
        if not path.is_file():
            raise FileNotFoundError(
                f"predicted mask16 not found for obj_id={obj_id} angle_idx={angle_idx} "
                f"part_name={part_name}: {path}"
            )
        arr = np.asarray(np.load(path))
        if tuple(arr.shape) != (16, 16, 16):
            raise ValueError(f"{path} expected shape (16,16,16), got {arr.shape}")
        return torch.from_numpy(arr.astype(np.float32, copy=False))

    @staticmethod
    def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        out = PartSSLatentFlowDataset.collate_fn(batch)
        max_parts = out["x_1_parts"].shape[1]
        gt_masks = torch.zeros((len(batch), max_parts, 16, 16, 16), dtype=torch.float32)
        cond_masks = torch.zeros((len(batch), max_parts, 16, 16, 16), dtype=torch.float32)
        for row, sample in enumerate(batch):
            k = int(sample["part_mask16_gt"].shape[0])
            gt_masks[row, :k] = sample["part_mask16_gt"]
            cond_masks[row, :k] = sample["part_mask16"]
        out["part_mask16_gt"] = gt_masks
        out["part_mask16"] = cond_masks
        return out
