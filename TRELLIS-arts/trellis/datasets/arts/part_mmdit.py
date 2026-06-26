"""PartMMDiT object-level dataset.

Each sample is one object/angle with K target parts. It emits dense SS latents
for target parts, one global SS latent canvas, shared DINOv2 image tokens,
frozen CLIP name token sequences, and visible-masked 2D bbox anchors.

This dataset is intentionally independent from ``part_ss_latent_flow``. It does
not emit soft-role mask labels, target slots, or part-token weights.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


__all__ = ["PartMMDiTDataset", "raw_coords_to_part_fg_mask"]


LATENT_SHAPE = (8, 16, 16, 16)
FG_MASK_SHAPE = (16, 16, 16)
DEFAULT_NAME_SEQ_CACHE = "clip_vitl14_seq.pt"


def raw_coords_to_part_fg_mask(
    coords: torch.Tensor,
    *,
    raw_resolution: int = 64,
    latent_resolution: int = 16,
    dilate: int = 0,
) -> torch.Tensor:
    """Max-pool sparse raw occupancy coords into a dense latent-grid mask."""

    if int(raw_resolution) % int(latent_resolution) != 0:
        raise ValueError(
            f"raw_resolution={raw_resolution} must be divisible by "
            f"latent_resolution={latent_resolution}"
        )
    coords = torch.as_tensor(coords, dtype=torch.long)
    if coords.dim() != 2 or coords.shape[1] != 3:
        raise ValueError(f"coords expected [N,3], got {tuple(coords.shape)}")
    mask = torch.zeros(
        (int(latent_resolution), int(latent_resolution), int(latent_resolution)),
        dtype=torch.bool,
    )
    if coords.numel() == 0:
        return mask
    if int(coords.min().item()) < 0 or int(coords.max().item()) >= int(raw_resolution):
        raise ValueError(
            f"raw coords must be in [0,{int(raw_resolution) - 1}], "
            f"got min={int(coords.min().item())} max={int(coords.max().item())}"
        )
    stride = int(raw_resolution) // int(latent_resolution)
    latent_coords = torch.div(coords, stride, rounding_mode="floor")
    mask[latent_coords[:, 0], latent_coords[:, 1], latent_coords[:, 2]] = True
    dilate = int(dilate)
    if dilate < 0:
        raise ValueError(f"dilate must be >= 0, got {dilate}")
    if dilate:
        pooled = F.max_pool3d(
            mask.float().view(1, 1, *mask.shape),
            kernel_size=2 * dilate + 1,
            stride=1,
            padding=dilate,
        )
        mask = pooled[0, 0].bool()
    return mask


class PartMMDiTDataset(Dataset):
    """Manifest-driven object-level PartMMDiT dataset."""

    def __init__(self, data_config: dict):
        super().__init__()
        self.data_root = Path(data_config["data_root"])
        self.recon_subdir = data_config.get("recon_subdir", "reconstruction")
        self.mask_subdir = data_config.get("mask_subdir", "renders")
        self.manifest_path = data_config["manifest_path"]
        self.num_views = int(data_config.get("num_views", 4))
        self.foreground_mask_dilate = int(data_config.get("foreground_mask_dilate", 0))
        self.max_samples = data_config.get("max_samples")
        self.include_obj_ids = self._normalize_obj_ids(data_config.get("include_obj_ids"))
        self.exclude_obj_ids = self._normalize_obj_ids(data_config.get("exclude_obj_ids"))
        if self.foreground_mask_dilate < 0:
            raise ValueError(
                f"foreground_mask_dilate must be >= 0, got {self.foreground_mask_dilate}"
            )

        self.recon_root = self.data_root / self.recon_subdir
        self.mask_root = self.data_root / self.mask_subdir
        self.name_seq_cache_path = self._resolve_name_cache_path(data_config)
        self.name_seq_table, self.name_dim = self._load_name_seq_cache(
            self.name_seq_cache_path
        )

        self.samples = self._enumerate_manifest()
        self.samples = self._filter_samples_by_obj_id(self.samples)
        if self.max_samples is not None:
            self.samples = self.samples[: int(self.max_samples)]
        if not self.samples:
            raise RuntimeError("PartMMDiTDataset produced 0 object samples")

        total_parts = sum(len(sample["parts"]) for sample in self.samples)
        print(
            f"[PartMMDiTDataset] {len(self.samples)} object samples / "
            f"{total_parts} target parts from manifest {self.manifest_path}"
        )

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

    def _resolve_name_cache_path(self, data_config: dict) -> Path:
        cache_path = data_config.get("name_emb_cache_path")
        if cache_path is None:
            return (
                self.recon_root
                / "name_emb_cache"
                / DEFAULT_NAME_SEQ_CACHE
            )
        path = Path(cache_path)
        return path if path.is_absolute() else self.data_root / path

    @staticmethod
    def _load_name_seq_cache(
        cache_path: Path,
    ) -> tuple[Dict[str, Dict[str, torch.Tensor]], int]:
        if not cache_path.is_file():
            raise FileNotFoundError(
                "name token cache missing (run precompute_clip_name_embeddings.py): "
                f"{cache_path}"
            )
        blob = torch.load(cache_path, map_location="cpu")
        if "seq" not in blob:
            raise KeyError(f"{cache_path} expected key 'seq'")
        if "dim" not in blob:
            raise KeyError(f"{cache_path} expected key 'dim'")
        dim = int(blob["dim"])
        table: Dict[str, Dict[str, torch.Tensor]] = {}
        for name, item in dict(blob["seq"]).items():
            item = dict(item)
            if "tokens" not in item or "mask" not in item:
                raise KeyError(f"{cache_path} sequence {name!r} expected tokens/mask")
            tokens = torch.as_tensor(item["tokens"], dtype=torch.float32)
            mask = torch.as_tensor(item["mask"], dtype=torch.bool)
            if tokens.dim() != 2 or tokens.shape[1] != dim:
                raise ValueError(
                    f"{cache_path} sequence {name!r} expected tokens [L,{dim}], "
                    f"got {tuple(tokens.shape)}"
                )
            if mask.dim() != 1 or mask.shape[0] != tokens.shape[0]:
                raise ValueError(
                    f"{cache_path} sequence {name!r} expected mask [{tokens.shape[0]}], "
                    f"got {tuple(mask.shape)}"
                )
            if not bool(mask.any()):
                raise ValueError(f"{cache_path} sequence {name!r} has empty mask")
            table[str(name)] = {"tokens": tokens, "mask": mask}
        return table, dim

    def _manifest_abs(self) -> Path:
        path = Path(self.manifest_path)
        return path if path.is_absolute() else self.data_root / path

    def _rooted(self, rel_or_abs: str) -> Path:
        path = Path(rel_or_abs)
        return path if path.is_absolute() else self.data_root / path

    def _enumerate_manifest(self) -> List[Dict[str, Any]]:
        manifest_abs = self._manifest_abs()
        if not manifest_abs.is_file():
            raise FileNotFoundError(f"Manifest not found: {manifest_abs}")

        rows: List[Dict[str, Any]] = []
        with manifest_abs.open("r", encoding="utf-8") as manifest_file:
            for line_no, line in enumerate(manifest_file, 1):
                if not line.strip():
                    continue
                record = json.loads(line)
                for key in (
                    "object_id",
                    "angle_idx",
                    "sample_id",
                    "target_part_names",
                    "view_indices",
                ):
                    if key not in record:
                        raise KeyError(f"{manifest_abs}:{line_no} missing field {key!r}")

                obj_id = str(record["object_id"])
                angle_idx = int(record["angle_idx"])
                target_names = [str(name) for name in record["target_part_names"]]
                view_indices = [int(view_idx) for view_idx in record["view_indices"]]
                if len(view_indices) != self.num_views:
                    raise ValueError(
                        f"{manifest_abs}:{line_no} expected {self.num_views} views, "
                        f"got {view_indices}"
                    )

                paths = dict(record.get("paths", {}))
                part_info_rel = paths.get(
                    "part_info",
                    f"{self.recon_subdir}/part_info/{obj_id}/part_info.json",
                )
                part_info = self._load_part_info(
                    self._rooted(part_info_rel),
                    obj_id=obj_id,
                )

                target_parts = record.get("target_parts") or []
                part_by_name = {str(part.get("name")): part for part in target_parts}
                parts = []
                for part_name in target_names:
                    target_part = dict(part_by_name.get(part_name, {}))
                    part_paths = dict(target_part.get("paths", {}))
                    part_type = self._part_type(part_info, target_part, part_name)
                    parts.append(
                        {
                            "part_name": part_name,
                            "part_type": part_type,
                            "target_part": target_part,
                            "z_part_rel": part_paths.get(
                                "part_latent",
                                (
                                    f"{self.recon_subdir}/ss_latents_per_part/"
                                    f"{obj_id}/angle_{angle_idx}/{part_name}.npy"
                                ),
                            ),
                            "raw_ind_rel": part_paths.get(
                                "part_voxel",
                                (
                                    f"{self.recon_subdir}/voxel_expanded/"
                                    f"{obj_id}/angle_{angle_idx}/64/ind_{part_name}.npy"
                                ),
                            ),
                        }
                    )

                rows.append(
                    {
                        "obj_id": obj_id,
                        "angle_idx": angle_idx,
                        "sample_id": str(record["sample_id"]),
                        "target_part_names": target_names,
                        "parts": parts,
                        "view_indices": view_indices,
                        "image_paths": list(record.get("image_paths", [])),
                        "z_global_rel": paths.get(
                            "overall_latent",
                            (
                                f"{self.recon_subdir}/ss_latents_expanded/"
                                f"{obj_id}/angle_{angle_idx}/latent.npz"
                            ),
                        ),
                        "tokens_rel": paths.get(
                            "dinov2_tokens",
                            (
                                f"{self.recon_subdir}/dinov2_tokens/"
                                f"{obj_id}/angle_{angle_idx}/tokens.npz"
                            ),
                        ),
                    }
                )
        return rows

    @staticmethod
    def _load_part_info(path: Path, *, obj_id: str) -> Dict[str, Any]:
        if not path.is_file():
            raise FileNotFoundError(f"part_info.json missing for obj_id={obj_id}: {path}")
        with path.open("r", encoding="utf-8") as part_info_file:
            part_info = json.load(part_info_file)
        if "parts" not in part_info:
            raise KeyError(f"{path} expected key 'parts'")
        return part_info

    @staticmethod
    def _part_type(
        part_info: Dict[str, Any],
        target_part: Dict[str, Any],
        part_name: str,
    ) -> str:
        if "type" in target_part:
            return str(target_part["type"])
        parts = part_info["parts"]
        if part_name not in parts:
            raise KeyError(f"part_info missing part {part_name!r}")
        part = dict(parts[part_name])
        if "type" not in part:
            raise KeyError(f"part_info[{part_name!r}] missing field 'type'")
        return str(part["type"])

    @staticmethod
    def _load_dense_latent(
        path: Path,
        *,
        obj_id: str,
        field: str,
        part_name: str | None = None,
    ) -> torch.Tensor:
        if not path.is_file():
            msg = f"{field} not found for obj_id={obj_id}: {path}"
            if part_name is not None:
                msg = f"{field} not found for obj_id={obj_id} part_name={part_name}: {path}"
            raise FileNotFoundError(msg)
        if path.suffix == ".npz":
            data = np.load(path)
            if "mean" not in data.files:
                raise KeyError(f"{path} expected key 'mean', found keys {data.files}")
            array = data["mean"]
        else:
            array = np.load(path)
        tensor = torch.from_numpy(np.asarray(array)).float()
        if tuple(tensor.shape) != LATENT_SHAPE:
            raise ValueError(
                f"{path} expected latent shape {LATENT_SHAPE}, got {tuple(tensor.shape)}"
            )
        return tensor

    def _load_cond_tokens(self, sample: Dict[str, Any]) -> torch.Tensor:
        path = self._rooted(sample["tokens_rel"])
        if not path.is_file():
            raise FileNotFoundError(
                f"DINOv2 tokens not found for obj_id={sample['obj_id']}: {path}"
            )
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

    def _load_raw_ind_coords(
        self,
        sample: Dict[str, Any],
        part: Dict[str, Any],
    ) -> torch.Tensor:
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

    def _name_seq(self, part_type: str) -> Dict[str, torch.Tensor]:
        if part_type not in self.name_seq_table:
            raise KeyError(
                f"part type {part_type!r} not in name token cache; re-run precompute"
            )
        return self.name_seq_table[part_type]

    def _pad_name_sequences(
        self,
        items: List[Dict[str, torch.Tensor]],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        max_len = max(int(item["tokens"].shape[0]) for item in items)
        tokens = torch.zeros(len(items), max_len, self.name_dim, dtype=torch.float32)
        mask = torch.zeros(len(items), max_len, dtype=torch.bool)
        for row, item in enumerate(items):
            seq_tokens = item["tokens"]
            seq_mask = item["mask"]
            length = int(seq_tokens.shape[0])
            tokens[row, :length] = seq_tokens
            mask[row, :length] = seq_mask
        return tokens, mask

    def _anchor_for_part(
        self,
        obj_id: str,
        angle_idx: int,
        part_name: str,
        view_indices: List[int],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        bbox_path = self.mask_root / obj_id / f"angle_{angle_idx}" / "bbox_gt.json"
        if not bbox_path.is_file():
            raise FileNotFoundError(f"bbox_gt.json missing: {bbox_path}")
        with bbox_path.open("r", encoding="utf-8") as bbox_file:
            bbox_gt = json.load(bbox_file)
        resolution = float(bbox_gt.get("resolution", 512))
        if "parts" not in bbox_gt:
            raise KeyError(f"{bbox_path} expected key 'parts'")
        if part_name not in bbox_gt["parts"]:
            raise KeyError(f"{bbox_path} missing part {part_name!r}")
        views = dict(bbox_gt["parts"][part_name].get("views", {}))

        anchor = torch.zeros(len(view_indices), 4, dtype=torch.float32)
        anchor_valid = torch.zeros(len(view_indices), dtype=torch.bool)
        for row, view_idx in enumerate(view_indices):
            view = views.get(str(view_idx))
            if view is None or not bool(view.get("visible", False)):
                continue
            if "bbox" not in view:
                raise KeyError(f"{bbox_path} part {part_name!r} view {view_idx} missing 'bbox'")
            x0, y0, x1, y1 = [float(coord) / resolution for coord in view["bbox"]]
            anchor[row] = torch.tensor(
                [
                    (x0 + x1) * 0.5,
                    (y0 + y1) * 0.5,
                    x1 - x0,
                    y1 - y0,
                ],
                dtype=torch.float32,
            )
            anchor_valid[row] = True
        return anchor, anchor_valid

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self.samples[idx]
        obj_id = sample["obj_id"]
        angle_idx = int(sample["angle_idx"])
        view_indices = list(sample["view_indices"])

        z_global = self._load_dense_latent(
            self._rooted(sample["z_global_rel"]),
            obj_id=obj_id,
            field="z_global",
        )
        cond = self._load_cond_tokens(sample)

        x_1_parts = []
        name_sequences = []
        anchors = []
        anchor_valids = []
        raw_ind_coords = []
        target_part_names = []
        target_part_types = []
        target_parts = []
        for part in sample["parts"]:
            part_name = part["part_name"]
            part_type = part["part_type"]
            x_1_parts.append(
                self._load_dense_latent(
                    self._rooted(part["z_part_rel"]),
                    obj_id=obj_id,
                    field="z_part",
                    part_name=part_name,
                )
            )
            name_sequences.append(self._name_seq(part_type))
            anchor, anchor_valid = self._anchor_for_part(
                obj_id,
                angle_idx,
                part_name,
                view_indices,
            )
            anchors.append(anchor)
            anchor_valids.append(anchor_valid)
            coords = self._load_raw_ind_coords(sample, part)
            raw_ind_coords.append(coords)
            target_part_names.append(part_name)
            target_part_types.append(part_type)
            target_parts.append(dict(part["target_part"]))

        part_raw_voxel_counts = torch.tensor(
            [coords.shape[0] for coords in raw_ind_coords],
            dtype=torch.float32,
        )
        part_fg_mask = torch.stack(
            [
                raw_coords_to_part_fg_mask(
                    coords,
                    raw_resolution=64,
                    latent_resolution=16,
                    dilate=self.foreground_mask_dilate,
                )
                for coords in raw_ind_coords
            ],
            dim=0,
        )
        name_tokens, name_mask = self._pad_name_sequences(name_sequences)

        return {
            "x_1_parts": torch.stack(x_1_parts, dim=0),
            "z_global": z_global,
            "cond": cond,
            "name_tokens": name_tokens,
            "name_mask": name_mask,
            "anchor": torch.stack(anchors, dim=0),
            "anchor_valid": torch.stack(anchor_valids, dim=0),
            "part_valid": torch.ones(len(x_1_parts), dtype=torch.bool),
            "part_raw_voxel_counts": part_raw_voxel_counts,
            "part_fg_mask": part_fg_mask,
            "target_part_names": target_part_names,
            "target_part_types": target_part_types,
            "target_parts": target_parts,
            "raw_ind_coords": raw_ind_coords,
            "obj_id": obj_id,
            "angle_idx": angle_idx,
            "sample_id": sample["sample_id"],
            "view_indices": view_indices,
            "image_paths": list(sample.get("image_paths", [])),
        }

    @staticmethod
    def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        max_parts = max(int(sample["x_1_parts"].shape[0]) for sample in batch)
        batch_size = len(batch)
        latent_shape = batch[0]["x_1_parts"].shape[1:]
        name_dim = int(batch[0]["name_tokens"].shape[-1])
        max_name_len = max(int(sample["name_tokens"].shape[1]) for sample in batch)
        num_views = int(batch[0]["anchor"].shape[1])

        x_1_parts = torch.zeros(
            (batch_size, max_parts) + latent_shape,
            dtype=batch[0]["x_1_parts"].dtype,
        )
        part_valid = torch.zeros((batch_size, max_parts), dtype=torch.bool)
        part_raw_voxel_counts = torch.zeros(
            (batch_size, max_parts),
            dtype=torch.float32,
        )
        part_fg_mask = torch.zeros(
            (batch_size, max_parts) + FG_MASK_SHAPE,
            dtype=torch.bool,
        )
        name_tokens = torch.zeros(
            (batch_size, max_parts, max_name_len, name_dim),
            dtype=batch[0]["name_tokens"].dtype,
        )
        name_mask = torch.zeros(
            (batch_size, max_parts, max_name_len),
            dtype=torch.bool,
        )
        anchor = torch.zeros(
            (batch_size, max_parts, num_views, 4),
            dtype=batch[0]["anchor"].dtype,
        )
        anchor_valid = torch.zeros(
            (batch_size, max_parts, num_views),
            dtype=torch.bool,
        )
        for row, sample in enumerate(batch):
            part_count = int(sample["x_1_parts"].shape[0])
            x_1_parts[row, :part_count] = sample["x_1_parts"]
            part_valid[row, :part_count] = sample["part_valid"]
            part_raw_voxel_counts[row, :part_count] = sample["part_raw_voxel_counts"]
            part_fg_mask[row, :part_count] = sample["part_fg_mask"]
            name_len = int(sample["name_tokens"].shape[1])
            name_tokens[row, :part_count, :name_len] = sample["name_tokens"]
            name_mask[row, :part_count, :name_len] = sample["name_mask"]
            anchor[row, :part_count] = sample["anchor"]
            anchor_valid[row, :part_count] = sample["anchor_valid"]

        out: Dict[str, Any] = {
            "x_1_parts": x_1_parts,
            "z_global": torch.stack([sample["z_global"] for sample in batch], dim=0),
            "cond": torch.stack([sample["cond"] for sample in batch], dim=0),
            "name_tokens": name_tokens,
            "name_mask": name_mask,
            "anchor": anchor,
            "anchor_valid": anchor_valid,
            "part_valid": part_valid,
            "part_raw_voxel_counts": part_raw_voxel_counts,
            "part_fg_mask": part_fg_mask,
        }
        for key in (
            "target_part_names",
            "target_part_types",
            "target_parts",
            "raw_ind_coords",
            "obj_id",
            "angle_idx",
            "sample_id",
            "view_indices",
            "image_paths",
        ):
            out[key] = [sample[key] for sample in batch]
        return out
