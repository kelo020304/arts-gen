"""Shared helpers for promptable part latent segmentation experiments."""

from __future__ import annotations

import json
import math
import os
import random
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import torch
import torch.nn.functional as F
from safetensors.torch import load_file
from torch.utils.data import Dataset, Sampler


PROJECT_ROOT = Path(__file__).resolve().parents[3]
TRELLIS_PATH = PROJECT_ROOT / "TRELLIS-arts"
if str(TRELLIS_PATH) not in sys.path:
    sys.path.insert(0, str(TRELLIS_PATH))

import train_arts  # noqa: F401
from trellis.datasets.arts.part_ss_latent_flow import PartSSLatentFlowDataset
from trellis.models.sparse_structure_vae import SparseStructureDecoder, SparseStructureEncoder
from trellis.trainers.arts.part_ss_latent_flow_eval import coords_iou


VEPFS_ROOT = Path(os.environ.get("VEPFS_ROOT", "/robot/data-lab/jzh"))
DATA_ROOT = Path(
    os.environ.get(
        "DATA_ROOT",
        str(
            VEPFS_ROOT
            / "art-gen/data/PhysX-Mobility-full-4view-0511/PhysX-Mobility-full-4view-0511"
        ),
    )
)
MANIFEST_REL = "manifests/part_completion/arts_mllm_physx-mobility.train.jsonl"
OFFICIAL_SPLIT_PATH = Path(
    os.environ.get(
        "SPLIT_JSON",
        str(
            VEPFS_ROOT
            / "art-gen/data/part_promptable_seg_manifests/v6/split_official_verse_realappliance_0511dd_v6.json"
        ),
    )
)
PACKED_DATA_ROOT = Path(
    os.environ.get("PACKED_DIR", str(VEPFS_ROOT / "art-gen/data/part_promptable_seg_packed_v6"))
)
SS_ENCODER_CKPT = Path(
    os.environ.get(
        "PROMPTSEG_SS_ENCODER_CKPT",
        os.environ.get(
            "SS_ENCODER_CKPT",
            str(PROJECT_ROOT / "pretrained/TRELLIS-image-large/ckpts/ss_enc_conv3d_16l8_fp16.safetensors"),
        ),
    )
)
SS_DECODER_CKPT = Path(
    os.environ.get(
        "PROMPTSEG_SS_DECODER_CKPT",
        os.environ.get(
            "SS_DECODER_CKPT",
            str(PROJECT_ROOT / "pretrained/TRELLIS-image-large/ckpts/ss_dec_conv3d_16l8_fp16.safetensors"),
        ),
    )
)
DECODE_THRESHOLD = 0.0


@dataclass(frozen=True)
class PartRow:
    sample_idx: int
    part_idx: int
    obj_id: str
    angle_idx: int
    sample_id: str
    part_name: str
    semantic_type: str
    original_label: int
    raw_count: int
    view_indices: tuple[int, ...]
    dataset_id: str = ""
    data_root: str = ""
    manifest_path: str = ""
    category: str = ""
    object_name: str = ""
    part_item_name: str = ""
    part_joint: str = ""
    sample_part_names: str = ""
    visible_view_count: int = 0


@dataclass(frozen=True)
class DatasetSpec:
    dataset_id: str
    data_root: Path
    manifest_paths: tuple[str, ...]


def _slug_from_path(path: Path) -> str:
    text = re.sub(r"[^A-Za-z0-9_.-]+", "-", path.name.strip())
    return text.strip("-") or "dataset"


def _split_env_list(value: str | None) -> list[str]:
    if value is None:
        return []
    return [item.strip() for item in value.split(":") if item.strip()]


def _default_manifest_paths(data_root: Path) -> tuple[str, ...]:
    standard = data_root / MANIFEST_REL
    if standard.is_file():
        return (MANIFEST_REL,)
    realappliance = data_root / "manifests/part_completion/arts_mllm_realappliance.train.jsonl"
    if realappliance.is_file():
        return ("manifests/part_completion/arts_mllm_realappliance.train.jsonl",)
    generated = sorted(
        data_root.glob("manifests/full_run_logs/batch_*/final_outputs/part_completion/arts_mllm_*.train.jsonl")
    )
    if generated:
        return tuple(str(path) for path in generated)
    return (MANIFEST_REL,)


def parse_dataset_specs_from_env() -> list[DatasetSpec]:
    roots = _split_env_list(os.environ.get("DATA_ROOTS"))
    roots_from_multi_env = bool(roots)
    if not roots:
        roots = [str(DATA_ROOT)]
    ids = _split_env_list(os.environ.get("DATASET_IDS"))
    manifest_groups = _split_env_list(os.environ.get("DATA_MANIFESTS"))
    specs: list[DatasetSpec] = []
    used_ids: set[str] = set()
    for idx, root_text in enumerate(roots):
        data_root = Path(root_text)
        if idx < len(ids):
            dataset_id = ids[idx]
        elif roots_from_multi_env:
            dataset_id = _slug_from_path(data_root)
        else:
            dataset_id = ""
        base_id = dataset_id
        suffix = 2
        while dataset_id in used_ids:
            dataset_id = f"{base_id}-{suffix}"
            suffix += 1
        used_ids.add(dataset_id)
        if idx < len(manifest_groups):
            manifest_paths = tuple(item.strip() for item in manifest_groups[idx].split(",") if item.strip())
        else:
            manifest_paths = _default_manifest_paths(data_root)
        specs.append(DatasetSpec(dataset_id=dataset_id, data_root=data_root, manifest_paths=manifest_paths))
    if not specs:
        raise RuntimeError("no dataset specs configured")
    return specs


def dataset_specs_from_split(split: Mapping[str, Any]) -> list[DatasetSpec]:
    raw_specs = split.get("datasets")
    if isinstance(raw_specs, list) and raw_specs:
        specs = []
        for idx, item in enumerate(raw_specs):
            if not isinstance(item, Mapping):
                raise TypeError(f"split datasets[{idx}] must be an object, got {type(item).__name__}")
            data_root = Path(str(item["data_root"]))
            manifest_value = item.get("manifest_paths", item.get("manifest_path", MANIFEST_REL))
            if isinstance(manifest_value, list):
                manifest_paths = tuple(str(path) for path in manifest_value)
            else:
                manifest_paths = (str(manifest_value),)
            specs.append(
                DatasetSpec(
                    dataset_id=str(item.get("dataset_id") or _slug_from_path(data_root)),
                    data_root=data_root,
                    manifest_paths=manifest_paths,
                )
            )
        return specs
    if "data_root" in split:
        data_root = Path(str(split["data_root"]))
        manifest_path = str(split.get("manifest_path") or MANIFEST_REL)
        return [DatasetSpec(dataset_id=str(split.get("dataset_id") or ""), data_root=data_root, manifest_paths=(manifest_path,))]
    return parse_dataset_specs_from_env()


class MultiPromptableBaseDataset:
    def __init__(self, datasets: Mapping[str, PartSSLatentFlowDataset]) -> None:
        self.datasets = dict(datasets)
        if not self.datasets:
            raise RuntimeError("MultiPromptableBaseDataset got zero datasets")

    def dataset_for_row(self, row: PartRow) -> PartSSLatentFlowDataset:
        dataset_id = row.dataset_id or next(iter(self.datasets))
        if dataset_id not in self.datasets:
            raise KeyError(f"unknown dataset_id={dataset_id!r}; available={sorted(self.datasets)}")
        return self.datasets[dataset_id]


def seed_all(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def make_base_dataset(
    *,
    data_root: Path = DATA_ROOT,
    manifest_path: str | Path | Iterable[str | Path] = MANIFEST_REL,
    include_obj_ids: Iterable[str] | None = None,
) -> PartSSLatentFlowDataset:
    manifest_paths = [manifest_path] if isinstance(manifest_path, (str, Path)) else list(manifest_path)
    if not manifest_paths:
        raise ValueError("manifest_path must contain at least one path")
    data_cfg = {
        "data_root": str(data_root),
        "recon_subdir": "reconstruction",
        "mask_subdir": "renders",
        "manifest_path": str(manifest_paths[0]),
        "num_views": 4,
        "allow_missing_masks": False,
        "require_part_token": False,
        "use_mask_overlap_pooling": False,
        "filter_zero_mask_coverage": False,
    }
    if include_obj_ids is not None:
        data_cfg["include_obj_ids"] = [str(x) for x in include_obj_ids]
    base = PartSSLatentFlowDataset(data_cfg)
    if len(manifest_paths) == 1:
        return base
    for extra_path in manifest_paths[1:]:
        extra_cfg = dict(data_cfg)
        extra_cfg["manifest_path"] = str(extra_path)
        extra = PartSSLatentFlowDataset(extra_cfg)
        base.samples.extend(extra.samples)
        if hasattr(base, "loads") and hasattr(extra, "loads"):
            base.loads.extend(extra.loads)
    print(
        f"[PromptSeg-data] merged {len(manifest_paths)} manifests from {data_root}: "
        f"{len(base.samples)} object samples / {sum(len(sample['parts']) for sample in base.samples)} target parts"
    )
    return base


def make_base_datasets(
    specs: Iterable[DatasetSpec],
    *,
    include_obj_ids_by_dataset: Mapping[str, Iterable[str]] | None = None,
) -> dict[str, PartSSLatentFlowDataset]:
    include_obj_ids_by_dataset = include_obj_ids_by_dataset or {}
    out: dict[str, PartSSLatentFlowDataset] = {}
    for spec in specs:
        out[spec.dataset_id] = make_base_dataset(
            data_root=spec.data_root,
            manifest_path=spec.manifest_paths,
            include_obj_ids=include_obj_ids_by_dataset.get(spec.dataset_id),
        )
    return out


def enumerate_part_rows(base_ds: PartSSLatentFlowDataset, *, dataset_id: str = "") -> list[PartRow]:
    rows: list[PartRow] = []
    for sample_idx, sample in enumerate(base_ds.samples):
        for part_idx, part in enumerate(sample["parts"]):
            raw = base_ds._load_raw_ind_coords(sample, part)
            target_part = part.get("target_part") if isinstance(part.get("target_part"), dict) else {}
            rows.append(
                PartRow(
                    sample_idx=int(sample_idx),
                    part_idx=int(part_idx),
                    obj_id=str(sample["obj_id"]),
                    angle_idx=int(sample["angle_idx"]),
                    sample_id=str(sample["sample_id"]),
                    part_name=str(part["part_name"]),
                    semantic_type=semantic_type(str(part["part_name"]), part),
                    original_label=int(base_ds._part_original_label(sample, part)),
                    raw_count=int(raw.shape[0]),
                    view_indices=tuple(int(v) for v in sample["view_indices"]),
                    dataset_id=str(dataset_id),
                    data_root=str(base_ds.data_root),
                    manifest_path=str(base_ds._manifest_abs()),
                    category=str(sample.get("category", "")),
                    object_name=str(sample.get("name", "")),
                    part_item_name=str(target_part.get("item_name", "")),
                    part_joint=str(target_part.get("joint", target_part.get("joint_type", ""))),
                    sample_part_names=" ".join(str(name) for name in sample.get("target_part_names", [])),
                    visible_view_count=int(target_part.get("visible_view_count", 0) or 0),
                )
            )
    return rows


def enumerate_part_rows_multi(base_datasets: Mapping[str, PartSSLatentFlowDataset]) -> list[PartRow]:
    rows: list[PartRow] = []
    for dataset_id, base_ds in base_datasets.items():
        rows.extend(enumerate_part_rows(base_ds, dataset_id=str(dataset_id)))
    return rows


def bucket_name(raw_count: int) -> str:
    raw_count = int(raw_count)
    if raw_count < 50:
        return "tiny"
    if raw_count < 500:
        return "small"
    if raw_count < 3000:
        return "medium"
    return "large"


def semantic_type(part_name: str, part: dict[str, Any] | None = None) -> str:
    if part is not None:
        value = str(part.get("type") or "")
        if value:
            return value
        target_part = part.get("target_part")
        if isinstance(target_part, dict):
            value = str(target_part.get("type") or "")
            if value:
                return value
    return str(part_name).split("_")[0]


def build_semantic_vocab(rows: Iterable[PartRow]) -> dict[str, int]:
    names = sorted({row.semantic_type for row in rows})
    return {name: idx for idx, name in enumerate(names)}


def raw_coords_to_mask16(coords: torch.Tensor | np.ndarray) -> torch.Tensor:
    coords_t = torch.as_tensor(coords, dtype=torch.long)
    mask = torch.zeros((16, 16, 16), dtype=torch.float32)
    if coords_t.numel() == 0:
        return mask
    latent = torch.div(coords_t.clamp(0, 63), 4, rounding_mode="floor")
    mask[latent[:, 0], latent[:, 1], latent[:, 2]] = 1.0
    return mask


def raw_coords_centroid_extent(coords: torch.Tensor | np.ndarray) -> torch.Tensor:
    coords_t = torch.as_tensor(coords, dtype=torch.float32)
    if coords_t.numel() == 0:
        return torch.zeros((6,), dtype=torch.float32)
    coords_f = coords_t.clamp(0.0, 63.0) / 63.0
    lo = coords_f.min(dim=0).values
    hi = coords_f.max(dim=0).values
    return torch.cat([(lo + hi) * 0.5, (hi - lo).clamp_min(0.0)], dim=0).to(dtype=torch.float32)


def load_angle_ind_union_coords(base_ds: PartSSLatentFlowDataset, sample: Mapping[str, Any]) -> torch.Tensor:
    angle_dir = (
        base_ds.recon_root
        / "voxel_expanded"
        / str(sample["obj_id"])
        / f"angle_{int(sample['angle_idx'])}"
        / "64"
    )
    paths = sorted(angle_dir.glob("ind_*.npy"))
    if not paths:
        raise FileNotFoundError(f"no ind_*.npy files found for whole-occ union: {angle_dir}")
    coords = [torch.from_numpy(np.asarray(np.load(path))).long() for path in paths]
    nonempty = [item for item in coords if item.numel() > 0]
    if not nonempty:
        return torch.empty((0, 3), dtype=torch.long)
    merged = torch.cat(nonempty, dim=0)
    if merged.dim() != 2 or merged.shape[1] != 3:
        raise ValueError(f"{angle_dir} expected ind coords with shape [N,3], got {tuple(merged.shape)}")
    return torch.unique(merged, dim=0)


def boundary_band_mask(mask: torch.Tensor | np.ndarray, *, radius: int = 1) -> torch.Tensor:
    m = torch.as_tensor(mask).bool()
    squeeze_batch = False
    if m.dim() == 3:
        m_work = m.unsqueeze(0)
        squeeze_batch = True
    elif m.dim() == 4:
        m_work = m
    else:
        raise ValueError(f"boundary_band_mask expected [D,H,W] or [B,D,H,W], got {tuple(m.shape)}")
    padded = F.pad(m_work.unsqueeze(1).float(), (1, 1, 1, 1, 1, 1), value=0.0).bool()
    center = padded[:, :, 1:-1, 1:-1, 1:-1]
    boundary = torch.zeros_like(center, dtype=torch.bool)
    for neighbor in (
        padded[:, :, :-2, 1:-1, 1:-1],
        padded[:, :, 2:, 1:-1, 1:-1],
        padded[:, :, 1:-1, :-2, 1:-1],
        padded[:, :, 1:-1, 2:, 1:-1],
        padded[:, :, 1:-1, 1:-1, :-2],
        padded[:, :, 1:-1, 1:-1, 2:],
    ):
        boundary.logical_or_(center != neighbor)
    boundary = boundary.squeeze(1)
    if int(radius) > 0:
        kernel = 2 * int(radius) + 1
        boundary = F.max_pool3d(
            boundary.unsqueeze(1).float(),
            kernel_size=kernel,
            stride=1,
            padding=int(radius),
        ).squeeze(1).bool()
    if squeeze_batch:
        return boundary[0]
    return boundary


def downsample_binary_mask(mask: np.ndarray, target_size: int = 512) -> np.ndarray:
    if mask.ndim != 2:
        raise ValueError(f"expected 2D mask, got {mask.shape}")
    mask = np.asarray(mask > 0, dtype=np.float32)
    h, w = mask.shape
    if h == target_size and w == target_size:
        return mask
    if h % target_size == 0 and w % target_size == 0:
        sh = h // target_size
        sw = w // target_size
        return mask.reshape(target_size, sh, target_size, sw).max(axis=(1, 3)).astype(np.float32, copy=False)
    ten = torch.from_numpy(mask).view(1, 1, h, w)
    pooled = F.adaptive_max_pool2d(ten, output_size=(target_size, target_size))
    return pooled.view(target_size, target_size).numpy().astype(np.float32, copy=False)


_MASK_FILE_RE = re.compile(r"mask_(\d+)\.npy$")


def all_mask_paths_for_sample(
    base_ds: PartSSLatentFlowDataset,
    sample: dict[str, Any],
    *,
    expected_views: int = 12,
) -> list[tuple[int, Path]]:
    obj_id = str(sample["obj_id"])
    angle_dir = f"angle_{int(sample['angle_idx'])}"
    mask_dir = base_ds.mask_root / obj_id / angle_dir / "mask"
    found: dict[int, Path] = {}
    if mask_dir.is_dir():
        for path in mask_dir.glob("mask_*.npy"):
            match = _MASK_FILE_RE.fullmatch(path.name)
            if match is None:
                continue
            found[int(match.group(1))] = path
    if int(expected_views) > 0:
        view_ids = set(range(int(expected_views)))
        view_ids.update(found)
    else:
        view_ids = set(found)
    return [(view_idx, found.get(view_idx, mask_dir / f"mask_{view_idx}.npy")) for view_idx in sorted(view_ids)]


def mask_label_counts_for_sample(
    base_ds: PartSSLatentFlowDataset,
    sample: dict[str, Any],
    *,
    expected_views: int = 12,
) -> tuple[dict[int, dict[int, int]], list[int]]:
    counts_by_view: dict[int, dict[int, int]] = {}
    missing_views: list[int] = []
    for view_idx, path in all_mask_paths_for_sample(base_ds, sample, expected_views=expected_views):
        if not path.is_file():
            counts_by_view[int(view_idx)] = {}
            missing_views.append(int(view_idx))
            continue
        label_map = np.asarray(np.load(path))
        if label_map.ndim != 2:
            raise ValueError(f"{path} expected [H,W] mask, got {label_map.shape}")
        labels, counts = np.unique(label_map, return_counts=True)
        counts_by_view[int(view_idx)] = {
            int(label): int(count)
            for label, count in zip(labels.tolist(), counts.tolist())
        }
    return counts_by_view, missing_views


def audit_promptable_mask_visibility(
    base_ds: PartSSLatentFlowDataset | MultiPromptableBaseDataset,
    rows: Iterable[PartRow],
    *,
    expected_views: int = 12,
) -> dict[str, Any]:
    cache: dict[tuple[str, int], tuple[dict[int, dict[int, int]], list[int]]] = {}
    records: list[dict[str, Any]] = []
    class_counts: dict[str, int] = {}
    for row in rows:
        if isinstance(base_ds, MultiPromptableBaseDataset):
            row_base = base_ds.dataset_for_row(row)
        else:
            row_base = base_ds
        sample_idx = int(row.sample_idx)
        cache_key = (str(row.dataset_id), sample_idx)
        sample = row_base.samples[sample_idx]
        if cache_key not in cache:
            cache[cache_key] = mask_label_counts_for_sample(
                row_base,
                sample,
                expected_views=int(expected_views),
            )
        counts_by_view, missing_views = cache[cache_key]
        if hasattr(row_base, "_part_original_labels"):
            part = sample["parts"][int(row.part_idx)]
            labels = [int(label) for label in row_base._part_original_labels(sample, part)]
        else:
            labels = [int(row.original_label)]
        selected_counts = {
            int(view_idx): int(sum(counts_by_view.get(int(view_idx), {}).get(label, 0) for label in labels))
            for view_idx in row.view_indices
        }
        all_counts = {
            int(view_idx): int(sum(view_counts.get(label, 0) for label in labels))
            for view_idx, view_counts in sorted(counts_by_view.items())
        }
        selected_nonempty = [view_idx for view_idx, count in selected_counts.items() if count > 0]
        all_nonempty = [view_idx for view_idx, count in all_counts.items() if count > 0]
        selected_pixels = int(sum(selected_counts.values()))
        all_pixels = int(sum(all_counts.values()))
        if selected_pixels > 0:
            cls = "visible_selected_views"
        elif all_pixels > 0:
            cls = "undetectable_selected_views"
        else:
            cls = "undetectable_all_views"
        class_counts[cls] = class_counts.get(cls, 0) + 1
        records.append({
            "key": part_row_key(row),
            "classification": cls,
            "dataset_id": row.dataset_id,
            "obj_id": row.obj_id,
            "angle_idx": int(row.angle_idx),
            "sample_id": row.sample_id,
            "part_name": row.part_name,
            "part_idx": int(row.part_idx),
            "semantic_type": row.semantic_type,
            "original_label": int(row.original_label),
            "prompt_original_labels": labels,
            "raw_count": int(row.raw_count),
            "selected_view_indices": [int(v) for v in row.view_indices],
            "selected_visible_pixels": selected_pixels,
            "selected_visible_by_view": selected_counts,
            "selected_nonempty_views": selected_nonempty,
            "all_visible_pixels": all_pixels,
            "all_visible_by_view": all_counts,
            "all_nonempty_views": all_nonempty,
            "missing_mask_views": [int(v) for v in missing_views],
        })

    total = len(records)
    class_ratios = {
        name: float(count / total) if total else 0.0
        for name, count in sorted(class_counts.items())
    }
    return {
        "expected_views": int(expected_views),
        "total_rows": total,
        "class_counts": dict(sorted(class_counts.items())),
        "class_ratios": class_ratios,
        "records": records,
    }


def _part_row_key_text(row: PartRow) -> str:
    prefix = f"{row.dataset_id}::" if row.dataset_id else ""
    return f"{prefix}{row.obj_id}|{int(row.angle_idx)}|{row.part_name}"


def _object_key_text(row: PartRow) -> str:
    return f"{row.dataset_id}::{row.obj_id}" if row.dataset_id else row.obj_id


def is_realappliance_row(row: Any) -> bool:
    text = " ".join(
        str(getattr(row, name, "") or "")
        for name in ("dataset_id", "data_root", "manifest_path", "category")
    ).lower()
    return "realappliance" in text or "real appliance" in text


_APPLIANCE_FOCUS_TERMS = (
    "appliance",
    "kitchen",
    "kitchenware",
    "bathroom fixture",
    "fixture",
    "microwave",
    "oven",
    "refrigerator",
    "fridge",
    "washer",
    "washing",
    "dryer",
    "dishwasher",
    "toaster",
    "faucet",
)
_ARTICULATED_FOCUS_TERMS = ("door", "drawer", "lid")


def is_verse_focus_row(row: Any) -> bool:
    dataset_text = " ".join(
        str(getattr(row, name, "") or "")
        for name in ("dataset_id", "data_root", "manifest_path")
    ).lower()
    if "phyx-verse" not in dataset_text and "phyx_verse" not in dataset_text and "verse" not in dataset_text:
        return False
    category_text = " ".join(
        str(getattr(row, name, "") or "")
        for name in ("category", "object_name")
    ).lower()
    part_text = " ".join(
        str(getattr(row, name, "") or "")
        for name in ("part_name", "semantic_type", "part_item_name", "sample_part_names")
    ).lower()
    return (
        any(term in category_text for term in _APPLIANCE_FOCUS_TERMS)
        or any(term in part_text for term in _ARTICULATED_FOCUS_TERMS)
    )


def sampling_tier(row: Any) -> str:
    if is_realappliance_row(row):
        return "realappliance"
    if is_verse_focus_row(row):
        return "verse_focus"
    return "base"


def is_small_prompt_row(row: Any) -> bool:
    part_text = f"{getattr(row, 'part_name', '')} {getattr(row, 'semantic_type', '')}".lower()
    return int(getattr(row, "raw_count", 0)) < 50 or "button" in part_text


def oversample_repeat_for_row(
    row: Any,
    *,
    realappliance_oversample: int = 1,
    verse_focus_oversample: int = 1,
    small_oversample: int = 1,
) -> int:
    tier = sampling_tier(row)
    if tier == "realappliance":
        repeat = max(1, int(realappliance_oversample))
    elif tier == "verse_focus":
        repeat = max(1, int(verse_focus_oversample))
    else:
        repeat = 1
    if is_small_prompt_row(row):
        repeat *= max(1, int(small_oversample))
    return int(repeat)


def build_oversampling_plan(
    rows: Iterable[Any],
    *,
    small_oversample: int = 2,
    realappliance_oversample: int = 0,
    realappliance_target_share: float = 0.22,
    realappliance_max_oversample: int = 8,
    verse_focus_oversample: int = 2,
) -> dict[str, Any]:
    rows_list = list(rows)
    small_oversample = max(1, int(small_oversample))
    verse_focus_oversample = max(1, int(verse_focus_oversample))
    realappliance_max_oversample = max(1, int(realappliance_max_oversample))

    def non_ra_effective() -> int:
        total = 0
        for row in rows_list:
            if sampling_tier(row) == "realappliance":
                continue
            total += oversample_repeat_for_row(
                row,
                realappliance_oversample=1,
                verse_focus_oversample=verse_focus_oversample,
                small_oversample=small_oversample,
            )
        return int(total)

    ra_small_weight = sum(
        small_oversample if is_small_prompt_row(row) else 1
        for row in rows_list
        if sampling_tier(row) == "realappliance"
    )
    non_ra_weight = non_ra_effective()
    if int(realappliance_oversample) > 0:
        ra_repeat = max(1, int(realappliance_oversample))
        auto_capped = False
    elif ra_small_weight > 0 and float(realappliance_target_share) > 0:
        target = min(max(float(realappliance_target_share), 0.0), 0.95)
        needed = math.ceil((target * non_ra_weight) / max((1.0 - target) * ra_small_weight, 1.0e-9))
        ra_repeat = max(1, int(needed))
        auto_capped = ra_repeat > realappliance_max_oversample
        ra_repeat = min(ra_repeat, realappliance_max_oversample)
    else:
        ra_repeat = 1
        auto_capped = False

    repeat_by_key: dict[str, int] = {}
    tier_rows: dict[str, dict[str, Any]] = {
        "realappliance": {"tier": "realappliance", "rows": 0, "objects": set(), "small_rows": 0, "base_repeat": ra_repeat, "effective_rows": 0},
        "verse_focus": {"tier": "verse_focus", "rows": 0, "objects": set(), "small_rows": 0, "base_repeat": verse_focus_oversample, "effective_rows": 0},
        "base": {"tier": "base", "rows": 0, "objects": set(), "small_rows": 0, "base_repeat": 1, "effective_rows": 0},
    }
    total_effective = 0
    for row in rows_list:
        tier = sampling_tier(row)
        repeat = oversample_repeat_for_row(
            row,
            realappliance_oversample=ra_repeat,
            verse_focus_oversample=verse_focus_oversample,
            small_oversample=small_oversample,
        )
        repeat_by_key[_part_row_key_text(row)] = int(repeat)
        info = tier_rows[tier]
        info["rows"] += 1
        info["objects"].add(_object_key_text(row))
        if is_small_prompt_row(row):
            info["small_rows"] += 1
        info["effective_rows"] += int(repeat)
        total_effective += int(repeat)

    table = []
    for tier in ("realappliance", "verse_focus", "base"):
        info = tier_rows[tier]
        effective = int(info["effective_rows"])
        table.append({
            "tier": tier,
            "rows": int(info["rows"]),
            "objects": len(info["objects"]),
            "small_rows": int(info["small_rows"]),
            "base_repeat": int(info["base_repeat"]),
            "effective_rows": effective,
            "effective_share": float(effective / max(1, total_effective)),
        })
        info["objects"] = len(info["objects"])

    return {
        "small_oversample": int(small_oversample),
        "realappliance_oversample": int(ra_repeat),
        "realappliance_target_share": float(realappliance_target_share),
        "realappliance_auto_capped": bool(auto_capped),
        "realappliance_max_oversample": int(realappliance_max_oversample),
        "verse_focus_oversample": int(verse_focus_oversample),
        "rows": len(rows_list),
        "effective_rows": int(total_effective),
        "repeat_by_key": repeat_by_key,
        "tiers": table,
    }


class PromptablePartDataset(Dataset):
    def __init__(
        self,
        base_ds: PartSSLatentFlowDataset | MultiPromptableBaseDataset,
        rows: list[PartRow],
        *,
        mask_size: int = 512,
        semantic_vocab: dict[str, int] | None = None,
        include_whole_coords: bool = False,
    ) -> None:
        self.base_ds = base_ds
        self.rows = list(rows)
        self.mask_size = int(mask_size)
        self.semantic_vocab = dict(semantic_vocab or {})
        self.include_whole_coords = bool(include_whole_coords)
        if not self.rows:
            raise RuntimeError("PromptablePartDataset got zero rows")

    def __len__(self) -> int:
        return len(self.rows)

    @staticmethod
    def _normalize_original_labels(original_label: int | Iterable[int]) -> np.ndarray:
        if isinstance(original_label, (int, np.integer)):
            labels = [int(original_label)]
        else:
            labels = [int(label) for label in original_label]
        if not labels:
            raise ValueError("expected at least one original label for prompt mask")
        return np.asarray(labels, dtype=np.int64)

    def _load_masks2d(self, sample: dict[str, Any], original_label: int | Iterable[int]) -> torch.Tensor:
        if isinstance(self.base_ds, MultiPromptableBaseDataset):
            raise TypeError("_load_masks2d requires a concrete PartSSLatentFlowDataset")
        labels = self._normalize_original_labels(original_label)
        views = []
        for mask_path in self.base_ds._iter_mask_paths(sample):
            label_map = np.asarray(np.load(mask_path))
            views.append(downsample_binary_mask(np.isin(label_map, labels), self.mask_size))
        return torch.from_numpy(np.stack(views, axis=0)).float()

    def _load_masks2d_from_base(
        self,
        base_ds: PartSSLatentFlowDataset,
        sample: dict[str, Any],
        original_label: int | Iterable[int],
    ) -> torch.Tensor:
        labels = self._normalize_original_labels(original_label)
        views = []
        for mask_path in base_ds._iter_mask_paths(sample):
            label_map = np.asarray(np.load(mask_path))
            views.append(downsample_binary_mask(np.isin(label_map, labels), self.mask_size))
        return torch.from_numpy(np.stack(views, axis=0)).float()

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row = self.rows[int(idx)]
        if isinstance(self.base_ds, MultiPromptableBaseDataset):
            base_ds = self.base_ds.dataset_for_row(row)
        else:
            base_ds = self.base_ds
        sample = base_ds.samples[row.sample_idx]
        part = sample["parts"][row.part_idx]
        if hasattr(base_ds, "_part_original_labels"):
            prompt_original_labels = [int(label) for label in base_ds._part_original_labels(sample, part)]
        else:
            prompt_original_labels = [int(row.original_label)]
        raw_coords = base_ds._load_raw_ind_coords(sample, part)
        z_global = base_ds._load_dense_latent(
            base_ds._rooted(sample["z_global_rel"]),
            obj_id=row.obj_id,
            field="z_global",
        )
        z_part = base_ds._load_dense_latent(
            base_ds._rooted(part["z_part_rel"]),
            obj_id=row.obj_id,
            field="z_part",
            part_name=row.part_name,
        )
        out = {
            "z_global": z_global,
            "latent_gt": z_part,
            "masks2d": self._load_masks2d_from_base(base_ds, sample, prompt_original_labels),
            "m_gt": raw_coords_to_mask16(raw_coords),
            "raw_coords": raw_coords,
            "raw_count": torch.tensor(float(row.raw_count), dtype=torch.float32),
            "obj_id": row.obj_id,
            "dataset_id": row.dataset_id,
            "angle_idx": row.angle_idx,
            "sample_id": row.sample_id,
            "part_name": row.part_name,
            "semantic_type": row.semantic_type,
            "semantic_type_id": torch.tensor(self.semantic_vocab.get(row.semantic_type, -1), dtype=torch.long),
            "part_idx": row.part_idx,
            "original_label": row.original_label,
            "view_indices": torch.tensor(row.view_indices, dtype=torch.long),
            "data_root": row.data_root,
            "manifest_path": row.manifest_path,
            "category": row.category,
            "object_name": row.object_name,
            "part_item_name": row.part_item_name,
            "part_joint": row.part_joint,
        }
        out["m_boundary"] = boundary_band_mask(out["m_gt"], radius=1).to(dtype=torch.uint8)
        if self.include_whole_coords:
            out["whole_coords"] = load_angle_ind_union_coords(base_ds, sample)
        return out


class PackedPromptablePartDataset(Dataset):
    """Shard-backed version of PromptablePartDataset for queue training.

    The packed format is intentionally plain torch serialization:
    ``index.json`` maps each row to ``shard_XXXXXX.pt`` and an item offset.
    Each shard stores a list of already materialized sample dicts.
    """

    def __init__(
        self,
        packed_dir: Path,
        rows: list[PartRow],
        *,
        semantic_vocab: dict[str, int] | None = None,
    ) -> None:
        self.packed_dir = Path(packed_dir)
        index_path = self.packed_dir / "index.json"
        if not index_path.is_file():
            raise FileNotFoundError(f"packed index not found: {index_path}")
        index = json.loads(index_path.read_text(encoding="utf-8"))
        if int(index.get("format_version", 0)) != 1:
            raise ValueError(f"{index_path} unsupported format_version={index.get('format_version')!r}")
        self.entries_by_key = {
            str(entry["key"]): entry
            for entry in index.get("entries", [])
        }
        self.rows = list(rows)
        self.semantic_vocab = dict(semantic_vocab or {})
        if not self.rows:
            raise RuntimeError("PackedPromptablePartDataset got zero rows")
        missing = [part_row_key(row) for row in self.rows if part_row_key(row) not in self.entries_by_key]
        if missing:
            preview = ", ".join(missing[:5])
            raise FileNotFoundError(f"packed dataset is missing {len(missing)} requested rows; first: {preview}")
        self.entries_for_rows = [self.entries_by_key[part_row_key(row)] for row in self.rows]
        self._cached_shard_name: str | None = None
        self._cached_items: list[dict[str, Any]] | None = None

    def __len__(self) -> int:
        return len(self.rows)

    def _load_shard(self, shard_name: str) -> list[dict[str, Any]]:
        if shard_name != self._cached_shard_name:
            shard_path = self.packed_dir / shard_name
            if not shard_path.is_file():
                raise FileNotFoundError(f"packed shard not found: {shard_path}")
            payload = torch.load(shard_path, map_location="cpu", weights_only=False)
            if not isinstance(payload, list):
                raise ValueError(f"{shard_path} expected a list of samples, got {type(payload).__name__}")
            self._cached_shard_name = shard_name
            self._cached_items = payload
        if self._cached_items is None:
            raise RuntimeError("packed shard cache was not populated")
        return self._cached_items

    def __getitem__(self, idx: int) -> dict[str, Any]:
        entry = self.entries_for_rows[int(idx)]
        items = self._load_shard(str(entry["shard"]))
        sample = dict(items[int(entry["index"])])
        sample["z_global"] = sample["z_global"].float()
        sample["latent_gt"] = sample["latent_gt"].float()
        sample["masks2d"] = sample["masks2d"].float()
        sample["m_gt"] = sample["m_gt"].float()
        if "m_boundary" in sample:
            sample["m_boundary"] = sample["m_boundary"].float()
        else:
            sample["m_boundary"] = boundary_band_mask(sample["m_gt"], radius=1).float()
        sample["raw_coords"] = sample["raw_coords"].long()
        if "whole_coords" in sample:
            sample["whole_coords"] = sample["whole_coords"].long()
        sample["view_indices"] = sample["view_indices"].long()
        sample["raw_count"] = torch.tensor(float(sample["raw_count"]), dtype=torch.float32)
        sample.setdefault("dataset_id", entry.get("dataset_id", ""))
        row = self.rows[int(idx)]
        sample.setdefault("data_root", row.data_root)
        sample.setdefault("manifest_path", row.manifest_path)
        sample.setdefault("category", row.category)
        sample.setdefault("object_name", row.object_name)
        sample.setdefault("part_item_name", row.part_item_name)
        sample.setdefault("part_joint", row.part_joint)
        sample["semantic_type_id"] = torch.tensor(
            self.semantic_vocab.get(str(sample["semantic_type"]), -1),
            dtype=torch.long,
        )
        return sample


class PackedShardBatchSampler(Sampler[list[int]]):
    """Yield single-shard batches to avoid repeatedly loading large packed shards."""

    def __init__(
        self,
        dataset: PackedPromptablePartDataset,
        *,
        batch_size: int,
        shuffle: bool,
        seed: int = 0,
        drop_last: bool = False,
        num_replicas: int = 1,
        rank: int = 0,
        small_oversample: int = 1,
        repeat_by_key: Mapping[str, int] | None = None,
    ) -> None:
        self.dataset = dataset
        self.batch_size = int(batch_size)
        self.shuffle = bool(shuffle)
        self.seed = int(seed)
        self.drop_last = bool(drop_last)
        self.num_replicas = int(num_replicas)
        self.rank = int(rank)
        self.small_oversample = max(1, int(small_oversample))
        self.repeat_by_key = {str(key): max(1, int(value)) for key, value in dict(repeat_by_key or {}).items()}
        self.epoch = 0
        if self.batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {self.batch_size}")
        if self.num_replicas <= 0:
            raise ValueError(f"num_replicas must be positive, got {self.num_replicas}")
        if not 0 <= self.rank < self.num_replicas:
            raise ValueError(f"rank={self.rank} must be in [0, {self.num_replicas})")
        self.shard_to_indices: dict[str, list[int]] = {}
        for idx, entry in enumerate(dataset.entries_for_rows):
            row = dataset.rows[idx]
            repeat = self.repeat_by_key.get(str(entry.get("key", "")))
            if repeat is None:
                repeat = oversample_repeat_for_row(
                    row,
                    realappliance_oversample=1,
                    verse_focus_oversample=1,
                    small_oversample=self.small_oversample,
                )
            self.shard_to_indices.setdefault(str(entry["shard"]), []).extend([idx] * int(repeat))
        if not self.shard_to_indices:
            raise RuntimeError("PackedShardBatchSampler got zero shards")

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def _all_batches(self) -> list[list[int]]:
        rng = random.Random(self.seed + self.epoch)
        shard_names = sorted(self.shard_to_indices)
        if self.shuffle:
            rng.shuffle(shard_names)
        batches: list[list[int]] = []
        for shard_name in shard_names:
            indices = list(self.shard_to_indices[shard_name])
            if self.shuffle:
                rng.shuffle(indices)
            for start in range(0, len(indices), self.batch_size):
                batch = indices[start : start + self.batch_size]
                if len(batch) < self.batch_size and self.drop_last:
                    continue
                batches.append(batch)
        return batches

    def __iter__(self):
        batches = self._all_batches()
        if self.num_replicas > 1:
            if self.drop_last:
                total = (len(batches) // self.num_replicas) * self.num_replicas
                batches = batches[:total]
            else:
                total = math.ceil(len(batches) / self.num_replicas) * self.num_replicas
                if len(batches) < total:
                    if not batches:
                        raise RuntimeError("PackedShardBatchSampler has no batches")
                    repeats = [list(batches[i % len(batches)]) for i in range(total - len(batches))]
                    batches = [*batches, *repeats]
            batches = batches[self.rank :: self.num_replicas]
        return iter(batches)

    def __len__(self) -> int:
        count = len(self._all_batches())
        if self.num_replicas <= 1:
            return count
        if self.drop_last:
            return count // self.num_replicas
        return int(math.ceil(count / self.num_replicas))


def part_row_key(row: PartRow) -> str:
    prefix = f"{row.dataset_id}::" if row.dataset_id else ""
    return f"{prefix}{row.obj_id}|{int(row.angle_idx)}|{row.part_name}"


def object_key(row: PartRow) -> str:
    return f"{row.dataset_id}::{row.obj_id}" if row.dataset_id else row.obj_id


def split_ref_matches_row(ref: Any, row: PartRow) -> bool:
    if isinstance(ref, Mapping):
        dataset_id = ref.get("dataset_id")
        if dataset_id is not None and str(dataset_id) != str(row.dataset_id):
            return False
        if "object_key" in ref:
            return str(ref["object_key"]) == object_key(row)
        if "obj_id" in ref:
            return str(ref["obj_id"]) == row.obj_id
        raise KeyError(f"split object ref must contain obj_id or object_key, got {ref}")
    text = str(ref)
    if "::" in text:
        return text == object_key(row)
    return text == row.obj_id


def collate_promptable_parts(batch: list[dict[str, Any]]) -> dict[str, Any]:
    out = {
        "z_global": torch.stack([item["z_global"] for item in batch], dim=0),
        "latent_gt": torch.stack([item["latent_gt"] for item in batch], dim=0),
        "masks2d": torch.stack([item["masks2d"] for item in batch], dim=0),
        "m_gt": torch.stack([item["m_gt"] for item in batch], dim=0),
        "m_boundary": torch.stack([item["m_boundary"] for item in batch], dim=0),
        "raw_count": torch.stack([item["raw_count"] for item in batch], dim=0),
        "view_indices": torch.stack([item["view_indices"] for item in batch], dim=0),
        "raw_coords": [item["raw_coords"] for item in batch],
        "raw_geom": torch.stack([raw_coords_centroid_extent(item["raw_coords"]) for item in batch], dim=0),
        "obj_id": [item["obj_id"] for item in batch],
        "dataset_id": [item.get("dataset_id", "") for item in batch],
        "angle_idx": [item["angle_idx"] for item in batch],
        "sample_id": [item["sample_id"] for item in batch],
        "part_name": [item["part_name"] for item in batch],
        "semantic_type": [item["semantic_type"] for item in batch],
        "semantic_type_id": torch.stack([item["semantic_type_id"] for item in batch], dim=0),
        "part_idx": [item["part_idx"] for item in batch],
        "original_label": [item["original_label"] for item in batch],
        "data_root": [str(item.get("data_root", "")) for item in batch],
        "manifest_path": [str(item.get("manifest_path", "")) for item in batch],
        "category": [str(item.get("category", "")) for item in batch],
        "object_name": [str(item.get("object_name", "")) for item in batch],
        "part_item_name": [str(item.get("part_item_name", "")) for item in batch],
        "part_joint": [str(item.get("part_joint", "")) for item in batch],
    }
    if all("whole_coords" in item for item in batch):
        out["whole_coords"] = [item["whole_coords"] for item in batch]
    if any("motion_valid" in item for item in batch):
        out["motion_valid"] = torch.stack([
            torch.as_tensor(item.get("motion_valid", False), dtype=torch.bool)
            for item in batch
        ], dim=0)
        out["motion_angle_b"] = torch.stack([
            torch.as_tensor(item.get("motion_angle_b", -1), dtype=torch.long)
            for item in batch
        ], dim=0)
        out["motion_joint_type"] = [str(item.get("motion_joint_type", "")) for item in batch]
        out["motion_transform_ab"] = torch.stack([
            item.get("motion_transform_ab", torch.eye(4, dtype=torch.float32)).float()
            for item in batch
        ], dim=0)
        out["motion_target_coords_b"] = [
            item.get("motion_target_coords_b", torch.empty((0, 3), dtype=torch.long)).long()
            for item in batch
        ]
        out["motion_scale_a"] = torch.stack([
            torch.as_tensor(item.get("motion_scale_a", 1.0), dtype=torch.float32)
            for item in batch
        ], dim=0)
        out["motion_offset_a"] = torch.stack([
            item.get("motion_offset_a", torch.zeros((3,), dtype=torch.float32)).float()
            for item in batch
        ], dim=0)
        out["motion_scale_b"] = torch.stack([
            torch.as_tensor(item.get("motion_scale_b", 1.0), dtype=torch.float32)
            for item in batch
        ], dim=0)
        out["motion_offset_b"] = torch.stack([
            item.get("motion_offset_b", torch.zeros((3,), dtype=torch.float32)).float()
            for item in batch
        ], dim=0)
    return out


def load_official_split(path: Path = OFFICIAL_SPLIT_PATH) -> dict[str, Any]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"official split not found: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    train_key = "train_keys" if "train_keys" in data else "train_ids"
    heldout_key = "heldout_keys" if "heldout_keys" in data else "heldout_ids"
    for key in (train_key, heldout_key):
        if key not in data:
            raise KeyError(f"{path} missing {key!r}")
        if not isinstance(data[key], list) or not data[key]:
            raise ValueError(f"{path} {key!r} must be a non-empty list")
    overlap = set(map(str, data[train_key])) & set(map(str, data[heldout_key]))
    if overlap:
        raise ValueError(f"{path} {train_key} and {heldout_key} overlap: {sorted(overlap)[:10]}")
    return data


def rows_for_obj_ids(rows: list[PartRow], obj_ids: Iterable[str]) -> list[PartRow]:
    wanted = list(obj_ids)
    wanted_object_keys: set[str] = set()
    wanted_obj_ids: set[str] = set()
    wanted_pairs: set[tuple[str, str]] = set()
    for ref in wanted:
        if isinstance(ref, Mapping):
            if "object_key" in ref:
                wanted_object_keys.add(str(ref["object_key"]))
            elif "obj_id" in ref and "dataset_id" in ref:
                wanted_pairs.add((str(ref["dataset_id"]), str(ref["obj_id"])))
            elif "obj_id" in ref:
                wanted_obj_ids.add(str(ref["obj_id"]))
            else:
                raise KeyError(f"split object ref must contain obj_id or object_key, got {ref}")
            continue
        text = str(ref)
        if "::" in text:
            wanted_object_keys.add(text)
        else:
            wanted_obj_ids.add(text)
    selected = [
        row
        for row in rows
        if object_key(row) in wanted_object_keys
        or row.obj_id in wanted_obj_ids
        or (row.dataset_id, row.obj_id) in wanted_pairs
    ]
    available_object_keys = {object_key(row) for row in selected}
    available_obj_ids = {row.obj_id for row in selected}
    available_pairs = {(row.dataset_id, row.obj_id) for row in selected}
    missing = []
    for ref in wanted:
        if isinstance(ref, Mapping):
            if "object_key" in ref:
                present = str(ref["object_key"]) in available_object_keys
            elif "obj_id" in ref and "dataset_id" in ref:
                present = (str(ref["dataset_id"]), str(ref["obj_id"])) in available_pairs
            elif "obj_id" in ref:
                present = str(ref["obj_id"]) in available_obj_ids
            else:
                present = False
        else:
            text = str(ref)
            present = text in available_object_keys if "::" in text else text in available_obj_ids
        if not present:
            missing.append(str(ref))
    if missing:
        raise RuntimeError(f"split references {len(missing)} object ids not present in rows; first: {missing[:10]}")
    return selected


def split_rows_by_obj(
    rows: list[PartRow],
    *,
    heldout_fraction: float = 0.2,
    seed: int = 20260611,
) -> tuple[list[PartRow], list[PartRow]]:
    obj_ids = sorted({object_key(row) for row in rows})
    rng = random.Random(int(seed))
    rng.shuffle(obj_ids)
    heldout_count = max(1, int(round(len(obj_ids) * float(heldout_fraction))))
    heldout = set(obj_ids[:heldout_count])
    train = [row for row in rows if object_key(row) not in heldout]
    val = [row for row in rows if object_key(row) in heldout]
    if {object_key(row) for row in train} & {object_key(row) for row in val}:
        raise RuntimeError("object split overlap detected")
    return train, val


def pick_gate1_obj_ids(rows: list[PartRow]) -> list[str]:
    required = ["102276"]
    preferred = ["100283", "101943", "102701", "101049", "101106", "101253", "100279", "100058"]
    obj_to_rows: dict[str, list[PartRow]] = {}
    for row in rows:
        obj_to_rows.setdefault(object_key(row), []).append(row)
    chosen: list[str] = []
    for obj_id in [*required, *preferred]:
        if obj_id in obj_to_rows and obj_id not in chosen:
            chosen.append(obj_id)
    multi_small = [
        obj_id
        for obj_id, obj_rows in obj_to_rows.items()
        if obj_id not in chosen and len(obj_rows) >= 3 and any(row.raw_count < 500 for row in obj_rows)
    ]
    multi_small.sort(key=lambda oid: (min(row.raw_count for row in obj_to_rows[oid]), -len(obj_to_rows[oid])))
    for obj_id in multi_small:
        if len(chosen) >= 9:
            break
        chosen.append(obj_id)
    single = [obj_id for obj_id, obj_rows in obj_to_rows.items() if len(obj_rows) == 1 and obj_id not in chosen]
    if single:
        chosen.append(single[0])
    for obj_id in sorted(obj_to_rows):
        if len(chosen) >= 10:
            break
        if obj_id not in chosen:
            chosen.append(obj_id)
    return chosen[:10]


def pick_gate1_rows(rows: list[PartRow]) -> tuple[list[PartRow], list[dict[str, Any]]]:
    obj_ids = pick_gate1_obj_ids(rows)
    preferred_angles = {
        "100283": 0,
        "101943": 0,
        "102701": 0,
        "101049": 5,
        "101106": 0,
        "101253": 9,
        "100279": 0,
        "100058": 0,
        "102276": 0,
    }
    out: list[PartRow] = []
    meta: list[dict[str, Any]] = []
    for obj_id in obj_ids:
        obj_rows = [row for row in rows if object_key(row) == obj_id]
        angles = sorted({row.angle_idx for row in obj_rows})
        angle = preferred_angles.get(obj_id)
        if angle not in angles:
            angle = max(
                angles,
                key=lambda a: (
                    sum(1 for row in obj_rows if row.angle_idx == a and row.raw_count < 50),
                    sum(1 for row in obj_rows if row.angle_idx == a),
                ),
            )
        chosen = [row for row in obj_rows if row.angle_idx == angle]
        out.extend(chosen)
        meta.append(
            {
                "obj_id": obj_id,
                "dataset_id": chosen[0].dataset_id if chosen else "",
                "angle_idx": int(angle),
                "parts": len(chosen),
                "min_raw_count": int(min(row.raw_count for row in chosen)),
                "tiny_parts": int(sum(row.raw_count < 50 for row in chosen)),
                "part_names": [row.part_name for row in chosen],
            }
        )
    return out, meta


def ckpt_paths(path: Path) -> tuple[Path, Path]:
    if path.suffix == ".safetensors":
        weights = path
        config = path.with_suffix(".json")
    else:
        weights = path.with_suffix(".safetensors")
        config = path.with_suffix(".json")
    if not config.is_file():
        raise FileNotFoundError(f"config not found: {config}")
    if not weights.is_file():
        raise FileNotFoundError(f"weights not found: {weights}")
    return config, weights


def load_ss_encoder(path: Path = SS_ENCODER_CKPT, *, device: torch.device, fp32: bool = True) -> SparseStructureEncoder:
    config_path, weights_path = ckpt_paths(path)
    cfg = json.loads(config_path.read_text(encoding="utf-8"))
    if cfg.get("name") != "SparseStructureEncoder":
        raise ValueError(f"{config_path} expected SparseStructureEncoder, got {cfg.get('name')!r}")
    model = SparseStructureEncoder(**cfg["args"]).to(device).eval()
    model.load_state_dict(load_file(str(weights_path), device=str(device)), strict=True)
    if fp32 and getattr(model, "use_fp16", False):
        model.convert_to_fp32()
    for param in model.parameters():
        param.requires_grad_(False)
    return model


def load_ss_decoder(path: Path = SS_DECODER_CKPT, *, device: torch.device, fp32: bool = True) -> SparseStructureDecoder:
    config_path, weights_path = ckpt_paths(path)
    cfg = json.loads(config_path.read_text(encoding="utf-8"))
    if cfg.get("name") != "SparseStructureDecoder":
        raise ValueError(f"{config_path} expected SparseStructureDecoder, got {cfg.get('name')!r}")
    model = SparseStructureDecoder(**cfg["args"]).to(device).eval()
    model.load_state_dict(load_file(str(weights_path), device=str(device)), strict=True)
    if fp32 and getattr(model, "use_fp16", False):
        model.convert_to_fp32()
    for param in model.parameters():
        param.requires_grad_(False)
    return model


@torch.no_grad()
def compute_empty_code(encoder: SparseStructureEncoder, *, device: torch.device) -> torch.Tensor:
    grid = torch.zeros((1, 1, 64, 64, 64), dtype=torch.float32, device=device)
    return encoder(grid, sample_posterior=False)[0].detach().float().cpu()


def dense_occ_from_coords(coords_list: list[torch.Tensor | np.ndarray], *, device: torch.device) -> torch.Tensor:
    out = torch.zeros((len(coords_list), 1, 64, 64, 64), dtype=torch.float32, device=device)
    for idx, coords in enumerate(coords_list):
        coords_t = torch.as_tensor(coords, dtype=torch.long, device=device)
        if coords_t.numel() == 0:
            continue
        out[idx, 0, coords_t[:, 0], coords_t[:, 1], coords_t[:, 2]] = 1.0
    return out


def mask_metrics_from_logits(logits: torch.Tensor, target: torch.Tensor, threshold: float = 0.5) -> dict[str, float]:
    prob = logits.sigmoid() if logits.dtype.is_floating_point else logits.float()
    pred = prob > float(threshold)
    gt = target.bool()
    inter = (pred & gt).sum(dim=1).float()
    union = (pred | gt).sum(dim=1).float()
    pred_count = pred.sum(dim=1).float()
    gt_count = gt.sum(dim=1).float()
    iou = torch.where(union > 0, inter / union.clamp_min(1.0), torch.ones_like(union))
    precision = torch.where(pred_count > 0, inter / pred_count.clamp_min(1.0), (gt_count == 0).float())
    recall = torch.where(gt_count > 0, inter / gt_count.clamp_min(1.0), torch.ones_like(gt_count))
    return {
        "cell_iou": float(iou.mean().detach().item()),
        "cell_precision": float(precision.mean().detach().item()),
        "cell_recall": float(recall.mean().detach().item()),
        "cell_pred_count": float(pred_count.mean().detach().item()),
        "cell_gt_count": float(gt_count.mean().detach().item()),
    }


@torch.no_grad()
def decode_latents_to_coords(decoder: SparseStructureDecoder, latents: torch.Tensor, threshold: float = DECODE_THRESHOLD) -> list[torch.Tensor]:
    device = next(decoder.parameters()).device
    logits = decoder(latents.to(device=device, dtype=torch.float32)).float()
    occ = logits[:, 0] > float(threshold)
    return [torch.nonzero(occ[idx], as_tuple=False).long().cpu() for idx in range(occ.shape[0])]


def decode_metrics_for_batch(pred_coords: list[torch.Tensor], raw_coords: list[torch.Tensor | np.ndarray]) -> list[dict[str, float]]:
    rows = []
    for pred, raw in zip(pred_coords, raw_coords):
        metric = coords_iou(pred, raw)
        rows.append(
            {
                "decode_iou": float(metric["iou"]),
                "decode_precision": float(metric["precision"]),
                "decode_recall": float(metric["recall"]),
                "pred_count": int(metric["pred_count"]),
                "raw_count": int(metric["gt_count"]),
            }
        )
    return rows


def summarize_rows(rows: list[dict[str, Any]], value_keys: tuple[str, ...]) -> dict[str, Any]:
    out: dict[str, Any] = {"n": len(rows)}
    for key in value_keys:
        values = [float(row[key]) for row in rows if key in row]
        out[key] = float(np.mean(values)) if values else float("nan")
    return out


def summarize_by_bucket(rows: list[dict[str, Any]], value_keys: tuple[str, ...]) -> dict[str, Any]:
    out = {}
    for bucket in ("tiny", "small", "medium", "large"):
        group = [row for row in rows if bucket_name(int(row.get("raw_count", row.get("raw_ind_count", 0)))) == bucket]
        out[bucket] = summarize_rows(group, value_keys)
    buttons = [row for row in rows if "button" in str(row.get("part_name", "")).lower()]
    out["button"] = summarize_rows(buttons, value_keys)
    return out


def sinusoidal_position_2d(height: int, width: int, dim: int) -> torch.Tensor:
    if dim % 4 != 0:
        raise ValueError(f"2D sincos dim must be divisible by 4, got {dim}")
    y, x = torch.meshgrid(torch.arange(height), torch.arange(width), indexing="ij")
    omega = torch.arange(dim // 4, dtype=torch.float32) / float(dim // 4)
    omega = 1.0 / (10000.0 ** omega)
    out = []
    for coord in (x.reshape(-1).float(), y.reshape(-1).float()):
        scaled = coord[:, None] * omega[None, :]
        out.extend([scaled.sin(), scaled.cos()])
    return torch.cat(out, dim=1)


def format_table(rows: list[dict[str, Any]], headers: list[str]) -> str:
    widths = [len(header) for header in headers]
    values = []
    for row in rows:
        vals = [str(row.get(header, "")) for header in headers]
        values.append(vals)
        widths = [max(width, len(value)) for width, value in zip(widths, vals)]
    line = " | ".join(header.ljust(width) for header, width in zip(headers, widths))
    sep = " | ".join("-" * width for width in widths)
    body = [" | ".join(value.ljust(width) for value, width in zip(vals, widths)) for vals in values]
    return "\n".join([line, sep, *body])


def approx_param_count(module: torch.nn.Module) -> int:
    return int(sum(p.numel() for p in module.parameters()))


def mask_morphology(mask: torch.Tensor, mode: str) -> torch.Tensor:
    if mode == "none":
        return mask
    x = mask.float().unsqueeze(1)
    if mode == "dilate":
        return F.max_pool3d(x, kernel_size=3, stride=1, padding=1).squeeze(1)
    if mode == "erode":
        inv = 1.0 - x
        return (1.0 - F.max_pool3d(inv, kernel_size=3, stride=1, padding=1)).squeeze(1)
    raise ValueError(f"unknown morphology mode: {mode}")


def bias_from_probability(prob: float) -> float:
    prob = min(max(float(prob), 1.0e-6), 1.0 - 1.0e-6)
    return float(math.log(prob / (1.0 - prob)))


def latent_support_mask(
    latent_gt: torch.Tensor,
    empty_code: torch.Tensor,
    raw_mask: torch.Tensor,
    *,
    multiplier: float = 4.0,
    far_radius: int = 2,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return data-driven latent support mask and per-sample noise/threshold."""
    if latent_gt.dim() != 5:
        raise ValueError(f"latent_gt expected [B,8,16,16,16], got {tuple(latent_gt.shape)}")
    if raw_mask.dim() != 4:
        raise ValueError(f"raw_mask expected [B,16,16,16], got {tuple(raw_mask.shape)}")
    empty = empty_code.to(device=latent_gt.device, dtype=latent_gt.dtype)
    if empty.dim() == 4:
        empty = empty.unsqueeze(0)
    diff = (latent_gt - empty).abs().mean(dim=1)
    m = raw_mask.bool().float().unsqueeze(1)
    for _ in range(int(far_radius)):
        m = F.max_pool3d(m, kernel_size=3, stride=1, padding=1)
    far = m[:, 0] <= 0
    noises = []
    thresholds = []
    support = torch.zeros_like(raw_mask, dtype=torch.bool)
    for idx in range(latent_gt.shape[0]):
        if far[idx].any():
            noise = diff[idx][far[idx]].mean()
        else:
            noise = diff[idx].mean()
        threshold = float(multiplier) * noise
        support[idx] = diff[idx] > threshold
        noises.append(noise)
        thresholds.append(threshold)
    return support.float(), torch.stack(noises), torch.stack(thresholds)
