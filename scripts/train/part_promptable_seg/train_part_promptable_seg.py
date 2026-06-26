#!/usr/bin/env python3
"""Train promptable discriminative part SS-latent segmentation."""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import subprocess
import sys
import time
from datetime import timedelta
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, Dataset, Sampler
from torch.utils.data.distributed import DistributedSampler


PROJECT_ROOT = Path(__file__).resolve().parents[3]
TRELLIS_PATH = PROJECT_ROOT / "TRELLIS-arts"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(TRELLIS_PATH) not in sys.path:
    sys.path.insert(0, str(TRELLIS_PATH))

from scripts.train.part_promptable_seg.part_promptable_seg_utils import (  # noqa: E402
    OFFICIAL_SPLIT_PATH,
    PACKED_DATA_ROOT,
    MultiPromptableBaseDataset,
    PartRow,
    PackedPromptablePartDataset,
    PackedShardBatchSampler,
    PromptablePartDataset,
    audit_promptable_mask_visibility,
    approx_param_count,
    bucket_name,
    build_oversampling_plan,
    build_semantic_vocab,
    collate_promptable_parts,
    compute_empty_code,
    decode_latents_to_coords,
    decode_metrics_for_batch,
    dense_occ_from_coords,
    dataset_specs_from_split,
    enumerate_part_rows_multi,
    format_table,
    load_ss_decoder,
    load_ss_encoder,
    latent_support_mask,
    load_official_split,
    make_base_datasets,
    mask_metrics_from_logits,
    mask_morphology,
    object_key,
    part_row_key,
    pick_gate1_rows,
    rows_for_obj_ids,
    seed_all,
    split_rows_by_obj,
    summarize_rows,
    summarize_by_bucket,
)
from scripts.train.part_promptable_seg.pack_part_promptable_seg_dataset import (  # noqa: E402
    ensure_packed_dataset,
    optional_path_arg,
    pack_completion_status,
    source_fingerprint,
)
from trellis.models.part_seg.promptable_latent_seg import PromptablePartLatentSegNet  # noqa: E402


DEFAULT_GATE1_OUT = Path("/mnt/robot-data-lab/jzh/art-gen-output/debug/part_promptable_seg_gate1_overfit")
DEFAULT_GATE2_OUT = Path("/mnt/robot-data-lab/jzh/art-gen-output/debug/part_promptable_seg_gate2_256obj")
DEFAULT_PACKED_V4 = Path("/mnt/robot-data-lab/jzh/art-gen/data/part_promptable_seg_packed_v4")

NO_PROMPT_TRACKER: dict[str, Any] | None = None


class SmallOversampleSampler(Sampler[int]):
    def __init__(
        self,
        rows: list[Any],
        *,
        small_oversample: int,
        shuffle: bool,
        seed: int,
        num_replicas: int = 1,
        rank: int = 0,
        repeat_by_key: dict[str, int] | None = None,
    ) -> None:
        self.rows = list(rows)
        self.small_oversample = max(1, int(small_oversample))
        self.repeat_by_key = {str(key): max(1, int(value)) for key, value in dict(repeat_by_key or {}).items()}
        self.shuffle = bool(shuffle)
        self.seed = int(seed)
        self.num_replicas = int(num_replicas)
        self.rank = int(rank)
        self.epoch = 0
        expanded: list[int] = []
        for idx, row in enumerate(self.rows):
            repeat = self.repeat_by_key.get(part_row_key(row))
            if repeat is None:
                part_text = f"{row.part_name} {row.semantic_type}".lower()
                repeat = self.small_oversample if (int(row.raw_count) < 50 or "button" in part_text) else 1
            expanded.extend([idx] * int(repeat))
        self.indices = expanded

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self.epoch)
        indices = list(self.indices)
        if self.shuffle:
            rng.shuffle(indices)
        if self.num_replicas > 1:
            total = int(math.ceil(len(indices) / self.num_replicas) * self.num_replicas)
            if len(indices) < total:
                indices.extend(indices[: total - len(indices)])
            indices = indices[self.rank :: self.num_replicas]
        return iter(indices)

    def __len__(self) -> int:
        if self.num_replicas <= 1:
            return len(self.indices)
        return int(math.ceil(len(self.indices) / self.num_replicas))


class ObjectGroupBatchSampler(Sampler[list[int]]):
    """Batch rows by object/angle so cross-part losses see complete local groups."""

    def __init__(
        self,
        rows: list[Any],
        *,
        batch_size: int,
        shuffle: bool,
        seed: int,
        num_replicas: int = 1,
        rank: int = 0,
        repeat_by_key: dict[str, int] | None = None,
    ) -> None:
        self.rows = list(rows)
        self.batch_size = int(batch_size)
        self.shuffle = bool(shuffle)
        self.seed = int(seed)
        self.num_replicas = int(num_replicas)
        self.rank = int(rank)
        self.repeat_by_key = {str(key): max(1, int(value)) for key, value in dict(repeat_by_key or {}).items()}
        self.epoch = 0
        if self.batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {self.batch_size}")
        if self.num_replicas <= 0:
            raise ValueError(f"num_replicas must be positive, got {self.num_replicas}")
        if not 0 <= self.rank < self.num_replicas:
            raise ValueError(f"rank={self.rank} must be in [0, {self.num_replicas})")
        groups: dict[str, list[int]] = {}
        for idx, row in enumerate(self.rows):
            key = f"{object_key(row)}|angle_{int(row.angle_idx)}"
            groups.setdefault(key, []).append(idx)
        self.groups = list(groups.values())

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def _group_repeat(self, group: list[int]) -> int:
        repeats = [
            self.repeat_by_key.get(part_row_key(self.rows[idx]), 1)
            for idx in group
        ]
        return max(1, max(repeats) if repeats else 1)

    def _all_batches(self) -> list[list[int]]:
        rng = random.Random(self.seed + self.epoch)
        groups = [list(group) for group in self.groups]
        if self.shuffle:
            rng.shuffle(groups)
            for group in groups:
                rng.shuffle(group)
        expanded_groups: list[list[int]] = []
        for group in groups:
            expanded_groups.extend([group] * self._group_repeat(group))
        batches: list[list[int]] = []
        current: list[int] = []
        for group in expanded_groups:
            if len(group) > self.batch_size:
                if current:
                    batches.append(current)
                    current = []
                batches.append(group)
                continue
            if current and len(current) + len(group) > self.batch_size:
                batches.append(current)
                current = []
            current.extend(group)
        if current:
            batches.append(current)
        if self.num_replicas > 1:
            total = int(math.ceil(len(batches) / self.num_replicas) * self.num_replicas)
            if len(batches) < total and batches:
                batches.extend([list(batches[i % len(batches)]) for i in range(total - len(batches))])
            batches = batches[self.rank :: self.num_replicas]
        return batches

    def __iter__(self):
        return iter(self._all_batches())

    def __len__(self) -> int:
        count = len(self._all_batches())
        return count


class PackedObjectGroupBatchSampler(Sampler[list[int]]):
    """Shard-aware object/angle batches for packed data and cross-part losses."""

    def __init__(
        self,
        dataset: PackedPromptablePartDataset,
        *,
        batch_size: int,
        shuffle: bool,
        seed: int,
        num_replicas: int = 1,
        rank: int = 0,
        repeat_by_key: dict[str, int] | None = None,
    ) -> None:
        self.dataset = dataset
        self.rows = list(dataset.rows)
        self.entries = list(dataset.entries_for_rows)
        self.batch_size = int(batch_size)
        self.shuffle = bool(shuffle)
        self.seed = int(seed)
        self.num_replicas = int(num_replicas)
        self.rank = int(rank)
        self.repeat_by_key = {str(key): max(1, int(value)) for key, value in dict(repeat_by_key or {}).items()}
        self.epoch = 0
        if self.batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {self.batch_size}")
        if self.num_replicas <= 0:
            raise ValueError(f"num_replicas must be positive, got {self.num_replicas}")
        if not 0 <= self.rank < self.num_replicas:
            raise ValueError(f"rank={self.rank} must be in [0, {self.num_replicas})")
        grouped: dict[str, list[int]] = {}
        for idx, row in enumerate(self.rows):
            key = f"{object_key(row)}|angle_{int(row.angle_idx)}"
            grouped.setdefault(key, []).append(idx)
        by_shard: dict[str, list[list[int]]] = {}
        for group in grouped.values():
            shard_counts: dict[str, int] = {}
            for idx in group:
                shard = str(self.entries[idx]["shard"])
                shard_counts[shard] = shard_counts.get(shard, 0) + 1
            shard = max(shard_counts.items(), key=lambda item: item[1])[0]
            by_shard.setdefault(shard, []).append(group)
        self.by_shard = by_shard
        self.shards = sorted(by_shard)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def _group_repeat(self, group: list[int]) -> int:
        repeats = [
            self.repeat_by_key.get(part_row_key(self.rows[idx]), 1)
            for idx in group
        ]
        return max(1, max(repeats) if repeats else 1)

    def _pack_groups(self, groups: list[list[int]]) -> list[list[int]]:
        batches: list[list[int]] = []
        current: list[int] = []
        for group in groups:
            if len(group) > self.batch_size:
                if current:
                    batches.append(current)
                    current = []
                batches.append(group)
                continue
            if current and len(current) + len(group) > self.batch_size:
                batches.append(current)
                current = []
            current.extend(group)
        if current:
            batches.append(current)
        return batches

    def _all_batches(self) -> list[list[int]]:
        rng = random.Random(self.seed + self.epoch)
        shards = list(self.shards)
        if self.shuffle:
            rng.shuffle(shards)
        batches: list[list[int]] = []
        for shard in shards:
            groups = [list(group) for group in self.by_shard[shard]]
            if self.shuffle:
                rng.shuffle(groups)
                for group in groups:
                    rng.shuffle(group)
            expanded: list[list[int]] = []
            for group in groups:
                expanded.extend([group] * self._group_repeat(group))
            batches.extend(self._pack_groups(expanded))
        if self.shuffle:
            rng.shuffle(batches)
        if self.num_replicas > 1:
            total = int(math.ceil(len(batches) / self.num_replicas) * self.num_replicas)
            if len(batches) < total and batches:
                batches.extend([list(batches[i % len(batches)]) for i in range(total - len(batches))])
            batches = batches[self.rank :: self.num_replicas]
        return batches

    def __iter__(self):
        return iter(self._all_batches())

    def __len__(self) -> int:
        return len(self._all_batches())


Y_UP_TO_Z_UP_3 = np.asarray([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]], dtype=np.float64)
Y_UP_TO_Z_UP_4 = np.eye(4, dtype=np.float64)
Y_UP_TO_Z_UP_4[:3, :3] = Y_UP_TO_Z_UP_3


def _motion_needs_y_up_fix(dataset_id: str | None, data_root: str | Path | None) -> bool:
    text = " ".join([str(dataset_id or ""), str(data_root or "")]).lower()
    return "phyx-verse" in text or "phyx_verse" in text


def _motion_transform_to_voxel_frame(matrix: np.ndarray, *, dataset_id: str, data_root: str) -> np.ndarray:
    mat = np.asarray(matrix, dtype=np.float64)
    if mat.shape != (4, 4):
        raise ValueError(f"expected 4x4 part transform, got {mat.shape}")
    if _motion_needs_y_up_fix(dataset_id, data_root):
        return Y_UP_TO_Z_UP_4 @ mat @ np.linalg.inv(Y_UP_TO_Z_UP_4)
    return mat.copy()


def _read_json_file(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_motion_camera_frame(data_root: str | Path, obj_id: str, angle_idx: int) -> tuple[float, np.ndarray] | None:
    path = Path(data_root) / "renders" / str(obj_id) / f"angle_{int(angle_idx)}" / "camera_transforms.json"
    if not path.is_file():
        return None
    data = _read_json_file(path)
    if "scale" not in data or "offset" not in data:
        return None
    return float(data["scale"]), np.asarray(data["offset"], dtype=np.float32)


def _load_motion_raw_coords(data_root: str | Path, obj_id: str, angle_idx: int, part_name: str) -> torch.Tensor | None:
    path = (
        Path(data_root)
        / "reconstruction"
        / "voxel_expanded"
        / str(obj_id)
        / f"angle_{int(angle_idx)}"
        / "64"
        / f"ind_{part_name}.npy"
    )
    if not path.is_file():
        return None
    coords = np.asarray(np.load(path), dtype=np.int64)
    if coords.ndim != 2 or coords.shape[1] != 3:
        return None
    return torch.from_numpy(coords).long()


def _part_info_by_name(data_root: str | Path, obj_id: str) -> dict[str, dict[str, Any]]:
    path = Path(data_root) / "reconstruction" / "part_info" / str(obj_id) / "part_info.json"
    if not path.is_file():
        return {}
    data = _read_json_file(path)
    parts = data.get("parts")
    return dict(parts) if isinstance(parts, dict) else {}


def _joint_transforms_for_object(data_root: str | Path, obj_id: str) -> dict[str, Any] | None:
    path = Path(data_root) / "joint_transforms" / f"{obj_id}.json"
    if not path.is_file():
        return None
    data = _read_json_file(path)
    angles = data.get("angles")
    return data if isinstance(angles, dict) else None


def _motion_object_key(row: Any) -> str:
    return f"{getattr(row, 'dataset_id', '')}::{getattr(row, 'obj_id', '')}"


def _part_motion_score(meta: dict[str, Any] | None, angle_meta: dict[str, Any] | None) -> float:
    if not meta or not angle_meta:
        return 0.0
    group_id = str(meta.get("joint_group_id", ""))
    if not group_id:
        return 0.0
    states = angle_meta.get("joint_states")
    if not isinstance(states, dict):
        return 0.0
    return float(states.get(group_id, 0.0) or 0.0)


def build_motion_sidecar(
    rows: list[PartRow],
    *,
    max_angle_delta: int = 0,
    min_state_delta: float = 1.0e-6,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    """Build per-row GT motion targets without changing packed sample files."""

    unique_rows: list[PartRow] = []
    seen_keys: set[str] = set()
    for row in rows:
        key = part_row_key(row)
        if key in seen_keys:
            continue
        seen_keys.add(key)
        unique_rows.append(row)

    rows_by_object: dict[str, list[PartRow]] = {}
    for row in unique_rows:
        rows_by_object.setdefault(_motion_object_key(row), []).append(row)

    raw_coord_cache: dict[tuple[str, str, int, str], torch.Tensor | None] = {}

    def load_cached_raw_coords(data_root: str, obj_id: str, angle_idx: int, part_name: str) -> torch.Tensor | None:
        cache_key = (str(data_root), str(obj_id), int(angle_idx), str(part_name))
        if cache_key not in raw_coord_cache:
            raw_coord_cache[cache_key] = _load_motion_raw_coords(data_root, obj_id, angle_idx, part_name)
        return raw_coord_cache[cache_key]

    sidecar: dict[str, dict[str, Any]] = {}
    stats: dict[str, Any] = {
        "rows": len(rows),
        "unique_rows": len(unique_rows),
        "objects": len(rows_by_object),
        "enabled_rows": 0,
        "skipped": {},
        "max_angle_delta": int(max_angle_delta),
        "min_state_delta": float(min_state_delta),
    }

    def skip(reason: str) -> None:
        skipped = stats.setdefault("skipped", {})
        skipped[reason] = int(skipped.get(reason, 0)) + 1

    for _obj_key, obj_rows in rows_by_object.items():
        first = obj_rows[0]
        data_root = str(first.data_root)
        dataset_id = str(first.dataset_id)
        obj_id = str(first.obj_id)
        part_info = _part_info_by_name(data_root, obj_id)
        jt = _joint_transforms_for_object(data_root, obj_id)
        if not part_info:
            for _row in obj_rows:
                skip("missing_part_info")
            continue
        if jt is None:
            for _row in obj_rows:
                skip("missing_joint_transforms")
            continue
        angles_meta = jt.get("angles", {})
        available_angles = sorted({int(row.angle_idx) for row in obj_rows if str(int(row.angle_idx)) in angles_meta})
        if len(available_angles) < 2:
            for _row in obj_rows:
                skip("lt_two_angles")
            continue
        camera_cache: dict[int, tuple[float, np.ndarray]] = {}
        for angle in available_angles:
            cam = _load_motion_camera_frame(data_root, obj_id, angle)
            if cam is not None:
                camera_cache[int(angle)] = cam
        for row in obj_rows:
            key = part_row_key(row)
            part_meta = part_info.get(str(row.part_name), {})
            joint_type = str(part_meta.get("joint_type") or part_meta.get("joint") or "")
            if joint_type not in {"B", "C", "prismatic", "revolute"}:
                skip("static_or_unsupported_joint")
                continue
            part_index = part_meta.get("part_index", part_meta.get("label"))
            if part_index is None:
                skip("missing_part_index")
                continue
            angle_a = int(row.angle_idx)
            if angle_a not in camera_cache:
                skip("missing_camera_a")
                continue
            angle_a_meta = angles_meta.get(str(angle_a))
            if not isinstance(angle_a_meta, dict):
                skip("missing_angle_a")
                continue
            part_trans_a = (angle_a_meta.get("part_transforms") or {}).get(str(int(part_index)))
            if part_trans_a is None:
                skip("missing_transform_a")
                continue
            state_a = _part_motion_score(part_meta, angle_a_meta)

            candidates: list[tuple[float, int, int, Any]] = []
            for angle_b in available_angles:
                if angle_b == angle_a or angle_b not in camera_cache:
                    continue
                if int(max_angle_delta) > 0 and abs(int(angle_b) - angle_a) > int(max_angle_delta):
                    continue
                angle_b_meta = angles_meta.get(str(angle_b))
                if not isinstance(angle_b_meta, dict):
                    continue
                part_trans_b = (angle_b_meta.get("part_transforms") or {}).get(str(int(part_index)))
                if part_trans_b is None:
                    continue
                state_delta = abs(float(_part_motion_score(part_meta, angle_b_meta)) - float(state_a))
                if state_delta < float(min_state_delta):
                    continue
                candidates.append((state_delta, abs(int(angle_b) - angle_a), int(angle_b), part_trans_b))

            if not candidates:
                skip("no_valid_angle_b")
                continue
            angle_b = -1
            part_trans_b = None
            target_b = None
            for _state_delta, _angle_delta, cand_angle_b, cand_part_trans_b in sorted(candidates, key=lambda x: (x[0], x[1], x[2])):
                cand_target_b = load_cached_raw_coords(data_root, obj_id, cand_angle_b, str(row.part_name))
                if cand_target_b is not None and cand_target_b.numel() > 0:
                    angle_b = int(cand_angle_b)
                    part_trans_b = cand_part_trans_b
                    target_b = cand_target_b
                    break
            if angle_b < 0 or part_trans_b is None or target_b is None:
                skip("no_target_b")
                continue
            transform_a = _motion_transform_to_voxel_frame(np.asarray(part_trans_a), dataset_id=dataset_id, data_root=data_root)
            transform_b = _motion_transform_to_voxel_frame(np.asarray(part_trans_b), dataset_id=dataset_id, data_root=data_root)
            g_ab = transform_b @ np.linalg.inv(transform_a)
            scale_a, offset_a = camera_cache[angle_a]
            scale_b, offset_b = camera_cache[angle_b]
            sidecar[key] = {
                "motion_valid": True,
                "motion_angle_b": int(angle_b),
                "motion_joint_type": joint_type,
                "motion_transform_ab": torch.from_numpy(g_ab.astype(np.float32)),
                "motion_target_coords_b": target_b.long(),
                "motion_scale_a": torch.tensor(float(scale_a), dtype=torch.float32),
                "motion_offset_a": torch.from_numpy(offset_a.astype(np.float32)),
                "motion_scale_b": torch.tensor(float(scale_b), dtype=torch.float32),
                "motion_offset_b": torch.from_numpy(offset_b.astype(np.float32)),
            }
            stats["enabled_rows"] = int(stats["enabled_rows"]) + 1

    stats["raw_coord_cache_entries"] = int(len(raw_coord_cache))
    stats["enabled_ratio"] = float(stats["enabled_rows"] / max(1, stats["unique_rows"]))
    return sidecar, stats


class MotionSidecarDataset(Dataset):
    def __init__(self, base: Dataset, rows: list[PartRow], sidecar: dict[str, dict[str, Any]]) -> None:
        self.base = base
        self.rows = list(rows)
        self.sidecar = dict(sidecar)
        if len(self.rows) != len(self.base):
            raise ValueError(f"MotionSidecarDataset rows/base length mismatch: {len(self.rows)} vs {len(self.base)}")

    def __getattr__(self, name: str) -> Any:
        return getattr(self.base, name)

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        sample = dict(self.base[int(idx)])
        meta = self.sidecar.get(part_row_key(self.rows[int(idx)]))
        if meta is None:
            sample["motion_valid"] = torch.tensor(False, dtype=torch.bool)
            sample["motion_angle_b"] = torch.tensor(-1, dtype=torch.long)
            sample["motion_joint_type"] = ""
            sample["motion_transform_ab"] = torch.eye(4, dtype=torch.float32)
            sample["motion_target_coords_b"] = torch.empty((0, 3), dtype=torch.long)
            sample["motion_scale_a"] = torch.tensor(1.0, dtype=torch.float32)
            sample["motion_offset_a"] = torch.zeros((3,), dtype=torch.float32)
            sample["motion_scale_b"] = torch.tensor(1.0, dtype=torch.float32)
            sample["motion_offset_b"] = torch.zeros((3,), dtype=torch.float32)
        else:
            sample.update(meta)
        return sample


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("gate1", "gate2"), default="gate1")
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=20260611)
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--eval-batch-size", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1.0e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-steps", type=int, default=200)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--dim", type=int, default=256)
    parser.add_argument("--depth", type=int, default=6)
    parser.add_argument("--head-depth", type=int, default=2)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--voxel-depth", type=int, default=3)
    parser.add_argument("--decode-dice-weight", type=float, default=0.5)
    parser.add_argument("--latent-part-weight", type=float, default=8.0)
    parser.add_argument("--latent-loss-mode", choices=("weighted", "signal_normalized"), default="weighted")
    parser.add_argument("--route", choices=("latent", "voxel"), default="latent")
    parser.add_argument("--voxel-loss-weight", type=float, default=1.0)
    parser.add_argument("--voxel-max-tokens", type=int, default=0)
    parser.add_argument("--refine-mode", choices=("token", "spconv"), default="token")
    parser.add_argument("--spconv-depth", type=int, default=4)
    parser.add_argument("--xpart-ce-weight", type=float, default=0.0)
    parser.add_argument("--infer-resolve", choices=("independent", "argmax"), default="independent")
    parser.add_argument("--motion-loss-weight", type=float, default=0.0)
    parser.add_argument("--motion-loss-kind", choices=("bce_dice", "bce"), default="bce_dice")
    parser.add_argument("--motion-max-angle-delta", type=int, default=0)
    parser.add_argument("--motion-sanity-only", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--motion-sanity-max-bce", type=float, default=0.01)
    parser.add_argument("--motion-sanity-max-dice", type=float, default=0.05)
    parser.add_argument("--warm-start", type=Path, default=None)
    parser.add_argument("--warm-start-freeze-steps", type=int, default=0)
    parser.add_argument("--embed-loss-weight", "--embed_loss_weight", dest="embed_loss_weight", type=float, default=0.0)
    parser.add_argument("--voxel-embedding-dim", type=int, default=16)
    parser.add_argument("--embed-pull-margin", type=float, default=0.5)
    parser.add_argument("--embed-push-margin", type=float, default=1.5)
    parser.add_argument("--embed-max-voxels-per-part", type=int, default=512)
    parser.add_argument("--object-group-batches", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--memorize-threshold", type=float, default=0.0)
    parser.add_argument("--mask-augment", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--view-dropout", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--min-views", type=int, default=1)
    parser.add_argument("--min-prompt-views", "--min_prompt_views", type=int, default=2)
    parser.add_argument("--view-dropout-start-step", "--view_dropout_start_step", type=int, default=0)
    parser.add_argument("--mask-only", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--mask-target", choices=("raw", "support"), default="raw")
    parser.add_argument("--support-multiplier", type=float, default=4.0)
    parser.add_argument("--mask-encoder", choices=("cnn_grid", "fg_points"), default="cnn_grid")
    parser.add_argument("--point-k-boundary", type=int, default=32)
    parser.add_argument("--point-k-interior", type=int, default=32)
    parser.add_argument("--point-resample-points", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--semantic-aux", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--semantic-loss-weight", type=float, default=0.1)
    parser.add_argument("--single-obj-id", default=None)
    parser.add_argument("--single-angle-idx", type=int, default=None)
    parser.add_argument("--single-part-name", default=None)
    parser.add_argument("--selection-json", type=Path, default=None)
    parser.add_argument("--fp16", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--compile", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--eval-every", type=int, default=500)
    parser.add_argument("--ckpt-every", type=int, default=1000)
    parser.add_argument("--eval-max-rows", type=int, default=0)
    parser.add_argument("--gate2-objects", type=int, default=256)
    parser.add_argument("--gate2-train-objects", type=int, default=224)
    parser.add_argument("--gate2-heldout-objects", type=int, default=32)
    parser.add_argument("--heldout-fraction", type=float, default=0.125)
    parser.add_argument("--decode-eval-steps", default="")
    parser.add_argument("--train-eval-max-rows", type=int, default=0)
    parser.add_argument("--heldout-eval-max-rows", type=int, default=0)
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--auto-resume", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--split-json", type=Path, default=None)
    parser.add_argument("--proxy-json", type=Path, default=None)
    parser.add_argument("--packed-dir", type=Path, default=None)
    parser.add_argument("--use-packed-whole-occ", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--auto-pack", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--base-packed-dir", type=optional_path_arg, default=None)
    parser.add_argument("--pack-shard-size", type=int, default=512)
    parser.add_argument("--pack-progress-every", type=int, default=1000)
    parser.add_argument("--pack-limit", type=int, default=0)
    parser.add_argument("--pack-barrier-timeout-s", type=int, default=7200)
    parser.add_argument("--small-oversample", type=int, default=2)
    parser.add_argument("--realappliance-oversample", type=int, default=0, help="0 means auto-compute from --realappliance-target-share")
    parser.add_argument("--realappliance-target-share", type=float, default=0.22)
    parser.add_argument("--realappliance-max-oversample", type=int, default=8)
    parser.add_argument("--verse-focus-oversample", type=int, default=2)
    parser.add_argument("--focal-gamma", type=float, default=1.5)
    parser.add_argument("--boundary-weight", "--boundary_weight", dest="boundary_weight", type=float, default=1.0)
    parser.add_argument("--full-eval-every", type=int, default=0)
    parser.add_argument("--final-full-eval", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--early-stop-patience", type=int, default=0)
    parser.add_argument("--early-stop-metric", default="heldout/e2e_decode_iou")
    parser.add_argument("--early-stop-min-delta", type=float, default=0.0)
    parser.add_argument("--mask-audit-views", type=int, default=0)
    parser.add_argument("--filter-undetectable", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--fail-label-absent-ratio", type=float, default=0.02)
    return parser.parse_args()


def init_distributed_if_needed() -> tuple[bool, int, int, int]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size <= 1:
        return False, 0, 1, 0
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    timeout_s = int(os.environ.get("PROMPTSEG_DDP_TIMEOUT_S", "7200"))
    dist.init_process_group(backend="nccl", timeout=timedelta(seconds=max(3600, timeout_s)))
    return True, rank, world_size, local_rank


def lr_for_step(base_lr: float, step: int, *, warmup_steps: int, total_steps: int) -> float:
    if step <= int(warmup_steps):
        return float(base_lr) * step / float(max(1, int(warmup_steps)))
    progress = min(1.0, (step - int(warmup_steps)) / float(max(1, int(total_steps) - int(warmup_steps))))
    return float(base_lr) * 0.5 * (1.0 + math.cos(math.pi * progress))


def dice_loss_prob(prob: torch.Tensor, target: torch.Tensor, dims: tuple[int, ...]) -> torch.Tensor:
    inter = (prob * target).sum(dim=dims)
    denom = prob.sum(dim=dims) + target.sum(dim=dims)
    return (1.0 - (2.0 * inter + 1.0e-6) / (denom + 1.0e-6)).mean()


def mask_loss(
    logits: torch.Tensor,
    target_flat: torch.Tensor,
    *,
    focal_gamma: float = 0.0,
    boundary_flat: torch.Tensor | None = None,
    boundary_weight: float = 1.0,
    collect_stats: bool = True,
) -> tuple[torch.Tensor, dict[str, float]]:
    target = target_flat.float()
    boundary = None
    if boundary_flat is not None and float(boundary_weight) > 1.0:
        boundary = boundary_flat.to(device=target.device, dtype=torch.bool)
        if boundary.shape != target.shape:
            raise ValueError(f"boundary_flat shape {tuple(boundary.shape)} must match target {tuple(target.shape)}")
    pos = target.sum(dim=1).clamp_min(1.0)
    neg = target.shape[1] - pos
    weights = (neg / pos).clamp(4.0, 1000.0)
    bce_elem = F.binary_cross_entropy_with_logits(
        logits,
        target,
        pos_weight=weights.view(-1, 1),
        reduction="none",
    )
    if float(focal_gamma) > 0:
        prob = torch.sigmoid(logits)
        pt = torch.where(target > 0.5, prob, 1.0 - prob).clamp(1.0e-6, 1.0 - 1.0e-6)
        bce_elem = bce_elem * torch.pow(1.0 - pt, float(focal_gamma))
    if boundary is not None:
        bce_elem = bce_elem * torch.where(
            boundary,
            bce_elem.new_full((), float(boundary_weight)),
            bce_elem.new_ones(()),
        )
    bce = bce_elem.mean(dim=1).mean()
    dice = dice_loss_prob(logits.sigmoid(), target, dims=(1,))
    if not collect_stats:
        return bce + dice, {}
    return bce + dice, {
        "mask_bce": float(bce.detach().item()),
        "mask_dice": float(dice.detach().item()),
        "focal_gamma": float(focal_gamma),
        "boundary_weight": float(boundary_weight),
        "boundary_voxel_ratio": float(boundary.float().mean().detach().item()) if boundary is not None else 0.0,
    }


def decode_dice_loss(decoder, latents: torch.Tensor, raw_coords: list[torch.Tensor], *, device: torch.device) -> torch.Tensor:
    target = dense_occ_from_coords(raw_coords, device=device)
    with torch.cuda.amp.autocast(enabled=False):
        logits = decoder(latents.float()).float()
    prob = logits.sigmoid()
    return dice_loss_prob(prob, target, dims=(1, 2, 3, 4))


@torch.no_grad()
def decode_full_occ(decoder, z_global: torch.Tensor, threshold: float = 0.0) -> torch.Tensor:
    with torch.cuda.amp.autocast(enabled=False):
        logits = decoder(z_global.float()).float()
    return (logits > float(threshold)).float()


def visible_view_counts(masks2d: torch.Tensor) -> torch.Tensor:
    if masks2d.dim() != 4:
        raise ValueError(f"masks2d expected [B,V,H,W], got {tuple(masks2d.shape)}")
    return (masks2d.flatten(2).sum(dim=2) > 0).sum(dim=1)


def apply_view_dropout(
    masks2d: torch.Tensor,
    *,
    min_views: int = 1,
    min_prompt_views: int = 1,
) -> tuple[torch.Tensor, dict[str, float]]:
    if masks2d.dim() != 4:
        raise ValueError(f"masks2d expected [B,V,H,W], got {tuple(masks2d.shape)}")
    bsz, views = masks2d.shape[:2]
    legacy_keep_min = max(1, min(int(min_views), int(views)))
    prompt_keep_min = max(1, min(int(min_prompt_views), int(views)))
    out = masks2d.clone()
    before_counts = visible_view_counts(masks2d)
    dropped = 0
    skipped_guard = 0
    for idx in range(bsz):
        nonempty = torch.nonzero(masks2d[idx].flatten(1).sum(dim=1) > 0, as_tuple=False).flatten()
        if nonempty.numel() == 0:
            continue
        max_keep = int(nonempty.numel())
        if max_keep < prompt_keep_min:
            skipped_guard += 1
            continue
        keep_min = min(max(legacy_keep_min, prompt_keep_min), max_keep)
        keep_count = int(torch.randint(keep_min, max_keep + 1, (), device=masks2d.device).item())
        if keep_count < max_keep:
            dropped += 1
        perm = nonempty[torch.randperm(max_keep, device=masks2d.device)]
        keep = perm[:keep_count]
        mask = torch.zeros((views,), dtype=torch.bool, device=masks2d.device)
        mask[keep] = True
        out[idx, ~mask] = 0.0
    after_counts = visible_view_counts(out)
    stats = {
        "view_dropout_single_before": float((before_counts == 1).sum().detach().item()),
        "view_dropout_single_after": float((after_counts == 1).sum().detach().item()),
        "view_dropout_min_prompt_views": float(prompt_keep_min),
        "view_dropout_skipped_guard": float(skipped_guard),
        "view_dropout_dropped_prompts": float(dropped),
    }
    return out, stats


def semantic_loss_from_logits(logits: torch.Tensor | None, target: torch.Tensor | None, *, weight: float) -> tuple[torch.Tensor | None, dict[str, float]]:
    if logits is None or target is None or float(weight) <= 0:
        return None, {"semantic_ce": 0.0, "semantic_acc": 0.0}
    valid = target >= 0
    if not bool(valid.any()):
        return None, {"semantic_ce": 0.0, "semantic_acc": 0.0}
    ce = F.cross_entropy(logits[valid].float(), target[valid].long())
    pred = logits[valid].argmax(dim=1)
    acc = (pred == target[valid]).float().mean()
    return float(weight) * ce, {"semantic_ce": float(ce.detach().item()), "semantic_acc": float(acc.detach().item())}


def parse_step_set(text: str) -> set[int]:
    out: set[int] = set()
    for piece in str(text or "").split(","):
        piece = piece.strip()
        if not piece:
            continue
        out.add(int(piece))
    return out


def dense_occ_from_batch_whole(batch: dict[str, Any], *, device: torch.device) -> torch.Tensor:
    if "whole_coords" not in batch:
        raise KeyError("--use-packed-whole-occ requires batch['whole_coords']; pack with include_whole_coords")
    return dense_occ_from_coords(batch["whole_coords"], device=device)


@torch.no_grad()
def full_occ_for_batch(
    decoder,
    z_global: torch.Tensor,
    batch: dict[str, Any],
    *,
    device: torch.device,
    use_packed_whole_occ: bool,
) -> torch.Tensor:
    if bool(use_packed_whole_occ):
        return dense_occ_from_batch_whole(batch, device=device)
    return decode_full_occ(decoder, z_global, threshold=0.0)


def voxel_loss_from_logits(
    logits: torch.Tensor,
    coords_list: list[torch.Tensor],
    raw_coords: list[torch.Tensor],
    *,
    device: torch.device,
    collect_stats: bool = True,
) -> tuple[torch.Tensor, dict[str, float]]:
    losses = []
    bce_items = []
    dice_items = []
    ious = []
    precisions = []
    recalls = []
    pred_counts = []
    gt_counts = []
    for idx, coords in enumerate(coords_list):
        valid = coords.shape[0] > 0
        if not valid:
            continue
        coords = coords.to(device=device, dtype=torch.long)
        keys = (coords[:, 0].clamp(0, 63) * 4096 + coords[:, 1].clamp(0, 63) * 64 + coords[:, 2].clamp(0, 63)).long()
        raw = torch.as_tensor(raw_coords[idx], dtype=torch.long, device=device)
        target_dense = torch.zeros((64 * 64 * 64,), dtype=torch.float32, device=device)
        if raw.numel() > 0:
            raw = raw.view(-1, 3)
            raw_keys = (raw[:, 0].clamp(0, 63) * 4096 + raw[:, 1].clamp(0, 63) * 64 + raw[:, 2].clamp(0, 63)).long()
            target_dense.index_fill_(0, raw_keys, 1.0)
        target_vals = target_dense.index_select(0, keys)
        logit = logits[idx, : coords.shape[0]].float()
        if target_vals.numel() == 0:
            continue
        pos = target_vals.sum().clamp_min(1.0)
        neg = (target_vals.numel() - pos).clamp_min(1.0)
        pos_weight = (neg / pos).clamp(1.0, 1000.0)
        bce = F.binary_cross_entropy_with_logits(logit, target_vals, pos_weight=pos_weight)
        prob = logit.sigmoid()
        dice = dice_loss_prob(prob.view(1, -1), target_vals.view(1, -1), dims=(1,))
        losses.append(bce + dice)
        if collect_stats:
            bce_items.append(float(bce.detach().item()))
            dice_items.append(float(dice.detach().item()))

        if collect_stats:
            pred = prob > 0.5
            gt = target_vals > 0.5
            inter = (pred & gt).sum().float()
            union = (pred | gt).sum().float()
            pred_count = pred.sum().float()
            gt_count = gt.sum().float()
            ious.append(float((inter / union.clamp_min(1.0)).detach().item()))
            precisions.append(float(torch.where(pred_count > 0, inter / pred_count.clamp_min(1.0), (gt_count == 0).float()).detach().item()))
            recalls.append(float(torch.where(gt_count > 0, inter / gt_count.clamp_min(1.0), torch.ones_like(gt_count)).detach().item()))
            pred_counts.append(float(pred_count.detach().item()))
            gt_counts.append(float(gt_count.detach().item()))
    if not losses:
        zero = logits.sum() * 0.0
        return zero, {
            "voxel_bce": 0.0,
            "voxel_dice": 0.0,
            "voxel_iou": 1.0,
            "voxel_precision": 1.0,
            "voxel_recall": 1.0,
            "voxel_pred_count": 0.0,
            "voxel_gt_count": 0.0,
        }
    if not collect_stats:
        return torch.stack(losses).mean(), {}
    return torch.stack(losses).mean(), {
        "voxel_bce": float(np.mean(bce_items)),
        "voxel_dice": float(np.mean(dice_items)),
        "voxel_iou": float(np.mean(ious)),
        "voxel_precision": float(np.mean(precisions)),
        "voxel_recall": float(np.mean(recalls)),
        "voxel_pred_count": float(np.mean(pred_counts)),
        "voxel_gt_count": float(np.mean(gt_counts)),
    }


def _target_dense_from_coords(coords: torch.Tensor, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    target = torch.zeros((64, 64, 64), dtype=dtype, device=device)
    coords = torch.as_tensor(coords, dtype=torch.long, device=device)
    if coords.numel() > 0:
        target[coords[:, 0].clamp(0, 63), coords[:, 1].clamp(0, 63), coords[:, 2].clamp(0, 63)] = 1.0
    return target


def trilinear_splat_motion(
    coords: torch.Tensor,
    weights: torch.Tensor,
    transform_ab: torch.Tensor,
    scale_a: torch.Tensor,
    offset_a: torch.Tensor,
    scale_b: torch.Tensor,
    offset_b: torch.Tensor,
    *,
    resolution: int = 64,
) -> torch.Tensor:
    """Splat weights from angle-a voxel centers into angle-b voxel grid."""

    device = weights.device
    dtype = weights.dtype
    coords_f = torch.as_tensor(coords, dtype=dtype, device=device)
    if coords_f.numel() == 0:
        return torch.zeros((resolution, resolution, resolution), dtype=dtype, device=device)
    if coords_f.dim() != 2 or coords_f.shape[1] != 3:
        raise ValueError(f"motion coords must be [N,3], got {tuple(coords_f.shape)}")
    scale_a = torch.as_tensor(scale_a, dtype=dtype, device=device)
    scale_b = torch.as_tensor(scale_b, dtype=dtype, device=device)
    offset_a = torch.as_tensor(offset_a, dtype=dtype, device=device)
    offset_b = torch.as_tensor(offset_b, dtype=dtype, device=device)
    transform_ab = torch.as_tensor(transform_ab, dtype=dtype, device=device)

    unit_a = (coords_f + 0.5) / float(resolution) - 0.5
    world_a = (unit_a - offset_a.view(1, 3)) / scale_a.clamp_min(1.0e-8)
    ones = torch.ones((world_a.shape[0], 1), dtype=dtype, device=device)
    world_b = (torch.cat([world_a, ones], dim=1) @ transform_ab.t())[:, :3]
    unit_b = world_b * scale_b.clamp_min(1.0e-8) + offset_b.view(1, 3)
    grid_b = (unit_b + 0.5) * float(resolution) - 0.5

    base = torch.floor(grid_b).long()
    frac = (grid_b - base.to(dtype=dtype)).clamp(0.0, 1.0)
    volume = torch.zeros((resolution, resolution, resolution), dtype=dtype, device=device)
    weights = weights.to(device=device, dtype=dtype).view(-1)
    for dx in (0, 1):
        wx = (1.0 - frac[:, 0]) if dx == 0 else frac[:, 0]
        ix = base[:, 0] + dx
        for dy in (0, 1):
            wy = (1.0 - frac[:, 1]) if dy == 0 else frac[:, 1]
            iy = base[:, 1] + dy
            for dz in (0, 1):
                wz = (1.0 - frac[:, 2]) if dz == 0 else frac[:, 2]
                iz = base[:, 2] + dz
                valid = (ix >= 0) & (ix < resolution) & (iy >= 0) & (iy < resolution) & (iz >= 0) & (iz < resolution)
                if bool(valid.any()):
                    vals = weights * wx * wy * wz
                    volume.index_put_((ix[valid], iy[valid], iz[valid]), vals[valid], accumulate=True)
    return volume.clamp(0.0, 1.0)


def motion_consistency_loss_from_logits(
    logits: torch.Tensor,
    coords_list: list[torch.Tensor],
    batch: dict[str, Any],
    *,
    device: torch.device,
    use_gt_membership: bool = False,
    loss_kind: str = "bce_dice",
) -> tuple[torch.Tensor, dict[str, float]]:
    loss_kind = str(loss_kind)
    if loss_kind not in {"bce_dice", "bce"}:
        raise ValueError(f"unknown motion loss kind={loss_kind!r}")
    valid_flags = batch.get("motion_valid")
    if valid_flags is None:
        zero = logits.sum() * 0.0
        return zero, {
            "motion_loss": 0.0,
            "motion_bce": 0.0,
            "motion_dice": 0.0,
            "motion_items": 0.0,
            "motion_gt_sanity": 1.0 if use_gt_membership else 0.0,
        }
    valid_flags = valid_flags.to(device=device, dtype=torch.bool)
    losses = []
    bce_items = []
    dice_items = []
    for idx, coords in enumerate(coords_list):
        if idx >= int(valid_flags.shape[0]) or not bool(valid_flags[idx].detach().item()):
            continue
        coords = coords.to(device=device, dtype=torch.long)
        if coords.numel() == 0:
            continue
        if use_gt_membership:
            raw = torch.as_tensor(batch["raw_coords"][idx], dtype=torch.long, device=device)
            raw_keys = set()
            if raw.numel() > 0:
                raw_key_tensor = raw[:, 0] * 4096 + raw[:, 1] * 64 + raw[:, 2]
                raw_keys = {int(v) for v in raw_key_tensor.detach().cpu().tolist()}
            coord_keys = coords[:, 0] * 4096 + coords[:, 1] * 64 + coords[:, 2]
            weights = torch.tensor(
                [1.0 if int(v) in raw_keys else 0.0 for v in coord_keys.detach().cpu().tolist()],
                dtype=torch.float32,
                device=device,
            )
        else:
            weights = logits[idx, : coords.shape[0]].float().sigmoid()
        target_coords = batch["motion_target_coords_b"][idx].to(device=device, dtype=torch.long)
        if target_coords.numel() == 0:
            continue
        pred = trilinear_splat_motion(
            coords,
            weights,
            batch["motion_transform_ab"][idx].to(device=device),
            batch["motion_scale_a"][idx].to(device=device),
            batch["motion_offset_a"][idx].to(device=device),
            batch["motion_scale_b"][idx].to(device=device),
            batch["motion_offset_b"][idx].to(device=device),
            resolution=64,
        ).float()
        target = _target_dense_from_coords(target_coords, device=device, dtype=pred.dtype)
        with torch.cuda.amp.autocast(enabled=False):
            pred_bce = pred.float().clamp(1.0e-4, 1.0 - 1.0e-4)
            target_bce = target.float()
            bce = F.binary_cross_entropy(pred_bce, target_bce)
        dice = dice_loss_prob(pred.view(1, -1), target.view(1, -1), dims=(1,))
        losses.append(bce if loss_kind == "bce" else bce + dice)
        bce_items.append(float(bce.detach().item()))
        dice_items.append(float(dice.detach().item()))
    if not losses:
        zero = logits.sum() * 0.0
        return zero, {
            "motion_loss": 0.0,
            "motion_bce": 0.0,
            "motion_dice": 0.0,
            "motion_items": 0.0,
            "motion_gt_sanity": 1.0 if use_gt_membership else 0.0,
        }
    loss = torch.stack(losses).mean()
    return loss, {
        "motion_loss": float(loss.detach().item()),
        "motion_bce": float(np.mean(bce_items)),
        "motion_dice": float(np.mean(dice_items)),
        "motion_items": float(len(losses)),
        "motion_gt_sanity": 1.0 if use_gt_membership else 0.0,
    }


def xpart_ce_loss_from_logits(
    logits: torch.Tensor,
    coords_list: list[torch.Tensor],
    raw_coords: list[torch.Tensor | np.ndarray],
    batch: dict[str, Any],
    *,
    device: torch.device,
    background_logit: float = 0.0,
) -> tuple[torch.Tensor | None, dict[str, float]]:
    groups: dict[str, list[int]] = {}
    for idx, coords in enumerate(coords_list):
        if coords.numel() == 0:
            continue
        groups.setdefault(_object_angle_key_from_batch(batch, idx), []).append(idx)

    losses: list[torch.Tensor] = []
    voxel_count = 0
    fg_count = 0
    bg_count = 0
    group_count = 0
    for indices in groups.values():
        if not indices:
            continue
        key_tensors: list[torch.Tensor] = []
        for idx in indices:
            coords = coords_list[idx].to(device=device, dtype=torch.long)
            if coords.numel() == 0:
                key_tensors.append(torch.empty((0,), dtype=torch.long, device=device))
                continue
            key_tensors.append(coords[:, 0] * 4096 + coords[:, 1] * 64 + coords[:, 2])
        nonempty_keys = [keys for keys in key_tensors if keys.numel() > 0]
        if not nonempty_keys:
            continue
        unique_keys = torch.unique(torch.cat(nonempty_keys), sorted=True)
        group_count += 1
        class_logits = torch.full(
            (unique_keys.shape[0], len(indices) + 1),
            -1.0e4,
            dtype=torch.float32,
            device=device,
        )
        class_logits[:, 0] = float(background_logit)
        target = torch.zeros((unique_keys.shape[0],), dtype=torch.long, device=device)
        for class_idx, idx in enumerate(indices, start=1):
            keys = key_tensors[class_idx - 1]
            if keys.numel() == 0:
                continue
            positions = torch.searchsorted(unique_keys, keys)
            class_logits[positions, class_idx] = logits[idx, : keys.shape[0]].float()
            raw = torch.as_tensor(raw_coords[idx], dtype=torch.long, device=device)
            if raw.numel() == 0:
                continue
            raw_keys = raw[:, 0] * 4096 + raw[:, 1] * 64 + raw[:, 2]
            raw_positions = torch.searchsorted(unique_keys, raw_keys.clamp_min(0))
            in_range = raw_positions < unique_keys.shape[0]
            raw_positions = raw_positions[in_range]
            raw_keys = raw_keys[in_range]
            match = unique_keys[raw_positions] == raw_keys
            raw_positions = raw_positions[match]
            if raw_positions.numel() == 0:
                continue
            competitor = class_logits[raw_positions, class_idx] > -9999.0
            raw_positions = raw_positions[competitor]
            target[raw_positions] = class_idx
        losses.append(F.cross_entropy(class_logits, target))
        voxel_count += int(unique_keys.shape[0])
        fg = int((target > 0).sum().detach().item())
        fg_count += fg
        bg_count += int(target.numel()) - fg

    if not losses:
        return None, {
            "xpart_ce": 0.0,
            "xpart_groups": 0.0,
            "xpart_voxels": 0.0,
            "xpart_fg_ratio": 0.0,
        }
    ce = torch.stack(losses).mean()
    return ce, {
        "xpart_ce": float(ce.detach().item()),
        "xpart_groups": float(group_count),
        "xpart_voxels": float(voxel_count),
        "xpart_fg_ratio": float(fg_count / max(1, fg_count + bg_count)),
    }


@torch.no_grad()
def voxel_decode_metrics_from_forward(
    logits: torch.Tensor,
    coords_list: list[torch.Tensor],
    raw_coords: list[torch.Tensor],
    *,
    threshold: float = 0.5,
) -> list[dict[str, float]]:
    pred_coords = []
    for idx, coords in enumerate(coords_list):
        if coords.numel() == 0:
            pred_coords.append(torch.empty((0, 3), dtype=torch.long))
            continue
        prob = logits[idx, : coords.shape[0]].float().sigmoid()
        keep = prob > float(threshold)
        pred_coords.append(coords[keep].detach().long().cpu())
    return decode_metrics_for_batch(pred_coords, raw_coords)


@torch.no_grad()
def resolve_argmax_coords_from_forward(
    logits: torch.Tensor,
    coords_list: list[torch.Tensor],
    batch: dict[str, Any],
    *,
    threshold: float = 0.5,
) -> list[torch.Tensor]:
    pred_coords = _pred_coords_from_forward(logits, coords_list, threshold=threshold)
    by_group: dict[str, list[int]] = {}
    for idx, coords in enumerate(pred_coords):
        if coords.numel() == 0:
            continue
        by_group.setdefault(_object_angle_key_from_batch(batch, idx), []).append(idx)
    out = [coords.clone() for coords in pred_coords]
    for indices in by_group.values():
        coord_to_claims: dict[int, list[tuple[int, int, float]]] = {}
        for idx in indices:
            coords = pred_coords[idx]
            if coords.numel() == 0:
                continue
            orig = coords_list[idx]
            prob = logits[idx, : orig.shape[0]].float().sigmoid()
            keep = prob > float(threshold)
            kept_positions = torch.nonzero(keep, as_tuple=False).flatten()
            keys = coords[:, 0] * 4096 + coords[:, 1] * 64 + coords[:, 2]
            orig_keys = orig[:, 0] * 4096 + orig[:, 1] * 64 + orig[:, 2]
            key_to_prob = {
                int(orig_keys[int(pos)].detach().cpu().item()): float(prob[int(pos)].detach().cpu().item())
                for pos in kept_positions
            }
            for local_pos, key in enumerate(keys.tolist()):
                coord_to_claims.setdefault(int(key), []).append((idx, local_pos, key_to_prob.get(int(key), 0.0)))
        keep_positions: dict[int, set[int]] = {idx: set(range(pred_coords[idx].shape[0])) for idx in indices}
        for claims in coord_to_claims.values():
            if len(claims) <= 1:
                continue
            best = max(claims, key=lambda item: item[2])
            for idx, local_pos, _prob in claims:
                if idx != best[0]:
                    keep_positions[idx].discard(local_pos)
        for idx in indices:
            positions = sorted(keep_positions[idx])
            out[idx] = pred_coords[idx][positions] if positions else torch.empty((0, 3), dtype=torch.long)
    return out


def _coords_to_key_set(coords: torch.Tensor | np.ndarray) -> set[int]:
    coords_t = torch.as_tensor(coords, dtype=torch.long)
    if coords_t.numel() == 0:
        return set()
    keys = coords_t[:, 0] * 4096 + coords_t[:, 1] * 64 + coords_t[:, 2]
    return {int(v) for v in keys.detach().cpu().tolist()}


def _object_angle_key_from_batch(batch: dict[str, Any], idx: int) -> str:
    dataset_ids = batch.get("dataset_id", [""] * len(batch["obj_id"]))
    return f"{dataset_ids[idx]}::{batch['obj_id'][idx]}|angle_{int(batch['angle_idx'][idx])}"


def embedding_partition_loss(
    embeddings: torch.Tensor | None,
    coords_list: list[torch.Tensor],
    raw_coords: list[torch.Tensor | np.ndarray],
    batch: dict[str, Any],
    *,
    pull_margin: float = 0.5,
    push_margin: float = 1.5,
    max_voxels_per_part: int = 512,
) -> tuple[torch.Tensor | None, dict[str, float]]:
    if embeddings is None:
        return None, {
            "embed_pull": 0.0,
            "embed_push": 0.0,
            "embed_loss": 0.0,
            "embed_groups": 0.0,
            "embed_parts": 0.0,
            "embed_voxels": 0.0,
        }
    device = embeddings.device
    raw_sets = [_coords_to_key_set(raw) for raw in raw_coords]
    by_group: dict[str, list[dict[str, torch.Tensor]]] = {}
    total_voxels = 0
    for idx, coords in enumerate(coords_list):
        if coords.numel() == 0 or not raw_sets[idx]:
            continue
        keys = coords[:, 0] * 4096 + coords[:, 1] * 64 + coords[:, 2]
        target = torch.tensor(
            [int(v) in raw_sets[idx] for v in keys.detach().cpu().tolist()],
            dtype=torch.bool,
            device=device,
        )
        if not bool(target.any()):
            continue
        emb = embeddings[idx, : coords.shape[0]][target].float()
        if emb.numel() == 0:
            continue
        if int(max_voxels_per_part) > 0 and emb.shape[0] > int(max_voxels_per_part):
            perm = torch.randperm(emb.shape[0], device=device)[: int(max_voxels_per_part)]
            emb = emb[perm]
        emb = F.normalize(emb, dim=-1, eps=1.0e-6)
        total_voxels += int(emb.shape[0])
        by_group.setdefault(_object_angle_key_from_batch(batch, idx), []).append({"emb": emb})

    pull_terms: list[torch.Tensor] = []
    prototypes_by_group: list[list[torch.Tensor]] = []
    for parts in by_group.values():
        prototypes: list[torch.Tensor] = []
        for item in parts:
            emb = item["emb"]
            proto = F.normalize(emb.mean(dim=0), dim=0, eps=1.0e-6)
            prototypes.append(proto)
            dist = torch.linalg.vector_norm(emb - proto.unsqueeze(0), dim=-1)
            pull_terms.append(torch.clamp(dist - float(pull_margin), min=0.0).pow(2).mean())
        if len(prototypes) >= 2:
            prototypes_by_group.append(prototypes)

    push_terms: list[torch.Tensor] = []
    for prototypes in prototypes_by_group:
        for i, a in enumerate(prototypes):
            for b in prototypes[i + 1 :]:
                dist = torch.linalg.vector_norm(a - b, dim=-1)
                push_terms.append(torch.clamp(float(push_margin) - dist, min=0.0).pow(2))

    if not pull_terms and not push_terms:
        zero = embeddings.sum() * 0.0
        return zero, {
            "embed_pull": 0.0,
            "embed_push": 0.0,
            "embed_loss": 0.0,
            "embed_groups": float(len(by_group)),
            "embed_parts": 0.0,
            "embed_voxels": float(total_voxels),
        }
    pull = torch.stack(pull_terms).mean() if pull_terms else embeddings.sum() * 0.0
    push = torch.stack(push_terms).mean() if push_terms else embeddings.sum() * 0.0
    loss = pull + push
    return loss, {
        "embed_pull": float(pull.detach().item()),
        "embed_push": float(push.detach().item()),
        "embed_loss": float(loss.detach().item()),
        "embed_groups": float(sum(1 for parts in by_group.values() if len(parts) >= 2)),
        "embed_parts": float(sum(len(parts) for parts in by_group.values())),
        "embed_voxels": float(total_voxels),
    }


def _pred_coords_from_forward(
    logits: torch.Tensor,
    coords_list: list[torch.Tensor],
    *,
    threshold: float = 0.5,
) -> list[torch.Tensor]:
    pred_coords: list[torch.Tensor] = []
    for idx, coords in enumerate(coords_list):
        if coords.numel() == 0:
            pred_coords.append(torch.empty((0, 3), dtype=torch.long))
            continue
        prob = logits[idx, : coords.shape[0]].float().sigmoid()
        pred_coords.append(coords[prob > float(threshold)].detach().long().cpu())
    return pred_coords


def pairwise_overlap_from_coords(pred_coords: list[torch.Tensor], batch: dict[str, Any]) -> dict[int, dict[str, int]]:
    by_group: dict[str, list[int]] = {}
    for idx in range(len(pred_coords)):
        by_group.setdefault(_object_angle_key_from_batch(batch, idx), []).append(idx)
    out = {
        idx: {
            "part_overlap_voxels": 0,
            "object_overlap_voxels": 0,
            "object_overlap_max_pair": 0,
            "object_group_parts": len(by_group[_object_angle_key_from_batch(batch, idx)]),
        }
        for idx in range(len(pred_coords))
    }
    key_sets = [_coords_to_key_set(coords) for coords in pred_coords]
    for indices in by_group.values():
        object_total = 0
        object_max = 0
        part_totals = {idx: 0 for idx in indices}
        for pos, a in enumerate(indices):
            for b in indices[pos + 1 :]:
                count = len(key_sets[a] & key_sets[b])
                object_total += count
                object_max = max(object_max, count)
                part_totals[a] += count
                part_totals[b] += count
        for idx in indices:
            out[idx]["part_overlap_voxels"] = int(part_totals[idx])
            out[idx]["object_overlap_voxels"] = int(object_total)
            out[idx]["object_overlap_max_pair"] = int(object_max)
    return out


def partition_coords_by_embedding(
    logits: torch.Tensor,
    coords_list: list[torch.Tensor],
    embeddings: torch.Tensor | None,
    batch: dict[str, Any],
    *,
    threshold: float = 0.5,
) -> list[torch.Tensor]:
    pred_coords = _pred_coords_from_forward(logits, coords_list, threshold=threshold)
    if embeddings is None:
        return pred_coords

    by_group: dict[str, list[int]] = {}
    for idx, coords in enumerate(pred_coords):
        if coords.numel() == 0:
            continue
        by_group.setdefault(_object_angle_key_from_batch(batch, idx), []).append(idx)
    out = [coords.clone() for coords in pred_coords]
    for indices in by_group.values():
        coord_to_claims: dict[int, list[tuple[int, int]]] = {}
        prototypes: dict[int, torch.Tensor] = {}
        for idx in indices:
            coords = pred_coords[idx]
            if coords.numel() == 0:
                continue
            orig = coords_list[idx]
            prob = logits[idx, : orig.shape[0]].float().sigmoid()
            keep = prob > float(threshold)
            emb = embeddings[idx, : orig.shape[0]][keep].float()
            if emb.numel() == 0:
                continue
            prototypes[idx] = F.normalize(emb.mean(dim=0), dim=0, eps=1.0e-6)
            keys = coords[:, 0] * 4096 + coords[:, 1] * 64 + coords[:, 2]
            for local_pos, key in enumerate(keys.tolist()):
                coord_to_claims.setdefault(int(key), []).append((idx, local_pos))
        keep_positions: dict[int, set[int]] = {idx: set(range(pred_coords[idx].shape[0])) for idx in indices}
        for claims in coord_to_claims.values():
            if len(claims) <= 1:
                continue
            best_idx = claims[0][0]
            best_score = None
            for idx, local_pos in claims:
                if idx not in prototypes:
                    score = logits.new_tensor(float("-inf"))
                else:
                    orig = coords_list[idx]
                    pred = pred_coords[idx][local_pos].to(device=orig.device)
                    matches = (orig == pred).all(dim=1).nonzero(as_tuple=False).flatten()
                    if matches.numel() == 0:
                        score = logits.new_tensor(float("-inf"))
                    else:
                        emb = F.normalize(embeddings[idx, int(matches[0])].float(), dim=0, eps=1.0e-6)
                        score = (emb * prototypes[idx]).sum()
                if best_score is None or float(score.detach().cpu().item()) > float(best_score.detach().cpu().item()):
                    best_score = score
                    best_idx = idx
            for idx, local_pos in claims:
                if idx != best_idx:
                    keep_positions[idx].discard(local_pos)
        for idx in indices:
            if pred_coords[idx].numel() == 0:
                continue
            positions = sorted(keep_positions[idx])
            out[idx] = pred_coords[idx][positions] if positions else torch.empty((0, 3), dtype=torch.long)
    return out


def latent_signal_stats(
    pred: torch.Tensor,
    target: torch.Tensor,
    empty_code: torch.Tensor,
    support_mask: torch.Tensor,
) -> dict[str, torch.Tensor]:
    empty = empty_code.to(device=target.device, dtype=target.dtype)
    if empty.dim() == 4:
        empty = empty.unsqueeze(0)
    abs_err = (pred - target).abs()
    full_l1 = abs_err.mean(dim=(1, 2, 3, 4))
    signal = (target - empty).abs().mean(dim=(1, 2, 3, 4)).clamp_min(1.0e-6)
    support = support_mask.to(device=target.device, dtype=target.dtype).unsqueeze(1)
    support_denom = (support.sum(dim=(1, 2, 3, 4)) * target.shape[1]).clamp_min(1.0)
    support_l1 = (abs_err * support).sum(dim=(1, 2, 3, 4)) / support_denom
    return {
        "full_l1": full_l1,
        "signal": signal,
        "signal_norm_l1": full_l1 / signal,
        "support_l1": support_l1,
    }


def train_step(
    model: PromptablePartLatentSegNet,
    decoder,
    batch: dict[str, Any],
    empty_code: torch.Tensor,
    *,
    device: torch.device,
    decode_weight: float,
    latent_part_weight: float,
    mask_augment: bool,
    mask_only: bool,
    mask_target: str,
    support_multiplier: float,
    latent_loss_mode: str,
    route: str,
    voxel_loss_weight: float,
    voxel_max_tokens: int,
    xpart_ce_weight: float,
    motion_loss_weight: float,
    motion_loss_kind: str,
    embed_loss_weight: float,
    embed_pull_margin: float,
    embed_push_margin: float,
    embed_max_voxels_per_part: int,
    view_dropout: bool,
    min_views: int,
    min_prompt_views: int,
    view_dropout_start_step: int,
    focal_gamma: float,
    boundary_weight: float,
    semantic_loss_weight: float,
    use_packed_whole_occ: bool,
    step: int | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    z_global = batch["z_global"].to(device=device, dtype=torch.float32)
    masks2d = batch["masks2d"].to(device=device, dtype=torch.float32)
    no_prompt_before_dropout = warn_all_empty_prompt_masks(batch, masks2d, context="train/pre_dropout")
    dropout_stats = {
        "view_dropout_active": 0.0,
        "view_dropout_single_before": float((visible_view_counts(masks2d) == 1).sum().detach().item()),
        "view_dropout_single_after": float((visible_view_counts(masks2d) == 1).sum().detach().item()),
        "view_dropout_min_prompt_views": float(max(1, int(min_prompt_views))),
        "view_dropout_skipped_guard": 0.0,
        "view_dropout_dropped_prompts": 0.0,
    }
    dropout_enabled = bool(view_dropout) and (step is None or int(step) >= int(view_dropout_start_step))
    if dropout_enabled:
        masks2d, dropout_stats = apply_view_dropout(
            masks2d,
            min_views=int(min_views),
            min_prompt_views=int(min_prompt_views),
        )
        dropout_stats["view_dropout_active"] = 1.0
    no_prompt_after_dropout = warn_all_empty_prompt_masks(batch, masks2d, context="train/post_dropout")
    latent_gt = batch["latent_gt"].to(device=device, dtype=torch.float32)
    m_raw = batch["m_gt"].to(device=device, dtype=torch.float32)
    boundary_raw = batch.get("m_boundary")
    boundary_flat = None
    if boundary_raw is not None and float(boundary_weight) > 1.0:
        boundary_flat = boundary_raw.to(device=device, dtype=torch.float32).reshape(boundary_raw.shape[0], -1)
    if mask_target == "support":
        m_gt, support_noise, support_threshold = latent_support_mask(
            latent_gt,
            empty_code.to(device),
            m_raw,
            multiplier=float(support_multiplier),
        )
    else:
        m_gt = m_raw
        support_noise = torch.zeros((m_gt.shape[0],), device=device)
        support_threshold = torch.zeros((m_gt.shape[0],), device=device)
    modes = ["none", "dilate", "erode"] if bool(mask_augment) else ["none"]
    mode = modes[int(torch.randint(0, len(modes), ()).item())]
    m_head2 = mask_morphology(m_gt, mode)
    if str(route) == "voxel":
        out_v = model(
            z_global,
            masks2d,
            candidate_cells=mask_morphology(m_gt, "dilate"),
            full_occ=full_occ_for_batch(
                decoder,
                z_global,
                batch,
                device=device,
                use_packed_whole_occ=bool(use_packed_whole_occ),
            ),
            max_voxels_per_sample=int(voxel_max_tokens),
        )
        no_prompt_hits = record_no_prompt_from_output(out_v, batch, context="train/forward", step=step)
        m_flat = m_gt.reshape(m_gt.shape[0], -1)
        l_mask, mask_items = mask_loss(
            out_v["m_logit"],
            m_flat,
            focal_gamma=float(focal_gamma),
            boundary_flat=boundary_flat,
            boundary_weight=float(boundary_weight),
        )
        l_voxel, voxel_items = voxel_loss_from_logits(
            out_v["voxel_logits"],
            out_v["voxel_coords"],
            batch["raw_coords"],
            device=device,
        )
        if float(xpart_ce_weight) > 0:
            l_xpart, xpart_items = xpart_ce_loss_from_logits(
                out_v["voxel_logits"],
                out_v["voxel_coords"],
                batch["raw_coords"],
                batch,
                device=device,
            )
        else:
            l_xpart = None
            xpart_items = {
                "xpart_ce": 0.0,
                "xpart_groups": 0.0,
                "xpart_voxels": 0.0,
                "xpart_fg_ratio": 0.0,
            }
        if float(motion_loss_weight) > 0:
            l_motion, motion_items = motion_consistency_loss_from_logits(
                out_v["voxel_logits"],
                out_v["voxel_coords"],
                batch,
                device=device,
                use_gt_membership=False,
                loss_kind=str(motion_loss_kind),
            )
        else:
            l_motion = None
            motion_items = {
                "motion_loss": 0.0,
                "motion_bce": 0.0,
                "motion_dice": 0.0,
                "motion_items": 0.0,
                "motion_gt_sanity": 0.0,
            }
        l_sem, sem_items = semantic_loss_from_logits(
            out_v.get("semantic_logits"),
            batch.get("semantic_type_id").to(device) if "semantic_type_id" in batch else None,
            weight=float(semantic_loss_weight),
        )
        if float(embed_loss_weight) > 0:
            l_embed, embed_items = embedding_partition_loss(
                out_v.get("voxel_embeddings"),
                out_v["voxel_coords"],
                batch["raw_coords"],
                batch,
                pull_margin=float(embed_pull_margin),
                push_margin=float(embed_push_margin),
                max_voxels_per_part=int(embed_max_voxels_per_part),
            )
        else:
            l_embed = None
            embed_items = {
                "embed_pull": 0.0,
                "embed_push": 0.0,
                "embed_loss": 0.0,
                "embed_groups": 0.0,
                "embed_parts": 0.0,
                "embed_voxels": 0.0,
            }
        loss = l_mask + float(voxel_loss_weight) * l_voxel
        if l_xpart is not None:
            loss = loss + float(xpart_ce_weight) * l_xpart
        if l_motion is not None:
            loss = loss + float(motion_loss_weight) * l_motion
        if l_sem is not None:
            loss = loss + l_sem
        if l_embed is not None:
            loss = loss + float(embed_loss_weight) * l_embed
        with torch.no_grad():
            mm = mask_metrics_from_logits(out_v["m_logit"].detach(), m_flat)
        return loss, {
            **mask_items,
            **voxel_items,
            **xpart_items,
            **motion_items,
            **sem_items,
            **embed_items,
            "xpart_ce_weight": float(xpart_ce_weight),
            "motion_loss_weight": float(motion_loss_weight),
            "embed_loss_weight": float(embed_loss_weight),
            "latent_l1_weighted": 0.0,
            "latent_l1_unweighted": 0.0,
            "latent_l1_signal_norm": 0.0,
            "support_l1": 0.0,
            "decode_dice": 0.0,
            "total": float(loss.detach().item()),
            "morph": "voxel_dilate",
            "support_noise": float(support_noise.detach().mean().item()),
            "support_threshold": float(support_threshold.detach().mean().item()),
            "support_cell_count": float(m_gt.detach().sum(dim=(1, 2, 3)).mean().item()),
            "no_prompt_before_dropout": float(no_prompt_before_dropout),
            "no_prompt_after_dropout": float(no_prompt_after_dropout),
            "no_prompt_forward": float(no_prompt_hits),
            **dropout_stats,
            **mm,
        }

    out = model(z_global, masks2d, empty_code.to(device), m_override=m_head2)
    no_prompt_hits = record_no_prompt_from_output(out, batch, context="train/forward", step=step)
    m_flat = m_gt.reshape(m_gt.shape[0], -1)
    l_mask, mask_items = mask_loss(
        out["m_logit"],
        m_flat,
        focal_gamma=float(focal_gamma),
        boundary_flat=boundary_flat,
        boundary_weight=float(boundary_weight),
    )
    if bool(mask_only):
        with torch.no_grad():
            mm = mask_metrics_from_logits(out["m_logit"].detach(), m_flat)
        return l_mask, {
            **mask_items,
            "latent_l1_weighted": 0.0,
            "latent_l1_unweighted": 0.0,
            "decode_dice": 0.0,
            "total": float(l_mask.detach().item()),
            "morph": mode,
            "no_prompt_before_dropout": float(no_prompt_before_dropout),
            "no_prompt_after_dropout": float(no_prompt_after_dropout),
            "no_prompt_forward": float(no_prompt_hits),
            **mm,
        }
    stats = latent_signal_stats(out["part_latent"], latent_gt, empty_code.to(device), m_gt)
    if str(latent_loss_mode) == "signal_normalized":
        l_latent = stats["signal_norm_l1"].mean()
    else:
        cell_weight = 1.0 + (float(latent_part_weight) - 1.0) * m_gt.unsqueeze(1)
        l_latent = ((out["part_latent"] - latent_gt).abs() * cell_weight).mean()
    l_decode = decode_dice_loss(decoder, out["part_latent"], batch["raw_coords"], device=device)
    l_sem, sem_items = semantic_loss_from_logits(
        out.get("semantic_logits"),
        batch.get("semantic_type_id").to(device) if "semantic_type_id" in batch else None,
        weight=float(semantic_loss_weight),
    )
    loss = l_mask + l_latent + float(decode_weight) * l_decode
    if l_sem is not None:
        loss = loss + l_sem
    with torch.no_grad():
        pred = out["m_logit"].detach()
        mm = mask_metrics_from_logits(pred, m_flat)
    return loss, {
        **mask_items,
        "latent_loss": float(l_latent.detach().item()),
        "latent_l1_weighted": float(l_latent.detach().item()),
        "latent_l1_unweighted": float(stats["full_l1"].mean().detach().item()),
        "latent_l1_signal_norm": float(stats["signal_norm_l1"].mean().detach().item()),
        "support_l1": float(stats["support_l1"].mean().detach().item()),
        "latent_signal": float(stats["signal"].mean().detach().item()),
        "decode_dice": float(l_decode.detach().item()),
        **sem_items,
        "total": float(loss.detach().item()),
        "morph": mode,
        "support_noise": float(support_noise.detach().mean().item()),
        "support_threshold": float(support_threshold.detach().mean().item()),
        "support_cell_count": float(m_gt.detach().sum(dim=(1, 2, 3)).mean().item()),
        "no_prompt_before_dropout": float(no_prompt_before_dropout),
        "no_prompt_after_dropout": float(no_prompt_after_dropout),
        "no_prompt_forward": float(no_prompt_hits),
        **dropout_stats,
        **mm,
    }


@torch.no_grad()
def evaluate(
    model: PromptablePartLatentSegNet,
    decoder,
    loader: DataLoader,
    empty_code: torch.Tensor,
    *,
    device: torch.device,
    max_rows: int = 0,
    write_visuals_dir: Path | None = None,
    mask_only: bool = False,
    mask_target: str = "raw",
    support_multiplier: float = 4.0,
    route: str = "latent",
    voxel_max_tokens: int = 0,
    use_packed_whole_occ: bool = False,
    infer_resolve: str = "independent",
) -> dict[str, Any]:
    model.eval()
    rows: list[dict[str, Any]] = []
    count = 0
    for batch in loader:
        z_global = batch["z_global"].to(device=device, dtype=torch.float32)
        masks2d = batch["masks2d"].to(device=device, dtype=torch.float32)
        warn_all_empty_prompt_masks(batch, masks2d, context="eval")
        latent_gt = batch["latent_gt"].to(device=device, dtype=torch.float32)
        m_raw = batch["m_gt"].to(device=device, dtype=torch.float32)
        if mask_target == "support":
            m_gt, _noise, _threshold = latent_support_mask(
                latent_gt,
                empty_code.to(device),
                m_raw,
                multiplier=float(support_multiplier),
            )
        else:
            m_gt = m_raw
        m_flat = m_gt.reshape(m_gt.shape[0], -1)

        if str(route) == "voxel":
            full_occ = full_occ_for_batch(
                decoder,
                z_global,
                batch,
                device=device,
                use_packed_whole_occ=bool(use_packed_whole_occ),
            )
            out_gt = model(
                z_global,
                masks2d,
                candidate_cells=mask_morphology(m_gt, "dilate"),
                full_occ=full_occ,
                max_voxels_per_sample=int(voxel_max_tokens),
            )
            record_no_prompt_from_output(out_gt, batch, context="eval/gt")
            pred_m = (out_gt["m_logit"].sigmoid() > 0.5).float().view(m_gt.shape)
            out_pred = model(
                z_global,
                masks2d,
                candidate_cells=mask_morphology(pred_m, "dilate"),
                full_occ=full_occ,
                max_voxels_per_sample=int(voxel_max_tokens),
            )
            record_no_prompt_from_output(out_pred, batch, context="eval/pred")
            gt_metrics = voxel_decode_metrics_from_forward(out_gt["voxel_logits"], out_gt["voxel_coords"], batch["raw_coords"])
            if str(infer_resolve) == "argmax":
                pred_coords = resolve_argmax_coords_from_forward(out_pred["voxel_logits"], out_pred["voxel_coords"], batch)
                pred_metrics = decode_metrics_for_batch(pred_coords, batch["raw_coords"])
            else:
                pred_metrics = voxel_decode_metrics_from_forward(out_pred["voxel_logits"], out_pred["voxel_coords"], batch["raw_coords"])
                pred_coords = _pred_coords_from_forward(out_pred["voxel_logits"], out_pred["voxel_coords"])
            partition_coords = partition_coords_by_embedding(
                out_pred["voxel_logits"],
                out_pred["voxel_coords"],
                out_pred.get("voxel_embeddings"),
                batch,
            )
            pred_overlap = pairwise_overlap_from_coords(pred_coords, batch)
            partition_overlap = pairwise_overlap_from_coords(partition_coords, batch)
            partition_metrics = decode_metrics_for_batch(partition_coords, batch["raw_coords"])
            mask_prob = out_gt["m_logit"].sigmoid().reshape(m_gt.shape[0], -1)
            mask_pred = mask_prob > 0.5
            mask_gt = m_flat.bool()
            for idx in range(z_global.shape[0]):
                inter = (mask_pred[idx] & mask_gt[idx]).sum().float()
                union = (mask_pred[idx] | mask_gt[idx]).sum().float()
                raw_count = int(batch["raw_count"][idx].item())
                row = {
                    "obj_id": batch["obj_id"][idx],
                    "dataset_id": batch["dataset_id"][idx],
                    "angle_idx": int(batch["angle_idx"][idx]),
                    "sample_id": batch["sample_id"][idx],
                    "part_name": batch["part_name"][idx],
                    "semantic_type": batch["semantic_type"][idx],
                    "raw_count": raw_count,
                    "mask_visible_pixels": int(masks2d[idx].sum().detach().cpu().item()),
                    "bucket": bucket_name(raw_count),
                    "cell_iou": float((inter / union.clamp_min(1.0)).item()),
                    "cell_pred_count": int(mask_pred[idx].sum().item()),
                    "cell_gt_count": int(mask_gt[idx].sum().item()),
                    "support_l1": float("nan"),
                    "latent_l1": float("nan"),
                    "latent_l1_signal_norm": float("nan"),
                    "latent_signal": float("nan"),
                    "mask_target": str(mask_target),
                    "head2_gtm_decode_iou": gt_metrics[idx]["decode_iou"],
                    "head2_gtm_precision": gt_metrics[idx]["decode_precision"],
                    "head2_gtm_recall": gt_metrics[idx]["decode_recall"],
                    "e2e_decode_iou": pred_metrics[idx]["decode_iou"],
                    "e2e_precision": pred_metrics[idx]["decode_precision"],
                    "e2e_recall": pred_metrics[idx]["decode_recall"],
                    "partition_e2e_decode_iou": partition_metrics[idx]["decode_iou"],
                    "partition_e2e_precision": partition_metrics[idx]["decode_precision"],
                    "partition_e2e_recall": partition_metrics[idx]["decode_recall"],
                    "gtm_pred_count": gt_metrics[idx]["pred_count"],
                    "e2e_pred_count": pred_metrics[idx]["pred_count"],
                    "partition_e2e_pred_count": partition_metrics[idx]["pred_count"],
                    "part_overlap_voxels": pred_overlap[idx]["part_overlap_voxels"],
                    "object_overlap_voxels": pred_overlap[idx]["object_overlap_voxels"],
                    "object_overlap_max_pair": pred_overlap[idx]["object_overlap_max_pair"],
                    "partition_part_overlap_voxels": partition_overlap[idx]["part_overlap_voxels"],
                    "partition_object_overlap_voxels": partition_overlap[idx]["object_overlap_voxels"],
                    "object_group_parts": pred_overlap[idx]["object_group_parts"],
                    "route": "voxel",
                }
                rows.append(row)
                count += 1
            if int(max_rows) > 0 and count >= int(max_rows):
                break
            continue

        out_gt = model(z_global, masks2d, empty_code.to(device), m_override=m_gt)
        record_no_prompt_from_output(out_gt, batch, context="eval/gt")
        latent_stats = latent_signal_stats(out_gt["part_latent"], latent_gt, empty_code.to(device), m_gt)
        pred_m = (out_gt["m_logit"].sigmoid() > 0.5).float().view(m_gt.shape)
        if bool(mask_only):
            gt_metrics = [{"decode_iou": float("nan"), "decode_precision": float("nan"), "decode_recall": float("nan"), "pred_count": 0} for _ in range(z_global.shape[0])]
            pred_metrics = [{"decode_iou": float("nan"), "decode_precision": float("nan"), "decode_recall": float("nan"), "pred_count": 0} for _ in range(z_global.shape[0])]
        else:
            out_pred = model(z_global, masks2d, empty_code.to(device), m_override=pred_m)
            record_no_prompt_from_output(out_pred, batch, context="eval/pred")
            gt_coords = decode_latents_to_coords(decoder, out_gt["part_latent"])
            pred_coords = decode_latents_to_coords(decoder, out_pred["part_latent"])
            gt_metrics = decode_metrics_for_batch(gt_coords, batch["raw_coords"])
            pred_metrics = decode_metrics_for_batch(pred_coords, batch["raw_coords"])

        mask_prob = out_gt["m_logit"].sigmoid().reshape(m_gt.shape[0], -1)
        mask_pred = mask_prob > 0.5
        mask_gt = m_flat.bool()
        for idx in range(z_global.shape[0]):
            inter = (mask_pred[idx] & mask_gt[idx]).sum().float()
            union = (mask_pred[idx] | mask_gt[idx]).sum().float()
            raw_count = int(batch["raw_count"][idx].item())
            row = {
                "obj_id": batch["obj_id"][idx],
                "dataset_id": batch["dataset_id"][idx],
                "angle_idx": int(batch["angle_idx"][idx]),
                "sample_id": batch["sample_id"][idx],
                "part_name": batch["part_name"][idx],
                "semantic_type": batch["semantic_type"][idx],
                "raw_count": raw_count,
                "mask_visible_pixels": int(masks2d[idx].sum().detach().cpu().item()),
                "bucket": bucket_name(raw_count),
                "cell_iou": float((inter / union.clamp_min(1.0)).item()),
                "cell_pred_count": int(mask_pred[idx].sum().item()),
                "cell_gt_count": int(mask_gt[idx].sum().item()),
                "support_l1": float(latent_stats["support_l1"][idx].item()) if not bool(mask_only) else float("nan"),
                "latent_l1": float(latent_stats["full_l1"][idx].item()) if not bool(mask_only) else float("nan"),
                "latent_l1_signal_norm": float(latent_stats["signal_norm_l1"][idx].item()) if not bool(mask_only) else float("nan"),
                "latent_signal": float(latent_stats["signal"][idx].item()) if not bool(mask_only) else float("nan"),
                "mask_target": str(mask_target),
                "head2_gtm_decode_iou": gt_metrics[idx]["decode_iou"],
                "head2_gtm_precision": gt_metrics[idx]["decode_precision"],
                "head2_gtm_recall": gt_metrics[idx]["decode_recall"],
                "e2e_decode_iou": pred_metrics[idx]["decode_iou"],
                "e2e_precision": pred_metrics[idx]["decode_precision"],
                "e2e_recall": pred_metrics[idx]["decode_recall"],
                "partition_e2e_decode_iou": float("nan"),
                "partition_e2e_precision": float("nan"),
                "partition_e2e_recall": float("nan"),
                "gtm_pred_count": gt_metrics[idx]["pred_count"],
                "e2e_pred_count": pred_metrics[idx]["pred_count"],
                "partition_e2e_pred_count": 0,
                "part_overlap_voxels": 0,
                "object_overlap_voxels": 0,
                "object_overlap_max_pair": 0,
                "partition_part_overlap_voxels": 0,
                "partition_object_overlap_voxels": 0,
                "object_group_parts": 1,
                "route": "latent",
            }
            rows.append(row)
            count += 1
        if int(max_rows) > 0 and count >= int(max_rows):
            break

    summary = summarize_by_bucket(
        rows,
        (
            "cell_iou",
            "support_l1",
            "latent_l1_signal_norm",
            "head2_gtm_decode_iou",
            "head2_gtm_precision",
            "head2_gtm_recall",
            "e2e_decode_iou",
            "e2e_precision",
            "e2e_recall",
            "partition_e2e_decode_iou",
            "partition_e2e_precision",
            "partition_e2e_recall",
            "part_overlap_voxels",
            "object_overlap_voxels",
            "partition_object_overlap_voxels",
        ),
    )
    all_summary = {
        "n": len(rows),
        "cell_iou": float(np.mean([r["cell_iou"] for r in rows])) if rows else float("nan"),
        "support_l1": float(np.mean([r["support_l1"] for r in rows])) if rows else float("nan"),
        "latent_l1_signal_norm": float(np.mean([r["latent_l1_signal_norm"] for r in rows])) if rows else float("nan"),
        "head2_gtm_decode_iou": float(np.mean([r["head2_gtm_decode_iou"] for r in rows])) if rows else float("nan"),
        "e2e_decode_iou": float(np.mean([r["e2e_decode_iou"] for r in rows])) if rows else float("nan"),
        "partition_e2e_decode_iou": float(np.nanmean([r.get("partition_e2e_decode_iou", float("nan")) for r in rows])) if rows else float("nan"),
        "part_overlap_voxels": float(np.mean([r.get("part_overlap_voxels", 0) for r in rows])) if rows else float("nan"),
        "object_overlap_voxels": float(np.mean([r.get("object_overlap_voxels", 0) for r in rows])) if rows else float("nan"),
        "partition_object_overlap_voxels": float(np.mean([r.get("partition_object_overlap_voxels", 0) for r in rows])) if rows else float("nan"),
    }
    summary["all"] = all_summary

    if write_visuals_dir is not None and rows:
        write_visuals_dir.mkdir(parents=True, exist_ok=True)
        write_eval_table_png(rows[: min(24, len(rows))], write_visuals_dir / "eval_table.png")
    return {"rows": rows, "summary": summary}


def write_eval_table_png(rows: list[dict[str, Any]], out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(14, max(3, 0.35 * len(rows))), dpi=150)
    try:
        ax.axis("off")
        headers = ["obj", "part", "raw", "cell", "gtm_iou", "e2e_iou"]
        data = [
            [
                row["obj_id"],
                row["part_name"][:24],
                row["raw_count"],
                f"{row['cell_iou']:.3f}",
                f"{row['head2_gtm_decode_iou']:.3f}",
                f"{row['e2e_decode_iou']:.3f}",
            ]
            for row in rows
        ]
        table = ax.table(cellText=data, colLabels=headers, loc="center", cellLoc="left")
        table.auto_set_font_size(False)
        table.set_fontsize(7)
        table.scale(1, 1.3)
        fig.tight_layout()
        try:
            fig.savefig(out_path)
        except OSError as exc:
            print(f"[PromptSeg-warning] failed to write eval table png {out_path}: {exc}", flush=True)
    finally:
        plt.close(fig)


def choose_gate2_object_ids(rows: list[Any], *, total: int, heldout: int) -> tuple[list[str], list[str], list[str]]:
    obj_ids = []
    for row in rows:
        key = object_key(row)
        if key not in obj_ids:
            obj_ids.append(key)
    obj_ids = obj_ids[: int(total)]
    obj_to_rows: dict[str, list[Any]] = {}
    for row in rows:
        key = object_key(row)
        if key in obj_ids:
            obj_to_rows.setdefault(key, []).append(row)

    def has_multi_button(oid: str) -> bool:
        return sum(1 for row in obj_to_rows.get(oid, []) if "button" in row.part_name.lower()) >= 2

    def has_door_lid(oid: str) -> bool:
        return any(("door" in row.part_name.lower()) or ("lid" in row.part_name.lower()) for row in obj_to_rows.get(oid, []))

    heldout_ids: list[str] = []
    for pred in (has_multi_button, has_door_lid):
        for oid in obj_ids:
            if oid not in heldout_ids and pred(oid):
                heldout_ids.append(oid)
                break
    for oid in reversed(obj_ids):
        if len(heldout_ids) >= int(heldout):
            break
        if oid not in heldout_ids:
            heldout_ids.append(oid)
    heldout_set = set(heldout_ids[: int(heldout)])
    train_ids = [oid for oid in obj_ids if oid not in heldout_set]
    return obj_ids, train_ids, [oid for oid in obj_ids if oid in heldout_set]


def rows_from_proxy_spec(rows: list[Any], specs: list[dict[str, Any]]) -> list[Any]:
    selected = []
    for spec in specs:
        if "object_key" in spec and "angle_idx" in spec and "part_name" in spec:
            matches = [
                row
                for row in rows
                if object_key(row) == str(spec["object_key"])
                and row.angle_idx == int(spec["angle_idx"])
                and row.part_name == str(spec["part_name"])
            ]
        elif "obj_id" in spec and "angle_idx" in spec and "part_name" in spec:
            matches = [
                row
                for row in rows
                if row.obj_id == str(spec["obj_id"])
                and ("dataset_id" not in spec or row.dataset_id == str(spec["dataset_id"]))
                and row.angle_idx == int(spec["angle_idx"])
                and row.part_name == str(spec["part_name"])
            ]
        elif "object_key" in spec:
            matches = [row for row in rows if object_key(row) == str(spec["object_key"])]
        elif "obj_id" in spec:
            matches = [
                row
                for row in rows
                if row.obj_id == str(spec["obj_id"])
                and ("dataset_id" not in spec or row.dataset_id == str(spec["dataset_id"]))
            ]
        else:
            raise KeyError(f"proxy spec must contain obj_id or object_key, got {spec}")
        if not matches:
            raise RuntimeError(f"proxy spec matched zero rows: {spec}")
        selected.extend(matches)
    return selected


def load_proxy_rows(path: Path, rows: list[Any]) -> tuple[list[Any], list[Any], dict[str, Any]]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"proxy json not found: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if "train" not in data or "heldout" not in data:
        raise KeyError(f"{path} must contain train and heldout lists")
    train_rows = rows_from_proxy_spec(rows, data["train"])
    heldout_rows = rows_from_proxy_spec(rows, data["heldout"])
    return train_rows, heldout_rows, data


def rows_from_packed_index(packed_dir: Path) -> tuple[list[PartRow], dict[str, Any]]:
    packed_dir = Path(packed_dir)
    index_path = packed_dir / "index.json"
    if not index_path.is_file():
        raise FileNotFoundError(f"packed index not found: {index_path}")
    index = json.loads(index_path.read_text(encoding="utf-8"))
    entries = list(index.get("entries", []))
    if not entries:
        raise RuntimeError(f"packed index has zero entries: {index_path}")
    datasets = {
        str(item.get("dataset_id", "")): item
        for item in index.get("datasets", [])
        if isinstance(item, dict)
    }
    rows: list[PartRow] = []
    for global_idx, entry in enumerate(entries):
        dataset_id = str(entry.get("dataset_id", ""))
        dataset_info = datasets.get(dataset_id, {})
        manifest_paths = dataset_info.get("manifest_paths", [])
        if isinstance(manifest_paths, list):
            manifest_path = ",".join(str(path) for path in manifest_paths)
        else:
            manifest_path = str(manifest_paths or "")
        part_name = str(entry.get("part_name", ""))
        rows.append(
            PartRow(
                sample_idx=int(global_idx),
                part_idx=0,
                obj_id=str(entry.get("obj_id", "")),
                angle_idx=int(entry.get("angle_idx", 0)),
                sample_id=str(entry.get("sample_id", "")),
                part_name=part_name,
                semantic_type=str(entry.get("semantic_type", part_name.split("_")[0] if part_name else "")),
                original_label=int(entry.get("original_label", 0)),
                raw_count=int(entry.get("raw_count", 0) or 0),
                view_indices=(),
                dataset_id=dataset_id,
                data_root=str(dataset_info.get("data_root", "")),
                manifest_path=manifest_path,
                category=str(entry.get("category", "")),
                object_name=str(entry.get("object_name", "")),
                part_item_name=str(entry.get("part_item_name", "")),
                part_joint=str(entry.get("part_joint", "")),
                sample_part_names=str(entry.get("sample_part_names", "")),
                visible_view_count=int(entry.get("visible_view_count", 0) or 0),
            )
        )
    meta = {
        "packed_index": str(index_path),
        "packed_rows": len(rows),
        "datasets": index.get("datasets", []),
        "split_json": index.get("split_json"),
    }
    return rows, meta


def select_rows(args: argparse.Namespace):
    split = load_official_split(args.split_json) if args.split_json is not None else None
    if args.packed_dir is not None and (Path(args.packed_dir) / "index.json").is_file():
        rows, packed_meta = rows_from_packed_index(args.packed_dir)
        base_ds = None
        if args.selection_json is not None:
            specs = json.loads(args.selection_json.read_text(encoding="utf-8"))
            selected = []
            for spec in specs:
                matches = [
                    row
                    for row in rows
                    if row.obj_id == str(spec["obj_id"])
                    and ("dataset_id" not in spec or row.dataset_id == str(spec["dataset_id"]))
                    and row.angle_idx == int(spec["angle_idx"])
                    and row.part_name == str(spec["part_name"])
                ]
                if len(matches) != 1:
                    raise RuntimeError(f"selection spec matched {len(matches)} rows: {spec}")
                selected.append(matches[0])
            return base_ds, selected, selected, selected, selected, {
                "selection_json": str(args.selection_json),
                "rows": specs,
                "packed": packed_meta,
            }
        if args.mode == "gate1":
            selected, gate1_meta = pick_gate1_rows(rows)
            return base_ds, selected, selected, selected, selected, {
                "gate1_selection": gate1_meta,
                "packed": packed_meta,
            }
        if args.split_json is not None:
            assert split is not None
            train_refs = split.get("train_keys", split["train_ids"])
            heldout_refs = split.get("heldout_keys", split["heldout_ids"])
            train = rows_for_obj_ids(rows, train_refs)
            val = rows_for_obj_ids(rows, heldout_refs)
            train_set = {object_key(row) for row in train}
            heldout_set = {object_key(row) for row in val}
            if train_set & heldout_set:
                raise RuntimeError("official split object overlap detected")
            proxy_train = train
            proxy_val = val
            if args.proxy_json is not None:
                proxy_train, proxy_val, proxy_meta = load_proxy_rows(args.proxy_json, rows)
            else:
                proxy_meta = None
            return base_ds, train, proxy_train, proxy_val, val, {
                "split_json": str(args.split_json),
                "train_obj_ids": list(map(str, train_refs)),
                "heldout_obj_ids": list(map(str, heldout_refs)),
                "train_obj_count": len(train_set),
                "heldout_obj_count": len(heldout_set),
                "proxy_json": str(args.proxy_json) if args.proxy_json is not None else None,
                "proxy": proxy_meta,
                "coverage": split.get("coverage", {}),
                "packed": packed_meta,
            }
        train, val = split_rows_by_obj(rows, heldout_fraction=float(args.heldout_fraction), seed=int(args.seed))
        return base_ds, train, train, val, val, {"packed": packed_meta}

    specs = dataset_specs_from_split(split or {})
    base_datasets = make_base_datasets(specs)
    base_ds = MultiPromptableBaseDataset(base_datasets)
    rows = enumerate_part_rows_multi(base_datasets)
    if args.selection_json is not None:
        specs = json.loads(args.selection_json.read_text(encoding="utf-8"))
        selected = []
        for spec in specs:
            matches = [
                row
                for row in rows
                if row.obj_id == str(spec["obj_id"])
                and ("dataset_id" not in spec or row.dataset_id == str(spec["dataset_id"]))
                and row.angle_idx == int(spec["angle_idx"])
                and row.part_name == str(spec["part_name"])
            ]
            if len(matches) != 1:
                raise RuntimeError(f"selection spec matched {len(matches)} rows: {spec}")
            selected.append(matches[0])
        return base_ds, selected, selected, selected, selected, {"selection_json": str(args.selection_json), "rows": specs}
    if args.mode == "gate1":
        selected, gate1_meta = pick_gate1_rows(rows)
        if args.single_obj_id is not None:
            if args.single_angle_idx is None or args.single_part_name is None:
                raise ValueError("--single-obj-id requires --single-angle-idx and --single-part-name")
            selected = [
                row
                for row in selected
                if row.obj_id == str(args.single_obj_id)
                and row.angle_idx == int(args.single_angle_idx)
                and row.part_name == str(args.single_part_name)
            ]
            if not selected:
                raise RuntimeError("single-row selection matched zero rows")
            gate1_meta = [{"single_row": selected[0].__dict__}]
        return base_ds, selected, selected, selected, selected, {"gate1_selection": gate1_meta}

    if args.split_json is not None:
        assert split is not None
        train_refs = split.get("train_keys", split["train_ids"])
        heldout_refs = split.get("heldout_keys", split["heldout_ids"])
        train = rows_for_obj_ids(rows, train_refs)
        val = rows_for_obj_ids(rows, heldout_refs)
        train_set = {object_key(row) for row in train}
        heldout_set = {object_key(row) for row in val}
        gate2_required = {str(x) for x in split.get("gate2_heldout_keys", split.get("gate2_heldout_ids", []))}
        if gate2_required and not gate2_required.issubset(heldout_set):
            missing = sorted(gate2_required - heldout_set)
            raise RuntimeError(f"official split is not a superset of Gate2 heldout ids; missing={missing}")
        if train_set & heldout_set:
            raise RuntimeError("official split object overlap detected")
        proxy_train = train
        proxy_val = val
        if args.proxy_json is not None:
            proxy_train, proxy_val, proxy_meta = load_proxy_rows(args.proxy_json, rows)
        else:
            proxy_meta = None
        return base_ds, train, proxy_train, proxy_val, val, {
            "split_json": str(args.split_json),
            "datasets": [
                {
                    "dataset_id": spec.dataset_id,
                    "data_root": str(spec.data_root),
                    "manifest_paths": [str(path) for path in spec.manifest_paths],
                }
                for spec in specs
            ],
            "train_obj_ids": list(map(str, train_refs)),
            "heldout_obj_ids": list(map(str, heldout_refs)),
            "train_obj_count": len(train_set),
            "heldout_obj_count": len(heldout_set),
            "proxy_json": str(args.proxy_json) if args.proxy_json is not None else None,
            "proxy": proxy_meta,
            "coverage": split.get("coverage", {}),
        }

    if int(args.gate2_objects) <= 0:
        obj_ids_all = []
        for row in rows:
            key = object_key(row)
            if key not in obj_ids_all:
                obj_ids_all.append(key)
        total_count = len(obj_ids_all)
        heldout_count = max(1, int(round(total_count * float(args.heldout_fraction))))
    else:
        heldout_count = int(args.gate2_heldout_objects)
        total_count = int(args.gate2_train_objects) + heldout_count
    if int(args.gate2_objects) > 0 and int(args.gate2_objects) != 256:
        total_count = int(args.gate2_objects)
        heldout_count = max(1, int(round(total_count * float(args.heldout_fraction))))
    obj_ids, train_obj_ids, heldout_obj_ids = choose_gate2_object_ids(rows, total=total_count, heldout=heldout_count)
    train_set = set(train_obj_ids)
    heldout_set = set(heldout_obj_ids)
    train = [row for row in rows if object_key(row) in train_set]
    val = [row for row in rows if object_key(row) in heldout_set]
    if train_set & heldout_set:
        raise RuntimeError("gate2 object split overlap detected")
    return base_ds, train, train, val, val, {
        "gate2_obj_ids": obj_ids,
        "train_obj_ids": train_obj_ids,
        "heldout_obj_ids": heldout_obj_ids,
        "train_obj_count": len(train_set),
        "heldout_obj_count": len(heldout_set),
        "heldout_has_multi_button": any(
            sum(1 for row in val if object_key(row) == oid and "button" in row.part_name.lower()) >= 2
            for oid in heldout_set
        ),
        "heldout_has_door_lid": any(("door" in row.part_name.lower()) or ("lid" in row.part_name.lower()) for row in val),
    }


def filter_rows_to_packed_index(
    packed_dir: Path,
    *,
    train_rows: list[Any],
    proxy_train_rows: list[Any],
    proxy_eval_rows: list[Any],
    full_eval_rows: list[Any],
    min_train: int = 1,
    min_eval: int = 1,
) -> tuple[list[Any], list[Any], list[Any], list[Any]]:
    index_path = Path(packed_dir) / "index.json"
    if not index_path.is_file():
        raise FileNotFoundError(f"packed index not found for pack-limit filtering: {index_path}")
    index = json.loads(index_path.read_text(encoding="utf-8"))
    available = {str(entry["key"]) for entry in index.get("entries", [])}

    def keep(rows: list[Any]) -> list[Any]:
        return [row for row in rows if part_row_key(row) in available]

    train = keep(train_rows)
    proxy_train = keep(proxy_train_rows)
    proxy_eval = keep(proxy_eval_rows)
    full_eval = keep(full_eval_rows)
    if len(train) < int(min_train):
        raise RuntimeError(f"pack-limit smoke has too few train rows in packed index: {len(train)}")
    if len(proxy_eval) < int(min_eval) and len(full_eval) < int(min_eval):
        raise RuntimeError(
            f"pack-limit smoke has too few eval rows in packed index: proxy={len(proxy_eval)} full={len(full_eval)}"
        )
    if not proxy_train:
        proxy_train = train[: min(len(train), max(1, int(min_train)))]
    if not proxy_eval:
        proxy_eval = full_eval[: min(len(full_eval), max(1, int(min_eval)))]
    if not full_eval:
        full_eval = proxy_eval
    return train, proxy_train, proxy_eval, full_eval


def save_checkpoint(
    path: Path,
    *,
    model: PromptablePartLatentSegNet,
    optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler | None,
    step: int,
    args: argparse.Namespace,
    empty_code: torch.Tensor,
    metadata: dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "step": int(step),
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict() if scaler is not None else None,
            "args": vars(args),
            "empty_code": empty_code.detach().cpu(),
            "metadata": metadata,
        },
        path,
    )


def load_checkpoint(path: Path, model, optimizer, scaler, device: torch.device) -> tuple[int, torch.Tensor]:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    target = model.module if isinstance(model, DistributedDataParallel) else model
    target.load_state_dict(ckpt["model"], strict=True)
    optimizer.load_state_dict(ckpt["optimizer"])
    if scaler is not None and ckpt.get("scaler") is not None:
        scaler.load_state_dict(ckpt["scaler"])
    return int(ckpt.get("step", 0)), ckpt["empty_code"].float()


def load_warm_start(
    path: Path,
    model,
    *,
    device: torch.device,
    is_rank0: bool,
) -> tuple[torch.Tensor | None, dict[str, Any]]:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    source = ckpt.get("model", {})
    if not isinstance(source, dict):
        raise RuntimeError(f"warm-start checkpoint has no model state dict: {path}")
    target = model.module if isinstance(model, DistributedDataParallel) else model
    target_state = target.state_dict()
    loadable: dict[str, torch.Tensor] = {}
    skipped_shape: list[dict[str, Any]] = []
    unexpected: list[str] = []
    for key, value in source.items():
        if key not in target_state:
            unexpected.append(str(key))
            continue
        if tuple(value.shape) != tuple(target_state[key].shape):
            skipped_shape.append({
                "key": str(key),
                "ckpt_shape": list(value.shape),
                "model_shape": list(target_state[key].shape),
            })
            continue
        loadable[str(key)] = value
    result = target.load_state_dict(loadable, strict=False)
    missing = list(result.missing_keys)
    unexpected_after = list(result.unexpected_keys)
    meta = {
        "path": str(path),
        "ckpt_step": int(ckpt.get("step", 0)),
        "loaded_count": len(loadable),
        "missing_count": len(missing),
        "unexpected_count": len(unexpected) + len(unexpected_after),
        "skipped_shape_count": len(skipped_shape),
        "loaded": sorted(loadable.keys()),
        "missing": missing,
        "unexpected": sorted([*unexpected, *unexpected_after]),
        "skipped_shape": skipped_shape,
    }
    if is_rank0:
        def preview(items: list[Any], limit: int = 24) -> str:
            text = [str(item) for item in items[:limit]]
            if len(items) > limit:
                text.append(f"...(+{len(items) - limit})")
            return ", ".join(text) if text else "(none)"

        print(
            f"[PromptSeg-warm-start] path={path} ckpt_step={meta['ckpt_step']} "
            f"loaded={meta['loaded_count']} missing={meta['missing_count']} "
            f"unexpected={meta['unexpected_count']} skipped_shape={meta['skipped_shape_count']}",
            flush=True,
        )
        print(f"[PromptSeg-warm-start] missing: {preview(missing)}", flush=True)
        print(f"[PromptSeg-warm-start] unexpected: {preview(meta['unexpected'])}", flush=True)
        if skipped_shape:
            print(f"[PromptSeg-warm-start] skipped_shape: {preview(skipped_shape)}", flush=True)
    empty = ckpt.get("empty_code")
    if torch.is_tensor(empty):
        return empty.float(), meta
    return None, meta


def append_code_update(text: str) -> None:
    path = TRELLIS_PATH / "code_update" / "part_promptable_seg.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write("\n\n")
        f.write(text.rstrip())
        f.write("\n")


def summary_table(summary: dict[str, Any]) -> str:
    rows = []
    for bucket in ("tiny", "small", "medium", "large", "button", "all"):
        if bucket not in summary:
            continue
        item = summary[bucket]
        rows.append(
            {
                "bucket": bucket,
                "n": item["n"],
                "cell_iou": f"{item.get('cell_iou', float('nan')):.4f}",
                "support_l1": f"{item.get('support_l1', float('nan')):.5f}",
                "gtm_decode": f"{item.get('head2_gtm_decode_iou', float('nan')):.4f}",
                "e2e_decode": f"{item.get('e2e_decode_iou', float('nan')):.4f}",
            }
        )
    return format_table(rows, ["bucket", "n", "cell_iou", "support_l1", "gtm_decode", "e2e_decode"])


def worst_rows_table(rows: list[dict[str, Any]], *, key: str = "e2e_decode_iou", n: int = 5) -> str:
    ordered = sorted(rows, key=lambda row: float(row.get(key, float("nan"))))
    out = []
    for row in ordered[: int(n)]:
        out.append(
            {
                "id": f"{row.get('obj_id')}/a{row.get('angle_idx')}/{row.get('part_name')}",
                "bucket": row.get("bucket", ""),
                "raw": row.get("raw_count", ""),
                "cell": f"{float(row.get('cell_iou', float('nan'))):.4f}",
                "GTcand": f"{float(row.get('head2_gtm_decode_iou', float('nan'))):.4f}",
                "Predcand": f"{float(row.get('e2e_decode_iou', float('nan'))):.4f}",
            }
        )
    return format_table(out, ["id", "bucket", "raw", "cell", "GTcand", "Predcand"])


def predcand_zero_evidence_table(rows: list[dict[str, Any]], *, n: int = 16) -> str:
    bad = [
        row
        for row in rows
        if float(row.get("e2e_decode_iou", 0.0)) <= 0.0
    ]
    bad.sort(key=lambda row: (
        float(row.get("head2_gtm_decode_iou", 0.0)),
        float(row.get("cell_iou", 0.0)),
    ))
    out = []
    for row in bad[: int(n)]:
        cell_gt = float(row.get("cell_gt_count", 0.0))
        cell_pred = float(row.get("cell_pred_count", 0.0))
        coverage = cell_pred / max(cell_gt, 1.0)
        out.append(
            {
                "id": f"{row.get('obj_id')}/a{row.get('angle_idx')}/{row.get('part_name')}",
                "bucket": row.get("bucket", ""),
                "raw": row.get("raw_count", ""),
                "GTcand": f"{float(row.get('head2_gtm_decode_iou', float('nan'))):.4f}",
                "cell": f"{float(row.get('cell_iou', float('nan'))):.4f}",
                "cell_cov": f"{coverage:.4f}",
                "mask_px": int(row.get("mask_visible_pixels", 0)),
            }
        )
    return format_table(out, ["id", "bucket", "raw", "GTcand", "cell", "cell_cov", "mask_px"])


def combined_eval_table(train_summary: dict[str, Any], heldout_summary: dict[str, Any]) -> str:
    rows = []
    for bucket in ("tiny", "small", "medium", "large", "button", "all"):
        tr = train_summary.get(bucket, {})
        ho = heldout_summary.get(bucket, {})
        rows.append(
            {
                "bucket": bucket,
                "train_n": tr.get("n", 0),
                "train_cell": f"{float(tr.get('cell_iou', float('nan'))):.4f}",
                "train_GTcand": f"{float(tr.get('head2_gtm_decode_iou', float('nan'))):.4f}",
                "train_Predcand": f"{float(tr.get('e2e_decode_iou', float('nan'))):.4f}",
                "train_part": f"{float(tr.get('partition_e2e_decode_iou', float('nan'))):.4f}",
                "train_ov": f"{float(tr.get('part_overlap_voxels', float('nan'))):.1f}",
                "train_part_ov": f"{float(tr.get('partition_object_overlap_voxels', float('nan'))):.1f}",
                "held_n": ho.get("n", 0),
                "held_cell": f"{float(ho.get('cell_iou', float('nan'))):.4f}",
                "held_GTcand": f"{float(ho.get('head2_gtm_decode_iou', float('nan'))):.4f}",
                "held_Predcand": f"{float(ho.get('e2e_decode_iou', float('nan'))):.4f}",
                "held_part": f"{float(ho.get('partition_e2e_decode_iou', float('nan'))):.4f}",
                "held_ov": f"{float(ho.get('part_overlap_voxels', float('nan'))):.1f}",
                "held_part_ov": f"{float(ho.get('partition_object_overlap_voxels', float('nan'))):.1f}",
            }
        )
    return format_table(rows, ["bucket", "train_n", "train_cell", "train_GTcand", "train_Predcand", "train_part", "train_ov", "train_part_ov", "held_n", "held_cell", "held_GTcand", "held_Predcand", "held_part", "held_ov", "held_part_ov"])


def proxy_metric_row(
    *,
    step: int,
    loss: float,
    train_summary: dict[str, Any],
    heldout_summary: dict[str, Any],
    sem_acc: float,
    peak_gb: float,
    s_per_step: float,
    util: float,
) -> dict[str, Any]:
    tr = train_summary.get("all", {})
    ho = heldout_summary.get("all", {})
    return {
        "step": int(step),
        "loss": float(loss),
        "train_cell": float(tr.get("cell_iou", float("nan"))),
        "train_GTcand": float(tr.get("head2_gtm_decode_iou", float("nan"))),
        "train_Predcand": float(tr.get("e2e_decode_iou", float("nan"))),
        "train_partition_Predcand": float(tr.get("partition_e2e_decode_iou", float("nan"))),
        "train_part_overlap": float(tr.get("part_overlap_voxels", float("nan"))),
        "train_partition_overlap": float(tr.get("partition_object_overlap_voxels", float("nan"))),
        "held_cell": float(ho.get("cell_iou", float("nan"))),
        "held_GTcand": float(ho.get("head2_gtm_decode_iou", float("nan"))),
        "held_Predcand": float(ho.get("e2e_decode_iou", float("nan"))),
        "held_partition_Predcand": float(ho.get("partition_e2e_decode_iou", float("nan"))),
        "held_part_overlap": float(ho.get("part_overlap_voxels", float("nan"))),
        "held_partition_overlap": float(ho.get("partition_object_overlap_voxels", float("nan"))),
        "sem_acc": float(sem_acc),
        "peakGB": float(peak_gb),
        "s/step": float(s_per_step),
        "util": float(util),
        **no_prompt_tracker_snapshot(),
    }


EVAL_VALUE_KEYS = (
    "cell_iou",
    "support_l1",
    "latent_l1_signal_norm",
    "head2_gtm_decode_iou",
    "head2_gtm_precision",
    "head2_gtm_recall",
    "e2e_decode_iou",
    "e2e_precision",
    "e2e_recall",
    "partition_e2e_decode_iou",
    "partition_e2e_precision",
    "partition_e2e_recall",
    "part_overlap_voxels",
    "object_overlap_voxels",
    "partition_object_overlap_voxels",
)


def dataset_eval_summary(rows: list[dict[str, Any]], dataset_id: str) -> dict[str, Any]:
    selected = [row for row in rows if str(row.get("dataset_id", "")) == str(dataset_id)]
    summary = summarize_by_bucket(selected, EVAL_VALUE_KEYS)
    summary["all"] = summarize_rows(selected, EVAL_VALUE_KEYS)
    return summary


def overfit_small_warnings(
    heldout_summary: dict[str, Any],
    best_by_bucket: dict[str, float],
    *,
    threshold: float = 0.03,
) -> list[str]:
    warnings = []
    for bucket in ("medium", "large"):
        value = float(heldout_summary.get(bucket, {}).get("e2e_decode_iou", float("nan")))
        if not math.isfinite(value):
            continue
        best = best_by_bucket.get(bucket)
        if best is not None and math.isfinite(best) and value < best - float(threshold):
            warnings.append(
                f"[WARN overfit-small?] bucket={bucket} held_Predcand={value:.4f} "
                f"best={best:.4f} drop={best - value:.4f}; consider lowering --small-oversample or --focal-gamma"
            )
        if best is None or value > best:
            best_by_bucket[bucket] = value
    return warnings


@torch.no_grad()
def run_motion_gt_sanity(
    model: PromptablePartLatentSegNet,
    decoder,
    loader: DataLoader,
    *,
    device: torch.device,
    use_packed_whole_occ: bool,
    loss_kind: str = "bce_dice",
    max_batches: int = 8,
) -> dict[str, float]:
    model.eval()
    bce_items: list[float] = []
    dice_items: list[float] = []
    loss_items: list[float] = []
    motion_items = 0
    batches = 0
    for batch in loader:
        batches += 1
        z_global = batch["z_global"].to(device=device, dtype=torch.float32)
        masks2d = batch["masks2d"].to(device=device, dtype=torch.float32)
        m_gt = batch["m_gt"].to(device=device, dtype=torch.float32)
        out_v = model(
            z_global,
            masks2d,
            candidate_cells=mask_morphology(m_gt, "dilate"),
            full_occ=full_occ_for_batch(
                decoder,
                z_global,
                batch,
                device=device,
                use_packed_whole_occ=bool(use_packed_whole_occ),
            ),
        )
        l_motion, items = motion_consistency_loss_from_logits(
            out_v["voxel_logits"],
            out_v["voxel_coords"],
            batch,
            device=device,
            use_gt_membership=True,
            loss_kind=str(loss_kind),
        )
        if float(items.get("motion_items", 0.0)) > 0:
            motion_items += int(items["motion_items"])
            bce_items.append(float(items["motion_bce"]))
            dice_items.append(float(items["motion_dice"]))
            loss_items.append(float(l_motion.detach().item()))
        if batches >= int(max_batches) and motion_items > 0:
            break
    return {
        "batches": float(batches),
        "items": float(motion_items),
        "motion_loss": float(np.mean(loss_items)) if loss_items else 0.0,
        "motion_bce": float(np.mean(bce_items)) if bce_items else 0.0,
        "motion_dice": float(np.mean(dice_items)) if dice_items else 0.0,
    }


def proxy_metric_table(row: dict[str, Any]) -> str:
    headers = [
        "step",
        "loss",
        "train_cell",
        "train_GTcand",
        "train_Predcand",
        "held_cell",
        "held_GTcand",
        "held_Predcand",
        "ra_held_n",
        "ra_held_cell",
        "ra_held_GTcand",
        "ra_held_Predcand",
        "sem_acc",
        "peakGB",
        "s/step",
        "util",
        "no_prompt_count",
        "samples_seen",
        "no_prompt_ratio",
    ]
    formatted = dict(row)
    for key, value in list(formatted.items()):
        if isinstance(value, float):
            formatted[key] = f"{value:.4f}"
    return format_table([formatted], headers)


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def append_tsv(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    headers = list(row.keys())
    write_header = not path.exists()
    with path.open("a", encoding="utf-8") as f:
        if write_header:
            f.write("\t".join(headers) + "\n")
        f.write("\t".join(str(row[key]) for key in headers) + "\n")


def part_row_id(row: Any) -> str:
    return part_row_key(row)


def write_mask_visibility_tsv(path: Path, records: list[dict[str, Any]]) -> None:
    fields = [
        "classification",
        "dataset_id",
        "obj_id",
        "angle_idx",
        "sample_id",
        "part_name",
        "original_label",
        "raw_count",
        "selected_view_indices",
        "selected_visible_pixels",
        "selected_nonempty_views",
        "all_visible_pixels",
        "all_nonempty_views",
        "missing_mask_views",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write("\t".join(fields) + "\n")
        for rec in records:
            values = []
            for field in fields:
                value = rec.get(field, "")
                if isinstance(value, (list, dict)):
                    value = json.dumps(value, ensure_ascii=False, sort_keys=True)
                values.append(str(value))
            f.write("\t".join(values) + "\n")


def mask_visibility_split_summary(rows: list[Any], records_by_key: dict[str, dict[str, Any]]) -> dict[str, Any]:
    class_counts: dict[str, int] = {}
    selected_all_empty = 0
    for row in rows:
        rec = records_by_key.get(part_row_id(row))
        if rec is None:
            continue
        cls = str(rec["classification"])
        class_counts[cls] = class_counts.get(cls, 0) + 1
        if int(rec.get("selected_visible_pixels", 0)) <= 0:
            selected_all_empty += 1
    total = len(rows)
    return {
        "total_rows": total,
        "selected_all_empty_rows": selected_all_empty,
        "selected_all_empty_ratio": float(selected_all_empty / max(1, total)),
        "class_counts": dict(sorted(class_counts.items())),
        "class_ratios": {
            name: float(count / max(1, total))
            for name, count in sorted(class_counts.items())
        },
    }


def _dist_broadcast_object(obj: Any, *, src: int = 0) -> Any:
    payload = [obj]
    dist.broadcast_object_list(payload, src=src)
    return payload[0]


def audit_and_filter_mask_visibility(
    base_ds,
    *,
    train_rows: list[Any],
    proxy_train_rows: list[Any],
    proxy_eval_rows: list[Any],
    full_eval_rows: list[Any],
    args: argparse.Namespace,
    distributed: bool,
    is_rank0: bool,
) -> tuple[list[Any], list[Any], list[Any], list[Any], dict[str, Any]]:
    all_rows_by_key: dict[str, Any] = {}
    for row in [*train_rows, *proxy_train_rows, *proxy_eval_rows, *full_eval_rows]:
        all_rows_by_key.setdefault(part_row_id(row), row)
    all_rows = list(all_rows_by_key.values())
    if is_rank0:
        audit = audit_promptable_mask_visibility(
            base_ds,
            all_rows,
            expected_views=int(args.mask_audit_views),
        )
        records = list(audit["records"])
        records_by_key = {str(rec["key"]): rec for rec in records}
        undetectable = [rec for rec in records if rec["classification"] == "undetectable_selected_views"]
        absent = [rec for rec in records if rec["classification"] == "label_absent_all_views"]
        split_summaries = {
            "train": mask_visibility_split_summary(train_rows, records_by_key),
            "proxy_train": mask_visibility_split_summary(proxy_train_rows, records_by_key),
            "proxy_eval": mask_visibility_split_summary(proxy_eval_rows, records_by_key),
            "full_eval": mask_visibility_split_summary(full_eval_rows, records_by_key),
        }
        report_dir = args.out_dir / "mask_audit"
        report_dir.mkdir(parents=True, exist_ok=True)
        summary = {key: value for key, value in audit.items() if key != "records"}
        summary.update({
            "filter_undetectable": bool(args.filter_undetectable),
            "fail_label_absent_ratio": float(args.fail_label_absent_ratio),
            "undetectable_rows": len(undetectable),
            "label_absent_rows": len(absent),
            "splits": split_summaries,
        })
        (report_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        (report_dir / "records.json").write_text(json.dumps(records, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        (report_dir / "undetectable.json").write_text(json.dumps(undetectable, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        (report_dir / "label_absent_all_views.json").write_text(json.dumps(absent, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        write_mask_visibility_tsv(report_dir / "records.tsv", records)
        write_mask_visibility_tsv(report_dir / "undetectable.tsv", undetectable)
        write_mask_visibility_tsv(report_dir / "label_absent_all_views.tsv", absent)
        label_absent_ratio = float(len(absent) / max(1, len(records)))
        undetectable_ratio = float(len(undetectable) / max(1, len(records)))
        print(
            f"[PromptSeg-mask-audit] total={len(records)} visible={audit['class_counts'].get('visible_selected_views', 0)} "
            f"undetectable={len(undetectable)} ({undetectable_ratio:.4%}) "
            f"label_absent={len(absent)} ({label_absent_ratio:.4%}) report={report_dir}",
            flush=True,
        )
        train_empty = split_summaries["train"]["selected_all_empty_rows"]
        train_total = split_summaries["train"]["total_rows"]
        print(
            f"[PromptSeg-mask-audit] train_selected_all_empty={train_empty}/{train_total} "
            f"({train_empty / max(1, train_total):.4%})",
            flush=True,
        )
        if absent:
            preview = ", ".join(
                f"{rec['obj_id']}/angle_{rec['angle_idx']}/{rec['part_name']} label={rec['original_label']}"
                for rec in absent[:8]
            )
            print(f"[PromptSeg-mask-audit] label_absent preview: {preview}", flush=True)
        error = None
        if label_absent_ratio > float(args.fail_label_absent_ratio):
            error = (
                f"label_absent_all_views ratio {label_absent_ratio:.4%} exceeds "
                f"--fail-label-absent-ratio={float(args.fail_label_absent_ratio):.4%}; "
                f"inspect {report_dir / 'label_absent_all_views.json'} before training"
            )
        filter_keys = {
            str(rec["key"])
            for rec in undetectable
            if bool(args.filter_undetectable)
        }
        audit_meta = {
            **summary,
            "report_dir": str(report_dir),
            "filtered_undetectable_rows": len(filter_keys),
        }
        payload = {"filter_keys": sorted(filter_keys), "audit_meta": audit_meta, "error": error}
    else:
        payload = None
    if distributed:
        payload = _dist_broadcast_object(payload, src=0)
    if payload is None:
        payload = {"filter_keys": [], "audit_meta": {}, "error": None}
    if payload.get("error"):
        raise RuntimeError(str(payload["error"]))
    filter_keys = set(payload["filter_keys"])

    def keep(rows: list[Any]) -> list[Any]:
        return [row for row in rows if part_row_id(row) not in filter_keys]

    filtered = (
        keep(train_rows),
        keep(proxy_train_rows),
        keep(proxy_eval_rows),
        keep(full_eval_rows),
    )
    if is_rank0 and filter_keys:
        print(
            f"[PromptSeg-mask-audit] filtered undetectable rows: train {len(train_rows)}->{len(filtered[0])} "
            f"proxy_train {len(proxy_train_rows)}->{len(filtered[1])} proxy_eval {len(proxy_eval_rows)}->{len(filtered[2])} "
            f"full_eval {len(full_eval_rows)}->{len(filtered[3])}",
            flush=True,
        )
    return (*filtered, payload["audit_meta"])


def warn_all_empty_prompt_masks(batch: dict[str, Any], masks2d: torch.Tensor, *, context: str) -> int:
    per_sample = masks2d.detach().flatten(2).sum(dim=2).sum(dim=1)
    empty = torch.nonzero(per_sample <= 0, as_tuple=False).flatten().detach().cpu().tolist()
    dataset_ids = batch.get("dataset_id", [""] * len(batch.get("obj_id", [])))
    for idx in empty:
        print(
            f"[PromptSeg-NO_PROMPT] context={context} obj_id={batch['obj_id'][idx]} "
            f"dataset_id={dataset_ids[idx]} "
            f"angle_idx={int(batch['angle_idx'][idx])} sample_id={batch['sample_id'][idx]} "
            f"part_name={batch['part_name'][idx]} original_label={batch['original_label'][idx]} "
            f"view_indices={batch['view_indices'][idx].detach().cpu().tolist()}",
            flush=True,
        )
    return len(empty)


def configure_no_prompt_tracker(out_dir: Path, *, rank: int) -> None:
    global NO_PROMPT_TRACKER
    path = Path(out_dir) / "logs" / f"no_prompt_rank{int(rank)}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    NO_PROMPT_TRACKER = {
        "rank": int(rank),
        "path": path,
        "samples_seen": 0,
        "no_prompt_count": 0,
    }


def record_no_prompt_from_output(
    out: dict[str, Any],
    batch: dict[str, Any],
    *,
    context: str,
    step: int | None = None,
) -> int:
    tracker = NO_PROMPT_TRACKER
    mask = out.get("no_prompt_mask")
    bsz = len(batch.get("obj_id", []))
    if tracker is not None:
        tracker["samples_seen"] = int(tracker.get("samples_seen", 0)) + int(bsz)
    if mask is None:
        return 0
    if not isinstance(mask, torch.Tensor):
        mask = torch.as_tensor(mask)
    hit_indices = torch.nonzero(mask.detach().bool().cpu(), as_tuple=False).flatten().tolist()
    if tracker is not None:
        tracker["no_prompt_count"] = int(tracker.get("no_prompt_count", 0)) + len(hit_indices)
    if not hit_indices:
        return 0
    path = tracker["path"] if tracker is not None else None
    rows = []
    for idx in hit_indices:
        view_indices = batch["view_indices"][idx]
        if isinstance(view_indices, torch.Tensor):
            view_indices = view_indices.detach().cpu().tolist()
        rows.append({
            "rank": int(tracker.get("rank", -1)) if tracker is not None else -1,
            "step": int(step) if step is not None else None,
            "context": str(context),
            "dataset_id": batch.get("dataset_id", [""] * bsz)[idx],
            "obj_id": batch["obj_id"][idx],
            "angle_idx": int(batch["angle_idx"][idx]),
            "sample_id": batch["sample_id"][idx],
            "part_name": batch["part_name"][idx],
            "original_label": int(batch["original_label"][idx]),
            "view_indices": [int(v) for v in view_indices],
        })
    if path is not None:
        with Path(path).open("a", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
    for row in rows:
        print(
            f"[PromptSeg-NO_PROMPT] rank={row['rank']} context={row['context']} step={row['step']} "
            f"dataset_id={row.get('dataset_id', '')} obj_id={row['obj_id']} angle_idx={row['angle_idx']} sample_id={row['sample_id']} "
            f"part_name={row['part_name']} original_label={row['original_label']} view_indices={row['view_indices']}",
            flush=True,
        )
    return len(rows)


def no_prompt_tracker_snapshot() -> dict[str, float]:
    tracker = NO_PROMPT_TRACKER or {}
    samples_seen = int(tracker.get("samples_seen", 0))
    count = int(tracker.get("no_prompt_count", 0))
    return {
        "no_prompt_count": float(count),
        "samples_seen": float(samples_seen),
        "no_prompt_ratio": float(count / max(1, samples_seen)),
    }


def current_gpu_util() -> float:
    if not torch.cuda.is_available():
        return 0.0
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
            text=True,
            timeout=5,
        )
    except Exception:
        return float("nan")
    vals = []
    for line in out.splitlines():
        line = line.strip()
        if not line:
            continue
        match = re.search(r"\d+", line)
        if match:
            vals.append(float(match.group(0)))
    return float(np.mean(vals)) if vals else float("nan")


def final_decision(train_summary: dict[str, Any], heldout_summary: dict[str, Any], history: list[dict[str, Any]]) -> dict[str, Any]:
    train_pred = float(train_summary.get("all", {}).get("e2e_decode_iou", float("nan")))
    held_pred = float(heldout_summary.get("all", {}).get("e2e_decode_iou", float("nan")))
    if train_pred >= 0.9 and held_pred < train_pred - 0.05:
        verdict = "capacity_ok_gap_is_data_or_regularization"
        action = "keep S as draft final; improve data/regularization before changing capacity"
    elif train_pred < 0.85:
        verdict = "underfit_signature"
        action = "queue M run with the same image and only MODEL_SIZE changed"
    else:
        recent = history[-3:]
        if len(recent) >= 3 and recent[-1]["held_Predcand"] > recent[0]["held_Predcand"] + 0.005:
            verdict = "still_improving"
            action = "extend steps before final capacity call"
        else:
            verdict = "plateau_or_borderline"
            action = "review bucket/worst-5 before changing capacity"
    return {
        "train_Predcand": train_pred,
        "held_Predcand": held_pred,
        "verdict": verdict,
        "action": action,
    }


def latest_checkpoint_path(out_dir: Path) -> Path | None:
    ckpt_dir = Path(out_dir) / "ckpts"
    latest = ckpt_dir / "latest.pt"
    if latest.is_file():
        return latest
    if not ckpt_dir.is_dir():
        return None
    candidates = []
    for path in ckpt_dir.glob("step_*.pt"):
        match = re.search(r"step_(\d+)\.pt$", path.name)
        if match:
            candidates.append((int(match.group(1)), path))
    if not candidates:
        return None
    return sorted(candidates)[-1][1]


@torch.no_grad()
def measure_peak_memory(
    model: PromptablePartLatentSegNet,
    decoder,
    dataset: PromptablePartDataset,
    empty_code: torch.Tensor,
    *,
    device: torch.device,
    mask_target: str,
    support_multiplier: float,
    route: str,
    voxel_max_tokens: int,
    use_packed_whole_occ: bool = False,
) -> float:
    if not torch.cuda.is_available() or not str(device).startswith("cuda"):
        return 0.0
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    batch = collate_promptable_parts([dataset[0]])
    _ = evaluate(
        model,
        decoder,
        DataLoader([dataset[0]], batch_size=1, shuffle=False, collate_fn=collate_promptable_parts),
        empty_code,
        device=device,
        max_rows=1,
        mask_only=False,
        mask_target=mask_target,
        support_multiplier=support_multiplier,
        route=route,
        voxel_max_tokens=voxel_max_tokens,
        use_packed_whole_occ=bool(use_packed_whole_occ),
    )
    peak_gb = float(torch.cuda.max_memory_allocated(device) / (1024 ** 3))
    del batch
    torch.cuda.empty_cache()
    return peak_gb


def memorized_eval(result: dict[str, Any], threshold: float) -> tuple[bool, dict[str, float]]:
    rows = result.get("rows", [])
    if not rows:
        return False, {}
    thr = float(threshold)
    mins = {
        "min_cell_iou": min(float(row.get("cell_iou", 0.0)) for row in rows),
        "min_gtm_decode": min(float(row.get("head2_gtm_decode_iou", 0.0)) for row in rows),
    }
    e2e_min = min(float(row.get("e2e_decode_iou", 0.0)) for row in rows)
    ok = all(value >= thr for value in mins.values())
    mins["min_e2e_decode_observed"] = e2e_min
    return ok, mins


def freeze_unused_route_parameters(model: PromptablePartLatentSegNet, *, route: str, is_rank0: bool) -> dict[str, int]:
    frozen = 0
    frozen_params = 0
    if str(route) == "voxel":
        latent_only_modules = [
            ("m_emb", model.m_emb),
            ("head2_in", model.head2_in),
            ("head2_blocks", model.head2_blocks),
            ("head2_norm", model.head2_norm),
            ("delta", model.delta),
        ]
        for _name, module in latent_only_modules:
            for param in module.parameters():
                if param.requires_grad:
                    frozen += 1
                    frozen_params += int(param.numel())
                    param.requires_grad_(False)
    trainable = int(sum(p.numel() for p in model.parameters() if p.requires_grad))
    frozen_total = int(sum(p.numel() for p in model.parameters() if not p.requires_grad))
    if is_rank0:
        print(
            f"[PromptSeg-freeze] route={route} route_unused_tensors={frozen} "
            f"route_unused_params={frozen_params:,} trainable_model_params={trainable:,} "
            f"frozen_model_params={frozen_total:,}",
            flush=True,
        )
    return {
        "route_unused_tensors": int(frozen),
        "route_unused_params": int(frozen_params),
        "trainable_model_params": trainable,
        "frozen_model_params": frozen_total,
    }


def main() -> int:
    args = parse_args()
    if args.resume is not None and args.warm_start is not None:
        raise ValueError("--resume and --warm-start are mutually exclusive")
    os.environ["PROMPTSEG_DDP_TIMEOUT_S"] = str(max(3600, int(args.pack_barrier_timeout_s)))
    if args.packed_dir is None and bool(args.auto_pack):
        args.packed_dir = DEFAULT_PACKED_V4
    distributed, rank, world_size, local_rank = init_distributed_if_needed()
    is_rank0 = rank == 0
    if args.out_dir is None:
        args.out_dir = DEFAULT_GATE1_OUT if args.mode == "gate1" else DEFAULT_GATE2_OUT
    if is_rank0:
        args.out_dir.mkdir(parents=True, exist_ok=True)
    configure_no_prompt_tracker(args.out_dir, rank=rank)
    if str(args.route) == "voxel" and int(args.voxel_max_tokens) > 0:
        raise ValueError("Route-V voxel cap is retired; pass --voxel-max-tokens 0")
    if args.packed_dir is not None and bool(args.auto_pack):
        fp = source_fingerprint(
            args.split_json or OFFICIAL_SPLIT_PATH,
            base_packed_dir=args.base_packed_dir,
            pack_limit=int(args.pack_limit),
        )
        ok, reason, marker = pack_completion_status(args.packed_dir, expected_fingerprint=fp)
        if is_rank0:
            if ok:
                print(f"[PromptSeg-auto-pack] packed complete: {args.packed_dir} rows={marker.get('rows') if marker else 'unknown'}", flush=True)
            else:
                print(f"[PromptSeg-auto-pack] packed incomplete: {args.packed_dir} reason={reason}; rank0 will build", flush=True)
                ensure_packed_dataset(
                    split_json=args.split_json or OFFICIAL_SPLIT_PATH,
                    out_dir=args.packed_dir,
                    shard_size=int(args.pack_shard_size),
                    include_heldout=True,
                    overwrite_incomplete=True,
                    base_packed_dir=args.base_packed_dir,
                    progress_every=int(args.pack_progress_every),
                    mask_audit_views=0,
                    limit=int(args.pack_limit),
                )
        if distributed:
            dist.barrier()
        ok, reason, _marker = pack_completion_status(args.packed_dir, expected_fingerprint=fp)
        if not ok:
            raise RuntimeError(f"packed dataset is still incomplete after auto-pack: {args.packed_dir}: {reason}")
    seed_all(int(args.seed) + rank)
    device = torch.device(f"cuda:{local_rank}" if distributed else (args.device if torch.cuda.is_available() or not args.device.startswith("cuda") else "cpu"))

    base_ds, train_rows, proxy_train_rows, proxy_eval_rows, full_eval_rows, split_meta = select_rows(args)
    if args.packed_dir is not None and int(args.pack_limit) > 0:
        before_counts = (len(train_rows), len(proxy_train_rows), len(proxy_eval_rows), len(full_eval_rows))
        train_rows, proxy_train_rows, proxy_eval_rows, full_eval_rows = filter_rows_to_packed_index(
            args.packed_dir,
            train_rows=train_rows,
            proxy_train_rows=proxy_train_rows,
            proxy_eval_rows=proxy_eval_rows,
            full_eval_rows=full_eval_rows,
            min_train=max(1, int(args.batch_size)),
            min_eval=max(1, int(args.eval_batch_size) if int(args.eval_batch_size) > 0 else 1),
        )
        if is_rank0:
            print(
                "[PromptSeg-pack-limit] filtered rows "
                f"train {before_counts[0]}->{len(train_rows)} "
                f"proxy_train {before_counts[1]}->{len(proxy_train_rows)} "
                f"proxy_eval {before_counts[2]}->{len(proxy_eval_rows)} "
                f"full_eval {before_counts[3]}->{len(full_eval_rows)}",
                flush=True,
            )
    mask_audit_meta: dict[str, Any] = {}
    if int(args.mask_audit_views) > 0:
        train_rows, proxy_train_rows, proxy_eval_rows, full_eval_rows, mask_audit_meta = audit_and_filter_mask_visibility(
            base_ds,
            train_rows=train_rows,
            proxy_train_rows=proxy_train_rows,
            proxy_eval_rows=proxy_eval_rows,
            full_eval_rows=full_eval_rows,
            args=args,
            distributed=distributed,
            is_rank0=is_rank0,
        )
        if not train_rows:
            raise RuntimeError("mask visibility filtering removed all train rows")
        if not proxy_train_rows:
            raise RuntimeError("mask visibility filtering removed all proxy train rows")
        if not proxy_eval_rows:
            raise RuntimeError("mask visibility filtering removed all proxy eval rows")
        if not full_eval_rows:
            raise RuntimeError("mask visibility filtering removed all full eval rows")
    motion_sidecar: dict[str, dict[str, Any]] = {}
    motion_sidecar_meta: dict[str, Any] = {}
    if bool(args.motion_sanity_only) or float(args.motion_loss_weight) > 0:
        motion_cache_path = args.out_dir / "motion_sidecar_cache.pt"
        if is_rank0:
            motion_rows = [*train_rows, *proxy_train_rows, *proxy_eval_rows, *full_eval_rows]
            motion_sidecar, motion_sidecar_meta = build_motion_sidecar(
                motion_rows,
                max_angle_delta=int(args.motion_max_angle_delta),
            )
            torch.save({"sidecar": motion_sidecar, "meta": motion_sidecar_meta}, motion_cache_path)
        if distributed:
            dist.barrier()
        if not is_rank0:
            payload = torch.load(motion_cache_path, map_location="cpu")
            motion_sidecar = dict(payload.get("sidecar", {}))
            motion_sidecar_meta = dict(payload.get("meta", {}))
        if is_rank0:
            print(
                "[PromptSeg-motion-sidecar] "
                f"rows={motion_sidecar_meta.get('rows', 0)} enabled={motion_sidecar_meta.get('enabled_rows', 0)} "
                f"ratio={float(motion_sidecar_meta.get('enabled_ratio', 0.0)):.4f} "
                f"skipped={motion_sidecar_meta.get('skipped', {})}",
                flush=True,
            )
    semantic_vocab = build_semantic_vocab([*train_rows, *full_eval_rows]) if bool(args.semantic_aux) else {}
    if args.packed_dir is not None:
        train_ds = PackedPromptablePartDataset(args.packed_dir, train_rows, semantic_vocab=semantic_vocab)
        proxy_train_ds = PackedPromptablePartDataset(args.packed_dir, proxy_train_rows, semantic_vocab=semantic_vocab)
        eval_ds = PackedPromptablePartDataset(args.packed_dir, proxy_eval_rows, semantic_vocab=semantic_vocab)
        full_eval_ds = PackedPromptablePartDataset(args.packed_dir, full_eval_rows, semantic_vocab=semantic_vocab)
    else:
        include_whole = bool(args.use_packed_whole_occ)
        train_ds = PromptablePartDataset(base_ds, train_rows, mask_size=512, semantic_vocab=semantic_vocab, include_whole_coords=include_whole)
        proxy_train_ds = PromptablePartDataset(base_ds, proxy_train_rows, mask_size=512, semantic_vocab=semantic_vocab, include_whole_coords=include_whole)
        eval_ds = PromptablePartDataset(base_ds, proxy_eval_rows, mask_size=512, semantic_vocab=semantic_vocab, include_whole_coords=include_whole)
        full_eval_ds = PromptablePartDataset(base_ds, full_eval_rows, mask_size=512, semantic_vocab=semantic_vocab, include_whole_coords=include_whole)
    if motion_sidecar:
        train_ds = MotionSidecarDataset(train_ds, train_rows, motion_sidecar)
        proxy_train_ds = MotionSidecarDataset(proxy_train_ds, proxy_train_rows, motion_sidecar)
        eval_ds = MotionSidecarDataset(eval_ds, proxy_eval_rows, motion_sidecar)
        full_eval_ds = MotionSidecarDataset(full_eval_ds, full_eval_rows, motion_sidecar)
    oversampling_plan = build_oversampling_plan(
        train_rows,
        small_oversample=int(args.small_oversample),
        realappliance_oversample=int(args.realappliance_oversample),
        realappliance_target_share=float(args.realappliance_target_share),
        realappliance_max_oversample=int(args.realappliance_max_oversample),
        verse_focus_oversample=int(args.verse_focus_oversample),
    )
    repeat_by_key = dict(oversampling_plan.get("repeat_by_key", {}))
    eval_batch_size = int(args.eval_batch_size) if int(args.eval_batch_size) > 0 else int(args.batch_size)
    group_batches = (
        bool(args.object_group_batches)
        if args.object_group_batches is not None
        else (float(args.embed_loss_weight) > 0 or float(args.xpart_ce_weight) > 0)
    )
    eval_group_batches = str(args.route) == "voxel"
    if args.packed_dir is not None:
        if group_batches:
            train_sampler = PackedObjectGroupBatchSampler(
                train_ds,
                batch_size=int(args.batch_size),
                shuffle=True,
                seed=int(args.seed),
                num_replicas=world_size if distributed else 1,
                rank=rank if distributed else 0,
                repeat_by_key=repeat_by_key,
            )
        else:
            train_sampler = PackedShardBatchSampler(
                train_ds,
                batch_size=int(args.batch_size),
                shuffle=True,
                seed=int(args.seed),
                num_replicas=world_size if distributed else 1,
                rank=rank if distributed else 0,
                small_oversample=int(args.small_oversample),
                repeat_by_key=repeat_by_key,
            )
        train_loader = DataLoader(
            train_ds,
            batch_sampler=train_sampler,
            num_workers=int(args.num_workers),
            collate_fn=collate_promptable_parts,
            pin_memory=torch.cuda.is_available(),
        )
        eval_sampler = (
            ObjectGroupBatchSampler(proxy_eval_rows, batch_size=eval_batch_size, shuffle=False, seed=int(args.seed))
            if eval_group_batches
            else PackedShardBatchSampler(eval_ds, batch_size=eval_batch_size, shuffle=False, small_oversample=1)
        )
        train_eval_sampler = (
            ObjectGroupBatchSampler(proxy_train_rows, batch_size=eval_batch_size, shuffle=False, seed=int(args.seed))
            if eval_group_batches
            else PackedShardBatchSampler(proxy_train_ds, batch_size=eval_batch_size, shuffle=False, small_oversample=1)
        )
        full_eval_sampler = (
            ObjectGroupBatchSampler(full_eval_rows, batch_size=eval_batch_size, shuffle=False, seed=int(args.seed))
            if eval_group_batches
            else PackedShardBatchSampler(full_eval_ds, batch_size=eval_batch_size, shuffle=False, small_oversample=1)
        )
        eval_loader = DataLoader(
            eval_ds,
            batch_sampler=eval_sampler,
            num_workers=0,
            collate_fn=collate_promptable_parts,
            pin_memory=torch.cuda.is_available(),
        )
        train_eval_loader = DataLoader(
            proxy_train_ds,
            batch_sampler=train_eval_sampler,
            num_workers=0,
            collate_fn=collate_promptable_parts,
            pin_memory=torch.cuda.is_available(),
        )
        full_eval_loader = DataLoader(
            full_eval_ds,
            batch_sampler=full_eval_sampler,
            num_workers=0,
            collate_fn=collate_promptable_parts,
            pin_memory=torch.cuda.is_available(),
        )
    else:
        if group_batches:
            train_sampler = ObjectGroupBatchSampler(
                train_rows,
                batch_size=int(args.batch_size),
                shuffle=True,
                seed=int(args.seed),
                num_replicas=world_size if distributed else 1,
                rank=rank if distributed else 0,
                repeat_by_key=repeat_by_key,
            )
            train_loader = DataLoader(
                train_ds,
                batch_sampler=train_sampler,
                num_workers=int(args.num_workers),
                collate_fn=collate_promptable_parts,
                pin_memory=torch.cuda.is_available(),
            )
        else:
            train_sampler = SmallOversampleSampler(
                train_rows,
                small_oversample=int(args.small_oversample),
                repeat_by_key=repeat_by_key,
                shuffle=True,
                seed=int(args.seed),
                num_replicas=world_size if distributed else 1,
                rank=rank if distributed else 0,
            )
            train_loader = DataLoader(
                train_ds,
                batch_size=int(args.batch_size),
                shuffle=False,
                sampler=train_sampler,
                num_workers=int(args.num_workers),
                collate_fn=collate_promptable_parts,
                pin_memory=torch.cuda.is_available(),
                drop_last=False,
            )
        if eval_group_batches:
            eval_loader = DataLoader(
                eval_ds,
                batch_sampler=ObjectGroupBatchSampler(proxy_eval_rows, batch_size=eval_batch_size, shuffle=False, seed=int(args.seed)),
                num_workers=0,
                collate_fn=collate_promptable_parts,
                pin_memory=torch.cuda.is_available(),
            )
            train_eval_loader = DataLoader(
                proxy_train_ds,
                batch_sampler=ObjectGroupBatchSampler(proxy_train_rows, batch_size=eval_batch_size, shuffle=False, seed=int(args.seed)),
                num_workers=0,
                collate_fn=collate_promptable_parts,
                pin_memory=torch.cuda.is_available(),
            )
            full_eval_loader = DataLoader(
                full_eval_ds,
                batch_sampler=ObjectGroupBatchSampler(full_eval_rows, batch_size=eval_batch_size, shuffle=False, seed=int(args.seed)),
                num_workers=0,
                collate_fn=collate_promptable_parts,
                pin_memory=torch.cuda.is_available(),
            )
        else:
            eval_loader = DataLoader(
                eval_ds,
                batch_size=eval_batch_size,
                shuffle=False,
                num_workers=0,
                collate_fn=collate_promptable_parts,
                pin_memory=torch.cuda.is_available(),
                drop_last=False,
            )
            train_eval_loader = DataLoader(
                proxy_train_ds,
                batch_size=eval_batch_size,
                shuffle=False,
                num_workers=0,
                collate_fn=collate_promptable_parts,
                pin_memory=torch.cuda.is_available(),
                drop_last=False,
            )
            full_eval_loader = DataLoader(
                full_eval_ds,
                batch_size=eval_batch_size,
                shuffle=False,
                num_workers=0,
                collate_fn=collate_promptable_parts,
                pin_memory=torch.cuda.is_available(),
                drop_last=False,
            )

    model = PromptablePartLatentSegNet(
        dim=int(args.dim),
        depth=int(args.depth),
        head_depth=int(args.head_depth),
        heads=int(args.heads),
        use_voxel_head=str(args.route) == "voxel",
        voxel_depth=int(args.voxel_depth),
        refine_mode=str(args.refine_mode),
        spconv_depth=int(args.spconv_depth),
        mask_encoder=str(args.mask_encoder),
        point_k_boundary=int(args.point_k_boundary),
        point_k_interior=int(args.point_k_interior),
        point_resample_points=bool(args.point_resample_points),
        semantic_classes=len(semantic_vocab) if bool(args.semantic_aux) else 0,
        voxel_embedding_dim=int(args.voxel_embedding_dim) if str(args.route) == "voxel" else 0,
    ).to(device)
    route_freeze_meta = freeze_unused_route_parameters(model, route=str(args.route), is_rank0=is_rank0)
    if bool(args.compile):
        model = torch.compile(model)
    encoder = load_ss_encoder(device=device, fp32=True)
    decoder = load_ss_decoder(device=device, fp32=True)
    frozen_encoder = sum(p.numel() for p in encoder.parameters() if not p.requires_grad)
    frozen_decoder = sum(p.numel() for p in decoder.parameters() if not p.requires_grad)
    if any(p.requires_grad for p in encoder.parameters()) or any(p.requires_grad for p in decoder.parameters()):
        raise RuntimeError("SS encoder/decoder must be frozen before promptable segmentation training")
    empty_code = compute_empty_code(encoder, device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))
    scaler = torch.cuda.amp.GradScaler(enabled=bool(args.fp16) and torch.cuda.is_available()) if torch.cuda.is_available() else None

    warm_start_meta: dict[str, Any] | None = None
    if args.warm_start is not None:
        loaded_empty, warm_start_meta = load_warm_start(args.warm_start, model, device=device, is_rank0=is_rank0)
        if loaded_empty is not None:
            empty_code = loaded_empty

    metadata = {
        "mode": args.mode,
        "train_rows": len(train_rows),
        "proxy_train_rows": len(proxy_train_rows),
        "proxy_eval_rows": len(proxy_eval_rows),
        "full_eval_rows": len(full_eval_rows),
        "split": split_meta,
        "mask_audit": mask_audit_meta,
        "params": approx_param_count(model),
        "trainable_params": int(sum(p.numel() for p in model.parameters() if p.requires_grad)),
        "frozen_model_params": int(sum(p.numel() for p in model.parameters() if not p.requires_grad)),
        "route_freeze": route_freeze_meta,
        "frozen_ss_encoder_params": int(frozen_encoder),
        "frozen_ss_decoder_params": int(frozen_decoder),
        "distributed": distributed,
        "rank": rank,
        "world_size": world_size,
        "ddp_find_unused_parameters": bool(str(args.route) == "voxel" and int(args.voxel_embedding_dim) > 0 and float(args.embed_loss_weight) <= 0),
        "semantic_vocab": semantic_vocab,
        "metric_kind": "voxel_iou_proxy" if str(args.route) == "voxel" else "ss_decode_iou",
        "refine_mode": str(args.refine_mode),
        "spconv_depth": int(args.spconv_depth),
        "xpart_ce_weight": float(args.xpart_ce_weight),
        "motion_loss_weight": float(args.motion_loss_weight),
        "motion_loss_kind": str(args.motion_loss_kind),
        "motion_max_angle_delta": int(args.motion_max_angle_delta),
        "motion_sidecar": motion_sidecar_meta,
        "infer_resolve": str(args.infer_resolve),
        "voxel_embedding_dim": int(args.voxel_embedding_dim) if str(args.route) == "voxel" else 0,
        "embed_loss_weight": float(args.embed_loss_weight),
        "embed_pull_margin": float(args.embed_pull_margin),
        "embed_push_margin": float(args.embed_push_margin),
        "embed_max_voxels_per_part": int(args.embed_max_voxels_per_part),
        "object_group_batches": group_batches,
        "warm_start": warm_start_meta,
        "small_oversample": int(args.small_oversample),
        "realappliance_oversample": int(oversampling_plan.get("realappliance_oversample", 1)),
        "realappliance_target_share": float(args.realappliance_target_share),
        "realappliance_max_oversample": int(args.realappliance_max_oversample),
        "verse_focus_oversample": int(args.verse_focus_oversample),
        "oversampling_plan": {k: v for k, v in oversampling_plan.items() if k != "repeat_by_key"},
        "focal_gamma": float(args.focal_gamma),
        "boundary_weight": float(args.boundary_weight),
        "auto_pack": bool(args.auto_pack),
        "base_packed_dir": str(args.base_packed_dir) if args.base_packed_dir is not None else None,
        "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
    }
    if distributed:
        ddp_find_unused = bool(str(args.route) == "voxel" and int(args.voxel_embedding_dim) > 0 and float(args.embed_loss_weight) <= 0)
        model = DistributedDataParallel(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=ddp_find_unused,
        )
    if is_rank0:
        (args.out_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        (args.out_dir / "train_rows.json").write_text(json.dumps([row.__dict__ for row in train_rows], indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        (args.out_dir / "proxy_train_rows.json").write_text(json.dumps([row.__dict__ for row in proxy_train_rows], indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        (args.out_dir / "proxy_eval_rows.json").write_text(json.dumps([row.__dict__ for row in proxy_eval_rows], indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        (args.out_dir / "full_eval_rows.json").write_text(json.dumps([row.__dict__ for row in full_eval_rows], indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(
            f"[PromptSeg-config] mode={args.mode} route={args.route} model_dim={args.dim} depth={args.depth} "
            f"batch_per_rank={args.batch_size} global_batch={int(args.batch_size) * world_size} lr={args.lr} "
            f"warmup={args.warmup_steps} steps={args.steps} split_json={args.split_json} proxy_json={args.proxy_json} "
            f"packed_dir={args.packed_dir} use_packed_whole_occ={args.use_packed_whole_occ} "
            f"refine_mode={args.refine_mode} spconv_depth={args.spconv_depth} "
            f"xpart_ce_weight={args.xpart_ce_weight} motion_loss_weight={args.motion_loss_weight} "
            f"motion_loss_kind={args.motion_loss_kind} "
            f"infer_resolve={args.infer_resolve} "
            f"view_dropout={args.view_dropout} min_views={args.min_views} min_prompt_views={args.min_prompt_views} "
            f"view_dropout_start_step={args.view_dropout_start_step} small_oversample={args.small_oversample} "
            f"realappliance_oversample={oversampling_plan.get('realappliance_oversample', 1)} "
            f"realappliance_target_share={args.realappliance_target_share} "
            f"verse_focus_oversample={args.verse_focus_oversample} "
            f"focal_gamma={args.focal_gamma} boundary_weight={args.boundary_weight} "
            f"auto_pack={args.auto_pack} out_dir={args.out_dir}",
            flush=True,
        )
        oversample_rows = []
        for item in oversampling_plan.get("tiers", []):
            oversample_rows.append({
                "tier": item["tier"],
                "rows": item["rows"],
                "objects": item["objects"],
                "small": item["small_rows"],
                "repeat": item["base_repeat"],
                "effective": item["effective_rows"],
                "share": f"{float(item['effective_share']) * 100.0:.2f}%",
            })
        print(
            "[PromptSeg-oversample]\n"
            + format_table(oversample_rows, ["tier", "rows", "objects", "small", "repeat", "effective", "share"]),
            flush=True,
        )
        print(
            f"[PromptSeg] device={device} train_rows={len(train_rows)} proxy_train_rows={len(proxy_train_rows)} "
            f"proxy_eval_rows={len(proxy_eval_rows)} full_eval_rows={len(full_eval_rows)} params={metadata['params']:,} "
            f"trainable_params={metadata['trainable_params']:,} frozen_ss_encoder={frozen_encoder:,} "
            f"frozen_ss_decoder={frozen_decoder:,} world_size={world_size}",
            flush=True,
        )

    start_step = 0
    if bool(args.motion_sanity_only) or float(args.motion_loss_weight) > 0:
        sanity = run_motion_gt_sanity(
            model.module if isinstance(model, DistributedDataParallel) else model,
            decoder,
            train_eval_loader,
            device=device,
            use_packed_whole_occ=bool(args.use_packed_whole_occ),
            loss_kind=str(args.motion_loss_kind),
            max_batches=8,
        )
        if is_rank0:
            (args.out_dir / "motion_gt_sanity.json").write_text(
                json.dumps(sanity, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            print(
                "[PromptSeg-motion-sanity] "
                f"items={sanity['items']:.0f} loss={sanity['motion_loss']:.4f} "
                f"bce={sanity['motion_bce']:.5f} dice={sanity['motion_dice']:.4f} "
                f"kind={args.motion_loss_kind} max_bce={float(args.motion_sanity_max_bce):.5f} "
                f"max_dice={float(args.motion_sanity_max_dice):.4f}",
                flush=True,
            )
        if sanity["items"] <= 0:
            raise RuntimeError("motion sanity found zero movable samples; refusing to train motion loss")
        if str(args.motion_loss_kind) == "bce" and float(sanity["motion_bce"]) > float(args.motion_sanity_max_bce):
            raise RuntimeError(
                "motion GT-membership sanity failed: "
                f"bce={sanity['motion_bce']:.5f} > max={float(args.motion_sanity_max_bce):.5f}. "
                "Do not train motion loss until voxel/frame alignment is fixed."
            )
        if str(args.motion_loss_kind) != "bce" and float(sanity["motion_dice"]) > float(args.motion_sanity_max_dice):
            raise RuntimeError(
                "motion GT-membership sanity failed: "
                f"dice={sanity['motion_dice']:.4f} > max={float(args.motion_sanity_max_dice):.4f}. "
                "Do not train motion loss until voxel/frame alignment is fixed."
            )
        if bool(args.motion_sanity_only):
            if is_rank0:
                print("[PromptSeg-motion-sanity] sanity-only requested; exiting before training", flush=True)
            return

    resume_path = args.resume
    if resume_path is None and bool(args.auto_resume):
        resume_path = latest_checkpoint_path(args.out_dir)
        if is_rank0:
            print(f"[PromptSeg-resume] auto_resume={bool(resume_path)} path={resume_path}", flush=True)
    if resume_path is not None:
        start_step, empty_code = load_checkpoint(resume_path, model, optimizer, scaler, device)
        if is_rank0:
            print(f"[PromptSeg] resumed {resume_path} at step={start_step}", flush=True)

    decode_eval_steps = parse_step_set(str(args.decode_eval_steps))
    data_iter = iter(train_loader)
    metrics_tail: list[dict[str, Any]] = []
    t0 = time.time()
    last_log_step = int(start_step)
    last_log_time = t0
    proxy_history: list[dict[str, Any]] = []
    held_predcand_best_by_bucket: dict[str, float] = {}
    best_early_metric = -float("inf")
    early_bad_count = 0
    for step in range(start_step + 1, int(args.steps) + 1):
        stop_after_eval = False
        model.train()
        if train_sampler is not None:
            train_sampler.set_epoch(step)
        step_time_start = time.perf_counter()
        data_wait_start = time.perf_counter()
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            batch = next(data_iter)
        data_wait_s = time.perf_counter() - data_wait_start

        compute_start = time.perf_counter()
        lr = lr_for_step(float(args.lr), step, warmup_steps=int(args.warmup_steps), total_steps=int(args.steps))
        for group in optimizer.param_groups:
            group["lr"] = lr
        optimizer.zero_grad(set_to_none=True)
        if scaler is not None and bool(args.fp16):
            with torch.cuda.amp.autocast(dtype=torch.float16):
                loss, items = train_step(
                    model,
                    decoder,
                    batch,
                    empty_code,
                    device=device,
                    decode_weight=float(args.decode_dice_weight),
                    latent_part_weight=float(args.latent_part_weight),
                    mask_augment=bool(args.mask_augment),
                    mask_only=bool(args.mask_only),
                    mask_target=str(args.mask_target),
                    support_multiplier=float(args.support_multiplier),
                    latent_loss_mode=str(args.latent_loss_mode),
                    route=str(args.route),
                    voxel_loss_weight=float(args.voxel_loss_weight),
                    voxel_max_tokens=int(args.voxel_max_tokens),
                    xpart_ce_weight=float(args.xpart_ce_weight),
                    motion_loss_weight=float(args.motion_loss_weight),
                    motion_loss_kind=str(args.motion_loss_kind),
                    embed_loss_weight=float(args.embed_loss_weight),
                    embed_pull_margin=float(args.embed_pull_margin),
                    embed_push_margin=float(args.embed_push_margin),
                    embed_max_voxels_per_part=int(args.embed_max_voxels_per_part),
                    view_dropout=bool(args.view_dropout),
                    min_views=int(args.min_views),
                    min_prompt_views=int(args.min_prompt_views),
                    view_dropout_start_step=int(args.view_dropout_start_step),
                    focal_gamma=float(args.focal_gamma),
                    boundary_weight=float(args.boundary_weight),
                    semantic_loss_weight=float(args.semantic_loss_weight) if bool(args.semantic_aux) else 0.0,
                    use_packed_whole_occ=bool(args.use_packed_whole_occ),
                    step=step,
                )
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            if float(args.grad_clip) > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(args.grad_clip))
            scaler.step(optimizer)
            scaler.update()
        else:
            loss, items = train_step(
                model,
                decoder,
                batch,
                empty_code,
                device=device,
                decode_weight=float(args.decode_dice_weight),
                latent_part_weight=float(args.latent_part_weight),
                mask_augment=bool(args.mask_augment),
                mask_only=bool(args.mask_only),
                mask_target=str(args.mask_target),
                support_multiplier=float(args.support_multiplier),
                latent_loss_mode=str(args.latent_loss_mode),
                route=str(args.route),
                voxel_loss_weight=float(args.voxel_loss_weight),
                voxel_max_tokens=int(args.voxel_max_tokens),
                xpart_ce_weight=float(args.xpart_ce_weight),
                motion_loss_weight=float(args.motion_loss_weight),
                motion_loss_kind=str(args.motion_loss_kind),
                embed_loss_weight=float(args.embed_loss_weight),
                embed_pull_margin=float(args.embed_pull_margin),
                embed_push_margin=float(args.embed_push_margin),
                embed_max_voxels_per_part=int(args.embed_max_voxels_per_part),
                view_dropout=bool(args.view_dropout),
                min_views=int(args.min_views),
                min_prompt_views=int(args.min_prompt_views),
                view_dropout_start_step=int(args.view_dropout_start_step),
                focal_gamma=float(args.focal_gamma),
                boundary_weight=float(args.boundary_weight),
                semantic_loss_weight=float(args.semantic_loss_weight) if bool(args.semantic_aux) else 0.0,
                use_packed_whole_occ=bool(args.use_packed_whole_occ),
                step=step,
            )
            loss.backward()
            if float(args.grad_clip) > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(args.grad_clip))
            optimizer.step()
        compute_s = time.perf_counter() - compute_start
        step_wall_s = time.perf_counter() - step_time_start
        timed_s = max(data_wait_s + compute_s, 1.0e-12)
        items.update({
            "step": step,
            "lr": lr,
            "step_wall_s": float(step_wall_s),
            "data_wait_s": float(data_wait_s),
            "compute_s": float(compute_s),
            "data_wait_pct": float(data_wait_s / timed_s * 100.0),
            "compute_pct": float(compute_s / timed_s * 100.0),
        })
        metrics_tail.append(items)

        if is_rank0 and (step == int(start_step) + 1 or step == 1 or step % int(args.log_every) == 0):
            elapsed = time.time() - t0
            now = time.time()
            recent_steps = max(1, step - last_log_step)
            recent_s_per_step = (now - last_log_time) / recent_steps
            recent_metrics = metrics_tail[-min(len(metrics_tail), recent_steps):]
            recent_data_wait = float(np.mean([m.get("data_wait_s", 0.0) for m in recent_metrics]))
            recent_compute = float(np.mean([m.get("compute_s", 0.0) for m in recent_metrics]))
            recent_timed = max(recent_data_wait + recent_compute, 1.0e-12)
            recent_data_wait_pct = recent_data_wait / recent_timed * 100.0
            recent_compute_pct = recent_compute / recent_timed * 100.0
            last_log_step = step
            last_log_time = now
            print(
                f"step {step}/{args.steps} total {items['total']:.4f} mask_bce {items['mask_bce']:.4f} "
                f"mask_dice {items['mask_dice']:.4f} latent_l1 {items['latent_l1_unweighted']:.4f} "
                f"decode_dice {items['decode_dice']:.4f} cell_iou {items['cell_iou']:.4f} "
                f"sem_acc {items.get('semantic_acc', 0.0):.4f} lr {lr:.2e} "
                f"s_per_step {recent_s_per_step:.4f} data_wait {recent_data_wait:.4f}s "
                f"compute {recent_compute:.4f}s data_wait_pct {recent_data_wait_pct:.1f} "
                f"compute_pct {recent_compute_pct:.1f} min_prompt_views={int(args.min_prompt_views)} "
                f"small_oversample={int(args.small_oversample)} focal_gamma={float(args.focal_gamma):.2f} "
                f"boundary_weight={float(args.boundary_weight):.2f} "
                f"boundary_ratio={float(items.get('boundary_voxel_ratio', 0.0)):.4f} "
                f"xpart_ce {float(items.get('xpart_ce', 0.0)):.4f} "
                f"motion_loss {float(items.get('motion_loss', 0.0)):.4f} "
                f"motion_dice {float(items.get('motion_dice', 0.0)):.4f} "
                f"embed_loss {float(items.get('embed_loss', 0.0)):.4f} "
                f"embed_pull {float(items.get('embed_pull', 0.0)):.4f} "
                f"embed_push {float(items.get('embed_push', 0.0)):.4f} "
                f"drop_single_view_after {int(items.get('view_dropout_single_after', 0.0))} "
                f"drop_active {int(items.get('view_dropout_active', 0.0))} "
                f"elapsed {elapsed/60:.1f}m",
                flush=True,
            )
            append_jsonl(args.out_dir / "logs" / "train_metrics.jsonl", dict(items))

        do_eval = step % int(args.eval_every) == 0 or step == int(args.steps)
        if distributed and do_eval:
            dist.barrier()
        if is_rank0 and do_eval:
            eval_model = model.module if isinstance(model, DistributedDataParallel) else model
            full_eval = (
                step in decode_eval_steps
                or (bool(args.final_full_eval) and step == int(args.steps))
                or (int(args.full_eval_every) > 0 and step % int(args.full_eval_every) == 0)
            )
            heldout_loader = full_eval_loader if full_eval else eval_loader
            heldout_max_rows = 0 if full_eval else (
                int(args.heldout_eval_max_rows) if int(args.heldout_eval_max_rows) > 0 else int(args.eval_max_rows)
            )
            train_result = evaluate(
                eval_model,
                decoder,
                train_eval_loader,
                empty_code,
                device=device,
                max_rows=int(args.train_eval_max_rows),
                write_visuals_dir=args.out_dir / "visuals" / f"step_{step:06d}" / "train",
                mask_only=bool(args.mask_only),
                mask_target=str(args.mask_target),
                support_multiplier=float(args.support_multiplier),
                route=str(args.route),
                voxel_max_tokens=int(args.voxel_max_tokens),
                use_packed_whole_occ=bool(args.use_packed_whole_occ),
                infer_resolve=str(args.infer_resolve),
            )
            heldout_result = evaluate(
                eval_model,
                decoder,
                heldout_loader,
                empty_code,
                device=device,
                max_rows=heldout_max_rows,
                write_visuals_dir=args.out_dir / "visuals" / f"step_{step:06d}" / "heldout",
                mask_only=bool(args.mask_only),
                mask_target=str(args.mask_target),
                support_multiplier=float(args.support_multiplier),
                route=str(args.route),
                voxel_max_tokens=int(args.voxel_max_tokens),
                use_packed_whole_occ=bool(args.use_packed_whole_occ),
                infer_resolve=str(args.infer_resolve),
            )
            peak_gb = measure_peak_memory(
                eval_model,
                decoder,
                eval_ds,
                empty_code,
                device=device,
                mask_target=str(args.mask_target),
                support_multiplier=float(args.support_multiplier),
                route=str(args.route),
                voxel_max_tokens=int(args.voxel_max_tokens),
                use_packed_whole_occ=bool(args.use_packed_whole_occ),
            )
            realappliance_heldout_summary = dataset_eval_summary(heldout_result["rows"], "realappliance")
            realappliance_table = summary_table(realappliance_heldout_summary)
            eval_dir = args.out_dir / "eval" / f"step_{step:06d}"
            eval_dir.mkdir(parents=True, exist_ok=True)
            train_dir = eval_dir / "train"
            heldout_dir = eval_dir / "heldout"
            train_dir.mkdir(parents=True, exist_ok=True)
            heldout_dir.mkdir(parents=True, exist_ok=True)
            (train_dir / "rows.json").write_text(json.dumps(train_result["rows"], indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
            (heldout_dir / "rows.json").write_text(json.dumps(heldout_result["rows"], indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
            (train_dir / "summary.json").write_text(json.dumps(train_result["summary"], indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
            (heldout_dir / "summary.json").write_text(json.dumps(heldout_result["summary"], indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
            (heldout_dir / "realappliance_summary.json").write_text(json.dumps(realappliance_heldout_summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
            table = combined_eval_table(train_result["summary"], heldout_result["summary"])
            worst_train = worst_rows_table(train_result["rows"], key="e2e_decode_iou", n=5)
            worst_heldout = worst_rows_table(heldout_result["rows"], key="e2e_decode_iou", n=5)
            pred_zero = predcand_zero_evidence_table(heldout_result["rows"], n=16)
            eval_meta = {
                "step": int(step),
                "full_eval": bool(full_eval),
                "metric_kind": metadata["metric_kind"],
                "peak_memory_gb_batch1": peak_gb,
                "redline_4090_gb": 8.0,
            }
            warn_lines = overfit_small_warnings(heldout_result["summary"], held_predcand_best_by_bucket)
            eval_meta["overfit_small_warnings"] = warn_lines
            elapsed = time.time() - t0
            avg_s_per_step = elapsed / max(1, step - start_step)
            proxy_row = proxy_metric_row(
                step=step,
                loss=float(items.get("total", float("nan"))),
                train_summary=train_result["summary"],
                heldout_summary=heldout_result["summary"],
                sem_acc=float(items.get("semantic_acc", 0.0)),
                peak_gb=peak_gb,
                s_per_step=avg_s_per_step,
                util=current_gpu_util(),
            )
            ra_all = realappliance_heldout_summary.get("all", {})
            proxy_row.update({
                "ra_held_n": int(ra_all.get("n", 0)),
                "ra_held_cell": float(ra_all.get("cell_iou", float("nan"))),
                "ra_held_GTcand": float(ra_all.get("head2_gtm_decode_iou", float("nan"))),
                "ra_held_Predcand": float(ra_all.get("e2e_decode_iou", float("nan"))),
            })
            if warn_lines:
                proxy_row["overfit_small_warn"] = " | ".join(warn_lines)
            proxy_history.append(proxy_row)
            eval_meta["proxy_row"] = proxy_row
            append_jsonl(args.out_dir / "logs" / "eval_proxy.jsonl", proxy_row)
            append_tsv(args.out_dir / "logs" / "eval_proxy.tsv", proxy_row)
            (eval_dir / "summary.json").write_text(
                json.dumps(
                    {
                        "train": train_result["summary"],
                        "heldout": heldout_result["summary"],
                        "realappliance_heldout": realappliance_heldout_summary,
                        "meta": eval_meta,
                    },
                    indent=2,
                    ensure_ascii=False,
                )
                + "\n",
                encoding="utf-8",
            )
            (eval_dir / "summary.txt").write_text(table + "\n\n[RealAppliance heldout]\n" + realappliance_table + "\n", encoding="utf-8")
            (eval_dir / "proxy_row.txt").write_text(proxy_metric_table(proxy_row) + "\n", encoding="utf-8")
            (eval_dir / "worst_train.txt").write_text(worst_train + "\n", encoding="utf-8")
            (eval_dir / "worst_heldout.txt").write_text(worst_heldout + "\n", encoding="utf-8")
            (eval_dir / "predcand_zero_evidence.txt").write_text(pred_zero + "\n", encoding="utf-8")
            print(
                f"[PromptSeg-eval] step={step} metric={metadata['metric_kind']} full_eval={full_eval} peak_gb={peak_gb:.3f}\n"
                f"{proxy_metric_table(proxy_row)}\n\n{table}\n\n[RealAppliance heldout]\n{realappliance_table}\n\n[worst train]\n{worst_train}\n\n"
                f"[worst heldout]\n{worst_heldout}\n\n[Predcand=0 evidence]\n{pred_zero}",
                flush=True,
            )
            for warn_line in warn_lines:
                print(warn_line, flush=True)
            append_code_update(
                f"# Promptable Part Seg Gate2 Eval step {step}\n\n"
                f"out_dir: `{eval_dir}`\nmetric: `{metadata['metric_kind']}`\nfull_eval: `{full_eval}`\n"
                f"peak_memory_gb_batch1: `{peak_gb:.3f}`\n\n```\n{table}\n```\n\nRealAppliance heldout:\n```\n{realappliance_table}\n```\n\nworst heldout:\n```\n{worst_heldout}\n```"
            )
            if float(args.memorize_threshold) > 0:
                ok, mins = memorized_eval(heldout_result, float(args.memorize_threshold))
                print(
                    f"[PromptSeg-memorize] step={step} threshold={float(args.memorize_threshold):.4f} "
                    f"ok={ok} mins={mins}",
                    flush=True,
                )
                if ok:
                    save_checkpoint(
                        args.out_dir / "ckpts" / f"step_{step}.pt",
                        model=model.module if isinstance(model, DistributedDataParallel) else model,
                        optimizer=optimizer,
                        scaler=scaler,
                        step=step,
                        args=args,
                        empty_code=empty_code,
                        metadata=metadata,
                    )
                    save_checkpoint(
                        args.out_dir / "ckpts" / "latest.pt",
                        model=model.module if isinstance(model, DistributedDataParallel) else model,
                        optimizer=optimizer,
                        scaler=scaler,
                        step=step,
                        args=args,
                        empty_code=empty_code,
                        metadata=metadata,
                    )
                    append_code_update(
                        f"# Promptable Part Seg Memorized\n\n"
                        f"mode: `{args.mode}`\nroute: `{args.route}`\n"
                        f"out_dir: `{args.out_dir}`\nstep: `{step}`\nthreshold: `{float(args.memorize_threshold)}`\nmins: `{mins}`"
                    )
                    stop_after_eval = True
            if step == int(args.steps):
                decision = final_decision(train_result["summary"], heldout_result["summary"], proxy_history)
                (args.out_dir / "final_decision.json").write_text(
                    json.dumps(decision, indent=2, ensure_ascii=False) + "\n",
                    encoding="utf-8",
                )
                (args.out_dir / "final_decision.txt").write_text(
                    f"verdict: {decision['verdict']}\naction: {decision['action']}\n"
                    f"train_Predcand: {decision['train_Predcand']:.4f}\n"
                    f"held_Predcand: {decision['held_Predcand']:.4f}\n",
                    encoding="utf-8",
                )
                print(f"[PromptSeg-final-decision] {decision}", flush=True)
            if int(args.early_stop_patience) > 0:
                metric_name = str(args.early_stop_metric)
                metric_value = float(proxy_row["held_Predcand"])
                if metric_name == "train/e2e_decode_iou":
                    metric_value = float(proxy_row["train_Predcand"])
                elif metric_name not in ("heldout/e2e_decode_iou", "held_Predcand"):
                    raise ValueError(f"unsupported early_stop_metric={metric_name!r}")
                if metric_value > best_early_metric + float(args.early_stop_min_delta):
                    best_early_metric = metric_value
                    early_bad_count = 0
                else:
                    early_bad_count += 1
                if early_bad_count >= int(args.early_stop_patience):
                    print(
                        f"[PromptSeg-early-stop] step={step} metric={metric_name} best={best_early_metric:.4f} "
                        f"bad_count={early_bad_count}",
                        flush=True,
                    )
                    save_checkpoint(
                        args.out_dir / "ckpts" / f"step_{step}.pt",
                        model=model.module if isinstance(model, DistributedDataParallel) else model,
                        optimizer=optimizer,
                        scaler=scaler,
                        step=step,
                        args=args,
                        empty_code=empty_code,
                        metadata=metadata,
                    )
                    save_checkpoint(
                        args.out_dir / "ckpts" / "latest.pt",
                        model=model.module if isinstance(model, DistributedDataParallel) else model,
                        optimizer=optimizer,
                        scaler=scaler,
                        step=step,
                        args=args,
                        empty_code=empty_code,
                        metadata=metadata,
                    )
                    stop_after_eval = True
        if distributed and do_eval:
            dist.barrier()
            flag = torch.tensor([1 if stop_after_eval else 0], device=device, dtype=torch.int32)
            dist.broadcast(flag, src=0)
            if int(flag.item()) != 0:
                dist.destroy_process_group()
                return 0
        elif stop_after_eval:
            return 0

        if is_rank0 and (step % int(args.ckpt_every) == 0 or step == int(args.steps)):
            save_checkpoint(
                args.out_dir / "ckpts" / f"step_{step}.pt",
                model=model.module if isinstance(model, DistributedDataParallel) else model,
                optimizer=optimizer,
                scaler=scaler,
                step=step,
                args=args,
                empty_code=empty_code,
                metadata=metadata,
            )
        if distributed and (step % int(args.ckpt_every) == 0 or step == int(args.steps)):
            dist.barrier()
        if is_rank0 and (step % int(args.ckpt_every) == 0 or step == int(args.steps)):
            save_checkpoint(
                args.out_dir / "ckpts" / "latest.pt",
                model=model.module if isinstance(model, DistributedDataParallel) else model,
                optimizer=optimizer,
                scaler=scaler,
                step=step,
                args=args,
                empty_code=empty_code,
                metadata=metadata,
            )

    if is_rank0:
        append_code_update(
            f"# Promptable Part Seg Train Complete\n\nmode: `{args.mode}`\nout_dir: `{args.out_dir}`\nlatest: `{args.out_dir / 'ckpts' / 'latest.pt'}`"
        )
    if distributed:
        dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
