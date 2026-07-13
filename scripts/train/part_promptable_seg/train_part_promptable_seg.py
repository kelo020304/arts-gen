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
    boundary_band_mask,
    bucket_name,
    build_oversampling_plan,
    build_semantic_vocab,
    collate_promptable_parts,
    compute_empty_code,
    decode_latents_to_coords,
    decode_metrics_for_batch,
    dense_occ_from_coords,
    dataset_specs_from_split,
    downsample_binary_mask,
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
DEFAULT_PACKED_V6 = Path("/robot/data-lab/jzh/art-gen/data/part_promptable_seg_packed_v6")
DEFAULT_JOINT_SMALL_PART_WEIGHT = 1.5

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

    def _group_cost(self, group: list[int]) -> int:
        # Keep the DDP rank-balancing estimate in sync with the cost-budgeted
        # sampler: joint token refine cost is roughly classes * candidate voxels.
        raw_total = sum(max(1, int(getattr(self.rows[idx], "raw_count", 1) or 1)) for idx in group)
        return max(1, raw_total) * max(1, len(group) + 1)

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
            # Keep per-rank work balanced at each global step. Sort by estimated
            # cost inside a shuffled run, then distribute adjacent-cost batches
            # across ranks after a per-step shuffle.
            batches = sorted(batches, key=lambda batch: sum(self._group_cost([idx]) for idx in batch), reverse=True)
            balanced: list[list[int]] = []
            for start in range(0, len(batches), self.num_replicas):
                block = batches[start : start + self.num_replicas]
                if self.shuffle:
                    rng.shuffle(block)
                balanced.extend(block)
            batches = balanced
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


class PackedObjectGroupCostBatchSampler(Sampler[list[int]]):
    """Shard-aware object/angle batches bounded by an estimated voxel token cost."""

    def __init__(
        self,
        dataset: PackedPromptablePartDataset,
        *,
        group_cost_budget: int,
        max_groups_per_batch: int,
        shuffle: bool,
        seed: int,
        num_replicas: int = 1,
        rank: int = 0,
        repeat_by_key: dict[str, int] | None = None,
    ) -> None:
        self.dataset = dataset
        self.rows = list(dataset.rows)
        self.entries = list(dataset.entries_for_rows)
        self.group_cost_budget = int(group_cost_budget)
        self.max_groups_per_batch = int(max_groups_per_batch)
        self.shuffle = bool(shuffle)
        self.seed = int(seed)
        self.num_replicas = int(num_replicas)
        self.rank = int(rank)
        self.repeat_by_key = {str(key): max(1, int(value)) for key, value in dict(repeat_by_key or {}).items()}
        self.epoch = 0
        if self.group_cost_budget <= 0:
            raise ValueError(f"group_cost_budget must be positive, got {self.group_cost_budget}")
        if self.max_groups_per_batch <= 0:
            raise ValueError(f"max_groups_per_batch must be positive, got {self.max_groups_per_batch}")
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

    def _group_cost(self, group: list[int]) -> int:
        # Joint token refine cost is dominated by classes * shared candidate S.
        # Index entries only expose per-part raw_count, so use their union-size
        # upper bound. The +1 class accounts for the body/background class.
        raw_total = sum(max(1, int(getattr(self.rows[idx], "raw_count", 1) or 1)) for idx in group)
        return max(1, raw_total) * max(1, len(group) + 1)

    def _pack_groups(self, groups: list[list[int]]) -> list[list[int]]:
        batches: list[list[int]] = []
        current: list[int] = []
        current_cost = 0
        current_groups = 0
        for group in groups:
            cost = self._group_cost(group)
            would_exceed_cost = current and current_cost + cost > self.group_cost_budget
            would_exceed_groups = current_groups >= self.max_groups_per_batch
            if would_exceed_cost or would_exceed_groups:
                batches.append(current)
                current = []
                current_cost = 0
                current_groups = 0
            current.extend(group)
            current_cost += cost
            current_groups += 1
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
    parser.add_argument("--mode", choices=("gate1", "gate2", "train"), default="gate1")
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=20260611)
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--eval-batch-size", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--prefetch-factor", type=int, default=0)
    parser.add_argument("--persistent-workers", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--pin-memory", action=argparse.BooleanOptionalAction, default=True)
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
    parser.add_argument("--joint-seg", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--body-class-weight", type=float, default=0.25)
    parser.add_argument("--joint-kmax", type=int, default=0)
    parser.add_argument("--joint-small-part-threshold", type=int, default=32)
    parser.add_argument("--joint-small-part-weight", type=float, default=DEFAULT_JOINT_SMALL_PART_WEIGHT)
    parser.add_argument("--joint-smooth-weight", type=float, default=0.0)
    parser.add_argument("--joint-smooth-same-label-weight", type=float, default=1.0)
    parser.add_argument("--joint-smooth-all-label-weight", type=float, default=0.0)
    parser.add_argument("--joint-smooth-cross-label-weight", type=float, default=0.0)
    parser.add_argument("--joint-smooth-neighborhood", type=int, choices=(6, 18, 26), default=6)
    parser.add_argument("--joint-crf-eval", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--joint-crf-iters", type=int, default=5)
    parser.add_argument("--joint-crf-pairwise", type=float, default=0.3)
    parser.add_argument("--joint-crf-neighborhood", type=int, choices=(6, 18, 26), default=6)
    parser.add_argument("--use-checkpoint", action=argparse.BooleanOptionalAction, default=False)
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
    parser.add_argument("--group-cost-budget", type=int, default=0)
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
    parser.add_argument("--precision", choices=("fp32", "fp16", "bf16"), default=None)
    parser.add_argument("--tf32", action=argparse.BooleanOptionalAction, default=None)
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
    parser.add_argument("--boundary-band-radius", "--boundary_band_radius", dest="boundary_band_radius", type=int, default=1)
    parser.add_argument("--boundary-hard-mining", "--boundary_hard_mining", dest="boundary_hard_mining", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--boundary-hard-mining-topk", "--boundary_hard_mining_topk", dest="boundary_hard_mining_topk", type=float, default=0.2)
    parser.add_argument("--boundary-hard-mining-weight", "--boundary_hard_mining_weight", dest="boundary_hard_mining_weight", type=float, default=2.0)
    parser.add_argument("--negative-prompt-channel", "--negative_prompt_channel", dest="negative_prompt_channel", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--negative-prompt-equivalence-check", "--negative_prompt_equivalence_check", dest="negative_prompt_equivalence_check", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--voxel-corrupt", "--voxel_corrupt", dest="voxel_corrupt", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--voxel-corrupt-drop-prob", "--voxel_corrupt_drop_prob", dest="voxel_corrupt_drop_prob", type=float, default=0.03)
    parser.add_argument("--voxel-corrupt-shell-prob", "--voxel_corrupt_shell_prob", dest="voxel_corrupt_shell_prob", type=float, default=0.08)
    parser.add_argument("--voxel-corrupt-speckle-prob", "--voxel_corrupt_speckle_prob", dest="voxel_corrupt_speckle_prob", type=float, default=0.0003)
    parser.add_argument("--voxel-corrupt-visualize-dir", "--voxel_corrupt_visualize_dir", dest="voxel_corrupt_visualize_dir", type=Path, default=None)
    parser.add_argument("--voxel-corrupt-visualize-count", "--voxel_corrupt_visualize_count", dest="voxel_corrupt_visualize_count", type=int, default=3)
    parser.add_argument("--voxel-corrupt-visualize-only", "--voxel_corrupt_visualize_only", dest="voxel_corrupt_visualize_only", action=argparse.BooleanOptionalAction, default=False)
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


def train_loader_kwargs(args: argparse.Namespace) -> dict[str, Any]:
    num_workers = max(0, int(args.num_workers))
    kwargs: dict[str, Any] = {
        "num_workers": num_workers,
        "collate_fn": collate_promptable_parts,
        "pin_memory": bool(args.pin_memory) and torch.cuda.is_available(),
    }
    if num_workers > 0:
        if int(args.prefetch_factor) > 0:
            kwargs["prefetch_factor"] = int(args.prefetch_factor)
        kwargs["persistent_workers"] = bool(args.persistent_workers)
    return kwargs


def eval_loader_kwargs(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "num_workers": 0,
        "collate_fn": collate_promptable_parts,
        "pin_memory": bool(args.pin_memory) and torch.cuda.is_available(),
    }


def resolve_precision(args: argparse.Namespace) -> str:
    if args.precision is not None:
        return str(args.precision)
    return "fp16" if bool(args.fp16) else "fp32"


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
    boundary_hard_mining: bool = False,
    boundary_hard_mining_topk: float = 0.2,
    boundary_hard_mining_weight: float = 2.0,
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
    hard_voxels = 0
    if boundary is not None and bool(boundary_hard_mining) and float(boundary_hard_mining_weight) > 1.0:
        topk = min(max(float(boundary_hard_mining_topk), 0.0), 1.0)
        if topk > 0.0:
            hard = torch.zeros_like(boundary, dtype=torch.bool)
            with torch.no_grad():
                hardness = bce_elem.detach()
                for idx in range(boundary.shape[0]):
                    band_idx = torch.nonzero(boundary[idx], as_tuple=False).flatten()
                    if band_idx.numel() == 0:
                        continue
                    k = max(1, int(math.ceil(float(band_idx.numel()) * topk)))
                    band_loss = hardness[idx, band_idx]
                    _, rel = torch.topk(band_loss, k=min(k, int(band_loss.numel())), largest=True)
                    hard[idx, band_idx[rel]] = True
            hard_voxels = int(hard.sum().detach().item())
            bce_elem = bce_elem * torch.where(
                hard,
                bce_elem.new_full((), float(boundary_hard_mining_weight)),
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
        "boundary_hard_mining": float(bool(boundary_hard_mining)),
        "boundary_hard_mining_topk": float(boundary_hard_mining_topk),
        "boundary_hard_mining_weight": float(boundary_hard_mining_weight),
        "boundary_hard_voxels": float(hard_voxels),
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


def _batch_angle_values(batch: dict[str, Any], count: int) -> list[int]:
    angle_raw = batch.get("angle_idx", [0] * int(count))
    if torch.is_tensor(angle_raw):
        return [int(v) for v in angle_raw.detach().cpu().view(-1).tolist()]
    return [int(v) for v in list(angle_raw)]


def _batch_group_keys(batch: dict[str, Any], count: int) -> list[tuple[str, str, int]]:
    dataset_ids = list(batch.get("dataset_id", [""] * int(count)))
    obj_ids = list(batch.get("obj_id", [""] * int(count)))
    angles = _batch_angle_values(batch, count)
    keys: list[tuple[str, str, int]] = []
    for idx in range(int(count)):
        keys.append((
            str(dataset_ids[idx]) if idx < len(dataset_ids) else "",
            str(obj_ids[idx]) if idx < len(obj_ids) else "",
            int(angles[idx]) if idx < len(angles) else 0,
        ))
    return keys


def build_negative_prompt_masks(
    batch: dict[str, Any],
    masks2d: torch.Tensor,
    *,
    enabled: bool,
) -> tuple[torch.Tensor | None, dict[str, float]]:
    if not bool(enabled):
        return None, {
            "negative_prompt_channel": 0.0,
            "negative_prompt_groups": 0.0,
            "negative_prompt_samples_with_other": 0.0,
            "negative_prompt_visible_pixels_mean": 0.0,
        }
    if masks2d.dim() != 4:
        raise ValueError(f"masks2d expected [B,V,H,W], got {tuple(masks2d.shape)}")
    bsz = int(masks2d.shape[0])
    groups: dict[tuple[str, str, int], list[int]] = {}
    for idx, key in enumerate(_batch_group_keys(batch, bsz)):
        groups.setdefault(key, []).append(idx)
    source = masks2d > 0.5
    negative = torch.zeros_like(masks2d, dtype=torch.float32)
    samples_with_other = 0
    for group in groups.values():
        if len(group) <= 1:
            continue
        for idx in group:
            others = [other for other in group if other != idx]
            if not others:
                continue
            union = source[others].any(dim=0)
            negative[idx] = union.to(dtype=negative.dtype)
            if bool(union.any().detach().item()):
                samples_with_other += 1
    return negative, {
        "negative_prompt_channel": 1.0,
        "negative_prompt_groups": float(sum(1 for group in groups.values() if len(group) > 1)),
        "negative_prompt_samples_with_other": float(samples_with_other),
        "negative_prompt_visible_pixels_mean": float(negative.flatten(1).sum(dim=1).mean().detach().item()) if bsz else 0.0,
    }


def boundary_flat_for_loss(
    m_gt: torch.Tensor,
    batch: dict[str, Any],
    *,
    radius: int,
    device: torch.device,
    boundary_weight: float,
    boundary_hard_mining: bool,
) -> torch.Tensor | None:
    if float(boundary_weight) <= 1.0 and not bool(boundary_hard_mining):
        return None
    radius = max(0, int(radius))
    if radius <= 0:
        return None
    if radius == 1 and "m_boundary" in batch:
        boundary = batch["m_boundary"].to(device=device, dtype=torch.float32)
    else:
        boundary = boundary_band_mask(m_gt.detach().float().cpu(), radius=radius).to(device=device, dtype=torch.float32)
    return boundary.reshape(boundary.shape[0], -1)


def corrupt_voxel_occ(
    full_occ: torch.Tensor,
    *,
    enabled: bool,
    drop_prob: float,
    shell_prob: float,
    speckle_prob: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    if not bool(enabled):
        count = float(full_occ.detach().sum().item())
        return full_occ, {
            "voxel_corrupt": 0.0,
            "voxel_corrupt_drop_prob": float(drop_prob),
            "voxel_corrupt_shell_prob": float(shell_prob),
            "voxel_corrupt_speckle_prob": float(speckle_prob),
            "voxel_corrupt_before_count": count,
            "voxel_corrupt_after_count": count,
            "voxel_corrupt_added": 0.0,
            "voxel_corrupt_dropped": 0.0,
        }
    if full_occ.dim() != 5 or full_occ.shape[1] != 1:
        raise ValueError(f"voxel corruption expects full_occ [B,1,D,H,W], got {tuple(full_occ.shape)}")
    drop_prob = min(max(float(drop_prob), 0.0), 1.0)
    shell_prob = min(max(float(shell_prob), 0.0), 1.0)
    speckle_prob = min(max(float(speckle_prob), 0.0), 1.0)
    occ = full_occ > 0.5
    before_count = float(occ.detach().sum().item())
    if before_count <= 0.0:
        return full_occ, {
            "voxel_corrupt": 1.0,
            "voxel_corrupt_drop_prob": drop_prob,
            "voxel_corrupt_shell_prob": shell_prob,
            "voxel_corrupt_speckle_prob": speckle_prob,
            "voxel_corrupt_before_count": 0.0,
            "voxel_corrupt_after_count": 0.0,
            "voxel_corrupt_added": 0.0,
            "voxel_corrupt_dropped": 0.0,
        }
    kept = occ & (torch.rand_like(full_occ) >= drop_prob)
    dilated = F.max_pool3d(occ.float(), kernel_size=3, stride=1, padding=1) > 0.5
    shell = dilated & (~occ)
    shell_add = shell & (torch.rand_like(full_occ) < shell_prob)
    speckle_add = (~dilated) & (torch.rand_like(full_occ) < speckle_prob)
    corrupted = kept | shell_add | speckle_add
    empty = corrupted.flatten(1).sum(dim=1) == 0
    if bool(empty.any().detach().item()):
        corrupted[empty] = occ[empty]
    after_count = float(corrupted.detach().sum().item())
    added = float((corrupted & (~occ)).detach().sum().item())
    dropped = float((occ & (~corrupted)).detach().sum().item())
    return corrupted.to(dtype=full_occ.dtype), {
        "voxel_corrupt": 1.0,
        "voxel_corrupt_drop_prob": drop_prob,
        "voxel_corrupt_shell_prob": shell_prob,
        "voxel_corrupt_speckle_prob": speckle_prob,
        "voxel_corrupt_before_count": before_count,
        "voxel_corrupt_after_count": after_count,
        "voxel_corrupt_added": added,
        "voxel_corrupt_dropped": dropped,
    }


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


def _coords_to_keys_tensor(coords: torch.Tensor | np.ndarray, *, device: torch.device) -> torch.Tensor:
    coords_t = torch.as_tensor(coords, dtype=torch.long, device=device).view(-1, 3)
    if coords_t.numel() == 0:
        return torch.empty((0,), dtype=torch.long, device=device)
    lo = int(coords_t.min().detach().item())
    hi = int(coords_t.max().detach().item())
    if lo < 0 or hi >= 64:
        raise ValueError(f"joint segmentation coords out of [0,64): min={lo} max={hi}")
    return coords_t[:, 0] * 4096 + coords_t[:, 1] * 64 + coords_t[:, 2]


def _candidate_cells_from_coords(coords: torch.Tensor | np.ndarray, *, device: torch.device) -> torch.Tensor:
    coords_t = torch.as_tensor(coords, dtype=torch.long, device=device).view(-1, 3)
    cells = torch.zeros((1, 16, 16, 16), dtype=torch.float32, device=device)
    if coords_t.numel() == 0:
        return cells
    lo = int(coords_t.min().detach().item())
    hi = int(coords_t.max().detach().item())
    if lo < 0 or hi >= 64:
        raise ValueError(f"joint candidate coords out of [0,64): min={lo} max={hi}")
    cell = torch.div(coords_t, 4, rounding_mode="floor").clamp(0, 15)
    cells[0, cell[:, 0], cell[:, 1], cell[:, 2]] = 1.0
    return cells


def _candidate_cells_from_latent_mask(mask: torch.Tensor, *, device: torch.device) -> torch.Tensor:
    mask_t = torch.as_tensor(mask, dtype=torch.float32, device=device)
    if mask_t.dim() == 3:
        mask_t = mask_t.unsqueeze(0)
    if mask_t.dim() != 4 or tuple(mask_t.shape[1:]) != (16, 16, 16):
        raise ValueError(f"joint candidate mask expected [B,16,16,16], got {tuple(mask_t.shape)}")
    return (mask_t > 0.5).float()


def _positions_for_keys(sorted_keys: torch.Tensor, query_keys: torch.Tensor) -> torch.Tensor:
    if query_keys.numel() == 0 or sorted_keys.numel() == 0:
        return torch.empty((0,), dtype=torch.long, device=sorted_keys.device)
    query_keys = torch.unique(query_keys.long(), sorted=True)
    pos = torch.searchsorted(sorted_keys, query_keys)
    in_range = pos < sorted_keys.shape[0]
    pos = pos[in_range]
    query_keys = query_keys[in_range]
    return pos[sorted_keys[pos] == query_keys]


def _build_joint_target(
    coord_keys: torch.Tensor,
    *,
    kept_part_keys: list[torch.Tensor],
    dropped_part_keys: list[torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    target = torch.zeros((coord_keys.shape[0],), dtype=torch.long, device=coord_keys.device)
    claim_count = torch.zeros_like(target)
    dropped_claim_count = torch.zeros_like(target)
    claim_mask = torch.zeros(
        (coord_keys.shape[0], len(kept_part_keys) + 1),
        dtype=torch.bool,
        device=coord_keys.device,
    )
    for class_idx, part_keys in enumerate(kept_part_keys, start=1):
        pos = _positions_for_keys(coord_keys, part_keys)
        if pos.numel() > 0:
            claim_count[pos] += 1
            target[pos] = int(class_idx)
            claim_mask[pos, int(class_idx)] = True
    for part_keys in dropped_part_keys:
        pos = _positions_for_keys(coord_keys, part_keys)
        if pos.numel() > 0:
            claim_count[pos] += 1
            dropped_claim_count[pos] += 1
            target[pos] = -100
    overlap = claim_count > 1
    target[overlap] = -100
    partial = overlap & (dropped_claim_count == 0) & (claim_mask.sum(dim=1) > 1)
    claim_mask[~partial] = False
    return target, overlap, claim_mask


def _joint_group_indices(batch: dict[str, Any]) -> list[list[int]]:
    order: list[str] = []
    groups: dict[str, list[int]] = {}
    for idx in range(len(batch["obj_id"])):
        key = _object_angle_key_from_batch(batch, idx)
        if key not in groups:
            groups[key] = []
            order.append(key)
        groups[key].append(idx)
    return [groups[key] for key in order]


def _dataset_root_for_joint_group(batch: dict[str, Any], idx: int) -> Path | None:
    root = str(batch.get("data_root", [""])[idx] if "data_root" in batch else "").strip()
    if root:
        path = Path(root)
        if path.is_dir():
            return path
    dataset_id = str(batch.get("dataset_id", [""])[idx] if "dataset_id" in batch else "")
    candidates: list[Path] = []
    if dataset_id == "physx-0511-drawer-door":
        candidates.append(
            Path("/robot/data-lab/jzh/art-gen/data/PhysX-Mobility-full-4view-0511/PhysX-Mobility-full-4view-0511")
        )
    if dataset_id:
        candidates.append(Path("/robot/data-lab/jzh/art-gen/data") / dataset_id)
        candidates.append(Path("/mnt/robot-data-lab/jzh/art-gen/data") / dataset_id)
    for candidate in candidates:
        if candidate.is_dir():
            return candidate
    return None


def _joint_body_prompt_from_raw_assets(
    batch: dict[str, Any],
    indices: list[int],
    *,
    device: torch.device,
) -> dict[str, Any] | None:
    first = int(indices[0])
    root = _dataset_root_for_joint_group(batch, first)
    if root is None:
        return None
    obj_id = str(batch["obj_id"][first])
    angle_idx = int(batch["angle_idx"][first])
    voxel_dir = root / "reconstruction" / "voxel_expanded" / obj_id / f"angle_{angle_idx}" / "64"
    if not voxel_dir.is_dir():
        return None
    part_names = {str(batch["part_name"][idx]) for idx in indices}
    body_paths = [
        path
        for path in sorted(voxel_dir.glob("ind_*.npy"))
        if path.stem.removeprefix("ind_") not in part_names
        and any(term in path.stem.lower() for term in ("body", "base"))
    ]
    if not body_paths:
        return None

    raw_arrays = [np.asarray(np.load(path), dtype=np.int64).reshape(-1, 3) for path in body_paths]
    raw_arrays = [arr for arr in raw_arrays if arr.size > 0]
    if not raw_arrays:
        return None
    raw_coords_np = np.unique(np.concatenate(raw_arrays, axis=0), axis=0)
    raw_coords = torch.as_tensor(raw_coords_np, dtype=torch.long, device=device)
    if raw_coords.numel() == 0:
        return None
    if int(raw_coords.min().detach().item()) < 0 or int(raw_coords.max().detach().item()) >= 64:
        raise ValueError(f"body raw coords out of [0,64): {voxel_dir}")

    part_labels = {int(batch["original_label"][idx]) for idx in indices}
    label_to_key: dict[int, str] = {}
    part_info_path = root / "reconstruction" / "part_info" / obj_id / "part_info.json"
    if part_info_path.is_file():
        payload = json.loads(part_info_path.read_text(encoding="utf-8"))
        parts = payload.get("parts", {})
        if isinstance(parts, dict):
            for name, meta in parts.items():
                if isinstance(meta, dict) and "label" in meta:
                    label_to_key[int(meta["label"])] = str(name)
    body_labels = sorted(
        label
        for label, name in label_to_key.items()
        if label not in part_labels and any(term in name.lower() for term in ("body", "base"))
    )
    mask_root = root / "renders" / obj_id / f"angle_{angle_idx}" / "mask"
    mask_views: list[np.ndarray] = []
    if body_labels and mask_root.is_dir():
        for view_idx in batch["view_indices"][first].detach().cpu().tolist():
            mask_path = mask_root / f"mask_{int(view_idx)}.npy"
            if not mask_path.is_file():
                raise FileNotFoundError(f"body prompt mask view not found: {mask_path}")
            label_map = np.asarray(np.load(mask_path))
            body_mask = np.isin(label_map, np.asarray(body_labels, dtype=label_map.dtype))
            mask_views.append(downsample_binary_mask(body_mask, 512))
    if not mask_views:
        return None
    masks2d = torch.from_numpy(np.stack(mask_views, axis=0)).float().unsqueeze(0).to(device=device)
    if int((masks2d.flatten(2).sum(dim=2) > 0).sum().detach().item()) <= 0:
        return None
    return {
        "mode": "prompted-part",
        "name": ",".join(path.stem.removeprefix("ind_") for path in body_paths),
        "masks2d": masks2d,
        "raw_coords": raw_coords,
        "raw_count": int(raw_coords.shape[0]),
        "original_labels": body_labels,
        "paths": [str(path) for path in body_paths],
    }


def _joint_class_bucket(meta: dict[str, Any]) -> str:
    if meta.get("kind") == "body":
        return "body"
    text = f"{meta.get('name', '')} {meta.get('semantic_type', '')}".lower()
    if "door" in text:
        return "door"
    if "drawer" in text:
        return "drawer"
    if any(term in text for term in ("knob", "button", "handle", "switch")):
        return "small"
    semantic = str(meta.get("semantic_type") or "").strip()
    return semantic if semantic else "part"


def _build_joint_group_prediction(
    model: PromptablePartLatentSegNet,
    *,
    z_global: torch.Tensor,
    masks2d: torch.Tensor,
    full_occ: torch.Tensor,
    batch: dict[str, Any],
    indices: list[int],
    device: torch.device,
    voxel_max_tokens: int,
    joint_kmax: int = 0,
    body_class_weight: float = 0.25,
    small_part_threshold: int = 32,
    small_part_weight: float = DEFAULT_JOINT_SMALL_PART_WEIGHT,
    subsample_parts: bool = False,
) -> dict[str, Any]:
    if int(voxel_max_tokens) != 0:
        raise ValueError("--joint-seg uses full shared candidate S; pass --voxel-max-tokens 0")
    if "whole_coords" not in batch:
        raise KeyError("--joint-seg requires batch['whole_coords']; use packed data with include_whole_coords and --use-packed-whole-occ")
    if not indices:
        raise ValueError("joint group has no part rows")
    first = int(indices[0])
    whole_coords = batch["whole_coords"][first]
    whole_keys = torch.unique(_coords_to_keys_tensor(whole_coords, device=device), sorted=True)
    if whole_keys.numel() == 0:
        raise ValueError(f"joint group {_object_angle_key_from_batch(batch, first)} has empty whole_coords")
    for idx in indices[1:]:
        other = torch.unique(_coords_to_keys_tensor(batch["whole_coords"][idx], device=device), sorted=True)
        if other.shape != whole_keys.shape or not bool(torch.equal(other, whole_keys)):
            raise ValueError(f"joint group {_object_angle_key_from_batch(batch, first)} has inconsistent whole_coords across rows")

    all_part_key_tensors: dict[int, torch.Tensor] = {}
    union_part_keys: list[torch.Tensor] = []
    for idx in indices:
        raw_keys = torch.unique(_coords_to_keys_tensor(batch["raw_coords"][idx], device=device), sorted=True)
        all_part_key_tensors[int(idx)] = raw_keys
        if raw_keys.numel() > 0:
            pos = _positions_for_keys(whole_keys, raw_keys)
            if int(pos.numel()) != int(raw_keys.numel()):
                raise ValueError(
                    f"joint group {_object_angle_key_from_batch(batch, first)} raw_coords for "
                    f"{batch['part_name'][idx]} are not a subset of whole_coords"
                )
            union_part_keys.append(raw_keys)
    union_keys = torch.unique(torch.cat(union_part_keys), sorted=True) if union_part_keys else torch.empty((0,), dtype=torch.long, device=device)
    body_count_total = int(whole_keys.numel() - _positions_for_keys(whole_keys, union_keys).numel())

    keep_positions = list(range(len(indices)))
    if bool(subsample_parts) and int(joint_kmax) > 0 and len(indices) > int(joint_kmax):
        perm = torch.randperm(len(indices), device=device)[: int(joint_kmax)]
        keep_positions = sorted(int(v) for v in perm.detach().cpu().tolist())
    kept_indices = [int(indices[pos]) for pos in keep_positions]
    kept_set = set(kept_indices)
    dropped_indices = [int(idx) for idx in indices if int(idx) not in kept_set]

    group_m = batch["m_gt"][indices].to(device=device, dtype=torch.float32)
    group_union_m = (group_m > 0.5).any(dim=0, keepdim=True).float()
    candidate_cells = _candidate_cells_from_latent_mask(mask_morphology(group_union_m, "dilate"), device=device)
    z_one = z_global[first : first + 1]
    full_one = full_occ[first : first + 1]
    cell_mask64 = candidate_cells.bool().unsqueeze(1)
    cell_mask64 = cell_mask64.repeat_interleave(4, dim=2).repeat_interleave(4, dim=3).repeat_interleave(4, dim=4)
    shared_s_raw = int(((full_one > 0.5) & cell_mask64).sum().detach().item())
    if shared_s_raw <= 0:
        raise ValueError(f"joint group {_object_angle_key_from_batch(batch, first)} has empty shared candidate S")
    class_meta: list[dict[str, Any]] = [
        {
            "kind": "body",
            "name": "body",
            "semantic_type": "body",
            "batch_idx": None,
            "raw_count": body_count_total,
            "body_mode": "learned-token",
            "body_prompt_name": "",
            "body_prompt_raw_count": 0,
            "body_prompt_original_labels": [],
            "render_label": 1,
        }
    ]
    prompt_masks = []
    for local_col, idx in enumerate(kept_indices, start=1):
        prompt_masks.append(masks2d[idx : idx + 1])
        class_meta.append(
            {
                "kind": "part",
                "name": str(batch["part_name"][idx]),
                "semantic_type": str(batch["semantic_type"][idx]),
                "batch_idx": int(idx),
                "original_label": int(batch["original_label"][idx]),
                "group_col": int(local_col),
                "render_label": int(local_col) + 1,
                "raw_count": int(torch.as_tensor(batch["raw_count"][idx]).item()),
            }
        )

    class_count = len(class_meta)
    prompt_batch = torch.cat(prompt_masks, dim=0)
    out = model(
        z_one,
        prompt_batch,
        candidate_cells=candidate_cells,
        full_occ=full_one,
        max_voxels_per_sample=int(voxel_max_tokens),
        joint_voxels=True,
    )

    coords0 = out["joint_coords"].to(device=device, dtype=torch.long)
    if coords0.numel() == 0:
        raise ValueError(f"joint group {_object_angle_key_from_batch(batch, first)} has no shared candidate voxels after cap")
    class_logits = out["joint_logits"].float()
    if tuple(class_logits.shape) != (coords0.shape[0], class_count):
        raise RuntimeError(
            f"joint voxel logits expected [{coords0.shape[0]},{class_count}], got {tuple(class_logits.shape)}"
        )
    coord_keys = coords0[:, 0] * 4096 + coords0[:, 1] * 64 + coords0[:, 2]
    order = torch.argsort(coord_keys)
    coord_keys = coord_keys[order]
    coords0 = coords0[order]
    class_logits = class_logits[order]

    target, overlap, overlap_claim_mask = _build_joint_target(
        coord_keys,
        kept_part_keys=[all_part_key_tensors[int(idx)] for idx in kept_indices],
        dropped_part_keys=[all_part_key_tensors[int(idx)] for idx in dropped_indices],
    )
    overlap_count = int(overlap.sum().detach().item())
    body_eval_count = int((target == 0).sum().detach().item())

    class_weight = torch.ones((len(class_meta),), dtype=torch.float32, device=device)
    class_weight[0] = float(body_class_weight)
    for class_idx, meta in enumerate(class_meta[1:], start=1):
        if int(meta.get("raw_count", 0)) < int(small_part_threshold):
            class_weight[class_idx] = float(small_part_weight)

    valid_target = target >= 0
    pred = class_logits.detach().float().argmax(dim=1)
    valid_pred = pred[valid_target]
    valid_gt = target[valid_target]
    inter_sum = 0
    union_sum = 0
    for class_idx in range(len(class_meta)):
        p = valid_pred == int(class_idx)
        g = valid_gt == int(class_idx)
        inter_sum += int((p & g).sum().item())
        union_sum += int((p | g).sum().item())
    boundary_stats = joint_boundary_metrics(
        {"class_logits": class_logits, "target": target, "coords": coords0},
        neighborhood=6,
    )

    return {
        "class_logits": class_logits,
        "target": target,
        "overlap_claim_mask": overlap_claim_mask,
        "coords": coords0,
        "class_meta": class_meta,
        "class_weight": class_weight,
        "stats": {
            "joint_whole_voxels": float(whole_keys.numel()),
            "joint_s_raw": float(shared_s_raw),
            "joint_s_eval": float(coord_keys.numel()),
            "joint_total_parts": float(len(indices)),
            "joint_kept_parts": float(len(kept_indices)),
            "joint_dropped_parts": float(len(dropped_indices)),
            "joint_body_voxels": float(body_eval_count),
            "joint_body_total_voxels": float(body_count_total),
            "joint_body_mode_prompted": 0.0,
            "joint_valid_voxels": float(valid_target.sum().detach().item()),
            "joint_overlap_voxels": float(overlap_count),
            "joint_overlap_ratio": float(overlap_count / max(1, coord_keys.numel())),
            "joint_argmax_has_body": float(bool(((pred == 0) & valid_target).any().detach().item())),
            "joint_argmax_mean_iou": float(inter_sum / max(1, union_sum)),
            "joint_group_cost": float((class_count + 1) * coord_keys.numel()),
            "fwd_count": 1.0,
            **boundary_stats,
        },
    }


def _joint_neighbor_offsets(neighborhood: int) -> list[tuple[int, int, int, float]]:
    out: list[tuple[int, int, int, float]] = []
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            for dz in (-1, 0, 1):
                if dx == 0 and dy == 0 and dz == 0:
                    continue
                nonzero = int(dx != 0) + int(dy != 0) + int(dz != 0)
                if int(neighborhood) == 6 and nonzero != 1:
                    continue
                if int(neighborhood) == 18 and nonzero > 2:
                    continue
                if (dx, dy, dz) <= (0, 0, 0):
                    continue
                out.append((dx, dy, dz, 1.0 / math.sqrt(float(dx * dx + dy * dy + dz * dz))))
    return out


@torch.no_grad()
def joint_boundary_metrics(pred: dict[str, Any], *, neighborhood: int = 6) -> dict[str, float]:
    logits = pred["class_logits"].float()
    coords = pred["coords"].to(device=logits.device, dtype=torch.long)
    target = pred["target"].to(device=logits.device, dtype=torch.long)
    pred_label = logits.detach().argmax(dim=1)
    return joint_boundary_metrics_from_labels(
        pred_label,
        coords,
        target,
        neighborhood=int(neighborhood),
    )


@torch.no_grad()
def joint_boundary_metrics_from_labels(
    pred_label: torch.Tensor,
    coords: torch.Tensor,
    target: torch.Tensor,
    *,
    neighborhood: int = 6,
) -> dict[str, float]:
    pred_label = pred_label.detach().to(device=coords.device, dtype=torch.long)
    coords = coords.to(device=pred_label.device, dtype=torch.long)
    target = target.to(device=pred_label.device, dtype=torch.long)
    out = {
        "joint_boundary_voxels": 0.0,
        "joint_boundary_correct": 0.0,
        "joint_boundary_acc": 0.0,
        "joint_boundary_error": 0.0,
        "joint_cross_label_pairs": 0.0,
        "joint_cross_label_pair_correct": 0.0,
        "joint_cross_label_pair_acc": 0.0,
        "joint_cross_label_same_pred": 0.0,
        "joint_cross_label_same_pred_rate": 0.0,
    }
    if pred_label.numel() == 0 or coords.shape[0] <= 1:
        return out

    valid = target >= 0
    if int(valid.sum().item()) <= 1:
        return out

    coord_keys = coords[:, 0] * 4096 + coords[:, 1] * 64 + coords[:, 2]
    source_ids = torch.arange(coords.shape[0], device=coords.device)
    boundary_mask = torch.zeros((coords.shape[0],), dtype=torch.bool, device=coords.device)
    cross_pair_count = 0
    cross_pair_correct = 0
    cross_pair_same_pred = 0
    for dx, dy, dz, _pair_w in _joint_neighbor_offsets(int(neighborhood)):
        delta = torch.tensor((dx, dy, dz), dtype=torch.long, device=coords.device)
        query = coords + delta.view(1, 3)
        in_bounds = ((query >= 0) & (query < 64)).all(dim=1)
        if not bool(in_bounds.any().item()):
            continue
        src = source_ids[in_bounds]
        q = query[in_bounds]
        q_keys = q[:, 0] * 4096 + q[:, 1] * 64 + q[:, 2]
        pos = torch.searchsorted(coord_keys, q_keys)
        pos_in = pos < coord_keys.shape[0]
        if not bool(pos_in.any().item()):
            continue
        src = src[pos_in]
        q_keys = q_keys[pos_in]
        pos = pos[pos_in]
        matched = coord_keys[pos] == q_keys
        if not bool(matched.any().item()):
            continue
        a = src[matched]
        b = pos[matched]
        valid_pair = valid[a] & valid[b]
        if not bool(valid_pair.any().item()):
            continue
        a = a[valid_pair]
        b = b[valid_pair]
        cross = target[a] != target[b]
        if not bool(cross.any().item()):
            continue
        a = a[cross]
        b = b[cross]
        boundary_mask[a] = True
        boundary_mask[b] = True
        pair_correct = (pred_label[a] == target[a]) & (pred_label[b] == target[b])
        cross_pair_count += int(a.shape[0])
        cross_pair_correct += int(pair_correct.sum().item())
        cross_pair_same_pred += int((pred_label[a] == pred_label[b]).sum().item())

    boundary = boundary_mask & valid
    boundary_count = int(boundary.sum().item())
    boundary_correct = int(((pred_label == target) & boundary).sum().item()) if boundary_count > 0 else 0
    boundary_acc = float(boundary_correct / max(1, boundary_count))
    cross_pair_acc = float(cross_pair_correct / max(1, cross_pair_count))
    cross_same_rate = float(cross_pair_same_pred / max(1, cross_pair_count))
    out.update(
        {
            "joint_boundary_voxels": float(boundary_count),
            "joint_boundary_correct": float(boundary_correct),
            "joint_boundary_acc": boundary_acc,
            "joint_boundary_error": float(1.0 - boundary_acc) if boundary_count > 0 else 0.0,
            "joint_cross_label_pairs": float(cross_pair_count),
            "joint_cross_label_pair_correct": float(cross_pair_correct),
            "joint_cross_label_pair_acc": cross_pair_acc,
            "joint_cross_label_same_pred": float(cross_pair_same_pred),
            "joint_cross_label_same_pred_rate": cross_same_rate,
        }
    )
    return out


@torch.no_grad()
def joint_crf_refine_labels(
    logits: torch.Tensor,
    coords: torch.Tensor,
    target: torch.Tensor,
    *,
    iterations: int,
    pairwise_weight: float,
    neighborhood: int,
) -> torch.Tensor:
    logits = logits.float()
    pred_label = logits.detach().argmax(dim=1)
    if (
        logits.numel() == 0
        or logits.shape[0] <= 1
        or logits.shape[1] <= 1
        or int(iterations) <= 0
        or float(pairwise_weight) <= 0.0
    ):
        return pred_label

    coords = coords.to(device=logits.device, dtype=torch.long)
    target = target.to(device=logits.device, dtype=torch.long)
    valid = target >= 0
    if int(valid.sum().detach().item()) <= 1:
        return pred_label

    unary = -F.log_softmax(logits, dim=1)
    coord_keys = coords[:, 0] * 4096 + coords[:, 1] * 64 + coords[:, 2]
    source_ids = torch.arange(coords.shape[0], device=coords.device)
    current = pred_label
    num_classes = int(logits.shape[1])
    for _ in range(int(iterations)):
        votes = logits.new_zeros((coords.shape[0], num_classes))
        for dx, dy, dz, pair_w in _joint_neighbor_offsets(int(neighborhood)):
            delta = torch.tensor((dx, dy, dz), dtype=torch.long, device=coords.device)
            query = coords + delta.view(1, 3)
            in_bounds = ((query >= 0) & (query < 64)).all(dim=1)
            if not bool(in_bounds.any().detach().item()):
                continue
            src = source_ids[in_bounds]
            q = query[in_bounds]
            q_keys = q[:, 0] * 4096 + q[:, 1] * 64 + q[:, 2]
            pos = torch.searchsorted(coord_keys, q_keys)
            pos_in = pos < coord_keys.shape[0]
            if not bool(pos_in.any().detach().item()):
                continue
            src = src[pos_in]
            q_keys = q_keys[pos_in]
            pos = pos[pos_in]
            matched = coord_keys[pos] == q_keys
            if not bool(matched.any().detach().item()):
                continue
            a = src[matched]
            b = pos[matched]
            valid_pair = valid[a] & valid[b]
            if not bool(valid_pair.any().detach().item()):
                continue
            a = a[valid_pair]
            b = b[valid_pair]
            w = logits.new_full((a.shape[0],), float(pair_w))
            votes.index_put_((a, current[b]), w, accumulate=True)
            votes.index_put_((b, current[a]), w, accumulate=True)
        updated = (unary - float(pairwise_weight) * votes).argmin(dim=1)
        updated = torch.where(valid, updated, current)
        if bool(torch.equal(updated, current)):
            current = updated
            break
        current = updated
    return current


def joint_partial_label_unary_loss(
    pred: dict[str, Any],
) -> tuple[torch.Tensor | None, dict[str, float]]:
    logits = pred["class_logits"].float()
    claim_mask = pred.get("overlap_claim_mask")
    if claim_mask is None or logits.numel() == 0:
        return None, {
            "joint_overlap_partial_unary": 0.0,
            "joint_overlap_supervised_voxels": 0.0,
            "joint_overlap_claim_mass": 0.0,
        }
    claim_mask = claim_mask.to(device=logits.device, dtype=torch.bool)
    if tuple(claim_mask.shape) != tuple(logits.shape):
        raise ValueError(
            f"joint overlap claim mask expected {tuple(logits.shape)}, got {tuple(claim_mask.shape)}"
        )
    supervised = claim_mask.any(dim=1)
    supervised_count = int(supervised.sum().detach().item())
    if supervised_count <= 0:
        return None, {
            "joint_overlap_partial_unary": 0.0,
            "joint_overlap_supervised_voxels": 0.0,
            "joint_overlap_claim_mass": 0.0,
        }
    allowed_log_probs = F.log_softmax(logits[supervised], dim=1).masked_fill(
        ~claim_mask[supervised],
        -torch.inf,
    )
    allowed_log_mass = torch.logsumexp(allowed_log_probs, dim=1)
    loss = -allowed_log_mass.mean()
    return loss, {
        "joint_overlap_partial_unary": float(loss.detach().item()),
        "joint_overlap_supervised_voxels": float(supervised_count),
        "joint_overlap_claim_mass": float(allowed_log_mass.detach().exp().mean().item()),
    }


def joint_pairwise_smooth_loss(
    pred: dict[str, Any],
    *,
    same_label_weight: float,
    all_label_weight: float,
    neighborhood: int,
    cross_label_weight: float = 0.0,
) -> tuple[torch.Tensor | None, dict[str, float]]:
    logits = pred["class_logits"].float()
    coords = pred["coords"].to(device=logits.device, dtype=torch.long)
    target = pred["target"].to(device=logits.device, dtype=torch.long)
    claim_mask_raw = pred.get("overlap_claim_mask")
    if claim_mask_raw is None:
        claim_mask = torch.zeros_like(logits, dtype=torch.bool)
    else:
        claim_mask = claim_mask_raw.to(device=logits.device, dtype=torch.bool)
        if tuple(claim_mask.shape) != tuple(logits.shape):
            raise ValueError(
                f"joint overlap claim mask expected {tuple(logits.shape)}, got {tuple(claim_mask.shape)}"
            )
    overlap_voxel = claim_mask.any(dim=1)
    has_overlap = bool(overlap_voxel.any().detach().item())
    if logits.numel() == 0 or coords.shape[0] <= 1:
        return None, {
            "joint_smooth": 0.0,
            "joint_smooth_pairs": 0.0,
            "joint_smooth_same_pairs": 0.0,
            "joint_smooth_cross_pairs": 0.0,
            "joint_smooth_overlap_pairs": 0.0,
        }
    if (
        float(same_label_weight) <= 0.0
        and float(all_label_weight) <= 0.0
        and float(cross_label_weight) <= 0.0
    ):
        return None, {
            "joint_smooth": 0.0,
            "joint_smooth_pairs": 0.0,
            "joint_smooth_same_pairs": 0.0,
            "joint_smooth_cross_pairs": 0.0,
            "joint_smooth_overlap_pairs": 0.0,
        }

    probs = F.softmax(logits, dim=1)
    coord_keys = coords[:, 0] * 4096 + coords[:, 1] * 64 + coords[:, 2]
    source_ids = torch.arange(coords.shape[0], device=coords.device)
    all_sum = logits.new_zeros(())
    all_weight = logits.new_zeros(())
    same_sum = logits.new_zeros(())
    same_weight = logits.new_zeros(())
    cross_sum = logits.new_zeros(())
    cross_weight = logits.new_zeros(())
    overlap_sum = logits.new_zeros(())
    overlap_weight = logits.new_zeros(())
    same_pair_count = 0
    cross_pair_count = 0
    overlap_pair_count = 0
    pair_count = 0
    for dx, dy, dz, pair_w in _joint_neighbor_offsets(int(neighborhood)):
        delta = torch.tensor((dx, dy, dz), dtype=torch.long, device=coords.device)
        query = coords + delta.view(1, 3)
        in_bounds = ((query >= 0) & (query < 64)).all(dim=1)
        if not bool(in_bounds.any().detach().item()):
            continue
        src = source_ids[in_bounds]
        q = query[in_bounds]
        q_keys = q[:, 0] * 4096 + q[:, 1] * 64 + q[:, 2]
        pos = torch.searchsorted(coord_keys, q_keys)
        pos_in = pos < coord_keys.shape[0]
        if not bool(pos_in.any().detach().item()):
            continue
        src = src[pos_in]
        q_keys = q_keys[pos_in]
        pos = pos[pos_in]
        matched = coord_keys[pos] == q_keys
        if not bool(matched.any().detach().item()):
            continue
        a = src[matched]
        b = pos[matched]
        hard_pair = (target[a] >= 0) & (target[b] >= 0)
        if bool(hard_pair.any().detach().item()):
            hard_a = a[hard_pair]
            hard_b = b[hard_pair]
            potts = 1.0 - (probs[hard_a] * probs[hard_b]).sum(dim=1)
            weight = logits.new_full((potts.shape[0],), float(pair_w))
            same = target[hard_a] == target[hard_b]
            if float(all_label_weight) > 0.0 and bool(same.any().detach().item()):
                all_potts = potts[same]
                all_w = weight[same]
                all_sum = all_sum + (all_potts * all_w).sum()
                all_weight = all_weight + all_w.sum()
            if float(same_label_weight) > 0.0 and bool(same.any().detach().item()):
                same_potts = potts[same]
                same_w = weight[same]
                same_sum = same_sum + (same_potts * same_w).sum()
                same_weight = same_weight + same_w.sum()
                same_pair_count += int(same_potts.shape[0])
            cross = ~same
            if float(cross_label_weight) > 0.0 and bool(cross.any().detach().item()):
                cross_dot = 1.0 - potts[cross]
                cross_w = weight[cross]
                cross_sum = cross_sum + (cross_dot * cross_w).sum()
                cross_weight = cross_weight + cross_w.sum()
                cross_pair_count += int(cross_dot.shape[0])
            pair_count += int(potts.shape[0])

        if float(same_label_weight) <= 0.0 or not has_overlap:
            continue

        a_overlap_b_hard = overlap_voxel[a] & (target[b] >= 0)
        if bool(a_overlap_b_hard.any().detach().item()):
            overlap_idx = a[a_overlap_b_hard]
            hard_idx = b[a_overlap_b_hard]
            allowed = claim_mask[overlap_idx, target[hard_idx]]
            if bool(allowed.any().detach().item()):
                overlap_idx = overlap_idx[allowed]
                hard_idx = hard_idx[allowed]
                hard_one_hot = F.one_hot(target[hard_idx], num_classes=logits.shape[1]).to(dtype=probs.dtype)
                spatial = 0.5 * (probs[overlap_idx] - hard_one_hot).square().sum(dim=1)
                spatial_w = logits.new_full((spatial.shape[0],), float(pair_w))
                overlap_sum = overlap_sum + (spatial * spatial_w).sum()
                overlap_weight = overlap_weight + spatial_w.sum()
                overlap_pair_count += int(spatial.shape[0])

        a_hard_b_overlap = (target[a] >= 0) & overlap_voxel[b]
        if bool(a_hard_b_overlap.any().detach().item()):
            hard_idx = a[a_hard_b_overlap]
            overlap_idx = b[a_hard_b_overlap]
            allowed = claim_mask[overlap_idx, target[hard_idx]]
            if bool(allowed.any().detach().item()):
                hard_idx = hard_idx[allowed]
                overlap_idx = overlap_idx[allowed]
                hard_one_hot = F.one_hot(target[hard_idx], num_classes=logits.shape[1]).to(dtype=probs.dtype)
                spatial = 0.5 * (probs[overlap_idx] - hard_one_hot).square().sum(dim=1)
                spatial_w = logits.new_full((spatial.shape[0],), float(pair_w))
                overlap_sum = overlap_sum + (spatial * spatial_w).sum()
                overlap_weight = overlap_weight + spatial_w.sum()
                overlap_pair_count += int(spatial.shape[0])

        both_overlap = overlap_voxel[a] & overlap_voxel[b]
        if bool(both_overlap.any().detach().item()):
            overlap_a = a[both_overlap]
            overlap_b = b[both_overlap]
            same_claims = (claim_mask[overlap_a] == claim_mask[overlap_b]).all(dim=1)
            if bool(same_claims.any().detach().item()):
                overlap_a = overlap_a[same_claims]
                overlap_b = overlap_b[same_claims]
                spatial = 0.5 * (probs[overlap_a] - probs[overlap_b]).square().sum(dim=1)
                spatial_w = logits.new_full((spatial.shape[0],), float(pair_w))
                overlap_sum = overlap_sum + (spatial * spatial_w).sum()
                overlap_weight = overlap_weight + spatial_w.sum()
                overlap_pair_count += int(spatial.shape[0])

    terms: list[torch.Tensor] = []
    if float(all_label_weight) > 0.0 and float(all_weight.detach().item()) > 0.0:
        terms.append(float(all_label_weight) * all_sum / all_weight.clamp_min(1.0))
    if float(same_label_weight) > 0.0 and float(same_weight.detach().item()) > 0.0:
        terms.append(float(same_label_weight) * same_sum / same_weight.clamp_min(1.0))
    if float(cross_label_weight) > 0.0 and float(cross_weight.detach().item()) > 0.0:
        terms.append(float(cross_label_weight) * cross_sum / cross_weight.clamp_min(1.0))
    if float(same_label_weight) > 0.0 and float(overlap_weight.detach().item()) > 0.0:
        terms.append(float(same_label_weight) * overlap_sum / overlap_weight.clamp_min(1.0))
    if not terms:
        return None, {
            "joint_smooth": 0.0,
            "joint_smooth_pairs": float(pair_count),
            "joint_smooth_same_pairs": float(same_pair_count),
            "joint_smooth_cross_pairs": float(cross_pair_count),
            "joint_smooth_overlap_pairs": float(overlap_pair_count),
        }
    loss = torch.stack(terms).sum()
    return loss, {
        "joint_smooth": float(loss.detach().item()),
        "joint_smooth_pairs": float(pair_count),
        "joint_smooth_same_pairs": float(same_pair_count),
        "joint_smooth_cross_pairs": float(cross_pair_count),
        "joint_smooth_overlap_pairs": float(overlap_pair_count),
    }


def joint_seg_loss(
    model: PromptablePartLatentSegNet,
    *,
    z_global: torch.Tensor,
    masks2d: torch.Tensor,
    full_occ: torch.Tensor,
    batch: dict[str, Any],
    device: torch.device,
    voxel_max_tokens: int,
    body_class_weight: float,
    joint_kmax: int,
    small_part_threshold: int,
    small_part_weight: float,
    joint_smooth_weight: float,
    joint_smooth_same_label_weight: float,
    joint_smooth_all_label_weight: float,
    joint_smooth_cross_label_weight: float,
    joint_smooth_neighborhood: int,
) -> tuple[torch.Tensor, dict[str, float]]:
    ce_total: torch.Tensor | None = None
    partial_unary_total: torch.Tensor | None = None
    smooth_items: list[dict[str, float]] = []
    smooth_losses: list[torch.Tensor] = []
    valid_total = 0
    supervised_total = 0
    overlap_supervised_total = 0
    overlap_partial_unary_sum = 0.0
    overlap_claim_mass_sum = 0.0
    group_count = 0
    stats_items: list[dict[str, float]] = []
    for indices in _joint_group_indices(batch):
        pred = _build_joint_group_prediction(
            model,
            z_global=z_global,
            masks2d=masks2d,
            full_occ=full_occ,
            batch=batch,
            indices=indices,
            device=device,
            voxel_max_tokens=int(voxel_max_tokens),
            joint_kmax=int(joint_kmax),
            body_class_weight=float(body_class_weight),
            small_part_threshold=int(small_part_threshold),
            small_part_weight=float(small_part_weight),
            subsample_parts=True,
        )
        target = pred["target"]
        valid = target != -100
        valid_count = int(valid.sum().detach().item())
        partial_unary, partial_item = joint_partial_label_unary_loss(pred)
        partial_count = int(partial_item["joint_overlap_supervised_voxels"])
        if valid_count + partial_count <= 0:
            continue
        if valid_count > 0:
            with torch.cuda.amp.autocast(enabled=False):
                ce_sum = F.cross_entropy(
                    pred["class_logits"].float(),
                    target.long(),
                    weight=pred["class_weight"].float(),
                    ignore_index=-100,
                    reduction="sum",
                )
        else:
            ce_sum = pred["class_logits"].sum() * 0.0
        if partial_unary is not None and partial_count > 0:
            partial_sum = partial_unary * float(partial_count)
            partial_unary_total = (
                partial_sum if partial_unary_total is None else partial_unary_total + partial_sum
            )
            overlap_partial_unary_sum += float(partial_unary.detach().item()) * float(partial_count)
            overlap_claim_mass_sum += float(partial_item["joint_overlap_claim_mass"]) * float(partial_count)
        ce_total = ce_sum if ce_total is None else ce_total + ce_sum
        if float(joint_smooth_weight) > 0.0:
            l_smooth, smooth_item = joint_pairwise_smooth_loss(
                pred,
                same_label_weight=float(joint_smooth_same_label_weight),
                all_label_weight=float(joint_smooth_all_label_weight),
                neighborhood=int(joint_smooth_neighborhood),
                cross_label_weight=float(joint_smooth_cross_label_weight),
            )
            smooth_items.append(smooth_item)
            if l_smooth is not None:
                smooth_losses.append(l_smooth)
        valid_total += valid_count
        supervised_total += valid_count + partial_count
        overlap_supervised_total += partial_count
        group_count += 1
        stats_items.append(pred["stats"])

    if ce_total is None or supervised_total <= 0:
        zero = z_global.sum() * 0.0
        return zero, {
            "joint_ce": 0.0,
            "joint_groups": 0.0,
            "joint_voxels": 0.0,
            "joint_supervised_voxels": 0.0,
            "joint_s_raw_min": 0.0,
            "joint_s_raw_max": 0.0,
            "joint_s_eval_min": 0.0,
            "joint_s_eval_max": 0.0,
            "joint_kept_parts": 0.0,
            "joint_total_parts": 0.0,
            "joint_dropped_parts": 0.0,
            "joint_overlap_voxels": 0.0,
            "joint_overlap_ratio": 0.0,
            "joint_overlap_supervised_voxels": 0.0,
            "joint_overlap_partial_unary": 0.0,
            "joint_overlap_claim_mass": 0.0,
            "joint_body_ratio": 0.0,
            "joint_body_prompted_groups": 0.0,
            "joint_argmax_has_body": 0.0,
            "joint_argmax_mean_iou": 0.0,
            "joint_group_cost": 0.0,
            "joint_group_cost_std": 0.0,
            "joint_boundary_voxels": 0.0,
            "joint_boundary_correct": 0.0,
            "joint_boundary_acc": 0.0,
            "joint_boundary_error": 0.0,
            "joint_cross_label_pairs": 0.0,
            "joint_cross_label_pair_correct": 0.0,
            "joint_cross_label_pair_acc": 0.0,
            "joint_cross_label_same_pred": 0.0,
            "joint_cross_label_same_pred_rate": 0.0,
            "joint_smooth": 0.0,
            "joint_pairwise_smooth": 0.0,
            "joint_smooth_weight": float(joint_smooth_weight),
            "joint_smooth_cross_label_weight": float(joint_smooth_cross_label_weight),
            "joint_smooth_pairs": 0.0,
            "joint_smooth_same_pairs": 0.0,
            "joint_smooth_cross_pairs": 0.0,
            "joint_smooth_overlap_pairs": 0.0,
            "fwd_count": 0.0,
        }
    ce_loss = ce_total / float(max(1, valid_total))
    pairwise_smooth_loss = torch.stack(smooth_losses).mean() if smooth_losses else ce_loss.detach() * 0.0
    partial_unary_loss = (
        partial_unary_total / float(overlap_supervised_total)
        if partial_unary_total is not None and overlap_supervised_total > 0
        else ce_loss.detach() * 0.0
    )
    smooth_loss = pairwise_smooth_loss + partial_unary_loss
    loss = ce_loss + float(joint_smooth_weight) * smooth_loss
    s_raw = [item["joint_s_raw"] for item in stats_items]
    s_eval = [item["joint_s_eval"] for item in stats_items]
    body = sum(item["joint_body_voxels"] for item in stats_items)
    raw = sum(item["joint_s_raw"] for item in stats_items)
    overlap_voxels = float(sum(item["joint_overlap_voxels"] for item in stats_items))
    eval_voxels = float(sum(item["joint_s_eval"] for item in stats_items))
    boundary_voxels = float(sum(item["joint_boundary_voxels"] for item in stats_items))
    boundary_correct = float(sum(item["joint_boundary_correct"] for item in stats_items))
    cross_pairs = float(sum(item["joint_cross_label_pairs"] for item in stats_items))
    cross_pair_correct = float(sum(item["joint_cross_label_pair_correct"] for item in stats_items))
    cross_same_pred = float(sum(item["joint_cross_label_same_pred"] for item in stats_items))
    boundary_acc = float(boundary_correct / max(1.0, boundary_voxels))
    cross_pair_acc = float(cross_pair_correct / max(1.0, cross_pairs))
    cross_same_rate = float(cross_same_pred / max(1.0, cross_pairs))
    return loss, {
        "joint_ce": float(ce_loss.detach().item()),
        "joint_groups": float(group_count),
        "joint_voxels": float(valid_total),
        "joint_supervised_voxels": float(supervised_total),
        "joint_s_raw_min": float(min(s_raw)),
        "joint_s_raw_max": float(max(s_raw)),
        "joint_s_eval_min": float(min(s_eval)),
        "joint_s_eval_max": float(max(s_eval)),
        "joint_kept_parts": float(np.mean([item["joint_kept_parts"] for item in stats_items])),
        "joint_total_parts": float(np.mean([item["joint_total_parts"] for item in stats_items])),
        "joint_dropped_parts": float(np.mean([item["joint_dropped_parts"] for item in stats_items])),
        "joint_overlap_voxels": overlap_voxels,
        "joint_overlap_ratio": float(overlap_voxels / max(1.0, eval_voxels)),
        "joint_overlap_supervised_voxels": float(overlap_supervised_total),
        "joint_overlap_partial_unary": float(
            overlap_partial_unary_sum / max(1, overlap_supervised_total)
        ),
        "joint_overlap_claim_mass": float(
            overlap_claim_mass_sum / max(1, overlap_supervised_total)
        ),
        "joint_body_ratio": float(body / max(1.0, raw)),
        "joint_body_prompted_groups": float(sum(item["joint_body_mode_prompted"] for item in stats_items)),
        "joint_argmax_has_body": float(min(item["joint_argmax_has_body"] for item in stats_items)),
        "joint_argmax_mean_iou": float(np.mean([item["joint_argmax_mean_iou"] for item in stats_items])),
        "joint_group_cost": float(np.mean([item["joint_group_cost"] for item in stats_items])),
        "joint_group_cost_std": float(np.std([item["joint_group_cost"] for item in stats_items])),
        "joint_boundary_voxels": boundary_voxels,
        "joint_boundary_correct": boundary_correct,
        "joint_boundary_acc": boundary_acc,
        "joint_boundary_error": float(1.0 - boundary_acc) if boundary_voxels > 0.0 else 0.0,
        "joint_cross_label_pairs": cross_pairs,
        "joint_cross_label_pair_correct": cross_pair_correct,
        "joint_cross_label_pair_acc": cross_pair_acc,
        "joint_cross_label_same_pred": cross_same_pred,
        "joint_cross_label_same_pred_rate": cross_same_rate,
        "joint_smooth": float(smooth_loss.detach().item()),
        "joint_pairwise_smooth": float(pairwise_smooth_loss.detach().item()),
        "joint_smooth_weight": float(joint_smooth_weight),
        "joint_smooth_cross_label_weight": float(joint_smooth_cross_label_weight),
        "joint_smooth_pairs": float(np.mean([item["joint_smooth_pairs"] for item in smooth_items])) if smooth_items else 0.0,
        "joint_smooth_same_pairs": float(np.mean([item["joint_smooth_same_pairs"] for item in smooth_items])) if smooth_items else 0.0,
        "joint_smooth_cross_pairs": float(np.mean([item["joint_smooth_cross_pairs"] for item in smooth_items])) if smooth_items else 0.0,
        "joint_smooth_overlap_pairs": float(np.mean([item["joint_smooth_overlap_pairs"] for item in smooth_items])) if smooth_items else 0.0,
        "fwd_count": float(np.mean([item["fwd_count"] for item in stats_items])),
    }


@torch.no_grad()
def joint_seg_eval_rows(
    model: PromptablePartLatentSegNet,
    *,
    z_global: torch.Tensor,
    masks2d: torch.Tensor,
    full_occ: torch.Tensor,
    batch: dict[str, Any],
    device: torch.device,
    voxel_max_tokens: int,
    body_class_weight: float,
    joint_kmax: int,
    small_part_threshold: int,
    small_part_weight: float,
    joint_crf_eval: bool = False,
    joint_crf_iters: int = 5,
    joint_crf_pairwise: float = 0.3,
    joint_crf_neighborhood: int = 6,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    groups = _joint_group_indices(batch)
    s_sizes: list[int] = []
    argmax_has_body = False
    body_modes: list[str] = []
    for indices in groups:
        pred = _build_joint_group_prediction(
            model,
            z_global=z_global,
            masks2d=masks2d,
            full_occ=full_occ,
            batch=batch,
            indices=indices,
            device=device,
            voxel_max_tokens=int(voxel_max_tokens),
            joint_kmax=int(joint_kmax),
            body_class_weight=float(body_class_weight),
            small_part_threshold=int(small_part_threshold),
            small_part_weight=float(small_part_weight),
            subsample_parts=False,
        )
        logits = pred["class_logits"].float()
        target = pred["target"].long()
        valid = target != -100
        pred_label = logits.argmax(dim=1)
        s_sizes.append(int(logits.shape[0]))
        if bool(((pred_label == 0) & valid).any().detach().item()):
            argmax_has_body = True
        class_meta = pred["class_meta"]
        stats = pred["stats"]
        crf_label = None
        crf_stats = None
        if bool(joint_crf_eval):
            crf_label = joint_crf_refine_labels(
                logits,
                pred["coords"].to(device=logits.device, dtype=torch.long),
                target,
                iterations=int(joint_crf_iters),
                pairwise_weight=float(joint_crf_pairwise),
                neighborhood=int(joint_crf_neighborhood),
            )
            crf_stats = joint_boundary_metrics_from_labels(
                crf_label,
                pred["coords"].to(device=logits.device, dtype=torch.long),
                target,
                neighborhood=int(joint_crf_neighborhood),
            )
        joint_group_key = _object_angle_key_from_batch(batch, int(indices[0]))
        body_modes.extend(str(meta.get("body_mode", "")) for meta in class_meta if meta.get("kind") == "body")
        for class_idx, meta in enumerate(class_meta):
            p = (pred_label == int(class_idx)) & valid
            g = (target == int(class_idx)) & valid
            inter = int((p & g).sum().detach().item())
            union = int((p | g).sum().detach().item())
            gt_count = int(g.sum().detach().item())
            pred_count = int(p.sum().detach().item())
            crf_iou = float("nan")
            crf_recall = float("nan")
            crf_precision = float("nan")
            crf_pred_count = 0
            if crf_label is not None:
                cp = (crf_label == int(class_idx)) & valid
                cinter = int((cp & g).sum().detach().item())
                cunion = int((cp | g).sum().detach().item())
                crf_pred_count = int(cp.sum().detach().item())
                crf_iou = float(cinter / max(1, cunion))
                crf_recall = float(cinter / max(1, gt_count))
                crf_precision = float(cinter / max(1, crf_pred_count))
            batch_idx = None if meta.get("batch_idx") is None else int(meta["batch_idx"])
            rows.append(
                {
                    "obj_id": batch["obj_id"][indices[0]],
                    "dataset_id": batch["dataset_id"][indices[0]],
                    "angle_idx": int(batch["angle_idx"][indices[0]]),
                    "class_name": str(meta["name"]),
                    "class_kind": _joint_class_bucket(meta),
                    "group_col": int(class_idx),
                    "render_label": int(meta.get("render_label", class_idx + 1)),
                    "batch_idx": batch_idx,
                    "part_id": None if batch_idx is None else int(batch["part_idx"][batch_idx]),
                    "original_label": None if batch_idx is None else int(batch["original_label"][batch_idx]),
                    "iou": float(inter / max(1, union)),
                    "recall": float(inter / max(1, gt_count)),
                    "precision": float(inter / max(1, pred_count)),
                    "gt_count": gt_count,
                    "pred_count": pred_count,
                    "voxel_share": float(gt_count / max(1, int(valid.sum().detach().item()))),
                    "joint_group_key": joint_group_key,
                    "joint_boundary_voxels": float(stats.get("joint_boundary_voxels", 0.0)),
                    "joint_boundary_correct": float(stats.get("joint_boundary_correct", 0.0)),
                    "joint_boundary_acc": float(stats.get("joint_boundary_acc", 0.0)),
                    "joint_boundary_error": float(stats.get("joint_boundary_error", 0.0)),
                    "joint_cross_label_pairs": float(stats.get("joint_cross_label_pairs", 0.0)),
                    "joint_cross_label_pair_correct": float(stats.get("joint_cross_label_pair_correct", 0.0)),
                    "joint_cross_label_pair_acc": float(stats.get("joint_cross_label_pair_acc", 0.0)),
                    "joint_cross_label_same_pred": float(stats.get("joint_cross_label_same_pred", 0.0)),
                    "joint_cross_label_same_pred_rate": float(stats.get("joint_cross_label_same_pred_rate", 0.0)),
                    "joint_crf_iou": crf_iou,
                    "joint_crf_recall": crf_recall,
                    "joint_crf_precision": crf_precision,
                    "joint_crf_pred_count": crf_pred_count,
                    "joint_crf_boundary_voxels": 0.0 if crf_stats is None else float(crf_stats.get("joint_boundary_voxels", 0.0)),
                    "joint_crf_boundary_correct": 0.0 if crf_stats is None else float(crf_stats.get("joint_boundary_correct", 0.0)),
                    "joint_crf_boundary_acc": 0.0 if crf_stats is None else float(crf_stats.get("joint_boundary_acc", 0.0)),
                    "joint_crf_boundary_error": 0.0 if crf_stats is None else float(crf_stats.get("joint_boundary_error", 0.0)),
                    "joint_crf_cross_label_pairs": 0.0 if crf_stats is None else float(crf_stats.get("joint_cross_label_pairs", 0.0)),
                    "joint_crf_cross_label_pair_correct": 0.0 if crf_stats is None else float(crf_stats.get("joint_cross_label_pair_correct", 0.0)),
                    "joint_crf_cross_label_pair_acc": 0.0 if crf_stats is None else float(crf_stats.get("joint_cross_label_pair_acc", 0.0)),
                    "joint_crf_cross_label_same_pred": 0.0 if crf_stats is None else float(crf_stats.get("joint_cross_label_same_pred", 0.0)),
                    "joint_crf_cross_label_same_pred_rate": 0.0 if crf_stats is None else float(crf_stats.get("joint_cross_label_same_pred_rate", 0.0)),
                }
            )
    meta = {
        "groups": int(len(groups)),
        "s_min": int(min(s_sizes)) if s_sizes else 0,
        "s_max": int(max(s_sizes)) if s_sizes else 0,
        "argmax_has_body": bool(argmax_has_body),
        "body_modes": sorted({mode for mode in body_modes if mode}),
    }
    return rows, meta


def summarize_joint_eval_rows(rows: list[dict[str, Any]], *, metric_prefix: str = "") -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    iou_key = f"{metric_prefix}_iou" if metric_prefix else "iou"
    recall_key = f"{metric_prefix}_recall" if metric_prefix else "recall"
    for kind in ("body", "door", "drawer", "small", "part"):
        selected = [
            row
            for row in rows
            if row.get("class_kind", row.get("joint_class_kind")) == kind
            and (iou_key in row or (not metric_prefix and "cell_iou" in row))
        ]
        if not selected:
            continue
        out[kind] = {
            "n": float(len(selected)),
            "iou": float(np.mean([float(row.get(iou_key, row.get("cell_iou", 0.0))) for row in selected])),
            "recall": float(np.mean([float(row.get(recall_key, row.get("e2e_recall", 0.0))) for row in selected])),
            "voxel_share": float(np.mean([float(row.get("voxel_share", row.get("joint_voxel_share", 0.0))) for row in selected])),
        }
    return out


def summarize_joint_boundary_rows(rows: list[dict[str, Any]], *, metric_prefix: str = "") -> dict[str, float]:
    seen: dict[str, dict[str, Any]] = {}
    key_prefix = metric_prefix or "joint"
    boundary_voxels_key = f"{key_prefix}_boundary_voxels"
    boundary_correct_key = f"{key_prefix}_boundary_correct"
    cross_pairs_key = f"{key_prefix}_cross_label_pairs"
    cross_correct_key = f"{key_prefix}_cross_label_pair_correct"
    cross_same_key = f"{key_prefix}_cross_label_same_pred"
    for row in rows:
        if boundary_voxels_key not in row and cross_pairs_key not in row:
            continue
        key = str(
            row.get("joint_group_key")
            or f"{row.get('dataset_id', '')}|{row.get('obj_id', '')}|{row.get('angle_idx', '')}"
        )
        if key not in seen:
            seen[key] = row
    if not seen:
        return {
            "groups": 0.0,
            "boundary_voxels": 0.0,
            "boundary_acc": float("nan"),
            "boundary_error": float("nan"),
            "cross_label_pairs": 0.0,
            "cross_label_pair_acc": float("nan"),
            "cross_label_same_pred_rate": float("nan"),
        }
    unique_rows = list(seen.values())
    boundary_voxels = float(sum(float(row.get(boundary_voxels_key, 0.0)) for row in unique_rows))
    boundary_correct = float(sum(float(row.get(boundary_correct_key, 0.0)) for row in unique_rows))
    cross_pairs = float(sum(float(row.get(cross_pairs_key, 0.0)) for row in unique_rows))
    cross_correct = float(sum(float(row.get(cross_correct_key, 0.0)) for row in unique_rows))
    cross_same = float(sum(float(row.get(cross_same_key, 0.0)) for row in unique_rows))
    boundary_acc = float(boundary_correct / boundary_voxels) if boundary_voxels > 0.0 else float("nan")
    cross_pair_acc = float(cross_correct / cross_pairs) if cross_pairs > 0.0 else float("nan")
    cross_same_rate = float(cross_same / cross_pairs) if cross_pairs > 0.0 else float("nan")
    return {
        "groups": float(len(unique_rows)),
        "boundary_voxels": boundary_voxels,
        "boundary_acc": boundary_acc,
        "boundary_error": float(1.0 - boundary_acc) if math.isfinite(boundary_acc) else float("nan"),
        "cross_label_pairs": cross_pairs,
        "cross_label_pair_acc": cross_pair_acc,
        "cross_label_same_pred_rate": cross_same_rate,
    }


def joint_eval_table(
    *,
    step: int,
    lr: float,
    loss_total: float,
    loss_joint_ce: float,
    grad_norm: float,
    train_rows: list[dict[str, Any]],
    heldout_rows: list[dict[str, Any]],
    metric_prefix: str = "",
) -> str:
    train_summary = summarize_joint_eval_rows(train_rows, metric_prefix=metric_prefix)
    held_summary = summarize_joint_eval_rows(heldout_rows, metric_prefix=metric_prefix)
    train_boundary = summarize_joint_boundary_rows(train_rows, metric_prefix=metric_prefix)
    held_boundary = summarize_joint_boundary_rows(heldout_rows, metric_prefix=metric_prefix)
    rows = []
    for kind in ("body", "door", "drawer", "small", "part"):
        tr = train_summary.get(kind, {})
        ho = held_summary.get(kind, {})
        if not tr and not ho:
            continue
        rows.append(
            {
                "class": kind,
                "train_IoU": f"{float(tr.get('iou', float('nan'))):.4f}",
                "held_IoU": f"{float(ho.get('iou', float('nan'))):.4f}",
                "voxel_share": f"{float(ho.get('voxel_share', tr.get('voxel_share', float('nan')))):.4f}",
                "recall": f"{float(ho.get('recall', tr.get('recall', float('nan')))):.4f}",
            }
        )
    body_iou = float(held_summary.get("body", {}).get("iou", float("nan")))
    part_values = [
        float(item["iou"])
        for kind, item in held_summary.items()
        if kind != "body" and math.isfinite(float(item.get("iou", float("nan"))))
    ]
    held_mean_part = float(np.mean(part_values)) if part_values else float("nan")
    header_prefix = "crf_" if metric_prefix else ""
    header = (
        f"step={int(step)} lr={float(lr):.6g} loss_total={float(loss_total):.6f} "
        f"loss_jointCE={float(loss_joint_ce):.6f} grad_norm={float(grad_norm):.6f} "
        f"{header_prefix}held_body_IoU={body_iou:.4f} {header_prefix}held_mean_part_IoU={held_mean_part:.4f} "
        f"{header_prefix}train_boundary_err={float(train_boundary.get('boundary_error', float('nan'))):.4f} "
        f"{header_prefix}held_boundary_err={float(held_boundary.get('boundary_error', float('nan'))):.4f} "
        f"{header_prefix}held_cross_same={float(held_boundary.get('cross_label_same_pred_rate', float('nan'))):.4f}"
    )
    return header + "\n" + format_table(rows, ["class", "train_IoU", "held_IoU", "voxel_share", "recall"])


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
    boundary_band_radius: int,
    boundary_hard_mining: bool,
    boundary_hard_mining_topk: float,
    boundary_hard_mining_weight: float,
    negative_prompt_channel: bool,
    voxel_corrupt: bool,
    voxel_corrupt_drop_prob: float,
    voxel_corrupt_shell_prob: float,
    voxel_corrupt_speckle_prob: float,
    semantic_loss_weight: float,
    use_packed_whole_occ: bool,
    joint_seg: bool = False,
    body_class_weight: float = 0.25,
    joint_kmax: int = 0,
    joint_small_part_threshold: int = 32,
    joint_small_part_weight: float = DEFAULT_JOINT_SMALL_PART_WEIGHT,
    joint_smooth_weight: float = 0.0,
    joint_smooth_same_label_weight: float = 1.0,
    joint_smooth_all_label_weight: float = 0.0,
    joint_smooth_cross_label_weight: float = 0.0,
    joint_smooth_neighborhood: int = 6,
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
    negative_masks2d, negative_prompt_stats = build_negative_prompt_masks(
        batch,
        masks2d,
        enabled=bool(negative_prompt_channel),
    )
    no_prompt_after_dropout = warn_all_empty_prompt_masks(batch, masks2d, context="train/post_dropout")
    latent_gt = batch["latent_gt"].to(device=device, dtype=torch.float32)
    m_raw = batch["m_gt"].to(device=device, dtype=torch.float32)
    boundary_flat = boundary_flat_for_loss(
        m_raw,
        batch,
        radius=int(boundary_band_radius),
        device=device,
        boundary_weight=float(boundary_weight),
        boundary_hard_mining=bool(boundary_hard_mining),
    )
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
        if bool(joint_seg):
            if not bool(use_packed_whole_occ):
                raise ValueError("--joint-seg requires --use-packed-whole-occ so body GT comes from packed whole_coords")
            full_occ = full_occ_for_batch(
                decoder,
                z_global,
                batch,
                device=device,
                use_packed_whole_occ=True,
            )
            l_joint, joint_items = joint_seg_loss(
                model,
                z_global=z_global,
                masks2d=masks2d,
                full_occ=full_occ,
                batch=batch,
                device=device,
                voxel_max_tokens=int(voxel_max_tokens),
                body_class_weight=float(body_class_weight),
                joint_kmax=int(joint_kmax),
                small_part_threshold=int(joint_small_part_threshold),
                small_part_weight=float(joint_small_part_weight),
                joint_smooth_weight=float(joint_smooth_weight),
                joint_smooth_same_label_weight=float(joint_smooth_same_label_weight),
                joint_smooth_all_label_weight=float(joint_smooth_all_label_weight),
                joint_smooth_cross_label_weight=float(joint_smooth_cross_label_weight),
                joint_smooth_neighborhood=int(joint_smooth_neighborhood),
            )
            if not torch.isfinite(l_joint.detach()):
                raise RuntimeError(f"joint segmentation loss is not finite: {float(l_joint.detach().item())}")
            return l_joint, {
                "mask_bce": 0.0,
                "mask_dice": 0.0,
                "voxel_bce": 0.0,
                "voxel_dice": 0.0,
                "voxel_iou": float(joint_items.get("joint_argmax_mean_iou", 0.0)),
                "voxel_precision": 0.0,
                "voxel_recall": 0.0,
                "voxel_pred_count": 0.0,
                "voxel_gt_count": 0.0,
                "xpart_ce": 0.0,
                "xpart_groups": 0.0,
                "xpart_voxels": 0.0,
                "xpart_fg_ratio": 0.0,
                "motion_loss": 0.0,
                "motion_bce": 0.0,
                "motion_dice": 0.0,
                "motion_items": 0.0,
                "motion_gt_sanity": 0.0,
                "semantic_ce": 0.0,
                "semantic_acc": 0.0,
                "embed_pull": 0.0,
                "embed_push": 0.0,
                "embed_loss": 0.0,
                "embed_groups": 0.0,
                "embed_parts": 0.0,
                "embed_voxels": 0.0,
                "xpart_ce_weight": 0.0,
                "motion_loss_weight": 0.0,
                "embed_loss_weight": 0.0,
                "latent_l1_weighted": 0.0,
                "latent_l1_unweighted": 0.0,
                "latent_l1_signal_norm": 0.0,
                "support_l1": 0.0,
                "decode_dice": 0.0,
                "cell_iou": float(joint_items.get("joint_argmax_mean_iou", 0.0)),
                "cell_precision": 0.0,
                "cell_recall": 0.0,
                "cell_pred_count": 0.0,
                "cell_gt_count": 0.0,
                "total": float(l_joint.detach().item()),
                "joint_seg": 1.0,
                "loss_jointCE": float(joint_items.get("joint_ce", 0.0)),
                "loss_jointSmooth": float(joint_items.get("joint_smooth", 0.0)),
                "morph": "joint_shared_whole_occ",
                "support_noise": float(support_noise.detach().mean().item()),
                "support_threshold": float(support_threshold.detach().mean().item()),
                "support_cell_count": float(m_gt.detach().sum(dim=(1, 2, 3)).mean().item()),
                "no_prompt_before_dropout": float(no_prompt_before_dropout),
                "no_prompt_after_dropout": float(no_prompt_after_dropout),
                "no_prompt_forward": 0.0,
                "boundary_band_radius": float(boundary_band_radius),
                **negative_prompt_stats,
                **dropout_stats,
                **joint_items,
            }
        full_occ = full_occ_for_batch(
            decoder,
            z_global,
            batch,
            device=device,
            use_packed_whole_occ=bool(use_packed_whole_occ),
        )
        full_occ, voxel_corrupt_stats = corrupt_voxel_occ(
            full_occ,
            enabled=bool(voxel_corrupt),
            drop_prob=float(voxel_corrupt_drop_prob),
            shell_prob=float(voxel_corrupt_shell_prob),
            speckle_prob=float(voxel_corrupt_speckle_prob),
        )
        out_v = model(
            z_global,
            masks2d,
            candidate_cells=mask_morphology(m_gt, "dilate"),
            full_occ=full_occ,
            max_voxels_per_sample=int(voxel_max_tokens),
            negative_masks2d=negative_masks2d,
        )
        no_prompt_hits = record_no_prompt_from_output(out_v, batch, context="train/forward", step=step)
        m_flat = m_gt.reshape(m_gt.shape[0], -1)
        l_mask, mask_items = mask_loss(
            out_v["m_logit"],
            m_flat,
            focal_gamma=float(focal_gamma),
            boundary_flat=boundary_flat,
            boundary_weight=float(boundary_weight),
            boundary_hard_mining=bool(boundary_hard_mining),
            boundary_hard_mining_topk=float(boundary_hard_mining_topk),
            boundary_hard_mining_weight=float(boundary_hard_mining_weight),
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
            "boundary_band_radius": float(boundary_band_radius),
            **negative_prompt_stats,
            **voxel_corrupt_stats,
            **dropout_stats,
            **mm,
        }

    out = model(z_global, masks2d, empty_code.to(device), m_override=m_head2, negative_masks2d=negative_masks2d)
    no_prompt_hits = record_no_prompt_from_output(out, batch, context="train/forward", step=step)
    m_flat = m_gt.reshape(m_gt.shape[0], -1)
    l_mask, mask_items = mask_loss(
        out["m_logit"],
        m_flat,
        focal_gamma=float(focal_gamma),
        boundary_flat=boundary_flat,
        boundary_weight=float(boundary_weight),
        boundary_hard_mining=bool(boundary_hard_mining),
        boundary_hard_mining_topk=float(boundary_hard_mining_topk),
        boundary_hard_mining_weight=float(boundary_hard_mining_weight),
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
            "boundary_band_radius": float(boundary_band_radius),
            **negative_prompt_stats,
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
        "boundary_band_radius": float(boundary_band_radius),
        **negative_prompt_stats,
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
    negative_prompt_channel: bool = False,
    joint_seg: bool = False,
    body_class_weight: float = 0.25,
    joint_kmax: int = 0,
    joint_small_part_threshold: int = 32,
    joint_small_part_weight: float = DEFAULT_JOINT_SMALL_PART_WEIGHT,
    joint_crf_eval: bool = False,
    joint_crf_iters: int = 5,
    joint_crf_pairwise: float = 0.3,
    joint_crf_neighborhood: int = 6,
) -> dict[str, Any]:
    model.eval()
    rows: list[dict[str, Any]] = []
    count = 0
    for batch in loader:
        z_global = batch["z_global"].to(device=device, dtype=torch.float32)
        masks2d = batch["masks2d"].to(device=device, dtype=torch.float32)
        negative_masks2d, _negative_prompt_stats = build_negative_prompt_masks(
            batch,
            masks2d,
            enabled=bool(negative_prompt_channel),
        )
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
            if bool(joint_seg):
                class_rows, joint_meta = joint_seg_eval_rows(
                    model,
                    z_global=z_global,
                    masks2d=masks2d,
                    full_occ=full_occ,
                    batch=batch,
                    device=device,
                    voxel_max_tokens=int(voxel_max_tokens),
                    body_class_weight=float(body_class_weight),
                    joint_kmax=int(joint_kmax),
                    small_part_threshold=int(joint_small_part_threshold),
                    small_part_weight=float(joint_small_part_weight),
                    joint_crf_eval=bool(joint_crf_eval),
                    joint_crf_iters=int(joint_crf_iters),
                    joint_crf_pairwise=float(joint_crf_pairwise),
                    joint_crf_neighborhood=int(joint_crf_neighborhood),
                )
                for class_row in class_rows:
                    raw_count = int(class_row["gt_count"])
                    class_batch_idx = class_row.get("batch_idx")
                    mask_visible_pixels = (
                        0
                        if class_batch_idx is None
                        else int(masks2d[int(class_batch_idx)].sum().detach().cpu().item())
                    )
                    row = {
                        "obj_id": class_row["obj_id"],
                        "dataset_id": class_row["dataset_id"],
                        "angle_idx": int(class_row["angle_idx"]),
                        "sample_id": f"{class_row['obj_id']}_angle_{int(class_row['angle_idx'])}",
                        "part_name": class_row["class_name"],
                        "semantic_type": class_row["class_kind"],
                        "raw_count": raw_count,
                        "mask_visible_pixels": mask_visible_pixels,
                        "bucket": bucket_name(raw_count),
                        "cell_iou": float(class_row["iou"]),
                        "cell_pred_count": int(class_row["pred_count"]),
                        "cell_gt_count": raw_count,
                        "support_l1": float("nan"),
                        "latent_l1": float("nan"),
                        "latent_l1_signal_norm": float("nan"),
                        "latent_signal": float("nan"),
                        "mask_target": "joint_single_owner",
                        "head2_gtm_decode_iou": float(class_row["iou"]),
                        "head2_gtm_precision": float(class_row["precision"]),
                        "head2_gtm_recall": float(class_row["recall"]),
                        "e2e_decode_iou": float(class_row["iou"]),
                        "e2e_precision": float(class_row["precision"]),
                        "e2e_recall": float(class_row["recall"]),
                        "partition_e2e_decode_iou": float(class_row["iou"]),
                        "partition_e2e_precision": float(class_row["precision"]),
                        "partition_e2e_recall": float(class_row["recall"]),
                        "gtm_pred_count": int(class_row["pred_count"]),
                        "e2e_pred_count": int(class_row["pred_count"]),
                        "partition_e2e_pred_count": int(class_row["pred_count"]),
                        "part_overlap_voxels": 0,
                        "object_overlap_voxels": 0,
                        "object_overlap_max_pair": 0,
                        "partition_part_overlap_voxels": 0,
                        "partition_object_overlap_voxels": 0,
                        "object_group_parts": int(joint_meta["groups"]),
                        "route": "voxel_joint",
                        "joint_class_kind": class_row["class_kind"],
                        "joint_group_col": int(class_row["group_col"]),
                        "joint_render_label": int(class_row["render_label"]),
                        "joint_batch_idx": class_batch_idx,
                        "joint_voxel_share": float(class_row["voxel_share"]),
                        "joint_group_key": str(class_row.get("joint_group_key", "")),
                        "joint_boundary_voxels": float(class_row.get("joint_boundary_voxels", 0.0)),
                        "joint_boundary_correct": float(class_row.get("joint_boundary_correct", 0.0)),
                        "joint_boundary_acc": float(class_row.get("joint_boundary_acc", 0.0)),
                        "joint_boundary_error": float(class_row.get("joint_boundary_error", 0.0)),
                        "joint_cross_label_pairs": float(class_row.get("joint_cross_label_pairs", 0.0)),
                        "joint_cross_label_pair_correct": float(class_row.get("joint_cross_label_pair_correct", 0.0)),
                        "joint_cross_label_pair_acc": float(class_row.get("joint_cross_label_pair_acc", 0.0)),
                        "joint_cross_label_same_pred": float(class_row.get("joint_cross_label_same_pred", 0.0)),
                        "joint_cross_label_same_pred_rate": float(class_row.get("joint_cross_label_same_pred_rate", 0.0)),
                        "joint_crf_iou": float(class_row.get("joint_crf_iou", float("nan"))),
                        "joint_crf_recall": float(class_row.get("joint_crf_recall", float("nan"))),
                        "joint_crf_precision": float(class_row.get("joint_crf_precision", float("nan"))),
                        "joint_crf_pred_count": int(class_row.get("joint_crf_pred_count", 0)),
                        "joint_crf_boundary_voxels": float(class_row.get("joint_crf_boundary_voxels", 0.0)),
                        "joint_crf_boundary_correct": float(class_row.get("joint_crf_boundary_correct", 0.0)),
                        "joint_crf_boundary_acc": float(class_row.get("joint_crf_boundary_acc", 0.0)),
                        "joint_crf_boundary_error": float(class_row.get("joint_crf_boundary_error", 0.0)),
                        "joint_crf_cross_label_pairs": float(class_row.get("joint_crf_cross_label_pairs", 0.0)),
                        "joint_crf_cross_label_pair_correct": float(class_row.get("joint_crf_cross_label_pair_correct", 0.0)),
                        "joint_crf_cross_label_pair_acc": float(class_row.get("joint_crf_cross_label_pair_acc", 0.0)),
                        "joint_crf_cross_label_same_pred": float(class_row.get("joint_crf_cross_label_same_pred", 0.0)),
                        "joint_crf_cross_label_same_pred_rate": float(class_row.get("joint_crf_cross_label_same_pred_rate", 0.0)),
                    }
                    rows.append(row)
                    count += 1
                if int(max_rows) > 0 and count >= int(max_rows):
                    break
                continue
            out_gt = model(
                z_global,
                masks2d,
                candidate_cells=mask_morphology(m_gt, "dilate"),
                full_occ=full_occ,
                max_voxels_per_sample=int(voxel_max_tokens),
                negative_masks2d=negative_masks2d,
            )
            record_no_prompt_from_output(out_gt, batch, context="eval/gt")
            pred_m = (out_gt["m_logit"].sigmoid() > 0.5).float().view(m_gt.shape)
            out_pred = model(
                z_global,
                masks2d,
                candidate_cells=mask_morphology(pred_m, "dilate"),
                full_occ=full_occ,
                max_voxels_per_sample=int(voxel_max_tokens),
                negative_masks2d=negative_masks2d,
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

        out_gt = model(z_global, masks2d, empty_code.to(device), m_override=m_gt, negative_masks2d=negative_masks2d)
        record_no_prompt_from_output(out_gt, batch, context="eval/gt")
        latent_stats = latent_signal_stats(out_gt["part_latent"], latent_gt, empty_code.to(device), m_gt)
        pred_m = (out_gt["m_logit"].sigmoid() > 0.5).float().view(m_gt.shape)
        if bool(mask_only):
            gt_metrics = [{"decode_iou": float("nan"), "decode_precision": float("nan"), "decode_recall": float("nan"), "pred_count": 0} for _ in range(z_global.shape[0])]
            pred_metrics = [{"decode_iou": float("nan"), "decode_precision": float("nan"), "decode_recall": float("nan"), "pred_count": 0} for _ in range(z_global.shape[0])]
        else:
            out_pred = model(z_global, masks2d, empty_code.to(device), m_override=pred_m, negative_masks2d=negative_masks2d)
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
    source_param_count = int(sum(value.numel() for value in source.values() if torch.is_tensor(value)))
    target_param_count = int(sum(value.numel() for value in target_state.values() if torch.is_tensor(value)))
    loadable_param_count = int(sum(value.numel() for value in loadable.values() if torch.is_tensor(value)))
    meta = {
        "path": str(path),
        "ckpt_step": int(ckpt.get("step", 0)),
        "loaded_count": len(loadable),
        "source_param_count": source_param_count,
        "target_param_count": target_param_count,
        "loadable_param_count": loadable_param_count,
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
        print(
            f"[PromptSeg-warm-start] params source={source_param_count:,} "
            f"target={target_param_count:,} loadable={loadable_param_count:,} "
            f"delta_target_source={target_param_count - source_param_count:,}",
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


@torch.no_grad()
def assert_negative_prompt_zero_equivalence(
    model,
    decoder,
    loader: DataLoader,
    empty_code: torch.Tensor,
    *,
    args: argparse.Namespace,
    device: torch.device,
    rank: int,
) -> None:
    if not bool(args.negative_prompt_channel) or not bool(args.negative_prompt_equivalence_check):
        return
    target = model.module if isinstance(model, DistributedDataParallel) else model
    was_training = bool(target.training)
    target.eval()
    try:
        batch = next(iter(loader))
        z_global = batch["z_global"].to(device=device, dtype=torch.float32)
        masks2d = batch["masks2d"].to(device=device, dtype=torch.float32)
        negative_masks2d, neg_stats = build_negative_prompt_masks(batch, masks2d, enabled=True)
        if negative_masks2d is None:
            raise RuntimeError("negative prompt equivalence check could not build negative masks")
        m_gt = batch["m_gt"].to(device=device, dtype=torch.float32)
        compare: list[tuple[str, torch.Tensor, torch.Tensor]] = []
        with torch.cuda.amp.autocast(enabled=False):
            if str(args.route) == "voxel":
                full_occ = full_occ_for_batch(
                    decoder,
                    z_global,
                    batch,
                    device=device,
                    use_packed_whole_occ=bool(args.use_packed_whole_occ),
                )
                candidate_cells = mask_morphology(m_gt, "dilate")
                base = target(
                    z_global,
                    masks2d,
                    candidate_cells=candidate_cells,
                    full_occ=full_occ,
                    max_voxels_per_sample=int(args.voxel_max_tokens),
                    negative_masks2d=None,
                )
                with_neg = target(
                    z_global,
                    masks2d,
                    candidate_cells=candidate_cells,
                    full_occ=full_occ,
                    max_voxels_per_sample=int(args.voxel_max_tokens),
                    negative_masks2d=negative_masks2d,
                )
                compare.append(("m_logit", base["m_logit"], with_neg["m_logit"]))
                compare.append(("voxel_logits", base["voxel_logits"], with_neg["voxel_logits"]))
            else:
                base = target(z_global, masks2d, empty_code.to(device), m_override=m_gt, negative_masks2d=None)
                with_neg = target(z_global, masks2d, empty_code.to(device), m_override=m_gt, negative_masks2d=negative_masks2d)
                compare.append(("m_logit", base["m_logit"], with_neg["m_logit"]))
                compare.append(("part_latent", base["part_latent"], with_neg["part_latent"]))
        max_abs = 0.0
        worst = ""
        for name, lhs, rhs in compare:
            delta = float((lhs.float() - rhs.float()).abs().max().detach().item())
            if delta > max_abs:
                max_abs = delta
                worst = name
        tol = 1.0e-6
        if max_abs > tol:
            raise RuntimeError(
                f"negative prompt zero-init equivalence failed on rank={rank}: "
                f"max_abs={max_abs:.6e} tensor={worst} tol={tol:.1e}"
            )
        if rank == 0:
            print(
                "[PromptSeg-negative-prompt] zero_init_equivalence=passed "
                f"max_abs={max_abs:.6e} groups={int(neg_stats['negative_prompt_groups'])} "
                f"samples_with_other={int(neg_stats['negative_prompt_samples_with_other'])}",
                flush=True,
            )
    finally:
        if was_training:
            target.train()


def _voxel_projection_image(occ: torch.Tensor) -> list[np.ndarray]:
    vol = occ.detach().float().cpu()
    if vol.dim() == 4:
        vol = vol.squeeze(0)
    if vol.dim() != 3:
        raise ValueError(f"expected one voxel volume [D,H,W], got {tuple(vol.shape)}")
    return [
        vol.max(dim=0).values.numpy(),
        vol.max(dim=1).values.numpy(),
        vol.max(dim=2).values.numpy(),
    ]


def write_voxel_corrupt_visuals(
    loader: DataLoader,
    *,
    args: argparse.Namespace,
    device: torch.device,
) -> list[dict[str, Any]]:
    if args.voxel_corrupt_visualize_dir is None:
        return []
    out_dir = Path(args.voxel_corrupt_visualize_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    batch = next(iter(loader))
    full_occ = dense_occ_from_batch_whole(batch, device=device)
    corrupted, stats = corrupt_voxel_occ(
        full_occ,
        enabled=True,
        drop_prob=float(args.voxel_corrupt_drop_prob),
        shell_prob=float(args.voxel_corrupt_shell_prob),
        speckle_prob=float(args.voxel_corrupt_speckle_prob),
    )
    count = min(int(args.voxel_corrupt_visualize_count), int(full_occ.shape[0]))
    rows: list[dict[str, Any]] = []
    obj_ids = list(batch.get("obj_id", [""] * count))
    part_names = list(batch.get("part_name", [""] * count))
    angle_values = _batch_angle_values(batch, int(full_occ.shape[0]))
    for idx in range(count):
        before = full_occ[idx, 0] > 0.5
        after = corrupted[idx, 0] > 0.5
        added = after & (~before)
        dropped = before & (~after)
        before_proj = _voxel_projection_image(before.float())
        after_proj = _voxel_projection_image(after.float())
        delta_proj = _voxel_projection_image((added | dropped).float())
        fig, axes = plt.subplots(3, 3, figsize=(7.5, 7.5))
        titles = ("xy", "xz", "yz")
        for col, title in enumerate(titles):
            axes[0, col].imshow(before_proj[col], cmap="gray", interpolation="nearest")
            axes[0, col].set_title(f"before {title}")
            axes[1, col].imshow(after_proj[col], cmap="gray", interpolation="nearest")
            axes[1, col].set_title(f"after {title}")
            axes[2, col].imshow(delta_proj[col], cmap="magma", interpolation="nearest")
            axes[2, col].set_title(f"delta {title}")
        for ax in axes.flat:
            ax.set_xticks([])
            ax.set_yticks([])
        obj_id = str(obj_ids[idx]) if idx < len(obj_ids) else ""
        part_name = str(part_names[idx]) if idx < len(part_names) else ""
        angle_idx = int(angle_values[idx]) if idx < len(angle_values) else 0
        before_count = int(before.sum().item())
        after_count = int(after.sum().item())
        added_count = int(added.sum().item())
        dropped_count = int(dropped.sum().item())
        fig.suptitle(
            f"{obj_id} a{angle_idx} {part_name}\n"
            f"before={before_count} after={after_count} added={added_count} dropped={dropped_count}",
            fontsize=8,
        )
        fig.tight_layout()
        out_path = out_dir / f"voxel_corrupt_{idx:02d}.png"
        fig.savefig(out_path, dpi=160)
        plt.close(fig)
        rows.append({
            "idx": int(idx),
            "path": str(out_path),
            "obj_id": obj_id,
            "angle_idx": angle_idx,
            "part_name": part_name,
            "before_count": before_count,
            "after_count": after_count,
            "added_count": added_count,
            "dropped_count": dropped_count,
        })
    meta = {
        "voxel_corrupt_stats": stats,
        "rows": rows,
        "drop_prob": float(args.voxel_corrupt_drop_prob),
        "shell_prob": float(args.voxel_corrupt_shell_prob),
        "speckle_prob": float(args.voxel_corrupt_speckle_prob),
    }
    (out_dir / "manifest.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(
        f"[PromptSeg-voxel-corrupt-vis] dir={out_dir} count={len(rows)} "
        f"drop_prob={float(args.voxel_corrupt_drop_prob)} "
        f"shell_prob={float(args.voxel_corrupt_shell_prob)} "
        f"speckle_prob={float(args.voxel_corrupt_speckle_prob)}",
        flush=True,
    )
    for row in rows:
        print(
            "[PromptSeg-voxel-corrupt-vis] "
            f"path={row['path']} before={row['before_count']} after={row['after_count']} "
            f"added={row['added_count']} dropped={row['dropped_count']} "
            f"id={row['obj_id']}/a{row['angle_idx']}/{row['part_name']}",
            flush=True,
        )
    return rows


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


def _sample_ids_for_nonfinite(batch: dict[str, Any], limit: int = 32) -> list[dict[str, Any]]:
    obj_ids = list(batch.get("obj_id", []))
    dataset_ids = list(batch.get("dataset_id", [""] * len(obj_ids)))
    sample_ids = list(batch.get("sample_id", [""] * len(obj_ids)))
    part_names = list(batch.get("part_name", [""] * len(obj_ids)))
    angle_raw = batch.get("angle_idx", [0] * len(obj_ids))
    angles = angle_raw.detach().cpu().tolist() if torch.is_tensor(angle_raw) else list(angle_raw)
    out: list[dict[str, Any]] = []
    for idx, obj_id in enumerate(obj_ids[: max(0, int(limit))]):
        out.append(
            {
                "dataset_id": str(dataset_ids[idx]) if idx < len(dataset_ids) else "",
                "obj_id": str(obj_id),
                "angle_idx": int(angles[idx]) if idx < len(angles) else 0,
                "sample_id": str(sample_ids[idx]) if idx < len(sample_ids) else "",
                "part_name": str(part_names[idx]) if idx < len(part_names) else "",
            }
        )
    return out


def _float_or_string(value: Any) -> float | str:
    try:
        out = float(value.detach().float().item() if torch.is_tensor(value) else value)
    except Exception:
        return str(value)
    if math.isfinite(out):
        return out
    if math.isnan(out):
        return "nan"
    return "inf" if out > 0 else "-inf"


def _record_nonfinite_grad(
    *,
    out_dir: Path,
    step: int,
    rank: int,
    grad_norm: Any,
    scaler: torch.cuda.amp.GradScaler | None,
    loss: torch.Tensor,
    items: dict[str, Any],
    batch: dict[str, Any],
    streak: int,
) -> list[dict[str, Any]]:
    sample_ids = _sample_ids_for_nonfinite(batch)
    loss_items = {
        str(key): _float_or_string(value)
        for key, value in items.items()
        if isinstance(value, (int, float)) or torch.is_tensor(value)
    }
    scale_value = float(scaler.get_scale()) if scaler is not None and scaler.is_enabled() else None
    append_jsonl(
        Path(out_dir) / "nonfinite_batches.jsonl",
        {
            "step": int(step),
            "rank": int(rank),
            "grad_norm": _float_or_string(grad_norm),
            "scale": scale_value,
            "loss": _float_or_string(loss),
            "loss_items": loss_items,
            "sample_ids": sample_ids,
            "streak": int(streak),
        },
    )
    return sample_ids


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
        undetectable_selected = [rec for rec in records if rec["classification"] == "undetectable_selected_views"]
        undetectable_all = [rec for rec in records if rec["classification"] == "undetectable_all_views"]
        undetectable = [*undetectable_selected, *undetectable_all]
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
            "undetectable_selected_rows": len(undetectable_selected),
            "undetectable_all_views_rows": len(undetectable_all),
            "label_absent_rows": len(absent),
            "splits": split_summaries,
        })
        (report_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        (report_dir / "records.json").write_text(json.dumps(records, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        (report_dir / "undetectable.json").write_text(json.dumps(undetectable, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        (report_dir / "undetectable_selected_views.json").write_text(json.dumps(undetectable_selected, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        (report_dir / "undetectable_all_views.json").write_text(json.dumps(undetectable_all, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        (report_dir / "label_absent_all_views.json").write_text(json.dumps(absent, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        write_mask_visibility_tsv(report_dir / "records.tsv", records)
        write_mask_visibility_tsv(report_dir / "undetectable.tsv", undetectable)
        write_mask_visibility_tsv(report_dir / "undetectable_selected_views.tsv", undetectable_selected)
        write_mask_visibility_tsv(report_dir / "undetectable_all_views.tsv", undetectable_all)
        write_mask_visibility_tsv(report_dir / "label_absent_all_views.tsv", absent)
        label_absent_ratio = float(len(absent) / max(1, len(records)))
        undetectable_ratio = float(len(undetectable) / max(1, len(records)))
        print(
            f"[PromptSeg-mask-audit] total={len(records)} visible={audit['class_counts'].get('visible_selected_views', 0)} "
            f"undetectable={len(undetectable)} ({undetectable_ratio:.4%}; "
            f"selected={len(undetectable_selected)} all_views={len(undetectable_all)}) "
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
    negative_prompt_channel: bool = False,
) -> float:
    if not torch.cuda.is_available() or not str(device).startswith("cuda"):
        return 0.0
    torch.cuda.set_device(device)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(torch.cuda.current_device())
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
        negative_prompt_channel=bool(negative_prompt_channel),
        joint_seg=False,
    )
    peak_gb = float(torch.cuda.max_memory_allocated(torch.cuda.current_device()) / (1024 ** 3))
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


def freeze_unused_route_parameters(
    model: PromptablePartLatentSegNet,
    *,
    route: str,
    joint_seg: bool = False,
    is_rank0: bool,
) -> dict[str, int]:
    frozen = 0
    frozen_params = 0
    frozen_names: list[str] = []

    def freeze_module(name: str, module: torch.nn.Module | None) -> None:
        nonlocal frozen, frozen_params
        if module is None:
            return
        for param in module.parameters():
            if param.requires_grad:
                frozen += 1
                frozen_params += int(param.numel())
                param.requires_grad_(False)
                frozen_names.append(str(name))

    if str(route) == "voxel":
        latent_only_modules = [
            ("m_emb", model.m_emb),
            ("head2_in", model.head2_in),
            ("head2_blocks", model.head2_blocks),
            ("head2_norm", model.head2_norm),
            ("delta", model.delta),
        ]
        for name, module in latent_only_modules:
            freeze_module(name, module)
        if bool(joint_seg):
            joint_unused_modules = [
                ("head1_norm", getattr(model, "head1_norm", None)),
                ("head1", getattr(model, "head1", None)),
                ("voxel_blocks", getattr(model, "voxel_blocks", None)),
                ("spconv_refine", getattr(model, "spconv_refine", None)),
                ("voxel_norm", getattr(model, "voxel_norm", None)),
                ("voxel_out", getattr(model, "voxel_out", None)),
                ("voxel_embed_out", getattr(model, "voxel_embed_out", None)),
                ("semantic_norm", getattr(model, "semantic_norm", None)),
                ("semantic_head", getattr(model, "semantic_head", None)),
            ]
            for name, module in joint_unused_modules:
                freeze_module(name, module)
    trainable = int(sum(p.numel() for p in model.parameters() if p.requires_grad))
    frozen_total = int(sum(p.numel() for p in model.parameters() if not p.requires_grad))
    if is_rank0:
        print(
            f"[PromptSeg-freeze] route={route} joint_seg={bool(joint_seg)} route_unused_tensors={frozen} "
            f"route_unused_params={frozen_params:,} trainable_model_params={trainable:,} "
            f"frozen_model_params={frozen_total:,}",
            flush=True,
        )
    return {
        "route_unused_tensors": int(frozen),
        "route_unused_params": int(frozen_params),
        "trainable_model_params": trainable,
        "frozen_model_params": frozen_total,
        "frozen_module_names": sorted(set(frozen_names)),
    }


def main() -> int:
    args = parse_args()
    if args.resume is not None and args.warm_start is not None:
        raise ValueError("--resume and --warm-start are mutually exclusive")
    os.environ["PROMPTSEG_DDP_TIMEOUT_S"] = str(max(3600, int(args.pack_barrier_timeout_s)))
    precision = resolve_precision(args)
    args.fp16 = precision in {"fp16", "bf16"}
    if args.tf32 is not None and torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = bool(args.tf32)
        torch.backends.cudnn.allow_tf32 = bool(args.tf32)
    if args.packed_dir is None and bool(args.auto_pack):
        args.packed_dir = DEFAULT_PACKED_V6
    distributed, rank, world_size, local_rank = init_distributed_if_needed()
    is_rank0 = rank == 0
    if args.out_dir is None:
        args.out_dir = DEFAULT_GATE1_OUT if args.mode == "gate1" else DEFAULT_GATE2_OUT
    if is_rank0:
        args.out_dir.mkdir(parents=True, exist_ok=True)
    configure_no_prompt_tracker(args.out_dir, rank=rank)
    if str(args.route) == "voxel" and int(args.voxel_max_tokens) != 0:
        raise ValueError("Route-V voxel cap is retired; pass --voxel-max-tokens 0")
    if int(args.boundary_band_radius) < 0:
        raise ValueError(f"--boundary-band-radius must be >= 0, got {args.boundary_band_radius}")
    if not 0.0 <= float(args.boundary_hard_mining_topk) <= 1.0:
        raise ValueError(f"--boundary-hard-mining-topk must be in [0,1], got {args.boundary_hard_mining_topk}")
    if bool(args.negative_prompt_channel) and bool(args.joint_seg):
        raise ValueError("--negative-prompt-channel is for the old per-part promptable route; do not combine it with --joint-seg")
    if bool(args.voxel_corrupt) and str(args.route) != "voxel":
        raise ValueError("--voxel-corrupt requires --route voxel")
    if bool(args.voxel_corrupt) and bool(args.joint_seg):
        raise ValueError("--voxel-corrupt is for the old per-part promptable route; do not combine it with --joint-seg")
    for name in ("voxel_corrupt_drop_prob", "voxel_corrupt_shell_prob", "voxel_corrupt_speckle_prob"):
        value = float(getattr(args, name))
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"--{name.replace('_', '-')} must be in [0,1], got {value}")
    if bool(args.joint_seg):
        if str(args.route) != "voxel":
            raise ValueError("--joint-seg requires --route voxel")
        if not bool(args.use_packed_whole_occ):
            raise ValueError("--joint-seg requires --use-packed-whole-occ")
        if int(args.joint_small_part_threshold) < 0:
            raise ValueError(f"--joint-small-part-threshold must be >= 0, got {args.joint_small_part_threshold}")
        if float(args.joint_small_part_weight) < 0.0:
            raise ValueError(f"--joint-small-part-weight must be >= 0, got {args.joint_small_part_weight}")
        if (
            is_rank0
            and int(args.joint_small_part_threshold) > 0
            and 0.0 <= float(args.joint_small_part_weight) < 1.0
        ):
            print(
                f"[PromptSeg-warning] joint_small_part_weight={args.joint_small_part_weight} "
                "downweights small parts; use >=1.0 for class-imbalance compensation.",
                flush=True,
            )
    for name in (
        "joint_smooth_weight",
        "joint_smooth_same_label_weight",
        "joint_smooth_all_label_weight",
        "joint_smooth_cross_label_weight",
    ):
        value = float(getattr(args, name))
        if value < 0.0:
            raise ValueError(f"--{name.replace('_', '-')} must be >= 0, got {value}")
    if int(args.joint_crf_iters) < 0:
        raise ValueError(f"--joint-crf-iters must be >= 0, got {args.joint_crf_iters}")
    if float(args.joint_crf_pairwise) < 0.0:
        raise ValueError(f"--joint-crf-pairwise must be >= 0, got {args.joint_crf_pairwise}")
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
        else (
            bool(args.joint_seg)
            or bool(args.negative_prompt_channel)
            or float(args.embed_loss_weight) > 0
            or float(args.xpart_ce_weight) > 0
        )
    )
    eval_group_batches = str(args.route) == "voxel"
    if args.packed_dir is not None:
        if group_batches and int(args.group_cost_budget) > 0:
            train_sampler = PackedObjectGroupCostBatchSampler(
                train_ds,
                group_cost_budget=int(args.group_cost_budget),
                max_groups_per_batch=int(args.batch_size),
                shuffle=True,
                seed=int(args.seed),
                num_replicas=world_size if distributed else 1,
                rank=rank if distributed else 0,
                repeat_by_key=repeat_by_key,
            )
        elif group_batches:
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
            **train_loader_kwargs(args),
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
            **eval_loader_kwargs(args),
        )
        train_eval_loader = DataLoader(
            proxy_train_ds,
            batch_sampler=train_eval_sampler,
            **eval_loader_kwargs(args),
        )
        full_eval_loader = DataLoader(
            full_eval_ds,
            batch_sampler=full_eval_sampler,
            **eval_loader_kwargs(args),
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
                **train_loader_kwargs(args),
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
                **train_loader_kwargs(args),
                drop_last=False,
            )
        if eval_group_batches:
            eval_loader = DataLoader(
                eval_ds,
                batch_sampler=ObjectGroupBatchSampler(proxy_eval_rows, batch_size=eval_batch_size, shuffle=False, seed=int(args.seed)),
                **eval_loader_kwargs(args),
            )
            train_eval_loader = DataLoader(
                proxy_train_ds,
                batch_sampler=ObjectGroupBatchSampler(proxy_train_rows, batch_size=eval_batch_size, shuffle=False, seed=int(args.seed)),
                **eval_loader_kwargs(args),
            )
            full_eval_loader = DataLoader(
                full_eval_ds,
                batch_sampler=ObjectGroupBatchSampler(full_eval_rows, batch_size=eval_batch_size, shuffle=False, seed=int(args.seed)),
                **eval_loader_kwargs(args),
            )
        else:
            eval_loader = DataLoader(
                eval_ds,
                batch_size=eval_batch_size,
                shuffle=False,
                **eval_loader_kwargs(args),
                drop_last=False,
            )
            train_eval_loader = DataLoader(
                proxy_train_ds,
                batch_size=eval_batch_size,
                shuffle=False,
                **eval_loader_kwargs(args),
                drop_last=False,
            )
            full_eval_loader = DataLoader(
                full_eval_ds,
                batch_size=eval_batch_size,
                shuffle=False,
                **eval_loader_kwargs(args),
                drop_last=False,
            )

    if args.voxel_corrupt_visualize_dir is not None:
        if is_rank0:
            write_voxel_corrupt_visuals(train_loader, args=args, device=device)
        if distributed:
            dist.barrier()
        if bool(args.voxel_corrupt_visualize_only):
            if distributed:
                dist.destroy_process_group()
            return 0

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
        use_body_prompt=bool(args.joint_seg),
        negative_prompt_channel=bool(args.negative_prompt_channel),
        use_checkpoint=bool(args.use_checkpoint),
    ).to(device)
    route_freeze_meta = freeze_unused_route_parameters(
        model,
        route=str(args.route),
        joint_seg=bool(args.joint_seg),
        is_rank0=is_rank0,
    )
    if bool(args.compile):
        model = torch.compile(model)
    encoder = None
    decoder = None
    frozen_encoder = 0
    frozen_decoder = 0
    if bool(args.joint_seg) and bool(args.use_packed_whole_occ):
        empty_code = torch.zeros((8, 16, 16, 16), dtype=torch.float32)
        if is_rank0:
            print(
                "[PromptSeg-ss-vae] skipped SS encoder/decoder load for joint packed-whole-occ training",
                flush=True,
            )
    else:
        encoder = load_ss_encoder(device=device, fp32=True)
        decoder = load_ss_decoder(device=device, fp32=True)
        frozen_encoder = sum(p.numel() for p in encoder.parameters() if not p.requires_grad)
        frozen_decoder = sum(p.numel() for p in decoder.parameters() if not p.requires_grad)
        if any(p.requires_grad for p in encoder.parameters()) or any(p.requires_grad for p in decoder.parameters()):
            raise RuntimeError("SS encoder/decoder must be frozen before promptable segmentation training")
        empty_code = compute_empty_code(encoder, device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))
    fp16_init_scale = float(os.environ.get("PROMPTSEG_FP16_INIT_SCALE", "128.0"))
    scaler = (
        torch.cuda.amp.GradScaler(
            enabled=precision == "fp16" and torch.cuda.is_available(),
            init_scale=fp16_init_scale,
        )
        if torch.cuda.is_available()
        else None
    )

    warm_start_meta: dict[str, Any] | None = None
    if args.warm_start is not None:
        loaded_empty, warm_start_meta = load_warm_start(args.warm_start, model, device=device, is_rank0=is_rank0)
        if loaded_empty is not None:
            empty_code = loaded_empty
    ddp_find_unused = bool(
        (str(args.route) == "voxel" and int(args.voxel_embedding_dim) > 0 and float(args.embed_loss_weight) <= 0)
        or bool(args.semantic_aux)
    )

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
        "ddp_find_unused_parameters": ddp_find_unused,
        "ddp_broadcast_buffers": not bool(args.joint_seg),
        "semantic_vocab": semantic_vocab,
        "metric_kind": "voxel_iou_proxy" if str(args.route) == "voxel" else "ss_decode_iou",
        "refine_mode": str(args.refine_mode),
        "spconv_depth": int(args.spconv_depth),
        "xpart_ce_weight": float(args.xpart_ce_weight),
        "joint_seg": bool(args.joint_seg),
        "body_mode": "auto-prompted-part-else-learned-token" if bool(args.joint_seg) else "none",
        "body_class_weight": float(args.body_class_weight),
        "joint_kmax": int(args.joint_kmax),
        "joint_small_part_threshold": int(args.joint_small_part_threshold),
        "joint_small_part_weight": float(args.joint_small_part_weight),
        "joint_smooth_weight": float(args.joint_smooth_weight),
        "joint_smooth_same_label_weight": float(args.joint_smooth_same_label_weight),
        "joint_smooth_all_label_weight": float(args.joint_smooth_all_label_weight),
        "joint_smooth_cross_label_weight": float(args.joint_smooth_cross_label_weight),
        "joint_smooth_neighborhood": int(args.joint_smooth_neighborhood),
        "joint_crf_eval": bool(args.joint_crf_eval),
        "joint_crf_iters": int(args.joint_crf_iters),
        "joint_crf_pairwise": float(args.joint_crf_pairwise),
        "joint_crf_neighborhood": int(args.joint_crf_neighborhood),
        "use_checkpoint": bool(args.use_checkpoint),
        "precision": precision,
        "autocast_dtype": "bfloat16" if precision == "bf16" else ("float16" if precision == "fp16" else "fp32"),
        "grad_scaler_enabled": bool(scaler is not None and scaler.is_enabled()),
        "grad_scaler_init_scale": fp16_init_scale if precision == "fp16" else 0.0,
        "dataloader": {
            "num_workers": int(args.num_workers),
            "prefetch_factor": int(args.prefetch_factor),
            "persistent_workers": bool(args.persistent_workers) and int(args.num_workers) > 0,
            "pin_memory": bool(args.pin_memory) and torch.cuda.is_available(),
            "group_cost_budget": int(args.group_cost_budget),
        },
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
        "boundary_band_radius": int(args.boundary_band_radius),
        "boundary_hard_mining": bool(args.boundary_hard_mining),
        "boundary_hard_mining_topk": float(args.boundary_hard_mining_topk),
        "boundary_hard_mining_weight": float(args.boundary_hard_mining_weight),
        "negative_prompt_channel": bool(args.negative_prompt_channel),
        "negative_prompt_equivalence_check": bool(args.negative_prompt_equivalence_check),
        "voxel_corrupt": bool(args.voxel_corrupt),
        "voxel_corrupt_drop_prob": float(args.voxel_corrupt_drop_prob),
        "voxel_corrupt_shell_prob": float(args.voxel_corrupt_shell_prob),
        "voxel_corrupt_speckle_prob": float(args.voxel_corrupt_speckle_prob),
        "auto_pack": bool(args.auto_pack),
        "base_packed_dir": str(args.base_packed_dir) if args.base_packed_dir is not None else None,
        "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
    }
    if distributed:
        model = DistributedDataParallel(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=ddp_find_unused,
            broadcast_buffers=not bool(args.joint_seg),
        )
    start_step = 0
    resume_path = args.resume
    if resume_path is None and bool(args.auto_resume):
        resume_path = latest_checkpoint_path(args.out_dir)
        if is_rank0:
            print(f"[PromptSeg-resume] auto_resume={bool(resume_path)} path={resume_path}", flush=True)
    if resume_path is not None:
        if args.resume is not None and not Path(resume_path).is_file():
            raise FileNotFoundError(f"--resume checkpoint does not exist: {resume_path}")
        try:
            start_step, empty_code = load_checkpoint(resume_path, model, optimizer, scaler, device)
        except Exception as exc:
            if args.resume is not None:
                raise RuntimeError(f"failed to load --resume checkpoint: {resume_path}") from exc
            raise
        if is_rank0:
            scaler_scale = (
                float(scaler.get_scale())
                if scaler is not None and scaler.is_enabled()
                else None
            )
            print(f"[PromptSeg-resume] loaded step={start_step} scaler_scale={scaler_scale}", flush=True)
    if is_rank0:
        (args.out_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        (args.out_dir / "train_rows.json").write_text(json.dumps([row.__dict__ for row in train_rows], indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        (args.out_dir / "proxy_train_rows.json").write_text(json.dumps([row.__dict__ for row in proxy_train_rows], indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        (args.out_dir / "proxy_eval_rows.json").write_text(json.dumps([row.__dict__ for row in proxy_eval_rows], indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        (args.out_dir / "full_eval_rows.json").write_text(json.dumps([row.__dict__ for row in full_eval_rows], indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(
            f"[PromptSeg-config] mode={args.mode} route={args.route} model_dim={args.dim} depth={args.depth} "
            f"resume={resume_path if resume_path is not None else 'none'} start_step={start_step} "
            f"batch_per_rank={args.batch_size} global_batch={int(args.batch_size) * world_size} lr={args.lr} "
            f"warmup={args.warmup_steps} steps={args.steps} split_json={args.split_json} proxy_json={args.proxy_json} "
            f"packed_dir={args.packed_dir} use_packed_whole_occ={args.use_packed_whole_occ} "
            f"refine_mode={args.refine_mode} spconv_depth={args.spconv_depth} "
            f"xpart_ce_weight={args.xpart_ce_weight} joint_seg={args.joint_seg} "
            f"body_mode={'auto-prompted-part-else-learned-token' if bool(args.joint_seg) else 'none'} "
            f"body_class_weight={args.body_class_weight} joint_kmax={args.joint_kmax} "
            f"joint_small_part_threshold={args.joint_small_part_threshold} "
            f"joint_small_part_weight={args.joint_small_part_weight} "
            f"joint_smooth_weight={args.joint_smooth_weight} "
            f"joint_smooth_same={args.joint_smooth_same_label_weight} "
            f"joint_smooth_all={args.joint_smooth_all_label_weight} "
            f"joint_smooth_cross={args.joint_smooth_cross_label_weight} "
            f"joint_smooth_n={args.joint_smooth_neighborhood} "
            f"joint_crf_eval={args.joint_crf_eval} "
            f"joint_crf_iters={args.joint_crf_iters} "
            f"joint_crf_pairwise={args.joint_crf_pairwise} "
            f"joint_crf_n={args.joint_crf_neighborhood} "
            f"voxel_max_tokens={args.voxel_max_tokens} use_checkpoint={args.use_checkpoint} "
            f"precision={precision} autocast_dtype={metadata['autocast_dtype']} "
            f"GradScaler enabled={metadata['grad_scaler_enabled']} init_scale={metadata['grad_scaler_init_scale']} "
            f"num_workers={args.num_workers} prefetch_factor={args.prefetch_factor} "
            f"persistent_workers={bool(args.persistent_workers) and int(args.num_workers) > 0} "
            f"pin_memory={bool(args.pin_memory) and torch.cuda.is_available()} "
            f"group_cost_budget={args.group_cost_budget} "
            f"motion_loss_weight={args.motion_loss_weight} "
            f"motion_loss_kind={args.motion_loss_kind} "
            f"infer_resolve={args.infer_resolve} "
            f"view_dropout={args.view_dropout} min_views={args.min_views} min_prompt_views={args.min_prompt_views} "
            f"view_dropout_start_step={args.view_dropout_start_step} small_oversample={args.small_oversample} "
            f"realappliance_oversample={oversampling_plan.get('realappliance_oversample', 1)} "
            f"realappliance_target_share={args.realappliance_target_share} "
            f"verse_focus_oversample={args.verse_focus_oversample} "
            f"focal_gamma={args.focal_gamma} boundary_weight={args.boundary_weight} "
            f"boundary_band_radius={args.boundary_band_radius} "
            f"boundary_hard_mining={args.boundary_hard_mining} "
            f"boundary_hard_mining_topk={args.boundary_hard_mining_topk} "
            f"boundary_hard_mining_weight={args.boundary_hard_mining_weight} "
            f"negative_prompt_channel={args.negative_prompt_channel} "
            f"negative_prompt_equivalence_check={args.negative_prompt_equivalence_check} "
            f"voxel_corrupt={args.voxel_corrupt} "
            f"voxel_corrupt_drop_prob={args.voxel_corrupt_drop_prob} "
            f"voxel_corrupt_shell_prob={args.voxel_corrupt_shell_prob} "
            f"voxel_corrupt_speckle_prob={args.voxel_corrupt_speckle_prob} "
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

    assert_negative_prompt_zero_equivalence(
        model,
        decoder,
        train_loader,
        empty_code,
        args=args,
        device=device,
        rank=rank,
    )

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
    nonfinite_grad_streak = 0
    max_nonfinite_grad_streak = 50
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
        grad_norm_value = 0.0
        optimizer_stepped = False
        grad_clip_max_norm = float(args.grad_clip) if float(args.grad_clip) > 0 else float("inf")
        if scaler is not None and scaler.is_enabled() and precision == "fp16":
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
                    boundary_band_radius=int(args.boundary_band_radius),
                    boundary_hard_mining=bool(args.boundary_hard_mining),
                    boundary_hard_mining_topk=float(args.boundary_hard_mining_topk),
                    boundary_hard_mining_weight=float(args.boundary_hard_mining_weight),
                    negative_prompt_channel=bool(args.negative_prompt_channel),
                    voxel_corrupt=bool(args.voxel_corrupt),
                    voxel_corrupt_drop_prob=float(args.voxel_corrupt_drop_prob),
                    voxel_corrupt_shell_prob=float(args.voxel_corrupt_shell_prob),
                    voxel_corrupt_speckle_prob=float(args.voxel_corrupt_speckle_prob),
                    semantic_loss_weight=float(args.semantic_loss_weight) if bool(args.semantic_aux) else 0.0,
                    use_packed_whole_occ=bool(args.use_packed_whole_occ),
                    joint_seg=bool(args.joint_seg),
                    body_class_weight=float(args.body_class_weight),
                    joint_kmax=int(args.joint_kmax),
                    joint_small_part_threshold=int(args.joint_small_part_threshold),
                    joint_small_part_weight=float(args.joint_small_part_weight),
                    joint_smooth_weight=float(args.joint_smooth_weight),
                    joint_smooth_same_label_weight=float(args.joint_smooth_same_label_weight),
                    joint_smooth_all_label_weight=float(args.joint_smooth_all_label_weight),
                    joint_smooth_cross_label_weight=float(args.joint_smooth_cross_label_weight),
                    joint_smooth_neighborhood=int(args.joint_smooth_neighborhood),
                    step=step,
                )
            if not torch.isfinite(loss.detach()).all():
                raise RuntimeError(f"non-finite loss at step={step} loss={float(loss.detach().float().item())}")
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_max_norm)
            grad_norm_value = float(grad_norm.detach().item() if torch.is_tensor(grad_norm) else grad_norm)
            grad_is_finite = bool(torch.isfinite(grad_norm.detach()).all()) if torch.is_tensor(grad_norm) else math.isfinite(float(grad_norm))
            if not grad_is_finite:
                nonfinite_grad_streak += 1
                sample_ids = _record_nonfinite_grad(
                    out_dir=args.out_dir,
                    step=step,
                    rank=rank,
                    grad_norm=grad_norm,
                    scaler=scaler,
                    loss=loss,
                    items=items,
                    batch=batch,
                    streak=nonfinite_grad_streak,
                )
                if nonfinite_grad_streak >= max_nonfinite_grad_streak:
                    raise RuntimeError(
                        f"non-finite grad_norm persisted for {nonfinite_grad_streak} consecutive steps "
                        f"at step={step}; recent_sample_ids={sample_ids}"
                    )
            else:
                nonfinite_grad_streak = 0
            scale_before = float(scaler.get_scale())
            scaler.step(optimizer)
            scaler.update()
            scale_after = float(scaler.get_scale())
            optimizer_stepped = grad_is_finite and scale_after >= scale_before
        else:
            with torch.cuda.amp.autocast(
                dtype=torch.bfloat16,
                enabled=precision == "bf16" and torch.cuda.is_available(),
            ):
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
                    boundary_band_radius=int(args.boundary_band_radius),
                    boundary_hard_mining=bool(args.boundary_hard_mining),
                    boundary_hard_mining_topk=float(args.boundary_hard_mining_topk),
                    boundary_hard_mining_weight=float(args.boundary_hard_mining_weight),
                    negative_prompt_channel=bool(args.negative_prompt_channel),
                    voxel_corrupt=bool(args.voxel_corrupt),
                    voxel_corrupt_drop_prob=float(args.voxel_corrupt_drop_prob),
                    voxel_corrupt_shell_prob=float(args.voxel_corrupt_shell_prob),
                    voxel_corrupt_speckle_prob=float(args.voxel_corrupt_speckle_prob),
                    semantic_loss_weight=float(args.semantic_loss_weight) if bool(args.semantic_aux) else 0.0,
                    use_packed_whole_occ=bool(args.use_packed_whole_occ),
                    joint_seg=bool(args.joint_seg),
                    body_class_weight=float(args.body_class_weight),
                    joint_kmax=int(args.joint_kmax),
                    joint_small_part_threshold=int(args.joint_small_part_threshold),
                    joint_small_part_weight=float(args.joint_small_part_weight),
                    joint_smooth_weight=float(args.joint_smooth_weight),
                    joint_smooth_same_label_weight=float(args.joint_smooth_same_label_weight),
                    joint_smooth_all_label_weight=float(args.joint_smooth_all_label_weight),
                    joint_smooth_cross_label_weight=float(args.joint_smooth_cross_label_weight),
                    joint_smooth_neighborhood=int(args.joint_smooth_neighborhood),
                    step=step,
                )
            if not torch.isfinite(loss.detach()).all():
                raise RuntimeError(f"non-finite loss at step={step} loss={float(loss.detach().float().item())}")
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_max_norm)
            grad_norm_value = float(grad_norm.detach().item() if torch.is_tensor(grad_norm) else grad_norm)
            grad_is_finite = bool(torch.isfinite(grad_norm.detach()).all()) if torch.is_tensor(grad_norm) else math.isfinite(float(grad_norm))
            if not grad_is_finite:
                nonfinite_grad_streak += 1
                sample_ids = _record_nonfinite_grad(
                    out_dir=args.out_dir,
                    step=step,
                    rank=rank,
                    grad_norm=grad_norm,
                    scaler=None,
                    loss=loss,
                    items=items,
                    batch=batch,
                    streak=nonfinite_grad_streak,
                )
                if nonfinite_grad_streak >= max_nonfinite_grad_streak:
                    raise RuntimeError(
                        f"non-finite grad_norm persisted for {nonfinite_grad_streak} consecutive steps "
                        f"at step={step}; recent_sample_ids={sample_ids}"
                    )
            else:
                nonfinite_grad_streak = 0
                optimizer.step()
                optimizer_stepped = True
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
            "grad_norm": float(grad_norm_value),
            "nonfinite_grad_streak": float(nonfinite_grad_streak),
            "optimizer_stepped": float(optimizer_stepped),
        })
        if bool(args.joint_seg):
            fwd_count_value = float(items.get("fwd_count", -1.0))
            if fwd_count_value != 1.0:
                raise RuntimeError(
                    f"fwd_count assertion failed at step={step}: expected 1.0, got {fwd_count_value}"
                )
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
                f"joint_ce {float(items.get('joint_ce', 0.0)):.4f} "
                f"joint_S {int(items.get('joint_s_eval_min', 0.0))}-{int(items.get('joint_s_eval_max', 0.0))} "
                f"fwd_count {float(items.get('fwd_count', 0.0)):.1f} "
                f"rank_step_std {float(items.get('joint_group_cost_std', 0.0)):.3f} "
                f"joint_body {float(items.get('joint_body_ratio', 0.0)):.4f} "
                f"joint_berr {float(items.get('joint_boundary_error', 0.0)):.4f} "
                f"joint_cross_same {float(items.get('joint_cross_label_same_pred_rate', 0.0)):.4f} "
                f"grad_norm {float(items.get('grad_norm', 0.0)):.4f} "
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
                negative_prompt_channel=bool(args.negative_prompt_channel),
                joint_seg=bool(args.joint_seg),
                body_class_weight=float(args.body_class_weight),
                joint_kmax=int(args.joint_kmax),
                joint_small_part_threshold=int(args.joint_small_part_threshold),
                joint_small_part_weight=float(args.joint_small_part_weight),
                joint_crf_eval=bool(args.joint_crf_eval),
                joint_crf_iters=int(args.joint_crf_iters),
                joint_crf_pairwise=float(args.joint_crf_pairwise),
                joint_crf_neighborhood=int(args.joint_crf_neighborhood),
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
                negative_prompt_channel=bool(args.negative_prompt_channel),
                joint_seg=bool(args.joint_seg),
                body_class_weight=float(args.body_class_weight),
                joint_kmax=int(args.joint_kmax),
                joint_small_part_threshold=int(args.joint_small_part_threshold),
                joint_small_part_weight=float(args.joint_small_part_weight),
                joint_crf_eval=bool(args.joint_crf_eval),
                joint_crf_iters=int(args.joint_crf_iters),
                joint_crf_pairwise=float(args.joint_crf_pairwise),
                joint_crf_neighborhood=int(args.joint_crf_neighborhood),
            )
            peak_gb = (
                float(torch.cuda.max_memory_allocated(torch.cuda.current_device()) / (1024 ** 3))
                if bool(args.joint_seg) and torch.cuda.is_available() and str(device).startswith("cuda")
                else measure_peak_memory(
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
                    negative_prompt_channel=bool(args.negative_prompt_channel),
                )
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
            joint_table_text = ""
            joint_crf_table_text = ""
            if bool(args.joint_seg):
                joint_table_text = joint_eval_table(
                    step=step,
                    lr=lr,
                    loss_total=float(items.get("total", float("nan"))),
                    loss_joint_ce=float(items.get("joint_ce", items.get("loss_jointCE", float("nan")))),
                    grad_norm=float(items.get("grad_norm", float("nan"))),
                    train_rows=train_result["rows"],
                    heldout_rows=heldout_result["rows"],
                )
                if bool(args.joint_crf_eval):
                    joint_crf_table_text = joint_eval_table(
                        step=step,
                        lr=lr,
                        loss_total=float(items.get("total", float("nan"))),
                        loss_joint_ce=float(items.get("joint_ce", items.get("loss_jointCE", float("nan")))),
                        grad_norm=float(items.get("grad_norm", float("nan"))),
                        train_rows=train_result["rows"],
                        heldout_rows=heldout_result["rows"],
                        metric_prefix="joint_crf",
                    )
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
            summary_text = table + "\n\n[RealAppliance heldout]\n" + realappliance_table + "\n"
            if joint_table_text:
                summary_text += "\n[Joint-body eval]\n" + joint_table_text + "\n"
                (eval_dir / "joint_eval_table.txt").write_text(joint_table_text + "\n", encoding="utf-8")
            if joint_crf_table_text:
                summary_text += "\n[Joint-CRF eval]\n" + joint_crf_table_text + "\n"
                (eval_dir / "joint_crf_eval_table.txt").write_text(joint_crf_table_text + "\n", encoding="utf-8")
            (eval_dir / "summary.txt").write_text(summary_text, encoding="utf-8")
            (eval_dir / "proxy_row.txt").write_text(proxy_metric_table(proxy_row) + "\n", encoding="utf-8")
            (eval_dir / "worst_train.txt").write_text(worst_train + "\n", encoding="utf-8")
            (eval_dir / "worst_heldout.txt").write_text(worst_heldout + "\n", encoding="utf-8")
            (eval_dir / "predcand_zero_evidence.txt").write_text(pred_zero + "\n", encoding="utf-8")
            joint_print = "\n[Joint-body eval]\n" + joint_table_text + "\n" if joint_table_text else ""
            joint_crf_print = "\n[Joint-CRF eval]\n" + joint_crf_table_text + "\n" if joint_crf_table_text else ""
            print(
                f"[PromptSeg-eval] step={step} metric={metadata['metric_kind']} full_eval={full_eval} peak_gb={peak_gb:.3f}\n"
                f"{proxy_metric_table(proxy_row)}\n\n{table}\n"
                f"{joint_print}{joint_crf_print}\n[RealAppliance heldout]\n{realappliance_table}\n\n[worst train]\n{worst_train}\n\n"
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
