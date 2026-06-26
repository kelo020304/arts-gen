"""Kinematic joint-head data/model/loss utilities.

This module is intentionally independent from part-seg training.  It consumes
GT part voxels and precomputed SS latents, and predicts object kinematic fields
from the frozen feature products already on disk.
"""

from __future__ import annotations

import json
import math
import os
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import Dataset


TYPE_TO_CLASS = {"B": 0, "C": 1, "E": 2}
CLASS_TO_TYPE = {0: "B", 1: "C", 2: "E"}
IGNORE_TYPE_CLASS = 3
BODY_PARENT_SLOT = 0
MAX_PARTS_DEFAULT = 32
DEFAULT_DATA_ROOTS = (
    "/mnt/robot-data-lab/jzh/art-gen/data/phyx-verse",
    "/mnt/robot-data-lab/jzh/art-gen/data/realappliance",
)
DEFAULT_SPLIT_JSON = "/mnt/robot-data-lab/jzh/art-gen-output/part_promptable_seg/manifests/split_official_verse_realappliance_0511dd_v4.json"
Y_UP_TO_Z_UP_3 = np.asarray(
    [
        [1.0, 0.0, 0.0],
        [0.0, 0.0, -1.0],
        [0.0, 1.0, 0.0],
    ],
    dtype=np.float64,
)
Y_UP_TO_Z_UP_4 = np.eye(4, dtype=np.float64)
Y_UP_TO_Z_UP_4[:3, :3] = Y_UP_TO_Z_UP_3
VERSE_Y_UP_DATASET_IDS = frozenset({"phyx-verse", "physx-mobility", "physx-0511-drawer-door"})


@dataclass(frozen=True)
class JointObjectRef:
    dataset_id: str
    data_root: str
    object_id: str


@dataclass(frozen=True)
class JointPartMeta:
    name: str
    label: int
    raw_label: int
    part_index: int
    joint_group_id: str
    parent_group: str | None
    joint_type: str
    joint_params: tuple[float, ...]
    ignored: bool
    ignore_reason: str
    axis: tuple[float, float, float]
    pivot: tuple[float, float, float]
    limits: tuple[float, float]


def seed_all(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise TypeError(f"{path} expected JSON object, got {type(value).__name__}")
    return value


def _object_key(dataset_id: str, object_id: str) -> str:
    return f"{dataset_id}::{object_id}" if dataset_id else str(object_id)


def infer_dataset_id(data_root: str | Path | None, dataset_id: str | None = None) -> str:
    if dataset_id:
        return str(dataset_id)
    if data_root is None:
        return ""
    return Path(data_root).name


def needs_y_up_joint_fix(dataset_id: str | None = None, data_root: str | Path | None = None) -> bool:
    did = infer_dataset_id(data_root, dataset_id).lower()
    return did in VERSE_Y_UP_DATASET_IDS


def joint_frame_rotation(dataset_id: str | None = None, data_root: str | Path | None = None) -> np.ndarray:
    if needs_y_up_joint_fix(dataset_id=dataset_id, data_root=data_root):
        return Y_UP_TO_Z_UP_3.copy()
    return np.eye(3, dtype=np.float64)


def transform_joint_params_to_dataset_frame(
    params: tuple[float, ...],
    *,
    dataset_id: str | None = None,
    data_root: str | Path | None = None,
) -> tuple[float, ...]:
    if len(params) != 8 or not needs_y_up_joint_fix(dataset_id=dataset_id, data_root=data_root):
        return params
    rot = Y_UP_TO_Z_UP_3
    axis = rot @ np.asarray(params[:3], dtype=np.float64)
    pivot = rot @ np.asarray(params[3:6], dtype=np.float64)
    return tuple(float(x) for x in (*axis.tolist(), *pivot.tolist(), params[6], params[7]))


def canonicalize_joint_axis_sign(params: tuple[float, ...]) -> tuple[float, ...]:
    if len(params) != 8:
        return params
    lo = float(params[6])
    hi = float(params[7])
    if hi >= -lo:
        return params
    axis = [-float(x) for x in params[:3]]
    return tuple(float(x) for x in (*axis, *params[3:6], -hi, -lo))


def transform_joint_matrix_to_dataset_frame(
    matrix: np.ndarray,
    *,
    dataset_id: str | None = None,
    data_root: str | Path | None = None,
) -> np.ndarray:
    mat = np.asarray(matrix, dtype=np.float64)
    if mat.shape != (4, 4):
        raise ValueError(f"joint transform matrix must be 4x4, got {mat.shape}")
    if not needs_y_up_joint_fix(dataset_id=dataset_id, data_root=data_root):
        return mat.copy()
    return Y_UP_TO_Z_UP_4 @ mat @ np.linalg.inv(Y_UP_TO_Z_UP_4)


def _split_dataset_roots(split: Mapping[str, Any]) -> dict[str, str]:
    roots: dict[str, str] = {}
    for item in split.get("datasets", []) or []:
        if not isinstance(item, Mapping):
            continue
        did = str(item.get("dataset_id") or "")
        root = item.get("data_root")
        if did and root:
            roots[did] = str(root)
    return roots


def refs_from_split(
    split_json: str | Path = DEFAULT_SPLIT_JSON,
    *,
    dataset_ids: Iterable[str] = ("phyx-verse", "realappliance"),
) -> tuple[list[JointObjectRef], list[JointObjectRef]]:
    path = Path(split_json)
    if not path.is_file():
        raise FileNotFoundError(path)
    split = _read_json(path)
    roots = _split_dataset_roots(split)
    allowed = {str(x) for x in dataset_ids}

    def parse(keys: Iterable[Any]) -> list[JointObjectRef]:
        out: list[JointObjectRef] = []
        for raw in keys:
            text = str(raw)
            if "::" not in text:
                continue
            dataset_id, object_id = text.split("::", 1)
            if dataset_id not in allowed:
                continue
            root = roots.get(dataset_id)
            if root is None:
                raise KeyError(f"split dataset root missing for dataset_id={dataset_id!r}")
            out.append(JointObjectRef(dataset_id=dataset_id, data_root=root, object_id=object_id))
        return out

    train = parse(split.get("train_keys", split.get("train_ids", [])))
    held = parse(split.get("heldout_keys", split.get("heldout_ids", [])))
    if not train:
        raise RuntimeError(f"{path} produced zero train refs for {sorted(allowed)}")
    if not held:
        raise RuntimeError(f"{path} produced zero heldout refs for {sorted(allowed)}")
    return train, held


def refs_from_roots(
    roots: Iterable[str | Path] = DEFAULT_DATA_ROOTS,
    *,
    heldout_fraction: float = 0.12,
    seed: int = 20260617,
) -> tuple[list[JointObjectRef], list[JointObjectRef]]:
    refs: list[JointObjectRef] = []
    for root in roots:
        root_path = Path(root)
        dataset_id = root_path.name
        base = root_path / "reconstruction" / "part_info"
        if not base.is_dir():
            raise FileNotFoundError(base)
        for obj_dir in sorted(p for p in base.iterdir() if p.is_dir()):
            if (obj_dir / "part_info.json").is_file():
                refs.append(JointObjectRef(dataset_id=dataset_id, data_root=str(root_path), object_id=obj_dir.name))
    if not refs:
        raise RuntimeError("refs_from_roots produced zero refs")
    rng = random.Random(int(seed))
    rng.shuffle(refs)
    held_count = max(1, int(round(len(refs) * float(heldout_fraction))))
    held = refs[:held_count]
    train = refs[held_count:]
    return train, held


def load_part_info(data_root: str | Path, object_id: str) -> dict[str, Any]:
    path = Path(data_root) / "reconstruction" / "part_info" / str(object_id) / "part_info.json"
    if not path.is_file():
        raise FileNotFoundError(path)
    return _read_json(path)


def _axis_normalize(values: Iterable[Any]) -> tuple[tuple[float, float, float], float]:
    vals = [float(x) for x in values]
    if len(vals) != 3:
        return (0.0, 0.0, 0.0), 0.0
    norm = math.sqrt(sum(v * v for v in vals))
    if norm <= 0.0:
        return (0.0, 0.0, 0.0), norm
    return tuple(float(v / norm) for v in vals), norm  # type: ignore[return-value]


def parse_joint_part(
    name: str,
    info: Mapping[str, Any],
    *,
    object_id: str,
    dataset_id: str | None = None,
    data_root: str | Path | None = None,
) -> JointPartMeta:
    joint_type = str(info.get("joint_type", "") or "")
    raw_params = info.get("joint_params") or []
    params = tuple(float(x) for x in raw_params) if isinstance(raw_params, list) else tuple()
    ignored = False
    reasons: list[str] = []
    axis = (0.0, 0.0, 0.0)
    pivot = (0.0, 0.0, 0.0)
    limits = (0.0, 0.0)

    if joint_type == "CB":
        ignored = True
        reasons.append("legacy_CB")
    elif joint_type not in TYPE_TO_CLASS:
        ignored = True
        reasons.append(f"unsupported_type:{joint_type or '<empty>'}")

    if joint_type == "E":
        if params:
            reasons.append("fixed_params_present")
        params = tuple()
    elif joint_type in ("B", "C"):
        if len(params) != 8:
            ignored = True
            reasons.append(f"params_len:{len(params)}")
        else:
            params = transform_joint_params_to_dataset_frame(params, dataset_id=dataset_id, data_root=data_root)
            params = canonicalize_joint_axis_sign(params)
            axis, norm = _axis_normalize(params[:3])
            if norm < 1.0e-6:
                ignored = True
                reasons.append("degenerate_axis")
            pivot = tuple(float(x) for x in params[3:6])  # type: ignore[assignment]
            limits = (float(params[6]), float(params[7]))
            if object_id == "052" and limits[0] > limits[1]:
                ignored = True
                reasons.append("ra_052_limit_lo_gt_hi")
    elif len(params) not in (0, 8, 16):
        reasons.append(f"unexpected_params_len:{len(params)}")

    return JointPartMeta(
        name=str(name),
        label=int(info.get("label", -1)),
        raw_label=int(info.get("raw_label", -1)),
        part_index=int(info.get("part_index", -1)),
        joint_group_id=str(info.get("joint_group_id", "")),
        parent_group=None if info.get("parent_group") is None else str(info.get("parent_group")),
        joint_type=joint_type,
        joint_params=params,
        ignored=bool(ignored),
        ignore_reason=";".join(reasons),
        axis=axis,
        pivot=pivot,
        limits=limits,
    )


def parse_object_parts(
    data_root: str | Path,
    object_id: str,
    *,
    dataset_id: str | None = None,
) -> list[JointPartMeta]:
    data = load_part_info(data_root, object_id)
    parts = data.get("parts")
    if not isinstance(parts, Mapping):
        raise TypeError(f"{data_root}/{object_id} part_info['parts'] must be a mapping")
    resolved_dataset_id = infer_dataset_id(data_root, dataset_id)
    metas = [
        parse_joint_part(
            str(name),
            info,
            object_id=str(object_id),
            dataset_id=resolved_dataset_id,
            data_root=data_root,
        )
        for name, info in parts.items()
    ]
    metas.sort(key=lambda item: (int(item.part_index), item.name))
    return metas


def parent_slot_for_part(meta: JointPartMeta, parts: list[JointPartMeta], name_to_idx: Mapping[str, int]) -> int:
    if meta.parent_group is None:
        return BODY_PARENT_SLOT
    parent_group = str(meta.parent_group)
    candidates = [p for p in parts if str(p.joint_group_id) == parent_group]
    body_candidates = [p for p in candidates if p.joint_type == "E" and p.parent_group is None]
    if body_candidates:
        return BODY_PARENT_SLOT
    candidates = [p for p in candidates if p.name != meta.name]
    if not candidates:
        return BODY_PARENT_SLOT
    candidates.sort(key=lambda p: (0 if not p.ignored else 1, p.part_index, p.name))
    parent = candidates[0]
    return int(name_to_idx[parent.name]) + 1


def _load_dense_latent(path: Path) -> torch.Tensor:
    if not path.is_file():
        raise FileNotFoundError(path)
    if path.suffix == ".npz":
        data = np.load(path)
        if "mean" not in data.files:
            raise KeyError(f"{path} expected key 'mean', found {data.files}")
        arr = data["mean"]
    else:
        arr = np.load(path)
    tensor = torch.from_numpy(np.asarray(arr)).float()
    if tuple(tensor.shape) != (8, 16, 16, 16):
        raise ValueError(f"{path} expected latent shape (8,16,16,16), got {tuple(tensor.shape)}")
    return tensor


def _load_raw_coords(path: Path) -> torch.Tensor:
    if not path.is_file():
        raise FileNotFoundError(path)
    coords = torch.from_numpy(np.asarray(np.load(path))).long()
    if coords.dim() != 2 or coords.shape[1] != 3:
        raise ValueError(f"{path} expected [N,3], got {tuple(coords.shape)}")
    return coords


def _load_voxel_points_np(path: Path) -> np.ndarray:
    arr = np.asarray(np.load(path))
    if arr.ndim == 2 and arr.shape[1] >= 3:
        return arr[:, :3].astype(np.float64, copy=False)
    return np.argwhere(arr > 0).astype(np.float64, copy=False)


def _angle_error_deg_abs(a: np.ndarray, b: np.ndarray) -> float:
    va = np.asarray(a, dtype=np.float64)
    vb = np.asarray(b, dtype=np.float64)
    na = float(np.linalg.norm(va))
    nb = float(np.linalg.norm(vb))
    if na <= 1.0e-9 or nb <= 1.0e-9:
        return float("nan")
    cos = abs(float(np.clip(np.dot(va / na, vb / nb), -1.0, 1.0)))
    return float(math.degrees(math.acos(cos)))


def prismatic_axis_voxel_sanity(
    refs: Iterable[JointObjectRef],
    *,
    sample_parts: int = 16,
    max_error_deg: float = 5.0,
    min_motion_voxels: float = 0.5,
    name_keywords: Iterable[str] | None = ("drawer",),
) -> dict[str, Any]:
    checked: list[dict[str, Any]] = []
    skipped = Counter()
    keywords = tuple(str(x).lower() for x in (name_keywords or ()))
    for ref in refs:
        if not needs_y_up_joint_fix(dataset_id=ref.dataset_id, data_root=ref.data_root):
            continue
        if len(checked) >= int(sample_parts):
            break
        root = Path(ref.data_root)
        voxel_root = root / "reconstruction" / "voxel_expanded" / ref.object_id
        if not voxel_root.is_dir():
            skipped["missing_voxel_root"] += 1
            continue
        angle_dirs = sorted(
            [p for p in voxel_root.iterdir() if p.is_dir() and p.name.startswith("angle_")],
            key=lambda p: int(p.name.rsplit("_", 1)[-1]) if p.name.rsplit("_", 1)[-1].isdigit() else 10**9,
        )
        if len(angle_dirs) < 2:
            skipped["lt_two_angles"] += 1
            continue
        try:
            parts = parse_object_parts(ref.data_root, ref.object_id, dataset_id=ref.dataset_id)
        except Exception as exc:
            skipped[f"part_info:{type(exc).__name__}"] += 1
            continue
        for part in parts:
            if len(checked) >= int(sample_parts):
                break
            if part.ignored or part.joint_type != "B":
                continue
            if keywords and not any(key in part.name.lower() for key in keywords):
                skipped["keyword_filter"] += 1
                continue
            base_path = angle_dirs[0] / "64" / f"ind_{part.name}.npy"
            if not base_path.is_file():
                skipped["missing_base_part_voxel"] += 1
                continue
            pts0 = _load_voxel_points_np(base_path)
            if pts0.size == 0:
                skipped["empty_base_part_voxel"] += 1
                continue
            c0 = pts0.mean(axis=0)
            best: tuple[float, np.ndarray, str] | None = None
            for angle_dir in angle_dirs[1:]:
                path = angle_dir / "64" / f"ind_{part.name}.npy"
                if not path.is_file():
                    continue
                pts = _load_voxel_points_np(path)
                if pts.size == 0:
                    continue
                disp = pts.mean(axis=0) - c0
                dist = float(np.linalg.norm(disp))
                if best is None or dist > best[0]:
                    best = (dist, disp, angle_dir.name)
            if best is None or best[0] < float(min_motion_voxels):
                skipped["low_motion"] += 1
                continue
            err = _angle_error_deg_abs(np.asarray(part.axis, dtype=np.float64), best[1])
            checked.append(
                {
                    "dataset_id": ref.dataset_id,
                    "object_id": ref.object_id,
                    "part_name": part.name,
                    "axis": [round(float(x), 6) for x in part.axis],
                    "motion_axis": [round(float(x), 6) for x in (best[1] / max(best[0], 1.0e-9)).tolist()],
                    "angle": best[2],
                    "motion_voxels": round(best[0], 6),
                    "axis_err_deg": round(float(err), 6),
                }
            )
    if not checked:
        raise RuntimeError(f"verse prismatic axis sanity found zero checkable parts; skipped={dict(skipped)}")
    worst = max(float(row["axis_err_deg"]) for row in checked)
    mean_err = sum(float(row["axis_err_deg"]) for row in checked) / len(checked)
    summary = {
        "checked": len(checked),
        "max_error_deg": worst,
        "mean_error_deg": mean_err,
        "threshold_deg": float(max_error_deg),
        "examples": checked[: min(5, len(checked))],
        "skipped": dict(skipped),
    }
    if worst > float(max_error_deg):
        raise AssertionError(f"verse prismatic axis sanity failed: {json_dumps(summary)}")
    return summary


def coords_to_mask64(coords: torch.Tensor) -> torch.Tensor:
    mask = torch.zeros((64, 64, 64), dtype=torch.bool)
    coords = coords.long()
    if coords.numel() > 0:
        coords = coords.clamp(0, 63)
        mask[coords[:, 0], coords[:, 1], coords[:, 2]] = True
    return mask


def coords_to_mask16(coords: torch.Tensor) -> torch.Tensor:
    mask = torch.zeros((16, 16, 16), dtype=torch.bool)
    coords = coords.long()
    if coords.numel() > 0:
        latent = torch.div(coords.clamp(0, 63), 4, rounding_mode="floor")
        mask[latent[:, 0], latent[:, 1], latent[:, 2]] = True
    return mask


def mask_centroid_extent(coords: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if coords.numel() == 0:
        return torch.zeros(3, dtype=torch.float32), torch.zeros(3, dtype=torch.float32)
    coords_f = coords.float() / 63.0
    lo = coords_f.min(dim=0).values
    hi = coords_f.max(dim=0).values
    return (lo + hi) * 0.5, (hi - lo).clamp_min(0.0)


def masked_pool_latent(latent: torch.Tensor, mask16: torch.Tensor) -> torch.Tensor:
    if mask16.any():
        values = latent[:, mask16.bool()]
        return values.mean(dim=1)
    return latent.flatten(1).mean(dim=1)


def pivot_to_grid_index(pivot: torch.Tensor, resolution: int = 16) -> torch.Tensor:
    # Canonical part_info pivots are already in the normalized scene frame.
    # Map [-0.5, 0.5] to [0, resolution-1].  Values outside are clamped, and
    # the dataset reports that rate for audit.
    grid = torch.round((pivot + 0.5) * float(resolution - 1)).long()
    return grid.clamp(0, resolution - 1)


def pivot_to_offset(pivot: torch.Tensor, resolution: int = 16) -> torch.Tensor:
    idx = pivot_to_grid_index(pivot, resolution=resolution).float()
    center = idx / float(resolution - 1) - 0.5
    return pivot - center


def pivot_in_bounds(pivot: Iterable[float]) -> bool:
    return all(-0.5 <= float(v) <= 0.5 for v in pivot)


class KinematicDataset(Dataset):
    def __init__(
        self,
        refs: Iterable[JointObjectRef],
        *,
        max_parts: int = MAX_PARTS_DEFAULT,
        include_ignored_parts: bool = True,
        limit_objects: int = 0,
    ) -> None:
        self.refs = list(refs)
        if int(limit_objects) > 0:
            self.refs = self.refs[: int(limit_objects)]
        self.max_parts = int(max_parts)
        self.include_ignored_parts = bool(include_ignored_parts)
        self.items: list[JointObjectRef] = []
        self.skip_reasons: Counter[str] = Counter()
        self.type_counts: Counter[str] = Counter()
        self.supervised_type_counts: Counter[str] = Counter()
        self.parent_nested_count = 0
        self.pivot_oob_count = 0
        self._build_index()

    def _build_index(self) -> None:
        for ref in self.refs:
            try:
                parts = parse_object_parts(ref.data_root, ref.object_id, dataset_id=ref.dataset_id)
            except Exception as exc:
                self.skip_reasons[f"part_info:{type(exc).__name__}"] += 1
                continue
            if not self.include_ignored_parts:
                parts = [p for p in parts if not p.ignored]
            if not parts:
                self.skip_reasons["zero_parts"] += 1
                continue
            if len(parts) > self.max_parts:
                self.skip_reasons["too_many_parts"] += 1
                continue
            missing = False
            root = Path(ref.data_root)
            for part in parts:
                if not (root / "reconstruction" / "voxel_expanded" / ref.object_id / "angle_0" / "64" / f"ind_{part.name}.npy").is_file():
                    missing = True
                    self.skip_reasons["missing_voxel"] += 1
                    break
                if not (root / "reconstruction" / "ss_latents_per_part" / ref.object_id / "angle_0" / f"{part.name}.npy").is_file():
                    missing = True
                    self.skip_reasons["missing_part_latent"] += 1
                    break
            if missing:
                continue
            z_path = root / "reconstruction" / "ss_latents_expanded" / ref.object_id / "angle_0" / "latent.npz"
            if not z_path.is_file():
                self.skip_reasons["missing_global_latent"] += 1
                continue
            self.items.append(ref)
            root_groups = {p.joint_group_id for p in parts if p.parent_group is None}
            for part in parts:
                self.type_counts[part.joint_type or "<empty>"] += 1
                if not part.ignored:
                    self.supervised_type_counts[part.joint_type] += 1
                if part.parent_group is not None and str(part.parent_group) not in root_groups:
                    self.parent_nested_count += 1
                if part.joint_type == "C" and not part.ignored and not pivot_in_bounds(part.pivot):
                    self.pivot_oob_count += 1
        if not self.items:
            raise RuntimeError(f"KinematicDataset produced zero usable objects; skips={dict(self.skip_reasons)}")

    def summary(self) -> dict[str, Any]:
        return {
            "objects": len(self.items),
            "refs": len(self.refs),
            "max_parts": self.max_parts,
            "type_counts": dict(self.type_counts),
            "supervised_type_counts": dict(self.supervised_type_counts),
            "skip_reasons": dict(self.skip_reasons),
            "nested_parent_edges": int(self.parent_nested_count),
            "pivot_oob_revolute": int(self.pivot_oob_count),
        }

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        ref = self.items[idx]
        root = Path(ref.data_root)
        parts = parse_object_parts(root, ref.object_id, dataset_id=ref.dataset_id)
        if not self.include_ignored_parts:
            parts = [p for p in parts if not p.ignored]
        parts = parts[: self.max_parts]
        name_to_idx = {p.name: i for i, p in enumerate(parts)}
        k = len(parts)

        z_global = _load_dense_latent(
            root / "reconstruction" / "ss_latents_expanded" / ref.object_id / "angle_0" / "latent.npz"
        )
        global_pool = z_global.flatten(1).mean(dim=1)

        x_part = torch.zeros((self.max_parts, 8, 16, 16, 16), dtype=torch.float32)
        part_mask16 = torch.zeros((self.max_parts, 16, 16, 16), dtype=torch.bool)
        pooled = torch.zeros((self.max_parts, 8), dtype=torch.float32)
        geom = torch.zeros((self.max_parts, 6), dtype=torch.float32)
        type_target = torch.full((self.max_parts,), IGNORE_TYPE_CLASS, dtype=torch.long)
        axis = torch.zeros((self.max_parts, 3), dtype=torch.float32)
        pivot = torch.zeros((self.max_parts, 3), dtype=torch.float32)
        pivot_index = torch.zeros((self.max_parts, 3), dtype=torch.long)
        pivot_offset = torch.zeros((self.max_parts, 3), dtype=torch.float32)
        limits = torch.zeros((self.max_parts, 2), dtype=torch.float32)
        parent = torch.zeros((self.max_parts,), dtype=torch.long)
        valid = torch.zeros((self.max_parts,), dtype=torch.bool)
        ignored = torch.ones((self.max_parts,), dtype=torch.bool)
        raw_counts = torch.zeros((self.max_parts,), dtype=torch.float32)

        part_names: list[str] = []
        ignore_reasons: list[str] = []
        joint_types: list[str] = []
        for part_idx, part in enumerate(parts):
            valid[part_idx] = True
            part_names.append(part.name)
            ignore_reasons.append(part.ignore_reason)
            joint_types.append(part.joint_type)
            latent = _load_dense_latent(
                root / "reconstruction" / "ss_latents_per_part" / ref.object_id / "angle_0" / f"{part.name}.npy"
            )
            coords = _load_raw_coords(
                root / "reconstruction" / "voxel_expanded" / ref.object_id / "angle_0" / "64" / f"ind_{part.name}.npy"
            )
            mask16 = coords_to_mask16(coords)
            centroid, extent = mask_centroid_extent(coords)
            x_part[part_idx] = latent
            part_mask16[part_idx] = mask16
            pooled[part_idx] = masked_pool_latent(z_global, mask16)
            geom[part_idx, :3] = centroid
            geom[part_idx, 3:] = extent
            raw_counts[part_idx] = float(coords.shape[0])
            ignored[part_idx] = bool(part.ignored)
            if not part.ignored and part.joint_type in TYPE_TO_CLASS:
                type_target[part_idx] = TYPE_TO_CLASS[part.joint_type]
            if part.joint_type in ("B", "C") and len(part.joint_params) == 8:
                axis[part_idx] = torch.tensor(part.axis, dtype=torch.float32)
                pivot[part_idx] = torch.tensor(part.pivot, dtype=torch.float32)
                pivot_index[part_idx] = pivot_to_grid_index(pivot[part_idx])
                pivot_offset[part_idx] = pivot_to_offset(pivot[part_idx])
                limits[part_idx] = torch.tensor(part.limits, dtype=torch.float32)
            parent[part_idx] = parent_slot_for_part(part, parts, name_to_idx)

        return {
            "z_global": z_global,
            "global_pool": global_pool,
            "x_part": x_part,
            "part_mask16": part_mask16,
            "masked_pool": pooled,
            "geom": geom,
            "type_target": type_target,
            "axis": axis,
            "pivot": pivot,
            "pivot_index": pivot_index,
            "pivot_offset": pivot_offset,
            "limits": limits,
            "parent": parent,
            "part_valid": valid,
            "ignored": ignored,
            "raw_counts": raw_counts,
            "obj_id": ref.object_id,
            "dataset_id": ref.dataset_id,
            "part_names": part_names,
            "joint_types": joint_types,
            "ignore_reasons": ignore_reasons,
        }


def collate_kinematic(batch: list[dict[str, Any]]) -> dict[str, Any]:
    tensor_keys = (
        "z_global",
        "global_pool",
        "x_part",
        "part_mask16",
        "masked_pool",
        "geom",
        "type_target",
        "axis",
        "pivot",
        "pivot_index",
        "pivot_offset",
        "limits",
        "parent",
        "part_valid",
        "ignored",
        "raw_counts",
    )
    out = {key: torch.stack([item[key] for item in batch], dim=0) for key in tensor_keys}
    for key in ("obj_id", "dataset_id", "part_names", "joint_types", "ignore_reasons"):
        out[key] = [item[key] for item in batch]
    return out


def _anchor_vectors(mode: str = "6") -> torch.Tensor:
    if str(mode) == "6":
        vecs = [
            (1, 0, 0),
            (-1, 0, 0),
            (0, 1, 0),
            (0, -1, 0),
            (0, 0, 1),
            (0, 0, -1),
        ]
    elif str(mode) == "26":
        vecs = []
        for x in (-1, 0, 1):
            for y in (-1, 0, 1):
                for z in (-1, 0, 1):
                    if x == y == z == 0:
                        continue
                    vecs.append((x, y, z))
    else:
        raise ValueError(f"axis_anchor_mode must be '6' or '26', got {mode!r}")
    tensor = torch.tensor(vecs, dtype=torch.float32)
    return F.normalize(tensor, dim=-1)


class JointHeadNet(nn.Module):
    def __init__(
        self,
        *,
        max_parts: int = MAX_PARTS_DEFAULT,
        dim: int = 256,
        axis_anchor_mode: str = "6",
        type_embed_dim: int = 32,
        num_heads: int = 8,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.max_parts = int(max_parts)
        self.dim = int(dim)
        self.dropout_p = float(dropout)
        self.register_buffer("axis_anchors", _anchor_vectors(axis_anchor_mode), persistent=False)
        input_dim = 8 + 8 + 8 + 6
        self.token_mlp = nn.Sequential(
            nn.Linear(input_dim, dim),
            nn.LayerNorm(dim),
            nn.GELU(),
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.GELU(),
        )
        self.global_proj = nn.Linear(8, dim)
        self.global_token_proj = nn.Linear(8, dim)
        self.cross_attn = nn.MultiheadAttention(dim, int(num_heads), batch_first=True)
        self.post = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim), nn.GELU())
        self.dropout = nn.Dropout(p=max(0.0, min(1.0, float(dropout))))
        self.type_head = nn.Linear(dim, 4)
        self.type_embed = nn.Embedding(4, int(type_embed_dim))
        branch_dim = dim + int(type_embed_dim)
        self.axis_anchor_head = nn.Linear(branch_dim, int(self.axis_anchors.shape[0]))
        self.axis_residual_head = nn.Linear(branch_dim, 3)
        self.pivot_heatmap_head = nn.Linear(branch_dim, 16 * 16 * 16)
        self.pivot_offset_head = nn.Linear(branch_dim, 3)
        self.limit_head = nn.Linear(branch_dim, 2)
        self.parent_head = nn.Linear(branch_dim, self.max_parts + 1)

    def _condition_type(self, type_logits: torch.Tensor, type_target: torch.Tensor | None, teacher_force: bool) -> torch.Tensor:
        if teacher_force and type_target is not None:
            ids = type_target.clamp(0, IGNORE_TYPE_CLASS)
        else:
            ids = type_logits.argmax(dim=-1).clamp(0, IGNORE_TYPE_CLASS)
        return self.type_embed(ids.long())

    def forward(
        self,
        z_global: torch.Tensor,
        masked_pool: torch.Tensor,
        x_part: torch.Tensor,
        geom: torch.Tensor,
        part_valid: torch.Tensor,
        type_target: torch.Tensor | None = None,
        *,
        teacher_force_type: bool = True,
    ) -> dict[str, torch.Tensor]:
        bsz, max_parts = part_valid.shape
        part_lat = x_part.flatten(3).mean(dim=-1)
        global_pool = z_global.flatten(2).mean(dim=-1)
        global_feat = self.global_proj(global_pool).unsqueeze(1).expand(-1, max_parts, -1)
        token_in = torch.cat([masked_pool, part_lat, global_pool.unsqueeze(1).expand(-1, max_parts, -1), geom], dim=-1)
        tokens = self.dropout(self.token_mlp(token_in) + global_feat)

        z_tokens = z_global.flatten(2).transpose(1, 2)
        memory = self.global_token_proj(z_tokens)
        key_padding = ~part_valid.bool()
        attn_out, _ = self.cross_attn(tokens, memory, memory, need_weights=False)
        tokens = self.dropout(self.post(tokens + attn_out))
        tokens = tokens.masked_fill(key_padding.unsqueeze(-1), 0.0)

        type_logits = self.type_head(tokens)
        type_emb = self._condition_type(type_logits, type_target, teacher_force_type)
        branch = self.dropout(torch.cat([tokens, type_emb], dim=-1))
        return {
            "type_logits": type_logits,
            "axis_anchor_logits": self.axis_anchor_head(branch),
            "axis_residual": self.axis_residual_head(branch),
            "pivot_heatmap_logits": self.pivot_heatmap_head(branch),
            "pivot_offset": self.pivot_offset_head(branch),
            "limits": self.limit_head(branch),
            "parent_logits": self.parent_head(branch),
        }

    def predict_axis(self, out: Mapping[str, torch.Tensor]) -> torch.Tensor:
        anchor_idx = out["axis_anchor_logits"].argmax(dim=-1)
        anchors = self.axis_anchors.to(device=anchor_idx.device)[anchor_idx]
        return F.normalize(anchors + out["axis_residual"], dim=-1)


def axis_anchor_targets(axis: torch.Tensor, anchors: torch.Tensor) -> torch.Tensor:
    anchors = anchors.to(device=axis.device, dtype=axis.dtype)
    sim = torch.einsum("...d,ad->...a", F.normalize(axis, dim=-1), anchors)
    return sim.argmax(dim=-1)


def joint_head_loss(
    model: JointHeadNet,
    out: Mapping[str, torch.Tensor],
    batch: Mapping[str, torch.Tensor],
    *,
    weights: Mapping[str, float] | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    weights = dict(weights or {})
    part_valid = batch["part_valid"].bool()
    type_target = batch["type_target"].long()
    ignored = batch["ignored"].bool()
    type_loss_mask = part_valid
    train_type = part_valid & (~ignored) & (type_target != IGNORE_TYPE_CLASS)
    movable = train_type & ((type_target == TYPE_TO_CLASS["B"]) | (type_target == TYPE_TO_CLASS["C"]))
    revolute = train_type & (type_target == TYPE_TO_CLASS["C"])

    losses: list[torch.Tensor] = []
    stats: dict[str, float] = {}
    if bool(type_loss_mask.any()):
        type_class_weights = weights.get("type_class_weights")
        if isinstance(type_class_weights, torch.Tensor):
            type_class_weights = type_class_weights.to(device=out["type_logits"].device, dtype=out["type_logits"].dtype)
        else:
            type_class_weights = None
        type_ce = F.cross_entropy(out["type_logits"][type_loss_mask], type_target[type_loss_mask], weight=type_class_weights)
    else:
        type_ce = out["type_logits"].sum() * 0.0
    losses.append(float(weights.get("type", 1.0)) * type_ce)
    stats["loss_type"] = float(type_ce.detach().item())

    if bool(movable.any()):
        anchor_t = axis_anchor_targets(batch["axis"], model.axis_anchors)
        axis_anchor_ce = F.cross_entropy(out["axis_anchor_logits"][movable], anchor_t[movable])
        pred_axis = model.predict_axis(out)
        axis_cos = (pred_axis[movable] * batch["axis"][movable]).sum(dim=-1).clamp(-1.0, 1.0)
        axis_loss = (1.0 - axis_cos).mean()
        limit_loss = F.smooth_l1_loss(out["limits"][movable], batch["limits"][movable])
    else:
        axis_anchor_ce = out["axis_anchor_logits"].sum() * 0.0
        axis_loss = out["axis_residual"].sum() * 0.0
        limit_loss = out["limits"].sum() * 0.0
    losses.append(float(weights.get("axis_anchor", 0.5)) * axis_anchor_ce)
    losses.append(float(weights.get("axis", 1.0)) * axis_loss)
    losses.append(float(weights.get("limit", 1.0)) * limit_loss)
    stats["loss_axis_anchor"] = float(axis_anchor_ce.detach().item())
    stats["loss_axis"] = float(axis_loss.detach().item())
    stats["loss_limit"] = float(limit_loss.detach().item())

    if bool(revolute.any()):
        flat_idx = (
            batch["pivot_index"][..., 0] * 16 * 16
            + batch["pivot_index"][..., 1] * 16
            + batch["pivot_index"][..., 2]
        ).long()
        pivot_ce = F.cross_entropy(out["pivot_heatmap_logits"][revolute], flat_idx[revolute])
        pivot_off = F.smooth_l1_loss(out["pivot_offset"][revolute], batch["pivot_offset"][revolute])
    else:
        pivot_ce = out["pivot_heatmap_logits"].sum() * 0.0
        pivot_off = out["pivot_offset"].sum() * 0.0
    losses.append(float(weights.get("pivot_heatmap", 1.0)) * pivot_ce)
    losses.append(float(weights.get("pivot_offset", 1.0)) * pivot_off)
    stats["loss_pivot_heatmap"] = float(pivot_ce.detach().item())
    stats["loss_pivot_offset"] = float(pivot_off.detach().item())

    if bool(train_type.any()):
        parent_logits = out["parent_logits"].clone()
        max_parts = part_valid.shape[1]
        allowed = torch.zeros((part_valid.shape[0], max_parts + 1), dtype=torch.bool, device=part_valid.device)
        allowed[:, BODY_PARENT_SLOT] = True
        allowed[:, 1:] = part_valid
        parent_logits = parent_logits.masked_fill(~allowed.unsqueeze(1), -1.0e4)
        parent_ce = F.cross_entropy(parent_logits[train_type], batch["parent"].long()[train_type])
    else:
        parent_ce = out["parent_logits"].sum() * 0.0
    losses.append(float(weights.get("parent", 1.0)) * parent_ce)
    stats["loss_parent"] = float(parent_ce.detach().item())
    total = torch.stack([x if x.dim() == 0 else x.mean() for x in losses]).sum()
    stats["loss"] = float(total.detach().item())
    stats["n_type"] = float(type_loss_mask.sum().detach().item())
    stats["n_type_main"] = float(train_type.sum().detach().item())
    stats["n_movable"] = float(movable.sum().detach().item())
    stats["n_revolute"] = float(revolute.sum().detach().item())
    return total, stats


@torch.no_grad()
def joint_head_metrics(model: JointHeadNet, out: Mapping[str, torch.Tensor], batch: Mapping[str, torch.Tensor]) -> dict[str, float]:
    part_valid = batch["part_valid"].bool()
    type_target = batch["type_target"].long()
    ignored = batch["ignored"].bool()
    train_type = part_valid & (~ignored) & (type_target != IGNORE_TYPE_CLASS)
    movable = train_type & ((type_target == TYPE_TO_CLASS["B"]) | (type_target == TYPE_TO_CLASS["C"]))
    revolute = train_type & (type_target == TYPE_TO_CLASS["C"])
    metrics: dict[str, float] = {
        "n": float(train_type.sum().item()),
        "n_movable": float(movable.sum().item()),
        "n_revolute": float(revolute.sum().item()),
    }
    if bool(train_type.any()):
        pred_type = out["type_logits"].argmax(dim=-1)
        metrics["type_acc"] = float((pred_type[train_type] == type_target[train_type]).float().mean().item())
        for code, cls in TYPE_TO_CLASS.items():
            cls_mask = train_type & (type_target == cls)
            if bool(cls_mask.any()):
                metrics[f"type_acc_{code}"] = float((pred_type[cls_mask] == type_target[cls_mask]).float().mean().item())
        parent_logits = out["parent_logits"].clone()
        allowed = torch.zeros((part_valid.shape[0], part_valid.shape[1] + 1), dtype=torch.bool, device=part_valid.device)
        allowed[:, BODY_PARENT_SLOT] = True
        allowed[:, 1:] = part_valid
        parent_logits = parent_logits.masked_fill(~allowed.unsqueeze(1), -1.0e4)
        parent_pred = parent_logits.argmax(dim=-1)
        metrics["parent_acc"] = float((parent_pred[train_type] == batch["parent"].long()[train_type]).float().mean().item())
    else:
        metrics["type_acc"] = 0.0
        metrics["parent_acc"] = 0.0
    if bool(movable.any()):
        pred_axis = model.predict_axis(out)
        cos_vec = (pred_axis[movable] * batch["axis"][movable]).sum(dim=-1).clamp(-1.0, 1.0)
        metrics["axis_vector_err_deg"] = float(torch.rad2deg(torch.acos(cos_vec)).mean().item())
        metrics["axis_err_deg"] = float(torch.rad2deg(torch.acos(cos_vec.abs().clamp(0.0, 1.0))).mean().item())
        metrics["limit_err"] = float((out["limits"][movable] - batch["limits"][movable]).abs().mean().item())
        for code in ("B", "C"):
            cls_mask = movable & (type_target == TYPE_TO_CLASS[code])
            if bool(cls_mask.any()):
                cls_cos = (pred_axis[cls_mask] * batch["axis"][cls_mask]).sum(dim=-1).clamp(-1.0, 1.0)
                metrics[f"axis_vector_err_deg_{code}"] = float(torch.rad2deg(torch.acos(cls_cos)).mean().item())
                metrics[f"axis_err_deg_{code}"] = float(torch.rad2deg(torch.acos(cls_cos.abs().clamp(0.0, 1.0))).mean().item())
                metrics[f"limit_err_{code}"] = float((out["limits"][cls_mask] - batch["limits"][cls_mask]).abs().mean().item())
    else:
        metrics["axis_err_deg"] = 0.0
        metrics["axis_vector_err_deg"] = 0.0
        metrics["limit_err"] = 0.0
    if bool(revolute.any()):
        flat_idx = out["pivot_heatmap_logits"].argmax(dim=-1)
        x = torch.div(flat_idx, 16 * 16, rounding_mode="floor")
        y = torch.div(flat_idx - x * 16 * 16, 16, rounding_mode="floor")
        z = flat_idx - x * 16 * 16 - y * 16
        center = torch.stack([x, y, z], dim=-1).float() / 15.0 - 0.5
        pred_pivot = center + out["pivot_offset"]
        gt_pivot = batch["pivot"]
        axis = F.normalize(batch["axis"], dim=-1)
        delta = pred_pivot[revolute] - gt_pivot[revolute]
        cross = torch.linalg.norm(torch.cross(delta, axis[revolute], dim=-1), dim=-1)
        metrics["pivot_axis_err"] = float(cross.mean().item())
    else:
        metrics["pivot_axis_err"] = 0.0
    return metrics


def aggregate_stats(rows: Iterable[Mapping[str, float]]) -> dict[str, float]:
    sums: defaultdict[str, float] = defaultdict(float)
    counts: defaultdict[str, int] = defaultdict(int)
    for row in rows:
        for key, value in row.items():
            if isinstance(value, (int, float)) and math.isfinite(float(value)):
                sums[key] += float(value)
                counts[key] += 1
    return {key: sums[key] / max(1, counts[key]) for key in sorted(sums)}


def json_dumps(data: Mapping[str, Any]) -> str:
    return json.dumps(data, ensure_ascii=False, sort_keys=True)
