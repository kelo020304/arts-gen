#!/usr/bin/env python3
"""Track1 data/trainer utilities for online GT mesh rendering supervision.

This module intentionally does not use dataset RGB/alpha or offline component
renders.  It feeds GT mesh vertices/faces to the native
`SLatVaeMeshDecoderTrainer.geometry_losses`, which renders mask/depth/normal
inside the training step with nvdiffrast.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset
from easydict import EasyDict as edict

REPO_ROOT = Path(__file__).resolve().parents[3]
TRELLIS_ROOT = REPO_ROOT / "TRELLIS-arts"
if str(TRELLIS_ROOT) not in sys.path:
    sys.path.insert(0, str(TRELLIS_ROOT))

import utils3d.torch  # noqa: E402
from safetensors.torch import load_file  # noqa: E402

from trellis.modules.sparse import SparseTensor  # noqa: E402
from trellis.models.structured_latent_vae.decoder_mesh_part_masked import (  # noqa: E402
    PartMaskedSLatMeshDecoder,
)
from trellis.trainers.vae.structured_latent_vae_mesh_dec import (  # noqa: E402
    SLatVaeMeshDecoderTrainer,
)
from trellis.representations import MeshExtractResult  # noqa: E402
from trellis.utils.loss_utils import l1_loss, lpips, smooth_l1_loss, ssim  # noqa: E402


DEFAULT_CACHE_MANIFEST = Path("/robot/data-lab/jzh/art-gen/data/slat_dec_part_cache/phase2_shared/phase2_shared_cache_manifest.json")
DEFAULT_DECODER_CKPT = REPO_ROOT / "pretrained/TRELLIS-image-large/ckpts/slat_dec_mesh_swin8_B_64l8m256c_fp16.safetensors"
DEGRADABLE_COMPONENT_KEYWORDS = (
    "door",
    "drawer",
    "lid",
    "panel",
    "flap",
    "cover",
    "gate",
)
TINY_COMPONENT_VOXEL_THRESHOLD = 200
MIN_DEGRADED_ABS_VOXELS = 32
MIN_DEGRADED_FRACTION = 0.20


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_obj_mesh(paths: list[Path]) -> dict[str, torch.Tensor]:
    vertices_all: list[np.ndarray] = []
    faces_all: list[np.ndarray] = []
    vertex_offset = 0
    for path in paths:
        vertices: list[list[float]] = []
        faces: list[list[int]] = []
        with path.open("r", encoding="utf-8", errors="ignore") as handle:
            for line in handle:
                if line.startswith("v "):
                    parts = line.strip().split()
                    if len(parts) < 4:
                        continue
                    vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
                elif line.startswith("f "):
                    idxs = []
                    for token in line.strip().split()[1:]:
                        raw = token.split("/")[0]
                        if not raw:
                            continue
                        idx = int(raw)
                        if idx < 0:
                            idx = len(vertices) + idx
                        else:
                            idx = idx - 1
                        idxs.append(idx + vertex_offset)
                    if len(idxs) == 3:
                        faces.append(idxs)
                    elif len(idxs) > 3:
                        for i in range(1, len(idxs) - 1):
                            faces.append([idxs[0], idxs[i], idxs[i + 1]])
        if not vertices or not faces:
            raise ValueError(f"{path}: expected non-empty vertices/faces, got v={len(vertices)} f={len(faces)}")
        vertices_np = np.asarray(vertices, dtype=np.float32)
        faces_np = np.asarray(faces, dtype=np.int64)
        vertices_all.append(vertices_np)
        faces_all.append(faces_np)
        vertex_offset += int(vertices_np.shape[0])
    v = np.concatenate(vertices_all, axis=0)
    f = np.concatenate(faces_all, axis=0)
    return {
        "vertices": torch.from_numpy(v.astype(np.float32, copy=False)),
        "faces": torch.from_numpy(f.astype(np.int64, copy=False)),
    }


def mesh_bounds(mesh: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    vertices = mesh["vertices"].float()
    lo = vertices.min(dim=0).values
    hi = vertices.max(dim=0).values
    return lo, hi


def normalize_mesh_with_object_bounds(
    mesh: dict[str, torch.Tensor],
    *,
    center: torch.Tensor,
    scale: torch.Tensor,
) -> dict[str, torch.Tensor]:
    vertices = mesh["vertices"].float()
    return {
        "vertices": (vertices - center.to(vertices)) / scale.to(vertices).clamp_min(1.0e-6),
        "faces": mesh["faces"].long(),
    }


def normalize_mesh_to_unit_cube(mesh: dict[str, torch.Tensor]) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    vertices = mesh["vertices"].float()
    lo, hi = mesh_bounds(mesh)
    center = (lo + hi) * 0.5
    scale = torch.clamp((hi - lo).max(), min=torch.tensor(1.0e-6, dtype=vertices.dtype))
    out = normalize_mesh_with_object_bounds(mesh, center=center, scale=scale)
    meta = {
        "bbox_min": [float(x) for x in lo.tolist()],
        "bbox_max": [float(x) for x in hi.tolist()],
        "center": [float(x) for x in center.tolist()],
        "scale": float(scale.item()),
    }
    return out, meta


def component_mask_for_slat_coords(slat_coords64: np.ndarray, component_coords64: np.ndarray) -> np.ndarray:
    comp = {tuple(map(int, row)) for row in np.asarray(component_coords64, dtype=np.int64).reshape(-1, 3)}
    coords = np.asarray(slat_coords64, dtype=np.int64).reshape(-1, 3)
    return np.asarray([1.0 if tuple(map(int, row)) in comp else 0.0 for row in coords], dtype=np.float32)


def dilate_coords64(coords64: np.ndarray, radius: int, *, resolution: int = 64) -> np.ndarray:
    coords = np.asarray(coords64, dtype=np.int64).reshape(-1, 3)
    radius = int(radius)
    if radius <= 0 or coords.size == 0:
        return coords.astype(np.int16, copy=False)
    offsets = np.asarray(
        [
            (dx, dy, dz)
            for dx in range(-radius, radius + 1)
            for dy in range(-radius, radius + 1)
            for dz in range(-radius, radius + 1)
            if max(abs(dx), abs(dy), abs(dz)) <= radius
        ],
        dtype=np.int64,
    )
    expanded = coords[:, None, :] + offsets[None, :, :]
    expanded = expanded.reshape(-1, 3)
    keep = np.all((expanded >= 0) & (expanded < int(resolution)), axis=1)
    if not np.any(keep):
        return coords.astype(np.int16, copy=False)
    return np.unique(expanded[keep], axis=0).astype(np.int16, copy=False)


def subset_slat_by_coords64(
    coords: np.ndarray,
    feats: np.ndarray,
    keep_coords64: np.ndarray,
    *,
    label: str,
) -> tuple[np.ndarray, np.ndarray, int]:
    coords_np = np.asarray(coords, dtype=np.int64).reshape(-1, 3)
    feats_np = np.asarray(feats, dtype=np.float32)
    keep_np = np.asarray(keep_coords64, dtype=np.int64).reshape(-1, 3)
    if keep_np.size == 0:
        raise ValueError(f"{label}: empty subset coords")
    if coords_np.ndim != 2 or coords_np.shape[1] != 3:
        raise ValueError(f"{label}: SLat coords expected [N,3], got {coords_np.shape}")
    if feats_np.ndim != 2 or feats_np.shape[0] != coords_np.shape[0]:
        raise ValueError(f"{label}: SLat feats shape mismatch coords={coords_np.shape} feats={feats_np.shape}")
    resolution = int(max(int(coords_np.max(initial=0)) + 1, int(keep_np.max(initial=0)) + 1, 64))
    slat_keys = coords_np[:, 0] * resolution * resolution + coords_np[:, 1] * resolution + coords_np[:, 2]
    keep_keys = keep_np[:, 0] * resolution * resolution + keep_np[:, 1] * resolution + keep_np[:, 2]
    keep_set = set(int(x) for x in keep_keys.tolist())
    mask = np.fromiter((int(k) in keep_set for k in slat_keys.tolist()), dtype=bool, count=len(slat_keys))
    matched = int(mask.sum())
    if matched == 0:
        raise ValueError(f"{label}: expanded subset has no overlap with whole SLat coords")
    return coords_np[mask].astype(np.int32, copy=False), feats_np[mask].astype(np.float32, copy=False), matched


def is_degradable_component(component_name: str, semantic_type: str, component_role: str) -> tuple[bool, str]:
    if str(component_role) != "part":
        return False, "not_part"
    text = f"{component_name} {semantic_type}".lower()
    blocked = ("handle", "knob", "pull", "hinge", "spout", "nozzle", "stopper", "mechanism", "leg", "arm", "wing", "head")
    for keyword in blocked:
        if keyword in text:
            return False, f"blocked_keyword:{keyword}"
    for keyword in DEGRADABLE_COMPONENT_KEYWORDS:
        if keyword in text:
            return True, f"matched_keyword:{keyword}"
    return False, "semantic_not_degradable"


def build_degradation_options(
    *,
    component_name: str,
    semantic_type: str,
    component_role: str,
    gt_count: int,
    erode_count: int,
    front_count: int,
) -> tuple[list[str], str, dict[str, int]]:
    threshold = max(MIN_DEGRADED_ABS_VOXELS, int(math.ceil(float(gt_count) * MIN_DEGRADED_FRACTION)))
    counts = {"gt": int(gt_count), "erode": int(erode_count), "front_only": int(front_count), "threshold": int(threshold)}
    if int(gt_count) < TINY_COMPONENT_VOXEL_THRESHOLD:
        return [], "tiny_component_gt_only", counts
    degradable, reason = is_degradable_component(component_name, semantic_type, component_role)
    if not degradable:
        return [], reason, counts
    options: list[str] = []
    if int(front_count) >= threshold:
        options.append("front_only")
    if int(erode_count) >= threshold:
        options.append("erode")
    if not options:
        return [], f"{reason};no_degraded_mask_meets_threshold", counts
    return options, reason, counts


def random_spherical_camera(
    *,
    device: torch.device,
    radius: float = 2.0,
    fov_deg: float = 30.0,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if generator is None:
        yaw = torch.rand((), device=device) * (2.0 * math.pi)
        z = torch.rand((), device=device) * 1.6 - 0.8
    else:
        yaw = torch.rand((), device=device, generator=generator) * (2.0 * math.pi)
        z = torch.rand((), device=device, generator=generator) * 1.6 - 0.8
    xy = torch.sqrt(torch.clamp(1.0 - z * z, min=1.0e-4))
    origin = torch.stack([torch.sin(yaw) * xy, torch.cos(yaw) * xy, z]) * float(radius)
    fov = torch.deg2rad(torch.tensor(float(fov_deg), device=device))
    extrinsics = utils3d.torch.extrinsics_look_at(
        origin.float(),
        torch.zeros(3, dtype=torch.float32, device=device),
        torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32, device=device),
    )
    intrinsics = utils3d.torch.intrinsics_from_fov_xy(fov, fov)
    return extrinsics.float(), intrinsics.float()


class PartMaskedOnlineRenderDataset(Dataset):
    def __init__(
        self,
        manifest_path: Path = DEFAULT_CACHE_MANIFEST,
        *,
        resolution: int = 512,
        include_body: bool = True,
        normalize_gt_mesh: bool = True,
        max_components: int = 0,
        mask_degrade_prob: float = 0.0,
        front_only_prob: float = 0.5,
        latent_input_mode: str = "whole",
        subset_dilation: int = 1,
    ) -> None:
        self.manifest_path = manifest_path.resolve()
        self.cache_root = self.manifest_path.parent
        payload = load_json(self.manifest_path)
        self.resolution = int(resolution)
        self.normalize_gt_mesh = bool(normalize_gt_mesh)
        self.mask_degrade_prob = float(mask_degrade_prob)
        self.front_only_prob = float(front_only_prob)
        self.latent_input_mode = str(latent_input_mode)
        if self.latent_input_mode not in {"whole", "expanded_subset"}:
            raise ValueError(f"unsupported latent_input_mode={self.latent_input_mode!r}")
        self.subset_dilation = int(subset_dilation)
        object_norms: dict[tuple[str, str, int], dict[str, Any]] = {}
        self.samples: list[dict[str, Any]] = []
        self.degradation_report: list[dict[str, Any]] = []
        for obj in payload.get("objects", []):
            slat_path = self.cache_root / str(obj["overall_slat_rel"])
            obj_key = (str(obj["dataset_id"]), str(obj["obj_id"]), int(obj["angle_idx"]))
            if self.normalize_gt_mesh:
                lows = []
                highs = []
                for component in obj.get("components", []):
                    paths = [Path(record["cache_path"]) for record in component.get("gt_mesh_records", [])]
                    if not paths:
                        continue
                    mesh = load_obj_mesh(paths)
                    lo, hi = mesh_bounds(mesh)
                    lows.append(lo)
                    highs.append(hi)
                if not lows:
                    raise ValueError(f"{obj_key}: cannot compute object-level GT mesh normalization; no mesh paths")
                lo_all = torch.stack(lows, dim=0).min(dim=0).values
                hi_all = torch.stack(highs, dim=0).max(dim=0).values
                center = (lo_all + hi_all) * 0.5
                scale = torch.clamp((hi_all - lo_all).max(), min=torch.tensor(1.0e-6, dtype=lo_all.dtype))
                object_norms[obj_key] = {
                    "bbox_min": [float(x) for x in lo_all.tolist()],
                    "bbox_max": [float(x) for x in hi_all.tolist()],
                    "center": center,
                    "scale": scale,
                    "scale_float": float(scale.item()),
                }
            for component in obj.get("components", []):
                if not include_body and component.get("component_role") == "body":
                    continue
                comp_cache = self.cache_root / str(component["component_cache_rel"])
                mesh_paths = [Path(record["cache_path"]) for record in component.get("gt_mesh_records", [])]
                if not slat_path.is_file():
                    raise FileNotFoundError(f"overall SLat missing: {slat_path}")
                if not comp_cache.is_file():
                    raise FileNotFoundError(f"component cache missing: {comp_cache}")
                if not mesh_paths:
                    raise ValueError(f"{obj['dataset_id']}::{obj['obj_id']} {component['component_name']}: no GT mesh records")
                for mesh_path in mesh_paths:
                    if not mesh_path.is_file():
                        raise FileNotFoundError(f"GT mesh missing: {mesh_path}")
                with np.load(comp_cache, allow_pickle=False) as comp_data:
                    gt_count = int(np.asarray(comp_data["coords64"]).shape[0])
                    erode_count = int(np.asarray(comp_data["coords64_erode"]).shape[0])
                    front_count = int(np.asarray(comp_data["coords64_front_only"]).shape[0])
                degrade_options, degrade_reason, degrade_counts = build_degradation_options(
                    component_name=str(component["component_name"]),
                    semantic_type=str(component["semantic_type"]),
                    component_role=str(component["component_role"]),
                    gt_count=gt_count,
                    erode_count=erode_count,
                    front_count=front_count,
                )
                self.degradation_report.append(
                    {
                        "tag": obj["tag"],
                        "dataset_id": obj["dataset_id"],
                        "obj_id": obj["obj_id"],
                        "angle_idx": int(obj["angle_idx"]),
                        "component_name": component["component_name"],
                        "component_role": component["component_role"],
                        "semantic_type": component["semantic_type"],
                        "gt_voxels": gt_count,
                        "front_only_voxels": front_count,
                        "erode_voxels": erode_count,
                        "threshold": degrade_counts["threshold"],
                        "degrade_options": list(degrade_options),
                        "degrade_reason": degrade_reason,
                    }
                )
                self.samples.append(
                    {
                        "dataset_id": obj["dataset_id"],
                        "obj_id": obj["obj_id"],
                        "tag": obj["tag"],
                        "angle_idx": int(obj["angle_idx"]),
                        "component_name": component["component_name"],
                        "component_role": component["component_role"],
                        "semantic_type": component["semantic_type"],
                        "slat_path": slat_path,
                        "component_cache": comp_cache,
                        "mesh_paths": mesh_paths,
                        "object_norm": object_norms.get(obj_key),
                        "degrade_options": list(degrade_options),
                        "degrade_reason": degrade_reason,
                        "degrade_counts": degrade_counts,
                    }
                )
        if max_components and int(max_components) > 0:
            self.samples = self.samples[: int(max_components)]
        if not self.samples:
            raise ValueError(f"{self.manifest_path}: no components selected")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        sample = self.samples[idx]
        with np.load(sample["slat_path"], allow_pickle=False) as data:
            coords = np.asarray(data["coords"], dtype=np.int32)
            feats = np.asarray(data["feats"], dtype=np.float32)
        with np.load(sample["component_cache"], allow_pickle=False) as data:
            comp_coords_gt = np.asarray(data["coords64"], dtype=np.int16)
            comp_coords_erode = np.asarray(data["coords64_erode"], dtype=np.int16)
            comp_coords_front = np.asarray(data["coords64_front_only"], dtype=np.int16)
        if coords.ndim != 2 or coords.shape[1] != 3:
            raise ValueError(f"{sample['slat_path']}: coords expected [N,3], got {coords.shape}")
        if feats.ndim != 2 or feats.shape != (coords.shape[0], 8):
            raise ValueError(f"{sample['slat_path']}: feats expected [{coords.shape[0]},8], got {feats.shape}")
        comp_coords = comp_coords_gt
        mask_mode = "gt"
        degrade_options = list(sample.get("degrade_options") or [])
        if self.mask_degrade_prob > 0 and degrade_options and random.random() < self.mask_degrade_prob:
            if "front_only" in degrade_options and "erode" in degrade_options:
                mask_mode = "front_only" if random.random() < self.front_only_prob else "erode"
            elif "front_only" in degrade_options:
                mask_mode = "front_only"
            elif "erode" in degrade_options:
                mask_mode = "erode"
            if mask_mode == "front_only":
                comp_coords = comp_coords_front
            elif mask_mode == "erode":
                comp_coords = comp_coords_erode
        original_slat_voxels = int(coords.shape[0])
        subset_matched = original_slat_voxels
        subset_coords64_count = 0
        if self.latent_input_mode == "expanded_subset":
            expanded_coords = dilate_coords64(comp_coords, self.subset_dilation)
            subset_coords64_count = int(expanded_coords.shape[0])
            coords, feats, subset_matched = subset_slat_by_coords64(
                coords,
                feats,
                expanded_coords,
                label=f"{sample['dataset_id']}::{sample['obj_id']} {sample['component_name']}",
            )
        mask = component_mask_for_slat_coords(coords, comp_coords).reshape(-1, 1)
        if float(mask.sum()) <= 0.0:
            raise ValueError(
                f"{sample['dataset_id']}::{sample['obj_id']} {sample['component_name']}: mask has no overlap with SLat coords"
            )
        mesh = load_obj_mesh(sample["mesh_paths"])
        mesh_meta = None
        if self.normalize_gt_mesh:
            norm = sample.get("object_norm")
            if norm is None:
                mesh, mesh_meta = normalize_mesh_to_unit_cube(mesh)
            else:
                center = norm["center"]
                scale = norm["scale"]
                mesh = normalize_mesh_with_object_bounds(mesh, center=center, scale=scale)
                mesh_meta = {
                    "mode": "object_level_unit_cube",
                    "bbox_min": norm["bbox_min"],
                    "bbox_max": norm["bbox_max"],
                    "center": [float(x) for x in center.tolist()],
                    "scale": float(norm["scale_float"]),
                }
        return {
            "coords": torch.from_numpy(coords),
            "feats": torch.from_numpy(np.concatenate([feats, mask], axis=1).astype(np.float32, copy=False)),
            "image": torch.zeros((3, self.resolution, self.resolution), dtype=torch.float32),
            "alpha": torch.zeros((self.resolution, self.resolution), dtype=torch.float32),
            "mesh": mesh,
            "sample_meta": {
                **{k: sample[k] for k in ("dataset_id", "obj_id", "tag", "angle_idx", "component_name", "component_role", "semantic_type")},
                "mask_mode": mask_mode,
                "mask_voxels_on_slat": int(mask.sum()),
                "mask_gt_coords64": int(len(comp_coords_gt)),
                "mask_used_coords64": int(len(comp_coords)),
                "mask_degrade_options": list(degrade_options),
                "mask_degrade_reason": sample.get("degrade_reason"),
                "mask_degrade_threshold": int(sample.get("degrade_counts", {}).get("threshold", 0)),
                "slat_voxels": int(coords.shape[0]),
                "original_slat_voxels": original_slat_voxels,
                "latent_input_mode": self.latent_input_mode,
                "subset_dilation": int(self.subset_dilation),
                "subset_coords64": int(subset_coords64_count),
                "subset_matched_slat_voxels": int(subset_matched),
                "gt_mesh_vertices": int(mesh["vertices"].shape[0]),
                "gt_mesh_faces": int(mesh["faces"].shape[0]),
                "gt_mesh_normalize": mesh_meta,
            },
        }

    @staticmethod
    def collate_fn(batch: list[dict[str, Any]]) -> dict[str, Any]:
        coords_parts = []
        feats_parts = []
        for batch_idx, sample in enumerate(batch):
            coords = sample["coords"].to(dtype=torch.int32)
            batch_col = torch.full((coords.shape[0], 1), batch_idx, dtype=torch.int32)
            coords_parts.append(torch.cat([batch_col, coords], dim=1))
            feats_parts.append(sample["feats"].float())
        return {
            "coords": torch.cat(coords_parts, dim=0),
            "feats": torch.cat(feats_parts, dim=0),
            "image": torch.stack([sample["image"] for sample in batch], dim=0),
            "alpha": torch.stack([sample["alpha"] for sample in batch], dim=0),
            "mesh": [sample["mesh"] for sample in batch],
            "sample_meta": [sample["sample_meta"] for sample in batch],
        }


class PartMaskedSLatVaeMeshDecoderTrainer(SLatVaeMeshDecoderTrainer):
    """Native mesh decoder trainer with random online cameras for part meshes."""

    def __init__(self, *args: Any, render_resolution: int = 512, camera_radius: float = 2.0, camera_fov_deg: float = 30.0, **kwargs: Any) -> None:
        kwargs.setdefault("lambda_color", 0.0)
        kwargs.setdefault("lambda_lpips", 0.0)
        super().__init__(*args, **kwargs)
        self.render_resolution = int(render_resolution)
        self.camera_radius = float(camera_radius)
        self.camera_fov_deg = float(camera_fov_deg)

    def _perceptual_loss(self, gt: torch.Tensor, pred: torch.Tensor, name: str) -> dict[str, torch.Tensor]:
        if gt.shape[1] != 3:
            if gt.shape[-1] != 3:
                raise ValueError(f"{name}: expected 3 channels, got gt={tuple(gt.shape)}")
            gt = gt.permute(0, 3, 1, 2)
        if pred.shape[1] != 3:
            if pred.shape[-1] != 3:
                raise ValueError(f"{name}: expected 3 channels, got pred={tuple(pred.shape)}")
            pred = pred.permute(0, 3, 1, 2)
        terms = {
            f"{name}_loss": l1_loss(gt, pred),
            f"{name}_loss_ssim": 1 - ssim(gt, pred),
        }
        perceptual = terms[f"{name}_loss"] + terms[f"{name}_loss_ssim"] * self.lambda_ssim
        if self.lambda_lpips > 0:
            terms[f"{name}_loss_lpips"] = lpips(gt, pred)
            perceptual = perceptual + terms[f"{name}_loss_lpips"] * self.lambda_lpips
        else:
            terms[f"{name}_loss_lpips"] = torch.zeros((), dtype=gt.dtype, device=gt.device)
        terms[f"{name}_loss_perceptual"] = perceptual
        return terms

    def _tsdf_reg_loss_guarded(
        self,
        rep: Any,
        depth_map: torch.Tensor,
        extrinsics: torch.Tensor,
        intrinsics: torch.Tensor,
    ) -> tuple[torch.Tensor, bool]:
        zero = depth_map.new_zeros(())
        if not bool(getattr(rep, "success", False)):
            return zero, False
        if getattr(rep, "tsdf_v", None) is None or getattr(rep, "tsdf_s", None) is None:
            return zero, False
        if rep.tsdf_v.numel() == 0 or rep.tsdf_s.numel() == 0:
            return zero, False
        with torch.no_grad():
            projected_pts, pts_depth = utils3d.torch.project_cv(extrinsics=extrinsics, intrinsics=intrinsics, points=rep.tsdf_v)
            projected_pts = (projected_pts - 0.5) * 2.0
            depth_map_res = int(depth_map.shape[1])
            gt_depth = torch.nn.functional.grid_sample(
                depth_map.reshape(1, 1, depth_map_res, depth_map_res),
                projected_pts.reshape(1, 1, -1, 2),
                mode="bilinear",
                padding_mode="border",
                align_corners=True,
            )
            pseudo_sdf = gt_depth.flatten() - pts_depth.flatten()
            delta = 1 / rep.res * 3.0
            trunc_mask = pseudo_sdf > -delta
        if int(trunc_mask.sum().item()) == 0:
            return zero, False
        gt_tsdf = torch.clamp(pseudo_sdf[trunc_mask], -delta, delta)
        tsdf = rep.tsdf_s.flatten()[trunc_mask]
        if gt_tsdf.numel() == 0 or tsdf.numel() == 0:
            return zero, False
        return torch.mean((tsdf - gt_tsdf) ** 2), True

    def _calc_tsdf_loss_guarded(
        self,
        reps: list[Any],
        depth_maps: torch.Tensor,
        extrinsics: torch.Tensor,
        intrinsics: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        total = len(reps)
        if total == 0:
            zero = depth_maps.new_zeros(())
            return zero, {"tsdf_valid": zero.detach(), "tsdf_skipped": zero.detach(), "tsdf_total": zero.detach()}
        valid_losses: list[torch.Tensor] = []
        for i, rep in enumerate(reps):
            loss, valid = self._tsdf_reg_loss_guarded(rep, depth_maps[i], extrinsics[i], intrinsics[i])
            if valid:
                valid_losses.append(loss)
        if valid_losses:
            tsdf_loss = torch.stack(valid_losses).mean()
        else:
            tsdf_loss = depth_maps.new_zeros(())
        valid_count = len(valid_losses)
        stats = {
            "tsdf_valid": depth_maps.new_tensor(float(valid_count)),
            "tsdf_skipped": depth_maps.new_tensor(float(total - valid_count)),
            "tsdf_total": depth_maps.new_tensor(float(total)),
        }
        return tsdf_loss, stats

    def geometry_losses(
        self,
        reps: list[Any],
        mesh: list[dict[str, torch.Tensor]],
        normal_map: torch.Tensor,
        extrinsics: torch.Tensor,
        intrinsics: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        with torch.no_grad():
            gt_meshes = []
            for i in range(len(reps)):
                gt_mesh = MeshExtractResult(mesh[i]["vertices"].to(self.device), mesh[i]["faces"].to(self.device))
                gt_meshes.append(gt_mesh)
            target = self._render_batch(gt_meshes, extrinsics, intrinsics, return_types=["mask", "depth", "normal"])
            target["normal"] = self._flip_normal(target["normal"], extrinsics, intrinsics)

        terms: dict[str, torch.Tensor] = edict(geo_loss=extrinsics.new_zeros(()))
        if self.lambda_tsdf > 0:
            tsdf_loss, tsdf_stats = self._calc_tsdf_loss_guarded(reps, target["depth"], extrinsics, intrinsics)
            terms["tsdf_loss"] = tsdf_loss
            terms.update(tsdf_stats)
            terms["geo_loss"] = terms["geo_loss"] + tsdf_loss * self.lambda_tsdf
        else:
            zero = extrinsics.new_zeros(())
            terms.update({"tsdf_loss": zero, "tsdf_valid": zero, "tsdf_skipped": zero, "tsdf_total": zero})

        return_types = ["mask", "depth", "normal", "normal_map"] if self.use_color else ["mask", "depth", "normal"]
        buffer = self._render_batch(reps, extrinsics, intrinsics, return_types=return_types)

        success_mask = torch.tensor([bool(rep.success) for rep in reps], device=self.device)
        if success_mask.sum() != 0:
            for k, v in buffer.items():
                buffer[k] = v[success_mask]
            for k, v in target.items():
                target[k] = v[success_mask]

            terms["mask_loss"] = l1_loss(buffer["mask"], target["mask"])
            if self.depth_loss_type == "l1":
                terms["depth_loss"] = l1_loss(buffer["depth"] * target["mask"], target["depth"] * target["mask"])
            elif self.depth_loss_type == "smooth_l1":
                terms["depth_loss"] = smooth_l1_loss(
                    buffer["depth"] * target["mask"],
                    target["depth"] * target["mask"],
                    beta=1.0 / (2 * reps[0].res),
                )
            else:
                raise ValueError(f"Unsupported depth loss type: {self.depth_loss_type}")
            terms.update(self._perceptual_loss(buffer["normal"] * target["mask"], target["normal"] * target["mask"], "normal"))
            terms["geo_loss"] = terms["geo_loss"] + terms["mask_loss"] + terms["depth_loss"] * self.lambda_depth + terms["normal_loss_perceptual"]
            if self.use_color and normal_map is not None:
                terms.update(self._perceptual_loss(normal_map[success_mask], buffer["normal_map"], "normal_map"))
                terms["geo_loss"] = terms["geo_loss"] + terms["normal_map_loss_perceptual"] * self.lambda_color
        return terms

    def _regularization_loss_guarded(self, reps: list[Any]) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if not reps:
            zero = torch.zeros((), device=self.device)
            return zero, {"reg_valid": zero.detach(), "reg_skipped": zero.detach(), "reg_total": zero.detach()}
        valid_losses: list[torch.Tensor] = []
        for rep in reps:
            reg = getattr(rep, "reg_loss", None)
            if reg is None:
                continue
            if not bool(getattr(rep, "success", False)):
                continue
            if torch.is_tensor(reg) and torch.isfinite(reg.detach()).all():
                valid_losses.append(reg)
        if valid_losses:
            reg_loss = torch.stack([x.reshape(()) for x in valid_losses]).mean()
        else:
            reg_loss = reps[0].vertices.new_zeros(()) if torch.is_tensor(getattr(reps[0], "vertices", None)) else torch.zeros((), device=self.device)
        valid_count = len(valid_losses)
        stats = {
            "reg_valid": reg_loss.new_tensor(float(valid_count)),
            "reg_skipped": reg_loss.new_tensor(float(len(reps) - valid_count)),
            "reg_total": reg_loss.new_tensor(float(len(reps))),
        }
        return reg_loss, stats

    def training_losses(
        self,
        latents: SparseTensor,
        image: torch.Tensor,
        alpha: torch.Tensor,
        mesh: list[dict[str, torch.Tensor]],
        extrinsics: torch.Tensor,
        intrinsics: torch.Tensor,
        normal_map: torch.Tensor | None = None,
    ) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
        reps = self.training_models["decoder"](latents)
        self.renderer.rendering_options.resolution = image.shape[-1]

        terms: dict[str, torch.Tensor] = edict(loss=image.new_zeros(()), rec=image.new_zeros(()))
        reg_loss, reg_stats = self._regularization_loss_guarded(reps)
        terms["reg_loss"] = reg_loss
        terms.update(reg_stats)
        terms["loss"] = terms["loss"] + reg_loss

        geo_terms = self.geometry_losses(reps, mesh, normal_map, extrinsics, intrinsics)
        terms.update(geo_terms)
        terms["loss"] = terms["loss"] + terms["geo_loss"]

        if self.use_color:
            color_terms = self.color_losses(reps, image, alpha, extrinsics, intrinsics)
            terms.update(color_terms)
            terms["loss"] = terms["loss"] + terms["color_loss"]
        return terms, {
            "rep_success": sum(1 for rep in reps if bool(getattr(rep, "success", False))),
            "rep_total": len(reps),
        }

    def prepare_batch(self, batch: dict[str, Any], *, device: torch.device | str) -> dict[str, Any]:
        coords = batch["coords"].to(device=device, dtype=torch.int32)
        feats = batch["feats"].to(device=device, dtype=torch.float32)
        latents = SparseTensor(coords=coords, feats=feats)
        batch_size = int(batch["image"].shape[0])
        extrinsics = []
        intrinsics = []
        for _idx in range(batch_size):
            ext, intr = random_spherical_camera(device=torch.device(device), radius=self.camera_radius, fov_deg=self.camera_fov_deg)
            extrinsics.append(ext)
            intrinsics.append(intr)
        return {
            "latents": latents,
            "image": batch["image"].to(device=device, dtype=torch.float32),
            "alpha": batch["alpha"].to(device=device, dtype=torch.float32),
            "mesh": [
                {"vertices": mesh["vertices"].to(device=device, dtype=torch.float32), "faces": mesh["faces"].to(device=device, dtype=torch.long)}
                for mesh in batch["mesh"]
            ],
            "extrinsics": torch.stack(extrinsics, dim=0),
            "intrinsics": torch.stack(intrinsics, dim=0),
        }


class OnlineRenderLossProbe(PartMaskedSLatVaeMeshDecoderTrainer):
    """Small wrapper that can be constructed without BasicTrainer.__init__."""

    def __init__(
        self,
        decoder: torch.nn.Module,
        *,
        device: torch.device,
        render_resolution: int,
        lambda_tsdf: float,
        lambda_ssim: float,
        lambda_lpips: float,
    ) -> None:
        self.training_models = {"decoder": decoder}
        self.models = self.training_models
        self.depth_loss_type = "smooth_l1"
        self.lambda_depth = 10.0
        self.lambda_ssim = float(lambda_ssim)
        self.lambda_lpips = float(lambda_lpips)
        self.lambda_tsdf = float(lambda_tsdf)
        self.lambda_color = 0.0
        self.use_color = False
        self.render_resolution = int(render_resolution)
        self.camera_radius = 2.0
        self.camera_fov_deg = 30.0
        self._init_renderer()


def build_partmasked_decoder_from_pretrained(
    ckpt: Path,
    *,
    device: torch.device,
    train: bool = True,
    mask_modulation: str = "none",
) -> PartMaskedSLatMeshDecoder:
    cfg_path = ckpt.with_suffix(".json")
    if not cfg_path.is_file():
        raise FileNotFoundError(f"mesh decoder config missing: {cfg_path}")
    if not ckpt.is_file():
        raise FileNotFoundError(f"mesh decoder weights missing: {ckpt}")
    cfg = load_json(cfg_path)
    args = dict(cfg["args"])
    args.pop("latent_channels", None)
    decoder = PartMaskedSLatMeshDecoder(
        **args,
        base_latent_channels=8,
        mask_channels=1,
        mask_modulation=str(mask_modulation),
    ).to(device)
    state = load_file(str(ckpt), device=str(device))
    decoder.load_partmasked_state_dict_from_base(state, strict=True)
    if train:
        decoder.train()
    else:
        decoder.eval()
    return decoder


def run_smoke(args: argparse.Namespace) -> None:
    if args.seed is not None:
        seed = int(args.seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    device = torch.device(f"cuda:{int(args.gpu)}" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("Track1 online render smoke requires CUDA/nvdiffrast")
    torch.cuda.set_device(device)
    dataset = PartMaskedOnlineRenderDataset(
        args.manifest,
        resolution=int(args.resolution),
        include_body=bool(args.include_body),
        normalize_gt_mesh=not bool(args.no_normalize_gt_mesh),
        max_components=int(args.max_components),
        latent_input_mode=str(args.latent_input_mode),
        subset_dilation=int(args.subset_dilation),
    )
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=int(args.batch_size),
        shuffle=False,
        num_workers=0,
        collate_fn=dataset.collate_fn,
    )
    batch = next(iter(loader))
    decoder = build_partmasked_decoder_from_pretrained(args.decoder_ckpt, device=device, train=True)
    probe = OnlineRenderLossProbe(
        decoder,
        device=device,
        render_resolution=int(args.resolution),
        lambda_tsdf=float(args.lambda_tsdf),
        lambda_ssim=float(args.lambda_ssim),
        lambda_lpips=float(args.lambda_lpips),
    )
    prepared = probe.prepare_batch(batch, device=device)
    prepared["image"] = torch.zeros_like(prepared["image"][:, :, : int(args.resolution), : int(args.resolution)])
    with torch.set_grad_enabled(True):
        losses, status = probe.training_losses(**prepared)
        loss = losses["loss"]
        if not torch.isfinite(loss):
            raise RuntimeError(f"non-finite loss: {loss}")
        if bool(args.backward):
            loss.backward()
    report = {
        "manifest": str(args.manifest.resolve()),
        "decoder_ckpt": str(args.decoder_ckpt.resolve()),
        "gpu": int(args.gpu),
        "resolution": int(args.resolution),
        "batch_size": int(args.batch_size),
        "latent_input_mode": str(args.latent_input_mode),
        "subset_dilation": int(args.subset_dilation),
        "lambda_color": 0.0,
        "lambda_tsdf": float(args.lambda_tsdf),
        "lambda_ssim": float(args.lambda_ssim),
        "lambda_lpips": float(args.lambda_lpips),
        "backward": bool(args.backward),
        "samples": batch["sample_meta"],
        "losses": {k: float(v.detach().float().cpu().item()) if torch.is_tensor(v) else float(v) for k, v in losses.items()},
        "status": status,
        "peak_mem_mb": float(torch.cuda.max_memory_allocated(device) / (1024.0 * 1024.0)),
    }
    out_path = args.out_json
    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=DEFAULT_CACHE_MANIFEST)
    parser.add_argument("--decoder-ckpt", type=Path, default=DEFAULT_DECODER_CKPT)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-components", type=int, default=1)
    parser.add_argument("--lambda-tsdf", type=float, default=0.01)
    parser.add_argument("--lambda-ssim", type=float, default=0.2)
    parser.add_argument("--lambda-lpips", type=float, default=0.0)
    parser.add_argument("--include-body", action="store_true")
    parser.add_argument("--no-normalize-gt-mesh", action="store_true")
    parser.add_argument("--latent-input-mode", choices=["whole", "expanded_subset"], default="whole")
    parser.add_argument("--subset-dilation", type=int, default=1)
    parser.add_argument("--backward", action="store_true")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--out-json", type=Path, default=Path("/robot/data-lab/jzh/art-gen/data/slat_dec_part_cache/track1_online_render_smoke.json"))
    run_smoke(parser.parse_args())


if __name__ == "__main__":
    os.environ.setdefault("SPCONV_ALGO", "native")
    main()
