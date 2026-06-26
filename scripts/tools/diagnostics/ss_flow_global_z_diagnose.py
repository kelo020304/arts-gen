#!/usr/bin/env python3
"""Diagnose SS global-z flow checkpoints without retraining.

This script is intentionally strict: missing files, shape mismatches, or
unsupported metric inputs raise immediately instead of falling back.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import sys
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader


PROJECT_ROOT = Path(__file__).resolve().parents[2]
TRELLIS_PATH = PROJECT_ROOT / "TRELLIS-arts"
if str(TRELLIS_PATH) not in sys.path:
    sys.path.insert(0, str(TRELLIS_PATH))


def _register_trellis_minimal() -> None:
    pkg = types.ModuleType("trellis")
    pkg.__path__ = [str(TRELLIS_PATH / "trellis")]
    pkg.__package__ = "trellis"
    sys.modules.setdefault("trellis", pkg)

    for subpackage in ("models", "modules", "trainers", "utils", "datasets"):
        module = types.ModuleType(f"trellis.{subpackage}")
        module.__path__ = [str(TRELLIS_PATH / "trellis" / subpackage)]
        module.__package__ = f"trellis.{subpackage}"
        sys.modules.setdefault(f"trellis.{subpackage}", module)

    pipelines = types.ModuleType("trellis.pipelines")
    pipelines.__path__ = [str(TRELLIS_PATH / "trellis" / "pipelines")]
    pipelines.__package__ = "trellis.pipelines"
    sys.modules.setdefault("trellis.pipelines", pipelines)


_register_trellis_minimal()

os.environ.setdefault("TORCH_HOME", str(PROJECT_ROOT / "submodules" / "TRELLIS.1"))
os.environ.setdefault("ATTN_BACKEND", "sdpa")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

from trellis.datasets.arts.ss_flow_global_z import SSFlowGlobalZDataset  # noqa: E402
from trellis.models.sparse_structure_flow import SparseStructureFlowModel  # noqa: E402
from trellis.pipelines.samplers.flow_euler import FlowEulerCfgSampler  # noqa: E402
from trellis.trainers.arts.ss_flow_global_z import (  # noqa: E402
    coords_iou,
    load_ss_decoder,
)
from trellis.utils.arts.config_utils import config_to_dict, load_config  # noqa: E402


class BatchedFlowEulerCfgSampler(FlowEulerCfgSampler):
    """FlowEulerCfgSampler with cond/neg forward batched into one model call.

    The formula is identical to ClassifierFreeGuidanceSamplerMixin:
    ``(1 + cfg_strength) * pred - cfg_strength * neg_pred``.
    """

    def _inference_model(self, model, x_t, t, cond, neg_cond, cfg_strength, **kwargs):
        batch = int(x_t.shape[0])

        def _match_batch(value: torch.Tensor, name: str) -> torch.Tensor:
            if value.shape[0] == 1 and batch > 1:
                return value.repeat(batch, *([1] * (value.ndim - 1)))
            if value.shape[0] != batch:
                raise ValueError(f"{name} batch {value.shape[0]} does not match x_t batch {batch}")
            return value

        cond = _match_batch(cond, "cond")
        neg_cond = _match_batch(neg_cond, "neg_cond")
        cam_pose = kwargs.pop("cam_pose", None)
        if cam_pose is not None:
            cam_pose = _match_batch(cam_pose, "cam_pose")
            kwargs["cam_pose"] = torch.cat([cam_pose, cam_pose], dim=0)
        x_cat = torch.cat([x_t, x_t], dim=0)
        cond_cat = torch.cat([cond, neg_cond], dim=0)
        t_cat = torch.full((batch * 2,), 1000 * float(t), device=x_t.device, dtype=torch.float32)
        pred_cat = model(x_cat, t_cat, cond_cat, **kwargs)
        pred, neg_pred = pred_cat.chunk(2, dim=0)
        return (1 + cfg_strength) * pred - cfg_strength * neg_pred


@dataclass(frozen=True)
class CoordMetrics:
    iou: float
    precision: float
    recall: float
    intersection: int
    pred_count: int
    gt_count: int


def _parse_cfg_values(value: str) -> list[float]:
    out = []
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        out.append(float(item))
    if not out:
        raise ValueError("--cfg-values produced an empty list")
    return out


def _load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def _resolve_path(path: str | os.PathLike[str]) -> Path:
    p = Path(path)
    return p if p.is_absolute() else PROJECT_ROOT / p


def _cfg_dict(cfg: Any) -> dict[str, Any]:
    return config_to_dict(cfg) if not isinstance(cfg, dict) else dict(cfg)


def _build_snapshot_dataset(cfg: Any) -> SSFlowGlobalZDataset:
    snapshot_cfg = _cfg_dict(cfg.snapshot)
    data_cfg = _cfg_dict(cfg.data)
    if "data" not in snapshot_cfg:
        raise KeyError("snapshot.data is required to reproduce val snapshot set")
    data_cfg.update(dict(snapshot_cfg["data"]))
    data_cfg.pop("exclude_obj_ids", None)
    data_cfg.pop("exclude_obj_ids_file", None)
    return SSFlowGlobalZDataset(data_cfg)


def _build_train_dataset_for_mean(cfg: Any, max_samples: int | None) -> SSFlowGlobalZDataset:
    data_cfg = _cfg_dict(cfg.data)
    if max_samples is not None:
        data_cfg["max_samples"] = int(max_samples)
    return SSFlowGlobalZDataset(data_cfg)


def _build_model(cfg: Any, device: torch.device) -> SparseStructureFlowModel:
    model_cfg = _cfg_dict(cfg.model)
    model_cfg.pop("name", None)
    args = model_cfg.pop("args", model_cfg)
    model = SparseStructureFlowModel(**args).to(device)
    return model


def _load_model_checkpoint(model: SparseStructureFlowModel, ckpt_path: Path) -> None:
    if not ckpt_path.is_file():
        raise FileNotFoundError(ckpt_path)
    state = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    missing, unexpected = model.load_state_dict(state, strict=False)
    allowed_missing = {"view_pose_proj.weight", "view_pose_proj.bias"}
    bad_missing = [key for key in missing if key not in allowed_missing]
    if bad_missing:
        raise RuntimeError(f"checkpoint load missing keys: {bad_missing[:20]}")
    if unexpected:
        raise RuntimeError(f"checkpoint load unexpected keys: {unexpected[:20]}")
    if missing:
        print(f"[diagnose] allowed missing zero-init keys: {missing}")
    if getattr(model, "use_fp16", False):
        model.convert_to_fp16()
    model.eval()


def _latent_from_sample(sample: dict[str, Any], data_root: Path) -> torch.Tensor:
    path = Path(sample["z_global_rel"])
    path = path if path.is_absolute() else data_root / path
    if not path.is_file():
        raise FileNotFoundError(path)
    with np.load(path) as payload:
        if "mean" not in payload.files:
            raise KeyError(f"{path} missing 'mean'; keys={payload.files}")
        latent_np = np.asarray(payload["mean"], dtype=np.float32)
    if latent_np.shape != (8, 16, 16, 16):
        raise ValueError(f"{path} latent shape {latent_np.shape}, expected (8,16,16,16)")
    return torch.from_numpy(latent_np).float()


def _raw_surface_from_sample(sample: dict[str, Any], data_root: Path) -> np.ndarray:
    rel = sample.get("surface_rel") or sample.get("overall_surface_rel")
    if rel is None:
        obj_id = str(sample["obj_id"])
        angle_idx = int(sample["angle_idx"])
        rel = f"reconstruction/voxel_expanded/{obj_id}/angle_{angle_idx}/64/surface.npy"
    path = Path(rel)
    path = path if path.is_absolute() else data_root / path
    if not path.is_file():
        raise FileNotFoundError(path)
    coords = np.load(path)
    if coords.ndim != 2 or coords.shape[1] != 3:
        raise ValueError(f"{path} raw surface coords shape {coords.shape}, expected [N,3]")
    if coords.size and (coords.min() < 0 or coords.max() > 63):
        raise ValueError(f"{path} raw surface coords out of [0,63]: min={coords.min()} max={coords.max()}")
    return coords.astype(np.int64, copy=False)


def _coord_metrics(pred: np.ndarray, gt: np.ndarray) -> CoordMetrics:
    if pred.ndim != 2 or pred.shape[1] != 3:
        raise ValueError(f"pred coords shape {pred.shape}, expected [N,3]")
    if gt.ndim != 2 or gt.shape[1] != 3:
        raise ValueError(f"gt coords shape {gt.shape}, expected [N,3]")
    pred_keys = {
        int(x) * 4096 + int(y) * 64 + int(z)
        for x, y, z in pred.astype(np.int64, copy=False)
    }
    gt_keys = {
        int(x) * 4096 + int(y) * 64 + int(z)
        for x, y, z in gt.astype(np.int64, copy=False)
    }
    inter = len(pred_keys & gt_keys)
    union = len(pred_keys | gt_keys)
    pred_count = len(pred_keys)
    gt_count = len(gt_keys)
    return CoordMetrics(
        iou=float(inter / union) if union else 1.0,
        precision=float(inter / pred_count) if pred_count else 0.0,
        recall=float(inter / gt_count) if gt_count else 0.0,
        intersection=int(inter),
        pred_count=int(pred_count),
        gt_count=int(gt_count),
    )


@torch.no_grad()
def _decode_latents_to_masks(decoder, latents: torch.Tensor, *, threshold: float, device: torch.device) -> torch.Tensor:
    if latents.ndim != 5 or tuple(latents.shape[1:]) != (8, 16, 16, 16):
        raise ValueError(f"latents expected [B,8,16,16,16], got {tuple(latents.shape)}")
    logits = decoder(latents.to(device=device, dtype=torch.float32))[:, 0].detach().float().cpu()
    if tuple(logits.shape[1:]) != (64, 64, 64):
        raise ValueError(f"decoder logits expected [B,64,64,64], got {tuple(logits.shape)}")
    return logits > float(threshold)


def _coords_to_mask(coords: np.ndarray) -> torch.Tensor:
    if coords.ndim != 2 or coords.shape[1] != 3:
        raise ValueError(f"coords shape {coords.shape}, expected [N,3]")
    mask = torch.zeros(64, 64, 64, dtype=torch.bool)
    if coords.size:
        coords_t = torch.from_numpy(coords.astype(np.int64, copy=False))
        if coords_t.min().item() < 0 or coords_t.max().item() > 63:
            raise ValueError(f"coords out of [0,63]: min={coords_t.min().item()} max={coords_t.max().item()}")
        mask[coords_t[:, 0], coords_t[:, 1], coords_t[:, 2]] = True
    return mask


def _mask_metrics(pred: torch.Tensor, gt: torch.Tensor) -> CoordMetrics:
    if pred.shape != (64, 64, 64) or gt.shape != (64, 64, 64):
        raise ValueError(f"mask metrics expected [64,64,64], got pred={tuple(pred.shape)} gt={tuple(gt.shape)}")
    pred = pred.bool()
    gt = gt.bool()
    intersection = int(torch.logical_and(pred, gt).sum().item())
    union = int(torch.logical_or(pred, gt).sum().item())
    pred_count = int(pred.sum().item())
    gt_count = int(gt.sum().item())
    return CoordMetrics(
        iou=float(intersection / union) if union else 1.0,
        precision=float(intersection / pred_count) if pred_count else 0.0,
        recall=float(intersection / gt_count) if gt_count else 0.0,
        intersection=intersection,
        pred_count=pred_count,
        gt_count=gt_count,
    )


def _summary(values: Iterable[float]) -> dict[str, float]:
    arr = np.asarray(list(values), dtype=np.float64)
    if arr.size == 0:
        raise ValueError("cannot summarize empty values")
    return {
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "min": float(arr.min()),
        "max": float(arr.max()),
    }


def _mean_latent(dataset: SSFlowGlobalZDataset, data_root: Path, max_samples: int | None) -> torch.Tensor:
    count = min(len(dataset), int(max_samples)) if max_samples is not None else len(dataset)
    if count <= 0:
        raise ValueError("mean latent dataset has zero samples")
    acc = torch.zeros(8, 16, 16, 16, dtype=torch.float64)
    for idx in range(count):
        acc += _latent_from_sample(dataset.samples[idx], data_root).double()
        if (idx + 1) % 1000 == 0:
            print(f"[mean] loaded {idx + 1}/{count}")
    return (acc / float(count)).float()


def _channel_rms(diff: torch.Tensor) -> list[float]:
    if diff.shape != (8, 16, 16, 16):
        raise ValueError(f"channel RMS expected latent diff (8,16,16,16), got {tuple(diff.shape)}")
    return [float(torch.sqrt(torch.mean(diff[c].float() ** 2)).item()) for c in range(8)]


def _cos(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(F.cosine_similarity(a.reshape(1, -1).float(), b.reshape(1, -1).float()).item())


def _blend_fraction(pred: torch.Tensor, gt: torch.Tensor, mean: torch.Tensor) -> float:
    # Project pred onto the line mean -> gt. 0 means exactly mean, 1 means GT.
    direction = (gt - mean).reshape(-1).float()
    denom = float(torch.dot(direction, direction).item())
    if denom <= 1.0e-12:
        return math.nan
    numer = float(torch.dot((pred - mean).reshape(-1).float(), direction).item())
    return float(numer / denom)


def _safe_name(value: float) -> str:
    return ("%g" % value).replace(".", "p").replace("-", "m")


def _load_existing_baseline(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    payload = _load_json(path)
    samples = payload.get("samples")
    if not isinstance(samples, list) or not samples:
        raise ValueError(f"{path} does not look like a global_z snapshot JSON with samples")
    ious = [float(row["sample_vs_gt_iou"]) for row in samples]
    return {
        "path": str(path),
        "threshold": float(payload["threshold"]),
        "count": len(ious),
        "mean_iou": float(np.mean(ious)),
        "median_iou": float(np.median(ious)),
    }


def _rooted(data_root: Path, rel_or_abs: str | os.PathLike[str]) -> Path:
    path = Path(rel_or_abs)
    return path if path.is_absolute() else data_root / path


def _camera_total_views(sample: dict[str, Any], data_root: Path) -> int:
    camera_rel = sample.get("camera_rel")
    if camera_rel is None:
        return 12
    payload = _load_json(_rooted(data_root, camera_rel))
    if isinstance(payload, dict) and "total_views" in payload:
        return int(payload["total_views"])
    frames = payload.get("frames") if isinstance(payload, dict) else payload
    if isinstance(frames, list) and frames:
        return len(frames)
    raise ValueError(f"cannot infer total views for sample={sample.get('sample_id')}")


def _apply_view_mode(
    samples: list[dict[str, Any]],
    *,
    data_root: Path,
    view_mode: str,
    clustered_start: str,
) -> list[dict[str, Any]]:
    if view_mode == "manifest":
        return samples
    if view_mode != "clustered":
        raise ValueError(f"unsupported view_mode={view_mode!r}")
    out = []
    for sample in samples:
        updated = dict(sample)
        original = [int(view_idx) for view_idx in updated["view_indices"]]
        total_views = _camera_total_views(updated, data_root)
        start = int(original[0]) if clustered_start == "first" else int(clustered_start)
        clustered = [(start + offset) % total_views for offset in range(len(original))]
        if len(set(clustered)) != len(clustered):
            raise ValueError(
                f"clustered selection duplicated views for sample={sample.get('sample_id')} "
                f"total_views={total_views}: {clustered}"
            )
        updated["manifest_view_indices"] = original
        updated["view_indices"] = clustered
        updated["view_mode"] = "clustered"
        out.append(updated)
    return out


def run(args: argparse.Namespace) -> None:
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{args.device}")
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    cfg = load_config(args.config)
    val_dataset = _build_snapshot_dataset(cfg)
    val_count = min(int(args.val_samples), len(val_dataset))
    if val_count <= 0:
        raise ValueError("val sample count must be positive")
    data_root = Path(_cfg_dict(cfg.data)["data_root"])
    val_dataset.samples = _apply_view_mode(
        val_dataset.samples,
        data_root=data_root,
        view_mode=str(args.view_mode),
        clustered_start=str(args.clustered_start),
    )

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    latent_dump_dir = out_dir / "latent_dumps"
    latent_dump_dir.mkdir(parents=True, exist_ok=True)

    decoder_ckpt = _resolve_path(args.decoder_ckpt or _cfg_dict(cfg.snapshot)["ss_decoder_ckpt"])
    decoder = load_ss_decoder(decoder_ckpt, device=device)

    model = _build_model(cfg, device)
    ckpt_path = _resolve_path(args.ckpt)
    _load_model_checkpoint(model, ckpt_path)
    sampler_cls = FlowEulerCfgSampler if bool(args.sequential_cfg) else BatchedFlowEulerCfgSampler
    sampler = sampler_cls(sigma_min=float(_cfg_dict(cfg.training).get("sigma_min", 1.0e-5)))

    print(f"[diagnose] checkpoint={ckpt_path}")
    print("[diagnose] checkpoint_kind=EMA expected filename contains denoiser_ema")
    print(f"[diagnose] sampler={sampler.__class__.__module__}.{sampler.__class__.__name__}")
    print(f"[diagnose] sampler_cfg_formula=(1+cfg_strength)*pred - cfg_strength*neg_pred")
    print(f"[diagnose] sequential_cfg={bool(args.sequential_cfg)}")
    print(f"[diagnose] view_mode={args.view_mode} clustered_start={args.clustered_start}")
    print(f"[diagnose] val_samples={val_count} threshold={args.threshold} decoder={decoder_ckpt}")

    loader = DataLoader(
        val_dataset,
        batch_size=int(args.batch_size),
        shuffle=False,
        num_workers=0,
        collate_fn=val_dataset.collate_fn,
    )

    gt_latents: list[torch.Tensor] = []
    raw_masks: list[torch.Tensor] = []
    sample_meta = val_dataset.samples[:val_count]
    for idx, sample in enumerate(sample_meta):
        gt_latent = _latent_from_sample(sample, data_root)
        gt_latents.append(gt_latent)
        raw_masks.append(_coords_to_mask(_raw_surface_from_sample(sample, data_root)))
        if idx + 1 >= val_count:
            break
    gt_latents_tensor = torch.stack(gt_latents, dim=0)
    gt_masks = _decode_latents_to_masks(decoder, gt_latents_tensor, threshold=float(args.threshold), device=device)

    ceiling_rows = []
    for sample, decoded_mask, raw_mask in zip(sample_meta, gt_masks, raw_masks):
        m = _mask_metrics(decoded_mask, raw_mask)
        ceiling_rows.append({
            "obj_id": str(sample["obj_id"]),
            "angle_idx": int(sample["angle_idx"]),
            "iou": m.iou,
            "precision": m.precision,
            "recall": m.recall,
            "decoded_count": m.pred_count,
            "raw_count": m.gt_count,
            "intersection": m.intersection,
        })
    ceiling = {
        "count": len(ceiling_rows),
        "iou": _summary(row["iou"] for row in ceiling_rows),
        "precision": _summary(row["precision"] for row in ceiling_rows),
        "recall": _summary(row["recall"] for row in ceiling_rows),
        "decoded_count": _summary(row["decoded_count"] for row in ceiling_rows),
        "raw_count": _summary(row["raw_count"] for row in ceiling_rows),
    }
    print(
        "[ceiling] "
        f"mean_iou={ceiling['iou']['mean']:.6f} median_iou={ceiling['iou']['median']:.6f} "
        f"mean_precision={ceiling['precision']['mean']:.6f} mean_recall={ceiling['recall']['mean']:.6f}"
    )

    cfg_values = _parse_cfg_values(args.cfg_values)
    pred_by_cfg: dict[float, list[torch.Tensor]] = {cfg_value: [] for cfg_value in cfg_values}
    rows_by_cfg: dict[float, list[dict[str, Any]]] = {cfg_value: [] for cfg_value in cfg_values}

    for cfg_value in cfg_values:
        noise_generator = torch.Generator(device=device)
        noise_generator.manual_seed(int(args.seed))
        collected = 0
        print(f"[sweep] cfg={cfg_value:g} start")
        with torch.no_grad():
            for batch_idx, batch in enumerate(loader):
                take = min(val_count - collected, int(batch["x_0"].shape[0]))
                if take <= 0:
                    break
                cond = batch["cond"][:take].to(device, non_blocking=True).float()
                cam_pose = batch["cam_pose"][:take].to(device, non_blocking=True).float()
                noise = torch.randn(
                    (take, 8, 16, 16, 16),
                    dtype=torch.float32,
                    device=device,
                    generator=noise_generator,
                )
                result = sampler.sample(
                    model=model,
                    noise=noise,
                    cond=cond,
                    neg_cond=torch.zeros_like(cond),
                    cam_pose=cam_pose,
                    steps=int(args.num_steps),
                    cfg_strength=float(cfg_value),
                    verbose=False,
                )
                pred_batch = result.samples.detach().float().cpu()
                pred_masks = _decode_latents_to_masks(
                    decoder,
                    pred_batch,
                    threshold=float(args.threshold),
                    device=device,
                )
                for local_idx in range(take):
                    row_idx = collected + local_idx
                    pred_latent = pred_batch[local_idx]
                    pred_by_cfg[cfg_value].append(pred_latent)
                    m = _mask_metrics(pred_masks[local_idx], gt_masks[row_idx])
                    sample = sample_meta[row_idx]
                    rows_by_cfg[cfg_value].append({
                        "cfg_strength": float(cfg_value),
                        "obj_id": str(sample["obj_id"]),
                        "angle_idx": int(sample["angle_idx"]),
                        "view_mode": str(sample.get("view_mode", args.view_mode)),
                        "view_indices": list(sample["view_indices"]),
                        "manifest_view_indices": list(sample.get("manifest_view_indices", sample["view_indices"])),
                        "iou": m.iou,
                        "precision": m.precision,
                        "recall": m.recall,
                        "pred_count": m.pred_count,
                        "gt_count": m.gt_count,
                        "intersection": m.intersection,
                    })
                collected += take
                if bool(args.progress) and (
                    collected >= val_count or (batch_idx + 1) % max(1, int(args.progress_every)) == 0
                ):
                    partial = rows_by_cfg[cfg_value]
                    print(
                        f"[sweep] cfg={cfg_value:g} progress={collected}/{val_count} "
                        f"mean_iou={np.mean([r['iou'] for r in partial]):.6f}",
                        flush=True,
                    )
                if collected >= val_count:
                    break
        if collected != val_count:
            raise RuntimeError(f"cfg={cfg_value:g} collected {collected}, expected {val_count}")
        cfg_rows = rows_by_cfg[cfg_value]
        print(
            f"[sweep] cfg={cfg_value:g} "
            f"mean_iou={np.mean([r['iou'] for r in cfg_rows]):.6f} "
            f"median_iou={np.median([r['iou'] for r in cfg_rows]):.6f} "
            f"mean_p={np.mean([r['precision'] for r in cfg_rows]):.6f} "
            f"mean_r={np.mean([r['recall'] for r in cfg_rows]):.6f} "
            f"mean_vox={np.mean([r['pred_count'] for r in cfg_rows]):.1f}"
        )
        torch.save(
            {
                "cfg_strength": float(cfg_value),
                "pred": torch.stack(pred_by_cfg[cfg_value], dim=0),
                "gt": gt_latents_tensor,
                "sample_meta": sample_meta,
            },
            latent_dump_dir / f"pred_gt_cfg{_safe_name(cfg_value)}.pt",
        )

    cfg_summary = []
    for cfg_value in cfg_values:
        cfg_rows = rows_by_cfg[cfg_value]
        cfg_summary.append({
            "cfg_strength": float(cfg_value),
            "count": len(cfg_rows),
            "iou": _summary(row["iou"] for row in cfg_rows),
            "precision": _summary(row["precision"] for row in cfg_rows),
            "recall": _summary(row["recall"] for row in cfg_rows),
            "pred_count": _summary(row["pred_count"] for row in cfg_rows),
            "gt_count": _summary(row["gt_count"] for row in cfg_rows),
        })

    best_cfg_row = max(cfg_summary, key=lambda row: row["iou"]["mean"])
    best_cfg = float(best_cfg_row["cfg_strength"])

    mean_dataset = None
    mean_train = None
    mean_val = gt_latents_tensor.mean(dim=0)
    if args.mean_source in ("train", "both"):
        mean_dataset = _build_train_dataset_for_mean(cfg, args.mean_train_samples)
        mean_train = _mean_latent(mean_dataset, data_root, args.mean_train_samples)
    if args.mean_source == "train":
        mean_latent = mean_train
        mean_count = min(len(mean_dataset), args.mean_train_samples) if args.mean_train_samples else len(mean_dataset)
    elif args.mean_source == "val":
        mean_latent = mean_val
        mean_count = len(gt_latents)
    else:
        if mean_train is None:
            raise RuntimeError("internal error: mean_train missing")
        mean_latent = mean_train
        mean_count = min(len(mean_dataset), args.mean_train_samples) if args.mean_train_samples else len(mean_dataset)

    diag_count = min(int(args.regression_samples), val_count)
    best_preds = pred_by_cfg[best_cfg]
    regression_rows = []
    for idx in range(diag_count):
        pred = best_preds[idx]
        gt = gt_latents[idx]
        sample = sample_meta[idx]
        pred_centered = pred - mean_latent
        gt_centered = gt - mean_latent
        pred_to_gt = pred - gt
        regression_rows.append({
            "obj_id": str(sample["obj_id"]),
            "angle_idx": int(sample["angle_idx"]),
            "cfg_strength": best_cfg,
            "channel_rms_error": _channel_rms(pred_to_gt),
            "rms_error": float(torch.sqrt(torch.mean(pred_to_gt.float() ** 2)).item()),
            "pred_vs_gt_cosine": _cos(pred, gt),
            "pred_centered_vs_gt_centered_cosine": _cos(pred_centered, gt_centered),
            "pred_vs_mean_cosine": _cos(pred, mean_latent),
            "gt_vs_mean_cosine": _cos(gt, mean_latent),
            "pred_centered_norm": float(torch.linalg.vector_norm(pred_centered.reshape(-1).float()).item()),
            "gt_centered_norm": float(torch.linalg.vector_norm(gt_centered.reshape(-1).float()).item()),
            "pred_centered_norm_over_gt_centered_norm": float(
                torch.linalg.vector_norm(pred_centered.reshape(-1).float()).item()
                / max(torch.linalg.vector_norm(gt_centered.reshape(-1).float()).item(), 1.0e-12)
            ),
            "blend_fraction_mean_to_gt": _blend_fraction(pred, gt, mean_latent),
        })

    regression_summary = {
        "count": len(regression_rows),
        "mean_source": args.mean_source,
        "mean_latent_sample_count": int(mean_count),
        "best_cfg_strength": best_cfg,
        "rms_error": _summary(row["rms_error"] for row in regression_rows),
        "pred_vs_gt_cosine": _summary(row["pred_vs_gt_cosine"] for row in regression_rows),
        "pred_centered_vs_gt_centered_cosine": _summary(
            row["pred_centered_vs_gt_centered_cosine"] for row in regression_rows
        ),
        "pred_vs_mean_cosine": _summary(row["pred_vs_mean_cosine"] for row in regression_rows),
        "gt_vs_mean_cosine": _summary(row["gt_vs_mean_cosine"] for row in regression_rows),
        "pred_centered_norm_over_gt_centered_norm": _summary(
            row["pred_centered_norm_over_gt_centered_norm"] for row in regression_rows
        ),
        "blend_fraction_mean_to_gt": _summary(row["blend_fraction_mean_to_gt"] for row in regression_rows),
        "channel_rms_error_mean": [
            float(np.mean([row["channel_rms_error"][channel] for row in regression_rows]))
            for channel in range(8)
        ],
    }
    print(
        "[regression] "
        f"best_cfg={best_cfg:g} rms={regression_summary['rms_error']['mean']:.6f} "
        f"pred_vs_mean_cos={regression_summary['pred_vs_mean_cosine']['mean']:.6f} "
        f"blend={regression_summary['blend_fraction_mean_to_gt']['mean']:.6f} "
        f"centered_norm_ratio={regression_summary['pred_centered_norm_over_gt_centered_norm']['mean']:.6f}"
    )

    existing_baseline = _load_existing_baseline(Path(args.existing_snapshot_json) if args.existing_snapshot_json else None)
    report = {
        "config": str(Path(args.config).resolve()),
        "checkpoint": str(ckpt_path),
        "checkpoint_kind": "ema" if "ema" in ckpt_path.name else "non_ema",
        "decoder_ckpt": str(decoder_ckpt),
        "threshold": float(args.threshold),
        "num_steps": int(args.num_steps),
        "view_mode": str(args.view_mode),
        "clustered_start": str(args.clustered_start),
        "sampler": {
            "class": f"{sampler.__class__.__module__}.{sampler.__class__.__name__}",
            "cfg_formula": "(1+cfg_strength)*pred - cfg_strength*neg_pred",
            "cfg_zero_is_no_guidance": True,
            "sequential_cfg": bool(args.sequential_cfg),
            "batched_cfg_equivalent_formula": not bool(args.sequential_cfg),
        },
        "val_count": int(val_count),
        "existing_snapshot": existing_baseline,
        "decode_ceiling_gt_decode_vs_raw_surface": ceiling,
        "cfg_sweep": cfg_summary,
        "best_cfg_strength": best_cfg,
        "regression_summary": regression_summary,
        "regression_rows": regression_rows,
        "outputs": {
            "latent_dump_dir": str(latent_dump_dir),
            "per_cfg_csv": str(out_dir / "per_cfg_metrics.csv"),
            "regression_csv": str(out_dir / "regression_metrics.csv"),
        },
    }

    report_path = out_dir / "summary.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    with (out_dir / "per_cfg_metrics.csv").open("w", encoding="utf-8", newline="") as fh:
        fieldnames = [
            "cfg_strength",
            "obj_id",
            "angle_idx",
            "view_mode",
            "view_indices",
            "manifest_view_indices",
            "iou",
            "precision",
            "recall",
            "pred_count",
            "gt_count",
            "intersection",
        ]
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for cfg_value in cfg_values:
            for row in rows_by_cfg[cfg_value]:
                out_row = dict(row)
                out_row["view_indices"] = json.dumps(out_row["view_indices"])
                out_row["manifest_view_indices"] = json.dumps(out_row["manifest_view_indices"])
                writer.writerow(out_row)

    with (out_dir / "regression_metrics.csv").open("w", encoding="utf-8", newline="") as fh:
        fieldnames = [
            "obj_id",
            "angle_idx",
            "cfg_strength",
            "rms_error",
            "pred_vs_gt_cosine",
            "pred_centered_vs_gt_centered_cosine",
            "pred_vs_mean_cosine",
            "gt_vs_mean_cosine",
            "pred_centered_norm",
            "gt_centered_norm",
            "pred_centered_norm_over_gt_centered_norm",
            "blend_fraction_mean_to_gt",
            "channel_rms_error",
        ]
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in regression_rows:
            out_row = dict(row)
            out_row["channel_rms_error"] = json.dumps(out_row["channel_rms_error"])
            writer.writerow(out_row)

    print(f"[done] wrote {report_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="TRELLIS-arts/configs/arts/ss_flow_global_z/full_train.yaml")
    parser.add_argument("--ckpt", required=True, help="Path to denoiser_ema...pt checkpoint")
    parser.add_argument("--decoder-ckpt", default=None)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--existing-snapshot-json", default=None)
    parser.add_argument("--cfg-values", default="0,1,2,3,5,7.5")
    parser.add_argument("--val-samples", type=int, default=200)
    parser.add_argument("--regression-samples", type=int, default=20)
    parser.add_argument("--mean-source", choices=("train", "val", "both"), default="train")
    parser.add_argument("--mean-train-samples", type=int, default=2000)
    parser.add_argument("--num-steps", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--threshold", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--view-mode", choices=("manifest", "clustered"), default="manifest")
    parser.add_argument("--clustered-start", default="first")
    parser.add_argument("--progress", action="store_true")
    parser.add_argument("--progress-every", type=int, default=1)
    parser.add_argument("--sequential-cfg", action="store_true", help="Use stock FlowEulerCfgSampler two-forward CFG path")
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
