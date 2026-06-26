#!/usr/bin/env python3
"""Diagnose concat SS-flow against the verified multiflow sampler.

This script intentionally keeps the multiflow path as per-view single-cond
forward passes followed by velocity averaging.  It never feeds concatenated
4-view tokens to the old multiflow checkpoint.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import types
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
TRELLIS_ROOT = PROJECT_ROOT / "TRELLIS-arts"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(TRELLIS_ROOT) not in sys.path:
    sys.path.insert(0, str(TRELLIS_ROOT))

os.environ.setdefault("ATTN_BACKEND", "sdpa")
os.environ.setdefault("SPCONV_ALGO", "native")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")


def register_trellis_package() -> None:
    if "trellis" not in sys.modules:
        pkg = types.ModuleType("trellis")
        pkg.__path__ = [str(TRELLIS_ROOT / "trellis")]
        pkg.__package__ = "trellis"
        sys.modules["trellis"] = pkg
    for subpkg in ("models", "modules", "pipelines", "utils", "datasets", "trainers"):
        name = f"trellis.{subpkg}"
        if name not in sys.modules:
            mod = types.ModuleType(name)
            mod.__path__ = [str(TRELLIS_ROOT / "trellis" / subpkg)]
            mod.__package__ = name
            sys.modules[name] = mod


register_trellis_package()

from safetensors.torch import load_file  # noqa: E402
from trellis.models.sparse_structure_flow import SparseStructureFlowModel  # noqa: E402
from trellis.models.sparse_structure_vae import SparseStructureDecoder  # noqa: E402


DEFAULT_MODEL_CONFIG = Path("/robot/data-lab/jzh/art-gen/weights/ss_flow_img_dit_L_16l8_fp16.json")
DEFAULT_DECODER_CKPT = PROJECT_ROOT / "pretrained/TRELLIS-image-large/ckpts/ss_dec_conv3d_16l8_fp16.safetensors"
DEFAULT_CONCAT_CKPT = Path(
    "/robot/data-lab/jzh/art-gen-output/tre-ss-concat-0616-1/ckpts/denoiser_step0012500.pt"
)
DEFAULT_MULTIFLOW_RAW_CKPT = Path(
    "/robot/data-lab/jzh/art-gen-output/tre_mf_4view_multiflow_0611/ckpts/denoiser_step0020000.pt"
)
DEFAULT_MULTIFLOW_EMA_CKPT = Path(
    "/robot/data-lab/jzh/art-gen-output/tre_mf_4view_multiflow_0611/ckpts/denoiser_ema0.9999_step0020000.pt"
)
DEFAULT_OUT_DIR = Path("/robot/data-lab/jzh/art-gen-output/tre-ss-eval/0617-concat-vs-multiflow")

DATASETS = [
    {
        "source": "physx-mobility",
        "data_root": Path(
            "/robot/data-lab/jzh/art-gen/data/PhysX-Mobility-full-4view-0511/PhysX-Mobility-full-4view-0511"
        ),
        "manifest": Path(
            "/robot/data-lab/jzh/art-gen/data/PhysX-Mobility-full-4view-0511/PhysX-Mobility-full-4view-0511/manifests/part_completion/arts_mllm_physx-mobility.train.jsonl"
        ),
        "tokens_subdirs": ["reconstruction/dinov2_tokens_official_prenorm1374", "reconstruction/dinov2_tokens"],
    },
    {
        "source": "phyx-verse",
        "data_root": Path("/robot/data-lab/jzh/art-gen/data/phyx-verse"),
        "manifest": Path("/robot/data-lab/jzh/art-gen/data/phyx-verse/manifests/part_completion/arts_mllm_phyx-verse.train.jsonl"),
        "tokens_subdirs": ["reconstruction/dinov2_tokens"],
    },
]


def require_file(path: Path, label: str) -> Path:
    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"{label} not found: {path}")
    return path


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(require_file(path, "json").read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: expected JSON object")
    return payload


def rooted(root: Path, rel_or_abs: str | Path) -> Path:
    path = Path(rel_or_abs)
    return path if path.is_absolute() else root / path


def object_sort_key(obj_id: str) -> tuple[int, str]:
    try:
        return (0, f"{int(obj_id):012d}")
    except ValueError:
        return (1, obj_id)


def find_token_path(data_root: Path, obj_id: str, angle_idx: int, tokens_subdirs: list[str], manifest_rel: str) -> Path:
    for subdir in tokens_subdirs:
        candidate = data_root / subdir / obj_id / f"angle_{angle_idx}" / "tokens.npz"
        if candidate.is_file():
            return candidate.resolve()
    candidate = rooted(data_root, manifest_rel)
    if candidate.is_file():
        return candidate.resolve()
    raise FileNotFoundError(f"tokens not found for {obj_id} angle_{angle_idx}: tried {tokens_subdirs} and {candidate}")


def build_available_samples() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[tuple[str, str, int]] = set()
    for ds in DATASETS:
        data_root = Path(ds["data_root"])
        manifest = require_file(Path(ds["manifest"]), f"{ds['source']} manifest")
        with manifest.open("r", encoding="utf-8") as handle:
            for line_no, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                rec = json.loads(line)
                obj_id = str(rec.get("object_id", rec.get("obj_id", "")))
                if not obj_id:
                    continue
                angle_idx = int(rec.get("angle_idx", rec.get("angle", 0)))
                key = (str(ds["source"]), obj_id, angle_idx)
                if key in seen:
                    continue
                seen.add(key)
                paths = dict(rec.get("paths", {}))
                view_indices = [int(v) for v in rec.get("view_indices", [])]
                if len(view_indices) != 4 or len(set(view_indices)) != 4:
                    continue
                rows.append({
                    "source": str(ds["source"]),
                    "data_root": str(data_root.resolve()),
                    "tokens_subdirs": list(ds["tokens_subdirs"]),
                    "manifest": str(manifest.resolve()),
                    "manifest_line": line_no,
                    "object_id": obj_id,
                    "angle_idx": angle_idx,
                    "category": rec.get("category"),
                    "name": rec.get("name"),
                    "target_part_count": int(rec.get("target_part_count", 0) or 0),
                    "view_indices": view_indices,
                    "paths": paths,
                })
    rows.sort(key=lambda r: (r["source"], object_sort_key(str(r["object_id"])), int(r["angle_idx"])))
    return rows


def materialize_sample(row: dict[str, Any]) -> dict[str, Any] | None:
    data_root = Path(str(row["data_root"]))
    obj_id = str(row["object_id"])
    angle_idx = int(row["angle_idx"])
    paths = dict(row.get("paths", {}))
    try:
        token_path = find_token_path(
            data_root,
            obj_id,
            angle_idx,
            [str(x) for x in row["tokens_subdirs"]],
            str(paths.get("dinov2_tokens", "")),
        )
    except FileNotFoundError:
        return None
    latent_path = rooted(data_root, paths.get(
        "overall_latent",
        f"reconstruction/ss_latents_expanded/{obj_id}/angle_{angle_idx}/latent.npz",
    ))
    surface_path = rooted(data_root, paths.get(
        "overall_surface",
        f"reconstruction/voxel_expanded/{obj_id}/angle_{angle_idx}/64/surface.npy",
    ))
    if not latent_path.is_file() or not surface_path.is_file():
        return None
    out = dict(row)
    out.pop("data_root", None)
    out.pop("tokens_subdirs", None)
    out.pop("paths", None)
    out["token_path"] = str(token_path)
    out["latent_path"] = str(latent_path.resolve())
    out["surface_path"] = str(surface_path.resolve())
    return out


def select_samples(rows: list[dict[str, Any]], max_extra_objects: int) -> list[dict[str, Any]]:
    by_obj: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_obj[(str(row["source"]), str(row["object_id"]))].append(row)
    for obj_rows in by_obj.values():
        obj_rows.sort(key=lambda r: int(r["angle_idx"]))

    selected: list[dict[str, Any]] = []
    wanted_obj = ("phyx-verse", "004d1e9e13934e319094151a4fad823f")
    for angle_idx in (0, 1, 2, 3):
        match = [r for r in by_obj.get(wanted_obj, []) if int(r["angle_idx"]) == angle_idx]
        if not match:
            raise RuntimeError(f"required target sample missing: {wanted_obj[1]} angle_{angle_idx}")
        row = materialize_sample(match[0])
        if row is None:
            raise RuntimeError(f"required target sample files missing: {wanted_obj[1]} angle_{angle_idx}")
        row["selection_group"] = "target_same_object_4angle"
        selected.append(row)

    extra_objects: list[tuple[str, str]] = []
    # Mix simple/default-pose and articulated/nonzero-pose examples across both sources.
    preferred = [
        ("physx-mobility", 1, 0),
        ("physx-mobility", 2, 0),
        ("physx-mobility", 3, 0),
        ("physx-mobility", 4, 0),
        ("physx-mobility", 1, 9),
        ("physx-mobility", 3, 9),
        ("phyx-verse", 2, 0),
        ("phyx-verse", 4, 0),
        ("phyx-verse", 2, 9),
        ("phyx-verse", 4, 9),
    ]
    selected_keys = {(r["source"], r["object_id"], int(r["angle_idx"])) for r in selected}
    for source, min_parts, desired_angle in preferred:
        if len(extra_objects) >= max_extra_objects:
            break
        for obj_key, obj_rows in by_obj.items():
            if obj_key == wanted_obj or obj_key in extra_objects or obj_key[0] != source:
                continue
            candidates = [
                r for r in obj_rows
                if int(r["target_part_count"]) >= min_parts and int(r["angle_idx"]) == desired_angle
            ]
            if not candidates:
                continue
            row = materialize_sample(candidates[0])
            if row is None:
                continue
            key = (row["source"], row["object_id"], int(row["angle_idx"]))
            if key in selected_keys:
                continue
            row["selection_group"] = "multi_object_default" if desired_angle == 0 else "multi_object_articulated"
            selected.append(row)
            selected_keys.add(key)
            extra_objects.append(obj_key)
            break

    if len(extra_objects) < max_extra_objects:
        for obj_key, obj_rows in by_obj.items():
            if obj_key == wanted_obj or obj_key in extra_objects:
                continue
            angle0 = [r for r in obj_rows if int(r["angle_idx"]) == 0]
            if not angle0:
                continue
            row = materialize_sample(angle0[0])
            if row is None:
                continue
            row["selection_group"] = "multi_object_default"
            selected.append(row)
            extra_objects.append(obj_key)
            if len(extra_objects) >= max_extra_objects:
                break

    return selected


def load_tokens(path: Path, view_indices: list[int], device: torch.device) -> torch.Tensor:
    with np.load(require_file(path, "tokens")) as payload:
        if "tokens" not in payload.files:
            raise KeyError(f"{path}: expected key 'tokens', got {payload.files}")
        arr = np.asarray(payload["tokens"])
    if arr.ndim != 3 or arr.shape[1:] != (1374, 1024):
        raise ValueError(f"{path}: expected [V,1374,1024], got {arr.shape}")
    if min(view_indices) < 0 or max(view_indices) >= arr.shape[0]:
        raise ValueError(f"{path}: cannot select views {view_indices} from shape {arr.shape}")
    out = torch.from_numpy(np.ascontiguousarray(arr[view_indices])).to(device=device, dtype=torch.float32)
    if tuple(out.shape) != (4, 1374, 1024):
        raise ValueError(f"{path}: selected token shape {tuple(out.shape)} != (4,1374,1024)")
    return out


def load_gt_latent(path: Path) -> torch.Tensor:
    with np.load(require_file(path, "GT latent")) as payload:
        if "mean" not in payload.files:
            raise KeyError(f"{path}: expected key 'mean', got {payload.files}")
        latent = torch.from_numpy(np.asarray(payload["mean"])).float()
    if tuple(latent.shape) != (8, 16, 16, 16):
        raise ValueError(f"{path}: expected latent shape (8,16,16,16), got {tuple(latent.shape)}")
    return latent


def load_gt_surface(path: Path) -> np.ndarray:
    coords = np.load(require_file(path, "GT surface"))
    if coords.ndim != 2 or coords.shape[1] != 3:
        raise ValueError(f"{path}: expected coords [N,3], got {coords.shape}")
    return np.ascontiguousarray(coords.astype(np.int64, copy=False))


def load_model(ckpt_path: Path, *, device: torch.device, use_view_id_embedding: bool) -> SparseStructureFlowModel:
    config = read_json(DEFAULT_MODEL_CONFIG)
    args = dict(config.get("args", config.get("models", {}).get("denoiser", {}).get("args", {})))
    if not args:
        raise KeyError(f"{DEFAULT_MODEL_CONFIG}: missing model args")
    args["use_camera_pose"] = False
    args["use_view_id_embedding"] = bool(use_view_id_embedding)
    args["num_view_embeddings"] = 4
    model = SparseStructureFlowModel(**args).to(device).eval()
    state = torch.load(require_file(ckpt_path, "SS-flow checkpoint"), map_location="cpu", weights_only=True)
    missing, unexpected = model.load_state_dict(state, strict=False)
    allowed_missing = {"view_id_embedding.weight"} if use_view_id_embedding else set()
    bad_missing = [key for key in missing if key not in allowed_missing]
    if bad_missing or unexpected:
        raise RuntimeError(f"{ckpt_path}: missing={bad_missing[:20]} unexpected={unexpected[:20]}")
    if getattr(model, "use_fp16", False):
        model.convert_to_fp16()
    for param in model.parameters():
        param.requires_grad_(False)
    return model


def load_decoder(ckpt_path: Path, *, device: torch.device) -> SparseStructureDecoder:
    ckpt_path = require_file(ckpt_path, "SS decoder checkpoint")
    config = read_json(ckpt_path.with_suffix(".json"))
    decoder = SparseStructureDecoder(**dict(config["args"])).to(device).eval()
    decoder.load_state_dict(load_file(str(ckpt_path)), strict=True)
    for param in decoder.parameters():
        param.requires_grad_(False)
    return decoder


@torch.no_grad()
def sample_latent(
    model: SparseStructureFlowModel,
    tokens: torch.Tensor,
    *,
    mode: str,
    seed: int,
    steps: int,
    cfg_strength: float,
) -> torch.Tensor:
    device = next(model.parameters()).device
    generator = torch.Generator(device=device)
    generator.manual_seed(int(seed))
    sample = torch.randn(
        1,
        model.in_channels,
        model.resolution,
        model.resolution,
        model.resolution,
        generator=generator,
        device=device,
        dtype=torch.float32,
    )
    t_seq = np.linspace(1.0, 0.0, int(steps) + 1)
    if mode == "multidiffusion":
        cond = tokens
        neg_cond = torch.zeros(1, 1374, 1024, device=device, dtype=cond.dtype)
        for t, t_prev in zip(t_seq[:-1], t_seq[1:]):
            t_model = torch.tensor([1000.0 * float(t)], device=device, dtype=torch.float32)
            preds = [model(sample, t_model, cond[i : i + 1]) for i in range(cond.shape[0])]
            pred = torch.stack(preds, dim=0).mean(dim=0)
            neg_pred = model(sample, t_model, neg_cond)
            pred_v = (1.0 + float(cfg_strength)) * pred - float(cfg_strength) * neg_pred
            sample = sample - (float(t) - float(t_prev)) * pred_v
    elif mode == "concat":
        cond = tokens.unsqueeze(0)
        neg_cond = torch.zeros_like(cond)
        for t, t_prev in zip(t_seq[:-1], t_seq[1:]):
            t_model = torch.tensor([1000.0 * float(t)], device=device, dtype=torch.float32)
            pred = model(sample, t_model, cond)
            neg_pred = model(sample, t_model, neg_cond)
            pred_v = (1.0 + float(cfg_strength)) * pred - float(cfg_strength) * neg_pred
            sample = sample - (float(t) - float(t_prev)) * pred_v
    else:
        raise ValueError(f"unknown sampling mode: {mode}")
    latent = sample[0].detach().float().cpu()
    if tuple(latent.shape) != (8, 16, 16, 16):
        raise RuntimeError(f"sampled latent shape {tuple(latent.shape)} != (8,16,16,16)")
    if not torch.isfinite(latent).all():
        raise RuntimeError("sampled latent contains NaN/Inf")
    return latent


@torch.no_grad()
def decode_latent_to_coords(decoder: SparseStructureDecoder, latent: torch.Tensor, threshold: float) -> tuple[np.ndarray, dict[str, Any]]:
    device = next(decoder.parameters()).device
    logits = decoder(latent.unsqueeze(0).to(device=device))[0, 0].detach().float().cpu()
    coords = torch.nonzero(logits > float(threshold), as_tuple=False).long().numpy()
    flat = logits.reshape(-1)
    return np.ascontiguousarray(coords.astype(np.int64, copy=False)), {
        "voxels": int(coords.shape[0]),
        "logit_min": float(flat.min().item()),
        "logit_max": float(flat.max().item()),
        "logit_mean": float(flat.mean().item()),
    }


def coord_keys(coords: np.ndarray) -> set[int]:
    if coords.size == 0:
        return set()
    arr = coords.astype(np.int64, copy=False)
    return {int(x) * 4096 + int(y) * 64 + int(z) for x, y, z in arr}


def coords_metrics(pred: np.ndarray, gt: np.ndarray) -> dict[str, float]:
    pred_keys = coord_keys(pred)
    gt_keys = coord_keys(gt)
    inter = len(pred_keys & gt_keys)
    union = len(pred_keys | gt_keys)
    return {
        "iou": float(inter / union) if union else 1.0,
        "precision": float(inter / len(pred_keys)) if pred_keys else 0.0,
        "recall": float(inter / len(gt_keys)) if gt_keys else 0.0,
        "intersection": float(inter),
        "pred_voxels": float(len(pred_keys)),
        "gt_voxels": float(len(gt_keys)),
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"no rows for {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def mean(values: list[float]) -> float:
    return float(np.mean(values)) if values else float("nan")


def median(values: list[float]) -> float:
    return float(np.median(values)) if values else float("nan")


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(row["threshold"], row["model"], row["selection_group"])].append(row)
        groups[(row["threshold"], row["model"], "all")].append(row)
    for (threshold, model, selection_group), items in sorted(groups.items(), key=lambda x: (float(x[0][0]), str(x[0][1]), str(x[0][2]))):
        ious = [float(r["iou_vs_gt_decoded"]) for r in items]
        ratios = [float(r["pred_gt_voxel_ratio"]) for r in items if float(r["gt_voxels_decoded"]) > 0]
        out.append({
            "threshold": threshold,
            "model": model,
            "selection_group": selection_group,
            "n": len(items),
            "mean_iou": mean(ious),
            "median_iou": median(ious),
            "min_iou": float(np.min(ious)),
            "max_iou": float(np.max(ious)),
            "frac_iou_gt_0p7": mean([1.0 if v > 0.7 else 0.0 for v in ious]),
            "mean_pred_gt_voxel_ratio": mean(ratios),
            "median_pred_gt_voxel_ratio": median(ratios),
        })
    return out


def snapshot_trend(concat_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for json_path in sorted((concat_dir / "samples").glob("step*/global_z_decode_step*.json")):
        payload = json.loads(json_path.read_text(encoding="utf-8"))
        step_text = json_path.parent.name.replace("step", "")
        step = int(step_text)
        for sample in payload.get("samples", []):
            gt_count = int(sample["gt"]["count"])
            pred_count = int(sample["sample"]["count"])
            rows.append({
                "step": step,
                "object_id": sample.get("obj_id"),
                "angle_idx": int(sample.get("angle_idx", 0)),
                "iou": float(sample.get("sample_vs_gt_iou", 0.0)),
                "gt_voxels": gt_count,
                "sample_voxels": pred_count,
                "sample_gt_voxel_ratio": float(pred_count / gt_count) if gt_count else float("nan"),
                "json_path": str(json_path),
            })
    return rows


def dataframe_to_markdown(rows: list[dict[str, Any]], columns: list[str], limit: int | None = None) -> str:
    shown = rows if limit is None else rows[:limit]
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in shown:
        vals = []
        for col in columns:
            val = row.get(col, "")
            if isinstance(val, float):
                vals.append(f"{val:.4g}")
            else:
                vals.append(str(val))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def write_report(out_dir: Path, *, args: argparse.Namespace, sample_rows: list[dict[str, Any]], summary_rows: list[dict[str, Any]], trend_rows: list[dict[str, Any]]) -> None:
    threshold0 = [r for r in sample_rows if abs(float(r["threshold"]) - 0.0) < 1.0e-9]
    target0 = [r for r in threshold0 if r["selection_group"] == "target_same_object_4angle"]
    target0.sort(key=lambda r: (int(r["angle_idx"]), str(r["model"])))
    multi0 = [r for r in threshold0 if r["selection_group"] != "target_same_object_4angle"]
    multi0.sort(key=lambda r: (str(r["source"]), str(r["object_id"]), int(r["angle_idx"]), str(r["model"])))
    threshold_summary = [
        r for r in summary_rows
        if r["selection_group"] == "all" and r["model"] in {"concat_raw_step12500", "multiflow_raw_step20000"}
    ]
    threshold_summary.sort(key=lambda r: (str(r["model"]), float(r["threshold"])))
    trend_summary: list[dict[str, Any]] = []
    by_step: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in trend_rows:
        by_step[int(row["step"])].append(row)
    for step in sorted(by_step):
        items = by_step[step]
        trend_summary.append({
            "step": step,
            "mean_iou": mean([float(r["iou"]) for r in items]),
            "median_iou": median([float(r["iou"]) for r in items]),
            "mean_sample_gt_voxel_ratio": mean([float(r["sample_gt_voxel_ratio"]) for r in items]),
            "angles": ",".join(str(int(r["angle_idx"])) for r in items),
        })

    report = f"""# Concat SS-flow Diagnosis 0617

## Protocol

- concat ckpt: `{args.concat_ckpt}`
- multiflow raw ckpt: `{args.multiflow_raw_ckpt}`
- multiflow EMA ckpt: `{args.multiflow_ema_ckpt}`
- multiflow mode: per denoise step, 4 single-view forwards, then mean velocity; `use_view_id_embedding=False`.
- concat mode: one `[1,4,1374,1024]` condition into concat/view-id model; `use_view_id_embedding=True`.
- seed/steps/cfg: `{args.seed}` / `{args.steps}` / `{args.cfg_strength}`
- thresholds: `{','.join(str(x) for x in args.thresholds)}`
- IoU threshold口径: sample latent 和 GT latent 都用同一个 decoder threshold 解码后比较；额外记录 GT surface IoU 只作参考。

## Same Object, Same 4 Angles, Threshold 0.0

{dataframe_to_markdown(target0, ["object_id", "angle_idx", "model", "iou_vs_gt_decoded", "iou_vs_gt_surface", "pred_voxels", "gt_voxels_decoded", "pred_gt_voxel_ratio", "precision", "recall"])}

## Multi-object Distribution, Threshold 0.0

{dataframe_to_markdown([r for r in summary_rows if abs(float(r["threshold"])) < 1.0e-9], ["threshold", "model", "selection_group", "n", "mean_iou", "median_iou", "min_iou", "max_iou", "frac_iou_gt_0p7", "mean_pred_gt_voxel_ratio"])}

## Threshold Effect

{dataframe_to_markdown(threshold_summary, ["threshold", "model", "selection_group", "n", "mean_iou", "median_iou", "mean_pred_gt_voxel_ratio", "frac_iou_gt_0p7"])}

## Concat Snapshot Trend

{dataframe_to_markdown(trend_summary, ["step", "mean_iou", "median_iou", "mean_sample_gt_voxel_ratio", "angles"])}

## Artifacts

- per-sample metrics: `{out_dir / "per_sample_metrics.csv"}`
- summary: `{out_dir / "summary_metrics.csv"}`
- selected samples: `{out_dir / "selected_samples.json"}`
- snapshot trend: `{out_dir / "concat_snapshot_trend.csv"}`
"""
    (out_dir / "REPORT.md").write_text(report, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--cfg-strength", type=float, default=3.0)
    parser.add_argument("--thresholds", type=float, nargs="+", default=[0.0, 0.3, 0.5])
    parser.add_argument("--extra-objects", type=int, default=8)
    parser.add_argument("--concat-ckpt", type=Path, default=DEFAULT_CONCAT_CKPT)
    parser.add_argument("--multiflow-raw-ckpt", type=Path, default=DEFAULT_MULTIFLOW_RAW_CKPT)
    parser.add_argument("--multiflow-ema-ckpt", type=Path, default=DEFAULT_MULTIFLOW_EMA_CKPT)
    parser.add_argument("--decoder-ckpt", type=Path, default=DEFAULT_DECODER_CKPT)
    parser.add_argument("--concat-dir", type=Path, default=Path("/robot/data-lab/jzh/art-gen-output/tre-ss-concat-0616-1"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir = args.out_dir.resolve()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    if not torch.cuda.is_available() and str(args.device).startswith("cuda"):
        raise RuntimeError(f"CUDA requested ({args.device}) but unavailable")
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.cuda.manual_seed_all(int(args.seed))
    torch.manual_seed(int(args.seed))

    available = build_available_samples()
    selected = select_samples(available, int(args.extra_objects))
    (args.out_dir / "selected_samples.json").write_text(json.dumps(selected, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"[select] available={len(available)} selected={len(selected)}", flush=True)
    for sample in selected:
        print(
            f"[select] {sample['selection_group']} {sample['source']} {sample['object_id']} "
            f"angle_{sample['angle_idx']} parts={sample['target_part_count']} views={sample['view_indices']}",
            flush=True,
        )

    decoder = load_decoder(args.decoder_ckpt, device=device)
    models = [
        ("concat_raw_step12500", load_model(args.concat_ckpt, device=device, use_view_id_embedding=True), "concat"),
        ("multiflow_raw_step20000", load_model(args.multiflow_raw_ckpt, device=device, use_view_id_embedding=False), "multidiffusion"),
        ("multiflow_ema9999_step20000", load_model(args.multiflow_ema_ckpt, device=device, use_view_id_embedding=False), "multidiffusion"),
    ]

    rows: list[dict[str, Any]] = []
    for idx, sample in enumerate(selected, 1):
        tokens = load_tokens(Path(sample["token_path"]), [int(v) for v in sample["view_indices"]], device)
        gt_latent = load_gt_latent(Path(sample["latent_path"]))
        gt_surface = load_gt_surface(Path(sample["surface_path"]))
        gt_by_threshold: dict[float, tuple[np.ndarray, dict[str, Any]]] = {}
        for threshold in args.thresholds:
            gt_by_threshold[float(threshold)] = decode_latent_to_coords(decoder, gt_latent, float(threshold))

        for model_name, model, mode in models:
            latent = sample_latent(
                model,
                tokens,
                mode=mode,
                seed=int(args.seed),
                steps=int(args.steps),
                cfg_strength=float(args.cfg_strength),
            )
            for threshold in args.thresholds:
                threshold = float(threshold)
                pred_coords, pred_stats = decode_latent_to_coords(decoder, latent, threshold)
                gt_coords, gt_stats = gt_by_threshold[threshold]
                decoded_metrics = coords_metrics(pred_coords, gt_coords)
                surface_metrics = coords_metrics(pred_coords, gt_surface)
                rows.append({
                    "source": sample["source"],
                    "selection_group": sample["selection_group"],
                    "object_id": sample["object_id"],
                    "angle_idx": int(sample["angle_idx"]),
                    "category": sample.get("category"),
                    "name": sample.get("name"),
                    "target_part_count": int(sample.get("target_part_count", 0)),
                    "view_indices": json.dumps(sample["view_indices"]),
                    "model": model_name,
                    "sampler_mode": mode,
                    "threshold": threshold,
                    "iou_vs_gt_decoded": float(decoded_metrics["iou"]),
                    "iou_vs_gt_surface": float(surface_metrics["iou"]),
                    "precision": float(decoded_metrics["precision"]),
                    "recall": float(decoded_metrics["recall"]),
                    "intersection": int(decoded_metrics["intersection"]),
                    "pred_voxels": int(decoded_metrics["pred_voxels"]),
                    "gt_voxels_decoded": int(decoded_metrics["gt_voxels"]),
                    "gt_surface_voxels": int(surface_metrics["gt_voxels"]),
                    "pred_gt_voxel_ratio": float(decoded_metrics["pred_voxels"] / decoded_metrics["gt_voxels"]) if decoded_metrics["gt_voxels"] else float("nan"),
                    "gt_decoded_surface_iou": float(coords_metrics(gt_coords, gt_surface)["iou"]),
                    "pred_logit_mean": float(pred_stats["logit_mean"]),
                    "gt_logit_mean": float(gt_stats["logit_mean"]),
                    "token_path": sample["token_path"],
                    "latent_path": sample["latent_path"],
                    "surface_path": sample["surface_path"],
                    "manifest": sample["manifest"],
                    "manifest_line": int(sample["manifest_line"]),
                })
        print(f"[eval] {idx}/{len(selected)} {sample['source']} {sample['object_id']} angle_{sample['angle_idx']}", flush=True)

    summary_rows = summarize(rows)
    trend_rows = snapshot_trend(args.concat_dir)
    write_csv(args.out_dir / "per_sample_metrics.csv", rows)
    write_csv(args.out_dir / "summary_metrics.csv", summary_rows)
    write_csv(args.out_dir / "concat_snapshot_trend.csv", trend_rows)
    protocol = {
        "concat_ckpt": str(args.concat_ckpt.resolve()),
        "multiflow_raw_ckpt": str(args.multiflow_raw_ckpt.resolve()),
        "multiflow_ema_ckpt": str(args.multiflow_ema_ckpt.resolve()),
        "decoder_ckpt": str(args.decoder_ckpt.resolve()),
        "multiflow_correct_path": "per denoise step: preds=[model(x,t,cond[i:i+1]) for i in 4]; pred=mean(preds)",
        "multiflow_use_view_id_embedding": False,
        "concat_use_view_id_embedding": True,
        "seed": int(args.seed),
        "steps": int(args.steps),
        "cfg_strength": float(args.cfg_strength),
        "thresholds": [float(x) for x in args.thresholds],
    }
    (args.out_dir / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n", encoding="utf-8")
    write_report(args.out_dir, args=args, sample_rows=rows, summary_rows=summary_rows, trend_rows=trend_rows)
    print(json.dumps({
        "report": str((args.out_dir / "REPORT.md").resolve()),
        "per_sample": str((args.out_dir / "per_sample_metrics.csv").resolve()),
        "summary": str((args.out_dir / "summary_metrics.csv").resolve()),
        "selected": str((args.out_dir / "selected_samples.json").resolve()),
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
