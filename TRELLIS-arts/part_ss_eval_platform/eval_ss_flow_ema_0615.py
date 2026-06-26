from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import subprocess
import sys
import threading
import time
import types
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader


REPO_ROOT = Path(__file__).resolve().parents[2]
TRELLIS_PATH = REPO_ROOT / "TRELLIS-arts"
if str(TRELLIS_PATH) not in sys.path:
    sys.path.insert(0, str(TRELLIS_PATH))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

os.environ.setdefault("TORCH_HOME", str(REPO_ROOT / "submodules" / "TRELLIS.1"))
os.environ.setdefault("ATTN_BACKEND", "sdpa")
os.environ.setdefault("SPCONV_ALGO", "native")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")


def _setup_trellis_imports() -> None:
    pkg = types.ModuleType("trellis")
    pkg.__path__ = [str(TRELLIS_PATH / "trellis")]
    pkg.__package__ = "trellis"
    sys.modules.setdefault("trellis", pkg)
    for sp in ("models", "modules", "trainers", "utils", "datasets", "pipelines", "renderers"):
        mod = types.ModuleType(f"trellis.{sp}")
        mod.__path__ = [str(TRELLIS_PATH / "trellis" / sp)]
        mod.__package__ = f"trellis.{sp}"
        sys.modules.setdefault(f"trellis.{sp}", mod)


_setup_trellis_imports()

from part_ss_eval_platform.eval_0615 import (  # noqa: E402
    BUCKETS,
    DEFAULT_DATA_CONFIG,
    DEFAULT_SPLIT_JSON,
    DEFAULT_SS_DECODER_CKPT,
    DEFAULT_SS_FLOW_CKPT,
    build_selection,
    load_data_config,
    _dataset_for,
)
from scripts.tools.render.render_voxel_eval_tripanel_flat import render_tripanel  # noqa: E402
from trellis.datasets.arts.ss_flow_global_z import SSFlowGlobalZDataset  # noqa: E402
from trellis.models.sparse_structure_flow import SparseStructureFlowModel  # noqa: E402
from trellis.pipelines.samplers.flow_euler import FlowEulerCfgSampler  # noqa: E402
from trellis.trainers.arts.ss_flow_global_z import load_ss_decoder  # noqa: E402
from trellis.utils.arts.config_utils import config_to_dict, load_config  # noqa: E402


DEFAULT_CONFIG = TRELLIS_PATH / "configs/arts/ss_flow_global_z/official_multiflow_train_full_0611.yaml"
DEFAULT_BARE_CKPT = Path("/robot/data-lab/jzh/art-gen-output/tre_mf_4view_multiflow_0611/ckpts/denoiser_step0020000.pt")
DEFAULT_EMA0999_CKPT = Path(
    "/robot/data-lab/jzh/art-gen-output/tre_mf_4view_multiflow_0611/ckpts/denoiser_ema0.999_step0020000.pt"
)
DEFAULT_OUT_ROOT = Path("/robot/data-lab/jzh/art-gen-output/EE-eval")


class BatchedFlowEulerCfgSampler(FlowEulerCfgSampler):
    """FlowEuler CFG sampler with conditional/unconditional passes batched."""

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
        x_cat = torch.cat([x_t, x_t], dim=0)
        cond_cat = torch.cat([cond, neg_cond], dim=0)
        t_cat = torch.full((batch * 2,), 1000 * float(t), device=x_t.device, dtype=torch.float32)
        pred_cat = model(x_cat, t_cat, cond_cat, **kwargs)
        pred, neg_pred = pred_cat.chunk(2, dim=0)
        return (1.0 + float(cfg_strength)) * pred - float(cfg_strength) * neg_pred


class MultiflowWrapper(torch.nn.Module):
    """Apply the 0611 multiflow training rule: per-view forward, then velocity mean."""

    def __init__(self, model: torch.nn.Module):
        super().__init__()
        self.model = model

    def forward(self, x_t: torch.Tensor, t: torch.Tensor, cond: torch.Tensor, **kwargs):
        if not isinstance(cond, torch.Tensor) or cond.ndim != 4:
            return self.model(x_t, t, cond, **kwargs)
        batch, num_views, token_count, token_dim = cond.shape
        cond_flat = cond.reshape(batch * num_views, token_count, token_dim).contiguous()
        x_flat = x_t[:, None].expand(-1, num_views, -1, -1, -1, -1).reshape(
            batch * num_views, *x_t.shape[1:]
        ).contiguous()
        if isinstance(t, torch.Tensor):
            if t.ndim == 0:
                t_flat = t.reshape(1).expand(batch * num_views).contiguous()
            else:
                t_flat = t[:, None].expand(-1, num_views).reshape(batch * num_views).contiguous()
        else:
            t_flat = t
        pred = self.model(x_flat, t_flat, cond_flat, **kwargs)
        return pred.reshape(batch, num_views, *x_t.shape[1:]).mean(dim=1)


class VramSampler:
    def __init__(self, gpu: str, interval: float = 0.5) -> None:
        self.gpu = str(gpu).split(",")[0]
        self.interval = float(interval)
        self.max_mib = 0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def __enter__(self) -> "VramSampler":
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)

    def _run(self) -> None:
        while not self._stop.is_set():
            value = self._query()
            if value is not None:
                self.max_mib = max(self.max_mib, value)
            self._stop.wait(self.interval)

    def _query(self) -> int | None:
        try:
            result = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=memory.used",
                    "--format=csv,noheader,nounits",
                    "-i",
                    self.gpu,
                ],
                capture_output=True,
                text=True,
                timeout=2,
            )
        except Exception:
            return None
        if result.returncode != 0:
            return None
        values: list[int] = []
        for line in result.stdout.splitlines():
            try:
                values.append(int(line.strip()))
            except ValueError:
                pass
        return max(values) if values else None


def _cfg_dict(cfg: Any) -> dict[str, Any]:
    return config_to_dict(cfg) if not isinstance(cfg, dict) else dict(cfg)


def _require_file(path: Path, label: str) -> Path:
    path = Path(path).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"{label} not found: {path}")
    return path


def _resolve_path(path: str | os.PathLike[str]) -> Path:
    p = Path(path)
    return p if p.is_absolute() else REPO_ROOT / p


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _parse_cfg_values(value: str) -> list[float]:
    values = []
    for item in value.split(","):
        item = item.strip()
        if item:
            values.append(float(item))
    if not values:
        raise ValueError("--cfg-values is empty")
    return values


def _safe_cfg(value: float) -> str:
    return ("%g" % value).replace(".", "p").replace("-", "m")


def _label_from_ckpt(path: Path) -> str:
    stem = Path(path).stem
    if stem.startswith("denoiser_"):
        stem = stem[len("denoiser_") :]
    return stem


def _rooted(data_root: Path, rel_or_abs: str | Path) -> Path:
    path = Path(rel_or_abs)
    return path if path.is_absolute() else data_root / path


def _make_ss_eval_dataset(cfg: Any, selected: list[Any]) -> SSFlowGlobalZDataset:
    data_cfg = _cfg_dict(cfg.data)
    data_cfg["condition_mode"] = "multiflow_view"
    data_cfg["tokens_subdir"] = "dinov2_tokens_official_prenorm1374"
    data_cfg["ignore_manifest_dinov2_tokens_path"] = True
    data_cfg["test_samples"] = [
        {"obj_id": sample.obj_id, "angle_idx": int(sample.angle_idx)}
        for sample in selected
    ]
    data_cfg.pop("test_obj_ids", None)
    data_cfg.pop("test_obj_ids_file", None)
    data_cfg.pop("exclude_obj_ids", None)
    data_cfg.pop("exclude_obj_ids_file", None)
    data_cfg.pop("max_samples", None)
    return SSFlowGlobalZDataset(data_cfg)


def _load_model(cfg: Any, ckpt_path: Path, device: torch.device) -> SparseStructureFlowModel:
    model_cfg = _cfg_dict(cfg.model)
    model_cfg.pop("name", None)
    args = model_cfg.pop("args", model_cfg)
    model = SparseStructureFlowModel(**args).to(device)
    state = torch.load(_require_file(ckpt_path, "SS-flow checkpoint"), map_location="cpu", weights_only=True)
    missing, unexpected = model.load_state_dict(state, strict=False)
    allowed_missing = {"view_pose_proj.weight", "view_pose_proj.bias"}
    bad_missing = [key for key in missing if key not in allowed_missing]
    if bad_missing:
        raise RuntimeError(f"{ckpt_path} missing keys: {bad_missing[:20]}")
    if unexpected:
        raise RuntimeError(f"{ckpt_path} unexpected keys: {unexpected[:20]}")
    if getattr(model, "use_fp16", False):
        model.convert_to_fp16()
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)
    return model


def _latent_from_sample(sample: dict[str, Any], data_root: Path) -> torch.Tensor:
    path = _rooted(data_root, sample["z_global_rel"])
    with np.load(_require_file(path, "GT z_global latent")) as payload:
        latent_np = np.asarray(payload["mean"], dtype=np.float32)
    if latent_np.shape != (8, 16, 16, 16):
        raise ValueError(f"{path} latent shape {latent_np.shape}, expected (8,16,16,16)")
    return torch.from_numpy(latent_np).float()


def _surface_from_sample(sample: dict[str, Any], data_root: Path) -> np.ndarray:
    rel = sample.get("surface_rel") or sample.get("overall_surface_rel")
    if rel is None:
        rel = f"reconstruction/voxel_expanded/{sample['obj_id']}/angle_{int(sample['angle_idx'])}/64/surface.npy"
    path = _rooted(data_root, rel)
    coords = np.load(_require_file(path, "GT whole surface coords"))
    coords = np.asarray(coords, dtype=np.int64).reshape(-1, 3)
    if coords.size and (coords.min() < 0 or coords.max() > 63):
        raise ValueError(f"{path} coords out of [0,63]")
    return np.unique(coords, axis=0)


def _mask_from_coords(coords: np.ndarray) -> torch.Tensor:
    mask = torch.zeros(64, 64, 64, dtype=torch.bool)
    coords = np.asarray(coords, dtype=np.int64).reshape(-1, 3)
    if coords.size:
        mask[coords[:, 0], coords[:, 1], coords[:, 2]] = True
    return mask


@torch.no_grad()
def _decode_latents(decoder: torch.nn.Module, latents: torch.Tensor, *, threshold: float, device: torch.device) -> torch.Tensor:
    logits = decoder(latents.to(device=device, dtype=torch.float32))[:, 0].detach().float().cpu()
    if tuple(logits.shape[1:]) != (64, 64, 64):
        raise ValueError(f"decoded logits expected [B,64,64,64], got {tuple(logits.shape)}")
    return logits > float(threshold)


def _mask_to_coords(mask: torch.Tensor) -> np.ndarray:
    return torch.nonzero(mask.bool(), as_tuple=False).cpu().numpy().astype(np.int64, copy=False)


def _mask_metrics(pred: torch.Tensor, gt: torch.Tensor) -> dict[str, Any]:
    pred = pred.bool()
    gt = gt.bool()
    intersection = int(torch.logical_and(pred, gt).sum().item())
    union = int(torch.logical_or(pred, gt).sum().item())
    pred_count = int(pred.sum().item())
    gt_count = int(gt.sum().item())
    return {
        "iou": float(intersection / union) if union else 1.0,
        "precision": float(intersection / pred_count) if pred_count else 0.0,
        "recall": float(intersection / gt_count) if gt_count else 0.0,
        "intersection": intersection,
        "pred_count": pred_count,
        "gt_count": gt_count,
    }


def _cos(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(F.cosine_similarity(a.reshape(1, -1).float(), b.reshape(1, -1).float()).item())


def _rel_l2(pred: torch.Tensor, gt: torch.Tensor) -> float:
    numer = torch.linalg.vector_norm((pred - gt).reshape(-1).float()).item()
    denom = max(torch.linalg.vector_norm(gt.reshape(-1).float()).item(), 1.0e-12)
    return float(numer / denom)


def _summary(values: list[float]) -> dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return {"mean": math.nan, "median": math.nan, "min": math.nan, "max": math.nan}
    return {
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "min": float(arr.min()),
        "max": float(arr.max()),
    }


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def _render_table_png(path: Path, rows: list[dict[str, Any]], *, title: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    columns = [
        "split",
        "bucket",
        "cfg_strength",
        "n",
        "mean_latent_cos",
        "mean_pred_vs_gt_decoded_iou",
        "mean_pred_vs_raw_surface_iou",
        "mean_pred_voxels",
    ]
    data = []
    for row in rows:
        values = []
        for col in columns:
            value = row.get(col, "")
            if isinstance(value, float):
                values.append(f"{value:.4f}")
            else:
                values.append(str(value))
        data.append(values)
    fig_h = max(2.6, 0.30 * (len(data) + 2))
    fig_w = 14
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.axis("off")
    ax.set_title(title, fontsize=10, pad=8)
    table = ax.table(cellText=data, colLabels=columns, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(7)
    table.scale(1.0, 1.15)
    for (row_idx, _col_idx), cell in table.get_celld().items():
        if row_idx == 0:
            cell.set_facecolor("#eeeeee")
            cell.set_text_props(weight="bold")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _group_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, float], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = (str(row["split"]), str(row["bucket"]), float(row["cfg_strength"]))
        groups[key].append(row)
        groups[(str(row["split"]), "all", float(row["cfg_strength"]))].append(row)

    out: list[dict[str, Any]] = []
    for split in ("train", "held"):
        cfg_values = sorted({float(row["cfg_strength"]) for row in rows if row["split"] == split})
        for bucket in (*BUCKETS, "all"):
            for cfg_value in cfg_values:
                group = groups.get((split, bucket, cfg_value), [])
                if not group:
                    continue
                out.append(
                    {
                        "split": split,
                        "bucket": bucket,
                        "cfg_strength": float(cfg_value),
                        "n": len(group),
                        "mean_latent_cos": _summary([float(r["latent_cos"]) for r in group])["mean"],
                        "median_latent_cos": _summary([float(r["latent_cos"]) for r in group])["median"],
                        "mean_rel_l2": _summary([float(r["latent_rel_l2"]) for r in group])["mean"],
                        "mean_pred_vs_gt_decoded_iou": _summary([float(r["pred_vs_gt_decoded_iou"]) for r in group])["mean"],
                        "mean_pred_vs_raw_surface_iou": _summary([float(r["pred_vs_raw_surface_iou"]) for r in group])["mean"],
                        "mean_gt_decoded_vs_raw_surface_iou": _summary([float(r["gt_decoded_vs_raw_surface_iou"]) for r in group])["mean"],
                        "mean_pred_voxels": _summary([float(r["pred_voxels"]) for r in group])["mean"],
                        "mean_gt_decoded_voxels": _summary([float(r["gt_decoded_voxels"]) for r in group])["mean"],
                    }
                )
    return out


def _render_preview(
    out_dir: Path,
    *,
    split: str,
    bucket: str,
    obj_id: str,
    angle_idx: int,
    cfg_strength: float,
    gt_coords: np.ndarray,
    pred_coords: np.ndarray,
    metrics: dict[str, Any],
) -> None:
    path = out_dir / "previews" / split / bucket / f"{obj_id}_angle{int(angle_idx):02d}_cfg{_safe_cfg(cfg_strength)}.png"
    render_tripanel(
        gt_coords,
        pred_coords,
        title=f"{split} {bucket} {obj_id} angle_{int(angle_idx)} cfg={cfg_strength:g}",
        metrics={
            "iou": float(metrics["iou"]),
            "precision": float(metrics["precision"]),
            "recall": float(metrics["recall"]),
        },
        out_path=path,
        width=2100,
        height=860,
    )


def _setup_rng(seed: int, device: torch.device) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)


def _build_selection(args: argparse.Namespace) -> tuple[list[Any], dict[str, Any]]:
    dc = load_data_config(Path(args.data_config))
    part_ds = _dataset_for("four", dc)
    selected, manifest = build_selection(part_ds, Path(args.split_json), per_split=int(args.per_split))
    if int(args.limit_samples) > 0:
        selected = selected[: int(args.limit_samples)]
        manifest["limit_samples"] = int(args.limit_samples)
    return selected, manifest


def _build_rows_meta(selected: list[Any]) -> dict[tuple[str, int], dict[str, Any]]:
    return {
        (str(sample.obj_id), int(sample.angle_idx)): {
            "split": sample.split,
            "bucket": sample.bucket,
            "part_count": int(sample.part_count),
            "min_raw_voxels": int(sample.min_raw_voxels),
            "has_button": bool(sample.has_button),
            "forced_reason": sample.forced_reason,
        }
        for sample in selected
    }


def run_one(args: argparse.Namespace, *, out_dir: Path, ckpt_path: Path, ckpt_label: str) -> None:
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{args.gpu}")
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")
    _setup_rng(int(args.seed), device)

    cfg = load_config(str(args.config))
    selected, selection_manifest = _build_selection(args)
    selected_meta = _build_rows_meta(selected)
    ss_dataset = _make_ss_eval_dataset(cfg, selected)
    data_root = Path(_cfg_dict(cfg.data)["data_root"])
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_json(
        out_dir / "selection.json",
        {
            **selection_manifest,
            "samples_flat": [asdict(sample) for sample in selected],
            "ss_flow_dataset_count": len(ss_dataset),
        },
    )
    print(f"[eval_ss_flow_ema] {ckpt_label} selected={len(selected)} out={out_dir}", flush=True)

    if bool(args.plan_only):
        return

    decoder_ckpt = _resolve_path(args.ss_decoder_ckpt)
    decoder = load_ss_decoder(_require_file(decoder_ckpt, "SS decoder checkpoint"), device=device)
    model = _load_model(cfg, ckpt_path, device)
    sampler = BatchedFlowEulerCfgSampler(sigma_min=float(_cfg_dict(cfg.training).get("sigma_min", 1.0e-5)))
    sampler_model = MultiflowWrapper(model)

    loader = DataLoader(
        ss_dataset,
        batch_size=int(args.batch_size),
        shuffle=False,
        num_workers=0,
        collate_fn=ss_dataset.collate_fn,
    )

    gt_latents = [_latent_from_sample(sample, data_root) for sample in ss_dataset.samples]
    raw_coords = [_surface_from_sample(sample, data_root) for sample in ss_dataset.samples]
    raw_masks = [_mask_from_coords(coords) for coords in raw_coords]
    gt_latents_tensor = torch.stack(gt_latents, dim=0)
    gt_masks = _decode_latents(decoder, gt_latents_tensor, threshold=float(args.decode_threshold), device=device)
    gt_vs_raw = [_mask_metrics(gt_masks[idx], raw_masks[idx]) for idx in range(len(raw_masks))]

    cfg_values = _parse_cfg_values(args.cfg_values)
    rows: list[dict[str, Any]] = []
    records: list[dict[str, Any]] = []
    started = time.time()
    peak_vram_mib = 0
    preview_remaining_by_bucket: dict[tuple[str, str, float], int] = defaultdict(lambda: int(args.previews_per_bucket))
    with VramSampler(str(args.gpu)) as vram:
        for cfg_value in cfg_values:
            noise_generator = torch.Generator(device=device)
            noise_generator.manual_seed(int(args.seed))
            collected = 0
            print(f"[eval_ss_flow_ema] {ckpt_label} cfg={cfg_value:g} start", flush=True)
            for batch_idx, batch in enumerate(loader):
                take = min(len(ss_dataset) - collected, int(batch["x_0"].shape[0]))
                if take <= 0:
                    break
                cond = batch["cond"][:take].to(device, non_blocking=True).float()
                noise = torch.randn(
                    (take, 8, 16, 16, 16),
                    dtype=torch.float32,
                    device=device,
                    generator=noise_generator,
                )
                with torch.no_grad():
                    result = sampler.sample(
                        model=sampler_model,
                        noise=noise,
                        cond=cond,
                        neg_cond=torch.zeros_like(cond),
                        steps=int(args.steps),
                        cfg_strength=float(cfg_value),
                        verbose=False,
                    )
                pred_latents = result.samples.detach().float().cpu()
                pred_masks = _decode_latents(
                    decoder,
                    pred_latents,
                    threshold=float(args.decode_threshold),
                    device=device,
                )
                for local_idx in range(take):
                    row_idx = collected + local_idx
                    sample = ss_dataset.samples[row_idx]
                    obj_id = str(sample["obj_id"])
                    angle_idx = int(sample["angle_idx"])
                    meta = selected_meta[(obj_id, angle_idx)]
                    pred_vs_gt = _mask_metrics(pred_masks[local_idx], gt_masks[row_idx])
                    pred_vs_raw = _mask_metrics(pred_masks[local_idx], raw_masks[row_idx])
                    pred_latent = pred_latents[local_idx]
                    gt_latent = gt_latents[row_idx]
                    row = {
                        "ckpt_label": ckpt_label,
                        "ckpt_path": str(ckpt_path),
                        "cfg_strength": float(cfg_value),
                        "split": meta["split"],
                        "obj_id": obj_id,
                        "angle": angle_idx,
                        "bucket": meta["bucket"],
                        "part_count": int(meta["part_count"]),
                        "min_raw_voxels": int(meta["min_raw_voxels"]),
                        "has_button": int(meta["has_button"]),
                        "view_indices": json.dumps([int(v) for v in sample["view_indices"]]),
                        "latent_cos": _cos(pred_latent, gt_latent),
                        "latent_rel_l2": _rel_l2(pred_latent, gt_latent),
                        "pred_vs_gt_decoded_iou": float(pred_vs_gt["iou"]),
                        "pred_vs_gt_decoded_precision": float(pred_vs_gt["precision"]),
                        "pred_vs_gt_decoded_recall": float(pred_vs_gt["recall"]),
                        "pred_vs_raw_surface_iou": float(pred_vs_raw["iou"]),
                        "pred_vs_raw_surface_precision": float(pred_vs_raw["precision"]),
                        "pred_vs_raw_surface_recall": float(pred_vs_raw["recall"]),
                        "gt_decoded_vs_raw_surface_iou": float(gt_vs_raw[row_idx]["iou"]),
                        "gt_decoded_vs_raw_surface_precision": float(gt_vs_raw[row_idx]["precision"]),
                        "gt_decoded_vs_raw_surface_recall": float(gt_vs_raw[row_idx]["recall"]),
                        "pred_voxels": int(pred_vs_gt["pred_count"]),
                        "gt_decoded_voxels": int(pred_vs_gt["gt_count"]),
                        "raw_surface_voxels": int(raw_masks[row_idx].sum().item()),
                    }
                    rows.append(row)
                    preview_key = (meta["split"], meta["bucket"], float(cfg_value))
                    if preview_remaining_by_bucket[preview_key] > 0:
                        pred_coords = _mask_to_coords(pred_masks[local_idx])
                        _render_preview(
                            out_dir,
                            split=meta["split"],
                            bucket=meta["bucket"],
                            obj_id=obj_id,
                            angle_idx=angle_idx,
                            cfg_strength=float(cfg_value),
                            gt_coords=raw_coords[row_idx],
                            pred_coords=pred_coords,
                            metrics=pred_vs_raw,
                        )
                        preview_remaining_by_bucket[preview_key] -= 1
                collected += take
                if collected >= len(ss_dataset) or (batch_idx + 1) % max(1, int(args.progress_every)) == 0:
                    cfg_rows = [row for row in rows if float(row["cfg_strength"]) == float(cfg_value)]
                    print(
                        f"[eval_ss_flow_ema] {ckpt_label} cfg={cfg_value:g} "
                        f"progress={collected}/{len(ss_dataset)} "
                        f"mean_cos={np.mean([r['latent_cos'] for r in cfg_rows]):.4f} "
                        f"mean_raw_iou={np.mean([r['pred_vs_raw_surface_iou'] for r in cfg_rows]):.4f}",
                        flush=True,
                    )
                if collected >= len(ss_dataset):
                    break
            records.append(
                {
                    "ckpt_label": ckpt_label,
                    "cfg_strength": float(cfg_value),
                    "completed": int(collected),
                    "seconds_since_start": round(time.time() - started, 3),
                }
            )
        peak_vram_mib = int(vram.max_mib)

    summary_rows = _group_summary(rows)
    fieldnames = [
        "ckpt_label",
        "cfg_strength",
        "split",
        "obj_id",
        "angle",
        "bucket",
        "part_count",
        "min_raw_voxels",
        "has_button",
        "view_indices",
        "latent_cos",
        "latent_rel_l2",
        "pred_vs_gt_decoded_iou",
        "pred_vs_gt_decoded_precision",
        "pred_vs_gt_decoded_recall",
        "pred_vs_raw_surface_iou",
        "pred_vs_raw_surface_precision",
        "pred_vs_raw_surface_recall",
        "gt_decoded_vs_raw_surface_iou",
        "gt_decoded_vs_raw_surface_precision",
        "gt_decoded_vs_raw_surface_recall",
        "pred_voxels",
        "gt_decoded_voxels",
        "raw_surface_voxels",
        "ckpt_path",
    ]
    summary_fields = [
        "split",
        "bucket",
        "cfg_strength",
        "n",
        "mean_latent_cos",
        "median_latent_cos",
        "mean_rel_l2",
        "mean_pred_vs_gt_decoded_iou",
        "mean_pred_vs_raw_surface_iou",
        "mean_gt_decoded_vs_raw_surface_iou",
        "mean_pred_voxels",
        "mean_gt_decoded_voxels",
    ]
    _write_csv(out_dir / "metrics.csv", rows, fieldnames)
    _write_csv(out_dir / "metrics_summary.csv", summary_rows, summary_fields)
    _write_json(out_dir / "metrics.json", rows)
    _write_json(out_dir / "metrics_summary.json", summary_rows)
    _render_table_png(out_dir / "metrics_summary.png", summary_rows, title=f"{ckpt_label} SS-flow whole voxel summary")

    best = max(
        [row for row in summary_rows if row["bucket"] == "all"],
        key=lambda row: (float(row["mean_pred_vs_raw_surface_iou"]), float(row["mean_latent_cos"])),
    )
    run_meta = {
        "entry": "python -m part_ss_eval_platform.eval_ss_flow_ema_0615",
        "config": str(Path(args.config).resolve()),
        "ckpt_label": ckpt_label,
        "ckpt_path": str(ckpt_path),
        "out_dir": str(out_dir),
        "condition_dropout": {
            "training_p_uncond": float(_cfg_dict(cfg.training).get("p_uncond", 0.0)),
            "cfg_formula": "(1+cfg_strength)*pred - cfg_strength*uncond",
            "negative_condition": "zero DINO token tensor",
            "cfg_values": cfg_values,
        },
        "condition_mode": {
            "training": str(_cfg_dict(cfg.data).get("condition_mode")),
            "eval": "multiflow_view",
            "inference_rule": "4 physical view tokens -> per-view denoiser velocity -> mean velocity per Euler step",
            "tokens_subdir": "dinov2_tokens_official_prenorm1374",
        },
        "sample_count": len(ss_dataset),
        "steps": int(args.steps),
        "decode_threshold": float(args.decode_threshold),
        "peak_vram_mib_sampled": peak_vram_mib,
        "seconds": round(time.time() - started, 3),
        "best_all_row_by_raw_surface_iou": best,
        "records": records,
    }
    _write_json(out_dir / "run_meta.json", run_meta)
    (out_dir / "peak_vram.txt").write_text(f"{peak_vram_mib}\n", encoding="utf-8")
    print(
        f"[eval_ss_flow_ema] {ckpt_label} done best_cfg={best['cfg_strength']:g} "
        f"best_all_raw_iou={best['mean_pred_vs_raw_surface_iou']:.4f} "
        f"best_all_cos={best['mean_latent_cos']:.4f} peak_vram={peak_vram_mib}MiB",
        flush=True,
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="0615 SS-flow bare/EMA CFG whole-voxel eval")
    p.add_argument("--config", default=str(DEFAULT_CONFIG))
    p.add_argument("--split-json", default=str(DEFAULT_SPLIT_JSON))
    p.add_argument("--data-config", default=str(DEFAULT_DATA_CONFIG))
    p.add_argument("--ss-decoder-ckpt", default=str(DEFAULT_SS_DECODER_CKPT))
    p.add_argument("--bare-ckpt", default=str(DEFAULT_BARE_CKPT))
    p.add_argument("--ema-ckpt", default=str(DEFAULT_EMA0999_CKPT))
    p.add_argument("--bare-label", default="")
    p.add_argument("--ema-label", default="")
    p.add_argument("--bare-out-dir", default=str(DEFAULT_OUT_ROOT / "0615-ss-ema1"))
    p.add_argument("--ema-out-dir", default=str(DEFAULT_OUT_ROOT / "0615-ss-ema2"))
    p.add_argument("--which", choices=("bare", "ema", "both"), default="both")
    p.add_argument("--per-split", type=int, default=64)
    p.add_argument("--limit-samples", type=int, default=0)
    p.add_argument("--cfg-values", default="0,3,5")
    p.add_argument("--steps", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--decode-threshold", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--gpu", default="0")
    p.add_argument("--progress-every", type=int, default=8)
    p.add_argument("--previews-per-bucket", type=int, default=1)
    p.add_argument("--plan-only", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    _require_file(Path(args.config), "SS-flow eval config")
    _require_file(Path(args.split_json), "split json")
    _require_file(Path(args.data_config), "part eval data config")
    _require_file(_resolve_path(args.ss_decoder_ckpt), "SS decoder ckpt")
    jobs = []
    if args.which in ("bare", "both"):
        bare_ckpt = Path(args.bare_ckpt)
        jobs.append((str(args.bare_label or _label_from_ckpt(bare_ckpt)), bare_ckpt, Path(args.bare_out_dir)))
    if args.which in ("ema", "both"):
        ema_ckpt = Path(args.ema_ckpt)
        jobs.append((str(args.ema_label or _label_from_ckpt(ema_ckpt)), ema_ckpt, Path(args.ema_out_dir)))
    for label, ckpt, out_dir in jobs:
        _require_file(ckpt, f"{label} checkpoint")
        run_one(args, out_dir=out_dir, ckpt_path=ckpt, ckpt_label=label)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
