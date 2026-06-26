#!/usr/bin/env python3
"""Trainer for 4-view token conditioned whole-object SS latent flow."""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import os
import random
from pathlib import Path
from typing import Any, Dict

_HERE = Path(__file__).resolve()
PROJECT_ROOT = _HERE.parents[4]

os.environ.setdefault("TORCH_HOME", str(PROJECT_ROOT / "submodules" / "TRELLIS.1"))
os.environ.setdefault("ATTN_BACKEND", "sdpa")

import numpy as np
import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data import DataLoader
from easydict import EasyDict as edict

from trellis.datasets.arts.ss_flow_global_z import SSFlowGlobalZDataset
from trellis.models.sparse_structure_flow import SparseStructureFlowModel
from trellis.models.sparse_structure_vae import SparseStructureDecoder
from trellis.trainers.arts.mixins.wandb import WandbMixin
from trellis.trainers.flow_matching.flow_matching import ImageConditionedFlowMatchingCFGTrainer
from trellis.trainers.flow_matching.mixins.classifier_free_guidance import ClassifierFreeGuidanceMixin
from trellis.utils.arts.anchor_utils import L2SPAnchor
from trellis.utils.arts.config_utils import config_to_dict, load_config
from trellis.utils.arts.ddp_utils import setup_ddp


class SSFlowGlobalZTrainer(WandbMixin, ImageConditionedFlowMatchingCFGTrainer):
    """Dense SS flow trainer that consumes already encoded DINOv2 tokens."""

    def __init__(
        self,
        *args,
        snapshot_config: dict | None = None,
        snapshot_source_dataset: SSFlowGlobalZDataset | None = None,
        fusion_mode: str = "multidiffusion",
        **kwargs,
    ):
        self.snapshot_config = snapshot_config or {}
        self.snapshot_source_dataset = snapshot_source_dataset
        self._ss_decoder = None
        self.fusion_mode = self._normalize_fusion_mode(fusion_mode)
        super().__init__(*args, **kwargs)

    def get_cond(self, cond, **kwargs):
        neg_cond = torch.zeros_like(cond)
        return ClassifierFreeGuidanceMixin.get_cond(self, cond, neg_cond=neg_cond, **kwargs)

    def get_inference_cond(self, cond, **kwargs):
        neg_cond = torch.zeros_like(cond)
        return ClassifierFreeGuidanceMixin.get_inference_cond(self, cond, neg_cond=neg_cond, **kwargs)

    def _is_multiflow_cond(self, cond: torch.Tensor | None) -> bool:
        return isinstance(cond, torch.Tensor) and cond.ndim == 4

    @staticmethod
    def _normalize_fusion_mode(fusion_mode: str | None) -> str:
        mode = str(fusion_mode or "multidiffusion").lower()
        aliases = {
            "multi": "multidiffusion",
            "multiflow": "multidiffusion",
            "avg": "multidiffusion",
            "average": "multidiffusion",
            "concat4": "concat",
            "concat_view": "concat",
            "concat_views": "concat",
        }
        mode = aliases.get(mode, mode)
        if mode not in {"multidiffusion", "concat"}:
            raise ValueError(f"unsupported fusion_mode={fusion_mode!r}; expected 'multidiffusion' or 'concat'")
        return mode

    @classmethod
    def predict_multiview(
        cls,
        model: torch.nn.Module,
        x_t: torch.Tensor,
        t: torch.Tensor,
        cond: torch.Tensor,
        *,
        fusion_mode: str = "multidiffusion",
        **kwargs,
    ) -> torch.Tensor:
        mode = cls._normalize_fusion_mode(fusion_mode)
        if not isinstance(cond, torch.Tensor) or cond.ndim != 4:
            return model(x_t, t, cond, **kwargs)
        if cond.shape[0] != x_t.shape[0]:
            raise ValueError(f"multiview cond batch {cond.shape[0]} does not match x_t batch {x_t.shape[0]}")
        if mode == "concat":
            return model(x_t, t, cond, **kwargs)

        bsz, num_views, token_count, token_dim = cond.shape
        cond_flat = cond.reshape(bsz * num_views, token_count, token_dim).contiguous()
        x_flat = x_t[:, None].expand(-1, num_views, -1, -1, -1, -1).reshape(
            bsz * num_views, *x_t.shape[1:]
        ).contiguous()
        if isinstance(t, torch.Tensor):
            if t.ndim == 0:
                t_flat = t.reshape(1).expand(bsz * num_views).contiguous()
            else:
                t_flat = t[:, None].expand(-1, num_views).reshape(bsz * num_views).contiguous()
        else:
            t_flat = t
        pred_flat = model(x_flat, t_flat, cond_flat, **kwargs)
        return pred_flat.reshape(bsz, num_views, *x_t.shape[1:]).mean(dim=1)

    def training_losses(
        self,
        x_0: torch.Tensor,
        cond=None,
        **kwargs,
    ):
        if not self._is_multiflow_cond(cond):
            return super().training_losses(x_0=x_0, cond=cond, **kwargs)

        if cond.shape[0] != x_0.shape[0]:
            raise ValueError(f"multiflow cond batch {cond.shape[0]} does not match x_0 batch {x_0.shape[0]}")
        bsz = int(cond.shape[0])
        num_views = int(cond.shape[1])
        noise = torch.randn_like(x_0)
        t = self.sample_t(bsz).to(x_0.device).float()
        x_t = self.diffuse(x_0, t, noise=noise)
        target = self.get_v(x_0, noise, t)

        if self.p_uncond > 0:
            drop = torch.rand(bsz, device=cond.device) < float(self.p_uncond)
            cond = torch.where(drop.view(bsz, 1, 1, 1), torch.zeros_like(cond), cond)

        pred = self.predict_multiview(
            self.training_models["denoiser"],
            x_t,
            t * 1000,
            cond,
            fusion_mode=self.fusion_mode,
            **kwargs,
        )
        if pred.shape != target.shape:
            raise RuntimeError(f"multiflow pred shape {tuple(pred.shape)} != target {tuple(target.shape)}")

        terms = edict()
        terms["mse"] = F.mse_loss(pred, target)
        terms["loss"] = terms["mse"]
        terms["view_count"] = torch.tensor(float(num_views), device=x_0.device)
        terms["fusion_mode_concat"] = torch.tensor(float(self.fusion_mode == "concat"), device=x_0.device)

        mse_per_instance = np.array([
            F.mse_loss(pred[i], target[i]).item()
            for i in range(bsz)
        ])
        time_bin = np.digitize(t.detach().cpu().numpy(), np.linspace(0, 1, 11)) - 1
        for i in range(10):
            if (time_bin == i).sum() != 0:
                terms[f"bin_{i}"] = {"mse": mse_per_instance[time_bin == i].mean()}
        return terms, {}

    def prepare_dataloader(self, *, num_workers: int | None = None, **kwargs):
        from trellis.utils.data_utils import ResumableSampler, cycle

        self.data_sampler = ResumableSampler(
            self.dataset,
            shuffle=True,
        )
        if num_workers is None:
            worker_count = int(np.ceil(os.cpu_count() / torch.cuda.device_count()))
        else:
            worker_count = int(num_workers)
        if worker_count < 0:
            raise ValueError(f"num_workers must be >= 0, got {worker_count}")
        self.dataloader = DataLoader(
            self.dataset,
            batch_size=self.batch_size_per_gpu,
            num_workers=worker_count,
            pin_memory=True,
            drop_last=True,
            persistent_workers=worker_count > 0,
            collate_fn=self.dataset.collate_fn if hasattr(self.dataset, "collate_fn") else None,
            sampler=self.data_sampler,
        )
        self.data_iterator = cycle(self.dataloader)

    def snapshot_dataset(self, num_samples=100):
        if self.is_master:
            print("[SSFlowGlobalZTrainer] snapshot_dataset skipped (pre-encoded token dataset)")

    def _load_snapshot_decoder(self):
        if self._ss_decoder is None:
            ckpt_path = self.snapshot_config.get("ss_decoder_ckpt")
            if not ckpt_path:
                raise ValueError("snapshot.ss_decoder_ckpt is required for global-z decode snapshots")
            self._ss_decoder = load_ss_decoder(_resolve_path(ckpt_path), device=self.device)
        return self._ss_decoder

    @staticmethod
    def _snapshot_barrier() -> None:
        if dist.is_available() and dist.is_initialized():
            dist.barrier()

    def _select_ema_params_for_snapshot(self):
        if not bool(self.snapshot_config.get("use_ema", False)):
            return None
        if not hasattr(self, "ema_params"):
            raise RuntimeError("snapshot.use_ema=true but trainer has no ema_params")
        requested = float(self.snapshot_config.get("ema_rate", self.ema_rate[0]))
        for idx, ema_rate in enumerate(self.ema_rate):
            if abs(float(ema_rate) - requested) < 1.0e-12:
                return self.ema_params[idx]
        raise RuntimeError(f"snapshot ema_rate={requested} not found in trainer ema rates {self.ema_rate}")

    def snapshot(self, suffix=None, num_samples=64, batch_size=4, verbose=False):
        suffix = suffix or f"step{self.step:07d}"
        if not bool(self.snapshot_config.get("enabled", True)):
            if self.is_master:
                print(f"[SSFlowGlobalZTrainer] snapshot({suffix}) skipped (snapshot.enabled=false)")
            self._snapshot_barrier()
            return
        if not self.is_master:
            self._snapshot_barrier()
            return

        source_dataset = self.snapshot_source_dataset or self.dataset
        sample_count = max(1, min(int(self.snapshot_config.get("num_samples", 1)), len(source_dataset)))
        snapshot_batch_size = max(1, min(int(self.snapshot_config.get("batch_size", batch_size)), sample_count))
        steps = int(self.snapshot_config.get("num_steps", 20))
        cfg_strength = float(self.snapshot_config.get("cfg_strength", 3.0))
        threshold = float(self.snapshot_config.get("decode_threshold", 0.0))
        verbose_sampling = bool(self.snapshot_config.get("verbose", False)) or bool(verbose)
        snapshot_seed = self.snapshot_config.get("seed")
        noise_generator = None
        if snapshot_seed not in (None, "", "null"):
            noise_generator = torch.Generator(device=self.device)
            noise_generator.manual_seed(int(snapshot_seed))

        dataloader = DataLoader(
            source_dataset,
            batch_size=snapshot_batch_size,
            shuffle=False,
            num_workers=0,
            collate_fn=source_dataset.collate_fn,
        )
        raw_model = self.models["denoiser"]
        original_state = None
        ema_params = self._select_ema_params_for_snapshot()
        if ema_params is not None:
            original_state = {
                key: value.detach().cpu().clone()
                for key, value in raw_model.state_dict().items()
            }
            raw_model.load_state_dict(self._master_params_to_state_dicts(ema_params)["denoiser"])
        raw_model.eval()
        class _MultiflowWrapper(torch.nn.Module):
            def __init__(self, model: torch.nn.Module, fusion_mode: str):
                super().__init__()
                self.model = model
                self.fusion_mode = fusion_mode

            def forward(self, x_t, t, cond, **kwargs):
                return SSFlowGlobalZTrainer.predict_multiview(
                    self.model,
                    x_t,
                    t,
                    cond,
                    fusion_mode=self.fusion_mode,
                    **kwargs,
                )

        try:
            sampler = self.get_sampler()
            sampler_model = (
                _MultiflowWrapper(raw_model, self.fusion_mode)
                if getattr(source_dataset, "condition_mode", "") == "multiflow_view"
                else raw_model
            )
            gt_batches = []
            pred_batches = []
            collected = 0
            for batch in dataloader:
                take = min(sample_count - collected, int(batch["x_0"].shape[0]))
                if take <= 0:
                    break
                data = {
                    key: value[:take].to(self.device, non_blocking=True)
                    for key, value in batch.items()
                }
                noise = torch.randn(
                    data["x_0"].shape,
                    dtype=data["x_0"].dtype,
                    device=data["x_0"].device,
                    generator=noise_generator,
                )
                inference_cond = self.get_inference_cond(data["cond"], cam_pose=data.get("cam_pose"))
                result = sampler.sample(
                    sampler_model,
                    noise=noise,
                    steps=steps,
                    cfg_strength=cfg_strength,
                    verbose=verbose_sampling,
                    **inference_cond,
                )
                pred_batches.append(result.samples.detach().float().cpu())
                gt_batches.append(data["x_0"].detach().float().cpu())
                collected += take
                if collected >= sample_count:
                    break
            if collected != sample_count:
                raise RuntimeError(f"snapshot collected {collected} samples, expected {sample_count}")
            pred = torch.cat(pred_batches, dim=0)
            gt = torch.cat(gt_batches, dim=0)
        finally:
            if original_state is not None:
                raw_model.load_state_dict(original_state)
            raw_model.train()

        decoder = self._load_snapshot_decoder()
        out_dir = Path(self.output_dir) / "samples" / suffix
        out_dir.mkdir(parents=True, exist_ok=True)
        png_path = out_dir / f"global_z_decode_{suffix}.png"
        json_path = out_dir / f"global_z_decode_{suffix}.json"
        summary = write_global_z_snapshot(
            decoder,
            gt=gt,
            pred=pred,
            out_png=png_path,
            threshold=threshold,
            sample_meta=source_dataset.samples[:sample_count],
            render_samples=self.snapshot_config.get("render_samples"),
        )
        json_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        print(f"[SSFlowGlobalZTrainer] snapshot({suffix}) wrote {png_path}")
        self._snapshot_barrier()


def _cfg_dict(cfg) -> dict:
    return config_to_dict(cfg) if not isinstance(cfg, dict) else dict(cfg)


def _setup_rng(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _resolve_path(path: str | os.PathLike[str]) -> Path:
    resolved = Path(path)
    return resolved if resolved.is_absolute() else PROJECT_ROOT / resolved


def _load_state_dict(path: Path) -> Dict[str, torch.Tensor]:
    if not path.is_file():
        raise FileNotFoundError(f"pretrained SS flow checkpoint not found: {path}")
    if path.suffix == ".safetensors":
        from safetensors.torch import load_file

        return load_file(str(path))
    return torch.load(path, map_location="cpu", weights_only=True)


def _decoder_args_from_yaml(path: Path) -> dict:
    from omegaconf import OmegaConf

    cfg = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
    if not isinstance(cfg, dict):
        raise ValueError(f"decoder yaml must contain a mapping: {path}")
    allowed = set(inspect.signature(SparseStructureDecoder.__init__).parameters)
    allowed.discard("self")
    return {key: value for key, value in cfg.items() if key in allowed}


def _decoder_args_from_json(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if "args" in payload:
        return dict(payload["args"])
    if "models" in payload and "decoder" in payload["models"]:
        return dict(payload["models"]["decoder"]["args"])
    raise KeyError(f"{path} must contain either args or models.decoder.args")


def load_ss_decoder(ckpt_path: Path, *, device: torch.device) -> SparseStructureDecoder:
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"SS decoder checkpoint not found: {ckpt_path}")

    if ckpt_path.suffix == ".ckpt":
        config_path = ckpt_path.with_suffix(".yaml")
        if not config_path.is_file():
            raise FileNotFoundError(f"SS decoder yaml not found: {config_path}")
        args = _decoder_args_from_yaml(config_path)
        state = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    elif ckpt_path.suffix == ".safetensors":
        from safetensors.torch import load_file

        config_path = ckpt_path.with_suffix(".json")
        if not config_path.is_file():
            raise FileNotFoundError(f"SS decoder json not found: {config_path}")
        args = _decoder_args_from_json(config_path)
        state = load_file(str(ckpt_path))
    else:
        raise ValueError(f"unsupported SS decoder checkpoint suffix: {ckpt_path}")

    decoder = SparseStructureDecoder(**args).to(device).eval()
    decoder.load_state_dict(state, strict=True)
    for param in decoder.parameters():
        param.requires_grad_(False)
    return decoder


@torch.no_grad()
def decode_latent_to_coords(
    decoder: SparseStructureDecoder,
    latent: torch.Tensor,
    *,
    threshold: float,
) -> tuple[np.ndarray, Dict[str, Any]]:
    if tuple(latent.shape) != (8, 16, 16, 16):
        raise ValueError(f"latent expected shape (8,16,16,16), got {tuple(latent.shape)}")
    device = next(decoder.parameters()).device
    logits = decoder(latent.unsqueeze(0).to(device=device))[0, 0].detach().float().cpu()
    coords = torch.nonzero(logits > float(threshold), as_tuple=False).long().numpy()
    flat = logits.reshape(-1)
    stats = {
        "count": int(coords.shape[0]),
        "logit_min": float(flat.min().item()),
        "logit_max": float(flat.max().item()),
        "logit_mean": float(flat.mean().item()),
        "count_threshold": float(threshold),
    }
    return coords, stats


def coords_iou(a: np.ndarray, b: np.ndarray) -> float:
    if a.size == 0 and b.size == 0:
        return 1.0
    if a.size == 0 or b.size == 0:
        return 0.0
    a_keys = {
        int(x) * 4096 + int(y) * 64 + int(z)
        for x, y, z in a.astype(np.int64, copy=False)
    }
    b_keys = {
        int(x) * 4096 + int(y) * 64 + int(z)
        for x, y, z in b.astype(np.int64, copy=False)
    }
    union = len(a_keys | b_keys)
    return float(len(a_keys & b_keys) / union) if union > 0 else 1.0


def _plot_coords(ax, coords: np.ndarray, title: str, color: str) -> None:
    ax.set_title(title, fontsize=9)
    ax.set_xlim(0, 63)
    ax.set_ylim(0, 63)
    ax.set_zlim(0, 63)
    ax.set_box_aspect((1, 1, 1))
    ax.view_init(elev=24, azim=-58)
    ax.tick_params(axis="both", which="major", labelsize=6, pad=-2)
    if coords.size == 0:
        ax.text2D(0.38, 0.48, "empty", transform=ax.transAxes, color="#b00020", fontsize=10)
        return
    max_points = 9000
    points = coords
    if points.shape[0] > max_points:
        idx = np.linspace(0, points.shape[0] - 1, max_points, dtype=np.int64)
        points = points[idx]
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=3, c=color, alpha=0.72, depthshade=False)


def write_global_z_snapshot(
    decoder: SparseStructureDecoder,
    *,
    gt: torch.Tensor,
    pred: torch.Tensor,
    out_png: Path,
    threshold: float,
    sample_meta: list[dict[str, Any]],
    render_samples: int | None = None,
) -> Dict[str, Any]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    count = min(int(gt.shape[0]), int(pred.shape[0]), len(sample_meta))
    if count <= 0:
        raise ValueError("snapshot has no samples to render")
    render_count = count if render_samples is None else max(0, min(int(render_samples), count))
    fig = plt.figure(figsize=(8.8, 3.8 * max(1, render_count)), dpi=140)
    summary: Dict[str, Any] = {
        "threshold": float(threshold),
        "rendered_samples": int(render_count),
        "samples": [],
        "gt_vs_gt_iou_matrix": [],
        "sample_vs_sample_iou_matrix": [],
        "pred_vs_gt_iou_matrix": [],
        "png_path": str(out_png),
    }
    gt_coords_list = []
    pred_coords_list = []
    for row in range(count):
        gt_coords, gt_stats = decode_latent_to_coords(decoder, gt[row], threshold=threshold)
        pred_coords, pred_stats = decode_latent_to_coords(decoder, pred[row], threshold=threshold)
        own_iou = coords_iou(pred_coords, gt_coords)
        gt_coords_list.append(gt_coords)
        pred_coords_list.append(pred_coords)
        meta = sample_meta[row]
        obj_id = str(meta.get("obj_id", row))
        angle_idx = int(meta.get("angle_idx", 0))
        if row < render_count:
            ax = fig.add_subplot(render_count, 2, row * 2 + 1, projection="3d")
            _plot_coords(ax, gt_coords, f"GT {obj_id} angle_{angle_idx}\nvoxels={gt_stats['count']}", "#1f77b4")
            ax = fig.add_subplot(render_count, 2, row * 2 + 2, projection="3d")
            _plot_coords(ax, pred_coords, f"Sample {obj_id} angle_{angle_idx}\nvoxels={pred_stats['count']}", "#d62728")
        summary["samples"].append({
            "obj_id": obj_id,
            "angle_idx": angle_idx,
            "gt": gt_stats,
            "sample": pred_stats,
            "sample_vs_gt_iou": own_iou,
        })
    summary["pred_vs_gt_iou_matrix"] = [
        [coords_iou(pred_coords_list[i], gt_coords_list[j]) for j in range(count)]
        for i in range(count)
    ]
    summary["gt_vs_gt_iou_matrix"] = [
        [coords_iou(gt_coords_list[i], gt_coords_list[j]) for j in range(count)]
        for i in range(count)
    ]
    summary["sample_vs_sample_iou_matrix"] = [
        [coords_iou(pred_coords_list[i], pred_coords_list[j]) for j in range(count)]
        for i in range(count)
    ]
    if render_count > 0:
        fig.tight_layout()
        fig.savefig(out_png)
    plt.close(fig)
    return summary


def _load_pretrained(model: SparseStructureFlowModel, pretrained_ckpt: str | None, *, rank: int) -> None:
    if not pretrained_ckpt:
        raise ValueError("training.pretrained_ckpt is required for ss_flow_global_z fine-tuning")
    ckpt_path = _resolve_path(pretrained_ckpt)
    state = _load_state_dict(ckpt_path)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if rank == 0:
        print(f"[SSFlowGlobalZ] loaded pretrained SS flow: {ckpt_path}")
        print(f"  missing keys: {len(missing)}")
        print(f"  unexpected keys: {len(unexpected)}")
    allowed_missing = {"view_pose_proj.weight", "view_pose_proj.bias", "view_id_embedding.weight"}
    bad_missing = [key for key in missing if key not in allowed_missing]
    if bad_missing:
        raise RuntimeError(f"pretrained SS flow load has missing model keys: {bad_missing[:20]}")
    if unexpected:
        raise RuntimeError(f"pretrained SS flow load has unexpected model keys: {unexpected[:20]}")
    if missing and rank == 0:
        print(f"  allowed new condition keys initialized from model init: {list(missing)}")


def _build_model(model_cfg: Dict[str, Any], *, device: torch.device) -> SparseStructureFlowModel:
    cfg = dict(model_cfg)
    cfg.pop("name", None)
    args = cfg.pop("args", cfg)
    return SparseStructureFlowModel(**args).to(device)


def _apply_lora_if_configured(model, cfg, *, rank: int):
    lora_cfg = _cfg_dict(cfg.lora) if "lora" in cfg else {}
    if not bool(lora_cfg.get("enabled", False)):
        return model
    from trellis.utils.arts.lora_utils import apply_lora_to_model

    keep_trainable = list(lora_cfg.get("keep_trainable") or [])
    if "view_pose_proj" not in keep_trainable:
        keep_trainable.append("view_pose_proj")
    if "view_id_embedding" not in keep_trainable:
        keep_trainable.append("view_id_embedding")
    lora_cfg["keep_trainable"] = keep_trainable
    model = apply_lora_to_model(model, lora_cfg)
    if rank == 0:
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        print(
            f"[SSFlowGlobalZ] LoRA enabled: trainable={trainable:,} / {total:,} "
            f"({trainable / total * 100:.2f}%)"
        )
    return model


def _attach_anchor_if_configured(model, cfg, pretrained_ckpt: str | None, *, rank: int):
    anchor_cfg = _cfg_dict(cfg.anchor) if "anchor" in cfg else {}
    if not bool(anchor_cfg.get("enabled", False)):
        return None
    if not pretrained_ckpt:
        raise ValueError("anchor.enabled=True requires training.pretrained_ckpt")
    state = _load_state_dict(_resolve_path(pretrained_ckpt))
    anchor = L2SPAnchor.from_state_dict(
        model,
        state,
        lambda_=float(anchor_cfg.get("lambda", 1.0e-4)),
        target=str(anchor_cfg.get("target", "trainable")),
    )
    anchor.attach()
    if rank == 0:
        print(f"[SSFlowGlobalZ] L2-SP anchor enabled: lambda={anchor.lambda_} target={anchor.target}")
    return anchor


def train(
    config,
    *,
    load_dir: str | None = None,
    resume_step: int | None = None,
    dump_param_stats: bool = False,
) -> None:
    cfg = config
    rank, local_rank, world_size = setup_ddp()
    is_distributed = world_size > 1
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    seed = int(getattr(cfg.training, "seed", 42))
    _setup_rng(seed + rank)

    if rank == 0:
        print("\n[SSFlowGlobalZ] config loaded:")
        print(f"  distributed={is_distributed} world_size={world_size} device={device}")

    dataset = SSFlowGlobalZDataset(_cfg_dict(cfg.data))
    model = _build_model(_cfg_dict(cfg.model), device=device)

    training_cfg = _cfg_dict(cfg.training)
    cfg_load_dir = training_cfg.pop("load_dir", None)
    cfg_resume_step = training_cfg.pop("resume_step", None)
    load_dir = load_dir if load_dir is not None else cfg_load_dir
    resume_step = resume_step if resume_step is not None else cfg_resume_step
    output_dir = str(training_cfg.pop("output_dir", "output/ss_flow_global_z"))
    pretrained_ckpt = training_cfg.pop("pretrained_ckpt", None)
    is_resuming = load_dir is not None and resume_step is not None
    if not is_resuming:
        _load_pretrained(model, pretrained_ckpt, rank=rank)
    elif rank == 0:
        print("[SSFlowGlobalZ] resume mode: checkpoint load will replace pretrained weights")

    model = _apply_lora_if_configured(model, cfg, rank=rank)
    _attach_anchor_if_configured(model, cfg, pretrained_ckpt, rank=rank)

    wandb_config = _cfg_dict(cfg.wandb) if "wandb" in cfg else None
    snapshot_config = _cfg_dict(cfg.snapshot) if "snapshot" in cfg else {}
    snapshot_source_dataset = None
    if bool(snapshot_config.get("enabled", True)) and "data" in snapshot_config:
        snapshot_data_config = dict(_cfg_dict(cfg.data))
        snapshot_data_override = dict(snapshot_config["data"])
        snapshot_data_config.update(snapshot_data_override)
        if "exclude_obj_ids" not in snapshot_data_override:
            snapshot_data_config.pop("exclude_obj_ids", None)
        if "exclude_obj_ids_file" not in snapshot_data_override:
            snapshot_data_config.pop("exclude_obj_ids_file", None)
        snapshot_source_dataset = SSFlowGlobalZDataset(snapshot_data_config)
    trainer = SSFlowGlobalZTrainer(
        models={"denoiser": model},
        dataset=dataset,
        output_dir=output_dir,
        load_dir=load_dir,
        step=resume_step,
        wandb_config=wandb_config,
        snapshot_config=snapshot_config,
        snapshot_source_dataset=snapshot_source_dataset,
        **training_cfg,
    )

    param_snapshot_before = None
    if dump_param_stats and rank == 0:
        param_snapshot_before = {
            name: hashlib.md5(param.detach().cpu().numpy().tobytes()).hexdigest()
            for name, param in model.named_parameters()
        }
        total = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"[PARAM_STATS_BEFORE] total={total} trainable={trainable} ratio={trainable / total * 100:.4f}%")

    if rank == 0:
        print("\n[SSFlowGlobalZ] starting training...")
    trainer.run()

    if dump_param_stats and rank == 0 and param_snapshot_before is not None:
        changed = []
        for name, param in model.named_parameters():
            digest = hashlib.md5(param.detach().cpu().numpy().tobytes()).hexdigest()
            if digest != param_snapshot_before[name]:
                changed.append(name)
        print(f"[PARAM_STATS_AFTER] changed={len(changed)}")
        lora_changed = [name for name in changed if "lora_" in name]
        non_lora_changed = [name for name in changed if "lora_" not in name]
        print(f"[PARAM_STATS_AFTER] lora_changed={len(lora_changed)} non_lora_changed={len(non_lora_changed)}")

    if dist.is_available() and dist.is_initialized():
        dist.barrier()


def parse_args():
    parser = argparse.ArgumentParser(description="4-view token -> global z SS flow fine-tuning")
    parser.add_argument("--config", required=True)
    parser.add_argument("--load-dir", default=None)
    parser.add_argument("--resume-step", type=int, default=None)
    parser.add_argument("--dump-param-stats", action="store_true", default=False)
    parser.add_argument("overrides", nargs="*", default=[])
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config, overrides=args.overrides if args.overrides else None)
    train(
        cfg,
        load_dir=args.load_dir,
        resume_step=args.resume_step,
        dump_param_stats=args.dump_param_stats,
    )


if __name__ == "__main__":
    main()
