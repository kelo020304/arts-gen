#!/usr/bin/env python3
"""Track2 4-view SLat flow overfit entry.

Uses Phase 2 shared cache for overall SLat coords/feats, and by default rebuilds
4-view DINOv2 tokens through the same live official TRELLIS RGBA path as
ee-eval.  View dropout preserves the original 4 view slots, and a zero-
initialized per-view identity embedding is added before flattening.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_file
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, DistributedSampler

REPO_ROOT = Path(__file__).resolve().parents[3]
TRELLIS_ROOT = REPO_ROOT / "TRELLIS-arts"
if str(TRELLIS_ROOT) not in sys.path:
    sys.path.insert(0, str(TRELLIS_ROOT))

from trellis.models.structured_latent_flow import SLatFlowModel  # noqa: E402
from trellis.modules.sparse import SparseTensor  # noqa: E402
from official_trellis_cond import encode_official_rgba_tokens  # noqa: E402


DEFAULT_CACHE_MANIFEST = Path("/robot/data-lab/jzh/art-gen/data/slat_dec_part_cache/phase2_shared/phase2_shared_cache_manifest.json")
DEFAULT_FLOW_CKPT = REPO_ROOT / "pretrained/TRELLIS-image-large/ckpts/slat_flow_img_dit_L_64l8p2_fp16.safetensors"
DEFAULT_CKPT_DIR = Path("/robot/data-lab/jzh/art-gen/ckpts/slat-flow-mv/overfit-0703")
DEFAULT_MEAN = torch.tensor(
    [-2.1687545776367188, -0.004347046371549368, -0.13352349400520325, -0.08418072760105133,
     -0.5271206498146057, 0.7238689064979553, -1.1414450407028198, 1.2039363384246826],
    dtype=torch.float32,
).view(1, 8)
DEFAULT_STD = torch.tensor(
    [2.377650737762451, 2.386378288269043, 2.124418020248413, 2.1748552322387695,
     2.663944721221924, 2.371192216873169, 2.6217446327209473, 2.684523105621338],
    dtype=torch.float32,
).view(1, 8)


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


class Phase2SLatFlowDataset(Dataset):
    def __init__(
        self,
        manifest: Path = DEFAULT_CACHE_MANIFEST,
        *,
        max_num_voxels: int = 0,
        view_dropout: bool = True,
        min_views: int = 1,
        cond_layer_norm: bool = False,
        cond_source: str = "live_official_trellis_rgba",
    ) -> None:
        self.manifest = manifest.resolve()
        self.cache_root = self.manifest.parent
        payload = load_json(self.manifest)
        self.max_num_voxels = int(max_num_voxels)
        self.view_dropout = bool(view_dropout)
        self.min_views = int(min_views)
        self.cond_layer_norm = bool(cond_layer_norm)
        self.cond_source = _normalize_cond_source(cond_source)
        self._live_token_cache: dict[int, tuple[torch.Tensor, dict[str, Any]]] = {}
        self.samples = []
        for sample_idx, obj in enumerate(payload.get("objects", [])):
            slat = self.cache_root / str(obj["overall_slat_rel"])
            cond = self.cache_root / str(obj["cond_4view_tokens_rel"])
            if not slat.is_file():
                raise FileNotFoundError(f"SLat cache missing: {slat}")
            if self.cond_source == "cache" and not cond.is_file():
                raise FileNotFoundError(f"cond cache missing: {cond}")
            reference_rgb = [str(path) for path in obj.get("reference_rgb", [])]
            reference_masks = [str(path) for path in obj.get("reference_masks", [])]
            view_indices = [int(v) for v in obj.get("view_indices", [])]
            if self.cond_source == "live_official_trellis_rgba":
                if len(reference_rgb) != 4 or len(reference_masks) != 4 or len(view_indices) != 4:
                    raise ValueError(
                        f"{self.manifest}: live official cond requires 4 reference_rgb/masks/view_indices "
                        f"for {obj.get('dataset_id')}::{obj.get('obj_id')}, got "
                        f"{len(reference_rgb)}/{len(reference_masks)}/{len(view_indices)}"
                    )
            self.samples.append(
                {
                    "sample_idx": int(sample_idx),
                    "dataset_id": obj["dataset_id"],
                    "obj_id": obj["obj_id"],
                    "tag": obj["tag"],
                    "angle_idx": int(obj["angle_idx"]),
                    "slat": slat,
                    "cond": cond,
                    "reference_rgb": reference_rgb,
                    "reference_masks": reference_masks,
                    "view_indices": view_indices,
                }
            )
        if len(self.samples) < 8:
            raise RuntimeError(f"{self.manifest}: expected at least 8 samples, got {len(self.samples)}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        sample = self.samples[idx]
        with np.load(sample["slat"], allow_pickle=False) as data:
            coords = np.asarray(data["coords"], dtype=np.int32)
            feats = np.asarray(data["feats"], dtype=np.float32)
        if coords.ndim != 2 or coords.shape[1] != 3:
            raise ValueError(f"{sample['slat']}: coords expected [N,3], got {coords.shape}")
        if feats.shape != (coords.shape[0], 8):
            raise ValueError(f"{sample['slat']}: feats expected [{coords.shape[0]},8], got {feats.shape}")
        if self.max_num_voxels and coords.shape[0] > self.max_num_voxels:
            keep = np.linspace(0, coords.shape[0] - 1, self.max_num_voxels).round().astype(np.int64)
            coords = coords[keep]
            feats = feats[keep]
        feats = (torch.from_numpy(feats).float() - DEFAULT_MEAN) / DEFAULT_STD
        if self.cond_source == "live_official_trellis_rgba":
            cache_key = int(sample["sample_idx"])
            cached = self._live_token_cache.get(cache_key)
            if cached is None:
                tokens, token_meta = encode_official_rgba_tokens(
                    sample["reference_rgb"],
                    sample["reference_masks"],
                    view_indices=sample["view_indices"],
                )
                cached = (tokens, token_meta)
                self._live_token_cache[cache_key] = cached
            tokens, token_meta = cached
            tokens = tokens.clone()
            view_indices = [int(v) for v in token_meta["view_indices"]]
        else:
            with np.load(sample["cond"], allow_pickle=False) as data:
                tokens = torch.from_numpy(np.asarray(data["tokens"], dtype=np.float32)).float()
                view_indices = np.asarray(data["view_indices"], dtype=np.int16).astype(int).tolist()
            token_meta = {
                "token_source": "cache",
                "token_path": str(sample["cond"].resolve()),
                "token_shape": list(tokens.shape),
                "view_indices": view_indices,
            }
        if tokens.ndim != 3 or tokens.shape[0] != 4 or tokens.shape[-1] != 1024:
            raise ValueError(f"{sample['cond']}: tokens expected [4,T,1024], got {tuple(tokens.shape)}")
        keep_mask = torch.ones(4, dtype=torch.bool)
        if self.view_dropout and self.min_views < 4:
            n_keep = random.randint(self.min_views, 4)
            keep_positions = sorted(random.sample(range(4), n_keep))
            keep_mask[:] = False
            keep_mask[keep_positions] = True
            tokens = tokens.clone()
            tokens[~keep_mask] = 0
        if self.cond_layer_norm and self.cond_source == "cache":
            tokens = F.layer_norm(tokens, tokens.shape[-1:])
        return {
            "coords": torch.from_numpy(coords.astype(np.int32, copy=False)),
            "feats": feats,
            "cond_tokens": tokens,
            "keep_mask": keep_mask,
            "meta": {k: sample[k] for k in ("dataset_id", "obj_id", "tag", "angle_idx", "reference_rgb", "reference_masks")} | {
                "view_indices": view_indices,
                "token_source": token_meta["token_source"],
                "token_meta": token_meta,
            },
        }

    @staticmethod
    def collate_fn(batch: list[dict[str, Any]]) -> dict[str, Any]:
        coords_parts = []
        feats_parts = []
        for i, sample in enumerate(batch):
            coords = sample["coords"].to(torch.int32)
            batch_col = torch.full((coords.shape[0], 1), i, dtype=torch.int32)
            coords_parts.append(torch.cat([batch_col, coords], dim=1))
            feats_parts.append(sample["feats"].float())
        return {
            "coords": torch.cat(coords_parts, dim=0),
            "feats": torch.cat(feats_parts, dim=0),
            "cond_tokens": torch.stack([sample["cond_tokens"] for sample in batch], dim=0),
            "keep_mask": torch.stack([sample["keep_mask"] for sample in batch], dim=0),
            "meta": [sample["meta"] for sample in batch],
        }


class SLatFlowWithViewEmbedding(nn.Module):
    def __init__(self, base: SLatFlowModel, *, num_views: int = 4, cond_dim: int = 1024) -> None:
        super().__init__()
        self.base = base
        self.view_embed = nn.Embedding(int(num_views), int(cond_dim))
        nn.init.zeros_(self.view_embed.weight)

    def forward(self, x: SparseTensor, t: torch.Tensor, cond_tokens: torch.Tensor) -> SparseTensor:
        if cond_tokens.ndim != 4:
            raise ValueError(f"cond_tokens expected [B,V,T,D], got {tuple(cond_tokens.shape)}")
        b, v, token_count, dim = cond_tokens.shape
        view_ids = torch.arange(v, device=cond_tokens.device)
        cond = cond_tokens + self.view_embed(view_ids).view(1, v, 1, dim)
        cond = cond.reshape(b, v * token_count, dim)
        return self.base(x, t, cond)


def _normalize_cond_source(cond_source: str) -> str:
    value = str(cond_source).strip().lower()
    aliases = {
        "live": "live_official_trellis_rgba",
        "official": "live_official_trellis_rgba",
        "live_official": "live_official_trellis_rgba",
        "live_official_trellis_rgba": "live_official_trellis_rgba",
        "cache": "cache",
        "cached": "cache",
        "phase2_cache": "cache",
    }
    if value not in aliases:
        raise ValueError(f"unsupported cond_source={cond_source!r}; expected live_official_trellis_rgba or cache")
    return aliases[value]


def setup_dist() -> tuple[int, int, int, torch.device]:
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        dist.init_process_group(backend="nccl")
        rank = dist.get_rank()
        world = dist.get_world_size()
        local = int(os.environ.get("LOCAL_RANK", rank % torch.cuda.device_count()))
    else:
        rank = 0
        world = 1
        local = 0
    torch.cuda.set_device(local)
    return rank, world, local, torch.device("cuda", local)


def cleanup_dist() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


def build_model(ckpt: Path, device: torch.device) -> SLatFlowWithViewEmbedding:
    cfg = load_json(ckpt.with_suffix(".json"))
    if cfg.get("name") not in {"SLatFlowModel", "ElasticSLatFlowModel"}:
        raise ValueError(f"{ckpt.with_suffix('.json')}: expected SLatFlowModel, got {cfg.get('name')}")
    base = SLatFlowModel(**cfg["args"]).to(device)
    state = load_file(str(ckpt), device=str(device))
    missing, unexpected = base.load_state_dict(state, strict=True)
    if missing or unexpected:
        raise RuntimeError(f"base flow load mismatch missing={missing} unexpected={unexpected}")
    return SLatFlowWithViewEmbedding(base).to(device)


def apply_trainable_mode(model: SLatFlowWithViewEmbedding, mode: str) -> None:
    mode = str(mode)
    if mode == "all":
        for param in model.parameters():
            param.requires_grad_(True)
        return
    if mode == "view_embed_only":
        for param in model.parameters():
            param.requires_grad_(False)
        model.view_embed.weight.requires_grad_(True)
        return
    raise ValueError(f"unsupported trainable_mode={mode!r}")


def sparse_from_batch(batch: dict[str, Any], device: torch.device) -> SparseTensor:
    return SparseTensor(
        coords=batch["coords"].to(device=device, dtype=torch.int32),
        feats=batch["feats"].to(device=device, dtype=torch.float32),
    )


def diffuse(x0: SparseTensor, t: torch.Tensor, noise: SparseTensor) -> SparseTensor:
    t_b = torch.cat([torch.full((x0.layout[i].stop - x0.layout[i].start, 1), t[i], device=x0.device) for i in range(x0.shape[0])], dim=0)
    return x0.replace((1.0 - t_b) * x0.feats + t_b * noise.feats)


def velocity_target(x0: SparseTensor, noise: SparseTensor) -> SparseTensor:
    return x0.replace(noise.feats - x0.feats)


def reduce_float(value: torch.Tensor, world: int) -> float:
    v = value.detach().float()
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(v, op=dist.ReduceOp.SUM)
        v = v / float(world)
    return float(v.cpu().item())


def train(args: argparse.Namespace) -> None:
    rank, world, local_rank, device = setup_dist()
    random.seed(int(args.seed) + rank)
    np.random.seed(int(args.seed) + rank)
    torch.manual_seed(int(args.seed) + rank)
    is_master = rank == 0
    if is_master:
        args.ckpt_dir.mkdir(parents=True, exist_ok=True)
        print(f"[track2] start world={world} max_steps={args.max_steps}", flush=True)
    dataset = Phase2SLatFlowDataset(
        args.manifest,
        max_num_voxels=int(args.max_num_voxels),
        view_dropout=bool(args.view_dropout),
        min_views=int(args.min_views),
        cond_layer_norm=bool(args.cond_layer_norm),
        cond_source=str(args.cond_source),
    )
    if dataset.cond_source == "live_official_trellis_rgba" and int(args.num_workers) != 0:
        raise ValueError("live_official_trellis_rgba runs CUDA DINO in __getitem__; set --num-workers 0")
    if is_master:
        print(
            f"[track2] cond_source={dataset.cond_source} "
            f"slat_norm_mean={DEFAULT_MEAN.flatten().tolist()} slat_norm_std={DEFAULT_STD.flatten().tolist()}",
            flush=True,
        )
    sampler = DistributedSampler(dataset, num_replicas=world, rank=rank, shuffle=True, drop_last=True) if world > 1 else None
    loader = DataLoader(dataset, batch_size=int(args.batch_size_per_gpu), sampler=sampler, shuffle=sampler is None, num_workers=int(args.num_workers), collate_fn=dataset.collate_fn, drop_last=True)
    iterator = iter(loader)
    model = build_model(args.flow_ckpt, device)
    if args.init_ckpt is not None:
        init_ckpt = Path(args.init_ckpt)
        if not init_ckpt.is_file():
            raise FileNotFoundError(f"Track2 init checkpoint missing: {init_ckpt}")
        payload = torch.load(init_ckpt, map_location=device, weights_only=False)
        missing, unexpected = model.load_state_dict(payload["model"], strict=True)
        if missing or unexpected:
            raise RuntimeError(f"Track2 init load mismatch missing={missing} unexpected={unexpected}")
        if is_master:
            print(f"[track2] warm-started from {init_ckpt} step={payload.get('step')}", flush=True)
    apply_trainable_mode(model, str(args.trainable_mode))
    wrapped: nn.Module = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False) if world > 1 else model
    optimizer = torch.optim.AdamW(wrapped.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))
    neg_dropout = float(args.cond_dropout)

    for step in range(1, int(args.max_steps) + 1):
        if sampler is not None:
            sampler.set_epoch(step)
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            batch = next(iterator)
        x0 = sparse_from_batch(batch, device)
        noise = x0.replace(torch.randn_like(x0.feats))
        t = torch.rand(x0.shape[0], device=device)
        xt = diffuse(x0, t, noise)
        target = velocity_target(x0, noise)
        cond = batch["cond_tokens"].to(device=device, dtype=torch.float32)
        if neg_dropout > 0:
            drop = torch.rand(cond.shape[0], device=device) < neg_dropout
            cond = cond.clone()
            cond[drop] = 0
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=bool(args.bf16)):
            pred = wrapped(xt, t * 1000.0, cond)
            loss = F.mse_loss(pred.feats.float(), target.feats.float())
        if not torch.isfinite(loss):
            raise RuntimeError(f"non-finite Track2 loss step={step}: {loss}")
        loss.backward()
        if float(args.grad_clip) > 0:
            torch.nn.utils.clip_grad_norm_(wrapped.parameters(), float(args.grad_clip))
        optimizer.step()
        loss_value = reduce_float(loss, world)
        if is_master and (step == 1 or step % int(args.log_every) == 0 or step == int(args.max_steps)):
            row = {
                "step": step,
                "loss": loss_value,
                "time": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
                "peak_mem_mb": float(torch.cuda.max_memory_allocated(device) / (1024.0 * 1024.0)),
                "views_kept": [int(mask.sum().item()) for mask in batch["keep_mask"]],
            }
            with (args.ckpt_dir / "losses.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row, sort_keys=True) + "\n")
            print(f"[track2] step={step} loss={loss_value:.6f}", flush=True)
        if is_master and (step % int(args.save_every) == 0 or step == int(args.max_steps)):
            payload = {
                "step": step,
                "model": (wrapped.module if isinstance(wrapped, DDP) else wrapped).state_dict(),
                "optimizer": optimizer.state_dict(),
                "args": vars(args),
            }
            torch.save(payload, args.ckpt_dir / f"step_{step:07d}.pt")
            torch.save(payload, args.ckpt_dir / "latest.pt")
    if is_master:
        print(f"[track2] complete max_steps={args.max_steps}", flush=True)
    cleanup_dist()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=DEFAULT_CACHE_MANIFEST)
    parser.add_argument("--flow-ckpt", type=Path, default=DEFAULT_FLOW_CKPT)
    parser.add_argument("--ckpt-dir", type=Path, default=DEFAULT_CKPT_DIR)
    parser.add_argument("--init-ckpt", type=Path, default=None)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--batch-size-per-gpu", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-num-voxels", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1.0e-5)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--view-dropout", action="store_true", default=True)
    parser.add_argument("--no-view-dropout", dest="view_dropout", action="store_false")
    parser.add_argument("--min-views", type=int, default=1)
    parser.add_argument("--cond-dropout", type=float, default=0.1)
    parser.add_argument("--cond-layer-norm", action="store_true")
    parser.add_argument("--cond-source", choices=["live_official_trellis_rgba", "cache"], default="live_official_trellis_rgba")
    parser.add_argument("--trainable-mode", choices=["view_embed_only", "all"], default="view_embed_only")
    parser.add_argument("--bf16", action="store_true", default=True)
    parser.add_argument("--no-bf16", dest="bf16", action="store_false")
    parser.add_argument("--seed", type=int, default=20260703)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--save-every", type=int, default=100)
    train(parser.parse_args())


if __name__ == "__main__":
    os.environ.setdefault("SPCONV_ALGO", "native")
    main()
