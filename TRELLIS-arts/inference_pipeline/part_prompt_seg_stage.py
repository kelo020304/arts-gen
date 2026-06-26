from __future__ import annotations

import functools
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from inference_pipeline.part_flow_stage import save_part_latents, save_part_voxels
from inference_pipeline.voxel_io import load_voxel


def _clean_state_dict(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    out: dict[str, torch.Tensor] = {}
    for key, value in state.items():
        out[key.removeprefix("module.")] = value
    return out


def _semantic_class_count(ckpt: dict[str, Any]) -> int:
    from trellis.models.part_seg.promptable_latent_seg import semantic_classes_from_ckpt

    return semantic_classes_from_ckpt(ckpt)


def _model_args_from_ckpt(ckpt: dict[str, Any]) -> dict[str, Any]:
    from trellis.models.part_seg.promptable_latent_seg import voxel_embedding_dim_from_ckpt

    args = dict(ckpt.get("args") or {})
    state = ckpt.get("model") if isinstance(ckpt.get("model"), dict) else {}
    stem = state.get("stem.weight")
    return {
        "dim": int(args.get("dim", 256)),
        "depth": int(args.get("depth", 6)),
        "head_depth": int(args.get("head_depth", 2)),
        "heads": int(args.get("heads", 8)),
        "use_xyz": bool(torch.is_tensor(stem) and int(stem.shape[1]) == 11),
        "use_voxel_head": str(args.get("route", "latent")) == "voxel" or "voxel_out.weight" in state,
        "voxel_depth": int(args.get("voxel_depth", 3)),
        "mask_encoder": str(args.get("mask_encoder", "cnn_grid")),
        "point_k_boundary": int(args.get("point_k_boundary", 32)),
        "point_k_interior": int(args.get("point_k_interior", 32)),
        "point_resample_points": bool(args.get("point_resample_points", False)),
        "semantic_classes": _semantic_class_count(ckpt),
        "voxel_embedding_dim": voxel_embedding_dim_from_ckpt(ckpt),
    }


@functools.lru_cache(maxsize=2)
def _load_prompt_seg_model(ckpt_path: str) -> tuple[torch.nn.Module, torch.Tensor, dict[str, Any]]:
    from trellis.models.part_seg.promptable_latent_seg import PromptablePartLatentSegNet

    ckpt_abs = str(Path(ckpt_path).expanduser().resolve())
    if not Path(ckpt_abs).is_file():
        raise FileNotFoundError(f"part promptable seg ckpt not found: {ckpt_abs}")
    ckpt = torch.load(ckpt_abs, map_location="cpu", weights_only=False)
    if not isinstance(ckpt, dict) or "model" not in ckpt:
        raise RuntimeError(f"{ckpt_abs} is not a PromptablePartLatentSegNet checkpoint")

    model = PromptablePartLatentSegNet(**_model_args_from_ckpt(ckpt)).cuda().eval()
    model.load_state_dict(_clean_state_dict(ckpt["model"]), strict=True)
    for param in model.parameters():
        param.requires_grad_(False)

    empty_code = ckpt.get("empty_code")
    if not torch.is_tensor(empty_code):
        raise RuntimeError(f"{ckpt_abs} missing tensor empty_code")
    if tuple(empty_code.shape[-4:]) != (8, 16, 16, 16):
        raise ValueError(f"{ckpt_abs} empty_code shape {tuple(empty_code.shape)} is invalid")

    args = dict(ckpt.get("args") or {})
    print(
        "[part_prompt_seg_stage] loaded "
        f"{ckpt_abs}: step={ckpt.get('step')} route={args.get('route', 'latent')} "
        f"dim={args.get('dim')} depth={args.get('depth')} mask_encoder={args.get('mask_encoder')}"
    )
    return model, empty_code.float().cuda(), args


def _downsample_binary_mask(mask: np.ndarray, target_size: int = 512) -> np.ndarray:
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


def _load_part_masks2d(dataset: Any, sample: dict[str, Any], part: dict[str, Any]) -> torch.Tensor:
    original_label = int(dataset._part_original_label(sample, part))
    views = []
    for mask_path in dataset._iter_mask_paths(sample):
        if not mask_path.is_file():
            raise FileNotFoundError(f"part prompt mask not found: {mask_path}")
        label_map = np.asarray(np.load(mask_path))
        views.append(_downsample_binary_mask(label_map == original_label, 512))
    if not views:
        raise ValueError("part promptable seg requires at least one prompt view")
    return torch.from_numpy(np.stack(views, axis=0)).float()


def _dense_occ_from_voxel_npz(voxel_path: Path, *, device: torch.device) -> torch.Tensor:
    voxel = load_voxel(voxel_path)
    coords = torch.as_tensor(voxel["coords"], dtype=torch.long, device=device)
    resolution = int(voxel["resolution"])
    if resolution != 64:
        raise ValueError(f"promptable seg expects voxel resolution 64, got {resolution}: {voxel_path}")
    occ = torch.zeros((1, 1, 64, 64, 64), dtype=torch.float32, device=device)
    if coords.numel() > 0:
        occ[0, 0, coords[:, 0], coords[:, 1], coords[:, 2]] = 1.0
    return occ


def _mask_morphology(mask: torch.Tensor, mode: str) -> torch.Tensor:
    if mode == "none":
        return mask
    x = mask.float().unsqueeze(1)
    if mode == "dilate":
        return F.max_pool3d(x, kernel_size=3, stride=1, padding=1).squeeze(1)
    if mode == "erode":
        inv = 1.0 - x
        return (1.0 - F.max_pool3d(inv, kernel_size=3, stride=1, padding=1)).squeeze(1)
    raise ValueError(f"unknown morphology mode: {mode}")


def _coords_from_voxel_output(out: dict[str, Any], *, threshold: float) -> np.ndarray:
    logits = out["voxel_logits"][0].float().sigmoid()
    pad_mask = out["voxel_pad_mask"][0].bool()
    coords = out["voxel_coords"][0].long()
    valid_len = min(coords.shape[0], logits.shape[0], pad_mask.shape[0])
    keep = (logits[:valid_len] > float(threshold)) & (~pad_mask[:valid_len])
    return coords[:valid_len][keep].detach().cpu().numpy().astype(np.int32)


def _decode_latent_to_coords(latent: torch.Tensor, ss_decoder_ckpt: str) -> np.ndarray:
    from inference import decode_ss

    coords = decode_ss(latent.detach().cpu(), ss_decoder_ckpt, threshold=0.0)
    return coords.numpy().astype(np.int32)


def _object_sample_and_dataset(data_config: dict, *, object_id: str, angle_idx: int, view_mode: str):
    from inference_pipeline.object_inputs import _dataset_for

    dataset = _dataset_for(view_mode, data_config)
    idx = next(
        (
            i
            for i, sample in enumerate(dataset.samples)
            if str(sample.get("obj_id", sample.get("object_id"))) == str(object_id)
            and int(sample["angle_idx"]) == int(angle_idx)
        ),
        None,
    )
    if idx is None:
        raise KeyError(f"manifest 中无 object_id={object_id} angle_idx={angle_idx}")
    return dataset, dataset.samples[idx]


def _write_prompt_seg_meta(
    parts_dir: Path,
    *,
    part_index: int,
    part_name: str,
    backend: str,
    ckpt_path: str,
    route: str,
    extra: dict[str, Any],
) -> None:
    path = parts_dir / f"part_{part_index:02d}_meta.json"
    payload = {
        "part_index": int(part_index),
        "target_part_name": str(part_name),
        "backend": str(backend),
        "ckpt": str(ckpt_path),
        "route": str(route),
        **extra,
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def run(
    run_dir,
    data_config: dict,
    *,
    object_id: str,
    angle_idx: int,
    view_mode: str,
    part_seg_ckpt: str,
    ss_decoder_ckpt: str,
    decode_backend: str = "trellis",
    voxel_threshold: float = 0.5,
) -> list[str]:
    """Run promptable part segmentation as the platform's part stage.

    The output contract intentionally matches part_flow_stage: parts/part_NN_voxel.npz
    for trellis decoding, or parts/part_NN_latent.npy + meta for sam3d decoding.
    """
    if not part_seg_ckpt:
        raise ValueError("part_prompt_seg_stage requires part_seg_ckpt")
    if decode_backend not in ("trellis", "sam3d"):
        raise ValueError(f"decode_backend must be 'trellis' or 'sam3d', got {decode_backend!r}")

    run_dir = Path(run_dir)
    z_global_path = run_dir / "ss_latent.npy"
    if not z_global_path.is_file():
        raise FileNotFoundError(f"part promptable seg missing ss_latent.npy: {z_global_path}")
    z_global = torch.from_numpy(np.load(z_global_path)).float().unsqueeze(0).cuda()
    if tuple(z_global.shape) != (1, 8, 16, 16, 16):
        raise ValueError(f"ss_latent.npy shape {tuple(z_global.shape)} is invalid")

    model, empty_code, ckpt_args = _load_prompt_seg_model(part_seg_ckpt)
    route = str(ckpt_args.get("route", "latent"))
    dataset, sample = _object_sample_and_dataset(
        data_config,
        object_id=object_id,
        angle_idx=angle_idx,
        view_mode=view_mode,
    )
    target_part_names = [str(part["part_name"]) for part in sample["parts"]]
    parts_dir = run_dir / "parts"
    parts_dir.mkdir(parents=True, exist_ok=True)

    part_coords: dict[str, np.ndarray] = {}
    part_latents: dict[str, np.ndarray] = {}
    meta_extra: dict[str, dict[str, Any]] = {}
    full_occ = None
    if route == "voxel":
        full_occ = _dense_occ_from_voxel_npz(run_dir / "voxel.npz", device=z_global.device)

    with torch.no_grad():
        for part_index, part in enumerate(sample["parts"]):
            part_name = str(part["part_name"])
            masks2d = _load_part_masks2d(dataset, sample, part).unsqueeze(0).cuda()
            visible_views = int((masks2d.flatten(2).sum(dim=2) > 0).sum().item())
            if route == "voxel":
                out_cell = model(z_global, masks2d, candidate_cells=torch.ones(
                    (1, 16, 16, 16),
                    dtype=torch.float32,
                    device=z_global.device,
                ), full_occ=full_occ)
                pred_m = (out_cell["m_logit"].sigmoid() > 0.5).float().view(1, 16, 16, 16)
                out_voxel = model(
                    z_global,
                    masks2d,
                    candidate_cells=_mask_morphology(pred_m, "dilate"),
                    full_occ=full_occ,
                )
                coords = _coords_from_voxel_output(out_voxel, threshold=float(voxel_threshold))
                part_coords[part_name] = coords
                meta_extra[part_name] = {
                    "visible_prompt_views": visible_views,
                    "voxel_threshold": float(voxel_threshold),
                    "pred_count": int(coords.shape[0]),
                }
            else:
                out = model(z_global, masks2d, empty_code)
                pred_m = (out["m_logit"].sigmoid() > 0.5).float().view(1, 16, 16, 16)
                out = model(z_global, masks2d, empty_code, m_override=pred_m)
                latent = out["part_latent"][0].detach().float().cpu()
                part_latents[part_name] = latent.numpy().astype(np.float32)
                if decode_backend == "trellis":
                    if not ss_decoder_ckpt:
                        raise ValueError("ss_decoder_ckpt is required for promptable latent route decode_backend=trellis")
                    part_coords[part_name] = _decode_latent_to_coords(latent, ss_decoder_ckpt)
                meta_extra[part_name] = {"visible_prompt_views": visible_views}

    if route == "latent" and decode_backend == "sam3d":
        written = save_part_latents(run_dir, part_latents, target_part_names=target_part_names)
    else:
        written = save_part_voxels(run_dir, part_coords, target_part_names=target_part_names, resolution=64)
        for part_index, part_name in enumerate(target_part_names):
            extra = dict(meta_extra.get(part_name, {}))
            extra.update({
                "decode_backend": decode_backend,
                "voxel_threshold": float(voxel_threshold),
                "pred_count": int(np.asarray(part_coords[part_name]).shape[0]),
            })
            _write_prompt_seg_meta(
                parts_dir,
                part_index=part_index,
                part_name=part_name,
                backend="promptable_seg",
                ckpt_path=part_seg_ckpt,
                route=route,
                extra=extra,
            )
    if not written:
        raise ValueError("promptable part segmentation wrote no parts")
    print(
        f"[part_prompt_seg_stage] wrote {len(written)} parts route={route} "
        f"decode_backend={decode_backend} -> {parts_dir}"
    )
    return written
