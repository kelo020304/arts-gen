#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import asdict, dataclass, is_dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[2]
TRELLIS_ROOT = REPO_ROOT / "TRELLIS-arts"
for _path in (REPO_ROOT, TRELLIS_ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

os.environ.setdefault("SPCONV_ALGO", "native")
os.environ.setdefault("ATTN_BACKEND", "sdpa")
os.environ.setdefault("SPARSE_ATTN_BACKEND", "sdpa")
os.environ.setdefault("SS_FLOW_FUSION_MODE", "concat")

import inference  # noqa: E402
from inference_pipeline.part_prompt_seg_stage import (  # noqa: E402
    _coords_from_voxel_output,
    _decode_latent_to_coords,
    _downsample_binary_mask,
    _load_prompt_seg_model,
    _mask_morphology,
)
from inference_pipeline.voxel_io import save_voxel  # noqa: E402
from trellis.modules.sparse import SparseTensor  # noqa: E402
from trellis.utils.arts.slat_asset_writer import save_decoded_slat_assets  # noqa: E402


DEFAULT_SS_FLOW_CKPT = Path(
    "/mnt/robot-data-lab/jzh/art-gen/ckpt/tre-ss-flow/tre-ss-concat-0616-1/ckpts/denoiser_ema0.999_step0012500.pt"
)
DEFAULT_PART_SEG_CKPT = Path(
    "/mnt/robot-data-lab/jzh/art-gen/ckpt/part-prompt-seg/part_promptable_seg_full_S_0616-1/ckpts/step_50000.pt"
)
DEFAULT_SS_DECODER_CKPT = Path(
    "pretrained/TRELLIS-image-large/ckpts/ss_dec_conv3d_16l8_fp16.safetensors"
)
DEFAULT_SLAT_FLOW_CKPT = Path(
    "pretrained/TRELLIS-image-large/ckpts/slat_flow_img_dit_L_64l8p2_fp16.safetensors"
)
DEFAULT_SLAT_MESH_DECODER_CKPT = Path(
    "pretrained/TRELLIS-image-large/ckpts/slat_dec_mesh_swin8_B_64l8m256c_fp16.safetensors"
)
DEFAULT_SLAT_GAUSSIAN_DECODER_CKPT = Path(
    "pretrained/TRELLIS-image-large/ckpts/slat_dec_gs_swin8_B_64l8gs32_fp16.safetensors"
)


@dataclass(frozen=True)
class CkptConfig:
    ss_flow_ckpt: str | Path = DEFAULT_SS_FLOW_CKPT
    part_seg_ckpt: str | Path = DEFAULT_PART_SEG_CKPT
    ss_decoder_ckpt: str | Path = DEFAULT_SS_DECODER_CKPT
    slat_flow_ckpt: str | Path = DEFAULT_SLAT_FLOW_CKPT
    slat_mesh_decoder_ckpt: str | Path = DEFAULT_SLAT_MESH_DECODER_CKPT
    slat_gaussian_decoder_ckpt: str | Path = DEFAULT_SLAT_GAUSSIAN_DECODER_CKPT
    ss_steps: int = 20
    ss_cfg_strength: float = 7.5
    ss_fusion_mode: str = "concat"
    slat_steps: int = 25
    slat_seed: int = 42
    part_voxel_threshold: float = 0.5
    output_dir: str | Path | None = None


@dataclass(frozen=True)
class ReconstructInput:
    images: Sequence[str | Path | Image.Image]
    masks: Sequence[str | Path | np.ndarray]
    part_info: Mapping[str, Any] | str | Path | None = None
    ckpt_config: CkptConfig | Mapping[str, Any] | None = None


@dataclass(frozen=True)
class Part:
    part_id: int
    label: str
    voxel_coords: np.ndarray
    mesh: Any | None
    gaussian: Any | None
    mesh_path: Path | None
    gaussian_path: Path | None
    joint: dict[str, Any] | None
    metadata: dict[str, Any]


@dataclass(frozen=True)
class ArtObject:
    labeled_voxel: np.ndarray
    whole_voxel_coords: np.ndarray
    parts: list[Part]
    scale: dict[str, Any]
    metadata: dict[str, Any]


def _dataclass_from_mapping(cls: type[CkptConfig], value: CkptConfig | Mapping[str, Any] | None) -> CkptConfig:
    if value is None:
        return cls()
    if isinstance(value, cls):
        return value
    if not isinstance(value, Mapping):
        raise TypeError(f"ckpt_config must be CkptConfig or mapping, got {type(value).__name__}")
    allowed = set(cls.__dataclass_fields__)
    extra = sorted(set(value) - allowed)
    if extra:
        raise KeyError(f"unknown CkptConfig fields: {extra}")
    return cls(**dict(value))


def _path_candidates(path: str | Path) -> list[Path]:
    raw = Path(path).expanduser()
    candidates = [raw]
    text = str(raw)
    swaps = [
        ("/robot/data-lab/jzh/art-gen-output", "/mnt/robot-data-lab/jzh/art-gen-output"),
        ("/robot/data-lab/jzh/art-gen", "/mnt/robot-data-lab/jzh/art-gen"),
        ("/mnt/robot-data-lab/jzh/art-gen-output", "/robot/data-lab/jzh/art-gen-output"),
        ("/mnt/robot-data-lab/jzh/art-gen", "/robot/data-lab/jzh/art-gen"),
    ]
    for src, dst in swaps:
        if text.startswith(src):
            candidates.append(Path(dst + text[len(src):]))
    if not raw.is_absolute():
        candidates.append(REPO_ROOT / raw)
    out: list[Path] = []
    seen: set[str] = set()
    for candidate in candidates:
        key = str(candidate)
        if key not in seen:
            seen.add(key)
            out.append(candidate)
    return out


def _resolve_existing(path: str | Path, label: str) -> Path:
    candidates = _path_candidates(path)
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(
        f"{label} not found. Checked: " + ", ".join(str(candidate) for candidate in candidates)
    )


def _load_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _load_image(value: str | Path | Image.Image) -> Image.Image:
    if isinstance(value, Image.Image):
        return value.copy()
    path = Path(value).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"image not found: {path}")
    return Image.open(path).convert("RGB")


def _load_mask(value: str | Path | np.ndarray) -> np.ndarray:
    if isinstance(value, (str, Path)):
        path = Path(value).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"mask not found: {path}")
        if path.suffix == ".npy":
            arr = np.load(path)
        elif path.suffix == ".npz":
            with np.load(path, allow_pickle=False) as data:
                key = "mask" if "mask" in data.files else "labels" if "labels" in data.files else data.files[0]
                arr = data[key]
        else:
            arr = np.asarray(Image.open(path))
    else:
        arr = np.asarray(value)
    if arr.ndim == 3:
        if arr.shape[-1] == 1:
            arr = arr[..., 0]
        else:
            raise ValueError(f"mask must be [H,W] int label map, got {arr.shape}")
    if arr.ndim != 2:
        raise ValueError(f"mask must be [H,W], got {arr.shape}")
    if not np.issubdtype(arr.dtype, np.integer):
        if np.all(np.equal(arr, np.round(arr))):
            arr = arr.astype(np.int32)
        else:
            raise TypeError(f"mask dtype must be integer, got {arr.dtype}")
    arr = np.asarray(arr, dtype=np.int32)
    if arr.size and int(arr.min()) < 0:
        raise ValueError(f"mask labels must be >=0, got min={int(arr.min())}")
    return arr


def _load_part_info(value: Mapping[str, Any] | str | Path | None) -> dict[str, Any] | None:
    if value is None:
        return None
    if isinstance(value, Mapping):
        return dict(value)
    path = Path(value).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"part_info not found: {path}")
    payload = _load_json(path)
    if not isinstance(payload, dict):
        raise ValueError(f"part_info must be a JSON object: {path}")
    return payload


def _part_labels_from_info(part_info: dict[str, Any] | None) -> dict[int, str]:
    if not part_info:
        return {}
    parts = part_info.get("parts")
    if not isinstance(parts, Mapping):
        return {}
    out: dict[int, str] = {}
    for key, value in parts.items():
        if not isinstance(value, Mapping) or "label" not in value:
            continue
        label_id = int(value["label"])
        text = str(value.get("type") or key)
        out[label_id] = text
    return out


def _validate_inputs(
    images: Sequence[Image.Image],
    masks: Sequence[np.ndarray],
    part_info: dict[str, Any] | None,
) -> list[int]:
    if len(images) != len(masks):
        raise ValueError(f"images/masks length mismatch: {len(images)} != {len(masks)}")
    if len(images) != 4:
        raise ValueError(f"0617 reconstruct expects exactly 4 views, got {len(images)}")
    for idx, (image, mask) in enumerate(zip(images, masks)):
        if image.size != (int(mask.shape[1]), int(mask.shape[0])):
            raise ValueError(
                f"view {idx} image size {image.size} does not match mask shape {mask.shape}"
            )
    labels = sorted({int(v) for mask in masks for v in np.unique(mask).tolist() if int(v) > 0})
    if not labels:
        raise ValueError("masks contain no positive part labels")
    if part_info:
        known = set(_part_labels_from_info(part_info))
        if known:
            missing = sorted(set(labels) - known)
            if missing:
                raise ValueError(f"mask labels not present in part_info parts[*].label: {missing}")
    for label_id in labels:
        pixels = sum(int((mask == int(label_id)).sum()) for mask in masks)
        if pixels <= 0:
            raise ValueError(f"part label {label_id} has zero pixels across all views")
    return labels


def _rgba_with_mask_alpha(image: Image.Image, mask: np.ndarray) -> Image.Image:
    rgb = image.convert("RGB")
    alpha = Image.fromarray((np.asarray(mask) > 0).astype(np.uint8) * 255, mode="L")
    rgba = rgb.convert("RGBA")
    rgba.putalpha(alpha)
    return rgba


def _dense_occ_from_coords(coords: np.ndarray, *, device: torch.device) -> torch.Tensor:
    coords = np.asarray(coords, dtype=np.int64).reshape(-1, 3)
    occ = torch.zeros((1, 1, 64, 64, 64), dtype=torch.float32, device=device)
    if coords.size:
        if int(coords.min()) < 0 or int(coords.max()) >= 64:
            raise ValueError(f"whole voxel coords out of [0,64): min={int(coords.min())} max={int(coords.max())}")
        ct = torch.as_tensor(coords, dtype=torch.long, device=device)
        occ[0, 0, ct[:, 0], ct[:, 1], ct[:, 2]] = 1.0
    return occ


def _part_masks_tensor(masks: Sequence[np.ndarray], part_id: int) -> torch.Tensor:
    views = [_downsample_binary_mask(mask == int(part_id), 512) for mask in masks]
    return torch.from_numpy(np.stack(views, axis=0)).float()


def _predict_part_voxels(
    *,
    z_global: torch.Tensor,
    whole_coords: np.ndarray,
    masks: Sequence[np.ndarray],
    part_ids: Sequence[int],
    part_names: Mapping[int, str],
    part_seg_ckpt: Path,
    ss_decoder_ckpt: Path,
    voxel_threshold: float,
) -> tuple[dict[int, np.ndarray], dict[int, dict[str, Any]]]:
    model, empty_code, ckpt_args = _load_prompt_seg_model(str(part_seg_ckpt))
    route = str(ckpt_args.get("route", "latent"))
    z_global_b = z_global.unsqueeze(0).float().cuda()
    if tuple(z_global_b.shape) != (1, 8, 16, 16, 16):
        raise ValueError(f"z_global shape invalid: {tuple(z_global_b.shape)}")
    full_occ = _dense_occ_from_coords(whole_coords, device=z_global_b.device) if route == "voxel" else None
    part_coords: dict[int, np.ndarray] = {}
    meta: dict[int, dict[str, Any]] = {}
    with torch.no_grad():
        for part_id in part_ids:
            masks2d = _part_masks_tensor(masks, int(part_id)).unsqueeze(0).cuda()
            visible_views = int((masks2d.flatten(2).sum(dim=2) > 0).sum().item())
            if visible_views <= 0:
                raise ValueError(f"part_id={part_id} has no visible prompt views")
            if route == "voxel":
                out_cell = model(
                    z_global_b,
                    masks2d,
                    candidate_cells=torch.ones((1, 16, 16, 16), dtype=torch.float32, device=z_global_b.device),
                    full_occ=full_occ,
                )
                pred_m = (out_cell["m_logit"].sigmoid() > 0.5).float().view(1, 16, 16, 16)
                out_voxel = model(
                    z_global_b,
                    masks2d,
                    candidate_cells=_mask_morphology(pred_m, "dilate"),
                    full_occ=full_occ,
                )
                coords = _coords_from_voxel_output(out_voxel, threshold=float(voxel_threshold))
            else:
                out = model(z_global_b, masks2d, empty_code)
                pred_m = (out["m_logit"].sigmoid() > 0.5).float().view(1, 16, 16, 16)
                out = model(z_global_b, masks2d, empty_code, m_override=pred_m)
                latent = out["part_latent"][0].detach().float().cpu()
                coords = _decode_latent_to_coords(latent, str(ss_decoder_ckpt))
            coords = np.asarray(coords, dtype=np.int32).reshape(-1, 3)
            if coords.size and (int(coords.min()) < 0 or int(coords.max()) >= 64):
                raise ValueError(
                    f"part_id={part_id} predicted coords out of [0,64): "
                    f"min={int(coords.min())} max={int(coords.max())}"
                )
            part_coords[int(part_id)] = coords
            meta[int(part_id)] = {
                "label": str(part_names.get(int(part_id), f"part_{int(part_id):02d}")),
                "route": route,
                "visible_prompt_views": visible_views,
                "pred_voxel_count": int(coords.shape[0]),
            }
    return part_coords, meta


def _sparse_subset_from_coords(slat: SparseTensor, coords: np.ndarray, label: str) -> tuple[SparseTensor, int]:
    coords_np = np.asarray(coords, dtype=np.int64).reshape(-1, 3)
    if coords_np.size == 0:
        raise ValueError(f"{label}: empty part coords")
    spatial = slat.coords[:, 1:].detach().long().cpu().numpy()
    resolution = int(max(int(spatial.max(initial=0)) + 1, int(coords_np.max(initial=0)) + 1, 64))
    slat_keys = spatial[:, 0] * resolution * resolution + spatial[:, 1] * resolution + spatial[:, 2]
    part_keys = coords_np[:, 0] * resolution * resolution + coords_np[:, 1] * resolution + coords_np[:, 2]
    part_set = set(int(x) for x in part_keys.tolist())
    keep_np = np.fromiter((int(k) in part_set for k in slat_keys.tolist()), dtype=bool, count=len(slat_keys))
    matched = int(keep_np.sum())
    if matched == 0:
        raise ValueError(f"{label}: no part coords matched whole SLat coords ({len(coords_np)} requested)")
    keep = torch.from_numpy(keep_np).to(device=slat.feats.device)
    return SparseTensor(feats=slat.feats[keep].contiguous(), coords=slat.coords[keep].contiguous()), matched


def _labeled_voxel(part_coords: Mapping[int, np.ndarray]) -> np.ndarray:
    volume = np.zeros((64, 64, 64), dtype=np.int32)
    for part_id, coords in part_coords.items():
        c = np.asarray(coords, dtype=np.int64).reshape(-1, 3)
        if c.size:
            volume[c[:, 0], c[:, 1], c[:, 2]] = int(part_id)
    return volume


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _as_jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return {"shape": list(value.shape), "dtype": str(value.dtype)}
    if is_dataclass(value):
        return _as_jsonable(asdict(value))
    if isinstance(value, Mapping):
        return {str(k): _as_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_as_jsonable(v) for v in value]
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    return value


def _summary_payload(result: ArtObject) -> dict[str, Any]:
    return {
        "metadata": _as_jsonable(result.metadata),
        "scale": _as_jsonable(result.scale),
        "labeled_voxel": {
            "path": "labeled_voxel.npy",
            "shape": list(result.labeled_voxel.shape),
            "dtype": str(result.labeled_voxel.dtype),
        },
        "whole_voxel_coords": {
            "shape": list(result.whole_voxel_coords.shape),
            "dtype": str(result.whole_voxel_coords.dtype),
        },
        "parts": [
            {
                "part_id": int(part.part_id),
                "label": str(part.label),
                "voxel_count": int(part.voxel_coords.shape[0]),
                "mesh_path": None if part.mesh_path is None else str(part.mesh_path),
                "gaussian_path": None if part.gaussian_path is None else str(part.gaussian_path),
                "joint": part.joint,
                "metadata": _as_jsonable(part.metadata),
            }
            for part in result.parts
        ],
    }


def reconstruct(
    images: Sequence[str | Path | Image.Image] | ReconstructInput,
    masks: Sequence[str | Path | np.ndarray] | None = None,
    ckpt_config: CkptConfig | Mapping[str, Any] | None = None,
    *,
    part_info: Mapping[str, Any] | str | Path | None = None,
) -> ArtObject:
    """Run the 0617 e2e reconstruction path with caller-provided part masks.

    The only segmentation source is ``masks``. This function does not run or
    import SAM3D.
    """
    started = time.time()
    if isinstance(images, ReconstructInput):
        req = images
        raw_images = req.images
        raw_masks = req.masks
        raw_part_info = req.part_info
        cfg = _dataclass_from_mapping(CkptConfig, req.ckpt_config)
    else:
        if masks is None:
            raise TypeError("masks is required when images is not ReconstructInput")
        raw_images = images
        raw_masks = masks
        raw_part_info = part_info
        cfg = _dataclass_from_mapping(CkptConfig, ckpt_config)

    pil_images = [_load_image(item) for item in raw_images]
    np_masks = [_load_mask(item) for item in raw_masks]
    info = _load_part_info(raw_part_info)
    part_ids = _validate_inputs(pil_images, np_masks, info)
    part_names = _part_labels_from_info(info)
    rgba_images = [_rgba_with_mask_alpha(image, mask) for image, mask in zip(pil_images, np_masks)]

    ss_flow_ckpt = _resolve_existing(cfg.ss_flow_ckpt, "SS-flow ckpt")
    part_seg_ckpt = _resolve_existing(cfg.part_seg_ckpt, "part promptable seg ckpt")
    ss_decoder_ckpt = _resolve_existing(cfg.ss_decoder_ckpt, "SS decoder ckpt")
    slat_flow_ckpt = _resolve_existing(cfg.slat_flow_ckpt, "SLat flow ckpt")
    slat_mesh_decoder_ckpt = _resolve_existing(cfg.slat_mesh_decoder_ckpt, "SLat mesh decoder ckpt")
    slat_gaussian_decoder_ckpt = _resolve_existing(cfg.slat_gaussian_decoder_ckpt, "SLat gaussian decoder ckpt")

    cond_tokens = inference._images_to_tokens(rgba_images).detach().float().cpu()
    z_global = inference.run_ss_flow_from_tokens(
        cond_tokens,
        str(ss_flow_ckpt),
        num_steps=int(cfg.ss_steps),
        cfg_strength=float(cfg.ss_cfg_strength),
        fusion_mode=str(cfg.ss_fusion_mode),
    )
    whole_coords = inference.decode_ss(z_global, str(ss_decoder_ckpt), threshold=0.0).numpy().astype(np.int32)
    part_coords, part_meta = _predict_part_voxels(
        z_global=z_global,
        whole_coords=whole_coords,
        masks=np_masks,
        part_ids=part_ids,
        part_names=part_names,
        part_seg_ckpt=part_seg_ckpt,
        ss_decoder_ckpt=ss_decoder_ckpt,
        voxel_threshold=float(cfg.part_voxel_threshold),
    )

    whole_coords_t = torch.from_numpy(np.ascontiguousarray(whole_coords.astype(np.int64, copy=False))).long()
    overall_slat = inference.run_slat_flow_from_tokens(
        cond_tokens,
        whole_coords_t,
        str(slat_flow_ckpt),
        num_steps=int(cfg.slat_steps),
        seed=int(cfg.slat_seed),
    )

    out_dir = None if cfg.output_dir is None else Path(cfg.output_dir).expanduser().resolve()
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        save_voxel(out_dir, whole_coords, resolution=64, source="reconstruct_ss_flow", basename="whole_voxel")

    decoded_overall = inference.decode_slat_assets(
        overall_slat,
        gaussian_decoder_ckpt=str(slat_gaussian_decoder_ckpt),
        mesh_decoder_ckpt=str(slat_mesh_decoder_ckpt),
        slat_is_normalized=True,
    )
    overall_assets: dict[str, str] = {}
    if out_dir is not None:
        overall_assets = save_decoded_slat_assets(
            decoded_overall,
            out_dir / "overall",
            mesh_name="overall.glb",
            gaussian_name="overall.ply",
        )

    parts: list[Part] = []
    for part_id in part_ids:
        label = str(part_names.get(int(part_id), f"part_{int(part_id):02d}"))
        part_slat, matched = _sparse_subset_from_coords(overall_slat, part_coords[int(part_id)], label)
        decoded = inference.decode_slat_assets(
            part_slat,
            gaussian_decoder_ckpt=str(slat_gaussian_decoder_ckpt),
            mesh_decoder_ckpt=str(slat_mesh_decoder_ckpt),
            slat_is_normalized=True,
        )
        mesh_path = None
        gaussian_path = None
        if out_dir is not None:
            part_dir = out_dir / f"part_{int(part_id):02d}_{label.replace('/', '_')}"
            assets = save_decoded_slat_assets(
                decoded,
                part_dir,
                mesh_name="mesh.glb",
                gaussian_name="gaussian.ply",
            )
            if "mesh" in assets:
                mesh_path = (part_dir / assets["mesh"]).resolve()
            if "gaussian" in assets:
                gaussian_path = (part_dir / assets["gaussian"]).resolve()
            save_voxel(part_dir, part_coords[int(part_id)], resolution=64, source="promptable_seg", basename="voxel")
        part_record = Part(
            part_id=int(part_id),
            label=label,
            voxel_coords=part_coords[int(part_id)],
            mesh=decoded.get("mesh"),
            gaussian=decoded.get("gaussian"),
            mesh_path=mesh_path,
            gaussian_path=gaussian_path,
            joint=None,
            metadata={
                **part_meta[int(part_id)],
                "slat_source": "subset_from_whole_slat_by_coords",
                "matched_slat_coords": int(matched),
                "joint_status": "TODO_no_cotrain_ckpt",
            },
        )
        parts.append(part_record)

    labeled = _labeled_voxel(part_coords)
    if out_dir is not None:
        np.save(out_dir / "labeled_voxel.npy", labeled.astype(np.int32))

    metadata = {
        "interface": "FROZEN v1",
        "view_count": len(pil_images),
        "input_resolution": [int(pil_images[0].height), int(pil_images[0].width)],
        "mask_contract": "[H,W] int32; 0=bg; positive labels are stable cross-view part ids",
        "ckpts": {
            "ss_flow_ckpt": str(ss_flow_ckpt),
            "part_seg_ckpt": str(part_seg_ckpt),
            "ss_decoder_ckpt": str(ss_decoder_ckpt),
            "slat_flow_ckpt": str(slat_flow_ckpt),
            "slat_mesh_decoder_ckpt": str(slat_mesh_decoder_ckpt),
            "slat_gaussian_decoder_ckpt": str(slat_gaussian_decoder_ckpt),
        },
        "ss_stage": {
            "fusion_mode": str(cfg.ss_fusion_mode),
            "steps": int(cfg.ss_steps),
            "cfg_strength": float(cfg.ss_cfg_strength),
            "whole_voxel_count": int(whole_coords.shape[0]),
        },
        "slat_stage": {
            "flow_calls": 1,
            "part_rule": "whole SLat flow once, then exact sparse-coordinate subset per part",
            "steps": int(cfg.slat_steps),
            "seed": int(cfg.slat_seed),
            "overall_assets": overall_assets,
        },
        "joint_status": "TODO_no_cotrain_ckpt",
        "seconds": round(time.time() - started, 3),
    }
    result = ArtObject(
        labeled_voxel=labeled,
        whole_voxel_coords=whole_coords,
        parts=parts,
        scale={"resolution": 64, "coord_frame": "canonical_grid"},
        metadata=metadata,
    )
    if out_dir is not None:
        _write_json(out_dir / "summary.json", _summary_payload(result))
    return result


def _default_smoke_sample() -> tuple[list[Path], list[Path], Path]:
    root = Path("/mnt/robot-data-lab/jzh/art-gen/data/phyx-verse")
    object_id = "0023687e90394c3e97ab19b0160cafb3"
    angle = 0
    views = [0, 1, 2, 3]
    base = root / "renders" / object_id / f"angle_{angle}"
    images = [base / "rgb" / f"view_{view}.png" for view in views]
    masks = [base / "mask" / f"mask_{view}.npy" for view in views]
    part_info = root / "reconstruction" / "part_info" / object_id / "part_info.json"
    return images, masks, part_info


def _load_ckpt_config_json(path: str | Path | None) -> dict[str, Any]:
    if not path:
        return {}
    payload = _load_json(path)
    if not isinstance(payload, dict):
        raise ValueError(f"ckpt config must be a JSON object: {path}")
    return payload


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="VLM-facing reconstruct API smoke/CLI.")
    parser.add_argument("--images", nargs="+", default=None)
    parser.add_argument("--masks", nargs="+", default=None)
    parser.add_argument("--part-info", default=None)
    parser.add_argument("--ckpt-config-json", default=None)
    parser.add_argument("--out-dir", default="/mnt/robot-data-lab/jzh/art-gen/vlm-reconstruct-smoke/part0")
    parser.add_argument("--smoke-from-dataset", action="store_true")
    parser.add_argument("--part-seg-ckpt", default=None, help="Explicit override, useful if runbook ckpt path is not mounted.")
    parser.add_argument("--ss-flow-ckpt", default=None)
    parser.add_argument("--quick-steps", action="store_true", help="Use 2 SS/SLat steps for interface smoke only.")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if args.smoke_from_dataset:
        images, masks, part_info = _default_smoke_sample()
    else:
        if not args.images or not args.masks:
            raise SystemExit("--images and --masks are required unless --smoke-from-dataset is set")
        images = [Path(item) for item in args.images]
        masks = [Path(item) for item in args.masks]
        part_info = Path(args.part_info) if args.part_info else None

    cfg_payload = _load_ckpt_config_json(args.ckpt_config_json)
    cfg_payload["output_dir"] = args.out_dir
    if args.part_seg_ckpt:
        cfg_payload["part_seg_ckpt"] = args.part_seg_ckpt
    if args.ss_flow_ckpt:
        cfg_payload["ss_flow_ckpt"] = args.ss_flow_ckpt
    if args.quick_steps:
        cfg_payload.setdefault("ss_steps", 2)
        cfg_payload.setdefault("slat_steps", 2)

    result = reconstruct(
        ReconstructInput(
            images=images,
            masks=masks,
            part_info=part_info,
            ckpt_config=CkptConfig(**cfg_payload),
        )
    )
    print(
        json.dumps(
            {
                "status": "done",
                "out_dir": str(Path(args.out_dir).expanduser().resolve()),
                "whole_voxels": int(result.whole_voxel_coords.shape[0]),
                "parts": [
                    {
                        "part_id": part.part_id,
                        "label": part.label,
                        "voxels": int(part.voxel_coords.shape[0]),
                        "mesh_path": None if part.mesh_path is None else str(part.mesh_path),
                        "gaussian_path": None if part.gaussian_path is None else str(part.gaussian_path),
                    }
                    for part in result.parts
                ],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
