#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.inference import reconstruct as core  # noqa: E402


STAGES = ("dino_ss_flow", "ss_decode", "part_prompt_seg", "slat_decode")
STAGE_DEPENDENCY = {
    "dino_ss_flow": None,
    "ss_decode": "dino_ss_flow",
    "part_prompt_seg": "ss_decode",
    "slat_decode": "part_prompt_seg",
}
STAGE_ARTIFACTS = {
    "dino_ss_flow": ("cond_tokens.npz", "z_global.pt", "token_pca.png", "token_norm.png", "metadata.json"),
    "ss_decode": ("whole_coords.npy", "metadata.json"),
    "part_prompt_seg": ("part_coords.npz", "part_metadata.json", "labeled_voxel.npy", "metadata.json"),
    "slat_decode": ("summary.json", "overall/overall.glb", "overall/overall.ply"),
}


def _json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    staged = path.with_name(f".{path.name}.{time.time_ns()}.tmp")
    staged.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    staged.replace(path)


def _read_json(path: Path, default: Any = None) -> Any:
    if not path.is_file():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def _fingerprint_file(path: Path) -> dict[str, Any]:
    resolved = path.expanduser().resolve()
    stat = resolved.stat()
    digest = hashlib.sha256()
    with resolved.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return {"path": str(resolved), "size": stat.st_size, "sha256": digest.hexdigest()}


def _signature(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def stage_dir(pipeline_root: Path, stage: str) -> Path:
    if stage not in STAGES:
        raise ValueError(f"unknown stage {stage!r}; expected one of {STAGES}")
    return pipeline_root / stage


def stage_status(pipeline_root: Path, stage: str) -> dict[str, Any]:
    root = stage_dir(pipeline_root, stage)
    payload = _read_json(root / "status.json", {})
    if not isinstance(payload, dict):
        payload = {}
    return {"stage": stage, "state": "not_started", "progress": 0, **payload}


def pipeline_status(pipeline_root: Path) -> dict[str, Any]:
    return {stage: stage_status(pipeline_root, stage) for stage in STAGES}


def invalidate_downstream(pipeline_root: Path, stage: str) -> list[str]:
    index = STAGES.index(stage)
    removed: list[str] = []
    for name in STAGES[index + 1 :]:
        target = stage_dir(pipeline_root, name)
        if target.exists():
            shutil.rmtree(target)
            removed.append(name)
    return removed


def _write_status(root: Path, stage: str, state: str, progress: int, **extra: Any) -> None:
    previous = _read_json(root / "status.json", {}) or {}
    payload = {
        **previous,
        "stage": stage,
        "state": state,
        "progress": max(0, min(100, int(progress))),
        "updated_unix": time.time(),
        **extra,
    }
    if state == "running" and "started_unix" not in payload:
        payload["started_unix"] = time.time()
    if state in {"complete", "failed", "cached"}:
        payload["finished_unix"] = time.time()
    _json(root / "status.json", payload)


def _stage_complete(root: Path, signature: str, stage: str) -> bool:
    status = _read_json(root / "status.json", {}) or {}
    return (
        status.get("state") in {"complete", "cached"}
        and status.get("signature") == signature
        and all((root / rel).is_file() for rel in STAGE_ARTIFACTS[stage])
    )


def _require_dependency(pipeline_root: Path, stage: str) -> dict[str, Any] | None:
    dependency = STAGE_DEPENDENCY[stage]
    if dependency is None:
        return None
    status = stage_status(pipeline_root, dependency)
    if status.get("state") not in {"complete", "cached"}:
        raise RuntimeError(f"stage {stage} requires completed {dependency}; current state={status.get('state')}")
    return status


def _load_context(
    images: Sequence[str | Path], masks: Sequence[str | Path], part_info: str | Path | None
) -> tuple[list[Any], list[np.ndarray], dict[str, Any] | None, list[int], dict[int, str], int | None]:
    pil_images = [core._load_image(value) for value in images]
    np_masks = [core._load_mask(value) for value in masks]
    info = core._load_part_info(part_info)
    all_part_ids = core._validate_inputs(pil_images, np_masks, info)
    part_names = core._part_labels_from_info(info)
    body_part_id = core._body_label_from_info(info)
    part_ids = [value for value in all_part_ids if body_part_id is None or value != body_part_id]
    if not part_ids:
        raise ValueError("no non-body part labels remain")
    return pil_images, np_masks, info, part_ids, part_names, body_part_id


def _input_signature(images: Sequence[str | Path], masks: Sequence[str | Path], part_info: str | Path | None) -> dict[str, Any]:
    return {
        "images": [_fingerprint_file(Path(value)) for value in images],
        "masks": [_fingerprint_file(Path(value)) for value in masks],
        "part_info": None if part_info is None else _fingerprint_file(Path(part_info)),
    }


def run_stage(
    stage: str,
    *,
    pipeline_root: str | Path,
    images: Sequence[str | Path],
    masks: Sequence[str | Path],
    object_masks: Sequence[str | Path] | None = None,
    cond_tokens: str | Path | None = None,
    part_info: str | Path | None,
    ckpt_config: core.CkptConfig | Mapping[str, Any],
    force: bool = False,
) -> dict[str, Any]:
    if stage not in STAGES:
        raise ValueError(f"unknown stage {stage!r}; expected one of {STAGES}")
    pipeline_root = Path(pipeline_root).expanduser().resolve()
    root = stage_dir(pipeline_root, stage)
    root.mkdir(parents=True, exist_ok=True)
    cfg = core._dataclass_from_mapping(core.CkptConfig, ckpt_config)
    inputs_payload = _input_signature(images, masks, part_info)
    inputs_payload["object_masks"] = [
        _fingerprint_file(Path(value)) for value in (object_masks or masks)
    ]
    inputs_payload["cond_tokens"] = None if cond_tokens is None else _fingerprint_file(Path(cond_tokens))
    input_signature = _signature(inputs_payload)
    try:
        dependency_status = _require_dependency(pipeline_root, stage)
        if dependency_status is not None and dependency_status.get("input_signature") != input_signature:
            raise RuntimeError(
                f"stage {stage} inputs changed since {STAGE_DEPENDENCY[stage]} completed; rerun upstream stages"
            )
    except Exception as exc:
        _write_status(root, stage, "failed", 0, input_signature=input_signature, error=str(exc), message="Dependency check failed")
        raise
    signature_payload = {
        "interface": "ee_eval_staged_v1",
        "stage": stage,
        "inputs": inputs_payload,
        "dependency_signature": None if dependency_status is None else dependency_status.get("signature"),
        "config": _stage_config(stage, cfg),
    }
    signature = _signature(signature_payload)
    if not force and _stage_complete(root, signature, stage):
        _write_status(root, stage, "cached", 100, signature=signature, input_signature=input_signature, message="Using matching cached artifacts")
        return stage_status(pipeline_root, stage)

    invalidated = invalidate_downstream(pipeline_root, stage)
    for child in root.iterdir():
        if child.name in {"status.json", "run.log", "ckpt_config.json"}:
            continue
        shutil.rmtree(child) if child.is_dir() else child.unlink()
    _write_status(root, stage, "running", 1, signature=signature, input_signature=input_signature, invalidated=invalidated, message="Loading inputs")
    try:
        pil_images, np_masks, _info, part_ids, part_names, body_part_id = _load_context(images, masks, part_info)
        if stage == "dino_ss_flow":
            np_object_masks = [core._load_mask(value) for value in (object_masks or masks)]
            _run_dino_ss_flow(root, pil_images, np_object_masks, cfg, cond_tokens=cond_tokens)
        elif stage == "ss_decode":
            _run_ss_decode(pipeline_root, root, cfg)
        elif stage == "part_prompt_seg":
            _run_part_prompt_seg(pipeline_root, root, np_masks, part_ids, part_names, body_part_id, cfg)
        else:
            _run_slat_decode(pipeline_root, root, part_ids, part_names, body_part_id, cfg)
        _write_status(root, stage, "complete", 100, signature=signature, invalidated=invalidated, message="Complete")
        return stage_status(pipeline_root, stage)
    except Exception as exc:
        _write_status(
            root,
            stage,
            "failed",
            stage_status(pipeline_root, stage).get("progress", 0),
            signature=signature,
            error=str(exc),
            traceback=traceback.format_exc(),
        )
        raise


def _stage_config(stage: str, cfg: core.CkptConfig) -> dict[str, Any]:
    common = {"stage": stage}
    if stage == "dino_ss_flow":
        common.update(
            ss_flow_ckpt=str(cfg.ss_flow_ckpt), ss_steps=cfg.ss_steps,
            cfg_strength=cfg.ss_cfg_strength, fusion_mode=cfg.ss_fusion_mode,
            seed=cfg.ss_seed,
        )
    elif stage == "ss_decode":
        common.update(ss_decoder_ckpt=str(cfg.ss_decoder_ckpt))
    elif stage == "part_prompt_seg":
        for key in (
            "part_seg_ckpt", "ss_decoder_ckpt", "part_voxel_threshold", "part_joint_candidate_mode",
            "part_joint_refine", "part_joint_refine_iters", "part_joint_refine_pairwise",
            "part_joint_refine_margin", "part_joint_refine_margin_quantile", "part_joint_refine_neighborhood",
            "part_joint_refine_min_vote_gain", "part_joint_refine_preserve_small_classes", "part_cc_filter",
            "part_cc_min_component_voxels", "part_cc_min_component_fraction", "part_cc_max_component_distance",
            "part_cc_max_large_component_distance",
        ):
            common[key] = getattr(cfg, key)
        checkpoint = Path(str(cfg.part_seg_ckpt)).expanduser().resolve()
        checkpoint_stat = checkpoint.stat()
        common["part_seg_ckpt"] = {
            "path": str(checkpoint),
            "size": checkpoint_stat.st_size,
            "mtime_ns": checkpoint_stat.st_mtime_ns,
        }
    else:
        common.update(
            slat_flow_ckpt=str(cfg.slat_flow_ckpt), mesh_decoder=str(cfg.slat_mesh_decoder_ckpt),
            gaussian_decoder=str(cfg.slat_gaussian_decoder_ckpt), slat_steps=cfg.slat_steps, slat_seed=cfg.slat_seed,
        )
    return common


def _run_dino_ss_flow(
    root: Path,
    images: Sequence[Any],
    object_masks: Sequence[np.ndarray],
    cfg: core.CkptConfig,
    *,
    cond_tokens: str | Path | None = None,
) -> None:
    _write_status(root, "dino_ss_flow", "running", 10, message="Extracting DINO tokens")
    if cond_tokens is not None:
        with np.load(Path(cond_tokens), allow_pickle=False) as payload:
            token_tensor = torch.from_numpy(np.asarray(payload["tokens"], dtype=np.float32)).contiguous()
        token_source = "canonical_dataset_cache"
    else:
        rgba = [core._rgba_with_mask_alpha(image, mask) for image, mask in zip(images, object_masks)]
        token_tensor = core.inference._images_to_tokens(rgba).detach().float().cpu()
        token_source = "dino_from_whole_object_foreground"
    if tuple(token_tensor.shape) != (4, 1374, 1024):
        raise ValueError(f"DINO token shape must be [4,1374,1024], got {tuple(token_tensor.shape)}")
    np.savez_compressed(root / "cond_tokens.npz", tokens=token_tensor.numpy())
    vis_meta = _save_token_visualizations(token_tensor.numpy(), root)
    _write_status(
        root,
        "dino_ss_flow",
        "running",
        25,
        message="Saved real DINO patch-token visualizations",
        artifacts={
            "tokens": "cond_tokens.npz",
            "pca": "token_pca.png",
            "norm": "token_norm.png",
            "pca_views": [f"token_pca_view_{index}.png" for index in range(token_tensor.shape[0])],
            "norm_views": [f"token_norm_view_{index}.png" for index in range(token_tensor.shape[0])],
        },
    )
    _write_status(root, "dino_ss_flow", "running", 35, message="Running sparse-structure flow")
    ckpt = core._resolve_existing(cfg.ss_flow_ckpt, "SS-flow ckpt")
    z_global = core.inference.run_ss_flow_from_tokens(
        token_tensor, str(ckpt), num_steps=int(cfg.ss_steps), cfg_strength=float(cfg.ss_cfg_strength),
        fusion_mode=str(cfg.ss_fusion_mode), seed=int(cfg.ss_seed),
    )
    torch.save(z_global.detach().float().cpu(), root / "z_global.pt")
    _json(root / "metadata.json", {
        "token_shape": list(token_tensor.shape), "token_source": token_source,
        "latent_shape": list(z_global.shape), "checkpoint": str(ckpt),
        "steps": int(cfg.ss_steps), "cfg_strength": float(cfg.ss_cfg_strength),
        "fusion_mode": str(cfg.ss_fusion_mode), "seed": int(cfg.ss_seed), **vis_meta,
    })
    _write_status(root, "dino_ss_flow", "running", 95, message="Saved SS latent")


def _run_ss_decode(pipeline_root: Path, root: Path, cfg: core.CkptConfig) -> None:
    _write_status(root, "ss_decode", "running", 15, message="Loading SS latent")
    z_global = torch.load(stage_dir(pipeline_root, "dino_ss_flow") / "z_global.pt", map_location="cpu", weights_only=True)
    ckpt = core._resolve_existing(cfg.ss_decoder_ckpt, "SS decoder ckpt")
    _write_status(root, "ss_decode", "running", 35, message="Decoding whole-object voxels")
    coords = core.inference.decode_ss(z_global, str(ckpt), threshold=0.0).numpy().astype(np.int32)
    np.save(root / "whole_coords.npy", coords)
    core.save_voxel(root, coords, resolution=64, source="reconstruct_ss_flow", basename="whole_voxel")
    _json(root / "metadata.json", {"whole_voxel_count": int(coords.shape[0]), "checkpoint": str(ckpt)})
    _write_status(root, "ss_decode", "running", 95, message="Saved whole-object voxel")


def _save_token_visualizations(tokens: np.ndarray, root: Path) -> dict[str, Any]:
    """Render DINO patch features themselves, never the source RGB image."""
    tokens = np.asarray(tokens, dtype=np.float32)
    patch_count = int(tokens.shape[1])
    grid = int(round(np.sqrt(patch_count)))
    special_tokens = 0
    if grid * grid != patch_count:
        # DINOv2 used here emits 5 non-spatial tokens followed by 37x37 patches.
        for candidate in range(1, min(32, patch_count)):
            spatial = patch_count - candidate
            candidate_grid = int(round(np.sqrt(spatial)))
            if candidate_grid * candidate_grid == spatial:
                special_tokens = candidate
                grid = candidate_grid
                break
        else:
            raise ValueError(f"cannot infer spatial DINO token grid from T={patch_count}")
    patches = tokens[:, special_tokens:, :]
    flat = patches.reshape(-1, patches.shape[-1]).astype(np.float64)
    centered = flat - flat.mean(axis=0, keepdims=True)
    _u, _s, vh = np.linalg.svd(centered, full_matrices=False)
    pca = (centered @ vh[:3].T).reshape(tokens.shape[0], grid, grid, 3)
    lo = np.percentile(pca, 2, axis=(0, 1, 2), keepdims=True)
    hi = np.percentile(pca, 98, axis=(0, 1, 2), keepdims=True)
    pca_u8 = (np.clip((pca - lo) / np.maximum(hi - lo, 1e-8), 0, 1) * 255).astype(np.uint8)
    norms = np.linalg.norm(patches, axis=-1).reshape(tokens.shape[0], grid, grid)
    norm_lo, norm_hi = float(np.percentile(norms, 2)), float(np.percentile(norms, 98))
    norm_u8 = (np.clip((norms - norm_lo) / max(norm_hi - norm_lo, 1e-8), 0, 1) * 255).astype(np.uint8)
    pca_views: list[np.ndarray] = []
    norm_views: list[np.ndarray] = []
    for index in range(tokens.shape[0]):
        pca_image = Image.fromarray(pca_u8[index], mode="RGB").resize((296, 296), Image.Resampling.NEAREST)
        norm_rgb = np.stack([norm_u8[index], np.zeros_like(norm_u8[index]), 255 - norm_u8[index]], axis=-1)
        norm_image = Image.fromarray(norm_rgb, mode="RGB").resize((296, 296), Image.Resampling.NEAREST)
        pca_image.save(root / f"token_pca_view_{index}.png")
        norm_image.save(root / f"token_norm_view_{index}.png")
        pca_views.append(np.asarray(pca_image))
        norm_views.append(np.asarray(norm_image))
    Image.fromarray(np.concatenate(pca_views, axis=1), mode="RGB").save(root / "token_pca.png")
    Image.fromarray(np.concatenate(norm_views, axis=1), mode="RGB").save(root / "token_norm.png")
    return {
        "token_visualization": {
            "semantics": "PCA RGB of normalized DINOv2 spatial patch-token features; colors are feature projections, not source RGB",
            "norm_semantics": "blue-to-red visualization of per-patch L2 feature norm",
            "special_tokens_excluded": special_tokens,
            "spatial_grid": [grid, grid],
            "pca_path": "token_pca.png",
            "norm_path": "token_norm.png",
        }
    }


def _run_part_prompt_seg(
    pipeline_root: Path, root: Path, masks: Sequence[np.ndarray], part_ids: Sequence[int], part_names: Mapping[int, str],
    body_part_id: int | None, cfg: core.CkptConfig,
) -> None:
    z_global = torch.load(stage_dir(pipeline_root, "dino_ss_flow") / "z_global.pt", map_location="cpu", weights_only=True)
    whole_coords = np.load(stage_dir(pipeline_root, "ss_decode") / "whole_coords.npy")
    _write_status(root, "part_prompt_seg", "running", 15, message="Loading joint part segmentation checkpoint")
    part_coords, part_meta = core._predict_part_voxels(
        z_global=z_global, whole_coords=whole_coords, masks=masks, part_ids=part_ids, part_names=part_names,
        body_part_id=body_part_id, part_seg_ckpt=core._resolve_existing(cfg.part_seg_ckpt, "part promptable seg ckpt"),
        ss_decoder_ckpt=core._resolve_existing(cfg.ss_decoder_ckpt, "SS decoder ckpt"), voxel_threshold=float(cfg.part_voxel_threshold),
        joint_candidate_mode=str(cfg.part_joint_candidate_mode), joint_refine=bool(cfg.part_joint_refine),
        joint_refine_iters=int(cfg.part_joint_refine_iters), joint_refine_pairwise=float(cfg.part_joint_refine_pairwise),
        joint_refine_margin=float(cfg.part_joint_refine_margin), joint_refine_margin_quantile=float(cfg.part_joint_refine_margin_quantile),
        joint_refine_neighborhood=int(cfg.part_joint_refine_neighborhood), joint_refine_min_vote_gain=float(cfg.part_joint_refine_min_vote_gain),
        joint_refine_preserve_small_classes=int(cfg.part_joint_refine_preserve_small_classes),
        joint_save_logits_path=(root / "joint_partition.npz") if cfg.part_joint_save_logits else None,
    )
    _write_status(root, "part_prompt_seg", "running", 70, message="Applying connected-component cleanup")
    cc_records: list[dict[str, Any]] = []
    if cfg.part_cc_filter:
        cc_records = core._apply_part_cc_filter(
            part_coords, whole_coords=whole_coords, part_ids=part_ids, part_names=part_names,
            min_component_voxels=cfg.part_cc_min_component_voxels, min_component_fraction=cfg.part_cc_min_component_fraction,
            max_component_distance=cfg.part_cc_max_component_distance, max_large_component_distance=cfg.part_cc_max_large_component_distance,
        )
    np.savez_compressed(root / "part_coords.npz", **{str(key): value for key, value in part_coords.items()})
    _json(root / "part_metadata.json", {str(key): core._as_jsonable(value) for key, value in part_meta.items()})
    labeled = core._joint_labeled_voxel(part_coords, part_ids) if -1 in part_coords else core._labeled_voxel(part_coords)
    np.save(root / "labeled_voxel.npy", labeled.astype(np.int32))
    for part_id in part_ids:
        label = str(part_names.get(part_id, f"part_{part_id:02d}"))
        core.save_voxel(root / f"part_{part_id:02d}_{label.replace('/', '_')}", part_coords[part_id], resolution=64, source="promptable_seg", basename="voxel")
    _json(root / "metadata.json", {"part_ids": list(part_ids), "part_names": {str(k): v for k, v in part_names.items()}, "cc_filter_records": cc_records})
    _write_status(root, "part_prompt_seg", "running", 95, message="Saved refined part voxels")


def _load_part_coords(path: Path) -> dict[int, np.ndarray]:
    with np.load(path, allow_pickle=False) as payload:
        return {int(key): np.asarray(payload[key], dtype=np.int32) for key in payload.files}


def _run_slat_decode(
    pipeline_root: Path,
    root: Path,
    part_ids: Sequence[int],
    part_names: Mapping[int, str],
    body_part_id: int | None,
    cfg: core.CkptConfig,
) -> None:
    with np.load(stage_dir(pipeline_root, "dino_ss_flow") / "cond_tokens.npz", allow_pickle=False) as payload:
        cond_tokens = torch.from_numpy(np.asarray(payload["tokens"], dtype=np.float32))
    whole_coords = np.load(stage_dir(pipeline_root, "ss_decode") / "whole_coords.npy")
    part_coords = _load_part_coords(stage_dir(pipeline_root, "part_prompt_seg") / "part_coords.npz")
    if -1 not in part_coords or not part_coords[-1].size:
        part_coords[-1] = core._residual_body_coords(whole_coords, part_coords, part_ids)
    _write_status(root, "slat_decode", "running", 10, message="Running whole-object SLat flow")
    slat = core.inference.run_slat_flow_from_tokens(
        cond_tokens, torch.from_numpy(whole_coords.astype(np.int64)), str(core._resolve_existing(cfg.slat_flow_ckpt, "SLat flow ckpt")),
        num_steps=int(cfg.slat_steps), seed=int(cfg.slat_seed),
    )
    torch.save({"coords": slat.coords.detach().cpu(), "feats": slat.feats.detach().cpu()}, root / "overall_slat.pt")
    mesh_ckpt = str(core._resolve_existing(cfg.slat_mesh_decoder_ckpt, "SLat mesh decoder ckpt"))
    gs_ckpt = str(core._resolve_existing(cfg.slat_gaussian_decoder_ckpt, "SLat gaussian decoder ckpt"))
    _write_status(root, "slat_decode", "running", 45, message="Decoding complete mesh and Gaussian")
    overall = core.inference.decode_slat_assets(slat, gaussian_decoder_ckpt=gs_ckpt, mesh_decoder_ckpt=mesh_ckpt, slat_is_normalized=True)
    overall_assets = core.save_decoded_slat_assets(overall, root / "overall", mesh_name="overall.glb", gaussian_name="overall.ply")
    decode_ids = list(part_ids)
    if part_coords[-1].size:
        decode_ids.insert(0, -1)
    parts: list[dict[str, Any]] = []
    count = max(1, len(decode_ids))
    for index, part_id in enumerate(decode_ids):
        is_body = part_id == -1
        output_part_id = body_part_id if is_body and body_part_id is not None else part_id
        label = str(part_names.get(body_part_id, "body")) if is_body else str(part_names.get(part_id, f"part_{part_id:02d}"))
        _write_status(root, "slat_decode", "running", 55 + int(38 * index / count), message=f"Decoding {label}")
        part_slat, matched = core._sparse_subset_from_coords(slat, part_coords[part_id], label)
        decoded = core.inference.decode_slat_assets(part_slat, gaussian_decoder_ckpt=gs_ckpt, mesh_decoder_ckpt=mesh_ckpt, slat_is_normalized=True)
        directory_id = "body" if is_body else f"part_{part_id:02d}"
        part_root = root / f"{directory_id}_{label.replace('/', '_')}"
        assets = core.save_decoded_slat_assets(decoded, part_root, mesh_name="mesh.glb", gaussian_name="gaussian.ply")
        core.save_voxel(part_root, part_coords[part_id], resolution=64, source="promptable_seg", basename="voxel")
        parts.append({
            "part_id": output_part_id, "label": label, "kind": "body" if is_body else "part",
            "voxel_count": int(part_coords[part_id].shape[0]),
            "matched_slat_coords": matched,
            "mesh_path": str((part_root / assets["mesh"]).resolve()) if "mesh" in assets else None,
            "gaussian_path": str((part_root / assets["gaussian"]).resolve()) if "gaussian" in assets else None,
        })
    summary = {
        "interface": "ee_eval_staged_v1", "whole_voxel_count": int(whole_coords.shape[0]), "overall_assets": overall_assets,
        "parts": parts, "pipeline_root": str(pipeline_root),
    }
    _json(root / "summary.json", summary)
    _write_status(root, "slat_decode", "running", 98, message="Saved decoded assets")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one cached EE-eval reconstruction stage")
    parser.add_argument("--stage", required=True, choices=STAGES)
    parser.add_argument("--pipeline-root", required=True)
    parser.add_argument("--images", nargs="+", required=True)
    parser.add_argument("--masks", nargs="+", required=True)
    parser.add_argument("--object-masks", nargs="+")
    parser.add_argument("--cond-tokens")
    parser.add_argument("--part-info")
    parser.add_argument("--ckpt-config-json", required=True)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    config = core._load_ckpt_config_json(args.ckpt_config_json)
    config.pop("output_dir", None)
    result = run_stage(
        args.stage, pipeline_root=args.pipeline_root, images=args.images, masks=args.masks,
        object_masks=args.object_masks, cond_tokens=args.cond_tokens,
        part_info=args.part_info, ckpt_config=core.CkptConfig(**config), force=args.force,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
