#!/usr/bin/env python
"""SAM 3D Objects Sparse-Structure (SS) stage glue script.

Runs the SAM 3D Objects sparse-structure stage for ONE object and captures
BOTH the SS latent (the global z used as the conditioning target by the
Part SS-Latent Flow training data) and the decoded voxel surface.

Why this exists
---------------
The upstream `surface_voxel.SurfaceVoxelPipeline` wrapper only persists the
decoded voxel coords + pose (see ``generate_surface_voxel/surface_voxel``).
For our training data contract we ALSO need the raw SS latent ``z_global`` of
shape ``(8, 16, 16, 16)`` -- this is exactly what
``reconstruction/ss_latents_expanded/<id>/angle_<a>/latent.npz`` stores under
the key ``"mean"``. This script mirrors the wrapper's setup but additionally
captures ``ss_return_dict["shape"]`` and reshapes it to that ``z_global``
layout before saving.

Environment
-----------
This MUST run inside the SAM 3D Objects venv (CUDA 12.1 / cu121). It imports
``sam3d_objects`` (the heavy pipeline) which is NOT installable in the
arts-reconstruction ``arts-gen`` env (cu118). Do not try to run it there.

Output layout (``--out <run_dir>``)
-----------------------------------
    ss_latent.npy   (8, 16, 16, 16) float32  -- matches ss_latents_expanded "mean"
    voxel.npz       {coords:int32[N,3], resolution:64,
                     coord_frame:"canonical_grid", source:"pred"}
    voxel.bin       little-endian uint16 flat x,y,z interleaved (mirrors voxel.npz)
    pose.json       rotation/translation/scale/intrinsics/downsample_factor/seed
                    -- so the downstream SLat stage can reuse the Stage-A pose.

Usage
-----
    python ss_stage.py --image img.png --mask mask.png \
        --config <pipeline.yaml> --out <run_dir> [--seed 42] [--device cuda]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image

# ---------------------------------------------------------------------------
# Reuse the canonical voxel writer from TRELLIS-arts if it is importable; this
# keeps voxel.npz / voxel.bin byte-for-byte identical to the rest of the
# pipeline. We add the TRELLIS-arts repo root to sys.path so that
# `inference_pipeline.voxel_io` resolves. If that import fails (e.g. the repo
# is laid out differently), we fall back to an INLINE writer that produces the
# exact same files -- not a silent error-swallowing fallback, just the same
# bytes written by local code.
# ---------------------------------------------------------------------------
_THIS_DIR = Path(__file__).resolve().parent
# infer_glue/ -> sam3d-stage/ -> submodules/ -> arts-reconstruction/ -> TRELLIS-arts/
_TRELLIS_ARTS_ROOT = (_THIS_DIR / ".." / ".." / ".." / "TRELLIS-arts").resolve()
if _TRELLIS_ARTS_ROOT.is_dir():
    sys.path.insert(0, str(_TRELLIS_ARTS_ROOT))

DEFAULT_OFFLINE_MOGE_MODEL = Path(
    "/robot/data-lab/jzh/art-gen/weights/hub/models--Ruicheng--moge-vitl/"
    "snapshots/979e84da9415762c30e6c0cf8dc0962896c793df/model.pt"
)

try:
    from inference_pipeline.voxel_io import save_voxel as _save_voxel  # type: ignore
    _SAVE_VOXEL_SOURCE = f"inference_pipeline.voxel_io ({_TRELLIS_ARTS_ROOT})"
except Exception:  # noqa: BLE001 -- import-availability probe, re-implemented below
    _save_voxel = None
    _SAVE_VOXEL_SOURCE = "inline (TRELLIS-arts inference_pipeline.voxel_io not importable)"


def _save_voxel_inline(run_dir: Path, coords: np.ndarray, *, resolution: int, source: str) -> None:
    """Inline copy of inference_pipeline.voxel_io.save_voxel.

    Writes voxel.npz + voxel.bin with byte-identical content to the canonical
    writer. Kept in sync with TRELLIS-arts/inference_pipeline/voxel_io.py.
    """
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    c = np.asarray(coords).astype(np.int32).reshape(-1, 3)
    if c.size and (int(c.min()) < 0 or int(c.max()) >= int(resolution)):
        raise ValueError(f"voxel coords 越界 [0,{resolution}): min={int(c.min())} max={int(c.max())}")
    np.savez_compressed(
        run_dir / "voxel.npz",
        coords=c,
        resolution=np.int32(resolution),
        coord_frame="canonical_grid",
        source=str(source),
    )
    (run_dir / "voxel.bin").write_bytes(c.astype("<u2").tobytes())


def save_voxel(run_dir: Path, coords: np.ndarray, *, resolution: int, source: str) -> None:
    if _save_voxel is not None:
        _save_voxel(run_dir, coords, resolution=resolution, source=source)
    else:
        _save_voxel_inline(run_dir, coords, resolution=resolution, source=source)


# ---------------------------------------------------------------------------
# Image / mask loaders (mirror surface_voxel.cli._load_rgb / _load_mask).
# ---------------------------------------------------------------------------
def _load_rgb(path: Path) -> np.ndarray:
    img = Image.open(path).convert("RGB")
    arr = np.array(img, dtype=np.uint8)
    if arr.ndim != 3 or arr.shape[-1] != 3:
        raise ValueError(f"expected RGB image at {path}, got shape {arr.shape}")
    return arr


def _load_mask(path: Path) -> np.ndarray:
    img = Image.open(path)
    arr = np.array(img)
    if arr.ndim == 3:
        # take last channel (alpha if RGBA/LA, else collapse via max)
        arr = arr[..., -1] if arr.shape[-1] in (2, 4) else arr.max(axis=-1)
    if arr.dtype == bool:
        return arr
    return arr > 127


def _merge_mask_to_rgba(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Concatenate RGB + (mask*255) alpha channel.

    Identical to surface_voxel.pipeline._merge_mask_to_rgba so the SS
    preprocessor sees exactly the same RGBA input as the upstream wrapper.
    """
    mask_u8 = (mask.astype(bool).astype(np.uint8)) * 255
    return np.concatenate([image[..., :3], mask_u8[..., None]], axis=-1)


class _StageASSOnlyPipeline:
    """Minimal SAM3D StageA pipeline: pointmap, SS conditioner/generator/decoder."""

    def __init__(
        self,
        *,
        models,
        ss_condition_embedder,
        ss_preprocessor,
        pose_decoder,
        depth_model,
        device,
        dtype,
        downsample_ss_dist,
        ss_condition_input_mapping,
        ss_cfg_strength=7,
        ss_cfg_interval=(0, 500),
        ss_cfg_strength_pm=0.0,
        ss_inference_steps=25,
        ss_rescale_t=3,
    ):
        import torch

        self.models = models
        self.condition_embedders = {"ss_condition_embedder": ss_condition_embedder}
        self.ss_preprocessor = ss_preprocessor
        self.pose_decoder = pose_decoder
        self.depth_model = depth_model
        self.device = torch.device(device)
        self.dtype = dtype
        self.shape_model_dtype = dtype
        self.downsample_ss_dist = int(downsample_ss_dist)
        self.ss_condition_input_mapping = list(ss_condition_input_mapping or [])
        self.ss_cfg_strength = ss_cfg_strength
        self.ss_cfg_interval = list(ss_cfg_interval)
        self.ss_cfg_strength_pm = ss_cfg_strength_pm

        ss_generator = self.models["ss_generator"]
        ss_generator.inference_steps = ss_inference_steps
        ss_generator.reverse_fn.strength = ss_cfg_strength
        ss_generator.reverse_fn.interval = list(ss_cfg_interval)
        ss_generator.rescale_t = ss_rescale_t
        ss_generator.reverse_fn.unconditional_handling = "add_flag"
        ss_generator.reverse_fn.strength_pm = ss_cfg_strength_pm

    def image_to_float(self, image):
        image = np.array(image)
        image = image / 255
        return image.astype(np.float32)

    def _synthetic_pointmap(self, loaded_image, loaded_mask):
        import torch

        _, h, w = loaded_image.shape
        yy, xx = torch.meshgrid(
            torch.linspace(-1.0, 1.0, h, device=self.device),
            torch.linspace(-1.0, 1.0, w, device=self.device),
            indexing="ij",
        )
        aspect = float(w) / max(float(h), 1.0)
        pointmap = torch.stack([xx * aspect, -yy, torch.ones_like(xx)], dim=0)
        pointmap = torch.where(
            loaded_mask.to(self.device)[None] > 0,
            pointmap,
            torch.full_like(pointmap, float("nan")),
        )
        intrinsics = torch.tensor(
            [[1.0, 0.0, 0.5], [0.0, 1.0, 0.5], [0.0, 0.0, 1.0]],
            dtype=torch.float32,
            device=self.device,
        )
        return {"pts_color": loaded_image.to(self.device), "intrinsics": intrinsics, "pointmap": pointmap}

    def compute_pointmap(self, image, pointmap=None):
        import torch

        loaded_image = torch.from_numpy(self.image_to_float(image))
        loaded_mask = loaded_image[..., -1]
        loaded_image = loaded_image.permute(2, 0, 1).contiguous()[:3]

        if pointmap is None and self.depth_model is None:
            print("[WARN] SAM3D_MOGE_MODEL_PATH not set; using synthetic plane pointmap fallback")
            return self._synthetic_pointmap(loaded_image, loaded_mask)

        if pointmap is not None:
            points_tensor = pointmap.to(self.device)
            intrinsics = None
        else:
            from sam3d_objects.pipeline.inference_pipeline_pointmap import (
                Transform3d,
                camera_to_pytorch3d_camera,
            )

            with torch.no_grad():
                with torch.autocast(device_type="cuda", dtype=self.dtype):
                    output = self.depth_model(loaded_image)
            camera_convention_transform = (
                Transform3d()
                .rotate(camera_to_pytorch3d_camera(device=self.device).rotation)
                .to(self.device)
            )
            points_tensor = camera_convention_transform.transform_points(output["pointmaps"])
            intrinsics = output.get("intrinsics", None)

        if loaded_image.shape != points_tensor.shape:
            points_tensor = torch.nn.functional.interpolate(
                points_tensor.permute(2, 0, 1).unsqueeze(0),
                size=(loaded_image.shape[1], loaded_image.shape[2]),
                mode="nearest",
            ).squeeze(0).permute(1, 2, 0)

        if intrinsics is None:
            intrinsics = torch.tensor(
                [[1.0, 0.0, 0.5], [0.0, 1.0, 0.5], [0.0, 0.0, 1.0]],
                dtype=torch.float32,
                device=self.device,
            )
        return {
            "pts_color": loaded_image.to(self.device),
            "intrinsics": intrinsics.to(self.device),
            "pointmap": points_tensor.permute(2, 0, 1).to(self.device),
        }

    def preprocess_image(self, image, preprocessor, pointmap=None):
        import torch
        from sam3d_objects.data.dataset.tdfy.img_and_mask_transforms import get_mask

        if not isinstance(image, np.ndarray):
            image = np.array(image)
        rgba_image = torch.from_numpy(self.image_to_float(image))
        rgba_image = rgba_image.permute(2, 0, 1).contiguous()
        rgb_image = rgba_image[:3]
        rgb_image_mask = get_mask(rgba_image, None, "ALPHA_CHANNEL")

        item = preprocessor._process_image_mask_pointmap_mess(
            rgb_image, rgb_image_mask, pointmap
        )
        out = {
            "mask": item["mask"][None].to(self.device),
            "image": item["image"][None].to(self.device),
            "rgb_image": item["rgb_image"][None].to(self.device),
            "rgb_image_mask": item["rgb_image_mask"][None].to(self.device),
        }
        if pointmap is not None and preprocessor.pointmap_transform != (None,):
            out.update(
                {
                    "pointmap": item["pointmap"][None].to(self.device),
                    "rgb_pointmap": item["rgb_pointmap"][None].to(self.device),
                    "pointmap_scale": item["pointmap_scale"][None].to(self.device),
                    "pointmap_shift": item["pointmap_shift"][None].to(self.device),
                    "rgb_pointmap_scale": item["rgb_pointmap_scale"][None].to(self.device),
                    "rgb_pointmap_shift": item["rgb_pointmap_shift"][None].to(self.device),
                    "rgb_pointmap_unnorm": preprocessor._apply_transform(
                        pointmap, preprocessor.pointmap_transform
                    )[None].to(self.device),
                }
            )
        return out

    def _condition_input(self, input_dict):
        condition_args = [input_dict[k] for k in self.ss_condition_input_mapping]
        condition_kwargs = {
            k: v for k, v in input_dict.items() if k not in self.ss_condition_input_mapping
        }
        tokens = self.condition_embedders["ss_condition_embedder"](
            *condition_args, **condition_kwargs
        )
        return (tokens,), {}

    def sample_sparse_structure(self, ss_input_dict, inference_steps=None, use_distillation=False):
        import torch
        from sam3d_objects.pipeline.inference_utils import (
            downsample_sparse_structure,
            prune_sparse_structure,
        )

        ss_generator = self.models["ss_generator"]
        ss_decoder = self.models["ss_decoder"]
        if use_distillation:
            ss_generator.no_shortcut = False
            ss_generator.reverse_fn.strength = 0
            ss_generator.reverse_fn.strength_pm = 0
        else:
            ss_generator.no_shortcut = True
            ss_generator.reverse_fn.strength = self.ss_cfg_strength
            ss_generator.reverse_fn.strength_pm = self.ss_cfg_strength_pm

        prev_inference_steps = ss_generator.inference_steps
        if inference_steps:
            ss_generator.inference_steps = inference_steps

        image = ss_input_dict["image"]
        bs = image.shape[0]
        latent_shape_dict = {
            k: (bs,) + (v.pos_emb.shape[0], v.input_layer.in_features)
            for k, v in ss_generator.reverse_fn.backbone.latent_mapping.items()
        }
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=self.shape_model_dtype):
            condition_args, condition_kwargs = self._condition_input(ss_input_dict)
            return_dict = ss_generator(
                latent_shape_dict,
                image.device,
                *condition_args,
                **condition_kwargs,
            )
            shape_latent = return_dict["shape"]
            ss = ss_decoder(
                shape_latent.permute(0, 2, 1)
                .contiguous()
                .view(shape_latent.shape[0], 8, 16, 16, 16)
            )
            coords = torch.argwhere(ss > 0)[:, [0, 2, 3, 4]].int()
            return_dict["coords_original"] = coords
            original_shape = coords.shape
            if self.downsample_ss_dist > 0 and coords.shape[0] > 0:
                coords = prune_sparse_structure(
                    coords,
                    max_neighbor_axes_dist=self.downsample_ss_dist,
                )
            coords, downsample_factor = downsample_sparse_structure(coords)
            print(f"[INFO] Downsampled coords from {original_shape[0]} to {coords.shape[0]}")
            return_dict["coords"] = coords
            return_dict["downsample_factor"] = downsample_factor

        ss_generator.inference_steps = prev_inference_steps
        return return_dict


def _filter_prefixed_state_dict(state_dict, prefix: str):
    n = len(prefix)
    out = {k[n:]: v for k, v in state_dict.items() if k.startswith(prefix)}
    if not out:
        raise KeyError(f"checkpoint has no weights with prefix {prefix!r}")
    return out


def _disable_dino_pretrained(config_node) -> None:
    """Avoid torch.hub downloads; condition embedder weights come from ss_generator.ckpt."""
    for embedder_entry in config_node["embedder_list"]:
        embedder_cfg = embedder_entry[0]
        if str(embedder_cfg.get("_target_", "")).endswith(".dino.Dino"):
            embedder_cfg["repo_or_dir"] = "/root/code/arts-gen/pretrained/dinov2"
            embedder_cfg["source"] = "local"
            embedder_cfg["share_backbone_key"] = "sam3d_ss_dinov2_vitl14_reg"
            embedder_cfg.setdefault("backbone_kwargs", {})
            embedder_cfg["backbone_kwargs"]["pretrained"] = False


def _patch_torch_pytree_tree_map_compat() -> None:
    """PyTorch 2.1 has tree_map(fn, tree); SAM3D expects tree_map(fn, *trees)."""
    import inspect
    from torch.utils import _pytree

    original_tree_map = _pytree.tree_map
    if getattr(original_tree_map, "_sam3d_multi_tree_compat", False):
        return
    try:
        if len(inspect.signature(original_tree_map).parameters) > 2:
            return
    except Exception:  # noqa: BLE001
        pass

    def _map_multi_tree(fn, tree, *rests):
        if isinstance(tree, dict):
            for other in rests:
                if not isinstance(other, dict) or set(other.keys()) != set(tree.keys()):
                    raise ValueError("tree_map dict pytrees have different keys")
            return {
                key: _map_multi_tree(fn, tree[key], *(other[key] for other in rests))
                for key in tree.keys()
            }
        if isinstance(tree, tuple) and hasattr(tree, "_fields"):
            for other in rests:
                if not isinstance(other, type(tree)) or len(other) != len(tree):
                    raise ValueError("tree_map namedtuple pytrees have different structures")
            return type(tree)(
                *[
                    _map_multi_tree(fn, tree[idx], *(other[idx] for other in rests))
                    for idx in range(len(tree))
                ]
            )
        if isinstance(tree, (list, tuple)):
            for other in rests:
                if not isinstance(other, type(tree)) or len(other) != len(tree):
                    raise ValueError("tree_map sequence pytrees have different structures")
            values = [
                _map_multi_tree(fn, tree[idx], *(other[idx] for other in rests))
                for idx in range(len(tree))
            ]
            return type(tree)(values)
        return fn(tree, *rests)

    def _tree_map(fn, tree, *rests):
        if not rests:
            return original_tree_map(fn, tree)
        return _map_multi_tree(fn, tree, *rests)

    _tree_map._sam3d_multi_tree_compat = True  # type: ignore[attr-defined]
    _pytree.tree_map = _tree_map


def _build_pipeline(config_path: Path, device: str, ss_generator_ckpt: Path | None = None):
    """Construct only the SAM3D StageA components needed for SS generation.

    The heavy imports happen INSIDE this function so that ``--help`` stays fast
    and does not pull in torch / sam3d_objects.
    """
    # The public sam3d_objects release lacks a `sam3d_objects.init` module that
    # the top-level package tries to import; this env flag short-circuits it.
    # MUST be set BEFORE importing sam3d_objects (see notebook/inference.py:6
    # and surface_voxel/pipeline.py).
    os.environ.setdefault("LIDRA_SKIP_INIT", "true")

    import torch
    from omegaconf import OmegaConf
    import hydra

    _patch_torch_pytree_tree_map_compat()
    import sam3d_objects  # noqa: F401 -- triggers package init side-effects
    from sam3d_objects.pipeline.inference_utils import get_pose_decoder

    config_path = Path(config_path)
    workspace = config_path.parent
    pipe_cfg = OmegaConf.load(str(config_path))
    gen_cfg_path = workspace / str(pipe_cfg["ss_generator_config_path"])
    gen_ckpt_path = (
        Path(ss_generator_ckpt)
        if ss_generator_ckpt is not None
        else workspace / str(pipe_cfg["ss_generator_ckpt_path"])
    )
    dec_cfg_path = workspace / str(pipe_cfg["ss_decoder_config_path"])
    dec_ckpt_path = workspace / str(pipe_cfg["ss_decoder_ckpt_path"])

    gen_cfg = OmegaConf.load(str(gen_cfg_path))
    gen_ckpt = torch.load(str(gen_ckpt_path), map_location="cpu", weights_only=False)
    gen_state = gen_ckpt.get("state_dict", gen_ckpt)
    ss_generator = hydra.utils.instantiate(gen_cfg["module"]["generator"]["backbone"])
    missing, unexpected = ss_generator.load_state_dict(
        _filter_prefixed_state_dict(gen_state, "_base_models.generator."),
        strict=True,
    )
    if missing or unexpected:
        raise RuntimeError(f"ss_generator load failed: missing={missing}, unexpected={unexpected}")

    condition_cfg = gen_cfg["module"]["condition_embedder"]["backbone"]
    _disable_dino_pretrained(condition_cfg)
    ss_condition_embedder = hydra.utils.instantiate(condition_cfg)
    missing, unexpected = ss_condition_embedder.load_state_dict(
        _filter_prefixed_state_dict(gen_state, "_base_models.condition_embedder."),
        strict=True,
    )
    if missing or unexpected:
        raise RuntimeError(
            f"ss_condition_embedder load failed: missing={missing}, unexpected={unexpected}"
        )

    ss_decoder = hydra.utils.instantiate(OmegaConf.load(str(dec_cfg_path)))
    dec_state = torch.load(str(dec_ckpt_path), map_location="cpu", weights_only=True)
    missing, unexpected = ss_decoder.load_state_dict(dec_state, strict=True)
    if missing or unexpected:
        raise RuntimeError(f"ss_decoder load failed: missing={missing}, unexpected={unexpected}")

    device_t = torch.device(device)
    ss_generator = ss_generator.to(device_t).eval()
    ss_condition_embedder = ss_condition_embedder.to(device_t).eval()
    ss_decoder = ss_decoder.to(device_t).eval()
    ss_preprocessor = hydra.utils.instantiate(pipe_cfg["ss_preprocessor"])

    depth_model = None
    moge_model_path = os.environ.get("SAM3D_MOGE_MODEL_PATH", "").strip()
    if not moge_model_path and DEFAULT_OFFLINE_MOGE_MODEL.is_file():
        moge_model_path = str(DEFAULT_OFFLINE_MOGE_MODEL)
    if moge_model_path:
        depth_cfg = OmegaConf.create(OmegaConf.to_container(pipe_cfg["depth_model"], resolve=False))
        depth_cfg.model.pretrained_model_name_or_path = moge_model_path
        depth_model = hydra.utils.instantiate(depth_cfg)
        depth_model.device = device_t
    else:
        print("[WARN] MoGe offline model not found; StageA 将使用 synthetic pointmap fallback")

    return _StageASSOnlyPipeline(
        models={"ss_generator": ss_generator, "ss_decoder": ss_decoder},
        ss_condition_embedder=ss_condition_embedder,
        ss_preprocessor=ss_preprocessor,
        pose_decoder=get_pose_decoder(str(pipe_cfg.get("pose_decoder_name", "ScaleShiftInvariant"))),
        depth_model=depth_model,
        device=device_t,
        dtype=torch.float16 if str(pipe_cfg.get("dtype", "float16")) == "float16" else torch.bfloat16,
        downsample_ss_dist=int(pipe_cfg.get("downsample_ss_dist", 0)),
        ss_condition_input_mapping=list(pipe_cfg.get("ss_condition_input_mapping", [])),
    )


def run_ss_stage(
    *,
    image: np.ndarray,
    mask: np.ndarray,
    config_path: Path,
    out_dir: Path,
    seed: int,
    device: str,
    ss_generator_ckpt: Path | None = None,
) -> dict:
    """Run the SS stage and persist ss_latent.npy + voxel.npz/.bin + pose.json.

    Returns a small summary dict for logging.
    """
    import torch

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rgba = _merge_mask_to_rgba(image, mask)
    print(f"[INFO] rgba input shape={rgba.shape} dtype={rgba.dtype}")

    print(
        f"[INFO] building pipeline from config={config_path} on device={device} "
        f"ss_generator_override={ss_generator_ckpt or '<pipeline.yaml>'}"
    )
    pipe = _build_pipeline(config_path, device, ss_generator_ckpt=ss_generator_ckpt)
    device_t = torch.device(device)

    try:
        with device_t:
            # 1) Pointmap (depth/MoGe) + intrinsics.
            pointmap_dict = pipe.compute_pointmap(rgba, pointmap=None)
            pointmap = pointmap_dict["pointmap"]
            intrinsics = pointmap_dict["intrinsics"]

            # 2) SS preprocessing.
            ss_input_dict = pipe.preprocess_image(
                rgba, pipe.ss_preprocessor, pointmap=pointmap
            )

            # 3) Sample the sparse structure (seeded for reproducibility).
            torch.manual_seed(seed)
            ss_return_dict = pipe.sample_sparse_structure(
                ss_input_dict, inference_steps=None, use_distillation=False
            )

            # 4) Decode pose so we can persist rotation/translation/scale like
            #    the upstream VoxelOutput (mirrors SurfaceVoxelPipeline.__call__).
            pointmap_scale = ss_input_dict.get("pointmap_scale", None)
            pointmap_shift = ss_input_dict.get("pointmap_shift", None)
            ss_return_dict.update(
                pipe.pose_decoder(
                    ss_return_dict,
                    scene_scale=pointmap_scale,
                    scene_shift=pointmap_shift,
                )
            )
            ss_return_dict["scale"] = (
                ss_return_dict["scale"] * ss_return_dict["downsample_factor"]
            )

            # 5) CAPTURE THE LATENT.
            #    return_dict["shape"] is [bs, 4096, 8]; reshape to the
            #    z_global layout (bs, 8, 16, 16, 16) used by the SS decoder and
            #    by the ss_latents_expanded "mean" array.
            z = ss_return_dict["shape"]  # [bs, 4096, 8]
            if z.ndim != 3 or z.shape[1] != 4096 or z.shape[2] != 8:
                raise ValueError(
                    f"unexpected SS latent shape {tuple(z.shape)}; expected (bs, 4096, 8)"
                )
            z_global = (
                z.permute(0, 2, 1)
                .contiguous()
                .view(z.shape[0], 8, 16, 16, 16)[0]
                .detach()
                .float()
                .cpu()
                .numpy()
            )  # (8, 16, 16, 16) float32, bs=1

            # 6) Extract voxel coords (drop the batch column).
            coords = (
                ss_return_dict["coords"][:, 1:].detach().to(int).cpu().numpy()
            )  # (N, 3)

            # Pull pose tensors onto CPU for JSON serialization.
            rotation = ss_return_dict["rotation"].detach().to(torch.float32).cpu()
            translation = ss_return_dict["translation"].detach().to(torch.float32).cpu()
            scale = ss_return_dict["scale"].detach().to(torch.float32).cpu()
            intrinsics_cpu = intrinsics.detach().to(torch.float32).cpu()
            downsample_factor = int(ss_return_dict["downsample_factor"])
    finally:
        # Free VRAM like the wrapper's unload().
        del pipe
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # --- Persist ss_latent.npy (matches ss_latents_expanded format) ---
    z_global = np.ascontiguousarray(z_global, dtype=np.float32)
    latent_path = out_dir / "ss_latent.npy"
    np.save(latent_path, z_global)
    print(f"[INFO] wrote SS latent {z_global.shape} {z_global.dtype} -> {latent_path}")

    # --- Persist voxel.npz + voxel.bin ---
    n_vox = int(coords.shape[0])
    if n_vox == 0:
        raise ValueError("SS stage produced 0 voxels (coords is empty); inference failed")
    save_voxel(out_dir, coords, resolution=64, source="pred")
    print(
        f"[INFO] wrote {n_vox} voxels (resolution=64, source='pred') via "
        f"{_SAVE_VOXEL_SOURCE} -> {out_dir / 'voxel.npz'} + {out_dir / 'voxel.bin'}"
    )

    # --- Persist pose.json (so the SLat stage can reuse Stage-A pose) ---
    pose = {
        "rotation": rotation.tolist(),
        "translation": translation.tolist(),
        "scale": scale.tolist(),
        "intrinsics": intrinsics_cpu.tolist(),
        "downsample_factor": downsample_factor,
        "seed": int(seed),
    }
    pose_path = out_dir / "pose.json"
    with pose_path.open("w") as f:
        json.dump(pose, f, indent=2)
    print(f"[INFO] wrote pose.json (downsample_factor={downsample_factor}) -> {pose_path}")

    return {"num_voxels": n_vox, "latent_shape": tuple(z_global.shape), "out_dir": str(out_dir)}


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="ss-stage",
        description=(
            "Run SAM 3D Objects sparse-structure stage for ONE object; save the "
            "SS latent (8,16,16,16) AND the decoded voxel + pose. Run inside the "
            "sam3d venv (cu121)."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--image", type=Path, required=True, help="Path to RGB image (PNG/JPG).")
    parser.add_argument("--mask", type=Path, required=True, help="Path to binary mask PNG.")
    parser.add_argument(
        "--config", type=Path, required=True, help="Path to SAM 3D pipeline.yaml."
    )
    parser.add_argument(
        "--out", type=Path, required=True, dest="out",
        help="Output run directory (created); receives ss_latent.npy, voxel.npz/.bin, pose.json.",
    )
    parser.add_argument(
        "--ss-generator-ckpt",
        type=Path,
        default=None,
        help=(
            "Optional SS generator checkpoint override. When omitted, the path in "
            "pipeline.yaml is used."
        ),
    )
    parser.add_argument("--seed", type=int, default=42, help="RNG seed for SS sampling.")
    parser.add_argument("--device", type=str, default="cuda", help="Torch device.")
    args = parser.parse_args()

    # Fail loudly on missing inputs (no silent fallback).
    if not args.image.exists():
        raise FileNotFoundError(f"--image not found: {args.image}")
    if not args.mask.exists():
        raise FileNotFoundError(f"--mask not found: {args.mask}")
    if not args.config.exists():
        raise FileNotFoundError(f"--config not found: {args.config}")
    if args.ss_generator_ckpt is not None and not args.ss_generator_ckpt.exists():
        raise FileNotFoundError(f"--ss-generator-ckpt not found: {args.ss_generator_ckpt}")

    image = _load_rgb(args.image)
    mask = _load_mask(args.mask)
    if mask.shape != image.shape[:2]:
        raise ValueError(
            f"mask shape {mask.shape} does not match image HxW {image.shape[:2]}"
        )

    summary = run_ss_stage(
        image=image,
        mask=mask,
        config_path=args.config,
        out_dir=args.out,
        seed=args.seed,
        device=args.device,
        ss_generator_ckpt=args.ss_generator_ckpt,
    )
    print(
        f"[INFO] DONE: {summary['num_voxels']} voxels, "
        f"SS latent {summary['latent_shape']} -> {summary['out_dir']}"
    )


if __name__ == "__main__":
    main()
