from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch

# Must be set BEFORE importing sam3d_objects (mirrors notebook/inference.py:6).
os.environ.setdefault("LIDRA_SKIP_INIT", "true")

from omegaconf import OmegaConf  # noqa: E402
from hydra.utils import instantiate  # noqa: E402

import sam3d_objects  # noqa: F401, E402  -- triggers package init side-effects
from sam3d_objects.pipeline.inference_pipeline_pointmap import (  # noqa: E402
    InferencePipelinePointMap,
)

from texture.types import AppearanceOutput


def _merge_mask_to_rgba(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    mask_u8 = (mask.astype(bool).astype(np.uint8)) * 255
    return np.concatenate([image[..., :3], mask_u8[..., None]], axis=-1)


def _load_voxel_dir(voxel_dir: Path) -> tuple[torch.IntTensor, dict]:
    """Read surface.npy + pose.json (and optional pointmap_unnorm.npy).

    Returns (coords (N,4) int32, pose_dict).
    """
    voxel_dir = Path(voxel_dir)

    surface = np.load(voxel_dir / "surface.npy")
    if surface.ndim != 2 or surface.shape[1] != 3:
        raise ValueError(
            f"surface.npy must have shape (N, 3); got {surface.shape}"
        )
    lo, hi = int(surface.min()), int(surface.max())
    if lo < 0 or hi > 63:
        raise ValueError(
            f"surface.npy coords out of [0, 63] range: min={lo}, max={hi}"
        )

    n = surface.shape[0]
    batch_col = np.zeros((n, 1), dtype=np.int64)
    full = np.concatenate([batch_col, surface.astype(np.int64)], axis=1).astype(np.int32)
    coords = torch.from_numpy(full)

    with (voxel_dir / "pose.json").open() as f:
        pose = json.load(f)

    return coords, pose


class TexturePipeline:
    """Wraps SAM 3D Objects Stage B (slat sampling + decoding).

    Currently loads the full `InferencePipelinePointMap` because
    `instantiate(config)` builds every model declared in pipeline.yaml.
    TODO: skip ss_* loads to save VRAM.
    """

    def __init__(
        self,
        config_path: Path,
        device: str = "cuda",
        *,
        load_mesh_decoder: bool = True,
        load_gs4_decoder: bool = False,
    ):
        config_path = Path(config_path)
        config = OmegaConf.load(str(config_path))
        config.rendering_engine = "pytorch3d"
        config.compile_model = False
        config.workspace_dir = str(config_path.parent)
        config.device = device

        if not load_gs4_decoder:
            # Avoid loading slat_decoder_gs_4.ckpt (~163MB) unless requested.
            config.slat_decoder_gs_4_config_path = None
            config.slat_decoder_gs_4_ckpt_path = None

        self._load_mesh_decoder = load_mesh_decoder
        self._load_gs4_decoder = load_gs4_decoder
        self._pipeline: InferencePipelinePointMap = instantiate(config)
        self._device = torch.device(device)

    def __call__(
        self,
        voxel_dir: Path,
        image: np.ndarray,
        mask: np.ndarray,
        *,
        seed: int | None = None,
        formats: tuple[str, ...] = ("gaussian", "mesh"),
        with_layout_postprocess: bool = False,
    ) -> AppearanceOutput:
        coords, pose = _load_voxel_dir(voxel_dir)

        if not self._load_mesh_decoder and "mesh" in formats:
            raise ValueError(
                "formats includes 'mesh' but mesh decoder was not loaded "
                "(construct TexturePipeline with load_mesh_decoder=True)"
            )
        if "gaussian_4" in formats and not self._load_gs4_decoder:
            raise ValueError(
                "formats includes 'gaussian_4' but gs_4 decoder was not loaded "
                "(construct TexturePipeline with load_gs4_decoder=True)"
            )

        rgba = _merge_mask_to_rgba(image, mask)
        pipe = self._pipeline

        seed_base = int(pose.get("seed", 42)) if seed is None else int(seed)
        slat_seed = seed_base + 1  # offset to decouple from monolithic-run RNG state.

        with self._device:
            coords_dev = coords.to(self._device)

            slat_input_dict = pipe.preprocess_image(rgba, pipe.slat_preprocessor)

            torch.manual_seed(slat_seed)
            slat = pipe.sample_slat(
                slat_input_dict,
                coords_dev,
                inference_steps=None,
                use_distillation=False,
            )

            outputs = pipe.decode_slat(slat, list(formats))
            # Skip pipe.postprocess_slat_output: it eagerly runs to_glb (texture
            # baking / mesh decimation) whenever "mesh" is decoded, which is
            # heavy and not what we want here. We surface the raw mesh and let
            # AppearanceOutput.save() write a vertex-colored .glb via trimesh.
            gs = outputs["gaussian"][0] if "gaussian" in outputs else None
            mesh = outputs["mesh"][0] if "mesh" in outputs else None

            if with_layout_postprocess:
                self._run_layout_postprocess(gs, pose, slat_input_dict)

        num_gaussians = None
        if gs is not None:
            num_gaussians = int(gs.get_xyz.shape[0])

        return AppearanceOutput(
            gs=gs,
            mesh=mesh,
            num_gaussians=num_gaussians,
        )

    def _run_layout_postprocess(
        self,
        gs: Any | None,
        pose: dict,
        slat_input_dict: dict,
    ) -> None:
        """Mirror `inference_pipeline_pointmap.py` ~line 467 layout postprocess.

        Note: this mutates nothing on the AppearanceOutput today because Stage B
        does not own the pose - the refined pose belongs upstream. We surface
        the call so callers who want IoU diagnostics can wire it in.
        """
        if gs is None:
            raise ValueError(
                "with_layout_postprocess=True requires gaussian output (gs is None)"
            )
        if "rgb_pointmap_unnorm" not in slat_input_dict:
            # slat_preprocessor doesn't produce pointmap_unnorm; the layout
            # post-optim needs it. The Stage A pose.json carries intrinsics but
            # not the pointmap - require it explicitly via the voxel_dir.
            raise ValueError(
                "with_layout_postprocess=True requires pointmap_unnorm in "
                "slat_input_dict, which the slat_preprocessor does not produce. "
                "Run post-optim from Stage A instead, or extend this pipeline "
                "to plumb pointmap_unnorm.npy from voxel_dir through."
            )
        intrinsics = pose.get("intrinsics", None)
        if intrinsics is None:
            raise ValueError(
                "with_layout_postprocess=True requires 'intrinsics' in pose.json"
            )

        intrinsics_t = torch.tensor(intrinsics, dtype=torch.float32, device=self._device)
        pose_dict = {
            "rotation": torch.tensor(pose["rotation"], dtype=torch.float32, device=self._device),
            "translation": torch.tensor(pose["translation"], dtype=torch.float32, device=self._device),
            "scale": torch.tensor(pose["scale"], dtype=torch.float32, device=self._device),
        }
        from copy import deepcopy
        self._pipeline.run_post_optimization_GS(
            deepcopy(gs),
            intrinsics_t,
            pose_dict,
            slat_input_dict,
            backend="gsplat",
        )

    def unload(self) -> None:
        del self._pipeline
        self._pipeline = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
