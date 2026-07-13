from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import torch

# The public sam3d_objects release lacks a `sam3d_objects.init` module that the
# top-level package tries to import; this env flag short-circuits that path.
# Must be set BEFORE importing sam3d_objects (see notebook/inference.py:6).
os.environ.setdefault("LIDRA_SKIP_INIT", "true")

from omegaconf import OmegaConf  # noqa: E402
from hydra.utils import instantiate  # noqa: E402

import sam3d_objects  # noqa: F401, E402  -- triggers package init side-effects
from sam3d_objects.pipeline.inference_pipeline_pointmap import (  # noqa: E402
    InferencePipelinePointMap,
)

from surface_voxel.types import VoxelOutput


def _merge_mask_to_rgba(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    mask_u8 = (mask.astype(bool).astype(np.uint8)) * 255
    return np.concatenate([image[..., :3], mask_u8[..., None]], axis=-1)


class SurfaceVoxelPipeline:
    """Wraps SAM 3D Objects sparse-structure stage.

    Loads ss_generator + ss_decoder + depth (MoGe). Note: due to how
    sam3d_objects' InferencePipelinePointMap is structured, this currently
    loads the FULL pipeline (~13GB VRAM) because `instantiate(config)`
    builds every sub-model declared in pipeline.yaml. A future optimization
    could prune `slat_*` entries from the config before instantiating to
    cut memory roughly in half.
    """

    def __init__(self, config_path: Path, device: str = "cuda"):
        config_path = Path(config_path)
        config = OmegaConf.load(str(config_path))
        config.rendering_engine = "pytorch3d"
        config.compile_model = False
        config.workspace_dir = str(config_path.parent)
        config.device = device

        self._pipeline: InferencePipelinePointMap = instantiate(config)
        self._device = torch.device(device)

    def __call__(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        *,
        seed: int = 42,
        keep_layout_aux: bool = True,
    ) -> VoxelOutput:
        rgba = _merge_mask_to_rgba(image, mask)
        pipe = self._pipeline

        with self._device:
            pointmap_dict = pipe.compute_pointmap(rgba, pointmap=None)
            pointmap = pointmap_dict["pointmap"]
            intrinsics = pointmap_dict["intrinsics"]

            ss_input_dict = pipe.preprocess_image(
                rgba, pipe.ss_preprocessor, pointmap=pointmap
            )

            torch.manual_seed(seed)
            ss_return_dict = pipe.sample_sparse_structure(
                ss_input_dict, inference_steps=None, use_distillation=False
            )

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

        pointmap_unnorm = None
        if keep_layout_aux:
            # (1, 3, H, W) -> (H, W, 3); see preprocess_image line ~218
            full = ss_input_dict.get("rgb_pointmap_unnorm", None)
            if full is not None:
                pointmap_unnorm = full[0].detach().permute(1, 2, 0).contiguous().cpu()

        return VoxelOutput(
            coords=ss_return_dict["coords"].detach().to(torch.int32).cpu(),
            downsample_factor=int(ss_return_dict["downsample_factor"]),
            rotation=ss_return_dict["rotation"].detach().to(torch.float32).cpu(),
            translation=ss_return_dict["translation"].detach().to(torch.float32).cpu(),
            scale=ss_return_dict["scale"].detach().to(torch.float32).cpu(),
            intrinsics=intrinsics.detach().to(torch.float32).cpu(),
            pointmap_unnorm=pointmap_unnorm,
            seed=int(seed),
        )

    def unload(self) -> None:
        del self._pipeline
        self._pipeline = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
