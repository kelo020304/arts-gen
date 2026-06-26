"""Stage 4 rendering utilities (shared by eval_slat_render.py and
Stage4Trainer.run_snapshot).

WARNING -- GT camera alignment pending:
  The Blender GT render camera metadata at /sda1 has NOT been confirmed.
  eval_slat_render.py uses TRELLIS canonical cameras (radius=2, fov=40 deg,
  up=[0,0,1], yaws 4-split) to render generated Gaussian representations;
  PSNR/SSIM vs GT Blender renders at smoke scale has NO statistical meaning.
  Real camera alignment + paper numbers are delivered by Phase 5.

Functions:
  load_slat_decoder(ckpt_path)        -> nn.Module (SLatGaussianDecoder, frozen)
  load_gaussian_renderer()            -> GaussianRenderer
  get_canonical_cameras(num_views)    -> (extrinsics [V,4,4], intrinsics [V,3,3])
  un_normalize_slat(z, mean, std)     -> SparseTensor (feats * std + mean)
  render_sample_to_views(gs, ext, intr, renderer) -> Tensor [V,3,H,W] in [0,1]

Sources:
  - RESEARCH section 5 SLat Decoder + Renderer Integration
  - TRELLIS canonical renderer: trellis/utils/render_utils.py:43-70
  - SLatVisMixin camera: trellis/datasets/structured_latent.py:62-79
"""

import math
from typing import Tuple

import torch
import torch.nn as nn


# NOTE: trellis stub must be set up by caller (eval_slat_render.py or train.py)
# before importing this module. render_utils does NOT set up stubs itself.


def load_slat_decoder(
    ckpt_basename: str = 'pretrained/ckpts/slat_dec_gs_swin8_B_64l8gs32_fp16',
) -> nn.Module:
    """Load frozen SLat Gaussian decoder directly from json+safetensors.

    Mirrors losses.py:DecodeAwareLoss._load_frozen_decoder() — does NOT use
    trellis.models.from_pretrained (that requires __init__.py to execute, but
    trellis.models is registered as a stub in train.py to avoid rembg/gradio).

    Args:
        ckpt_basename: path without extension (.json + .safetensors are appended)

    Returns:
        SLatGaussianDecoder on cuda, .eval() mode, requires_grad=False
    """
    import json
    from safetensors.torch import load_file
    from trellis.models.structured_latent_vae.decoder_gs import SLatGaussianDecoder

    config_path = f'{ckpt_basename}.json'
    weights_path = f'{ckpt_basename}.safetensors'

    with open(config_path, 'r') as f:
        config = json.load(f)

    decoder = SLatGaussianDecoder(**config['args'])
    state_dict = load_file(weights_path)
    decoder.load_state_dict(state_dict, strict=True)
    decoder = decoder.cuda().eval()
    decoder.requires_grad_(False)
    print(f'[render_utils] loaded SLat decoder: {config.get("name", type(decoder).__name__)}')
    return decoder


def load_gaussian_renderer():
    """Construct GaussianRenderer with TRELLIS canonical defaults.

    Values from trellis/utils/render_utils.py:52-60 (get_renderer for Gaussian type).
    """
    from trellis.renderers import GaussianRenderer
    renderer = GaussianRenderer()
    renderer.rendering_options.resolution = 512
    renderer.rendering_options.near = 0.8
    renderer.rendering_options.far = 1.6
    renderer.rendering_options.bg_color = (0, 0, 0)
    renderer.rendering_options.ssaa = 1
    renderer.pipe.kernel_size = 0.1
    renderer.pipe.use_mip_gaussian = True
    print('[render_utils] GaussianRenderer ready (res=512, fov derived per-call, bg=black)')
    return renderer


def get_canonical_cameras(
    num_views: int = 4,
    radius: float = 2.0,
    fov_deg: float = 40.0,
    pitch_rad: float = 0.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Generate TRELLIS-canonical multi-view cameras.

    Yaws evenly split the full circle (0, 2pi/V, 4pi/V, ...).
    Pitch fixed at 0 for determinism (SLatVisMixin uses random +/-pi/4, we use fixed).
    Radius = 2 (object centered at origin, on unit-ish sphere at 2x).
    FoV = 40 deg (both x and y).
    Up vector = [0, 0, 1] (Z-up, TRELLIS convention).

    Args:
        num_views: number of views (4 default)
        radius: camera distance from origin
        fov_deg: field of view in degrees (used for both x and y)
        pitch_rad: fixed pitch in radians (0 = horizontal)

    Returns:
        extrinsics: Tensor [V, 4, 4]
        intrinsics: Tensor [V, 3, 3]
    """
    from trellis.utils.render_utils import yaw_pitch_r_fov_to_extrinsics_intrinsics

    # Pass Python lists -- the TRELLIS function expects list[scalar] or scalar,
    # NOT torch tensors (float(tensor) fails for multi-element tensors).
    # fov must be in DEGREES: the function applies deg2rad internally.
    yaws_list = [i * 2 * math.pi / num_views for i in range(num_views)]
    pitches_list = [pitch_rad] * num_views

    extrinsics, intrinsics = yaw_pitch_r_fov_to_extrinsics_intrinsics(
        yaws_list, pitches_list, rs=radius, fovs=fov_deg,
    )
    return torch.stack(extrinsics), torch.stack(intrinsics)


def un_normalize_slat(z_slat, mean: torch.Tensor, std: torch.Tensor):
    """Un-normalize SLat sparse features before sending to frozen decoder.

    CRITICAL: dataset applies (feats - mean) / std during __getitem__;
    decoder expects UN-NORMALIZED features (trained on raw SLat VAE output).
    See RESEARCH section 5.2 and SLatVisMixin.decode_latent:45-55.

    Args:
        z_slat: SparseTensor with normalized feats [sum(N), 8]
        mean: Tensor [8] or [1,8]
        std:  Tensor [8] or [1,8]

    Returns:
        SparseTensor with feats un-normalized (preserves coords/layout)
    """
    mean = mean.to(z_slat.feats.device).view(1, -1)
    std = std.to(z_slat.feats.device).view(1, -1)
    raw_feats = z_slat.feats * std + mean
    return z_slat.replace(raw_feats)


@torch.no_grad()
def render_sample_to_views(
    gaussian,
    extrinsics: torch.Tensor,
    intrinsics: torch.Tensor,
    renderer,
) -> torch.Tensor:
    """Render a Gaussian representation from multiple cameras.

    Args:
        gaussian: single Gaussian object (for one sample in batch)
        extrinsics: [V, 4, 4]
        intrinsics: [V, 3, 3]
        renderer: GaussianRenderer

    Returns:
        Tensor [V, 3, H, W] in [0, 1]
    """
    views = []
    for v in range(extrinsics.shape[0]):
        res = renderer.render(gaussian, extrinsics[v], intrinsics[v], colors_overwrite=None)
        views.append(res['color'])  # [3, H, W]
    return torch.stack(views, dim=0).clamp(0, 1)
