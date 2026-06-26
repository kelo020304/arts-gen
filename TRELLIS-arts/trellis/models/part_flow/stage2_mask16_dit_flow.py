"""Single-part mask16-conditioned DiT flow for SS part latents."""

from __future__ import annotations

import torch
import torch.nn as nn

from trellis.models.sparse_structure_flow import SparseStructureFlowModel
from trellis.modules.spatial import patchify
from trellis.modules.transformer import AbsolutePositionEmbedder


__all__ = ["Stage2Mask16DiTFlow"]


class Stage2Mask16DiTFlow(nn.Module):
    """Generate one target part SS latent from noise, global SS latent, and mask16.

    This is intentionally single-part and condition-only:

    - target stream: concat(x_t_part, part_mask16)
    - condition stream: patchified z_global
    - output: velocity for x_t_part

    There is no DINO token, part name, 3D box, canonical crop, decode loss, or
    cross-part identity machinery in this model.
    """

    def __init__(
        self,
        *,
        resolution: int = 16,
        latent_channels: int = 8,
        model_channels: int = 1024,
        num_blocks: int = 12,
        num_heads: int = 16,
        patch_size: int = 1,
        global_cond_patch_size: int = 1,
        use_checkpoint: bool = False,
        use_fp16: bool = False,
    ):
        super().__init__()
        self.resolution = int(resolution)
        self.latent_channels = int(latent_channels)
        self.model_channels = int(model_channels)
        self.patch_size = int(patch_size)
        self.global_cond_patch_size = int(global_cond_patch_size)
        if self.resolution != 16:
            raise ValueError("Stage2Mask16DiTFlow currently expects resolution=16")
        if self.resolution % self.patch_size != 0:
            raise ValueError(f"resolution={self.resolution} must be divisible by patch_size={self.patch_size}")
        if self.resolution % self.global_cond_patch_size != 0:
            raise ValueError(
                f"resolution={self.resolution} must be divisible by "
                f"global_cond_patch_size={self.global_cond_patch_size}"
            )

        cond_patch_dim = self.latent_channels * self.global_cond_patch_size ** 3
        self.global_cond_layer = nn.Linear(cond_patch_dim, self.model_channels)
        self.global_pos_embedder = AbsolutePositionEmbedder(self.model_channels, 3)
        grid = self.resolution // self.global_cond_patch_size
        coords = torch.meshgrid(
            torch.arange(grid, dtype=torch.float32),
            torch.arange(grid, dtype=torch.float32),
            torch.arange(grid, dtype=torch.float32),
            indexing="ij",
        )
        self.register_buffer("_global_coords", torch.stack(coords, dim=-1).reshape(-1, 3), persistent=False)
        self.global_type = nn.Parameter(torch.randn(1, self.model_channels) * 0.02)

        self.backbone = SparseStructureFlowModel(
            resolution=self.resolution,
            in_channels=self.latent_channels + 1,
            model_channels=self.model_channels,
            cond_channels=self.model_channels,
            out_channels=self.latent_channels,
            num_blocks=int(num_blocks),
            num_heads=int(num_heads),
            patch_size=self.patch_size,
            pe_mode="ape",
            use_fp16=bool(use_fp16),
            use_checkpoint=bool(use_checkpoint),
            qk_rms_norm=True,
            qk_rms_norm_cross=False,
            use_camera_pose=False,
        )

    def encode_global(self, z_global: torch.Tensor) -> torch.Tensor:
        if z_global.dim() != 5:
            raise ValueError(f"z_global expected [B,C,16,16,16], got {tuple(z_global.shape)}")
        b, c, d, h, w = z_global.shape
        if c != self.latent_channels or (d, h, w) != (self.resolution, self.resolution, self.resolution):
            raise ValueError(
                f"z_global expected [B,{self.latent_channels},{self.resolution},{self.resolution},{self.resolution}], "
                f"got {tuple(z_global.shape)}"
            )
        patches = patchify(z_global, self.global_cond_patch_size)
        tokens = patches.view(patches.shape[0], patches.shape[1], -1).permute(0, 2, 1).contiguous()
        tokens = self.global_cond_layer(tokens)
        pos = self.global_pos_embedder(self._global_coords.to(device=z_global.device)).to(dtype=tokens.dtype)
        return tokens + pos.unsqueeze(0) + self.global_type.to(dtype=tokens.dtype).view(1, 1, -1)

    def _normalize_mask(self, part_mask16: torch.Tensor, *, batch_size: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        if part_mask16.dim() == 4:
            expected = (batch_size, self.resolution, self.resolution, self.resolution)
            if tuple(part_mask16.shape) != expected:
                raise ValueError(f"part_mask16 expected {expected}, got {tuple(part_mask16.shape)}")
            mask = part_mask16.unsqueeze(1)
        elif part_mask16.dim() == 5:
            expected = (batch_size, 1, self.resolution, self.resolution, self.resolution)
            if tuple(part_mask16.shape) != expected:
                raise ValueError(f"part_mask16 expected {expected}, got {tuple(part_mask16.shape)}")
            mask = part_mask16
        else:
            raise ValueError(
                f"part_mask16 expected [B,16,16,16] or [B,1,16,16,16], got {tuple(part_mask16.shape)}"
            )
        return mask.to(device=device, dtype=dtype)

    def forward(
        self,
        x_t_part: torch.Tensor,
        t: torch.Tensor,
        z_global: torch.Tensor,
        part_mask16: torch.Tensor,
    ) -> torch.Tensor:
        if x_t_part.dim() != 5:
            raise ValueError(f"x_t_part expected [B,C,16,16,16], got {tuple(x_t_part.shape)}")
        b, c, d, h, w = x_t_part.shape
        if c != self.latent_channels or (d, h, w) != (self.resolution, self.resolution, self.resolution):
            raise ValueError(
                f"x_t_part expected [B,{self.latent_channels},{self.resolution},{self.resolution},{self.resolution}], "
                f"got {tuple(x_t_part.shape)}"
            )
        if tuple(z_global.shape) != tuple(x_t_part.shape):
            raise ValueError(f"z_global shape {tuple(z_global.shape)} must match x_t_part {tuple(x_t_part.shape)}")
        if t.dim() != 1 or t.shape[0] != b:
            raise ValueError(f"t expected [B], got {tuple(t.shape)}")
        mask = self._normalize_mask(part_mask16, batch_size=b, device=x_t_part.device, dtype=x_t_part.dtype)
        cond = self.encode_global(z_global)
        x_in = torch.cat([x_t_part, mask], dim=1)
        return self.backbone(x_in, t, cond)
