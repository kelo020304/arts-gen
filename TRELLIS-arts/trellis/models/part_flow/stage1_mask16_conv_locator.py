"""High-resolution Conv3D Stage1 locator: 2D masks + global latent -> mask16."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .mask16_unet_blocks import make_coords3d


__all__ = ["Stage1Mask16ConvLocator"]


def _num_groups(channels: int) -> int:
    for groups in (32, 16, 8, 4, 2, 1):
        if channels % groups == 0:
            return groups
    return 1


class ConvFiLMResBlock3D(nn.Module):
    def __init__(self, channels: int, cond_dim: int, *, dilation: int = 1):
        super().__init__()
        self.channels = int(channels)
        self.norm1 = nn.GroupNorm(_num_groups(self.channels), self.channels)
        self.conv1 = nn.Conv3d(
            self.channels,
            self.channels,
            3,
            padding=int(dilation),
            dilation=int(dilation),
        )
        self.norm2 = nn.GroupNorm(_num_groups(self.channels), self.channels)
        self.cond = nn.Linear(int(cond_dim), self.channels * 2)
        self.conv2 = nn.Conv3d(self.channels, self.channels, 3, padding=1)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        gamma, beta = self.cond(cond).chunk(2, dim=1)
        h = self.norm2(h)
        h = h * (1.0 + gamma.view(gamma.shape[0], -1, 1, 1, 1)) + beta.view(beta.shape[0], -1, 1, 1, 1)
        h = self.conv2(F.silu(h))
        return x + h


class ParallelDilatedMixer3D(nn.Module):
    """Fuse same-resolution local and dilated 3D context without downsampling."""

    def __init__(self, channels: int):
        super().__init__()
        self.channels = int(channels)
        branch_channels = max(1, self.channels // 4)
        self.norm = nn.GroupNorm(_num_groups(self.channels), self.channels)
        self.branches = nn.ModuleList(
            [
                nn.Conv3d(self.channels, branch_channels, 3, padding=d, dilation=d)
                for d in (1, 2, 3)
            ]
        )
        self.local = nn.Conv3d(self.channels, branch_channels, 1)
        self.fuse = nn.Conv3d(branch_channels * 4, self.channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.silu(self.norm(x))
        parts = [self.local(h)]
        parts.extend(branch(h) for branch in self.branches)
        return x + self.fuse(torch.cat(parts, dim=1))


class Stage1Mask16ConvLocator(nn.Module):
    """Pure high-resolution Conv3D locator.

    The 3D trunk never downsamples below 16^3, preserving tiny-part support
    detail. 2D masks are encoded by shared Conv2D and fused by an MLP; no 3D
    transformer or 4^3 bottleneck is used.
    """

    def __init__(
        self,
        *,
        latent_channels: int = 8,
        base_channels: int = 256,
        cond_dim: int = 256,
        num_views: int = 4,
        mask_size: int = 64,
        num_blocks: int = 12,
        dilations: tuple[int, ...] | list[int] | None = None,
        use_view_index: bool = False,
        max_view_index: int = 64,
    ):
        super().__init__()
        self.latent_channels = int(latent_channels)
        self.base_channels = int(base_channels)
        self.cond_dim = int(cond_dim)
        self.num_views = int(num_views)
        self.mask_size = int(mask_size)
        self.num_blocks = int(num_blocks)
        self.use_view_index = bool(use_view_index)
        y, x = torch.meshgrid(
            torch.linspace(-1.0, 1.0, self.mask_size),
            torch.linspace(-1.0, 1.0, self.mask_size),
            indexing="ij",
        )
        self.register_buffer("coords2d", torch.stack([x, y], dim=0), persistent=False)
        self.register_buffer("coords3d", make_coords3d(16), persistent=False)

        self.mask_encoder = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.GroupNorm(8, 64),
            nn.SiLU(),
            nn.Conv2d(64, 128, 4, stride=2, padding=1),
            nn.GroupNorm(16, 128),
            nn.SiLU(),
            nn.Conv2d(128, 192, 4, stride=2, padding=1),
            nn.GroupNorm(16, 192),
            nn.SiLU(),
            nn.Conv2d(192, cond_dim, 4, stride=2, padding=1),
            nn.GroupNorm(16, cond_dim),
            nn.SiLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.view_slot = nn.Embedding(self.num_views, cond_dim)
        self.view_index_embed = nn.Embedding(max_view_index, cond_dim) if self.use_view_index else None
        self.view_fuse = nn.Sequential(
            nn.Linear(cond_dim * self.num_views, cond_dim * 2),
            nn.SiLU(),
            nn.Linear(cond_dim * 2, cond_dim),
            nn.SiLU(),
            nn.Linear(cond_dim, cond_dim),
        )
        self.in_proj = nn.Conv3d(self.latent_channels + 3, self.base_channels, 3, padding=1)
        if dilations is None:
            dilations = (1, 1, 2, 2, 3, 3, 2, 2, 1, 1, 1, 1)
        if len(dilations) < self.num_blocks:
            repeats = (self.num_blocks + len(dilations) - 1) // len(dilations)
            dilations = tuple(dilations) * repeats
        self.blocks = nn.ModuleList(
            [
                ConvFiLMResBlock3D(self.base_channels, self.cond_dim, dilation=int(dilations[i]))
                for i in range(self.num_blocks)
            ]
        )
        self.mixers = nn.ModuleDict({
            str(i): ParallelDilatedMixer3D(self.base_channels)
            for i in range(2, self.num_blocks, 3)
        })
        self.skip_fuse = nn.Conv3d(self.base_channels * 2, self.base_channels, 1)
        self.out = nn.Sequential(
            nn.GroupNorm(_num_groups(self.base_channels), self.base_channels),
            nn.SiLU(),
            nn.Conv3d(self.base_channels, 1, 3, padding=1),
        )

    def encode_views(self, masks2d: torch.Tensor, view_indices: torch.Tensor | None = None) -> torch.Tensor:
        if masks2d.dim() != 4:
            raise ValueError(f"masks2d expected [B,V,H,W], got {tuple(masks2d.shape)}")
        b, v, h, w = masks2d.shape
        if v != self.num_views:
            raise ValueError(f"model num_views={self.num_views}, got {v}")
        if h != self.mask_size or w != self.mask_size:
            raise ValueError(f"model mask_size={self.mask_size}, got {h}x{w}")
        coords = self.coords2d.to(device=masks2d.device, dtype=masks2d.dtype)
        coords = coords.view(1, 1, 2, h, w).expand(b, v, -1, -1, -1)
        x2 = torch.cat([masks2d.float().view(b, v, 1, h, w), coords.float()], dim=2)
        feat = self.mask_encoder(x2.reshape(b * v, 3, h, w)).view(b, v, self.cond_dim)
        slot_ids = torch.arange(v, device=masks2d.device)
        feat = feat + self.view_slot(slot_ids).view(1, v, self.cond_dim)
        if self.view_index_embed is not None and view_indices is not None:
            feat = feat + self.view_index_embed(
                view_indices.clamp_min(0).clamp_max(self.view_index_embed.num_embeddings - 1)
            )
        return self.view_fuse(feat.reshape(b, v * self.cond_dim))

    def forward(
        self,
        z_global: torch.Tensor,
        masks2d: torch.Tensor,
        view_indices: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if z_global.dim() != 5:
            raise ValueError(f"z_global expected [B,C,16,16,16], got {tuple(z_global.shape)}")
        b, c, d, h, w = z_global.shape
        if c != self.latent_channels or (d, h, w) != (16, 16, 16):
            raise ValueError(f"z_global expected [B,{self.latent_channels},16,16,16], got {tuple(z_global.shape)}")
        cond = self.encode_views(masks2d, view_indices=view_indices)
        coords = self.coords3d.to(device=z_global.device, dtype=z_global.dtype).unsqueeze(0).expand(b, -1, -1, -1, -1)
        x0 = self.in_proj(torch.cat([z_global.float(), coords.float()], dim=1))
        x = x0
        for idx, block in enumerate(self.blocks):
            x = block(x, cond)
            key = str(idx)
            if key in self.mixers:
                x = self.mixers[key](x)
        x = self.skip_fuse(torch.cat([x, x0], dim=1))
        return self.out(x).squeeze(1)
