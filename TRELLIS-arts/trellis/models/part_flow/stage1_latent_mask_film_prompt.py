"""FiLM-prompt latent-mask decoder for TRELLIS SS latents.

The model predicts part membership on the 16^3 SS latent lattice. It keeps the
spatial inductive bias that worked in the small Stage1 locator while adding a
prompt-conditioned dynamic mask head inspired by promptable mask decoders.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .mask16_unet_blocks import make_coords3d


__all__ = ["Stage1LatentMaskFiLMPrompt"]


def _num_groups(channels: int) -> int:
    for groups in (32, 16, 8, 4, 2, 1):
        if channels % groups == 0:
            return groups
    return 1


class ZeroInitFiLMResBlock3D(nn.Module):
    """Full-resolution 3D ResBlock with conservative FiLM conditioning."""

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

        nn.init.zeros_(self.cond.weight)
        nn.init.zeros_(self.cond.bias)
        nn.init.zeros_(self.conv2.weight)
        nn.init.zeros_(self.conv2.bias)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        gamma, beta = self.cond(cond).chunk(2, dim=1)
        h = self.norm2(h)
        h = h * (1.0 + gamma.view(gamma.shape[0], -1, 1, 1, 1)) + beta.view(beta.shape[0], -1, 1, 1, 1)
        h = self.conv2(F.silu(h))
        return x + h


class Stage1LatentMaskFiLMPrompt(nn.Module):
    """Predict a target part mask on the 16^3 SS latent lattice.

    Inputs:
      z_global: [B, 8, 16, 16, 16]
      masks2d: [B, V, H, W]

    Output:
      latent_mask_logits: [B, 16, 16, 16]
    """

    def __init__(
        self,
        *,
        latent_channels: int = 8,
        model_channels: int = 256,
        cond_dim: int = 256,
        num_views: int = 4,
        mask_size: int = 64,
        num_blocks: int = 12,
        use_view_index: bool = False,
        max_view_index: int = 64,
    ):
        super().__init__()
        self.latent_channels = int(latent_channels)
        self.model_channels = int(model_channels)
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
            nn.Conv2d(192, self.cond_dim, 4, stride=2, padding=1),
            nn.GroupNorm(_num_groups(self.cond_dim), self.cond_dim),
            nn.SiLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.view_slot = nn.Embedding(self.num_views, self.cond_dim)
        self.view_index_embed = nn.Embedding(max_view_index, self.cond_dim) if self.use_view_index else None
        self.view_fuse = nn.Sequential(
            nn.Linear(self.cond_dim * self.num_views, self.cond_dim * 2),
            nn.SiLU(),
            nn.Linear(self.cond_dim * 2, self.cond_dim),
            nn.SiLU(),
            nn.Linear(self.cond_dim, self.cond_dim),
        )

        self.in_proj = nn.Sequential(
            nn.Conv3d(self.latent_channels + 3, self.model_channels, 3, padding=1),
            nn.GroupNorm(_num_groups(self.model_channels), self.model_channels),
            nn.SiLU(),
            nn.Conv3d(self.model_channels, self.model_channels, 3, padding=1),
        )
        base_pattern = (1, 1, 2, 1, 1, 2, 1, 1, 2, 1, 1, 1)
        repeats = (self.num_blocks + len(base_pattern) - 1) // len(base_pattern)
        dilations = (base_pattern * repeats)[: self.num_blocks]
        self.blocks = nn.ModuleList(
            [
                ZeroInitFiLMResBlock3D(self.model_channels, self.cond_dim, dilation=int(dilation))
                for dilation in dilations
            ]
        )

        self.feat_norm = nn.GroupNorm(_num_groups(self.model_channels), self.model_channels)
        self.feat_proj = nn.Conv3d(self.model_channels, self.model_channels, 1)
        self.static_head = nn.Sequential(
            nn.GroupNorm(_num_groups(self.model_channels), self.model_channels),
            nn.SiLU(),
            nn.Conv3d(self.model_channels, self.model_channels // 2, 3, padding=1),
            nn.GroupNorm(_num_groups(self.model_channels // 2), self.model_channels // 2),
            nn.SiLU(),
            nn.Conv3d(self.model_channels // 2, 1, 1),
        )
        self.dynamic_head = nn.Sequential(
            nn.LayerNorm(self.cond_dim),
            nn.Linear(self.cond_dim, self.model_channels * 2),
            nn.GELU(),
            nn.Linear(self.model_channels * 2, self.model_channels + 1),
        )
        with torch.no_grad():
            self.static_head[-1].bias.fill_(-2.0)
            self.dynamic_head[-1].bias[-1].fill_(-2.0)

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
        feat = feat + self.view_slot(slot_ids).to(dtype=feat.dtype).view(1, v, self.cond_dim)
        if self.view_index_embed is not None and view_indices is not None:
            idx = view_indices.clamp_min(0).clamp_max(self.view_index_embed.num_embeddings - 1)
            feat = feat + self.view_index_embed(idx).to(dtype=feat.dtype)
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
        coords = self.coords3d.to(device=z_global.device, dtype=z_global.dtype)
        coords = coords.unsqueeze(0).expand(b, -1, -1, -1, -1)
        x = self.in_proj(torch.cat([z_global.float(), coords.float()], dim=1))
        for block in self.blocks:
            x = block(x, cond)

        static_logits = self.static_head(x).squeeze(1)
        feat = self.feat_proj(F.silu(self.feat_norm(x))).flatten(2).transpose(1, 2).contiguous()
        dynamic = self.dynamic_head(cond)
        weight = dynamic[:, :-1]
        bias = dynamic[:, -1]
        dynamic_logits = (feat * weight.unsqueeze(1)).sum(dim=-1) / math.sqrt(float(self.model_channels))
        dynamic_logits = dynamic_logits + bias.unsqueeze(1)
        dynamic_logits = dynamic_logits.view(b, 16, 16, 16)
        return static_logits + dynamic_logits
