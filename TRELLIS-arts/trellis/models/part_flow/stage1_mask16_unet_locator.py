"""Full-capacity Stage1 locator: 2D masks + global SS latent -> mask16."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .mask16_unet_blocks import Downsample3D, FiLMResBlock3D, TokenTransformer3D, Upsample3D, make_coords3d


__all__ = ["Stage1Mask16UNetLocator"]


class Stage1Mask16UNetLocator(nn.Module):
    """Predict a target part's 16^3 support mask.

    Train/infer contract:
      input  = complete object SS latent + ordered binary 2D masks
      output = target part mask16 logits

    No camera matrices, 3D boxes, DINO tokens, or part names are inputs.
    """

    def __init__(
        self,
        *,
        latent_channels: int = 8,
        base_channels: int = 256,
        cond_dim: int = 256,
        num_views: int = 4,
        mask_size: int = 64,
        view_transformer_layers: int = 4,
        view_transformer_heads: int = 8,
        bottleneck_transformer_layers: int = 2,
        bottleneck_transformer_heads: int = 8,
        use_view_index: bool = False,
        max_view_index: int = 64,
    ):
        super().__init__()
        self.latent_channels = int(latent_channels)
        self.base_channels = int(base_channels)
        self.cond_dim = int(cond_dim)
        self.num_views = int(num_views)
        self.mask_size = int(mask_size)
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
            nn.Conv2d(cond_dim, cond_dim, 4, stride=2, padding=1),
            nn.GroupNorm(16, cond_dim),
            nn.SiLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.view_slot = nn.Embedding(self.num_views, cond_dim)
        self.view_index_embed = nn.Embedding(max_view_index, cond_dim) if self.use_view_index else None
        self.view_cls = nn.Parameter(torch.zeros(1, 1, cond_dim))
        view_layer = nn.TransformerEncoderLayer(
            d_model=cond_dim,
            nhead=int(view_transformer_heads),
            dim_feedforward=cond_dim * 4,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.view_transformer = nn.TransformerEncoder(view_layer, num_layers=int(view_transformer_layers))
        self.view_norm = nn.LayerNorm(cond_dim)

        c0 = self.base_channels
        c1 = self.base_channels * 2
        c2 = self.base_channels * 4
        self.in_proj = nn.Conv3d(self.latent_channels + 3, c0, 3, padding=1)
        self.enc0 = nn.ModuleList([
            FiLMResBlock3D(c0, c0, cond_dim, zero_init_cond=False),
            FiLMResBlock3D(c0, c0, cond_dim, zero_init_cond=False),
        ])
        self.down0 = Downsample3D(c0, c1)
        self.enc1 = nn.ModuleList([
            FiLMResBlock3D(c1, c1, cond_dim, zero_init_cond=False),
            FiLMResBlock3D(c1, c1, cond_dim, zero_init_cond=False),
        ])
        self.down1 = Downsample3D(c1, c2)
        self.mid = nn.ModuleList([
            FiLMResBlock3D(c2, c2, cond_dim, zero_init_cond=False),
            FiLMResBlock3D(c2, c2, cond_dim, zero_init_cond=False),
        ])
        self.mid_attn = TokenTransformer3D(
            c2,
            spatial_size=4,
            num_layers=int(bottleneck_transformer_layers),
            num_heads=int(bottleneck_transformer_heads),
        )
        self.up1 = Upsample3D(c2, c1)
        self.dec1 = nn.ModuleList([
            FiLMResBlock3D(c1 + c1, c1, cond_dim, zero_init_cond=False),
            FiLMResBlock3D(c1, c1, cond_dim, zero_init_cond=False),
        ])
        self.up0 = Upsample3D(c1, c0)
        self.dec0 = nn.ModuleList([
            FiLMResBlock3D(c0 + c0, c0, cond_dim, zero_init_cond=False),
            FiLMResBlock3D(c0, c0, cond_dim, zero_init_cond=False),
        ])
        self.out = nn.Sequential(
            nn.GroupNorm(32, c0),
            nn.SiLU(),
            nn.Conv3d(c0, 1, 3, padding=1),
        )
        nn.init.zeros_(self.view_cls)
        nn.init.constant_(self.out[-1].bias, -4.0)

    @staticmethod
    def _fourier_view_embed(view_indices: torch.Tensor, dim: int) -> torch.Tensor:
        half = dim // 2
        device = view_indices.device
        dtype = torch.float32
        freqs = torch.arange(half, device=device, dtype=dtype)
        freqs = torch.pow(2.0, freqs / max(half, 1))
        angles = (view_indices.float().unsqueeze(-1) / 64.0) * (2.0 * math.pi)
        emb = torch.cat([torch.sin(angles * freqs), torch.cos(angles * freqs)], dim=-1)
        if emb.shape[-1] < dim:
            emb = F.pad(emb, (0, dim - emb.shape[-1]))
        return emb

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
        tokens = self.mask_encoder(x2.reshape(b * v, 3, h, w)).view(b, v, self.cond_dim)
        slot_ids = torch.arange(v, device=masks2d.device)
        tokens = tokens + self.view_slot(slot_ids).view(1, v, self.cond_dim)
        if view_indices is not None:
            tokens = tokens + self._fourier_view_embed(view_indices, self.cond_dim).to(dtype=tokens.dtype)
            if self.view_index_embed is not None:
                tokens = tokens + self.view_index_embed(
                    view_indices.clamp_min(0).clamp_max(self.view_index_embed.num_embeddings - 1)
                )
        cls = self.view_cls.to(device=tokens.device, dtype=tokens.dtype).expand(b, -1, -1)
        encoded = self.view_transformer(torch.cat([cls, tokens], dim=1))
        return self.view_norm(encoded[:, 0])

    @staticmethod
    def _run_blocks(blocks: nn.ModuleList, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        for block in blocks:
            x = block(x, cond)
        return x

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
        x = self.in_proj(torch.cat([z_global.float(), coords.float()], dim=1))
        e0 = self._run_blocks(self.enc0, x, cond)
        e1 = self._run_blocks(self.enc1, self.down0(e0), cond)
        m = self._run_blocks(self.mid, self.down1(e1), cond)
        m = self.mid_attn(m)
        d1 = self._run_blocks(self.dec1, torch.cat([self.up1(m), e1], dim=1), cond)
        d0 = self._run_blocks(self.dec0, torch.cat([self.up0(d1), e0], dim=1), cond)
        return self.out(d0).squeeze(1)
