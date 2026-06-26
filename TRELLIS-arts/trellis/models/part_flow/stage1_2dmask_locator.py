"""Stage1 locator: 2D part masks + global SS latent -> part mask16."""

from __future__ import annotations

import torch
import torch.nn as nn


__all__ = ["Stage1TwoDMaskLocator"]


class Stage1TwoDMaskLocator(nn.Module):
    """Predict a part's 16^3 support mask from z_global and 2D part masks.

    This module intentionally has no camera-matrix or 3D-box inputs. The caller
    supplies ordered binary 2D part masks using the same view ordering at train
    and inference time.
    """

    def __init__(
        self,
        *,
        latent_channels: int = 8,
        hidden_channels: int = 128,
        num_views: int = 4,
        mask_size: int = 64,
        use_view_index: bool = False,
        max_view_index: int = 64,
    ):
        super().__init__()
        self.num_views = int(num_views)
        self.mask_size = int(mask_size)
        self.hidden_channels = int(hidden_channels)
        self.use_view_index = bool(use_view_index)

        y, x = torch.meshgrid(
            torch.linspace(-1.0, 1.0, self.mask_size),
            torch.linspace(-1.0, 1.0, self.mask_size),
            indexing="ij",
        )
        self.register_buffer("coords2d", torch.stack([x, y], dim=0), persistent=False)

        c3 = torch.stack(
            torch.meshgrid(
                torch.linspace(-1.0, 1.0, 16),
                torch.linspace(-1.0, 1.0, 16),
                torch.linspace(-1.0, 1.0, 16),
                indexing="ij",
            ),
            dim=0,
        )
        self.register_buffer("coords3d", c3, persistent=False)

        self.mask_encoder = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.GroupNorm(8, 32),
            nn.SiLU(),
            nn.Conv2d(32, 64, 4, stride=2, padding=1),
            nn.GroupNorm(8, 64),
            nn.SiLU(),
            nn.Conv2d(64, hidden_channels, 4, stride=2, padding=1),
            nn.GroupNorm(8, hidden_channels),
            nn.SiLU(),
            nn.Conv2d(hidden_channels, hidden_channels, 4, stride=2, padding=1),
            nn.GroupNorm(8, hidden_channels),
            nn.SiLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.slot_embed = nn.Embedding(self.num_views, hidden_channels)
        self.view_embed = nn.Embedding(max_view_index, hidden_channels) if self.use_view_index else None
        self.view_fuse = nn.Sequential(
            nn.Linear(hidden_channels * self.num_views, hidden_channels),
            nn.SiLU(),
            nn.Linear(hidden_channels, hidden_channels),
        )
        self.encoder3d = nn.Sequential(
            nn.Conv3d(latent_channels + 3, hidden_channels, 3, padding=1),
            nn.GroupNorm(8, hidden_channels),
            nn.SiLU(),
            nn.Conv3d(hidden_channels, hidden_channels, 3, padding=1),
            nn.GroupNorm(8, hidden_channels),
            nn.SiLU(),
            nn.Conv3d(hidden_channels, hidden_channels, 3, padding=1),
            nn.GroupNorm(8, hidden_channels),
            nn.SiLU(),
        )
        self.film = nn.Linear(hidden_channels, hidden_channels * 2)
        self.head = nn.Sequential(
            nn.Conv3d(hidden_channels, hidden_channels, 3, padding=1),
            nn.GroupNorm(8, hidden_channels),
            nn.SiLU(),
            nn.Conv3d(hidden_channels, 1, 1),
        )

    def forward(
        self,
        z_global: torch.Tensor,
        masks2d: torch.Tensor,
        view_indices: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if masks2d.dim() != 4:
            raise ValueError(f"masks2d expected [B,V,H,W], got {tuple(masks2d.shape)}")
        B, V, H, W = masks2d.shape
        if V != self.num_views:
            raise ValueError(f"model num_views={self.num_views}, got {V}")
        if H != self.mask_size or W != self.mask_size:
            raise ValueError(f"model mask_size={self.mask_size}, got {H}x{W}")

        coords2d = self.coords2d.to(device=masks2d.device, dtype=masks2d.dtype)
        coords2d = coords2d.view(1, 1, 2, H, W).expand(B, V, -1, -1, -1)
        x2 = torch.cat([masks2d.float().view(B, V, 1, H, W), coords2d.float()], dim=2)
        view_feat = self.mask_encoder(x2.reshape(B * V, 3, H, W)).view(B, V, self.hidden_channels)
        slot_ids = torch.arange(V, device=masks2d.device)
        view_feat = view_feat + self.slot_embed(slot_ids).view(1, V, self.hidden_channels)
        if self.view_embed is not None and view_indices is not None:
            view_feat = view_feat + self.view_embed(
                view_indices.clamp_min(0).clamp_max(self.view_embed.num_embeddings - 1)
            )
        cond = self.view_fuse(view_feat.reshape(B, V * self.hidden_channels))

        coords3d = self.coords3d.to(device=z_global.device, dtype=z_global.dtype)
        coords3d = coords3d.unsqueeze(0).expand(B, -1, -1, -1, -1)
        feat3d = self.encoder3d(torch.cat([z_global.float(), coords3d.float()], dim=1))
        gamma, beta = self.film(cond.float()).chunk(2, dim=1)
        feat3d = feat3d * (1.0 + gamma.view(B, -1, 1, 1, 1)) + beta.view(B, -1, 1, 1, 1)
        return self.head(feat3d).squeeze(1)
