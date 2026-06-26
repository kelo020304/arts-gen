"""Shared 3D U-Net blocks for mask16-conditioned part models."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


__all__ = [
    "FiLMResBlock3D",
    "Downsample3D",
    "Upsample3D",
    "TokenTransformer3D",
    "SinusoidalTimeEmbedding",
    "make_coords3d",
]


def _num_groups(channels: int) -> int:
    for groups in (32, 16, 8, 4, 2, 1):
        if channels % groups == 0:
            return groups
    return 1


def make_coords3d(resolution: int = 16) -> torch.Tensor:
    coords = torch.stack(
        torch.meshgrid(
            torch.linspace(-1.0, 1.0, int(resolution)),
            torch.linspace(-1.0, 1.0, int(resolution)),
            torch.linspace(-1.0, 1.0, int(resolution)),
            indexing="ij",
        ),
        dim=0,
    )
    return coords


class FiLMResBlock3D(nn.Module):
    """3D residual block with one FiLM modulation vector."""

    def __init__(self, in_channels: int, out_channels: int, cond_dim: int, *, zero_init_cond: bool = True):
        super().__init__()
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.norm1 = nn.GroupNorm(_num_groups(self.in_channels), self.in_channels)
        self.conv1 = nn.Conv3d(self.in_channels, self.out_channels, 3, padding=1)
        self.norm2 = nn.GroupNorm(_num_groups(self.out_channels), self.out_channels)
        self.cond = nn.Linear(int(cond_dim), self.out_channels * 2)
        self.conv2 = nn.Conv3d(self.out_channels, self.out_channels, 3, padding=1)
        self.skip = (
            nn.Identity()
            if self.in_channels == self.out_channels
            else nn.Conv3d(self.in_channels, self.out_channels, 1)
        )
        if zero_init_cond:
            nn.init.zeros_(self.cond.weight)
            nn.init.zeros_(self.cond.bias)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        gamma, beta = self.cond(cond).chunk(2, dim=1)
        h = self.norm2(h)
        h = h * (1.0 + gamma.view(gamma.shape[0], -1, 1, 1, 1)) + beta.view(beta.shape[0], -1, 1, 1, 1)
        h = self.conv2(F.silu(h))
        return self.skip(x) + h


class Downsample3D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Conv3d(int(in_channels), int(out_channels), 3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Upsample3D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Conv3d(int(in_channels), int(out_channels), 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        return self.conv(x)


class TokenTransformer3D(nn.Module):
    """Transformer over low-resolution 3D tokens, used only at the 4^3 bottleneck."""

    def __init__(
        self,
        channels: int,
        *,
        spatial_size: int = 4,
        num_layers: int = 2,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
    ):
        super().__init__()
        self.channels = int(channels)
        self.spatial_size = int(spatial_size)
        self.num_tokens = self.spatial_size ** 3
        self.pos = nn.Parameter(torch.zeros(1, self.num_tokens, self.channels))
        layer = nn.TransformerEncoderLayer(
            d_model=self.channels,
            nhead=int(num_heads),
            dim_feedforward=int(self.channels * float(mlp_ratio)),
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=int(num_layers))
        self.norm = nn.LayerNorm(self.channels)
        nn.init.normal_(self.pos, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, d, h, w = x.shape
        if c != self.channels:
            raise ValueError(f"expected {self.channels} channels, got {c}")
        if (d, h, w) != (self.spatial_size, self.spatial_size, self.spatial_size):
            raise ValueError(f"expected spatial {self.spatial_size}^3, got {(d, h, w)}")
        tokens = x.flatten(2).transpose(1, 2).contiguous()
        tokens = self.encoder(tokens + self.pos.to(device=x.device, dtype=x.dtype))
        tokens = self.norm(tokens)
        return tokens.transpose(1, 2).reshape(b, c, d, h, w).contiguous()


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int, *, max_period: float = 10000.0):
        super().__init__()
        self.dim = int(dim)
        self.max_period = float(max_period)
        self.mlp = nn.Sequential(
            nn.Linear(self.dim, self.dim * 4),
            nn.SiLU(),
            nn.Linear(self.dim * 4, self.dim),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        if t.dim() != 1:
            raise ValueError(f"t must be [B], got {tuple(t.shape)}")
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(self.max_period)
            * torch.arange(half, device=t.device, dtype=t.dtype)
            / max(half, 1)
        )
        args = t[:, None] * freqs[None] * self.max_period
        emb = torch.cat([torch.cos(args), torch.sin(args)], dim=1)
        if emb.shape[1] < self.dim:
            emb = F.pad(emb, (0, self.dim - emb.shape[1]))
        return self.mlp(emb)
