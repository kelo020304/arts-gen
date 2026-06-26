"""Prompted latent-mask decoder for TRELLIS SS latent grids.

This module predicts a soft part-membership mask on the 16^3 SS latent lattice.
It does not decode voxels and does not treat the lattice as a 64^3 occupancy
grid. The output logits are intended to condition the part-latent flow.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .mask16_unet_blocks import make_coords3d


__all__ = ["Stage1LatentMaskPromptDecoder"]


def _num_groups(channels: int) -> int:
    for groups in (32, 16, 8, 4, 2, 1):
        if channels % groups == 0:
            return groups
    return 1


class PreActResBlock3D(nn.Module):
    """Full-resolution 3D residual block over the 16^3 latent lattice."""

    def __init__(self, channels: int, *, dilation: int = 1, residual_scale: float = 1.0):
        super().__init__()
        self.channels = int(channels)
        self.residual_scale = float(residual_scale)
        self.norm1 = nn.GroupNorm(_num_groups(self.channels), self.channels)
        self.conv1 = nn.Conv3d(
            self.channels,
            self.channels,
            3,
            padding=int(dilation),
            dilation=int(dilation),
        )
        self.norm2 = nn.GroupNorm(_num_groups(self.channels), self.channels)
        self.conv2 = nn.Conv3d(self.channels, self.channels, 3, padding=1)
        nn.init.zeros_(self.conv2.weight)
        nn.init.zeros_(self.conv2.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        h = self.conv2(F.silu(self.norm2(h)))
        return x + h * self.residual_scale


class PromptCrossAttentionBlock(nn.Module):
    """Latent tokens attend to compact 2D-mask prompt tokens."""

    def __init__(self, channels: int, *, num_heads: int = 8, mlp_ratio: float = 4.0):
        super().__init__()
        self.channels = int(channels)
        self.norm_latent = nn.LayerNorm(self.channels)
        self.norm_prompt = nn.LayerNorm(self.channels)
        self.cross_attn = nn.MultiheadAttention(
            self.channels,
            int(num_heads),
            dropout=0.0,
            batch_first=True,
        )
        self.norm_mlp = nn.LayerNorm(self.channels)
        hidden = int(round(self.channels * float(mlp_ratio)))
        self.mlp = nn.Sequential(
            nn.Linear(self.channels, hidden),
            nn.GELU(),
            nn.Linear(hidden, self.channels),
        )

    def forward(self, latent_tokens: torch.Tensor, prompt_tokens: torch.Tensor) -> torch.Tensor:
        q = self.norm_latent(latent_tokens)
        kv = self.norm_prompt(prompt_tokens)
        h, _ = self.cross_attn(q, kv, kv, need_weights=False)
        latent_tokens = latent_tokens + h
        latent_tokens = latent_tokens + self.mlp(self.norm_mlp(latent_tokens))
        return latent_tokens


class TwoWayMaskDecoderBlock(nn.Module):
    """SAM-style two-way block between latent tokens and prompt/mask tokens."""

    def __init__(self, channels: int, *, num_heads: int = 8, mlp_ratio: float = 4.0):
        super().__init__()
        self.channels = int(channels)
        self.norm_cond_to_latent_q = nn.LayerNorm(self.channels)
        self.norm_cond_to_latent_kv = nn.LayerNorm(self.channels)
        self.cond_to_latent = nn.MultiheadAttention(
            self.channels,
            int(num_heads),
            dropout=0.0,
            batch_first=True,
        )
        self.norm_cond_mlp = nn.LayerNorm(self.channels)
        self.norm_latent_to_cond_q = nn.LayerNorm(self.channels)
        self.norm_latent_to_cond_kv = nn.LayerNorm(self.channels)
        self.latent_to_cond = nn.MultiheadAttention(
            self.channels,
            int(num_heads),
            dropout=0.0,
            batch_first=True,
        )
        self.norm_latent_mlp = nn.LayerNorm(self.channels)
        hidden = int(round(self.channels * float(mlp_ratio)))
        self.cond_mlp = nn.Sequential(
            nn.Linear(self.channels, hidden),
            nn.GELU(),
            nn.Linear(hidden, self.channels),
        )
        self.latent_mlp = nn.Sequential(
            nn.Linear(self.channels, hidden),
            nn.GELU(),
            nn.Linear(hidden, self.channels),
        )

    def forward(
        self,
        latent_tokens: torch.Tensor,
        cond_tokens: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        q = self.norm_cond_to_latent_q(cond_tokens)
        kv = self.norm_cond_to_latent_kv(latent_tokens)
        h, _ = self.cond_to_latent(q, kv, kv, need_weights=False)
        cond_tokens = cond_tokens + h
        cond_tokens = cond_tokens + self.cond_mlp(self.norm_cond_mlp(cond_tokens))

        q = self.norm_latent_to_cond_q(latent_tokens)
        kv = self.norm_latent_to_cond_kv(cond_tokens)
        h, _ = self.latent_to_cond(q, kv, kv, need_weights=False)
        latent_tokens = latent_tokens + h
        latent_tokens = latent_tokens + self.latent_mlp(self.norm_latent_mlp(latent_tokens))
        return latent_tokens, cond_tokens


class PromptTokenPooler(nn.Module):
    """Compress spatial 2D-mask tokens into a small prompt-token set."""

    def __init__(
        self,
        channels: int,
        *,
        num_prompt_tokens: int = 32,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
    ):
        super().__init__()
        self.channels = int(channels)
        self.num_prompt_tokens = int(num_prompt_tokens)
        self.queries = nn.Parameter(torch.randn(1, self.num_prompt_tokens, self.channels) * 0.02)
        self.norm_tokens = nn.LayerNorm(self.channels)
        self.norm_queries = nn.LayerNorm(self.channels)
        self.attn = nn.MultiheadAttention(self.channels, int(num_heads), dropout=0.0, batch_first=True)
        self.norm_mlp = nn.LayerNorm(self.channels)
        hidden = int(round(self.channels * float(mlp_ratio)))
        self.mlp = nn.Sequential(
            nn.Linear(self.channels, hidden),
            nn.GELU(),
            nn.Linear(hidden, self.channels),
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        b = tokens.shape[0]
        queries = self.queries.to(device=tokens.device, dtype=tokens.dtype).expand(b, -1, -1)
        q = self.norm_queries(queries)
        kv = self.norm_tokens(tokens)
        pooled, _ = self.attn(q, kv, kv, need_weights=False)
        pooled = queries + pooled
        pooled = pooled + self.mlp(self.norm_mlp(pooled))
        return pooled


class Stage1LatentMaskPromptDecoder(nn.Module):
    """Predict part membership logits on the 16^3 TRELLIS SS latent lattice.

    Inputs:
      z_global: full-object SS latent, [B, 8, 16, 16, 16]
      masks2d: ordered binary 2D part masks, [B, V, H, W]

    Output:
      latent_mask_logits: [B, 16, 16, 16]
    """

    def __init__(
        self,
        *,
        latent_channels: int = 8,
        model_channels: int = 256,
        num_views: int = 4,
        mask_size: int = 64,
        prompt_grid_size: int = 16,
        num_res_blocks: int = 12,
        decoder_layers: int = 4,
        num_heads: int = 8,
        num_prompt_tokens: int = 32,
        use_view_index: bool = False,
        max_view_index: int = 64,
    ):
        super().__init__()
        self.latent_channels = int(latent_channels)
        self.model_channels = int(model_channels)
        self.num_views = int(num_views)
        self.mask_size = int(mask_size)
        self.prompt_grid_size = int(prompt_grid_size)
        self.num_res_blocks = int(num_res_blocks)
        self.decoder_layers = int(decoder_layers)
        self.use_view_index = bool(use_view_index)

        y, x = torch.meshgrid(
            torch.linspace(-1.0, 1.0, self.mask_size),
            torch.linspace(-1.0, 1.0, self.mask_size),
            indexing="ij",
        )
        self.register_buffer("coords2d", torch.stack([x, y], dim=0), persistent=False)
        self.register_buffer("coords3d", make_coords3d(16), persistent=False)

        self.z_proj = nn.Conv3d(self.latent_channels + 3, self.model_channels, 3, padding=1)
        if num_res_blocks <= 0:
            dilations: tuple[int, ...] = ()
        else:
            base_pattern = (1, 1, 2, 1, 1, 2, 1, 1, 2, 1, 1, 1)
            repeats = (num_res_blocks + len(base_pattern) - 1) // len(base_pattern)
            dilations = (base_pattern * repeats)[:num_res_blocks]
        self.res_blocks = nn.ModuleList(
            [PreActResBlock3D(self.model_channels, dilation=int(dilation)) for dilation in dilations]
        )
        self.latent_pos = nn.Parameter(torch.randn(1, 16 * 16 * 16, self.model_channels) * 0.02)

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
            nn.Conv2d(192, self.model_channels, 3, padding=1),
            nn.GroupNorm(_num_groups(self.model_channels), self.model_channels),
            nn.SiLU(),
        )
        self.prompt_pos = nn.Parameter(
            torch.randn(1, self.num_views, self.model_channels, self.prompt_grid_size, self.prompt_grid_size) * 0.02
        )
        self.view_slot = nn.Embedding(self.num_views, self.model_channels)
        self.view_index_embed = nn.Embedding(max_view_index, self.model_channels) if self.use_view_index else None
        self.prompt_pool = PromptTokenPooler(
            self.model_channels,
            num_prompt_tokens=int(num_prompt_tokens),
            num_heads=int(num_heads),
        )
        self.mask_token = nn.Parameter(torch.randn(1, 1, self.model_channels) * 0.02)
        self.mask_seed = nn.Sequential(
            nn.LayerNorm(self.model_channels),
            nn.Linear(self.model_channels, self.model_channels),
            nn.GELU(),
            nn.Linear(self.model_channels, self.model_channels),
        )
        self.decoder = nn.ModuleList(
            [
                TwoWayMaskDecoderBlock(
                    self.model_channels,
                    num_heads=int(num_heads),
                )
                for _ in range(self.decoder_layers)
            ]
        )
        self.out_norm = nn.LayerNorm(self.model_channels)
        self.cond_norm = nn.LayerNorm(self.model_channels)
        self.mask_feat = nn.Linear(self.model_channels, self.model_channels)
        self.hyper_head = nn.Sequential(
            nn.LayerNorm(self.model_channels),
            nn.Linear(self.model_channels, self.model_channels * 2),
            nn.GELU(),
            nn.Linear(self.model_channels * 2, self.model_channels + 1),
        )
        with torch.no_grad():
            self.hyper_head[-1].bias[-1].fill_(-4.0)

    def encode_prompt_tokens(
        self,
        masks2d: torch.Tensor,
        view_indices: torch.Tensor | None = None,
    ) -> torch.Tensor:
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
        feat = self.mask_encoder(x2.reshape(b * v, 3, h, w))
        if feat.shape[-2:] != (self.prompt_grid_size, self.prompt_grid_size):
            feat = F.interpolate(
                feat,
                size=(self.prompt_grid_size, self.prompt_grid_size),
                mode="bilinear",
                align_corners=False,
            )
        feat = feat.view(b, v, self.model_channels, self.prompt_grid_size, self.prompt_grid_size)
        feat = feat + self.prompt_pos.to(device=feat.device, dtype=feat.dtype)
        slot_ids = torch.arange(v, device=masks2d.device)
        feat = feat + self.view_slot(slot_ids).to(dtype=feat.dtype).view(1, v, self.model_channels, 1, 1)
        if self.view_index_embed is not None and view_indices is not None:
            idx = view_indices.clamp_min(0).clamp_max(self.view_index_embed.num_embeddings - 1)
            feat = feat + self.view_index_embed(idx).to(dtype=feat.dtype).view(b, v, self.model_channels, 1, 1)
        tokens = feat.permute(0, 1, 3, 4, 2).reshape(
            b,
            v * self.prompt_grid_size * self.prompt_grid_size,
            self.model_channels,
        )
        return self.prompt_pool(tokens)

    def encode_latent_tokens(self, z_global: torch.Tensor) -> torch.Tensor:
        if z_global.dim() != 5:
            raise ValueError(f"z_global expected [B,C,16,16,16], got {tuple(z_global.shape)}")
        b, c, d, h, w = z_global.shape
        if c != self.latent_channels or (d, h, w) != (16, 16, 16):
            raise ValueError(f"z_global expected [B,{self.latent_channels},16,16,16], got {tuple(z_global.shape)}")
        coords = self.coords3d.to(device=z_global.device, dtype=z_global.dtype)
        coords = coords.unsqueeze(0).expand(b, -1, -1, -1, -1)
        x = self.z_proj(torch.cat([z_global.float(), coords.float()], dim=1))
        for block in self.res_blocks:
            x = block(x)
        tokens = x.flatten(2).transpose(1, 2).contiguous()
        return tokens + self.latent_pos.to(device=tokens.device, dtype=tokens.dtype)

    def forward(
        self,
        z_global: torch.Tensor,
        masks2d: torch.Tensor,
        view_indices: torch.Tensor | None = None,
    ) -> torch.Tensor:
        latent_tokens = self.encode_latent_tokens(z_global)
        prompt_tokens = self.encode_prompt_tokens(masks2d, view_indices=view_indices)
        mask_seed = self.mask_seed(prompt_tokens.mean(dim=1)).unsqueeze(1)
        mask_token = self.mask_token.to(device=prompt_tokens.device, dtype=prompt_tokens.dtype) + mask_seed
        cond_tokens = torch.cat([mask_token, prompt_tokens], dim=1)
        for block in self.decoder:
            latent_tokens, cond_tokens = block(latent_tokens, cond_tokens)
        latent_tokens = self.out_norm(latent_tokens)
        mask_token = self.cond_norm(cond_tokens[:, 0])
        dynamic = self.hyper_head(mask_token)
        weight = dynamic[:, :-1]
        bias = dynamic[:, -1]
        mask_feat = self.mask_feat(latent_tokens)
        logits = (mask_feat * weight.unsqueeze(1)).sum(dim=-1) / math.sqrt(float(self.model_channels))
        logits = logits + bias.unsqueeze(1)
        return logits.view(z_global.shape[0], 16, 16, 16)
