"""PartMMDiT model components.

Task 3 introduces the conditioning path and zero-init gated cross-part block.
The full PartMMDiTModel assembly is added in the following task.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.utils.checkpoint

from trellis.models.sparse_structure_flow import TimestepEmbedder
from trellis.modules.attention.full_attn import scaled_dot_product_attention
from trellis.modules.attention import MultiHeadAttention
from trellis.modules.norm import LayerNorm32
from trellis.modules.spatial import patchify, unpatchify
from trellis.modules.transformer import AbsolutePositionEmbedder
from trellis.modules.transformer.blocks import FeedForwardNet


__all__ = ["PartConditioner", "GatedCrossPartBlock", "PartMMDiTModel"]


class PartConditioner(nn.Module):
    """Build per-part adaLN condition vectors.

    ``cond_vec = t_emb + proj_name(name_or_null) + proj_anchor(anchor_or_null)``.
    The nulls are learned in model-channel space and are selected independently
    for name and anchor CFG dropout.
    """

    def __init__(self, dim: int, name_dim: int = 768, anchor_in: int = 4):
        super().__init__()
        self.dim = int(dim)
        self.t_embed = TimestepEmbedder(self.dim)
        self.name_proj = nn.Linear(int(name_dim), self.dim)
        self.anchor_mlp = nn.Sequential(
            nn.Linear(int(anchor_in), self.dim),
            nn.SiLU(),
            nn.Linear(self.dim, self.dim),
        )
        self.name_null = nn.Parameter(torch.zeros(self.dim))
        self.anchor_null = nn.Parameter(torch.zeros(self.dim))

    def encode_anchor(
        self,
        anchor: torch.Tensor,
        anchor_valid: torch.Tensor,
    ) -> torch.Tensor:
        """Encode bbox anchors with visible-view masked mean.

        Args:
            anchor: ``[..., V, 4]`` normalized ``cx,cy,w,h``.
            anchor_valid: ``[..., V]`` bool visible mask.
        """

        feat = self.anchor_mlp(anchor)
        mask = anchor_valid.to(dtype=feat.dtype).unsqueeze(-1)
        denom = mask.sum(dim=-2).clamp_min(1.0)
        return (feat * mask).sum(dim=-2) / denom

    def forward(
        self,
        t: torch.Tensor,
        name_emb: torch.Tensor,
        anchor: torch.Tensor,
        anchor_valid: torch.Tensor,
        drop_name: torch.Tensor | None = None,
        drop_anchor: torch.Tensor | None = None,
    ) -> torch.Tensor:
        time_cond = self.t_embed(t)
        name_cond = self.name_proj(name_emb)
        if drop_name is not None:
            name_cond = torch.where(
                drop_name.unsqueeze(-1),
                self.name_null.to(dtype=name_cond.dtype, device=name_cond.device),
                name_cond,
            )

        anchor_cond = self.encode_anchor(anchor, anchor_valid)
        if drop_anchor is not None:
            anchor_cond = torch.where(
                drop_anchor.unsqueeze(-1),
                self.anchor_null.to(dtype=anchor_cond.dtype, device=anchor_cond.device),
                anchor_cond,
            )
        return time_cond + name_cond + anchor_cond


class _MaskedSelfAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int):
        super().__init__()
        if int(dim) % int(num_heads) != 0:
            raise ValueError(f"dim={dim} must be divisible by num_heads={num_heads}")
        self.dim = int(dim)
        self.num_heads = int(num_heads)
        self.head_dim = self.dim // self.num_heads
        self.to_qkv = nn.Linear(self.dim, self.dim * 3)
        self.to_out = nn.Linear(self.dim, self.dim)

    def forward(self, x: torch.Tensor, key_padding_mask: torch.Tensor) -> torch.Tensor:
        batch_size, token_count, _ = x.shape
        qkv = self.to_qkv(x).view(
            batch_size,
            token_count,
            3,
            self.num_heads,
            self.head_dim,
        )
        if not bool(key_padding_mask.any()):
            out = scaled_dot_product_attention(qkv).reshape(batch_size, token_count, self.dim)
            return self.to_out(out)

        q, k, v = qkv.unbind(dim=2)
        q = q.permute(0, 2, 1, 3)
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)
        attn_logits = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        valid_mask = (~key_padding_mask).view(batch_size, 1, 1, token_count)
        attn_logits = attn_logits.masked_fill(~valid_mask, torch.finfo(attn_logits.dtype).min)
        attn = torch.softmax(attn_logits, dim=-1)
        attn = torch.where(valid_mask, attn, torch.zeros_like(attn))
        out = torch.matmul(attn, v)
        out = out.permute(0, 2, 1, 3).reshape(batch_size, token_count, self.dim)
        return self.to_out(out)


class GatedCrossPartBlock(nn.Module):
    """Zero-init gated self-attention over all valid part tokens of one object."""

    def __init__(self, dim: int, num_heads: int):
        super().__init__()
        self.norm = LayerNorm32(int(dim), elementwise_affine=False, eps=1e-6)
        self.attn = _MaskedSelfAttention(int(dim), int(num_heads))
        self.gate = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor, part_valid: torch.Tensor) -> torch.Tensor:
        """Apply cross-part attention to valid parts only.

        Args:
            x: ``[K, T, C]`` part tokens for one object.
            part_valid: ``[K]`` bool mask, True for real parts.
        """

        if x.dim() != 3:
            raise ValueError(f"x expected [K,T,C], got {tuple(x.shape)}")
        if part_valid.dim() != 1 or part_valid.shape[0] != x.shape[0]:
            raise ValueError(
                f"part_valid expected [{x.shape[0]}], got {tuple(part_valid.shape)}"
            )
        if not bool(part_valid.any()):
            return x

        part_count, token_count, channels = x.shape
        valid_parts = part_valid.to(device=x.device).bool()
        valid_x = x[valid_parts]
        flat = self.norm(valid_x).reshape(1, valid_x.shape[0] * token_count, channels)
        attn_out = self.attn(
            flat,
            key_padding_mask=torch.zeros(
                1,
                flat.shape[1],
                dtype=torch.bool,
                device=x.device,
            ),
        ).reshape(valid_x.shape[0], token_count, channels)
        out = torch.zeros(
            part_count,
            token_count,
            channels,
            dtype=attn_out.dtype,
            device=x.device,
        )
        out[valid_parts] = attn_out
        return x + (self.gate.to(dtype=out.dtype) * out).to(dtype=x.dtype)


class PartMMDiTBlock(nn.Module):
    """Dense adaLN block with self-attn, shared-memory cross-attn, and MLP."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        use_checkpoint: bool = False,
    ):
        super().__init__()
        self.use_checkpoint = bool(use_checkpoint)
        self.norm1 = LayerNorm32(dim, elementwise_affine=False, eps=1e-6)
        self.norm_cross = LayerNorm32(dim, elementwise_affine=True, eps=1e-6)
        self.norm2 = LayerNorm32(dim, elementwise_affine=False, eps=1e-6)
        self.self_attn = MultiHeadAttention(
            dim,
            num_heads=num_heads,
            type="self",
            attn_mode="full",
            qk_rms_norm=True,
        )
        self.cross_attn = MultiHeadAttention(
            dim,
            ctx_channels=dim,
            num_heads=num_heads,
            type="cross",
            attn_mode="full",
            qk_rms_norm=False,
        )
        self.mlp = FeedForwardNet(dim, mlp_ratio=mlp_ratio)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim, 6 * dim, bias=True),
        )

    def _forward(
        self,
        x: torch.Tensor,
        cond_vec: torch.Tensor,
        memory: torch.Tensor,
    ) -> torch.Tensor:
        shift_attn, scale_attn, gate_attn, shift_mlp, scale_mlp, gate_mlp = (
            self.adaLN_modulation(cond_vec).chunk(6, dim=1)
        )
        h = self.norm1(x)
        h = h * (1 + scale_attn.unsqueeze(1)) + shift_attn.unsqueeze(1)
        h = self.self_attn(h)
        h = h + self.cross_attn(self.norm_cross(x), memory)
        x = x + gate_attn.unsqueeze(1) * h

        h = self.norm2(x)
        h = h * (1 + scale_mlp.unsqueeze(1)) + shift_mlp.unsqueeze(1)
        h = self.mlp(h)
        x = x + gate_mlp.unsqueeze(1) * h
        return x

    def forward(
        self,
        x: torch.Tensor,
        cond_vec: torch.Tensor,
        memory: torch.Tensor,
    ) -> torch.Tensor:
        if self.use_checkpoint:
            return torch.utils.checkpoint.checkpoint(
                self._forward,
                x,
                cond_vec,
                memory,
                use_reentrant=False,
            )
        return self._forward(x, cond_vec, memory)


class PartMMDiTModel(nn.Module):
    """Part latent flow model with name->adaLN identity and 2D anchors."""

    def __init__(
        self,
        resolution: int,
        latent_channels: int,
        model_channels: int,
        cond_dim: int,
        num_blocks: int,
        num_heads: int,
        patch_size: int,
        num_views: int,
        max_parts: int,
        cross_part_layers: tuple[int, ...] | list[int] = (3, 6, 9),
        clip_name_dim: int = 768,
        mlp_ratio: float = 4.0,
        use_fp16: bool = False,
        use_checkpoint: bool = False,
    ):
        super().__init__()
        self.resolution = int(resolution)
        self.latent_channels = int(latent_channels)
        self.model_channels = int(model_channels)
        self.cond_dim = int(cond_dim)
        self.num_blocks = int(num_blocks)
        self.num_heads = int(num_heads)
        self.patch_size = int(patch_size)
        self.num_views = int(num_views)
        self.max_parts = int(max_parts)
        self.cross_part_layers = tuple(int(layer) for layer in cross_part_layers)
        self.clip_name_dim = int(clip_name_dim)
        self.use_fp16 = bool(use_fp16)
        self.use_checkpoint = bool(use_checkpoint)
        self.dtype = torch.float16 if self.use_fp16 else torch.float32

        if self.resolution % self.patch_size != 0:
            raise ValueError(
                f"resolution={self.resolution} must be divisible by patch_size={self.patch_size}"
            )
        if self.model_channels % self.num_heads != 0:
            raise ValueError(
                f"model_channels={self.model_channels} must be divisible by num_heads={self.num_heads}"
            )

        patch_dim = self.latent_channels * self.patch_size ** 3
        self.input_layer = nn.Linear(patch_dim, self.model_channels)
        self.global_layer = nn.Linear(patch_dim, self.model_channels)
        self.cond_proj = nn.Linear(self.cond_dim, self.model_channels)
        self.out_layer = nn.Linear(self.model_channels, patch_dim)
        self.conditioner = PartConditioner(
            dim=self.model_channels,
            name_dim=self.clip_name_dim,
            anchor_in=4,
        )
        self.blocks = nn.ModuleList(
            [
                PartMMDiTBlock(
                    dim=self.model_channels,
                    num_heads=self.num_heads,
                    mlp_ratio=mlp_ratio,
                    use_checkpoint=self.use_checkpoint,
                )
                for _ in range(self.num_blocks)
            ]
        )
        self.cross_part_blocks = nn.ModuleDict(
            {
                str(layer): GatedCrossPartBlock(self.model_channels, self.num_heads)
                for layer in self.cross_part_layers
            }
        )

        grid = self.resolution // self.patch_size
        coords = torch.meshgrid(
            torch.arange(grid, dtype=torch.float32),
            torch.arange(grid, dtype=torch.float32),
            torch.arange(grid, dtype=torch.float32),
            indexing="ij",
        )
        coords = torch.stack(coords, dim=-1).reshape(-1, 3)
        pos_embedder = AbsolutePositionEmbedder(self.model_channels, 3)
        self.register_buffer("pos_emb", pos_embedder(coords), persistent=False)

        self.initialize_weights()
        if self.use_fp16:
            self.convert_to_fp16()

    def initialize_weights(self) -> None:
        def _basic_init(module: nn.Module) -> None:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)
        nn.init.normal_(self.conditioner.t_embed.mlp[0].weight, std=0.02)
        nn.init.normal_(self.conditioner.t_embed.mlp[2].weight, std=0.02)
        for block in self.blocks:
            mod = block.adaLN_modulation[-1]
            gate_attn_start = 2 * self.model_channels
            gate_attn_end = 3 * self.model_channels
            gate_mlp_start = 5 * self.model_channels
            gate_mlp_end = 6 * self.model_channels
            nn.init.constant_(mod.bias, 0)
            nn.init.constant_(mod.weight[gate_attn_start:gate_attn_end], 0)
            nn.init.constant_(mod.weight[gate_mlp_start:gate_mlp_end], 0)
            nn.init.constant_(mod.bias[gate_attn_start:gate_attn_end], 1e-3)
            nn.init.constant_(mod.bias[gate_mlp_start:gate_mlp_end], 1e-3)
        nn.init.xavier_uniform_(self.out_layer.weight)
        nn.init.constant_(self.out_layer.bias, 0)

    def convert_to_fp16(self) -> None:
        self.blocks.apply(lambda module: module.half() if isinstance(module, nn.Linear) else module)
        self.cross_part_blocks.apply(lambda module: module.half() if isinstance(module, nn.Linear) else module)

    def patchify_latent(self, x: torch.Tensor) -> torch.Tensor:
        patches = patchify(x, self.patch_size)
        patches = patches.view(patches.shape[0], patches.shape[1], -1).permute(0, 2, 1)
        return self.input_layer(patches) + self.pos_emb.to(device=x.device, dtype=patches.dtype)[None]

    def unpatchify_latent(self, tokens: torch.Tensor) -> torch.Tensor:
        grid = self.resolution // self.patch_size
        patches = self.out_layer(tokens)
        patches = patches.permute(0, 2, 1).contiguous().view(
            tokens.shape[0],
            self.latent_channels * self.patch_size ** 3,
            grid,
            grid,
            grid,
        )
        return unpatchify(patches, self.patch_size).contiguous()

    def _global_tokens(self, z_global: torch.Tensor) -> torch.Tensor:
        patches = patchify(z_global, self.patch_size)
        patches = patches.view(patches.shape[0], patches.shape[1], -1).permute(0, 2, 1)
        return self.global_layer(patches) + self.pos_emb.to(device=z_global.device, dtype=patches.dtype)[None]

    def _memory_tokens(self, z_global: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        global_tokens = self._global_tokens(z_global)
        image_tokens = self.cond_proj(cond)
        return torch.cat([global_tokens, image_tokens], dim=1)

    def forward(
        self,
        x_t_parts: torch.Tensor,
        t: torch.Tensor,
        z_global: torch.Tensor,
        cond: torch.Tensor,
        name_emb: torch.Tensor,
        anchor: torch.Tensor,
        anchor_valid: torch.Tensor,
        part_valid: torch.Tensor,
        drop_name: torch.Tensor | None = None,
        drop_anchor: torch.Tensor | None = None,
        x_self_cond: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if x_self_cond is not None:
            raise NotImplementedError("PartMMDiTModel does not use self-conditioning")
        if x_t_parts.dim() != 6:
            raise ValueError(f"x_t_parts expected [B,K,C,R,R,R], got {tuple(x_t_parts.shape)}")
        batch_size, part_count = x_t_parts.shape[:2]
        expected_latent = (
            batch_size,
            part_count,
            self.latent_channels,
            self.resolution,
            self.resolution,
            self.resolution,
        )
        if tuple(x_t_parts.shape) != expected_latent:
            raise ValueError(f"x_t_parts expected {expected_latent}, got {tuple(x_t_parts.shape)}")
        if tuple(part_valid.shape) != (batch_size, part_count):
            raise ValueError(
                f"part_valid expected {(batch_size, part_count)}, got {tuple(part_valid.shape)}"
            )

        flat_x = x_t_parts.reshape(
            batch_size * part_count,
            self.latent_channels,
            self.resolution,
            self.resolution,
            self.resolution,
        )
        h = self.patchify_latent(flat_x)
        token_count = h.shape[1]

        memory = self._memory_tokens(z_global, cond)
        memory = memory[:, None].expand(batch_size, part_count, -1, -1).reshape(
            batch_size * part_count,
            memory.shape[1],
            self.model_channels,
        )

        t_part = t[:, None].expand(batch_size, part_count).reshape(-1)
        cond_vec = self.conditioner(
            t_part,
            name_emb.reshape(batch_size * part_count, self.clip_name_dim),
            anchor.reshape(batch_size * part_count, anchor.shape[-2], anchor.shape[-1]),
            anchor_valid.reshape(batch_size * part_count, anchor_valid.shape[-1]),
            drop_name=drop_name.reshape(-1) if drop_name is not None else None,
            drop_anchor=drop_anchor.reshape(-1) if drop_anchor is not None else None,
        )

        h = h.to(dtype=self.dtype)
        memory = memory.to(dtype=self.dtype)
        cond_vec = cond_vec.to(dtype=self.dtype)
        for layer_idx, block in enumerate(self.blocks):
            h = block(h, cond_vec, memory)
            if layer_idx in self.cross_part_layers:
                h_obj = h.reshape(batch_size, part_count, token_count, self.model_channels)
                out = []
                cross_block = self.cross_part_blocks[str(layer_idx)]
                for batch_idx in range(batch_size):
                    out.append(cross_block(h_obj[batch_idx], part_valid[batch_idx]))
                h = torch.stack(out, dim=0).reshape(
                    batch_size * part_count,
                    token_count,
                    self.model_channels,
                )

        h = h.to(dtype=x_t_parts.dtype)
        out = self.unpatchify_latent(h)
        return out.reshape(
            batch_size,
            part_count,
            self.latent_channels,
            self.resolution,
            self.resolution,
            self.resolution,
        )
