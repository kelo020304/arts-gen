"""PartMMDiT v2 true dual-stream MMDiT components."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.utils.checkpoint

from trellis.models.sparse_structure_flow import TimestepEmbedder
from trellis.modules.attention import MultiHeadAttention
from trellis.modules.attention.full_attn import scaled_dot_product_attention
from trellis.modules.norm import LayerNorm32
from trellis.modules.spatial import patchify, unpatchify
from trellis.modules.transformer import AbsolutePositionEmbedder
from trellis.modules.transformer.blocks import FeedForwardNet


__all__ = [
    "PartConditionTokenBuilder",
    "DualStreamMMDiTBlock",
    "GatedCrossPartBlock",
    "PartMMDiTModel",
]


class AdaLNZero(nn.Module):
    """Timestep-only adaLN modulation."""

    def __init__(self, dim: int):
        super().__init__()
        self.net = nn.Sequential(nn.SiLU(), nn.Linear(int(dim), 6 * int(dim), bias=True))
        nn.init.constant_(self.net[-1].weight, 0)
        nn.init.constant_(self.net[-1].bias, 0)

    def forward(self, t_emb: torch.Tensor) -> tuple[torch.Tensor, ...]:
        return self.net(t_emb).chunk(6, dim=1)


class PartConditionTokenBuilder(nn.Module):
    """Build per-part condition stream tokens from name tokens and view anchors."""

    def __init__(self, dim: int, name_dim: int = 768, anchor_in: int = 4):
        super().__init__()
        self.dim = int(dim)
        self.name_dim = int(name_dim)
        self.name_proj = nn.Linear(self.name_dim, self.dim)
        self.anchor_mlp = nn.Sequential(
            nn.Linear(int(anchor_in), self.dim),
            nn.SiLU(),
            nn.Linear(self.dim, self.dim),
        )
        self.invisible_anchor = nn.Parameter(torch.zeros(self.dim))
        self.null_name = nn.Parameter(torch.zeros(1, self.name_dim))
        self.null_anchor = nn.Parameter(torch.zeros(1, self.dim))

    def forward(
        self,
        name_tokens: torch.Tensor,
        name_mask: torch.Tensor,
        anchor: torch.Tensor,
        anchor_valid: torch.Tensor,
        *,
        drop_name: torch.Tensor | None = None,
        drop_anchor: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return condition tokens and bool mask.

        Args:
            name_tokens: ``[B,K,L,768]`` or flattened ``[N,L,768]``.
            name_mask: matching ``[B,K,L]`` or ``[N,L]`` bool mask.
            anchor: matching ``[B,K,V,4]`` or ``[N,V,4]``.
            anchor_valid: matching ``[B,K,V]`` or ``[N,V]`` bool mask.
        """

        original_shape = name_tokens.shape
        if name_tokens.dim() == 4:
            batch_size, part_count, name_len, name_dim = name_tokens.shape
            flat_count = batch_size * part_count
            name_tokens = name_tokens.reshape(flat_count, name_len, name_dim)
            name_mask = name_mask.reshape(flat_count, name_len)
            anchor = anchor.reshape(flat_count, anchor.shape[-2], anchor.shape[-1])
            anchor_valid = anchor_valid.reshape(flat_count, anchor_valid.shape[-1])
            if drop_name is not None:
                drop_name = drop_name.reshape(flat_count)
            if drop_anchor is not None:
                drop_anchor = drop_anchor.reshape(flat_count)
        elif name_tokens.dim() != 3:
            raise ValueError(
                f"name_tokens expected [B,K,L,D] or [N,L,D], got {tuple(original_shape)}"
            )

        if name_tokens.shape[-1] != self.name_dim:
            raise ValueError(
                f"name_tokens dim expected {self.name_dim}, got {name_tokens.shape[-1]}"
            )
        if name_mask.shape != name_tokens.shape[:2]:
            raise ValueError(
                f"name_mask shape {tuple(name_mask.shape)} must match "
                f"name_tokens[:2] {tuple(name_tokens.shape[:2])}"
            )
        if anchor.dim() != 3 or anchor.shape[-1] != 4:
            raise ValueError(f"anchor expected [N,V,4], got {tuple(anchor.shape)}")
        if anchor_valid.shape != anchor.shape[:2]:
            raise ValueError(
                f"anchor_valid shape {tuple(anchor_valid.shape)} must match "
                f"anchor[:2] {tuple(anchor.shape[:2])}"
            )
        if name_tokens.shape[0] != anchor.shape[0]:
            raise ValueError("name and anchor streams must have the same flat part count")

        name_mask = name_mask.bool()
        anchor_valid = anchor_valid.bool()
        name_tokens = name_tokens.to(dtype=self.name_proj.weight.dtype)
        anchor = anchor.to(dtype=self.anchor_mlp[0].weight.dtype)
        name_stream = self.name_proj(name_tokens)
        if drop_name is not None:
            drop_name = drop_name.to(device=name_stream.device).bool()
            null_name = self.name_proj(
                self.null_name.to(device=name_tokens.device, dtype=name_tokens.dtype)
            ).to(dtype=name_stream.dtype)
            name_stream = torch.where(drop_name.view(-1, 1, 1), null_name.view(1, 1, -1), name_stream)
            name_mask = torch.where(
                drop_name.view(-1, 1),
                torch.ones_like(name_mask),
                name_mask,
            )

        anchor_stream = self.anchor_mlp(anchor)
        invisible = self.invisible_anchor.to(
            device=anchor_stream.device,
            dtype=anchor_stream.dtype,
        ).view(1, 1, -1)
        anchor_stream = torch.where(anchor_valid.unsqueeze(-1), anchor_stream, invisible)
        anchor_mask = torch.ones(
            anchor_stream.shape[:2],
            dtype=torch.bool,
            device=anchor_stream.device,
        )
        if drop_anchor is not None:
            drop_anchor = drop_anchor.to(device=anchor_stream.device).bool()
            null_anchor = self.null_anchor.to(
                device=anchor_stream.device,
                dtype=anchor_stream.dtype,
            ).view(1, 1, -1)
            anchor_stream = torch.where(drop_anchor.view(-1, 1, 1), null_anchor, anchor_stream)

        cond_tokens = torch.cat([name_stream, anchor_stream], dim=1)
        cond_mask = torch.cat([name_mask.to(device=cond_tokens.device), anchor_mask], dim=1)
        return cond_tokens, cond_mask


class JointDualStreamAttention(nn.Module):
    """Two projection streams, one joint bidirectional self-attention."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        qkv_bias: bool = True,
        update_cond: bool = True,
    ):
        super().__init__()
        dim = int(dim)
        num_heads = int(num_heads)
        if dim % num_heads != 0:
            raise ValueError(f"dim={dim} must be divisible by num_heads={num_heads}")
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.update_cond = bool(update_cond)
        self.part_qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.cond_qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.part_out = nn.Linear(dim, dim)
        self.cond_out = nn.Linear(dim, dim) if self.update_cond else None

    def forward(
        self,
        part: torch.Tensor,
        cond: torch.Tensor,
        cond_mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, part_len, _ = part.shape
        cond_len = int(cond.shape[1])
        part_qkv = self.part_qkv(part).reshape(
            batch_size,
            part_len,
            3,
            self.num_heads,
            self.head_dim,
        )
        cond_qkv = self.cond_qkv(cond).reshape(
            batch_size,
            cond_len,
            3,
            self.num_heads,
            self.head_dim,
        )
        q = torch.cat([cond_qkv[:, :, 0], part_qkv[:, :, 0]], dim=1)
        k = torch.cat([cond_qkv[:, :, 1], part_qkv[:, :, 1]], dim=1)
        v = torch.cat([cond_qkv[:, :, 2], part_qkv[:, :, 2]], dim=1)
        if cond_mask is None or bool(cond_mask.all()):
            joint = scaled_dot_product_attention(q, k, v)
        else:
            valid = torch.cat(
                [
                    cond_mask.to(device=part.device).bool(),
                    torch.ones(batch_size, part_len, dtype=torch.bool, device=part.device),
                ],
                dim=1,
            )
            joint = self._masked_attention(q, k, v, valid)
        cond_out, part_out = joint[:, :cond_len], joint[:, cond_len:]
        part_out = self.part_out(part_out.reshape(batch_size, part_len, self.dim))
        if self.update_cond:
            cond_out = self.cond_out(cond_out.reshape(batch_size, cond_len, self.dim))
        else:
            cond_out = None
        return part_out, cond_out

    def _masked_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        valid: torch.Tensor,
    ) -> torch.Tensor:
        q = q.permute(0, 2, 1, 3)
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)
        logits = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        key_valid = valid.view(valid.shape[0], 1, 1, valid.shape[1])
        logits = logits.masked_fill(~key_valid, torch.finfo(logits.dtype).min)
        attn = torch.softmax(logits, dim=-1)
        query_valid = valid.view(valid.shape[0], 1, valid.shape[1], 1)
        attn = torch.where(query_valid, attn, torch.zeros_like(attn))
        out = torch.matmul(attn, v)
        return out.permute(0, 2, 1, 3)


class DualStreamMMDiTBlock(nn.Module):
    """True dual-stream MMDiT block for one part and its name/anchor tokens."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        use_checkpoint: bool = False,
        update_cond: bool = True,
    ):
        super().__init__()
        self.use_checkpoint = bool(use_checkpoint)
        self.update_cond = bool(update_cond)
        self.part_norm1 = LayerNorm32(dim, elementwise_affine=False, eps=1e-6)
        self.cond_norm1 = LayerNorm32(dim, elementwise_affine=False, eps=1e-6)
        self.part_norm2 = LayerNorm32(dim, elementwise_affine=False, eps=1e-6)
        if self.update_cond:
            self.cond_norm2 = LayerNorm32(dim, elementwise_affine=False, eps=1e-6)
        self.joint_attn = JointDualStreamAttention(dim, num_heads, update_cond=self.update_cond)
        self.part_mlp = FeedForwardNet(dim, mlp_ratio=mlp_ratio)
        if self.update_cond:
            self.cond_mlp = FeedForwardNet(dim, mlp_ratio=mlp_ratio)
        self.part_mod = AdaLNZero(dim)
        self.cond_mod = AdaLNZero(dim)

    @staticmethod
    def _modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

    def _forward(
        self,
        part: torch.Tensor,
        cond: torch.Tensor,
        t_emb: torch.Tensor,
        cond_mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        p_shift_a, p_scale_a, p_gate_a, p_shift_m, p_scale_m, p_gate_m = self.part_mod(t_emb)
        c_shift_a, c_scale_a, c_gate_a, c_shift_m, c_scale_m, c_gate_m = self.cond_mod(t_emb)

        part_h = self._modulate(self.part_norm1(part), p_shift_a, p_scale_a)
        cond_h = self._modulate(self.cond_norm1(cond), c_shift_a, c_scale_a)
        part_attn, cond_attn = self.joint_attn(part_h, cond_h, cond_mask)
        part = part + p_gate_a.unsqueeze(1) * part_attn
        if self.update_cond:
            cond = cond + c_gate_a.unsqueeze(1) * cond_attn

        part_h = self._modulate(self.part_norm2(part), p_shift_m, p_scale_m)
        part = part + p_gate_m.unsqueeze(1) * self.part_mlp(part_h)
        if self.update_cond:
            cond_h = self._modulate(self.cond_norm2(cond), c_shift_m, c_scale_m)
            cond = cond + c_gate_m.unsqueeze(1) * self.cond_mlp(cond_h)
        if cond_mask is not None:
            cond = cond * cond_mask.to(device=cond.device, dtype=cond.dtype).unsqueeze(-1)
        return part, cond

    def forward(
        self,
        part: torch.Tensor,
        cond: torch.Tensor,
        t_emb: torch.Tensor,
        cond_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.use_checkpoint:
            return torch.utils.checkpoint.checkpoint(
                self._forward,
                part,
                cond,
                t_emb,
                cond_mask,
                use_reentrant=False,
            )
        return self._forward(part, cond, t_emb, cond_mask)


class SharedMemoryCrossAttentionBlock(nn.Module):
    """Part-stream-only cross-attention into shared global/image memory."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        use_checkpoint: bool = False,
    ):
        super().__init__()
        self.use_checkpoint = bool(use_checkpoint)
        self.norm_cross = LayerNorm32(dim, elementwise_affine=True, eps=1e-6)
        self.cross_attn = MultiHeadAttention(
            dim,
            ctx_channels=dim,
            num_heads=num_heads,
            type="cross",
            attn_mode="full",
            qk_rms_norm=False,
        )

    def _forward(self, part: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        return part + self.cross_attn(self.norm_cross(part), memory)

    def forward(self, part: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        if self.use_checkpoint:
            return torch.utils.checkpoint.checkpoint(
                self._forward,
                part,
                memory,
                use_reentrant=False,
            )
        return self._forward(part, memory)


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
        """Apply cross-part attention to valid parts only."""

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


class PartMMDiTModel(nn.Module):
    """Part latent flow model with true dual-stream name/anchor MMDiT."""

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
        if max(self.cross_part_layers, default=-1) >= self.num_blocks:
            raise ValueError(
                f"cross_part_layers={self.cross_part_layers} exceed num_blocks={self.num_blocks}"
            )

        patch_dim = self.latent_channels * self.patch_size ** 3
        self.input_layer = nn.Linear(patch_dim, self.model_channels)
        self.global_layer = nn.Linear(patch_dim, self.model_channels)
        self.cond_proj = nn.Linear(self.cond_dim, self.model_channels)
        self.out_layer = nn.Linear(self.model_channels, patch_dim)
        self.t_embed = TimestepEmbedder(self.model_channels)
        self.cond_token_builder = PartConditionTokenBuilder(
            dim=self.model_channels,
            name_dim=self.clip_name_dim,
            anchor_in=4,
        )
        self.blocks = nn.ModuleList(
            [
                DualStreamMMDiTBlock(
                    dim=self.model_channels,
                    num_heads=self.num_heads,
                    mlp_ratio=mlp_ratio,
                    use_checkpoint=self.use_checkpoint,
                    update_cond=layer_idx < self.num_blocks - 1,
                )
                for layer_idx in range(self.num_blocks)
            ]
        )
        self.memory_blocks = nn.ModuleList(
            [
                SharedMemoryCrossAttentionBlock(
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
        nn.init.normal_(self.t_embed.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embed.mlp[2].weight, std=0.02)
        for block in self.blocks:
            for mod in (block.part_mod.net[-1], block.cond_mod.net[-1]):
                nn.init.constant_(mod.weight, 0)
                nn.init.constant_(mod.bias, 0)
                dim = self.model_channels
                nn.init.constant_(mod.bias[2 * dim : 3 * dim], 1e-3)
                nn.init.constant_(mod.bias[5 * dim : 6 * dim], 1e-3)
        nn.init.xavier_uniform_(self.out_layer.weight)
        nn.init.constant_(self.out_layer.bias, 0)

    def convert_to_fp16(self) -> None:
        for module in (self.blocks, self.memory_blocks, self.cross_part_blocks):
            module.apply(lambda submodule: submodule.half() if isinstance(submodule, nn.Linear) else submodule)

    def _pos_emb(self, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        return self.pos_emb.to(device=device, dtype=dtype)[None]

    def patchify_latent(self, x: torch.Tensor) -> torch.Tensor:
        patches = patchify(x, self.patch_size)
        patches = patches.view(patches.shape[0], patches.shape[1], -1).permute(0, 2, 1)
        return self.input_layer(patches) + self._pos_emb(x.device, patches.dtype)

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
        return self.global_layer(patches) + self._pos_emb(z_global.device, patches.dtype)

    def _memory_tokens(self, z_global: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        if cond.dim() != 3 or cond.shape[-1] != self.cond_dim:
            raise ValueError(f"cond expected [B,N,{self.cond_dim}], got {tuple(cond.shape)}")
        global_tokens = self._global_tokens(z_global)
        image_tokens = self.cond_proj(cond)
        return torch.cat([global_tokens, image_tokens], dim=1)

    def _validate_forward_inputs(
        self,
        x_t_parts: torch.Tensor,
        t: torch.Tensor,
        z_global: torch.Tensor,
        name_tokens: torch.Tensor,
        name_mask: torch.Tensor,
        anchor: torch.Tensor,
        anchor_valid: torch.Tensor,
        part_valid: torch.Tensor,
    ) -> tuple[int, int]:
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
        if part_count > self.max_parts:
            raise ValueError(f"part_count={part_count} exceeds max_parts={self.max_parts}")
        if tuple(t.shape) != (batch_size,):
            raise ValueError(f"t expected [{batch_size}], got {tuple(t.shape)}")
        if tuple(z_global.shape) != (
            batch_size,
            self.latent_channels,
            self.resolution,
            self.resolution,
            self.resolution,
        ):
            raise ValueError(
                f"z_global expected [B,{self.latent_channels},{self.resolution},{self.resolution},{self.resolution}], "
                f"got {tuple(z_global.shape)}"
            )
        if name_tokens.dim() != 4 or name_tokens.shape[:2] != (batch_size, part_count):
            raise ValueError(
                f"name_tokens expected [B,K,L,{self.clip_name_dim}], got {tuple(name_tokens.shape)}"
            )
        if name_tokens.shape[-1] != self.clip_name_dim:
            raise ValueError(
                f"name_tokens dim expected {self.clip_name_dim}, got {name_tokens.shape[-1]}"
            )
        if tuple(name_mask.shape) != tuple(name_tokens.shape[:3]):
            raise ValueError(
                f"name_mask shape {tuple(name_mask.shape)} must match name_tokens[:3] "
                f"{tuple(name_tokens.shape[:3])}"
            )
        if anchor.dim() != 4 or anchor.shape[:2] != (batch_size, part_count) or anchor.shape[-1] != 4:
            raise ValueError(f"anchor expected [B,K,V,4], got {tuple(anchor.shape)}")
        if anchor.shape[2] != self.num_views:
            raise ValueError(f"anchor view count expected {self.num_views}, got {anchor.shape[2]}")
        if tuple(anchor_valid.shape) != tuple(anchor.shape[:3]):
            raise ValueError(
                f"anchor_valid shape {tuple(anchor_valid.shape)} must match anchor[:3] "
                f"{tuple(anchor.shape[:3])}"
            )
        if tuple(part_valid.shape) != (batch_size, part_count):
            raise ValueError(
                f"part_valid expected {(batch_size, part_count)}, got {tuple(part_valid.shape)}"
            )
        return batch_size, part_count

    def forward(
        self,
        x_t_parts: torch.Tensor,
        t: torch.Tensor,
        z_global: torch.Tensor,
        cond: torch.Tensor,
        name_tokens: torch.Tensor,
        name_mask: torch.Tensor,
        anchor: torch.Tensor,
        anchor_valid: torch.Tensor,
        part_valid: torch.Tensor,
        drop_name: torch.Tensor | None = None,
        drop_anchor: torch.Tensor | None = None,
        x_self_cond: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if x_self_cond is not None:
            raise NotImplementedError("PartMMDiTModel does not use self-conditioning")
        batch_size, part_count = self._validate_forward_inputs(
            x_t_parts,
            t,
            z_global,
            name_tokens,
            name_mask,
            anchor,
            anchor_valid,
            part_valid,
        )
        flat_x = x_t_parts.reshape(
            batch_size * part_count,
            self.latent_channels,
            self.resolution,
            self.resolution,
            self.resolution,
        )
        part_tokens = self.patchify_latent(flat_x)
        token_count = part_tokens.shape[1]

        cond_tokens, cond_mask = self.cond_token_builder(
            name_tokens,
            name_mask,
            anchor,
            anchor_valid,
            drop_name=drop_name,
            drop_anchor=drop_anchor,
        )
        t_part = t[:, None].expand(batch_size, part_count).reshape(-1)
        t_emb = self.t_embed(t_part)
        memory = self._memory_tokens(z_global, cond)
        memory = memory[:, None].expand(batch_size, part_count, -1, -1).reshape(
            batch_size * part_count,
            memory.shape[1],
            self.model_channels,
        )

        part_tokens = part_tokens.to(dtype=self.dtype)
        cond_tokens = cond_tokens.to(dtype=self.dtype)
        memory = memory.to(dtype=self.dtype)
        t_emb = t_emb.to(dtype=self.dtype)
        for layer_idx, block in enumerate(self.blocks):
            part_tokens, cond_tokens = block(part_tokens, cond_tokens, t_emb, cond_mask)
            part_tokens = self.memory_blocks[layer_idx](part_tokens, memory)
            if layer_idx in self.cross_part_layers:
                part_obj = part_tokens.reshape(
                    batch_size,
                    part_count,
                    token_count,
                    self.model_channels,
                )
                out = []
                cross_block = self.cross_part_blocks[str(layer_idx)]
                for batch_idx in range(batch_size):
                    out.append(cross_block(part_obj[batch_idx], part_valid[batch_idx]))
                part_tokens = torch.stack(out, dim=0).reshape(
                    batch_size * part_count,
                    token_count,
                    self.model_channels,
                )

        part_tokens = part_tokens.to(dtype=x_t_parts.dtype)
        out = self.unpatchify_latent(part_tokens)
        return out.reshape(
            batch_size,
            part_count,
            self.latent_channels,
            self.resolution,
            self.resolution,
            self.resolution,
        )
