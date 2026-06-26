"""Variable-K Part Flow Predictor.

Key design differences from the pre-Phase-8 fixed-K predictor:

1. **No fixed-K classification head**. Part ids are object-local (in object
   A, part-id 1 might be "door"; in object B, part-id 1 might be "handle").
   We do NOT learn a global "Linear(H, K)" head that assigns shared
   semantics to each channel.

2. **No fixed-K simplex input embedding**. Encoding ``x_t`` as a ``Linear(K, H)``
   would give each simplex dim a learned global semantic — same problem.
   Instead, ``x_t`` is encoded as a weighted combination of the SAMPLE-SPECIFIC
   part tokens: ``x_t_emb[n] = sum_j x_t[n, j] * part_tokens[b, j]``.

3. **Sample-specific part tokens** are built by mask-guided pooling of DINOv2
   tokens:

   For each sample b and part_id j in {1..K_b}:
       part_tokens[b, j] = mean(cond[b, t] for t where mask_token_labels[b, t] == j)

   Slot 0 is reserved for empty and is never mask-pooled. Bg tokens
   (``mask_token_labels == 0``) are excluded. If part j has zero
   2D mask coverage (fully occluded), a learnable slot embedding is used as
   fallback (weak auxiliary).

4. **Output is sample-specific logits** via voxel·part_token dot-product:

       logits[n, j] = voxel_hidden[n] · part_tokens[batch_idx[n], j] / sqrt(H)

   Padding dims (j >= num_parts[b]) are masked to -inf before softmax.

This model works together with a :class:`bridges.BaseCategoricalFlowBridge` —
the model produces endpoint logits, the bridge handles conditional path
sampling, loss, and ODE step. See ``scripts/train/part_flow/flow_losses.py``.

Inputs to forward:
    x_t:                [N_total, K_max]  padded simplex (0 on invalid dims)
    t:                  [B]               time per sample
    coords:             [N_total, 4]      col0=batch_idx
    cond:               [B, V*T, cond_dim]  DINOv2 tokens
    mask_token_labels:  [B, V*T]          int64 in {0..K_b}; 0=bg, 1..K_b=part slots
    num_parts:          List[int]         per-sample K_b+1 (Phase 8: includes empty slot)
    is_on_surface:      [N_total]         int64 in {0, 1}  (Phase 8 D-05)

Outputs:
    endpoint_logits:    [N_total, K_max]   masked logits; -inf on invalid
    part_valid_mask:    [B, K_max]         bool, True on valid dims
    part_tokens:        [B, K_max, hidden_dim]  (for debugging / downstream)

Phase 8 conventions:
- Simplex slot 0 is the dedicated "empty" class (D-03).
- ``num_parts[b]`` now means K_b + 1 (real parts + empty slot); slot 0
  is always valid (D-10).
- ``build_part_tokens`` overwrites slot 0 with ``empty_token`` (D-08).
- ``surface_emb(is_on_surface)`` is summed into ``voxel_tokens`` (D-05).
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .attention_utils import (  # noqa: F401
    _get_attn_backend,
    _varlen_attention,
    TimestepEmbedder,
)


__all__ = [
    'ConditionTokenCompressor',
    'PartTokenTransformer',
    'DenseVoxelDecoderLayer',
    'PartFlowPredictor',
]


class ConditionTokenCompressor(nn.Module):
    """Compress per-view condition tokens to 3 tokens/view + 1 global token."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        num_view_tokens: int = 3,
        max_views: int = 16,
        dropout: float = 0.1,
    ):
        super().__init__()
        assert dim % num_heads == 0, f'dim {dim} not divisible by heads {num_heads}'
        self.dim = dim
        self.num_heads = num_heads
        self.num_view_tokens = int(num_view_tokens)
        self.max_views = int(max_views)
        self.view_queries = nn.Parameter(torch.randn(num_view_tokens, dim) * 0.02)
        self.global_query = nn.Parameter(torch.randn(1, dim) * 0.02)
        self.view_id_emb = nn.Embedding(max_views, dim)
        self.view_token_emb = nn.Embedding(num_view_tokens, dim)
        self.global_type_emb = nn.Parameter(torch.randn(1, dim) * 0.02)
        self.view_attn = nn.MultiheadAttention(
            dim, num_heads, dropout=dropout, batch_first=True,
        )
        self.global_attn = nn.MultiheadAttention(
            dim, num_heads, dropout=dropout, batch_first=True,
        )
        self.view_norm = nn.LayerNorm(dim, eps=1e-6)
        self.global_norm = nn.LayerNorm(dim, eps=1e-6)

    def forward(self, cond_proj: torch.Tensor, num_views: int) -> torch.Tensor:
        B, VT, H = cond_proj.shape
        assert H == self.dim
        assert 1 <= num_views <= self.max_views
        assert VT % num_views == 0, (
            f'cond token count {VT} not divisible by num_views={num_views}'
        )
        tokens_per_view = VT // num_views
        cond_by_view = cond_proj.view(B, num_views, tokens_per_view, H)

        view_outputs = []
        token_ids = torch.arange(self.num_view_tokens, device=cond_proj.device)
        token_pos = self.view_token_emb(token_ids).unsqueeze(0)
        for v in range(num_views):
            q = self.view_queries.unsqueeze(0).expand(B, -1, -1)
            q = q + token_pos
            q = q + self.view_id_emb.weight[v].view(1, 1, H)
            out, _ = self.view_attn(
                q, cond_by_view[:, v], cond_by_view[:, v], need_weights=False,
            )
            view_outputs.append(self.view_norm(out))
        view_tokens = torch.cat(view_outputs, dim=1)

        q_global = (
            self.global_query.unsqueeze(0).expand(B, -1, -1)
            + self.global_type_emb.unsqueeze(0)
        )
        global_token, _ = self.global_attn(
            q_global, view_tokens, view_tokens, need_weights=False,
        )
        global_token = self.global_norm(global_token)
        return torch.cat([view_tokens, global_token], dim=1)


class PartTokenTransformer(nn.Module):
    """Bidirectional transformer encoder over sample-local valid part slots."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=num_heads,
            dim_feedforward=dim * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)

    def forward(
        self,
        part_tokens: torch.Tensor,
        part_valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        out = self.encoder(part_tokens, src_key_padding_mask=~part_valid_mask)
        return out * part_valid_mask.unsqueeze(-1).to(out.dtype)


class DenseVoxelDecoderLayer(nn.Module):
    """Voxel decoder layer: part cross-attn -> condition cross-attn -> FFN."""

    def __init__(self, dim: int = 256, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        assert dim % num_heads == 0, f'dim {dim} not divisible by heads {num_heads}'
        self.ada_ln_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim, 6 * dim, bias=True),
        )
        self.norm_part = nn.LayerNorm(dim, eps=1e-6)
        self.part_attn = nn.MultiheadAttention(
            dim, num_heads, dropout=dropout, batch_first=True,
        )
        self.norm_cond = nn.LayerNorm(dim, eps=1e-6)
        self.cond_attn = nn.MultiheadAttention(
            dim, num_heads, dropout=dropout, batch_first=True,
        )
        self.norm_ffn = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim),
        )
        self.dropout = nn.Dropout(dropout)

    def _modulate(self, x, shift, scale):
        return x * (1.0 + scale) + shift

    def forward(
        self,
        voxel_tokens: torch.Tensor,
        t_emb_per_voxel: torch.Tensor,
        part_tokens: torch.Tensor,
        part_valid_mask: torch.Tensor,
        cond_tokens: torch.Tensor,
    ) -> torch.Tensor:
        mod = self.ada_ln_modulation(t_emb_per_voxel)
        shift_part, scale_part, gate_part, shift_ffn, scale_ffn, gate_ffn = mod.chunk(
            6, dim=-1,
        )

        q = self._modulate(
            self.norm_part(voxel_tokens), shift_part, scale_part,
        ).unsqueeze(0)
        part_out, _ = self.part_attn(
            q,
            part_tokens.unsqueeze(0),
            part_tokens.unsqueeze(0),
            key_padding_mask=(~part_valid_mask).unsqueeze(0),
            need_weights=False,
        )
        voxel_tokens = voxel_tokens + gate_part * self.dropout(part_out.squeeze(0))

        q = self.norm_cond(voxel_tokens).unsqueeze(0)
        cond_out, _ = self.cond_attn(
            q,
            cond_tokens.unsqueeze(0),
            cond_tokens.unsqueeze(0),
            need_weights=False,
        )
        voxel_tokens = voxel_tokens + self.dropout(cond_out.squeeze(0))

        h = self._modulate(self.norm_ffn(voxel_tokens), shift_ffn, scale_ffn)
        voxel_tokens = voxel_tokens + gate_ffn * self.dropout(self.ffn(h))
        return voxel_tokens


class PartFlowPredictor(nn.Module):
    """Variable-K Part Flow Predictor.

    Args:
        k_max: padding upper bound for num_parts per sample. NOT a semantic
            class count — different samples have different K_b <= k_max.
        hidden_dim: transformer hidden dim.
        num_layers: transformer depth.
        num_heads: attention heads.
        cond_dim: DINOv2 feature dim.
        dropout: dropout rate.
        use_slot_embedding_fallback: if True, parts with zero 2D mask coverage
            (fully occluded in all views) get a learnable slot_emb fallback
            vector instead of a zero part token. Default True.
    """

    def __init__(
        self,
        k_max: int = 128,
        hidden_dim: int = 256,
        num_layers: int = 4,
        num_heads: int = 8,
        cond_dim: int = 1024,
        dropout: float = 0.1,
        use_slot_embedding_fallback: bool = True,
        num_views: int = 4,
        part_token_layers: int = 2,
        condition_tokens_per_view: int = 3,
        voxel_chunk_size: int = 32768,
        use_slot_id_embedding: bool = False,
        slot_id_embedding_scale: float = 0.1,
        use_gradient_checkpointing: bool = False,
    ):
        super().__init__()
        self.k_max = k_max
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.cond_dim = cond_dim
        self.use_slot_embedding_fallback = use_slot_embedding_fallback
        self.num_views = int(num_views)
        self.voxel_chunk_size = int(voxel_chunk_size)
        self.use_slot_id_embedding = bool(use_slot_id_embedding)
        self.slot_id_embedding_scale = float(slot_id_embedding_scale)
        self.use_gradient_checkpointing = bool(use_gradient_checkpointing)

        # Positional embedding for voxel coordinates
        self.pos_embed = nn.Sequential(
            nn.Linear(3, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        # Timestep embedding (shared with legacy model — reused from sibling file)
        self.time_embed = TimestepEmbedder(hidden_dim)

        # Project DINOv2 cond -> hidden_dim (used for both RGB branch and
        # part-token pooling)
        self.rgb_proj = nn.Linear(cond_dim, hidden_dim)

        # Optional fallback slot embedding for occluded parts (k_max slots)
        if use_slot_embedding_fallback:
            self.slot_emb = nn.Parameter(torch.randn(k_max, hidden_dim) * 0.02)
        else:
            self.register_parameter('slot_emb', None)

        if self.use_slot_id_embedding:
            self.slot_id_emb = nn.Embedding(k_max, hidden_dim)
        else:
            self.slot_id_emb = None

        # Phase 8 (D-05, D-09): binary surface condition embedding.
        # is_on_surface in {0=off-surface, 1=on-surface} -> hidden_dim vector,
        # summed into voxel_tokens. Zero init so phase 8 starts indifferent
        # to surface and learns the useful signal from scratch.
        self.surface_emb = nn.Embedding(2, hidden_dim)
        nn.init.zeros_(self.surface_emb.weight)

        # Phase 8 (D-08): learnable empty-class token for simplex slot 0.
        # Slot 0 is the reserved empty class (D-03) and is NOT pooled from
        # mask tokens — empty has no corresponding 2D mask region. We
        # instantiate a fresh randn*0.02 Parameter (instead of reusing
        # self.slot_emb[0]) so that the empty semantics are explicit and
        # do not leak through the occluded-part fallback path.
        self.empty_token = nn.Parameter(torch.randn(hidden_dim) * 0.02)

        # Voxel-side and part-side projections for the final scoring head.
        # Keeping separate projections lets the model learn "which part-feature
        # axes matter" vs "which voxel-feature axes matter".
        self.voxel_score_proj = nn.Linear(hidden_dim, hidden_dim)
        self.part_score_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_norm = nn.LayerNorm(hidden_dim, eps=1e-6)

        self.condition_compressor = ConditionTokenCompressor(
            hidden_dim,
            num_heads,
            num_view_tokens=condition_tokens_per_view,
            max_views=max(16, self.num_views),
            dropout=dropout,
        )
        self.part_token_transformer = PartTokenTransformer(
            hidden_dim,
            num_heads,
            num_layers=part_token_layers,
            dropout=dropout,
        )

        # Decoder stack
        self.decoder_layers = nn.ModuleList([
            DenseVoxelDecoderLayer(hidden_dim, num_heads, dropout)
            for _ in range(num_layers)
        ])

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        for layer in self.decoder_layers:
            # Zero-init AdaLN final projection (DiT stability)
            nn.init.zeros_(layer.ada_ln_modulation[-1].weight)
            nn.init.zeros_(layer.ada_ln_modulation[-1].bias)

    # ------------------------------------------------------------------ #
    # Part token pooling                                                 #
    # ------------------------------------------------------------------ #

    def build_part_tokens(
        self,
        cond_proj: torch.Tensor,       # [B, VT, H]
        mask_token_labels: torch.Tensor,  # [B, VT] int64 in {0..K_b}
        num_parts: List[int],
    ) -> torch.Tensor:
        """Mask-guided average pooling of DINOv2 cond by 2D part id.

        Returns ``part_tokens [B, k_max, H]``. For sample b:
            part_tokens[b, 0] = empty_token
            part_tokens[b, j] = mean(cond_proj[b, t]) where mask_token_labels[b,t]==j,
            for j in {1..num_parts[b]-1}. Padding slots (j >= num_parts[b]) are 0
            unless ``use_slot_embedding_fallback`` is True AND the slot is valid
            but had zero mask coverage (occluded). Background tokens (mask==0)
            are excluded.
        """
        B, VT, H = cond_proj.shape
        device = cond_proj.device
        dtype = cond_proj.dtype

        # Slot 0 is background in mask_token_labels and empty in the simplex; it
        # must not be pooled from image tokens. Real part labels 1..K map
        # directly to simplex slots 1..K. Invalid labels are data-contract
        # errors, not values to clamp into a valid-looking slot.
        if mask_token_labels.numel() > 0:
            min_label = int(mask_token_labels.min().item())
            max_label = int(mask_token_labels.max().item())
            if min_label < 0 or max_label >= self.k_max:
                raise ValueError(
                    f'mask_token_labels must be in [0, {self.k_max - 1}], '
                    f'got [{min_label}, {max_label}]'
                )
        labels = mask_token_labels  # [B, VT]
        one_hot = F.one_hot(labels, num_classes=self.k_max).to(dtype)  # [B, VT, k_max]
        one_hot[..., 0] = 0  # exclude bg/CLS from pooling; slot 0 is empty_token

        counts = one_hot.sum(dim=1)  # [B, k_max]
        # Weighted sum
        sums = torch.einsum('bvk,bvh->bkh', one_hot, cond_proj)  # [B, k_max, H]
        counts_safe = counts.clamp(min=1.0).unsqueeze(-1)
        pooled = sums / counts_safe                              # [B, k_max, H]

        # Valid mask (per num_parts) and "present" mask (actually has 2D coverage)
        valid = self.build_part_valid_mask(num_parts, device)     # [B, k_max] bool
        present = counts > 0                                      # [B, k_max] bool

        # If a part slot is valid but not present (occluded), substitute fallback
        if self.slot_emb is not None:
            need_fallback = valid & (~present)                    # [B, k_max]
            if need_fallback.any():
                fallback = self.slot_emb.unsqueeze(0).expand(B, -1, -1)  # [B, k_max, H]
                pooled = torch.where(
                    need_fallback.unsqueeze(-1).expand_as(pooled),
                    fallback.to(dtype),
                    pooled,
                )
        # Zero out invalid (padding) slots
        pooled = pooled * valid.unsqueeze(-1).to(dtype)

        # Phase 8 (D-08): override slot 0 with learnable empty_token, for every sample.
        # Slot 0 is the reserved empty class (D-03), not a mask-pooled feature.
        # We do this AFTER the zero-out-invalid-slots step so the empty_token never
        # gets silenced by `valid.unsqueeze(-1)` — slot 0 is always valid (D-10).
        pooled[:, 0, :] = self.empty_token.to(pooled.dtype).unsqueeze(0).expand(B, H)
        return pooled

    def build_part_valid_mask(self, num_parts: List[int], device: torch.device) -> torch.Tensor:
        B = len(num_parts)
        idx = torch.arange(self.k_max, device=device).unsqueeze(0).expand(B, self.k_max)
        nps = torch.tensor(num_parts, device=device).unsqueeze(-1)
        return idx < nps

    # ------------------------------------------------------------------ #
    # Forward                                                            #
    # ------------------------------------------------------------------ #

    def forward(
        self,
        x_t: torch.Tensor,                # [N_total, k_max] padded simplex
        t: torch.Tensor,                  # [B] time per sample
        coords: torch.Tensor,             # [N_total, 4]
        cond: torch.Tensor,               # [B, V*T, cond_dim]
        mask_token_labels: torch.Tensor,  # [B, V*T] int64
        num_parts: List[int],             # per-sample K_b+1 (Phase 8: includes empty)
        is_on_surface: torch.Tensor,      # [N_total] int64 in {0, 1}  (Phase 8 D-05)
    ) -> Dict[str, torch.Tensor]:
        """Return dict with keys:
        - endpoint_logits [N_total, k_max] (masked to -inf on invalid dims)
        - part_valid_mask [B, k_max] bool
        - part_tokens     [B, k_max, hidden_dim]
        - valid_per_voxel [N_total, k_max] bool
        """
        assert x_t.dim() == 2 and x_t.shape[1] == self.k_max, \
            f'x_t shape {x_t.shape} mismatch k_max={self.k_max}'
        assert coords.dim() == 2 and coords.shape[1] == 4
        B, VT, D_cond = cond.shape
        assert D_cond == self.cond_dim
        assert t.shape[0] == B
        assert mask_token_labels.shape == (B, VT)
        N_total = x_t.shape[0]
        assert coords.shape[0] == N_total
        assert len(num_parts) == B

        # Phase 8 (D-05): surface condition invariants.
        assert is_on_surface.dim() == 1 and is_on_surface.shape[0] == N_total, \
            f'is_on_surface shape {tuple(is_on_surface.shape)} mismatch, expected [{N_total}]'
        assert is_on_surface.dtype in (torch.int64, torch.int32, torch.long), \
            f'is_on_surface dtype must be int64/int32, got {is_on_surface.dtype}'
        assert ((is_on_surface == 0) | (is_on_surface == 1)).all(), \
            'is_on_surface values must be in {0, 1}'

        device = x_t.device
        dtype = x_t.dtype
        batch_idx = coords[:, 0].long().to(device)
        xyz = coords[:, 1:4].to(device).float() / 64.0

        # Per-sample voxel seq lens. The loss labels share this row order, so
        # silently sorting here would desynchronize logits from supervision.
        assert torch.all(batch_idx[1:] >= batch_idx[:-1]), \
            'coords must be sorted by batch_idx before PartFlowPredictor.forward'

        n_seqlen: List[int] = []
        for b in range(B):
            n_seqlen.append(int((batch_idx == b).sum().item()))
        assert sum(n_seqlen) == N_total

        # --- Build masks and part tokens ---
        cond_proj = self.rgb_proj(cond)                        # [B, VT, H]
        part_tokens = self.build_part_tokens(
            cond_proj, mask_token_labels, num_parts,
        )                                                      # [B, k_max, H]
        part_valid_mask = self.build_part_valid_mask(num_parts, device)  # [B, k_max]
        valid_per_voxel = part_valid_mask[batch_idx]           # [N_total, k_max]

        if self.slot_id_emb is not None:
            slot_ids = torch.arange(self.k_max, device=device)
            slot_hint = self.slot_id_emb(slot_ids).unsqueeze(0).to(part_tokens.dtype)
            real_part_mask = part_valid_mask.clone()
            real_part_mask[:, 0] = False
            part_tokens = part_tokens + (
                self.slot_id_embedding_scale
                * slot_hint
                * real_part_mask.unsqueeze(-1).to(part_tokens.dtype)
            )

        part_tokens = self.part_token_transformer(part_tokens, part_valid_mask)
        part_tokens[:, 0, :] = self.empty_token.to(part_tokens.dtype).unsqueeze(0).expand(
            B, self.hidden_dim,
        )
        part_tokens = part_tokens * part_valid_mask.unsqueeze(-1).to(part_tokens.dtype)
        cond_tokens = self.condition_compressor(cond_proj, num_views=self.num_views)

        # --- Encode x_t via weighted sum of sample-specific part tokens ---
        # Compute per sample to avoid materializing [N_total, k_max, H].
        x_t_emb = torch.empty(N_total, self.hidden_dim, device=device, dtype=dtype)
        offset = 0
        for b, n_b in enumerate(n_seqlen):
            K_b = int(num_parts[b])
            sl = slice(offset, offset + n_b)
            x_t_emb[sl] = x_t[sl, :K_b] @ part_tokens[b, :K_b].to(dtype)
            offset += n_b

        # --- Voxel tokens: x_t encoding + positional ---
        surf_emb = self.surface_emb(is_on_surface.long().to(device))  # [N_total, H]
        voxel_tokens = x_t_emb + self.pos_embed(xyz) + surf_emb       # [N_total, H]

        # --- Time embedding per voxel ---
        t_emb = self.time_embed(t)                             # [B, H]
        t_emb_per_voxel = t_emb[batch_idx]                     # [N_total, H]

        c_seqlen_part = part_valid_mask.sum(dim=1).tolist()    # per-sample K_b
        assert all(c > 0 for c in c_seqlen_part), (
            'Every sample must have >= 1 part (num_parts > 0 expected).'
        )

        # --- Decoder stack ---
        decoded_chunks = []
        offset = 0
        for b, n_b in enumerate(n_seqlen):
            sl = slice(offset, offset + n_b)
            h = voxel_tokens[sl]
            t_h = t_emb_per_voxel[sl]
            part_b = part_tokens[b]
            valid_b = part_valid_mask[b]
            cond_b = cond_tokens[b]
            for layer in self.decoder_layers:
                if self.use_gradient_checkpointing and self.training:
                    from torch.utils.checkpoint import checkpoint

                    h = checkpoint(
                        layer, h, t_h, part_b, valid_b, cond_b,
                        use_reentrant=False,
                    )
                else:
                    h = layer(h, t_h, part_b, valid_b, cond_b)
            decoded_chunks.append(h)
            offset += n_b
        voxel_tokens = torch.cat(decoded_chunks, dim=0)

        # --- Endpoint logits via voxel·part_token dot-product ---
        voxel_h = self.out_norm(voxel_tokens)                  # [N_total, H]
        voxel_q = self.voxel_score_proj(voxel_h)               # [N_total, H]
        part_k = self.part_score_proj(part_tokens)             # [B, k_max, H]
        logits = torch.full(
            (N_total, self.k_max), -1e4, device=device, dtype=voxel_q.dtype,
        )
        offset = 0
        scale = math.sqrt(self.hidden_dim)
        for b, n_b in enumerate(n_seqlen):
            K_b = int(num_parts[b])
            sl = slice(offset, offset + n_b)
            logits[sl, :K_b] = (voxel_q[sl] @ part_k[b, :K_b].T) / scale
            offset += n_b

        # Mask padding dims to -inf so they never enter softmax
        logits = logits.masked_fill(~valid_per_voxel, -1e4)

        return {
            'endpoint_logits': logits,
            'part_valid_mask': part_valid_mask,
            'part_tokens': part_tokens,
            'valid_per_voxel': valid_per_voxel,
        }
