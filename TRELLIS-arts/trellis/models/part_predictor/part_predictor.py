"""
Query-based Part Predictor V2 (MMoT dual cross-attention + Learnable Queries).

Architecture per decoder layer (5 sub-layers):
    1. self-attn over K queries
    2. cross-attn queries -> voxel features (pos-embed of coords)
    3. cross-attn queries -> rgb feats (DINOv2 tokens, cond_proj)
    4. cross-attn queries -> mask feats (mask_embed → mask_proj), if available
    5. FFN
    mask head: per-voxel soft mask [K, N] via dot-product

MMoT-style dual branch (replaces additive fusion):
    2D part masks are downsampled to DINOv2 patch grid and encoded via
    nn.Embedding(max_k+1, cond_dim, padding_idx=0). The rgb and mask
    signals go through INDEPENDENT projection + cross-attention branches,
    keeping DINOv2 features uncontaminated while still injecting part
    identity. No camera projection needed — operates entirely in 2D.

Query initialization:
    Pure learnable queries (nn.Parameter), sliced by num_parts per sample.

Mask head: dot-product between refined queries [K, d] and voxel features [N, d],
    followed by softmax over K dim (per-voxel distribution over parts).

Convention: All mask tensors use [K, N] layout (Mask2Former standard:
    K=num_queries, N=num_voxels). softmax applied over dim=0 (K dim).

Multi-batch:
    - coords: [N_total, 4] where col0=batch_idx (SLat-style packed layout)
    - voxel cross-attn / self-attn / cond cross-attn all use packed varlen
      attention with block-diagonal mask (xformers) or cu_seqlens (flash_attn).
    - A pure-PyTorch per-sample-loop fallback is provided for environments
      without varlen kernels.
    - Returns List[Dict] (one per sample) for B>=2. B==1 legacy call signature
      (coords [N,3], num_parts:int) still returns a single dict.

Fusion modes (selectable via model.fusion_mode in YAML):
    - 'serial'    (default): 3 independent cross-attn steps (voxel -> rgb -> mask),
                             independent residuals. Current behavior, stable baseline.
    - 'concat_kv': shared Q/K/V/O, concat all modality KV, single softmax.
                   Highest scale-mismatch risk, kept for ablation only.
    - 'mmdit':    MMDiT-style. Per-modality K/V projections, joint softmax over
                  concatenated KV, RMSNorm QK-Norm (shared on Q, per-modality on K).
                  joint_o zero-init so initial residual is 0 (matches 'serial' start).

Serial mode modality order:
    Under fusion_mode='serial', the injection order of the three cross-attn
    steps (voxel / rgb / mask) is controlled by `model.serial_order` in YAML.
    Default ['voxel', 'rgb', 'mask'] keeps current behavior. Any permutation
    of the three names is accepted (6 possible orders total) — useful for
    ablating order bias in the additive residual chain.
"""

import warnings
from typing import Dict, List, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


__all__ = [
    'QueryPartPredictor',
    'PartDecoderLayer',
    'PartDecoderLayerConcatKV',
    'PartDecoderLayerMMDiT',
]


# --------------------------------------------------------------------------- #
# Varlen attention helper                                                     #
# --------------------------------------------------------------------------- #

def _get_attn_backend() -> str:
    """Return one of 'xformers', 'flash_attn', 'sdpa_loop'.

    Reads trellis sparse ATTN constant; if neither xformers nor flash_attn can
    be imported, falls back to a pure-PyTorch per-sample loop ('sdpa_loop').
    """
    try:
        from trellis.modules.sparse import ATTN as _ATTN
    except Exception:
        _ATTN = None

    if _ATTN == 'xformers':
        try:
            import xformers.ops  # noqa: F401
            return 'xformers'
        except Exception:
            pass
    if _ATTN == 'flash_attn':
        try:
            import flash_attn  # noqa: F401
            return 'flash_attn'
        except Exception:
            pass
    # Try any available backend before falling back.
    try:
        import xformers.ops  # noqa: F401
        return 'xformers'
    except Exception:
        pass
    try:
        import flash_attn  # noqa: F401
        return 'flash_attn'
    except Exception:
        pass
    return 'sdpa_loop'


def _varlen_attention(
    q: torch.Tensor,         # [T_q, H, C]
    k: torch.Tensor,         # [T_kv, H, C]
    v: torch.Tensor,         # [T_kv, H, C]
    q_seqlen: List[int],
    kv_seqlen: List[int],
) -> torch.Tensor:
    """Varlen multi-head attention producing packed output [T_q, H, C].

    Mirrors `TRELLIS.modules.sparse.attention.full_attn.sparse_scaled_dot_product_attention`
    backend branching. Used for three call sites in PartDecoderLayer.
    """
    backend = _get_attn_backend()
    assert q.dim() == 3 and k.dim() == 3 and v.dim() == 3, \
        f"Expected packed [T, H, C] tensors, got {q.shape}, {k.shape}, {v.shape}"
    assert sum(q_seqlen) == q.shape[0], \
        f"q seqlen sum mismatch: {sum(q_seqlen)} vs {q.shape[0]}"
    assert sum(kv_seqlen) == k.shape[0] == v.shape[0], \
        f"kv seqlen sum mismatch: {sum(kv_seqlen)} vs {k.shape[0]}/{v.shape[0]}"

    if backend == 'xformers':
        import xformers.ops as xops
        qB = q.unsqueeze(0)  # [1, T_q, H, C]
        kB = k.unsqueeze(0)
        vB = v.unsqueeze(0)
        mask = xops.fmha.BlockDiagonalMask.from_seqlens(q_seqlen, kv_seqlen)
        out = xops.memory_efficient_attention(qB, kB, vB, mask)[0]  # [T_q, H, C]
        return out

    if backend == 'flash_attn':
        import flash_attn
        device = q.device
        # flash_attn only supports fp16/bf16; cast fp32 inputs and restore after.
        orig_dtype = q.dtype
        if orig_dtype == torch.float32:
            q, k, v = q.half(), k.half(), v.half()
        cu_q = torch.cat([
            torch.zeros(1, dtype=torch.int32, device=device),
            torch.cumsum(torch.tensor(q_seqlen, dtype=torch.int32, device=device), 0),
        ]).int()
        cu_kv = torch.cat([
            torch.zeros(1, dtype=torch.int32, device=device),
            torch.cumsum(torch.tensor(kv_seqlen, dtype=torch.int32, device=device), 0),
        ]).int()
        out = flash_attn.flash_attn_varlen_func(
            q, k, v, cu_q, cu_kv, max(q_seqlen), max(kv_seqlen),
        )
        return out.to(orig_dtype)  # [T_q, H, C]

    # Fallback: per-sample SDPA loop (works on any torch >= 2.0, no extra deps).
    outs = []
    q_off = 0
    kv_off = 0
    for ql, kvl in zip(q_seqlen, kv_seqlen):
        q_b = q[q_off:q_off + ql]                      # [ql, H, C]
        k_b = k[kv_off:kv_off + kvl]                   # [kvl, H, C]
        v_b = v[kv_off:kv_off + kvl]                   # [kvl, H, C]
        # SDPA expects [..., seq, dim] with heads dim as batch-like; use
        # [H, ql, C] and [H, kvl, C] so heads attend independently.
        q_h = q_b.transpose(0, 1).unsqueeze(0)          # [1, H, ql, C]
        k_h = k_b.transpose(0, 1).unsqueeze(0)          # [1, H, kvl, C]
        v_h = v_b.transpose(0, 1).unsqueeze(0)          # [1, H, kvl, C]
        out_h = F.scaled_dot_product_attention(q_h, k_h, v_h)  # [1, H, ql, C]
        outs.append(out_h.squeeze(0).transpose(0, 1))   # [ql, H, C]
        q_off += ql
        kv_off += kvl
    return torch.cat(outs, dim=0)                       # [T_q, H, C]


# --------------------------------------------------------------------------- #
# PartDecoderLayer with packed varlen attention                               #
# --------------------------------------------------------------------------- #

class PartDecoderLayer(nn.Module):
    """Single Transformer Decoder layer for Part Predictor (packed multi-batch).

    Sub-layers (pre-norm style):
        1. Self-attention over part queries (block-diagonal per sample)
        2. Cross-attention: queries -> voxel features (varlen per sample)
        3. Cross-attention: queries -> DINOv2 tokens (varlen; uniform kv len)
        4. Feed-forward network

    All three attention calls go through `_varlen_attention` so the same code
    path works under xformers / flash_attn / pure-torch-fallback.

    Modality injection order (steps 2-4) is controlled by `serial_order`. Default
    ['voxel', 'rgb', 'mask'] matches the original hard-coded order. Any permutation
    of these three names is valid; the mask step is still a conditional no-op when
    `mask_feats is None`.

    Args:
        query_dim: Dimension of query/key/value vectors.
        num_heads: Number of attention heads.
        dropout: Dropout rate for attention and FFN.
        serial_order: Permutation of {'voxel', 'rgb', 'mask'} controlling the
            cross-attention step order in forward. None → default
            ['voxel', 'rgb', 'mask'].
    """

    def __init__(
        self,
        query_dim: int = 512,
        num_heads: int = 8,
        dropout: float = 0.1,
        serial_order: Optional[List[str]] = None,
    ):
        super().__init__()
        assert query_dim % num_heads == 0, 'query_dim must be divisible by num_heads'

        if serial_order is None:
            serial_order = ['voxel', 'rgb', 'mask']
        expected = {'voxel', 'rgb', 'mask'}
        if set(serial_order) != expected or len(serial_order) != 3:
            raise ValueError(
                f"serial_order must be a permutation of {sorted(expected)}, "
                f"got {serial_order!r}"
            )
        self.serial_order = list(serial_order)

        self.query_dim = query_dim
        self.num_heads = num_heads
        self.head_dim = query_dim // num_heads

        # -- Self-attention over queries --
        self.norm_self = nn.LayerNorm(query_dim)
        self.self_q = nn.Linear(query_dim, query_dim)
        self.self_k = nn.Linear(query_dim, query_dim)
        self.self_v = nn.Linear(query_dim, query_dim)
        self.self_o = nn.Linear(query_dim, query_dim)

        # -- Cross-attention: queries -> voxel features --
        self.norm_cross_voxel = nn.LayerNorm(query_dim)
        self.voxel_q = nn.Linear(query_dim, query_dim)
        self.voxel_k = nn.Linear(query_dim, query_dim)
        self.voxel_v = nn.Linear(query_dim, query_dim)
        self.voxel_o = nn.Linear(query_dim, query_dim)

        # -- Cross-attention: queries -> rgb (DINOv2) tokens --
        self.norm_cross_rgb = nn.LayerNorm(query_dim)
        self.cross_rgb_q = nn.Linear(query_dim, query_dim)
        self.cross_rgb_k = nn.Linear(query_dim, query_dim)
        self.cross_rgb_v = nn.Linear(query_dim, query_dim)
        self.cross_rgb_o = nn.Linear(query_dim, query_dim)

        # -- Cross-attention: queries -> mask tokens (MMoT dual branch) --
        self.norm_cross_mask = nn.LayerNorm(query_dim)
        self.cross_mask_q = nn.Linear(query_dim, query_dim)
        self.cross_mask_k = nn.Linear(query_dim, query_dim)
        self.cross_mask_v = nn.Linear(query_dim, query_dim)
        self.cross_mask_o = nn.Linear(query_dim, query_dim)

        # -- Feed-forward --
        self.norm_ffn = nn.LayerNorm(query_dim)
        self.ffn = nn.Sequential(
            nn.Linear(query_dim, query_dim * 4),
            nn.GELU(),
            nn.Linear(query_dim * 4, query_dim),
        )

        self.dropout = nn.Dropout(dropout)

    def _reshape_heads(self, x: torch.Tensor) -> torch.Tensor:
        """[T, D] -> [T, H, C]."""
        T, D = x.shape
        return x.view(T, self.num_heads, self.head_dim)

    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        """[T, H, C] -> [T, D]."""
        T, H, C = x.shape
        return x.reshape(T, H * C)

    def forward(
        self,
        queries: torch.Tensor,                     # [sum_K, D] packed queries
        voxel_feats: torch.Tensor,                 # [sum_N, D] packed voxel feats
        rgb_feats: torch.Tensor,                   # [sum_rgb, D] packed DINOv2 feats
        mask_feats: Optional[torch.Tensor],        # [sum_mask, D] gathered label>0 feats, or None
        k_seqlen: List[int],                       # per-sample query count
        n_seqlen: List[int],                       # per-sample voxel count
        c_seqlen_rgb: List[int],                   # per-sample rgb count (uniform VT)
        c_seqlen_mask: Optional[List[int]] = None, # per-sample mask count (after gather)
    ) -> torch.Tensor:
        """Return refined queries [sum_K, D].

        Step 4 (mask cross-attn) only attends tokens with label>0 (gathered
        upstream). label=0 (bg/CLS/dropped-view/no-mask) positions are strictly
        excluded from the mask KV set — they don't enter the softmax denominator.
        """
        # --- 1. Self-attention over queries (block-diagonal, k_seqlen) ---
        q_in = self.norm_self(queries)
        q = self._reshape_heads(self.self_q(q_in))
        k = self._reshape_heads(self.self_k(q_in))
        v = self._reshape_heads(self.self_v(q_in))
        out = _varlen_attention(q, k, v, k_seqlen, k_seqlen)
        out = self.self_o(self._merge_heads(out))
        queries = queries + self.dropout(out)

        # --- 2-4. Cross-attention blocks dispatched by self.serial_order ---
        # Supported modality names: 'voxel', 'rgb', 'mask'.
        # Order is a permutation of these three, configured via __init__.
        # Step 4 (mask cross-attn) only attends tokens with label>0 (gathered
        # upstream). label=0 (bg/CLS/dropped-view/no-mask) positions are strictly
        # excluded from the mask KV set — they don't enter the softmax denominator.
        for modality in self.serial_order:
            if modality == 'voxel':
                q_in = self.norm_cross_voxel(queries)
                q = self._reshape_heads(self.voxel_q(q_in))
                k = self._reshape_heads(self.voxel_k(voxel_feats))
                v = self._reshape_heads(self.voxel_v(voxel_feats))
                out = _varlen_attention(q, k, v, k_seqlen, n_seqlen)
                out = self.voxel_o(self._merge_heads(out))
                queries = queries + self.dropout(out)
            elif modality == 'rgb':
                q_in = self.norm_cross_rgb(queries)
                q = self._reshape_heads(self.cross_rgb_q(q_in))
                k = self._reshape_heads(self.cross_rgb_k(rgb_feats))
                v = self._reshape_heads(self.cross_rgb_v(rgb_feats))
                out = _varlen_attention(q, k, v, k_seqlen, c_seqlen_rgb)
                out = self.cross_rgb_o(self._merge_heads(out))
                queries = queries + self.dropout(out)
            elif modality == 'mask':
                if mask_feats is not None:
                    assert c_seqlen_mask is not None, \
                        "c_seqlen_mask must be provided when mask_feats is not None"
                    q_in = self.norm_cross_mask(queries)
                    q = self._reshape_heads(self.cross_mask_q(q_in))
                    k = self._reshape_heads(self.cross_mask_k(mask_feats))
                    v = self._reshape_heads(self.cross_mask_v(mask_feats))
                    out = _varlen_attention(q, k, v, k_seqlen, c_seqlen_mask)
                    out = self.cross_mask_o(self._merge_heads(out))
                    queries = queries + self.dropout(out)
                # else: mask step is a no-op (no mask info available)

        # --- 5. Feed-forward ---
        q_in = self.norm_ffn(queries)
        queries = queries + self.dropout(self.ffn(q_in))

        return queries


# --------------------------------------------------------------------------- #
# PartDecoderLayerConcatKV (scheme B: concat KV + shared Q/K/V/O)             #
# --------------------------------------------------------------------------- #

class PartDecoderLayerConcatKV(nn.Module):
    """Alternative decoder layer — scheme B: Concat KV + shared Q/K/V/O.

    Replaces the three serial cross-attn steps (voxel / rgb / mask) with a
    single joint cross-attention. All three modalities' KV are concatenated
    and go through ONE softmax, via shared Q/K/V/O projections.

    Sub-layers:
        1. Self-attention over queries (unchanged vs PartDecoderLayer)
        2. Joint cross-attention: queries -> concat(voxel, rgb, mask?)
           [single softmax, single residual]
        3. FFN (unchanged)

    Warning:
        Modality scale mismatch (voxel pos_embed vs DINOv2 token amplitude
        vs mask_embed) likely dominates the joint softmax. This scheme is
        included for ablation completeness, not recommended as a daily
        driver.

    Args: same as PartDecoderLayer.
    """

    def __init__(self, query_dim: int = 512, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        assert query_dim % num_heads == 0, 'query_dim must be divisible by num_heads'
        self.query_dim = query_dim
        self.num_heads = num_heads
        self.head_dim = query_dim // num_heads

        # Self-attn (same as PartDecoderLayer)
        self.norm_self = nn.LayerNorm(query_dim)
        self.self_q = nn.Linear(query_dim, query_dim)
        self.self_k = nn.Linear(query_dim, query_dim)
        self.self_v = nn.Linear(query_dim, query_dim)
        self.self_o = nn.Linear(query_dim, query_dim)

        # Joint cross-attn (shared Q/K/V/O across all modalities)
        self.norm_joint = nn.LayerNorm(query_dim)
        self.joint_q = nn.Linear(query_dim, query_dim)
        self.joint_k = nn.Linear(query_dim, query_dim)
        self.joint_v = nn.Linear(query_dim, query_dim)
        self.joint_o = nn.Linear(query_dim, query_dim)

        # FFN (same)
        self.norm_ffn = nn.LayerNorm(query_dim)
        self.ffn = nn.Sequential(
            nn.Linear(query_dim, query_dim * 4),
            nn.GELU(),
            nn.Linear(query_dim * 4, query_dim),
        )
        self.dropout = nn.Dropout(dropout)

    def _reshape_heads(self, x: torch.Tensor) -> torch.Tensor:
        T, D = x.shape
        return x.view(T, self.num_heads, self.head_dim)

    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        T, H, C = x.shape
        return x.reshape(T, H * C)

    def forward(
        self,
        queries: torch.Tensor,
        voxel_feats: torch.Tensor,
        rgb_feats: torch.Tensor,
        mask_feats: Optional[torch.Tensor],
        k_seqlen: List[int],
        n_seqlen: List[int],
        c_seqlen_rgb: List[int],
        c_seqlen_mask: Optional[List[int]] = None,
    ) -> torch.Tensor:
        B = len(k_seqlen)
        assert len(n_seqlen) == B and len(c_seqlen_rgb) == B

        # --- 1. Self-attn ---
        q_in = self.norm_self(queries)
        q = self._reshape_heads(self.self_q(q_in))
        k = self._reshape_heads(self.self_k(q_in))
        v = self._reshape_heads(self.self_v(q_in))
        out = _varlen_attention(q, k, v, k_seqlen, k_seqlen)
        out = self.self_o(self._merge_heads(out))
        queries = queries + self.dropout(out)

        # --- 2. Joint cross-attn: concat [voxel, rgb, (mask)] per sample ---
        # Per-sample KV layout: [n_b | VT_b | (mask_b)], dynamically drop mask
        # slot when mask_feats is None. Reassemble packed KV by interleaving
        # per-sample slices so _varlen_attention's block-diagonal mask sees the
        # right concatenated chunk per sample.
        has_mask = mask_feats is not None
        if has_mask:
            assert c_seqlen_mask is not None and len(c_seqlen_mask) == B, (
                "c_seqlen_mask must be provided when mask_feats is not None"
            )

        kv_chunks: List[torch.Tensor] = []
        joint_kv_seqlen: List[int] = []
        voxel_off = 0
        rgb_off = 0
        mask_off = 0
        for b in range(B):
            nb = n_seqlen[b]
            vt_b = c_seqlen_rgb[b]
            kv_chunks.append(voxel_feats[voxel_off:voxel_off + nb])
            kv_chunks.append(rgb_feats[rgb_off:rgb_off + vt_b])
            total = nb + vt_b
            if has_mask:
                mb = c_seqlen_mask[b]
                kv_chunks.append(mask_feats[mask_off:mask_off + mb])
                mask_off += mb
                total += mb
            joint_kv_seqlen.append(total)
            voxel_off += nb
            rgb_off += vt_b
        kv_cat = torch.cat(kv_chunks, dim=0) if len(kv_chunks) > 1 else kv_chunks[0]

        q_in = self.norm_joint(queries)
        q = self._reshape_heads(self.joint_q(q_in))
        k = self._reshape_heads(self.joint_k(kv_cat))
        v = self._reshape_heads(self.joint_v(kv_cat))
        out = _varlen_attention(q, k, v, k_seqlen, joint_kv_seqlen)
        out = self.joint_o(self._merge_heads(out))
        queries = queries + self.dropout(out)

        # --- 3. FFN ---
        q_in = self.norm_ffn(queries)
        queries = queries + self.dropout(self.ffn(q_in))
        return queries


# --------------------------------------------------------------------------- #
# PartDecoderLayerMMDiT (scheme C: per-modality K/V + QK-Norm + joint softmax) #
# --------------------------------------------------------------------------- #

class _RMSNorm(nn.Module):
    """RMSNorm for QK-Norm in MMDiT-style fusion.

    Applied per-head after Q/K projection (on the last dim = head_dim), before
    attention. Standard RMSNorm with a learnable per-dim scale and no bias.
    """

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [..., dim]
        rms = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).rsqrt()
        return x * rms * self.weight


class PartDecoderLayerMMDiT(nn.Module):
    """Alternative decoder layer — scheme C: MMDiT-style joint cross-attention.

    Per-modality independent K/V projections + QK-Norm (RMSNorm on Q/K) +
    joint softmax over all modalities. Query projection is shared (single
    `joint_q`) since the query stream is single (only queries get updated).

    Sub-layers:
        1. Self-attention over queries
        2. Joint cross-attention with independent K/V per modality:
              K_cat = concat([voxel_k, rgb_k, mask_k])
              V_cat = concat([voxel_v, rgb_v, mask_v])
              out = Attn(joint_q, K_cat, V_cat)  # single softmax
           QK-Norm: shared on Q side (single `qk_norm_q`), per-modality on K
           side (`qk_norm_voxel_k` / `qk_norm_rgb_k` / `qk_norm_mask_k`).
           joint_o zero-init: residual starts at 0, matching scheme A's
           initial behavior.
        3. FFN

    Model-dropout semantics (mask missing): mask branch is dynamically removed
    from the concat; kv_seqlen adjusted per sample. This preserves the Round 5
    gather invariant (label=0 positions never enter the softmax denominator).

    QK-Norm design note:
        Single Q stream cannot wear three modality-specific Q norms while
        preserving block-diagonal varlen-attn alignment (would require
        Q-repeat + output aggregation). We adopt the "shared Q norm +
        per-modality K norm" simplification common in MMDiT-for-single-stream
        (e.g. single-branch FLUX / SD3 adaptations), which keeps the main K-side
        scale-calibration benefit.

    Args: same as PartDecoderLayer.
    """

    def __init__(self, query_dim: int = 512, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        assert query_dim % num_heads == 0, 'query_dim must be divisible by num_heads'
        self.query_dim = query_dim
        self.num_heads = num_heads
        self.head_dim = query_dim // num_heads

        # Self-attn
        self.norm_self = nn.LayerNorm(query_dim)
        self.self_q = nn.Linear(query_dim, query_dim)
        self.self_k = nn.Linear(query_dim, query_dim)
        self.self_v = nn.Linear(query_dim, query_dim)
        self.self_o = nn.Linear(query_dim, query_dim)

        # Joint cross-attn projections (shared Q, per-modality K/V)
        self.norm_joint = nn.LayerNorm(query_dim)
        self.joint_q = nn.Linear(query_dim, query_dim)
        self.voxel_k = nn.Linear(query_dim, query_dim)
        self.voxel_v = nn.Linear(query_dim, query_dim)
        self.cross_rgb_k = nn.Linear(query_dim, query_dim)
        self.cross_rgb_v = nn.Linear(query_dim, query_dim)
        self.cross_mask_k = nn.Linear(query_dim, query_dim)
        self.cross_mask_v = nn.Linear(query_dim, query_dim)
        self.joint_o = nn.Linear(query_dim, query_dim)

        # QK-Norm (RMSNorm per head-dim).
        # One shared Q-side norm (single query stream) + three per-modality
        # K-side norms. This keeps block-diagonal varlen attention trivially
        # correct while still calibrating each modality's K scale.
        self.qk_norm_q = _RMSNorm(self.head_dim)
        self.qk_norm_voxel_k = _RMSNorm(self.head_dim)
        self.qk_norm_rgb_k = _RMSNorm(self.head_dim)
        self.qk_norm_mask_k = _RMSNorm(self.head_dim)

        # FFN
        self.norm_ffn = nn.LayerNorm(query_dim)
        self.ffn = nn.Sequential(
            nn.Linear(query_dim, query_dim * 4),
            nn.GELU(),
            nn.Linear(query_dim * 4, query_dim),
        )
        self.dropout = nn.Dropout(dropout)

    def _reshape_heads(self, x: torch.Tensor) -> torch.Tensor:
        T, D = x.shape
        return x.view(T, self.num_heads, self.head_dim)

    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        T, H, C = x.shape
        return x.reshape(T, H * C)

    def forward(
        self,
        queries: torch.Tensor,
        voxel_feats: torch.Tensor,
        rgb_feats: torch.Tensor,
        mask_feats: Optional[torch.Tensor],
        k_seqlen: List[int],
        n_seqlen: List[int],
        c_seqlen_rgb: List[int],
        c_seqlen_mask: Optional[List[int]] = None,
    ) -> torch.Tensor:
        B = len(k_seqlen)
        has_mask = mask_feats is not None
        if has_mask:
            assert c_seqlen_mask is not None and len(c_seqlen_mask) == B, (
                "c_seqlen_mask must be provided when mask_feats is not None"
            )

        # --- 1. Self-attn (same as scheme A) ---
        q_in = self.norm_self(queries)
        q = self._reshape_heads(self.self_q(q_in))
        k = self._reshape_heads(self.self_k(q_in))
        v = self._reshape_heads(self.self_v(q_in))
        out = _varlen_attention(q, k, v, k_seqlen, k_seqlen)
        queries = queries + self.dropout(self.self_o(self._merge_heads(out)))

        # --- 2. MMDiT joint cross-attn ---
        q_in = self.norm_joint(queries)
        q_proj = self._reshape_heads(self.joint_q(q_in))

        # Per-modality K/V projections
        voxel_k = self._reshape_heads(self.voxel_k(voxel_feats))
        voxel_v = self._reshape_heads(self.voxel_v(voxel_feats))
        rgb_k = self._reshape_heads(self.cross_rgb_k(rgb_feats))
        rgb_v = self._reshape_heads(self.cross_rgb_v(rgb_feats))
        if has_mask:
            mask_k = self._reshape_heads(self.cross_mask_k(mask_feats))
            mask_v = self._reshape_heads(self.cross_mask_v(mask_feats))

        # QK-Norm: Q-side shared (single query stream), K-side per-modality
        q_proj = self.qk_norm_q(q_proj)
        voxel_k = self.qk_norm_voxel_k(voxel_k)
        rgb_k = self.qk_norm_rgb_k(rgb_k)
        if has_mask:
            mask_k = self.qk_norm_mask_k(mask_k)

        # Per-sample KV concat (block-diagonal); mask slot dynamic when None.
        k_parts: List[torch.Tensor] = []
        v_parts: List[torch.Tensor] = []
        joint_kv_seqlen: List[int] = []
        voxel_off = 0
        rgb_off = 0
        mask_off = 0
        for b in range(B):
            nb = n_seqlen[b]
            vt_b = c_seqlen_rgb[b]
            k_parts.append(voxel_k[voxel_off:voxel_off + nb])
            v_parts.append(voxel_v[voxel_off:voxel_off + nb])
            k_parts.append(rgb_k[rgb_off:rgb_off + vt_b])
            v_parts.append(rgb_v[rgb_off:rgb_off + vt_b])
            total = nb + vt_b
            if has_mask:
                mb = c_seqlen_mask[b]
                k_parts.append(mask_k[mask_off:mask_off + mb])
                v_parts.append(mask_v[mask_off:mask_off + mb])
                mask_off += mb
                total += mb
            joint_kv_seqlen.append(total)
            voxel_off += nb
            rgb_off += vt_b

        k_cat = torch.cat(k_parts, dim=0)
        v_cat = torch.cat(v_parts, dim=0)
        out = _varlen_attention(q_proj, k_cat, v_cat, k_seqlen, joint_kv_seqlen)
        queries = queries + self.dropout(self.joint_o(self._merge_heads(out)))

        # --- 3. FFN ---
        q_in = self.norm_ffn(queries)
        queries = queries + self.dropout(self.ffn(q_in))
        return queries


# --------------------------------------------------------------------------- #
# QueryPartPredictor (multi-batch)                                            #
# --------------------------------------------------------------------------- #

class QueryPartPredictor(nn.Module):
    """Query-based Part Predictor V2 (Mask-in-KV + Learnable Queries).

    Inputs (training, B>=1):
        coords: [N_total, 4] int (col0=batch_idx, col1-3=xyz in 64-cube)
        cond:   [B, V*T, cond_dim] float (view-dropout applied upstream)
        mask_token_labels: [B, V*T] int64 (0=bg, 1..K=part id)
        num_parts: List[int]  (per-sample K_b)

    Voxel tokens = positional encoding of coords (MLP). No per-voxel features.

    Backward-compat (B=1 legacy, coords [N,3] + int num_parts):
        auto-detected: the old single-dict output is returned.

    Output (multi-batch):
        List[Dict] of length B, each dict has:
            mask_logits:  [K_b, N_b]
            soft_masks:   [K_b, N_b]  (softmax over K dim)
            class_logits: [K_b, num_part_types + 1]
            query_embs:   [K_b, query_dim]
    """

    def __init__(
        self,
        num_layers: int = 4,
        query_dim: int = 512,
        num_heads: int = 8,
        num_part_types: int = 32,
        max_k: int = 40,
        cond_dim: int = 1024,
        dropout: float = 0.1,
        fusion_mode: str = 'serial',  # 'serial' | 'concat_kv' | 'mmdit'
        serial_order: Optional[List[str]] = None,  # only used when fusion_mode=='serial'
    ):
        super().__init__()

        self.query_dim = query_dim
        self.num_part_types = num_part_types
        self.max_k = max_k

        # Learnable part queries — sliced by num_parts per sample at runtime.
        self.learnable_queries = nn.Parameter(torch.randn(max_k, query_dim) * 0.02)

        # Mask embedding: encodes 2D mask part identity (id 0 = bg/CLS/dropped).
        # Index 0 is strict zero (padding_idx); 1..max_k = part ids.
        # Projected into query space via mask_proj as an independent KV branch.
        self.mask_embed = nn.Embedding(max_k + 1, cond_dim, padding_idx=0)

        # RGB branch: projects DINOv2 tokens → query space
        self.cond_proj = nn.Linear(cond_dim, query_dim)
        # Mask branch: projects mask_embed lookup → query space (MMoT dual branch).
        # bias=False so mask_proj(0) == 0: preserves the label-0 = neutral
        # invariant (padding_idx=0 gives mask_embed(0)=0, and we must not
        # reintroduce a learnable constant vector via a bias here).
        self.mask_proj = nn.Linear(cond_dim, query_dim, bias=False)
        # Voxel token = pure positional encoding of coords (no per-voxel features).
        # MLP gives richer encoding than a single Linear(3, D).
        self.pos_embed = nn.Sequential(
            nn.Linear(3, query_dim),
            nn.SiLU(),
            nn.Linear(query_dim, query_dim),
        )

        # Fusion-mode dispatch: pick decoder layer class by YAML switch.
        layer_cls_map = {
            'serial': PartDecoderLayer,
            'concat_kv': PartDecoderLayerConcatKV,
            'mmdit': PartDecoderLayerMMDiT,
        }
        if fusion_mode not in layer_cls_map:
            raise ValueError(
                f"fusion_mode must be one of {list(layer_cls_map)}, "
                f"got {fusion_mode!r}"
            )
        self.fusion_mode = fusion_mode
        LayerCls = layer_cls_map[fusion_mode]

        # serial_order is only meaningful for fusion_mode == 'serial'.
        # For concat_kv / mmdit the three modalities go through a joint
        # softmax with no inherent ordering — silently ignore the field,
        # but warn if the user explicitly passed a non-default value.
        default_order = ['voxel', 'rgb', 'mask']
        effective_order = default_order if serial_order is None else list(serial_order)

        if fusion_mode == 'serial':
            self.serial_order = effective_order
            self.decoder_layers = nn.ModuleList([
                LayerCls(query_dim, num_heads, dropout, serial_order=effective_order)
                for _ in range(num_layers)
            ])
        else:
            if serial_order is not None and list(serial_order) != default_order:
                warnings.warn(
                    f"serial_order={serial_order!r} is ignored because "
                    f"fusion_mode={fusion_mode!r} (only used when fusion_mode='serial')",
                    UserWarning,
                    stacklevel=2,
                )
            self.serial_order = None
            self.decoder_layers = nn.ModuleList([
                LayerCls(query_dim, num_heads, dropout)
                for _ in range(num_layers)
            ])

        self.class_norm = nn.LayerNorm(query_dim)
        self.class_head = nn.Linear(query_dim, num_part_types + 1)

        self._init_weights()

    # ------------------------------------------------------------------ #
    # Shared prediction heads (used for final + aux intermediate layers) #
    # ------------------------------------------------------------------ #

    def _predict_sample(
        self,
        queries_b: torch.Tensor,      # [K_b, D]
        voxel_feats_b: torch.Tensor,  # [N_b, D]
    ) -> Dict[str, torch.Tensor]:
        """Compute mask + class predictions for a single sample."""
        mask_logits = torch.einsum('kd,nd->kn', queries_b, voxel_feats_b) / (self.query_dim ** 0.5)
        soft_masks = F.softmax(mask_logits, dim=0)
        class_logits = self.class_head(self.class_norm(queries_b))
        return {
            'mask_logits': mask_logits,
            'soft_masks': soft_masks,
            'class_logits': class_logits,
            'query_embs': queries_b,
        }

    def _predict_packed(
        self,
        queries: torch.Tensor,          # [sum_K, D]
        voxel_feats: torch.Tensor,      # [N_total, D]
        voxel_layouts: List[slice],
        k_seqlen: List[int],
    ) -> List[Dict[str, torch.Tensor]]:
        """Compute mask + class predictions for packed multi-batch queries."""
        outputs: List[Dict[str, torch.Tensor]] = []
        q_off = 0
        for b in range(len(k_seqlen)):
            K_b = k_seqlen[b]
            q_b = queries[q_off:q_off + K_b]
            vf_b = voxel_feats[voxel_layouts[b]]
            outputs.append(self._predict_sample(q_b, vf_b))
            q_off += K_b
        return outputs

    def _init_weights(self):
        for name, p in self.named_parameters():
            # Skip mask_embed — handled separately below to preserve padding_idx=0.
            if 'mask_embed' in name:
                continue
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        nn.init.normal_(self.learnable_queries, mean=0.0, std=0.02)
        # Init part embeddings (1..max_k); index 0 must stay zero (padding_idx).
        nn.init.zeros_(self.mask_embed.weight)
        nn.init.normal_(self.mask_embed.weight[1:], mean=0.0, std=0.02)

        # Zero-init joint_o in MMDiT decoder layers so the initial residual is 0,
        # matching scheme A's layer-0 starting point (self-attn + FFN only).
        # Scheme B's joint_o keeps xavier because its parameter count is ~1/3
        # and it needs a non-zero starting signal to learn.
        # RMSNorm weights are ones (already initialized in _RMSNorm.__init__).
        if self.fusion_mode == 'mmdit':
            for layer in self.decoder_layers:
                nn.init.zeros_(layer.joint_o.weight)
                if layer.joint_o.bias is not None:
                    nn.init.zeros_(layer.joint_o.bias)

    # ------------------------------------------------------------------ #
    # Forward                                                            #
    # ------------------------------------------------------------------ #

    def forward(
        self,
        coords: torch.Tensor,
        cond: torch.Tensor,
        num_parts: Union[int, List[int]] = 0,
        mask_token_labels: Optional[torch.Tensor] = None,
    ) -> Union[Dict[str, torch.Tensor], List[Dict[str, torch.Tensor]]]:
        """Forward pass.

        Args:
            coords: [N_total, 4] int (col0=batch_idx) or [N, 3] (legacy B=1)
            cond: [B, V*T, cond_dim] or [V*T, cond_dim] (legacy B=1)
            num_parts: List[int] or int — number of parts per sample
            mask_token_labels: [B, V*T] int64 — per-token mask part ids
                (0=bg, 1..K=part). None = no mask info (all-zero embedding).
        """
        # Legacy B=1 path: coords [N,3] + int num_parts.
        legacy = (
            coords.dim() == 2 and coords.shape[1] == 3
            and (isinstance(num_parts, int) or (hasattr(num_parts, 'dim') and num_parts.dim() == 0))
        )
        if legacy:
            mtl = mask_token_labels[0] if mask_token_labels is not None and mask_token_labels.dim() == 2 else mask_token_labels
            return self._forward_single(coords, cond, int(num_parts), mtl)

        # -------- Multi-batch packed path --------
        assert coords.dim() == 2 and coords.shape[1] == 4, \
            f"Expected coords [N,4] (col0=batch_idx) for multi-batch, got {coords.shape}"
        assert cond.dim() == 3, f"Expected cond [B, V*T, D], got {cond.shape}"
        B, VT, _ = cond.shape
        if isinstance(num_parts, int):
            num_parts = [num_parts] * B
        assert len(num_parts) == B, f"num_parts list len {len(num_parts)} != B {B}"

        # Per-sample voxel counts & layouts
        batch_idx = coords[:, 0].long()
        xyz = coords[:, 1:4]  # [N_total, 3]
        n_seqlen: List[int] = []
        voxel_layouts: List[slice] = []
        start = 0
        for b in range(B):
            nb = int((batch_idx == b).sum().item())
            n_seqlen.append(nb)
            voxel_layouts.append(slice(start, start + nb))
            start += nb
        assert start == coords.shape[0], \
            f"batch_idx coverage mismatch: {start} vs {coords.shape[0]}"

        # Verify coords already grouped by batch_idx (SLat collate_fn guarantees this).
        # If not grouped, sort once.
        if not torch.all(batch_idx[1:] >= batch_idx[:-1]):
            order = torch.argsort(batch_idx, stable=True)
            coords = coords[order]
            batch_idx = coords[:, 0].long()
            xyz = coords[:, 1:4]

        # Voxel tokens = pure positional encoding of coords, no per-voxel features.
        voxel_feats = self.pos_embed(xyz.float() / 64.0)  # [N_total, D]

        # Queries: learnable queries sliced by num_parts per sample
        k_seqlen: List[int] = []
        query_list = []
        for b in range(B):
            K_b = int(num_parts[b])
            assert K_b > 0, f"num_parts[{b}] must be > 0"
            assert K_b <= self.max_k, f"num_parts[{b}]={K_b} exceeds max_k={self.max_k}"
            k_seqlen.append(K_b)
            query_list.append(self.learnable_queries[:K_b])  # [K_b, D]
        queries = torch.cat(query_list, dim=0)              # [sum_K, D]

        # Cond: dual branch — rgb (DINOv2) and optional mask (MMoT-style)
        rgb_feats = self.cond_proj(cond)                        # [B, VT, D]
        rgb_feats_packed = rgb_feats.reshape(B * VT, -1)        # [B*VT, D]
        c_seqlen_rgb = [VT] * B
        # Strict label-0 neutrality:
        #   1. all-zero labels (or labels=None) → skip Step 4 entirely
        #   2. mixed labels → gather only label>0 positions into the mask KV;
        #      label=0 tokens (bg/CLS/dropped-view) do NOT enter the softmax.
        # This is stronger than just "mask_proj(0)=0" — it guarantees label=0
        # positions contribute ZERO to the mask cross-attention output.
        use_mask = (
            mask_token_labels is not None
            and int(mask_token_labels.max().item()) > 0
        )
        if use_mask:
            # Gather label>0 positions per sample. Result is packed tightly,
            # each sample's count in c_seqlen_mask.
            valid = mask_token_labels > 0                       # [B, VT] bool
            c_seqlen_mask: Optional[List[int]] = valid.sum(dim=1).tolist()
            assert all(c > 0 for c in c_seqlen_mask), (
                "use_mask=True but some sample has zero label>0 positions. "
                "This should not happen in normal training — a sample has "
                "parts so labels must contain at least one id>0."
            )
            mask_emb = self.mask_embed(mask_token_labels.clamp(max=self.max_k))  # [B, VT, cond_dim]
            mask_feats = self.mask_proj(mask_emb)               # [B, VT, D]
            # Packed gather: [sum(c_seqlen_mask), D]
            mask_feats_packed: Optional[torch.Tensor] = mask_feats[valid]
        else:
            mask_feats_packed = None
            c_seqlen_mask = None

        # Transformer decoder layers with auxiliary intermediate predictions
        aux_per_layer: List[List[Dict[str, torch.Tensor]]] = []
        for layer_idx, layer in enumerate(self.decoder_layers):
            queries = layer(
                queries, voxel_feats,
                rgb_feats_packed, mask_feats_packed,
                k_seqlen=k_seqlen, n_seqlen=n_seqlen,
                c_seqlen_rgb=c_seqlen_rgb,
                c_seqlen_mask=c_seqlen_mask,
            )
            # Auxiliary predictions from intermediate layers (training only)
            if self.training and layer_idx < len(self.decoder_layers) - 1:
                aux_per_layer.append(
                    self._predict_packed(queries, voxel_feats, voxel_layouts, k_seqlen)
                )

        # Final predictions from last layer
        outputs = self._predict_packed(queries, voxel_feats, voxel_layouts, k_seqlen)

        # Attach per-sample auxiliary outputs: List[Dict] per intermediate layer
        if aux_per_layer:
            for b in range(B):
                outputs[b]['aux_outputs'] = [
                    layer_preds[b] for layer_preds in aux_per_layer
                ]

        return outputs

    # ------------------------------------------------------------------ #
    # Legacy single-sample path (coords [N,3], unchanged semantics)      #
    # ------------------------------------------------------------------ #

    def _forward_single(
        self,
        coords: torch.Tensor,        # [N, 3]
        cond: torch.Tensor,          # [V*T, D] or [1, V*T, D]
        num_parts: int,
        mask_token_labels: Optional[torch.Tensor] = None,  # [V*T] int64
    ) -> Dict[str, torch.Tensor]:
        # Normalize cond to [V*T, D]
        if cond.dim() == 3:
            assert cond.shape[0] == 1, "legacy path expects B=1 cond"
            cond = cond[0]
        N = coords.shape[0]
        VT = cond.shape[0]

        voxel_feats = self.pos_embed(coords.float() / 64.0)  # [N, D]  (pure PE, no feats)

        # Dual branch: rgb (DINOv2) and optional mask (gather label>0 only).
        # All-zero labels ≡ None (skip Step 4); mixed labels → gather positions
        # with label>0, so label=0 tokens contribute zero to mask cross-attn.
        rgb_feats = self.cond_proj(cond)                     # [VT, D]
        c_seqlen_rgb = [VT]
        use_mask = (
            mask_token_labels is not None
            and int(mask_token_labels.max().item()) > 0
        )
        if use_mask:
            mtl = mask_token_labels.to(coords.device)
            valid = mtl > 0                                  # [VT] bool
            valid_count = int(valid.sum().item())
            assert valid_count > 0, (
                "use_mask=True but zero label>0 positions; this is a logic bug"
            )
            mask_emb = self.mask_embed(mtl.clamp(max=self.max_k))  # [VT, cond_dim]
            mask_feats_full = self.mask_proj(mask_emb)       # [VT, D]
            mask_feats: Optional[torch.Tensor] = mask_feats_full[valid]  # [valid_count, D]
            c_seqlen_mask: Optional[List[int]] = [valid_count]
        else:
            mask_feats = None
            c_seqlen_mask = None

        assert num_parts > 0, "num_parts must be > 0"
        queries = self.learnable_queries[:num_parts]         # [K, D]

        K = queries.shape[0]
        k_seqlen = [K]
        n_seqlen = [N]

        # Decoder layers with auxiliary intermediate predictions
        aux_outputs: List[Dict[str, torch.Tensor]] = []
        for layer_idx, layer in enumerate(self.decoder_layers):
            queries = layer(
                queries, voxel_feats, rgb_feats, mask_feats,
                k_seqlen=k_seqlen, n_seqlen=n_seqlen,
                c_seqlen_rgb=c_seqlen_rgb,
                c_seqlen_mask=c_seqlen_mask,
            )
            if self.training and layer_idx < len(self.decoder_layers) - 1:
                aux_outputs.append(self._predict_sample(queries, voxel_feats))

        result = self._predict_sample(queries, voxel_feats)
        if aux_outputs:
            result['aux_outputs'] = aux_outputs
        return result
