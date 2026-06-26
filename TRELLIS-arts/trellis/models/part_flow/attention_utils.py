"""Attention + timestep utilities shared by Part Flow models.

Moved out of dirichlet_flow_predictor.py in Phase 8 (08-01) so that the
legacy Dirichlet files can be deleted (08-08) without breaking
part_flow_predictor.py's import chain.

Three public names:
    - _get_attn_backend() -> 'xformers' | 'flash_attn' | 'sdpa_loop'
    - _varlen_attention(q, k, v, q_seqlen, kv_seqlen) -> Tensor
    - TimestepEmbedder (DiT-style sinusoidal + MLP)
"""

from __future__ import annotations

from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


__all__ = ['_get_attn_backend', '_varlen_attention', 'TimestepEmbedder']


# =========================================================================== #
# Varlen attention helpers                                                    #
# =========================================================================== #

def _get_attn_backend() -> str:
    """Return one of 'xformers', 'flash_attn', 'sdpa_loop'."""
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
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    q_seqlen: List[int],
    kv_seqlen: List[int],
) -> torch.Tensor:
    """Varlen multi-head attention with packed [T, H, C] tensors."""
    backend = _get_attn_backend()
    assert q.dim() == 3 and k.dim() == 3 and v.dim() == 3, \
        f'Expected packed [T, H, C] tensors, got {q.shape}, {k.shape}, {v.shape}'
    assert sum(q_seqlen) == q.shape[0]
    assert sum(kv_seqlen) == k.shape[0] == v.shape[0]

    if backend == 'xformers':
        import xformers.ops as xops
        mask = xops.fmha.BlockDiagonalMask.from_seqlens(q_seqlen, kv_seqlen)
        out = xops.memory_efficient_attention(
            q.unsqueeze(0), k.unsqueeze(0), v.unsqueeze(0), mask,
        )[0]
        return out

    if backend == 'flash_attn':
        import flash_attn
        orig_dtype = q.dtype
        if orig_dtype == torch.float32:
            q, k, v = q.half(), k.half(), v.half()
        device = q.device
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
        return out.to(orig_dtype)

    # Fallback: per-sample SDPA loop
    outs = []
    q_off = kv_off = 0
    for ql, kvl in zip(q_seqlen, kv_seqlen):
        q_b = q[q_off:q_off + ql]
        k_b = k[kv_off:kv_off + kvl]
        v_b = v[kv_off:kv_off + kvl]
        q_h = q_b.transpose(0, 1).unsqueeze(0)
        k_h = k_b.transpose(0, 1).unsqueeze(0)
        v_h = v_b.transpose(0, 1).unsqueeze(0)
        out_h = F.scaled_dot_product_attention(q_h, k_h, v_h)
        outs.append(out_h.squeeze(0).transpose(0, 1))
        q_off += ql
        kv_off += kvl
    return torch.cat(outs, dim=0)


# =========================================================================== #
# Timestep embedding (standard DiT pattern)                                   #
# =========================================================================== #

class TimestepEmbedder(nn.Module):
    """Sinusoidal timestep embedding + 2-layer MLP -> hidden_dim.

    Matches TRELLIS's TimestepEmbedder (sparse_structure_flow.py:11) —
    inlined here to keep part_flow self-contained.
    """

    def __init__(self, hidden_dim: int, frequency_embedding_size: int = 256):
        super().__init__()
        self.frequency_embedding_size = frequency_embedding_size
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_dim, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim, bias=True),
        )

    @staticmethod
    def timestep_embedding(t: torch.Tensor, dim: int, max_period: int = 10000) -> torch.Tensor:
        half = dim // 2
        freqs = torch.exp(
            -np.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half,
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        return self.mlp(t_freq)
