from typing import *
import torch
import math
from . import DEBUG, BACKEND
from torch.nn.functional import scaled_dot_product_attention as torch_sdpa

if BACKEND == 'xformers':
    import xformers.ops as xops
elif BACKEND == 'flash_attn':
    import flash_attn
elif BACKEND == 'sdpa':
    sdpa = torch_sdpa
elif BACKEND == 'naive':
    pass
else:
    raise ValueError(f"Unknown attention backend: {BACKEND}")


__all__ = [
    'scaled_dot_product_attention',
]


def _naive_sdpa(q, k, v, attn_bias=None):
    """
    Naive implementation of scaled dot product attention.
    """
    q = q.permute(0, 2, 1, 3)   # [N, H, L, C]
    k = k.permute(0, 2, 1, 3)   # [N, H, L, C]
    v = v.permute(0, 2, 1, 3)   # [N, H, L, C]
    scale_factor = 1 / math.sqrt(q.size(-1))
    attn_weight = q @ k.transpose(-2, -1) * scale_factor
    if attn_bias is not None:
        attn_weight = attn_weight + attn_bias
    attn_weight = torch.softmax(attn_weight, dim=-1)
    out = attn_weight @ v
    out = out.permute(0, 2, 1, 3)   # [N, L, H, C]
    return out


@overload
def scaled_dot_product_attention(qkv: torch.Tensor) -> torch.Tensor:
    """
    Apply scaled dot product attention.

    Args:
        qkv (torch.Tensor): A [N, L, 3, H, C] tensor containing Qs, Ks, and Vs.
    """
    ...

@overload
def scaled_dot_product_attention(q: torch.Tensor, kv: torch.Tensor) -> torch.Tensor:
    """
    Apply scaled dot product attention.

    Args:
        q (torch.Tensor): A [N, L, H, C] tensor containing Qs.
        kv (torch.Tensor): A [N, L, 2, H, C] tensor containing Ks and Vs.
    """
    ...

@overload
def scaled_dot_product_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """
    Apply scaled dot product attention.

    Args:
        q (torch.Tensor): A [N, L, H, Ci] tensor containing Qs.
        k (torch.Tensor): A [N, L, H, Ci] tensor containing Ks.
        v (torch.Tensor): A [N, L, H, Co] tensor containing Vs.

    Note:
        k and v are assumed to have the same coordinate map.
    """
    ...

def _normalize_attn_bias(attn_bias, *, device, dtype):
    if attn_bias is None:
        return None
    return attn_bias.to(device=device, dtype=dtype)


def _sdpa_no_mask(q, k, v):
    q = q.permute(0, 2, 1, 3)   # [N, H, L, C]
    k = k.permute(0, 2, 1, 3)   # [N, H, L, C]
    v = v.permute(0, 2, 1, 3)   # [N, H, L, C]
    out = torch_sdpa(q, k, v)
    return out.permute(0, 2, 1, 3)   # [N, L, H, C]


def _apply_key_logit_bias(q, k, attn_bias):
    """Encode query-independent additive key logits in extra Q/K dimensions.

    For original head dim d and augmented dim D, choose
    q'=[a*q,1,0...] and k'=[a*k,bias*sqrt(D),0...] with a=(D/d)^0.25.
    Then q'k'/sqrt(D) == qk/sqrt(d) + bias, without materializing a
    [B,H,Q,K] attention mask. D is rounded to a small multiple of 8 so PyTorch
    keeps the efficient SDPA kernel without doubling memory for d=64.
    """
    attn_bias = _normalize_attn_bias(attn_bias, device=q.device, dtype=q.dtype)
    if attn_bias.dim() != 4:
        raise ValueError(f"attn_bias must be [B,H_or_1,1,K], got {tuple(attn_bias.shape)}")
    if attn_bias.shape[0] not in (1, q.shape[0]):
        raise ValueError(f"attn_bias batch {attn_bias.shape[0]} cannot broadcast to {q.shape[0]}")
    if attn_bias.shape[1] not in (1, q.shape[2]):
        raise ValueError(f"attn_bias heads {attn_bias.shape[1]} cannot broadcast to {q.shape[2]}")
    if attn_bias.shape[2] != 1:
        raise ValueError("attn_bias must be query-independent (shape[2] == 1)")
    if attn_bias.shape[3] != k.shape[1]:
        raise ValueError(f"attn_bias key length {attn_bias.shape[3]} != {k.shape[1]}")
    bias = attn_bias.expand(q.shape[0], q.shape[2], 1, k.shape[1]).squeeze(2)
    bias = bias.permute(0, 2, 1).contiguous()  # [B, K, H]
    d = int(q.shape[-1])
    d_aug = ((d + 1 + 7) // 8) * 8
    scale = (float(d_aug) / float(d)) ** 0.25
    q_extra = torch.zeros(*q.shape[:-1], d_aug - d, device=q.device, dtype=q.dtype)
    k_extra = torch.zeros(*k.shape[:-1], d_aug - d, device=k.device, dtype=k.dtype)
    q_extra[..., 0] = 1.0
    k_extra[..., 0] = bias * math.sqrt(float(d_aug))
    q_aug = torch.cat([q * scale, q_extra], dim=-1)
    k_aug = torch.cat([k * scale, k_extra], dim=-1)
    return q_aug, k_aug


def scaled_dot_product_attention(*args, **kwargs):
    attn_bias = kwargs.pop('attn_bias', None)
    arg_names_dict = {
        1: ['qkv'],
        2: ['q', 'kv'],
        3: ['q', 'k', 'v']
    }
    num_all_args = len(args) + len(kwargs)
    assert num_all_args in arg_names_dict, f"Invalid number of arguments, got {num_all_args}, expected 1, 2, or 3"
    for key in arg_names_dict[num_all_args][len(args):]:
        assert key in kwargs, f"Missing argument {key}"

    if num_all_args == 1:
        qkv = args[0] if len(args) > 0 else kwargs['qkv']
        assert len(qkv.shape) == 5 and qkv.shape[2] == 3, f"Invalid shape for qkv, got {qkv.shape}, expected [N, L, 3, H, C]"
        device = qkv.device

    elif num_all_args == 2:
        q = args[0] if len(args) > 0 else kwargs['q']
        kv = args[1] if len(args) > 1 else kwargs['kv']
        assert q.shape[0] == kv.shape[0], f"Batch size mismatch, got {q.shape[0]} and {kv.shape[0]}"
        assert len(q.shape) == 4, f"Invalid shape for q, got {q.shape}, expected [N, L, H, C]"
        assert len(kv.shape) == 5, f"Invalid shape for kv, got {kv.shape}, expected [N, L, 2, H, C]"
        device = q.device

    elif num_all_args == 3:
        q = args[0] if len(args) > 0 else kwargs['q']
        k = args[1] if len(args) > 1 else kwargs['k']
        v = args[2] if len(args) > 2 else kwargs['v']
        assert q.shape[0] == k.shape[0] == v.shape[0], f"Batch size mismatch, got {q.shape[0]}, {k.shape[0]}, and {v.shape[0]}"
        assert len(q.shape) == 4, f"Invalid shape for q, got {q.shape}, expected [N, L, H, Ci]"
        assert len(k.shape) == 4, f"Invalid shape for k, got {k.shape}, expected [N, L, H, Ci]"
        assert len(v.shape) == 4, f"Invalid shape for v, got {v.shape}, expected [N, L, H, Co]"
        device = q.device    

    if attn_bias is not None:
        if num_all_args == 1:
            q, k, v = qkv.unbind(dim=2)
        elif num_all_args == 2:
            k, v = kv.unbind(dim=2)
        q, k = _apply_key_logit_bias(q, k, attn_bias)
        return _sdpa_no_mask(q, k, v)

    if BACKEND == 'xformers':
        if num_all_args == 1:
            q, k, v = qkv.unbind(dim=2)
        elif num_all_args == 2:
            k, v = kv.unbind(dim=2)
        out = xops.memory_efficient_attention(q, k, v)
    elif BACKEND == 'flash_attn':
        if num_all_args == 1:
            out = flash_attn.flash_attn_qkvpacked_func(qkv)
        elif num_all_args == 2:
            out = flash_attn.flash_attn_kvpacked_func(q, kv)
        elif num_all_args == 3:
            out = flash_attn.flash_attn_func(q, k, v)
    elif BACKEND == 'sdpa':
        if num_all_args == 1:
            q, k, v = qkv.unbind(dim=2)
        elif num_all_args == 2:
            k, v = kv.unbind(dim=2)
        q = q.permute(0, 2, 1, 3)   # [N, H, L, C]
        k = k.permute(0, 2, 1, 3)   # [N, H, L, C]
        v = v.permute(0, 2, 1, 3)   # [N, H, L, C]
        attn_mask = _normalize_attn_bias(attn_bias, device=q.device, dtype=q.dtype)
        out = sdpa(q, k, v, attn_mask=attn_mask)         # [N, H, L, C]
        out = out.permute(0, 2, 1, 3)   # [N, L, H, C]
    elif BACKEND == 'naive':
        if num_all_args == 1:
            q, k, v = qkv.unbind(dim=2)
        elif num_all_args == 2:
            k, v = kv.unbind(dim=2)
        out = _naive_sdpa(q, k, v, attn_bias=attn_bias)
    else:
        raise ValueError(f"Unknown attention module: {BACKEND}")
    
    return out
