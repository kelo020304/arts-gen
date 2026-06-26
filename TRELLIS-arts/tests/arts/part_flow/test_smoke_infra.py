"""Smoke tests for Phase 8 Part Flow infra."""

import inspect

import pytest
import torch


def test_imports():
    from trellis.models.part_flow import attention_utils as au
    assert callable(au._get_attn_backend)
    assert callable(au._varlen_attention)
    assert isinstance(au.TimestepEmbedder, type)


def test_backend_roundtrip():
    from trellis.models.part_flow import attention_utils as au
    assert au._get_attn_backend() in {'xformers', 'flash_attn', 'sdpa_loop'}


def test_pfp_uses_attention_utils():
    # part_flow_predictor should now import these from attention_utils
    from trellis.models.part_flow import attention_utils as au
    from trellis.models.part_flow import part_flow_predictor as pfp
    assert inspect.getfile(pfp._varlen_attention) == inspect.getfile(au._varlen_attention)


def test_varlen_cpu_shape(monkeypatch):
    # [T=8, H=4, C=16] single-sample packed tensors.
    # Force the sdpa_loop fallback so this test runs on CPU even when
    # flash_attn / xformers are installed (they require CUDA tensors).
    from trellis.models.part_flow import attention_utils as au
    monkeypatch.setattr(au, '_get_attn_backend', lambda: 'sdpa_loop')
    q = torch.randn(8, 4, 16)
    k = torch.randn(8, 4, 16)
    v = torch.randn(8, 4, 16)
    out = au._varlen_attention(q, k, v, [8], [8])
    assert out.shape == (8, 4, 16)
