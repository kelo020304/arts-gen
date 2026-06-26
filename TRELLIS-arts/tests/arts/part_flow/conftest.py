"""Shared pytest fixtures for Phase 8 Part Flow tests.

sys.path injection + ATTN_BACKEND default are handled by the parent conftest
at TRELLIS-arts/tests/arts/conftest.py — this file only owns dummy fixtures.
"""

import numpy as np
import pytest

from ._dummy_solid import (  # noqa: E402
    make_dummy_solid,
    make_dummy_surface_indices,
)


@pytest.fixture
def dummy_solid_volume() -> np.ndarray:
    return make_dummy_solid(resolution=64, num_parts=5, empty_ratio=0.95, seed=42)


@pytest.fixture
def dummy_surface_indices(dummy_solid_volume) -> np.ndarray:
    return make_dummy_surface_indices(dummy_solid_volume)


@pytest.fixture
def dummy_dinov2_tokens():
    import torch
    V, T, D = 4, 1370, 1024
    return torch.randn(V, T, D, dtype=torch.float32)


@pytest.fixture
def dummy_part_info() -> dict:
    return {
        'num_parts': 5,
        'parts': {f'part_{i}': {'label': i, 'type': 'generic'} for i in range(1, 6)},
    }


@pytest.fixture(autouse=True)
def force_part_flow_cpu_attention(monkeypatch):
    """Keep model tests on the CPU-safe attention implementation."""
    try:
        from trellis.models.part_flow import attention_utils as au
    except Exception:
        return
    monkeypatch.setattr(au, '_get_attn_backend', lambda: 'sdpa_loop')
