"""Integration tests for L2SPAnchor + YAML parsing (Phase 7, Rule E-01 + E-03)."""

import hashlib
import os
import sys

import torch
import torch.nn as nn

try:
    import pytest
except ImportError:
    class _Mark:
        @staticmethod
        def parametrize(argnames, argvalues):
            if isinstance(argnames, str):
                names = [name.strip() for name in argnames.split(',')]
            else:
                names = list(argnames)

            def decorator(fn):
                setattr(fn, '_parametrize', (names, list(argvalues)))
                return fn

            return decorator

    class _PytestFallback:
        mark = _Mark()

    pytest = _PytestFallback()

# 顶级 conftest.py 已注入 TRELLIS-arts/ 到 sys.path。
PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', '..', '..')
)

from trellis.utils.arts.anchor_utils import L2SPAnchor
from trellis.utils.arts.config_utils import load_config
from trellis.utils.arts.lora_utils import apply_lora_to_model


def _hash_state(state):
    """SHA256 on concatenated tensor bytes, order-independent."""
    h = hashlib.sha256()
    for key in sorted(state.keys()):
        h.update(key.encode('utf-8'))
        h.update(
            state[key]
            .detach()
            .cpu()
            .contiguous()
            .view(-1)
            .to(torch.float32)
            .numpy()
            .tobytes()
        )
    return h.hexdigest()


def _make_tiny_mlp():
    return nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 2))


class _TinyLoRAModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_layer = nn.Linear(4, 4)
        self.blocks = nn.ModuleList([nn.Module()])
        self.blocks[0].to_qkv = nn.Linear(4, 12)
        self.out_layer = nn.Linear(4, 4)


def test_I1_model_body_not_polluted_by_anchor():
    model = _make_tiny_mlp()
    with torch.no_grad():
        for param in model.parameters():
            param.fill_(5.0)
    resumed_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    resumed_hash_before = _hash_state(resumed_state)

    pretrained_state = {k: torch.full_like(v, -1.0) for k, v in resumed_state.items()}
    anchor = L2SPAnchor.from_state_dict(model, pretrained_state, lambda_=1e-4)
    anchor.attach()

    resumed_hash_after = _hash_state({k: v for k, v in model.state_dict().items()})
    assert resumed_hash_after == resumed_hash_before


def test_I2_init_theta_0_matches_pretrained_state():
    model = _make_tiny_mlp()
    pretrained_state = {k: torch.randn_like(v) for k, v in model.state_dict().items()}
    anchor = L2SPAnchor.from_state_dict(model, pretrained_state, lambda_=1e-4)
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        assert name in anchor._theta_0
        expected = pretrained_state[name].to(dtype=param.dtype, device=param.device)
        assert torch.allclose(anchor._theta_0[name], expected)


def test_I2_invariant_theta_0_frozen_during_training():
    model = _make_tiny_mlp()
    pretrained_state = {k: torch.randn_like(v) for k, v in model.state_dict().items()}
    anchor = L2SPAnchor.from_state_dict(model, pretrained_state, lambda_=1e-4)
    anchor.attach()

    hash_before = _hash_state(anchor._theta_0)

    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    for _ in range(100):
        x = torch.randn(4, 4)
        y = model(x).sum()
        optimizer.zero_grad()
        y.backward()
        optimizer.step()

    hash_after = _hash_state(anchor._theta_0)
    assert hash_before == hash_after


def test_lora_plus_anchor_resolves_peft_prefixed_keep_trainable_keys():
    base_model = _TinyLoRAModel()
    pretrained_state = {k: v.detach().clone() for k, v in base_model.state_dict().items()}
    wrapped = apply_lora_to_model(
        base_model,
        {
            'enabled': True,
            'rank': 2,
            'alpha': 4,
            'dropout': 0.0,
            'target_modules': ['to_qkv'],
            'keep_trainable': ['input_layer', 'out_layer'],
        },
    )
    anchor = L2SPAnchor.from_state_dict(wrapped, pretrained_state, lambda_=1e-4)
    anchored_keys = set(anchor._theta_0.keys())
    assert 'base_model.model.input_layer.weight' in anchored_keys
    assert 'base_model.model.input_layer.bias' in anchored_keys
    assert 'base_model.model.out_layer.weight' in anchored_keys
    assert 'base_model.model.out_layer.bias' in anchored_keys
    assert not any('lora_' in key for key in anchored_keys)


@pytest.mark.parametrize(
    'yaml_rel',
    [
        'TRELLIS-arts/configs/arts/ss_flow_art/mv_4view.yaml',
        'TRELLIS-arts/configs/arts/ss_flow_art/mv_4view_lora.yaml',
        'TRELLIS-arts/configs/arts/slat_flow_art/mv_4view.yaml',
        'TRELLIS-arts/configs/arts/slat_flow_art/mv_4view_lora.yaml',
    ],
)
def test_legacy_yaml_has_no_anchor_section(yaml_rel):
    cfg = load_config(os.path.join(PROJECT_ROOT, yaml_rel))
    assert 'anchor' not in cfg


@pytest.mark.parametrize(
    'yaml_rel',
    [
        'TRELLIS-arts/configs/arts/ss_flow_art/mv_4view_l2sp.yaml',
        'TRELLIS-arts/configs/arts/slat_flow_art/mv_4view_l2sp.yaml',
    ],
)
def test_l2sp_yaml_anchor_section_complete(yaml_rel):
    cfg = load_config(os.path.join(PROJECT_ROOT, yaml_rel))
    anchor = cfg['anchor']
    assert anchor['enabled'] is True
    assert anchor['mode'] == 'l2_sp'
    assert abs(float(anchor['lambda']) - 1e-4) < 1e-10
    assert anchor['target'] == 'trainable'
    assert anchor['log_summary_every'] == 500
    assert anchor['log_per_layer_every'] == 2000


@pytest.mark.parametrize(
    'yaml_rel,expected_keep',
    [
        (
            'TRELLIS-arts/configs/arts/ss_flow_art/mv_4view_lora_ext.yaml',
            ['input_layer', 'out_layer'],
        ),
        (
            'TRELLIS-arts/configs/arts/slat_flow_art/mv_4view_lora_ext.yaml',
            ['input_layer', 'input_blocks', 'out_blocks', 'out_layer'],
        ),
    ],
)
def test_lora_ext_yaml_keep_trainable_present(yaml_rel, expected_keep):
    cfg = load_config(os.path.join(PROJECT_ROOT, yaml_rel))
    lora = cfg['lora']
    assert lora['enabled'] is True
    assert list(lora['keep_trainable']) == expected_keep


def main():
    for name, fn in sorted(globals().items()):
        if not name.startswith('test_') or not callable(fn):
            continue
        parametrize = getattr(fn, '_parametrize', None)
        if parametrize is None:
            fn()
            print(f'[PASS] {name}')
            continue
        argnames, argvalues = parametrize
        for args in argvalues:
            if not isinstance(args, tuple):
                args = (args,)
            kwargs = dict(zip(argnames, args))
            fn(**kwargs)
            print(f'[PASS] {name} {kwargs}')


if __name__ == '__main__':
    main()
