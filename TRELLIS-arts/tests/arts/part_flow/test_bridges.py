"""Bridge conformance tests: Fisher / GumbelSoftmax.

Every active bridge must satisfy the padding-discipline contract regardless
of flow family. Tests run on synthetic multi-K batches (CPU only).

Contract:
  1. sample_source -> simplex, sum=1 on valid dims, 0 on padding
  2. sample_conditional_path at t=0 ~ source, at t=t_max close to x_1
  3. sample_conditional_path preserves padding=0 invariant
  4. step preserves padding=0 invariant and simplex constraint
  5. compute_loss masks padding dims out of the softmax denominator

Legacy dirichlet/sfm bridges are NOT tested here — they're not active.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

# Trainer code now lives inside the trellis package — parent conftest injects
# TRELLIS-arts/ into sys.path, so direct imports work.
from trellis.models.part_flow import bridges as _BRIDGES  # noqa: E402


def _mixed_K_setup(k_max=8, num_parts=(2, 4, 7), n_per=(10, 15, 20), device='cpu'):
    """Build x_1 (one-hot), voxel_layout for a 3-sample mixed-K batch."""
    N_total = sum(n_per)
    x_1_one_hot = torch.zeros(N_total, k_max, device=device)
    voxel_layout = []
    offset = 0
    labels = []
    for K_b, n_b in zip(num_parts, n_per):
        voxel_layout.append(slice(offset, offset + n_b))
        cls = torch.randint(0, K_b, (n_b,), device=device)
        for i in range(n_b):
            x_1_one_hot[offset + i, cls[i]] = 1.0
        labels.append(cls)
        offset += n_b
    labels_all = torch.cat(labels, dim=0)
    return x_1_one_hot, voxel_layout, labels_all


def _check_padding_invariant(x, valid_per_voxel, name):
    pad_mass = x[~valid_per_voxel].abs().max().item()
    assert pad_mass < 1e-5, f'{name}: padding dims have mass {pad_mass}'


def _check_simplex(x, valid_per_voxel, name, atol=1e-3):
    sums = x.sum(dim=-1)
    expected = valid_per_voxel.any(dim=-1).float()
    assert torch.allclose(sums, expected, atol=atol), \
        f'{name}: sum-to-one violated (max err {(sums-expected).abs().max()})'


def _run_contract_suite(bridge, name, k_max=8):
    torch.manual_seed(0)
    num_parts = [2, 4, 7]
    n_per = [10, 15, 20]
    x_1, voxel_layout, labels = _mixed_K_setup(k_max, num_parts, n_per)
    N_total = sum(n_per)

    valid_pv = torch.zeros(N_total, k_max, dtype=torch.bool)
    for sl, K_b in zip(voxel_layout, num_parts):
        valid_pv[sl, :K_b] = True

    # 1. source
    x_0 = bridge.sample_source(num_parts, n_per, torch.device('cpu'))
    _check_padding_invariant(x_0, valid_pv, f'{name}: x_0')
    _check_simplex(x_0, valid_pv, f'{name}: x_0')

    # 2a. path at t=0 (should be ~source or near-uniform)
    t_zero = torch.zeros(N_total)
    x_t_0 = bridge.sample_conditional_path(x_1, t_zero, voxel_layout, num_parts, x_0=x_0)
    _check_padding_invariant(x_t_0, valid_pv, f'{name}: x_t (t=0)')
    _check_simplex(x_t_0, valid_pv, f'{name}: x_t (t=0)')

    # 2b. path at t=t_max (should concentrate on target)
    t_full = torch.full((N_total,), bridge.t_max)
    x_t_1 = bridge.sample_conditional_path(x_1, t_full, voxel_layout, num_parts, x_0=x_0)
    _check_padding_invariant(x_t_1, valid_pv, f'{name}: x_t (t=t_max)')
    _check_simplex(x_t_1, valid_pv, f'{name}: x_t (t=t_max)')
    target_prob = x_t_1.gather(1, labels.unsqueeze(-1)).squeeze(-1)
    # fisher hits exact vertex; gumbel with tau_min=0.01 gets ~0.99+
    assert target_prob.mean() > 0.5, \
        f'{name}: at t=t_max target prob mean {target_prob.mean():.3f} < 0.5'

    # 3. middle t
    t_mid = torch.full((N_total,), bridge.t_max / 2)
    x_t_mid = bridge.sample_conditional_path(x_1, t_mid, voxel_layout, num_parts, x_0=x_0)
    _check_padding_invariant(x_t_mid, valid_pv, f'{name}: x_t (mid)')
    _check_simplex(x_t_mid, valid_pv, f'{name}: x_t (mid)')

    # 4. step: oracle endpoint, expect target prob non-decreasing
    endpoint_probs = x_1.clone()
    x_next = bridge.step(
        x_t_mid, endpoint_probs,
        t_val=bridge.t_max / 2, dt=bridge.t_max / 10,
        voxel_layout=voxel_layout, num_parts=num_parts,
    )
    _check_padding_invariant(x_next, valid_pv, f'{name}: x_next (step)')
    _check_simplex(x_next, valid_pv, f'{name}: x_next (step)')
    p_before = x_t_mid.gather(1, labels.unsqueeze(-1)).squeeze(-1).mean()
    p_after = x_next.gather(1, labels.unsqueeze(-1)).squeeze(-1).mean()
    assert p_after >= p_before - 1e-3, \
        f'{name}: step with oracle regressed: {p_before:.3f} -> {p_after:.3f}'

    # 5. masked CE: padding-dim logit perturbations must not change loss
    logits = torch.randn(N_total, k_max)
    loss, metrics = bridge.compute_loss(logits, labels, valid_pv)
    assert torch.isfinite(loss), f'{name}: loss not finite'
    assert 'gt_prob_mean' in metrics and 'endpoint_acc' in metrics

    logits_alt = logits.clone()
    logits_alt[~valid_pv] = torch.randn_like(logits_alt[~valid_pv]) * 1e3
    loss_alt, _ = bridge.compute_loss(logits_alt, labels, valid_pv)
    assert torch.allclose(loss, loss_alt, atol=1e-4), \
        f'{name}: padding perturbation changed loss ({loss} vs {loss_alt})'

    print(f'[PASS] {name} contract suite (K_b={num_parts})')


def test_fisher_bridge_contract():
    br = _BRIDGES.build_bridge('fisher', k_max=8, t_max=1.0)
    _run_contract_suite(br, 'Fisher')


def test_gumbel_bridge_contract():
    # Use paper defaults; tau_min relatively small so t=t_max concentrates well
    br = _BRIDGES.build_bridge(
        'gumbel', k_max=8, t_max=1.0,
        tau_max=10.0, decay_rate=3.0, noise_scale=2.0, tau_min=0.01,
    )
    _run_contract_suite(br, 'Gumbel-Softmax')


def test_gumbel_temperature_schedule():
    """Paper Eq. 8: tau(t) = tau_max * exp(-lambda t). Spot-check endpoints."""
    br = _BRIDGES.build_bridge(
        'gumbel', k_max=4, t_max=1.0,
        tau_max=10.0, decay_rate=3.0, noise_scale=2.0, tau_min=0.001,
    )
    assert abs(br.tau_at(0.0) - 10.0) < 1e-6, f'tau(0) should be 10 got {br.tau_at(0.0)}'
    # tau(1) = 10 * exp(-3) ≈ 0.4979
    import math
    expected = 10.0 * math.exp(-3.0)
    assert abs(br.tau_at(1.0) - expected) < 1e-4, f'tau(1) should be ~{expected} got {br.tau_at(1.0)}'
    print(f'[PASS] Gumbel temperature schedule: tau(0)=10, tau(1)={expected:.4f}')


def test_build_bridge_rejects_dirichlet_and_sfm():
    for legacy in ('dirichlet', 'sfm', 'unknown'):
        try:
            _BRIDGES.build_bridge(legacy, k_max=8)
            raise AssertionError(f'build_bridge should reject {legacy!r}')
        except ValueError as e:
            assert legacy in str(e) or 'active options' in str(e)
    print('[PASS] build_bridge rejects legacy dirichlet/sfm and unknown types')


def test_build_bridge_drops_unused_kwargs():
    """Fisher config can inherit gumbel's tau_* kwargs; build_bridge drops them."""
    br = _BRIDGES.build_bridge(
        'fisher', k_max=8, t_max=1.0,
        tau_max=10.0, decay_rate=3.0, noise_scale=2.0, tau_min=0.01,
    )
    assert isinstance(br, _BRIDGES.FisherBridge)
    print('[PASS] build_bridge silently drops unused kwargs (fisher ignores tau_*)')


if __name__ == '__main__':
    test_fisher_bridge_contract()
    test_gumbel_bridge_contract()
    test_gumbel_temperature_schedule()
    test_build_bridge_rejects_dirichlet_and_sfm()
    test_build_bridge_drops_unused_kwargs()
    print('All bridge contract tests passed.')
