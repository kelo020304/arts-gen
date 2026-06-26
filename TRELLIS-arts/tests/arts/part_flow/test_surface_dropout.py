"""D-21: surface dropout rate Uniform(0.05, 0.20), only flips 1 -> 0.

These tests exercise the per-iter dropout logic that PartFlowDataset.__getitem__
applies to `is_on_surface`. The implementation is duplicated here (small
`_apply_surface_dropout`) so we can sweep seeds without instantiating the full
dataset (which requires DINOv2 + render data).

Contract verified:
  - rate samples fall inside [lo, hi] (with margin for float endpoints)
  - dropout flips only 1 -> 0, never 0 -> 1
  - mean fraction dropped tracks mean rate
  - deterministic given seed
"""

import numpy as np
import torch


def _apply_surface_dropout(
    is_on_surface: torch.Tensor,
    lo: float,
    hi: float,
    seed: int,
) -> tuple[torch.Tensor, float]:
    """Mirror of the logic inside PartFlowDataset.__getitem__."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    rate = float(np.random.uniform(lo, hi))
    is_on_surface = is_on_surface.clone()
    surf_idx = torch.nonzero(is_on_surface, as_tuple=False).squeeze(-1)
    n_drop = int(len(surf_idx) * rate)
    if n_drop > 0:
        perm = torch.randperm(len(surf_idx))[:n_drop]
        is_on_surface[surf_idx[perm]] = 0
    return is_on_surface, rate


def test_surface_dropout_only_flips_1_to_0():
    torch.manual_seed(0)
    before = (torch.rand(262144) < 0.05).long()  # ~5% surface
    after, _ = _apply_surface_dropout(before, 0.05, 0.20, seed=0)
    # after <= before elementwise (no 0 -> 1 flip)
    assert (after <= before).all()
    # Must actually drop SOME (rate * n_surf >= 1 for this seed/rate)
    assert int(after.sum()) < int(before.sum())


def test_surface_dropout_rate_in_range_over_many_iters():
    """Rate samples must fall in [0.05, 0.20]."""
    before = (torch.rand(262144) < 0.05).long()
    rates = []
    for seed in range(50):
        _, rate = _apply_surface_dropout(before, 0.05, 0.20, seed=seed)
        rates.append(rate)
    rates_arr = np.array(rates)
    # strict bounds (Uniform(lo, hi) is closed-open but we allow tiny float slop)
    assert rates_arr.min() >= 0.05 - 1e-9
    assert rates_arr.max() <= 0.20 + 1e-9
    # mean of U(0.05, 0.20) = 0.125
    assert 0.09 <= rates_arr.mean() <= 0.16, f'mean rate {rates_arr.mean()}'


def test_surface_dropout_fraction_reasonable():
    """Over 50 iters on a fixed surface, fraction actually dropped ~ rate."""
    torch.manual_seed(0)
    before = (torch.rand(262144) < 0.05).long()
    n_before = int(before.sum().item())
    frac_dropped = []
    for seed in range(50):
        after, _ = _apply_surface_dropout(before, 0.05, 0.20, seed=seed)
        n_after = int(after.sum().item())
        frac_dropped.append((n_before - n_after) / max(1, n_before))
    mean_frac = float(np.mean(frac_dropped))
    # mean expected frac ~= mean rate ~= 0.125; allow ±0.04 band to absorb
    # int-truncation bias and Monte-Carlo spread across 50 samples
    assert 0.08 < mean_frac < 0.17, f'mean dropped frac {mean_frac}'


def test_surface_dropout_deterministic_given_seed():
    before = (torch.rand(262144) < 0.05).long()
    a1, r1 = _apply_surface_dropout(before, 0.05, 0.20, seed=1234)
    a2, r2 = _apply_surface_dropout(before, 0.05, 0.20, seed=1234)
    assert r1 == r2
    assert torch.equal(a1, a2)


def test_surface_dropout_does_not_touch_non_surface_voxels():
    """Only positions where before == 1 can change to 0; positions where
    before == 0 must remain 0 in the output."""
    before = (torch.rand(262144) < 0.05).long()
    off_mask = before == 0
    for seed in range(10):
        after, _ = _apply_surface_dropout(before, 0.05, 0.20, seed=seed)
        assert int(after[off_mask].sum()) == 0
