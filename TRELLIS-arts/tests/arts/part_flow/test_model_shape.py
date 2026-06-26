"""CPU shape-level sanity check for PartFlowPredictor (variable-K).

Covers:
- Single-sample forward at various K_b (=1, 2, 4)
- Batch forward with mixed K_b per sample (variable-K)
- Endpoint logits shape + masked to -inf on padding
- Part token pooling (bg excluded, occluded parts get slot fallback)
- Gradient flow through all sub-layers (incl. voxel-part scoring)
"""

from __future__ import annotations

import torch


def _setup_imports():
    """Trainer code now lives inside the trellis package — parent conftest
    injects TRELLIS-arts/ into sys.path. Force CPU-safe attention backend."""
    from trellis.models.part_flow import attention_utils as au
    au._get_attn_backend = lambda: 'sdpa_loop'
    from trellis.models.part_flow.part_flow_predictor import PartFlowPredictor
    return PartFlowPredictor


def _make_mask_labels(B: int, VT: int, num_parts_per_sample) -> torch.Tensor:
    """Per-sample mask_token_labels in {0..K_b}; at least one token per valid part."""
    labels = torch.zeros(B, VT, dtype=torch.long)
    for b, K_b in enumerate(num_parts_per_sample):
        # Phase 8: K_b includes empty slot 0; real mask labels are 1..K_b-1.
        active = torch.rand(VT) < 0.1
        ids = torch.randint(1, K_b, (VT,), dtype=torch.long)
        labels[b] = torch.where(active, ids, torch.zeros_like(ids))
        # Guarantee at least one token per real part
        for j in range(1, K_b):
            if (labels[b] == j).sum() == 0:
                labels[b, j - 1] = j
    return labels


def test_variable_K_per_sample_forward():
    """Batch with [K_b=2, K_b=4] — verify shape + valid mask."""
    PartFlowPredictor = _setup_imports()
    torch.manual_seed(0)

    k_max = 8
    B = 2
    num_parts = [2, 4]
    VT = 5480
    n_per = [50, 80]
    N_total = sum(n_per)

    model = PartFlowPredictor(
        k_max=k_max, hidden_dim=64, num_layers=2, num_heads=4, cond_dim=1024,
    )
    with torch.no_grad():
        model.voxel_score_proj.weight.add_(torch.randn_like(model.voxel_score_proj.weight) * 0.02)
    model.eval()

    # x_t padded to k_max, zero on padding dims; valid dims sum to ~1
    x_t = torch.zeros(N_total, k_max)
    offset = 0
    for b, (K_b, n_b) in enumerate(zip(num_parts, n_per)):
        v = torch.softmax(torch.randn(n_b, K_b), dim=-1)
        x_t[offset:offset + n_b, :K_b] = v
        offset += n_b

    batch_idx = torch.cat([torch.full((n_per[b],), b, dtype=torch.long) for b in range(B)])
    coords = torch.cat([batch_idx.unsqueeze(-1), torch.randint(0, 64, (N_total, 3))], dim=-1).int()
    t = torch.tensor([2.0, 5.0])
    cond = torch.randn(B, VT, 1024)
    mask_labels = _make_mask_labels(B, VT, num_parts)
    is_on_surface = torch.zeros(N_total, dtype=torch.long)

    with torch.no_grad():
        out = model(x_t, t, coords, cond, mask_labels, num_parts, is_on_surface)

    assert out['endpoint_logits'].shape == (N_total, k_max), \
        f'logits shape {out["endpoint_logits"].shape} != ({N_total}, {k_max})'

    # Padding dims should be -inf-like (< -1e3)
    pv = out['valid_per_voxel']
    logits = out['endpoint_logits']
    assert (logits[~pv] < -1e3).all(), 'Padding dims not properly masked to -inf'

    # Valid dims should have finite values
    assert torch.isfinite(logits[pv]).all(), 'Valid dims have non-finite logits'

    # part_valid_mask
    pvm = out['part_valid_mask']
    assert pvm.shape == (B, k_max)
    assert pvm[0, :2].all() and not pvm[0, 2:].any(), 'Sample 0 K_b=2 mask wrong'
    assert pvm[1, :4].all() and not pvm[1, 4:].any(), 'Sample 1 K_b=4 mask wrong'

    # Softmax on logits should sum=1 on valid dims only
    from torch.nn import functional as F
    probs = F.softmax(logits, dim=-1)
    for b, sl in enumerate([slice(0, 50), slice(50, 130)]):
        K_b = num_parts[b]
        # Invalid dims should contribute ~0
        inv_sum = probs[sl, K_b:].sum(dim=-1)
        assert inv_sum.max() < 1e-3, f'Padding softmax leak: max={inv_sum.max()}'
    print(f'[PASS] variable-K forward: K_b={num_parts}, padding masked to -inf')


def test_part_token_pooling_bg_excluded():
    """mask_token_labels bg (==0) must NOT contribute to part token pooling."""
    PartFlowPredictor = _setup_imports()
    torch.manual_seed(1)

    k_max = 5
    model = PartFlowPredictor(k_max=k_max, hidden_dim=32, num_layers=1, num_heads=2, cond_dim=16)
    model.eval()

    # Construct two batches where part tokens at ids 1-3 are at the same
    # positions (identical cond entries), but bg tokens differ wildly.
    # Part token pooling should be identical between the two.
    B = 1
    VT_a = 100
    VT_b = 500
    torch.manual_seed(100)
    cond_part = torch.randn(1, 3, 16)  # part 1, 2, 3 features

    # Build two batches with different VT but the same part positions
    def build(VT):
        torch.manual_seed(42 + VT)
        cond = torch.randn(B, VT, 16) * 5.0  # large magnitude bg
        labels = torch.zeros(B, VT, dtype=torch.long)
        # Put part 1 at idx 0, part 2 at idx 1, part 3 at idx 2
        cond[0, :3] = cond_part[0]
        labels[0, 0] = 1
        labels[0, 1] = 2
        labels[0, 2] = 3
        # All other tokens are bg with random large values — should be EXCLUDED
        return cond, labels

    cond_a, labels_a = build(VT_a)
    cond_b, labels_b = build(VT_b)

    # build_part_tokens uses rgb_proj first; reproduce that here
    with torch.no_grad():
        cp_a = model.rgb_proj(cond_a)
        cp_b = model.rgb_proj(cond_b)
        pt_a = model.build_part_tokens(cp_a, labels_a, [4])
        pt_b = model.build_part_tokens(cp_b, labels_b, [4])

    assert torch.allclose(pt_a, pt_b, atol=1e-5), \
        f'Part tokens differ between VT={VT_a} and VT={VT_b} — bg leaking into pooling'
    # Slot 4 is padding (invalid) — should be 0
    assert pt_a[0, 4].abs().max() < 1e-6
    print(f'[PASS] part-token pooling: bg excluded, VT={VT_a} vs {VT_b} identical')


def test_occluded_part_fallback():
    """If a part has zero 2D mask tokens (fully occluded), slot_emb is used as fallback."""
    PartFlowPredictor = _setup_imports()
    torch.manual_seed(2)

    k_max = 5
    model = PartFlowPredictor(
        k_max=k_max, hidden_dim=32, num_layers=1, num_heads=2, cond_dim=16,
        use_slot_embedding_fallback=True,
    )
    model.eval()

    VT = 50
    # Part 1 has tokens, parts 2, 3 are fully occluded (no mask coverage)
    cond = torch.randn(1, VT, 16)
    labels = torch.zeros(1, VT, dtype=torch.long)
    labels[0, :10] = 1  # only part 1 is visible

    with torch.no_grad():
        cp = model.rgb_proj(cond)
        # num_parts=4: empty + parts 1, 2, 3; parts 2, 3 are occluded.
        pt = model.build_part_tokens(cp, labels, [4])

    # Slot 0: dedicated empty token.
    assert torch.allclose(pt[0, 0], model.empty_token.detach(), atol=1e-6)

    # Part 1: mean of tokens 0-9 (via rgb_proj), stored in simplex slot 1.
    expected_pt_1 = cp[0, :10].mean(dim=0)
    assert torch.allclose(pt[0, 1], expected_pt_1, atol=1e-5)

    # Parts 2, 3: slot_emb fallback (non-zero, since use_slot_embedding_fallback=True)
    assert pt[0, 2].abs().max() > 1e-5, 'Part 2 (occluded) should have fallback slot_emb'
    assert pt[0, 3].abs().max() > 1e-5, 'Part 3 (occluded) should have fallback slot_emb'

    # Slot 4: padding — should be 0
    assert pt[0, 4].abs().max() < 1e-6
    print(f'[PASS] occluded-part fallback: visible part pooled, occluded uses slot_emb')


def test_gradient_flow_variable_K():
    """Mixed-K batch — all relevant params should receive gradient."""
    PartFlowPredictor = _setup_imports()
    torch.manual_seed(3)

    k_max = 6
    model = PartFlowPredictor(k_max=k_max, hidden_dim=32, num_layers=2, num_heads=4, cond_dim=64)
    with torch.no_grad():
        model.voxel_score_proj.weight.add_(torch.randn_like(model.voxel_score_proj.weight) * 0.02)
        model.part_score_proj.weight.add_(torch.randn_like(model.part_score_proj.weight) * 0.02)

    B = 2
    num_parts = [3, 5]
    VT = 200
    n_per = [20, 30]
    N_total = sum(n_per)

    x_t = torch.zeros(N_total, k_max)
    offset = 0
    for b, (K_b, n_b) in enumerate(zip(num_parts, n_per)):
        x_t[offset:offset + n_b, :K_b] = torch.softmax(torch.randn(n_b, K_b), dim=-1)
        offset += n_b

    batch_idx = torch.cat([torch.full((n_per[b],), b, dtype=torch.long) for b in range(B)])
    coords = torch.cat([batch_idx.unsqueeze(-1), torch.randint(0, 64, (N_total, 3))], dim=-1).int()
    t = torch.tensor([1.0, 5.0])
    cond = torch.randn(B, VT, 64)
    mask_labels = _make_mask_labels(B, VT, num_parts)
    is_on_surface = torch.zeros(N_total, dtype=torch.long)

    out = model(x_t, t, coords, cond, mask_labels, num_parts, is_on_surface)
    loss = out['endpoint_logits'][out['valid_per_voxel']].pow(2).sum()
    loss.backward()

    n_with_grad = sum(
        1 for p in model.parameters()
        if p.grad is not None and p.grad.abs().max() > 0
    )
    total = sum(1 for _ in model.parameters())
    assert n_with_grad > 0.5 * total, f'Only {n_with_grad}/{total} params got grad'

    # Check critical paths received gradient
    crit = {'voxel_score_proj', 'part_score_proj', 'cross_part_q', 'cross_part_k'}
    for name, p in model.named_parameters():
        if any(c in name for c in crit) and 'weight' in name:
            assert p.grad is not None and p.grad.abs().max() > 0, \
                f'{name} did not receive gradient'
    print(f'[PASS] gradient flow variable-K: {n_with_grad}/{total} params, critical paths wired')


if __name__ == '__main__':
    test_variable_K_per_sample_forward()
    test_part_token_pooling_bg_excluded()
    test_occluded_part_fallback()
    test_gradient_flow_variable_K()
    print('All model shape tests passed.')
